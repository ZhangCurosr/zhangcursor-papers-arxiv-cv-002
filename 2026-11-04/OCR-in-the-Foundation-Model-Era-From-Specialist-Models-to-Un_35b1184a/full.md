# OCR in the Foundation Model Era: From Specialist Models to Unified Models

Zhiheng Li , Zongyang Ma , Jiaxian Chen , Cheng Cui , Yi Liu , Ziqi Zhang , Chunfeng Yuan , Bing Li , Xiang Bai , and Weiming Hu

Abstract—Optical Character Recognition (OCR) is among the oldest and most widely deployed applications of artificial intelligence. This survey reviews its development, focusing on the Foundation Model Era. We define OCR through faithful transcription: within a specified visual scope, OCR should completely reconstruct the text, notation, and document structure observable in the source image, and every textual, notational, or structural element in the output must be verifiable from visual evidence in that image. We trace the development from specialist OCR models toward unified models, examining joint text detection and recognition, structured sequence generation, and the use of pretrained vision–language models (VLMs). We organize contemporary OCR systems into five coexisting families: modular toolkits and task-specific models, specialized document VLMs, hybrid workflows, general-purpose VLMs, and commercial products and managed services. We compare their architectures, data, and training, examining hallucination and omission, insufficient spatial grounding, and inference cost in generative OCR. We review evaluation from character and scene text recognition to parsing individual structural elements and full-page transcription, highlighting gaps in consistent output definitions, attribution of performance to individual components, and coverage of real capture conditions and visible content and structure. Finally, we summarize the challenges facing OCR and discuss directions for future research. We continuously track related work at https://github.com/zhihengli-casia/Awesome-OCR-in-the-Foundation-Model-Era.

Index Terms—Optical Character Recognition, Generative Document Parsing, Vision–Language Models.

## 1 INTRODUCTION

O <sup>PTICAL</sup> <sup>Character</sup> <sup>Recognition</sup> <sup>(OCR)</sup> <sup>is</sup> <sup>among</sup> <sup>the</sup> <sup>old-</sup> est and most widely deployed applications of artificial intelligence. Efforts to read printed text by machine date back to the 1910s [1], and by the 1950s dedicated OCR devices matched optically sensed character patterns against stored templates to produce machine-readable codes [2].

Over the following decades, OCR moved well beyond its original reliance on dedicated hardware. The targets of recognition expanded along several axes: from printed to handwritten text [3], from scanned pages to text captured in natural scenes as cameras became ubiquitous [4], and from plain passages to documents with rich structures such as tables, formulas, and layouts with multiple columns [5], [6]. Generative models, increasingly built on pretrained vision–language models (VLMs), produce JSON, Markdown, HTML, or LAT X directly from document images [7], [8].

What counts as OCR? The literature uses OCR with different boundaries: technology for scanned documents [4], camera-based OCR [9], artificial optical signals [8], text and visual structure recovery in holistic OCR [10], and multimodal evaluations that include information extraction and document question answering [11]. In this survey, we define OCR through faithful transcription. Within a specified visual scope, OCR aims to completely reconstruct the text, notation, and document structure observable in the source image. Every textual, notational, or structural element in the output must be verifiable from visual evidence in that image (Fig. 1). Normalization must follow a declared output convention. Our main coverage includes printed, handwritten, and scene text, mathematical formulas, tables, layout, and reading order. Chart data, flowchart connections, musical notation, and chemical structures can also be recovered from visual evidence; we discuss these as related extensions with selective coverage.

![](images/a91680ce3db5b799ce7d79ce61b314fe12ab40bcb08df29a4ced9ae48ed61d16.jpg)  
Fig. 1. Scope of this survey. Within a specified visual scope, OCR aims to completely recover visible text, notation, and document structure, with every output element verifiable against the source image. Main coverage spans printed, handwritten, and scene text; formulas as notation; and tables, layout, and reading order as structure. Key information extraction, question answering, translation, and retrieval are downstream tasks.

Recovering a table’s cells, rows, and merged cells, or the values a chart encodes through marks and axes, reconstructs the source document; computing a statistic from those values or explaining a trend goes beyond it. Document understanding assigns semantic fields [12], answers questions [13], or draws conclusions about a document [14], [15]. These downstream tasks have influenced OCR architectures, training, and evaluation.

![](images/0ac5c9fdc47b957fa1f745e19411ec6cbca1152d0ee69c705c0694c67bc44d03.jpg)  
Fig. 2. OCR and related document modeling across five historical tracks and three eras. Columns show representative pipelines; braces mark three eras: I. Classical OCR Era, II. Deep Learning Era, and III. Foundation Model Era. The dashed Document Pretraining box provides related background: these models combine upstream OCR text, layout, and visual features for document understanding. Generative parsing reconstructs text, formulas, and tables within a unified structured document, using formats such as Markdown, HTML, L<sup>A</sup>T X, or JSON. Dates indicate approximate emergence or historical prominence; the tracks overlap in time and several remain active in contemporary systems.

From specialist models to unified models: Fig. 2 traces three overlapping eras: the Classical OCR, Deep Learning, and Foundation Model eras. Classical methods relied on templates, handcrafted features, and statistical models. Deep learning brought learned features and sequence prediction, while pipelines continued to assemble outputs from specialist components [16]. Joint modeling also developed: Mask TextSpotter [17] integrated detection and recognition in one network, SPTS [18] represented locations and text in one output sequence, and OmniParser [19] supported text spotting, table recognition, and key information extraction within a common framework.

The Foundation Model Era expanded task sharing through large-scale pretraining and structured generation. Donut [7] generated structured results for document tasks directly from images, and Nougat [20] adapted this approach to full-page scientific transcription. GOT-OCR [8] and HunyuanOCR [21] used pretrained large language models (LLMs) to support multiple tasks and content types in one model. The development from specialist to unified models thus spans these eras, with increasing sharing of networks, representations, and decoders.

In practice, unified models can generate complete pages or work with layout analysis to generate regions before document assembly. Specialist models, whole-page generation, and regional generation continue to coexist; Section 3 compares their architectures.

Why does OCR need a new survey? Mori et al. [2] reviewed the historical development of OCR. Ye and Doermann [4], Zhu et al. [25], Long et al. [26], and Chen et al. [9] surveyed scene text detection and recognition across different stages of its development. Liu et al. [27] provided a broader review of intelligent document analysis and recognition. Recent surveys examine OCR with large models [28], document intelligence and parsing [106], [107], [108], text-centric image understanding [109], visual text processing and manipulation [29], [110], or general vision–language models that include OCR among many capabilities [30]. They adopt different boundaries for OCR.

In the Foundation Model Era, a single system can perform transcription, structural recovery, and downstream understanding, while specialist models, unified models, and hybrid workflows continue to coexist. This survey connects research on specialist OCR and unified models to explain how OCR has changed in the Foundation Model Era and which problems warrant further research. Fig. 3 outlines the survey’s structure.

Contributions: (i) We systematically review the major stages and technical transitions of OCR, tracing its development from specialist models toward unified models with a focus on the Foundation Model Era. (ii) We define OCR through faithful transcription and organize contemporary systems into five families, comparing their architectures, data, and training. (iii) We trace OCR evaluation from text recognition to full-page parsing, analyze recognition and structural recovery using published results, and identify gaps in output definitions, score attribution, and coverage of multilingual content, long documents, and real capture conditions. (iv) Focusing on faithful transcription, we discuss output reliability, the coexistence of specialist, unified, and hybrid approaches, and the expanding scope of content recovery.

![](images/f5f5863f1a2ab7b29bca459e6d83bca1819b9d23e3fefe62881713c0697601b9.jpg)  
Fig. 3. Review framework and evidence map. The figure connects OCR’s historical evolution and five contemporary system families with benchmark evidence, current limitations, and the three expectations of the outlook. Citations identify representative references.

This survey aims to help newcomers understand OCR and provide experienced researchers with a reference for recent advances and open problems.

## 2 HISTORICAL EVOLUTION OF OCR

This section traces how OCR recovered visible text and its location, and how these predictions became increasingly integrated for document reconstruction. Detailed surveys cover these eras [2], [4], [9], [25], [26], [27]; Figs. 2 and 4 summarize their evolution.

## 2.1 Classical OCR Era

Early optical reading devices included the optophone (1914), which encoded printed patterns as tones for a listener to interpret, and Tauschek’s template-matching machine (1929). Commercial OCR later used standardized fonts and controlled imaging [1], [2]. HMMs [31] modeled character sequences, SVMs [32] classified characters, and LeNet [33] learned recognition features from pixels. FineReader [111] and Tesseract [36] supported document

![](images/9e11085febf8a68247cd62c551b96487e76ce45f4d880ab17c961c4cd4d9fa0f.jpg)  
Fig. 4. Representative milestones in OCR and related document modeling, following the five tracks in Fig. 2. Years mark publications, public announcements, or system releases; horizontal spacing is schematic. Asterisks identify precursors: the optophone as a reading aid, and Donut and Pix2Struct in document understanding and image-to-sequence pretraining. Section 2 covers classical and deep learning methods; Section 3 examines unified models and generative parsing.

OCR, while recognition expanded to handwriting and non-Latin scripts [3], [34], [35].

Scene text introduced uncontrolled typography, backgrounds, and lighting. MSER [37] located stable regions, SWT [38] exploited consistent stroke width, and Neumann and Matas [39] connected localization with recognition. PhotoOCR [112] combined handcrafted features with a deep character classifier, marking a transition toward learned representations.

## 2.2 Deep Learning Era

Deep learning replaced handcrafted features with learned representations [113], [114]. In detection, CTPN [115] linked local text proposals, EAST [40] regressed oriented regions, PSENet [116] and CRAFT [117] supported irregular text through segmentation, and DBNet [41] made binarization differentiable.

For cropped text recognition, CRNN [23] combined CNN, BiLSTM, and CTC. RARE [118] and ASTER [42] jointly trained a thin-plate-spline rectifier and an attentionbased recognizer using transcription supervision. ABI-Net [119] added iterative language modeling; TrOCR [43] used pretrained vision and language Transformers; and SVTR [120] replaced the CNN–RNN backbone with a visual Transformer. Training data scale and curation also substantially affect recognition [72], [73].

Separately trained detectors and recognizers can duplicate feature extraction, while localization errors affect the image regions available for transcription. End-to-end spotters couple the two tasks through shared features and joint supervision. FOTS [44] and Mask TextSpotter [17] combined both tasks in one network; Mask TextSpotter used character segmentation to recognize irregular text. ABCNet [121] represented curved text with Bezier curves, while DeepSolo [45] jointly predicted geometry and characters through point queries.

Sequence generation further connected localization and recognition through a common output representation. SPTS [18] generated a point and its transcription for each instance. To reduce the cost of long autoregressive sequences, SPTS v2 [46] generated instance points first, then recognized instances in parallel with shared decoder parameters. UNITS [122] used prompts to select points, boxes, or polygons as geometric outputs alongside transcriptions. Omni-Parser [19] extended sequence modeling to table recognition and downstream key information extraction (§3.1).

Task-specific recognition continues to develop: CPPD and IPAD study efficient decoding [123], [124]; SMTR addresses text longer than training instances [125]; LMIM uses linguistically informed self-supervision [126]; UnionST improves text synthesis [127]; and WATERec targets artistic typography [128]. SCUT-EPT highlights long text, editing marks, and noisy backgrounds [129]. EasyOCR [130], docTR [131], MMOCR [132], and PaddleOCR [16] package components into multilingual pipelines, with PaddleOCR also covering layout, tables, and formulas.

Document pretraining addressed relationships beyond isolated text recognition. LayoutLM and its successors [24], [133], [134], DocFormer [47], and UDOP [48] combine upstream OCR text, layout coordinates, and visual features into shared representations for document understanding. OCR supplies the transcriptions, while these models learn how text, position, and appearance interact in downstream tasks. Document reconstruction also requires relating recognized content to its layout. The cost and error propagation of the external OCR stage also motivated direct image-tosequence approaches such as Donut [7] (§3.1).

## 2.3 Lessons and Transition to Generative Parsing

Faithful transcription of a complete page requires accurate local recognition and recovery of the visible structural relations among regions. Errors can therefore arise within individual components and when their outputs are assembled. Modular pipelines offer independently testable and upgradable components, supporting reliability and cost control. Their coordination challenges depend on representation sharing, joint training, and feedback [16], [26]. (1) Error propagation: missed detections or incorrect layout labels can affect later recognition and assembly, while global correction across components is difficult [7], [8]. (2) Training fragmentation: different annotations and objectives complicate pagelevel optimization; shared encoders and multitask training help reuse data and representations [48]. (3) Output integration: coherent documents require aligning regions, text, reading order, hierarchy, and specialized markup through interfaces and postprocessing [8], [20].

The Deep Learning Era brought both learned representations and closer integration of OCR tasks. Joint text spotters reduce detection–recognition handoffs [44], [45], while sequence prediction connects locations and transcriptions in a common output. The Foundation Model Era builds on these developments through large-scale pretraining and shared generation across more content types and document structures. Whole-page generation and layout-guided regional generation organize this integration differently (Section 3).

## 3 OCR IN THE FOUNDATION MODEL ERA

In the Foundation Model Era, large-scale pretraining supplies visual and language representations that can be reused across OCR tasks. Building on earlier joint prediction and sequence modeling, generative document models recover text, notation, and structure through shared decoders. We examine three aspects of this unification: which tasks share a model, how their outputs represent content and structure, and how inference coordinates the page and its regions. These models can generate whole pages or operate in stages, as in layout prediction followed by regional recognition. We assess these choices through transcription fidelity, spatial and structural recovery, and inference cost, accounting for differences in architecture, training data, and decoding strategy.

This section traces how generative document parsing emerged (§3.1), organizes the current ecosystem into five coexisting system families and analyzes their architectural differences (§3.2), examines the data and training pipelines that further differentiate them (§3.3), and characterizes the limitations that define the current frontier (§3.4). The Generative Parsing track of Fig. 4 charts representative systems from Donut to recent models including OCRVerse [10], Unlimited-OCR [135], and TeleOCR [61].

## 3.1 Emergence of Generative Document Parsing

Generative document parsing drew on three advances outside OCR. Large-scale pretraining produced transferable representations from unlabeled text [136] and naturallanguage supervision of images [74]; Vision Transformers [137] extended Transformer-based pretraining to image patches; and language-model pretraining at scale supplied general-purpose text decoders [138], while instruction tuning with human feedback improved their ability to follow user instructions [139]. These advances expanded the capacity and task coverage of generative OCR.

Three developments adapted general-purpose pretrained models to the demands of document OCR. First, vision encoders reached the resolution that dense document text demands: Swin Transformer [49] keeps attention cost tractable at page scale through hierarchical windowing, NaViT [50] processes images at native resolution and aspect ratio, and tiling schemes split a page into subimages so that encoders with a fixed input resolution, such as SigLIP [140], can be reused. Earlier encoders designed for scene understanding operated at 224–384 pixels [74], [137] and could miss small characters or fine table lines. Monkey [141] reused a pretrained encoder on local crops with separate low-rank adapters and a shared resampler, retaining a global image view. TextMonkey [142] introduced shifted-window attention for communication across windows and resampling guided by similarity to reduce redundant visual tokens. Second, the vision–language model (VLM) architecture emerged as a general framework for multimodal tasks [143], [144]: a vision encoder converts the image into visual tokens, a learned projector aligns them with the embedding space of a pretrained large language model, and the LLM generates text conditioned on them. We use VLM broadly for models that connect visual representations with language generation. The models discussed here include compact encoder–decoder precursors, such as Donut and Nougat, and recent systems built on pretrained LLM decoders. This architecture gave OCR a visual frontend with high capacity and a unified interface for structured generation. Third, training data for whole pages with structural annotations became available at scale. Rendering markup into images and automatically annotating real documents were established techniques [5], [6], [92], but they had supplied only markup for cropped regions or layout boxes for individual modules. Whole pages can be paired with transcription or markup sequences through synthetic document generation and alignment of existing PDFs with structured sources [20], [51], or through existing OCR pipelines with quality filtering [8], [76], supporting training on whole pages at scale.

Early models that map images to sequences without a separate OCR engine, such as Donut [7] and Pix2Struct [52], provided important technical foundations. Donut used an encoder–decoder architecture with a decoder initialized from multilingual BART for task-specific document understanding. Pix2Struct used HTML generation from screenshots for pretraining across visual and linguistic tasks. Nougat [20] built on Donut’s architecture to transcribe complete pages of scientific documents. Its target sequence represents both recognized content and document structure.

OmniParser [19], introduced in §2.2, unified prediction across document tasks through explicit spatial representations without a pretrained LLM. It used structure points to organize text spotting, key information extraction, and table recognition, followed by separate contour and text decoders. OmniParser V2 [53] added layout analysis and introduced a shared decoder with separate feedforward branches for structure, localization, and recognition. In its architectural comparison with the same Swin-B backbone, V2 uses 110M parameters versus 144M for OmniParser, with end-to-end text spotting scores of 84.3 versus 84.0 on Total-Text without a lexicon [53]. This illustrates how partial decoder sharing can reduce model size while preserving text spotting performance. It also used its structured-pointsof-thought representation to fine-tune multimodal language models.

![](images/ab71a7e5cfd0f6a43a768a0055f8057053aa3c006f37ae607ca04c1c79fd7bfd.jpg)  
Fig. 5. Five system families in the OCR ecosystem of the Foundation Model Era, showing how each converts visual input into structured output. Each flow represents a family: modular toolkits and task-specific models combine task-specific prediction with modular processing, specialized document VLMs perform full-page generation with a model trained for documents, hybrid workflows scaffold generation with explicit layout analysis and VLM recognition of individual regions, general-purpose VLMs expose transcription through open prompting, and commercial products and managed services expose closed stacks through product interfaces or managed endpoints. Table 1 lists representative systems.

In recent OCR VLMs, task unification also draws on vision–language pretraining and multitask training. HunyuanOCR-1.5, for example, combines transcription and document parsing with downstream question answering and translation in one model [57].

For complete document reconstruction, whole-page generation places content and structure in a common decoding context. Dense pages, however, may require high input resolution and long output sequences. Layout-guided regional generation [80], [81], [82] addresses these costs by identifying regions and decoding their contents separately, retaining an explicit document assembly stage. These two routes draw on shared generative models and training methods and continue to develop in parallel.

## 3.2 System Architectures and Ecosystem

Recent benchmarks and technical reports have begun categorizing today’s coexisting OCR systems into coarse groups such as pipelines, end-to-end systems, and VLMs [59], [96], [98]. A classical detection–recognition pipeline and a hybrid workflow that pairs a layout detector with generative recognizers are both pipelines, yet differ fundamentally in how they produce output; a specialized document VLM and a general-purpose VLM are both “end-to-end,” yet differ in training objective, output consistency, and deployment constraints. We therefore refine the existing groupings into five families by asking two questions of each system. The first is how recognition and structure are integrated: through task-specific predictions produced within one model or combined by a toolkit or application (Family 1, including joint text spotters and specialized table or formula recognizers); at the region level, where an explicit layout stage selects regions for recognition and subsequent assembly (Family 3); or through structured generation over the whole page (Families 2 and 4). The second question separates the two families that generate whole pages by training objective: specialized document VLMs trained specifically for document OCR form Family 2, while general-purpose VLMs with OCR as one capability among many form Family 4. Commercial products and managed services often disclose too little architectural or training detail for reliable placement on either dimension. We therefore treat their delivery model as

## TABLE 1

Representative systems in the OCR ecosystem across the five families (F1–F5). Date gives the first documented public appearance of each work version, or product family (YYYY.MM; YYYY.– indicates an unavailable month). Outputs list representative content types or formats: Text includes answers and captions for downstream tasks, and Geometry covers boxes, polygons, and points. Params gives official nominal counts for the named model or family (the core VLM in F3); “–” marks undisclosed or non-unique values. † marks compact encoder–decoder precursors: Donut for document understanding and Nougat for transcription. Access icons link to official code, model hubs, papers, or services. Full inventory at https://github.com/zhihengli-casia/Awesome-OCR-in-the-Foundation-Model-Era.
<table><tr><td>Model / System</td><td>Date</td><td>Typical output</td><td>Params</td><td>Access</td></tr><tr><td colspan="5">F1: Modular toolkits and task-specific models</td></tr><tr><td></td><td>Modular toolkits</td><td></td><td></td><td></td></tr><tr><td>PP-OCRv6 [54]</td><td>2026.06</td><td>Text, Geometry</td><td>1.5–34.5M</td><td>CCCCC</td></tr><tr><td>PP-StructureV3 [145]</td><td>2025.05</td><td>Markdown, JSÓN</td><td></td><td></td></tr><tr><td>Docling [55]</td><td>2024.07</td><td>Markdown, JSON</td><td></td><td></td></tr><tr><td>docTR[131]</td><td>2021.03</td><td>Text, Geometry</td><td></td><td></td></tr><tr><td>EasyOCR [130]</td><td>2020.04</td><td>Text, Geometry</td><td></td><td></td></tr><tr><td>Tesseract [36]]</td><td>2005.-</td><td>Text, Geometry</td><td></td><td></td></tr><tr><td colspan="5">Task-specific models</td></tr><tr><td>SVTRv2 [56]</td><td>2024.11</td><td>Text</td><td>5.1/11.3/19.8M</td><td></td></tr><tr><td>DocLayout-YOLO [146]</td><td>2024.10</td><td>Geometry</td><td>19.5M</td><td></td></tr><tr><td>UniMERNet [93]</td><td>2024.04</td><td>LATEX</td><td>107/202/325M</td><td></td></tr><tr><td>OmniParser [19]</td><td>2024.03</td><td>Text, Geometry, Markup</td><td></td><td></td></tr><tr><td>DeepSolo [45]</td><td>2022.11</td><td>Text, Geometry</td><td></td><td></td></tr><tr><td>DBNet++ [147]</td><td>2022.02</td><td>Geometry</td><td></td><td></td></tr><tr><td>TrOCR [43]</td><td>2021.09</td><td>Text</td><td>62/334/558M</td><td></td></tr><tr><td>ASTER [42]</td><td>2018.06</td><td>Text</td><td></td><td></td></tr><tr><td>CRNN [23]</td><td>2015.07</td><td>Text</td><td>8.3M</td><td>0</td></tr><tr><td colspan="3"></td><td></td><td></td></tr><tr><td>HunyuanOCR-1.5 [57]</td><td>2026.07</td><td>F2: Specialized document VLMs Text, Markdown, Geometry</td><td>1B</td><td></td></tr><tr><td>ÓvisOCR2 [58]</td><td>2026.07</td><td>Markdown</td><td>0.8B</td><td></td></tr><tr><td>Unlimited-OCR [135]</td><td>2026.06</td><td>Text, Geometry</td><td>3B (0.5B act.)</td><td></td></tr><tr><td>Qianfan-OCR [59]</td><td>2026.03</td><td>Markdown, HTML, JSON</td><td>4B</td><td></td></tr><tr><td>dots.mocr [148]</td><td>2026.03</td><td>Markdown, JSON, SVG</td><td>3B</td><td></td></tr><tr><td>FireRed-OCR [149]</td><td>2026.02</td><td>Markdown</td><td>2B</td><td></td></tr><tr><td>DeepSeek-OCR-2 [60]</td><td>2026.01</td><td>Markdown</td><td>3B (0.5B act.)</td><td></td></tr><tr><td>olmOCR 2 [76]</td><td>2025.10</td><td>Text, Markdown</td><td>7B</td><td></td></tr><tr><td>Granite-Docling [150]</td><td>2025.09</td><td>DocTags</td><td>258M</td><td></td></tr><tr><td>GOT-OCR 2.0 [8]</td><td>2024.09</td><td>Text, Markup</td><td>580M</td><td></td></tr><tr><td>Nougat† [20]</td><td>2023.08</td><td>Markdown</td><td>250/350M</td><td></td></tr><tr><td>Donut† [7]</td><td>2021.11</td><td>Text, JSON</td><td>202M</td><td>0</td></tr><tr><td colspan="3">F3: Hybrid workflows</td><td></td><td></td></tr><tr><td>TeleOCR [61]</td><td>2026.08</td><td>Markdown, JSON</td><td>1.2B</td><td></td></tr><tr><td>MonkeyOCRv2-Parsing [75]</td><td>2026.07</td><td>Markdown, JSON</td><td>0.6/0.7B</td><td></td></tr><tr><td>PaddleOCR-VL-1.6 [62]</td><td>2026.05</td><td>Markdown, JSON</td><td>0.9B</td><td></td></tr><tr><td>MinerU 2.5-Pro [63]</td><td>2026.04</td><td>Markdown, JSON</td><td>1.2B</td><td></td></tr><tr><td>GLM-OCR [64]</td><td>2026.02</td><td>Markdown, JSON</td><td>0.9B</td><td></td></tr><tr><td>Youtu-Parsing [83]</td><td>2026.01</td><td>Text, IATEX, HTML, Geometry</td><td>2.5B</td><td></td></tr><tr><td>Dolphin-v2 [82]</td><td>2025.12</td><td>Markdown, JSON</td><td>3B</td><td>CCCCOCC</td></tr><tr><td>MonkeyOCR [81]</td><td>2025.06</td><td>Markdown, JSON</td><td>3B</td><td>0</td></tr><tr><td colspan="3">F4: General-purpose VLMs</td><td></td><td></td></tr><tr><td>GPT-6 [68]</td><td>2026.10</td><td>Text, Markup</td><td></td><td></td></tr><tr><td>Qwen3.8 [66]</td><td>2026.08</td><td>Text, Markup</td><td>27B</td><td></td></tr><tr><td>Gemini 3.1 Pro [65] InternVL3.5 [67]</td><td>2026.02 2025.08</td><td>Text, Markup</td><td></td><td>00ΦC</td></tr><tr><td></td><td></td><td>Text, Markup</td><td>1.1-241B</td><td></td></tr><tr><td colspan="6">F5: Commercial products and managed services</td></tr><tr><td>Mistral OCR 4 [69]</td><td>2026.06</td><td>Markdown, JSON</td><td></td><td></td></tr><tr><td>Azure AI Doc Intelligence [71]</td><td>2019.05</td><td>JSON</td><td></td><td>一00我书书</td></tr><tr><td>Google Document AI [70]</td><td>2019.04</td><td>JSON</td><td></td><td></td></tr><tr><td>Amazon Textract [151]</td><td>2018.11</td><td>JSON</td><td></td><td></td></tr><tr><td>Mathpix [152]</td><td>2016.03</td><td>Markdown</td><td></td><td></td></tr><tr><td>ABBYY FineReader [111]</td><td>1993.07</td><td>Text, PDF</td><td></td><td></td></tr></table>

a separate category (Family 5). The families describe OCR workflows; task sharing can occur within each family. Fig. 5 and Table 1 summarize the five families. The table selects historical landmarks, representative current architectures, and widely deployed platforms, and compares their typical outputs, parameter scale, and availability.

Family 1: Modular toolkits and task-specific models. This family includes modular toolkits (Tesseract [36], PaddleOCR [16], [145], EasyOCR [130], docTR [131]) that chain detection, recognition, and postprocessing, as well as taskspecific models (SVTR [120], UniMERNet [93], DocLayout-YOLO [146]) and joint text spotters. Such models solve defined recognition or localization tasks and can also serve as building blocks in Family 3. Their task-specific interfaces commonly support constrained or deterministic decoding: detection–recognition pipelines return text with spatial coordinates, table and formula modules emit structured markup, and layout models produce bounding boxes with category labels. These systems retain task-specific supervision and output formats and are widely used for stable workloads with clear requirements, where they offer predictable outputs and high throughput. Dedicated recognition remains active alongside OCR with foundation models. On WordArt-Bench, WATERec [128] substantially outperforms the VLM baselines evaluated under the same protocol, showing the continuing value of data and architectures tailored to the domain for challenging text distributions.

Family 2: Specialized document VLMs. A specialized document VLM, designed or fine-tuned specifically for document OCR, maps page images to structured output (Markdown, HTML, LAT X, or JSON), directly targeting the requirement for faithful parsing (Table 1). These systems commonly combine training focused on documents, visual encoding designed for OCR, and a predefined output schema. Structural relations such as heading hierarchy, table boundaries, and reading order are handled within the generative process, with designs ranging from implicit decoding [20] through chain-of-thought layout [59] to grounded multitask parsing [21]. GOT-OCR shares a language decoder across text and notation formats [8]; HunyuanOCR-1.5 uses task prompts to generate transcriptions, text coordinates, or document markup [57]. Unification therefore concerns both shared model parameters and support for different output requirements.

Family 3: Hybrid workflows. These systems bridge Families 1 and 2: an explicit layout stage, run by a dedicated detector (often a Family 1 component) or by the parsing model itself, segments the page into typed regions, and each region is decoded by a VLM recognizer before a postprocessor assembles the results into a structured document [63], [81], [82]. Working on bounded regions can shorten generation sequences, localize some errors, and make computational cost easier to budget. Reliability depends on layout recall and crop quality: missed or inaccurate regions may produce omissions that subsequent assembly cannot recover.

Dolphin (2025) generated the page’s element sequence with a compact encoder–decoder and then parsed the elements in parallel with element-specific prompts [153]. MonkeyOCR (June 2025) built a hybrid workflow on a pretrained LLM decoder: a dedicated detector predicts the block structure, a 3B VLM recognizes each block, and a relation stage orders the blocks [81]. dots.ocr (July 2025) took the whole-page route, predicting layout and content with one VLM in a single pass [154]. Subsequent releases explored smaller recognizers and different degrees of sharing across stages: MinerU2.5 (September 2025) used a 1.2B VLM for layout at reduced resolution and for recognition from crops at native resolution [80]; PaddleOCR-VL-1.5 (January 2026) kept the two-stage form with a 0.9B VLM [77]; and Dolphin-v2 fine-tuned one VLM for both layout analysis and content parsing, with regional recognition for digital documents and whole-page parsing for photographed ones [82]. Releases in 2026 kept this structure and moved effort to training and decoding: MinerU2.5-Pro added fine-tuning on difficult samples and PaddleOCR-VL-1.6 progressive post-training [62], [63]; MonkeyOCRv2 replaced the reused vision encoder with one pretrained from scratch on document images [75] (§3.3);

GLM-OCR and Youtu-Parsing cut sequential decoding steps with multi-token and parallel decoding [64], [83] (§3.4); and TeleOCR extended the workflow to camera-captured documents [61]. Recognizers in this family now range from 0.6B to 3B parameters, with most 2026 releases below 1.5B.

Family 4: General-purpose VLMs. General-purpose VLMs (GPT-6 [68], Gemini 3.1 Pro [65], Qwen3.8 [66], InternVL3.5 [67]) are trained for broad understanding of visual and linguistic content, with OCR as one capability among many. These models support transcription through prompts, alongside applications such as question answering about images and semantic field extraction. The share of documents in their training mixtures is generally undisclosed. On OmniDocBench v1.6 full, the general-purpose VLMs on the official leaderboard span 83.76 to 93.70 Overall, from InternVL3.5-241B to Ovis2.6-30B-A3B, and the strongest of them falls within the 93.26–96.91 range of the specialized and hybrid systems in Table 6 [155].

Family 5: Commercial products and managed services. Commercial products and managed services (ABBYY, Mathpix, TextIn, Mistral OCR 4 [69], Google Document AI [70], Azure [71]) typically emphasize operational reliability, format control, integration support, and enterprise deployment features. Where documentation describes the processing, it maps onto the preceding families: Document AI and Azure AI Document Intelligence expose separate reading, layout, and form or table processors in the manner of Family 1 toolkits, while Mistral OCR 4 serves one document model that returns Markdown and JSON.

## 3.3 Data and Training Pipelines

The capabilities of each system family depend on both its architecture and the data and training signals from which those capabilities are learned. Fig. 6 compares these two dimensions across the four trainable families. Local transcription labels supervise recognition, while page reconstruction also requires reading order, hierarchy, and relations between elements.

Data sources: Data quantity and quality are established performance variables in scene text recognition [72], [73], and the four trainable families draw on progressively broader data (Fig. 6, left). Family 1 trains each module on task-specific data: scene text images, word and line crops, pages with layout annotations, table crops, and formula crops. Public corpora range from roughly $1 0 ^ { 5 }$ annotated pages or formulas to nearly $1 0 ^ { 7 }$ synthetic word images per task [5], [6], [92], [156]. Family 2 assembles data centered on documents, with reported training mixtures spanning several million to tens of millions of page and region image–text pairs [7], [8], [20], [157]. These data combine full pages with region crops through synthetic rendering [51], engine bootstrapping, deduplication using CLIP [74], quality filtering, and curriculum schedules. Family 3 combines layout annotations (pages with layout labels and bounding boxes) with cropped regions paired with transcription targets. Representative systems report several million to several tens of millions of mixed training samples, while their subsets for layout supervision range from tens of thousands to a few million pages [77], [80], [81]. Family 4 trains on broad multimodal corpora spanning photographs, charts, conversations, code, and documents, where document understanding is one of many target capabilities. Open technical reports describe hundreds of billions to several trillion pretraining tokens, followed by roughly one to tens of millions of supervised multimodal examples [158], [159], [160]; the proportions of documents are generally undisclosed.

![](images/d4e314fb97e8828d4bd4142f00f263f908705c1f92b3ef16d49ae659c7b2284e.jpg)  
Fig. 6. Data sources (left) and training pipelines (right) for the first four system families. Modular toolkits and task-specific models use taskspecific data and supervision; specialized document VLMs use pages and region crops for pretraining, supervised fine-tuning, and optiona reinforcement learning with verifiable rewards; hybrid workflows train separate layout and recognition models or a shared VLM for both tasks; and general-purpose VLMs use broad multimodal corpora with alignment, joint pretraining, instruction tuning, and optional post-training.

Training pipelines: Training methods also diverge across families (Fig. 6, right). Family 1 uses task-specific supervision: CTC or token cross-entropy for recognizers [23], [120], and detection or layout losses for other modules [41], [92]. Components may be trained separately, while end-toend spotters share features and jointly optimize localization and recognition [17], [18]. Family 2 is typically trained in two stages, pretraining for alignment between document images and text followed by supervised fine-tuning (SFT) on document tasks covering text fidelity, table structure, and formula syntax; a growing number of systems add a third stage of reinforcement learning with verifiable rewards (RLVR) that exploits OCR’s distinctive property of producing verifiable outputs. These objectives include rewards from executable unit tests for text presence and absence, reading order, tables, formulas, and abnormal repetition in olmOCR2 [76] and rewards that account for output format [79]. Family 3 uses either separate models or a shared VLM for layout analysis and recognition. With separate models, the layout detector is trained on annotated pages, while region parsers follow Family 2’s training pipeline or are reused from existing models. Dolphin-v2 fine-tunes the same VLM for both layout analysis and content parsing [82]. PaddleOCR-VL-1.5 also uses group relative policy optimization (GRPO) to reduce inconsistency in output style [77], [78]. With its 1.2B architecture fixed, MinerU2.5- Pro improves table TEDS on OmniDocBench v1.6 from 90.37 to 92.87 by adding supervised fine-tuning on hard examples mixed with earlier training data [63]. Within Family 3, MonkeyOCRv2 pretrains standalone ViT-Small/Base encoders from scratch on document images, using text and pixel reconstruction decoders that share the visual tokens [75]. For parsing, it freezes the encoder and connects it to Qwen3-0.6B through an MLP projector. At a long-side input resolution of 448 pixels, adding pixel reconstruction during pretraining raises its S-Parsing pipeline’s recognition accuracy on scrambled text from 55.4% to 72.1%, showing gains when linguistic context is disrupted [75]. Family 4 follows a pipeline with four stages: visual alignment (connector warmup), joint pretraining on multimodal and text data, SFT via instruction tuning, and optional post-training with direct preference optimization (DPO), reinforcement learning from human feedback (RLHF), or RLVR. In downstream document question answering, DocThinker [161] uses rule-based reinforcement learning with rewards for answer correctness, supporting-region localization, question reformulation, and output format.

## 3.4 Current Limitations

Families 2–4 generate output conditioned on the image. Learned linguistic and layout regularities should support recovery of the content in the source image, but when compression, limited resolution, or unfamiliar layouts weaken the visual evidence, generation may rely on learned priors, producing unsupported content or omitting difficult regions. Localization, reading order, and relations across regions add requirements beyond recognizing the text, and high input resolution with long structured outputs increases computation. Multilingual text and documents with many pages combine these difficulties.

Hallucination and omission: Hallucination and omission can arise when visual detail is lost, complex layouts are only partly covered, or decoding errors accumulate in long sequences [7], [8], [96]. Published evaluations report fabricated text and phantom table rows, as well as omissions, collapsed columns, and incorrect heading levels [96], [98]. MinerU2.5, MonkeyOCR, PaddleOCR-VL, and Dolphin-v2 use layout analysis and parsing of individual regions to shorten generation sequences and preserve local detail [77], [80], [81], [82]; MinerU2.5-Pro and olmOCR2 strengthen output fidelity through data verification and executable rewards [63], [76]. The olmOCR2 study reports that a data loader bug excluded blank pages from training and caused hallucinations on blank inputs, an error that benchmark scores did not register, since fixing it and retraining left them unchanged [76].

Spatial grounding and structural relations: Spatial grounding associates each output element, such as text, a formula, a table, or a layout block, with its region in the source image through a bounding box, coordinates, or a region identifier. It makes generated content verifiable against the image and supports reconstruction of reading order, hierarchy, and relations across regions. Its precision is bounded by the visual detail the encoder retains and by coordinate quantization in the output [21], [59]. Benchmark evaluations show that generalpurpose VLMs remain limited in text localization, layout perception, and parsing of complex elements [95], and plain Markdown, HTML, or LAT<sub>E</sub>X output generally omits the correspondence between elements and image regions altogether [8], [20]. Hybrid workflows add explicit structure: MonkeyOCR separates structure detection, region recognition, and reading-order prediction through its Where–What–How formulation [81]; Dolphin-v2 uses absolute coordinates and layout anchors to improve crop precision; and PaddleOCR-VL predicts regions defined by multiple points jointly with reading order and grounded text spotting [77], [82].

Inference cost: Inference cost arises from visual input at high resolution and from long structured output: dense pages produce many visual tokens, and generating one token at a time introduces latency that grows with output length. GLM-OCR combines multi-token prediction with parallel region recognition [64], Youtu-Parsing introduces token parallelism and query parallelism over shared page features [83], and P-MTP progressively increases lookahead depth and dynamically selects speculative lengths during inference [84]. Parallel processing of regions and multi-token decoding reduce sequential generation steps, while layout processing, scheduling, verification, and assembly add system overhead.

Multilingual text: Language coverage expanded in the earlier eras through dedicated datasets: CASIA-HWDB and IFN/ENIT for Chinese and Arabic handwriting [34], [162], the ICDAR 2019 MLT challenge for detection and recognition across ten languages in scene images [163], and XFUND for form understanding in seven languages [164]. Generative parsers inherit their coverage from the pretrained language model and from the documents added in fine-tuning (§3.3), and evaluations show where it falls short: the question-answering benchmark MTVQA reports low accuracy of leading VLMs on text-centric questions in nine languages [165], CC-OCR includes a multilingual reading track [166], and MDPBench reports diacritic omissions, character substitutions across writing systems, and errors in ordering text from right to left in document parsing [85].

Long and multipage documents: Most parsing benchmarks and training data use single pages (Table 2). A document with many pages requires continuing paragraphs and tables across page breaks, recovering a heading hierarchy over the whole document, and maintaining reading order across pages, while visual tokens and output length accumulate with page count. MPDocBench-Parse evaluates these requirements and reports weaknesses in merging content across pages and in recovering hierarchy [86]. Longdocument understanding also addresses the cost of processing many pages: DUDE and MMLongBench-Doc pose questions whose evidence is spread over many pages [167], [168], PDF-WuKong parses the document into text and image blocks and then selects evidence by sparse sampling conditioned on the query [87], and DocSeeker learns to locate evidence pages through distilled supervision and reinforcement learning [169].

## 4 EVOLUTION OF OCR BENCHMARKS

The system evolution of Section 3 changed what OCR benchmarks must measure. The required output has grown from a class label or a character string to a structured reconstruction of a whole page or document, with locations, reading order, tables, formulas, and markup returned together. Evaluation followed this growth in stages, each opened when systems outgrew the measure before it, and none closed by its successors. Section 4.1 traces the five stages; Section 4.2 reads published results as of October 2026 against them.

![](images/0f9f961d4e775a087354e56af385dbdba8240011d3b2b372abae7acad09581e7.jpg)  
Fig. 7. Evolution of evaluation relevant to OCR across five stages, combining representative input–output contracts with a chronological timeline of benchmark datasets. ❶ Text recognition: controlled characters, words, and lines (MNIST, IAM, CASIA-HWDB). ❷ Scene text: detection and recognition in natural images (ICDAR, SVT, Total-Text, CTW1500). ❸ Document structure components: tables and formulas as structured markup, plus layouts as labeled spatial regions (PubTabNet, PubLayNet, DocLayNet, HME100K). ❹ Downstream document understanding: question answering about documents (DocVQA, ChartQA, DUDE). ❺ Foundation Model OCR Evaluation, split into two tracks: VLM OCR ability diagnostics (OCRBench and OCRBench v2) and benchmarks for parsing full pages or multipage documents (OmniDocBench, Real5-OmniDocBench, MDPBench, MPDocBench-Parse, and PureDocBench). These evaluation directions expand and coexist; VLM diagnostic suites may combine transcription and understanding tasks. Stars mark selected landmark benchmarks; Table 2 lists dataset details and metrics.

## 4.1 How OCR Benchmarks Evolved

Five stages trace this development (Fig. 7 and Table 2): the input expands from isolated characters to complete documents, and the output from class labels through region transcripts and component structures to full-document markup.

The first stage tested controlled text recognition. MNIST [33] reduced the problem to classifying single digits, while IAM [170] and CASIA-HWDB [34] extended evaluation to handwritten characters and complete text lines. These protocols receive the text already located and cropped; finding it defined the next stage.

The second stage brought OCR into natural scenes. The ICDAR Robust Reading competitions [89], [171] and later datasets such as Total-Text [90] and CTW1500 [91] established protocols for scene text detection and end-to-end spotting, and word-crop benchmarks IIIT5K [172], SVT [88], SVTP [173], and CUTE80 [174] isolated scene text recognition. Video text was probed through the ICDAR 2013 video challenge [171] and BOVText [175]. The unit of judgment remained the word or region; HierText [176] grouped words into lines and paragraphs, while tables, formulas, and page-level structure stayed outside these protocols.

The third stage, document structure components, brought structure into the metric, one component at a time. PubTabNet [5] evaluated table structure as HTML with TEDS, FinTabNet [177] paired cell boxes with logical row–column structure and evaluated cell adjacency, and CROHME [178] and UniMER-Test [93] evaluated formula recognition under structural or LAT X conventions. Layout benchmarks followed a different contract: PubLayNet [92] and DocLayNet [179] formulate the task as localization and classification of predefined page regions and report mAP within their respective label spaces. Each component is scored under its own definition of structure, with no shared protocol across components.

The fourth stage, document understanding, judges systems by their answers. DocVQA [13], ChartQA [94], and DUDE [167] pair document images with natural language questions, while FUNSD [180] and CORD [181] target key information extraction. These downstream benchmarks evaluate answer or field accuracy and have influenced VLM system design.

The fifth stage, OCR evaluation in the Foundation Model Era, returned to reconstruction and divides into two tracks. OCRBench [11] and OCRBench v2 [95] diagnose VLM abilities across transcription, localization, and understanding tasks. OmniDocBench [96], [155], Real5- OmniDocBench [97], PureDocBench [98], MDPBench [85], and MPDocBench-Parse [86] evaluate reconstruction of complete pages and multipage documents across content fidelity, reading order, and document structure, placing multiple system families on shared pages while retaining metrics for individual components. With this stage the unit of evaluation caught up with the unit of system output: the page, and most recently the document.

## 4.2 What the Benchmarks Reveal

## 4.2.1 Quantitative Performance Trends

Published results show high scores on several common OCR benchmarks, with small recent gains on the selected scene text benchmarks. Table structure within pages and physical capture conditions still distinguish systems.

Scene text. CTW1500 detection Hmean increased from 75.6 for TextSnake to 88.5 for TPD-STR; under the MJ+ST setting, IIIT5K recognition accuracy increased from 82.9 for CRNN to 97.7 for SVTRv2; and Total-Text F1 without a lexicon increased from 65.3 for Mask TextSpotter v2 to 84.6 for TiCLS (Tables 3–5). From the 2022 entries to the latest ones, the gains are 1.4, 0.7, and 1.0 points: most of the distance was covered within the Deep Learning Era, and the open problems moved to harder text distributions and to full pages.

Document structure components. Reported scores are also high on several document structure benchmarks. Reported TEDS scores include 88.3 for EDD on the original PubTabNet test set [5] and 96.3 for TableMASTER on the ICDAR 2021 test split [219]; the different test sets prevent a direct estimate of improvement. On PubLayNet, reported mAP includes 91.0 for Mask R-CNN in the RoDLA evaluation [220] and 96.2 for VGT [221], and UniMERNet reached 96.8 CDM on UniMER-Test in a 2025 metric study [101]. These scores characterize individual components; full-page benchmarks additionally assess how their outputs preserve reading order and relations between elements.

VLM OCR diagnostics. Shared diagnostic suites measure the reading ability that general-purpose VLMs expose through prompts. Selected OCRBench results range from 736 for GPT-4o to 931 for Qwen3.5-397B-A17B [11], [222], an increase of 195 points that leaves the leading score 69 points below the nominal ceiling of 1,000. The total aggregates 1,000 question-level judgments spanning text recognition, scene-text and document question answering, key information extraction, and handwritten mathematical expression recognition [11], so a change in the total cannot be attributed to any single ability, and reconstruction of complete pages lies outside the task set.

Parsing complete pages. Benchmarks for parsing complete pages evaluate reconstruction under a common output protocol. On OmniDocBench v1.6 full, the selected results span 93.26–96.91 Overall, 95.60–97.35 for text, 88.04–96.82 for tables, and 95.08–97.53 for formulas (Table 6) [58], [61], [155]. Text scores of the eight systems lie within 1.75 points of one another, and table scores spread over 8.78 points, so tables are the component that still separates them. TeleOCR and OvisOCR2 hold the top two positions, one from layoutguided regional generation and one from whole-page generation. Every system above 95 Overall has at most 1.2B parameters, while the 2B to 4B systems score at most 94.00. Within these selected results, the choice between the two designs does not decide the ranking, and larger models do not rank higher.

Physical robustness. Real5-OmniDocBench evaluates scanning, warping, screen photography, illumination change, and skew over the same 1,355 source pages from OmniDocBench v1.5. Across the selected systems on its current official leaderboard, overall scores span 64.45–93.19 (Table 7) [216]. The modular toolkit PP-StructureV3 falls from 84.68 on scans to 37.98 under skew, while every listed generative system stays above 85 in all five conditions, and their overall scores lie within 4.2 points of one another. Each image is a re-capture of one source page under a single acquisition condition [97], so documents that exist only as camera captures lie outside the benchmark’s coverage.

## 4.2.2 What Current Metrics Miss

Across protocols, interpretation is limited by three recurring issues: differences in output definitions and matching rules, aggregation across heterogeneous components, and incomplete coverage of content, document conditions, and reconstruction errors.

Ambiguity in output definitions and protocols. Document structure admits multiple reasonable decompositions and serializations. Borderless tables, aligned fields, forms, and layouts with multiple columns can cross category boundaries, while visually equivalent tables may have different HTML or LAT X representations. Benchmarks also use different region ontologies, annotation granularities, and matching rules, as illustrated by PubLayNet, DocLayNet, and OmniDocBench [92], [96], [155], [179]. A reported score therefore reflects both the system’s reconstruction and the protocol’s definition of structure.

Composite score attribution. OmniDocBench v1.6 illustrates the attribution problem: its Overall score combines text, table, and formula metrics, while reading order is reported separately [155]. Similar overall scores can therefore represent different component profiles.

Coverage boundaries. Current benchmarks for complete pages concentrate on text, mathematical formulas, tables, layout regions, and reading order. Charts, flowcharts, chemical structures, and sheet music also encode content and structure that can be recovered from visual evidence; GOT, Logics-Parsing, and OCRVerse already output some of them

## TABLE 2

Representative benchmarks across five stages of evaluation relevant to OCR (newest first within each subgroup). Year gives the first public paper or dataset release, or the named version release. Input uses Page for a single page, Multipage for a document, and Mixed for heterogeneous   
inputs. Geometry denotes boxes or polygons. Main metrics follow official protocols; ED denotes edit distance, CDM evaluates formulas, and TEDS evaluates tables. Scale refers to the named release or split; K/M counts are rounded and test splits are marked. Access icons link to official repositories, dataset pages, or papers. Full list: project page.

<table><tr><td>Benchmark</td><td>Year</td><td>Input</td><td>Output</td><td>Main metrics</td><td>Scale</td><td>Access</td></tr><tr><td colspan="7">0 Controlled text recognition</td></tr><tr><td colspan="7">Character classification</td></tr><tr><td>EMNIST [182]</td><td>2017</td><td>Crop</td><td>Labels</td><td>Accuracy</td><td>814K images</td><td>000</td></tr><tr><td>SVHN [183]</td><td>2011</td><td>Crop</td><td>Labels</td><td>Accuracy</td><td>600K+ images</td><td></td></tr><tr><td>MNIST [33]</td><td>1998</td><td>Crop</td><td>Labels Handwriting character, line, and word recognition</td><td>Accuracy</td><td>70K images</td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>CASIA-HWDB 1.x/2.x [34]</td><td>2011</td><td>Page, Crop</td><td>Text, Labels</td><td>Accuracy, CR/AR</td><td>3.9M chars, 5.1K pages</td><td>我进进井</td></tr><tr><td>RIMES 2011 (line) [184]</td><td>2011</td><td>Crop</td><td>Text</td><td>CER, WER</td><td>12,111 lines</td><td></td></tr><tr><td>IAM [170]</td><td>2002</td><td>Crop</td><td>Text</td><td>CER, WER</td><td>13.4K lines</td><td></td></tr><tr><td>IFN/ENIT [162]</td><td>2002</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>26,459 images</td><td></td></tr><tr><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td>Θ Scene text Detection and spotting</td><td></td><td></td><td></td></tr><tr><td>HierText [176]</td><td>2022</td><td>Scene</td><td>Masks, Hierarchy</td><td>PQ</td><td>11,639 images</td><td></td></tr><tr><td>TextOCR [185]</td><td>2021</td><td>Scene</td><td>Text, Geometry</td><td>Det. F1, Word acc.</td><td>28,134 images</td><td></td></tr><tr><td>ReCTS [186]</td><td>2019</td><td>Scene</td><td>Text, Geometry</td><td>Hmean, 1-NED</td><td>25K images</td><td></td></tr><tr><td>LSVT [187]</td><td>2019</td><td>Scene</td><td>Text, Geometry</td><td>Hmean, 1-NED</td><td>450K images (400K weak)</td><td></td></tr><tr><td>ArT [188]</td><td>2019</td><td>Scene</td><td>Text, Geometry</td><td>Hmean, 1-NED</td><td>10K images</td><td></td></tr><tr><td>MLT 2019 [163]</td><td>2019</td><td>Scene</td><td>Text, Geometry, Script</td><td>Hmean, Script acc., 1-NED</td><td>20K images</td><td></td></tr><tr><td>CTW1500 [91]</td><td>2017</td><td>Scene</td><td>Geometry</td><td>P/R/F1</td><td>1.5K images</td><td></td></tr><tr><td>Total-Text [90]</td><td>2017</td><td>Scene</td><td>Geometry</td><td>P/R/F1</td><td>1.6K images</td><td></td></tr><tr><td>MLT 2017 [189]</td><td>2017</td><td>Scene</td><td>Geometry, Script</td><td>Hmean, Script acc.</td><td>18K images</td><td></td></tr><tr><td>COCO-Text [190]</td><td>2016</td><td>Scene</td><td>Geometry</td><td>P/R/F1</td><td>63K images</td><td></td></tr><tr><td>ICDAR 2015 [89]</td><td>2015</td><td>Scene</td><td>Geometry</td><td>P/R/F1</td><td>1.5K images</td><td></td></tr><tr><td>ICDAR 2013 [171]</td><td>2013</td><td>Scene</td><td>Geometry</td><td>P/R/F1</td><td>462 images</td><td></td></tr><tr><td>MSRA-TD500 [191]</td><td>2012 Scene</td><td></td><td>Geometry</td><td>P/R/F1</td><td>500 images</td><td>C井安共安共CC共安共共弁</td></tr><tr><td></td><td></td><td></td><td>Cropped word recognition</td><td></td><td></td><td></td></tr><tr><td>Union14M-Bench [73]</td><td>2023</td><td>Crop</td><td>Text</td><td>Word acc., WAICS</td><td>409K images (test)</td><td>0</td></tr><tr><td>ICDAR 2015 (rec.) [89]</td><td>2015</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>2,077 images (test)</td><td></td></tr><tr><td>CUTE80 [174]</td><td>2014</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>288 images (test)</td><td></td></tr><tr><td>SVTP [173]</td><td>2013</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>639 images (test)</td><td></td></tr><tr><td>ICDAR 2013 (rec.) [171]</td><td>2013</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>1,095 images (test)</td><td></td></tr><tr><td>IIIT5K [172] SVT [88], [192]</td><td>2012</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>3,000 images (test)</td><td>我我进进进</td></tr><tr><td></td><td>2010</td><td>Crop</td><td>Text</td><td>Word acc.</td><td>647 images (test)</td><td></td></tr><tr><td>BOVText [175]</td><td>2021</td><td>Video</td><td>Video text spotting Text, Tracks</td><td>MOTA, MOTP, IDF1</td><td>2,021 videos, 1.75M frames</td><td>0</td></tr><tr><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td>© Document structure components</td><td></td><td></td><td></td></tr><tr><td>PubTables-1M [193]</td><td></td><td></td><td>Tables</td><td>AP, GriTS</td><td>947.6K tables</td><td></td></tr><tr><td>WTW [194]</td><td>2021 2021</td><td>Page, Crop</td><td>Geometry, Structure Geometry, Structure</td><td>F1, TEDS</td><td>14,581 images</td><td></td></tr><tr><td>FinTabNet [177]</td><td>2020</td><td>Scene, Page</td><td>Geometry, Structure</td><td>Adjacency F1</td><td>112,887 tables</td><td></td></tr><tr><td>PubTabNet [5]</td><td>2019</td><td>Crop</td><td>HTML</td><td>TEDS</td><td>568K tables</td><td></td></tr><tr><td>SciTSR [195]</td><td></td><td>Crop</td><td>Cell graph</td><td>Structure F1</td><td>15K tables</td><td>Ccccc</td></tr><tr><td>TableBank [196]</td><td>2019 2019</td><td>Crop Page, Crop</td><td>HTML, Geometry</td><td>P/R/F1, BLEU</td><td>417K det., 145K struct.</td><td></td></tr><tr><td></td><td></td><td></td><td>Formulas</td><td></td><td></td><td></td></tr><tr><td>UniMER-Test [93]</td><td>2024</td><td>Crop</td><td>LTEX</td><td>ExpRate, ED, CDM</td><td>23,757 images</td><td>0</td></tr><tr><td>HME100K [197]</td><td>2022</td><td>Crop</td><td>LTEX</td><td>ExpRate</td><td>100K images</td><td></td></tr><tr><td>CROHME 2016 [178]</td><td>2016</td><td>Strokes</td><td>Symbols, Relations</td><td>ExpRate</td><td>1,147 expressions (test)</td><td> O0C</td></tr><tr><td>Im2LaTeX-100K [6]</td><td>2016</td><td>Crop</td><td>LATEX</td><td>BLEU, Image ED/match</td><td>100K images</td><td></td></tr><tr><td></td><td></td><td></td><td>Layout element detection</td><td></td><td></td><td></td></tr><tr><td>DocLayNet [179]</td><td>2022</td><td>Page</td><td>Geometry, Labels</td><td>COCO mAP</td><td>80,863 pages</td><td></td></tr><tr><td>DocBank [198] PubLayNet [92]</td><td>2020</td><td>Page</td><td>Token labels</td><td>F1</td><td>500K pages</td><td>CCC</td></tr><tr><td></td><td>2019</td><td>Page</td><td>Geometry, Labels</td><td>COCO mAP</td><td>360K pages</td><td></td></tr><tr><td colspan="7"> Document understanding†</td></tr><tr><td></td><td></td><td></td><td>Document and scene text VQA</td><td></td><td></td><td></td></tr><tr><td>OCR-Reasoning [199]</td><td>2025</td><td>Mixed</td><td>Answer, Rationale</td><td>Accuracy, Reasoning score</td><td>1,069 QA</td><td></td></tr><tr><td>DUDE [167]</td><td>2023</td><td>Multipage</td><td>Answer, Confidence</td><td>ANLS, ECE, AURC</td><td>41.5K QA</td><td></td></tr><tr><td>ChartQA [94]</td><td>2022</td><td>Chart</td><td>Answer</td><td>Relaxed acc.</td><td>32.7K QA 50K QA</td><td>CCCOO</td></tr><tr><td>DocVQA [13] TextVQA [200]</td><td>2020 2019</td><td>Page Scene</td><td>Answer Answer</td><td>ANLS VQA acc.</td></table>

<sup>†</sup> Stage ❹ is a transition through downstream document understanding; marked suites involve semantic extraction or reasoning beyond faithful transcription.

TABLE 3  
Scene text detection Hmean/F1 (%).
<table><tr><td>Method</td><td>Date</td><td>IC15</td><td>Total</td><td>CTW</td></tr><tr><td>PSENet [116]</td><td>2018.06</td><td>85.7</td><td>80.9</td><td>82.2</td></tr><tr><td>TextSnake [203]</td><td>2018.07</td><td>82.6</td><td>78.4</td><td>75.6</td></tr><tr><td>PAN [204]</td><td>2019.08</td><td>82.9</td><td>85.0</td><td>83.7</td></tr><tr><td>DBNet [41]</td><td>2019.11</td><td>87.3</td><td>84.7</td><td>84.9</td></tr><tr><td>TextFuseNet [205]</td><td>2020.07</td><td>92.1</td><td>87.1</td><td>86.6</td></tr><tr><td>FCENet [206]</td><td>2021.04</td><td>86.2</td><td>85.8</td><td>85.5</td></tr><tr><td>TESTR [207]</td><td>2022.04</td><td>90.0</td><td>86.9</td><td>87.1</td></tr><tr><td>TPD-STR [208]</td><td>2025.02</td><td>89.4</td><td>88.7</td><td>88.5</td></tr></table>

TABLE 4

Scene text recognition accuracy (%) under MJ+ST, following the comparison compiled in SVTRv2 [56]. IC13/IC15 use the 857/1811-image subsets.

<table><tr><td colspan="6"></td></tr><tr><td>Model</td><td>Date</td><td>Params</td><td><img src="images/03bee39301c5c96baedb16e546c5e3359d21863fc4c3e8a98800f01931a71985.jpg"/></td><td><img src="images/03d600f2ac2e45cf569cc4d5ccc65c0691e4d2c3e5dcbcd92f54a578a62e2d3a.jpg"/> <img src="images/608fcaba8859fe9fe8028bb11947acb44b2aab286ec2cf02d222735cc3d3587b.jpg"/></td><td><img src="images/615c2fa0ccbc8457f476583aae6ebed4cd4f3c85a0d61990e6c036a2d1a678e2.jpg"/></td><td><img src="images/3d991d2a58daea24cb6d122e7ff2c2d4ecc900a4d112d5e30cec70fc4f6dbfa9.jpg"/> <img src="images/a2ead471c6e8739bae41e908a827a2af70a63975de5b3991cd2506253eb98261.jpg"/></td></tr><tr><td>CRNN [23]</td><td>2015.07</td><td>8.3M</td><td>82.9</td><td>81.6 91.1 69.4 70.0</td><td></td><td>65.5</td></tr><tr><td>ASTER [42]</td><td>2018.06</td><td>27.2M</td><td>93.3</td><td>90.0 90.8 74.7 80.2</td><td></td><td>80.9</td></tr><tr><td>ABINet [119]</td><td>2021.03</td><td>36.7M</td><td>96.2</td><td></td><td>93.5 97.4 86.089.3</td><td>89.2</td></tr><tr><td>SVTR-B [120]</td><td>2022.05</td><td>24.6M</td><td>96.0</td><td></td><td>91.5 97.1 85.2 89.9</td><td>91.7</td></tr><tr><td>PARSeq [105]</td><td>2022.07 SVTRv2-B [56] 2024.11</td><td>23.8M 19.8M</td><td>97.0 97.7</td><td>94.0 97.3 88.1 91.2</td><td>93.6 97.0 86.588.9</td><td>92.2 95.8</td></tr></table>

TABLE 5

End-to-end scene text spotting F1 (%). Dates refer to the first public papers. “None” is lexicon-free; “Full” uses all words in the test set.
<table><tr><td></td><td></td><td>IC15</td><td colspan="2">Total-Text</td><td colspan="2">CTW1500</td></tr><tr><td>Method</td><td>Date</td><td>None</td><td>None</td><td>Full</td><td>None</td><td>Full</td></tr><tr><td>FOTS [44]</td><td>2018</td><td></td><td>一</td><td></td><td>21.1</td><td>39.7</td></tr><tr><td>Mask TextSpotter v2 [209]</td><td>2019</td><td></td><td>65.3</td><td>77.4</td><td></td><td></td></tr><tr><td>TextDragon [210]</td><td>2019</td><td>一</td><td>48.8</td><td>74.8</td><td>39.7</td><td>72.4</td></tr><tr><td>ABCNet [121]</td><td>2020</td><td>一</td><td>64.2</td><td>75.7</td><td>45.2</td><td>74.1</td></tr><tr><td>ABCNet v2 [211]</td><td>2021</td><td></td><td>70.4</td><td>78.1</td><td>57.5</td><td>77.2</td></tr><tr><td>TESTR [207]</td><td>2022</td><td>65.3</td><td>73.3</td><td>83.9</td><td>56.0</td><td>81.5</td></tr><tr><td>DeepSolo [45]</td><td>2022</td><td>73.8</td><td>83.6</td><td>89.6</td><td>64.2</td><td>81.4</td></tr><tr><td>ESTextSpotter [212]</td><td>2023</td><td></td><td>80.8</td><td>87.1</td><td>64.9</td><td>83.9</td></tr><tr><td>DG-Bridge [213]</td><td>2024</td><td></td><td>83.3</td><td>88.3</td><td>69.8</td><td>83.9</td></tr><tr><td>TiCLS [214]</td><td>2026</td><td>79.1</td><td>84.6</td><td>90.4</td><td>66.2</td><td>88.1</td></tr></table>

TABLE 6

Selected OmniDocBench v1.6 full scores [155] as of October 2026. Overall/Text/Table/Fml. are % (↑); Text is (1 − Edit)×100, Table is TEDS, Fml. is CDM, and RO is reading order Edit (↓).
<table><tr><td>System</td><td>Date</td><td>Params</td><td>Overall</td><td>Text</td><td>Table</td><td>Fml.</td><td>RO</td></tr><tr><td colspan="6">Specialized document VLMs</td><td></td><td></td></tr><tr><td>OvisOCR2 [58]</td><td>2026.07</td><td>0.8B</td><td>96.47</td><td>97.35</td><td>94.58</td><td>97.49</td><td>0.112</td></tr><tr><td>Unlimited-OCR [135]</td><td>2026.06</td><td>3B</td><td>94.00</td><td>96.06</td><td>90.21</td><td>95.72</td><td>0.128</td></tr><tr><td>Qianfan-OCR [59]</td><td>2026.03</td><td>4B</td><td>93.90</td><td>96.00</td><td>90.53</td><td>95.08</td><td>0.130</td></tr><tr><td>FireRed-OCR [149]</td><td>2026.02</td><td>2B</td><td>93.26</td><td>96.30</td><td>88.04</td><td>95.44</td><td>0.131</td></tr><tr><td colspan="8">Hybrid workflows</td></tr><tr><td>TeleOCR [61]</td><td>2026.08</td><td>1.2 B</td><td>96.91</td><td>97.33</td><td>96.82</td><td>96.59</td><td>0.118</td></tr><tr><td>PaddleOCR-VL-1.6 [62]</td><td>2026.05</td><td>0.9B</td><td>96.34</td><td>96.74</td><td>94.76</td><td>97.53</td><td>0.128</td></tr><tr><td>MinerU2.5-Pro-2605 [215]</td><td>2026.05</td><td>1.2B</td><td>95.72</td><td>96.40</td><td>93.62</td><td>97.15</td><td>0.123</td></tr><tr><td>GLM-OCR [64]</td><td>2026.02</td><td>0.9B</td><td>95.22</td><td>95.60</td><td>92.83</td><td>97.18</td><td>0.133</td></tr></table>

from page images [8], [10], [223]. ParseBench extends evaluation to charts by checking sampled data points and their

TABLE 7  
Selected Real5-OmniDocBench leaderboard scores [216] (%,↑), as of October 2026, using the pinned OmniDocBench v1.5 protocol. S/W/SP/I/SK = scan/warp/screen/illum./skew; All averages the five condition scores; Worst is their minimum.
<table><tr><td>System</td><td>All</td><td>S</td><td>W</td><td>SP</td><td>I</td><td>SK</td><td>Worst</td></tr><tr><td colspan="6">Modular toolkits and task-specific models</td><td></td><td></td><td></td></tr><tr><td>PP-StructureV3 [145]</td><td>64.45</td><td>84.68</td><td>59.34</td><td>66.89</td><td>73.38</td><td>37.98</td><td></td><td>37.98</td></tr><tr><td colspan="9">Specialized document VLMs</td></tr><tr><td>OvisOCR2 [58]</td><td>92.29</td><td>93.77</td><td>91.40</td><td></td><td>93.09</td><td>92.88</td><td>90.33</td><td>90.33</td></tr><tr><td colspan="9">Hybrid workflows</td></tr><tr><td>PaddleOCR-VL-1.6 [62]</td><td>93.19</td><td>94.74</td><td></td><td>92.48</td><td>92.78</td><td>93.28</td><td>92.66</td><td>92.48</td></tr><tr><td>Xiaomi-OCR-0 [217]</td><td>92.73</td><td>93.73</td><td></td><td>92.76</td><td>92.80</td><td>92.68</td><td>91.69</td><td>91.69</td></tr><tr><td>TeleOCR [61]</td><td>90.72</td><td>92.33</td><td></td><td>90.63</td><td>89.09</td><td>90.90</td><td>90.63</td><td>89.09</td></tr><tr><td>GLM-OCR [64]</td><td>90.32</td><td>92.67</td><td></td><td>90.68</td><td>91.75</td><td>91.12</td><td>85.39</td><td>85.39</td></tr><tr><td>MonkeyOCRv2-B-Parsing [75]</td><td>89.22</td><td>89.49</td><td></td><td>89.70</td><td>88.40</td><td>88.54</td><td>89.97</td><td>88.40</td></tr><tr><td colspan="9">General-purpose VLMs</td></tr><tr><td>Doubao-Seed-2.1-Pro [218]</td><td>89.02</td><td>88.85</td><td></td><td>89.36</td><td>88.99</td><td>89.13</td><td>88.79</td><td>88.79</td></tr></table>

labels [202], leaving the completeness of chart reconstruction outside this metric.

Coverage also varies by language, document length, and capture condition. MDPBench evaluates digital and photographed documents in 17 languages and reports larger errors on non-Latin scripts and photographed pages, particularly among open-source models [85]. Together with the controlled recaptures in Real5-OmniDocBench [97], these results motivate reporting performance by language and capture condition. MPDocBench-Parse evaluates cross-page text and table merging and heading hierarchies in English and Chinese [86]. Extending these document-level tests to more languages would help establish whether multilingual reading also preserves structure across page breaks.

Faithfulness also requires targeted error checks. CHAOS-Bench measures whether models preserve words with deliberately altered characters when visual evidence conflicts with language priors [57]; a missed match can reflect omission, correction, or another substitution. ParseBench separately includes checks for missing, unexpected, and repeated content [202]. Reporting these error types alongside aggregate text scores would show whether a system alters, omits, or invents content.

## 5 DISCUSSION AND OUTLOOK

This survey defines OCR through faithful transcription, traces its development from specialist to unified models, and examines recognition and structural recovery using published benchmark results. Looking ahead, three expectations stand out.

First, as models and agents increasingly use OCR outputs, hallucinations, omissions, and structural errors can affect downstream results. Future work should strengthen verification of content accuracy, completeness, and structural relations against the source image. Second, requirements for accuracy, cost, and control differ across uses, so specialist, unified, and hybrid approaches will continue to coexist. Third, the object of reconstruction will keep widening, from text to the structure of whole documents, from single pages to many, from a few scripts to all writing systems, and from clean scans to whatever a camera captures, and evaluation must widen with it.

OCR’s lasting value lies in turning visually recorded information into accurate digital content that can be searched, edited, processed, and reused.

## REFERENCES

[1] H. F. Schantz, The History of OCR, Optical Character Recognition. Manchester Center, VT: Recognition Technologies Users Association, 1982.

[2] S. Mori, C. Y. Suen, and K. Yamamoto, “Historical review of OCR research and development,” Proceedings of the IEEE, vol. 80, no. 7, pp. 1029–1058, 1992.

[3] R. Plamondon and S. N. Srihari, “Online and off-line handwriting recognition: A comprehensive survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 1, pp. 63–84, 2000.

[4] Q. Ye and D. Doermann, “Text detection and recognition in imagery: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 7, pp. 1480–1500, 2015.

[5] X. Zhong, E. ShafieiBavani, and A. Jimeno-Yepes, “Image-based table recognition: Data, model, and evaluation,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020.

[6] Y. Deng, A. Kanervisto, J. Ling, and A. M. Rush, “Image-tomarkup generation with coarse-to-fine attention,” Proceedings of the 34th International Conference on Machine Learning (ICML), pp. 980–989, 2017.

[7] G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, and S. Park, “OCR-free document understanding transformer,” in Proceedings ofthe European Conference on Computer Vision (ECCV), 2022.

[8] H. Wei, C. Liu, J. Chen, J. Wang, L. Kong, Y. Xu, Z. Ge, L. Zhao, J. Sun, Y. Peng, C. Han, and X. Zhang, “General OCR theory: Towards OCR-2.0 via a unified end-to-end model,” arXiv preprint arXiv:2409.01704, 2024.

[9] X. Chen, L. Jin, Y. Zhu, C. Luo, and T. Wang, “Text recognition in the wild: A survey,” ACM Computing Surveys, vol. 54, no. 2, 2021.

[10] Y. Zhong, L. Chen, X. Zhao et al., “OCRVerse: Towards holistic OCR in end-to-end vision-language models,” arXiv preprint arXiv:2601.21639, 2026.

[11] Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X. Yin, C.-L. Liu, L. Jin, and X. Bai, “OCRBench: On the hidden mystery of OCR in large multimodal models,” Science China Information Sciences, 2024.

[12] W. Hwang, J. Yim, S. Park, S. Yang, and M. Seo, “Spatial dependency parsing for semi-structured document information extraction,” in Findings of the Association for Computational Linguistics: ACL 2021, 2021.

[13] M. Mathew, D. Karatzas, and C. V. Jawahar, “DocVQA: A dataset for VQA on document images,” in Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2021, pp. 2200–2209.

[14] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. tau Yih, T. Rockt¨ aschel, S. Riedel,¨ and D. Kiela, “Retrieval-augmented generation for knowledgeintensive NLP tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.

[15] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding, and J. Tang, “CogAgent: A visual language model for GUI agents,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 281–14 290.

[16] Y. Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y. Bai, Z. Yu, Y. Yang, Q. Dang, and H. Wang, “PP-OCR: A practical ultra lightweight OCR system,” arXiv preprint arXiv:2009.09941, 2020.

[17] P. Lyu, M. Liao, C. Yao, W. Wu, and X. Bai, “Mask TextSpotter: An end-to-end trainable neural network for spotting text with arbitrary shapes,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 67–83. [Online]. Available: https://arxiv.org/abs/1807.02242

[18] D. Peng, X. Wang, Y. Liu, J. Zhang, M. Huang, S. Lai, J. Li, S. Zhu, D. Lin, C. Shen, X. Bai, and L. Jin, “SPTS: Single-point text spotting,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 4272–4281. [Online]. Available: https://arxiv.org/abs/2112.07917

[19] J. Wan, S. Song, W. Yu, Y. Liu, W. Cheng, F. Huang, X. Bai, C. Yao, and Z. Yang, “OmniParser: A unified framework for text spotting, key information extraction and table recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 15 641–15 653. [Online]. Available: https://arxiv.org/abs/2403.19128

[20] L. Blecher, G. Cucurull, T. Scialom, and R. Stojnic, “Nougat: Neural optical understanding for academic documents,” arXiv preprint arXiv:2308.13418, 2023.

[21] Hunyuan Vision Team, P. Lyu, X. Wan, G. Li, S. Peng, W. Wang, L. Wu, H. Shen, Y. Zhou, C. Tang, Q. Yang, Q. Peng, B. Luo, H. Yang, X. Zhang, J. Zhang, H. Peng, H. Yang, S. Xie, L. Zhou, G. Pei, B. Wu, R. Yan, K. Wu, J. Yang, B. Wang, K. Liu, J. Zhu, J. Jiang, Linus, H. Hu, and C. Zhang, “HunyuanOCR technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.19575

[22] S. Impedovo, L. Ottaviano, and S. Occhinegro, “Optical character recognition — a survey,” International Journal of Pattern Recognition and Artificial Intelligence, vol. 5, no. 01n02, pp. 1–24, 1991.

[23] B. Shi, X. Bai, and C. Yao, “An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 11, pp. 2298–2304, 2017.

[24] Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, “LayoutLM: Pre-training of text and layout for document image understanding,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020, pp. 1192–1200.

[25] Y. Zhu, C. Yao, and X. Bai, “Scene text detection and recognition: Recent advances and future trends,” Frontiers of Computer Science, vol. 10, no. 1, pp. 19–36, 2016.

[26] S. Long, X. He, and C. Yao, “Scene text detection and recognition: The deep learning era,” International Journal of Computer Vision, vol. 129, pp. 161–184, 2021.

[27] C. Liu, L. Jin, X. Bai, X. Li, and F. Yin, “Frontiers of intelligent document analysis and recognition: Review and prospects,” Journal of Image and Graphics, vol. 28, no. 8, pp. 2223–2252, 2023.

[28] H. Li, Y. Liu, W. Liao, M. Huang, S. Zhang, and L. Jin, “OCR in the era of large models: Current status and prospects,” Journal of Image and Graphics, vol. 30, no. 6, pp. 2023–2050, 2025.

[29] Y. Shu, W. Zeng, F. Zhao, Z. Chen, Z. Li, X. Yang, Y. Zhou, P. Rota, X. Bai, L. Jin, X.-C. Yin, and N. Sebe, “Visual text processing: A comprehensive review and unified evaluation,” arXiv preprint arXiv:2504.21682, 2025.

[30] P. Fu, T. Guan, Z. Wang, Z. Guo, C. Duan, H. Sun, B. Chen, J. Ma, Q. Jiang, K. Zhou, and J. Luo, “Multimodal large language models for text-rich image understanding: A comprehensive review,” arXiv preprint arXiv:2502.16586, 2025.

[31] L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition,” Proceedings ofthe IEEE, vol. 77, no. 2, pp. 257–286, 1989.

[32] C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

[33] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[34] C.-L. Liu, F. Yin, D.-H. Wang, and Q.-F. Wang, “CASIA online and offline Chinese handwriting databases,” in Proceedings of the 11th International Conference on Document Analysis and Recognition (ICDAR), 2011, pp. 37–41.

[35] U. Pal and B. B. Chaudhuri, “Indian script character recognition: A survey,” Pattern Recognition, vol. 37, no. 9, pp. 1887–1899, 2004.

[36] R. Smith, “An overview of the Tesseract OCR engine,” Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR), pp. 629–633, 2007.

[37] J. Matas, O. Chum, M. Urban, and T. Pajdla, “Robust widebaseline stereo from maximally stable extremal regions,” Image and Vision Computing, vol. 22, no. 10, pp. 761–767, 2004.

[38] B. Epshtein, E. Ofek, and Y. Wexler, “Detecting text in natural scenes with stroke width transform,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2010, pp. 2963–2970.

[39] L. Neumann and J. Matas, “Real-time scene text localization and recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3538–3545.

[40] X. Zhou, C. Yao, H. Wen, Y. Wang, S. Zhou, W. He, and J. Liang, “EAST: An efficient and accurate scene text detector,” in Pro-

ceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2642–2651.

[41] M. Liao, Z. Wan, C. Yao, K. Chen, and X. Bai, “Real-time scene text detection with differentiable binarization,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 7, 2020, pp. 11 474–11 481.

[42] B. Shi, M. Yang, X. Wang, P. Lyu, C. Yao, and X. Bai, “ASTER: An attentional scene text recognizer with flexible rectification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 9, pp. 2035–2048, 2019.

[43] M. Li, T. Lv, J. Chen, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, “TrOCR: Transformer-based optical character recognition with pre-trained models,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

[44] X. Liu, D. Liang, S. Yan, D. Chen, Y. Qiao, and J. Yan, “FOTS: Fast oriented text spotting with a unified network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 5676–5685.

[45] M. Ye, J. Zhang, S. Zhao, J. Liu, T. Liu, B. Du, and D. Tao, “DeepSolo: Let transformer decoder with explicit points solo for text spotting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[46] Y. Liu, J. Zhang, D. Peng, M. Huang, X. Wang, J. Tang, C. Huang, D. Lin, C. Shen, X. Bai, and L. Jin, “SPTS v2: Single-point scene text spotting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 15 665–15 679, 2023. [Online]. Available: https://arxiv.org/abs/2301.01635

[47] S. Appalaraju, B. Jasani, B. U. Kota, Y. Xie, and R. Manmatha, “DocFormer: End-to-end transformer for document understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 993–1003.

[48] Z. Tang, Z. Yang, G. Wang, Y. Fang, Y. Liu, C. Zhu, M. Zeng, C. Zhang, and M. Bansal, “Unifying vision, text, and layout for universal document processing,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[49] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10 012–10 022.

[50] M. Dehghani, B. Mustafa, J. Djolonga, J. Heek, M. Minderer, M. Caron, A. Steiner, J. Puigcerver, R. Geirhos, I. Alabdulmohsin, A. Oliver, P. Padlewski, A. Gritsenko, M. Luciˇ c, and N. Houlsby,´ “Patch n’ pack: NaViT, a vision transformer for any aspect ratio and resolution,” Advances in Neural Information Processing Systems, vol. 36, 2023.

[51] NAVER CLOVA AI Research, “SynthDoG: Synthetic document generator,” GitHub repository, 2021. [Online]. Available: https://github.com/clovaai/donut/tree/master/synthdog

[52] K. Lee, M. Joshi, I. Turc, H. Hu, F. Liu, J. Eisenschlos, U. Khandelwal, P. Shaw, M.-W. Chang, and K. Toutanova, “Pix2Struct: Screenshot parsing as pretraining for visual language understanding,” Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.

[53] W. Yu, Z. Yang, J. Wan, S. Song, J. Tang, W. Cheng, Y. Liu, and X. Bai, “OmniParser V2: Structured-points-of-thought for unified visual text parsing and its generality to multimodal large language models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 8, pp. 9210–9227, 2026. [Online]. Available: https://arxiv.org/abs/2502.16161

[54] Y. Zhang, X. Wang, M. Lin, Y. Zhang, P. Deng, T. Sun, T. Gao, Z. Zhang, J. Liu, C. Zhou, H. Liu, S. Liang, C. Cui, Y. Liu, D. Yu, and Y. Ma, “PP-OCRv6: From 1.5M to 34.5M parameters, surpassing billion-scale VLMs on OCR tasks,” arXiv preprint arXiv:2606.13108, 2026.

[55] C. Auer, M. Lysak, A. Nassar, M. Dolfi, N. Livathinos, P. Vagenas, C. B. Ramis, M. Omenetti, F. Lindlbauer, K. Dinkla, L. Mishra, Y. Kim, S. Gupta, R. T. de Lima, V. Weber, L. Morin, I. Meijer, V. Kuropiatnyk, and P. W. J. Staar, “Docling technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2408.09869

[56] Y. Du, Z. Chen, H. Xie, C. Jia, and Y.-G. Jiang, “SVTRv2: CTC beats encoder-decoder models in scene text recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[57] G. Li, X. Wan, S. Peng, W. Wang, H. Feng, Y. Du, B. Wu, Z. Ruan, Z. Lu, L. Wu, P. Lyu, H. Shen, Z. Lin, S. Hu, J. Yang, H. Wen, G. Yu, H. Liu, B. Wang, C. Ma, H. Hu, C. Zhang, and Y. Zhou,

“HunyuanOCR-1.5: Making lightweight OCR VLMs faster and better,” arXiv preprint arXiv:2607.04884, 2026.

[58] S. Lu, Y. Li, Y. Xia, Y. Chen, A.-Y. Ji, J.-P. Jiang, Q.-G. Chen, J. Zhao, E. Lin, H. Li, C. Qin, Z. Xu, and W. Luo, “OvisOCR2 technical report,” arXiv preprint arXiv:2607.13639, 2026.

[59] D. Dong, M. Zheng, D. Xu, C. Luo, B. Zhuang, Y. Li, R. He, H. Wang, W. Zhang, W. Wang, Y. Wang, X. Xiong, A. Zheng, X. Zuo, Z. Ou, J. Gu, Q. Guo, J. Wu, D. Yin, and D. Shen, “Qianfan-OCR: A unified end-to-end model for document intelligence,” arXiv preprint arXiv:2603.13398, 2026.

[60] H. Wei, Y. Sun, and Y. Li, “DeepSeek-OCR 2: Visual causal flow,” arXiv preprint arXiv:2601.20552, 2026.

[61] P. Cai, Z. Zou, S. Liu, Y. Wang, J. Tang, K. Yang, M. Tong, M. Jiang, Z. He, and H. Sun, “TeleOCR: Navigating document parsing across digital and camera-captured documents,” arXiv preprint arXiv:2608.12898, 2026.

[62] Z. Zhang, H. Liu, S. Liang, Y. Zhang, Y. Xiang, J. Liu, T. Sun, M. Lin, Y. Zhang, C. Zhou, T. Gao, C. Cui, Y. Liu, D. Yu, and Y. Ma, “PaddleOCR-VL-1.6: Expanding the frontier of document parsing with under-optimized region refinement and progressive post-training,” arXiv preprint arXiv:2606.03264, 2026.

[63] B. Wang, T. He, L. Ouyang, F. Wu, Z. Zhao, T. Chu, Y. Qu, Z. Jin, W. Zeng, Z. Miao, B. Xu, J. Niu, M. Cai, J. Qiu, Q. Zhang, D. Ma, Y. Sun, H. Dong, W. Zhang, J. Xiao, J. Shi, P. Liao, X. Zhao, H. Zhong, L. Wei, J. Yu, J. Yang, W. Li, S. Wang, Q. Wu, X. Zhou, W. Li, Z. Li, Z. Tu, J. Wu, L. Wu, C. Xu, K. Chen, W. Zhang, Y. Qiao, B. Zhou, D. Lin, and C. He, “MinerU2.5-Pro: Pushing the limits of data-centric document parsing at scale,” arXiv preprint arXiv:2604.04771, 2026.

[64] S. Duan, Y. Xue, W. Wang, Z. Su, H. Liu, S. Yang, G. Gan, G. Wang, Z. Wang, S. Yan, D. Jin, Y. Zhang, G. Wen, Y. Wang, Y. Zhang, X. Zhang, W. Hong, Y. Cen, D. Yin, B. Chen, W. Yu, X. Gu, and J. Tang, “GLM-OCR technical report,” 2026. [Online]. Available: https://arxiv.org/abs/2603.10910

[65] Google DeepMind, “Gemini 3.1 Pro,” Model documentation, 2026. [Online]. Available: https://deepmind.google/models/ gemini/pro/

[66] Qwen Team, “Qwen3.8-27B,” Hugging Face model card, Aug. 2026. [Online]. Available: https://huggingface.co/Qwen/ Qwen3.8-27B

[67] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, Z. Wang, Z. Chen et al., “InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[68] OpenAI, “GPT-6 and intelligent UI for everyone,” Official model release, Oct. 2026. [Online]. Available: https://openai.com/ index/gpt-6-for-everyone/

[69] Mistral AI, “Introducing Mistral OCR 4,” Official model release, Jun. 2026. [Online]. Available: https://mistral.ai/news/ocr-4/

[70] Google Cloud, “Document AI documentation,” Product documentation, 2025. [Online]. Available: https://cloud.google. com/document-ai/docs

[71] Microsoft Azure, “Azure AI Document Intelligence documentation,” Product documentation, 2025. [Online]. Available: https://learn.microsoft.com/en-us/azure/ ai-services/document-intelligence/

[72] J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, “What is wrong with scene text recognition model comparisons? Dataset and model analysis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 4715– 4723.

[73] Q. Jiang, J. Wang, D. Peng, C. Liu, and L. Jin, “Revisiting scene text recognition: A data perspective,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[74] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

[75] Y. Liu, Z. Li, Z. Zhang, S. Zhang, Q. Liu, J. Song, Z. Guo, X. Wang, H. Zheng, Y. Liu, D. Luo, Z. Ma, J. Zhang, and X. Bai, “MonkeyOCRv2: A visual-text foundation model for document AI,” arXiv preprint arXiv:2607.11562, 2026.

[76] J. Poznanski, L. Soldaini, and K. Lo, “olmOCR 2: Unit test rewards for document OCR,” arXiv preprint arXiv:2510.19817, 2025.

[77] C. Cui, T. Sun, S. Liang, T. Gao, Z. Zhang, J. Liu, X. Wang, C. Zhou, H. Liu, M. Lin, Y. Zhang, Y. Zhang, Y. Liu, D. Yu, and Y. Ma, “PaddleOCR-VL-1.5: Towards a multi-task 0.9B VLM for robust in-the-wild document parsing,” arXiv preprint arXiv:2601.21957, 2026.

[78] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[79] Y. Zhong, L. Chen, Z. Zeng, X. Zhao, D. Jiang, L. Zheng, J. Huang, H. Qiu, P. Shi, S. Yang, and L. Ma, “Reading or reasoning? Format decoupled reinforcement learning for document OCR,” arXiv preprint arXiv:2601.08834, 2026.

[80] J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He, F. Wu, Q. Zhang et al., “MinerU2.5: A decoupled visionlanguage model for efficient high-resolution document parsing,” arXiv preprint arXiv:2509.22186, 2025.

[81] Z. Li, Y. Liu, Q. Liu, Z. Ma, Z. Zhang, S. Zhang, B. Yang, Z. Guo, J. Zhang, X. Wang, and X. Bai, “MonkeyOCR: Document parsing with a structure-recognition-relation triplet paradigm,” arXiv preprint arXiv:2506.05218, 2025.

[82] H. Feng, W. Shi, K. Zhang, X. Fei, L. Liao, D. Yang, Y. Du, X. Wu, J. Tang, Y. Liu, H. Chen, and C. Huang, “Dolphin-v2: Universal document parsing via scalable anchor prompting,” arXiv preprint arXiv:2602.05384, 2026.

[83] H. Cao, K. Yin, Y. Wu, B. Liu, Z. Cai, X. Li, H. Chen, X. Li, Y. Liu, D. Jiang, X. Sun, Y. Wu, Q. Li, A. Guo, Y. Liao, Y. Qu, H. Lin, C. He, and S. Liu, “Youtu-Parsing: Perception, structuring and recognition via high-parallelism decoding,” arXiv preprint arXiv:2601.20430, 2026.

[84] L. Xiang, C. Zhai, S. Wei, J. Wu, Q. Xie, X. Tan, K. Chen, and W. He, “P-MTP: Efficient document parsing via multitoken prediction with progressive depth scaling,” arXiv preprint arXiv:2606.24447, 2026.

[85] Z. Li, Z. Lin, Q. Liu, Z. Zhang, S. Zhang, Z. Guo, J. Song, J. Zhang, X. Bai, and Y. Liu, “MDPBench: A benchmark for multilingual document parsing in real-world scenarios,” arXiv preprint arXiv:2603.28130, 2026.

[86] B. Zhou, H. Xing, Y. Chen, J. Xu, Q. Zheng, F. Gao, Z. Yang, S. Bai, M. Yan, J. Ye, and H. Xie, “MPDocBench-Parse: Benchmarking practical multi-page document parsing,” arXiv preprint arXiv:2605.22100, 2026.

[87] X. Xie, H. Yan, L. Yin, Y. Liu, J. Ding, M. Liao, Y. Liu, W. Chen, and X. Bai, “PDF-WuKong: A large multimodal model for efficient long PDF reading with end-to-end sparse sampling,” International Journal of Computer Vision, vol. 134, 2026, art. no. 324. [Online]. Available: https://doi.org/10.1007/s11263-026-02860-w

[88] K. Wang, B. Babenko, and S. Belongie, “End-to-end scene text recognition,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2011, pp. 1457–1464.

[89] D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R. Chandrasekhar, S. Lu, F. Shafait, S. Uchida, and E. Valveny, “ICDAR 2015 competition on robust reading,” in Proceedings of the 13th International Conference on Document Analysis and Recognition (ICDAR), 2015, pp. 1156–1160.

[90] C. K. Chng and C. S. Chan, “Total-Text: A comprehensive dataset for scene text detection and recognition,” in Proceedings of the International Conference on Document Analysis and Recognition (IC-DAR), 2017.

[91] Y. Liu, L. Jin, S. Zhang, C. Luo, and S. Zhang, “Curved scene text detection via transverse and longitudinal sequence connection,” Pattern Recognition, vol. 90, pp. 337–345, 2019.

[92] X. Zhong, J. Tang, and A. Jimeno-Yepes, “PubLayNet: Largest dataset ever for document layout analysis,” Proceedings of the International Conference on Document Analysis and Recognition (IC-DAR), pp. 1015–1022, 2019.

[93] Z. Gu, G. Liang, B. Wang, Z. Zhao, Q. Zhang, W. Li, C. Xu, B. Zhang, B. Shi, J. Wu, W. Zhang, and C. He, “UniMERNet: A universal network for real-world mathematical expression recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 34 106–34 115.

[94] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, “ChartQA: A benchmark for question answering about charts with visual and logical reasoning,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022.

[95] L. Fu, Z. Kuang, J. Song, M. Huang, B. Yang, Y. Li, L. Zhu, Q. Luo, X. Wang, H. Lu, Z. Li, G. Tang, B. Shan, C. Lin, Q. Liu, B. Wu, H. Feng, H. Liu, C. Huang, J. Tang, W. Chen, L. Jin, Y. Liu, and X. Bai, “OCRBench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning,” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2025.

[96] L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, J. Shi, F. Wu, P. Chu, M. Liu, Z. Li, C. Xu, B. Zhang, B. Shi, Z. Tu, and C. He, “OmniDocBench: Benchmarking diverse PDF document parsing with comprehensive annotations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[97] C. Zhou, Z. Gao, X. Wang, T. Gao, C. Cui, J. Tang, and Y. Liu, “Real5-OmniDocBench: A full-scale physical reconstruction benchmark for robust document parsing in the wild,” in Proceedings of the European Conference on Computer Vision (ECCV), 2026. [Online]. Available: https://arxiv.org/abs/2603.04205

[98] Z. Li, Z. Ma, J. Chen, J. Zhang, Z. Su, Y. Zhang, Z. Yu, R. Liu, X. Lv, B. Li, J. Gao, Z. Zhang et al., “How far is document parsing from solved? PureDocBench: A source-traceable benchmark across clean, degraded, and real-world settings,” arXiv preprint arXiv:2605.07492, 2026.

[99] C. Cui, T. Gao, X. Wang, C. Zhou, H. Liu, T. Sun, Y. Zhang, Z. Zhang, J. Liu, M. Lin, Y. Zhang, S. Liang, Y. Xiang, and Y. Liu, “RT-DocLayout: Real-time end-to-end document layout analysis with reading order in the wild,” arXiv preprint arXiv:2606.23344, 2026.

[100] T. Kumar and H. S. Bhatt, “Evaluating table structure recognition: A new perspective,” arXiv preprint arXiv:2208.00385, 2022.

[101] B. Wang, F. Wu, L. Ouyang, Z. Gu, R. Zhang, R. Xia, B. Shi, B. Zhang, and C. He, “Image over text: Transforming formula recognition evaluation with character detection matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 19 681–19 690.

[102] J. Lee, S. Park, J. Baek, S. J. Oh, S. Kim, and H. Lee, “On recognizing texts of arbitrary shapes with 2D self-attention,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020. [Online]. Available: https://arxiv.org/abs/1910.04396

[103] Y. Li, Y. Qian, Y. Yu, X. Qin, C. Zhang, Y. Liu, K. Yao, J. Han, J. Liu, and E. Ding, “StrucTexT: Structured text understanding with multi-modal transformers,” in Proceedings of the ACM International Conference on Multimedia (MM), 2021, pp. 1912–1920.

[104] J. Wang, L. Jin, and K. Ding, “LiLT: A simple yet effective language-independent layout transformer for structured document understanding,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2022, pp. 7747– 7757.

[105] D. Bautista and R. Atienza, “Scene text recognition with permuted autoregressive sequence models,” in Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[106] L. Cui, Y. Xu, T. Lv, and F. Wei, “Document AI: Benchmarks, models and applications,” arXiv preprint arXiv:2111.08609, 2021.

[107] Q. Zhang, B. Wang, V. S.-J. Huang, J. Zhang, Z. Wang, H. Liang, C. He, and W. Zhang, “Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction,” arXiv preprint arXiv:2410.21169, 2024.

[108] X.-H. Li, Y. Zhou, L.-R. Peng, S.-X. Chen, Z.-H. Lian, L.-C. Gao, X.-C. Yin, and C.-L. Liu, “The technological evolution from text recognition to intelligent document parsing—a survey of advances from the “document image micro-salon” seminar series,” Journal of Image and Graphics, 2026, online first, May 12, 2026.

[109] Y. Zhang, Q. Li, H. Shen, G. Zeng, Y. Zhou, C. Ma, Y. Zhang, and W. Wang, “Text-centric image analysis techniques: A critical review,” Journal of Image and Graphics, vol. 28, no. 8, pp. 2253– 2275, 2023.

[110] Y. Shu, W. Zeng, Z. Li, F. Zhao, and Y. Zhou, “Visual text meets low-level vision: A comprehensive survey on visual text processing,” arXiv preprint arXiv:2402.03082, 2024.

[111] ABBYY, “ABBYY FineReader PDF: User guide and product documentation,” Product documentation, 2024. [Online]. Available: https://help.abbyy.com/en-us/finereader/16/user guide/overview/

[112] A. Bissacco, M. Cummins, Y. Netzer, and H. Neven, “PhotoOCR: Reading text in uncontrolled conditions,” in Proceedings of the

IEEE International Conference on Computer Vision (ICCV), 2013, pp. 785–792.

[113] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Advances in Neural Information Processing Systems, vol. 25, 2012.

[114] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770– 778.

[115] Z. Tian, W. Huang, T. He, P. He, and Y. Qiao, “Detecting text in natural image with connectionist text proposal network,” in Proceedings of the European Conference on Computer Vision (ECCV), 2016, pp. 56–72.

[116] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao, “Shape robust text detection with progressive scale expansion network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9328–9337.

[117] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, “Character region awareness for text detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9357–9366.

[118] B. Shi, X. Wang, P. Lyu, C. Yao, and X. Bai, “Robust scene text recognition with automatic rectification,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4168–4176, 2016.

[119] S. Fang, H. Xie, Y. Wang, Z. Mao, and Y. Zhang, “Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 7098–7107.

[120] Y. Du, Z. Chen, C. Jia, X. Yin, T. Zheng, C. Li, Y. Du, and Y.-G. Jiang, “SVTR: Scene text recognition with a single visual model,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI), 2022, pp. 884–890.

[121] Y. Liu, H. Chen, C. Shen, T. He, L. Jin, and L. Wang, “ABC-Net: Real-time scene text spotting with adaptive Bezier-curve network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 9809–9818.

[122] T. Kil, S. Kim, S. Seo, Y. Kim, and D. Kim, “Towards unified scene text spotting based on sequence generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15 223–15 232. [Online]. Available: https://arxiv.org/abs/2304.03435

[123] Y. Du, Z. Chen, C. Jia, X. Yin, C. Li, Y. Du, and Y.-G. Jiang, “Context perception parallel decoder for scene text recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 6, pp. 4668–4683, 2025.

[124] X. Yang, Z. Qiao, and Y. Zhou, “IPAD: Iterative, parallel, and diffusion-based network for scene text recognition,” International Journal of Computer Vision, vol. 133, no. 8, pp. 5589–5609, 2025.

[125] Y. Du, Z. Chen, C. Jia, X. Gao, and Y.-G. Jiang, “Out of length text recognition with sub-string matching,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 2798–2806.

[126] Y. Zhang, C. Liu, J. Wei, X. Yang, Y. Zhou, C. Ma, and X. Ji, “Linguistics-aware masked image modeling for self-supervised scene text recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 9318– 9328.

[127] X. Ye, Y. Du, J. Zhang, C. Li, J. Lyu, and Z. Chen, “What’s wrong with synthetic data for scene text recognition? A strong synthetic engine with diverse simulations and self-evolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2026, pp. 16 645–16 654. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/ html/Ye Whats Wrong with Synthetic Data for Scene Text Recognition A CVPR 2026 paper.html

[128] X. Ye, Y. Du, J. Zhang, H. Zhang, C. Sun, C. Li, J. Lyu, and Z. Chen, “Advancing WordArt-oriented scene text recognition: Datasets and methods,” in Proceedings ofthe European Conference on Computer Vision (ECCV), 2026. [Online]. Available: https://arxiv.org/abs/2606.24484

[129] Y. Zhu, Z. Xie, L. Jin, X. Chen, Y. Huang, and M. Zhang, “SCUT-EPT: New dataset and benchmark for offline Chinese text recognition in examination paper,” IEEE Access, vol. 7, pp. 370–382, 2019.

[130] JaidedAI, “EasyOCR: Ready-to-use OCR with 80+ supported languages,” GitHub Repository, 2020. [Online]. Available: https://github.com/JaidedAI/EasyOCR

[131] Mindee, “docTR: Document text recognition,” GitHub Repository, 2021. [Online]. Available: https://github.com/mindee/doctr

[132] Z. Kuang, H. Sun, Z. Li, X. Yue, T. H. Lin, J. Chen, H. Wei, Y. Zhu, T. Gao, W. Zhang, K. Chen, W. Zhang, and D. Lin, “MMOCR: A comprehensive toolbox for text detection, recognition and understanding,” arXiv preprint arXiv:2108.06543, 2021.

[133] Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang, W. Che, M. Zhang, and L. Zhou, “LayoutLMv2: Multimodal pre-training for visually-rich document understanding,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, 2021.

[134] Y. Huang, T. Lv, L. Cui, Y. Lu, and F. Wei, “LayoutLMv3: Pretraining for document AI with unified text and image masking,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022.

[135] Y. Yin, H. Liu, YY, Q. Xie, C. Liu, S. Yang, S. Wang, Z. Liu, H. Zou, J. Chen, S. Wei, J. Wu, M. Huang, Z. Wu, G. Wang, T. Du, and L. Jia, “Unlimited OCR works,” arXiv preprint arXiv:2606.23050, 2026.

[136] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” Proceedings of NAACL-HLT, 2019.

[137] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021.

[138] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Ro-\` driguez, A. Joulin, E. Grave, and G. Lample, “LLaMA: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[139] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, 2022.

[140] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[141] Z. Li, B. Yang, Q. Liu, Z. Ma, S. Zhang, J. Yang, Y. Sun, Y. Liu, and X. Bai, “Monkey: Image resolution and text label are important things for large multi-modal models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [Online]. Available: https://arxiv.org/abs/2311.06607

[142] Y. Liu, B. Yang, Q. Liu, Z. Li, Z. Ma, S. Zhang, and X. Bai, “TextMonkey: An OCR-free large multimodal model for understanding document,” arXiv preprint arXiv:2403.04473, 2024.

[143] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023.

[144] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond,” arXiv preprint arXiv:2308.12966, 2023.

[145] C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, C. Zhou, H. Liu, Y. Zhang, W. Lv, K. Huang, Y. Zhang, J. Zhang, J. Zhang, Y. Liu, D. Yu, and Y. Ma, “PaddleOCR 3.0 technical report,” arXiv preprint arXiv:2507.05595, 2025.

[146] Z. Zhao, H. Kang, B. Wang, and C. He, “DocLayout-YOLO: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception,” arXiv preprint arXiv:2410.12628, 2024.

[147] M. Liao, Z. Zou, Z. Wan, C. Yao, and X. Bai, “Real-time scene text detection with differentiable binarization and adaptive scale fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 919–931, 2023.

[148] H. Zheng, Y. Li, K. Zhang et al., “Multimodal OCR: Parse anything from documents,” arXiv preprint arXiv:2603.13032, 2026. [Online]. Available: https://arxiv.org/abs/2603.13032

[149] H. Wu, H. Lou, X. Li, Z. Zhong, Z. Sun, P. Chen, X. Zhou, K. Zuo, Y. Chen, X. Tang, Y. Hu, B. Zhou, J. Wu, Y. Wu, W. Yu, Y. Liu,

Y. Huang, M. Xu, G. Liu, Y. Ma, Z. Sun, and C. Qiao, “FireRed-OCR technical report,” arXiv preprint arXiv:2603.01840, 2026.

[150] IBM Granite Team, “Granite-Docling-258M,” Hugging Face model card, Sep. 2025. [Online]. Available: https://huggingface. co/ibm-granite/granite-docling-258M

[151] Amazon Web Services, “Amazon Textract documentation,” Product documentation, 2025. [Online]. Available: https: //docs.aws.amazon.com/textract/

[152] Mathpix, “Mathpix documentation,” Product documentation, 2026. [Online]. Available: https://docs.mathpix.com/

[153] H. Feng, S. Wei, X. Fei, W. Shi, Y. Han, L. Liao, J. Lu, B. Wu, Q. Liu, C. Lin, J. Tang, H. Liu, and C. Huang, “Dolphin: Document image parsing via heterogeneous anchor prompting,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 21 919–21 936. [Online]. Available: https://aclanthology.org/2025.findings-acl.1130/

[154] Y. Li, G. Yang, H. Liu, B. Wang, and C. Zhang, “dots.ocr: Multilingual document layout parsing in a single visionlanguage model,” 2025. [Online]. Available: https://arxiv.org/ abs/2512.02498

[155] OpenDataLab Contributors, “OmniDocBench: Official evaluation repository,” GitHub repository, 2026, version v1.6-full leaderboard; accessed: 2026-10-18. [Online]. Available: https://github.com/opendatalab/OmniDocBench

[156] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman, “Synthetic data and artificial neural networks for natural scene text recognition,” arXiv preprint arXiv:1406.2227, 2014.

[157] H. Wei, Y. Sun, and Y. Li, “DeepSeek-OCR: Contexts optical compression,” arXiv preprint arXiv:2510.18234, 2025.

[158] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[159] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, Z. Gao, E. Cui, X. Wang, Y. Cao, Y. Liu, X. Wei, H. Zhang, H. Wang, W. Xu, H. Li, J. Wang, N. Deng, S. Li, Y. He, T. Jiang, J. Luo, Y. Wang, C. He, B. Shi, X. Zhang, W. Shao, J. He, Y. Xiong, W. Qu, P. Sun, P. Jiao, H. Lv, L. Wu, K. Zhang, H. Deng, J. Ge, K. Chen, L. Wang, M. Dou, L. Lu, X. Zhu, T. Lu, D. Lin, Y. Qiao, J. Dai, and W. Wang, “InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[160] S. Bai et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[161] W. Yu, Z. Yang, Y. Liu, and X. Bai, “DocThinker: Explainable multimodal large language models with rule-based reinforcement learning for document understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 837–847.

[162] M. Pechwitz, S. S. Maddouri, V. Margner, N. Ellouze, and¨ H. Amiri, “IFN/ENIT-database of handwritten Arabic words,” in Proceedings of the Colloque International Francophone sur l’Ecrit et<sup>´</sup> le Document (CIFED), 2002, pp. 127–136.

[163] N. Nayef, Y. Patel, M. Busta, P. N. Chowdhury, D. Karatzas, W. Khlif, J. Matas, U. Pal, J.-C. Burie, C.-L. Liu, and J.-M. Ogier, “ICDAR2019 robust reading challenge on multi-lingual scene text detection and recognition — RRC-MLT-2019,” in Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 1582–1587.

[164] Y. Xu, T. Lv, L. Cui, G. Wang, Y. Lu, D. Florencio, C. Zhang, and F. Wei, “XFUND: A benchmark dataset for multilingual visually rich form understanding,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 3214–3224.

[165] J. Tang, Q. Liu, Y. Ye, J. Lu, S. Wei, A.-L. Wang, C. Lin, H. Feng, Z. Zhao, Y. Wang, Y. Liu, H. Liu, X. Bai, and C. Huang, “MTVQA: Benchmarking multilingual text-centric visual question answering,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 7748–7763. [Online]. Available: https://aclanthology.org/2025.findings-acl. 404/

[166] Z. Yang, J. Tang, Z. Li, P. Wang, J. Wan, H. Zhong, X. Liu, M. Yang, P. Wang, S. Bai, L. Jin, and J. Lin, “CC-OCR: A comprehensive and challenging OCR benchmark for evaluating large multimodal models in literacy,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[167] J. Van Landeghem, R. Tito, Ł. Borchmann, M. Pietruszka, P. Joziak, R. Powalski, D. Jurkiewicz, M. Coustaty, B. Anckaert, E. Valveny, M. B. Blaschko, S. Moens, and T. Stanislawek, “Document understanding dataset and evaluation (DUDE),” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19 528–19 540.

[168] Y. Ma, Y. Zang, L. Chen, M. Chen, Y. Jiao, X. Li, X. Lu, Z. Liu, Y. Ma, X. Dong, P. Zhang, L. Pan, Y.-G. Jiang, J. Wang, Y. Cao, and A. Sun, “MMLongBench-Doc: Benchmarking long-context document understanding with visualizations,” arXiv preprint arXiv:2407.01523, 2024.

[169] H. Yan, Y. Liu, X. Liu, Y. Zhang, M. Liao, J. Wu, W. Chen, and X. Bai, “DocSeeker: Structured visual reasoning with evidence grounding for long document understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 41 140–41 149.

[170] U.-V. Marti and H. Bunke, “The IAM-database: An English sentence database for offline handwriting recognition,” International Journal on Document Analysis and Recognition, vol. 5, no. 1, pp. 39–46, 2002.

[171] D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Bigorda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, and L. P. de las Heras, “ICDAR 2013 robust reading competition,” in Proceedings of the 12th International Conference on Document Analysis and Recognition (ICDAR), 2013, pp. 1484–1493.

[172] A. Mishra, K. Alahari, and C. V. Jawahar, “Scene text recognition using higher order language priors,” in Proceedings of the British Machine Vision Conference (BMVC), 2012.

[173] T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan, “Recognizing text with perspective distortion in natural scenes,” Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 569–576, 2013.

[174] A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan, “A robust arbitrary text detection system for natural scene images,” Expert Systems with Applications, vol. 41, no. 18, pp. 8027–8048, 2014.

[175] W. Wu, Y. Cai, D. Zhang, S. Wang, Z. Li, J. Li, Y. Tang, and H. Zhou, “A bilingual, OpenWorld video text dataset and endto-end video text spotter with transformer,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2021.

[176] S. Long, S. Qin, D. Panteleev, A. Bissacco, Y. Fujii, and M. Raptis, “Towards end-to-end unified scene text detection and layout analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[177] X. Zheng, D. Burdick, L. Popa, X. Zhong, and N. X. R. Wang, “Global table extractor (GTE): A framework for joint table identification and cell structure recognition using visual context,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 697–706.

[178] H. Mouchere \` et al., “ICFHR 2016 CROHME: Competition on recognition of online handwritten mathematical expressions,” in Proceedings of the International Conference on Frontiers in Handwriting Recognition (ICFHR), 2016, pp. 607–612.

[179] B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, and P. W. J. Staar, “DocLayNet: A large human-annotated dataset for documentlayout segmentation,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2022, pp. 3743–3751.

[180] G. Jaume, H. K. Ekenel, and J.-P. Thiran, “FUNSD: A dataset for form understanding in noisy scanned documents,” in Proceedings ofthe International Conference on Document Analysis and Recognition Workshops (ICDARW), 2019, pp. 1–6.

[181] S. Park, S. Shin, B. Lee, J. Lee, J. Surh, M. Seo, and H. Lee, “CORD: A consolidated receipt dataset for post-OCR parsing,” in Workshop on Document Intelligence at NeurIPS 2019, 2019.

[182] G. Cohen, S. Afshar, J. Tapson, and A. van Schaik, “EMNIST: Extending MNIST to handwritten letters,” in Proceedings of the International Joint Conference on Neural Networks (IJCNN), 2017, pp. 2921–2926.

[183] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” in NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

[184] E. Grosicki and H. El Abed, “ICDAR 2011—French handwriting recognition competition,” in Proceedings of the International Con-

ference on Document Analysis and Recognition (ICDAR), 2011, pp. 1459–1463.

[185] A. Singh, G. Pang, M. Toh, J. Huang, W. Galuba, and T. Hassner, “TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8802–8812.

[186] R. Zhang, M. Yang, X. Bai, B. Shi, D. Karatzas, S. Lu, C. V. Jawahar, Y. Zhou, Q. Jiang, Q. Song, N. Li, K. Zhou, L. Wang, D. Wang, and M. Liao, “ICDAR 2019 robust reading challenge on reading Chinese text on signboard,” in Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 1577–1581.

[187] Y. Sun et al., “ICDAR 2019 competition on large-scale street view text with partial labeling – RRC-LSVT,” in Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019.

[188] C. K. Chng et al., “ICDAR 2019 robust reading challenge on arbitrary-shaped text – RRC-ArT,” in Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 1571–1576.

[189] N. Nayef, F. Yin, I. Bizid, H. Choi, Y. Feng, D. Karatzas, Z. Luo, U. Pal, C. Rigaud, J. Chazalon, W. Khlif, M. M. Luqman, J.-C. Burie, C.-L. Liu, and J.-M. Ogier, “ICDAR2017 robust reading challenge on multi-lingual scene text detection and script identification — RRC-MLT,” in Proceedings ofthe 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), 2017, pp. 1454–1459.

[190] A. Veit, T. Matera, L. Neumann, J. Matas, and S. Belongie, “COCO-Text: Dataset and benchmark for text detection and recognition in natural images,” arXiv preprint arXiv:1601.07140, 2016.

[191] C. Yao, X. Bai, W. Liu, Y. Ma, and Z. Tu, “Detecting texts of arbitrary orientations in natural images,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1083–1090, 2012.

[192] K. Wang and S. Belongie, “Word spotting in the wild,” in Computer Vision – ECCV 2010, ser. Lecture Notes in Computer Science, vol. 6311. Springer, 2010, pp. 591–604.

[193] B. Smock, R. Pesala, and R. Abraham, “PubTables-1M: Towards comprehensive table extraction from unstructured documents,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 4634–4642.

[194] R. Long et al., “Parsing table structures in the wild,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[195] Z. Chi et al., “Complicated table structure recognition,” arXiv preprint arXiv:1908.04729, 2019.

[196] M. Li, L. Cui, S. Huang, F. Wei, M. Zhou, and Z. Li, “TableBank: A benchmark dataset for table detection and recognition,” in Proceedings of the International Conference on Language Resources and Evaluation (LREC), 2020, pp. 1918–1925.

[197] Y. Yuan et al., “Syntax-aware network for handwritten mathematical expression recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 4553–4562.

[198] M. Li, Y. Xu, L. Cui, S. Huang, F. Wei, Z. Li, and M. Zhou, “DocBank: A benchmark dataset for document layout analysis,” Proceedings of the 28th International Conference on Computational Linguistics, pp. 949–960, 2020.

[199] M. Huang, Y. Shi, D. Peng, S. Lai, Z. Xie, and L. Jin, “OCR-Reasoning benchmark: Unveiling the true capabilities of MLLMs in complex text-rich image reasoning,” in International Conference on Learning Representations (ICLR), 2026.

[200] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards VQA models that can read,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 8317–8326.

[201] C. Peng, Z. Xu, Y. Xiong, Z. Chen, J. Ji, Q. Liu, Z. Liu, Z. Qin, B. Zhao, J. Wan, J. Tang, Z. Yang, S. Bai, D. Liu, and M. Sun, “CC-OCR V2: Fine-grained attribution of LMM failures in real-world visual document understanding,” arXiv preprint arXiv:2605.03903, 2026.

[202] B. Zhang, S. G. Acosta, P. Carlson, S. Bron, P.-L. Doulcet, D. B. Ospina, and S. Suo, “ParseBench: A document parsing benchmark for AI agents,” arXiv preprint arXiv:2604.08538, 2026.

[203] S. Long, J. Ruan, W. Zhang, X. He, W. Wu, and C. Yao, “TextSnake: A flexible representation for detecting text of arbitrary shapes,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 20–36.

[204] W. Wang, E. Xie, X. Song, Y. Zang, W. Wang, T. Lu, G. Yu, and C. Shen, “Efficient and accurate arbitrary-shaped text detection with pixel aggregation network,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 8440– 8449.

[205] J. Ye, Z. Chen, J. Liu, and B. Du, “TextFuseNet: Scene text detection with richer fused features,” in Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2020, pp. 516–522.

[206] Y. Zhu, J. Chen, L. Liang, Z. Kuang, L. Jin, and W. Zhang, “Fourier contour embedding for arbitrary-shaped text detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 3123–3131.

[207] X. Zhang, Y. Su, S. Tripathi, and Z. Tu, “Text spotting transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9519–9528.

[208] S. Kim, S. Lee, J. Kim, and N. Kwak, “TPD-STR: Text polygon detection with split transformers,” in Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2025, pp. 8940–8949.

[209] M. Liao, P. Lyu, M. He, C. Yao, W. Wu, and X. Bai, “Mask TextSpotter: An end-to-end trainable neural network for spotting text with arbitrary shapes,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 2, pp. 532–548, 2021. [Online]. Available: https://arxiv.org/abs/1908.08207

[210] W. Feng, W. He, F. Yin, X.-Y. Zhang, and C.-L. Liu, “TextDragon: An end-to-end framework for arbitrary shaped text spotting,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9076–9085.

[211] Y. Liu, C. Shen, L. Jin, T. He, P. Chen, C. Liu, and H. Chen, “ABCNet v2: Adaptive Bezier-curve network for real-time endto-end text spotting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8048–8064, 2022.

[212] M. Huang, J. Zhang, D. Peng, H. Lu, C. Huang, Y. Liu, X. Bai, and L. Jin, “ESTextSpotter: Towards better scene text spotting with explicit synergy in transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/html/ Huang ESTextSpotter Towards Better Scene Text Spotting with Explicit Synergy in ICCV 2023 paper.html

[213] M. Huang, H. Li, Y. Liu, X. Bai, and L. Jin, “Bridging the gap between end-to-end and two-step text spotting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 15 608–15 618. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/ html/Huang Bridging the Gap Between End-to-End and Two-Step Text Spotting CVPR 2024 paper.html

[214] L. Jang, Y. Lin, Y.-Y. Chiang, and J. Weinman, “TiCLS: Tightly coupled language text spotter,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026. [Online]. Available: https://openaccess.thecvf. com/content/WACV2026/html/Jang TiCLS Tightly Coupled Language Text Spotter WACV 2026 paper.html

[215] OpenDataLab, “MinerU2.5-Pro-2605-1.2B model card,” Hugging Face model card, 2026, checkpoint 2605; OmniDocBench v1.6-full; accessed: 2026-09-13. [Online]. Available: https: //huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B

[216] PaddlePaddle Contributors, “Real5-OmniDocBench: Official dataset and leaderboard,” Hugging Face dataset repository, 2026, accessed: 2026-10-18. [Online]. Available: https://huggingface. co/datasets/PaddlePaddle/Real5-OmniDocBench

[217] X. Chen, A. Du, F. Feng, P. Fu, J. Luan, L. Xu, S. Zhang, H. Li, H. Qu, and C. Tan, “Xiaomi-OCR-0 technical report,” arXiv preprint arXiv:2609.36136, 2026.

[218] ByteDance Seed Team, “Seed2.1 officially released: Advancing AI productivity,” Official model release, 2026. [Online]. Available: https://seed.bytedance.com/en/blog/ seed2-1-officially-released-advancing-ai-productivity

[219] J. Ye, X. Qi, Y. He, Y. Chen, D. Gu, P. Gao, and R. Xiao, “PingAn-VCGroup’s solution for ICDAR 2021 competition on scientific literature parsing Task B: Table recognition to HTML,” arXiv preprint arXiv:2105.01848, 2021.

[220] Y. Chen, J. Zhang, K. Peng, J. Zheng, R. Liu, P. Torr, and R. Stiefelhagen, “RoDLA: Benchmarking the robustness of document layout analysis models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 15 556–15 566. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/ Chen RoDLA Benchmarking the Robustness of Document Layout Analysis Models CVPR 2024 paper.html

[221] C. Da, C. Luo, Q. Zheng, and C. Yao, “Vision grid transformer for document layout analysis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[222] Qwen Team, “Qwen3.5: Towards native multimodal agents,” Technical report, 2026. [Online]. Available: https://qwen.ai/blog? id=29c9028b4203455095b262fd99b46f53

[223] X. Chen, S. Li, X. Zhu et al., “Logics-Parsing technical report,” arXiv preprint arXiv:2509.19760, 2025.