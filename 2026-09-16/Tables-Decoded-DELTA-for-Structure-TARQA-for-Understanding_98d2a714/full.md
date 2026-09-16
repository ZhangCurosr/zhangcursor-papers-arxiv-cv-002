# Tables Decoded: DELTA for Structure, TARQA for Understanding

Jahanvi Rajput<sup>∗1</sup>, Dhruv Kudale<sup>\*1†</sup>, Saikiran Kasturi<sup>1</sup>, Utkarsh Verma<sup>1</sup>, Ganesh Ramakrishnan<sup>1,2</sup>

{23d0378, 22m2116, 24m2157, 24m2153, ganramkr}@iitb.ac.in

<sup>1</sup>Indian Institute of Technology Bombay, <sup>2</sup>BharatGen

## Abstract

Table understanding is a core task in document intelligence, encompassing two key subtasks: table reconstruction and table visual question answering (TabVQA). While recent approaches predominantly rely on visionlanguage models (VLMs) operating on table images, we propose a more scalable and effective alternative based on structured textual representations. These representations are easier to process, align more naturally with LLMs, and eliminate the need for language-specific visual encoders, making them particularly suitable for multilingual documents. We present DELTA, which separates physical structure recognition, logical structure recognition, and OCR to extract both layout and content accurately. DELTA outputs tables in Optimised Table Structure Language (OTSL), a compact and unified format that encodes cell arrangements and textual content. On table structure recognition (TSR), DELTA achieves TEDS-Structure scores comparable with state-of-the-art methods across FinTabNet, PubTabNet, and PubTables-1M. We further establish its robustness on non-English tables through our curated Hindi benchmark, TORQUE. Building on this, we introduce TARQA, an LLM fine-tuned on OTSL sequences. Our approach yields gains of 9.3 p.p. on WTQ (TabQA) and 9.2 p.p. on FinTabNetQA (TabVQA), respectively. On TORQUE, our method ranks second among all VLMs and DELTA + LLM variants. We release our code, models, and benchmark at: https://github.com/ Tihiitborg/Tables-Decoded.

## 1. Introduction

Tables are a vital medium for structured information in domains like science, finance, and administration. Their visual layouts encode relational data, supporting tasks such as retrieval, parsing, summarization, and Table Visual Question Answering (TabVQA). Real-world tables, however, present challenges: diverse layouts, formatting variations, multiple languages, and complications like merged cells, spanning, or noisy scans. Broadly, as seen in Figure (1), existing approaches to table understanding follow one of the many distinct paradigms. The first relies on Vision Language Models (VLMs), which directly consume table images and jointly model visual and textual features using architectures pretrained on large multimodal datasets. These models are typically fine-tuned end-to-end for tasks such as table detection, reconstruction, or TabVQA. The second paradigm leverages Large Language Models (LLMs), which operate entirely in the language space. In this setting, tables are first serialized into textual formats, such as HTML, Markdown, or plain-text formats, before being provided as input prompts to LLMs (Figure (1)). These models then reason over these serialized tables to answer queries. While VLMs benefit from strong visual grounding, they are often monolithic and inflexible, tightly coupling layout reasoning, content extraction, and question answering into a single opaque pipeline. This end-to-end dependency makes them difficult to debug or adapt. Moreover, they require extensive pretraining with image-text pairs. VLMs also show a performance gap, especially for non-Latin scripts or lowresource languages. Crucially, the internal representations they learn are often not interpretable or easily transferable to new tasks. In contrast, LLM-based approaches offer a modular, language-agnostic alternative. Once tables are converted into structured textual formats, general-purpose LLMs can perform downstream tasks like TabVQA with minimal fine-tuning. However, the challenge lies in the table serialization process itself. An important factor that significantly impacts table understanding performance, especially when leveraging LLMs, is the choice of table representation. Simple plain-text formats fail to preserve structural information such as rows, columns, and cells, making it difficult for models to infer relationships between table elements. While formats like LaTeX and HTML can accurately capture table structures, they are often verbose and lead to long token sequences, which strain the input limitations of most LLMs and reduce overall efficiency. Markdown, though more compact, cannot faithfully represent complex structures involving merged cells, making it unsuitable for complex tables. These limitations highlight the need for a compact format, one that preserves both structure and content while remaining well-suited for LLMbased processing. To address this, we advocate the use of the recently proposed Optimised Table Structure Language (OTSL) sequences [29]. OTSL provides a compact yet expressive linear representation of tables, capturing hierarchical layouts, spanning cells, and textual content in a form shorter than HTML while preserving structure. We design a modular pipeline that is doubly decoupled (Figure (1)): first, by separating Table Structure Recognition (TSR) and Optical Character Recognition (OCR) to handle structure and text independently, and second, within TSR itself by disentangling physical (rows, columns) from logical (cell arrangement) structure. This design improves interpretability, robustness across document styles and languages, and scalability for TabVQA. It outperforms recent end-to-end VLMs on standard benchmarks and enables multilingual TabVQA, including low-resource scripts, by shifting reasoning to text-based LLMs using OTSL. Our main contributions are as follows:

![](images/aa160cfd8a53d1716e366df7c30869800c89f2f955c776298ea48ccf5d03d995.jpg)  
Figure 1. TabVQA paradigms: (i) end-to-end, where a VLM directly processes the table image and question; (ii) decoupled, where TSR + OCR extracts structure and content into text (HTML, OTSL) for an LLM; and (iii) our doubly decoupled approach, which further separates physical and logical TSR for better control. The steps following OCR remain the same as for the conventional decoupled approach.

• We present DELTA, a Doubly dEcoupled tabLe reconsTruction Approach that doubly decouples TSR followed by OCR to produce a compact OTSL sequence with high structural fidelity, outperforming recent VLMs.

• To support downstream tasks, we introduce a lossless algorithm to convert widely used HTML format into OTSL, effectively capturing both table structure and content.

• We propose TARQA, TAble structuRe-aware Question Answering, an LLM fine-tuned on OTSL-formatted tables for TabVQA, demonstrating stronger performance over models fine-tuned on HTML and other baselines on WikiTableQuestions (WTQ) and FinTabNetQA datasets.

• Finally, we introduce and release TORQUE, Table Oriented Reconstruction and Question-answering Upon dEvanagari, a new Hindi benchmark to demonstrate the multilingual capability of our framework, showcasing effective performance on both table reconstruction and Tab-VQA tasks for the Hindi language.

## 2. Related Literature

End-to-End Table Reconstruction leverages VLMs for holistic table understanding, aiming to directly generate structured representations (HTML or JSON) from table images. These models use pre-trained vision-language encoders to jointly process visual and textual cues, enabling the prediction of table structures and content. Approaches like MTLTabNet [27], SmolDocLing [35], SmolVLM [31], and Granite-Vision [55] fall into this category. The key advantage of this paradigm is its ability to unify structure and content prediction without relying on intermediate OCR outputs or handcrafted rules. However, these models often struggle with multilingual scripts, complex layouts, and noisy scanned documents, particularly when such variations are underrepresented in training data.

Decoupled Table Reconstruction involves two key steps: TSR and OCR. TSR models are responsible for identifying cell boundaries and relationships. Object detectionbased TSR methods focus on reconstructing the physical layout of tables by localizing cells using models like Faster R-CNN [48], Mask R-CNN [7], YOLO [47], etc. Transformer-based variants such as DETR [5], TATR [50], TableFormer [34], and TSRFormer [22] have also been equipped for this task. However, such object detectionbased TSR often depends on post-processing for mapping cells to row and column numbers, making them sensitive to detection errors. Later, Im2Seq-based TSR methods dominated the field of TSR, where they directly predict the logical structures (cell mapping) from images using encoder-decoder architectures, outputting formats like HTML or LATEX. Recent approaches adopt compact representations like OTSL [19, 29] to improve inference efficiency and structural consistency. Hybrid TSR methods, including graph-based models like GTE [62] and TGRNet [59], treat table reconstruction as a graph problem. Others combine visual detection with token-level generation, such as EDD [64], local attention-based TSR [28], or use visual and positional cues to refine predictions [12, 44, 45]. The emergence of Im2Seq models has often offered better endto-end consistency by synchronising physical and logical structure predictions in a unified pipeline. Table reconstruction datasets such as PubTabNet [64], FinTabNet [63], Pub Tables [50], SynthTabNet [33], and TabRecSet [60] provide diverse annotations, with evaluation metrics like TEDS [64] and GriTS [51] assessing both spatial and structural accu racy. Once the structure is extracted, OCR engines are used to recognize the textual content within each cell. Popular OCR engines suitable for this task include Tesseract [49], DocTR [32], EasyOCR [13], and several others [37, 39].

Table-based Question Answering (TabQA) focuses on reasoning over structured table data, typically provided in CSV or HTML format, to answer natural language questions. Recently, LLMs [3, 6, 15] have been fine-tuned or prompted for direct reasoning over tabular inputs, exhibiting strong zero-shot and few-shot capabilities. However, their performance heavily depends on the quality and format of both the table and the prompt. Thus, the choice of representation used to encode tabular data becomes a critical design decision in decoupled TabVQA pipelines. TabVQA, in contrast, integrates visual understanding with question answering, operating directly on table images. Several datasets, such as ComTQA [61], WTQ [40], and FinTabNetQA [18], have been introduced to support this task. As discussed earlier, there are two main approaches to TabVQA [18]: (i) VLM-based models [3, 9, 10, 21, 25, 57] that jointly encode visual and textual information to generate answers, and (ii) decoupled pipelines that first extract table structure and content via TSR + OCR, then apply LLMs for textual question answering. We build upon this paradigm, where DELTA handles the doubly decoupled table reconstruction, followed by TARQA, our finetuned LLM, for question answering. Both components of our approach are detailed in the upcoming Section (3).

## 3. Our Methodology

Our methodology has two components: DELTA for generating OTSL from table images, and TARQA for TabVQA on these sequences extracted from input table images.

## 3.1. DELTA for Structure

Figure (2) presents an overview of DELTA. Given a table image, we perform TSR followed by OCR to extract the table content, where TSR is also divided into two components: physical structure and logical structure. The final output after TSR and OCR is an HTML representation, which is then transformed into an OTSL sequence in a lossless manner using Algorithm (1). Overall, our methodology for DELTA consists of the following key steps:

## 3.1.1. TSR for Cell Demarcation

For TSR, we employ a combination of SPRINT [19] and TATR [52], which together provide a comprehensive understanding of table layouts. We chose SPRINT as it has shown fast, robust, and language-agnostic performance on logical TSR, making it ideal for capturing cell relationships across diverse scripts. SPRINT provides an HTML sequence corresponding to the input table’s logical structure. TATR, also a recent state-of-the-art model, complements this by focusing on physical structure recognition, accurately identifying rows and columns to demarcate individual cells in the table image. The combined output is a structured HTML tag sequence that represents the table, including complex features such as cell merges, row spans, and column spans, with each <td> tag attributed with precise bounding box coordinates. The output of the TSR step with cells highlighted is seen in Figure (2). We provide a more detailed explanation of how TATR and SPRINT work together to generate the final HTML sequence in the supplementary material.

## 3.1.2. OCR for Content Extraction

The TSR output, an HTML tag sequence with bounding box annotations, is used to perform OCR on each cell. We employ EasyOCR [13] for its seamless integration with TSR, GPU compatibility, and fast processing, while remaining modular and easily replaceable. For each <td> tag, the bounding box crops the corresponding cell from the table image, and EasyOCR extracts the text, which is then embedded back into the HTML string. This enriches the structure with content and extends naturally to multilingual tables. The choice of EasyOCR is further validated through an ablation study with other OCR models, detailed in the supplementary material.

![](images/31e579ece935b820009eeb09ef1a1e1f22208f11bb57317c6ecc38cbccf1e694.jpg)  
Figure 2. DELTA overview: given a table image, the system first performs decoupled TSR (physical structure through TATR and logical structure through SPRINT), followed by OCR to extract table content. The resulting HTML is converted into a compact OTSL sequence.

## 3.1.3. HTML to OTSL conversion

As highlighted in prior work [29], HTML representations are often large and noisy due to their verbose, repetitive tag structures, making them difficult to use directly with LLMs in TabQA tasks. To overcome these limitations, we convert the HTML output into an OTSL sequence, a compact, structured format that efficiently captures both the table’s layout and content. This conversion significantly reduces input length while preserving all necessary information, making it more suitable for LLM-based inference and improving performance. We present Algorithm (1), which outlines the process of converting an HTML sequence into its corresponding OTSL representation in a lossless manner. Figure (3) provides an example of a complex table, showcasing the HTML generated by DELTA, and the resulting OTSL sequence. This comparison demonstrates the compactness and structured clarity of OTSL, which enhances downstream task performance. We refer readers to the original OTSL specification [29] for a detailed understanding of its syntax and token design.

## 3.2. TARQA for Question Answering

वैशिष्ट्येWe introduce TARQA, an LLM fine-tuned for TabQA di-<sup>का</sup>  ा सू र्यंrectly on OTSL sequences. Its key contribution is to reframe <sup>ं चा</sup> <sup>हा</sup> <sup>दा</sup> table understanding as a text-to-text problem, leveraging <sup>मा</sup> <sup>र्च</sup>OTSL, to eliminate the reliance on table images for downstream reasoning tasks. By operating purely on structured OTSL inputs, TARQA naturally supports decoupled processing and multilingual extension. TARQA can be seamlessly paired with DELTA, which produces OTSL representations, thereby enabling a complete TabVQA pipeline as illustrated in Figure (4). The choice of OTSL over alter-<sub>वा</sub>native formats, fine-tuning details, and other related experimentation is described in the upcoming Section (4).

![](images/17fa279a61db1b7f425916eefabd077a89e9fe91f825211de66bd7263a812a48.jpg)  
Figure 3. Illustration of converting a table grid into HTML and then into the OTSL sequence using Algorithm (1).

Algorithm 1 Extract OTSL Matrix from HTML string   
Require: HTML string   
Ensure: OTSL matrix string   
1: Parse HTML, find <table>   
2: Compute R (rows) and C (columns) using rowspan,   
colspan   
3: Initialize otsl matrix[R][C] ← "<ecel>"   
4: Initialize cell map[R][C] ← 0   
5: for each row i do   
6: col idx ← 0   
7: for each cell in row do   
8: while cell map[i][col idx] = 1 do   
9: col idx ← col idx + 1   
10: end while   
11: Extract rowspan, colspan, text   
12: Assign "<fcel>" if text exists, else   
"<ecel>"   
13: Fill merged cells: "<lcel>" (left),   
"<ucel>" (up), "<xcel>" (cross)   
14: Update cell map, move to next column   
15: end for   
16: end for   
17: Convert otsl matrix to string with "<nl>" separators   
18: return OTSL matrix

![](images/0167e2ddb2fcd3cff5c8bb540a7ed69524240be86de8b305116f762bc100316c.jpg)  
Figure 4. TARQA, an LLM fine-tuned on OTSL sequence, uses output from DELTA along with the query to generate answers.

## 4. Experiments

Now that we have described DELTA for producing OTSL sequences, we proceed to validate the effectiveness of OTSL as the most suitable representation for TabQA (and eventually TabVQA). We conduct a series of experiments comparing OTSL with other common table representations. We first describe the datasets used, followed by a detailed explanation of the experimental setup.

## 4.1. Datasets

We begin by describing the English and non-English tablebased datasets we use in our experimental setup.

<table><tr><td>Dataset</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>FinTabNet</td><td></td><td>一</td><td>10305</td></tr><tr><td>PubTabNet</td><td></td><td>6942</td><td></td></tr><tr><td>PubTables</td><td></td><td></td><td>92841</td></tr><tr><td>WTQ</td><td>11321</td><td>一</td><td>7175</td></tr><tr><td>FinTabNetQA</td><td></td><td>一</td><td>250</td></tr></table>

Table 1. Datasets used for reconstruction, TabQA and TabVQA

## 4.1.1. English Datasets

As DELTA is a framework composed of multiple off-theshelf blocks, we skip training and directly infer and report results on FinTabNet (test), PubTables (test), and PubTab-Net (validation, for fair comparison with prior work). For TabQA, we fine-tune LLMs on WTQ with different table representations (HTML, plain text, OTSL) and select the best-performing variant of TARQA. Finally, we evaluate the full TabVQA pipeline of DELTA followed by TARQA on FinTabNetQA. Dataset statistics are present in Table (1).

## 4.1.2. TORQUEDataset

We introduce TORQUE, a Hindi benchmark for evaluating both Hindi table reconstruction and Hindi TabVQA. It contains 210 tables : 109 scanned and 101 digital-born, cropped, and sourced from government circulars [30, 46] and spiritual books from MUSTARD [19], with a mix of simple (149) and complex (61) table structures. Figure (5) shows a few sample images from the dataset. The dataset also includes 422 manually verified QA pairs, generated using GPT-oss-20B [1]. For table reconstruction, ChatGPT-4o [36] outputs were manually post-corrected to obtain the exact ground-truth HTML sequences. TORQUE is used to benchmark open-source models and our proposed pipeline, showing that our approach is language-agnostic and outperforms conventional decoupled pipelines and significant VLMs without any Hindi-specific finetuning (zero-shot).

## 4.2. Generating Table Representations

The WTQ training set already contains HTML sequences (ground truth). To create inputs in other formats, we apply format-specific conversions: for OTSL, we use our proposed Algorithm (1) to convert the HTML into a compact sequence; and for plain text, we parse the HTML content row-wise (left to right) and flatten it into a simple text string. Based on the chosen representation format, the corresponding LLM is fine-tuned for the TabQA task.

## 4.3. Finetuning TARQA Variants

We fine-tuned the Meta-LLaMA-3-8B-Instruct [2] for the question-answering task to generate concise, accurate natural language answers based on a table OTSL sequence and a question. The model was fine-tuned using a custom instruction-tuned setup, where each example followed a structured prompt template that included an instruction, a serialised table, a natural language question, and the ground truth answer. Table (2) summarises the fine-tuning configuration of TARQA on the WTQ dataset. The dataset was preprocessed using the HuggingFace AutoTokenizer with padding applied up to a maximum sequence length of 4096 tokens and truncation enabled. Model inference was carried out using greedy decoding. All experiments were conducted using a CUDA-enabled NVIDIA H100 80GB GPU device. The prompt used for fine-tuning is as follows:

<table><tr><td colspan="6">yaiaR  TT 3RGR HRA HHS</td></tr><tr><td>可</td><td colspan="2">HR  3TT 啊国</td><td>( 耐式)</td><td>37 ( 耐式)</td><td>AAR</td></tr><tr><td colspan="4">3T 3HH </td><td>34.17 69</td><td>4,330</td></tr><tr><td colspan="4">2,19,092</td><td>1,287.39 4,907</td><td>4,87,871</td></tr><tr><td colspan="3">25,283</td><td>403.67</td><td>703</td><td>1,53,715</td></tr><tr><td colspan="3">54,101</td><td>164.45</td><td>481</td><td>75,607</td></tr><tr><td colspan="3">12,529</td><td>139.89</td><td>207</td><td>28,622</td></tr><tr><td colspan="3">15,623</td><td>378.55</td><td>650</td><td>66,466</td></tr><tr><td colspan="3">415</td><td>12.60</td><td>44</td><td>1,580</td></tr><tr><td colspan="3">27,448</td><td>329.27</td><td>461</td><td>62,861</td></tr><tr><td colspan="3">3,56,002</td><td>2,749.99</td><td>7,412</td><td>8,81,052</td></tr><tr><td colspan="3">3  119.50 </td><td>1,78,699.00</td><td>4,18,263</td><td>282.57 </td></tr><tr><td colspan="6">3</td></tr><tr><td>页 ·</td><td>44-</td><td>3 F</td><td>PR R RT 同向对</td><td></td><td>AH</td></tr><tr><td>1.</td><td>RFRER()</td><td>01</td><td>3()5/2001/21-可() 16.07.2002</td><td></td><td>On Scale</td></tr><tr><td>2.</td><td>TRRTETR</td><td>03</td><td>3()13/1994/21-4(5) 11/13.10.1994</td><td></td><td>On Scale</td></tr><tr><td>3.</td><td></td><td>01</td><td>17()2/1988/21-可() 22/23.04.1994</td><td></td><td>70290-1540— 76450</td></tr><tr><td>4.</td><td>3ifaR a</td><td>01</td><td>17(支)2/1988/21-可() 22/23.04.1994</td><td></td><td>On Scale</td></tr><tr><td>5.</td><td>T 3ITETTRT (Peet STRE .3T.T.3TR.3T度)</td><td>01</td><td>17(3)2/1988/21-可() 22/23.04.1994</td><td></td><td>On Scale</td></tr></table>

Figure 5. Sample Hindi images from the TORQUE dataset

```markdown
### Prompt Used for Finetuning:
Given the following table, answer the
question in one word or a short phrase.
Do not provide an explanation.
### Table: {OTSL sequence}
### Question: {User query}
### Answer: {answer}
```

## 4.4. Ablation Study for OTSL Format

The effectiveness of OTSL for TabVQA is demonstrated through our ablation study (Table (3)). We compare different input formats used to fine-tune as well as infer across both off-the-shelf and fine-tuned LLMs (denoted by the prefix ’TARQA’ followed by the format used for finetuning). We show that fine-tuning using OTSL led to the best scores. For each format, we finetune using the same configuration as detailed in Section (4.3) and report both ANLS and Ex-

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Model</td><td>LLaMA-3-8B-Instruct</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>2e−5</td></tr><tr><td>Batch Size</td><td>1 (per GPU)</td></tr><tr><td>Number of Epochs</td><td>4</td></tr><tr><td>Sequence Length</td><td>4096 tokens</td></tr><tr><td>Tokenizer Pad Token</td><td>Set to eos_token</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Loss</td><td>Cross-Entropy</td></tr></table>

Table 2. Fine-tuning parameter details for TARQA

act Match (EM) metrics. TARQA-OTSL shows a gain of around 14 p.p. over TARQA-HTML on both metrics.
<table><tr><td>Inferred On</td><td>LLM Model [2] ANLS</td><td>EM</td></tr><tr><td rowspan="2">OTSL</td><td>Off the Shelf 25.4</td><td>16.9</td></tr><tr><td>TARQA-OTSL 56.5</td><td>54.0</td></tr><tr><td rowspan="2">HTML</td><td>Off the Shelf</td><td>11.0 10.3</td></tr><tr><td>TARQA-HTML 42.3</td><td>39.9</td></tr><tr><td rowspan="2">Plain Text</td><td>Off the Shelf</td><td>29.8 27.8</td></tr><tr><td>TARQA-Plain</td><td>52.8 50.2</td></tr></table>

Table 3. Comparing the impact of fine-tuning on different table representations on the WTQ test set with best scores highlighted.

## 5. Results and Discussions

On Table Reconstruction: We evaluate the effectiveness of our approach, DELTA, across multiple table reconstruction benchmarks. Specifically, DELTA generates an HTML tag sequence, which is subsequently converted into OTSL, capturing both structure and content. Table (4) presents a comparison of DELTA with state-of-the-art methods, grouped into end-to-end VLMs, conventional decoupled approaches, and our proposed doubly decoupled framework. The evaluation spans FinTabNet, PubTabNet, PubTables, and TORQUE datasets using both TEDS-S and TEDS metrics. Results show that DELTA achieves competitive structural alignment, as reflected in consistently strong TEDS-S scores across all benchmarks, while also highlighting the advantages of our doubly decoupled design. While DELTA outperforms most VLMs in TEDS, it does not surpass MTL-TabNet, which lacks generalizability, or methods like VAST [12] and TableFormer [34] that exploit PDF parsing for near-perfect text fidelity. Our reliance on OCR introduces the main bottleneck. Although DELTA’s modularity allows easy replacement of the OCR component to boost TEDS scores. As shown in the supplementary material, OCR ablation demonstrates that ground-truth–mapped content achieves near-perfect TEDS, confirming that reconstruction itself is reliable and OCR is the only limitation. Overall, DELTA shows strong structural understanding and achieves solid performance on TORQUE, even with challenging images, while VLM-based methods such as SmolVLM [31] and SmolDocling [35] remain difficult to extend to multilingual data.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Dataset</td><td colspan="2">FinTabNet</td><td colspan="2">PubTabNet</td><td colspan="2">PubTables</td><td colspan="2">TORQUE</td></tr><tr><td>TEDS-S TEDS</td><td></td><td>TEDS-S TEDS</td><td></td><td>TEDS-S TEDS</td><td></td><td>TEDS-S TEDS</td><td></td></tr><tr><td rowspan="4">End-to-end VLMs</td><td>SmolVLM[31]</td><td></td><td>18.0</td><td></td><td></td><td></td><td>32.0</td><td>0.00</td><td>0.05</td></tr><tr><td>SmolDocling [35]</td><td>81.0</td><td>52.0</td><td></td><td></td><td>65.0</td><td>88.0</td><td>0.05</td><td>4.50</td></tr><tr><td>Granite Vision [55]</td><td></td><td>54.0</td><td></td><td></td><td></td><td>70.0</td><td></td><td></td></tr><tr><td>MTL-TabNet [26]</td><td>98.8</td><td></td><td>97.9</td><td>96.7</td><td>96.7</td><td>96.2</td><td>一</td><td>一</td></tr><tr><td rowspan="2">TSR + OCR</td><td>EDD [64]</td><td>90.6</td><td>=</td><td>89.9</td><td>88.3</td><td>一</td><td>=</td><td>=</td><td></td></tr><tr><td>LGPMA [44]</td><td></td><td></td><td>96.7</td><td>94.6</td><td>一</td><td>1</td><td></td><td></td></tr><tr><td rowspan="2">TSR + PDF Parsing</td><td>TableFormer*[34]</td><td>96.8</td><td>89.0</td><td>96.7</td><td>93.6</td><td>一</td><td>84.0</td><td>一</td><td></td></tr><tr><td>VAST*[11]</td><td>98.6</td><td>98.2</td><td>97.2</td><td>96.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Doubly Decoupled</td><td>DELTA(Ours)</td><td>98.2</td><td>55.9</td><td>97.5</td><td>54.4</td><td>97.7</td><td>54.8</td><td>85.4</td><td>63.6</td></tr></table>

Table 4. Comparison of TEDS-S and TEDS scores for different table reconstruction methods across popular benchmarks, along with TORQUE. Approaches marked by \* use PDF-parsing for content extraction. The best and second-best results are highlighted.

<table><tr><td rowspan="2">Baseline</td><td colspan="2">Fine-tuned on</td></tr><tr><td>HTML</td><td>OTSL</td></tr><tr><td>UDOP [54]</td><td>47.2</td><td>一</td></tr><tr><td>Pix2struct [20]</td><td>39.8</td><td></td></tr><tr><td>DocOwl [10]</td><td>26.9</td><td></td></tr><tr><td>Kosmos [41]</td><td>32.4</td><td></td></tr><tr><td>Donut [17]</td><td>18.8</td><td></td></tr><tr><td>TAPAS [8]</td><td>46.4</td><td></td></tr><tr><td>Mistral [14]</td><td></td><td>29.9</td></tr><tr><td>SOLAR [16]</td><td></td><td>12.3</td></tr><tr><td>TARQA</td><td>42.3</td><td>56.5</td></tr></table>

Table 5. Comparison of ANLS scores on the WTQ test set. The best and second-best results are highlighted for clarity.

On TabQA and TabVQA: In Table (5), we evaluate the impact of structured outputs on the downstream TabQA task. Specifically, we compare TARQA-OTSL and TARQA-HTML, both of which take DELTA’s correspond ing reconstructed outputs. To ensure consistency, we report HTML baseline results from prior work, all fine-tuned on the WTQ dataset. For OTSL baselines, we strictly finetune LLMs using the same setup as TARQA. The results clearly show that OTSL-based inputs achieve significantly higher ANLS scores, outperforming the strongest HTML baseline by 9.3 p.p.. We evaluate the performance of our integrated pipeline, DELTA + TARQA, on the FinTabNetQA dataset [18] for the TabVQA task. As shown in Table (6), our approach achieves higher relieved accuracy than all other baselines. We adopt relieved accuracy as the evaluation metric because it accounts for semantically equivalent answers that may differ in surface form (e.g., numerical formatting, currency symbols, or minor textual variations). The skylines (with ground-truth sequences) outperform all other methods. Notably, the OTSL-based skyline achieves the highest score, indicating that more accurate OTSL-based table reconstruction (higher TEDS) directly translates to stronger downstream performance in TabVQA.

<table><tr><td>Type</td><td>Model</td><td>FinTabNetQA</td></tr><tr><td rowspan="6">Open-source</td><td>BLIP-2 [21]</td><td>0.4</td></tr><tr><td>CogVLM-1k [57]</td><td>4.8</td></tr><tr><td>CogAgent-VQA [9]</td><td>22.8</td></tr><tr><td>SPHINX-v1-1k [23]</td><td>3.2</td></tr><tr><td>LLaVA-1.5 [24]</td><td>0.8</td></tr><tr><td>QWEN-VL-Chat [3] QWEN-VL [3]</td><td>29.6</td></tr><tr><td rowspan="3"></td><td>SPHINX-MoE-1k [18]</td><td>34.0 36.0</td></tr><tr><td>Closed-source SPHINX-v2-1k [18]</td><td>31.2</td></tr><tr><td>SPHINX-MoE [18]</td><td>2.8</td></tr><tr><td rowspan="2">Ours</td><td>DELTA + TARQA-HTML</td><td>29.2</td></tr><tr><td>DELTA + TARQA-OTSL</td><td>45.2</td></tr><tr><td rowspan="2">Skyline</td><td>GT-HTML + TARQA-HTML</td><td>51.2</td></tr><tr><td>GT-OTSL + TARQA-OTSL</td><td>69.2</td></tr></table>

Table 6. Comparison of relieved accuracy scores for TabVQA on FinTabNetQA. The results include many baselines alongside our method. Additionally, for skyline, we provide Ground Truth (GT) HTML and OTSL inputs to the corresponding TARQA variant. The best and second-best scores are highlighted.

<table><tr><td>Approach</td><td>VLM/LLM</td><td>Model Parameters</td><td>Relieved-Acc.</td><td>EM</td><td>ANLS</td></tr><tr><td rowspan="4">VLMs (End-to-End)</td><td>Qwen-2.5-VL Instruct [4]</td><td>7B</td><td>47.40</td><td>46.20</td><td>66.40</td></tr><tr><td>InternVL-3_5 [58]</td><td>8B</td><td>17.54</td><td>16.35</td><td>19.19</td></tr><tr><td>Paligemma-2 [53]</td><td>3B</td><td>10.20</td><td>07.10</td><td>13.90</td></tr><tr><td>SmolVLM-Instruct [31]</td><td>3B</td><td>14.45</td><td>00.95</td><td>14.45</td></tr><tr><td rowspan="6">DELTA + LLM</td><td>Qwen-2.5-Hindi [56]</td><td>14B</td><td>21.30</td><td>20.40</td><td>26.50</td></tr><tr><td>Mistral [14]</td><td>7B</td><td>6.64</td><td>6.16</td><td>9.00</td></tr><tr><td>HiTQA-mBart [38]</td><td>611M</td><td>11.22</td><td>1.43</td><td>1.91</td></tr><tr><td>HiTQA-M2M [38]</td><td>484M</td><td>5.73</td><td>0.24</td><td>0.24</td></tr><tr><td>mBERT [42]</td><td>179M</td><td>0.47</td><td>0.24</td><td>0.95</td></tr><tr><td>TAPAS [8]</td><td>110M</td><td>4.50</td><td>4.50</td><td>6.60</td></tr><tr><td rowspan="2">Ours</td><td>DELTA(HTML) + TARQA-HTML</td><td>8B</td><td>12.80</td><td>12.09</td><td>16.11</td></tr><tr><td>DELTA(OTSL) + TARQA-OTSL</td><td>8B</td><td>27.49</td><td>23.93</td><td>36.49</td></tr><tr><td rowspan="2">Skyline</td><td>GT HTML + TARQA-HTML</td><td>8B</td><td>28.44</td><td>28.20</td><td>31.75</td></tr><tr><td>GT OTSL + TARQA-OTSL</td><td>8B</td><td>63.51</td><td>60.43</td><td>76.30</td></tr></table>

Table 7. Comparative results on TORQUE for the TabVQA task. We also report our results with both HTML and OTSL variants of TARQA, along with ground-truth inputs to TARQA as the skyline. The best and second-best results are highlighted for clarity.

On Multilingual TabVQA: To showcase the multilingual capability of our framework, we report results on the TORQUE dataset using different categories of models. Specifically, we evaluate both end-to-end VLMs and several LLMs, where the latter take as input the outputs of DELTA along with the question to be asked. Table (7) presents the results, demonstrating that our approach consistently outperforms all VLMs by more than 10 p.p. except Qwen-2.5 VL Instruct [4]. Qwen achieves stronger performance only because its pre-training corpus includes Hindi data, which gives it an inherent advantage. For LLMs with Hindi capability, none can surpass our approach. This improvement is notable as DELTA achieves zero-shot performance on TORQUE, with no components trained on Hindi. Furthermore, the skylines (obtained by feeding the ground-truth sequences to TARQA) achieve substantially higher performance than Qwen [4] and all other VLMs, highlighting that a decoupled module is considerably more effective than end-to-end VLMs. This also indicates that if the inputs to TARQA are of higher quality (i.e., better OCR), as in the case of ground-truth HTML, the downstream performance can be significantly benefited. Qualitative examples for all tasks are included in the supplementary material.

## 6. Conclusion

In summary, we present a comprehensive pipeline for table understanding and reasoning. We begin with DELTA, a doubly decoupled table reconstruction framework that separates structure recognition from OCR and disentangles physical and logical TSR, producing a compact OTSL representation. DELTA achieves high TEDS-S scores and surpasses recent VLM-based approaches. We further introduce a lossless HTML-to-OTSL conversion for interoperability and TARQA, an LLM fine-tuned on OTSL for TabVQA, which achieves strong results on WTQ and FinTabNetQA. Finally, we showcase the performance of DELTA and DELTA + TARQA on the curated Hindi benchmark TORQUE, where our zero-shot approach significantly outperforms other models, including LLMs with inherent Hindi understanding. Collectively, these contributions define a flexible and extensible framework for highfidelity multilingual table reconstruction and TabVQA.

## 7. Limitations and Future Work

While DELTA demonstrates good TEDS-S scores, the overall TEDS scores can still be improved. This is due to the current OCR support in the pipeline, which introduces errors that not only affect the TEDS score but can also impact downstream TabVQA performance. Improving OCR quality will push TEDS closer to 100%. This, in turn, would enable DELTA + TARQA to consistently achieve scores close to skyline performance. In this way, we will distil the challenges of black-box VLMs into a more well-defined formulation, making Multilingual TabVQA both transparent and debuggable. Our experiments are limited to English and Hindi benchmarks, but the modular design and OCR support will enable extension to other languages. Future work will focus on enhancing reasoning for complex queries, such as abstractive QA, thereby broadening the scope and impact of our approach.

## 8. Acknowledgement

We acknowledge BharatGen and the Indian Institute of Technology Bombay for providing resources and support for the project. Jahanvi Rajput’s PhD is supported by the Prime Minister’s Research Fellowship (PMRF).

## References

[1] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025. 5

[2] AI@Meta. Llama 3 model card. 2024. 5, 6

[3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 3, 7

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 8

[5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 3, 2

[6] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 3

[7] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross B.´ Girshick. Mask R-CNN. CoRR, abs/1703.06870, 2017. 3

[8] Johan Holmgren, Paul Davidsson, Jan A Persson, and Linda Ramstedt. Tapas: A multi-agent-based model for simulation of transport chains. Simulation Modelling Practice and Theory, 23:1–18, 2012. 7, 8

[9] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023. 3, 7

[10] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024. 3, 7

[11] Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng. Improving table structure recognition with visual-alignment sequential coordinate modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11134–11143, 2023. 7

[12] Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng. Improving table structure recognition with visual-alignment sequential coordinate modeling. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11134–11143, 2023. 3, 6

[13] JaidedAI. Easyocr: Ready-to-use ocr with 80+ supported languages and all popular writing scripts. https:// github.com/JaidedAI/EasyOCR, 2020. Accessed: 2025-04-20. 3

[14] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne´ Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed.´ Mistral 7b: Efficient high-performance open weights. arXiv preprint, arXiv:2310.06825, 2023. Accessed: 2025-09-13. 7, 8

[15] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. 3

[16] Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. Solar 10.7b: Scaling large language models with simple yet effective depth up-scaling, 2023. 7

[17] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sang doo Yun, Dongyoon Han, and Seunghyun Park. Ocrfree document understanding transformer. arXiv preprint arXiv:2111.15664, 2021. 7

[18] Yoonsik Kim, Moonbin Yim, and Ka Yeon Song. Tablevqa bench: A visual question answering benchmark on multiple table domains. arXiv preprint arXiv:2404.19205, 2024. 3, 7

[19] Dhruv Kudale, Badri Vishal Kasuba, Venkatapathy Sub ramanian, Parag Chaudhuri, and Ganesh Ramakrishnan. Sprint: Script-agnostic structure recognition in tables. arXiv preprint arXiv:2503.11932, 2025. 3, 5, 1

[20] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language under standing. arXiv preprint arXiv:2210.03347, 2022. 7

[21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3, 7

[22] Weihong Lin, Zheng Sun, Chixiang Ma, Mingze Li, Jiawei Wang, Lei Sun, and Qiang Huo. Tsrformer: Table structure recognition with transformers. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6473– 6482, 2022. 3

[23] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 7

[24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 7

[25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3

[26] Nam Tuan Ly and Atsuhiro Takasu. An end-to-end multitask learning model for image-based table recognition. arXiv preprint arXiv:2303.08648, 2023. 7

[27] Nam Tuan Ly and Atsuhiro Takasu. An end-to-end multitask learning model for image-based table recognition. arXiv preprint arXiv:2303.08648, 2023. 2

[28] Nam Tuan Ly and Atsuhiro Takasu. An end-to-end local attention based model for table recognition. In Document Analysis and Recognition - ICDAR 2023, pages 20–36, Cham, 2023. Springer Nature Switzerland. 3

[29] Maksym Lysak, Ahmed Nassar, Nikolaos Livathinos, Christoph Auer, and Peter Staar. Optimized table tokenization for table structure recognition. In Document Analysis and Recognition - ICDAR 2023, pages 37–50, Cham, 2023. Springer Nature Switzerland. 2, 3, 4

[30] Madhya Pradesh High Court. Circulars and orders, 2025. Accessed: 2025-03-25. 5

[31] Andres Marafioti, Orr Zohar, Miquel Farr ´ e, Merve Noyan,´ Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025. 2, 7, 8

[32] Mindee. doctr: Document text recognition. https:// github.com/mindee/doctr, 2021. 3

[33] Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. Tableformer: Table structure understanding with transformers. arXiv preprint arXiv:2203.01017, 2022. 3

[34] Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. Tableformer: Table structure understanding with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4614– 4623, 2022. 3, 6, 7, 1

[35] Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer, Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A. Said Gurbuz, Michele Dolfi, Miquel Farre, and Peter W. J.´ Staar. Smoldocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion. arXiv preprint arXiv:2503.11576, 2025. 2, 7

[36] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, and Lama Ahmad et al. Gpt-4 technical report, 2024. 5

[37] PaddlePaddle. Paddleocr: A rich, practical and productionready ocr library based on paddlepaddle. https : //github.com/PaddlePaddle/PaddleOCR, 2020. Accessed: 2025-04-20. 3

[38] Vaishali Pal, Evangelos Kanoulas, Andrew Yates, and Maarten de Rijke. Table question answering for low-resourced indic languages. arXiv preprint arXiv:2410.03576, 2024. 8

[39] Vik Paruchuri. Surya: Ocr, layout analysis, reading order, and table recognition in 90+ languages. https: //github.com/VikParuchuri/surya, 2023. Accessed: 2025-04-20. 3

[40] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages

1470–1480, Beijing, China, 2015. Association for Computational Linguistics. 3

[41] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 7

[42] Telmo Pires, Eva Schlinger, and Dan Garrette. How multilingual is multilingual bert? arXiv preprint arXiv:1906.01502, 2019. 8

[43] Devashish Prasad, Ayan Gadpal, Kshitij Kapadni, Manish Visave, and Kavita Sultanpure. Cascadetabnet: An approach for end to end table detection and structure recognition from image-based documents, 2020. 1

[44] Liang Qiao, Zaisheng Li, Zhanzhan Cheng, Peng Zhang, Shiliang Pu, Yi Niu, Wenqi Ren, Wenming Tan, and Fei Wu. Lgpma: Complicated table structure recognition with local and global pyramid mask alignment. In International confer ence on document analysis and recognition, pages 99–114. Springer, 2021. 3, 7, 1

[45] Sachin Raja, Ajoy Mondal, and CV Jawahar. Table structure recognition using top-down and bottom-up cues. In European conference on computer vision, pages 70–86. Springer, 2020. 3

[46] Rajbhasha Department. Orders and circulars, 2025. Accessed: 2025-03-25. 5

[47] Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. CoRR, abs/1506.02640, 2015. 3

[48] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. CoRR, abs/1506.01497, 2015. 3

[49] R. Smith. An overview of the tesseract ocr engine. In Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), pages 629–633, 2007. 3

[50] Brandon Smock, Rohith Pesala, and Robin Abraham. Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4634–4642, 2022. 3, 1

[51] Brandon Smock, Rohith Pesala, and Robin Abraham. Grits: Grid table similarity metric for table structure recognition. In International Conference on Document Analysis and Recognition, pages 535–549. Springer, 2023. 3

[52] Brandon Smock, Rohith Pesala, and Robin Abraham. Aligning benchmark datasets for table structure recognition. arXiv preprint arXiv:2303.00716, 2023. 3

[53] Andreas Steiner, Andre Susano Pinto, Michael Tschannen,´ Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024. 8

[54] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. arXiv preprint arXiv:2212.02623, 2022. 7

[55] Granite Vision Team, Leonid Karlinsky, Assaf Arbelle, Abraham Daniels, Ahmed Nassar, Amit Alfassi, Bo Wu, Eli Schwartz, Dhiraj Joshi, Jovana Kondic, et al. Granite vision: a lightweight, open-source multimodal model for enterprise intelligence. arXiv preprint arXiv:2502.09927, 2025. 2, 7

[56] Traversaal.ai and 1-800-LLMs. Qwen-2.5-14b-hindi. https://huggingface.co/large-traversaal/ Qwen-2.5-14B-Hindi, 2024. Accessed: 2025-09-13. 8

[57] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3, 7

[58] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 8

[59] Wenyuan Xue, Baosheng Yu, Wen Wang, Dacheng Tao, and Qingyong Li. Tgrnet: A table graph reconstruction network for table structure recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1295–1304, 2021. 3

[60] Fan Yang, Lei Hu, Xinwu Liu, Shuangping Huang, and Zhenghui Gu. A large-scale dataset for end-to-end table recognition in the wild. Scientific Data, 10(1):110, 2023. 3

[61] Weichao Zhao, Hao Feng, Qi Liu, Jingqun Tang, Binghong Wu, Lei Liao, Shu Wei, Yongjie Ye, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. Tabpedia: Towards comprehensive visual table understanding with concept synergy. In Advances in Neural Information Processing Systems, 2024. 3

[62] Xinyi Zheng, Doug Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang. Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context, 2020. 3, 1

[63] Xinyi Zheng, Doug Burdick, Lucian Popa, Peter Zhong, and Nancy Xin Ru Wang. Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. Winter Conference for Applications in Computer Vision (WACV), 2021. 3

[64] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020. 3, 7

# Tables Decoded: DELTA for Structure, TARQA for Understanding Supplementary Material

## 9. Symbols and abbreviations

Table (8) presents the terminology used in the proposed approaches, along with their full forms and the corresponding tasks in which they are applied. Table (9) lists all the abbreviations used throughout the paper, providing readers with an easy reference to understand the terminology better.

## 10. Motivation for Doubly Decoupled Approach

Table Reconstruction approaches can be divided into:

• End-to-end VLMs In conventional table understanding pipelines, physical structure, logical structure, and content recognition (OCR) are often bundled together into a monolithic framework. We refer to this as the VLMs paradigm, where all three aspects: cell boundaries, spanning relations, and textual extraction are jointly modeled. While such end-to-end systems simplify design, they are typically hard to interpret, debug, and adapt across domains and languages.

• Conventional Decoupling: Recent approaches such as CascadeTabNet [43], TATR [50], GTE [62], TableFormer [34], and LGPMA [44] attempt to decouple structure recognition from content recognition. However, within the structure recognition stage, the physical structure (rows, columns, cell boundaries) and the logical structure (spanning cells, header associations, merged cells) remain entangled. This coupling often limits flexibility and makes it challenging to localise errors, as mispredictions in geometry and semantics influence each other.

• Our Doubly-Decoupled Framework: (DELTA) We introduce a finer decomposition by separating physical and logical structure recognition into independent stages. In the structure recognition stage, the physical structure (cell boundaries) and the logical structure (cell arrangements) should be loosely coupled. The physical structure is inherently tied to image coordinates, whereas the logical structure only requires predicting the arrangement and relationships among cells. By decoupling the two, we can keep the logical layer language-agnostic, while isolating coordinate-dependent errors in the physical layer. This separation improves flexibility, simplifies debugging, and allows each component to be strengthened independently. Specifically, TATR models the physical layout, while SPRINT captures logical relations. Their outputs are then combined into a complete table structure, which is subsequently passed to OCR for content extraction. This double decoupling offers greater control and modularity.

## 11. DELTA in detail

Our proposed framework, DELTA, tackles the longstanding challenge of disentangling physical and logical structures in table structure recognition. Unlike traditiona approaches, where geometric layout (rows, columns, cell boundaries) and logical layout (cell arrangement, spans, header associations) are tightly coupled, DELTA predicts them independently and then combines their outputs in a principled manner. Specifically, we employ SPRINT for logical structure prediction, which generates compact OTSL/HTML sequences and offers an ideal balance of speed, accuracy, and language independence, making it robust across multilingual and noisy documents. For the physical structure, we use TATR, a DETR-based model pretrained on large-scale datasets, that leverages only row and column predictions to ensure clean grid alignment. These two components are integrated to reconstruct complete HTML tables with explicit bounding boxes, row spans, and column spans, making the system modular, interpretable, and extensible. Beyond structure prediction, we also examine the role of OCR for content recognition and conduct comprehensive ablations to validate and justify our design choices.

## 11.1. SPRINT for Logical Structure

SPRINT is an image-to-sequence model that employs a Global Context Attention (GCA)-based encoder and a transformer-based decoder to generate compact sequence representations (OTSL/HTML) of logical table structures. We adopt SPRINT for logical structure prediction because it achieves an ideal balance of speed, accuracy, and language independence. Unlike methods that rely heavily on OCR or language-specific cues, SPRINT focuses solely on the structural layout, making it inherently robust across multilingual and noisy document settings. This design aligns perfectly with our objective of decoupling physical and logical structure, as SPRINT cleanly predicts the cell arrangement without being confounded by text semantics. Moreover, since it has already demonstrated state-of-the-art performance on table structure recognition benchmarks, it provides a reliable backbone for our framework. We report ablations reported on SPRINT to demonstrate these strengths in Table (10). For further implementation details, we refer the reader to the original SPRINT [19] paper.

## 11.2. TATR for Physical Structure

To extract the physical structure, we use the TATR [50] V1.1 model pre-trained on FinTabNet, PubTabNet, and

<table><tr><td>Abbreviation</td><td>Description</td><td>Task Used For</td></tr><tr><td>DELTA</td><td>Doubly dEcoupled tabLe reconsTruction Approach (a proposed approach)</td><td>Table Reconstruction, measured by TEDS-Structure and TEDS Scores</td></tr><tr><td>TARQA</td><td>TAble structuRe-aware Question Answering (a fine-tuned LLM on OTSL sequences)</td><td>TabQA Table-based Question Answering</td></tr><tr><td>TORQUE</td><td>Table Oriented Reconstruction and Question- answering Upon dEvanagari (curated benchmark)</td><td>Hindi Table Reconstruction Hindi TabVQA</td></tr><tr><td>DELTA + TARQA</td><td>DELTA converts Table Image to OTSL TARQA answers Questions with OTSL</td><td>Decoupled VQA task</td></tr><tr><td>TARQA-OTSL</td><td>TARQA Fine-tuned on OTSL sequences.</td><td>TabVQA</td></tr><tr><td>TARQA-HTML GT OTSL +</td><td>TARQA Fine-tuned on HTML sequences.</td><td>TabVQA</td></tr><tr><td>TARQA-OTSL</td><td>Ground Truth OTSL given to TARQA-OTSL</td><td>TabVQA</td></tr><tr><td>GT HTML + TARQA-HTML</td><td>Ground Truth OTSL given to TARQA-OTSL</td><td>TabVQA</td></tr></table>

Table 8. Abbreviations and Their Descriptions

<table><tr><td>Abbreviation</td><td>Description</td></tr><tr><td>ANLS</td><td>Average Normalized Levenshtein Similarity</td></tr><tr><td>EM</td><td>Exact Match</td></tr><tr><td>GT</td><td>Ground Truth</td></tr><tr><td>LLMs</td><td>Large Language Models</td></tr><tr><td>OCR</td><td>Optical Character Recognition</td></tr><tr><td>OTSL</td><td>Optimized Table</td></tr><tr><td>p.p.</td><td>Structure Language percentage point</td></tr><tr><td>SPRINT</td><td>Script-agnostic Structure</td></tr><tr><td></td><td>Recognition in Tables</td></tr><tr><td>TabQA</td><td>Table Question Answering</td></tr><tr><td>TabVQA</td><td>Table Visual Question Answering</td></tr><tr><td>TATR</td><td>Table Transformer Tree Edit Distance</td></tr><tr><td>TEDS</td><td>-based Similarity</td></tr><tr><td>TSR</td><td>Table Structure Recoginition</td></tr><tr><td>VLMs</td><td>Vision Language Models</td></tr><tr><td>WTQ</td><td>WikiTableQuestions</td></tr></table>

Table 9. General abbreviations used in the paper.

PubTables-1M. TATR, built on DETR [5], predicts six classes, of which we only leverage table-row and table-column to estimate the rows and columns. For inference, we set the detection threshold to 0.25 and apply non-maximum suppression (NMS) with an IoU threshold of 0.25 on table-row predictions to minimize overlap and improve consistency. The resulting values are then aligned with the output sequence predicted by SPRINT, ensuring coherence between physical and logical structures.

## 11.3. TATR and SPRINT for Complete TSR

This step is responsible for integrating the logical structure (tag sequence predicted by SPRINT) with the physical structure (list of bounding boxes corresponding to detected rows and columns) to produce a final HTML representation of the table. Each <td> element in the output is annotated with its bounding box coordinates, as well as rowspan and colspan attributes whenever merged cells are detected. While row and column bounding boxes intersect to form candidate cells, the crucial constraint is that there exists a one-to-one mapping between each logical cell predicted by SPRINT and its corresponding physical bounding box. The algorithm enforces this alignment to guarantee that both spatial positioning and spanning attributes are preserved.

Formally, as seen in Algorithm (2), it takes as input an OTSL matrix M (generated by SPRINT, note that it can be converted to HTML in a lossless manner, but since SPRINT directly gives an OTSL string, we leverage that initially) and a set of bounding boxes Cells (derived from TATR). The OTSL matrix encodes the table layout, where each entry specifies whether a position corresponds to a cell (C), a new row marker (N), or is empty. The process begins by initializing an empty HTML string H. For each entry (i, j) in M:

• If it corresponds to a cell, the bounding box cell is retrieved from Cells. The function get cell spans computes the extent of row and column spans by checking consecutive overlaps in the SPRINT output.

<table><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Training</td><td rowspan=1 colspan=1>SPRINT Config</td><td rowspan=1 colspan=1>TEDS-SSimple</td><td rowspan=1 colspan=1>TEDS-SComplex</td><td rowspan=1 colspan=1>TEDS-SOverall</td></tr><tr><td rowspan=4 colspan=1>PubTabNet</td><td rowspan=1 colspan=1>PubTabNet</td><td rowspan=1 colspan=1>*Layers: 3, Shape: 32*128</td><td rowspan=1 colspan=1>97.91</td><td rowspan=1 colspan=1>91.17</td><td rowspan=1 colspan=1>94.61</td></tr><tr><td rowspan=1 colspan=1>PubTabNet</td><td rowspan=1 colspan=1>Layers: 4, Shape: 32*128</td><td rowspan=1 colspan=1>98.12</td><td rowspan=1 colspan=1>92.84</td><td rowspan=1 colspan=1>95.53</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 6, Shape: 32*128</td><td rowspan=1 colspan=1>98.11</td><td rowspan=1 colspan=1>92.98</td><td rowspan=1 colspan=1>95.60</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 6, Shape: 128*128</td><td rowspan=1 colspan=1>98.00</td><td rowspan=1 colspan=1>93.32</td><td rowspan=1 colspan=1>95.71</td></tr><tr><td rowspan=4 colspan=1>FinTabNet</td><td rowspan=1 colspan=1>FinTabNet</td><td rowspan=1 colspan=1>Layers: 6, Shape: 32*32</td><td rowspan=1 colspan=1>98.39</td><td rowspan=1 colspan=1>94.57</td><td rowspan=1 colspan=1>96.41</td></tr><tr><td rowspan=1 colspan=1>FinTabNet</td><td rowspan=1 colspan=1>Layers: 6, Shape: 32*128</td><td rowspan=1 colspan=1>98.30</td><td rowspan=1 colspan=1>97.46</td><td rowspan=1 colspan=1>97.88</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 6, Shape: 128*128</td><td rowspan=1 colspan=1>98.31</td><td rowspan=1 colspan=1>97.73</td><td rowspan=1 colspan=1>98.01</td></tr><tr><td rowspan=1 colspan=1>FinTabNet</td><td rowspan=1 colspan=1>Layers: 6, Shape: 128*128</td><td rowspan=1 colspan=1>98.35</td><td rowspan=1 colspan=1>97.74</td><td rowspan=1 colspan=1>98.03</td></tr><tr><td rowspan=4 colspan=1>PubTables-1M</td><td rowspan=1 colspan=1>PubTables-1M</td><td rowspan=1 colspan=1>Layers: 6, Shape: 32*128</td><td rowspan=1 colspan=1>98.19</td><td rowspan=1 colspan=1>92.69</td><td rowspan=1 colspan=1>95.50</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 8, Shape: 32*128</td><td rowspan=1 colspan=1>98.88</td><td rowspan=1 colspan=1>93.34</td><td rowspan=1 colspan=1>96.00</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 6, Shape: 32*128</td><td rowspan=1 colspan=1>98.87</td><td rowspan=1 colspan=1>94.80</td><td rowspan=1 colspan=1>96.75</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>Layers: 6, Shape: 128*128</td><td rowspan=1 colspan=1>98.92</td><td rowspan=1 colspan=1>96.54</td><td rowspan=1 colspan=1>97.68</td></tr></table>

Table 10. Results on different test sets for the SPRINT component of DELTA trained on various datasets. The training set of ’All’ refers to the combined training dataset of all three datasets. The config is dictated by two parameters, mainly the number of decoder layers and the shape (dimensions) of the input image, which is resized in the preprocessing stage. \* indicates that the maximum permissible length of prediction was set to 192 for that experiment, and the length was set to 224 otherwise. All the results are reported on the canonical validation set of PubTabNet [29] and canonical test sets of FinTabNet [29] and PubTables-1M [29]

• If spans are present, the bounding box is extended accordingly, and an HTML td tag with the appropriate rowspan and/or colspan attributes is generated and appended to H.

• If no spans are present, a simple td tag with its bounding box is appended. In both cases, the processed bounding box is stored in the bbox attribute.

Whenever an entry is marked as N, the algorithm closes the current row and begins a new one. After all entries are processed, the final HTML string is completed.

The output consists of the structured HTML string H, which encodes the table with explicit row and column spans that maintain the bounding boxes associated with each logical cell. This design ensures that even complex tables with merged rows/columns are reconstructed faithfully, while retaining both logical order (from SPRINT) and spatial grounding (from TATR).

## 11.4. OCR Ablations

The TSR module of DELTA achieves highly reliable structure predictions, with TEDS-Structure scores exceeding 95% on standard datasets. However, the overall TEDS score is hampered by OCR quality, making OCR the primary performance bottleneck. OCR errors arise from noise, complex layouts, slanted text, and special symbols. Since OCR follows TSR in our pipeline, its choice is critical. To assess this impact, we compare two widely used engines: Tesseract [49] and EasyOCR [13]. Tesseract provides broad multilingual support, modularity, and ease of integration but suffers from lower accuracy on noisy data, lacks GPU acceleration, and is slow at inference. In contrast, EasyOCR is GPU-compatible, nearly three times faster, and consistently more accurate. It integrates seamlessly with detection and structure recognition modules, offers greater control over outputs, and is modular enough to be replaced with CRNN-based or fine-tuned models. Empirically, EasyOCR achieves consistently stronger results (Table 11), improving TEDS scores of almost all the datasets. The weighted average increases by 12.45 p.p. across FinTabNet, PubTab-Net, FinTabNetQA, and TORQUE, which also translates into higher TabVQA accuracy. The slight drop in performance on PubTabNet stems from its relatively clean images with few OCR errors, rather than a limitation of EasyOCR. In this setting, Tesseract and EasyOCR perform nearly identically, with negligible differences in TEDS scores. In contrast, PubTables-1M poses a greater challenge: its large scale makes Tesseract impractically slow for experiments, whereas EasyOCR strikes a balance between speed and accuracy, achieving a reasonable TEDS score of 54.8. Table (12) highlights the latency comparison, underscoring EasyOCR’s superior efficiency. Accordingly, EasyOCR is used as the default OCR module in our DELTA pipeline, though it can be readily replaced with a stronger alternative.

To isolate OCR as the primary performance bottleneck, we prepared Ground Truth (GT) mapped predictions, where the content of each predicted HTML cell was replaced with the corresponding ground-truth content while preserving the predicted structure. Care was taken to ensure accurate row-wise fidelity and cell mapping. This setup is equivalent to DELTA predictions followed by perfect OCR. As shown by the resulting TEDS scores (Table 11) above 90% in most cases. This shows that the content errors are almost entirely attributable to OCR. This study clearly indicates that integrating a stronger OCR module can substantially boost the table reconstruction quality of DELTA, effectively narrowing the problem.

Algorithm 2 Converting OTSL Matrix (logical structure)   
and cell boxes (physical structure) into HTML Sequence.   
1: Input: OTSL matrix M of size $R \times C$ , list of cell   
bounding boxes Cells   
2: Output: HTML table string H, list of structured cells   
S   
3: H ← “<table><tbody>”   
4: for $i = 1  R$ do   
5: Append $^ { 6 6 } < t \tt { r } > ^ { \prime \prime }$ to H   
6: for $j = 1  C$ do   
7: if $M [ i , j ] = { \mathrm { C } }$ then   
8: cell $ C e l l s [ i , j ]$   
9: $( r s , c s ) \gets \mathsf { g e t \_ c e l l } \mathsf { 1 \_ s p a n s } ( M , i , j )$   
10: $\mathbf { i f } r s > 0 \land c s > 0$ then   
11: Extend td to cover row and column   
spans   
12: Append <td rowspan=rs+1   
colspan=cs+1 bbox=cell> to H   
13: else if $r s > 0$ then   
14: Extend td vertically   
15: Append <td rowspan=rs+1   
bbox=cell> to H   
16: else if cs > 0 then   
17: Extend td horizontally   
18: Append <td colspan=cs+1   
bbox=cell> to H   
19: else   
20: Append <td bbox=cell> to H   
21: end if   
22: else if $M [ i , j ] = \mathbb { N }$ then   
23: Append $^ { 6 6 } < / \pm \underline { { { \Upsilon } } } > ^ { , 5 }$ to H   
24: end if   
25: end for   
26: end for   
27: Append “</tbody></table>” to H   
28: return H

## 12. Evaluation Metrics

Table (13) lists the metrics used for evaluation. TEDS and TEDS-S are applied to the table reconstruction task on PubTabNet, PubTables-1M, FinTabNet, and TORQUE; TEDS-S captures structural fidelity, while TEDS score takes into account both structure and content, making them well-suited for measuring both layout accuracy and content alignment. For the TabQA and TabVQA tasks on WTQ, FinTabNetQA, and TORQUE, we report ANLS, EM, and Relieved Accuracy, all of which range from 0 to 100. These metrics are standard in QA benchmarks, reflecting exact correctness (EM), tolerance to minor variations (ANLS), and robustness to different semantic answer formats (Relieved Accuracy). Together, they ensure fair and comprehensive comparisons across diverse methods.

<table><tr><td>Dataset</td><td>Tesseract</td><td>Easy OCR</td><td>GT Mapped</td></tr><tr><td>FinTabNet</td><td>41.5</td><td>55.9</td><td>91.2</td></tr><tr><td>PubTabNet</td><td>54.4</td><td>53.0</td><td>87.8</td></tr><tr><td>FinTabNetQA</td><td>70.0</td><td>84.0</td><td>92.5</td></tr><tr><td>PubTables-1M</td><td></td><td>54.8</td><td>86.8</td></tr><tr><td>TORQUE</td><td>40.8</td><td>63.6</td><td>83.5</td></tr></table>

Table 11. TEDS scores across different OCR modules and ground truth content mapped to the structure of DELTA. Results show consistent improvements on FinTabNet, FinTabNetQA, and TORQUE datasets.
<table><tr><td>OCR Engine</td><td>Average Latency per image (in secs)</td></tr><tr><td>Tesseract</td><td>13.9</td></tr><tr><td>EasyOCR</td><td>4.3</td></tr></table>

Table 12. Average Latency comparison of Tesseract and EasyOCR per table image in seconds calculated on the FinTabNet test set.

## 13. OTSL Ablation

Figure (6) presents qualitative examples illustrating the compactness of the OTSL format relative to HTML. We include two scenarios: one featuring a large table and another involving a large table with a more complex structure. These examples demonstrate that OTSL not only preserves structural fidelity but also provides a significantly more compact representation than HTML. The compactness of OTSL further benefits LLMs by reducing context length, enabling more efficient and accurate table understanding. In our analysis, several input tables that were incorrectly processed in HTML format were correctly interpreted when encoded in OTSL, highlighting its effectiveness. This reduced representation allows larger tables to be processed without exceeding context limits, ultimately contributing to improved overall accuracy. Finally, the handling of mathematical expressions depends on the OCR module applied after structure recognition; therefore, OTSL itself does not impose any inherent limitations on the extraction of mathematical equations.

## 14. Qualitative Results

In this section, we present qualitative results for table reconstruction on FinTabNet and TORQUE datasets, as well as for TabQA and TabVQA tasks.

<table><tr><td>Metric</td><td>Definition</td></tr><tr><td>Tree Edit Distance-based Similarity (TEDS)</td><td>Measures the similarity between predicted and ground truth HTML table struc- tures using tree edit distance. It evaluates both the structure and content of ta- bles, which we use to evaluate DELTA pipeline.</td></tr><tr><td>TEDS-Structure (TEDS-S)</td><td>A TEDS variant that evaluates only the HTML tag sequence, ignoring content. We use it to assess DELTA&#x27;s table structure recognition.</td></tr><tr><td>Average Normalized Levenshtein Similarity (ANLS)</td><td>Measures textual similarity between predicted and ground truth answers using the normalised Levenshtein distance, providing partial credit for near matches. We use this metric to evaluate TARQA on the WTQ dataset.</td></tr><tr><td>Exact Match (EM)</td><td>A strict binary metric that returns 1 if the predicted answer matches the ground truth exactly, and 0 otherwise.</td></tr><tr><td>Relieved-Accuracy</td><td>This metric considers predictions correct if any normalised form matches the ground truth, ignoring units and formatting. It emphasizes semantic equiva- lence while evaluating DELTA + TARQA on the FinTabNetQA dataset for the TabVQA task.</td></tr></table>

Table 13. Overview of metrics used for evaluating DELTA and TARQA.  
![](images/bce33a66dafccc7568ac25feb1d14732e0a4b88bfc6796cf87c469664fe39b23.jpg)

(a) HTML Character Count: 438, OTSL Character Count: 409  
![](images/8b1273909c92d790057c0f8eaf3b0c13b708abbe5d9ad015883d22fe39f25179.jpg)  
(b) HTML Character Count: 1607, OTSL Character Count: 1430  
Figure 6. Qualitative examples from the TORQUE dataset illustrating the compactness of the OTSL format in terms of Character Counts from the input image.

## 14.1. Table Reconstruction

Figure (7a) demonstrates a successful case of DELTA, where the table image has a clean layout with multiple columns and well-separated numeric content. This results in perfect structural accuracy (TEDS-S = 100) and a high overall score $( \mathrm { T E D S } = 8 0 . 9 8 )$ . Similarly, Figure (7b) shows another positive example, with a clear two-column structure and well-defined row–column divisions, yielding TEDS-S = 100 and TEDS = 88.11. In contrast, Figure (8a) highlights a failure case arising from OCR extraction errors: while the structural score remains high $( \mathrm { T E D S - S } = 9 5 . 1 5 )$ , the overall content fidelity is poor (TEDS = 19.35). Figure (8b) presents another challenging example, where low image resolution and a borderless table with bullet points hinder accurate reconstruction, leading to low scores (TEDS-S = 26.09, TEDS = 19.95).

Similarly, Figure (9) presents qualitative examples of table reconstruction results on the TORQUE dataset. Figure (9a) demonstrates a successful reconstruction, reflected in a high TEDS-S and TEDS score of 97.22 and 96.06, respectively. In contrast, Figure (9b) highlights a failure case where slanted text and low-resolution input hinder accurate prediction, resulting in a significantly lower TEDS-S and TEDS score of 44.44 and 14.60, respectively.

## 14.2. Table Question Answering

Figure (10a) and Figure (10b) present qualitative examples from the WTQ dataset for TabQA tasks, where the answers predicted by the proposed TARQA framework are consistent with the ground truth. In contrast, Figure (11a) and Figure (11b) illustrate cases where the predictions deviate from the ground truth. For Figure (11a), the model outputs a value occurring immediately after the correct answer, indicating a limitation in its contextual understanding. For Figure (11b), the prediction is incorrect because the question is of a comparative type, for which the model has not been explicitly fine-tuned.

## 14.3. Table Visual Question Answering

Figure (12a) and Figure (12b) show illustrative Tab-VQA examples from the FintabnetQA dataset using the DELTA+TARQA pipeline. These examples highlight that even with difficult, borderless tables, the framework can accurately predict both numerical and textual answers, aligning with the ground truth. On the other hand, Figure (13a) and Figure (13b) showcase instances of divergence. In Figure (13a), the model becomes confused due to the complexity of the question type, while in Figure (13b), poor image quality leads to incorrect answer. These cases highlight both the strengths and current limitations of the approach, while also indicating clear directions, improved OCR, and enhanced reasoning for future enhancements.

![](images/cf1dfeeda24bea4e7285bb10848b6844aa5f834f1c8da0fb78925e5cab1670c9.jpg)

## Rendered HTML image

![](images/e8c5105ebf021baf3006fc89bbc5ef08dac3302d1aac978c5cbda29ee2a0d684.jpg)

![](images/2288ccc4a5a8f64747bfeba58127e25353b8dd1fe7891e0320d92fc97de8ea1b.jpg)

## Input Table Image

![](images/23ac4f66aa77d805f2b9346ef49ad90acd72c2fde754d9f5fb946f41c8d0f5d3.jpg)

## Rendered HTML image

![](images/94b481fb93409871bfab4055de7cb0669cf9a41d8f6af39088f27c93eafcb03d.jpg)  
(b) TEDS-S = 100 and TEDS = 88.11.

![](images/15ef001cb0b8abe23fca761fe16848c400b523357a9362c23e11379df1bea6cf.jpg)  
Figure 7. Qualitative examples from the FinTabNet dataset illustrating table reconstruction outputs produced by DELTA framework.

## Input Table Image

DELTA Output  
![](images/91bb9f11703e6872f35b06c1f027ff21eb6b71b1d67a2c3d4c7d7e5c33d6ecdc.jpg)

## Rendered HTML image

![](images/112785568348dc936f898d1ae1ee4ebf7d0b089fc1265c0761f900730a4da65a.jpg)  
(a) TEDS-S = 95.15 and TEDS = 19.35

![](images/b8426cccd512951f98af07a65b2b6ac63afd6e19c9ccc253b028ae2cd49a1efd.jpg)

## Input Table Image

![](images/b521908db126a8d83ac4bb4af170823ab140d21e8e7fa34e9158a5208d48b8e4.jpg)  
(b) TEDS-S = 26.09 and TEDS = 19.95  
Figure 8. Inaccurate qualitative examples from the FinTabNet dataset illustrating table reconstruction outputs generated by DELTA.

## Rendered HTML image

Frodutts und Senutes

AIR MILES Reward Pgtam

Short-term Loyahty Progtn

What was the return on assets in plans for the pension cost in 2011 technology setvices DJta tvices \~Suategy ad aalytical stvices Traditicn 1 and digital mztketing

el[ss AEeTe Pracessing Services New account pocesing \~Bill prucessing RcmllnCe processing Customet cilt

Makctng Scwviccs

## DELTA Output

<fcel> AIR\_MILES\_Reward\_Pgtam <nl> <ecel> <fcel> Short-<sub>11</sub> <sub><fcel></sub> <sub>2010</sub> <sub><nl></sub> <sub><fcel></sub> <sub>Pension\_funded\_status:</sub> <sub><lcel></sub> <sub><lcel></sub> <sub><lcel></sub> <sub><nl></sub> <sub><fcel></sub> <sub>Discount\_rate</sub> <sub><fcel></sub> <sub>3.65%</sub>term\_Loyahty\_Progtn <nl> <ecel> <fcel>

<table><tr><td></td><td>RE 2017-18 ()</td><td>BE 2018-19 f ()</td><td>uftadia</td></tr><tr><td> R</td><td>14478</td><td>15799</td><td>9.124188</td></tr><tr><td>y明(可市)市 faT </td><td>330</td><td>600</td><td>81.81818</td></tr><tr><td>a    </td><td>9466</td><td>10317</td><td>8.99007</td></tr><tr><td>ifr i epf  a</td><td>87319</td><td>89210</td><td>2.165623</td></tr><tr><td>far steit sa</td><td>64318</td><td>53469</td><td>-16.8678</td></tr><tr><td>Re 3E54 H</td><td>2543</td><td>4086</td><td>60.67637</td></tr><tr><td>TaR fa</td><td>9786</td><td>16 9 86</td><td>73.57449</td></tr><tr><td>S</td><td>80000</td><td>93440</td><td>16.8</td></tr><tr><td>    </td><td>15193</td><td>399 37</td><td>162.8645</td></tr><tr><td>  R  </td><td>59279</td><td>62000</td><td>4.5 9 0158</td></tr><tr><td> HE</td><td>3165</td><td>4042</td><td>27.70932</td></tr><tr><td>AR H</td><td>11428</td><td>11294</td><td>-1.17256</td></tr><tr><td></td><td>357305</td><td>401180</td><td>12,2794 ait:   2018</td></tr><tr><td colspan="4"></td></tr></table>

![](images/8140d4b2ab2ed2905c7ef2a88fd6f06dda09cd89ed61ae522e5df5bf7126c708.jpg)  
(b) Depicts a challenging case with slanted text and low-resolution input, leading to notable differences between the predicted and input tables and a lowe TEDS-S and TEDS score of 44.44 and 14.60, respectively.  
Figure 9. Qualitative results of the proposed DELTA framework on a TORQUE sample for the table reconstruction task.

<table><tr><td rowspan=15 colspan=8>Table Image1                                  164,1192Koodi        Robin          117,1263                                  81,7254Chillaa       Robin          73,439521            Adele          44,2976Yhdestä puusta Jukka Poika    42,4297                Jesse Kaikuranta38,9858                Chisu          31,5419                                  29,08010Hunningolla  Erin            27,655</td><td rowspan=1 colspan=5>HTML String from WTQ</td><td rowspan=7 colspan=1>Input QuestionWhich album has thehighest number of salesbut doesn&#x27;t have adesignated artist?</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>Album</td><td rowspan=2 colspan=2>Artist(s)</td><td rowspan=2 colspan=1>Sales</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=8 colspan=5>&lt;table border=\&quot;1\&quot; cellspacing=\&quot;0\&quot; cellpadding=\&quot;5\&quot;&gt;\n &lt;thead&gt;\n &lt;tr&gt;\n &lt;th&gt;&lt;/th&gt;\n&lt;th&gt;Album&lt;/th&gt;\n &lt;th&gt;Artist(s)&lt;/th&gt;\n &lt;th&gt;Sales&lt;/th&gt;\n &lt;/tr&gt;\n &lt;/thead&gt;\n &lt;tbodv&gt;\n &lt;tr&gt;\n &lt;td&gt;1&lt;/td&gt;\r&lt;td&gt;Vain elämää&lt;/td&gt;\n &lt;td&gt;various artists&lt;/td&gt;\n &lt;td&gt;164,119&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2&lt;/td&gt;\n&lt;td&gt;Koodi&lt;/td&gt;\n &lt;td&gt;Robin&lt;/td&gt;\n &lt;td&gt;117,126&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;3&lt;/td&gt;\n &lt;td&gt;Vainelämää&lt;/td&gt;\n &lt;td&gt;various artists&lt;/td&gt;\n &lt;td&gt;81.725&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;4&lt;/td&gt;\n&lt;td&gt;Chillaa&lt;/td&gt;\n &lt;td&gt;Robin&lt;/td&gt;\n &lt;td&gt;73,439&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;5&lt;/td&gt;\n &lt;td&gt;21&lt;/td&gt;\n&lt;td&gt;Adele&lt;/td&gt;\n &lt;td&gt;44,297&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;6&lt;/td&gt;\n &lt;td&gt;Yhdestä puusta&lt;/td&gt;\n &lt;td&gt;JukkaPoika&lt;/td&gt;\n &lt;td&gt;42,429&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;7&lt;/td&gt;\n &lt;td&gt;Vie mut kotiin&lt;/td&gt;\n &lt;td&gt;JesseKaikuranta&lt;/td&gt;\n &lt;td&gt;38,985&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;8&lt;/td&gt;\n &lt;td&gt;Kun valaistun&lt;/td&gt;\n&lt;td&gt;Chisu&lt;/td&gt;\n &lt;td&gt;31,541&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;9&lt;/td&gt;\n &lt;td&gt;Joululauluja&lt;/td&gt;\n &lt;td&gt;JuhaTapio&lt;/td&gt;\n &lt;td&gt;29,080&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;10&lt;/td&gt;\n &lt;td&gt;Hunningolla&lt;/td&gt;\n &lt;td&gt;Erin&lt;/td&gt;\n&lt;td&gt;27,655&lt;/td&gt;\n &lt;/tr&gt;\n &lt;/tbody&gt;\n&lt;/table&gt;</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Vain elämää</td><td rowspan=1 colspan=2>various artists</td><td rowspan=1 colspan=1>164,119</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=2>Koodi</td><td rowspan=1 colspan=2>Robin</td><td rowspan=1 colspan=1>117,126</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=2>Vain elämää</td><td rowspan=1 colspan=2>various artists</td><td rowspan=1 colspan=1>81,725</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>td>Adele</td></td><td rowspan=1 colspan=2>ele</td>|n <td>44,29</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=2>Chillaa</td><td rowspan=1 colspan=2>Robin</td><td rowspan=1 colspan=1>73,439</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=2>Adele</td><td rowspan=1 colspan=1>44,297</td><td></td></tr><tr><td rowspan=3 colspan=1>6</td><td rowspan=3 colspan=2>Yhdestä puusta</td><td rowspan=3 colspan=2>Jukka Poika</td><td rowspan=3 colspan=1>42,429</td><td></td></tr><tr><td rowspan=4 colspan=1>Answer from TARQAvain elämää</td></tr><tr><td rowspan=1 colspan=5>Input OTSL Sequence to TARQA</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=2>Vie mut kotiin</td><td rowspan=1 colspan=2>Jesse Kaikuranta</td><td rowspan=1 colspan=1>38,985</td><td rowspan=4 colspan=5>&lt;otsl&gt;&lt;ecel&gt;&lt;fcel&gt;Album&lt;fcel&gt;Artist(s)&lt;fcel&gt;Sales &lt;nl&gt;&lt;fcel&gt;1&lt;fcel&gt;Vain_elämää&lt;fcel&gt;various_artists &lt;fcel&gt;164,119 &lt;nl&gt;&lt;fcel&gt;2 &lt;fcel&gt;Koodi &lt;fcel&gt;Robin &lt;fcel&gt;117,126 &lt;nl&gt; &lt;fcel&gt; 3&lt;fcel&gt;Vain_elämää&lt;fcel&gt;various_artists &lt;fcel&gt;81,725 &lt;nl&gt;&lt;fcel&gt;4&lt;fcel&gt;Chillaa &lt;fcel&gt;Robin &lt;fcel&gt;73,439&lt;n|&gt; &lt;fcel&gt; 5 &lt;fcel&gt;21 &lt;fcel&gt;Adele&lt;fcel&gt;44,297 &lt;n|&gt; &lt;fcel&gt;6 &lt;fcel&gt; Yhdestä_puusta &lt;fcel&gt;Jukka_Poika &lt;fcel&gt; 42,429 &lt;nl&gt;&lt;fcel&gt;7 &lt;fcel&gt; Vie_mut_kotiin &lt;fcel&gt; Jesse_Kaikuranta &lt;fcel&gt;38,985&lt;nl&gt; &lt;fcel&gt; 8 &lt;fcel&gt; Kun valaistun &lt;fcel&gt; Chisu &lt;fcel&gt;31,541 &lt;nl&gt; &lt;fcel&gt; 9 &lt;fcel&gt; Joululauluja &lt;fcel&gt;Juha_Tapio &lt;fcel&gt;29,080&lt;n|&gt; &lt;fcel&gt;10 &lt;fcel&gt;Hunningolla&lt;fcel&gt;Erin &lt;fcel&gt;27,655 &lt;nl&gt;&lt;/otsl&gt;</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=2>Kun valaistun</td><td rowspan=1 colspan=2>Chisu</td><td rowspan=1 colspan=1>31,541</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=2>Joululauluja</td><td rowspan=1 colspan=2>Juha Tapio</td><td rowspan=1 colspan=1>29,080</td><td rowspan=2 colspan=1>Groundtruth Answervain elämää</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=2>Hunningolla</td><td rowspan=1 colspan=2>Erin</td><td rowspan=1 colspan=1>27,655</td></tr><tr><td rowspan=1 colspan=14>(a) The predicted answers align with the ground truth</td></tr><tr><td rowspan=1 colspan=7>Table Image</td><td></td><td rowspan=1 colspan=5>HTML String from WTQ</td><td rowspan=7 colspan=1>Input QuestionIn whichcompetition didhopley finish first?</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=7 colspan=5>&lt;table border=\&quot;1\&quot; cellspacing=\&quot;0\&quot; cellpadding=\&quot;5\&quot;&gt;\n &lt;thead&gt;\n &lt;tr&gt;\n &lt;th&gt;Year&lt;/th&gt;\n &lt;th&gt;Competition&lt;/th&gt;\n&lt;th&gt;Venue&lt;/th&gt;\n &lt;th&gt;Position&lt;/th&gt;\n &lt;th&gt;Event&lt;/th&gt;\n &lt;th&gt;Notes&lt;/th&gt;\n &lt;/tr&gt;\n &lt;/thead&gt;\n &lt;tbody&gt;\n &lt;tr&gt;\n&lt;td&gt;2000&lt;/td&gt;\n &lt;td&gt;World Junior Championships&lt;/td&gt;\n &lt;td&gt;Santiago, Chile&lt;/td&gt;\n &lt;td&gt;1st&lt;/td&gt;\n &lt;td&gt;Discusthrow&lt;/td&gt;\n &lt;td&gt;59.51 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2003&lt;/td&gt;\n &lt;td&gt;All-Africa Games&lt;/td&gt;\n &lt;td&gt;Abuja, Nigeria&lt;/td&gt;\n&lt;td&gt;5th&lt;/td&gt;\n &lt;td&gt;Shot put&lt;/td&gt;\n &lt;td&gt;17.76 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2003&lt;/td&gt;\n &lt;td&gt;All-Africa Games&lt;/td&gt;\n&lt;td&gt;Abuja, Nigeria&lt;/td&gt;\n &lt;td&gt;2nd&lt;/td&gt;\n &lt;td&gt;Discus throw&lt;/td&gt;\n &lt;td&gt;62.86 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2004&lt;/td&gt;\n&lt;td&gt;African Championships&lt;/td&gt;\n &lt;td&gt;Brazzaville, Republic of the Congo&lt;/td&gt;\n &lt;td&gt;2nd&lt;/td&gt;\n &lt;td&gt;Discus throw&lt;/td&gt;\n&lt;td&gt;63.50 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2004&lt;/td&gt;\n &lt;td&gt;0lympic Games&lt;/td&gt;\n &lt;td&gt;Athens, Greece&lt;/td&gt;\n &lt;td&gt;8th&lt;/td&gt;\r&lt;td&gt;Discus throw&lt;/td&gt;\n &lt;td&gt;62.58 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2006&lt;/td&gt;\n &lt;td&gt;Commonwealth Games&lt;/td&gt;\n&lt;td&gt;Melbourne, Australia&lt;/td&gt;\n &lt;td&gt;7th&lt;/td&gt;\n &lt;td&gt;Shot put&lt;/td&gt;\n &lt;td&gt;18.44 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2006&lt;/td&gt;\n&lt;td&gt;Commonwealth Games&lt;/td&gt;\n &lt;td&gt;Melbourne, Australia&lt;/td&gt;\n &lt;td&gt;4th&lt;/td&gt;\n &lt;td&gt;Discus throw&lt;/td&gt;\n &lt;td&gt;60.99m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2007&lt;/td&gt;\n &lt;td&gt;All-Africa Games&lt;/td&gt;\n &lt;td&gt;Algiers. Algeria&lt;/td&gt;\n &lt;td&gt;3rd&lt;/td&gt;\n&lt;td&gt;Discus throw&lt;/td&gt;\n &lt;td&gt;57.79 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;tr&gt;\n &lt;td&gt;2008&lt;/td&gt;\n &lt;td&gt;African Championships&lt;/td&gt;\n &lt;td&gt;AddisAbaba, Ethiopia&lt;/td&gt;\n &lt;td&gt;2nd&lt;/td&gt;\n &lt;td&gt;Discus throw&lt;/td&gt;\n &lt;td&gt;56.98 m&lt;/td&gt;\n &lt;/tr&gt;\n &lt;/tbody&gt;\n&lt;/table&gt;</td></tr><tr><td rowspan=1 colspan=1>2000WorldCh</td><td></td><td rowspan=1 colspan=1>Junior      Saampionships</td><td rowspan=1 colspan=2>ntiago, Chile       1</td><td rowspan=1 colspan=1>st</td><td rowspan=1 colspan=2>Discus 59.51throw  m</td></tr><tr><td rowspan=1 colspan=1>2003Al</td><td></td><td rowspan=1 colspan=1>l-Africa Games    A</td><td rowspan=1 colspan=2>buja, Nigeria       5</td><td rowspan=1 colspan=1>th Shot</td><td rowspan=1 colspan=2>17.76mput</td></tr><tr><td rowspan=1 colspan=1>2003Al</td><td></td><td rowspan=1 colspan=1>l-Africa Games    A</td><td rowspan=1 colspan=2>buja, Nigeria       2</td><td rowspan=1 colspan=1>nd</td><td rowspan=1 colspan=2>Discus 62.86throw  m</td></tr><tr><td rowspan=1 colspan=1>2004Af</td><td></td><td rowspan=1 colspan=1>rican Championships</td><td rowspan=1 colspan=2>Brazzavill, Republic of theCongo           2</td><td rowspan=1 colspan=1>nd th</td><td rowspan=1 colspan=2>Discus 63.50row  m</td></tr><tr><td rowspan=1 colspan=1>2004 O</td><td></td><td rowspan=1 colspan=1>lympic Games     A</td><td rowspan=1 colspan=2>thens, Greece       8</td><td rowspan=1 colspan=1>thD</td><td rowspan=1 colspan=2>62.58hsu m</td></tr><tr><td rowspan=1 colspan=1>2006C</td><td></td><td rowspan=1 colspan=1>ommonwealth Games</td><td rowspan=1 colspan=2>Melbourne, Australia    7</td><td rowspan=1 colspan=1>th Shot</td><td rowspan=1 colspan=2>put 18.44m</td><td></td></tr><tr><td rowspan=3 colspan=1>2006C</td><td></td><td rowspan=3 colspan=1>ommonwealth Games</td><td rowspan=3 colspan=2>Melbourne, Australia    4</td><td rowspan=3 colspan=1>th Dt</td><td rowspan=3 colspan=2>iscus 60.99hrow  m</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=5 colspan=1>Answer from TARQAWorld JuniorChampionships</td></tr><tr><td></td><td rowspan=1 colspan=5>Input OTSL Sequence to TARQA</td></tr><tr><td rowspan=1 colspan=1>2007Al</td><td></td><td rowspan=1 colspan=1>l-Africa Games</td><td rowspan=1 colspan=2>Algiers, Algeria      3</td><td rowspan=1 colspan=1>rd t</td><td rowspan=1 colspan=2>Discus 57.79hrow</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>2008Af</td><td></td><td rowspan=2 colspan=1>rican Championships</td><td rowspan=2 colspan=2>Addis Ababa, Ethiopia   2</td><td rowspan=2 colspan=1>nd D</td><td rowspan=2 colspan=2>iscus 56.988throw</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=2 colspan=5>&lt;otsl&gt; &lt;fcel&gt; Year &lt;fcel&gt; Competition &lt;fcel&gt; Venue &lt;fcel&gt; Position &lt;fcel&gt;Event &lt;fcel&gt;Notes &lt;nl&gt; &lt;fcel&gt; 2000 &lt;fcel&gt;World_Junior_Championships &lt;fcel&gt;Santiago,_Chile &lt;fcel&gt;1st &lt;fcel&gt;Discus_throw &lt;fcel&gt;59.51_m &lt;nl&gt;&lt;fcel&gt;2003&lt;fcel&gt;All-Africa_Games&lt;fcel&gt; Abuja,_Nigeria &lt;fcel&gt;5th &lt;fcel&gt;Shot_put &lt;fcel&gt;17.76_m &lt;nl&gt; &lt;fcel&gt;2003 &lt;fcel&gt;All-Africa_Games &lt;fcel&gt; Abuja,_Nigeria &lt;fcel&gt; 2nd&lt;fcel&gt; Discus_throw &lt;fcel&gt; 62.86_m &lt;nl&gt; &lt;fcel&gt; 2004 &lt;fcel&gt; African_Championships &lt;fcel&gt; Brazzaville,_Republic_of_the_Congo &lt;fcel&gt; 2nd&lt;fcel&gt; Discus_throw &lt;fcel&gt; 63.50_m &lt;nl&gt; &lt;fcel&gt;2004 &lt;fcel&gt; Olympic_Games &lt;fcel&gt; Athens,_Greece &lt;fcel&gt;8th &lt;fcel&gt; Discus_throw &lt;fcel&gt;62.58_m&lt;nl&gt; &lt;fcel&gt; 2006 &lt;fcel&gt; Commonwealth_Games &lt;fcel&gt;Melbourne,_Australia &lt;fcel&gt;7th &lt;fcel&gt; Shot_put&lt;fcel&gt;18.44_m&lt;nl&gt;&lt;fcel&gt;2006 &lt;fcel&gt; Commonwealth_Games &lt;fcel&gt;Melbourne,_Australia &lt;fcel&gt; 4th &lt;fcel&gt;Discus_throw &lt;fcel&gt;60.99_m &lt;nl&gt; &lt;fcel&gt; 2007 &lt;fcel&gt; All-Africa_Games&lt;fcel&gt; Algiers,_Algeria &lt;fcel&gt;3rd&lt;fcel&gt;Discus_throw &lt;fcel&gt;57.79_m &lt;nl&gt;&lt;fcel&gt;2008 &lt;fcel&gt; African_Championships &lt;fcel&gt;Addis_Ababa,_Ethiopia &lt;fcel&gt; 2nd &lt;fcel&gt; Discus_throw &lt;fcel&gt; 56.98_m &lt;nl&gt; &lt;/otsl&gt;</td></tr><tr><td rowspan=1 colspan=7></td><td></td><td rowspan=1 colspan=1>Groundtruth AnswerWorld JuniorChampionships</td></tr></table>

(b) The predicted answers align with the ground truth  
Figure 10. Qualitative examples from the WTQ dataset for TabQA tasks, illustrating answers generated TARQA-OTSL.

![](images/7146489489986a082bba4b950cdc126905185c33d5e7d6ca3227020765abe32c.jpg)  
(b) The predictions differ from the ground truth  
Figure 11. Qualitative examples from the WTQ dataset for TabQA tasks, illustrating answers generated by TARQA-OTSL.

![](images/249bfe7f684a0a29b0762f8fbbf0681d69a0e996b8ad11dfdf66001f29b1f1c9.jpg)

(a) The predicted answers coincide with the ground truth  
![](images/16263ba87f79276b6e1bedea82c9a688f660689e2d13675b5e9388a542ebad82.jpg)  
(b) The predicted answers coincide with the ground truth

Figure 12. Illustrative TabVQA examples from the FintabnetQA dataset showcasing the performance of DELTA+TARQA-OTSL.

<table><tr><td colspan="6">Input Table Image</td></tr><tr><td colspan="3">Computed tax at statutory tax rate State income taxes, net of federal tax benefit (1) . . . . . . . Non-deductible expenses and other (2) Foreign taxes</td><td>2014 $297 22 8 (17) $310</td><td>Year ended December 31, 2013 $212 15 8 (17) $218</td><td>2012 $31 5 (8) (15) $13</td></tr><tr><td>Total</td><td colspan="2">Input Question What was the total for the year ended December 31,</td><td colspan="2">Answer from TARQA $1,080,544.00</td><td></td></tr><tr><td></td><td>2013?</td><td colspan="3">Groundtruth Answer $218</td><td></td></tr></table>

(a) The ground truth and predicted answers are diverging from each other.

![](images/68132b7175d4ff3d721ef518c615ee7973f14d2a2563175e4526a8cf98ec8071.jpg)  
(b) The ground truth and predicted answers are diverging from each other.  
Figure 13. Illustrative TabVQA examples from the FintabnetQA dataset showcasing the performance of DELTA+TARQA-OTSL.