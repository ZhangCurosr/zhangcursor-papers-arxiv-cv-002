# HANS: A Handwritten Answer Sheet Dataset for Noisy Hybrid Document Parsing

Xiazhen Wu, Wansong Qin, Yangbin Zheng, Liangda Fang, Zhan Li, Xiujie Huang, Liushen Zhou, and Quanlong Guan

Jinan University, Guangzhou, China wuxiazhen@stu2024.jnu.edu.cn {songbai,zybbyz}@stu2025.jnu.edu.cn {fangld,lizhan,t\_xiujie,crystal,gql}@jnu.edu.cn

Abstract. Intelligent grading and automated scoring technologies constitute critical infrastructure for smart education. However, existing document parsing and handwriting recognition benchmarks are predominantly designed for well-structured printed documents or isolated mathematical expressions, lacking datasets that capture the complex characteristics inherent to student answer sheets, including multi-line derivation processes, heterogeneous mixtures of text and mathematical formulae, and noise artifacts such as strikethroughs. To address this gap, we introduce HANS, the first dataset explicitly constructed for real-world educational scenarios, encompassing mathematical expressions, natural language text, hand-drawn tables, and diverse noise patterns including corrections and deletions, accompanied by fine-grained annotations that establish a reliable foundation for robust recognition research. Building upon HANS, we propose NA-GOT, an end-to-end framework that achieves two-stage noise suppression through a lightweight noise suppression module operating at the feature level, complemented by a noiseaware attention mechanism incorporated into the decoding stage. Experimental results demonstrate that HANS poses substantial challenges to existing methods, while NA-GOT achieves significant improvements in both accuracy and stability for answer process recognition. The dataset will be made publicly available upon publication.

Keywords: Smart education · Handwritten document parsing · HANS dataset

## 1 Introduction

Intelligent grading and automated scoring technologies are increasingly emerging as pivotal driving forces in the advancement of smart education. In contrast to outcome-oriented evaluation paradigms that focus exclusively on final answers, modern pedagogical practice, particularly in mathematics education, places greater emphasis on understanding and diagnosing students’ problemsolving thought processes. This demands that intelligent systems be capable of recognizing the derivation steps and natural language explanations written by students on answer sheets, thereby enabling step-wise scoring, error cause analysis, and personalized feedback. Consequently, the accurate recognition of students’ handwritten problem-solving processes constitutes a critical component of intelligent grading systems, as its accuracy and robustness directly determine the viability of downstream scoring and diagnostic tasks.

![](images/1db84e05a882c84d7c473f6242ecc108dab44c0e3f96244d6c9987d136dc7dfa.jpg)  
Fig. 1: Illustration of representative characteristics of the HANS dataset. HANS is a real-world student handwritten mathematics answer sheet dataset, encompassing the dual challenges of complex formula-text interleaving (blue bounding boxes) and correction noise artifacts (red bounding boxes).

Recent years have witnessed substantial progress in document parsing and handwriting recognition. Document parsing methods are predominantly designed for well-structured printed documents, with representative approaches including multi-stage pipeline [8] processing and end-to-end vision-language models [10,15,16,20,23], exemplified by benchmark datasets such as OmniDocBench [13] and FOX [11]. Concurrently, research on handwriting recognition using classical benchmarks — including IAM [12], CROHME [24], and ICDAR Math — has driven an architectural evolution from CNN and RNN-CTC paradigms toward Transformer-based [17] encoders with Seq2Seq decoders and unified visionlanguage modeling frameworks.

Nevertheless, existing benchmarks exhibit several critical limitations. 1) Prevailing datasets predominantly focus on printed academic documents, neglecting real-world handwritten document types such as student answer sheets. 2) There is a systematic absence of research on handwriting noise: most benchmarks assume high document quality and lack systematic annotation and targeted evaluation of common noise artifacts such as strikethroughs, erasures, and overwriting. 3) Existing handwriting datasets are confined to either pure text lines or isolated mathematical expressions; in particular, handwritten mathematical expression recognition has largely been restricted to single-line expressions, with limited investigation into continuous multi-line writing scenarios involving interleaved text and formulae.

To address these limitations, we introduce HANS, the first dataset explicitly constructed for real-world educational scenarios. HANS is sourced from scanned student answer sheets collected during eighth-grade mathematics quality assessment examinations conducted over three semesters in a municipal school district, specifically from the open-ended mathematical problem-solving sections. To protect student privacy, all personally identifiable information was cropped prior to data release and use. Unlike prior datasets comprising pure text or isolated formulae, the student problem-solving processes in HANS encompass mathematical expressions, natural language text, hand-drawn tables, and manual strikethroughs, faithfully reflecting the typical interplay between mixed handwriting and structured writing characteristic of authentic examination contexts, as shown in Fig. 1 . More critically, we provide fine-grained annotations across distinct writing modalities within the problem-solving process, including content-level annotations for expressions, text, and tables, as well as spatial annotations for noise elements such as strikethroughs and deletions. These annotations furnish a supervisory foundation for explicitly distinguishing noise from valid handwriting, thereby supporting subsequent robust modeling and analysis.

Building upon the noise annotations provided by HANS, we further propose NA-GOT, an end-to-end noise-aware recognition framework extending the GOT [21] architecture. The proposed framework suppresses noise interference through a two-stage mechanism. In the first stage, we design a Lightweight Noise Suppression Module (LNSM) that attenuates noise at the feature level. In the second stage, we introduce a noise-aware attention mechanism that reduces attention weights assigned to noise-bearing regions via learned noise bias terms. Experimental results demonstrate that incorporating the noise-aware mechanism yields significant improvements in both recognition accuracy and output stability for answer sheet problem-solving process recognition.

In summary, the contributions of this paper are as follows:

1. We construct HANS, a real-world educational answer sheet dataset encompassing mixed handwritten problem-solving processes with heterogeneous content types including mathematical expressions, natural language text, and tables. Fine-grained annotations are provided, with particular emphasis on spatial bounding box annotations for real-world noise artifacts such as strikethroughs and deletions, establishing a foundation for robust recognition research.

2. We propose NA-GOT, an end-to-end noise-aware recognition framework that achieves two-stage noise suppression through a Lightweight Noise Suppression Module(LNSM) and a noise-aware attention mechanism.

3. Extensive experimental results demonstrate that the HANS dataset presents substantial challenges to current state-of-the-art methods, and validate that the proposed approach significantly outperforms existing methods.

## 2 Related Work

## 2.1 Document Parsing and Handwriting Recognition Datasets

In the domain of document parsing, datasets such as DocLayNet [14] and OmniDocBench [13] provide important benchmarks for document layout segmentation and multi-task learning, with particular significance for layout element detection and segmentation. Concurrently, datasets including FUNSD [7] and FOX have advanced the parsing of form and handwritten documents, encompassing diverse components such as tables, text, and formulae while extending support to more complex document structures. The data distributions of these benchmarks are predominantly composed of printed text, with relatively limited representation of handwritten content, and systematic investigation of noise-related challenges remains largely absent. Existing handwriting recognition datasets, such as IAM, CROHME, and MathWriting [5] , primarily focus on ofline handwriting recognition and mathematical formula transcription tasks. Although substantial progress has been achieved in handwritten text and mathematical expression recognition, these datasets continue to sufer from limited diversity in writing modalities.

## 2.2 Document Parsing and Handwriting Recognition Models

In the area of generative document understanding, models such as Nougat [2] and GOT-OCR2.0 realize end-to-end conversion from document images to structured text via visual Transformer architectures, supporting joint modeling of heterogeneous content types. LayoutLM [25] achieves strong performance through joint modeling of layout and content by integrating visual and textual information within a unified framework. In the field of handwriting recognition, TrOCR [9] adopts an architecture comprising an image Transformer encoder coupled with a text Transformer decoder, enabling eficient end-to-end handwritten text transcription. In the domain of handwritten mathematical expression recognition, works such as TAMER introduce tree-structure-aware modules to jointly optimize sequence prediction and tree structure prediction tasks, thereby improving model generalization over complex mathematical expression structures and demonstrating strong adaptability — particularly for the recognition of intricate formulae and expressions.

Despite the remarkable progress achieved in existing document parsing and handwriting recognition techniques, current methodologies remain inadequate in addressing the noise-corrupted and structurally complex handwriting characteristics inherent to educational answer sheets. Consequently, future research should build upon existing datasets and methods to develop more specialized datasets and more adaptive model frameworks, with the aim of enhancing robustness and accuracy in the educational answer sheet recognition task.

Table 1: Comparison of HANS against existing datasets.HS denotes the number of handwritten samples,and Level indicates the sample granularity(page or line). HME, HT, and NA indicate the presence of handwritten mathematical expressions,handwritten text, and noise annotations,respectively.
<table><tr><td>Dataset</td><td>HS</td><td>Level</td><td>HME</td><td>HT</td><td>NA</td></tr><tr><td>OmniDocBench 116</td><td></td><td>Pages</td><td>√</td><td>√</td><td>x</td></tr><tr><td>CC-OCR [26]</td><td>200</td><td>Pages</td><td>√</td><td>x</td><td>x</td></tr><tr><td>FOX</td><td>0</td><td>Pages</td><td>x</td><td>x</td><td>x</td></tr><tr><td>IAM</td><td>1539</td><td>Lines</td><td>x</td><td>√</td><td>x</td></tr><tr><td>CROHME</td><td>8836</td><td>Lines</td><td>√</td><td>x</td><td>x</td></tr><tr><td>HANS(ours)</td><td>5213</td><td>Pages</td><td>√</td><td>√</td><td>√</td></tr></table>

## 3 HANS Dataset

## 3.1 Dataset Construction and Statistics

We construct HANS, a dataset designed for real-world educational answer sheet scenarios. The data are sourced from eighth-grade mathematics quality assessment examinations conducted over three semesters in a municipal middle school district in China, specifically from the open-ended problem-solving sections comprising six mathematical questions per examination, drawn from scanned student answer sheets, as shown in Tab. 1. The dataset contains 5,213 answer question images, covering 745 students across 18 distinct problem types. To protect student privacy, all personally identifiable information was cropped prior to data release and use.

The writing patterns in HANS exhibit considerable complexity. Within a single page, natural language explanations and mathematical expressions frequently appear in interleaved configurations, accompanied by extensive multiline derivation steps, chains of equations, and cross-line intermediate results that collectively encode rich procedural information. The mathematical expressions themselves present high structural complexity, commonly featuring fractions, radicals, superscripts and subscripts, nested parentheses, and diverse geometric and logical symbols, further compounding the dificulty of symbol-level accurate transcription and structural consistency constraints. In addition, HANS contains a non-trivial proportion of hand-drawn tables, whose cells typically contain both natural language and mathematical content simultaneously, substantially increasing the coupling between structural layout recognition and content transcription. Furthermore, the answer sheets exhibit pervasive real-world noise and correction behaviors characteristic of authentic examination contexts, including erasures, strikethroughs, and overwriting, which introduce stroke discontinuities, stroke superposition, local occlusion, and structural degradation. These artifacts result in severe spatial entanglement between valid handwriting and noise traces, significantly elevating the uncertainty and dificulty faced by recognition models.

![](images/29ca32c6e35346c0723c764e012d4a9ed15dccd8567506528051b3578b63747f.jpg)  
Fig. 2: Characteristics of the HANS annotated dataset. Each sample in the dataset is annotated with both the complete problem-solving process transcription and the corresponding correction noise regions. Blue bounding boxes denote the annotation of interleaved mathematical expressions and natural language text, while red bounding boxes indicate the spatial annotations of correction noise artifacts.

## 3.2 Annotation Details

Unlike benchmarks that provide only final answer transcriptions or annotations for isolated formulae, we provide annotations more closely aligned with the requirements of automated grading. The fundamental sample unit of the dataset is defined as the answer region image corresponding to each open-ended problem, with each sample containing two categories of annotations: (1) problem-solving process recognition annotations, and (2) non-semantic noise region annotations for erasures and strikethroughs. Full details are illustrated in Fig. 2.

Problem-Solving Process Recognition Annotations Each sample is provided with a recognition transcription corresponding to the complete problemsolving process visible in the image; content that has been deleted by the student is excluded from the final transcription. In terms of structural representation, the problem-solving process is segmented according to the semantic information of each handwritten line, strictly preserving the original line breaks as written. The logical ordering between lines is inferred from handwriting layout conventions, with line break symbols employed to denote logical sequencing.

Noise Region Detection Annotations For images containing noise, we provide pixel-level annotation files for noise regions, stored in JSON format. All noise instances within an image are uniformly assigned the label noise and are permitted to adopt one of two geometric annotation structures: block-level erasures or strikethroughs are annotated using rectangle, defined by two diagonal corner coordinates, while irregularly shaped corrections or localized ink clusters are annotated using polygon, defined by a closed multi-point contour. In cases where noise regions partially overlap with valid content, conservative boundary fitting is applied to minimize erroneous coverage of efective derivation areas. For strikethrough scenarios, annotations primarily delineate the extent of visible occlusion caused by deletion strokes. For overwriting scenarios, annotations encompass both the discarded content and the superimposed strokes, while endeavoring to preserve the final legible and correct writing. For erasure scenarios, annotations cover both erasure residuals and regions rendered illegible by the erasing action.

## 3.3 Annotation Pipeline

To ensure both annotation quality and operational eficiency, we design a threestage pipeline for the HANS annotation task, comprising automatic pre-annotation, annotator correction, and expert quality inspection.

Automatic pre-annotation. We employ state-of-the-art detection and recognition models to generate pre-annotations for both noise region detection and problem-solving process transcription. Specifically, YOLOv8 [6] is adopted for the automatic localization of student correction regions, while DeepSeek-OCR [22] is employed for preliminary content recognition of answer sheets, encompassing annotations for text, mathematical expressions, and tables.

Annotator correction. Following automatic pre-annotation, human annotators systematically revise the initial annotation results. For student correction regions, annotators identify and supplement any missed noise areas, and refine the bounding box boundaries to more precisely conform to the extents of the correction artifacts. Each character is individually verified to ensure transcription accuracy. With respect to the overall problem-solving process, annotators perform character-level inspection and correction of the preliminary recognition drafts, ensuring the accurate representation of text, mathematical symbols, and their structural relationships. Omitted content is supplemented where necessary, and output formatting is standardized to satisfy the requirements of subsequent model training and evaluation.

Quality inspection. Despite the comprehensive corrections performed by annotators, the structural complexity of mathematical formulae and tables may nonetheless introduce residual errors. Accordingly, domain experts conduct a dedicated quality inspection pass to verify annotation accuracy and ensure consistency across the dataset.

## 4 Method

We propose NA-GOT, a noise-aware end-to-end answer sheet recognition framework, as illustrated in Fig. 3, designed to mitigate the interference of real-world noise artifacts, including erasures, strikethroughs, and overwriting on the accurate recognition of valid handwritten characters. Building upon GOT, the proposed framework suppresses noise through a two-stage mechanism. In the first stage, we design a Lightweight Noise Suppression Module (LNSM) that attenuates noise influence at the feature level. In the second stage, we introduce a Noise-aware attention mechanism that reduces attention weights assigned to noise-bearing regions via learned noise bias terms. Specifically, the input image is first processed by a visual encoder to extract patch-level features, while a mask matrix generated from noise annotations provides noise-aware supervision signals during training. The noise-aware module performs weighted suppression of features according to the predicted noise probabilities, and subsequently modulates the decoder’s attention distribution to direct the model’s focus toward valid handwriting regions. The resulting visual features are then projected via a linear layer and fed into the decoder, which generates the final Markdown-formatted output, achieving complete transcription of natural language text, mathematical formulae, and step-by-step derivations.

![](images/40ceb61a1c7c8c756b9c92c376072dc43bfcd7c6fa4e27d18fb318f8dc8f1422.jpg)  
Fig. 3: Overall architecture of the NA-GOT framework.

## 4.1 Lightweight Noise Suppression Module

We first construct patch-level supervision signals aligned with the visual patch tokens from the noise annotations. Specifically, for each image, all annotations are first converted into a binary pixel mask matrix M according to the original image resolution. To maintain consistency with the patch token layout of the GOT visual encoder, M is uniformly resized to a resolution of 1024×1024, followed by 16×16 average pooling to compute the noise occupancy ratio over the patch grid. The resulting grid is subsequently flattened into patch-level noise labels m<sup>g</sup>t in one-to-one correspondence with the visual tokens, where each entry in m<sup>g</sup>t represents the proportion of noise pixels within the corresponding patch.

We takes the patch-level noise labels m<sup>g</sup>t and the output features V of the visual encoder as inputs to the Noise Mask Head (NMH), which determines whether each patch constitutes a noise patch. Concretely, NMH employs a twolayer MLP to predict a noise confidence score p for each patch token:

$$
p = \sigma ( { \mathrm { M L P } } ( V ) )
$$

This design enables the model to adaptively predict noise distributions during inference without relying on manual annotations or additional noise inputs, while simultaneously providing stable noise priors for subsequent token gating and the decoder’s noise-aware attention bias, thereby enhancing the robustness of downstream sequence generation.

Leveraging the obtained noise confidence scores $p ,$ we further suppress the representational magnitude of noise tokens at the visual token level. Specifically, the noise confidence scores p are mapped to gating coeficients, which are applied to scale the visual tokens on a per-token basis:

$$
\tilde { V } = V \odot \left( 1 - \alpha \cdot p \right)
$$

where $\odot$ denotes element-wise multiplication, and α is a hyperparameter controlling the gating strength.

## 4.2 Weighted Attention Correction

While LNSM modulates the representational magnitude of visual tokens, the decoder may nevertheless assign substantial attention weights to noise tokens. Since NA-GOT concatenates visual patch tokens and text tokens into a unified sequence, cross-modal alignment occurs within the decoder. To suppress the interference of noise visual tokens on the generation process, Weighted Attention Correction (WAC) introduces a negative bias derived from the noise probabilities p output by NMH into the decoder’s self-attention layers:

$$
b = - \beta \cdot p
$$

where b denotes the bias term and $\beta$ is a hyperparameter controlling the penalty strength. A higher noise probability yields a larger bias, resulting in a lower attention weight for the corresponding patch following the softmax operation.

This mechanism penalizes noise patches during attention computation, directing the decoder’s focus toward valid handwriting regions, thereby improving both the accuracy and structural integrity of the generated Markdown output in complex handwritten answer sheet scenarios.

## 4.3 Loss Function

The NMH predicts a noise confidence score p for each visual patch token, and the noise labels $m ^ { g t }$ are constructed from the noise annotations. A binary crossentropy loss with positive sample weighting is applied to compute the noise mask loss $L _ { m a s k }$ . Given the input image and the corresponding prompt, the model autoregressively generates the target sequence, with cross-entropy loss computed as the primary recognition loss $L _ { o c r }$ . The final training objective is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { o c r } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } }
$$

where $\lambda _ { \mathrm { m a s k } }$ denotes the noise branch weight, controlling the influence of noise supervision on the primary recognition task.

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details We employ the proposed NA-GOT model for both training and evaluation. For the LNSM, patch-level noise supervision signals are generated from noise annotations during training to supervise the NMH in predicting noise distributions. The key hyperparameters adopted are $\alpha = 0 . 7$ and $\beta = 2 . 0$

Dataset and Evaluation Metrics We conduct all experiments on the proposed HANS dataset to evaluate whether the proposed method efectively addresses the robustness challenges inherent to answer sheet recognition. The dataset is partitioned into training and test sets at an 8:2 ratio, with samples containing noise regions comprising three-quarters of the training set. Evaluation is performed using sequence alignment metrics commonly adopted in document parsing research, including BLEU, Edit Distance, METEOR, Precision, Recall, and F1-score.

Baselines To comprehensively evaluate the proposed method, we select three categories of representative baselines for comparison: (1) General VLMs: Qwen3- VL [1] and InternVL3.5 [19]; (2) Specialized VLMs: DeepSeek-OCR2, PaddleOCR-VL1.5 [3], MinerU2.7.6-VLM [18], and GOT; (3) Pipeline Tools: PaddleOCR PP-StructureV3 [4] and MinerU2.7.6-Pipeline. To provide a rigorous comparison, GOT is additionally fine-tuned on the full HANS training set prior to evaluation, thereby demonstrating the superiority of the proposed approach.

## 5.2 Comparison to Previous Approaches

We present a comprehensive comparison of NA-GOT against prior methods on the HANS dataset in Tab. 2, with all baselines organized into three groups by model category. NA-GOT consistently surpasses all baselines across General VLMs, Specialized VLMs, and Pipeline Tools. Pipeline Tools exhibit notably inferior performance on this task, while General VLMs and Specialized VLMs achieve moderate performance. Among these, Qwen3-VL delivers the strongest results, indicating that powerful general-purpose VLMs possess considerable document understanding and generation capabilities, yet remain susceptible to noise interference in the challenging noisy answer sheet scenario. GOT transferred directly to this dataset yields relatively low performance, suggesting a substantial domain shift between this task distribution and the original training data — particularly with respect to noise patterns and the heterogeneous output requirements imposed by interleaved formula-text handwriting. This further underscores the value of HANS in filling a critical data gap in the document parsing domain. In contrast, GOT fine-tuned on HANS achieves significant performance gains, validating the efectiveness of our annotations in enabling the model to rapidly adapt to the target data distribution. Compared to the fine-tuned GOT baseline, our proposed method achieves consistent improvements across all evaluation metrics, with particularly pronounced gains on heavily noisy samples. These results demonstrate that the proposed Lightweight Noise Suppression Module(LNSM) efectively learns to extract noise-invariant features, while the Noise-aware attention mechanism penalizes noise-bearing positions at the attention layer, thereby mitigating noise-induced misalignment and repetitive generation.

Table 2: Experimental results of NA-GOT and baseline models on the HANS dataset.
<table><tr><td>Method</td><td colspan="4">BLEU↑ Edit Distance↓ METEOR↑ Precision↑ Recall↑ F1-score↑</td></tr><tr><td>PaddleOCR PP-V3</td><td>0.177 0.659</td><td>0.336</td><td>0.624 0.546</td><td>0.566</td></tr><tr><td>MinerU2.7.6-pipeline</td><td>0.064 0.734</td><td>0.218</td><td>0.482 0.431</td><td>0.439</td></tr><tr><td>Qwen3-VL</td><td>0.433 0.445</td><td>0.586</td><td>0.734 0.731</td><td>0.724</td></tr><tr><td>InternVL3.5</td><td>0.196 0.650</td><td>0.364</td><td>0.660 0.554</td><td>0.650</td></tr><tr><td>Paddle-VL1.5</td><td>0.349 0.526</td><td>0.497</td><td>0.722 0.665</td><td>0.678</td></tr><tr><td>MinerU2.7.6-VLM</td><td>0.274 0.584</td><td>0.485</td><td>0.611 0.677</td><td>0.629</td></tr><tr><td>Deepseek-OCR2</td><td>0.338 0.533</td><td>0.505</td><td>0.655</td><td>0.665 0.649</td></tr><tr><td>GOT</td><td>0.098 0.789</td><td>0.231</td><td>0.426</td><td>0.470 0.409</td></tr><tr><td>GOT(fine tuned)</td><td>0.577 0.324</td><td>0.687</td><td>0.812</td><td>0.806 0.801</td></tr><tr><td>NA-GOT(fine tuned) 0.594</td><td>0.306</td><td>0.701</td><td>0.821</td><td>0.813 0.810</td></tr></table>

## 5.3 Ablation Study

We conduct an ablation study by removing individual noise-aware components from the full NA-GOT model to evaluate their respective contributions, as reported in Tab. 3. The $\mathrm { w / o }$ WAC variant achieves a BLEU score of 0.589 and an Edit Distance of 0.314, indicating that the noise prediction and token gating mechanism alone can efectively suppress noise interference at the feature level. The $\mathrm { w / o }$ token gating variant yields lower BLEU and F1-score than the $\mathrm { w / o }$ WAC variant, suggesting that feature-level token gating contributes more substantially to robust recognition than decoder-level attention correction alone. Nevertheless, both ablated variants underperform the full NA-GOT model. The full model achieves the best overall performance across all evaluation metrics, demonstrating that token gating and WAC provide complementary benefits at the representation and decoding stages, respectively.

Table 3: Ablation study of token gating and Weighted Attention Correction (WAC) in NA-GOT.
<table><tr><td>Method</td><td>BLEU↑</td><td>Edit Distance↓</td><td>METEOR↑</td><td>Precision↑</td><td>Recall↑</td><td>F1-score↑</td></tr><tr><td>NA-GOT</td><td>0.594</td><td>0.306</td><td>0.701</td><td>0.821</td><td>0.813</td><td>0.810</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  WAC</td><td>0.589</td><td>0.314</td><td>0.698</td><td>0.820</td><td>0.809</td><td>0.808</td></tr><tr><td>w/o Token Gating 0.580</td><td></td><td>0.321</td><td>0.691</td><td>0.815</td><td>0.808</td><td>0.803</td></tr></table>

Table 4: Analysis of the influence of hyperparameter α on model performance.
<table><tr><td>α BLEU↑</td><td>Edit Distance↓</td><td>METEOR↑</td><td>Precision↑</td><td>Recall↑</td><td>F1-score↑</td></tr><tr><td>0.5 0.592</td><td>0.307</td><td>0.698</td><td>0.818</td><td>0.806</td><td>0.807</td></tr><tr><td>0.7 0.594</td><td>0.306</td><td>0.701</td><td>0.821</td><td>0.813</td><td>0.810</td></tr><tr><td>0.90.593</td><td>0.306</td><td>0.699</td><td>0.820</td><td>0.810</td><td>0.809</td></tr></table>

## 5.4 Hyperparameter experimentation

Furthermore, we conduct ablation experiments on the two key hyperparameters, α and β. As reported in Tab. 4, the gating strength coeficient α is found to exert a significant influence on model performance. When $\alpha = 0 . 5$ , noise suppression is insuficient, allowing noise tokens to propagate into the decoder with considerable magnitude, resulting in marginally lower BLEU and F1-score. When $\alpha = 0 . 9 ,$ , excessive suppression impairs a portion of valid features, leading to slight degradation in BLEU and METEOR. The model achieves optimal overall performance at $\alpha = 0 . 7 \mathrm { . }$ , indicating that moderate representation-level noise suppression efectively reduces noise-induced erroneous generation without sacrificing coverage of critical information.

The hyperparameter β governs the penalty strength imposed on noise patches at the attention layer, with results reported in Tab. 5. When $\beta = 1 . 0$ , the penalty is insuficient, causing the model to produce erroneous outputs in certain noisebearing regions. When $\beta = 3 . 0$ , the excessively large penalty introduces mild over-suppression, marginally weakening the decoder’s attention allocation toward valid patches. At $\beta = 2 . 0$ , the model exhibits a stronger tendency to aggregate information from non-noise regions, yielding optimal performance across all evaluation metrics.

## 5.5 Visual Analysis

The recognition results from four models—NA-GOT, Qwen3-VL, DeepSeek-OCR2, and MinerU2.7.6-pipeline—are compared in Figure 4. NA-GOT demonstrates stronger robustness and generation stability in noisy areas, making fewer errors in misinterpreting erasure textures or strikethrough lines as characters or symbols. It maintains a reasonable line-level structure and derivation order.Additionally, NA-GOT shows a higher fidelity to structured formulas, with fractions, parentheses, and equality chains more consistent, and key variables and operators more coherent, thus reducing semantic bias caused by omissions, mistakes, or structural damage.

Table 5: Analysis of the influence of hyperparameter β on model performance.
<table><tr><td>β BLEU↑</td><td>Edit Distance↓</td><td>METEOR↑</td><td>Precision↑</td><td>Recall↑</td><td>F1-score↑</td></tr><tr><td>1.00.593</td><td>0.307</td><td>0.700</td><td>0.819</td><td>0.811</td><td>0.810</td></tr><tr><td>2.0 0.594</td><td>0.306</td><td>0.701</td><td>0.821</td><td>0.813</td><td>0.810</td></tr><tr><td>3.00.591</td><td>0.308</td><td>0.697</td><td>0.817</td><td>0.807</td><td>0.808</td></tr></table>

![](images/ed19ef1d522306d72005c5f70bb725f0c915202c184368595bbb936304dfc7f1.jpg)  
Fig. 4: Visualization Results.Figure 4 presents a sample of answer sheet image recognition. The upper part shows the original answer sheet image, with the red box indicating the noise areas due to student corrections. The lower part displays the recognition results from four models—NA-GOT, Qwen3-VL, DeepSeek-OCR2, and MinerU2.7.6- pipeline—with red boxes highlighting the recognition errors caused by noise regions in the image

In contrast, general multimodal models like Qwen3-VL, although generally readable, are more prone to misreading similar-looking variables and numbers near noise boundaries, and tend to experience local misalignment or omission in complex derivations. Specialized VLMs, such as DeepSeek-OCR2, are more prone to decoding instability or structural misreadings under strong noise conditions. For instance, in the red box, DeepSeek-OCR2 shows noticeable repetitive outputs, rendering the entire derivation unusable. Pipeline methods, such as MinerU2.7.6-pipeline, are more likely to miss content or make incorrect identifications, leading to missing steps, disjointed content, or even unrelated structures being introduced.

Overall, this visualization case demonstrates that NA-GOT efectively mitigates the misattention and misalignment issues induced by noise, significantly suppressing the degradation caused by repetitive generation in long sequences. As a result, it provides more stable, structured, and accurate recognition outputs, which are closer to being usable for subsequent automatic grading and diagnosis in real answer sheet scenarios.

## 6 Conclusion

In this paper, we address the problem of recognizing students’ handwritten problem-solving processes on answer sheets in smart education scenarios. We introduce HANS, the first dataset specifically constructed for this task, encompassing mathematical expressions, natural language text, hand-drawn tables, and diverse noise artifacts including strikethroughs and deletions, with finegrained annotations that provide a reliable foundation for robust recognition research. Furthermore, we propose NA-GOT, an end-to-end noise-aware recognition framework that achieves two-stage noise suppression through a lightweight noise suppression module and a noise-aware attention mechanism. Experimental results demonstrate that the HANS dataset presents substantial challenges to existing methods, while NA-GOT achieves significant improvements in both recognition accuracy and output stability for problem-solving process transcription, ofering an efective solution for handwritten document parsing in real-world educational scenarios.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

2. Blecher, L., Cucurull, G., Scialom, T., Stojnic, R.: Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418 (2023)

3. Cui, C., Sun, T., Liang, S., Gao, T., Zhang, Z., Liu, J., Wang, X., Zhou, C., Liu, H., Lin, M., et al.: Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957 (2026)

4. Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., Wang, X., Zhang, Z., Zhou, C., Liu, H., et al.: Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595 (2025)

5. Gervais, P., Fadeeva, A., Maksai, A.: Mathwriting: A dataset for handwritten mathematical expression recognition. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. pp. 5459–5469 (2025)

6. Hussain, M.: Yolo-v1 to yolo-v8, the rise of yolo and its complementary nature toward digital manufacturing and industrial defect detection. Machines 11(7), 677 (2023)

7. Jaume, G., Ekenel, H.K., Thiran, J.P.: Funsd: A dataset for form understanding in noisy scanned documents. arXiv preprint arXiv:1905.13538 (2019)

8. Li, C., Guo, R., Zhou, J., An, M., Du, Y., Zhu, L., Liu, Y., Hu, X., Yu, D.: Pp-structurev2: A stronger document analysis system. arXiv preprint arXiv:2210.05391 (2022)

9. Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., Wei, F.: Trocr: Transformer-based optical character recognition with pre-trained models. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 13094– 13102 (2023)

10. Li, Z., Liu, Y., Liu, Q., Ma, Z., Zhang, Z., Zhang, S., Guo, Z., Zhang, J., Wang, X., Bai, X.: Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218 (2025)

11. Liu, C., Wei, H., Chen, J., Kong, L., Ge, Z., Zhu, Z., Zhao, L., Sun, J., Han, C., Zhang, X.: Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295 (2024)

12. Marti, U.V., Bunke, H.: The iam-database: an english sentence database for ofline handwriting recognition. International journal on document analysis and recognition 5(1), 39–46 (2002)

13. Ouyang, L., Qu, Y., Zhou, H., Zhu, J., Zhang, R., Lin, Q., Wang, B., Zhao, Z., Jiang, M., Zhao, X., et al.: Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24838–24848 (2025)

14. Pfitzmann, B., Auer, C., Dolfi, M., Nassar, A.S., Staar, P.: Doclaynet: A large human-annotated dataset for document-layout segmentation. In: Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. pp. 3743–3751 (2022)

15. Poznanski, J., Rangapur, A., Borchardt, J., Dunkelberger, J., Huf, R., Lin, D., Wilhelm, C., Lo, K., Soldaini, L.: olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443 (2025)

16. Poznanski, J., Soldaini, L., Lo, K.: olmocr 2: Unit test rewards for document ocr. arXiv preprint arXiv:2510.19817 (2025)

17. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

18. Wang, B., Xu, C., Zhao, X., Ouyang, L., Wu, F., Zhao, Z., Xu, R., Liu, K., Qu, Y., Shang, F., et al.: Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839 (2024)

19. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025)

20. Wei, H., Kong, L., Chen, J., Zhao, L., Ge, Z., Yang, J., Sun, J., Han, C., Zhang, X.: Vary: Scaling up the vision vocabulary for large vision-language model. In: European Conference on Computer Vision. pp. 408–424. Springer (2024)

21. Wei, H., Liu, C., Chen, J., Wang, J., Kong, L., Xu, Y., Ge, Z., Zhao, L., Sun, J., Peng, Y., et al.: General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704 (2024)

22. Wei, H., Sun, Y., Li, Y.: Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234 (2025)

23. Xia, R., Ye, H., Yan, X., Liu, Q., Zhou, H., Chen, Z., Shi, B., Yan, J., Zhang, B.: Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. IEEE Transactions on Image Processing (2025)

24. Xie, Y., Mouchère, H., Simistira Liwicki, F., Rakesh, S., Saini, R., Nakagawa, M., Nguyen, C.T., Truong, T.N.: Icdar 2023 crohme: Competition on recognition of handwritten mathematical expressions. In: International Conference on Document Analysis and Recognition. pp. 553–565. Springer (2023)

25. Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of text and layout for document image understanding. In: Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. pp. 1192–1200 (2020)

26. Yang, Z., Tang, J., Li, Z., Wang, P., Wan, J., Zhong, H., Liu, X., Yang, M., Wang, P., Bai, S., et al.: Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 21744–21754 (2025)