# MedPAO-VL: Interpretable and Protocol-Compliant Agent for Structured Radiology Report Generation

Shrish Shrinath Vaidya<sup>1</sup> <sup>⋆</sup>, Gowthamaan Palani<sup>1</sup> , Minmini Selvam<sup>2</sup>, Gokulraja Srinivasaraja<sup>3</sup>, and Ganapathy Krishnamurthi<sup>1</sup>

<sup>1</sup> Department of Data Science and Artificial Intelligence, Indian Institute of Technology Madras, Chennai, India

<sup>2</sup> TANUH – The AI Centre of Excellence in Healthcare, Indian Institute of Science, Bengaluru, India

3 Department of Radiology (Division of Interventional Radiology), Ramaiah Medical College and Memorial hospital, Bengaluru, India

Abstract. Despite the remarkable generalization capabilities of Large Language Models, generating clinically accurate and structured radiology reports from medical images remains a significant challenge. Current end-to-end approaches function as black boxes, prone to factual hallucinations and generating ungrounded findings. This lack of reliability and failure to adhere to domain-specific structuring protocols, fundamentally limits their clinical utility. We present MedPAO-VL, a multimodal agent that shifts the paradigm from monolithic generation to explainable, iterative, protocol-compliant structured report generation. Unlike deterministic models, our plan-act-observe loop functions as a goal-oriented agent: it first perceives the Chest X-Ray through an explainable vision module designed to capture inherent diagnostic uncertainty, then iteratively refines the generated report by observing semantic discrepancies and self-correcting its generation for better alignment with visually detected clinical findings. Evaluation of our framework demonstrates that this agentic reasoning improves clinical correctness and completeness, achieving an expert-rated accuracy score of 3.5/5 and a structural score of 4.3/5. By prioritizing pathology specific explainability and iterative refinement over opaque end-to-end mapping, MedPAO-VL ofers a transparent and clinically viable alternative for automated structured radiology report generation. The code is available at: https://github.com/MiRL-IITM/medpao-vl

Keywords: Explainability · Structured Report Generation · Multimodal agent · Clinical Protocols · Iterative refinement

## 1 Introduction

Generating accurate and structured radiology reports remains a significant bottleneck in clinical workflows. Escalating volumes of medical imaging procedures exert immense pressure on radiologists, creating a critical need for automated systems that can reduce turnaround times and standardize reporting quality [14]. Yet in high-stakes clinical settings, automation alone is insuficient. Any deployed system must also be trustworthy: its outputs must be interpretable, its reasoning auditable, and its failure modes clinically safe. This demand has catalyzed two parallel, yet largely distinct, streams of research: end-to-end report generation from images and the post-hoc structuring of unstructured reports. The first stream is dominated by multimodal models that generate reports directly from medical images. While these models demonstrate considerable promise, they function as black-boxes, ofering clinicians little insight into how a conclusion was reached. A further fundamental limitation lies in their training data, which operates on the flawed assumption that any pathology not explicitly mentioned in a ground-truth report is implicitly absent. This is clinically unsound: the absence of a finding may reflect a lack of observation rather than its confirmed absence, and encoding this ambiguity as ground truth undermines both model reliability and clinical trust. In contrast, the second stream prioritizes verifiable reasoning and strict protocol compliance. Frameworks such as Med-PAO [55] employ an agentic loop to parse unstructured reports and transform them into a structured, protocol-compliant format like ABCDEF [24](details in Supplementary material), producing outputs that are auditable and adherent to standardized guidelines. However, its primary limitation is its post-hoc nature: it structures existing text rather than reasoning from primary visual evidence, leaving the opacity of the upstream generative process unaddressed.

![](images/4ab86c86ba90a23c0e10b68adf9b10d7406f49611082f471dcb1d28d33070bf8.jpg)  
Fig. 1: MedPAO-VL integrates explainable, uncertainty-aware visual perception and iterative self-correction, overcoming limitations of traditional VLMs and post-hoc report structuring methods.

This dichotomy exposes a critical gap at the intersection of explainability and generative capability. No existing solution integrates the visual reasoning power of multimodal models with the transparent, protocol-driven auditability of agentic frameworks, a gap that directly compromises the trustworthiness of automated radiology pipelines. To bridge this gap, we introduce MedPAO-VL (MedPAO-Vision-Language), a multimodal agent which is both end-to-end and transparent by design to generate structured, protocol-compliant reports from CXRs. By extending the MedPAO framework [55] with multimodal capabilities, we embed explainability into the generative process itself rather than as an afterthought. The key contributions of this work are as follows:

1. We introduce MedPAO-VL, a multimodal agent that unifies visual perception with a Plan-Act-Observe loop, yielding protocol-compliant report generation with step-wise auditable reasoning.

2. A novel Vision Concept Module employing a uncertainty aware labeling scheme for pathologies to provide reliable and explainable visual recognition that more faithfully reflects clinical epistemology.

3. Extensive evaluation demonstrating that our hybrid architecture produces clinically sound reports that are factually consistent and protocol-compliant alongwith interpretable vision module, advancing the trustworthiness of automated radiology reporting.

## 2 Related Works

## 2.1 Radiology Report Generation

Automated radiology report generation has emerged as a significant area of research, drawing initial inspiration from the successes of natural image captioning [38,47,63]. Foundational approaches to report generation established the encoder-decoder framework, initially employing Convolutional Neural Networks (CNNs) paired with Long Short-Term Memory (LSTM) networks [21, 22, 31]. With the advent of the Transformer [56] architecture, the field evolved to replace recurrent decoders with LLMs, yielding CNN-LLM configurations [17, 27, 65]. This evolution has culminated in Vision Transformers (ViTs)-LLM configurations [7, 35, 36, 51, 60]. Beyond the standard encoder-decoder paradigm, various lines of research have introduced architectural modifications and auxiliary techniques to achieve logical coherence and factual correctness by integrating memory-driven modules [4, 13, 49], retrieval methods [11] with contrastive learning [10, 32, 44], fine-grained lesion features [29, 52], and medical knowledge graphs [26, 34, 66, 68]. More recently, the field is advancing towards native Vision-Language Models (VLMs) [41, 48, 54, 64] and Generalist Medical AI (GMAI) [40, 54] architectures. Unlike modular approaches that bridge separate pre-trained vision and language encoders, these foundation models are often trained end-to-end on vast, multimodal datasets. Agent-based frameworks ofer a compelling alternative to monolithic LLMs by enabling task decomposition through specialized tools. For instance, a multi-agent architecture was recently proposed to handle disease-specific report generation by assigning specialized tasks to individual agents [58]. This concept was later extended by incorporating a Concept Bottleneck Model [25] to extract critical pathologies; these findings then serve as the foundation for a multi-agent RAG system designed to generate clinically grounded and transparent medical documentation [1]. While Foundational models, LLMs and VLMs excel at producing fluent and descriptive narratives, they fail to generate structured reports grounded in clinical protocols. Though existing agentic frameworks are robust, they often struggle with complex reasoning and factual hallucinations.

## 2.2 Explainable AI in Report Generation

While recent models for automated report generation have achieved strong quantitative performance, their adoption in clinical practice hinges on meeting the critical requirement of explainability. To address this, a prominent line of research has focused on improving visual interpretability [57, 62]. Several studies employed region-guided encoder networks that identify pathological areas and use these localized detections as grounded cues for generation [2, 52, 67]. Other approaches integrated radiologists’ eye-gaze data as a supervisory signal to train more explainable image encoders [45]. Further enhancing this paradigm, methods with learnable query tokens optimized visual processing while yielding interpretable spatial attention maps through cross-attention mechanisms [9, 59]. In parallel, a distinct set of approaches prioritized concept-level explainability [11,20] over spatial heatmaps. These methods output discrete, scalar intermediate representations, to construct more informative prompts for an LLM-based report generator. Extending this idea, other works explicitly extracted key clinical findings from reports using frameworks like RadGraph [19] and leveraged these structured entities to guide the generation process [30, 61]. With the exception of PromptMRG [20], existing methodologies generally operate under the implicit assumption that pathologies not explicitly documented in the reference reports are absent. Furthermore, all of these approaches rely on general purpose LLMs to extract clinical concepts, which lack verification by radiologists.

## 3 Methodology

We introduce MedPAO-VL, an interpretable, multimodal agent designed to generate protocol-compliant structured radiology reports through an iterative refinement process. Extending the architecture established in MedPAO [55], our system integrates three additional components, the Vision Concept Module, Concept Grounded Report Generator and an Agentic PAO(Plan-Act-Observe) loop. The overall system is as shown in Fig.2. An overview of tools used in MedPAO architecture can be seen in Supplementary material.

## 3.1 Vision Concept Module

Mitigating Implicit Negative Bias Radiology reports frequently omit unresolved or expected findings, reflecting clinical prioritization rather than documented absence [46]. However, prominent AI frameworks [16,57] typically enforce a closed-world assumption by applying U-Zeros strategies [18] to map unmentioned or "uncertain" annotations to definite negatives. This introduces an implicit negative bias that corrupts training signals, suppresses uncertainty, and distorts spatial grounding. To eliminate this bias, we formulate pathology prediction as a three-class problem. Given a CXR image x, the model predicts a label from {present, absent, uncertain} for each of n target pathologies. This tripartite scheme mirrors clinical diferential diagnosis by explicitly distinguishing true absence from reporting omission. Fig. 2(a) represents an overview of proposed vision model.

![](images/6b7fc6245dbe549cd69e802f086b8e9da798b0dec08884a3d5ff8202765528a5.jpg)  
Fig. 2: Overview of proposed MedPAO-VL architecture. It mainly comprises of (a)Vision Concept Module for extracting the clinical findings, (b)Pathology slot module for extracting explainable pathology embeddings and (c)An iterative PAO loop to handle report correction and Structuring.

Explainable Pathology Slot Module The Pathology Slot Module (PSM) is motivated by object-centric learning [8, 37] and slot attention-based approaches for explainable image classification [28]. Given backbone feature representations, the PSM instantiates n pathology-specific slots, each designed to extract features relevant to its corresponding pathology. For an input feature map $\boldsymbol { F } \in \mathbb { R } ^ { m \times d }$ where m is the number of spatial tokens and d is the feature dimension, the PSM disentangles the global representation into n pathology-specific context vectors $U \in \mathbb { R } ^ { n \times d }$ via iterative slot refinement over T iterations, with slots initialized as $S _ { 0 } \sim \mathcal { N } ( 0 , 1 ) \in \mathbb { R } ^ { n \times d }$ . The full forward pass is illustrated in Fig. 2(b).

Key Attention Design. The central contribution of the PSM lies in its attention mechanism. At each iteration t, pathology-specific slots $S _ { t - 1 }$ attend over the spatial feature map to compute attention logits via scaled dot-product attention:

$$
{ \cal A } _ { \mathrm { l o g i t s } } = \frac { Q K ^ { \top } } { \sqrt { d } } \in \mathbb { R } ^ { n \times m } , \qquad { \cal A } _ { t } = \mathrm { s o f t m a x } \biggl ( \frac { { \cal A } _ { \mathrm { l o g i t s } } } { \tau } , ~ \mathrm { d i m } = - 1 \biggr ) ,
$$

where Q and K are projections of the current slots and input features respectively, and τ is a temperature hyperparameter. Crucially, unlike the original Slot Attention [37], which applies softmax over the slot dimension to enforce competition between slots per patch, we apply softmax over the patch dimension. This design choice is motivated by the multi-label nature of pathology prediction: a single image patch may carry evidence for multiple co-occurring conditions, and forcing slot competition would suppress these overlapping signals. Beyond enabling multi-label recognition, this formulation yields a direct interpretability benefit - each $A _ { t } \in \mathbb { R } ^ { n \times m }$ constitutes a per-pathology spatial attention map that is an inherent product of the forward pass, providing faithful, pathology-specific visual grounding without any post-hoc approximation. The slot state is subsequently updated via a GRU cell, integrating the attended context with prior slot state:

$$
U _ { t } = A _ { t } V \in \mathbb { R } ^ { n \times d } , \qquad S _ { t } = \mathrm { G R U } ( U _ { t } , \ S _ { t - 1 } ) .
$$

After T iterations, the final context vectors $U _ { T } \in \mathbb { R } ^ { n \times d }$ encode semantically rich, pathology-specific representations for eficient downstream three-class classification task.

Shared Classification Head. A single shared head operates on the final context vectors $U _ { T }$ from the PSM and performs three-class (present, absent, uncertain) prediction for each pathology i:

$$
p ^ { i } = \mathrm { s o f t m a x } \big ( \mathrm { D r o p o u t } \big ( \mathrm { R e L U } ( U _ { T } ^ { i } W _ { 1 } + b _ { 1 } ) \big ) W _ { 2 } + b _ { 2 } \big ) ,
$$

yielding Probability $= \{ p ^ { 1 } , \ldots , p ^ { n } \}$ . Rather than n independent classifiers, this parameter-eficient design enforces a common decision boundary, placing greater representational burden on the PSM while incentivising the slot vectors to be maximally discriminative.

Interpretability-Aware Training Objective The primary training objective is the sum of cross-entropy losses across all n pathologies:

$$
{ \mathcal { L } } _ { \mathrm { c l s } } = \sum _ { i = 1 } ^ { n } { \mathrm { C r o s s E n t r o p y } } \left( p ^ { i } , g t ^ { i } \right) ,
$$

where $p ^ { i }$ and $g t ^ { i }$ denote the predicted probabilities and ground-truth label for pathology i.

Similarity Loss as an Interpretability Regularizer. Optimizing ${ \mathcal L } _ { \mathrm { c l s } }$ alone is insuficient for producing meaningful explanations. Since the PSM applies softmax over the patch dimension, the slot attention maps are susceptible to representational collapse, converging toward near-identical distributions across all n slots. To explicitly regularize against this failure mode and enforce per-pathology spatial diversity, we introduce $\mathcal { L } _ { \mathrm { s i m } } ,$ , a redundancy penalty over the attention maps. Following [42], each normalized attention map $A _ { k }$ is $\ell _ { 2 } \cdot$ -normalized along the patch dimension, and a pairwise cos-cubed similarity matrix $M \in \mathbb { R } ^ { n \times n }$ is computed by element-wise cubing the normalized maps and taking their inner product:

$$
\hat { A } _ { k } = \frac { A _ { k } } { \vert \vert A _ { k } \vert \vert _ { 2 } } \in \mathbb { R } ^ { n \times m } , \qquad M = \hat { A } ^ { 3 } \Big ( \hat { A } ^ { 3 } \Big ) ^ { \top } ,
$$

where $\hat { A } ^ { 3 }$ denotes element-wise cubing, and each entry $M _ { i , j }$ measures the coscubed similarity between the attention maps of pathology i and pathology j. The cubic transformation amplifies high-similarity signals while suppressing noise, making the penalty sensitive to genuine redundancy rather than incidental overlap. To penalize only redundancy while permitting benign anti-correlation between anatomically distinct pathologies, ReLU is applied exclusively to ofdiagonal entries:

$$
{ \mathcal { L } } _ { \mathrm { s i m } } = { \frac { 1 } { n ( n - 1 ) } } \sum _ { i = 1 } ^ { n } \sum _ { \underset { j \neq i } { \sum } } ^ { n } \operatorname { R e L U } ( M _ { i , j } ) .
$$

The complete training loss is a linear combination of both terms:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { s i m } } \mathcal { L } _ { \mathrm { s i m } } , } \end{array}
$$

where $\lambda _ { \mathrm { s i m } }$ balances classification performance against interpretability quality. The result is a training objective that jointly optimizes for diagnostic accuracy and spatial interpretability, ensuring that the PSM’s explanations are both clinically faithful and visually diverse across pathologies. For further analysis of $L _ { \mathrm { s i m } }$ refer to Supplementary material.

## 3.2 Concept Grounded Report Generator

The Concept Grounded Report Generator is a multimodal model tasked at synthesizing a fluent, unstructured report. Unlike unconstrained generation, this module adopts a concept-driven approach: it receives both the CXR image and semantic constraints, a list of present and absent clinical concepts from the Vision Concept Module and renders these findings into medically coherent report. This dual-input design serves two complementary purposes. First, by grounding the generation on a curated concept vocabulary rather than raw image features alone, the outputs become traceable to discrete, verifiable clinical findings. This traceability is a critical prerequisite for clinical trustworthiness, allowing practitioners to audit the specific concepts driving any narrative statement. Second, utilizing the image as an active visual reference provides the necessary multimodal context to produce naturally expressive descriptions, avoiding the disjointedness of mere concept concatenation.

## 3.3 The Agentic PAO Loop

The PAO loop shown in Fig. $. 2 ( \mathrm { c } )$ is an iterative loop that adapts to the complexity of the clinical findings, by ofering the following capabilities:

Self-Correction Loop and Report Structuring. After extracting the visual findings, the agent begins by initiating the Concept Grounded Report Generator to produce an initial findings narrative. To ensure clinical fidelity, the Observe phase employs Semantic Overlap Checker. This module extracts clinical entities from the generated narrative and computes the semantic similarity against the concept list $\hat { C }$ from the Vision Concept Module. If the checker identifies a discrepancy, the agent triggers a Re-Plan and Act cycle. In this phase, the generator is re-prompted to refine the specific section containing the discrepancy. This loop continues for up to N iterations or until the vision concept list $\hat { C } \subseteq$ CurrentFindings. After loop termination, the agent uses various tools from MedPAO like concept extraction, ontology mapping, concept categorizer and structured report generator to structure the free-style report according to a given clinical protocol. Refer to Supplementary for an end to end example of MedPAO-VL.

Amortized Eficiency via Concept Caching. To mitigate the computational overhead of iterative agentic loops, the PAO framework employs a Dynamic Concept Cache for ABCDEF protocol categorization. By storing and retrieving mappings for frequently recurring CXR findings, the agent bypasses high-latency Ontology Mapping, and Categorizer tool invocations on cache hits. New mappings are dynamically cached, allowing the system to reuse prior knowledge and optimize inference (see Supplementary for further details on PAO implementation).

## 4 Experiments

## 4.1 Dataset & Pre-processing

We utilize a subset of MIMIC-CXR [23] to train both the Vision Concept Module and the Concept Grounded Report Generator. We selected studies where the reports explicitly contain HISTORY, FINDINGS, and IMPRESSION sections, yielding approximately 53,000 CXR–report pairs from the train set. The FINDINGS section serves as the primary input to our concept extraction pipeline, which identifies salient clinical findings from free-form text. We leveraged the same concept extraction pipeline from [55] which was validated on a gold standard dataset annotated by radiologists. Since a same observation may be expressed in many linguistic forms, the extracted concepts are normalized to canonical SNOMED CT ontology using the medical entity normalization model [33], enabling uniform grouping across heterogeneous representations of similar findings. An LLM (DeepSeek-R1-Distill-Llama-70B [12]) is used to separate the present and absent concepts within each group. After radiologists validation, low-frequency or incoherent groups are filtered out, producing a final vocabulary of 236 unique, standardized concept groups. Additional details on radiologist validation is in Supplementary material. At the end of pre-processing steps, the curated dataset comprises approximately 50,000 (CXR, report, concept) triplets that form the basis for all models presented in this work. We utilize MIMIC-CXR [23] test set and IU-XRAY [6] test set for evaluation and the same pre-processing pipeline is applied on the test set to obtain ground-truth concepts from radiology reports, against which all models are validated. Additional details on data curation are provided in Supplementary material.

## 4.2 Implementation Details

For training the Vision Concept Module, the dataset obtained in sub-section 4.1 is partitioned into a 90:10 train–validation split using stratified sampling on the clear lungs pathology. We use the MedSigLIP vision encoder [48] as a fixed backbone, operating on an image resolution of 448x448. The PSM block produces 236 pathology-specific slots corresponding to the 236 target concepts. The similarity loss weight is set to $\lambda _ { \mathrm { s i m } } ~ = ~ 3$ . We train for 50 epochs using AdamW with a cosine learning rate schedule, an initial learning rate of $5 \times 1 0 ^ { - 5 }$ and a batch size of 32. For the Concept Grounded Report Generator, we finetune MedGemma 4B [48] on 10,000 samples for ablation studies and utilize the best hyperparameters to scale the training over 50,000 samples. All the experiments were conducted on a single NVIDIA A6000 (48 GB) GPU. Further implementation details are provided in Supplementary material.

## 4.3 Report Generation Evaluation

Unstructured Report Evaluation As shown in Table 1 and Table 2, the proposed Concept Grounded Report Generator sets a new benchmark for clinical accuracy in chest X-ray report generation. Our 10K and 50K variants, which represent the report generator fine-tuned on 10,000 and 50,000 samples respectively, demonstrate the scalability of our approach. Specifically, the 50K variant achieves a state-of-the-art CheXbert-F1 [50] of 0.617 and a RadGraph-F1 [19] of 0.427 on the MIMIC-CXR dataset, substantially outperforming significantly larger models such as MAIRA-2 [2] and Flamingo-CXR [53]. This superior performance in clinical metrics, paired with a high CheXbert Micro-F1 of 0.706, underscores the model’s ability to correctly identify pathologies and map complex anatomical-clinical relationships. Furthermore, even the 10K variant remains competitive against SOTA baselines, validating the efectiveness of our radiologist-validated concept standardization even when fine-tuned on a smaller subset of data (compared to 1M samples in MAIRA-2 and 500K in Flamingo-CXR). While our model remains competitive in NLG metrics, the high GREEN [43] mean scores (0.733 for IU X-RAY) highlight that our generated reports maintain high clinical utility and diagnostic integrity, efectively minimizing critical omissions compared to existing generative approaches.

Structured Report Evaluation To evaluate the quality of the generated structured radiology reports, we defined two primary scoring metrics: (1) an Accuracy Score, which measures the clinical correctness of the generated findings, and (2) a Structural Score, which assesses the extent to which the generated reports conform to the predefined reporting protocol. The evaluation was conducted by two senior radiologists on a balanced subset of 50 cases drawn from the MIMIC-CXR test set. To ensure a representative evaluation, these cases were selected via stratified random sampling, which preserved the underlying distribution of both CheXpert labels [18] and report complexity (utilizing an is complex flag for reports containing $\geq 9$ concepts; refer to Supplementary material). For each case, the radiologists were presented with samples consisting of reports generated by our agent and reports produced by the MedGemma-27B multimodal model, along with the corresponding CXR image and the groundtruth report. To ensure an unbiased assessment, the radiologists underwent a blinded evaluation, without knowledge of the method responsible for generating the report. The mean evaluation results across all assessed cases are reported in Table 3. Full inter-rater statistics, score distributions and the scoring criteria are described in Supplementary material.

Table 1: Performance Comparison on MIMIC-CXR and IU X-RAY Datasets. This table presents the performance of various models against our Concept Grounded report Generator on generating unstructured radiology reports(without PAO loop) for the MIMIC-CXR and IU X-RAY test sets. The Sections column indicates whether the model generates the FINDINGS(F) or IMPRESSION(I) section of the report, with metrics sourced from published research.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Methods</td><td rowspan="2"></td><td rowspan="2">Year Section</td><td colspan="2">NLG Metrics</td><td colspan="2">Clinical Metrics</td></tr><tr><td>BLEU-4 ROUGE-L CheXbert-F1 RadGraph-F1</td><td></td><td></td><td></td></tr><tr><td rowspan="13">MIMIC-CXR IU X-RAY</td><td>R2Gen [4]</td><td>2020</td><td>F+I</td><td>0.103</td><td>0.277</td><td>0.276</td><td>0.134</td></tr><tr><td>CXR-RePaiR [10]</td><td>2021</td><td>F</td><td>0.021</td><td>0.143</td><td>0.274</td><td>0.091</td></tr><tr><td>M2 Transformer [39]</td><td>2021</td><td>F</td><td>0.114</td><td></td><td></td><td>0.220</td></tr><tr><td>RGRG [52]</td><td>2023</td><td>F</td><td>0.126</td><td>0.264</td><td>0.447</td><td></td></tr><tr><td>METransformer [59]</td><td>2023</td><td>F</td><td>0.124</td><td>0.291</td><td>0.311</td><td></td></tr><tr><td>R2GenGPT [60]</td><td>2023</td><td>F+I</td><td>0.134</td><td>0.297</td><td>0.389</td><td></td></tr><tr><td>Flamingo-CXR [53]</td><td>2023</td><td>F+I</td><td>0.101</td><td>0.297</td><td>0.519</td><td>0.205</td></tr><tr><td>Med-PaLM M, 84B [54]</td><td>2024</td><td>F</td><td>0.113</td><td>0.273</td><td></td><td>0.267</td></tr><tr><td>Med-Gemini-2D [64]</td><td>2024</td><td>F+I</td><td>0.205</td><td>0.283</td><td></td><td>0.244</td></tr><tr><td>PromptMRG [20]</td><td>2024</td><td>F</td><td>0.112</td><td>0.268</td><td>0.476</td><td></td></tr><tr><td>InVERGe [7]</td><td>2024</td><td>F</td><td>0.100</td><td>0.309</td><td></td><td></td></tr><tr><td>LLaVA-Rad [3]</td><td>2024</td><td>F+I</td><td>0.154</td><td>0.306</td><td>0.395</td><td>0.294</td></tr><tr><td>Chexagent [5]</td><td>2024</td><td>F+I</td><td></td><td>0.450</td><td>0.444</td><td>0.266</td></tr><tr><td>MAIRA-2 [2]</td><td>2024</td><td>F</td><td>0.231</td><td>0.384</td><td>0.416</td><td>0.346</td></tr><tr><td>MedGemma, 4B [48]</td><td>2025</td><td>F+I</td><td>0.019</td><td>0.131</td><td></td><td>0.295</td></tr><tr><td>Ours (10K)</td><td></td><td>F</td><td>0.102</td><td>0.298</td><td>0.563</td><td>0.376</td></tr><tr><td>Ours (50K)</td><td></td><td>F</td><td>0.136</td><td>0.328</td><td>0.617</td><td>0.427</td></tr><tr><td rowspan="9"></td><td>R2Gen [4]</td><td>2020</td><td>F+I</td><td>0.165</td><td>0.371</td><td>0.136</td><td></td></tr><tr><td>M2 Transformer [39]</td><td>2021</td><td>F</td><td>0.131</td><td></td><td></td><td></td></tr><tr><td>RGRG [52]</td><td>2023</td><td>F</td><td>0.063</td><td>0.180</td><td>0.180</td><td></td></tr><tr><td>R2GenGPT [60]</td><td>2023</td><td>F+I</td><td>0.173</td><td>0.377</td><td></td><td></td></tr><tr><td>PromptMRG [20]</td><td>2024</td><td>F</td><td>0.112</td><td>0.268</td><td>0.476</td><td></td></tr><tr><td>InVERGe [7]</td><td>2024</td><td>F</td><td>0.100</td><td>0.309</td><td></td><td></td></tr><tr><td>MAIRA-2 [2]</td><td>2024</td><td>F</td><td>0.117</td><td>0.274</td><td>0.319</td><td>0.271</td></tr><tr><td>Ours (10K)</td><td></td><td>F</td><td>0.083</td><td>0.327</td><td>0.203</td><td>0.342</td></tr><tr><td>Ours (50K)</td><td></td><td>F</td><td>0.097</td><td>0.341</td><td>0.424</td><td>0.401</td></tr></table>

Table 2: Comprehensive Evaluation Metrics of Concept Grounded report Generator (50K variant, without PAO loop) on MIMIC-CXR and IU X-RAY test-sets
<table><tr><td rowspan="2">Dataset</td><td colspan="6">NLG</td><td colspan="2"> $\mathbf { C h e X b e r t }$ </td><td colspan="2">GREEN</td></tr><tr><td>BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L Micro-F1 Macro-F1 Mean STD</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\mathrm { R G } _ { \mathrm { E } } \ \mathrm { R G } _ { \mathrm { E R } } \ \overline { { \mathrm { R G } } } _ { \mathrm { E R } }$ </td><td></td></tr><tr><td>MIMIC-CXR</td><td>0.431</td><td>0.272</td><td>0.188</td><td>0.136</td><td>0.407</td><td>0.328</td><td>0.706</td><td>0.617</td><td>0.472 0.214 0.403 0.3740.303</td><td></td></tr><tr><td>IU X-RAY</td><td>0.416</td><td>0.239</td><td>0.150</td><td>0.097</td><td>0.469</td><td>0.341</td><td>0.619</td><td>0.424</td><td>0.733 0.225 0.4950.4580.375</td><td></td></tr></table>

Table 3: Radiologist evaluation of structured reports with ±95% CI
<table><tr><td rowspan="2"></td><td colspan="2">MedGemma 27B</td><td colspan="2">MedPAO-VL</td></tr><tr><td>Accuracy score Structure score Accuracy score Structure score</td><td></td><td></td><td></td></tr><tr><td>Radiologist1</td><td> $\overline { { 2 . 8 8 \pm 0 . 4 4 } }$ </td><td> $\overline { { 3 . 8 4 \pm 0 . 3 6 } }$ </td><td> $\overline { { 3 . 4 0 \pm 0 . 3 2 } }$ </td><td> $\overline { { 4 . 4 4 \pm 0 . 2 4 } }$ </td></tr><tr><td>Radiologist2</td><td> $3 . 4 8 \pm 0 . 5 6$ </td><td> $4 . 1 2 \pm 0 . 2 8$ </td><td> $3 . 6 0 \pm 0 . 4 0$ </td><td> $4 . 2 2 \pm 0 . 3 2$ </td></tr><tr><td>Combined Mean</td><td> $3 . 1 8 \pm 0 . 3 6$ </td><td> $3 . 9 8 \pm 0 . 2 4$ </td><td> ${ \bf 3 . 5 0 \pm 0 . 2 8 }$ </td><td> ${ \bf 4 . 2 8 \pm 0 . 2 0 }$ </td></tr></table>

![](images/99557e939ca30227683b14f54c55561ecbdc46027a0853a49d5a0deef590c262.jpg)  
(a)

![](images/e44c9149fbf5bb7bb9e7571f08e7e06c33730a1d27abc49a800737c954bd35c4.jpg)  
(b)  
Fig. 3: Latent pathology slot vector similarity. 2D cosine similarity projections for (a) “mild $\mathrm { C H F } ^ { \prime \prime }$ and (b) “likely pneumonia” and related pathologies.

## 4.4 Interpretability and Representational Analysis

Latent Orthogonality To evaluate the representational fidelity of the PSM, we analyze inter-slot cosine similarity of the final context vectors $U _ { T }$ . Context vectors are unit-normalized and projected onto a 2D plane anchored to a reference pathology, revealing clinically coherent directional structure: related findings cluster near the reference axis while opposing pathologies are displaced into the negative half-space. For likely pneumonia $\mathrm { ( F i g . 3 b ) }$ , hazy bilateral parenchymal opacities $\left( \cos = + 0 . 9 8 5 , 9 . 7 ^ { \circ } \right)$ and likely loculated fissural fluid $( \cos = + 0 . 9 8 5$ $3 . 1 ^ { \circ } )$ align tightly with the reference, consistent with classical radiographic presentation, whereas pneumothorax (cos =−0.281) and normal pleural surfaces (cos =−0.348) are geometrically opposed. For mild CHF (Fig.3a), mild pulmonary vascular congestion (cos =+0.893) and tiny pleural efusion (cos =+0.696) project positively as expected sequelae of elevated hydrostatic pressure, while emphysema with right apical bulla (cos =−0.360) and low lung volumes (cos =−0.377) are displaced into the opposing quadrant, reflecting their association with obstructive rather than congestive physiology. These geometric relationships demonstrate that the PSM encodes clinically meaningful diferentials as directional structure in embedding space. For further analysis on clinical co-occurrence of learnt pathology embeddings refer to Supplementary material.

![](images/8915f122fb87b200bfad0602b722080bcb66b96be9c7cb8104cafb30724b7954.jpg)

Fig. 4: t-SNE visualization of the slot activation space against ABCDEF clinical protocol.  
![](images/d7579f8b62b20b11a2509941fd2a8541f24db20c766858d88baeca8414600230.jpg)  
Fig. 5: Input CXR and Explainable Attention maps from PSM block

Emergent Anatomical Organization in Latent Manifold We evaluated the representational fidelity of the PSM using t-SNE over the 236-dimensional slot activation space. A significant emergent property is observed: while the model was never trained to categorize findings into the ABCDEF protocol, the latent vectors naturally organize into clusters that coincide with these anatomical categories. To ensure a balanced representation, we utilized a stratified subset of 4,000 slot instances per category from the MIMIC-CXR test set. We employed a cosine distance metric for the projection to prioritize vector similarity over absolute magnitude, efectively capturing the diagnostic relationships within the high-dimensional manifold. As shown in Fig.4, the distinct spatial separation between Airways (A), Breathing (B), and Circulation (C) clusters validates that our concept-driven architecture disentangles co-occurring pathologies based on their anatomical origin. This alignment provides a verifiable diagnostic manifold that mirrors radiological reasoning, bridging the gap between deep vision features and clinical interpretability. Extended qualitative analysis of per-slot tsne plot can be found in Supplementary material.

Spatial Explainability As discussed in Section 3.1, the PSM block is designed not only to capture pathology-specific representations but also to generate interpretable attention maps as an inherent product of its forward pass, requiring no post-hoc approximation. Fig. 5 illustrates this capability on a representative complex CXR containing four co-occurring pathologies. Crucially, each slot attends to a distinct and anatomically faithful region: the blunt costophrenic angles slot correctly localizes attention to the lower lung bases where pleural blunting is evident; the calcified aortic knob slot concentrates activation at the upper mediastinum near the aortic arch; the right pleural efusion slot exhibits a lateralized response confined to the right lower hemithorax; and the cardiomegaly slot distributes attention broadly across the cardiac silhouette, consistent with difuse enlargement.

## 5 Ablations

Table 4: Evaluation results of Vision Concept module from its baseline model to the proposed architecture
<table><tr><td>Dataset</td><td>Methods</td><td colspan="5">Trainable Params. AUC F1-score Precision Recall</td></tr><tr><td></td><td>Baseline model</td><td>2.7M</td><td>0.64</td><td>0.485</td><td>0.42</td><td>0.574</td></tr><tr><td>MIMIC-CXR</td><td>Baseline + PSM</td><td>47.2M</td><td>0.679</td><td>0.484</td><td>0.41</td><td>0.59</td></tr><tr><td></td><td>Baseline + PSM + Shared Classifier Head</td><td>6.1M</td><td>0.77</td><td>0.467</td><td>0.44</td><td>0.497</td></tr><tr><td></td><td>Proposed(Baseline + PSM + Shared Classifier + Similarity loss)</td><td>6.1M</td><td>0.784</td><td>0.48</td><td>0.478</td><td>0.567</td></tr></table>

Table 5: Evaluation results on the impact of model scale, fine-tuning, and the proposed clinical concept integration in Concept Grounded Report Generator
<table><tr><td>Dataset</td><td>Methods</td><td colspan="6">BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE-L METEOR</td></tr><tr><td></td><td>MedGemma-4B MM zero shot</td><td>0.139</td><td>0.068</td><td>0.035</td><td>0.019</td><td>0.131</td><td>0.253</td></tr><tr><td>MIMIC-CXR</td><td>MedGemma-27B MM zero shot</td><td>0.235</td><td>0.109</td><td>0.051</td><td>0.026</td><td>0.165</td><td>0.234</td></tr><tr><td></td><td>MedGemma-4B MM Finetuned w/o Concepts (10K)</td><td>0.275</td><td>0.163</td><td>0.104</td><td>0.071</td><td>0.227</td><td>0.265</td></tr><tr><td></td><td>Proposed (MedGemma-4B Finetuned with Concepts, 10K)</td><td>0.377</td><td>0.224</td><td>0.147</td><td>0.102</td><td>0.298</td><td>0.358</td></tr></table>

Table 6: Evaluation of the Plan-Act-Observe (PAO) loop across various configurations.
<table><tr><td>System Configuration CheXbert-F1 ↑ RadGraph-F1 ↑ GREEN ↑ φ+ Ratio ↑ Mean Iter. LLM Calls ↓ Time (s) ↓</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Deterministic Baseline</td><td>0.617</td><td>0.427</td><td>0.47 ± 0.21</td><td>0.602</td><td></td><td></td></tr><tr><td>PAO Full (Cold Cache)</td><td>0.634</td><td>0.448</td><td> $0 . 6 1 \pm 0 . 2 0$ </td><td>0.803</td><td></td><td>218 295</td></tr><tr><td>PAO Full (Warm Cache)</td><td>0.634</td><td>0.448</td><td> $0 . 6 1 \pm 0 . 2 0$ </td><td>0.803</td><td>11.2</td><td>163</td></tr></table>

## 5.1 Vision model ablations

As presented in Table 4, we compare successive stages of the vision model design using standard classification metrics. The Baseline configuration corresponds to directly using the vision backbone features followed by 236 independent classification heads for multi-label prediction. Incorporating the PSM module yields a modest improvement over the Baseline while additionally enabling the generation of interpretable attention maps. The PSM variant with a shared classifier further improves performance while reducing the overall parameter count. Finally, introducing the similarity loss has minimal impact on the quantitative metrics but noticeably enhances the quality of the attention maps by encouraging more diverse, pathology-specific focus patterns.

## 5.2 Impact of Concept Grounding

As shown in Table 5, the finetuned MedGemma-4B model significantly outperforms both the 4B and 27B zero-shot baselines. This result strongly suggests that task-specific adaptation is far more critical for generating high-quality reports than raw model scale alone, making finetuning an essential step. However, the most critical comparison is between the finetuned model without concepts and our complete proposed model. As the results demonstrate, incorporating medical concepts yields performance boost, positioning our proposed model clearly ahead of all other variants. This jump highlights that the concept integration provides the crucial semantic grounding to achieve the highest level of clinical accuracy and fluency.

## 5.3 PAO Loop Ablations

The PAO ablations shown in Table 6 comprises of three diferent combinations tested on MIMIC-CXR test. For all the experiments the maximum number of refinement steps is set to 3. The deterministic baseline experiment runs all the tools sequentially and achieves a concept coverage ratio $\cdot ( \phi ^ { + } )$ of 0.602, indicating that approximately 40% of vision module predicted findings are silently omitted from the final freestyle report. Introducing the Self Correction loop(PAO Full, Cold Cache) raises $\phi ^ { + }$ to 0.803 and improves ChexBERT-F1 [50], Radgraph-F1 [19] and GREEN [43] scores, with a mean iteration count of 2.8 confirming that the loop triggers genuine re-planning in the large majority of reports rather than terminating in a single pass. While iterative repair increases mean LLM calls from 6.0 to 11.2, the warm-cache condition recovers this overhead substantially, by retrieving previously computed concept-to-ABCDEF mappings from the cache. Through caching the frequent findings, inference time is reduced from 295s to 163s, a 45% reduction while quality metrics remain unchanged.

## 6 Conclusion

We introduced MedPAO-VL, a transparent, multimodal agent designed to generate protocol-compliant, structured radiology reports. Unlike opaque end-to-end models, MedPAO-VL integrates generative capabilities with verifiable agentic reasoning to yield explicit explainability maps and auditable intermediate reasoning traces. The architecture’s three core components - the Vision Concept Module, the Concept Grounded Report Generator, and the iterative PAO loop - work in concert to preserve the inherent uncertainty of clinical observations, provide faithful visual localization, and enforce strict adherence to medical ontologies. Extensive evaluations confirm that our approach produces reliable, factually consistent, and protocol-compliant structured radiology reports. Despite these strengths, the system’s current diagnostic reach is constrained by a predefined vocabulary of 236 concept groups, limiting the documentation of ultra-rare anomalies. Future work will address this limitation by dynamically expanding the clinical vocabulary and extending the agentic framework to accommodate multi-modal imaging and broader diagnostic workflows.

## References

1. Alam, H.M.T., Srivastav, D., Kadir, M.A., Sonntag, D.: Towards interpretable radiology report generation via concept bottlenecks using a multi-agentic rag. In: Hauf, C., Macdonald, C., Jannach, D., Kazai, G., Nardini, F.M., Pinelli, F., Silvestri, F., Tonellotto, N. (eds.) Advances in Information Retrieval. pp. 201–209. Springer Nature Switzerland, Cham (2025)

2. Bannur, S., Bouzid, K., Coelho de Castro, D., Schwaighofer, A., Bond-Taylor, S., Ilse, M., Pérez-García, F., Salvatelli, V., Sharma, H., Meissen, F., Ranjit, M., Srivastav, S., Gong, J., Falck, F., Oktay, O., Thieme, A., Lungren, M.P., Wetscherek, M.T., Alvarez-Valle, J., Hyland, S.: Maira-2: Grounded radiology report generation. Tech. Rep. MSR-TR-2024-18, Microsoft (June 2024), https://www. microsoft.com/en-us/research/publication/maira-2-grounded-radiologyreport-generation/

3. Chaves, J.M.Z., Huang, S.C., Xu, Y., Xu, H., Usuyama, N., Zhang, S., Wang, F., Xie, Y., Khademi, M., Yang, Z., et al.: Towards a clinically accessible radiology foundation model: open-access and lightweight, with automated evaluation. arXiv preprint arXiv:2403.08002 (2024)

4. Chen, Z., Song, Y., Chang, T.H., Wan, X.: Generating radiology reports via memory-driven transformer. In: Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP). pp. 1439–1449 (2020)

5. Chen, Z., Varma, M., Xu, J., Paschali, M., Veen, D.V., Johnston, A., Youssef, A., Blankemeier, L., Bluethgen, C., Altmayer, S., Valanarasu, J.M.J., Muneer, M.S.E., Reis, E.P., Cohen, J.P., Olsen, C., Abraham, T.M., Tsai, E.B., Beaulieu,

C.F., Jitsev, J., Gatidis, S., Delbrouck, J.B., Chaudhari, A.S., Langlotz, C.P.: A vision-language foundation model to enhance eficiency of chest x-ray interpretation (2024), https://arxiv.org/abs/2401.12208

6. Demner-Fushman, D., Kohli, M.D., Rosenman, M.B., Shooshan, S.E., Rodriguez, L., Antani, S., Thoma, G.R., McDonald, C.J.: Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association 23(2), 304–310 (2016)

7. Deria, A., Kumar, K., Chakraborty, S., Mahapatra, D., Roy, S.: Inverge: Intelligent visual encoder for bridging modalities in report generation. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 2028–2038 (2024). https://doi.org/10.1109/CVPRW63382.2024.00208

8. Didolkar, A.R., Zadaianchuk, A., Awal, R., Seitzer, M., Gavves, E., Agrawal, A.: Ctrl-o: Language-controllable object-centric visual representation learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025)

9. Edirisinghe, D., Nimalsiri, W., Hennayake, M., Meedeniya, D., Lim, G.: Chest X-Ray report generation using abnormality guided vision language model. IEEE Access 13, 157651–157673 (2025)

10. Endo, M., Krishnan, R., Krishna, V., Ng, A.Y., Rajpurkar, P.: Retrieval-based chest x-ray report generation using a pre-trained contrastive language-image model. In: Machine learning for health. pp. 209–219. PMLR (2021)

11. Gu, D., Gao, Y., Zhou, Y., Zhou, M., Metaxas, D.: Radalign: Advancing radiology report generation with vision-language concept alignment. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 484–494. Springer (2025)

12. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645(8081), 633–638 (2025)

13. Hadjiyski, N., Kasturi, A., Vosoughi, A., Wismüller, A.: Leveraging a memorydriven transformer for eficient radiology report generation from chest x-rays to establish a quantitative metric. In: Emerging Topics in Artificial Intelligence (ETAI) 2024. vol. 13118, pp. 10–22. SPIE (2024)

14. Hong, E.K., Suh, C.h., Nukala, M., Esfahani, A., Licaros, A., Madan, R., Hunsaker, A., Hammer, M.: Radiologist interaction with ai-generated preliminary reports: A longitudinal multi-reader study. Journal of the American College of Radiology (2025)

15. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)

16. Huang, S.C., Shen, L., Lungren, M.P., Yeung, S.: Gloria: A multimodal global-local representation learning framework for label-eficient medical image recognition. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 3922– 3931 (2021). https://doi.org/10.1109/ICCV48922.2021.00391

17. Huang, Z., Zhang, X., Zhang, S.: Kiut: Knowledge-injected u-transformer for radiology report generation. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19809–19818 (2023). https://doi.org/10. 1109/CVPR52729.2023.01897

18. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al.: Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 590–597 (2019)

19. Jain, S., Agrawal, A., Saporta, A., Truong, S., Duong, D.N.D.N., Bui, T., Chambon, P., Zhang, Y., Lungren, M., Ng, A., Langlotz, C., Rajpurkar, P., Rajpurkar, P.: Radgraph: Extracting clinical entities and relations from radiology reports. In: Vanschoren, J., Yeung, S. (eds.) Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks. vol. 1. Curran (2021), https://datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/ 2021/file/c8ffe9a587b126f152ed3d89a146b445-Paper-round1.pdf

20. Jin, H., Che, H., Lin, Y., Chen, H.: Promptmrg: Diagnosis-driven prompts for medical report generation. Proceedings of the AAAI Conference on Artificial Intelligence 38(3), 2607–2615 (Mar 2024). https://doi.org/10.1609/aaai.v38i3.28038, https://ojs.aaai.org/index.php/AAAI/article/view/28038

21. Jing, B., Wang, Z., Xing, E.: Show, describe and conclude: On exploiting the structure information of chest x-ray reports. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics (2019). https://doi.org/10.18653/v1/p19- 1657, http://dx.doi.org/10.18653/v1/P19-1657

22. Jing, B., Xie, P., Xing, E.: On the automatic generation of medical imaging reports. In: Gurevych, I., Miyao, Y. (eds.) Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2577– 2586. Association for Computational Linguistics, Melbourne, Australia (Jul 2018). https://doi.org/10.18653/v1/P18- 1240, https://aclanthology.org/P18- 1240/

23. Johnson, A., Pollard, T., Mark, R., Berkowitz, S., Horng, S.: MIMIC-CXR Database. PhysioNet (Jul 2024). https://doi.org/10.13026/4jqj-jw95, https: //doi.org/10.13026/4jqj-jw95, version 2.1.0

24. Jones, J., Silverstone, L., Bell, D., et al.: Chest x-ray review: ABCDEF. https: //radiopaedia.org/articles/41125. https://doi.org/10.53347/rID-41125, reference article. Accessed: 2025-06-24

25. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept bottleneck models. In: International conference on machine learning. pp. 5338–5348. PMLR (2020)

26. Li, C.Y., Liang, X., Hu, Z., Xing, E.P.: Knowledge-driven encode, retrieve, paraphrase for medical image report generation. In: Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence. AAAI’19/IAAI’19/EAAI’19, AAAI Press (2019). https://doi.org/10.1609/aaai.v33i01.33016666, https://doi. org/10.1609/aaai.v33i01.33016666

27. Li, J., Li, S., Hu, Y., Tao, H.: A self-guided framework for radiology report generation. In: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li, S. (eds.) Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. pp. 588–598. Springer Nature Switzerland, Cham (2022)

28. Li, L., Wang, B., Verma, M., Nakashima, Y., Kawasaki, R., Nagahara, H.: Scouter: Slot attention-based classifier for explainable image recognition. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 1026–1035 (2021). https://doi.org/10.1109/ICCV48922.2021.00108

29. Li, M., Cai, W., Liu, R., Weng, Y., Zhao, X., Wang, C., Chen, X., Liu, Z., Pan, C., Li, M., et al.: Ffa-ir: Towards an explainable and reliable medical report generation benchmark. In: Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2) (2021)

30. Li, M., Lin, B., Chen, Z., Lin, H., Liang, X., Chang, X.: Dynamic graph enhanced contrastive learning for chest x-ray report generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3334– 3343 (2023)

31. Li, Y., Liang, X., Hu, Z., Xing, E.P.: Hybrid retrieval-generation reinforced agent for medical image report generation. Advances in neural information processing systems 31 (2018)

32. Liu, C., Tian, Y., Chen, W., Song, Y., Zhang, Y.: Bootstrapping large language models for radiology report generation. Proceedings of the AAAI Conference on Artificial Intelligence 38(17), 18635–18643 (Mar 2024). https://doi.org/10.1609/ aaai.v38i17.29826, https://ojs.aaai.org/index.php/AAAI/article/view/ 29826

33. Liu, F., Shareghi, E., Meng, Z., Basaldella, M., Collier, N.: Self-alignment pretraining for biomedical entity representations. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 4228–4238. Association for Computational Linguistics, Online (Jun 2021), https://www.aclweb.org/anthology/2021.naaclmain.334

34. Liu, F., Wu, X., Ge, S., Fan, W., Zou, Y.: Exploring and distilling posterior and prior knowledge for radiology report generation. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13748–13757 (2021). https://doi.org/10.1109/CVPR46437.2021.01354

35. Liu, K., Ma, Z., Kang, X., Li, Y., Xie, K., Jiao, Z., Miao, Q.: Enhanced contrastive learning with multi-view longitudinal data for chest x-ray report generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10348–10359 (2025)

36. Liu, R., Li, M., Zhao, S., Chen, L., Chang, X., Yao, L.: In-context learning for zeroshot medical report generation. In: Proceedings of the 32nd ACM International Conference on Multimedia. p. 8721–8730. MM ’24, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3664647. 3680760, https://doi.org/10.1145/3664647.3680760

37. Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., Kipf, T.: Object-centric learning with slot attention. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 11525–11538. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper\_files/paper/2020/ file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf

38. Lu, J., Xiong, C., Parikh, D., Socher, R.: Knowing when to look: Adaptive attention via a visual sentinel for image captioning. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3242–3250 (2017). https://doi. org/10.1109/CVPR.2017.345

39. Miura, Y., Zhang, Y., Tsai, E., Langlotz, C., Jurafsky, D.: Improving factual completeness and consistency of image-to-text radiology report generation. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 5288–5304 (2021)

40. Moor, M., Banerjee, O., Abad, Z.S.H., Krumholz, H.M., Leskovec, J., Topol, E.J., Rajpurkar, P.: Foundation models for generalist medical artificial intelligence. Nature 616(7956), 259–265 (2023)

41. Nie, Y., He, S., Bie, Y., Wang, Y., Chen, Z., Yang, S., Cai, Z., Wang, H., Wang, X., Luo, L., Wu, M., Wu, X., Chan, R.C.K., Lau, Y.M., Zheng, Y., Rajpurkar, P., Chen,

H.: An explainable biomedical foundation model via large-scale concept-enhanced vision-language pre-training (2025), https://arxiv.org/abs/2501.15579

42. Oikarinen, T., Das, S., Nguyen, L.M., Weng, T.W.: Label-free concept bottleneck models. In: International Conference on Learning Representations (2023)

43. Ostmeier, S., Xu, J., Chen, Z., Varma, M., Blankemeier, L., Bluethgen, C., Michalson, A.E., Moseley, M., Langlotz, C., Chaudhari, A.S., Delbrouck, J.B.: GREEN: Generative radiology report evaluation and error notation. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 374–390. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024. findings-emnlp.21, https://aclanthology.org/2024.findings-emnlp.21/

44. Park, S.J., Heo, K.S., Shin, D.H., Son, Y.H., Oh, J.H., Kam, T.E.: Dart: Diseaseaware image-text alignment and self-correcting re-alignment for trustworthy radiology report generation. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15580–15589 (2025). https://doi.org/10. 1109/CVPR52734.2025.01452

45. Peng, P., Fan, W., Shen, Y., Liu, W., Yang, X., Zhang, Q., Wei, X., Zhou, D.: Eye gaze guided Cross-Modal alignment network for radiology report generation. IEEE Journal of Biomedical and Health Informatics 28, 7406–7419 (2024)

46. Rabaey, P., Moon, J.H., Lee, J.O., Kim, M.G., Yoon, H., Demeester, T., Choi, E.: Modeling clinical uncertainty in radiology reports: from explicit uncertainty markers to implicit reasoning pathways (2026), https://arxiv.org/abs/2511. 04506

47. Rennie, S.J., Marcheret, E., Mroueh, Y., Ross, J., Goel, V.: Self-critical sequence training for image captioning. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1179–1195 (2017). https://doi.org/10.1109/ CVPR.2017.131

48. Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., Traverse, M., Kohlberger, T., Xu, S., Jamil, F., Hughes, C., Lau, C., Chen, J., Mahvar, F., Yatziv, L., Chen, T., Sterling, B., Baby, S.A., Baby, S.M., Lai, J., Schmidgall, S., Yang, L., Chen, K., Bjornsson, P., Reddy, S., Brush, R., Philbrick, K., Asiedu, M., Mezerreg, I., Hu, H., Yang, H., Tiwari, R., Jansen, S., Singh, P., Liu, Y., Azizi, S., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Riviere, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Grill, J., Ramos, S., Yvinec, E., Casbon, M., Buchatskaya, E., Alayrac, J.B., Lepikhin, D., Feinberg, V., Borgeaud, S., Andreev, A., Hardin, C., Dadashi, R., Hussenot, L., Joulin, A., Bachem, O., Matias, Y., Chou, K., Hassidim, A., Goel, K., Farabet, C., Barral, J., Warkentin, T., Shlens, J., Fleet, D., Cotruta, V., Sanseviero, O., Martins, G., Kirk, P., Rao, A., Shetty, S., Steiner, D.F., Kirmizibayrak, C., Pilgrim, R., Golden, D., Yang, L.: Medgemma technical report (2025), https://arxiv.org/abs/2507. 05201

49. Shahzadi, I., Madni, T.M., Janjua, U.I., Batool, G., Naz, B., Ali, M.Q.: Csamdt: Conditional self attention memory-driven transformers for radiology report generation from chest x-ray. Journal of Imaging Informatics in Medicine 37(6), 2825–2837 (2024)

50. Smit, A., Jain, S., Rajpurkar, P., Pareek, A., Ng, A.Y., Lungren, M.: Combining automatic labelers and expert annotations for accurate radiology report labeling using bert. In: Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP). pp. 1500–1519 (2020)

51. Srivastav, S., Ranjit, M., Pérez-García, F., Bouzid, K., Bannur, S., Castro, D.C., Schwaighofer, A., Sharma, H., Ilse, M., Salvatelli, V., Bond-Taylor, S., Falck, F., Thieme, A., Richardson, H., Lungren, M.P., Hyland, S.L., Alvarez-Valle, J.: MAIRA at RRG24: A specialised large multimodal model for radiology report generation. In: Demner-Fushman, D., Ananiadou, S., Miwa, M., Roberts, K., Tsujii, J. (eds.) Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. pp. 597–602. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/v1/2024.bionlp- 1.50, https: //aclanthology.org/2024.bionlp-1.50/

52. Tanida, T., Müller, P., Kaissis, G., Rueckert, D.: Interactive and explainable regionguided radiology report generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7433–7442 (2023)

53. Tanno, R., Barrett, D.G., Sellergren, A., Ghaisas, S., Dathathri, S., See, A., Welbl, J., Singhal, K., Azizi, S., Tu, T., et al.: Consensus, dissensus and synergy between clinicians and specialist foundation models in radiology report generation. arXiv preprint arXiv:2311.18260 (2023)

54. Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.C., Carroll, A., Lau, C., Tanno, R., Ktena, I., Palepu, A., Mustafa, B., Chowdhery, A., Liu, Y., Kornblith, S., Fleet, D., Mansfield, P., Prakash, S., Wong, R., Virmani, S., Semturs, C., Mahdavi, S., Green, B., Dominowska, E., Aguera-Arcas, B., Barral, J., Webster, D., Corrado, G., Matias, Y., Singhal, K., Florence, P., Karthikesalingam, A., Natarajan, V.: Towards generalist biomedical ai. NEJM AI (2024), https: //ai.nejm.org/doi/full/10.1056/AIoa2300138

55. Vaidya, S.S., Palani, G., Ramesh, S., Balasubramanian, V., Selvam, M., Srinivasaraja, G., Krishnamurthi, G.: Medpao: A protocol-driven agent for structuring medical reports. In: Qiu, J., Wu, J., Langlotz, C., Huang, B., Lei, Z., Wu, H., Liu, H., Xie, W. (eds.) AI for Clinical Applications. pp. 33–45. Springer Nature Switzerland, Cham (2026)

56. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

57. Wang, F., Zhou, Y., Wang, S., Vardhanabhuti, V., Yu, L.: Multi-granularity crossmodal alignment for generalized medical visual representation learning. Advances in neural information processing systems 35, 33536–33549 (2022)

58. Wang, P., Ye, S., Naseem, U., Kim, J.: Mrgagents: A multi-agent framework for improved medical report generation with med-lvlms. In: 2025 International Conference on Digital Image Computing: Techniques and Applications (DICTA). pp. 1–7. IEEE (2025)

59. Wang, Z., Liu, L., Wang, L., Zhou, L.: Metransformer: Radiology report generation by transformer with multiple learnable expert tokens. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11558–11567 (2023). https://doi.org/10.1109/CVPR52729.2023.01112

60. Wang, Z., Liu, L., Wang, L., Zhou, L.: R2gengpt: Radiology report generation with frozen llms. Meta-Radiology 1(3), 100033 (2023). https://doi.org/https: //doi.org/10.1016/j.metrad.2023.100033, https://www.sciencedirect.com/ science/article/pii/S2950162823000334

61. Wang, Z., Tang, M., Wang, L., Li, X., Zhou, L.: A medical semantic-assisted transformer for radiographic report generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 655–664. Springer (2022)

62. Wu, C., Zhang, X., Zhang, Y., Wang, Y., Xie, W.: Medklip: Medical knowledge enhanced language-image pre-training for x-ray diagnosis. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 21372–21383 (2023)

63. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., Bengio, Y.: Show, attend and tell: Neural image caption generation with visual attention. In: Bach, F., Blei, D. (eds.) Proceedings of the 32nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 37, pp. 2048–2057. PMLR, Lille, France (07–09 Jul 2015), https: //proceedings.mlr.press/v37/xuc15.html

64. Yang, L., Xu, S., Sellergren, A., Kohlberger, T., Zhou, Y., Ktena, I., Kiraly, A., Ahmed, F., Hormozdiari, F., Jaroensri, T., et al.: Advancing multimodal medical capabilities of gemini. arXiv preprint arXiv:2405.03162 (2024)

65. Yang, S., Wu, X., Ge, S., Zheng, Z., Zhou, S.K., Xiao, L.: Radiology report generation with a learned knowledge base and multi-modal alignment. Medical Image Analysis 86, 102798 (2023). https://doi.org/https://doi.org/10.1016/j. media.2023.102798, https://www.sciencedirect.com/science/article/pii/ S1361841523000592

66. Yang, S., Wu, X., Ge, S., Zhou, S.K., Xiao, L.: Knowledge matters: Chest radiology report generation with general and specific knowledge. Medical Image Analysis 80, 102510 (2022). https://doi.org/https://doi.org/10.1016/j. media.2022.102510, https://www.sciencedirect.com/science/article/pii/ S1361841522001578

67. Zhang, K., Yang, Y., Yu, J., Fan, J., Jiang, H., Huang, Q., Han, W.: Attribute prototype-guided iterative scene graph for explainable radiology report generation. IEEE Transactions on Medical Imaging 43(12), 4470–4482 (2024). https://doi. org/10.1109/TMI.2024.3424505

68. Zhang, Y., Wang, X., Xu, Z., Yu, Q., Yuille, A., Xu, D.: When radiology report generation meets knowledge graph. Proceedings of the AAAI Conference on Artificial Intelligence 34(07), 12910–12917 (Apr 2020). https://doi.org/10.1609/aaai. v34i07.6989, https://ojs.aaai.org/index.php/AAAI/article/view/6989

## Supplementary Material

## A The ABCDEF Protocol for Systematic CXR Interpretation

Chest radiograph interpretation is a foundational clinical skill that is prone to errors of omission when approached unsystematically. To mitigate this risk, the ABCDEF mnemonic has been widely adopted as a structured, sequential framework for reviewing chest X-rays (CXRs), ensuring that all anatomical regions and radiological domains are evaluated before a diagnostic impression is formed [24]. Each letter of the mnemonic corresponds to a distinct category of findings, as summarised in Table 7.

A — Airways. Evaluation begins at the midline with the trachea, which should be central at the sternal notch, with a permissible slight rightward deviation at the aortic arch. The trachea is traced inferiorly to the carina, where it bifurcates into the right and left main bronchi; the normal carinal angle is less than 105<sup>◦</sup> (mean 80<sup>◦</sup>). Deviation of the trachea toward a hemithorax may indicate ipsilateral volume loss (e.g., lobar collapse or fibrosis), while contralateral displacement suggests a space-occupying process such as a large pleural efusion or tension pneumothorax. Intraluminal masses, extrinsic narrowing, and splaying of the carina are additional findings sought at this stage.

B — Breathing (Lungs, Pulmonary Vessels, and Pleural Spaces). Both lungs should be symmetrically expanded and of similar volume. The lung parenchyma is assessed for focal consolidation, ground-glass opacification, interstitial changes, nodules, and masses. Pulmonary vascularity is examined for redistribution (upper lobe diversion) or congestion. The pleural spaces are inspected for efusion (blunting of the costophrenic angles), pneumothorax (visible pleural line with absent lung markings peripherally), or pleural thickening.

C — Circulation (Cardiomediastinal Contour and Great Vessels). The cardiac silhouette is assessed for size and morphology. On a standard posteroanterior (PA) film, the cardiothoracic ratio should not exceed 0.5; values above this threshold indicate cardiomegaly. Individual cardiac borders are inspected for clarity, as loss of definition localises underlying parenchymal pathology (e.g., right heart border efacement in right middle lobe consolidation). The mediastinal contours are reviewed for widening, masses, and lymphadenopathy. The aortic knuckle, aortopulmonary window, and hilar structures are specifically assessed, along with the pulmonary vasculature for signs of hypertension or oedema.

D — Diaphragm and Below. The right hemidiaphragm normally sits 1–2 cm higher than the left. Both domes should have smooth contours with sharp, acute costophrenic angles bilaterally. Elevation of a hemidiaphragm may indicate phrenic nerve palsy, lobar collapse, or a subphrenic process. Free subdiaphragmatic air (pneumoperitoneum) is a critical finding indicative of visceral perforation and is seen as a radiolucent crescent beneath the diaphragm on erect films. Subdiaphragmatic structures visible on the CXR—including gastric distension, hepatosplenomegaly, and abdominal calculi—are also reviewed at this stage.

E — External (Chest Wall, Bones, and Soft Tissues). The osseous structures are examined systematically: the clavicles, scapulae, ribs (posterior and anterior arches), sternum, and visible thoracic vertebrae are assessed for fractures, lytic lesions, and deformities. Rib fractures are of particular clinical significance as they may indicate an underlying pulmonary contusion or haemopneumothorax. Soft tissues are reviewed for subcutaneous emphysema, asymmetric breast shadows, and abnormal calcifications or masses.

F — Foreign Material (Devices, Lines, and Tubes). The final step involves cataloguing all implanted or inserted devices and verifying correct positioning. Endotracheal tube tips should lie 3–5 cm above the carina. Nasogastric tubes should course below the diaphragm into the gastric body. Central venous catheters should terminate at the cavoatrial junction. Pacemaker and implantable cardioverter-defibrillator (ICD) leads are traced from the generator (typically in the left infraclavicular region) to their respective cardiac chambers. Surgical clips, prosthetic heart valves, and other foreign bodies are also documented.

Table 7: Summary of the ABCDEF systematic chest radiograph review protocol.
<table><tr><td>Letter</td><td>Category</td><td>Key Findings to Assess</td></tr><tr><td>A</td><td>Airways</td><td>Tracheal position and deviation; carinal angle (&lt; 105°); main bronchi; intraluminal masses or narrowing</td></tr><tr><td>B</td><td>Breathing</td><td>Lung parenchyma (consolidation, nodules, interstitial changes); pulmonary vascularity; pleural effusion; pneumoth- orax</td></tr><tr><td>C</td><td>Circulation</td><td>Cardiothoracic ratio (&lt; 0.5 on PA); cardiac borders; medi- astinal width and contour; aortic knuckle; hilar structures</td></tr><tr><td>D</td><td>Diaphragm</td><td>Hemidiaphragm height; costophrenic angles; pneumoperi- toneum; subdiaphragmatic pathology (hepatosplenomegaly, calculi)</td></tr><tr><td>E</td><td>External</td><td>Ribs, clavicles, scapulae, vertebrae for fractures or lesions; soft tissues (subcutaneous emphysema, masses)</td></tr><tr><td>F</td><td></td><td>Foreign Material Endotracheal tubes; central venous catheters; nasogastric tubes; pacemaker/ICD leads; prostheses; surgical clips</td></tr></table>

## B Overview of MedPAO Components

The MedPAO framework [55] introduced an LLM-driven agent designed to transform unstructured, free-text Chest X-Ray (CXR) findings into standardized, protocol-compliant structured reports. Coordinated by a central LLM orchestrator, the system utilizes a suite of clinically validated tools:

1. Concept Extractor: This tool distills raw clinical text from radiologists into discrete key findings. By shifting the architecture to a concept-centric paradigm, the system significantly reduces the noise and redundancy inherent in natural language. Powered by a medical LLM fine-tuned on a mixture of expert-annotated and synthetic corpora, this tool mitigates hallucinations by outputting both the discrete concepts and their corresponding source sentences. These sentence pairings subsequently serve as explainable reasoning traces for clinical auditing.

2. Ontology Mapper: This module grounds the extracted concepts within standardized medical ontologies, specifically SNOMED CT. Operating via an external API call, it queries the ontology to retrieve a candidate set of matching terms, along with their associated descriptions and hierarchical ancestors, for each extracted clinical finding.

3. Ontology Filter: Due to the one-to-many nature of ontological mapping, a reasoning model is employed to disambiguate the candidate terms. Utilizing the contextual evidence provided by the source sentences from the Concept Extractor, this tool classifies the retrieved SNOMED CT candidates into primary and secondary matches for each concept.

4. Concept Categorizer: This reasoning module maps the filtered, grounded concepts to appropriate target categories defined by the specified clinical protocol (e.g., the ABCDEF format).

5. Structured Report Generator: A general-purpose LLM synthesizes the final output. Rather than merely concatenating the extracted findings, it leverages the categorized concepts, their source sentences, and the structural guidelines of the clinical protocol to generate a cohesive, medically accurate, and naturally phrased structured report.

The proposed MedPAO-VL architecture inherits and integrates several of these core tools for ontology grounding, concept extraction, and clinical NLP tasks. Furthermore, Fig. 6 illustrates the comparative efectiveness of the Med-PAO Agent against general-purpose LLMs on a representative CXR report.

## C Implementation Details

## C.1 Plan-Act-Observe: Technical Implementation

Semantic Matching Logic The overlap checker utilizes the SapBERT [33] model to embed both the vision-module labels and the concepts extracted from generated free-style reports. We utilize the embedding based cosine similarity metric to assess the overlapping or semantically matching concepts. It follows the algorithm shown in Algorithm 1.

![](images/2087e8c830eca73d0ec51547b73e01fb72a2ef70fd217298f74623061c4c1fcd.jpg)  
Fig. 6: Illustration of MedPAO Agent performance against other models on a hypothetical report. (In the above, the protocol placeholder represents the injected protocolrelated instructions.)

Self-Correction Prompting Strategy After performing the overlap verification, if the set of missing concepts is non-empty, the Orchestrator receives a signal from the Observe state indicating that narratives corresponding to the uncovered concepts must be generated. The orchestrator then prompts the conceptgrounded report generator with the CXR image along with the list of missing concepts, instructing it to generate narrative descriptions only for those missing findings.

Following this extended report generation step, concepts are again extracted from the updated report and the overlap checker is executed to verify whether all vision-derived concepts are now present in the generated narrative. If some concepts remain uncovered, the procedure is repeated for a predefined number of iterations N. Conversely, once the generated freestyle report successfully incorporates all concepts predicted by the vision module, the iterative loop terminates and the orchestrator proceeds to the subsequent protocol-based report generation stage.

Concept Cache Architecture and Persistence The Concept Cache is implemented as a persistent Key-Value store. Keys are the raw concepts the system has encountered previously, and values are standardized ABCDEF category assignments. The process of concept categorization is the same as proposed in [55], which has been validated on a radiologist annotated gold standard dataset. The Concept cache is updated after each successful execution of the agent(i.e., after generation of each structured report).

Algorithm 1 Identification of Missed Concepts using Semantic Similarity   
Require: Vision concepts V, Freestyle concepts F, threshold τ   
Ensure: Missed concepts list M   
M ← ∅   
for each v ∈ V do   
if predicted(v) = Absent then   
v ← “no ” ∥ v   
end if   
e<sub>v</sub> ← SapBERT(v)   
j ← 0   
for each f ∈ F do   
e<sub>f</sub> ← SapBERT(f)   
if cosine\_similarity(e<sub>v</sub>, e<sub>f</sub>) > τ then   
break   
end if   
j ← j + 1   
end for   
if j = |F| then   
M ← M ∪ {v}   
end if   
end for   
return M

Table 8 lists a subset of frequently observed concepts stored in the Concept Cache along with their corresponding ABCDEF category assignments. These concepts represent commonly occurring findings and normal observations encountered during chest X-ray report generation.

## C.2 Concept Grounded Report Generator Implementation

We used a finetuned Medgemma-4b [48] Multimodal model for the Concept Grounded Report Generation Tasks. To reduce memory overhead, we loaded the Medgemma-4b [48] in 4-bit NormalFloat (NF4) quantization. We applied Parameter-Eficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) [15], with a rank(r) of 32 and an alpha of 64, and a dropout probability of 0.05. Finetuning was performed on a 50,000-report subset for four epochs using AdamW as optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ , a cosine schedule, a batch size of 4, and gradient accumulation over 64 steps. During inference, reports were generated with a maximum of 150 new tokens. All the experiments are done with a random seed of 42. This quantized PEFT setup reduces memory consumption suficiently to fine-tune on a single NVIDIA A6000 (48 GB) GPU.

Table 8: Examples of frequently occurring concepts stored in the Concept Cache.
<table><tr><td>Concept</td><td>Category</td></tr><tr><td>Well inflated lungs</td><td>B</td></tr><tr><td>No focal airspace consolidation</td><td>B</td></tr><tr><td>No pleural effusions</td><td>B</td></tr><tr><td>No pulmonary edema</td><td>B</td></tr><tr><td>No pneumothorax</td><td>B</td></tr><tr><td>Normal cardiac contours</td><td>C</td></tr><tr><td>Normal mediastinal contours</td><td>C</td></tr><tr><td>Normal pulmonary vasculature</td><td>B</td></tr><tr><td>No acute bony abnormality</td><td>E</td></tr><tr><td>No acute osseous abnormalities</td><td>E</td></tr><tr><td>Left-sided pacemaker device</td><td>F</td></tr><tr><td>Clips in right upper quadrant of abdomen</td><td>F</td></tr><tr><td>Unchanged eventration of right hemidiaphragm D</td><td></td></tr><tr><td>Normal mediastinal and hilar contours</td><td>C</td></tr><tr><td>Normal hilar contours</td><td>C</td></tr></table>

## D Radiologist Evaluation

## D.1 Structured Report Evaluation Metrics

To assess the quality of the generated structured radiology reports, we introduced two key scoring metrics: (1) an Accuracy Score, which measures the clinical correctness of the generated report against the original MIMIC-CXR reference reports, and (2) a Structural Score, which evaluates the degree to which the generated reports adhere to the defined reporting protocol. These metrics provide a comprehensive assessment of both the medical validity and standardized formatting of the outputs. The assessment process involved presenting radiologists with paired samples - our system-generated structured reports, MedGemma-27b [48] multimodal model generated structured reports alongside their corresponding MIMIC-CXR images and reference reports. This side-by-side comparison enabled direct evaluation of report quality against established clinical standards. Radiologists evaluated each report pair blindly (without knowing which report corresponds to which method) using the standardized metrics outlined in Tables 9 and 10.

## D.2 Extended Radiologist score evaluation

Inter-rater Agreement. Inter-rater agreement ranged from fair to moderate $( \kappa =$ 0.22–0.51, quadratic weighted), consistent with published radiologist agreement rates on subjective report quality assessment. Tanno et al. [53] highlighted this inherent subjectivity by demonstrating that a panel of 27 radiologists reached unanimous agreement on report preference in only 27.4% of cases for complex datasets, reflecting the nuanced nature of clinical quality judgments. The lower κ observed for MedPAO-VL accuracy (0.222) compared to MedGemma-27B

Table 9: Scoring criteria for report quality
<table><tr><td>Score Definition</td><td></td></tr><tr><td>5</td><td>Perfect Match: The Generated Report is a perfect and complete reflec- tion of the Groundtruth Report. All key details are present, and there is no missing, extraneous, or fabricated (hallucinated) information.</td></tr><tr><td>4</td><td>Generally Accurate with Minor Omissions: The Generated Report is highly accurate and contains all key details. It may be missing a few minor, non-critical details that do not impact the overall meaning or interpretation. No hallucinations.</td></tr><tr><td>3</td><td>Key Details Present but Incomplete or with Minor Hallucinations: The Generated Report captures the main points but has significant omis- sions of secondary details. Alternatively, it might contain minor fabri- cated details that do not fundamentally contradict the groundtruth but add noise. Requires some interpretation to align with the groundtruth.</td></tr><tr><td>2</td><td>Missing Key Details or Contradictory Information: The Generated Re- port fails to mention one or more key details from the groundtruth. Alternatively, it contains information that directly contradicts the groundtruth. This report could be misleading but is not immediately dangerous.</td></tr><tr><td>1</td><td>Critically Flawed or Dangerous: The Generated Report has major omis- sions of critical information or contains significant, dangerous halluci- nations that could lead to incorrect and harmful decisions.</td></tr></table>

## Table 10: Scoring criteria for report structure

<table><tr><td></td><td>Score Definition</td></tr><tr><td>5</td><td>Perfect Structure: The report is perfectly structured. All specified Key Concepts are present as distinct sections/headers, in the correct order, and contain the relevant information as seen in the groundtruth.</td></tr><tr><td>4</td><td>Good Structure: All Key Concepts are present and correctly populated, but there might be minor deviations.</td></tr><tr><td>3</td><td>Acceptable Structure: Most Key Concepts are present, but one or two may be missing or merged. The information is generally in the right place, with some minor deviations.</td></tr><tr><td>2</td><td>Poor Structure: Multiple Key Concepts are missing. Information is dis- organized, placed under incorrect headings, making the report difficult to follow and non-compliant with the protocol.</td></tr><tr><td>1</td><td>No Structure: The report is a free-form block of text. It completely ignores the Key Concepts and shows no attempt to follow the required protocol-driven structure.</td></tr></table>

Table 11: Extended inter-rater statistics for structured report evaluation
<table><tr><td>Metric</td><td colspan="2">MedGemma-27B</td><td colspan="2">MedPAO-VL</td></tr><tr><td></td><td>Accuracy Structure</td><td></td><td>Accuracy</td><td>Structure</td></tr><tr><td colspan="5">Combined Mean  $3 . 1 8 \pm 0 . 3 6 3 . 9 8 \pm 0 . 2 4 3 . 5 0 \pm \mathbf { 0 . 2 8 4 . 2 8 } \pm \mathbf { 0 . 2 0 }$ </td></tr><tr><td>Cohen&#x27;s κ</td><td>0.510</td><td>0.394</td><td>0.222</td><td>0.369</td></tr></table>

![](images/c6f7e7c71531127b7a2da07757e612df91dbef26d2eff91d613af2b68f41fe9a.jpg)  
Fig. 7: Radiologist score distribution

(0.510) is consistent with this phenomenon: outputs of greater clinical nuance are harder to score consistently than reports with clearly identifiable deficiencies. Score distribution analysis (Fig 7) provides a complementary view: 60% of MedPAO-VL accuracy assessments received a score of 4 or 5, compared to 44% for MedGemma-27B, and 90% of MedPAO-VL structure scores reached this threshold versus 70% for MedGemma-27B. Due to the evaluation cohort size (N=50 per rater), formal significance testing is limited in power; a larger multi-reader study is planned as future work.

## E Data Curation pipeline and Analysis

## E.1 Data Curation

The data curation pipeline comprises four key stages, as illustrated in Figure 8.

1. Concept Extraction: We leverage the MedPAO concept extractor [55] to automatically identify clinical findings within radiology reports. A few-shot prompting strategy is used to query the model for candidate concepts along with the corresponding source sentences, thereby minimizing hallucinations and providing inherent explainability by linking each concept to its originating text span. If the extractor produces invalid or incomplete outputs, the model is re-prompted until a valid set of concepts is obtained. This procedure is applied to all reports, resulting in a structured collection of report–concept–evidence triplets.

![](images/ed94e49f1ba2ac0165751b17f51d38241fb554831e40e8144144d43db44c3ff6.jpg)  
Fig. 8: The entire Data preprocessing pipeline for extracting standardized concepts from clinical reports

2. Concept Segregation: Once all concepts are extracted for each report, the next step involves grouping them into Present and Absent categories to support downstream standardization. We begin by aggregating the full set of unique extracted concepts (shown in row 1 of Table 12) and pass them to a large language model with instructions to assign each concept or pathology to one of the two categories. For this classification step, we utilize the DeepSeek-R1-Distill-Llama-70B model [12]. This yields two structured outputs: a list of concepts classified as Present and another comprising those inferred as Absent. The counts of unique Present and Absent concepts are provided in rows 2 and 3 of Table 12.

3. Concept Standardization: To enable consistent normalization, we first convert all Absent concepts into their corresponding Present forms using an LLM (e.g., transforming “no pneumothorax” into “pneumothorax”). These generated Present counterparts, together with the originally extracted Present concepts, form the unified set used for standardization. For ontology mapping, we employ the state-of-the-art concept normalization model SapBERT [33], using SNOMED CT as the target ontology due to its comprehensive coverage of clinical findings. SapBERT embeds each concept and matches it to the closest SNOMED CT entry (e.g., CONCEPT ID, ANCESTORS) based on embedding similarity, producing a standardized medical term. Finally, we post-process the outputs by restoring the standardized Present counterparts back to their original Absent forms. This yields a complete set of concepts—each labeled as Present or Absent and linked to a corresponding SNOMED CT identifier.

4. Concept Filtering: After obtaining the SNOMED CT mappings for each concept along with their Present/Absent labels, we construct an inverted index in which each SNOMED CT identifier serves as a key, and its value is a dictionary grouping the corresponding Present and Absent concepts. This restructuring yields SNOMED CT–centric concept groups (the unique number of such groups is mentioned in row 4 of Table 12), each representing all variants mapped to the same ontology term. Using the original extracted concepts, we compute the occurrence frequency of each SNOMED CT group across the dataset. These frequencies are then used to filter out low-support groups based on a predefined threshold. Specifically, we discard any group with a total frequency below 100. Applying this criterion reduces the overall set of groups to 236, which serve as the final classification labels for our vision model, which can also be seen in row 5 of Table 12.

## E.2 Radiologist validation on frequency filtering

Here is the validation given by the radiologist after observing the filtered and rejected concept groups:

Table 12: Number of concepts in each categories
<table><tr><td>Categories</td><td>Number</td></tr><tr><td>Unique raw concepts</td><td>60,531</td></tr><tr><td>Present raw concepts</td><td>57,410</td></tr><tr><td>Absent raw concepts</td><td>3,123</td></tr><tr><td>Unique SNOMEDCT groups before filtering 9,450</td><td></td></tr><tr><td>Unique SNOMEDCT groups after filtering 236</td><td></td></tr></table>

## Radiologist Comment

The 236 accepted concept groups represent clinically meaningful findings commonly encountered in routine practice. The rejected concepts were predominantly technical descriptors (e.g., view type, patient positioning, redundant labels) or extremely low-frequency entities with insuficient samples for reliable learning. Excluding these low-frequency and non-pathologic concepts does not materially afect clinical usability. Their removal reduces class imbalance and noise, improving model robustness and reliability. Rare entities can be incorporated in future iterations as more data become available.

## E.3 Data Analysis

The Data Analysis part mainly includes the below 3 categories useful for the eventual data splitting and training the models.

Concept Distribution per Report: Figure 9 illustrates the distribution of both raw and standardized concepts across all reports in the dataset. As observed, the overall shape of the distributions remains similar; however, the standardized concept distribution (Figure 9b) exhibits a noticeable leftward shift, reflecting a reduced mean number of concepts per report. This reduction primarily results from the removal of low-frequency concepts during the filtering stage. Also Figure 10 represents the Top 10 raw concepts based on their frequency of occurance.

Segregation of Normal and Complex Reports: To facilitate efective training of the language decoder, we partition the dataset into Normal and Complex reports based on the number of extracted concepts. Normal reports tend to be shorter, containing fewer concepts and fewer words, whereas Complex reports exhibit higher concept density and longer narrative descriptions. Using a threshold of nine concepts, we categorize each report into one of these two groups. Figure 11 shows the word-count distributions for both categories. As expected, the Complex reports exhibit higher mean and median word counts compared to Normal reports, supporting the validity of our concept-based segmentation strategy. Summary statistics for both groups are provided in Table 13. Notably, the dataset contains fewer Complex reports, suggesting that many studies correspond to normal or near-normal findings, or that reporting styles across data sources frequently favor concise descriptions. This stratified separation is instrumental in constructing balanced train-validation-test splits for fine-tuning the grounded report generation model.

![](images/4931fe619ebd372fa67c952fe4660cf09d6e11cecc3ac0e7e5a3c18d83899edb.jpg)  
(a)

![](images/19f08e12164ae91814abeb3616ce416c7a2795616e99eec46f135df21fd93d46.jpg)  
(b)

Fig. 9: Comparison of concept distributions across all reports: (a) raw extracted concepts, and (b) standardized concepts after post-processing.  
![](images/707be74e97a1af3550fa610e9fd1a7143a26d036f060b93bbcd2e2db2c96e9e5.jpg)  
Fig. 10: Top 10 raw Concept and their Frequency

Standardized Concept Counts and Frequency Analysis: Figure 12 illustrates the distribution of raw concepts across the SNOMED CT–standardized concept groups. As shown, each group contains a varying number of concepts classified under both the Present and Absent categories. Following this grouping, a frequency-level analysis was performed to examine how often each concept group appears within the report findings across the dataset. The resulting frequency distribution is depicted in Figure 13, highlighting the variation in occurrence patterns between the Present and Absent classes within each standardized concept group.

Table 13: Statistic of Normal and Complex reports
<table><tr><td colspan="3">Normal Reports Complex Reports</td></tr><tr><td>Number of reports</td><td>47,176</td><td>6,721</td></tr><tr><td>Mean (word count)</td><td>45.6</td><td>74.0</td></tr><tr><td>Median (word count)</td><td>70</td><td>70.0</td></tr><tr><td>Std Deviation (word count) 21.5</td><td></td><td>19.0</td></tr><tr><td>1st Quartile (word count)</td><td>32.0</td><td>59.0</td></tr><tr><td>3rd Quartile (word count)</td><td>56.0</td><td>85.0</td></tr></table>

![](images/8f4c7f1c8ae87de636980ff0509e9eae242834dc70a3a85c7885f4908cef8c7c.jpg)

![](images/a6fbbd36ceb79f69217d58aef232f00a19bb114070c18231413456ccb21db3fb.jpg)  
Fig. 11: Distribution of no. of concepts across normal and complex reports in the current dataset.

![](images/cf2ee65164b505fc16d33949f75f0bd9306319bcc6fb9a2a61e22d562145cdde.jpg)  
Fig. 12: Top SNOMEDCT groups and their Present and Absent concept counts

![](images/b4436086b329cd4b05594cd0f809e3d72d2805675458e5d5d2a78fc9953740d6.jpg)  
Fig. 13: Top SnomedCT concept groups and their Frequency of occurance in the given dataset.

## F Extended Qualitative evaluation

## F.1 Emergent Clinical Co-occurrence in Latent Space

![](images/6c8eeda8834eb14db04d40ef6ff05df8c15264417105ee0108f55da595edd5d1.jpg)  
(a)

![](images/57540b44550c0573ca72150811b02d455218390f8638eb368ed64a8e55786c88.jpg)  
(b)  
Fig. 14: Heatmap showing Cosine similarity amongst the observed pathologies. (a)Cosine similarity amongst all the pathologies. (b)A cross-section of B(Breathing)

To evaluate whether the learned pathology embeddings of the Pathology Slot Module (PSM) capture real-world clinical relationships, we analyze the cosine similarity matrix of these learned pathology embeddings. As illustrated in the global heatmap encompassing all target concepts (Fig.14a), the latent space predominantly maintains orthogonality, indicated by the widespread neutral and negative similarities. This is a critical property which confirms that the slots representations successfully resisted feature collapse, thereby preserving the strict diagnostic specificity required to diferentiate distinct pathologies.

Amidst this orthogonality, the model exhibits localized blocks of high positive correlation. A high-resolution cross-section of the "Breathing" (Category B) pathologies (Fig. 14b) explicitly validates that these correlations are not noise, but rather emergent encodings of clinical co-occurrence. Rather than isolating every finding independently, the manifold tightly aligns embeddings where clinical presentation frequently overlaps. For instance, biapical scarring and bibasilar airspace opacities demonstrate a strong semantic alignment with a cosine similarity of 0.91. This efectively mirrors the common co-occurrence of these findings in patients with chronic pulmonary architectural changes, confirming that MedPAO-VL’s internal representations natively align with established medical epistemology.

## F.2 Clinical Manifold Analysis and Concept Specialization

To further investigate the discriminative power of the Pathology Slot Module (PSM), we perform per-pathology t-SNE projections of the context vectors $U _ { T }$ across the test set. A notable observation from figure 15 shows the high degree of latent separability achieved for rare but visually distinct objects, such as dual lead $l e f t$ -sided pacer devices and chest tubes. Despite the "Uncertain" class dominating the majority of the latent space, the positive clinical instances congregate in highly localized, high-density clusters. Furthermore, for acute pathologies like pulmonary edema, the model exhibits a clear class-conditional separation. The positive clusters are spatially removed from the negative background, proving that the PSM maintains diagnostic specificity even in complex, multipathological images. These analysis confirms that each slot successfully specializes in a unique visual signature, efectively disentangling medical findings from the global image features.

## F.3 Impact of Similarity loss on Attention Maps

As discussed Loss Formulation, the initial formulation of the model incorporating the Pathology Slot Module failed to produce spatially distinct attention maps for diferent pathologies. To mitigate this issue, a similarity loss $\left( L _ { s i m } \right)$ was introduced to discourage redundant attention patterns across slots. The qualitative impact of this modification is illustrated in Figure 16, which presents a side-by-side comparison of the attention maps generated with and without the inclusion of $L _ { s i m }$

In the absence of the similarity loss, the model tends to adopt a “lazy” strategy by concentrating its attention on a limited region of the image. This behavior is evident in Figure 16a, where the model consistently focuses on the top-left corner irrespective of the pathology being considered. In contrast, incorporating $L _ { s i m }$ encourages the model to distribute its attention more diversely across relevant regions.

![](images/58af65483a93cd106a2f51092c95918ae99906e72e49f12e9607a2b8fc50a35e.jpg)  
Fig. 15: Per-slot t-SNE visualizations for representative clinical findings.

A similar pattern is observed in Figure 16b, where the model without the similarity loss predominantly attends to the top-right edge of the image. The introduction of $L _ { s i m }$ alleviates this issue, resulting in more spatially meaningful attention maps that better correspond to the underlying pathologies. Therefore, beyond improving the quantitative performance of the model, the similarity loss plays a crucial role in enhancing the interpretability and quality of the generated explanation maps.

In this degenerate regime, the model satisfies the classification objective through a “lazy” strategy that ignores slot-specific distinctions, yielding attention maps that are visually indistinguishable and therefore clinically uninformative.

## G End-to-End System Walkthrough

To demonstrate the auditable reasoning capabilities of MedPAO-VL, we provide a step-by-step trace of the system processing a highly complex case. This example highlights how the agent handles diagnostic uncertainty and self-corrects semantic discrepancies before generating the final protocol-compliant report.

## Phase 1: Visual Perception

## • Module: Vision Concept Module

– Input: Raw CXR Image

– Process: Extracts visual features, projects to pathology slots, and estimates present/absent/uncertain probabilities per pathology along with spatial attention maps. The attention maps extracted for this particular sample under consideration is shown in Fig 17

no tiny pleural effusion

![](images/9fcec4e03fd793dfaa2fea7f7c32ea821d908cf6a13dd8155f32295fab31d106.jpg)  
(b)  
Fig. 16: Attention map comparison between models with and without $L _ { s i m }$

![](images/58582272f75234d980e0b95a3d463b4a5618d7785a4f0f86a04a2d994d65ac9a.jpg)  
Input CXR

![](images/59e596bdd7f32e57c3f7fd957f530d845d06bc5786d71cf12f5c6960f7a778c6.jpg)

![](images/be5c2dc3ae018dfea19e580eb5e4c40fd0bd1a58f3d1f7f6d307df2c10587513.jpg)

![](images/3aa4d83789ce8ebb16a6d967fac5a121f4adf68171cc117641125baf38f2c1f3.jpg)  
Fig. 17: Input CXR and the explainable heatmaps from Vision Concept Module

![](images/609b9dda6f7082fd49a46b15a3895653ca9118ab8bb57bc1935023c443c668e2.jpg)  
no pneumothorax

Output: Present: clear lungs, normal cardiomediastinal silhouette, status post sternotomy and coronary artery bypass graft surgery; Absent: tiny pleural efusion, pneumothorax

## Phase 2: Initial Report Generation

## • Module: Concept-Grounded Report Generator

– Input: CXR Image; Present: clear lungs, normal cardiomediastinal silhouette, status post sternotomy and coronary artery bypass graft surgery; Absent: tiny pleural efusion, pneumothorax

– Process: Generates initial narrative findings based strictly on the grounded visual concepts and CXR analysis.

– Output:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are stable. Median   
sternotomy wires and mediastinal clips are noted.

## Phase 3: Self Correction Loop Iteration 1

## • Module: Concept Extractor – Input:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are unremarkable.   
Median sternotomy wires and mediastinal clips are noted.   
There is no pleural effusion or pneumothorax.

– Process: Extracts clinical findings and corresponding source sentences from the previously generated report.

– Output:

clear lungs: The lungs are well expanded and clear.   
stable cardiomediastinal contours: Cardiomediastinal   
and hilar contours are stable.   
stable hilar contours: Cardiomediastinal and hilar   
contours are stable.   
sternotomy wires: Median sternotomy wires and   
mediastinal clips are noted   
mediastinal clips: Median sternotomy wires and   
mediastinal clips are noted

## • Module: Overlap Checker

Input: two concept sets to compare

Report concepts:

[clear lungs, stable cardiomediastinal contours, stable   
hilar contours, sternotomy wires, mediastinal clips]

## Vision concepts:

[clear lungs, normal cardiomediastinal silhouette, status post sternotomy and coronary artery bypass graft surgery, no tiny pleural effusion, no pneumothorax]

– Process: Utilizes SapBERT for embedding-based matching between the concepts extracted from the report and the concepts from the vision module, and returns the missing concepts.

– Output:

[no tiny pleural effusion, no pneumothorax]

## • Module: Re-prompt Report Generator

– Input:

Previous Report:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are unremarkable.   
Median sternotomy wires and mediastinal clips are noted.

## Missing Concepts:

[no tiny pleural effusion, no pneumothorax]

– Process: Takes the previously generated report and the set of missing concepts from the vision module, and generates the missing piece of the report.

Output:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are unremarkable.   
Median sternotomy wires and mediastinal clips are noted.   
There is no pleural effusion or pneumothorax.

Iteration 2

## • Module: Concept Extractor

– Input:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are unremarkable.

Median sternotomy wires and mediastinal clips are noted.   
There is no pleural effusion or pneumothorax.

– Process: Extracts clinical findings and corresponding source sentences from the previously generated report.

– Output:

clear lungs: The lungs are well expanded and clear.   
stable cardiomediastinal contours: Cardiomediastinal   
and hilar contours are stable.   
stable hilar contours: Cardiomediastinal and hilar   
contours are stable.   
sternotomy wires: Median sternotomy wires and   
mediastinal clips are noted.   
mediastinal clips: Median sternotomy wires and   
mediastinal clips are noted.   
no pleural effusion: There is no pleural effusion or   
pneumothorax.   
no pneumothorax: There is no pleural effusion or   
pneumothorax.

## • Module: Overlap Checker

– Input: two concept sets to compare — Report concepts:

[clear lungs, stable cardiomediastinal contours, stable hilar contours, sternotomy wires, mediastinal clips, no pleural effusion, no pneumothorax]

## Vision concepts:

[clear lungs, normal cardiomediastinal silhouette, status post sternotomy and coronary artery bypass graft surgery, no tiny pleural effusion, no pneumothorax]

– Process: Utilizes SapBERT for embedding-based matching between the concepts extracted from the report and the concepts from the vision module, and returns the missing concepts.

– Output: [] → Loop exits

## Phase 4: Report Structuring

• Module: Concept Extractor

– Input:

FINDINGS: The lungs are well expanded and clear.   
Cardiomediastinal and hilar contours are unremarkable.   
Median sternotomy wires and mediastinal clips are noted.   
There is no pleural effusion or pneumothorax.

– Process: Extracts clinical findings and corresponding source sentences from the previously generated report.

## – Output:

clear lungs: The lungs are well expanded and clear.   
stable cardiomediastinal contours: Cardiomediastinal   
and hilar contours are stable.   
stable hilar contours: Cardiomediastinal and hilar   
contours are stable.   
sternotomy wires: Median sternotomy wires and   
mediastinal clips are noted.   
mediastinal clips: Median sternotomy wires and   
mediastinal clips are noted.   
no pleural effusion: There is no pleural effusion or   
pneumothorax.   
no pneumothorax: There is no pleural effusion or   
pneumothorax.

## • Module: Ontology Mapper – Input:

[clear lungs, stable cardiomediastinal silhouette, stable   
hilar contours, sternotomy wires, mediastinal clips, no   
tiny pleural effusion, no pneumothorax]

– Process: Maps each of the extracted concepts to the SNOMED CT ontology to obtain further clinical information for structural grounding in text.

## – Output:

```json
[
{
"query": "clear lungs",
"concept_id": "444859001",
"matched_surface": "Lungs in normal
arrangement",
"preferred_desc": "Lungs in normal
arrangement (finding)",
"all_descs": [
"Lungs in normal arrangement (finding)",
```

"Lungs in normal arrangement"   
],   
"ancestors": [   
"Normal lung (finding)",   
"Normal respiratory system (finding)",   
"Lung finding (finding)",   
"Respiratory finding (finding)",   
"Finding of thoracic region (finding)",   
"Lower respiratory tract finding (   
finding)"   
]   
},   
{   
"query": "stable cardiomediastinal   
silhouette",   
"concept\_id": "233819005",   
"matched\_surface": "Stable angina (   
disorder)",   
"preferred\_desc": "Stable angina (disorder   
)",   
"all\_descs": [   
"Stable angina (disorder)",   
"Stable angina"   
],   
"ancestors": [   
"Chronic ischemic heart disease (   
disorder)",   
"Angina (disorder)",   
"Chronic chest pain (finding)",   
"Ischemic heart disease (disorder)",   
"Chronic heart disease (disorder)"   
]   
},<sub>{</sub>   
"query": "no pneumothorax",   
"concept\_id": "49756002",   
"matched\_surface": "Perinatal pneumothorax   
(disorder)",   
"preferred\_desc": "Perinatal pneumothorax   
(disorder)",   
"all\_descs": [   
"Perinatal pneumothorax (disorder)",   
"Perinatal pneumothorax"   
],   
"ancestors": [   
"Pneumothorax (disorder)"   
"Neonatal respiratory system disorder (   
disorder)"

"Fetal intrapartum and/or early   
neonatal disorder (disorder)"   
"Disorder of pleura AND/OR pleura   
cavity (disorder)"   
"Neonatal disorder (disorder)"   
]   
},   
]

## • Module: Concept Categorizer – Input: Concept-source sentence:

clear lungs: The lungs are well expanded and clear.   
stable cardiomediastinal contours: Cardiomediastinal   
and hilar contours are stable.   
stable hilar contours: Cardiomediastinal and hilar   
contours are stable.   
sternotomy wires: Median sternotomy wires and   
mediastinal clips are noted.   
mediastinal clips: Median sternotomy wires and   
mediastinal clips are noted.   
no pleural effusion: There is no pleural effusion or   
pneumothorax.   
no pneumothorax: There is no pleural effusion or   
pneumothorax.

## Ontology information:

[   
{   
"query": "clear lungs",   
"concept\_id": "444859001",   
"matched\_surface": "Lungs in normal   
arrangement",   
"preferred\_desc": "Lungs in normal   
arrangement (finding)",   
"all\_descs": [   
"Lungs in normal arrangement (finding)",   
"Lungs in normal arrangement"   
],   
"ancestors": [   
"Normal lung (finding)",   
"Normal respiratory system (finding)",   
"Lung finding (finding)",

"Respiratory finding (finding)",   
"Finding of thoracic region (finding)",   
"Lower respiratory tract finding (   
finding)"   
]   
},   
{   
"query": "stable cardiomediastinal   
silhouette",   
"concept\_id": "233819005",   
"matched\_surface": "Stable angina (   
disorder)",   
"preferred\_desc": "Stable angina (disorder   
)",   
"all\_descs": [   
"Stable angina (disorder)",   
"Stable angina"   
],   
"ancestors": [   
"Chronic ischemic heart disease (   
disorder)",   
"Angina (disorder)",   
"Chronic chest pain (finding)",   
"Ischemic heart disease (disorder)",   
"Chronic heart disease (disorder)"   
]   
},   
{   
"query": "no pneumothorax",   
"concept\_id": "49756002",   
"matched\_surface": "Perinatal pneumothorax   
(disorder)".   
"preferred\_desc": "Perinatal pneumothorax   
(disorder)",   
"all\_descs": [   
"Perinatal pneumothorax (disorder)",   
"Perinatal pneumothorax"   
],   
"ancestors": [   
"Pneumothorax (disorder)"   
"Neonatal respiratory system disorder (   
disorder)"   
"Fetal intrapartum and/or early   
neonatal disorder (disorder)"   
"Disorder of pleura AND/OR pleural   
cavity (disorder)"   
"Neonatal disorder (disorder)"   
]

– Process: Takes each concept and its ontology information(matched terms, description and the ancestors), along with the protocol (ABCDEF) information, as input and categorizes each concept into an ABCDEF category.

## – Output:

```yaml
clear lungs: B
stable cardiomediastinal contours: C
stable hilar contours: C
sternotomy wires: F
mediastinal clips: F
no pleural effusion: B
no pneumothorax: B
```

## • Module: Structured Report Generator

– Input: Categorized concepts:

```yaml
clear lungs: B
stable cardiomediastinal contours: C
stable hilar contours: C
sternotomy wires: F
mediastinal clips: F
no pleural effusion: B
no pneumothorax: B
```

## Concept-source sentence:

clear lungs: The lungs are well expanded and clear.   
stable cardiomediastinal contours: Cardiomediastinal   
and hilar contours are stable.   
stable hilar contours: Cardiomediastinal and hilar   
contours are stable.   
sternotomy wires: Median sternotomy wires and   
mediastinal clips are noted.   
mediastinal clips: Median sternotomy wires and   
mediastinal clips are noted.   
no pleural effusion: There is no pleural effusion or   
pneumothorax.

no pneumothorax: There is no pleural effusion or pneumothorax.

Process: Takes the categorized set of concepts, along with the concepts and their corresponding source sentences, and outputs a structured findings section of the report.

## – Output:

A: No findings

B: The lungs are well expanded and clear with no pleural effusion or pneumothorax.

C: Cardiomediastinal and hilar contours are stable.

D: No findings

E: No findings

F: Median sternotomy wires and mediastinal clips are noted.

## H LLM prompts

MedPAO-VL follows a PAO loop logic and strategically employs the LLM engine during both the Plan and Observe states of the agent’s operational cycle. In the Plan state, the MCP client queries the LLM by providing contextual information including available tools and any previously executed steps, prompting (Figure 18) the engine to determine the next appropriate tool along with its specific parameters. During the Observe state, the system presents (Figure 19) the LLM with both the tool-generated output and the original user query, requesting an evaluation of whether the response adequately addresses the initial request. Based on this assessment, the LLM decides whether to continue iterating through the cycle or terminate the process and return the current output as final. All the individual tool calls of the system uses diferent prompt to carry out the tasks: Concept Extraction (Figure 20), Concept categorization (Figure 21) and the Structured report generation (Figure 22). The Concept Grounded Report Generator generates a fluent, unstructured narrative from the discrete clinical findings and the image. The generation process is designed as a "concept-driven" synthesis, utilizing a finetuned multimodal model. The prompt used to generate this report is depicted in Figure 23. Furthermore, the same concept grounded report generator is also used self refinement loop to generate the report of missing findings, as indicated by the prompt in Fig 24 .

Planning Prompt with Self-Correction   
You are in FULL PIPELINE PLANNING mode of a Self-Correction PAO   
agent.   
USER’S ORIGINAL QUERY: {state.user\_query}   
AVAILABLE TOOLS: {tools\_description}   
TASK: Create a complete pipeline of actions to fully satisfy the   
user’s query. Be prepared to perform self-correction and replanning   
when prompted by the Observe phase.   
You must respond with a JSON array of action steps in the following   
format:   
[   
{   
"step\_number": 1,   
"tool\_name": "tool-name",   
"arguments": {   
"argument-name": "value"   
},   
"description": "What this step accomplishes"   
},   
{   
"step\_number": 2,   
"tool\_name": "another-tool",   
"arguments": {   
"argument-name": "value"   
},   
"description": "What this step accomplishes"   
}   
1   
Requirements:   
1. Plan all steps required to complete the user’s request.   
2. Ensure each step logically follows from the previous one.   
3. Select the appropriate tool for each step.   
4. Provide valid arguments matching the corresponding tool schema.   
5. Include a concise description of the purpose of each step.   
6. Ensure the resulting pipeline fully satisfies the user’s   
request.   
7. If the Observe phase requests a REPLAN, generate an updated   
pipeline accordingly.   
IMPORTANT: Respond ONLY with the JSON array. Do not include   
explanations, markdown, or any additional text.  
Fig. 18: The system prompt used for the Planning Agent.

Observation Prompt with Replanning   
# Dynamic Context Injection:   
mode\_context = "This was step X of Y in the planned pipeline."   
You are operating in the OBSERVE phase of a Plan–Act–Observe (PAO)   
agent.   
USER QUERY: {state.user\_query}   
CURRENT PLAN MODE: {state.planning\_mode}   
ACTION EXECUTED: {state.action\_taken}   
TOOL OUTPUT: {tool\_result}   
ALL PREVIOUS RESULTS: {state.action\_results if state.action\_results   
else ’None’}   
{mode\_context}   
TASK   
Analyze the result of the executed tool and determine the correct   
next action.   
You must evaluate:   
1. Whether the tool execution was successful.   
2. Whether the output contains the expected information.   
3. Whether the user’s goal has been fully achieved.   
4. Whether regeneration or self-correction is required.   
Special handling rules   
– If the comparison tool reports missing concepts, instruct the   
system to regenerate narratives for those missing concepts.   
– If a tool fails or produces invalid output, retry the same step   
with corrected arguments.   
– If the current plan is no longer valid due to unexpected   
outputs, request REPLAN.   
– If all required concepts are covered and a structured report is   
generated, mark the task as complete.   
Respond strictly in the following format:   
ANALYSIS:   
Your reasoning about the result.   
GOAL ACHIEVED:   
YES or NO.   
NEXT ACTION:   
One of the following:   
– Execute the next planned step.   
– Retry the previous tool with corrected input.   
– Regenerate missing concept narratives.   
– REPLAN if the current plan is invalid.   
– COMPLETE if the goal has been achieved.  
Fig. 19: Observation prompt used by the PAO agent to analyze tool outputs and decide the next step, including retry and replanning mechanisms.

Concept Extraction Few-Shot Prompt   
INSTRUCTION: Assume you are an expert radiology practitioner. You   
have to extract the medical concepts from the given report and   
indicate which sentence each concept came from.   
# Example 1 (Negative findings)   
Extracted concepts and their corresponding sentence from "No focal   
consolidation, pleural effusion, or pneumothorax is seen. Heart and   
mediastinal contours are within normal limits. There is no evidence   
for pulmonary edema." are   
{   
"no focal consolidation": "No focal consolidation, pleural   
effusion, or pneumothorax is seen.",   
"no pleural effusion": "No focal consolidation, pleural   
effusion, or pneumothorax is seen.",   
"no pneumothorax": "No focal consolidation, pleural effusion, or   
pneumothorax is seen.",   
"normal heart contours": "Heart and mediastinal contours are   
within normal limits.",   
"no pulmonary edema": "There is no evidence for pulmonary   
edema."   
} ##   
[... Examples 2, 3, and 4 omitted for brevity ...]   
# Example 5 (Complex pathology)   
Extracted concepts and their corresponding sentence from "Cardiac   
silhouette remains moderately enlarged slightly increased from   
prior exam. There has been interval increase in central pulmonary   
vascular en or ement as well as interstitial edema. A focal ri ht   
lower lung consolidation has increased in severity and is worrisome   
for pneumonia. [...]" are   
{   
"enlarged cardiac silhouette": "Cardiac silhouette remains   
moderately enlarged slightly increased from prior exam.",   
"pulmonary vascular engorgement": "There has been interval   
increase in central pulmonary vascular engorgement as well as   
interstitial edema.",   
"interstitial edema": "There has been interval increase in   
central pulmonary vascular engorgement as well as interstitial   
edema.",   
"pneumonia": "A focal right lower lung consolidation has   
increased in severity and is worrisome for pneumonia.",   
"no pleural effusion": "There is no large pleural effusion or   
pneumothorax."   
} ##   
TARGET QUERY:   
Extracted concepts and their corresponding sentence from   
{medical\_final} are  
Fig. 20: The few-shot prompt used for extracting clinical concepts and mapping them to their source sentences. The model is conditioned with 5 diverse examples ranging from normal findings to complex pathologies.

ABCDEF Categorization Prompt   
SYSTEM ROLE: You are a radiology expert with the following   
knowledge:   
<ABCDEF Protocol Description>   
TASK: You will be given:   
1. A concept-source sentence extracted from a radiology report.   
2. A list of medical concepts with their ontology information.   
3. A protocol category.   
Your task is to categorize each medical concept into one of   
the protocol categories: A, B, C, D, E, or F. Use both the   
concept-source sentence for contextual understanding, and the   
ontology information (description and ancestors) for semantic   
understanding.   
Ontology information follows the structure:   
{   
"concept": "<medical concept>","description":"<ontology description>"   
"ancestors": ["<ancestor 1>","<ancestor 2>",...]   
}   
Carefully analyze the concept, its ontology description, its   
ancestors, and the report context before assigning a category.   
TARGET QUERY — now categorize all concepts.   
Concept-source sentence:   
{concepts\_report}   
Ontology data:   
{ontology\_data}   
OUTPUT FORMAT: Return ONLY a valid JSON object in the format:   
{   
"<concept\_1>": "<category>",   
"<concept\_2>": "<category>"   
  
Do not provide explanations, reasoning, markdown, or additional   
text.  
Fig. 21: The prompt used to classify extracted concepts into the standard ABCDEF radiological schema. The <ABCDEF Protocol Description> placeholder contains same protocol information as shown in Fig 22

![](images/e1562e1871f9b60e54a8e1fa8778601f9c036a73833255da0300942c554f0e83.jpg)  
Fig. 22: The prompt used to generate the final structured radiology report. The model is conditioned on the ABCDEF protocol to ensure clinical completeness and is explicitly constrained to avoid hallucinating information not present in the input concepts.

![](images/3b052bcf05e05f282d21533ff5bd8cb6bc79254c9e4e69ccbccf58375b1d6c31.jpg)  
Fig. 23: The multimodal prompt for concept grounded radiology report generation. The model receives a chest X-ray image and a textual prompt containing observed clinical concepts (pre-processed to list present and absent findings), and is tasked with generating a comprehensive report.

![](images/7a458200e76de0f26ccd71d674129d4174cc5b6dad216e74ef4fd555f66bd001.jpg)  
Fig. 24: The prompt used in the re-prompting step of the self-correction loop. Given the previously generated report and the set of concepts flagged as missing by the overlap checker, the model is constrained to generate only the missing findings, without repeating or regenerating the full report.