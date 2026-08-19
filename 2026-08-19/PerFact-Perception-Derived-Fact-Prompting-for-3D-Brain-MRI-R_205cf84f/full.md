# PerFact: Perception-Derived Fact Prompting for 3D Brain MRI Report Generation

Jianyu Sun<sup>1,\*</sup> Zhenxuan Zhang<sup>1,2,\*</sup> Guang Yang<sup>1,2,3,†</sup> Peter J. Lally<sup>1,†</sup>

<sup>1</sup>Department of Bioengineering, Imperial College London, London, UK

<sup>2</sup>Imperial-X, Imperial College London, London, UK

<sup>3</sup>School of Biomedical Engineering & Imaging Sciences, King’s College London, London, UK

<sup>\*</sup>These authors contributed equally to this work. <sup>†</sup> Corresponding authors.

## Abstract

Radiology report generation has matured almost entirely on 2D chest radiographs, where the default route to better reports is a larger backbone or a pre-training one on medical data. We revisit that assumption on 3D multi-sequence brain MRI, a volumetric multi-disease regime, and find that the model is not the lever. Zero-shot medical and radiology vision-language models transfer poorly to brain MRI, with chest radiograph specialists failing most conspicuously, and five backbones fine-tuned identically across three model families and an order of magnitude in scale difer only marginally. What determines the quality of the report is the information injected into the prompt. We delegate perception to upstream 3D segmentation and classification, serialize their outputs into a structured fact sentence, and prompt a LoRA-adapted vision-language model with it; we call this PerFact. In a controlled study that fixes the backbone, data split, target reports, and adaptation while varying only the injected grounding, perception-derived facts outperform retrieved prior reports, retrieval becomes redundant once facts are present, and end-to-end predicted facts remain efective without any ground-truth annotation at inference. The residual gap between predicted and oracle facts is explained by the granularity of the facts rather than by the generator. Closed-ended visual question answering comes at no measurable cost to report quality, though the grounding source has little efect on it. On 3D brain MRI, grounding information, not model choice, is the dominant controllable factor in report quality.

## Introduction

A brain Magnetic Resonance Imaging (MRI) report is read for a few findings that change management: where a lesion is, which side it is on, whether structures are displaced and in which direction, and what the lesion most likely is. A vision-language model is asked for those findings from three slices of a study that holds several hundred (Figure 1). Fluent text that is wrong on any of these carries no mark of its own unreliability. Figure 2 shows released systems describing the wrong organ, placing a right frontal glioblastoma in the left hemisphere, and inverting the direction of midline shift, all in well-formed prose.

Automated radiology report generation has matured on a single modality: the systems that define the state of the art were built, trained and validated on 2D chest radiographs (Bannur et al. 2024; Zambrano Chaves et al. 2025; Chen et al. 2024b). Brain MRI is a diferent regime. A study comprises several co-registered 3D volumes with diferent tissue contrasts rather than one projection, findings are volumetric and lateralized, and a realistic corpus spans diseases as different as glioma, meningioma, acute stroke and white-matter hyperintensity. Recent work shows that a health-system-scale neuroimaging archive can support strong report generation (Kondepudi et al. 2026); what drives report quality in this regime, at the corpus sizes available to most groups, has not been established.

(a) Three of 496 slices reach the model  
![](images/04af37d90f3756519bbc0522b2dce546ba2dc3d387868b41950d2a9f40b060fc.jpg)

(b) PerFact adds one machine-generated fact sentence  
![](images/a44ef3ec15d50dc00ed21617896316bc695d06e2990a7df9ff2bb48b1d0c95a3.jpg)  
Figure 1: The information gap PerFact closes. This acute stroke study holds 496 axial slices and the prompt carries three of them, 0.6% of the study, with a fixed instruction. The fact sentence comes from an upstream model that reads all 496; the vision-language model still sees only the same three slices.

Two routes dominate attempts to improve such systems, and we find that neither is where the gain lies. The first is the model, either a larger backbone or pretraining on medical data: six released medical and radiology vision-language models degrade sharply zero-shot on brain MRI, the chestspecialized ones producing thoracic sentences for a brain study, and across five backbones fine-tuned identically, from

![](images/a9b6129d63e6a09a247729ff30014f7df49954b09960aa05a0e1f2e424d4428d.jpg)  
Injected fact (PerFact only): right frontal lobe; ring enhancement; edema; midline shift; ventricle compression; glioblastoma. No direction given.

<table><tr><td>Setting</td><td>Verbatim output on this study</td></tr><tr><td>Reference</td><td>Abnormal signal in the right frontal lobe . . . midline shifted to the left . . . glioblastoma.</td></tr><tr><td>CheXagent ×</td><td>There is no evidence of intracranial hemorrhage or mass effect. (all of it)</td></tr><tr><td>MAIRA-2 ×</td><td>. . . the endotracheal tube terminating in the right mainstem bronchus</td></tr><tr><td>LLaVA-Rad ×</td><td>. opacities within the right middle and lower lobes, an infectious process</td></tr><tr><td>LLaVA-Med-1.5×</td><td>. a large left frontal arachnoid cyst, a small left frontal subdural hematoma..</td></tr><tr><td>Qwen2.5-VL (zero-shot)×†</td><td>. a well-defined mass in the left parietal lobe . . . no mass effect or midline shift . .</td></tr><tr><td>+ Plain SFT ×</td><td>. . the left frontal region . . . a slight rightward shift of the midline . . . meningioma.</td></tr><tr><td>+ PerFact (ours)√</td><td>. . . the right frontal lobe . . . leftward midline shift . . . Glioblastoma.</td></tr></table>

Figure 2: Injected facts correct a hallucinated location (case BraTS-GLI-00599). Left: four sequences of one slice, radiological orientation, reference segmentation outlined, with the fact given to PerFact. Right: verbatim output, . . . marking an elision, green agreeing with the reference and orange contradicting it. The lower block varies only the injected text. † A diferential and a management plan rather than a report. ‡ A thoracic structure.

3B to 32B parameters in three model families, clinical entity F1 score varies by only 0.015 once the same facts are injected. The second is retrieval, conditioning on similar prior reports or on curated knowledge: prior reports supply the form of a report rather than its content, and retrieved knowledge is no better than plain fine-tuning. Neither supplies what the generator is missing, a reliable statement of what is in the images.

What determines report quality is instead the information supplied to the model. We delegate perception to upstream 3D models, running segmentation and classification on the volumes, serialize their outputs into a compactfact sentence, and leave the vision-language model the task it performs reliably: rendering structured facts as fluent clinical prose. We call this PerFact, for perception-derived facts. Holding backbone, split, target reports, adaptation recipe and decoding fixed and varying only what is injected into the prompt, clinical entity F1 score increases from 0.644 with images alone to 0.775 with predicted facts, against an oracle ceiling of 0.879. That span of 0.235 is more than an order of magnitude beyond the 0.015 spanned by changing the model. Our contributions are summarized as follows:

• We present a controlled study of grounding on 3D brain MRI, over four diseases and three image contrasts, for both report generation and closed-ended Visual Question Answering (VQA), varying only the injected auxiliary information.

• We show that model choice has minimal impact: report F1 score spans 0.015 across five backbones fine-tuned identically on the same facts, while six zero-shot medical and radiology vision-language models, most of all the chest radiology report generators, fail on brain MRI.

• We show that the injected information has the greatest impact: F1 score spans 0.235 across the grounding conditions; predicted facts outperform retrieval, retrieval is redundant once facts are present, and predicted facts recover about half the oracle gain.

• We identify where further efort is productive: no retrieval design survived reseeding, and the residual oracle gap is set by the granularity of the fact schema rather than by the generator.

## Related Work

Radiology report generation: Radiology report generation grew out of image captioning (Vinyals et al. 2015) and matured on chest radiographs, first with encoderdecoder and memory-driven architectures (Jing, Xie, and Xing 2018; Chen et al. 2020, 2021) and more recently with report-specialized vision-language models such as MAIRA-2, LLaVA-Rad, and CheXagent (Bannur et al. 2024; Zambrano Chaves et al. 2025; Chen et al. 2024b). Benchmarks (Johnson et al. 2019; Demner-Fushman et al. 2016), entity extractors (Jain et al. 2021), and clinical-eficacy metrics (Smit et al. 2020) were all designed around that modality. Volumetric work is thinner and diferently shaped: the 3D radiology models that exist mostly target general-purpose captioning (Wu et al. 2025), while neuroimaging report generation has been approached either from a single disease family (Lei et al. 2024) or by pretraining on one health system’s uncurated archive of millions of volumes (Kondepudi et al. 2026). Neither settles which grounding signal drives report quality at ordinary corpus sizes, and we test the transfer of chest-trained systems rather than assuming it.

Grounding generation in facts and in retrieval: A recurring remedy for hallucination is to condition generation on structured intermediate representations: detected attributes (Jin et al. 2024), anatomical graphs (Zhang et al. 2020), or knowledge injected from a curated ontology (Huang, Zhang, and Zhang 2023; Liu et al. 2021). Retrieval (Lewis et al. 2020) is the other common source of context, and medical variants retrieve either similar prior cases or curated domain knowledge (Xia et al. 2025, 2024). Both families usually compare one conditioned model against an unconditioned baseline, and their interaction is unexamined because the two are deployed in isolation. We contribute not a new conditioning mechanism but a controlled measurement of what each signal is worth against the alternatives.

## Method

## Problem Formulation

Let the dataset be $\mathcal { D } = \{ ( X _ { i } , y _ { i } , \mathcal { Q } _ { i } ) \} _ { i = 1 } ^ { N }$ , where $X _ { i } \ =$ $\{ x _ { i } ^ { ( m ) } \} _ { m = 1 } ^ { M _ { i } }$ is a brain MRI study with a variable number of co-registered 3D volumes with diferent image contrasts, y<sub>i</sub> is the target radiology report, and $\mathcal { Q } _ { i } = \{ ( q _ { i , j } , a _ { i , j } ) \} _ { i = 1 } ^ { J _ { i } }$ is a set of closed-ended VQA pairs. Given $X _ { i } ,$ , the model must generate a report $\hat { y } _ { i }$ and, when VQA supervision is included, predict the correct answer tokens for each question. We condition generation on an auxiliary grounding signal G<sub>i</sub> and model the report distribution as $\dot { p } _ { \theta } ( y _ { i } \mid X _ { i } , \bar { \mathcal { G } } _ { i } )$ . The central question is which choice of $\mathcal { G } _ { i }$ best improves p<sub>θ</sub> under a fixed backbone, split, target report set, and decoding strategy. We write $d _ { i }$ for the disease family of case i, T for the training split, and B for the curated neuroimaging knowledge base.

## PerFact Pipeline

PerFact decomposes the problem into two stages (Figure 3). First, upstream 3D perception maps the study to a structured latent summary. Second, that summary is serialized into a short fact sentence and used as prompt-level grounding for a vision-language generator. We write the upstream predictor as $z _ { i } = \bar { h } _ { \phi } ( \bar { X } _ { i } )$ , where $z _ { i }$ collects segmentation masks, lesion attributes, and disease labels predicted from the volumes. A deterministic serializer $\psi ( \cdot )$ then converts these outputs into a fact sentence $s _ { i } = \psi ( z _ { i } )$ . This sentence is the factual interface between the perception model and the report generator. When ground truth is available for analysis, we also define an oracle sentence $s _ { i } ^ { * } = \psi ( z _ { i } ^ { * } )$ , which upper bounds the efect of conditioning without requiring inference-time annotation.

## Upstream 3D Perception

Perception is trained on the full 3D study rather than on the 2D slices consumed by the generator. The upstream network is a multi-task model trained on four pre-tasks: lesion segmentation, missing-modality completion, image-text matching, and attribute classification. If we denote the corresponding losses by $\mathcal { L } _ { \mathrm { s e g } } , \mathcal { L } _ { \mathrm { c m p } } , \mathcal { L } _ { \mathrm { i t m } } ,$ and $\mathcal { L } _ { \mathrm { c l s } } .$ , the upstream objective can be written in the generic weighted form

$$
\mathcal { L } _ { \mathrm { u p } } = \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { c m p } } \mathcal { L } _ { \mathrm { c m p } } + \lambda _ { \mathrm { i t m } } \mathcal { L } _ { \mathrm { i t m } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } ,\tag{1}
$$

where the $\lambda$ terms control the relative contribution of each pre-task. Training follows a three-stage curriculum $s _ { 1 } $ $\bar { \cal S } _ { 2 } \  \ { \cal S } _ { 3 } ,$ , where $S _ { 1 }$ is image-only pretraining, $S _ { 2 }$ adds image-text alignment, and $S _ { 3 }$ adds classification supervision. We train each stage from scratch rather than warm-starting it from the previous stage, so that stage efects are not confounded with extra optimization. For non-tumor datasets, whose masks are binary rather than multi-region, we use a separate binary lesion segmenter per dataset, which avoids forcing the tumor label scheme onto stroke and white-matter disease (Myronenko 2018).

The serializer ψ extracts disease-relevant fields from $z _ { i } ,$ including laterality, lesion burden, lesion volume, lobe, enhancement pattern, and predicted disease category. These grouped attributes, $f _ { i } \ = \ \bar { \left| f _ { i } ^ { \mathrm { l o c } } , f _ { i } ^ { \mathrm { v o l } } , f _ { i } ^ { \mathrm { a t t r } } , f _ { i } ^ { \mathrm { c a \ " t } } \right| }$ , are concatenated into a compact natural-language sentence $s _ { i } =$ $\psi ( z _ { i } )$ . Coarse fields such as side, volume band, and burden are shared across diseases, while fine-grained fields such as multi-region volumes and enhancement pattern are only produced when the annotation supports them, which keeps the injected prompt human-readable.

## Retrieval and Prompt Construction

The fact sentence is also the retrieval query. Let $e ( \cdot )$ be the frozen text encoder learned by the image-text matching pretask, and let $u _ { i } = e ( s _ { i } )$ be the query embedding for case i. We use two retrieval sources.

For case retrieval, we define the candidate pool as the training split $\tau$ and retrieve the top-k reports within the same disease family. With cosine similarity sim $( i , j ) = \cos ( u _ { i } , u _ { j } )$ the retrieval set is

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { C R } } ( i ) = \mathrm { T o p K } _ { j \in \mathcal { T } , d _ { j } = d _ { i } } \mathrm { s i m } ( i , j ) , } \end{array}\tag{2}
$$

with the query case excluded from its own neighborhood. Retrieved reports are wrapped as reference text from similar prior cases rather than from the patient under study. For knowledge retrieval, we query a curated neuroimaging knowledge base B using the same sentence embedding:

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { K R } } ( i ) = \mathrm { T o p K } _ { b \in \mathcal { B } } \cos ( u _ { i } , e ( b ) ) . } \end{array}\tag{3}
$$

Retrieval is thus aligned with the structured fact representation rather than an image embedding or a sparse lexical match.

The final prompt concatenates the image tokens, a fixed instruction, and an optional grounding block, $P _ { i } =$ $[ \mathrm { I M G } ( X _ { i } )$ ; INST; G<sub>i</sub>]. The grounding block takes one of seven forms:

$$
\begin{array} { r } { \mathcal { G } _ { i } \in \left\{ \begin{array} { l l } { \varnothing , \mathcal { R } _ { \mathrm { K R } } ( i ) , \mathcal { R } _ { \mathrm { C R } } ( i ) , } \\ { s _ { i } , s _ { i } ^ { * } , s _ { i } \oplus \mathcal { R } _ { \mathrm { K R } } ( i ) , s _ { i } \oplus \mathcal { R } _ { \mathrm { C R } } ( i ) } \end{array} \right\} . } \end{array}\tag{4}
$$

where ⊕ denotes concatenation. The PerFact condition is the simplest non-empty case, namely $\mathcal { G } _ { i } = s _ { i }$

## Training Objective

We adapt a general-purpose vision-language backbone with Low-Rank Adaptation (LoRA) (Hu et al. 2021) and keep the base model frozen. Let θ denote the backbone parameters and ∆θ the LoRA updates. Report generation and closed-ended VQA are trained with a mixture objective $\mathscr { L } = \left( { 1 - \rho } \right) \mathscr { L } _ { \mathrm { r e p } } +$ $\rho \mathcal { L } _ { \mathrm { v q a } } ,$ where $\rho = 0 . 2$ in the main setting. The report loss is the standard autoregressive negative log-likelihood,

$$
\mathcal { L } _ { \mathrm { r e p } } = - \sum _ { i = 1 } ^ { N } \log p _ { \theta + \Delta \theta } ( y _ { i } \mid X _ { i } , P _ { i } ) ,\tag{5}
$$

and the VQA loss is the same token-level cross-entropy applied to the answer string

$$
\mathcal { L } _ { \mathrm { v q a } } = - \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { J _ { i } } \log p _ { \theta + \Delta \theta } ( a _ { i , j } \mid X _ { i } , q _ { i , j } , P _ { i } ) .\tag{6}
$$

![](images/0ab179cd962d9c82adc7ac08b7ea061630d7ba0ab410cdc629fb9db42d613f19.jpg)  
Figure 3: Overview of PerFact. Upstream 3D perception (top) turns a multi-contrast brain MRI study into a structured fact sentence (SF); the same sentence is the query for retrieval over training-split reports (CR) or a knowledge base (KR) (bottom). The shared encoder drawn here is the tumor multi-task network; stroke and white-matter hyperintensity each use a standalone binary segmenter instead. The fact sentence, generated report, and VQA pairs are verbatim system output for one glioma case.

Closed-ended VQA is scored without free-form generation: we read out the yes/no token mass at the answer position, normalise it, and optimize the model to place the higher mass on the correct answer.

## Evaluation

For report generation, we report the clinical entity (CE) metrics, precision, recall and F1 score over the entities a report names, alongside the natural language generation (NLG) metrics BLEU (Papineni et al. 2002), METEOR (Banerjee and Lavie 2005), and ROUGE-L (Lin 2004). Let CE(·) denote the clinical entity extractor applied to generated and reference reports; then

$$
\begin{array} { r } { \mathrm { P r e c } = \frac { \vert \mathrm { C E } ( \hat { y } _ { i } ) \cap \mathrm { C E } ( y _ { i } ) \vert } { \vert \mathrm { C E } ( \hat { y } _ { i } ) \vert } , \quad \mathrm { R e c } = \frac { \vert \mathrm { C E } ( \hat { y } _ { i } ) \cap \mathrm { C E } ( y _ { i } ) \vert } { \vert \mathrm { C E } ( y _ { i } ) \vert } , } \end{array}\tag{7}
$$

with F1 computed in the usual way from precision and recall. For VQA, we threshold the normalized yes-probability at 0.5 to compute accuracy and F1 score, and report the area under the receiver operating characteristic curve (AUROC) over the same score. Upstream perception is evaluated with Dice for segmentation and per-attribute AUC for classification.

## Experimental Setup

## Data

We build a unified multi-disease brain MRI corpus by reusing the radiology reports released with RadGenome-

Brain MRI (Lei et al. 2024) across four public imaging sources: glioma (230 reported cases, T1-weighted/T1- weighted with contrast/T2-weighted/fluid attenuated inversion recovery [FLAIR]) (Baid et al. 2021), meningioma (230, same sequences) (LaBella et al. 2023), acute stroke (250, difusion weighted imaging/apparent difusion coeficient/FLAIR) (Hernandez Petzsche et al. 2022), and white-matter hyperintensity (60 annotated training cases, FLAIR/T1-weighted) (Kuijf et al. 2019). The corpus spans four diseases and three image contrasts; the report test set contains 130 cases and the closed-ended VQA bank 1,549 questions. Because the sequence set difers by disease, the prompt carries a variable number of images and names the sequences present rather than padding to a fixed set.

We follow the oficial split where one exists and otherwise generate a 70/15/15 per-dataset split with seed 42, recorded in the released manifest. Cases are filtered for image-text quality per dataset rather than globally, so that a domain gap between datasets cannot silently remove one disease.

An external cohort without reference reports: We intended report-level external validation on a post-operative glioma cohort (Fields et al. 2024), but its 298 studies provide segmentation labels and a clinical table with no free-text reports. We report this rather than substituting templated text, and confine external evidence to the upstream perception task.

Table 1: Grounding source is the lever (report $n { = } 1 3 0 , \mathrm { V Q A } n { = } 1 5 4 9 )$ . Expands the Qwen2.5-VL block of Table 3 across eight grounding conditions: +KR retrieved knowledge, +CR retrieved prior case reports, +SF predicted facts, ${ + } \mathrm { S F ^ { * } }$ ground-truth facts; only the injected information difers. Every cell is a single run, read against the seed band of the Experimental Setup. Best and second best among deployable methods; $+ \bar { \mathrm { S F ^ { * } } }$ is an upper bound, not a competitor.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Injected grounding</td><td colspan="3">CE metrics</td><td colspan="3">NLG metrics</td><td colspan="3">Closed-ended VQA</td></tr><tr><td>PREC</td><td>REC</td><td>F-1</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>Acc</td><td>F-1</td><td>AUROC</td></tr><tr><td>Zero-shot</td><td>none, no adaptation</td><td>0.541</td><td>0.480</td><td>0.509</td><td>2.68</td><td>20.32</td><td>15.52</td><td>0.567 0.352</td><td></td><td>0.530</td></tr><tr><td>SFT</td><td>none</td><td>0.671</td><td>0.619</td><td>0.644</td><td>25.83</td><td>43.71</td><td>49.85</td><td>0.650</td><td>0.567</td><td>0.697</td></tr><tr><td>+KR</td><td>knowledge</td><td>0.682</td><td>0.601</td><td>0.639</td><td>25.98</td><td>43.96</td><td>51.31</td><td>0.644</td><td>0.583</td><td>0.701</td></tr><tr><td>+CR</td><td>prior case reports</td><td>0.701</td><td>0.736</td><td>0.718</td><td>28.15</td><td>46.70</td><td>51.68</td><td>0.683</td><td>0.488</td><td>0.701</td></tr><tr><td>+SF (PerFact)</td><td>predicted facts</td><td>0.771</td><td>0.779</td><td>0.775</td><td>29.52</td><td>47.30</td><td>53.55</td><td>0.693</td><td>0.535</td><td>0.720</td></tr><tr><td>+SF +KR</td><td>facts + knowledge</td><td>0.763</td><td>0.791</td><td>0.777</td><td>29.45</td><td>47.31</td><td>53.55</td><td>0.676</td><td>0.465</td><td>0.711</td></tr><tr><td>+SF+CR</td><td>facts + case reports</td><td>0.766</td><td>0.754</td><td>0.760</td><td>29.20</td><td>46.85</td><td>52.63</td><td>0.674</td><td>0.409</td><td>0.698</td></tr><tr><td> $+ S F ^ { * }$ </td><td>oracle facts</td><td>0.912</td><td>0.849</td><td>0.879</td><td>34.28</td><td>51.90</td><td>56.81</td><td>0.655</td><td>0.546</td><td>0.666</td></tr></table>

## Implementation

We use Qwen2.5-VL-7B-Instruct as the main backbone, and compare Qwen2.5-VL 3B/32B, LLaVA-1.5-7B, and InternVL3-8B under the same recipe. Fine-tuning uses LoRA (Hu et al. 2021) with r=16, α=32, and all linear targets, trained with lr 1e−4, cosine decay, 0.05 warmup, and 3 epochs. Runs use bf16 precision, gradient checkpointing, batch size 1 with accumulation 8, a 4096–6144 token context, and image\_max\_pixels 262,144. We decode greedily throughout, and training runs on a single NVIDIA GH200 96GB GPU under aarch64 Linux, with LLaMA-Factory (Zheng et al. 2024), CUDA 12.6, Python 3.11.7 and Transformers 4.49–4.53; upstream perception is implemented in PyTorch with MONAI (Cardoso et al. 2022). Early runs used sampling, which altered F1 score by up to 0.03 in either direction; all reported results come from the greedy rerun.

Retrieval queries a frozen biomedical image-text encoder (Zhang et al. 2023) with the fact sentence. Closed-ended VQA is scored by the next-token yes/no probability mass rather than by generation, following the protocol of recent medical RAG work (Xia et al. 2025), so that accuracy, F1 score, and AUROC remain comparable to published radiology-VQA numbers.

## Measuring the Noise Band

Retraining fixed configurations under training seeds 42, 1234 and 2024, report clinical entity F1 score has a standard deviation near 0.003 (0.001–0.005 across backbones) and VQA AUROC near 0.010, giving conservative bands of ±0.01 and ±0.02–0.03. Every comparison below is read against these bands, and diferences inside them are reported as ties.

## Comparisons

We vary three axes, each changing only one factor. (i) Grounding source: supervised fine-tuning on images alone (SFT) / +KR / +CR / +SF / +SF +CR / +SF<sup>∗</sup>; (ii) backbone: Qwen2.5-VL at 3B, 7B, and 32B (Qwen Team 2025), LLaVA-1.5-7B (Liu et al. 2024), and InternVL3-8B (Zhu et al. 2025), all under fixed SF grounding; (iii) domain pretraining: six medical and radiology vision-language models evaluated zero-shot on the same test set and the same metric suite, namely MAIRA-2 (Bannur et al. 2024), LLaVA-Rad (Zambrano Chaves et al. 2025), CheXagent (Chen et al. 2024b), LLaVA-Med (Li et al. 2023), HuatuoGPT-Vision (Chen et al. 2024a), and RadFM (Wu et al. 2025). Singleimage models receive a 2 × 2 montage of the sequences; RadFM, the only natively volumetric model, receives the volumes.

## Results

## Efect of the Grounding Source

Holding everything else fixed and varying only what is injected, clinical entity F1 score varied over a range of 0.235 (Table 1). Three observations follow. First, facts outperform retrieval: SF reaches 0.775 against 0.718 for case retrieval, while knowledge retrieval (0.639) is no better than plain finetuning (0.644); retrieval helps only when it supplies theform of a report, not as a source of domain knowledge. Second, retrieval is redundant once facts are present: adding knowledge to SF is flat (+0.002) and adding case reports slightly hurts (−0.015), because the two sources overlap in content rather than accumulate, which is why PerFact is defined as SF alone. Third, imperfect upstream perception remains worthwhile: SF recovers 0.131 of the 0.235 available above SFT, more than half the oracle gain, without ground-truth annotation at inference; the residual oracle gap is 0.104.

Retrieval’s contribution is stylistic, the 0.074 that CR alone adds over SFT. Sweeping its design choices (query, retriever, k, reranking, filtering to content absent from the facts) changed nothing: one variant appeared to gain +0.021 but did not survive reseeding.

The gain is not only aggregate. The injected fact corrects all three errors in Figure 2, and the direction of midline shift is recovered even though the fact records only that a shift is present.

Table 2: Fine-tuned backbones under identical PerFact grounding, difering only in the backbone; every column is a mean over the same three seeds. Best and second best in each column.
<table><tr><td rowspan="2">Method</td><td rowspan="2">LLM</td><td rowspan="2">Size</td><td colspan="3">CE metrics</td><td colspan="3">NLG metrics</td><td colspan="3">Closed-ended VQA</td></tr><tr><td>PREC</td><td>REC</td><td>F-1</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>Acc</td><td>F-1</td><td>AUROC</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5</td><td>3B</td><td>0.762</td><td>0.782</td><td>0.772</td><td>29.67</td><td>47.46</td><td>53.86</td><td>0.618</td><td>0.570</td><td>0.683</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5</td><td>7B</td><td>0.759</td><td>0.793</td><td>0.775</td><td>30.42</td><td>48.63</td><td>54.14</td><td>0.653</td><td>0.564</td><td>0.702</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5</td><td>32B</td><td>0.766</td><td>0.782</td><td>0.774</td><td>29.90</td><td>47.75</td><td>53.60</td><td>0.640</td><td>0.582</td><td>0.715</td></tr><tr><td>LLaVA-1.5</td><td>Vicuna</td><td>7B</td><td>0.750</td><td>0.786</td><td>0.767</td><td>29.99</td><td>47.56</td><td>53.64</td><td>0.662</td><td>0.494</td><td>0.685</td></tr><tr><td>InternVL3</td><td>Qwen2.5</td><td>8B</td><td>0.754</td><td>0.811</td><td>0.782</td><td>29.80</td><td>47.90</td><td>53.44</td><td>0.675</td><td>0.603</td><td>0.714</td></tr></table>

## Efect of Model Choice

Backbone choice does not matter. Fixing grounding to SF and swapping only the backbone, clinical entity F1 score spans 0.767–0.782 across five models from three families and an order of magnitude in scale, over three seeds each (Table 2; the best are the two PerFact rows of Table 3). That 0.015 spread is wider than reseeding alone predicts (one-way ANOVA over the fifteen runs, F(4, 10)=3.60, p=0.046), so the backbones are not interchangeable: InternVL3 separates, and the remaining four span 0.008. Read on a single scale, the three axes we can vary are 0.006 wide for reseeding, 0.015 for the backbone and 0.235 for what is injected. Scale alone (3B, 7B, 32B) is flat at 0.772, 0.775, 0.774.

One asymmetry: report generation is flat across backbones, but VQA shows a mild, monotone scale efect within the Qwen family . Rendering structured facts as prose is within reach of every model tried; answering a closed question about the image is not.

Medical pretraining does not transfer. Table 3 evaluates six medical and radiology vision-language models zero-shot. None approaches a fine-tuned system on any of the nine metrics (Figure 4), and performance orders cleanly by how far the pretraining domain sits from brain MRI: general medical above biomedical and volumetric radiology, with the chest radiology report generators at the bottom. The failure is qualitative too, the chest-specialized models producing findings such as “The heart size is normal” or references to PA/lateral views for a brain MRI study, and it takes more than one form (Figure 5).

The same ordering holds for closed-ended VQA (Table 3): the best zero-shot model reaches AUROC 0.631, below Per-Fact’s 0.720, and two fall below chance. MAIRA-2 is not applicable rather than a number: it exposes only report generation and phrase grounding, so a forced forward pass scores the first token of a report, not an answer.

## Upstream Perception

The upstream models the facts depend on are competent in their own right. On the tumor subset, adding image-text alignment to the multi-task network helps most: mean attribute AUC over 17 attributes rises from 0.758 to 0.829 and tumor Dice from 0.755/0.746/0.731 to 0.764/0.767/0.767 (enhancing/core/whole), after which classification saturates and matching recall at 10 rises from 0.07 to 0.48. Per-dataset segmenters reach Dice 0.697 on stroke and 0.756 on the official 110-case white-matter split (Kuijf et al. 2019), whose two unseen scanners cost only 0.015 Dice. The white-matter segmenter was the one that most repaid attention: retraining it at 500 epochs with best-validation checkpointing lifted its Dice from 0.592 to 0.750, a 27% relative gain. See the Supplementary Material for details.

![](images/846c8bdbf8651b2cbf327762e8d27d6bd043d9dec8d9ec23037014852c071d08.jpg)  
Figure 4: PerFact against released models on all nine metrics (zero-shot, brain MRI). The general medical model and the two chest radiology report generators all collapse, most sharply on the NLG axes, while PerFact traces a full profile.

## Specificity to Report Generation

Closed-ended VQA behaves diferently. Across the six nonoracle conditions ofTable 1 the VQA AUROC range is 0.023, against 0.138 for clinical entity F1 score over those same conditions, so the grounding lever is specific to the text the model writes. Answering ability instead requires having seen the answer format: without it every condition sits near chance (AUROC 0.53–0.58), and the 20% mixture we adopt lifts AUROC to 0.696 at a cost to report F1 of −0.002, inside the noise band.

Table 3: Report generation and closed-ended VQA on brain MRI (n=130 reports, n=1549 questions). Blocks: medical pretrained VLMs, chest radiology report generators (both zero-shot), then ours with one un-adapted anchor per backbone; Table 1 walks the intermediate conditions. LLaVA-Rad’s report F-1 is inflated by laterality words coinciding with brain entities; its recall of 0.108 is the honest figure. Answering no throughout scores 0.655 on the VQA bank, above every zero-shot model here, so AUROC is the column to read. Dashes: MAIRA-2 exposes no question-answering interface. Best and second best among deployable rows; zero-shot and Qwen2.5-VL cells are single runs, the InternVL3 PerFact row a three-seed mean.
<table><tr><td rowspan="2">Method</td><td rowspan="2">LLM</td><td rowspan="2">Size</td><td colspan="3">CE metrics</td><td colspan="3">NLG metrics</td><td colspan="3">Closed-ended VQA</td></tr><tr><td>PREC</td><td>REC</td><td>F-1</td><td></td><td>BLEU-4 METEOR ROUGE-L</td><td></td><td>Acc</td><td>F-1</td><td>AUROC</td></tr><tr><td>HuatuoGPT-Vision</td><td>Qwen2</td><td>7B</td><td>0.544</td><td>0.346</td><td>0.423</td><td>2.87</td><td>17.46</td><td>14.43</td><td>0.550 0.532</td><td></td><td>0.631</td></tr><tr><td>RadFM</td><td>MedLLaMA</td><td>14B</td><td>0.486</td><td>0.122</td><td>0.195</td><td>2.66</td><td>11.05</td><td>15.88</td><td></td><td>0.566 0.106</td><td>0.438</td></tr><tr><td>LLaVA-Med-1.5</td><td>Mistral</td><td>7B</td><td>0.503</td><td>0.090</td><td>0.153</td><td>0.70</td><td>8.13</td><td>11.84</td><td></td><td>0.394 0.525</td><td>0.547</td></tr><tr><td>LLaVA-Rad</td><td>Vicuna</td><td>7B</td><td>0.500</td><td>0.108</td><td>0.178</td><td>1.15</td><td>8.33</td><td>11.69</td><td>0.467 0.482</td><td></td><td>0.564</td></tr><tr><td>MAIRA-2</td><td>Vicuna</td><td>6.9B</td><td>0.455</td><td>0.041</td><td>0.075</td><td>1.12</td><td>7.79</td><td>12.39</td><td></td><td></td><td></td></tr><tr><td>CheXagent</td><td>Phi-2</td><td>8B</td><td>0.172</td><td>0.013</td><td>0.024</td><td>0.52</td><td>3.31</td><td>6.04</td><td></td><td>0.612 0.294</td><td>0.478</td></tr><tr><td rowspan="2">Qwen2.5-VL (zero-shot) PerFact (ours)</td><td>Qwen2.5</td><td>7B</td><td>0.541</td><td></td><td>0.480 0.509</td><td>2.68</td><td>20.32</td><td>15.52</td><td>0.567 0.352</td><td></td><td>0.530</td></tr><tr><td></td><td></td><td>0.771</td><td>0.779</td><td>0.775</td><td>29.52</td><td>47.30</td><td>53.55</td><td>0.693 0.535</td><td></td><td>0.720</td></tr><tr><td rowspan="2">InternVL3 (zero-shot) PerFact (ours)</td><td rowspan="2">Qwen2.5</td><td>8B</td><td>0.499</td><td>0.455</td><td>0.476</td><td>2.05</td><td>20.57</td><td>17.06</td><td>0.522 0.467</td><td></td><td>0.576</td></tr><tr><td></td><td>0.754</td><td>0.811 0.782</td><td></td><td>29.80</td><td>47.90</td><td>53.44</td><td>0.675 0.603</td><td></td><td>0.714</td></tr></table>

![](images/9c0010607803fc0a98577d0e0196a5cb106a7c531d025056665ecbde1e2a28be.jpg)  
Figure 5: t-SNE of clinical entity vectors against the reference reports (brain MRI, n=130 each). All 1,170 vectors share one embedding, so the reference distribution is identical in every panel. Shading outlines each cloud and titles give the median Jaccard distance $d _ { J }$ . All panels but the last are zero-shot, and only PerFact’s outline coincides with the reference’s. The cluster left of center is the all-empty vector: 86 of LLaVA-Med-1.5’s reports name no recognized brain finding, against 3 of PerFact’s.

## Sources of the Residual Oracle Gap

The residual −0.104 between SF and $\mathrm { S F ^ { * } }$ in Table 1 is not uniform; it tracks how fine-grained the facts are (per-disease numbers in the supplement). Tumor cases carry the richest facts (three region volumes, enhancement pattern, disease type) and lose the most when predicted, F1 falling from 0.919 to 0.782. Stroke facts are coarser and lose almost nothing (0.782 against 0.762); white-matter facts are a single burden band and lose nothing at all. The bottleneck is therefore the precision of what upstream can assert, not the generator: the gap is widest exactly where the schema asks upstream for the most detail: three nested tumor regions and an enhancement pattern against a single white-matter burden band. The whitematter row rests on nine reported cases and is indicative rather than conclusive. Whether a better segmenter closes the gap is not something we measure here, since every end-toend condition reported above uses one fixed set of upstream checkpoints.

## Conclusion

We asked what determines report quality on 3D multicontrast brain MRI data, and answered it by holding everything fixed except one variable at a time. Model choice is not the answer. The injected information is: it varies clinical entity F1 score within a range of 0.235 against the 0.015 spanned by the backbone, with structured facts from upstream 3D perception outperforming retrieved prior reports, and predicted facts recovering approximately half of the oracle headroom without inference-time annotation. We instantiate this as PerFact, the simplest configuration that carries the gain.

The remaining gap is upstream and structured: it scales with how much detail the fact schema asks upstream for, widest for tumor cases and absent for white-matter cases. This directs further work to the perception-language interface rather than to the generator. Two constraints bound our claims: all reports come from a single annotation source, so the learned conventions are its own; and the generator consumes 2D slices, so a natively volumetric model might close part of the oracle gap. Both are left to future work.

## References

Baid, U.; Ghodasara, S.; Mohan, S.; Bilello, M.; Calabrese, E.; Colak, E.; et al. 2021. The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. arXiv:2107.02314.

Banerjee, S.; and Lavie, A. 2005. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 65–72.

Bannur, S.; Bouzid, K.; Castro, D. C.; Schwaighofer, A.; Thieme, A.; Bond-Taylor, S.; Ilse, M.; Pérez-García, F.; Salvatelli, V.; Sharma, H.; Meissen, F.; Ranjit, M.; Srivastav, S.; Gong, J.; Codella, N. C. F.; Falck, F.; Oktay, O.; Lungren, M. P.; Wetscherek, M. T.; Alvarez-Valle, J.; and Hyland, S. L. 2024. MAIRA-2: Grounded Radiology Report Generation. arXiv:2406.04449.

Cardoso, M. J.; Li, W.; Brown, R.; Ma, N.; Kerfoot, E.; et al. 2022. MONAI: An open-source framework for deep learning in healthcare. arXiv:2211.02701.

Chen, J.; Gui, C.; Ouyang, R.; Gao, A.; Chen, S.; Chen, G. H.; Wang, X.; Zhang, R.; Cai, Z.; Ji, K.; Yu, G.; Wan, X.; and Wang, B. 2024a. HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale. arXiv:2406.19280.

Chen, Z.; Shen, Y.; Song, Y.; and Wan, X. 2021. Crossmodal Memory Networks for Radiology Report Generation. In Proceedings ofthe 59thAnnual Meeting oftheAssociation for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP), 5904–5914.

Chen, Z.; Song, Y.; Chang, T.-H.; and Wan, X. 2020. Generating Radiology Reports via Memory-driven Transformer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1439–1449.

Chen, Z.; Varma, M.; Delbrouck, J.-B.; Paschali, M.; Blankemeier, L.; Van Veen, D.; Valanarasu, J. M. J.; Youssef, A.; Cohen, J. P.; Reis, E. P.; Tsai, E. B.; Johnston, A.; Olsen, C.; Abraham, T. M.; Gatidis, S.; Chaudhari, A. S.; and Langlotz, C. 2024b. CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation. arXiv:2401.12208.

Demner-Fushman, D.; Kohli, M. D.; Rosenman, M. B.; Shooshan, S. E.; Rodriguez, L.; Antani, S.; Thoma, G. R.; and McDonald, C. J. 2016. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23(2): 304–310.

Fields, B. K. K.; Calabrese, E.; Mongan, J.; Cha, S.; Hess, C. P.; Sugrue, L. P.; Chang, S. M.; Luks, T. L.; Villanueva-Meyer, J. E.; Rauschecker, A. M.; and Rudie, J. D. 2024. The University of California San Francisco Adult Longitudinal Post-Treatment Difuse Glioma MRI Dataset. Radiology: Artificial Intelligence, 6(4): e230182.

Hernandez Petzsche, M. R.; de la Rosa, E.; Hanning, U.; Wiest, R.; et al. 2022. ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. Scientific Data, 9. ArXiv:2206.06694; article s41597-022- 01875-5.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Huang, Z.; Zhang, X.; and Zhang, S. 2023. KiUT: Knowledge-injected U-Transformer for Radiology Report Generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19809–19818. ArXiv:2306.11345.

Jain, S.; Agrawal, A.; Saporta, A.; Truong, S. Q. H.; Duong, D. N.; Bui, T.; Chambon, P.; Zhang, Y.; Lungren, M. P.; Ng, A. Y.; Langlotz, C. P.; and Rajpurkar, P. 2021. RadGraph: Extracting Clinical Entities and Relations from Radiology Reports. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks).

Jin, H.; Che, H.; Lin, Y.; and Chen, H. 2024. PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2607–2615. ArXiv:2308.12604.

Jing, B.; Xie, P.; and Xing, E. 2018. On the Automatic Generation of Medical Imaging Reports. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2577–2586.

Johnson, A. E. W.; Pollard, T. J.; Berkowitz, S. J.; Greenbaum, N. R.; Lungren, M. P.; Deng, C.-y.; Mark, R. G.; and Horng, S. 2019. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data, 6(1): 317.

Kondepudi, A.; Rao, A.; Zhao, C.; Lyu, Y.; Harake, S.; Banerjee, S.; Joshi, R.; Meissner, A.-K.; Hou, R.; Jiang, C.; Chowdury, A.; Srinivasan, A.; Athey, B.; Gulani, V.; Pandey, A.; Lee, H.; and Hollon, T. 2026. Health system learning enables generalist neuroimaging models. Nature Medicine. Published online 10 July 2026; doi:10.1038/s41591-026-04497- 1.

Kuijf, H. J.; Biesbroek, J. M.; De Bresser, J.; Heinen, R.; Andermatt, S.; Bento, M.; Berseth, M.; Belyaev, M.; Cardoso, M. J.; Casamitjana, A.; et al. 2019. Standardized assessment of automatic segmentation of white matter hyperintensities and results of the WMH segmentation challenge. IEEE transactions on medical imaging, 38(11): 2556–2568.

LaBella, D.; Adewole, M.; Alonso-Basanta, M.; et al. 2023. The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial Meningioma. arXiv:2305.07642.

Lei, J.; Zhang, X.; Wu, C.; Dai, L.; Zhang, Y.; Zhang, Y.; Wang, Y.; Xie, W.; and Li, Y. 2024. AutoRG-Brain: Grounded Report Generation for Brain MRI.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 9459–9474. ArXiv:2005.11401.

Li, C.; Wong, C.; Zhang, S.; Usuyama, N.; Liu, H.; Yang, J.; Naumann, T.; Poon, H.; and Gao, J. 2023. LLaVA-

Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. arXiv:2306.00890.

Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, 74–81.

Liu, F.; Wu, X.; Ge, S.; Fan, W.; and Zou, Y. 2021. Exploring and Distilling Posterior and Prior Knowledge for Radiology Report Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). ArXiv:2106.06963.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024. Improved Baselines with Visual Instruction Tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). ArXiv:2310.03744.

Myronenko, A. 2018. 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization. In International MIC-CAI Brainlesion Workshop (BrainLes). ArXiv:1810.11654.

Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. In Proceedings ofthe 40thAnnual Meeting ofthe Associationfor Computational Linguistics (ACL), 311–318.

Qwen Team. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Smit, A.; Jain, S.; Rajpurkar, P.; Pareek, A.; Ng, A. Y.; and Lungren, M. P. 2020. CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). ArXiv:2004.09167.

Vinyals, O.; Toshev, A.; Bengio, S.; and Erhan, D. 2015. Show and Tell: A Neural Image Caption Generator. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3156–3164.

Wu, C.; Zhang, X.; Zhang, Y.; Wang, Y.; and Xie, W. 2025. Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data. Nature Communications. ArXiv:2308.02463; article s41467-025-62385-7.

Xia, P.; Zhu, K.; Li, H.; Wang, T.; Shi, W.; Wang, S.; Zhang, L.; Zou, J.; and Yao, H. 2025. MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models. In International Conference on Learning Representations (ICLR). ArXiv:2410.13085.

Xia, P.; Zhu, K.; Li, H.; Zhu, H.; Li, Y.; Li, G.; Zhang, L.; and Yao, H. 2024. RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1081–1093. ArXiv:2407.05131.

Zambrano Chaves, J. M.; et al. 2025. A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings. Nature Communications. Article s41467-025-58344-x.

Zhang, S.; Xu, Y.; Usuyama, N.; et al. 2023. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv:2303.00915.

Zhang, Y.; Wang, X.; Xu, Z.; Yu, Q.; Yuille, A.; and Xu, D. 2020. When Radiology Report Generation Meets Knowledge Graph. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 12910–12917. ArXiv:2002.08277.

Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; Luo, Z.; and Ma, Y. 2024. LlamaFactory: Unified Eficient Fine-Tuning of 100+ Language Models. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (System Demonstrations). ArXiv:2403.13372.

Zhu, J.; Wang, W.; Chen, Z.; et al. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479.