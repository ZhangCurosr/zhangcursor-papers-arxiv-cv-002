# Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters

Taha Sabir<sup>1[0009</sup>−<sup>0002</sup>−<sup>4870</sup>−<sup>0017]</sup>, Alberto Montero Molina<sup>2,</sup>†<sup>[0009</sup>−<sup>0008</sup>−<sup>6798</sup>−<sup>0943]</sup>, Thinh Quang Ly<sup>3,</sup>†<sup>[0009</sup>−<sup>0006</sup>−<sup>9039</sup>−<sup>5887]</sup>, Veronika Dmitrenko<sup>4,</sup>†<sup>[0009</sup>−<sup>0006</sup>−<sup>4636</sup>−<sup>7440]</sup>, Mohammad Yaqub<sup>5[0000</sup>−<sup>0001</sup>−<sup>6896</sup>−<sup>1105]</sup>, and Tausifa Jan Saleem<sup>5(B)[0000</sup>−<sup>0002</sup>−<sup>0827</sup>−<sup>0043]</sup>

<sup>1</sup> Department of Computer Science, Gettysburg College, Gettysburg, PA 17325, USA <sup>2</sup> Higher School of Computing (ESCOM), Instituto Politécnico Nacional (IPN), Mexico City 07738, Mexico

<sup>3</sup> Department of Computer Science and Engineering, Michigan State University, East Lansing, MI 48824, USA

<sup>4</sup> Faculty of Computer Science and Faculty of Economic Sciences, HSE University, Moscow, Russia

<sup>5</sup> Computing and Mathematical Sciences Division, Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, United Arab Emirates tausifa.saleem@mbzuai.ac.ae †These authors contributed equally to this work.

Abstract. Adapting a single vision-language model to medical image parsing requires training choices that reflect the demands of each task. We study task-specific adaptation with Qwen3.5-9B for FLARE 2026 Task 3, covering six task families and seven imaging modalities. Our system couples a shared, frozen 4-bit backbone with LoRA adapters selected from the input question. We customize adapter ranks and prompts, rebalance rare classification targets, and supplement regression training with dermatology classification examples. On the fixed internal holdout, the task-specific adapters achieve a dataset-macro balanced accuracy of 0.7791 for classification, a dataset-macro micro-F1 of 0.6286 for multilabel classification, a counting mean absolute error (MAE) of 11.73, a dataset-macro detection F1 of 0.8914, a dataset-macro regression MAE of 9.78, and a report-generation CRIMSON score of 0.8204. On the organizer’s 1,703-example hidden-validation set, the submitted system achieves 0.8496 classification balanced accuracy, 0.5338 sample-averaged multi-label F1, and 14.5950 regression MAE. The final 30.1 GB container averages 3.1 seconds per case, including model loading, and produces identical outputs across repeated verification runs. The benefits vary across tasks, supporting the use of task-specific adapter configurations and image budgets within a shared medical vision-language model. Code is available at https://github.com/TahaSabir0/FLARE26-Hexapus.

Keywords: Medical image parsing · Vision-language models · Parametereficient fine-tuning · LoRA · Task routing · FLARE 2026.

## 1 Introduction

Unlike conventional visual question answering (VQA), where the expected response is primarily textual, medical image parsing requires a general medical vision-language model (VLM) to interpret heterogeneous visual inputs and produce task-appropriate outputs ranging from categorical and multi-label predictions to numerical estimates, bounding boxes, and free-form reports, across modalities with diferent visual characteristics and anatomical morphologies. Medical VLMs range from biomedical dialogue systems such as LLaVA-Med [10] to models pretrained across clinical imaging domains such as MedGemma [16], and benchmarks compare generalist and specialist performance across clinical domains [24]. Med-MoE [9] illustrates the need to combine discriminative and generative medical tasks, while a systematic task-transfer study [15] shows that fine-tuning one visual capability can unpredictably improve or degrade others.

FLARE 2026 Task 3 [6] provides a practical setting for studying this problem. After data cleaning, its 51,154 question-answer pairs and 50,996 images span 19 datasets, eight imaging modalities, and seven task types; excluding unscored instance detection and endoscopy, our evaluation covers 17 datasets, seven modalities, and six scored task families: classification, multi-label classification, counting, detection, regression, and report generation. A single shared adaptation may sufer from interference between these tasks, whereas a separate fine-tuned VLM per task would substantially increase training, storage, and deployment costs.

Related Work. General-domain VLMs such as BLIP-2 [11] and Qwen2.5-VL [2] align visual and language representations through lightweight bridging modules or dynamic-resolution processing, while medical systems including LLaVA-Med [10], HuatuoGPT-Vision [4], and MedGemma [16] add medical image–text training or domain-specific visual representations. Yet parsing outputs difer in visual evidence, supervision, and decoding constraints. Previous challenge solutions to FLARE 2025 Task 5 [12] use diferent forms of specialization: Qwen-VLConnector [13] fuses multi-level visual features, ME-VLIP [17] routes questions among task-specific QLoRA adapters and a multi-task fallback, and Sun et al. [19] use a medically pretrained vision encoder with a shared LoRA-adapted language model. These approaches show that broad task coverage is possible but leave open how task-specific adaptation, prompting, and visual resolution should be allocated within one shared, deployable backbone. LoRA [8], QLoRA [5], and LLaMA-Factory [23] make such modular specialization practical.

Contributions. Our system uses Qwen3.5-9B [14] with a question-only GLi-Class router [18] (DeBERTa-V3 [7]) selecting one of six specialist adapters or a general fallback on one shared 4-bit backbone. Our contributions are:

– We develop and validate a parameter-eficient system for six medical imageparsing tasks using one quantized Qwen3.5-9B backbone and lightweight task-specific LoRA adapters.

– We compare the adapted system with its zero-shot Qwen3.5-9B backbone on an internal holdout and public validation data.

– We compare specialist and general adapters, measure task-dependent resolution efects, and analyze errors hidden by aggregate metrics.

## 2 Method

We use one quantized Qwen3.5-9B backbone with task-specific LoRA adapters. A question-only router selects the adapter, image budget, and prompt for each case (Fig. 1).

## 2.1 Data preparation and image handling

We audited the oficial FLARE 2026 Task 3 2D data [6] and excluded 73 records that failed task-specific validity checks: 47 endoscopy records with invalid labels, 17 BUSI detection records with inverted or zero-area boxes, seven periapical records with contradictory labels, one duplicated periapical record, and one Boneresorption record with an extreme target value of 2,085 (dataset median: 28). We also excluded all 1,248 instance-detection samples because this task was not scored, and omitted endoscopy from adapter training because of its few usable examples, long multi-frame inputs, and noisy labels.

Following ME-VLIP [17], we expanded each multi-image training example into separate single-image examples, repeating the question and answer for each image. Questions sharing an image remained in the same partition. In earlier Qwen3-VL-8B experiments, grouping three neojaundice images reduced balanced accuracy in two complementary train–validation folds; this motivated, but does not validate, the policy for Qwen3.5-9B. At inference, the system accepts up to four images and selects four evenly spaced images from longer sequences.

Image preprocessing consists of RGB conversion and aspect-ratio-preserving resizing when an image exceeds its route’s pixel-area cap. The cap is a maximum area, not a forced square shape. We used no additional image transformations, unlabeled examples, or partial labels.

## 2.2 Shared backbone and task routing

All routes use one Qwen3.5-9B backbone loaded with 4-bit NF4 quantization and bfloat16 computation. The pretrained backbone remains frozen. Six taskspecific LoRA adapters and one general adapter, trained across all scored tasks, are loaded into the same model; switching tasks changes the active adapter without reloading the backbone.

We reuse the GLiClass classifier released with ME-VLIP [17], without further training. It uses a DeBERTa-v3-small encoder to predict a task label and confidence from the question text alone. The corresponding specialist is activated when the top confidence is at least 0.3 and the label has a configured route; otherwise, the general adapter is used. Supplied TaskType and Modality metadata do not influence routing.

## 2.3 Task-specific adaptation

We train each adapter by supervised fine-tuning, optimizing LoRA parameters with loss applied only to the answer tokens. Table 1 lists the deployed configurations. In the table, the full labeled pool comprises A ∪ B, the internal holdout, and validation-public; prompt variants are defined in Section 2.4. Shared optimization settings are given in Section 3.2.

![](images/df797c0782cae405fd64942a115d9879b854726ff23aea9d59453128946a38b8.jpg)  
Fig. 1. Routed inference with a shared quantized backbone. The question selects a specialist or the general fallback, together with the image budget and prompt used for answer generation.

Table 1. Per-route configuration of the proposed system. Pixel values denote maximum image-area budgets, not forced square resizing.
<table><tr><td>Route</td><td>Data scope</td><td>Pixels</td><td>LoRA r/α</td><td>Prompt</td><td>Output</td></tr><tr><td>Classification</td><td>Full labeled pool, oversam- pled</td><td>20482</td><td>32/64</td><td>Bare</td><td>Label string</td></tr><tr><td>Multi-label classifi- cation</td><td>Full labeled pool</td><td>20482</td><td>64/128</td><td>Medical</td><td>Label set</td></tr><tr><td>Counting</td><td>Full labeled pool</td><td>20482</td><td>8/16</td><td>Counting</td><td>Numeric string</td></tr><tr><td>Detection</td><td>Full labeled pool</td><td>20482</td><td>8/16</td><td>Bare</td><td>Nested boxes</td></tr><tr><td>Regression</td><td>A∪B + bcn20000 donor mix- 20482 ture</td><td></td><td>8/16</td><td>Bare</td><td>Numeric string</td></tr><tr><td>Report generation</td><td>A∪B</td><td>7682</td><td>8/16</td><td>Bare</td><td>Free text</td></tr><tr><td>General adapter</td><td>A∪B, all scored tasks</td><td>7682</td><td>8/16</td><td>Bare</td><td>Task- dependent</td></tr></table>

Preliminary Qwen3-VL-8B experiments informed the LoRA [8] ranks retained for Qwen3.5-9B: 32 for classification, 64 for multi-label classification, and 8 for the remaining adapters, with $\alpha = 2 r$ throughout.

For classification, we use capped oversampling of rare dataset–answer groups. A group containing $n < 2 5 0$ training examples is expanded by duplication to min(6n, 250) examples; larger groups are unchanged. For regression, we combine 5,265 examples from A ∪ B with 5,000 bcn20000 dermatology-classification examples. The A/B partitions and internal holdout are defined in Section 3.1.

Adapters trained on A ∪ B provide the internal comparisons. For deployment, classification, multi-label classification, counting, and detection were subsequently refitted on the full labeled development pool. Regression, report generation, and the general fallback retain their A ∪ B adapters.

## 2.4 Prompting and inference

Each user message contains indexed image markers followed by the original question. Counting and multi-label classification also use the instructions below, placed before the question. They are inspired by chain-of-thought prompting [21], but training supervises only final answers. The remaining routes use the bare image–question prompt.

<table><tr><td>Task-specific prompts</td></tr><tr><td>Counting System You are a medical image analysis expert. Think carefully before answering. Task instruction Methodically count each object in the image (scan left-to-right, row by row)</td></tr><tr><td>Multi-label classification System You are an expert radiologist and medical image analyst specialising in</td></tr><tr><td>X-ray, CT, MRI, ultrasound, pathology, endoscopy, and microscopy. Think carefully and systematically before answering.</td></tr><tr><td>Task instruction For each listed finding, assess whether it is clearly visible in the image. List only confirmed findings separated by commas.</td></tr></table>

Generation uses greedy decoding (do\_sample=False), at most 512 new tokens, fixed end-of-sequence and padding identifiers, and Qwen3.5 thinking mode disabled. Post-processing removes any thinking tags, retains the final line for classification, normalizes multi-label delimiters, extracts detection boxes and numeric answers, and retains report text. If no numeric value is found, the numeric parser returns zero. We apply no calibration, re-querying, or manual answer correction.

The container packages the backbone, adapters, tokenizer, and router for fully ofline execution. Its runtime, size, and repeated-run checks are reported in Section 4.4.

## 3 Experiments

## 3.1 Dataset and evaluation measures

Dataset. The cleaned source pool for FLARE 2026 Task 3 [6] contained 51,154 question–answer pairs from the organizer training and public-validation data. After the exclusions described in Section 2.1, our evaluation covers 17 datasets, seven imaging modalities, and six scored task families.

Following recommended cross-validation practice for medical-imaging AI [3], we partitioned the organizer training data by image group, stratified by dataset and task combination, into an approximately 10% internal holdout and two 45% subsets, A and B (Fig. 2). Smaller subsets $A ^ { \prime } \subset A$ and $B ^ { \prime } \subset B .$ , each containing 30% of its parent partition, supported inexpensive preliminary screening before selected configurations were evaluated on the full $A / B$ partitions. Full $A  B$ and B → A experiments were run where available. These partitions were fixed before expanding multi-image examples into single-image training rows.

Adapted models in the internal and public-validation comparisons are trained on A ∪ B; public-validation images overlapping the organizer training data were excluded. Although internal-holdout images were excluded from these adapters training, holdout results informed some development choices. We therefore treat the holdout results as development comparisons rather than independent estimates of generalization [20]. Selected deployment routes were later refitted with the holdout and public-validation data (Table 1); scores on these previously seen cases are not reported as held-out performance.

![](images/5969c85ce1bad56645faa2f7f1b45c65d934e59ef6321edf9d663a87b25b05e7.jpg)  
Fig. 2. Image-grouped development partitions and configuration-screening subsets. Percentages are approximate; $A ^ { \prime }$ and $B ^ { \prime }$ each contain 30% of their parent partition. Public-validation images overlapping the organizer training data were removed.

Evaluation. We use balanced accuracy (mean class recall) for classification and micro-averaged F1 for multi-label classification. Detection uses F1 with box matching at intersection-over-union $> 0 . 5$ . Counting and regression use mean absolute error (MAE), and report generation uses the language-model-based CRIMSON metric [1].

We evaluate the internal and public splits separately, computing each metric per dataset and reporting its unweighted mean within each task. Counting and report generation each contain one evaluated dataset. Excluding endoscopy, internal/public classification covers ten/six datasets, multi-label classification two/one, and detection three/two. Regression has two internal datasets and no public cases. Organizer hidden-validation scores (Section 4.1) instead pool cases within each task. Diferences in dataset composition and aggregation prevent direct comparison with our dataset-macro results.

Statistical uncertainty. Paired 95% confidence intervals in the configuration ablations use 10,000 bootstrap resamples of matched question–answer cases, sampled with replacement within each dataset using the same sampled indices for both configurations. For the general-versus-specialist regression comparison, interval bounds are the 2.5th and 97.5th percentiles of the resampled MAE differences. This case-level resampling treats question–answer cases as independent; the intervals do not account for dependence between questions sharing an image or variation across training seeds.

Comparison protocols. We compare A ∪ B adapters with two zero-shot baselines on the same splits. The first uses the deployed prompts, answer prefixes, extraction rules, and 512-token limit. The format-assisted baseline adds explicit output-format instructions, removes the answer prefix, changes answer extraction, and allows 1,024 tokens. Both retain the same 4-bit Qwen3.5-9B backbone, greedy decoding, disabled thinking mode, and task-specific image budgets (768<sup>2</sup> for reports, 2048<sup>2</sup> otherwise). Thus, their diference measures a combined inference-protocol change, not a prompt-only efect. These evaluations select configurations using known task assignments and do not measure router accuracy or end-to-end container performance. Report generation uses only the first baseline; comparisons retain reports scored successfully for both models (780 internal and 1,943 public cases).

## 3.2 Implementation details

Environment settings Table 2 summarizes the single-GPU development environments used with LLaMA-Factory.

Table 2. Development environment.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>System</td><td>Ubuntu 22.04.5 LTS (A5000 workstation)</td></tr><tr><td>CPU / RAM</td><td>32 cores / 256 GB (A5000 workstation)</td></tr><tr><td>GPUs</td><td>NVIDIA RTX A5000 (24 GB)</td></tr><tr><td>Framework</td><td>NVIDIA RTX PRO 6000 Blackwell (96 GB) PyTorch 2.11.0, CUDA 12.8</td></tr><tr><td>Training library LLaMA-Factory 0.9.5</td><td></td></tr><tr><td>Dependencies</td><td>Transformers 5.16.1, PEFT 0.18.1</td></tr><tr><td></td><td>bitsandbytes 0.50.2</td></tr><tr><td>Code</td><td>https://github.com/TahaSabir0/FLARE26-Hexapus</td></tr></table>

Training protocols We applied QLoRA to all linear layers of the language model while keeping the pretrained weights, vision encoder, and multimodal projector frozen. Supervised fine-tuning used next-token cross-entropy loss on answer tokens only. Table 3 lists the shared settings. Runs used one GPU and an efective batch size of 8: batch size 2 with four accumulation steps on Blackwell, or batch size 1 with eight steps on the A5000.

An earlier single-seed Qwen3-VL-8B classification screen favored three epochs over two or four (0.5964 versus 0.5697 and 0.5876 dataset-macro balanced accuracy on $A ^ { \prime }  B ^ { \prime }$ at 768<sup>2</sup>). We transferred this schedule to Qwen3.5-9B without treating it as optimal for the final backbone or every task.

Table 3. Shared training settings for the QLoRA adapters.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Effective batch size</td><td>8</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Initial learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate schedule</td><td>Linear decay, 3% warm-up</td></tr><tr><td>Maximum sequence length 8,192 tokens</td><td></td></tr><tr><td>Compute precision</td><td>BF16</td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td></tr><tr><td>Random seed</td><td>1337</td></tr></table>

Each adapter was evaluated at its training resolution using the final checkpoint, without early stopping, extra augmentation, or patch sampling. All reported Qwen3.5-9B experiments use one random seed; Section 3.1 describes configuration selection.

## 4 Results and discussion

## 4.1 Results on the validation sets

## Internal and public validation.

Figure 3 compares the route-matched A ∪ B adapters with the two zero-shot protocols. These adapters exclude internal holdout and public-validation images from training; they are proxies for deployment routes subsequently refitted on the full labeled pool. Table 4 gives the internal per-dataset results.

Output conventions materially afect zero-shot performance. Format assistance raises classification balanced accuracy from 0.0035 to 0.3982 internally and from 0.0055 to 0.4229 publicly; adaptation then reaches 0.7791 and 0.7418. The same protocol change does not help multi-label classification or counting, and near-zero detection scores remain sensitive to coordinate conventions and answer extraction rather than measuring localization alone.

![](images/d5ab2c27a3674861532e8d9aa0234dadefb1cf40f2ec137b101b164e8150e078.jpg)

![](images/ba2068a78c4a9b9fddee9ffc613e9fd65b3c13a8c9c1dced044b0a0d4fa60c4f.jpg)

![](images/bdb142859b05af4f62e3f992ee700f6b536efe105a79cf448f3c61f5d633f90d.jpg)

![](images/340257dfcd05e311fc08cfdaa3e28c4df494c51a767074a4cd072d079daa41bd.jpg)

![](images/f50cb5ecc6b7511ff5bcd2d9f78e7a46972c5cf4469cd1626bc2895751d0e775.jpg)

![](images/b93efa3d85ebde31352ea6aa1972c66963b027f79d67a98dc90b4bb899fd6798.jpg)  
Fig. 3. Zero-shot and adapted Qwen3.5-9B on internal and public-validation data, using the protocols in Section 3.1. BA denotes balanced accuracy; multi-label F1 is micro-averaged within each dataset. Task scores are dataset macros, except singledataset counting and report generation. Report means use the same successfully scored cases for both models; no format-assisted report run was performed. Public validation has no regression cases. Arrows indicate the better direction.

Adaptation improves every evaluated task over the available zero-shot protocols, although large errors remain. Relative to the deployed-settings baseline, counting MAE falls from 118.14 to 11.73 internally and from 405.60 to 213.66 publicly, where dense cell images remain dificult. On jointly scored reports, CRIMSON rises from 0.7141 to 0.8208 internally and from 0.8151 to 0.8622 publicly.

Organizer hidden validation. The hidden-validation submission was scored on 1,703 cases spanning classification, multi-label classification, and regression (Table 5). The oficial overall score was 0.4825. These metrics are computed over each task’s full scoring surface; in particular, the challenge’s sample-averaged multi-label F1 is distinct from the per-dataset micro-F1 used in our internal and public-validation comparisons.

Table 4. Per-dataset results of the route-matched Qwen3.5-9B adapters on the internal holdout. The report score uses all 781 adapted-model scores; Fig. 3 uses the 780 reports jointly scored with zero-shot.
<table><tr><td>Task</td><td>Dataset</td><td>Metric</td><td>n</td><td>Resolution</td><td>Qwen3.5-9B</td></tr><tr><td>Classification</td><td>CMMD</td><td>BA</td><td>357</td><td> $\overline { { 2 0 4 8 ^ { 2 } } }$ </td><td>0.5970</td></tr><tr><td></td><td>BCN20000</td><td>BA</td><td>286</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.6900</td></tr><tr><td></td><td>Dental</td><td>BA</td><td>420</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.6700</td></tr><tr><td></td><td>Neojaundice</td><td>BA</td><td>60</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.8100</td></tr><tr><td></td><td>Bone marrow</td><td>BA</td><td>72</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.8660</td></tr><tr><td></td><td>BUS-UCLM</td><td>BA</td><td>52</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.9190</td></tr><tr><td></td><td>Fundus</td><td>BA</td><td>138</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.6870</td></tr><tr><td></td><td>IUGC</td><td>BA</td><td>131</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.9300</td></tr><tr><td></td><td>Retino</td><td>BA</td><td>111</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.7610</td></tr><tr><td>Multi-label</td><td>BUSI</td><td>BA</td><td>62</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.8610</td></tr><tr><td></td><td>ChestDR</td><td>F1</td><td>388</td><td> $\overline { { 2 0 4 8 ^ { 2 } } }$ </td><td>0.5625</td></tr><tr><td></td><td>Periapical</td><td>F1</td><td>315</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.6947</td></tr><tr><td>Counting</td><td>NeurIPS22Cell</td><td>MAE</td><td>100</td><td> $\overline { { 2 0 4 8 ^ { 2 } } }$ </td><td>11.7300</td></tr><tr><td>Detection</td><td>IUGC</td><td>F1</td><td>516</td><td> $\overline { { 2 0 4 8 ^ { 2 } } }$ </td><td>0.9981</td></tr><tr><td></td><td>BUSI-det</td><td>F1</td><td>50</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.8317</td></tr><tr><td></td><td>BUS-UCLM- det</td><td>F1</td><td>22</td><td> $2 0 4 8 ^ { 2 }$ </td><td>0.8444</td></tr><tr><td>Regression</td><td>Boneresorption MAE</td><td></td><td>70</td><td> $\overline { { 2 0 4 8 ^ { 2 } } }$ </td><td>17.1600</td></tr><tr><td></td><td>IUGC</td><td>MAE</td><td>516</td><td> $2 0 4 8 ^ { 2 }$ </td><td>2.4100</td></tr><tr><td>Report generation</td><td>IU-XRay</td><td>CRIMSON</td><td>781</td><td> $7 6 8 ^ { 2 }$ </td><td>0.8204</td></tr></table>

Table 5. Oficial hidden-validation performance of the submitted Qwen3.5-9B system. Scores are pooled by task.
<table><tr><td>Task</td><td>Metric</td><td>n</td><td>Score</td></tr><tr><td>Classification</td><td>Balanced accuracy ↑</td><td>1,126</td><td>0.8496</td></tr><tr><td>Multi-label classification</td><td>Sample-averaged F1 ↑</td><td>477</td><td>0.5338</td></tr><tr><td>Regression</td><td>MAE ↓</td><td>100</td><td>14.5950</td></tr></table>

## 4.2 Configuration ablations

General versus task-specific adapters. Task-specific configurations produced uneven benefits over the shared general adapter (Table 6(a)). All rows compare matched $7 6 8 ^ { 2 }$ input. Specialists score higher on four of six tasks, including the largest scored slice, classification, while the multi-label and reportgeneration comparisons show no clear diference.

On Boneresorption, the specialist reduces MAE by 3.93 (paired 95% confidence interval [0.83, 7.06]) and produces 12 exact-zero answers versus 29 for the general adapter. This supports the specialist recipe on that dataset, but does not measure router accuracy.

Table 6. Configuration ablations on the internal holdout. (a) General adapter versus task specialists at matched $7 6 8 ^ { 2 }$ input. (b) Resolution comparisons for task specialists. Regression is the mean MAE over two datasets.
<table><tr><td colspan="4">(a) General versus task-specific adapters</td></tr><tr><td>Task</td><td>Metric</td><td>General</td><td>Specialist</td></tr><tr><td>Classification</td><td>BA↑</td><td>0.7348</td><td>0.7695</td></tr><tr><td>Multi-label classification</td><td>F1 ↑</td><td>0.6157</td><td>0.6130</td></tr><tr><td>Counting</td><td>MAE↓</td><td>14.3400</td><td>13.4500</td></tr><tr><td>Detection</td><td>F1 ↑</td><td>0.8876</td><td>0.9194</td></tr><tr><td>Regression</td><td>MAE↓</td><td>12.1600</td><td>10.1800</td></tr><tr><td>Report generation</td><td>CRIMSON ↑</td><td>0.8307</td><td>0.8204</td></tr><tr><td>(b) Image-resolution budget</td><td></td><td></td><td></td></tr><tr><td>Task</td><td>Metric</td><td>768²</td><td>2048²</td></tr><tr><td>Classification</td><td>BA↑</td><td>0.7695</td><td>0.7791</td></tr><tr><td>Multi-label classification</td><td>F1 ↑</td><td>0.6130</td><td>0.6286</td></tr><tr><td>Detection</td><td>F1 ↑</td><td>0.9194</td><td>0.8914</td></tr><tr><td>Regression</td><td>MAE↓</td><td>10.1800</td><td>9.7800</td></tr><tr><td>Counting</td><td>MAE ↓</td><td>13.4500</td><td>11.7300</td></tr></table>

Image resolution. Raising the image budget from $7 6 8 ^ { 2 }$ to $2 0 4 8 ^ { 2 }$ has mixed efects (Table 6(b)). Classification BA increases by 0.0096, multi-label F1 increases by 0.0156, counting MAE decreases by 13%, and regression macro MAE decreases by 0.40, while detection F1 decreases by 0.0280. The paired 95% interval for the multi-label change is $[ - 0 . 0 0 4 , + 0 . 0 3 5 ]$ , and both per-dataset regression intervals include zero. Runs also used diferent GPUs and micro-batch shapes at the same efective batch size, so the changes cannot be attributed to image detail alone. Although the 768<sup>2</sup> A ∪ B detection adapter scored 0.9194, it is not the deployed route: the full-data detection adapter was trained at 2048<sup>2</sup>. Table 4 therefore reports the route-matched A ∪ B proxy at $2 0 4 8 ^ { 2 }$ (0.8914 F1). Regression likewise remains at $2 0 4 8 ^ { 2 }$ , while report generation uses 768<sup>2</sup>.

## 4.3 Strengths and remaining failure modes

An output-format audit of internal adapted-model predictions found valid nested coordinate lists in all 588 detection responses, valid integers in all counting responses, and no empty answers or leaked reasoning tags. The remaining errors mainly reflect visual recognition and numerical estimation rather than syntax.

Figure 4 illustrates representative successes and task-specific failures across categorical, numerical, and spatial outputs. All displayed outputs are syntactically valid; the remaining errors arise from visual category confusion, a degenerate numerical estimate, or incomplete localization.

Report generation remains the main semantic limitation. Among 223 cases with abnormal reference reports, 137 (61.4%) receive normal-only predictions. Only 16.4% of predictions are unique, and the ten most frequent outputs account for 60.6%, indicating heavy reuse of report templates. The resulting omissions are not apparent from the aggregate CRIMSON score of 0.8204.

![](images/5ca37e2ebfce28f447663a0548d879108bcdc4bb8cb94c2cf01129c1876fbd55.jpg)

![](images/518126d8e4fbe11ce1593c36bd72e186ed6033aa9910580baef693620e5bebff.jpg)  
(a) Classification, BUS-UCLM. Question: What does the contour-defined zone in the breast ultrasound scan signify? Options: A) Benign lesion present; B) Malignant lesion present; C) No lesion, only normal tissue. Ground truth: C. Prediction: C.  
(b) Classification, BUS-UCLM. Question: What does the contour-defined zone in the breast ultrasound scan signify? Options: A) Benign lesion present; B) Malignant lesion present; C) No lesion, only normal tissue. Ground truth: B. Prediction: A.

![](images/c81f5ee917c839eca90bf26cb7f0615f3e4c4e1261b9f5e8284e5e99b005345b.jpg)

![](images/3b0f1850a744cbb4ce84c5477a79ee48e2d77c71f4ed45c0d60ca1a09d42076c.jpg)  
(c) Regression, Boneresorption. Question: Assess the rate of alveolar bone resorption (ABR%) from this orthopantomogram. Ground truth: 24.2. Prediction: 25.0. Absolute error: 0.8.  
(d) Regression, Boneresorption. Question: Estimate the alveolar bone resorption percentage (ABR%) based on the radiographic findings in this panoramic image. Ground truth: 31.1. Prediction: 0.0. Absolute error: 31.1.

![](images/d6de9bccafa6a53cbecdfbbf7f0a48ea556ac4cefcbf998cbbfa5d2cf72e3f4e.jpg)  
(e) Detection, BUS-UCLM-det. IoU: 0.68. Question: Report the spatial coordinates of lesions in the ultrasound scan as [x<sub>min</sub>, y<sub>min</sub>, x<sub>max</sub>, y<sub>max</sub>], structuring multiple lesions as a nested list. Ground truth: [[274,90,651,469]]. Prediction: [[252,82,752,500]].

![](images/42d5d002fdc1457bc4b3ef91cb050d8f45f4fe6e73af8169e30381007fd33745.jpg)  
(f) Detection, BUS-UCLM-det. IoU: 0.25. Question: Pinpoint lesion regions in the ultrasound image with [x<sub>min</sub>, y<sub>min</sub>, x<sub>max</sub>, y<sub>max</sub>] coordinates, nesting lists for multiple findings. Ground truth: [[226,84,689,465]]. Prediction: [[261,100,508,282]].  
Fig. 4. Representative successes or low-error predictions (left) and failures (right). Classification and detection use non-overlapping validation-public cases; regression uses the internal holdout because public validation has no regression cases. Blue boxes show ground truth and orange boxes show predictions. All examples use task-specific A ∪ B adapters.

Table 7. Final container verification on 32 cases covering all six tasks.
<table><tr><td>Measure</td><td>Observed</td><td>Limit or check</td></tr><tr><td>Mean runtime (incl. model loading)</td><td>3.1 s/case</td><td>≤ 6.57 s/case</td></tr><tr><td>Container image size</td><td>30.1 GB</td><td>≤ 35 GB</td></tr><tr><td>Repeated-run identity</td><td>32/32 cases</td><td>Identical outputs</td></tr></table>

## 4.4 Deployment eficiency

The final container meets the runtime and size limits and produces identical outputs across two runs (Table 7).

## 4.5 Results on the Final Testing Set

Table 8 reports the organizer-announced final-testing results for our team, Hexapus, in the 2D track. These results are distinct from the hidden-validation and internal dataset-macro results above.

Table 8. Organizer-reported final-testing results for Hexapus.
<table><tr><td>Balanced accuracy</td><td>Multi-label F1</td><td>Detection F1</td><td>MAE</td></tr><tr><td>0.8585</td><td>0.5310</td><td>0.8633</td><td>18.5286</td></tr></table>

## 4.6 Limitations and future work

The organizer hidden-validation score covers classification, multi-label classification, and regression only. The announced final-testing results additionally reports detection F1, but no counting or report-generation scores. Evidence for these two tasks is therefore limited to the internal and public-validation sets, which difer in composition and are not directly comparable. Our Qwen3.5-9B experiments also use a single training seed. The 2048<sup>2</sup> production-adapter runs peaked at approximately 40–44 GB of GPU memory, requiring a 48 GB-class or larger GPU, whereas the lower-resolution runs fit on our 24 GB workstation GPUs; the resolution comparisons also difered in GPU class and micro-batch shape at the same efective batch size. Future work should confirm these task-dependent efects across seeds, quantify errors introduced by the question-only router, and investigate selective high-resolution processing or adaptive tiling under matched compute budgets. A promising extension would combine our adapter routing with the multi-level visual feature fusion of QwenVLConnector [13], letting the router also select a specialized visual pathway for fine spatial detail.

## 5 Conclusion

We demonstrate that a single quantized VLM can support six heterogeneous medical image-parsing tasks without a separate backbone for each one. The submitted 30.1 GB system averaged 3.1 seconds per case and achieved an overall score of 0.4825 on organizer hidden validation across classification, multilabel classification, and regression. Our controlled comparisons show no universal adaptation recipe: specialist adapters improved selected structured tasks, higher resolution substantially benefited dense counting but produced smaller or inconsistent gains elsewhere, and the general adapter remained competitive for some routes. Report generation exposed a diferent limitation: structurally valid outputs could still omit clinically relevant findings, with 61.4% of abnormalreference cases receiving normal-only predictions. A shared backbone should therefore unify deployment, not force uniform training or inference; future generalist medical VLMs should allocate adapter capacity and visual detail by task demand and be evaluated per dataset and failure mode, not aggregate scores.

Acknowledgments The authors of this paper declare that the proposed solution is fully automatic without any manual intervention. We acknowledge the Mohamed bin Zayed University of Artificial Intelligence Undergraduate Research Internship Program (MBZUAI UGRIP) for supporting this work. We thank all data owners and contributors for making the data publicly available and Codabench [22] for hosting the challenge platform.

## Disclosure of Interests

The authors declare no competing interests.

## References

1. Baharoon, M., Heintz, T., Raissi, S., et al.: CRIMSON: A clinically-grounded LLM-based metric for generative radiology report evaluation. arXiv preprint arXiv:2603.06183 (2026), https://arxiv.org/abs/2603.06183 6

2. Bai, S., Chen, K., Liu, X., et al.: Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923 (2025) 2

3. Bradshaw, T.J., Huemann, Z., Hu, J., Rahmim, A.: A guide to cross-validation for artificial intelligence in medical imaging. Radiology: Artificial Intelligence 5(4), e220232 (2023). https://doi.org/10.1148/ryai.220232 6

4. Chen, J., Gui, C., Ouyang, R., et al.: Towards injecting medical visual knowledge into multimodal LLMs at scale. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 7346–7370 (2024) 2

5. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: QLoRA: Eficient finetuning of quantized llms. In: Advances in Neural Information Processing Systems. vol. 36, pp. 10088–10115. Neural Information Processing Systems Foundation, Inc. (2023). https://doi.org/10.52202/075280-0441 2

6. MICCAI FLARE 2026 Task 3: Multimodal model for medical image parsing. Codabench (2026), https://www.codabench.org/competitions/7151/, accessed 21 September 2026 2, 3, 6

7. He, P., Gao, J., Chen, W.: DeBERTaV3: Improving DeBERTa using ELECTRAstyle pre-training with gradient-disentangled embedding sharing. In: The Eleventh International Conference on Learning Representations (2023), https: //openreview.net/forum?id=sE7-XhLxHA 2

8. Hu, E.J., Shen, Y., Wallis, P., et al.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022) 2, 4

9. Jiang, S., Zheng, T., Zhang, Y., Jin, Y., Yuan, L., Liu, Z.: Med-MoE: Mixture of domain-specific experts for lightweight medical vision-language models. arXiv preprint arXiv:2404.10237 (2024), https://arxiv.org/abs/2404.10237 2

10. Li, C., Wong, C., Zhang, S., et al.: LLaVA-Med: Training a large language-andvision assistant for biomedicine in one day. In: Advances in Neural Information Processing Systems (2023) 2

11. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (2023) 2

12. Ma, J., Gu, S., Wang, B.: Fast, low-resource, accurate, robust, and efectual medical image analysis (flare) 2025. Zenodo (2025). https://doi.org/10.5281/ zenodo.15094799, https://zenodo.org/records/15094799, medical Image Computing and Computer Assisted Intervention 2025 (MICCAI) 2

13. Nguyen, L.T.P., Nguyen, T., Nguyen, T.H., Hoang, G.M., Vu, A.M., Bagci, U.: QwenVLConnector: A fast, unified medical VLM chatbot for fine-grained clinical perception and text generation. arXiv preprint arXiv:2609.23139 (2026), https: //arxiv.org/abs/2609.23139 2, 13

14. Qwen Team: Qwen3.5: Towards native multimodal agents (2026), https://qwen. ai/blog?id=qwen3.5 2

15. Sachdeva, B., Uppal, K., Java, A., Balasubramanian, V.N.: Understanding task transfer in vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026), https://arxiv.org/abs/ 2511.18787, oral presentation 2

16. Sellergren, A., Kazemzadeh, S., Jaroensri, T., et al.: MedGemma technical report. arXiv preprint arXiv:2507.05201 (2025) 2

17. Shaaban, M.A., Saqib, A., Hardan, S.E., Taratynova, D., Saleem, T.J., Yaqub, M.: ME-VLIP: A modular and eficient vision-language framework for generalizable medical image parsing. MICCAI 2025 FLARE Challenge paper (2025), https: //openreview.net/forum?id=0LYAs4b8T6 2, 3

18. Stepanov, I., Shtopko, M., Vodianytskyi, D., Lukashov, O., Yavorskyi, A., Yaroshenko, M.: GLiClass: Generalist lightweight model for sequence classification tasks. arXiv preprint arXiv:2508.07662 (2025), https://arxiv.org/abs/ 2508.07662 2

19. Sun, H., Xue, L., Jia, L.: Vision-language model for multitask medical text generation. MICCAI 2025 FLARE Challenge paper (2025), https://openreview.net/ forum?id=ovT0OkUDXg 2

20. Varma, S., Simon, R.: Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7, 91 (2006). https://doi.org/10.1186/ 1471-2105-7-91 6

21. Wei, J., Wang, X., Schuurmans, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. In: Advances in Neural Information Processing Systems. vol. 35, pp. 24824–24837. Neural Information Processing Systems Foundation, Inc. (2022). https://doi.org/10.52202/068431-1800 5

22. Xu, Z., Escalera, S., Pavão, A., et al.: Codabench: Flexible, easy-to-use, and reproducible meta-benchmark platform. Patterns 3(7), 100543 (2022) 14

23. Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z.: LlamaFactory: Unified eficient fine-tuning of 100+ language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). pp. 400–410. Association for Computational Linguistics (2024). https://doi.org/10.18653/v1/2024.acl-demos.38, https:// aclanthology.org/2024.acl-demos.38/ 2

24. Zhong, Y., Jin, R., Dou, Q., Li, X.: Can generalist vision language models (VLMs) rival specialist medical VLMs? benchmarking and strategic insights. arXiv preprint arXiv:2506.17337 (2025), https://arxiv.org/abs/2506.17337 2

Table 9. Checklist Table. Please fill out this checklist table in the answer column.
<table><tr><td>Requirements</td><td>Answer</td></tr><tr><td>A meaningful title</td><td>Yes</td></tr><tr><td>The number of authors (≤6)</td><td>6</td></tr><tr><td>Author affiliations and ORCID</td><td>Yes</td></tr><tr><td>Corresponding author email is presented</td><td>Yes</td></tr><tr><td>Validation scores are presented in the abstract Introduction includes at least three parts:</td><td>Yes</td></tr><tr><td>background, related work, and motivation</td><td>Yes</td></tr><tr><td>A pipeline/network figure is provided</td><td>Figure 1</td></tr><tr><td>Pre-processing</td><td>Section 2.1</td></tr><tr><td>Strategies to use the partial label</td><td>N/A</td></tr><tr><td>Strategies to use the unlabeled images.</td><td>N/A</td></tr><tr><td>Strategies to improve model inference</td><td>Section 2.4</td></tr><tr><td>Post-processing</td><td>Section 2.4</td></tr><tr><td>The dataset and evaluation metric section are presented Section 3.1</td><td></td></tr><tr><td>Environment setting table is provided</td><td>Table 2</td></tr><tr><td>Training protocol table is provided</td><td>Table 3</td></tr><tr><td>Ablation study</td><td>Table 6</td></tr><tr><td>Efficiency evaluation results are provided</td><td>Table 7</td></tr><tr><td>Visualized segmentation example is provided</td><td>N/A (no segmentation task)</td></tr><tr><td>Limitation and future work are presented</td><td>Yes</td></tr><tr><td>Reference format is consistent.</td><td>Yes</td></tr></table>