# Vision-Language Models for Analog Gauge Reading: An Empirical Study of Specialization, Transfer and Reliability

Abdul Mueez<sup>a</sup>, Aaditya Baranwal<sup>a</sup>, Junior Chaj-Mejia<sup>a</sup>, Guneet Bhatia<sup>b</sup>, Jason T. Voelker<sup>b</sup>, Shruti Vyas<sup>a,∗</sup>

<sup>a</sup>University of Central Florida, Orlando, FL, United States <sup>b</sup>Siemens Energy, Orlando, FL, United States

## Abstract

Analog gauges remain common in industrial environments where manual inspection is costly or hazardous. The engineering application addressed here is direct numerical reading of single-target analog-gauge images, while the artificial-intelligence contribution is a systematic evaluation of specialization, transfer, robustness and reliability for a general-purpose vision-language model (VLM) without an explicit pointer-segmentation and geometric-reading pipeline. The Qwen2.5-VL-7B-Instruct model is evaluated using zero-shot prompting, in-context learning (ICL) and parameter-eficient fine-tuning with Quantized Low-Rank Adaptation (QLoRA) on a public synthetic dataset, a video-derived Pressure Gauge dataset and a proprietary industrial dataset. All fine-tuning experiments use a fixed 20-epoch protocol with the final epoch used for analysis; separate models with and without supplied gauge ranges remove prompt-setting confounds. The primary metric is range-normalized mean

percentage error (MPE). The best fine-tuned MPE values are 2.39% on the synthetic dataset, with a 95% bootstrap confidence interval (CI) of 1.43–3.90%; 2.61% on the Pressure Gauge dataset, with a CI of 1.66–3.80%; and 4.43% on the proprietary industrial dataset, with a CI of 2.31–7.14%. Leave-one-datasetout experiments reveal substantial transfer degradation on held-out synthetic and proprietary data, while robustness tests identify Gaussian blur as the strongest tested corruption. Reliability analysis shows that high-confidence errors remain possible, motivating abstention and independent validation in safety-critical use. These results support QLoRA-specialized VLMs for direct single-gauge reading but not yet a deployment-ready plant-monitoring pipeline.

Keywords: Industrial automation, Analog gauge reading, Vision-language model, Quantized low-rank adaptation, Parameter-eficient fine-tuning

## 1. Introduction

Analog gauges are ubiquitous instruments for monitoring critical parameters such as pressure, temperature and flow rate across manufacturing facilities, power substations, process plants and other industrial environments [1, 2, 3, 4, 5]. Accurate and timely readings are important for operational safety, maintenance and process oversight, yet repeated manual inspection is labor intensive, vulnerable to transcription error and potentially hazardous in dificult-to-access locations [1, 4].

Analog gauges persist because they are simple, locally readable and can operate without an electrical data interface. Their mechanical simplicity can also be attractive in harsh or electromagnetically noisy environments [6].

Rather than replacing this installed base, an image-based monitoring layer can digitize readings while preserving the underlying instrument.

Traditional automated gauge readers commonly decompose the problem into gauge localization, perspective correction, pointer or keypoint detection, scale extraction and geometric conversion [7, 5, 8, 1]. These pipelines can be efective, but each stage introduces assumptions about geometry, image quality or gauge design. Direct regression approaches reduce some of this complexity, although data scarcity and domain shift remain important concerns [2, 9, 10].

Large vision-language models (VLMs) provide a diferent starting point: a pretrained multimodal representation can be adapted to produce the gauge value directly from an image and textual instruction. However, strong generic visual-language capability does not imply metrological reliability. The central objective of this work is therefore to determine how much task-specific specialization is required for accurate analog-gauge reading and to characterize where a specialized VLM remains unreliable.

This study makes four contributions:

1. It systematically compares zero-shot prompting, in-context learning (ICL) and QLoRA (Quantized Low-Rank Adaptation)-based parametereficient fine-tuning of Qwen2.5-VL-7B-Instruct.

2. It trains separate with-range and without-range models, providing a clean metadata ablation.

3. It adds leave-one-dataset-out (LODO) and adverse-condition corruption analyses to probe transfer and robustness.

4. It expands evaluation beyond MPE through bootstrap confidence intervals, MAE, RMSE, tail-error statistics, operational tolerance rates, parsing failures, calibration metrics and explanation-method comparisons.

The scope is deliberately narrower than full plant-scene automation. Each evaluated input contains one target gauge, but it is not necessarily a tight gauge crop: multiple images retain substantial surrounding scene context and the gauge occupies only a small portion of the frame. Only source images containing multiple gauges in one of the datasets were cropped to isolate one gauge per sample. The model predicts the reading directly from these single-target-gauge images; detecting and selecting among multiple gauges and plant-level acquisition or alarm integration remain outside the evaluated pipeline.

## 2. Related Work

## 2.1. Multi-stage computer-vision gauge readers

Classical analog-gauge readers typically use geometric image processing. Circular boundaries can be estimated with Hough transforms or ellipse fitting, the pointer can be extracted with line detection, polar projection or principalcomponent analysis and the final value can be computed from the pointer angle and scale geometry [11, 12, 13, 7]. These methods are interpretable and computationally eficient, but sensitivity to glare, viewing angle, occlusion and non-standard dial layouts can propagate through the pipeline.

Recent learning-based systems retain much of this staged structure while replacing individual components with neural networks. Examples include segmentation-based pointer-meter reading [14], detection and recognition systems for substations [15], keypoint-based mobile gauge transcription [5],

GaugeTracker [4] and learning-based gauge reading in unconstrained imagery [8]. A recent outdoor-substation method combines learned dial segmentation with distortion correction and scale recognition [1]. Collectively, these approaches show that strong performance is possible, but they also illustrate the diversity of preprocessing, geometric assumptions and supervision required across gauge-reading systems.

## 2.2. Direct regression and compact learned baselines

An alternative is to regress the reading directly, avoiding explicit pointer geometry. CNN-based direct regressors and multi-task regression systems have demonstrated the feasibility of this formulation [2, 9, 10]. MobileNetV2 [16] is an attractive compact backbone because of its low computational cost. However, direct regression can become dificult when the same apparent pointer position corresponds to diferent engineering values across gauge ranges and evaluation can be confounded when diferent target parameterizations are used across experimental conditions. For this reason, the MobileNetV2 comparison in this work is treated as a lightweight reference rather than a clean test of whether range metadata is visually inferred.

## 2.3. Vision-language models and model selection

VLMs couple visual representations with autoregressive language generation, providing a flexible interface for tasks that combine visual parsing and textual instructions [17]. Qwen2.5-VL-7B-Instruct was selected because it is an open-weight model with documented dynamic-resolution visual processing and strong general-purpose localization, document and structured visual-understanding capability [18]. These properties are relevant to analog gauges, where small ticks, printed scale values and pointer geometry must be interpreted jointly. The 7B-scale model is also small enough to adapt locally with QLoRA while retaining substantially greater capacity than compact VLM families such as SmolVLM2.

The present study does not claim an exhaustive same-split comparison against every contemporary industrial inspection architecture. Instead, it asks whether specialization can turn a general-purpose VLM into an accurate direct gauge reader and then subjects that model to controlled metadata ablation, cross-domain transfer, robustness and reliability analyses. This distinction is important because published gauge-reading systems often use diferent datasets, crop policies, sampling strategies and unavailable implementations, preventing a fully controlled cross-paper ranking.

## 3. Datasets

This work is evaluated on three datasets: the Synthetic dataset, the Pressure Gauge video dataset and the Siemens Energy (SE) dataset. Descriptive dataset names are used throughout the paper for readability. For compact labels in tables and figures, Pressure denotes the Pressure Gauge dataset.

## 3.1. Synthetic Dataset

A synthetic dataset titled “Synthetic Data for Precision Gauge Reading” created by Endava and distributed through Kaggle [19] was used. The “Synthetic DS6.0” subset used here contains 501 gauge images with annotations for the reading and scale information. A random train-test split produced 375 training images and 126 test images. The source dataset contains additional component masks and keypoints, but only the gauge reading and scale endpoints are required by the present experiments. Representative Synthetic samples are shown in Figure 1.

![](images/81895eccda01a40865bc499d93d16b144db6c91abb2d3010583bf85c31731c56.jpg)

![](images/52db4009a2789ef14bfe3de3623105a2c3918bf17a8de83fcf3b4e25158d56f6.jpg)

![](images/6fd3d278805b846cdbe80bb286a3a1f0934ef416a6023f3d73de9eb3e039bc1a.jpg)  
Figure 1: Examples from the public Synthetic Data for Precision Gauge Reading dataset [19].

## 3.2. Pressure Gauge Video Dataset

The Pressure Gauge dataset is derived from the “Pressure Gauge Reader Data” video collection [20, 12]. Frames were extracted at one-second intervals and then filtered to retain unique needle positions. The source training partition consists of three videos of the same gauge captured under slightly diferent lighting and framing conditions. The source test partition contains seven videos: two show the same gauge with minor lighting/framing changes, two introduce angle variation and three feature diferent gauges. The final split contains 57 training images and 36 test images. Importantly, the source dataset separates training and test videos: the training images used here come from videos out2, out3 and out5, whereas the test images come from man1 through man7. Consequently, no video source is shared between the Pressure Gauge training and test partitions. This establishes video-level separation, but does not by itself establish gauge-identity or acquisitioncondition independence. The exact filename-level split will be included in the public release described in Section ??.

The source dataset does not provide the numerical annotations required by the present regression evaluation. A first annotator assigned the gauge reading and scale endpoints for each selected frame. A second person then verified all annotations and made changes as necessary. Representative Pressure Gauge frames are shown in Figure 2.

![](images/1c3505084d73e31ecf0f7b4484a7ded265c2222506456150b48cb2e57ea3094e.jpg)

![](images/4bdcee988f78268bb79d8bd7d82cebc3a7c417f0f0fa797d96ff239402ea5abc.jpg)

![](images/e32dc4c6ff34ba8147a7f07334d6edaa760a9534d4360464c81a6bec83031ddb.jpg)  
Figure 2: Example frames from the Pressure Gauge video collection [20].

## 3.3. Siemens Energy Dataset

The SE dataset contains gauge photographs acquired in real industrial environments. Raw images cannot be publicly distributed because of proprietary and site-sensitivity constraints. The image collection was randomly divided into approximately 75% training and 25% testing, yielding 132 original training images and 43 test images. Only training images containing more than one gauge were cropped to isolate individual gauge instances, increasing the training count to 152. Images that already contained one gauge were retained without imposing a tight gauge crop; consequently, some samples include substantial surrounding scene context and the target gauge occupies only a small portion of the image. The 43 test images each contain one target gauge and required no multi-gauge instance extraction.

Each SE sample used for training or evaluation is associated with the numerical reading and minimum/maximum scale values. A first annotator assigned the reading and scale endpoints, after which a second person verified all annotations and made changes as necessary. To improve reproducibility without releasing sensitive imagery, the final adapters, exact split manifests, gauge-range metadata, ground-truth readings and evaluation scripts will be released.

The SE split is random rather than source or gauge-family-held-out. Therefore, visually related gauge types or acquisition environments may occur in both partitions. The LODO experiment that excludes all SE images from fine-tuning provides complementary evidence about transfer to the SE domain, but it cannot determine whether particular SE gauge types overlap between the random training and test partitions. A validated gauge-family taxonomy and a future gauge-family or plant-held-out split are required for that question. Representative SE images are shown in Figure 3.

## 3.4. Dataset Summary

Table 1 summarizes the final dataset split sizes, the number of distinct scale configurations, overall scale endpoints and test-set ground-truth value ranges.

![](images/97090b5cf66db82c86178802a408959470137b815b81d0409340a802cddbb7a3.jpg)

![](images/57c92c28a43b8b664cc6b2cb7ba4bf6f989c92d5acfd216847d30bb630b989ca.jpg)

![](images/fe1e4c3e5100164ce73cf238cf592e73f7adefca748d30fed32a7389226135cb.jpg)  
Figure 3: Examples from the proprietary SE dataset, illustrating variation in gauge appearance, background, scale and viewpoint.

Table 1: Summary of the final dataset splits, with scale-range and value statistics. “# Ranges” is the number of distinct (min, max) scale configurations present in the split; “Value Range” is the span of ground-truth needle readings.
<table><tr><td>Dataset</td><td>Train</td><td>Test</td><td># Ranges (train/test)</td><td>Scale Endpoints (min-max)</td><td>Value Range (test)</td></tr><tr><td>Synthetic</td><td>375</td><td>126</td><td>7/7</td><td>-2 to 10</td><td>-1.9 to 9.0</td></tr><tr><td>Pressure</td><td>57</td><td>36</td><td>1/3</td><td>-1 to 6</td><td>-0.42 to 3.2</td></tr><tr><td>SE (Ours)</td><td>152</td><td>43</td><td>32 / 18</td><td>-30 to 5000</td><td>0 to 2300</td></tr></table>

## 4. Methodology

## 4.1. Qwen2.5-VL-7B-Instruct

The core model is Qwen/Qwen2.5-VL-7B-Instruct [18]. A parameter audit of the final checkpoint after 20 training epochs reports 8,315,961,344 parameters in total: 7,615,616,512 in the language-model component, 631,975,680 in the vision encoder, 44,574,464 in the visual projector/merger and 23,794,688 LoRA parameters. The LoRA parameters therefore represent 0.29% of the loaded parameter count. The checkpoint is loaded for inference with parameters frozen; the LoRA count is identified from the adapter parameter names rather than from requires\_grad at inference time. Figure 4 illustrates the vision-encoder, projector/merger and language-model structure used in the VLM formulation.

![](images/92c33bccf70bdbe38c2c3e27d99e6766562da2d99caaa391ae15eb7212caf6ec.jpg)  
Figure 4: General VLM structure: a vision encoder extracts visual tokens, a projector/merger maps them to the language-model representation and the autoregressive language model produces the numerical answer.

## 4.2. Zero-shot and in-context learning

For zero-shot evaluation, the model receives the gauge image and the task instruction without demonstrations. ICL prepends nested sets of training examples. For each dataset, the 50-example pool was selected by random sampling from the corresponding training partition and the smaller example sets were nested subsets of that sampled pool. Thus, example sets of size $k \in \{ 1 , 5 , 1 0 , 2 0 , 5 0 \}$ satisfy $E _ { 1 } \subset E _ { 5 } \subset E _ { 1 0 } \subset E _ { 2 0 } \subset E _ { 5 0 }$ , allowing sensitivity to context size to be inspected without changing the smaller sets when additional examples are added.

## 4.3. QLoRA specialization and fixed-epoch training protocol

QLoRA [21] combines low-rank adaptation [22] with quantized frozen base weights. During training, the Qwen base model is loaded with 4-bit NF4 quantization, double quantization and bfloat16 compute. LoRA uses rank r = 8, α = 16 and dropout 0.05 on q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj and down\_proj modules.

Five Qwen2.5-VL-7B models are trained: an all-domain model trained and evaluated with range, an all-domain model trained and evaluated without range and three LODO models in which Synthetic, Pressure Gauge or SE is excluded from training and then used only for testing. The LODO models use the with-range prompt so that the held-out-domain comparison changes training-domain coverage rather than prompt formulation.

The reported protocol uses no separate validation set and no early stopping. All five revised fine-tuning runs were restarted from the base Qwen2.5-VL-7B-Instruct checkpoint, trained for 20 fixed epochs and the final checkpoint after epoch 20 is used for every reported analysis. No checkpoint is selected using test performance.

Training uses a batch size of 1 with gradient accumulation over 8 steps, AdamW with a learning rate of $2 \times 1 0 ^ { - 4 }$ , bfloat16 mixed precision and two Nvidia RTX A5500 GPUs. Training images are augmented with random rotation (±15<sup>◦</sup>, probability 0.7), color jitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.1; probability 0.8), Gaussian blur (kernel 3, σ = 0.1–1.5; probability 0.3) and random perspective transformation (distortion scale 0.2, probability 0.5, bilinear interpolation). Test images are not augmented. The Qwen processor uses dynamic image resolution with total-pixel bounds

256\*28\*28 to 640\*28\*28. A separate audit was not performed to verify label validity for random-crop outcomes; accordingly, crop-related label preservation is not claimed as an independently validated property of the augmentation pipeline.

## 4.4. Prompt templates

The reported experiments use the templates in Table 2. Preliminary prompt variants were not systematically archived and are therefore not presented as experimental results.

## 4.5. Metrics and statistical analysis

For sample i, range-normalized percentage error is

$$
e _ { i } = \frac { | y _ { i } - \hat { y } _ { i } | } { y _ { \operatorname* { m a x } , i } - y _ { \operatorname* { m i n } , i } } \times 1 0 0 \%
$$

The primary MPE is the mean of $e _ { i }$ . Because capping errors at 100% can hide catastrophic failures, this study reports both capped and uncapped MPE. Capped MPE uses min $( e _ { i } , 1 0 0 )$ ; uncapped MPE retains the complete tail. Additional metrics include raw engineering-value MAE and RMSE, maximum and 95th-percentile absolute/full-scale errors, failure rate above 5% of full scale, pass rates within ±1%, ±2% and ±5% of full scale and parsing-failure rate. The MAE/RMSE values are computed directly on the numerical targets and are reported as raw engineering-value errors rather than as physical-unit-specific metrology metrics.

MPE uncertainty is quantified with 10,000-sample nonparametric bootstrap 95% confidence intervals. Paired model comparisons align predictions by filename and use both a paired bootstrap CI for the mean per-image error diference and a Wilcoxon signed-rank test.

Table 2: Prompt templates used in the reported experiments. The same range condition is used during training and inference for the fine-tuned models.
<table><tr><td>Task</td><td>Role / Condition Template</td><td></td></tr><tr><td>Value Reading</td><td>System</td><td>You are an assistant that reads analog gauges. Respond with *only* the numerical value the needle is</td></tr><tr><td rowspan="2"></td><td></td><td>pointing to. For example, if the needle points to 5.5, respond ‘5.5&#x27;.</td></tr><tr><td></td><td>User (With range) What is the current reading on this analog gauge? The scale ranges from</td></tr><tr><td></td><td>User (No range)</td><td>{min_val} to {max_val}. What is the current reading on this analog gauge?</td></tr><tr><td>Scale Probe</td><td>System</td><td>You are a vision assistant that reads analog dial gauges. Look only at the printed scale numbers on the</td></tr><tr><td></td><td></td><td>dial face. Report the smallest and largest numbers printed on the scale. Respond ONLY as JSON: {&quot;min&quot;:</td></tr><tr><td></td><td>User</td><td>&lt;number&gt;, &quot;max&quot;: &lt;number&gt;}. What are the minimum and maximum scale values printed on this gauge&#x27;s dial?</td></tr></table>

Model output parsing first attempts strict floating-point conversion and then a numeric-expression extraction. Any remaining nonnumeric output is counted as a parsing failure rather than silently assigned a value.

## 4.6. Confidence and calibration

For a generated numeric answer containing m non-EOS output tokens, sequence confidence is defined as the geometric mean of the token probabilities assigned to the greedily generated tokens:

$$
C = \exp \left( \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \log p _ { k } \right) .
$$

This is a token-generation score, not a calibrated probability of numeric correctness. Calibration is therefore evaluated explicitly using 10-bin expected calibration error (ECE), maximum calibration error (MCE), reliability diagrams and risk-coverage curves with area under the risk-coverage curve (AURC). A prediction is treated as operationally correct for calibration when its absolute error is within 5% of full scale. This 5% threshold is an evaluation tolerance for the reliability analysis, not a claim of compliance with a metrology or instrument-accuracy standard. High-confidence (HC) errors are defined as incorrect predictions (error exceeding 5% of full scale) with sequence confidence $C \geq 0 . 9 9$

## 5. Results

## 5.1. Zero-shot and ICL baselines

Figure 5 shows zero-shot and ICL benchmark performance. ICL is evaluated with 1, 5, 10, 20 and 50 nested examples.

Pressure  
![](images/8b57ba4dcbc4d4b270320f64b60c7435afe8ff779070c6bb6afbb75823d31757.jpg)

![](images/f433acf42720baecc42b89e75a3394f3d757acd31f1ec176764acaa4669fefba.jpg)

SE  
![](images/c6954e324094e524672793175a1b6376a0866f7b93c7fbcf63e7321b6d791201.jpg)  
Figure 5: Zero-shot (0 examples) and ICL sensitivity to the number of contextual examples. Lower MPE is better.

The baseline trends are unstable. The Synthetic dataset reaches its best ICL result at 20 examples without range (15.83% MPE), Pressure Gauge at 10 examples without range (5.89%) and SE at 10 examples with range (27.07%). Increasing context beyond the best point does not monotonically improve accuracy, motivating weight adaptation rather than relying on context size

alone.

## 5.2. Fine-tuned QLoRA performance and clean range ablation

Table 3 reports the fine-tuned results obtained from the final checkpoint after 20 fixed training epochs. The best MPE is 2.39% for Synthetic with range, 2.61% for Pressure Gauge without range and 4.43% for SE with range. Figure 6 provides a visual comparison of the best zero-shot, ICL and fine-tuned results.

Table 3: MPE (%) for zero-shot, best-case ICL and fine-tuned Qwen2.5-VL-7B models. Fine-tuned entries include the 95% bootstrap CI in brackets and come from separately trained with-range and without-range models; all fine-tuned results use the final checkpoint after 20 fixed epochs.
<table><tr><td>Dataset</td><td>Method</td><td>With Range</td><td>Without Range</td></tr><tr><td rowspan="3"></td><td>Zero-shot</td><td>21.33</td><td>18.12</td></tr><tr><td>Synthetic ICL (best)</td><td>18.43 (k = 20)</td><td>15.83 (k = 20)</td></tr><tr><td>Fine-tuned</td><td>2.39 [1.43, 3.90]</td><td>2.73 [1.57, 4.58]</td></tr><tr><td rowspan="4">Pressure</td><td>Zero-shot</td><td>25.29</td><td>15.69</td></tr><tr><td>ICL (best)</td><td>8.34 (k = 5)</td><td>5.89 (k = 10)</td></tr><tr><td>Fine-tuned</td><td>3.31</td><td>2.61</td></tr><tr><td></td><td>[2.27, 4.57]</td><td>[1.66, 3.80]</td></tr><tr><td rowspan="3">SE</td><td>Zero-shot</td><td>39.12</td><td>31.06</td></tr><tr><td></td><td>ICL (best) 27.07 (k = 10)</td><td>29.87 (k = 20)</td></tr><tr><td>Fine-tuned</td><td>4.43 [2.31, 7.14]</td><td>6.28 [2.47, 11.84]</td></tr></table>

![](images/cc65915a617f3e828e903ebfff4b21fb2d1f55c3e3af817463997b9fe16b549d.jpg)  
Figure 6: Best observed MPE for each learning strategy. For Pressure, the best QLoRA result is the separately trained no-range model; for Synthetic and SE, the best QLoRA result uses range metadata.

The paired range ablation in Table 4 shows that the average capped-MPE diferences between the separately trained range/no-range models are not statistically significant at $p < 0 . 0 5$ for any dataset. However, capped MPE alone obscures tail behavior. On SE, the no-range model has an uncapped MPE of 13.42% compared with 4.43% with range and its maximum full-scale error reaches 407.14% compared with 41.25%. Thus, range metadata is best interpreted as reducing extreme SE failures rather than as providing a statistically established mean-error improvement in this small test set.

## 5.3. Extended error and reliability metrics

Table 5 complements the primary capped-MPE results with uncapped MPE, raw engineering-value MAE/RMSE, full-scale tail errors and operational tolerance pass rates. Across the nine all-domain and LODO evaluations, all 615 generated answers were numerically parseable (0 parsing failures).

The no-range SE and Synthetic results illustrate why uncapped reporting matters: their capped MPEs are 6.28% and 2.73% (Table 3), but uncapped MPE rises to 13.42% and 5.94%, respectively, because of extreme outliers.

Table 4: Paired range-ablation statistics for the separately trained fine-tuned Qwen2.5- VL-7B models. The mean diference is with-range minus without-range capped per-image percentage error; the corresponding MPE values are reported in Table 3.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Mean difference (percentage points)</td><td rowspan="2">95% CI</td><td rowspan="2">Wilcoxon p</td></tr><tr><td></td></tr><tr><td>SE</td><td>-1.85</td><td>[−7.17, 1.72]</td><td>0.974</td></tr><tr><td>Synthetic</td><td>-0.34</td><td>[-0.80, 0.03]</td><td>0.356</td></tr><tr><td>Pressure</td><td>0.70</td><td>[−0.22, 1.74]</td><td>0.106</td></tr></table>

Table 5: Extended error, tail-error and operational tolerance metrics for fine-tuned Qwen2.5- VL-7B models. MAE and RMSE use dataset-specific engineering units; P95 FS and Max FS indicate 95th-percentile and maximum errors relative to full scale (FS). Pass rates denote the percentage of predictions within operational tolerances (±1%, ±2%, ±5% FS). No parsing failures occurred across evaluated samples (0/205 per model condition).
<table><tr><td rowspan="2">Dataset Range</td><td rowspan="2"></td><td rowspan="2">Uncapped MPE (%)</td><td rowspan="2">MAE RMSE</td><td rowspan="2"></td><td rowspan="2">P95 FS Max FS Pass Pass Pass</td><td rowspan="2">(%)</td><td rowspan="2">±1%</td><td rowspan="2"> $\pm 2 \% \pm 5 \%$ </td></tr><tr><td>(%)</td></tr><tr><td>SE</td><td>with</td><td>4.43</td><td>48.39</td><td>176.47</td><td>19.90</td><td>41.25</td><td>44.2% 60.5% 79.1%</td><td></td></tr><tr><td>Synthetic with</td><td></td><td>2.39</td><td>0.17</td><td>0.61</td><td>5.00</td><td>80.00</td><td>37.3% 64.3% 96.0%</td><td></td></tr><tr><td>Pressure with</td><td></td><td>3.31</td><td>0.13</td><td>0.17</td><td>9.38</td><td>16.40</td><td>19.4% 36.1% 83.3%</td><td></td></tr><tr><td>SE</td><td>without</td><td>13.42</td><td>62.25</td><td>278.93</td><td>26.44</td><td>407.14</td><td>46.5% 55.8% 81.4%</td><td></td></tr><tr><td>Synthetic without</td><td></td><td>5.94</td><td>0.45</td><td>3.61</td><td>9.06</td><td>505.00</td><td>37.3% 64.3% 92.9%</td><td></td></tr><tr><td>Pressure without</td><td></td><td>2.61</td><td>0.10</td><td>0.16</td><td>8.53</td><td>16.50</td><td>38.9% 61.1% 91.7%</td><td></td></tr></table>

Dataset-level tail statistics therefore reveal failures that capped MPE alone can conceal.

## 5.4. Leave-one-dataset-out transfer

The LODO experiment directly tests whether performance survives removal of the target domain from fine-tuning. Table 6 compares each heldout-domain model with the all-domain with-range model on the same test images.

Table 6: Leave-one-dataset-out (LODO) cross-domain transfer evaluation for fine-tuned Qwen2.5-VL-7B. LODO MPE is the range-normalized mean percentage error on the heldout test partition with its 95% nonparametric bootstrap confidence interval [CI]. “Increase (pp)” represents the performance degradation in percentage points (LODO MPE minus the all-domain with-range baseline from Table 3), where positive values indicate increased error from domain exclusion. Wilcoxon p is the two-tailed significance value from a paired Wilcoxon signed-rank test on per-image errors.
<table><tr><td>Held-out dataset</td><td>LODO MPE [95% CI]</td><td>Increase (pp)</td><td>Increase 95% CI</td><td>Wilcoxon</td></tr><tr><td>SE</td><td>13.25 [8.38, 19.00]</td><td>8.82</td><td>[4.07, 14.20]</td><td>p &lt; 0.001</td></tr><tr><td>Synthetic</td><td>14.99 [11.14, 19.25]</td><td>12.60</td><td>[8.37, 17.21]</td><td>&lt; 0.001</td></tr><tr><td>Pressure</td><td>4.99 [3.09, 7.26]</td><td>1.68</td><td>[-0.06, 3.93]</td><td>0.360</td></tr></table>

Excluding SE or Synthetic from training causes a large and statistically significant increase in error, while the Pressure Gauge degradation is not significant in the present 36-image test set. For Pressure Gauge, the reported 1.68-percentage-point increase is relative to the all-domain with-range baseline of 3.31%, not the 2.61% no-range result that is the best all-domain result for that dataset. These results constrain the generalization claim: the specialized model transfers imperfectly across domains and benefits materially from exposure to visually related training data.

## 5.5. Adverse-condition robustness

Robustness was evaluated on all 43 SE test images by applying deterministic synthetic corruptions at three severity levels. Blur uses Gaussian radii 1.5, 3.0 and 5.0 pixels; low light multiplies brightness by 0.6, 0.4 and 0.25; reflection overlays a blurred bright ellipse with increasing opacity; perspective perturbation increases corner displacement; and occlusion adds one, two or three small randomly positioned black patches without targeting any semantic region. The clean with-range SE baseline is 4.43% MPE. Table 7 reports the corruption results numerically, while Figure 7 shows the MPE trend across severity levels.

Table 7: SE corruption robustness for the fine-tuned with-range Qwen2.5-VL-7B model. Values are capped MPE (%). The final column is the severity-3 failure rate above 5% full scale.
<table><tr><td>Corruption</td><td>Severity 1</td><td>Severity 2</td><td>Severity 3</td><td>Fail &gt; 5% at S3</td></tr><tr><td>Gaussian blur</td><td>9.24</td><td>15.51</td><td>19.92</td><td>65.12%</td></tr><tr><td>Low light</td><td>4.05</td><td>4.27</td><td>4.00</td><td>20.93%</td></tr><tr><td>Occlusion</td><td>4.68</td><td>5.69</td><td>5.47</td><td>23.26%</td></tr><tr><td>Perspective</td><td>5.07</td><td>3.11</td><td>4.14</td><td>16.28%</td></tr><tr><td>Reflection</td><td>4.26</td><td>4.47</td><td>4.86</td><td>23.26%</td></tr></table>

The model is notably sensitive to blur, whereas the tested low-light, reflection, perspective and small-occlusion perturbations remain closer to the clean baseline. The strong blur degradation persists despite the moderate Gaussian-blur augmentation used during training, indicating that generic blur augmentation alone does not provide adequate robustness at the tested severe blur levels. These tests increase controlled coverage of adverse conditions but should not be interpreted as a substitute for a larger real-world environmental benchmark.

![](images/450d963bf764b72e26cce9c1bd07d146e440598f90246c4e2dafeb1c2ec358d1.jpg)  
Figure 7: MPE under synthetic adverse-condition corruptions on the SE test set. Blur is the dominant tested degradation.

## 5.6. Visual endpoint probe

Using the scale probe prompt defined in Table 2, the visual endpoint experiment evaluates whether the fine-tuned model can extract minimum and maximum scale numbers directly from the dial image without supplying them in the prompt. Across 43 SE test images, the minimum endpoint is parsed on 100% of images with 95.3% exact-match accuracy and MAE 2.09, while the maximum endpoint has 100% parse coverage, 83.7% exact-match accuracy and MAE 44.21. This supports the narrow conclusion that the VLM often reads printed scale endpoints visually; it does not demonstrate reliable understanding of tick intervals, units or every scale label. Range metadata therefore remains useful in applications where a gauge registry or trusted configuration is available.

## 5.7. Calibration and high-confidence errors

Table 8 reports calibration metrics evaluated under the formulation detailed in Section 4.6. Across 205 pooled test predictions, the with-range model has ECE 0.053 and AURC 0.064, with two incorrect predictions at sequence confidence $\geq 0 . 9 9$ . The no-range model has ECE 0.047 and AURC 0.055, with one such high-confidence error. Thus, confidence is informative enough to rank some predictions but is not a reliable stand-alone safety signal. Figure 8 shows the pooled reliability diagram and risk-coverage curve for the with-range model.

![](images/347dbfbe10d68a369c2ed428b13634ede96393eb974813f573fdc2c87cc68f30.jpg)  
(a) Pooled reliability diagram.

![](images/21fbc62a353813b77c3a55486522d36b5e6488a56ee80a565cd8a93100efdb5f.jpg)  
(b) Pooled risk-coverage curve.  
Figure 8: Calibration analysis for the final with-range model. Corresponding no-range plots will be included in the public result package described in Section ??.

Table 8: Confidence calibration and reliability metrics for the fine-tuned Qwen2.5-VL-7B models evaluated on individual datasets and the pooled test set (N = 205). Accuracy@5% denotes the proportion of predictions meeting the ±5% full-scale operational correctness threshold. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) quantify average and worst-case calibration gaps across 10 equal-width confidence bins. AURC represents the Area Under the Risk-Coverage Curve measuring error rejection eficacy under confidence filtering. Mean Confidence denotes the average token-sequence confidence (C) and High-Confidence (HC) errors count critical failures where absolute error exceeds 5% of full scale despite sequence confidence $C \geq 0 . 9 9$
<table><tr><td>Model</td><td>Dataset</td><td>Accuracy @5% (%)</td><td></td><td>ECE MCE AURC</td><td></td><td>Mean confidence errors</td><td>HC</td></tr><tr><td>With range SE</td><td></td><td>79.1</td><td>0.158</td><td>0.311</td><td>0.177</td><td>0.920</td><td>1</td></tr><tr><td>With range Synthetic</td><td></td><td>96.0</td><td>0.036 0.357</td><td></td><td>0.021</td><td>0.925</td><td>1</td></tr><tr><td>With range Pressure</td><td></td><td>83.3</td><td>0.138</td><td>0.198</td><td>0.175</td><td>0.803</td><td>0</td></tr><tr><td>With range Pooled</td><td></td><td>90.2</td><td>0.053</td><td>0.219</td><td>0.064</td><td>0.902</td><td>2</td></tr><tr><td>No range</td><td>SE</td><td>81.4</td><td>0.131</td><td>0.353</td><td>0.066</td><td>0.945</td><td>0</td></tr><tr><td>No range</td><td>Synthetic</td><td>92.1</td><td>0.045 0.068</td><td></td><td>0.049</td><td>0.926</td><td>1</td></tr><tr><td>No range</td><td>Pressure</td><td>91.7</td><td>0.079</td><td>0.328</td><td>0.035</td><td>0.851</td><td>0</td></tr><tr><td>No range</td><td>Pooled</td><td>89.8</td><td>0.047</td><td>0.328</td><td>0.055</td><td>0.917</td><td>1</td></tr></table>

## 5.8. Pressure Gauge video-level analysis and cross-study comparison

The Pressure Gauge test set contains 36 images drawn from seven videos that are distinct from the training videos. As previously noted, our evaluation partition was deliberately curated to include only frames exhibiting distinct gauge readings, yielding a more diverse and arguably more demanding benchmark than the uniform, fixed-interval frame-sampling protocol used in the reference study [5]. Because their purpose-built multi-stage mobile gaugereading pipeline is not publicly available, their method cannot be re-executed directly on our specific 36-frame test split. Table 9 therefore combines the requested per-video breakdown with a transparent cross-study comparison using the relative-error metric $\mu _ { R }$ defined in the reference study [5]. The fine-tuned Qwen2.5-VL-7B metrics reflect the no-range condition, where the reported confidence intervals correspond to per-video bootstrap MPE intervals scaled to the relative-error domain (divided by 100).

Table 9: Per-video Pressure Gauge comparison with the reference method [5]. Values are mean relative error $\mu _ { R }$ (lower is better). The Qwen column includes a 95% bootstrap CI. Sampling protocols are not identical.
<table><tr><td>Video</td><td>n</td><td>Reference  $\mu _ { R }$ </td><td>Fine-tuned Qwen2.5-VL-7B [95% CI]  $\mu _ { R }$ </td></tr><tr><td>test1/man1</td><td>13</td><td>0.003</td><td>0.0096 [0.0058, 0.0138]</td></tr><tr><td>test2/man2</td><td>12</td><td>0.010</td><td>0.0319 [0.0137, 0.0596]</td></tr><tr><td>test3/man3</td><td>5</td><td>0.036</td><td>0.0320 [0.0240, 0.0408]</td></tr><tr><td>test4/man4</td><td>1</td><td>0.008</td><td>0.0080 [0.0080, 0.0080]</td></tr><tr><td>test5/man5</td><td>2</td><td>No prediction</td><td>0.0520 [0.0320, 0.0720]</td></tr><tr><td>test6/man6</td><td>1</td><td>0.045</td><td>0.1250 [0.1250, 0.1250]</td></tr><tr><td>test7/man7</td><td>2</td><td>0.042</td><td>0.0167 [0.0167, 0.0167]</td></tr><tr><td>Overall</td><td>36</td><td>0.024ª</td><td>0.0261 [0.0166, 0.0380]</td></tr></table>

<sup>a</sup>The reference-study mean excludes test5, for which no prediction was produced.

The comparison is mixed rather than uniformly favorable. The reference method uses a task-specific multi-stage pipeline for the Pressure Gauge video setting, including localization, geometric processing and reading stages [5]. Against that dataset-specific methodology, the general VLM-based reader has lower mean relative error on test3 and test7, matches test4, has higher error on test1, test2 and test6 and returns predictions for test5 where the multi-stage method reports no result. Its overall relative error is therefore similar despite using a direct cross-dataset reading formulation rather than a pipeline specifically designed for this dataset. Because the sampled frames are not identical, this is evidence of competitive performance and broader coverage, not a controlled claim of superiority.

## 5.9. Other model baselines and computational cost

The study also evaluates Qwen2.5-VL-3B, compact SmolVLM2 models and a MobileNetV2 regressor on SE. Table 10 reports these additional baseline accuracies; the primary Qwen2.5-VL-7B SE results are reported once, in Table 3. Because per-image outputs for the additional baselines are not available under the same fixed-epoch Qwen2.5-VL-7B protocol, the table is used for architectural context rather than paired significance claims. The MobileNetV2 with-range and without-range experiments also use diferent target formulations (normalized position versus raw-value regression), so their diference is not interpreted as a clean metadata ablation.

A protocol-matched eficiency benchmark was then run for all five architectures on the same Nvidia RTX A5500 using the same source image, batch size of one, five warm-up iterations, 30 timed trials, CUDA synchronization and peak-memory procedure, with preprocessing excluded from the timed region. VLM latency measures greedy generation with a maximum of 20 new tokens; MobileNetV2 latency measures a single forward pass. Table 11 reports mean and 95th-percentile latency and peak allocated GPU memory during inference. Training duration is reported on a per-epoch basis from these later protocol-matched benchmark runs rather than extrapolated from the main accuracy-training logs or to a total multi-epoch training time. These timing measurements were collected after the main accuracy experiments and were not used for model or checkpoint selection. Peak GPU memory during training was not recorded in this benchmark; the memory values in Table 11 are inference-only peak allocations.

Table 10: Additional model baselines on the SE dataset. MPE is reported in percent. The primary Qwen2.5-VL-7B values (4.43% with range; 6.28% without range) are reported in Table 3 and are not duplicated here.
<table><tr><td rowspan="2">Model</td><td colspan="2">MPE (%)</td></tr><tr><td>With</td><td>Without</td></tr><tr><td></td><td>range</td><td>range</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>4.98</td><td>7.33</td></tr><tr><td>SmolVLM2-500M-Instruct</td><td>24.14</td><td>21.12</td></tr><tr><td>SmolVLM2-256M-Instruct</td><td>26.22</td><td>17.78</td></tr><tr><td>MobileNetV2 Regressor</td><td>11.12</td><td>27.82</td></tr></table>

The VLMs have broadly similar sub-second generation latency under this protocol (approximately 537–600 ms), whereas MobileNetV2 is much faster at 3.95 ms. Peak inference memory decreases more consistently with model size, from 8101.53 MB for Qwen2.5-VL-7B to 1132.43 MB for SmolVLM2- 256M; MobileNetV2 uses 143.73 MB. The near-equality of Qwen2.5-VL-7B and 3B latency illustrates that short autoregressive generation latency is not determined by parameter count alone. Likewise, the measured per-epoch training durations depend on implementation and processor overhead and do not decrease monotonically with model size. These timings exclude image acquisition, multi-gauge detection or instance selection, preprocessing, output parsing, network/database writes and alarm handling.

Table 11: Protocol-matched computational benchmark on an Nvidia RTX A5500. Peak memory is peak allocated GPU memory during inference; training time is measured per epoch under the benchmark configuration.
<table><tr><td>Model</td><td>Mean latency P95 latency (ms)</td><td>(ms)</td><td>Peak GPU memory (MB) per epoch (min)</td><td>Training time</td></tr><tr><td>Qwen2.5-VL-7B</td><td>594.98</td><td>597.30</td><td>8101.53</td><td>8.05</td></tr><tr><td>Qwen2.5-VL-3B</td><td>599.74</td><td>604.12</td><td>6668.66</td><td>7.65</td></tr><tr><td>SmolVLM2-500M</td><td>582.55</td><td>587.18</td><td>2882.85</td><td>16.81</td></tr><tr><td>SmolVLM2-256M</td><td>536.53</td><td>539.52</td><td>1132.43</td><td>16.17</td></tr><tr><td>MobileNetV2</td><td>3.95</td><td>4.07</td><td>143.73</td><td>0.65</td></tr></table>

## 5.10. Interpretability: blurred-patch sensitivity and attention rollout

To examine which image regions influence the fine-tuned model’s numerical predictions, we compare two complementary spatial diagnostics: blurredpatch perturbation and attention rollout. The blurred-patch analysis divides each image into a 10 × 10 grid and locally blurs one cell at a time. The resulting sensitivity map reflects the change in prediction error relative to the unperturbed image, thereby identifying regions for which loss of local visual detail most afects the generated reading. Unlike hard occlusion with a black patch, local blurring preserves the approximate image structure and reduces the possibility that the explanation is dominated by an artificial occlusion boundary.

Attention rollout provides a second, non-perturbative diagnostic. The rollout is computed from the final four transformer layers, beginning from positions associated with the generated non-EOS answer tokens and propagating mean-head self-attention, with residual normalization, back to the visual-token representation. To make eager-attention extraction computationally tractable, this diagnostic uses an aspect-ratio-preserving 448-pixel short side and a processor cap of 262,144 pixels. The resulting attention map is projected back to the source-image aspect ratio for visualization. Because this difers from the standard evaluation resolution, the attention maps are used only as qualitative spatial diagnostics; all numerical reading results are obtained from the standard evaluation pipeline.

Figure 9 presents three representative cases. Under the standard withrange evaluation, the perspective case has an actual reading of 12.5 and is predicted as 15.0 (0.76% MPE), while the dense-scale case has an actual reading of 75 and is predicted as 70.0 (0.83% MPE). In both successful examples, the blurred-patch maps concentrate their strongest sensitivity within the gauge face rather than the surrounding background. For the perspective case, the most influential region lies in the lower-left portion of the dial, close to the portion of the pointer and scale involved in determining the reading. The attention-rollout map is less spatially concentrated, but also assigns substantial attention to regions of the dial face. For the dense-scale case, blurred-patch sensitivity is again concentrated on the left side of the scale near the visually relevant dial markings, whereas the attention rollout is distributed across several locations on the gauge face, including both left- and right-side scale regions. Thus, the two explanation methods provide partially convergent rather than identical spatial evidence.

Quantitatively, the blurred-patch and attention-rollout maps have Pearson correlations of 0.775 for the perspective example and 0.659 for the dense-scale example and their dominant grid cell agrees in both cases. These values indicate moderate spatial agreement, while the visibly broader attention distributions caution against interpreting attention as a precise localization of the needle or a specific tick mark. In particular, no formal claim is made that either method identifies the exact pointer tip or corresponding scale label because spatial needle and tick-level annotations are not available for these examples.

The anomalous example illustrates the limitation of such post-hoc explanations. The image is a comparatively clear, approximately head-on industrial scene for which the actual reading is 10, yet the model predicts 41.0, corresponding to 31.0% MPE, with sequence confidence 0.985. Its blurred-patch sensitivity map is efectively flat across the image: locally removing fine detail from any single grid cell produces no distinguishable sensitivity peak. Consequently, no meaningful perturbation-based localization can be inferred for this failure. In contrast, the attention-rollout map still places visible attention on the gauge face, particularly around the upper portion of the dial. This discrepancy is important: attention to the gauge does not establish that the model extracted the correct pointer geometry or numerical scale relationship. Because the perturbation-map variance is zero, a correlation between the two explanation maps is undefined for this example.

Taken together, the examples support a deliberately limited interpretation. For two accurate predictions, blurred-patch sensitivity and attention rollout show moderate spatial agreement and both emphasize regions of the gauge face, although attention is comparatively difuse. For the anomalous prediction, attention remains concentrated on the gauge while perturbation sensitivity provides no meaningful localization. The analysis therefore does not support a causal claim that the erroneous numerical prediction arises specifically from the vision encoder, scale interpretation, or autoregressive decoding stage. Rather, it shows that apparently relevant attention can coexist with a large numerical error, reinforcing the need to treat post-hoc attention visualizations as diagnostic evidence rather than as proof of correct visual reasoning.

## 6. Discussion: Industrial Risk Controls

The experiments expose several practical failure mechanisms that should shape deployment design. First, blur causes the largest tested corruption degradation, motivating an image-quality gate that rejects insuficiently focused inputs before numerical interpretation. Second, the nonzero highconfidence error rate shows that autoregressive token confidence must not be used as a sole validity signal. Third, LODO degradation indicates that a gauge reader should be validated on the target site or gauge family rather than assumed to transfer automatically.

A practical safety architecture could combine: (i) confidence-based abstention calibrated on the target environment; (ii) deterministic plausibility checks against the configured scale range and known process bounds; (iii) temporal consistency checks for video streams; (iv) an image-quality or blur detector; (v) redundant verification by OCR/geometric reasoning or a second model when a reading would trigger an alarm; and (vi) human confirmation for safety-critical decisions. These mechanisms are proposed safeguards, not components validated by the present experiments and should be evaluated before operational use.

![](images/c44f044272fede5955ec7006c56b45c3b094f0ca7d8182a6a23698589a693a8a.jpg)  
(a) Perspective input

![](images/74f21001ead3ba6963deb07dcadf10e4c2efa0789cede57f2e919a57c168f545.jpg)  
(b) Dense-scale input

![](images/db7926bfc3bb2d6c8eb627a832aabd7345a4dd0f313627265d65c273daa0c8fa.jpg)  
(c) Anomalous input

![](images/8ec67be30a6a89c45955845a308f35fe9997233a78b2be604cafc4a9f60dc2a0.jpg)  
(d) Blurred-patch sensitivity

![](images/ceedd9424d718cfbbb354c98670dbcca3066c26d18c21e0c30ed430afc280001.jpg)  
(e) Blurred-patch sensitivity

![](images/878f42609639ce94c00a279ebeb61e97b68aba65c898cbf75063c9ed55e5867a.jpg)  
(f) Blurred-patch sensitivity

![](images/0274db9471aeb439177c1e228f16f425c6ad9d5d2f2dd5b4ea2c160093d677f2.jpg)  
(g) Attention rollout

![](images/a0ef8850720c3d1d099e48e6eda0d3e4d065a2eaed03a6342a7e5d829ceea1f5.jpg)  
(h) Attention rollout

![](images/5be813edbd4d7bcd64c2d30c762f1da5af1e0b961c1cd681351f46dac55bb139.jpg)  
(i) Attention rollout  
Figure 9: Interpretability analysis for three representative SE examples. Top: original inputs. Middle: blurred-patch sensitivity maps. Bottom: attention-rollout maps. The two successful examples emphasize similar gauge-face regions, though attention rollout is more spatially difuse. The anomalous example shows a nearly flat sensitivity map despite attention on the gauge face, indicating that attention alone does not guarantee a correct numerical reading.

## 7. Limitations

Several limitations bound the conclusions of this study.

• Dataset size. The test sets contain 126 Synthetic, 36 Pressure Gauge and 43 SE images. Bootstrap intervals quantify sampling uncertainty but do not replace a larger independent industrial benchmark.

• Split structure. Pressure Gauge is video-separated, but Synthetic and SE use random splits. The SE partition is not gauge-family- or plant-held-out, so related gauges or environments may occur in both training and test data. LODO measures transfer when the entire SE domain is excluded from training, but it does not identify gauge-type overlap within the random SE split.

• Single-target-gauge input assumption. Each input contains one target gauge, but images are not necessarily tight crops and may retain substantial scene context. Only multi-gauge source images were cropped to isolate individual instances. Automatic detection and selection among multiple gauges are not evaluated.

• Range metadata. The model can often read scale endpoints visually, but maximum-endpoint exact match is 83.7% and tick intervals/units were not directly tested. A trusted range registry remains useful.

• Cross-domain transfer. LODO performance degrades substantially on Synthetic and SE, showing that the present model is not domain invariant.

• Reliability. High-confidence errors remain possible and blur sensitivity is substantial. Calibration and abstention must be validated for each deployment environment.

## 8. Conclusion

This study demonstrates that QLoRA specialization can substantially improve direct numerical reading by a general-purpose VLM for images containing a single target analog gauge, including images in which the gauge occupies only a small portion of the frame. Under a fixed 20-epoch protocol using the final checkpoint, the best MPEs are 2.39% on Synthetic, 2.61% on Pressure Gauge and 4.43% on SE. Separate range/no-range models remove prompt-setting confounds, while expanded metrics expose tail errors that capped MPE alone would conceal.

The additional experiments also make the limits of the approach clear. Leave-one-dataset-out evaluation shows large transfer degradation on heldout Synthetic and SE; Gaussian blur is a strong failure mode; and highconfidence errors persist despite generally low average error. The evidence therefore supports a promising specialized single-target-gauge reader, not a deployment-ready system for detecting, selecting and reading multiple gauges throughout a plant. Future work should prioritize larger source-heldout industrial datasets, gauge-family-level evaluation, reliable scale/OCR extraction, calibrated abstention and integration with independent validation and multi-gauge detection.

## 9. Acknowledgements

The authors gratefully acknowledge the valuable guidance and support provided by Siemens Energy which was instrumental in the completion of this research. This research was also supported by The Florida High Tech Corridor’s Matching Grants Research Program (MGRP) under grant number GR109914.

## References

[1] Y. Yang, W. Liao, S. Fan, J. Hou, H. Tang, Automatic reading method for analog dial gauges with diferent measurement ranges in outdoor substation scenarios, Information 16 (3) (2025) 226.

[2] V. Trairattanapa, S. Phimsiri, C. Utintu, R. Cherdchusakulcha, T. Tosawadi, E. Thamwiwatthana, S. Tungjitnob, P. Tangamonsiri, A. Takutruea, A. Keomeesuan, et al., Real-time multiple analog gauges reader for an autonomous robot application, in: 2022 17th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP), IEEE, 2022, pp. 1–6.

[3] X. Li, P. Yin, C. Duan, Y. Zhi, Analog gauge reader based on image recognition, in: Journal of Physics: Conference Series, Vol. 1650, IOP Publishing, 2020, p. 032061.

[4] B. Tian, M. Wu, R. Zhang, H. Zheng, B. Chen, Y. Wang, S. Trivedi, S. Zhang, R. B. Kaufman, L. Espenhahn, et al., Gaugetracker: Aipowered cost-efective analog gauge monitoring system, in: 2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR), IEEE, 2024, pp. 477–483.

[5] B. Howells, J. Charles, R. Cipolla, Real-time analogue gauge transcription on mobile phone, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2369–2377.

[6] H. Kappert, S. Schopferer, N. Saeidi, S. Ziesche, A. Olowinsky, F. Naumann, M. Jaegle, M. Spanier, Sensor systems for extremely harsh environments, in: Sensors and Measuring Systems; 21th ITG/GMA-Symposium, VDE, 2022, pp. 1–3.

[7] J. Peixoto, J. Sousa, R. Carvalho, G. Santos, J. Mendes, R. Cardoso, A. Reis, Development of an analog gauge reading solution based on computer vision and deep learning for an iot application, in: Telecom, Vol. 3, MDPI, 2022, pp. 564–580.

[8] M. Reitsma, J. Keller, K. Blomqvist, R. Siegwart, Under pressure: learning-based analog gauge reading in the wild, in: 2024 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2024, pp. 14–20.

[9] H. Ninama, J. Raikwal, A. Ravuri, D. Sukheja, S. K. Bhoi, N. Jhanjhi, A. A. H. Elnour, A. Abdelmaboud, Computer vision and deep transfer

learning for automatic gauge reading detection, Scientific Reports 14 (1) (2024) 23019.

[10] D. Ji, W. Zhang, W. Yang, Q. Zhao, Another way: Direct regression of meter readings for circular pointer meter images, Engineering Applications of Artificial Intelligence 136 (2024) 108863.

[11] M. K. Gellaboina, G. Swaminathan, V. Venkoparao, Analog dial gauge reader for handheld devices, in: 2013 IEEE 8th Conference on Industrial Electronics and Applications (ICIEA), IEEE, 2013, pp. 1147–1150.

[12] J. S. Lauridsen, J. A. Graasmé, M. Pedersen, D. G. Jensen, S. H. Andersen, T. B. Moeslund, Reading circular analogue gauges using digital image processing, in: 14th International joint conference on computer vision, imaging and computer graphics theory and applications (Visigrapp 2019), SCITEPRESS Digital Library, 2019, pp. 373–382.

[13] B. Li, J. Yang, X. Zeng, H. Yue, W. Xiang, Automatic gauge detection via geometric fitting for safety inspection, IEEE Access 7 (2019) 87042– 87048.

[14] L. Zuo, P. He, C. Zhang, Z. Zhang, A robust approach to reading recognition of pointer meters based on improved mask-rcnn, Neurocomputing 388 (2020) 90–101.

[15] Y. Liu, J. Liu, Y. Ke, A detection and recognition system of pointer meters in substations based on computer vision, Measurement 152 (2020) 107333.

[16] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510– 4520.

[17] F. Bordes, R. Y. Pang, A. Ajay, A. C. Li, A. Bardes, S. Petryk, O. Mañas, Z. Lin, A. Mahmoud, B. Jayaraman, et al., An introduction to visionlanguage modeling, arXiv preprint arXiv:2405.17247 (2024).

[18] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al., Qwen2. 5-vl technical report, arXiv preprint arXiv:2502.13923 (2025).

[19] J. Hanzelka, L. Tiday, A. Buinoschi-Tirpescu, D. Danciu, Synthetic data for precision gauge reading, https://www.kaggle.com/datasets/ endava/synthetic-data-for-precision-gauge-reading/data, kaggle. Accessed: May 2025.

[20] J. Grassme, Pressure gauge reader data, https://www.kaggle.com/ datasets/juliusgrassme/pressure-gauge-reader-data, kaggle, Accessed: May 2025 (2022).

[21] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, Qlora: Eficient finetuning of quantized llms, Advances in neural information processing systems 36 (2023) 10088–10115.

[22] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al., Lora: Low-rank adaptation of large language models., ICLR 1 (2) (2022) 3.