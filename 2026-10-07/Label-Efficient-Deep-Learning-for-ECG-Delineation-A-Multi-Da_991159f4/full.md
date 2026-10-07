# Label-Efficient Deep Learning for ECG Delineation: A Multi-Dataset Benchmark against Widely Used Delineation Tools

Jeonghwa Lim<sup>∗</sup>, Minje Park<sup>∗</sup>, Yeongyeon Na, Yujin Eom, Soyeon Lim, Young Ho Lee, Yu Jeong Kim, Sunghoon Joo<sup>†</sup>, and Ki Hong Lee<sup>‡</sup>

Abstract— Electrocardiogram (ECG) delineation, the identification of waveform boundaries, is a foundational step that translates raw ECG signals into clinically interpretable measurements. Deep learning has advanced this task but remains dependent on costly expert annotations. Label-efficient strategies such as self-supervised pretraining and semi-supervised learning are expected to ease this burden, yet it remains unclear whether they yield reliable delineation and whether the deep models they produce outperform the delineation tools used in practice. We address this in two stages. First, comparing selfsupervised objectives with supervised or semi-supervised fine-tuning across one internal and four external datasets, we find that pretraining helps but the objective matters, and that the value of semi-supervised fine-tuning depends on the pretraining objective. Second, we benchmark the selected deep learning model against widely used opensource (NeuroKit2, Prominence, ECGdeli) and commercial (CalECG) tools using three complementary metrics. The model ranks best on every metric and dataset, outperforming the strongest tool by a clear margin on the rhythmdiverse set (mIoU 71.3 vs. 54.8%; averaged point-wise sensitivity 92.6 vs. 76.4%), and degrades the least from sinus to arrhythmia. A rhythm-stratified and point-wise analysis further characterizes the distinctive behavior of each tool, yielding practical guidance for tool selection. These results provide systematic, multi-dataset evidence that selfsupervised pretraining is effective for ECG delineation and

Index Terms— Electrocardiography, Semantic segmentation, Electrocardiogram delineation, Label-efficient learning, Self-supervised learning, Semisupervised learning, Benchmarking

## I. INTRODUCTION

enables a label-efficiently trained deep learning model to outperform widely used delineation tools by leveraging abundant unlabeled data. This supports adopting such models in diverse, real-world clinical settings.

N Electrocardiogram (ECG) delineation, the segmentation of the ECG signal into its component P wave, QRS complex, and T wave, forms the basis for deriving clinically essential measurements such as the PR interval, QRS duration, and QT interval [1]. These intervals are fundamental to the diagnosis and monitoring of a wide range of cardiac conditions, including conduction abnormalities, arrhythmias, and other electrophysiologic disorders such as QT prolongation [2]. Accordingly, accurate and reliable ECG delineation is essential, as it directly affects the quality of downstream clinical decision-making [3]. Yet achieving robust, generalizable ECG delineation remains challenging, owing to substantial morphological variability and signal-quality degradation across patients and recording conditions [4], [5].

Automated delineation has followed two broad approaches: rule-based signal processing and deep learning. Rule-based methods are interpretable, lightweight, and widely deployed, but their accuracy tends to degrade on morphologically atypical beats and under residual noise, such as baseline wander and muscle artifacts, that distorts the waveform features on which they rely [4], [5]. Recently, deep learning has been increasingly applied to delineation, learning waveform morphology directly from data and achieving improved accuracy [6]–[8]. The key limitation of deep learning, however, is a reliance on large volumes of expert-annotated data: precise delineation labels are costly and time-consuming to obtain, which constrains the practical scalability of fully supervised training.

Label-efficient deep learning [9] can be a promising approach to mitigate the annotation bottleneck. Semi-supervised learning combines a limited labeled set with abundant unlabeled data, improving over supervised training on the labeled subset alone [10]. Self-supervised pretraining, which learns transferable representations from large-scale unlabeled data before task-specific fine-tuning, has driven progress across various domains [11], [12], including ECG tasks such as arrhythmia detection and disease screening [13]–[15]. The effectiveness of label-efficient learning for ECG delineation, however, remains unexamined on two fronts. First, ECG delineation is a dense prediction task at the level of individual time points, unlike the record-level classification tasks where pretraining has proven effective [16]. It is therefore unclear whether self-supervised pretraining helps at all, how different pretraining objectives compare, and how the choice of objective interacts with supervised versus semi-supervised finetuning. Second, setting aside how these models are trained, label-efficient learning approaches have, to our knowledge, never been directly benchmarked against the open-source and commercial tools currently used in research and clinical practice.

![](images/d7fe2ca5bbd0754bcdf848aaf69fdeea5bc79a04dbb0bb8999b36d7bb8c89831.jpg)  
Fig. 1. Study overview. Stage 1 pretrains an encoder on unlabeled ECG data with self-supervised learning (e.g., Masked Autoencoder) and fine-tunes it with a segmentation decoder (e.g., FCN) under supervised and semi-supervised learning (e.g., Mean Teacher). Stage 2 benchmarks the resulting model against existing delineation tools on multiple datasets using three complementary metrics: mIoU, measurement error on the PR, QRS, and QT intervals, and fiducial point localization accuracy.

We address these questions in two stages, illustrated in Fig. 1. In the first stage, we compare several pretraining objectives, each followed by supervised or semi-supervised fine-tuning, to assess the effect of self-supervised pretraining and the fine-tuning strategy on delineation performance. In the second stage, we benchmark the model selected in the first stage against established open-source and commercial tools.

Our contributions are summarized as follows:

• We provide systematic, multi-dataset evidence that selfsupervised pretraining benefits ECG delineation.

• We present the first benchmark of label-efficiently trained deep learning models against multiple open-source libraries as well as a commercial package, with rhythmstratified analysis across diverse external datasets.

• We characterize the distinctive behavior of each tool across datasets, and show that the deep learning model achieves the best overall performance while remaining robust under arrhythmia.

## II. RELATED WORKS

## A. Classical ECG Delineation Tools

Rule-based signal processing has been the standard approach to automated ECG delineation. Among these methods, the dominant family builds on the wavelet transform, which decomposes the signal into multiple scales so that the peaks and boundaries of the P wave, QRS complex, and T wave can be localized from characteristic slopes [17]. Many of these algorithms are distributed as open-source and are widely used in practice. NeuroKit2 [18] and ECGdeli [19] implement wavelet-based delineation, while Prominence [20] is a recent physiology-informed method that localizes fiducial points via peak prominence with linear runtime. In regulatory and pharmaceutical settings, commercial packages such as CalECG (AMPS LLC, New York, NY) instead produce an initial delineation that a trained operator reviews and corrects [21], [22].

## B. Deep Learning for ECG Delineation

Deep learning approaches formulate ECG delineation as a dense prediction task, assigning each time point to a wave class (P, QRS, T, or background) with encoder-decoder segmentation networks [6]–[8]. These models surpass rule-based methods in accuracy and extend to diverse arrhythmias [8], but they are predominantly supervised and depend on dense, timepoint-level annotations that are costly to produce, especially across the varied morphologies seen in arrhythmia. Semisupervised learning offers a natural solution, supplementing a small labeled set with abundant unlabeled data [10]. A recent multi-dataset study applies it to delineation and reports gains over supervised training on the labeled subset alone [23], confirming that unlabeled recordings are a useful resource.

TABLE I  
CHARACTERISTICS OF THE ECG DATASETS USED FOR DEVELOPING AND EVALUATING THE DELINEATION MODEL. LEAD CONFIGURATIONS RANGE FROM 12-LEAD (SIX LIMB AND SIX PRECORDIAL LEADS) TO 6-LEAD (LIMB ONLY) AND 2-LEAD (ARBITRARY PAIRS, E.G., MLII AND V1).
<table><tr><td>Role</td><td>Source</td><td>#Subjects</td><td>#ECGs</td><td>Duration (labeled)</td><td>Sample rate</td><td>Lead type</td><td>#Samples</td></tr><tr><td rowspan="3">Training</td><td>QTDB</td><td>105</td><td>105</td><td>5.9–253.6 s</td><td>250 Hz</td><td>2-lead</td><td>718</td></tr><tr><td>ISP</td><td>499</td><td>499</td><td>10 s</td><td>1000 Hz</td><td>12-lead</td><td>5,988</td></tr><tr><td>PTB-XL</td><td>18,885</td><td>21,837</td><td>— (unlabeled)</td><td>500 Hz</td><td>12-lead</td><td>262,044</td></tr><tr><td rowspan="4">Evaluation</td><td>LUDB</td><td>200</td><td>200</td><td>10 s</td><td>500 Hz</td><td>12-lead</td><td>2,369</td></tr><tr><td>Zhejiang</td><td>334</td><td>334</td><td>1.3–7.1 s</td><td>2000 Hz</td><td>12-lead</td><td>4,008</td></tr><tr><td>RDB</td><td>2,399</td><td>2,399</td><td>10 s</td><td>500 Hz</td><td>12-lead</td><td>28,788</td></tr><tr><td>mECGDB</td><td>205</td><td>205</td><td>2-10 s</td><td>250 Hz</td><td>6-lead</td><td>205</td></tr></table>

## C. Self-Supervised Learning

Self-supervised learning (SSL) reduces the reliance on manual annotations by pretraining on large unlabeled datasets through pretext tasks constructed from the data itself, yielding general-purpose representations. These representations can then be transferred to the downstream task using only a small labeled set [13], [24]. Several categories of SSL methods exist, including reconstruction-based and contrastive approaches [15], [25], [26]. Although SSL has been widely applied to ECG, existing studies have largely targeted classification tasks such as arrhythmia detection and disease screening [13]–[15]; to our knowledge, none has treated delineation as a downstream task. Consequently, whether SSL benefits delineation and how pretraining interacts with supervised versus semi-supervised fine-tuning all remain underexamined.

## D. Benchmarking Gaps in ECG Delineation

Jaeger et al. benchmarked rule-based methods across multiple datasets but excluded deep learning [27], whereas Chuma et al. included deep learning methods but evaluated on only a single external dataset [28]. Both assessed only waveformlevel accuracy at the fiducial point or segment level, without reporting the clinical interval measurements (PR interval, QRS duration, and QT interval) that delineation ultimately serves. A recent semi-supervised benchmark [23] did report such interval metrics, but evaluated deep learning models alone and, by merging the labeled datasets into a single training pool with only one held-out external set, left generalization to unseen datasets underexplored. To our knowledge, no prior work has jointly evaluated rule-based and deep learning methods across multiple external datasets while assessing these clinical intervals.

## III. BENCHMARK DESIGN

## A. Datasets and Preprocessing

We use seven public datasets, namely QTDB [29], [30], ISP [31], PTB-XL [32], MIMIC-IV-ECG [33], LUDB [34], Zhejiang [30], [35], and RDB [36], [37], together with a private dataset, the mobile ECG database (mECGDB). To ensure a fair and reproducible comparison, we follow the standardized benchmark configuration of a prior open benchmark study [23]. RDB and MIMIC-IV-ECG, which are not included in the benchmark, were processed following the same protocol. Table I summarizes the characteristics of these datasets, and further details are provided in the Supplementary Material (Section VIII).

1) Pretraining data: For self-supervised pretraining, we use MIMIC-IV-ECG, a dataset that provides large-scale raw ECG signals alongside clinical text reports and has been widely adopted for self-supervised ECG representation learning [38]. The original dataset contains 800,035 diagnostic 12-lead ECGs from 161,352 subjects. After filtering out corrupted recordings (e.g., those with saturated or NaN values) and those whose report is empty or contains fewer than three words [15], we retain 761,139 recordings from 154,094 subjects. Each recording is 10 seconds long and is sampled at 500 Hz.

2) Training data: Three datasets are used to develop the delineation model: QTDB and ISP as the labeled sets and PTB-XL as the unlabeled set. We choose QTDB and ISP for their complementary characteristics: QTDB is an established delineation benchmark, while ISP offers larger scale, more realistic noise, and greater rhythm diversity. They are split independently into training, validation, and test sets and merged into a single labeled pool (QTDB+ISP). Its training split provides the labeled data for supervised and semi-supervised training. PTB-XL is a widely used clinical 12-lead ECG dataset spanning diverse morphologies and rhythms. We therefore use it as the unlabeled set for semi-supervised training.

3) Evaluation data: The QTDB and ISP test splits together form our internal test set. We additionally use four external test sets, chosen to span different sources, acquisition devices, and populations, so that robustness can be assessed under the diverse distribution shifts arising in practice. Each external set contributes a distinct aspect of evaluation. LUDB is a widely used benchmark for ECG delineation, providing high-quality, per-lead annotations. Zhejiang consists of recordings that each contain arrhythmic beats, predominantly premature ventricular complexes (PVCs), enabling delineation assessment under arrhythmic conditions. RDB offers delineation labels across eight rhythm types, from sinus rhythms to various arrhythmias, supporting rhythm-stratified subgroup analysis. Finally, mECGDB comprises recordings collected with a portable device outside clinical settings, testing robustness under less controlled conditions. All external sets provide dense timepoint-level delineation labels, except mECGDB, which offers only interval-level annotations (PR interval, QRS duration, and QT interval). The collection of mECGDB was approved by the Institutional Review Board (IRB) of Chonnam National University Hospital (IRB number: CNUH-2025-127), and informed consent was obtained from all participants.

4) Preprocessing: Each ECG is adjusted to a fixed 10- second duration by cropping or zero-padding as needed, resampled to 250 Hz, band-pass filtered (0.67–40 Hz) to suppress baseline wander and high-frequency noise, and Zscore normalized before being passed to the models.

## B. Evaluation Protocol

We evaluate all methods with three complementary metrics: mean Intersection over Union (mIoU), an averaged interval error, and averaged point-wise sensitivity. First, mIoU quantifies the overlap between predicted and reference segments across the P, QRS, and T classes, reflecting segmentation accuracy per time point. It is computed per sample and then averaged across the test set. Second, for each clinical interval (PR interval, QRS duration, and QT interval), we compute its mean absolute error against the reference annotations, averaged over the samples where that interval is available; averaging the three perinterval errors yields a single averaged interval error, capturing the accuracy of the measurements used in clinical practice. Third, we report point-wise sensitivity at the onset and offset of the P, QRS, and T waves, counting a predicted point as correct if it falls within a fixed tolerance of the reference. Sensitivity is conventionally reported per fiducial point, but for a concise comparison across methods and test sets, we report the average over the six onset and offset points and present the sensitivity for each point separately in the point-wise performance analysis. We evaluate the averaged point-wise sensitivity at a 40 ms tolerance, in line with recent delineation studies [28], [39]. For a more detailed point-wise analysis, we additionally report the positive predictive value (PPV) as a precision counterpart to sensitivity, extend the tolerance to 150 ms for both metrics following earlier studies [40], [41], and evaluate localization error as the signed time difference between each predicted onset and offset point and its reference annotation. Full implementations of all metrics are provided in the Supplementary Material (Section IX).

For the comparison among learning strategies, we report the mean and standard deviation over runs with different random seeds to account for training stochasticity (e.g., weight initialization, mini-batch ordering). The statistical significance of the comparison is assessed via paired bootstrap and paired Wilcoxon signed-rank tests, with Holm correction over each family of comparisons [42]–[44]. Details are provided in the Supplementary Material (Section XII), which also reports a seed-level robustness check of these comparisons.

## IV. EXPERIMENTAL SETUP

Our study proceeds in two stages: (1) developing deep learning delineation models under supervised, semi-supervised, and self-supervised strategies and selecting the best-performing pretraining objective, and (2) benchmarking the selected model against existing delineation tools.

## A. Comparing Learning Strategies for ECG Delineation

The first stage compares learning strategies to see whether self-supervised pretraining benefits delineation and how it interacts with the use of unlabeled data during fine-tuning. Strategies vary along two factors: (1) whether the encoder is initialized by self-supervised pretraining on a large-scale unlabeled dataset from a separate source or trained from scratch; and (2) whether training uses labeled data only (supervised) or additionally exploits an unlabeled set (semi-supervised). Crossing the two factors lets us isolate the individual effects of self-supervised pretraining and of semi-supervised training by varying each factor in turn. For pretraining, we select one representative objective from each of the three major families of SSL:

• MAE [25], a generative objective that masks a large fraction of the input and trains the encoder to reconstruct it.

• MoCo [45], a contrastive objective that pulls together augmented views of the same signal and pushes apart views of different signals.

• MERL [15], a multimodal objective that aligns each ECG with its paired textual report in a shared representation space.

For semi-supervised learning, we adopt Mean Teacher (MT) [46], a consistency-regularization method shown to be strong and robust for ECG delineation [23], with a weak-strong augmentation scheme.

## B. Benchmarking Against Existing Delineation Tools

The second stage benchmarks the deep learning model developed with the best-performing pretraining objective from the first stage, under both supervised and semi-supervised finetuning, against existing open-source (rule-based) and commercial delineation tools. We evaluate four widely used delineation tools, three open-source and one commercial:

• NeuroKit2 [18], a Python toolbox for biosignal processing that includes wavelet-based ECG delineation.

• ECGdeli [19], a MATLAB toolbox providing waveletbased delineation.

• Prominence [20], a Python implementation of peakprominence delineation.

• CalECG [47], a commercial package used in regulatory and pharmaceutical settings.

## C. Implementation Details

For the delineation tools, we use the default configurations without manual correction. For NeuroKit2, two wavelet variants are evaluated: the continuous wavelet transform (CWT)

TABLE II  
BENCHMARKING RESULTS OF LEARNING STRATEGIES FOR ECG DELINEATION. PER-DATASET ENTRIES ARE MIOU, REPORTED AS MEAN ± STANDARD DEVIATION OVER THREE SEEDS, WHERE INTERNAL DENOTES THE $Q T D B + I S P$ TEST SPLIT. UNDER SUMMARY, AVG. IS THE MIOU AVERAGED ACROSS THE FOUR DATASETS FOR EACH FINE-TUNING STRATEGY, AND RANK IS THE AVERAGE RANK AMONG THE FOUR PRETRAINING STRATEGIES, AGGREGATED OVER FINE-TUNING AND DATASETS. THE BEST VALUE IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td></td><td></td><td colspan="4">mIoU (%) ↑</td><td colspan="2">Summary</td></tr><tr><td>Pretraining</td><td>Fine-tuning</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>RDB</td><td>Avg. (%) ↑</td><td>Rank↓</td></tr><tr><td rowspan="2"></td><td>Supervised</td><td> $7 5 . 2 \pm 0 . 2$ </td><td> $6 4 . 8 \pm 0 . 8$ </td><td> $5 3 . 3 \pm 2 . 4$ </td><td> $6 0 . 5 \pm 0 . 5$ </td><td>63.5</td><td>3.50</td></tr><tr><td>Semi-supervised</td><td> $8 1 . 6 \pm 0 . 2$ </td><td> $7 0 . 0 \pm 5 . 9$ </td><td> $6 6 . 0 \pm 5 . 3$ </td><td> $6 7 . 9 \pm 0 . 9$ </td><td>71.4</td><td></td></tr><tr><td rowspan="2">MAE*</td><td>Supervised</td><td> ${ \bf 8 2 . 8 \pm 0 . 3 }$ </td><td> $7 1 . 3 \pm 2 . 3$ </td><td> ${ \bf 7 5 . 2 \pm 1 . 5 }$ </td><td> ${ \bf 7 1 . 9 \pm 1 . 2 }$ </td><td>75.3</td><td>1.25</td></tr><tr><td>Semi-supervised</td><td> $8 2 . 3 \pm 0 . 5$ </td><td> $7 2 . 8 \pm 5 . 3$ </td><td> $7 4 . 8 \pm 2 . 1$ </td><td> $7 1 . 1 \pm 0 . 4$ </td><td>75.2</td><td></td></tr><tr><td rowspan="2"> $\mathbf { M o C o ^ { * } }$ </td><td>Supervised</td><td> $8 2 . 2 \pm 0 . 1$ </td><td> $6 9 . 5 \pm 3 . 3$ </td><td> $6 6 . 2 \pm 3 . 7$ </td><td> $7 1 . 6 \pm 1 . 5$ </td><td>72.4</td><td>1.75</td></tr><tr><td>Semi-supervised</td><td> $8 2 . 5 \pm 0 . 1$ </td><td> ${ \bf 7 3 . 3 \pm 3 . 9 }$ </td><td> $7 3 . 3 \pm 1 . 0$ </td><td> $7 0 . 9 \pm 0 . 7$ </td><td>75.0</td><td></td></tr><tr><td rowspan="2">MERL</td><td>Supervised</td><td> $7 6 . 3 \pm 0 . 2$ </td><td> $6 7 . 0 \pm 0 . 7$ </td><td> $5 2 . 3 \pm 0 . 7$ </td><td> $6 1 . 7 \pm 0 . 3$ </td><td>64.3</td><td></td></tr><tr><td>Semi-supervised</td><td> $8 1 . 0 \pm 0 . 1$ </td><td> $6 1 . 8 \pm 2 . 1$ </td><td> $6 2 . 6 \pm 5 . 0$ </td><td> $6 9 . 4 \pm 0 . 5$ </td><td>68.7</td><td>3.50</td></tr></table>

<sup>∗</sup>: significantly better than without pretraining.

and the discrete wavelet transform (DWT). For CalECG, we use the four fiducial points that define the clinical intervals (P onset, QRS onset, QRS offset, and T offset). It is therefore evaluated on these four fiducial points and on the intervallevel metrics, but not on the remaining fiducial points assessed for the other tools. The configuration used for each tool is provided in the Supplementary Material (Section X).

The deep learning model architecture is a ViT-Tiny [48], [49] encoder with a two-layer FCN decoder [50], using a 128-dimensional hidden layer and a dropout rate of $p =$ 0.1 [51]. We adopt this encoder for its strong performance in ECG analysis and its effectiveness for semi-supervised ECG segmentation [23]. We keep the architecture fixed across all learning strategies, so that any performance difference is attributable to the strategy rather than to architectural factors.

For pretraining, we follow the training schedule from each objective’s original work. For delineation, both supervised and semi-supervised models are trained for 100 epochs with a batch size of 16, using AdamW [52] with weight decay 0.05 and a cosine learning rate schedule [53] that warms up to 0.001 over 10 epochs before annealing to 0.0001. For semi-supervised learning, an unlabeled batch of equal size is sampled alongside the labeled batch for consistency regularization. Each model is trained with three random seeds, and we use the run with the median validation mIoU for the tool comparison. These experiments are implemented in PyTorch 1.11 with Python 3.9 and run on four NVIDIA RTX 4090 GPUs. Full training configurations are provided in the Supplementary Material (Section XI)

## V. RESULTS

## A. Comparison of Learning Strategies

Table II reports the delineation performance (mIoU) of the different learning strategies across all test sets, with the averaged interval error and the averaged point-wise sensitivity results provided in the Supplementary Material (Tables S8– S9).

Without pretraining, semi-supervised learning improves over supervised learning. The improvement is largely consistent across metrics and datasets. mIoU improves on all datasets (63.5 to 71.4% on average), and averaged point-wise sensitivity shows a similar trend. The averaged interval error also improves on all datasets except RDB, where supervised learning gives a lower error. Zhejiang shows the largest gains across all three metrics, with mIoU, for example, rising from 53.3 to 66.0%. Given that Zhejiang consists entirely of recordings with arrhythmic beats, this suggests that the benefit is pronounced under arrhythmic conditions.

Self-supervised pretraining helps, but the objective matters. Across all datasets, the best performance is attained by a pretrained strategy. The benefit of pretraining, however, depends on the objective. MAE and MoCo significantly outperform no pretraining, whereas MERL does not. Averaged across datasets, MERL underperforms the other objectives by a clear margin (mIoU 68.7 vs. 75.2% for MAE and 75.0% for MoCo under semi-supervised fine-tuning). On the internal test set, which is derived from the same data source as the training data, MAE and MoCo are near saturation (mIoU 82.2–82.8%) and even MERL approaches them under semi-supervised finetuning (81.0 vs. 82.3–82.5%). These differences among the objectives widen on the external sets. Furthermore, the two effective objectives interact differently with fine-tuning. MAE performs comparably under supervised and semi-supervised fine-tuning (mIoU 75.3 vs. 75.2%), whereas MoCo benefits substantially from semi-supervised fine-tuning (mIoU 72.4 vs. 75.0%), most clearly on Zhejiang (mIoU $6 6 . 2 ~  ~ 7 3 . 3 \% )$ with gains of similar magnitude in averaged interval error and averaged point-wise sensitivity.

MAE and MoCo, the two effective objectives, show no significant difference. We adopt MAE for its higher mIoU on average and better average rank, as well as its stability under both supervised and semi-supervised fine-tuning, which MoCo lacks. The resulting models under supervised and semisupervised fine-tuning, denoted DL DELINEATOR (sup) and (semi), serve as the deep learning models in the analyses that follow.

## B. Comparison with Existing Tools

Table III reports the benchmark against existing delineation tools across the internal and three external test sets (LUDB,

TABLE III  
COMPARISON OF THE DL DELINEATOR WITH OPEN-SOURCE DELINEATION TOOLS. THE BEST VALUE IN EACH COLUMN IS IN BOLD, AND THE SECOND-BEST IS UNDERLINED. AVG. RANK IS THE MEAN RANK ACROSS ALL COLUMNS. CALECG, CONFIGURED FOR INTERVAL OUTPUTS ONLY, IS COMPARED SEPARATELY ON INTERVAL ERROR IN THE SUPPLEMENTARY MATERIAL (TABLE S10).
<table><tr><td></td><td colspan="4">mIoU (%) ↑</td><td colspan="4">Averaged interval error (ms) ↓</td><td colspan="4">Averaged point-wise sensitivity (%) ↑</td><td></td></tr><tr><td>Method</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>RDB</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>RDB</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>RDB</td><td> $\operatorname { A v g } .$  Rank↓</td></tr><tr><td>NeuroKit2 (CWT)</td><td>41.3</td><td>44.9</td><td>7.3</td><td>40.0</td><td>70.9</td><td>108.3</td><td>72.8</td><td>30.2</td><td>59.4</td><td>59.9</td><td>15.7</td><td>54.9</td><td>5.8</td></tr><tr><td>NeuroKit2 (DWT)</td><td>43.4</td><td>45.6</td><td>8.6</td><td>38.6</td><td>43.6</td><td>45.6</td><td>52.7</td><td>40.5</td><td>60.3</td><td>62.2</td><td>17.8</td><td>58.5</td><td>5.2</td></tr><tr><td>ECGdeli</td><td>67.5</td><td>59.4</td><td>41.9</td><td>54.8</td><td>24.6</td><td>29.3</td><td>30.4</td><td>30.0</td><td>82.4</td><td>80.6</td><td>56.3</td><td>76.4</td><td>3.4</td></tr><tr><td>Prominence</td><td>60.4</td><td>63.9</td><td>42.7</td><td>51.7</td><td>28.1</td><td>22.3</td><td>35.9</td><td>23.0</td><td>78.1</td><td>82.9</td><td>54.6</td><td>73.7</td><td>3.6</td></tr><tr><td>DL DELINEATOR (sup)</td><td>83.0</td><td>71.6</td><td>73.5</td><td>71.1</td><td>11.5</td><td>19.0</td><td>16.3</td><td>22.2</td><td>96.1</td><td>86.6</td><td>82.9</td><td>92.6</td><td>1.8</td></tr><tr><td>DL DELINEATOR (semi)</td><td>82.4</td><td>75.8</td><td>75.1</td><td>71.3</td><td>12.0</td><td>16.8</td><td>15.3</td><td>21.7</td><td>96.1</td><td>91.1</td><td>86.0</td><td>92.5</td><td>1.3</td></tr></table>

TABLE IV

COMPARISON OF THE DL DELINEATOR ON mECGDB WITH OPEN-SOURCE AND COMMERCIAL DELINEATION TOOLS, EVALUATED BY MEAN ABSOLUTE ERROR (MS) FOR THE PR INTERVAL, QRS DURATION, QT INTERVAL, AND THEIR AVERAGE. THE BEST VALUE IS IN BOLD AND THE SECOND-BEST IS UNDERLINED.
<table><tr><td>Method</td><td>Average</td><td>PR</td><td>QRS</td><td>QT</td></tr><tr><td>NeuroKit2 (CWT)</td><td>97.3</td><td>207.4</td><td>32.1</td><td>52.3</td></tr><tr><td>NeuroKit2 (DWT)</td><td>51.3</td><td>45.3</td><td>17.0</td><td>91.5</td></tr><tr><td>ECGdeli</td><td>32.9</td><td>47.0</td><td>18.0</td><td>33.6</td></tr><tr><td>CalECG</td><td>37.9</td><td>41.1</td><td>32.7</td><td>40.0</td></tr><tr><td>Prominence</td><td>25.4</td><td>29.6</td><td>25.0</td><td>21.7</td></tr><tr><td>DL DELINEATOR (sup)</td><td>15.0</td><td>13.8</td><td>12.2</td><td>19.1</td></tr><tr><td>DL DELINEATOR (semi)</td><td>15.4</td><td>15.3</td><td>11.8</td><td>19.1</td></tr></table>

Zhejiang, and RDB). Among the existing tools, ECGdeli and Prominence are the top two, with average ranks of 3.4 and 3.6, respectively. The DL DELINEATOR significantly surpasses all the benchmarked tools across every metric and dataset, with its two fine-tuning configurations taking the best and secondbest average ranks (1.3 and 1.7, respectively).

The advantage of the DL DELINEATOR is most pronounced on datasets with arrhythmias. On RDB, which spans a wide range of rhythms, the DL DELINEATOR clearly outperforms the strongest tool, ECGdeli (mIoU 71.3 vs. 54.8%; averaged point-wise sensitivity 92.6 vs. 76.4%). The margin is largest on Zhejiang (mIoU 75.1 vs. 42.7% for Prominence; averaged interval error 15.3 vs. 30.4 ms and averaged pointwise sensitivity 86.0 vs. 56.3% for ECGdeli). For the pointwise sensitivity, these margins narrow under the less stringent 150 ms tolerance, but their ordering is unchanged; the same pattern holds for the point-wise PPV in the Supplementary Material (Table S11).

Table IV reports the averaged interval error on mECGDB. The DL DELINEATOR again achieves the lowest error, reducing the average from 25.4 ms for the best tool (Prominence) to 15.0 ms. Both configurations improve significantly on the average, the PR interval, and the QRS duration. For the QT interval, both remain the most accurate, each with an error of 19.1 ms compared with 21.7 ms for Prominence, and the improvement is significant in all cases except the supervised configuration against Prominence. Its advantage therefore holds even for portable, out-of-clinic recordings.

## C. Point-wise Performance Characteristics

To characterize each tool’s distinctive behavior, we examine point-wise sensitivity on LUDB and Zhejiang (Table V), chosen for their contrasting morphology: LUDB is dominated by sinus rhythms, whereas Zhejiang consists entirely of recordings with ventricular ectopic and tachycardic beats. The corresponding point-wise PPV and localization errors (signed mean ± SD) are reported in the Supplementary Material (Table S12–S13).

The two NeuroKit2 variants are each limited at a different fiducial point of the beat. The CWT-based method shows particularly low sensitivity at $\mathrm { \bf P _ { o n } }$ (43.7 and 14.7% on LUDB and Zhejiang, respectively), whereas the DWT-based method struggles at $\mathrm { Q R S _ { o n } }$ and $\mathrm { T _ { o f f } }$ , the two fiducial points that define the QT interval (e.g., 66.2 and 55.8% on LUDB). These observations are consistent with their errors on mECGDB, where the CWT-based and DWT-based methods show large PR and QT interval errors, respectively (207.4 and 91.5 ms; Table IV).

CalECG was configured to output only the fiducial points required to define the clinical intervals; it therefore provides $\mathrm { P _ { o n } , \ Q R S _ { o n } , \ Q R S _ { o f f } , }$ , and $\mathrm { T _ { o f f } }$ , but not $\mathrm { P _ { o f f } }$ or $\mathrm { T _ { o n } }$ , and was evaluated only at these four points. Among them, its sensitivity was moderate on LUDB (68.1% overall) but well below ECGdeli and Prominence, and it dropped sharply under arrhythmia, reaching only 7.0% overall on Zhejiang.

ECGdeli and Prominence achieve the highest performance among the evaluated tools, although their superiority varies across datasets and fiducial points. ECGdeli is more accurate on the QRS complex, giving the highest $\mathrm { Q R S _ { o n } }$ sensitivity among the tools on both LUDB and Zhejiang (97.3 and 83.2%, respectively), whereas Prominence is more reliable at $\mathrm { \bf P _ { o n } }$ (79.2 and 65.4%, respectively). For the remaining points, the leading tool depends on the dataset.

By contrast, the DL DELINEATOR surpasses every tool at nearly all fiducial points. The two fine-tuning configurations differ only at a few points on LUDB: supervised fine-tuning falls short of the best tool at $\mathrm { P _ { o f f } }$ (81.3 vs. 87.2% for ECGdeli) and $\mathrm { T _ { o n } }$ (77.8 vs. 79.1% for Prominence), whereas semisupervised fine-tuning exceeds all tools at both (90.3 and 79.2%). On Zhejiang, both configurations lead at every point by a wide margin (e.g., $\mathrm { T _ { o f f } }$ 85.4 vs. 45.4% for ECGdeli).

TABLE V  
COMPARISON OF THE DL DELINEATOR ON EXTERNAL TEST SETS (LUDB AND Zhejiang) WITH OPEN-SOURCE AND COMMERCIAL DELINEATION TOOLS, EVALUATED BY POINT-WISE SENSITIVITY (%) PER FIDUCIAL POINT. "TOTAL" IS THE AVERAGE OVER THE AVAILABLE ONSET/OFFSET POINTS. THE BEST VALUE IS IN BOLD AND THE SECOND-BEST IS UNDERLINED.
<table><tr><td>Dataset</td><td>Method</td><td> $\mathrm { P _ { o n } }$ </td><td> $\mathrm { P _ { o f f } }$ </td><td> $\mathrm { Q R S _ { o n } }$ </td><td> $\mathrm { Q R S _ { o f f } }$ </td><td> $\mathrm { T _ { o n } }$ </td><td> $\mathrm { T _ { o f f } }$ </td><td>Total</td></tr><tr><td rowspan="7">LUDB</td><td>NeuroKit2 2 (CWT)</td><td>43.7</td><td>43.1</td><td>80.8</td><td>72.2</td><td>56.4</td><td>63.4</td><td>59.9</td></tr><tr><td>NeuroKit2 (DWT)</td><td>73.3</td><td>63.8</td><td>66.2</td><td>81.8</td><td>32.5</td><td>55.8</td><td>62.2</td></tr><tr><td>ECGdeli</td><td>71.8</td><td>87.2</td><td>97.3</td><td>85.1</td><td>70.7</td><td>71.7</td><td>80.6</td></tr><tr><td>CalECG</td><td>74.0</td><td></td><td>67.7</td><td>69.1</td><td></td><td>61.5</td><td>68.1</td></tr><tr><td>Prominence</td><td>79.2</td><td>80.8</td><td>91.9</td><td>85.8</td><td>79.1</td><td>80.7</td><td>82.9</td></tr><tr><td>DL DELINEATOR (sup)</td><td>80.4</td><td>81.3</td><td>99.1</td><td>98.9</td><td>77.8</td><td>81.6</td><td>86.6</td></tr><tr><td>DL DELINEATOR (semi)</td><td>94.0</td><td>90.3</td><td>99.4</td><td>99.2</td><td>79.2</td><td>84.6</td><td>91.2</td></tr><tr><td rowspan="7">Zhejiang</td><td>NeuroKit2 (CWT)</td><td>14.7</td><td>13.6</td><td>20.8</td><td>24.5</td><td>9.8</td><td>11.0</td><td>15.7</td></tr><tr><td>NeuroKit2 (DWT)</td><td>24.4</td><td>24.6</td><td>14.3</td><td>20.0</td><td>11.4</td><td>11.8</td><td>17.8</td></tr><tr><td>ECGdeli</td><td>57.7</td><td>37.3</td><td>83.2</td><td>67.4</td><td>46.5</td><td>45.4</td><td>56.3</td></tr><tr><td>CalECG</td><td>5.6</td><td></td><td>5.7</td><td>9.6</td><td></td><td>7.2</td><td>7.0</td></tr><tr><td>Prominence</td><td>65.4</td><td>64.8</td><td>68.0</td><td>66.4</td><td>32.5</td><td>30.4</td><td>54.6</td></tr><tr><td>DL DELINEATOR (sup)</td><td>82.6</td><td>79.9</td><td>92.0</td><td>93.2</td><td>69.1</td><td>80.9</td><td>82.9</td></tr><tr><td>DL DELINEATOR (semi)</td><td>87.5</td><td>84.1</td><td>94.6</td><td>94.6</td><td>69.6</td><td>85.4</td><td>86.0</td></tr></table>

TABLE VI

RHYTHM-STRATIFIED SUBGROUP PERFORMANCE ON RDB FOR THE SINUS AND ARRHYTHMIA GROUPS, WITH THE RELATIVE PERFORMANCE DEGRADATION FROM SINUS TO ARRHYTHMIA (SMALLER IS BETTER). THE BEST VALUE IS IN BOLD AND THE SECOND-BEST IS UNDERLINED.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="2">Subgroup performance</td><td rowspan="2">Degradation</td></tr><tr><td>Sinus</td><td>Arrhythmia</td></tr><tr><td rowspan="4">mIoU (%) ↑</td><td>ECGdeli</td><td>66.3</td><td>38.7</td><td>41.6%</td></tr><tr><td>Prominence</td><td>64.4</td><td>33.2</td><td>48.5%</td></tr><tr><td>DL DELINEATOR (sup)</td><td>76.1</td><td>64.9</td><td>14.7%</td></tr><tr><td>DL DELINEATOR (semi)</td><td>76.6</td><td>64.2</td><td>16.2%</td></tr><tr><td rowspan="4">Averaged interval error (ms) ↓</td><td>ECGdeli</td><td>28.9</td><td>40.1</td><td>38.8%</td></tr><tr><td>Prominence</td><td>18.2</td><td>38.2</td><td>110.0%</td></tr><tr><td>DL DELINEATOR (sup)</td><td>20.6</td><td>28.5</td><td>38.4%</td></tr><tr><td>DL DELINEATOR (semi)</td><td>19.4</td><td>30.2</td><td>56.0%</td></tr><tr><td rowspan="4">Averaged point-wise sensitivity (%) ↑</td><td>ECGdeli</td><td>81.1</td><td>67.3</td><td>17.1%</td></tr><tr><td>Prominence</td><td>83.5</td><td>58.3</td><td>30.2%</td></tr><tr><td>DL DELINEATOR (sup)</td><td>96.4</td><td>80.1</td><td>16.9%</td></tr><tr><td>DL DELINEATOR (semi)</td><td>96.5</td><td>80.4</td><td>16.7%</td></tr></table>

## D. Rhythm-stratified Subgroup Analysis

RDB contains a large and diverse set of rhythms, making it well-suited for a rhythm-stratified subgroup analysis. We evaluate all metrics for each rhythm and compare the DL DELINEATOR in detail with the two leading tools, ECGdeli and Prominence. In Table VI, rhythms are grouped into sinus and arrhythmia; we report each group’s mean and the relative degradation between them, with per-rhythm results in the Supplementary Material (Table S14).

In the sinus group, both configurations of the DL DE-LINEATOR lead in mIoU and averaged point-wise sensitivity by a wide margin (e.g., mIoU 76.1 and 76.6 vs. 66.3% for ECGdeli), and trail only Prominence on averaged interval error, by a small margin (19.4 vs. 18.2 ms). Under arrhythmia, both configurations are the most accurate on every metric, outperforming the best tool on each (mIoU 64.9 vs. 38.7%; interval error 28.5 vs. 38.2 ms; sensitivity 80.4 vs. 67.3%).

The DL DELINEATOR is also the most robust to sinusto-arrhythmia shift. On mIoU, its relative degradation is far smaller than that of either tool (14.7 and 16.2 vs. 41.6 and 48.5%), and on sensitivity it is again the smallest (16.7 vs.

17.1% for ECGdeli). Notably, the DL DELINEATOR combines strong accuracy on sinus rhythms with the smallest relative degradation, maintaining its lead under arrhythmia. For interval error, Prominence degrades sharply (110.0%) and ECGdeli starts from the worst error in both groups (28.9 and 40.1 ms), whereas the DL DELINEATOR keeps the lowest interval error under arrhythmia.

## E. Qualitative Analysis

Fig. 2 shows representative delineation results across methods and datasets, with additional per-dataset examples in the Supplementary Material (Fig. S1–S3), which follow similar trends. The examples make the differences between methods visually apparent and align with the quantitative results.

On sinus rhythms (LUDB and RDB), most methods recover the overall P, QRS, and T structure, and the differences appear mainly at the P and QRS onsets and offsets. NeuroKit2 is the exception, remaining unreliable even on sinus rhythms: the CWT-based method misses the first P wave on RDB, and the DWT-based method misses the first QRS on LUDB and delineates the T wave too narrowly. These failures are consistent with the low point-wise sensitivity reported earlier, at $\mathrm { \Delta P _ { o n } }$ for the CWT-based method and at the QRS and T boundaries for the DWT-based method.

![](images/99cb06faa59ac440722ee67abd606055f9249c38112c649866ec5c54218bdc7e.jpg)  
Fig. 2. Qualitative examples of ECG delineation results across methods and datasets. The rhythm classes shown reflect the characteristics of each dataset. Rows are grouped into the reference annotation (GT), rule-based tools, and the DL DELINEATOR; shaded bands mark each method’s detected P (blue), QRS (orange), and T (green) waves on the same input signal. SR, sinus rhythm; PVC, premature ventricular contraction; AFIB, atrial fibrillation; SVT, supraventricular tachycardia.

The differences become pronounced under arrhythmia. On Zhejiang, which contains premature ventricular complexes, both NeuroKit2 variants fail to delineate; ECGdeli misses the P wave; and Prominence misses the P wave and QRS complex of the beat preceding the ectopic beat, whereas the DL DELINEATOR stays close to the reference. On RDB, the existing tools produce characteristic false positives under non-sinus rhythm. For example, under atrial fibrillation and supraventricular tachycardia, both ECGdeli and Prominence mark spurious P waves, in the latter case by labeling T waves as P waves. In contrast, both configurations of the DL DELINEATOR remain aligned with the reference across all rhythm types.

Overall, these qualitative results mirror the point-wise and rhythm-stratified findings: the existing tools are reliable on sinus rhythms but lose accuracy at the P and T boundaries and break down under arrhythmia, whereas the DL DELINEATOR stays robust across datasets and rhythm types.

## VI. DISCUSSION

## A. Effectiveness of Label-Efficient Deep Learning

Self-supervised pretraining is effective for ECG delineation. Pretraining on a large unlabeled dataset with a suitable objective improves performance across all test sets, with the clearest gains on the external sets (Table II). A plausible interpretation is that unlabeled data exposes the model to a wider range of morphologies, rhythms, and noise than the labeled set alone, which should matter most when the test distribution deviates from that of the labeled training set. These results establish self-supervised pretraining as a practical approach to labelefficient delineation and broaden the strategies available for this task.

The choice of pretraining objective also matters. The reconstruction objective (MAE) and the unimodal contrastive objective (MoCo) transfer well, whereas the multimodal contrastive objective (MERL), which additionally uses paired text reports, underperformed. Because MoCo is also contrastive, the gap does not appear to stem from contrastive learning itself; one possibility is that aligning ECGs to text encourages invariance to the fine morphological detail that delineation depends on. The benefit of self-supervised pretraining therefore depends on the design of the pretraining objective, not on self-supervised learning alone, making an objective tailored to per-time-point localization a promising direction for future work.

Self-supervised pretraining and semi-supervised learning each improve delineation over supervised learning, but their benefits do not simply add up (Table II). The contribution of semi-supervised learning depends on how much room the supervised baseline leaves: it is the largest without pretraining, remains substantial for MoCo, and is negligible for MAE, whose supervised fine-tuning performance is already saturated. A plausible interpretation is that a sufficiently effective pretraining objective already captures much of what the unlabeled recordings offer, leaving little for semi-supervised fine-tuning to add. Because it is difficult to know in advance whether a given objective leaves such room, semi-supervised fine-tuning is a safe default across the objectives we examine: it helps when room remains and at least does not degrade performance.

## B. The Advantage of DL DELINEATOR over Existing Tools

Across the internal and three external sets, the DL DELIN-EATOR outperforms every tool on all three metrics, and its improvements are statistically significant in all comparisons (Table III). On a further external set recorded outside clinical conditions, mECGDB, it again achieves the lowest error (Table IV). The advantage holds under both fine-tuning strategies, consistent with our choice of MAE, which we select for its stability across fine-tuning schemes. Together, these results provide systematic evidence that label-efficient deep learning offers a measurable and consistent advantage over existing open-source and commercial tools, addressing whether it is worth adopting for delineation in practice.

## C. Performance across Fiducial Points and Rhythms

A point-wise analysis (Table V) localizes where the two fine-tuning strategies differ: semi-supervised fine-tuning closes the residual gaps that supervised fine-tuning leaves at $\mathrm { P _ { o f f } }$ and $\mathrm { T _ { o n } }$ on LUDB. Because this comparison uses a single selected model per configuration, we read these differences as tendencies rather than precise magnitudes. This refines, rather than contradicts, the saturation reported in Section VI-A: that saturation is in the averaged metric, whereas these differences are confined to individual boundaries. Even when the averaged metric shows no measurable improvement, semi-supervised fine-tuning thus improves a few boundaries without degrading others, supporting its use as a low-risk default. Beyond this difference, the DL DELINEATOR (semi) is consistent across fiducial points; its weakest region is the T-wave boundaries, and $\mathrm { T _ { o n } }$ in particular, where the sensitivity is the lowest of the six points. These are the hardest points for every tool, reflecting the gradual, low-amplitude onset of the T-wave rather than a limitation specific to our model, so improving localization there while maintaining overall performance is a worthwhile direction for future work.

We next stratify the same results by rhythm (Table VI), a property of direct practical importance, since abnormal rhythms are common in clinical recordings and are harder to delineate. The DL DELINEATOR retains its accuracy under arrhythmia, remaining the most accurate method on every metric. This is consistent with learned representations capturing morphological variability that fixed heuristics cannot. Moreover, rule-based tools are often evaluated on predominantly sinus recordings, so their reported accuracy may not carry over to the arrhythmic conditions common in practice.

## D. From Tool Selection to a Single Robust Delineator

Among the tools, ECGdeli and Prominence are the strongest and essentially indistinguishable in average rank (3.5 vs. 3.7). For users who rely on them, the choice is better framed by what is measured and on which rhythm than by which tool is better overall. On predominantly sinus recordings, Prominence is a reasonable default: it leads ECGdeli across metrics on LUDB and in the sinus subgroup of RDB, and is more reliable at $\mathrm { \bf P _ { o n } } .$ , making it preferable when P-onset localization is required (Tables III, V, VI). ECGdeli is preferable when the QRS complex is the target (at $\mathrm { Q R S _ { o n } }$ and QRS duration) and is the safer choice when rhythm composition is mixed or unknown, degrading less from sinus to arrhythmia and holding up better on the T wave under arrhythmia (Tables IV, V, VI).

In practice, however, such tool selection is hard to apply. Under arrhythmia, both tools degraded sharply (Table VI), and choosing the right tool presupposes knowing the rhythm of each incoming recording in advance, which requires an additional classification step. A single learned delineator avoids this: the DL DELINEATOR remains robust across acquisition settings and rhythm types, so it can be deployed without characterizing each recording or switching tools between settings, while remaining more accurate than either tool. Moreover, it runs faster than these tools (Supplementary Material, Table S7), so its accuracy does not come at the cost of speed.

## E. Limitations and Future Work

Our study has several limitations. First, we examined only a subset of possible self-supervised approaches (three objectives with a single, established backbone), so the strategies we identify are effective, practical directions rather than the optimal choice. A broader exploration of self-supervised objectives is therefore a natural next step. Because the benefit of pretraining depends strongly on the objective, with MAE (a dense reconstruction task) transferring best, a promising direction is to move beyond generic reconstruction toward pretext tasks that target temporal localization more directly, such as objectives built around the wave boundaries that delineation must predict. Second, mECGDB provides interval-level labels alone, so the evidence under uncontrolled conditions rests on a single metric; recordings annotated across a wider range of devices would allow the full metric set to be assessed under such conditions.

## VII. CONCLUSION

We studied label-efficient deep learning for ECG delineation and compared it, under identical conditions, against widely used open-source and commercial tools on one internal and four external test sets using three complementary metrics. Selfsupervised pretraining is effective, though its benefit depends on the objective: with MAE pretraining, the resulting DL DELINEATOR outperforms every tool on all three metrics across datasets under both fine-tuning configurations, with the improvement statistically significant in nearly all comparisons and its margin widening under arrhythmia. Resolving performance by fiducial point and rhythm further shows that the tools behave differently, concentrating at the P and T waves and under arrhythmia, whereas the DL DELINEATOR remains robust throughout. Overall, self-supervised pretraining enables deep learning models to surpass existing delineation tools using only a modest amount of labeled data. This work supports wider adoption of label-efficient learning and lays the groundwork for reliable ECG delineation in clinical practice.

## DATA AVAILABILITY

All datasets except mECGDB are publicly available. For QTDB, ISP, PTB-XL, LUDB, and Zhejiang, we used the preprocessed versions provided by the SemiSegECG benchmark<sup>1</sup>. $R D B ^ { 2 }$ and $M I M I C - I V - E C G ^ { 3 }$ are available from their original sources. mECGDB is a private dataset and cannot be shared due to user privacy and ethical restrictions. However, it can be requested from the corresponding author, subject to institutional approval and user consent.

## REFERENCES

[1] A. Gacek and W. Pedrycz, ECG signal processing, classification and interpretation: a comprehensive framework of computational intelligence. Springer Science & Business Media, 2011.

[2] P. M. Rautaharju, B. Surawicz, and L. S. Gettes, “Aha/accf/hrs recommendations for the standardization and interpretation of the electrocardiogram: part iv: the st segment, t and u waves, and the qt interval: a scientific statement from the american heart association electrocardiography and arrhythmias committee, council on clinical cardiology; the american college of cardiology foundation; and the heart rhythm society: endorsed by the international society for computerized electrocardiology,” Circulation, vol. 119, no. 10, pp. e241–e250, 2009.

[3] P. Kligfield, L. S. Gettes, J. J. Bailey, R. Childers, B. J. Deal, E. W. Hancock, G. Van Herpen, J. A. Kors, P. Macfarlane, D. M. Mirvis et al., “Recommendations for the standardization and interpretation of the electrocardiogram: part i: the electrocardiogram and its technology: a scientific statement from the american heart association electrocardiography and arrhythmias committee, council on clinical cardiology; the american college of cardiology foundation; and the heart rhythm society endorsed by the international society for computerized electrocardiology,” Circulation, vol. 115, no. 10, pp. 1306–1324, 2007.

[4] M. Elgendi, B. Eskofier, S. Dokos, and D. Abbott, “Revisiting qrs detection methodologies for portable, wearable, battery-operated, and wireless ecg systems,” PloS one, vol. 9, no. 1, p. e84018, 2014.

[5] F. Liu, C. Liu, X. Jiang, Z. Zhang, Y. Zhang, J. Li, and S. Wei, “Performance analysis of ten common qrs detectors on different ecg application cases,” Journal of healthcare engineering, vol. 2018, no. 1, p. 9050812, 2018.

[6] G. Jimenez-Perez, A. Alcaine, and O. Camara, “U-net architecture for the automatic detection and delineation of the electrocardiogram,” in 2019 Computing in Cardiology (CinC). IEEE, 2019, pp. Page–1.

[7] X. Liang, L. Li, Y. Liu, D. Chen, X. Wang, S. Hu, J. Wang, H. Zhang, C. Sun, and C. Liu, “Ecg segnet: An ecg delineation model based on the encoder-decoder structure,” Computers in biology and medicine, vol. 145, p. 105445, 2022.

[8] C. Joung, M. Kim, T. Paik, S.-H. Kong, S.-Y. Oh, W. K. Jeon, J.-h. Jeon, J.-S. Hong, W.-J. Kim, W. Kook et al., “Deep learning based ecg segmentation for delineation of diverse arrhythmias,” PloS one, vol. 19, no. 6, p. e0303178, 2024.

[9] C. Jin, Z. Guo, Y. Lin, L. Luo, and H. Chen, “Label-efficient deep learning in medical image analysis: Challenges and future directions,” arXiv preprint arXiv:2303.12484, 2023.

[10] A. Pelaez-Vegas, P. Mesejo, and J. Luengo, “A survey on semi-´ supervised semantic segmentation,” arXiv preprint arXiv:2302.09899, 2023.

[11] R. Krishnan, P. Rajpurkar, and E. J. Topol, “Self-supervised learning in medicine and healthcare,” Nature Biomedical Engineering, vol. 6, no. 12, pp. 1346–1352, 2022.

[12] A. L. Goldberger, L. A. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals,” circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[13] D. Kiyasseh, T. Zhu, and D. A. Clifton, “Clocs: Contrastive learning of cardiac signals across space, time, and patients,” in International conference on machine learning. PmLR, 2021, pp. 5606–5615.

[14] Y. Na, M. Park, Y. Tae, and S. Joo, “Guiding masked representation learning to capture spatio-temporal relationship of electrocardiogram,” in International conference on learning representations, vol. 2024, 2024, pp. 15 012–15 035.

[15] C. Liu, Z. Wan, C. Ouyang, A. Shah, W. Bai, and R. Arcucci, “Zeroshot ecg classification with multimodal learning and test-time clinical knowledge enhancement,” arXiv preprint arXiv:2403.06659, 2024.

[16] M. Al-Masud, J. M. Lopez Alcaraz, and N. Strodthoff, “Benchmarking ecg fms: A reality check across clinical tasks,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 146 689– 146 715.

[17] J. P. Mart´ınez, R. Almeida, S. Olmos, A. P. Rocha, and P. Laguna, “A wavelet-based ecg delineator: evaluation on standard databases,” IEEE Transactions on biomedical engineering, vol. 51, no. 4, pp. 570–581, 2004.

[18] D. Makowski, T. Pham, Z. J. Lau, J. C. Brammer, F. Lespinasse, H. Pham, C. Scholzel, and S. A. Chen, “Neurokit2: A python toolbox¨ for neurophysiological signal processing,” Behavior research methods, vol. 53, no. 4, pp. 1689–1696, 2021.

[19] N. Pilia, C. Nagel, G. Lenis, S. Becker, O. Dossel, and A. Loewe,¨ “Ecgdeli- an open source ecg delineation toolbox for matlab,” SoftwareX, vol. 13, p. 100639, 2021.

[20] J. Emrich, A. Gargano, T. Koka, and M. Muma, “Physiology-informed ecg delineation based on peak prominence,” in 2024 32nd European Signal Processing Conference (EUSIPCO). IEEE, 2024, pp. 1402– 1406.

[21] A. P. Porretta, C. Morgat, E. Surget, V. Fressart, A. Bloch, N. Neyroud, F. Badilini, M. Vaglio, I. Denjoy, and F. Extramiana, “Software-based analysis of t-wave morphology: identifying the electrocardiogram signature of high-risk long qt syndrome,” Europace, vol. 27, no. 9, p. euaf213, 2025.

[22] N. Engstrom, G. P. Dobson, K. Ng, K. Lander, K. Win, A. Gupta, and H. L. Letson, “Validation of calecg software for primary prevention heart failure patients: Reducing inter-observer measurement variability,” Journal of electrocardiology, vol. 74, pp. 128–133, 2022.

[23] M. Park, J. Lim, T. Yu, and S. Joo, “Semisegecg: A multi-dataset benchmark for semi-supervised semantic segmentation in ecg delineation,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 5099–5104.

[24] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597–1607.

[25] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 15 979–15 988.

[26] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2020, pp. 9726–9735.

[27] K. M. Jaeger, M. Nissen, M. Flaucher, L. Graf, J. Joanidopoulos, L. Anneken, H. Huebner, C. Goossens, A. Titzmann, C. Pontones et al., “Systematic comparison of ecg delineation algorithm performance on smartwatch data,” IEEE Access, vol. 12, pp. 160 794–160 804, 2024.

[28] A. T. Chuma, A. S. Youssef, M. H. Asmare, C. Wang, D. M. Kassie, J.- U. Voigt, and B. Vanrumste, “Performance assessment of ecg delineators on single-lead wearable ambulatory data,” medRxiv, pp. 2026–03, 2026.

[29] P. Laguna, R. G. Mark, A. Goldberg, and G. B. Moody, “A database for evaluation of algorithms for measurement of qt and other waveform intervals in the ecg,” in Computers in cardiology 1997. IEEE, 1997, pp. 673–676.

[30] G. Jimenez-Perez, J. Acosta, A. Alcaine, and O. Camara, “Generalising electrocardiogram detection and delineation: training convolutional neural networks with synthetic data augmentation,” Frontiers in Cardiovascular Medicine, vol. 11, p. 1341786, 2024.

[31] A. Avetisyan, N. Khachaturov, A. Asatryan, S. Tigranyan, and Y. Markin, “Isp ecg delineation dataset,” 2024. [Online]. Available: https://zenodo.org/doi/10.5281/zenodo.11472366

[52] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[32] P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, “Ptb-xl, a large publicly available electrocardiography dataset,” Scientific data, vol. 7, no. 1, pp. 1–15, 2020.

[53] ——, “Sgdr: Stochastic gradient descent with warm restarts,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/forum?id=Skq89Scxx

[33] B. Gow, T. Pollard, L. A. Nathanson, A. Johnson, B. Moody, C. Fernandes, N. Greenbaum, J. W. Waks, P. Eslami, T. Carbonati et al., “Mimiciv-ecg: Diagnostic electrocardiogram matched subset,” Type: dataset, vol. 6, pp. 13–14, 2023.

[34] A. I. Kalyakulina, I. I. Yusipov, V. A. Moskalenko, A. V. Nikolskiy, K. A. Kosonogov, G. V. Osipov, N. Y. Zolotykh, and M. V. Ivanchenko, “Ludb: a new open-access validation tool for electrocardiogram delineation algorithms,” IEEE access, vol. 8, pp. 186 181–186 190, 2020.

[35] J. Zheng, G. Fu, K. Anderson, H. Chu, and C. Rakovski, “A 12-lead ecg database to identify origins of idiopathic ventricular arrhythmia containing 334 patients,” Scientific data, vol. 7, no. 1, p. 98, 2020.

[36] J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, and C. Rakovski, “A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients,” Scientific data, vol. 7, no. 1, p. 48, 2020.

[37] Y. Liu, P. Zhang, X. Feng, D. Hu, D. Zhou, J. Li, K. Huang, Y. Zhao, Z. Fu, Q. Zheng et al., “Y-net-ecg: A multi-lead informed and interpretable architecture for ecg segmentation across diverse rhythms,” Expert Systems with Applications, vol. 283, p. 127955, 2025.

[38] A. Han, T. Tohyama, D. Yoon, K. Paik, B. Gow, L. A. Celi, H. Lee, and H.-C. Lee, “Review of open foundation models and datasets for ecg and ppg waveforms,” npj Digital Medicine, 2026.

[39] Z. Chen, M. Wang, M. Zhang, W. Huang, H. Gu, and J. Xu, “Postprocessing refined ecg delineation based on 1d-unet,” Biomedical Signal Processing and Control, vol. 79, p. 104106, 2023.

[40] A. I. Kalyakulina, I. I. Yusipov, V. A. Moskalenko, A. V. Nikolskiy, A. A. Kozlov, N. Y. Zolotykh, and M. V. Ivanchenko, “Finding morphology points of electrocardiographic-signal waves using wavelet analysis,” Radiophysics and Quantum Electronics, vol. 61, no. 8-9, p. 689–703, Jan. 2019. [Online]. Available: http://dx.doi.org/10.1007/s11141-019-09929-2

[41] Association for the Advancement of Medical Instrumentation, “Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms,” Association for the Advancement of Medical Instrumentation, Arlington, VA, USA, Tech. Rep. ANSI/AAMI EC57:1998/(R2020), 2020.

[42] R. J. Tibshirani and B. Efron, “An introduction to the bootstrap,” Monographs on statistics and applied probability, vol. 57, no. 1, pp. 1–436, 1993.

[43] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics bulletin, vol. 1, no. 6, pp. 80–83, 1945.

[44] S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian journal of statistics, pp. 65–70, 1979.

[45] X. Chen, S. Xie, and K. He, “An empirical study of training selfsupervised vision transformers,” in 2021 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2021, pp. 9620–9629.

[46] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” Advances in neural information processing systems, vol. 30, 2017.

[47] S. D. Cohen, M. Robert-Halabi, A. Procureur, M. Jamelot, M. Vaglio, F. Badilini, E. Prifti, and J.-E. Salem, “Validation of a novel semiautomated ecg quantification tool, applied to a cardio-oncology: Semiautomated ecg tool applied to cardio-oncology,” Cardio-Oncology, vol. 11, no. 1, p. 113, 2025.

[48] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[49] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[50] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3431–3440.

[51] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The journal of machine learning research, vol. 15, no. 1, pp. 1929–1958, 2014.

## VIII. DATASET DETAILS

This section provides further details on the ECG datasets introduced in Section III-A of the main paper and summarized in Table I. We adopt the standardized benchmark configuration of a prior open benchmark study,<sup>4</sup> which publicly released both single-lead ECG recordings and the corresponding split index files. Under this configuration, QTDB is partitioned subject-wise into training, validation, and test sets at a 6:2:2 ratio, whereas ISP follows its own predefined test split, with the remaining recordings assigned to training and validation at a 6:2 ratio. Table S1 reports the sample counts of the internal (QTDB + ISP) dataset. Tables S2 and S3 detail the rhythm composition of the external test sets.

TABLE S1  
THE DETAIL NUMBER OF THE INTERNAL (QTDB + ISP) SET SAMPLE.
<table><tr><td>Source</td><td>Train</td><td>Validation</td><td>Test</td><td>Total</td></tr><tr><td>QTDB</td><td>422</td><td>148</td><td>148</td><td>718</td></tr><tr><td>ISP</td><td>3,792</td><td>1,272</td><td>924</td><td>5,988</td></tr><tr><td>Internal (QTDB+ISP)</td><td>4,214</td><td>1,420</td><td>1,072</td><td>6,706</td></tr></table>

TABLE S2  
RHYTHM COMPOSITION OF THE LUDB AND Zhejiang USED FOR EXTERNAL TEST.
<table><tr><td>Dataset</td><td>Rhythm</td><td>#ECGs</td></tr><tr><td rowspan="8">LUDB</td><td>Sinus rhythm</td><td>143</td></tr><tr><td>Sinus tachycardia</td><td>4</td></tr><tr><td>Sinus bradycardia</td><td>25</td></tr><tr><td>Sinus arrhythmia</td><td>8</td></tr><tr><td>Irregular sinus rhythm</td><td>2</td></tr><tr><td>Abnormal rhythm (AFib, flutter)</td><td>18</td></tr><tr><td>Total</td><td>200</td></tr><tr><td></td><td>PVC (premature ventricular complex) 329</td></tr><tr><td rowspan="2">Zhejiang</td><td>VT (ventricular tachycardia)</td><td>5</td></tr><tr><td>Total</td><td>334</td></tr></table>

TABLE S3

RHYTHM COMPOSITION OF THE RDB USED FOR THE RHYTHM-STRATIFIED SUBGROUP ANALYSIS. RECORDINGS ARE GROUPED INTO SINUS AND ARRHYTHMIA CATEGORIES.

<table><tr><td>Group</td><td>Rhythm</td><td>#ECGs</td></tr><tr><td rowspan="4">Sinus</td><td>Sinus rhythm</td><td>400</td></tr><tr><td>Sinus bradycardia</td><td>400</td></tr><tr><td>Sinus tachycardia</td><td>140</td></tr><tr><td>Sinus irregularity</td><td>399</td></tr><tr><td rowspan="4">Arrhythmia</td><td>Atrial flutter</td><td>400</td></tr><tr><td>Atrial fibrillation</td><td>400</td></tr><tr><td>Atrial tachycardia</td><td>121</td></tr><tr><td>Supraventricular tachycardia</td><td>139</td></tr><tr><td>Total</td><td></td><td>2,399</td></tr></table>

<sup>4</sup>https://github.com/vuno/semi-seg-ecg

## IX. EVALUATION METRICS

The three complementary metrics used to produce the quantitative results are implemented in the Supplementary Code : (i) the mean Intersection over Union (mIoU); (ii) the mean absolute error of the PR, QRS, and QT intervals, averaged into a single averaged interval error; and (iii) the point-wise sensitivity of the P, QRS, and T onsets and offsets, averaged into a single averaged point-wise sensitivity, together with the corresponding positive predictive value (PPV) and localization error, evaluated at tolerances of 40 ms and 150 ms. All metrics are derived from the same predicted segmentation results and share a common set of functions, ensuring that the evaluation is reproducible and applied consistently across datasets.

## A. Rhythm-stratified Subgroup Analysis

We quantify the relative performance degradation from sinus to arrhythmia. For a metric $m ,$ the degradation is defined as

$$
\Delta = \frac { m _ { \mathrm { { s i n u s } } } - m _ { \mathrm { { a r r } } } } { m _ { \mathrm { { s i n u s } } } } \times 1 0 0 \% ,
$$

where $m _ { \mathrm { s i n u s } }$ and $m _ { \mathrm { a r r } }$ denote the value of m on the sinus and arrhythmia subgroups, respectively. For mIoU and averaged point-wise sensitivity, for which higher values indicate better performance, a positive $\Delta$ corresponds to degradation under arrhythmia. For averaged interval error, for which lower values indicate better performance, the sign is reversed,

$$
\Delta = \frac { m _ { \mathrm { a r r } } - m _ { \mathrm { s i n u s } } } { m _ { \mathrm { s i n u s } } } \times 1 0 0 \% ,
$$

so that a positive $\Delta$ likewise indicates degradation under arrhythmia. Under this convention, a negative $\Delta$ denotes improved performance on the arrhythmia subgroup.

## X. DELINEATAION TOOLS

## A. Open-source tools

We use the publicly available implementations of NeuroKit2<sup>5</sup>, ECGdeli<sup>6</sup> and Prominence<sup>7</sup>. All open-source tools are implemented following their documented example configurations.

## B. CalECG (AMPS LLC, New York, NY)

ECG waveforms are analyzed using CalECG (v4.2.0). Because CalECG requires a supported input format, the recordings in each dataset are converted to a CalECG-readable XML format (e.g., GE MUSE XML), preserving the ECG waveform and the dataset’s sampling rate; the software then verifies the sampling rate and signal duration of each recording (e.g., 10 s at 500 Hz for LUDB). Under Tools > Set Up Protocol, the number of beats to measure is set larger than that in any recording, and QRS detection is configured per lead (”QRS detected separately on each lead”) to minimize missed detections. After verifying signal quality on the full waveform, beats are labeled semi-automatically according to the protocol (the ”Toggles on SemiAutoomatic Mode” button), and the per-beat measurements are exported as a text file for downstream analysis. The CalECG results should be read as an unedited algorithmic baseline rather than the tool’s intendeduse accuracy. CalECG is intended primarily as an annotation environment, in which the built-in algorithm provides an initial delineation that a human inspects and corrects; we report its output without any such correction. They remain informative nonetheless: among the points it extracts, CalECG performs reasonably on the sinus-dominated LUDB but markedly worse on the arrhythmic Zhejiang, in both point-wise sensitivity (Table V) and averaged interval error (Table S10). This indicates that the initial algorithmic delineation would require substantially more manual correction under arrhythmia than in sinus rhythm.

## XI. MODEL TRAINING CONFIGURATIONS

This section details the semi-supervised and self-supervised configurations used in our experiments. For semi-supervised learning, we adopt the Mean Teacher (MT) framework (Table S4). For self-supervised pretraining, we evaluate three algorithms, MAE, MoCo, and MERL; their optimization and architecture-specific hyperparameters are given in Tables S5 and S6, respectively, following each original work adapted to the ECG domain.

TABLE S4  
SEMI-SUPERVISED LEARNING CONFIGURATION.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Framework</td><td>Mean Teacher (MT)</td></tr><tr><td>Teacher EMA decay</td><td>0.99</td></tr><tr><td>Consistency loss weight</td><td>1.0</td></tr><tr><td>Weak augmentation</td><td>Random resized cropping</td></tr><tr><td>Strong augmentation</td><td>RandAugment(N = 3, p = 0.5)</td></tr><tr><td rowspan="4">Candidate transforms</td><td>Powerline noise</td></tr><tr><td>Sine-wave noise</td></tr><tr><td>Amplitude scaling</td></tr><tr><td>White noise</td></tr></table>

N: number of transforms, p: probability

TABLE S5  
OPTIMIZATION HYPERPARAMETERS FOR SELF-SUPERVISED PRETRAINING: MAE, MOCO, AND MERL. ALL ALGORITHMS ARE TRAINED WITH THE ADAMW OPTIMIZER AND COSINE LEARNING RATE SCHEDULING.
<table><tr><td>Hyperparameter</td><td>MAE</td><td>MoCo</td><td>MERL</td></tr><tr><td>Epochs</td><td>800</td><td>300</td><td>50</td></tr><tr><td>Warm-up epochs</td><td>40</td><td>40</td><td>5</td></tr><tr><td>Batch size</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>Learning rate</td><td>1.2e-3</td><td>1.2e-3</td><td>2.0e-4</td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.1</td><td>1e-5</td></tr><tr><td> $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9, 0.95)</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr></table>

## XII. DETAILED QUANTITATIVE RESULTS

## A. Statistical Analysis and Seed Robustness

For metrics computed from per-ECG results (mIoU, averaged interval error, averaged point-wise sensitivity), we assess significance by a paired bootstrap over ECG records. For the comparison of pretraining objectives and average ranks, we use paired Wilcoxon signed-rank tests, pairing on dataset×finetuning-configuration blocks and on metric×dataset conditions, respectively. All p-values are Holm-corrected over the corresponding family of comparisons.

TABLE S6  
MODEL ARCHITECTURE AND ALGORITHM-SPECIFIC HYPERPARAMETERS FOR SELF-SUPERVISED PRETRAINING.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>MAE</td><td>0.75</td></tr><tr><td>Masking ratio Decoder embedding dimension Decoder depth Decoder attention heads</td><td>96 4 3</td></tr><tr><td>MoCo Projection hidden dimension</td><td>4096 256</td></tr><tr><td>Projection output dimension Predictor hidden dimension Momentum schedule Momentum coefficient (start → end) Temperature</td><td>4096 cosine 0.99 → 1.00 0.2</td></tr><tr><td>MERL Text encoder</td><td>MedCPT</td></tr><tr><td>ECG projection hidden dimension ECG projection output dimension Uni-modal alignment dropout Temperature</td><td>256 256 0.1 0.07</td></tr></table>

Tables III and IV in the main paper report results of a single seed. Repeating the same procedure independently for three seeds, both fine-tuning configurations remain significantly superior in 94 of 96 combinations for Table III (2 configurations×4 datasets×3 metrics×4 tools) and in 38 of 40 for Table IV (2 configurations×4 metrics×5 tools), and mIoU is significantly superior in every combination and every seed. All exceptions involve Prominence: averaged point-wise sensitivity on LUDB for DL DELINEATOR (sup) and averaged interval error on RDB for DL DELINEATOR (semi) in Table III, and QT interval error on mECGDB for both configurations in Table IV. The margins are small. For LUDB sensitivity, DL DELINEATOR (sup) narrows to +0.5 %p in its weakest seed, the only seed that does not reach significance. For mECGDB QT error, DL DELINEATOR (semi) drops to -0.1 ms in its single non-significant seed, while DL DELINEATOR (sup) stays between +1.9 and +2.6 ms without reaching significance in any seed. Only the averaged interval error on RDB reverses direction, with Prominence more accurate by 0.6 ms in one of the three seeds (+1.3 and +0.2 ms in the other two).

## B. Computational Cost

We measure the computation time of the DL DELINEATOR and four rule-based methods (NeuroKit2 (CWT), NeuroKit2 (DWT), ECGdeli, and Prominence) on LUDB, running each method on a single CPU (Intel i7-10750H) with one sample at a time. For every method, we record the time taken to produce the P, QRS, and T fiducial points, and report the mean and standard deviation (SD) over the test set.

The DL DELINEATOR is the fastest of the five, runs at a mean of 17.1 ms per sample (Table S7), ahead of Prominence (20.0 ms) and several times faster than the two NeuroKit2 variants and ECGdeli (56.3, 63.6, and 141.0 ms). Its runtime is also the most stable (SD 1.9 ms versus 6 ms or more for the others), as a learned model performs a fixed computation regardless of the signal, whereas the search in the rulebased methods scales with waveform complexity. NeuroKit2 additionally fails to return an output on 7 samples, which are excluded from its statistics.

TABLE S7  
COMPUTATION TIME ANALYSIS ON LUDB, REPORTED AS A MEAN ± STANDARD DEVIATION. THE BEST VALUE IS BOLDED.
<table><tr><td>Method</td><td>Time (ms) ↓</td></tr><tr><td>NeuroKit2 (CWT)</td><td> $5 6 . 3 { \pm } 1 1 . 9 $ </td></tr><tr><td>NeuroKit2 (DWT)</td><td> $6 3 . 6 { \pm } 1 1 . 1$ </td></tr><tr><td>ECGdeli</td><td>141.0±57.5</td></tr><tr><td>Prominence</td><td>20.0±6.0</td></tr><tr><td>DL DELINEATOR</td><td>17.1±1.9</td></tr></table>

## C. Full Benchmarking Results

We present the full benchmarking results summarized in the main paper. Tables S8 and S9 report the averaged interval error and averaged point-wise sensitivity for each combination of pretraining (none, MAE, MoCo, MERL) and finetuning (supervised, semi-supervised), where MAE pretraining generally performs best. Table S10 compares the DL DELIN-EATOR with CalECG on averaged interval error (Internal, LUDB, Zhejiang), reported separately because CalECG was configured to output only the fiducial points required for interval computation. Table S11 additionally compares the DL DELINEATOR with existing open-source tools by averaged point-wise sensitivity and PPV at the 40 and 150 ms tolerances, and Table S13 reports the per-fiducial localization error $( \mu \pm \sigma )$ on the LUDB and Zhejiang external test sets. Table S14 presents the rhythm-stratified subgroup analysis on RDB, showing that the DL DELINEATOR maintains the smallest performance degradation from sinus to arrhythmia.

TABLE S8  
BENCHMARKING RESULTS OF LEARNING STRATEGIES FOR ECG DELINEATION. PER-DATASET ENTRIES ARE AVERAGED INTERVAL ERROR (MS), REPORTED AS MEAN ± STD OVER THREE SEEDS, WHERE INTERNAL DENOTES THE $Q T D B + I S P$ TEST SPLIT. THE BEST VALUE IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td>Pretraining</td><td>Fine-tuning</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>mECGDB</td><td>RDB</td></tr><tr><td rowspan="2"></td><td>Supervised</td><td> $1 6 . 2 \pm 0 . 3$ </td><td> $2 1 . 1 \pm 0 . 7$ </td><td> $3 3 . 1 \pm 2 . 0$ </td><td> $2 6 . 1 \pm 0 . 9$ </td><td> $2 3 . 4 \pm 0 . 3$ </td></tr><tr><td>Semi-supervised</td><td> $1 2 . 8 \pm 0 . 4$ </td><td> $2 0 . 8 \pm 2 . 4$ </td><td> $2 1 . 7 \pm 3 . 0$ </td><td> $1 6 . 2 \pm 0 . 6$ </td><td> $2 9 . 8 \pm 4 . 3$ </td></tr><tr><td rowspan="2">MAE</td><td>Supervised</td><td> ${ \bf 1 1 . 6 \pm 0 . 2 }$ </td><td> $1 9 . 2 \pm 1 . 7$ </td><td> $1 6 . 5 \pm 1 . 2$ </td><td> ${ \bf 1 5 . 0 \pm 0 . 2 }$ </td><td> ${ \bf 2 0 . 9 \pm 1 . 2 }$ </td></tr><tr><td>Semi-supervised</td><td> $1 1 . 7 \pm 0 . 3$ </td><td> $1 8 . 4 \pm 2 . 0$ </td><td> ${ \bf 1 5 . 9 \pm 0 . 6 }$ </td><td> $1 5 . 7 \pm 0 . 7$ </td><td> $2 2 . 7 \pm 1 . 0$ </td></tr><tr><td rowspan="2">MoCo</td><td>Supervised</td><td> $1 2 . 0 \pm 0 . 3$ </td><td> $2 1 . 3 \pm 2 . 5$ </td><td> $2 1 . 3 \pm 1 . 0$ </td><td> $1 5 . 2 \pm 0 . 8$ </td><td> $2 2 . 4 \pm 1 . 2$ </td></tr><tr><td>Semi-supervised</td><td> $1 1 . 8 \pm 0 . 3$ </td><td> ${ \bf 1 7 . 2 \pm 0 . 8 }$ </td><td> $1 6 . 9 \pm 0 . 8$ </td><td> $1 5 . 4 \pm 0 . 7$ </td><td> $2 2 . 3 \pm 0 . 5$ </td></tr><tr><td rowspan="2">MERL</td><td>Supervised</td><td> $1 5 . 3 \pm 0 . 2$ </td><td> $2 0 . 3 \pm 0 . 7$ </td><td> $3 4 . 6 \pm 2 . 5$ </td><td> $2 4 . 1 \pm 0 . 7$ </td><td> $2 3 . 3 \pm 0 . 4$ </td></tr><tr><td>Semi-supervised</td><td> $1 6 . 4 \pm 3 . 5$ </td><td> $5 0 . 3 \pm 1 . 4$ </td><td> $2 3 . 9 \pm 1 . 7$ </td><td> $1 6 . 0 \pm 0 . 9$ </td><td> $2 5 . 3 \pm 2 . 9$ </td></tr></table>

TABLE S9  
BENCHMARKING RESULTS OF LEARNING STRATEGIES FOR ECG DELINEATION. PER-DATASET ENTRIES ARE AVERAGED POINT-WISE SENSITIVITY (%), REPORTED AS MEAN ± STD OVER THREE SEEDS, WHERE INTERNAL DENOTES THE $Q T D B + I S P$ TEST SPLIT. THE BEST VALUE IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td>Pretraining</td><td>Fine-tuning</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td><td>RDB</td></tr><tr><td></td><td>Supervised Semi-supervised</td><td> $9 1 . 3 \pm 0 . 2$   $9 5 . 4 \pm 0 . 2$ </td><td> $8 2 . 5 \pm 1 . 2$   $8 6 . 5 \pm 7 . 0$ </td><td> $6 6 . 5 \pm 2 . 6$   $7 6 . 6 \pm 6 . 2$ </td><td> $8 7 . 1 \pm 0 . 7$   $9 0 . 8 \pm 0 . 6$ </td></tr><tr><td>MAE</td><td>Supervised</td><td> $9 6 . 1 \pm 0 . 2$ </td><td> $8 6 . 4 \pm 3 . 0$ </td><td> $8 5 . 4 \pm 2 . 2$ </td><td> ${ \bf 9 3 . 3 \pm 0 . 6 }$ </td></tr><tr><td>MoCo</td><td>Semi-supervised Supervised</td><td> ${ \bf 9 6 . 2 \pm 0 . 3 }$   $9 5 . 7 \pm 0 . 1$ </td><td> $8 9 . 2 \pm 3 . 3$   $8 5 . 1 \pm 3 . 5$ </td><td> ${ \bf 8 5 . 9 \pm 1 . 5 }$   $7 7 . 5 \pm 4 . 2$ </td><td> $9 2 . 6 \pm 0 . 3$   $9 2 . 4 \pm 1 . 0$ </td></tr><tr><td>MERL</td><td>Semi-supervised Supervised</td><td> $9 6 . 1 \pm 0 . 2$   $9 2 . 4 \pm 0 . 2$ </td><td> ${ \bf 9 0 . 3 \pm 3 . 5 }$   $8 5 . 8 \pm 0 . 8$ </td><td> $8 4 . 5 \pm 0 . 7$   $6 5 . 9 \pm 0 . 3$ </td><td> $9 2 . 9 \pm 1 . 0$   $8 8 . 6 \pm 0 . 1$ </td></tr></table>

TABLE S10
<table><tr><td></td><td colspan="3">Averaged interval error (ms) ↓</td></tr><tr><td>Method</td><td>Internal</td><td>LUDB</td><td>Zhejiang</td></tr><tr><td>CalECG</td><td>37.9</td><td>20.1</td><td>58.2</td></tr><tr><td>DL DELINEATOR (sup)</td><td>11.5</td><td>19.0</td><td>16.3</td></tr><tr><td>DL DELINEATOR (semi)</td><td>12.0</td><td>16.8</td><td>15.3</td></tr></table>

COMPARISON OF THE DL DELINEATOR WITH CALECG ON AVERAGED INTERVAL ERROR. THE BEST VALUE IS IN BOLD AND THE SECOND BEST IS UNDERLINED.

COMPARISON OF THE DL DELINEATOR WITH OPEN-SOURCE DELINEATION TOOLS, EVALUATED BY AVERAGED POINT-WISE TOTAL SENSITIVITY AND POSITIVE PREDICTIVE VALUE (PPV) (%) AT TOLERANCES OF 40 AND 150 MS. THE BEST VALUE IN EACH COLUMN IS IN BOLD.  
TABLE S11
<table><tr><td></td><td></td><td colspan="2">Sensitivity (%)</td><td colspan="2">PPV (%)</td></tr><tr><td>Dataset</td><td>Method</td><td>40 ms</td><td>150ms</td><td>40 ms</td><td>150 ms</td></tr><tr><td rowspan="6">Internal</td><td>NeuroKit2 (CWT)</td><td>59.4</td><td>73.7</td><td>66.1</td><td>82.1</td></tr><tr><td>NeuroKit2 (DWT)</td><td>60.3</td><td>88.4</td><td>60.5</td><td>88.0</td></tr><tr><td>ECGdeli</td><td>82.4</td><td>97.0</td><td>77.3</td><td>90.8</td></tr><tr><td>Prominence</td><td>78.1</td><td>95.9</td><td>74.2</td><td>91.1</td></tr><tr><td>DL DELINEATOR (sup)</td><td>96.1</td><td>99.0</td><td>95.6</td><td>98.6</td></tr><tr><td>DL DELINEATOR (semi)</td><td>96.1</td><td>99.1</td><td>95.4</td><td>98.3</td></tr><tr><td rowspan="6">LUDB</td><td>NeuroKit2 (CWT)</td><td>59.9</td><td>68.2</td><td>68.5</td><td>77.9</td></tr><tr><td>NeuroKit2 (DWT)</td><td>62.2</td><td>88.2</td><td>62.1</td><td>87.2</td></tr><tr><td>ECGdeli</td><td>80.6</td><td>97.9</td><td>71.8</td><td>86.7</td></tr><tr><td>Prominence</td><td>82.9</td><td>97.3</td><td>77.7</td><td>90.9</td></tr><tr><td>DL DELINEATOR (sup)</td><td>86.6</td><td>94.5</td><td>90.4</td><td>98.8</td></tr><tr><td>DL DELINEATOR (semi)</td><td>91.1</td><td>98.7</td><td>91.4</td><td>99.0</td></tr><tr><td rowspan="7">Zhejiang</td><td>NeuroKit2 (CWT)</td><td>15.7</td><td>21.9</td><td>52.2</td><td>74.0</td></tr><tr><td>NeuroKit2 (DWT)</td><td>17.8</td><td>25.2</td><td>54.3</td><td>79.6</td></tr><tr><td>ECGdeli</td><td>56.3</td><td>83.5</td><td>48.2</td><td>71.1</td></tr><tr><td>Prominence</td><td>54.6</td><td>74.4</td><td>57.9</td><td>81.9</td></tr><tr><td>DL DELINEATOR (sup)</td><td>82.9</td><td>92.2</td><td>88.1</td><td>97.7</td></tr><tr><td>DL DELINEATOR (semi)</td><td>86.0</td><td>94.9</td><td>88.3</td><td>97.4</td></tr><tr><td>NeuroKit2 (CWT)</td><td>54.9</td><td>66.1</td><td>60.8</td><td>73.6</td></tr><tr><td rowspan="6">RDB</td><td>NeuroKit2 (DWT)</td><td>58.5</td><td>81.3</td><td>54.8</td><td>77.2</td></tr><tr><td>ECGdeli</td><td>76.4</td><td>95.3</td><td>63.0</td><td>78.4</td></tr><tr><td>Prominence</td><td>73.7</td><td>87.8</td><td>66.9</td><td>79.9</td></tr><tr><td>DL DELINEATOR (sup)</td><td>92.6</td><td>97.5</td><td>88.9</td><td>93.7</td></tr><tr><td>DL DELINEATOR (semi)</td><td></td><td>98.3</td><td>87.7</td><td>93.3</td></tr><tr><td></td><td>92.5</td><td></td><td></td><td></td></tr></table>

TABLE S12  
COMPARISON OF THE DL DELINEATOR ON EXTERNAL TEST SETS (LUDB AND Zhejiang) WITH OPEN-SOURCE AND COMMERCIAL DELINEATION TOOLS, EVALUATED BY POINT-WISE POSITIVE PREDICTIVE VALUE (PPV, %) PER FIDUCIAL POINT. "TOTAL" IS THE AVERAGE OVER THE AVAILABLE ONSET/OFFSET POINTS. THE BEST VALUE IS IN BOLD.
<table><tr><td>Dataset</td><td>Method</td><td> $\mathrm { P _ { o n } }$ </td><td> $\mathrm { P _ { o f f } }$ </td><td> $\mathrm { Q R S _ { o n } }$ </td><td> $\mathrm { Q R S _ { o f f } }$ </td><td> $\mathrm { T _ { o n } }$ </td><td> $\mathrm { T _ { o f f } }$ </td><td>Total</td></tr><tr><td rowspan="7"></td><td>NeuroKit2 2 (CWT)</td><td>43.5</td><td>42.8</td><td>87.4</td><td>93.8</td><td>67.5</td><td>76.0</td><td>68.5</td></tr><tr><td>NeuroKit2 2 (DWT)</td><td>62.4</td><td>53.6</td><td>75.1</td><td>92.3</td><td>32.7</td><td>56.5</td><td>62.1</td></tr><tr><td>ECGdeli</td><td>55.1</td><td>66.9</td><td>96.8</td><td>84.6</td><td>63.2</td><td>64.1</td><td>71.8</td></tr><tr><td>CalECG</td><td>83.8</td><td></td><td>95.0</td><td>96.9</td><td></td><td>84.5</td><td>90.1</td></tr><tr><td>Prominence</td><td>64.6</td><td>65.9</td><td>91.0</td><td>84.9</td><td>79.0</td><td>80.7</td><td>77.7</td></tr><tr><td>DL DELINEATOR (sup)</td><td>88.1</td><td>89.1</td><td>99.2</td><td>99.0</td><td>81.4</td><td>85.4</td><td>90.4</td></tr><tr><td>DL DELINEATOR (semi)</td><td>92.8</td><td>89.1</td><td>99.2</td><td>99.0</td><td>81.3</td><td>86.8</td><td>91.4</td></tr><tr><td rowspan="7">Zhejiang</td><td>NeuroKit2 2 (CWT)</td><td>34.1</td><td>30.5</td><td>77.2</td><td>83.3</td><td>41.9</td><td>46.3</td><td>52.2</td></tr><tr><td>NeuroKit2 (DWT)</td><td>44.2</td><td>43.7</td><td>53.8</td><td>85.8</td><td>48.1</td><td>50.2</td><td>54.3</td></tr><tr><td>ECGdeli</td><td>33.5</td><td>21.7</td><td>79.3</td><td>64.2</td><td>45.7</td><td>44.6</td><td>48.2</td></tr><tr><td>CalECG</td><td>30.5</td><td></td><td>52.8</td><td>89.1</td><td></td><td>74.0</td><td>61.6</td></tr><tr><td>Prominence</td><td>47.3</td><td>46.8</td><td>77.2</td><td>75.4</td><td>52.1</td><td>48.7</td><td>57.9</td></tr><tr><td>DL DELINEATOR (sup)</td><td>92.6</td><td>89.6</td><td>95.8</td><td>97.1</td><td>70.7</td><td>82.7</td><td>88.1</td></tr><tr><td>DL DELINEATOR (semi)</td><td>91.1</td><td>87.6</td><td>97.3</td><td>97.3</td><td>70.2</td><td>86.2</td><td>88.3</td></tr></table>

TABLE S13  
COMPARISON OF THE DL DELINEATOR ON LUDB AND Zhejiang EXTERNAL TEST SETS WITH WIDELY USED OPEN-SOURCE AND COMMERCIAL DELINEATION TOOLS, EVALUATED BY POINT-WISE LOCALIZATION ERROR (MS) PER FIDUCIAL POINT AT THE 40 MS TOLERANCE. THE SIGNED MEAN ERROR AND ITS STANDARD DEVIATION ARE REPORTED.
<table><tr><td>Dataset</td><td>Method</td><td> $\mathrm { P _ { o n } }$ </td><td> $\mathrm { P _ { o f f } }$ </td><td> $\mathrm { Q R S _ { o n } }$ </td><td> $\mathrm { Q R S _ { o f f } }$ </td><td> $\mathrm { T _ { o n } }$ </td><td> $\mathrm { T _ { o f f } }$ </td></tr><tr><td rowspan="7">LUDB</td><td>NeuroKit2 (CWT)</td><td> $- 7 . 6 { \pm } 9 . 4 $ </td><td> $0 . 1 { \pm } 9 . 5 $ </td><td> $- 8 . 8 { \pm } 1 2 . 3$ </td><td> $- 1 2 . 0 { \pm } 1 3 . 3 $ </td><td> $5 . 0 { \pm } 1 7 . 4 $ </td><td> $6 . 1 { \pm } 1 0 . 9$ </td></tr><tr><td>NeuroKit2 (DWT)</td><td> $3 . 5 { \pm } 1 7 . 6 $ </td><td> $- 3 . 0 { \pm } 1 7 . 3 $ </td><td>-3.6±13.7</td><td> $- 3 . 1 { \pm } 1 3 . 4 $ </td><td> $1 7 . 7 { \pm } 2 0 . 1 $ </td><td> $- 1 0 . 0 { \pm } 1 3 . 2 $ </td></tr><tr><td>ECGdeli</td><td> $- 1 3 . 0 { \pm } 1 1 . 3$ </td><td>16.8±11.4</td><td>-7.2±13.9</td><td>16.2±16.0</td><td>-10.4±19.9</td><td>4.5±15.9</td></tr><tr><td>CalECG</td><td>0.3±11.5</td><td></td><td>8.1±13.5</td><td>-2.3±11.8</td><td></td><td>5.3±12.1</td></tr><tr><td>Prominence</td><td>1.9±12.8</td><td>3.1±13.9</td><td>-0.0±15.9</td><td>-11.4±18.1</td><td>2.8±18.2</td><td>13.2±14.8</td></tr><tr><td>DL DELINEATOR (sup)</td><td>-5.4±14.3</td><td>7.6±15.1</td><td>-8.4±10.1</td><td>10.4±11.4</td><td>4.0±20.1</td><td>12.5±14.1</td></tr><tr><td>DL DELINEATOR (semi)</td><td>-6.5±13.1</td><td>10.0±14.6</td><td>-8.2±10.0</td><td>9.3±12.1</td><td>1.3±20.4</td><td>13.0±14.5</td></tr><tr><td rowspan="7">Zhejiang</td><td>NeuroKit2 (CWT)</td><td>6.3±14.1</td><td>-13.4±12.8</td><td>-6.8±15.4</td><td>-12.5±19.7</td><td>2.9±19.3</td><td>-4.6±16.2</td></tr><tr><td>NeuroKit2 (DWT)</td><td>2.7±14.4</td><td>-2.9±16.1</td><td>-5.0±16.4</td><td>-8.2±14.5</td><td>7.6±20.5</td><td>-7.1±14.5</td></tr><tr><td>ECGdeli</td><td>-3.9±12.0</td><td>6.4±17.1</td><td>-1.5±18.2</td><td>13.3±18.0</td><td>1.4±23.3</td><td>-4.7±19.5</td></tr><tr><td>CalECG</td><td>2.8±16.1</td><td></td><td>25.7±10.7</td><td>-7.2±13.6</td><td></td><td>-2.4±15.1</td></tr><tr><td>Prominence</td><td>4.8±13.8</td><td>-4.5±15.4</td><td>2.1±18.9</td><td>-10.3±17.4</td><td>3.1±20.3</td><td>-0.3±19.3</td></tr><tr><td>DL DELINEATOR (sup)</td><td>-1.1±11.2</td><td>-5.3±15.7</td><td>-4.6±12.9</td><td>5.5±13.8</td><td>8.2±20.1</td><td>1.4±17.5</td></tr><tr><td>DL DELINEATOR (semi)</td><td>-0.7±10.6</td><td>-7.1±15.3</td><td>-4.5±13.1</td><td>4.1±13.8</td><td>8.9±19.9</td><td>-2.3±17.6</td></tr></table>

TABLE S14  
RHYTHM-STRATIFIED SUBGROUP ANALYSIS ON THE RDB, COMPARING THE DL DELINEATOR WITH THE TWO STRONGEST TOOLS (PROMINENCE, ECGDELI). RHYTHMS ARE GROUPED INTO SINUS (SR, SB, ST, SI) AND ARRHYTHMIA (AF, AFIB, AT, SVT). FOR EACH METRIC, WE REPORT THE GROUP MEAN PERFORMANCE AND ITS RELATIVE DEGRADATION (DEG.%) FROM SINUS TO ARRHYTHMIA. THE BEST VALUE IS IN BOLD.
<table><tr><td></td><td></td><td colspan="5">Sinus group</td><td colspan="5">Arrhythmia group</td><td></td></tr><tr><td>Metric</td><td>Method</td><td>SR</td><td>SB</td><td>ST</td><td>SI</td><td>Avg.</td><td>AF</td><td>AFIB</td><td>AT</td><td>SVT</td><td>Avg.</td><td>Deg.%</td></tr><tr><td></td><td>ECGdeli</td><td>67.8</td><td>67.2</td><td>63.2</td><td>66.9</td><td>66.3</td><td>38.8</td><td>42.1</td><td>45.3</td><td>28.8</td><td>38.7</td><td>41.5</td></tr><tr><td>mIoU (%) ↑</td><td>Prominence</td><td>67.2</td><td>65.4</td><td>58.9</td><td>66.3</td><td>64.4</td><td>33.1</td><td>38.1</td><td>38.6</td><td>22.8</td><td>33.2</td><td>48.6</td></tr><tr><td></td><td>DL DELINEATOR (sup)</td><td>76.4</td><td>77.0</td><td>75.0</td><td>75.9</td><td>76.1</td><td>61.1</td><td>67.6</td><td>63.5</td><td>67.4</td><td>64.9</td><td>14.7</td></tr><tr><td></td><td>DL DELINEATOR (semi)</td><td>77.3</td><td>78.0</td><td>74.2</td><td>76.9</td><td>76.6</td><td>59.4</td><td>67.8</td><td>63.4</td><td>66.3</td><td>64.2</td><td>16.2</td></tr><tr><td></td><td>ECGdeli</td><td>28.3</td><td>30.1</td><td>28.4</td><td>28.6</td><td>28.9</td><td>34.1</td><td>29.8</td><td>34.4</td><td>62.0</td><td>40.1</td><td>65.8</td></tr><tr><td>Averaged interval error (ms) ↓</td><td>Prominence</td><td>17.4</td><td>20.1</td><td>18.0</td><td>17.3</td><td>18.2</td><td>35.7</td><td>25.7</td><td>34.9</td><td>56.4</td><td>38.2</td><td>135.6</td></tr><tr><td></td><td>DL DELINEATOR (sup)</td><td>22.0</td><td>21.7</td><td>18.1</td><td>20.5</td><td>20.6</td><td>35.0</td><td>28.4</td><td>22.9</td><td>27.7</td><td>28.5</td><td>71.7</td></tr><tr><td></td><td>DL DELINEATOR (semi)</td><td>20.3</td><td>20.1</td><td>18.7</td><td>18.4</td><td>19.4</td><td>38.2</td><td>28.5</td><td>24.6</td><td>29.5</td><td>30.2</td><td>92.6</td></tr><tr><td></td><td>ECGdeli</td><td>82.3</td><td>79.0</td><td>80.0</td><td>83.2</td><td>81.1</td><td>70.2</td><td>74.1</td><td>69.1</td><td>55.7</td><td>67.3</td><td>17.1</td></tr><tr><td>Average point-wise sensitivity (%) ↑</td><td>Prominence</td><td>86.1</td><td>82.4</td><td>79.9</td><td>85.7</td><td>83.5</td><td>59.7</td><td>69.1</td><td>57.9</td><td>46.3</td><td>58.2</td><td>30.3</td></tr><tr><td></td><td>DL DELINEATOR (sup)</td><td>97.2</td><td>95.7</td><td>95.7</td><td>96.9</td><td>96.4</td><td>84.4</td><td>86.4</td><td>86.5</td><td>63.0</td><td>80.1</td><td>16.9</td></tr><tr><td></td><td>DL DELINEATOR (semi)</td><td>97.5</td><td>96.1</td><td>95.0</td><td>97.4</td><td>96.5</td><td>81.8</td><td>86.3</td><td>88.3</td><td>65.1</td><td>80.4</td><td>16.7</td></tr></table>

SR: sinus rhythm; SB: sinus bradycardia; ST: sinus tachycardia; SI: sinus irregularity;  
AF: atrial flutter; AFIB: atrial fibrillation; AT: atrial tachycardia; SVT: supraventricular tachycardia

XIII. QUALITATIVE RESULTS  
![](images/269b8b1274f74c3003c766c3803b56e4cd6321f10d18554a06834cfe2cab81dd.jpg)  
Fig. S1. Qualitative P/QRS/T delineation on a representative 2.5 s Lead II segment from LUDB. Rows are grouped into the reference annotation (GT), rule-based tools, and the DL DELINEATOR; shaded bands mark each method’s detected P (blue), QRS (orange), and T (green) waves on the same input signal.

![](images/c3c2048494dc00e3d7b4c11aaf110755af699377b71aa7223922c3ba9fc31d4b.jpg)  
Fig. S2. Qualitative P/QRS/T delineation on a representative 2.5 s Lead II segment from Zhejiang. Rows are grouped into the reference annotation (GT), rule-based tools, and the DL DELINEATOR; shaded bands mark each method’s detected P (blue), QRS (orange), and T (green) waves on the same input signal.

![](images/72d9915992e5b6d079d6439c24e86a317648894b21cfbd6175603c3151eab312.jpg)  
Fig. S3. Qualitative P/QRS/T delineation on a representative 2.5 s Lead II segment from RDB. Rows are grouped into the reference annotation (GT), rule-based tools, and the DL DELINEATOR; shaded bands mark each method’s detected P (blue), QRS (orange), and T (green) waves on the same input signal. SR, sinus rhythm; AFIB, atrial fibrillation; SVT, supraventricular tachycardia.