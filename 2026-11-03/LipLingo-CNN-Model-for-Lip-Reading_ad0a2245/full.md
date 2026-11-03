# LipLingo: CNN Model for Lip Reading

Mahmoud S. Mayaleh<sup>1</sup>, Liban Ali Achkir<sup>2</sup>, Nijat Mehdiyev<sup>2</sup>, Ahmet Ozdil<sup>¨</sup> <sup>2</sup>

<sup>1</sup>National Conservatory of Arts and Crafts (CNAM), Paris, France <sup>2</sup>Ostim Technical University, Ankara, Turkiye ¨

mahmoud.mayaleh@lecnam.net, liban.ali.achkir@ostimteknik.edu.tr, 240910604@ostimteknik.edu.tr, ahmet.ozdil@ostimteknik.edu.tr

## Abstract

Lip reading plays a crucial role in applications such as speech recognition, assistive technologies for the hearing-impaired, and biometric authentication. However, performance often degrades when the speaker varies and when certain speech sounds are visually similar. To address these challenges, we propose LipLingo, a robust lip-reading system based on the LipNet architecture, enhanced with a standardized preprocessing pipeline for consistent mouth-region representation, a rotational validation strategy for improved generalization to unseen speakers, and a flexible similarity evaluation for more reliable sentence-level assessment. The model architecture combines 3D convolutional layers for spatio-temporal feature extraction and bidirectional recurrent layers for sequence modeling, optimized using the Adam optimizer and Connectionist Temporal Classification (CTC) loss. On the GRID dataset, LipLingo achieved 95.0% character-leve accuracy (CER 4.99%) and 87.2% sentence-level accuracy. At the corpus level, the system obtained a (WER 13.1%), outperforming the baseline LipNet and demonstrating competitive performance relative to recent 3D-ResNet + Bi-GRU + MixUp approaches. These results provide insights into strategies that enhance lip-reading robustness and highlight directions for future work toward real-world deployment.

## 1 Introduction

## 1.1 General Overview of Lip Reading

Human language understanding relies on a combination of auditory and visual cues. While spoken language is primarily transmitted via sound waves, visual information—particularly lip movements—provides complementary data that improves comprehension. Lip reading, also referred to as speech reading, is the process of interpreting spoken words by observing these visual cues. It has several practical applications, including:

• Speech Recognition: Enhances transcription accuracy when integrated with audio-based systems.

• Assistive Technologies: Enables hearing-impaired individuals to better understand spoken communication.

• Biometric Authentication: Utilizes unique lip movement patterns for speaker verification.

Despite recent advances in machine learning, lip reading systems face challenges due to variations in facial features, illumination conditions, and the visual similarity of certain phonemes. These factors complicate accurate interpretation and limit performance in real-world scenarios.

## 1.2 A Deep Dive into Improved Speech Reading

This study explores deep learning-based approaches to automatic lip reading, aiming to achieve higher accuracy and improved generalization compared with existing pre-trained systems. Our approach is based on the LipNet architecture [3], which has shown strong performance on sentence-level lip reading tasks.

Recent advances in lip reading are based on modern sequence models and pre-training. Transformer/Conformer encoders now have very low error rates on large, in-the-wild benchmarks such as LRS3 [5], and self-supervised methods (e.g., AV-HuBERT) learn strong visual speech features with far less labeled data [23]. Landmarkguided transformers and efficient windowed-attention encoders further improve generalization to new speakers and reduce computational effort [25, 20]. Unified audio–visual models can also run in audio-only[12], videoonly or fused modes, improving robustness when one modality is unreliable.

Most of the above systems rely on very large, diverse corpora and significant computational power; they are not directly comparable on GRID’s controlled grammar and scale [9]. Therefore, we use LipNet as a reproducible, end-to-end baseline on GRID so that we can isolate the effects of our contributions without confounding factors from large-scale pre-training, while discussing broader trends in the literature.

## 1.3 Our Specific Objectives

This work is guided by four primary objectives:

1. Evaluate the potential of the LipNet architecture for customized training on our dataset.

2. Develop a tailored model that integrates additional preprocessing and architectural enhancements.

3. Conduct rigorous performance evaluation and compare against baseline and state-of-the-art models.

4. Identify key factors that influence accuracy and robustness to guide future improvements.

While LipNet and similar architectures achieve good performance under constrained settings, their generalization to diverse speakers and environments remains limited. This study focuses on addressing these challenges by proposing improvements that enhance model robustness and practical usability, which is crucial for assistive technologies and biometric applications.

## 2 Literature Review

## 2.1 Overview of Related Work in Lip Reading

Lip reading, also referred to as speech reading, involves decoding spoken language by interpreting the visible movements of a speaker’s lips and face. The advent of deep learning has significantly improved the ability of models to learn complex spatio-temporal representations. Earlier research focused primarily on recognizing isolated words or limited vocabularies. Recent efforts, however, have shifted toward sentence-level transcription, speaker-independent recognition, and end-to-end architectures that minimize the need for manual feature engineering.

## 2.2 Existing Models, Techniques, and Datasets

LipNet. Assael et al. [3] introduced LipNet, a pioneering end-to-end model for sentence-level lip reading that set a new benchmark on the GRID dataset. LipNet combines spatio-temporal convolutional layers to capture both spatial features (e.g., lip shape) and temporal dynamics (e.g., mouth movements across frames). These features are then processed by Long Short-Term Memory (LSTM) layers [13], which model the sequential nature of speech. The use of Connectionist Temporal Classification (CTC) loss [11] enables alignment-free mapping between video frames and character sequences. LipNet achieved significant performance gains over previous approaches and inspired numerous follow-up studies.

3D Convolutional Neural Networks. Exarchos et al. [10] proposed a 3D CNN architecture capable of modeling both spatial and temporal dimensions simultaneously. Unlike 2D CNNs, which process individual frames independently, 3D convolutions operate across sequences of frames, enabling the network to capture motion dynamics inherent to speech. Some variants further incorporate recurrent layers, such as LSTMs or Gated Recurrent Units (GRUs) [6], to enhance temporal modeling. While powerful, 3D CNNs are computationally demanding and prone to overfitting, particularly on smaller datasets.

Viseme-Based Recognition with Hidden Markov Models. Earlier approaches often relied on hand-crafted features and statistical models. Lu et al. [16] and Baum and Petrie [4] developed systems for classifying visemes, which are visually distinct clusters of phonemes [21]. Hidden Markov Models (HMMs) modeled the probabilistic transitions between visemes over time. Although viseme-based recognition simplified classification, it lacked the fine-grained detail required for accurate word recognition in continuous speech.

Deep Autoencoders. Zhou and Paffenroth [27] investigated stacked denoising autoencoders to learn hierarchical feature representations without requiring labeled data. These models compress and reconstruct input video sequences, learning robust encodings even in noisy conditions. Unsupervised pre-training can be particularly valuable for low-resource datasets, although autoencoders typically require additional sequence-labeling components for complete speech recognition pipelines.

While LipNet set an important benchmark, it also has clear limitations. Its training relies on the GRID dataset’s constrained grammar and vocabulary, which makes results less representative of natural speech. In addition, LipNet struggles with visually similar phonemes (e.g., ’p’ vs. ’b’) and has limited robustness to speaker variation or environmental changes. Later architectures such as 3D CNNs and ResNet–BiGRU hybrids improve spatio-temporal feature capture, but at the cost of higher computational complexity and larger datasets. Transformer-based methods (e.g., Conformer VSR, LipFormer) show further accuracy gains, yet depend on large-scale pretraining and benchmark datasets (LRS2 [8], LRS3), which are not always practical. These largescale datasets contain tens to hundreds of hours of unconstrained speech from diverse speakers, but they demand substantial computational resources (multi-GPU training for weeks) that are not accessible in many research or applied settings, they introduce licensing and storage constraints, since distributing and processing hundreds of gigabytes is not feasible for all institutions. For these reasons, when focusing on controlled datasets such as GRID, using transformer-based systems pre-trained on LRS2/LRS3 may be excessive or misaligned with the study goals. LipLingo addresses part of the gap by proposing a standardized preprocessing pipeline that reduces speaker-specific variability and a rotational validation strategy to improve generalization on GRID, while keeping the model lightweight.

Datasets. The GRID audiovisual sentence corpus remains one of the most widely used benchmarks for lip reading research. Its structured sentence format and consistent recording conditions support reproducible experiments, yet its limited speaker diversity and restricted vocabulary hinder generalization to natural, unconstrained speech.

Beyond GRID, the field now evaluates models on more diverse datasets such as LRW (word-level, 500 classes, short clips) [7], LRW-1000 (large-scale, naturally distributed, Mandarin) [26], and LRS2/LRS3 (sentence-level, unconstrained speech from BBC/TED). These benchmark datasets introduce challenges such as large vocabulary, varying sentence structures, and uncontrolled recording conditions. They highlight the gap between controlled benchmarks like GRID and real-world deployment. In this study, we focus on GRID for reproducibility and fair comparison with LipNet, but we acknowledge the importance of testing on larger benchmark datasets in future work.

Table 1: Sentence-level lip-reading models and their reported performance (WER-focused).
<table><tr><td>Model</td><td>Dataset</td><td>Performance</td></tr><tr><td>LipNet (2016) 3D ResNet + Bi-GRU + MixUp (2024)</td><td>GRID GRID</td><td>WER 11.4% WER 11.3%</td></tr><tr><td>Conformer VSR (2023)</td><td>LRS3 (TED)</td><td>WER 12.8%</td></tr><tr><td>SwinLip (2025)</td><td>LRS2</td><td>WER 37.01%</td></tr><tr><td>AV-HuBERT (2022)</td><td>LRS2 / LRS3</td><td>WER 26.9%</td></tr><tr><td>LipLingo (ours)</td><td>GRID</td><td>WER 13.1%</td></tr></table>

## 2.3 Gaps in Current Research

Despite substantial progress, several challenges persist:

• Vocabulary Scale and Speaker Independence: Many models achieve high accuracy on constrained datasets but degrade when applied to unconstrained speech or unseen speakers. Achieving true speaker independence remains an open problem [18].

• Environmental Robustness: Variations in lighting, occlusions (e.g., facial hair, masks), and speaking styles significantly affect performance. Data augmentation and multimodal sensor fusion have been proposed as mitigations but require further validation.

• Computational accessibility: Transformer-based and self-supervised models achieve state-of-the-art results, but they require large-scale datasets (LRS2, LRS3, LRW-1000) and weeks of multi-GPU training, making them impractical for smaller labs or real-world deployments where compute and data are limited.

• Interpretability and Transparency: Deep learning architectures often function as black boxes, providing limited insight into decision-making. Incorporating domain knowledge, such as biomechanical models of speech production, may improve interpretability.

Given these limitations, our work builds upon LipNet but introduces enhanced preprocessing steps and architectural refinements designed to improve accuracy, robustness, and real-world applicability.

## 3 Methodology

The complete implementation, including preprocessing pipelines, model training scripts, and evaluation tools, is available at our public repository: https://github.com/mahmoudmayaleh/LipLingo.

## 3.1 Dataset

The experiments were conducted using the GRID audiovisual sentence corpus [9], which consists of video and audio recordings from 34 speakers (18 male, 16 female), each uttering 1,000 sentences, resulting in a total of 34,000 samples. The sentences follow a fixed syntactic structure (e.g., “place red at G9 now”), which facilitates supervised learning while providing speaker and articulation variability.

## 3.2 Data Preprocessing

Each video was decomposed into individual frames using OpenCV [19]. Face detection and extraction of 468 facial landmarks were performed using MediaPipe [17]. A bounding box around the mouth region was computed based on these landmarks to isolate relevant visual features and minimize background noise (Figure 1).

![](images/35d6e3104c4cb8a60f506e04d7b0b913b847eb98adc8fbdea973821a082661d3.jpg)  
Figure 1: Facial landmark detection using MediaPipe for isolating the mouth region. [17].

The extracted mouth regions were resized to 100 × 50 pixels to standardize input dimensions. To reduce

computational complexity, frames were converted to grayscale:

$$
Y = 0 . 2 9 9 R + 0 . 5 8 7 G + 0 . 1 1 4 B .\tag{1}
$$

Sequences shorter than the maximum length were zero-padded, and pixel intensities were normalized by subtracting the mean and dividing by the standard deviation. Examples of preprocessed frames are shown in Figure 2.

![](images/f7c48b2cb43d229759fe35ffe5bc9d161a9642d63ec49dcd471b4e321c1650dd.jpg)  
Figure 2: Example of mouth region extraction and grayscale conversion used in preprocessing. This step standardizes input features and reduces color-related variability, improving model robustness to lighting differences.

Although the GRID dataset was collected under controlled laboratory conditions with minimal background noise and stable illumination, we did not perform any explicit denoising. Our preprocessing pipeline (normalization, padding, grayscale conversion) standardizes the inputs across all sessions, but does not simulate environmental variability such as occlusions, motion blur or background noise. These factors are considered limitations of the present study and will be addressed in future work through targeted data augmentation and multimodal fusion strategies.

## 3.3 Data Pipeline

The TensorFlow tf.data API [1] was employed to construct an efficient data pipeline:

• Loading of video frame sequences and alignment labels.

• Application of preprocessing functions in parallel.

• Batching of samples with automatic sequence padding.

• Prefetching to optimize GPU utilization.

This design ensured consistently formatted and efficiently streamed data during training.

## 3.4 Model Architecture

The LipLingo model architecture extends LipNet [3] with modifications to improve feature extraction and sequence modeling. As shown in (Figure 3), the architecture includes three 3D convolutional layers, two bidirectional GRU layers, and a final dense-softmax output layer optimized with CTC loss.

![](images/4f599d1c7cbc8dd632a25d83cba3e206755b239d4532786382e82aea0574cc4e.jpg)  
Figure 3: Lip reading model architecture combining CNN, GRU, and CTC loss.

• Grayscale input. All video frames were converted to grayscale before being forwarded to the network. This choice reduces computational costs and memory usage while standardizing the input for all speakers and recording conditions. Although this step removes potentially informative color cues (e.g., tongue visibility, contrast between teeth and lips), previous work has shown that lip contour and motion provide the dominant features for visual speech recognition on datasets such as GRID. We acknowledge this trade-off and note that the inclusion of color information is a possibility for future work.

• 3D Convolutional layers. Three spatio-temporal convolutional layers were used to capture both spatial lip features and temporal dynamics across consecutive frames. This depth was selected as a balance: Deeper stacks provided only marginal improvements in accuracy but resulted in overfitting and increased computational costs, while shallower models underfit the task.

• Bidirectional GRU Layers: Two bidirectional GRU layers model sequential dependencies in forward and backward directions [6]. GRUs were chosen instead of LSTMs to reduce the number of parameters and training costs while maintaining a strong temporal modeling capacity. Experience has shown that adding more recurrent layers does not improve the accuracy on GRID and increases training instability, so the architecture was fixed at two GRU layers.

• Dense and SoftMax Layers: A fully connected (dense) layer projects the recurrent features onto the character class space. This is followed by a SoftMax activation, that generates a probability distribution over all possible characters at each time step. This design allows the model to output flexible sequence predictions aligned with the CTC loss, while keeping the final layer lightweight to avoid overfitting on the relatively small GRID dataset.

## 3.5 Connectionist Temporal Classification (CTC) Loss

Because the alignment between input frames and output characters is unknown, the Connectionist Temporal Classification (CTC) loss [11] was adopted:

$$
\mathcal { L } _ { \mathrm { C T C } } = - \log \sum _ { \pi \in B ^ { - 1 } ( y ) } P ( \pi | x ) ,\tag{2}
$$

where $B ^ { - 1 } ( y )$ is the set of all valid alignments that collapse to the target sequence y, and π denotes an alignment path. This allows end-to-end training without frame-level annotations.

## 3.6 Training Setup and Optimization

The model was trained for 98 epochs using the Adam optimizer [14] with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . A scheduler reduced the learning rate by a factor of 0.5 when validation loss plateaued for 10 epochs. The batch size was 32, and dropout layers were applied after recurrent layers to mitigate overfitting.

Data split strategy. The dataset was split into 80% training, 10% validation and 10% test sets. A rotating validation strategy was applied so that each speaker was included in the validation at least once during training

to ensure that different speaker characteristics were taken into account. The final test set was strictly speakerindependent: The speakers used for the test did not appear in either the training or the validation. This approach was intended to allow a fair assessment of generalization to unseen speakers.

Validation was performed every five epochs using a rotating validation set to evaluate generalization on unseen samples. Early stopping was triggered if no improvement was observed for 20 consecutive epochs.

## 3.7 Evaluation Strategy

The final model was evaluated on an independent test set of 1,000 samples. Predictions were compared to ground-truth transcripts using edit distance metrics. Evaluation metrics included:

• Character Error Rate (CER)

• Word Error Rate (WER)

• Precision

• Recall

• F1-Score

These metrics were selected to provide complementary perspectives. Character Error Rate (CER) and Word Error Rate (WER) are widely used in speech recognition as they directly quantify transcription fidelity by counting substitutions, deletions and insertions at the sequence level [24]. They indicate how close the predicted output is to the real transcription, which is essential for evaluating sentence intelligibility. However, CER and WER alone can overlook effects of class imbalance: For example, frequent characters or short words may dominate the error count. Therefore, we also include Precision, Recall and F1-score [22], which evaluate the model’s ability to correctly identify less frequent characters and balance false positives and false negatives. Together, this combination of metrics provides both a global overview of the transcription quality and a fine-grained analysis of the classification reliability. Example outputs and confusion matrices are reported in Section 4.

## 4 Experiments & Results

This section presents the performance of the LipLingo model under different training configurations. Results are reported using the evaluation metrics defined in the previous section. Confusion matrices and example predictions illustrate both the effectiveness of the approach and the remaining challenges.

## 4.1 Training Configurations

Three experimental configurations were evaluated:

• Model A: 100 epochs without any validation dataset.

• Model B: 70 epochs with validation starting after epoch 40.

• Model C: 98 epochs with rotating validation data every 5 epochs.

## 4.2 Quantitative Performance

Table 2 summarizes character-level performance for each configuration.

Table 2: Character-level performance comparison across training configurations.
<table><tr><td>Model</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td>F1-Score</td></tr><tr><td>100 epochs (no validation)</td><td>0.3524</td><td>0.3524</td><td>0.3603</td><td>0.3516</td></tr><tr><td>70 epochs (mid-validation)</td><td>0.3964</td><td>0.3964</td><td>0.4028</td><td>0.3936</td></tr><tr><td>98 epochs (rotating validation)</td><td>0.9501</td><td>0.9488</td><td>0.9501</td><td>0.9490</td></tr></table>

The absence of validation in Model A led to overfitting, as the network continued optimizing on training data without monitoring generalization. Model B improved slightly by introducing validation after epoch 40, but

by that point overfitting had already occurred. Model C, employing rotational validation, consistently exposed the model to unseen data, producing robust and generalizable features.

## 4.3 Sentence-Level Performance

Table 3 lists sentence-level results for the best-performing configuration (Model C).

Table 3: Sentence-level evaluation metrics for Model C (98 epochs) over three runs $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Accuracy</td><td> $0 . 8 7 2 0 \pm 0 . 0 0 2 3$ </td></tr><tr><td>Recall</td><td> $0 . 8 6 0 0 \pm 0 . 0 0 2 1$ </td></tr><tr><td>Precision</td><td> $0 . 8 8 0 0 \pm 0 . 0 0 1 9$ </td></tr><tr><td>F1-Score</td><td> $0 . 8 7 0 0 \pm 0 . 0 0 2 5$ </td></tr></table>

In one example, the words green and nine were predicted as gren and ain, respectively. Despite these substitutions, the sentence retained high overall similarity.

## 4.4 Confusion Matrices

Figure 4 shows the character-level confusion matrix for Model C, where most predictions align with the diagonal, indicating consistent classification. Figure 5 shows the word-level confusion matrix, where most words were correctly predicted.

$$
\begin{array} { r l } & { \quad \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes Q \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ & { \quad \times \frac { 1 } { 2 } Q _ { 0 } ( \Omega \otimes \Omega ) } \\ &  \quad \times \frac { 1 } { 2 } Q _ { 0 } (  \end{array}
$$

Figure 4: Character-level confusion matrix of the best-performing model (Model C).

A closer look at the confusion matrices shows that most of the errors were not evenly distributed, but concentrated on a few frequent cases. The space character was the most problematic, as it was misclassified four times and replaced by other characters five times. Characters with rounded or visually similar articulations also caused confusion, such as ‘o’, which was predicted as ‘w’, and ‘e’ was occasionally confused with an ‘a’. These patterns suggest that visually ambiguous shapes and high-frequency symbols dominate the model’s errors. This emphasises the potential benefit of integrating language models or temporal context to reduce such systematic confusions.

$$
\begin{array} { r l } { \gamma _ { 1 } ^ { \mathrm { o u t } } = \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } \gamma _ { 1 } ^ { \mathrm { o u t } } } & { = \gamma _ { 2 } ^ { \mathrm { o u t } } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 1 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } } \\ { \gamma _ { 2 } ^ { \mathrm { o u t } } = \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } \gamma _ { 2 } ^ { \mathrm { o u t } } } & { = \gamma _ { 2 } ^ { \mathrm { o u t } } \gamma _ { 1 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } } \\ { \gamma _ { 3 } ^ { \mathrm { o u t } } = \gamma _ { 1 } ^ { \mathrm { o u t } } \gamma _ { 1 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 3 } ^ { \mathrm { o u t } } } &  = \gamma _ { 2 } ^ { \mathrm { o u t } } \gamma _ { 2 } ^ { \mathrm { o u t } } + \frac { \gamma _ { 1 } ^ { \mathrm { o u t } } } { 2 } \gamma _ { 3 } ^  \ \end{array}
$$

Figure 5: Word-level confusion matrix for Model C.

The word-level matrix shows that the errors cluster around a few frequent patterns rather than being evenly distributed. The word green acted as an attractor class: it was predicted instead of at (3×), lay (3×), bin (2×) and in (2×), and the near-miss gren was frequently decoded as green (5×). This indicates vowel deletion/insertion around the “ee” sound, This behavior is consistent with the CTC decoding process, which can merge repeated symbols and therefore struggles with vowel lengthening. We also observed confusions caused by vowel length or closely related phoneme sequences, e.g., son→soon (3×) andfive→soon (2×), as well as occasional ambiguity of boundaries between letters and words $( \mathbf { e } . \mathbf { g } . , f {  } f i \nu e )$ and short high-frequency terms (now→nine; c→e). Overall, these patterns indicate that visually ambiguous vowel lengths and onset clusters dominate the errors at the word level. The inclusion of a lightweight word-level language model or lexicon constraints should help to regularize such cases.

## 4.5 Flexible Word Similarity Evaluation

A flexible similarity measure was applied to account for alignment differences between predicted and reference sentences. For each predicted word $w _ { p }$ at position i, we compared it to the neighboring reference words:

$$
\operatorname { C a n d i d a t e s } ( i ) = \{ w _ { r } ( i - 1 ) , w _ { r } ( i ) , w _ { r } ( i + 1 ) \} .\tag{3}
$$

Similarity was defined as:

$$
\mathrm { S i m i l a r i t y } ( w _ { p } ) = \operatorname* { m a x } _ { c \in \mathrm { C a n d i d a t e s } ( i ) } \left[ 1 - \frac { \mathrm { L e v e n s h t e i n } ( w _ { p } , c ) } { \operatorname* { m a x } ( | w _ { p } | , | c | ) } \right] .\tag{4}
$$

[15]

Table 4 shows an evaluation example.

Table 4: Example sentence with flexible word similarity evaluation with ${ \bf a } \pm 1$ window.
<table><tr><td>Word</td><td>Candidate Matches</td><td>Similarity (%)</td></tr><tr><td>set</td><td>{set, white}</td><td>100</td></tr><tr><td>white</td><td>{set, white, sp}</td><td>100</td></tr><tr><td>sp</td><td>{white, sp, with}</td><td>100</td></tr><tr><td>with</td><td>{sp, with, f}</td><td>100</td></tr><tr><td>f</td><td>{with, f, five}</td><td>100</td></tr><tr><td>five</td><td>{f, five, soon}</td><td>100</td></tr><tr><td>son</td><td>{five, soon}</td><td>75</td></tr></table>

The neighborhood window was set to ±1 word position, which was chosen to accommodate typical minor misalignments seen in Connectionist Temporal Classification (CTC) decoding, its known tendency to collapse repeated tokens or shift alignment without frame-level supervision [11]. A ±1 window represents a practical balance: It captures near-miss predictions that preserve sentence intelligibility without loosening the metric too much and inflating performance.

## 4.6 Error Rates

CER and WER were computed as:

$$
\mathrm { C E R } = \frac { \mathrm { S u b s t i t u t i o n s } + \mathrm { D e l e t i o n s } + \mathrm { I n s e r t i o n s } } { \mathrm { T o t a l } \mathrm { C h a r a c t e r s } }\tag{5}
$$

$$
\mathrm { W E R } = { \frac { \mathrm { S u b s t i t u t i o n s } + \mathrm { D e l e t i o n s } + \mathrm { I n s e r t i o n s } } { \mathrm { T o t a l ~ W o r d s } } }\tag{6}
$$

To illustrate how character-level errors are quantified, consider the following example:

Example sentence prediction:

• Original: lay green in e nine please

• Predicted: lay gren in e ain please

• Errors:

– “green” → “gren” (1 deletion)

– “nine” → “ain” (1 substitution, 1 deletion)

• Character Error Rate: 3 errors / 26 characters = 12%

Word Error Rate (WER) evaluation:

• Original: lay — green — in — e — nine — please

• Predicted: lay — gren — in — e — ain — please

• Errors:

– “green” → “gren” (1 substitution)

– “nine” → “ain” (1 substitution)

• Word Error Rate: 2 errors / 6 words = 33%

Model C achieved a CER of 4.99% and a WER of 13.1%, demonstrating strong transcription fidelity.

## 4.7 Comparison with Baselines

Table 5 compares LipLingo with LipNet and the 3D-ResNet + Bi-GRU + MixUp model.

Table 5: Comparison with baseline and state-of-the-art models.
<table><tr><td>Model</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td>F1-Score</td></tr><tr><td>LipNet (Unseen Speakers)</td><td>0.9381</td><td>0.8821</td><td>0.8821</td><td>0.8821</td></tr><tr><td>3D-ResNet + Bi-GRU + MixUp</td><td>0.9490</td><td>0.8950</td><td>0.8950</td><td>0.8950</td></tr><tr><td>LipLingo (Characters)</td><td>0.9501</td><td>0.9488</td><td>0.9501</td><td>0.9490</td></tr><tr><td>LipLingo (Sentences)</td><td>0.8720</td><td>0.8600</td><td>0.8800</td><td>0.8700</td></tr></table>

To ensure fairness, all models were trained and evaluated on the same partitions of the GRID dataset using a standardized evaluation protocol. Baseline implementations followed their originally reported preprocessing steps, while LipLingo introduced a standardized pipeline with automatic mouth region detection, resizing, and grayscale conversion. This distinction clarifies that performance differences arise from both the architectural design and the proposed preprocessing improvements, rather than inconsistencies in the data splits.

Beyond GRID, recent transformer-based models such as Conformer VSR [5], AV-HuBERT [23], and Swin-Lip [20] achieve significantly lower WERs on large, unconstrained datasets (e.g., LRS3). While these results are not directly comparable due to differences in vocabulary size, recording conditions, and dataset scale, they provide useful context: LipLingo attains strong accuracy on GRID while maintaining a lightweight architecture, whereas state-of-the-art systems rely on extensive pretraining and massive corpora. This highlights the trade-off between reproducibility on controlled benchmarks and scalability to unconstrained real-world scenarios.

## 4.8 Error Analysis and Performance Summary

Beyond overall metrics, an error analysis was conducted to identify model weaknesses. Most misclassifications involved visually similar phonemes (e.g., /b/ vs. /p/) and short function words (e.g., “in” vs. “an”), which are inherently difficult to distinguish using only visual cues. The proposed preprocessing steps, including mouth isolation and normalization, reduced speaker-specific lip shape errors, highlighting the value of consistent input representation. However, errors related to rapid speech and partial occlusion (e.g., head movement) persisted, suggesting the need for temporal attention mechanisms and multimodal integration in future work.

Overall, the model trained with rotational validation achieved the best performance, combining high characterlevel accuracy with strong sentence-level generalization. Compared with baseline systems, LipLingo matched or outperformed state-of-the-art approaches while maintaining stable error rates across different speakers and sentence structures.

## 5 Discussion

## 5.1 Interpretation of Results

The experiments show that LipLingo achieves high accuracy at the character level and strong performance at the sentence level, especially when trained with rotating validation data. The final model achieved a Character Error

Rate (CER) of 4.99% and a Word Error Rate (WER) of 13.1%, matching or exceeding previous benchmarks on the GRID dataset [3, 10]. A key finding is the effect of the rotating validation strategy, which consistently improved the generalization of the model compared to fixed validation splits. By continuously exposing the model to different validation subsets, this approach reduces overfitting to speaker- or session-specific patterns and better reflects real-world usage conditions where unseen variations are common. This result has broader implications beyond lip reading and suggests that dynamic validation strategies may benefit other areas of sequence learning, such as gesture recognition, sign language translation or human activity detection, where dataset homogeneity often limits model robustness.

The gap between the performance at character and sentence level is made even clearer by the worked example in section 4.6. While the prediction “lay gren in e ain please” resulted in a Character Error Rate (CER) of only 12%, the corresponding Word Error Rate (WER) was 33%. This shows how small errors at the character level can escalate to whole word substitutions, which in turn reduce accuracy at the sentence level. This phenomenon explains why LipLingo achieves high character accuracy (95.0%) but comparatively lower sentence accuracy (87.2%).

Recent advances suggest a promising path to narrowing the character-to-sentence gap by integrating linguistic modeling directly into VSR systems. For instance, the VALLR framework [2] first predicts phonemes with a transformer-CTC model and then reconstructs full sentences using a finetuned LLM. Despite using 99% less labeled data, it achieves a state-of-the-art WER of 18.7% on LRS3. This demonstrates how language modeling can substantially reduce word-level errors, especially under data-constrained conditions.

The flexible similarity analysis also confirmed the robustness of the model to minor word omissions or shifts in alignment, as the high similarity scores were maintained even when the predictions deviated slightly from the reference sentences.

Analysis of the confusion matrices (figures 4 and 5) provides additional insight into the limitations of the model. At the character level, the space token was the most error-prone. It was frequently confused with vowels such as ‘e’ or ‘a’, indicating ambiguities between words. At the word level, ‘green’ frequently acted as an attractor, replacing or being replaced by phonetically and visually similar forms such as ‘gren’, ‘lay’ or ‘in’. Other recurring errors were short functional words (e.g., ‘now’→‘nine’) and vowel- length confusions (e.g., ‘son’→‘soon’). These error patterns confirm that the remaining weaknesses of LipLingo are concentrated in visually ambiguous cases, suggesting that the inclusion of linguistic context or external language models could further reduce such errors.

Taken together, these analyses validate the key design choices of LipLingo. The proposed preprocessing pipeline (automatic mouth region detection and grayscale conversion) improved robustness by standardizing the input across all speakers, while the rotational validation strategy consistently improved generalization. The combination of quantitative benchmarks, error analysis and qualitative examples shows that LipLingo’s performance improvements are supported by multiple lines of evidence, strengthening confidence in the proposed approach.

## 5.2 Limitations

Despite these encouraging results, several limitations remain:

• Dataset Scope: The GRID corpus consists of syntactically regular and restricted sentences, which simplifies the lip-reading task. Consequently, the model’s performance on unconstrained, natural conversational speech has not been evaluated.

• Speaker Diversity: Although the dataset includes multiple speakers, it does not fully capture demographic and linguistic variability such as accents, dialects, or individual speaking styles, which may limit generalization.

• Environmental Variability: All recordings were captured in controlled laboratory conditions. Real-

world scenarios involving varied lighting, partial occlusion (e.g., facial hair, masks), and head movements remain unaddressed.

• Computational Cost: Training required significant computational resources (e.g., an NVIDIA T4 GPU). Scaling the system to larger, more diverse datasets may require algorithmic optimizations or more powerful hardware.

Moreover, evaluation was conducted only on the GRID dataset, which has a restricted vocabulary and controlled recording conditions. This limits the generalization of our findings to more diverse, real-world settings where vocabulary size, speaker variation, and environmental noise play a much greater role.

## 5.3 Future Work

Future research directions aim to address these limitations and further improve the robustness of LipLingo:

• Dataset Expansion: Extend the evaluation to larger and more diverse datasets, such as LRW, LRW-1000 [26], LRS2 [8], and LRS3, which capture natural, unconstrained speech and provide a stronger benchmark for cross-domain generalization.

• Multimodal Integration: Combine visual information with complementary modalities such as audio signals or head pose estimation, to improve robustness under challenging conditions. Self-supervised multimodal approaches such as AV-HuBERT [23] demonstrate the potential of this direction.

• Transfer Learning: Utilize pre-training on large-scale video datasets followed by fine-tuning for lipreading tasks to improve performance on limited-domain data. Recent transformer-based models such as Conformer VSR [5], LipFormer [25] and SwinLip [20] demonstrate the effectiveness of this strategy.

• Model Optimization: Investigate efficient deployment strategies, including pruning, quantization, and knowledge distillation, to reduce inference latency and computational requirements while maintaining accuracy.

• Explainability: Incorporate attention mechanisms or visual mapping methods to improve model interpretability and identify the most influential visual cues for lip-reading predictions, supporting transparent deployment in assistive or biometric applications.

## 6 Conclusion

This study presented LipLingo, an end-to-end deep learning model for automatic lip reading based on an enhanced version of the LipNet architecture. Through comprehensive preprocessing, robust spatio-temporal feature extraction, and carefully designed validation strategies, the model achieved 95.01% character-level accuracy and 87.20% sentence-level accuracy on the GRID dataset.

Comparisons with established baseline systems demonstrated that LipLingo matches or surpasses the state of the art while maintaining consistent performance across diverse speakers. The flexible similarity evaluation further confirmed the system’s robustness to minor alignment discrepancies.

While these results are promising, further work is required to handle more diverse and unconstrained speech data. Future research will focus on expanding training datasets, integrating audiovisual modalities, and enhancing model interpretability to support the deployment of practical lip-reading systems in real-world applications.

## References

[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, and X. Zheng. Tensorflow: Large-scale machine learning on heterogeneous systems. https://www.tensorflow.org/, 2015.

[2] Anonymous. Vallr: Visual asr language model for lip reading. arXiv preprint arXiv:2503.21408, 2025. two-stage phoneme-CTC + LLM model achieves 18.7

[3] Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas. Lipnet: Sentence-level lipreading. arXiv preprint arXiv:1611.01599, 2016.

[4] L. E. Baum and T. Petrie. Statistical inference for probabilistic functions of finite state markov chains. Annals ofMathematical Statistics, 37(6):1554–1563, 1966.

[5] Oscar Chang, Pingchuan Ma, Alexandros Haliassos, Stavros Petridis, and Andrew Zisserman. Conformers are all you need for visual speech recognition. arXiv preprint arXiv:2302.10915, 2023.

[6] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger¨ Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder–decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

[7] Joon Son Chung and Andrew Zisserman. Lip reading in the wild (lrw). In Asian Conference on Computer Vision (ACCV), pages 87–103. Springer, 2016.

[8] Joon Son Chung and Andrew Zisserman. Lip reading in the wild. In IEEE International Conference on Computer Vision Workshops (ICCVW), pages 2871–2879, 2017.

[9] M. Cooke, J. Barker, S. Cunningham, and X. Shao. The grid audio-visual speech corpus. Zenodo, 2006.

[10] T. Exarchos, G. N. Dimitrakopoulos, A. G. Vrahatis, G. Chrysovitsiotis, Z. Zachou, and E. Kyrodimos. Lip-reading advancements: A 3d cnn/lstm fusion for precise word recognition. BioMedInformatics, 4(1):410–422, 2024.

[11] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: Labelling´ unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning, pages 369–376, 2006.

[12] Alexandros Haliassos, Pingchuan Ma, Andrew Zisserman, Themos Stafylakis, and Stavros Petridis. Unified speech recognition: A single model for auditory, visual, and audiovisual inputs. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[13] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.

[14] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[15] V. I. Levenshtein. Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8):707–710, 1966.

[16] S. Lu, Y. Tong, and H. Liu. Automatic speechreading recognition using visemes and hidden markov models. Speech Communication, 44(3):269–280, 2006.

[17] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, and M. Grundmann. Mediapipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172, 2019.

[18] Marzieh Oghbaie, Arian Sabaghi, Kooshan Hashemifard, and Mohammad Akbari. Advances and challenges in deep lip reading. arXiv preprint arXiv:2110.07879, 2021.

[19] OpenCV. Color space conversions. https://docs.opencv.org/3.4/de/d25/imgproc\_ color\_conversions.html, 2023.

[20] Yoon-Ha Park, Min-Jae Kim, Hyeon-Ji Lee, and Jun-Ho Choi. Swinlip: An efficient visual speech encoder for lip reading using swin transformer. IEEE Access, 2025. Early Access.

[21] S. Petridis, T. Stafylakis, P. Ma, F. Cai, G. Tzimiropoulos, and M. Pantic. End-to-end audiovisual speech recognition. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6548–6552, 2018.

[22] D. M. Powers. Evaluation: From precision, recall and f-measure to roc, informedness, markedness and correlation. Journal ofMachine Learning Technologies, 2(1):37–63, 2011.

[23] Bowen Shi, Wei-Ning Hsu, and Abdelrahman Mohamed. Learning audio-visual speech representation by masked multimodal cluster prediction. arXiv preprint arXiv:2201.02184, 2022.

[24] Y. Wang, A. Acero, and C. Chelba. That matches the optimization objective for understanding: A study on word error rate and language understanding. Technical report, Microsoft Research, 2003.

[25] Feng Xue, Huibin Xu, Jianhua Tao, Jing Yi, Peng Liu, Wenjing Xu, and Lianhong Chen. Lipformer: Learning to lipread unseen speakers based on visual-landmark transformers. IEEE Transactions on Circuits and Systemsfor Video Technology, 33(12):7376–7389, 2023.

[26] Shuang Yang, Yuanhang Zhang, Dalu Feng, Mingmin Yang, Chenhao Wang, Jingyun Xiao, Keyu Long, Shiguang Shan, and Xilin Chen. Lrw-1000: A naturally-distributed large-scale benchmark for lip reading in the wild. In 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019), page 1–8. IEEE Press, 2019.

[27] C. Zhou and R. Paffenroth. Anomaly detection with robust deep autoencoders. Unpublished manuscript, 2017.