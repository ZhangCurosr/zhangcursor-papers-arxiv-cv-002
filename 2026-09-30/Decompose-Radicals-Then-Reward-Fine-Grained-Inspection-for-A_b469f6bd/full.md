# Decompose Radicals, Then Reward: Fine-Grained Inspection for Accurate Chinese Text Rendering

Yazhen Xie<sup>1</sup>,<sup>2</sup>,<sup>∗</sup>, Xingsong Ye<sup>1</sup>,<sup>2</sup>,<sup>∗</sup>, Zhineng Chen<sup>1</sup>,<sup>2</sup>,<sup>†</sup>

<sup>1</sup>Institute of Trustworthy Embodied AI, Fudan University

<sup>2</sup>Shanghai Key Laboratory of Multimodal Embodied AI

<sup>∗</sup>Equal contribution, <sup>†</sup>Corresponding author

## Abstract

Rendering accurate Chinese text remains challenging for text-to-image models. Existing OCR-based reinforcement-learning rewards compare decoded transcripts with target strings. Such rewards overlook the compositional nature of Chinese writing: an ideograph consists of reusable components arranged through explicit spatial relations, yet OCR evaluates it as an atomic character. Consequently, visually diferent radical-level errors may receive equally coarse feedback, encouraging glyphs that merely resemble the target instead of faithfully reproducing its internal structure. We employ Ideographic Description Sequences (IDS), which comprise spatial operators and character components, and train an expert IDS recognizer to transcribe rendered Chinese text into this representation. Building on this recognizer, we introduce IDSpect, which deterministically decomposes the target text into IDS tokens and aligns crop-level visual IDS predictions with the target sequence. Globally unique token credit makes this comparison robust to the order of detected text regions. Combined with a whole-character semantic reward, IDSpect supplies fine-grained credit with component and spatial-relation without changing the image generator or adding inference-time cost. Experiments with GRPO post-training of Qwen-Image demonstrate that IDSpect achieves leading structural quality and semantic alignment on LongText and GenTextEval.

Keywords: Text-to-Image Generation, Chinese Visual Text Rendering, OCR Reward Correspondence: zhinchen@fudan.edu.cn

## 1 Introduction

Text-to-image generation has advanced rapidly [13, 22]. Specialized encoders and explicit glyph conditioning have further improved visual text rendering [2, 20, 25]. Nevertheless, rendering specified text inside an image remains a persistent failure mode, particularly for Chinese ideographs [31]. Unlike alphabetic characters, a Chinese ideograph can encode a two-dimensional composition of reusable components under left–right, top–bottom, enclosing, and other spatial relations [18]. This large and long-tailed output space makes both generation and evaluation dificult: a glyph may preserve several correct components while corrupting only one component or their spatial arrangement.

Reinforcement-learning post-training ofers a direct way to optimize text rendering [10], and OCR similarity is a natural reward for this objective [32]. However, an OCR reward first compresses the image into a character transcript and then evaluates that transcript. Consequently, it provides weak credit for partial structural progress: repairing one radical of a still-misrecognized ideograph may leave the reward unchanged. Conversely, linguistic priors can occasionally recover the intended character from a malformed glyph, producing a high semantic score without adequate visual evidence.

![](images/7544add62f4436ffdc7b29ce06e9520f10a081cba3a8e8c5c268245470ad5071.jpg)  
Figure 1 Local reward responses to a malformed ideograph. Character-level rewards produce discrete decisions, whereas IDSpect retains partial structural credit via IDS matching.

Fig. 1 shows both failure modes in the same generated image. For the highlighted target ideograph, a specialist OCR model follows the local appearance and predicts a visually plausible but incorrect character, contributing 0.00 to the character-level reward. A global MLLM OCR instead recovers the target from the surrounding linguistic context despite its malformed structure, producing a false-positive local contribution of 1.00. TextPecker detects or quantifies anomalous glyphs and is representative of recent structure-aware evaluators for visual text rendering [32]. TextPecker correctly identifies the same glyph as structurally anomalous and replaces it with #, avoiding the MLLM’s false positive but assigning a local contribution of 0.00. Once the glyph is collapsed into an anomaly marker, however, its correct and incorrect internal structures are no longer distinguished.

Prior Chinese OCR work uses radicals, strokes, and IDS to recognize rare or unseen characters [1, 29, 30]. We build on these representations to assess how much of a generated ideograph’s internal structure matches its target, providing partial credit even when the whole character is incorrect. To this end, we introduce IDSpect, illustrated in Fig. 2. We first train an expert IDS recognizer that transcribes detected text regions into IDS token sequences. During post-training, the detected crops feed two complementary reward branches: the semantic branch compares OCR transcripts with the target text, while the structural branch aligns visual IDS predictions with the deterministic target IDS. Their weighted sum combines transcript-level fidelity with intra-character structural credit. In Fig. 1, IDSpect retains credit for matched IDS tokens while penalizing mismatched ones, assigning a standalone IDS reward of 0.22. Thus, anomaly-aware rewards identify which glyph is problematic, whereas a compositional reward additionally quantifies how much of its internal structure already matches the target.

Our contributions are threefold:

• We identify the coarse-credit limitation of atomic OCR rewards for Chinese visual text generation, under which structurally distinct intra-character errors can receive indistinguishable feedback.

• We develop an expert IDS recognizer that decomposes rendered text in generated images into sequences

of spatial operators and character components.

• We construct IDSpect as a fine-grained reward for GRPO. On LongText and GenTextEval, it improves both structural quality and semantic alignment over general OCR-based rewards and TextPecker.

## 2 Related Work

## 2.1 Visual text generation and post-training

Text-to-image models have progressively improved spelling and layout through specialized encoders and explicit glyph conditions [2, 3, 11, 19, 20, 22]. More broadly, recent difusion-based generation methods have explored fine-grained visual conditioning and text-driven control to achieve more precise correspondence between visual prompts and generated content [23, 24]. Recent methods introduce hierarchical rewards and region-level preference optimization for visual text rendering [5, 16]. When used as rewards, specialist OCR models [6, 9, 27, 28] such as PP-OCRv5 [4] directly optimize transcription accuracy but inherit the recognizer’s atomic label space. TextPecker [32] introduces structural anomaly quantification for reward-guided generation. However, it roughly recognizes all incorrectly written characters as “#”, which is overly coarse. This motivates us to measure edit progress over the internal composition of a target ideograph rather than only assigning a quality judgment to the whole glyph or text region.

## 2.2 Structure-aware Chinese text recognition

Radicals, strokes, and IDS have long been used to improve recognition of rare or unseen Chinese characters [1, 29, 30, 33]. These studies use decomposition to recognize text. Inspired by them, we instead train an IDS reader as a reward model and use its structured output to optimize a generative model. Reward utility depends not only on final recognition accuracy, but also on whether intermediate scores rank partially correct and malformed glyphs in a useful order.

## 3 Method

## 3.1 Problem formulation

Let � be an image description and $y = \left( c _ { 1 } , \ldots , c _ { M } \right)$ the text that should be visible in the generated image. A generator $G _ { \theta }$ samples $I \sim G _ { \theta } ( p , y )$ . During group-relative post-training, multiple images are sampled under the same condition and scored by a reward �(�<sub>,</sub> �) [10, 14]. Our goal is to construct a reward that preserves semantic correctness while providing fine-grained supervision for the internal composition of Chinese ideographs.

An OCR model $f _ { \mathrm { o c r } }$ is applied to the generated image to obtain the recognized transcript $\hat { y }$ . Let $\bar { y }$ and $\bar { \hat { y } }$ denote the target and recognized strings after removing spaces and lowercasing. We measure semantic fidelity using normalized character edit similarity:

$$
r _ { \mathrm { s e m } } ( I , y ) = \left\{ \begin{array} { l l } { 1 , } & { \bar { y } \mathrm { o c c u r s i n } \bar { \hat { y } } , } \\ { 1 - \frac { \operatorname* { m i n } \{ \mathrm { L e v } ( \bar { \hat { y } } , \bar { y } ) , | \bar { y } | \} } { | \bar { y } | } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

All training targets are nonempty, and the truncated edit distance keeps the reward within [0<sub>,</sub> 1]. The substring condition gives full credit when the target text is correctly recognized even if the OCR output contains additional characters. However, this semantic reward treats each character as an atomic symbol and therefore provides limited information about partially correct character structures. We address this limitation by introducing an IDS-based compositional reward in the following section.

![](images/c9a6d11d82ca85ad4ee02b334d4591cbe091b960820cd0419d762e5eeca2e238.jpg)  
Figure 2 Framework of IDSpect. An OCR detector extracts text crops from each generated image. The semantic branch compares OCR transcripts with the target text, while the IDS branch compares visually predicted and target IDS tokens for intra-character feedback. Their weighted sum guides GRPO post-training of Qwen-Image.

## 3.2 Balanced IDS recognizer

IDS represents a Chinese ideograph as a hierarchical tree, where internal nodes correspond to spatial operators and leaf nodes correspond to reusable character components [18]. In this representation, a normal Chinese character can be recursively parsed into a sequence of structural operations and primitive components, providing an explicit description of how the character is spatially composed. For example, a character formed by placing two components side by side is represented by a horizontal composition operator together with the IDS representations of its two subcomponents. The decomposition can then be recursively applied to each subcomponent until reaching indivisible or reusable components. In this way, IDS converts the implicit two-dimensional composition of a Chinese character into an explicit hierarchical structure that can be serialized as a token sequence.

Using the Unicode 16.0 BabelStone lexicon [18, 21], we recursively decompose the Chinese characters covered by GB 18030–2022 [17]. Let $D ( c )$ denote the flattened IDS sequence obtained by recursively decomposing a character �. For a text string $y = \left( c _ { 1 } , \ldots , c _ { M } \right)$ , its target structural representation is constructed by concatenating the IDS sequences of all characters, i.e., $D ( y ) = D ( c _ { 1 } ) \oplus \cdots \oplus D ( c _ { M } )$ . We cap the recursive decomposition depth at 10 and construct the decoder vocabulary from the resulting spatial operators and reusable components.

Based on this representation, the IDS recognizer $f _ { \mathrm { i d s } }$ is trained to directly predict the IDS sequence from a rendered text image rather than first recognizing the image as a character sequence and then performing symbolic decomposition. Given an input text image �, the recognizer produces $\hat { D } = f _ { \mathrm { i d s } } ( x )$ , where �<sup>ˆ</sup> is a sequence of spatial operators and character components. Thus, the recognition process transforms the visual appearance of the text directly into its underlying structural representation. This formulation shifts the recognition target from a flat character identity to the internal composition of each ideograph, enabling the model to explicitly recover the structural primitives and their spatial relationships from visual evidence.

Specifically, we train an IDS recognizer $f _ { \mathrm { i d s } }$ that maps a text-region image $I _ { k }$ to an IDS sequence $\hat { s } _ { k }$ . Building on the SVTRv2 visual backbone [6], a recent state-of-the-art architecture for scene text recognition, we reformulate character transcription as autoregressive prediction over the IDS vocabulary. The resulting recognizer couples an SVTRv2 encoder with a Transformer-style decoder following NRTR [15] to directly

Table 1 Quantitative comparison of Qwen-Image variants on Chinese visual text rendering benchmarks. Base denotes the frozen model. OCR, TextPecker [32], and IDSpect denote GRPO post-training with the corresponding rewards. Avg.: original benchmark text score, Qua.: structural quality, Sem.: semantic alignment. Qua. and Sem. are evaluated by TextPecker.
<table><tr><td rowspan="2">一 Rewards</td><td colspan="3">LongText</td><td colspan="2">一 GenTextEval</td></tr><tr><td>一 Avg.</td><td>Qua.</td><td>Sem. 一</td><td>Qua.</td><td>Sem.</td></tr><tr><td>Base</td><td>0.920</td><td>0.924</td><td>0.834</td><td>0.933</td><td>0.810</td></tr><tr><td>OCR</td><td>0.967</td><td>0.956</td><td>0.886</td><td>0.953</td><td>0.874</td></tr><tr><td>TextPecker</td><td>0.974</td><td>0.969</td><td>0.908</td><td>0.973</td><td>0.897</td></tr><tr><td>IDSpect</td><td>0.972</td><td>0.975</td><td>0.928</td><td>0.979</td><td>0.911</td></tr></table>

recover the compositional structure of rendered text.

Crucially, training data determine whether this recognizer can reliably parse the long tail of Chinese ideographs. Natural scene-text corpora are strongly biased toward frequent characters, and their IDS annotations consequently provide highly imbalanced supervision for radicals and other components. We therefore adapt the UnionST rendering engine [26] to construct a balanced synthetic training set (IDSynth-1M). We form a rare-ideograph-enriched corpus with balanced frequencies across covered characters, render its strings into text images, and pair each image with its IDS sequence. This design does more than increase rare-character coverage. With naturally distributed text, the recognizer can exploit character-frequency and linguistic shortcuts: it first recognizes a glyph as a frequent character and then reproduces its canonical IDS, without grounding the output in the observed components. Such behavior reduces IDS prediction to character recognition and cannot expose internal glyph errors. Balanced, randomly composed transcripts discourage this shortcut and promote component-grounded prediction.

## 3.3 Target-conditioned compositional reward

The OCR detector yields text crops $\{ I _ { k } \} _ { k = 1 } ^ { K } ,$ , and the IDS recognizer produces a structural token sequence $\hat { s } _ { k } = f _ { \mathrm { i d s } } ( I _ { k } )$ for each crop. Let $s = D ( y )$ denote the complete IDS sequence of the target text. Rather than concatenating the predicted sequences in detector order, we align each $\hat { s } _ { k }$ with candidate spans of �, since detected crops may appear in an arbitrary order and may cover diferent portions of the target text. Candidate spans are generated by semi-global edit alignment and filtered by non-maximum suppression to remove near-duplicate matches. A global assignment then selects at most one target span for each predicted crop, while each target position can receive exact-match credit only once. This makes the matching independent of crop order while preserving the IDS-token order within each crop and its aligned target span.

Let $A ^ { \star }$ denote the selected global assignment and $C ( A ^ { \star } )$ the number of uniquely credited target IDS tokens. We define the compositional reward as a token-level F1 score

$$
r _ { \mathrm { i d s } } ( I , y ) = \frac { 2 C ( A ^ { \star } ) } { | D ( y ) | + \sum _ { k = 1 } ^ { K } | \hat { s } _ { k } | } .\tag{2}
$$

The reward provides fine-grained structural supervision because matching individual components and spatial operators can increase the score even when the atomic OCR output remains unchanged. At the same time, missing or extraneous predictions are penalized through the F1 denominator and the unique-credit constraint. Finally, we combine the semantic and compositional rewards as

$$
\begin{array} { r } { R ( I , y ) = \lambda r _ { \mathrm { s e m } } ( I , y ) + ( 1 - \lambda ) r _ { \mathrm { i d s } } ( I , y ) , } \end{array}\tag{3}
$$

where $r _ { \mathrm { s e m } }$ provides target-level semantic supervision and $r _ { \mathrm { i d s } }$ provides fine-grained structural supervision. The fused reward is used to guide GRPO post-training of $G _ { \theta }$ . Both reward branches are discarded after training, so IDSpect introduces no additional cost during generator inference.

## 4 Experiments

## 4.1 Experimental setup

Generator and optimization. We post-train Qwen-Image [22] with Flow-GRPO [10] using the Flow-Factory framework [12]. Following its default Qwen-Image configuration, we optimize LoRA adapters [8] with rank � = 64 and scaling factor � = 128 using AdamW, with a learning rate of $3 \times 1 0 ^ { - 4 }$ and weight decay of $1 0 ^ { - 4 }$ All remaining hyperparameters follow the framework defaults. For IDSpect, we set $\lambda = 0 . 5$ , assigning equal weights to the semantic and IDS rewards throughout all experiments.

IDS-recognizer data. We construct IDSynth-1M, a dataset of one million rendered scene text images, to train $f _ { \mathrm { i d s } }$ . To mitigate the long-tailed distribution of natural Chinese text, we sample characters approximately uniformly from the entire character set covered by the selected Chinese fonts inherited from UnionST to form transcripts. This sampling strategy also suppresses natural lexical co-occurrence, making it dificult for the model to infer complete characters from linguistic context alone and thereby encouraging it to parse the IDS sequence primarily from visual information without interference from language modeling. We then convert each sampled transcript into its corresponding IDS representation as the training target. The maximum decoded IDS sequence length is set to 100 tokens.

Baselines and metrics. Baselines include the frozen model, OCR-only GRPO, and TextPecker-guided GRPO [32]. We compare them with IDSpect-guided GRPO. Following TextPecker’s Chinese evaluation protocol, we report results on LongText-Bench [7] and GenTextEval-Bench [32] using the original benchmark text score (Avg.), structural quality (Qua.), and semantic alignment (Sem.). The published Qwen-Image, OCR-reward, and TextPecker-reward results are included for direct comparison. IDSpect is evaluated using the same benchmark protocol.

## 4.2 Experimental Results

Quantitative results. Tab. 1 summarizes the quantitative comparison on LongText and GenTextEval. Relative to the frozen Qwen-Image model, OCR-guided GRPO substantially improves the original benchmark text score as well as both TextPecker-based structural quality and semantic alignment. On LongText, the original text score increases from 0.920 to 0.967, while Qua. and Sem. improve from 0.924 and 0.834 to 0.956 and 0.886, respectively. Replacing the OCR reward with the TextPecker reward brings further improvements, achieving an Avg. score of 0.974 on LongText together with Qua. and Sem. scores of 0.969 and 0.908. On GenTextEval, TextPecker-guided GRPO reaches 0.973 Qua. and 0.897 Sem. Our IDSpect further improves both structural and semantic metrics, achieving 0.979 Qua. and 0.911 Sem. on GenTextEval. These results exceed the OCR reward by 0.026 and 0.037, respectively, and improve over the TextPecker reward by 0.006 and 0.014. On LongText, IDSpect achieves the highest Qua. and Sem. scores of 0.975 and 0.928, respectively, while maintaining an Avg. score of 0.972 that is comparable to TextPecker. These results indicate that IDS-based supervision provides a finer-grained signal for Chinese character rendering. Rather than treating each character as a single categorical recognition target, the IDS reward evaluates whether its internal components and spatial composition are correctly rendered, which is particularly beneficial for correcting localized structural errors that may be overlooked by conventional OCR-based rewards.

Qualitative results. Fig. 3 compares the frozen Qwen-Image model with OCR-, TextPecker-, and IDSpect guided GRPO on three prompts containing multiple Chinese text regions with diferent lengths and spatial arrangements. The frozen model frequently produces incomplete or corrupted characters, particularly in longer text lines. OCR-guided GRPO improves the overall readability of the generated text, while TextPecker-guided GRPO further improves character-level fidelity. In comparison, IDSpect renders the requested main and secondary text more completely and better preserves the internal structure of individual Chinese characters. The improvement is especially apparent in longer lines on signs and posters, where structural errors can accumulate across characters and become dificult to capture with a coarse text-level reward. At the same time, IDSpect maintains coherent scene layouts and the intended spatial organization of multiple text regions. These observations are consistent with the quantitative results and suggest that the IDS reward provides more detailed structural feedback during GRPO optimization.

![](images/74c65eae7814e7cff863a5584212d2bcaaf8bd5f9246eae2d2fec74237b24a62.jpg)  
Figure 3 Qualitative comparison of Qwen-Image and OCR-, TextPecker-, and IDSpect-guided GRPO on Chinese text-rendering prompts. The rightmost column lists the target text.

## 4.3 Ablation of IDS reward construction

IDS reference. We first compare two types of IDS references. Per-crop consistency uses the IDS prediction of the OCR transcript recognized from each generated crop as the reference. This formulation measures whether the visual content is structurally consistent with the OCR result, but does not directly enforce agreement with the requested target text. In contrast, the other three variants use the canonical IDS decomposition of the target text as the reference, which directly connects the structural reward to the desired output. The semantic OCR term remains target-conditioned in all experiments. Per-crop consistency achieves the highest Qua. score of 0.980, showing that comparison against an OCR-derived structural representation provides a strong signal for local structural quality. However, its Sem. score is 0.901, which is lower than the 0.911 achieved by our target-based crop-wise alignment. This gap indicates that structural consistency with the OCR prediction does not necessarily guarantee alignment with the requested text, motivating the use of target IDS as the reference for structural supervision.

Target-based comparison. We next compare three strategies for matching predicted IDS sequences against the target IDS representation. Direct concatenation compares crop-level predictions with the target IDS sequence according to detector order, making the reward sensitive to mismatches between detector order and target text order. Character-wise matching follows the Chinese matching strategy released with TextPecker [32], which reduces dependence on crop ordering but treats characters independently and therefore discards useful inter-character ordering information. Our crop-wise alignment instead matches each detected text crop to a corresponding span of the target text while preserving the order of IDS tokens within each crop and enforcing unique target-token credit. As shown in Tab. 2, crop-wise alignment achieves the highest Sem. score of 0.911, improving over direct concatenation and character-wise matching by 0.025 and 0.012, respectively. Its Qua. score reaches 0.979, only 0.001 below per-crop consistency while providing substantially stronger semantic alignment. These results demonstrate that the efectiveness of the IDS reward depends not only on the structural representation itself but also on how the predicted structures are aligned with the target. Target-based supervision grounds the structural reward in the requested text, while crop-wise alignment preserves the correspondence between local text regions and their target spans.

Table 2 Ablation of IDS reward construction on GenTextEval. All variants use the same balanced IDS recognizer, semantic OCR term, and training configuration.
<table><tr><td>IDS reference</td><td>Comparison</td><td>Qua. ↑</td><td>Sem. ↑</td></tr><tr><td>OCR prediction</td><td>Per-crop consistency</td><td>0.980</td><td>0.901</td></tr><tr><td>Target text</td><td>Direct concatenation</td><td>0.976</td><td>0.886</td></tr><tr><td>Target text</td><td>Character-wise matching</td><td>0.971</td><td>0.899</td></tr><tr><td>Target text</td><td>Crop-wise alignment (ours)</td><td>0.979</td><td>0.911</td></tr></table>

## 5 Conclusion

In this paper, we addressed a fundamental limitation of OCR-based rewards for Chinese visual text rendering: treating each ideograph as an atomic label provides insuficient credit for partial improvements to its internal structure. We introduced IDSpect, a target-conditioned compositional reward that represents Chinese text with Ideographic Description Sequences and combines character-level semantic fidelity with fine-grained supervision over components and spatial operators. Experiments with GRPO post-training of Qwen-Image show that IDSpect improves structural quality and semantic alignment on LongText-Bench and GenTextEval-Bench, outperforming general OCR-based rewards and providing stronger supervision than glyph-level anomaly-aware evaluation. These results demonstrate that decomposing ideographs into reusable components and explicit spatial relations ofers a practical route to more faithful Chinese text generation. In future work, we plan to extend the framework to broader Chinese character coverage and multilingual scripts like Japanese and Korean.

## References

[1] Jingye Chen, Bin Li, and Xiangyang Xue. Zero-shot chinese character recognition with stroke-level decomposition. In ĲCAI, 2021.

[2] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdifuser: difusion models as text painters. In NeurIPS, 2023.

[3] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdifuser-2: Unleashing the power of language models for text rendering. In ECCV, 2024.

[4] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. CoRR, abs/2507.05595, 2025.

[5] Mingxuan Cui, Jingpu Yang, Fengxian Ji, Qian Jiang, Zhecheng Shi, Jiaming Wang, Zirui Song, Fajri Koto, and Xiuying Chen. Textalign: Preference alignment for text rendering with hierarchical rewards. arXiv preprint arXiv:2605.19320, 2026.

[6] Yongkun Du, Zhineng Chen, Hongtao Xie, Caiyan Jia, and Yu-Gang Jiang. Svtrv2: Ctc beats encoder-decoder models in scene text recognition. In ICCV, 2025.

[7] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025.

[8] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

[9] Gengluo Li, Xingyu Wan, Shangpin Peng, Weinong Wang, Hao Feng, Yongkun Du, Binghong Wu, Zheng Ruan, Zhiqiong Lu, Liang Wu, et al. Hunyuanocr-1.5: Making lightweight ocr vlms faster and better. arXiv preprint arXiv:2607.04884, 2026.

[10] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di ZHANG, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In NeurIPS, 2025.

[11] Jian Ma, Yonglin Deng, Chen Chen, Nanyang Du, Haonan Lu, and Zhenyu Yang. Glyphdraw2: Automatic generation of complex glyph posters with difusion models and large language models. In AAAI, 2025.

[12] Bowen Ping, Chengyou Jia, Minnan Luo, Hangwei Qian, and Ivor Tsang. Flow-factory: A unified framework for reinforcement learning in flow-matching models. arXiv preprint arXiv:2602.12529, 2026.

[13] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In CVPR, 2022.

[14] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[15] Fenfen Sheng, Zhineng Chen, and Bo Xu. Nrtr: A no-recurrence sequence-to-sequence model for scene text recognition. In ICDAR, 2019.

[16] Xincheng Shuai, Ziye Li, Henghui Ding, and Dacheng Tao. Glyphprinter: Region-grouped direct preference optimization for glyph-accurate visual text rendering. In CVPR, 2026.

[17] State Administration for Market Regulation and Standardization Administration of China. GB 18030–2022: Information Technology—Chinese Coded Character Set. National Standard of the People’s Republic of China, 2022. URL https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=A1931A578FE14957104988029B0833D3.

[18] The Unicode Consortium. The Unicode Standard, Version 16.0: Core Specification, Chapter 18, East Asia. https://unicode.org/versions/Unicode16.0.0/core-spec/chapter-18/, 2024.

[19] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245, 2024.

[20] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. In ICLR, 2024.

[21] Andrew West. BabelStone IDS Database. https://www.babelstone.co.uk/CJK/IDS.HTML, 2025.

[22] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

[23] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, and Tao Mei. 3dstyle-difusion: Pursuing fine-grained text-driven 3d stylization with 2d difusion models. In ACM MM, 2023.

[24] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, Chong-Wah Ngo, and Tao Mei. Hi3d: Pursuing high-resolution image-to-3d generation with video difusion models. In ACM MM, 2024.

[25] Xingsong Ye, Yongkun Du, Yunbo Tao, and Zhineng Chen. Textssr: Difusion-based data synthesis for scene text recognition. In ICCV, 2025.

[26] Xingsong Ye, Yongkun Du, JiaXin Zhang, Chen Li, Jing LYU, and Zhineng Chen. What’s wrong with synthetic data for scene text recognition? a strong synthetic engine with diverse simulations and self-evolution. In CVPR, 2026.

[27] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Zhixian Li, Chong Sun, Chen Li, Jing Lyu, Lianwen Jin, and Zhineng Chen. All-in-one multilingual scene text recognition with script-aware mixture-of-experts. arXiv preprint arXiv:2609.24058, 2026.

[28] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Haojie Zhang, Chong Sun, Chen Li, Jing Lyu, and Zhineng Chen. Advancing wordart-oriented scene text recognition: Datasets and methods. In ECCV, 2026.

[29] Haiyang Yu, Xiaocong Wang, Bin Li, and Xiangyang Xue. Chinese text recognition with a pre-trained clip-like model through image-ids aligning. In ICCV, 2023.

[30] Jianshu Zhang, Yixing Zhu, Jun Du, and Lirong Dai. Radical analysis network for zero-shot learning in printed chinese character recognition. In ICME, 2018.

[31] Peirong Zhang, Haowei Xu, Jiaxin Zhang, Xuhan Zheng, Guitao Xu, Yuyi Zhang, Junle Liu, Zhenhua Yang, Wei Zhou, and Lianwen Jin. Ocrgenbench: A comprehensive benchmark for evaluating ocr generative capabilities. arXiv preprint arXiv:2507.15085, 2025.

[32] Hanshen Zhu, Yuliang Liu, Xuecheng Wu, An-Lan Wang, Hao Feng, Dingkang Yang, Chao Feng, Can Huang, Jingqun Tang, and Xiang Bai. Textpecker: Rewarding structural anomaly quantification for enhancing visual text rendering. In CVPR, 2026.

[33] Yinglian Zhu, Haiyang Yu, Qizao Wang, Wei Lu, Xiangyang Xue, and Bin Li. Zero-shot chinese character recognition with hierarchical multi-granularity image-text aligning. arXiv preprint arXiv:2505.24837, 2025.