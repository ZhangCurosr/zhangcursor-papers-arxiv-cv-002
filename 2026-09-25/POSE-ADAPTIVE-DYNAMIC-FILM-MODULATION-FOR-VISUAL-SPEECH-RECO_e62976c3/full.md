# POSE ADAPTIVE DYNAMIC FILM MODULATION FOR VISUAL SPEECH RECOGNITION

Matthew Kit Khinn Teng, Haibo Zhang, Takeshi Saitoh

Kyushu Institute of Technology, Japan Kyushu Institute of Technology, Fukuoka 820-8502, Japan

## ABSTRACT

Head-pose variation introduces substantial appearance transformations in visual speech recognition (VSR), making pose-aware feature modulation desirable. However, performance degradation and unwanted feature interactions may result from using numerous Feature-wise Linear Modulation (FiLM) circuits with fixed modulation intensity. We propose a Pose Adaptive Dynamic FiLM framework with a Dynamic Residual FiLM (DR-FiLM) modulator that predicts input-dependent weights to adaptively control the strength of pose-conditioned modulation. Experiments on LRS2 and LRS3 demonstrate that unweighted multi-pathway modulation substantially degrades phoneme recognition, increasing PER to 20.33% and 29.42%, respectively, compared with 16.20% and 20.96% for the single ResFiLM configuration. In contrast, the proposed DR-FiLM with dynamic Deep–Res weighting reduces PER to 15.74% on LRS2 and 23.91% on LRS3, substantially mitigating the adverse effects of unweighted modulation. The analysis of the learned weights further reveals a consistent tendency to assign greater weight to the deeper FiLM pathway as head-pose variation increases. These results show that merging pose-conditioned FiLM circuits is more efficient when the modulation strength is dynamically controlled.

Index Terms— Lip-reading, Deep Learning, Featurewise Linear Modulation (FiLM), Dynamic Feature Modulation, Adaptive Multi-Pathway Routing

## 1. INTRODUCTION

Lip-reading is a form of V-ASR that derives spoken words from visual cues of facial articulatory features, particularly lip movements. Many existing VSR approaches directly predict words or entire sentences from visual speech [1–5]. However, direct word-level prediction can be challenged by speaker-dependent variations, large vocabularies, and subtle visual differences between words. This has motivated phonetic representations, in which phoneme-level prediction provides a more fine-grained representation of visual speech and reduces word-level ambiguity. Despite this advantage, phoneme-level VSR remains sensitive to appearance variations caused by head-pose changes. Such variations alter the spatial configuration and appearance of the lip region, making recognition more challenging under large or unconstrained viewing angles. Prior studies have addressed this difficulty through multi-view representation learning [6–9] and poseoriented data augmentation or synthetic view generation [10– 13]; however, these approaches either rely on predefined viewpoints or synthetic transformations, rather than explicitly modeling pose-induced representation changes within the VSR network.

To solve this limitation, our prior research presented HP-VSR-ResFiLM [14], which conditions visual feature modulation on estimated head posture. While it generates sample-adaptive modulation parameters (α and γ) based on head pose, it applies these modulations through a rigid, unweighted structure where all FiLM pathways contribute uniformly regardless of posture variation. This fixed weighting setup may not be optimal for all utterances, as varied head-pose conditions may necessitate different levels of feature modification across layers.This raises a crucial question: can the model dynamically select or weight which poseconditioned feature pathways are best suited to each input?

Motivated by this observation, our proposed framework extends HP-VSR-ResFiLM [14] by adaptively weighting multiple pose-conditioned FiLM pathways according to the input. The main contributions of this study are: 1) We introduce a Dynamic Residual FiLM (DR-FiLM) Modulator with a dynamic weighting mechanism that adaptively controls the modulation strength of multiple pose-conditioned FiLM pathways. 2) We study several configurations of Deep FiLM and ResFiLM pathways and show that adaptive weighting can reduce performance loss caused by unweighted multi-pathway modulation. 3) We examine the acquired dynamic weights over several yaw ranges, gaining insight into how the model adjusts its dependence on distinct FiLM routes under varied head-position situations.

## 2. METHODOLOGY

Framework Overview. The proposed framework is shown in Figure 1. The visual and head-pose encoders retain the same architecture as in [14]. In the visual pathway, the existing FiLM modules at 2D ResNet 18 frontend Layers 3 and 4 (Deep FiLM) are retained, and an additional ResBlock-FiLM (ResFiLM) module is added after the 2D ResNet 18. Moreover, the encoded head-pose features are used by two components: a FiLM generator that produces the modulation parameters $\gamma$ and $\beta ,$ and a residual weight generator that predicts input-dependent weights $w _ { \mathrm { d e e p } }$ and $w _ { \mathrm { r e s } }$ for the Deep FiLM and ResFiLM pathways, respectively. These parameters and weights are used by the proposed DR-FiLM modulator to adaptively control the strength of pose-conditioned modulation. The resulting visual features are subsequently combined with the head-pose features through an MLP, while the remaining phoneme prediction architecture follows the previous framework.

![](images/41a65d6be42f8941b0e503bd3a3e7172e6dca2bc392b09ae2250217e9cf82b19.jpg)  
Fig. 1. Overview of the proposed architecture. Head-pose features dynamically determine the contribution of the Deep FiLM pathways at ResNet 18 Layers 3–4 and the ResFiLM pathway for pose-adaptive feature modulation.

Residual Weight Generator. The residual weight generator predicts the relative modulation strength of the Deep FiLM and ResFiLM (Deep + Res) pathways from the encoded headpose feature. It consists of two linear layers with an intermediate ReLU activation. Given the head-pose feature h, the generator produces two unnormalized weights:

$$
[ \tilde { w } _ { \mathrm { d e e p } } , \tilde { w } _ { \mathrm { r e s } } ] = f _ { \mathrm { w } } ( \mathbf { h } ) ,\tag{1}
$$

which are normalized using a softmax function:

$$
[ w _ { \mathrm { d e e p } } , w _ { \mathrm { r e s } } ] = \mathrm { s o f t m a x } ( [ \tilde { w } _ { \mathrm { d e e p } } , \tilde { w } _ { \mathrm { r e s } } ] ) .\tag{2}
$$

Thus,

$$
w _ { \mathrm { d e e p } } + w _ { \mathrm { r e s } } = 1 .\tag{3}
$$

These input-dependent weights are provided to the DR-FiLM Modulator to adaptively control the strength of poseconditioned modulation in each pathway.

DR-FiLM Modulator. This modulator combines poseconditioned feature modulation with input-dependent modulation strength. Two pose-conditioned modulation pathways are employed: a Deep FiLM pathway applied at ResNet 18 Layers 3 and 4, and an additional ResFiLM pathway.

For the 2D ResNet 18 (Deep FiLM) pathway, the proposed DR-FiLM operation is defined as

$$
{ \bf F } _ { \mathrm { D R - F i L M } } ^ { \mathrm { d e e p } } = { \bf F } _ { \mathrm { o u t } } ^ { \mathrm { d e e p } } + w _ { \mathrm { d e e p } } \left( { \bf F } _ { \mathrm { o u t } } ^ { \mathrm { d e e p } } \odot \gamma _ { \mathrm { d e e p } } + \beta _ { \mathrm { d e e p } } \right) ,\tag{4}
$$

where $\mathbf { F } _ { \mathrm { o u t } } ^ { \mathrm { d e e p } }$ denotes the output feature of the corresponding convolutional transformation. This operation is applied sequentially at 2D ResNet 18 Layers 3 and 4.

Similarly, the additional ResBlock-FiLM (ResFiLM) pathway is defined as

$$
\mathbf { F } _ { \mathrm { D R - F i L M } } ^ { \mathrm { r e s } } = \mathbf { F } _ { \mathrm { o u t } } ^ { \mathrm { r e s } } + w _ { \mathrm { r e s } } \left( \mathbf { F } _ { \mathrm { o u t } } ^ { \mathrm { r e s } } \odot \gamma _ { \mathrm { r e s } } + \beta _ { \mathrm { r e s } } \right) .\tag{5}
$$

Unlike conventional FiLM [15] and directly gated FiLM formulations [16], which were originally developed for different applications, the proposed DR-FiLM adopts their feature modulation formulation while introducing a dynamically weighted pose-conditioned modulation term for VSR. Specifically, DR-FiLM retains the original feature representation as a base term and introduces dynamically weighted modulation through the Deep FiLM and ResFiLM pathways. The dynamic weights $w _ { \mathrm { d e e p } }$ and $w _ { \mathrm { r e s } }$ therefore control the strength of pose-conditioned modulation in the respective pathways.

Absolute-Pose FiLM Baseline. To investigate whether the performance of dynamic weighting can be attributed directly to head-pose magnitude, we additionally construct an absolute-pose-based FiLM baseline (Abs FiLM). Unlike DR-FiLM, the weights in Abs FiLM are not predicted by the residual weight generator. Instead, they are assigned deterministically based on the average absolute yaw angle across all frames. Specifically, let $\begin{array} { r } { p = \frac { 1 } { T } \sum \left| \mathbf { y } \mathbf { a w } _ { t } \right| } \end{array}$ denote the mean absolute yaw over T frames, and define

$$
\alpha = \left\{ \begin{array} { l l } { 0 , } & { p = 0 ^ { \circ } , } \\ { p / 3 0 ^ { \circ } , } & { 0 ^ { \circ } < p < 3 0 ^ { \circ } , } \\ { 1 , } & { p \geq 3 0 ^ { \circ } . } \end{array} \right.\tag{6}
$$

Two complementary weighting directions are considered. For Abs FiLM (R→D), the weights are defined as $w _ { \mathrm { r e s } } = 1 - \alpha$ and $w _ { \mathrm { d e e p } } = \alpha$ , whereas for Abs FiLM (D→R), they are reversed as $w _ { \mathrm { r e s } } = \alpha$ and $w _ { \mathrm { d e e p } } = 1 - \alpha$ . Thus, the two variants respectively shift the weighting from ResFiLM to Deep FiLM and from Deep FiLM to ResFiLM as the absolute yaw magnitude increases, while maintaining $w _ { \mathrm { d e e p } } + w _ { \mathrm { r e s } } = 1$

Table 1. Performance comparison (PER % and WER %) of representative VSR methods and our proposed framework on the LRS2 and LRS3 datasets. The total hours indicate the combined duration or size of the pretraining and training datasets.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Modality</td><td rowspan="2">FiLM Layers</td><td colspan="2">LRS2</td><td colspan="2">LRS3</td></tr><tr><td>Total Hours</td><td>WER↓</td><td>Total Hours</td><td>WER↓</td></tr><tr><td>Hyb.-Conf. [6]</td><td>Video</td><td></td><td>223</td><td>39.1</td><td>438</td><td>46.9</td></tr><tr><td>Hyb.-Conf. [6]</td><td>Video</td><td></td><td>381</td><td>37.9</td><td>590</td><td>43.3</td></tr><tr><td>VTP [1]</td><td>Video</td><td></td><td>2676</td><td>22.6</td><td>2676</td><td>30.7</td></tr><tr><td>Auto-AVSR [2]</td><td>Video</td><td></td><td>818</td><td>27.9</td><td>818</td><td>33.0</td></tr><tr><td>Auto-AVSR [2]</td><td>Video</td><td></td><td>3448</td><td>14.6</td><td>3448</td><td>19.1</td></tr><tr><td>CM-aux [17]</td><td>Video</td><td></td><td>223</td><td>32.9</td><td>438</td><td>37.9</td></tr><tr><td>SyncVSR [18]</td><td>Video†</td><td></td><td>223</td><td>28.9</td><td>438</td><td>31.2</td></tr><tr><td>GLip [19]</td><td>Video</td><td></td><td>223</td><td>27.4</td><td>438</td><td>30.1</td></tr><tr><td>PV-ASR [14]</td><td>Video + 117-Point</td><td></td><td>223</td><td>26.6</td><td>438</td><td>36.7</td></tr><tr><td>HP-VSR-ResFiLM [14]</td><td>Video + Headpose</td><td>ResFiLM</td><td>223</td><td>24.7</td><td>438</td><td>30.3</td></tr><tr><td>HP-VSR-FiLMFuse [14]</td><td>Video + Headpose</td><td>L3-L4</td><td>223</td><td>25.4</td><td>438</td><td>32.5</td></tr><tr><td>Ours (Baseline)</td><td>Video + Headpose</td><td> $_ { \mathrm { L 3 - L 4 + R e s F i L M } }$ </td><td>223</td><td>28.8</td><td>438</td><td>39.0</td></tr><tr><td>Ours (Abs FiLM (R→D))</td><td>Video + Headpose</td><td> $_ { \mathrm { L 3 - L 4 + R e s F i L M } }$ </td><td>223</td><td>27.6</td><td>438</td><td>33.9</td></tr><tr><td>Ours (DR-FiLM (Deep + Res))</td><td>Video + Headpose</td><td> $_ { \mathrm { L 3 - L 4 + R e s F i L M } }$ </td><td>223</td><td>23.2</td><td>438</td><td>34.4</td></tr></table>

<sup>†</sup>Audio used only as an auxiliary training signal (crossmodal token prediction); inference is video-only.

## 3. EXPERIMENTAL SETUP

Datasets. Experiments are conducted on the publicly available English visual speech datasets LRS2 [20], LRS3 [21]. LRS2 contains approximately 144k utterances (224.5 h), while LRS3 comprises approximately 152k utterances (438.9 h), with both datasets providing separate pretraining, training/validation, and test partitions.

Experimental Configuration. The training and evaluation protocols follow our previous HP-VSR framework [14], including the Stage 1 initialization, optimization settings, batch configuration, and checkpoint averaging. Stage 2 uses the same fixed pretrained NLLB model as in [22] without additional training or fine-tuning.

Evaluation Metrics. VSR performance is evaluated using Word Error Rate (WER) [23], which measures the proportion of word-level deletions, insertions, and substitutions relative to the reference text. Phoneme Error Rate (PER) is reported to assess recognition performance at the phoneme level.

## 4. EXPERIMENTAL RESULTS

Comparison of Recent Representative Methods. Table 1 shows that our DR-FiLM (Deep + Res) configuration achieves the strongest LRS2 result among all video-only methods at comparable data scale (23.2% WER), outperforming GLip [19], CM-aux [17], and our own previous work, HP-VSR-ResFiLM [14], while substantially recovering the degradation observed in the unweighted multi-pathway configuration (28.8% → 23.2% on LRS2, 39.0% → 34.4% on LRS3). This confirms that dynamic, input-dependent weighting is essential when combining multiple pose-conditioned FiLM pathways, consistent with our ablation findings in Table 2. However, on LRS3, DR-FiLM (34.4%) does not surpass GLip (30.1%) or our prior single-location ResFiLM variant (30.3%), despite its clear advantage on LRS2. We attribute this discrepancy to differences in recording conditions between the two benchmarks: LRS2’s BBC broadcast footage and LRS3’s TED-talk footage differ in pose distribution, lighting, and speaker framing, which may affect how the dual-pathway modulation interacts with each domain. We view improving cross-dataset consistency of DR-FiLM as an important direction for future work.

Table 2. Ablation study of DR-FiLM configurations and dynamic weighting on LRS2 and LRS3. D denotes the Deep FiLM pathway applied at ResNet 18 Layers 3–4, while R denotes the ResFiLM pathway applied after the 2D CNN frontend. R→D shifts the weighting from ResFiLM toward Deep FiLM as the absolute yaw angle increases, whereas D→R applies the reverse weighting.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Dynamic Weights</td><td colspan="2">LRS2</td><td colspan="2">LRS3</td></tr><tr><td>PER↓</td><td>WER↓</td><td>PER↓</td><td>WER↓</td></tr><tr><td>Baseline</td><td>x</td><td>20.33</td><td>28.84</td><td>29.42</td><td>38.98</td></tr><tr><td>DR-FiLM (Layer-wise)</td><td>√</td><td>20.14</td><td>29.01</td><td>25.88</td><td>35.54</td></tr><tr><td>DR-FiLM (Deep + Res)</td><td>√</td><td>15.74</td><td>23.24</td><td>23.91</td><td>34.43</td></tr><tr><td>DR-FiLM (L4 + Res)</td><td>√</td><td>16.79</td><td>24.98</td><td>27.08</td><td>37.58</td></tr><tr><td>Abs FiLM (R→D)</td><td>√</td><td>18.79</td><td>27.55</td><td>23.67</td><td>33.90</td></tr><tr><td>Abs FiLM (D→R)</td><td>√</td><td>19.76</td><td>28.44</td><td>23.77</td><td>33.82</td></tr></table>

Analysis of DR-FiLM Configurations.Table 2 evaluates various dynamic-weighting configurations—including layer-wise weighting $( w _ { 3 } + w _ { 4 } + w _ { \mathrm { r e s } } = 1 )$ and L4+Res weighting $( w _ { 4 } \ + \ w _ { \mathrm { r e s } } \ = \ 1 ) { \ - \mathrm { - t o } }$ analyze the impact of weighting granularity and modulation depth. Simply combining multiple FiLM pathways without adaptive weighting degrades performance significantly. For instance, the unweighted L3–L4 + ResFiLM baseline increases error rates on LRS2 (PER/WER rising from 16.20%/24.68% to 20.33%/28.84%) and LRS3 (from 20.96%/30.32% to 29.42%/38.98% compared to single HP-VSR-ResFiLM). This indicates that concurrently activating multiple poseconditioned routes without coordination leads to feature interference. Dynamic weighting effectively mitigates this degradation, though performance varies by setup. While the layerwise configuration yields mixed results (20.14%/29.01% on LRS2; 25.88%/35.54% on LRS3), the proposed DR-FiLM (Deep + Res) configuration achieves the best overall performance among dynamic variants (15.74%/23.24% on LRS2; 23.91%/34.43% on LRS3). Comparing this with the DR-FiLM (L4+Res) setup (16.79%/24.98% and 27.08%/37.58%) confirms that jointly weighting the deeper L3–L4 layers is superior to restricting adaptation to L4 alone. Finally, the Abs-FiLM variants test deterministic weighting based on absolute yaw. While both directional rules $( R \to D \colon$ 18.79%/27.55% and 23.67%/33.90%; $D \ \to \ R :$ 19.76%/28.44% and 23.77%/33.82%) outperform the unweighted baseline, they remain inferior to learned DR-FiLM (Deep + Res). This demonstrates that input-dependent learned weighting is more robust than a rigid, heuristic poserule. Overall, DR-FiLM (Deep + Res) successfully recovers the degradation caused by unweighted multi-pathway modulation.

Table 3. Average DR-FiLM (Deep + Res) weights and PER across different head-pose ranges on LRS2 and LRS3. $w _ { \mathrm { d e e p } }$ and $w _ { \mathrm { r e s } }$ denote the weights assigned to the deeper L3–L4 FiLM and ResFiLM branches, respectively. The standard deviation (Std.) is identical for both weights because $w _ { \mathrm { d e e p } } + w _ { \mathrm { r e s } } = 1$
<table><tr><td>Dataset</td><td>Yaw Pose</td><td> $\operatorname { A v g } .$   $w _ { \mathrm { d e e p } }$ </td><td> $\operatorname { A v g } .$   $w _ { \mathrm { r e s } }$ </td><td>Std.</td><td>Avg. PER</td><td>No. of Samples</td></tr><tr><td>LRS2</td><td> $< 1 5 ^ { \circ }$ </td><td>0.7948</td><td>0.2052</td><td>0.0043</td><td>14.03</td><td>693</td></tr><tr><td></td><td> $1 5 { - } 3 0 ^ { \circ }$ </td><td>0.8027</td><td>0.1973</td><td>0.0068</td><td>17.50</td><td>341</td></tr><tr><td></td><td> $\geq 3 0 ^ { \circ }$ </td><td>0.8234</td><td>0.1766</td><td>0.0162</td><td>19.36</td><td>209</td></tr><tr><td>LRS3</td><td> $< 1 5 ^ { \circ }$ </td><td>0.7586</td><td>0.2414</td><td>0.0031</td><td>23.22</td><td>430</td></tr><tr><td></td><td> $1 5 { - } 3 0 ^ { \circ }$ </td><td>0.7631</td><td>0.2369</td><td>0.0050</td><td>23.35</td><td>488</td></tr><tr><td></td><td> $\geq 3 0 ^ { \circ }$ </td><td>0.7794</td><td>0.2206</td><td>0.0168</td><td>25.42</td><td>403</td></tr></table>

Analysis of Dynamic Weights. Table 3 presents the average learned dynamic weights across different head-pose ranges for the LRS2 and LRS3 datasets. On LRS2, the average $w _ { \mathrm { d e e p } }$ increases from 0.7948 for samples with $| y a w | < 1 5 ^ { \circ }$ to 0.8027 for $1 5 ^ { \circ } \leq | y a w | < 3 0 ^ { \circ }$ and 0.8234 for $| y a w | \geq 3 0 ^ { \circ }$ A similar trend is observed on LRS3, where $w _ { \mathrm { d e e p } }$ increases from 0.7586 to 0.7631 and 0.7794 across the same pose ranges. Correspondingly, $w _ { \mathrm { r e s } }$ decreases as the pose range increases, since the two weights are constrained by $w _ { \mathrm { d e e p } } + w _ { \mathrm { r e s } } = 1$

These findings show that the model assigns greater modulation strength to the deeper FiLM route under larger headpose fluctuations, while decreasing the relative strength of the ResFiLM pathway. This behavior indicates that the suggested dynamic weighting method learns to alter modulation strength based on the input pose, rather than applying a set weighting to all samples. The increase in $w _ { \mathrm { d e e p } }$ is more noticeable for highly posed samples, notably on LRS2. This suggests that deeper feature modulation may become more relevant as pose variation grows.

## 5. LIMITATIONS AND FUTURE WORK

Although our Dynamic FiLM framework mitigates unweighted multi-pathway degradation, it does not uniformly outperform the strongest HP-VSR-ResFiLM baseline across both benchmarks: while DR-FiLM (Deep + Res) surpasses the baseline on LRS2, a performance gap persists on LRS3. This indicates that dynamic weighting efficacy is sensitive to dataset characteristics like pose distribution and video quality.

Future work will explore more expressive weighting mechanisms, extend utterance-level routing to frame-level temporal adaptation for within-utterance pose changes, and incorporate additional geometry (pitch and roll). Finally, broader cross-domain evaluations will help verify generalization across diverse recording conditions.

## 6. CONCLUSIONS

This study proposes a Pose-Adaptive Dynamic FiLM framework with DR-FiLM to dynamically control multiple poseconditioned FiLM pathways for VSR. Experiments on LRS2 and LRS3 show that unweighted multi-pathway modulation can substantially degrade recognition performance, while dynamic weighting effectively mitigates this degradation. The proposed DR-FiLM (Deep + Res) configuration achieves PER/WERs of 15.74%/23.24% on LRS2 and 23.91%/34.43% on LRS3, demonstrating the effectiveness of dynamically balancing the contributions of Deep FiLM and ResFiLM. The analysis reveals that the pathway weights vary with head-pose severity, implying that DR-FiLM can adapt feature modulation according to pose conditions rather than relying on a fixed configuration. These results highlight dynamic poseconditioned weighting as an effective approach for improving the performance of phoneme-based VSR under varying head poses.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number JP23H03787.

## References

[1] K. R. Prajwal, T. Afouras, and A. Zisserman, “Sub-word level lip reading with visual attention,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5162–5172.

[2] P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, and M. Pantic, “Auto-AVSR: Audio-visual speech recognition with automatic labels,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[3] X. Liu, E. Lakomkin, K. Vougioukas, P. Ma, H. Chen, R. Xie, M. Doulaty, N. Moritz, J. Kolar, S. Petridis, M. Pantic, and C. Fuegen, “SynthVSR: Scaling up visual speech recognition with synthetic supervision,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 18 806–18 815.

[4] Y. A. D. Djilali, S. Narayan, H. Boussaid, E. Almazrouei, and M. Debbah, “Lip2Vec: Efficient and robust visual speech recognition via latent-to-latent visual to audio representation mapping,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 13 744–13 755.

[5] H. Laux, E. Mededovic, A. Hallawa, L. Martin, A. Peine, and A. Schmeink, “LITEVSR: Efficient visual speech recognition by learning from speech representations of unlabeled data,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 10 391–10 395.

[6] P. Ma, S. Petridis, and M. Pantic, “End-to-end audiovisual speech recognition with conformers,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 7613–7617.

[7] S. Isobe, S. Tamura, S. Hayamizu, Y. Gotoh, and M. Nose, “Multi-angle lipreading with angle classification-based feature extraction and its application to audio-visual speech recognition,” Future Internet, vol. 13, no. 7, p. 182, 2021.

[8] T. Maeda and S. Tamura, “Multi-view convolution for lipreading,” in 2021 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2021, pp. 1092–1096.

[9] S. Jeon and M. S. Kim, “End-to-end sentence-level multi-view lipreading architecture with spatial attention module integrated multiple cnns and cascaded local selfattention-ctc,” Sensors, vol. 22, no. 9, p. 3597, 2022.

[10] S. Cheng, P. Ma, G. Tzimiropoulos, S. Petridis, A. Bulat, J. Shen, and M. Pantic, “Towards pose-invariant lip-reading,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 4357–4361.

[11] B. Hao, D. Zhou, X. Li, X. Zhang, L. Xie, J. Wu, and E. Yin, “Lipgen: Viseme-guided lip video generation for enhancing visual speech recognition,” in ICASSP

2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[12] A. Fernandez-Lopez, H. Chen, P. Ma, A. Haliassos, S. Petridis, and M. Pantic, “Sparsevsr: Lightweight and noise robust visual speech recognition,” arXiv preprint arXiv:2307.04552, 2023.

[13] Z. Kang, M. Sadeghi, R. Horaud, and X. Alameda-Pineda, “Expression-preserving face frontalization improves visually assisted speech processing,” International Journal of Computer Vision, vol. 131, no. 5, pp. 1122–1140, 2023.

[14] M. K. K. Teng, H. Zhang, and T. Saitoh, “Head-poseaware visual speech recognition via residual film modulation,” arXiv preprint arXiv:2606.00751, 2026.

[15] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[16] X. Lin, J. Wu, C. Zhou, S. Pan, Y. Cao, and B. Wang, “Task-adaptive neural process for user cold-start recommendation,” in Proceedings of the web conference 2021, 2021, pp. 1306–1316.

[17] P. Ma, S. Petridis, and M. Pantic, “Visual speech recognition for multiple languages in the wild,” Nature Machine Intelligence, vol. 4, no. 11, pp. 930–939, 2022.

[18] Y. J. Ahn, J. Park, S. Park, J. Choi, and K.-E. Kim, “Syncvsr: Data-efficient visual speech recognition with end-to-end crossmodal audio token synchronization,” in Interspeech 2024. ISCA, 2024, pp. 867–871.

[19] T. Wang, S. Yang, S. Shan, and X. Chen, “Glip: a global-local integrated progressive framework for robust visual speech recognition,” arXiv preprint arXiv:2509.16031, 2025.

[20] J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, “Lip reading sentences in the wild,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 3444–3453.

[21] T. Afouras, J. S. Chung, and A. Zisserman, “LRS3- TED: a large-scale dataset for visual speech recognition,” arXiv preprint arXiv:1809.00496, 2018.

[22] M. K. K. Teng, H. Zhang, and T. Saitoh, “Phoneme-level visual speech recognition via point-visual fusion and language model reconstruction,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 10 477–10 481.

[23] F. Jelinek, L. Bahl, and R. Mercer, “Design of a linguistic statistical decoder for the recognition of continuous speech,” IEEE Transactions on Information Theory, vol. 21, no. 3, pp. 250–256, 1975.