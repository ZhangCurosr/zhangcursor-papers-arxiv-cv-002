# MR-JEPA: A General Purpose Video Foundation Model for Cardiac MRI

Athira J. Jacob<sup>1,2</sup> , Puneet Sharma<sup>1</sup> , Dorin Comaniciu<sup>1</sup> , and Daniel Rueckert<sup>2,3</sup>

<sup>1</sup> Digital Technology and Innovation, Siemens Healthineers, Princeton, NJ, USA

2 Chair for Al in Healthcare and Medicine, Technical University of Munich (TUM) and TUM University Hospital, Munich, Germany

Department of Computing, Imperial College London, UK

Abstract. Cardiac magnetic resonance imaging (CMR) produces rich sequential data such as temporal cine videos and spatial LGE/mapping stacks, yet most deep learning approaches process individual 2D slices, discarding this context. We present MR-JEPA, a self-supervised video foundation model for CMR that extends LeJEPA to 3D spatiotemporal inputs through tubelet tokenization, spatiotemporal masking augmentation, and initialization from a 2D CMR foundation model. Unlike prior CMR video models limited to cine data, MR-JEPA is pretrained on multi-sequence data (cine, LGE, mapping) from 10,505 patients across two centers without annotations. We evaluate the frozen encoder on six downstream tasks using a unified multi-view gated attention architecture: LV ejection fraction, RV ejection fraction, three myocardial strains (GLS, GCS, GRS), and four-class disease detection. MR-JEPA outperforms other compared methods on all five regression tasks, including both a domain-specific CMR model pretrained on more data with text supervision and a natural-video foundation model, achieving an LV EF MAE of 4.79% (r =0.764) and a GLS MAE of 1.87 (r=0.805), with 21-27% MAE reductions over baselines on strain tasks. For disease detection, MR-JEPA achieved a macro AUG of 0.868, remaining competitive with the domain-specific baseline despite using a fully self-supervised pretraining objective. These results demonstrate the potential of a unified video encoder for robust, multi-view utilization of diverse CMR sequences in clinical cardiac quantification and diagnosis.

Keywords: Cardiac MRI • Video Foundation Model • Strain Regression • Disease Detection.

## 1 Introduction

Cardiac magnetic resonance (CMR) provides complementary assessment of cardiac anatomy, function, and tissue characterization through multiple sequence types, including cine, late gadolinium enhancement (LGE), and parametric mapping. These sequences contain rich temporal and spatial information relevant to cardiac function, myocardial mechanics, and disease characterization [16].

Most deep learning methods for CMR operate on individual 2D slices [9], disregarding the sequential context inherent in cine and multi-slice acquisition. Recently, self-supervised video foundation models have demonstrated strong spatiotemporal representation learning from unlabeled data [4, 24]. However, adapting these to CMR remains challenging due to limited medical video data, heterogeneous imaging sequences, & the need to integrate information across multiple views and contrasts.

Prior work on CMR video foundation models has largely focused on cine imaging and often relies on auxiliary text supervision. In this work, we introduce MR-JEPA (Magnetic Resonance Joint-Embedding Predictive Architecture) (Fig. 1), a fully self-supervised video foundation model for CMR based on LeJEPA (Latent-Euclidean JEPA) [5]. We extend LeJEPA to spatiotemporal CMR inputs using tubelet embeddings and spatiotemporal masking, and leverage a pretrained 2D CMR FM for improved accuracy. We pretrain on 160,172 multi-sequence CMR clips from 10,505 patients, including cine, LGE and T1/T2 images. Using a unified frozen-encoder architecture with gated attention aggregation, MR-JEPA is evaluated on six downstream tasks spanning ventricular function, myocardial strain, and disease classification. MR-JEPA outperforms both a natural-video foundation model (V-JEPA2, [4]) and a prior CMR-specific foundation model [22] on all regression tasks while remaining competitive for disease classification, despite using 5x fewer pretraining videos and a smaller architecture.

## 2 Related Work

## 2.1 Self-Supervised Learning for Video

Self-supervised learning (SSL) has become a powerful paradigm for learning visual representations without labels. In images, contrastive and self-distillation methods such as DINO [20, 23] learn invariant representations from augmented views using a teacher-student framework. For video, two major SSL paradigms have emerged. Masked autoencoders, including VideoMAE [25,26], reconstruct masked spatiotemporal patches but may emphasize low-level details over semantic content. Joint-Embedding Predictive Architectures (JEPA) instead predict latent representations. I-JEPA [3] introduced this approach for images, while V-JEPA [6] and V-JEPA 2 [4] extended it to large-scale video learning, achieving strong performance on motion understanding tasks. However, these methods rely on momentum teachers, stop-gradients, and predictor networks. LeJEPA [5] recently removed these components, using a single shared encoder and Sketched Isotropic Gaussian Regularization (SIGReg) to prevent collapse, resulting in a simpler and theoretically grounded objective.

## 2.2 Foundation Models for Cardiac MRI

Several foundation models have been developed for cardiac MRI. Jacob et al. [14] introduced a DINO-based model trained on multi-sequence CMR, demonstrating benefits across segmentation and classification. However, it operates on 2D images and do not capture temporal dynamics. CineMA [12] used masked autoencoder pretraining on 15 million cine images, supporting segmentation, EF estimation, and disease detection. For video-based representation learning, Shad et al. [22] proposed a multimodal foundation model using contrastive learning over cine MRI & clinical reports. However, multimodal approaches require paired reportimage data, which is often unavailable and limits scalability.

![](images/fc2317236f575d95472662c75ffb7264ce65036abbbead96a77bdb65b2891f41.jpg)  
Fig. 1. Summary of the method - pretraining and finetuning for downstream tasks

## 3 Method

## 3.1 Pretraining

MR-JEPA extends LeJEPA [5], a joint-embedding self-supervised learning framework, from natural images to CMR video. Unlike teacher-student approaches [3, 4], LeJEPA trains a single shared encoder to learn invariant representations across multiple augmented views. Given projected embeddings $\{ \mathbf { z } _ { v } \} _ { v = 1 } ^ { V }$ of all V views, where the first $V _ { g }$ correspond to global crops, the objective is

$$
\mathcal { L } = \left( 1 - \lambda \right) \mathcal { L } _ { \mathrm { i n v } } + \lambda \mathcal { L } _ { \mathrm { S I G R e g } } ,\tag{1}
$$

with an invariance loss

$$
\mathcal { L } _ { \mathrm { i n v } } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } \left. \mathbf { z } _ { v } - \bar { \mathbf { z } } \right. ^ { 2 } , \quad \bar { \mathbf { z } } = \frac { 1 } { V _ { g } } \sum _ { v = 1 } ^ { V _ { g } } \mathbf { z } _ { v } ,\tag{<sup>2</sup>}
$$

and $\mathcal { L } _ { \mathrm { S I G R e g } }$ regularization term [5] that prevents representation collapse by encouraging an isotropic embedding distribution without negative pairs, momentum encoders, or predictor networks.

To adapt LeJEPA to CMR video, we introduce three modifications. First, the original 2D patch embedding is replaced with a 3D tubelet embedding that jointly encodes spatial and temporal information [4,2], Second, we apply spatiotemporal masking to local crops, randomly masking subsets of tubelet tokens to encourage fine-grained feature learning from incomplete inputs [4]. Third, the encoder is initialized from a 2D CMR foundation model pretrained using the DIN0v3 [23] objective (similar to [14]); 2D patch embeddings are inflated along the temporal dimension [8], while positional embeddings are replaced with sinusoidal spatiotemporal encodings [25]

The encoder is a Vision Transformer [11] adapted for video input via a 3D tubelet patch embedding. Input clips of consecutive grayscale CMR frames are divided into spatiotemporal tubelets, each spanning multiple frames and a spatial patch, yielding a sequence of tokens that jointly encode spatial and temporal information. A prepended classification [CLS] token aggregates global information, and sinusoidal positional encodings are applied separately along the spatial and temporal dimensions. The [CLS] output is passed through an MLP projector for the pretraining loss. MR-JEPA is pretrained on diverse CMR sequences, including cine, LGE, and T1/T2 mapping data. By jointly modeling temporal videos and spatial image stacks, the model learns a unified representation space capturing both cardiac dynamics and anatomical structure.

## 3.2 Downstream Tasks

We evaluate the frozen MR-JEPA encoder on six downstream tasks spanning cardiac function assessment and disease classification.

Cardiac Function Regression. Five regression tasks are considered: (1) left ventricular ejection fraction (LV EF), estimated from five cine views (2CH, 4CH, and basal, mid, and apical SAX); (2) right ventricular ejection fraction (RV EF), from 4CH and mid-SAX views; and three strain measures: (3) global longitudinal strain (GLS) from 2CH and 4CH views, and (4-5) global circumferential strain (GCS) and global radial strain (GRS) from 2CH, 4CH, and mid-SAX views.

Cardiac Disease Classification. The sixth task classifies patients into four categories: normal, dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), and ischemic heart disease (IHD). Inputs comprise three cine views (2CH, 4CH, mid-SAX) and a single-shot LGE (PSIR) view.

For all tasks, the frozen encoder extracts a [CLS] embedding from each view. View-specific embeddings are projected and aggregated using gated attention [13], enabling task-dependent weighting of complementary views. A taskspecific linear head then produces either a scalar regression output or class probabilities.

## 4 Data and Experiments

## 4.1 Data

Pretraining data. MR-JEPA is pretrained on CMR data from two clinical centers (Center 1 and Center 2), comprising 10,505 unique patients and 160,172 video clips. The data was acquired between 2013 and 2022 on 1.5T and 3T Siemens scanners (MAGNETOM Aera, Avanto, and Skyra). Imaging protocols comprised cine bSSFP acquisitions in standard short- and long-axis views, LGE imaging using Tl-weighted inversion-recovery sequences following gadolinium administration. Additional sequences included Tl- and T2-weighted imaging, T2 mapping, T2\* imaging etc. in Center 1, but not in Center 2. The pretraining dataset consisted predominantly of cine acquisitions (77%), followed by LGE (23%) and a small proportion of mapping data (0.4%), reflecting the distribution of routine clinical CMR examinations.

## Downstream data

— LV EF: Kaggle Second Annual Data Science Bowl [19] (697 patients, USA), with expert-annotated LV volumes; split: 487/196/427. Inputs to the model consisted of 5 cine views (each view is one video): 2CH, 4CH, and 3 slices from SAX at base, mid and apex.

RV EF, GLS, GCS, GRS: Center 1 (994/112/273 patients); Reference values for the measurements were obtained using a previously validated CMR analysis pipeline [10] that performed cardiac chamber segmentations on SAX and LAX views, with testing dice of LV 96.2%, myocardium 89.4%, RV 92.9%, LA 92.8%, RA 94.7% on a test set consisting of more than 1700 patients [10]. The derived measurements, including RV EF and strains, were further validated in [1, 7, 21, 15]. The input views for each individual task were chosen to be appropriate to the particular task. GCS and GRS used three input cine views: 2CH, 4CH and one SAX slice at mid-stack. GLS used 2CH and 4CH cine videos. RV EF used two views: 4CH and one SAX slice at mid-stack.

— Disease detection: Center 1 (942/108/256 patients). Patients were classified as normal, ischemic heart disease (IHD), hypertrophic cardiomyopathy (HCM), or dilated cardiomyopathy (DCM) by two experienced cardiologists. Class distribution was 42% normal, 33% IHD, 13% HCM, and 12% DCM. The input views consisted of three cine videos (4CH, 2CH, SAX mid-slice) and one LGE stack (singleshot stack covering the whole heart).

Data for Model Analyses. We study the effect of various design choices using two auxiliary tasks (Refer to Section: 4.3 for more details):

Cine view classification: six-class classification (SAX, 2CH, 3CH, 4CH, aorta, other) on 471 cine sequences from 133 patients (Center 3), split $3 1 8 / 7 5 / 7 8$

LGE scar detection: binary scar classification on the EMIDEC dataset [17] (100 patients; 67% scar-positive), split 70/15/15.

## 4.2 Implementation

Pretraining setup. The encoder (ViT-S/16, 27.0M parameters) is trained for 300 epochs using AdamW [18] (Ir $5 \times 1 0 ^ { - 4 }$ , weight decay 0.05, 5,000-iteration warmup, batch size 64x4 GPUs, bfloatl6). Eight input frames are used. Frames are randomly sampled during training for augmentation and uniformly sampled across the full sequence during inference. Two global crops (224x224) and six local crops (96x96) are generated per sample $( V _ { g } { = } 2 , V { = } 8 )$ . The loss weighting is $\lambda { = } 0 . 0 5$ . Training proceeds in two stages: first at half spatial resolution, then at full 224x224, loading all weights except positional embeddings (re-initialized for the new grid).

Downstream training. For all methods, the encoder is frozen; only the downstream head is trained (Fig. 1). Each view’s [CLS] embedding (384-dim) is linearly projected to 512 dimensions. The gated attention mechanism uses two parallel pathways (Tanh and Sigmoid) whose element-wise product yields attention logits; softmax-normalized attention weights produce a single studylevel representation. For regression, a linear head maps to a scalar and is trained using Huber loss. For classification, a linear head maps to class logits and is trained using cross-entropy with class-balanced sampling. All tasks use AdamW (Ir 10“  ), 300 epochs, encoder frozen.

## 4.3 Experiments

Baselines. The baselines use the same frozen-encoder, gated-attention downstream architecture. All compared methods are finetuned on the downstream tasks (identical data, splits) for fair comparison.

a) Shad et al. [22]: Multiscale ViT (MViT, 36.3M params) pretrained with video-text contrastive learning on 293K OMR videos from 17K patients (three US centers). Pretrained projection head is frozen and uses 16 input frames. The model was pretrained on SAX, 2CH, 3CH and 4CH by the authors.

b)V-JEPA2 [4]: ViT-B/16 pretrained on natural video using the JEPA framework, which forms a strong natural-domain baseline without medical knowledge.

Metrics. Regression: MAE, Pearson correlation (r), Bland-Altman bias and limits of agreement. Classification: macro AUG, weighted AUG, per-class onevs-rest AUG.

Model Analyses We validate three design choices by comparing pretraining variants (all other settings fixed): (a) no encoder initialization from 2D CMR FM (random init); (b) no masking augmentation; (c) 16 input frames instead of 8. Each variant is evaluated via training loss and two linear probing tasks:

Table 1. Regression performance on the test set. r: Pearson correlation coefficient. Best results are shown in bold for MAE and r. Bias and limits of agreement (LoA) are reported jointly as Bland-Altman statistics. V-JEPA2 [4], Shad et al [22], MR-JEPA (ours).
<table><tr><td>Task</td><td>Method</td><td>MAE↓</td><td>r↑</td><td></td><td>Bias (LoA)</td></tr><tr><td rowspan="3">LV EF (n = 427)</td><td>V-JEPA2</td><td>6.70</td><td>0.189</td><td></td><td>0.92 [−17.6, 19.4]</td></tr><tr><td>Shad et al.</td><td>6.07</td><td>0.614</td><td></td><td>3.40 [-11.5, 18.3]</td></tr><tr><td>MR-JEPA</td><td>4.79</td><td>0.764</td><td>0.22</td><td>[−12.1, 12.5]</td></tr><tr><td rowspan="3">RV EF (n = 273)</td><td>V-JEPA2</td><td>9.65</td><td>0.323</td><td>-0.22</td><td>[−26.3, 25.8]</td></tr><tr><td>Shad et al.</td><td>9.35</td><td>0.410</td><td>0.83</td><td>[−23.5, 25.2]</td></tr><tr><td>MR-JEPA</td><td>8.67</td><td>0.506</td><td>1.62</td><td>[−21.5, 24.7]</td></tr><tr><td rowspan="3">GLS (n = 273)</td><td>V-JEPA2</td><td>2.38</td><td>0.648</td><td></td><td>-0.64 [−7.1, 5.9]</td></tr><tr><td>Shad et al.</td><td>2.90</td><td>0.713</td><td></td><td>-2.35 [−8.3, 3.6]</td></tr><tr><td>MR-JEPA</td><td>1.87</td><td>0.805</td><td>-0.49</td><td>[−5.5, 4.6]</td></tr><tr><td rowspan="3">GCS (n = 273)</td><td>V-JEPA2</td><td>3.03</td><td>0.564</td><td>-0.71</td><td>[−9.3,7.9]</td></tr><tr><td>Shad et al.</td><td>2.80</td><td>0.677</td><td>-1.80</td><td>[-9.5,5.9]</td></tr><tr><td>MR-JEPA</td><td>2.39</td><td>0.726</td><td></td><td>−0.46 [−7.6, 6.7]</td></tr><tr><td rowspan="3">GRS (n = 273)</td><td>V-JEPA2</td><td>7.45</td><td>0.598</td><td></td><td>-0.44 [−19.3, 18.4]</td></tr><tr><td>Shad et al.</td><td>7.26</td><td>0.717</td><td></td><td>4.16 [-12.2, 20.5]</td></tr><tr><td>MR-JEPA</td><td>5.33</td><td>0.802</td><td></td><td>-0.97 [-15.0, 13.1]</td></tr></table>

cine view classification representing temporal sequences, and LGE scar detection representing spatial sequences, (a) uses cine videos as inputs while (b) uses the stack of 2D LGE slices covering the spatial extent of the heart as a video input. Unlike the downstream evaluations, the model analysis uses a frozen encoder with a linear classifier to isolate representation quality rather than maximize downstream performance. Tasks from multiple CMR sequences are included to assess generalization beyond cine imaging.

## 5 Results

## 5.1 Cardiac Function Regression

Table 1 summarizes the regression results across all five cardiac function tasks.   
MR-JEPA achieves the best MAE and Pearson correlation on every task.

LV Ejection Fraction. On the Kaggle dataset (n=427), MR-JEPA achieves an MAE of 4.79% with a Pearson correlation of 0.764, substantially outperforming Shad et al. (MAE 6.07, r=0.614) and V-JEPA2 (MAE 6.70, r=0.189). Bland-Altman analysis shows MR-JEPA has minimal systematic bias (0.22%) and symmetric limits of agreement ([—12.1,12.5]), whereas Shad et al. exhibits a systematic overestimation of 3.40%.

RV Ejection Fraction. RV EF is inherently more challenging due to the complex right ventricular geometry. MR-JEPA achieves the best MAE (8.67%) and correlation (r=0.506), improving over Shad et al. by 0.68 percentage points. All methods show wider limits of agreement compared to LV EF, reflecting the greater measurement difficulty.

Table 2. Disease classification performance on the held-out test set $( n = 2 5 6 )$ , reported as macro AUC, weighted AUG, and one-vs-rest (OvR) AUC for each disease class. Best results are shown in bold. V-JEPA2 [4], Shad et al [22], MR-JEPA (ours).
<table><tr><td>Metric</td><td>V-JEPA2</td><td>Shad et al.</td><td>MR-JEPA</td></tr><tr><td>Macro AUC</td><td>0.741</td><td>0.882</td><td>0.868</td></tr><tr><td>Weighted AUC</td><td>0.743</td><td>0.864</td><td>0.854</td></tr><tr><td>OvR AUC – Normal</td><td>0.773</td><td>0.888</td><td>0.889</td></tr><tr><td>OvR AUC – DCM</td><td>0.777</td><td>0.905</td><td>0.890</td></tr><tr><td>OvR AUC – HCM</td><td>0.708</td><td>0.950</td><td>0.922</td></tr><tr><td>OvR AUC – IHD</td><td>0.705</td><td>0.785</td><td>0.771</td></tr></table>

Myocardial Strain. The largest improvements were observed for myocardial strain estimation. MR-JEPA achieved the best performance on GLS, GCS, and GRS, with MAEs of 1.87, 2.39, and 5.33, respectively. Relative to the strongest baseline, this corresponds to MAE reductions of 15-27% and higher correlations across all strain tasks $( r = 0 . 7 2 6 \mathrm { - } 0 . 8 0 5 )$ . MR-JEPA also maintained low systematic bias $( | \mathrm { b i a s } | \leq 0 . 9 7 )$

## 5.2 Cardiac Disease Detection

Table 2 presents the disease classification results. Unlike the regression tasks, Shad et al. achieves the highest macro AUC (0.882 vs. 0.868 for MR-JEPA), although MR-JEPA substantially outperforms V-JEPA2 (0.741). MR-JEPA achieves the highest one-vs-rest AUC for the normal class (0.889) and remains within 0.01-0.03 AUC of Shad et al. across all disease classes. These results suggest that MR-JEPA learns representations that transfer effectively to disease classification despite being optimized using a fully self-supervised objective. The stronger performance of Shad et al. on disease classification, despite not being pretrained on LGE, may be attributable to several differences in the pretraining setup, including a larger pretraining dataset (293K vs. 160K clips), longer input sequences (16 vs. 8 frames), and the use of video-text contrastive learning with clinical report supervision. Nevertheless, MR-JEPA remains competitive despite using fewer pretraining videos, a smaller model (27M vs. 36.3M parameters), and no text supervision.

## 5.3 Model Analysis

Table 3 evaluates three design choices of MR-JEPA. Although pretraining losses were similar across configurations (0.207-0.221), downstream performance differed markedly. Removing 2D-CMR foundation model initialization or spatiotemporal masking substantially reduced transferability, with scar detection dropping from 0.850 to chance level (0.500) in both cases. Increasing clip length from 8 to 16 frames reduced optimization efficiency (103 vs. 190 epochs) and lowered view classification accuracy (0.841 vs. 0.932). Overall, 2D-CMR initialization, spatiotemporal masking, and 8-frame clips yielded the strongest representations.

Table 3. Hyperparameter study. Linear probing accuracy for cine view classification and LGE scar detection under different pretraining configurations.
<table><tr><td>Configuration</td><td>Pretrain Loss</td><td>Epochs</td><td>View Cls. Acc.↑</td><td>Scar Det. Acc.↑</td></tr><tr><td>No DINOv3 init</td><td>0.208</td><td>207</td><td>0.890</td><td>0.500</td></tr><tr><td>No masking</td><td>0.207</td><td>223</td><td>0.908</td><td>0.500</td></tr><tr><td>16 frames</td><td>0.221</td><td>103</td><td>0.841</td><td>0.750</td></tr><tr><td>Ours (full)</td><td>0.215</td><td>190</td><td>0.932</td><td>0.850</td></tr></table>

## 6 Conclusions

We present MR-JEPA, a video foundation model for cardiac MRI that extends LeJEPA to 3D spatiotemporal inputs. Using a simple joint-embedding objective with spatiotemporal masking and initialization from a 2D CMR foundation model, MR-JEPA learns representations from 160K multi-sequence CMR clips across 10,505 patients without annotations. With a frozen encoder and gated attention aggregation, it achieves state-of-the-art performance on five cardiac function regression tasks (LV EF, RV EF, GLS, GCS, GRS), outperforming domain-specific and natural video baselines. Compared with prior CMR video foundation models trained exclusively on cine imaging, pretraining on multiple CMR sequences may improve representation learning across heterogeneous image contrasts. On disease detection, it remains competitive despite using less data and a simpler, unsupervised training objective. Ablations confirm the importance of encoder initialization, spatiotemporal masking, and 8-frame inputs, while embedding analysis shows that the model naturally separates temporal and spatial sequences and generalizes to unseen spatial cine volumes.

Limitations: Pretraining was performed using single-vendor CMR data. Although the LV EF task was evaluated on an independent multi-center dataset, the remaining downstream tasks were evaluated on data from one pretraining center; however, no labels were used during pretraining. Ground truth for RV EF and strain estimation was obtained from a previously validated deep learning model rather than manual expert annotations. Finally, the pretraining dataset was heavily dominated by cine acquisitions, with only a small proportion of mapping data, that too from a single center. Future work should collect more mapping data and evaluate on mapping-specific downstream tasks (e.g., fibrosis/edema quantification). Finally, we compared MR-JEPA primarily against existing foundation models under a unified downstream evaluation protocol. Comparison with task-specific state-of-the-art models would provide additional clinical context and is left for future work.

Disclaimer: The concepts and information presented in this paper are based on research results that are not commercially available. Future commercial availability cannot be guaranteed. Disclosure of Interests: AJ, PS and DC are employees of Siemens Healthineers. The other authors declare that they have no competing interests.

## References

1. Afana, A.S., Garot, J., Duhamel, S., Hovasse, T., Champagne, S., Unterseeh, T., Garot, P., Akodad, M., Chitiboi, T., Sharma, P., et al.: Prediction of cardiovascular events using fully automated global longitudinal and circumferential strain in patients undergoing stress cmr. Circulation: Cardiovascular Imaging 18(10), e018350 (2025)

2. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lucic, M., Schmid, C.: Vivit: A video vision transformer. In: 2021 IEEE/CVF international conference on computer vision (ICCV). pp. 6816-6826. IEEE (2021)

3. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N.: Self-supervised learning from images with a joint-embedding predictive architecture. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15619-15629 (2023)

4. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al.: V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025)

5. Balestriero, R., LeCun, Y.: Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544 (2025)

6. Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., Ballas, N.: Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471 (2024)

7. Canuti, E., Toupin, S., Sanguineti, F., Hovasse, T., Garot, P., Champagne, S., Chitiboi, T., Jacob, A., Borgohain, L, Sharma, P., et al.: Artificial intelligence fully automated assessment of global longitudinal strain predicts death in patients undergoing stress cmr. Archives of Cardiovascular Diseases 117(1), S57-S58 (2024)

8. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: 2017 IEEE conference on computer vision and pattern recognition (CVPR). pp. 4724-4733. IEEE (2017)

9. Chen, C., Qin, C., Qiu, H., Tarroni, G., Duan, J., Bai, W., Rueckert, D.: Deep learning for cardiac image segmentation: a review. Frontiers in cardiovascular medicine 7, 508599 (2020)

10. Chitiboi, T., Georgescu, B., Wetzl, J., Borgohain, L, Geppert, C., Piechnik, S.K., Neubauer, S., Petersen, S., Sharma, P.: Deep learning-based strain quantification from cine cardiac mri. In: ISMRM Annual Meeting (2020)

11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Geliy, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

12. Fu, Y., Bai, W., Yi, W., Manisty, C., Bhuva, A.N., Treibel, T.A., Moon, J.C., Clarkson, M.J., Davies, R.H., Hu, Y.: A versatile foundation model for cine cardiac magnetic resonance image analysis tasks. arXiv preprint arXiv:2506.00679 (2025)

13. Use, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: International conference on machine learning, pp. 2127—2136. PMLR (2018)

14. Jacob, A.J., Borgohain, L, Chitiboi, T., Sharma, P., Comaniciu, D., Rueckert, D.: Towards a cardiovascular magnetic resonance foundation model for multi-task cardiac image analysis. Journal of Cardiovascular Magnetic Resonance 27(2), 101967 (2025)

15. Jacob, A.J., Chitiboi, T., Schoepf, U.J., Sharma, P., Aldinger, J., Baker, C., Lautenschlager, C., Emrich, T., Varga-Szemes, A.: Deep-learning-based disease classification in patients undergoing cine cardiac mri. Journal of Magnetic Resonance Imaging 61(4), 1635-1647 (2025)

16. Kramer, C.M., Barkhausen, J., Bucciarelli-Ducci, C., Flamm, S.D., Kim, R.J., Nagel, E.: Standardized cardiovascular magnetic resonance imaging (cmr) protocols: 2020 update. Journal of Cardiovascular Magnetic Resonance 22(1), 17 (2020)

17. Lalande, A., Chen, Z., Decourselle, T., Qayyum, A., Pommier, T., Lorgis, L., de La Rosa, E., Cochet, A., Cottin, Y., Ginhac, D., et al.: Emidec: a database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac mri. Data 5(4), 89 (2020)

18. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

19. Newton, A., Buckeye, A., Arai, A., Mulholland, J., BoozAllen, J., kungvu, O’Connell, M., Kim, M., Salasznyk, R., Shannon, Steve Mills, W.C.: Second annual data science bowl (2015), https://kaggle.com/competitions/second-annualdata-science-bowl

20. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

21. Pezel, T., Garot, P., Toupin, S., Sanguineti, F., Hovasse, T., Unterseeh, T., Chitiboi, T., Jacob, A., Borgohain, L, Sharma, P., et al.: Incremental prognostic value of fully-automatic machine-learning based global circumferential strain during a stress cmr exam. European Heart Journal 44(Supplement\_2), ehad655-214 (2023)

22. Shad, R., Zakka, C., Kaur, D., Mathur, M., Fong, R., Cho, J., Filice, R.W., Mongan, J., Kallianos, K., Khandwala, N., et al.: A generalizable deep learning system for cardiac mri. Nature Biomedical Engineering pp. 1—16 (2026)

23. Simeoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

24. Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are dataeflicient learners for self-supervised video pre-training. Advances in neural information processing systems 35, 10078-10093 (2022)

25. Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are dataefficient learners for self-supervised video pre-training. Advances in neural information processing systems 35, 10078-10093 (2022)

26. Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., Qiao, Y.: Videomae v2: Scaling video masked autoencoders with dual masking. In: Proceed ings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14549-14560 (2023)