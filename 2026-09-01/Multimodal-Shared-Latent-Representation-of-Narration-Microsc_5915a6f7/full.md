# Multimodal Shared Latent Representation of Narration, Microscope and iOCT Images for Phase Recognition in Vitreoretinal Surgery

Onur Izmitlioglu<sup>1</sup>, Shervin Dehghani<sup>1,2</sup>, Tarek Ghannoum<sup>3</sup>, Benedikt Schworm<sup>3</sup>, and Nassir Navab<sup>1</sup>

<sup>1</sup> Computer Aided Medical Procedures, Technical University of Munich, Germany {onur.izmitlioglu, shervin.dehghani, nassir.navab}@tum.de <sup>2</sup> SynthesEyes GmbH, Munich, Germany

3 Department of Ophthalmology, LMU University Hospital, Munich, Germany t.ghannoum@med.uni-muenchen.de, benedikt.schworm@med.uni-muenchen.de

Abstract. Surgical phase recognition is key to context-aware computerassisted feedback in vitreoretinal procedures, yet the scarcity of synchronized multimodal intraoperative data, particularly microscope views and intraoperative OCT, limits approaches that aim to replicate the multimodal integration surgeons perform naturally. Surgical narration, by contrast, is abundantly available online and ofers rich semantic supervision. Prior work has mainly explored pairwise contrastive learning (e.g., intraoperative OCT–microscope or microscope–narration), leaving the joint modeling of all three modalities largely unexplored. We introduce a framework that uses microscope views as a shared anchor to bridge surgical narrations and intraoperative OCT (iOCT) without requiring a fully synchronized tri-modal dataset, leveraging real microscope–narration videos and a synthetic dataset of synchronized microscope video and tool-aligned iOCT pairs. Contrastive alignment transfers structural priors from the synthetic domain to real videos lacking iOCT, and a dual-head MS-TCN++ integrates the resulting embeddings for joint macro- and micro-phase prediction. Evaluated on real vitreoretinal surgeries, our framework improves macro-phase recognition over a zero-shot baseline (mean F1 0.38 → 0.53) and provides an exploratory route to estimating fine-grained instrument– tissue measurements that are not directly observable in real microscope video alone; these micro-phase estimates are validated quantitatively on synthetic data and shown only qualitatively on real surgery. To our knowledge, this is the first work to unify microscope view, iOCT Bscans, and surgical narrations in a shared latent space for surgical phase recognition.

Keywords: Vitreoretinal Surgery · Surgical Phase Recognition · Video-Language Modeling · Multimodal Alignment

## 1 Introduction

Vitreoretinal surgery involves high-precision microsurgical procedures that demand exceptional skill and real-time decision-making. Enhancing a surgeon’s contextual awareness through automated systems is critical for improving clinical outcomes, reducing cognitive load, and standardizing surgical training [15, 18]. Surgical phase recognition serves as a foundational capability for such contextaware systems; however, it remains challenging in vitreoretinal procedures due to subtle tool-tissue interactions and the scarcity of annotated multimodal datasets. We distinguish between two levels of surgical phases: macro-phases, corresponding to the conventional high-level stages of the procedure, and micro-phases, capturing fine-grained instrument states such as forceps jaw closure and tool-toretina distance. Current AI-driven vitreoretinal surgery analysis often relies on single-modality video data, neglecting the depth-resolved structural insights provided by iOCT, which ofers critical subsurface tissue information and tool-tissue proximity that video alone cannot capture.

![](images/5c9263e26a71ac4946221175da781c32ef0ab1128be60478c10ae616d485afe7.jpg)  
Fig. 1. Overview of the proposed framework. We use three modality-specific encoders: Encoder 1 (surgical narration), Encoder 2 (microscope view), and Encoder 3 (iOCT B-scans). Stage A: Encoder 2 is aligned to the frozen Encoder 1 via constraint C1, bringing microscope frames and narrations into a shared domain. Stage B: with Encoder 2 now frozen, Encoder 3 is trained via constraint C2 to project iOCT Bscans into the same domain. Stage C: the resulting embeddings train the dual-head MS-TCN++ for macro- and micro-phase prediction. At inference, because semantically similar modalities occupy neighboring regions of the latent space, the MS-TCN++ predicts both macro- and micro-phases even when only the microscope view is available.

Surgical narrations ofer rich semantic supervision [27], yet datasets jointly containing narrations, microscope views, and iOCT B-scans are largely absent from the literature. Curating such a tri-modal dataset is itself a significant challenge, requiring synchronized acquisition across heterogeneous imaging systems in a demanding intraoperative setting. Moreover, clinical iOCT B-scans are not constrained to anatomically informative regions, and current technology does not support tool-following acquisition, which leaves much of the captured crosssectional information contextually decoupled from the surgical procedure. To address this, we leverage a synthetic dataset of tool-aligned iOCT B-scans, ensuring that the acquired cross-sections are anchored to the surgical site and thus carry procedure-relevant context. Utilizing microscope views as a common anchor, we then bridge two complementary datasets, one of microscope–narration pairs and one of microscope–iOCT pairs, aligning all three modalities in a shared latent space without requiring a fully annotated tri-modal dataset.

![](images/fdb42ddbf1d102be373fcec5fb9c28c989aac46065c06120e8f1fde3ed3d6e7f.jpg)  
Fig. 2. Overview of the dataset. (a) A sample frame from a youtube video taken from Dr. Simon Chen channel, containing the narration with the microscope view. (b) A sample frame from the synthetic dataset, containing the microscope view, the toolaligned iOCT cross-sections, and forceps properties (tool-to-retina distance and forceps jaw closure).

Once the shared latent space is learned, supervised by surgical narrations and fine-grained annotations of the synthetic dataset, any single modality carries enough semantic structure to support inference about the others at inference time, as the joint supervision brings their representations into close proximity. For instance, a microscope view alone could in principle support an approximation of tool-to-retina distance without iOCT, to the extent that the two representations come to lie near each other in the latent space; we regard such cross-modal inference as an exploratory direction and evaluate it directly only where ground truth is available (Sec. 4).

## 2 Related Works

Deep Learning in Surgical Contexts Deep learning, particularly CNNs and RNNs, shifted surgical workflow analysis toward robust, data-driven methods. [24] introduced EndoNet for laparoscopic videos, proving that deep models could outperform handcrafted features by learning directly from raw data. To better capture temporal dependencies, [4] developed TECNO, utilizing multi-stage convolutional networks, while [23] advanced clinical integration with real-time phase recognition for cholecystectomies. [5] synthesized these trends, highlighting the shift toward multimodal, context-aware systems while noting that a lack of labeled data remains a significant bottleneck.

Ophthalmic Surgery and Domain-specific Challenges While progress has been made in general surgery, ophthalmic procedures—particularly vitreoretinal—remain underrepresented. [1] addressed this through the CATARACTS challenge for tool annotation, yet most research still prioritizes laparoscopic or robotic surgery, overlooking the unique visual traits of eye procedures. Recently, [13] introduced OphNet, a large-scale benchmark for ophthalmic workflows, enabling more targeted, fine-grained temporal analysis. Despite this, the field still lacks narrated or context-rich data, a gap the current study aims to bridge. More recently, synthetic datasets have opened new avenues for multimodal learning in surgical settings. [21] propose training on tool-aligned iOCT cross-sections paired with fundus images to estimate tool-to-retina distance.

Language as Context in Surgical Modeling The integration of language into surgical video analysis has introduced vital semantic context. [11] showed that speech-based phase recognition provides non-intrusive insights for skill assessment. Furthering the language as context approach, [28] leveraged multimodal learning from video lectures, using narration as a rich supervisory signal for temporal alignment. Similarly, [9] reinforced that expert language is a crucial supervisory signal for enhancing video-based surgical understanding, training a video-language model with contrastive alignment to learn short-term spatiotemporal and multimodal representations for surgical phase detection, which are then passed to an MS-TCN++ to capture long-range temporal relationships across the full video.

Vision-Language Pretraining and Contrastive Learning Vision-language pretraining and contrastive learning have shifted surgical modeling toward zeroshot and few-shot capabilities. [27] and [26] introduced hierarchical models that align video frames with text to improve contextual transfer. This study specifically builds on PeskaVLP, which uses narrated content and contrastive alignment to bridge the visual-semantic gap and overcome annotation scarcity. Furthermore, frameworks like IMAGEBIND [10] demonstrate that contrastive objectives can align diverse modalities beyond text, such as audio and images, in a shared latent space, which underscores the versatility of modality-agnostic pretraining, supporting our objective of aligning OCT and microscope view representations for vitreoretinal surgery.

## 3 Methodology

## 3.1 Macro- and Micro-Phase Definitions

In this work, we focus on epiretinal membrane (ERM) peeling as a representative vitreoretinal procedure, chosen for its well-defined sequential structure and the availability of both real and synthetic data in our dataset. Our schema follows a two-tier hierarchy: macro-phases for procedural structure and micro-phases for instrument-tissue interaction.

Macro-phase We decompose ERM peeling into five macro-phases, defined by keystep-level timestamps and established in consultation with vitreoretinal surgical experts to ensure clinical validity:

1. Anterior Preparation: Initial surgical setup, including trocar insertion and port placement.

2. Posterior Preparation: Establishment of intraocular visualization and core vitrectomy.

3. Membrane Identification: Localization of the target membrane, typically aided by chromodye staining.

4. Membrane Peeling: Active removal of the epiretinal membrane through forceps manipulation.

5. Retina Stabilization: Post-peeling assessment and restoration, including procedures such as air-fluid exchange.

Micro-phase Micro-phases characterize instrument-tissue interaction through two continuous attributes available in the synthetic dataset: the tool-to-retina distance and forceps jaw closure.

## 3.2 Dataset Description

The real dataset comprises 33 ERM peeling procedures sourced from online platforms, featuring full surgical microscope views (avg. 3 min) with varying resolutions. Whisper [19] generated three annotation granularities: Abstractlevel: video-level semantic summaries, Keystep-level: timestamped milestones for macro-phase segmentation and Narration-level: synchronous transcriptions of real-time surgeon commentary. Keystep timestamps used for macro-phase segmentation were reviewed and, where needed, corrected by a vitreoretinal surgeon, while narration-level transcripts were kept as produced by Whisper. Videos are split at the procedure level into 28 training procedures and 5 held-out procedures, with all frames of a given procedure kept within a single split. The 5 held-out procedures are used both to evaluate the Stage A zero-shot baseline (Table 1) and to report final macro-phase performance (Table 2); no hyperparameter tuning is performed on them.

For the synthetic data, five ERM peeling procedures were provided by SynthesEyes GmbH <sup>4</sup>, covering membrane interaction and instrument manipulation. We additionally use two synthetic subretinal-injection sequences performed with a straight cannula. These sequences share the retinal setting but difer in instrument and task; they provide a contrasting procedural context that sharpens the phase boundaries of ERM peeling and act as hard negatives during Stage B alignment (Sec. 3.4). In all synthetic sequences the iOCT cross-sections are continuously aligned to the primary instrument, tracking its movement throughout the procedure. For all synthetic sequences, accompanying metadata provides per-frame tool-to-retina distance, a normalized value in the range [0, 1] representing the vertical proximity of the instrument tip to the retinal surface. The ERM peeling sequences additionally provide forceps jaw closure, a continuous value in the range [0, 1] where 0 corresponds to fully open and 1 to fully closed; the subretinal-injection sequences use a straight cannula and therefore carry no jaw-closure attribute. A sample from the dataset is shown in Fig. 2.

![](images/0f1192d6a7dfc7b505ccd9e46b5b4a9aa7d8b6c23ef860208bc50dadf17f5e97.jpg)  
Fig. 3. Overview of the models. (a) Stage A: vision-language pretraining with hierarchical (narration-, keystep-, and abstract-level) text-image pairs. f = Encoder 2, a trainable ResNet-50 microscope-image encoder; g = Encoder 1, a frozen Bio\_ClinicalBERT text encoder. (b) Stage B: microscope-iOCT alignment. f = Encoder 2, frozen from Stage A; g = Encoder 3, a trainable iOCT encoder. Both branches project to a shared 512-dimensional space and are aligned with a hard triplet objective (Sec. 3.4).

## 3.3 Stage A: Vision-Language Model

The baseline model for surgical phase recognition utilizes Vision-Language Pretraining (VLP) to derive semantic representations from microscope frames. This approach adapts PeskaVLP [27] to the vitreoretinal domain. Dual-Encoder Architecture projects visual and textual inputs into a shared embedding space, enabling the alignment of surgical frames with descriptions through contrastive learning, as shown in Fig. 3.

The text encoder (Encoder 1 in Fig. 1) utilizes Bio\_ClinicalBERT [2], pretrained on the MIMIC-III dataset to provide deep understanding of surgical terminology. It processes tokens through 12 transformer layers, summing the final four to capture both low-level and high-level semantics. During fine-tuning, the BERT parameters are frozen to maintain clinical knowledge, while the image encoder is trained to adapt to the vitreoretinal domain. The image encoder (Encoder 2 in Fig. 1) uses a ResNet-50 backbone [12], initialized with ImageNet weights. Microscope frames are resized to 336 × 336 pixels and normalized. The architecture extracts spatial features through residual blocks, followed by global average pooling and a linear projection layer that maps the output to the 512- dimensional latent space.

## 3.4 Stage B: Multimodal Alignment with iOCT Data

To close the gap between synthetic iOCT data and real-world microscope videos, we adopt a dual-encoder architecture grounded in metric learning and crossmodal alignment principles [22, 14], projecting both modalities into a unified 512-dimensional embedding space. Encoder 2 remains frozen from Stage A to preserve its pretrained semantic procedural knowledge, while a trainable iOCT encoder (Encoder 3 in Fig. 1) is jointly optimized on synthetic scans, enriching the microscope representations with depth-resolved structural information and bridging the two modalities without requiring real iOCT data. The iOCT encoder is a CNN that maps each pair of tool-aligned cross-sections to this shared embedding space, with alignment learned via a hard triplet loss that pulls distances toward 0 for temporally matched microscope-iOCT pairs and toward 1 for unmatched pairs. Triplets are sampled by phase: given an anchor microscope frame and its temporally matched iOCT pair drawn from a synthetic sequence performed with one instrument (tool A), the negative is the iOCT pair of a randomly chosen frame from a sequence performed with the other instrument (tool B). Concretely, microscope frames showing forceps (ERM peeling) are contrasted against iOCT from cannula-based subretinal injection and vice versa, encouraging the encoder to represent instrument- and task-specific structure rather than generic retinal appearance.

## 3.5 Stage C: Multi-Task Temporal Modeling

Frame-level representations from Stage A and Stage B ignore temporal logic, leading to workflow inconsistencies and rapid phase oscillations [6, 7] that median filtering cannot adequately resolve [5]. While RNNs and LSTMs [16] have traditionally modeled temporal sequences, they sufer from parallelization bottlenecks [16, 20], vanishing gradients [3], and fixed-size memory constraints [20]. TCNs [16, 3] overcome these by using dilated convolutions [25] to exponentially expand the receptive field, capturing multi-scale procedural structures [25, 4].

The MS-TCN++ architecture [17] improves upon initial multi-stage refinement [8] by employing dual dilated layers to capture both short-term landmarks and long-term constraints [4, 8, 17]. We implement a dual-head multi-task architecture to simultaneously address macro-phase classification and micro-phase regression.

Head-1 regresses to five high-level phases using manual integer annotations from real surgical videos. While it successfully captures the overall procedural structure of ERM peeling, it lacks fine-grained details regarding instrument states or anatomical measurements.

Head-2 utilizes synthetic videos and the accompanying metadata to regress to two continuous values, forceps jaw closure and tool-to-retina distance. Unlike the discrete classification of Head-1, this head characterizes subtle instrumenttissue interactions that are inherently unobservable in microscope footage but are observable in iOCT data.

## 4 Experimental Results

The implementation is done using PyTorch 2.7.0 and Python 3.11. The codebase leverages GPUs with CUDA capabilities and standard deep learning libraries, including NumPy 2.2.7 for numerical computations, OpenCV 4.11.0 for video processing, and Pillow 11.2.1 for image handling.

The held-out set of five surgical videos is used to assess the trained PeskaVLP model in order to create a baseline for surgical phase recognition. Phase predictions were made without any temporal modeling, following a zero-shot procedure: for each of the five macro-phases, a descriptive text prompt is designed and projected into the latent space via Encoder 1, while each microscope frame is independently projected into the same latent space via Encoder 2. The predicted phase for a given frame is then assigned as the class whose text embedding is closest to the frame embedding, with closeness measured via cosine similarity. The baseline model’s performance is displayed in Table 1.

<table><tr><td>Phase Prec. Rec. F1</td></tr><tr><td>Anterior Prep. 0.59 0.38 0.46 Posterior Prep.</td></tr><tr><td>0.20 0.03 0.06 Membrane Id. 0.28 0.39 0.33</td></tr><tr><td>Membrane Peeling 0.77 0.64 0.70</td></tr><tr><td>Retina Stab. 0.24 0.55 0.33</td></tr><tr><td>Mean 0.42 0.40 0.38</td></tr></table>

Table 1. Baseline zero-shot macro-phase recognition performance after Stage A

<table><tr><td>Phase Prec. Rec. F1</td></tr><tr><td>Anterior Prep. 0.53 0.70 0.61</td></tr><tr><td>Posterior Prep. 0.50 0.84 0.63</td></tr><tr><td>Membrane Id. 0.31 0.32 0.31 Membrane Peeling 0.76 0.68 0.72</td></tr><tr><td>Retina Stab. 0.42 0.36 0.38 Mean 0.50 0.58 0.53</td></tr></table>

Table 2. Macro-phase recognition performance on real surgical videos with refined labels followed by Stage C

For macro-phase recognition, the dual-head MS-TCN++ model with label refinement outperformed the zero-shot baseline, achieving a mean F1 score of 53% with mean precision of 50.3% and mean recall of 57.9%. The performance breakdown by surgical phases is shown in Table 2. The model shows particularly strong gains for Anterior and Posterior Preparation, while Membrane Identification and Retina Stabilization remain comparatively harder to recognize. This configuration jointly introduces temporal modeling, label refinement, and multitask learning relative to the Stage A baseline; disentangling their individual contributions requires dedicated ablations, which we leave to future work.

Micro-phase prediction was assessed quantitatively on the held-out synthetic ERM peeling sequences, where per-frame ground truth is available, and only qualitatively on real surgical videos. On synthetic data, the model achieved an MAE of $\mathbf { 0 . 0 5 0 \ : \pm { \ : 0 . 0 0 1 } }$ for tool-to-retina distance and ${ \bf 0 . 1 6 7 \pm 0 . 1 3 5 }$ for forceps jaw closure. This ordering is counterintuitive, since the lower MAE for distance would ordinarily indicate the easier target. The two signals, however, have very diferent statistics. Tool-to-retina distance changes slowly and stays within a narrow band for most of the procedure, so a near-constant, over-smoothed prediction already yields a small absolute error while failing to follow the instrument’s true vertical motion, an expected consequence of the missing depth cues in 2D imagery. Forceps jaw closure instead spans the full [0, 1] range with frequent open/close transitions; the model recovers the timing and direction of these transitions but incurs larger errors and high per-sequence variance (±0.135) around them. In this sense forceps closure is predicted more informatively, since the clinically meaningful state changes are captured, even though its MAE is higher. On real videos the same qualitative pattern holds: forceps-state estimates vary meaningfully over time, whereas distance estimates stay stable but conservative. As no real-domain ground truth is available, these real-video micro-phase results should be read as exploratory rather than as validated measurements. Overall, discrete state changes appear learnable under weak cross-modal supervision, while continuous depth estimation remains challenging without explicit 3D priors.

Limitations. Several limitations should be noted. The study uses 33 real videos and seven synthetic procedures (five ERM peeling, two subretinal injection), with results reported on a single procedure-level split (28 training, 5 held-out real videos). The synthetic data is also limited in coverage: it does not naturally span all macro-phases, so only a few sequences are available per phase and some phases are not represented at all, which constrains the micro-phase supervision. The macro-phase improvement in Table 2 reflects the combined efect of temporal modeling, label refinement, and multi-task learning; our current experiments do not isolate the individual contribution of the narration/microscope/iOCT alignment through ablations. Micro-phase targets are validated only on synthetic data, and the structure of the learned shared latent space is not analyzed directly, either geometrically (e.g., via cross-modal retrieval or matched-versusunmatched embedding distances) or topologically (e.g., via persistent homology of how the modalities and phases are connected in the embedding). Our comparison is against a zero-shot vision-language baseline; supervised video-only temporal models and standard domain-adaptation methods are not included. Ablations per modality and training stage, stronger supervised baselines, realdomain micro-phase annotation, multi-split/multi-seed reporting, and quantitative latent-space analysis are the focus of ongoing work.

## 5 Conclusion

Our results suggest that the surgical microscope image can serve as a common anchor to bridge otherwise disjoint datasets, allowing structural priors learned from synthetic iOCT to propagate into real surgical video without ever requiring synchronized tri-modal recordings. As synthetic data generation matures and the curation of real narrated surgical videos expands, both the fidelity of learned representations and the overall performance are expected to improve substantially. Rigorous evaluation on larger and more diverse datasets, including real surgical videos annotated with forceps properties to directly assess the method’s adaptability, will be essential to disentangle the contribution of each modality, inform more principled training strategies, and determine under which clinical conditions each modality provides the greatest benefit. Loss functions that more explicitly encourage cross-modal attention between the modalities, rather than only pulling matched pairs together, would also be a promising direction. This is a natural next step toward building richer, more robust multimodal representations for surgical AI, a form of cross-modal reasoning that surgeons perform implicitly throughout a procedure.

## 6 Acknowledgments

This work was supported by the Bavarian Research Foundation (BFS) under Grant AZ-1569-22. We gratefully acknowledge Hessam Roodaki and Ghazal Ghazaei for their support and helpful discussions.

Competing interests S.D. and N.N. are shareholders of SynthesEyes GmbH, the company that provided the synthetic dataset used in this work. All other authors declare that they have no competing interests.

## References

1. Al Hajj, H., Lamard, M., Conze, P.H., Roychowdhury, S., Hu, X., Maršalkait˙e, G., Zisimopoulos, O., Dedmari, M.A., Zhao, F., Prellberg, J.: Cataracts: Challenge on automatic tool annotation for cataract surgery. Medical image analysis 52, 24–41 (2019)

2. Alsentzer, E., Murphy, J., Boag, W., Weng, W.H., Jindi, D., Naumann, T., Mc-Dermott, M.: Publicly available clinical bert embeddings. In: Proceedings of the 2nd clinical natural language processing workshop. pp. 72–78 (2019)

3. Bai, S., Kolter, J.Z., Koltun, V.: An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271 (2018)

4. Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S.T., Navab, N.: Tecno: Surgical phase recognition with multi-stage temporal convolutional networks. In: International conference on medical image computing and computer-assisted intervention. pp. 343–352. Springer (2020)

5. Demir, K.C., Schieber, H., Weise, T., Roth, D., May, M., Maier, A., Yang, S.H.: Deep learning in surgical workflow analysis: a review of phase and step recognition. IEEE Journal of Biomedical and Health Informatics 27(11), 5405–5417 (2023)

6. Ding, H., Gao, Z., Planche, B., Luan, T., Sharma, A., Zheng, M., Lou, A., Chen, T., Unberath, M., Wu, Z.: Neural finite-state machines for surgical phase recognition (2024)

7. Ding, H., Lian, X., Unberath, M.: Mosformer: Augmenting temporal context with memory of surgery for surgical phase recognition (2025), https://arxiv.org/abs/2503.00695

8. Farha, Y.A., Gall, J.: Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 3575–3584 (2019)

9. Gastager, D., Ghazaei, G., Patsch, C.: Watch and learn: leveraging expert knowledge and language for surgical video understanding. International Journal of Computer Assisted Radiology and Surgery 21(1), 185–194 (2026)

10. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15180– 15190 (2023)

11. Guzmán-García, C., Gómez-Tome, M., Sanchez-Gonzalez, P., Oropesa, I., Gómez, E.J.: Speech-based surgical phase recognition for non-intrusive surgical skills’ assessment in educational contexts. Sensors 21(4), 1330 (2021)

12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

13. Hu, M., Xia, P., Wang, L., Yan, S., Tang, F., Xu, Z., Luo, Y., Song, K., Leitner, J., Cheng, X.: Ophnet: A large-scale video benchmark for ophthalmic surgical workflow understanding. In: European Conference on Computer Vision. pp. 481– 500. Springer (2024)

14. Ketabi, S., Wagner, M.W., Hawkins, C., Tabori, U., Ertl-Wagner, B.B., Khalvati, F.: Multimodal contrastive learning for enhanced explainability in pediatric brain tumor molecular diagnosis. Scientific Reports 15(1), 10943 (2025)

15. Lalys, F., Bouget, D., Rifaud, L., Jannin, P.: Automatic knowledge-based recognition of low-level tasks in ophthalmological procedures. International journal of computer assisted radiology and surgery 8(1), 39–49 (2013)

16. Lea, C., Flynn, M.D., Vidal, R., Reiter, A., Hager, G.D.: Temporal convolutional networks for action segmentation and detection. In: proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 156–165 (2017)

17. Li, S., Farha, Y.A., Liu, Y., Cheng, M.M., Gall, J.: Ms-tcn++: Multistage temporal convolutional network for action segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(6), 6647–6658 (2023). https://doi.org/10.1109/TPAMI.2020.3021756

18. Quellec, G., Charrière, K., Lamard, M., Droueche, Z., Roux, C., Cochener, B., Cazuguel, G.: Real-time recognition of surgical tasks in eye surgery videos. Medical image analysis 18(3), 579–590 (2014)

19. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision. In: Proceedings of the 40th International Conference on Machine Learning. PMLR (2023)

20. Ramesh, S., Dall’Alba, D., Gonzalez, C., Yu, T., Mascagni, P., Mutter, D., Marescaux, J., Fiorini, P., Padoy, N.: Multi-task temporal convolutional networks for joint recognition of surgical phases and steps in gastric bypass procedures. International Journal of Computer Assisted Radiology and Surgery 16, 1111–1119 (2021)

21. Rohrmoser, N., Ghazaei, G., Sommersperger, M., Navab, N.: Toward comprehensive real-time scene understanding in ophthalmic surgery through multimodal image fusion. International Journal of Computer Assisted Radiology and Surgery pp. 1–9 (2026)

22. Schrof, F., Kalenichenko, D., Philbin, J.: Facenet: A unified embedding for face recognition and clustering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 815–823 (2015)

23. Shinozuka, K., Turuda, S., Fujinaga, A., Nakanuma, H., Kawamura, M., Matsunobu, Y., Tanaka, Y., Kamiyama, T., Ebe, K., Endo, Y.: Artificial intelligence software available for medical devices: surgical phase recognition in laparoscopic cholecystectomy. Surgical Endoscopy 36(10), 7444–7452 (2022)

24. Twinanda, A.P., Shehata, S., Mutter, D., Marescaux, J., De Mathelin, M., Padoy, N.: Endonet: a deep architecture for recognition tasks on laparoscopic videos. IEEE transactions on medical imaging 36(1), 86–97 (2016)

25. Yu, F., Koltun, V.: Multi-Scale Context Aggregation by Dilated Convolutions. arXiv e-prints p. arXiv:1511.07122 (2015). https://doi.org/10.48550/arXiv.1511.07122

26. Yuan, K., Srivastav, V., Navab, N., Padoy, N.: Hecvl: Hierarchical video-language pretraining for zero-shot surgical phase recognition. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 306–316. Springer (2024)

27. Yuan, K., Srivastav, V., Navab, N., Padoy, N.: Procedure-aware surgical videolanguage pretraining with hierarchical knowledge augmentation. In: Advances in Neural Information Processing Systems. vol. 37, pp. 122952–122983 (2024)

28. Yuan, K., Srivastav, V., Yu, T., Lavanchy, J.L., Marescaux, J., Mascagni, P., Navab, N., Padoy, N.: Learning multi-modal representations by watching hundreds of surgical video lectures. Medical Image Analysis p. 103644 (2025)