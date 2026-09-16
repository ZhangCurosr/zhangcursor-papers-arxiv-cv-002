# PSMP-CLIP: PATCH-PROMPT SAM AND MULTI-SEMANTIC PROMPTING FOR CLIP-BASED ZERO-SHOT ANOMALY DETECTION

Xuezhi Xiang<sup>1,2,\*</sup>, Guanghao Wu<sup>1</sup>, Heqi Xiang<sup>3</sup>, Jiayao Liu<sup>1</sup>, Xiaoheng Li<sup>1</sup>, Yiming Chen<sup>1</sup>, Shanjun Zhang<sup>4</sup>

<sup>1</sup>Information and Communication Engineering, Harbin Engineering University, Harbin, China <sup>2</sup> Key Laboratory of Advanced Marine Communication and Information Technology, Harbin, China <sup>3</sup>Department of Computer Science, University of Toronto, Toronto, ON M5S 2E4, Canada <sup>4</sup>The Department of Computer Science, Kanagawa University, Kanagawa, 221-8686, Japan E-mails: xiangxuezhi@hrbeu.edu.cn, wuguanghao@hrbeu.edu.cn, claire.xiang@mail.utoronto.ca

## ABSTRACT

Zero-shot anomaly detection aims to localize anomalies without target-domain samples. Existing CLIP-based methods suffer from coarse anomaly maps and limited semantic prompts. We propose PSMP-CLIP, integrating patch-prompt SAM2 segmentation (PPSS) and multi-semantic guided prompt regularization (MSGPR). PPSS samples prompts directly from intermediate patch features, avoiding threshold drift and guiding SAM2 to produce precise masks. MSGPR uses multiple learnable prompts constrained by semantic anchors to preserve generalization. Experiments on 14 datasets show highly competitive performance, achieving the best pixel-level AUROC on MVTec AD, BTAD, DTD-Synthetic, CVC-ClinicDB, TN3K, Endo, and Kvasir.

Index Terms— Zero-shot anomaly detection, CLIP, SAM, prompt learning, multi-semantic guided prompt regularization

## 1. INTRODUCTION

Anomaly detection is critical in industrial inspection and medical imaging, where anomalous samples are scarce and expensive to annotate. Zero-shot anomaly detection (ZSAD) [1] aims to identify and localize anomalies without target-domain training data. Vision-language models such as CLIP [2] provide strong semantic alignment, offering a promising foundation for ZSAD. However, existing CLIPbased methods still face two key limitations: coarse anomaly maps with imprecise boundaries, and prompt representations that converge to a narrow semantic subspace.

CLIP-based zero-shot anomaly detection approaches, including AnomalyCLIP [3], AdaCLIP [4], AA-CLIP [5],

Bayes-PFL [6], and MRAD [7], align normal/abnormal text prompts with image features to produce anomaly maps. While these methods improve generalization, their outputs remain coarse and boundary-insensitive. To enhance localization, SAM [8] and SAM2 [9] are combined with CLIP. Clip-SAM [10] extracts spatial prompts by applying pixel-level thresholding to CLIP’s coarse segmentation maps. However, such strategies rely on thresholding low-resolution outputs, leading to prompt drift when anomalies are absent or subtle. In prompt learning, MSGCoOp [11] adapts CLIP via multisemantic guided prompts, but it is not specifically adapted for the anomaly detection task, making it hard to adequately meet the demand of anomaly detection.

To address these issues, we propose PSMP-CLIP, a collaborative network that integrates patch-prompt SAM2 segmentation (PPSS) and multi-semantic guided prompt regularization (MSGPR). PPSS directly samples spatial prompts from intermediate patch features, avoiding up-sampling and thresholding on coarse masks. MSGPR uses four learnable prompt groups constrained by semantic anchors, where four groups of anchors are generated for normal and abnormal texts, respectively, to preserve generalization and enrich taskspecific semantics.

Our contributions are as follows:

1) We propose PPSS, which directly samples spatial prompts from intermediate patch features to avoid prompt coordinate drift, improving fine-grained anomaly localization.

2) We propose MSGPR, which uses multiple learnable prompt groups constrained by semantic anchors to suppress catastrophic forgetting and enrich task-specific semantics.

3) We conduct comprehensive experiments on 14 datasets. The results show that our method achieves highly competitive performance on pixel-level metrics under the zero-shot setting.

![](images/88f48ab79162e41be240a09b3c8835f8a223ed9a5d020e3750730d0321ce6154.jpg)  
Fig. 1. Overview of the proposed PSMP-CLIP network. PPSS enables SAM2 to produce precise boundary segmentations and MSGPR preserves CLIP’s general knowledge and enriches task-specific semantics.

## 2. METHOD

## 2.1. Overview

Fig. 1 illustrates the overall architecture. PSMP-CLIP builds on AA-CLIP as baseline and comprises two modules: PPSS and MSGPR. During training, the model jointly optimizes classification, segmentation, diversity, semantic guidance, and orthogonality losses. At inference, the final anomaly map is obtained by fusing CLIP coarse prediction with SAM2 fine segmentation.

## 2.2. Patch-Prompt SAM2 Segmentation (PPSS)

Multi-level patch feature extraction. From the 6th, 12th, 18th, and 24th layers of the CLIP image encoder, we extract features $\mathbf { F } _ { i }$ and align them to text dimension via trainable projections, then aggregate:

$$
\mathbf { V } _ { \mathrm { p a t c h } } = \sum _ { i = 1 } ^ { 4 } \operatorname { P r o j } _ { i } ( \mathbf { F } _ { i } ) .\tag{1}
$$

Patch-level anomaly scoring. Compute cosine similarity between $\mathbf { V } _ { \mathrm { p a t c h } }$ and the fused normal/abnormal text embeddings $\mathbf { T } ^ { \mathrm { { N } } } / \mathbf { \dot { T } } ^ { \mathrm { { A } } }$

$$
\mathbf { p } _ { \mathrm { s e g } } ^ { \mathrm { o } } = \cos \left( \mathbf { V } _ { \mathrm { p a t c h } } \cdot [ \mathbf { T } ^ { \mathrm { N } } , \mathbf { T } ^ { \mathrm { A } } ] \right) ,\tag{2}
$$

and define anomaly score ${ \mathbf s } ( n ) = { \mathbf p } _ { \mathrm { s e g } } ^ { \mathrm { A } } ( n ) - { \mathbf p } _ { \mathrm { s e g } } ^ { \mathrm { N } } ( n )$

Hybrid prompt generation. We sample positive points where $\mathbf { s } ( n ) > \tau$ and negative points where $\mathbf { s } ( n ) < - \tau$ directly on the patch grid:

$$
\mathcal { P } ^ { \mathrm { p a t c h } } = \{ { \bf p } _ { n } = 1 | \mathbf { s } ( n ) > \tau \} \cup \big \{ { \bf p } _ { n } = 0 | \mathbf { s } ( n ) < - \tau \big \} .\tag{3}
$$

The coordinates of each patch token are mapped to the original pixel space by taking the geometric center:

$$
x _ { n } ^ { \mathrm { c } } = \left( u _ { n } + \frac { 1 } { 2 } \right) \frac { W } { W _ { \mathrm { p a t c h } } } , \qquad y _ { n } ^ { \mathrm { c } } = \left( v _ { n } + \frac { 1 } { 2 } \right) \frac { H } { H _ { \mathrm { p a t c h } } } .\tag{4}
$$

We compute connected components of positive points in the patch grid, take the minimum bounding rectangle of each

Table 1. Pixel-level ZSAD (P-AUROC% / PRO%)
<table><tr><td>Domain</td><td>Datasets</td><td>AnomalyCLIP</td><td>AdaCLIP</td><td>AA-CLIP</td><td>Bayes-PFL</td><td>MRAD</td><td>Ours</td></tr><tr><td rowspan="6">Industrial</td><td>MVTec AD</td><td>(91.1,81.4)</td><td>(86.8,33.8)</td><td>(90.6,85.0)</td><td>(91.9,87.8)</td><td>(93.0,86.8)</td><td>(93.7,85.9)</td></tr><tr><td>VisA</td><td>(95.5,87.0)</td><td>(95.1,71.3)</td><td>(94.5,80.0)</td><td>(95.7,90.1)</td><td>(95.9,88.0)</td><td>(95.4,87.0)</td></tr><tr><td>BTAD</td><td>(94.2,74.8)</td><td>(87.7,17.1)</td><td>(95.2,72.3)</td><td>(93.9,76.6)</td><td>(95.4,72.8)</td><td>(95.5,74.8)</td></tr><tr><td>MPDD</td><td>(95.9,85.5)</td><td>(-,-)</td><td>(95.7,86.2)</td><td>(-,-)</td><td>(97.9,90.6)</td><td>(96.1,87.3)</td></tr><tr><td>DTD-Synthetic</td><td>(97.9,92.3)</td><td>(94.1,24.9)</td><td>(96.6,84.1)</td><td>(97.8,94.3)</td><td>(98.1,89.8)</td><td>(98.6,90.0)</td></tr><tr><td>DAGM</td><td>(95.6,91.0)</td><td>(97.0,40.9)</td><td>(93.1,82.7)</td><td>(99.3,98.0)</td><td>(97.4,90.3)</td><td>(96.6,89.0)</td></tr><tr><td rowspan="5">Medical</td><td>CVC-ClinicDB</td><td>(82.9,67.8)</td><td>(83.6,11.5)</td><td>(88.1,74.7)</td><td>(86.1,69.8)</td><td>(87.3,73.9)</td><td>(88.2,75.2)</td></tr><tr><td>CVC-ColonDB</td><td>(81.9,71.3)</td><td>(78.5,10.1)</td><td>(82.9,73.5)</td><td>(80.6,72.6)</td><td>(84.7,73.9)</td><td>(83.2,75.1)</td></tr><tr><td>TN3K</td><td>(81.5,50.4)</td><td>(82.1,49.8)</td><td>(81.6,47.7)</td><td>(-,-)</td><td>(-,-)</td><td>(84.8,51.9)</td></tr><tr><td>Endo</td><td>(84.1,63.6)</td><td>(84.5,12.1)</td><td>(90.4,75.7)</td><td>(84.8,63.2)</td><td>(88.3,71.6)</td><td>(91.3,77.8)</td></tr><tr><td>Kvasir</td><td>(78.9,45.6)</td><td>(-,-)</td><td>(86.7,55.1)</td><td>(85.4,63.9)</td><td>(84.3,52.7)</td><td>(89.3,56.3)</td></tr></table>

component, and map these rectangles to the original image as box prompts.

Mask confidence-weighted fusion. SAM2 produces K candidate masks with confidence scores $\{ S _ { k } \}$ . After softmax normalization, the fused SAM2 probability map is:

$$
\mathbf { P } _ { \mathrm { S A M 2 } } = \sum _ { k = 1 } ^ { K } { \frac { \exp ( S _ { k } ) } { \sum _ { i = 1 } ^ { K } \exp ( S _ { i } ) } } \mathbf { P } _ { k } ,\tag{5}
$$

where $\mathbf { P } _ { k } = { \mathrm { S i g m o i d } } ( \mathbf { Z } _ { k } )$ . The final result is $\mathbf { P } _ { \mathrm { f i n a l } } = ( 1 -$ $w _ { 0 } ) \mathbf { P } _ { \mathrm { C L I P } } + w _ { 0 } \mathbf { P } _ { \mathrm { S A M 2 } }$ , with $w _ { 0 } = 0 . 8$

## 2.3. Multi-Semantic Guided Prompt Regularization (MS-GPR)

We construct k=4 parallel learnable prompt groups for normal and abnormal texts. Each group is encoded by the adapter-equipped text encoder, and the fused features are obtained by averaging:

$$
\mathbf { T } ^ { \mathrm { N } } = { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } \mathbf { t } _ { i } ^ { \mathrm { N } } , \qquad \mathbf { T } ^ { \mathrm { A } } = { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } \mathbf { t } _ { i } ^ { \mathrm { A } } .\tag{6}
$$

To preserve general knowledge, we build k fixed semantic anchors for normal and abnormal states from templates like “A photo of [status] [CLS]” and encode them with the frozen CLIP text encoder. The averaged anchors are:

$$
\mathbf { A } ^ { \mathrm { N } } = { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } \mathbf { a } _ { i } ^ { \mathrm { N } } , \qquad \mathbf { A } ^ { \mathrm { A } } = { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } \mathbf { a } _ { i } ^ { \mathrm { A } } .\tag{7}
$$

The semantic guidance loss pulls the learnable prompt features toward the corresponding anchor features:

$$
\mathcal { L } _ { \mathrm { s g } } = \frac { 1 } { 2 } \left( 1 - \cos ( \mathbf { T } ^ { \mathrm { { N } } } , \mathbf { A } ^ { \mathrm { { N } } } ) + 1 - \cos ( \mathbf { T } ^ { \mathrm { { A } } } , \mathbf { A } ^ { \mathrm { { A } } } ) \right) .\tag{8}
$$

Diversity regularization penalizes pairwise similarity among prompt groups to encourage complementary semantics:

$$
\mathcal { L } _ { \mathrm { d i v } } ^ { \mathrm { t y p e } } = \frac { 1 } { k ( k - 1 ) } \sum _ { 1 \le i < j \le k } \cos ^ { 2 } \left( \mathbf { t } _ { i } ^ { \mathrm { t y p e } } , \mathbf { t } _ { j } ^ { \mathrm { t y p e } } \right) ,\tag{9}
$$

with ${ \mathcal { L } } _ { \mathrm { d i v } } = { \textstyle \frac { 1 } { 2 } } ( { \mathcal { L } } _ { \mathrm { d i v } } ^ { \mathrm { N } } + { \mathcal { L } } _ { \mathrm { d i v } } ^ { \mathrm { A } } )$ . Orthogonality loss separates normal and abnormal embeddings:

$$
\mathcal { L } _ { \mathrm { o r t h } } = \left| \langle \mathbf { T } ^ { \mathrm { N } } , \mathbf { T } ^ { \mathrm { A } } \rangle \right| ^ { 2 } .\tag{10}
$$

## 2.4. Training Objectives

Following the baseline [5] for alignment (BCE for classification, Dice+Focal for segmentation) and orthogonality, the total loss additionally includes diversity and semantic guidance:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } + \lambda _ { \mathrm { s g } } \mathcal { L } _ { \mathrm { s g } } + \lambda _ { \mathrm { o r t h } } \mathcal { L } _ { \mathrm { o r t h } } . } \\ { \mathbb { W } \mathrm { e } \mathrm { ~ s e t } \lambda _ { \mathrm { a l i g n } } = 1 , \lambda _ { \mathrm { d i v } } = 0 . 1 5 , \lambda _ { \mathrm { s g } } = 0 . 5 , \lambda _ { \mathrm { o r t h } } = 0 . 1 . } \end{array}\tag{11}
$$

## 3. EXPERIMENTS

## 3.1. Setup

We evaluate on 14 datasets: MVTec AD [12], VisA [13], BTAD [14], MPDD [15], DTD-Synthetic [16], DAGM [17], and CVC-ClinicDB [18], CVC-ColonDB [19], Endo [20], TN3K [21], HeadCT [22], BrainMRI [22], Br35H [23], Kvasir [24] (medical). Following the cross-dataset protocol, models are trained on VisA and evaluated on all others (for VisA, trained on MVTec AD). We use ViT-L-14-336 CLIP backbone and SAM2.1 hiera-large, with only adapters and prompts trainable. Metrics are pixel-level AUROC/AUPRO and image-level AUROC/AP.

## 3.2. Comparison with State-of-the-Art

Tables 1 and 2 provide a full comparison with recent methods. Our method achieves the best pixel AUROC on MVTec AD (93.7), BTAD (95.5), DTD-Synthetic (98.6), CVC-ClinicDB (88.2), TN3K (84.8), Endo (91.3) and Kvasir (89.3), and best pixel AUPRO on CVC-ClinicDB (75.2), CVC-ColonDB (75.1), TN3K (51.9) and Endo (77.8). The improvements stem from PPSS providing precise boundary refinement and MSGPR enhancing coarse localization.

Table 2. Image-level ZSAD (I-AUROC% / AP%)
<table><tr><td>Domain</td><td>Datasets</td><td>AnomalyCLIP</td><td>AdaCLIP</td><td>AA-CLIP</td><td>Bayes-PFL</td><td>MRAD</td><td>Ours</td></tr><tr><td rowspan="6">Industrial</td><td>MVTec AD</td><td>(91.5,96.1)</td><td>(92.0,96.4)</td><td>(91.6,96.1)</td><td>(92.0,96.2)</td><td>(94.0,97.4)</td><td>(93.8,97.1)</td></tr><tr><td>VisA</td><td>(82.1,85.4)</td><td>(83.0,84.9)</td><td>(81.5,85.1)</td><td>(87.0,89.0)</td><td>(85.7,88.3)</td><td>(83.0,85.5)</td></tr><tr><td>BTAD</td><td>(88.3,87.3)</td><td>(91.6,92.4)</td><td>(93.2,96.8)</td><td>(92.8,94.1)</td><td>(92.8,94.2)</td><td>(94.4,96.4)</td></tr><tr><td>MPDD</td><td>(74.1,78.2)</td><td>(-,-)</td><td>(73.0,79.3)</td><td>(-,-)</td><td>(81.8,83.4)</td><td>(77.8,82.5)</td></tr><tr><td>DTD-Synthetic</td><td>(93.5,97.0)</td><td>(92.8,97.0)</td><td>(91.9,91.3)</td><td>(97.2,99.0)</td><td>(96.0,98.4)</td><td>(94.0,93.5)</td></tr><tr><td>DAGM</td><td>(97.5,92.3)</td><td>(96.5,95.7)</td><td>(93.1,82.7)</td><td>(97.7,95.7)</td><td>(98.4,98.6)</td><td>(96.6,89.7)</td></tr><tr><td rowspan="3">Medical</td><td>BrainMRI</td><td>(90.3,92.2)</td><td>(94.4,93.1)</td><td>(91.5,91.3)</td><td>(94.3,88.4)</td><td>(97.0,97.4)</td><td>(93.5,93.5)</td></tr><tr><td>HeadCT</td><td>(93.4,91.6)</td><td>(91.4,92.2)</td><td>(95.7,92.2)</td><td>(91.9,91.1)</td><td>(97.1,97.6)</td><td>(96.7,96.5)</td></tr><tr><td>Br35H</td><td>(94.6,94.7)</td><td>(96.1,94.3)</td><td>(95.3,94.1)</td><td>(97.8,93.6)</td><td>(97.9,97.6)</td><td>(96.5,95.5)</td></tr></table>

![](images/19034462410ab096cf05297a919a59c8c5c61928e786f5621ba8cf822156313c.jpg)  
Fig. 2. Qualitative comparison of anomaly segmentation across industrial (left) and medical (right) domains.

Table 3. Module ablation results. P-AUC/P-PRO averaged over 11 datasets with pixel-level GT and I-AUC/I-AP averaged over 9 datasets with image-level labels. Best results are highlighted in bold.
<table><tr><td>Method</td><td>P-AUC</td><td>P-PRO</td><td>I-AUC</td><td>I-AP</td></tr><tr><td>Baseline (AA-CLIP)</td><td>90.5</td><td>74.3</td><td>89.6</td><td>89.9</td></tr><tr><td>+ PPSS</td><td>91.3</td><td>76.5</td><td>91.7</td><td>92.0</td></tr><tr><td>+ MSGPR</td><td>90.9</td><td>75.0</td><td>90.4</td><td>90.7</td></tr><tr><td>+ PPSS + MSGPR</td><td>92.1</td><td>77.3</td><td>91.8</td><td>92.2</td></tr></table>

## 3.3. Ablation Studies

Table 3 shows that adding PPSS alone improves P-AUC from 90.5 to 91.3 and P-PRO from 74.3 to 76.5, while I-AUC and I-AP rise to 91.7 and 92.0, indicating that SAM2 refinement helps both local and global discrimination. Adding MSGPR alone yields smaller pixel-level gains but improves imagelevel metrics to 90.4 and 90.7, mainly strengthening global semantics. Combining both achieves the best results (P-AUC 92.1, P-PRO 77.3, I-AUC 91.8, I-AP 92.2). The additional gains over individual components confirm the complementary synergy between PPSS and MSGPR.

Fig. 2 provides a visual comparison of anomaly segmentation results. In industrial scenarios, PSMP-CLIP yields clearer boundaries and better alignment with ground truth, even for subtle defects under complex backgrounds. Benefiting from rich multi-semantic prompts and patch-guided SAM2 refinement, the model identifies anomalies more accurately and suppresses false positives. These qualitative results further confirm the advantage of combining multi-semantic guided prompt regularization with fine-grained SAM2 segmentation.

## 4. CONCLUSION

We proposed PSMP-CLIP, a zero-shot anomaly detection network that synergizes patch-prompt SAM2 segmentation and multi-semantic prompt regularization. PPSS avoids prompt drift by sampling directly from patch anomaly scores, while MSGPR preserves general knowledge and enriches semantics. Experiments on 14 datasets show highly competitive pixel-level performance and strong cross-domain generalization. Future work will extend to video and few-shot settings.

## 5. REFERENCES

[1] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “Winclip: Zero-/few-shot anomaly classification and segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 19606–19616.

[2] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 8748–8763.

[3] Q. Zhou, G. Pang, Y. Tian, S. He, and J. Chen, “Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024, pp. 49705–49737.

[4] Y. Cao, J. Zhang, L. Frittoli, Y. Cheng, W. Shen, and G. Boracchi, “Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 55–72.

[5] W. Ma et al., “Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 4744–4754.

[6] Z. Qu et al., “Bayesian prompt flow learning for zero-shot anomaly detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 30398–30408.

[7] C. Xu, C. Lv, Q. Chen, F. Zhang, and Z. Zhang, “Mrad: Zero-shot anomaly detection with memory-driven retrieval,” arXiv preprint arXiv:2602.00522, 2026.

[8] A. Kirillov et al., “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 3992–4003.

[9] N. Ravi et al., “Sam 2: Segment anything in images and videos,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025, pp. 28085–28128.

[10] S. Li, J. Cao, P. Ye, Y. Ding, C. Tu, and T. Chen, “Clipsam: Clip and sam collaboration for zero-shot anomaly segmentation,” Neurocomputing, vol. 618, pp. 129122, 2025.

[11] Z. Wang, T. Sun, M. Du, Y. Huang, X. Xu, and Z. Li, “Msgcoop: Multiple semantic-guided context optimization for few-shot learning,” in Proc. Int. Conf. Virtual Reality Vis. (ICVRV), 2025, pp. 384–389.

[12] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Mvtec ad – a comprehensive real-world dataset for unsupervised anomaly detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9584–9592.

[13] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-the-difference self-supervised pre-training for anomaly detection and segmentation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 392–408.

[14] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, “Vt-adl: A vision transformer network for image anomaly detection and localization,” in Proc. IEEE Int. Symp. Ind. Electron. (ISIE), 2021, pp. 1–6.

[15] S. Jezek, M. Jonak, R. Burget, P. Dvorak, and M. Skotak, “Deep learning-based defect detection of metal parts: Evaluating current methods in complex conditions,” in Proc. Int. Congr. Ultra Modern Telecommun. Control Syst. Workshops (ICUMT), 2021, pp. 66–71.

[16] T. Aota, L. T. T. Tong, and T. Okatani, “Zero-shot versus many-shot: Unsupervised texture anomaly detection,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2023, pp. 5553–5561.

[17] M. Wieler, T. Hahn, and F. A. Hamprecht, “Weakly supervised learning for industrial optical inspection,” in Proc. DAGM Symp., 2007.

[18] J. Bernal, F. J. Sanchez, G. Fern´ andez-Esparrach, D. Gil,´ C. Rodr´ıguez, and F. Vilarino, “Wm-dova maps for ac- ˜ curate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” Computerized Medical Imaging and Graphics, vol. 43, pp. 99–111, 2015.

[19] J. Bernal, J. Sanchez, and F. Vilarino, “Towards auto-´ matic polyp detection with a polyp appearance model,” Pattern Recognition, vol. 45, no. 9, pp. 3166–3182, 2012.

[20] S. A. Hicks, D. Jha, V. Thambawita, P. Halvorsen, H. L. Hammer, and M. A. Riegler, “The endotect 2020 challenge: Evaluation and comparison of classification, segmentation and inference time for endoscopy,” in Proc. Int. Conf. Pattern Recognit. (ICPR), 2021, pp. 263–274.

[21] H. Gong et al., “Multi-task learning for thyroid nodule segmentation with thyroid region prior,” in Proc. IEEE Int. Symp. Biomed. Imaging (ISBI), 2021, pp. 257–261.

[22] M. Salehi, N. Sadjadi, S. Baselizadeh, M. H. Rohban, and H. R. Rabiee, “Multiresolution knowledge distillation for anomaly detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 14902–14912.

[23] A. Hamada, “Br35h :: Brain tumor detection 2020,” IEEE Dataport, 2025.

[24] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D. Johansen, “Kvasirseg: A segmented polyp dataset,” in Proc. Int. Conf. Multimedia Modeling (MMM), 2019, pp. 451–462.