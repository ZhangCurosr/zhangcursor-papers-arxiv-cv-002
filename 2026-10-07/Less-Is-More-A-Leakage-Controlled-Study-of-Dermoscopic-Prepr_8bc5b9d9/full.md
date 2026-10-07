# Less Is More: A Leakage-Controlled Study of Dermoscopic Preprocessing for Joint Skin Lesion Classification and Segmentation with YOLO26

Truong Viet Vu, Nguyen Chi Hai, Nguyen Phuc Nguyen, Ngo Hoang Tu, Vo Nguyen Quoc Bao, and Nguyen Thai Anh Faculty of Information Technology, Van Lang School of Technology, Van Lang University, Ho Chi Minh City, Vietnam Email: vu.2274802011045@vanlanguni.vn, hai.2274802010214@vanlanguni.vn, nguyen.2274802010586@vanlanguni.vn, tu.nh@vlu.edu.vn, bao.vnq@vlu.edu.vn, anh.nt@vlu.edu.vn

Abstract—Handcrafted preprocessing is widely employed in automated dermoscopic analysis to suppress imaging artifacts and enhance lesion visibility. Nevertheless, its actual contribution to modern real-time models remains unclear, particularly when evaluation protocols do not adequately control correlations among images of the same lesion. This study presents a leakagecontrolled, lesion-disjoint evaluation of dermoscopic preprocessing and augmentation for joint multi-class lesion classification and instance segmentation using a fixed nano-scale YOLO26 segmentation model (YOLO26n-seg). From HAM10000 (10,015 images), quality control yields 10,013 valid image–mask pairs from 7,468 unique lesions, partitioned into mutually exclusive sets by lesion identity. With the architecture, resolution, training budget, and evaluation protocol held fixed, we compare minimally processed images plus online augmentation against offline class balancing, DullRazor–CLAHE preprocessing, and raw–processed hybrid views, over three random seeds. On the lesion-disjoint test set, the raw baseline achieves a mask mAP<sub>50:95</sub> of 0.5636±0.0234, a Dice score of 0.9356 ± 0.0024, and a macro-F1 score of 0.6917 ± 0.0202. Offline augmentation does not improve the<sup>[</sup> mean performance, while the combined and hybrid strategies reduce both class-aware segmentation and classification accuracy. At only 2.69 million parameters, the model runs at approximately 50 frames per second. Under a leakage-controlled, lesiondisjoint protocol with all non-input factors held fixed, minimally processed dermoscopic images combined with standard online augmentation deliver a better accuracy–efficiency trade-off than increasingly complex deterministic preprocessing, which yields no consistent joint benefit across three seeds on HAM10000.

Index Terms—dermoscopy, skin lesion classification, instance segmentation, YOLO26, image preprocessing, lesion-disjoint evaluation

## I. INTRODUCTION

Skin lesion assessment relies strongly on visual characteristics such as color, texture, shape, and border structure. Dermoscopy provides non-invasive visualization of subsurface features and improves melanoma diagnostic accuracy over unaided examination [1], yet interpretation remains challenging because different lesion categories may appear similar while lesions within a category vary considerably. Deep neural networks have therefore emerged as decision-support tools that learn discriminative representations directly from dermoscopic images [2].

Automated dermoscopic analysis commonly involves lesion segmentation and classification. Segmentation delineates ab normal tissue, whereas classification assigns the lesion to a diagnostic category. Research on both tasks has been accelerated by public challenges organized by the International Skin Imaging Collaboration (ISIC) [3]. The Human Against Machine with 10,000 training images (HAM10000) dataset is a widely used benchmark comprising 10,015 dermoscopic images from seven diagnostic categories [4]. Because multiple images may represent the same lesion under different acquisition conditions, random image-level partitioning can place correlated observations in separate subsets and yield optimistic estimates of generalization. Lesion-level separation is therefore important for reliable medical machine-learning evaluation [5].

Dermoscopic images may contain hairs, uneven illumination, low contrast, and acquisition artifacts. Common preprocessing techniques include DullRazor hair removal [6], contrast-limited adaptive histogram equalization (CLAHE) [7], Gray World color normalization [8], and lesion-centered region-of-interest (ROI) extraction. Although visually motivated, these transformations may alter diagnostically relevant information or introduce artificial patterns. Consistent with this concern, Mahbod et al. found that segmentation-based image manipulation did not consistently improve dermoscopic classification and that removing surrounding context could reduce performance [9].

Data augmentation raises a related concern. Online augmentation applies stochastic transformations during training, whereas offline augmentation explicitly creates additional samples that can target minority classes but remain correlated with their sources and must be restricted to the training set [10]. Distinguishing the two is essential to avoid confounding preprocessing, sampling, and augmentation effects.

Classification and segmentation are often studied separately, although both objectives can be learned from shared dermoscopic representations [11]. Their performance is not interchangeable: accurate lesion overlap does not guarantee correct diagnosis, and correct classification may accompany an incomplete boundary. A joint system should therefore be assessed using both spatial and class-aware metrics [12], [13]. The YOLO family provides an efficient foundation for this setting by combining localization and class prediction within a unified framework [14], and its recent extensions add instance segmentation while preserving practical inference speed. In particular, the nano-scale YOLO26 segmentation variant (YOLO26n-seg) jointly predicts lesion location, diagnostic class, confidence, and a pixel-level mask [15]. Com parisons among recent YOLO generations further demonstrate that architecture and model scale affect accuracy–latency trade-offs, emphasizing the need for matched experimental conditions [16], [17].

Despite these advances, the benefit of dermoscopic preprocessing remains unclear because previous evaluations often vary data partitioning, architecture, resolution, augmentation, sampling, and prediction policies simultaneously. It is therefore necessary to determine whether increasingly complex preprocessing provides a genuine advantage under leakagecontrolled and otherwise matched conditions.

This study evaluates handcrafted dermoscopic preprocessing for lightweight joint multi-class lesion classification and instance segmentation under a lesion-disjoint protocol. Using a fixed YOLO26n-seg architecture, we compare minimally processed images, offline class-balancing augmentation, DullRazor–CLAHE preprocessing, individual preprocessing components, ROI transformations, and hybrid raw–processed training views, evaluating the principal configurations across three random seeds.

The main contributions of this study are as follows:

• Leakage-controlled evaluation: We show that the preexisting image-level split leaks 765 lesions (of 7,468) across subsets, and we construct a SHA-256–frozen, lesion-disjoint partition with zero cross-subset overlap.

• Controlled preprocessing analysis: Holding the architecture, resolution, training budget, thresholds, and evaluation procedure fixed, we isolate preprocessing and augmentation as the only varied factors across four principal configurations and seven component ablations.

• Finding and reproducible baseline: Over three seeds we jointly report classification, class-aware segmentation, boundary quality, model size, and speed, and we show that minimally processed inputs with online augmentation give the best accuracy–efficiency trade-off, establishing a reproducible lightweight baseline (2.69 M parameters, approximately 50 FPS).

The remainder of this paper is organized as follows. Section II presents the dataset preparation, preprocessing configurations, training protocol, and evaluation criteria. Section III reports and discusses the experimental results. Section IV concludes the paper and outlines directions for future research.

## II. METHODOLOGY

## A. Dataset and Annotations

The experimental corpus is constructed by matching HAM10000 metadata, dermoscopic images, and segmentation masks through $\mathrm { i m a g e \_ i d } .$ HAM10000 contains 10,015 images from seven classes [4]: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc).

Table I: Class distribution of the lesion-disjoint dataset.
<table><tr><td>Class</td><td>Train</td><td>Validation</td><td>Test</td><td>Total</td></tr><tr><td>akiec</td><td>229</td><td>49</td><td>49</td><td>327</td></tr><tr><td>bcc</td><td>360</td><td>77</td><td>77</td><td>514</td></tr><tr><td>bkl</td><td>768</td><td>165</td><td>165</td><td>1,098</td></tr><tr><td>df</td><td>80</td><td>18</td><td>17</td><td>115</td></tr><tr><td>mel</td><td>778</td><td>167</td><td>167</td><td>1,112</td></tr><tr><td>nv</td><td>4,693</td><td>1,006</td><td>1,006</td><td>6,705</td></tr><tr><td>vasc</td><td>98</td><td>22</td><td>22</td><td>142</td></tr><tr><td>Total images</td><td>7,006</td><td>1,504</td><td>1,503</td><td>10,013</td></tr></table>

Corresponding publicly released ISIC segmentation masks, which serve as the segmentation ground truth, are matched by image identifier [3].

Quality control excludes two uniformly foreground masks, ISIC\_0026042 and ISIC\_0029819, because they do not delineate lesion boundaries. The final corpus therefore contains 10,013 valid image–mask pairs from 7,468 lesions, distributed as shown in Table I.

Each binary mask is converted into a class-aware YOLO polygon from its largest valid external contour and simplified using the Douglas–Peucker algorithm [18]. The audit verifies image and mask readability, polygon validity, label completeness, and consistency with diagnostic metadata.

## B. Lesion-Disjoint Split and Leakage Audit

HAM10000 contains multiple images of some lesions; consequently, image-level splitting can place correlated observations in different subsets and overestimate generalization [5]. Indeed, the pre-existing image-level split contains 765 lesions spanning more than one subset.

We therefore group samples by lesion\_id and generate an approximately class-stratified 70%/15%/15% split using seed 42. The training, validation, and test sets contain 7,006, 1,504, and 1,503 images from 5,231, 1,117, and 1,120 lesions, respectively. All classes are represented in each subset, with zero lesion overlap. The assignments and file provenance are frozen in a manifest whose split-defining contents are recorded using a 256-bit Secure Hash Algorithm (SHA-256) digest.

## C. Preprocessing and Augmentation Strategies

Fig. 1 summarizes the pipeline, and Table II defines the four principal input strategies. Images and masks are letterboxed to 640 × 640 pixels using bilinear and nearest-neighbor interpolation, respectively.

DullRazor uses a $1 7 \times 1 7$ black-hat operator with threshold 10 [6], followed by Telea inpainting with radius 1 [19]. Contrast-limited adaptive histogram equalization (CLAHE) is applied to the lightness channel of the CIE $L ^ { * } a ^ { * } b ^ { * }$ (CIELAB) representation using a clip limit of 2.0 and an $8 \times 8$ grid [7].

Following the Gray World assumption [8], let $I _ { c } ( x , y )$ denote the intensity at location $( x , y )$ in channel $c \in \{ R , G , B \}$ of an image with height H and width W. The channel mean and common target mean are given, respectively, as

Table II: Principal preprocessing and augmentation configurations.
<table><tr><td>Configuration</td><td>Training views</td><td>Validation and test views</td><td>Offline augmentation</td></tr><tr><td>Raw + online augmentation</td><td>Raw image followed by label-safe letterboxing</td><td>Raw image followed by the same letterboxing</td><td>None</td></tr><tr><td>Offline class balancing</td><td>Baseline view plus three generated variants</td><td>Identical to the raw baseline</td><td>Train only</td></tr><tr><td>DullRazor-CLAHE</td><td>for every non-nv training image DullRazor hair removal followed by CIELAB CLAHE and letterboxing</td><td>The same deterministic preprocessing pipeline</td><td>None</td></tr><tr><td>Raw-processed hybrid</td><td>One raw view and one ROI-DullRazor-Gray World view for each training image</td><td>The minimally processed raw view</td><td>None</td></tr></table>

![](images/63acfd076e5cf9c6092c0ddbd35846dc5af55a0ca8d87d24472b13ad422f1836.jpg)  
Figure 1: Overview of the experimental pipeline.

$$
\mu _ { c } = \frac { 1 } { H W } \sum _ { y = 1 } ^ { H } \sum _ { x = 1 } ^ { W } I _ { c } ( x , y ) , \bar { \mu } = \frac { 1 } { 3 } \sum _ { c \in \{ R , G , B \} } \mu _ { c } .\tag{1}
$$

For $\mu _ { c } > 0 .$ , color normalization is computed as

$$
I _ { c } ^ { \prime } ( x , y ) = \mathrm { c l i p } _ { [ 0 , 2 5 5 ] } \left( I _ { c } ( x , y ) \frac { \bar { \mu } } { \mu _ { c } } \right) ,\tag{2}
$$

where $I _ { c } ^ { \prime } ( x , y )$ denotes the normalized intensity. If any channel mean is zero, the original image is retained.

Region-of-interest (ROI) extraction uses the bounding rectangle of the largest contour. A crop is accepted only if it covers at least 50% of the image and preserves all foreground pixels; accepted crops receive a 2% margin.

Offline augmentation jointly transforms images and masks using flips, affine and color transformations, and Gaussian blur. Three variants are generated for each non-nv training image using preparation seed 42; no offline variant is introduced into validation or test data.

## D. YOLO26 Joint Classification and Instance Segmentation

YOLO26n-seg is initialized from yolo26n-seg.pt and fine-tuned for the seven lesion classes [15]. Each predicted instance contains a box, confidence, class, and mask, enabling lesion classification and instance segmentation in one inference pass.

Because each image contains one primary lesion, only the single highest-confidence instance is retained for perimage evaluation, and each ground-truth mask is reduced to its largest external contour, so both masks are single-lesion by construction. A missing detection produces an explicit no\_detection classification outcome and an empty predicted mask, ensuring that detection failures remain in both task evaluations.

YOLO26n-seg is fixed across the principal preprocessing strategies. YOLO11n-seg and YOLO12n-seg serve as rawinput architecture controls, while YOLO26s-seg assesses the effect of increased capacity.

## E. Training Protocol

Except for the deterministic no-augmentation control, all principal configurations use the same 640 × 640 resolution, training budget, optimizer, and online augmentation policy. Training lasts up to 300 epochs with batch size 16. AdamW [20] uses an initial learning rate of $1 0 ^ { - 3 }$ , weight decay of $5 \times 1 0 ^ { - 4 }$ , three warm-up epochs, and cosine decay [21]. Early stopping uses a patience of 50 epochs, and the checkpoint with the best validation fitness is retained.

Online augmentation includes hue–saturation–value perturbation, rotations within ±30<sup>◦</sup>, 10% translation, 30% scaling, horizontal and vertical flips, and mosaic augmentation [22]; mosaic is disabled during the final ten epochs. A deterministic raw-image control disables all stochastic online augmentation.

Principal configurations are trained with seeds 42, 52, and 62 and component ablations with seed 42; validation data guide checkpoint selection, and test data are used only for final evaluation.

## F. Evaluation Metrics and Efficiency Measurement

Validation and test evaluation use a confidence threshold of 0.25, an intersection over union (IoU) threshold of 0.7 for non-maximum suppression (NMS), and a mask threshold of 0.5. Let P denote the highest-confidence predicted mask and G the union of ground-truth polygons. The true-positive (TP), false-positive (FP), and false-negative (FN) pixel counts are $\mathrm { T P } = | P \cap G | , \mathrm { F P } = | P \setminus G |$ , and $\operatorname { F N } = | G \setminus P |$ . Following Taha and Hanbury [23], the pixel-level overlap metrics are given by

$$
\mathrm { D i c e } = { \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } , \quad \mathrm { I o U } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .\tag{3}
$$

Given boundary precision $P _ { b }$ , boundary recall $R _ { b } ,$ , and a two-pixel matching tolerance, boundary quality is measured using the boundary $F _ { 1 }$ score (BF<sub>1</sub>), mathematically expressed as [24]

$$
\mathrm { B F } _ { 1 } = \frac { 2 P _ { b } R _ { b } } { P _ { b } + R _ { b } } .\tag{4}
$$

It is worth mentioning that for a missed detection, $P$ is empty, so Dice, IoU, and BF are zero.

Following the Microsoft Common Objects in Context (COCO) protocol [25], class-aware segmentation is measured using mask mean average precision (mAP) at an IoU threshold of 0.50, denoted by m $\mathrm { A P _ { 5 0 } } ,$ and across thresholds from 0.50 to 0.95, denoted by $\mathrm { m A P _ { 5 0 : 9 5 } }$ . Ultralytics mAP evaluation uses a confidence threshold of 0.001 and an NMS IoU threshold of 0.7.

Classification metrics include accuracy, balanced accuracy, macro-F1, per-class precision, recall, F1, and confusion matrices [26], [27]. The no\_detection category is retained as an incorrect outcome but excluded from the seven-class macro average.

Efficiency is measured at batch size 1 with 20 warm-up iterations and 100 memory-resident test images; timings cover Ultralytics preprocessing, inference, and postprocessing but exclude image loading. Following [28], we report parameter count, checkpoint size, mean latency, and throughput in frames per second (FPS). For mean latency $\bar { \ell } _ { \mathrm { m s } } .$ , FPS is computed by

$$
\mathrm { F P S } = \frac { 1 0 0 0 } { \bar { \ell } _ { \mathrm { m s } } } .\tag{5}
$$

## III. NUMERICAL RESULTS AND DISCUSSION

All results are obtained on the lesion-disjoint test set. Principal configurations are reported as mean ± standard deviation over the three seeds, and component ablations use seed 42.

## A. Main Preprocessing and Augmentation Comparison

As shown in Table III and summarized in Fig. 2, the raw baseline (raw + online augmentation) achieves the strongest overall performance, including the highest accuracy, balanced accuracy, macro-F1, boundary F1, and mask mAP. Removing online augmentation causes the largest decline, reducing macro-F1 from 0.6917 to 0.5893 and mask m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ from 0.5636 to 0.4271, confirming that the “less is more” finding concerns deterministic preprocessing rather than augmentation itself. DullRazor–CLAHE reduces both classification and class-aware segmentation performance, whereas the hybrid strategy attains the best binary overlap (Dice 0.9366, IoU 0.8978) but lower balanced accuracy, macro-F1, and mask mAP, reinforcing that overlap quality and diagnostic recognition are not interchangeable. Given the three-seed standard deviations, offline class balancing and the hybrid stay within one standard deviation of the baseline on the class-aware metrics, so the added complexity yields no measurable joint benefit; differences among the principal configurations are interpreted descriptively rather than as significant. All configurations share the same 2.69 M-parameter model, and the raw baseline gives the best accuracy–efficiency balance.

![](images/dd4048d8e568515328c34d558e52901d214852cf0a57ad1316f6b6bda8807a3e.jpg)  
Figure 2: Main test-set comparison of the four principal input strategies and the deterministic no-augmentation control. Points denote three-seed means, horizontal error bars denote standard deviations, and dashed lines mark the raw baseline (raw + online augmentation).

## B. Component Ablation Study

Table IV shows that ROI extraction produces the highest balanced accuracy, macro-F1, and mask mAP, whereas Gray World achieves the best Dice, IoU, and boundary F1. CLAHE slightly improves mask mAP over the matched raw reference but reduces accuracy and macro-F1, while DullRazor consistently weakens the main metrics. Combining ROI with DullRazor and either CLAHE or Gray World produces no additive benefit and generally performs below the raw reference, indicating that successive transformations may alter useful color, texture, or boundary cues. Since these ablations use a single seed, their differences are interpreted directionally: individual operations such as ROI extraction may lead several isolated metrics, yet these gains do not survive composition and are not reproduced as three-seed principal configurations. The seed-averaged evidence in Table III therefore governs the conclusion that increasingly complex preprocessing does not provide a consistent joint classification–segmentation advantage.

Table III: Three-seed test performance of the main YOLO26n-seg configurations. Acc., BAcc., and $\mathrm { B F _ { 1 } }$ denote accuracy, balanced accuracy, and boundary $\mathrm { F _ { 1 } }$ score, respectively; bold values indicate the best mean.
<table><tr><td>Input strategy</td><td>Acc.</td><td>BAcc.</td><td>Macro-F1</td><td>Dice</td><td>IoU</td><td>BF1</td><td> $\overline { { \mathrm { \mathbf { M a s k } \ m A P _ { 5 0 } } } }$ </td><td> $\mathrm { M a s k \ m A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>Raw + online augmentation</td><td>0.8210±0.0111</td><td>0.6672±0.0203</td><td>0.6917±0.0202</td><td>0.9356±0.0024</td><td>0.8971±0.0027</td><td> $\mathbf { \overline { { 0 . 4 9 2 2 } } } \pm \mathbf { 0 . 0 0 4 8 }$ </td><td> $\overline { { \mathbf { 0 . 7 3 6 4 } \pm \mathbf { 0 . 0 2 3 2 } } }$ </td><td>0.5636±0.0234</td></tr><tr><td>Raw without online augmentation</td><td>0.7753±0.0070</td><td>0.5707±0.0373</td><td>0.5893±0.0279</td><td>0.9228±0.0006</td><td>0.8815±0.0008</td><td> $0 . 4 5 8 2 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $0 . 5 6 5 4 { \scriptstyle \pm 0 . 0 3 6 6 }$ </td><td> $0 . 4 2 7 1 { \scriptstyle \pm 0 . 0 2 5 2 }$ </td></tr><tr><td>Offline class balancing</td><td>0.8150±0.0047</td><td>0.6654±0.0154</td><td>0.6870±0.0251</td><td>0.9332±0.0021</td><td>0.8943±0.0020</td><td> $0 . 4 8 4 1 { \scriptstyle \pm 0 . 0 0 5 1 }$ </td><td>0.7305±0.0168</td><td>0.5614±0.0131</td></tr><tr><td>DullRazor-CLAHE</td><td>0.8026±0.0010</td><td>0.6090±0.0147</td><td>0.6356±0.0129</td><td>0.9304±0.0036</td><td>0.8900±0.0038</td><td> $0 . 4 7 1 3 { \scriptstyle \pm 0 . 0 0 2 1 }$ </td><td>0.6940±0.0133</td><td>0.5192±0.0122</td></tr><tr><td>Raw-processed hybrid</td><td>0.8186±0.0058</td><td>0.6252±0.0161</td><td>0.6537±0.0257</td><td>0.9366±0.0006</td><td>0.8978±0.0006</td><td>0.4846±0.0050</td><td> $0 . 7 1 1 9 { \scriptstyle \pm 0 . 0 0 8 9 }$ </td><td>0.5466±0.0074</td></tr></table>

Table IV: Component ablation on the lesion-disjoint test set using seed 42. The raw reference is the single-seed counterpart of the three-seed raw baseline in Table III. Bold values indicate the best result in each column.
<table><tr><td>Configuration</td><td> $\operatorname { A c c } .$ </td><td>BAcc.</td><td>Macro-F1</td><td>Dice</td><td>IoU</td><td> $\mathrm { B F _ { 1 } }$ </td><td> $\mathrm { M a s k \ m A P _ { 5 0 } }$ </td><td> $\mathbf { M a s k \ m A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>Raw (seed 42)</td><td>0.8310</td><td>0.6489</td><td>0.6796</td><td>0.9331</td><td>0.8943</td><td>0.4869</td><td>0.7340</td><td>0.5520</td></tr><tr><td>DullRazor only</td><td>0.8097</td><td>0.6020</td><td>0.6440</td><td>0.9329</td><td>0.8927</td><td>0.4821</td><td>0.6967</td><td>0.5224</td></tr><tr><td>CLAHE only</td><td>0.8150</td><td>0.6505</td><td>0.6623</td><td>0.9315</td><td>0.8927</td><td>0.4846</td><td>0.7366</td><td>0.5664</td></tr><tr><td>Gray World only</td><td>0.8064</td><td>0.6499</td><td>0.6730</td><td>0.9371</td><td>0.8987</td><td>0.4940</td><td>0.7261</td><td>0.5605</td></tr><tr><td>ROI only</td><td>0.8263</td><td>0.6524</td><td>0.6945</td><td>0.9341</td><td>0.8962</td><td>0.4934</td><td>0.7405</td><td>0.5729</td></tr><tr><td> $\mathrm { R O I } + \mathrm { D u l l R a z o r } + \mathrm { C L A H E }$ </td><td>0.8097</td><td>0.6310</td><td>0.6580</td><td>0.9303</td><td>0.8902</td><td>0.4736</td><td>0.6942</td><td>0.5123</td></tr><tr><td> $\mathrm { R O I + D u l l R a z o r + G r a y \ W o r l d }$ </td><td>0.8104</td><td>0.6225</td><td>0.6586</td><td>0.9293</td><td>0.8899</td><td>0.4823</td><td>0.7076</td><td>0.5396</td></tr></table>

Table V: Raw-input architecture and efficiency comparison. M and MiB denote millions of parameters and mebibytes; results are mean ± standard deviation over three seeds, and YOLO26s-seg is reported for seed 42. FPS is the per-seed mean of (5), which differs slightly from $1 0 0 0 / \bar { \ell } _ { \mathrm { m s } }$ as the reciprocal and mean do not commute.
<table><tr><td>Architecture</td><td>Params (M)</td><td>Size (MiB)</td><td>Macro-F1</td><td>Mask mAP50:95</td><td>Latency (ms)</td><td>FPS</td></tr><tr><td>YOLO11n-seg</td><td>2.84</td><td>5.76</td><td>0.6734±0.0068</td><td>0.5501±0.0150</td><td>25.51±4.39</td><td> $\overline { { 3 9 . 9 2 \pm 6 . 2 8 } }$ </td></tr><tr><td>YOL012n-seg</td><td>2.78</td><td>5.74</td><td>0.6962±0.0178</td><td>0.5758±0.0067</td><td>20.17±6.65</td><td>54.33±21.76</td></tr><tr><td>YOLO26n-seg</td><td>2.69</td><td>6.28</td><td>0.6917±0.0202</td><td>0.5636±0.0234</td><td>20.48±4.24</td><td>50.14±9.42</td></tr><tr><td>YOLO26s-seg</td><td>10.37</td><td>22.29</td><td>0.7239</td><td>0.5945</td><td>14.26</td><td>70.13</td></tr></table>

## C. Architecture and Efficiency Analysis

Among the nano-scale models, YOLO12n-seg achieves the highest mean macro-F1 (0.6962) and mask $\mathrm { m A P _ { 5 0 : 9 5 } }$ (0.5758), whereas YOLO26n-seg uses the fewest parameters (2.69 M). YOLO12n-seg attains a measured throughput of 54.33±21.76 FPS, but the large run-to-run variation, which reflects system contention on a single shared GPU, together with the separately executed benchmarks, makes latency rankings descriptive. YOLO26s-seg further improves the seed-42 scores but requires 3.85 times as many parameters and a checkpoint 3.55 times as large as YOLO26n-seg. Overall, YOLO12nseg and YOLO26n-seg are statistically indistinguishable on the accuracy metrics across three seeds (overlapping standard deviations), with YOLO26n-seg the most parameter-efficient; this equivalence does not alter the preprocessing comparison, for which YOLO26n-seg is fixed across all strategies.

## D. Per-Class and Qualitative Analysis

For the raw baseline, the mean per-class F1 scores over three seeds are 0.9127 for nv, 0.8008 for vasc, 0.7449 for bcc, 0.7238 for df, 0.6971 for bkl, 0.5423 for mel, and 0.4204 for akiec. The low akiec F1 is primarily associated with a recall of 0.2789 despite a precision of 0.8585, meaning that most akiec lesions are missed and assigned to other classes; this is a safety-relevant limitation for a pre-malignant category, driven by its low support and overlap with bkl. The confusion matrices further show that the dominant melanoma error is mel→nv (melanoma predicted as benign nevus), the most safety-critical failure mode, together with akiec→bkl confusion, consistent with class imbalance and overlapping dermoscopic appearance. Fig. 3 illustrates this task dependence: the model accurately delineates representative mel and nv lesions, but some highly overlapping masks are assigned to the wrong diagnostic class. In particular, Dice scores of 0.991 and 0.936 accompany mel→nv and nv→bkl errors, respectively, confirming that geometric agreement alone is insufficient for evaluating joint lesion analysis and that a segmentation-only score would conceal these diagnostic errors.

![](images/71b9666e7513481477d7c55ca68dae1928c0879d34aa199904ab3011a4fd4752.jpg)  
Figure 3: Qualitative test predictions of the raw YOLO26n-seg baseline using seed 52. The upper examples are correctly classified, whereas the lower examples retain high mask overlap but receive incorrect diagnostic labels.

## E. Discussion and Limitations

The results favor minimally processed dermoscopic images combined with online augmentation: only online augmentation yields a broad gain, whereas deterministic preprocessing trades isolated overlap improvements for weaker class-aware recognition. Thus, “less is more” refers to limiting deterministic preprocessing rather than removing augmentation. These results characterize a research baseline for method comparison and do not establish diagnostic or triage utility; the high

Dice partly reflects the single-lesion, lesion-filling geometry of HAM10000 rather than clinically difficult boundary cases. The findings also remain bounded by the HAM10000 acquisition domain, class imbalance, and limited minority-class support. Moreover, three seeds provide limited evidence for small differences, while the ablations and YOLO26s control are directional seed-42 comparisons. The single-lesion prediction policy reflects the dataset structure but not multi-lesion scenes, and the reported throughput excludes image loading and offline preprocessing, representing model execution rather than complete workflow latency.

## IV. CONCLUDING REMARKS AND OPEN PROBLEMS

This study presented a leakage-controlled, lesion-disjoint evaluation of dermoscopic preprocessing for joint multiclass classification and instance segmentation. Under a fixed YOLO26n-seg with all non-input factors held constant, minimally processed images with online augmentation delivered the strongest overall balance of classification, class-aware segmentation, boundary quality, and efficiency (2.69 M parameters, approximately 50 FPS), whereas increasingly complex deterministic preprocessing produced no consistent joint benefit across three seeds. This does not imply that preprocessing is unhelpful in general: individual operations improved isolated metrics, and cross-device acquisition may shift the balance. The architecture controls showed YOLO12n-seg and YOLO26n-seg to be statistically indistinguishable on accuracy, with YOLO26n-seg the most parameter-efficient. These results establish a reproducible lightweight baseline and emphasize evaluating recognition and segmentation together. Future work should examine cross-dataset and cross-device generalization, improve minority-class recognition, extend evaluation to multilesion settings, and investigate adaptive or learnable normalization under the same leakage-controlled protocol.

## ACKNOWLEDGMENT

This research is funded by Van Lang University, Vietnam under grant number 2510-DT-VLT-KCT-SV-003.

## REFERENCES

[1] H. Kittler, H. Pehamberger, K. Wolff, and M. Binder, “Diagnostic accuracy of dermoscopy,” Lancet Oncol., vol. 3, no. 3, pp. 159–165, 2002.

[2] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun, “Dermatologist-level classification of skin cancer with deep neural networks,” Nature, vol. 542, no. 7639, pp. 115–118, 2017.

[3] N. C. F. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. W. Dusza, D. A. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. A. Marchetti, H. Kittler, and A. Halpern, “Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC),” arXiv preprint arXiv:1902.03368, 2019. [Online]. Available: https://arxiv.org/abs/1902.03368

[4] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Sci. Data, vol. 5, p. 180161, 2018.

[5] P. Rouzrokh, B. Khosravi, S. Faghani, M. Moassefi, D. V. Vera Garcia, Y. Singh, K. Zhang, G. M. Conte, and B. J. Erickson, “Mitigating bias in radiology machine learning: 1. data handling,” Radiol. Artif. Intell., vol. 4, no. 5, p. e210290, 2022.

[6] T. K. Lee, V. Ng, R. P. Gallagher, A. Coldman, and D. I. McLean, “DullRazor: A software approach to hair removal from images,” Comput. Biol. Med., vol. 27, no. 6, pp. 533–543, 1997.

[7] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in Graphics Gems IV, P. S. Heckbert, Ed. San Diego, CA, USA: Academic, 1994, pp. 474–485.

[8] G. Buchsbaum, “A spatial processor model for object colour perception,” J. Franklin Inst., vol. 310, no. 1, pp. 1–26, 1980.

[9] A. Mahbod, P. Tschandl, G. Langs, R. Ecker, and I. Ellinger, “The effects of skin lesion segmentation on the performance of dermatoscopic image classification,” Comput. Methods Programs Biomed., vol. 197, p. 105725, 2020.

[10] C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,” J. Big Data, vol. 6, no. 1, p. 60, 2019.

[11] X. Yang, Z. Zeng, S. Y. Yeo, C. Tan, H. L. Tey, and Y. Su, “A novel multi-task deep learning model for skin lesion segmentation and classification,” arXiv preprint arXiv:1703.01025, 2017. [Online]. Available: https://arxiv.org/abs/1703.01025

[12] P. N. Nguyen, T. T. H. Dang, V. V. Truong, H. T. Ngo, and T. A. Nguyen, “Swin-PResU: Hybrid SwinViT and pruned residual CNN for precise brain tumor MRI segmentation,” in Proc. IEEE Int. Conf. Comput. Commun. Technol. (RIVF), 2025, pp. 149–154.

[13] V. V. Truong, P. N. Nguyen, T. T. H. Dang, A. T. Nguyen, T. T. Tran, N. Q. B. Vo, and H. T. Ngo, “Automated wood and leaf separation: A comprehensive review on advanced methods and forestry applications,” IEEE Access, vol. 14, 2026.

[14] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788.

[15] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, “Ultralytics YOLO26: Unified real-time end-to-end vision models,” arXiv preprint arXiv:2606.03748, 2026. [Online]. Available: https://arxiv.org/abs/2606.03748

[16] Q. M. Pham, N. T. Lang, H. T. Ngo, and T. A. Nguyen, “Fruit ripeness detection: A comparative analysis of state-of-the-art YOLO models,” in Proc. IEEE Int. Conf. Comput. Commun. Technol. (RIVF), 2025, pp. 699–704.

[17] C. H. Nguyen, H. S. H. Le, M. P. Nguyen, H. T. Ngo, and T. A. Nguyen, “A hybrid framework for semi-automated traffic annotation: Integrating YOLOv11 object detection with CLIP semantic verification,” in Proc. IEEE Int. Conf. Comput. Commun. Technol. (RIVF), 2025, pp. 203–208.

[18] D. H. Douglas and T. K. Peucker, “Algorithms for the reduction of the number of points required to represent a digitized line or its caricature,” Cartographica, Int. J. Geogr. Inf. Geovis., vol. 10, no. 2, pp. 112–122, 1973.

[19] A. Telea, “An image inpainting technique based on the fast marching method,” J. Graph. Tools, vol. 9, no. 1, pp. 23–34, 2004.

[20] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[21] ——, “SGDR: Stochastic gradient descent with warm restarts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017. [Online]. Available: https://openreview.net/forum?id=Skq89Scxx

[22] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal speed and accuracy of object detection,” arXiv preprint arXiv:2004.10934, 2020. [Online]. Available: https://arxiv.org/abs/2004. 10934

[23] A. A. Taha and A. Hanbury, “Metrics for evaluating 3d medical image segmentation: Analysis, selection, and tool,” BMC Med. Imaging, vol. 15, no. 1, p. 29, 2015.

[24] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 724–732.

[25] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Proc. Eur. Conf. Comput. Vis. (ECCV). Springer, 2014, pp. 740–755.

[26] M. Sokolova and G. Lapalme, “A systematic analysis of performance measures for classification tasks,” Inf. Process. Manage., vol. 45, no. 4, pp. 427–437, 2009.

[27] K. H. Brodersen, C. S. Ong, K. E. Stephan, and J. M. Buhmann, “The balanced accuracy and its posterior distribution,” in Proc. IEEE 20th Int. Conf. Pattern Recognit. (ICPR), 2010, pp. 3121–3124.

[28] S. Bianco, R. Cadene, L. Celona, and P. Napoletano, “Benchmark analysis of representative deep neural network architectures,” IEEE Access, vol. 6, pp. 64 270–64 277, 2018.