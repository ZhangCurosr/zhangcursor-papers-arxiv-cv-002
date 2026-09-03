# Morphology signal in whole slide image foundation models can automatically triage slides

Ayushi Sinha<sup>1</sup>, Shashank Yadav<sup>2</sup>, Benjamin Holmes<sup>3</sup>, Pravat Das<sup>3</sup>, Aaron W.   
Bogan<sup>4</sup>, James S. Lewis Jr.<sup>5</sup>, Santiago Romero-Brufau<sup>6</sup>, Andrew Y. K. Foong<sup>2</sup>, Scott H. Kaufmann<sup>7</sup>, Kathryn M. Van Abel<sup>6</sup>, David M. Routman<sup>2</sup>, and Michael R. Lucas<sup>3</sup>

<sup>1</sup> Department of Radiology, Mayo Clinic, Rochester MN Department of Radiation Oncology, Mayo Clinic, Rochester MN <sup>3</sup> AI Program, Mayo Clinic, Rochester MN

4 Department of Quantitative Health Sciences, Mayo Clinic, Scottsdale AZ Department of Laboratory Medicine and Pathology, Mayo Clinic, Scottsdale AZ Department of Otolaryngology-Head and Neck Surgery, Mayo Clinic, Rochester MN 7 Department of Oncology, Mayo Clinic, Rochester MN lucas.michael2@mayo.edu

Abstract. Patient exams in the cancer diagnosis and staging process typically generate several whole slide images (WSIs). One of the initial steps in training models on WSI data is identifying one or a few slides containing tumor or other diagnostic biomarkers necessary for downstream prediction tasks such as estimating recurrence risk or progressionfree survival. This step requires tedious manual curation by experienced pathologists. Many published datasets make the artificial assumption of 1 slide per patient. Alternatively, all slides per patient may be used for model training, which may dilute the signal from the few slides containing tumor or other relevant information. In this paper, we present a pipeline to overcome these challenges using publicly available WSI foundation models (FMs). Our evaluations show that ranking WSIs based on predictions from zero-shot classification using WSI FMs accurately identifies slides with the most tumor, indicating that WSI FMs contain suficient morphology signal to automatically triage slides. We also present a formulation for ranked evaluation to benchmark FM performance in slide triage. We show, on multiple datasets, that tumor slides are identified in the top-2 ranked slides for patients with up to 43 slides.

Keywords: whole slide image · slide triage · foundation models

## 1 Introduction

Cancer diagnosis and staging typically involve biopsy and resection specimens, which are prosected, sectioned, processed into parafin-embedded blocks, processed into glass slide sections, and finally reviewed by the pathologist for diagnosis. With the growth of digital pathology, slides are increasingly being digitized to generate whole slide images (WSIs). This process typically generates a large number of WSIs per patient, particularly for resection specimens. However, only a few WSIs contain tumor or other diagnostic biomarkers that may be useful for downstream prediction tasks such as estimating recurrence risk or progression-free survival. Current practice requires a pathologist to triage all generated WSIs per patient by visually analyzing and identifying those with primary tumor and/or lymph nodes involved by metastasis, which can take anywhere from about 30−90s per slide [8]. Automatically identifying the one or few most relevant slides per patient is a crucial, albeit overlooked, step in WSI model training. In this paper, we show that foundation models (FMs) trained jointly on histopathology image and text data may contain suficient morphology signal to automatically triage slides without the need for manual curation.

![](images/563cd284b0caf0f17ae64a91a0b84f8121d26011da6a4a58859338061bc4fbfb.jpg)  
Fig. 1: Schematic diagram of the presented pipeline which takes takes all slides per patient and ranks them to enable selection of the top-k slides. The shade of blue depicts, as an example, the percent tumor in each slide with darker blue denoting more tumor. From left to right, the slides are first segmented to remove background pixels, after which patches are extracted from the tissue regions. FMs are used to generate image embeddings at the patch- and slide-level, which are then used along with text embeddings generated from class specific prompts to classify the image. At the patch-level, classification results are aggregated by computing the percent patches in a class per slide, which are then used to rank the slides. At the patch-level, class probabilities are used directly to rank slides.

WSIs present several computational challenges. A typical WSI is digitized at various magnifications ranging from 0.25 microns per pixel (mpp) to 2mpp. At the highest magnification levels, images may be tens of thousands of pixels in resolution. To process images of such magnitude, WSIs are split into smaller patches $( \mathrm { e . g . , 2 5 6 \times 2 5 6 ~ p x }$ or $5 1 2 \times 5 1 2 ~ \mathrm { p x } )$ . Histopathology models are subsequently trained at the patch level, with FMs generating patch-level embeddings [4, 11, 14, 3, 2]. Patch-level embeddings can be aggregated across WSIs to generate slidelevel embeddings [6, 12, 16, 15, 18]. Downstream training tasks, such as predicting recurrence risk, also typically use patch-level embeddings along with aggregation methods such as multiple instance learning (MIL) to learn slide-level labels [5]. Attention-based MIL (AB-MIL) proposes using attention to create weighted representations for each slide or patient by learning which patches contribute most to the prediction [9]. When datasets have not been manually curated to select a single slide per patient for prediction tasks, patches from all slides may be used [17, 13]. However, each slide can generate thousands of patches, diluting the signal from the few patches containing tumor or other relevant information.

To overcome this limitation, we perform zero-shot classification using publicly available pathology FMs by computing cosine similarity between image and text embeddings to assess tumor content in WSIs. While FMs have been used to classify slides both by fine-tuning [7] and using zero-shot classification [1], these have primarily been done for cancer subtyping and not for slide triage. Using the pipeline presented in this paper (Fig. 1), we explore whether pathology FMs contain suficient morphology signal to automatically triage slides. We evaluate patch- and slide-level embeddings, and present a formulation for ranked evaluation that takes into consideration the number of slides per patient, the number of slides with tumor content, and the rank of the accurately identified slides. This formulation is used to benchmark models for slide triage on 2 datasets. We show that by ranking WSIs by tumor content, we can accurately identify slides with the most tumor, demonstrating that automatic slide triage is possible.

## 2 Methods

## 2.1 Datasets

Two datasets consisting of WSIs and accompanying pathology reports were used to evaluate our pipeline. Pathology reports were used to extract annotations that were used for evaluation. Pathology reports are organized by anatomical specimens received into parts, which are further organized, after sectioning, processing, and embedding, into blocks of tissue. Parts are assigned letters and blocks are assigned numerical values to generate block codes (e.g., A1, A2, B5, etc.). Each block may generate multiple slides, although most blocks in our datasets only contained a single slide. Each slide is identified only by its block code. A single pathology report representing findings from all slides is generated per patient exam. The presence or absence of cancer is typically noted at the part level. Details of the two datasets are provided below:

HN: The head & neck cancer dataset was compiled from 1233 patients consisting of 55597 WSIs. These slides were drawn from patients with cancer specimens mainly from the neck, tonsils, tongue, pharynx, and cervical lymph nodes. Slide-level annotations (e.g., slide with most tumor, etc.) were not available for this dataset. However, pathology reports were used to extract coarse part-level annotations. That is, if part A was positive for cancer, then all blocks in part A (i.e., A1, A2, . . . ) were labeled as cancer positive, although only one or few of these blocks may actually be cancer positive. When pathology reports identified cancer positivity at specific complexity levels, the annotation was refined to only include block codes associated with those complexity levels within the part.

For a subset of the data, lymph nodes involved by metastases were identified at the block-level in the pathology report. This enabled us to curate a HN-node subset of the HN dataset consisting of 3960 lymph node slides where, for 199 patients, involved lymph nodes were identified at a more granular level than in the full HN dataset.

OV: The ovarian cancer dataset was compiled from 35 patients consisting of 1068 slides drawn mainly from the uterus, ovaries, peritoneum, omentum, and pelvic lymph nodes. It was manually curated by a physician by visually evaluating each slide and identifying one or few blocks per patient with the most tumor content. The block with the most tumor content was annotated as the primary block, while any additional blocks identified per patient as also containing tumor were annotated as secondary blocks. All annotations, verified by a pathologist, were used only for evaluation.

## 2.2 Data processing and embedding generation

All WSIs were segmented using the HEST segmenter [10] to remove background pixels, and 512 × 512 patches were extracted from the tissue region at 20× magnification (0.5mpp). Embeddings were generated using the following FMs:

– CONCH, which uses contrastive learning objectives in a ViT-based architecture to align image and text modalities into a shared embedding space [11]. CONCH generates patch-level embeddings.

mean-CONCH, which mean-pools patch-level CONCH embeddings into slide-level embeddings.

– TITAN, which leverages patch-level embeddings from CONCH v1.5 in a vision-only self-supervised learning paradigm and fine-tunes the resulting model using contrastive alignment with text data [6]. TITAN generates slidelevel embeddings.

## 2.3 Zero-shot classification and slide ranking

Zero-shot classification was performed using text prompts designed to classify image embeddings into the following classes: {tumor, lymphocyte, other}. Prompts for tumor and other classes were adapted per cancer type (Tables A1, A2, A3). An ensemble of prompts was used for each class to account for the variability in verbiage across pathologists (Table A4). Text embeddings were generated for each prompt and a mean embedding was computed per class.

Cosine similarity was computed between image and text embeddings and class probabilities were calculated by applying softmax over the similarity scores. Class predictions were assigned to the class with the highest probability. For mean-CONCH and TITAN, class probabilities were used directly to rank slides. Patch-level embeddings from CONCH were further aggregated using class predictions per patch to compute the percent patches per slide assigned to a class, which was then used to rank slides. The top-k slides were selected based on these rankings and any slides with class probability 0 or 0% patches in a class are excluded from the top-k. Therefore, a top-k selection can generate a set of slides smaller than k.

## 3 Evaluation

## 3.1 Recall@k

We use recall@k to evaluate whether any of the annotated block codes are included in the top-k slides ranked by the 3 models. We use a variant of recall@k common in retrieval tasks, where recall per patient is defined as:

$$
\mathrm { r e c a l } 1 _ { i } ^ { k } = \left\{ \begin{array} { l l } { { 1 } } & { { \mathrm { i f ~ a t ~ l e a s t ~ 1 ~ a n n o t a t e d ~ b l o c k ~ c o d e ~ i s ~ i n ~ t h e ~ t o p - } k } } \\ { { 0 } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.\tag{1}
$$

Recall@k across a dataset of N patients can then be calculated as:

$$
\mathtt { r e c a l 1 } ^ { k } = \frac { \sum _ { 0 < i < N } \mathtt { r e c a l 1 } _ { i } ^ { k } } { N }\tag{2}
$$

This evaluation generates insights into the number of patients in a dataset for whom an annotated block code was in the top-k ranked slides.

## 3.2 Agreement between models

Agreement among models is computed at each k to assess whether they produce the same or similar ranking across patients. Agreement is calculated by computing the intersection of the sets of top block codes ranked by each model at each $k ,$ and taking the cardinality of the intersection. This value is normalized by the cardinality of the ideal intersected set, which is k × num\_patients for each k.

## 3.3 Ranked evaluation

Finally, we perform a ranked evaluation since the dificulty of the ranking task varies per patient due to 3 main factors:

Pool size: Each patient exam results in a diferent number of WSIs (Fig. 2A). The top-k slide selection task becomes more dificult as the number of slides increases. Therefore, slide triage from a larger pool gets a higher weight:

$$
\mathtt { p o o l \_ r e w a r d } _ { i } = \left( \log ( \mathtt { p o o l \_ s i z e } _ { i } ) / \log ( \mathtt { M A X \_ P O D L \_ S I Z E } ) \right) ^  0 . 5\tag{3}
$$

Annotated set size: Coarse annotations available at the part level present an easier evaluation task. Conversely, fine-grained annotations consisting of one or a few block codes present a more dificult evaluation task. Therefore, smaller annotated sets are assigned a higher weight:

$$
\mathsf { s e t \_ s i z e \_ r e w a r d } _ { i } = \frac { \log ( \mathsf { p o o l \_ s i z e } _ { i } - \mathsf { s e t \_ s i z e } _ { i } + 1 ) } { \log ( \mathsf { p o o l \_ s i z e } _ { i } ) }\tag{4}
$$

The +1 term ensures a positive numerator and that set\_size\_reward is 0 when the set\_size equals the pool\_size, in which case any ranking would yield block codes in the annotated set.

![](images/045e7bbce0573ca72a729a778fe9e936c97832129c791f5980d4541cd8ac2537.jpg)  
B

![](images/efc180a73c2d7ff75f3d23a2eddfe3a58af845e1abfeffd81a24b238c4c0018e.jpg)

![](images/b30f43c6433bac7b846bb0b17b69463efc80479e62b5ffcc86d231bf07c9376d.jpg)

![](images/844b4d862200be10bf8f7b358b48a92916b056b4eadc4692d77c048adcd19367.jpg)  
D

![](images/2ba11982dd90b401f78d33f8ebafb9d7061fd9cce549a68d390082f78892a581.jpg)

![](images/d37fed52e09905ba30cd967cd7ec8a06137581efa61135ebf65d84713cc51687.jpg)  
Fig. 2: A: Distribution of slides per patient across the HN (top) and OV (bottom) datasets. The HN dataset has slides in the range [1, 154] per patient with an average of $3 6 . 4 \pm 1 5 . 8$ slides per patient, while the HN-node dataset has slides in the range [6, 54] per patient with an average of $1 9 . 9 \pm 8 . 9$ slides per patient. The OV dataset has slides in the range [1, 81] per patient with an average of $3 0 . 5 \pm 1 8 . 4$ slides per patient. B: All models show similarly high recall at all k on the HN dataset, where slides with tumor are labeled coarsely at the part level. C: Patch-level classification from CONCH aligns accurately with slide morphology. D: On the OV dataset, where only 1 or a few slides with tumor are annotated per patient, recall at lower k is low but quickly increases with higher k.

Rank: Finally, higher rank for correctly triaged slide(s) is assigned higher weight:

$$
\mathbf { r a n k _ { - p e n a l t y } } _ { i } = 1 / \log _ { 2 } ( \mathbf { r a n k } _ { i } + 1 )\tag{5}
$$

The $+ 1$ in the denominator ensures that there is no division by 0. The final score for the ith patient is simply:

$$
\mathtt { s c o r e } _ { i } = \mathtt { p o o l \_ r e w a r d } _ { i } \times \mathtt { s e t \_ s i z e } _ { - } \mathtt { r e w a r d } _ { i } \times \mathtt { r a n k \_ p e n a l t y } _ { i }\tag{6}
$$

We ignore all patients with a single slide since pool\_size $= ~ 1$ does not present a selection task. The score is bounded in the range $[ 0 , 1 ]$ , with 1 being the optimal score, and final scores are averaged across the dataset for each k.

## 4 Results

## 4.1 All models achieve high recall

All models show similar performance on the HN dataset with CONCH (top recall of 97.89% at $k = 8 )$ performing marginally better than mean-CONCH and TITAN (top recall of $9 7 . 7 4 \%$ at $k = 9$ and $9 7 . 3 1 \%$ at $k = 8 ,$ , respectively) (Fig. 2B). Since the HN dataset is coarsely labeled with annotated sets containing on average $1 5 . 2 \pm 1 0 . 1$ block codes, it presents an easier evaluation task. This is reflected in the high recall at $k = 1 \left( \sim 9 2 \% \right)$ . Evaluation on the HN-node subset, which contains more fine-grained labels for involved lymph nodes with annotated sets containing on average 3.6±3.4 block codes, yielded a recall of 100% at $k = 1$ for all models. This performance on a more realistic evaluation task indicates that pathology FMs can accurately identify tumor content in slides. Visualizing patchlevel classification generated by CONCH quantitatively demonstrates accurate identification of slide morphology, as confirmed by a pathologist (Fig. 2C).

Performance on the OV dataset, which also presents a more realistic evaluation task with annotated sets containing an average of $1 . 9 \pm 1 . 1$ block codes, is worse at lower $k .$ However, at higher k, performance of all models matches the performance observed on the easier HN dataset. When evaluated against primary blocks alone, recall tops out at 85% at $k = 7$ for CONCH and TITAN and at $k = 9$ for mean-CONCH (Fig. 2D, left). When evaluated against primary and secondary blocks, we achieve recall of 100% at $k = 7 , 9$ , and 6 for CONCH, mean-CONCH, and TITAN, respectively (Fig. 2D, right). This result implies that including the top 9 slides ranked by any of these models guarantees the inclusion of at least one slide containing tumor. Repeating the evaluation with the dataset filtered to remove parts identified in the pathology report to be negative for tumor yielded minimal changes, with TITAN achieving a recall of 100% at $k = 5$ when evaluated against primary and secondary blocks, demonstrating that these models accurately identify slides without tumor.

## 4.2 CONCH and mean-CONCH show highest agreement

Agreement between CONCH and mean-CONCH is highest across both datasets, as is expected since they represent diferent aggregations of the same embeddings. Agreement between CONCH/mean-CONCH and TITAN is lower, implying that there is a mismatch between the top-k slides ranked by TITAN when compared to those ranked by CONCH and mean-CONCH, especially at lower k $\left( \mathrm { F i g . 3 A } \right)$

## 4.3 Ranked evaluation score highest for CONCH and mean-CONCH

Ranked evaluation shows each model’s performance across pool sizes (number of slides per patient), which were bucketed into quartiles to ensure roughly equalsized buckets. We also computed performance using the mean pool size and mean annotated set size per bucket at $k = 1$ and 10 to determine the expected performance for each dataset (Fig. 3B). The expected performance at $k = 1$ is lower for the HN dataset than for the OV dataset due to the larger annotated set sizes in the HN dataset. All models show similar performance on the HN dataset due to the coarsely labeled annotated set, which induces a higher likelihood that block codes from the annotated set appear at or near $k = 1$ . In the smallest pool-size bucket, all models perform better than the expected performance at $k = 1$ . On the OV dataset, CONCH and mean-CONCH outperform TITAN, with all models showing a drop in performance at the largest pool sizes. The hit rank, or k at which a block code from the annotated set is identified is, on average, 1.4 and 1.3 across the 3 smaller pool-size buckets for CONCH and mean-CONCH, respectively (Fig. 3C). However, the hit rank drops to 2.4 and 2.6 in the largest bucket for CONCH and mean-CONCH, respectively. Average hit rank for TITAN, on the other hand, drops consistently from 1.5 to 2, 3 and 3.1 as pool sizes increase (Fig. 3C).

![](images/e23d7393543418a63f0d968d2b58d194d5dcdb007ace7198e2c404d82fc7951f.jpg)  
Fig. 3: A: Agreement between rankings from CONCH and mean-CONCH is highest on both the HN and OV datasets. B: Ranked evaluation shows comparable performance across models on the HN dataset, while CONCH and mean-CONCH show better performance on the OV dataset. C: Hit rank is also similar across models on the HN dataset, while CONCH and mean-CONCH show better performance on the OV dataset, with the tumor slide identified within the top-2 ranked slides for patients with up to 43 slides.

## 5 Discussion

Automatic triage of digital pathology slides is critical not only for streamlining pathologist evaluation of cancer patient cases, but also for curating and labeling datasets for model training. In this paper, we show that publicly available FMs trained jointly on pathology image and text data can successfully triage slides without manual curation. Patch-level FMs like CONCH show better performance than slide-level FMs like TITAN, which may be because patch-level FMs classify slides at a more granular level and therefore may be better able to identify smaller tumors or tumors in slides where lymphocytes or other cell types are more dominant. Slide-level FMs, like TITAN, on the other hand, must aggregate these fine-grained patch embeddings into a single slide embedding, potentially suppressing some features present in the slide. This may also explain the disagreement between TITAN and CONCH/mean-CONCH, as well as the lower average hit rank achieved by TITAN for patients with larger sets of slides.

Future work includes evaluation of this pipeline using additional models and exploration into diferences in performance between the studied models. For instance, exploration into representations learned by diferent models using mechanistic interpretability methods could shed light on the diferences in performance. Further, evaluation may be performed on additional datasets to study generalizability across cancer types. Preliminary results on survival prediction using the top-1 slide ranked by CONCH show improvement over predictions using all slides per patient [17, 13], and evaluation using top-k, k > 1 slides is ongoing.

Acknowledgments. This study was funded by the Mayo Clinic Comprehensive Cancer Center, Susan Morrow Legacy Foundation, and Heidi Diercks Krause Fund in AI Innovation for Cancer. We would also like to thank Jake Matras and Dr. Derek Alexander for their feedback, and Suzanne Lee-Trejo for managing collaboration across teams.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Ahmed, F., Sellergren, A., Yang, L., Xu, S., Babenko, B., Ward, A., Olson, N., Mohtashamian, A., Matias, Y., Corrado, G.S., et al.: Pathalign: A vision-language model for whole slide images in histopathology. arXiv preprint arXiv:2406.19578 (2024)

2. Alber, M., Milbich, T., Carpen-Amarie, A., Tietz, S., Dippel, J., Muttenthaler, L., Cancer, B.P., Benetti, A., Korfiatis, P., Eulig, E., Lüscher, J., Wu, J., Hashimi, S.A., Dernbach, G., Schallenberg, S., Shah, N., Krügener, M., Jammoria, A., Matras, J., Dufy, P., Redlon, M., Jurmeister, P., Horst, D., Ruf, L., Müller, K.R., Klauschen, F., Norgan, A.: Atlas 2 – Foundation models for clinical deployment (Jan 2026). https://doi.org/10.48550/arXiv.2601.05148

3. Alber, M., Tietz, S., Dippel, J., Milbich, T., Lesort, T., Korfiatis, P., Krü- gener, M., Cancer, B.P., Shah, N., Möllers, A., Seegerer, P., Carpen-Amarie, A., Standvoss, K., Dernbach, G., Jong, E.d., Schallenberg, S., Kunft, A., Ankershofen, H.H.v., Schaeferle, G., Dufy, P., Redlon, M., Jurmeister, P., Horst, D., Ruf, L., Müller, K.R., Klauschen, F., Norgan, A.: Atlas: A Novel Pathology Foundation Model by Mayo Clinic, Charité, and Aignostics (Jan 2025). https://doi.org/10.48550/arXiv.2501.05409

4. Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Chen, B., Zhang, A., Shao, D., Song, A.H., Shaban, M., et al.: Towards a general-purpose foundation model for computational pathology. Nature Medicine (2024)

5. Dietterich, T.G., Lathrop, R.H., Lozano-Pérez, T.: Solving the multiple instance problem with axis-parallel rectangles. Artificial Intelligence 89(1), 31–71 (1997). https://doi.org/https://doi.org/10.1016/S0004-3702(96)00034-3

6. Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J., Jaume, G., Shaban, M., Kim, A., et al.: A multimodal whole-slide foundation model for pathology. Nature Medicine pp. 1–13 (2025)

7. Guo, Z., Zhang, Z., Ma, J., Wang, Y., Zhou, F., Xu, Y., Liang, L., Zhao, C., Xie, Q., Li, J., et al.: A clinically validated foundation model for comprehensive lung pathology interpretation. arXiv preprint arXiv:2605.25878 (2026)

8. Hanna, M.G., Reuter, V.E., Hameed, M.R., Tan, L.K., Chiang, S., Sigel, C., Hollmann, T., Giri, D., Samboy, J., Moradel, C., Rosado, A., Otilano, J.R., England, C., Corsale, L., Stamelos, E., Yagi, Y., Schüfler, P.J., Fuchs, T., Klimstra, D.S., Sirintrapun, S.J.: Whole slide imaging equivalency and eficiency study: experience at a large academic center. Modern Pathology 32(7), 916–928 (2019). https://doi.org/https://doi.org/10.1038/s41379-019-0205-0

9. Ilse, M., Tomczak, J., Welling, M.: Attention-based deep multiple instance learning. In: Dy, J., Krause, A. (eds.) Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 2127– 2136. PMLR (10–15 Jul 2018), https://proceedings.mlr.press/v80/ilse18a.html

10. Jaume, G., Doucet, P., Song, A.H., Lu, M.Y., Almagro-Perez, C., Wagner, S.J., Vaidya, A.J., Chen, R.J., Williamson, D.F.K., Kim, A., Mahmood, F.: HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis. arXiv (Jun 2024), https://arxiv.org/abs/2406.16192v1

11. Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., et al.: A visual-language foundation model for computational pathology. Nature Medicine 30, 863–874 (2024)

12. Shaikovski, G., Casson, A., Severson, K., Zimmermann, E., Wang, Y.K., Kunz, J.D., Retamero, J.A., Oakley, G., Klimstra, D., Kanan, C., et al.: Prism: A multimodal generative foundation model for slide-level histopathology. arXiv preprint arXiv:2405.10254 (2024)

13. Shao, Z., Chen, Y., Bian, H., Zhang, J., Liu, G., Zhang, Y.: Hvtsurv: Hierarchical vision transformer for patient-level survival prediction from whole slide image. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 2209– 2217 (2023)

14. Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., Yang, E., Mathieu, P., van Eck, A., Lee, D., Viret, J., Robert, E., Wang, Y.K., Kunz, J.D., Lee, M.C.H., Bernhard, J.H., Godrich, R.A., Oakley, G., Millar, E., Hanna, M., Wen, H., Retamero, J.A., Moye, W.A., Yousfi, R., Kanan, C., Klimstra, D.S., Rothrock, B., Liu, S., Fuchs, T.J.: A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine (2024)

15. Wang, X., Zhao, J., Marostica, E., Yuan, W., Jin, J., Zhang, J., Li, R., Tang, H., Wang, K., Li, Y., Wang, F., Peng, Y., Zhu, J., Zhang, J., Jackson, C.R., Zhang, J., Dillon, D., Lin, N.U., Sholl, L., Denize, T., Meredith, D., Ligon, K.L., Signoretti, S., Ogino, S., Golden, J.A., Nasrallah, M.P., Han, X., Yang, S., Yu, K.H.: A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634(8035), 970–978 (Oct 2024). https://doi.org/10.1038/s41586-024-07894-z

16. Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., González, J., Gu, Y., Xu, Y., Wei, M., Wang, W., Ma, S., Wei, F., Yang, J., Li, C., Gao, J., Rosemon, J., Bower, T., Lee, S., Weerasinghe, R., Wright, B.J., Robicsek, A., Piening, B., Bifulco, C., Wang, S., Poon, H.: A whole-slide foundation model for digital pathology from real-world data. Nature (2024)

17. Yao, J., Zhu, X., Jonnagaddala, J., Hawkins, N., Huang, J.: Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks. Medical Image Analysis 65, 101789 (2020)

18. Zhang, D., Gong, Z., Pang, X., Liu, J., Lu, J., Cui, H., Ge, J., Zeng, Z., Yi, K., Li, Y., et al.: Care: A molecular-guided foundation model with adaptive region modeling for whole slide image analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21078–21088 (2026)

## 6 Appendix

The appendix contains details about the prompts used for zero-shot classification.

Table A1: Prompts used for tumor class for the HN and OV datasets.

tumor   
HN "squamous cell carcinoma", "head and neck squamous cell carcinoma", "HNSCC", "head and neck SCC", "oropharyngeal squamous cell carcinoma", "laryngeal squamous cell carcinoma", "hypopharyngeal squamous cell carcinoma", "nasopharyngeal squamous cell carcinoma", "oral cavity squamous cell carcinoma", "HNSC"   
OV "high-grade serous ovarian cancer", "high-grade serous carcinoma", "HGSOC", "serous papillary carcinoma", "serous ovarian carcinoma"

Table A2: Prompts used for lymphocyte class for the HN and OV datasets.
<table><tr><td></td><td>lymphocyte</td></tr><tr><td>HN</td><td>&quot;lymphocytes&quot;, &quot;lymphoid aggregate&quot;, &quot;immune cells&quot;, &quot;lymphoid infiltrate&quot;, &quot;inflammatory cells&quot;, &quot;tumor-infiltrating lymphocytes&quot;, &quot;TILs&quot;,</td></tr><tr><td>OV</td><td>&quot;tumor-infiltrating immune cells&quot;</td></tr></table>

Table A3: Prompts used for other class for the HN and OV datasets.
<table><tr><td></td><td>other</td></tr><tr><td>HN</td><td>&quot;resection negative for carcinoma&quot;, &quot;resection negative for tumor&quot;, &quot;normal oropharyngeal mucosa&quot;, &quot;uninvolved oropharyngeal mucosa&quot;, &quot;normal oropharyngeal epithelium&quot;, &quot;benign epithelium&quot;</td></tr><tr><td>OV</td><td>&quot;resection negative for carcinoma&quot;, &quot;resection negative for tumor&quot;, &quot;normal ovarian tissue&quot;, &quot;uninvolved ovarian tissue&quot;, &quot;normal ovarian epithelium&quot;, &quot;benign epithelium&quot;</td></tr></table>

Table A4: Templates used to generate an ensemble of prompts. Prompts were generated by combining template text with class prompts by replacing ‘CLASS-NAME’ with the class prompts shown in Tables A1, A2, A3. Adapted from example in the CONCH repository.

"CLASSNAME.", "a photomicrograph showing CLASSNAME.", "a photomicrograph of CLASSNAME.", "an image of CLASSNAME.", "an image showing CLASSNAME.", "an example of CLASSNAME.", "CLASSNAME is   
shown.", "this is CLASSNAME.", "there is CLASSNAME.", "a histopathological   
image showing CLASSNAME.", "a histopathological image of CLASSNAME.", "a   
histopathological photograph of CLASSNAME.", "a histopathological photograph   
showing CLASSNAME.", "shows CLASSNAME.", "presence of CLASSNAME.",   
"CLASSNAME is present.", "an H&E stained image of CLASSNAME.", "an H&E   
stained image showing CLASSNAME.", "an H&E image showing CLASSNAME.",   
"an H&E image of CLASSNAME.", "CLASSNAME, H&E stain.", "CLASSNAME,