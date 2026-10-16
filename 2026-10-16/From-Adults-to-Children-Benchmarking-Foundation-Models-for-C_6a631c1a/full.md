# From Adults to Children: Benchmarking Foundation Models for Chest X-Ray Classification under Domain Shift

Sara Inacio´ <sup>1,3</sup>, Hugo Proenc¸a<sup>1,2</sup>, Joao C. Neves˜ <sup>1,3</sup>

<sup>1</sup>University of Beira Interior, Portugal <sup>2</sup>IT: Instituto de Telecomunicac¸oes ˜ <sup>3</sup>NOVA LINCS sara.inacio@ubi.pt, hugomcp@ubi.pt, jcneves@ubi.pt

## Abstract

Foundation models have shown significant potential across diverse fields and have been applied to medical imaging, but their robustness across patient populations remains unclear. In this work, we investigate foundation models’ abil ity to generalize from adult to pediatric chest X-ray classification. To the best of our knowledge, this is the first benchmark to compare multiplefoundation models under a common adult-to-pediatric evaluation setting while also analyzing the effect ofadaptation depth. We harmonize VinDr-CXR and PediCXR into a five-label classification task and evaluate sixfoundation models and two CNN baselines. All models are adapted using adult data only and tested in both the in-domain and pediatric domains using linear probing, partial fine-tuning, and full fine-tuning. The benchmark reveals a clear and consistent trend: all models suffer a substantialperformance degradation when transferredfrom adult to pediatric data, with drops exceeding 10 percentage points. While Macro-AUROC values reach approximately 92% on the adult test set, performance decreases to around 78% on the pediatric domain. Another important finding is that increasing the depth of adaptation does not significantly improve cross-population generalization. Moreover, strong in-domain performance does not translate into better performance on pediatric data.

## 1. Introduction

Medical imaging plays an important role in clinical diagnosis, and deep learning has shown considerable potential for automated medical image analysis. Chest radiography (CXR) is widely used in pediatric care due to its low cost, availability, and relatively low radiation exposure [18]. However, the development of deep learning systems for chest X-ray analysis has been largely supported by adult datasets. Previous studies have shown that models trained on adult chest radiographs can experience substantial performance degradation when evaluated on pediatric populations [5, 6, 36]. As illustrated in Fig. 1, this problem is closely related to domain shift [13], where the data distribution at test time differs from that observed during training. In medical imaging, such differences may arise from acquisition devices, imaging protocols, clinical institutions, annotation practices, or patient populations [8]. Age is a particularly relevant factor in population shifts in chest radiography, as adult and pediatric examinations differ in anatomical characteristics, disease prevalence, and the presentation of several radiographic findings. Foundation models [3] have recently emerged as a promising approach for learning transferable representations from largescale datasets. In medical imaging, these models are commonly pretrained using self-supervised, weakly supervised, or multimodal objectives and can then be adapted to downstream clinical tasks [28]. Several foundation models have also been developed specifically for medical imaging and chest radiography [1, 20, 35]. Their transfer capabilities make them particularly relevant for studying whether largescale pretraining can improve robustness across patient populations. Despite this progress, the behavior of foundation models under an adult-to-pediatric CXR domain shift remains poorly characterized. Existing work has mainly focused on individual models, specific tasks, or single adaptation settings. In particular, it remains unclear how foundation models with different pretraining domains and learning objectives compare when evaluated under the same adultto-pediatric protocol, and how the level of downstream adaptation affects their cross-domain performance. In this work, we study adult-to-pediatric generalization in chest X-ray classification by conducting a standardized comparison between conventional CNNs and foundation models. We evaluate two CNN baselines and six foundation models spanning different pretraining domains and learning objectives. All models are adapted exclusively using adult VinDr-CXR [16] data and evaluated both in-domain on adult radiographs and cross-domain on PediCXR [21], without using pediatric data during downstream adaptation. We evaluate each model under three adaptation strategies: linear probing, partial fine-tuning, and full fine-tuning. To the best of our knowledge, this is the first systematic benchmark of multiple foundation models under a common adult-topediatric CXR evaluation protocol. Our main contributions are:

![](images/4714b7d58acd1e62cfdb44dce66aa55e9665fbcdaed502ff28a0869e52d50766.jpg)  
Figure 1. Illustration of the domain shift between adult and pediatric chest X-rays. The model is trained on the adult source domain and evaluated on the pediatric target domain.

• We provide a standardized benchmark of conventional CNNs and foundation models under an adult-to-pediatric chest X-ray domain shift.

• We evaluate three downstream adaptation strategies: linear probing, partial fine-tuning, and full fine-tuning, and study how they affect in-domain and cross-domain performance.

• We analyze how model family, pretraining domain, and learning objective relate to the performance degradation observed between adult and pediatric populations.

The code will be made publicly available upon acceptance to facilitate reproducibility of the benchmark.

## 2. Related Work

## 2.1. Foundation Models for Medical Imaging

Foundation models have become increasingly adopted in medical imaging, offering an alternative to task-specific supervised representation learning. Early work in this area focused on vision-language pretraining, following the success of models such as CLIP [22] in learning transferable representations from large collections of image–text pairs. This idea translated to medical data, where images are often accompanied by free-text reports, and showed that radiology reports can provide useful semantic supervision and support transfer to downstream imaging tasks without relying on extensive manual annotation. More recent methods have explored stronger forms of domain specialization. MedCLIP [29] adapts contrastive vision-language learning to medical data while reducing the dependence on strictly paired image–text samples, whereas BioMedCLIP [35] expands the scale of biomedical pretraining using millions of image–text pairs drawn from the scientific literature. At the same time, image-only foundation models have shown that textual supervision is not a prerequisite for transferable visual representations. DINOv2 [17], despite being trained on general-domain images, achieves strong transfer across visual tasks through self-supervised learning. These models therefore provide a useful basis for studying how architecture, pretraining modality, and domain specificity affect transfer to medical imaging tasks.

## 2.2. Foundation Models for Chest X-rays

Chest radiography has become an important application area for foundation models because several large datasets provide radiographs together with their corresponding reports. This has made vision-language pretraining particularly attractive. BioViL [2], for instance, learns radiologyspecific representations from paired chest X-rays and reports, and BioViL-T [1] extends this idea to longitudinal studies by incorporating temporal information. CheXzero [27] and CXR-CLIP [34] also rely on image–text contrastive learning, with a focus on zero-shot and transferable recognition. ELIXR [31] takes a somewhat different direction, linking chest radiograph representations with large language models. Image-only pretraining has also received increasing attention in chest radiography. RAD-DINO [20] applies self-supervised DINO-style training to large chest X-ray collections, allowing visual representations to be learned without using the associated reports. More recent models such as RayDINO [14] and EVA-X [33] follow a similar direction, further exploring large-scale self-supervised learning for radiographic images.

## 2.3. Domain Shift in Medical Imaging

In many deep learning settings, it is implicitly assumed that the training and test data follow the same distribution, or at least a very similar one. Medical images are produced in complex clinical environments, and their appearance is affected by several factors [8]. Differences in scanners, acquisition protocols, institutions, patient populations, and data collection periods can lead to variations between datasets, even when they address the same clinical problem. As a result, a model that performs well on data from one domain may show a considerable drop in performance when evaluated on another domain. Previous studies have reported this effect across different sources of domain shift in medical imaging, including acquisition-related differences, changes between clinical sites, and variations in patient demographics [15]. Population shift is particularly relevant in this context, since factors such as age, disease prevalence, and other patient characteristics can affect both the appearance of medical images and the distribution of the corresponding findings. This makes cross-population evaluation important when assessing whether a model trained on one patient group can generalize to another.

Table 1. Label harmonization between VinDr-CXR [16] and PediCXR [21] for the five-label multi-label classification task. Labels without a corresponding category in the other dataset are excluded from the target space.
<table><tr><td>Target label</td><td>VinDr-CXR</td><td>PediCXR</td></tr><tr><td>No Finding</td><td>No finding</td><td>No finding</td></tr><tr><td>Other Diseases</td><td>Other diseases</td><td>Other diseases</td></tr><tr><td>Lung Tumor</td><td>Lung tumor</td><td>Lung tumor Pneumonia,</td></tr><tr><td>Pneumonia</td><td>Pneumonia</td><td>Broncho-pneumonia, Pleuro-pneumonia</td></tr><tr><td>Tuberculosis</td><td>Tuberculosis</td><td>Tuberculosis</td></tr><tr><td>Excluded</td><td>COPD</td><td>Remaining global labels</td></tr></table>

## 2.4. Adult-to-Pediatric Domain Shift

Age-related domain shift is a relevant issue in chest X-ray analysis, particularly given the limited availability of large pediatric datasets [18]. Rollan-Martinez-Herrera et al. [24] reported a clear drop in pneumonia classification performance when a CNN trained on adult data was evaluated in children, with AUC decreasing from 0.95 to 0.82. Tang et al. [26] explored unsupervised domain adaptation and improved pediatric AUC from 0.893, without adaptation, to 0.963 using unlabeled pediatric images. Similar transfer issues have also been studied in lung segmentation [10, 23], where performance was shown to vary with pediatric age and anatomy. Capellan-Martin et al. [4] further investigated this problem using self-supervised ViT pretraining for zero-shot pediatric tuberculosis detection. Previous studies have mainly considered task-specific CNNs, individual architectures, or single adaptation settings. Recent work has evaluated foundation models across adult and pediatric CXR populations [7, 30]. However, the effect of downstream adaptation depth has not been systematically compared across multiple foundation-model visual encoders under the same adult-to-pediatric transfer protocol. To the best of our knowledge, no systematic comparison of multiple foundation models with different pretraining domains and objectives under the same adult-to-pediatric CXR evaluation setting has been reported.

## 3. Adult-to-Pediatric CXR Benchmark

## 3.1. Problem Formulation

We consider multi-label chest X-ray classification under an adult-to-pediatric domain shift. Let $\mathcal { D } _ { S }$ and $\mathcal { D } _ { T }$ denote the adult source and pediatric target domains, respectively. After label harmonization, both domains share a common set of C clinical findings, with labels $y \in \{ 0 , 1 \} ^ { C }$ . Samples from the two domains follow distributions $P _ { S } ( X , Y )$ and

$P _ { T } ( X , Y )$ , where $P s ( X , Y ) \neq P _ { T } ( X , Y )$ . For each adaptation strategy, a model $f _ { \theta }$ is optimized using only the adult source training set, $\mathcal { D } _ { S } ^ { \mathrm { t r a i n } }$ , and evaluated on both the adult source test set, $\mathcal { D } _ { S } ^ { \mathrm { t e s t } }$ , and the pediatric target test set, $\mathcal { D } _ { T } ^ { \mathrm { t e s t } }$ No samples from the pediatric target dataset are used during downstream adaptation or model selection. Fig. 2 shows an overview of the evaluation protocol.

## 3.2. Datasets

We use two chest X-ray datasets to study the adult-topediatric domain shift: VinDr-CXR [16] for the adult population and PediCXR [21] for the pediatric population. PediCXR [21] contains 9,125 pediatric chest X-rays, with 7,728 images in the training set and 1,397 in the test set, annotated with 15 global diagnostic labels. Each image has a multi-label, image-level annotation produced by a single assigned radiologist. VinDr-CXR [16] contains 18,000 adult chest X-rays, divided into 15,000 training and 3,000 tes images, and provides six image-level labels: Lung Tumor, Pneumonia, Tuberculosis, Other Diseases, COPD, and No Finding. Before constructing the benchmark splits, we performed age-based population screening using the available DICOM metadata. Further details are provided in the Supplementary Material. Since the two datasets use different annotation vocabularies, we harmonize their labels into a common five-label target space consisting of No Finding, Other Diseases, Lung Tumor, Pneumonia, and Tuberculosis. In PediCXR [21], Pneumonia, Broncho-pneumonia, and Pleuro-pneumonia are merged into a single Pneumonia label. COPD is excluded from VinDr-CXR [16] because no corresponding label is available in PediCXR [21], while the remaining PediCXR[21] labels without a counterpart in VinDr-CXR [16] are not considered as target labels. Table 1 summarizes this mapping. For the VinDr-CXR [16] training set, each image is independently annotated by three radiologists. Their annotations are aggregated into a single multi-label target vector using an element-wise OR operation, implemented as the maximum across the three binary annotation vectors. Thus, a pathology is considered presen if it is reported by at least one radiologist, allowing multiple findings from different annotators to be retained simultaneously. To ensure label consistency, No Finding is set to zero whenever any pathological finding is present. In contrast, no annotator aggregation is required for PediCXR [21], since each image has a single annotation. This labe harmonization procedure is applied consistently to both the training and test splits of each dataset.

## 3.3. Models

We select a diverse set of models to investigate how different pretraining strategies and domains relate to robustness under the adult-to-pediatric chest X-ray domain shift. Our benchmark spans two main dimensions: the learning paradigm used during pretraining, ranging from supervised learning to image-only self-supervision and imagetext contrastive learning, and the domain of the pretrain ing data, ranging from natural images to biomedical and radiological data. We first consider DenseNet-121 [9] and EfficientNet-B0 [25] as conventional CNN baselines. Both models are initialized with weights obtained through su pervised pretraining on natural images and provide a ref erence for architectures that do not rely on large-scale foundation-model pretraining. We then include DINOv2 [17] and CLIP [22] as general-purpose foundation mod els pretrained on natural-image data. DINOv2 [17] relies on image-only self-supervised learning, whereas CLIP [22] uses image-text contrastive pretraining, representing two widely adopted foundation-model pretraining paradigms. To extend the comparison toward domain-specific pretrain ing, we include BiomedCLIP [35], which is pretrained on biomedical image-text pairs. Finally, we consider models whose pretraining is more closely aligned with radio logical data: RAD-DINO [20], based on image-only self supervised learning, and BioViL-T [1] and MedCLIP [29], which use image-text pretraining. For all models, we use only the visual encoder and attach the same downstream classification framework. For multimodal models, we do not use the text encoder during downstream training or in ference. This common setup ensures that the comparison focuses on the visual representations acquired during pretraining rather than on differences in downstream classification strategies. In this way, these models provide representative points across different pretraining paradigms and lev els of domain specificity. Rather than isolating either factor independently, our benchmark lets us assess whether models pretrained with different learning objectives and increasingly domain-relevant data show different levels of robustness when transferring from adult to pediatric chest X-rays.

![](images/f55ee383ec3e5860f0c2fd2505438a8ee7c4b190b737bd311818091bbc378cc3.jpg)  
Figure 2. Overview of the proposed domain shift evaluation methodology. Models are trained or adapted using chest X-rays from the seen adult source domain and subsequently evaluated on the unseen pediatric target domain, without target-domain training.

## 3.4. Adaptation Protocols

To evaluate how different levels of model adaptation affect performance under the adult-to-pediatric domain shift, we consider three adaptation protocols: linear probing (LP), partial fine-tuning (PFT), and full fine-tuning (FFT). These protocols progressively increase the number of trainable parameters, ranging from adapting only the classification head to updating the entire visual encoder, allowing us to assess how much modification of the pretrained representations is required for the downstream task [11, 12]. In linear probing, the visual encoder is kept frozen, and only the classification head is optimized. This setting evaluates the transferability of the pretrained visual representations without modifying the encoder. In partial fine-tuning, the final encoder block or stage, together with the classification head, is optimized, while the remaining encoder parameters are kept frozen. This provides an intermediate level of adaptation in which high-level representations can adjust to the downstream task while preserving most of the pretrained representation. PFT is defined structurally by unfreezing the final encoder block or stage and is therefore not parameter-matched across architectures. Finally, in full fine-tuning, the entire visual encoder and classification head are jointly optimized, providing the highest degree of adaptation to the training data. Table 2 reports the total number of parameters for each model, together with the number of trainable parameters under LP and PFT. Under FFT, all model parameters are trainable by definition.

## 4. Evaluation

## 4.1. In-Domain & Cross-Domain

We evaluate all models under two settings: in-domain and cross-domain evaluation. For each adaptation protocol, models are trained exclusively on the VinDr-CXR [16] adult training set. In-domain performance is measured on the held-out VinDr-CXR [16] test set, which follows the same source domain as the training data. Cross-domain performance is evaluated on PediCXR [21], without any additional training or adaptation using pediatric data. This evaluation protocol allows us to assess not only the predictive performance of each model on the source domain, but also how well it generalizes when transferred to a pediatric population. We compare the in-domain and cross-domain results to quantify the performance degradation associated with the adult-to-pediatric domain shift. The same evaluation procedure is applied to all CNN baselines and foundation models across the three adaptation strategies.

Table 2. Model configurations across adaptation strategies. Model architectures, total parameter counts, and trainable parameters under linear probing (LP) and partial fine-tuning (PFT). Under full fine-tuning, all model parameters are updated.
<table><tr><td>Model</td><td>Visual Backbone</td><td>Total params.</td><td>LP trainable</td><td>PFT trainable</td></tr><tr><td>DenseNet-121 [9]</td><td></td><td>6.95M</td><td>5.1K</td><td>2.17M</td></tr><tr><td>EfficientNet-B0 [25]</td><td></td><td>4.01M</td><td>6.4K</td><td>1.14M</td></tr><tr><td>CLIP [22]</td><td>ViT-B/16</td><td>86.20M</td><td>2.6K</td><td>7.49M</td></tr><tr><td>DINOv2 [17]</td><td>ViT-B/14</td><td>86.58M</td><td>3.8K</td><td>7.09M</td></tr><tr><td>BiomedCLIP [35]</td><td>ViT-B/16-224</td><td>86.19M</td><td>2.6K</td><td>7.49M</td></tr><tr><td>MedCLIP [29]</td><td>ViT-Swin-tiny</td><td>27.52M</td><td>3.8K</td><td>14.19M</td></tr><tr><td>BioViL-T [1]</td><td>ResNet-50</td><td>27.27M</td><td>2.6K</td><td>15.49M</td></tr><tr><td>RAD-DINO [20]</td><td>ViT-B/14</td><td>86.58M</td><td>3.8K</td><td>7.09M</td></tr></table>

## 4.2. Metrics

We evaluate performance using Macro-AUROC, mean Average Precision (mAP), and Macro-F1, computed over the five target labels. Macro-AUROC and mAP are obtained by independently computing AUROC and AP for each label and taking their unweighted mean. Macro-F1 is computed analogously after binarizing the predictions using a fixed threshold of 0.5. These metrics provide complementary threshold-independent and threshold-dependent evaluations of the multi-label classification performance. Macro-AUROC is used as the primary metric for the domain-shift analysis, while mAP and Macro-F1 are reported as complementary metrics. To quantify the degradation under the adult-to-pediatric domain shift, we additionally report

$$
\Delta _ { \mathrm { A U R O C } } = \mathrm { A U R O C } _ { A  A } - \mathrm { A U R O C } _ { A  P } ,\tag{1}
$$

where $A \  \ A$ denotes in-domain adult evaluation and $A \  \ P$ denotes evaluation on the pediatric domain after training on adult data. Lower $\Delta _ { \mathrm { A U R O C } }$ indicates a smaller performance degradation under the domain shift. We distinguish cross-domain performance from robustness to the shift. Cross-domain performance is measured directly by the pediatric AUROC, whereas robustness reflects how much performance is retained relative to the adult indomain setting and is summarized by $\Delta _ { \mathrm { A U R O C } }$ . We therefore consider both quantities jointly, since a small domainshift gap alone does not necessarily imply strong performance on the pediatric target domain.

## 4.3. Implementation Details

All experiments were implemented in PyTorch [19] and conducted on an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPU. All input images were resized to $2 2 4 \times 2 2 4$ pixels, while retaining the model-specific normalization used by each pretrained encoder. No additional data augmentation was applied during training. Models were trained under the three adaptation protocols described above: linear probing, partial fine-tuning, and full fine-tuning. Following [32], the classification head was optimized with a learning rate of $1 0 ^ { - 3 }$ in all experiments. For protocols in which the encoder was updated, a learning rate of $1 0 ^ { - 4 }$ was used for CNN-based backbones and $1 0 ^ { - 5 }$ for Vision Transformer-based backbones. AdamW was used as the optimizer with a weight decay of $1 0 ^ { - 4 }$ . A learningrate scheduler was applied to all trainable parameter groups, multiplying the learning rate by 0.7 every 15 epochs. After age-based screening, a fixed 80/20 split resulted in 11,894 training and 2,974 validation images. Supplementary Tables report detailed split statistics. This split was generated once and kept fixed across all models and adaptation protocols to ensure a consistent comparison. The official VinDr-CXR test set was kept separate and used only for in-domain evaluation. All models were trained for 100 epochs with a batch size of 32. Model selection was performed on the source-domain validation set, with the checkpoint achieving the highest macro-averaged AUROC retained for evaluation. Binary cross-entropy with logits was used as the training objective. To account for class imbalance, class-specific positive weights were computed from the adult training set as the ratio of negative to positive samples for each class and incorporated into the loss. Each experiment was repeated using three different random seeds, and results are reported across the three runs.

Baselines. We include two conventional CNN architectures as baselines to provide a reference for comparison with the evaluated foundation models. DenseNet-121 [9] uses dense connections between layers, which encourage feature reuse and improve information and gradient propagation throughout the network. EfficientNet [25] follows a compound scaling strategy that jointly balances network depth, width, and input resolution, providing a computationally efficient convolutional architecture. These models allow us to assess whether foundation models provide an advantage over established CNN-based approaches under the same training and evaluation settings.

## 5. Results

We structure our analysis around four research questions. RQ1 asks which models transfer most effectively from adult to pediatric data. RQ2 examines how pretraining domain and supervision paradigm relate to robustness under this shift. RQ3 evaluates how linear probing, partial finetuning, and full fine-tuning affect pediatric performance and cross-domain robustness. Finally, RQ4 assesses whether in-domain performance reliably reflects cross-domain performance and robustness.

## 5.1. Foundation Models vs. CNNs

RQ1 compares the transfer of ImageNet-pretrained CNNs and foundation models from adult to pediatric domains. As shown in Table 3, all models degrade under the domain shift, but the magnitude of the drop varies considerably. Under linear probing, the strongest foundation models clearly outperform DenseNet-121 [9] and EfficientNet-B0 [25] on the pediatric set and exhibit smaller domain-shift gaps. BiomedCLIP [35], MedCLIP [29], and RAD-DINO [20] achieve pediatric AUROCs approximately 75% or higher, compared with 56.11% and 46.84% for DenseNet-121 [9] and EfficientNet-B0 [25]. This advantage is not univer sal: DINOv2 [17] shows a similar gap to the CNN baselines, while BioViL-T [1] suffers the largest drop under linear probing. The advantage changes with stronger adaptation. In particular, DenseNet-121[9] becomes competitive with several foundation models under full fine-tuning, whereas EfficientNet-B0 [25] remains less robust. Overall, the strongest foundation models transfer better than the ImageNet-pretrained CNNs, especially under limited adaptation, although foundation-model pretraining alone does not guarantee robustness.

## 5.2. Influence of Pretraining on Domain-Shift

RQ2 compares general-purpose foundation models (CLIP [22] and DINOv2 [17]), medical vision-language models (BiomedCLIP [35], MedCLIP [29], and BioViL-T [1]), and the CXR-specific image-only RAD-DINO [20]. Clear differences appear across these pretraining settings. RAD-DINO [20] shows the most consistent transfer, maintaining a pediatric AUROC of approximately 80% across all three adaptation protocols. MedCLIP [29] and BiomedCLIP [35] also perform strongly with linear probing, but their pediatric performance decreases as more parameters are finetuned. In contrast, BioViL-T [1] benefits substantially from stronger adaptation, improving from 39.23% with linear probing to 68.16% with full fine-tuning. General-purpose models are similarly sensitive to the adaptation strategy, with DINOv2 [17] performing best under partial fine-tuning and CLIP [22] under full fine-tuning. Among the evaluated models, the CXR-specific image-only RAD-DINO [20] shows the most stable behavior across adaptation protocols, while general-purpose and medical vision-language models exhibit greater variability.

![](images/428d232712d39ad7d5cb6367095ec9a488cd2cd7f96ee87f9084936573c707ca.jpg)  
Linear Probing Partial Fine-TuningΔ Full Fine-Tuning  
Figure 3. Pediatric AUROC vs. Domain-shift gap. Colors denote models; circles, squares, and triangles denote linear probing, partial fine-tuning, and full fine-tuning, respectively. Lines connect the three protocols for each model. Lower gaps and higher AUROC are better.

## 5.3. Adaptation on Domain-shift Robustness

RQ3 examines how linear probing, partial fine-tuning, and full fine-tuning affect pediatric performance and robustness to the adult-to-pediatric shift (Fig. 3). On average, pediatric AUROC increases from 60.72% with linear probing to 65.84% with partial fine-tuning and 69.00% with full finetuning. The corresponding domain-shift gap decreases from 21.65 to 18.39 and 17.21, indicating that most of the reduction is achieved with partial fine-tuning. However, this varies considerably across models. DenseNet-121 [9] and BioViL-T [1] benefit from stronger adaptation, whereas DI-NOv2 performs best with partial fine-tuning. RAD-DINO [20] remains almost unchanged across the three protocols, maintaining a pediatric AUROC close to 80%. In contrast, MedCLIP [29] and BiomedCLIP [35] achieve their best pediatric performance with linear probing and degrade with further fine-tuning. No single adaptation strategy is optimal for all models: additional fine-tuning can improve transfer, have little effect, or reduce cross-domain performance depending on the pretrained representation.

## 5.4. In-Domain vs. Cross-Domain Robustness

RQ4 examines whether in-domain performance reliably indicates robustness under the adult-to-pediatric shift. As shown in Fig. 4, all configurations lie below the identity line, confirming a systematic drop in pediatric performance. Although stronger in-domain models generally tend to perform better on pediatric data, the size of the domain-shift gap varies substantially. BioViL-T [1] illustrates this clearly under linear probing, dropping from 80.66% in-domain AUROC to 39.23% on the pediatric set. In contrast, RAD-DINO [20] and BiomedCLIP [35] maintain much smaller

Full Fine-Tuning

Table 3. Performance under the adult-to-pediatric domain shift (A → P) and in-domain adult evaluation (A → A) across the three adaptation protocols. Performance metrics (%) are reported as mean ± standard deviation over three random seeds. ∆<sub>AUROC</sub> is computed from the corresponding unrounded mean AUROC values and is defined as $\Delta _ { \mathrm { A U R O C } } = \mathrm { A U R O C } _ { A  A } - \mathrm { A U R O C } _ { A  P } ,$ , where lower values indicate a smaller performance degradation under the domain shift. Best and second-best mean values within each adaptation strategy are shown in bold and underlined, respectively.

$$
\Delta _ { \mathrm { A U R O C } } \downarrow
$$

$$
\overline { { 5 6 . 1 1 \pm 0 . 7 7 } }
$$

$$
4 6 . 8 4 \pm 2 . 1 7
$$

$$
\overline { { 1 7 . 6 3 \pm 0 . 1 1 } }
$$

$$
1 8 . 4 2 \pm 0 . 2 1
$$

$$
1 5 . 3 8 \pm 0 . 5 0
$$

$$
7 7 . 6 0 \pm 0 . 0 9
$$

$$
2 0 . 4 7 \pm 0 . 4 7
$$

$$
5 2 . 9 8 \pm 0 . 1 9
$$

$$
\overline { { 4 1 . 5 8 \pm 0 . 0 4 } }
$$

$$
1 8 . 2 9 \pm 0 . 0 1
$$

$$
7 4 . 4 0 \pm 0 . 5 1
$$

$$
\overline { { 3 8 . 2 4 \pm 1 . 0 2 } }
$$

$$
2 1 . 0 7 \pm 0 . 2 6
$$

$$
3 6 . 8 1 \pm 0 . 3 4
$$

$$
5 7 . 1 5 \pm 0 . 1 1
$$

$$
2 0 . 2 7 \pm 0 . 0 7
$$

$$
7 3 . 2 3 \pm 0 . 0 1
$$

$$
3 7 . 9 1 \pm 0 . 2 6
$$

$$
2 2 . 0 9 \pm 0 . 7 4
$$

$$
3 7 . 4 3 \pm 0 . 0 4
$$

$$
8 3 . 3 9 \pm 0 . 0 2
$$

$$
3 8 . 9 4 \pm 0 . 1 6
$$

$$
2 0 . 9 6 \pm 0 . 9 2
$$

$$
7 4 . 9 5 \pm 0 . 4 1
$$

$$
2 1 . 2 4 \pm 1 . 7 1
$$

$$
4 7 . 3 1 \pm 0 . 0 6
$$

$$
8 5 . 4 3 \pm 0 . 1 5
$$

$$
4 5 . 2 1 \pm 0 . 5 9
$$

$$
5 2 . 6 9 \pm 0 . 6 8
$$

$$
\mathbf { 7 9 . 3 1 \pm 0 . 2 6 }
$$

$$
{ \bf 3 5 . 3 1 \pm 0 . 5 6 }
$$

$$
\pm 5 . 7 5 \pm 0 . 1 6
$$

$$
9 1 . 9 9 \pm 0 . 0 5
$$

$$
4 7 . 5 6 \pm 3 . 4 3
$$

$$
6 4 . 0 0 \pm 0 . 2 5
$$

$$
3 9 . 2 3 \pm 0 . 1 0
$$

$$
{ \bf 5 9 . 0 2 \pm 0 . 5 2 }
$$

$$
1 7 . 3 9 \pm 0 . 0 1
$$

$$
1 3 . 6 7 \pm 0 . 0 4
$$

$$
8 0 . 6 6 \pm 0 . 0 1
$$

$$
4 2 . 9 8 \pm 0 . 0 0
$$

$$
\underline { { 7 9 . 1 9 \pm 0 . 3 1 } }
$$

$$
4 1 . 9 8 \pm 0 . 0 3
$$

$$
2 4 . 5 9 \pm 0 . 2 2
$$

$$
\mathbf { 9 2 . 2 6 \ : \pm 0 . 0 6 }
$$

$$
2 3 . 9 8 \pm 0 . 1 7
$$

$$
{ \bf 6 4 . 0 9 \pm 0 . 1 7 }
$$

$$
5 6 . 5 0 \pm 0 . 4 8
$$

$$
\overline { { 6 3 . 0 5 \pm 4 . 8 4 } }
$$

$$
\overline { { 1 9 . 4 7 \pm 0 . 4 2 } }
$$

$$
\overline { { 1 6 . 1 8 \pm 0 . 2 1 } }
$$

$$
5 0 . 3 6 \pm 1 0 . 5 2
$$

$$
\overline { { 8 1 . 4 1 \pm 1 . 6 9 } }
$$

$$
1 9 . 1 0 \pm 0 . 3 8
$$

$$
\overline { { 4 8 . 4 2 \pm 2 . 7 2 } }
$$

$$
2 0 . 4 1 \pm 1 . 0 7
$$

$$
5 0 . 6 7 \pm 4 . 0 4
$$

$$
\overline { { 4 4 . 0 1 \pm 1 . 3 8 } }
$$

$$
7 6 . 3 8 \pm 1 . 0 7
$$

$$
1 9 . 1 7 \pm 0 . 2 3
$$

$$
3 9 . 7 3 \pm 1 . 0 9
$$

$$
1 9 . 7 8 \pm 1 . 4 5
$$

$$
7 3 . 4 4 \pm 1 . 5 8
$$

$$
3 8 . 1 0 \pm 0 . 8 9
$$

$$
7 2 . 5 1 \pm 1 . 4 7
$$

$$
4 0 . 1 7 \pm 0 . 1 4
$$

$$
2 3 . 1 5 \pm 0 . 2 9
$$

$$
3 9 . 4 6 \pm 1 . 0 7
$$

$$
2 4 . 3 1 \pm 0 . 7 2
$$

$$
8 7 . 2 5 \pm 0 . 0 9
$$

$$
5 7 . 3 5 \pm 0 . 4 3
$$

$$
7 4 . 3 2 \pm 1 . 0 0
$$

$$
5 3 . 8 4 \pm 1 . 4 7
$$

$$
\mathbf { 3 1 . 3 7 \pm 0 . 3 0 }
$$

$$
{ \underline { { 7 6 . 8 1 \pm 1 . 0 2 } } }
$$

$$
2 4 . 2 3 \pm 0 . 7 7
$$

$$
2 6 . 6 5 \pm 0 . 8 9
$$

$$
8 6 . 2 9 \pm 0 . 4 4
$$

$$
{ \bf 2 6 . 8 3 \pm 0 . 7 3 }
$$

$$
5 5 . 3 1 \pm 0 . 4 3
$$

$$
5 1 . 3 0 \pm 2 . 4 5
$$

$$
5 8 . 3 4 \pm 1 . 0 0
$$

$$
2 0 . 9 8 \pm 2 . 3 0
$$

$$
\underline { { 9 1 . 7 9 \pm 0 . 1 2 } }
$$

$$
1 9 . 6 0 \pm 1 . 4 8
$$

$$
{ \bf 6 6 . 6 1 \pm 0 . 6 6 }
$$

$$
{ \bf 6 4 . 0 3 \pm 0 . 7 1 }
$$

$$
\mathbf { 7 9 . 7 6 \pm 0 . 0 8 }
$$

$$
2 6 . 3 7 \pm 0 . 4 5
$$

$$
8 5 . 8 0 \pm 0 . 5 0
$$

$$
2 5 . 7 2 \pm 0 . 2 4
$$

$$
5 2 . 7 3 \pm 3 . 0 7
$$

$$
\mathbf { 9 2 . 4 2 \ : \pm 0 . 0 5 }
$$

$$
5 4 . 6 3 \pm 1 . 4 1
$$

$$
6 6 . 1 1 \pm 0 . 2 7
$$

$$
\underline { { 5 9 . 3 9 \pm 1 . 1 1 } }
$$

$$
\overline { { 6 6 . 3 5 \pm 7 . 9 2 } }
$$

$$
2 0 . 8 9 \pm 0 . 5 6
$$

$$
2 1 . 1 6 \pm 0 . 9 3
$$

$$
5 0 . 8 2 \pm 6 . 1 7
$$

$$
1 8 . 3 9 \pm 0 . 3 3
$$

$$
\overline { { 8 8 . 3 5 \pm 0 . 5 1 } }
$$

$$
\overline { { 5 4 . 7 8 \pm 4 . 0 8 } }
$$

$$
\overline { { 5 8 . 7 0 \pm 0 . 9 5 } }
$$

$$
1 6 . 3 2 \pm 2 . 5 6
$$

$$
7 5 . 4 4 \pm 2 . 2 1
$$

$$
2 0 . 2 7 \pm 0 . 7 9
$$

$$
6 8 . 5 2 \pm 1 . 0 2 
$$

$$
4 0 . 8 1 \pm 4 . 4 5
$$

$$
3 5 . 1 8 \pm 4 . 7 8
$$

$$
2 1 . 2 6 \pm 1 . 0 9
$$

$$
8 1 . 7 0 \pm 1 . 7 4
$$

$$
2 3 . 1 7 \pm 2 . 3 4
$$

$$
7 1 . 1 2 \pm 1 . 0 7
$$

$$
5 0 . 1 6 \pm 4 . 0 8
$$

$$
4 9 . 6 6 \pm 3 . 8 8
$$

$$
2 1 . 3 1 \pm 1 . 9 0
$$

$$
8 4 . 7 1 \pm 2 . 1 1
$$

$$
2 1 . 9 1 \pm 0 . 6 3
$$

$$
7 1 . 7 5 \pm 1 . 9 7
$$

$$
5 6 . 9 1 \pm 3 . 0 0
$$

$$
2 1 . 7 9 \pm 0 . 5 6
$$

$$
5 5 . 0 5 \pm 1 . 8 1
$$

$$
8 6 . 0 4 \pm 1 . 2 8
$$

$$
2 5 . 7 8 \pm 0 . 9 2
$$

$$
{ underline { { 7 6 . 5 4 \pm 1 . 3 3 } } }
$$

$$
5 7 . 4 1 \pm 1 . 2 8
$$

$$
5 5 . 8 2 \pm 2 . 0 5
$$

$$
2 4 . 5 6 \pm 0 . 6 3
$$

$$
9 2 . 3 9 \pm 0 . 4 0
$$

$$
2 1 . 0 0 \pm 0 . 1 8
$$

$$
6 8 . 1 6 \pm 4 . 9 4
$$

$$
\mathbf { 7 0 . 3 1 \pm 0 . 4 4 }
$$

$$
2 2 . 9 8 \pm 1 . 1 1
$$

$$
8 7 . 9 0 \pm 0 . 1 8
$$

$$
\mathbf { 3 0 . 9 7 \ : \pm 5 . 0 0 }
$$

$$
{ \bf 6 5 . 0 1 } \pm 3 . 2 5
$$

$$
\mathbf { 7 8 . 7 7 \pm 0 . 8 3 }
$$

$$
2 4 . 5 3 \pm 0 . 6 2
$$

$$
6 0 . 9 4 \pm 0 . 8 2
$$

$$
\mathbf { 9 3 . 1 5 \ : \pm 0 . 4 1 }
$$

$$
6 8 . 6 2 \pm 0 . 2 8 
$$

$$
6 0 . 1 7 \pm 0 . 8 1
$$

$$
6 4 . 9 2 \pm 2 . 4 0
$$

![](images/4503f465fed59a15ff949b6148ba867a866f7eb4e19c9e7eef01b47c9a7c8811.jpg)

![](images/76f6abc1ba2f254680c09bff009655abe2f26f1ce5488ff17f9b330a775f7db8.jpg)

![](images/df31d792dc6459c695e7ccb320140c887c28f1200c7042ad386ccb4bc2a18c48.jpg)  
Figure 4. Adult in-domain versus pediatric AUROC across adaptation strategies. The dashed line indicates equal in-domain and cross-domain performance, and the vertical dotted lines show the corresponding domain-shift gap.

gaps while also achieving strong pediatric performance. These results show that in-domain performance alone is not sufficient to assess robustness. Models with similar adult performance can behave very differently after transfer to the pediatric domain.

## 6. Discussion

Our results show that foundation models are not uniformly robust to the adult-to-pediatric shift. RAD-DINO [20], MedCLIP [29], and BiomedCLIP [35] preserve perfor mance relatively well under limited adaptation, whereas BioViL-T [1] exhibits the largest domain-shift gap under linear probing. This is notable because BioViL-T [1] is also pretrained on medical image-text data. As shown in Table 2, it is the only foundation model in our comparison that does not use a ViT-based visual encoder and instead relies on a ResNet-50 backbone. This architectural difference, together with differences in the pretraining objective and data, may partly contribute to its distinct behavior. How ever, the present experiments do not isolate these factors, so medical-domain pretraining alone cannot be considered suf ficient for robustness across adult and pediatric populations. RAD-DINO [20] stands out for its stability across adapta tion strategies, with pediatric AUROC remaining close to 0.80% under linear probing, partial fine-tuning, and full fine-tuning. RAD-DINO and DINOv2 [17] provide a particularly informative comparison, as they share the same ViT-B/14 backbone, model size, and adaptation parameter counts. Despite these similarities, DINOv2 shows a sub stantially larger domain-shift gap under linear probing and benefits more from downstream adaptation. Their different linear-probing behavior is consistent with a benefit from CXR-specific pretraining, although differences in pretrain ing data and learning objective prevent a causal attribution. The adaptation experiments also show that more fine-tuning is not always better. Partial fine-tuning captures most of the average reduction in the domain-shift gap observed with full fine-tuning, while the effect of updating additional parameters varies substantially across models. DenseNet-121 [9] and BioViL-T [1] benefit from stronger adaptation, whereas RAD-DINO [20] changes very little across protocols. Med CLIP [29] and BiomedCLIP [35], by contrast, achieve their best pediatric performance with linear probing and degrade as more parameters are updated. The choice of adapta tion strategy therefore appears to depend strongly on the pretrained representation. Finally, strong in-domain perfor mance does not necessarily imply robustness to the adult-to pediatric shift. Adult and pediatric performance follow the same general trend, but similar in-domain scores do not al ways translate into similar behavior after the domain shift. BioViL-T [1] is a clear example: despite performing well on the adult test set, its AUROC drops substantially on the pediatric data. This gap shows that in-domain performance alone gives an incomplete view of how well a model transfers across populations. Evaluating models on shifted populations is therefore important when assessing their robustness and transferability.

## 7. Limitations

Although some comparisons are relatively well controlled, most notably RAD-DINO [20] and DINOv2 [17], which share the same backbone, model size, and adaptation budget, the full set of models differs in architecture, pretraining data, scale, supervision, and learning objective. Moreover, PFT is defined at the block/stage level and is therefore not parameter-matched across architectures. The observed differences should thus be interpreted as associations rather than as evidence that any single factor, such as image-only supervision or CXR-specific pretraining, is responsible for improved robustness. In addition, our evaluation considers a single adult-to-pediatric cross-dataset domain shift. Because the adult and pediatric populations are drawn primarily from different datasets, age cannot be isolated from dataset-specific differences such as image acquisition. The same trends may therefore not hold across other hospitals, acquisition devices, pediatric populations, or types of distribution shift. Further evaluation across additional datasets and clinical settings is needed to determine how these findings generalize.

## 8. Conclusion

This is the first work to compare several foundation models under the same adult-to-pediatric setting while also varying the level of model adaptation. We benchmark six representative foundation models and two conventional CNNs for chest X-ray classification, systematically evaluating linear probing, partial fine-tuning, and full fine-tuning across models with different pretraining domains and supervision paradigms. Across this benchmark, all models are affected by the shift from adult to pediatric data, but the extent of the degradation varies substantially across pretrained representations. RAD-DINO [20], MedCLIP [29], and BiomedCLIP [35] retain comparatively strong pediatric performance with limited adaptation, whereas other models are much more sensitive to the shift. More adaptation is not necessarily better: depending on the model, fine-tuning more of the encoder can improve transfer, have little effect, or even reduce pediatric performance. We also observe that models performing well on the adult test set are not always the ones that transfer best to pediatric data. These results suggest that cross-population robustness depends on both the pretrained representation and how it is adapted to the downstream task. This reinforces the need to evaluate foundation models directly across patient populations, rather than using in-domain performance as a proxy for cross-domain robustness.

## References

[1] Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Perez-Garcia, Maximilian Ilse, Daniel C Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, et al. Learning to exploit temporal structure for biomedical vision-language processing. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15016–15027. IEEE, 2023. 1, 2, 4, 5, 6, 7, 8

[2] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Stephanie Hyland, Maria Wetscherek, Tristan Naumann, Aditya Nori, Javier Alvarez-Valle, et al. Making the most of text semantics to improve biomedical vision–language processing. In European Conference on Computer Vision, pages 1–21. Springer, 2022. 2

[3] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 1

[4] Daniel Capellan-Mart´ ´ın, Abhijeet Parida, Juan J Gomez-´ Valverde, Ramon Sanchez-Jacob, Pooneh Roshanitabrizi, Marius G Linguraru, Mar´ıa J Ledesma-Carbayo, and Syed M Anwar. Zero-shot pediatric tuberculosis detection in chest xrays using self-supervised learning. In 2024 IEEE International Symposium on Biomedical Imaging (ISBI), pages 1–5. IEEE, 2024. 3

[5] Devina Chatterjee, Adway Kanhere, Florence X Doo, Jerry Zhao, Andrew Chan, Alexander Welsh, Pranav Kulkarni, Annie Trang, Vishwa S Parekh, and Paul H Yi. Children are not small adults: addressing limited generalizability of an adult deep learning ct organ segmentation model to the pediatric population. Journal of Imaging Informatics in Medicine, 38(3):1628–1641, 2025. 1

[6] Sheng Cheng, Zbigniew A Starosolski, and Devika Subramanian. Self-supervised learning for building robust pediatric chest x-ray classification models. arXiv preprint arXiv:2409.00231, 2024. 1

[7] Louisa Fay, Jean-Benoit Delbrouck, Thomas Kustner, Bin¨ Yang, Noel C Codella, Matthew P Lungren, Curtis Langlotz, and Sergios Gatidis. Beyond the prompt: deploying medical foundation models on diverse chest x-ray populations. In Medical Imaging with Deep Learning, 2025. 3

[8] Hao Guan and Mingxia Liu. Domain adaptation for medical image analysis: a survey. IEEE Transactions on Biomedical Engineering, 69(3):1173–1185, 2021. 1, 2

[9] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2261–2269. Ieee, 2017. 4, 5, 6, 7, 8

[10] Tristan Kirscher, Sylvain Faisan, Xavier Coubez, Loris Barrier, and Philippe Meyer. Psat: Pediatric segmentation approaches via adult augmentations and transfer learning. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 474–483. Springer, 2025. 3

[11] Simon Kornblith, Jonathon Shlens, and Quoc V. Le. Do better imagenet models transfer better? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 4

[12] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution, 2022. 28. 4

[13] Jose G Moreno-Torres, Troy Raeder, Roc´ıo Alaiz-Rodr´ıguez, Nitesh V Chawla, and Francisco Herrera. A unifying view on dataset shift in classification. Pattern Recognition, 45(1):521–530, 2012. 1

[14] Theo Moutakanni, Piotr Bojanowski, Guillaume Chas-´ sagnon, Celine Hudelot, Armand Joulin, Yann LeCun,´ Matthew Muckley, Maxime Oquab, Marie-Pierre Revel, and Maria Vakalopoulou. Advancing human-centric ai for robust x-ray analysis through holistic self-supervised learning. arXiv preprint arXiv:2405.01469, 2024. 2

[15] Aminu Musa, Mariya Ibrahim Adamu, Habeebah Adamu Kakudi, Monica Hernandez, and Yusuf Lawal. Analyzing cross-population domain shift in chest x-ray image classifi cation and mitigating the gap with deep supervised domain adaptation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 585– 595. Springer, 2024. 2

[16] Ha Q Nguyen, Khanh Lam, Linh T Le, Hieu H Pham, Dat Q Tran, Dung B Nguyen, Dung D Le, Chi M Pham, Hang TT Tong, Diep H Dinh, et al. Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations. Scientific Data, 9(1):429, 2022. 2, 3, 4

[17] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 4, 5, 6, 7, 8

[18] Sirwa Padash, Mohammad Reza Mohebbian, Scott J Adams, Robert DE Henderson, and Paul Babyn. Pediatric chest radiograph interpretation: how far has artificial intelligence come? a systematic literature review. Pediatric Radiology, 52(8):1568–1580, 2022. 1, 3

[19] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im perative style, high-performance deep learning library. Ad vances in neural information processing systems, 32, 2019. 5

[20] Fernando Perez-Garc´ ´ıa, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al. Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence, 7(1): 119–130, 2025. 1, 2, 4, 5, 6, 7, 8

[21] Hieu H Pham, Ngoc H Nguyen, Thanh T Tran, Tuan NM Nguyen, and Ha Q Nguyen. Pedicxr: an open, large-scale chest radiograph dataset for interpretation of common thoracic diseases in children. Scientific data, 10(1):240, 2023. 2, 3, 5

[22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2, 4, 5, 6, 7

[23] Sivaramakrishnan Rajaraman, Feng Yang, Ghada Zamzmi, Zhiyun Xue, and Sameer Antani. Can deep adult lung segmentation models generalize to the pediatric population? Ex pert systems with applications, 229:120531, 2023. 3

[24] Maria Rollan-Martinez-Herrera, Alejandro A D´ıaz, Ruben´ San Jose Est´ epar, Gonzalo Vegas Sanchez-Ferrero, James C´ Ross, Raul San Jos ´ e Est ´ epar, and Pietro Nardelli. Cnns´ trained with adult data are useful in pediatrics. a pneumonia classification example. Plos one, 19(7):e0306703, 2024. 3

[25] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114. PmLR, 2019. 4, 5, 6, 7

[26] Yuxing Tang, Youbao Tang, Veit Sandfort, Jing Xiao, and Ronald M Summers. Tuna-net: Task-oriented unsupervised adversarial network for disease recognition in cross-domain chest x-rays. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 431– 440. Springer, 2019. 3

[27] Ekin Tiu, Ellie Talius, Pujan Patel, Curtis P Langlotz, Andrew Y Ng, and Pranav Rajpurkar. Expert-level detection of pathologies from unannotated chest x-ray images via selfsupervised learning. Nature Biomedical Engineering, 6(12): 1399–1406, 2022. 2

[28] Vivien van Veldhuizen, Vanessa Botha, Chunyao Lu, Melis Erdal Cesur, Kevin Groot Lipman, Edwin D de Jong, Hugo Horlings, Clarisa I Sanchez, Cees GM Snoek,´ Lodewyk Wessels, et al. Foundation models in medical imaging: A review and outlook. arXiv preprint arXiv:2506.09095, 2025. 1

[29] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. In Proceedings of Conference on Empirical Methods in Natural Language Processing, pages 3876–3887, 2022. 2, 4, 5, 6, 7, 8

[30] Qinmei Xu, Yiheng Li, Xianghao Zhan, Ahmet Gorkem Er, Brittany Dashevsky, Chuanjun Xu, Mohammed Alawad, Mengya Yang, Liu Ya, Changsheng Zhou, Xiao Li, Haruka Itakura, and Olivier Gevaert. Benchmarking chest x-ray diagnosis models across multinational datasets. arXiv preprint arXiv:2505.16027, 2025. 3

[31] Shawn Xu, Lin Yang, Christopher Kelly, Marcin Sieniek, Timo Kohlberger, Martin Ma, Wei-Hung Weng, Atilla Kiraly, Sahar Kazemzadeh, Zakkai Melamed, et al. Elixr: Towards a general purpose x-ray artificial intelligence system through alignment of large language models and radiology vision encoders. arXiv preprint arXiv:2308.01317, 2023. 2

[32] X Yang, R Shi, T He, Y Wang, L Wang, and L Wen. Rethinking foundation models for medical image classification through a benchmark study on medmnist. arXiv preprint arXiv:2312.06600, 2023. 5

[33] Jingfeng Yao, Xinggang Wang, Yuehao Song, Huangxuan Zhao, Jun Ma, Yajie Chen, Wenyu Liu, and Bo Wang. Eva-

x: A foundation model for general chest x-ray analysis with self-supervised learning. npj Digital Medicine, 8(1):678, 2025. 2

[34] Kihyun You, Jawook Gu, Jiyeon Ham, Beomhee Park, Jiho Kim, Eun K Hong, Woonhyuk Baek, and Byungseok Roh. Cxr-clip: Toward large scale chest x-ray language-image pre-training. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 101–111. Springer, 2023. 2

[35] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: A multimodal biomedical foundation model pretrained from fifteen mil lion scientific image-text pairs. arxiv 2023. arXiv preprint arXiv:2303.00915, 70, 2023. 1, 2, 4, 5, 6, 7, 8

[36] Mohammad Zunaed, Anwarul Hasan, and Taufiq Hasan. Improving pediatric pneumonia diagnosis with adult chest x-ray images utilizing contrastive learning and embedding similar ity. arXiv preprint arXiv:2404.12958, 2024. 1