# Real-World Perception for Autonomous Driving in Adverse Weather: Enhancing Standard Detectors via Foundation-Guided

Auto-Annotation

Sepideh Gohari, Goodarz Mehr, Azim Eskandarian, Fellow, IEEE

Abstract—Standard deployment-ready object detectors for autonomous vehicles degrade in adverse weather and lighting conditions without being trained on extensive domain-specific data. While large-scale vision foundation models offer robust zero-shot generalization, their high computational cost makes them impractical for real-time deployment. To bridge this gap, we propose a foundation-guided auto-annotation pipeline that enhances standard detectors without architectural changes. We first benchmark three distinct models, YOLOv8, Co-DETR, and SAM3, on our custom real-world driving dataset spanning 25 unique operational scenarios across various route, weather, and lighting conditions. Based on our analysis, SAM3 demonstrates superior accuracy and resilience across all scenarios. Thus, we deploy it as an offline auto-annotator to generate pseudolabels on the unannotated subset of our dataset. Fine-tuning the baseline YOLOv8 on these annotations yields a 16.04% higher overall mean Average Precision (mAP) and improves cross-environmental stability compared to the baseline model, highlighted by a 32.73% and 28.65% mAP increase in Residential Direct Sunlight and Highway Fog, respectively. These results demonstrate that standard detectors can achieve environmental resilience without the need for extensive manual annotation or architectural modifications.

Index Terms—Adverse weather perception, object detection, real-world dataset, auto-annotation, pseudo-labeling, autonomous driving.

## I. INTRODUCTION

UTONOMOUS vehicles have made significant progress in recent years, performing reliably within their designated operation areas. However, the unreliability of visionbased perception systems in adverse weather conditions hindered their expansion into broader environments [1]. While modern 2D object detection algorithms, such as YOLO and its variants [2], perform well under ideal weather and lighting conditions [3], their performance degrades when they are subjected to real-world physical noise such as rain and snow [1]. Since these algorithms are based on convolutional neural networks (CNNs) and rely on localized receptive fields, they depend heavily on fine-grained textures and clear edges. In adverse weather and lighting conditions, physical noise degrades these visual cues by corrupting the high-frequency textures and fine structural geometries. Lacking broader spatial context, these localized features are no longer sufficient for reliable

![](images/87f45e391f8800605adb9481245104a71db745db6b95adbda205f0b5b60011ee.jpg)  
Fig. 1. End-to-end framework of the proposed foundation-guided perception pipeline.

detection [4]–[6]. As a result, the number of misclassifications and missed detections increases as environmental conditions deteriorate [7].

Overcoming these limitations requires addressing challenges within both data curation and the model structure pipeline. On the data front, existing approaches face a dual bottleneck. Researchers frequently rely on synthetically generated adverse weather datasets to avoid manual labeling [8], [9]. However, such algorithmic augmentations cannot fully replicate the complex physical noise and sensor artifacts present in realworld data [10]. On the other hand, collecting real-world adverse weather data requires manual ground-truth annotation, which is prohibitively labor-intensive [10]. A solution to these problems is a framework that can exploit unannotated realworld driving data without relying on extensive manual effort.

On the model structure front, perception pipelines have increasingly adopted attention-based paradigms, ranging from Detection Transformers (DETR) [11] to large-scale vision foundation models like the Segment Anything Model (SAM) [12] and its more recent variant, SAM3 [13]. Unlike traditional CNNs, these models replace localized receptive fields with local (windowed) and global attention. By capturing short- and long-range contextual relationships across the entire image, these models can infer object presence even when highfrequency textures are degraded by physical noise [14]. Furthermore, pre-training on large-scale datasets equips them with rich spatial priors and robust zero-shot generalization across challenging environments [15]. However, their large size and heavy computational requirements make direct deployment on autonomous vehicles for real-time inference impractical [16], [17].

A practical solution to this trade-off is offline knowledge transfer, which combines the resilience of foundation models to physical noise with the speed of standard detectors. In this framework, a large-scale foundation model acts as the offline pseudo-labeler of unannotated, visually challenging data [18], [19]. Fine-tuning lightweight detectors on these generated annotations enables them to capture complex environmental representations [20], which improves their performance in adverse conditions without modifying their architecture or increasing inference latency.

To implement this strategy, we propose the framework illustrated in Fig. 1. We use a real-world driving dataset specifically collected by our research group for adverse weather perception, covering diverse weather, lighting, and route conditions [21]. On a manually annotated subset, we evaluate three architecturally distinct candidate models: YOLOv8 (CNNbased) [22], Co-DETR (transformer-based) [23], and SAM3 (vision foundation) [13]. Among these candidates, SAM3 achieves optimal overall performance with the highest mAP and low standard deviation across all scenarios, confirming its viability as an offline auto-annotator. We then use SAM3 to generate pseudo-labels for the unannotated keyframes of the dataset and fine-tune baseline YOLOv8 on this data. This pipeline substantially improves YOLOv8’s detection accuracy in adverse weather and lighting conditions while preserving its original architecture.

In summary, our main contributions are as follows:

• We present a comparison of three architecturally distinct detectors: YOLOv8, Co-DETR, and SAM3. We evaluate these models across our custom real-world driving dataset [21], covering 25 combinations of route (e.g., Campus, Highway), weather (e.g., Rain, Fog), and lighting (e.g., Direct sunlight, Low light) conditions.

• We implement a foundation-guided auto-annotation pipeline that uses SAM3 to generate offline pseudo-labels for unannotated adverse weather driving data, eliminating the need for costly manual annotation.

• We show that fine-tuning baseline YOLOv8 on these pseudo-labels increases overall mAP in adverse conditions by 16.04% (from a 34.19% baseline mAP to 50.23%), with peak gains of 32.73% in Residential Direct Sunlight and 28.65% in Highway Fog, all without modifying YOLOv8’s original architecture.

## II. METHODOLOGY

This section presents our end-to-end pipeline, starting from dataset curation and stratified ground-truth annotation to foundation-guided auto-annotation and fine-tuning. An overview of the framework is shown in Fig. 1.

## A. Dataset Overview and Adverse Conditions Taxonomy

The dataset used in this work was specifically designed by our research group [21] as a small-scale, controlled benchmark for driving under adverse weather and lighting conditions. This dataset was collected using the X-CAR research platform [24], which is equipped with five surround-view cameras (2880 × 1860 resolution). Consequently, each recorded timestamp in the dataset consists of five high-definition surroundview images. The dataset spans four distinct route topologies (Campus, Highway, Residential, Rural) with six independent conditions, comprising three weather domains (Fog, Rain, Snow) and three lighting domains (Direct Sunlight, Low Light, No Light). Including an additional Sufficient Light for the Campus route, this structured taxonomy yields 25 unique operational scenarios.

## B. Dataset Pruning and Keyframe Extraction

To adapt this raw dataset for our specific evaluation and auto-annotation pipelines, we implemented a temporal pruning process and eliminated out-of-domain data. While the original continuous collection comprised 1,218,990 raw frames across the five surround-view cameras (representing 243,798 distinct timestamps), off-route transit periods at the beginning and end of each recording sequence (e.g., initial startup or final parking maneuvers) were removed to ensure the data strictly reflected the target operational route. This boundary isolation reduced the dataset to 849,795 active, in-domain driving frames (169,959 timestamps per camera).

The raw sensor data was originally captured at a frequency of 10 Hz. Sampling at this rate introduces visual redundancy, as frames captured 0.1 seconds apart are nearly identical. To resolve this, we applied a temporal downsampling protocol and extracted keyframes at a rate of 2 Hz. This interval was selected to create an optimal balance, it eliminates consecutive frame duplication while ensuring that scene transitions remain continuous.

This filtering, followed by the downsampling process, reduced the data into a pool of 31,269 timestamps, yielding a total multi-view collection of 156,345 keyframes across the five cameras. This processed dataset is then used for subsequent manual annotation and pseudo-labeling, with the keyframe distribution divided across the target routes: Campus (58,075), Residential (48,830), Rural (34,285), and Highway (15,155).

## C. Stratified Sampling and Ground-Truth Curation

To construct an unbiased, reliable baseline for evaluating the candidate object detectors, we extracted a subset of the keyframe pool for manual ground-truth annotation. We established a sampling protocol to prevent environmental or spatial bias across the 25 operational scenarios. For each scenario, exactly 30 samples were selected. We distributed the sampling across the sensor suite by extracting six keyframes from each of the five surround-view cameras per scenario to guarantee complete 360° spatial representation. This structure yielded a balanced, multi-view evaluation set of 750 sampled keyframes (25 scenarios × 30 samples). The sample selection process was guided by two primary goals, maximizing visual diversity by enforcing a meaningful temporal distance between samples, and ensuring the presence of at least one target object of interest (e.g., a vehicle or pedestrian). However, satisfying both criteria while maintaining the rigid 30-sample criteria occasionally conflicted with real-world data distributions. Therefore, we relaxed both the temporal distance and object presence constraints in shorter recording sequences or sparsely populated environments. As a result, this sampling protocol yielded a small subset of samples containing no target objects. Retaining these background-only samples strengthens the evaluation by testing how well models handle empty scenes without triggering false-positive detections.

![](images/1826de8144cc56d986fd1e79d6537769ef665e29892060b913e679cbe301ebea.jpg)  
(a)

![](images/887303a417c3cd358d5e12400c21d4f1826ecf0c6549db0cee4f7de771d47be0.jpg)  
(b)  
Fig. 2. Class distribution of the manually annotated ground-truth subset (N = 3, 169 total instances). (a) Instance count for each object class broken down by route type (Campus, Highway, Residential, Rural) and total count. (b) Overall percentage distribution of object classes across the ground-truth.

## D. Manual Annotation Protocol and Taxonomy

After curating the subset of 750 samples, we manually annotated each sample using the Computer Vision Annotation Tool (CVAT) [25] to construct our ground-truth. This groundtruth served as the benchmark to evaluate the performance of YOLOv8, Co-DETR, and SAM3 as three structurally distinct object detection frameworks under adverse conditions. To ensure a fair comparison, the ground-truth annotation categories had to directly match the supported output categories of all candidate models. SAM3 is an open-vocabulary foundation model with no constraints on its output categories. However, both YOLOv8 and Co-DETR are closed-vocabulary models and their detection output is restricted to the 80 standard categories of the Microsoft COCO dataset [26]. Consequently, our ground-truth categories were bounded by the COCO category set to maintain consistency across our candidate models.

Among the standard COCO categories, only a subset of eight classes is directly relevant to driving perception tasks: car, truck, bus, bicycle, motorcycle, person, traffic light, and stop sign. To ensure the long-term utility and scalability of our curated ground-truth, we annotated this subset using 20 fine-grained classes. However, for the scope of this study, we merged these granular classes through a mapping layer to match the eight baseline COCO categories and excluded all non-target classes.

During annotation, we drew tight bounding boxes around each target instance to exclude unnecessary background pixels.

To maintain consistent evaluation standards under adverse visibility, we annotated all identifiable objects where the bounding box dimensions were greater than 20 × 20 pixels, regardless of heavy occlusion or frame-edge truncation. For instances that were visually identifiable but fell below the $2 0 ~ \times ~ 2 0$ pixel threshold, we assigned them to the ignore category corresponding to their class. However, we removed all these ignore classes alongside the non-target categories prior to the evaluation process in this study. This annotation protocol ensures that small, ambiguous background objects do not artificially skew the precision and recall across our candidate detectors.

In total, this annotation procedure yielded 3,169 bounding box instances across our 750 samples. Fig. 2a details the distribution of ground-truth categories across the four routes, as well as cumulative totals, and Fig. 2b shows the overall class proportions across the entire ground-truth. With an average density of 4.23 instances per frame, the dataset is dominated by standard vehicular actors (59.9% cars) and vulnerable road users (17.7% persons). This class imbalance reflects real-world driving conditions, enabling us to test how well detectors locate and classify objects across diverse operational domains.

## E. Evaluation Models

We selected three object detection frameworks representing distinct architectural paradigms, namely a real-time detector (YOLOv8), a high-performance transformer (Co-DETR), and an open-vocabulary foundation model (SAM3). We evaluated these models against our manually annotated ground-truth to test their robustness under adverse weather and lighting conditions.

Our first candidate model is YOLOv8, a real-time, singlestage detector widely adopted for autonomous driving and perception tasks due to its recognized balance of computational efficiency and accuracy. Among its five scaled configurations (Nano, Small, Medium, Large, and Extra-Large), we selected the YOLOv8-Large (YOLOv8l) variant. While the Extra-Large (YOLOv8x) version offers a 1.0% mAP gain on the COCO dataset, it introduces a 58.6% latency penalty, increasing runtime from 9.06 to 14.37 ms. Selecting YOLOv8l gives us strong multi-class detection capabilities without overparameterizing the model.

For our second candidate, we chose Co-DETR, a collaborative detection transformer that optimizes encoder learning through parallel auxiliary heads. To maintain consistency with the closed-vocabulary COCO classes used in the YOLOv8 setup, we selected a COCO-pretrained configuration featuring a Vision Transformer Large (ViT-L) backbone. This architecture achieves a detection accuracy of 65.9% Average Precision on the COCO dataset. Deploying this high-capacity model establishes a performance upper bound, allowing us to compare performance against real-time detectors and open-vocabulary architectures.

Our third candidate model is SAM3, an open-vocabulary foundation model optimized for Promptable Concept Segmentation (PCS) using a text-conditioned, DETR-based detector. Despite its zero-shot setup, SAM3 achieves a competitive 56.4% mAP on the COCO dataset. To ensure a fair comparison among the candidates, we evaluated SAM3 using text-only prompting without manual point guidance or image exemplars, extracting bounding boxes directly from its box prediction head. For resolving semantic ambiguity from broad openvocabulary queries, we prompted the model with 14 finegrained class names, which we subsequently mapped and merged back into the 8 baseline COCO driving categories during post-processing. Finally, we eliminated spatial redundancies caused by merging sub-classes, such as overlapping bounding boxes when merging ”rider” and ”pedestrian” into ”person”, by applying Non-Maximum Suppression (NMS) to generate clean, non-duplicated predictions.

## F. Auto-Annotation Framework and Fine-Tuning

After evaluating the three candidate detectors under adverse weather and lighting conditions, we selected the architecture demonstrating the highest accuracy and environmental robustness as our offline auto-annotator. We then used this selected model to process unannotated keyframes across our entire dataset, generating a comprehensive set of pseudo-labels. To prevent data leakage and maintain a fair evaluation, we kept the manually annotated subset entirely separate from this autoannotation pipeline, reserving it strictly as our ground-truth set for final evaluation.

Following this stage, we used the pool of keyframes annotated with these pseudo-labels to fine-tune the YOLOv8l model, adapting this general-domain detector for specialized perception under adverse weather and lighting conditions. With this setup, we evaluated how effectively knowledge transfers across models. Specifically, we examined whether a real-time, lightweight detector can learn from a large, highcapacity offline model to improve its detection performance under visually degraded conditions.

## III. EXPERIMENTS AND DISCUSSION

## A. Implementation Details

We ran all evaluations on an NVIDIA GeForce RTX 3090 GPU. To prevent dependency conflicts, we isolated each candidate model within its dedicated Docker environment. During inference, we set a unified confidence threshold of $\tau = 0 . 5$ across all models to balance precision and recall.

TABLE I  
PERFORMANCE COMPARISON (MAP<sub>50:95</sub> (%)) UNDER VARYING WEATHER AND LIGHTING CONDITIONS ACROSS FOUR ROUTE TYPES.
<table><tr><td>Environment</td><td>Condition</td><td>YOLOv8</td><td>CoDETR</td><td>SAM3</td></tr><tr><td rowspan="7">Campus</td><td>Direct Sunlight</td><td>29.71</td><td>48.44</td><td>53.47</td></tr><tr><td>Fog</td><td>36.68</td><td>68.66</td><td>68.93</td></tr><tr><td>Low Light</td><td>32.61</td><td>54.10</td><td>46.06</td></tr><tr><td>No Light</td><td>37.19</td><td>38.63</td><td>51.72</td></tr><tr><td>Rain</td><td>38.77</td><td>57.55</td><td>55.27</td></tr><tr><td>Snow</td><td>30.88</td><td>52.10</td><td>55.58</td></tr><tr><td>Sufficient Light*</td><td>41.72</td><td>60.47</td><td>49.61</td></tr><tr><td rowspan="6">Highway</td><td>Direct Sunlight</td><td>34.86</td><td>54.64</td><td>50.13</td></tr><tr><td>Fog</td><td>16.83</td><td>42.78</td><td>55.71</td></tr><tr><td>Low Light</td><td>36.12</td><td>62.06</td><td>60.53</td></tr><tr><td>No Light</td><td>12.88</td><td>25.62</td><td>26.82</td></tr><tr><td>Rain</td><td>19.51</td><td>47.21</td><td>43.45</td></tr><tr><td>Snow</td><td>25.74</td><td>53.28</td><td>48.40</td></tr><tr><td rowspan="6">Residential</td><td>Direct Sunlight</td><td>27.01</td><td>62.38</td><td>60.39</td></tr><tr><td>Fog</td><td>26.40</td><td>42.39</td><td>45.43</td></tr><tr><td>Low Light</td><td>49.98</td><td>54.97</td><td>68.45</td></tr><tr><td>No Light</td><td>23.65</td><td>30.42</td><td>47.87</td></tr><tr><td>Rain</td><td>44.78</td><td>56.63</td><td>65.55</td></tr><tr><td>Snow</td><td>35.03</td><td>51.58</td><td>48.18</td></tr><tr><td rowspan="6">Rural</td><td>Direct Sunlight</td><td>48.80</td><td>76.36</td><td>72.36</td></tr><tr><td>Fog</td><td>53.07</td><td>78.05</td><td>73.07</td></tr><tr><td>Low Light</td><td>39.75</td><td>47.32</td><td>63.70</td></tr><tr><td>No Light</td><td>28.23</td><td>29.04</td><td>53.70</td></tr><tr><td>Rain</td><td>44.34</td><td>76.65</td><td>69.35</td></tr><tr><td>Snow</td><td>47.68</td><td>66.29</td><td>63.19</td></tr><tr><td rowspan="2">Overall Summary</td><td>Mean (µ)</td><td>34.19</td><td>53.21</td><td>56.14</td></tr><tr><td>STD (σ)</td><td>10.68</td><td>14.31</td><td>10.98</td></tr></table>

Serves strictly as Campus baseline reference and is excluded from the overall metrics $( \mu , \sigma )$ to preserve cross-environmental symmetry.

This value prevents false positives caused by lower thresholds while avoiding the suppression of distant or occluded objects, such as vulnerable road users, at higher thresholds.

## B. Evaluation Metrics

We evaluated model performance using the standard COCO protocol, adopting $\mathrm { m A P _ { 5 0 : 9 5 } }$ as our primary metric. For a given object class c and Intersection over Union (IoU) threshold i, Average Precision (AP) integrates the Area Under the Curve (AUC) of the Precision-Recall distribution:

$$
A P _ { c , i } = \int _ { 0 } ^ { 1 } P ( R ) d R\tag{1}
$$

The m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ metric averages these AP values across all target classes and IoU thresholds ranging from 0.50 to 0.95.

## C. Candidate Detector Evaluation under Adverse Conditions

Table I presents the quantitative evaluation of YOLOv8l, Co-DETR (ViT-L), and SAM3 across varying weather and lighting conditions, measured via standard $\mathrm { m A P _ { 5 0 : 9 5 } }$ at a unified confidence threshold $( \tau = 0 . 5 )$

Across all evaluated conditions, YOLOv8 consistently underperforms the transformer architectures, recording an overall mean of 34.19% mAP compared to 53.21% mAP for Co-DETR and 56.14% mAP for SAM3. On average, YOLOv8 lags behind by roughly 20% mAP, with the gap reaching a maximum difference of 38.88% against SAM3 under Highway

Fog (16.83% vs. 55.71% mAP). This gap stems from YOLO’s dependence on local receptive fields. When adverse conditions such as fog or glare degrade fine texture cues, local features become unreliable. In contrast, Co-DETR and SAM3 use attention mechanisms, allowing them to infer bounding boxes using the broader spatial and semantic context.

Further differences are observed between the two transformer architectures under No Light conditions. Although performance drops across all models in darkness, SAM3 outperforms Co-DETR, particularly in No Light Rural settings (53.70% vs. 29.04% mAP). Since Co-DETR relies solely on visual features, it is vulnerable when underexposure eliminates image contrast. However, SAM3 uses text-conditioned semantic embeddings to guide visual cross-attention. Combined with its large-scale pre-training, this helps stabilize object localization even when visual features are degraded.

Additionally, all models experience a sharp accuracy drop under the Highway No Light scenario. In this setting, high driving speeds induce motion blur while a lack of proper lighting eliminates image contrast. Their combined effect yields the lowest scores across the dataset, 12.88% for YOLO, 25.62% for Co-DETR, and 26.82% mAP for SAM3. This decline highlights a limitation of camera-only perception, showing that passive vision alone fails when adverse environmental conditions become severe.

Evaluating overall accuracy (µ) and standard deviation (σ) across all conditions and routes reveals the differences in model stability. YOLO records the lowest standard deviation $( \sigma = 1 0 . 6 8 )$ , which reflects its consistently low performance $( \mu = 3 4 . 1 9 \% )$ . Between the two transformer-based models, Co-DETR achieves a high mean accuracy $( \mu ~ = ~ 5 3 . 2 1 \% )$ but exhibits a high standard deviation $( \sigma = 1 4 . 3 1 )$ , showing that its predictions are sensitive to environmental changes. SAM3 offers the optimal balance, achieving the highest overall performance $( \mu ~ = ~ 5 6 . 1 4 \% )$ alongside high crossenvironmental stability $( \sigma = 1 0 . 9 8 )$ . Nevertheless, an overall average of 56.14% mAP remains insufficient for safety-critical autonomous driving. These results indicate that even state-ofthe-art vision models cannot overcome extreme environmental degradations on their own, reinforcing the need to integrate active sensors like LiDAR or Radar to improve safety and real-world reliability.

## D. Efficacy of Foundation-Guided Fine-Tuning

Based on our initial evaluation, we selected SAM3 as our offline model due to its optimal balance of accuracy and stability across varying weather, lighting, and route conditions. We used this model to automatically annotate the unannotated keyframes, and applied these generated pseudo-annotations to fine-tune the baseline YOLO architecture. To evaluate the efficacy of this foundation-guided fine-tuning, we compared the baseline pre-trained YOLO against the fine-tuned model across all test scenarios, with overall metrics summarized in Table II.

Fine-tuning improved overall detection performance, raising the average mAP from 34.19% to 50.23%. Alongside this 16.04% improvement, the standard deviation decreased from

TABLE II  
OVERALL PERFORMANCE SUMMARY: BASELINE VS. FINE-TUNED YOLOV8.
<table><tr><td>Model Architecture</td><td>Mean mAP (µ)</td><td>STD (σ)</td></tr><tr><td>Baseline YOLO (Pre-trained)</td><td>34.19</td><td>10.68</td></tr><tr><td>Fine-Tuned YOLO (SAM3-Guided)</td><td>50.23</td><td>9.96</td></tr><tr><td>Overall Improvement (∆)</td><td>+16.04</td><td>-0.72</td></tr></table>

![](images/c746f483c253b2b8c92373c9ba815adeb0cda2ba2797cae1e938776101e95f18.jpg)  
Fig. 3. Performance improvements $( \mathrm { m A P _ { 5 0 - 9 5 } ) }$ of the SAM3-guided finetuned YOLOv8 model versus the pre-trained baseline, sorted by the magnitude of improvement. Grey markers indicate the baseline accuracy, while colored markers denote the fine-tuned model across all 25 scenarios.

10.68 to 9.96. The lower standard deviation indicates that the fine-tuned YOLO model inherited the cross-environmental stability of SAM3, yielding more consistent predictions across changing weather and lighting conditions.

Fig. 3 illustrates performance changes across all 25 environmental conditions. The fine-tuning approach was particularly effective in adverse scenarios where the baseline model previously failed. Most notably, the fine-tuned model achieved major gains in conditions like Residential Direct Sunlight (+32.73% mAP) and Highway Fog (+28.65% mAP). These findings show that using a large offline foundation model for auto-annotation allows lightweight detectors to overcome feature representation limits without manual labeling effort, maintaining robust perception in adverse weather.

## E. Qualitative Comparison

To visually evaluate our fine-tuning approach, Fig. 4 compares predictions from the three candidate models and the finetuned detector against the ground-truth across four representative scenarios: Residential Direct Sunlight, Highway No Light,

![](images/2fd99a7364013127a3dc0edddddaed42e5a1ced73605bc77fec648219044b454.jpg)  
Fig. 4. Qualitative detection comparison across four operational scenarios: Residential Direct Sunlight, Highway No Light, Campus Fog, and Rural Rain. Columns display (a) Ground Truth manual annotations alongside outputs from (b) Baseline YOLOv8, (c) Co-DETR, (d) SAM3, and (e) Fine-tuned YOLOv8. Note that while baseline YOLOv8 misses heavily obscured targets, the foundation-guided fine-tuned model successfully recovers those bounding boxes (Best viewed in color and zoomed in).

Campus Fog, and Rural Rain. We selected these specific scenes to capture pronounced instances of each condition across all four route types. Residential Direct Sunlight demonstrates our largest improvement after fine-tuning (+32.73% mAP), whereas Highway No Light represents the lowest performance across all models. Finally, Campus Fog and Rural Rain add environmental diversity and illustrate detector behavior under distinct forms of visual degradation.

Across all four scenarios, baseline YOLOv8 performs worst among the evaluated models due to its sensitivity to environmental noise. Its detections remain restricted to well-lit, closerange targets, failing to detect occluded, distant, or poorly illuminated objects in adverse conditions like Highway No Light and Rural Rain. In contrast, the fine-tuned YOLOv8 model detects more objects across every scenario than its baseline counterpart, demonstrating a clear accuracy improvement.

Comparing predictions between Co-DETR and SAM3 confirms SAM3’s greater overall robustness in adverse conditions, supporting its role as our offline auto-annotator. While Co-DETR occasionally identifies targets SAM3 misses or misclassifies, such as the truck in Rural Rain, its performance degrades in severe conditions, where it fails to detect any vehicles in Highway No Light.

Comparing fine-tuned YOLOv8 against SAM3 shows that predictions made by the fine-tuned detector closely match those of SAM3, rather than those of the baseline YOLOv8. However, the fine-tuned model does not blindly replicate SAM3’s pseudo-labels. In Campus Fog, it eliminates a falsepositive vehicle predicted by SAM3, and in Rural Rain, it corrects a mislabeled vehicle class. This demonstrates that finetuning allows YOLOv8 to absorb SAM3’s detection strengths without compromising its own classification reliability.

## IV. CONCLUSION

In this paper, we demonstrated that standard object detectors can adapt to adverse real-world conditions without modifying their architecture or extensive training with manually annotated data. We first compared the performance of three distinct candidate models on our custom dataset covering diverse routes, weather, and lighting conditions. Due to SAM3’s optimal balance of accuracy and stability across these 25 real world driving scenarios, we selected it as our offline autoannotator. Fine-tuning a baseline YOLOv8 model on these pseudo-labels generated by SAM3 yielded performance gains, particularly in adverse environments like Highway Fog and Residential Direct Sunlight, resulting in mAP gains of 28.65% and 32.73%, respectively. Overall, the fine-tuned YOLOv8 achieved a 16.04% mAP improvement compared to the baseline model, reaching an average of 50.23% mAP across all scenarios. However, this overall performance indicates that passive camera systems still face physical limits in the presence of real-world physical noise. Achieving true operational safety ultimately requires integrating these optimized pipelines with active multi-sensor fusion algorithms to ensure reliable perception.

## ACKNOWLEDGMENTS

This work builds upon the master’s thesis of Xuelai Du [21]. The authors greatly appreciate the extensive data collection and initial experimental setup that enabled this research.

## REFERENCES

[1] Y. Yuan, W. Dong, S. Yang, and T. Wu, “Awd-yolo enhancing autonomous driving perception reliability in adverse weather,” Scientific Reports, 2026.

[2] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 779– 788.

[3] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, “Object detection in 20 years: A survey,” Proceedings of the IEEE, vol. 111, no. 3, pp. 257–276, 2023.

[4] B.-T. Tran-Le, V. Patel, V.-T. Huynh, M.-K. Tran, K. Agrawal, M.-T. Tran, and T. V. Nguyen, “Towards safer roads: benchmarking object detection models in complex weather scenarios,” Machine Vision and Applications, vol. 36, no. 4, p. 94, 2025.

[5] K. Vinciguerra and L. Marchegiani, “Clouded judgments: An empirical analysis of vision-based object detection systems in harsh weather conditions,” in 2026 IEEE 23rd International Multi-Conference on Systems, Signals & Devices (SSD). IEEE, 2026, pp. 816–821.

[6] V. S. Patel, K. Agrawal, and T. V. Nguyen, “A comprehensive analysis of object detectors in adverse weather conditions,” in 2024 58th Annual Conference on Information Sciences and Systems (CISS). IEEE, 2024, pp. 1–6.

[7] F. Pettersen and H. Zhu, “Robustness of object detection of autonomous vehicles in adverse weather conditions,” arXiv preprint arXiv:2602.12902, 2026.

[8] C. Sakaridis, D. Dai, and L. Van Gool, “Semantic foggy scene understanding with synthetic data,” International Journal of Computer Vision, vol. 126, no. 9, pp. 973–992, 2018.

[9] S. S. Halder, J.-F. Lalonde, and R. d. Charette, “Physics-based rendering for improving robustness to rain,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 10 203–10 212.

[10] C. Sakaridis, D. Dai, and L. Van Gool, “Acdc: The adverse conditions dataset with correspondences for semantic driving scene understanding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 765–10 775.

[11] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision. Springer, 2020, pp. 213– 229.

[12] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[13] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[14] M. M. Naseer, K. Ranasinghe, S. H. Khan, M. Hayat, F. Shahbaz Khan, and M.-H. Yang, “Intriguing properties of vision transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 23 296– 23 308, 2021.

[15] Z. Wang, Y. Zhang, Z. Zhang, Z. Jiang, Y. Yu, L. Li, and L. Li, “Exploring semantic prompts in the segment anything model for domain adaptation,” Remote Sensing, vol. 16, no. 5, p. 758, 2024.

[16] S. Saha and L. Xu, “Vision transformers on the edge: A comprehensive survey of model compression and acceleration strategies,” Neurocomputing, vol. 643, p. 130417, 2025.

[17] Z. Lin, R. Tous, and B. Otero, “Deploying vision foundation ai models on the edge. the sam2 experience,” in International Work-Conference on Artificial Neural Networks. Springer, 2025, pp. 423–434.

[18] I. Radosavovic, P. Dollar, R. Girshick, G. Gkioxari, and K. He, “Data´ distillation: Towards omni-supervised learning,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4119–4128.

[19] N. F. Chen, “Pseudo-labels for supervised learning on dynamic vision sensor data, applied to object detection under ego-motion,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 644–653.

[20] B. Xu, M. Chen, W. Guan, and L. Hu, “Efficient teacher: Semi-supervised object detection for yolov5,” arXiv preprint arXiv:2302.07577, 2023.

[21] X. Du, “Development and analysis of a small-scale controlled dataset with various weather conditions, lighting, and route types for autonomous driving,” Master’s thesis, Virginia Polytechnic Institute and State University, Blacksburg, VA, 2024.

[22] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[23] Z. Zong, G. Song, and Y. Liu, “Detrs with collaborative hybrid assignments training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 6748–6758.

[24] G. Mehr, P. Ghorai, C. Zhang, A. Nayak, D. Patel, S. Sivashangaran, and A. Eskandarian, “X-car: An experimental vehicle platform for connected autonomy research,” IEEE Intelligent Transportation Systems Magazine, vol. 15, no. 2, pp. 41–57, 2022.

[25] CVAT.ai Corporation, “Computer Vision Annotation Tool (CVAT),” Nov. 2026. [Online]. Available: https://www.cvat.ai/

[26] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European conference on computer vision. Springer, 2014, pp. 740–755.

Sepideh Gohari received the B.S. degree in electrical engineering and the M.S. degree in artificial intelligence and robotics from Ferdowsi University of Mashhad, Mashhad, Iran. She is currently pursuing the Ph.D. degree in electrical and computer engineering at the Autonomous Robots and Vehicles Lab (ARVL), at Virginia Commonwealth University (VCU), Richmond, VA, USA. Her research interests include computer vision, autonomous driving, robotics, and artificial intelligence.

Goodarz Mehr received the B.Sc. degree in mechanical engineering from Sharif University of Technology, Tehran, Iran, in 2016 and the M.Sc. and Ph.D. degrees in mechanical engineering from Virginia Tech, Blacksburg, VA, USA, in 2023 and 2024, respectively. He is currently a postdoctoral research associate at Autonomous Robots and Vehicles Lab (ARVL) at Virginia Commonwealth University (VCU), Richmond, VA, USA. His research interests include multi-agent robotics, stochastic planning models, and cooperative perception.

Azim Eskandarian (IEEE Fellow) received the B.S. and D.Sc. degrees in mechanical engineering from George Washington University (GWU), Washington, D.C., USA, and the M.S. degree in mechanical engineering from Virginia Tech, Blacksburg, VA, USA.

He has been the Alice T. and William H. Goodwin Jr. Dean of the College of Engineering at Virginia Commonwealth University (VCU), Richmond, USA, since 2023, where he also established the Autonomous Robots and Vehicle Laboratory (ARVL). Before that, he was a Professor and the Head of the Department of Mechanical Engineering at Virginia Tech since 2015, where he became the Nicholas and Rebecca Des Champs Chair Professor in 2018, and where he established the Autonomous Systems and the Intelligent Machines Laboratory, to conduct research in intelligent and autonomous vehicles and mobile robotics. As a Professor at George Washington University, he was the Co-Founder of the National Crash Analysis Center in 1992, the founding director of the Center for Intelligent Systems Research from 1995 to 2015, and the director of the University’s area of excellence in Transportation Safety and Security from 2003 to 2015. He was an Assistant Professor at The Pennsylvania State University in York, PA, USA, from 1989 to 1992 and an Engineer/Project Manager in the industry from 1983 to 1989.

Dr. Eskandarian is a Fellow of IEEE, ASME, and SAE. He was elected to the Virginia Academy of Science, Engineering, and Medicine (VASEM) in 2026. He received the IEEE Intelligent Transportation Society Outstanding Researcher Award in 2017 and the GWU School of Engineering Outstanding Researcher Award in 2013. He served as Editor-in-Chief of the IEEE Transactions on Intelligent Transportation Systems from 2019 to 2023.