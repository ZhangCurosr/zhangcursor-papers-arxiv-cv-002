# GA-EIRFS: A GEOMETRY-AUGMENTED REPEAT-FACTOR SAMPLING METHOD FOR LONG-TAILED LIDAR 3D OBJECT DETECTION

Taufiq Ahmed<sup>†</sup> , Constantino Alvarez Casado<sup>´</sup> <sup>†</sup> , Daniel Herrera Castro<sup>†</sup> , Sasan Sharifipour<sup>†</sup> , Abhishek Kumar<sup>⋆</sup> , Miguel Bordallo Lopez ´ <sup>†</sup>

<sup>†</sup>Center for Machine Vision and Signal Analysis (CMVS), University of Oulu, Finland <sup>⋆</sup>University of Jyvaskyl ¨ a, Jyv ¨ askyl ¨ a, Finland¨

## ABSTRACT

Long-tailed 3D object detection is treated as a class-frequency problem, but LiDAR supervision quality depends on object observability: similar frequencies can hide different geometric evidence. We introduce Geometry-Augmented Exponentially Weighted Instance-Aware Repeat Factor Sampling (GA-EIRFS), a detector-agnostic method that modulates a frequency-based repeat factor with a fixed geometry score combining point count, surface-normal entropy, and surface coverage. GA-EIRFS changes only frame-sampling probabilities, leaving the detector and inference unchanged. On nuScenes it improves mean average precision (mAP) and the nuScenes detection score (NDS) in four converged experiments with CenterPoint and PointPillars over two seeds; for CenterPoint at seed 666, mAP rises from 0.552 to 0.563 and bicycle AP from 0.306 to 0.359. Per-class gains correlate with the class sampling-weight increase (Spearman ρ = 0.70, p = 0.025) but not with geometry score alone $( \rho ~ = ~ 0 . 3 2 , ~ p ~ = ~ 0 . 3 7 )$ , so geometry amplifies frequency-driven need. KITTI results vary across seeds, most for the rarest class. Code: https://github.com/Multimodal-Sensing-Lab/GA-EIRFS.

Index Terms— Long-tailed 3D object detection, LiDAR point clouds, class imbalance, repeat factor sampling, data resampling

## 1. INTRODUCTION

LiDAR-based 3D object detection supports autonomous vehicles, mobile robots, inspection drones and roadside sensing, where missing an uncommon but safety-relevant object can have serious consequences. Annotated datasets often contain far fewer examples of such objects than of common categories. For example, in the nuScenes database training split [1], annotated cars outnumber bicycles by 41.5:1, which presents a challenge in learning to detect underrepresented classes [2, 3]. We evaluate on driving benchmarks, but the underlying problem of imbalanced training data extends to other LiDAR applications.

Sampling-based rebalancing addresses the imbalance without modifying the detector. Repeat Factor Sampling (RFS) [4] repeats an image according to the frequency of the rarest category it contains. Instance-Aware RFS (IRFS) [5] combines image and instance frequency through a geometric mean, and E-IRFS [6] applies an exponential function to that mean so that extremely rare categories receive a stronger adjustment. In 3D detection, Class-Balanced Grouping and Sampling (CBGS) duplicates frames by category occurrence [7]. Every method in this family reads annotation counts only.

Frequency is not the same as observability. LiDAR returns depend on object size, range, orientation, occlusion and surface structure. In the nuScenes training split a bicycle box contains 16.9 points on average, against 98.1 for a car and 240.3 for a trailer. Bicycle and trailer are both minority classes and a frequency-based rule gives them similar sampling weight, yet the evidence available per instance differs by more than an order of magnitude. Across the ten classes the frequency term and the geometry score are close to uncorrelated (Spearman $\rho = 0 . 1 6 , p = 0 . 6 5 )$ , as Figure 1 shows, so a rule defined on counts alone cannot separate a rare class that is easy to observe from one that is not.

![](images/1c0249502eb6fc6bd37592c00839a76fa4e2a160e210c47e72eab81430642192.jpg)  
Fig. 1. Rarity and geometric difficulty are close to independent across the ten nuScenes classes. Markers are classes, placed by the E-IRFS frequency term $q _ { c }$ and the geometry score $G _ { c }$ of Eq. (5). Grey curves are contours of $r _ { c } ( \alpha = 2 . 0 , \beta = 1 . 0 ) \colon$ vertical distance at fixed rarity is the sampling weight that geometry adds.

We propose Geometry-Augmented E-IRFS (GA-EIRFS), which estimates a fixed class-level geometry score from the annotated Li-DAR points before training and uses it to modulate the exponent of the E-IRFS repeat factor. Geometry amplifies an existing frequencydriven adjustment rather than acting on its own, and one parameter controls how much it contributes. The contributions are as follows:

• A geometry-augmented repeat factor that combines class frequency with class-level LiDAR observability and reduces exactly to E-IRFS when the geometry term is disabled.

• A geometry score built from three measurable properties of the annotated point sets, computed once before training, with no change to the detector, the loss or the inference path.

• An evaluation on nuScenes with two detectors and two seeds, a sweep of the geometry strength, an analysis of which classes benefit and why, and a transfer experiment on KITTI in which the gains are not consistent.

## 2. RELATED WORK

Rebalancing for long-tailed 3D detection. Long-tail mitigation for point clouds acts on the data distribution, on the training objective, or on the detector. CBGS builds a more balanced training distribution and groups related categories [7], Lee and Kim combine categoryspecific heads, dynamic loss averaging and contextual ground-truth sampling [8], and Peri et al. formalize long-tailed 3D detection with hierarchical supervision and evaluation [2]. Later work adds multimodal fusion [9], neighbour-based confidence adjustment [3], loss reweighting [10] or few-shot adaptation [11]. These methods change the objective, the architecture or the modalities, and none of them changes which frames the sampler draws. Frequency and difficulty have been separated before, but outside 3D geometry: CDB-S resamples classes by measured validation difficulty [12] and geometric priors have shaped the feature space in long-tailed classification [13]. Both derive difficulty from model behaviour rather than from the sensor evidence.

Geometry and difficulty-aware training. Point density has been used to guide internal point selection and attention in DA-3DSSD [14] and in density-aware set abstraction [15]. PointDrop generates sparse adversarial examples [16], and density-adaptive augmentation modifies point-cloud content [17]. All of these act inside the detector or inside a scene. The closest operational method is Curricular Object Manipulation (COM) [18], which groups database objects by distance, dimensions, orientation and occupancy, then progressively selects harder objects for ground-truth copy-paste augmentation. Its sampling unit is an object pasted into a scene and its difficulty is updated from model scores during training. Rare Example Mining [19] also separates rareness from difficulty, but it selects unlabelled tracks for annotation rather than rebalancing a labelled set, as do active-annotation methods [20, 21].

Gap and positioning. Two gaps follow. The sampling methods decide how often a frame is drawn from annotation counts alone, so they cannot separate a rare class the sensor observes well from one it observes poorly. The geometry-aware methods do read sensor structure, but they act inside the detector or inside a synthesised scene, and those that separate rarity from difficulty estimate difficulty from model behaviour or select data for annotation. We found no method that inserts a fixed, sensor-derived geometry score directly into a frame repeat factor. GA-EIRFS targets that gap: it scores class-level observability once from the annotated points and multiplies it into the exponent of an existing frequency-based repeat factor, leaving the detector, the loss and inference untouched. Section 3 defines the two priors and how they combine.

## 3. GEOMETRY-AUGMENTED REPEAT FACTOR SAMPLING

GA-EIRFS takes an annotated training split as input and returns one sampling probability per frame. Figure 2 shows the two priors, their combination, and the single point at which the training loop is affected.

## 3.1. Frequency prior

Let $f _ { i , c }$ be the fraction of training frames containing class c and $f _ { b , c }$ be the fraction of 3D annotated boxes assigned to c. Following E-IRFS [6], the frequency term is given by:

$$
q _ { c } = \sqrt { \frac { t } { \sqrt { f _ { i , c } f _ { b , c } } } } ,\tag{1}
$$

![](images/13901cbdba6fc668c0ad0e87a184bcdc2ed106ac19a8cbd1bf4db281a260edfc.jpg)  
Fig. 2. GA-EIRFS pipeline. The priors q<sub>c</sub> and $G _ { c }$ of Eqs. (1) and (5) are estimated once from the training split and enter the class repeat factor of Eq. (6), which becomes a frame sampling probability.

where t sets the threshold at which oversampling becomes active. The term grows as either frequency falls and treats classes with comparable occurrence statistics identically.

## 3.2. Class-level geometry prior

For every annotated instance, we collect the LiDAR points inside its ground-truth 3D box. Let $\mathcal { N } ( \cdot )$ denote min-max normalization across the classes of the dataset. Each component is constructed to lie in [0, 1], with higher values indicating a greater estimated geometric difficulty. The first component penalizes low point support and is defined as:

$$
D _ { c } = 1 - \mathcal { N } ( \log ( 1 + \bar { n } _ { c } ) ) ,\tag{2}
$$

where $\bar { n } _ { c }$ is the mean number of in-box points per instance of class c. The logarithm reduces the influence of classes with very high point counts on the normalization. In nuScenes, the traffic cone class has $D _ { c } = 1 . 0 0$ , with 9.2 points per instance on average, while the bus class has $D _ { c } = 0 . 0 0$ , with 312.4 points per instance.

The second component measures surface diversity through the distribution of local surface-normal orientations. Normals are estimated for each instance and assigned to 256 orientation bins. The normalized class-level entropy is defined as:

$$
H _ { c } = \mathcal { N } \Bigg ( - \sum _ { k = 1 } ^ { 2 5 6 } p _ { c , k } \log p _ { c , k } \Bigg ) ,\tag{3}
$$

where $p _ { c , k }$ is the probability assigned to orientation bin k for class c. A flat panel exhibits similar normal orientations and low entropy, whereas an irregular structure such as a bicycle frame can exhibit more diverse orientations and higher entropy. Instances with fewer than ten points are excluded from the entropy calculation because their normal estimates are unstable. Assigning zero entropy to these instances would artificially lower the estimated surface diversity of sparsely sampled classes.

The third component measures how sparsely the observed surface is covered,

$$
S _ { c } = 1 - \mathcal { N } \Big ( \overline { { ( n / A ) } } _ { c } \Big ) , \qquad A = 2 ( l w + l h + w h ) ,\tag{4}
$$

where n is the in-box point count and A the surface area of a box of dimensions $l ,$ w and h. The surface area replaces volume because a LiDAR sensor observes the outer surfaces, and normalizing by volume makes large hollow boxes such as the trailer and bus appear artificially empty. The geometry score is the convex combination

$$
G _ { c } = \lambda _ { D } D _ { c } + \lambda _ { H } H _ { c } + \lambda _ { S } S _ { c } , \qquad \lambda _ { D } + \lambda _ { H } + \lambda _ { S } = 1 .\tag{5}
$$

## 3.3. Geometry-augmented repeat factor

GA-EIRFS modulates the E-IRFS exponent with the geometry score computed as:

$$
r _ { c } = \exp ( \alpha q _ { c } \left( 1 + \beta G _ { c } \right) ) ,\tag{6}
$$

where α controls the exponential scaling and β the geometry contribution. Setting $\beta = 0$ recovers E-IRFS exactly, allowing direct ablation of the geometry term. For a frame i, the sampling weight is $\begin{array} { r } { r _ { i } = \operatorname* { m a x } _ { c \in i } r _ { c } , } \end{array}$ and its probability of being selected for training is $\begin{array} { r } { p _ { i } = r _ { i } / \sum _ { j } r _ { j } , } \end{array}$ , where the sum runs over all training frames.

The product form in Eq. (6) is deliberate. Geometry scales a weight that rarity has already requested, so a common class that happens to be geometrically difficult is not promoted on its own. Table 1 reports the effect. Bicycle and trailer receive similar sampling weight under E-IRFS, at 2.58 and 2.06, a ratio of 1.25. Under GA-EIRFS the factors become 5.61 and 2.63 and the ratio widens to 2.14, while car moves only from 1.29 to 1.49. Taking the maximum over the classes in a frame, rather than a product, avoids compounding when several minority classes co-occur and preserves the framelevel behaviour of E-IRFS, so that $\beta$ isolates geometry. The cost is one offline pass over the annotated in-box points.

## 4. EXPERIMENTAL SETUP

Datasets. The main evaluation uses nuScenes [1], comprised of 1000 driving scenes recorded with a 32-beam LiDAR at 20 Hz. We use the standard split of 700 training and 150 validation scenes, that is 28,130 and 6,019 annotated keyframes, and the ten official detection classes. Transfer is assessed on KITTI [22], which uses a 64-beam sensor, single sweeps and three classes, and whose Car to Cyclist ratio of about 19.6 to 1 is milder than the 41.5 to 1 ratio on nuScenes.

Models, protocol and metrics. We evaluate CenterPoint [23] and PointPillars [24] in OpenPCDet [25]. The principal nuScenes runs use 20 epochs and the seeds 666 and 1337, that is two repetitions per detector and four paired baseline-to-GA-EIRFS comparisons in total. We set $\alpha = 2 . 0 , t = 0 . 0 1 , \beta = 1 . 0$ and $( \lambda _ { D } , \lambda _ { H } , \lambda _ { S } ) =$ (0.5, 0.3, 0.2), weighting density most because it is measured directly rather than estimated. The geometry sweep uses 12 epochs and $\beta \in \{ 0 , 0 . 5 , 1 . 0 , 2 . 0 \}$ . For KITTI we keep α, t and $\beta$ unchanged, recompute $G _ { c }$ from the KITTI point clouds, use PointPillars with the reference OpenPCDet configuration and no road-plane augmentation, and evaluate the seeds 666, 1337 and 42 (3 repetitions). We report nuScenes mean average precision (mAP) and the nuScenes detection score (NDS): mAP matches predictions to ground truth by center distance, while NDS combines mAP with the true-positive errors in translation, scale, orientation, velocity and attribute [1]; perclass AP; and KITTI moderate-difficulty 3D AP via box IoU [22]. Every comparison is paired: baseline and GA-EIRFS share detector, schedule, augmentation and seed, and differ only in the sampler.

## 5. RESULTS AND DISCUSSION

Overall performance. Table 2 reports the four fully converged nuScenes comparisons. GA-EIRFS increases both mAP and NDS in every run, by 0.4 to 1.3 pp mAP and 0.3 to 0.7 pp NDS. Over the four paired runs the mean gain is +0.91 pp mAP (95% confidence interval [0.27, 1.55], paired t test $p = 0 . 0 2 0 )$ and +0.49 pp NDS $( [ 0 . 1 2 , 0 . 8 6 ] , p = 0 . 0 2 5 )$ . Both intervals exclude zero, although four observations pooled over two architectures bound the effect rather than fix it. The two detectors differ in backbone, feature representation and detection head, so the agreement indicates that the effect is not tied to one architecture. The 12-epoch runs isolate the geometry

Table 1. nuScenes class statistics and repeat factors, ordered by instance count. Eq. (6) uses α = 2, t = 0.01: β = 0 gives E-IRFS and $\beta = 1$ our default.
<table><tr><td colspan="3">Points/</td><td colspan="3">Repeat factor</td></tr><tr><td>Class</td><td>Instances</td><td>obj.</td><td> $G _ { c }$ </td><td> $\beta { = } 0$ </td><td> $\beta { = } 1$ </td></tr><tr><td>Car</td><td>339,949</td><td>98.1</td><td>0.601</td><td>1.29</td><td>1.49</td></tr><tr><td>Pedestrian</td><td>161,928</td><td>11.7</td><td>0.807</td><td>1.37</td><td>1.78</td></tr><tr><td>Barrier</td><td>107,507</td><td>62.6</td><td>0.252</td><td>1.56</td><td>1.74</td></tr><tr><td>Truck</td><td>65,262</td><td>209.8</td><td>0.431</td><td>1.51</td><td>1.80</td></tr><tr><td>Traffic cone</td><td>62,964</td><td>9.2</td><td>0.588</td><td>1.61</td><td>2.14</td></tr><tr><td>Trailer</td><td>19,202</td><td>240.3</td><td>0.337</td><td>2.06</td><td>2.63</td></tr><tr><td>Bus</td><td>12,286</td><td>312.4</td><td>0.342</td><td>2.15</td><td>2.79</td></tr><tr><td>Construction vehicle</td><td>11,050</td><td>103.9</td><td>0.518</td><td>2.34</td><td>3.64</td></tr><tr><td>Motorcycle</td><td>8,846</td><td>41.4</td><td>0.718</td><td>2.51</td><td>4.86</td></tr><tr><td>Bicycle</td><td>8,185</td><td>16.9</td><td>0.823</td><td>2.58</td><td>5.61</td></tr></table>

term from the exponential frequency weighting. E-IRFS reproduces the vanilla mAP exactly under both seeds, 0.5303 and 0.5361, while adding geometry raises it to 0.5398 and 0.5428, so the improvement is attributable to $G _ { c }$ and not to the reweighting it modulates.

Table 2. nuScenes validation results after 20 epochs. V is training without rebalancing and GA is GA-EIRFS. Higher is better, $\Delta$ is in percentage points computed before rounding, and the better value of each metric within a run is in bold. The last row is the mean of the four paired differences, with 95% confidence intervals [+0.27, +1.55] for mAP and [+0.12, +0.86] for NDS.
<table><tr><td colspan="2"></td><td colspan="3">mAP↑</td><td colspan="3">NDS ↑</td></tr><tr><td>Detector</td><td>Seed</td><td>V</td><td>GA</td><td> $\Delta$ </td><td>V</td><td>GA</td><td> $\Delta$ </td></tr><tr><td>CenterPoint</td><td>666</td><td>0.552</td><td>0.563</td><td>+1.1</td><td>0.635</td><td>0.642</td><td>+0.7</td></tr><tr><td>CenterPoint</td><td>1337</td><td>0.554</td><td>0.563</td><td>+0.9</td><td>0.638</td><td>0.641</td><td>+0.3</td></tr><tr><td>PointPillars</td><td>666</td><td>0.385</td><td>0.388</td><td>+0.4</td><td>0.536</td><td>0.538</td><td>+0.3</td></tr><tr><td>PointPillars</td><td>1337</td><td>0.382</td><td>0.395</td><td>+1.3</td><td>0.534</td><td>0.541</td><td>+0.7</td></tr><tr><td>Mean, 4 runs</td><td></td><td></td><td></td><td>+0.9</td><td></td><td></td><td>+0.5</td></tr></table>

Which classes benefit. We divide the ten classes of Table 1 into the $^ { 5 }$ rarest and the 5 most frequent. For CenterPoint at seed 666 the mean AP of the rare group rises from 0.393 to 0.410, a gain of 1.7 pp, whereas the frequent group moves from 0.712 to 0.716. Bicycle improves from 0.306 to 0.359, a gain of 5.3 pp or 17%. Figure 3(a) shows how consistently these effects repeat. Across the four runs, overall mAP and NDS improve in four of four, as do bicycle and motorcycle, while barrier, traffic cone, bus, truck and construction vehicle improve in three of four. Trailer splits two against two, so its behaviour is better described as run-to-run variation than as a systematic loss. Car and pedestrian, which already saturate, never degrade.

Why they benefit. The geometry score alone does not predict the per-class outcome. Over the ten classes the rank correlation between $G _ { c }$ and AP change is weak $( \rho = 0 . 3 2 , p = 0 . 3 7 )$ , whereas the gain in the class sampling weight $\Delta r _ { c } = r _ { c } ( \beta = 1 ) - r _ { c } ( \beta = 0 )$ correlates considerably better $( \rho = 0 . 7 0 , p = 0 . 0 2 5 ) ,$ as plotted in Figure 3(c). This quantity measures a change in class sampling weight, not observed training exposure: frame weights are determined by the maximum class weight and normalized across all training frames, so actual class exposure also depends on co-occurrence. This supports the mechanism the product form was designed for: geometry is useful when it amplifies a frequency-driven need for exposure and has little effect when the frequency term is already small. The sweep in

(c) Sampling weight and AP gain  
![](images/b9d9543715b9afc754fb540e9ece5ae1ef6dd8d1fddde2a326e0dd214642b730.jpg)

![](images/fabe4cab8ccc14933fe237421bd23cc82e379e42ba2e1062a68df4064662fb72.jpg)

![](images/edb2781a5124ba2c39f4cf708ebf90678e1f9a62971002a558d24beb67a42660.jpg)

$$
\Delta r _ { c }
$$

Fig. 3. Quantitative evidence on nuScenes. (a) Per-class AP change at 20 epochs, in percentage points (pp). Bars are the mean of the two seeds, vertical markers are the individual seeds, and classes are ordered by $G _ { c } ,$ lowest at the bottom. (b) Geometry-strength sweep at 12 epochs, CenterPoint, seed 666. (c) AP change against the exposure gain $\Delta r _ { c } = r _ { c } ( \beta = 1 ) - r _ { c } ( \beta = 0 )$ , with the Spearman rank correlation over the ten classes.  
![](images/33514d20d8d7b099f8350dbbd68ab28b2fdd273160efaea8c76e00b1dfb12832.jpg)  
Fig. 4. Top-down view of nuScenes validation sample 3896, in which the two annotated bicycles are far apart and sparsely sampled. Green boxes are ground truth, dashed red boxes are predictions scoring at least 0.3. Vanilla and E-IRFS return no bicycle above the threshold, whereas GA-EIRFS recovers both, at 0.33 and 0.52. The panels differ only in the sampler.

Figure 3(b) gives the most direct evidence that $G _ { c }$ drives the effect: bicycle AP rises monotonically from 0.283 at $\beta \ : = \ : 0$ to 0.354 at $\beta = 2 ,$ motorcycle AP follows the same ordering, and overall mAP stays within 0.34 pp for every nonzero β. A range-corrected G<sub>c</sub> preserves the class ranking $( \rho = 0 . 9 3 9 )$ , so the score is not merely a proxy for the distance distribution of a class. With ten class observations, these p-values are diagnostic rather than confirmatory.

Qualitative behaviour. Figure 4 shows a validation frame with two bicycles at different ranges. Neither vanilla training nor E-IRFS produces a bicycle prediction above the operating threshold, while GA-EIRFS recovers both. The frame was selected as a case in which the frequency-only sampler and the baseline behave identically, the condition the geometry term is meant to change. One frame is not evidence of the size of the effect.

Transfer to KITTI. On KITTI, Cyclist is the rarest class and, independently, has the highest geometry score at $G _ { c } = 0 . 9 3 5$ , giving it the largest repeat factor at 4.05 against 1.92 for Pedestrian and 1.38 for Car, so the pattern that motivates the method on nuScenes reappears under a different sensor. The detection results do not follow as cleanly. Table 3 shows that both minority classes lean positive, but neither improvement holds across the three seeds. Pedestrian gains 0.69 pp on average, while Cyclist loses 0.16 pp because the seed 1337 drop outweighs its two gains, and Car is unchanged. We treat KITTI as a boundary condition rather than a generalisation result. A likely contributor is pool size: Cyclist has 734 instances in 514 frames, the smallest pool in either dataset, so its large repeat factor concentrates exposure on few frames, the same mechanism behind trailer’s run-to-run variation on nuScenes.

Limitations. GA-EIRFS uses hand-set mixture weights and one score per class. It cannot represent within-class variation caused by range, weather or occlusion, and repeating a frame also repeats the frequent objects and background it contains, which is the most likely reason why classes such as trailer respond inconsistently. The nuScenes evidence covers two detectors and two seeds, so the associations constrain the mechanism but do not establish it, and the method was not compared directly against curriculum-based augmentation under a shared protocol.

Table 3. KITTI 3D AP (%) at moderate difficulty, PointPillars, three seeds. Better value of each pair in bold.
<table><tr><td></td><td colspan="2">Car</td><td colspan="2">Pedestrian</td><td colspan="2">Cyclist</td></tr><tr><td>Seed</td><td>V</td><td>GA</td><td>V</td><td>GA</td><td>V</td><td>GA</td></tr><tr><td>666</td><td>75.88</td><td>75.87</td><td>44.19</td><td>45.34</td><td>61.90</td><td>62.29</td></tr><tr><td>1337</td><td>76.28</td><td>75.81</td><td>41.93</td><td>43.74</td><td>62.69</td><td>59.99</td></tr><tr><td>42</td><td>75.34</td><td>75.61</td><td>44.89</td><td>44.00</td><td>61.05</td><td>62.88</td></tr><tr><td>Mean</td><td>75.83</td><td>75.76</td><td>43.67</td><td>44.36</td><td>61.88</td><td>61.72</td></tr><tr><td>SD</td><td>0.47</td><td>0.14</td><td>1.55</td><td>0.86</td><td>0.82</td><td>1.53</td></tr></table>

## 6. CONCLUSION

We presented GA-EIRFS, a rebalancing method that inserts a fixed class-level LiDAR geometry prior into a frequency-based frame repeat factor. On nuScenes, mAP and NDS increased in all four converged detector and seed combinations, bicycle AP increased by 5.3 pp for CenterPoint at seed 666, and it grew monotonically with β. The gains track the realised sampling-weight change rather than the geometry score itself, which supports reading the method as an amplifier of frequency-driven sampling rather than as an independent difficulty sampler. The evidence is limited to two seeds per detector and to one dataset with a severe imbalance, since the KITTI transfer was not consistent across seeds. Learning the mixture weights and conditioning the score on range or on the instance are the next steps.

## ACKNOWLEDGMENT

The research was supported by the Business Finland WiSeCom project (Grant 3630/31/2024), the University of Oulu, the Research Council of Finland 6G Flagship Programme (Grant 346208), and the Business Finland 6GSoft project (Grant 8541/31/2022). Research supported by the NVIDIA Academic Grant Program using nVidia RTX6000 Blackwell GPUs.

## 7. REFERENCES

[1] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11618–11628.

[2] Neehar Peri, Achal Dave, Deva Ramanan, and Shu Kong, “Towards Long-Tailed 3d Detection,” in Proceedings of the 6th Conference on Robot Learning. 2023, vol. 205 of Proceedings ofMachine Learning Research, pp. 1904–1915, PMLR.

[3] Jooyoung Lee, Jaeyoon Lee, and Jongwon Choi, “Nba3d: Neighbor-Based Confidence Adjustment for 3d Rare Object Detection Using LiDAR,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 4, pp. 4508–4516, apr 11 2025.

[4] Agrim Gupta, Piotr Dollar, and Ross Girshick, “Lvis: A dataset for large vocabulary instance segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, p. 5356.

[5] Burhaneddin Yaman, Tanvir Mahmud, and Chun-Hao Liu, “Instance-aware repeat factor sampling for long-tailed object detection,” arXiv preprint arXiv:2305.08069, 2023.

[6] Taufiq Ahmed, Abhishek Kumar, Constantino Alvarez Casado,<sup>´</sup> Anlan Zhang, Tuomo Hanninen, Lauri Lov ¨ en, Miguel Bordallo´ Lopez, and Sasu Tarkoma, “Exponentially weighted instance-´ aware repeat factor sampling for long-tailed object detection model training in unmanned aerial vehicles surveillance scenarios,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2025, Hangzhou, China, October 19-25, 2025. 2025, pp. 11546–11552, IEEE.

[7] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu, “Class-balanced grouping and sampling for point cloud 3d object detection,” arXiv preprint arXiv:1908.09492, 2019.

[8] Daeun Lee and Jinkyu Kim, “Resolving Class Imbalance for LiDAR-based Object Detector by Dynamic Weight Average and Contextual Ground Truth Sampling,” in 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 1 2023, pp. 682–691.

[9] Yechi Ma, Neehar Peri, Achal Dave, Wei Hua, Deva Ramanan, and Shu Kong, “Long-tailed 3d detection via multi-modal fusion,” arXiv preprint arXiv:2312.10986, 2023.

[10] Chun Chieh Chang, Ta Chun Tai, Van Tin Luu, Hong Han Shuai, Wen Huang Cheng, Yung Hui Li, and Ching Chun Huang, Optimizing 3D Object Detection with Data Importance-Based Loss Reweighting, pp. 179–194, Springer Nature Singapore, 2024.

[11] Jiawei Liu, Xingping Dong, Sanyuan Zhao, and Jianbing Shen,

“Generalized Few-Shot 3d Object Detection of LiDAR Point Cloud for Autonomous Driving,” arXiv.org, 2023.

[12] Saptarshi Sinha, Hiroki Ohashi, and Katsuyuki Nakamura, “Class-Difficulty Based Methods for Long-Tailed Visual Recognition,” International Journal of Computer Vision, vol. 130, no. 10, pp. 2517–2531, aug 18 2022.

[13] Yanbiao Ma, Licheng Jiao, Fang Liu, Shuyuan Yang, Xu Liu, and Puhua Chen, “Geometric Prior Guided Feature Representation Learning for Long-Tailed Classification,” International Journal of Computer Vision, vol. 132, no. 7, pp. 2493–2510, feb 5 2024.

[14] Jingmei Ning, Feipeng Da, and Shaoyan Gai, “Density Aware 3d Object Single Stage Detector,” IEEE Sensors Journal, vol. 21, no. 20, pp. 23108–23117, oct 15 2021.

[15] Tingyu Zhang, Jian Wang, and Xinyu Yang, “Boosting 3d Object Detection with Density-Aware Semantics-Augmented Set Abstraction,” Sensors, vol. 23, no. 12, pp. 5757, jun 20 2023.

[16] Wenxin Ma, Jian Chen, Qing Du, and Wei Jia, “Pointdrop: Improving Object Detection from Sparse Point Clouds via Adversarial Data Augmentation,” in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, jan 10 2021, pp. 10004–10009.

[17] Joohyun Lee, Jin-Hee Lee, Jae-Keun Lee, Je-Seok Kim, Soon Kwon, and Sangdong Kim, “Dual Adaptive Data Augmentation for 3d Object Detection,” Information and Communication Technology Convergence, 2023.

[18] Ziyue Zhu, Qiang Meng, Xiao Wang, Ke Wang, Liujiang Yan, and Jian Yang, “Curricular Object Manipulation in LiDARbased Object Detection,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 6 2023, pp. 1125–1135.

[19] Chiyu Max Jiang, Mahyar Najibi, Charles R. Qi, Yin Zhou, and Dragomir Anguelov, “Improving the Intra-class Long-tail in 3d Detection via Rare Example Mining,” European Conference on Computer Vision, 2022.

[20] Di Feng, Xiao Wei, Lars Rosenbaum, Atsuto Maki, and Klaus Dietmayer, “Deep Active Learning for Efficient Training of a LiDAR 3d Object Detector,” in 2019 IEEE Intelligent Vehicles Symposium (IV). IEEE, 6 2019, pp. 667–674.

[21] Dong Liang, Jing-Wei Zhang, Ying-Peng Tang, and Sheng-Jun Huang, “Mus-cdb: Mixed uncertainty sampling with class distribution balancing for active annotation in aerial object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–13, 2023.

[22] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231– 1237, 2013.

[23] Tianwei Yin, Xingyi Zhou, and Philipp Krahenb ¨ uhl, “Center- ¨ based 3d object detection and tracking,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11784–11793.

[24] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom, “Pointpillars: Fast encoders for object detection from point clouds,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 12689–12697.

[25] OpenPCDet Development Team, “Openpcdet: An open-source toolbox for 3d object detection from point clouds,” https: //github.com/open-mmlab/OpenPCDet, 2020.