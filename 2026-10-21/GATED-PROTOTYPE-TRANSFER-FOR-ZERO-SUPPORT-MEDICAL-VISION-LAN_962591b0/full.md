# GATED PROTOTYPE TRANSFER FOR ZERO-SUPPORT MEDICAL VISION LANGUAGE MODEL

Sumaiya Fazal<sup>∗</sup> FAST-NUCES p200401@nu.edu.pk

Laleh Behjat University of Calgary laleh@ucalgary.ca

Rizwan Qureshi   
Massachusetts General Hospital,   
Harvard University   
rizwan.qureshi@mgh.harvard.edu   
Samira Ebrahimi Kahou   
University of Calgary/Mila   
samira.ebrahimikahou@ucalgary.ca

## ABSTRACT

Realistic few-shot adaptation of medical vision-language models (VLMs) can leave low-prevalence classes without any labeled support. The training-free SS-Text+ solver remains operational in this setting, but its zero-support prototypes receive no visual adaptation and fall back to a purely textual class prototype. We introduce Gated Knowledge-Guided Prototype Transfer (GProtoT), a pre-repulsion module that transfers supported-class visual residuals to missing classes using VLM similarity and an ontology-based clinical distance. A validation-free evidence gate scales transfer based on semantic relevance and residual agreement, thereby limiting unsupported extrapolation. In 50 paired realistic one-shot trials on MESSIDOR with FLAIR features, severe and proliferative diabetic retinopathy were absent from every support set. Relative to SS-Text+, gated ProtoT improved missing-class recall by 5.23 points, average class accuracy by 1.37 points, and the missing/supported recall harmonic mean by 1.83 points. It reduced negativetransfer frequency from 46-48% for ungated transfer to 2%, while at the cost of reductions of 1.20 percentage points in supported-class recall and 0.45 percentage points in top-1 accuracy. These results identify that evidence-based gating provides a safety mechanism for the knowledge source. GProtoT is based on SS-Text+ and requires no gradients, validation data, or additional model inference, offering a low-compute path toward accountable adaptation under incomplete clinical supervision. Code is available: https://github.com/sumaiyaafridi/GProtoT

## 1 INTRODUCTION

Medical VLMs provide transferable representations and text-derived class prototypes, making black-box few-shot adaptation possible without updating a large encoder (Shakeri et al., 2024). Yet the conventional balanced K-shot protocol is poorly aligned with disease prevalence. Common findings may appear repeatedly while rare or severe categories receive no labeled examples, and aggregate accuracy can then remain high even when clinically important categories are unrecognized.

Silva-Rodr´ıguez et al. (2025a) introduced SS-Text+, a closed-form text-based linear probe to formalize realistic, imbalanced, validation-free adaptation. For a class with no support, SS-Text+ correctly retains its text prototype rather than fitting an undefined visual evidence. This behavior is robust, but limited-supported classes become task-adapted while a missing class remains purely zero-shot. An alternate method is to borrow a related supported prototype. Our experiments show that borrowing a related supported prototype strategy is unsafe and, due to ungated VLM and ontology-guided transfer, reduced balanced performance and missing-class recall is nearly half of support trials.

We therefore ask whether cross-class transfer can improve a zero-support prototype without turning external knowledge into a source of negative transfer? We propose evidence-based GProtoT, a training-free module applied after SS-Text+ forms class prototypes and before its standard repulsion step. The main formulation transfers visual residuals, preserving the missing class’s text semantics, and the evidence-based gate shrinks the update when semantic relevance or source agreement is weak. Our contributions are:

• a non-recursive residual-transfer rule that employs only originally supported classes

• a VLM, ontology, and hybrid neighbor construction with a validation-free evidence-based gate

• a paired safety evaluation that reports missing-class recall, supported-class recall, their harmonic mean, and negative-transfer frequency, not only overall accuracy

## 2 METHODOLOGY

## 2.1 KNOWLEDGE GUIDED PROTOTYPE TRANSFER

Building on SS-Text+, let $t _ { c } \in \mathbb { R } ^ { D }$ be the text prototype of class $c , K _ { c }$ its support count, and $\tilde { \mu } _ { c }$ its pre-repulsion SS-Text+ prototype. Class-adaptive regularization gives $\tilde { \mu } _ { c } = t _ { c }$ when $K _ { c } = 0$ (Silva-Rodr´ıguez et al., 2025a). Thus, supported classes receive visual adaptation, whereas zerosupport classes remain text-only. We define the supported and missing sets as $\mathcal { S } = \{ i : K _ { i } > 0 \}$ and $\mathcal { M } = \{ c : K _ { c } = 0 \}$ , respectively.

For Knowledge-guided neighbors, for each missing target $c ^ { * } \in { \mathcal { M } } .$ , ProtoT selects sources only from $s ;$ updated missing classes are never reused as sources. Candidate sources are ranked using textspace VLM similarity, $s _ { c ^ { * } , i } ^ { \mathrm { v l m } } = \cos ( t _ { c ^ { * } } , t _ { i } )$ , ontology proximity, $s _ { c ^ { * } , i } ^ { \mathrm { o n t } } = - d ( c ^ { * } , i )$ , or the average of their z-scored values for hybrid ranking. After selecting the top-k neighbors $\mathcal { N } ( c ^ { * } )$ , their scores are converted into normalized contribution weights:

$$
w _ { i } = \frac { \exp ( s _ { i } / T ) } { \sum _ { j \in \mathcal { N } ( c ^ { * } ) } \exp ( s _ { j } / T ) } , \qquad \sum _ { i \in \mathcal { N } ( c ^ { * } ) } w _ { i } = 1 .\tag{1}
$$

equation 1 assigns greater influence to more relevant sources while keeping the total contribution controlled. For a non-positive user temperature, ProtoT uses the validation-free value $T =$ $\operatorname* { m a x } ( \mathrm { s t d } \{ s _ { i } \} , 1 0 ^ { - 3 } )$

For residual transfer and evidence gate, rather than copying complete source prototypes, ProtoT transfers only their task-specific visual corrections. With ${ \hat { x } } = x / \| x \| _ { 2 }$ , the residual of supported class i is $\boldsymbol { r } _ { i } = \hat { \tilde { \mu } } _ { i } - \boldsymbol { \hat { t } _ { i } }$ . The weights in equation 1 then produce the proposed target correction $\bar { r } _ { c ^ { * } }$ while $a _ { c ^ { * } }$ measures whether the selected residuals agree:

$$
\bar { r } _ { c ^ { * } } = \sum _ { i \in \mathcal { N } ( c ^ { * } ) } w _ { i } r _ { i } , \qquad a _ { c ^ { * } } = \frac { \| \bar { r } _ { c ^ { * } } \| _ { 2 } } { \sum _ { i \in \mathcal { N } ( c ^ { * } ) } w _ { i } \| r _ { i } \| _ { 2 } + \epsilon } .\tag{2}
$$

Aligned residuals give $a _ { c ^ { * } } \approx 1$ , whereas conflicting residuals cancel and reduce agreement. Because agreement alone does not establish target relevance, ProtoT also computes $q _ { c ^ { * } } \mathrm { : }$ : VLM relevance is the clipped nearest-neighbor margin above the candidate-row median, ontology relevance is $1 / \operatorname* { m a x } ( d _ { \operatorname* { m i n } } , \bar { 1 } )$ , and hybrid relevance is their geometric mean. The gate and effective strength are $g _ { c ^ { * } } = \mathrm { c l i p } _ { [ 0 , 1 ] } ( a _ { c ^ { * } } q _ { c ^ { * } } )$ and $\beta _ { c ^ { * } } = \beta g _ { c ^ { * } }$

The missing prototype is updated by adding the gated residual to its original text direction:

$$
p _ { c ^ { * } } = \| t _ { c ^ { * } } \| _ { 2 } \frac { \hat { t } _ { c ^ { * } } + \beta _ { c ^ { * } } \bar { r } _ { c ^ { * } } } { \| \hat { t } _ { c ^ { * } } + \beta _ { c ^ { * } } \bar { r } _ { c ^ { * } } \| _ { 2 } } .\tag{3}
$$

This preserves the target’s text semantics while incorporating only evidence-supported visual information. If evidence is weak, $g _ { c ^ { * } } = 0$ and equation 3 recovers the original text prototype. ProtoT leaves supported prototypes unchanged, and the original SS-Text+ repulsion and inference normalization proceed as before. It requires no gradients, validation data, extra labels, or additional VLM inference, preserving the efficiency of training-free black-box adaptation (Schwartz et al., 2020).

Table 1: Fifty-seed realistic one-shot MESSIDOR performance. Parentheses report paired changes from SS-Text+ in percentage points. NTR and severe NTR denote the percentages of seeds with ∆MCR < 0 and ∆MCR < −5 points, respectively
<table><tr><td>Method</td><td>Top-1</td><td>ACA</td><td>MCR</td><td>SCR</td><td>H</td><td>NTR</td><td>Sev.</td></tr><tr><td>SS-Text+</td><td>57.45</td><td>54.64</td><td>54.90</td><td>54.46</td><td>52.87</td><td></td><td>一</td></tr><tr><td>ProtoT-VLM</td><td>54.83</td><td>51.91</td><td>52.73</td><td>51.37</td><td>50.25</td><td>46</td><td>38</td></tr><tr><td>ProtoT-Ont</td><td>(−2.62) 54.92</td><td>(-2.72) 51.90</td><td>(−2.16) 52.47</td><td>(-3.10) 51.52</td><td>(−2.63) 50.20</td><td>48</td><td>34</td></tr><tr><td></td><td>(−2.53)</td><td>(-2.74)</td><td>(−2.43)</td><td>(-2.94)</td><td>(-2.67)</td><td></td><td></td></tr><tr><td>Gated-ProtoT</td><td>57.00 (−0.45)</td><td>56.01 (+1.37)</td><td>60.12 (+5.23)</td><td>53.27 (-1.20)</td><td>54.71 (+1.83)</td><td>2</td><td>2</td></tr></table>

## 3 EXPERIMENTAL SETUP

We build upon the official SS-Text+ implementation and utilize its cached FLAIR representations for grading diabetic retinopathy in MESSIDOR (Decenciere et al. \` , 2014; Silva-Rodr´ıguez et al., 2025b). The classification includes five ordinal categories: no retinopathy, mild, moderate, severe, and proliferative diabetic retinopathy. Following the realistic K = 1 protocol, the total dataset for support consists of five images sampled according to the training prevalence. Across seeds 0–49, each support set contained counts of [3, 1, 1, 0, 0], while the identities of support images varied; severe and proliferative disease categories were consistently absent. Prevalence-weighted sampling at k = 1 yields the same class count pattern in every seed, so seed variance measures only sensitivity to support selection. Because the identity of the absent classes is not varied, missing-class results should be interpreted as specific to this particular zero-support configuration.

All techniques utilize the same support sets and initial prototypes. ProtoT employs residual transfer with parameters set to $k = 3 , \beta = 1$ , automatic temperature control, and the original all-class repulsion. Both ProtoT-VLM and ProtoT-Ont operate without gating. GProtoT incorporates hybrid ranking along with the proposed gating mechanism. Each experiment is conducted alongside SS-Text+ using the identical support set. The ontology is a manually specified ordinal class graph based on the established diabetic-retinopathy severity ordering (Wilkinson et al., 2003).

Because typical instances predominate in the test set, top-1 accuracy is insufficient. In light of the focus on class-aware medical validation (Maier-Hein et al., 2024), we therefore present average class accuracy (ACA), mean recall over missing classes (MCR), mean recall over supported classes (SCR), and harmonic mean (H).

## 4 RESULTS

Table 1 shows that knowledge transfer alone does not guarantee safe transfer. Relative to SS-Text+, both ungated variants yield lower mean performance across all reported metrics. ProtoT-VLM reduces ACA, MCR, SCR, H, and top-1 accuracy by 2.72, 2.16, 3.10, 2.63, and 2.62 percentage points, respectively and the corresponding reductions for ProtoT-Ont are 2.74, 2.43, 2.94, 2.67, and 2.53 points. Negative transfer is frequent where ProtoT-VLM and ProtoT-Ont reduce MCR relative to the paired baseline in 46% and 48% of seeds, respectively, with reductions exceeding five points in 38% and 34% of seeds. For ProtoT-VLM, the mean MCR change is negative but statistically inconclusive, mean change = −2.16 points, 95% CI [−4.73, 0.32], Holm-adjusted p = .103, because the interval includes zero. In contrast, ProtoT-Ont produces a statistically significant, although borderline, reduction in MCR, mean change = −2.43 points, 95% CI [−4.68, −0.21], Holm-adjusted $p = . 0 4 9 .$

Gated-ProtoT reverses the missing-class pattern. It raises MCR to 60.12%, corresponding to a mean paired improvement of 5.23 percentage points, 95% CI [3.84, 6.69]. It also increases ACA to 56.01%, +1.37 points, 95% CI [0.78, 1.99] and H to 54.71%, +1.83 points, 95% CI [1.06, 2.64]. MCR improves in 36 of the 50 seeds and remains unchanged in 13, leaving only one seed with negative transfer, NTR = 2%. These gains are accompanied by modest reductions of 1.20 points in SCR, 95% CI [−1.52, −0.88] and 0.45 points in top-1 accuracy, 95% CI [−0.71, −0.19]. Thus, the combined hybrid-ranking and evidence-gating variant substantially improves recognition of zerosupport classes.

Table 2: Skin histology results with CONCH (Lu et al., 2024) over 50 paired realistic nominal K = 1 seeds. Performance values are mean percentages; ∆ reports paired changes from SS-Text+ in percentage points. NTR and Sev. denote the percentages of seeds with $\Delta \mathrm { M C R } < 0$ and ∆MCR < −5 points, respectively.
<table><tr><td>Method</td><td>Top-1</td><td>ACA</td><td>MCR</td><td>SCR</td><td>H</td><td>NTR</td><td>Sev.</td></tr><tr><td>SS-Text+</td><td>77.577</td><td>57.166</td><td>12.276</td><td>84.100</td><td>21.344</td><td>一</td><td>1</td></tr><tr><td>Gated-ProtoT</td><td>79.444</td><td>63.193</td><td>28.709</td><td>83.884</td><td>42.729</td><td>0</td><td>0</td></tr><tr><td>∆ (points)</td><td>+1.867</td><td>+6.027</td><td>+16.433</td><td>-0.216</td><td>+21.385</td><td>一</td><td>一</td></tr></table>

Moderate DR is the highest-weighted source for both missing classes across all 50 seeds. The mean gate is $0 . 5 8 2 { \pm } 0 . 0 1 0$ for severe DR and 0.214±0.003 for proliferative DR. The weaker proliferative update reflects its larger nearest-supported ontology distance of two and lower VLM relevance of 0.102, compared with distance one and relevance 0.381 for severe DR. Nevertheless, gate magnitude controls the strength of the prototype update rather than directly determining final recall, which also depends on the baseline prototype, class separability, and resulting decision boundaries. Under Gated-ProtoT, per-class recall is 46.53% for severe DR and 73.71% for proliferative DR, compared with 37.47% and 68.00% under ProtoT-VLM and 36.93% and 68.00% under ProtoT-Ont.

Table 2 extends the evaluation to Skin histology with CONCH (Lu et al., 2024) under 50 paired realistic nominal K = 1 seeds. We retain the MESSIDOR transfer settings: hybrid ranking, k = 3, β = 1, automatic temperature, residual transfer, and all-class repulsion. Every seed contains zerosupport classes. Relative to SS-Text+, Gated-ProtoT raises MCR from 12.276% to 28.709%, a gain of 16.433 percentage points, and improves ACA and H by 6.027 and 21.385 points, respectively. Top-1 accuracy increases by 1.867 points, while SCR decreases by 0.216 points. No seed has lower MCR than its paired baseline $( \mathrm { N T R } = 0 / 5 0 = 0 \% )$ , and severe negative transfer is also absent. Both Skin and MESSIDOR show improved missing-class recall, although their top-1 trade-offs differ: Skin gains 1.867 points, whereas MESSIDOR loses 0.45 points. These observations extend the combined variant to a non-ordinal histology label space and a second encoder.

## 5 CONCLUSION

We introduced knowledge-guided prototype transfer under a realistic one-shot protocol in which prevalence-weighted support allocation leaves the classes with no samples entirely unsupported. Knowledge proximity alone did not ensure safe transfer: both VLM and ontology ungated variants lowered every mean performance metric and produced negative transfer in 46 to 48% of the supportsampling seeds. Evidence gating substantially reduced this failure. Gated-ProtoT improved mean missing-class recall by 5.23 percentage points and ACA by 1.37 points, while reducing negative transfer to one of 50 seeds. These improvements were accompanied by modest observed reductions of 1.20 points in supported-class recall and 0.45 points in top-1 accuracy.

The results suggest that calibrating transfer strength can reduce the risks of knowledge-guided transfer. However, because GProtoT combines hybrid ranking with evidence gating, the present comparison does not fully isolate the gate’s contribution. Skin histology with CONCH (Lu et al., 2024) further shows a 16.43-point MCR gain and a 1.87-point top-1 gain, with no negative MCR changes in 50 paired seeds. These findings extend the evaluation beyond retinal grading, while broader class-rotated validation and isolating the gate’s contribution remain necessary.

## REFERENCES

Etienne Decenciere et al. Feedback on a publicly distributed image database: The messidor database.\` Image Analysis & Stereology, 33(3):231–234, 2014.

Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. A visual-language foundation model for computational pathology. Nature medicine, 30(3):863–874, 2024.

Lena Maier-Hein et al. Metrics reloaded: Recommendations for image analysis validation. Nature Methods, 21:195–212, 2024.

Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green ai. Communications of the ACM, 63(12):54–63, 2020.

Fereshteh Shakeri et al. Few-shot adaptation of medical vision-language models. In Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 553–563, 2024.

Julio Silva-Rodr´ıguez et al. Few-shot, now for real: Medical vlms adaptation without balanced sets or validation. In Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 237–247, 2025a. doi: 10.1007/978-3-032-04981-0 23.

Julio Silva-Rodr´ıguez et al. A foundation language-image model of the retina (flair): Encoding expert knowledge in text supervision. Medical Image Analysis, 99:103357, 2025b.

Charles P Wilkinson, Frederick L Ferris III, Ronald E Klein, Paul P Lee, Carl David Agardh, Matthew Davis, Diana Dills, Anselm Kampik, Rangasamy Pararajasegaram, Juan T Verdaguer, et al. Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. Ophthalmology, 110(9):1677–1682, 2003.