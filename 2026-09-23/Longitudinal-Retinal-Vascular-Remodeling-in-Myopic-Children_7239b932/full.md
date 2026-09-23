# Longitudinal Retinal Vascular Remodeling in Myopic Children Treated with Orthokeratology or Defocus Lenses: A Two-Year Comparative Study

Zhihao Zhao <sup>b,</sup> <sup>†</sup>, Yinzheng Zhao <sup>b,</sup> <sup>†</sup>, Jie Zhang <sup>a,</sup> <sup>†</sup>, Huiqin Jiang <sup>d</sup>, Yanyu Shangguan <sup>a</sup>, Yanfei Sun <sup>d</sup>, Li Chen <sup>c</sup>, Yanlong Bi <sup>a</sup>, M.Ali Nasseri <sup>b,</sup> <sup>e,</sup> <sup>f,</sup> <sup>\*</sup>, Bing Li <sup>a,</sup> <sup>\*</sup>

<sup>a</sup> Department of ophthalmology, Tongji Hospital, School of Medicine, Tongji University, Shanghai, 200065, China

<sup>b</sup> TUM University Hospital Rechts der Isar, Technische Universität München, 81675 Munich, Germany

<sup>c</sup> Department of ophthalmology, Yangpu Hospital, School of Medicine, Tongji University, Shanghai, 200090, China

<sup>d</sup> Department of ophthalmology, Shanghai Demu Youmei Ophthalmology Outpatient Department Co., Ltd. Shanghai, 200001, China

<sup>e</sup> Zhongshan Ophthalmic Center, Sun Yat-sen University, Guangzhou, 510623, China

<sup>f</sup> Department of Biomedical Engineering, University of Alberta, T6G 2R3 Alberta, Canada

†These authors contributed equally.

<sup>\*</sup>Corresponding author(s): Prof. M.Ali Nasser

## Abstract

Purposes: To characterize longitudinal retinal vascular changes in myopic children treated with orthokeratology (OK) or multifocal defocus lenses (Defocus) and to examine their association with axial elongation.

Methods: In this retrospective cohort study, 43 myopic children underwent comprehensive clinical examination and fundus photography at baseline, 12 months, and 24 months. Axial length (AL) and spherical equivalent refraction (SER) were recorded at baseline, 6, 12, and 24 months. An automated segmentation model extracted vascular parameters, main vessel angle (MA), branching angle (BA), bifurcation edge angle (BEA), crossover point (COP), and terminal vessel count (TVC). Repeatedmeasures ANOVA assessed temporal changes. Pearson or Spearman correlations evaluated associations between AL and vascular metrics.

Results: Over 24 months, the OK group exhibited significantly slower axial elongation than the Defocus group (0.214 mm and 0.522 mm, p < 0.01). In the OK group, MA and BA decreased modestly, BEA in arteries declined gradually, but COP and TVC remained relatively stable. The Defocus group demonstrated more pronounced decreases in MA and BA, an increase in BEA, and significant reductions in COP and TVC (p < 0.05). Correlation analysis revealed stronger associations between AL and vascular parameters, especially COP and TVC, in the Defocus group at all time points, whereas only BA and BEA correlated with AL in the OK group.

Conclusions: OK lenses mitigate axial elongation and induce milder retinal vascular remodeling compared to Defocus lenses. Distinct temporal patterns of vascular metrics changes were observed between the two interventions, and correlate differentially with axial growth.

Keywords: Myopia control; Retinal vascular remodeling; Orthokeratology; Multifocal defocus lenses; Axial elongation

## 1. Introduction

Myopia has emerged as a global public health concern, with increasing prevalence among children and adolescents, particularly in East Asia.<sup>1-3</sup> Excessive axial elongation, referring to axial growth beyond the expected age-related physiological range and commonly accompanying progressive or high myopia, is a major risk factor for sight threatening complications,<sup>4,</sup> <sup>5</sup> such as myopic maculopathy,<sup>6</sup> choroidal neovascularization<sup>7</sup> and retinal detachment.<sup>8</sup> Several myopia control strategies have been developed, among which Orthokeratology (OK) lenses and multifocal defocus lenses have demonstrated clinical efficacy in slowing axial elongation.<sup>9-12</sup> However, the underlying mechanisms of these interventions remain incompletely understood.

Recent advances in retinal imaging and image analysis have made it possible to noninvasively evaluate structural and microvascular changes in the posterior pole, including the retinal vasculature.<sup>13-15</sup> Retinal vascular morphology, such as vessel angles, branching complexity, and vascular density, is known to reflect underlying ocular and systemic physiological changes.<sup>16,</sup> <sup>17</sup> In the context of myopia progression, alterations in vascular geometry may serve as a potential biomarker for disease severity and therapeutic response. Advanced imaging techniques, particularly optical coherence tomography angiography, have further demonstrated that myopia-related microvascular alterations may occur in different retinal and choroidal vascular layers. Collectively, these findings suggest that retinal vascular characteristics may provide complementary information beyond conventional refractive and biometric measurements. Some researchers showed that tensile force might influence changes in the retinal vascular system during axial elongation in high myopia, leading to atrophy of the vascular layer, narrowing of vessels, altered vessel angles, and reduced vascular density.<sup>18,</sup> <sup>19</sup> Lim et al.<sup>20</sup> found that higher myopic refractive error and longer axial length were associated with a sharper bifurcation angle in arteries among middle aged and older Malaysians, along with increased branching coefficients in both arteries and veins. However, in a small cohort of pseudophakic individuals, Patton et al.<sup>21</sup> did not observe any association between AL and vessel bifurcation angles or connectivity indices.

Most existing studies on myopia control have focused predominantly on refractive and axial outcomes, with limited investigation into the underlying structural or vascular changes in the retina.<sup>22,</sup> <sup>23</sup> Studies of myopia-control interventions have mainly focused on changes in spherical equivalent refraction and axial length, while the longitudinal patterns of retinal vascular change during treatment remain insufficiently characterized. In particular, it remains unclear whether children receiving different myopia-control interventions exhibit distinct patterns of retinal vascular remodeling over time and whether these patterns are associated with axial elongation. While retinal vascular alterations have been observed in myopic eyes, the temporal dynamics of these changes, especially in response to specific interventions, remain poorly understood. Furthermore, few studies have directly compared the impact of different myopia control strategies on the retinal microvasculature. Understanding whethe these treatment modalities induce distinct patterns of vascular remodeling may offer insights into their mechanisms of action and provide potential biomarkers for treatment efficacy. This knowledge gap underscores the need for longitudinal, image-based vascular analyses to complement traditional biometric measures in myopia research.

In this study, we used an automated retinal image analysis pipeline to quantify vascular features from fundus photographs of myopic children receiving either orthokeratology or multifocal Defocus lens treatment over two years. The evaluated parameters included main vessel angle, branching angle, branching edge angle, crossover point count, and terminal vessel count. We hypothesized that the two treatment groups would exhibit distinct longitudinal patterns of retinal vascular change and that changes in these vascular parameters would be associated with axial elongation. By testing this hypothesis, the study aimed to identify potentially informative vascular parameters and provide longitudinal evidence to support the design of future prospective studies investigating retinal vascular remodeling during myopia control.

## 2. Methods

## 2.1 Study Design and Subjects

This was a retrospective, observational cohort study conducted at the affiliated Yangpu District Central Hospital of Tongji University. We reviewed the clinical records of a consecutive series of pediatric patients who initiated either OK lens or Defocus lens treatment between January 2022 and January 2024. This consecutive sampling resulted in a total of 43 subjects who met the eligibility criteria. Treatment selection was determined before study inclusion based on clinical assessment and discussions with the children and their guardians, taking into consideration refractive status, ocular characteristics, lifestyle requirements, treatment suitability, and family preference. The study included 25 patients in the OK lens group (Eyebright Medical Technology, Beijing, Co., Ltd.) and 18 patients in the Defocus lens group (Eyepol Optical Technology, Xiamen, Co., Ltd.). Inclusion criteria were: (1) a clinical diagnosis of myopia (SER between -0.75 D and -6.00 D); (2) availability of complete clinical records and high-quality fundus photographs at baseline, 12 months, and 24 months; (3) age between 8 and 16 years at the start of treatment. Patients were excluded if they had: (1) astigmatism > 1.50 D; (2) any ocular pathology other than myopia; (3) a history of prior ophthalmic surgery; or (4) fundus images of insufficient quality for reliable automated analysis (e.g., due to poor focus, media opacity, or artifacts).

This study adhered to the principles of the Declaration of Helsinki and was approved by the Ethics Committee of Yangpu District Central Hospital (Approval Number: LL-2025-LW-003). Given the retrospective nature of the study using anonymized data, the requirement for written informed consent was waived by the committee.

## 2.2 Data Collection and Image Analysis

Clinical data included age, sex, age at initial myopia diagnosis, SER, AL, and corneal curvature. AL and SER were recorded at baseline, 6 months, 12 months, 18 months and 24 months. Fundus photographs were acquired using Non-mydriatic Fundus Camera (Topcon, Japan) at baseline, 1 year, and 2 years. Images were captured at a $4 5 ^ { \circ }$ field of view, centered on the macula.

Retinal vascular parameters, including vessel angle, fractal dimension, vessel diameter, and vascular coefficient, were extracted from fundus images using a validated model based on segmentation and quantization. Axial elongation and refraction progression were calculated as the difference between baseline and follow-up values at each time point.

Retinal vascular parameters were extracted using a validated, automated analysis pipeline based on a deep learning segmentation model. Firstly, raw fundus images underwent preprocessing, including brightness normalization and contrast enhancement to standardize image quality. Secondly, a U-Net based deep learning architecture was employed to segment the vascular network and generate a binary vessel map. The system automatically differentiated arteries from veins. Finally, the segmented vessel map was skeletonized to extract the network's topology and then calculate the quantitative parameters. When the loss function of our data classification model converged to 0.09, the model accuracy reached 94.19%. To ensure data quality, all segmented images were visually inspected by a trained researcher, and images with significant segmentation errors were excluded. The core research focus is on comparing longitudinal trends and relative differences in vascular parameters between groups, which can reduce image magnification differences caused by axial length growth or corneal curvature changes due to OK lens use, thereby improving the accuracy of absolute measurements.

## 2.3 Statistical Analysis

In this study, normality tests were first conducted. For continuous variables following a normal distribution, they are expressed as mean ± standard deviation $( \mathsf { X } \pm \mathsf { S } )$ . Effect sizes were reported alongside P values where applicable. Partial eta-squared was used to describe the magnitude of effects in repeated-measures ANOVA. Correlation analyses were reported using Pearson’s correlation coefficient or Spearman’s rank correlation coefficient, together with the corresponding P values. Adjusted odds ratios and 95% confidence intervals were reported for the binary logistic regression analysis. If the data did not satisfy normality, the Friedman rank-sum test was used as an alternative. Comparisons between different treatment groups were performed using the Mann–Whitney U test. Categorical variables are presented as frequencies and percentages, with intergroup comparisons conducted using $\mathsf { X } ^ { 2 }$ tests or Fisher's exact test. The correlation between vascular parameters and axial length, along with refractive changes, was analyzed using Pearson or Spearman correlation analysis. A binary logistic regression model was further used to evaluate the independent association between changes in vascular parameters and axial elongation, while controlling for confounding factors such as age, sex, baseline axial length, and treatment methods. The logistic regression analysis was performed to identify factors associated with greater cumulative axial elongation over the 24-month follow-up. The dependent variable was coded as 0 for cumulative axial elongation $\leqslant 0 . 3$ mm and 1 for cumulative axial elongation >0.3 mm. The cutoff was selected as a clinically interpretable threshold with reference to previously reported axial-growth ranges in children and its use in studies evaluating responses to myopia-control interventions. This stratification enabled the identification of demographic and retinal vascular factors associated with more pronounced cumulative axial growth. Differences were considered statistically significant when P values $< 0 . 0 5$ . All data analyses were conducted using SPSS (version 27.0) and Python (version 3.5).

## 3. Results

A total of 43 children with myopia were included, with 25 patients in the OK lenses group and 18 patients in the Defocus lenses group (Table 1). The gender distribution was comparable between the two groups. However, patients in the OK lens group were significantly older $( 1 3 . 2 8 \pm 1 . 8 1 \mathrm { y e a r s } )$ than those in the Defocus lens group $( 1 1 . 7 2 \pm 1 . 9 9 \mathrm { ~ y e a r s }$ $\mathsf { P } < 0 . 0 0 1 $ . Similarly, the age at first myopia diagnosis was higher in the OK lens group $( 1 0 . 4 0 \pm 1 . 6 1 \mathrm { y e a r s } )$ compared to the Defocus group (9.33 $\pm \ : 1 . 6 4 $ years, $\mathsf { P } < 0 . 0 0 1 )$ . The OK lens group had a higher mean baseline age than the Defocus lens group. As age is closely associated with ocular growth and retinal vascular development in children, this baseline difference was considered when interpreting the longitudinal axial-length and vascular trajectories. Age was subsequently included as a covariate in the multivariable regression analysis. There were no statistically significant differences between the two groups in terms of baseline cycloplegic spherical equivalent refraction or axial length $( \mathsf { P } \ : = \ : 0 . 6 2 3 ;$ $\mathsf { P } = 0 . 8 3 4 )$ . Longitudinal measurements of axial length at 6, 12, and 24 months after treatment initiation also showed no significant differences between groups.

## 3.1 Changes in Axial Length between Different Groups

The longitudinal changes in AL between two groups were evaluated at multiple follow-up time points. As shown in Figure 2A, both groups exhibited progressive axial elongation over the 24-month period. However, the rate of axial length increase was significantly lower in the OK lens group compared to the Defocus lens group. At 6, 12, and 24 months, the mean AL increases from baseline in the OK group were 0.093 mm, 0.132 mm, and 0.214 mm, respectively, whereas the corresponding increases in the Defocus group were 0.066 mm, 0.203 mm, and 0.522 mm.

Additionally, to better understand the short-term dynamics of axial elongation, inter-visit AL differences were plotted (Figure 2B). Between 6 and 12-month follow-ups, the Defocus group showed a continued and relatively stable increase in AL (0.138 mm), while the OK lens group exhibited a lower increment (0.039 mm). During the second year (months 12 to 24), the Defocus lens group again showed a notable increase in AL (0.204 mm), in contrast to the OK group, which maintained minimal elongation (0.089 mm).

## 3.2 Longitudinal Changes in Major Vascular Parameters Across Different Groups

Repeated-measures ANOVA revealed statistically significant interaction effects between group and time for main vascular parameters, indicating distinct longitudinal trends between the OK lens and Defocus lens groups (P < 0.001).

The temporal trends of vascular parameters were analyzed separately for arterioles and venules in both treatment groups (Figure 3). In the OK lens group, the arterial MA showed a slight and relatively stable decrease over the 24-month follow-up period (Figure 3A). In contrast, the Defocus lens group exhibited a more pronounced reduction in arterial MA, particularly after the first year. Venular MA remained relatively stable in both groups.

The branching angle (BA) of retinal vessels demonstrated a decreasing trend in both treatment groups (Figure 3B). In the OK lens group, both arteriolar and venular BA values showed a steady and progressive decline throughout the observation period. In the Defocus lens group, arteriolar BA also decreased over time, with a mild reversal trend between 12 and 24 months. Interestingly, venular BA remained relatively stable, with a slight decrease observed in the second year.

The bifurcation edge angle (BEA) exhibited distinct temporal patterns across groups and vessel types (Figure 3C). In the OK lens group, venular BEA remained relatively stable, while arteriolar BEA showed a progressive and marked decrease throughout the follow-up. In contrast, the Defocus lens group demonstrated an obvious decreasing trend in arteriolar and increasing trend in venular BEA values, particularly between 12 and 24 months.

As shown in Figure 3D, the mean number of crossover points (COP) exhibited distinct temporal patterns between the OK lens and Defocus lens groups. In both arterial and venous measurements, the previous group showed relatively stable or mildly increasing values over the 24-month follow-up period. The second group exhibited an initial decline in both arterial and venous crossover point counts at 12 months, which then stabilized through 24 months.

Figure 3E demonstrates the longitudinal changes in terminal vascular counts (TVC) across both treatment groups. At all measured time points, a gradual reduction in terminal vessel numbers was observed in both groups. However, the decline was more pronounced in the Defocus lens group, particularly at 12 months, with minimal recovery observed thereafter.

## 3.3 Correlation Between Axial Length and Vascular Parameters

Correlation analyses between AL and retinal vascular parameters at each follow-up time point revealed group-specific patterns (Table 2). In the Defocus lens group, AL was significantly correlated with most arterial and venous parameters across all time points, particularly with COP and TVC (P <

0.05). Significant correlations were also observed between AL and MA at all time points (P < 0.05), while other parameters such as BA, BEA, and BEC showed inconsistent associations.

The OK lens group exhibited fewer significant correlations (Table 3). For arterial parameters, only the BA and BEA consistently showed correlations with AL (P < 0.05), while COP and TVC were only partially correlated at baseline and early follow-up. For venous parameters, correlations with AL were generally weak and less consistent in the OK lens group.

A binary logistic regression analysis was performed to evaluate the independent association between changes in retinal vascular parameters and axial elongation, controlling for confounding factors such as age, gender, and baseline axial length (Table 4). The dependent variable was coded as 0 for AL changes ≤0.3 mm and 1 for changes >0.3 mm. This cutoff was used as a pragmatic analytical threshold and was not considered a universally established clinical definition of rapid axial elongation. Age itself is a key factor influencing the progression of myopia. To correct for this confounding effect, we adjusted for age as a covariate in the regression analysis. Independent variables included age, gender, and various morphological parameters for both arteries and veins. The regression analysis revealed that arterial BA were statistically significant predictors (P < 0.05). After controlling for all other variables, the odds of significant myopia progression increased by 2.3 times for every one-year increment in age. The OR value less than 1 for arterial BA indicates a negative correlation between arterial BA and significant axial elongation, suggesting that a larger arterial BA may have a protective effect against excessive axial elongation.

Collectively, these findings demonstrate distinct and quantifiable longitudinal patterns of retinal vascular remodeling across the two treatment groups. Although the clinical significance and applicable thresholds of these vascular changes require further validation, the identified parameters provide an empirical basis for subsequent studies evaluating their potential value in monitoring axial growth and myopia-control outcomes.

## 4. Discussion

Retinal fundus imaging offers a noninvasive and reproducible approach combined with quantitative vascular analysis to assess microvascular alterations in vivo. The retinal vasculature shares embryological, anatomical, and physiological characteristics with the cerebral and systemic microcirculation, making it an accessible surrogate for evaluating vascular health.<sup>24,</sup> <sup>25</sup> Quantitative analysis of retinal vascular features, including vessel diameter,<sup>26</sup> fractal dimension,<sup>27</sup> and vascular density, has been increasingly utilized to detect early vascular changes in a variety of ocular and systemic diseases, such as diabetic retinopathy, hypertensive retinopathy, glaucoma, and age-related macular degeneration.<sup>28-30</sup>

Our findings reveal distinct patterns of retinal vascular remodeling associated with different myopia control strategies. These vascular changes may reflect underlying differences in the biomechanical and physiological responses to axial elongation modulated by optical interventions. By quantifying these changes at multiple time points, fundus-based vascular analysis not only enables the identification of early biomarkers of disease progression but also allows for monitoring treatment effects and comparing different therapeutic modalities.

Compared with the defocus lens group, the OK lens group exhibited significantly less axial elongation over 24 months. This finding is broadly consistent with previous studies reporting slower axial elongation among children treated with OK lenses.<sup>10</sup> Notably, vascular structural changes, including reductions in BA, TVC and COP, were generally milder in the OK lens group, suggesting that axial elongation may play a mechanistic role in the stability of peripheral retinal vessels and their nearly normal branching morphology.<sup>31</sup> The reduced mechanical stretch on the retina and sclera likely minimizes tractional forces on retinal vessels, thus preserving their geometric configurations. Furthermore, the relative stability of vascular parameters may reflect preserved retinal perfusion and autoregulatory function under slower myopic progression. The coordinated longitudinal patterns observed in axial length and retinal vascular morphology indicate a close association between ocular growth and vascular remodeling during myopia-control treatment. Changes in retinal geometry, tissue tension, perfusion demand, and vascular autoregulation may provide biologically plausible contexts for understanding these longitudinal associations. The decline in vascular COP and TVC signifies peripheral capillary rarefaction, a reduction that was notably more prominent in the Defocus lens group. This pattern is frequently associated with progressive myopia, likely attributed to mechanisms such as tissue remodeling, diminished pro-angiogenic cues, or localized hypoxic conditions resulting from retinal stretching or compromised metabolic support in the periphery. The diminished topological complexity of the retinal vascular network may affect local perfusion and oxygenation.

Venular BA in the Defocus group showed an obvious decrease, but arterial BA increased, which may represent compensatory dilation or vascular adaptation to maintain retinal perfusion in the setting of progressive structural remodeling. However, the OK group showed more gradual, controlled changes, consistent with attenuated retinal distortion. This may reflect more localized and controlled remodeling in peripheral vascular system under OK lens treatment. A study involving 493 patients indicated that the higher the degree of myopia, the smaller the artery branching angle. Different refractive errors in myopia are associated with different vascular bifurcation patterns. Conversely, the Defocus lens group exhibited increased BEA in venous branches and decreased BEA in arterial branches, suggesting peripheral vessel dilation or disruption of normal branching geometry. The OK lens group showed smaller changes. This divergence may reflect differing impacts of the two treatments on peripheral retinal biomechanics or oxygen demand. Li et al.<sup>32</sup> also indicates a reduced density of retinal micro vessels in both the superficial and deep vascular plexuses among individuals with high myopia. This conclusion aligns with our findings.

Despite the valuable insights provided by our findings, several limitations of this study should be acknowledged. First, the study failed to control for other potential confounding factors, such as circadian variations in axial length, systemic vascular health, or lifestyle habits, all of which could influence retinal vascular morphology. Second, the sample size was small, and the retrospective design analyzing consecutive cases within a specific time period limits the generalizability of the findings to a broader population and prevents establishing causality. Although our analysis provides valuable preliminary longitudinal evidence regarding retinal vascular development in myopic children undergoing treatment, future prospective studies incorporating age-matched untreated, non-myopic, and alternative treatment control groups are warranted. Our results should be regarded as exploratory associations rather than definitive treatment effects. Given these limitations, future research directions are clear. We will design large scale, age matched prospective randomized controlled trials combined with more advanced imaging techniques such as optical coherence tomography angiography (OCT-A) to obtain quantitative information on retinal vascular networks at different layers, thereby deepening our understanding of vascular remodeling mechanisms.

## 5. Conclusion

In summary, this study provides preliminary evidence indicating that OK lens treatment is associated with slower axial elongation and less pronounced retinal vascular remodeling over a twoyear follow-up period compared to defocus lens treatment. These observed associations lay a foundation for the design of larger, more rigorous prospective clinical trials and help identify vascular parameters that merit further investigation. Future large scale, age-matched prospective studies utilizing proper control groups and multimodal imaging modalities are required to validate these findings and elucidate their potential clinical implications.

## Abbreviations

OK Orthokeratology   
AL Axial length   
SER Spherical Equivalent Refraction   
MA Main vessel angle   
BA Branching angle   
BEA Bifurcation edge angle   
BEC Bifurcation edge coefficient   
COP Crossover point   
TVC Terminal vessel count

## Tables

Table 1. Clinical Characteristics and Baseline Parameters
<table><tr><td rowspan="2">Characteristics</td><td colspan="2">Groups</td><td rowspan="2">P Value</td></tr><tr><td>OK Lens (n=25)</td><td>Defocus Lens (n=18)</td></tr><tr><td>Gender</td><td></td><td></td><td>0.435</td></tr><tr><td>Male</td><td>16</td><td>10</td><td></td></tr><tr><td>Female</td><td>9</td><td>8</td><td></td></tr><tr><td>Age</td><td>13.28±1.81</td><td>11.72±1.99</td><td>&lt; 0.001</td></tr><tr><td>First Diagnosis Age</td><td>10.40±1.61</td><td>9.33±1.64</td><td>&lt; 0.001</td></tr><tr><td>Baseline Values</td><td></td><td></td><td></td></tr><tr><td>Cycloplegic Refraction</td><td>-2.45±1.08</td><td>-2.69±2.48</td><td>0.623</td></tr><tr><td>Axial Length</td><td>24.44±0.85</td><td>24.39±0.85</td><td>0.834</td></tr><tr><td>Longitudinal Values</td><td></td><td></td><td></td></tr><tr><td>Axial Length After 6 months</td><td>24.54±0.81</td><td>24.46±0.92</td><td>0.872</td></tr><tr><td>Axial Length After 12 months</td><td>24.58±0.80</td><td>24.60±0.90</td><td>0.606</td></tr><tr><td>Axial Length After 24 months</td><td>24.75±0.99</td><td>24.92±0.64</td><td>0.595</td></tr></table>

Table 2. Analysis of Correlation Between Axial Length and Retinal Arterial Vascular Parameters at Different Time Points
<table><tr><td>Artery</td><td>MA</td><td>BA</td><td>BEA</td><td>BEC</td><td>COP</td><td>TVC</td></tr><tr><td>Defocus Lens Group</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T0 Axial Length</td><td>&lt; 0.05</td><td>0.998</td><td>0.626</td><td>0.673</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T6m Axial Length</td><td>&lt; 0.05</td><td>0.961</td><td>0.228</td><td>0.939</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T12m Axial Length</td><td>&lt; 0.05</td><td>0.901</td><td>0.474</td><td>0.665</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T24m Axial Length</td><td>&lt; 0.05</td><td>0.499</td><td>0.173</td><td>0.860</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>OK Lens Group</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T0 Axial Length</td><td>0.783</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.158</td><td>0.556</td><td>0.616</td></tr><tr><td>T6m Axial Length</td><td>0.881</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.160</td><td>0.494</td><td>0.555</td></tr><tr><td>T12m Axial Length</td><td>0.880</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.162</td><td>0.605</td><td>0.691</td></tr><tr><td>T24m Axial Length</td><td>0.738</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.154</td><td>0.258</td><td>0.324</td></tr></table>

MA: main angle; BA: branching angle; BEA: bifurcation edge angle; BEC: Bifurcation edge coefficient; COP: crossover point; TVC: terminal vessel count

Table 3. Analysis of Correlation Between Axial Length and Retinal Venous Vascular Parameters at Different Time Points
<table><tr><td>Vein</td><td>MA</td><td>BA</td><td>BA (asymmetry)</td><td>BEA</td><td>BEA (asymmetry)</td><td>COP</td><td>TVC</td></tr><tr><td>Defocus Lens Group</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T0 Axial Length</td><td>0.360</td><td>0.894</td><td>0.561</td><td>0.157</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T6m Axial Length</td><td>0.574</td><td>0.856</td><td>0.623</td><td>0.095</td><td>0.094</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T12m Axial Length</td><td>0.648</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.154</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>T24m Axial Length</td><td>0.098</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>0.148</td><td>&lt; 0.05</td><td>&lt; 0.05</td><td>&lt; 0.05</td></tr><tr><td>OK Lens Group</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T0 Axial Length</td><td>0.440</td><td>0.731</td><td>0.680</td><td>0.478</td><td>0.698</td><td>0.758</td><td>0.722</td></tr><tr><td>T6m Axial Length</td><td>0.335</td><td>0.685</td><td>0.744</td><td>0.066</td><td>0.286</td><td>0.391</td><td>0.377</td></tr><tr><td>T12m Axial Length</td><td>0.315</td><td>0.627</td><td>0.621</td><td>0.515</td><td>0.767</td><td>0.733</td><td>0.719</td></tr><tr><td>T24m Axial Length</td><td>0.089</td><td>0.665</td><td>0.861</td><td>0.483</td><td>0.791</td><td>0.474</td><td>0.449</td></tr></table>

MA: main angle; BA: branching angle; BEA: bifurcation edge angle; COP: crossover point; TVC: terminal vessel count

Table 4. Binary Logistic Regression Analysis for Factors Associated with Significant Axial Length Change
<table><tr><td rowspan="2"></td><td rowspan="2">B</td><td rowspan="2">S.E.</td><td rowspan="2">Wald</td><td rowspan="2">Sig. (P value)</td><td rowspan="2">Exp (B) (OR)</td><td colspan="2">Exp (B) 95% Cl</td></tr><tr><td>Lower Limits</td><td>Upper Limits</td></tr><tr><td>Age</td><td>0.842</td><td>0.298</td><td>8.007</td><td>0.005</td><td>2.322</td><td>1.296</td><td>4.162</td></tr><tr><td>Gender</td><td>-0.496</td><td>0.891</td><td>0.31</td><td>0.578</td><td>0.609</td><td>0.106</td><td>3.494</td></tr><tr><td>AL Baseline</td><td>-0.297</td><td>0.552</td><td>0.288</td><td>0.591</td><td>0.743</td><td>0.252</td><td>2.195</td></tr><tr><td>Artery MA</td><td>-0.019</td><td>0.015</td><td>1.588</td><td>0.208</td><td>0.982</td><td>0.954</td><td>1.01</td></tr><tr><td>Artery BA</td><td>-0.084</td><td>0.041</td><td>4.137</td><td>&lt; 0.05</td><td>0.919</td><td>0.848</td><td>0.997</td></tr><tr><td>Artery BEA</td><td>0.021</td><td>0.014</td><td>2.137</td><td>0.144</td><td>1.021</td><td>0.993</td><td>1.05</td></tr><tr><td>Artery COP</td><td>-0.352</td><td>0.401</td><td>0.772</td><td>0.38</td><td>0.703</td><td>0.321</td><td>1.542</td></tr><tr><td>Artery TVC</td><td>0.236</td><td>0.382</td><td>0.382</td><td>0.537</td><td>1.266</td><td>0.599</td><td>2.677</td></tr><tr><td>Vein MA</td><td>0.012</td><td>0.016</td><td>0.551</td><td>0.458</td><td>1.012</td><td>0.981</td><td>1.044</td></tr><tr><td>Vein BA</td><td>-0.047</td><td>0.036</td><td>1.708</td><td>0.191</td><td>0.954</td><td>0.89</td><td>1.024</td></tr><tr><td>Vein BEA</td><td>-0.021</td><td>0.018</td><td>1.327</td><td>0.249</td><td>0.98</td><td>0.946</td><td>1.015</td></tr><tr><td>Vein COP</td><td>-0.462</td><td>0.285</td><td>2.64</td><td>0.104</td><td>0.63</td><td>0.361</td><td>1.1</td></tr><tr><td>Vein TVC</td><td>0.56</td><td>0.293</td><td>3.651</td><td>0.056</td><td>1.751</td><td>0.986</td><td>3.110</td></tr></table>

95% CI: 95% Confidence Interval； S.E.: Standard Error； P<0.05 means significant

## Figure Captions

Figure 1. Flowchart of the Study Design and Data Analysis Pipeline. This study included 43 subjects with longitudinal images after applying exclusion criteria, divided into the OK lens group and Defocus lens group based on their treatments. Arteries and veins were extracted from the corresponding images using a segmentation model, and retinal vessel parameters were obtained through an automated quantification system. These parameters were then analyzed for correlation with changes in axial length of the eye.

![](images/778c4e8686a64f020e97b8cd0677713941dd998ee27e5ad0f27771557e842caf.jpg)  
Figure 2. Longitudinal changes in axial length over 24 months in the OK lens group and Defocus lens group. (A) shows changes in axial length compared to baseline, and (B) displays changes in axial length between adjacent follow-up time points. The red line represents the OK lens group, and the blue line represents the Defocus lens group.

A.  
![](images/2df5d819150992269222c6209395361f70a1e597f833449d2d478e65e229cbc7.jpg)

B.  
![](images/b164c1fd5b60270802715682a4a7d3d3a5c5301bc8deac2657fe118a7de92d01.jpg)

Figure 3. Changes in retinal artery and vein parameters over time across different groups. Red lines represent arteries; blue lines represent veins. Solid lines indicate the OK lens group, while dashed lines indicate the Defocus lens group. The analysis of retinal vessel parameters primarily focused on five aspects: (A) main vessel angle, (B) branching angle, (C) bifurcation edge angle, (D) number of crossover points, and (E) number of terminal vessels, expressed as averages.  
A.  
![](images/b0837dfd03f3aced73b72f5560d8854291cdc925691212db8adbf59166fc5ec5.jpg)

B.  
![](images/b6d2d857844f39fbd8751181d4d920a2a285224dc768e36ceb7baf1a0b200217.jpg)

C.  
![](images/60490c4180de4b5effd556d087c2658ee58d2c67ddf39c010110812918676427.jpg)

D.  
![](images/3b249e0039ab4b3819ec9ad07165ee94e68137f4abd906abe401f48e17dafdc8.jpg)

E.  
![](images/384e59676cda0fdb6d26e181505878bba949238acaf063872dea26c800e63d7f.jpg)  
Artery of OK Lens Group --Artery of Defocus Lens Group Vein of OK Lens Group ---Vein of Defocus Lens Group

## References

1. J. B. Jonas, M. Ang, P. Cho, et al. IMI prevention of myopia and its progression. Investigative ophthalmology & visual science. 2021; 62: 6-6.

2. J. J. Walline Myopia control: a review. Eye & contact lens. 2016; 42: 3-8.

3. P. K. Verkicharla, P. Kammari and A. V. Das Myopia progression varies with age and severity of myopia. Plos one. 2020; 15: e0241759.

4. N. Saka, K. Ohno-Matsui, N. Shimada, et al. Long-term changes in axial length in adult eyes with pathologic myopia. American journal of ophthalmology. 2010; 150: 562-568. e561.

5. S. Zhang, Y. Chen, Z. Li, et al. Axial elongation trajectories in Chinese children and adults with high myopia. JAMA ophthalmology. 2024; 142: 87-94.

6. J. Ruiz-Medrano, J. A. Montero, I. Flores-Moreno, et al. Myopic maculopathy: current status and proposal for a new classification and grading system (ATN). Progress in retinal and eye research. 2019; 69: 80-115.

7. H. E. Grossniklaus and W. R. Green Choroidal neovascularization. American journal of ophthalmology. 2004; 137: 496-503.

8. N. Ghazi and W. Green Pathology and pathogenesis of retinal detachment. Eye. 2002; 16: 411- 421.

9. J. K. Lau, S. J. Vincent, S.-W. Cheung and P. Cho Higher-order aberrations and axial elongation in myopic children treated with orthokeratology. Investigative ophthalmology & visual science. 2020; 61: 22-22.

10. P. Cho and S.-W. Cheung Protective role of orthokeratology in reducing risk of rapid axial elongation: a reanalysis of data from the ROMIO and TO-SEE studies. Investigative ophthalmology & visual science. 2017; 58: 1411-1416.

11. C. S. Y. Lam, W. C. Tang, D. Y.-y. Tse, et al. Defocus Incorporated Multiple Segments (DIMS) spectacle lenses slow myopia progression: a 2-year randomised clinical trial. British Journal of Ophthalmology. 2020; 104: 363-368.

12. J. Liu, Y. Lu, D. Huang, et al. The eficacy of defocus incorporated multiple segments lenses in slowing myopia progression: results from diverse clinical circumstances. Ophthalmology. 2023; 130: 542-550.

13. S. Asrani, S. Zou, S. d’Anna, S. Vitale and R. Zeimer Noninvasive mapping of the normal retinal thickness at the posterior pole. Ophthalmology. 1999; 106: 269-273.

14. T. Bek Regional morphology and pathophysiology of retinal vascular disease. Progress in retinal and eye research. 2013; 36: 247-259.

15. C. Y.-l. Cheung, S. Ong, M. K. Ikram, et al. Retinal vascular fractal dimension is associated with cognitive dysfunction. Journal of Stroke and Cerebrovascular Diseases. 2014; 23: 43-50.

16. Z. Li, Y. He, S. Keel, et al. Eficacy of a deep learning system for detecting glaucomatous optic neuropathy based on color fundus photographs. Ophthalmology. 2018; 125: 1199-1206.

17. D. Shi, W. Zhang, X. Chen, et al. Eyefound: a multimodal generalist foundation model for ophthalmic imaging. arXiv preprint arXiv:2405.11338. 2024;

18. J. Rebhan, L. P. Parker, L. J. Kelsey, F. K. Chen and B. J. Doyle A computational framework to investigate retinal haemodynamics and tissue stress. Biomechanics and Modeling in Mechanobiology. 2019; 18: 1745-1757.

19. C. J. Pournaras, E. Rungger-Brändle, C. E. Riva, S. H. Hardarson and E. Stefansson Regulation of retinal blood flow in health and disease. Progress in retinal and eye research. 2008; 27: 284- 330.

20. L. S. Lim, C. Y.-l. Cheung, X. Lin, et al. Influence of refractive error and axial length on retinal vessel geometric characteristics. Investigative ophthalmology & visual science. 2011; 52: 669- 678.

21. N. Paton, R. Maini, T. MacGillivary, et al. Efect of axial length on retinal vascular network geometry. American journal of ophthalmology. 2005; 140: 648. e641-648. e647.

22. A. Benavente-Perez Evidence of vascular involvement in myopia: a review. Frontiers in Medicine. 2023; 10: 1112996.

23. D. Ng, C. Cheung, F. Luk, et al. Advances of optical coherence tomography in myopia and pathologic myopia. Eye. 2016; 30: 901-916.

24. S. A. Burns, A. E. Elsner and T. J. Gast Imaging the retinal vasculature. Annual review of vision science. 2021; 7: 129-153.

25. D. Cabrera DeBuc, G. M. Somfai and A. Koller Retinal microvascular network alterations: potential biomarkers of cerebrovascular and neural diseases. American Journal of Physiology-Heart and Circulatory Physiology. 2017; 312: H201-H212.

26. M. M. Fraz, R. Welikala, A. R. Rudnicka, et al. QUARTZ: Quantitative Analysis of Retinal Vessel Topology and size–An automated system for quantification of retinal vessels morphology. Expert Systems with Applications. 2015; 42: 7221-7234.

27. N. Cheung, K. C. Donaghue, G. Liew, et al. Quantitative assessment of early diabetic retinopathy using fractal analysis. Diabetes care. 2009; 32: 106-110.

28. H. Chung, A. Harrisa, T. Ciulla and L. Kagemann Progress in measurement of ocular blood flow and relevance to our understanding of glaucoma and age-related macular degeneration. Progress in retinal and eye research. 1999; 18: 669-687.

29. M. K. Ikram, Y. T. Ong, C. Y. Cheung and T. Y. Wong Retinal vascular caliber measurements: clinical significance, current knowledge and future perspectives. Ophthalmologica. 2013; 229: 125-136.

30. M. K. Ikram, C. Y. Cheung, M. Lorenzi, et al. Retinal vascular caliber as a biomarker for diabetes microvascular complications. Diabetes care. 2013; 36: 750.

31. Y. Zhao, Z. Zhao, J. Yang, et al. AI-based fully automatic analysis of retinal vascular morphology in pediatric high myopia. BMC ophthalmology. 2024; 24: 415.

32. M. Li, Y. Yang, H. Jiang, et al. Retinal microvascular network and microcirculation assessments in high myopia. American journal of ophthalmology. 2017; 174: 56-67.