# LED-BENCH: A Benchmark for Large-format Engineering Drawing Understanding

Ke Li, Tianyi Wang, Enning You, Chaoqun Wan, Zhen Huang, Shuxian Liang<sup>†</sup>, Xian-Sheng Hua Institute of AI for Engineering, Tongji University <sup>§</sup> https://github.com/Wedadek/LED-BENCH

![](images/0fc1c8422599738c6c5cf108ff451ca5837084f173a7e27c2b7dc226b2ed26cb.jpg)  
Figure 1: Overview of LED-BENCH. The schematic illustrates target localization, object association, and evidence integration over complete engineering drawings. Questions range from localized evidence understanding to cross-region or cross-view reasoning.

## Abstract

Real-world engineering drawings are often large-format and information-dense. Answering practical engineering questions therefore requires both fine-grained understanding of engineering information and complex reasoning over objects, relations, and evidence distributed across different regions or views. Existing benchmarks cover a range of engineering-drawing understanding tasks involving local regions, individual views, specific elements, or more constrained visual settings. However, systematic evaluation of understanding over realistic, large-format engineering drawings remains limited. To address this gap, we introduce LED-BENCH, a multidisciplinary benchmark designed to evaluate fine-grained understanding and complex reasoning over large-format engineering drawings, spanning the mechanical, architectural, and transportation domains. Extensive evaluation of 11 representative multimodal large language models, including 8 proprietary models and 3 open-source models, reveals substantial remaining difficulty: GPT-5.6 Sol achieves 52.65% average accuracy, performance varies across engineering domains, and all evaluated models perform worse on questions requiring broader evidence integration. These results highlight fine-grained understanding, evidence integration, and complex reasoning over large-format engineering drawings as persistent challenges for current multimodal models.

## 1 Introduction

Engineering drawings communicate technical specifications through standardized graphical conventions used across mechanical engineering, architecture, and construction (International Organization for Standardization, 2020). Beyond geometric shape, they carry dimensions, tolerances, materials, manufacturing requirements, and other information needed to define and realize a design (Quintana et al., 2010). A complete large-format drawing may combine geometry, symbols, annotations, tables, notes, and information distributed across multiple views or distant regions. Answering an engineering question can therefore require reading fine-grained visual elements, associating them with the correct objects or views, reconstructing relations among engineering entities, and integrating evidence from different parts of the sheet. This setting combines the challenges of visually rich document understanding (Mathew et al., 2021; 2022) with the loss of fine detail that can arise when multimodal models process high-resolution images (Wang et al., 2025).

Recent benchmarks establish important but complementary evaluation settings. MechVQA studies recognition, reasoning, and judging on mechanical drawings, while TriView2CAD evaluates orthographic-projection reasoning for CAD (Kou et al., 2026; Niu et al., 2025). DesignQA evaluates whether multimodal models can interpret and apply requirements from engineering documentation (Doris et al., 2025). In architecture and construction, AECV-Bench evaluates object counting and drawing-grounded document question answering, ArchPlanVQA focuses on architectural floor-plan CAD drawings, and DrawingVQA organizes questions over real construction drawings by professional reasoning depth (Kondratenko et al., 2026; Li et al., 2026; Jung et al., 2026). AEC-Bench further studies tool-using agents over construction document sets (Mankodiya et al., 2026). LED-BENCH provides a common evaluation setting for finegrained understanding and complex reasoning over complete, large-format engineering drawings spanning mechanical, architectural, and transportation domains.

The benchmark is grounded in practical drawing-reading and engineering-analysis needs, covering both detailed local understanding and more complex questions requiring evidence association and integration across different regions or views. Mechanical drawings organize information through projections and detailed annotations, architectural drawings through spatial layouts, and transportation drawings through alignments and infrastructure networks. Evaluating these representations together provides a common basis for analyzing engineering drawing understanding across disciplines. Figure 1 illustrates the setting.

Table 1 compares LED-BENCH with representative engineering-drawing benchmarks in terms of domain coverage, dataset scale, visual inputs, and task focus. In constructing LED-BENCH, engineering-background annotators define question requirements, review model-assisted drafts, and verify reference answers against the drawings.

Table 1: Comparison with representative engineering-oriented multimodal benchmarks.
<table><tr><td>Benchmark</td><td>Domain</td><td>Images</td><td>Tasks</td><td>Visual Source</td><td>Task Focus</td></tr><tr><td>MechVQA</td><td>Mech.</td><td>3,281</td><td>20,778</td><td>Mechanical drawings</td><td>Recognition / Reasoning / Judging</td></tr><tr><td>AECV-Bench</td><td>AEC</td><td>21</td><td>192</td><td>AEC floor plans</td><td>OCR / Counting / Spatial / Comparative</td></tr><tr><td>ArchPlanVQA</td><td>Arch.</td><td>11,477</td><td>26,266</td><td>Architectural floor plans</td><td>Semantic understanding / Engineering reasoning</td></tr><tr><td>DrawingVQA</td><td>AEC</td><td>33</td><td>92</td><td>Construction drawings</td><td>Perceptual / Contextual / Domain-expert</td></tr><tr><td>DesignQA</td><td>Mech.</td><td>230</td><td>1,451</td><td>Engineering drawings / CAD</td><td>Requirement understanding / Compliance</td></tr><tr><td>TriView2CAD</td><td>Civil / CAD</td><td>203k</td><td>≈645K</td><td>Bridge-pier three-view CAD</td><td>Orthographic-projection reasoning</td></tr><tr><td>LED-BENCH</td><td>Mech. / Arch. / Transp.</td><td>107</td><td>377</td><td>Large-format drawings</td><td>Fine-grained understanding / Complex reasoning</td></tr></table>

Our evaluation covers 11 models over three independent runs. Results reveal distinct domain-specific performance patterns and consistently lower accuracy on the Hard subset. Supplementary experiments quantify the gains obtained by providing additional access to relevant drawing regions.

Our main contributions are threefold:

1. A multidisciplinary benchmark for large-format engineering drawings. We introduce 107 complete drawing sheets and 377 questions for evaluating fine-grained understanding and complex reasoning across mechanical, architectural, and transportation engineering.

2. An expert-led, model-assisted benchmark construction process. We construct practical drawing-reading and engineering-analysis questions through an expert-led process in which model assistance supports candidate drafting and refinement, followed by drawing-grounded human verification.

3. A systematic evaluation of current multimodal models. We evaluate 11 representative MLLMs under a unified threerun protocol, analyze performance across engineering domains and difficulty levels, and examine three visual-access interventions on the same question set.

## 2 Related Work

Multimodal reasoning and visually rich document understanding. General-purpose multimodal benchmarks have progressively moved beyond object recognition toward knowledge-intensive and compositional reasoning. MMMU (Yue et al., 2024) evaluates expert-level multimodal understanding across a broad range of academic disciplines, while MM-Vet (Yu et al., 2024) examines the integration of multiple visual and linguistic capabilities. A closely related line of work studies visually rich documents, where answering questions depends not only on recognizing text but also on interpreting layout, graphical structure, and relations among multiple visual elements. DocVQA (Mathew et al., 2021) established document-image question answering as a major evaluation setting; InfographicVQA (Mathew et al., 2022) and ChartQA (Masry et al., 2022) further require models to combine textual, graphical, and numerical evidence. OCRBench (Liu et al., 2024) systematically evaluates text-centric visual capabilities, while MMLongBench-Doc (Ma et al., 2024) extends document reasoning to long-context settings in which relevant evidence may be distributed across multiple pages and heterogeneous document modalities.

Together, these studies show that multimodal document understanding requires substantially more than OCR. Engineering drawings share this need for localization and evidence integration, but introduce an additional representational layer: dimensions, projection conventions, technical symbols, geometric relations, stationing systems, and domain-specific drafting conventions jointly determine the meaning of the visual evidence.

Engineering drawing and design benchmarks. Recent benchmarks evaluate multimodal models directly on engineering representations, but differ substantially in domain, visual unit, and reasoning scope. MechVQA (Kou et al., 2026) studies large-scale mechanical-drawing understanding through fine-grained tasks spanning recognition, reasoning, and judging. DesignQA (Doris et al., 2025) evaluates the interpretation and application of engineering requirements using textual technical documents, CAD-related imagery, and engineering drawings. TriView2CAD, introduced alongside CReFT-CAD (Niu et al., 2025), focuses on orthographic-projection reasoning and correspondence between multi-view representations and underlying CAD or geometric structure.

In architecture and construction, AECV-Bench (Kondratenko et al., 2026) evaluates drawing-grounded capabilities including text interpretation, object counting, spatial reasoning, and comparison. ArchPlanVQA (Li et al., 2026) studies visual question answering over architectural floor plans, while DrawingVQA (Jung et al., 2026) emphasizes expertcurated reasoning questions grounded in real construction drawings. AEC-Bench (Mankodiya et al., 2026) further extends evaluation toward agentic reasoning over real-world AEC document sets, spanning intra-sheet, intra-drawing, and intra-project tasks.

Taken together, prior benchmarks occupy complementary points along three axes: the engineering domain being represented, the visual unit supplied to the model, and the depth of reasoning required. LED-BENCH focuses on their intersection in a complete-sheet setting, covering heterogeneous engineering domains and large-format drawings, with questions ranging from localized evidence understanding to cases in which information must be associated across different regions or views and integrated into a complete engineering answer.

## 3 LED-BENCH

## 3.1 Drawing Collection

LED-BENCH is constructed as a reasoning-intensive benchmark. We adopt a purposive, need-driven curation strategy focused on large-format engineering drawings that support fine-grained understanding and complex reasoning. Candidate drawings are collected from publicly accessible educational, technical, and project-document sources, including engineering textbooks and handbooks, public drawing repositories, and released design documents. Annotators inspect each candidate drawing and retain those that support meaningful drawing-reading or engineering-analysis questions involving detailed engineering information, relations among engineering objects, and, where applicable, evidence distributed across different regions or views. A drawing is retained only when it is sufficiently legible for reliable human interpretation, answerrelevant evidence can be grounded unambiguously, and it supports multiple distinct and determinate questions. Incomplete, corrupted, severely blurred, visually duplicated, or insufficiently legible drawings are excluded according to these curation criteria.

## 3.2 Expert-Led Question Construction

Figure 2 illustrates the expert-led, model-assisted construction process. For each drawing, engineering-background annotators identify a practical drawing-reading or engineering-analysis need and specify the target engineering object or task, the required evidence, the relations or reasoning involved, and the requirements for a complete answer.

![](images/de12f67117c17f59a9ae6011dc9e3179cf1adba38769a71b2e3bad588846725c.jpg)  
Figure 2: Expert-led, model-assisted benchmark construction. Domain experts specify questions and supporting evidence, language models expand the candidate set, and experts review each retained question and reference answer against its source drawing.

Language models assist candidate drafting, answer proposals, and wording refinement. Engineering-background annotators verify each candidate and its reference answer against the corresponding source drawing, checking grounding, target-object association, required evidence, answer correctness and completeness, engineering relevance, ambiguity, and unintended answer cues. Candidates are revised or excluded when these criteria are not met. The retained questions are labeled Medium or Hard according to their evidence structure and reasoning requirements, as defined in Section 3.3. Appendix F details the authoring process.

## 3.3 Dataset Overview

Table 2 summarizes the frozen benchmark. LED-BENCH contains 107 complete engineering drawing sheets and 377 questions: 45 mechanical drawings with 174 questions, 21 architectural drawings with 75 questions, and 41 transportation drawings with 128 questions. Multiple questions may be associated with the same source sheet, but each question is evaluated independently with a fresh model context.

Table 2: Dataset composition across engineering domains. Q/Draw. denotes the average number of questions associated with each complete drawing. Medium and Hard are relative difficulty levels within LED-Bench.
<table><tr><td>Domain</td><td>Drawings</td><td>Questions</td><td>Q/Draw.</td><td>Medium</td><td>Hard</td></tr><tr><td>Mechanical</td><td>45</td><td>174</td><td>3.87</td><td>119</td><td>55</td></tr><tr><td>Architecture</td><td>21</td><td>75</td><td>3.57</td><td>37</td><td>38</td></tr><tr><td>Transportation</td><td>41</td><td>128</td><td>3.12</td><td>55</td><td>73</td></tr><tr><td>Total</td><td>107</td><td>377</td><td>3.52</td><td>211</td><td>166</td></tr></table>

Figure 3 summarizes the domain composition of the benchmark and the capabilities targeted by its questions.

Difficulty levels. LED-BENCH uses two relative difficulty levels, Medium and Hard. Medium questions typically rely on a bounded set of evidence and require operations such as combining several annotations, resolving a local structural or assembly relation, establishing a simple cross-region or cross-view correspondence, or performing a one-step engineering calculation. Hard questions require broader or more interdependent evidence, such as associating information across different regions or multiple views, reconstructing spatial or topological relations, tracing ordered dependencies, performing multi-step inference or calculation, or jointly applying multiple engineering constraints. Both levels remain drawing-grounded reasoning tasks; the distinction reflects the scope and interdependence of evidence and reasoning rather than a separation between recognition and reasoning. Engineering-background experts review assignments against the corresponding drawings.

![](images/31749f65ab38e9bdb10d32bc1a0e59442bde0c7d539341c8ac435a3f9ed48e59.jpg)  
Figure 3: Dataset composition and benchmark focus. (a) Question distribution and Medium/Hard split across the three domains. (b) Conceptual hierarchy of capabilities involved in large-format engineering drawing understanding and reasoning.

Drawing scale. LED-BENCH exhibits substantial variation in image scale across domains, with median long-side lengths of approximately 1,450 pixels for mechanical drawings, 6,800 pixels for architectural drawings, and 3,400 pixels for transportation drawings. This variation reflects the heterogeneous spatial organization of large-format engineering drawings. Appendix A.1 further examines model sensitivity to input rasterization resolution.

## 4 Evaluation Results

## 4.1 Evaluation Setup

Testing models. We evaluate 11 representative multimodal large language models (MLLMs), including eight proprietary models and three open-weight models. The proprietary models are GPT-5.6 Sol, GPT-5.6 Luna, GPT-5.5, Gemini-3.5 Flash, Qwen3.8-Max, Qwen3.8-Flash, Qwen3.7-Flash, and Qwen3.6-Flash; the open-weight models are Kimi K2.5, Qwen3.6-27B, and Qwen3.8-27B.

Automatic scoring. LED-BENCH uses strict binary semantic scoring for open-ended answers. Each response is independently evaluated by three LLM judges using the question, reference answer, and required answer content, while the original drawing and evaluated-model identity are hidden from the judges. A response receives a score of 1 only when all indispensable engineering content is correct; semantically equivalent wording, numerical representations, and units are accepted, whereas substantive omissions, incorrect object or relation associations, numerical errors, or contradictions receive 0. Scores are aggregated from the judges’ valid votes as detailed in Appendix B. In a blinded review of 90 sampled responses, human labels agree with the LLM-majority decisions in 95.6% of cases (Cohen’s $\kappa = 0 . 9 0 2 ) \mathrm { ; }$ Appendix C details this reference-based validation. Each model is evaluated in three independent runs over the full benchmark. Response scores are pooled across the runs within each domain, without majority voting across model responses. Because the three domains contain different numbers of questions, we use the unweighted domain average $A _ { \mathrm { a v g } } = \textstyle { \frac { 1 } { 3 } } ( A _ { \mathrm { m e c h } } + A _ { \mathrm { a r c h } } + A _ { \mathrm { t r a n s } } )$ as the primary metric.

Difficulty-level results are computed in the same way within the Medium and Hard subsets. Run-level results and stability statistics are reported in Appendix D.

## 4.2 Overall Performance

Table 3 reports performance across the three engineering domains. The highest domain-avg accuracy is 52.65%, achieved by GPT-5.6 Sol, with 58.81% on mechanical, 48.89% on architectural, and 50.26% on transportation drawings.

Table 3: Performance across engineering domains. The best result in each column is shown in bold, and the second-best is underlined. Domain Avg uses unrounded domain scores; recomputing from the two-decimal entries can differ by 0.01 percentage points.
<table><tr><td>Model</td><td>Mechanical↑</td><td>Architecture↑</td><td>Transportation↑</td><td>Domain Avg↑</td></tr><tr><td colspan="5">Closed-source MLLMs</td></tr><tr><td>GPT-5.6 Sol</td><td>58.81</td><td>48.89</td><td>50.26</td><td>52.65</td></tr><tr><td>GPT-5.6 Luna</td><td>50.38</td><td>47.11</td><td>44.27</td><td>47.26</td></tr><tr><td>Qwen3.8-Max</td><td>46.36</td><td>48.00</td><td>46.09</td><td>46.82</td></tr><tr><td>Gemini-3.5 Flash</td><td>57.47</td><td>35.11</td><td>31.77</td><td>41.45</td></tr><tr><td>GPT-5.5</td><td>41.00</td><td>17.78</td><td>39.84</td><td>32.87</td></tr><tr><td>Qwen3.7-Flash</td><td>32.57</td><td>16.44</td><td>27.86</td><td>25.63</td></tr><tr><td>Qwen3.8-Flash</td><td>29.89</td><td>16.44</td><td>27.34</td><td>24.56</td></tr><tr><td>Qwen3.6-Flash</td><td>18.39</td><td>6.22</td><td>15.36</td><td>13.33</td></tr><tr><td colspan="5">Open-weight MLLMs</td></tr><tr><td>Kimi-K2.5</td><td>32.18</td><td>15.11</td><td>27.08</td><td>24.79</td></tr><tr><td>Qwen3.8-27B</td><td>22.41</td><td>20.44</td><td>24.22</td><td>22.36</td></tr><tr><td>Qwen3.6-27B</td><td>31.99</td><td>6.22</td><td>22.92</td><td>20.38</td></tr></table>

Performance also varies substantially across domains. For example, Gemini-3.5 Flash reaches 57.47% on mechanical drawings but 35.11% and 31.77% on architectural and transportation drawings, respectively. In contrast, Qwen3.8-Max remains within a narrower range of 46.09–48.00% across the three domains. These results indicate that aggregate performance alone can obscure substantially different domain-specific behavior.

## 4.3 Performance across Difficulty Levels

Table 4 compares performance on Medium and Hard questions using domain-avg accuracy. All evaluated models perform worse on the Hard subset. GPT-5.6 Sol decreases from 56.57% to 48.47%, Qwen3.8-Max from 49.97% to 42.09%, and Gemini-3.5 Flash from 47.39% to 34.49%, the largest absolute drop among the evaluated models at 12.90 percentage points.

Table 4: Domain-avg accuracy (%) across the two difficulty levels. The best and second-best accuracies in each difficulty level are shown in bold and underlined, respectively. ∆ denotes the absolute decrease from Medium to Hard.
<table><tr><td>Model</td><td>Medium↑</td><td>Hard↑</td><td>Δ</td></tr><tr><td colspan="4">Closed-source MLLMs</td></tr><tr><td>GPT-5.6 Sol</td><td>56.57</td><td>48.47</td><td>8.10</td></tr><tr><td>GPT-5.6 Luna</td><td>48.59</td><td>44.60</td><td>3.99</td></tr><tr><td>Qwen3.8-Max</td><td>49.97</td><td>42.09</td><td>7.88</td></tr><tr><td>Gemini-3.5 Flash</td><td>47.39</td><td>34.49</td><td>12.90</td></tr><tr><td>GPT-5.5</td><td>34.34</td><td>29.36</td><td>4.98</td></tr><tr><td>Qwen3.7-Flash</td><td>27.01</td><td>23.91</td><td>3.10</td></tr><tr><td>Qwen3.8-Flash</td><td>27.21</td><td>20.63</td><td>6.58</td></tr><tr><td>Qwen3.6-Flash</td><td>16.94</td><td>8.57</td><td>8.37</td></tr><tr><td colspan="4">Open-weight MLLMs</td></tr><tr><td>Kimi-K2.5</td><td>25.98</td><td>22.68</td><td>3.30</td></tr><tr><td>Qwen3.8-27B</td><td>25.80</td><td>18.04</td><td>7.76</td></tr><tr><td>Qwen3.6-27B</td><td>23.59</td><td>16.86</td><td>6.73</td></tr></table>

The accuracy gap is consistent with the intended Medium/Hard distinction: Hard questions generally require broader or more interdependent evidence and more involved reasoning.

## 4.4 Failure Stages and Diagnostic Interventions

We analyze model failures at four conceptual stages: Local Evidence Understanding, Object Association, Relation Reconstruction, and Integrated Engineering Reasoning. In the coded error sample shown in Figure 4, Relation Reconstruction is the largest category (47.3%), followed by Integrated Engineering Reasoning (23.4%), Object Association (17.2%), and Local Evidence Understanding (12.1%).

To examine whether improved visual access can alleviate these failures, we conduct a supplementary diagnostic study on 377 questions from 107 drawings, comprising 174 mechanical, 75 architectural, and 128 transportation questions. Four response models—GPT-5.4, Qwen3.8-Flash, Gemini-3.5 Flash, and GPT-5.6 Sol—are evaluated for three runs under a whole-sheet baseline and three visual-access interventions. The fixed overlapping crops setting supplements the full drawing with four predefined overlapping crops. Locate then reread first identifies likely target regions and then rereads the corresponding crops. Search then reread searches for multiple relevant regions before rereading and integrating the resulting observations.

![](images/45ff22d3d48fd52c632704b891a3d84140b9350508c6c4fde03cb0b5a60c74ba.jpg)  
(a) Failure-stage distribution

![](images/5b09541948654a87d5dc108d17d41f7e5f1e9dc61710d49828b3535a620d20ad.jpg)  
(b) Accuracy across domains  
Figure 4: Failure stages and domain-level intervention performance. (a) Distribution of the four failure stages in the coded error sample used for the diagnostic analysis. (b) Four-model mean accuracy across Mechanical, Architecture, and Transportation for whole-sheet direct answering and three visual-access interventions.

Table 5: Domain-avg accuracy (%) for the whole-sheet baseline and three visual-access interventions. ∆ denotes the absolute change relative to whole-sheet direct answering.
<table><tr><td>Method</td><td>Intended support</td><td>Domain Avg↑</td><td>∆</td></tr><tr><td>Whole-sheet direct answering</td><td>Baseline</td><td>36.38</td><td></td></tr><tr><td>Fixed overlapping crops</td><td>Local evidence access</td><td>49.34</td><td>12.96</td></tr><tr><td>Locate then reread</td><td>Target localization + rereading</td><td>41.43</td><td>5.05</td></tr><tr><td>Search then reread</td><td>Multi-region search + rereading</td><td>49.74</td><td>13.35</td></tr></table>

The three interventions exhibit different cross-domain behavior. Fixed overlapping crops improve accuracy in all three domains and raise the domain-avg score from 36.38% to 49.34%. Search then reread reaches a similar overall level (49.74%) and gives the highest architectural accuracy (53.45%). Locate then reread is more mixed: it improves Architecture and Transportation but decreases Mechanical from 44.12% to 36.47%. Relative to whole-sheet direct answering, the visual-access interventions achieve higher overall accuracy, with the largest gains differing across domains and methods Their input procedures and model-call configurations are detailed in Appendix G.

## 5 Conclusion

LED-BENCH establishes a multidisciplinary benchmark for fine-grained understanding and complex reasoning over complete engineering drawings across mechanical, architectural, and transportation domains. Under a unified three-run evaluation protocol, the 11 evaluated MLLMs show substantial cross-domain variation and consistently lower accuracy

on questions requiring broader and more interdependent evidence. Supplementary visual-access interventions improve performance in several settings, with different effects across methods and domains. These findings highlight the importance of locating, associating, and integrating engineering evidence across complete drawing sheets.

## References

Anna C. Doris, Daniele Grandi, Ryan Tomich, Md Ferdous Alam, Mohammadmehdi Ataei, Hyunmin Cheong, and Faez Ahmed Designqa: A multimodal benchmark for evaluating large language models’ understanding of engineering documentation. Journal of Computing and Information Science in Engineering, 25(2):021009, 2025. doi: 10.1115/1.4067333.

International Organization for Standardization. Technical product documentation (TPD)—general principles of representation—part 1: Introduction and fundamental requirements. Technical Report ISO 128-1:2020, International Organization for Standardization, 2020. URL https://www.iso.org/standard/65296.html.

Yoonhwa Jung, Junryu Fu, and Mani Golparvar-Fard. Drawingvqa: A real-world benchmark for multi-depth visual-textual reasoning on construction drawings. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 2121–2130, June 2026. URL https://arxiv.org/abs/2607.15418.

Aleksei Kondratenko, Mussie Birhane, Houssame E. Hsain, and Guido Maciocci. Aecv-bench: Benchmarking multimodal models on architectural and engineering drawings understanding. arXiv preprint arXiv:2601.04819, 2026. doi: 10.48550/arXiv.2601.04819.

Qian Kou, Xiaofeng Shi, Yulin Li, Xiaosong Qiu, Xinyang Wang, Hua Zhou, and Dongxing Cao. Mechvqa: Benchmarking and enhancing multimodal llms on comprehensive mechanical drawing understanding. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2605.30794.

Pengyang Li, Hainan Chen, Zhuang Zheng, Juntao He, and Hai Yang. Archplanvqa: Benchmark and comprehensive evaluation of vision-language models for understanding architectural floor plan cad drawings. Journal ofComputing in Civil Engineering, 40(5): 04026060, 2026. doi: 10.1061/JCCEE5.CPENG-7571.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. doi: 10.1007/s11432-024-4235-6.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. In Advances in Neural Information Processing Systems, volume 37, pp. 95963–96010, 2024. doi: 10.52202/079017-3041. Datasets and Benchmarks Track.

Harsh Mankodiya, Chase Gallik, Theodoros Galanos, and Andriy Mulyar. Aec-bench: A multimodal benchmark for agentic systems in architecture, engineering, and construction. arXiv preprint arXiv:2603.29199, 2026. doi: 10.48550/arXiv.2603.29199.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pp. 2263–2279, 2022. doi: 10.18653/v1/2022.findings-acl.177.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pp. 2200–2209, 2021.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pp. 1697–1706, 2022.

Ke Niu, Zhuofan Chen, Haiyang Yu, Yuwen Chen, Teng Fu, Mengyang Zhao, Bin Li, and Xiangyang Xue. Creft-cad: Boosting orthographic projection reasoning for cad via reinforcement fine-tuning. In Advances in Neural Information Processing Systems, volume 38, pp. 92007–92029, 2025. doi: 10.52202/085713-2765.

Virgilio Quintana, Louis Rivest, Robert Pellerin, Frédérick Venne, and Fawzi Kheddouci. Will model-based definition replace engineering drawings throughout the product lifecycle? a global perspective from aerospace industry. Computers in Industry, 61(5): 497–508, 2010. doi: 10.1016/j.compind.2010.01.005.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025. doi: 10.1609/aaai.v39i8.32852.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 57730–57754, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

A. Strict accuracy by input resolution

## A Drawing Scale and Visual Input Considerations

The three domains differ substantially in raster dimensions, reflecting both their heterogeneous source types and drawing conventions. For native mechanical rasters and the frozen 200-DPI architectural/transportation renders, the median long-side lengths are approximately 1,450 pixels for mechanical drawings, 6,800 pixels for architectural drawings, and 3,400 pixels for transportation drawings. Mechanical sheets often contain dense multi-view geometry and technical annotations, while architectural and transportation sheets distribute evidence across larger spatial extents.

These dimensions describe images at submission. Model APIs apply their own visual preprocessing, including resizing, tiling, and visual tokenization.

## A.1 Exploratory Rasterization Results

Figure 5 summarizes an exploratory rasterization study on architectural and transportation PDF-source drawings. We compare multiple client-side rendering resolutions and report the strict accuracy among adjudicated answers for four response models. Higher input resolution generally improves accuracy, particularly for GPT-5.6 Sol and Gemini 3.5 Flash, with variation across models and rendering levels.

The main benchmark uses 200 DPI for PDF-source drawings and preserves native mechanical raster images. The exploratory study characterizes how client-side rendering resolution affects model performance.

![](images/1514de19fdecae34cab2276e4652626df949f13759a33810e4852c9a953fbfee.jpg)  
Resolution level (PDF DPI; mechanical scaling below)  
Figure 5: Exploratory rasterization study. Strict accuracy among adjudicated answers as a function of client-side PDF rasterization resolution for four representative response models. Results are reported across the tested rendering resolutions.

## B Evaluation Protocol Details

Each question receives a fresh request with the complete image and no retrieval tools, demonstrations, or reference answer.   
The inference client preserves the submitted image bytes.

Its wrapper sends max\_completion\_tokens=32768 and normally requests response\_format=json\_ object. Temperature, top-p, seed, reasoning effort, and image-detail parameters are omitted from the requests; their effective values are provider defaults. The client requests a short final answer, and an interface-error fallback can remove the JSON response-format constraint. The completion-token limit applies to provider-accounted output tokens, including reasoning tokens where applicable.

R1–R3 use the same request template with independent calls and response-level scoring. Initial pass-level deadlines are 240/600/600 seconds, with deferred technical retries; recovery schedules and attempt metadata are recorded in the run artifacts. Successful responses are retained, and retries address technical failures. Per-attempt records contain parsing status, returned model IDs, finish reasons, and retry information.

The eleven displayed main-table API aliases are gpt-5.6-sol, gpt-5.6-luna, gpt-chat-latest, gemini-3.5-flash, qwen3.8-max, qwen3.8-flash, qwen3.7-flash, qwen3.6-flash, kimi-k2.5, qwen3.8-27b, and qwen3.6-27b. The gateway alias gpt-chat-latest is reported as GPT-5.5.

## B.1 Exact answering prompt

Discipline: {domain\_label}.   
Answer only from evidence visible in the attached complete drawing.   
Do not use external standards or assumptions to fill missing information.   
Do not describe your reasoning. Preserve identifiers, specifications, numbers, units, tolerances,   
,→ directions, object-value bindings, and necessary order.   
Answer in the same language as the question. Include every necessary answer field.   
Question: {question}   
Return exactly one JSON object with no Markdown:   
{"answer":"...","answerability":"answerable|partial|insufficient|ambiguous","confidence":0.0}

The parser accepts a JSON object, including one within a code fence, and can recover a decodable object containing an answer field. Empty, truncated, or unusable responses are retained as technical or response failures. Confidence and answerability fields are stored but do not weight correctness.

## B.2 Exact binary judge prompt

Each judge receives only the question, full reference answer, mandatory components, and final candidate answer. Candidatemodel identity and the source image are absent. The component list contains the full reference answer as one item, and judges return an integer score of 0 or 1 with a brief explanation.

```jsonl
You are a strict evaluator for an engineering-drawing question-answering benchmark. Judge only
whether the candidate answer is fully correct with respect to the provided reference answer and,→
mandatory components. Do not infer the candidate model identity. Return JSON only.,→
Judge whether the candidate answer is fully correct.
Question:
{question}
Reference answer:
{reference_answer}
Mandatory components:
{mandatory_components}
Candidate answer:
{candidate_answer}
Rules:
- Score 1 only if the candidate is semantically correct and ALL mandatory components are satisfied.
- Score 0 if any mandatory component is missing or wrong.
- Score 0 for a wrong engineering object/instance, swapped values, wrong
,→ relation/direction/identifier, or a material contradiction.
- Equivalent wording, standard engineering synonyms, and exactly equivalent unit conversions are
,→ acceptable.
- Do not require extra reasoning or details that are not mandatory.
- Use binary scoring only. Do not output partial credit.
Return JSON only:
{"score": 0 or 1, "reason": "brief explanation"}
```

Judge prompts, panel identifiers, and run-manifest identifiers are recorded as evaluation metadata.

Three valid votes use majority decision. Two valid agreeing votes also yield a score; conflicting pairs or a single valid vote remain NA. Missing terminal candidate answers are zeros, whereas unresolved judge decisions are excluded from the scoring denominator. Judges assess semantic equivalence to the provided reference answer.

## B.3 Scoring and aggregation formulas

For question i in run r, each of the three judges returns a binary decision $s _ { i , r } ^ { ( j ) } \in \{ 0 , 1 \}$ when valid. The final response score is

$$
s _ { i , r } = { \left\{ \begin{array} { l l } { { \mathrm { m a j o r i t y } } \{ s _ { i , r } ^ { ( j ) } \} _ { j = 1 } ^ { 3 } , } & { { \mathrm { t h r e e ~ v a l i d ~ v o t e s } } , } \\ { v , } & { { \mathrm { t w o ~ v a l i d ~ v o t e s } } , { \mathrm { b o t h ~ e q u a l ~ t o ~ } } v , } \\ { { \mathrm { N A } } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

Missing terminal candidate answers are scored as 0 under the frozen policy, whereas unresolved judge outcomes remain NA and are excluded from the denominator.

For domain d containing $N _ { d }$ questions, pooled domain accuracy over the three independent runs is

$$
A _ { d } = \frac { \sum _ { i \in d } \sum _ { r = 1 } ^ { 3 } z _ { i , r } s _ { i , r } } { \sum _ { i \in d } \sum _ { r = 1 } ^ { 3 } z _ { i , r } } , \qquad z _ { i , r } = \mathbb { I } [ s _ { i , r } \neq \mathrm { N A } ] .\tag{2}
$$

The primary benchmark metric is the unweighted domain average

$$
A _ { \mathrm { a v g } } = \frac { 1 } { 3 } \left( A _ { \mathrm { m e c h } } + A _ { \mathrm { a r c h } } + A _ { \mathrm { t r a n s } } \right) .\tag{3}
$$

Difficulty-level results are computed analogously by restricting the sums in Eq. (2) to the corresponding Medium or Hard subset within each domain and then averaging the resulting domain accuracies with equal weight.

## C Human Validation of the LLM Scoring Protocol

We assess the reliability of the automatic majority-vote scorer using a blinded human review of 90 model responses, with 30 responses sampled from each of the Mechanical, Architecture, and Transportation subsets. The review set was constructed from stored R2/R3 responses using a fixed random seed. Within each domain, sampled questions were distinct, sampling did not use the LLM-judge label, and candidate model/run assignments were approximately balanced. During review, the candidate-model identity, run identifier, individual judge votes, and majority decision were hidden. The human reviewer saw the question, reference answer, and candidate response and assigned a binary correctness label under the same complete-answer criterion as the automatic scorer. Cohen’s κ is computed over binary human labels; no sampled response was marked unjudgeable.

Table 6: Blinded human validation of the LLM-based majority-vote scoring protocol. Agreement is the fraction of responses for which the human binary label matches the three-judge majority decision.
<table><tr><td>Subset</td><td>N</td><td>Agreement (%)</td><td>Cohen&#x27;s κ</td></tr><tr><td>Overall</td><td>90</td><td>95.6</td><td>0.902</td></tr><tr><td>Mechanical</td><td>30</td><td>93.3</td><td>0.815</td></tr><tr><td>Architecture</td><td>30</td><td>96.7</td><td>0.932</td></tr><tr><td>Transportation</td><td>30</td><td>96.7</td><td>0.930</td></tr><tr><td>Unanimous judge votes</td><td>71</td><td>97.2</td><td>0.934</td></tr><tr><td>2:1 split judge votes</td><td>19</td><td>89.5</td><td>0.791</td></tr></table>

Across the 90 reviewed responses, 86 receive the same binary decision from the human reviewer and the LLM-majority scorer. Agreement is higher when all three judges agree than when the vote is split 2:1.

## D Run-to-Run Stability

To characterize repeated-run variability under the same primary aggregation used in the main evaluation, we compute domain-avg accuracy separately for each of the three runs. Within each run, accuracy is first computed independently on Mechanical, Architecture, and Transportation, and the three domain accuracies are then averaged with equal weight. Table 7 reports the resulting run-wise domain-avg scores and their arithmetic mean and sample standard deviation across

Table 7: Run-to-run stability under the primary domain-avg metric. R1–R3 report domain-avg accuracy (%) for each independent run; the final column reports the arithmetic mean and sample standard deviation across the three runs.
<table><tr><td>Model</td><td>R1↑</td><td>R2↑</td><td>R3↑</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td colspan="5">Closed-source MLLMs</td></tr><tr><td>GPT-5.6 Sol</td><td>54.15</td><td>53.79</td><td>50.02</td><td> $5 2 . 6 5 \pm 2 . 2 9$ </td></tr><tr><td>GPT-5.6 Luna</td><td>50.02</td><td>47.72</td><td>44.02</td><td> $4 7 . 2 6 \pm 3 . 0 3$ </td></tr><tr><td>Qwen3.8-Max</td><td>46.88</td><td>47.09</td><td>46.48</td><td> $4 6 . 8 2 \pm 0 . 3 1$ </td></tr><tr><td>Gemini-3.5 Flash</td><td>39.45</td><td>44.02</td><td>40.88</td><td> $4 1 . 4 5 \pm 2 . 3 4$ </td></tr><tr><td>GPT-5.5</td><td>32.98</td><td>33.28</td><td>32.35</td><td> $3 2 . 8 7 \pm 0 . 4 7$ </td></tr><tr><td>Qwen3.7-Flash</td><td>25.64</td><td>26.77</td><td>24.46</td><td> $2 5 . 6 3 \pm 1 . 1 5$ </td></tr><tr><td>Qwen3.8-Flash</td><td>26.52</td><td>25.11</td><td>22.04</td><td> $2 4 . 5 6 \pm 2 . 2 9$ </td></tr><tr><td>Qwen3.6-Flash</td><td>13.68</td><td>12.99</td><td>13.32</td><td> $1 3 . 3 3 \pm 0 . 3 5$ </td></tr><tr><td colspan="5">Open-weight MLLMs</td></tr><tr><td>Kimi-K2.5</td><td>23.70</td><td>24.99</td><td>25.68</td><td> $2 4 . 7 9 \pm 1 . 0 0$ </td></tr><tr><td>Qwen3.8-27B</td><td>22.85</td><td>22.66</td><td>21.57</td><td> $2 2 . 3 6 \pm 0 . 6 9$ </td></tr><tr><td>Qwen3.6-27B</td><td>22.80</td><td>19.88</td><td>18.45</td><td> $2 0 . 3 8 \pm 2 . 2 2$ </td></tr></table>

the three runs $( n = 3 )$ . The reported sample standard deviation summarizes run-to-run variation; results are aggregated across responses rather than by voting across runs.  
Table 8 gives the corresponding domain-level stability statistics underlying the domain-avg scores. For each mode and domain, the reported mean is computed from the three run-wise domain accuracies and matches the pooled domain accuracy in Table 3; the Domain Avg column is therefore identical to the final column of Table 7.

Table 8: Domain-level run-to-run stability. Values are mean ± sample standard deviation of the three run-wise accuracies (%).
<table><tr><td>Model</td><td>Mechanical↑</td><td>Architecture↑</td><td>Transportation↑</td><td>Domain Avg↑</td></tr><tr><td colspan="5">Closed-source MLLMs</td></tr><tr><td>GPT-5.6 Sol</td><td> $5 8 . 8 1 \pm 0 . 8 8$ </td><td> $4 8 . 8 9 \pm 4 . 0 7$ </td><td> $5 0 . 2 6 \pm 4 . 3 0$ </td><td> $5 2 . 6 5 \pm 2 . 2 9$ </td></tr><tr><td>GPT-5.6 Luna</td><td> $5 0 . 3 8 \pm 1 . 7 6$ </td><td> $4 7 . 1 1 \pm 4 . 2 9$ </td><td> $4 4 . 2 7 \pm 6 . 6 4$ </td><td> $4 7 . 2 6 \pm 3 . 0 3$ </td></tr><tr><td>Qwen3.8-Max</td><td> $4 6 . 3 6 \pm 2 . 0 2$ </td><td> $4 8 . 0 0 \pm 2 . 6 7$ </td><td> $4 6 . 0 9 \pm 1 . 5 6$ </td><td> $4 6 . 8 2 \pm 0 . 3 1$ </td></tr><tr><td>Gemini-3.5 Flash</td><td> $5 7 . 4 7 \pm 1 . 0 0$ </td><td> $3 5 . 1 1 \pm 4 . 2 9$ </td><td> $3 1 . 7 7 \pm 3 . 5 2$ </td><td> $4 1 . 4 5 \pm 2 . 3 4$ </td></tr><tr><td>GPT-5.5</td><td> $4 1 . 0 0 \pm 1 . 8 5$ </td><td> $1 7 . 7 8 \pm 4 . 0 7$ </td><td> $3 9 . 8 4 \pm 1 . 3 5$ </td><td> $3 2 . 8 7 \pm 0 . 4 7$ </td></tr><tr><td>Qwen3.7-Flash</td><td> $3 2 . 5 7 \pm 1 . 2 0$ </td><td> $1 6 . 4 4 \pm 3 . 3 6$ </td><td> $2 7 . 8 6 \pm 1 . 9 7$ </td><td> $2 5 . 6 3 \pm 1 . 1 5$ </td></tr><tr><td>Qwen3.8-Flash</td><td> $2 9 . 8 9 \pm 0 . 5 7$ </td><td> $1 6 . 4 4 \pm 5 . 3 9$ </td><td> $2 7 . 3 4 \pm 2 . 0 7$ </td><td> $2 4 . 5 6 \pm 2 . 2 9$ </td></tr><tr><td>Qwen3.6-Flash</td><td> $1 8 . 3 9 \pm 1 . 5 2$ </td><td> $6 . 2 2 \pm 2 . 0 4$ </td><td> $1 5 . 3 6 \pm 1 . 1 9$ </td><td> $1 3 . 3 3 \pm 0 . 3 5$ </td></tr><tr><td colspan="5">Open-weight MLLMs</td></tr><tr><td>Kimi-K2.5</td><td> $3 2 . 1 8 \pm 1 . 1 5$ </td><td> $1 5 . 1 1 \pm 3 . 3 6$ </td><td> $2 7 . 0 8 \pm 1 . 1 9$ </td><td> $2 4 . 7 9 \pm 1 . 0 0$ </td></tr><tr><td>Qwen3.8-27B</td><td> $2 2 . 4 1 \pm 2 . 6 3$ </td><td> $2 0 . 4 4 \pm 2 . 7 8$ </td><td> $2 4 . 2 2 \pm 1 . 5 6$ </td><td> $2 2 . 3 6 \pm 0 . 6 9$ </td></tr><tr><td>Qwen3.6-27B</td><td> $3 1 . 9 9 \pm 3 . 2 7$ </td><td> $6 . 2 2 \pm 2 . 7 8$ </td><td> $2 2 . 9 2 \pm 2 . 3 9$ </td><td> $2 0 . 3 8 \pm 2 . 2 2$ </td></tr></table>

Across the eleven reported models, domain-avg run-to-run sample standard deviations range from 0.31 to 3.03 percentage points. Some model–domain combinations, particularly in Architecture and Transportation, exhibit larger variability across runs.

## E Operational Interpretation of the Failure Stages

Table 9: Operational interpretation of the four conceptual failure stages used in the diagnostic analysis.
<table><tr><td>Stage</td><td>Operational interpretation</td></tr><tr><td></td><td>Local Evidence Understanding Relevant local text, numbers, identifiers, specifications, symbols, or geometric cues are read or interpreted incorrectly.</td></tr><tr><td>Object Association</td><td>Evidence is plausibly read but associated with the wrong engineering object, instance, space, component, annotation target, or view-specific counterpart.</td></tr><tr><td>Relation Reconstruction</td><td>Relevant objects are identified, but connectivity, adjacency, path, direction, order, continuity, scope, datum, or cross-region/view correspondence is reconstructed incorrectly.</td></tr><tr><td>Integrated Engineering Reasoning</td><td>Upstream evidence is substantially available, but the final answer fails during evidence combination, multi-step derivation, arithmetic or unit handling, aggregation, constraint application, or answer completion.</td></tr></table>

These operational categories organize the response-level error analysis reported in Section 4.4.

## F Question Authoring and Quality-Control Details

Archived authoring materials. Candidate construction was iterative. The archived development prompts ask a model to propose questions, candidate answers, evidence dependencies, and wording refinements from a complete engineering drawing. Engineering-background annotators review candidate questions and select the final items using drawing-grounded criteria.

Operational construction criteria. For each candidate, annotators identify a practical information need, the target engineering object or task, the drawing evidence required to answer it, the relations or inference involved, and the requirements for a complete answer. Candidates are revised or removed when grounding is unclear, the reference answer is incorrect or incomplete, wording creates unintended answer cues, engineering relevance is weak, or substantial semantic redundancy remains.

Difficulty distinction. The Medium/Hard distinction reflects evidence structure and reasoning demand. Medium questions typically use a bounded evidence set and relatively direct combinations or calculations. Hard questions require broader or more interdependent evidence, such as cross-region or cross-view association, spatial or topologica reconstruction, ordered-dependency tracing, multi-step derivation, or joint application of multiple engineering constraints. Engineering-background experts review this distinction with reference to the source drawing and the evidence required to answer each question.

## G Supplementary Method Study: Execution and Results

## G.1 Inputs and intervention procedures

The supplementary study compares whole-sheet direct answering with three visual-access interventions on the same 377 questions and answer requirements. Reference answers are withheld during inference, and each model–question–method combination is evaluated in three independent runs, without voting across runs or selecting the best response

Fixed overlapping crops provide the largest point estimate in Mechanical and Transportation, while search then reread performs best in Architecture and has the highest domain-avg point estimate. Locate then reread improves Architecture and Transportation over whole-sheet direct answering but is lower in Mechanical. These comparisons characterize performance under the respective visual-access input and inference procedures. Whole-sheet and fixed-crop settings normally use one model request, whereas locate/search methods normally use three, as summarized in the procedure table.

Table 10: Visual-access intervention inputs and normal deployment calls per response. Failures and fallback paths may change the realized count.
<table><tr><td>Method</td><td>Input / procedure</td><td>Calls</td></tr><tr><td>Whole-sheet direct answering</td><td>Complete drawing and question; answer directly from the full visual input</td><td>1</td></tr><tr><td>Fixed overlapping crops</td><td>Complete drawing plus four predefined overlapping crops in one request</td><td>1</td></tr><tr><td>Locate then reread</td><td>Locate at most two likely target regions, reread the corresponding crops, and answer using the full drawing and reread observations</td><td>3</td></tr><tr><td>Search then reread</td><td>Search for multiple relevant regions, reread up to two selected regions, and integrate the resulting observations</td><td>3</td></tr></table>

## H Evaluation Scope and Interpretation

LED-Bench evaluates a curated set of 107 complete engineering drawing sheets and 377 questions across three disciplines.   
The selection emphasizes drawing-grounded evidence association, relation reconstruction, and integrated reasoning.   
Domain-specific drafting conventions, drawing scales, and question composition are reflected in the reported results.

Questions originating from the same sheet share a visual context. The evaluation reports pooled accuracy across three independent model runs and run-level variation; the supplementary methods study reports accuracy under its documented input and inference-call configurations. The scoring protocol evaluates answer correctness against drawing-grounded reference answers, with a separate blinded comparison between human and LLM-majority scoring decisions.

Source drawings were collected from public educational, technical, and project-document materials, while the question specifications and reference answers were constructed for LED-Bench. Release materials will document source provenance and redistribution permissions.