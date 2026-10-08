# From Pixel to Coding: Evaluating the Figure Reproduction Capabilities of MLLMs

Zijian Chen<sup>1,2</sup>, Zhengyu Chen<sup>1</sup>, Bohan Liang<sup>2</sup>, Lirong Deng<sup>3</sup>, Yushuo Zheng<sup>1,2</sup>, Yanwei Jiang<sup>1,2</sup>, Qi Jia<sup>2</sup>, Kaiwei Zhang<sup>2</sup>, Wenjun Zhang<sup>1</sup>, Guangtao Zhai<sup>1,2,\*</sup>

<sup>1</sup>Institute of Image Communication and Information Processing, Shanghai Jiao Tong University, Shanghai, 200240, China. <sup>2</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, 200030, China. <sup>3</sup>Macao Polytechnic University, Macao, 999078, China. Corresponding author: Guangtao Zhai.

Contributing authors: zijian.chen@sjtu.edu.cn; chenzy828@sjtu.edu.cn;   
bobliang@sjtu.edu.cn; p2523406@mpu.edu.mo; yushuozheng@sjtu.edu.cn;   
jiang-yan-wei@sjtu.edu.cn; jiaqi@pjlab.org.cn; zhangkaiwei@pjlab.org.cn; zhangwenjun@sjtu.edu.cn; zhaiguangtao@sjtu.edu.cn;

## Abstract

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in both visual understanding and code generation. However, existing benchmarks typically evaluate these two modalities in isolation, lacking a dedicated assessment of their unification, i.e., how a model can perceive complex visual structures and synthesize them into precise, executable code. Moreover, current visual code generation benchmarks often rely on simplified layouts within single programming environments, falling short of evaluating true unified multimodal reasoning. To bridge this gap, we propose FigCodeBench, a comprehensive framework for rigorously evaluating MLLMs on figure reproduction, integrating multimodal comprehension and generation. We first design a systematic dataset construction pipeline, resulting in a total of 6,194 instances that cover 7 functional categories and 4 types of programming languages (Python, Matlab, R, and Latex). We further categorize figure reproduction into three tiers (Easy, Medium, Hard) with visual and code complexity modeling, specifically targeting complex structural reasoning, varying aspect ratios, and dense geometric constraints. We introduce a multi-dimensional evaluation protocol, encompassing visual fidelity (e.g., PSNR, SSIM, LPIPS) and syntactic isomorphism (e.g., abstract syntax tree

similarity), that aligns highly with the Mean Machine Opinion Score (MMOS) and human preferences. Based on our framework, we conducted extensive experiments on 24 widely used proprietary and open-source MLLMs (e.g., Gemini 3.1 Pro, GPT-5.4, and Kimi-K2.5), where we observed a universal, non-linear performance clif across diferent programming languages and dificulty scenarios for all models, and gained several insights, such as the significant metric decline in rigid declarative languages. Our benchmark ofers (1) a comprehensive dataset and multi-dimensional evaluation pipeline; (2) an up-to-date leaderboard on MLLM figure reproduction proficiency; and (3) a nuanced understanding of the perceiving and reasoning bottlenecks hindering unified comprehension and generation in current MLLMs.

Keywords: Multimodal Large Language Models, Figure Reproduction, Image Understanding, Code Generation, Benchmark

## 1 Introduction

Multimodal Large Language Models (MLLMs) have demonstrated extraordinary advances in image understanding and code generation. Current models (Xu et al. 2025; Qwen Team 2026a; Zeng et al. 2026; xAI 2026; OpenAI 2026d) can now efortlessly complete Visual Question Answering (VQA) (Liu et al. 2024) and achieve near-perfect accuracy on visual math problems (Wang et al. 2024a), coding challenges (Chen et al. 2021; Jimenez et al. 2024a), and strong performance on more recent multimodal reasoning benchmarks such as RealWorldQA (xAI 2024) and OmniDocBench 1.5 (Ouyang et al. 2025). Recent releases touted as vision-language breakthroughs, such as Gemini 3.1 Pro (Google 2026a) and Qwen3.8-Max (Qwen Team 2026c), have pushed performance towards the upper limits of these benchmarks. For example, Qwen3.8-Max achieves over 95% accuracy on MathVision (Wang et al. 2024a) and over 92% pass@1 on LiveCodeBench v6 (Jain et al. 2025).

Yet, existing evaluations predominantly operate within isolated paradigms that evaluate modalities in a disjointed manner, failing to assess the true unification of comprehension and generation. On one hand, traditional code generation benchmarks (Jain et al. 2025; Zheng et al. 2025) rely exclusively on textual specifications as input, neglecting the reality that real-world programming is inherently a multimodal endeavor, where developers frequently reference visual materials. On the other hand, existing multimodal evaluations (Yue et al. 2025; Fu et al. 2026; Zhang et al. 2025b) are largely confined to perceptual comprehension, which is typically assessed through VQA or multiple-choice formats. They fall short of validating a model’s capacity for further structural representation, leaving it unknown whether an MLLM can accurately synthesize its visual understanding into complex, executable generative outputs. Benchmarks like CharXiv (Wang et al. 2024c) ofer visual chart comprehension challenges, but sufer from passive question-answering, rely on rigid text-matching metrics susceptible to superficial guessing, lack executable validation, and the inability to verify active figure reproduction.

![](images/bb749c79f0fa9db6a627807cf329c16ed12d015cf72fecfe1b2cc3e9ed51a7c2.jpg)  
Fig. 1 FigCodeBench leaderboard. Left: Mean Machine Opinion Score (MMOS) versus average cost per problem for various models. Right: Performance comparison of six representative models on diferent image categories.

To bridge this gap, some pioneering eforts have begun to explore cross-modal visual code generation, tasking models with translating images into executable scripts. Plot2Code (Wu et al. 2025a), ChartCoder (Zhao et al. 2025), and ChartMimic (Yang et al. 2025) utilize information-intensive visual charts and textual instructions as inputs, requiring MLLMs to generate the corresponding code for chart rendering. Design2Code (Si et al. 2025) assesses how well MLLMs can generate code that renders into webpages matching given reference screenshots. BioMotion Arena (Chen et al. 2025b) draws inspiration from the remarkable capacity of human vision to perceive biological motions and prompts MLLMs to generate code for human animation that satisfies the combination of action requirements. However, these benchmarks predominantly focus on overly simplified, single-language data plots or front-end web development (e.g., HTML/CSS rendering). They inherently lack the mathematical rigor, dense geometric constraints, and multi-paradigm complexity required for professional scientific figure reproduction. Furthermore, their evaluation protocols often rely on unidimensional visual similarity or heuristic script matching, failing to capture the true structural fidelity across diverse programming environments.

To this end, we introduce FigCodeBench, a visual programming benchmark for figure reproduction, which challenges MLLMs to generate the corresponding code for reproducing a given figure. FigCodeBench is characterized by its (1) real-world fidelity, (2) more programming languages, (3) diverse figure types, and (4) multi-dimensional evaluation metrics. Specifically, prior benchmarks predominantly focus on simplistic charts (e.g., standard bar or pie graphs), while FigCodeBench expands the horizon to complex scientific figures in diferent coding scenarios. Our benchmark comprehensively evaluates the model’s ability to synthesize intricate typographic layouts, geospatial projections, mathematical vector fields, flowcharts, and other abstract diagrams, moving beyond simple chart reading to holistic code-driven scientific visualization. Through the collection of academic documents and online repositories, we identify four common programming languages, i.e., Python, Matlab, R, and Latex, as well as 7 functionfocused figure subcategories. Subsequently, we manually collect a total of 6,194 highquality figure-code pairs for these types with human-in-the-loop augmentation to mitigate data contamination. Furthermore, we establish a suite of evaluation metrics from both image-oriented and code-oriented perspectives to thoroughly assess the performance of MLLMs. We also propose to extract the visual quality achieved relative to the amount of code required as a joint fidelity-eficiency metric that quantifies the reproduction eficiency.

We evaluate a suite of frontier models on FigCodeBench, including proprietary models such as Gemini 3.1 Pro, GPT-5.4, Claude Opus 4.7, and open-source models such as Kimi-K2.5 and Qwen3.5-122B (Fig. 1). We observe a significant performance gap across diferent programming languages, where rendering via descriptive Latex constitutes a formidable bottleneck for all evaluated models. In-depth experiments on diferent image types further reveal that current MLLMs lack the 2D spatial reasoning capabilities required for code-based object depiction and the representation of abstract set relationships, once detached from a regular grid coordinate system. Meanwhile, we find a significant divergence in the coding strategies adopted by diferent MLLMs, highlighting that few models can strike a perfect balance between precise visual comprehension and reliable code generation. We also compare reasoning models with their non-reasoning counterparts and identify the impact of model size in handling such problems. Correlation analysis demonstrates a high correlation between our adopted metrics and human evaluation, validating the efectiveness of these metrics. Further case study reveals that although current MLLMs have largely mastered basic data-togeometry mapping, they still exhibit substantial weaknesses in fine-grained aesthetic rendering.

Overall, our contributions can be summarized as follows:

• This paper introduces FigCodeBench, a comprehensive framework designed for evaluating the unified visual comprehension and code generation capabilities of MLLMs, consisting of a large-scale figure reproduction dataset and a multiperspective evaluation protocol.

• We design a systematic dataset construction pipeline resulting in 4 programming languages and 7 functional categories with 6,194 multimodal figure-code pairs, alongside a multi-dimensional metric system that provides reliable assessments highly aligned with both machine judgments and human evaluations.

• We conduct large-scale experiments on 11 proprietary and 13 open-source MLLMs, revealing that existing models still face substantial challenges in figure reproduction tasks. We observe widespread performance clifs across scenarios and derive several insights into the perceptual and generative bottlenecks of current MLLMs.

## 2 Related Work

## 2.1 Multimodal Large Language Models

Multimodal Large Language Models (MLLMs) (Hurst et al. 2024; Li et al. 2024b; Chen et al. 2024c; Team et al. 2023, 2024; Wang et al. 2024b; Google 2025a) have demonstrated remarkable performance across a broad spectrum of tasks, evolving from foundational visual perception to complicated multimodal reasoning across images, video, and 3D space (Chen et al. 2025f; Fei et al. 2024; Wu et al. 2025b). The integration of the visual capabilities of vision transformers (e.g., QwenViT (Bai et al. 2025b)) and the generative capabilities of LLMs enables the seamless bridging of raw visual perception with high-level cognitive reasoning, allowing models to interpret complex visual scenes and follow open-ended instructions. Notably, in the AI for Science (AI4S) domain, the Intern-S1 series (Bai et al. 2025a; Zou et al. 2026) equips an MoE LLM with a vision encoder, a time-series encoder, and a dynamic tokenizer that switches the tokenization and embedding strategies for natural language and scientific inputs, achieving leading results across scientific domains (e.g., chemistry, materials, life-science, earth, etc.). In the medicine domain, Med-MLLM (Liu et al. 2023a) supports medical data across visual modalities (e.g., chest X-ray and CT) and textual modalities (e.g., medical reports and free-text clinical notes), thereby facilitating downstream clinical tasks. For more routine user interface (GUI) automation control, CoCo-Agent (Ma et al. 2024) and MP-GUI (Wang et al. 2025c) enhance GUI perception across diferent aspects and granularities, including screenshots and detailed layouts for the visual channel, and historical actions for the textual channel. Recently, building upon the rapid advancements in the understanding and reasoning capabilities of unimodal large models, researchers have further extended MLLMs to audio (Cappellazzo et al. 2025; Hu et al. 2026), code (Cao et al. 2026; OpenAI 2026a), action (Qi et al. 2025; Szot et al. 2024), and even Omni modalities (Xu et al. 2025; Jiang et al. 2025), which further expanded the application scope of MLLMs.

## 2.2 Evaluation of MLLMs

Early multimodal benchmarks, such as VQA v2 (Goyal et al. 2017), LVLM-eHub (Xu et al. 2024), MME (Fu et al. 2025), and Seed-Bench (Li et al. 2024a), primarily focus on Visual Question Answering (VQA). Later, a greater number of visual-language tasks have emerged to assess MLLMs’ capabilities across various dimensions, including Optical Character Recognition (OCR) (Liu et al. 2023b), chart understanding (Masry et al. 2022; Wang et al. 2024c), abstract visual reasoning (Chen et al. 2025b; Zhang et al. 2025a), long video reasoning (Wu et al. 2024; Wang et al. 2025b), scientific and multidisciplinary tasks (Zhou et al. 2025; Yue et al. 2025; Chen et al. 2025a,c; Rong et al. 2025; Zheng et al. 2026). In contrast, our FigCodeBench bridges the evaluation of the perception capabilities of MLLMs with the code generation abilities of their language components, enabling a cross-modality evaluation that aligns with the practical capabilities required in actual scientific analysis workflows.

## 2.3 Code Generation Benchmarks

Natural language to code generation has gained growing attention recently, with a proliferation of code-specific models (Fakhoury et al. 2024; Cao et al. 2026; OpenAI 2026c; Inc 2026), necessitating the development of more comprehensive and diverse evaluation benchmarks. For example, HumanEval (Chen et al. 2021), HumanEval-XL (Peng et al. 2024), and DSCodeBench (Ouyang et al. 2026) focus on code generation for various programming languages. LiveBench (White et al. 2025) assesses a model’s ability to parse a coding competition question statement and write a correct answer, with completion ability concerned. LiveCodeBench (Jain et al. 2025) expands the usage scenarios of code LLMs to code generation, self-repair, code execution, and test case output prediction. LiveCodeBench Pro (Zheng et al. 2025) further elevates code problems to human grandmaster levels, featuring sophisticated algorithmic reasoning. SWE-Bench (Jimenez et al. 2024b), on the other hand, focuses more on assessing software engineering scenarios for code maintenance rather than algorithm design. However, these benchmarks predominantly serve LLMs, leaving MLLMs insuficiently scrutinized regarding the intricate synergy between visual grounding and code generation.

Table 1 FigCodeBench statistics for annotated tags with corresponding functional-focus taxonomy. Breakdown by the number of figure plotting topics, query image counts, percentages, average code length (tokens calculated by the LLama3 tokenizer), and dificulty scores. The slash separates statistics of exemplary data (left) and user-generated data (right).
<table><tr><td>Language</td><td>Package</td><td>Tag</td><td># Prototype</td><td># Query Imgs</td><td>Pct.%</td><td>Code Len.</td><td>Difficulty</td></tr><tr><td rowspan="6">Python</td><td rowspan="6">Matplotlib &amp;Seaborn</td><td>Statistical</td><td>56/51</td><td>168/148</td><td>9.72/3.31</td><td>379.7/772.4</td><td>[30.52,60.25]/[2.42,61.69]</td></tr><tr><td>Relational</td><td>29/44</td><td>87/131</td><td>5.03/2.93</td><td>337.3/625.2</td><td>[31.35,60.92]/[36.72,64.56]</td></tr><tr><td>Temporal</td><td>17/34</td><td>51/97</td><td>2.95/2.17</td><td>377.1/514.2</td><td>[38.09,52.42]/[16.76,59.65]</td></tr><tr><td>Compositional</td><td>9/8</td><td>27/24</td><td>1.56/0.54</td><td>305.9/485.1</td><td>[39.13,52.50]/[43.46,52.31]</td></tr><tr><td>Geospatial</td><td>8/1</td><td>24/3</td><td>1.39/0.07</td><td>338.6/566.3</td><td>[34.39,55.02]/[44.87,50.88]</td></tr><tr><td>Mathematical</td><td>37/39</td><td>111/116</td><td>6.42/2.60</td><td>376.2/496.1</td><td>[32.40,55.47]/[41.84,60.91]</td></tr><tr><td rowspan="6">Matlab</td><td rowspan="6">Built-in</td><td>Statistical</td><td>8/33</td><td>24/98</td><td>1.39/2.19</td><td>508.9/704.5</td><td>[33.33,54.25]/[36.73,57.21]</td></tr><tr><td>Relational</td><td>8/21</td><td>21/77</td><td>1.21/1.72</td><td>482.8/663.0</td><td>[32.38,52.61]/[42.48,56.92]</td></tr><tr><td>Temporal</td><td>12/25</td><td>35/75</td><td>2.02/1.68</td><td>451.2/842.8</td><td>[34.48,51.12]/[38.98,58.41]</td></tr><tr><td>Compositional</td><td>5/11</td><td>13/33</td><td>0.75/0.74</td><td>388.2/1005.4</td><td>[38.80,50.14]/[46.62,61.78]</td></tr><tr><td>Geospatial</td><td>3/4</td><td>9/10</td><td>0.52/0.22</td><td>615.4/193.1</td><td>[41.14,54.01]/[ /[33.18,47.44]</td></tr><tr><td>Mathematical</td><td>35/34</td><td>104/101</td><td>6.02/2.26</td><td>475.7/789.8</td><td>[31.94,55.38]/[43.57,59.08]</td></tr><tr><td rowspan="6">R</td><td rowspan="6">ggplot2</td><td>Statistical</td><td>47/97</td><td>141/289</td><td>8.16/6.47</td><td>310.3/717.9</td><td>[28.08,54.43]/[34.18,63.13]</td></tr><tr><td>Relational</td><td>33/41</td><td>99/122</td><td>5.73/2.73</td><td>341.0/1108.2</td><td>[31.66,53.65]/[36.83,61.15]</td></tr><tr><td>Temporal</td><td>26/43</td><td>78/126</td><td>4.51/2.82</td><td>407.1/1032.1</td><td>[34.03,52.34]/[40.84,72.86]</td></tr><tr><td>Compositional</td><td>13/37</td><td>39/107</td><td>2.26/2.40</td><td>297.8/768.2</td><td>[37.77,52.39]/[38.61,61.25]</td></tr><tr><td>Geospatial</td><td>4/0</td><td>12/0</td><td>0.69/0.00</td><td>181.2/0.0</td><td>[18.68,42.35]/[0.00,0.00]</td></tr><tr><td>Mathematical</td><td>15/24</td><td>45/72</td><td>2.60/1.61</td><td>398.0/613.9</td><td>[34.56,49.27]/[38.39,59.52]</td></tr><tr><td>Latex</td><td>TikZ</td><td>Conceptual</td><td>285/1,345</td><td>641/2,836</td><td>37.07/63.52</td><td>1344.2/1668.6</td><td>[27.83,76.98]/[25.33,73.99]</td></tr><tr><td>Total</td><td></td><td></td><td>651/1,897</td><td>1,729/4,465</td><td>100%/100%</td><td></td><td>[2.42, 76.98]</td></tr></table>

## 3 Benchmark Curation

In this section, we first introduce the design philosophy of FigCodeBench (Sec. 3.1), and elaborate on the data curation process (Sec. 3.2). Then, we quantify the data diversity and dificulty tiers (Sec. 3.3). Afterwards, we define the evaluation taxonomy (Sec. 3.4) and make comparisons with existing related benchmarks (Sec. 3.5).

## 3.1 Design Philosophy

FigCodeBench is designed to evaluate MLLMs’ ability to generate the precise programmatic representation of a scientific visualization from its visual manifestation, thereby diferentiating their visual perception, coding, and cross-modal reasoning skills. Currently, the lack of systematic, reproducible benchmarks hinders progress in scientific visualization models. Specifically, an input tuple $\mathcal { X } = \{ I , L , A \}$ includes the target reference image I to be reproduced, the target programming language L, and a set of auxiliary instructions A, providing textual context or specific constraints (e.g., dataset or package requirements). Given X , the model $M _ { \theta }$ aims to generate code C by estimating the conditional probability distribution:

$$
C \sim P _ { M _ { \theta } } ( C \mid I , L , A )\tag{1}
$$

Afterwards, the corresponding code compiler maps the generated script C to a rendered image <sup>ˆ</sup>I, which should be semantically and visually consistent with I across multiple fidelity dimensions. Guided by this objective, we construct the benchmark around two core principles.

Comprehensiveness and Fidelity. Unlike existing chart-related MLLM benchmarks (Yang et al. 2024; Wu et al. 2025a; Yang et al. 2025) that aim at Python only, the tasks in FigCodeBench further cover three mainstream programming languages for figure visualization, i.e., Matlab, R, and LaTeX. These languages dominate the landscape of scientific computing and academic publishing, yet they exhibit fundamentally diferent syntactic structures, rendering paradigms, and API logic. By expanding the evaluation scope across these diverse compiler spaces, FigCodeBench rigorously assesses the unified multimodal comprehension and generation abilities of MLLMs, mitigating the risk of evaluating models that merely overfit to the idiosyncratic APIs of a single ecosystem. Furthermore, to guarantee real-world fidelity, the reference figures and user queries in our benchmark are strictly curated from authentic, in-the-wild sources rather than being synthetically generated via predefined templates.

Reproducibility and Determinism. Reproducibility is a core design criterion. Each task is anchored by an explicit visual outcome that serves as the basis for evaluation. The execution workflows for all four programming spaces are encapsulated within a containerized sandbox environment. By strictly fixing random seeds and versioncontrolling the backend rendering engines and their auxiliary libraries, we ensure a one-to-one mapping between a given script and its generated figure and avoid stochastic model behavior. Meanwhile, FigCodeBench is designed with a robust, multidimensional evaluation framework that equips objective measurements and a multimodal LLM jury mechanism to ensure highly reproducible, fair, and definitive assessments. Beyond visual and semantic accuracy, we incorporate understanding-generation eficiency as a fundamental facet of our evaluation protocol, which is assessed through execution cost, including money, token expenses, and consistency measures, to quantify variability and facilitate repeated and reproducible evaluations across the research community.

## 3.2 Data Collection Pipeline

As illustrated in Fig. 2, constructing FigCodeBench progresses through four distinct stages: figure-code scraping, diversity perturbation, auxiliary information extraction, and quality control. Here, we provide an overview.

Data Sources. We collect source figures and their generating codes from arXiv, peer-reviewed scientific publications, professional visualization galleries (e.g., Matplotlib, Seaborn, TikZ, PGFPlots, and ggplot2), and open-source repositories (e.g., GitHub and Reddit) that hold a CC BY 4.0 license, resulting in an initial collection of over 4,100 figures. The collected instances exhibit the genuine complexities of professional scientific communication. This includes complex multi-axis layouts, specific visual representations, sophisticated aesthetic configurations (e.g., customized legends, nonstandard aspect ratios, and perceptually uniform color schemes), and dense textual annotations.

![](images/0c78021f15325e6b2bfd810a290860fedffdc09a308eabf66fb6f444fbbcb01b.jpg)  
Fig. 2 Overview of FigCodeBench. Upper: The construction pipeline of FigCodeBench includes source data collection, attribute-guided perturbation, auxiliary information extraction, and quality control. Bottom left: This gallery displays 16 representative cases, organized into seven functional categories, i.e., statistical, relational, temporal, compositional, geospatial, mathematical, and conceptual visualizations. We summarize the key features of this benchmark in the bottom right part.

Function-focused Taxonomy and Tags. Instead of categorizing figures solely by their morphological appearance, we identify 7 functional categories that span the common requirements of scientific communication, i.e., statistical (e.g., distributions and boxplots), relational (e.g., scatter and radar charts), temporal (e.g., time-series and area plots), Compositional (e.g., pie and tree maps), geospatial (e.g., map projections), mathematical (e.g., 3D surfaces and vector fields), and conceptual (e.g., flowcharts and various patterns). Furthermore, each sample is annotated with the programming space (i.e., Python, LaTeX, R, and Matlab), the expertise level of the source code (i.e., exemplary and user-generated), and inherent complexity metrics, serving a critical diagnostic purpose to decouple baseline API memorization from authentic compositional generalization. We list the distribution of each tag in Tab. 1.

Attribute-guided Perturbation. To mitigate the risk of data contamination, we augment the seed instances through controlled, attribute-guided perturbations. We develop an automated Abstract Syntax Tree (AST) parser and regex-based pipeline to isolate and modify specific semantic or stylistic attributes, such as axis, color, themes, legend, annotation, background, and fonts, within the source scripts, generating figure-code variants. We also invited five scientific researchers to manually contribute instances with personal preferences through an interactive graphical user interface (GUI) for human-guided curation. For each seed figure-code pair, we create 2 additional augmented pairs. Ultimately, this perturbation strategy enhances our dataset, yielding over 12,000 pairs that reflect a wide range of realistic and practical chart use cases.

![](images/834adc28a2c9125cddb69d0003612fc1751edf684f4b1732fcb5b22ba863fca3.jpg)  
Fig. 3 Scatter plot and marginal distribution of the image-code complexity of FigCodeBench. The red dotted lines separate the easy ([2.42, 44.73]), medium ([44.73, 50.05]) and hard tiers [50.05, 76.98]. We also visualize some image-code cases (points that are neither zero-sum nor on the coordinate axes) at the corresponding minimum and maximum dificulty tiers.

Auxiliary Information Extraction. Beyond the source code, ensuring absolute reproducibility requires isolating the execution from environmental variance. To guarantee that the generation process is hardware-agnostic and strictly determinis tic, we extract and explicitly anchor a comprehensive set of auxiliary information for each instance. First, we identify and lock explicit dependencies, including third-party packages, library versions, and random seeds. More importantly, we resolve implicit rendering ambiguities by specifying standard graphic backends (e.g., applying a specific backend in Matplotlib to prevent interactive rendering collapse), enforcing uniform canvas geometries (e.g., DPI and aspect ratio configurations), and standardizing font assets to prevent layout corruption caused by Operating System (OS)-level font fallbacks.

Quality Control and Filtering. In this phase, we deploy a two-stage quality control and filtering process. First, all scripts are injected into isolated, predefined sandbox environments encompassing standardized compiler toolchains. Any instance that triggers compilation errors (e.g., image-saving errors caused by interactive maps), missing dependencies, or runtime exceptions is automatically pruned. Besides, we update the code blocks that use local data with simulated data to avoid errors stemming from unresolvable file I/O operations within the sandbox environment. Second, we implement a human-in-the-loop inspection protocol. Domain experts systematically review the rendered images to identify and eliminate pathological visual artifacts that automated scripts might overlook, such as severe textual overlap, axis clutter, large areas of blank space, pure background, or disproportionate aspect ratios, guaranteeing that every reference figure in FigCodeBench represents a flawless, having scientific significance standard. This process results in a refined collection of 6,194 figure-code pairs.

## 3.3 Data Statistics and Division

As depicted in Tab. 1, FigCodeBench encompasses a total of 6,194 instances, covering 4 types of programming languages and 7 functional categories. Given the variety of their internal combination elements, we treat 2,548 unique instances as prototypes. We provide a detailed breakdown of dataset composition, including the number of query images, category proportions, and tokenized code lengths, for both exemplary and user-generated samples. Notably, the latter features longer code sequences and increased complexity, exhibiting the diversity of FigCodeBench. Note that we do not deliberately impose sample uniformity, but rather reflect the usage frequency of diferent programming languages in specific scenarios.

Dificulty Tiers. To streamline navigation and ensure diversity, we design a figurecode complexity rating heuristic for dificulty tiers. We quantify the intrinsic dificulty of each benchmark instance from two complementary perspectives, $\mathrm { i . e . }$ , the visual complexity of the target figure and the structural complexity of its reference code. We characterize the visual complexity of the figure using five features, including edge density, line-orientation diversity, color diversity, number of foreground components, and foreground spatial occupancy:

• Edge Density: Edge density is defined as $F _ { \mathrm { e d g e } } = N _ { \mathrm { e d g e } } / N _ { \mathrm { v a l i d } }$ , where $N _ { \mathrm { e d g e } }$ is the number of Canny operator detected edge pixels and $N _ { \mathrm { v a l i d } }$ is the number of valid, non-padding pixels. A higher edge density generally indicates that the figure contains more curves, axes, text, markers, or other graphical structures.

• Line-Orientation Diversity: We compute horizontal and vertical image gradients using Sobel filters and evaluate their orientations only at Canny edge pixels. Gradient orientations are treated as unsigned line orientations and quantized into 18 equally spaced bins. Let $p _ { b }$ denotes the proportion of edge pixels assigned to orientation bin b. The normalized orientation entropy is $F _ { \mathrm { o r i e n t a t i o n } } =$ $\textstyle - \sum _ { b = 1 } ^ { 1 8 } p _ { b } \log p _ { b } / \log 1 8$ . The resulting value lies in [0, 1]. A figure dominated by parallel or similarly oriented lines produces a low score, whereas a figure containing horizontal, vertical, curved, and oblique structures produces a high score.

• Color Diversity: Color diversity is measured using the entropy of a quantized CIELAB color histogram. The CIELAB space is divided into $8 \times 8 \times 8$ bins, giving $B = 5 1 2$ possible color bins. Let $p _ { b }$ denote the proportion of foreground pixels falling into color bin b. We define $\begin{array} { r } { \dot { F } _ { \mathrm { c o l o r } } = - \sum _ { b = 1 } ^ { \tilde { B } } } \end{array}$ p<sub>b</sub> log $p _ { b } / \log B$ . Consequently, a monochromatic line plot receives a lower score than a figure containing multiple colored curves, regions, or heatmaps (Rosenholtz et al. 2007).

• Number of Foreground Components: We construct a foreground mask by comparing each valid pixel with the estimated background color $M ( x , y ) =$ $\mathbf { 1 } \left[ \Delta E ( \bar { I } ( x , y ) , C _ { \mathrm { b g } } ) > \tau \right]$ , where $\Delta E$ denotes the color diference in CIELAB space and the initial threshold is set to $\tau = 5 . \mathrm { ~ A ~ 3 ~ } \times 3$ morphological closing operation is subsequently applied to connect small discontinuities, and connected regions containing fewer than 20 pixels are removed as noise. Let $N _ { \mathrm { c o m p o n e n t } }$ denote the number of remaining foreground connected components. The componentcomplexity feature is defined as $F _ { \mathrm { c o m p o n e n t } } = \log \left( 1 + N _ { \mathrm { c o m p o n e n t } } \right)$ , providing a representation of spatially separated image regions (Rosenfeld and Pfaltz 1966), where a larger number of components generally indicates more independent graphical elements, such as markers, text glyphs, annotations, and legend.

• Foreground Spatial Occupancy: The spatial occupancy of a figure is defined as $F _ { \mathrm { o c c u p a n c y } } = N _ { \mathrm { f o r e g r o u n d } } / N _ { \mathrm { v a l i d } } ,$ where $N _ { \mathrm { f o r e g r o u n d } }$ is the number of foreground pixels in the cleaned mask and $N _ { \mathrm { v a l i d } }$ excludes padded pixels. This feature distinguishes visually sparse figures from figures whose graphical elements occupy a large fraction of the canvas.

To make the extracted visual features comparable across figures with diferent native resolutions, each ground-truth image is resized such that its longer side is 1,024 pixels while preserving the original aspect ratio. The resized image may be padded to a 1024 × 1024 canvas, but padded pixels are excluded from all subsequent measurements. For images containing transparency, the alpha channel is composited over a white background. We estimate the dominant background color, denoted by $C _ { \mathrm { b g } }$ , as the component-wise median of the CIELAB values of the pixels along the image boundary. Compared with using a fixed white background, this procedure is more robust to ofwhite canvases and other background colors. The overall image-complexity score $D _ { \mathrm { i m a g e } }$ is then defined as the averaged values of the five features and mapped to [0, 100].

Before extracting code features, we remove comments, blank lines, import statements, package-loading commands, and automatically generated boilerplate. Large literal data blocks and long coordinate arrays are also excluded. These elements may substantially increase file length without increasing the complexity of the graphical construction procedure. The resulting representation is intended to quantify plotting logic rather than embedded data volume. Here, we describe code complexity using three feature groups, including code size, Abstract Syntax Tree (AST) structure, and plotting-API usage.

• Code Size: We measure the number of efective lexical tokens as the code size $F _ { \mathrm { s i z e } } = \log { ( 1 + N _ { \mathrm { e f f . ~ t o k e n } } ) }$ . Efective tokens include function calls, operators, control-flow statements, parameters, and variable references.

• AST Structural Complexity: The abstract syntax tree provides a languageaware representation of the syntactic organization of a program. We use both the total number of AST nodes and the maximum AST depth, i.e., $F _ { \mathrm { A S T } } =$ 0.7 log $( 1 + N _ { \mathrm { A S T ~ n o d e } } ) + 0 . 3 D _ { \mathrm { A S T ~ d e p t h } }$ . We assign a larger weight to node count as it summarizes structural information over the complete syntax tree, whereas maximum depth depends only on the deepest root-to-leaf path and may be disproportionately afected by a single nested expression (Zuse 2019). We parse Python programs using the built-in ast module. Matlab, R, and LaTeX programs are parsed using the corresponding language-specific parser.

• Plotting-Function Calls: We also count function or method calls that directly afect graphical output. These include the creation of graphical primitives, axes, annotations, legends, color mappings, and figure layouts, but exclude utility functions unrelated to rendering. The plotting-operation feature is computed as $F _ { \mathrm { A P I } } = \log \left( 1 + N _ { \mathrm { A P I ~ c a l l s } } \right)$

Finally, we perform equal-weighted averaging to compute the code complexity score, and the final dificulty score is $D _ { \mathrm { o v e r a l l } } = 0 . 5 D _ { \mathrm { i m a g e } } + 0 . 5 D _ { \mathrm { c o d e } }$ . Finally, instances below the 33rd percentile of $D _ { \mathrm { o v e r a l l } }$ are assigned to the Easy tier, those between the 33rd and 67th percentiles to the Medium tier, and the remaining instances to the Hard tier. A valid ranking scheme should exhibit monotonically decreasing execution rates and perceptual similarity from the Easy to Hard tiers (Sec. 4.7). Fig. 3 shows the overall distributions and provides examples of image-code pairs at diferent dificulty tiers.

## 3.4 Evaluation Protocol

FigCodeBench incorporates multi-level evaluators that yield deterministic and reproducible assessment signals from both image-oriented and code-oriented perspectives by grounding evaluations in executable checks, quantitative, and qualitative measurements.

Image-Oriented Metrics. To evaluate the visual equivalence between the rendered output of a generated script $\hat { I }$ and its ground-truth visualization $I ,$ we construct a multi-scale visual assessment suite that spans low-level pixel fidelity, structural layout integrity, and high-level human perception. At the pixel and structural level, we employ Peak Signal-to-Noise Ratio (PSNR) to measure absolute reconstruction fidelity, and the Structural Similarity Index Measure (SSIM) (Wang et al. 2004) to capture spatial alignment and structural continuity. To align with human perceptual judgment, we integrate Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al. 2018) alongside the CIEDE2000 color diference formula (∆E) (Luo et al. 2001), which strictly quantifies global chromatic shifts in a perceptually uniform color space. Crucially, to prevent models from inflating visual metrics through a small sample of fortuitously compiled scripts, the final aggregate score is scaled as an execution-rate-weighted average, explicitly penalizing models with low compilation success rates.

Code-Oriented Metrics. In addition to the execution rate, we evaluate the generated code from lexical, structural, and plotting-semantic perspectives. Since visually equivalent figures can be produced by substantially diferent programs, no single reference-based code metric provides a complete characterization of code quality. We therefore adopt CodeBERTScore (Zhou et al. 2023), CrystalBLEU (Eghbali and Pradel 2022), normalized AST similarity, and plotting-API F1 as complementary diagnostic metrics. CodeBERTScore is used to measure the contextual semantic correspondence in Python only, due to the fact that it does not provide validated language-specific models for Matlab, R, or LaTeX. Thus, we further compute CrystalBLEU for all four languages to quantify lexical correspondence, which excludes shared n-grams that occur frequently across unrelated programs, such as common import statements, figure-initialization routines, and standard rendering commands, preventing ubiquitous plotting templates from inflating similarity scores. To measure structural correspondence, we compute a normalized AST similarity between the generated and ground-truth code. Specifically, we strip comments and formatting-related artifacts while preserving plotting-function names and arguments critical to visual rendering. Given the normalized syntax trees $T _ { g }$ and $T _ { r }$ of the generated and reference code, respectively, we define

$$
S _ { \mathrm { A S T } } = 1 - \frac { \mathrm { T E D } ( T _ { g } , T _ { r } ) } { | T _ { g } | + | T _ { r } | } ,\tag{2}
$$

where $\operatorname { T E D } ( \cdot , \cdot )$ denotes the unit-cost tree-edit distance and $| T |$ is the number of nodes in tree T. This normalization bounds the score between zero and one, with higher values indicating stronger structural similarity. Unlike token-based metrics, normalized AST similarity captures correspondence in structures, nesting patterns, expressions, and function-call organization while remaining relatively insensitive to variable naming and formatting choices. Finally, we introduce plotting-API F1 to measure whether the generated code invokes the plotting operations required to reconstruct the reference figure. Function calls are first mapped to a language-independent plotting ontology, such as LINE, SCATTER, BAR, AXIS CONFIG, LEGEND, ANNOTATION, SUB-PLOT, and STYLE. For example, matplotlib.pyplot.scatter, Matlab scatter, R geom point, and the TikZ/PGFPlots option only marks are mapped to the canonical SCATTER operation. Let $c _ { g } ( a )$ and $c _ { r } ( a )$ denote the number of occurrences of canonical plotting operation a in the generated and reference code. Their multiset intersection is

$$
I _ { \mathrm { A P I } } = \sum _ { a } \operatorname* { m i n } ( c _ { g } ( a ) , c _ { r } ( a ) ) .\tag{3}
$$

Plotting-API precision, recall, and F-score are then defined as

$$
P _ { \mathrm { A P I } } = \frac { I _ { \mathrm { A P I } } } { \sum _ { a } c _ { g } ( a ) } , \qquad R _ { \mathrm { A P I } } = \frac { I _ { \mathrm { A P I } } } { \sum _ { a } c _ { r } ( a ) } , \qquad F _ { \mathrm { A P I } } = \frac { 2 P _ { \mathrm { A P I } } R _ { \mathrm { A P I } } } { P _ { \mathrm { A P I } } + R _ { \mathrm { A P I } } } ,\tag{4}
$$

which provides a task-specific and interpretable measure of whether the model recovered the essential graphical construction operations.

Mean Machine Opinion Score. Following the successful use of MLLMs for evaluation in text analysis and visual understanding tasks (Chen et al. 2024a; Pu et al. 2025; Chen et al. 2025e,g), we employ the MLLM-as-a-Judge mechanism to evaluate the high-level logic. To avoid the bias caused by a single model’s judgment, we refer to the subjective mean opinion score (MOS) in the field of image quality assessment (Chen et al. 2025d) and adopt the model jury form for rating. Specifically, we input both the ground-truth figure and the generated figure into three MLLMs, and instruct them to output a high-level similarity score ranging from 0 to 100 based on the chart types, layout, text content, data, and style (Yang et al. 2025). After that, we average these scores across MLLM judges to get the overall mean machine opinion score (MMOS). The robustness demonstration of this rating strategy is provided in Sec. 10.

The Figure-Code Fidelity Metric. The above unimodal metrics characterize the visual similarity of the rendered figures and the similarity of the generated code, respectively. Here, we further introduce Figure–Code Fidelity (FCF), a joint fidelity–eficiency metric that evaluates the visual quality achieved relative to the amount of code required. Let $Q _ { i } \in [ 0 , 1 ]$ denote the composite visual-similarity score of the i-th generated figure, and let $\bar { L } _ { i } ^ { \mathrm { g e \bar { n } } }$ and $L _ { i } ^ { \mathrm { g t } }$ denote the efective token counts of the generated and ground-truth code, respectively. We define the relative code-length ratio as $R _ { i } ^ { \mathrm { { l e n } } } = L _ { i } ^ { \mathrm { { g e n } } } / L _ { i } ^ { \mathrm { { g t } } }$ and the corresponding conciseness factor as

$$
C _ { i } ^ { \mathrm { l e n } } = \frac { 1 } { \operatorname* { m a x } ( 1 , R _ { i } ^ { \mathrm { l e n } } ) } .\tag{5}
$$

Table 2 Comparison of our FigCodeBench with other related datasets and benchmarks. “I”, “T”, and “C” denote the image, text, and code modalities, respectively. “EM” denotes the exact match.
<table><tr><td>Dataset/Benchmark</td><td>Source</td><td># Lang.</td><td># Fig. Type</td><td># Test Inst.</td><td>Reso.</td><td>Eval. Format</td><td>Metric</td></tr><tr><td>ChartLlama (Han et al. 2023)</td><td>Synthesized</td><td>1</td><td>10</td><td>458</td><td>704×516</td><td>I+T→T</td><td>GPT Score, EM</td></tr><tr><td>MMCode (Li et al. 2024c)</td><td>Crawl</td><td>1</td><td>12</td><td>263</td><td>534×302</td><td>I+T→C</td><td>Pass rate</td></tr><tr><td>MatPlotBench (Yang et al. 2024)</td><td>Crawl</td><td>1</td><td>13</td><td>100</td><td>849×639</td><td>T→C→I</td><td>GPT Score</td></tr><tr><td>ChartX (Xia et al. 2025)</td><td>Synthesized</td><td>1</td><td>18</td><td>6,000</td><td>1176×819</td><td>I+T→T&amp;C→I</td><td>Multiple</td></tr><tr><td>Plot2Code (Wu et al. 2025a)</td><td>Crawl</td><td>2</td><td>6</td><td>368</td><td>903×664</td><td>I+T→C→I</td><td>Multiple</td></tr><tr><td>Design2Code (Si et al. 2025)</td><td>Crawl</td><td>1</td><td>HTML</td><td>484</td><td>1280×1482</td><td>I+T→C→I</td><td>Multiple</td></tr><tr><td>ChartMimic (Yang et al. 2025)</td><td>Crawl+Synthesized</td><td>1</td><td>22</td><td>2,400</td><td>1124×782</td><td>I+T→C→I</td><td>Multiple</td></tr><tr><td>FigCodeBench (Ours)</td><td>Crawl+Synthesized</td><td>4</td><td>30</td><td>6,194</td><td>2022×1552</td><td>I+T→C→I</td><td>Multiple</td></tr></table>

The instance-level FCF score is then defined as

$$
\mathrm { F C F } _ { i } = E _ { i } Q _ { i } C _ { i } ^ { \mathrm { l e n } } = E _ { i } \frac { Q _ { i } } { \operatorname* { m a x } \left( 1 , L _ { i } ^ { \mathrm { g e n } } / L _ { i } ^ { \mathrm { g t } } \right) } ,\tag{6}
$$

where $E _ { i }$ is an execution indicator that equals one when the generated code is successfully executed and produces a valid figure, and zero otherwise. The reference-relative normalization controls for diferences in intrinsic figure complexity and language-specific verbosity. Consequently, the FCF score remains bounded by the underlying visualsimilarity score and reaches a high value only when the generated program is both visually faithful and concise.

Since the four image-oriented metrics operate on heterogeneous numerical scales, we transform each metric into a dimensionless similarity score for which a larger value consistently indicates greater visual fidelity. Specifically, we define the normalized PSNR similarity as

$$
q _ { i } ^ { \mathrm { P S N R } } = 1 - 1 0 ^ { - \mathrm { P S N R } _ { i } / 2 0 } .\tag{7}
$$

For SSIM, we retain the original score without further transformation, while converting LPIPS into a bounded similarity score using an inverse transformation:

$$
q _ { i } ^ { \mathrm { L P I P S } } = \frac { 1 } { 1 + \mathrm { L P I P S } _ { i } } .\tag{8}
$$

For the CIEDE2000 diference, we adopt the Just Noticeable Diference (JND)-based nonlinear mapping proposed by (Yang et al. 2012), where $\Delta E$ is converted into an objective score conforming to subjective color perception:

$$
q _ { i } ^ { \Delta E } = \frac { \mathcal { I } \left( \overline { { \Delta E } } _ { i } \right) } { 5 } .\tag{9}
$$

At last, we aggregate them using an equally weighted geometric mean, preventing an exceptionally high score on one dimension from completely ofsetting a substantial discrepancy on another:

$$
\mathrm { Q } _ { i } = \left( q _ { i } ^ { \mathrm { P S N R } } q _ { i } ^ { \mathrm { S S I M } } q _ { i } ^ { \mathrm { L P I P S } } q _ { i } ^ { \Delta E } \right) ^ { 1 / 4 } .\tag{10}
$$

![](images/a823636846604509ca7b098a329cc0b1928646c7f6a8a0a9da91cc7afdc43703.jpg)  
(a) ChartLlama

![](images/893f4c7c89a8cbd934dddda2e7f28c2c9f084065c404f4afb53ea01280e64fa9.jpg)

(b) MMCode  
![](images/eed1561232f95b24e293716d7305d86851f4d6348aa6c4f7c2768b869ff3cf03.jpg)  
(d) ChartX

(c) MatPlotBench  
![](images/ab5b3366820a6ccd190031ac1f02be1301507d16479095dce8a71de137e274f8.jpg)  
(e) Plot2Code

![](images/ffec161dc7f39669308b7ca961cfd6abba49fe1071b3f10a844c3a014298f5ce.jpg)  
(f) Design2Code

![](images/09a228f2c6156180e502d35e18ad27d2b5ec2e7ed163ad71282bfb74a29db454.jpg)  
(g) ChartMimic

![](images/3c03199ad1802bc390f0aa048d32caaf88fee0c8cb4b59f84c45cf7b43c5cecf.jpg)  
(h) FigCodeBench  
Fig. 4 A gallery of random samples from seven compared benchmarks (i.e., ChartLlama, MMCode, MatPlotBench, ChartX, Plot2Code, Design2Code, and ChartMimic) and our proposed FigCodeBench. FigCodeBench covers 4 diferent languages, 7 categories, and 2,548 prototypes with 6,194 query images for standardized evaluation of MLLMs, significantly surpassing the existing benchmarks in terms of scale and coverage. Zoom-in for better visualization.

![](images/eea76e38fac14ed5b8ab775d68f61193445377a109e64ae73ce6b8f125575d5b.jpg)  
(a) Brightness

![](images/8d6547eb8aa62d66e303e3594d077eb739ab117b4a678c0cd4a6eab946de7ffa.jpg)  
(b) Contrast

![](images/5c7160120dba2da7b873fc4f713b375d4fa6a89f6984e8b50c9bbb30fa02851d.jpg)  
(c) Colorfulness

![](images/91355d6ccc09180fb318ab68476ce39c5e3ed42c7031b972bdf7a4d229cc7a4f.jpg)  
(d) Sharpness

![](images/c3a646c04a49ca3cbe493d9996702b31090a30fc8d7ea3bc188a06f4b2cd555d.jpg)  
(e) SI  
Fig. 5 Feature distribution comparisons among eight benchmarks: ChartLlama, MMCode, MatPlot-Bench, ChartX, Plot2Code, Design2Code, ChartMimic, and our FigCodeBench.

## 3.5 Comparisons with Existing Benchmarks

To further distinguish the diferences between FigCodeBench and the most related works (Han et al. 2023; Xia et al. 2025; Li et al. 2024c; Yang et al. 2024; Wu et al. 2025a; Si et al. 2025; Yang et al. 2025), we summarize their main characteristics in Tab. 2. FigCodeBench distinguishes itself from the following perspectives: (1) Real-world fidelity and less data contamination. Unlike the synthesized data in ChartLlama (Han et al. 2023) and ChartX (Xia et al. 2025) that may deviate from the distribution of real-world data, or raw crawled data in MMCode (Li et al. 2024c) and Design2Code (Si et al. 2025) that may sufer from data contamination, we not only enrich the data sources but also apply both LLM- and human-involved data augmentations to avoid these problems. (2) Covering more programming languages. The existing chartrelated code generation benchmarks mainly focus on Python, while we additionally include Matlab, R, and LaTeX languages. (3) More diverse figure types. FigCodeBench significantly surpasses existing benchmarks in both the diversity of figure types and the total data volume. (4) Hybrid evaluators. Recent benchmarks like ChartLlama (Han et al. 2023), MatPlotBench (Yang et al. 2024), and ChartMimic (Yang et al. 2025) rely heavily on GPTScore-based metrics (Fu et al. 2024), which may raise concerns about the ranking robustness and model bias. In contrast, we aim to evaluate the figure reproduction capabilities of MLLMs by combining deterministic evaluators with an MLLM-as-Judge framework, including image-oriented and code-oriented metrics, to ensure comprehensiveness, accuracy, eficiency, and reproducibility.

![](images/389c79b47747b7c6cdf9e967fa6005e7282f6cbd7a4bd5be9d325dea6f51495d.jpg)

![](images/8864a7192d3636d0cd54fde7eeabe32f85f1e894cb7d8196bcc72b899058e8f7.jpg)  
(a) ChartLlama

![](images/ec1ec75360fc4c0ce95b14d2fcdfda16ae973227081a79f1cee46dd63b2252a0.jpg)

![](images/28ba14157a4adc4d25a61cb1bd87bdda40f42bc0bd11df66e875357ca7e5bb47.jpg)  
(b) ChartMimic

![](images/fc7e40e9b8e00696166b39c987580dbafbb24d4ba0ef15b34e98fbe15f4af115.jpg)

![](images/01e6b882942b121197b9658a76f75a87bbb2b2854d6870e60d51b66f28e1ae83.jpg)  
(c) ChartX

![](images/773a35296d4a00a0d6a1054d06b44a120c277802923350da094dfb7f11662569.jpg)

![](images/38b0a6a43905cd9c6301a2380be328c3b0f32b06686e3b7f2d2e1a665af19165.jpg)

![](images/50bdd9797e778e4c8680e46df0c833a68f662604420661d6bd52d5109197ce15.jpg)

![](images/22f9b2dd67b69726e21358a87a04561929a3d95593702e3e83e17cf15e0ce682.jpg)  
(e) MMCode

(d) Design2Code  
![](images/433e56f53165b63ce01d547f51654426da0a5d00e0fc6c8a064fa2f16fb24edf.jpg)

![](images/cf94e161d613a595815a724f7866be97d8581068e7dcf1653c9d49734230e7be.jpg)  
(f) MatPlotBench

![](images/72164d0f697b26572fbf2beac4706ef3bbea08cb2c92f69c3c38740253b02c4b.jpg)

![](images/98f403e08b36814dce7944542ae2c2e24da6227033ce75724132ce16e9cb90c5.jpg)  
(g) Plot2Code

![](images/f5e800458d47429215459486a2fb39bd0124a16d563c4ef8c2750a572c52afeb.jpg)

![](images/a16fab0a791b488c684276985da2141eebc407c30b9164a2383d96d489712ea5.jpg)  
(h) FigCodeBench  
Fig. 6 Query image (blue ‘x’) distribution in paired feature space with corresponding convex hulls (red boundaries). Left: Brightness (BR)×Contrast (CT), right: Colorfulness (CF)×Sharpness (SR).

Content Diversity and Observations. To characterize the content diversity of the test images in each dataset, we include five low-level features (Tu et al. 2021; Chen et al. 2024b), including brightness, contrast, colorfulness, sharpness, and spatial information (SI), thereby providing a larger visual space in which to plot and analyze content diversity of the eight benchmarks. Fig. 4 provides a thumbnail of the compared benchmarks, and Fig. 5 shows the fitted kernel distribution of each selected feature. We also plotted the convex hulls of paired features to show the feature coverage of each dataset in Fig. 6. We make some observations from the above plots. As seen in Fig. 4 and the corresponding convex hulls in Fig. 6, FigCodeBench and ChartX exhibit similar styles and coverage in brightness and contrast, while the styles of ChartLlama, MMCode, Design2Code, and MatPlotBench are relatively simple and their distribution is uneven. On the sharpness and SI histograms, our FigCodeBench is spread most widely, while Plot2Code is concentrated on lower values. The overall distribution and coverage comparisons demonstrate that FigCodeBench possesses better diversity.

Table 3 List of models evaluated and their respective organizations, cut-of date, and citations.
<table><tr><td>Model Name</td><td>Organizations</td><td>Cut-off Date</td><td>Citation</td></tr><tr><td>gpt-5.4</td><td>OpenAI</td><td>August 2025</td><td>(OpenAI 2026d)</td></tr><tr><td>gpt-5.4-mini</td><td>OpenAI</td><td>August 2025</td><td>(OpenAI 2026d)</td></tr><tr><td>gpt-5.2</td><td>OpenAI</td><td>August 2025</td><td>(OpenAI 2025b)</td></tr><tr><td>claude-opus-4.7</td><td>Anthropic</td><td>January 2026</td><td>(Anthropic 2026)</td></tr><tr><td>gemini-3.1-pro-preview</td><td>Google</td><td>January 2025</td><td>(Google 2026a)</td></tr><tr><td>gemini-3-flash-preview-nothinking gemini-3-flash-preview-thinking</td><td>Google</td><td>January 2025</td><td>(Google 2025b)</td></tr><tr><td></td><td>Google</td><td>January 2025</td><td>(Google 2025b)</td></tr><tr><td>gemini-2.5-flash-nothinking</td><td>Google</td><td>January 2025</td><td>(Google 2025a)</td></tr><tr><td>gemini-2.5-flash-thinking-16384</td><td>Google</td><td>January 2025</td><td>(Google 2025a)</td></tr><tr><td>grok-4.3</td><td>xAI</td><td>December 2025</td><td>(xAI 2026)</td></tr><tr><td>qwen3.6-plus</td><td>Alibaba</td><td>2026</td><td>(Qwen Team 2026b)</td></tr><tr><td>glm-5.1</td><td>Zhipu</td><td>Not specified</td><td>(Zeng et al. 2026)</td></tr><tr><td>glm-4.6v</td><td>Zhipu</td><td>Not specified</td><td>(Hong et al. 2025)</td></tr><tr><td>MiniMax-M3</td><td>MiniMaxAI</td><td>January 2026</td><td>(MiniMax 2026)</td></tr><tr><td>kimi-k2.5</td><td>Moonshot AI</td><td>Not specified</td><td>(Kimi Team et al. 2026)</td></tr><tr><td>doubao-seed-2-0-pro-260215</td><td>ByteDance</td><td>Not specified</td><td>(Bytedance Seed 2026)</td></tr><tr><td>qwen3.5-122b-a10b</td><td>Alibaba</td><td>2026</td><td>(Qwen Team 2026a)</td></tr><tr><td>qwen3.5-35b-a3b</td><td>Alibaba</td><td>2026</td><td>(Qwen Team 2026a)</td></tr><tr><td>qwen3-vl-235b-a22b-instruct</td><td>Alibaba</td><td>March 2025</td><td>(Bai et al. 2025b)</td></tr><tr><td>qwen3-vl-30b-a3b-instruct</td><td>Alibaba</td><td>March 2025</td><td>(Bai et al. 2025b)</td></tr><tr><td>qwen3-vl-8b-instruct</td><td>Alibaba</td><td>March 2025</td><td>(Bai et al. 2025b)</td></tr><tr><td>qwen2.5-vl-72b-instruct</td><td>Alibaba</td><td>June 2024</td><td>(Qwen Team 2025)</td></tr><tr><td>internvl3.5-38b-instruct</td><td>Shanghai AI Lab</td><td>Not specified</td><td>(Wang et al. 2025a)</td></tr><tr><td>internvl3.5-8b-instruct</td><td>Shanghai AI Lab</td><td>Not specified</td><td>(Wang et al. 2025a)</td></tr></table>

## 4 Experiments

## 4.1 Evaluation Setup

Evaluated Models. Our experiments include 24 MLLMs in total, with a mix of frontier proprietary models and open-source models. In particular, for proprietary models, we include models from OpenAI, Anthropic, Google, xAI, and Alibaba, e.g., GPT-5.4 (OpenAI 2026d), GPT-5.2 (OpenAI 2025b), Claude Opus 4.7 (Anthropic 2026), Gemini 3.1 Pro (Google 2026a), Gemini 3 Flash (Google 2025b), Gemini 2.5 Pro (Google 2025a), Grok 4.3 (xAI 2026), and Qwen3.6-Plus (Qwen Team 2026b). For open-source models, we include models from Zhipu, MiniMax, Moonshot AI, ByteDance, Alibaba, and Shanghai AI Lab, such as GLM-5.1 (Zeng et al. 2026), MiniMax-M3 (MiniMax

Table 4 The evaluation resource consumption of FigCodeBench. AvgTok is the average number of tokens generated per query and AvgCost is the approximate \$-cost per query.
<table><tr><td rowspan="2">Model</td><td colspan="2">Python (n=987)</td><td colspan="2">Matlab (n=600)</td><td colspan="2">R (n=1130)</td><td colspan="2">Latex (n=3477)</td></tr><tr><td>AvgTok</td><td>AvgCost</td><td>AvgTok</td><td>AvgCost</td><td>AvgTok</td><td>AvgCost</td><td>AvgTok</td><td>AvgCost</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>GPT-5.4</td><td>814.98</td><td>$0.0117</td><td>946.19</td><td>$0.0122</td><td>898.51</td><td>$0.0119</td><td>1176.65</td><td>$0.0154</td></tr><tr><td>GPT-5.4-Mini</td><td>822.65</td><td>$0.0034</td><td>702.30</td><td>$0.0029</td><td>831.93</td><td>$0.0040</td><td>1075.91</td><td>$0.0043</td></tr><tr><td>GPT-5.2</td><td>792.12</td><td>$0.0088</td><td>951.72</td><td>$0.0110</td><td>830.21</td><td>$0.0103</td><td>1026.63</td><td>$0.0105</td></tr><tr><td>Claude Opus 4.7</td><td>597.10</td><td>$0.0223</td><td>590.41</td><td>$0.0204</td><td>341.74</td><td>$0.0128</td><td>537.90</td><td>$0.0192</td></tr><tr><td>Gemini 3.1 Pro</td><td>580.26</td><td>$0.1222</td><td>619.17</td><td>$0.0887</td><td>643.99</td><td>$0.0899</td><td>861.01</td><td>$0.1739</td></tr><tr><td>Gemini 3 Flash</td><td>558.61</td><td>$0.0031</td><td>695.48</td><td>$0.0030</td><td>629.90</td><td>$0.0032</td><td>822.92</td><td>$0.0035</td></tr><tr><td>Gemini 3 Flash-TK</td><td>640.40</td><td>$0.0254</td><td>720.22</td><td>$0.0226</td><td>655.11</td><td>$0.0239</td><td>804.60</td><td>$0.0323</td></tr><tr><td>Gemini 2.5 Flash</td><td>958.33</td><td>$0.0019</td><td>1230.49</td><td>$0.0019</td><td>765.14</td><td>$0.0024</td><td>1236.69</td><td>$0.0020</td></tr><tr><td>Gemini 2.5 Flash-TK</td><td>844.54</td><td>$0.0200</td><td>1133.81</td><td>$0.0191</td><td>912.07</td><td>$0.0202</td><td>703.56</td><td>$0.0209</td></tr><tr><td>Grok-4.3</td><td>644.26</td><td>$0.0069</td><td>495.09</td><td>$0.0059</td><td>572.09</td><td>$0.0066</td><td>776.76</td><td>$0.0052</td></tr><tr><td>Qwen3.6-Plus</td><td>686.35</td><td>$0.0176</td><td>763.18</td><td>$0.0217</td><td>660.09</td><td>$0.0166</td><td>851.62</td><td>$0.0108</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>GLM-5.1</td><td>1408.62</td><td>$0.0244</td><td>1181.09</td><td>$0.0168</td><td>774.71</td><td>$0.0148</td><td>1080.14</td><td>$0.0163</td></tr><tr><td>GLM-4.6V</td><td>497.02</td><td>$0.0037</td><td>435.42</td><td>$0.0028</td><td>386.01</td><td>$0.0035</td><td>469.69</td><td>$0.0037</td></tr><tr><td>MiniMax-M3</td><td>860.15</td><td>$0.0014</td><td>667.46</td><td>$0.0031</td><td>637.89</td><td>$0.0023</td><td>807.62</td><td>$0.0049</td></tr><tr><td>Kimi-K2.5</td><td>698.12</td><td>$0.0218</td><td>674.08</td><td>$0.0220</td><td>644.64</td><td>$0.0207</td><td>796.87</td><td>$0.0172</td></tr><tr><td>DouBao-Seed-2-0-Pro</td><td>586.81</td><td>$0.0069</td><td>459.35</td><td>$0.0060</td><td>567.29</td><td>$0.0084</td><td>716.40</td><td>$0.0058</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>529.24</td><td>$0.0009</td><td>696.89</td><td>$0.0010</td><td>577.12</td><td>$0.0010</td><td>855.94</td><td>$0.0012</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>742.45</td><td></td><td>875.25</td><td></td><td>681.65</td><td></td><td>841.01</td><td></td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>475.23</td><td>$0.0014</td><td>461.04</td><td>$0.0012</td><td>477.29</td><td>$0.0014</td><td>690.68</td><td>$0.0013</td></tr><tr><td>Qwen3-VL-32B</td><td>968.89</td><td></td><td>1670.08</td><td></td><td>2043.48</td><td></td><td>3440.89</td><td></td></tr><tr><td>Qwen3-VL-8B</td><td>1309.17</td><td></td><td>8301.24</td><td></td><td>2820.2</td><td></td><td>6753.37</td><td></td></tr><tr><td>Qwen2.5-VL-72B</td><td>366.18</td><td></td><td>273.80</td><td></td><td>379.81</td><td></td><td>630.96</td><td></td></tr><tr><td>InternVL3.5-38B</td><td>535.13</td><td></td><td>277.18</td><td></td><td>428.75</td><td></td><td>726.10</td><td></td></tr><tr><td>InternVL3.5-8B</td><td>1089.31</td><td></td><td>2063.26</td><td></td><td>2053.12</td><td></td><td>2798.35</td><td></td></tr></table>

2026), Kimi-K2.5 (Kimi Team et al. 2026), DouBao-Seed-2-0-Pro (Bytedance Seed 2026), Qwen3.5-122B-A10B (Qwen Team 2026a), Qwen3-VL-30B (Bai et al. 2025b), and InternVL3.5-38B (Wang et al. 2025a). We provide detailed information with the corresponding versions, organizations, knowledge cut-of date, and citations for all participating models in Tab. 3.

Implementation Details. Apart from the proprietary models and those with a parameter size greater than 100B that are deployed via API, all other models are run using up to 4 Nvidia H200 141GB GPUs. We use the default parameter settings (e.g., temperature, top k, and top p) of the respective models and do not apply an output token limit for better performance release. Regarding the generation errors caused by the absence of the package or library files, we have updated the local evaluation environment to prevent such occurrences. For GLM-5.1, although the oficial documentation does not explicitly state that it supports multimodal input, we use base64-encoded images as input.

Jury Models. We employ three MLLMs to form the judging panel, i.e., gpt-5.1 (OpenAI 2025a), gpt-5.6-luna (OpenAI 2026b), and gemini-3.5-flash (Google 2026b), which assess the extent to which the generated figure corresponds to the ground-truth figure while mitigating individual opinion bias. The specific prompt templates are presented in App. A. Specifically, we input both the generated and the ground-truth figures into the judge models simultaneously. Then, each model is instructed to evaluate the similarity between the two figures, taking into account five dimensions, including text, layout, type, data, and style. Subsequently, each judge outputs an overall score ranging from 0 to 100 to represent the degree of similarity between the figures. Failed executions are directly assigned a score of 0.

Table 5 Image-oriented reproduction performance on FigCodeBench. PSNR/SSIM (↑) and LPIPS/∆E (↓) represent the pixel-level and perceptual metrics, respectively. “TK” denotes the thinking version. The best and second-best results are in bold and underlined.
<table><tr><td rowspan="2">Model</td><td colspan="3">Python</td><td colspan="3"></td><td colspan="3"></td><td colspan="3">Latex</td></tr><tr><td>Exec.</td><td>PSNR/SSIM</td><td>LPIPS/∆E</td><td>Exec.</td><td>PSNR/SSIM</td><td>LPIPS/ΔE</td><td>Exec.</td><td>PSNR/SSIM</td><td>LPIPS/ΔE</td><td>Exec.</td><td>PSNR/SSIM</td><td>LPIPS/ΔE</td></tr><tr><td colspan="10">Proprietary Models</td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>91.48</td><td>12.807/0.628</td><td>0.411/16.607</td><td>44.67</td><td>5.697/0.307</td><td>0.724/59.897</td><td>62.48</td><td>8.080/0.460</td><td>0.664/44.177</td><td>52.43</td><td>8.151/0.416</td><td>0.668/50.986</td></tr><tr><td>GPT-5.4-Mini</td><td>90.87</td><td>12.545/0.614</td><td>0.467/16.912</td><td>75.00</td><td>9.103/0.500</td><td>0.592/33.133</td><td>78.85</td><td>10.340/0.581</td><td>0.571/28.265</td><td>48.12</td><td>7.764/0.388</td><td>0.702/54.294</td></tr><tr><td>GPT-5.2</td><td>91.38</td><td>12.452/0.614</td><td>0.487/16.459</td><td>68.17</td><td>8.239/0.446</td><td>0.649/39.135</td><td>41.33</td><td>5.248/0.297</td><td>0.786/62.414</td><td>39.69</td><td>6.218/0.315</td><td>0.760/62.473</td></tr><tr><td>Claude Opus 4.7</td><td>99.29</td><td>10.189/0.583</td><td>0.726/17.188</td><td>98.17</td><td>10.205/0.585</td><td>0.701/15.603</td><td>97.35</td><td>10.714/0.689</td><td>0.633/15.794</td><td>88.47</td><td>12.822/0.683</td><td>0.594/18.398</td></tr><tr><td>Gemini 3.1 Pro</td><td>93.61</td><td>13.432/0.648</td><td>0.395/13.504</td><td>82.50</td><td>10.414/0.562</td><td>0.495/25.122</td><td>76.19</td><td>9.934/0.560</td><td>0.587/30.874</td><td>59.76</td><td>9.772/0.486</td><td>0.610/43.148</td></tr><tr><td>Gemini 3 Flash</td><td>91.18</td><td>12.702/0.619</td><td>0.445/16.472</td><td>81.00</td><td>9.939/0.544</td><td>0.536/26.808</td><td>90.44</td><td>12.040/0.677</td><td>0.504/17.161</td><td>64.74</td><td>10.221/0.519</td><td>0.593/38.630</td></tr><tr><td>Gemini 3 Flash-TK</td><td>90.06</td><td>12.495/0.608</td><td>0.459/17.254</td><td>88.33</td><td>11.115/0.595</td><td>0.479/19.864</td><td>93.81</td><td>11.966/0.690</td><td>0.515/15.025</td><td>60.22</td><td>9.511/0.484</td><td>0.620/42.786</td></tr><tr><td>Gemini 2.5 Flash</td><td>87.93</td><td>9.128/0.545</td><td>0.737/26.392</td><td>78.50</td><td>8.439/0.490</td><td>0.748/32.129</td><td>84.78</td><td>9.359/0.615</td><td>0.679/26.564</td><td>24.96</td><td>3.511/0.191</td><td>0.884/76.987</td></tr><tr><td>Gemini 2.5 Flash-TK</td><td>81.03</td><td>8.789/0.502</td><td>0.748/31.137</td><td>76.67</td><td>8.036/0.465</td><td>0.756/33.857</td><td>74.78</td><td>8.044/0.515</td><td>0.735/36.108</td><td>15.65</td><td>2.320/0.124</td><td>0.921/85.388</td></tr><tr><td>Grok-4.3 Qwen3.6-Plus</td><td>90.97</td><td>11.528/0.583</td><td>0.591/19.219</td><td>83.83</td><td>11.003/0.381</td><td>0.605/25.803</td><td>85.49</td><td>11.569/0.646</td><td>0.521/21.817</td><td>68.85</td><td>10.865/0.550</td><td>0.605/35.138</td></tr><tr><td>Open-Source Models</td><td>88.54</td><td>11.806/0.584</td><td>0.540/19.986</td><td>73.17</td><td>8.955/0.491</td><td>0.605/34.289</td><td>87.43</td><td>12.003/0.667</td><td>0.490/19.679</td><td>60.05</td><td>9.694/0.486</td><td>0.630/43.067</td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>GLM-5.1</td><td>83.67</td><td>9.037/0.497</td><td>0.754/29.444</td><td>64.33</td><td>6.550/0.364 = =</td><td>0.813/45.116</td><td>64.16 = = = =</td><td>6.873/0.431</td><td>0.770/45.551</td><td>40.41</td><td>5.839/0.311</td><td>0.808/62.874</td></tr><tr><td>GLM-4.6V</td><td>86.11</td><td>10.580/0.568</td><td>0.706/23.927</td><td>83.83</td><td>8.956/0.470</td><td>0.753/27.531</td><td>86.55</td><td>10.844/0.615</td><td>0.668/22.373</td><td>23.12</td><td>3.332/0.179</td><td>0.893/78.779</td></tr><tr><td>MiniMax-M3</td><td>88.34</td><td>9.125/0.514</td><td>0.748/27.301</td><td>77.50</td><td>7.283/0.451</td><td>0.773/36.782</td><td>76.73</td><td>8.778/0.525</td><td>0.721/33.622</td><td>63.19</td><td>8.556/0.479</td><td>0.722/42.355</td></tr><tr><td>Kimi-K2.5</td><td>92.29</td><td>12.297/0.614</td><td>0.509/16.474</td><td>79.00</td><td>9.338/0.519</td><td>0.579/29.552</td><td>91.68</td><td>12.610/0.699</td><td>0.459/15.764</td><td>48.75</td><td>7.846/0.393</td><td>0.707/53.895</td></tr><tr><td>DouBao-Seed-2-0-Pro</td><td>91.48</td><td>12.070/0.599</td><td>0.547/17.575</td><td>69.83</td><td>8.644/0.464</td><td>0.632/36.932</td><td>86.55</td><td>11.270/0.639</td><td>0.527/21.257</td><td>49.01</td><td>7.675/0.395</td><td>0.707/53.448</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>82.56</td><td>10.635/0.563</td><td>0.649/26.799</td><td>43.00</td><td>4.971/0.281</td><td>0.800/62.103</td><td>69.20</td><td>8.905/0.518</td><td>0.636/37.478</td><td>59.30</td><td>9.621/0.483</td><td>0.646/43.939</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>70.18</td><td>9.236/0.460</td><td>0.665/37.017</td><td>32.33</td><td>3.939/0.219</td><td>0.839/71.110</td><td>81.15</td><td>10.755/0.614</td><td>0.559/26.062</td><td>53.12</td><td>8.699/0.436</td><td>0.677/49.483</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>82.45</td><td>10.216/0.521</td><td>0.646/27.232</td><td>67.17</td><td>7.749/0.428</td><td>0.698/41.487</td><td>85.22</td><td>11.489/0.645</td><td>0.538/22.313</td><td>64.19</td><td>11.001/0.534</td><td>0.614/38.996</td></tr><tr><td>Qwen3-VL-32B</td><td>75.76</td><td>9.319/0.483</td><td>0.677/33.363</td><td>40.33</td><td>4.429/0.245</td><td>0.824/65.247</td><td>69.20</td><td>9.066/0.519</td><td>0.636/37.212</td><td>53.98</td><td>9.074/0.450</td><td>0.677/48.717</td></tr><tr><td>Qwen3-VL-8B</td><td>74.65</td><td>8.733/0.461</td><td>0.710/35.249</td><td>31.00</td><td>3.542/0.193</td><td>0.864/73.063</td><td>63.63</td><td>8.170/0.478</td><td>0.690/42.810</td><td>41.47</td><td>6.732/0.341</td><td>0.752/60.976</td></tr><tr><td>Qwen2.5-VL-72B</td><td>79.51</td><td>9.152/0.506</td><td>0.670/31.460</td><td>59.17</td><td>6.654/0.379</td><td>0.721/48.506</td><td>81.15</td><td>10.912/0.619</td><td>0.574/26.688</td><td>75.87</td><td>12.762/0.633</td><td>0.556/28.190</td></tr><tr><td>InternVL3.5-38B InternVL3.5-8B</td><td>77.69 65.42</td><td>9.242/0.492 7.483/0.409</td><td>0.689/32.572 0.747/43.868</td><td>46.33 37.83</td><td>5.273/0.294 4.246/0.240</td><td>0.790/59.581 0.831/67.195</td><td>73.10 56.90</td><td>9.642/0.557 7.402/0.435</td><td>0.645/34.269 0.730/49.011</td><td>57.89 43.20</td><td>9.063/0.468 6.655/0.347</td><td>0.680/45.645 0.760/59.443</td></tr></table>

## 4.2 Analysis of Evaluation Resource Consumption

Tab. 4 reports the average resource consumption of 24 evaluated MLLMs in Fig-CodeBench. Based on this, we conduct a statistical analysis on the generation verbosity (AvgTok) and monetary overhead (AvgCost) to investigate whether the choice of programming language biases these overheads.

Setup. For the AvgTok analysis, all 24 evaluated MLLMs serve as independent subjects. For the AvgCost analysis, we evaluate 18 API-used models. A preliminary normality check via the Shapiro-Wilk test indicates that both token and cost distributions are heavily right-skewed. This skewness is driven by two distinct factors: (1) severe length hallucinations in specific smaller-scale open-source models (e.g., Qwen3-VL-8B outputting over 8,300 tokens in MATLAB), and (2) orders-of-magnitude diferences in vendor pricing paradigms (e.g., Gemini 3.1 Pro costing significantly more than standard models). To prevent these cross-model variances and outliers from violating parametric assumptions, we employ the non-parametric Friedman rank-sum test, which isolates the intra-model language diferences by ranking the four languages within each model, mathematically neutralizing the confounding efect of varying baseline prices across vendors.

Results. As observed, Latex demands the highest token volume (Median ≈ 831), sequentially followed by Matlab (Median ≈ 699), Python (Median ≈ 692), and R (Median ≈ 649). The Friedman test confirms a statistically significant main efect of the programming language on the generated token length $( \chi ^ { 2 } ( 3 , N = 2 4 ) = 1 9 . 3 5 , p <$ 0.001). Subsequent pairwise post-hoc Nemenyi testing indicates that Latex exhibits a significantly higher AvgTok compared to both R $\left( p < 0 . 0 1 \right)$ and Python $\left( p < 0 . 0 5 \right)$ (Pereira et al. 2015). Meanwhile, the Friedman test on AvgCost reveals a significant main efect $( \chi ^ { 2 } ( 3 , N = 1 8 ) = 1 4 . 8 2 , p < 0 . 0 1 )$ . Ranking analysis confirms that LaTeX incurs the highest evaluation cost across the majority of models. For instance, the AvgCost for Gemini 3.1 Pro peaks at \$0.1739 per instance in Latex, representing an approximate 42% cost inflation compared to its Python equivalent (\$0.1222). R emerges as the most economically eficient language to evaluate. The significant divergence in evaluation resource consumption stems from the syntactic architecture of the respective languages. High-level data manipulation environments like Python and R leverage concise syntax, ensuring highly economical generation, while Latex (specifically TikZ/PGFPlots) functions as a declarative markup language, necessitating extensive boilerplate code and explicit coordinate-level geometric definitions.

Table 6 Code-level fidelity of figure reproduction performance on FigCodeBench. We report CodeBERTScore (CBS; Python only), CrystalBLEU (CB), normalized AST similarity $( S _ { \mathrm { A S T } } )$ and plotting-API F1 $( F _ { \mathrm { A P I } } )$ between the generated code and the corresponding ground-truth implementation. Higher values indicate greater similarity to the corresponding ground-truth code. Missing or unmatched code outputs are assigned a score of zero when computing the all-instance averages. The best and second-best results are in bold and underlined.
<table><tr><td rowspan="2">Model</td><td colspan="4">Python</td><td colspan="3">Matlab</td><td colspan="3">R</td><td colspan="3">Latex</td></tr><tr><td>CBS</td><td>CB</td><td> $S _ { \mathrm { A S T } }$ </td><td> $\overline { { F _ { \mathrm { A P I } } } }$ </td><td>CB</td><td> $\underline { { S _ { \mathrm { A S T } } } }$ </td><td> $\underline { { F _ { \mathrm { A P I } } } }$ </td><td>CB</td><td> $\underline { { S _ { \mathrm { A S T } } } }$ </td><td> $\underline { { F _ { \mathrm { A P I } } } }$ </td><td>CB</td><td> $S _ { \mathrm { A S T } }$ </td><td> $\overline { { F _ { \mathrm { A P I } } } }$ </td></tr><tr><td colspan="10">Proprietary Models</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>0.818</td><td>0.150</td><td>0.649</td><td>0.537</td><td>0.090</td><td>0.538</td><td>0.342</td><td>0.216</td><td>0.719</td><td>0.506</td><td>0.162</td><td>0.596</td><td>0.630</td></tr><tr><td>GPT-5.4-Mini</td><td>0.812</td><td>0.138</td><td>0.630</td><td>0.518</td><td>0.111</td><td>0.590</td><td>0.415</td><td>0.215</td><td>0.725</td><td>0.494</td><td>0.152</td><td>0.599</td><td>0.617</td></tr><tr><td>GPT-5.2</td><td>0.819</td><td>0.143</td><td>0.645</td><td>0.547</td><td>0.081</td><td>0.581</td><td>0.403</td><td>0.221</td><td>0.727</td><td>0.525</td><td>0.159</td><td>0.595</td><td>0.630</td></tr><tr><td>Claude Opus 4.7</td><td>0.753</td><td>0.036</td><td>0.618</td><td>0.312</td><td>0.033</td><td>0.565</td><td>0.314</td><td>0.045</td><td>0.688</td><td>0.447</td><td>0.028</td><td>0.480</td><td>0.437</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.845</td><td>0.274</td><td>0.741</td><td>0.646</td><td>0.136</td><td>0.653</td><td>0.490</td><td>0.306</td><td>0.797</td><td>0.591</td><td>0.146</td><td>0.603</td><td>0.628</td></tr><tr><td>Gemini 3 Flash</td><td>0.842</td><td>0.265</td><td>0.735</td><td>0.605</td><td>0.121</td><td>0.669</td><td>0.510</td><td>0.298</td><td>0.787</td><td>0.555</td><td>0.166</td><td>0.656</td><td>0.686</td></tr><tr><td>Gemini 3 Flash-TK</td><td>0.838</td><td>0.229</td><td>0.708</td><td>0.581</td><td>0.116</td><td>0.663</td><td>0.507</td><td>0.288</td><td>0.773</td><td>0.542</td><td>0.134</td><td>0.611</td><td>0.648</td></tr><tr><td>Gemini 2.5 Flash</td><td>0.752</td><td>0.027</td><td>0.559</td><td>0.254</td><td>0.020</td><td>0.550</td><td>0.297</td><td>0.063</td><td>0.676</td><td>0.377</td><td>0.050</td><td>0.456</td><td>0.354</td></tr><tr><td>Gemini 2.5 Flash-TK</td><td>0.745</td><td>0.027</td><td>0.525</td><td>0.269</td><td>0.021</td><td>0.559</td><td>0.278</td><td>0.045</td><td>0.554</td><td>0.335</td><td>0.029</td><td>0.447</td><td>0.326</td></tr><tr><td>Grok-4.3</td><td>0.819</td><td>0.180</td><td>0.680</td><td>0.533</td><td>0.111</td><td>0.595</td><td>0.501</td><td>0.263</td><td>0.786</td><td>0.570</td><td>0.144</td><td>0.592</td><td>0.623</td></tr><tr><td>Qwen3.6-Plus</td><td>0.824</td><td>0.190</td><td>0.669</td><td>0.536</td><td>0.105</td><td>0.638</td><td>0.476</td><td>0.273</td><td>0.772</td><td>0.534</td><td>0.168</td><td>0.645</td><td>0.666</td></tr><tr><td colspan="10">Open-Source Models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-5.1</td><td>0.671</td><td>0.022</td><td>0.402</td><td>0.219</td><td>0.022</td><td>0.454</td><td>0.273</td><td>0.056</td><td>0.653</td><td>0.376</td><td>0.039</td><td>0.415</td><td>0.332</td></tr><tr><td>GLM-4.6V</td><td>0.763</td><td>0.044</td><td>0.571</td><td>0.225</td><td>0.035</td><td>0.588</td><td>0.373</td><td>0.045</td><td>0.685</td><td>0.398</td><td>0.013</td><td>0.401</td><td>0.235</td></tr><tr><td>MiniMax-M3</td><td>0.751</td><td>0.028</td><td>0.521</td><td>0.224</td><td>0.031</td><td>0.513</td><td>0.287</td><td>0.060</td><td>0.663</td><td>0.363</td><td>0.034</td><td>0.407</td><td>0.350</td></tr><tr><td>Kimi-K2.5</td><td>0.816</td><td>0.192</td><td>0.658</td><td>0.538</td><td>0.130</td><td>0.629</td><td>0.490</td><td>0.280</td><td>0.765</td><td>0.543</td><td>0.121</td><td>0.462</td><td>0.471</td></tr><tr><td>DouBao-Seed-2-0-Pro</td><td>0.811</td><td>0.186</td><td>0.689</td><td>0.464</td><td>0.120</td><td>0.640</td><td>0.477</td><td>0.264</td><td>0.787</td><td>0.544</td><td>0.124</td><td>0.539</td><td>0.572</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>0.815</td><td>0.200</td><td>0.698</td><td>0.557</td><td>0.104</td><td>0.640</td><td>0.481</td><td>0.252</td><td>0.773</td><td>0.542</td><td>0.155</td><td>0.643</td><td>0.649</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.797</td><td>0.151</td><td>0.608</td><td>0.482</td><td>0.084</td><td>0.630</td><td>0.453</td><td>0.245</td><td>0.758</td><td>0.520</td><td>0.158</td><td>0.613</td><td>0.648</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>0.806</td><td>0.219</td><td>0.692</td><td>0.535</td><td>0.112</td><td>0.627</td><td>0.525</td><td>0.233</td><td>0.784</td><td>0.563</td><td>0.136</td><td>0.613</td><td>0.643</td></tr><tr><td>Qwen3-VL-32B</td><td>0.817</td><td>0.136</td><td>0.663</td><td>0.484</td><td>0.042</td><td>0.590</td><td>0.504</td><td>0.086</td><td>0.721</td><td>0.459</td><td>0.069</td><td>0.582</td><td>0.604</td></tr><tr><td>Qwen3-VL-8B</td><td>0.811</td><td>0.075</td><td>0.655</td><td>0.454</td><td>0.008</td><td>0.418</td><td>0.363</td><td>0.054</td><td>0.678</td><td>0.452</td><td>0.031</td><td>0.492</td><td>0.525</td></tr><tr><td>Qwen2.5-VL-72B</td><td>0.812</td><td>0.177</td><td>0.666</td><td>0.502</td><td>0.063</td><td>0.536</td><td>0.516</td><td>0.157</td><td>0.710</td><td>0.565</td><td>0.118</td><td>0.602</td><td>0.642</td></tr><tr><td>InternVL3.5-38B</td><td>0.809</td><td>0.155</td><td>0.652</td><td>0.420</td><td>0.072</td><td>0.537</td><td>0.450</td><td>0.152</td><td>0.686</td><td>0.529</td><td>0.102</td><td>0.523</td><td>0.572</td></tr><tr><td>InternVL3.5-8B</td><td>0.800</td><td>0.080</td><td>0.638</td><td>0.366</td><td>0.032</td><td>0.503</td><td>0.434</td><td>0.054</td><td>0.593</td><td>0.489</td><td>0.061</td><td>0.535</td><td>0.583</td></tr></table>

## 4.3 Benchmark Results

Tab. 5, Tab. 6, Tab. 7, and Fig. 11 show the main results in terms of image-oriented metrics, code-oriented metrics, mean model opinion score (MMOS), and fig-code fidelity (FCF) scores, respectively.

Performance Comparison on Code Executability. First, the results in Tab. 5 reveal a significant disparity in code executability. In Python evaluations, proprietary models show a highly concentrated, superior execution distribution $( M e d i a n = 9 0 . 9 7 \% )$ whereas open-source models exhibit lower and variable reliability $( M e d i a n = 8 2 . 4 5 \% )$ as supported by the Mann-Whitney U test $( U = 1 1 8 . 5 , p < 0 . 0 1 )$ . This gap further widens in less prevalent domains like Matlab, where proprietary models maintain a median execution rate of 78.50%, the open-source median collapses to 46.33%, yielding an even more pronounced statistical divergence $( U = 1 3 2 . 0 , p < 0 . 0 0 1 )$ ). Fig. 7 further demonstrates that code executability is statistically biased by the target programming language, showing a performance hierarchy of Python $\succ \mathrm { R } \succ \mathrm { M A T L A B } \succ \mathrm { L a T e X }$ From the perspective of the model, Claude Opus 4.7 consistently achieves the highest execution rate across all 4 languages, followed by Gemini 3 Flash-TK and Grok-4.3. While flagship open-source models like Kimi-K2.5 and DouBao-Seed-2-0-Pro achieve highly competitive execution rates in Python $( > 9 1 \% )$ and $\mathrm { ~ R ~ } ( > 8 6 \% )$ , their reliability also collapses when interacting with declarative markup (Latex) or domain-specific environments (Matlab). Such execution degradation suggests that current pre-training paradigms may overweight Python data, leaving them poorly equipped to generate executable code for other language ecosystems. Additionally, a longitudinal comparison between the Gemini 2.5 and 3 series highlights a significant generational leap in crossdomain compilation resilience, with execution gains observed across all four languages, including an improvement of nearly 40% in execution rate for Latex.

Table 7 Performance comparison on mean model opinion score (MMOS). We report the scores rated by three MLLM judges. J<sub>1</sub>, J<sub>2</sub>, and J<sub>3</sub> represent GPT-5.1, GPT-5.6-Luna, and Gemini-3.5-Flash, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="4">Python</td><td colspan="4">Matlab</td><td colspan="4">R</td><td colspan="4">Latex</td></tr><tr><td>J1</td><td>J2</td><td>J3</td><td>MMOS</td><td>J1</td><td>J2</td><td>J3</td><td>MMOS</td><td>J1</td><td>J2</td><td>J3</td><td>MMOS</td><td>J1</td><td>J2</td><td>J3</td><td>MMOS</td></tr><tr><td colspan="10">Proprietary Models</td><td colspan="3"></td><td colspan="4"></td></tr><tr><td>GPT-5.4</td><td>85.59</td><td>75.87</td><td>77.74</td><td>79.73</td><td>41.58</td><td>36.83</td><td>35.52</td><td>37.98</td><td>53.65</td><td>45.70</td><td>41.57</td><td>46.97</td><td>46.40</td><td>39.04</td><td>35.86</td><td>40.43</td></tr><tr><td>GPT-5.4-Mini</td><td>84.31</td><td>74.60</td><td>66.76</td><td>75.22</td><td>65.70</td><td>56.62</td><td>56.64</td><td>59.65</td><td>67.46</td><td>57.88</td><td>51.12</td><td>58.82</td><td>41.27</td><td>33.86</td><td>30.57</td><td>35.23</td></tr><tr><td>GPT-5.2</td><td>85.43</td><td>75.52</td><td>68.04</td><td>76.33</td><td>61.77</td><td>53.87</td><td>52.06</td><td>55.90</td><td>34.97</td><td>29.59</td><td>26.43</td><td>30.33</td><td>35.86</td><td>29.77</td><td>26.97</td><td>30.87</td></tr><tr><td>Claude Opus 4.7</td><td>86.33</td><td>77.59</td><td>73.56</td><td>76.19</td><td>72.37</td><td>61.04</td><td>58.73</td><td>64.05</td><td>68.35</td><td>62.17</td><td>54.66</td><td>61.73</td><td>55.14</td><td>50.93</td><td>49.35</td><td>51.81</td></tr><tr><td>Gemini 3.1 Pro</td><td>88.20</td><td>80.66</td><td>82.24</td><td>83.70</td><td>76.29</td><td>67.98</td><td>65.77</td><td>70.01</td><td>65.48</td><td>56.51</td><td>51.74</td><td>57.91</td><td>56.24</td><td>50.44</td><td>45.47</td><td>50.72</td></tr><tr><td>Gemini 3 Flash</td><td>85.45</td><td>76.83</td><td>68.09</td><td>76.79</td><td>74.36</td><td>65.23</td><td>65.62</td><td>68.40</td><td>79.11</td><td>69.17</td><td>61.00</td><td>69.76</td><td>59.96</td><td>52.28</td><td>45.04</td><td>52.43</td></tr><tr><td>Gemini 3 Flash-TK</td><td>85.12</td><td>76.32</td><td>74.09</td><td>78.51</td><td>82.02</td><td>73.27</td><td>69.69</td><td>74.99</td><td>83.07</td><td>69.12</td><td>66.85</td><td>73.01</td><td>56.62</td><td>49.93</td><td>42.78</td><td>49.78</td></tr><tr><td>Gemini 2.5 Flash</td><td>62.44</td><td>51.28</td><td>53.71</td><td>55.81</td><td>57.70</td><td>49.37</td><td>46.52</td><td>51.20</td><td>63.55</td><td>49.36</td><td>46.27</td><td>53.06</td><td>12.21</td><td>7.93</td><td>5.74</td><td>8.63</td></tr><tr><td>Gemini 2.5 Flash-TK</td><td>63.79</td><td>54.06</td><td>58.68</td><td>58.84</td><td>53.66</td><td>47.68</td><td>45.80</td><td>49.05</td><td>58.21</td><td>51.60</td><td>50.32</td><td>53.38</td><td>8.94</td><td>6.58</td><td>3.87</td><td>6.46</td></tr><tr><td>Grok-4.3</td><td>77.39</td><td>64.15</td><td>66.91</td><td>69.48</td><td>69.64</td><td>58.09</td><td>57.25</td><td>61.66</td><td>74.90</td><td>63.52</td><td>56.75</td><td>65.06</td><td>58.54</td><td>46.70</td><td>38.62</td><td>47.95</td></tr><tr><td>Qwen3.6-Plus</td><td>80.89</td><td>70.33</td><td>70.58</td><td>73.93</td><td>65.35</td><td>57.36</td><td>55.71</td><td>59.47</td><td>74.08</td><td></td><td>63.36</td><td>67.53</td><td>54.65</td><td></td><td>43.45</td><td>48.26</td></tr><tr><td>Open-Source Models</td><td colspan="4"></td><td colspan="4"></td><td colspan="4">65.14</td><td colspan="4">46.67</td></tr><tr><td>GLM-5.1</td><td>47.86</td><td>41.83</td><td>42.92</td><td>44.20</td><td>27.84</td><td>19.23</td><td>21.28</td><td>22.78</td><td>52.77</td><td>38.33</td><td>32.64</td><td>41.25</td><td>27.65</td><td>18.35</td><td>16.77</td><td>20.92</td></tr><tr><td>GLM-4.6V</td><td>53.44</td><td>47.92</td><td>49.31</td><td>50.22</td><td>34.11</td><td>29.73</td><td>29.70</td><td>31.18</td><td>74.13</td><td>58.83</td><td>48.22</td><td>60.39</td><td>12.47</td><td>7.31</td><td>6.85</td><td>8.88</td></tr><tr><td>MiniMax-M3</td><td>58.60</td><td>51.64</td><td>54.74</td><td>54.99</td><td>29.05</td><td>23.18</td><td>25.63</td><td>25.95</td><td>57.28</td><td>43.37</td><td>37.29</td><td>45.98</td><td>45.76</td><td>37.57</td><td>32.60</td><td>38.64</td></tr><tr><td>Kimi-K2.5</td><td>81.33</td><td>70.93</td><td>63.84</td><td>72.03</td><td>70.88</td><td>61.32</td><td>60.50</td><td>64.23</td><td>82.60</td><td>72.31</td><td>62.64</td><td>72.52</td><td>43.87</td><td>36.60</td><td>32.12</td><td>37.53</td></tr><tr><td>DouBao-Seed-2-0-Pro</td><td>82.27</td><td>70.20</td><td>71.00</td><td>74.49</td><td>62.02</td><td>53.23</td><td>50.90</td><td>55.38</td><td>76.33</td><td>64.48</td><td>58.18</td><td>66.33</td><td>44.22</td><td>37.31</td><td>35.29</td><td>38.94</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>42.58</td><td>35.48</td><td>39.27</td><td>39.11</td><td>32.02</td><td>26.40</td><td>27.52</td><td>28.65</td><td>51.60</td><td>43.12</td><td>40.98</td><td>45.23</td><td>49.61</td><td>40.90</td><td>37.36</td><td>42.62</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>61.10</td><td>52.52</td><td>53.81</td><td>55.81</td><td>26.73</td><td>22.34</td><td>21.79</td><td>23.62</td><td>63.64</td><td>54.59</td><td>51.90</td><td>56.71</td><td>46.64</td><td>38.69</td><td>34.49</td><td>39.94</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>66.59</td><td>54.89</td><td>57.37</td><td>59.62</td><td>50.53</td><td>41.63</td><td>45.19</td><td>45.78</td><td>63.81</td><td>53.08</td><td>52.43</td><td>56.44</td><td>50.60</td><td>40.05</td><td>38.86</td><td>43.17</td></tr><tr><td>Qwen3-VL-32B</td><td>59.19</td><td>48.11</td><td>49.77</td><td>52.36</td><td>29.08</td><td>23.77</td><td>25.12</td><td>25.99</td><td>50.01</td><td>41.33</td><td>40.35</td><td>43.90</td><td>43.21</td><td>34.38</td><td>32.53</td><td>36.71</td></tr><tr><td>Qwen3-VL-8B</td><td>50.55</td><td>41.05</td><td>43.41</td><td>45.00</td><td>20.33</td><td>16.54</td><td>18.87</td><td>18.58</td><td>42.15</td><td>34.74</td><td>34.89</td><td>37.26</td><td>29.10</td><td>23.16</td><td>22.96</td><td>25.07</td></tr><tr><td>Qwen2.5-VL-72B</td><td>46.31</td><td>38.62</td><td>43.61</td><td>42.85</td><td>41.62</td><td>33.65 25.54</td><td>35.98 27.92</td><td>37.08 27.99</td><td>55.17 38.87</td><td>46.04 32.33</td><td>47.00 36.51</td><td>49.40 35.90</td><td>53.12 36.74</td><td>41.28 28.67</td><td>41.05 30.15</td><td>45.15 31.85</td></tr><tr><td>InternVL3.5-38B InternVL3.5-8B</td><td>50.27 38.66</td><td>41.30 31.67</td><td>43.80 34.79</td><td>45.12 35.04</td><td>30.51 23.41</td></table>

![](images/7c6f0554b2d2a34b2609c8c4fce25e63abfe0d12a5f50b4235d36663f1e4fc41.jpg)

![](images/1dd58e1013ece4de2933ce63de838363e1c914093f3c5d729014f6a1ad0c7d04.jpg)

![](images/d0891a0da76c63d7341adcf567150399c5cec9e6825c0feaefdf0f9750bffe5c.jpg)

![](images/a52a6dd0419c21877e8ff41b54a7c60e7590d20d1bfdbfd2d24730ec5f33cedb.jpg)

![](images/c842f31b1953b423e89f4d88ac04f007f963a708545331ba061b8c505108d339.jpg)  
Fig. 7 Statistical significance of cross-language diferences in image-oriented performance. Pairwise diferences among Python, Matlab, R, and LaTeX are examined for execution success rate (Exec), PSNR, SSIM, LPIPS, and CIEDE2000 color diference (∆E). Bars and vertically displayed numbers show the mean performance across 24 evaluated models, and error bars denote the standard error of the mean. All P values were calculated using two-sided Wilcoxon signed-rank tests on paired model-level measurements, followed by Holm–Bonferroni correction. NS, not significant; \*≤ 0.05, $^ { * * } \leq 0 . 0 1 , \mathrm { ~ a n d ~ } ^ { * * * } \leq 0 . 0 0 1$

Performance Comparison on Image-oriented Metrics. From the perspective of image-oriented reproduction metrics, first, high code execution reliability does not necessarily translate into superior image reproduction quality, indicating that successful code generation and visual rendering constitute related but distinct capabilities. Specifically, while Claude Opus 4.7 achieves a state-of-the-art execution rate in Python tasks, it is substantially outperformed by Gemini 3.1 Pro and GPT-5.4 across all visual assessment criteria. In both Matlab and R languages, Claude Opus 4.7 maintains commanding execution rates (> 97%) but is consistently eclipsed in visual similarity by the Gemini 3 Flash series. This phenomenon highlights a trade-of in code generation strategies, i.e., compilation conservatism versus visual risk-taking. Models like Claude Opus 4.7 appear to prioritize syntactic safety, thereby guaranteeing compilation success but resulting in generic charts that fail to capture the fine-grained aesthetic intricacies of the ground truth. Second, in languages like Python and R, models generally achieve high SSIM (> 0.550) and low LPIPS (< 0.600), suggesting that the underlying coordinate mapping and structural layout algorithms are well-learned. However, transitioning to Matlab and Latex triggers a severe collapse in visual fidelity. As demonstrated in Fig. 7, this degradation is most apparent in the ∆E metric. For instance, while GPT-5.4 maintains a $\Delta E$ of 16.607 in Python, its perceptual color error increases to 59.897 in MATLAB and 50.986 in Latex, indicating that even when models successfully compile low-resource code, they often fall back on hallucinated color schemes, failing to achieve the strict pixel-level determinism required for scientific reproduction. Furthermore, while the open-source MLLMs generally lag behind proprietary models, remarkable domain-specific breakthroughs by leading open-source architectures are also observed. Notably, Kimi-K2.5 sets a new benchmark record in the R domain. It not only achieves a formidable execution rate (91.68%) but also entirely dominates the image-oriented metrics with the highest PSNR (12.610), SSIM (0.699), and lowest LPIPS (0.459) across all 24 evaluated models.

Performance Comparison on Code-oriented Metrics. We quantify the syntactic and structural similarity of the generated code against the ground-truth scripts in Tab. 6. Surprisingly, Claude Opus 4.7, despite achieving the state-of-the-art in execution reliability $( > 9 8 \% )$ and image-oriented fidelity across all domains, exhibits near-bottom code similarity scores. In Python, its CrystalBLEU is only 0.036, and its API F1-score is 0.312, substantially lower than comparable models such as Gemini 3.1 Pro and even lower than much smaller models like Qwen3-VL-8B. This indicates that Claude Opus 4.7 does not memorize or mimic the canonical, textbook-style snippets in the ground-truth dataset. Instead, it adopts alternative programmatic paradigms that are structurally divergent (low $S _ { \mathrm { A S T } } )$ but visually isomorphic and computationally robust. Meanwhile, models that perform well on image metrics unexpectedly also achieve high scores on these metrics. For example, Gemini 3.1 Pro achieves the highest $S _ { \mathrm { A S T } }$ (0.741) and $F _ { \mathrm { A P I } } \left( 0 . 6 4 6 \right)$ in Python, followed by Gemini 3 Flash. A similar pattern holds in Matlab and R, exhibiting a strong tendency to reproduce canonical syntax from human-written code. Moreover, a cross-metric comparison reveals that exact lexical matching (CrystalBLEU) yields uniformly low scores across all models and domains (peaking at only 0.306 for Gemini 3.1 Pro in R). This highlights the inferiority of tokenlevel evaluation for code generation, as variable renaming or superficial formatting changes heavily penalize the score. In contrast, $S _ { \mathrm { A S T } }$ and $F _ { \mathrm { A P I } }$ yield significantly higher and more discriminative distributions, confirming that while models rarely guess the exact tokens of the ground truth, they successfully capture the essential plotting functions required for visual synthesis.

![](images/5065ad9ef8cd62b9d7698c4973fb2fdca3ddd5dee580a12801e53f9182faf6fb.jpg)  
Fig. 8 Fine-grained score distribution generated by MLLM judges. We report the average scores of three MLLM judges for each evaluated model across four programming languages.

![](images/88aaca335ddfcd746921597e0b428e043505e3bdcf79a382e2b91598f40a24e1.jpg)  
Fig. 9 Pearson correlation between diferent MLLM judges across four programming languages.

![](images/3b9e065b79b0caab069d38c8ac08d023c9296100ee40a89727c469b260cfb219.jpg)  
Fig. 10 Qualitative visualization of MLLM judges. All judge models produce human-aligned scores and mutually consistent rankings. From top to bottom, the reproduced figures show progressively decreasing similarity to the reference.

MLLM Judgment. As reported in Tab. 7, Gemini 3.1 Pro and Gemini 3 Flash series achieve the highest MMOS in Python and the other environments, respectively, which marks a substantial performance leap over their predecessor Gemini 2.5. Although open-source MLLMs generally lag behind proprietary ones, a few frontier models achieve remarkable domain-specific alignment. Specifically, Kimi-K2.5 achieves an exceptional MMOS of 72.52 in R, significantly outperforming numerous peers and even surpassing top-tier proprietary models like Claude Opus 4.7 (61.73). DouBao-Seed-2-0-Pro similarly demonstrates highly competitive visual fidelity in Python (74.49). However, most open-source models $( \mathrm { e . g . }$ , the GLM and smaller InternVL families) exhibit catastrophic performance degradation on Matlab and Latex tasks, with scores frequently falling into the 20–30 range. This extreme variance highlights a severe skew in open-source instruction-tuning distributions, indicating a critical scarcity of syntactic and visual priors in long-tail, low-resource plotting ecosystems. Consistent with prior multi-dimensional evaluations, rendering via Latex constitutes a formidable bottleneck for all evaluated models. The highest MMOS values, achieved by Gemini 3 Flash (52.43) and Claude Opus 4.7 (51.81), barely cross the threshold of marginal acceptability, underscoring that current MLLMs, regardless of parameter scale or architectural openness, have yet to master the explicit mathematical coordinate mapping, precise node anchoring, and syntactic invariants demanded by low-level declarative frameworks like TikZ and PGFPlots.

![](images/91b08ef9de12969bc1874c2b791b0484354ef9c570ea34e81424e50ccee9a995.jpg)

![](images/2fceca199c5d58c48f1b7a2a3319b1b0b33abe97f5a28a73362713f304f39183.jpg)

![](images/a722b8dcb448bb9786c4a01b32fc9150fa5f55a810d6957baddbef2b54041d31.jpg)  
(c) R

![](images/0e11a0cc90ed80eb75a9207f492dd94b9a4389ffdb971b93dcb6dc14cd19a6a8.jpg)  
(d) Latex  
Fig. 11 Comparison of Fig-Code Fidelity (FCF) results across four programming languages.

Fig. 8 further provides the rating results for five sub-dimensions, i.e., chart types, layout, text content, data, and style. We can observe that across nearly all evaluated models, the chart types (blue) and layout (purple) dimensions have the highest scores, while data (red) consistently performs as the weakest performance bottleneck. This indicates that MLLMs are generally proficient at identifying the correct base plotting API while sufering a profound degradation when tasked with microscopic precision, specifically the accurate mathematical projection of data distributions and geometric mappings into the coordinate space. Such discrepancy is magnified in mid-tier and earlier-generation models, such as Claude Opus 4.7, Gemini 2.5 Flash, and GLM-4.6V, which maintain competent layout scores but experience a significant performance decrease in both text content and data. In contrast, models like Gemini 3 Flash-TK that achieve the highest overall MMOS also demonstrate a balanced integration across all five axes. A similarly balanced capability profile is mirrored by top-tier open-source models like Kimi 2.5 and DouBao Seed 2.0 Pro, indicating that frontier multimodal alignment techniques are successfully bridging the gap between simply writing executable code and orchestrating a deterministic, pixel-perfect visual rendering engine.

Table 8 Performance comparison across diferent functional categories of figures. MMOS is used as the indicator.
<table><tr><td>Model</td><td>Statistical</td><td>Relational</td><td>Temporal</td><td>Compositional</td><td>Geospatial</td><td>Mathematical</td><td>Conceptual</td></tr><tr><td colspan="8">Proprietary Models</td></tr><tr><td>GPT-5.4</td><td>61.15</td><td>55.01</td><td>54.30</td><td>55.16</td><td>41.08</td><td>56.56</td><td>40.43</td></tr><tr><td>GPT-5.4-Mini</td><td>69.28</td><td>61.66</td><td>65.81</td><td>59.63</td><td>36.47</td><td>66.02</td><td>35.23</td></tr><tr><td>GPT-5.2</td><td>52.19</td><td>51.09</td><td>55.25</td><td>43.77</td><td>21.67</td><td>60.06</td><td>30.87</td></tr><tr><td>Claude Opus 4.7</td><td>68.76</td><td>62.31</td><td>72.58</td><td>66.74</td><td>30.75</td><td>61.48</td><td>51.81</td></tr><tr><td>Gemini 3.1 Pro</td><td>70.84</td><td>66.92</td><td>72.42</td><td>64.00</td><td>65.57</td><td>72.50</td><td>50.72</td></tr><tr><td>Gemini 3 Flash</td><td>75.31</td><td>69.82</td><td>74.60</td><td>74.55</td><td>53.90</td><td>67.56</td><td>52.43</td></tr><tr><td>Gemini 3 Flash-TK</td><td>78.33</td><td>72.44</td><td>79.60</td><td>74.59</td><td>49.08</td><td>74.76</td><td>49.78</td></tr><tr><td>Gemini 2.5 Flash</td><td>52.39</td><td>47.41</td><td>54.61</td><td>42.98</td><td>26.57</td><td>45.28</td><td>8.63</td></tr><tr><td>Gemini 2.5 Flash-TK</td><td>54.36</td><td>47.30</td><td>56.27</td><td>42.56</td><td>24.36</td><td>48.55</td><td>6.46</td></tr><tr><td>Grok-4.3</td><td>71.45</td><td>62.06</td><td>68.43</td><td>65.54</td><td>41.62</td><td>61.56</td><td>47.95</td></tr><tr><td>Qwen3.6-Plus Open-Source Models</td><td>74.49</td><td>62.20</td><td>72.28</td><td>64.38</td><td>35.56</td><td>65.20</td><td>48.26</td></tr><tr><td colspan="8"></td></tr><tr><td>GLM-5.1</td><td>42.37</td><td>35.18</td><td>48.94</td><td>32.84</td><td>19.28</td><td>41.25</td><td>20.92</td></tr><tr><td>GLM-4.6V</td><td>46.89</td><td>40.43</td><td>54.67</td><td>35.14</td><td>24.35</td><td>39.89</td><td>8.88</td></tr><tr><td>MiniMax-M3</td><td>51.50</td><td>45.66</td><td>52.92</td><td>38.60</td><td>29.07</td><td>46.74</td><td>38.64</td></tr><tr><td>Kimi-K2.5</td><td>76.05</td><td>66.67</td><td>75.92</td><td>71.74</td><td>38.86</td><td>63.77</td><td>37.53</td></tr><tr><td>DouBao-Seed-2-0-Pro</td><td>72.01</td><td>63.28</td><td>72.44</td><td>71.33</td><td>37.99</td><td>58.68</td><td>38.94</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>44.38</td><td>33.99</td><td>40.23</td><td>42.52</td><td>17.82</td><td>36.76</td><td>42.62</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>55.95</td><td>46.80</td><td>53.59</td><td>39.00</td><td>21.98</td><td>43.95</td><td>39.94</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>59.59</td><td>52.17</td><td>57.30</td><td>46.93</td><td>35.36</td><td>55.41</td><td>43.17</td></tr><tr><td>Qwen3-VL-32B</td><td>49.53</td><td>37.07</td><td>43.75</td><td>43.14</td><td>16.93</td><td>40.61</td><td>36.71</td></tr><tr><td>Qwen3-VL-8B</td><td>40.55</td><td>29.51</td><td>37.97</td><td>31.76</td><td>16.68</td><td>37.15</td><td>25.07</td></tr><tr><td>Qwen2.5-VL-72B</td><td>49.89</td><td>38.45</td><td>43.71</td><td>46.71</td><td>25.48</td><td>42.61</td><td>45.15</td></tr><tr><td>InternVL3.5-38B</td><td>40.62</td><td>32.37</td><td>35.80</td><td>33.64</td><td>21.83</td><td>42.40</td><td>31.85</td></tr><tr><td>InternVL3.5-8B</td><td>33.29</td><td>20.14</td><td>27.49</td><td>28.03</td><td>4.40</td><td>32.01</td><td>21.64</td></tr></table>

Furthermore, we observe slight deviations in evaluation preference. GPT-5.1 (J<sub>1</sub>) exhibits the most lenient scoring bias, awarding the highest scores across all model categories. In contrast, GPT-5.6-Luna (J<sub>2</sub>) and Gemini-3.5-Flash (J<sub>3</sub>) exhibit markedly more conservative and rigorous baselines. This intrinsic score variance strongly validates the necessity of the MMOS formulation, which calibrates individual judge leniency to yield a reliable and robust metric for visual generation fidelity. As shown in Fig. 9, all three MLLM judges correlate positively with each other within all languages, demonstrating the feasibility of our rating strategy. Meanwhile, the low correlations in cross-language scoring reflect domain brittleness and a severe divergence in the models’ capabilities. Fig. 10 further illustrates some qualitative results of MLLM judges, showing that their assessments are largely consistent across the five evaluation aspects.

Performance Comparison on Generation Eficiency. While image-oriented metrics evaluate the absolute visual quality and code-oriented metrics assess syntactic alignment, true practical utility in code generation hinges on token economy. To evaluate this, we analyze the Figure–Code Fidelity (FCF), a joint metric that quantifies the visual quality achieved relative to the volume of code generated. The FCF distributions across the four programming domains, visualized in Fig. 11, reveal a highly asymmetric landscape of generative eficiency. In the Python and Matlab tasks, proprietary models establish a clear Pareto frontier for generation eficiency. Gemini 3.1 Pro consistently achieves the highest FCF scores in both domains, followed closely by the Gemini 3 Flash series. This indicates that these proprietary architectures have successfully internalized the canonical, high-density APIs of libraries like matplotlib and base Matlab, allowing them to achieve maximum visual fidelity without resorting to verbose workarounds.

Conversely, transitioning to R and Latex reveals a striking inversion of the performance hierarchy. In these domains, the open-source MLLMs outperform proprietary models in generative eficiency. Most notably, Qwen2.5-VL-72B obtains the highest FCF scores in both R and Latex, while GPT-5.2 and Gemini 2.5 Flash are in last place. Given the high degree of similarity in their code structures (Tab. 6) and the relatively smaller number of generated tokens (Tab. 4), we speculate that these models execute complex geometric mapping with concise macro commands, whereas many proprietary models sufer from length penalties by over-generating explicit coordinate logic.

Performance on Diferent Image Types. In Tab. 8, we investigate the figure reproduction performance of models across seven functional categories. First, most models show the highest proficiency in charting tasks that rely on standard data mapping, namely the statistical, temporal, and compositional categories. Notably, Gemini 3 Flash-TK, augmented with test-time reasoning, dominates, achieving the highest scores of 78.33, 79.60, and 74.59 across all three categories, respectively. Second, the geospatial category constitutes the most significant obstacle in current visual code generation tasks. Except for Gemini 3.1 Pro, which achieves a relatively high score of 65.57, almost all other models sufer a clif-like performance collapse. For instance, GPT-5.2 obtains a mere 21.67, while InternVL3.5-8B reaches only 4.40. This failure can be attributed to the underlying logic of map rendering, which requires models to process non-standard map projections and frequently relies on implicit, external geographic polygon data (such as Shapefiles or specific map packages). Without explicit definitions of these complex coordinate contours in their pre-training corpora, models can produce severe logical hallucinations when generating such code. Furthermore, the conceptual and relational categories, which emphasize abstract logic and relationships, also exhibit severe capability degradation. This further exposes a severe architectural flaw, i.e., current MLLMs lack the 2D spatial reasoning capabilities required for codebased object depiction and the representation of abstract set relationships (e.g., Venn diagrams or hierarchical trees), once detached from a regular grid coordinate system.

Comparison Between Reasoning Models and Their Non-reasoning Counterparts. We also examine the impact of enabling reasoning in MLLMs on diferent languages. In particular, we choose to compare Gemini 2.5 Flash Nothinking versus Thinking and their successors, Gemini 3 Flash Nothinking versus Thinking. First, immature test-time reasoning induces performance degradation in code-oriented visual generation tasks. As reported in Tab. 5, the integration of reasoning in Gemini 2.5 Flash-TK triggers a decline in both execution rates and structural similarity across all four programming languages. We suspect that this efect originates from an overthinking bias inherent in early models, which increases the risk of compilation collapse and causes the generated visualizations to deviate from the standard visual priors of the ground truth. Conversely, the Gemini 3 Flash series demonstrates a remarkable generational improvement in reasoning mechanisms. Although activating reasoning capability decreases pixel-level image metrics and code similarity metrics, the humanaligned perceptual fidelity metric, MMOS, achieves large improvements in Python (from 76.69 to 78.51), Matlab (from 68.40 to 74.99), and R (from 69.76 to 73.01). This divergence reflects the nature of reasoning in visual code generation. Rather than relying on rote memorization of boilerplate syntax and default API aesthetics, reasoning models synthesize charts through complex, explicit intermediate logic. Although such an approach incurs mathematical penalties under strict string-matching and absolute pixel-level metrics, it ultimately produces visualizations of superior logical coherence, information completeness, and semantic correctness, thereby earning higher recognition within the perception-based MLLM evaluation framework.

![](images/b3b7932c2d763264d6b4b3db705c5e5a6b9bf6d9cd0adb2dba45e6cdbcb3f579.jpg)  
Fig. 12 Model size versus MMOS. Trend lines are added to show the performance trend of each model family across diferent parameter scales. Note that we only investigate open-source MLLMs with publicly available parameters.

Model Size Analysis. Fig. 12 presents scatter plots of model size versus MMOS across four languages. We observe that increasing the parameter size generally yields consistent performance improvements within the same model family. In both the Qwen3-VL series (8B, 32B, 235B) and the InternVL3.5 series (8B, 38B), the logarithmic expansion of model size produces distinct upward trend lines in their MMOS across Python, Matlab, and R. As one of the largest evaluated open-source models, Kimi-K2.5 consistently operates at or near the performance ceiling across all four languages, representing the upper bound achievable through pure parameter scaling. Furthermore, we observe a counterintuitive inverse scaling phenomenon in the Qwen3.5 and GLM series, where generative performance degrades as model size increases. We hypothesize that this anomaly arises because their visual perception capabilities fail to scale proportionally with the expanding language backbone, creating a multimodal bottleneck in coordinate-based rendering tasks. Interestingly, a notable exception emerges in the LaTeX domain. Since Latex is a declarative and descriptive language, strengthening the language backbone can efectively compensate for lagging visual priors. In this scenario, advanced textual reasoning translates directly into improved geometric reproducibility, demonstrating that language scaling remains highly beneficial for purely logic-driven, text-intensive rendering paradigms.

![](images/fdc13e2e860b8cd277750b76269fa679f596aaecb31bb4609f690d519066daf7.jpg)  
Fig. 13 Correlation analysis of image-oriented metrics.

![](images/cd7ced45ebe54ba7b5b953b59cf4ce1d3add583965b56a6f00fcc4d372be0f11.jpg)

![](images/0f190be6f1557d9f3aac49d4a1552e9ffc5176ab33a0b19fdba475ea001bd4f5.jpg)  
Fig. 14 Correlation analysis of code-oriented metrics.

![](images/6453f7f5cbb7e9334ab822bd21a61154cec4357a42d595bee565b63deb87be32.jpg)

## 4.4 Correlation Between Diferent Metrics

To validate the rationality of our evaluation protocol, we conduct a comprehensive meta-evaluation of the intra-metric correlation, comparing MMOS against objective image-oriented metrics. We use Spearman’s rank correlation ρ and Kendall’s rank correlation τ as indicators. First, as shown in Fig. 13, in the diagonal blocks of the four coding environments, PSNR and SSIM show consistently strong positive correlations, indicating that they capture overlapping low-level reconstruction errors with equivalent results. Second, the deep perceptual metric LPIPS and the color metric ∆E demonstrate strong negative correlation trends with PSNR and SSIM. It is worth noting that this negative correlation is not an absolutely perfect linear divergence (e.g., the Kendall’s τ between ∆E and PSNR in R is -0.34), showing that LPIPS and ∆E are capable of detecting deep semantic misalignments and subtle color gamut shifts that are easily overlooked by traditional pixel metrics. Meanwhile, we notice that the pronounced decay in cross-language correlations highlights that the multimodal figure reproduction capability of MLLMs is highly dependent on language-specific syntactic priors, substantiating the necessity of constructing benchmarks that span multiple languages.

Table 9 Correlation between image-oriented metrics and MMOS. Given that code-oriented metrics primarily quantify similarity and thus do not directly reflect final graphical fidelity, we exclude them in this experiment.
<table><tr><td rowspan="2">Model</td><td colspan="4">Python</td><td colspan="4">Matlab</td><td colspan="4">R</td><td colspan="4">Latex</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>∆E</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>∆E</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>∆E</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>∆E</td></tr><tr><td>Spearman&#x27;s ρ</td><td>0.7687</td><td>0.7856</td><td>-0.7380</td><td>-0.7730</td><td>0.8035</td><td>0.7617</td><td>-0.8915</td><td>-0.5290</td><td>0.6783</td><td>0.7180</td><td>-0.7423</td><td>-0.7278</td><td>0.5313</td><td>0.5462</td><td>-0.6269</td><td>-0.5235</td></tr><tr><td>Kendall&#x27;s τ</td><td>0.5725</td><td>0.5621</td><td>-0.5699</td><td>-0.5435</td><td>0.6232</td><td>0.5725</td><td>-0.7441</td><td>-0.7104</td><td>0.4928</td><td>0.5554</td><td>-0.5554</td><td>-0.5870</td><td>0.3333</td><td>0.3376</td><td>-0.4335</td><td>-0.3406</td></tr></table>

![](images/09074e6db853b49e0c91711c5b2b3b8fb70baa925aff343ffc1e560f7452b484.jpg)

![](images/a5f46c1dbf5e8ae7231d11e7e4fe89aaedb12c7ebf5cb5766342fa29d8781589.jpg)  
Fig. 15 Performance correlation of FigCodeBench with benchmarks evaluating chart understanding and code generation capabilities.

In Fig. 14, we observe a high consistency during cross-language comparisons, in contrast to the heterogeneity exhibited by image-oriented metrics. This reveals that the general programming proficiency of frontier MLLMs has strong cross-environment transferability. Tab. 9 further reports the correlation between image-oriented metrics and MMOS. Consistent with the above analysis, deep semantic feature-based LPIPS aligns better with human preferences compared to the others. Meanwhile, all imageoriented metrics demonstrate moderate to strong correlations with MMOS across four programming languages, validating the rationality of employing image-level visual fidelity metrics for figure reproduction assessment.

![](images/012ca8444d166c7f00e2f5383e02db606942d53bbb313a8daef66c5a5cb0cd8f.jpg)  
Fig. 16 Screenshot of the user interface (UI) for subjective evaluation.

Table 10 Pearson correlation coeficient r between multi-level MMOS and human evaluation. J<sub>1</sub>, J<sub>2</sub>, and J<sub>3</sub> represent GPT-5.1, GPT-5.6-Luna, and Gemini-3.5-Flash, respectively.
<table><tr><td>Metric</td><td>Chart Type</td><td>Layout</td><td>Text Content</td><td>Data</td><td>Style</td><td>Overall</td></tr><tr><td>J1</td><td>0.863</td><td>0.804</td><td>0.852</td><td>0.752</td><td>0.738</td><td>0.793</td></tr><tr><td>J2</td><td>0.881</td><td>0.822</td><td>0.846</td><td>0.774</td><td>0.744</td><td>0.823</td></tr><tr><td>J3</td><td>0.897</td><td>0.815</td><td>0.863</td><td>0.807</td><td>0.761</td><td>0.837</td></tr><tr><td>MMOS</td><td>0.883</td><td>0.817</td><td>0.855</td><td>0.783</td><td>0.747</td><td>0.826</td></tr></table>

## 4.5 Correlation Between FigCodeBench and Other Benchmarks

To uncover the factors influencing FigCodeBench performance, we decompose the capabilities required for figure replication into chart understanding and code generation, and investigate the performance correlation between FigCodeBench and existing benchmarks targeting these two dimensions. Specifically, we adopt ChartXiv (reasoning question) (Wang et al. 2024c) and LiveCodeBench (v6) (Zheng et al. 2025) as representative benchmarks for chart understanding and code generation, respectively. Pearson correlation coeficients r are computed to quantify these relationships. As depicted in 15, FigCodeBench shows a more stronger linear positive correlation with ChartXiv (r = 0.797) than LiveCodeBench v6 (r = 0.537), which indicates that proficiency in generic algorithmic programming does not directly translate to visual code generation. Unlike pure logic-driven coding tasks, reproducing figures requires deterministic coordinate spatial reasoning and precise geometric mapping. Therefore, the strong alignment with ChartXiv underscores that accurate visual comprehension serves as the indispensable basis for cross-modal image reproduction. These results further demonstrate the necessity of FigCodeBench, which fills the gap in the evaluation of cross-modal perception and generation.

![](images/eda6ee850a484266ad1d15ea43128e5bc00082c224087f50730939f277c999d7.jpg)  
Fig. 17 The MMOS of six representative models at diferent dificulty tiers.

## 4.6 Human Evaluation

To assess the reliability of the designed multi-level MLLM judges, we further conducted human evaluation for subjective consistency analysis. Specifically, we recruited 12 experts from the university. Each of them has over three years of experience in using related programming languages and creating scientific diagrams. Following the same protocol as the MLLM judges, each human evaluator rates a generated-reference image pair on five dimensions (i.e., type, layout, style, text, and data), with scores ranging from 0 to 20. Subsequently, we calculate the Pearson correlation coeficient between the aggregate scores and the MMOS. A total of 1,000 generated–reference image pairs are uniformly sampled from the four language-specific subsets of FigCodeBench. Fig. 16 provides a screenshot of the UI in the subjective evaluation. Before the annotation, we instructed all participants to have a clear and consistent understanding of all evaluated aspects and tested their eligibility via a 10-image pre-labeling.

As shown in Tab. 10, this evaluation paradigm demonstrates strong alignment with human intuition across multiple dimensions. Notably, Gemini-3.5-Flash $\left( J _ { 3 } \right)$ achieves the highest correlation $( r = 0 . 8 3 7 )$ . By ensembling the preferences of diferent models, the aggregated MMOS achieves a strong overall correlation of 0.826, which efectively mitigates the evaluation biases introduced by a single weaker model $( \mathrm { e . g . , ~ } J _ { 1 }$ at 0.793), thereby providing a robust and highly human-aligned quantitative benchmark. Moreover, we find that highly objective dimensions such as chart type and text content demonstrate a higher correlation with human judgments compared to the high-level style and rigorous data dimensions. This is partly attributed to the inherent variance of abstract attributes and partly attributed to the fundamental limitations of current visual encoders in performing fine-grained spatial mappings. Besides, we assess the inter-rater reliability among human subjects using Krippendorf’s alpha (Hayes and Krippendorf 2007), yielding a value of $\alpha = 0 . 7 6 6$ , which indicates strong agreement.

## 4.7 Performance on Diferent Dificulty Tiers

We provide the performance of six representative models, including GPT-5.4, Gemini 3.1 Pro, Qwen 3.6-Plus, Kimi-K2.5, Qwen3.5-122B, and Qwen3-VL-32B, across different dificulty tiers for FigCodeBench in Fig. 17. We observe a consistent decline in performance across all languages as the dificulty increases. For example, the performance (MMOS) of Qwen3.6-Plus at easy, medium, and hard levels is 57.90, 49.82, and 31.11, respectively. These results demonstrate that FigCodeBench is inherently challenging and confirm the eficacy of the established dificulty levels. Additionally, we observe a more significant performance decrease in relatively weaker MLLMs, such as Qwen3.5-122B and Qwen3-VL-32B, than in top-tier models. Specifically, Qwen3-VL-32B sufers a relative decline of 77.2%, from an average Easy score of 47.69 to a mere 10.85 on Hard tasks. In contrast, leading proprietary models show better resilience against figure complexity. For instance, Gemini 3.1 Pro and Qwen3.6-Plus maintain MMOS scores of 30.04 and 31.11 on Hard tasks, sufering relative drops of 57.21% and 46.27%, respectively, from the Easy level. Meanwhile, we find that all models degrade only moderately from Easy to Medium, but much more sharply from Medium to Hard. This pattern suggests that the structural requirements in the Hard tier, such as extreme aspect ratios, intricate mathematical annotations, and densely overlapping geometric constraints, mark a level of complexity that current MLLMs still struggle with.

## 4.8 Case Study

To further elucidate the limitations of MLLMs on FigCodeBench, we conduct an indepth qualitative case study of the generation results from six representative models (GPT-5.4, Gemini 3.1 Pro, Grok 4.3, GLM-5.1, Kimi-K2.5, and Qwen3-VL-235B-A22B), as shown in Fig. 18. Through a cross-domain visual comparison against the ground-truth references, we identify several critical dimensions of deficiency in contemporary codeto-figure reproduction tasks as follows. First, Gemini 3.1 Pro achieves the overall best visual reproduction performance in four programming languages. In the Python task, it perfectly reconstructs the complex continuous gradient background (transitioning from orange to green) and maintains accurate bar proportions. Similarly, in the Matlab area plot and Latex relationship tree, its alignment precision, typography, and structural topology are nearly indistinguishable from the reference images. While several models successfully synthesize fundamental chart skeletons, they frequently struggle with micro-level visual styling. In the Python example, Kimi-K2.5 and Grok 4.3 fail to reproduce the smooth continuous gradient, erroneously discretizing the background into solid, segregated color blocks. Meanwhile, the color accuracy of Qwen3-VL-235B-A22B is even worse. In the Matlab stacked area chart, Grok 4.3 and Kimi-K2.5 generate prominent visual artifacts by excessively thickening or weakening the black dashed lines and altering the title proportions. Second, current MLLMs exhibit severe degradation in spatial reasoning when managing dense text configurations or strict aspect ratios. In the R heatmap case, GPT-5.4, Kimi-K2.5, and Qwen3-VL-235B sufer a severe typographic failure, where a large amount of text elements overlap into an illegible black blob, entirely destroying chart readability.

As demonstrated by the previous quantitative experiments, challenging environments like Matlab and LaTeX induce frequent compilation failures, e.g., GLM-5.1 failing in both Python and MATLAB, and Grok 4.3 failing in Latex. In addition, in the Latex relationship tree, GLM-5.1 generates a topologically distinct and contextually irrelevant tree, replacing the reference human names with stylistic attributes such as ”Style”, ”Color”, and ”Font”. Furthermore, Qwen3-VL-235B incorrectly modifies the node fill styles and the global layout spacing of the tree. These results suggest that, although current MLLMs have largely mastered basic data-to-geometry mapping, they still exhibit substantial structural weaknesses, such as spatial layout planning (e.g., collision avoidance and aspect-ratio control) and fine-grained aesthetic rendering (e.g., continuous colormaps).

![](images/395e92b85c50b2b327f9033731a0cc20b5e9d9fe5fa2770a45346157d53041c0.jpg)  
Fig. 18 Comparison of figures reproduced by diferent models. We visualize results from six representative MLLMs, including GPT-5.4, Gemini 3.1 Pro, Grok 4.3, GLM-5.1, Kimi-K2.5, and Qwen3-VL-235B-A22B. The rectangle with diagonal lines indicates a failure in generation.

## 5 Conclusions

In this work, we introduce FigCodeBench, a rigorously curated benchmark designed to evaluate the unified perception and coding capabilities of MLLMs in figure reproduction. We construct FigCodeBench based on four primary principles, i.e., comprehensiveness, fidelity, reproducibility, and determinism. FigCodeBench encompasses scientific drawing tasks with 6,194 figure-code pairs, spanning four programming languages and 7 functionfocused figure categories that closely reflect common research scenarios. FigCodeBench is developed through a human-in-the-loop and semi-automatic process that consists of comprehensive data sources and multidimensional evaluation protocols, addressing the isolation characteristics in current image understanding and code generation benchmarks. Based on our framework, we conducted extensive experiments on 11 widely used proprietary and 13 open-source MLLMs, revealing their inferiority in precise figure reproduction and the inconsistency between figure reproduction and code generation capabilities. We also investigate the performance correlation across metrics, programming languages, and ability dimensions, providing several valuable insights. We anticipate that FigCodeBench will drive future research in the field of perception-generation unified MLLMs.

Limitations. Despite our best eforts, we acknowledge the following limitations in our study. First, since FigCodeBench is tailored to evaluate code generation for plotting, its findings may not fully generalize to other types of programming tasks. Second, although our evaluation incorporates objective metrics with MLLM-as-ajudge semantics, it can inherently carry the risk of subtle alignment biases. For highly specialized domain figures, where a tiny geometric shift might fundamentally alter a physical or biological conclusion, general-purpose MLLM judges may lack the granular domain expertise required to penalize scientifically fatal, yet visually negligible, hallucinations. Future work will explore the integration of domain-specific models to further calibrate the evaluation tier.

## Declarations

• Availability of data and materials: The FigCodeBench website will be available at https://zijianchen98.github.io/FigCodeBench/. The FigCodeBench codebase will be available at https://github.com/zijianchen98/FigCodeBench. The Fig-CodeBench dataset will be hosted and version-tracked on Hugging Face, and will be permanently accessible at https://huggingface.co/datasets/CCZZJJ/ FigCodeBench. We encourage the community to provide feedback and engage in scrutinizing the benchmark.

• Competing interests: The authors declare no competing interests.

• Funding: This work was supported in part by the National Key R&D Program of China (2025ZD0124104) in collaboration with Shanghai Artificial Intelligence Laboratory, and in part by the Shanghai Municipal Special Program for Basic

Research on General AI Foundation Models (Grant No. 2025SHZDZX025D09), in collaboration with Shanghai Artificial Intelligence Laboratory.

## Appendix

## A Prompt Templates

In this section, we provide the prompt templates used for the experiment.

## A.1 Prompt for Code Generation

Note that we enforce uniform canvas geometries, e.g., DPI and aspect ratio configurations, to explicitly control the resolution of rendered figures.

## <|System Prompt|>

You are an expert developer who specializes in writing {code type} code based on a given figure.

## <|User Prompt|>

I found a very nice figure in a paper, but there is no corresponding source code available. I need your help to generate the corresponding {code type} code that can precisely reproduce the figure based on the image I provide.

Requirement:

1. Ensure figsize=(width: {width}, height: {height}) to match the original image size.

2. Here is some auxiliary information to help you:

(a) The additional package dependencies include {package 1, ..., package n}.

(b) The dataset used in generating figure includes {dataset 1, ..., dataset n}.

(c) The random seed used in generating this figure is {random seed}.

Now, return only the code, without any additional explanations.

## A.2 Prompt for Jury Models

<|System Prompt|>   
You are an expert acting as a rigorous judge for evaluating visualization chart plots.   
The first image (Reference Image) is the ground truth plot generated by various   
programming languages (e.g., Python, MATLAB, R, or LaTeX). The second image   
(AI-Generated Image) is rendered from the code generated by an AI assistant. Your   
task is to evaluate how accurately the AI-generated plot replicates the reference plot.   
<|User Prompt|>   
# Scoring Methodology (Total: 100 points): Evaluate the AI-generated image based   
on the following 5 criteria. Each criterion is worth up to 20 points.   
1. Chart Types (20 points): Does the generated image use the exact same chart types   
(e.g., line, bar, scatter, pie, 3D surface) and coordinate systems as the reference image?   
2. Layout (20 points): Does the spatial arrangement of subplots, aspect ratios, and   
the positioning of legends or insets match the reference image exactly (e.g., number of   
rows and columns)?   
3. Text Content (20 points): Does the generated image precisely include all semantic   
text elements from the reference image (e.g., main titles, subplot titles, annotations,   
axis labels, and legend text), excluding axis tick numeric labels?   
4. Data (20 points): How accurately do the data trends, distributions, number of data   
groups/series, and geometric mappings in the generated image resemble the reference   
image?   
5. Style (20 points): Does the generated image match the original in terms of visual   
aesthetics, including color palettes (line/fill colors), marker types (point shapes, line   
styles), grid presence, and background colors?   
# Output Instructions:   
Analyze the images step-by-step, then provide the final scores. You MUST output   
your response entirely in the following valid JSON format, without any surrounding   
markdown blocks or additional text:   
“evaluation”: {   
“chart types”: {“score”: <int between 0 and 20>},   
“layout”: {“score”: <between 0 and 20>},   
“text content”: {“score”: <between 0 and 20>},   
“data”: {“score”: <between 0 and 20>},   
“style”: {“score”: <between 0 and 20>}   
},   
“final total score”: <sum of the 5 scores>

## References

Anthropic (2026) Introducing claude opus 4.7. https://www.anthropic.com/news/ claude-opus-4-7, accessed: 2026-06-09

Bai L, Cai Z, Cao Y, et al (2025a) Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:250815763

Bai S, Cai Y, Chen R, et al (2025b) Qwen3-vl technical report. arXiv preprint arXiv:251121631

Bytedance Seed (2026) Seed 2.0 oficial launch. URL https://seed.bytedance.com/seed2

Cao R, Chen M, Chen J, et al (2026) Qwen3-coder-next technical report. arXiv preprint arXiv:260300729

Cappellazzo U, Kim M, Chen H, et al (2025) Large language models are strong audio-visual speech recognition learners. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, pp 1–5

Chen D, Chen R, Zhang S, et al (2024a) MLLM-as-a-judge: Assessing multimodal LLMas-a-judge with vision-language benchmark. In: Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol 235. PMLR, pp 6562–6595, URL https://proceedings.mlr.press/v235/chen24h.html

Chen M, Tworek J, Jun H, et al (2021) Evaluating large language models trained on code. arXiv preprint arXiv:210703374

Chen Z, Sun W, Tian Y, et al (2024b) Gaia: Rethinking action quality assessment for aigenerated videos. Advances in neural information processing systems 37:40111–40144

Chen Z, Wu J, Wang W, et al (2024c) Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 24185–24198

Chen Z, Chen T, Zhang W, et al (2025a) Obi-bench: Can lmms aid in study of ancient script on oracle bones? In: The Thirteenth International Conference on Learning Representations

Chen Z, Deng L, Chen Z, et al (2025b) Can large models fool the eye? a new turing test for biological animation. arXiv preprint arXiv:250806072

Chen Z, Hua W, Li J, et al (2025c) Pictobi-20k: Unveiling large multimodal models in visual decipherment for pictographic oracle bone characters. arXiv preprint arXiv:250905773

Chen Z, Sun W, Wu H, et al (2025d) Study of subjective and objective naturalness assessment of ai-generated images. IEEE Transactions on Circuits and Systems for Video Technology 35(4):3573–3588

Chen Z, Sun Y, Tian Y, et al (2025e) Maceval: A multi-agent continual evaluation network for large models. arXiv preprint arXiv:251109139

Chen Z, Tian Y, Sun Y, et al (2025f) Just noticeable diference for large multimodal models. arXiv preprint arXiv:250700490

Chen Z, Zhang W, Zhai G (2025g) Evaluating from benign to dynamic adversarial: A squid game for large language models. arXiv preprint arXiv:251110691

Eghbali A, Pradel M (2022) Crystalbleu: precisely and eficiently measuring the similarity of code. In: Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, pp 1–12

Fakhoury S, Naik A, Sakkas G, et al (2024) Llm-based test-driven interactive code generation: User study and empirical evaluation. IEEE Transactions on Software Engineering 50(9):2254–2268

Fei H, Wu S, Ji W, et al (2024) Video-of-thought: step-by-step video reasoning from perception to cognition. In: Proceedings of the 41st International Conference on Machine Learning, pp 13109–13125

Fu C, Chen P, Shen Y, et al (2025) Mme: A comprehensive evaluation benchmark for multimodal large language models. URL https://arxiv.org/abs/2306.13394, arXiv:2306.13394

Fu J, Ng SK, Jiang Z, et al (2024) Gptscore: Evaluate as you desire. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp 6556–6576

Fu L, Kuang Z, Song J, et al (2026) Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. Advances in Neural Information Processing Systems 38

Google (2025a) Gemini 2.5: Our most intelligent ai model. https: //blog.google/innovation-and-ai/models-and-research/google-deepmind/ gemini-model-thinking-updates-march-2025/, accessed: 2026-04-09

Google (2025b) Gemini 3 flash: frontier intelligence built for speed. https://blog.google/ products-and-platforms/products/gemini/gemini-3-flash/, accessed: 2026-04-09

Google (2026a) Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog.google/innovation-and-ai/models-and-research/gemini-models/ gemini-3-1-pro/, accessed: 2026-04-09

Google (2026b) Gemini 3.5: frontier intelligence with action. https://blog.google/ innovation-and-ai/models-and-research/gemini-models/gemini-3-5/, accessed: 2026- 09-09

Goyal Y, Khot T, Summers-Stay D, et al (2017) Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 6904–6913

Han Y, Zhang C, Chen X, et al (2023) Chartllama: A multimodal llm for chart understanding and generation. arXiv preprint arXiv:231116483

Hayes AF, Krippendorf K (2007) Answering the call for a standard reliability measure for coding data. Communication methods and measures 1(1):77–89

Hong W, Yu W, Gu X, et al (2025) Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:250701006

Hu H, Zhu X, He T, et al (2026) Qwen3-tts technical report. arXiv preprint arXiv:260115621

Hurst A, Lerer A, Goucher AP, et al (2024) Gpt-4o system card. arXiv preprint arXiv:241021276

Inc A (2026) Agentic coding - best ai for coding. https://cursor.com/, accessed: 2026-04-19

Jain N, Han K, Gu A, et al (2025) Livecodebench: Holistic and contamination free evaluation of large language models for code. In: The Thirteenth International Conference on Learning Representations

Jiang S, Liang J, Wang J, et al (2025) From specific-mllms to omni-mllms: a survey on mllms aligned with multi-modalities. In: Findings of the Association for Computational Linguistics: ACL 2025, pp 8617–8652

Jimenez CE, Yang J, Wettig A, et al (2024a) Swe-bench: Can language models resolve real-world github issues? In: International Conference on Learning Representations, pp 54107–54157

Jimenez CE, Yang J, Wettig A, et al (2024b) Swe-bench: Can language models resolve real-world github issues? In: The Twelfth International Conference on Learning Representations

Kimi Team, Bai T, Bai Y, et al (2026) Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:260202276

Li B, Ge Y, Ge Y, et al (2024a) Seed-bench: Benchmarking multimodal large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 13299–13308

Li B, Zhang Y, Guo D, et al (2024b) Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:240803326

Li K, Tian Y, Hu Q, et al (2024c) Mmcode: Benchmarking multimodal large language models for code generation with visually rich programming problems. In: Findings of the Association for Computational Linguistics: EMNLP 2024, pp 736–783

Liu F, Zhu T, Wu X, et al (2023a) A medical multimodal large language model for future pandemics. NPJ Digital Medicine 6(1):226

Liu Y, Li Z, Li H, et al (2023b) On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:230507895

Liu Y, Duan H, Zhang Y, et al (2024) Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision, Springer, pp 216–233

Luo MR, Cui G, Rigg B (2001) The development of the cie 2000 colour-diference formula: Ciede2000. Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Fran¸cais de la Couleur 26(5):340–350

Ma X, Zhang Z, Zhao H (2024) Coco-agent: A comprehensive cognitive mllm agent for smartphone gui automation. In: Findings of the Association for Computational Linguistics: ACL 2024, pp 9097–9110

Masry A, Do XL, Tan JQ, et al (2022) Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In: Findings of the association for computational linguistics: ACL 2022, pp 2263–2279

MiniMax (2026) Minimax m3: Coding and agentic frontier. 1m-context msa. native multimodality. https://www.minimax.io/models/text/m3, accessed: 2026-06-23

OpenAI (2025a) Gpt-5.1: A smarter, more conversational chatgpt. https://openai.com/ index/gpt-5-1/, accessed: 2026-08-09

OpenAI (2025b) Introducing gpt-5.2. https://openai.com/index/introducing-gpt-5-2/, accessed: 2026-06-09

OpenAI (2026a) Codex — ai coding partner from openai. https://openai.com/codex/, accessed: 2026-04-13

OpenAI (2026b) Gpt-5.6: Frontier intelligence that scales with your ambition. https: //openai.com/index/gpt-5-6/, accessed: 2026-09-09

OpenAI (2026c) Introducing gpt-5.3-codex. https://openai.com/index/ introducing-gpt-5-3-codex/, accessed: 2026-04-19

OpenAI (2026d) Introducing gpt-5.4. https://openai.com/index/introducing-gpt-5-4/, accessed: 2026-04-09

Ouyang L, Qu Y, Zhou H, et al (2025) Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, pp 24838–24848

Ouyang S, Huang D, Guo J, et al (2026) Dscodebench: A realistic benchmark for data science code generation. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp 32628–32636

Peng Q, Chai Y, Li X (2024) Humaneval-xl: A multilingual code generation benchmark for cross-lingual natural language generalization. In: Proceedings of the 2024 joint international conference on computational linguistics, language resources and evaluation (LREC-COLING 2024), pp 8383–8394

Pereira DG, Afonso A, Medeiros FM (2015) Overview of friedman’s test and post-hoc analysis. Communications in Statistics-Simulation and Computation 44(10):2636– 2653

Pu S, Wang Y, Chen D, et al (2025) Judge anything: Mllm as a judge across any modality. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp 5742–5753

Qi H, Ye S, Mathis A, et al (2025) Llavaction: evaluating and training multi-modal large language models for action understanding. arXiv preprint arXiv:250318712

Qwen Team (2025) Qwen2.5-vl. URL https://qwenlm.github.io/blog/qwen2.5-vl/

Qwen Team (2026a) Qwen3.5: Towards native multimodal agents. URL https://qwen. ai/blog?id=qwen3.5

Qwen Team (2026b) Qwen3.6-Plus: Towards real world agents. URL https://qwen.ai/ blog?id=qwen3.6

Qwen Team (2026c) Qwen3.8-max: A new bar for coding and cowork. URL https: //qwen.ai/blog?id=qwen3.8

Rong D, Chen Z, Jia Q, et al (2025) Liveproteinbench: A contamination-free benchmark for assessing models’ specialized capabilities in protein science. arXiv preprint arXiv:251222257

Rosenfeld A, Pfaltz JL (1966) Sequential operations in digital picture processing. Journal of the ACM (JACM) 13(4):471–494

Rosenholtz R, Li Y, Nakano L (2007) Measuring visual clutter. Journal of vision 7(2):17–17

Si C, Zhang Y, Li R, et al (2025) Design2code: Benchmarking multimodal code generation for automated front-end engineering. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational

Linguistics: Human Language Technologies (Volume 1: Long Papers), pp 3956–3974

Szot A, Mazoure B, Agrawal H, et al (2024) Grounding multimodal large language models in actions. Advances in Neural Information Processing Systems 37:20198– 20224

Team G, Anil R, Borgeaud S, et al (2023) Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:231211805

Team G, Georgiev P, Lei VI, et al (2024) Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:240305530

Tu Z, Wang Y, Birkbeck N, et al (2021) Ugc-vqa: Benchmarking blind video quality assessment for user generated content. IEEE Transactions on Image Processing 30:4449–4464

Wang K, Pan J, Shi W, et al (2024a) Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems 37:95095–95169

Wang P, Bai S, Tan S, et al (2024b) Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:240912191

Wang W, Gao Z, Gu L, et al (2025a) Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:250818265

Wang W, He Z, Hong W, et al (2025b) Lvbench: An extreme long video understanding benchmark. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 22958–22967

Wang Z, Bovik AC, Sheikh HR, et al (2004) Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4):600– 612

Wang Z, Xia M, He L, et al (2024c) Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems 37:113569–113697

Wang Z, Chen W, Yang L, et al (2025c) Mp-gui: Modality perception with mllms for gui understanding. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp 29711–29721

White C, Dooley S, Roberts M, et al (2025) Livebench: A challenging, contaminationfree llm benchmark. In: The Thirteenth International Conference on Learning Representations

Wu C, Liang Z, Ge Y, et al (2025a) Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific

plots. In: Findings of the Association for Computational Linguistics: NAACL 2025, pp 3006–3028

Wu D, Liu F, Hung YH, et al (2025b) Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

Wu H, Li D, Chen B, et al (2024) Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems 37:28828–28857

xAI (2024) Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model. https://x.ai/news/grok-1.5v, accessed: 2026-09-09

xAI (2026) Grok 4.3. https://docs.x.ai/developers/models/grok-4.3, accessed: 2026-06- 09

Xia R, Ye H, Yan X, et al (2025) Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. IEEE Transactions on Image Processing

Xu J, Guo Z, Hu H, et al (2025) Qwen3-omni technical report. arXiv preprint arXiv:250917765

Xu P, Shao W, Zhang K, et al (2024) Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(3):1877–1893

Yang C, Shi C, Liu Y, et al (2025) Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. In: The Thirteenth International Conference on Learning Representations

Yang Y, Ming J, Yu N (2012) Color image quality assessment based on ciede2000. Advances in Multimedia 2012(1):273723

Yang Z, Zhou Z, Wang S, et al (2024) Matplotagent: Method and evaluation for llm-based agentic scientific data visualization. In: Findings of the Association for Computational Linguistics: ACL 2024, pp 11789–11804

Yue X, Zheng T, Ni Y, et al (2025) Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp 15134–15186

Zeng A, Lv X, Hou Z, et al (2026) Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:260215763

Zhang R, Isola P, Efros AA, et al (2018) The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 586–595

Zhang Z, Chen Z, Zhang Z, et al (2025a) Puzzlebench: A fully dynamic evaluation framework for large multimodal models on puzzle solving. arXiv preprint arXiv:250410885

Zhang Z, Wang J, Wen F, et al (2025b) Large multimodal models evaluation: A survey. Science China Information Sciences 68(12):221301

Zhao X, Luo X, Shi Q, et al (2025) Chartcoder: Advancing multimodal large language model for chart-to-code generation. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp 7333–7348

Zheng Y, Chen Z, Jia Q (2026) Benchmarking cross-scale perception ability of large multimodal models in material science. arXiv preprint arXiv:260319327

Zheng Z, Cheng Z, Shen Z, et al (2025) Livecodebench pro: How do olympiad medalists judge llms in competitive programming? In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track

Zhou S, Alon U, Agarwal S, et al (2023) Codebertscore: Evaluating code generation with pretrained models of code. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp 13921–13937

Zhou Y, Wang Y, He X, et al (2025) Scientists’ first exam: Probing cognitive abilities of mllm via perception, understanding, and reasoning. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track

Zou Y, Zhu D, Zhu L, et al (2026) Intern-s1-pro: Scientific multimodal foundation model at trillion scale. arXiv preprint arXiv:260325040

Zuse H (2019) Software complexity: measures and methods, vol 4. Walter de Gruyter GmbH & Co KG