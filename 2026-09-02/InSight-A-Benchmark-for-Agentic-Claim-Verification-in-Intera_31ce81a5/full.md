# InSight: A Benchmark for Agentic Claim Verification in Interactive Visualizations

Maeve Hutchinson<sup>1</sup>, Syed Mahbubul Huq<sup>1</sup>, Mohammad Albinhassan<sup>2</sup>, Radu Jianu<sup>1</sup>, Aidan Slingsby<sup>1</sup>, Pranava Madhyastha<sup>1,3,2</sup>

<sup>1</sup>City, University of London, <sup>2</sup>Imperial College London, <sup>3</sup>The Alan Turing Institute

Correspondence: {maeve.hutchinson, pranava.madhyastha}@city.ac.uk

## Abstract

Vision Language Models have demonstrated remarkable proficiency in interpreting static visual artifacts, but modern data analysis is inherently dynamic, requiring the active interrogation of interactive environments. Existing benchmarks are predominantly constrained to static imagery and one-shot question answering and fail to capture the epistemic demands of this domain, where evidence is frequently occluded, distributed across linked views, or conditionally revealed through user agency. In this paper, we introduce InSight, a benchmark for agentic claim verification over interactive visualizations. The dataset consists of 21,349 claims derived from human-authored analytical narratives and grounded in fully interactive web-based environments. Agents must navigate these environments to determine whether a natural language claim is supported, refuted or not verifiable given the available evidence. Unlike traditional evaluations, InSight treats interaction traces as intrinsic proxies for reasoning, enabling a rigorous audit of how models seek and synthesize visual evidence. We evaluate state-of-the-art models, revealing that interactive verification remains a non-trivial challenge. We release InSight at https://github.com/ maevehutch/insight.

## 1 Introduction

The recent advances of vision-language models (VLMs) have shown impressive performance in chart interpretation and visual reasoning (e.g., Google, 2025b; OpenAI, 2025). These systems are being increasingly integrated into analytical workflows, evolving from mere curiosities into fully fledged intelligent assistants tasked with conducting visual analysis, generating design advice and captioning complex imagery (Kim et al., 2025; Kaur et al., 2025). However, while the deployment of these models has indeed accelerated, the premise that they possess a robust, ground-level understanding of visual data remains contentious. Recent empirical evidence suggests that despite their fluency, VLMs frequently lack fundamental visualization literacy, which is the ability to accurately read, understand, and interpret encoded data without relying on parametric knowledge. Hong et al. (2025) demonstrate that state-of-the-art models often fail to ground their answers in visual features, instead confabulating responses based on textual context or prior training data, particularly when confronted with modified or decontextualised charts.

This discrepancy between perceived capability and actual visual grounding is exacerbated by the limitations of current evaluation methodologies. The evaluation of these models has traditionally relied on static benchmarks, where the model is presented with a fixed image of a chart and queried via a question-answer pair (Methani et al., 2020; Masry et al., 2022) or a summarisation task or captioning task (Tang et al., 2023). While valuable for assessing perceptual acuity in closed settings, this static paradigm ignores the fundamental ecological validity of the process of data analysis where meaningful visualizations are rarely static artifacts. They are interactive environments, such as dashboards, faceted browsers, and exploratory tools, where information is conditionally revealed through user agency. In real-world analytical scenarios, evidence is frequently occluded behind tooltips, distributed across linked views, or accessible only through filtering and zooming.

In this paper, we introduce InSight, a benchmark for agentic claim verification over interactive visualizations. InSight reframes visual reasoning as a sequential decision-making process. We present an evaluation framework where an agent must actively navigate web-based visualization environments through executing actions such as clicking, hovering, and scrolling to verify data-related claims. Our work departs from most previous attempts in building visualization understanding benchmarks in two key ways. Firstly, unlike datasets that rely on synthetic templates or task setups (Methani et al., 2020; Masry et al., 2022), or crowd-sourced descriptions of static images (Akhtar et al., 2024), InSight is fundamentally grounded in human-authored analytical notebooks focused on real, analytical, data exploration. We extract claims directly from expert narratives and preserving the corresponding custom vega-lite visualizations (Satyanarayan et al., 2017), allowing us to preserve authentic analytical workflows rather than contrived proxies. Secondly, InSight treats the interaction trace as a first-class citizen. In traditional text-based fact-checking (Thorne et al., 2018; Aly et al., 2021) or static visual question answering (Methani et al., 2020; Masry et al., 2022), the reasoning process is often opaque, inferred only from post-hoc interpretability methods or chain of thought as a proxy. In InSight, the sequence of actions, such as, the mouse interaction or the navigation path taken, provides an explicit, traceable proxy for the model’s attentional focus and reasoning strategy. This allows us to determine not only if a model is correct, but whether it engaged with the necessary evidence to justify its conclusion. Our experiments with state-of-the-art models reveal that interactive verification remains a difficult challenge for all models. We also demonstrate that interaction is particularly critical for falsification, where targeted visual evidence is required to overturn lin guistically plausible but unsupported claims. We also observe that high-performing models exhibit distinct interaction signatures, engaging in deeper, more targeted exploration of the visual topology.

We make the following contributions: We formulate the task of Interactive Visual Claim Verification, extending the epistemic requirements of factchecking to agentic, multimodal environments. We present InSight, an ecologically valid benchmark comprising over 21k claims grounded in humanauthored interactive visualizations, explicitly designed to penalise passive perception. We provide a comprehensive evaluation of current VLMs, offering a granular analysis of how interaction strategies correlate with reasoning fidelity.

## 2 Background

## 2.1 Agentic Vision-Language Models

Recent work has explored VLMs deployed as agents that operate in interactive environments, such as web browsers, desktop GUIs, and simulated interfaces. Benchmarks such as OSWorld (Xie et al., 2024), MiniWoB++ (Furuta et al., 2024), VisualWebArena (Koh et al., 2024) and Browser-Gym (Chezelles et al., 2025) frame perception and action as a sequential decision-making problem, where models must execute actions (e.g., clicking, scrolling, typing) to complete complex tasks. These efforts have driven rapid progress in generalpurpose multimodal agents and highlighted the challenges of long-horizon planning, grounding, and robustness in realistic environments.

However, existing agentic benchmarks primarily evaluate task completion, with success defined by whether a goal state is reached. In contrast, our work focuses on a single, tightly scoped task, visual claim verification, where interaction is used explicitly to acquire evidence, enabling controlled analysis of how agents seek, evaluate, and integrate information across modalities.

## 2.2 Chart Understanding

A large body of work addresses question answering and reasoning over charts and visualizations. Early datasets such as FigureQA (Kahou et al., 2018) and DVQA (Kafle et al., 2018) rely on synthetically generated charts and templated questions, enabling controlled evaluation of specific reasoning skills but offering limited ecological validity. Later datasets, including PlotQA (Methani et al., 2020) and ChartQA (Masry et al., 2022), move toward greater realism by using real or semi-real data and more diverse visual styles, but still frame the task as one-shot question answering over static images.

Datasets such as ChartCheck (Akhtar et al., 2024) focus on chart reasoning through claim verification rather than QA, requiring models to determine whether a textual statement is supported or contradicted by a chart. This shift aligns more closely with real-world misinformation scenarios, but these datasets continue to rely on static visual inputs and typically involve claims written or mutated specifically for the benchmark. As a result, they do not capture how analysts actually interact with visualizations to uncover evidence, nor do they reflect the compositional, multi-view, and interactive nature of real analytical dashboards.

InSight differs from prior chart datasets in two key respects: it preserves interactive visualization environments rather than static renderings, and its claims are grounded in human-authored analytical narratives, rather than being synthetically constructed to fit a predefined schema. This allows the benchmark to capture realistic tasks that are absent from existing chart understanding datasets.

## 2.3 Claim Verification and Fact-Checking

Claim verification has been extensively studied in text-only settings, most notably in datasets such as FEVER (Thorne et al., 2018), SciFact (Wadden et al., 2020), and FEVEROUS (Aly et al., 2021). These benchmarks emphasize evidence retrieval and textual entailment, introducing the Not Enough Information (NEI) label to capture epistemic uncertainty. In these datasets, relevant evidence is explicitly available in textual form and do not model the process of actively uncovering evidence and updating beliefs about claims.

Multimodal fact-checking datasets extend this paradigm to images. Datasets such as NewsCLIPpings (Luo et al., 2021) and AVerImaTeC (Cao et al., 2025) require models to assess the consistency across textual and image modalities often in closed-world settings where all evidence is contained in the input. These tasks emphasize crossmodal alignment and visual-textual reasoning but still rely on static inputs and do not support sequential evidence acquisition.

InSight bridges these lines of work by combining the epistemic framing of claim verification, including NEI supervision, with interactive visual environments. In this setting, the environment is partially observable with evidence distributed across views, or conditionally revealed through interaction, introducing challenges that are absent from both text-only verification and static multimodal fact-checking.

## 3 InSight: Task and Dataset

## 3.1 Task Definition

We define interactive claim verification over visualizations as a decision-making task in which a model or a human must determine the veracity of a natural language claim, c, in relation to an interactive visualization environment, E.

Each instance in the dataset consists of a natural language claim made regarding the data presented in the environment, and the environment itself. The latter is rendered as a web-based interface containing multiple views (e.g., a dashboard) and interactive UI elements.

The objective of the task is to assign one of three labels to the claim: True, meaning the visualizations support the claim; False, meaning the visualizations contradict the claim; and Not Enough Information (NEI) meaning that the visualizations do not provide sufficient evidence to either verify or refute the claim.

The environment E is partially observable. Relevant visual evidence may be occluded behind interactions, distributed across multiple views or panels, or conditionally revealed based upon prior states (of interactions). Thus, to access this evidence, the model must execute a sequence of actions $a _ { 1 } , . . . , a _ { T }$ from a fixed action space (e.g., clicking, hovering, scrolling) to update the environment state and generate new visual observations. The model operates within a turn-based interaction loop with a bounded budget of actions. We emphasise that no textual metadata regarding the underlying data is provided; all verification must be grounded in pixel-level evidence revealed through interaction.

We formalize this interactive claim verification task as a process of sequential evidence acquisition. As each action potentially exposes new information, the resulting interaction trace serves as an intrinsic proxy for the model’s internal reasoning process, revealing what information the model deems relevant, how it navigates the topology of the visualization, and the stopping criteria it employs when judging evidence sufficiency.

## 3.2 Dataset Construction

InSight is derived from a corpus of human-authored analytical notebooks that combine interactive Vega-Lite (Satyanarayan et al., 2017) visualizations with natural language narratives describing the analytical insights they reveal (Wood et al., 2019). These notebooks were authored independently by analysts with formal training in data visualization, who selected their own datasets, formulated analytical questions, and designed custom visualizations. As a result, the corpus reflects authentic analytical reasoning rather than constructed descriptions. We apply rigorous filtering criteria, retaining N = 297 notebooks for downstream processing.

The construction pipeline proceeds in four stages. Full implementation details, including all prompts, thresholds, and lexicons, are provided in Appendix A.

Stage 1: Span Extraction. We extract candidate spans from each narrative using Lundgard and Satyanarayan (2022)’s four-level semantic model, retaining only spans at levels 2 (statistical insights) and 3 (visual/perceptual insights), which express content verifiable against visualizations. To ensure extraction stability, we perform three independent LLM runs per text segment (Wang et al., 2023) and consolidate spans via a multi-stage agreement procedure based on trigram overlap. Spans extracted by only a single run are discarded, and the final span text is constructed from tokens selected by at least two runs. This procedure filters out hallucinated extractions and semantically irrelevant text.

Stage 2: Claim Decomposition. Retained spans may contain compound statements or implicit references. Following Metropolitansky and Larson (2025), we decompose each span into atomic, fully decontextualized propositions that can be independently verified against the visualizations. A selfverification step confirms that each decomposed claim is entailed by the source text (Weng et al., 2023), and semantic labels are reassigned via majority vote across three passes. Claims that fail grounding verification or exhibit unstable labeling are discarded. The resulting claims constitute the ground-truth TRUE set.

Stage 3: Claim Mutation. To construct a balanced verification benchmark, we generate FALSE and NEI claims through controlled mutation of the true claims, following the contrastive construction paradigm of Thorne et al. (2018). We apply three mutation strategies, each targeting a distinct source of semantic change:

• Antonym substitution reverses directional or quantitative language (e.g., increased → decreased) using a curated antonym lexicon, producing FALSE claims.

• In-lexicon argument substitution replaces data arguments (categorical values, numerical quantities) with alternatives from the same dataset, producing FALSE claims.

• Out-of-lexicon argument substitution replaces arguments with values or attributes absent from the dataset, producing NEI claims that refer to entities for which no visual evidence exists.

All mutations are constrained by XML-style tagging of substitutable spans to prevent lexical drift and preserve structural similarity with the source claim. Crucially, each mutated claim is validated using a pretrained NLI model (He et al., 2023): FALSE claims must exhibit high bidirectional contradiction scores (> 0.9) with the original, while NEI claims must exhibit low entailment and low contradiction (< 0.2). Claims failing validation are discarded.

Stage 4: Human Validation. We conduct a human validation study with 13 expert annotators, who produced 475 annotations across 294 sampled claims. Annotators performed the same interactive verification task defined in §3.1, blind to the original labels and construction process. Raw agreement with the dataset labels is 81.3%. A Bayesian Dawid–Skene model (Dawid and Skene, 1979; Paun et al., 2018) applied to the annotation data recovers class prevalences within 3 percentage points of the ground truth, with a posterior mean accuracy of 66.2% (95% CI: [59.5%, 71.8%]) and an MAP estimate of 77.6%. These results confirm that the controlled construction and NLI-based validation procedures produce labels broadly consistent with expert human judgment, despite the inherent difficulty of the task.

## 3.3 Dataset Analysis

InSight contains 21,349 claims derived from 297 human-authored analytical notebooks. Claims are distributed across the three verification labels, with 41.6% True, 45.0% False, and 13.4% Not Enough Information (NEI) cases.

Claim properties. Claims are concise but nontrivial, with an average length of 18 tokens (median 17). 56.8% correspond to semantic level 2, relating to statistical and numerical insights (e.g., value retrieval, extrema), and 43.2% to level 3, relating to visual and perceptual insights (e.g., trends, patterns). This distribution reflects a mix of precise data lookup and higher-order relational reasoning, both of which are common in real analytical workflows.

Visualization diversity. The visualizations are authored in Vega-Lite (Satyanarayan et al., 2017), a declarative grammar based on composable marks and encodings. As a result, the space of visualization forms is not bounded by a predefined taxonomy. Across the corpus, visualizations combine a wide range of marks—bars, lines, points, areas, geoshapes, text, and rules—and frequently involve layering and multi-view composition. On average, each notebook contains 4.5 views and 3.7 visualization specifications, with 67% of notebooks exhibiting explicit multi-view composition. This means that part of the verification task is determining which visualization is most relevant to the claim, a challenge absent from single-chart bench-

marks.

Interaction characteristics. Interactivity is intrinsic to the visualization environments in InSight. The most prevalent mechanism is hover-triggered tooltips, which conditionally reveal details-ondemand, such as precise values, labels, or additional information. Click-based and point-selection interactions are also common, enabling filtering, highlighting, or cross-view coordination. Less frequent primitives (interval selection, zoom/pan) nevertheless play an important role for claims requiring range-based or spatial reasoning. Importantly, these statistics capture interaction capabilities designed into the visualizations themselves; scrolling and basic mouse actions are additionally available as part of the agent action space (§4.1).

Partial observability and interaction necessity. Every environment in InSight is partially observable, and therefore every claim requires exploration of the environment. Environments contain on average 4.5 views and 3.7 specifications (Figure 3), and cannot be rendered in a single fixed viewport, so the initial screenshot never exposes the full set of visualizations. Even when the visualization relevant to a claim happens to appear first, an agent cannot know this without inspecting the remaining views: it must establish that no other view supplies stronger, contradicting, or disambiguating evidence, and for NEI claims it must establish that no view supplies evidence at all. In addition, the most prevalent interaction mechanism, hover-triggered tooltips, hides precise values behind an action, so even claims about a visible chart typically cannot be resolved from the static rendering.

Additionally, visualizations here were built by analysts to explore their own datasets, and the claims were derived from insights they reached through interactive use, not from what a single frame conveys. Consequently, no evaluation claim is verifiable from the initial viewport alone, and a correct answer produced without interaction should be attributed to parametric knowledge or guessing rather than to visual grounding. This property is what allows InSight to penalize passive perception, and it motivates the design of the Interaction Efficiency Score in 4.2.

Comparison with prior datasets. Unlike existing chart understanding and claim verification benchmarks, which typically rely on static images or preextracted data, InSight jointly supports: (i) fully interactive visual environments, (ii) claims grounded in real-world analytical provenance rather than synthetic construction, (iii) explicit NEI supervision, and (iv) the collection of action traces during verification. See Appendix C for additional distributional analyses, including mark-type frequencies and specification density histograms.

## 4 Evaluation and Behavioural Analysis

## 4.1 Experimental Setup

Evaluation Environment. Each visualization environment E is rendered as an HTML page containing one or more interactive visualizations authored in Vega-Lite and executed in a Chromium-based headless browser using Playwright with a fixed viewport. This setup renders the visualization notebooks exactly as a human expert would see them and ensures consistent rendering and interaction behavior across models. The fixed viewport is deliberately smaller than the rendered page: because notebooks contain multiple views (§3.3), only a portion of the environment is visible at any time, and the remainder must be reached by scrolling, while details-on-demand (tooltips, selections) must be triggered by mouse actions. Every environment is therefore partially observable, and no claim in the evaluation set can be verified from the initial screenshot alone, regardless of which visualization contains the relevant evidence.

Crucially, we treat the web page as a purely visual surface. The model is provided only with the RGB screenshot of the current viewport; it does not have access to the underlying DOM tree, HTML source code, or accessibility tree. This design choice aligns with recent work in GUI agents (e.g. Qin et al., 2025; Luo et al., 2025), ensuring that the model must ground its reasoning in the visual rendering of data rather than parsing textual data representations embedded in the code.

Agent Action Space. The model interacts with the environment through a defined high-level action space A, drawing on prior work in web and UI agent benchmarks (Xie et al., 2024; Qin et al., 2025, interalia). The action space mimics standard mouse and keyboard inputs, requiring the model to predict pixel coordinates (x, y).

We define the action space A as follows: Navigation: scroll(dx,dy) for precise pixel scrolling, and macro-actions such as page\_up(n), page\_down(n), and directional arrow keys (e.g., arrow\_right(n)) to navigate the webpage. Mouse Interaction: drag((x1,y1), (x2,y2)), click(x,y), shift\_click(x,y), and

The sum of crimes in London decreased between 2015 and 2016

![](images/622eb69004b0bacc52672ab514f9626c53df7048624df2f17094051e00ae6743.jpg)  
Figure 1: Interaction trace for Gemini 3.5 Flash verifying a false claim about London crime trends. The model hovers to inspect a tooltip, then scrolls through the multi-view notebook to locate relevant evidence, correctly predicting FALSE with an IES of 0.833.

hover $( x , y )$ These are essential for interacting with visualizations to reveal more evidence, such as tooltip inspection (hover), range selection (drag), or multi-selection (shift-click). Termination: answer(label), where label ∈ {True, False, NEI}. This action terminates the episode and submits the verification decision.

Interaction Protocol. Models interact with the environment through a standard turn-based protocol. At each turn $t \leq T _ { \operatorname* { m a x } }$ , the model receives the current screenshot $s _ { t }$ and the claim c, and must output a single action $a _ { t } \in A .$ If $a _ { t }$ is not a terminal answer action, the environment executes a<sub>t</sub>, updates its state, and provides the next screenshot $s _ { t + 1 }$ . The process continues until the model outputs an answer action or the maximum turn budget $T _ { \mathrm { m a x } }$ is reached.

## 4.2 Evaluation Metrics

We introduce an Interaction Efficiency Score (IES), designed to capture whether agents interact productively with the visualization environment while performing claim verification. Existing agentic benchmarks primarily evaluate task completion alone, whereas InSight explicitly treats interaction traces as part of the reasoning process. Our metric is conceptually related to Success weighted by Path Length (SPL) (Anderson et al., 2018) used in embodied navigation benchmarks, but adapted to browser-based visual evidence acquisition.

For an episode i, let $A _ { i } \in [ 0 , 1 ]$ denote answer accuracy, where $A _ { i } = 1$ if the predicted label is correct and $A _ { i } = 0$ otherwise. Let $T _ { i }$ denote the total number of actions executed by the agent, and let $E _ { i }$ denote the number of effective actions. We define an effective action as an action that produces an observable change in the environment state, i.e., interactions that reveal new visual evidence.

Formally, we compute effective actions as:

$$
E _ { i } = \sum _ { t = 1 } ^ { T _ { i } } \mathbf { 1 } [ s _ { t + 1 } \neq s _ { t } ]\tag{1}
$$

where $s _ { t }$ denotes the observable environment state after action t, and 1[·] is the indicator function.

We then define the Interaction Efficiency Score as:

$$
\mathrm { I E S } _ { i } = A _ { i } \cdot { \frac { E _ { i } } { T _ { i } } }\tag{2}
$$

where $\frac { E _ { i } } { T _ { i } }$ is the state change ratio (SCR) and we report the dataset-level score as:

$$
\mathrm { I E S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } A _ { i } \frac { E _ { i } } { T _ { i } }\tag{3}
$$

This formulation rewards agents that both answer correctly and interact productively with the environment. Incorrect answers receive a score of

0 regardless of interaction behavior, while correct predictions are additionally differentiated by the proportion of actions that meaningfully advance the observable state of the environment. IES thus provides a behavioural signal indicating whether an agent’s interaction trajectory acquires visual evidence during verification, complementing accuracy rather than replacing it.

Correct answers without interaction. An agent that submits an answer at the first turn has $T _ { i } = 0$ and $E _ { i } = 0 ;$ we define $E _ { i } / T _ { i } = 0$ in this case, so the episode receives IES = 0 even if the answer is correct. This is deliberate: since no claim is resolvable from the initial screenshot, a correct answer with no interaction reflects parametric knowledge or chance rather than visual grounding, which is precisely the failure mode IES is meant to expose. Qwen 3.5 0.8B, which never interacts, receives IES = 0 for this reason.

Scrolling. Scrolling almost always changes the rendered pixels and is therefore almost always counted as effective. This is by design, because environments contain more views than fit in a viewport, scrolling is the action by which an agent discovers where relevant evidence resides and by which it returns to a relevant view after surveying the alternatives.

NEI claims. For an NEI claim, the ideal trajectory is to explore the environment and find nothing. Such exploration still changes the observable state and is counted as effective. Since for NEI claims there is by construction no evidence directly relevant to the claim’s veracity, no metric can score whether the answer depended on revealed evidence, therefore IES scores whether the agent looked. The same holds to a lesser degree for True and False claims, because the agent cannot know in advance which view is relevant, actions that reveal irrelevant views are still useful exploratory work in a partially observable environment, and we intentionally do not penalize them.

Limitation. IES credits any state-changing action regardless of whether the final answer used the revealed evidence. It therefore cannot fully rule out correct-but-ungrounded predictions by agents that interact but ignore what they see. IES identifies agents that do not interact at all, and ranks interacting agents by how much of their budget produces new observations. Finer-grained grounding would require claim-specific evidence annotations and is a natural extension we leave for future work.

## 4.3 Model Results

We benchmark a selection of closed-source and open-weight state-of-the-art models. We evaluate models on a stratified subset of 500 claims, a similar sample size to equivalent multi-turn agentic benchmarking (Xie et al., 2024). The subset consists of 167 True, 167 False, and 166 NEI claims. We do not benchmark across the whole dataset to avoid potential data contamination issues and reserve samples for future rigorous benchmarking (Deng et al., 2024).

We evaluate fourteen model configurations spanning four model families: Gemini, GPT, Gemma, and Qwen. For all models we use an interaction budget of 10, following other similar CUA benchmark action budgets (Xie et al., 2024). We additionally do ablations with interaction budgets of $( T _ { \mathrm { m a x } } \in \{ 1 , 2 5 \}$ ) for some models to compare behavior across action budgets. Table 1 reports accuracy, interaction volume $( T / N$ , the mean number of actions per episode), the effective action ratio $( E / T )$ , and the Interaction Efficiency Score (IES). Both accuracy and IES are reported overall and disaggregated by claim label.

Overall Performance At $T _ { m a x } = 1 0$ , GPT-5.5 achieves the highest overall accuracy at 57.2%, followed by Gemma 4 31B (47.2%) and Qwen 3.5 27B (45.0%). These results are notable given that random performance on a balanced three-class task is 33.3%, indicating that all models meaningfully exceed chance, but none approach human-level reliability. A clear scaling trend is visible within model families: Qwen accuracy degrades monotonically from 44.8% at 35B parameters to 34.0% at 0.8B, and Gemma drops from 47.2% (31B) to 31.2% (E2B). At the smallest scales, models barely exceed random baselines, suggesting that interactive visual reasoning imposes a minimum capacity threshold that sub-billion parameter models consistently fail to meet.

The Role of Interaction The $T _ { \mathrm { m a x } } = 1$ setting serves as a no-interaction baseline. In this ablation, the model receives the claim and the initial viewport and must answer at the first turn, which reduces the task to static visual question answering over a partial view of the environment. Performance in this setting reflects what the model can infer from parametric knowledge, linguistic plausibility of the claim, and whatever happens to be visible in the first screenshot. The results are mixed. Gemini 3.5 Flash is slightly more accurate without interaction (44.2% vs. 41.6%), as is Qwen 3.5 27B (46.2% vs. 45.0%), whereas Gemma 4 31B drops from 47.2% to 45.4% when interaction is removed. For the two models that do not benefit, the gains at $T _ { \mathrm { m a x } } = 1$ are within a few points and suggest that unrestricted interaction can be counterproductive when a model lacks a plan for exploration and may commit on partial evidence, or lose track of the claim across turns. That the pattern does not hold for Gemma indicates the effect is model-specific rather than a property of the task, and it is worth noting that the accuracy gap between the two settings is small for all three models, so parametric knowledge and the initial viewport account for a large share of the accuracy that interaction-capable models achieve.

Table 1: Comparison of model efficiency and interaction metrics across model families. Accuracy and IES are reported overall and disaggregated by claim label.
<table><tr><td>Model</td><td> $T _ { m a x }$ </td><td> $T / N$ </td><td>Accuracy</td><td> $\mathbf { A c c _ { F a l s e } }$ </td><td> $\mathbf { A c c y } _ { \mathbf { E I } }$ </td><td> $\mathbf { A c c _ { T r u e } }$ </td><td> $E / T$ </td><td>IES</td><td> $\mathbf { I E S } _ { \mathbf { F a l s e } }$ </td><td> $\mathbf { I E S } _ { \mathbf { N E I } }$ </td><td> $\mathbf { I E S _ { T r u e } }$ </td></tr><tr><td>Gemini 3.5 Flash</td><td>25</td><td>9.33</td><td>50.0%</td><td>54.8%</td><td>50.6%</td><td>44.6%</td><td>56.7%</td><td>31.0%</td><td>29.87%</td><td>36.10%</td><td>27.03%</td></tr><tr><td>Gemini 3.5 Flash</td><td>10</td><td>6.27</td><td>41.6%</td><td>45.8%</td><td>45.8%</td><td>33.3%</td><td>61.3%</td><td>24.93%</td><td>23.62%</td><td>32.63%</td><td>18.63%</td></tr><tr><td>Gemini 3.5 Flash</td><td>1</td><td>一</td><td>44.2%</td><td>33.1%</td><td>73.5%</td><td>26.2%</td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5</td><td>10</td><td>3.13</td><td>57.2%</td><td>59.6%</td><td>59.0%</td><td>53.0%</td><td>45.4%</td><td>26.98%</td><td>29.24%</td><td>23.98%</td><td>27.70%</td></tr><tr><td>Gemma 431B</td><td>10</td><td>3.16</td><td>47.2%</td><td>40.4%</td><td>63.9%</td><td>37.5%</td><td>43.8%</td><td>20.57%</td><td>15.12%</td><td>28.48%</td><td>18.13%</td></tr><tr><td>Gemma 431B</td><td>1</td><td></td><td>45.4%</td><td>26.5%</td><td>84.9%</td><td>25.0%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma 4 E4B</td><td>10</td><td>0.89</td><td>43.2%</td><td>18.7%</td><td>88.0%</td><td>23.2%</td><td>18.3%</td><td>6.87%</td><td>2.81%</td><td>15.66%</td><td>2.18%</td></tr><tr><td>Gemma 4 E2B</td><td>10</td><td>0.16</td><td>31.2%</td><td>7.8%</td><td>71.7%</td><td>14.3%</td><td>5.0%</td><td>0.90%</td><td>0.00%</td><td>2.71%</td><td>0.00%</td></tr><tr><td>Qwen 3.6 35B</td><td>10</td><td>1.78</td><td>44.8%</td><td>42.8%</td><td>62.7%</td><td>29.2%</td><td>21.9%</td><td>9.07%</td><td>7.54%</td><td>12.67%</td><td>7.01%</td></tr><tr><td>Qwen 3.5 27B</td><td>10</td><td>1.57</td><td>45.0%</td><td>38.0%</td><td>66.3%</td><td>31.0%</td><td>17.7%</td><td>7.57%</td><td>5.65%</td><td>10.20%</td><td>6.87%</td></tr><tr><td>Qwen 3.5 27B</td><td>1</td><td></td><td>46.2%</td><td>35.5%</td><td>79.5%</td><td>23.8%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen 3.5 9B</td><td>10</td><td>0.50</td><td>38.0%</td><td>50.6%</td><td>26.5%</td><td>36.9%</td><td>9.56%</td><td>3.52%</td><td>4.67%</td><td>3.82%</td><td>2.08%</td></tr><tr><td>Qwen 3.5 2B</td><td>10</td><td>0.24</td><td>37.0%</td><td>30.7%</td><td>51.8%</td><td>28.6%</td><td>4.10%</td><td>1.17%</td><td>1.20%</td><td>0.24%</td><td>2.06%</td></tr><tr><td>Qwen 3.5 0.8B</td><td>10</td><td>0.00</td><td>34.0%</td><td>21.7%</td><td>54.8%</td><td>25.6%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td></tr></table>

The Gemini budget ablation is informative here because it is non-monotonic: 44.2% at $T _ { \mathrm { m a x } } = 1$ 41.6% at 10, and 50.0% at 25. Under the 10-action budget Gemini is the most active model, and many of its trajectories are still mid-exploration when the budget forces an answer or when the model chooses to stop. A partially completed survey of a multiview environment appears to be worse than none, presumably because the model has seen enough to abandon its prior but not enough to replace it. Given 25 actions it uses 9.33 on average, completes more of its exploration, and reaches the highest accuracy of any Gemini configuration, with IES rising from 24.93% to 31.0% even though its effective action ratio falls slightly, consistent with later actions being used to return to and inspect views already discovered.

The interaction traces reveal a pronounced disparity in how models allocate their action budgets. Gemini 3.5 Flash is the most active agent, averaging 6.27 actions per episode with an effective action ratio of 61.3%, indicating that a majority of its interactions produce observable state changes. GPT-5.5 is comparatively economical, averaging 3.13 actions but achieving the highest accuracy, suggesting more deliberate and targeted exploration. At the other extreme, Qwen 3.5 0.8B averages zero actions per episode, defaulting to immediate answer submission without any environmental interaction, a behaviour that reduces the task to uninformed guessing from the initial viewport.

Interaction Efficiency The IES metric disentangles accuracy from interaction quality, rewarding models that both answer correctly and interact productively. GPT-5.5 leads with an IES of 26.98%, followed closely by Gemini 3.5 Flash (24.93%) and Gemma 4 31B (20.57%). The gap between accuracy and IES rankings is informative: while Gemma 4 31B achieves comparable accuracy to Gemini 3.5 Flash, its lower effective action ratio reduces its IES, indicating that a substantial proportion of its actions fail to advance the observable environment state, meaning it is not making use of the visual information.

Small models exhibit near-zero IES scores. Gemma 4 E2B achieves 0.90% and Qwen 3.5 0.8B achieves 0.00%, indicating that even when these models occasionally produce correct answers, they do so without meaningful environmental engagement. This finding validates the design of IES as a metric that penalizes correct-but-ungrounded predictions, which accuracy alone cannot detect.

Label-Specific Analysis Disaggregating IES by claim label reveals asymmetric difficulty across verification categories. For GPT-5.5, IES is relatively balanced across labels, suggesting robust interaction strategies across all claim types. In contrast, most other models exhibit a characteristic pattern where NEI claims yield the highest IES. Gemini 3.5 Flash achieves $\mathrm { I E S _ { N E I } } = 3 2 . 6 3 \%$ but only $\mathrm { I E S } _ { \mathrm { T r u e } } = 1 8 . 6 3 \%$ , and this gap is even more pronounced in smaller models such as Gemma 4 E4B $( \mathrm { I E S _ { N E I } } = 1 5 . 6 6 \% , \mathrm { I E S _ { T r u e } } = 2 . 1 8 \% )$

This pattern admits a plausible explanation: verifying a True claim requires locating and correctly interpreting specific visual evidence, whereas recognizing that evidence is insufficient for an NEI claim may require less precise interaction; the model needs only to fail to find confirming or contradicting evidence across the environment. Conversely, False claims, which require discovering evidence that contradicts a linguistically plausible statement, are particularly challenging for weaker models, indicating difficulty to productively acquire disconfirming evidence. This is consistent with prior observations in textual fact-checking that falsification is cognitively harder than verification, and extends that finding to the agentic multimodal setting (Thorne et al., 2018).

Actions and Behavioral Signatures Analyzing the actions taken by the models reveals qualitative differences in exploration strategies. Gemini 3.5 Flash employs the broadest action vocabulary, with substantial use of scroll (2.51 per sample), hover (1.42), click (1.15), and page\_down (0.73), as well as non-trivial use of arrow keys and drag actions. This diverse repertoire enables it to achieve the highest effective action ratio. GPT-5.5 similarly prioritises hover (1.07) and scroll (1.02) but uses fewer total actions, consistent with a more efficient exploration strategy.

In contrast, smaller models exhibit a smaller range of action types taken. Gemma 4 E2B and Qwen 3.5 0.8B use almost exclusively zero actions across all types, while intermediate models such as Qwen 3.5 9B show limited use of scroll and click but negligible use of hover, which is a critical action for tooltip inspection. Given that hover-triggered tooltips are the most prevalent interaction mechanism in the InSight environments (Appendix C), this omission likely explains these models’ inability to access conditionally revealed evidence.

Failure Modes Several systematic failure patterns emerge from the interaction traces. First, premature commitment: smaller models frequently terminate after one or two actions, submitting answers before engaging with the environment. Second, unfocused exploration: some incorrect predictions by larger models involve many actions with low effective action ratios, suggesting exploration that fails to target relevant visual elements. Third, actiontype rigidity: mid-range models tend to default to scrolling and clicking without employing hover or drag actions, missing evidence encoded in tooltips or selection-based interactions.

These findings collectively demonstrate that interactive claim verification is not simply a harder version of static visual question answering. It requires coordinated perception, hypothesis formation, and action selection, which are capabilities that scale with model capacity but remain far from saturated even at the frontier.

## 5 Conclusion

We have presented InSight, a benchmark for agentic claim verification over interactive visualizations comprising 21,349 claims grounded in humanauthored analytical notebooks. By requiring models to actively navigate web-based environments to gather evidence, InSight moves beyond static visual question answering and captures the sequential, exploratory nature of real analytical workflows.

Our evaluation reveals several findings. First, interactive claim verification remains challenging: the best-performing model, GPT-5.5, achieves only 57.2% accuracy. Second, IES exposes models that answer correctly without engaging with the environment at all, which indicates ungrounded prediction rather than evidence-based reasoning. Third, falsification is disproportionately difficult, requiring targeted evidence-seeking interaction that most models struggle to produce, extending prior observations from textual fact-checking to the agentic multimodal setting.

These findings point to several directions for future work. The gap between static and interactive performance suggests that explicit training for evidence-seeking behavior, rather than one-shot visual reasoning, may be necessary. The difficulty of falsification motivates investigation into how models form and update hypotheses during multi-turn visual exploration. More broadly, InSight provides a foundation for studying multimodal argumentation in agentic systems, where the reasoning process is not opaque but traceable through interaction.

## 6 Limitations

InSight has several limitations that point to directions for future work. The benchmark assumes a fixed, high-level action space that, while grounded in prior work on agentic benchmarks, abstracts common mouse and navigation interactions. While this enables controlled comparison across models, it may not capture all interaction modalities used by humans, such as complex gesture-based interactions or semantic shortcuts provided by specific interfaces. Different action space designs may yield different interaction strategies and performance profiles.

Although InSight emphasizes interaction as a proxy for reasoning, interaction traces do not fully capture internal model deliberation. A model may perform correct actions for the wrong reasons or fail despite internally plausible reasoning. Thus, interaction traces should be viewed as complementary, not exhaustive, evidence of reasoning behavior. However, we conjecture that these traces still provide evidence for examining model behavior.

## 7 Ethical Considerations

The construction of this corpus was subject to strict institutional governance and was formally ratified by the university’s Research Ethics Committee (REC). The corpus comprises artefacts originating from the capstone projects of a graduate-level module in Advanced Data Visualisation. Unlike uncurated web-scraped datasets, which often suffer from variable quality and noisy alignment, our data collection protocol imposed a rigorous high-pass quality filter. The inclusion criteria were determined by a two-stage validation process. Each submission was assessed by a senior academic specialising in data visualisation and visual analytics. This evaluation served as our primary signal of quality; only submissions demonstrating substantial technical and design proficiency were selected for inclusion. The initial assessment was subsequently audited and validated by the departmental Academic Board, ensuring that the selection serves as a gold-standard representation of domain concepts. Following ethical approval, we engaged in a retrospective consent acquisition process. We contacted graduates of the programme to articulate the study’s objectives, specifically the contribution of their work to the advancement of multimodal machine learning. The annotation procedure was also subject to ratification by the REC. We recruited data science master’s students via email to participate in the study. Participation was voluntary and participants were not paid.

## Acknowledgments

This work was supported in part by the Alan Turing Institute under Fundamental Research Project No. PP00029.

## References

Mubashara Akhtar, Nikesh Subedi, Vivek Gupta, Sahar Tahmasebi, Oana Cocarascu, and Elena Simperl. 2024. ChartCheck: Explainable Fact-Checking over Real-World Chart Images. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 13921–13937, Bangkok, Thailand. Association for Computational Linguistics.

Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, and Amir R. Zamir. 2018. On evaluation of embodied navigation agents. Preprint, arXiv:1807.06757.

Rui Cao, Zifeng Ding, Zhijiang Guo, Michael Sejr Schlichtkrull, and Andreas Vlachos. 2025. AVer-ImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web.

Thibault Le Sellier De Chezelles, Maxime Gasse, Alexandre Drouin, Massimo Caccia, Léo Boisvert, Megh Thakkar, Tom Marty, Rim Assouel, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F. Xu, Siva Reddy, Quentin Cappart, Graham Neubig, Ruslan Salakhutdinov, Nicolas Chapados, and Alexandre Lacoste. 2025. The BrowserGym Ecosystem for Web Agent Research. arXiv preprint. ArXiv:2412.05467 [cs].

A. P. Dawid and A. M. Skene. 1979. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. Journal ofthe Royal Statistical Society. Series C (Applied Statistics), 28(1):20–28. Publisher: [Royal Statistical Society, Oxford University Press].

Chunyuan Deng, Yilun Zhao, Yuzhao Heng, Yitong Li, Jiannan Cao, Xiangru Tang, and Arman Cohan. 2024. Unveiling the Spectrum of Data Contamination in Language Model: A Survey from Detection to Remediation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16078–16092,

Bangkok, Thailand. Association for Computational Linguistics.

Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. 2024. Multimodal Web Navigation with Instruction-Finetuned Foundation Models. arXiv preprint. ArXiv:2305.11854 [cs].

Google. 2025a. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. Technical Report, Google.

Google. 2025b. A new era of intelligence with Gemini 3.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv preprint. ArXiv:2111.09543 [cs].

Jiayi Hong, Christian Seto, Arlen Fan, and Ross Maciejewski. 2025. Do LLMs Have Visualization Literacy? An Evaluation on Modified Visualizations to Test Generalization in Data Interpretation. IEEE Transactions on Visualization and Computer Graphics, 31(10):7004–7018.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrialstrength Natural Language Processing in Python.

Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. 2018. DVQA: Understanding Data Visualizations via Question Answering. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5648–5656. ISSN: 2575-7075.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. 2018. FigureQA: An Annotated Figure Dataset for Visual Reasoning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings. OpenReview.net.

Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Sumitra Ganesh, and Manuela Veloso. 2025. ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering. arXiv preprint. ArXiv:2510.04514 [cs].

Nam Wook Kim, Yongsu Ahn, Grace Myers, and Benjamin Bach. 2025. How Good Is ChatGPT in Giving Advice on Your Visualization Design? ACM Trans. Comput.-Hum. Interact., 32(5):46:1–46:33.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. 2024. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1:

Long Papers), pages 881–905, Bangkok, Thailand. Association for Computational Linguistics.

Alan Lundgard and Arvind Satyanarayan. 2022. Accessible Visualization via Natural Language Descriptions: A Four-Level Model of Semantic Content. IEEE Transactions on Visualization and Computer Graphics, 28(1):1073–1083.

Grace Luo, Trevor Darrell, and Anna Rohrbach. 2021. NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6801–6817, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tiange Luo, Lajanugen Logeswaran, Justin Johnson, and Honglak Lee. 2025. Visual Test-time Scaling for GUI Agent Grounding. arXiv preprint. ArXiv:2505.00684 [cs].

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland. Association for Computational Linguistics.

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. 2020. PlotQA: Reasoning over Scientific Plots. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1516–1525. ISSN: 2642-9381.

Dasha Metropolitansky and Jonathan Larson. 2025. Towards Effective Extraction and Evaluation of Factual Claims. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6996–7045, Vienna, Austria. Association for Computational Linguistics.

OpenAI. 2025. Introducing GPT-5.2.

Silviu Paun, Bob Carpenter, Jon Chamberlain, Dirk Hovy, Udo Kruschwitz, and Massimo Poesio. 2018. Comparing Bayesian Models of Annotation. Transactions ofthe Associationfor Computational Linguistics, 6:571–585. Place: Cambridge, MA Publisher: MIT Press.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, and 1 others. 2025. Uitars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326.

Arvind Satyanarayan, Dominik Moritz, Kanit Wongsuphasawat, and Jeffrey Heer. 2017. Vega-Lite: A Grammar of Interactive Graphics. IEEE Transactions on Visualization and Computer Graphics, 23(1):341– 350.

Benny Tang, Angie Boggust, and Arvind Satyanarayan. 2023. VisText: A Benchmark for Semantically Rich Chart Captioning. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7268–7298, Toronto, Canada. Association for Computational Linguistics.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for Fact Extraction and VERification. arXiv preprint. ArXiv:1803.05355 [cs].

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific Claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, Online. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv preprint. ArXiv:2203.11171 [cs].

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2023. Large Language Models are Better Reasoners with Self-Verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2550–2575, Singapore. Association for Computational Linguistics.

Jo Wood, Alexander Kachkaev, and Jason Dykes. 2019. Design Exposition with Literate Visualization. IEEE Transactions on Visualization and Computer Graphics, 25(1):759–768.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. Osworld: Benchmarking multimodal agents for openended tasks in real computer environments. Preprint, arXiv:2404.07972.

## A Dataset Construction Details

This appendix provides full implementation details for the four-stage dataset construction pipeline summarized in §3.2.

## A.1 Data Sources and Filtering

InSight is derived from human-authored analytical notebooks that combine interactive Vega-Lite (Satyanarayan et al., 2017) visualizations with rich natural language narratives (Wood et al., 2019). These notebooks were authored independently by analysts with formal training in data visualization. They selected their own datasets, formulated analytical questions, and designed custom visualizations to explore them. The resulting notebooks reflect authentic analytical reasoning about data rather than constructed descriptions.

We apply a series of filtering criteria to ensure quality and relevance, retaining $N = 2 9 7$ notebooks. We extract two components from each: the human-authored analytical narrative, which serves as the source of grounded claims, and the corresponding interactive visualization environment.

## A.2 Span Extraction and Semantic Labeling

The analytical narratives contain a mixture of descriptive context, commentary, and claims grounded in visual evidence. We isolate spans that express verifiable claims about the underlying data using an agreement-based extraction procedure.

Candidate Span Extraction. We adopt Lundgard and Satyanarayan (2022)’s four-level semantic model of natural language about data visualizations, specifically levels 2 and 3. Level 2 captures statistical insights about the underlying datasets, such as extrema or value retrieval, while level 3 captures visual insights revealed by the visualizations, such as trends or patterns. This ensures that candidate spans relate to content that could be verified or refuted by visual evidence.

We prompt Gemini 2.5 Flash (Google, 2025a) to extract text spans from each analytical narrative containing level 2/3 semantic content. To reduce the potential impact of variability in model outputs, we perform three independent runs per text segment (Wang et al., 2023).

Agreement-Based Span Consolidation. We consolidate spans across the three model runs using a multi-stage agreement procedure. Any spans that are not extractive (i.e., not verbatim substrings of the original text) are discarded. This ensures that no new content is introduced by the LLM at this stage.

We group spans across runs based on lexical overlap by computing the word trigram F1 overlap score and greedily matching spans from different model runs with high lexical similarity, using a fixed minimum threshold for matching. Spans that are not grouped with any other span, meaning only one model run extracted that span, are discarded.

From the groups of matched spans, we construct a final span by retaining tokens chosen by at least two model runs. If this produces a non-contiguous sequence, we expand it to the minimal contiguous substring in the original text. The semantic label assigned to this span is the majority label across the runs. This procedure enforces cross-run stability, filtering out hallucinated extractions and semantically irrelevant text.

## A.3 Span Decomposition into Verifiable Claims

The spans extracted in the previous stage may contain multiple propositions, implicit references, or compound statements that are not directly suitable for claim verification. We decompose each retained span into a set of atomic, verifiable claims that can be independently assessed against the visualizations. This stage is adapted from Claimify (Metropolitansky and Larson, 2025), tailored to interactive visual evidence.

We prompt an LLM to identify all specific and verifiable propositions expressed in each span. The model is instructed to decompose compound statements into their simplest discrete units and to discard non-verifiable content. Each claim must be fully decontextualized – interpretable in isolation, without access to the surrounding narrative – while preserving its original meaning. The model is provided with the span’s immediate textual context and required to explicitly resolve all referents (e.g., pronouns, comparatives, implicit subjects). Claims are restricted to information present in the original span or its local context; no external knowledge is permitted.

To mitigate potential meaning shifts introduced by LLM-based decomposition, we apply a selfverification step in which the model is shown the original span and the decomposed claims and asked to verify that each claim is entailed by the source text (Weng et al., 2023). We also reassign semantic labels to each claim using the same semantic schema, performing 3 passes and retaining the majority label. Claims for which semantic labeling is unstable or that fail grounding verification are discarded.

The claims produced by this stage constitute the set of ground-truth TRUE claims in InSight. Each is directly derived from human-authored analytical text, verified to preserve its original meaning after decomposition and decontextualization. These true claims form the semantic anchor of the dataset and are subsequently used as the source for controlled generation of FALSE and NEI variants.

## A.4 Claim Mutation

We generate FALSE and NEI claims through controlled mutation of the true claims, applying minimal, semantics-preserving perturbations designed to alter veracity while retaining linguistic plausibility and structural similarity. This reduces annotation artifacts by ensuring that true, false, and NEI claims differ primarily in their semantic relation to the underlying visual evidence.

Antonym Substitution. This strategy targets claims whose veracity depends on scalar, directional, or quantitative language. We construct a lexicon of lemmas that have a clear antonym which, when substituted, induces a semantically opposed claim. The lexicon is curated from high-frequency lemmas in the true claim set, focusing on auxiliary verbs, determiners, adjectives, adverbs, and nouns expressing quantitative judgments (e.g., increase/decrease, all/none). Mappings are curated to avoid substitutions that merely weaken a claim or introduce ambiguity, ensuring clear semantic opposition.

We tag occurrences of lexicon entries in each claim using spaCy (Honnibal et al., 2020) and custom XML-style markers that encode the lemma identity. This constrains the mutation process, preventing lexical drift. We then prompt an LLM to generate a mutated false claim by substituting only the tagged tokens with an appropriate antonym, preserving all other content verbatim and making only minimal grammatical adjustments when strictly necessary.

Contradiction Validation. Each original– mutated claim pair is evaluated using a pretrained NLI model (He et al., 2023). We compute contradiction scores in both directions and retain only claims whose average contradiction score exceeds a threshold of 0.9. This filters out cases where antonym substitution fails to invert meaning cleanly or produces ambiguous statements.

In-Lexicon Argument Substitution. Many claims relate to specific data arguments—attributes, categorical values, or numerical quantities. We identify candidate arguments by generating a structured summary of each visualization’s underlying dataset, including attribute names, categorical values, and numerical ranges. An LLM tags spans in each claim corresponding to attributes, values, and numbers using XML-style markers encoding the associated attribute name. This tagging procedure is shared across both in-lexicon and out-of-lexicon substitution strategies.

In-lexicon substitution replaces one or more tagged arguments with an alternative from the same dataset lexicon. We apply two forms: categorical value substitution, in which a categorical value is replaced with a different value from the same attribute; and numerical substitution, in which a numerical value is replaced with a different value within the observed range of the attribute. We do not substitute attribute names, as doing so can unintentionally produce true claims when attributes share similar distributions. Only mutated claims exhibiting high NLI contradiction scores (> 0.9) are retained and labeled FALSE.

Out-of-Lexicon Argument Substitution. This strategy generates NEI claims by replacing tagged arguments with values or attributes absent from the underlying dataset, introducing references that cannot be verified or refuted. We apply two types: attribute substitution, replacing an attribute name with one not observed in the dataset; and categorical value substitution, replacing a value with one not observed. We do not substitute numerical values, as this can only produce false claims.

NLI-based validation ensures that the resulting claim is neither supported nor contradicted by the original source text. We compute entailment and contradiction scores and retain only claims where both fall below a threshold of 0.2, removing cases where the substitution introduces an unintended true or false statement.

## A.5 Human Validation Details

A total of 13 annotators with expertise in data visualization participated in the validation study, producing 475 annotations across a sample of 294 claims, with up to three independent annotations per claim. Annotators performed the same task defined in §3.1: interacting with the visualizations as needed and assigning one of the three labels based solely on the available visual evidence. They were primed with example tasks and given feedback before starting annotation on real samples. Annotators were blind to the original dataset labels and to the construction process.

Raw agreement between human annotators and the original labels is 81.3%. We additionally applied a Bayesian Dawid–Skene model (Dawid and Skene, 1979; Paun et al., 2018) to the annotation data, inferring latent labels solely from annotator responses. The inferred class prevalence closely matches the dataset distribution: estimated proportions for TRUE and FALSE claims differ by less than 3 percentage points from the ground truth, while NEI prevalence is recovered within the model’s uncertainty bounds. The posterior mean accuracy of inferred labels relative to the dataset labels is 66.2% (95% credible interval: [59.5%, 71.8%]), with an MAP estimate of 77.6%.

## B Dataset Construction Method Example

This appendix illustrates the complete pipeline for constructing claims from human-authored analytical narratives, using a real example from the corpus.

## B.1 Original Text

"...Firstly, the number of fatal traffic accidents per month has been decreasedfrom 2005 to 2015. From the last bar chart, In 2005 to 2007, there were roughly about 350 records monthly. The number drops gradually, and startingfrom 2010, there were about 200 records in each month. This decrease in records may be a result of advancements in protectivefeatures ofcars, legislation against mobile phone use while driving or other traffic safety related legislations..."

## B.2 Span Extraction and Semantic Labeling

Candidate spans are extracted from the narrative and labeled according to their semantic level (Level 2: statistical/relational, Level 3: perceptual/cognitive):

Label 3: “Firstly, the number of fatal traffic accidents per month has been decreased from 2005 to 2015.”

Label 2: “In 2005 to 2007, there were roughly about 350 records monthly.”

Label 3: “The number drops gradually.”

Label 2: “and starting from 2010, there were about 200 records in each month.”

## B.3 Decomposition and Decontextualization

Each span is decomposed into atomic, verifiable propositions with all implicit references resolved:

Span: “Firstly, the number of fatal traffic accidents per month has been decreased from 2005 to 2015.” Proposition: “The number of fatal traffic accidents per month has been decreased from 2005 to 2015.”

Span: “In 2005 to 2007, there were roughly about 350 records monthly.”

Proposition: “In 2005 to 2007, there were roughly about 350 fatal traffic accidents monthly [as shown in the last bar chart].”

Span: “The number drops gradually.”

Proposition: “The number of [fatal traffic accidents per month] drops gradually [from 2005 to 2015].”

Span: “and starting from 2010, there were about 200 records in each month.”

Proposition: “Starting from 2010, there were about 200 [fatal traffic accident] records in each month.”

## B.4 Claim Mutation: Contrary Swap

Antonym substitution generates false claims by reversing directional language. NLI contradiction scores validate semantic opposition:

Input: The number of fatal traffic accidents per month has been decreased from 2005 to 2015.

Mutated (False): The number of fatal traffic accidents per month has been increased from 2005 to 2015. (contradiction score: 0.998)

Input: The number of fatal traffic accidents per month drops gradually from 2005 to 2015.

Mutated (False): The number of fatal traffic accidents per month rises gradually from 2005 to 2015. (contradiction score: 0.999)

## B.5 Claim Mutation: In-Lexicon Argument Swap

Arguments (dates, values) are replaced with alternatives from the dataset to generate false claims:

![](images/eebee5b96c7ae7fbd4b0cc1400185c28f93df23a58441e2955efe28bdf95b996.jpg)  
Figure 2: Worked example of the dataset construction pipeline. Starting from human-authored analytical text about traffic accidents (A), we extract semantically labeled spans (B), decompose them into atomic propositions with resolved references (C), and generate False claims through antonym substitution (D1) and in-lexicon argument replacement (D2), as well as NEI claims through out-of-lexicon substitution (D3). NLI scores validate that mutations produce the intended semantic relationships.

Mutated (False): In 2008 to 2007, there were roughly about 350 fatal traffic accidents monthly as shown in the last bar chart. (Contradiction score: 0.984)

Mutated (False): In 2005 to 2007, there were roughly about 315 fatal traffic accidents monthly as shown in the last bar chart. (Contradiction score: 0.996)

Mutated (False): Starting from 2010, there were about 50 fatal traffic accident records in each month. (Contradiction score: 0.068 - rejected)

## B.6 Claim Mutation: Out-of-Lexicon Argument Swap

Arguments not present in the dataset are substituted to generate NEI (Not Enough Information) claims:

Mutated (NEI): The number of road incidents per month drops gradually from 2005 to 2015. (Entailment score: 0.885 - rejected)

Mutated (NEI): Starting from 2010, there were about 200 fatal traffic accident records in each week. (Entailment score: 0.010, Contradiction score: 0.916 - rejected)

Note: Bold text indicates substituted elements. NLI scores validate that mutations produce the intended semantic relationship (high contradiction forfalse claims, low entailment/contradictionfor NEI claims).

## C Dataset Statistics

This appendix provides additional distributional detail complementing the dataset analysis in §3.2.

Mark types. The mark-type distribution (Figure 3, top right) illustrates that InSight is not dominated by a single visualization form. While bars are the most common mark, a substantial fraction of notebooks include lines, geoshapes, circles, text marks, and layered combinations thereof. This diversity is a natural consequence of using Vega-Lite specifications authored by analysts rather than enforcing a fixed chart taxonomy. Importantly, many claims require reasoning across multiple mark types within the same environment (e.g., relating a geographic map to a linked bar chart).

Specification and interaction density. The histograms of visualization specifications per file and interaction types per file (Figure 3, bottom row) show that most notebooks contain multiple specifications and multiple distinct interaction mechanisms. The long right tails indicate the presence of highly complex environments with many views and interaction affordances. This reinforces that In-Sight environments are not single-chart snapshots but multi-view analytical spaces in which part of the verification task is identifying where relevant evidence resides.

![](images/f582414872d0357bfe226613a0a135eaf33958f283007fefabd1f586fbc2e4a7.jpg)

![](images/53b15b334ba343e045f82766d4fd734df69ab37808e327954ce1ed3b0d07d03e.jpg)

![](images/c52df59d845258d8ed839d2d5ab08c4fbddd8cf3611ca2401585898918fedd31.jpg)

![](images/d9f6f6b09704315c7b206828f1d87647c3116c1265cd38858b0869c9d11f3a6b.jpg)  
Figure 3: Dataset composition. Top left: distribution of interaction primitives across notebooks. Top right: distribution of Vega-Lite mark types. Bottom left: number of visualization specifications per notebook. Bottom right: number of distinct interaction types per notebook.

## D Model Evaluation Examples

![](images/10751c4ca4333bd2ddfde3c10fd302dce611036d9a1faf57911c0cda6c896a06.jpg)  
Figure 4: Interaction trace for GPT 5.5 correctly verifying a true claim

Gemma\_4\_31B  
![](images/87071b61884346061705eaa2fa49905e407b8bce6c3a609b67349be9f0008aa0.jpg)  
Figure 5: Interaction trace for Gemma 4 31B Flash verifying a true claim, incorrectly labelling the claim as NEI.

## Gemini\_3.5-flash

In Wales in 2021 for ages 25-35, the proportion of women smokers has either surpassed or matched the number of male smokers.

![](images/4246dba0d65171f42a238a589d22cc07b27494bb1f0d2f86bb023c319cbb7b28.jpg)  
click (135, 427)

![](images/e7055c8049b2dc1805ed28d34c230861ba5b721a503fabd554e15bf8f034b54d.jpg)  
click (646, 512)

![](images/0522d9e715a1679f4c9ce833dda68c0f2a161f5f622ed1a2440900d7576e1e5b.jpg)  
click (245, 293)  
click (85, 427)

![](images/7752ee164721561afd0fc75fc7090867d01fc322c89364358aced3e6a9d8157b.jpg)  
click (109, 293)

![](images/f54b874f00f153b7f2c65f9796db2ccf811e8994830f5adbbb39b0ebe53ef966.jpg)

![](images/e848cfa24958dd59a17c7df1ab3746dd48bd4e40b2dc9befb98c542c176590dc.jpg)

![](images/a812fc2050321a3808297cd8bab8f26ec2a3e9f1104c654769d4c3bf169e6e38.jpg)  
hover (754, 112)

![](images/474890a9bf52e3cb09d2b01a92632641b16fb76fd93ffb5ef4ad3e5a1bc65243.jpg)

![](images/a76f552c708f9e75f0f30e99e81438624d99d63d02d8352da597e3896281c299.jpg)  
Model answer: Not Enough Information Ground truth: True

![](images/60556a5b17519b903e22686ffa95207e822b13e618204e77dafcab80b3b10834.jpg)  
Figure 6: Interaction trace for Gemini 3.5 Flash verifying a true claim, incorrectly labelling the claim as NEI.  
GPT\_5.5  
In Wales in 2021 for ages 25-35, the proportion of women smokers has either surpassed or matched the number of male smokers.

![](images/710d65e61ebc350cb36faaac63480d80601fcc2fe3a0574f8ada6d19a21eb680.jpg)  
Model answer: True Ground truth: True

Figure 7: Interaction trace for GPT 5.5 correctly verifying a true claim.

Model answer: False Ground truth: NotEnoughInfo

## Gemma\_4\_31B

The pay gap between White employees and employees of indigenous people was 7.7% in 2015.

Initial state

![](images/ecd24918a749f3e1149451ef902209193b48c61c0d60bc4f52e19d45668768a5.jpg)  
hover (315, 300)

![](images/d8497237a2a16be0beb57d7dbc245d148e5509b4a5130af6246c1a1d8fa6a5f7.jpg)

![](images/8c39e8b4b201fa6a44ae265c7f6c06e50e30347ff2bb5c4cce5d8b671c6f0b75.jpg)  
× WRONG IES:0.000 SCR: 1.000

Figure 8: Interaction trace for Gemma 4 31B verifying an NEI claim, incorrectly labelling the claim as False.

## Gemini\_3.5-flash

The pay gap between White employees and employees of indigenous people was 7.7% in 2015.

Initial state

![](images/5b01e242b110ad407d3d05ecbba0f85e5c988da3c1dc15eec97687bd99e34ad6.jpg)  
hover (275, 450)

![](images/7dc0d7c28489ffe1930a57c4bc3f0294f2a7bb5fd79bbd5784d79698b5aa95e5.jpg)  
hover (270, 350)

![](images/7ef4b0de6f40e7abfe9359386f2dc1825f51e244f49752addf835b474c868f05.jpg)

![](images/9954d34ddb5103cb642afbf9b983566ab9234ee71606e85ac5295015bc02525e.jpg)

![](images/dab9af9aa462fc314d7e9e9962bbc8cd870dba084995ba85e811c96721533b38.jpg)

![](images/6ce983b35fb02dae595199ce7f532a9e1b5369b83199726067421562e12b2083.jpg)  
Model answer: False Ground truth: NotEnoughInfo  
× WRONG IES: 0.000 SCR: 1.000

Figure 9: Interaction trace for Gemini 3.5 Flash verifying an NEI claim, incorrectly labelling the claim as False.