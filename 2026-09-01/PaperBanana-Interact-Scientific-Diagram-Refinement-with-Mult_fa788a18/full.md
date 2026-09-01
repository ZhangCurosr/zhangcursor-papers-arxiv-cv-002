# PaperBanana-Interact: Scientific Diagram Refinement with Multi-Turn Human Feedback

Xueqing Wu<sup>1</sup> <sup>2</sup> <sup>\*</sup>, Ashwin Balasubramanian<sup>2</sup>, Bingxuan Li<sup>2</sup> <sup>3</sup>, Dawei Zhu<sup>4</sup>, Kai-Wei Chang<sup>1</sup>, Yale Song<sup>2</sup>, Yiwen Song<sup>2</sup>, Rui Meng<sup>2</sup>, Tomas Pfister<sup>2</sup> and Nanyun Peng<sup>2</sup>

<sup>1</sup>University of California, Los Angeles, <sup>2</sup>Google, <sup>3</sup>University of Illinois Urbana-Champaign, <sup>4</sup>Peking University https://shirley-wu.github.io/PaperBanana-Interact/

Recent eforts have aimed to automate scientific diagram generation from paper content (Lin et al., 2026; Zhu et al., 2026a). However, fully satisfying an author’s visual and communicative preferences in a single turn is challenging: in our formative user study (� = 14), all participants requested further revisions after viewing an initial draft, and 86% of them rated the refined diagrams as more satisfactory. Despite the clear demand, the multi-turn workflow remains largely underexplored. To bridge this gap, we present MTPaperBananaBench, a benchmark for multi-turn diagram generation containing 292 images annotated with 3,518 user requirements. To reduce expensive human studies and enable scalable benchmarking, we construct a user simulator that, at each turn, identifies unsatisfied requirements and converts � of them into natural language feedback. Evaluating both requirement satisfaction and overall diagram quality reveals two key failure modes shared across baseline multiturn systems: (1) quality drift, where diagram quality progressively declines over turns, and (2) forgetting, where previously implemented features are lost in subsequent turns. To address these issues, we introduce PaperBanana-Interact, a multi-agent system that refines diagrams via an internal critique-and-refine loop. PaperBanana-Interact consistently improves rather than degrades diagram quality across turns, outperforming baselines by 11.9–18.6 points in quality score and reducing forgetting by 3.7–6.2 points.

## 1. Introduction

Visual diagrams are essential for communicating complex scientific ideas. Motivated by this, recent work explores using multimodal generative models to generating diagrams directly from paper content (Zhao et al., 2026; Zhu et al., 2026a,b). However, satisfying an author’s nuanced visual and communicative preferences in a single pass remains challenging, necessitating multi-turn refinement. Furthermore, figure design is naturally iterative: reviewing an initial draft often reveals refinement opportunities that are dificult to anticipate upfront. Our formative study with 14 participants from diverse backgrounds confirms the strong need for a multi-turn diagram generation workflow: all participants requested further revisions after reviewing the initial draft, and 86% reported higher satisfaction after iterative refinement. These findings highlight multi-turn interaction as a critical yet largely unexplored direction for diagram generation.

To systematically investigate this challenge, we introduce MTPaperBananaBench, a benchmark for multi-turn diagram refinement with iterative user feedback. At the core of the benchmark is the interaction process in which user preferences gradually emerge as the diagram evolves. However, collecting such human interactions at scale is prohibitively expensive and dificult to standardize across evaluation runs. To enable scalable and reproducible benchmarking, we develop a user simulator that emulates this iterative feedback. The simulator is driven by a set of user preferences hidden from the diagram generator at the initial turn. At each subsequent turn, the simulator identifies unfulfilled preferences and formulates � of them into natural language feedback. To support this simulator, we annotate a diverse pool of 3,518 user requirements derived from 292 diagrams drawn by human experts, spanning content, layout, and visual representation. During evaluation, the generation

system interacts with the simulated user for a fixed interaction budget. The final output diagram is then evaluated across two dimensions: (1) satisfaction of the user requirements, and (2) overall diagram quality.

Using MTPaperBananaBench, we conduct a comprehensive evaluation with various baselines for diagram refinement, including frontier generative models such as NanoBananaPro (Google DeepMind, 2025) and GPT-Image-2 (OpenAI, 2026), as well as an agentic baseline, PaperBanana-DirectRefine. While these systems effectively address refinement requests on individual turns, achieving singleturn satisfaction rates of 68.6% for PaperBanana-DirectRefine and 56.2% for NanoBananaPro, their performance degrades significantly across multiturn interactions. Extensive failure analysis reveals two key failure modes: (1) quality drift, where image quality progressively deteriorates over turns (50.3→19.0 for NanoBananaPro,

![](images/40906e459e96c9f00856ce03fd645a342ed86030a001cfa1380172010b7e831a.jpg)  
Figure 1 | Diagram generators may fail to fully satisfy user preferences in a single turn, necessitating multi-turn refinement with human feedback. We introduce MTPaperBananaBench to benchmark this task, and propose PaperBanana-Interact to mitigate common failure modes in baseline systems. [Figure generated by PaperBanana-Interact]

50.3→47.1 for PaperBanana-DirectRefine);

and (2) forgetting, where later refinements overwrite previously satisfied requests, afecting 14.1– 22.9% of such requests across all baselines. These findings highlight the importance of developing diagram generation systems that can efectively interpret evolving user feedback and support iterative refinement across multiple turns.

To address these problems, we propose PaperBanana-Interact, a multi-agent system for diagram refinement. PaperBanana-Interact refines the diagram via an internal loop, where a critic reviews each revision and proposes suggestions, and a refiner and a visualizer generate the next revision accordingly. Importantly, PaperBanana-Interact features a multi-objective critic that assesses the diagram against multiple constraints, including the current user request, all prior user requests, adherence to the source context, and presentation quality. To handle long, multi-image interaction histories eficiently, we also introduce a summarizer that compresses the full history into a compact textual memory capturing relevant visual details, which is then shared across all subsequent agents. Experimental results show that PaperBanana-Interact mitigates quality drift and forgetting, improving quality scores by 11.9–18.6 percent over baselines while reducing the forgetting rate by 3.7–6.2 percent.

In summary, our contributions are threefold, summarized in Figure 1. (1) We introduce MTPaperBananaBench, the first multi-turn diagram generation benchmark with 292 samples and 3,518 user requirement annotations. (2) We systematically benchmark generative model baselines and agentic baselines, uncovering shared challenges of quality drift and forgetting for multiturn scientifc diagram generation. (3) We propose PaperBanana-Interact, a multi-agent refinement system that boosts diagram quality by 11.9–18.6 percent while lowering the forgetting rate by 3.7–6.2 percent.

## 2. Related Work

Generative models for image generation and refinement. Generative models have achieved strong capabilities in text-guided image generation (Cao et al., 2025; Google DeepMind, 2025; Meta, 2026; OpenAI, 2026). Beyond one-shot generation, these models increasingly support image-conditioned and instruction-guided refinement, enabling users to modify visual content while preserving desired elements (Brooks et al., 2023; Hertz et al., 2022; Kawar et al., 2023). Recent work further explores multi-turn interaction, where users iteratively provide feedback and corrections (Hahn et al., 2025; Nabati et al., 2024). However, such interaction remains underexplored for scientific diagrams, which impose stricter requirements on structure, semantics, and accuracy.

Scientific diagram generation. Generating scientific diagrams is challenging because it demands rigorous factual accuracy and the flexibility to capture nuanced author insights. Early methods typically rendered the diagrams by generating code, such as TikZ (Belouadi et al., 2024a,b, 2025; Zhang et al., 2025) or PythonPPTX (Pang et al., 2026; Zheng et al., 2025). With the growing capabilities of generative models, recent research has increasingly integrated them into the pipeline. For example, Zhu et al. (2026b) use code to draft an initial diagram before refining it with generative models. Other recent work directly generates the final image with generative models, and relies on large language models (LLMs) solely to optimize the text prompts (Zhao et al., 2026; Zhu et al., 2026a). Despite these advancements, existing frameworks lack support for multi-turn, iterative human feedback, a critical gap that we aim to address.

## 3. Task and Dataset

## 3.1. Task Formulation

In the standard single-turn setting, scientific diagram generation aims to produce an illustrative diagram, denoted as $I _ { 0 } \mathrm { { : } }$ , from a source context $x _ { 0 } ,$ , which typically includes the paper content and a communicative intent that specifies the diagram’s scope and focus. The multi-turn setting extends this task by allowing the initial diagram draft $I _ { 0 }$ to be refined iteratively. At each subsequent turn $t \geq 1$ the user provides natural language feedback $x _ { t }$ requesting further edits. The goal is then to generate an updated diagram $I _ { t }$ conditioned on the interaction history $\left[ \mathbf { I } _ { < t } , \mathbf { x } _ { \le t } \right]$ , where $\mathbf { x } _ { \le t } = \left[ x _ { 0 } , \ldots , x _ { t } \right]$ and $\mathbf { I } _ { < t } = \left[ I _ { 0 } , \ldots , I _ { t - 1 } \right]$ denote the sequence of user inputs and previously generated diagrams, respectively.

## 3.2. Formative User Study

To validate the practical need for the formulated task, we conducted a formative user study with 14 participants. All participants had substantial experience in academic writing and scientific illustration, with details reported in Appendix C.

Participants were asked to create a diagram for their own research. They first reviewed an initial draft generated by PaperBanana and then interacted with the baseline system PaperBanana-DirectRefine (described in §4.2) as needed to iteratively refine the diagram. Afterward, participants retrospectively rated their satisfaction with both the initial and final diagrams on a 1–5 scale. As shown in Figure 2, multi-turn refinement increased the average satisfaction score from 3.2 to 4.2. Overall, 12 of 14 participants (86%) assigned a higher rating to the diagram after refinement, with improvements ranging from 0.5 to 2 points. These results indicate a strong user need for multi-turn interactions when authoring scientific diagrams.

We further conducted follow-up interviews to analyze the interaction sessions and collect qualitative feedback. Notably, nine participants reported that the initial draft shaped or changed their

vision of how the diagram should be presented. Consequently, many refinement requests emerged only after participants viewed an initial or intermediate draft, a pattern observed in 64% of the sessions. This further highlights that multi-turn interaction closely reflects real-world authoring workflows. The interviews also surfaced recurring failure modes of the PaperBanana-DirectRefine system, including reduced factual faithfulness (29%), degraded aesthetic quality (43%), and failure to preserve previously satisfied requests (29%). These

![](images/949a0a5d008bf106992ae718b883d00bec125e79c9e58718de97d83f2f1222f9.jpg)  
Figure 2 | Multi-turn refinement improves user satisfaction. Per-participant ratings (1–5 scale) for the diagrams before and after multi-turn refinement, with score changes (Δ) annotated above each bar.

qualitative observations closely align with our quantitative evaluation on MTPaperBananaBench, further motivating the design of PaperBanana-Interact to address these limitations.

## 3.3. MTPaperBananaBench

We curate MTPaperBananaBench to provide a controlled benchmark for evaluating multi-turn diagram generation and refinement. To simulate iterative human-AI interaction, a user simulator operates over a list of pre-annotated requirements �, identifying the first � failed requirements and translating them into natural language feedback. Overall, the benchmark comprises 292 diagrams with 3,518 annotated requirements. We provide details of the user simulator, the dataset construction process, and our evaluation protocol in this section.

Dataset construction. For each instance in the PaperBananaBench (Zhu et al., 2026a) dataset, we annotate a requirement list � through a two-stage process: LLM-assisted candidate generation followed by manual review by human annotators, as illustrated at the top of Figure 3. At both stages, the requirement list � is grounded on both its source context $x _ { 0 }$ and the reference diagram $I _ { H }$ drawn by the original authors. As $I _ { H }$ reflects expert knowledge of the source content and its intended communication goal, we extract its salient design choices as requirement annotations rather than annotating them from scratch. To ensure broad coverage, we establish a comprehensive taxonomy (detailed in Appendix B) that categorizes requirements by category (content, organization, and visual representation) and violation severity. Both the LLM and human annotators are then explicitly instructed to annotate requirements spanning a diverse range of categories and severity levels.

The concrete annotation process is as follows, with further details provided in Appendix B. First, we prompt Gemini-3.1-Pro (Google DeepMind, 2026) to generate candidate requirements and label each according to the taxonomy. Then, three human annotators independently review the candidates, removing or revising unfaithful requirements and correcting inaccurate category labels. This manual review removes 0.5% and revises 24.7% of the candidates. The final MTPaperBananaBench contains 3,518 requirements across 292 diagrams, averaging 12 requirements per diagram, with the detailed distribution shown in Figure 8. To assess annotation quality, we measure inter-annotator agreement on a subset of 207 requirements, yielding a Cohen’s � of 0.767, which indicates substantial agreement. Finally, since the user simulator reveals requirements � in a predefined order, we randomly permute each requirement list and fix the resulting order.

User simulator. The user simulator operates over a list of � requirements, $R = [ r _ { 1 } , \ldots , r _ { N } ]$ , where each requirement represents a plausible visualization preference, such as “the bar chart must use distinct fill colors to visually distinguish between the two subject types” or “the Q, K, and V token sequences must be stacked vertically”. These requirements are initially hidden from the diagram generation system and are progressively revealed by the simulator, thereby simulating how a user’s request evolves over multiple interaction turns.

![](images/8c94f8cfd8fc85c8411f3dcf991a455916e25bb9671ca678449cf167559ad92f.jpg)  
Figure 3 | Annotation and inference pipeline of MTPaperBananaBench. The multi-turn interaction is driven by a user simulator; at each turn, the user simulator evaluates the diagram against all requirements, extracts � failed ones, and formulates them as natural language feedback. [Figure generated by PaperBanana-Interact]

Concretely, at turn �, an LLM-as-judge evaluator E (detailed at Appendix F) determines whether each requirement $r _ { i } \in R$ is satisfied by the latest diagram $I _ { t - 1 }$ , yielding a set of failed requirements. When the set is non-empty, we select the first � failed requirements following the predefined order of �, where $k \geq 1$ is a configurable hyperparameter. The user simulator then prompts a vision-language model to generate an utterance $x _ { t }$ targeting these failed requirements. To simplify the simulation, we additionally provide the user simulator with the evaluator rationales that detail why $I _ { t - 1 }$ fails each selected requirement. The interaction terminates when $F _ { t - 1 } = \emptyset$ or a turn budget $T _ { m a x }$ is reached.

Evaluation. We evaluate the final diagram along two main dimensions: requirement satisfaction and diagram quality. Concretely, we reuse the evaluator E to compute the requirement satisfaction rate by averaging the binary satisfaction scores across all requirements. To evaluate diagram quality, we adopt the metric introduced in PaperBananaBench (Zhu et al., 2026a), which compares the generated diagram against a human-created reference diagram across multiple dimensions and reports the resulting win rate. To characterize multi-turn behavior and measure common failure patterns, we report two additional diagnostic metrics: (1) per-turn requirement satisfaction isolates the system’s single-turn instruction-following capability by measuring the fraction of newly raised requirements that are satisfied within their introduction turn; (2) forgetting rate measures the fraction of requirements that are satisfied when introduced but become unsatisfied at any subsequent turn. Further details on all metrics are provided in Appendix F.

## 4. Methods

## 4.1. Generative Models as Refiners

Generative models are often inherently capable of image refinement (Brooks et al., 2023; Google DeepMind, 2025; Kawar et al., 2023; OpenAI, 2026), making them a natural baseline for refining

![](images/2b7a86b435724e2e00492920d4edf769f1327c295dfdc5d6d61511dacc28e211.jpg)  
Figure 4 | PaperBanana-Interact overview. PaperBanana-Interact first employs a summarizer to compress interaction history into a compact memory, followed by a refinement loop where a multiobjective critic evaluates the diagram against diferent requirements to guide subsequent updates and visualization. [Figure generated by PaperBanana-Interact]

diagrams. Denoting the generative model as $\boldsymbol { \mathcal { V } } ,$ , we prompt the model to refine the latest image $I _ { t - 1 }$ conditioned on all user utterances up to turn �:

$$
I _ { t } = \mathcal { V } ( I _ { t - 1 } , { \bf x } _ { \leq t } ) .
$$

## 4.2. PaperBanana-DirectRefine

Recent agentic diagram-generation systems (Zhao et al., 2026; Zhu et al., 2026a) typically employ a pipeline, denoted by ${ \mathcal { A } } ,$ , to generate and refine a detailed image-generation prompt. A generative model then serves as a visualizer $\boldsymbol { \mathcal { V } } ,$ producing the final image from the prompt �. Under this formulation, the initial turn is given by:

$$
P _ { 0 } = \mathcal { A } ( x _ { 0 } ) , I _ { 0 } = \mathcal { V } ( P _ { 0 } ) .
$$

Extending this design to a multi-turn setting, the image $I _ { t }$ at turn � is generated by first using a system $\mathcal { A } _ { r }$ to refine the previous prompt $P _ { t - 1 }$ into $P _ { t }$ , and subsequently rendering the image via $\mathbf { \mathcal { V } } \mathbf { : }$

$$
P _ { t } = \mathcal { A } _ { r } ( P _ { t - 1 } , \mathbf { I } _ { < t } , \mathbf { x } _ { \le t } ) , \quad I _ { t } = \mathcal { V } ( P _ { t } ) .
$$

The efectiveness of this framework thus depends critically on the design of ${ \mathcal { A } } _ { r }$ . As a simple baseline, we instantiate $\mathcal { A } _ { r }$ via direct prompting of a vision-language model, denoted by $M _ { \mathrm { v l m } }$ . To control computational cost and prevent long-context degradation, we provide the model with the full sequence of user utterances, while limiting the image and prompt inputs to only the most recent turn:

$$
P _ { t } = \mathcal { M } _ { \mathrm { v l m } } \big ( I _ { t - 1 } , P _ { t - 1 } , \mathbf { x } _ { \le t } \big ) .
$$

## 4.3. PaperBanana-Interact

In this section, we introduce PaperBanana-Interact, a multi-agent system that builds on the agentic design in §4.2. As shown in Figure 4, PaperBanana-Interact uses a summarizer to compress multiimage histories into a compact textual memory, followed by a critic-refiner loop to iteratively improve both the text prompt and the visualized image. The full prompts are provided in Appendix H.

Summarizer S. Requiring each agent to independently process the multi-image interaction history is computationally ineficient and risks distracting it from its core task. Furthermore, successive image revisions exhibit significant visual redundancy and can thus be substantially compressed. We therefore introduce a summarizer S that distills the interaction history into a textual memory $M _ { t } = S ( \mathbf { I } _ { < t } , \mathbf { X } _ { \leq t } )$ . �<sub>�</sub> is shared across all agents, providing them with critical context without requiring repeated processing of the raw images. Concretely, this memory summarizes key visual details, including what each diagram depicts, how it changed relative to the previous revision, and whether those modifications align with the user’s request at that turn.

Refinement loop. At turn �, PaperBanana-Interact generates �<sub>�</sub> through an internal agent loop starting from the previous turn’s image and prompt, i.e. $\mathbf { \Phi } _ { I _ { t } } ^ { ( 0 ) } = I _ { t - 1 } , \mathbf { \Phi } _ { P _ { t } } ^ { ( 0 ) } = P _ { t - 1 }$ . At iteration �, a critic C evaluates the current version $I _ { t } ^ { ( \tau - 1 ) }$ , identifies remaining defects, and yields a critique:

$$
c _ { t } ^ { ( \tau ) } = C \left( I _ { t } ^ { ( \tau - 1 ) } , \mathbf { x } _ { \leq t } , M _ { t } \right) .
$$

If C determines that further revision is needed, a refiner R incorporates $c _ { t } ^ { ( \tau ) }$ to update the prompt:

$$
\begin{array} { r } { \boldsymbol { P } _ { t } ^ { ( \tau ) } = \mathcal { R } \left( \boldsymbol { I } _ { t } ^ { ( \tau - 1 ) } , \boldsymbol { P } _ { t } ^ { ( \tau - 1 ) } , c _ { t } ^ { ( \tau ) } , \mathbf { x } _ { \le t } , M _ { t } \right) . } \end{array}
$$

The visualizer V then renders the updated prompt to produce the next image revision, $I _ { t } ^ { ( \tau ) } = \mathcal { V } \left( P _ { t } ^ { ( \tau ) } \right)$ which is fed back to C. The loop terminates when the critic determines no further edits are required, or when the iteration budget $\tau _ { \mathrm { m a x } }$ is reached.

Multi-objective critic. PaperBanana-Interact employs a multi-objective critic that systematically evaluates the generated image against all relevant constraints: the current and prior user requests $x _ { 1 } , \ldots , x _ { t }$ , faithfulness to the source context �<sub>0</sub>, and overall presentation quality. Concretely, the critic is instructed to produce a structured JSON with an explicit critique entry for each constraint, thereby guiding the system toward satisfying multi-turn constraints and mitigating quality drift and forgetting issues.

## 5. Experiments

Evaluated systems. To perform multi-turn diagram generation, we use existing single-turn generators to produce initial diagrams and diferent refiners to perform multi-turn iterations. To remain consistent with prior work (Zhu et al., 2026a) and ensure a fair comparison, all agentic systems adopt Gemini-3.1-Pro (Google DeepMind, 2026) and NanoBananaPro (Google DeepMind, 2025) as the language model and image generation backbones, respectively.

Single-turn generators: We consider generative models, including NanoBananaPro (Google DeepMind, 2025) and GPT-Image-2 (OpenAI, 2026), as well as agentic systems, including AutoFigure (Zhu et al., 2026b), AutoFigure-Edit (Lin et al., 2026), PaperBanana (Zhu et al., 2026a), and Crafter (Zhao et al., 2026).

Refiners: For generative models used as single-turn generators, we use the same model as the refiner. For agentic single-turn generators such as PaperBanana and Crafter, we compare PaperBanana-Interact against baseline refiners, including NanoBananaPro and PaperBanana-DirectRefine. For PaperBanana-Interact, we set the iteration budget for its internal loop to $\tau _ { m a x } = 1 0$ . We exclude AutoFigure and AutoFigure-Edit from the multi-turn evaluation due to their low performance.

User simulation setup. We implement the user simulator described in §3.3 using Gemini-3.1-Pro (Google DeepMind, 2026) as the backbone model, with full prompts provided in Appendix G. We perform up to $T _ { m a x } = 5$ refinement turns with user feedback while allowing early stopping when all user requirements are satisfied. We evaluate two simulation settings with diferent numbers of requirements raised per turn (�=1 and �=3), reported in Tables 1 and 2, respectively.

<table><tr><td>Single-Turn Generator</td><td>Multi-Turn Refiner</td><td>Qual↑</td><td>Req↑</td><td>Reqpt↑</td><td>Forget↓</td></tr><tr><td>AutoFigure</td><td>–</td><td>1.9</td><td>20.4</td><td>一</td><td></td></tr><tr><td>AutoFigure-Edit</td><td></td><td>0.3</td><td>16.5</td><td></td><td></td></tr><tr><td>NanoBananaPro</td><td>NanoBananaPro</td><td>22.4 12.5</td><td>18.1 39.4</td><td>53.8</td><td>20.9</td></tr><tr><td>GPT-Image-2</td><td></td><td>41.6</td><td>25.0</td><td></td><td></td></tr><tr><td></td><td>GPT-Image-2</td><td>25.7 35.4</td><td>52.9 18.4</td><td>67.4</td><td>15.5</td></tr><tr><td rowspan="4">Crafter</td><td>NanoBananaPro</td><td>6.2</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>39.4</td><td>56.2</td><td>19.2</td></tr><tr><td>PaperBanana-DirectRefine</td><td>30.5</td><td>47.6</td><td>68.1</td><td>15.0</td></tr><tr><td>PaperBanana-Interact</td><td>48.1</td><td>52.1</td><td>73.4</td><td>10.3</td></tr><tr><td rowspan="4">PaperBanana</td><td></td><td>50.3</td><td>25.2</td><td></td><td></td></tr><tr><td>NanoBananaPro</td><td>19.0</td><td>45.2</td><td>56.2</td><td>19.0</td></tr><tr><td>PaperBanana-DirectRefine</td><td>47.1</td><td>52.6</td><td>68.6</td><td>18.8</td></tr><tr><td>PaperBanana-Interact</td><td>61.2</td><td>58.0</td><td>77.2</td><td>12.6</td></tr></table>

Table 1 | Main results with the � =1 user simulator. We report quality (Qual↑) and requirement satisfaction (Req↑) as main metrics, and per-turn requirement satisfaction $( \mathrm { R e q ^ { p t } \uparrow } )$ and forgetting rate (Forget↓) as diagnostic metrics. For each single-turn generator, we report its single-turn performance (- as Multi-Turn Refiner) and its multi-turn performance when paired with diferent refiners.

Evaluation setup. As discussed in §3.3, we report two main metrics, requirement satisfaction and diagram quality, both using Gemini-3.1-Pro as the judge. For diagram quality, we adopt the metric from Zhu et al. (2026a) measuring four dimensions: faithfulness, conciseness, readability, and aesthetics. For requirement satisfaction, we prompt the VLM to view the generated diagram alongside its associated requirements and output a binary score and rationale for each requirement. To validate the reliability of this evaluation, a human evaluator independently judged the satisfaction of 200 requirements, achieving a Cohen’s � of 0.812 with the VLM judge, indicating near-perfect agreement. For diagnostic purposes, we also report the per-turn requirement satisfaction and forgetting rate. Full prompts and implementation details are in Appendix F.

## 5.1. Experimental Results

Main results. We report evaluation results for the �=1 and �=3 user simulators in Tables 1 and 2, respectively. As users reveal hidden requirements across turns, multi-turn interaction consistently improves requirement satisfaction (Req) compared to the single-turn setting. All refiners are reasonably capable of satisfying a user request in the turn it is raised, with per-turn requirement satisfaction rates (Req<sup>pt</sup>) ranging from 50 to 80 across all systems.

Beyond this overall trend, performance varies substantially across systems. Among single-turn generators, PaperBanana performs best overall, followed by GPT-Image-2. For multi-turn refiners, generative models such as NanoBananaPro perform poorly and consistently degrade diagram quality. Agentic systems, by contrast, achieve notably better results: PaperBanana-DirectRefine significantly improves both diagram quality and requirement satisfaction, while PaperBanana-Interact delivers further gains.

Quality drift. All baseline refiners exhibit a performance decline relative to their single-turn outputs,

<table><tr><td>Single-Turn Generator</td><td>Multi-Turn Refiner</td><td>Qual↑</td><td>Req↑</td><td>Reqpt↑</td><td>Forget↓</td></tr><tr><td rowspan="2">NanoBananaPro</td><td></td><td>22.4</td><td>18.1</td><td></td><td></td></tr><tr><td>NanoBananaPro</td><td>8.6</td><td>76.5</td><td>59.6</td><td>21.5</td></tr><tr><td>GPT-Image-2</td><td>GPT-Image-2</td><td>41.6 22.4</td><td>25.0 88.5</td><td>- 71.0</td><td>14.1</td></tr><tr><td rowspan="4">Crafter</td><td></td><td>35.4</td><td>18.4</td><td></td><td></td></tr><tr><td>NanoBananaPro</td><td>2.9</td><td>74.6</td><td>59.1</td><td>21.6</td></tr><tr><td>PaperBanana-DirectRefine</td><td>23.5</td><td>87.8</td><td>72.2</td><td>16.0</td></tr><tr><td>PaperBanana-Interact</td><td>42.1</td><td>93.4</td><td>79.0</td><td>12.3</td></tr><tr><td rowspan="4">PaperBanana</td><td></td><td>50.3</td><td>25.2</td><td></td><td></td></tr><tr><td>NanoBananaPro</td><td>18.3</td><td>76.9</td><td>59.8</td><td>22.9</td></tr><tr><td>PaperBanana-DirectRefine</td><td>42.6</td><td>87.6</td><td>71.0</td><td>19.1</td></tr><tr><td>PaperBanana-Interact</td><td>54.5</td><td>94.2</td><td>79.6</td><td>13.0</td></tr></table>

Table 2 | Main results with the �=3 user simulator, with notation following Table 1.

with generative models experiencing larger drops than PaperBanana-DirectRefine. As shown in Figure 5 (top), the quality score experiences a pronounced drop in early rounds and progressively declines throughout later turns. Comparing Tables 1 and 2 further shows that this drift is amplified in � = 3 settings, where user feedback is longer and contains multiple requirements. In contrast, PaperBanana-Interact consistently mitigates quality drift and enhances quality over single-turn outputs, despite more modest gains under �=3.

To dissect quality drift, we present a breakdown of quality scores across dimensions in Figure 5 (bottom). NanoBananaPro exhibits severe quality drift across all dimensions, with prominent failure modes including sudden drops in aesthetics (Figure 15) and distorted artifacts (Figure 16). While PaperBanana-DirectRefine better preserves overall quality, its quality drift often stems from awkward integration of user

![](images/c77cc78f60836214582b13d8b4c36a8991754759b09d4e2db8085b767851e312.jpg)

![](images/d5dae9bd6b7a81b244d02802c09296c9753552da94b0084d80adbec9e6cc6674.jpg)  
Figure 5 | Diagram quality across refinement turns (top) and final quality breakdown across dimensions (bottom) when refiner systems iterate on PaperBanana outputs for 5 turns with the �=3 user simulator.

requests (Figure 6) and unexpected stylistic drift (Figure 17), mostly degrading conciseness and readability. In contrast, PaperBanana-Interact mitigates quality drift by iteratively refining the visual design and correcting layout or styling issues across iterations, as shown in Figure 6. A pairwise evaluation by a human annotator across 150 samples further validates these gains: PaperBanana-Interact outputs were preferred over NanoBananaPro and PaperBanana-DirectRefine in 81.3% and 76.7% of cases, respectively.

Forgetting. All baseline systems consistently exhibit forgetting, with forgetting rates ranging from 14 to 23% under both � =1 and � =3 settings. Forgetting typically occurs when a model optimizes for the current request but inadvertently alters a feature previously specified by the user, as shown in Figure 7. PaperBanana-Interact addresses this by refining diagrams over multiple internal iterations while actively checking for forgetting at each step. As demonstrated in Table 3, ablated versions with smaller

[User] Reorganize the layout into three stacked horizontal sections going strictly from top to bottom: 'Training Process', then...  
![](images/f32dcdc735b7199bf065f38e89821828d608125385fcf6fedb07e00444014e2d.jpg)  
Figure 6 | Qualitative example of quality drift. PaperBanana-DirectRefine awkwardly integrates the user request, leaving an unbalanced layout and excessive empty space, while PaperBanana-Interact iteratively refines the layout and styling and produces a higher-quality output.

[User at t=1] ...Could you add "U-Net:" to the names so they read "U-Net: Panorama Branch" and "U-Net: Perspective Branch"?..  
![](images/462b071bb34dca9df375fade1f54d107b28d829935538254e08fa80769b4f35b.jpg)  
Figure 7 | Qualitative example of forgetting. PaperBanana-DirectRefine initially implements the requested changes. However, during later refinements that repeatedly update the styling of the relevant blocks, it fails to preserve the changes.

iteration budgets (�<sub>max</sub>) show monotonically higher forgetting rates. However, PaperBanana-Interact does not eliminate forgetting entirely, with a forgetting rate of 10–13%.

PaperBanana-Interact ablations. We investigate three design choices of PaperBanana-Interact: critic design, multi-image history, and iteration budget, with results shown in Table 3. For the critic, replacing our multi-objective critic with a naive, unstructured baseline lowers per-turn satisfaction by 4.9 points and quality by 2.3 points while slightly increasing forgetting, demonstrating the benefit of our proposed design. For multiimage history, omitting it from the input increases forgetting from 12.6 to 14.7 and brings a slight quality drop, confirming that the multi-image history provides

<table><tr><td>Critic</td><td>History</td><td> $\tau _ { m a x }$ </td><td>Qual↑</td><td>Req↑</td><td>Reqpt↑</td><td>Forget↓</td></tr><tr><td colspan="7">Default setup for PaperBanana-Interact</td></tr><tr><td>MO</td><td>Sum</td><td>10</td><td>61.2</td><td>58.0</td><td>77.2</td><td>12.6</td></tr><tr><td>Naive</td><td>Sum</td><td>10</td><td>58.9</td><td>57.1</td><td>72.3</td><td>13.1</td></tr><tr><td>MO MO</td><td>None Full</td><td>10 10</td><td>58.0</td><td>58.1 57.6</td><td>77.1</td><td>14.7</td></tr><tr><td>MO</td><td>Sum</td><td>5</td><td>58.9 60.6</td><td>57.4</td><td>73.5 75.5</td><td>11.6 13.4</td></tr><tr><td>MO</td><td>Sum</td><td>3</td><td>53.3</td><td>55.9</td><td>74.1</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>13.8</td></tr><tr><td>MO</td><td>Sum</td><td>1</td><td>43.5</td><td>54.2</td><td>72.3</td><td>16.8</td></tr></table>

Table 3 | Ablation study for PaperBanana-Interact. We study three design choices: (1) the critic design, comparing our multi-objective critic (MO) with a naive critic that does not enforce structured multi-objective critique (Naive); (2) the multi-image history provided to agents, including the summarized memory �<sub>�</sub> (Sum), no history beyond $I _ { t - 1 }$ (None), and the full history $\mathbf { I } _ { < t }$ (Full); and (3) the iteration budget $\tau _ { m a x }$ of the internal loop. All runs refine PaperBanana outputs for 5 turns with the $k = 1$ user simulator. The highest scores are shown in bold, and the second-highest scores are underlined.

critical context for the task. Conversely, feeding agents the full history (Full) slightly reduces forgetting but degrades per-turn requirement satisfaction (77.2 → 73.5) and final quality, while consuming 37.1% more input tokens. Overall, compact textual memory ofers the best trade-of. Finally, the iteration budget shows the most significant impact on performance, with a monotonic decline as �<sub>max</sub> decreases from 10 to 1, where $\tau _ { \operatorname* { m a x } } = 1$ denotes a single refinement pass without further iterations.

## 6. Conclusions

In this work, we investigate multi-turn scientific diagram generation and refinement guided by iterative human feedback. Motivated by a formative user study, we introduce MTPaperBananaBench to benchmark this task using a user simulator that progressively reveals requirements across turns. Evaluating requirement satisfaction and diagram quality reveals two key failure modes in existing systems: quality drift across refinement turns and the forgetting of previously satisfied requirements. To address these challenges, we present PaperBanana-Interact, a multi-agent refinement framework featuring a history summarizer and a multi-objective critic. Empirical results demonstrate that PaperBanana-Interact consistently mitigates quality drift and reduces requirement forgetting. Overall, this work provides a foundation for future research on interaction-aware scientific diagram generation and refinement.

## References

J. Belouadi, A. Lauscher, and S. Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. In International Conference on Learning Representations, volume 2024, pages 55917–55943, 2024a.

J. Belouadi, S. P. Ponzetto, and S. Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. Advances in Neural Information Processing Systems, 37:85074–85108, 2024b.

J. Belouadi, E. Ilg, M. Keuper, H. Tanaka, M. Utiyama, R. Dabre, S. Eger, and S. Ponzetto. Tikzero: Zero-shot text-guided graphics program synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17793–17806, 2025.

T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392– 18402, 2023.

S. Cao, H. Chen, P. Chen, Y. Cheng, Y. Cui, X. Deng, Y. Dong, K. Gong, T. Gu, X. Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Google DeepMind. Nano banana pro (Gemini 3 Pro Image). https://deepmind.google/models/ gemini-image/pro/, 2025. Image generation and editing model. Accessed: 2026-08-02.

Google DeepMind. Gemini 3.1 pro: Model card. https://deepmind.google/models/ model-cards/gemini-3-1-pro/, Feb. 2026. Accessed: 2026-07-31.

M. Hahn, W. Zeng, N. Kannen, R. Galt, K. Badola, B. Kim, and Z. Wang. Proactive agents for multi-turn text-to-image generation under uncertainty. In International Conference on Machine Learning, pages 21591–21628. PMLR, 2025.

A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani. Imagic: Text-based real image editing with difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017, 2023.

Z. Lin, Q. Xie, M. Zhu, S. Li, Q. Sun, E. Gu, Y. Ding, K. Sun, F. Guo, P. Lu, et al. Autofigure-edit: Generating editable scientific illustrations via reference-guided styling. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 57–67, 2026.

Meta. Introducing muse image: Image generation built for your world. https://about.fb.com/ news/2026/07/introducing-muse-image-meta-ai/, July 2026. Originally published July 7, 2026. Accessed July 22, 2026.

O. Nabati, G. Tennenholtz, C. Hsu, M. Ryu, D. Ramachandran, Y. Chow, X. Li, and C. Boutilier. Personalized and sequential text-to-image generation. arXiv e-prints, pages arXiv–2412, 2024.

OpenAI. GPT Image 2. OpenAI API Documentation, 2026. URL https://developers.openai. com/api/docs/models/gpt-image-2.

W. Pang, K. Q. Lin, X. Jian, X. He, and P. Torr. Paper2poster: Towards multimodal poster automation from scientific papers. Advances in Neural Information Processing Systems, 38, 2026.

L. Zhang, S. Eger, Y. Cheng, W. Zhai, J. Belouadi, F. Moafian, and Z. Zhao. Scimage: How good are multimodal large language models at scientific text-to-image generation? In International Conference on Learning Representations, volume 2025, pages 6923–6948, 2025.

H. Zhao, S. Si, Z. Wang, Z. Wang, L. Chen, X. Li, Z. Liang, M. Sun, and M. Zhang. Crafter: A multi-agent harness for editable scientific figure generation from diverse inputs. arXiv preprint arXiv:2605.30611, 2026.

H. Zheng, X. Guan, H. Kong, W. Zhang, J. Zheng, W. Zhou, H. Lin, Y. Lu, X. Han, and L. Sun. Pptagent: Generating and evaluating presentations beyond text-to-slides. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 14413–14429, 2025.

D. Zhu, R. Meng, Y. Song, X. Wei, S. Li, T. Pfister, and J. Yoon. Paperbanana: Automating academic illustration for ai scientists. In Forty-third International Conference on Machine Learning, 2026a.

M. Zhu, Z. Lin, Y. Weng, P. Lu, Q. Xie, Y. Wei, S. Liu, Q. Sun, and Y. Zhang. Autofigure: Generating and refining publication-ready scientific illustrations. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=5N3z9JQJKq.

## A. Potential Risks

We acknowledge the ethical risks associated with generative models, particularly the potential for “visual hallucinations”, in which representations of scientific content appear plausible but are factually incorrect. We will explicitly state that the tool is intended solely as an assistive system and will prominently display this disclaimer in both the README file and the main page of the demonstration interface. Following the practices outlined in Zhu et al. (2026b) and Lin et al. (2026), we intend to release the software under a dedicated license requiring that any academic publication using images generated by the tool must (a) clearly disclose that the image was generated by this system; and (b) include a section discussing the role of artificial intelligence in the research. We emphasize that users should not over-rely on such systems and must maintain rigorous human oversight to preserve the scientific accuracy and integrity of published illustrations.

![](images/c44d234698a66bc8804ec4e5b2ad4e654a2507c8bb96c05599a7e90c5c30c682.jpg)

![](images/d54c67a2bb34eda27ee0b7774a51fa335fc8c3d85a36d62bf157d59439c76312.jpg)

![](images/0bcbcbb39b310687c28c8f5c10dd8d95237c17732702151736cacfd9e9161c3c.jpg)  
Figure 8 | Distribution of the number of requirements per diagram, as well as the distribution of requirements across taxonomy categories in MTPaperBananaBench.

## B. Data Annotation Details

As described in §3.3, we annotate the requirements using a two-stage process. First, Gemini-3.1-Pro is prompted to generate candidate requirements; human annotators then review and revise them. Both the LLM and the human annotators are provided with the same taxonomy to promote broad and balanced coverage. The taxonomy is defined as follows:

• Semantic Category: (1) Content, specifying what information appears in the diagram, what is omitted, which functional relationships are shown, and the level of mathematical or functional detail; (2) Organization, specifying how information is spatially arranged, ordered, and grouped; and (3) Visual representation, specifying the visual encodings, styles, metaphors, and overall aesthetic treatment used to render the diagram.

• Impact Severity: (1) Critical, where a violation makes the figure scientifically, mathematically, or structurally incorrect; (2) Communicative, where a violation preserves technical correctness but makes the figure confusing, overwhelming, or prone to misinterpretation; and (3) Stylistic, where a violation afects neither accuracy nor comprehension, but reduces aesthetic quality or professional polish.

Annotators are instructed to (1) maintain an approximately even balance across the semantic categories and (2) ensure coverage of all three severity levels.

During manual review, annotators are provided with the following information: (1) the source context from the paper, (2) the human-drawn diagram, and (3) the model-generated requirement description and taxonomy labels. They revise the requirement annotations to ensure (1) faithfulness to the source context, (2) faithfulness to the diagram, and (3) correct taxonomy labeling. Because our goal is to capture the original authors’ design choices, we treat the human-drawn diagram as the primary source of evidence for visual and structural decisions. To further support the annotation process, we prompt both Gemini-3.1-Pro and Claude-Opus-4.8 to independently assess the faithfulness of each requirement and verify its taxonomy labels. Requirements are flagged for closer inspection whenever either model identifies a potential faithfulness issue or the two models disagree on the taxonomy labels. In addition, Gemini-3.1-Pro provides a concise summary of the paper to help annotators understand the relevant technical context. Finally, since figure captions may unintentionally reveal hidden requirements, we prompt Gemini-3.1-Pro to simplify each caption, removing content redundant with the annotated requirements.

In total, three annotators, all authors of this paper, participated in the annotation process. We began with annotation training, followed by the joint annotation of 15 data points, during which discrepancies were discussed and resolved immediately. The annotators then worked independently on separate batches containing overlapping samples, enabling us to compute inter-annotator agreement (IAA). As reported in §3.3, we computed IAA on a challenging subset of 207 requirements that were flagged as potentially problematic by at least one of Gemini-3.1-Pro or Claude-Opus-4.8. On this subset, the three annotators achieved substantial agreement, with a Cohen’s � of 0.767, supporting the reliability of the annotations. Detailed statistics of the final dataset are reported in Figure 8.

<table><tr><td></td><td>Min</td><td>Max</td><td>Median</td></tr><tr><td>Papers authored or significantly contributed to in total</td><td>3</td><td>30</td><td>11</td></tr><tr><td>Papers authored or significantly contributed to in the past two years</td><td>1</td><td>25</td><td>8</td></tr><tr><td>Paper diagrams personally created or substantially edited in total</td><td>2</td><td>50</td><td>10.5</td></tr><tr><td>Paper diagrams personally created or substantially edited in the past two years</td><td>1</td><td>25</td><td>6.5</td></tr></table>

Table 4 | Summary statistics for participants’ prior experience in paper drafting and diagram drawing.

## C. Details for Formative User Study

In this section, we report the detailed protocol for our formative user study. We recruited 14 participants internally from the same organization, with national backgrounds including the United States, China, and India. All participants had substantial experience in academic writing and scientific illustration. Specifically, participants were required to have (1) authored or made significant contributions to at least one research paper, and (2) personally created or substantially edited at least one paper diagram within the past two years. Statistics of participants’ self-reported experience are shown in Table 4. We obtained consent from all participants to use the results to motivate our study design and to report de-identified results in the final paper.

During the study, each participant selected a paper for the diagram creation task. To ensure suficient familiarity with the content, the selected paper was required to be one to which they had made substantial intellectual contributions. We encouraged participants to use an ongoing paper whose methodology section was largely complete but whose main figure had not yet been created. If no such paper was available, participants instead selected a previously published paper and were instructed to adopt a post-hoc mindset: rather than reproducing the original figure, they were asked to imagine redesigning or revising it for a keynote presentation or journal extension, with the goal of communicating the core idea more efectively. Among the 14 participants, 5 selected ongoing papers and 9 selected prior papers.

After paper selection, participants provided the methodology section together with a figure caption describing the intended purpose of the diagram. The interface for this stage is shown in Figure 9. PaperBanana then generated an initial diagram. Participants could subsequently interact with PaperBanana-DirectRefine to iteratively refine the result using natural language feedback. They were free to stop whenever they felt satisfied, with a maximum study duration of 30 minutes. The interaction interface is shown in Figure 10. Every participant requested at least one revision. Participants stopped after a median of 5 interaction turns, with the full distribution shown in Figure 12.

Following the interaction session, we conducted a semi-structured interview consisting of the following questions: (1) Rate your satisfaction with the initial diagram on a 1–5 scale. (2) Rate your satisfaction with the final diagram on a 1–5 scale. (3) Did seeing the initial diagram change or shape how you thought the method should be presented? (4) For each revision turn, why did you make this request? (5) For each revision turn, did the resulting diagram introduce any new problems? If so, why?

![](images/94dd67273634e9ccc144953e6ee571afb4b266a11db4fe031aae8e162657436d.jpg)

Figure 9 | Interface for the initial diagram generation in the formative user study. Participants provided the methodology section of their selected paper and a figure caption describing the intended diagram.  
![](images/6cda7c127560b51c589c9b6ce6d5c8f8835e2906b5e8ee1a8f6f7ddfd7d0f75c.jpg)  
Figure 10 | Interface for multi-turn diagram refinement in the formative user study. The generated diagram and its prior revisions are displayed on the left, while the chat panel on the right allows participants to iteratively refine the diagram through natural-language feedback. User-related information is hidden.

Overall satisfaction scores are summarized in Figure 2. Figure 11 further separates the results by whether participants used ongoing or prior papers; we observe no significant diference in the distributions between the two groups. Overall, participants consistently reported higher satisfaction with the refined diagrams than with the initial generations.

The interview responses provide further insight into the iterative design process. Nine of the 14 participants (64%) reported that viewing the initial diagram changed or refined their thinking about how the method should be presented. This observation is further reflected in participants’ explanations of their revision requests. When asked “Why did you raise this request?”, participants in 9 of the 14 sessions (64%) reported at least one revision that was prompted by new ideas about how or what to illustrate after viewing the previous diagram, rather than merely correcting an existing deficiency.

![](images/bebc17c0f088949b6bf8aef118e0bf24fa509b4de4fcaa09df51fb55c98e8da6.jpg)  
Figure 11 | Per-participant ratings (1–5 scale) for the diagrams before and after multi-turn refinement, split according to whether participants designed diagrams for prior or ongoing papers. Score changes (Δ) are annotated above each bar.

![](images/871771e51b50db42f57dfe59bb837f35308c3535c9ab033c7311c99799e43f46.jpg)  
Figure 12 | Number of refinement turns per participant before they stopped interacting with the system.

The interviews also reveal a complementary phenomenon: the iterative refinement process itself could introduce new problems. When asked whether a revision result had introduced another issue, participants in 11 of the 14 sessions (79%) reported at least one such occurrence. The detailed breakdown of these issues is reported in the main paper: reduced factual faithfulness in 29% of sessions, degraded aesthetic quality in 43% of sessions, and failure to preserve previously satisfied requests in 29% of sessions.

## D. Experimental Details

For Gemini-3.1-Pro, we use the gemini-3.1-pro-preview model with a temperature of 1.0 and a permissive maximum output length of 50,000 tokens. For NanoBananaPro, we use the gemini-3-pro-image model with a temperature of 1.0. Following the respective settings used in the oficial PaperBanana (Zhu et al., 2026a) and Crafter (Zhao et al., 2026) implementations, we generate Crafter images with a 16:9 aspect ratio at 2K resolution. For PaperBanana, we use the aspect ratio specified by each data point and generate images at 1K resolution. For GPT-Image-2, we use the gpt-image-2 model with the image size set to 1536 × 1024, quality set to high, background set to opaque, and output format set to PNG.

We then report the computational cost of generating an initial diagram with Paper-Banana and refining it with PaperBanana-Interact $( \tau _ { \operatorname* { m a x } } = 1 0 )$ for five turns using the �=1 user simulator. For each diagram, the initial generation round consumes 186.8�/6.2� input/output tokens for Gemini-3.1-Pro and 5.7�/5.3� input/output tokens for NanoBananaPro. In each subsequent refinement turn, PaperBanana-Interact consumes 159.4�/11.2� input/output tokens for Gemini-3.1-Pro and 7.1�/5.0� input/output tokens for NanoBananaPro per diagram. Most of the computational overhead

![](images/7744e62cf6c69c169064171f64700d267728192b0b85a43174f6fd3da5f0a867.jpg)  
Figure 13 | Distribution of the number of internal agentic iterations performed by PaperBanana-Interact.

arises from the internal refinement iterations. Figure 13 shows the distribution of the number of iterations performed by PaperBanana-Interact before stopping. The distribution is concentrated at small iteration counts and generally decreases as the number of iterations increases, with a spike (22%) at the maximum iteration limit, corresponding to challenging examples that cannot satisfy the stopping criterion within the given budget.

Due to computational constraints, we do not report full descriptive statistics for every experiment; most numbers in the main text are therefore from a single run. To estimate the variance, we rerun one of our main settings, refining PaperBanana outputs (fixed throughout five runs) with PaperBanana-Interact using the �=1 user simulator. Across five runs, we obtain (mean ± SD) a diagram quality of $6 1 . 2 \pm 1 . 8$ , requirement satisfaction of $5 8 . 0 \pm 0 . 4 _ { \mathrm { { i } } }$ , per-turn requirement satisfaction of $7 7 . 2 \pm 0 . 3$ and forgetting rate of $1 2 . 6 \pm 1 . 2$ . Diagram quality is slightly less stable, likely due to the subjective nature of the quality definition, while the other metrics show low variance. We report the mean values in Tables 1 and 3 in the main text.

## E. Qualitative Examples

We show a complete five-turn refinement trajectory of PaperBanana-Interact starting from the PaperBanana output in Figure 14, where user feedback is produced by the � =1 simulator. We show additional qualitative examples in Figures 15, 16, and 17, showing diferent quality drift examples for NanoBananaPro and PaperBanana-DirectRefine.

## F. Evaluation Details

## F.1. Diagram Quality

We adopt the evaluation metric from PaperBanana (Zhu et al., 2026a) to assess diagram quality in terms of both content and presentation. Specifically, the metric compares a model-generated diagram �<sub>�</sub> with a human reference diagram �<sub>�</sub> across four dimensions, including two primary dimensions: (1) faithfulness, which evaluates alignment with the methodology description and diagram caption, and (2) conciseness, which evaluates focusing on core information without visual clutter; and two secondary dimensions: (1) readability, which evaluates intelligible layouts, legible text, no excessive crossing lines, and so on, and (2) aesthetics, which evaluates adherence to the stylistic norms of academic manuscripts.

For each dimension, the generated diagram is assigned a win, tie, or loss relative to the reference diagram. These dimension-level outcomes are aggregated into an overall win, tie, or loss using a two-tier voting procedure. If the primary dimensions yield a decisive winner, meaning that one diagram wins both dimensions or wins one dimension while tying the other, that result determines the overall outcome. If the primary dimensions result in a tie, such as when each diagram wins one dimension or both dimensions are tied, the same rule is applied to the secondary dimensions. Finally, a win, tie, and loss are assigned scores of 100, 50, and 0, respectively. We use exactly the same evaluation prompt as reported by Zhu et al. (2026a) and provided in their codebase.

## F.2. Requirement Satisfaction

For requirement satisfaction evaluation, we prompt the VLM with the model-generated diagram and all associated requirements, and prompt it to generate a structured output containing a binary score and rationale for each requirement. This batched evaluation aims to save computation. The full prompt is reported in Listing 1.

In addition to the human evaluation reported in §5 and also in Appendix F.3, we also compare diferent VLMs as evaluators. We compare Gemini-3.1-Pro and Claude-Opus-4.8 as requirement

[t=0] Initial draft  
![](images/2dcf4dacce3b60adaced7bb33e3a8530a706639bc4eb780d55f848632860a009.jpg)

[User at t=2] Instead of just having text labels for the Sender and Receiver distributions in the middle, could you draw them as actual continuous curves? Please stack those curves over on the far right side so they line up vertically for the KL comparison.  
![](images/3786e444ace6d382ebd6e298549f4790abb38eb38c97617e6f40e41aa47c5d1c.jpg)

[User at t=4] The atom nodes in the top left zoomed-in box look good, but they are just connected by plain black lines right now. Could you change those lines into colored arrows to clearly show the direction of the structural vectors?  
![](images/f1062cf65b9ecef86570d0ba34cef99bd5ec2d90d62634058172c59eb67eeb8f.jpg)

[User at t=1] The bottom section is completely missing the Sender and Receiver distributions. Can you add those in and connect them with a double-headed arrow labeled 'KL' to show the loss objective?  
![](images/0777c0cece39d1890d56d31a61cda5712db2c7f74d5d4aed7a5ecbaa83b51713.jpg)

[User at t=3] For those pyramid shapes in the top part, can you make the inside lines dashed? Right now all the edges are solid, so it doesn't really look like a see-through 3D shape.

![](images/d52d89410c12828e494c78989d683d444c0ad24456cde7d09cfc960b2e82e802.jpg)

[User at t=5] Can you put the timestep sampling math and the noisy channel sections into their own colored boxes? Right now they're just floating text, but l'd like them to look like distinct modules with different background colors, similar to the GNN and Bayesian Update blocks.

![](images/0e6e195714bb5a5bebf6884f5ac3104255fee5ca1f347ae0b4eed481ffac4cdf.jpg)  
Figure 14 | A five-turn refinement trajectory of PaperBanana-Interact starting from the PaperBanana output. User feedback is produced by the �=1 simulator.

![](images/2e041496f78ce8d2d07be1b31940751a6c7170b8730edf44290c4315fd08b503.jpg)  
Figure 15 | NanoBananaPro refiner outputs with sudden drops in aesthetics.

![](images/99b5c718cf618e22c8de116777dad60ffd3aeda8297de8288fda1550b151bb46.jpg)  
Figure 16 | NanoBananaPro refiner produces distorted artifacts when editing an image repeatedly.

![](images/0a13b0260b636c0d0bb619e7fdf9038e02e80361a70f7d10de9451a45088427d.jpg)  
Figure 17 | PaperBanana-DirectRefine sometimes sufers from unexpected styling drift.

satisfaction evaluators on full outputs from PaperBanana-DirectRefine refining PaperBanana outputs for five turns with the � =3 user simulator. This comparison yields a Cohen’s � of 0.663, indicating substantial agreement and validating the robustness of the evaluation.

## F.3. Human Evaluation

We conducted two human evaluations, both performed by authors of this paper, to validate the reliability of our experimental results.

First, to validate our proposed requirement satisfaction metric, a human evaluator independently assessed 200 requirements against diagrams generated by PaperBanana-Interact after refining Paper-Banana outputs for 5 turns with the �=3 user simulator. For each case, the evaluator gave a binary judgment: satisfied or unsatisfied. Comparison with the VLM judge outputs yielded a Cohen’s � of 0.812, indicating near-perfect agreement.

Second, to validate the improved diagram quality of PaperBanana-Interact, a human evaluator ranked three outputs per sample, produced by the NanoBananaPro refiner, PaperBanana-DirectRefine, and PaperBanana-Interact after each refined PaperBanana outputs for 5 turns with the � = 3 user simulator. With system identities hidden, the evaluator ranked the three diagrams using the same criteria as in our main quality evaluation: faithfulness and conciseness as primary dimensions, followed by readability and aesthetics. Across 150 data points, PaperBanana-Interact was preferred over PaperBanana-DirectRefine in 76.7% of cases and over NanoBananaPro in 81.3% of cases, confirming the superior quality of PaperBanana-Interact.

<table><tr><td rowspan=1 colspan=1>Setup</td><td rowspan=1 colspan=1>Failed Requirements</td><td rowspan=1 colspan=1>User Utterance</td></tr><tr><td rowspan=1 colspan=1>k=1</td><td rowspan=1 colspan=1>Requirement: In Panel B, the continuous-time distribution curves for &#x27;Re-ceiver Distribution&#x27; and &#x27;Sender Distribution&#x27; must be vertically alignedon the far right to allow a direct vertical spatial connection for the KLdivergence comparison.Evaluation rationale: The &#x27;Receiver Distribution&#x27;and&#x27;Sender Distribution&#x27;are represented as text labels within a dashed box in the center-left ofthe panel, not as continuous-time distribution curves aligned on the farright.</td><td rowspan=1 colspan=1>Instead of just having text labels forthe Sender and Receiver distributionsin the middle, could you draw themas actual continuous curves? Pleasestack those curves over on the farright side so they line up verticallyfor the KL comparison.</td></tr><tr><td rowspan=1 colspan=1>k=3</td><td rowspan=1 colspan=1>Requirement # 1: The &#x27;Backbone Atoms&#x27; component must be styled asa node-and-edge graph utilizing circular nodes for atoms and explicitlycolored directional arrows to represent structural vectors (e.g., v1, v2).Evaluation rationale: Although the Backbone Atoms component usescircular nodes (spheres) to depict atoms, the edges connecting theminside the main tetrahedral structure are plain solid lines. The 1-vectordiagram features a basic black arrow, but the diagram strictly fails to use&quot;explicitly colored directional arrows&quot; to represent structural vectors,nor does it include labels like v1 or v2.Requirement # 2: The primary operational blocks in Panel B (BayesianUpdate, Geometric Algebra GNN, Sample timestep, Noisy channel) mustuse distinct, contrasting background fill colors (e.g., blue, orange, pink)to aesthetically differentiate the functional modules.Evaluation rationale: While the &quot;GNN Module&quot; (orange) and &quot;BayesianUpdate&quot; (beige) use background fill colors, the diagram does not containdesignated operational blocks for &quot;Sample timestep&quot; or &quot;Noisy channel&quot;rendered with distinct background fill colors (they only appear as float-ing text or graph titles). Consequently, the requirement is not fullyimplemented across all specified modules.Requirement # 3: The diagram must explicitly include two majorsub-sections labeled &#x27;Panel A&#x27; for geometric information extraction and&#x27;Panel B&#x27; for the Bayesian Flow Network pipeline.Evaluation rationale: While the diagram includes two major, visuallydistinct sub-sections for geometric information extraction (top) and theBayesian Flow Network (bottom), it fails to explicitly label them withthe required text &#x27;Panel A&#x27; and &#x27;Panel B&#x27;.</td><td rowspan=1 colspan=1>The connections between the atomsin that tetrahedron part need to becolored arrows rather than just plainsolid lines, and they should have la-bels like v1 and v2. Also, down in thebottom section, &quot;Sample timestep&#x27;and &quot;Noisy channel&quot; need to be drawnas actual colored boxes just like theGNN and Bayesian Update ones sothey clearly stand out. Oh, and onelast thing, could you explicitly add a&quot;Panel A&quot; label to the top section and&quot;Panel B&quot; to the bottom one?</td></tr></table>

Table 5 | Examples of user simulator inputs and outputs, with each input specifying the failed requirements and the corresponding evaluation rationale. We show two examples with � =1 and � =3, respectively. The �=1 example corresponds to the second turn in Figure 14.

## G. User Simulation Details

We show the system prompt for the user simulator in Listing 2. Examples of input information and simulated user utterances are shown in Table 5.

## H. PaperBanana-Interact Details

For the summarizer, we report the system prompt in Listing 3, with an example summarizer output in Listing 4. In the refinement loop, the critic and refiner are implemented as consecutive turns within a single multi-turn conversation. They use the shared system prompt shown in Listing 5 for the first iteration (� = 1) and Listing 6 for subsequent iterations (� > 1). The critic is invoked first using the prompt in Listing 7 for the first iteration (� = 1) or Listing 8 for later iterations (� > 1). Its response is then appended to the conversation history. The refiner is subsequently invoked using the prompt shown in Listing 9, conditioned on both the original input and the critic’s output. Finally, the system prompt for the visualizer is provided in Listing 10.

Listing 1: System prompt for requirement satisfaction evaluation.   
You are an expert judge in academic visual design. You will be presented with a   
list of user requirements, a diagram caption, a method section, and a   
model-generated diagram. You will carefully evaluate the correctness of EACH   
requirement separately and provide a binary (Satisfied/Unsatisfied) judgment.   
# Inputs   
1. \*\*User Requirements\*\*: [list]   
2. \*\*Diagram Caption\*\*: [text]   
3. \*\*Method Section\*\*: [text]   
4. \*\*Model-generated Diagram\*\*: [image]   
### Decision Criteria for EACH User Requirement   
\*\*1. Grounding via Context (Method & Caption):\*\*   
Before evaluating the visual execution, use the \*\*User Requirement\*\* as your   
primary source for the expected design. Then, rely on the provided \*\*Method   
Section\*\* and \*\*Diagram Caption\*\* as your definitive sources of truth for   
context, background information, and disambiguation. Use these texts to:   
\* Understand the underlying academic logic and architecture with absolute   
precision.   
\* Identify exactly what specific modules, variables, and mathematical notations   
represent.   
\* Strictly map the components mentioned in the User Requirement (e.g., "the   
secondary attention head," "the Discriminator module") to their corresponding   
visual elements in the diagram.   
\*\*2. Determine the Status (No Partial Credit):\*\*   
You must evaluate the diagram against the User Requirement as a strict   
pass/fail. There is absolutely zero partial credit.   
\* To be marked \*\*"Satisfied"\*\*, the execution must be \*\*absolute and   
flawless\*\*. Every single mandatory instruction, content rule, layout rule, and   
visual element must be perfectly implemented - \*\*zero errors, zero deviations,   
and zero violations are tolerated\*\*. Furthermore, the execution must \*\*fully   
and successfully convey\*\* the intended communicative goal.   
\* If even a single binding constraint is missing, misplaced, ambiguous, or only   
partially implemented, the status is immediately \*\*"Unsatisfied"\*\*.   
\*\*3. Handling OR-Conditions:\*\*   
If the User Requirement explicitly offers multiple acceptable options for an   
element (e.g., "integrate an icon representing vision, such as an eye or a   
magnifier"), the diagram must perfectly execute \*\*exactly one\*\* of those   
options. Furthermore, the chosen solution must be applied consistently

```markdown
throughout the entire diagram. Mixing multiple options inconsistently or
executing the chosen option poorly results in an **"Unsatisfied"** rating.
**4. Evaluating Non-Binding Recommendations and Examples:**
User Requirements may contain binding constraints alongside non-binding
recommendations or illustrative examples (e.g., "highlight the active path with
a bright color, e.g., red or orange"). If the diagram deviates from a specific
illustrative example but adopts a valid alternative design choice (e.g., using
bright yellow instead), you must evaluate it rigorously:
* The diagram passes only if the alternative design **flawlessly** achieves the
**exact same communicative goal** as the binding requirement (e.g., it still
successfully highlights the active path).
* The alternative design must maintain academic readability and perform **equal
to or better than** the proposed example. If the alternative introduces visual
f i k h l i f h di h i
**"Unsatisfied"**.
# Output Format (repeat this format for EVERY requirement, starting with a new
line)
Requirement: [Repeat the requirement, word for word, without making any
changes. Keep everything including punctuation and capitalization as-is.]
Rationale: [Explain your reasoning, detailing how the diagram satisfieds or
does not satisfy the requirement, or why the requirement is not applicable.]
Verdict: [Satisfied|Unsatisfied]
# Example
## Input
<requirements>
* The diagram must explicitly include nodes for the variables $x_t$ across at
least three timesteps, such as $x_1$, $x_2$, and $x_3$, to clearly convey
multi-step temporal behavior.
* The $x_t$ nodes for successive timesteps, e.g., $t=1,2,3,\\ldots$, must be
arranged in a strict left-to-right sequence to visually encode their temporal
order.
* The diagram should use a soft pastel color palette to support a polished,
publication-quality academic aesthetic.
</requirements>
(Context Omitted)
## Output
Requirement: The diagram must explicitly include nodes for the variables $x_t$
across at least three timesteps, such as $x_1$, $x_2$, and $x_3$, to clearly
convey multi-step temporal behavior.
Rationale: The diagram includes separate nodes labeled $x_1$, $x_2$, and $x_3$,
so it explicitly represents the variable across at least three timesteps. This
satisfies the requirement of showing multi-step temporal behavior at the node
level.
Verdict: Satisfied
```

Requirement: The \$x\_t\$ nodes for successive timesteps, e.g., \$t=1,2,3,\\ldots\$,   
must be arranged in a strict left-to-right sequence to visually encode their   
temporal order.   
Rationale: Although the diagram includes multiple \$x\_t\$ nodes, they are   
arranged in a two-dimensional grid rather than a strict left-to-right sequence.   
As a result, the temporal ordering is not visually encoded clearly and may be   
read as a structural grouping rather than a chronological progression.   
Verdict: Unsatisfied   
Requirement: The diagram should use a soft pastel color palette to support a   
polished, publication-quality academic aesthetic.   
Rationale: The dia ram uses stark, hi hl saturated colors rather than a soft   
pastel palette. This makes the visual style feel less subtle and less aligned   
with the intended polished academic aesthetic.   
Verdict: Unsatisfied

Listing 2: System prompt for user simulator. The highlighted text depends on the hyperparameter   
�.   
You are role-playing as the (non-expert) user who asked an assistant to create   
a figure for your research paper. You are now looking at the latest version of   
the figure and you are not fully happy with it.   
You are NOT a designer. You do not know the technical vocabulary of visual   
design, and you should not sound like an expert reviewer. You just roughly know   
what you wanted, and you can tell when something looks off.   
## Your task   
Look at the current figure and the notes about what is wrong with it, then   
write ONE short, casual message asking for those things to be fixed. React like   
a real, slightly impatient user - not a careful, exhaustive reviewer.   
## What you are given   
- Current figure: the attached image.   
- What’s wrong: internal evaluations explaining how the figure currently fails   
specific requirements. Use these ONLY to understand what is actually wrong. Do   
NOT quote them or mention that an evaluation exists.   
- What you wanted: the underlying requirements you originally cared about.   
These are the ground truth for what to ask for. Note the assistant never saw   
these spelled out - it only worked from the method section and caption below -   
so unless you already raised one of them in an earlier turn, it had no way of   
knowing they mattered to you.   
- Conversation so far: earlier turns between you ("User") and the assistant.   
Each of your past messages may be followed by an internal note recording which   
requirements you raised and why - use these only to remember what you already   
complained about; never quote or mention them.   
- Paper context (method section & figure caption): background only, so your   
complaint makes sense.   
## How to write the feedback   
- Keep it short: \*\*AT MOST TWO\*\* sentences for \*\*EACH\*\* requirement (k   
requirements in total), casual and conversational.   
- Raise all k issues you are given, in the order given (most important first).   
Do not raise problems beyond these.

```markdown
- Blend the k issues into one smooth, natural message. Connect the complaints
the way a real person would ("also", "and one more thing", "oh, and...") - and
vary how you connect them from message to message.
- Talk like a normal person ("the arrows are kind of confusing", "can you make
the boxes bigger?"), not like a designer ("increase the stroke weight", "fix
the visual hierarchy").
- Describe the problem or what you want, not the exact design fix. Don’t be
overly specific or prescriptive.
- Don’t mention the evaluation, the requirements, scores, rubrics, or that this
is a simulation. Just talk about the picture.
- Usually don’t repeat feedback you already gave in earlier turns. But
sometimes you’ll need to re-raise an issue you brought up before - either it
was never really fixed, or it got fixed once and then broke again. When that
happens, it’s fine to sound a bit annoyed, like a real person would.
## Output
Return ONLY the user’s message text - no quotes, no preamble, no explanation.
Listing 3: System prompt for summarizer. The highlighted text is constructed on the fly, depending
on the number of prior turns.
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS
2025), acting as the note-taker for a diagram revision conversation.
## TASK
You are given the full chat history between a user and an AI diagram-generation
system. Each [DiagramN] is attached as an image. The user’s initial request
(the ’Methodology Section’ and ’Figure Caption’) precedes [Diagram1], and each
user turn reacts to the diagram generated right before it.
Your task is to annotate the chat history so that it remains fully
interpretable without the images: write exactly one context note for every turn.
**For each assistant turn [DiagramN]:** summarize what the diagram shows;
enumerate the concrete changes made relative to [DiagramN-1], if a previous
version exists, and whether those changes addresses the request in user turn
N-1; and, judging from the user’s subsequent reply, state which aspects the
user appears to accept and which remain unsatisfactory.
**For each user turn:** identify the exact element(s) in the diagram that
the utterance refers to, including their location and current visual state;
state precisely what is wrong with them: and specify what change would resolve
the complaint.
Each note must be self-contained: an utterance together with its note must be
sufficient for a reader who has never seen the images to reconstruct what was
shown, what the user objected to, and why.
## OUTPUT
Provide your response strictly in the following JSON format, one key per turn
in conversation order.
‘‘‘json
{
```

```json
"context_diagram1": "Context note for [Diagram1].",
"context_user_turn1": "Context note for user turn 1.",
"context_diagram2": "Context note for [Diagram2].",
"context_user_turn2": "Context note for user turn 2.",
}
CCC
```

Listing 4: Example summarizer output for the five turns shown in Figure 14.   
{   
"context\_diagram1": "The diagram illustrates the RBFN pipeline in two main   
sections. The top section, ’Geometric Algebra GNN for 3D Structure Modeling’,   
shows a DNA/RNA double helix zooming into a magnified view of a tetrahedral   
atomic structure (P\_{i-1}, P\_i, C4’, N1/N9), which is broken down into   
multivector components (0-vector scalar, 1-vector vector, 2-vector bivector,   
3-vector pseudoscalar) that feed into a multi-conformation graph. The bottom   
section, ’Bayesian Flow Network (BFN) for RNA Generation’, shows a time axis   
from t=1 (noisy) to t=0 (clean), depicting a forward process adding noise to a   
sequence and a reverse process using a Geometric Algebra GNN Module and a   
Bayesian Update block to predict categorical distributions. Judging from the   
user’s subsequent reply, they accept the overall pipeline but are unsatisfied   
with the bottom section because it completely lacks representations of the   
Sender and Receiver distributions.",   
"context\_user\_turn1": "The user refers to the bottom section of the diagram,   
which currently shows the forward and reverse processes, the GNN module, the   
Bayesian Update, and categorical bar charts, but omits the Sender and Receiver   
distributions. The user complains that these distributions are missing and   
requests adding them and connecting them with a double-headed arrow labeled   
’KL’ to represent the loss objective.",   
"context\_diagram2": "The diagram retains the two-part structure but   
introduces several visual changes: the top-left double helix is replaced with   
an abstract single-stranded RNA structure, the multi-conformation graph’s   
visual style is altered, and a sampling formula is added to the bottom time   
axis. Relative to Diagram 1, this version addresses the user’s request by   
adding a dashed box in the middle of the bottom section containing the text   
labels ’Receiver Distribution (p\_\\theta)’ and ’Sender Distribution (q\_t)’   
connected by a vertical double-headed arrow labeled ’KL’. Judging from the   
user’s subsequent reply, they accept the inclusion of the distributions and the   
KL arrow, but are dissatisfied that the distributions are depicted merely as   
text labels in the center of the diagram.",   
"context\_user\_turn2": "The user refers to the newly added dashed box in the   
middle of the bottom section containing the text labels for ’Sender   
Distribution’ and ’Receiver Distribution’. The user complains that these are   
just plain text labels and requests that they be drawn as actual continuous   
curves and moved to the far right side of the diagram so they are stacked   
vertically for the KL comparison.",   
"context\_diagram3": "The diagram updates the visual style of the top-left RNA   
strand to a simpler teal curve with magenta dots. Relative to Diagram 2, the   
bottom section has been modified to address the user’s request: the text-only   
labels in the middle have been removed, and two vertically stacked plots   
showing continuous bell curves for the ’Receiver Distribution’ and ’Sender

Distribution’ have been added on the far right, connected by a solid   
double-headed arrow labeled ’KL’. Judging from the user’s subsequent reply,   
they accept the new layout of the bottom section but are now unsatisfied with   
the visual representation of the tetrahedral ’pyramid shapes’ in the top   
section.",   
"context\_user\_turn3": "The user refers to the 3D tetrahedral ’pyramid shapes’   
located in the top section’s magnified view and the ’3-vector (Pseudoscalar)’   
component. The user complains that all the edges of these shapes are drawn with   
solid black lines, which makes them look flat rather than like see-through 3D   
objects. The user requests changing the inside lines of these pyramids to   
dashed lines.",   
"context\_diagram4": "The diagram updates the structural elements in the top   
half. Relative to Diagram 3, the inner edges of the tetrahedral shape in the   
top-left magnified view and the 3-vector pseudoscalar have been changed from   
solid lines to dashed lines, which successfully addresses the user’s request by   
creating a see-through 3D effect. The 0-vector line and some internal graph   
edges were also made dashed. Judging from the user’s subsequent reply, they   
accept the dashed lines and the atom nodes but are unsatisfied with the fact   
that the outer connections between atoms in the top-left magnified view are   
still plain black lines.",   
"context\_user\_turn4": "The user refers to the lines connecting the atom nodes   
inside the top-left zoomed-in box (the tetrahedral structure). The user   
complains that these connections are currently just plain black lines and   
requests changing them into colored arrows to clearly illustrate the direction   
of the structural vectors.",   
"context\_diagram5": "The diagram updates the top-left magnified view.   
Relative to Diagram 4, the plain black lines connecting the atom nodes in the   
tetrahedron have been replaced with directional arrows colored orange, blue,   
green, and yellow. This directly addresses the user’s request to show the   
structural vector directions. Judging from the user’s subsequent reply, they   
are satisfied with the top section’s content but are now unsatisfied with the   
visual grouping of certain elements in the bottom section.",   
"context\_user\_turn5": "The user refers to the timestep sampling math formula   
(’p(t) = 0.02U(0,1) + 0.98B(1.0, 1.9)’) placed above the time axis and the   
noisy channel sections (the Sender and Receiver continuous curve plots) on the   
far right of the bottom section. The user complains that these elements are   
currently floating as text and unboxed plots on the background. The user   
requests placing them into their own distinct colored boxes to match the   
modular look of the ’GNN’ and ’Bayesian Update’ blocks."   
}

Listing 5: System prompt for the critic and refiner in the first iteration, i.e. � = 1. The highlighted   
text is constructed on the fly, depending on the current turn �.   
## ROLE   
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).   
## TASK

Your task is to provide a thorough critique and refinement of the target diagram, [Diagram2], based on the \*\*user’s request\*\*. You are provided with the initial request (’Methodology Section’ and ’Figure Caption’) and the iterative user feedback provided via ’Chat History’. Please preserve the correct elements from previous turns, and directly address the feedback given in the \*\*latest\*\* turn of the Chat History.

This is a multi-turn process carried out in three steps. At each step, follow the specific instruction given in that turn - do not run ahead or fold later steps into an earlier one.

1. \*\*Critique.\*\* Walk the target diagram against every constraint one by one and write up the per-aspect critique, noting for each issue what is wrong and how it should be fixed.

2. \*\*Decide.\*\* Judge whether the diagram still needs revision, and answer with a single word as instructed.

3. \*\*Refine.\*\* Only if revision is needed, you will then be shown the Detailed Description that renders the target diagram, and asked to synthesize your critique into concrete suggestions and a fully revised description.

## ## REVISION RULES - WHAT YOU MUST ENSURE

## 1. Chat history

Address the user’s requirements in both the latest turn and all prior turns. The target diagram builds upon all previous turns, so you should treat past requests differently from the most recent one.

## - \*\*Prior turns:\*\*

一 \*\*Assessing satisfaction:\*\* If a user stops mentioning a specific issue, it usually means the functionality was successfully implemented. Don’t assume it’s still broken.

\*\*Preserving elements:\*\* Inspect the diagram to find components that correspond to earlier requests. This reveals what the user likes. Be sure to preserve these successful elements faithfully, without degrading or undoing them.

\*\*The latest turn:\*\* The request in the \*\*latest\*\* turn directly targets [Diagram2] and, by definition, hasn’t been addressed yet. Identify exactly what is wrong or missing, determine how to fix it, and ensure your revised description fully resolves the issue.

## 2. Source Content

\*\*Fidelity & Alignment:\*\* Ensure the diagram accurately reflects the method described in the "Methodology Section" and aligns with the "Figure Caption." Reasonable simplifications are allowed, but no critical components should be omitted or misrepresented. Also, the diagram should not contain any hallucinated content. Consistency with the provided methodology section & figure caption is always the most important thing.

\*\*Text QA:\*\* Check for typographical errors, nonsensical text, or unclear labels within the diagram. Suggest specific corrections.

\*\*Validation of Examples:\*\* Verify the accuracy of illustrative examples. If the diagram includes specific examples to aid understanding (e.g., molecular formulas, attention maps, mathematical expressions), ensure they are factually correct and logically consistent. If an example is incorrect, provide the correct version.

\*\*Caption Exclusion:\*\* Ensure the figure caption text (e.g., "Figure 1: Overview...") is \*\*not\*\* included within the image visual itself. The caption should remain separate.

```tcl
3. Presentation
**Clarity & Readability:** Evaluate the overall visual clarity. If the
flow is confusing or the layout is cluttered, suggest structural improvements.
**Legend Management:** Be aware that the description and diagram may
include a text-based legend explaining color coding. Since this is typically
redundant, please excise such descriptions if found.
## HOW TO CRITIQUE - WALK EVERY CONSTRAINT ONE BY ONE
Do NOT lump everything into a single blob. Go through the constraints
sequentially and emit a SEPARATE critique entry for each, in this exact order:
1. "critique_latest_turn": critique [Diagram2] against the most recent user
turn (shown as "User (latest turn)" in the Chat History). Identify exactly WHAT
is wrong or missing for this request, its location and current visual state,
and how the revised description will fix it.
2. "critique_prior_turnN": emit one entry for EACH prior user turn, where N is
the user turn’s index shown as "User (turn N)" in the Chat History. In each,
verify the target diagram still satisfies that turn’s requirement, preserve the
successful elements, and flag anything undone or degraded. If there are no
prior turns, emit no such key.
3. "critique_source_content": critique the target diagram against the
’Methodology Section’ and ’Figure Caption’ - fidelity & alignment, text QA,
validation of examples, and caption exclusion.
4. "critique_presentation": critique the target diagram for its overall visual
clarity, presentation, and legend.
Each critique entry should be long and specific, grounded in visual details,
rather than a one-line verdict.
## INPUT DATA
**Target Diagram**: [Diagram2] [the generated figure, attached as an image]
**Detailed Description**: [The detailed description of the figure -
withheld for now; it will be provided only at the refinement step (step 3)]
**Methodology Section**: [Contextual content from the methodology section]
**Figure Caption**: [Target figure caption]
**Chat History**: [List of chat utterances from all turns. Each user turn
is labeled "User (turn N)" or "User (latest turn)". Each prior turn includes a
brief [Context: ...] note summarizing what that diagram showed or what that
user turn requested. Focus on addressing the **LATEST** turn]
```

Listing 6: System prompt for the critic and refiner in later iterations, i.e. � > 1. The highlighted   
text is constructed on the fly, depending on current turn � and iteration �.   
## ROLE   
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).   
## TASK   
Your task is to provide a thorough critique and refinement of the target   
diagram, [Diagram2.1], based on the user’s full request, the source content,   
and the overall presentation. You are provided with the initial request   
(’Methodology Section’ and ’Figure Caption’) and the iterative user feedback   
provided via ’Chat History’.

This is a multi-turn process carried out in three steps. At each step, follow the specific instruction given in that turn - do not run ahead or fold later steps into an earlier one.

1. \*\*Critique.\*\* Walk the target diagram against every constraint one by one and write up the per-aspect critique, noting for each issue what is wrong and how it should be fixed.

2. \*\*Decide.\*\* Judge whether the diagram still needs revision, and answer   
with a single word as instructed.

3. \*\*Refine.\*\* Only if revision is needed, you will then be shown the   
Detailed Description that renders the target diagram, and asked to synthesize   
your critique into concrete suggestions and a fully revised description.

## ## REVISION RULES - WHAT YOU MUST ENSURE

## 1. Chat history & requirements

## \*\*Prior turns:\*\*

一 \*\*Assessing satisfaction:\*\* If a user stops mentioning a specific   
issue, it usually means the functionality was successfully implemented. Don’t   
assume it’s still broken.

\*\*Preserving elements:\*\* Inspect the diagram to find components that correspond to earlier requests. This reveals what the user likes. Be sure to preserve these successful elements faithfully, without degrading or undoing them.

- \*\*The latest turn:\*\*

\*\*Assessing satisfaction:\*\* Step back and comprehensively verify if   
the target diagram satisfies the latest user request. The user’s latest request   
originally targets [Diagram2], and [Diagram2.1] has ALREADY been revised in   
response to the user’s most recent turn. It MAY already have been addressed by   
that revision. However, the user has NOT yet seen this revision, so you CANNOT   
apply the "user stopped mentioning it, so it must be fixed" heuristic - it may   
still be unmet.

\*\*Preserving elements:\*\* If the diagram has already successfully addressed the latest user request, be sure to preserve these successful elements faithfully, without degrading or undoing them.

\*\*Where this critique sits\*\*: Since the diagram has already been revised to address the latest user request, your job now is to step back and comprehensively verify that the target diagram simultaneously satisfies EVERY constraint across all turns and the main context, and to catch any remaining general issues. Do NOT single out the latest turn or prioritize it.

## 2. Source Content

\*\*Fidelity & Alignment:\*\* Ensure the diagram accurately reflects the method described in the "Methodology Section" and aligns with the "Figure Caption." Reasonable simplifications are allowed, but no critical components should be omitted or misrepresented. Also, the diagram should not contain any hallucinated content. Consistency with the provided methodology section & figure caption is always the most important thing.

\*\*Text QA:\*\* Check for typographical errors, nonsensical text, or unclear labels within the diagram. Suggest specific corrections.

\*\*Validation of Examples:\*\* Verify the accuracy of illustrative   
examples. If the diagram includes specific examples to aid understanding (e.g.,   
molecular formulas, attention maps, mathematical expressions), ensure they are   
factually correct and logically consistent. If an example is incorrect, provide   
the correct version.

\*\*Caption Exclusion:\*\* Ensure the figure caption text (e.g., "Figure 1:   
Overview...") is \*\*not\*\* included within the image visual itself. The caption   
should remain separate.   
3. Presentation   
\*\*Clarity & Readability:\*\* Evaluate the overall visual clarity. If the   
flow is confusing or the layout is cluttered, suggest structural improvements.   
\*\*Legend Management:\*\* Be aware that the description and diagram may   
include a text-based legend explaining color coding. Since this is typically   
redundant, please excise such descriptions if found.   
## HOW TO CRITIQUE - WALK EVERY CONSTRAINT ONE BY ONE   
Do NOT lump everything into a single blob. Go through the constraints   
sequentially and emit a SEPARATE critique entry for each, in this exact order:   
1. "critique\_latest\_turn": critique [Diagram2.1] against the most recent user   
turn (shown as "User (latest turn)" in the Chat History). Fold it in as one   
requirement among many - do not prioritize it - but do verify it is actually   
met, since the user has not reviewed this revision yet.   
2. "critique\_prior\_turnN": emit one entry for EACH prior user turn, where N is   
the user turn’s index shown as "User (turn N)" in the Chat History. In each,   
verify the target diagram still satisfies that turn’s requirement, preserve the   
successful elements, and flag anything undone or degraded. If there are no   
prior turns, emit no such key.   
3. "critique\_source\_content": critique the target diagram against the   
’Methodology Section’ and ’Figure Caption’ - fidelity & alignment, text QA,   
validation of examples, and caption exclusion.   
4. "critique\_presentation": critique the target diagram for its overall visual   
clarity, presentation, and legend.   
Each critique entry should be long and specific rather than a one-line verdict.   
## INPUT DATA   
\*\*Target Diagram\*\*: [Diagram2.1] [the generated figure, attached as an   
image - revised from [Diagram2] but not yet reviewed by the user]   
\*\*Detailed Description\*\*: [The detailed description of the figure -   
withheld for now; it will be provided only at the refinement step (step 3)]   
\*\*Methodology Section\*\*: [Contextual content from the methodology section]   
\*\*Figure Caption\*\*: [Target figure caption]   
\*\*Chat History\*\*: [List of chat utterances from all turns. Each user turn   
is labeled "User (turn N)" or "User (latest turn)". Each turn includes a brief   
[Context: ...] note summarizing what that diagram showed or what that user turn   
requested. Check the diagram against ALL turns, not just the latest]

Listing 7: Critic prompt in the first iteration, i.e. � = 1. The highlighted text is constructed on the   
fly, depending on current turn �.   
Please perform the first step: Critique. Walk [Diagram2] against every   
constraint one by one and emit the per-aspect critique, noting for each issue   
what is wrong and how it should be fixed.   
## OUTPUT   
Provide your response strictly in the following JSON format.   
‘‘‘json   
{

Please perform the third step: Refine. The Detailed Description that produced   
{DIAGRAM\_LABEL} is now provided below; revise it based on your critique above,   
producing these two fields sequentially:   
1. "critic\_suggestions": synthesize all of the above into concrete, actionable   
suggestions, trying to incorporate and balance all of the requirements above   
rather than fixing one at the expense of another.   
2. "revised\_description": the fully revised detailed description incorporating   
every suggestion. Follow the "POSITIVE, SPECIFIC DESCRIPTIONS" rule below -   
state what the diagram SHOULD contain, affirmatively and specifically, not what   
it should avoid.

"critique\_latest\_turn": "Your critique against the latest user turn.",   
"critique\_prior\_turn1": "Your critique against user turn 1.",   
"critique\_source\_content": "Your critique against the Methodology Section and   
Figure Caption.",   
"critique\_presentation": "Your critique of the overall visual clarity,   
presentation, and legend.",

```python
depending on current turn � and iteration �.
Please perform the first step: Critique. Walk [Diagram2.1] against every
constraint one by one and emit the per-aspect critique, prioritizing the LATEST
user turn, and noting for each issue what is wrong and how it should be fixed.
## OUTPUT
Provide your response strictly in the following JSON format.
‘‘‘json
{
"critique_latest_turn": "Your critique against the latest user turn.",
"critique_prior_turn1": "Your critique against user turn 1.",
"critique_source_content": "Your critique against the Methodology Section and
Figure Caption.",
"critique_presentation": "Your critique of the overall visual clarity,
presentation, and legend.",
```

Listing 10: System prompt for the visualizer.   
You are an expert scientific diagram illustrator. Generate high-quality   
scientific diagrams based on user requests.

Do NOT phrase the description as "do not ...", "avoid ...", "no ...",   
"without ...", or "remove ..." - an image model cannot render an absence, and   
naming an unwanted element often causes it to appear.   
Instead, describe positively what SHOULD be there: the exact element, its   
content, position, size, color, and style that replaces the problem.   
一 Prevent errors by being MORE specific, not by prohibiting. Where something   
previously went wrong, add more concrete detail (exact text, counts, geometry,   
spatial relations) so the generator has no room to reintroduce the mistake.   
## OUTPUT   
Provide your response strictly in the following JSON format.   
‘‘‘json   
{   
"critic\_suggestions": "Synthesize the above into concrete, actionable   
suggestions for improvement, trying to incorporate and balance all of the   
requirements above rather than fixing one at the expense of another. If the   
diagram is perfect, write ’No changes needed.’",   
"revised\_description": "Insert the fully revised detailed description here,   
incorporating all your suggestions. Phrase it positively and specifically -   
describe what the diagram SHOULD contain (exact content, position, size, color,   
style), never what to avoid or remove. If no changes are needed, write ’No   
changes needed.’"   
}