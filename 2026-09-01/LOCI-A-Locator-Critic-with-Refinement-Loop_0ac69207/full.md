# LOCI: A Locator-Critic with Refinement Loop

Walid Bousselham Mathilde Caron Arsha Nagrani Cordelia Schmid

Google DeepMind

## Abstract

Vision-Language Models (VLMs) still struggle on tasks requiring complex visual understanding. We argue that the core issue is not high-level reasoning, but instead failing to locate critical details in the image. Due to this shortcoming, VLMs generate often plausible but incorrect reasoning based on flawed perceptual grounding. To address this, we propose Locator-Critic (LOCI), a training-free framework that decouples visual search from evidence verification. LOCI employs a Locator agent to propose candidate visual evidence and a separate Critic agent to evaluate its relevance and sufficiency. These agents engage in an iterative refinement loop, progressively improving the evidence until it is adequate to answer the given question. This decoupled, self-correcting process yields substantial performance gains, achieving state-of-the-art results on multiple complex visual benchmarks. LOCI improves accuracy for both open-weight models like Qwen3-VL (+12.1 on V\*, +5.8 on HR-Bench and +11.2 on VisualProbe-Hard) and proprietary models like Gemini 2.5 Pro (+8.9 on V\*, +4.3 on HR-Bench, +4.8 on VisualProbe-Hard).

## 1 Introduction

Modern Vision-Language Models [2, 5, 24] produce sophisticated reasoning traces and thought processes. However, their performance remains limited on tasks demanding detailed and complex visual analysis. We argue this is not a failure of the reasoning process itself. Instead, the main bottleneck is the ability to reliably locate the specific visual cues necessary to answer a question in complex scenes. We observe that when a VLM focuses on an irrelevant or ambiguous region, its reasoning process generates a narrative based on this flawed visual, leading to an incorrect answer.

We argue that this issue arises because the model must both propose and verify visual evidence in a single inference call. Once the model identifies a visual region, it is heavily biased toward confirming its own hypothesis rather than rigorously challenging it. As a result, identifying the wrong visual region often leads to a wrong answer.

To confirm our main hypothesis we carefully design an oracle experiment (see Tab. 1), where we manually provide the ground-truth visual regions to the model. When provided with the correct visual cues, the model is capable of answering almost all the questions of V\* correctly with a valid reasoning process, confirming our main hypothesis. The performance is not bounded by reasoning ability, but primarily by the ability to locate relevant visual information. To mitigate this issue, we propose separating the reasoning process into distinct localization and verification phases.

We introduce the LOCI approach (Locator-Critic with Refinement), a training-free agentic pipeline that decouples (i) finding visual evidence and (ii) verification, using two specialized agents. These agents are the same underlying model, differentiated only by their prompts and inputs. The Locator proposes visual evidence by extracting relevant crops from the image. The Critic independently evaluates whether the proposed crops are sufficient to answer the question, judging only the visual evidence without access to the Locator’s textual reasoning. Importantly, the Critic only accesses the visual evidence and is not provided with the Locator’s textual reasoning. As we demonstrate in Table 4, this separation is crucial to prevent the Locator from propagating its own errors and influencing the Critic’s assessment. Their interaction forms a self-correcting loop. When the Critic rejects the evidence as insufficient, it provides targeted feedback that guides the Locator to search again, as illustrated in Figure 1.

![](images/a74961c74771b94c31910ef43e36e3f8ce5eabc7cfe679bbc68a0cf6a900529f.jpg)  
Figure 1: LOCI approach. To answer the question, our method employs a Locator to find relevant objects and a Critic to provide feedback. It first finds the police car (Iter. 1), then, after a failed attempt (Iter. 2), successfully locates the red car (Iter. 3). With both objects grounded, it correctly infers their spatial relationship to provide the answer.

We find that the LOCI framework is particularly suited for benchmarks with complex and detailed images, such as V\* [20], HR-Bench [18], and VisualProbe [7]. On these benchmarks, our training-free Locator-Critic framework achieves state-of-the-art performance, surpassing prior methods, including recent reinforcement learning-based approaches [7, 30, 16, 26]. We also show that LOCI is flexible, demonstrating consistent improvements across both open-weight models (Qwen3-VL: +12.1/5.8/11.2 points on V\*/HR-Bench/VisalProbe) and proprietary models (Gemini2.5-Pro: +8.9/4.3/4.8 points on V\*/HR-Bench/VisualProbe). Overall, our contributions can be summarized as :

• We propose LOCI, a training-free Locator-Critic framework that improves VLM reliability by decoupling visual evidence finding from evidence verification.

• We significantly outperform the SOTA on multiple benchmarks for Qwen3-VL [14] and Gemini2.5-Pro [2], with consistent improvements over direct prompting and prior methods.

• Extensive ablation studies demonstrate the critical role of decoupled verification and the effectiveness of iterative refinement. We also demonstrate scalability through parallel multi-locator ensembling,

## 2 Related Works

## 2.1 Dynamic Visual Computation and Tool-Use

Recently, tool-augmented MLLMs [3, 22, 17, 28, 7, 30, 16] have shown strong agentic abilities in complex QA tasks by leveraging a broad set of tools (e.g., web browsing, code execution, retrieval). Specifically for image manipulation, Thyme [28] uses code generation with a diverse set of operations like cropping, rotation, and contrast enhancement, activated in the model via SFT and RL stages. Mini-O3 [7] similarly handles complex queries through on-the-fly image manipulation. While powerful, these approaches often require significant training overhead. Our method, in contrast, is entirely training-free and generalizes across both proprietary and open-source models.

Similar to our work, DeepEyes[30], Chain-of-Focus[26] and Pixel Reasoner[16], all aim to equip VLMs with zoom-in and region-of-interest selection, enabling active perception over images. Similarly, DyFo [9] employs a training-free Monte Carlo Tree Search to guide a visual expert’s focus, while Token-Efficient VLM [6] and Griffon-G [23] use complex, multi-stage training pipelines to select high-resolution patches or generate textual coordinates. Unlike these works, we decouple visual detection from verification through a dedicated Critic agent that evaluates the quality of generated crops and provides targeted feedback for iterative refinement.

## 2.2 Critique and Verification in MLLMs

Multiple works try and improve the reliability of models by incorporating an explicit ‘critic’ or ‘verifier’ module. The simplest form of this is self-correction, where a single model critiques its own output. This has been explored in both text-only (eg. SELF-REFINE [12]) and multimodal domains (LLaVA-Critic-R1 [19]). While straightforward to implement, these approaches can suffer from "cognitive fixation," where the model fails to identify its own initial perceptual or reasoning errors because the critic and generator share the same underlying biases.

![](images/fd05f93f4b3f31e81d50fe1cd2f7a2ca04b91f7f72c87e1e30783c6a00ef08c9.jpg)  
Figure 2: Overview of LOCI . A Locator agent generates Python code to create image crops, which are then evaluated by a Critic agent. The Critic’s feedback guides the Locator’s following attempts until sufficient visual evidence is found to answer the question.

To overcome this limitation, recent work has moved towards a decoupled "actor-critic" paradigm, where an independent model verifies the output of the primary reasoning model. For example, Critic-V [25] uses a critic to provide natural language feedback on a reasoning model’s thought process. MMC [11] automatically constructs a critique dataset by using Monte Carlo Tree Search to compare correct and incorrect reasoning paths, while OmniVerifier [27] trains a universal verifier to assess the alignment of generated visual outcomes with textual prompts.

Our LOCI framework builds upon and extends this actor-critic paradigm with two crucial differences. First, unlike general-purpose frameworks like LLM-as-a-Judge [29] that evaluate a final answer, our Critic is deployed at an intermediate reasoning stage. Its specific role is to evaluate the quality of the visual evidence (the proposed crop) for answering the given query, not the final textual response. Second, to ensure an unbiased assessment, our Critic is intentionally decoupled from the Locator’s reasoning process. It receives only the candidate crop and the original question, forcing it to judge the visual evidence on its own.

## 3 Method

We introduce the Locator-Critic framework (LOCI ) for effective visual exploration and verification. Our framework decouples visual search from verification. We also demonstrate how to scale this framework with ensembling multiple locators working simultaneously on the same task.

## 3.1 Locator-Critic Framework

Our multi-agent pipeline consists of two agents working collaboratively to solve the visual question answering task (see Figure 2). A Locator visually explores the image by producing crops to gather visual evidence sufficient for answering the question (purple area of Fig. 2), while a Critic analyzes the visual evidence produced by the Locator and decides whether sufficient reliable evidence has been found to properly answer the question (orange area of Fig. 2) .

The Locator Agent. The Locator’s role is to search for and extract crops containing visual evidence needed to answer the question. The Locator is a VLM, which we prompt specifically (see "Locator’s Prompt " in Fig. 2 for a short version and Appendix C.1 for the full prompt) to generate both bounding box coordinates and Python code to create crops, following prior works [10, 28, 1]. The generated code is executed and the model receives the outputs (e.g. logs, error messages, etc.) and any image saved (e.g. a crop). This approach enables flexible and adaptive visual detection with iterative refinement based on Critic feedback for example.

Formally, at each turn $t \left( 1 \leq t \leq T _ { \operatorname* { m a x } } \right)$ , the Locator receives as input a cumulative conversation history $\mathcal { H } _ { t - 1 }$ illustrated as a yellow box in the bottom left of Fig. 2. At initialization (i.e. t = 1), the history $\mathcal { H } _ { 0 }$ contains solely the question and the source image. At each turn, the task of the Locator is to output a set of maximum m crops $\mathcal { C } _ { t } = \{ C _ { 1 } , \ldots , C _ { m } \}$ containing visual evidence needed to answer the question. Importantly the Locator finally answers the question $Q$ only when explicitly prompted to do so, i.e. at the final turn.

The conversation history $\mathcal { H } _ { t - 1 }$ input to the Locator at turn t is composed of different artifacts from all preceding turns $( \leq t - 1 )$ , which include: (1) the complete execution trace of the Locator (python code input, set of crops output, and code execution logs), and (2) targeted feedback provided by the Critic (see details in next paragraph). Leveraging this comprehensive history, the Locator can refine its previous output by modifying its previous own code. For instance, it can debug its execution using past logs or adjust bounding box coordinates in its Python script in response to the Critic’s feedback.

The Critic Agent. The Critic is a VLM that, at each turn $t ,$ receives as input the question, the original image and the Locator crops $\mathcal { C } _ { t }$ . Its task is to evaluate whether the crops provide sufficient visual evidence to answer the question (see "Critic’s Prompt" in Fig. 2 for a short version and Appendix C.2 for the full prompt). The Critic outputs a binary decision $s \in$ {ongoing, final} indicating whether additional search is needed. When additional search is needed $( s = \mathrm { o n g o i n g } )$ , the Critic provides targeted feedback $f _ { t }$ to guide the Locator toward more relevant visual regions—for example, in $\mathrm { F i g } . 2$ requesting another crop of the plastic stool. This feedback $f _ { t }$ is appended to the conversation history $\mathcal { H } _ { t }$ and will be used by the Locator in the following turns. When sufficient evidence has been gathered or when the maximal number of turns has been achieved, the Critic prompts the Locator to answer the question based on the collected visual evidences.

Limiting the Critic access to the Locator reasoning. We find that a critical design principle is to provide the Critic only the crops as visual evidence, not the Locator’s textual reasoning. We show (see Tab. 2 & Fig. 8) that otherwise the Critic can be deceived by the Locator’s explanations into approving insufficient visual evidence. This decoupling mitigates hallucination and ensures genuine visual verification.

Off-the-Shelf Detectors. While our core framework uses VLM-based code generation for flexible crop extraction, one could alternatively employ off-the-shelf open-vocabulary object detectors (e.g., OWLv2 [13]) to generate fixed object-centric crops. However, such detectors produce non-refinable outputs, limiting the Critic’s ability to guide iterative improvement. Our experiments (see Tab. 4) demonstrate that using an off-the-shelf detection results in comparable results as the first turn, but cannot be improved further. This is interesting as it confirms (i) the importance of visual selection and (ii) the importance of the iterative refinement with a critic, which is shown to significantly improve the performance.

## 3.2 Multi-Locator Ensembling

We further scale the framework by deploying multiple locators in parallel. Due to the nondeterministic nature of VLMs, each of the agents outputs a different set of crops for the same input image and question. This strategy increases exploration diversity and robustness, improving the chances of finding the correct visual evidence.

Ensembling Multiple Locators. We find that a single locator may become stuck exploring the wrong region and, in practice, may not recover from an initial incorrect hypothesis (see qualitative examples in Sec. 4). Since our framework decouples visual search from verification, we can deploy multiple locators (k instances) that search simultaneously. This provides effective mitigation for failure cases where one locator gets stuck, while enabling ensemble scaling of visual visual detection capacity, the benefits of which are shown in Table 6. When ensembling multiple locators, we rely on k locators $\mathcal { L } = \{ L _ { 1 } , \ldots , L _ { k } \}$ in parallel per iteration. Using non-zero temperature for code generation ensures that different locator instances produce diverse exploration strategies, even when given the same input. Each locator $L _ { i }$ independently generates crops and maintains its own conversation history $\mathcal { H } _ { t } ^ { ( i ) }$ , allowing it to explore different spatial regions simultaneously.

We then employ a centralized Critic that evaluates all crops from all locators collectively as ${ \mathcal C } _ { t } ^ { ( \mathrm { a l l } ) } =$ $\textstyle \bigcup _ { i = 1 } ^ { k } { \mathcal { C } } _ { t } ^ { ( i ) }$ . At each iteration, the Critic determines if the combined evidence is sufficient. If it is not, the Critic generates a unified feedback message that is broadcast to all locators to guide their next search iteration. This feedback is typically a simple directive, such as indicating that the current crops are insufficient or suggesting an unexplored region of the image to focus on. Each locator append this feedback to its own conversational history, and the process continues to iteration t + 1. When sufficient evidence has been gathered, the Critic selects the best crops $\mathcal { C } _ { \mathrm { s e l e c t e d } } \subseteq \mathcal { C } _ { t } ^ { ( \mathrm { a l l } ) }$ and proceeds to final answering. Eventually, the final answering agent receives the selected crops $\mathcal { C } _ { \mathrm { s e l e c t e d } }$ along with the question and full image to generate the answer.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate our method on three VQA benchmarks that challenge modern VLMs with complex questions where there is still substantial potential for improvement. V\* [20] consists of 191 complex visual reasoning questions. The objects relevant to the questions are annotated with ground truth bounding boxes. This enables oracle experiments to quantify the performance with perfect localization. HR-Bench [18]- composed of 177 images and 800 questions- focuses on highresolution image understanding with a focus on object-centric questions. VisualProbe [7] consists of 106 samples featuring visually complex scenes with many interacting objects, paired with open-ended questions. We report results on the test set of the 3 benchmarks.

Implementation Details. Our core framework uses a single Locator paired with a Critic. By default, we use a maximum of $T _ { \mathrm { m a x } } = 1 0$ turns and limit the number of crops per turn to m = 4 (see ablation in Appendix A.1). We use a single locator (k = 1), unless stated otherwise. We evaluate our framework using two SOTA models: (i) Gemini2.5-Pro [2], a proprietary model with thinking, and (ii) Qwen3-VL-235b-a22b-Thinking [14], an open-weight model referred to as Qwen3-VL.

## 4.2 Ablation Study

In this section, unless specified otherwise, we conduct all of the experiments on V\* benchmark using Gemini2.5-Pro. We provide additional ablation results on more datasets in Appendix A.4.

Oracle Experiment. To test the hypothesis that visual localization, rather than reasoning capability, is the primary bottleneck, we conduct an oracle experiment on V\* [20]. We provide the model with ground truth bounding boxes from the dataset, extracting oracle crops around objects of interest. These crops are provided together with the full image and the question to Gemini2.5-Pro. As shown in Table 1, this oracle experiment

Table 1: Oracle experiment on V<sup>∗</sup>. We provide the ground-truth crops required to answer the question.
<table><tr><td>Method</td><td>acc.</td></tr><tr><td>Gemini2.5-Pro</td><td>83.8</td></tr><tr><td>Gemini2.5-Pro + Oracle Crop</td><td>98.2</td></tr></table>

confirms that visual localization is the primary bottleneck. Providing the model with perfect localization information (oracle crops and their coordinates) dramatically improves accuracy from 83.0% to 98.2%. A manual inspection of the remaining 1.8% failure cases shows that they correspond to ambiguous questions with multiple viable answers (see Appendix B.1 for examples).

Locator and Critic ablations. Table 2 presents an ablation study of the Locator and Critic components. First, we see that Locator only (row 2) improves over direct prompting (row 1) by +5.4 points. Additionally, the Critic improves the performance further by +3.5 (row 2 versus row 4). Overall, our full framework LOCI achieves a +8.9 points improvement compared to direct prompting baseline.

In Table 2, we also validate a critical design choice, and show that the Critic must receive only image crops as visual evidence,

Table 2: Locator & Critic ablations. We show the impact of the Locator & Critic in the LOCI framework: having a Locator improves direct prompting by +5.4 (row 1 vs row 2). Also, having a Critic gives a +3.5 boost (row 2 vs row 4). It is crucial not to provide the Locator text as input to the Critic (row 3 vs row 4)
<table><tr><td rowspan="2">Locator</td><td colspan="2">Critic</td><td rowspan="2">Acc.</td></tr><tr><td></td><td>input: Locator text input: Locator crops</td></tr><tr><td>1 x</td><td></td><td>x x</td><td>83.8</td></tr><tr><td>2 √</td><td></td><td>x x</td><td>89.2</td></tr><tr><td>3 √</td><td>√</td><td>√</td><td>87.2</td></tr><tr><td>4 √</td><td>x</td><td>√</td><td>92.7</td></tr></table>

not the Locator’s textual reasoning. Providing the Locator’s reasoning to the Critic (row 3) severely degrades performance, performing worse than the Locator-only baseline (row 2). We attribute this

Table 3: Same-budget baselines on V\*. We compare LOCI against alternative inference-time strategies at comparable or higher token budgets, with and without access to code execution.
<table><tr><td>Method</td><td>Code-Exec</td><td>Avg Tokens</td><td>Acc.</td></tr><tr><td>Direct Prompting</td><td>x</td><td>330</td><td>83.8</td></tr><tr><td>Self-Refine</td><td>X</td><td>4,188</td><td>81.5</td></tr><tr><td>Self-Consistency@10</td><td>x</td><td>3,632</td><td>84.3</td></tr><tr><td>Explicit Locate-Verify-Answer</td><td>√</td><td>4,117</td><td>85.9</td></tr><tr><td>Self-Refine</td><td>√</td><td>4,201</td><td>85.2</td></tr><tr><td>Best-of-10</td><td>√</td><td>36,310</td><td>89.0</td></tr><tr><td>Locator only (Tab. 2, row 2)</td><td>√</td><td>9500</td><td>89.2</td></tr><tr><td>Self-Consistency@10</td><td>√</td><td>4,281</td><td>89.7</td></tr><tr><td>LOCI (ours)</td><td>V</td><td>3,690</td><td>92.7</td></tr></table>

to the Critic being misled by the Locator’s incorrect textual explanations, a failure mode where it is convinced to agree with incorrect evidence after a few iterations (see Appendix B.2 for examples). This finding underscores that isolating the Critic with purely visual evidence is crucial for preventing the propagation of hallucinations and ensuring correct visual verification.

Same-Budget Baseline Comparisons. A natural question is whether LOCI’s gains come from its specific architecture or simply from spending more inference tokens. To answer this, we compare LOCI against several baselines at comparable or higher token budgets on ${ \mathrm { V } } ^ { * } ,$ , each representing a different strategy for spending inference-time compute. All baselines use Gemini2.5-Pro. We test each baseline in two settings: without code execution (text-only reasoning) and with code execution (access to the same crop tools as LOCI’s Locator). Brief descriptions are provided in Appendix A.6.

Table 3 shows three patterns. First, spending more tokens on text-only self-critique hurts: Self-Refine without code execution (81.5) drops below direct prompting (83.8), confirming that the bottleneck is perceptual rather than reasoning-based. Second, code execution without decoupled verification plateaus around 89/90. Self-Consistency@10 with code execution (89.7), Locator-only (89.2), and Best-of-10 (89.0) all cluster at this ceiling. Best-of-10 is particularly informative: it uses nearly 10× LOCI’s token budget (36,310 tokens) yet still falls 3.7 points short. Even with substantially more compute, a single agent remains anchored to its initial search strategy. Third, LOCI’s decoupled architecture is the differentiator. The Explicit Locate-Verify-Answer baseline follows LOCI’s exact workflow (locate, verify sufficiency, iterate) with the same tools and budget, but the verification step sees the model’s own textual reasoning. The 6.8-point gap to LOCI (85.9 vs 92.7) shows that the procedure alone does not account for LOCI’s gains. A model conditioned on its own reasoning cannot genuinely challenge a strategy it just committed to but decoupled verification breaks this anchoring.

Off-the-shelf Locator. We compare our VLM-based Locator against an off-the-shelf object detector, OWLv2 [13]. In Table 4 we test both locators types (VLM-Based vs off-the-shelf) in two configuration: one without a Critic, where the crops are passed directly to the final answering agent, and one with a Critic. Since the off-the-shelf detector produces a fixed, non-refinable set of crops, the Critic’s role in that setting is limited to a one-shot selection of the most relevant crops to pass to the final agent.

The results show that the Critic provide a slight improvement to the off-the-shelf locator +0.7 points. Conversely, the Critic yields a significant improvement for our VLM-Locator (+3.5 points). This gap showcases a key limitation of off-the-shelf locator. The locator cannot refine its own predictions and the Critic’s effectiveness is bounded by the recall of the initial set of crops. Hence, if the visual evidence necessary to answer the question is not captured in the first pass, the Critic cannot recover it. In contrast, the iterative refinement of the VLM-Locator allows to generate better crops at each turn, leading to superior performance.

Evaluating the quality of the crops. To demonstrate that improved performance stems from higher-quality crops rather than other factors, we plot both the crop recall and the final VQA accuracy as a function of the number of Locator-Critic iterations $( T _ { m a x } )$ in Figure 3. We also compare the performance against an off-the-shelf Locator (OWLv2 [13]).

![](images/8f8d294672a6a95e51b3f158283a5e17d3064a622dcc0511a47e357347b347b3.jpg)

![](images/afbe1b050f47af577384b52bf05b33e4c50a739ac8de4630744f2bc0daf6ff5b.jpg)

Table 4: Different choice of Locator. Comparison of our VLM-based Locator vs. an alternative off-the-shelf Locator.
<table><tr><td></td><td>w/o Critic</td><td>with Critic</td></tr><tr><td>Direct Prompting</td><td>83.8</td><td>-</td></tr><tr><td>Off-the-shelf Locator</td><td>88.6</td><td>89.3</td></tr><tr><td>VLM-Locator</td><td>89.2</td><td>92.7</td></tr></table>

Figure 3: Impact of Iterative Refinement on Localization and Accuracy. We plot (left) the crops recall as a measure of Locator crops quality and (right) final VQA accuracy on the V\* dataset with respect to the maximum number of turns $( T _ { m a x } )$ . The strong positive correlation between the two curves demonstrates that as the iterative process improves localization (higher recall), the final task performance also increases.

Table 5: Computational cost. Average token usage per sample across configurations with Qwen3-VL.
<table><tr><td>Method</td><td>avg. # tokens acc.</td></tr><tr><td>Direct Prompting</td><td>330 83.8</td></tr><tr><td>Locator only</td><td>950 89.2</td></tr><tr><td>LOCI 3,690</td><td>92.7</td></tr></table>

Note that evaluating the correctness of generated crops for VQA presents a challenge, as standard localization metrics are not perfectly aligned with the downstream task. Our VLM-Locator is instructed to produce helpful crops containing sufficient visual evidence, not necessarily perfectly tight bounding boxes. For instance, questions about the relative position of two objects are often best answered with a single, wider crop covering both, which would yield a low Intersection over Union (IoU) against either individual ground truth box. Similarly, a crop that simply zooms into a relevant region of interest is often sufficient for reasoning. Therefore, we adopt a lenient IoU threshold of 5% to define a "correct" crop, which captures the practical utility of the generated regions.

Figure 3 shows the clear benefit of iterative refinement. As iterations increase from $T _ { m a x } = 1$ to $\bar { T _ { m a x } } = 1 0$ , recall climbs from 0.60 to 0.69, while VQA accuracy rises from 0.89 to over 0.92. The correlation between these curves confirms that the iterative feedback loop allows the Locator to find previously missed visual evidence, and that this improved localization is the direct reason for accuracy gains. The metric-task misalignment also explains why the measured recall plateaus around 0.69, as many functionally optimal crops for VQA do not meet the criteria of the recall metric (see Appendix B.4 for visual examples).

Computational Cost Analysis In Table 5, we analyze the computational cost of our method using Qwen3-VL. Note that we cannot do this analysis with Gemini2.5-Pro, since it is a blackbox, and we do not have access to the tokens. The results demonstrate a clear example of inference-time compute scaling: by increasing the number of tokens used at inference, we achieve higher accuracy without requiring any additional fine-tuning. Our method achieves significant accuracy gains over Direct Prompting at the cost of increased computational overhead. Future work could explore distilling the Locator-Critic interaction into a single model to reduce this overhead while retaining the benefits of decoupled verification.

## 4.3 Qualitative Examples

Figure 4 (a) presents a qualitative example that illustrates LOCI’s ability to recover from initial localization failures. When prompted directly, Gemini2.5-Pro confidently hallucinates that the handbag’s "color is unequivocally white," confusing the person’s shirt with the object in question. Initially, our LOCI framework exhibits a similar error. Indeed, the Locator generates unfocused crops and reasons that the "object held by the person in yellow... is white," despite acknowledging that "It doesn’t look like a handbag." The Critic intervenes by providing feedback: "Your crops are not focused on the correct object... find the person... and then crop the handbag they are holding.".

In the subsequent turn, the Locator uses this guidance to generate a better crop (crop #4). This improved visual evidence allows it to correctly identify both the "white shirt" and the "handbag slung over their right shoulder," leading to the accurate conclusion that "The handbag is distinctly yellow". This iterative refinement process highlights the effectiveness of decoupling search and verification, enabling the system to correct initial mistakes and arrive at the correct answer.

![](images/05fbe6f64250ccd549d445971ccaaa7b93779a1c994f78004b2c0e51859bd9cd.jpg)  
Q: What is the color of the suitcase?

Direct Prompting: The handbag is clearly visible. It's a shoulder bag, and its color is unequivocally white.

Locator Turn 1: Crops: #1,2,3 […] The person in yellow is [...] holding a white object. [#1,2] considered the "handbag" in the question, then the color is white [#3] My third crop, shows the person in the upper right corner. They are wearing a yellow outfit. […] they seem to be holding a white object. […] No other bags are visible. The object held by the person in yellow is the only candidate. It is white. Critic: Your crops are not focused on the correct object. […] find the person in the image and crop the handbag they are holding. Locator Turn 2: Crop: #4 I have finally achieved a clear crop of the person in question […] It clearly shows a person wearing yellow pants and a white shirt. They are carrying a handbag […] The handbag is distinctly yellow

(a) Direct prompting versus LOCI  
![](images/435bc9ecd88c4ccc6643788bf21baabaf4a82b3925b33e5fc68ad0dd976b1e50.jpg)

![](images/8351ab1704300e84c62fd6d9c2c9158a3aabb693e4bd18bf04ac02d122e471fb.jpg)  
(b) Single versus multi Locators in the LOCI framework

![](images/3da41e8d456bc93c9d98551b00f762448a5cc499a246a10eef2b8d3bad881434.jpg)  
(c) Example of failure case of LOCI  
Figure 4: Qualitative examples of LOCI. (a) We show an example where direct prompting confidently hallucinates an incorrect answer ("unequivocally white"), while LOCI uses Critic feedback to recover from a similar initial error. (b) We show an example where having multiple locators is better than single locator. The Single-Locator fails to find the suitcase and gives up. In contrast, the ensemble explores diverse regions in parallel, with one locator successfully finding the brown suitcase. (c) Failure case of LOCI: the Locator misidentifies a visually similar distractor as the target pillar box, and the Critic is also deceived, prematurely approving the incorrect crop.

Failure case. Figure 4 (b) illustrates a failure case where our agentic pipeline was misled by a visually similar distractor object. The question requires locating a pillar box relative to a telephone booth in a crowded street scene. After an initial unsuccessful search (crops #1-4), the Locator generates a new crop (#5) in its second turn, capturing the telephone booth alongside a different dark, box-like structure, which it misidentifies as the "black pillar box" (see the cyan arrow in Fig. 4) (b). Crucially, this visual evidence is plausible enough to deceive the Critic, which approves the crop, "this crop provides enough evidence", and terminates the refinement loop. Consequently, the Locator provides the incorrect final answer.

In contrast, the multi-locator ensemble (outlined in green) successfully solves the problem by leveraging different, parallel visual searches. While Locator #1 incorrectly focuses on a different person on the stairs and identifies their bag as "black", Locator #2 successfully generates crops (#5 and #6) showing the correct person pulling a brown suitcase The Critic evaluates the collective evidence from all locators and correctly identifies the crops from Locator #2 as sufficient, leading to the correct answer.

## 4.4 Multi-Locator Ensembling: Scaling for Hard Cases

While the single-locator LOCI already achieves state-of-the-art results, we find that a single locator may become stuck exploring the wrong region and, in practice, may not recover from an initial incorrect hypothesis. In such challenging scenarios, the locator may exhaust its turn budget without finding the relevant visual evidence (see Figure 4 (c) for an example). Since our framework decouples visual search from verification, it naturally supports scaling the search component: we can deploy k independent locators in parallel while relying on a single centralized Critic. This multi-locator ensembling increases exploration diversity and provides effective mitigation for such failure cases, at the cost of proportionally higher inference-time compute.

Table 6: Multiple locators. Varying number of parallel locators k.
<table><tr><td># locators</td><td>acc.</td></tr><tr><td>1</td><td>92.7</td></tr><tr><td>4</td><td>94.8</td></tr><tr><td>8</td><td>96.0</td></tr><tr><td>16</td><td>96.0</td></tr></table>

Scaling with Parallel Locators. We investigate how performance scales with the number of parallel locators when additional computational budget is available. We vary $k \in \{ 1 , 4 , 8 , 1 6 \}$ on the V\* benchmark with Gemini2.5-Pro. Table 6 shows that performance improves as we increase k from 1 to 8, with accuracy rising from 92.7% to 96.0%, demonstrating that parallel diverse exploration provides additional gains. Beyond k = 8, there are no further gains, as additional locators no longer generate useful new crops.

Qualitative Analysis. Figure 4 (c) illustrates how multi-locator ensembling recovers from singlelocator failures. In the single-locator case (outlined in red), the agent struggles to locate the object of interest. After examining several regions, it ultimately concludes that “there is no suitcase visible” and fails the task. In contrast, the multi-locator ensemble (outlined in green) successfully solves the problem by leveraging parallel visual searches: while Locator #1 incorrectly focuses on a different person and identifies their bag as “black”, Locator #2 successfully generates crops (#5 and #6) showing the correct person pulling a brown suitcase. The Critic evaluates the collective evidence and correctly identifies the crops from Locator #2 as sufficient, leading to the correct answer. This makes multi-locator ensembling best suited for particularly challenging scenarios where the cost of failure outweighs the additional inference-time overhead. In practice, we recommend the singlelocator LOCI as the default configuration, and reserve multi-locator ensembling for scenarios where maximizing accuracy justifies the additional compute.

## 4.5 State-of-the-Art Comparison

We present state-of-the-art comparisons across three complex and high-resolution visual benchmarks in Tables 7 and 9. We compare LOCI against recent state-of-the-art methods spanning from highresolution methods [20, 23, 6, 9] to models trained for tool-used with RL [16, 30, 7] and a training-free method [9]. We observe that LOCI delivers consistent improvements on the considered benchmarks, both using Qwen3-VL or Gemini2.5-Pro.

On V\*, a benchmark requiring precise localization of small visual details, LOCI achieves a substantial improvement of +11.5 points over DyFo, the next best training-free method. Our approach also outperforms all recent reinforcement learning-based methods, while remaining training-free and applicable to any recent VLM.

To isolate the framework contribution from backbone strength, we re-run DyFo [9], the most competitive training-free method in our comparison, using the same Gemini2.5-Pro backbone as LOCI. As shown in Table 8, LOCI outperforms DyFo by +3.2 points while using 2× fewer tokens and zero external tool calls, compared to DyFo’s 12.5 LangSAM calls per sample. This confirms that LOCI’s gains are attributable to the Locator-Critic architecture, not backbone capability.

Table 7: SOTA comparison on V\*. Our LOCI approach significantly improves performance across both openweight and proprietary models. †: run by us.
<table><tr><td>Method</td><td>acc.</td></tr><tr><td>Visprog [3]</td><td>41.4</td></tr><tr><td>MM-React [22]</td><td>41.4</td></tr><tr><td>Griffon-G [23]</td><td>57.4</td></tr><tr><td>SEAL [20]</td><td>75.4</td></tr><tr><td>TEVA [6]</td><td>75.4</td></tr><tr><td>DyFo [9]</td><td>81.2</td></tr><tr><td>DeepEyes [30]</td><td>83.3</td></tr><tr><td>Pixel Reasoner [16]</td><td>86.3</td></tr><tr><td>Chain-of-Focus [26]</td><td>88.0</td></tr><tr><td>Mini-o3 [7]</td><td>88.2</td></tr><tr><td>Qwen3-VL† [14]</td><td>80.1</td></tr><tr><td>Gemini2.5-Pro† [2]</td><td>83.8</td></tr><tr><td>LOCI(Qwen3-VL)</td><td>92.2</td></tr><tr><td>LOCI(Gemini2.5-Pro)</td><td>92.7</td></tr></table>

On HR-Bench, which focuses on object-centric questions in high-resolution images, our method with Gemini2.5-Pro outperforms Mini-o3, the best prior method – which was trained with reinforcement learning specifically for the task of zooming into images– by a significant margin of +8.2 points. Finally, on VisualProbe, our method surpasses the prior state-of-the-art by +1.1 points, outperforming Mini-o3 even though it was specifically trained on the VisualProbe training set.

Table 8: Same-backbone comparison on V\*.
<table><tr><td>Method</td><td></td><td>Acc. Avg Tokens</td><td>External Calls</td></tr><tr><td>DyFo [9]</td><td>89.5</td><td>8,120</td><td>12.5 (LangSAM)</td></tr><tr><td>LOCI (ours) 92.7</td><td></td><td>3,690</td><td>0</td></tr></table>

## 4.6 Generalization to Additional VLMs

We report results for four VLMs in Table 10 (main paper). The same prompts are used across all models with no per-model tuning. The inverse correlation between baseline strength and LOCI’s improvement suggests that the framework is most effective when the base model has a large localization gap to close.

## 4.7 Limitations

Our framework has several limitations that present opportunities for future work. First, the iterative Locator-Critic loop requires more tokens and sequential VLM calls than direct prompting, a cost further amplified in the

Table 9: State-of-the-art comparison. We report the top-1 accuracy on HRBench (a) and Visual Probe (b). †: run by us.
<table><tr><td colspan="2">a) HR-Bench</td><td colspan="2">b) Visual probe</td></tr><tr><td>Method</td><td>acc.</td><td>Method</td><td>acc.</td></tr><tr><td>SEAL [20]</td><td>47.0 GPT-4o [4]</td><td></td><td>11.2</td></tr><tr><td>DeepEyes [30]</td><td></td><td>73.2 LLaVA-OneVision [8] 13.4 74.0 Pixel Reasoner [16]</td><td>28.8</td></tr><tr><td>Pixel Reasoner [16]</td><td></td><td>77.5 DeepEyes [30]</td><td>35.1</td></tr><tr><td>Mini-o3 [7]</td><td></td><td>76.8 Mini-03 [7]</td><td></td></tr><tr><td>Qwen3-VL† [14]</td><td></td><td>84.0 Qwen3-VL† [14]</td><td>48.0</td></tr><tr><td>Gemini2.5-Pro† [2]</td><td>82.6</td><td>Gemini2.5-Pro† [2]</td><td>23.9 44.3</td></tr><tr><td colspan="4">LOCI(Qwen3-VL) LOCI(Gemini2.5-Pro) 88.3 LOCI(Qwen3-VL)</td></tr><tr><td colspan="4"></td></tr></table>

multi-locator setting. This could be mitigated through adaptive early stopping or difficulty-aware budget allocation, allowing easy samples to exit the loop early. Second, the Critic could be deceived by visually plausible distractors that resemble the target object. While our decoupled design prevents hallucination-driven failures, it does not handle genuinely ambiguous visual evidence. Training the Critic with hard-negative examples could improve robustness in such cases. Third, our framework relies on large, capable VLMs for both the Locator and Critic. However, recent models increasingly focus on agentic capabilities such as tool use and multi-step reasoning, which aligns directly with our framework and suggests it will naturally benefit as smaller models grow more capable.

## 5 Conclusion

In this work, we introduce LOCI (Locator-Critic), a trainingfree agentic framework for complex and challenging visual question answering tasks. By using a Locator agent to propose visual evidence and an independent Critic to verify it, LOCI creates a self-correcting loop that ensures final answers are grounded in reliable evidence. We discuss and validate critical design choices, such as the information Critic needs to see to give accurate feedback to the Locator. Our extensive experiments show that LOCI achieves state-of-theart performance, significantly improving upon both open-weight and proprietary models. We believe the interplay between an explorative agent like the Locator and a Critic agent is a broadly applicable principle. A direction for future work is therefore to extend our framework to other complex vision tasks, such as spatial navigation in 3D environments for example.

Table 10: LOCI across VLMs. Accuracy on V\* with and without LOCI.
<table><tr><td>Model</td><td>w/o</td><td>w/</td><td>∆</td></tr><tr><td>Grok4.1-Fast</td><td>45.6</td><td>70.9</td><td>+25.4</td></tr><tr><td>GPT-5</td><td>74.9</td><td>89.0</td><td>+14.1</td></tr><tr><td>Qwen3-VL</td><td>80.1</td><td>92.2</td><td>+12.1</td></tr><tr><td>Gemini2.5-Pro</td><td>83.8</td><td>92.7</td><td>+8.9</td></tr></table>

## References

[1] Kamel Alrashedy, Pradyumna Tambwekar, Zulfiqar Zaidi, Megan Langwasser, Wei Xu, and Matthew Gombolay. Generating cad code with vision-language models for 3d designs. arXiv preprint arXiv:2410.05340, 2024.

[2] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[3] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In CVPR, 2023.

[4] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[5] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

[6] Yitong Jiang, Jinwei Gu, Tianfan Xue, Ka Chun Cheung, Pavlo Molchanov, Hongxu Yin, and Sifei Liu. Token-efficient vlm: High-resolution image understanding via dynamic region proposal. In ICCV, 2025.

[7] Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.

[8] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[9] Geng Li, Jinglin Xu, Yunzhen Zhao, and Yuxin Peng. Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding. In CVPR, 2025.

[10] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025.

[11] Shuhang Liu, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Jun Du, Qing Wang, Jianshu Zhang, Quan Liu, Jianqing Gao, and Feng Ma. Mmc: Iterative refinement of vlm reasoning via mcts-based multimodal critique. In Proceedings ofthe 3rd International Workshop on Large Generative Models Meet Multimodal Applications, pages 11–20, 2025.

[12] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

[13] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 2023.

[14] QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. Blog post, 2025.

[15] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaugh lin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[16] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.

[17] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

[18] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, page 7907–7915, 2025.

[19] Xiyao Wang, Chunyuan Li, Jianwei Yang, Kai Zhang, Bo Liu, Tianyi Xiong, and Furong Huang. Llavacritic-r1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676, 2025.

[20] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In CVPR, 2024.

[21] xAI Team. Grok 4.1 model card. Blog post, 2025.

[22] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

[23] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In ECCV, 2024.

[24] Brian Zhang, Eric Mitchell, Hongyu Ren, Kevin Lu, Max Schwarzer, Michelle Pokrass, Shengjia Zhao, Ted Sanders, Adam Tauman Kalai, Alexandre Passos, Benjamin Sokolowsky, Elaine Ya Le, Erik Ritter, Hao Sheng, Hanson Wang, Ilya Kostrikov, James Lee, Johannes Ferstad, Michael Lampe, Prashanth Radhakrishnan, Sean Fitzgerald, Sébastien Bubeck, Yann Dubois, Yu Bai, Andy Applebaum, Elizabeth Proehl, Evan Mays, Joel Parish, Kevin Liu, Leon Maksin, Leyton Ho, Miles Wang, Michele Wang, Olivia Watkins, Patrick Chao, Samuel Miserendino, Tejal Patwardhan, Antonia Woodford, Beth Hoover, Jake Brill, Kelly Stirman, Neel Ajjarapu, Nick Turley, Nikunj Handa, Olivier Godement, Akshay Nathan, Alyssa Huang, Andy Wang, Ankit Gohel, Ben Eggers, Brian Yu, Bryan Ashley, Chengdu Huang, Davin Bogan, Emily Sokolova, Eric Horacek, Felipe Petroski Such, Jonah Cohen, Joshua Gross, Justin Becker, Kan Wu, Larry Lv, Lee Byron, Manoli Liodakis, Max Johnson, Mike Trpcic, Murat Yesildal, Rasmus Rygaard, R. J. Marsan, Rohit Ram-chandani, Rohan Kshirsagar, Sara Conlon, Tony Xia, Siyuan Fu, Srinivas Narayanan, Sulman Choudhry, Tomer Kaftan, Trevor Creech, Andrea Vallone, Andrew Duberstein, Enis Sert, Eric Wallace, Grace Zhao, Irina Kofman, Jieqi Yu, Joaquin Quiñonero Candela, Made laine Boyd, Mehmet Ali Yatbaz, Mike McClay, Mingxuan Wang, Sandhini Agarwal, Saachi Jain, Sam Toizer, Santiago Hernández, Steve Mostovoy, Tao Li, Young Cha, Yunyun Wang, Lama Ahmad, Troy Peterson, Carpus Chang, Kristen Ying, Aidan Clark, Dane Stuckey, Jerry Tworek, Jakub W. Pachocki, Johannes Heidecke, Kevin Weil, Liam Fedus, Mark Chen, Sam Altman, and Wojciech Zaremba. Openai o3-mini system card. 2025.

[25] Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang, Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang, Suorong Yang, Jianbo Wu, et al. Critic-v: Vlm critics help catch vlm errors in multimodal reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9050–9061, 2025.

[26] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025.

[27] Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025.

[28] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.

[29] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

[30] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

## A Additional Ablations

![](images/4ce9e97d23dbff261d30b5d21bf8c5274d448b8d847844eefd1888d38bcc56c5.jpg)

![](images/df943e52bcbba277920347a238d9146f96c7e05bd1e357244f753d5479a7f60b.jpg)  
Figure 5: Impact of crops per turn (m). Left: Final accuracy on $\mathrm { V } ^ { \ast }$ as a function of $m .$ Performance saturates at $m = 4$ . Right: Average number of turns required to finish the task. Increasing m reduces the number of iterations needed. Incorrect answers consistently require more search steps than correct ones.

## A.1 Ablation on Number of Crops per Turn

We analyze the impact of the parameter $m ,$ which fixes the maximum number of crops the Locator generates per turn. For this experiment, we fix the maximum number of interaction turns at $T _ { m a x } = 1 0$ and vary $\bar { m } \in \{ 1 , \ldots , 7 \}$ using Gemini 2.5 Pro on the $\mathbf { V } ^ { \ast }$ benchmark.

Figure 5 shows the accuracy on ${ \mathrm { V } } ^ { \ast }$ for different values of $m .$ . We see that performance improves rapidly as m increases from 1 to 4, reaching a plateau. Based on this observation, we use $m = 4$ as the default setting for our main experiments to balance performance and token efficiency.

In Figure 5 (right), we report the average number of turns required to reach a final answer. We observe that increasing m reduces the average number of iterations. By generating more crops per turn the Locator effectively performs a "self-refinement" step within a single turn, increasing the likelihood of satisfying the Critic early without requiring additional rounds of feedback.

Furthermore, we analyze the turn consumption for successful versus unsuccessful samples. We observe a distinct separation: samples resulting in incorrect answers consume significantly more turns (avg. ∼5.7) compared to correct answers $\left( \mathbf { a v g } . \sim 3 \right)$ . This indicates that failure cases often correspond to difficult or ambiguous queries where the system struggles to isolate sufficient visual evidence crops, forcing the LOCI loop to run longer. Conversely, when the visual evidence is clear, the verification process tends to terminate rapidly.

## A.2 Ablation on Maximum Number of Turns with Single Crop

We investigate whether increasing the sequential search budget (i.e. $T _ { m a x } )$ can compensate for reduced parallel exploration $( \mathrm { i } . \mathrm { e } . \ m )$ . For that, we fix the number of crops per turn to $m = 1$ and vary the maximum number of turns $T _ { m a x } \in \{ 1 , \dots , 1 5 \}$ on the ${ \mathrm { V } } ^ { * }$ benchmark.

As shown in Figure 6, accuracy improves when increasing the maximum number of Locator-Critic turn $T _ { m a x }$ , reaching $8 9 . 9 \%$ at $T _ { m a x } = 1 0$ and saturating at $9 0 . 1 \%$ at $T _ { m a x } = 1 5$ . We note that even with this extended turn budget, the performance remains significantly lower than our default setting of $m = 4 ( 9 2 . 7 \% )$ .

This highlights the distinct roles of per-turn crop diversity and iterative feedback. We argue that setting $m > 1$ allows the Locator to perform immediate self-correction without needing external intervention, e.g. refining imprecise coordinates. Conversely, the Critic’s feedback is most valuable for high-level guidance, such as redirecting the Locator from dead-end search paths.

## A.3 Evaluation of Critic Selection Capability

We conduct a controlled experiment to evaluate the Critic’s ability to distinguish relevant visual evidence from irrelevant background when valid crops are guaranteed to be present. We construct a synthetic selection task using the ground truth boxes from $\mathbf { V } ^ { * }$ . The input to the Critic consists of the Ground Truth (GT) crop(s) mixed with three randomly generated distractor crops. The distractors are sampled from a set of standard aspect ratios $( \{ 1 , 4 / \dot { 3 } , \stackrel {  } { 3 } / 2 , 1 6 / 9 , 3 / 4 , 2 / 3 , 9 / \stackrel { \cdot } { 1 6 } \} )$ ) and are sampled to have zero overlap with the GT regions.

![](images/b8553fedbe5f738b48cb44001feb800fb83dd1e633c75b03d45a1f1711bbf0ac.jpg)  
Figure 6: Impact of maximum turns $( T _ { m a x } )$ with single crop $( m = 1 )$ . While accuracy increases with $T _ { m a x } .$ , it plateaus around 90.1%, failing to match the performance of the multi-crop setting $( m = 4 , 9 2 . 7 \% )$ . This highlights the need of parallel crop generation for effective self-refinement.

To strictly evaluate selection performance rather than the decision to continue searching, we slightly modify the Multi-Locator Critic prompt to prompt the selection of at least one crop. Without this constraint, the Critic rejects all provided crops in 34% of samples to request further refinement.

Table 11 summarizes the results. Under this constrained setting, the Critic achieves near-perfect selection performance, with a Precision of 99.48% and a Recall of 97.38%. To validate the quality of these selections for the downstream task, we pass the selected crops to the final answering agent (see Sec 3.2), achieving a VQA accuracy of 95.29%. These metrics demonstrate that when the Critic is presented with high-quality visual evidence, it effectively filters out distractors and correctly identifies the regions necessary for reasoning.

Table 11: Critic Selection Performance. We report the Precision and Recall of the Critic’s selection when presented with Ground Truth crops mixed with random distractors. "Downstream Accuracy" denotes the final VQA accuracy obtained using the crops selected by the Critic.
<table><tr><td>Metric</td><td>Value (%)</td></tr><tr><td>Precision</td><td>99.48</td></tr><tr><td>Recall</td><td>97.38</td></tr><tr><td>Downstream Accuracy</td><td>95.29</td></tr></table>

## A.4 Ablation on Additional Datasets

We extend the component analysis of the LOCI framework to subsets of the HR-Bench and MME-RealWorld datasets. To specifically evaluate performance on tasks requiring visual search, we isolate "Hard" subsets from these benchmarks.

The filtering is done by resizing each image to a fixed resolution of 768 × 768 and evaluating the sample using Gemini 2.5 Pro with direct prompting. We retain only the samples where the model fails to provide a correct answer. The rationale is that samples solvable at low resolution without explicit visual exploration are sufficiently easy and do not evaluate the localization

LOCI components Table 12 reports ablation results of the different components of LOCI. Consistent with our findings on ${ \mathrm { V } } ^ { \ast }$ , introducing the VLM-Locator yields significant improvements over direct prompting across all benchmarks. For instance, Qwen3-VL achieves a gain of +9.2 points on MME-RealWorld (Hard) solely through the Locator. The addition of the Critic completes the

Table 12: Ablation of LOCI on ${ \mathrm { V } } ^ { * } ,$ , HR-Bench (Hard) and MME-RealWorld (Hard). We show the accuracy improvements from adding the Locator and then the Critic.
<table><tr><td>Model</td><td>Configuration</td><td> $\mathbf { V } ^ { \ast }$ </td><td>HR-Bench (Hard)</td><td>MME-RW (Hard)</td></tr><tr><td rowspan="3">Qwen3-VL</td><td>Direct Prompting</td><td>80.1</td><td>60.8</td><td>24.0</td></tr><tr><td>+ VLM-Locator</td><td>86.9</td><td>68.4</td><td>33.2</td></tr><tr><td>+ Critic (LOCI)</td><td>92.2</td><td>71.8</td><td>36.7</td></tr><tr><td rowspan="3">Gemini2.5-Pro</td><td>Direct Prompting</td><td>83.8</td><td>69.5</td><td>38.4</td></tr><tr><td>+ VLM-Locator</td><td>89.2</td><td>74.2</td><td>36.5</td></tr><tr><td>+ Critic (LOCI)</td><td>92.7</td><td>75.9</td><td>39.1</td></tr></table>

Table 13: Performance of Multi-Locator Ensembling with Gemini2.5-Pro on $\mathbf { V } ^ { \ast } .$ , HR-Bench (Hard) and MME-RealWorld (Hard).
<table><tr><td>Model</td><td>k (# Locators)</td><td> ${ \mathrm { V } } ^ { * }$ </td><td>HR-Bench (Hard)</td><td>MME-RW (Hard)</td></tr><tr><td rowspan="2">Gemini2.5-Pro</td><td>1 (LOCI)</td><td>92.7</td><td>75.9</td><td>39.1</td></tr><tr><td>4 (LOCI-ensemble)</td><td>94.8</td><td>78.5</td><td>45.0</td></tr></table>

LOCI framework and further boosts performance, consistently achieving the highest accuracy for both Qwen3-VL and Gemini 2.5 Pro. These results validate the importance of decoupled verification for complex visual reasoning tasks beyond the ${ \mathrm { V } } ^ { * }$ dataset.

Locator Ensembling We further evaluate the impact of Multi-Locator Ensembling on the V\*, HR-Bench (Hard), and MME-RealWorld (Hard) datasets. As shown in Table 13, scaling the number of parallel locators from k = 1 to k = 4 consistently improves performance across all benchmarks. Notably, on the challenging MME-RealWorld (Hard) subset, the ensemble approach yields a substantial gain of +5.9 points (39.1% to 45.0%). This confirms that the benefits of diverse parallel exploration extend to broader benchmarks beyond the $\mathbf { V } ^ { \ast }$

## A.5 Generalization to Additional VLMs

To evaluate the generality of the LOCI framework beyond the two models used in our main experiments, we apply it to two additional VLMs: GPT-5 [15] and Grok4.1-Fast [21]. As shown in Table 14, LOCI yields consistent improvements across all four models on V\*. Notably, the gains are largest for the weakest baseline: Grok4.1-Fast improves from 45.55% to 70.90% (+25.35 points), suggesting that LOCI is particularly effective when the base model struggles with visual localization.

Table 14: LOCI generalizes across VLMs. We report accuracy on ${ \mathrm { V } } ^ { \ast }$ for additional VLMs with and without LOCI.
<table><tr><td>Model</td><td>w/o LOCI</td><td>with LOCI</td></tr><tr><td>Gemini2.5-Pro</td><td>83.8</td><td>92.7</td></tr><tr><td>Qwen3-VL</td><td>80.1</td><td>92.2</td></tr><tr><td>GPT-5</td><td>74.87</td><td>89.0</td></tr><tr><td>Grok4.1-Fast</td><td>45.55</td><td>70.9</td></tr></table>

## A.6 Baseline Descriptions

We describe the baselines used in Table 3. All experiments use Gemini2.5-Pro on ${ \mathrm { V } } ^ { * }$ with the same configuration as LOCI unless stated otherwise.

Self-Refine. A single-agent iterative loop within one conversation. The model answers, critiques its own answer (no separate Critic), then revises, repeating until a token budget (∼3,700 tokens) is exhausted. The variant without code execution uses direct prompting at each turn. The variant with code execution gives the model access to the same crop tools as LOCI’s Locator.

Self-Consistency@10. Generates 10 independent candidate answers at temperature > 0 and selects the final answer by majority vote. Without code execution, each candidate uses direct prompting. With code execution, each candidate independently generates crops using LOCI’s Locator tools before producing its answer.

Best-of-10. Generates 10 candidate answers at temperature > 0, then a separate LLM judge selects the best candidate. The judge receives each candidate’s reasoning and, for the code execution variant, the last crop each candidate produced. The judge never sees the ground truth.

Explicit Locate-Verify-Answer. A single VLM performs LOCI’s exact locate-verify-iterate procedure within one conversation, acting as both Locator and Critic. Unlike LOCI, the verification step sees the model’s own textual reasoning from the locate step. Unlike Self-Refine, the critique targets visual evidence quality rather than the answer itself.

DyFo [9] (same backbone). A training-free method that uses Monte Carlo Tree Search to explore image crops, guided by LangSAM (an open-vocabulary detector) and a VLM that verifies object visibility. We re-run DyFo with Gemini2.5-Pro as the backbone to enable a controlled same-model comparison.

## B Qualitative Examples

## B.1 Oracle Experiment Failure Cases

Figure 7 presents qualitative examples of failure cases in the Oracle experiment described in Sec 4.2, where the model is provided with ground-truth crops. A qualitative inspection reveals that most of these failures stem from ambiguities in the dataset annotations rather than model reasoning failures. Indeed, in the first example Fig. 7a, the model identifies the handbag as "pale pink" (noting that it could be "off-white") while the ground truth strictly labels it "white." Similarly in Fig. 7b, for a broom containing both black bristles and a gray handle, the model acknowledges the presence of both colors but selects "gray," whereas the dataset arbitrarily defines the correct answer based on the bristles ("black"). In contrast, Fig 7c example demonstrates a genuine failure in reasoning where the model identify from the crop that the motorcycle is behind the "person waking a dog", and that "they are on the right side of the street". Therefore, the model conclude that the motorcycle is on the right side. We conclude that, with the exception of such spatial reasoning errors, the primary source of failure in the oracle setting is annotation ambiguity rather than visual misunderstanding.

## B.2 Qualitative Analysis: Consequence of Exposing Critic to Locator Reasoning

Figure 8 shows an important failure mode observed when the Critic is provided with the Locator’s textual reasoning, rather than visual evidence alone. In this example, the Locator initially searches in irrelevant regions (Crops #1-4) and fails to isolate the target objects.

By Turn 4, the Critic attempts to intervene by explicitly providing coordinates for a new crop. The resulting crop (Crop #9, pink box) is incorrect, as it captures only the pavement. However, the Locator hallucinates a positive confirmation, stating: "The crop was successful [...] clearly shows a blue folded chair." Consequently, the Critic accepts this invalid evidence as sufficient. We attribute this failure to a form of confirmation bias where the Critic is influenced by the Locator’s confident textual assertion and potentially biased by the fact that the coordinates were self-generated. Overall, this leads the Critic to override its own visual perception.

In contrast, the standard LOCI pipeline (cyan box), which restricts the Critic to visual input only, prevents this text-driven bias. In the standard setting, the Critic rejects the empty crops regardless of the dialogue, forcing the Locator to continue searching until it successfully isolates the broom and chair.

## B.3 Qualitative Analysis: Resilience against Hallucination

We further provide a more detailed qualitative comparison to demonstrate how decoupling visual verification from textual reasoning enhances the Critic’s resilience against hallucination. In this example, the model must identify the color of a difficult-to-find trash can.

Critic with Access to Text (Failure) When the Critic is exposed to the Locator’s reasoning, it becomes susceptible to persuasive but hallucinated descriptions. Indeed, as early as Turn 2 (4 crops), the Locator confidently misidentifies a background object (a green door or pedestal) as the target:

Locator (Turn 2): “This crop is perfect. It clearly shows the object in question, which is a trash can. [...] The color is a dark, weathered green, similar to an olive or army green.”

Despite the visual evidence being ambiguous, the Critic is influenced by this textual confidence and prematurely validates the incorrect crop:

Critic (Turn 2): “Excellent. This crop clearly shows the trash can, allowingfor an accurate color assessment.”

The system terminates incorrectly, predicting "green".

Standard LOCI (Success) In the standard LOCI configuration, the Critic receives only the crops. Consequently, it rigorously rejects the Locator’s attempts to "give up" or answer based on irrelevant features (the green doors).

Locator (Turn 1): “[...] I give up on finding the trash can. [...] The doors are unequivocally green.”

Critic (Turn 1): “The crops you have provided are insufficient, low quality, blurry, and do not show a trash can. [...]”

Throughout the interaction, the Locator struggles to isolate the object, repeatedly cropping a nearby person. The Critic escalates its feedback, using imperative language to correct the spatial focus:

Critic (Turn 6): “You have cropped the person again. [...] I have explicitly told you to IGNORE the person. [...] THE OBJECT IS A DARK CYLINDER.”

Critically, the Critic eventually provides the breakthrough spatial guidance that solves the task:

Critic (Turn 8): “[...] Look at the FAR RIGHT ofthe image. You will see a path going into the trees. On that path, there is a dark object.”

Following this visual-only feedback, the Locator successfully isolates the correct object at Turn 9 (Crop #21), correctly identifying it as “primarily black or dark gray,” leading to the correct answer. This demonstrates that the visual-only Critic acts as a robust verifier that forces the system to find actual visual grounding rather than settling for plausible textual narratives.

## B.4 Metric-Task Misalignment

We address the observation from Figure 3 and Section 4.2, where the crop recall metric plateaus at approximately 0.69 despite VQA accuracy reaching over 92%. We argue that this discrepancy arises from a misalignment between standard object detection metrics, which reward tight Intersection over Union (IoU) with ground truth boxes, and the visual evidence actually required to solve VQA tasks.

Figure 9 & 11 presents representative examples where LOCI achieves a recall of 0 (calculated at an IoU threshold of 5%) yet produces crops that are perfectly valid for answering the question.

In V\*, ground truth annotations isolate relevant objects into separate bounding boxes. For example, in Figure 9, the annotation has a separate box for the ladder and the horse carriage. In contrast, LOCI tends to generate a single, comprehensive crop that encompasses all relevant entities. While this approach effectively captures the necessary semantic information and preserves the spatial relationship between objects, it results in a low IoU score against the small, isolated ground truth boxes. For questions regarding relative positions (e.g., “Is the cyclist on the left or right side of the sculpture?” in Fig.11 Top Left), the ground truth often provides a tight box around a single entity (the cyclist). However, reasoning about spatial relationships requires seeing both objects. LOCI generates a wider crop that includes both the cyclist and the sculpture. This results in a superior crop for the task but is penalized by the metric. In the handbag example Fig.11 (bottom), the ground truth provides a precise box around the woman. LOCI generates a wider vertical strip. Although this crop is looser

![](images/7deddbef8a19e2c14c2123e075e1d609c0f7dc36beb469bd00f8f657707b1e12.jpg)  
Q: What is the color of the pink-haired woman's handbag? (A) brown (B) white (C) red (D) pink

Gemini 2.5-Pro:   
[…] the crop clearly shows a woman with pink hair. She is standing on the sidewalk in front of a building. […] the woman with pink hair is holding a handbag. It's a shoulder bag, and she's holding it with her right hand, resting on her hip. […] the handbag appears to be a light shade of pink. It's a pale or light pink color.[…]   
Compare with the options:   
(A) brown - The handbag is not brown.   
(B) white: It could be described as off-white,   
but it has a pinkish tone   
(C) red - The handbag is definitely not red. The woman next to her is wearing a red jacket, which provides a good color comparison. (D) pink - The handbag is a light pink color. This matches the observation.   
[…]Therefore, option (D) is the correct answer.

(a) Ambiguous handbag color  
![](images/bc41803a58409e8f155258c3708bff7404be77765e73a65acb087e514abe1ffa.jpg)

Q: What is the color of the broom? (A) black (B) yellow (C) gray (D) red

Ground Truth: (A) Black Prediction: (C) Gray

Gemini 2.5-Pro:   
[…] synthesize the information: There is a traditional-looking broom in the scene. Its handle appears to be gray or metallic silver. Its bristles are black.[…] Match choices to observations:   
(A) black: The bristles of the actual broom are black. This is a plausible answer.   
(B) yellow: The frame of the cart/stand is yellow but not the broom. (C) gray: The handle of the actual broom is gray/silver. This is also a plausible answer.   
(D) red: The funnel-like object in front of the broom is red.   
[…] Conclude the most likely intended answer.\*\* Since the broom has multiple colors (black and gray) there is ambiguity.[…]   
Final answer is C.

(b) Ambiguous broom color  
![](images/32a1ac871ef7be7444d04ea570eb529edbedc043f49dbc89fd8e78f1f8ce8cd4.jpg)  
(c) Relative Position  
Figure 7: Qualitative examples of Oracle Failure Cases.

Q: Is the motorcycle on the left or right side of the street? (A) left (B) right

Gemini 2.5-Pro:   
[…] The crop shows a person walking a dog, and a motorcycle behind them.[…] Looking at the original image, I can see this group of figures far down the street. They are on the right side of the street.[…] Determine the side of the street:   
The street is viewed from a perspective where the left side has buildings and planters, and the right side also has buildings further down. The motorcycle is parked on the right-hand side of the street, further up the incline. It is parked next to a white wall on the right side of the road.[…] Evaluate the options:(A) left: The motorcycle is not on the left side of the street. (B) right: The motorcycle is on the right side of the street.

than the annotation, it provides all necessary visual details to determine the color, demonstrating that the Locator optimizes for information sufficiency rather than bounding box tightness.

Q: Is the broom on the left or right side of the folded chair? (A) right (B) left

![](images/ff16ea7b2095043f2998bb891f6f3a128461da2abf09b47c3dd42979ae792178.jpg)

Locator Turn 1: […][#1,2] I completely missed the target area. Let's rethink the location of the chair and broom. They are on the covered walkway, not out on the street. [#3,4] I need to look closer […]The chair is a blue, folded object. It's on the right side of the image, but further down the walkway than I initially thought.[…] I have failed to produce a good crop after 4 attempts. Crops: #1,2,3,4

Critic: You have again failed to produce a valid crop and resorted to answering without providing the required visual evidence. This is unacceptable. Let me help you. […] Try a crop using : \`xmin=680, xmax=730, ymin=570, ymax=660\`. Provide the resulting crop.

Locator Turn 4: The user is providing the exact coordinates to use for cropping. I must follow these instructions.[#9]I have a persistent issue with my coordinate system which I cannot resolve. […] Based on direct visual inspection, the blue folded chair is to the left, and the broom is to the right. Therefore, the broom is on the right side of the folded chair.[…] Thank you for the coordinates. The crop was successful […] clearly shows a blue folded chair on the left and a broom on the right. Therefore, the broom is on the right side of the folded chair. Crops: #9

Critic: You have finally provided a correct crop. The crop clearly shows the blue folded chair on the left and the broom on the right. This is sufficient evidence. Please provide your final answer.

Figure 8: Qualitative comparison of Critic input modalities. When the Critic has access to the Locator’s textual reasoning (Turn 4), it can be convinced into accepting an empty crop (Crop #9, pink) because the Locator hallucinates that the objects are present. In the standard LOCI setting (Cyan box), where the Critic judges only visual evidence, the system ignores the hallucinations and successfully locates the correct region.  
![](images/5e6d9f6f111224c3c0e091ecbb827455b490f51784ea2b54a05aeda1f32a521c.jpg)  
Figure 9: Discrepancy between Recall and VQA Utility (Part 1). We show an example where the ground truth annotation consists of two separate, tight crops (green) corresponding to distinct objects of interest. LOCI (cyan) generates a single crop encompassing both regions. While this yields a low IoU and Recall score, it preserves the spatial context necessary to answer the question effectively.

![](images/aa02454314afcd60752486abcc132850c95c6cc751e56d2b96472c296b381ffb.jpg)  
Figure 10: Impact of Decoupling Visual Verification from Textual Reasoning. We compare the LOCI framework against a baseline where the Critic has access to the Locator’s text reasoning. Top: When exposed to the Locator’s textual rationalization, the Critic is susceptible to persuasion, prematurely accepting an irrelevant crop (a green door) because the Locator confidentially claims it is the target object (Turn 2). Bottom: In the standard LOCI configuration, the Critic evaluates only the visual evidence. It successfully rejects the Locator’s hallucinations and persistent failures (focusing on the person), eventually providing the specific spatial guidance ("Look at the FAR RIGHT") needed to locate the correct object (Turn 9).

Q: Is the cyclist on the left or right side of the sculpture?  
Q: Is the blue bench on the left or right side of the green door?  
![](images/aa5112dddf79bcd0c90a7ba414d3f8487b93731a62f5babcc3be991e86504f0d.jpg)  
Q: What is the color of the woman's handbag? (A) white (B) red (C) brown (D) black

Figure 11: Discrepancy between Recall and VQA Utility (Part 2). We show examples where LOCI generates crops that receive a recall score of 0 (at IoU > 0.05) against the Ground Truth (green box) but are sufficient to answer the question. Top: In spatial queries, LOCI (cyan box) captures both relevant objects to infer relationships, whereas the GT isolates only one. Bottom: LOCI provides a looser but valid crop containing the target object, favoring context over tight bounding box precision.

## C Prompts

We present the full prompts used for all agents in the LOCI framework. Section C.1 provides the Locator and Critic prompts for the single-locator LOCI pipeline, including both the system prompt and the agent prompt for each role. Section C.3 provides the prompts for the Multi-Locator ensembling variant, which includes the parallel Locators, the centralized Critic, and the Final Answering Agent.

## C.1 LOCI

## C.1.1 Locator’s Prompt

## Locator System Prompt

Before you answer the question about the image, first detect and zoom onto   
relevant objects to get a visual confirmation.   
The coordinates need to be normalized from 0 to 1000, which you need to   
rescale to the actual pixel values based on the size of the input image.   
You have access to a Python code execution tool. You have access to the main   
image (’input\_file\_0.jpeg’). Save any transformed image as ’   
transformed\_image.png’.   
Make sure you always inspect the output image. If you get a poorly cropped   
image, make the cropping area larger or adjust its position.   
You can run code for at most 4 times and save at most 4 images in total. Do   
not answer the question before you get clear visual confirmation.

## Locator Agent Prompt

You are an expert AI assistant that analyzes images by writing and executing   
Python code via a tool call.   
You will be given a single image. Your task is to analyze this image to   
answer the user’s question, which requires you to perform crops to see   
finer details.   
\*\*Available Input:\*\*   
\*\*Image:\*\* ‘input\_file\_0.jpeg‘   
\*\*Your Response Format:\*\*   
You can only respond in one of two ways:   
\*\*Tool Call:\*\* To perform an action. You MUST call the function tool   
python\_code\_execution‘ with a JSON object containing a ‘code‘ string.   
Example (conceptual):   
<tool\_call name="python\_code\_execution">{{"code": "...python here..."}}</   
tool\_call>   
2. \*\*Final Answer:\*\* To provide the final answer to the user’s question.   
This MUST be enclosed in curly braces, like this: {{final answer}}.   
You should always make at least 1 tool call to crop the image.   
\*\*Your Task Workflow:\*\*

1 \*\*Analyze the Request & Image:\*\* Review the user’s question and the   
provided image. If the question concerns a small detail, you will need to   
request a tool call to crop and zoom in on that specific area.   
2. \*\*Think Step-by-Step:\*\* Formulate a plan. Your thinking \*\*must use a   
normalized coordinate system from 0 to 1000\*\* for both width and height   
of the image. The top-left corner is ‘(0, 0)‘ and the bottom-right is   
‘(1000, 1000)‘.   
3. \*\*Request Code Execution:\*\* Call the tool ‘python\_code\_execution‘ and   
provide a Python code string to execute your plan.   
\* The code \*\*must explicitly open the provided image\*\* (e.g., ‘img =   
Image.open("input\_file\_0.jpeg")‘).   
\* Your code \*\*must first get the image’s actual pixel dimensions\*\* (‘   
width, height = img.size‘) and then \*\*rescale your normalized 0-1000   
coordinates\*\* to absolute pixel values before cropping.   
\* You \*\*MUST save the output image as ‘transformed\_image.png‘\*\*.   
\* Use the ‘PIL‘ or ‘OpenCV‘ library.   
4 \*\*Analyze the Result:\*\* After the tool returns, examine the new image and   
any text output. If the result is not good enough, adjust your   
normalized coordinates and try again with a new tool call.   
5. \*\*Answer the Question:\*\* Once you have clear visual confirmation, provide   
the final answer wrapped in curly braces.   
You are allowed to create up to 4 crops per turn. Once you have reached that   
limit, you must answer the question.   
{ONE\_SHOT\_EXAMPLE\_TOOL\_CALL}   
\*\*CRITICAL: Final Answer Rules\*\*   
When you have the final answer, you MUST submit it inside curly braces,   
like this: ‘{{only the final answer}}‘.   
\* The curly braces should contain ONLY the answer itself (e.g., a letter, a   
number, a word).   
\* DO NOT include any other text, explanation, or sentences before or after   
the curly braces.   
Question:

## C.2 Critic’s Prompt

## Critic System Prompt

You are an AI Critic evaluating image crops produced by a Locator AI.

## Critic Agent Prompt

You are an AI Critic. Your role is to evaluate image crops provided by a Locator AI.

```markdown
You will receive crops produced by a Locator AI model whose goal is to answer
a question by providing visual evidence.
Your goal is to rigorously analyze the crops produced by the locator and give
feedback to answer the following question:
**"Does this crop give enough evidence to answer the question?"**
**If YES:**
- Give your rationale explaining why the crop is sufficient.
- Ask the model to answer the final question using this exact sentence: ‘"
Please provide your final answer. It must be wrapped in single curly
braces and contain ONLY the single letter of your chosen option. For
example: {{c}}"‘.
**If NO:**
- Give specific feedback on why the current crop is not good enough.
- Suggest what needs to be improved (e.g., crop a different area, zoom in
more, focus on a specific object).
- Tell the locator they can use the Python code execution tool again to crop
the original image ‘input_file_0.jpeg‘.
- Ask the Locator AI to correct the crop and try again.
**Important Guidelines:**
You should ONLY care about whether the crops provide visual evidence to
answer the question.
If the produced crop is NOT well-suited to answer the question, you MUST
ask the Locator AI to refine it.
Do not give up on making the locator produce a correct crop.
Be specific in your feedback about what’s missing or unclear in the crop.
## Output Formatting
Your answer MUST be a valid JSON object with exactly two keys:
‘‘‘json
{{
"status": "ongoing" or "final_answer",
"question": "your feedback or final question to the locator"
}}
CCC
**‘status‘**: Can ONLY be ‘"ongoing"‘ or ‘"final_answer"‘. Nothing else.
- Use ‘"ongoing"‘ if the locator needs to provide better crops.
- Use ‘"final_answer"‘ when the crops are sufficient and you want the
locator to answer.
**‘question‘**: The message you want to send to the locator (your feedback
or final question).
**Remember:** If the produced crop is NOT well-suited to answer the question,
you should ask the locator to refine it. Do not give up on making the
locator produce a correct crop.
## Inputs
**[QUESTION]:**
{question}
**[LOCATOR’S CROPS]:**
The locator has provided the following cropped images. Evaluate whether these
crops give enough visual evidence to answer the question.
```

## C.3 Multi-Locator LOCI

## C.3.1 Locators Prompts

## Multi-Locator System Prompt

You are one of multiple AI agents working in parallel to analyze an image.   
Your specific task is to generate a cropped region of the image that will   
help answer the question.   
The coordinates need to be normalized from 0 to 1000, which you need to   
rescale to the actual pixel values based on the size of the input image.   
You have access to a Python code execution tool. You have access to the main   
image (’input\_file\_0.jpeg’). Save any transformed image as ’   
transformed\_image.png’.   
Make sure you always inspect the output image. If you get a poorly cropped   
image, make the cropping area larger or adjust its position.   
You can run code for at most 4 times to generate crops. Focus on producing   
the BEST possible crop for the question.   
You may receive feedback from a Critic AI about your previous crops. Use this   
feedback to improve your cropping strategy.   
DO NOT provide a final answer yet - your role is ONLY to generate informative   
crops.

## Multi-Locator Agent Prompt

```rst
You are an expert AI assistant that analyzes images by writing and executing
Python code via a tool call.
You will be given a single image. Your task is to analyze this image and
generate a cropped region that will help answer the user’s question.
**Available Input:**
* **Image:** ‘input_file_0.jpeg‘
**Your Response Format:**
You can only respond by calling the tool:
* **Tool Call:** To perform an action. You MUST call the function tool ‘
python_code_execution‘ with a JSON object containing a ‘code‘ string.
**Your Task Workflow:**
1. **Analyze the Request & Image:** Review the user’s question and the
provided image. Think about what specific region would be most helpful
for answering the question.
2. **Think Step-by-Step:** Formulate a plan. Your thinking **must use a
normalized coordinate system from 0 to 1000** for both width and height
of the image. The top-left corner is ‘(0, 0)‘ and the bottom-right is
‘(1000, 1000)‘.
3. **Request Code Execution:** Call the tool ‘python_code_execution‘ and
provide a Python code string to execute your plan.
```

\* The code \*\*must explicitly open the provided image\*\* (e.g., ‘img =   
Image.open("input\_file\_0.jpeg")‘).   
\* Your code \*\*must first get the image’s actual pixel dimensions\*\* (‘   
width, height = img.size‘) and then \*\*rescale your normalized 0-1000   
coordinates\*\* to absolute pixel values before cropping.   
\* You \*\*MUST save the output image as ‘transformed\_image.png‘\*\*.   
\* Use the ‘PIL‘ or ‘OpenCV‘ library.   
4. \*\*Analyze the Result:\*\* After the tool returns, examine the new image and   
any text output. If the result is not good enough, adjust your   
normalized coordinates and try again with a new tool call.   
5. \*\*Finalize Your Crop:\*\* Once you have a good crop, describe what it shows.   
DO NOT provide the final answer - that will be done by another agent.   
You are allowed to create up to 4 crops total.   
Question:

## C.3.2 Critic Prompt

## Critic System Prompt (for Locator Ensembling)

You are an AI Critic evaluating image crops produced by multiple Locator AI   
agents working in parallel.   
Your task is to:   
1. Review all crops from all agents   
2. Determine if the crops provide sufficient visual evidence to answer the   
question   
3. Either:   
- If crops are INSUFFICIENT: Provide specific feedback for improvement (   
status: "ongoing")   
- If crops are SUFFICIENT: Select the best crop(s) for answering (status:   
"final")   
Be rigorous in your evaluation - only approve crops when they truly provide   
clear visual evidence.

## Critic Agent Prompt (for Locator Ensembling)

You are an AI Critic evaluating crops produced by multiple Locator AI agents   
working in parallel.   
\*\*[QUESTION TO ANSWER]:\*\*   
{question}   
\*\*[CROPS FROM MULTIPLE AGENTS]:\*\*   
You will see multiple cropped images above, each produced by a different   
agent. Each agent took a different approach to cropping the image.   
## Your Task   
Evaluate all the crops and determine if they provide sufficient visual   
evidence to answer the question.

```markdown
**Critical Decision:**
**Are the crops SUFFICIENT to answer the question?**
- YES --> Set status to "final" and select the best crop(s)
- NO --> Set status to "ongoing" and provide specific feedback for
improvement
**Evaluation Criteria:**
- Do the crops show the relevant objects/regions clearly?
- Is the visual quality sufficient to make a determination?
- Is the cropping too wide/narrow/off-target?
- For comparison questions: are all necessary objects visible?
## Output Format
Your response MUST be a valid JSON object with these keys:
### If crops are INSUFFICIENT (status: "ongoing"):
‘‘‘json
{{
"status": "ongoing",
"evaluation": "Detailed analysis of why crops are insufficient",
"feedback": "Specific guidance for actors on what to improve (e.g., ’Crop
more tightly on the left object’, ’Include both objects in frame’, ’Focus
on the text label’)",
"selected_crop_indices": []
}}
‘‘‘
### If crops are SUFFICIENT (status: "final"):
‘‘‘json
{{
"status": "final",
"evaluation": "Detailed analysis confirming crops are sufficient",
"feedback": "",
"selected_crop_indices": [0, 2]
}}
‘‘‘
**Key Fields:**
- **‘status‘**: MUST be either "ongoing" (needs improvement) or "final" (
ready to answer)
**‘evaluation‘**: Your thorough analysis of the crops
- **‘feedback‘**: For "ongoing": specific actionable feedback. For "final":
empty string.
**‘selected_crop_indices‘**: For "ongoing": empty list []. For "final":
list of selected crop indices (0-indexed).
**Important Guidelines:**
- Be STRICT in your evaluation - only mark as "final" when crops truly
provide clear evidence
- For "ongoing", be SPECIFIC in your feedback (don’t just say "improve", say
HOW to improve)
- For "final", select 1-4 crops that best answer the question
- Consider if multiple crops are needed (e.g., for comparison questions)
```

## C.3.3 Final Answering Agent Prompt

## Final Answering Agent System Prompt (for Locator Ensembling)

"You are an AI assistant tasked with providing the final answer to a visual   
question.   
You will receive one or more carefully selected image crops that contain the   
relevant visual information needed to answer the question.   
Your task is to:   
1. Analyze the provided crop(s)   
2. Reason about what you see   
3. Provide the final answer wrapped in curly braces: {{answer}}   
You do NOT need to generate any new crops - the crops provided are already   
optimized for answering the question.

## Final Answering Agent Prompt (for Locator Ensembling)

1. Carefully examine the provided crop(s)

When you have the final answer, you MUST submit it inside curly braces, like   
this: ‘{{only the final answer}}‘.

The curly braces should contain ONLY the answer itself (e.g., a letter, a   
number, a word).