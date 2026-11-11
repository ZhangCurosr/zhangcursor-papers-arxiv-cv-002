# PhyRubric: Rubric-Grounded Diagnostic Evaluation of Physical Commonsense in Text-to-Video Generation

Huiyi Chen Weitai Kang Shicheng Fan Xinshuo Wang

Zhixiang Shen Yan Yan Philip S. Yu Baoxin Li

Department of Computer Science, University of Illinois Chicago

{hchen270, wkang126, sfan25, xwang921, zshen33, yyan55, psyu, baoxinli}@uic.edu

https://github.com/chenyil6/phyRubric

## Abstract

As text-to-video (T2V) models are increasingly positioned as world simulators, it is essential to evaluate whether their videos conform to physical commonsense. However, existing benchmarks rely on annotators who rate each video with a single score based on intuition, yielding ratings that are subjective and do not tell which physical principle is violated. We introduce PhyRubric, a rubric-grounded diagnostic framework for evaluating physical commonsense in T2V generation. For each caption, PhyRubric constructs a caption-specific rubric before any video is seen, decomposing the caption into semantic and physical items, and derives scores from item-level verdicts grounded in timestamped evidence. We instantiate this framework as PhyRubric-Bench, which spans 34 categories across five physical domains, and scale it up to PhyRubric-Train with 3,788 videos, on which we fine-tune the open-source PhyRubric-Judge. Empirical results across ten T2V models show that physical commonsense remains far from solved, and that PhyRubric-Judge correlates more closely with human judgments than existing open-source judges while matching the frontier proprietary Gemini 4 Argon. Further analysis provides actionable insightsfor T2V models.

## 1. Introduction

Text-to-video (T2V) generation has advanced rapidly, and recent T2V models [1, 13, 45, 57, 59, 62, 67] can now render complex scenes from text prompts. As these models are increasingly positioned as world simulators [1, 33, 68], a basic requirement is that their videos conform to physical commonsense, the shared expectations about the physical world that everyone acquires from everyday experience, without any textbook physics [37, 55]. A reliable and comprehensive evaluation of this ability is therefore essential for tracking progress and guiding model development.

![](images/05098ba87e777d135512b591cd15a6a45836716389e9c710fc2630eb04a30718.jpg)  
Figure 1. Intuition-based rating vs. rubric-grounded evaluation. (a) Two raters give different scores to the same video, and the scores alone do not say what went wrong. (b) Checking the video against a caption-specific rubric item by item yields scores with the violated expectation and its timestamp.

Existing benchmarks for physical commonsense have made notable progress [7, 8, 19, 47], but they share two limitations (Fig. 1). First, the ratings are subjective. Annotators rate each video and its caption with a single score on a discrete scale, but no rule defines how physical violations map to scores. Each rater thus applies an intuitive mapping, weighing the severity of violations by personal standards, so the same video may receive different scores. This inconsistency introduces noise into both benchmark results and the automatic evaluators trained on these ratings. Second, the ratings are uninterpretable. A discrete scale is easier to rate than a continuous one, since fractional scores would be harder to assign and more scattered. However, a single score reflects only how implausible a video is overall, not which physical principle is violated. Evaluations can thus only rank models and categories, yielding coarse findings that offer little guidance for improvement.

To address these limitations, we introduce PhyRubric, a rubric-grounded diagnostic framework for evaluating physical commonsense in T2V generation (Fig. 2). For each caption, PhyRubric constructs a caption-specific rubric before any video is seen, and reuses it for every video generated from that caption. The rubric decomposes the caption into verifiable items of two types: semantic items list the visible facts the caption requires, while physical items specify which visible events count as violations and are each assigned a severity level inspired by the order in which infants acquire physical knowledge [6, 41, 55]. Given a video, each item receives a verdict grounded in timestamped evidence, and scores are derived deterministically from these verdicts rather than rated directly. Every score thus comes with a diagnosis of which expectation failed and why.

We instantiate this framework as PhyRubric-Bench. Unlike prior benchmarks dominated by mechanics (Suppl. Tab. 4), it spans 34 categories across five physical domains: mechanics, optics, heat, electricity, and acoustics<sup>1</sup>. Every caption describes a complete causal event and states its outcome explicitly, so that semantic adherence requires rendering the cause, the action, and its physical consequence, e.g., “the key switch is turned, and the lamp lights” rather than “the key switch is turned.”

Beyond PhyRubric-Bench, we scale up to PhyRubric-Train, 947 captions that share the taxonomy of the benchmark but no scenes, each paired with its own rubric. Here the rubric plays a different role: rather than defining the evaluation as in PhyRubric-Bench, it guides an agentic annotator, telling it which expectations to check and where to look, yielding 3,788 labeled videos from four T2V models with 45.9K rubric-item verdicts. On these labels, we train PhyRubric-Judge, an open-source VLM that needs no rubric at inference. Given only a caption and a video, it predicts semantic adherence and physical commonsense scores together with a diagnosis of which physical expectation is violated and why.

We evaluate ten T2V models on PhyRubric-Bench and find that basic physical commonsense remains far from solved: even the strongest model, Seedance 2.5, produces videos that are both faithful to the caption and physically plausible in fewer than half of cases. Moreover, PhyRubric-Judge correlates more closely with human judgments than the judges released with existing benchmarks, generalizes well to unseen generators and domains, and provides a diagnosis that locates each violation.

Our contributions are summarized as follows:

• We propose PhyRubric, a rubric-grounded framework that constructs a caption-specific rubric before any video is seen and derives every score from evidence-grounded item verdicts, so that each score comes with a diagnosis.

• We construct PhyRubric-Bench, spanning 34 categories across five physical domains with human-verified rubrics and labels, and PhyRubric-Train, labeled by a rubricguided agentic annotator, on which we train the opensource PhyRubric-Judge that generalizes to unseen generators and domains.

• Rather than proposing a definitive standard, we show that our rubric-grounded evaluation serves as an analysis tool that yields deeper insights: semantic adherence is the main bottleneck, physical errors are all-or-nothing, and models often avoid physical errors by not depicting the event at all, a shortcut that prior protocols reward.

## 2. Related Work

## 2.1. Physical Commonsense Benchmarks for Textto-Video Generation

The rapid advancement of text-to-video (T2V) generation has motivated a range of benchmarks for its assessment [30, 31, 39, 43, 44, 56, 60, 66]. Moving beyond general visual quality, a growing line of work focuses on physical commonsense in T2V generation [7, 8, 19, 20, 36, 47, 53], while other works study physical understanding in settings where models are conditioned on observed frames and predict how the scene evolves [33, 40, 48]. However, most of these physical commonsense benchmarks are dominated by mechanics [7, 8, 20], and some include prompts that require specialized knowledge beyond everyday experience [19, 20, 36, 47]. In contrast, PhyRubric-Bench spans five physical domains with prompts grounded in physical commonsense, and pairs each prompt with a captionspecific rubric that enables diagnostic evaluation.

## 2.2. VLM-as-a-Judge for Physical Commonsense

Existing evaluation of video generation mainly follows two paradigms: standard metrics such as FVD [58], SSIM [61], and CLIP score [27], and VLM-based evaluation [12, 23, 52] such as VideoScore [25]. As standard metrics cannot capture physical plausibility, physical commonsense evaluation mainly adopts the latter, employing VLMs as dedicated judges. On the open-source side, judges such as VideoCon-Physics [7] and VideoPhy-2-AutoEval [8] are fine-tuned on the human annotations of their corresponding datasets. On the proprietary side, recent works prompt commercial models such as GPT-4o, Gemini-2.5-Pro, and GPT-5.5 as judges [19, 47, 51, 53]. Nevertheless, most of these judges output a holistic score rather than a detailed diagnosis, failing to reveal what goes wrong and why.

## 2.3. Rubric-Based Evaluation

A rubric is an explicit set of criteria that decomposes a holistic quality judgment into verifiable items, making the assessment transparent, consistent, and interpretable [14, 32]. Recently, rubrics have been widely adopted for fine-grained evaluation of large language models (LLMs) and visionlanguage models [34, 38], and increasingly tailor the criteria to each individual query rather than applying a generic rubric to all samples [4, 16, 35]. In text-to-image generation, a related line of work decomposes prompts into finegrained questions to assess text-image alignment [15, 29]. However, rubric-based evaluation remains underexplored for physical commonsense in video generation. In this work, rubrics play two roles: they serve as the evaluation criteria of PhyRubric-Bench, and they guide an agentic annotator to label training data for PhyRubric-Judge.

## 3. PhyRubric

As illustrated in Fig. 2, PhyRubric fixes a rubric for each caption before any video is seen and shares it across all videos from that caption. This keeps the criteria identical across videos, so that scores from different generators are directly comparable and cannot be swayed by what a particular video happens to show. Moreover, a rubric reviewed by humans once can be reused for any number of videos, including those from future generators.

## 3.1. Caption-Specific Rubric Formulation

We evaluate generated videos along the two dimensions introduced by VideoPhy-2 [8], semantic adherence (SA) and physical commonsense (PC), each on a 1–5 scale. Unlike prior benchmarks, where annotators assign a holistic score to each dimension, we pair every caption with a rubric that specifies what a video must satisfy to be faithful and physically plausible for that caption. The rubric consists of two groups of items, one for SA and one for PC, and each score is derived from item-level verdicts rather than rated directly. A simplified rubric example is shown in Fig. 3.

Semantic Adherence (SA). SA measures whether the video shows what the caption describes. Its rubric items are the visible facts the caption requires, in two tiers:

• Primary: the main objects and each event in the caption.

• Secondary: the remaining facts, including stated qualifiers (e.g., once, slowly, wooden), initial states implied by the caption, and the order of events.

Physical Commonsense (PC). This metric evaluates whether the events shown in the video follow physical commonsense. We define four severity levels for physical violations, inspired by the order in which infants acquire physical knowledge in developmental psychology [6, 41, 55]:

• Fatal: the scene is impossible as a whole, violating core object principles such as permanence, solidity, or continuity (e.g., objects passing through each other, or appearing or vanishing without cause).

• Core: the central interaction described by the caption does not happen or happens incorrectly (e.g., in reverse or through a broken mechanism).

• Major: a clear violation outside the central interaction, such as a missing secondary consequence (e.g., the lamp lights but the desk beneath it stays dark).

• Minor: a visual flaw that leaves the event intact, such as color or texture drift of untouched objects.

Each PC item is assigned one of these severity levels and one of ten violation types (e.g., permanence, solidity, causeless motion, process direction). Following the same developmental order, the ten types are grouped into L1 core object principles, L2 everyday dynamics, and L3 attribute experience (Suppl. Tab. 6). Each PC item is then judged by a pair of verifiable statements:

• Expect: what a physically correct video should show, specifying the moment, region, or event to inspect.

• Fail: the concrete visible signs that count as a violation, and the item fails only if one of these signs is observed.

Verdicts. An SA item is judged pass orfail. A PC item has a third verdict, not shown, which applies when the moment, region, or event specified by its Expect statement never appears in the video. Otherwise, the item fails if any sign in its Fail statement is observed, and passes if not. Note that not shown means the situation is absent, not that the expected behavior failed to occur. For example, for the caption “The key switch on the base of a desk lamp is turned, and the lamp lights” (Fig. 3), if the key is never turned in the video, P2 is not shown rather than failed, since the missing action is already penalized by the SA item S2. Since a fail can only be triggered by a listed sign, every negative verdict is grounded in a specific observation rather than an overall impression, and carries the severity and violation type of its item. Consistent with the nature of commonsense, PC items judge only the direction and order of change rather than its rate or extent, and never penalize a cause for being out of view (e.g., a heat source or power supply).

Scoring Rules. Both scores lie on a 1–5 scale and are derived from the verdicts by fixed rules. For SA, a video judged entirely unrelated to the caption scores 1. Otherwise, it scores 5 if all items pass, 4 if only secondary items fail, 3 if exactly one primary item fails, and 2 if two or more primary items fail. For PC, let F be the set of failed items. A video with $F = \emptyset$ scores 5; otherwise,

$$
\mathrm { P C } = \operatorname* { m a x } \big ( 1 , \operatorname* { m i n } _ { i \in F } c ( s _ { i } ) - \operatorname* { m a x } ( 0 , h - 1 ) \big ) ,\tag{1}
$$

where $s _ { i }$ is the severity of item $i , c ( \cdot )$ maps fatal, core, major, and minor to caps of 1, 2, 3, and 4, and h is the number of failed items of major severity or above. In words, the most severe failure sets the cap, and each additional failure of major severity or above lowers the score by one. Each rubric is checked to contain at least two primary and one secondary SA items and at least one PC item of every severity, so that every score from 1 to 5 is attainable.

## 3.2. PhyRubric-Bench

Suppl. Tab. 4 compares PhyRubric-Bench with prior benchmarks, which are dominated by mechanics. PhyRubric-

![](images/0b24a0625d3c562bfc413f40c8ef4a353a3ceb810b6ac5d89d02608a4125d844.jpg)  
Figure 2. Overview of PhyRubric. (1) Rubric formulation: each caption is paired with a rubric covering semantic adherence (SA) and physical commonsense (PC), constructed before any video is seen and reused to score videos from all models. (2) PhyRubric-Bench: captions spanning five physical domains and their rubrics are drafted by LLMs and reviewed by humans. Videos from 10 T2V models are pre-annotated by an agent and verified by humans. (3) Train & Judge: an agentic annotator labels training videos to build PhyRubric Train, on which we train PhyRubric-Judge. Given a video and its caption, the judge outputs an SA score, whether PC is assessable, a PC score, and a diagnosis of the violated expectation with timestamped evidence.

![](images/c4b6b20613bd19eb455282836d5d779bc61e0526b3ff5e01db6504f026dafb0e.jpg)  
Figure 3. A simplified rubric for the caption “The key switch on the base of a desk lamp is turned, and the lamp lights.” The full rubric is provided in Suppl. Sec. A.3.

Bench covers all five physical domains with 34 categories (Fig. 4), and every caption tests only physical commonsense and is paired with a human-verified rubric.

Caption and rubric generation. Captions are drafted by Claude Fable 5.1 [2] and GPT-5.6 Sol [49] and refined by human reviewers until each describes a single causal event that is visible within about five seconds and has an explicitly stated outcome. Stating the outcome serves two purposes. (1) Fixed expectations: what should happen is determined before any video is seen, so the rubric can specify the expected events and violations rather than guess among plausible outcomes. (2) Diagnostic failures: if a model is told what should happen yet fails to render it, the failure reflects missing physical knowledge rather than creative freedom.

In addition, captions include a person only when the human body itself is part of the physics, since current generators often produce distorted human bodies [46, 65, 66]. Each phenomenon is instantiated in three to five distinct scenes for diversity, yielding 1,192 captions, which are split into 245 for PhyRubric-Bench and 947 for PhyRubric-Train (Sec. 3.3) with no scene shared between the two. Each caption is then paired with a rubric, drafted by an LLM prompted with example rubrics from the same domain, checked automatically for format and score reachability, and revised by human reviewers until approved.

Video generation and annotation. For evaluation, the 245 benchmark captions are rendered by ten text-to-video models, six of which are unseen during judge training, yielding 2,433 videos<sup>2</sup>. Each video is first pre-annotated item by item by our agentic annotator (Sec. 3.3), and human annotators then verify every verdict against the video to obtain the ground truth. Since the rubric already specifies what counts as a violation, annotators only need to check whether the listed signs appear, rather than judging physical plausibility themselves. More construction details and statistics are given in Suppl. Sec. A.2.

![](images/bbf8fea37c05edbcec32cb8977e74c78e92f7428bb8061fdef7dbe1dc8e65ded.jpg)  
Figure 4. Representative examples from PhyRubric-Bench. Each caption describes a single everyday event and states its visible outcome.

## 3.3. PhyRubric-Train and PhyRubric-Judge

Training data annotation. Human verification does not scale to thousands of videos, so we label the training set with the same agentic annotator used to pre-annotate PhyRubric-Bench, built on Claude Sonnet 5.5 [3]. Instead of a fixed set of frames, the annotator first scans a timestamped overview of the clip and then zooms in at full frame rate where and when it needs to, until each rubric item can be decided from frames that clearly show the relevant event. This matters because physical violations often involve small objects or brief moments that uniform sampling misses. For every item, it returns a verdict with timestamped evidence, and the scores follow from the verdicts by the rules in Sec. 3.1. Applied to the 947 training captions, each rendered by four generators, it yields PhyRubric-Train with 3,788 videos and 45.9K item-level verdicts. Suppl. Sec. A.4 details its protocol and reliability.

Judge. The agentic annotator produces reliable labels but is impractical as a general-purpose evaluator: it relies on a proprietary model, makes repeated tool calls per video, and needs a rubric for every new caption. We therefore train PhyRubric-Judge, an open-source VLM, on PhyRubric-Train. Starting from Qwen3-VL-8B [5], we explore configurations that vary the rubric input, the output format, and the training objective (SFT or GRPO [54]). Our default judge takes only the caption and uniformly sampled frames, so that it applies to any new caption and receives the same input as existing judges. It outputs the SA score, whether PC is assessable, the PC score, and a diagnosis. The training targets come from PhyRubric-Train, with the diagnosis assembled from the item-level evidence: a summary of what the video shows, the missing caption facts, and each physical violation with its evidence. The judge thus learns not only what score to give but why (Suppl. Sec. C.3).

## 4. Experiments

## 4.1. Experimental Setup

Video Generative Models. PhyRubric-Bench is rendered by ten video generators. Four of them are open-source T2V models that also render PhyRubric-Train: CogVideoX1.5- 5B [64], HunyuanVideo-1.5-8B [62], LTX-Video-2B [21], and Wan2.2-T2V-A14B [59]. The other six are unseen during training and measure how well a judge generalizes to new generators: Wan2.2-TI2V-5B [59], LTX-2.3- 22B [22], Cosmos-Predict2.5-2B and Cosmos-Predict2.5- 14B [1], and the proprietary Seedance 2.5 [11] and Veo 3.1 [17]. All videos are five to six seconds long.

Evaluation Metrics. Each video is rated on a 1–5 scale for SA and PC. We report the mean SA over all videos and the mean PC over assessable videos only. We also report the joint performance (JP), the percentage of all videos with SA≥4 and PC≥4, i.e., videos that are both faithful to the caption and physically plausible. Following VideoPhy-2 [8], we use JP as the primary metric. A high PC alone only indicates that what is shown is plausible (e.g., a nearly static video can easily appear plausible), so SA is needed to confirm that the model produces the described event.

Implementation Details. For video generation, each opensource model is run with its official checkpoint and default sampling recipe, generating one video per caption with a fixed seed of 42. HunyuanVideo-1.5 and Wan2.2-T2V-A14B are rendered at 480p, CogVideoX1.5-5B at 768p, and the remaining open-source models at 704p. Seedance 2.5 and Veo 3.1 are accessed through their official APIs at 480p and 720p, respectively. Per-model settings are listed in Suppl. Sec. B.1. For PhyRubric-Judge, we fine-tune

Table 1. Physical commonsense of T2V models on PhyRubric-Bench. SA and PC are mean scores on a 1–5 scale; PC is computed ove assessable videos only. JP is the joint performance (%), the fraction of all videos with SA≥4 and PC≥4. Avg averages over domains. The best proprietary model is in bold and the best open-source model is underlined. For each model, pink and blue shading mark the JP of its best and worst domains, respectively.
<table><tr><td></td><td colspan="3">Mechanics</td><td colspan="3">Optics</td><td colspan="3">Heat</td><td colspan="3">Electricity</td><td colspan="3">Acoustics</td><td colspan="3">Avg</td></tr><tr><td>Model</td><td>SA</td><td>PC</td><td>JP</td><td>SA</td><td>PC</td><td>JP</td><td>SA</td><td>PC</td><td>JP SA</td><td>PC</td><td>JP</td><td>SA</td><td>PC</td><td>JP</td><td>SA</td><td>PC</td><td></td><td>JP</td></tr><tr><td></td><td colspan="10">Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedance 2.5 [11]</td><td>3.95</td><td>4.24</td><td>48.00</td><td>3.96</td><td>3.96</td><td>48.98</td><td>3.42</td><td>4.14 38.00</td><td>3.94</td><td>4.13</td><td>56.00</td><td>3.60</td><td>3.47</td><td>30.00</td><td></td><td>3.77</td><td>3.99</td><td>44.20</td></tr><tr><td>Veo 3.1 [17]</td><td>3.88</td><td>4.06</td><td>44.00</td><td>3.78</td><td>3.89</td><td>48.00 3.48</td><td>3.92</td><td>36.00</td><td>3.80</td><td>3.35</td><td>38.00</td><td>3.45</td><td>2.65</td><td>15.00</td><td>3.68</td><td>3.57</td><td></td><td>36.20</td></tr><tr><td></td><td colspan="10">Open-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LTX-Video-2B [21]</td><td>2.28</td><td>2.59</td><td>5.33</td><td>2.56</td><td>2.94</td><td>6.00</td><td>2.20</td><td>3.73</td><td>2.00</td><td>2.30</td><td>3.42</td><td>2.00</td><td>2.00</td><td>2.50</td><td>5.00</td><td>2.27</td><td>3.04</td><td>4.07</td></tr><tr><td>Cosmos-Predict2.5-2B [1]</td><td>2.29</td><td>3.16</td><td>5.56</td><td>2.67</td><td>4.36</td><td>12.50</td><td>2.32</td><td>3.86</td><td>4.00</td><td>2.38</td><td>3.22</td><td>6.38</td><td>2.60</td><td>2.23</td><td>5.00</td><td>2.45</td><td>3.37</td><td>6.69</td></tr><tr><td>Wan2.2-TI2V-5B [59]</td><td>2.73</td><td>2.72</td><td>9.33</td><td>2.88</td><td>3.41</td><td>16.00</td><td>2.44</td><td>3.29</td><td>10.00</td><td>2.64</td><td>2.83</td><td>4.00</td><td>2.45</td><td>2.00</td><td>0.00</td><td>2.63</td><td>2.85</td><td>7.87 5.27</td></tr><tr><td>CogVideoX1.5-5B [64]</td><td>2.21</td><td>2.52</td><td>1.33</td><td>2.64</td><td>2.63</td><td>6.00</td><td>2.34</td><td>3.78</td><td>8.00</td><td>2.34</td><td>2.76</td><td>6.00</td><td>2.50</td><td>2.71</td><td>5.00</td><td>2.41</td><td>2.88</td><td></td></tr><tr><td>HunyuanVideo-1.5-8B [62]</td><td>2.83</td><td>3.93</td><td>14.67</td><td>3.20</td><td>3.87</td><td>28.00</td><td>2.64</td><td>3.93</td><td>18.00</td><td>2.42</td><td>2.70</td><td>4.00</td><td>2.40</td><td>2.88</td><td>10.00</td><td>2.70</td><td>3.46</td><td>14.93 14.87</td></tr><tr><td>Wan2.2-T2V-A14B [59]</td><td>2.95</td><td>3.43</td><td>21.33</td><td>3.44</td><td>3.75</td><td>26.00</td><td>2.26</td><td>3.53</td><td>2.00</td><td>2.96</td><td>3.29</td><td>20.00</td><td>2.60</td><td>2.71</td><td>5.00</td><td>2.84 2.68</td><td>3.34</td><td>12.22</td></tr><tr><td>Cosmos-Predict2.5-14B [1]</td><td>2.69</td><td>3.68</td><td>13.89</td><td>2.96</td><td>3.58</td><td>14.58</td><td>2.46</td><td>3.69</td><td>12.00</td><td>2.62</td><td>3.15</td><td>10.64</td><td>2.65</td><td>2.29</td><td>10.00</td><td></td><td>3.28</td><td></td></tr><tr><td>LTX-2.3-22B [22]</td><td>2.96</td><td>3.31</td><td>17.33</td><td>2.96</td><td>3.62</td><td>18.00</td><td>2.48</td><td>3.15</td><td>8.00</td><td>2.50</td><td>3.20</td><td>14.00</td><td>2.45</td><td>2.82</td><td>0.00</td><td>2.67</td><td>3.22</td><td>11.47</td></tr></table>

Qwen3-VL-8B-Instruct [5] on PhyRubric-Train with LoRA (rank 16) [28] on all linear layers of the language model, keeping the vision encoder and projector frozen. The judge takes the caption and 16 uniformly sampled frames, each resized to at most $4 4 8 \times 4 4 8$ . We train for 2 epochs (900 steps) with an effective batch size of 8 and a cosine learning rate schedule peaking at $1 0 ^ { - 4 }$ . At inference, the judge decodes greedily, and we additionally compute a soft score [42, 63] as the expected rating under the probabilities of the five score tokens (Suppl. Sec. B.2). All experiments are run on NVIDIA GH200 Superchips.

## 4.2. Main Results

Tab. 1 reports the results of ten video generators on PhyRubric-Bench across five physics domains. We summarize the main findings below.

Physical commonsense remains far from solved. The two proprietary models lead by a wide margin: Seedance 2.5 achieves an average JP of 44.20% and Veo 3.1 36.20%, two to three times that of the strongest open-source model, HunyuanVideo-1.5-8B (14.93%). Even so, more than half of their videos are still not both faithful to the caption and physically plausible. Moreover, every open-source model has an average SA below 3, indicating that at least one primary element of the caption is typically missing from the generated video. Even among assessable videos, where the intended event does occur, their average PC never exceeds 3.5, suggesting that the event typically violates at least one expectation of major severity or above.

Video generative models handle optics best but struggle with heat and acoustics. As shown in Tab. 1, Optics is the strongest domain for eight of the ten evaluated models, and ranks second (or tied for second) for the remaining two models, Seedance 2.5 and CogVideoX1.5-5B. In contrast, Heat and Acoustics are the weakest domains for most models. Even the proprietary models reach only 30.00% (Seedance 2.5) and 15.00% (Veo 3.1) JP on Acoustics, and two open-source models score 0.00% in this domain. We conjecture that optical phenomena (e.g., reflection and shadows) are visually salient and well represented in training data, whereas thermal processes involve gradual state changes and acoustic phenomena must be conveyed through indirect visual cues (e.g., sound-induced ripples on a liquid surface), both of which are harder to render.

Semantic adherence is the main bottleneck, and physical errors are all-or-nothing. Fig. 5a shows the distribution of SA and PC scores. SA concentrates at 2 (50.8% of all videos), the score given when two or more primary elements of the caption are missing, while only 5.0% of videos receive a 4, the score for a video that misses only secondary details: most generated videos fail to depict the captioned event itself, before physics even comes into play. In contrast, once the event is fully shown, its physics is usually correct: as shown in Fig. 5b, only 5.9% of videos show the event fully but violate physics (SA≥4, PC<4), compared with 16.9% that pass, meaning that 74.1% of fully shown events are rendered with plausible physics. Physical errors themselves are all-or-nothing: PC over assessable videos is bimodal, with 52.9% scoring 5, i.e., no violation at all, and 40.8% scoring 1 or 2, i.e., a fatal or core violation, whereas scores of 3 and 4, i.e., only major or minor violations, together account for 6.3%. This suggests that when current models do show the event, they either render its physics correctly or break it at the most basic level, and rarely get the core event right but slip on peripheral details.

Models take a shortcut to high PC by not showing the event. Prior protocols rate PC on every video, so a video that never attempts the event leaves nothing for physics to contradict and receives a high PC. In PhyRubric, the PC items on the central event of such a video are judged not shown, so the video is marked unassessable and excluded from PC. We define the avoidance rate of a model as the share of its videos that never show the event. Such avoidance is common: 25.0% of all videos never show the event (Fig. 5b), and the avoidance rate ranges from 20.0% to 44.5% across open-source models, compared with 5.7% and 8.6% for Seedance 2.5 and Veo 3.1 (Fig. 5c). We regard avoidance as a failure of physical commonsense rather than merely of instruction following: every caption in PhyRubric-Bench states the outcome explicitly, so a model that still renders a static scene or a different event suggests that it does not know how the stated cause produces the stated effect. If avoided videos are scored as in prior protocols, the PC score of every open-source model rises by 0.16–0.59, and the gain grows with its avoidance rate (Fig. 5c). LTX-Video-2B, with the highest avoidance rate (44.5%), would then rank above Cosmos-Predict2.5-14B and Wan2.2-T2V-A14B. PC alone is most misleading for Cosmos-Predict2.5-2B: under prior protocols its PC would be second only to Seedance 2.5 and above Veo 3.1, yet its JP (6.69%) ranks eighth of ten, as it rarely depicts the event.

![](images/f9b64e7c9ab23bc74f0e66e89db7e5ce886608583d376f290fd70c09de3610df.jpg)  
(a) Distribution of SA and PC scores

![](images/6e23f1f36b4cd9154e3a81901dff52094bd4bf554f3d8f5ec1c28af225e37d72.jpg)  
(b) Outcome of all videos

![](images/467929775fb1e78d9aba2881c3d9d309aa11ff7877070d0f231e3e5562acad0e.jpg)  
(c) PC over assessable vs. all videos  
Figure 5. Statistics of PhyRubric-Bench. (a) SA over all videos and PC over assessable videos. (b) Outcome of each video: the event is not shown, shown incompletely (SA<4), shown but physically implausible (SA≥4, PC<4), or passing (SA≥4, PC≥4). (c) Mean PC per generator over assessable and all videos, sorted by avoidance rate (right column). Dashed lines mark two generators for reference.

## 4.3. Evaluation of PhyRubric-Judge

We assess PhyRubric-Judge by its agreement with the human labels of PhyRubric-Bench. We apply each judge to every video in PhyRubric-Bench and report two metrics on its soft score: Pearson correlation (r) [50] with the human scores, and ROC-AUC [24] for distinguishing videos that humans rate as pass (score≥4) from those rated as fail, separately for SA and PC. Suppl. Sec. B.2 reports the results of all judges on the full set of metrics adopted by prior work. Baselines. We compare PhyRubric-Judge with four groups of judges. (1) Zero-shot VLMs: Qwen3-VL-8B, Qwen3- VL-32B, and Qwen3-VL-235B-A22B [5]. The 8B model is also the untrained starting point of our judge. (2) General video evaluators: VideoScore [25] and VideoScore2 [26], fine-tuned to rate generated videos along general quality dimensions. (3) Physics-specific evaluators: VideoCon-Physics [7] and VideoPhy-2-AutoEval [8], each fine-tuned on human annotations of its own benchmark. (4) Proprietary VLM: Gemini 4 Argon [18], a frontier closed-source model, prompted with the same instructions as our judge. Details are in the Suppl. Sec. B.3.

Table 2. Agreement between judges and human labels on PhyRubric-Bench. r: Pearson correlation with human scores; AUC: ROC-AUC for distinguishing videos rated as pass (score≥4) from fail by humans. Both are computed on each judge’s soft score. Best results in bold, second best underlined.
<table><tr><td rowspan="2">Judge</td><td colspan="2">SA</td><td colspan="2">PC</td></tr><tr><td>r↑</td><td>AUC↑</td><td>r↑</td><td>AUC↑</td></tr><tr><td colspan="5">Zero-Shot VLMs</td></tr><tr><td>Qwen3-VL-8B [5]</td><td>0.46</td><td>0.78</td><td>0.02</td><td>0.53</td></tr><tr><td>Qwen3-VL-32B [5]</td><td>0.49</td><td>0.81</td><td>0.14</td><td>0.58</td></tr><tr><td>Qwen3-VL-235B [5]</td><td>0.46</td><td>0.81</td><td>0.14</td><td>0.57</td></tr><tr><td colspan="5">General Video Evaluators</td></tr><tr><td>VideoScore [25]</td><td>-0.01</td><td>0.48</td><td>0.11</td><td>0.59</td></tr><tr><td>VideoScore2 [26]</td><td>0.32</td><td>0.68</td><td>0.11</td><td>0.56</td></tr><tr><td colspan="5">Physics-Specific Evaluators</td></tr><tr><td>VideoCon-Physics [7]</td><td>0.16</td><td>0.59</td><td>0.18</td><td>0.61</td></tr><tr><td>VideoPhy-2-AutoEval [8]</td><td>0.21</td><td>0.63</td><td>0.23</td><td>0.63</td></tr><tr><td colspan="5">Proprietary VLM</td></tr><tr><td>Gemini 4 Argon [18]</td><td>0.60</td><td>0.84</td><td>0.31</td><td>0.68</td></tr><tr><td>PhyRubric-Judge (Ours)</td><td>0.61</td><td>0.84</td><td>0.32</td><td>0.68</td></tr></table>

PhyRubric-Judge aligns best with human judgments. As shown in Tab. 2, PhyRubric-Judge achieves the best agreement with human judgments on both SA and PC across all metrics. On SA, it reaches a correlation of 0.61, well above the best open-source baseline, Qwen3- VL-32B (0.49). PC is the harder dimension: open-source zero-shot VLMs perform close to chance (AUC 0.53–0.58), and even the strongest open-source evaluator, VideoPhy-2- AutoEval, reaches only r = 0.23, despite being fine-tuned in two stages on 62K human annotations from VideoPhy and VideoPhy-2. In contrast, PhyRubric-Judge reaches 0.32 while being trained on only 3,600 automatically labeled videos. Notably, this 8B model is on par with the proprietary Gemini 4 Argon (r = 0.60 on SA and 0.31 on PC), showing that rubric-grounded supervision enables a small open model to match a frontier proprietary model.

(a) Permanence & count (Wan2.2-T2V-A14B)  
Table 3. Generalization of PhyRubric-Judge. (a) Videos from the four generators used to build PhyRubric-Train (Seen, 980 videos) and from the six held-out generators (Unseen, 1,453 videos). (b) Per-domain agreement of the judge trained on all five domains (Full) and with that domain held out (w/o). Metrics as in Tab. 2.
<table><tr><td rowspan="3">(a) Seen vs. unseen generators</td><td colspan="4">Seen</td><td colspan="4">Unseen</td></tr><tr><td colspan="4">SA</td><td colspan="4">SA PC</td></tr><tr><td>r</td><td>AUC</td><td>r</td><td>PC AUC</td><td>T</td><td>AUC</td><td>T</td><td>AUC</td></tr><tr><td>Qwen3-VL-32B</td><td>0.48</td><td>0.82</td><td>0.18</td><td>0.60</td><td>0.49</td><td>0.80</td><td>0.11</td><td>0.57</td></tr><tr><td>VideoScore2</td><td>0.32</td><td>0.69</td><td>0.12</td><td>0.55</td><td>0.30</td><td>0.67</td><td>0.10</td><td>0.56</td></tr><tr><td>VideoPhy-2-AutoEval</td><td>0.25</td><td>0.66</td><td>0.24</td><td>0.64</td><td>0.18</td><td>0.62</td><td>0.23</td><td>0.63</td></tr><tr><td>PhyRubric-Judge (Ours)</td><td>0.62</td><td>0.86</td><td>0.35</td><td>0.69</td><td>0.59</td><td>0.83</td><td>0.29</td><td>0.67</td></tr></table>

<table><tr><td rowspan="3"></td><td colspan="4">Full</td><td colspan="4">w/o the domain</td></tr><tr><td colspan="2">SA</td><td colspan="2">PC</td><td colspan="2">SA</td><td colspan="2">PC</td></tr><tr><td>r</td><td>AUC</td><td>r</td><td>AUC</td><td>r</td><td>AUC</td><td>r</td><td>AUC</td></tr><tr><td>Mechanics</td><td>0.67</td><td>0.88</td><td>0.37</td><td>0.72</td><td>0.65</td><td>0.87</td><td>0.28</td><td>0.67</td></tr><tr><td>Optics</td><td>0.57</td><td>0.81</td><td>0.30</td><td>0.66</td><td>0.53</td><td>0.79</td><td>0.24</td><td>0.63</td></tr><tr><td>Heat</td><td>0.58</td><td>0.86</td><td>0.24</td><td>0.66</td><td>0.54</td><td>0.85</td><td>0.21</td><td>0.64</td></tr><tr><td>Electricity</td><td>0.59</td><td>0.83</td><td>0.34</td><td>0.70</td><td>0.56</td><td>0.82</td><td>0.30</td><td>0.69</td></tr><tr><td>Acoustics</td><td>0.54</td><td>0.81</td><td>0.19</td><td>0.58</td><td>0.52</td><td>0.79</td><td>0.23</td><td>0.60</td></tr></table>

PhyRubric-Judge generalizes to unseen generators and domains. PhyRubric-Train contains videos from only four of the ten generators. To test whether the judge learns physical commonsense rather than the artifacts of these four generators, Tab. 3 (a) compares its agreement on benchmark videos from these generators (Seen) and from the other six (Unseen). On unseen generators, PhyRubric-Judge loses only 0.03 in SA and 0.06 in PC correlation, and still outperforms every baseline. To test generalization across domains, Tab. 3 (b) trains five additional judges, each with one domain removed from PhyRubric-Train, and evaluates each on its held-out domain. SA correlation drops by at most 0.04, and PC correlation by at most 0.09 (on mechanics).

## 4.4. Error Analysis

To better understand the limitations of current T2V models, we categorize their failed items by the three levels and ten violation types in PhyRubric, with more analysis in Suppl. Sec. D.1.

The physical commonsense of current T2V models is still limited at the most basic level. As shown in Fig. 6 (a), 41.6% of the violations belong to L1 (core object principles) and 50.7% to L2 (everyday dynamics), while only 7.6% belong to L3 (attribute experience). This indicates that current models rarely fail on fine attributes, but instead break principles that hold for any object, such as persistence over time, and the way common events unfold. Fig. 6 (b) shows that the two most frequent types, missing consequence (24.0%) and permanence & count (20.4%), together account for nearly half of all violations. A permanence & count violation occurs when an object appears from nothing, vanishes, teleports, duplicates, or merges with another during the video. For example, in Fig. 7 (a), the sauce becomes fully stirred and the spoon teleports to a new position between 1.50 s and 1.62 s, a severe violation that any viewer would notice at a glance. A missing consequence violation occurs when the trigger of an event is shown but its effect never follows, or occurs at the wrong time. In Fig. 7(b), the lights are on from the start, before the finger even reaches the button. Both videos have high visual quality, yet their obvious physical errors make them distinguishable from real videos.

![](images/8c4087b8e4a3db60efd1e1e814c2eb1b7e6f672d3e6ffa2f8d7605512a059624.jpg)  
Figure 6. Distribution of PC violations in the assessable videos of PhyRubric-Bench, according to the human labels. Each violation type is counted once per video. (a) By level of the violation taxonomy. (b) By violation type, showing the five most frequent types and the rest as Other.

![](images/1d5df18d779145e409744c01f4859ca8b02b09bf177be93cbb2f9421f5bb03e2.jpg)  
“A squirt of yellow mustard and a squirt of red ketchup are stirred together on a plate, and the mixture turns orange.”

![](images/6ee91339b4df98ae997007e911ed4d080b7aef20ec5c20bbc6a3fa9162f92f52.jpg)  
“The big red button on a toy fire engine is pressed, and its roof lights start to flash.”  
Figure 7. Two failure cases from PhyRubric-Bench. (a) A small squirt of ketchup becomes a full bowl and the spoon jumps between 1.50 s and 1.62 s. (b) The roof light is lit from the first frame, before the finger reaches the button at 2.30 s.

## 5. Conclusion

We presented PhyRubric, a rubric-grounded framework for diagnostic evaluation of physical commonsense in T2V generation, which fixes a caption-specific rubric before any video is seen and derives every score from evidencegrounded item verdicts. We instantiated it as PhyRubric-Bench across five physical domains and scaled it up to PhyRubric-Train, on which we trained PhyRubric-Judge, an open-source judge that outperforms existing open-source judges, matches Gemini 4 Argon, and generalizes to unseen generators and domains. Our evaluation of T2V models reveals that physical commonsense remains far from solved, and our in-depth analysis provides actionable insights for developing future T2V models.

## LLM Usage Statement

Beyond the LLMs that are described in the paper, Claude models were used to assist with code implementation and to polish the writing. All LLM-generated code was reviewed by the authors.

## Acknowledgments

This work used the Delta and DeltaAI systems at the National Center for Supercomputing Applications (NCSA) through allocation CIS261732 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program [10], which is supported by U.S. National Science Foundation (NSF) grants #2138259, #2138286, #2138307, #2137603, and #2138296. Delta and DeltaAI [9] are supported by NSF awards OAC 2005572 and OAC 2320345, respectively, and the State of Illinois. This work also used Google Cloud through the CloudBank project, which is supported by NSF grant #1925001.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 1, 5, 6

[2] Anthropic. System card: Claude Fable 5.1 & Claude Mythos 5.1. https://www.anthropic.com/claudefable-and-mythos-5-1, 2026. Accessed: 2026-10- 05. 4

[3] Anthropic. System card: Claude Sonnet 5.5. https:// www.anthropic.com/claude-sonnet-5-5, 2026. Accessed: 2026-10-06. 5

[4] Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quinonero-Candela, Foivos Tsim-˜ pourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025. 3

[5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 5, 6, 7

[6] Renee Baillargeon. Infants’ physical world. ´ Current directions in psychological science, 13(3):89–94, 2004. 2, 3

[7] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, pages 102075– 102121, 2025. 1, 2, 7, 5, 6, 19

[8] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. In International Conference on Learning Representations, pages 118456–118470, 2026. 1, 2, 3, 5, 7, 6, 19

[9] Brett Bode, Gregory Bauer, Laura Herriott, Volodymyr Kin dratenko, and William Gropp. Deltaai: A national resource for ai/ml research. In Practice and Experience in Advanced Research Computing 2025: The Power of Collaboration, pages 1–4. 2025. 9

[10] Timothy J Boerner, Stephen Deems, Thomas R Furlani, Shelley L Knuth, and John Towns. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and experience in advanced research computing 2023: Computing for the common good, pages 173–176. 2023. 9

[11] ByteDance Seed. One-take creation, flexible referencing: Introducing seedance 2.5. https://seed.bytedance. com/en/blog/one-take-creation-flexiblereferencing - introducing - seedance - 2 - 5, 2026. Blog post; accessed 2026-10-23. 5, 6

[12] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. arXiv preprint arXiv:2402.04788, 2024. 2

[13] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 1

[14] Hao Chen, Ziyu Han, Yukun Yan, Qingfu Zhu, Maosong Sun, and Wanxiang Che. From holistic evaluation to structured criteria: Rubrics across the evolving llm landscape. arXiv preprint arXiv:2606.08625, 2026. 2

[15] Jaemin Cho, Yushi Hu, Jason Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In International conference on learning representations, pages 15625–15645, 2024. 3

[16] Jonathan Cook, Tim Rocktaschel, Jakob Foerster, Dennis¨ Aumiller, and Alex Wang. Ticking all the boxes: Generated checklists improve llm evaluation and generation. arXiv preprint arXiv:2410.03608, 2024. 3

[17] Google DeepMind. Veo 3.1. https://deepmind. google/models/veo/, 2025. Accessed: 2026-11-05. 5, 6

[18] Google DeepMind. Gemini 4 argon: Our next era of frontier intelligence. https : / / blog . google / innovation- and- ai/models- and- research/ gemini- models/gemini- 4- argon/, 2026. Accessed: 2026-11-05. 7

[19] Jing Gu, Xian Liu, Yu Zeng, Ashwin Nagarajan, Fangrui Zhu, Daniel Hong, Yue Fan, Qianqi Yan, Kaiwen Zhou, Ming-Yu Liu, et al. phyworldbench: A comprehensive evaluation of physical realism in text-to-video models. In International Conference on Learning Representations, pages 75130–75164, 2026. 1, 2

[20] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation. arXiv preprint arXiv:2505.00337, 2025. 2

[21] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 5, 6

[22] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026. 5, 6

[23] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Yufan Deng, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, et al. Video-bench: Human-aligned video generation benchmark. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18858– 18868. IEEE, 2025. 2

[24] James A Hanley and Barbara J McNeil. The meaning and use of the area under a receiver operating characteristic (roc) curve. Radiology, 143(1):29–36, 1982. 7

[25] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2105–2123, 2024. 2, 7, 5, 6, 19

[26] Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, et al. Videoscore2: Think before you score in generative video evaluation. arXiv preprint arXiv:2509.22799, 2025. 7, 5, 6, 20

[27] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021. 2

[28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 6

[29] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In 2023 ieee/cvf international confer ence on computer vision (iccv), pages 20349–20360. IEEE, 2023. 3

[30] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818. IEEE, 2024. 2

[31] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2

[32] Anders Jonsson and Gunilla Svingby. The use of scoring rubrics: Reliability, validity and educational consequences. Educational research review, 2(2):130–144, 2007. 2

[33] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 1, 2

[34] Seungone Kim, Jay Shin, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Ryan Shin, Sungdong Kim, James Thorne, Minjoon Seo, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In In ternational Conference on Learning Representations, pages 29927–29962, 2024. 3

[35] Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne Long pre, Chaeeun Kim, Dongkeun Yoon, Guijin Son, Yejin Cho, Sheikh Shafayat, Jinheon Baek, et al. The biggen bench: A principled benchmark for fine-grained evaluation of language models with language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5877– 5919, 2025. 3

[36] Megha Mariam KM, Aditya Arun, Zakaria Laskar, and CV Jawahar. Phyeduvideo: A benchmark for evaluating text-tovideo models for physics education. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8690–8699. IEEE, 2026. 2

[37] James R Kubricht, Keith J Holyoak, and Hongjing Lu. Intuitive physics: Current research and controversies. Trends in cognitive sciences, 21(10):749–759, 2017. 1

[38] Seongyun Lee, Seungone Kim, Sue Park, Geewook Kim, and Minjoon Seo. Prometheus-vision: Vision-language model as a judge for fine-grained evaluation. In Findings of the As sociation for Computational Linguistics: ACL 2024, pages 11286–11315, 2024. 3

[39] Mingxiang Liao, Hannan Lu, Xinyu Zhang, Fang Wan, Tianyu Wang, Yuzhong Zhao, Wangmeng Zuo, Qixiang Ye, and Jingdong Wang. Evaluation of text-to-video generation models: A dynamics perspective. Advances in Neural Infor mation Processing Systems, 37:109790–109816, 2024. 2

[40] Juyi Lin, Arash Akbari, Yumei He, Lin Zhao, Haichao Zhang, Arman Akbari, Xingchen Xu, Zoe Y Lu, Enfu Nan, Hokin Deng, et al. Phyground: Benchmarking physical reasoning in generative world models. arXiv preprint arXiv:2605.10806, 2026. 2

[41] Yi Lin, Maayan Stavans, and Renee Baillargeon. Infants´ physical reasoning and the cognitive architecture that supports it. Cambridge handbook of cognitive development, pages 168–194, 2022. 2, 3

[42] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522, 2023. 6, 5

[43] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-tovideo generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023. 2

[44] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22139–22149. IEEE, 2024. 2

[45] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025. 1

[46] Lu Ma, Kaibo Cao, Hao Liang, Jiaxin Lin, Zhuang Li, Yuhong Liu, Jihong Zhang, Wentao Zhang, and Bin Cui. Evaluating and predicting distorted human body parts for generated images. arXiv preprint arXiv:2503.00811, 2025. 4

[47] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024. 1, 2

[48] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 948– 958. IEEE, 2026. 2

[49] OpenAI. GPT-5.6 system card. https : //deploymentsafety.openai.com/gpt- 5- 6, 2026. Accessed: 2026-10-05. 4

[50] Karl Pearson. Note on regression and inheritance in the case of two parents. proceedings of the royal society of London, 58:240–242, 1895. 7

[51] Atin Pothiraj, Jaemin Cho, Yue Zhang, Elias Stengel-Eskin, and Mohit Bansal. Physics question scene graph: Finegrained evaluation of physical plausibility in text-to-video generation. In European Conference on Computer Vision, pages 301–318. Springer, 2026. 2

[52] Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang, Zhiyuan Zhang, Ze-

tong Zhou, Shuang Gong, et al. Judge anything: Mllm as a judge across any modality. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5742–5753, 2025. 2

[53] Enes Sanli, Baris Sarper Tezcan, Erkut Erdem, and Aykut Erdem. Physvidbench: Language-grounded evaluation of physical commonsense in text-to-video models. In Findings of the Association for Computational Linguistics: EMNLP 2026, pages 45311–45359, 2026. 2

[54] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 5

[55] Elizabeth S Spelke and Katherine D Kinzler. Core knowl edge. Developmental science, 10(1):89–96, 2007. 1, 2, 3

[56] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehen sive benchmark for compositional text-to-video generation. In 2025 IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 8406–8416. IEEE, 2025. 2

[57] Rui Sun, Yumin Zhang, Tejal Shah, Jiahao Sun, Shuoying Zhang, Wenqi Li, Haoran Duan, Bo Wei, and Rajiv Ranjan. From sora what we can see: A survey of text-to-video gener ation. arXiv preprint arXiv:2405.10674, 2024. 1

[58] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 2

[59] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 5, 6

[60] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddaus¨ Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, et al. A very big video reasoning suite. arXiv preprint arXiv:2602.20159, 2026. 2

[61] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 2

[62] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025. 1, 5, 6

[63] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023. 6, 5

[64] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. In International Conference on Learning Representations, pages 83048–83077, 2025. 5, 6

[65] Zhichao Zhang, Wei Sun, Xinyue Li, Yunhao Li, Qihang Ge, Jun Jia, Zicheng Zhang, Zhongpeng Ji, Fengyu Sun, Shangling Jui, et al. Human-activity agv quality assessment: A benchmark dataset and an objective evaluation metric. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 6771–6780, 2025. 4

[66] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 2, 4

[67] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 1

[68] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Bohan Li, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. arXiv preprint arXiv:2405.03520, 2024. 1

# PhyRubric: Rubric-Grounded Diagnostic Evaluation of Physical Commonsense in Text-to-Video Generation

Supplementary Material

## A. Dataset and Benchmark

## A.1. Comparison with Prior Benchmarks

Tab. 4 compares PhyRubric with prior benchmarks that evaluate physical commonsense in text-to-video models. Each cell is determined from the benchmark’s paper and released code and data, following the criteria below.

Domains. Prior benchmarks center on mechanics, such as collisions, falling, and fluid flow. Heat is a main category only in PhyGenBench. PhyWorldBench covers heat in two of its 50 subcategories and electricity in one, about 4% and 2% of its prompts, whereas each domain makes up about a fifth of PhyRubric’s captions. No prior benchmark covers acoustics. These under-covered domains are exactly those in which the cause (e.g., an electric current, heat, or sound) is invisible, so a model must link a visible trigger to a visible effect without seeing what connects them.

Commonsense only. We mark this column when every caption can be judged from everyday experience. PhyWorld-Bench includes an Anti-Physics category whose prompts deliberately violate physics, PhyGenBench organizes its prompts by 27 physical laws, and T2VPhysBench builds its prompts from first-principles laws such as energy conservation. Judging these videos requires treating physical violations as correct or applying formal physics knowledge, neither of which reflects how non-experts judge videos.

Rubric. Several benchmarks pair each prompt with questions or standards: PhyGenBench generates physics questions with GPT-4o, PhyWorldBench lists physics standards per prompt, and PhysVidBench derives yes/no questions from each prompt. None of them, however, assigns a severity to its items or specifies the visible signs that count as a violation, so a failed question weighs the same whether an object vanishes or a color drifts slightly, and the score is an average of answers rather than a rule over severities. PhyRubric fixes both, and every rubric is verified by humans before any video is scored.

Train set. Only VideoPhy and VideoPhy-2 release humanlabeled videos for training a judge, from which VideoCon-Physics and VideoPhy-2-AutoEval are trained. Their labels are holistic 1–5 ratings, whereas PhyRubric-Train provides item-level verdicts with timestamped evidence.

Explainable. A judge is explainable when it reports which requirement fails and what in the video supports that verdict. The judges of VideoPhy, VideoPhy-2, and PhyGen-Bench output scores only. PhyWorldBench and PhysVid-Bench are partially explainable: each yes/no answer refers to one requirement, so a negative answer indicates what is missing, but neither cites evidence from the video, and both judges answer from a text description of the video rather than from the video itself.

![](images/2aa8ad74b52aab38118888a97f05e3d0aab95ac6fc5c95d0184602834433ab85.jpg)  
Figure 8. Taxonomy of PhyRubric-Bench. The inner circle shows the five physical domains, sized by the number of captions, and the outer ring shows the 34 categories.

Open-source. We mark this column when a physicsspecific judge is trained and its weights are released. VideoPhy and VideoPhy-2 release their judges, VideoCon-Physics and VideoPhy-2-AutoEval. PhyWorldBench, Phy-GenBench, and PhysVidBench instead rely on generalpurpose models, mainly proprietary ones (OpenAI o1, GPT-4o, and Gemini 2.5 Flash, respectively). T2VPhysBench uses human evaluation only and has no judge, which we mark as not applicable.

## A.2. Statistics

Fig. 8 shows the taxonomy of PhyRubric-Bench, and Tab. 5 lists the categories of each domain and the number of training and evaluation captions. Captions average 21.1 words in PhyRubric-Train and 22.8 in PhyRubric-Bench. Each rubric contains 5.1 SA items and 7.0 PC items on average, and each PC item carries one of ten violation types (Tab. 6). The 3,788 training videos carry 45.9K item-level verdicts, and the 2,433 benchmark videos carry 29.3K.

Table 4. Comparison with prior physical commonsense benchmarks for video generation. Mechanics, Optics, Heat, Electricity, Acoustics: coverage of the five physical domains. Commonsense only: every caption tests physical commonsense that can be judged from everyday experience, without requiring formal physics knowledge. Rubric: each caption is paired with a human-verified rubric of verifiable items with explicit severities, from which the scores are derived rather than rated holistically. Train set: labeled training data released for training a VLM judge. Explainable: the judge reports which requirement fails and the supporting evidence. Open-source: a physics-specific judge is trained and its weights are released. ✓: yes; : partially; ✗: no; –: not applicable (human evaluation only, no VLM judge).
<table><tr><td></td><td colspan="8">Data</td><td colspan="2">VLM judge</td></tr><tr><td>Benchmark</td><td>Mechanics</td><td>Optics</td><td>Heat</td><td>Electricity</td><td>Acoustics</td><td>Commonsense only</td><td>Rubric</td><td>Train set</td><td>Explainable</td><td>Open-source</td></tr><tr><td>VideoPhy [7]</td><td>√</td><td>x</td><td>x</td><td>X</td><td>X</td><td>√</td><td>x</td><td>√</td><td>x</td><td>√</td></tr><tr><td>VideoPhy-2 [8]</td><td>V</td><td>x</td><td>x</td><td>x</td><td>X</td><td>√</td><td>x</td><td>√</td><td>x</td><td>√</td></tr><tr><td>PhyWorldBench [19]</td><td>√</td><td></td><td>0</td><td>0</td><td>X</td><td>X</td><td>x</td><td>x</td><td>0</td><td>x</td></tr><tr><td>PhyGenBench [47]</td><td>√</td><td>√</td><td>V</td><td>X</td><td>X</td><td>X</td><td>x</td><td>X</td><td>X</td><td>X</td></tr><tr><td>T2VPhysBench [20]</td><td>L</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>x</td><td>X</td><td></td><td></td></tr><tr><td>PhysVidBench [53]</td><td>√</td><td>x</td><td>X</td><td>x</td><td>X</td><td>√</td><td>x</td><td>x</td><td>0</td><td>x</td></tr><tr><td>PhyRubric (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 5. Taxonomy of PhyRubric-Bench. Each caption belongs to exactly one domain and one category.
<table><tr><td>Domain</td><td>Categories</td><td>#Train</td><td>#Eval</td></tr><tr><td>Mechanics</td><td>gravity and fall, inertia, collision, friction and rolling, projectile, balance and toppling, tension, deformation, cutting and tearing, fluid, granular, gas and wind, biomechanics,</td><td>242</td><td>75</td></tr><tr><td>Heat</td><td>soft material, occlusion and containment melting and freezing, boiling and steam, condensation, burning and smoke, heat re- sponse</td><td>200</td><td>50</td></tr><tr><td>Optics</td><td>shadow, reflection, refraction and transparency, illumination, color mixing</td><td>227</td><td>50</td></tr><tr><td>Electricity</td><td>switch and light, plug and power, lightning, spark, magnet, static</td><td>202</td><td>50</td></tr><tr><td>Acoustics</td><td>struck vibration, sound-driven vibration, source motion</td><td>76</td><td>20</td></tr><tr><td>Total</td><td>34 categories</td><td>947</td><td>245</td></tr></table>

## A.3. Caption and Rubric Construction

Caption Construction. Captions are drafted one category at a time. For each of the 34 categories, Claude Fable 5.1 and GPT-5.6 Sol receive the scope note of the category and the drafting instructions in Sec. E.3, and each model proposes a batch of candidates. Beyond the requirements in Sec. 3.2, the instructions add three rules. (1) A caption states the trigger and the visible outcome in plain words but never how the outcome unfolds. (2) It contains no law names, numbers, decorative adjectives, or camera directions. (3) Its outcome must hold under every reasonable rendering rather than depend on a threshold, such as how hard an object is pushed. Each candidate also comes with two hidden fields that neither the generator nor the VLM judge ever sees: one to three violation types that the event puts at risk, and an expectation briefly describing how the outcome should unfold. Both guide rubric drafting.

Candidates are first checked automatically for a wellformed record, a valid category, and the absence of numerals, and then reviewed manually in two stages. First, the categories are divided between two reviewers. Each reviewer marks every candidate in the assigned categories as keep, edit, or remove, provides a revised caption for each edit, and compares the captions within each category for repeated scenes, lack of variety, and events that belong to another domain. The other reviewer then cross-checks every decision and revision. Second, a third reviewer goes through all remaining captions to ensure consistent standards across categories. Disagreements are resolved by discussion. Review removed 242 candidates: 99 in heat, 48 in electricity, 48 in mechanics, 24 in acoustics, and 23 in optics.

Rubric Construction. Every caption in both splits receives a rubric by the same procedure. Claude Fable 5.1 drafts the rubric from the caption, the scope note of its category, its violation types and expectation, and approved rubrics from the same domain as examples, following the instructions in Sec. E.3. The instructions specify what each item may say: an SA item states a fact the caption requires and nothing else, and a PC item pairs a physical requirement with two to four visible signs of failure, constrains direction and order but never magnitude, and never requires an invisible cause such as heat, current, or sound to be seen.

Every draft is first run through a checking program, which flags a draft that is malformed, alters the caption, has too few or too many items, or makes some score from 1 to 5 unreachable. It also flags numeric thresholds, vague words, SA items that judge physics or are conditional, secondary SA items that overlap a PC item, and declared violation types that no PC item covers.

Table 6. Violation types of PhyRubric. Each PC item has one type. Types are grouped by the order in which infants acquire physical knowledge: L1 core object principles, L2 everyday dynamics, and L3 attribute experience. #Items counts the drafted PC items over all 1,192 rubrics, excluding the scene-wide item shared by every rubric.
<table><tr><td>Type</td><td>Level</td><td>Definition</td><td>#Items</td></tr><tr><td>Permanence &amp; count</td><td>L1</td><td>Objects appear from nothing, vanish, teleport, duplicate, or merge.</td><td>1,099</td></tr><tr><td>Shape &amp; identity</td><td>L1</td><td>Rigid objects morph or melt, change identity, or bodies grow or lose parts.</td><td>492</td></tr><tr><td>Solidity</td><td>L1</td><td>Matter passes through matter.</td><td>198</td></tr><tr><td>Causeless motion</td><td>L1</td><td>Inert objects start, stop, or change direction with nothing acting on them.</td><td>752</td></tr><tr><td>Gravity direction</td><td>L2</td><td>Unsupported objects hover, rise, or fall the wrong way.</td><td>735</td></tr><tr><td>Trajectory &amp; momentum</td><td>L2</td><td>Paths, speeds, or rebounds are inconsistent with how the object was set in motion.</td><td>317</td></tr><tr><td>Process direction</td><td>L2</td><td>A process runs backwards, oscillates, or resets (e.g., melted ice refreezes).</td><td>554</td></tr><tr><td>Missing consequence</td><td>L2</td><td>The central event does not produce its visible result, or produces it at the wrong place, time or direction.</td><td>911</td></tr><tr><td>Material response</td><td>L2</td><td>A material behaves like another class (e.g., glass bends, water acts as a solid).</td><td>800</td></tr><tr><td>Attribute drift</td><td>L3</td><td>The color, texture, or position of untouched objects changes between frames.</td><td>1,301</td></tr><tr><td>Total</td><td></td><td></td><td>7,159</td></tr></table>

The 1,192 drafts are then divided evenly among three reviewers regardless of category. Each reviewer reads every assigned draft together with its flags and decides whether it passes, checking whether each item tests what the caption puts at risk and and whether every sign of failure is visible in the video. A draft that does not pass is revised with the LLM, checked by the program again, and reviewed until it passes. Finally, for cross-checking, each reviewer re-reads a random 10% of the rubrics approved by another reviewer, in a cycle, and agrees with the original decision in 93% of cases. The remaining cases, mostly wording refinements rather than incorrect items, are discussed and revised. Fig. 9 shows the full rubric of the desk-lamp caption in Fig. 3.

## A.4. Annotation

Agentic annotator. Both splits are first labeled by Claude Sonnet 5.5 running as a tool-using agent. Videos are labeled by independent agent instances, each handling five videos from mixed generators. Every video is labeled in a fresh context, so the agent sees only its caption, rubric, and video, never the labels of other videos. The agent sees the video only through a frame-inspection tool with two operations. Overview returns 16 uniformly spaced frames of the whole clip, each stamped with its timestamp. Zoom returns up to 16 consecutive frames at the full frame rate within a time window chosen by the agent, optionally cropped to a region. For every video, the agent first takes an overview and writes a description of the whole clip before reading the rubric in detail. It then zooms at least once into the time window and region where the central event should occur, and zooms again whenever a decision hinges on a moment it has not yet seen at full frame rate. An item that involves a small object or a fast action is never decided from the overview alone. The full instructions are given in Sec. E.3.

The agent applies the verdict definitions of Sec. 3.1 with one additional requirement: an item passes only when the agent can point to the frames that show it, since the caption states what was requested, not what the video contains. For every item, the agent writes its verdict with one or two sentences of timestamped evidence, and for a failed PC item it also records which sign in the Fail statement was observed. The agent never outputs scores; SA and PC are computed from its verdicts by the rules in Sec. 3.1. PhyRubric-Train uses these labels as they are.

Human verification. For PhyRubric-Bench, the three reviewers who built the rubrics verify every verdict against the video. Each reviewer watches the video, reads every verdict together with its evidence, and keeps it or changes it; the agent’s original labels are stored separately so that every change can be counted. Generator names are hidden and videos are shuffled across generators, so that a reviewer cannot be biased toward or against a particular model. The three reviewers first verify 20 videos together to align their standards, and these are excluded from the statistics below. Next, 150 videos, 15 per generator, are each verified independently by two reviewers, with each pair of reviewers covering one third of them. The remaining videos are divided evenly among the three reviewers.

Overturn rate. Reviewers changed 2.8% of the SA verdicts and 5.6% of the PC verdicts. Most PC changes (61%) turn a pass into a fail, i.e., a violation the agent missed, typically a brief or small one, which matches the error patterns of Sec. 4.4. After verification, 88.1% of the videos keep both their SA and PC scores, and most of the remaining ones change by one point. On the doubly verified videos, the two reviewers agree on 95.3% of the verdicts (Cohen’s κ = 0.87), so the corrections reflect what the video shows rather than the preference of one reviewer. Because PhyRubric-Train is labeled by the same agent under the same instructions, the overturn rate also approximately estimates the error rate of its training labels, about 3% for SA and 6% for PC verdicts.

Caption: The key switch on the base of a desk lamp is turned, and the lamp lights.   
Semantic adherence (SA)   
S1 [primary] A desk lamp with a key switch on its base.   
S2 [primary] The key switch is turned.   
S3 [primary] The lamp lights.   
S4 [secondary] The lamp is dark before the turn.   
S5 [secondary] The order is dark, turn, lit.   
Physical commonsense (PC)   
P1 [fatal | permanence count]   
Expect: One lamp with one key switch; the lamp keeps its shape; the hand passes through   
nothing.   
Fail: A second lamp appears; or the lamp vanishes; or the hand passes through the base; or   
the lamp changes shape.   
P2 [fatal | permanence count] (scene-wide item, shared verbatim by all rubrics)   
Expect: Everything in view, named in the caption or not, keeps its number, identity and shape   
unless something visible acts on it, and nothing passes through anything else.   
Fail: An object that nothing visible acts on appears from nothing; or vanishes while still   
in frame and not occluded; or duplicates or merges with another; or turns into a   
different object or a different shape between frames; or any object passes through   
another. Changes brought about by the caption’s event or by a visible interaction,   
objects leaving the frame, and occlusion are not failures.   
P3 [core | missing consequence]   
Expect: The lamp lights at the moment the key clicks over, and only then.   
Fail: The key is turned and the lamp stays dark; or the lamp is lit before the key moves; or   
the lamp comes on with the key untouched; or the lamp comes on well after the turn with   
nothing else happening.   
P4 [core | process direction]   
Expect: Once on, the lamp stays lit steadily for the rest of the clip.   
Fail: The lamp goes off again by itself; or flickers on and off; or dims back to dark; or the   
key turns back by itself.   
P5 [major | missing consequence]   
Expect: The lamp’s light reaches its surroundings: the shade glows and the desk beneath   
brightens, with shadows pointing away from the lamp.   
Fail: The bulb glows while the shade and desk stay as dark as before; or the surroundings   
brighten while the lamp stays dark; or the light comes from a place other than the   
lamp.   
P6 [major | causeless motion]   
Expect: The lamp stays put as the key is turned; nothing else on the desk moves.   
Fail: The lamp slides, tips or lifts at the turn; or the shade swings by itself; or items on   
the desk jump.   
P7 [minor | attribute drift]   
Expect: The shade’s color, the base and the desk stay the same apart from the light.   
Fail: The shade changes color rather than brightness; or the base changes shape; or the   
desk’s contents rearrange.  
Figure 9. Full rubric for the caption “The key switch on the base of a desk lamp is turned, and the lamp lights.” Items are shown in descending order of severity, which differs from the drafting order in Sec. E.3 and does not affect scoring.

## B. Experimental Setup

## B.1. Generation Settings

Tab. 7 lists the settings of every generator. Each caption is passed verbatim as the prompt, with no prompt rewriting, style suffix, or negative prompt beyond a model’s own default. Clip length follows each model’s native recipe, between 5.0 and 6.0 seconds. Seedance 2.5 is run at 480p,

Table 7. Generation settings of the ten video generators. Opensource models run with their official checkpoints and default sampling recipes through the diffusers pipelines, one video per caption with seed 42; the two proprietary models are accessed through their official APIs, which expose neither a seed nor sampling parameters. CFG: classifier-free guidance scale; for Wan2.2- T2V-A14B the two values belong to the high- and low-noise experts. The distilled LTX models use a fixed timestep schedule without guidance.
<table><tr><td>Model</td><td>Resolution</td><td>Frames</td><td>FPS</td><td>Steps</td><td>CFG</td></tr><tr><td>CogVideoX1.5-5B</td><td>1360 × 768</td><td>81</td><td>16</td><td>50</td><td>6.0</td></tr><tr><td>HunyuanVideo-1.5-8B</td><td>848 × 480</td><td>121</td><td>24</td><td>50</td><td>6.0</td></tr><tr><td>LTX-Video-2B</td><td>1216 × 704</td><td>121</td><td>24</td><td>8+4</td><td></td></tr><tr><td>Wan2.2-T2V-A14B</td><td>832 × 480</td><td>81</td><td>16</td><td>40</td><td>4.0/ 3.0</td></tr><tr><td>Wan2.2-TI2V-5B</td><td>1280 × 704</td><td>121</td><td>24</td><td>50</td><td>5.0</td></tr><tr><td>LTX-2.3-22B</td><td>1280 × 704</td><td>121</td><td>24</td><td>8</td><td>一</td></tr><tr><td>Cosmos-Predict2.5-2B</td><td>1280 × 704</td><td>93</td><td>16</td><td>36</td><td>7.0</td></tr><tr><td>Cosmos-Predict2.5-14B</td><td>1280 × 704</td><td>93</td><td>16</td><td>36</td><td>7.0</td></tr><tr><td>Seedance 2.5</td><td>854 × 480</td><td>121</td><td>24</td><td>一</td><td>一</td></tr><tr><td>Veo 3.1</td><td>1280 × 720</td><td>144</td><td>24</td><td>一</td><td>一</td></tr></table>

16:9, and 5 seconds, with audio and watermark disabled. Veo 3.1 (veo-3.1-generate-001) is run at 720p and 6 seconds, without audio.

## B.2. Evaluation Metrics

Hard and soft judge scores. A generative judge emits an integer score, which we call the hard score. Following G-Eval [42] and Q-Align [63], we also extract the probabilities the model assigns to the tokens “1”–“5” at the position where it generates the score, and define the soft score as the expectation under the renormalised distribution:

$$
s ^ { \mathrm { s o f t } } = \sum _ { k = 1 } ^ { 5 } k \cdot \frac { p ( k ) } { \sum _ { j = 1 } ^ { 5 } p ( j ) } ,\tag{2}
$$

where $p ( k )$ is the probability of token $" k "$ . Unlike the hard score, the soft score can rank clips that receive the same integer, which reflects the judge’s confidence and avoids the many ties that a five-valued integer score inevitably produces, so that correlation and ranking metrics can measure finer-grained agreement with humans.

VideoCon-Physics [7] and VideoScore [25] output continuous scores by construction (a Yes-probability and a regression output, respectively), so they have a soft score but no hard score. VideoPhy-2-AutoEval [8], like PhyRubric-Judge, emits integer scores. We compute its soft score with the same construction, without changing its prompt or decoding, so that all judges are compared on soft scores.

Metrics used by prior judges. Existing judges are validated with different metrics [7, 8, 25, 26]. We report all of them for every judge (Tab. 8) and group them into three families below, where, for each clip i and each dimension (SA or $\mathrm { P C } ) , x _ { i }$ is the judge’s (hard or soft) score, $y _ { i } \in \{ 1 , \ldots , 5 \}$ the human label, and n the number of clips.

A. Correlation. These metrics measure whether the judge orders clips in the same way as humans, regardless of the score scale. We report them on both the hard and the soft score, because the prior judges validate with different score types: VideoPhy-2-AutoEval reports Pearson on its integer score, VideoScore2 on a probability-derived float.

• Pearson r is the linear correlation between x and $y \colon$

$$
r = { \frac { \sum _ { i } ( x _ { i } - { \bar { x } } ) ( y _ { i } - { \bar { y } } ) } { { \sqrt { \sum _ { i } ( x _ { i } - { \bar { x } } ) ^ { 2 } } } { \sqrt { \sum _ { i } ( y _ { i } - { \bar { y } } ) ^ { 2 } } } } } .\tag{3}
$$

It ranges in $[ - 1 , 1 ]$ , and a higher value indicates stronger linear agreement with human labels.

• Spearman $\rho$ is the Pearson correlation computed on ranks instead of raw scores, where clips with equal scores share the average of their ranks. It ranges in $[ - 1 , 1 ]$ , and a higher value indicates that the judge ranks clips more consistently with human labels, even if the two scores are not linearly related.

## B. Ranking.

• ROC-AUC splits clips by the human label into positives $\mathcal { P } = \{ i : y _ { i } \geq 4 \}$ and negatives ${ \mathcal { N } } ,$ and measures the probability that a positive clip receives a higher judge score than a negative one, counting ties as one half:

$$
\mathrm { A U C } = \frac { 1 } { | \mathcal { P } | | \mathcal { N } | } \sum _ { i \in \mathcal { P } } \sum _ { j \in \mathcal { N } } \big [ \mathbb { 1 } ( x _ { i } > x _ { j } ) + \frac { 1 } { 2 } \mathbb { 1 } ( x _ { i } = x _ { j } ) \big ] .\tag{4}
$$

It ranges in [0, 1], where 0.5 corresponds to random ordering and a higher value is better. It thresholds only the human label, so it applies to judge scores on any scale. Following VideoCon-Physics [7], which computes ROC-AUC on its continuous Yes-probability, we compute it on continuous scores: for judges that output integers, we use their soft scores (Eq. 2).

C. Agreement. These metrics check whether the judge’s decisions or scores match the human labels, in two ways.

(1) Binary agreement. Both scores are thresholded into pass/fail decisions, $z _ { i } = \mathbb { 1 } ( y _ { i } \geq 4 )$ and $\hat { z } _ { i } = \mathbb { 1 } ( x _ { i } \ge \theta )$ where θ is the judge’s own pass line (Tab. 9). We compute the metrics for SA and PC separately, and for their joint decision, under which a clip passes only if both pass.

• Binary accuracy is the fraction of clips with $\hat { z } _ { i } = z _ { i }$

• F1 is the harmonic mean of precision (the fraction of judge-passed clips that humans also pass) and recall (the fraction of human-passed clips that the judge also passes), with pass as the positive class.

(2) Score-level agreement. Exact and within-1 accuracy are reported for judges that output integer 1–5 scores, and are computed on the hard score. They measure whether the score can be used directly on the human scale, rather than how well it ranks clips.

• Exact accuracy is the fraction of clips with $x _ { i } = y _ { i }$ • Within-1 accuracy is the fraction of clips with $| x _ { i } - y _ { i } | \le$ 1, tolerating the one-point disagreements that are common among human annotators.

Table 8. Metrics reported by the original papers of prior judges (✓), and the full set we report (Ours), grouped into three families. Score: the judge score on which we compute each metric (S: soft; H: hard; pass line: each judge’s own threshold, Tab. 9).
<table><tr><td>Family</td><td>Metric</td><td>Score</td><td>VideoCon-Physics</td><td>VideoPhy-2-AutoEval</td><td>VideoScore</td><td>VideoScore2</td><td>Ours</td></tr><tr><td rowspan="2">A. Correlation</td><td>Pearson r</td><td>S/H</td><td></td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Spearman ρ</td><td>S/H</td><td></td><td></td><td>L</td><td></td><td>√</td></tr><tr><td>B. Ranking</td><td>ROC-AUC</td><td>S</td><td>√</td><td></td><td></td><td></td><td>√</td></tr><tr><td rowspan="3">C. Agreement</td><td>Binary Accuracy F1 (binary)</td><td>pass line pass line</td><td></td><td>√ √</td><td></td><td></td><td>√ √</td></tr><tr><td>Exact accuracy</td><td>H</td><td></td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>Within-1 accuracy</td><td>H</td><td></td><td></td><td></td><td>√</td><td>J</td></tr></table>

Both range in [0, 1], and a higher value is better.

## B.3. Baseline Setup

Tab. 9 lists the dimensions, native output and pass line of every judge. All baselines are run on the PhyRubric-Bench videos with their official checkpoints, prompts, frame sampling and decoding. Each baseline receives the caption and the video, but not the rubric.

VideoCon-Physics [7] and VideoPhy-2-AutoEval [8] share the mPLUG-Owl-video 7B architecture. We follow their official implementations, including the prompts (Sec. E.1) and 32 uniformly sampled frames at 224 × 224. Each is queried once for SA and once for PC. VideoCon-Physics uses the normalized probability of answering “Yes” as its soft score:

$$
s = \frac { p ( \mathrm { Y e s } ) } { p ( \mathrm { Y e s } ) + p ( \mathrm { N o } ) } .\tag{5}
$$

VideoPhy-2-AutoEval takes its greedily decoded digit as the hard score, and its soft score follows Eq. 2.

VideoScore [25] follows its official implementation, with up to 48 uniformly sampled frames and a single prompt containing the caption (Sec. E.1). Instead of generating text, it outputs five scores in [1, 4] from a regression head, one per evaluation dimension. We use text-to-video alignment as SA and factual consistency as PC.

VideoScore2 [26] follows its official implementation, with video sampled at 2 fps, its three-dimension prompt containing the caption (Sec. E.1), and sampling at temperature 0.7 with a fixed seed per clip. It first generates a reasoning trace and then an integer score for each dimension, which we parse as the hard score. Its soft score applies Eq. 2 at each score token. We use text-to-video alignment as SA and physical and common-sense consistency as PC.

Zero-shot VLMs. Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct and Qwen3-VL-235B-A22B-Instruct receive the same system prompt and input as PhyRubric-Judge (Sec. E.2), namely the SA and PC rating scales described in words, the caption, and 16 uniformly sampled frames at up to 448 × 448. They decode greedily and are parsed identically. The 8B model is the base of PhyRubric-Judge.

Proprietary VLM. Gemini 4 Argon receives the same system prompt and output format as PhyRubric-Judge (Sec. E.2). The video is sent directly and sampled by the model at 3 fps, which yields about 15 frames for a fivesecond clip, close to the 16 frames given to the local judges. Thinking cannot be disabled, so we set it to the lowest level the model allows (LOW), and we decode at temperature 0. The integer the model outputs is its hard score. The mode exposes no token probabilities, so Eq. 2 cannot be applied. Instead, in a second run, we add two fields to the requested JSON that ask the model for its probability of each score from 1 to 5, separately for SA and PC, and take the expectation of this stated distribution as the soft score.

## C. Additional Experimental Results

## C.1. Comparison with Existing VLM Judges

Tab. 10–12 report families A–C for every judge of Tab. 2. Judges whose dimensions differ from ours are mapped to SA and PC as listed in Tab. 9. All baselines use their officia checkpoints, prompts, frame sampling and decoding; implementation details are given in Sec. B. PhyRubric-Judge achieves the best result on every metric among open-source judges, and is on par with the proprietary Gemini 4 Argon.

Comparison with Gemini 4 Argon. In Tab. 10, Gemini 4 Argon leads on the hard-score rows. This is because its thinking cannot be disabled (Sec. B.3), so its integer score follows a reasoning trace, whereas the hard score of PhyRubric-Judge, like that of every other baseline, is a single greedy token without reasoning. A direct comparison of hard scores therefore favors Gemini 4 Argon. Moreover, a paired bootstrap over clips (2,000 resamples) shows that none of these gaps is significant, as all 95% confidence intervals include zero.

Table 9. Native output, dimensions and pass line of each judge. The pass line is used only by binary agreement (family C). Integer judges without an official pass line use 4, as for human labels. <sup>‡</sup>For exact and within-1 accuracy, VideoScore’s 1–4 output is mapped to 1–5 by round(1.25x). <sup>§</sup>Soft score from its stated 1–5 distribution (Sec. B.3).
<table><tr><td>Judge</td><td>Dimension used (SA / PC)</td><td>Native output</td><td>Range</td><td>Pass line</td></tr><tr><td>Zero-shot VLMs</td><td>SA / PC (our prompt)</td><td>generated integer</td><td>1-5</td><td>4</td></tr><tr><td>VideoScore‡ [25]</td><td>text-to-video alignment / factual consistency</td><td>regression output</td><td>[1, 4]</td><td>4 after mapping</td></tr><tr><td>VideoScore2 [26]</td><td>T2V alignment / physical &amp; common-sense consistency</td><td>generated integer</td><td>1-5</td><td>4</td></tr><tr><td>VideoCon-Physics</td><td>SA / PC prompts</td><td> $p ( \mathrm { Y e s } ) / ( p ( \mathrm { Y e s } ) + p ( \mathrm { N o } ) )$ </td><td>[0, 1]</td><td>0.5 (official)</td></tr><tr><td>VideoPhy-2-AutoEval</td><td>SA/PC</td><td>generated integer</td><td>1-5</td><td>4 (official)</td></tr><tr><td>Gemini 4 Argon§ [18]</td><td>SA / PC (our prompt)</td><td>generated integer</td><td>1-5</td><td>4</td></tr><tr><td>PhyRubric-Judge (ours)</td><td>SA/PC</td><td>generated integer</td><td>1-5</td><td>4</td></tr></table>

Table 10. Family A: correlation between judge scores and human labels on PhyRubric-Bench. Hard: the generated integer score. Soft: the probability-weighted score (Eq. 2), see Sec. B.3 for VideoCon-Physics, VideoScore, and Gemini 4 Argon. Higher is better. Per column, the best soft score is in bold and the best hard score is underlined.
<table><tr><td colspan="2"></td><td colspan="2">SA</td><td colspan="2">PC</td></tr><tr><td>Judge</td><td>Score</td><td>r</td><td>ρ</td><td>r</td><td>ρ</td></tr><tr><td>Zero-shot VLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-8B</td><td>hard</td><td>0.451</td><td>0.472</td><td>0.002</td><td>0.021</td></tr><tr><td></td><td>soft</td><td>0.459</td><td>0.495</td><td>-0.001</td><td>0.043</td></tr><tr><td>Qwen3-VL-32B</td><td>hard</td><td>0.467</td><td>0.502</td><td>0.143</td><td>0.146</td></tr><tr><td></td><td>soft</td><td>0.487</td><td>0.546</td><td>0.143</td><td>0.143</td></tr><tr><td>Qwen3-VL-235B</td><td>hard soft</td><td>0.449 0.464</td><td>0.482 0.540</td><td>0.129 0.137</td><td>0.123 0.130</td></tr><tr><td colspan="2">General video evaluators</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoScore</td><td>soft</td><td>-0.015</td><td>0.001</td><td>0.110</td><td>0.153</td></tr><tr><td>VideoScore2</td><td>hard soft</td><td>0.314 0.320</td><td>0.334 0.338</td><td>0.106 0.111</td><td>0.105 0.112</td></tr><tr><td colspan="2">Physics-specific evaluators</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoCon-Physics</td><td>soft</td><td>0.162</td><td>0.186</td><td>0.176</td><td>0.195</td></tr><tr><td>VideoPhy-2-AutoEval</td><td>hard soft</td><td>0.198 0.210</td><td>0.223 0.245</td><td>0.203 0.232</td><td>0.202 0.237</td></tr><tr><td colspan="2">Proprietary VLM</td><td></td><td></td><td>0.251</td><td></td></tr><tr><td>Gemini 4 Argon</td><td>hard soft</td><td>0.580 0.604</td><td>0.555 0.566</td><td>0.311</td><td>0.264 0.333</td></tr><tr><td>PhyRubric-Judge (ours)</td><td>hard soft</td><td>0.543 0.608</td><td>0.553 0.603</td><td>0.244 0.317</td><td>0.246 0.317</td></tr></table>

In Tab. 12, Gemini 4 Argon is best on binary accuracy, on SA and jointly. These numbers are inflated because it rarely predicts pass: it passes only 10% of clips on SA and 3% jointly, whereas humans pass 20% and 14%. Since most clips fail, even a trivial judge that fails every clip reaches an accuracy of 0.80 on SA and 0.86 jointly. F1 does not reward this behaviour, and PhyRubric-Judge achieves the higher F1 in every column. Moreover, its joint pass rate (17%) is close to the human rate (14%), so it yields usable per-generator scores for a leaderboard, whereas Gemini 4 Argon would leave most generators with a JP near zero (Tab. 13).

Table 11. Family B: ROC-AUC between soft judge scores and human labels, with labels binarised at $y \geq 4 .$ Notation as in Tab. 10. Higher is better (0.5: random); best per column in bold.
<table><tr><td>Judge</td><td>SA</td><td>PC</td></tr><tr><td>Zero-shot VLMs Qwen3-VL-8B</td><td>0.778</td><td>0.525</td></tr><tr><td>Qwen3-VL-32B</td><td>0.811</td><td>0.582</td></tr><tr><td>Qwen3-VL-235B</td><td>0.808</td><td>0.569</td></tr><tr><td>General video evaluators</td><td></td><td></td></tr><tr><td>VideoScore</td><td>0.477</td><td>0.589</td></tr><tr><td>VideoScore2</td><td>0.684</td><td>0.557</td></tr><tr><td>Physics-specific evaluators VideoCon-Physics</td><td></td><td></td></tr><tr><td>VideoPhy-2-AutoEval</td><td>0.595</td><td>0.611</td></tr><tr><td></td><td>0.633</td><td>0.632</td></tr><tr><td>Proprietary VLM</td><td></td><td></td></tr><tr><td>Gemini 4 Argon</td><td>0.839</td><td>0.680</td></tr><tr><td>PhyRubric-Judge (ours)</td><td>0.845</td><td>0.680</td></tr></table>

## C.2. Reproducing the Leaderboard with PhyRubric-Judge

The clip-level agreement of Sec. C.1 is a demanding test. In practice, a judge is used to score a generator, where per-clip errors average out. We therefore recompute the leaderboard of Tab. 1 from judge scores (Tab. 13).

PhyRubric-Judge reproduces the human leaderboard closely $( r = 0 . 9 3 , \rho = 0 . 8 7 )$ and keeps its JP values on the human scale, with an MAE of 4.65 points. It separates the two proprietary generators from the open-source ones by a wide margin, although it swaps their order, and it overestimates the weakest open-source generators. Gemini 4 Argon reaches a slightly higher Pearson correlation $( r = 0 . 9 4 )$ , but this is driven by the two proprietary generators alone: it compresses the other eight to a JP below 3%, so it ranks them less reliably $( \rho = 0 . 8 1 )$ and deviates far more from the human values (MAE 12.65). PhyRubric-Judge thus provides a leaderboard that is usable in absolute terms, not only in ranking.

Table 12. Family C: agreement with human labels on PhyRubric-Bench. (1) Binary: labels pass at y ≥ 4 and each judge at its own pass line on its native output (Tab. 9; the hard score for generative judges); joint: a clip passes only if both SA and PC pass. (2) Score-level: exact and within-1 accuracy of the hard score against the human 1–5 label. VideoCon-Physics outputs a probability, which has no 1–5 counterpart, so its score-level entries are left blank. Higher is better; best per column in bold.
<table><tr><td></td><td colspan="4">SA</td><td colspan="4">PC</td><td colspan="2">Joint</td></tr><tr><td></td><td colspan="2">(1) Binary</td><td colspan="2">(2) Score-level</td><td colspan="2">(1) Binary</td><td colspan="2">(2) Score-level</td><td colspan="2">(1) Binary</td></tr><tr><td>Judge</td><td>Acc</td><td>F1</td><td>Exact</td><td>W-1</td><td>Acc</td><td>F1</td><td>Exact</td><td>W-1</td><td>Acc</td><td>F1</td></tr><tr><td>Zero-shot VLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-8B</td><td>0.576</td><td>0.443</td><td>0.238</td><td>0.632</td><td>0.585</td><td>0.731</td><td>0.435</td><td>0.596</td><td>0.546</td><td>0.354</td></tr><tr><td>Qwen3-VL-32B</td><td>0.533</td><td>0.440</td><td>0.287</td><td>0.587</td><td>0.601</td><td>0.696</td><td>0.390</td><td>0.559</td><td>0.592</td><td>0.392</td></tr><tr><td>Qwen3-VL-235B</td><td>0.522</td><td>0.431</td><td>0.264</td><td>0.565</td><td>0.604</td><td>0.751</td><td>0.569</td><td>0.605</td><td>0.481</td><td>0.341</td></tr><tr><td>General video evaluators</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoScore</td><td>0.618</td><td>0.201</td><td>0.186</td><td>0.681</td><td>0.499</td><td>0.430</td><td>0.093</td><td>0.371</td><td>0.690</td><td>0.137</td></tr><tr><td>VideoScore2</td><td>0.605</td><td>0.416</td><td>0.214</td><td>0.716</td><td>0.567</td><td>0.650</td><td>0.148</td><td>0.522</td><td>0.625</td><td>0.317</td></tr><tr><td>Physics-specific evaluators</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoCon-Physics</td><td>0.679</td><td>0.295</td><td></td><td></td><td>0.470</td><td>0.314</td><td></td><td></td><td>0.814</td><td>0.125</td></tr><tr><td>VideoPhy-2-AutoEval</td><td>0.693</td><td>0.317</td><td>0.311</td><td>0.752</td><td>0.617</td><td>0.713</td><td>0.235</td><td>0.566</td><td>0.733</td><td>0.273</td></tr><tr><td>Proprietary VLM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 4 Argon</td><td>0.853</td><td>0.508</td><td>0.558</td><td>0.892</td><td>0.442</td><td>0.153</td><td>0.125</td><td>0.433</td><td>0.865</td><td>0.221</td></tr><tr><td>PhyRubric-Judge (ours)</td><td>0.836</td><td>0.547</td><td>0.614</td><td>0.846</td><td>0.648</td><td>0.757</td><td>0.599</td><td>0.636</td><td>0.844</td><td>0.498</td></tr></table>

Table 13. Leaderboard reproduced by judges. JP (%) of each generator on PhyRubric-Bench, computed as in Tab. 1 from human labels (Human) and from the hard scores of PhyRubric-Judge (Ours) and Gemini 4 Argon (Argon). The last three rows give the Pearson and Spearman correlation, and the mean absolute error (MAE, in JP points), of each judge’s column against the human column across the ten generators. Both judges score all 2,433 clips.
<table><tr><td>Generator</td><td>Human</td><td>Ours</td><td>Argon</td></tr><tr><td>Seedance 2.5</td><td>44.20</td><td>32.30</td><td>12.98</td></tr><tr><td>Veo 3.1</td><td>36.20</td><td>37.53</td><td>6.53</td></tr><tr><td>HunyuanVideo-1.5-8B</td><td>14.93</td><td>14.60</td><td>1.60</td></tr><tr><td>Wan2.2-T2V-A14B</td><td>14.87</td><td>20.13</td><td>1.60</td></tr><tr><td>Cosmos-Predict2.5-14B</td><td>12.22</td><td>19.34</td><td>2.93</td></tr><tr><td>LTX-2.3-22B</td><td>11.47</td><td>10.93</td><td>1.47</td></tr><tr><td>Wan2.2-TI2V-5B</td><td>7.87</td><td>15.27</td><td>1.20</td></tr><tr><td>Cosmos-Predict2.5-2B</td><td>6.69</td><td>11.87</td><td>2.19</td></tr><tr><td>CogVideoX1.5-5B</td><td>5.27</td><td>10.53</td><td>0.40</td></tr><tr><td>LTX-Video-2B</td><td>4.07</td><td>6.27</td><td>0.40</td></tr><tr><td>Pearson r vs. human</td><td>一</td><td>0.93</td><td>0.94</td></tr><tr><td>Spearman ρ vs. human</td><td>一</td><td>0.87</td><td>0.81</td></tr><tr><td>MAE vs. human</td><td>一</td><td>4.65</td><td>12.65</td></tr></table>

Table 14. Ablations of PhyRubric-Judge on PhyRubric-Bench. Pearson r and ROC-AUC of the soft score against human labels, as in Tab. 2. Every variant keeps the training recipe of Sec. 4.1 and changes only the stated factor.
<table><tr><td colspan="3">SA</td><td colspan="2">PC</td></tr><tr><td>Setting</td><td>r</td><td>AUC</td><td>r</td><td>AUC</td></tr><tr><td>Default (caption only)</td><td>0.608</td><td>0.845</td><td>0.317</td><td>0.680</td></tr><tr><td colspan="3">(a) Rubric input</td></tr><tr><td>+ rubric input, scores only</td><td>0.668</td><td>0.869</td><td>0.368 0.711</td></tr><tr><td>+ rubric input, item verdicts</td><td>0.568</td><td>0.821</td><td>0.254 0.629</td></tr><tr><td>Zero-shot, caption only</td><td>0.463</td><td>0.779 0.023</td><td>0.534</td></tr><tr><td>Zero-shot, + rubric input</td><td>0.368 0.688</td><td>-0.071</td><td>0.447</td></tr><tr><td colspan="3">(b) Output format</td></tr><tr><td>Diagnosis first</td><td>0.537</td><td>0.803 0.199</td><td>0.566</td></tr><tr><td>Diagnosis as thinking</td><td>0.547 0.804</td><td>0.199</td><td>0.570</td></tr><tr><td colspan="4">(c) Training objective</td></tr><tr><td>SFT, then GRPO (thinking-free)</td><td>0.595</td><td>0.841</td><td>0.314 0.684</td></tr><tr><td>SFT, then GRPO (with thinking)</td><td>0.543 0.805</td><td>0.086</td><td>0.501</td></tr><tr><td colspan="4">(d) Training data</td></tr><tr><td>25% of training data</td><td>0.581</td><td>0.830</td><td>0.215 0.635</td></tr><tr><td>50% of training data</td><td>0.588</td><td>0.837 0.247</td><td>0.653</td></tr></table>

## C.3. Ablations of PhyRubric-Judge

Tab. 14 examines four design choices of PhyRubric-Judge. Each variant trains Qwen3-VL-8B with the recipe of

Input and output of each variant in Tab. 14   
Default. Input: frames + caption.   
Output: {"sa score", "pc assessable",   
"pc score", "reason"}   
Rubric input, scores only. Input: frames +   
caption + rubric.   
Output: as Default.   
Rubric input, item verdicts. Input: frames   
+ caption + rubric.   
Output: {"summary", "items": [   
{"id": "S1", "status":   
"pass|fail", "evidence"}, ...,   
{"id": "P1", "status":   
"pass|fail|not shown", "evidence"}, ...]}   
Scores are derived from the verdicts with the   
scoring rules.   
Diagnosis first. Input: as Default.   
Output: {"reason", "sa score",   
"pc assessable", "pc score"}   
Diagnosis as thinking. Input: as Default.   
Output: <think> reason </think>   
{"sa score", "pc assessable",   
"pc score"}   
SFT, then GRPO. Input and output: as Default   
(thinking-free) or as Diagnosis as thinking   
(with thinking). GRPO changes the training   
objective, not the format.  
Figure 10. Input and output format of the ablation variants. The variants differ in whether the rubric is given and in where the scores and the diagnosis appear. The diagnosis text (reason) is identical wherever it is produced.

Sec. 4.1 and changes only one factor. Fig. 10 shows the input and output of each variant.

## Should the judge see the rubric?

The annotator needs the rubric, whereas the judge takes only the caption (Sec. 3.3). We test this choice in three ways. (1) Giving the judge the rubric as input and training it to output the scores only improves correlation by 0.06 on SA and 0.05 on PC. The rubric thus carries information that the judge cannot fully recover from the caption, but every new caption would then need a rubric before the judge can be applied. (2) Giving the rubric and training the judge to output item-wise verdicts instead of scores does not help, as the scores derived from its verdicts are less accurate than directly predicted ones. (3) The rubric is of no use to the untrained model. Given the rubric, zero-shot Qwen3-VL-8B becomes worse on both dimensions, and its PC correlation falls below zero. We therefore keep the caption-only input, so that the judge applies to any new caption, and use the rubric at annotation time only.

## Where should the diagnosis go?

The judge emits the scores first and the diagnosis after them. We compare two alternatives in which the diagnosis is generated first. (1) The diagnosis is the first field of the same JSON object. (2) The diagnosis is a chain of thought inside <think> tags, followed by a JSON object that holds the scores only. The diagnosis text is identical in all three cases, and only its position changes. Both alternatives lose 0.06–0.07 on SA and 0.12 on PC. The reason is that the soft score is the expectation over the score tokens (Eq. 2). When the score comes first, this distribution still reflects the uncertainty of the judge. After a committed diagnosis, it becomes nearly one-hot, and this uncertainty is lost. Scores first thus keeps an informative soft score and a diagnosis.

## Does reinforcement learning help?

Starting from the SFT judge, we continue training the same LoRA with GRPO for one epoch, with 4 samples per prompt at temperature 1, a batch of 8 prompts, a learning rate of $1 0 ^ { - 5 }$ , and no KL term. The reward is a weighted sum of four terms,

$$
R = r _ { \mathrm { S A } } + r _ { \mathrm { P C } } + 0 . 5 r _ { \mathrm { f m t } } + 0 . 5 r _ { \mathrm { a s s e s s } } ,\tag{6}
$$

where $r _ { \mathrm { f m t } }$ rewards a valid output format and $r _ { \mathrm { a s s e s s } }$ rewards a correct pc assessable flag. The score terms measure how close the predicted score $\hat { s } _ { d }$ is to the label $s _ { d } \colon$

$$
r _ { d } = 1 - \frac { | \hat { s } _ { d } - s _ { d } | } { 4 } , \quad d \in \{ \mathrm { S A } , \mathrm { P C } \} .\tag{7}
$$

We run this recipe twice, (1) without thinking, starting from Default, and (2) with thinking, starting from Diagnosis as thinking, and compare each run with its own starting point.

As shown in Tab. 14(c), neither run improves over its SFT starting point, and the thinking-free run is clearly better than the one with thinking, especially on PC (r of 0.314 vs. 0.086). A possible reason is that GRPO has mainly proven effective on reasoning tasks with verifiable answers, such as mathematics, and may bring less to a perceptual task such as judging physical commonsense in video. We leave this to future work and keep SFT as the default.

## How much training data is needed?

PhyRubric-Train is split into 3,600 videos for training and 188 videos from 47 held-out captions for monitoring the training loss. All judges in this paper are trained on the 3,600-video split. We train on random 25% and 50% subsets of it (900 and 1,800 videos) with the same number of epochs (Fig. 11). SA improves only slightly with data (0.58, 0.59, 0.61), whereas PC improves steadily (0.22, 0.25, 0.32). Physical plausibility is thus the dimension that benefits most from more supervision. Since the training labels are produced automatically by the annotator, the judge can be strengthened further at low cost, by generating and labeling more videos without human annotation.

![](images/d3ccfd1cd1027ea29bafcf06c23bf3e50efb9fba35bcb22d1cb94f0e574fc61d.jpg)  
Figure 11. Pearson r with human labels against the fraction of PhyRubric-Train used for training. The hollow point at 0% (dashed) is zero-shot Qwen3-VL-8B.

Table 15. Regeneration with a diagnosis-driven rewrite on 100 captions. Mean SA and PC and pass rates (%) of the blind labels for the original Wan2.2-T2V-A14B video and the two re-rendered conditions. <sup>∗</sup>: differs from the original video, <sup>†</sup>: differs from the new seed, at $p \ < \ 0 . 0 5$ (paired Wilcoxon signed-rank test over captions). No PC difference is significant.
<table><tr><td rowspan="2">Condition</td><td colspan="2">Mean</td><td colspan="3">Pass rate</td></tr><tr><td>SA</td><td>PC</td><td>SA</td><td>PC</td><td>Joint</td></tr><tr><td>Original video</td><td>2.55</td><td>3.37</td><td>11</td><td>51</td><td>6</td></tr><tr><td>New seed</td><td> $2 . 7 6 ^ { * }$ </td><td>3.29</td><td>18</td><td>50</td><td>10</td></tr><tr><td>Rewrite + diagnosis</td><td> $3 . 1 9 ^ { * \dagger }$ </td><td>3.37</td><td>31</td><td>51</td><td>18</td></tr></table>

## C.4. Can Prompting Fix Physical Violations?

Setup. We ask whether feeding the diagnosis back into the prompt helps a generator fix its failures. From the 245 Wan2.2-T2V-A14B videos of PhyRubric-Bench, we take the 193 that PhyRubric-Judge scores SA≤3 or PC≤3 and sample 100 of them, stratified by domain. Each caption is re-rendered once under two conditions with the same sampler settings: the original caption with a new seed, which controls for resampling, and a prompt of 45–70 words written by Claude Sonnet 5.5 from the caption and the judge’s diagnosis of the original video, which names the agent, states the cause before its effect, and fixes the order of events. All 200 new videos are labeled blind by the annotator with the original rubrics (Sec. A.4), and Tab. 15 compares them with the original videos.

The rewrite fixes adherence, not physics. The diagnosisdriven rewrite raises the mean SA from 2.55 to 3.19 and the SA pass rate from 11% to 31%, against 2.76 and 18% for resampling alone, and its gain over the new seed is significant. PC, in contrast, does not move: neither condition differs from the original in mean PC or in PC pass rate, and the joint pass rate rises only because more videos now pass

Table 16. Violation types by domain on PhyRubric-Bench. Each cell is the percentage of assessable videos in that domain with at least one failed PC item of the given type, according to the human labels. Types are ordered by overall frequency; the highest domain of each type is in bold.
<table><tr><td>Violation type</td><td>Mech.</td><td>Opt.</td><td>Heat</td><td>Elec.</td><td>Acou.</td><td>All</td></tr><tr><td>Missing consequence</td><td>9.1</td><td>16.1</td><td>13.5</td><td>24.6</td><td>44.4</td><td>17.3</td></tr><tr><td>Permanence &amp; count</td><td>18.5</td><td>14.2</td><td>8.3</td><td>15.8</td><td>15.5</td><td>14.7</td></tr><tr><td>Causeless motion</td><td>10.9</td><td>5.4</td><td>4.2</td><td>11.0</td><td>21.8</td><td>9.2</td></tr><tr><td>Process direction</td><td>1.8</td><td>4.1</td><td>11.2</td><td>13.0</td><td>1.4</td><td>6.4</td></tr><tr><td>Attribute drift</td><td>4.5</td><td>9.3</td><td>3.9</td><td>5.6</td><td>2.8</td><td>5.5</td></tr><tr><td>Material response</td><td>8.6</td><td>1.6</td><td>4.9</td><td>2.0</td><td>3.5</td><td>4.7</td></tr><tr><td>Gravity direction</td><td>5.9</td><td>1.0</td><td>4.4</td><td>6.5</td><td>5.6</td><td>4.7</td></tr><tr><td>Shape &amp; identity</td><td>4.1</td><td>11.7</td><td>2.3</td><td>2.3</td><td>0.0</td><td>4.7</td></tr><tr><td>Trajectory &amp; momentum</td><td>8.2</td><td>0.5</td><td>0.3</td><td>2.5</td><td>3.5</td><td>3.5</td></tr><tr><td>Solidity</td><td>2.5</td><td>0.3</td><td>0.0</td><td>2.8</td><td>0.0</td><td>1.4</td></tr><tr><td>#Assessable videos</td><td>558</td><td>386</td><td>384</td><td>354</td><td>142</td><td>1,824</td></tr></table>

SA. The rewrite can make the generator depict the caption, but it does not make it obey physics.

What the diagnosis changes. Fig. 12 shows an example. The diagnosis states that the balloon floats on a string and that no throw, no arc and therefore no order of events is seen. The rewrite is specific about exactly these points: it starts from the arm swinging and releasing the balloon, names the high arc above the fence and the splash, and keeps the scene of the original video. The generator follows this prompt, and SA rises from 2 to 5. The diagnosis turns the caption into a prompt that states what the generator left out.

## D. More Analysis

## D.1. Error Analysis

The ten violation types are defined in Tab. 6 (Sec. A.2). Tab. 16 reports, for each domain, the share of assessable videos with at least one failed PC item of each type, and Tab. 17 breaks the same rates down by generator.

Adherence decays along the causal episode. Every caption in PhyRubric describes a complete episode, namely a scene, a trigger, its effect, and their order, and the SA items follow this structure. Fig. 13(a) plots the pass rate of each stage. Models render the scene most reliably, but the pass rate falls for the trigger and its effect, and falls further for their order. The decay is steepest for the weakest models. Fig. 13(b) focuses on the step from trigger to effect. In mechanics and optics, where the effect follows directly from contact or light, 68% and 70% of triggered videos show it. In heat, electricity, and acoustics, where the cause reaches the effect through an invisible medium, only 43–49% do.

Violation types follow the physics of each domain. Tab. 16 shows that the violation types are not spread evenly over the domains but follow what each domain asks for. Missing consequence is the most frequent type overall (17.3% of assessable videos) and peaks in acoustics (44.4%) and electricity (24.6%), where an invisible sound or current must produce a visible effect, against 9.1% in mechanics. Acoustics also leads in causeless motion (21.8%), since a vibration that should follow a sound is often rendered as an object that shivers on its own. Process direction concentrates in heat and electricity (11.2% and 13.0%), where a melting process reverses or a lamp switches back off. Optics carries the appearance types, namely shape and identity (11.7%), where a reflection or a refracted object does not match its source, and attribute drift (9.3%), where the color of an untouched object changes. Mechanics, where several objects interact, leads in permanence and count (18.5%), material response (8.6%), and trajectory and momentum (8.2%). Each domain thus exposes a different class of failure, and a benchmark concentrated on mechanics would leave most of these failures unmeasured.

Caption: A water balloon is lobbed over a garden fence and arcs over into a paddling pool on the other side.  
![](images/e9df6177b4639909ccf37456955efffa18a1da73ee865b819cbd37066c992d04.jpg)

Inputs of the two re-renders   
Caption (A, with a new seed). A water balloon is lobbed over a garden fence and arcs over into a   
paddling pool on the other side.   
Diagnosis of the original video by PhyRubric-Judge (given to C). A pale green balloon on a white   
string floats in front of a black fence and moves right and down across the frame, the camera   
panning with it. It reaches the pool edge at about 2.7s and drops into the water by 3.0s, then   
floats on the surface with the string trailing to the left. No lob or throw is seen. Missing from   
the caption: The balloon floats on a string in front of the fence; no throw or lob is seen. The   
balloon drifts down to the pool edge and drops in; no arc over the fence is seen. No lob is seen,   
so no lob-then-arc order. No violation of everyday physics is visible.   
Rewrite + diagnosis (C). In a sunny garden, a person’s hand lobs a pale green water balloon over a   
tall black fence. First the arm swings and releases the balloon, then it flies in a high arc above   
the fence and drops down into a paddling pool on the other side, splashing into the water.  
Figure 12. An example of regeneration. Wan2.2-T2V-A14B re-renders the caption from the original caption (Original), with a new seed (New seed), and from a prompt rewritten with the judge’s diagnosis (Rewrite + diagnosis). Blue boxes mark SA violations, and the box lists the caption, the diagnosis, and the rewritten prompt. In the original video, the balloon hangs on a string and is never thrown, and a new seed changes little. The rewrite that sees the diagnosis produces the throw, the arc, and the splash.

![](images/bba8fdb0473fb5b4391bf72523ea20e4ed3c2b855933e274f3144b27cf55000c.jpg)  
(a) SA item pass rate along the causal episode

![](images/a805275621be8d390bf019f19f5b2af22e97c68424f2428a4c7ee6cfd21c35d1.jpg)  
(b) Effect shown given the trigger, per domain  
Figure 13. Semantic adherence along the causal episode on PhyRubric-Bench. (a) Pass rate of SA items at each stage, scene, trigger, effect and their order, for four generators. (b) Fraction of videos that show every effect among those that show the trigger, per domain.

Table 17. Violation types by generator on PhyRubric-Bench. Each cell is the percentage of assessable videos of that generator with at least one failed PC item of the given type, according to the human labels. The last row gives the number of assessable videos.
<table><tr><td>Violation type</td><td>Seedance 2.5</td><td>Veo 3.1</td><td>LTX-Video 2B</td><td>Cosmos- Predict2.5-2B</td><td>Wan2.2- TI2V-5B</td><td>CogVideoX 1.5-5B</td><td>Hunyuan Video-1.5-8B</td><td>Wan2.2- T2V-A14B</td><td>Cosmos- Predict2.5-14B</td><td>LTX-2.3 22B</td></tr><tr><td>Missing consequence</td><td>11.3</td><td>14.7</td><td>22.1</td><td>14.4</td><td>22.2</td><td>17.6</td><td>20.0</td><td>16.3</td><td>18.8</td><td>18.7</td></tr><tr><td>Permanence &amp; count</td><td>5.2</td><td>12.9</td><td>22.8</td><td>13.8</td><td>22.2</td><td>33.3</td><td>8.6</td><td>10.7</td><td>8.6</td><td>16.3</td></tr><tr><td>Causeless motion</td><td>5.7</td><td>6.7</td><td>10.3</td><td>10.0</td><td>10.8</td><td>12.1</td><td>7.6</td><td>12.8</td><td>9.1</td><td>9.0</td></tr><tr><td>Process direction</td><td>3.0</td><td>7.1</td><td>2.2</td><td>9.4</td><td>9.7</td><td>6.1</td><td>5.4</td><td>5.6</td><td>9.7</td><td>6.0</td></tr><tr><td>Attribute drift</td><td>2.6</td><td>2.7</td><td>8.8</td><td>3.8</td><td>9.7</td><td>12.7</td><td>3.8</td><td>4.1</td><td>4.3</td><td>5.4</td></tr><tr><td>Material response</td><td>2.6</td><td>0.9</td><td>7.4</td><td>6.2</td><td>9.7</td><td>7.9</td><td>2.7</td><td>4.6</td><td>3.8</td><td>3.6</td></tr><tr><td>Gravity direction</td><td>2.2</td><td>2.2</td><td>7.4</td><td>4.4</td><td>4.5</td><td>9.1</td><td>3.2</td><td>7.1</td><td>4.3</td><td>4.2</td></tr><tr><td>Shape &amp; identity</td><td>1.7</td><td>2.7</td><td>10.3</td><td>1.9</td><td>5.1</td><td>9.7</td><td>5.9</td><td>3.6</td><td>3.2</td><td>5.4</td></tr><tr><td>Trajectory &amp; momentum</td><td>1.7</td><td>4.0</td><td>4.4</td><td>3.1</td><td>4.5</td><td>4.2</td><td>3.2</td><td>3.6</td><td>3.8</td><td>2.4</td></tr><tr><td>Solidity</td><td>0.0</td><td>2.2</td><td>0.7</td><td>0.6</td><td>2.3</td><td>1.2</td><td>1.1</td><td>2.0</td><td>1.1</td><td>2.4</td></tr><tr><td>#Assessable videos</td><td>230</td><td>224</td><td>136</td><td>160</td><td>176</td><td>165</td><td>185</td><td>196</td><td>186</td><td>166</td></tr></table>

Object permanence separates the generators, while missing consequence is shared. Tab. 17 breaks the same rates down by generator, and the two most frequent types behave differently. Permanence and count failures vary by a factor of six. Objects appear, vanish, or duplicate in 33.3% of the assessable videos of CogVideoX1.5-5B and in 22– 23% of those of LTX-Video-2B and Wan2.2-TI2V-5B, but in only 5.2% for Seedance 2.5 and 8.6% for HunyuanVideo-1.5 and Cosmos-Predict2.5-14B, with Veo 3.1 in between at 12.9%. Missing consequence, in contrast, stays between 11% and 22% for every generator, and causeless motion between 6% and 13%. Missing consequence is the most frequent type for all generators except the two weakest, where permanence and count leads. Keeping objects consistent over time thus separates the generators, whereas linking a cause to its visible effect is a weakness they all share and remains the dominant failure even of the generators that largely keep objects consistent.

## D.2. Qualitative Examples

Generated videos. Figs. 14–18 show one caption per domain rendered by all ten generators. The two proprietary models render the event in every example, while opensource models most often fail before physics comes into play. Avoidance rarely looks like a frozen scene. Instead, the generator skips the action the caption describes: it either shows the outcome from the first frame, as in the fish already hanging from the rod (Cosmos-Predict2.5-14B, Fig. 17) or the phone resting on the pool edge throughout (CogVideoX1.5-5B, Fig. 14), or replaces the action with a related one, such as leafing through sheet music (LTX-2.3- 22B, Fig. 18). When the event is shown, the physical errors are brief and local, such as the sheet of paper that emerges from beneath the speaker (Cosmos-Predict2.5-2B, Fig. 18): a single rubric item catches it, and the verdict carries the item’s severity rather than a holistic impression.

Judge diagnoses. Fig. 19 shows the output of PhyRubric-Judge on four videos from PhyRubric-Bench. The diagnosis follows the structure of the training targets: a timestamped description of what the video shows, the caption facts that are missing, and each physical violation with the moment it occurs. When the event passes (a), the description is a complete timeline of the event. When the event never occurs (b), the judge says so and marks PC as not assessable. When physics breaks (c, d), the judge names the violation and when it occurs, such as fish that tumble with nothing touching them or balls that appear and vanish on the table.

![](images/620d426d78b82aa57e9a3cf7285dd4cf201668b2821f0b9d68fbe6ad1f6dc49b.jpg)  
Figure 14. Videos of ten generators for a mechanics caption. Each row shows six evenly spaced frames of one video, and the border color gives the human verdict. Green: the event is shown and physically plausible (SA≥4, PC≥4). Red: the event is shown but violates physics (SA≥4, PC<4). Blue: the event is shown incompletely (SA<4). Gray: the event is not shown.

฀A dog barks at its own reflection in a shop window, and the reflected dog barks back, faint over the goods on display behind the glass.”  
![](images/8a0467a5637548fe396863caaeaa734630a109bf8c20790179cd4563da9a062b.jpg)  
Figure 15. Videos of ten generators for an optics caption; border colors as in Fig. 14.

฀A slice of cheese is laid on a hot dog straight of the grill, and the cheese softens and slumps over the sides.”

![](images/a8a3cfcb052ce26f5bd451272b173aecc67c5018b4da5e02944a7b6c46a879a8.jpg)  
Figure 16. Videos of ten generators for a heat caption; border colors as in Fig. 14.

฀A toy fishing rod with a magnet on its line is dangled over paper fish fitted with steel clips, and a fish jumps up to the magnet and hangs from it.”

![](images/480036c1666b10a31f2418a0c1489aed5771577eeff1ce35402a091cd536372c.jpg)  
Figure 17. Videos of ten generators for an electricity caption; border colors as in Fig. 14.

฀A sheet of paper lies loosely over the grille of a large speaker, and it lifts and flaps with each beat of the bass.”  
![](images/b4581b8844bea7504aec89d63d936a5f46a1e5a47dd04b9fde4f024c39449602.jpg)  
Figure 18. Videos of ten generators for an acoustics caption; border colors as in Fig. 14.

![](images/006182c1630aa3d0b920f5686e5de545c6a335e95b33b33c79640a1aa55ce57e.jpg)  
Figure 19. Outputs of PhyRubric-Judge on four videos from PhyRubric-Bench; (a)–(c) also appear in Figs. 14–17. Each block shows six frames of the video (border color as in Fig. 14), the generator, the caption and the human scores, followed by the judge’s output verbatim. The diagnosis describes what the video shows with timestamps, lists what is missing from the caption, and names each physical violation: (a) the event is shown and passes; (b) the event never occurs, so PC is not assessable; (c) fish tumble in the air with nothing touching them; (d) balls appear and vanish on the table, so their number changes.

## E. Prompts

This section lists the prompts used by every judge, and the instructions used to draft captions and rubrics (Sec. E.3). Baseline prompts are reproduced verbatim from the official code; {caption} is replaced by the caption of the clip and <|video|> (or the equivalent placeholder) by the video tokens.

## E.1. Baseline Judges

VideoCon-Physics [7] is queried twice per clip, once for SA and once for PC, and scored as $p ( \mathrm { Y e s } ) / ( p ( \mathrm { Y e s } ) + p ( \mathrm { N o } ) )$ at the answer position.

VideoCon-Physics, SA prompt   
The following is a conversation between a curious human and AI assistant. The assistant gives   
helpful, detailed, and polite answers to the user’s questions.   
Human: <|video|>   
Human: Does this video entail the description: "{caption}"?   
AI:

The following is a conversation between a curious human and AI assistant. The assistant gives   
helpful, detailed, and polite answers to the user’s questions.   
Human: <|video|>   
Human: Does this video follow the physical laws?   
AI:

VideoPhy-2-AutoEval [8] is queried twice per clip; the first digit of the greedy output is the score.

VideoPhy-2-AutoEval, SA prompt   
The following is a conversation between a curious human and an AI assistant. The assistant gives   
helpful, detailed, and polite answers to the user’s questions.   
Human: <|video|>   
Human: Does this video match the description: "{caption}"? Please rate the video on a scale from   
1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.   
AI:

The following is a conversation between a curious human and an AI assistant. The assistant gives   
helpful, detailed, and polite answers to the user’s questions.   
Human: <|video|>   
Human: Does this video adhere to the physical laws? Rate the video on a scale from 1 to 5, where   
5 means full compliance and 1 means significant violations.   
AI:

VideoScore [25] uses one prompt for all five dimensions; the five scores are read from the regression head, and we use text-to-video alignment as SA and factual consistency as PC.

VideoScore, regression prompt   
Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please   
watch the following frames of a given video and see the text prompt for generating the video, then   
give scores from 5 different dimensions:   
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and   
color   
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or   
movements   
(3) dynamic degree, the degree of dynamic changes   
(4) text-to-video alignment, the alignment between the text prompt and the video content   
(5) factual consistency, the consistency of the video content with the common-sense and factual   
knowledge   
for each dimension, output a float number from 1.0 to 4.0, the higher the number is, the better the

video performs in that sub-score, the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the   
video is like a real video)   
Here is an output example:   
visual quality: 3.2   
temporal consistency: 2.7   
dynamic degree: 4.0   
text-to-video alignment: 2.3   
factual consistency: 1.8   
For this video, the text prompt is "{caption}", all the frames of video are as follows:

VideoScore2 [26] uses one prompt for its three dimensions and answers after a reasoning trace; we use text-to-video alignment as SA and physical/common-sense consistency as PC.

VideoScore2, prompt   
You are an expert for evaluating AI-generated videos from three dimensions:   
(1) visual quality -- clarity, smoothness, artifacts;   
(2) text-to-video alignment -- fidelity to the prompt;   
(3) physical/common-sense consistency -- naturalness and physics plausibility.   
Video prompt: {caption}   
Please output in this format:   
visual quality: <v score>;   
text-to-video alignment: <t score>,   
physical/common-sense consistency: <p score>

## E.2. PhyRubric-Judge and Zero-shot VLMs

PhyRubric-Judge and the zero-shot Qwen3-VL models receive the same system prompt and the same user turn (the frames followed by the caption). The judge is fine-tuned on this format, while the zero-shot models see it as is.

PhyRubric-Judge / zero-shot VLMs   
You are a judge of generated videos. Given the caption used to generate a video and uniformly   
sampled frames of it, score two things and explain.   
Semantic adherence (sa score, 1-5): 5 = everything the caption describes is visibly shown; 4 =   
the main objects and events are shown, only a minor detail is off; 3 = one main element of the   
caption is missing or wrong; 2 = two or more main elements are missing or wrong, but the clip is a   
recognisable attempt at the caption; 1 = no discernible relation to the caption.   
Physical commonsense (pc score, 1-5): 5 = no visible violation of everyday physics; 4 = only minor   
side details are wrong (colour, texture or position of untouched things); 3 = exactly one clear   
violation, and it is not the central interaction; 2 = the central interaction the caption is about   
does not happen, happens in reverse, or its mechanism is broken, and nothing else serious is wrong;   
or two clear violations that are both not central; 1 = the scene is impossible as a whole (matter   
passing through matter, objects appearing, vanishing, morphing or merging, gravity in the wrong   
direction); or the central interaction fails together with another clear violation; or three or   
more clear violations.   
pc assessable: true if the central physical event of the caption appears in the clip so that its   
physics can be judged; false if it never appears. When false, pc score only reflects scene-wide   
violations.   
Judge only what is visible. Do not infer unseen forces or causes. Do not penalise image quality,   
camera motion, style, how far a process has advanced, or anything that would need a numeric   
measurement.   
Answer with one JSON object: {"sa score": int, "pc assessable": bool, "pc score": int, "reason":   
str}. The reason states what the clip shows, what from the caption is missing, and which physical   
violations are seen, with timestamps.   
<video>   
Caption: {caption}

## E.3. Data Construction

The following instructions were given to the LLMs that drafted the captions and the rubrics (Sec. A.3), and to the agentic annotator (Sec. A.4); for the annotator, file paths and tool invocation details are omitted. {domain}, {category},

and {category note} are filled per category, and {caption}, {violation types}, {expectation}, and {examples} per caption.

Caption drafting   
You write captions that test the physical commonsense of text-to-video models. Each caption   
describes an everyday scene with one visible dynamic event, and will later be rendered by video   
models and judged against a rubric.   
Domain: {domain}. Category: {category}. Scope of this category: {category note}   
Write {n} captions for this category. Each caption must satisfy all of the following.   
1. One or two sentences in plain English. An everyday scene that a non-expert recognises from   
experience; no textbook demonstrations.   
2. State the triggering event AND its visible outcome (falls off, is thrown forward, shatters,   
slides a short way and stops), so the phenomenon is guaranteed to be in frame and the ending is not   
left open.   
3. Never state HOW the outcome happens physically (no ‘‘accelerating’’, ‘‘in an arc’’, ‘‘bounces   
lower each time’’, ‘‘at the same moment as’’). The manner belongs in the expectation, not the   
caption.   
4. The outcome must hold for every reasonable rendering of the caption. Do not write an event   
whose outcome depends on a threshold (fast enough, hard enough), on skill, or on details the   
caption leaves open. If the outcome is conditional, change the scene or drop it.   
5. No law names, no numbers or units, no decorative adjectives, no camera or shot directions   
(close-up, slow motion, the camera follows). Describe the scene, not how to film it.   
6. The visible change must fit in a clip of about five seconds from one fixed viewpoint. Describe   
slow processes as an ongoing state or a before-and-after result, not as a completed action.   
7. Include a person only when the human body itself is part of the physics (a hand warming ice,   
breath fogging a window). Otherwise use objects, animals, or an unseen actor.   
8. Vary the objects, settings, and triggers across captions; no two captions should share the same   
scene.   
For each caption also give:   
-- violation types: one to three tags from {permanence count, shape identity, solidity,   
causeless motion, gravity direction, trajectory momentum, process direction, missing consequence,   
material response, attribute drift}, naming the expectations this event puts at risk.   
-- expectation: one or two sentences on HOW the stated outcome should unfold: its timing relative   
to the cause, its direction, whether it is gradual or abrupt, and what must not happen. It must   
hold for every reasonable rendering. It is never shown to the video model or the judge.   
Example:   
caption: A block is pulled out of the bottom row of a tall wooden block tower and the tower   
collapses.   
violation types: gravity direction, missing consequence   
expectation: The collapse starts as the support is removed, the tower sways then falls, and the   
blocks scatter on the table; it does not stand for long on nothing.   
Return a JSON list of {"caption", "violation types", "expectation"}.

## Rubric drafting

You write the rubric for one caption. The rubric has two groups of items. SA items ask whether the clip shows what the caption says; PC items ask whether what the clip shows is physically possible. Scores are derived from the item verdicts afterwards and are never rated directly. Caption: {caption}. Category: {category}. Scope of this category: {category note}. Violation types: {violation types}. Expectation: {expectation}.   
Approved rubrics from the same domain, to match in structure, length and tone: {examples} SA items (3--6; at least two primary and one secondary).   
-- One item per visible fact the caption requires: the objects, the action, the stated outcome. Nothing the caption does not say, and nothing about physics or quality.   
-- Primary: facts without which the clip is not this caption. Secondary: qualifiers the caption states (once, slowly, wooden), stated counts, and, when the caption names several sub-actions, one item asking whether they happen in the caption’s order.   
-- A secondary item never restates a physical expectation, is never hypothetical, and never depends on an event the caption does not include. Every item must be decidable on any clip.   
PC items (six; a seventh scene-wide fatal item is appended automatically), in this order:   
1. Core items, one per violation type that is the central mechanism, built from the expectation, in the order the event unfolds.   
2. Major items: the accompanying physics of side elements that this scene puts at risk (gravity on a secondary object, material response, a secondary consequence).   
3. One fatal item naming the scene’s objects: nothing appears, vanishes, merges, morphs, or passes through anything.   
4. One minor attribute-drift item: colour, texture, label and position of untouched things. Each PC item has an expect and a fail. Expect says what a correct clip shows and thereby which moment, region or event the item is about. Fail is the only ground for a FAIL: two to four concrete visible events joined by ‘‘; or’’, each something a viewer can point at in the frames. Fail is not the negation of expect. Judge direction and order only, never magnitude or rate: a correct clip in which the process has barely advanced must pass. Never require an unseen cause (heat, current, sound, wind, a hand) to be visible. Blur, flicker, resolution, colour cast, style, and self-driven agents starting or stopping are not violations. Do not use vague words (unrealistic, implausible, properly, correctly, natural, smoothly); name the event. No two items share a fail text. Add a precondition only for scene content the caption does not guarantee; such an item is not shown when the precondition does not occur.   
Present tense, plain words, no numbers with units, no camera or image-quality language. Return the rubric as JSON in the format of the examples.

## Agentic annotator

You label generated videos against a rubric, from the video frames alone. Work independently: you never see the labels of other videos or other labellers.

Do not judge from the caption. Captions say what was requested; generated videos often do something else. Mark a pass only when you can point to what you saw.

-- SA item: pass if the clip visibly shows exactly what the check says, else fail. SA is never not shown. Read each check literally: a loosely similar action does not count (a hand resting on a car is not pushing it). A clip showing a different scene fails SA items and is NOT penalised in PC for that.

-- PC item, three steps. (1) Is the situation the item is about in the clip at all (the precondition occurs; the moment, region or event in expect appears)? If not: not shown. not shown means the situation is absent; it does not mean the correct behaviour is missing, and it does not mean unsure. (2) Situation present and you SEE an event listed in fail: fail, and copy the matching clause into matched fail. The fail text is the only ground for a fail. (3) Otherwise pass.

How to look (small objects and fast actions are easy to miss). Tools: overview(N=16): N evenly spaced frames of the whole clip, with timestamps. zoom(t0, t1, [box]): full-rate frames between t0 and t1 seconds, optionally cropped to a box   
given as fractions, at most 16 frames.

For EVERY clip: (a) make one overview and describe the whole clip to yourself before reading the items in detail; (b) then make at least one zoom on the region and time window where the key event should be; (c) zoom again if a decision hinges on a moment you have not seen at full rate. Do not decide an item from the overview alone when the object involved is small or the action is fast. Output per clip, as JSON:

"items": [{"id": "S1", "status": "pass|fail", "evidence": "one or two sentences saying what you SAW and when, not restating the rubric"},