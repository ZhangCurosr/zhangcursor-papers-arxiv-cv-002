# When Seeing Is Not Enough: Benchmarking Interactive Visual Grounding in LVLMs

Zhengxiang Wang   
Department of Linguistics & IACS Stony Brook University   
zhengxiang.wang@stonybrook.edu

Owen Rambow Department of Linguistics & IACS Stony Brook University owen.rambow@stonybrook.edu

## Abstract

Visual grounding is typically evaluated as a one-shot mapping from an informative referring expression to a visual target. This formulation misses a central property of real-world reference: target information is often incomplete, ambiguous, and established through interaction. We introduce a controlled evaluation framework for interactive visual grounding in large vision-language models (LVLMs), varying how much target information is provided upfront and how much must be acquired through dialogue. Across four human-grounded visual contexts and four interaction protocols, current LVLMs perform significantly below task-level human baselines. Interaction can help when follow-up questions refine or repair an initial target description. Performance is lowest when no initial description is provided and target information must be acquired through questions, indicating that proactive question-driven grounding remains difficult. LVLMs are also poorly calibrated, often reporting confidence that exceeds their empirical accuracy. Followup studies confirm these patterns across varied description sources (human versus AI), reasoning efforts, repeated interactions, description providers, and visual contexts. Overall, interactive visual grounding remains an important challenge, requiring visual matching, information seeking and synthesis.<sup>1</sup>

## 1 Introduction

Visual grounding is a core vision-language capability that requires a system to identify and localize regions or objects in a visual environment from a textual description (Xiao et al., 2026). It underlies applications such as visual question answering, visual retrieval, human–AI interaction, and embodied AI agents. Enabled by large-scale crossmodal training and alignment (Radford et al., 2021;

![](images/7af31a19cb1d3b3bfa9105fe96c5faa76f43c5094f35f1ea84fee3b4e949008c.jpg)  
Figure 1: Traditional visual grounding performs oneshot matching, assuming an informative input description. Interactive visual grounding removes this assumption, requiring the Matcher (M) to interact with the Director (D) to seek more information. Examples are adapted from Hawkins et al. (2020) to be illustrative.

Alayrac et al., 2022; Liu et al., 2023), recent large vision-language models (LVLMs) have demonstrated strong general-purpose vision-language capabilities (Wang et al., 2024; OpenAI et al., 2024; Gemini Team et al., 2025); in particular, they have largely saturated standard visual grounding benchmarks (Chen et al., 2024; Dong et al., 2026).

However, traditional visual grounding (Figure 1 upper part) relies on a restrictive assumption: the model is given a sufficiently informative referring expression before grounding begins. Most benchmarks therefore cast grounding as a one-shot, noninteractive task, where the model maps an already available description to a visual target (Mao et al., 2016). This setup primarily evaluates static visual matching. It does not test whether a model can recognize that a description is incomplete, ask for missing information, or update its grounding decision as new evidence becomes available.

This assumption departs from how reference works in real-world human communication (Figure 1, lower part). Referring expressions are frequently underspecified, ambiguous, or sufficient only relative to a particular visual context. Speakers and listeners establish reference collaboratively, using clarification questions and feedback to build common ground over time (Clark and Wilkes-Gibbs, 1986; Brennan and Clark, 1996).

We therefore study interactive visual grounding, where target information may be ambiguous or underspecified at first and must be established through interaction. We argue that interactive visual grounding requires two broad capabilities. First, models need visual matching ability: mapping a referring expression, or an accumulated target representation, to the intended visual item. Second, models need information seeking and synthesis ability: assessing whether the current information is sufficient, asking effective questions when it is not, and synthesizing answers across turns. Traditional grounding benchmarks primarily test the former; interactive grounding requires both.

Prior studies evaluate non-interactive referringexpression comprehension or object matching (Tang et al., 2024; Wang et al., 2025a), interactive reference games, and human–AI communication (Hua and Artzi, 2024; Imai et al., 2025; Zeng et al., 2026; Jones et al., 2026). They reveal important limitations but do not systematically vary whether target information is supplied or must be acquired, leaving visual-matching failures difficult to distinguish from information-seeking failures.

To address this gap, we introduce a controlled evaluation framework for interactive visual grounding. Our framework varies how much target information is available upfront and how much must be acquired through dialogue, thereby testing whether models can ground informative descriptions, use optional clarification, recover from underspecification, and construct target representations through their own questions. Across 38,848 target selections from eight LVLMs under diverse experimental conditions, current models perform significantly below task-level human baselines even when additional target information is available. They can use follow-up questions to refine or repair an initial description, but struggle to ground targets through proactive question-driven dialogue despite asking the most questions. Their reported confidence also frequently exceeds empirical accuracy. Together, these findings show that interactive visual grounding remains an important challenge for LVLMs beyond static visual matching.

## 2 Related Work

Static Visual Grounding Most visual grounding benchmarks evaluate grounding as a static identification or localization problem: given an image and a referring expression, a model selects the corresponding region or object (Lin et al., 2014; Kazemzadeh et al., 2014; Plummer et al., 2015; Yu et al., 2016; Chen et al., 2024; Dong et al., 2026). This formulation has been central to measuring visual-language alignment, but it assumes that the referring expression is already sufficiently informative. As a result, it primarily tests visual matching, rather than whether a model can recognize missing information or seek clarification.

Interactive Visual Grounding Prior work has introduced interaction into visual grounding through goal-oriented visual dialogue and clarificationbased object identification. GuessWhat?! frames object identification as a single-target game in which a questioner asks yes/no questions to identify an object (de Vries et al., 2017). INGRESS studies situated referring-expression grounding, where a robot asks clarification questions when a referent is ambiguous (Shridhar et al., 2020). InViG benchmarks open-ended multi-turn grounding for object disambiguation (Zhang et al., 2023). These settings show that interaction is important for grounding, but they typically focus on single-target identification or constrained question-answering. In contrast, we study multi-target grounding under controlled information conditions, allowing us to distinguish visual matching from information-seeking ability.

LVLMs in Reference Games Recent studies have evaluated LVLMs in repeated, multi-target reference games, building on classic work on referential communication and common ground (Krauss and Glucksberg, 1969; Clark and Wilkes-Gibbs, 1986; Schober and Clark, 1989). Some work examines whether LVLMs can comprehend referring expressions in human-human dialogues that become more efficient over repeated interaction (Wang et al., 2025a). Other work studies LVLM behavior in self-play (Hua and Artzi, 2024; Imai et al., 2025) or human–AI interaction (Zeng et al., 2026; Jones et al., 2026), revealing non-human-like patterns in both generation and comprehension. Our work differs by systematically controlling how much target information is provided upfront and how much must be acquired through dialogue. This lets us test whether model failures arise from visual matching or information seeking or their combination.

![](images/7fdf529dd1d5ec2e84b669e37eb1f287b991c58dab33304b5602991bdc7891c6.jpg)  
Figure 2: Overview of our proposed framework for controlled interactive visual grounding evaluation. (A) and (C) show the visual contexts and metrics used in our main experiments, respectively; (B) summarizes the interaction protocols and grounding process. The “Others” entries in (A) and (C) highlight the framework’s extensibility.

Multi-Turn Evaluation Multi-turn evaluation has been increasingly used to evaluate LLMs beyond isolated single-turn prompts (Zheng et al., 2023; Bai et al., 2024; Kwan et al., 2024; Wang et al., 2025b; Han et al., 2025). Recent work shows that LLMs can degrade across turns and often fail to seek clarification, causing misunderstandings to compound over interaction (Laban et al., 2026; Shaikh et al., 2025). Multimodal multi-turn evaluation extends these challenges to LVLMs, where models must track visual inputs, dialogue history, and evolving user intent (Liu et al., 2024; Lee et al., 2025; Xue et al., 2025; Feng et al., 2023; Tian et al., 2025). Our work instead tests whether models can use dialogue to determine and acquire missing target information and improve grounding accuracy.

## 3 Interactive Visual Grounding

Figure 2 visualizes our proposed evaluation framework, connecting the visual contexts (Panel A), multi-target visual grounding (Panel B), and evaluation metrics (Panel C). This section details the core designs for the overall task, the interaction protocols, and the evaluation metrics.

Task Formalization A Director sees an ordered target set $D = \left( d _ { 1 } , \ldots , d _ { k } \right)$ ; a Matcher sees a separately indexed candidate set $M = ( m _ { 1 } , \ldots , m _ { n } )$ containing the targets and possibly additional distractors. The players do not see each other’s view, and indices are randomized independently.

For each target position j, the players exchange utterances until the Matcher selects $\hat { d } _ { j } \in M$ . The Matcher may first ask clarification questions and may later revise a selection. The final output is the ordered sequence $\hat { D } = ( \hat { d } _ { 1 } , \dots , \hat { d } _ { k } )$ Thus, the task adds clarification, cross-turn integration, and revision to one-shot matching under partial observability (Clark and Wilkes-Gibbs, 1986).

Interaction Protocols We instantiate four protocols that vary how much target information is provided upfront and how much the Matcher must acquire through interaction.

In STATIC, the Matcher receives one humanproduced target description at a time and selects without interaction. “Full” means that the description comes from a high-accuracy human dialogue, not that it exhaustively specifies the target or was individually matched correctly. FULL provides the same description but permits optional follow-up questions, testing whether clarification can verify, refine, or repair an initially informative input. PARTIAL instead begins with an underspecified attribute, requiring the Matcher to seek missing information. NONE provides no initial description; the Director returns only yes/no/unclear answers, so the Matcher must decide what to request and construct a target representation through questions (de Vries et al., 2017). Together, the protocols progress from matching an informative description (STATIC) to optional clarification (FULL), recovery from underspecification (PARTIAL), and fully question-driven grounding (NONE).

<table><tr><td rowspan="2">Image Set</td><td rowspan="2">Source</td><td colspan="3">Dataset Size</td><td colspan="4">Context Properties</td></tr><tr><td># Items</td><td># Targets</td><td># Dialogues</td><td>Matching</td><td>Modality</td><td>Object Type</td><td>Nameable</td></tr><tr><td>Dogs</td><td>W</td><td>13</td><td>10</td><td>10</td><td>Free-form</td><td>SPOKEN</td><td>REAL</td><td>√</td></tr><tr><td>Baskets 1</td><td>W</td><td>13</td><td>10</td><td>10</td><td>Free-form</td><td>SPOKEN</td><td>REAL</td><td></td></tr><tr><td>Baskets 2</td><td>Z</td><td>18</td><td>12</td><td>14</td><td>Free-form</td><td>TEXT</td><td>REAL</td><td></td></tr><tr><td>Tangrams</td><td>H</td><td>12</td><td>12</td><td>14</td><td>Sequential</td><td>TEXT</td><td>ABSTRACT</td><td></td></tr></table>

Table 1: Dataset statistics and key properties of the four visual contexts used in our interactive visual grounding experiments. Source: W = Wang et al. (2025a); Z = Zeng et al. (2026); and H = Hawkins et al. (2020). In the source human tasks, free-form matching permitted revisions, whereas sequential matching proceeded one target at a time without revision; our simulated interaction rules are described in Section 2.

A description drawn from a high-performing human dialogue may still leave an LVLM or even another human uncertain or incorrect, motivating optional interaction even in the FULL condition.

Evaluation Metrics We primarily consider accuracy and question rate as our main metrics, reflecting models’ visual matching ability and tendency to seek information in interactive settings. Accuracy is the percentage of correctly matched items within a task round. Question rate is the average number of questions asked per item; by definition, it is 0 under STATIC. As a baseline, random accuracy is 1/n for a Matcher view with n candidates.

We also analyze calibration to assess whether models know when they have enough information to make a selection. Each Matcher is prompted to report a confidence score with its final selection, and we compare these stated confidence scores against empirical accuracy using calibration plots.<sup>2</sup>

Finally, we consider self-repairs as a secondary behavioral signal. Self-repairs occur when a Matcher revises an earlier selection after receiving additional information, indicating whether models can recognize and correct earlier grounding decisions during interaction.

## 4 Experimental Setup

## 4.1 Datasets

We evaluate interactive visual grounding across four human-grounded, object-level visual contexts (Table 1): dogs and baskets from Wang et al.

(2025a), baskets from Zeng et al. (2026), and tangrams from Hawkins et al. (2020). Appendix H provides more details on the four image sets, and Panel A of Figure 2 shows representative examples.

We select these datasets for two reasons. First, they provide human-to-human reference data over the same visual objects, allowing us to simulate human Directors and compare model behavior against human-grounded baselines. Second, they cover several dimensions central to referential communication: real versus abstract objects, spoken versus text-based references, free matching versus sequential target selection, and objects with versus without conventional names. The less nameable contexts are particularly challenging: baskets require finegrained attribute descriptions, while tangrams often elicit highly variable descriptions across speakers. We also include two basket image sets with different levels of initial-round human performance: 80% in Zeng et al. (2026) and 100% in Wang et al. (2025a). Together, these image sets provide a diverse, human-grounded testbed for evaluating LVLMs under varying degrees of visual ambiguity, nameability, and referential difficulty.

All four datasets contain repeated-round human reference data, where a round denotes one pass through the same image set by the same dyad. By default, we use only the initial round, where referring expressions are more self-contained and less dependent on dyad-specific common ground.

## 4.2 Referring Expression Processing

We reuse the referring expressions from the four datasets in Table 1, which contain human-produced descriptions for the same visual objects across multiple discourse participants. This supports a realistic and diverse evaluation setting, given the openended nature of referring expression generation.

Filtering To ensure that the retained descriptions are accurate and sufficiently informative, we keep only initial-round referring expressions from dialogues in which human participants achieved at least 80% matching accuracy for Zeng et al. (2026) and 90% for Hawkins et al. (2020), balancing description quality against sample size. For the two image sets from Wang et al. (2025a), initial-round accuracy is 100%, so we retain all dialogues. Although item-level success could provide a more fine-grained filter, we use dialogue-level accuracy because the referring expressions are drawn from complete interaction transcripts and may therefore depend on the broader dialogue context.

We further control description quality by selecting only participants whose matching performance does not decrease across rounds within each dataset. Since repeated reference games typically become easier as participants establish common ground, this filter removes dialogues with unstable engagement or inconsistent task behavior. It also ensures that the selected dialogues are suitable for our multi-round follow-up studies, where we examine whether LVLMs can handle realistic longhorizon interactive visual grounding. This filtering yields 48 high-quality human dialogues.

Full Target Item Description Extraction Following Zeng et al. (2026), we use GPT-5.4 (OpenAI, 2025) to automatically extract object-level referring expressions for each image set in Table 1, except for Tangrams, which already includes manually curated object-level descriptions. This process yields 536 diverse, high-quality referring expressions across the four image sets, which we use in STATIC and FULL.

Partial Target Item Description Extraction To construct underspecified referring expressions, we prompt GPT-5.4 with the full descriptions for all target items in the same image set and ask it to extract the least informative attribute from each description. We use these extracted attributes as the initial descriptions in PARTIAL. This LLMas-a-judge procedure is designed not to find optimal shortened descriptions, but to minimize initial informativeness so that the Matcher must decide whether clarification is needed. We verify that the extracted attributes are lexically supported by the source full expressions (see Appendix C). Our experiments further validate this design: all models ask questions under PARTIAL, indicating that the extracted attributes are underspecified enough to elicit clarification (e.g., Figure 3 in Section 5).

## 4.3 Director Simulation

Because our focus is the Matcher, we use GPT-5.4 as a fixed simulated Director. This choice is motivated by prior evidence that GPT-5.2 reaches about 85% initial-round accuracy with human Matchers (Zeng et al., 2026). Fixing the Director makes information access more comparable across Matchers and protocols; Section 6 tests sensitivity to an alternative Director, showing comparable results.

The Director is always conditioned on the same human-produced full target description, defined in Section 4.2. In STATIC, we programmatically present this preconfigured full description to the Matcher one target at a time. In FULL and PAR-TIAL, the Matcher first receives a preconfigured full or underspecified description, respectively, and the GPT-5.4 Director is called only when the Matcher asks a clarification question. In NONE, no initial description is provided; instead, the Director answers the Matcher’s questions while being prompted with the corresponding full target description. When a question asks about information not explicitly stated in the description, the prompt instructs the Director to answer based only on the provided description and visual input. See Appendix F for the full Director prompts.

## 4.4 Matcher Simulation

Models We evaluate LVLMs from different model families and scales to simulate matchers: (1) five proprietary LVLMs, namely GPT-5.4, GPT-5.4- Mini, and GPT-5.5 (OpenAI, 2026a,b), and Gemini-3.1-Pro and Gemini-3.1-Flash-Lite (Google Deep-Mind, 2026); and (2) three open-weight LVLMs, namely Gemma-4-31B-IT (Google, 2026) and Qwen3-VL-8B/32B-Instruct (Bai et al., 2025).

Prompting We use protocol-specific system prompts while keeping the task overview fixed for both the Director and Matcher across conditions. In interactive settings, the Matcher returns a JSON object containing a natural-language utterance and any selections it makes, with a 100-point confidence score for each selection. In the static setting, the Matcher returns a JSON selection with a confidence score. See Appendix F for full prompts.

Simulation Scale Across the four datasets, we obtain 536 item–description pairs. Under the four interaction protocols in the main experiments, each LVLM matcher makes 2,144 target selections based on information from the simulated Director. Across eight LVLM matchers, this yields 17,152 selections. Including the follow-up studies in Section 6, which add another 21,696 selections, our experiments produce 38,848 total target selections for analysis.

![](images/729e6c213f66151a496345f69bc319988432b515ee6e9fb0eec06dfa1eff6b0a.jpg)  
Figure 3: Overall results across the four image sets. Top: average matcher accuracy. Bottom: average questions per target item. Colors denote the four interaction protocols; horizontal lines mark human and random baselines. The human line is a task-level reference, but not a protocol-matched baseline, since the interaction conditions differ.

## 4.5 Multi-Turn Dialogue Simulation

Visual Input Format For each image set, the Director is shown only the target items, while the Matcher is shown the full candidate set, which may include distractors. We concatenate the individual item images into a single indexed grid and provide the grid as one image input. Director indices follow the item-description order in the original dataset, while Matcher indices are randomized to prevent trivial index matching or data contamination.

Turn Taking The Director speaks first in STATIC, FULL, and PARTIAL, whereas the Matcher speaks first by initiating a yes/no question in NONE. In the interactive protocols, Matchers are allowed— but not required—to ask clarification questions and to revise earlier selections when later information changes their decision. Each turn contains one natural-language utterance capped at 50 words, and a Matcher may ask as many questions as needed across the 10-turn budget, which is empirically sufficient. If the turn budget is exhausted, the Matcher is prompted to make a target selection. The Director advances to the next target only after the Matcher selects. The final prediction sequence is computed from the Matcher’s finalized selections.

<table><tr><td rowspan=1 colspan=6>Model         Dogs    Baskets 1 Baskets 2 TangramsAll</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>–14.0**</td><td rowspan=1 colspan=1>-15.0**</td><td rowspan=1 colspan=1>–18.5***</td><td rowspan=1 colspan=1>–40.5***</td><td rowspan=1 colspan=1>–25.3***</td></tr><tr><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1>–26.0***</td><td rowspan=1 colspan=1> $. 1 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1>–20.8***</td><td rowspan=1 colspan=1>–42.9***</td><td rowspan=1 colspan=1>–27.3***</td></tr><tr><td rowspan=1 colspan=1>GPT-5.4-Mini</td><td rowspan=1 colspan=1>–40.0***</td><td rowspan=1 colspan=1>-30.0***</td><td rowspan=1 colspan=1>–32.1***</td><td rowspan=1 colspan=1>–70.8***</td><td rowspan=1 colspan=1>–47.7***</td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Pro</td><td rowspan=1 colspan=1>–7.0*</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>–8.3*</td><td rowspan=1 colspan=1>-6.0</td><td rowspan=1 colspan=1>–6.8***</td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Flash</td><td rowspan=1 colspan=1>-18.0***</td><td rowspan=1 colspan=1>-23.0**</td><td rowspan=1 colspan=1>–22.6***</td><td rowspan=1 colspan=1>–47.0***</td><td rowspan=1 colspan=1>–28.9***</td></tr><tr><td rowspan=1 colspan=1>Gemma-4-31B-IT</td><td rowspan=1 colspan=1>-37.0***</td><td rowspan=1 colspan=1>-31.0***</td><td rowspan=1 colspan=1>–27.4***</td><td rowspan=1 colspan=1>–44.6***</td><td rowspan=1 colspan=1>-36.7***</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-32B</td><td rowspan=1 colspan=1>–25.0***</td><td rowspan=1 colspan=1>–26.0**</td><td rowspan=1 colspan=1>–41.7***</td><td rowspan=1 colspan=1>–72.6***</td><td rowspan=1 colspan=1>–46.2***</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1>–42.0***</td><td rowspan=1 colspan=1>–54.0***</td><td rowspan=1 colspan=1>–61.9***</td><td rowspan=1 colspan=1>–72.0***</td><td rowspan=1 colspan=1>–61.4***</td></tr></table>

Table 2: Best paired mean LVLM–human matcher accuracy gap (%) across datasets, selected from the bestperforming interaction protocol for each model–dataset pair. The All column is computed over all examples from the four datasets. Darker red denotes larger deficits relative to human matchers, and bold marks the smallest gap within each dataset. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1$ , and $^ { * * * } p \leq . 0 0 1$ See Appendix D.1 for protocol-specific results.

## 5 Results

## 5.1 Interactive Visual Grounding Poses a Challenging Task

LVLM matchers significantly lag behind human matchers in interactive visual grounding. The upper panels of Figure 3 show that nearly all tested LVLM matchers underperform the corresponding task-level human baselines across the four image sets, with the largest gaps on the Tangram Image Set. Table 2 further shows that these gaps are often substantial and statistically significant under paired t-tests, especially when results are pooled across all datasets. The only case of human parity is Gemini-3.1-Pro on Basket Image Set 2 under PARTIAL, but no model outperforms the human baselines. Moreover, the consistent gaps under STATIC and FULL suggest that current LVLMs still struggle to comprehend descriptions drawn from high-accuracy human dialogues (Tang et al., 2024; Wang et al., 2025a; Zeng et al., 2026).

<table><tr><td>Model</td><td>STATIC</td><td>FULL</td><td> $\mathrm { P A R T I A L }$ </td><td>NONE</td></tr><tr><td>GPT-5.5</td><td>65.8</td><td> $\underline { { 6 9 . 5 } } _ { + 3 . 6 }$ </td><td> $\underline { { 7 1 . 4 } } _ { + 5 . 5 }$ </td><td> ${ \bf 5 9 . 7 _ { - 6 . 1 } }$ </td></tr><tr><td>GPT-5.4</td><td>66.1</td><td> $6 9 . 4 _ { + 3 . 4 * }$ </td><td> $6 5 . 5 _ { - 0 . 4 }$ </td><td> $4 3 . 2 _ { - 2 3 . 6 * }$ </td></tr><tr><td>GPT-5.4-Mini</td><td>47.9</td><td> $4 9 . 0 _ { + 0 . 6 }$ </td><td> $3 7 . 1 _ { - 1 1 . 5 * }$ </td><td> $1 3 . 5 _ { - 3 5 . 2 * }$ </td></tr><tr><td>Gemini-3.1-Pro</td><td>87.5</td><td> $\mathbf { 8 9 . 9 _ { + 2 . 9 } }$ </td><td> $\mathbf { 8 9 . 0 _ { + 2 . 3 } }$ </td><td> $5 6 . 7 _ { - 2 8 . 6 * }$ </td></tr><tr><td>Gemini-3.1-Flash</td><td>67.8</td><td> $6 5 . 3 _ { - 2 . 7 }$ </td><td> $5 4 . 5 _ { - 1 2 . 9 * }$ </td><td> $3 7 . 9 _ { - 3 0 . 4 * * }$ </td></tr><tr><td>Gemma-4-31B-IT</td><td>56.0</td><td> $6 0 . 0 _ { + 4 . 3 }$ </td><td> $5 8 . 4 _ { + 2 . 6 }$ </td><td> $2 6 . 4 _ { - 2 9 . 0 * * }$ </td></tr><tr><td>Qwen3-VL-32B</td><td>48.8</td><td> $5 0 . 5 _ { + 2 . 8 }$ </td><td> $4 4 . 0 _ { - 4 . 0 }$ </td><td> $2 3 . 9 _ { - 2 5 . 5 * }$ </td></tr><tr><td>Qwen3-VL-8B</td><td>33.6</td><td> $3 5 . 0 _ { + 1 . 6 }$ </td><td> $3 5 . 3 _ { + 1 . 2 }$ </td><td> $2 1 . 6 _ { - 1 2 . 7 * }$ </td></tr></table>

Table 3: Micro-averaged accuracy (%) by model and interaction protocol. Subscripts report paired mean differences relative to STATIC; green indicates gains and red indicates losses. The best-performing model within each protocol is bolded, and the second-best model is underlined. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1 , \mathrm { a n d } ^ { * * * } p \leq . 0 0 1$

The task is challenging because it requires both visual matching and effective information seeking. GPT-5.4-Mini consistently asks the fewest questions. It outperforms Qwen3-VL-8B under STATIC and FULL, but its advantage nearly disappears under PARTIAL and reverses under NONE (Table 3), where question asking matters more. This suggests that limited question asking can hurt when the task requires acquiring missing information. However, asking more questions alone is not sufficient. Gemini-3.1-Pro asks more questions than GPT-5.5, yet GPT-5.5 achieves higher overall accuracy under NONE. These contrasts suggest that success requires not only visual matching or more questions, but useful questions and effective integration of answers across turns.

## 5.2 Stronger Models Perform Better, but Question-Only Grounding Remains Hard

Model performance exhibits a clear hierarchy. Figure 3 and Table 3 show that larger models generally outperform their smaller counterparts from the same model family, suggesting benefits from scaling, and proprietary models tend to outperform open-weight models. Gemini-3.1-Pro is strongest overall, achieving the best performance under STATIC, FULL, and PARTIAL, with especially large gains on the Tangram Image Set. GPT-5.5 is also competitive, particularly under NONE, suggesting stronger question-driven information seeking and integration capabilities.

![](images/a197f8a18e640e7f4f532ec93712072b009179b6ff4f3fe28573139f5ee69f62.jpg)  
Figure 4: Protocol-level calibration. Points show modelspecific confidence bins from 10 evenly spaced bins between 0 and $1 0 0 ;$ point size is proportional to bin count. The dashed diagonal denotes perfect calibration.

LVLMs can benefit from interaction, but question-driven grounding remains challenging. Table 3 shows that FULL improves over STATIC for seven of the eight models, suggesting that optional follow-up questions can help refine or repair initial descriptions. Under PARTIAL, results are mixed: some models recover from underspecification, while others decline. The clearest limitation appears under NONE, where all models perform below STATIC despite asking the most questions. This suggests that current LVLMs struggle to construct and ground target representations through proactive question-driven dialogue.

## 5.3 LVLMs Are Not Yet Calibrated for Interactive Visual Grounding

Current LVLMs are sensitive to information availability, but remain overconfident. All models ask more questions when less initial information is available: NONE > PARTIAL > FULL $> \mathrm { S T A T I C }$ (lower panels of Figure 3). Figure 4 shows a similar shift in confidence: models tend to report lower confidence in more interaction-heavy protocols. However, most points still fall below the diagonal, indicating that reported confidence often exceeds empirical accuracy. This overconfidence is especially pronounced under PARTIAL and NONE, suggesting that LVLMs are less reliable at judging information sufficiency when the initial referring expression is underspecified or must be constructed through interaction. Overall, this suggests that LVLMs adjust their behavior to the amount of available information, but their confidence does not yet reliably reflect whether they have enough evidence for accurate grounding.

<table><tr><td rowspan="2">Reasoning</td><td colspan="2">GPT-5.4</td><td colspan="2">GPT-5.4-Mini</td></tr><tr><td>STATIC</td><td>FULL</td><td>STATIC</td><td>FULL</td></tr><tr><td>None</td><td>50.6</td><td>52.4</td><td>24.4</td><td>17.3</td></tr><tr><td>Medium</td><td> $5 2 . 4 _ { + 1 . 8 }$ </td><td> $6 0 . 7 _ { + 8 . 3 }$ </td><td> $2 6 . 2 _ { + 1 . 8 }$ </td><td> $3 0 . 4 _ { + 1 3 . 1 }$ </td></tr><tr><td>High</td><td> $4 8 . 8 _ { - 1 . 8 }$ </td><td> $6 1 . 3 _ { + 8 . 9 }$ </td><td> $1 5 . 5 _ { - 8 . 9 }$ </td><td> $3 1 . 0 _ { + 1 3 . 7 }$ </td></tr></table>

Table 4: Accuracy (%) under different reasoning-effort settings on tangrams. Subscripts report differences relative to default no reasoning.

Self-repairs are rare and inconsistently reliable. Revising a selection could indicate that a model recognizes and acts on uncertainty, but repairs occur only under FULL and PARTIAL and account for under 2% of matches. Among the 49 observed repairs, 26 (53.1%) correct a wrong selection, 5 (10.2%) change a correct selection to a wrong one, and 18 (36.7%) replace one wrong answer with another. Outcomes are strongly model-dependent: Gemini-3.1-Pro corrects 13 of its 17 revisions (76.5%), whereas GPT-5.4 corrects none of six and nevertheless raises its confidence by 11.2 points on average. Thus, models can occasionally exploit later evidence to repair an error, but revision remains too sparse and its confidence changes too inconsistent to serve as reliable error awareness. Full modeland protocol-level results appear in Appendix D.2.

## 5.4 Summary

Overall, interactive visual grounding remains challenging for current LVLMs. Models ask more questions when information is limited and can sometimes benefit from interaction, especially when follow-up questions refine or repair an initial description. However, they still lag behind human matchers, struggle with proactive question-driven dialogue, and remain poorly calibrated for the task.

## 6 Follow-Up Studies

To test whether our conclusions depend on description source, test-time reasoning, repeatedreference structure, Director choice, or the original visual contexts, we conduct five follow-up studies. We (i) replace human descriptions with

![](images/b44e6c157dc812151550de0914140eba879bf53044be72bedf99e5ea9b906c06.jpg)

![](images/913258384864c2cf5400833171b8935b2ef5a938a15fef690156d01c662889d9.jpg)  
Figure 5: Accuracy (%) on AI-generated and humangenerated descriptions across models and interaction protocols, averaged over all four datasets.

GPT-5.4-generated descriptions across all Matchers and protocols; (ii) vary reasoning effort for GPT-5.4 and GPT-5.4-Mini on tangrams; (iii) evaluate three LVLMs over four rounds of repeated reference; (iv) replace GPT-5.4 with Gemini-3.1-Flash as the Director on Basket Set 2 and Tangrams; and (v) evaluate five LVLM Matchers (i.e., GPT-5.4, GPT-5.4-Mini, Gemini-3.1-Flash, Gemma-4-31B-IT, and Qwen3-VL-8B) on six public face image sets with GPT-5.4-generated descriptions (Holland et al., 2018), each containing 12 images of one individual with varied facial expressions.

LVLMs perform better with AI-generated descriptions, but result patterns remain comparable. Figure 5 shows that LVLMs generally achieve higher accuracy with AI-generated descriptions than with human-generated descriptions, especially under STATIC and FULL. This suggests that AI-generated descriptions are easier for LVLMs to interpret, while human referring expressions pose a distinct challenge. Importantly, however, the overall protocol-level patterns remain similar: STATIC and FULL are strongest, PARTIAL is more variable, and NONE remains most difficult.

Reasoning helps in interactive settings, but scaling plateaus quickly. Reasoning yields little benefit in STATIC, but substantially improves accuracy in FULL (Table 4). This suggests that stronger reasoning helps models gather and synthesize visual evidence through conversation. However, gains largely plateau from medium to high reasoning, indicating that additional test-time compute alone is not sufficient. Reasoning also modestly improves calibration, with similar plateaus from medium to high effort, as discussed in Appendix E.2.

![](images/e23c0ff732b9d68db0849eb5cfb945d88c9bf5faa6456b6c01eddac54adb137e.jpg)

Figure 6: Accuracy (%) across rounds on all four image sets under FULL. See Appendix E.3 for STATIC results.
<table><tr><td colspan="3">A. Alternative Director robustness Director / comparison FULL PARTIAL</td></tr><tr><td>GPT-5.4 Gemini-3.1-Flash ∆ (Gemini – GPT) Rank correlation ρ</td><td>51.5 50.4 -1.1 .929</td><td>48.3 44.8 -3.5</td></tr><tr><td colspan="3">.976 B. Face-domain robustness Metric STATIC FULL PARTIAL NONE</td></tr><tr><td colspan="3">Accuracy (%) 48.7 ∆ vs. STATIC</td></tr></table>

Table 5: Aggregate alternative-Director (A) and facedomain (B) results. Panel A averages across Matchers, datasets, and description sources; Panel B across Matchers and six face sets.

LVLMs can benefit from repeated interaction, but the gap to human matchers remains. Figure 6 shows that models can improve over rounds under FULL. Results for STATIC show similar patterns and are reported in Appendix E.3 for space reasons. This suggests that LVLMs can handle increasingly efficient referring expressions that emerge through entrainment. However, these improvements are not consistent across datasets, and human matchers continue to outperform LVLMs. While Gemini-3.1-Pro approaches human baselines in some settings, long-horizon interactive visual grounding remains a broad challenge for LVLMs.

Matcher rankings are stable across Director models. Table 5 A shows that replacing the GPT-5.4 Director with Gemini-3.1-Flash reduces mean

Matcher accuracy by only 1.1 points in FULL (51.5% to 50.4%), but by 3.5 points in PARTIAL (48.3% to 44.8%). The larger PARTIAL decrease is consistent with the Director carrying more responsibility when the initial description is underspecified. Despite these absolute changes, Matcher rankings remain highly correlated across Directors (Spearman’s ρ = .929 for FULL and .976 for PAR-TIAL). Thus, interaction quality affects accuracy, especially under underspecification, but the central Matcher hierarchy is not an artifact of a single simulated Director (Appendix E.4).

The protocol-level pattern extends to finegrained face matching. Across six face sets with AI-generated descriptions, Table 5B shows that FULL remains within 0.9 points of STATIC (47.8% vs. 48.7%), whereas PARTIAL and NONE fall 6.8 and 18.0 points below STATIC, respectively. Thus, optional clarification largely preserves performance when an informative description is available, but interaction does not fully compensate as initial information is removed. This graded pattern matches Figure 5 and extends the main protocol-level observation to a distinct fine-grained domain, confirming the core finding in Section 5 (Appendix E.5).

## 7 Conclusion

We introduce a controlled framework for evaluating interactive visual grounding beyond one-shot matching. Across four human-grounded visual contexts, current LVLMs perform significantly below task-level human references. Interaction can help when follow-up questions refine or repair an initial description, but models consistently struggle with proactive question-driven grounding. They are also poorly calibrated, often reporting confidence that exceeds accuracy. These findings are further validated across follow-up studies using AIgenerated descriptions, test-time scaling, exposure to repeated interaction, an alternative simulated Director, and a fine-grained face context. Overall, interactive visual grounding remains an important challenge for LVLMs, with potential implications for future embodied human–AI interaction.

## Acknowledgments

This material is based upon work supported by the National Science Foundation (NSF) under Grant No. 2125295 and by a seed grant from Stony Brook University. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the NSF.

We are grateful for support from the Institute for Advanced Computational Science (IACS) at Stony Brook University. We also thank the three anonymous reviewers for their valuable feedback.

## Limitations

Simulated Directors and Director Dependence. We use simulated rather than human Directors for scale and experimental control. Repeating FULL and PARTIAL with Gemini-3.1-Flash produces highly stable Matcher rankings, reducing concern that the main conclusions are specific to GPT-5.4. That said, this robustness check does not establish equivalence to human Directors, and individual model–protocol outcomes can still vary with the Director.

Attribution. The four protocols vary how much information must be acquired interactively. They are designed to create a controlled way to separate visual matching from information seeking and question-driven grounding, but do not uniquely isolate question selection, answer integration, dialogue memory, and visual matching etc. Our claim is also limited: these protocols increase the need for information seeking and synthesis beyond static visual matching, rather than attributing each failure to one precise subcomponent, which we leave to future studies.

Visual Context Coverage. The main testbed contains four object-level contexts, and the face followup adds another fine-grained within-category setting. Neither covers cluttered scenes, spatial or relational grounding, or open-world visual search. Our conclusions therefore concern proactive interactive grounding in controlled, fine-grained matching contexts rather than visual grounding in all settings.

Constrained Interaction under NONE. The Director provides yes/no answers under NONE to create a controlled information-seeking setting, following de Vries et al. (2017). Open-ended Director answers may support different questioning strategies and should be studied separately.

## Ethical Considerations

Data Use and Privacy. This work evaluates LVLMs on interactive visual grounding using existing research datasets and public or previously released visual materials. We do not collect new human-subject data. The face-domain robustness study uses public research images only for withinset visual matching; we neither infer identity nor predict sensitive attributes. The main evaluated images involve object-level visual contexts such as dogs, baskets, and abstract tangrams, which reduces risks related to privacy or demographic stereotyping.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, and 8 others. 2022. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems.

Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, and Wanli Ouyang. 2024. MT-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7421–7454, Bangkok, Thailand. Association for Computational Linguistics.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Susan E. Brennan and Herbert H. Clark. 1996. Conceptual pacts and lexical choice in conversation. Journal of Experimental Psychology: Learning, Memory, and Cognition, 22(6):1482–1493.

Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S. H. Gary Chan, and Hongyang Zhang. 2024. Revisiting referring expression comprehension evaluation in the era of large multimodal models. Preprint, arXiv:2406.16866.

Herbert H. Clark and Deanna Wilkes-Gibbs. 1986. Referring as a collaborative process. Cognition, 22(1):1– 39.

Harm de Vries, Florian Strub, Sarath Chandar, Olivier Pietquin, Hugo Larochelle, and Aaron Courville. 2017. Guesswhat?! visual object discovery through multi-modal dialogue. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Qihua Dong, Kuo Yang, Lin Ju, Handong Zhao, Yitian Zhang, Yizhou Wang, Huimin Zeng, Jianglin

Lu, and Yun Fu. 2026. Ref-adv: Exploring MLLM visual reasoning in referring expression tasks. In The Fourteenth International Conference on Learning Representations.

Jiazhan Feng, Qingfeng Sun, Can Xu, Pu Zhao, Yaming Yang, Chongyang Tao, Dongyan Zhao, and Qingwei Lin. 2023. MMDialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7348–7363, Toronto, Canada. Association for Computational Linguistics.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, and Angeliki Lazaridou and 1332 others. 2025. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

Google. 2026. Gemma 4 model overview. https://ai. google.dev/gemma/docs/core/model\_card\_4. Accessed: 2026-05-19.

Google DeepMind. 2026. Gemini 3.1 Pro model card. https://deepmind.google/models/ model-cards/gemini-3-1-pro/. Accessed: 2026-05-19.

Chi Han, Xin Liu, Haodong Wang, Shiyang Li, Jingfeng Yang, Haoming Jiang, Zhengyang Wang, Qingyu Yin, Liang Qiu, Changlong Yu, Yifan Gao, Zheng Li, Bing Yin, Jingbo Shang, and Heng Ji. 2025. Can language models follow multiple turns of entangled instructions? In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 25445–25460, Suzhou, China. Association for Computational Linguistics.

Robert D. Hawkins, Michael C. Frank, and Noah D. Goodman. 2020. Characterizing the dynamics of learning in repeated reference games. Cognitive Science, 44(6).

Catherine A. C. Holland, Natalie C. Ebner, Tian Lin, and Gregory R. Samanez-Larkin. 2018. Emotion identification across adulthood using the dynamic faces database of emotional expressions in younger, middle aged, and older adults. Cognition and Emotion, 33(2):245–257.

Yilun Hua and Yoav Artzi. 2024. Talk less, interact better: Evaluating in-context conversational adaptation in multimodal LLMs. In First Conference on Language Modeling.

Saki Imai, Mert Inan, Anthony B. Sicilia, and Malihe Alikhani. 2025. Measuring how (not just whether) VLMs build common ground. In Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing - Natural Language

Processing in the Generative AI Era, pages 441–451, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.

Cameron R. Jones, Agnese Lombardi, Kyle Mahowald, and Benjamin K. Bergen. 2026. Llms and people both learn to form conventions – just not with each other. Preprint, arXiv:2602.08208.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787– 798, Doha, Qatar. Association for Computational Linguistics.

Robert M. Krauss and Sam Glucksberg. 1969. The development of communication: Competence as a function of age. Child Development, 40(1):255.

Wai-Chung Kwan, Xingshan Zeng, Yuxin Jiang, Yufei Wang, Liangyou Li, Lifeng Shang, Xin Jiang, Qun Liu, and Kam-Fai Wong. 2024. MT-eval: A multiturn capabilities evaluation benchmark for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20153–20177, Miami, Florida, USA. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23, pages 611–626, New York, NY, USA. Association for Computing Machinery.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2026. LLMs get lost in multi-turn conversation. In The Fourteenth International Conference on Learning Representations.

Young-Jun Lee, Byung-Kwan Lee, Jianshu Zhang, Yechan Hwang, Byungsoo Ko, Han-Gyu Kim, Dongyu Yao, Xuankun Rong, Eojin Joo, Seung-Ho Han, Bowon Ko, and Ho-Jin Choi. 2025. Multiverse: A multi-turn conversation benchmark for evaluating large vision and language models. Preprint, arXiv:2510.16641.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. 2014. Microsoft COCO: Common Objects in Context, pages 740–755. Springer International Publishing.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc.

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024.

MMDU: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for LVLMs. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 11–20.

OpenAI. 2025. Gpt-5 is here. https://openai.com/ gpt-5/. Accessed: 2026-05-20.

OpenAI. 2026a. Introducing GPT-5.4. https:// openai.com/index/introducing-gpt-5-4/. Accessed: 2026-05-19.

OpenAI. 2026b. Introducing GPT-5.5. https:// openai.com/index/introducing-gpt-5-5/. Accessed: 2026-05-19.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, and Jeff Belgum and 262 others. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 2641– 2649.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Michael F Schober and Herbert H Clark. 1989. Understanding by addressees and overhearers. Cognitive Psychology, 21(2):211–232.

Omar Shaikh, Hussein Mozannar, Gagan Bansal, Adam Fourney, and Eric Horvitz. 2025. Navigating rifts in human-LLM grounding: Study and benchmark. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 20832–20847, Vienna, Austria. Association for Computational Linguistics.

Mohit Shridhar, Dixant Mittal, and David Hsu. 2020. Ingress: Interactive visual grounding of referring expressions. Int. J. Rob. Res., 39(2–3):217–232.

Zineng Tang, Lingjun Mao, and Alane Suhr. 2024. Grounding language in multi-perspective referential communication. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 19727–19741, Miami, Florida, USA. Association for Computational Linguistics.

Yunjie Tian, Tianren Ma, Lingxi Xie, and Qixiang Ye. 2025. Chatterbox: Multimodal referring and grounding with chain-of-questions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7401–7409.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Keqin Chen, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2024. Cogvlm: Visual expert for pretrained language models. In Advances in Neural Information Processing Systems, volume 37, pages 121475– 121499. Curran Associates, Inc.

Zhengxiang Wang, Weiling Li, Panagiotis Kaliosis, Owen Rambow, and Susan Brennan. 2025a. LVLMs are bad at overhearing human referential communication. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 16758–16782, Suzhou, China. Association for Computational Linguistics.

Zhengxiang Wang, Veronika Makarova, Zhi Li, Jordan Kodner, and Owen Rambow. 2025b. LLMs can perform multi-dimensional analytic writing assessments: A case study of L2 graduate-level academic English writing. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8637–8663, Vienna, Austria. Association for Computational Linguistics.

Linhui Xiao, Xiaoshan Yang, Xiangyuan Lan, Yaowei Wang, and Changsheng Xu. 2026. Toward visual grounding: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(3):2749– 2771.

Haochen Xue, Feilong Tang, Ming Hu, Yexin Liu, Qidong Huang, Yulong Li, Chengzhi Liu, Zhongxing Xu, Chong Zhang, Chun-Mei Feng, Yutong Xie, Imran Razzak, Zongyuan Ge, Jionglong Su, Junjun He, and Yu Qiao. 2025. MMRC: A large-scale benchmark for understanding multimodal large language model in real-world conversation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22477–22503, Vienna, Austria. Association for Computational Linguistics.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. 2016. Modeling context in referring expressions. Preprint, arXiv:1608.00272.

Peter Zeng, Weiling Li, Amie Paige, Zhengxiang Wang, Panagiotis Kaliosis, Dimitris Samaras, Gregory Zelinsky, Susan Brennan, and Owen Rambow. 2026. Lvlms and humans ground differently in referential communication. Preprint, arXiv:2601.19792.

Hanbo Zhang, Jie Xu, Yuchen Mo, and Tao Kong. 2023. Invig: Benchmarking interactive visual grounding with 500k human-robot interactions. Preprint, arXiv:2310.12147.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

## Appendix Table of Contents

A LVLM Details 13   
B Expected Calibration Error 13   
C Extracted Partial Description Validation 13   
D Results 14   
D.1 LVLM and Human Matcher Per  
formance Gap across Interaction   
Protocols 14   
D.2 Self-Repair Analysis 14   
E Follow-Up Studies 15   
E.1 AI-Generated Descriptions 15   
E.2 Reasoning Effort 17   
E.3 Multi-Round Repeated References 18   
E.4 Alternative Director Robustness 18   
E.5 Face-Domain Robustness 18   
F Prompts for Visual Grounding Experi  
ments 21   
F.1 Task Overview 21   
F.2 Matcher Output Format Instructions 21   
F.3 STATIC Protocol 22   
F.4 FULL Protocol . 22   
F.5 PARTIAL Protocol . 23   
F.6 NONE Protocol 24   
G Prompts for Referring Expression Pro  
cessing 25   
G.1 Referring Expression Extraction 25   
G.2 Partial Description Extraction . 26   
G.3 Object Description Generation 27   
H Dataset Details 27

## A LVLM Details

Table 6 lists the exact LVLM variants used in our experiments. We evaluate proprietary GPT and

Gemini models through their respective APIs, using the latest available variants at the time of experimentation. For open-weight models, we use the Hugging Face checkpoints listed in the table and run local inference with vLLM (Kwon et al., 2023). Unless otherwise specified, we use the default decoding configuration across models.

## B Expected Calibration Error

We compute Expected Calibration Error (ECE) to summarize the alignment between the Matcher’s stated confidence and empirical accuracy. For each final selection, the Matcher is prompted to report a confidence score $c _ { j } \in [ 0 , 1 0 0 ]$ , and we define correctness as whether the selected item matches the target item at the corresponding position. We partition predictions into 10 evenly spaced confidence bins over [0, 100]: [0, 10), [10, 20), . . . , [90, 100].

For each bin $B _ { \ell }$ , we compute the average empirical accuracy and the average stated confidence of all predictions in that bin. ECE is then computed as the weighted average absolute gap between accuracy and confidence across bins:

$$
\mathrm { E C E } = \sum _ { \ell = 1 } ^ { 1 0 } \frac { | { \cal B } _ { \ell } | } { N } \left| \operatorname { a c c } ( { \cal B } _ { \ell } ) - \operatorname { c o n f } ( { \cal B } _ { \ell } ) \right| ,
$$

where $N$ is the total number of predictions, $| B _ { \ell } |$ is the number of predictions in bin $B _ { \ell } , \operatorname { a c c } ( B _ { \ell } )$ is the percentage of correct predictions in the bin, and conf $( B _ { \ell } )$ is the average stated confidence in the bin. Lower ECE indicates better calibration.

In the main text, we report calibration primarily through calibration plots rather than ECE. This is because ECE is strongly negatively correlated with accuracy in our setting (Spearman’s $\rho = - 0 . 9 6 )$ making it less informative as a separate measure of calibration. Intuitively, models with lower accuracy tend to have larger confidence–accuracy gaps even when their confidence patterns are qualitatively similar. Calibration plots therefore provide a more interpretable view of whether models are overconfident or underconfident across confidence levels.

## C Extracted Partial Description Validation

Metric definitions. Let d denote the full description and let $A = \{ a _ { 1 } , \ldots , a _ { n } \}$ denote the generated attribute list for an item. Text is normalized by lowercasing and tokenized with the regular expression [A-Za-z0-9]+, so punctuation is ignored and only alphanumeric tokens are retained.

<table><tr><td>Model</td><td>Provider / Family</td><td>Variant used (snapshot / HF repo)</td><td>Inference backend</td></tr><tr><td>GPT-5.5 (OpenAI, 2026b)</td><td>OpenAI</td><td>gpt-5.5-2026-04-23</td><td>OpenAI API</td></tr><tr><td>GPT-5.4 (OpenAI, 2026a)</td><td>OpenAI</td><td> $\mathtt { g p t - 5 . 4 - 2 0 2 6 - } \theta 3 - \theta 5$ </td><td>OpenAI API</td></tr><tr><td>GPT-5.4-Mini (OpenAI, 2026a)</td><td>OpenAI</td><td>gpt-5.4-mini-2026-03-17</td><td>OpenAI API</td></tr><tr><td>Gemini-3.1-Pro (Google DeepMind, 2026)</td><td>Google Gemini</td><td>gemini-3.1-pro-preview</td><td>Gemini API</td></tr><tr><td>Gemini-3.1-Flash (Google DeepMind, 2026)</td><td>Google Gemini</td><td>gemini-3.1-flash-lite-preview</td><td>Gemini API</td></tr><tr><td>Gemma-4-31B-IT (Google, 2026)</td><td>Gemma</td><td>google/gemma-4-31b-it</td><td>vLLM</td></tr><tr><td>Qwen3-VL-32B (Bai et al., 2025)</td><td>Qwen</td><td>Qwen/Qwen3-VL-32B-Instruct</td><td>vLLM</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>Qwen</td><td>Qwen/Qwen3-VL-8B-Instruct</td><td>vLLM</td></tr></table>

Table 6: LVLM variants used in our experiments. API models are identified by their provider-facing model names, and open-weight models are identified by their Hugging Face repository IDs. Open-weight models are run locally with vLLM.

For each attribute $a _ { i } .$ , we compute an attribute support score $s ( a _ { i } , d )$ as the maximum of exact substring match and token overlap:

$$
s ( a _ { i } , d ) = \mathrm { m a x } \left( \mathrm { s u b s t r } ( a _ { i } , d ) , \mathrm { o v e r l a p } ( a _ { i } , d ) \right) .
$$

An attribute is marked as supported if $s ( a _ { i } , d ) \geq$ 0.45.

The attribute support rate for an item is the fraction of generated attributes whose support score exceeds this threshold:

$$
\operatorname { s u p p o r t \_ r a t e } ( A , d ) = { \frac { \sum _ { i = 1 } ^ { n } { \mathcal { k } } ^ { 2 } [ s ( a _ { i } , d ) \geq 0 . 4 5 ] } { n } } .
$$

Description coverage measures how much of the original full description is recovered by the generated attributes:

$$
\operatorname { c o v e r a g e } ( A , d ) = { \frac { \sum _ { w \in V } \operatorname* { m i n } \left( c _ { A } ( w ) , c _ { d } ( w ) \right) } { | T ( d ) | } } .
$$

For the selected underspecified expression, we compute the least-attribute support score as $s ( a _ { \mathrm { l e a s t } } , d )$ , which verifies that the initial partial expression is grounded in the original description despite being intentionally underspecified.

Validation results. The extracted attributes are strongly supported by the original full descriptions. Across all items, the mean attribute support rate is 0.998, indicating that nearly all generated attributes can be matched back to the source description. The mean attribute support score is 0.980, and the mean description coverage is 0.895, suggesting that the decomposition preserves most of the descriptive content while keeping the attributes grounded in the original expression. The selected least-informative attributes are also well supported, with a mean least-attribute support score of 0.981.

These results suggest that our procedure produces source-grounded partial descriptions: the initial expressions used in PARTIAL are intentionally underspecified, but not hallucinated or detached from the original human descriptions.

## D Results

## D.1 LVLM and Human Matcher Performance Gap across Interaction Protocols

Tables 7–10 report the full LVLM–human matcher accuracy gaps for each interaction protocol, complementing the best-over-protocol summary in Table 2. Across STATIC, FULL, and PARTIAL, nearly all LVLMs significantly lag behind human matchers, although the size of the gap varies by model, dataset, and protocol. The only non-significant cases occur for Gemini-3.1-Pro, including one case in which it reaches parity with human matchers.

Under NONE, all models significantly underperform human matchers. This comparison should be interpreted with caution, however, because the human reference data were not collected under an explicit NONE setting. Instead, the human interactions more closely resemble a mixture of the other protocols, depending on how much detail human Directors provided in their object descriptions and how well Matchers understood those descriptions.

## D.2 Self-Repair Analysis

Tables 12 and 11 summarize self-repair behavior for models that revise their selections at least once. Self-repairs are relatively rare, accounting for no more than 2% of the 536 matches in any interaction protocol, and occur only under FULL and PARTIAL, with no observed cases under NONE. When models do revise, outcomes vary substantially: some repairs change initially wrong selections to correct ones, especially for Gemini-3.1-Pro, while others introduce errors or replace one wrong answer with another, as in about half of GPT-5.5 cases. Confidence changes are also mixed, suggesting that revisions do not always correspond to increased certainty. Overall, self-repair is observable but limited in current LVLM matchers.

<table><tr><td>Model</td><td>Dogs</td><td>Baskets 1</td><td></td><td>Baskets 2 Tangrams</td><td>All</td></tr><tr><td>GPT-5.5</td><td> $- 2 3 . 0 ^ { * * * }$ </td><td> $- 1 6 . 0 ^ { * * }$ </td><td> $- 2 0 . 2 ^ { * * * }$ </td><td> $- 5 7 . 7 ^ { * * * }$ </td><td> $- 3 0 . 9 ^ { * * * }$ </td></tr><tr><td>GPT-5.4</td><td> $- 2 9 . 0 ^ { * * * }$ </td><td> $- 2 2 . 0 ^ { * * * }$ </td><td> $- 2 3 . 8 ^ { * * * }$ </td><td> $- 4 4 . 6 ^ { * * * }$ </td><td> $- 3 0 . 6 ^ { * * * }$ </td></tr><tr><td>GPT-5.4-Mini</td><td> $- 4 1 . 0 ^ { * * * }$ </td><td> $- 3 0 . 0 ^ { * * * }$ </td><td> $- 4 5 . 8 ^ { * * * }$ </td><td> $- 7 0 . 8 ^ { * * * }$ </td><td> $- 4 8 . 8 ^ { * * * }$ </td></tr><tr><td>Gemini-3.1-Pro</td><td> $\mathbf { - 1 2 . 0 ^ { * } }$ </td><td> $\mathbf { - 9 . 0 ^ { * * * } }$ </td><td> ${ \bf - 1 0 . 7 ^ { * } }$ </td><td>-6.0</td><td> $\mathbf { - 9 . 2 ^ { * * * } }$ </td></tr><tr><td>Gemini-3.1-Flash</td><td> $- 1 8 . 0 ^ { * * * }$ </td><td> $- 2 3 . 0 ^ { * * }$ </td><td> $- 2 2 . 6 ^ { * * * }$ </td><td> $\cdot 4 7 . 0 ^ { * * * }$ </td><td> $- 2 8 . 9 ^ { * * * }$ </td></tr><tr><td>Gemma-4-31B-IT</td><td> $- 4 1 . 0 ^ { * * * }$ </td><td> $- 4 0 . 0 ^ { * * * }$ </td><td> $- 3 2 . 7 ^ { * * * }$ </td><td> $- 4 8 . 8 ^ { * * * }$ </td><td> $- 4 0 . 7 ^ { * * * }$ </td></tr><tr><td>Qwen3-VL-32B</td><td> $- 3 1 . 0 ^ { * * * }$ </td><td> $- 3 9 . 0 ^ { * * * }$ </td><td> $- 4 1 . 7 ^ { * * * }$ </td><td> $- 7 2 . 6 ^ { * * * }$ </td><td> $- 4 7 . 9 ^ { * * * }$ </td></tr><tr><td>Qwen3-VL-8B</td><td> $- 4 7 . 0 ^ { * * * }$ </td><td> $- 5 6 . 0 ^ { * * * }$ </td><td> $- 6 9 . 0 ^ { * * * }$ </td><td> $- 7 3 . 8 ^ { * * * }$ </td><td> $- 6 3 . 1 ^ { * * * }$ </td></tr></table>

Table 7: Mean LVLM–human matcher accuracy gap (%) under the STATIC protocol. Darker red denotes larger deficits relative to human matchers, and bold marks the smallest gap within each dataset. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1$ $^ { * * * } p \leq . 0 0 1$

<table><tr><td rowspan=1 colspan=2>Model            Dogs</td><td rowspan=1 colspan=4>Baskets 1Baskets 2Tangrams     All</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1> $- 1 8 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $\cdot 1 5 . 0 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 1 8 . 5 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 5 1 . 2 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 7 . 2 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1> $- 2 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 1 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 0 . 8 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 2 . 9 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 7 . 3 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>GPT-5.4-Mini</td><td rowspan=1 colspan=1> $- 4 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 2 . 1 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 8 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 7 . 7 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Pro</td><td rowspan=1 colspan=1> $\mathbf { - 7 . 0 ^ { * } }$ </td><td rowspan=1 colspan=1>-3.0</td><td rowspan=1 colspan=1> $\mathbf { \cdot 8 . 3 ^ { \ast } }$ </td><td rowspan=1 colspan=1> $z . 7 ^ { * }$ </td><td rowspan=1 colspan=1> $\mathbf { - 6 . 8 ^ { \ast \ast \ast } }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Flash</td><td rowspan=1 colspan=1> $- 1 9 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 4 . 4 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 8 . 2 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 1 . 4 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemma-4-31B-IT</td><td rowspan=1 colspan=1> $- 3 7 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 1 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 7 . 4 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 6 . 7 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-32B</td><td rowspan=1 colspan=1> $- 2 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 6 . 0 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 4 2 . 3 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 9 . 8 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 6 . 2 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1> $- 4 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 6 . 1 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 7 6 . 2 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 1 . 7 ^ { * * * }$ </td></tr></table>

Table 8: Mean LVLM–human matcher accuracy gap (%) under the FULL protocol. Darker red denotes larger deficits relative to human matchers, and bold marks the smallest gap within each dataset. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1$ $^ { * * * } p \leq . 0 0 1$

<table><tr><td rowspan=1 colspan=6>Model            DogsBaskets 1Baskets 2 Tangrams     All</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1> $- 1 4 . 0 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 1 5 . 0 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 2 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 1 . 1 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 5 . 3 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1> $- 2 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 3 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 1 . 2 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>GPT-5.4-Mini</td><td rowspan=1 colspan=1> $- 6 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 5 . 6 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 9 . 6 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Pro</td><td rowspan=1 colspan=1>-7.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1> $\mathbf { - 8 . 3 ^ { * } }$ </td><td rowspan=1 colspan=1> ${ \bf - 1 3 . 1 ^ { * * } }$ </td><td rowspan=1 colspan=1> $- 7 . 7 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Flash</td><td rowspan=1 colspan=1> $- 3 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 2 9 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 0 . 5 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 0 . 7 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 2 . 2 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemma-4-31B-IT</td><td rowspan=1 colspan=1> $- 4 1 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 4 . 5 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 4 4 . 6 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 8 . 3 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-32B</td><td rowspan=1 colspan=1> $- 3 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 3 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> ${ } ^ { 5 0 . 6 ^ { * * * } }$ </td><td rowspan=1 colspan=1> $- 8 1 . 5 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 2 . 7 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1> $- 5 3 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 4 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 1 . 9 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 7 2 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 1 . 4 ^ { * * * }$ </td></tr></table>

Table 9: Mean LVLM–human matcher accuracy gap (%) under the PARTIAL protocol. Darker red denotes larger deficits relative to human matchers, and bold marks the smallest gap within each dataset. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1$ $^ { * * * } p \leq . 0 0 1$

<table><tr><td rowspan=1 colspan=6>Model            Dogs1Baskets 1 Baskets 2Tangrams      All</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1> ${ \bf - 2 6 . 0 ^ { * * * } }$ </td><td rowspan=1 colspan=1> $- 2 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $\mathbf { - 4 0 . 5 ^ { * * * } }$ </td><td rowspan=1 colspan=1> $\mathbf { - 3 7 . 0 ^ { * * * } }$ </td></tr><tr><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1> $- 4 8 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 8 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 1 . 8 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 3 . 5 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { G P T } { - } 5 . 4 { - } \mathrm { M i n i }$ </td><td rowspan=1 colspan=1> $- 7 0 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 1 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 7 . 5 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 9 . 9 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 3 . 2 ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Pro</td><td rowspan=1 colspan=1> $- 3 3 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $\mathbf { \cdot 1 9 . 0 ^ { \ast \ast } }$ </td><td rowspan=1 colspan=1> $\mathbf { - 4 4 . 0 ^ { * * * } }$ </td><td rowspan=1 colspan=1> $- 5 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $\cdot 4 0 . 0 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemini-3.1-Flash</td><td rowspan=1 colspan=1> $- 5 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 3 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 7 . 1 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 7 . 3 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 5 8 . 8 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1>Gemma-4-31B-IT</td><td rowspan=1 colspan=1> $- 6 5 . 0 ^ { * * }$ </td><td rowspan=1 colspan=1> $- 6 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 4 . 9 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 2 . 7 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 0 . 3 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { Q w e n 3 - V L } \mathrm { - } 3 2 \mathrm { B }$ </td><td rowspan=1 colspan=1> $- 6 3 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 6 5 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 8 . 6 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 9 . 8 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 2 . 8 ^ { * * * }$ </td></tr><tr><td rowspan=1 colspan=1> ${ \mathrm { Q w e n } } 3 { \mathrm { - } } \mathrm { V L } { \mathrm { - } } 8 \mathrm { B }$ </td><td rowspan=1 colspan=1> $- 6 1 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 6 . 0 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 8 3 . 9 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 5 . 6 ^ { * * * }$ </td><td rowspan=1 colspan=1> $- 7 5 . 1 ^ { * * * }$ </td></tr></table>

Table 10: Mean LVLM–human matcher accuracy gap (%) under the NONE protocol. Darker red denotes larger deficits relative to human matchers, and bold marks the smallest gap within each dataset. Significance is assessed with paired t-tests: $^ { * } p \leq . 0 5 , ^ { * * } p \leq . 0 1$ $^ { * * * } p \leq . 0 0 1$

<table><tr><td>Model</td><td>FULL</td><td>PARTIAL</td><td>Total</td></tr><tr><td>GPT-5.5</td><td>7</td><td>10</td><td>17</td></tr><tr><td>GPT-5.4</td><td>4</td><td>2</td><td>6</td></tr><tr><td>GPT-5.4-Mini</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Gemini-3.1-Pro</td><td>9</td><td>8</td><td>17</td></tr><tr><td>Gemini-3.1-Flash</td><td>1</td><td>4</td><td>5</td></tr><tr><td>Gemma-4-31B-IT</td><td>2</td><td>1</td><td>3</td></tr></table>

Table 11: Distribution of self-repair cases by model and interaction protocol. Counts indicate the number of observed self-repairs under FULL and PARTIAL. We did not observe self-repairs under NONE.

## E Follow-Up Studies

This section provides additional implementation details and results for the follow-up studies conducted in Section 6.

## E.1 AI-Generated Descriptions

Experimental Procedure For experiments on AI-generated descriptions, we replace humangenerated descriptions with GPT-5.4-generated ones and rerun the main experiments across all models and interaction protocols. We repeat each experiment three times using the same description map (see below) but with shuffled description order. This controls for variation in item ordering while isolating the effect of description source, allowing us to test whether the main results depend on the use of human referring expressions.

<table><tr><td>Model</td><td>Count</td><td>Wrong → Correct</td><td>Correct → Wrong</td><td>Wrong → Wrong</td><td>∆ Confidence</td></tr><tr><td>GPT-5.5</td><td>17</td><td>47.1</td><td>5.9</td><td>47.1</td><td>-14.9</td></tr><tr><td>GPT-5.4</td><td>6</td><td>0.0</td><td>50.0</td><td>50.0</td><td>+11.2</td></tr><tr><td>GPT-5.4-Mini</td><td>1</td><td>100.0</td><td>0.0</td><td>0.0</td><td>-15.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>17</td><td>76.5</td><td>0.0</td><td>23.5</td><td>-4.3</td></tr><tr><td>Gemini-3.1-Flash</td><td>5</td><td>60.0</td><td>20.0</td><td>20.0</td><td>+1.0</td></tr><tr><td>Gemma-4-31B-IT</td><td>3</td><td>33.3</td><td>0.0</td><td>66.7</td><td>-16.7</td></tr></table>

Table 12: Self-repair outcomes by model. Count denotes the number of self-repair cases. Transition columns report the percentage of repairs that change an initially wrong selection to correct, correct selection to wrong, or wrong selection to another wrong answer. ∆ Confidence denotes the mean change in confidence after repair; positive values indicate increased confidence, and negative values indicate decreased confidence.

![](images/a8d60e886a41a74fff41dc0df9c361ae0fb65ed6f664b5475d16a6909a35f80f.jpg)  
Figure 7: Overall results across the four image sets from the follow-up experiments with AI-generated descriptions. Top: average matcher accuracy. Bottom: average questions per target item. Colors denote the four interaction protocols; horizontal lines mark human and random baselines.

Additional Result Discussion Figure 7 reports the full results for each dataset. The overall patterns are comparable to those in the main experiments with human-generated descriptions (Figure 3; Section 5). LVLMs also remain overconfident and undercalibrated even with AI-generated descriptions, as shown in Figure 8.

For easier comparison, we aggregate results across the four datasets and visualize the differences in accuracy and question-asking behavior between AI- and human-generated descriptions in Figure 9. Figure 9 shows that LVLMs generally achieve higher accuracy with AI-generated descriptions than with human-generated descriptions, especially under STATIC and FULL. Models also tend to ask fewer questions under FULL, suggesting that AI-generated descriptions are easier for LVLMs to interpret. This contrast indicates that human referring expressions pose a distinct challenge, even when they are sufficiently informative for human matchers. Importantly, however, the overall protocol-level patterns remain similar: STATIC and FULL are strongest, PARTIAL is more variable, and NONE remains difficult.

<table><tr><td>Reasoning</td><td>GPT-5.4</td><td>GPT-5.4-Mini</td></tr><tr><td>None</td><td>0.6</td><td>0.2</td></tr><tr><td>Medium</td><td> $0 . 8 _ { + 0 . 2 }$ </td><td> $0 . 1 _ { - 0 . 1 }$ </td></tr><tr><td>High</td><td> $0 . 9 _ { + 0 . 3 }$ </td><td> $0 . 0 _ { - 0 . 2 }$ </td></tr></table>

Table 13: Average number of questions per target item under FULL for different reasoning-effort settings. Subscripts report differences relative to the default noreasoning setting; green indicates increases and red indicates decreases.

![](images/e9a4aa951cb78a04f3c066476ddf0b46a42d98e8100fc8ddc303a08a97be22e5.jpg)  
Figure 8: Protocol-level calibration from the followup experiments with AI-generated descriptions. Points show model-specific confidence bins from 10 evenly spaced bins between 0 and 100; point size is proportional to bin count. The dashed diagonal denotes perfect calibration.

![](images/8ca42855720cfbdc150cddd55e27374bb5b93ae7e217ba7743f7729c23a48f83.jpg)  
Figure 9: Differences in LVLM outcomes and questionasking behavior between AI- and human-generated descriptions, averaged across all four datasets. Bars show $\Delta = \mathrm { A I }$ -generated − human-generated descriptions for accuracy and questions per item.

![](images/07bcd23f32ca394457a49aeb6ba2153959fbfbb6894f402658dbfa549e904b7f.jpg)  
Figure 10: Calibration under different reasoning-effort settings for GPT-5.4 and GPT-5.4-Mini. Each point represents a confidence bin, with the x-axis showing mean reported confidence and the y-axis showing empirical accuracy. Point size indicates bin size, colors denote reasoning effort, and the dashed diagonal marks perfect calibration.

## E.2 Reasoning Effort

Reasoning-effort setup. Default reasoning effort is model-specific rather than uniform across GPT models. GPT-5.5 uses medium reasoning effort by default, whereas GPT-5.4 defaults to no reasoning. We therefore vary the reasoning effort of GPT-5.4 and GPT-5.4-Mini through the official Responses API to isolate the effect of test-time reasoning.

Reasoning effort changes question asking modestly. Table 13 shows that higher reasoning effort slightly increases question asking for GPT-5.4 under FULL, while GPT-5.4-Mini asks very few questions across settings. Together with Table 4, this suggests that reasoning improves interactive performance partly by supporting information seeking, but not simply by increasing question volume.

Reasoning effort slightly improves calibration, but gains plateau from medium to high. Figure 10 shows that higher reasoning effort generally brings more calibration bins closer to the diagonal, especially when accounting for bin sizes, indicating slightly better alignment between confidence and accuracy. However, the shift from medium to high reasoning is small, suggesting that scaling reasoning effort does not reliably improve calibration. Most bins still fall below the diagonal, so both models remain overconfident.

![](images/d4562301bf379dc6b5059f9420b8ae9ae6d2cb2be92db14c86596bfecf9bb004.jpg)  
Figure 11: Accuracy (%) across rounds on the four image sets under STATIC.

![](images/1cd52048ae1a056f1aadcecba9273cfcb92e38e69bab323984812382d9ad6e3a.jpg)  
Figure 12: Average number of questions per item across rounds on the tangrams under FULL.

## E.3 Multi-Round Repeated References

Per-dataset accuracy results under STATIC. Figure 11 shows model performance on tangrams across rounds under STATIC. Models improve matching performance even without the ability to ask questions, confirming that they can comprehend the increasingly efficient referring expressions generated by humans over repeated interaction.

Question asking under FULL. Figure 12 shows the average number of questions per item across rounds under FULL. Question rates are generally stable or decrease slightly over rounds for most models and datasets.

Model calibration across rounds. Figure 13 shows round-level calibration across datasets, aggregating results from STATIC and FULL because the two protocols exhibit broadly similar patterns. Models remain imperfectly calibrated across rounds: many confidence bins fall below the diagonal, indicating overconfidence. Calibration does not consistently improve with repeated exposure, even when accuracy increases over rounds. This suggests that LVLMs can sometimes benefit from repeated references without reliably improving their uncertainty estimates.

## E.4 Alternative Director Robustness

Experimental procedure. To test whether Matcher results depend on one simulated information source, we replace GPT-5.4 with Gemini-3.1- Flash as the Director and rerun FULL and PARTIAL on Basket Image Set 2 and Tangrams. We evaluate both human- and GPT-5.4-generated initial descriptions and otherwise preserve the original prompts, candidate views, and turn budget.

Results. Table 14 shows that individual model– protocol combinations vary with the Director, as expected in a collaborative task, but Matcher rankings remain highly stable. Rank correlations between results with GPT-5.4 and Gemini-3.1-Flash Directors are ρ = .929 under FULL and ρ = .976 under PARTIAL. GPT-5.4 is generally the stronger Director, yet the main protocol-level patterns and Matcher hierarchy do not depend on using GPT-5.4 alone. The correlations are computed over Matcherlevel accuracies aggregated across the two datasets and description sources.

## E.5 Face-Domain Robustness

Experimental procedure. We evaluate five representative Matchers on six image sets from the public FACES database (Holland et al., 2018). Each set contains 12 images of one individual displaying varied facial expressions. Because these image sets do not include human interaction data, we generate initial descriptions with GPT-5.4 and evaluate the same four protocols. Director and Matcher indices are randomized independently as in the main study. This experiment contains no protocol-matched human baseline and is intended as a robustness check in another finegrained within-category context.

Results and scope. Table 15 reproduces the main aggregate pattern: STATIC and FULL are comparable (48.7% and 47.8%), PARTIAL is lower (41.9%), and NONE is most difficult (30.7%). The result extends the protocol-level observation beyond dogs, baskets, and tangrams, but faces remain a controlled within-category matching context; this study does not establish generality to cluttered scenes, spatial relations, or open-world visual search.

![](images/f837d085f12dec9c1784cbe1cd3a3ad139f36c821ff2f21099a6ab1e69c382fc.jpg)

![](images/b40e98c45da931f6e1c55df7051b73bdac048848d9ae472b844ca12f31becbf9.jpg)  
Figure 13: Round-level calibration across datasets in the four-round follow-up experiments. Points show model specific confidence bins from 10 evenly spaced bins between 0 and 100; point size is proportional to bin count. The dashed diagonal denotes perfect calibration.

<table><tr><td rowspan="2">Matcher</td><td colspan="2">GPT-5.4 Director</td><td colspan="2">Gemini-3.1-Flash Director</td></tr><tr><td>FULL</td><td>PARTIAL</td><td>FULL</td><td>PARTIAL</td></tr><tr><td>GPT-5.5</td><td>59.5</td><td>61.3</td><td>59.5</td><td>55.7</td></tr><tr><td>GPT-5.4</td><td>62.8</td><td>58.3</td><td>57.4</td><td>51.2</td></tr><tr><td>GPT-5.4-Mini</td><td>39.3</td><td>28.6</td><td>38.7</td><td>28.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>81.5</td><td>83.6</td><td>86.3</td><td>85.1</td></tr><tr><td>Gemini-3.1-Flash</td><td>58.0</td><td>43.8</td><td>58.3</td><td>37.5</td></tr><tr><td>Gemma-4-31B-IT</td><td>54.5</td><td>54.8</td><td>53.9</td><td>46.4</td></tr><tr><td>Qwen3-VL-32B</td><td>33.3</td><td>28.3</td><td>33.0</td><td>33.0</td></tr><tr><td>Qwen3-VL-8B</td><td>23.2</td><td>27.4</td><td>16.1</td><td>21.1</td></tr></table>

Table 14: Accuracy (%) with GPT-5.4 and Gemini-3.1-Flash Directors, aggregated across Basket Image Set 2 and Tangrams and across human- and AI-generated descriptions. Matcher rankings across Directors remain highly correlated (ρ = .929 for FULL and .976 for PARTIAL).

<table><tr><td>Matcher</td><td>STATIC</td><td>FULL</td><td>PARTIAL</td><td>NONE</td></tr><tr><td>GPT-5.4</td><td>53.7</td><td>55.1</td><td>56.9</td><td>52.3</td></tr><tr><td>GPT-5.4-Mini</td><td>44.9</td><td>47.7</td><td>26.9</td><td>16.2</td></tr><tr><td>Gemini-3.1-Flash</td><td>55.6</td><td>50.0</td><td>38.0</td><td>17.6</td></tr><tr><td>Gemma-4-31B-IT</td><td>55.1</td><td>54.6</td><td>54.2</td><td>49.4</td></tr><tr><td>Qwen3-VL-8B</td><td>34.3</td><td>31.5</td><td>33.3</td><td>20.8</td></tr><tr><td>Overall</td><td>48.7</td><td>47.8</td><td>41.9</td><td>30.7</td></tr></table>

Table 15: Accuracy (%) on six face image sets with AI-generated descriptions. Each image set contains 12 images. There is no protocol-matched human baseline for this robustness experiment.

## F Prompts for Visual Grounding Experiments

This section reports the prompts used in the study for the visual grounding experiments.

## F.1 Task Overview

We reuse the same task overview prompt for both the Director and Matcher across all interaction protocols to maximize the comparability across the four interaction protocols.

```markdown
### Task Overview
You will participate in a visual reference task involving two roles: Director and Matcher. The two
participants may not share the same visual context.
Both participants are shown randomized sets of indexed visual items. These visual items may be
objects, scenes, or other image-based stimuli. The participants share a subset of target
items, but the Matcher may also see additional distractor items that are not visible to the
Director.
The goal is for the Director and Matcher to collaborate so that the Matcher can identify each
target item as accurately as possible.
The task may consist of multiple rounds. Please strictly follow the instructions for your assigned
role across all rounds.
```

## F.2 Matcher Output Format Instructions

Across all interaction protocols, Matcher is prompted to produce a JSON object containing the target item index and its confidence level (%) rated from 0 to 100.

The output format instruction for the STATIC protocol is as follows.

```markdown
### Output Format Instruction
Output your response in JSON format using the following structure:
{
"target_item_index": 5,
"confidence": 75
}
**Field Definitions**
- "target_item_index": the index of the item you are selecting as the match for the identified
target item.
"confidence": your confidence level (%) in the selection, rated from 0 (not confident at all) to
100 (most confident).
```

The output format instruction for the other three interactive protocols (FULL, PARTIAL, and NONE) is as follows.

```markdown
### Output Format Instruction
Output your response in JSON format using the following structure:
{
"utterance": "Your natural, concise utterance to the Director.",
"selections": [
{
"nth_item": 2,
"target_item_index": 5,
"confidence": 75
},
```

- Do not include duplicate entries for the same nth\_item in a single response.

- If you have enough information to make a selection for the current item, include that item in "   
selections".

]   
}   
\*\*Field Definitions\*\*   
"utterance" should contain your natural language utterance you say to the Director. Do not   
mention indices or positional cues here.   
"selections" should be a list of items you are selecting based on the Director's responses. Each   
item selection should include:   
- "nth\_item": the order of the target item being identified (starting from 1 for the first item   
identified in the current round).   
"target\_item\_index": the index of the item you are selecting as the match for the identified   
target item.   
- "confidence": your confidence level (%) in the selection, rated from 0 (not confident at all)   
to 100 (most confident).

## \*\*Output Rules\*\*

- If you are revising one or more earlier selections, include each revised item as a separate   
entry in "selections".

- If you are both revising earlier selections and selecting the current item, include all of those   
updates in "selections" in the same response.

- Unless you are revising an earlier choice, "selections" should just contain only one entry for   
the most recently identified item.

- If you revise a previous selection, your "utterance" may briefly acknowledge the correction, but   
it must not disclose any item index or positional information.

## F.3 STATIC Protocol

Since there is no interaction between the Director and Matcher under this protocol, we simulate the Director model by programmatically sending the Matcher a full item description at a time without actually prompting a Director LVLM. The Matcher system prompt is given below.

### Task Overview   
{Task Overview}   
### Role Instructions   
You are the Matcher. In this role, you must carefully analyze the Director's descriptions and   
choose the most likely match for each target item based on the visual items visible to you.   
### Output Format Instruction   
{Matcher Output Format Instruction}

## F.4 FULL Protocol

## Director system prompt.

```markdown
### Task Overview
{Task Overview}
### Role Instructions
You are the Director. In this role, you must describe each visual item visible to you from the
lowest index to the highest index one at a time and answer any questions the Matcher may have.
```

Analyze the Matcher's question carefully based on your visual context and answer the question to   
the best of your ability.   
### Communication Guidelines   
- Describe exactly one target item at a time.   
- Do not move on to the next item until the Matcher signals to proceed. Wait for them to signal   
when they are ready.   
- Use natural, human-like, simple, and clear language that supports efficient communication.   
- Try your best to make each description unambiguous and sufficiently informative for the Matcher   
to identify the related target item.   
- Keep each description within 50 words, favoring shorter descriptions when clarity is maintained.   
- Do not reference display position, display order, neighboring-item relations, or explicit item   
indices. You may describe within-item spatial structure when it is part of the item's   
appearance.

## Matcher system prompt.

<table><tr><td>### Task Overview</td></tr><tr><td>{Task Overview}</td></tr><tr><td>### Role Instructions</td></tr><tr><td>You are the Matcher. In this role, you must carefully analyze the Director&#x27;s descriptions and ask any clarification questions if needed to identify each target item based on the visual items visible to you.</td></tr><tr><td>### Communication Guidelines</td></tr><tr><td>- In each turn, you are allowed to ask a question or signal to proceed to the next target item,</td></tr><tr><td>but keep it no longer than 50 words. - If you ask a question or a confirmation, you must not internally select any target item until</td></tr><tr><td>the Director provides an answer to your question. - If you signal to proceed, you must have internally selected a target item in your output.</td></tr><tr><td>- Use concise, natural, human-like language and never mention item indices or positional cues in</td></tr><tr><td>your natural-language utterance. - You may ask as many questions as needed to clarify your understanding of the target items, but</td></tr><tr><td>avoid asking redundant or irrelevant questions.</td></tr><tr><td>- Once you have identified a target item, you can move on to the next one until all the target items have been identified.</td></tr><tr><td>- You can ask questions about your previous selection(s), in case you want to double check or revise them.</td></tr><tr><td>- Do not reference display position, display order, neighboring-item relations, or explicit item</td></tr><tr><td>indices. You may describe within-item spatial structure when it is part of the item&#x27;s</td></tr><tr><td></td></tr><tr><td>appearance.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>### Output Format Instruction</td></tr></table>

## F.5 PARTIAL Protocol

## Director system prompt.

```markdown
### Task Overview
{Task Overview}
### Role Instructions
You are the Director. In this role, you must describe each visual item visible to you from the
lowest index to the highest index one at a time and answer any questions the Matcher may have.
```

\### Communication Guidelines

## Matcher system prompt.

\### Task Overview

{Task Overview}

\### Role Instructions

\### Communication Guidelines

\### Output Format Instruction

{Matcher Output Format Instruction}

## F.6 NONE Protocol

## Director system prompt.

\### Task Overview

{Task Overview}

\### Role Instructions

You are the Director. In this role, you must answer the Matcher's questions using only one of the   
following responses: "Yes," "No," or "Unclear."   
Analyze the Matcher's question carefully based on your visual context and answer the question to   
the best of your ability.   
### Communication Guidelines   
- Just output "Yes," "No," or "Unclear" without any additional explanation or information in each   
turn.   
- If the Matcher's response indicates the target item index they selected, just respond with "   
Unclear."

## Matcher system prompt.

```markdown
### Task Overview
{Task Overview}
### Role Instructions
You are the Matcher. In this role, you must ask only yes or no questions to the Director to
identify each target item based on the visual items visible to you. There are a total of \
$num_targets target items for you to identify in each round.
### Communication Guidelines
- In each turn, you are only allowed to ask a clear and concise question at a time, no longer than
50 words.
- Avoid asking questions that require the Director to provide information beyond "Yes," "No," or "
Unclear."
Use concise, natural, human-like language and never mention item indices or positional cues in
your natural-language utterance.
You may ask up to \$max_questions questions as needed to clarify your understanding of the
target items, and avoid asking redundant or irrelevant questions.
Once you have identified a target item, you can move on to the next one until all the \
$num_targets target items have been identified.
You can ask questions about your previous selection(s), in case you want to double check or
revise them.
- For each question, make clear which target item it refers to, such as "the first target item," "
the fifth target item," and so on.
### Output Format Instruction
{Matcher Output Format Instruction}
```

## G Prompts for Referring Expression Processing

## G.1 Referring Expression Extraction

Prompt used for the automatic extraction of referring expressions for each target visual referent produced in a referential conversation. Words within “<>" denote placeholders.

This prompt is from Zeng et al. (2026) and we only use it for the dog and basket datasets. The tangrams dataset already comes with manually curated object descriptions for each target object.

This is an extractive task.   
You will be given a transcript of a conversation between two participants engaged in a   
collaborative object-matching task. There are exactly <num\_objects> target objects. One   
participant (the describer) describes each target object, and the other participant (the   
matcher) attempts to identify them.   
Your task is to extract the descriptive phrases used by the describer for each target object.

- Extract phrases verbatim from the transcript.   
- Do not extract the whole utterance, only the descriptive phrases.   
- Exclude disfluencies, fillers, and false starts (e.g., "um", "uh", "like").   
Do not paraphrase or infer missing information.   
- Each object may have one or multiple descriptive phrases.   
Return the results in the following JSON format:   
{   
"object\_#1": "descriptive phrases for object 1",   
"object\_#2": "descriptive phrases for object 2",   
"object\_#<num\_objects>": "descriptive phrases for object <num\_objects>"   
}   
Example description phrases:   
- doesn't have handle, tip of it is thicker than rest of body, brownish color, weaves are in   
squares if you look at it directly   
- half circle, no handles, top tip of it is a little bit thicker than rest of body   
tip which is a little bit thicker than rest of body   
tip that is a little bit larger than body, looks a little bit thicker   
Transcript:   
<transcript>   
Output only the JSON object. Do not include any additional text or explanations.

## G.2 Partial Description Extraction

Given a description map in JSON format, convert it into a new JSON object with the same top-level   
keys.   
1. "full": the original full description as a string, unchanged.   
2. "attributes": a list of atomic attributes extracted from the full description.   
3. "least": the least informative attribute for this visual item, selected from its "attributes"   
list.   
Attribute extraction rules:   
- Each attribute should be expressed in natural language.   
- Each attribute must describe only one visual feature.   
- Keep attributes as close as possible to the original wording; verbatim if possible.   
- Do not infer attributes that are not explicitly stated in the description.   
Do not treat missing attributes as negative evidence.   
- Remove duplicate attributes within the same item.   
- Split compound descriptions into separate atomic attributes.   
- Preserve relative/comparative attributes when explicitly stated, such as "tallest", "shortest",   
"second shortest", or "shorter than 9".   
- Resolve coreferences when extracting attributes, so that each attribute is self-contained and   
does not rely on pronouns or other references to other attributes.   
- If there is only one attribute in the description, simply reuse the full description as the   
single attribute.   
Least attribute selection rules:   
- For each item, choose exactly one attribute from that item's "attributes" list.   
- Choose the attribute that is least informative for identifying the item, taking into account all   
attributes across all items in the input map.   
- Do not choose an attribute that gives the gestalt impression of the whole item; instead, choose   
an attribute that describes a specific visual feature.   
- Prefer a generic, common, or weakly distinguishing attribute over an attribute that is   
distinctive or rare across items.   
- If several attributes are similarly generic, choose the one that would be least helpful on its   
own for distinguishing the item from the other items.   
- Do not invent, rewrite, or combine attributes for "least"; it must exactly match one string in   
that item's "attributes" list.

Return only valid JSON. Do not include explanations, markdown, or comments.   
Input format:   
{   
"image\_id": "full natural-language description"   
}   
Output format:   
{   
"image\_id": {   
"full": "original full natural-language description",   
"attributes": [   
"attribute 1",   
"attribute 2"   
],   
"least": "attribute 2"   
}   
}

## G.3 Object Description Generation

We use the following prompt to generate full descriptions for each object across the four image sets with GPT-5.4. To mirror the human Director setting, the model is shown only the target images. We prompt it to produce a description map that can be used directly in our experimental pipeline.

You are creating a full description map for a visual reference task.   
The task involves two roles: a Director and a Matcher. The Director sees a set of target visual   
items and must describe them so the Matcher can identify the same items among possible   
distractors. Your job is to write the Director-style initial descriptions for one round.   
You will receive all director images for the round in a single message. Each image is labeled with   
its filename only so the output can be keyed correctly.   
Return only valid JSON: an object mapping each exact filename to one short natural-language   
description.   
Description rules:   
- Describe every provided image exactly once.   
- Use the exact filenames as JSON keys.   
- Each description must be fewer than 50 tokens.   
- Use natural, human-like, simple, and clear language.   
- Make each description unambiguous and sufficiently informative for a Matcher to identify the   
item among similar items.   
- Describe the item itself: shape, pose, orientation, silhouette, distinctive parts, and within  
item spatial structure when useful.   
- Do not reference display position, display order, neighboring-item relations, explicit item   
indices, file paths, or filenames in the descriptions.   
- Do not say "image", "picture", "item number", or similar display-oriented labels in the   
descriptions.   
- Do not include markdown, comments, explanations, or extra keys.

## H Dataset Details

Figures 14, 15, 16, and 17 show the four object image sets used in our experiments. These sets span real-world objects (dogs and baskets) and abstract tangrams, and differ in the number of target and distractor items. For the basket and dog sets, the Director sees only the target items, while the Matcher sees the full candidate set. For the tangram set, both players see the full set. In all experiments, item orders for Director follow the item-description order in the original dataset, while Matcher’s item orders are randomized to prevent trivial index matching or data contamination.

We obtain the image sets and corresponding human–human interaction data from the GitHub repositories associated with the cited papers.

![](images/dfe4a5dd282b13d54dfeeff05079562453900d2c369fe5eb0ff05520bdeceade.jpg)  
Figure 14: Basket image set from Wang et al. (2025a). The first 10 baskets are target items visible to the Director; the Matcher sees the full set.

![](images/d8aedcd7d264b659b8c29bd23239715ad0794b0d754e7426a35e2736091971b2.jpg)  
Figure 15: Dog image set from Wang et al. (2025a). The first 10 dogs are target items visible to the Director; the Matcher sees the full set.

![](images/0c645c455aebd7a3f2ed69d6ee08f3e2f2322073866466f0dac22677de7bff6b.jpg)  
Figure 16: Basket image set from Zeng et al. (2026). The first 12 baskets are target items visible to the Director; the Matcher sees the full set.

![](images/7ccd38a2bcaf6412701aecdc9753e9388878afe06d3b679a4493f60d662fd498.jpg)  
Figure 17: Tangram image set from Hawkins et al. (2020). Both the Director and Matcher see the full set; item orders are randomized independently for the two views in our experiments.