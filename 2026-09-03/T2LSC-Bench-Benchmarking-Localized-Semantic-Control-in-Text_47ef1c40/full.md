# T2LSC-Bench: Benchmarking Localized Semantic Control in Text-to-Image Generation

Yan Wang, Xinyi Hou, Weiguo Lin, Junjun Si, and Siwei Ma Fellow, IEEE

Abstract—Recent text-to-image models have become increasingly capable of rendering explicit text, but reliable localized text control requires more than generating the correct string. In visual content creation tasks such as product labeling, signage, and interface design, designers typically require the target text be rendered within a designated text-bearing region without altering the predefined subject identity or surrounding scene semantics. We refer to violations of this requirement as target-text-associated semantic leakage, in which target-text semantics are expressed through non-textual visual content beyond the designated anchor. Existing visual-text benchmarks primarily evaluate readability, spelling accuracy, and layout, leaving this form of semantic leakage largely unexamined. We introduce T2LSC-BENCH (Textto-Image Localized Semantic Control Benchmark), a controlled diagnostic benchmark that tests whether T2I models adhere to the localized-text contract. T2LSC-BENCH comprises 50 seed subjects and 1,200 controlled prompt cases per model, yielding 7,160 evaluated images from 7,200 planned generations across six models. Its factorized design independently varies semantic relation, scene openness, prompt mode, and language. A dualbranch evaluation protocol combines OCR–VLM text verification with structured VLM semantic judgments to separately measure Text-at-Anchor Accuracy (TAA), Semantic Subject Preservation (SSP), Semantic Leakage Rate (SLR), and Conditional Semantic Leakage Rate (cSLR). Under the stress-test conditions, aggregate SLR increases from 1.2% to 18.1% and cSLR from 1.3% to 18.2%, whereas TAA decreases only slightly from 91.4% to 90.9%. This indicates that accurate text rendering does not, by itself, guarantee local containment of target-text semantics. Explicit anti-leakage prompting is associated with a reduction in SLR from 16.6% to 8.4% without degrading rendering accuracy, suggesting partial but model-dependent mitigation. A stratified human validation study on 420 images further demonstrates strong agreement between the automatic protocol and adjudicated human annotations. The project resources are available at https://github.com/LLMSecResearch/T2LSC-Bench.

Index Terms—Text-to-image generation, semantic leakage, visual text generation, benchmark, multimodal evaluation.

## I. INTRODUCTION

Recent text-to-image models have made substantial progress in image fidelity and instruction following [1], [2], as well as in explicit text rendering [3], [4]. In practical visual content creation tasks such as storefront signs, device nameplates, and product packaging, target text is typically confined to a designated text-bearing region of a particular subject. These tasks require models not only to render the target text accurately within the designated region, but also to keep its semantic influence strictly local without altering the subject identity or surrounding scene conditions independently specified in the prompt. Such fine-grained local controllability is important for producing multiple variants of advertising and design assets, generating controlled synthetic data [5], and creating personalized visual content [6]. We refer to this requirement as an explicit localized-text contract and define its violation as target-text-associated semantic leakage, in which targettext semantics are expressed through non-textual visual content beyond the designated anchor.

![](images/4cb3e591028c7ed067c67c446d53aae6390f150683960746842717d52522e0eb.jpg)  
Fig. 1. Examples of localized text rendering and target-text-associated semantic leakage

However, adhering to this contract is non-trivial. Target text contains both a visual form that must be rendered within the designated region and semantic information that may influence image generation. When the semantics of the target text are inconsistent with the established visual context, including the subject identity, function, and surrounding scene, the model may not treat the text solely as a local rendering constraint, and may alter the subject or scene to make the target text appear more plausible within the overall image. Fig. 1 illustrates this problem in three localized text-rendering scenarios. This raises an important question that remains underexplored: under an explicit localized-text contract, when target text is rendered correctly, can its semantic influence remain confined to the designated region?

Prior studies have identified related forms of semantic interference in generative models. Visual text generation methods may produce the visual concept denoted by a word instead of correctly rendering its glyphs [7], while multi-entity generation may transfer semantic attributes from one visual entity to another [8]. Neither line of work examines the setting studied here, in which the target text is rendered correctly at the designated anchor, yet its semantics propagate beyond that anchor to alter the text-bearing subject or surrounding nontextual content.

![](images/652cb362549c0bbea5c513ad1fe1a22676b5de24eaa68510c51c3ddfc3d6861d.jpg)  
Fig. 2. Overview of T2LSC-BENCH for benchmarking localized semantic control in text-to-image generation.

Despite the practical importance of semantic containment, existing evaluations are not designed to measure it directly. Benchmarks and methods for text-rich image generation primarily assess whether text is readable, correctly spelled, properly placed, and robustly rendered under complex prompts [9]. Related evaluations of subject preservation and conditional control focus on identity retention or local controllability [10]. These evaluation settings are important, but they do not explicitly separate text rendering accuracy from semantic containment. Consequently, a model may achieve high textrendering accuracy even when the rendered text alters the subject or surrounding non-textual scene.

To fill this gap, we present T2LSC-BENCH (Text-to-Image Localized Semantic Control Benchmark), a controlled diagnostic benchmark designed to stress-test whether text-toimage models can confine target-text semantics to a designated text anchor. T2LSC-BENCH selects subjects with stable visual identities and default functions, assigns a natural text anchor to each subject, and pairs every seed with one aligned and two conflicting target texts. It further varies scene openness, prompt mode, and language to enable controlled analysis of semantic leakage. The overall construction, generation, and evaluation pipeline of T2LSC-BENCH is illustrated in Fig. 2. The benchmark comprises 50 seed subjects and 1,200 prompt cases per model, yielding 7,200 planned generations across six models, of which 7,160 are available for evaluation. T2LSC-

BENCH measures a model’s ability to follow a specific instruction in which the user has localized the text and required the subject and non-textual context to be preserved. This contrasts with open-ended creative generation, where globally adapting the scene to accommodate new text is considered a reasonable behavior.

We further introduce a dual-branch evaluation protocol that separates anchor-text correctness from semantic containment. The text-rendering branch combines character-level OCR evidence with anchor-aware VLM verification. The semanticevaluation branch uses a blinded VLM judge to assess subject identity preservation and target-text-associated non-textual evidence outside the designated anchor. The protocol reports Text-at-Anchor Accuracy (TAA), Semantic Subject Preservation (SSP), Semantic Leakage Rate (SLR), and Conditional Semantic Leakage Rate (cSLR). A stratified human validation study on 420 images shows high agreement between the automatic protocol and adjudicated human annotations.

Experiments on six text-to-image models reveal a clear boundary of localized semantic control. When the target text matches the predefined subject context, models generally preserve both the localized rendering and the surrounding visual semantics. Under the stress-test conditions, however, SLR increases from 1.2% to 18.1% and cSLR from 1.3% to 18.2%, whereas TAA remains nearly unchanged at 91.4% and 90.9%. These results show that a model may follow the local text-rendering instruction while failing to isolate the semantic influence of the rendered text.

Contributions. In summary, we make the following contributions:

• We introduce T2LSC-BENCH, a controlled benchmark for evaluating localized semantic control under an explicit localized-text contract. Cross-domain target texts serve as diagnostic stress probes for revealing and quantifying target-text-associated semantic leakage. It comprises 50 seed subjects, 1,200 prompt cases per model, and 7,200 planned generations across six text-to-image models, with a factorized design that enables controlled comparison.

• We establish a dual-branch evaluation protocol that combines OCR–VLM text verification, blinded VLM-based semantic assessment, and human validation to separately measure anchor-text correctness and semantic containment.

• We benchmark six mainstream text-to-image models and demonstrate that stress-test cases expose substantial semantic leakage while having only a limited effect on text-rendering accuracy. We further analyze the effects of prompt mode, scene openness, and language.

## II. BACKGROUND AND RELATED WORK

Visual Text Generation and Evaluation. Visual text generation methods improve local textual fidelity through characteraware representations, glyph guidance, and spatial control [3], [11]–[13]. Subsequent work extends these capabilities to multilingual generation, layout planning, glyph-aligned encoding, text inpainting, and less constrained scenes [14]– [18]. TextCrafter further targets multi-region text rendering in complex scenes [19]. Corresponding benchmarks evaluate readability, spelling, layout, prompt complexity, and rendering robustness [9], [20]. However, these studies primarily assess whether the requested string is rendered correctly, rather than whether its semantic influence remains confined to the designated text anchor.

Subject Preservation, Compositionality, and Semantic Leakage. Subject-driven generation methods preserve identity through learned representations, subject-specific adaptation, and image-based conditioning [6], [10], [21]. RefVNLI further evaluates subject preservation together with textual alignment [22]. Localized generation methods improve region-specific control through explicit spatial conditions and contextual constraints [23]–[26]. Related compositional approaches address entity omission and prompt-structure misalignment [27]–[29], as well as incorrect attribute binding and unintended concept interactions [30]–[32]. T2I-CompBench and ConceptMix evaluate these failures across attributes, relations, and multiple concepts [33], [34]. FreeText examines interference between glyph generation and word-associated visual concepts, while DeLeaker studies semantic transfer between visual entities [7], [8]. ALE-Bench evaluates attribute leakage in image editing, whereas SemVarBench measures responses to controlled semantic variations [35], [36]. In contrast, T2LSC-BENCH isolates cases in which the requested text is correctly rendered at a designated anchor, yet its semantics alter the text-bearing subject or surrounding non-textual content.

Semantic Evaluation of Generated Images. Automatic evaluation has progressed from global image–text similarity toward structured semantic verification [37]. TIFA and GenEval assess prompt fidelity through question answering or objectlevel checks [38], [39], while HEIM, Gecko, and VQAScore broaden evaluation dimensions and compositional assessment [40]–[42]. These methods primarily measure general prompt– image alignment rather than the containment of rendered-text semantics. Our protocol therefore evaluates anchor-text accuracy, subject preservation, and non-textual semantic leakage separately.

![](images/d75801fe7cd50376a8c06860989899619a23be93fb10b065957df77f50cfa792.jpg)  
Fig. 3. Conceptual boundary of target-text-associated semantic leakage.

## III. BENCHMARK CONSTRUCTION

T2LSC-BENCH is a controlled diagnostic benchmark designed to stress-test whether text-to-image models can confine the semantics of target text to a designated text anchor. Its factorized design supports separate analysis of local text rendering, subject identity preservation, and target-text-associated changes to non-textual visual content under comparable generation conditions.

## A. Problem Definition

Each benchmark case specifies a text-bearing subject, its predefined identity and default context, a designated text anchor, and a target text. The subject definition is fixed before target-text assignment and serves as the reference for interpreting the generated image. The benchmark operationalizes an explicit localized-text contract: the target string is to be rendered within the anchor, and the subject identity and nontextual context are to be preserved. We define target-textassociated semantic leakage as the occurrence of visible nontextual cues outside the designated anchor that are semantically linked to the target text but cannot be naturally attributed to the predefined subject context.

This definition is deliberately normative about the contract but neutral about generation in general. A model that globally adapts a scene to accommodate new text under an underspecified prompt is not wrong; T2LSC-BENCH tests the narrower, practically important ability to respect an explicit localization instruction. Fig. 3 highlights two important distinctions. First, subject preservation and semantic leakage are separate properties: target-text-associated cues may appear on an otherwise recognizable subject or in its surrounding scene. Second, not every generation error constitutes semantic leakage. Extra or repeated text, target-related content confined to the designated anchor, cues expected in the default subject context, and subject degradation unrelated to the target text are not counted as semantic leakage. Text rendering, subject preservation, and semantic leakage are operationalized separately in Section IV-B.

![](images/16ef1939216b2ecdf0aef0fa09f0816e2227644833c5aac7100ec21030839d51.jpg)  
Fig. 4. Factorized case instantiation and quality control in T2LSC-BENCH.

## B. Seed and Text-Anchor Design

Seed selection emphasizes both the observability of semantic leakage and the reliability of its evaluation. Each seed must have a stable visual identity, a native text-bearing region, and a clear default semantic or functional interpretation. These properties enable us to distinguish subject-identity failures from ordinary generation variations and to determine whether non-textual changes express the target-text semantics beyond what is naturally expected from the default subject context.

To reduce ambiguity, we exclude abstract scene categories and place-level descriptions that do not specify a concrete visual subject or a clearly identifiable text-bearing region. Instead, each seed defines a concrete, visually recognizable subject and a native text anchor, such as a storefront facade with a mounted signboard, a service vehicle with a body-side marking region, a product container with a front label, or a warning sign with a predefined text panel. The anchor specifies the intended location of the target string; rendering the correct string elsewhere therefore does not count as successful anchor-text rendering. The subject definition establishes the reference identity and default context, whereas the text anchor defines the local boundary for evaluating text rendering and semantic leakage. The 50 seeds cover six descriptive subject categories. Representative seeds are shown in Fig. 2(a), while the complete seed inventory is provided in Table S1 of the supplementary material.

## C. Controlled Factors

T2LSC-BENCH varies four controlled factors to test complementary hypotheses about target-text-associated semantic leakage. Semantic relation tests whether localized semantic control remains reliable when the target text is semantically incongruent with the predefined subject context. Prompt mode tests whether explicit semantic-boundary constraints suppress leakage, while scene openness tests whether richer non-textual context provides more potential carriers for target-text semantics. Language tests whether the phenomenon is specific to a particular language or writing system. The subject and text anchor remain fixed across matched cases, supporting controlled comparisons of these factors. Their operational definitions are summarized in Table I.

1) Semantic Relation: Each seed is paired with one aligned target text and two conflicting target texts. The aligned text is compatible with the predefined subject identity and default function, providing a reference condition for ordinary text– subject consistency. By contrast, the conflicting texts are semantically incompatible with the default subject interpretation and short enough to fit within the designated anchor. Each text conveys a clear and specific meaning, reducing ambiguity in the subsequent leakage assessment. They cover functional, category, service-domain, product-domain, and safety-context mismatches. These mismatch categories are used only to diversify the conflict probes and reduce dependence on any single target text.

TABLE I  
OPERATIONAL SPECIFICATION OF THE CONTROLLED FACTORS IN T2LSC-BENCH.
<table><tr><td>Factor</td><td>Condition</td><td>Operational specification</td></tr><tr><td rowspan="2">Semantic relation</td><td>aligned</td><td>The target text is semantically compatible with the predefined subject identity and default function.</td></tr><tr><td>conflicting</td><td>The target text conflicts with the predefined subject interpretation. Each seed includes two distinct conflict probes.</td></tr><tr><td rowspan="2">Prompt mode</td><td>natural</td><td>A controlled base prompt specifying the subject, scene openness, text anchor, target text, readability, and native placement.</td></tr><tr><td>anti_leakage</td><td>The same base prompt with additional constraints against semantic propagation, subject or scene reinterpretation, supporting visual content, and scene reconstruction.</td></tr><tr><td rowspan="2">Scene openness</td><td>closed</td><td>A subject-centered composition with 1-3 surrounding objects, low background density, and a mid-shot view.</td></tr><tr><td>open</td><td>A context-rich composition with 4–6 surrounding objects, medium background density, and a mid-to-far-shot view, while keeping the subject identifiable.</td></tr><tr><td rowspan="2">Language</td><td>English</td><td>The subject description, text anchor, target text, and prompt template are instantiated in English.</td></tr><tr><td>Chinese</td><td>The corresponding components are instantiated in Chinese while preserving the same subject-relation design.</td></tr></table>

2) Prompt Mode: The natural condition uses a controlled base prompt that specifies the subject, scene openness, text anchor, target text, readability, and native-placement requirements. The anti leakage condition retains the same base prompt while adding constraints against semantic propagation from the target text, subject or scene reinterpretation, supporting non-textual content, and scene reconstruction. All remaining prompt content is unchanged, allowing the effect of semantic-boundary constraints to be isolated.

3) Scene Openness: Scene openness controls the amount of non-textual context available around the subject.The closed condition specifies a subject-centered mid-shot composition with 1–3 surrounding objects and low background density, whereas the open condition uses a mid-to-far-shot composition with 4–6 surrounding objects and medium background density while keeping the main subject identifiable. These settings provide a reproducible contrast in contextual richness while avoiding both isolated object views and excessively complex scenes. Comparing the two conditions tests whether richer surroundings provide more opportunities for target-textassociated objects or contextual cues to appear outside the designated text anchor.

4) Language: Each case is instantiated in English and Chinese using language-specific subject descriptions, text anchors, target texts, and prompt templates. The subject identity, semantic relation, scene openness, and prompt mode are preserved across the paired conditions, enabling a controlled comparison between the two distinct language conditions.

Together, these factors define the controlled condition space used for factorized case instantiation and subsequent analysis.

## D. Case Instantiation and Quality Control

T2LSC-BENCH constructs benchmark cases through a full factorial expansion of the controlled factors. Each seed specifies a subject description, a text-anchor specification, one aligned target text, and two conflicting target texts. For each of the 50 seeds, the three target-text instances are combined with two scene-openness settings, two prompt modes, and two languages. This process yields 1,200 prompt instances, all of which are applied to each evaluated model. The overall caseinstantiation process is illustrated in Fig. 4.

We conduct consistency checks at the subject, anchor, target-text, and cross-language levels. Subject descriptions are reviewed to ensure that they refer to concrete physical entities with recognizable identities. Text anchors are checked for naturalness, visual specificity, and compatibility with the corresponding subjects. Aligned target texts are verified to be consistent with the default subject interpretation, whereas conflicting target texts are checked to ensure that they are semantically incompatible with that interpretation while remaining concrete and suitable for the designated anchor. The English and Chinese versions are reviewed to preserve the same underlying subject, anchor, and semantic-relation design.

## IV. EVALUATION

## A. Experimental Setup

We evaluate six text-to-image models: Doubao Seedream 5.0, Gemini 3.1 Flash Image Preview, Gemini 2.5 Flash Image, GPT-image-1.5, Wan2.6 Image, and Qwen-Image-2.0. For brevity, we refer to them as Seedream 5.0, Gemini 3.1, GPT-image-1.5, Wan2.6, Qwen-2.0, and Gemini 2.5, respectively. Each model is assigned the same 1,200 benchmark cases, yielding 7,200 planned generations. The final evaluation includes 7,160 available images: 1,176 from GPT-image-1.5, 1,184 from Qwen-Image-2.0, and 1,200 from each of the other four models. The unavailable outputs all involve the Disney storefront seed and were blocked by provider-side contentsafety filters.

All images are evaluated using a shared protocol and judge configuration. The protocol combines OCR evidence, VLM-based anchor-text verification, and VLM-based semantic assessment. Only OCR–VLM disagreements that cannot be resolved by the predefined fusion rules are referred for manual review. Results are analyzed across the four controlled factors defined in the preceding section.

Separately, we validate the automatic protocol on a stratified subset of 420 images, with 70 images sampled from each model. Within each model, the subset contains 35 automati cally predicted SLR-positive and 35 SLR-negative cases while maintaining coverage of semantic relation, scene openness, prompt mode, and language. Two authors with experience in text-to-image evaluation independently annotate TAA, SSP, and SLR without access to the automatic predictions. Disagreements are adjudicated by a third author with experience in multimodal evaluation, and the resulting labels serve as references for human validation.

## B. Evaluation Protocol

As illustrated in Fig. 5, the protocol comprises two evaluation branches that are applied independently to each image. The text-rendering branch combines character-level OCR evidence with anchor-aware VLM verification to determine whether the target text is correctly rendered at the designated anchor. Independently of the text-rendering outcome, the semantic branch evaluates every image for subject identity preservation and target-text-associated semantic leakage. An uncertain outcome is assigned when the visible evidence is insufficient to support a reliable judgment. The conditional leakage measure is subsequently computed over samples with exact anchor-text rendering and a valid semantic-leakage judgment.

We use gemini-3.1-pro-preview with the temperature set to 0 for both text-rendering evaluation and semantic evaluation. The two tasks use separate fixed prompts and JSON output schemas. The VLM version, prompts, confidence thresholds, OCR–VLM fusion rules, and output-parsing procedures remain fixed across all images and evaluated generators.

1) Text-Rendering Evaluation: The text-rendering branch determines whether the complete target string is correctly rendered within the designated native text-bearing region. We first apply PaddleOCR using its Chinese recognition configuration, which supports both Chinese and Latin characters, with the angle classifier enabled. OCR detections with confidence below 0.5 are discarded to suppress spurious responses to background patterns, decorative elements, and unreadable text-like shapes. The retained detections are compared with the target string using exact match, normalized match, edit distance, and character error rate. Exact and normalized matches are used in the fusion decision, whereas edit distance and character error rate serve as diagnostic evidence. OCR provides character-level evidence but cannot reliably determine whether a detected string appears at the designated anchor. It may also fragment multi-line text or incorrectly recognize malformed glyphs as valid characters.

To verify both text content and placement, an anchoraware VLM examines the designated text region and reports the detected string, visibility, readability, placement validity, character-level deviations, ambiguity, and confidence $c _ { i } \in$ [0, 1]. Let $O _ { i } = 1$ denote an exact or normalized OCR match and $V _ { i } ~ = ~ 1$ denote the corresponding VLM match, with a confidence threshold of $\tau = 0 . 7 .$ A sample is provisionally labeled as exact if either $O _ { i } = V _ { i } = 1 ~ \mathrm { o r } ~ O _ { i } = 0 , V _ { i } = 1$ , and $c _ { i } \geq \tau$ . The second rule recovers likely OCR false negatives caused by missed or fragmented detections when the target text is nevertheless visibly correct at the designated anchor.

![](images/e942a15ebbda5a4b410273be56e80b384212673b5f9f76dcfca459b599c5fbea.jpg)  
Fig. 5. Overview of the automatic evaluation protocol.

All OCR–VLM disagreements that remain unresolved by the predefined fusion rules, including OCR matches rejected by the VLM and VLM matches below the confidence threshold, are referred for manual review. Following fusion and review, a non-exact sample with a small but visible characterlevel deviation is labeled as minor error, whereas other visibly incorrect renderings are labeled as failure. The label ambiguous is reserved for cases in which the available visual evidence remains insufficient after review. Only samples assigned the final label exact are counted as successful anchor-text renderings.

2) Semantic Evaluation: The semantic-evaluation branch evaluates whether the image contains target-text-associated non-textual visual evidence outside the designated anchor. The semantic judge receives only the generated image, subject description, text anchor, and target text. The semantic-relation label, prompt mode, complete generation prompt, and generator identity are withheld to prevent the controlled condition or model identity from influencing the judgment.

Rather than requesting a direct leakage decision, the prompt decomposes semantic evaluation into three evidence-grounded steps. The judge first locates the visual entity corresponding to the predefined subject and assesses whether its identity is preserved. It then describes candidate non-textual cues outside the designated anchor in neutral terms. For each cue, the judge determines whether it supports the target-text semantics and whether it would be naturally expected from the default subject context in the absence of the target text. A cue is treated as leakage evidence only when it supports the targettext semantics but cannot be naturally explained by the default subject context. This procedure produces separate SSP and SLR judgments together with concise evidence grounded in visible image content.

Text-only artifacts are treated separately from semantic leakage. Repeated target text and other readable text outside the designated anchor, including text on external signs, labels, and captions, do not constitute SLR by themselves. Likewise, generic blur, structural deformation, or loss of subject identity does not by itself indicate semantic leakage. Such a change contributes to SLR only when it provides visible non-textual evidence that supports the target-text semantics beyond the default subject context.

3) Evaluation Metrics: We report four complementary metrics.

• Text-at-Anchor Accuracy (TAA) measures the proportion of available images in which the complete target string is correctly rendered at the designated anchor.

• Semantic Subject Preservation (SSP) measures the proportion of valid subject-preservation judgments in which the predefined subject identity remains recognizable.

• Semantic Leakage Rate (SLR) measures the proportion of valid leakage judgments in which non-textual visual content outside the designated anchor expresses target-text semantics beyond what would naturally be expected from the default subject context.

• Conditional Semantic Leakage Rate (cSLR) measures the same leakage outcome specifically among samples with exact anchor-text rendering.

Let D denote the set of all available images, and let V<sub>S</sub> and $\gamma _ { L }$ denote the subsets for which valid binary SSP and SLR judgments are available, respectively. For each image $i \in$ $\mathcal { D } ,$ let $q _ { i }$ denote the final text-rendering label. For each $i \in$ $\nu _ { S }$ , let $s _ { i } \in \{ 0 , 1 \}$ denote the subject-preservation judgment, where $s _ { i } = 1$ indicates that the predefined subject identity is preserved. For each $i \in \mathcal { V } _ { L }$ , let $l _ { i } \in \{ 0 , 1 \}$ denote the leakage judgment, where $l _ { i } = 1$ indicates the presence of semantic leakage. Let I[·] denote the indicator function. We define

$$
\mathrm { T A A } = \frac { 1 0 0 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } \mathbb { I } [ q _ { i } = e x a c t ] ,\tag{1}
$$

$$
\mathrm { S S P } = \frac { 1 0 0 } { \left. \mathcal { V } _ { S } \right. } \sum _ { i \in \mathcal { V } _ { S } } \mathbb { I } [ s _ { i } = 1 ] ,\tag{2}
$$

$$
\mathrm { S L R } = \frac { 1 0 0 } { \vert \mathcal { V } _ { L } \vert } \sum _ { i \in \mathcal { V } _ { L } } \mathbb { I } [ l _ { i } = 1 ] .\tag{3}
$$

Because SLR is computed over both exact and non-exact text renderings, it does not isolate whether leakage persists after successful local text rendering. We therefore define

$$
\mathcal { C } _ { L } = \left\{ i \in \mathcal { V } _ { L } \ : | \ : q _ { i } = e x a c t \right\} ,\tag{4}
$$

where $\mathcal { C } _ { L }$ contains samples with exact anchor-text rendering and a valid leakage judgment. Conditional Semantic Leakage Rate is defined as

$$
\mathrm { c S L R } = \frac { 1 0 0 } { \left| \mathcal { C } _ { L } \right| } \sum _ { i \in \mathcal { C } _ { L } } \mathbb { I } [ l _ { i } = 1 ] .\tag{5}
$$

TABLE II  
OVERALL AND RELATION-SPECIFIC PERFORMANCE ON T2LSC-BENCH(%).
<table><tr><td></td><td></td><td colspan="2">Overall</td><td colspan="2">SLR↓</td><td colspan="2">cSLR↓</td><td>∆SLR↓</td></tr><tr><td>Model</td><td>N</td><td>TAA</td><td>SSP</td><td>Al.</td><td>Cf.</td><td>Al.</td><td>Cf.</td><td>pp</td></tr><tr><td>Seedream 5.0</td><td>1200</td><td>97.1</td><td>96.4</td><td>1.8</td><td>21.5</td><td>1.8</td><td>22.0</td><td>19.7</td></tr><tr><td>Gemini 3.1</td><td>1200</td><td>99.0</td><td>98.8</td><td>0.5</td><td>13.0</td><td>0.5</td><td>13.1</td><td>12.5</td></tr><tr><td>GPT-1.5</td><td>1176</td><td>98.3</td><td>96.9</td><td>0.5</td><td>14.0</td><td>0.5</td><td>14.2</td><td>13.5</td></tr><tr><td>Wan2.6</td><td>1200</td><td>94.2</td><td>91.7</td><td>2.0</td><td>22.4</td><td>2.1</td><td>23.4</td><td>20.4</td></tr><tr><td>Qwen-2.0</td><td>1184</td><td>98.8</td><td>96.5</td><td>1.5</td><td>15.8</td><td>1.6</td><td>15.9</td><td>14.3</td></tr><tr><td>Gemini 2.5</td><td>1200</td><td>59.2</td><td>96.7</td><td>1.0</td><td>21.9</td><td>1.3</td><td>22.7</td><td>20.9</td></tr><tr><td>All</td><td>7160</td><td>91.0</td><td>96.2</td><td>1.2</td><td>18.1</td><td>1.3</td><td>18.2</td><td>16.9</td></tr></table>

Note. N denotes the number of available images. Metric-specific valid denominators are reported in the supplementary material. The final row reports image-level micro-averages.

cSLR quantifies semantic leakage conditional on successful anchor-text rendering and therefore directly assesses whether target-text semantics remain locally contained after the target string has been rendered exactly.

All metrics are reported as percentages. Higher TAA and SSP indicate better performance, whereas lower SLR and cSLR indicate stronger semantic containment. An uncertain SSP judgment is excluded only from the denominator of SSP, whereas an uncertain SLR judgment is excluded from the denominators of both SLR and cSLR. cSLR further requires an exact text-rendering label. To quantify the semantic-relation contrast, we additionally report

$$
\Delta \mathrm { S L R } = \mathrm { S L R } _ { \mathrm { c f } } - \mathrm { S L R } _ { \mathrm { a l } } ,\tag{6}
$$

where al denotes the aligned condition and cf pools the two conflict probes defined for each seed.

## V. EXPERIMENTAL RESULTS

## A. Overall Results

Table II shows that text-rendering accuracy and semantic control do not necessarily vary together. TAA ranges from 59.2% to 99.0%, whereas SSP ranges from 91.7% to 98.8%. Under conflicting conditions, aggregate SLR reaches 18.1%, and restricting the analysis to exact text renderings yields a nearly identical cSLR of 18.2%. Thus, correctly rendering the requested text does not necessarily indicate effective control over its semantic influence.

The model-level results reveal distinct capability profiles. Gemini 3.1 achieves the strongest overall performance, although its conflicting cSLR remains 13.1%. Seedream 5.0, by contrast, combines a high TAA of 97.1% with a conflicting cSLR of 22.0%, indicating strong glyph rendering but limited containment of text semantics. Wan2.6 records both the lowest SSP and the highest conflicting cSLR, suggesting that semantic conflict affects both subject identity and surrounding visual content. Gemini 2.5 exhibits limitations in both text rendering and semantic containment, although its cSLR is computed from a smaller exact-rendering subset. These differences are consistent with varying degrees of coupling between local text rendering and global scene synthesis.

TABLE III  
PERFORMANCE UNDER THE CONTROLLED FACTORS OF T2LSC-BENCH (%).
<table><tr><td>Factor</td><td>Condition</td><td>N</td><td>TAA↑</td><td>SSP↑</td><td>SLR↓</td></tr><tr><td rowspan="2">Relation</td><td> $R _ { \mathrm { a l } }$ </td><td>2384</td><td>91.4</td><td>98.7(2381)</td><td>1.2(2381)</td></tr><tr><td> $R _ { \mathrm { c f } }$ </td><td>4776</td><td>90.9</td><td>94.9(4773)</td><td>18.1(4773)</td></tr><tr><td rowspan="2">Prompt</td><td> $P _ { \mathrm { n a t } }$ </td><td>3582</td><td>91.0</td><td>95.8(3581)</td><td>16.6(3581)</td></tr><tr><td> $P _ { \mathrm { a n t i } }$ </td><td>3578</td><td>91.1</td><td>96.5(3573)</td><td>8.4(3573)</td></tr><tr><td rowspan="2">Scene</td><td> $S _ { \mathrm { c l } }$ </td><td>3580</td><td>91.1</td><td>96.2(3577)</td><td>11.4(3577)</td></tr><tr><td> $S _ { \mathrm { o p } }$ </td><td>3580</td><td>91.0</td><td>96.1(3577)</td><td>13.6(3577)</td></tr><tr><td rowspan="2">Language</td><td> $L _ { \mathrm { e n } }$ </td><td>3582</td><td>97.9</td><td>95.6(3579)</td><td>14.1(3579)</td></tr><tr><td> $L _ { \mathrm { z h } }$ </td><td>3578</td><td>84.2</td><td>96.8(3575)</td><td>10.9(3575)</td></tr></table>

Note. N denotes the number of available images. Values in parentheses are valid denominators for SSP and SLR. The results are image-level micro-averages. Relation, prompt, and scene denote semantic relation, prompt mode, and scene openness, respectively.

Fig. 6 illustrates these differences under matched conditions. In the first-aid-box case, several models introduce food-related objects associated with “SNACK BOX,” while others preserve the default context. In the sports-bottle case, leakage appears as reinterpretation of the subject as an automotive-fluid container. The warning-sign and storefront cases further show target-text-associated modification of a non-textual pictogram and broader reconstruction of the storefront as a laundry, respectively. These examples demonstrate that the same semantic conflict can produce object insertion, subject reinterpretation, or scene reconstruction depending on the model, even when anchor-text accuracy is held constant.

Of the 7,200 planned generations, 7,160 are available for evaluation. The 40 unavailable outputs are associated with the Disney storefront seed and provider-side content-safety filtering. Excluding this seed from all models changes every reported metric by less than 0.5 percentage points, indicating that the main findings are not driven by the missing outputs.

## B. Effects of Controlled Factors

Table III reports image-level micro-averages over the six models. For each controlled factor, images from the remaining conditions are pooled, and each image receives equal weight. Model-level macro-averaging yields nearly identical SLR values and the same overall trends. To account for correlations among cases derived from the same subject, we compute paired confidence intervals using 10,000 bootstrap samples over the 50 subject seeds.

As shown in Figure 7, all confidence intervals exclude zero. Semantic relation yields the largest SLR difference, followed by prompt mode, whereas scene openness and language exhibit smaller differences. Overall, leakage varies most strongly with semantic relation, while explicit anti-leakage prompting substantially, but not completely, reduces it.

a) Semantic relation: The semantic-relation comparison provides the primary stress test of localized semantic control. Compared with the semantically matched condition $R _ { \mathrm { a l } }$ , the stress-test condition $R _ { \mathrm { c f } }$ increases SLR from 1.2% to 18.1%.

![](images/d7e46d2e9831711cbf9fbfc4cecda96ad442ef147787cce1dda2d568ccbf01be.jpg)  
Fig. 6. Cross-model comparison under matched benchmark conditions. All outputs achieve exact anchor-text rendering but differ in subject preservation and semantic leakage.

![](images/0f696551d0a68f160d49ca23f710007a60c8a97a4fe72e5a0cf80b3bd7a24894.jpg)  
Fig. 7. Paired seed-clustered bootstrap estimates of SLR differences between controlled conditions. Points indicate estimated differences, horizontal bars indicate 95% confidence intervals, and the dashed line denotes no difference.

The seed-clustered difference is 16.9 percentage points, with a 95% confidence interval of [13.4, 20.7]. By contrast, TAA decreases by only 0.5 percentage points. Moreover, cSLR increases from 1.3% to 18.2%, showing that leakage remains prevalent even when the target text is rendered exactly. Together, these results show that models may successfully follow the local text-rendering instruction while failing to confine the semantic influence of the target text to the designated anchor.

b) Prompt mode: Explicit anti-leakage instructions provide substantial but incomplete mitigation. SLR decreases from 16.6% under $P _ { \mathrm { n { a t } } }$ to 8.4% under $P _ { \mathrm { a n t i } }$ , while TAA remains nearly unchanged. The seed-clustered difference is −8.1 percentage points, with a 95% confidence interval of [−10.2, −6.2]. The magnitude of this reduction varies substantially across models: GPT-image-1.5 shows a large decrease, whereas Seedream 5.0 changes only marginally. Explicit instructions can therefore reduce semantic leakage, but they do not provide consistent semantic containment across models.

c) Scene openness: Open scenes exhibit a higher SLR than closed scenes (13.6% versus 11.4%). The seed-clustered difference is 2.1 percentage points, with a 95% confidence interval of [0.8, 3.5]. Richer visual contexts may provide more opportunities for target-text-associated non-textual cues to appear outside the designated anchor, although this difference is smaller than those associated with semantic relation and prompt mode.

d) Language: The language comparison reveals distinct patterns for text rendering and semantic leakage. Across all six models, English cases exhibit higher SLR than Chinese cases, with aggregate SLR values of 14.1% and 10.9%, respectively. The difference ranges from 0.7 percentage points for Gemini 3.1 to 5.2 percentage points for Wan2.6, indicating a consistent association whose magnitude varies across models. In contrast, the large aggregate TAA difference (97.9% versus 84.2%) is driven primarily by the lower Chinese text-rendering accuracy of Gemini 2.5 and therefore does not reflect a consistent cross-model language pattern. Overall, the language condition is consistently associated with semantic leakage across the evaluated models, whereas the aggregate difference in textrendering accuracy is dominated by model-specific capability.

## C. Human Validation Results

As shown in Table IV, the automatic protocol exhibits high agreement with the adjudicated human annotations across all three dimensions. TAA and SSP achieve agreement rates of 95.5% and 95.0%, with Cohen’s κ values of 0.842 and 0.823, respectively. Their precision, recall, and F1 scores all exceed 96%, indicating reliable assessment of anchor-text correctness and subject preservation.

SLR remains more challenging because its assessment requires distinguishing target-text-associated non-textual cues from the surrounding context that is naturally compatible with the predefined subject. Nevertheless, the automatic SLR judge achieves 90.0% agreement, a Cohen’s κ of 0.800, and an F1 score of 89.3%. Its recall of 96.2% shows that most human-confirmed leakage cases are detected, whereas its lower precision of 83.3% reflects a greater tendency toward falsepositive leakage judgments. Inspection of these false positives suggests that the main remaining difficulty is the overattribution of ambiguous or otherwise natural contextual cues to the target-text semantics. Overall, the automatic evaluation shows high agreement with human annotations, while further improvement is needed to distinguish target-text-associated semantic evidence from visual cues that are natural to the predefined subject context.

![](images/5eb54e3cad8e182661e568c2694237489ea44366b17ed6b2b605bda2945acf22.jpg)  
Fig. 8. Matched comparisons of semantic containment and leakage. Each pair shares the same subject, target text, text anchor, and controlled conditions, although the outputs may come from different models. All outputs satisfy $\mathrm { \bar { T A A } = S S P = 1 ; }$ only leakage cases satisfy SLR = 1.

TABLE IV  
HUMAN VALIDATION ON THE STRATIFIED 420-IMAGE SUBSET.
<table><tr><td colspan="4">(a) Confusion counts</td></tr><tr><td>Metric</td><td>TP</td><td>FP TN</td><td>FN</td></tr><tr><td>TAA</td><td>338</td><td>12</td><td>63</td></tr><tr><td>SSP</td><td>338</td><td>7 61</td><td>7 14</td></tr><tr><td>SLR</td><td>175 35</td><td>203</td><td>7</td></tr></table>

(b) Agreement and classification performance
<table><tr><td>Metric</td><td>Agr. (%)</td><td>κ</td><td>Prec. (%)</td><td>Rec. (%)</td><td>F1 (%)</td></tr><tr><td>TAA</td><td>95.5</td><td>0.842</td><td>96.6</td><td>98.0</td><td>97.3</td></tr><tr><td>SSP</td><td>95.0</td><td>0.823</td><td>98.0</td><td>96.0</td><td>97.0</td></tr><tr><td>SLR</td><td>90.0</td><td>0.800</td><td>83.3</td><td>96.2</td><td>89.3</td></tr></table>

Note. Human annotations are treated as reference labels and automatic judgments as predictions. Agr., Prec., and Rec. denote agreement, precision, and recall.

## D. Qualitative Observations

Fig. 8 compares semantic containment and leakage under matched benchmark conditions. Leakage is indicated by targettext-associated non-textual cues outside the designated anchor, including a water tap, pharmacy shelving, an ATM and coins, an excavator, gasoline pumps, and an ice-cream cone. By contrast, the corresponding contained outputs render the same target text without introducing such cues. These comparisons clarify the SLR decision boundary: Leakage is identified only when the semantics of the target text are reflected in visible non-textual content outside the designated anchor and cannot be naturally explained by the default subject context.

## VI. DISCUSSION

T2LSC-BENCH provides a controlled testbed for comparing how effectively text-to-image models confine targettext semantics to a designated region while preserving the subject and scene. It can support cross-model comparison, regression testing across model updates, and the evaluation of prompt or training-based strategies for reducing leakage, with applications to advertising and design variants, controlled synthetic data, and personalized visual content. However, the current benchmark focuses on subjects with predefined text anchors and stable visual priors, relatively short English and Chinese target strings. Future work could extend the evaluation to longer target strings, additional languages, less structured subjects, and image and video editing settings.

## VII. CONCLUSION

We introduced T2LSC-BENCH, a controlled diagnostic benchmark for evaluating localized-text semantic containment in T2I generation. By separately measuring anchor-text accuracy, subject preservation, and semantic leakage under an explicit localized-text contract, T2LSC-BENCH reveals that correct text rendering, by itself, does not guarantee local semantic containment. Across six models, semantically matched cases generally preserve localized control, whereas the stress-test cases expose substantial semantic leakage despite only minor changes in text-rendering accuracy. Explicit anti-leakage prompting offers partial, model-dependent mitigation. These findings establish semantic containment as an important and complementary dimension of visual-text generation and evaluation, and provide a reproducible protocol for measuring it in current and future multimedia content-generation systems.

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10 684–10 695.

[2] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, W. Manassra, P. Dhariwal, C. Chu, Y. Jiao, and A. Ramesh, “Improving image generation with better captions,” OpenAI, Tech. Rep., 2023. [Online]. Available: https://cdn.openai.com/papers/dall-e-3.pdf

[3] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser: Diffusion models as text painters,” Advances in Neural Information Processing Systems, vol. 36, pp. 9353–9387, 2023.

[4] Y. Tuo, W. Xiang, J. He, Y. Geng, and X. Xie, “Anytext: Multilingual visual text generation and editing,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. [Online]. Available: https://openreview.net/forum?id=ezBH9WE9s2

[5] Y. Zhu, Z. Li, T. Wang, M. He, and C. Yao, “Conditional text image generation with diffusion models,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 14 235–14 244.

[6] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “DreamBooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 500–22 510.

[7] R. Zhang, H. Wang, C. Liu, G. Wang, Z. Ma, and W. Zhang, “Freetext: Training-free text rendering via attention localization and spectral glyph injection,” in ICML, 2026.

[8] M. Ventura, M. Toker, O. Patashnik, Y. Belinkov, and R. Reichart, “Deleaker: Dynamic inference-time reweighting for semantic leakage mitigation in text-to-image models,” in ICLR, 2026.

[9] T. Zhang, X. Wang, L. Li, Z. Tai, J. Chi, J. Tian, H. He, and S. Wang, “STRICT: stress-test of rendering image containing text,” in EMNLP. Association for Computational Linguistics, 2025, pp. 21 137–21 150.

[10] D. Li, J. Li, and S. Hoi, “Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing,” Advances in Neural Information Processing Systems, vol. 36, pp. 30 146–30 166, 2023.

[11] R. Liu, D. Garrette, C. Saharia, W. Chan, A. Roberts, S. Narang, I. Blok, R. Mical, M. Norouzi, and N. Constant, “Character-aware models improve visual text rendering,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 16 270–16 297.

[12] J. Ma, M. Zhao, C. Chen, R. Wang, D. Niu, H. Lu, and X. Lin, “Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation,” arXiv preprint arXiv:2303.17870, 2023.

[13] Y. Yang, D. Gui, Y. Yuan, W. Liang, H. Ding, H. Hu, and K. Chen, “Glyphcontrol: Glyph conditional control for visual text generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 44 050–44 066, 2023.

[14] J. Zhang, Y. Zhou, J. Gu, C. Wigington, T. Yu, Y. Chen, T. Sun, and R. Zhang, “Artist: Improving the generation of text-rich images with disentangled diffusion models and large language models,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 1268–1278.

[15] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser-2: Unleashing the power of language models for text rendering,” in European Conference on Computer Vision. Springer, 2024, pp. 386– 402.

[16] Z. Liu, W. Liang, Z. Liang, C. Luo, J. Li, G. Huang, and Y. Yuan, “Glyph-byt5: A customized text encoder for accurate visual text rendering,” in European Conference on Computer Vision. Springer, 2024, pp. 361–377.

[17] Y. Zhao and Z. Lian, “Udifftext: A unified framework for high-quality text synthesis in arbitrary images via character-aware diffusion models,” in European conference on computer vision. Springer, 2024, pp. 217– 233.

[18] Y. Zhu, J. Liu, F. Gao, W. Liu, X. Wang, P. Wang, F. Huang, C. Yao, and Z. Yang, “Visual text generation in the wild,” in European Conference on Computer Vision. Springer, 2024, pp. 89–106.

[19] N. Du, Z. Chen, Z. Chen, S. Gao, X. Chen, Z. Jiang, J. Yang, and Y. Tai, “Textcrafter: Accurately rendering multiple texts in complex visual scenes,” CoRR, vol. abs/2503.23461, 2025.

[20] F. Fallah, M. Patel, A. Chatterjee, V. Morariu, C. Baral, and Y. Yang, “Textinvision: Text and prompt complexity driven visual text generation

benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 525–534.

[21] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, “Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv preprint arXiv:2308.06721, 2023.

[22] A. Slobodkin, H. Taitelbaum, Y. Bitton, B. Gordon, M. Sokolik, N. B. Guetta, A. Gueta, R. Rassin, D. Lischinski, and I. Szpektor, “Refvnli: Towards scalable evaluation of subject-driven text-to-image generation.” in EMNLP (Findings), 2025, pp. 8420–8438.

[23] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in ICCV, 2023, pp. 3836–3847.

[24] Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee, “Gligen: Open-set grounded text-to-image generation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 22 511–22 521.

[25] Z. Yang, J. Wang, Z. Gan, L. Li, K. Lin, C. Wu, N. Duan, Z. Liu, C. Liu, M. Zeng et al., “Reco: Region-controlled text-to-image generation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 14 246–14 255.

[26] S. Tang, P. Gong, K. Li, K. Guo, B. Wang, M. Ye, J. Zhang, and X. Zhu, “Consistent text-to-image generation via scene de-contextualization,” CoRR, vol. abs/2510.14553, 2025.

[27] H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, and D. Cohen-Or, “Attend-andexcite: Attention-based semantic guidance for text-to-image diffusion models,” ACM transactions on Graphics (TOG), vol. 42, no. 4, pp. 1– 10, 2023.

[28] W. Feng, X. He, T.-J. Fu, V. Jampani, A. Akula, P. Narayana, S. Basu, X. E. Wang, and W. Y. Wang, “Training-free structured diffusion guidance for compositional text-to-image synthesis,” arXiv preprint arXiv:2212.05032, 2022.

[29] R. Rassin, E. Hirsch, D. Glickman, S. Ravfogel, Y. Goldberg, and G. Chechik, “Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment,” Advances in Neural Information Processing Systems, vol. 36, pp. 3536–3559, 2023.

[30] Y. Li, M. Keuper, D. Zhang, and A. Khoreva, “Divide and bind your attention for improved generative semantic nursing,” in BMVC, 2023.

[31] T. H. S. Meral, E. Simsar, F. Tombari, and P. Yanardag, “CONFORM: Contrast is all you need for high-fidelity text-to-image diffusion models,” in CVPR, 2024, pp. 9005–9014.

[32] J. Kim, E. Esmaeili, and Q. Qiu, “Text embedding is not all you need: Attention control for text-to-image semantic alignment with text selfattention maps,” in CVPR, 2025, pp. 8031–8040.

[33] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation,” in NeurIPS Datasets and Benchmarks, 2023, pp. 78 723– 78 747.

[34] C.-E. Wu, L. Yu, Z. Huang, O. Russakovsky, and S. Arora, “Conceptmix: A compositional image generation benchmark with controllable difficulty,” in NeurIPS Datasets and Benchmarks, 2024.

[35] S. Mun, J. Nam, S. Cho, and J. Ok, “Addressing text embedding leakage in diffusion-based image editing,” in ICCV, 2025, pp. 16 451–16 460.

[36] X. Zhu, P. Sun, Y. Song, Y. Xiao, Z. Li, C. Wang, J. Huang, B. Yang, and X. Xu, “Evaluating semantic variation in text-to-image synthesis: A causal perspective,” in ICLR, 2025.

[37] J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi, “Clipscore: A reference-free evaluation metric for image captioning,” in EMNLP, 2021, pp. 7514–7528.

[38] Y. Hu, B. Liu, J. Kasai, Y. Wang, M. Ostendorf, R. Krishna, and N. A. Smith, “Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering,” in ICCV, 2023, pp. 20 406–20 417.

[39] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An object-focused framework for evaluating text-to-image alignment,” in NeurIPS Datasets and Benchmarks, 2023.

[40] T. Lee, M. Yasunaga, C. Meng, Y. Mai, J. S. Park, A. Gupta, Y. Zhang, D. Narayanan, H. Teufel, M. Bellagente, M. Kang, T. Park, J. Leskovec, J.-Y. Zhu, L. Fei-Fei, S. Ermon, and P. Liang, “Holistic evaluation of text-to-image models,” arXiv preprint arXiv:2311.04287, 2023.

[41] O. Wiles, C. Zhang, I. Albuquerque, I. Kajic, S. Wang, E. Bugliarello,´ Y. Onoe, C. Knutsen, C. Rashtchian, J. Pont-Tuset, and A. Nematzadeh, “Revisiting text-to-image evaluation with gecko: On metrics, prompts, and human ratings,” arXiv preprint arXiv:2404.16820, 2024.

[42] B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, X. Xia, P. Zhang, G. Neubig, and D. Ramanan, “Evaluating and improving compositional text-to-visual generation,” in CVPR Workshops, 2024, pp. 5290–5301.