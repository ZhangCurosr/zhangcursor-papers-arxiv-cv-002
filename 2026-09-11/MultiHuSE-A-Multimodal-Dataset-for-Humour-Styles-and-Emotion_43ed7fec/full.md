# MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions

1<sup>st</sup> Mary Ogbuka Kenneth

2<sup>nd</sup> Foaad Khosmood

Algorithmic Human Development group,

3<sup>rd</sup> Abbas Edalat

Department of Computing

Computer Engineering Department

Imperial College London

London, United Kingdom

m.kenneth22@imperial.ac.uk

California Polytechnic State University

San Luis Obispo, United States

foaad@calpoly.edu

Algorithmic Human Development group,

Department of Computing

Imperial College London

London, United Kingdom

a.edalat@imperial.ac.uk

Abstract—Computational recognition of verbal humour remains a challenging task, requiring an understanding of language, delivery style, emotions, and cultural context. Most existing approaches focus on binary classification and lack datasets that capture psychological dimensions of humour alongside variations in expression. We introduce MultiHuSE, a multimodal dataset comprising 2,407 high-definition videos of 50 demographically diverse actors performing 1,463 text samples across four psychological humour styles (affiliative, aggressive, self-enhancing, and self-deprecating), as well as neutral content. A subset is additionally annotated for underlying emotions. The dataset uniquely captures multiple actor interpretations of the same texts, enabling systematic analysis of expressive diversity. Baseline experiments show that multimodal fusion outperforms unimodal approaches (80.1% vs. 77.4% accuracy) in humour style classification, with particularly strong gains for affiliative humour (66% to 74%). While text provides the strongest individual signal, fusion models deliver meaningful improvements. We hope that MultiHuSE provides empirical support for psychological theories linking humour and emotion, while also opening new avenues for research in human communication, well-being, and AI-driven interaction. The dataset is available for academic use under an End-User Licence Agreement.

Index Terms—Computational humour, Multimodal emotion recognition, Humour style classification, Facial expression analysis, Affective computing

## I. INTRODUCTION

Humour recognition presents significant challenges for artificial intelligence systems, particularly in multimodal contexts that require integration of verbal, visual, and acoustic cues [1]– [3]. While research in multimodal humour detection has advanced [4]–[6], most approaches remain limited to binary classification (humour vs. non-humour) [7], overlooking deeper psychological dimensions of humour identified by Martin et al. [8]: self-enhancing, self-deprecating, affiliative, and aggressive styles. These categories have significant implications for psychological well-being: affiliative and self-enhancing styles generally foster positive outcomes [9]–[12], while aggressive and self-deprecating styles can be detrimental to mental health [13]–[15].

The growing adoption of laughter-based interventions in therapeutic settings for addressing depression and anxiety [16], [17], combined with evidence that specific humour styles differentially impact mental health outcomes [8], [18], [19], underscores the need for computational systems that detect humour styles and their associated emotional expressions. Unlike conventional emotion datasets capturing well-defined emotional states, our approach examines nuanced emotional undertones within humour delivery, emotions that may be subtly expressed or filtered through the comedic performance context [20]. This integration enables computational validation of psychological theories and the development of emotionallyaware systems for mental health applications.

Existing multimodal humour datasets present several limitations for analysing humour styles. UR-FUNNY [21] focuses on binary humour detection using TED talks; MHD [5] uses sitcom recordings with laughter tracks; and Passau-SFCH [22] provides style annotations but is restricted to German football press conferences. Additionally, these datasets typically capture only a single performance per instance, limiting the study of expressive variation.

To address these gaps, we introduce MultiHuSE, a new multimodal dataset of over 2,400 recordings featuring fifty diverse performers executing humour across four psychological categories and neutral content, with annotations of underlying emotion. The dataset uniquely captures multiple interpretations of the same texts, enabling systematic analysis of expression variation and multimodal humour perception. Our contributions include:

1) The first English-language multimodal dataset annotated with psychological humour styles and enriched with emotion labels.

2) Multiple video performances of the same text by different actors, enabling analysis of expressive and subjective variation in humour delivery

3) Benchmark experiments showing that multimodal fusion outperforms unimodal models (80% vs. 77% F1-score)

## II. RELATED WORK

Recent datasets have advanced multimodal humour detection, but with significant limitations (see Table I for comparison). UR-FUNNY [21] contains 16,514 video segments from TED Talks with binary classification, while MHD [5] provides

13,633 TV sitcom clips with laughter-based annotations. Both datasets offer valuable scale and naturalistic settings but focus solely on binary humour detection, overlooking psychologically grounded humour styles that have different implications for well-being [2], [8].

The Passau-SFCH dataset [22] addresses this gap by incorporating four humour styles across 39,682 German football press conference segments, though only 6% contains actual humour content. However, it is limited to male German speakers in a single domain and lacks controlled recording conditions, limiting its generalisability across diverse populations and contexts. Similarly, YouTube stand-up datasets [23] capture authentic performances but focus on laughter detection rather than humour style classification.

Multimodal emotion datasets provide methodological foundations for affective analysis but are limited for humour research. MELD [24] includes 13,708 video segments from Friends with rich emotion annotations, but only 38% contain humorous content and lacks humour-specific style labels. SHEMuD [25] combines binary humour detection with emotion labels across 6,191 TV dialogue segments, but its humour classification remains binary without distinguishing psychologically meaningful humour styles. Both datasets capture only single expressions of content, preventing analysis of how performers might express the same humorous material with varying emotional undertones.

## A. Dataset Gaps and Research Motivation

The analysis of existing datasets reveals three critical gaps: (1) predominant focus on binary classification rather than psychologically meaningful humour styles, (2) single-expression capture that overlooks interpretative diversity, and (3) limited integration with psychological theories linking humour to emotional expression and well-being. We developed Multi-HuSE with the goal of addressing these gaps and supporting research at the intersection of computation and psychological theory.

## III. DATASET CREATION

We present MultiHuSE, an unpublished multimodal dataset for humour style and emotion recognition. The collection comprises 2,407 high-definition recordings (1080p at 25fps) of fifty demographically diverse participants performing nearly 1,500 text instances. The texts were sourced from the humour style corpus of Kenneth et al. [2], which includes psychologically grounded humour style annotations. While the text annotations are from this prior work, all video recordings, actor performances, and emotion annotations are novel contributions of our dataset.

## A. Data Collection

a) Recording Setup: We employed a professional, controlled recording environment in a dedicated room at Imperial College London with a Canon EOS Rebel DSLR camera and Rode VideoMic Pro directional microphone. All recordings were conducted under identical setup conditionswhite backdrop, consistent lighting, and standardised participant positioning (head to knee)to ensure both facial expressions and body gestures were clearly visible while minimising environmental variability that could interfere with model training.

b) Participant Demographics: We recruited 50 actors (54% female, 40% male, 6% other genders; ages 18–69 years; 72% White, 12% Asian/Asian British, 6% Black/African/Caribbean, 10% other backgrounds) through a structured recruitment process.<sup>1</sup> Participants received a 50 compensation for their ∼ 2.5 hour commitment, including preparation and recording.

c) Participant Preparation: Participants received detailed instructions explaining each humour style (full instructions available online<sup>2</sup>), along with the following definitions:

• Affiliative: Humour used to build connections and social bonds, focusing on shared experiences.

• Self-enhancing: Humour involving finding amusement in adversity; a positive, adaptive style.

• Aggressive: Humour used to attack or belittle others through sarcasm or put-downs.

• Self-deprecating: Humour involving making fun of oneself, highlighting personal weaknesses.

• Neutral: Statements that are not jokes or do not fit other humour styles.

Participants were provided with the text samples prior to the recording day to familiarise themselves with the content, practise delivery with different emotions, and opt out of any samples they felt uncomfortable performing. They were instructed to consider six emotional categories (joy, empathy, anger, neutral, embarrassment, and superiority) while preparing their performances.

d) Dataset Composition: Our video collection contains over 2,400 samples distributed across five categories with nearbalanced representation: self-enhancing (500 videos, 20.8%), aggressive (529 videos, 22.0%), neutral (495 videos, 20.6%), self-deprecating (457 videos, 19.0%), and affiliative (426 videos, 17.7%) instances.

The dataset uniquely features 943 re-performed videos (approximately 40% of the dataset), capturing multiple interpretations of identical content for direct comparison of expression variations. This approach enables systematic analysis of the subjective nature of humour delivery, where performers bring unique interpretations to the same textual content. Fig. 1 illustrates this interpretative diversity through an example of the aggressive humour sample “I hear you were born on April 2, one day too late”. The first performer (Fig. 1a) delivers the joke with visible anger, whilst the second performer (Fig. 1b) expresses a sense of superiority—demonstrating how identical text can elicit distinct emotional expressions and performance styles. Such variation is particularly valuable for studying the nuanced relationship between humour styles, emotional expression, and delivery techniques.

TABLE I: Comparison of existing multimodal humour and emotion datasets. The table shows key characteristics, including data source, number of video samples, average video duration in seconds, modalities used, annotation granularity, and whethe datasets capture single or multiple performances of the same content
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Content Source</td><td rowspan=1 colspan=1># Videos</td><td rowspan=1 colspan=1>Avg Dur.[s]</td><td rowspan=1 colspan=1>Modalities</td><td rowspan=1 colspan=1>Annotation Type</td><td rowspan=1 colspan=1>PerformanceType</td><td rowspan=1 colspan=1>Key Limitations/Features</td></tr><tr><td rowspan=1 colspan=1>UR-FUNNY [21]</td><td rowspan=1 colspan=1>TED Talks</td><td rowspan=1 colspan=1>16,514</td><td rowspan=1 colspan=1>4.58</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Binary humour</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>English; No humour style differentiation</td></tr><tr><td rowspan=1 colspan=1>MHD [5]</td><td rowspan=1 colspan=1>TV sitcoms(Big Bang Theory)</td><td rowspan=1 colspan=1>13,633</td><td rowspan=1 colspan=1>3.83</td><td rowspan=1 colspan=1>Video, audio</td><td rowspan=1 colspan=1>Binary humour</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>English; Producer-added laughter; no styles</td></tr><tr><td rowspan=1 colspan=1>Passau-SFCH [22]</td><td rowspan=1 colspan=1>Football press conferences</td><td rowspan=1 colspan=1>39,682</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Four humour styles</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>German; Limited to male speakers in single domain</td></tr><tr><td rowspan=1 colspan=1>MELD [24]</td><td rowspan=1 colspan=1>TV series (Friends)</td><td rowspan=1 colspan=1>13,708</td><td rowspan=1 colspan=1>3.59</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Emotions only</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>English; No humour style annotations</td></tr><tr><td rowspan=1 colspan=1>SHEMuD [26]</td><td rowspan=1 colspan=1>TV dialogues</td><td rowspan=1 colspan=1>6,191</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Binary humour + emotions</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>Hindi/English; Single source; no style annotations</td></tr><tr><td rowspan=1 colspan=1>YouTube Stand-up [23]</td><td rowspan=1 colspan=1>YouTube stand-up comedy</td><td rowspan=1 colspan=1>39,127</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Laughter-based</td><td rowspan=1 colspan=1>Single</td><td rowspan=1 colspan=1>Russian/English; Focus on laughter, not humour styles</td></tr><tr><td rowspan=1 colspan=1>MultiHuSE (Ours)</td><td rowspan=1 colspan=1>Controlled studio</td><td rowspan=1 colspan=1>2,407</td><td rowspan=1 colspan=1>6.66</td><td rowspan=1 colspan=1>Video, audio, text</td><td rowspan=1 colspan=1>Four humour styles+ emotions</td><td rowspan=1 colspan=1>Multiple</td><td rowspan=1 colspan=1>English; Diverse performers; consistent quality;943 alternative performances ofidentical jokes with varied expressions</td></tr></table>

![](images/644111492ae24574b857d633e6e76b33f4d6560fce909f3a264a17c9dcf4e42e.jpg)  
(a) Performer 1 Expressing Anger

![](images/c2a42dfca14a724889cf4d6828e75e7874d023780e6a669f8b4e9653c2125f9d.jpg)  
(b) Performer 2 Expressing Superiority  
Fig. 1: Expression variability for the same aggressive joke: (a) anger versus (b) superiority delivery of joke: “I hear you were born on April 2, one day too late”.

e) Recording Duration: Videos range from 2 to 74.96 seconds (mean = 6.66 s), with 91.3% under 10 seconds, 6.7% (10–20 s), 1.2% (20–30 s), and 0.8% (> 30 s). Brief videos include concise jokes such as “Do your thing and don’t care if they like it” (self-enhancing, 2 seconds), “They say good things take time so that’s why I am always late” (self-enhancing, 3 seconds), and “I feel there is no more trash to burn but myself” (self-deprecating, 2 seconds). These brief clips capture complete but concise humorous expressions, while longer videos contain more elaborate joke narratives. This variation in duration reflects the natural diversity of humour delivery, from quick one-liners to extended anecdotes.

## B. Annotation and Performance Approach

a) Humour Style and Text Categories: We utilised the labels from the existing text dataset [2], which contains 1,463 instances across five categories: 1,129 jokes aligned with four humour styles framework (self-enhancing, self-deprecating, affiliative, and aggressive) [8], and 334 neutral non-jokes.

b) Performance Guidance and Interpretative Freedom: Actors received a preparation document<sup>3</sup> with brief definitions of each humour style, along with the actual jokes and their associated category labels. Crucially, the instructions explicitly stated: “Remember, there are no ‘correct’ interpretations. We’re interested in your natural portrayal ofthese jokes across different emotional spectrums.” Participants were encouraged to practise delivering each joke with different emotions from a provided list (joy, anger, neutral, superiority, embarrassment, and empathy) to explore varied interpretations. This balanced approach–providing category understanding while emphasising interpretative freedom– resulted in substantial performance diversity, even for identical text samples, as evidenced by the varying emotions displayed across performances (see Fig. 2, analysed in detail in Section III-C).

c) Emotion Annotation: Immediately after performing each text sample, performers provided self-reports of their own emotional state during delivery, selecting from six emotion categories (joy, anger, neutral, superiority, embarrassment, and empathy). These self-reported emotions represent the performers’ internal experience while expressing the humour, not anticipated audience reactions or observer perceptions. This approach aligns with psychological research [10], [27]– [29] that examines humour styles in relation to the expresser’s emotional state. These categories enable analysis of humourembedded emotional states distinct from general emotion datasets, supporting psychological theory validation (Section III-C, Figure 2). This performer-centred emotional assessment provides insight into the psychological experience of humour production rather than reception.

d) Emotion Inter-annotator Agreement: We evaluated agreement between emotion annotations from original performances and re-performances using paired data where the same jokes were performed by different actors or the same actor with different interpretations. Our analysis used 687 paired instances representing jokes with both original and reperformance emotion annotations.

Our analysis employed two key metrics:

1) Simple Agreement Rate (0.440): The proportion of instances where both annotators assigned the same emotion, calculated as the number of perfect matches (302) divided by the total paired instances (687).

2) Cohen’s Kappa (0.325): A robust measure accounting for chance agreement, indicating fair agreement above random chance [30], [31].

While all original performances include emotion annotations, only 687 of the 943 re-performances have emotion labels, allowing for paired analysis of these 687 jokes.

![](images/a251e937cc42b378076b303af1ba9d31abf7e6cd48b07020a7a746b897b06e99.jpg)

![](images/7c43d56876bf6e01c454ff650d8fe869f46ff89a0e853725dd03a586bde689d5.jpg)  
Fig. 2: Relationship between humour styles and emotions visualised through heatmaps. Left: original performance annotations; Right: re-performance annotations. Darker colours represent stronger associations between specific emotions and humour styles

## C. Humour Styles-Emotion Relationships

Figure 2 demonstrates consistent emotion-humour associations across original and re-performances using 687 paired instances. The stable patterns (e.g., aggressive humour predominantly paired with anger/superiority, self-deprecating humour paired with embarrassment) validate that these emotionhumour relationships are robust across different performers, supporting psychological theories that link specific humour styles to corresponding emotional states during expression. Key patterns include:

• Neutral humour: Strong association with neutral emotion (Original: 90; Re-performance: 98)

• Aggressive humour: Predominantly paired with Anger (Original: 60; Re-performance: 74) and Superiority (Original: 42; Re-performance: 40), reflecting consistent assertive or mocking interpretations.

• Self-deprecating humour: Strongly aligns with Embarrassment (Original: 70; Re-performance: 55), supporting its association with vulnerability and self-directed awkwardness.

• Self-enhancing humour: Co-occurs with Empathy (Original: 39; Re-performance: 31) and Joy (Original: 36; Reperformance: 45), indicating reflective and positive selfregard.

• Affiliative humour: Primarily linked to Joy (Original: 59; Re-performance: 67), supporting its role in positive social bonding.

The consistent patterns observed between original and reperformance data indicate emotion-humour associations that align with established psychological theories [10], [27]–[29] and offer potential support for computational approaches to recognising emotional aspects of humour styles.

## D. Ethical Considerations and Data Access

The MultiHuSE dataset was created with careful attention to ethical principles. All procedures received approval from the Science, Engineering and Technology Research Ethics Committee at Imperial College London. Participants provided informed consent, were compensated £50 for their time, and had the option to opt out of performing any content they found uncomfortable. While personal identifiers were pseudonymised, facial features remained visible as required for research purposes, with participants explicitly informed of this necessity.

The MultiHuSE dataset is available to researchers via a dedicated section of our project website<sup>4</sup>, under a custom End-User Licence Agreement (EULA) permitting use for academic and research purposes only. The website currently provides information about the project and access procedures.

To mitigate potential risks associated with offensive content in certain aggressive humour samples, the dataset documentation explicitly states that inclusion does not imply endorsement. Use of the dataset is restricted to research purposes aligned with the advancement of mental health understanding and computational humour analysis.

## IV. EXPERIMENTS

To demonstrate the viability and research utility of our multimodal collection, we performed baseline experiments evaluating humour style and emotion recognition across single and combined modalities. Direct comparison with existing datasets is not feasible as MultiHuSE represents the first English-language multimodal dataset with psychological humour style annotations, making our baselines foundational benchmarks for future research.

## A. Feature Extraction

We employed state-of-the-art encoders to extract robust multimodal features:

Text: BERT-base-uncased embeddings (768-D), using the [CLS] token from padded sequences (max 128 tokens) [32]

Audio: Dasheng-0.6B (1280D) [33], a powerful audio encoder pretrained on diverse audio data for capturing nuanced speech and sound features.

Video: MC3-18 (768D) [34], a spatiotemporal convolutional encoder optimised for short video clips and fine-grained motion representation.

## B. Data Splits and Preprocessing

We used an 80:20 train-test split stratified by humour style for humour style classification. Training data underwent 5-fold stratified cross-validation.

## C. Baseline Models

We evaluated XGBoost classifiers and Cross-Modal Attention Transformer:

XGBoost: XGBClassifier(use\_label\_encoder =False, random\_state=42,

eval\_metric=’mlogloss’) [35].

Transformer: Cross-modal attention network with multihead attention (8 heads) projecting text, audio, and visual features to a common 1024D space, trained end-to-end for humour style and emotions classification.

## D. Fusion Strategies

Single Modality: Individual XGBoost models trained for text, audio, and visual encoder features.

Exponential Weighted Fusion: Predictions from separate XGBoost modality models combined using exponential weighting (α = 2) based on individual modality performance, resulting in dynamic weights (humour: text 0.54, audio 0.31, visual 0.15; emotion: text 0.40, audio 0.32, visual 0.27).

Cross-Modal Attention: Uses an attention mechanism to weigh the relevance of features from one modality when processing another. This allows the model to learn complex relationships between modalities directly.

## E. Evaluation

We report accuracy, precision, recall, and F1-score (weighted averages). Accuracy measures overall prediction correctness, while precision reflects the proportion of true positives among predicted positives. Recall assesses the models ability to identify all actual positives, and the F1-score provides the harmonic mean of precision and recall, balancing both metrics.

## V. RESULTS

## A. Modality Effectiveness Analysis

Table II shows baseline results. Text features yielded the strongest performance (77.4%), confirming linguistic content as the primary signal for humour style classification. Audio (58.5%) and visual (40.0%) lagged behind, reflecting the subtle and subjective nature of emotional and humorous expression, which makes these cues harder to model. Still, they add complementary prosodic and expressive information.

Multimodal fusion outperformed unimodal baselines: exponential weighted fusion reached 80.1% accuracy and crossattention 79.7%. Though gains over text-only are modest (+2.7%, +2.3%), they highlight meaningful cross-modal contributions and demonstrate the datasets compatibility with modern architectures.

Table III shows humour-style variation: neutral content was most reliable (85 - 90% F1), while affiliative humour benefited most from multimodality (66% to 74% F1 with exponential fusion). Text remained dominant across all fusions, with audio-visual alone achieving only 58.3%. Thus, humour style recognition is semantically driven but enriched by multimodal expression, underscoring the capacity of the data set to provide a complete understanding of humour.

TABLE II: Comprehensive Multimodal Results on MultiHuSE Dataset
<table><tr><td rowspan="2">Method</td><td colspan="4">Humor Styles (%)</td><td colspan="4">Emotion (%)</td></tr><tr><td>Acc</td><td>Prec</td><td>Recall</td><td>F1</td><td>Acc</td><td>Prec</td><td>Recall</td><td>F1</td></tr><tr><td colspan="9">Single Modality Baselines</td></tr><tr><td>Text Only</td><td>77.4</td><td>78.0</td><td>77.0</td><td>77.0</td><td>36.5</td><td>35.0</td><td>37.0</td><td>35.0</td></tr><tr><td>Audio Only</td><td>58.5</td><td>59.0</td><td>58.0</td><td>58.0</td><td>32.8</td><td>33.0</td><td>33.0</td><td>32.0</td></tr><tr><td>Visual Only</td><td>40.0</td><td>40.0</td><td>40.0</td><td>39.0</td><td>30.1</td><td>30.0</td><td>30.0</td><td>29.0</td></tr><tr><td colspan="9">Multimodal Fusion</td></tr><tr><td>Equal Fusion</td><td>77.8</td><td>78.0</td><td>78.0</td><td>78.0</td><td>38.2</td><td>38.0</td><td>38.0</td><td>37.0</td></tr><tr><td>Exponential Fusion</td><td>80.1</td><td>80.0</td><td>80.0</td><td>80.0</td><td>38.6</td><td>38.0</td><td>39.0</td><td>37.0</td></tr><tr><td>Cross-Attention</td><td>79.7</td><td>80.0</td><td>79.0</td><td>79.0</td><td>32.2</td><td>32.0</td><td>32.0</td><td>32.0</td></tr></table>

TABLE III: F1-Scores for Individual Humour Styles
<table><tr><td>Method</td><td>Self-Enhancing</td><td>Self-Deprecating</td><td>Affiliative</td><td>Aggressive</td><td>Neutral</td></tr><tr><td>Text Only</td><td>77</td><td>80</td><td>66</td><td>77</td><td>85</td></tr><tr><td>Audio Only</td><td>55</td><td>57</td><td>47</td><td>54</td><td>79</td></tr><tr><td>Visual Only</td><td>39</td><td>29</td><td>33</td><td>40</td><td>56</td></tr><tr><td>Equal Fusion</td><td>76</td><td>79</td><td>66</td><td>76</td><td>90</td></tr><tr><td>Exponential Fusion</td><td>79</td><td>81</td><td>74</td><td>78</td><td>89</td></tr><tr><td>Cross-Attention</td><td>83</td><td>78</td><td>68</td><td>81</td><td>85</td></tr></table>

## B. Dataset Quality and Validation

To assess the reliability and research utility of the dataset, we conducted validation across three key dimensions: feature extraction reliability, re-performance variation, and annotation coverage.

1) Feature Extraction Reliability: To validate technical data quality, we assessed computational feature extraction success rates across all video samples using state-of-the-art encoders. Table IV presents the success rates for each modality.

TABLE IV: Feature extraction success rates
<table><tr><td>Modality</td><td>Encoder</td><td>Success Rate</td><td>Successful/Total</td></tr><tr><td>Text</td><td>BERT-base-uncased</td><td>100%</td><td>2,407/2,407</td></tr><tr><td>Audio</td><td>Dasheng-0.6B</td><td>100%</td><td>2,407/2,407</td></tr><tr><td>Visual</td><td>MC3-18</td><td>100%</td><td>2,407/2,407</td></tr></table>

All modalities achieved perfect extraction rates (100%) using modern pre-trained encoders, ensuring comprehensive coverage for multimodal fusion experiments.

2) Re-performance Variation: A central strength of our dataset is its capture of diverse interpretations of identical humorous content. We analysed 1,872 re-performance videos corresponding to 927 unique jokes, using 17 facial features extracted via MediaPipe Face Mesh [36]. For each joke performed by multiple actors, we computed pairwise Euclidean distances between facial feature vectors and compared them to random baseline distances.

Within-joke distances were significantly smaller than random baseline distances, indicating that re-performances of identical content maintain meaningful similarity whilst exhibiting substantial expressive diversity. This statistically significant difference suggests that re-performances capture both content consistency and interpretative variation, supporting MultiHuSE’s ability to study the subjective nature of humour expression.

TABLE V: Re-performance variation analysis
<table><tr><td>Distance Type</td><td>Mean SD</td><td>Interpretation</td></tr><tr><td>Within-joke</td><td>0.2181 0.1112</td><td>Same joke re-performances</td></tr><tr><td>Random baseline</td><td>0.2661 0.1181</td><td>Different jokes</td></tr><tr><td>Mann-Whitney U</td><td> $p < 0 . 0 0 0 0 0 1$ </td><td>Cohen&#x27;s  $\mathbf { d } = \mathbf { 0 . 4 3 }$ </td></tr></table>

3) Annotation Coverage: All 2,407 videos include complete humour style labels inherited from the source text dataset. Self-reported emotion annotations are available for all 1,463 original performances and 687 re-performances (72.7% of reperformed instances), providing 89.3% overall emotion coverage (2,150/2,407 instances). The partial emotion coverage resulted from methodological evolution during data collection, as the emotion annotation protocol for re-performance was introduced partway through the study after recognising its value for understanding expression variation.

## VI. APPLICATIONS OF THE MULTIHUSE DATASET

MultiHuSE enables recognition of psychologically grounded humour styles with established links to mental health outcomes. Meta-analytic evidence shows affiliative and self-enhancing humour protects against depression and anxiety [18], [19], while aggressive and self-deprecating styles increase mental health risks [15], [37]. Recent laughter-based interventions for treating depression [16], [17] highlight the therapeutic potential of humour style differentiation.

## A. Mental Health Applications

Digital Monitoring: Apps that track users’ humour expressions during therapy sessions–detecting predominant selfdeprecating humour (depression risk indicator) versus affiliative humour (protective factor)–enabling personalised intervention recommendations. For example, a system could alert therapists when patients consistently use self-deprecating humour, prompting targeted cognitive behavioural interventions.

Clinical Assessment: Tools supporting mental health professionals in evaluating humour-based coping mechanisms and therapeutic progress through automated analysis of patient expressions.

Content Moderation: Systems detecting harmful humour while preserving adaptive styles, reducing cyberbullying and enhancing digital well-being.

Emotionally Aware AI: Virtual assistants that differentiate supportive from harmful humour, enabling psychologically sensitive responses in therapeutic contexts.

## VII. CONCLUSION

We present MultiHuSE, a multimodal dataset for humour style and emotion recognition comprising over 2,400 highdefinition recordings of fifty diverse performers executing a balanced set of textual content across four psychologicallygrounded categories. The dataset includes 943 re-performed samples, enabling direct comparison of expression variations for identical content.

Our validation shows 100% feature extraction success with state-of-the-art encoders and significant performance variation reflecting interpretative diversity $( p < 0 . 0 0 0 0 0 1 )$ . Multimodal fusion outperformed single modalities, with exponential fusion achieving 80.1% accuracy vs. 77.4% for text-only; crossattention reached 79.7%, confirming compatibility with modern deep learning. Ablation highlights text dominance while showing complementary contributions from audio and visual cues. We have observed relationships between humour styles and specific emotions that appear consistent with psychological theories.

## VIII. LIMITATIONS AND FUTURE DIRECTIONS

We acknowledge several limitations. First, the moderate inter-annotator agreement (Cohen’s $\kappa ~ = ~ 0 . 3 2 5 )$ reflects the inherent subjectivity of emotional experiences in humour contexts. Second, despite demographic diversity within our sample 72% White, 12% Asian/Asian British, 6% Black/African/Caribbean, 10% other backgrounds), recruitment was geographically constrained to London, which may limit the dataset’s representation of global humour expression patterns and cultural interpretations of humour styles. This geographical limitation reinforces Western humour perspectives and may introduce cultural bias in computational models trained on this data.

Third, while our dataset contains 2,407 videos compared to larger collections, our controlled design emphasises quality and balance–79.4% humorous content with even distribution across styles, compared to Passau-SFCH’s 6.02% humorous content from a single domain. Direct comparison with existing datasets is challenging as MultiHuSE represents the first English-language multimodal dataset with psychological humour style annotations.

Future research will address these limitations through multiple recruitment centers across different countries and cultural contexts to capture diverse humour expression patterns and cultural interpretations of humour styles. Additional directions include exploring advanced vision-language models, longitudinal studies of humour style changes for intervention applications, and investigating relationships between automatically detected and human-perceived emotions in humour contexts.

## REFERENCES

[1] C. Strapparava, O. Stock, and R. Mihalcea, “Computational Humour,” in Emotion-Oriented systems, 2011, pp. 609–634.

[2] M. O. Kenneth, F. Khosmood, and A. Edalat, “A Two-Model Approach for Humour Style Recognition,” in Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities. Miami: Association for Computational Linguistics, 11 2024, pp. 259– 274. [Online]. Available: https://aclanthology.org/2024.nlp4dh-1.25/

[3] C. Shani, N. Borenstein, and D. Shahaf, “How Did This Get Funded?! Automatically Identifying Quirky Scientific Achievements,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. Association for Computational Linguistics, 6 2021, pp. 14–28. [Online]. Available: http://arxiv.org/abs/2106.03048

[4] Y. Kayatani, Z. Yang, M. Otani, N. Garcia, C. Chu, Y. Nakashima, and H. Takemura, “The laughing machine: Predicting humor in video,” in Proceedings - 2021 IEEE Winter Conference on Applications of Computer Vision, WACV 2021. Institute of Electrical and Electronics Engineers Inc., 1 2021, pp. 2072–2081.

[5] B. N. Patro, M. Lunayach, D. Srivastava, H. Singh, and V. P. Namboodiri, “Multimodal Humor Dataset: Predicting Laughter tracks for Sitcoms,” Computer Vision fundation, pp. 576–585, 2021. [Online]. Available: https://delta-lab-iitk.github.io/

[6] S. Castro, D. Hazarika, V. Perez-Rosas, R. Zimmermann, R. Mihalcea,´ and S. Poria, “Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper),” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 4619–4629. [Online]. Available: https://github.

[7] M. O. Kenneth, F. Khosmood, and A. Edalat, “Systematic Literature Review: Computational Approaches for Humour Style Classification,” ArXiv, 1 2024. [Online]. Available: https://arxiv.org/abs/2402.01759

[8] R. A. Martin, P. Puhlik-Doris, G. Larsen, J. Gray, and K. Weir, “Individual differences in uses of humor and their relation to psychological well-being: Development of the Humor Styles Questionnaire,” Journal of Research in Personality, vol. 37, pp. 48–75, 2003. [Online]. Available: www.elsevier.com/locate/jrp

[9] A. Edalat, “Self-initiated humour protocols: An algorithmic approach for learning to laugh,” PsyArXiv, vol. 5, pp. 1–14, 11 2023.

[10] W. P. Hampes, “The Relation Between Humor Styles and Empathy,” Europe’s Journal of Psychology, vol. 6, no. 3, pp. 34–45, 2007. [Online]. Available: www.ejop.org

[11] C. Y. Plessen, F. R. Franken, C. Ster, R. R. Schmid, C. Wolfmayr, A. M. Mayer, M. Sobisch, M. Kathofer, K. Rattner, E. Kotlyar, R. J. Maierwieser, and U. S. Tran, “Humor styles and personality: A systematic review and meta-analysis on the relations between humor styles and the Big Five personality traits,” Personality and Individual Differences, vol. 154, 2 2020.

[12] M. O. Kenneth, F. Khosmood, and A. Edalat, “Explaining Humour Style Classifications: An XAI Approach to Understanding Computational Humour Analysis,” Journal of Data Mining & Digital Humanities, NLP4DH, 1 2025. [Online]. Available: http://arxiv.org/abs/2501.02891

[13] L. Veselka, J. A. Schermer, R. A. Martin, and P. A. Vernon, “Relations between humor styles and the Dark Triad traits of personality,” Personality and Individual Differences, vol. 48, no. 6, pp. 772–774, 4 2010.

[14] I. I. Khramtsova and T. S. Chuykova, “Mindfulness and self-compassion as predictors of humor styles in US and Russia,” Social Psychology and Society, vol. 7, no. 2, pp. 93–108, 2016.

[15] N. Kuiper and N. McHale, “Humor styles as mediators between selfevaluative standards and psychological well-being,” Journal of Psychology: Interdisciplinary and Applied, vol. 143, no. 4, pp. 359–376, 7 2009.

[16] J. E. Yim, “Therapeutic benefits of laughter in mental health: A theoretical review,” pp. 243–249, 7 2016.

[17] N. S. Akimbekov and M. S. Razzaque, “Laughter therapy: A humorinduced hormonal intervention to reduce stress and anxiety,” pp. 135– 138, 1 2021.

[18] T. E. Ford, S. K. Lappi, E. C. O’Connor, and N. C. Banos, “Manipulating humor styles: Engaging in self-enhancing humor reduces state anxiety,” Humor, vol. 30, no. 2, pp. 169–191, 5 2017.

[19] . Menendez-Aller, . Postigo, P. Montes-´ Alvarez, F. J. Gonz<sup>´</sup> alez-Primo,´ and E. Garc´ıa-Cueto, “Humor as a protective factor against anxiety and depression,” International Journal of Clinical and Health Psychology, vol. 20, no. 1, pp. 38–45, 1 2020.

[20] G. Ritchie, The Comprehension of Jokes: A Cognitive Science Framework, 1st ed. London: Routledge Taylor & Francis Group, 7 2018.

[21] M. Kamrul Hasan, W. Rahman, A. Zadeh, J. Zhong, M. Iftekhar Tanveer, L.-P. Morency, and M. Hoque, “UR-FUNNY: A Multimodal Language Dataset for Understanding Humor,” in Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019, pp. 2046–2056. [Online]. Available: www.ted.com

[22] L. Christ, S. Amiriparian, A. Kathan, N. Muller, A. K¨ onig, and¨ B. W. Schuller, “Multimodal Prediction of Spontaneous Humour: A Novel Dataset and First Results,” Transactions on Affective Computing, vol. 20, no. 10, pp. 1–15, 9 2022. [Online]. Available: http://arxiv.org/abs/2209.14272

[23] A. Kuznetsova and C. Strapparava, “Multimodal and Multilingual Laughter Detection in Stand-Up Comedy Videos,” in The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation. ELRA Language Resource Association, 5 2024, pp. 11 884–11 889. [Online]. Available: https://github.com/ MontrealCorpusTools/

[24] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, “MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations,” in Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics. Florence, Italy: Association for Computational Linguistics, 8 2019, pp. 527–536.

[25] D. Singh Chauhan, G. Vikram Singh, A. Arora, A. Ekbal, and P. Bhattacharyya, “A Sentiment and Emotion aware Multimodal Multiparty Humor Recognition in Multilingual Conversational Setting,” in Proceedings of the 29th International Conference on Computational Linguistics,, 10 2022, pp. 6752–6761. [Online]. Available: https: //huggingface.co/

[26] D. S. Chauhan, G. V. Singh, A. Arora, A. Ekbal, and P. Bhattacharyya, “An emoji-aware multitask framework for multimodal sarcasm detection,” Knowledge-Based Systems, vol. 257, 12 2022.

[27] J. Torres-Mar´ın, G. Navarro-Carrillo, and H. Carretero-Dios, “Is the use of humor associated with anger management? The assessment of individual differences in humor styles in Spain,” Personality and Individual Differences, vol. 120, pp. 193–201, 1 2018.

[28] G. E. Weisfeld and M. B. Weisfeld, “Does a humorous element characterize embarrassment?” Humor, vol. 27, no. 1, pp. 65–85, 2 2014.

[29] M. Billig, “Humour and Embarrassment: Limits of ‘Nice-Guy’ Theories of Social Life,” Theory, Culture & Society (SAGE, London, Thousand oaks and new Delhi), vol. 18, no. 5, pp. 23–43, 2001.

[30] C. L. Sabharwal, “Cohen’s Kappa Statistic and newKappaStatistic for Measuring and Interpreting Inter-Rater Agreement,” International Journal of Research in Engineering and Science (IJRES) ISSN, vol. 9, no. 7, pp. 23–28, 7 2021. [Online]. Available: www.ijres.org

[31] S. Sun, “Meta-analysis of Cohen’s kappa,” Health Services and Outcomes Research Methodology, vol. 11, no. 3-4, pp. 145–163, 12 2011.

[32] J. Devlin, M.-W. Chang, K. Lee, K. T. Google, and A. I. Language, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of NAACL-HLT. Minnesota: Association for Computational Linguistics, 6 2019, pp. 4171–4186. [Online]. Available: https://github.com/tensorflow/tensor2tensor

[33] H. Dinkel, Z. Yan, Y. Wang, J. Zhang, Y. Wang, and B. Wang, “Scaling up masked audio encoder learning for general audio classification,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH. International Speech Communication Association, 2024, pp. 547–551.

[34] D. Tran, H. Wang, L. Torresani, J. Ray, Y. Lecun, and M. Paluri, “A Closer Look at Spatiotemporal Convolutions for Action Recognition,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 12 2018, pp. 6450–6459.

[35] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, vol. 13-17-August-2016. Association for Computing Machinery, 8 2016, pp. 785–794.

[36] Y. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, “Realtime Facial Surface Geometry from Monocular Video on Mobile GPUs,” arXiv, 7 2019. [Online]. Available: http://arxiv.org/abs/1907.06724

[37] A. Amjad and R. Dasti, “Humor styles, emotion regulation and subjective well-being in young adults,” Current Psychology, vol. 41, no. 9, pp. 6326–6335, 9 2022.