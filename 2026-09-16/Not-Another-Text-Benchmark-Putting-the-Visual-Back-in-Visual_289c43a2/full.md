# Not Another Text Benchmark: Putting the “Visual” Back in Visual Question Answering for Large Video Models

Rwiddhi Chakraborty<sup>1</sup> Yinong Oliver Wang<sup>2</sup> Cheng Zhang<sup>3</sup> Fan Bai<sup>4</sup> Zhuoran Yu<sup>5</sup> Michael Kampffmeyer<sup>6</sup> Yong Jae Lee<sup>5</sup> Fernando De la Torre<sup>2</sup> Robert Jenssen<sup>6</sup>

<sup>1</sup> University of Copenhagen <sup>2</sup> Carnegie Mellon University <sup>3</sup> Texas A&M University <sup>4</sup> Johns Hopkins University <sup>5</sup> University of Wisconsin-Madison <sup>6</sup> UiT The Arctic University of Norway

## Abstract

Large video models have exhibited impressive performance on a wide range of visual question answering tasks, owing to the rise of powerful, pretrained text and vision encoders. The usefulness of such models have also been demonstrated on a wide range of benchmarks, with an important caveat - the dominant approach in these benchmarks evaluates multiple choice reasoning via text options. This is a natural way to test text-based reasoning in these models, and has led to significant insights regarding model behavior in the community. In this work, we ask a different question - what happens when the evaluation modality is visual, rather than text? We introduce three new vision-centric evaluation benchmarks in temporalframe retrieval, videofuture prediction, and causal memory distortion, all designed around evaluating visual understanding capabilities in large video models. Our approach complements the existing approaches to evaluate video understanding in frontier models. We show that current frontier models exhibit significant weakness when attempting to reason through visual queries, rather than text. We conclude with an extended analysis section that provides pointers for future improvements in visual understanding for large video models.

## 1 Introduction

The rise of large video models in recent years is one of the most interesting developments in multimodal artifical intelligence [37, 8, 35, 45]. These models typically follow the architectural primitives of large vision-language models [42, 19, 17, 2], where pretrained vision and text encoders are aligned such that tokens in both modalities may be interleaved to be interpreted as inputs to the generative decoder. This approach, coupled with training on large amounts of data, has led to impressive visual understanding capabilities of these models [29].

We notice an exclusive design choice in the benchmarks that evaluate video understanding - examples include TemporalBench [5], TempCompass [20], Vinoground [48], MVBench [18], and LVBench [33]. They expose multiple choice questions exclusively in text. While using text as an evaluation modality is valid for the visual question answering (VQA) task, we are motivated to create a more direct evaluation suite for visual understanding, i.e., creating datasets where the evaluation modality is visual. This approach has two benefits - first, it gives us a complementary view in evaluating the same VQA datasets on the same models in a different modality than text. In this way, the visual modality appears as a useful control. Second, this approach allows us to investigate whether frontier video models could demonstrate inflated VQA performance by using language cues in the answer choices. There exists evidence that they do so [1, 13]. In fact, in TemporalBench, the authors in an ablation find that large language models (LLMs) will often use a “centralized” description of the options as a cue to make the final decision. Some of these results thus indicate a greater need for direct visual evaluation of such models. While in this work, we do not directly tackle language shortcuts, through our novel evaluation design, we are able to shed light on certain aspects of language-based biases in large video models. We illustrate our motivation through a simple experiment in Figure 1. On a task where we prompt a state-of-the-art model (Gemini-2.5-Pro [6]) to answer what happens after (or before) a particular action in the input video, the model performs well when the multiple choice options are in text. However, when we replace these text options with visual options (from the actual input video), the performance drop is stark. This phenomenon of incorrect visual grounding will be a recurrent theme in this paper. Targeting large video-language models, we create our own novel evaluation suite based on this theme.

![](images/2da38b5ce66c024d3aff80388d8f21b0a46bdee048fe55c783328d458e10afad.jpg)

![](images/24f1a78ea8e738dafc7d682851da273fcce8f94db0553654fe36655734b7bc9f.jpg)  
Figure 1: A simple illustration of the usefulness of our visual evaluation suite. Traditional evaluation suites in video understanding tasks use text as the evaluation modality (left). We introduce a dataset that evaluates visual understanding in large video models through the visual modality (right). On the temporal frame retrieval task in our dataset, Gemini 2.5 Pro, a frontier video model, performs significantly worse when the answer choices are visual than when they are text (the ∆ sign signifies the drop in accuracy, over 40 percentage points). This demonstrates a gap in the visual reasoning capabilities of large video models.

While recent works in cross-modal consistency have hinted at this problem [32, 49, 21], there are key differences - first, the current approach to check cross-modal consistency is to take images of data in other modalities, e.g., taking a picture of a graph, instead of taking the actual graph. In our work, vision is not a proxy for any other modality. It is the modality of evaluation. Second, none of the existing approaches address inconsistencies in video understanding.

In each of our tasks, the input is a video (a sequence of frames), and a set of visual questions. These questions are either in the format of multiple choices (where each choice is a sequence of frames taken from the context video), or a single, binary decision-making problem, with a single frame as input. Particularly, we choose three tasks:

• Temporalframe retrieval: the model is prompted to choose the correct sequence of frames after or before an anchor event. This task assesses frame retrieval and temporal reasoning of events.

• Videofuture prediction: given query frames, the model is prompted to choose the correct subsequent future frames. This task tests understanding of physical and causal dynamics across time.

• Causal memory distortion: inspired by an influential result in cognitive science [28], we evaluate memory distortion in large video-language models, which tests hallucination of a contact moment (e.g. throwing, kicking something) when the corresponding frame is removed from the input video. Our experimental design closely follows this original study, which shows that people often hallucinate the moment of contact when subsequent footage implies it, but not otherwise. To the best of our knowledge, this is the first evaluation of causal memory distortion in large video-language models.

Our results reveal substantial weaknesses in these core visual tasks across state-of-the-art videolanguage models, introducing new aspects against recent claims that these models are zero-shot reasoners [37]. We conclude with an extended analysis section that sheds insights into future improvement directions in visual understanding. In summary, our contributions include:

• A novel benchmark of visual questions, beyond the current dominant approach of text-based questions in querying video understanding in large video models.

• Evaluating causal memory distortion in large video models, a purely visual task, inspired from cognitive science. To the best of our knowledge, this study does not exist in the current large vision model (LVM) literature.

• Surprising results of poor model performance and extended analysis that sheds insights on model failure modes, paving the way for future improvements in model performance.

## 2 Related Work

Visual Understanding in Video Models. Before the advent of video large multi-modal models (LMMs), pretrained models such as CLIP [23] trained on video datasets exhibited high levels of single frame reliance [14, 4]. In [4], the authors present a CLIP-first approach to extracting a single, representative frame from the input video for a downstream video QA/retrieval task. In [25], the authors shuffle frames in videos and create temporal classes, classes that require temporal reasoning to solve, in contrast to static classes that are time-irrelevant. They show that most major benchmarks overwhelmingly contain static rather than temporal classes. Through our dataset design, we explicitly enforce the temporal aspect of video understanding since all three tasks require the model to reason across frames, rather than a single frame alone - whether it be for retrieval, memory distortion, or future prediction. With the introduction of video LMMs, in addition to the single frame bias, a large body of work showed that such models frequently rely on textual shortcuts when making decisions [30, 44, 36, 26, 15, 12]. More recent appraisals of multimodal LLMs (MLLMs) have focused on hallucination issues [31]. In this case, it seems even more relevant to design visual evaluation suites, to better ascertain the impact of language-based biases in model decision-making. While our work does not directly tackle shortcut learning, we do demonstrate the benefits of using a visual evaluation suite to shed insights on this problem.

Relevant Benchmarks. A number of benchmarks have been proposed to evaluate the capabilities of video LMMs [46, 5, 48, 18, 7, 10], and have been heavily used by the community for evaluation tasks. However, the key approach underlying all these works is the choice of text as the primary query modality. While evaluating text-based reasoning is quite natural for certain VQA tasks, they do not provide a mechanism to more directly evaluate visual reasoning and groundedness, i.e., reasoning through vision rather than text. Thus, our evaluation suite provides a more direct way to compare text-based and vision-based reasoning, and positions itself complementarily to existing appraoches. The most similar approach to ours is the work of [21], where the authors propose similar questions to evaluate audio models. However, there are key differences in our work - first, we operate in a different domain, i.e., video, and second, we note that the issue of memory distortion that we tackle here has not been explored in other works either. We believe this task is a novel way to detect hallucinations in large video models.

Cross-modal consistency. The issue of testing whether models behave consistently when evaluated on the same information across modalities is termed cross-modal consistency, particularly with the popularity of modern LMMs. This is not a new phenomenon. There are earlier works documenting the difficulty of training and evaluating multimodal architectures [34]. [50] train unimodal encoders in an alternative fashion to reduce multimodal interference. More recent works have shifted the focus from representation learning to inference, owing to the heavy cost of training LMMs. [32] evaluate cross-modal consistency using input text and pictures of the same text. [22] evaluate LMMs on visual algorithmic reasoning (latex code as input, as opposed to a picture of a table). Finally, [49] adopt a similar approach to [32], taking screenshots of textual content to use as the image modality. Our work exists in a different domain of video understanding, and does not use image as a proxy modality - we use visual information directly in our evaluation suite.

## 3 The Dataset

Video understanding is fundamentally temporal, causal, and grounded. A model must (i) recover event structure from how content evolves over time, (ii) anticipate what comes next from partial trajectories, and yet (iii) remain faithful to what was actually shown even when the context implies an unobserved event with high probability. These three competencies, although not comprehensive, are a necessary set of requirements that repeatedly arise in real video use cases (e.g., retrieval, monitoring, and summarization); and they expose failures that cannot be diagnosed with static recognition alone. Therefore, we accordingly design three visual-centric tasks that each targets one essential axis of video understanding while keeping the interface simple and comparable across models.

In this section, we detail the dataset design for our three visual tasks tackled in this paper: (1) temporal frame retrieval, (2) video future prediction, and (3) causal memory distortion. For each task, we specify the input/output modalities and construction procedure, and describe the underlying video sources used to generate the evaluation instances.

The unifying view Each task presented in our benchmark is a unique instantiation of the overarching schema: each instance as an input video/video prefix, a visual query, and optionally visual answer candidates. In every case, the model must pick the correct answer based on visual evidence. Based on the exact nature of visual reasoning, the particular task differs. We present the three tasks below.

## 3.1 Temporal Frame Retrieval

This task evaluates a model’s ability to understand temporal context in video. Given a video clip and an anchor action as input, the model is required to select the correct frames that occur either before or after the anchor action from a set of candidate frames. The task is purely visual: the model is required to select among visual options, not text.

We construct this dataset using video clips from the Charades dataset [27], which provides temporally annotated human action sequences, e.g., opening a door, tidying some clothes. These annotations enable the creation of meaningful temporal frame retrieval questions and form the basis for generating questions that require reasoning about temporal relationships between frames.

Given a video clip, we select an anchor action from its annotations and pose two temporal reasoning questions: (i) what happens before the anchor action, and (ii) what happens after the anchor action. Crucially, the model must retrieve the correct answer from a set of candidate visual options (i.e., a visual MCQ), rather than text descriptions.

In our visual MCQ setup, each question is accompanied with four candidate options (as demonstrated in Figure 2):

1. Correct: the true temporal neighbor frames,

2. Scrambled: frames sampled across the video and randomly shuffled,

3. Reversed: the same span of frames as the correct option, played backward,

4. Closest Distractor: the false temporal neighbor frames, i.e. previous action to the anchor if the question is of type “after", and vice versa.

There is only one correct answer per question, although future work could explore multiple correct options. We employ a small set of preprocessing and filtering techniques to make sure we only include high-quality samples.<sup>1</sup> We repurposed the Charades test set of 1862 videos and filtered down to 321 videos. After balancing the before and after questions, we constructed a total of 1108 questions. The correct answer for each question was assigned randomly among the four options. The dataset contains around 35,000 frames that we extract using this process.

## 3.2 Video Future Prediction

This task evaluates a model’s ability to predict future events in a video from past context. Given a video clip in which only an initial frame segment is visible (i.e. only a truncated video as input), the model must select the correct future frames from a set of candidate segments. This is different from the retrieval task, since the model does not have access to the entire video clip to make the decision. Instead, it has to reason based on the truncated context it receives as input. We construct this dataset using video clips from the Shapestacks dataset [11]. Shapestacks contains synthetic videos of 3D geometric objects interacting in a physics-based environment, making it well suited for future prediction tasks.

![](images/c3ce9e693b62b649c4cf431f799d78ccd497fe6598403be4d571218d50800952.jpg)

![](images/37acf3764434465baa43e24fd38a06e9d016eedcf3589295245ff195e67a926e.jpg)

![](images/f930acd1e11ca6eb7c9ca4231de93e20a0d9d0e815c091f1953b4f3776c51d4f.jpg)

![](images/7286187be77c32bb98b5b6b6e9c0fc787accb6eb7c308d15a6d80cdf3d29d274.jpg)  
Figure 2: Temporal Frame Retrieval. Given a video and an action (e.g. “someone goes from standing to sitting”), the model must select the correct sequence of frames that occur after or before the anchor action from four visual options. Green is correct, red incorrect.  
Figure 3: Video Future Prediction. Given a video input showing the initial context frames, the model must select the correct sequence of future frames from four visual options. Green is correct, red incorrect.

The dataset construction proceeds in two stages: (i) video frame extraction and (ii) question-answer generation. In the first stage, we randomly sample videos in a balanced manner across the CCS and Blocks categories, as well as the Easy and Hard difficulty levels, to ensure broad and fair coverage of scenarios. From each selected video, we extract the first eight frames as the context frames and the last eight frames as thefuture frames. All selected videos are manually verified to contain meaningful physical interactions, ensuring that the context frames do not trivially reveal the future frames. In the second stage, we generate multiple-choice questions for input videos, and each question consists of four candidate options:

1. Correct: the ground-truth future segment,

2. Reversed–I: the ground-truth future frames presented in reverse order,

3. Reversed–II: the input (context) frames presented in reverse order,

4. Scrambled: a variant of the correct future sequence in which the first and last frames are preserved, while the intermediate frames are randomly permuted.

As in the previous dataset, each question contains exactly one correct answer, which is randomly positioned among the four options. In total, we finalize 300 video clips, resulting in 300 questions. This process yields 12,000 extracted frames overall. We illustrate the task in Figure 3.

## 3.3 Causal Memory Distortion

This task is inspired by an influential cognitive science experiment by [28], which studies how humans form false memories about causal events in videos. In their setup, a video sequence contains a clear contact moment (e.g., kicking something or throwing something). The video is then slightly altered to remove the contact moment, but subjects are shown a candidate “contact moment” frame from the original video. The study found that humans often falsely report having seen the contact moment in the altered video, effectively “hallucinating”, when the remaining footage causally implies the missing contact. Crucially, this effect depends on causal implication; if the altered video contains evidence consistent with the contact having occurred (e.g., the ball rolls away after the removed kick), false recall is common; if the altered video contains no such implication (e.g., the removed kick is followed by an unrelated scene such as a ship sailing), false recall is much less likely. To the best of our knowledge, we are the first to adapt this idea to curate a dataset that evaluates models’ ability to detect causal inconsistencies in video sequences, i.e. evaluating whether models can indeed stay robust when contact frames are removed with causal implication, as opposed to when they are not. We use the term “causal memory distortion” as the phenomenon when models exhibit similar hallucination about the removed “contact” frame.

![](images/7f293fb4a0f4f06cf42bf108bc014ed25cded25dd8955b6eff5a1daeb32b5b22.jpg)  
Figure 4: An illustration of the Causal Memory Distortion task. The contact frame (the moment the shoe leaves the foot) is highlighted in green. We denote w/CI–with causal implication, and w/o CI–without causal implication. The contact frame is present in the complete versions across the three conditions, but is absent in the incomplete versions.

To construct this dataset, we again use the Charades dataset. Since this task requires fine-grained control on the video editing, we manually selected videos from the Charades test set that satisfied the following three criteria: (i) the contact moment must be clear and long enough to be visible, (ii) the video must have a clear causal implication after the contact moment, and (iii) the video must also have a moment of clear non-causal implication. We do not require that the non-causal implication frames necessarily follow the contact moment immediately. After an extensive manual search, we finalized 20 videos that satisfied these criteria. While the dataset size is seemingly small, it is three times larger than the original work by [28], which had only six videos. Further, the code to annotate more videos is extensible, and it is possible to add more videos beyond Charades for future work.

With appropriate raw videos selected, we manually constructed the task setup, including the contact moment and relevant frames. We categorize two types of videos: complete - contact moment is not removed, and incomplete - contact moment is removed. Then, for each video type, three conditions were created (as illustrated in Figure 4):

1. With Causal Implication: The video segment following the contact moment was preserved.

2. Without Causal Implication: The video segment following the contact moment was replaced with a random segment causally unrelated to the contact moment.

3. Scrambled: The video segments were randomly shuffled, providing the necessary control.

In the end, we had six variants per original video (two video types, three conditions). During evaluation, the complete and incomplete video types were randomly assigned to each model. The task thus evaluates whether the model is more likely to hallucinate the contact moment when it is removed from the original video but followed up with a clip that implies a causal connection.

A note on benchmark size and scaling The dataset currently has 641 videos. Temporal retrieval scales to any dataset with timestamped descriptions; video future prediction scales via the physics simulator. The causal memory distortion task in particular is a bit difficult to scale, given the heavy manual curation required. However, our dataset has three times the number of videos from the original study. As we are not targeting a new state-of-the-art, we believe our sample size is reasonable for analyzing frontier models. Our 95% confidence intervals (bootstrap B=10000 and Wilson score) are tight, confirming sufficient dataset size for reliable conclusions.

## 4 Experiments

## 4.1 Setup

Models For our work, we use Gemini 2.5 Pro and GLM4.6-V [6, 47] as the proprietary frontier video models, with LLaVA-OV-72B [16], Qwen-2.5-72B, Qwen-2.5-32B, and Qwen-3-32B [42], mPlug-Owl-3 [43], and Internvl3-78B [51] as the open-source frontier video models. We use fixed prompts for each task across the models to ensure consistency. We use models off the shelf through open-source downloads on Huggingface [38], or frontier model APIs.

Metrics Unless otherwise noted, we present the accuracy of the models on all our tasks. For MCQ, the accuracy is measured from the selection of the ground truth option. For the Causal Memory Distortion task, the accuracy is measured based on whether the model predicts yes or no regarding the frame in question. We observe that the evaluated models follows instruction perfectly and always output the selected option or yes/no depending on the task.

Human baseline To verify that the tasks are well-posed and solvable, we tested ten human subjects on a subset of the dataset and compared with the models. However, directly comparing humans and models here raises scientific control issues: humans have short-term memory providing semantic access to prior data, whereas each model inference is a fresh start. Some benchmarks include human studies, but these remain contentious, as designing sound human-model evaluations is an open question and a direction for future work. We therefore present the human results only to demonstrate that the tasks are solvable, and make no claims about human-model performance comparisons.

We leave more information on the exact evaluation formats, prompt structures, and data formats in Section B of the appendix.

## 4.2 Main Results

The main results are presented in Table 1. Overall, the results demonstrate a surprisingly significant decline by simply replacing text questions with visual questions. On the temporal frame retrieval task, we observe that all models perform poorly (slightly above random chance). On the future prediction task, we observe that Gemini significantly outperforms the open-source models, and Qwen models perform slightly above the random baseline. We also note that human performance is meaningfully above chance across all tasks, which we believe is the clearest evidence against the benchmark being ill-posed. Human accuracy on VFP (80%) and CMD (80%) is substantially higher than chance, indicating those tasks are well within human capability. The relatively lower human accuracy on TFR reflects genuine task difficulty: discriminating temporally adjacent action segments from a 30-second video is non-trivial even for humans. The results show a range of difficulty across tasks, but crucially, all remain meaningfully above random performance. Second, we acknowledge that certain evaluation components, particularly those involving shuffled or reversed frames, were not designed with human viewing in mind and may therefore hinder human performance on those specific tasks. This is a known property of such designs rather than a flaw unique to our benchmark. Shuffled and reversed frame orderings are well-established conventions in video understanding benchmarks, and notably, prior work adopting these designs has not always included human studies — for instance, AoT [41], RTime [9], employ shuffled and reversed video evaluation without a human baseline.

Finally, on the causal distortion task, we present a further analysis of statistical significance and the sensitivity-specificity rates to help in the interpretation of the results, owing to comparatively lower sample size. These results are in appendix F.1 and F.2. We observe the rather interesting result that the specificity rate of Gemini is 30 points lower than Qwen, suggesting that it is much more likely to predict “yes”, i.e., it hallucinated the contact frame at a much higher rate than Qwen.

Table 1: Accuracy (%) across the three tasks. Best result per column in bold. The causal memory distortion task is binary (“yes”/“no”), hence the 50% random baseline. The other two tasks have four choices each, hence the 25% random baseline.
<table><tr><td>Model</td><td>Temporal Frame Retrieval</td><td>Future Prediction</td><td>Causal Memory Distortion</td></tr><tr><td>Random Baseline</td><td>25.00</td><td>25.00</td><td>50.00</td></tr><tr><td>Human</td><td>44.00</td><td>80.00</td><td>80.00</td></tr><tr><td>LLaVA-OV-72B</td><td>23.62</td><td>25.75</td><td>52.00</td></tr><tr><td>Qwen2.5-32B</td><td>24.71</td><td>33.44</td><td>55.00</td></tr><tr><td>Qwen2.5-72B</td><td>25.83</td><td>31.70</td><td>60.00</td></tr><tr><td>mPLUG-Owl3</td><td>17.69</td><td>17.40</td><td>51.70</td></tr><tr><td>Qwen3-32B</td><td>21.57</td><td>41.80</td><td>50.00</td></tr><tr><td>InternVL3-72B</td><td>21.48</td><td>25.80</td><td>50.00</td></tr><tr><td>GLM-4.6V</td><td>29.78</td><td>32.44</td><td>51.70</td></tr><tr><td>Gemini-2.5-Pro</td><td>30.00</td><td>69.62</td><td>58.00</td></tr><tr><td>Kimi-K2.6</td><td>33.45</td><td>28.43</td><td>58.33</td></tr><tr><td>GLM-5V-Turbo</td><td>23.87</td><td>24.08</td><td>55.00</td></tr></table>

## 5 Analysis

The broad theme of our paper is to uncover the tension between two modalities towards the evaluation of frontier video models. We seek to elaborate on our main findings with key experiments further shedding insight on the interplay between the text and vision modalities.

## 5.1 Comparison with text-based evaluation

We begin with the most natural question: given the poor performance ofmodels on our visual-centric tasks, how well do models perform ifthe evaluation setup is text-based, which is the current approach taken towards benchmark design? We conduct this analysis along two axes. First, we use the temporal frame retrieval task built on Charades, which provides time-stamped action descriptions that can be used directly as text options<sup>2</sup>. This tests whether there is a meaningful difference if text options are used as-is. Second, to test whether the gap persists when visual content is converted to text rather than drawn from pre-existing annotations, we use GPT-5.4 to caption the visual options from the Shapestacks task and evaluate Gemini on this distilled text-based version of the dataset. This tests whether there is a meaningful difference if visual data is first converted into text, and then evaluated.

As illustrated in Figure 5, we observe a significant drop in performance across all models when the evaluation modality shifts from text to visual on the Charades task. However, in the distillation experiment on Shapestacks (Figure 6) accuracy drops from 69.62 with visual options to 45.12 when those options are distilled into text captions. This loss of fine-grained visual detail through textual distillation is consistent with prior observations [24]. Taken together, we see that text-based evaluations may inflate performance when text options are available as-is, but the opposite may happen if visual options are converted into text, owing to loss in fine-grained information. These results demonstrate the benefit of our visual benchmark as a complementary alternative to text-based evaluations.

## 5.2 Reasoning Patterns

We manually evaluate reasoning traces from Gemini 2.5 Pro and Qwen2.5-72B on the temporal frame retrieval task by extracting and annotating claims for factual grounding. The two models show distinct reasoning patterns. Gemini first describes the video broadly, then analyzes options one by one. While it correctly identifies semantic content and temporal structure, it frequently misidentifies reversed sequences and fails to map its reasoning to the correct visual option—a failure of option grounding, illustrated in Figure 7. Qwen, by contrast, ignores the visual options entirely, listing discrete events before selecting an answer. It correctly identifies events in ∼66% of cases, but without grounding to the options, performance suffers—a failure of query grounding, illustrated in Figure 8. Both patterns illustrate our core motivation: models reason well on text but struggle to ground reasoning in the visual modality. Addressing option grounding in Gemini and query grounding in Qwen are natural future directions, though prompting strategies remain unexplored. We present sample traces and other details on reasoning and error patterns in Section D, and Section E respectively.

![](images/173833ad457bf196357bcdd6d6239c8a27546628c678e0a46e2986ea71a43501.jpg)  
Figure 5: Comparison of model accuracies on the Temporal Frame Retrieval task when evaluated with text options vs. video options. The significant drop in performance (large ∆) when using visual options highlights the limitations of text-based evaluations.

![](images/4533338ce67db8225e444e8c6a24c88dc4adbf7442e36500a353b0cff6906b9f.jpg)  
Figure 6: Gemini performance on the future prediction task when visual options are distilled into text captions. The substantial drop (large ∆) indicates that captioning loses fine-grained visual information needed to disambiguate options.

![](images/628791e6c617e55dc6589a3ccd43815668daf148be4c2c8ba1e648eecb4ccffb.jpg)  
Figure 7: We analyze Gemini2.5-Pro’s reasoning traces and find strong event understanding for video clips, grounded in both the context video and visual options. However, it often confounds the final answer: in the figure above, it correctly identifies option B in its reasoning, yet outputs option C.  
Figure 8: We analyze Qwen2.5-72B’s reasoning trace and find strong event understanding for the context video (it identifies key actions), but it fails to ground its decision in the visual options. In the illustration, it skips analyzing the options entirely and selects option C, when the correct answer was B.

## 6 Limitations

First, we wish to further expand the dataset we created. This will not be trivial - in addition to the manual curation required for the causal memory distortion task, picking appropriate datasets for the other two tasks would required careful deliberation as well. Second, our analysis in this paper was entirely black-box, i.e. they do not include a key complementary view of our paper - those of model representations and internal mechanisms. In the future, we would like to explore white-box perspectives to this work, particularly the role of the vision encoder in inference.

## 7 Conclusion

In this work, we selected a particular domain, that of video understanding in VQA tasks, and introduced a new evaluation approach for frontier video models - visual queries. Through three fundamental tasks in temporal frame retrieval, video future prediction, and causal memory distortion, we attempted to shed light on the role visual information plays as an evaluation modality for frontier video models, as opposed to text. Our results demonstrated that there are still significant weaknesses in visual understanding capabilities for these models. We believe our work provides a complementary view to the existing text benchmarks - it does not replace them, but rather seeks to complement them to improve our own understanding of how these models operate. Finally, we also hope that this new evaluation approach can facilitate more analyses on language-based biases which text-based reasoning benchmarks may sometimes be susceptible to.

## References

[1] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. Don’t just assume; look and answer: Overcoming priors for visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4971–4980, 2018. 2

[2] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024. 1

[3] Hritik Bansal, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang, and Aditya Grover. Videocon: Robust video-language alignment via contrast captions. In Proceedings ofthe ieee/cvfconference on computer vision and pattern recognition, pages 13927–13937, 2024. 18

[4] Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the" video" in video-language understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2917–2927, 2022. 3

[5] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. ArXiv, abs/2410.10818, 2024. 1, 3

[6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2, 7

[7] Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees G. M. Snoek, and Yuki M. Asano. Lost in time: A new temporal benchmark for videollms. 2024. 3

[8] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104, 2025. 1

[9] Yang Du, Yuqi Liu, and Qin Jin. Reversed in time: A novel temporal-emphasized benchmark for cross-modal video-text retrieval. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 5260–5269, 2024. 7

[10] Rohit Girdhar and Deva Ramanan. Cater: A diagnostic dataset for compositional actions & temporal reasoning. In International Conference on Learning Representations. 3

[11] Oliver Groth, Fabian B Fuchs, Ingmar Posner, and Andrea Vedaldi. Shapestacks: Learning vision-based physical intuition for generalised object stacking. In Proceedings ofthe european conference on computer vision (eccv), pages 702–717, 2018. 4

[12] Jihyung Kil, Cheng Zhang, Dong Xuan, and Wei-Lun Chao. Discovering the unknown knowns: Turning implicit knowledge in the dataset into explicit training examples for visual question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6346–6361, 2021. 3

[13] Dohwan Ko, Ji Lee, Woo-Young Kang, Byungseok Roh, and Hyunwoo Kim. Large language models are temporal and causal reasoners for video question answering. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 4300–4316, 2023. 2

[14] Jie Lei, Tamara Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 487–507, 2023. 3

[15] Mosh Levy, Shauli Ravfogel, and Yoav Goldberg. Guiding llm to fool itself: Automatically manipulating machine reading comprehension shortcut triggers. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8495–8505, 2023. 3

[16] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research. 7

[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 1

[18] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 1, 3

[19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1

[20] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024. 1

[21] Le Thien Phuc Nguyen, Zhuoran Yu, Samuel Low Yu Hang, Subin An, Jeongik Lee, Yohan Ban, SeungEun Chung, Thanh-Huy Nguyen, JuWan Maeng, Soochahn Lee, et al. See, hear, and understand: Benchmarking audiovisual human speech understanding in multimodal large language models. arXiv preprint arXiv:2512.02231, 2025. 2, 3

[22] Simon Park, Abhishek Panigrahi, Yun Cheng, Dingli Yu, Anirudh Goyal, and Sanjeev Arora. Generalizing from simple to hard visual reasoning: Can we mitigate modality imbalance in vlms? In International Conference on Machine Learning, pages 48192–48244. PMLR, 2025. 3

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3

[24] Simon Schrodi, David T Hoffmann, Max Argus, Volker Fischer, and Thomas Brox. Two effects, one trigger: On the modality gap, object bias, and information imbalance in contrastive visionlanguage models. In The Thirteenth International Conference on Learning Representations. 8

[25] Laura Sevilla-Lara, Shengxin Zha, Zhicheng Yan, Vedanuj Goswami, Matt Feiszli, and Lorenzo Torresani. Only time can tell: Discovering temporal data for temporal modeling. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 535–544, 2021. 3

[26] Kazutoshi Shinoda, Saku Sugawara, and Akiko Aizawa. Which shortcut solution do question answering models prefer to learn? In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press, 2023. 3

[27] Gunnar A Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In European conference on computer vision, pages 510–526. Springer, 2016. 4

[28] Brent Strickland and Frank C. Keil. Event completion: event based inferences distort memory in a matter of seconds. Cognition, 121 3:409–15, 2011. 2, 5, 6, 16, 22

[29] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 1

[30] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022. 3

[31] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9568–9578, 2024. 3

[32] Angela van Sprang, Laurens Samson, Ana Lucic, Erman Acar, Sennay Ghebreab, and Yuki M Asano. Same content, different answers: Cross-modal inconsistency in mllms. arXiv preprint arXiv:2512.08923, 2025. 2, 3

[33] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958– 22967, 2025. 1

[34] Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networks hard? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12695–12705, 2020. 3

[35] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European conference on computer vision, pages 396–416. Springer, 2024. 1

[36] Zhecan Wang, Long Chen, Haoxuan You, Keyang Xu, Yicheng He, Wenhao Li, Noel Codella, Kai-Wei Chang, and Shih-Fu Chang. Dataset bias mitigation in multiple-choice visual question answering and beyond. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 8598–8617, 2023. 3

[37] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 1, 3

[38] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019. 7

[39] Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, and Hang Liu. A survey of uncertainty estimation methods on large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21381–21396, 2025. 21

[40] Miao Xiong, Andrea Santilli, Michael Kirchhof, Adam Golinski, and Sinead Williamson. Efficient and effective uncertainty quantification for LLMs. In Neurips Safe Generative AI Workshop 2024, 2024. 21

[41] Zihui Xue, Mi Luo, and Kristen Grauman. Seeing the arrow of time in large multimodal models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. 7

[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1, 7

[43] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. In The Thirteenth International Conference on Learning Representations. 7

[44] Keren Ye and Adriana Kovashka. A case study of the shortcut effects in visual commonsense reasoning. In Proceedings ofthe Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI), February 2021. 3

[45] Lin Yen-Chen. Video generation models explosion 2024. yenchenlin.me, Jan 2025. 1

[46] Chenyu Yi, Siyuan Yang, Haoliang Li, Yap-Peng Tan, and Alex C Kot. Benchmarking the robustness of spatial-temporal models against corruptions. In 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Neural Information Processing Systems (NeurIPS), 2021. 3

[47] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025. 7

[48] Jianrui Zhang, Mu Cai, and Yong Jae Lee. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763, 2024. 1, 3

[49] Xiang Zhang, Senyu Li, Ning Shi, Bradley Hauer, Zijun Wu, Grzegorz Kondrak, Muhammad Abdul-Mageed, and Laks VS Lakshmanan. Cross-modal consistency in multimodal large language models. arXiv preprint arXiv:2411.09273, 2024. 2, 3

[50] Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, and Huaxiu Yao. Multimodal representation learning by alternating unimodal adaptation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 27456–27466, 2024. 3

[51] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 7

## Appendix

## Impact Statement

This work advances the evaluation of video–language models with a special focus on video-centric tasks. By pushing evaluation toward stronger visual evidence requirements, our benchmark promotes robustness and reliability, helping extend the practical application boundary of video–language models. Beyond ranking models, it provides diagnostic value: our tasks and analyses offer clear grounds to identify specific strengths and weaknesses under vision-centric conditions, informing targeted model improvements. There are, however, always negatives associated with using frontier models, particularly with respect to biased/problematic content generation or interpretation. We have, through our experiments, ensured to the best of our abilities that no such possibilities arose, through the dataset design. To facilitate responsible adoption, we will release the benchmark with clear documentation and recommended evaluation practices, positioning it as a complement to existing model evaluations. The total compute used in this work is 240 GPU hours on NVIDIA GH200 (96GB) and 147 CPU hours.

## A The Dataset

In this section we furnish some additional details on the construction of the temporal retrieval task, the future prediction task, and the causal memory distortion task.

## A.1 Statistics

In Table 2, we present detailed statistics on our benchmarks, including the number of videos, questions, frames, duration, and difficulty based on human judgement. Note, the frames column describes the total number of frames the model sees in the input query over the whole data.

<table><tr><td>Dataset</td><td>Videos</td><td>Questions</td><td>Frames</td><td>Duration</td><td>Difficulty</td></tr><tr><td>Charades</td><td>321</td><td>1,108</td><td>35,000</td><td>~30s</td><td>Hard</td></tr><tr><td>Shapestacks</td><td>300</td><td>300</td><td>12,000</td><td>1-3s</td><td>Easy</td></tr><tr><td>Causal</td><td>20</td><td>60</td><td>20</td><td>~30s</td><td>Moderate</td></tr></table>

Table 2: Dataset statistics.

## A.2 Temporal Frame Retrieval

There are 157 action classes in the Charades dataset, with timestamped annotations for each video. The annotations look like so:

c077 12.10 18.00;c079 11.80 17.30;c080 13.00 18.00;

c076 11.80 17.50;c075 5.40 14.10

Given these annotations, we can map them to actual actions using the \_classes.txt file, where the text descriptions of these action ids reside.

## A.2.1 Preprocessing

We filter actions to keep only those with durations between 1 and 5 seconds.

1. Actions shorter than 1 s or longer than 5 s are dropped.

2. Videos themselves are never removed.

3. We set thresholds of less than 1s to remove micro-actions, and more than 5s to remove general actions that are true overall but not for specific contexts. For example, in

c015 0.00 32.10;c087 0.60 32.10;c016 0.00 32.10;c154 0.00 31.8

both c015 (Holding a phone), and c154 (Standing up somewhere) are included, which are not specific enough as actions. In fact, all four actions annotated in this video are true for the whole

video. We keep only discrete actions in their respective temporal windows, and not general actions that are true for the whole video.

We present an example JSON entry from the data:

```csv
"0HGNK": [
{
"id": "0HGNK_0000",
"video_id": "0HGNK",
"type": "after",
"question": "What happens after the following action: Someone is going from standing to
sitting?",
"options": {
"A": {
"frames": [],
"span": [
10.3,
15.2
],
"class_id": "c152",
"text_option": "Opening a book",
"label": "incorrect",
"reverse_order": true,
"distractor_type": "reversed"
},
"B": {
"frames": [],
"span": [
18.376751109861242,
22.95627907087428
],
"class_id": "c150",
"text_option": "Someone is running somewhere",
"label": "incorrect",
"distractor_type": "random",
"scrambled": true
},
"C": {
"frames": [],
"span": [
0.0,
3.8
],
"class_id": "c151",
"text_option": "Someone is going from standing to sitting",
"label": "incorrect",
"distractor_type": "neighbor"
},
"D": {
"frames": [],
"span": [
10.3,
15.2
],
"class_id": "c152",
"text_option": "Someone is smiling",
"label": "correct"
}
},
"metadata": {
"context_class_id": "c151",
"correct_class_id": "c152",
"context_span": [
0.0,
3.8
],
"correct_span": [
10.3,
```

15.2   
],   
"correct\_option": "D"   
}

All the necessary information required to load the correct options are apparent in this structure. We will release all the JSON files for the three tasks.

## A.3 Video Future Prediction

Shapestacks is a 3D simulation dataset that creates RGB renderings of geometrical primitives (e.g. cube, cuboids, cylinders, spheres) under two structurally unstable configurations - a center-of-mass violation, and a planar surface violation.

Each configuration is of the form:

env\_<subset\_name>-<height>-<vcom>-<vpsf>-<version>

## where:

1. subset\_name: Indicates whether the configuration contains only cubes (blocks) or cuboids, cylinders and spheres (ccs).

2. height h: the number of objects in the stack

3. vcom: the layer of the stack in which the stability violation occurs. Layers are counted from bottom (=1) to top (=6).

4. vpsf: annotates the layer of the stack in which a planar surface violation occurs. Layers are counted from bottom (=1) to top (=6).

5. version v: the version number of this scenario, since scenarios with the same parameters have been sampled multiple times.

There are both easy and hard versions of these configurations. In our dataset, we sample in a balanced manner across both the blocks and ccs types, and the easy and hard configurations. We truncate each video to eight input frames, and the remaining eight as label frames. Then, we create the four candidate options as discussed in the main paper.

Here is an example of the JSON entry:

"env\_ccs-hard-h=4-vcom=1-vpsf=0-v=98": {   
"correct\_option": "C",   
"options": {   
"C": "correct",   
"A": "rev\_correct",   
"B": "scrambled",   
"D": "rev\_input"   
}

Each of the options are saved as frames in their respective subdirectories under the main directory (here, env\_ccs-hard-h=4-vcom=1-vpsf=0-v=98). In the end, we have 300 videos.

## A.4 Causal Memory Distortion

Since the original study by [28] did not release the videos, we created our own annotation tool, and used the Charades dataset as the base to create 20 clips, following the design protocol in the original study. We do not include fillers or lures as these were added as additional controls for human distractions, which we believe do not readily apply to the model case. In our annotation tool, we manually select the temporal window where the contact (e.g. throwing a ball) occurs, and make sure to remove successive frames from the contact frame, as the task is not fine-grained frame retrieval. Rather, we want a set of frames following the contact frame that are either causally indicative of contact, or not. For example, if a person is throwing a ball, we do not need all the frames capturing the ball’s trajectory. We need the key frame, i.e. the moment where the ball leaves the hand (contact), and the following frames as the ball in the air/on the ground. This is a causal implication, and is thu not strictly a fine-grained frame retrieval task.

Here is an example JSON entry:

{   
"video\_id": "10M0F",   
"versions": {   
"complete": "causal/benchmark/videos/10M0F\_complete.mp4",   
"incomplete\_with\_implication": "causal/benchmark/videos/10   
M0F\_incomplete\_with\_implication.mp4",   
"incomplete\_without\_implication": "causal/benchmark/videos/10   
M0F\_incomplete\_without\_implication.mp4",   
"scrambled": "causal/benchmark/videos/10M0F\_scrambled.mp4"   
},   
"frames":   
{   
"frame\_id": "contact",   
"frame\_type": "contact",   
"path": "frames/10M0F\_contact.jpg",   
"answers": {   
"complete": "yes",   
"incomplete\_with\_implication": "no",   
"incomplete\_without\_implication": "no",   
"scrambled": "no"   
}   
}   
}

As in the original paper, we make sure the clips are all time-matched to within 1 second of the complete video with causal implication, ensuring that the number of frames the model sees will be approximately equal.

## B Evaluation

As opposed to the standard text-based evaluation, our dataset allows us to interleave visual inputs in the prompt. Here, we discuss the task specific prompt templates used in our paper. In general, we use the transformers library from HuggingFace for all the open-source models, which ha a convenient interface to interleave the frame options and pass it to the model. We take the same approach for the Gemini API as well.

## B.1 Temporal Frame Retrieval

In the temporal frame retrieval task, the example prompt looks like:

You are an expert at video understanding. You will be shown a video and multiple-choice   
question about events in it.\n\n"   
f"Question: {q}\n\n"   
"Options:"   
"A": <Load frames from Option A>   
"B": <Load frames from Option B>   
\n\n   
Show your reasoning steps, and then end your response with "   
"\"The final answer is: <your answer from options A/B/C/D>\"

Here, q is loaded from the JSON file presented earlier. We extract the answers and match them with the ground truths to compare the accuracy.

## B.2 Video Future Prediction

In this setting, we load the input frames from the appropriate paths for each video id, and the prompt structure looks like so:

"You are shown an initial state of a block tower, followed by four possible outcomes (A, B, C, D).\n\nQuestion: Which option shows the most likely outcome for this tower after some time passes?\n\nConsider physics and stability when making your prediction .\n\nInitial tower state:

<video frames>   
\n\nShow your reasoning steps, and then end your response with "   
"\"The final answer is: <your answer from options A/B/C/D>\

We extract the answers and match them with the ground truths to compute the accuracy.

## B.3 Causal Memory Distortion

Since this task is not multiple choice, we use a slightly different evaluation scheme.

As in the original paper, the complete and incomplete assignments to each model are randomized. Since there are twenty videos, we assign ten complete and ten incomplete videos to each model. Since each video type has three conditions (with implication, without implication, scrambled), we have sixty clips in total (thirty with the complete type, and thirty with the incomplete).

We first iterate through all the complete types, and then the incomplete types. In both cases, the prompt structure is the same:

"<video\_frames>   
\n Did this frame appear in the video?: <contact\_frame>   
\n Answer only Yes or No."

Note that in this task, the ground truth is simpler - for the complete video type, the ground truth is always “Yes", since by construction, the contact frame always appear. In the incomplete video type, the ground truth is always “No", since by construction, the contact frame does not appear.

## C Constructing the text options

For the text options, scrambled/reversed descriptions are sampled from different videos in the corpus to avoid confounders, since these sequences are not real actions in the video. The correct option and closest distractor use exact Charades timestamp descriptions. To ensure that this design does not create easy negatives, we use VideoCon [3], to create harder textual negatives and evaluate Gemini on it as a sanity check against artificial inflation of performance. If distractor difficulty rather than modality were the primary factor, we would expect a substantial drop in performance on this harder caption set. As shown in Table 3, this is not the case: accuracy drops only marginally from 73.87 (original text) to 71.48 (hard text), while the gap to visual querying (30.00) remains large. We believe this supports our hypothesis that visual querying exposes gaps in fine-grained temporal understanding that textual querying alone cannot.

<table><tr><td>Modality</td><td>Acc.</td></tr><tr><td>Text [Original]</td><td>73.87</td></tr><tr><td>Text [Hard]</td><td>71.48</td></tr><tr><td>Visual</td><td>30.00</td></tr></table>

Table 3: Accuracy for different text negatives

## D Reasoning Analysis

In this section, we outline the steps taken to generate the results in figures 8 and 7. Given the reasoning trace in the model output, we first extract claims using a large language model (GPT5.2). These models are quite capable today for this simple task. We ensure to extract only the claims (facts/assertions) in the trace, and not meta-resoning, decision justification, or final answer statements. We extract only the incorrect predictions to probe where the models went wrong. We present below an example entry from the claim JSON we create for Qwen72B on Charades:

![](images/ab1129d61e68d222e0429e9184e599fe77e0af2b2bd1d7c418afeff601b3a24b.jpg)  
Figure 9: The pipeline to analyze the reasoning traces in Gemini and Qwen72B

"id": "T5ECU\_0000",   
"predicted": "C",   
"correct": "B",   
"claims": [   
"1. Initially, there are two people standing near a window.",   
"2. One person appears to be holding a cup and drinking from it.",   
"3. The other person seems to be adjusting something on the window or interacting with   
the blinds.",   
"4. There is no clear indication of anyone sitting down in the provided frames."   
]   
},

## For Gemini, we present an example entry below:

"1. \*\*Analyze the initial action:\*\* The video begins in a dimly lit room. At timestamp 00:02, the person reaches up and the room becomes brightly lit. This is the action of \"Turning on a light.\"",

"2. \*\*Observe the subsequent events:\*\* Immediately after the light turns on, the person begins to remove their blue hoodie. They pull it over their head, revealing a white t-shirt underneath.",

"3. \*\*Compare with Option A:\*\* This option shows the person, in the now brightly lit room, taking off their blue hoodie. This sequence of movements directly follows the action of turning on the light.",

"4. \*\*Compare with Option B:\*\* This option shows the person after the hoodie is already off. They are holding a yellow ball and interacting with items on a dresser. This happens later in the video, after the hoodie has been removed.",

"5. \*\*Compare with Option C:\*\* This option is a reversed or jumbled version of the   
events shown in Option B. It does not correctly follow the initial action.",

"6. \*\*Compare with Option D:\*\* This option includes the moment the light is turned on and then shows the person taking off the hoodie. However, the question asks what happens \*after\* the light is turned on. Option A is a more precise answer as it depicts only the events that follow the specified action."

Given these claims, we manually watch the videos and annotate which of these are true, and which are false. This allows us to evaluate how specifically the models fail at this task. The revelations of option confounding in Gemini, and option skipping in Qwen72B were thus possible using this approach. We illustrate this pipeline in Figure 9.

Next, we also analyzed incorrect Gemini reasoning traces on Shapestacks and identified two dominant failure patterns. Gemini demonstrates strong physics reasoning in the majority of its errors: it correctly predicts that unstable towers will collapse. The core failure mode is sequence order confusion (52.9%): the model confounds a reversed or scrambled option as "smooth" and "physically plausible" while dismissing the correct option as "jumbled" or "reversed". It cannot reliably distinguish forward from backward frame order. The second mode, static ambiguity (34.5%), arises when all options appear visually identical (stable towers with no motion), making selection effectively random.

## E Error Analysis

On temporal frame retrieval, both models’ dominant error is the neighbor distractor (Gemini 37.5%, Qwen 41.7%) – the temporally adjacent but wrong action – suggesting models struggle most when fine-grained temporal discrimination is required. The reversed distractor is the least confusing (Gemini 29.4%, Qwen 24.8%), indicating some ability to detect temporal direction. Both models also perform substantially worse on ’before’ questions (Gemini 27.0% vs 34.6%, Qwen 19.8% vs 27.7%), revealing that backward temporal reasoning is harder than forward. On Shapestacks, the models exhibit different failure modes: Qwen’s dominant error is the reversed correct sequence (55.4%); it identifies the right visual content but cannot distinguish temporal direction. Gemini’s dominant error is the scrambled sequence (44.8%), suggesting weaker sensitivity to frame ordering. Finally, on causal memory distortion (Appendix D.1), Gemini hallucinates contact frames at higher rates than Qwen, in addition to higher false alarm rates on average.

![](images/2e0d353f10099aa98d6ef2813508a56ab5642cac4121b50a746b16cccfe9dc1c.jpg)

![](images/68fce0116989d9221dc5b6df44a265e4d14d6e1155ea4806f43267e17e46223b.jpg)  
Figure 10: Average hit rates and false alarm rates on Gemini (left) and Qwen72B (right)

## F Additional Analysis

## F.1 Variance on Causal Memory Distortion

Due to the low number of videos in the causal task (twenty), we re-ran the experiment three times to get the means and standard deviations for Qwen72B and Gemini. We did not choose the other models in this task as these two models turned out to be the best in the causal task. We visualize the average hit rates and false alarm rates on Qwen72B, in figure 10.

For Gemini, we observe higher false alarm rates on average than Qwen72B, suggesting it’s tendency to answer in the affirmative ("Yes") most of the time. On Qwen72B, we observe lower false alarm rates on average, explaining how the overall accuracy for this model was higher than Gemini. Overall, we find that the strong confounding effect of the incomplete condition with implication as reported in the original paper was only partially observed in Gemini, but not observed at all in Qwen72B. In fact, in the latter model, on average the false alarm rate on the incomplete condition with causal implication was lower than without. However, we make no claims of statistical significance here, only about the means. We can conclusively conclude that Gemini does hallucinate frames more on average then Qwen72B, but we cannot conclusively conclude that within condition differences are statistically significant.

## F.2 Sensitivity and Specificty on Causal Memory Distortion

Table 4: Signal detection performance across models. Sensitivity = hit rate; Specificity = 1 − false alarm rate. $d ^ { \prime } = z ( \mathrm { H i t } ) \bar { } - z ( \mathrm { F A } )$ measures discriminability; $\begin{array} { r } { c = - \frac { 1 } { 2 } ( z ( \mathrm { H i t } ) + z ( \mathrm { F A } ) ) , } \end{array}$ ) measures response bias, where c > 0 indicates a conservative $( ^ { 6 6 } \mathrm { n o } ^ { 3 3 } )$ bias and $c < 0$ indicates a liberal $( ^ { 6 6 } \mathbf { y e s } ^ { , 3 } )$ bias.
<table><tr><td>Model</td><td>Sensitivity (%)</td><td>Specificity (%)</td><td>Accuracy (%)</td><td> $d ^ { \prime }$ </td><td>C</td></tr><tr><td>Qwen2.5-72B</td><td>57</td><td>63</td><td>60</td><td>0.51</td><td>+0.08</td></tr><tr><td>Gemini2.5-Pro</td><td>83</td><td>33</td><td>58</td><td>0.51</td><td>-0.70</td></tr><tr><td>Chance</td><td>50</td><td>50</td><td>50</td><td>0.00</td><td>0.00</td></tr></table>

Finally, on the causal distortion task, we find the surprising result where Qwen2.5-72B outperforms Gemini 2.5 Pro slightly, suggesting that a further analysis of the sensitivity and specificity rates could help to understand this observation. The former measures how well the model performs on true positives, i.e., the model predicted “yes” when the ground truth was “yes” (the complete video type). The latter measures how well the model performs on true negatives (i.e., the model predicted “no” when the ground truth was “no”). Our results in Table 4 offer evidence towards why Gemini performs slightly worse - we observe that the specificity rate of Gemini is 30 points lower than Qwen, suggesting that it is much more likely to predict “yes”, i.e., it hallucinated the contact frame at a much higher rate than Qwen.

![](images/766d8503308837122fceb7cbdb2000eabf449c1008b4d8910dc076e5383f8e13.jpg)

Qwen2.5-72B  
![](images/b10675fff04f6ba38ad25d61e14aff52e327c3b2b0f25f08c26813d987ec1464.jpg)  
Figure 11: Uncertainty analysis comparing Gemini2.5-Pro and Qwen2.5-72B on text-option vs. visual-option questions. Perplexity score 1=absolute confidence, 4=absolute uncertainty. The boxplots of perplexity score reveal a stark contrast: Gemini is overconfident in its predictions regardless of the evaluation modality, while Qwen is more uncertain on both tasks, more uncertain on visual information than text. Gemini makes no such distinctions and confidently predicts the wrong answer.

## F.3 Modal Perplexity

Models on our benchmark often behave as if they are guessing when options are videos. However, we want to ask: are models aware that they do not understand the video options (high uncertainty), or are they confidently committing to essentially random choices? Specifically, we perform uncertainty analysis for the temporal frame retrieval task, where both visual and text options are available.

For each question, we treat the four answer options as a categorical distribution induced by the model’s logits and estimate model uncertainty over the options via perplexity [39, 40]. This lets us distinguish between (i) near-uniform, high-entropy predictions consistent with genuine uncertainty, and (ii) highly peaked, low-entropy predictions that nevertheless aggregate to near-random accuracy. By comparing these quantities across text-option vs. visual-option settings on the Gemini 2.5 Pro and Qwen2.5-72B responses in Figure 11, we find a stark contrast: we find that Gemini is overall more overconfident in its predictions, and does not calibrate the difference in evaluation modality (text and vision), i.e., there is a minimal gap in its uncertainty among the two modalities. Its predictions are overconfident with respect to both. Qwen, on the other hand, is more uncertain on both modalities (is not overconfident in either), and calibrated among modalities as well; it is much more uncertain on visual data than text. These are interesting results, as both models have low accuracies on the temporal frame retrieval task (Table 1), which allows us to frame their behavior in terms of confidence and calibration.

## G Dataset expansion roadmap

In the CMD annotation pipeline, the real time constraint is the selection of suitable video candidates - finding a usable video requires a human to judge whether the contact is unambiguous, whether the post-contact trajectory is visible and long enough to imply the outcome, and whether the scene contains causally unrelated footage. These issues were not encountered by the original study [28], as there were only six videos in their work.

But once these candidates are identified, the annotation cost itself from the human is not expensive - per source video the annotator marks four things: the sub-range containing one target action, the contact/release frame (the probe frame at test), the interval to exercise in the incomplete conditions, and an interval of unrelated same-scene footage. From these the tool deterministically synthesises all six stimuli, so effort scales with source videos rather than conditions. Marking a selected video takes 5–10 minutes.

Therefore, in the dataset expansion process, the crowdworkers would first need to filter out candidates based on the judgment described above, and only then engage in the annotation. Once the first step is achieved, the second step is quicker. We plan to engage crowdworkers in this process. We have also released the annotation tool as part of our code release.

The TFR and VFP tasks are relatively straightforward to expand. Both scripts automate the frame selection and generation process, while for VFP, potentially infinite videos could be generated as it is a 3D object simulator.

For all tasks, there is dataset flexibility - Something-Something V2, EPIC-KITCHENS, Ego4D, are some examples of other source datasets from which we can extract frames.

## H Prompt sensitivity

We run an ablation for Gemini and Kimi K2.6 on TFR using prompt variations. We modify the original prompt’s wording and structure, but keep the semantics unchanged. If prompt sensitivity were the key differentiator, we would observe a significant difference in the vision-text performance gap. However, consistent with our central claim, we observe a minimal difference in the vision-text performance gap, demonstrated in Table 5. It cannot, therefore, be attributed to variations in the prompt.

Table 5: Text and visual performance across model variants. ∆ denotes the difference between visual and text performance.
<table><tr><td>Variant</td><td>Text</td><td>Visual</td><td>∆</td></tr><tr><td>Gemini Original</td><td>0.739</td><td>0.307</td><td>-0.432</td></tr><tr><td>Gemini Modified</td><td>0.744</td><td>0.306</td><td>-0.438</td></tr><tr><td>Kimi Original</td><td>0.721</td><td>0.334</td><td>-0.386</td></tr><tr><td>Kimi Modified</td><td>0.714</td><td>0.256</td><td>-0.458</td></tr></table>

This is consistent with our central claim in the paper - visual querying surfaces gaps that text querying alone cannot elucidate in video understanding.