# Grounded Human-Attributed Description and Activity Recognition in Videos

Jay N. Paranjape<sup>1,2</sup>, Jurijs Nazarovs<sup>2</sup>, Rohan Sarkar<sup>2</sup>, Yue Guo<sup>2</sup>, Lipeng Ke<sup>2</sup>, Sankar Venkataraman<sup>2</sup>, Vishal M. Patel<sup>2</sup>, and Nataraj Jammalamadaka<sup>2</sup>

<sup>1</sup> Johns Hopkins University, USA 2 Amazon

Abstract. Understanding characteristics and activities of individuals in complex multi-person environments is crucial for real-world applications. However, existing datasets often simplify this problem to grounded group activity recognition, single-person activity recognition, or close-set activity recognition, thus limiting the generalizability of models trained on these datasets. In this work, we introduce the task of Grounded Human-Attributed Description and Activity Recognition (GHADAR), which involves describing the characteristics and activity descriptions of every person in a video, provided the location of the person in the video, which is more practical for real-world applications. To facilitate this, we introduce a new dataset derived from AVA-Actions by generating open-set captions for the person description and activity. In addition, we propose a novel method to efectively utilize the information contained in grounding during training by constraining the cross-attention masks during training in VLMs to improve performance for this task. Our experiments show that our method outperforms SOTA VLMs on this task. Finally, we demonstrate the limitations of existing evaluation metrics, which are overly reliant on human-annotations and exact texttext matching. As an added video-based evaluation, we propose a holistic VLM-based evaluation schema that compares concepts directly between the video and the generated predictions. Thus, in this work, we develop a complete framework for GHADAR, including a dataset, a novel method and an evaluation schema, thereby establishing a strong foundation for future research in this domain. Code and data will be made available.

Keywords: Video Understanding · Activity Recognition · Vision Language Models

## 1 Introduction

The rapid progress of vision–language models (VLMs) has expanded the scope of computer vision from recognition to rich, multimodal understanding. Modern applications increasingly demand systems that can not only detect people in complex scenes, realized using powerful detectors like YOLO ( [31]) and DINO ( [35]), but also describe their appearance, attribute fine-grained characteristics, and recognize ongoing activities in an open-set manner ( [6, 9, 11, 16, 29]). Such human-attributed understanding is critical for intelligent home assistants, autonomous surveillance and checkout systems [26, 27], and assistive technologies ( [29]).

We formalize this challenge as Grounded Human-Attributed Description and Activity Recognition (GHADAR): given the spatial location of people in a video (via bounding box), the goal is to generate two open-set captions — one describing each person’s attributes (e.g., an elderly man in white on the street, a child in blue uniform) and one describing their activity (e.g., attempting to cross a road, playing on the footpath). This formulation demands per-person reasoning in multi-actor scenes, integrating both local appearance cues and broader contextual signals.

Traditionally, research eforts in this direction have typically reduced the complexity of human-centric video understanding due to the scarcity of annotated data as shown in Figure 1. For example, some works reduce this task to understanding the activity of a group of people as a whole or generating one caption for the entire video ( [13]). Meanwhile, some other works reduce the scope of activity recognition to a closed set of limited action labels ( [4,16,20]) or focus exclusively on single-person scenarios ( [33]). In response, this paper introduces a new dataset called AVA-Captions, which extends AVA-Actions ( [9]) with open-set, per-person captions for both attributes and activities, and involves multi-person interactions. Starting from AVA-Actions’ per-person action labels, we employ a high-capacity VLM (Nova Pro v1 ( [1])) to generate natural-language descriptions, followed by identity-aware deduplication via person re-identification. This enables training and evaluation of models on the proposed task.

We further propose Constrained Attention Masking-based Pretraining (CAMP), a two-stage training strategy for VLMs that explicitly leverages grounding information. In Phase 1, we constrain the attention mask of the output tokens to the visual tokens within an expanded bounding box around the target person, enforcing localized reasoning. In Phase 2, we remove constraints and fine-tune at a reduced learning rate, allowing the model to integrate global context without eroding its person-specific focus.

Finally, we critique existing evaluation metrics in person and activity description. These metrics are largely text-text comparisons based on n-grams (BLEU, METEOR, ROUGE) ( [3, 18, 22]), or similarity between embeddings from pretrained models (SBERT, CLIP) ( [24, 25]). Thus, these metrics often penalize semantically correct predictions that diverge lexically from human-annotated ground truth ( [21]). To address this, we introduce a VLM-driven evaluation framework that directly compares conceptual alignment between the raw video content and the model prediction from the lens of completeness, uniqueness and correctness of the caption. In summary, our contributions are:

1. Task and Dataset: We define the task of Grounded Human-Attributed Description and Activity Recognition (GHADAR) and release AVA-Captions, the first large-scale dataset with open-set, per-person appearance and activity captions grounded in video.

![](images/fc5ad7bd974b3f3c056a6a8f79044f42d4b912686e499dc70417cfe9cf4564e1.jpg)  
Fig. 1: Our proposed task with respect to existing tasks. While existing tasks are more relaxed, we propose an open-set human description and activity recognition that is more generic.

2. Method: We propose CAMP, a two-stage attention-masking pretraining strategy that explicitly leverages grounding for improved per-person reasoning.

3. Evaluation: We design a VLM-based video–prediction alignment protocol to complement classical text-matching metrics.

4. Results: We demonstrate consistent gains over strong VLM baselines on AVA-Captions and the publicly available HC-STVG v2 ( [30]) datasets, across both standard and proposed metrics.

## 2 Related Work

Table 1: Existing datasets. Multi-Activity represents whether captions are available for multiple activities done by the person in the video. Multi-Person represents whether annotations are available for multiple people appearing in the video. Most of the video clips in these datasets are short (0-20s) except for AVA-Actions and MEVA, which have 15 min long and 5 min long annotated videos, respectively. \* - can be obtained with minimal processing
<table><tr><td>Dataset</td><td>Size (# clips)</td><td></td><td>Human-Centric? Grounded for every frame? Open-Set Captions Multi-Activity? Multi-Person?</td><td></td><td></td><td></td></tr><tr><td>HC-STVG ( [30])</td><td>15K</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>VidSTG ( [28, 36])</td><td>7.8K</td><td></td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>ActivityNet Entities ( [40])</td><td>15K</td><td>*</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>AVA-Actions ( [9])</td><td>450</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>AVA-Kinteics ( [16])</td><td>230k</td><td>√</td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>MSR-VTT ( [34])</td><td>10K</td><td>米</td><td></td><td>√</td><td></td><td></td></tr><tr><td>RefAVA ( [23])</td><td>36K</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>MEVA ([8])</td><td>1.7K</td><td>√</td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>AVA-Captions (Ours)</td><td>45K</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Existing Datasets: While we are among the first to propose grounded appearance and activity captions for every person in a video, this task has been explored in various relaxed settings in the literature. We tabulate some related datasets and their characteristics in Table 1.HC-STVG ( [30]) contains multi-person video clips but annotates only a single target individual per clip. It provides natural-language descriptions of the subject’s appearance and activity, along with frame-level bounding boxes, making it the closest existing benchmark to our setting. This dataset comes closest to our use case and hence, we also include results on this dataset in our experiments. VidSTG ( [28, 36]) is not a human-centric dataset and contains location captions for inanimate objects for most videos. In addition, it does not have person descriptions. ActivityNet Entities ( [40]) is not strictly human-centric but it is easy to filter out the other object annotations. However, it does not ofer grounding information for all frames and a majority of the videos contain only one person in the field of view. AVA-Actions ( [9]) has frame-wise annotations per person. However, it does not have person descriptions and only provides action labels for 80 comprehensive, predetermined classes. We use this dataset to generate natural language descriptions for the person and their activity using these action labels. AVA-Kinetics ( [16]) does not have open-set captions nor does it have bounding boxes for the persons in the video. RefAVA [23] extends AVA-Actions by adding manually-annotated open set person descriptions. However, it does not provide open-set captions for human actions, which are required for understanding person-object interactions. Finally, MEVA ( [8]) is a human-and-group-centric datasets that has activity labels for individuals and group activities. However, this dataset ofers a very limited range of activities and is more specific for the use-cases of surveillance. In summary, existing datasets either lack open-set captions, omit per-person grounding, or restrict to single-person or closed-set scenarios — motivating the need for AVA-Captions.

Human Activity Recognition Methods: Human activity recognition (HAR) is an essential component of fields like home assistants, embodied AI and assistive technology, and is gaining traction in Deep Learning (DL) literature ( [6,11,29]). Recent deep learning approaches treat HAR as a specialized form of video understanding, but often under constrained settings. ( [12, 13, 32, 33]). For instance, ( [33]) combines local and global video processing to capture activity cues, but assumes only one person is present. ( [12]) performs large-scale pretraining for caption generation from videos. But this method is not necessarily human-centric and is more suited toward scene description rather than human action description. ( [13]) focuses on recognizing activities of people as a group rather than focus on individual actions, whereas ( [4]) focuses on HAR using the human pose. Some papers also approach this problem by predefining action classes and trying to classify human actions in one of these classes ( [4,8,16,20]). As opposed to these methods, which employ a relaxed setting for HAR, we are one of the first methods to propose the HAR task in a more unconstrained way, allowing open-set captions, and human-attributed action recognition that allows multiple people in the field of view.

![](images/84b672fed65176f121f08727c3456454ef92c80dfc4051f5b28e3c62a766cc6d.jpg)  
Fig. 2: Examples from the AVA-Captions dataset. The captions in black are generated using expert-VLM while red action labels are provided in the original AVA-Actions dataset. More examples are provided in the supplementary.

## 3 AVA-Captions Dataset

Existing datasets focus on a single person or focus on a closed set of classes, as shown in Table 1. To address these limitations, we construct AVA-Captions, a large-scale benchmark for open-set, per-person appearance and activity description in video. We build AVA-Captions from AVA-Actions v2.2 ( [9]), which contains 15-minute annotated segments for 235 training and 64 validation videos. Each segment includes frame-level bounding boxes and action labels for every visible person, identified by a unique Person ID (PID), with actions drawn from 80 predefined classes. To create training samples, we segment each annotated video into 10-second clips with a maximum overlap of 5s, yielding approximately 36.8K training and 11.1K testing clips from the original training and validation splits. As AVA-Actions does not release test-set annotations, we do not use its oficial test split.

Once we have the extracted clips, we use a large VLM (called Expert-VLM) - Nova Pro ( [1]), to generate separate captions for person and activity descriptions for each person in the clip. For the person description, we provide the video along with an image crop extracted using the bounding box around the person to the VLM and prompt it to generate a caption describing the person including their apparel, age and location (for eg. a child in red hat standing near the stairs). For the activity description, we also provide the activity labels provided in the original dataset to the expert VLM and ask it to construct a complete caption for the activity of the person using the video and the action labels. The exact prompts used for this process are provided in the supplementary. This method can be adapted for other datasets as well to generate dense personspecific captions in the future.

However, there is one more concern regarding the AVA-Actions dataset. From our observations, we found that the same person is often assigned multiple PIDs. This occurs when the camera pans away from a person and comes back to the same person, who is then detected as a new entity. This causes the data to have multiple repeated entries for diferent PIDs since they represent the same person. To address this, we apply a Person-ReID-based filter using the torchreid library ( [37–39]). If the image crop corresponding to a given PID has a ReID similarity score greater than 0.75 with any previous PID, then we merge them and retain the previous PID. We use the Expert-VLM to merge the captions so that there is no repetition. Some examples are shown in Figure 2. We manually validated 1,000 randomly sampled datapoints to assess the quality of captions generated by the VLM. For each video, we examined whether:

1. all ground-truth action labels were incorporated into the caption,

2. individuals were described accurately,

3. the annotated actions were correctly reflected, and

4. the same person was consistently assigned a single subject ID.

Our evaluation revealed that in approximately 95% of cases, the VLM successfully utilized all action labels and produced accurate person and activity descriptions. However, in roughly 15% of cases, the captions contained repeated actions, primarily due to duplicated action labels across timestamps in the original dataset. In addition, for around 15% of the cases, the same subject was still given multiple PIDs. Adjusting the ReID similarity score threshold to 0.85 increased this number to 20%, whereas changing it to 0.65 led to false merging of distinct entities into a single subject ID for around 30% cases. Thus, we choose 0.75 as the threshold. While we note that VLM-based generation may introduce minor biases, our manual inspection suggests that such artifacts are limited and can be treated as noise without significantly impacting model development. Additional examples and dataset statistics are provided in the supplementary.

## 4 Constrained Attention Masking-based Pretraining (CAMP)

## 4.1 Problem Formulation

Given an input video and grounding information in the form of bounding boxes, the task of Grounded Human-Attributed Description and Activity Recognition expects a person description caption and an action caption for all people in the video. More formally, given video V, let there be S number of subjects in the video. Let the sampled frames from the video be denoted by $\{ F _ { 1 } , F _ { 2 } , . . . , F _ { D } \}$ ， where D denotes the total number of frames. The bounding boxes around a subject i are given by $B _ { i } = \{ b _ { 1 } , b _ { 2 } , . . . , b _ { D } \}$ , where each element can be a bounding box in the $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ format if the person is present in the frame or None otherwise. The bounding box format specifies the top-left and the bottomright points of the bounding box in the frame. Thus, the complete grounding information can be denoted by $\mathbb { B } = \{ B _ { 1 } , B _ { 2 } , . . . , B _ { S } \}$ . Given this information, the model is expected to produce two outputs $C ^ { P } = \{ c _ { 1 } ^ { P } , c _ { 2 } ^ { P } , . . . , c _ { S } ^ { P } \}$ and $C ^ { A } = \{ c _ { 1 } ^ { A } , c _ { 2 } ^ { A } , . . . , c _ { S } ^ { A } \}$ , where $C ^ { \bar { P } }$ represents the person descriptions and $\dot { C } ^ { A }$ denotes the activity captions of all the subjects in the video. Here, $c _ { i } ^ { P }$ and $c _ { i } ^ { A }$ denote the captions for person description and person activity for the subject.

In this study, we use a VLM for the model, which learns a function $f$ such that $A = f ( V , Q | \mathbb { B } )$ , where $Q$ denotes the question prompt provided to the VLM and

![](images/3782e68353b9b3bb3e2648c4e6aa2f6c26193cc451e905b9ec853444d38a121d.jpg)  
Fig. 3: The two phases of CAMP. In the first phase, the modified causal mask for the answer tokens makes sure they can only attend to the vision tokens corresponding to the relaxed bounding box around the person for which the caption is being generated. In the second phase, the model is trained with a lower learning rate with a default attention mask, which allows insertion of global context into the attention weights. Here, $T _ { V } { : }$ vision tokens, $T _ { Q } \colon$ question tokens, $T _ { A } \colon$ Answer tokens. The illustration shows only one frame but the masking is done using all frames of the video.

A represents the output answer from the VLM. To model the above formulation, the question prompt includes information about the Person ID (PID) of the person and the bounding boxes. The output answer combines the person and activity descriptions in the following form: $A = ^ { \mathrm { { * } } } \mathrm { P I D } _ { 1 } . \mathrm { ~ } c _ { 1 } ^ { P } : c _ { 1 } ^ { A } ; \mathrm { ~ } . . . \mathrm { { P I D } } _ { S } . \mathrm { ~ } c _ { S } ^ { P } : c _ { S } ^ { A } ; \mathrm { ~ } ^ { \mathrm { { * } } }$ The delimiters used in this format are useful for evaluation. Finally, an implicit constraint in this work is that the captions for person and activity description should be concise (a sentence instead of detailed descriptions), yet distinctive. This would help reduce further processing times for potential use cases. However, this is not a constraint that is inherent to the problem formulation and is enforced because of the ground truth annotations.

## 4.2 CAMP Phase 1

In phase 1 of CAMP, we train the VLM InternVL3-1B ( [7]) with a constraint on the attention mask during the decoding phase. In InternVL3, the video, question and answer are all converted into tokens using the VLM’s image and text encoders. The attention mask, in turn, defines which token can attend to which token during the computation of attention weights in the transformer blocks of the VLM. In phase 1, we attempt to restrict the model to only look at regions around a person when generating output for the person.

More formally, let the number of video tokens be denoted by $T _ { V } ,$ the number of question prompt tokens be denoted by $T _ { Q }$ , and the number of the ground truth answer prompt tokens be denoted by $T _ { A }$ . Then the attention mask for any layer of the VLM is a $N \times N$ matrix Att, where $N = T _ { V } + T _ { Q } + T _ { A }$ . In this

matrix, the element $A t t _ { i , j }$ is given by:

$$
A t t _ { i , j } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ t o k e n ~ } } i { \mathrm { ~ c a n ~ a t t e n d ~ t o ~ t o k e n ~ } } j } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{1}
$$

By default, VLMs follow a causal attention masking strategy, where the output (answer) tokens can attend to all tokens before the current token, to prevent information leakage. Note that we are only interested in the answer tokens since the loss is computed only for the model outputs for these tokens. More formally, by default:

$$
A t t _ { i , j } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } i \leq T _ { V } + T _ { Q } } \\ { 1 , } & { { \mathrm { i f ~ } } j \leq i } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{2}
$$

As mentioned previously, the annotated ground truth consists of captions from all people in the image. In the default setting, the output tokens for the parts corresponding to one person can also attend to the regions in the video not related to the person and the model is expected to learn the attention weights by itself to reduce the weightage of these regions. Instead, in phase 1 of CAMP, we explicitly constrain the model to generate outputs without attending to any regions except around the entity of interest during training. More formally, this translates to the following:

$$
A t t _ { i , j } = \left\{ { \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } i \leq T _ { V } + T _ { Q } } \\ { 1 , } & { { \mathrm { i f ~ } } j \leq i \ { \mathrm { A N D ~ } } j \in g _ { t s } ( i , \mathbb { B } ) } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{3}
$$

Here, $g _ { t s }$ represents the token selection function that outputs a list of tokens that the current token can attend to. For video tokens, this function masks out the tokens that are unrelated to the person for which the output is being generated at token i. Thus, $g _ { t s } ( i , \mathbb { B } ) = [ k _ { 1 } , k _ { 2 } , . . . , k _ { M } ]$ , such that (i) $M \leq N$ , (ii) $k _ { m } \leq N \forall k _ { m } \in g _ { t s } ( i , \mathbb { B } )$ , (iii) if $j > T _ { V } , j \in g _ { t s } ( i , \mathbb { B } )$ , (iv) $\mathrm { i f } \ j \le T _ { V } , j \in g _ { t s } ( i , \mathbb { B } )$ if and only if the token j corresponds to the region in the video contained in the bounding boxes $B _ { p }$ for person p for which the caption is being generated at token i. While training, we fix the input size of each video frame to $4 4 8 \times 4 4 8$ Hence $g _ { t s } ( i , \mathbb { B } )$ is a deterministic function. This is shown in Figure 3 where the attention mask values corresponding to the solid coloured lines are 1’s.

However, this is not enough to capture the activity of the person since the bounding box around a person may not include the local context around the person, which is required to gauge the semantic activity of the person. Hence, we use a relaxed bounding box, as shown by the dotted coloured lines in Figure 3. The relaxed bounding box for a given bounding box b is given by $\left[ m a x ( 0 , x _ { 1 } - w ) \right]$ , max $( 0 , y _ { 1 } - 0 . 2 h )$ , $m i n ( x _ { 2 } + w , 4 4 8 )$ , min $( y _ { 2 } + 0 . 2 h , 4 4 8 ) ]$ , where w and h represent the width and the height of the original bounding box $b ,$ and 448 denotes the size of the image. This function is motivated by the general observation in our dataset that the bounding boxes around people are generally higher than they are wider, and that significant context related to the activity is captured in the areas around the person.

## 4.3 CAMP Phase 2

During Phase 1 training, we constrain the attention masks for each individual, encouraging the model to generate appearance and activity captions by focusing primarily on the local context surrounding the person. While efective for learning localized descriptions, this setup has two key limitations: (i) activities that inherently depend on global context—such as interactions with other people or distant objects—cannot be captured accurately under strict local constraints; and (ii) constructing person-specific attention masks requires knowing the exact mapping between output tokens and individuals, which is feasible during training but impractical at inference. A potential workaround would involve dynamically monitoring each generated token and updating the attention mask whenever a new subject is introduced, but this approach significantly increases inference complexity and latency, rendering Phase 1 alone unsuitable for deployment.

To address these issues, Phase 2 training removes the constraints and finetunes the model with the default attention mask (Figure 3) at a reduced learning rate. This continuation serves two purposes: it enables the model to incorporate global contextual cues while learning to generate outputs under the standard attention mechanism. At the same time, a lower learning rate ensures that that the attention weights of the model are not changed drastically so that the model still focuses on the regions around the person of interest. Together, these phases yield a model that balances localized grounding with global contextual reasoning, while remaining practical for inference.

## 4.4 Motivation

We observed that when training a VLM on our dataset, the model frequently confused actions performed by diferent individuals (for example, action done by person B predicted for person A). To mitigate this, we introduce the CAMP framework, built on the hypothesis that for describing a person’s actions and appearance, the local context surrounding the individual is generally suficient, without requiring attention to the entire scene. By masking tokens outside the region of interest, the model is encouraged to learn mappings between localized visual features and their corresponding textual descriptions. While this hypothesis holds in many cases, modifying attention maps dynamically during inference is impractical. To address this limitation, we introduce Phase 2 training, where the constraints are removed and the model is fine-tuned with a reduced learning rate. Prior work has shown that lower learning rates yield more conservative weight updates and stabilize training ( [5, 15]). This second phase enables the model to incorporate global context while preserving its focus on the regions of interest, thereby eliminating the need for attention map modifications at inference time.

## 5 Experiments

Comparison with Similar Scale VLMs: We perform experiments on the HC-STVG and AVA-Captions datasets with baseline, open-source VLMs and our CAMP method. The experimental setup is detailed in the supplementary. We tabulate the results of CAMP with InternVL3-1B [7] and Qwen2.5VL-3B [2] backbone as compared to similar open source baselines in Tables 3 and 2. We see a clear improvement in all classical and embedding-based metrics for both datasets. Table 2 evaluates the model prediction for subject description while Table 3 evaluates the model prediction for activity description. As described before, the output is of the format: $A = ^ { \mathrm { * } } \mathrm { P I D } _ { 1 } . \ c _ { 1 } ^ { P } : c _ { 1 } ^ { A } ; \ . . . \mathrm { \bar { P I D } } _ { S } . \ c _ { S } ^ { P } : c _ { S } ^ { A } ; \ ^ { \mathfrak { m } }$ . We separate this output into subject description and activity description using the colon delimiter. For diferent subjects, we use the PID to map the predictions to the ground truth. The computed metrics are averaged across all videos and subjects. As compared to the LLAVA-One Vision (LLAVA-OV) baseline ( [17, 19]), which ofers a 0.5 B model, we see a significant improvement with InternVL3 1B model. While some of this can be attributed to the more number of parameters (since LLAVA-OV does not ofer 1B model), the multimodal encoder structure in InternVL3 ( [7]) style of models also plays a significant part. Unlike LLAVA, both InternVL3 and Qwen support frame-level injection, enabling bounding boxes to be passed separately for each frame. This design yields consistent improvements, which are further amplified by our CAMP framework through constrained attention in Phase 1 and unconstrained fine-tuning in Phase 2. Interestingly, we observe greater relative improvements for the smaller InternVL3-1B backbone compared to the larger Qwen2.5VL-3B. This is expected: larger models can already capture complex functions, making constrained training less impactful, whereas smaller models benefit more from CAMP’s inductive bias. Due to compute limitations, we were unable to compare with larger-scale models. However, we emphasize that the proposed task is most relevant for home assistants and surveillance systems, where inference typically occurs on edge devices. For this reason, our study focuses on small- to medium-scale models, which are more practical for such applications.

Table 2: Comparison of CAMP against existing methods for Subject Description
<table><tr><td rowspan="2">Method</td><td colspan="5">AVA-Captions</td><td colspan="5">HC-STVG</td></tr><tr><td>BLEU4 (↑)</td><td>METEOR (↑)</td><td>ROUGE (↑)</td><td>CLIP Sim (↑)</td><td>SBert Sim (↑)</td><td>BLEU4 (↑)</td><td>METEOR (↑)</td><td>ROUGE (↑)</td><td>CLIP Sim (↑)</td><td>Sbert Sim (↑)</td></tr><tr><td>LLAVA-OV 0.5B</td><td>28.29</td><td>54.22</td><td>59.47</td><td>0.75</td><td>0.54</td><td>18.70</td><td>53.94</td><td>57.47</td><td>0.86</td><td>0.48</td></tr><tr><td>InternVL3-1B</td><td>31.07</td><td>58.17</td><td>63.09</td><td>0.77</td><td>0.58</td><td>25.87</td><td>55.08</td><td>65.54</td><td>0.87</td><td>0.60</td></tr><tr><td>Qwen2.5VL-3B</td><td>36.46</td><td>63.17</td><td>68.06</td><td>0.82</td><td>0.65</td><td>25.66</td><td>55.07</td><td>65.81</td><td>0.88</td><td>0.62</td></tr><tr><td>InternVL3 + CAMP (Ours)</td><td>32.02</td><td>59.90</td><td>64.07</td><td>0.79</td><td>0.60</td><td>27.11</td><td>57.90</td><td>66.10</td><td>0.88</td><td>0.61</td></tr><tr><td>Qwen2.5VL-3B + CAMP (Ours)</td><td>36.48</td><td>63.62</td><td>68.43</td><td>0.83</td><td>0.67</td><td>27.33</td><td>57.84</td><td>67.82</td><td>0.88</td><td>0.64</td></tr></table>

Ablation on Relaxation of Bounding Box: During the attention masking in Phase 1 of CAMP, we use a relaxed bounding box around the person to capture local context for predicting activity. We expand it by 100% of the width on each side and by 20% of the height on each side based on the observation that humans are generally taller than wider. Hence, the bounding box around them captures lesser context along the width than the height. Thus, for the relaxed bounding box, we expand it more along the width. For added validation, we vary the proportions of these relaxations in Table 4. We observe that increasing the context in width is more efective than height as expected.

Table 3: Comparison of CAMP against existing methods for Activity Description
<table><tr><td rowspan="2">Method</td><td colspan="5">AVA-Captions</td><td colspan="5">HC-STVG</td></tr><tr><td>BLEU4 (↑) METEOR (↑) ROUGE (↑) CLIP Sim (↑)</td><td></td><td></td><td></td><td>SBert Sim (↑) </td><td>BLEU4 (↑)</td><td>METEOR (↑)</td><td>ROUGE (↑)</td><td>CLIP Sim (↑) Sbert Sim (↑)</td><td></td></tr><tr><td>LLAVA-OV 0.5B</td><td>13.39</td><td>37.71</td><td>39.69</td><td>0.85</td><td>0.53</td><td>8.34</td><td>32.43</td><td>41.78</td><td>0.82</td><td>0.40</td></tr><tr><td>InternVL3-1B</td><td>19.34</td><td>44.91</td><td>46.28</td><td>0.89</td><td>0.63</td><td>8.60</td><td>32.66</td><td>38.03</td><td>0.85</td><td>0.42</td></tr><tr><td>Qwen2.5VL-3B</td><td>19.30</td><td>43.02</td><td>47.35</td><td>0.90</td><td>0.63</td><td>8.97</td><td>31.52</td><td>40.01</td><td>0.86</td><td>0.47</td></tr><tr><td>InternVL3 + CAMP (Ours)</td><td>20.11</td><td>46.15</td><td>46.82</td><td>0.90</td><td>0.65</td><td>11.17</td><td>35.67</td><td>42.23</td><td>0.87</td><td>0.46</td></tr><tr><td>Qwen2.5VL-3B + CAMP (Ours)</td><td>20.14</td><td>43.40</td><td>47.90</td><td>0.91</td><td>0.65</td><td>10.64</td><td>33.08</td><td>40.65</td><td>0.86</td><td>0.47</td></tr></table>

Ablation on the phases of CAMP is provided in the supplementary due to space constraints. We find that employing only Phase 1 causes 10% drop due to lack of global context, while directly going to Phase 2 without the first phase causes a drop of 4%. This shows the necessity of both phases.

Table 4: Efect of relaxations of bounding box during CAMP on HC-STVG dataset. The columns represent the one-sided changes to width (\Delta w) and height (\Delta h) in terms of percentage of the total width and height. For example, (20,20) signifies 20% relaxation in width and height on each side, thus making the relaxed bounding box as $[ m a x ( 0 , x _ { 1 } -$ 0.2w), max(0, $y _ { 1 } - 0 . 2 h )$ , min $( x _ { 2 } + 0 . 2 w , 4 4 8 )$ $m i n ( y _ { 2 } + 0 . 2 h , 4 4 8 ) ]$
<table><tr><td></td><td colspan="7">(∆w, ∆h)</td></tr><tr><td></td><td>(0,0)</td><td>(20,20)</td><td>(50,50)</td><td>(100,100)</td><td>(100,0)</td><td>(100,20) (used)</td><td>(100,50)</td><td>(20,50)</td></tr><tr><td>Subject Sbert</td><td>0.54</td><td>0.54</td><td>0.57</td><td>0.58</td><td>0.60</td><td>0.61</td><td>0.60</td><td>0.55</td></tr><tr><td>Subject METEOR</td><td>52.21</td><td>52.54</td><td>54.96</td><td>57.09</td><td>56.98</td><td>57.90</td><td>56.59</td><td>53.64</td></tr><tr><td>Subject ROUGE</td><td>61.73</td><td>61.67</td><td>63.88</td><td>65.87</td><td>66.38</td><td>66.10</td><td>65.92</td><td>62.30</td></tr><tr><td>Activity Sbert</td><td>0.44</td><td>0.44</td><td>0.45</td><td>0.46</td><td>0.46</td><td>0.46</td><td>0.46</td><td>0.44</td></tr><tr><td>Activity METEOR</td><td>33.32</td><td>33.04</td><td>34.00</td><td>34.40</td><td>35.00</td><td>35.67</td><td>34.87</td><td>33.55</td></tr><tr><td>Activity ROUGE</td><td>40.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.07</td></tr><tr><td></td><td></td><td>40.44</td><td>41.40</td><td>41.84</td><td>42.06</td><td>42.23</td><td>41.87</td><td></td></tr></table>

## 6 VLM-based Metrics

Metric Definition: Text-text matching or embedding similarity metrics are inherently tied to the specific ground truth (GT) annotations ( [21]). As a result, they can undervalue predictions that are semantically correct but difer lexically from the GT, or that capture details omitted in the annotation. For example, consider Figure 5 which contains low-scoring predictions from our model. However, it can be seen from the video frame that the events described by the prediction are actually happening and noteworthy. We provide more such examples in the supplementary. This highlights a limitation: existing metrics cannot directly verify whether a predicted caption is factually grounded in the visual content. To address this, we propose using a large-scale model to act as a judge, given the actual video and the predictions. Thus, these metrics are computed without knowing the GT captions and designed to complement, rather than replace, the classical measures. More specifically, given a video, an image crop of the target subject, and the predicted caption, we query Nova-Pro v1 ( [1]) as well as Qwen2.5 VL 32B ( [2]) to assign a score from 1–5 (higher is better) for each of the following:

![](images/bff4831c4c36f2834d85dd38632ec620b4b2373a374d120627347decfcccca6c.jpg)  
Fig. 4: Qualitative examples from HC-STVG (left two) and AVA-Captions (right). HC-STVG has captions only for one person, whereas AVA-Captions has captions for both people in the video.

1. Subject / Activity Correctness (CRT): How much of the predicted caption is actually present in the video for the given subject. CRT measures factual alignment of the prediction with the video content.

2. Subject / Activity Completeness (CMP): How much of the subject description / activity is described by the caption. CMP measures whether all relevant actions are captured by the prediction.

3. Subject Uniqueness (UNQ): How discriminative is the predicted subject description in identifying the subject. UNQ measures ambiguity in the predicted caption.

CRT and CMP are analogous to precision and recall, respectively whereas UNQ is motivated for practical use cases and its definition is dependent on the video. For example, if a predicted caption reads: “The man : opens the door", it can be considered highly unique for describing the subject if there is only one man in the video, but not at all unique when there are multiple men in the video. We believe that UNQ is a better metric than CMP for subject description for practical purposes. In addition, CMP is ill-defined especially for the subject description since there can be various ways of describing the same person. While this is also true for the activities, CMP gives an approximate measure of whether all important activities are captured by the model and so we retain it as a metric. Some examples are shown in Figure 5. Based on these definitions, we compute (i) the average CRT, (ii) average CMP, (iii) average UNQ and (iv) accuracy based on CRT \ge 4 \text { AND } UNQ \ge 3 for sub ject and CRT \ge 4 \text { AND } CMP \ge 3 for activity (Acc), as VLM-based metrics. The full prompt used for querying the VLM for computing these metrics is provided in the supplementary.

![](images/5f507c2afe36f36df25606c5635ce4b3fbd4572c093384d5354146c5373edc14.jpg)  
Fig. 5: Examples of predictions wrongly given low metrics. In the first row, the predicted events occur in the video but are not captured by the GT annotation, leading to low SBert scores. Similarly, in the second row, the GT and predicted ways to describe the woman are valid, yet subject SBert score is low due to lexical mismatch. This necessitates the need for VLM-based metrics. More examples in the supplementary

Comparison with Similar Scale VLMs: We tabulate the subject description metrics in Table 5 using Nova Pro v1 and Qwen2.5 VL 32B as expert models respectively. Table 6 show activity caption metrics using Nova Pro v1 and Qwen2.5 VL 32B as expert models respectively. We see that our method consistently outperforms baselines and is able to produce better and more comprehensive outputs as shown in Figure 4. Even though the average CRT, UNQ, or CMP rise slightly, it generates a significant diference in accuracy after thresholding, which is due to the discrete nature of the VLM-based metrics. As seen in the examples, constrained attention using CAMP can guide the VLM towards localizing actions for a given individual instead of getting confused with the global context alone. For example, without CAMP, InternVL3 confuses between the two PIDs, whereas LLAVA-OV confuses the person description.

Limitations: Recent works have shown the efectiveness of using VLMs as judges [10, 14] where the proximity of the VLM results to the human results is heavily dependent on the prompt. However, VLMs can introduce inherent bias in their results as described in detail in the supplementary. Hence, we recommend using VLM-based metrics for relative comparison between models rather than the absolute performance of a single model.

Table 5: Comparison of CAMP against existing methods for Subject Description with VLM-based metrics using Nova Pro v1 and Qwen2.5 VL 32B as expert VLMs
<table><tr><td>VLM Evaluator</td><td colspan="6">Nova Pro v1</td><td colspan="6">Qwen2.5 VL 32B</td></tr><tr><td>Dataset</td><td colspan="3">AVA-Captions</td><td colspan="3">HC-STVG</td><td colspan="3">AVA-Captions</td><td colspan="3">HC-STVG</td></tr><tr><td>Method</td><td>Avg CRT (↑)</td><td>Avg UNQ(↑)</td><td>Acc (↑)</td><td>Avg CRT (↑)</td><td>Avg UNQ (↑)</td><td>Acc (↑)</td><td>Avg</td><td>Avg CRT (↑) UNQ(↑)</td><td>Acc (↑)</td><td>Avg CRT (↑)</td><td>Avg ) UNQ(↑)</td><td>Acc (↑)</td></tr><tr><td>LLAVA-OV 0.5B</td><td>2.84</td><td>2.06</td><td>0.29</td><td>3.99</td><td>1.94</td><td>0.34</td><td>2.89</td><td>2.29</td><td>0.38</td><td>2.86</td><td>2.46</td><td>0.30</td></tr><tr><td>InternVL3-1B</td><td>3.03</td><td>2.07</td><td>0.32</td><td>4.03</td><td>2.09</td><td>0.40</td><td>3.05</td><td>2.34</td><td>0.41</td><td>3.03</td><td>2.47</td><td>0.37</td></tr><tr><td>Qwen2.5VL-3B</td><td>3.46</td><td>2.35</td><td>0.43</td><td>4.36</td><td>2.04</td><td>0.39</td><td>3.44</td><td>2.58</td><td>0.66</td><td>3.25</td><td>2.53</td><td>0.33</td></tr><tr><td>InternVL3 + CAMP (Ours)</td><td>3.09</td><td>2.11</td><td>0.34</td><td>4.29</td><td>2.25</td><td>0.48</td><td>3.13</td><td>2.38</td><td>0.43</td><td>3.28</td><td>2.69</td><td>0.45</td></tr><tr><td>Qwen2.5VL-3B + CAMP (Ours)</td><td>3.51</td><td>2.36</td><td>0.43</td><td>4.27</td><td>2.11</td><td>0.41</td><td>3.47</td><td>2.63</td><td>0.67</td><td>3.29</td><td>2.56</td><td>0.38</td></tr></table>

Table 6: Comparison of CAMP against existing methods for Activity Description with VLM-based metrics using Nova Pro v1 and Qwen2.5 VL 32B as expert VLMs
<table><tr><td>VLM Evaluator</td><td colspan="6">Nova Pro v1</td><td colspan="6">Qwen2.5 VL 32B</td></tr><tr><td>Dataset</td><td colspan="3">AVA-Captions</td><td colspan="3">HC-STVG</td><td colspan="3">AVA-Captions</td><td colspan="3">HC-STVG</td></tr><tr><td>Method</td><td>Avg CRT (↑)</td><td>Avg CMP(↑)</td><td>Acc (↑)</td><td>Avg CRT (↑)</td><td>Avg CMP (↑)</td><td>Acc (↑)</td><td>Avg CRT (↑)</td><td>Avg CMP(↑)</td><td>Acc (↑)</td><td>Avg CRT (↑)</td><td>Avg CMP(↑)</td><td>Acc (↑)</td></tr><tr><td>LLAVA-OV 0.5B</td><td>3.24</td><td>2.39</td><td>0.54</td><td>2.21</td><td>1.17</td><td>0.08</td><td>3.59</td><td>3.17</td><td>0.75</td><td>3.07</td><td>2.43</td><td>0.53</td></tr><tr><td>InternVL3-1B</td><td>3.54</td><td>2.65</td><td>0.64</td><td>3.17</td><td>1.94</td><td>0.41</td><td>3.80</td><td>3.40</td><td>0.83</td><td>3.93</td><td>3.28</td><td>0.88</td></tr><tr><td>Qwen2.5VL-3B</td><td>3.61</td><td>2.83</td><td>0.68</td><td>3.51</td><td>2.07</td><td>0.48</td><td>3.86</td><td>3.51</td><td>0.85</td><td>3.96</td><td>3.18</td><td>0.87</td></tr><tr><td>InternVL3 + CAMP (Ours)</td><td>3.63</td><td>2.75</td><td>0.68</td><td>3.45</td><td>2.16</td><td>0.50</td><td>3.92</td><td>3.51</td><td>0.86</td><td>4.04</td><td>3.31</td><td>0.90</td></tr><tr><td>Qwen2.5VL-3B + CAMP (Ours)</td><td>3.63</td><td>2.86</td><td>0.71</td><td>3.43</td><td>2.05</td><td>0.46</td><td>3.86</td><td>3.52</td><td>0.85</td><td>4.02</td><td>3.20</td><td>0.89</td></tr></table>

## 7 Conclusion

In this paper, we introduce the task of Grounded Human-Attributed Description and Activity Recognition and propose a dataset called Ava-Captions for the same by modifying the AVA-Actions dataset. In order to efectively use the grounding information in this task, we propose the CAMP method that constrains the attention mask during training, thereby directing the model to focus more on the local context around a person to better caption their description and activity. Finally, we propose VLM-based metrics that directly compares predicted captions with the content of the video instead of incomplete human annotations and serves as a complement to classical metrics. A limitation in the VLM-based metric is that it is highly dependent on the ability of the VQA to understand the video and the instruction. With this work, we want to encourage future research in the human-attributed video understanding tasks that can be useful for various applications.

## References

1. AGI, A.: The amazon nova family of models: Technical report and model card (2025), https://arxiv.org/abs/2506.12103

2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

3. Banerjee, S., Lavie, A.: Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In: Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization. pp. 65–72 (2005)

4. Bian, X., Chang, D., Yang, Y., He, Z., Liang, K., Ma, Z.: Class-aware contrastive learning for fine-grained skeleton-based action recognition. In: Cho, M., Laptev, I., Tran, D., Yao, A., Zha, H. (eds.) Computer Vision – ACCV 2024. pp. 225–241. Springer Nature Singapore, Singapore (2025)

5. Bjorck, J., Weinberger, K.Q., Gomes, C.: Understanding decoupled and early weight decay. Proceedings of the AAAI Conference on Artificial Intelligence 35(8), 6777–6785 (May 2021). https://doi.org/10.1609/aaai.v35i8.16837, https://ojs.aaai.org/index.php/AAAI/article/view/16837

6. Chen, Y., Cui, W., Huang, Y., Liu, C., Zhu, T.: Open-set sensor human activity recognition based on reciprocal time series. In: Shi, Z., Torresen, J., Yang, S. (eds.) Intelligent Information Processing XII. pp. 101–115. Springer Nature Switzerland, Cham (2024)

7. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24185–24198 (2024)

8. Corona, K., Osterdahl, K., Collins, R., Hoogs, A.: Meva: A large-scale multiview, multimodal video dataset for activity detection. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1060–1068 (January 2021)

9. Gu, C., Sun, C., Ross, D.A., Vondrick, C., Pantofaru, C., Li, Y., Vijayanarasimhan, S., Toderici, G., Ricco, S., Sukthankar, R., Schmid, C., Malik, J.: Ava: A video dataset of spatio-temporally localized atomic visual actions. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

10. Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H., Wang, S., Zhang, K., Wang, Y., Gao, W., Ni, L., Guo, J.: A survey on llm-as-a-judge (2025), https://arxiv.org/abs/2411.15594

11. Hossain, M.M., Han, T.A., Ara, S.S., Shamszaman, Z.U.: Benchmarking classical, deep, and generative models for human activity recognition. arXiv preprint arXiv:2501.08471 (2025), https://doi.org/10.48550/arXiv.2501.08471

12. Kazakos, E., Schmid, C., Sivic, J.: Large-scale pre-training for grounded video caption generation. arXiv preprint arXiv:2503.10781 (2025)

13. Kim, D., Song, Y., Cho, M., Kwak, S.: Towards more practical group activity detection: A new benchmark and model. arXiv preprint arXiv:2312.02878 (2023)

14. Lee, S., Kim, S., Park, S.H., Kim, G., Seo, M.: Prometheus-vision: Vision-language model as a judge for fine-grained evaluation (2024), https://arxiv.org/abs/ 2401.06591

15. Lewkowycz, A.: How to decay your learning rate (2021), https://arxiv.org/abs/ 2103.12682

16. Li, A., Thotakuri, M., Ross, D.A., Carreira, J., Vostrikov, A., Zisserman, A.: The ava-kinetics localized human actions video dataset (2020), https://arxiv.org/ abs/2005.00214

17. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)

18. Lin, C.Y.: Rouge: A package for automatic evaluation of summaries. In: Text summarization branches out: Proceedings of the ACL-04 workshop. pp. 74–81 (2004)

19. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning (2023)

20. Monfort, M., Pan, B., Ramakrishnan, K., Andonian, A., McNamara, B.A., Lascelles, A., Fan, Q., Gutfreund, D., Feris, R.S., Oliva, A.: Multi-moments in time: Learning and interpreting models for multi-action video understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(12), 9434–9445 (2022). https://doi.org/10.1109/TPAMI.2021.3126682

21. Ohi, M., Kaneko, M., Okazaki, N., Inoue, N.: Multi-modal, multi-task, multicriteria automatic evaluation with vision language models (2025), https://arxiv. org/abs/2412.14613

22. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic evaluation of machine translation. In: Proceedings of the 40th annual meeting of the Association for Computational Linguistics. pp. 311–318 (2002)

23. Peng, K., Fu, J., Yang, K., Wen, D., Chen, Y., Liu, R., Zheng, J., Zhang, J., Sarfraz, M.S., Stiefelhagen, R., Roitberg, A.: Referring atomic video action recognition. In: European Conference on Computer Vision (ECCV) (2024)

24. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, S., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning (ICML) (2021)

25. Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bertnetworks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. pp. 3982–3992 (2019)

26. Sarkar, R., Kak, A.: Scalable event-driven software architecture for the automation of people-centric systems (Jun 06 2024), uS Patent 12,008,873

27. Sarkar, R., Kak, A.C.: Checksoft : A scalable event-driven software architecture for keeping track of people and things in people-centric spaces (2021), https: //arxiv.org/abs/2102.10513

28. Shang, X., Di, D., Xiao, J., Cao, Y., Yang, X., Chua, T.S.: Annotating objects and relations in user-generated videos. In: Proceedings of the 2019 on International Conference on Multimedia Retrieval. pp. 279–287. ACM (2019)

29. Shin, J., Hassan, N., Miah, A.S.M., Nishimura, S.: A comprehensive methodological survey of human activity recognition across diverse data modalities. Sensors 25(13), 4028 (2024), https://www.mdpi.com/1424-8220/25/13/4028

30. Tang, Z., Liao, Y., Liu, S., Li, G., Jin, X., Jiang, H., Yu, Q., Xu, D.: Human-centric spatio-temporal video grounding with visual transformers. IEEE Transactions on Circuits and Systems for Video Technology 32(12), 8238–8249 (2022). https:// doi.org/10.1109/TCSVT.2021.3085907

31. Tian, Y., Ye, Q., Doermann, D.: Yolov12: Attention-centric real-time object detectors. arXiv preprint arXiv:2502.12524 (2025)

32. Wang, H., Xu, Z., Cheng, Y., Diao, S., Zhou, Y., Cao, Y., Wang, Q., Ge, W., Huang, L.: Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290 (2024)

33. Wang, M., Huang, Z., Kong, X., Shen, G., Dai, G., Wang, J., Liu, Y.: Action detail matters: Refining video recognition with local action queries. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19132– 19142 (2025). https://doi.org/10.1109/CVPR52734.2025.01782

34. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

35. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection (2022)

36. Zhang, Z., Zhao, Z., Zhao, Y., Wang, Q., Liu, H., Gao, L.: Where does it exist: Spatio-temporal video grounding for multi-form sentences. In: CVPR (2020)

37. Zhou, K., Xiang, T.: Torchreid: A library for deep learning person re-identification in pytorch. arXiv preprint arXiv:1910.10093 (2019)

38. Zhou, K., Yang, Y., Cavallaro, A., Xiang, T.: Omni-scale feature learning for person re-identification. In: ICCV (2019)

39. Zhou, K., Yang, Y., Cavallaro, A., Xiang, T.: Learning generalisable omni-scale representations for person re-identification. TPAMI (2021)

40. Zhou, L., Kalantidis, Y., Chen, X., Corso, J.J., Rohrbach, M.: Grounded video description. In: CVPR (2019)

# Supplementary - Grounded Human-Attributed Description and Activity Recognition in Videos

Anonymous ECCV 2026 Submission

Paper ID #06775

## 1 Prompt for generating AVA-Captions

We use Nova Pro v1 as an expert model to generate full captions about person and activity descriptions, given the video, cropped image of the person and the activity labels provided in the AVA-Actions dataset. For multiple people in the video, the expert model is run for each person and the resulting captions are merged. The exact prompt used for this purpose is as follows:

<VIDEO> <PERSON-CROP> This is a 10 seconds video and an image. Consider the person in the image. The action labels associated with the person in the video are given by “<ACTION1>,<ACTION2>,...". These actions describe only one person. Describe the selected person in regards to clothing, location or other surprising appearance, for example "Sitting blindfolded woman in a black suit". Use all the action labels to create an activity caption only for the selected person in the video while maintaining the temporal action order. The output should be in the format: <person description>: <activity caption>. For example, "Blindfolded woman in black suit: sits on a chair, gets up, opens the door, hugs the man at the door and walks out".Structure the action caption in a conceptual manner, with slightly more focus on the meaning of actions rather than simple words.

## 2 Experimental Setup

We perform experiments on the HC-STVG and AVA-Captions datasets with baseline, open-source VLMs and our CAMP method. We use 3 A100-SXM4s for our experiments with a batch size of 3. We use an image size of 448X448 and sample 10 equidistant frames from the video. Since most videos are 10 seconds long, this essentially corresponds to an efective sampling rate of 1 FPS. For evaluation, we calculate the subject and activity metrics separately. Subject metrics evaluate the predicted human appearance captions and activity metrics evaluate the predicted activity captions against the ground truth. We use the standard metrics for text-text comparison, namely BLEU4 ( [3]), ROUGE ( [2]) and METEOR ( [1]). In addition, we also calculate the embedding similarity metrics namely CLIP Similarity Score (CSS) and SBert Similarity Score (SbSS). For these, we compute the CLIP ( [4]) embeddings and SentenceBert ( [5]) embeddings, respectively, for the predicted caption and the ground truth caption, and compute the cosine similarity between the two.

![](images/b1c51f390d658ad8e3f3cbf38614e5136466fd9d01d5934acb1e9bc03b64e4b4.jpg)  
Fig. 1: Examples from the AVA-Captions dataset. The captions in black are generated using expert-VLM while red action labels are provided in the original AVA-Actions dataset.

## 3 More examples from AVA-Captions Dataset

Figure 1 shows examples from the AVA-Captions dataset. The VLM is able to accurately describe the person description even though it is not provided in the original ground truth. At the same time, the activity caption follows the action labels provided in the ground truth but is an open-set caption. Data statistics are provided in Table 1.

Table 1: Statistics for the number of people per video clip in the AVA-Captions dataset
<table><tr><td>Phase</td><td></td><td></td><td>Total video clips Minimum # People Maximum # People Mean # People Median # People</td><td></td><td></td></tr><tr><td>Training</td><td>36815</td><td>1</td><td>10</td><td>1.96</td><td>2</td></tr><tr><td>Testing</td><td>11095</td><td>1</td><td>11</td><td>1.98</td><td>2</td></tr></table>

## 4 Ablation on Phases of CAMP

We perform an ablation with just Phase 1 of CAMP, where we apply constrained attention masking during training as well as inference on HC-STVG and AVA-Captions dataset. The results are shown in Table 2. During inference, we keep a conditional statement at every step of the generation that checks for special tokens for change of subject and modifies the attention mask accordingly. However, we find significantly lower scores on both datasets. This is not surprising since false outputs can lead to false attention masks for all the tokens after the false output.

Table 2: Ablation on the phases of CAMP. The first row denotes the baseline without CAMP, next row denotes only Phase 1 without Phase 2 and last row denotes the best method, which includes both Phase 1 and Phase 2,
<table><tr><td></td><td></td><td colspan="5">HC-STVG Subject</td><td colspan="5">HC-STVG Activity</td></tr><tr><td></td><td></td><td>Phase 1 Phase 2 BLEU4 (↑)</td><td>METEOR (↑)</td><td>ROUGE (↑)</td><td>CLIP Sim (↑)</td><td>SBert Sim (↑)</td><td>BLEU4 (↑)</td><td>METEOR (↑)</td><td>ROUGE (↑)</td><td>CLIP Sim (↑)</td><td>SBert Sim (↑)</td></tr><tr><td>√</td><td></td><td>20.09</td><td>45.05</td><td>52.65</td><td>0.58</td><td>0.36</td><td>5.15</td><td>23.88</td><td>24.55</td><td>0.57</td><td>0.26</td></tr><tr><td></td><td>√</td><td>25.87</td><td>55.08</td><td>65.54</td><td>0.87</td><td>0.60</td><td>8.60</td><td>32.66</td><td>38.03</td><td>0.85</td><td>0.42</td></tr><tr><td>√</td><td>√</td><td>27.11</td><td>57.90</td><td>66.10</td><td>0.88</td><td>0.61</td><td>11.17</td><td>35.67</td><td>42.23</td><td>0.87</td><td>0.46</td></tr><tr><td></td><td></td><td colspan="5"></td><td colspan="5"></td></tr><tr><td></td><td></td><td></td><td></td><td>AVA-Captions Subject</td><td></td><td></td><td></td><td></td><td>AVA-Captions Activity</td><td></td><td></td></tr><tr><td colspan="10">Phase 1 Phase 2 BLEU4 (↑) METEOR (↑) ROUGE (↑) CLIP Sim (↑) ) SBert Sim (↑)</td></tr><tr><td>√</td><td></td><td>10.27</td><td>20.81</td><td>29.74</td><td>0.59</td><td>0.49</td><td>BLEU4 (↑) 6.93</td><td>20.81</td><td>18.49</td><td>METEOR (↑) ROUGE (↑) CLIP Sim (↑) SBert Sim (↑) 0.64</td><td>0.46</td></tr><tr><td></td><td>√</td><td>31.07</td><td>58.17</td><td>63.09</td><td>0.77</td><td>0.58</td><td>19.34</td><td>44.91</td><td>46.28</td><td>0.89</td><td>0.63</td></tr><tr><td>√</td><td>√</td><td>32.02</td><td>59.90</td><td>64.07</td><td>0.79</td><td>0.60</td><td>20.11</td><td>46.15</td><td>46.82</td><td>0.90</td><td>0.65</td></tr></table>

## 5 Motivating the need for VLM-metrics

We present more examples of cases where the ground truth for a given subject in a video lexically difers from the equally correct predictions in Figures 2 and 3. A video contains a lot of information, not all of which is captured by a human annotator in most cases. Thus, classical metrics which directly compare with these are significantly pessimistic. This motivates the need for VLM-based metrics, where the video is directly analyzed to assess the predicted caption.

![](images/88dbe6a88b0bd33624d1131c7fe35d4107ed31710c3fb7173568d47ec234e8d4.jpg)  
Fig. 2: Examples of predictions which are lexically dissimilar from the ground truth but are equally valid for the given video

![](images/49757f9938b905760ee37fab4f3183fc0c550e130badc270db816be1950f083c.jpg)  
Fig. 3: A limiting case of VLM-based evaluation metrics. CMP and UNQ are pessimistically given a score of 1 even though the predicted caption describes the person uniquely and describes some of the activities of the subject.

## 6 Limitations of VLM-based Evaluation

While foundation models today perform extremely well for visual understanding tasks, using them for evaluation can still induce biases in the result. Consider the example shown in Figure 3. Here, the caption describes the subject as “the standing man". Since no other men are standing in the video, it should get a high UNQ value, which is not the case. Similarly, some of the activities of the person are accurately described, yet the VLM gives the lowest CMP scores to the prediction. In our observation, especially for CMP and UNQ, the VLM is biased towards more pessimistic values for these metrics.

Thus, while VLMs ofer crucial insights complementing the classical metrics, it is best to use the VLM-metrics for relative comparison between models rather than absolute performance of a model.

## 7 Prompt for VLM-based Metrics

We used the Nova Pro v1 model to generate the CRT, CMP, and UNQ metrics. Since UNQ is not suitable for activity metrics, we run the VLM separately for the subject and the activity. Since HC-STVG dataset has annotations for only a certain portion of the video, we also provide the starting and ending timestamp in the prompt. The exact prompts are as follows:

Subject: Consider only t=<start time> to t=<end time> seconds in the video. Consider only the appearance related details of the Person ID 1: <subject description from ground truth>. Next, consider this candidate caption which we want to evaluate against your caption and the events that are discernible from the video from t=<start time> to t=<end time> seconds:“<predicted subject>"

Your task is to rate the candidate caption for Person ID 1 in the video on a scale of 1 to 5 for each of the following criteria and rating scale INDEPEN-DENTLY.

Evaluation Criteria:- Completeness: How completely does the caption cover all the relevant and essential aspects of Person ID 1 in the video?

Rating Scale:

\- 1 Very Low Completeness: The caption includes almost none of the important attributes for the person, missing most details.

\- 5 Extremely High Completeness: The caption fully covers all relevant and essential aspects, leaving no important details unaddressed.

Evaluation Criterion: - Correctness: How accurately does the caption describe the aspects and activities of Person ID 1?

Rating Scale: - 1 Very Low Correctness: The caption is mostly or entirely incorrect; The person cannot be described by any or most of the predicted attributes

\- 5 Extremely High Correctness: All attributes mentioned in the predicted caption belong to the person in the video.

Note that a caption may be correct but not complete and vice-versa. For example, if a caption has 2 attributes while the person requires 6 attributes to be uniquely defined, it is low complete. But if those 2 predicted attributes actually belong to the person, it has extremely high correctness. Similarly, if the predicted caption has 3 attributes and the person has only one of them and nothing else, the caption has high completeness since the actual attribute was captured, but low correctness because incorrect attributes were also predicted. Thus, completeness is similar to recall and correctness is similar to precision.

Evaluation Criterion: - Uniqueness: First, look at the diferent people in the video. Given the candidate caption, how easy it is to locate Person ID 1 in the video? Note that a simple caption can be highly discriminative. For example, if the video has only men and one woman, a caption like "a woman" is extremely unique as it can only be used to describe that one woman in the video.

Rating Scale:

\- 1 Very Low Uniqueness: The same caption can be used to describe multiple entities in the video

\- 5 Extremely Uniqueness: There cannot be any shred of doubt about which person from the video is being described by the caption.’

Activity: Consider only t=<start time> to t=<end time> seconds in the video. Consider only the activities of the Person ID 1: <subject description from ground truth>. Next, consider this candidate caption which we want to evaluate against your caption and the events that are discernible from the video from t=<start time> to t=<end time> seconds: “<predicted activity capti

Your task is to rate the candidate caption for Person ID 1 in the video for each of the following criteria and rating scales INDEPENDENTLY.

Evaluation Criteria:- Completeness: How completely does the caption cover all the relevant activities of Person ID 1 in the video?

Rating Scale:

\- 1 Very Low Completeness: The caption includes almost none of the important activities, missing most activity details.

\- 5 Extremely High Completeness: The caption fully covers all relevant and essential aspects, leaving no important details unaddressed.

Evaluation Criterion: - Correctness: How accurate is the caption regarding the activities of Person ID 1?

Rating Scale:

\- 1 Very Low Correctness: The caption is mostly or entirely incorrect; The person is not doing any or most of the predicted activities

\- 5 Extremely High Correctness: Almost all activities mentioned in the predicted caption are done by the person in the video.

Note that a caption may be correct but not complete and vice-versa. For example, if a caption has 2 activities while the person is doing 6 activities, it is low complete. But if those 2 predicted activities are actually done by the person, it has extremely high correctness. Similarly, if the predicted caption has 3 activities and the person is doing only one of them, the caption has high completeness since the actual activity was captured, but low correctness because incorrect activities were also predicted. Thus, completeness is similar to recall and correctness is similar to precision.

## References

1. Banerjee, S., Lavie, A.: Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In: Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization. pp. 65–72 (2005) 1

2. Lin, C.Y.: Rouge: A package for automatic evaluation of summaries. In: Text summarization branches out: Proceedings of the ACL-04 workshop. pp. 74–81 (2004) 1

3. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic evaluation of machine translation. In: Proceedings of the 40th annual meeting of the Association for Computational Linguistics. pp. 311–318 (2002) 1

4. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, S., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning (ICML) (2021) 1

5. Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bertnetworks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. pp. 3982–3992 (2019) 1