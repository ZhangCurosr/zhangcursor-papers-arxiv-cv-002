# TempJail: Temporal Jailbreak Attack against Large Vision-Language Models via Subtitle Scheduling

Ling Zhou<sup>1</sup>, Yihao Huang<sup>2</sup>, Jingling Sun<sup>1</sup>, Zhiwen Tian<sup>1</sup>, Yi Zeng<sup>1</sup>, Qihe Liu<sup>1</sup>, Shijie Zhou<sup>1</sup>

1. University of Electronic Science and Technology of China, Chengdu, China

2. East China Normal University, Shanghai, China

Abstract—Large vision-language models (LVLMs) have achieved remarkable progress in video understanding and reasoning. Despite extensive studies on text- and image-based jailbreaks, video jailbreaks against LVLMs remain largely unexplored. Existing video jailbreak methods mainly manipulate textual content embedded in videos, while overlooking how such information is organized over time. Our analysis reveals that jailbreak effectiveness depends not only on the semantics of textual information but also on its temporal presentation, including duration and timing-slot allocation. Motivated by this finding, we use subtitles, which are common in real-world videos and allow semantic content to be presented under precise temporal control without appearing visually intrusive, as a natural attack medium. Based on this insight, we propose TempJail, a black-box video-based jailbreak framework that constructs query-aligned dialogue-style subtitle sequences and optimizes their temporal scheduling to exploit temporal vulnerabilities in LVLMs and elicit responses that satisfy the harmful intent of the source query. Extensive experiments on four representative LVLMs and two datasets demonstrate that TempJail achieves the highest attack success rate across all evaluated model–dataset settings, outperforming the strongest baseline by 53 and 18 percentage points in dataset-averaged ASR on GPT-5 and Gemini 3.5-Flash, respectively.

## I. INTRODUCTION

In recent years, large language models have rapidly evolved toward multimodal intelligence, giving rise to large visionlanguage models (LVLMs) that have demonstrated remarkable progress in image understanding, video understanding, crossmodal question answering, and multimodal reasoning [1], [2]. Compared with traditional text-only large language models, LVLMs substantially expand the perceptual boundary of AI systems by incorporating visual and temporal signals. However, this expanded capability also introduces new security risks. In particular, when models directly consume semantically rich visual inputs such as images and videos, their safety alignment mechanisms may exhibit vulnerabilities that differ from those observed in purely textual settings [3], [4], [5], [6], [7].

Existing jailbreak studies have primarily targeted the textual and visual modalities [8], [9], while the safety of LVLMs under video inputs remains less explored. Early work such as VideoJail extends image-based typographic attacks to videos by overlaying largely static text on video frames [10]. More recent methods explore content composition and variation: MCV combines semantically related but distinct clips to increase visual diversity [11], whereas SPTV constructs typographic videos using semantic variants of harmful queries and safety-proximal text frames [12]. Despite their methods being reasonable, these methods mainly investigate what content is presented, with limited attention to how it is organized over time.

![](images/39b9e90f72c8eb536f8bd76023900994f04f20022694cc27a1d0c0914a1e6a5d.jpg)  
Fig. 1: Comparison of existing video-based jailbreaks and TempJail: prior methods rely on video content composition while largely ignoring temporal factors, whereas TempJail leverages query-aligned dialogue context and subtitle-level temporal optimization to achieve more effective attacks.

Temporal structure, however, is a defining characteristic of video. Motivated by the prevalence of subtitles in realworld videos, we use them to probe temporal vulnerabilities in LVLMs. The effect of subtitles depends not only on their semantic content, but also on their order, timing, and duration. Our preliminary experiments show that these temporal factors can substantially affect LVLM responses, revealing an underexplored source of safety risk in video-based LVLMs.

Based on this observation, we propose TempJail, a blackbox temporal jailbreak attack against large vision-language models via subtitle scheduling. Given a harmful query, TempJail first reformulates it into a sequence of semantically connected questions, which naturally lead to the original query. Through multi-turn interaction with a substitute model, TempJail then obtains a coherent dialogue-like response sequence, which is combined with the original query to construct the final subtitles. Next, according to the dialogue content, TempJail employs a text-to-video model to generate a background video, making the overall video appear more natural. Finally, Temp-Jail applies Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [13] to optimize the display duration of each subtitle segment, thereby controlling how harmful semantics are temporally exposed to the target model. As illustrated in Figure 1, unlike existing methods that mainly focus on increasing video content diversity or concatenating static typographic frames, TempJail shifts the focus from content construction alone to the temporal scheduling of subtitle semantics in videos.

Extensive experiments across multiple datasets and models verify the effectiveness of TempJail in black-box settings, and systematically analyze the effects of different subtitle configurations on attack success rates.

The main contributions of this paper are as follows:

• We introduce a temporal perspective on video-based jailbreak attacks against LVLMs. To the best of our knowledge, we are the first to investigate how the temporal organization of textual content along the video timeline affects attack effectiveness. Our findings extend prior studies that have primarily focused on content construction, highlighting temporal organization as an important yet underexplored factor in multimodal safety.

• We propose TempJail, a black-box subtitle-based video jailbreak framework. Specifically, the method consists of three stages: subtitle construction, video generation, and subtitle timing optimization. By jointly modeling subtitle content and its temporal organization, TempJail effectively exploits the temporal sensitivity of LVLMs.

• Extensive experiments on two datasets, four LVLMs, and four baselines show that TempJail consistently achieves the best performance, outperforming the strongest baseline by 53 percentage points. Ablation analysis reveals that temporal scheduling alone improves GPT-5’s averaged ASR from 11% to 71%, highlighting the importance of temporal organization in LVLM jailbreaks.

## II. RELATED WORK

## A. Large Vision-Language Models

Large vision-language models (LVLMs) have rapidly evolved from early image-centric systems such as BLIP-2 [14], InstructBLIP [15], and LLaVA [16] into a broader family of general-purpose multimodal models. These early LVLMs primarily connected pretrained visual encoders with large language models to support tasks involving static images, including image captioning, visual question answering, and vision-language instruction following. More recent research has extended this paradigm to temporally structured visual inputs, enabling models to process and reason over sequences of video frames. Representative open-source video-capable LVLMs include Video-ChatGPT [17], Video-LLaMA [18], LLaVA-Video [19], and CogVLM2 [20]. These models explore different mechanisms for aggregating multi-frame visual information and incorporating temporal cues, thereby supporting tasks such as video question answering, video captioning, multimodal dialogue, and temporal event understanding. Meanwhile, widely deployed multimodal model families, including the GPT series [21], [22], [23], the Gemini family [24], [25], [26], and Qwen3-VL variants [27], have further broadened the practical scope of multimodal perception and reasoning.

## B. Jailbreaking Large Vision-Language Models

Jailbreaking attacks against LVLMs have been extensively studied in the image modality. Existing methods conceal harmful intent through typographic prompts, multimodal linkage, automatically generated jailbreak prompts, and interactive black-box optimization [28], [29], [30], [31]. These studies have revealed diverse vulnerabilities of LVLMs under static visual inputs.

By contrast, video jailbreak attacks remain comparatively underexplored. VideoJail [10] leverages video generation models to amplify harmful content embedded in images and uses carefully designed textual prompts to direct the model’s attention toward malicious queries. MCV [11] constructs videos by concatenating multiple short clips depicting diverse contexts related to a harmful query, demonstrating the effects of clip number, video dynamics, and contextual diversity on jailbreak effectiveness. SPTV [12] constructs safety-proximal typographic videos by selecting visually diverse typographic frames that preserve the harmful intent while remaining close to benign videos in the representation space. Although these methods leverage different aspects of video, they mainly focus on generation, clip composition, or frame-level content, rather than the temporal progression of harmful semantics. In contrast, our work constructs dialogue-style subtitle sequences and optimizes segment durations to expose the temporal vulnerabilities of LVLMs under video inputs.

## III. MOTIVATION

To motivate the design of our method, we conduct a series of controlled experiments to identify the key factors that influence subtitle-based video jailbreak attacks. Our analysis focuses on two questions: (1) How do subtitle display form and content affect jailbreak effectiveness? (2) How do subtitle duration and timing-slot allocation affect jailbreak effectiveness when the subtitle content remains unchanged?

We use HADES [32] and VLJailbreakBench [33], two multimodal safety datasets covering diverse harmful-request categories. We randomly sample 50 examples from each dataset while preserving their category distributions. We evaluate Qwen3-VL-Plus [34] and Qwen3-VL-32B-Instruct [27] as the target models and report attack success rate (ASR) and refusal rate (RR).

Unless otherwise specified, each video has a resolution of 1280 × 720, a duration of 5 seconds, and a frame rate of 10 FPS. During target-model inference, videos are sampled at the same rate of 10 FPS, ensuring that every rendered frame is included in the input. The decoding temperature is fixed to 0. All videos in this section use a solid-white background to exclude interference from non-textual visual content.

Each dialogue consists of five question–answer rounds, yielding ten subtitle segments, followed by the harmful query. The irrelevant dialogue, generated by GPT-5, discusses scenery unrelated to the query, whereas the relevant dialogue is constructed using the method mentioned in Section IV-C to remain semantically relevant and progressively establish the query context.

TABLE I: Effects of subtitle display form, content, and duration on HADES using Qwen3-VL-Plus. We report attack success rate (ASR) and refusal rate (RR). “–” indicates that, at a 1-second duration, dialogue + query is equivalent to query only because the query occupies the entire subtitle duration. Key results are highlighted in bold.
<table><tr><td rowspan="2">Display Pattern</td><td rowspan="2">Subtitle Content</td><td colspan="6">ASR (%)</td><td colspan="6">RR (%)</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>4s</td><td>5s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>4s</td><td>5s</td><td>Avg.</td></tr><tr><td rowspan="3">Subtitle-Blank</td><td>Query only</td><td>38</td><td>42</td><td>44</td><td>40</td><td>42</td><td>41.2</td><td>63</td><td>58</td><td>58</td><td>58</td><td>61</td><td>59.6</td></tr><tr><td>Irrelevant dialogue + query</td><td>一</td><td>40</td><td>44</td><td>38</td><td>34</td><td>39.0</td><td>一</td><td>65</td><td>58</td><td>60</td><td>64</td><td>61.8</td></tr><tr><td>Relevant dialogue + query</td><td>一</td><td>58</td><td>52</td><td>60</td><td>64</td><td>58.5</td><td>一</td><td>39</td><td>40</td><td>38</td><td>34</td><td>37.8</td></tr><tr><td rowspan="3">Blank-Subtitle</td><td>Query only</td><td>48</td><td>46</td><td>44</td><td>42</td><td>42</td><td>44.4</td><td>52</td><td>54</td><td>55</td><td>55</td><td>61</td><td>55.4</td></tr><tr><td>Irrelevant dialogue + query</td><td>一</td><td>26</td><td>38</td><td>30</td><td>34</td><td>32.0</td><td>一</td><td>70</td><td>62</td><td>65</td><td>64</td><td>65.3</td></tr><tr><td>Relevant dialogue + query</td><td>一</td><td>54</td><td>50</td><td>58</td><td>64</td><td>56.5</td><td>一</td><td>46</td><td>50</td><td>41</td><td>34</td><td>42.8</td></tr><tr><td rowspan="3">Blank-Subtitle-Blank</td><td>Query only</td><td>46</td><td>48</td><td>42</td><td>42</td><td>42</td><td>44.0</td><td>54</td><td>54</td><td>52</td><td>56</td><td>61</td><td>55.4</td></tr><tr><td>Irrelevant dialogue + query</td><td>一</td><td>38</td><td>40</td><td>36</td><td>34</td><td>37.0</td><td>一</td><td>57</td><td>62</td><td>63</td><td>64</td><td>61.5</td></tr><tr><td>Relevant dialogue + query</td><td>一</td><td>64</td><td>60</td><td>62</td><td>64</td><td>62.5</td><td>1</td><td>37</td><td>37</td><td>39</td><td>34</td><td>36.8</td></tr></table>

## A. Subtitle Display Form and Content

We first study how subtitle display form and content affect jailbreak performance. This experiment is conducted on HADES using Qwen3-VL-Plus. We compare three display forms—Subtitle–Blank, Blank–Subtitle, and Blank–Subtitle– Blank—and three content configurations—query only, irrelevant dialogue + query, and relevant dialogue + query.

The three display forms are defined as follows. If the subtitle display duration is 1 second, then under Blank–Subtitle, the first 4 seconds contain no subtitles and the subtitle appears only in the final second. Under Blank–Subtitle–Blank, the subtitle appears only in the middle second, with no subtitles in the first and last 2 seconds. Under Subtitle–Blank, the subtitle appears in the first second, followed by 4 seconds without subtitles. The same temporal construction is applied when the subtitle display duration is extended to 2-5 seconds.

Table I shows that subtitle content is the dominant factor. Across all three display forms, relevant dialogue + query achieves the highest average ASR and the lowest average RR. Its average ASRs are 58.5%, 56.5%, and 62.5%, compared with 41.2%, 44.4%, and 44.0% for query only. In contrast, irrelevant dialogue provides no consistent benefit and can substantially increase refusal. Under Blank–Subtitle, for example, it reduces the average ASR from 44.4% to 32.0% and increases the average RR from 55.4% to 65.3%.

Display form also affects performance, although less strongly than subtitle content. For relevant dialogue + query, Blank–Subtitle–Blank achieves the best average result, with an ASR of 62.5% and an RR of 36.8%.

Overall, effective subtitle-based attacks require both semantically relevant dialogue and an appropriate display form. These findings motivate TempJail to construct a query-relevant dialogue rather than merely adding unrelated text or directly repeating the harmful query.

![](images/07aa4e75d4d012ad7a68111da5cea3787c44dc1e5804aecd11da7646c87fe5ba.jpg)  
Fig. 2: Effect of timing-slot allocation.

## B. Subtitle Duration and Timing-Slot Allocation

We next examine how the total subtitle duration and its allocation across segments affect jailbreak effectiveness for a fixed subtitle sequence.

In Table I, the total subtitle duration varies from 1 to 5 seconds. For dialogue + query, the harmful query is displayed for 1 second, while the remaining time is divided among the ten dialogue segments. When the total subtitle duration reaches 5 seconds, the subtitles span the entire video and the three display forms become equivalent.

Increasing the subtitle duration does not consistently improve jailbreak effectiveness across subtitle-content settings. Nevertheless, when relevant dialogue + query is displayed throughout the full 5-second video, it achieves an ASR of 64% and an RR of 34%, yielding the best joint ASR–RR performance among all evaluated configurations. Building on the optimal setting identified above, we use relevant dialogue + query as the subtitle content and display the complete subtitle sequence throughout the entire 5-second video.

We further examine how the fixed 5-second duration is allocated across subtitle segments by comparing equal allocation with three random allocations generated using different seeds. With semantic content and visual input held constant, this experiment isolates the effect of timing-slot allocation.

Figure 2 shows that subtitle timing allocation substantially affects ASR, and equal allocation is not always optimal. For example, on HADES, random allocation improves ASR from 86% to 90% for Qwen3-VL-Plus, while on VLJailbreakBench, Qwen3-VL-32B-Instruct achieves an increase from 52% to 68%. Moreover, different random schedules lead to different attack effectiveness even with identical subtitle content, demonstrating that LVLM responses are sensitive to when information is presented during video playback.

These results indicate that temporal scheduling is a critical factor in subtitle-based video jailbreak attacks. Therefore, subtitle timing should be treated as an optimization variable rather than a simple presentation detail.

## IV. METHODOLOGY

## A. Problem Definition

Given an original harmful question $Q _ { h }$ and a target LVLM ${ \mathcal { C } } ,$ our goal is to construct a multimodal input that induces C to generate a response satisfying the harmful intent of $Q _ { h } .$ Each query consists of an adversarial video $V _ { \mathrm { a d v } }$ and a benign textual prompt $P _ { b }$ , where $P _ { b }$ does not explicitly contain the harmful request but only guides the model to process the video content. The model response is formulated as

$$
Y = { \mathcal { C } } ( V _ { \mathrm { a d v } } , P _ { b } ) .\tag{1}
$$

To evaluate attack success, we define a judge function $\mathcal { I } ( Y , Q _ { h } ) \in \{ 0 , 1 \}$ , where $\mathcal { I } ( Y , Q _ { h } ) = 1$ indicates that the response $Y$ satisfies the harmful intent of $Q _ { h }$ . The attack success is defined as

$$
\mathrm { S u c c } ( V _ { \mathrm { a d v } } , P _ { b } , Q _ { h } ; \mathcal { C } ) = \mathbb { I } [ \mathcal { I } ( \mathcal { C } ( V _ { \mathrm { a d v } } , P _ { b } ) , Q _ { h } ) = 1 ] .\tag{2}
$$

For a fixed benign prompt $P _ { b } .$ , the objective is to find an adversarial video:

$$
V _ { \mathrm { a d v } } ^ { * } = \arg \operatorname* { m a x } _ { V _ { \mathrm { a d v } } } \mathbb { E } [ \operatorname { S u c c } ( V _ { \mathrm { a d v } } , P _ { b } , Q _ { h } ; \mathcal { C } ) ] .\tag{3}
$$

## B. Overview of the Proposed Method

We construct the adversarial video by designing three components: subtitle sequence S, background video B, and subtitle timing configuration θ. Specifically, given a harmful query $Q _ { h }$ our method consists of three stages: constructing a subtitle sequence, generating a semantically consistent background video, and optimizing the subtitle timing (See Figure 3): $Q _ { h } \to S , S \to B , ( S , B ) \to \theta .$

The subtitle construction stage embeds the harmful query into a coherent dialogue-like sequence, the video generation stage provides a natural visual context, and the timing optimization stage searches for an effective temporal arrangement of subtitle segments. The components obtained are combined to form the final adversarial video $V _ { \mathrm { a d v } } ^ { * }$

## C. Dialogue Sequence and Subtitle Construction

The first stage transforms the original harmful query into a dialogue-style subtitle sequence. Drawing inspiration from the conversational decomposition strategy used in prior textonly multi-turn jailbreaks [35], we prepend a short multi-turn dialogue to provide conversational context, making the subtitle stream resemble a natural conversation unfolding over time.

Given an original harmful query $Q _ { h }$ , we first generate a sequence of related questions with a substitute model $\mathcal { C } _ { \mathrm { s u b } } \mathrm { : }$

$$
\mathcal { Q } = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { M } \} .\tag{4}
$$

These questions are sequentially answered by $\mathcal { C } _ { \mathrm { s u b } }$ , producing the dialogue responses:

$$
a _ { i } = \mathcal { C } _ { \mathrm { s u b } } ( q _ { i } | q _ { < i } , a _ { < i } ) .\tag{5}
$$

The resulting dialogue history is represented as

$$
D = \{ ( q _ { 1 } , a _ { 1 } ) , ( q _ { 2 } , a _ { 2 } ) , \dots , ( q _ { M } , a _ { M } ) \} .\tag{6}
$$

After converting the dialogue history into subtitle segments, the final subtitle sequence is constructed as

$$
S = D \oplus Q _ { h } ,\tag{7}
$$

where $\oplus$ denotes string concatenation.

## D. Background Video Generation

Given the constructed subtitle stream $S ,$ the second stage generates a background video B that provides visual context for the dialogue.

Specifically, a scene planner $\mathcal { G } _ { \mathrm { s c e n e } }$ first derives a high-level scene description P from the subtitle context:

$$
P = \mathcal { G } _ { \mathrm { s c e n e } } ( S ) ,
$$

where P represents the visual attributes of the scene.

A text-to-video generator $\mathcal { G } _ { \mathrm { v i d } }$ then produces the background video:

$$
B = { \mathcal { G } } _ { \mathrm { v i d } } ( P ) .
$$

The generated video remains benign and contains no additional textual or instructional information. Its purpose is to provide a natural carrier for the subtitle stream, while the main attack signal is conveyed through the subtitle content and its temporal presentation.

## E. Subtitle Timing Optimization via CMA-ES

Our preliminary analyses show that subtitle timing is a critical factor affecting attack performance. However, determining an effective timing configuration is challenging, as the impact of each subtitle segment is not independent. The duration assigned to one segment may influence the interpretation of subsequent segments, and the overall attack effectiveness depends on the joint temporal organization of the entire subtitle sequence. Therefore, we formulate subtitle scheduling as a black-box optimization problem and employ CMA-ES to search for an effective timing configuration.

Let the subtitle sequence S contain N segments, denoted as ${ \cal { S } } = \{ u _ { i } \} _ { i = 1 } ^ { N }$ . Given the total video duration $T ,$ the timing configuration is defined as

$$
\boldsymbol { \theta } = \{ t _ { i } \} _ { i = 1 } ^ { N } ,\tag{8}
$$

![](images/0bff8cd2d10907b1692800815d368a86a0b0d77c9db2db2ac0e221992a00297d.jpg)  
Fig. 3: Overview of our methodology. We first construct a multi-turn subtitle sequence, then generate a semantically matched background video, and finally optimize subtitle timing via CMA-ES.

where $t _ { i }$ represents the display duration of subtitle segment $u _ { i }$ . The durations satisfy

$$
\sum _ { i = 1 } ^ { N } t _ { i } = T ,\tag{9}
$$

Given a timing configuration θ, the attack effectiveness is evaluated by the objective function

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta \in \Theta } F ( \theta ; B , S , Q _ { h } ) ,\tag{10}
$$

where Θ denotes the feasible timing space of the video and $F ( \cdot )$ measures the attack success score under the corresponding timing configuration.

Since the objective is black-box and non-differentiable, we adopt CMA-ES to jointly optimize the timing configuration. Specifically, CMA-ES searches in a bounded N-dimensional latent space and maps each latent vector to a valid timing configuration:

$$
\theta = \Psi ( z ) ,\tag{11}
$$

where $z \in \mathbb { R } ^ { N }$ is the latent timing variable and $\Psi ( \cdot )$ enforces non-negativity, minimum-duration constraints, and the fixed total duration. By jointly optimizing the complete timing configuration, CMA-ES captures the temporal interactions among different subtitle segments. The detailed optimization procedure is presented in Algorithm 1.

## V. EXPERIMENTS

## A. Experimental Setup

Datasets: We use two representative multimodal safety datasets: HADES [32] and VLJailbreakBench [33], covering diverse harmful-request categories. We randomly sample 50 examples from each dataset while preserving their category distributions.

```powershell
Algorithm 1: Subtitle timing optimization with CMA-ES
Require: Subtitle sequence $S ,$ background video B, source
query $\textstyle Q _ { h } ,$ duration T, iterations $G ,$ population size λ
Ensure: Optimized timing configuration $\theta ^ { * }$
1: Initialize CMA-ES distribution and $f ^ { * } \gets - \infty$
2: for $g = 1 , \ldots , G$ do
3: Sample latent candidates $\{ z _ { k } ^ { ( g ) } \} _ { k = 1 } ^ { \lambda }$
4: for each candidate $z _ { k } ^ { ( g ) }$ do
5: Compute timing configuration $\theta _ { k } ^ { ( g ) } = \Psi ( z _ { k } ^ { ( g ) } )$
6: Obtain attack score $f _ { k } ^ { ( g ) }$
7: if $f _ { k } ^ { ( g ) } \geq f ^ { * }$ then
8: $\boldsymbol { \theta ^ { * } }  \boldsymbol { \theta } _ { k } ^ { ( g ) }$
9: $f ^ { * } \gets \tilde { f } _ { k } ^ { ( g ) }$
10: if $f _ { k } ^ { ( g ) } \ge f _ { \mathrm { t a r g e t } }$ then return $\theta ^ { * }$
11: end if
12: end for
13: Update CMA-ES distribution using candidate scores
14: end for
15: return $\theta ^ { * }$
```

Target models: Our evaluation covers four popular videocapable LVLMs: Qwen3-VL-Plus [34], GPT-5 [23], Gemini 3.5-Flash [26], and the open-source Qwen3-VL-32B-Instruct [27].

Baselines: We compare TempJail with four representative multimodal jailbreak baselines: FigStep [28], VideoJail [10], SPTV [12], and MCV [11]. Since FigStep is originally imagebased, we adapt it into a video-format variant for comparison under the same video configuration.

In addition to the full TempJail pipeline, we further evaluate TempJail-Uniform with uniformly allocated subtitle slots and TempJail-White with a solid-white background to isolate the effects of temporal scheduling and semantic scene generation, respectively.

TABLE II: Attack success rate (ASR, %) of different multimodal jailbreak methods on VLJailbreakBench and HADES. TJ U denotes TempJail-Uniform with uniformly allocated subtitle slots, and TJ-W denotes TempJail-White with a solid-white background. The best result in each setting is highlighted in bold.
<table><tr><td rowspan="3">Target LVLM</td><td colspan="7">VLJailbreakBench</td><td colspan="7">HADES</td></tr><tr><td colspan="4">Baselines</td><td colspan="3">Ours</td><td colspan="3">Baselines</td><td colspan="3"></td></tr><tr><td>FigStep</td><td>VideoJail</td><td>SPTV</td><td>MCV</td><td>TJ-U</td><td>TJ-W</td><td>TempJail</td><td>FigStep</td><td>VideoJail</td><td>SPTV</td><td>MCV</td><td>TJ-W</td><td>TempJail</td></tr><tr><td>GPT-5</td><td>6</td><td>4</td><td>20</td><td>18</td><td>12</td><td>78</td><td>70</td><td>4</td><td>4</td><td>16</td><td>8</td><td></td><td></td></tr><tr><td>Gemini 3.5-Flash</td><td>70</td><td>60</td><td>58</td><td>84</td><td>78</td><td></td><td></td><td></td><td>44</td><td>64</td><td>10 84</td><td>78 86</td><td>72 94</td></tr><tr><td>Qwen3-VL-Plus</td><td>48</td><td>86</td><td>64</td><td>94</td><td>92</td><td></td><td></td><td>50 64</td><td>72</td><td>80</td><td>80</td><td>90</td><td>96</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>56</td><td>48</td><td>64</td><td>90</td><td>90</td><td>92</td><td></td><td>66</td><td>66</td><td>84</td><td>80</td><td>94</td><td>98</td></tr><tr><td>Average</td><td>45.0</td><td>49.5</td><td>51.5</td><td>71.5</td><td>68.0</td><td>87.0</td><td>89.0</td><td>43.5</td><td>46.0</td><td>49.5</td><td>59.0 63.5</td><td>87.0</td><td>90.0</td></tr></table>

Evaluation metric: For each sample, we query the target model five times and use GPT-5 [23] as an automated judge. A response is considered successful if it substantively fulfills the harmful intent, rather than refusing or providing only generic safety guidance. A sample is considered successfully attacked if at least one of the five responses succeeds. We report the resulting sample-level attack success rate (ASR) as our primary metric., following standard jailbreak evaluation protocols.

Implementation details: Unless otherwise specified, we use Qwen3-VL-Flash [36] as the substitute model for subtitle construction, Qwen3.6-Flash [37] as the scene-prompt planner, and Runway Gen-4.5 [38] for background-video generation. Each TempJail video is rendered at a resolution of 1280×720, with a fixed duration of 5 seconds and a rendering frame rate of 24 FPS. For temporal optimization, we run CMA-ES for at most three generations with a population size of 10. The initial CMA-ES mean is randomly sampled from a standard Gaussian distribution, and the initial step size is set to 1.0. During evaluation through the explicit frame-input pipeline, videos are uniformly sampled at 4 FPS, and all target models use greedy decoding with a temperature of 0.

## B. Main Results

Table II compares TempJail with four representative videobased jailbreak baselines. TempJail achieves the highest average attack success rate (ASR) on both datasets, reaching 89.0% on VLJailbreakBench and 90.0% on HADES. On proprietary models, TempJail achieves 70% and 72% ASR on GPT-5 and 90% and 94% on Gemini 3.5-Flash across VLJailbreakBench and HADES, respectively. By comparison, the strongest prior results on GPT-5 are only 20% and 16%, showing that temporal subtitle organization remains effective in settings where existing video jailbreak baselines achieve relatively low ASR. TempJail also performs strongly on the Qwen3-VL family, achieving 100%/96% ASR on Qwen3-VL-Plus and 96%/98% on Qwen3-VL-32B-Instruct.

Across the eight model–dataset combinations, TempJail achieves the best result among the complete attack methods in every setting, indicating that its effectiveness is consistent across the evaluated architectures and access settings. The close average ASR on VLJailbreakBench and HADES also suggests that the improvement is not limited to one dataset.

TABLE III: Ablation study on Qwen3-VL-Plus and Qwen3- VL-32B-Instruct. “White” denotes a solid-white background, “TJ-Video” denotes the background video generated by Temp-Jail. “VLJB” denotes VLJailbreakBench. Results are reported in ASR (%). “Query” denotes the query-only subtitle setting, while“Dialogue” denotes the constructed dialogue + query subtitle sequence.
<table><tr><td>Background Subtitle</td><td></td><td>Timing</td><td>HADES</td><td>VLJB</td></tr><tr><td colspan="5">Qwen3-VL-Plus</td></tr><tr><td>White</td><td>Query</td><td></td><td>52</td><td>36</td></tr><tr><td>White</td><td>Dialogue</td><td>Uniform</td><td>86</td><td>84</td></tr><tr><td>TJ-Video</td><td>Dialogue</td><td>Uniform</td><td>80</td><td>92</td></tr><tr><td>White</td><td>Dialogue</td><td>CMA-ES</td><td>90</td><td>94</td></tr><tr><td>TJ-Video</td><td>Dialogue</td><td>CMA-ES</td><td>96</td><td>100</td></tr><tr><td colspan="5">Qwen3-VL-32B-Instruct</td></tr><tr><td>White</td><td>Query</td><td></td><td>32</td><td>12</td></tr><tr><td>White</td><td>Dialogue</td><td>Uniform</td><td>76</td><td>52</td></tr><tr><td>TJ-Video</td><td>Dialogue</td><td>Uniform</td><td>80</td><td>90</td></tr><tr><td>White</td><td>Dialogue</td><td>CMA-ES</td><td>94</td><td>92</td></tr><tr><td>TJ-Video</td><td>Dialogue</td><td>CMA-ES</td><td>98</td><td>96</td></tr></table>

The controlled variants further clarify the contributions of temporal scheduling and semantic scene generation. TempJail-Uniform performs worse than the full TempJail pipeline, demonstrating the benefit of optimizing subtitle-slot allocation. TempJail-White remains highly effective, indicating that subtitles carry the primary jailbreak signal, while the generated semantic scene provides complementary contextual information. Together, these comparisons show that temporal scheduling is the main contributor, and semantic scene generation provides a modest average gain, although its effect is model-dependent and it does not improve performance on GPT-5. Overall, these results support our central claim that how harmful semantics are scheduled over time is a critical attack factor beyond the textual content itself.

## C. Ablation Study

To better understand the contribution of each component in TempJail, we conduct an ablation study on subtitle construction, background generation, and temporal optimization. As shown in Table III, replacing the query-only subtitle with the dialogue-style sequence under a white background and uniform timing increases the average ASR from 33.0% to 74.5%, demonstrating the importance of constructing coherent conversational context. Under uniform timing, incorporating the generated semantic background further raises the average ASR to 85.5%, corresponding to the TempJail-Uniform configuration used in the main results. Temporal optimization yields the most consistent improvement: applying CMA-ES with a white background increases the average ASR to 92.5%, corresponding to TempJail-White, while combining CMA-ES with the generated semantic background raises it to 97.5%. The complete TempJail pipeline achieves the highest ASR in all four evaluation settings, indicating that subtitle construction and temporal optimization are the main sources of improvement, while semantic background generation provides a modest complementary gain.

![](images/28d773b9fe09c09a6390ab12080def7bc0fde4d35df3a0b9d8ec4d72897efad1.jpg)  
(a) Effect of Frame-Sampling Rate on ASR

![](images/e39b8a758a42c43577f014ea9a297b628faac9cb0d7dbb5054945b86ad5ca30f.jpg)  
(b) Effect of Iterations on ASR

![](images/a66e4b58f4f9fde3ea5799e9a842772b78661708ee70c02625a40c31c68198ff.jpg)  
(c) Effect of Temperature on ASR  
Fig. 4: Parameter analysis of TempJail. (a) Effect of the frame-sampling rate on Qwen3-VL-Plus over VLJailbreakBench and HADES. (b) Effect of the number of optimization iterations on Qwen3-VL-Plus over both datasets. (c) Effect of the decoding temperature on Qwen3-VL-Plus and Qwen3-VL-32B-Instruct over HADES. ASR denotes the attack success rate.

## D. Parameter Analysis

We analyze the sensitivity of TempJail to three key hyperparameters: the model frame-sampling rate (FPS), the number of optimization iterations, and the decoding temperature. The results are summarized in Figure 4.

Effect of the Model Frame-Sampling Rate: As shown in Figure 4(a), increasing the number of video frames sampled by the target model from 2 FPS to 4–6 FPS substantially improves the ASR on both datasets. On VLJailbreakBench, the ASR increases from 96% at 2 FPS to 100% at both 4 and 6 FPS. On HADES, it rises from 88% at 2 FPS to 96% at 4 FPS and reaches its highest value of 98% at 6 FPS. Further increasing the model frame-sampling rate to 8 or 10 FPS provides no additional improvement and instead results in a slight performance decrease. These results indicate that sampling frames at a moderate rate allows the target model to capture sufficient temporal information from the dynamically presented subtitles, whereas denser sampling contributes little additional useful information. We therefore set the model frame-sampling rate to 4 FPS in the main experiments, as it achieves the best performance on VLJailbreakBench and nearbest performance on HADES while limiting the computational cost associated with processing additional frames.

Effect of the Number of Iterations: Figure 4(b) shows that most performance gains are obtained within the first few optimization iterations. For Qwen3-VL-Plus, one iteration increases the ASR from 62% to 90% on VLJailbreakBench and from 60% to 92% on HADES. The ASR continues to increase and reaches 100% and 96%, respectively, after three iterations. Additional iterations yield only marginal improvements. These findings indicate that the temporal scheduling process converges quickly, as an effective subtitle arrangement can generally be identified within a small number of iterations. Accordingly, we use three iterations in the main experiments to achieve strong attack performance while limiting optimization overhead.

Effect of Temperature: As shown in Figure 4(c), the performance of TempJail varies only slightly across different decoding temperatures. When the temperature ranges from 0 to 1, the ASR remains between 96% and 100% for both Qwen3-VL-Plus and Qwen3-VL-32B-Instruct, with both models reaching 100% ASR at a decoding temperature of 1. These limited fluctuations indicate that TempJail does not depend on a narrowly selected decoding temperature. Instead, its effectiveness is primarily determined by the constructed video prompt and temporal subtitle schedule rather than by a specific decoding configuration.

## VI. CONCLUSION

In this paper, we reveal the temporal vulnerability of large vision-language models under video inputs and propose TempJail, a black-box video jailbreak framework via subtitle scheduling. Extensive experiments across multiple advanced LVLMs and two multimodal safety datasets demonstrate that TempJail consistently outperforms representative jailbreak baselines. Further analyses show that coherent subtitle construction and optimized temporal scheduling are the primary sources of its effectiveness. Ultimately, our findings expose temporal presentation as an important attack surface in videocapable LVLMs and highlight the need for future multimodal safety mechanisms to explicitly account for the temporal dynamics of textual and visual information in video inputs.

[1] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

[2] Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, and G. Shi, “A survey of state of the art large vision language models: Benchmark evaluations and challenges,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1587–1606.

[3] M. Ye, X. Rong, W. Huang, B. Du, N. Yu, and D. Tao, “A survey of safety on large vision-language models: Attacks, defenses and evaluations,” 2025. [Online]. Available: https://arxiv.org/abs/2502.14881

[4] X. Liu, Y. Zhu, Y. Lan, C. Yang, and Y. Qiao, “Safety of Multimodal Large Language Models on Images and Text,” in IJCAI, 2024.

[5] Y. Fan, Y. Cao, Z. Zhao, Z. Liu, and S. Li, “Unbridled icarus: A survey of the potential perils of image inputs in multimodal large language model security,” in 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2024, pp. 3428–3433.

[6] H. Jin, L. Hu, X. Li, P. Zhang, C. Chen, J. Zhuang, and H. Wang, “Jailbreakzoo: Survey, landscapes, and horizons in jailbreaking large language and vision-language models,” 2025. [Online]. Available: https://arxiv.org/abs/2407.01599

[7] D. Liu, M. Yang, X. Qu, P. Zhou, Y. Cheng, and W. Hu, “A Survey of Attacks on Large Vision–Language Models: Resources, Advances, and Future Trends,” IEEE Transactions on Neural Networks and Learning Systems, 2025.

[8] S. Yi, Y. Liu, Z. Sun, T. Cong, X. He, J. Song, K. Xu, and Q. Li, “Jailbreak attacks and defenses against large language models: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2407.04295

[9] S. Wang, Z. Long, Z. Fan, and Z. Wei, “From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 17 568–17 582.

[10] W. Hu, S. Gu, Y. Wang, and R. Hong, “Videojail: Exploiting videomodality vulnerabilities for jailbreak attacks on multimodal large language models,” in ICLR 2025 Workshop on Building Trust in Language Models and Applications, 2025.

[11] C. Kang, S. Sun, H. Jun, and J. H. Kim, “Jailbreaking multimodal large language models using multi-clip video,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). San Diego, California, United States: Association for Computational Linguistics, Jul. 2026, pp. 25 863–25 889. [Online]. Available: https://aclanthology.org/2026.acl-long.1186/

[12] D. Wang, X. He, X. Lyu, and B. Xiao, “Breaking Multimodal LLM Safety via Video-Driven Prompting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2026, pp. 8566–8576.

[13] N. Hansen and A. Ostermeier, “Completely derandomized selfadaptation in evolution strategies,” Evolutionary Computation, vol. 9, no. 2, pp. 159–195, Jun. 2001.

[14] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[15] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49 250–49 267, 2023.

[16] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.

[17] M. Maaz, H. Rasheed, S. Khan, and F. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 12 585– 12 602.

[18] H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audiovisual language model for video understanding,” in Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations, 2023, pp. 543–553.

[19] Y. Zhang, J. Wu, W. Li, B. Li, Z. MA, Z. Liu, and C. Li, “Llava-video: Video instruction tuning with synthetic data,” Transactions on Machine Learning Research, 2025.

[20] W. Hong, W. Wang, M. Ding, W. Yu, Q. Lv, Y. Wang et al., “Cogvlm2: Visual language models for image and video understanding,” 2024. [Online]. Available: https://arxiv.org/abs/2408.16500

[21] OpenAI, “GPT-4V(ision) system card,” https://openai.com/index/gpt-4vsystem-card/, Sep. 2023, accessed: 2026-07-29.

[22] A. Hurst, A. Lerer, A. P. Goucher et al., “Gpt-4o system card,” 2024. [Online]. Available: https://arxiv.org/abs/2410.21276

[23] A. Singh, A. Fry, A. Perelman et al., “Openai gpt-5 system card,” 2026. [Online]. Available: https://arxiv.org/abs/2601.03267

[24] Gemini Team, “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” 2024. [Online]. Available: https://arxiv.org/abs/2403.05530

[25] G. Comanici, E. Bieber, M. Schaekermann, et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” 2025. [Online]. Available: https://arxiv.org/abs/2507.06261

[26] Gemini Team, “Gemini 3.5 Flash Model Card,” https://deepmind.google/ models/model-cards/gemini-3-5-flash/, May 2026, accessed: 2026-07- 29.

[27] S. Bai, Y. Cai, R. Chen et al., “Qwen3-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[28] Y. Gong, D. Ran, J. Liu, C. Wang, T. Cong, A. Wang, S. Duan, and X. Wang, “Figstep: Jailbreaking large vision-language models via typographic visual prompts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, pp. 23 951–23 959.

[29] Y. Wang, X. Zhou, Y. Wang, G. Zhang, and T. He, “Jailbreak large vision-language models through multi-modal linkage,” 2025. [Online]. Available: https://arxiv.org/abs/2412.00473

[30] Y. Liu, C. Cai, X. Zhang, X. Yuan, and C. Wang, “Arondight: Red teaming large vision language models with auto-generated multi-modal jailbreak prompts,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 3578–3586.

[31] R. Cheng, Y. Ding, S. Cao, R. Duan, X. Jia, S. Yuan, S. Qin, Z. Wang, and X. Jia, “Pbi-attack: Prior-guided bimodal interactive black-box jailbreak attack for toxicity maximization,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 609–628.

[32] Y. Li, H. Guo, K. Zhou, W. X. Zhao, and J.-R. Wen, “Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 174–189.

[33] R. Wang, J. Li, Y. Wang, B. Wang, X. Wang, Y. Teng, Y. Wang, X. Ma, and Y.-G. Jiang, “Ideator: Jailbreaking and benchmarking large visionlanguage models using themselves,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 8875–8884.

[34] Alibaba Cloud, “Qwen3-VL-Plus Model Information,” https://help. aliyun.com/zh/model-studio/qwen3-vl-plus, Dec. 2025, accessed: 2026- 07-29.

[35] Q. Ren, H. Li, D. Liu, Z. Xie, X. Lu, Y. Qiao, L. Sha, J. Yan, L. Ma, and J. Shao, “Llms know their vulnerabilities: Uncover safety gaps through natural distribution shifts,” 2026. [Online]. Available: https://arxiv.org/abs/2410.10700

[36] Alibaba Cloud, “Qwen3-VL-Flash Model Information,” https://help. aliyun.com/zh/model-studio/qwen3-vl-flash, Jan. 2026, accessed: 2026- 07-29.

[37] ——, “Qwen3.6-Flash Model Information,” https://help.aliyun.com/zh/ model-studio/qwen3-6-flash, Apr. 2026, snapshot version qwen3.6- flash-2026-04-16. Accessed: 2026-07-29.

[38] Runway, “Runway Gen-4.5: State-of-the-Art AI Video Generation,” https://runway.com/research/introducing-runway-gen-4.5, Dec. 2025, accessed: 2026-07-29.