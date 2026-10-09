# FASTBENCH: CAN STREAMING VLMS PERCEIVE HIGH-DYNAMIC REAL-WORLD STREAMS?

Yuxuan Hu<sup>1∗</sup>, Weikang Shi<sup>1,\*</sup>, Yang Bo<sup>2</sup>, Xudong Lu<sup>1</sup>, Xintong Guo<sup>2</sup>, Shuhan LI<sup>2</sup>, Yuyang He<sup>2</sup>, Huankang Guan<sup>2</sup>, Peiwen Sun<sup>1</sup>, Yunqiao Yang<sup>1</sup>, Wenbo Li<sup>1</sup>, Rui Liu<sup>2</sup>, Hongsheng Li<sup>1</sup>

<sup>1</sup>CUHK MMLab

<sup>2</sup>Huawei Research

## ABSTRACT

Streaming Video Large Language Models (VLMs) have emerged as a promising paradigm for real-time, continuous video understanding. However, existing streaming benchmarks predominantly target low-dynamic scenarios, masking a critical challenge: under a bounded context budget, a model must trade off long temporal history, spatial resolution, and temporal granularity, and the sparse sampling rates (1–2 FPS) adopted by current systems inevitably miss fast-paced events. To bridge this gap, we introduce FastBench, a benchmark designed to evaluate streaming VLMs on high-dynamic real-world video streams. FastBench is built with a trajectory-grounded pipeline: a VLM proposes candidate QA pairs from native high-frame-rate clips, a low-FPS filter discards pseudo-dynamic questions that remain answerable at 2 FPS, and a Vision Expert Verification stage grounds answers on object trajectories extracted by SAM3 and CoTracker3 to suppress hallucinations, followed by three rounds of human inspection. The resulting 306 QA pairs span eight domains (e.g., sports, gaming, wildlife, and transportation), six capability dimensions, and forward, instant, and backward temporal scopes, each with human-annotated evidence intervals. Alongside the benchmark, we present ProactiveFrame, a training-free baseline that lets a model raise or restore the frame rate of incoming chunks through text tokens, supported by a dual-tier sliding window that buffers recent high-FPS chunks and gracefully degrades them into sparse history. Experiments show that high-dynamic perception remains far from solved: the strongest model, Gemini-3.5-Flash, reaches only 50.7%. Denser sampling helps substantially (Qwen3-VL-8B improves from 32.9% at 2 FPS to 44.6% at 24 FPS) but saturates as history is compressed. ProactiveFrame improves over sparse uniform sampling (+5.4% and +1.5%), yet stays well below oracle-guided focusing, revealing that current VLMs struggle to decide, from the visual stream alone, when finer temporal perception is needed. FastBench establishes a rigorous testbed and offers actionable insights for highdynamic streaming video understanding. Our project is open source and available at https://github.com/Ashone3/FastBench.

## 1 INTRODUCTION

The remarkable success of Large Language Models (LLMs) and Vision-Language Models (VLMs) has catalyzed significant progress in the domain of video understanding. Recent advancements in Video VLMs, such as VideoChat (Li et al., 2025a) and LLaVA-NeXT-Video (Zhang et al., 2024), have demonstrated impressive capabilities in comprehending temporally bounded, offline video clips. However, real-world applications often require interacting with continuous, infinite visual inputs on the fly. This necessity has driven a recent paradigm shift towards Streaming Video Large Language Models (Streaming VLMs), which are designed to process newly arriving visual streams in real time and maintain long-term context, as exemplified by StreamingVLM (Xu et al., 2026).

![](images/ca170ec8ca4d949ae55060958edc97618024dae4e5a28a93560dd619403ba08e.jpg)  
Question (14.00s): Why were there explosive sparks just now? GT Response : After the wooden board was removed, the iron rods at the ends of the capacitor came into contact, causing a short circuit and producing a spark.  
Gemini-3.5-Flash: The explosive sparks seen are caused by a sudden, high-current short-circuit discharge of a highvoltage capacitor.

Figure 1: A perception failure in a high-dynamic video event. A wooden separator (highlighted in purple using SAM 3) is removed, allowing the metal rods to touch and produce sparks. Both models provide generic electrical explanations that could be inferred from the sparks alone, but fail to identify the brief sequence of separator removal and subsequent contact that triggers the discharge. This illustrates the challenge of capturing critical visual details in fast-paced events.

Despite this rapid progress, existing streaming evaluation benchmarks, such as OVO-Bench (Niu et al., 2025) and PhoStream (Lu et al., 2026b), emphasize temporal reasoning and response timing without explicitly targeting high-dynamic perception. This leaves a critical challenge encountered in real-world streaming applications: accurately perceiving and comprehending highly dynamic, fastpaced events under strict context window constraints. In practice, video streams exhibit a heterogeneous mixture of low- and high-dynamic scenes. Operating under limited token budgets, current models must balance the need for maintaining a long temporal history, preserving high spatial resolution, and capturing fine-grained temporal information. Such constraints motivate sparse frame sampling; for example, the online baselines evaluated in OVO-Bench operate at 1–2 frames per second (Niu et al., 2025). Consequently, when faced with rapidly changing high-frequency actions—such as sports, extreme gaming, or wildlife hunting—these models often miss crucial visua details, leading to severe perception failures (as illustrated in Fig. 1). While recent studies have in troduced offline benchmarks focusing on motion comprehension and high frame-per-second (FPS) perception (e.g., MotionBench, Hong et al., 2025; FPS-Bench, Choudhury et al., 2026), evaluating the capability to perceive high-dynamic events within a continuous, untrimmed streaming context remains an open problem.

To bridge this critical gap, we introduce FastBench, a novel benchmark specifically designed to evaluate the performance of streaming VLMs on high-dynamic real-world video streams. FastBench is constructed via a meticulously designed pipeline that synergizes VLM-driven generation, agentic expert verification, and rigorous human inspection. To ensure true temporal dependency, we introduce a low-FPS verification step that proactively filters out pseudo-dynamic questions. Furthermore, to overcome the inherent hallucination risks of VLMs in fast-paced scenarios, we propose a novel Vision Expert Verification mechanism. By autonomously invoking vision foundation models, the VLM verifies and refines its QA pairs against objective, text-based physical trajectories rather than error-prone visual features. Spanning diverse domains including sports, gaming, nature, and traffic, FastBench establishes a stringent evaluation standard for high-dynamic streaming perception.

Along with the benchmark, we present ProactiveFrame, a simple, training-free baseline framework. Diverging from passive uniform sampling, ProactiveFrame empowers the agent to dynamically adjust its temporal perception granularity via text prompts based on the ongoing streaming context. To balance the need for fine-grained perception and long-term memory under strict token budgets, we introduce a dual-tier context sliding window mechanism. Operating on a single, unified chronological stream, this design proactively buffers high-FPS chunks for fleeting events, which subse quently undergo a graceful temporal degradation into a sparse historical context. This hierarchical management elegantly resolves the trade-off, enabling continuous high-dynamic perception without fragmenting the temporal horizon. Evaluations on FastBench show that high-dynamic perception benefits from denser temporal sampling, but remains constrained by the trade-off between temporal granularity and historical context. ProactiveFrame improves over sparse uniform sampling, yet its gap to oracle-guided sampling highlights the challenge of autonomously identifying when finer temporal perception is needed.

![](images/1bf84ad8ce44c30c67c09ae2f7f52395fba83d4627cacf77ac0b6e96f59a11b6.jpg)  
(a) Temporal QA examples. The backward query at 17 s refers to evidence at 10–11 s; the forward query at 33 s is answered after observing evidence at 36–37 s.  
(b) Dataset distributions. Counts and percentages of 306 QA pairs across six capability categories (top) and eight domains (bottom).  
Figure 2: Overview of FastBench. Evidence Start and Evidence End in (a) mark the humanannotated interval containing the visual evidence required to answer each question.

In summary, our main contributions are three-fold:

• High-Dynamic Streaming Benchmark: We introduce FastBench, a benchmark specifically designed to evaluate high-frame-rate-dependent perception in continuous video streams under bounded-context and response-timing constraints

• Trajectory-Grounded Data Pipeline: We propose an innovative data construction pipeline incorporating a low-FPS filtering mechanism to remove pseudo-dynamic questions, paired with a Vision Expert Verification module that grounds VLM annotations on objective physical trajectories to mitigate visual hallucinations.

• Training-Free Baseline Framework: We present ProactiveFrame, a simple yet effective training-free framework that features a dual-tier context sliding window mechanism. It dynamically buffers high-FPS chunks for fleeting events and allows them to gracefully degrade into sparse historical context, balancing fine-grained perception and long-term memory under tight token budgets.

## 2 RELATED WORK

## 2.1 STREAMING VIDEO UNDERSTANDING BENCHMARKS

Streaming video benchmarks evaluate understanding from incrementally available evidence. StreamingBench (Lin et al., 2026) assesses real-time visual, omni-source, and contextual understanding, while OVO-Bench (Niu et al., 2025) covers backward tracing, real-time understanding, and forward active responding. OVBench (Huang et al., 2025) evaluates online perception, memory, and reasoning across past, present, and future contexts. PhoStream (Lu et al., 2026b) extends evaluation to open-ended question answering in mobile scenarios, emphasizing audio-visual evidence and response timing. Building on these temporal task structures, FastBench focuses on high-framerate-dependent evidence in continuous streams, jointly evaluating fine-grained perception, historical retention, and response timing under a bounded visual context.

![](images/1034de4a6ee3e3579a9795de004cc9d52569a6f331e775bb056bdc6568420310.jpg)  
Figure 3: FastBench data construction pipeline. A VLM generates candidate QA pairs from short, high-frame-rate video clips. Questions answerable at 2 FPS are discarded. The remaining pairs undergo automated vision-expert verification, where the VLM uses SAM3 segmentation and CoTracker3 trajectories to verify or refine answers; pairs that cannot be verified or corrected are discarded. Retained QA pairs are mapped back to their source videos and undergo human quality checks to form FastBench. The lower panel illustrates how trajectory evidence supports correcting a punch count from two to three.

## 2.2 FINE-GRAINED MOTION AND HIGH-FRAME-RATE UNDERSTANDING

Fine-grained temporal benchmarks seek to distinguish motion understanding from reliance on static cues. TempCompass (Liu et al., 2024) uses videos with shared static content but contrasting temporal properties. MotionBench (Hong et al., 2025) evaluates fine-grained motion comprehension and investigates higher-frame-rate inputs and motion-preserving feature fusion. FPS-Bench (Choudhury et al., 2026) introduces minimum frames-per-second (minFPS), using human annotations to identify the lowest sampling rate needed to verify an answer. Complementing these efforts, FastBench uses low-FPS answerability filtering rather than per-question minFPS estimation to evaluate high-frame rate-dependent evidence within a causal streaming protocol.

## 2.3 PROACTIVE PERCEPTION AND STREAMING CONTEXT MANAGEMENT

Streaming systems must coordinate response timing, visual computation, and context retention. VideoLLM-online (Chen et al., 2024) learns from video streams to support temporally aligned dialogue. LION-FS (Li et al., 2025b) separates fast response determination from detail-enhanced generation, while StreamMind (Ding et al., 2025) uses event-gated LLM invocation to decouple perception from language reasoning. VideoChat3 (Li et al., 2026) adapts the spatial resolution of subsequent windows using predicted interaction states. For context management, StreamingVLM (Xu et al., 2026) maintains different KV-cache retention windows for visual and textual tokens, while simpleStream (Shen et al., 2026) demonstrates a strong recent-frame baseline and a perception– memory trade-off. In contrast, ProactiveFrame provides a training-free, prompt-driven baseline that adjusts future temporal sampling rates and uses a dual-tier window to retain recent high-FPS observations alongside sparse historical context.

## 3 FASTBENCH

In this section, we present the design of FastBench and our baseline framework, ProactiveFrame. We first detail the data curation pipeline in Section 3.1 and present the dataset statistics in Section 3.2. Finally, Section 3.3 describes the framework and core dynamic perception mechanisms of ProactiveFrame.

## 3.1 DATA CURATION

Raw Video Collection. To construct a diverse pool of candidate videos rich in high-dynamic, fast-paced events across domains such as sports, gaming, nature, and traffic, we introduce an automated, metadata-driven collection pipeline. We first employ a VLM, Gemini-3.1-Pro (Google DeepMind, 2026a), to brainstorm and generate a comprehensive set of search keywords corresponding to our target high-dynamic scenarios. Using these queries, we retrieve raw, untrimmed videos from popular video-sharing platforms, including YouTube and Bilibili. A metadata filter then uses timestamped comments and danmaku to localize high-dynamic segments within the retrieved videos (Appendix A.5).

Task Definition. We formulate FastBench as a Streaming Video Question Answering (QA) task over an untrimmed, continuous video stream V arriving online. At a query timestamp $t _ { q } ,$ a user question Q is presented, requiring evidence anchored within a temporal window $[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } ]$ . Following the temporal scopes considered in streaming benchmarks (Niu et al., 2025; Lu et al., 2026b), we distinguish three cases: Backward $( t _ { \mathrm { e n d } } < t _ { q } )$ , Instant $( t _ { \mathrm { s t a r t } } \leq t _ { q } \leq t _ { \mathrm { e n d } } )$ , and Forward $( t _ { \mathrm { s t a r t } } > t _ { q } )$ Fig. 2(a) illustrates two of these scopes: the backward example requires recalling whether an F1 tire contacted a track feature before the query, whereas the forward example requires observing and counting a kingfisher’s wing flaps after the query.

FastBench explicitly focuses on the unique difficulties of high-dynamic real-world scenarios. Specifically, the benchmark introduces two primary challenges inherent to continuous perception. First, it requires a significantly finer temporal granularity, demanding models to accurately comprehend rapidly evolving visual information across coherent continuous frames rather than isolated snapshots. Second, it highlights a critical trade-off between retaining a long-term historical context and processing high-frequency visual details. Under a limited context token budget, simultaneously accommodating both aspects becomes a severe bottleneck. Given the continuous visual input and query Q, the objective of the model is to effectively navigate this context-granularity trade-off to generate an accurate answer A.

Data Construction Pipeline We designed a comprehensive data construction pipeline that synergizes VLM-driven generation, agentic expert verification, and rigorous human inspection to extract and annotate high-quality question-answering pairs from high-dynamic video streams, as illustrated in Fig. 3. Initially, raw high-resolution videos are segmented into short clips with a duration of no more than 5 seconds. For each segment, we maintain a multi-stage offline conversation with a Vision-Language Model to automate the annotation process. In this initial phase, the VLM analyzes the native high-frame-rate video clip and proposes a candidate high-FPS QA pair.

To ensure the generated questions genuinely require continuous perception of high-dynamic events rather than static scene understanding, we introduce a low-FPS verification mechanism. The video clip is temporally sub-sampled at a sparse rate of 2 FPS and fed back into the VLM along with the candidate question. If the model can correctly answer the question relying solely on this low-FPS clip, the question is identified as pseudo-dynamic and is immediately dropped from the pipeline.

Questions that successfully pass the low-FPS filter then enter the Vision Expert Verification stage. The VLM first performs temporal grounding to isolate the target action interval and selects the initial frame of this segment. It then generates a text query for the target object, which is processed by SAM3 (Carion et al., 2026) to produce a segmentation mask. CoTracker3 (Karaev et al., 2025) tracks the resulting points over the video segment and outputs their trajectory coordinates. The VLM verifies or corrects the original answer from these trajectories. Unverifiable or uncorrectable QA pairs are discarded, while the successful ones advance as double-verified QA pairs.

Human Verification Double-verified QA pairs are mapped back to their longer source videos and reviewed by ten experts in multimodal video understanding. Each sample undergoes three round of inspection comparing the original high-FPS video with its 2-FPS subsample to verify: (1) answer correctness; (2) clear, unambiguous visual evidence and question targets; and (3) dependence on high-dynamic visual information, ruling out questions answerable at 2 FPS or without visual context. Experts also annotate the evidence interval and specific timestamps requiring high-FPS perception. Fig. 2(a) illustrates these intervals with Evidence Start and Evidence End markers. Only samples passing all three rounds are retained in FastBench.

![](images/86586af7a716797130529604258a49e055b091dad2f2cbf2d98f105130c5a746.jpg)  
Figure 4: Overview of ProactiveFrame. (a) The model switches between low- and high-FPS sampling through <Focus Start> and <Focus End>; <Silent> preserves the current mode. (b) A 120-frame General Window includes a 90-frame Focus Window. When a new high-FPS chunk arrives, the oldest focused chunk is downsampled into sparse history, and the oldest historical chunk is evicted to maintain the budget. Each chunk spans one second; F denotes retained frames.

## 3.2 DATA STATISTICS

In this section, we present the overall statistics of the FastBench dataset. We first detail the multistage data annotation and filtering process. Subsequently, we analyze the dataset’s diversity and comprehensively evaluate its distribution across different capability dimensions and real-world domains. The overall statistical distributions for both capabilities and domains are summarized in Fig. 2(b).

Data Annotation. To construct FastBench, we employed Gemini-3.1-Pro to generate 89 search keywords for high-dynamic scenarios. For each keyword, a VLM semantically filtered the metadata of the top 200 videos ranked by popularity. This pipeline yielded 766 source videos, segmented into 33,619 clips of up to 5 seconds. The VLM then generated 16,702 candidate question-answering (QA) pairs from these high-frame-rate clips. To ensure temporal dependency and mitigate visual hallucinations, low-FPS filtering and Vision Expert Verification reduced the pool to 3,102 double verified QAs. Finally, multi-round human inspections for answer correctness, visual clarity, and high-dynamic relevance yielded 306 high-quality samples.

Capability Taxonomy and Distribution. FastBench categorizes its 306 QA pairs into six capability dimensions (Fig. 2(b)): (1) Action & Physical Interaction (51.96%) covers fine-grained action recognition, physical contact detection, and rapid event counting; (2) Predictive & Causal Reasoning (14.38%) evaluates action anticipation and causal inference; (3) Motion & Spatiotemporal Tracking (13.40%) tests spatial reasoning and trajectory tracking under rapid motion; (4) Entity & Visual Perception (9.48%) assesses object and text recognition under motion blur; (5) Temporal & State Dynamics (8.17%) covers temporal sequencing and sudden state changes; and (6) Streaming & Online Detection (2.61%) targets real-time localization of action boundaries. Together, these dimensions assess fine-grained perception and higher-level reasoning under streaming constraints.

Domain Coverage and Diversity. FastBench spans eight domains featuring rapid motion and high visual dynamics: (1) Sports (29.08%), including motorsports, combat sports, and skateboarding; (2) Video Games (18.63%), featuring fast-paced action and esports gameplay; (3) Performing Arts (16.01%), including sleight-of-hand magic, card manipulation, and stage music; (4) Animals (12.75%), covering swift wildlife movements and hunting; (5) Lifestyle & Recreation (8.50%), encompassing acrobatic stunts, pen spinning, and active hobbies; (6) Transportation (8.17%), involving driving, drone operations, and aviation; (7) Science & Technology (3.59%), focusing on high

Table 1: Main evaluation results on FastBench. Capabilities are ordered from left to right by sample ratio in descending order. Scores are presented in percentage (%). Values in green or red parentheses indicate gains or drops relative to their respective base models.
<table><tr><td rowspan="2">Model</td><td colspan="6">Capability Breakdown (by Sample Ratio ↓)</td><td colspan="3">Temporal Scope</td><td rowspan="2">Overall</td></tr><tr><td>Action &amp; Phys. Inter. (51.96%)</td><td>Predictive &amp; Causal (14.38%)</td><td>Motion &amp; Spatiotemp. (13.40%)</td><td>Entity &amp; Visual (9.48%)</td><td>Temporal &amp; Dynamics (8.17%)</td><td>Streaming &amp; Online (2.61%)</td><td>Forward (54.90%)</td><td>Instant (23.86%)</td><td>Backward (21.24%)</td></tr><tr><td>Closed-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-Lite(ByteDance Seed, 2026)</td><td>25.0</td><td>51.4</td><td>16.6</td><td>35.2</td><td>30.4</td><td>42.5</td><td>33.2</td><td>24.9</td><td>25.2</td><td>29.5</td></tr><tr><td>GPT-5.6-Sol(OpenAI, 2026)</td><td>42.6</td><td>56.8</td><td>57.6</td><td>45.5</td><td>43.2</td><td>30.0</td><td>45.1</td><td>51.8</td><td>44.9</td><td>46.7</td></tr><tr><td>Gemini-3.5-Flash(Google DeepMind, 2026b)</td><td>43.6</td><td>67.3</td><td>62.9</td><td>44.8</td><td>52.0</td><td>52.5</td><td>52.7</td><td>48.5</td><td>47.7</td><td>50.7</td></tr><tr><td>Open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aura (8B)(Lu et al., 2026a)</td><td>31.1</td><td>32.3</td><td>47.8</td><td>21.4</td><td>28.8</td><td>20.0</td><td>24.9</td><td>41.4</td><td>40.3</td><td>32.1</td></tr><tr><td>Qwen3-VL (8B)(Bai et al., 2025)</td><td>30.6</td><td>34.1</td><td>47.3</td><td>32.4</td><td>32.0</td><td>2.5</td><td>33.9</td><td>34.5</td><td>28.3</td><td>32.9</td></tr><tr><td>MOSS-VL-Realtime (11B)(Wang et al., 2026)</td><td>27.3</td><td>34.5</td><td>27.8</td><td>16.6</td><td>31.2</td><td>35.0</td><td>36.1</td><td>22.2</td><td>13.2</td><td>27.9</td></tr><tr><td>JoyAI-VL-Interaction (8B)(Yao et al., 2026)</td><td>41.9</td><td>48.6</td><td>49.8</td><td>34.5</td><td>43.2</td><td>70.0</td><td>44.2</td><td>43.6</td><td>44.3</td><td>44.1</td></tr><tr><td>Proactive Perception Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoChat3 (4B)(Li et al., 2026)</td><td>15.7</td><td>29.5</td><td>18.5</td><td>19.3</td><td>26.4</td><td>67.5</td><td>37.6</td><td>0.0</td><td>0.0</td><td>20.7</td></tr><tr><td>ProactiveFrame (Qwen3-VL 8B)</td><td>36.1 (+5.5)</td><td>42.7 (+8.6)</td><td>47.3 (0.0)</td><td>33.8 (+1.4)</td><td>34.4 (+2.4)</td><td>40.0 (+37.5)</td><td>43.0 (+9.1)</td><td>32.3 (-2.2)</td><td>32.9 (+4.6)</td><td>38.3 (+5.4)</td></tr><tr><td>ProactiveFrame (Gemini-3.5-Flash)</td><td>47.0 (+3.4)</td><td>61.4 (-5.9)</td><td>56.1 (-6.8)</td><td>56.6 (+11.8)</td><td>54.4 (+2.4)</td><td>62.5 (+10.0)</td><td>53.6 (+0.9)</td><td>48.5 (0.0)</td><td>52.9 (+5.2)</td><td>52.2 (+1.5)</td></tr></table>

speed physics experiments; and (8) Food & Cooking (3.27%), covering rapid culinary preparation.   
This coverage supports evaluation across varied motion semantics and visual environments.

## 3.3 PROACTIVE FRAME

To resolve the conflict between retaining long-term history and capturing fine-grained temporal details under strict context budgets, we introduce ProactiveFrame, a simple yet effective training-free baseline that enables streaming VLMs to autonomously control their future temporal perception granularity. Instead of passive uniform sampling, the model is guided via text prompts to dynami cally switch the frame rate of the incoming stream based on the ongoing streaming context.

Under a fixed context window (e.g., 120 frames), a high frame rate severely compresses the retained history: a 2 FPS stream preserves 60 seconds (sixty 1-second chunks), whereas 30 FPS shrinks the temporal receptive field to merely 4 seconds. To preserve sufficient history while enabling highdynamic perception, we design a dual-tier context sliding window. As illustrated in Fig. 4, assuming a default 1080p resolution, the General Window holds a constant 120 frames, within which a highertier Focus Window buffers high-FPS chunks for fleeting events. Its size is adjustable; for example, 90 frames accommodate three consecutive high-FPS (e.g., native 30 FPS) chunks.

When the accumulated high-FPS frames exceed the Focus Window’s capacity, the earliest chunk is evicted following a First-In-First-Out (FIFO) principle. Crucially, rather than being discarded, it undergoes graceful temporal degradation: the chunk is down-sampled to a standard low-FPS chunk (e.g., 2 FPS) and enters the lower-tier General Window. This hierarchical management retains a sparse yet continuous long-term history without sacrificing the ability to proactively perceive tran sient, high-dynamic events.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETTING

Streaming Evaluation Protocol. We follow the streaming inference framework introduced by PhoStream Lu et al. (2026b), where videos are streamed to the model in 1-second chunks. At each step, the model must decide whether to continue observing or generate a final response. To achieve this, for general (non-streaming) models, we employ explicit text prompts instructing them to output a ”Silent” word if they require more temporal context to answer the query. Conversely, for native streaming models, we preserve their inherent interaction paradigms, evaluating them using their default, model-specific wait tokens to ensure fair and optimal performance. The quality of the final open-ended answers is assessed using an LLM-as-a-Judge.

Evaluation Criteria Early and late responses are assigned a score of 0. However, high-dynamic real-world streams present a unique challenge: for tasks like short-term counting, models typically cannot react at the exact millisecond an event ends. Instead, they require a brief subsequent observation period to confirm the transition into a static state and ensure the action has fully concluded. To accommodate this natural reasoning process, we adapt the strict evaluation criteria by introducing a relaxed late-response boundary, extending the valid temporal window to 4 seconds after the event concludes.

Context Window Constraints We define a unified maximum context budget of 248,832,000 pixels, which is equivalent to the information capacity of 120 frames at a 1080p resolution. For our proposed ProactiveFrame, we configure the focus window to 90 frames (at default 1080p resolution) to buffer high-FPS visual chunks, reserving the remaining capacity for at least 30 low-FPS frames to retain a continuous, sparse historical context.

## 4.2 MAIN RESULTS

Table 1 presents the main evaluation results on FastBench, revealing a noticeable performance gap between closed-source and open-source models in handling high-dynamic streaming scenarios. Among the evaluated models, Gemini-3.5-Flash achieves the highest overall accuracy within the closed-source category at 50.7%. In the open-source category, JoyAI-VL-Interaction (8B) leads the performance with an overall score of 44.1%. Other open-source models demonstrate more modest capabilities on this challenging benchmark, with Qwen3-VL (8B) scoring 32.9%, Aura achieving 32.1%, and MOSS-VL-Realtime (11B) yielding an overall score of 27.9%.

Within the scope of proactive perception methods, VideoChat3 (4B) yields a relatively low overall score of 20.7%. This underperformance can likely be attributed to the model’s primary focus on learning to dynamically control the spatial dimension rather than the temporal dimension. Furthermore, its limited 4B parameter capacity inherently restricts its ability to adequately process and reason over highly dynamic, fast-paced visual events.

In contrast, our proposed zero-shot approach, ProactiveFrame, demonstrates substantial effectiveness. Integrating ProactiveFrame with the Qwen3-VL (8B) baseline significantly improves the model’s overall performance to 38.3%, representing a notable +5.4% overall increase. This en hancement is driven by massive gains in specific capability breakdowns, highlighted by a remarkable +37.5% surge in the “Streaming & Online” category and an +8.6% increase in “Predictive & Causal” reasoning. Furthermore, combining Gemini-3.5-Flash with our proactive perception method surpasses all other evaluated models, achieving a peak overall score of 52.2%. This configuration also yields impressive sub-category gains, most notably a +11.8% jump in “Entity & Visual” perception and a +10.0% boost in “Streaming & Online” tasks.

However, performance drops are also observed in certain sub-categories—for instance, Gemini 3.5-Flash experiences declines in “Motion & Spatiotemp.” (-6.8%) and “Predictive & Causal” (- 5.9%), while Qwen3-VL exhibits a minor decrease in the “Instant” temporal scope (-2.2%). This suggests that dynamic temporal resampling can occasionally introduce frame-alignment noise or context redundancy during complex spatio-temporal reasoning. Overall, while these results strongly underscore the immense potential of empowering models with autonomous temporal control, they also reveal that current VLMs’ proactive perception mechanisms still have substantial room for improvement in precision and stability.

## 4.3 ABLATION ANALYSIS

Temporal Context Formulation To examine how streaming models utilize temporal context, Table 2 compares three context formulations: (1) Full 30 FPS, processing the stream at its native frame rate; (2) Mixed FPS (Focus), an oracle that provides 30 FPS chunks only within human-annotated focus intervals $( [ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } ] )$ and 2 FPS otherwise; and (3) ProactiveFrame, where the model triggers focus sampling autonomously from the visual stream. ProactiveFrame improves over the base models in Table 1 (38.3% vs. 32.9% for Qwen3-VL (8B) and 52.2% vs. 50.7% for Gemini-3.5-Flash), but a gap remains to the Mixed FPS oracle (40.7% and 57.5%), indicating that current VLMs still struggle to trigger focus mode at optimal timestamps.

Intriguingly, for Qwen3-VL (8B), Full 30 FPS (42.1%) surpasses the oracle Mixed FPS (40.7%). This reveals a bottleneck in current open-source VLMs: even with perfectly isolated high-FPS focus segments, their fine-grained dynamic understanding remains limited. Consequently, the visual context lost by downsampling background frames outweighs the benefit of longer historical memory.

Temporal scopes further exhibit distinct patterns. On Forward queries, Full 30 FPS exceeds Mixed FPS for both models (56.3% vs. 55.4% on Gemini-3.5-Flash and 42.0% vs. 39.4% on Qwen3-

Table 2: Ablation study on temporal context formulation using Qwen3-VL (8B) and Gemini-3.5- Flash on FastBench. Mixed FPS (Focus) provides original 30 FPS video chunks during the annotated focus timestamps and 2 FPS video chunks otherwise.
<table><tr><td rowspan="3">Method</td><td colspan="6">Capability Breakdown (by Sample Ratio ↓)</td><td colspan="3">Temporal Scope</td><td rowspan="3">Overall</td></tr><tr><td>Action &amp; Phys. Inter.</td><td>Predictive &amp; Causal</td><td>Motion &amp;</td><td>Entity &amp;</td><td>Temporal &amp;</td><td>Streaming &amp;</td><td>Forward</td><td>Instant</td><td>Backward</td></tr><tr><td>(51.96%)</td><td>(14.38%)</td><td>Spatiotemp. (13.40%)</td><td>Visual (9.48%)</td><td>Dynamics (8.17%)</td><td>Online (2.61%)</td><td>(54.90%)</td><td>(23.86%)</td><td>(21.24%)</td></tr><tr><td colspan="10">Qwen3-VL (8B)</td></tr><tr><td>Full 30 FPS</td><td>41.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>36.1</td><td>40.9 42.7</td><td>57.6 47.3</td><td>37.2 33.8</td><td>42.4 34.4</td><td>7.5 40.0</td><td>42.0 43.0</td><td>46.3 32.3</td><td>37.5 32.9</td><td>42.1 38.3</td></tr><tr><td>Proactive Frame Mixed FPS (Focus)</td><td>41.6</td><td>41.4</td><td>49.3</td><td>35.2</td><td>36.8</td><td>7.5</td><td>39.4</td><td>43.8</td><td>40.6</td><td>40.7</td></tr><tr><td colspan="10">Gemini-3.5-Flash</td></tr><tr><td>Full 30 FPS</td><td>49.4</td><td>61.4</td><td>65.9</td><td>60.0</td><td>56.8</td><td>62.5</td><td>56.3</td><td>58.1</td><td>49.5</td><td>55.3</td></tr><tr><td>Proactive Frame</td><td>47.0</td><td>61.4</td><td>56.1</td><td>56.6</td><td>54.4</td><td>62.5</td><td>53.6</td><td>48.5</td><td>52.9</td><td>52.2</td></tr><tr><td>Mixed FPS (Focus)</td><td>49.2</td><td>65.0</td><td>69.8</td><td>71.0</td><td>60.0</td><td>60.0</td><td>55.4</td><td>64.1</td><td>55.4</td><td>57.5</td></tr></table>

Table 3: Ablation study on input video frame rate (FPS) using Qwen3-VL (8B) on FastBench. Capabilities are ordered from left to right by sample ratio in descending order. Scores are presented in percentage (%). Values in green or red parentheses indicate performance gains or drops relative to the 2 FPS baseline.
<table><tr><td rowspan="2">FPS</td><td colspan="6">Capability Breakdown (by Sample Ratio ↓)</td><td colspan="3">Temporal Scope</td><td rowspan="2">Overall</td></tr><tr><td>Action &amp; Phys. Inter. (51.96%)</td><td>Predictive &amp; Causal (14.38%)</td><td>Motion &amp; Spatiotemp. (13.40%)</td><td>Entity &amp; Visual (9.48%)</td><td>Temporal &amp; Dynamics (8.17%)</td><td>Streaming &amp; Online (2.61%)</td><td>Forward (54.90%)</td><td>Instant (23.86%)</td><td>Backward (21.24%)</td></tr><tr><td>2 FPS</td><td>30.6</td><td>34.1</td><td>47.3</td><td>32.4</td><td>32.0</td><td>2.5</td><td>33.9</td><td>34.5</td><td>28.3</td><td>32.9</td></tr><tr><td>4FPS</td><td>37.0 (+6.4)</td><td>40.5 (+6.4)</td><td>50.2 (+2.9)</td><td>35.2 (+2.8)</td><td>41.6 (+9.6)</td><td>2.5 (0.0)</td><td>37.4 (+3.5)</td><td>41.1 (+6.6)</td><td>38.8 (+10.5)</td><td>38.6 (+5.7)</td></tr><tr><td>8FPS</td><td>40.3 (+9.7)</td><td>38.6 (+4.5)</td><td>54.1 (+6.8)</td><td>33.8 (+1.4)</td><td>43.2 (+11.2)</td><td>0.0 (-2.5)</td><td>38.1 (+4.2)</td><td>44.1 (+9.6)</td><td>42.5 (+14.2)</td><td>40.5 (+7.6)</td></tr><tr><td>16 FPS</td><td>43.0(+12.4)</td><td>44.5 (+10.4)</td><td>59.0 (+11.7)</td><td>40.0 (+7.6)</td><td>40.0 (+8.0)</td><td>2.5 (0.0)</td><td>41.8 (+7.9)</td><td>45.8 (+11.3)</td><td>46.8 (+18.5)</td><td>43.8 (+10.9)</td></tr><tr><td>24FPS</td><td>42.9 (+12.3)</td><td>48.6 (+14.5)</td><td>61.0 (+13.7)</td><td>39.3 (+6.9)</td><td>41.6 (+9.6)</td><td>2.5 (0.0)</td><td>43.5 (+9.6)</td><td>46.0 (+11.5)</td><td>46.2 (+17.9)</td><td>44.6 (+11.7)</td></tr><tr><td>30 FPS</td><td>41.0 (+10.4)</td><td>40.9 (+6.8)</td><td>57.6 (+10.3)</td><td>37.2 (+4.8)</td><td>42.4 (+10.4)</td><td>7.5 (+5.0)</td><td>42.0(+8.1)</td><td>46.3 (+11.8)</td><td>37.5 (+9.2)</td><td>42.1 (+9.2)</td></tr></table>

VL). Gemini-3.5-Flash peaks under Full 30 FPS, whereas Qwen3-VL peaks under ProactiveFrame (43.0%). Backward queries peak under the oracle Mixed FPS for both models (55.4% and 40.6%). This aligns with task requirements: Forward queries observe events immediately after the query and therefore benefit from dense frames, while Backward queries must retrieve past evidence and are thus sensitive to the memory–resolution trade-off that the oracle balances best.

Input Video Frame Rate To analyze the effect of temporal granularity, we evaluate Qwen3-VL (8B) at input frame rates from 2 to 30 FPS under a unified context budget (Table 3). Performance scales monotonically from the 2 FPS baseline (32.9%) up to 24 FPS, peaking at 44.6% (+11.7%), confirming that dense sampling is essential for capturing rapid state transitions and fleeting motion cues.

However, native 30 FPS slightly degrades performance to 42.1%. This saturation reflects a core trade-off in context-constrained streaming VLMs: maximal frame rate severely compresses the historical receptive field and introduces substantial inter-frame redundancy, so 24 FPS offers the best balance between context depth and high-frequency detail.

Capability and scope breakdowns support this observation. Motion-intensive tasks such as Motion & Spatiotemporal Tracking (+13.7%) and Predictive & Causal Reasoning (+14.5%) benefit most, peaking at 24 FPS. Meanwhile, the drop at 30 FPS is most pronounced for Backward queries (46.2% at 24 FPS to 37.5%), since their reliance on historical evidence suffers from the shortened temporal window, whereas Instant queries remain strong at 30 FPS (46.3%), depending mainly on immediate frame resolution.

## 5 CONCLUSION

We introduced FastBench, a benchmark for evaluating streaming VLMs on high-dynamic real-world streams, built with a trajectory-grounded pipeline that filters pseudo-dynamic questions and verifies answers against SAM3 and CoTracker3 trajectories before multi-round human inspection. We also presented ProactiveFrame, a training-free baseline that adjusts future frame rates via text tokens and retains history through a dual-tier sliding window. Experiments show that high-dynamic perception remains largely unsolved: the best model reaches only 50.7%, denser sampling helps but is bounded by the granularity–history trade-off, and ProactiveFrame improves over sparse sampling yet trails oracle-guided focusing, indicating that deciding when to look closer is the central open challenge. We hope FastBench and its annotated focus timestamps foster future research on proactive temporal control in streaming video understanding.

## AI USE STATEMENT

In this work, we used Gemini-3.1-Pro (Google DeepMind, 2026a) to construct the FastBench question-answer annotations in Figure 3. It proposed the search keywords used to collect highdynamic source videos, and then annotated each short high-frame-rate clip: it proposed a candidate question-answer pair, discarded the pair when the question was answerable from a 2 FPS subsample, and otherwise selected the action interval, queried SAM3 for a segmentation mask, and verified or revised the answer against CoTracker3 trajectories. Pairs that could not be verified or corrected were discarded. Gemini-3.5-Flash (Google DeepMind, 2026b) was used to read timestamped comments and danmaku when localizing candidate segments. Every pair retained in the benchmark was afterwards accepted or rejected by human experts. We also used an LLM-as-a-judge to score open-ended answers, following the same protocol as PhoStream (Lu et al., 2026b). We have not used generative AI tools to develop theoretical models, formulate or prove mathematical claims, propose hypotheses, design the method, or translate text. We reviewed the model-generated annotations against the source videos and the trajectory evidence, and human experts made the final decision for every released sample. We take responsibility for the final content of this work, including the annotations and evaluation judgments produced with the aid of generative AI.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/abs/2511.21631.

ByteDance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity. Model card, February 2026. URL https://seed.bytedance.com/en/seed2.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts. In International Conference on Learning Representations, volume 2026, pp. 138846–138923, 2026.

Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. VideoLLM-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18407–18418, June 2024.

Rohan Choudhury, Jean-Sebastien Dandurand, Kai Qiu, Kshitij Madhav Bhat, Kartik Sharma, Liza Dahiya, Yizhou Zhao, Souraja Kundu, Chun-Hsien Lin, Kris M. Kitani, and Laszl´ o A. Jeni. FPS-´ Bench: A benchmark for high frame-rate video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18598–18608, June 2026.

Xin Ding, Hao Wu, Yifan Yang, Shiqi Jiang, Qianxi Zhang, Donglin Bai, Zhibo Chen, and Ting Cao. StreamMind: Unlocking full frame rate streaming video dialogue through event-gated cognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 13448–13459, October 2025.

Google DeepMind. Gemini 3.1 Pro model card, February 2026a. URL https://deepmind. google/models/model-cards/gemini-3-1-pro/.

Google DeepMind. Gemini 3.5 Flash model card, May 2026b. URL https://deepmind. google/models/model-cards/gemini-3-5-flash/.

Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, and Jie Tang. MotionBench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8450–8460, June 2025.

Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: Ovbench and videochat-online. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3328–3338. IEEE, 2025.

Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker3: Simpler and better point tracking by pseudo-labelling real videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6013– 6022, October 2025. URL https://openaccess.thecvf.com/content/ICCV2025/ html/Karaev\_CoTracker3\_Simpler\_and\_Better\_Point\_Tracking\_by\_ Pseudo-Labelling\_Real\_Videos\_ICCV\_2025\_paper.html.

Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. VideoChat: Chat-centric video understanding. Science China Information Sciences, 68(10):200102, 2025a. doi: 10.1007/s11432-024-4321-9. URL https://doi.org/10. 1007/s11432-024-4321-9.

Wei Li, Bing Hu, Rui Shao, Leyang Shen, and Liqiang Nie. Lion-fs: Fast & slow video-language thinker as online video assistant. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3240–3251. IEEE, 2025b.

Xinhao Li, Yuhan Zhu, Xiangyu Zeng, Yuhao Dong, Haoning Wu, Zhiqiu Zhang, Yuandong Yang, Changlian Ma, Qingyu Zhang, Yansong Shi, et al. VideoChat3: Fully open video MLLM for efficient and generalist video understanding. arXiv preprint arXiv:2607.14935, 2026. URL https://arxiv.org/abs/2607.14935.

Junming Lin, Zheng Fang, Chi Chen, Haoxuan Cheng, Zihao Wan, Fuwen Luo, Ziyue Wang, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 12147–12151. IEEE, 2026.

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings ofthe Association for Computational Linguistics: ACL 2024, pp. 8731–8772, 2024.

Xudong Lu, Yang Bo, Jinpeng Chen, Shuhan Li, Xintong Guo, Huankang Guan, Fang Liu, Dunyuan Xu, Peiwen Sun, Heyang Sun, Rui Liu, and Hongsheng Li. AURA: Always-on understanding and real-time assistance via video streams. arXiv preprint arXiv:2604.04184, 2026a. URL https: //arxiv.org/abs/2604.04184.

Xudong Lu, Huankang Guan, Yang Bo, Jinpeng Chen, Xintong Guo, Shuhan Li, Fang Liu, Peiwen Sun, Xueying Lee, Wei Zhang, Xue Yang, Rui Liu, and Hongsheng Li. PhoStream: Benchmarking real-world streaming for omnimodal assistants in mobile scenarios. In International Conference on Machine Learning, 2026b. URL https://icml.cc/virtual/2026/poster/ 62888.

Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, and Jiaqi Wang. OVO-Bench: How far is your Video-LLMs from real-world online video understanding? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18902–18913, June 2025.

OpenAI. GPT-5.6 Sol. OpenAI API documentation, July 9 2026. URL https://developers. openai.com/api/docs/models/gpt-5-6-sol.

Yujiao Shen, Shulin Tian, Jingkang Yang, and Ziwei Liu. A simple baseline for streaming video understanding. arXiv preprint arXiv:2604.02317, 2026.

Pengyu Wang, Chenkun Tan, Shaojun Zhou, Qirui Zhou, Yanxin Chen, Xingyang He, Huazheng Zeng, Jijun Cheng, Chenghao Wang, Xiaomeng Qian, et al. MOSS-VL technical report. arXiv preprint arXiv:2608.15045, 2026. URL https://arxiv.org/abs/2608.15045.

Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. StreamingVLM: Real-time understanding for infinite video streams. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 61463–61475, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 6445dd88ebb9a6a3afa0b126ad87fe41-Paper-Conference.pdf.

Dingyu Yao, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Xiangyu Zeng, Yifei Li, Haowen Hou, Zheming Liang, Congcong Wang, Kaiwen Tuo, et al. JoyAI-VL-Interaction: Real-time visionlanguage interaction intelligence. arXiv preprint arXiv:2606.14777, 2026. URL https:// arxiv.org/abs/2606.14777.

Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. LLaVA-NeXT: A strong zero-shot video understanding model. LLaVA official technical blog, April 2024. URL https://llava-vl.github.io/blog/ 2024-04-30-llava-next-video/.

## A APPENDIX

## A.1 MORE STATISTICS OF FASTBENCH

FastBench contains 306 QA pairs over 300 video clips: 294 clips contain one QA pair each, and six contain two. The clips have a cumulative duration of 3.77 hours, with a mean of 45.29 seconds and a range of 7.32–79.68 seconds. Figure A.1 summarizes the distributions of clip durations and query timestamps. Queries are issued 19.34 seconds after the start of a clip on average, with timestamps ranging from 0 to 74 seconds. These statistics complement the capability and domain distributions in Section 3.2 by characterizing the temporal context in which questions are presented.

![](images/29399cd9a328fcff49e458bb9c01d64f7413923ca72876837299e1129a67f5d9.jpg)

![](images/4382cc4c00b9890e50e26d3af6fc1fe332479d0cef9fe3a23e61e21f73451b68.jpg)  
Figure A.1: Additional temporal statistics of FastBench. (a) Duration distribution of the 300 video clips. (b) Query timestamp distribution of the 306 QA pairs, measured from the start of each clip. Both panels use 10-second bins; bar labels indicate counts and dashed lines mark the means.

## A.2 DETAILS OF HUMAN ANNOTATORS

The human verification in Section 3.1 was conducted by ten annotators with experience in multimodal video understanding, each holding at least a master’s degree. Annotators were compensated at \$20 per hour.

## A.3 MORE DETAILED EVALUATION RESULTS

We provide additional evaluation results across domains and capability–temporal scope combinations, using the same protocol and scoring scale as Table 1.

Results across domains. Table A.1 reports scores across the eight domains in FastBench, ordered by sample count. Each domain score is averaged over its constituent QA pairs, while Overall is averaged over all 306 QA pairs.

Capability-wise results by temporal scope. Table A.2 further separates each capability into Forward, Instant, and Backward subsets, complementing the aggregate capability and temporal scores in Table 1. All eight Streaming & Online Detection samples belong to Forward; its Instant and Backward subsets are empty.

## A.4 QUALITATIVE DATASET EXAMPLES

Figure A.2 shows four FastBench items whose answers depend on a brief visual change. Each row contains three frames from the evidence interval, with the query time, focus timestamp, and annotated answer.

## A.5 METADATA FILTERING

This subsection details the metadata filter that follows keyword retrieval in Section 3.1.

Identifying fleeting, high-dynamic moments within hours of raw video using pure visual processing is computationally prohibitive. However, we empirically observe a strong correlation between specific user engagement patterns and the occurrence of rapid, visually complex events. Notably, timestamped comments (e.g., “01:02 What just flashed by?”) and time-aligned bullet chats frequently highlight moments that capture intense audience attention, serving as natural indicators of high-dynamic scenes. Leveraging this insight, we formulate a VLM-assisted filtering mechanism. We feed the retrieved video metadata—specifically the comments and danmaku—into Gemini-3.5- Flash (Google DeepMind, 2026b). The VLM acts as an efficient semantic filter, analyzing these textual cues to not only determine the presence of high-dynamic content but also to accurately pinpoint the temporal coordinates of these target segments. This step efficiently prunes the video corpus and extracts precise temporal windows for the subsequent verification stages.

![](images/8dff86d18e677caa176c390acecc29a8481572bb8b38814167daf6d09d09b981.jpg)

![](images/36186548077796e40cb09e968cb5789edce0229dc89bf689d631a977dff815e8.jpg)

![](images/cb4e62ec3912869ccdf95246dc044bfa4050835e900f6d9fe7dc49f57a141832.jpg)

(a) Performing arts, forward (sample-58). Query and focus at 3 s; the answer is available at 5 s. Frames at 4.0 s, 4.4 s, and 5.0 s. Q: Look closely at this actor, what surprising move will he make next? A: He raises both arms, covers his face with his sleeves, and as the arms drop his mask and costume change from green to red.  
![](images/d513f4f9ea22bf3f02682d94ae653e73ad16a41c9e49ddfb17e138171b33a3ec.jpg)

![](images/38bddc22fc72f4e843ec35ada44167b284ecb3ea5c914a74d5fa7473a10098c0.jpg)

![](images/e9119f22e7a576e64ef75bb30540335ecb952baf30a872251cab99ebdfc8efac.jpg)

(b) Lifestyle and recreation, backward (sample-14). Focus at 16 s; query at 17 s. Frames at 16.0 s, 16.3 s, and 17.3 s. Q: The white ball is rolling toward the red ball. Will the red ball be knocked into the top-left pocket? A: Yes. The white ball strikes the red ball, which rolls left and drops into the top-left pocket.  
![](images/38df43326200714bd17cdc41f9ad18b68d4ea2eee34a5a06b796a51fe723e343.jpg)

![](images/7703ba87dcd9a29217420c3c449fa175513eb183c75ae9dc77c46af1c3b32ab0.jpg)

![](images/d955584caa26d8ed069a5ebc5919c5dd213e92fd11a91e1c0ce42e9be87a4c19.jpg)  
(c) Transportation, backward (sample-212). Focus at 7 s; query at 8 s. Frames at 7.0 s, 7.4 s, and 7.8 s. Q: At the frontal collision, what happens to the black windshield wiper, and which way does it fly off? A: It jolts loose, rotates, and flies off toward the right side of the screen.

![](images/238344b09e40dfb14404c4c450939077f05c91fd8f6603006f42fae0006f8cfe.jpg)

![](images/2beb58e5f3fa5c7024fc5b3268b6b0d051af845d60b4055a38cf21bdb74df007.jpg)

![](images/1192c4c351212e6fd343512816cb7c148ddc2fdfa5f0a3ea69c6415af5271570.jpg)  
(d) Science and technology, backward (sample-31). Focus at 64 s; query at 67 s. Frames at 64 s, 65 s, and 67 s. Q: Was the disc just rotating clockwise or counterclockwise? A: Clockwise. The spinning frames are motion-blurred and the last frame is already still, so the direction is not recoverable from these three images.  
Figure A.2: Qualitative examples from FastBench. Each row is one QA pair. Focus is the annotated timestamp that requires high-frame-rate perception; the query time is when the question is asked.

Table A.1: Evaluation results across domains. Scores use a 0–100 scale. Parentheses in the header indicate the number of QA pairs in each domain. Best displayed scores in each column are bolded.
<table><tr><td>Model</td><td>Sports (89)</td><td>Video Games (57)</td><td>Performing Arts (49)</td><td>Animals (39)</td><td>Lifestyle &amp; Recreation (26)</td><td>Transpor- tation (25)</td><td>Science &amp; Technology (11)</td><td>Food &amp; Cooking (10)</td><td>Overall</td></tr><tr><td colspan="10">Closed-source Models</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>38.4</td><td>28.1</td><td>20.4</td><td>28.2</td><td>15.4</td><td>41.6</td><td>23.6</td><td>22.0</td><td>29.5</td></tr><tr><td>GPT-5.6-Sol</td><td>53.7</td><td>37.5</td><td>44.9</td><td>45.1</td><td>36.2</td><td>48.0</td><td>65.5</td><td>54.0</td><td>46.7</td></tr><tr><td>Gemini-3.5-Flash</td><td>56.4</td><td>43.2</td><td>49.8</td><td>52.3</td><td>46.2</td><td>49.6</td><td>65.5</td><td>38.0</td><td>50.7</td></tr><tr><td colspan="10">Open-source Models</td></tr><tr><td>Aura (8B)</td><td>36.0</td><td>29.1</td><td>29.0</td><td>27.7</td><td>40.0</td><td>28.0</td><td>41.8</td><td>26.0</td><td>32.1</td></tr><tr><td>Qwen3-VL (8B)</td><td>38.4</td><td>35.4</td><td>26.1</td><td>27.7</td><td>26.9</td><td>25.6</td><td>50.9</td><td>36.0</td><td>32.9</td></tr><tr><td>MOSS-VL-Realtime (11B)</td><td>28.5</td><td>25.3</td><td>29.8</td><td>31.3</td><td>24.6</td><td>23.2</td><td>27.3</td><td>36.0</td><td>27.9</td></tr><tr><td colspan="10">Proactive Perception Methods</td></tr><tr><td>VideoChat3 (4B)</td><td>21.8</td><td>21.1</td><td>13.9</td><td>30.3</td><td>13.1</td><td>21.6</td><td>27.3</td><td>14.0</td><td>20.7</td></tr><tr><td>ProactiveFrame (Qwen3-VL 8B)</td><td>40.4</td><td>35.8</td><td>37.1</td><td>47.7</td><td>36.2</td><td>29.6</td><td>41.8</td><td>26.0</td><td>38.3</td></tr><tr><td>ProactiveFrame (Gemini-3.5-Flash)</td><td>62.5</td><td>49.5</td><td>47.8</td><td>48.7</td><td>43.1</td><td>50.4</td><td>65.5</td><td>26.0</td><td>52.2</td></tr></table>

Figure A.3 gives the instruction used for this filter. The model must return a JSON match decision and, when the video matches, an approximate timestamp for the window passed to later verification.

## A.6 PROACTIVEFRAME INSTRUCTION

Figure A.4 shows the instruction used by ProactiveFrame during streaming inference (Section 3.3). On each new video chunk, the model answers the active question when the observed context is sufficient; otherwise it emits Focus Start or Focus End as the entire response to raise or restore the frame rate of the next chunk, or outputs Silent to keep the current mode.

Table A.2: Capability-wise evaluation results by temporal scope. Scores use a 0–100 scale. Fwd., Inst., and Back. denote Forward, Instant, and Backward, respectively. A dash indicates an empty subset. Best displayed scores in each column are bolded.
<table><tr><td rowspan="2">Model</td><td colspan="3">Action &amp; Physical Interaction</td><td colspan="3">Predictive &amp; Causal Reasoning</td><td colspan="3">Motion &amp; Spatiotemporal Tracking</td></tr><tr><td>Fwd.</td><td>Inst.</td><td>Back.</td><td>Fwd.</td><td>Inst.</td><td>Back.</td><td>Fwd.</td><td>Inst.</td><td>Back.</td></tr><tr><td>Closed-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>26.0</td><td>29.5</td><td>19.6</td><td>51.7</td><td>40.0</td><td>56.0</td><td>12.6</td><td>16.0</td><td>28.6</td></tr><tr><td>GPT-5.6-Sol</td><td>41.1</td><td>49.5</td><td>39.1</td><td>55.6</td><td>40.0</td><td>76.0</td><td>50.5</td><td>65.3</td><td>60.0</td></tr><tr><td>Gemini-3.5-Flash</td><td>43.8</td><td>48.0</td><td>39.6</td><td>69.4</td><td>26.7</td><td>76.0</td><td>60.0</td><td>60.0</td><td>77.1</td></tr><tr><td>Open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aura (8B)</td><td>23.0</td><td>41.0</td><td>35.2</td><td>28.9</td><td>20.0</td><td>64.0</td><td>40.0</td><td>48.0</td><td>68.6</td></tr><tr><td>Qwen3-VL (8B)</td><td>33.4</td><td>29.5</td><td>27.0</td><td>36.7</td><td>20.0</td><td>24.0</td><td>44.2</td><td>46.7</td><td>57.1</td></tr><tr><td>MOSS-VL-Realtime (11B)</td><td>37.0</td><td>22.0</td><td>16.5</td><td>36.7</td><td>40.0</td><td>16.0</td><td>34.7</td><td>30.7</td><td>2.9</td></tr><tr><td colspan="10">Proactive Perception Methods</td></tr><tr><td>VideoChat3 (4B) ProactiveFrame</td><td>34.2</td><td>0.0</td><td>0.0</td><td>36.1</td><td>0.0</td><td>0.0</td><td>40.0</td><td>0.0</td><td>0.0</td></tr><tr><td>(Qwen3-VL 8B)</td><td>44.9</td><td>28.5</td><td>28.7</td><td>45.0</td><td>40.0</td><td>28.0</td><td>46.3</td><td>37.3</td><td>71.4</td></tr><tr><td>ProactiveFrame (Gemini-3.5-Flash)</td><td>47.7</td><td>47.5</td><td>45.7</td><td>61.1</td><td>40.0</td><td>76.0</td><td>49.5</td><td>50.7</td><td>85.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Model</td></tr><tr><td rowspan="2"></td><td colspan="3">Visual Perception</td><td colspan="3">State Dynamics</td><td colspan="3">Streaming &amp; Online Detection</td></tr><tr><td></td><td>Fwd. Inst.</td><td>Back.</td><td>Fwd.</td><td>Inst.</td><td>Back.</td><td></td><td>Fwd. Inst.</td><td>Back.</td></tr><tr><td colspan="10">Closed-source Models Doubao-Seed-2.0-Lite</td></tr><tr><td></td><td>38.9</td><td>17.5</td><td>60.0</td><td>38.6</td><td>20.0</td><td>20.0</td><td>42.5</td><td></td><td></td></tr><tr><td>GPT-5.6-Sol</td><td>41.1</td><td>50.0</td><td>60.0</td><td>45.7</td><td>42.9</td><td>35.0</td><td></td><td>30.0</td><td></td></tr><tr><td>Gemini-3.5-Flash</td><td>51.1</td><td>25.0</td><td>60.0</td><td>48.6</td><td>62.9</td><td>45.0</td><td></td><td>52.5</td><td></td></tr><tr><td colspan="10">Open-source Models</td></tr><tr><td>Aura (8B)</td><td>13.3</td><td>40.0</td><td>20.0</td><td>21.4</td><td>40.0</td><td>35.0</td><td>20.0</td><td></td><td></td></tr><tr><td>Qwen3-VL (8B)</td><td>30.0</td><td>45.0</td><td>13.3</td><td>38.6</td><td>31.4</td><td>10.0</td><td></td><td>2.5</td><td></td></tr><tr><td>MOSS-VL-Realtime (11B)</td><td>24.4</td><td>5.0</td><td>0.0</td><td>47.1</td><td>17.1</td><td>0.0</td><td></td><td>35.0</td><td></td></tr><tr><td colspan="10">Proactive Perception Methods</td></tr><tr><td>VideoChat3 (4B)</td><td>31.1</td><td>0.0</td><td>0.0</td><td>47.1</td><td>0.0</td><td>0.0</td><td>67.5</td><td></td><td></td></tr><tr><td>ProactiveFrame (Qwen3-VL 8B)</td><td>33.3</td><td>40.0</td><td>20.0</td><td>37.1</td><td>31.4</td><td>30.0</td><td>40.0</td><td></td><td></td></tr><tr><td>ProactiveFrame (Gemini-3.5-Flash)</td><td>60.0</td><td>42.5</td><td>73.3</td><td>57.1</td><td>60.0</td><td>35.0</td><td></td><td>62.5</td><td></td></tr></table>

# Task   
Analyze the given video metadata (title, sampled danmaku, and   
comments) and determine whether the video contains a high-dynamic   
transient: a sudden switch from an ordinary, redundant state to a   
moment that requires high-FPS, frame-by-frame understanding (for   
example, an ultra-fast serve, bursting a balloon, an extreme dodge,   
or an exposed magic trick). If so, precisely localize the   
approximate time span in which this high-dynamic transient occurs.   
# Criteria (target scene characteristics)   
1. Abruptness: the video is mostly calm or shows only routine   
motion, but at one point an extremely fast and complex action   
happens suddenly.   
2. Low-FPS ambiguity: at a conventional low frame rate the moment   
is severely motion-blurred and can be resolved only frame by   
frame or in slow motion.   
# Signal Indicators (cues from multiple sources)   
- Title cues: words such as "extreme", "clutch", "sudden",   
"split-second capture", "instant", or "teleport".   
- Comment and danmaku behavior cues: mentions of "0.25x speed",   
"pause", "rewind", "couldn’t see it", "in a blink", or   
"couldn’t screenshot it".   
- Comment timestamps: explicit markers such as "01:15 watch this"   
or "the last 3 seconds".   
- Danmaku timing cues: a dense burst of "???", "so fast", or   
"what just happened".   
Filter the video according to the requirements above, and provide   
both the reasoning and the target segment time. The output must be   
valid JSON and contain exactly three fields:   
{"is\_match": <true|false>,   
"reasoning": "<at most 120 characters>",   
"target\_timestamp": "<MM:SS / HH:MM:SS / None>"}.   
If the time cannot be determined, output None. Do not output any   
additional text.

Figure A.3: Instruction for the metadata filter used before visual inspection. It reads the title, sampled danmaku, and comments, decides whether the video contains a sudden high-dynamic moment that requires frame-level perception, and returns an approximate timestamp of that moment.

You are an AI assistant for real-time video stream question   
answering. At any moment, there is at most one active question.   
Answer the active question as soon as the current video   
context|based on all frames observed so far and prior   
dialogue|contains enough information to support a complete and   
factually grounded answer.   
Stream input semantics:   
- New visual input is appended incrementally in chunks.   
- Each newly appended chunk covers at most 1 second of new   
video time.   
- The number of frames in a newly appended chunk can vary; do   
not assume a fixed frame count per update.   
- You can proactively request a temporary higher-frame-rate   
view when the current sampling rate may miss fine-grained   
temporal details.   
Follow these output rules exactly:   
- Focus control:   
1) Proactively monitor the video even when there is no active   
question. Output the exact token Focus\_Start whenever the   
current scene appears likely to contain an upcoming   
highlight, rapid motion, a brief event, a transition, an   
interaction, object manipulation, a subtle gesture, or any   
detail that could benefit from fine-grained temporal   
perception. This preserves richer evidence for possible   
future backward-looking or instant questions.   
2) Also output Focus\_Start when an active question requires   
closer temporal observation before it can be answered   
reliably. High-frame-rate sampling starts with the next   
video chunk.   
3) Use a low threshold for starting focus. When uncertain   
whether higher frame rate may help, prefer Focus\_Start;   
unnecessary focus is acceptable.   
4) While high-frame-rate sampling is active, inspect each new   
chunk and output the exact token Focus\_End once the   
interesting/high-motion/fine-grained interval has clearly   
ended and the scene has settled. Normal sampling resumes   
with the next video chunk. Until then, keep focus active   
by answering the active question if possible, or outputting   
Silent.   
5) Focus\_Start and Focus\_End must each be the entire response.   
Do not combine either token with an answer, Silent,   
punctuation, or explanation.   
6) Do not output Focus\_Start if high-frame-rate sampling is   
already active. Do not output Focus\_End unless it is   
active.   
- For multiple-choice questions (those that explicitly list   
options such as A, B, C, D):   
Respond only with the correct option letter (e.g., "A"). Do   
not add punctuation, explanation, or extra text.   
- For all other questions:   
Provide a complete answer using only information available in   
the stream up to this point.   
If the answer can be inferred with reasonable confidence,   
answer directly.   
- Output the exact word Silent only in one of these cases:   
1) There is no active question and neither Focus\_Start nor   
Focus\_End is appropriate.   
2) User explicitly asks to delay answering.   
3) Question unambiguously requires future content and no part   
can be determined yet.

Figure A.4: Instruction used by ProactiveFrame to control frame rate from the stream itself. Focus Start requests a higher frame rate for the next chunk, Focus End returns to normal sampling, and each must be the entire response; Silent continues the current mode when the model should not yet answer.