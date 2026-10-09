# From Video Clips to Creation Trajectory: Sora100K for AI-Native Video Creation

Sicong Yang<sup>1</sup>, Ruihuan Yang<sup>1</sup>, Jian Lu<sup>1</sup>, Jianfei Yuan<sup>1</sup>, Xiaodong Cun<sup>2</sup>, Xiuli Bi<sup>1</sup>

<sup>1</sup>Chongqing University of Posts and Telecommunications, Chongqing, China <sup>2</sup>GVC Lab, Great Bay University, Dongguan, China

![](images/741c849b17660fe8a676622ef9c6df9b983dc0ed7c66f8f4429746d34296162c.jpg)  
(b) Creation Trajectory Representation  
(c) Sora100K  
Figure 1: Overview of Sora100K. We represent the AI-Native video creation workflow as a video creation trajectory that connects the generation root, successive editing edges, and a final video. Based on this representation, Sora100K organizes diverse video creation trajectories into a structured video dataset while preserving source-to-edit lineage and editing order.

## Abstract

AI-Native video creation is shifting from isolated video clips toward iterative video creation workflows. However, existing datasets remain largely video clips, representing video generation and editing as separate tasks rather than connected stages of a video creation workflow. In this paper, we introduce Sora100K, a dataset that represents the AI-Native video creation workflow as a structured video creation trajectory. Specifically, we first identify video creation trajectories and decompose them into three subsets according to their structural roles: text-to-video generation records as roots, singleturn video editing records as editing edges, and multi-turn video editing records as complete trajectories. Then, we use a VLM to assign semantic annotations for generation roots and editing-operation annotations for editing edges. A strict construction pipeline further reconstructs source-to-edit lineage, editing order, and intermediate video states while ensuring data quality. Finally, we perform lightweight adaptation on LTX-2 models to assess the supervision value of Sora100K. The results show improvements in visual quality, multi-shot

generation, and cross-shot consistency, while successive-turn evaluation reveals that following multi-turn editing instructions remains challenging. Sora100K establishes a new data foundation for AI-Native video creation beyond isolated video clips and toward structured video creation trajectory. The dataset and supplementary materials are publicly available at https://huggingface.co/datasets/ysicong/Sora100K.

## Introduction

Commercial video generation models (OpenAI 2024; Google 2025; Runway 2025; Kling AI 2026; Seedance et al. 2026) continue to advance beyond short, isolated video clips toward cinematic videos with multi-shot structure and audiovisual synchronization. These advances are also changing how users create videos. Instead of relying on a single generation or editing step, users increasingly refine video content, visual style, camera motion, and audio through successive natural-language editing instructions. As illustrated in Figure 1, we represent this workflow from the initial prompt to the final edited video as a video creation trajectory. It connects the initial generation with successive editing operations and intermediate video states.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Structure</td><td colspan="2">Modality</td></tr><tr><td>Root 6</td><td>Edge 一</td><td>Trajectory E</td><td>Multi-shot 回</td><td>Audio 4</td></tr><tr><td>ActivityNet (CVPR&#x27;15)</td><td>√</td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>MSR-VTT (CVPR&#x27;16)</td><td>√</td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>WebVid-10M (ICCV&#x27;21)</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>HD-VILA-100M (CVPR’22)</td><td>√</td><td>×</td><td>×</td><td>√</td><td>×</td></tr><tr><td>Panda-70M (CVPR&#x27;24)</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Koala-36M (CVPR&#x27;25)</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>InstructVid2Vid (ICME&#x27;24)</td><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>VIVID-10M (arXiv&#x27;24)</td><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Señorita-2M (NeurIPS&#x27;25)</td><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>OpenVE-3M (arXiv&#x27;25)</td><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Sora100K (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Structural and modality comparison of representative video datasets. Root and Edge denote text-to-video generation records and single-turn editing records, respectively, while Trajectory indicates whether they are connected through editing order and intermediate video states. Multishot and Audio specify the presence of multi-scene transitions and aligned audio tracks.

However, as illustrated in Table 1, existing open academic datasets remained largely video clips. Text-to-video datasets (Wang et al. 2025; Chen et al. 2024) stored independent text–video pairs, while editing datasets (Qin et al. 2024; Zi et al. 2025) stored single-step source–instruction– target triplets. This organization disconnected initial generation from successive edits and omitted editing order and intermediate video states. The key gap therefore lies not only in scale, but also in data representation: isolated video clips do not capture connected video creation trajectory.

To bridge this gap, we introduce Sora100K, a large-scale dataset for video creation trajectory. As illustrated in Figure 1, it contains over 100K video records curated from the AI-Native social media platform Sora2 (OpenAI 2024) and organized into three structural subsets: text-to-video generation as generation roots, single-turn video editing as editing edges, and multi-turn video editing as complete video creation trajectories. These subsets jointly cover diferent stages of video creation trajectories. Beyond its trajectory-level organization, the proposed Sora100K explicitly preserves two properties that remain underexplored in open academic datasets, i.e., multi-shot structure and audio-visual synchronization. These characteristics make Sora100K closer to contemporary real-world video creation workflow. To make our dataset analyzable and reusable, we develop an annotation pipeline. We use Qwen2.5-VL-7B (Bai et al. 2025) to scale the annotation of generation semantics and editing operations, these annotations provide structured information at the three subsets. We use manual inspection to verify the accuracy of the automated annotations. To evaluate the practical utility of Sora100K, we further apply lightweight LoRA (Hu et al. 2022) adaptation to LTX-2 (HaCohen et al. 2026). The experimental results show that a small subset of Sora100K already provides efective supervision for video generation

and editing.

In summary, our main contributions are as follows:

• We introduce a representation that structures AI-Native video creation as a creation trajectory with generation roots and editing edges.

• We construct Sora100K, a large-scale dataset with semantic and editing annotations, source-to-edit lineage, intermediate video states, multi-shot structure, and audiovisual synchronization.

• We provide structured annotations and trajectory-level organization that support research on cross-turn dependencies, and maybe can support long-horizon video generation and editing in future research.

## Related Work

## Text-to-Video Generation Models and Datasets.

Early video generation methods (Vondrick, Pirsiavash, and Torralba 2016; Tulyakov et al. 2018), including GAN-based and autoregressive approaches, mainly focused on unconditional or weakly conditioned generation, limiting alignment with complex text prompts. Difusion-based models later improved text–video alignment through text encoders and large-scale text–image pretraining. Early video difusion models (Ho et al. 2022; Singer et al. 2023) still struggled with temporal consistency, while later latent difusion and spatiotemporal transformer architectures improved visual quality and temporal coherence. Recent models (Seedance et al. 2026; Kling AI 2026) further supported structured multi-shot generation and audio-visual synchronization.

Large-scale datasets such as WebVid-10M (Bain et al. 2021) and HD-VILA-100M (Xue et al. 2022) provided the text–video pairs required to train foundational text-tovideo models. Other datasets, including MSR-VTT (Xu et al. 2016) and ActivityNet (Caba Heilbron et al. 2015), were widely used for video understanding, captioning, and retrieval. These datasets generally organized samples as independent text–video pairs. In our workflow-centric representation, each pair corresponds to a generation root that can support subsequent editing operations. However, prior datasets did not connect generation roots with later editing edges and therefore did not preserve source-to-edit lineage, editing order, or intermediate video states.

## Video Editing Models and Datasets.

Instruction-driven visual editing first advanced in the image domain. InstructPix2Pix (Brooks, Holynski, and Efros 2023) used model-generated paired data to learn general editing instructions. Extending this setting to video required temporal consistency across frames. Tune-A-Video (Wu et al. 2023) and FateZero (Qi et al. 2023) used one-shot tuning or attention mechanisms to preserve source structure, while Token-Flow (Geyer et al. 2023) and CoDeF (Ouyang et al. 2023) modeled temporal correspondences for consistent editing. These methods mainly addressed individual editing operations, which correspond to editing edges in a video creation trajectory. Recent studies explored iterative editing and cross-turn consistency (Lee et al. 2026), but data preserving successive prompts and videos remained limited.

![](images/73b4d519b0ecc67656c809362e5972ce85d98115b444bc7b6da47617548a35d8.jpg)  
Figure 2: Comparison between representative video datasets and Sora100K. Prior datasets typically store video generation and editing as separate clip-level samples. Sora100K connects text-to-video generation roots, single-turn editing edges, and multi-turn editing trajectories, while preserving multi-shot structure and audio-visual synchronization.

Recent datasets expanded the scale and coverage of instruction-based video editing. InstructVid2Vid (Qin et al. 2024) constructed video–instruction triplets for general editing, while VIVID-10M (Hu et al. 2024) focused on largescale local editing. Señorita-2M (Zi et al. 2025) and OpenVE-3M (He et al. 2025) further improved data quality and editing diversity. However, these datasets generally stored each operation as an independent source–instruction–target triplet. Such triplets provide supervision for individual editing edges but do not connect them to the generation root or preserve editing order and intermediate states.

## Dataset Characteristics

In this section, we compared the diferences between Sora100K and previous datasets. As shown in Table 1 and Figure 2, Sora100K represents AI-Native video creation as a structured creation trajectory that connects the video generation, video editing and creation trajectory, Sora100K moves beyond isolated video clips by modeling creation trajectories and retaining multi-shot and audio-visual information.

## Structure-Level Characteristics

Video Creation Trajectories. AI-Native video creation begins with an initial generation and proceeds through successive editing operations. A creation tree represents all video states derived from the same initial prompt, while a creation trajectory is an ordered root-to-node path within the tree.

$$
\mathcal { C } _ { i } = \left( p _ { i } , \mathcal { V } _ { i } , \mathcal { E } _ { i } , r _ { i } \right) ,\tag{1}
$$

where $\mathcal { C } _ { i }$ denotes the i-th creation tree, p<sub>i</sub> denotes the initial generation prompt, $\nu _ { i }$ is the set of video states produced during the video creation trajectory, and $r _ { i } = v _ { i } ^ { 0 } \in \mathcal { V } _ { i }$ is the root node corresponding to the initial generated video. As defined in Eq. 1, the tree connects the initial prompt, generated video states, and editing operations within one video creation trajectory. Its edge set is defined as

$$
\mathcal { E } _ { i } = \{ ( v _ { i } ^ { a } , e _ { i } ^ { a  b } , v _ { i } ^ { b } ) \} ,\tag{2}
$$

where $v _ { i } ^ { a }$ is the source video state, $e _ { i } ^ { a  b }$ is the editing instruction applied to it, and $v _ { i } ^ { b }$ is the resulting video state.

A video creation trajectory connects the editing edges in Eq. 2 into a root-to-node path in $\mathcal { C } _ { i }$ :

$$
\mathcal { T } _ { i , k } = \left( p _ { i } , v _ { i } ^ { 0 } , e _ { i , k } ^ { 1 } , v _ { i , k } ^ { 1 } , \ldots , e _ { i , k } ^ { L _ { i , k } } , v _ { i , k } ^ { L _ { i , k } } \right) ,\tag{3}
$$

where $\mathcal { T } _ { i , k }$ denotes the k-th trajectory derived from $\mathcal { C } _ { i } , e _ { i , k } ^ { t }$ and $\boldsymbol { v } _ { i , k } ^ { t }$ denote the editing instruction and resulting video at turn t, respectively, and $L _ { i , k }$ is the total number of editing turns. When no branching occurs, the creation tree degenerates into a single trajectory as defined in Eq. 3.

The three subsets of Sora100K are not independent task collections. Instead, they provide three structural views of the creation trees defined in Eq. 1. Specifically, the text-tovideo generation subset records the initial prompt and root node, the single-turn video editing subset records individual editing edges, and the multi-turn video editing subset records complete root-to-node trajectories:

$$
\begin{array} { l } { \displaystyle \mathcal { D } _ { \mathrm { g e n } } = \big \{ \big ( p _ { i } , r _ { i } \big ) \big \} _ { i = 1 } ^ { N } , } \\ { \displaystyle \mathcal { D } _ { \mathrm { e d i t } } = \bigcup _ { i = 1 } ^ { N } \mathcal { E } _ { i } , } \\ { \displaystyle \mathcal { D } _ { \mathrm { t r a j } } = \big \{ \mathcal { T } _ { i , k } \big \} _ { i , k } , } \end{array}\tag{4}
$$

where N is the number of creation trees, and $\mathcal { D } _ { \mathrm { g e n } } , \mathcal { D } _ { \mathrm { e d i t } }$ , and $\mathcal { D } _ { \mathrm { t r a j } }$ denote the text-to-video generation, single-turn video editing, and multi-turn video editing subsets, respectively. As summarized in Eq. 4, generation serves as the trajectory root, each single-turn edit forms a transition edge, and each multi-turn sequence constitutes a complete trajectory.

![](images/f45c46814ec59ba306b126b20141f674af97643e4f7c63f02ba9207c3836683c.jpg)  
Text: A woman made of white yarn cries red threads, ...

![](images/08f8c86bf70d6d4b247107d065eef6b6d5dbb3b8970a968ec6f9c50f0364663d.jpg)

![](images/334a2af3975e4f59ffadb7114c400e3a1366bb5570cdcdf75625484bf6737165.jpg)

![](images/0a8b618426862fb05ceb9da388827a8c14ffdcc9548de897ae22dfd40fbca2ce.jpg)  
Text: A solitary man ascends a monumental staircase ...  
Edit Prompt 1: Replace environments inside the doorway...

![](images/1475db2bde121da0fdfccea8c936e5bcd3e0364cfa9ad28493cb122f21cfdae5.jpg)

![](images/b1347d71ea44f22f8e9eaf5d1ded97ec7937095e1dfcedf53d4f7579968dcc9c.jpg)  
Edit Prompt 2: Add a small yellow yarn cat beside the woman, ...  
Editing Prompt 3: Add multi colorful yarn cats ...

![](images/b9550ac31c304f00f93e76fbdea3ded24299a23b2f31444a20fb0b409d7bc6bb.jpg)

![](images/ba8c3ffedcfdb9a4a32c145fb82ebfe9798b030bbbf80045a29d7c6fb824989e.jpg)

![](images/03967d85f0a31faa3c4643199f63bc8554d13338752e03ebd9cdb47a99222066.jpg)  
Text: A girl with acoustic guitar + black cat; smile; ...

![](images/ac5bf1720ac4561a17cf603a1ce9567ec5e978d37f4a6a0cde59c1590a2d1417.jpg)  
Edit Prompt 3: Remove the second man and replace portal ...  
Edit Prompt 1: Give it a Christmas theme.  
Edit Prompt 2: Duplicate walking man and place the copy farther ...

![](images/b1e14f11a112b7e9dc61afb53b84e8908e60b4ae2560d1959cd2637eddb568cd.jpg)  
Edit Prompt 2: Make the cat dressed up too.

![](images/d0b40790249f2e79d8ef50d101cc160774340dc3e4423de0850fcce0aa4b829c.jpg)  
Edit Prompt 3: Japanese girl with acoustic guitar.

Figure 3: Sample video creation trajectories from Sora100K. In each row, the initial text-to-video generation record corresponds to the root. Each pair of adjacent video states and its editing instruction forms a single-turn editing edge, while the complete ordered sequence forms a video creation trajectory.

## Video-Level Characteristics

Sora100K exhibits two properties: audio-visual synchronization and multi-shot structure, which were weakly supported in prior video generation and editing datasets.

Audio-Visual Synchronization. Sora100K preserves audio signals that are closely aligned with the visual content. Earlier video datasets often contained visual-only samples because of the limited audio-generation capabilities of previous models, restricting their use forjoint audio-visual modeling. In contrast, Sora100K retains the original synchronized audio without excessive processing, providing a practical data foundation for audio-visual video generation.

Multi-Shot Structure. Sora100K contains a large proportion of videos with clear multi-shot organization. Earlier datasets were largely dominated by single-shot clips and provided limited scene transitions and shot changes. Although the Sora2 technical report did not explicitly emphasize multishot generation, our analysis shows that most collected videos contain multiple shots and generally maintain character consistency across shot transitions.

## Dataset Construction and Annotation

In this section, we describe how Sora100K is constructed, how its three subsets are annotated, and how each subset is composed within a unified and systematic framework. As illustrated in Figure 11, we present representative samples here and provide additional examples in the appendix.

## Data Curation and Quality Filtering

We construct Sora100K from approximately 180K public Sora2 (OpenAI 2024) records. After recovering prompts, editing instructions, and available source references, we apply accessibility, VBench-based quality, and traceability filtering, retaining approximately 137K, 111K, and finally 103.4K records. We organize the filtered records according to their recovered source-to-edit relations. Every valid initial generation is retained as a root in the text-to-video generation subset, even when no subsequent edit is available. Every valid source–instruction–target relation is retained as an edge in the single-turn video editing subset. Records containing only one edit therefore contribute both a generation root and an editing edge, while only connected paths with at least three video states are included in the multi-turn video editing subset. This process yields 18,451 generation roots, 76,964 editing edges, and 8,024 multi-turn trajectories. And we inspect representative trajectories and find their source-to-edit lineage, editing order, and intermediate states to be consistent with the recovered metadata. We use Qwen2.5-VL-7B (Bai et al. 2025) to annotate root semantics and edge-level editing operations, and further verify the automatic annotations through human evaluation.

## Dataset Annotation

A video creation trajectory contains information at diferent structural levels. We therefore adopt a hierarchical annotation scheme over the text-to-video generation, single-turn video editing, and multi-turn video editing subsets. These annotations characterize the generation roots, editing operations, and the video creation trajectories, respectively.

![](images/a8b96319eda3f48df54da9348f7c5ccef5eab51dddac17b5e7cebc2cd1f23cf4.jpg)  
(a) Overall composition

![](images/7bf2f109b525ce39d6e0a4483ea056e39479a659e951b6fb60541cfac70d48ac.jpg)

![](images/1bf03dc5454a60c39e9a6a09e3c2ffe4b026e11497b9d2610e0a5431626303ae.jpg)

![](images/bee53f9259408c7e9fdd8cc3e217bfac84f3c71d8e9078c03c04a9018a33aa18.jpg)

(b) Video statistics and generation prompt length  
![](images/6fe6341d78f6b5e87b917924a2bfb865f0c27435089ca5a770430a9dc77c74ed.jpg)  
Figure 4: Statistical analysis of Sora100K, including overall task proportions, video-level statistics, editing category distribution, generation category distribution, and prompt length distribution.

Root-Level Semantic Annotation. For the text-to-video generation subset, we characterize each generation root along five dimensions: semantic category, video duration, frame count, number of shots, and prompt length. We construct a single-label semantic taxonomy, and use Qwen2.5-VL-7B (Bai et al. 2025) to annotate the samples. Prompt length is extracted from the original metadata, while the video-level attributes are obtained through video analysis.

Edge-Level Operation Annotation. For the single-turn video editing subset, we characterize each editing edge by its editing category, video duration, frame count, and number of shots. We construct a multi-label taxonomy because one edited may involve multi-modifications. We use Qwen2.5- VL-7B (Bai et al. 2025) to annotate the samples.

Trajectory-Level Structural Annotation. For the multiturn video editing subset, we preserve the editing category, source-to-edit lineage, and chain length of each video creation trajectory. These structural annotations connect the generation roots and editing edges into ordered creation trajectory and support the analysis of multi-turn dependencies. This structural information connects the annotated generation roots and editing edges into complete trajectories.

## Dataset Statistics

We first examine the multi-turn video editing subset to characterize the trajectory-level structure of Sora100K, and followed by the generation roots and editing edges. The detailed distributions are shown in Figure 4. We present further statistical details and sample illustrations in the appendix.

Trajectory-Level Structural Statistics. Sora100K has 8,024 multi-turn video editing trajectories, accounting for 7.8% of its 103.4K structured records. As shown in Figure 4, we analyze editing-operation types rather than initialgeneration semantics. Object-level operations account for the largest proportion at 4.38%, followed by style-level (1.83%), plot-level (0.65%), scene-level (0.50%), and other operations (0.44%) of the full dataset.

We define chain length as the total number of video states in a trajectory, including the initial generation root and all subsequent edited states. Trajectories with 3, 4, and 5 states account for 61.5%, 18.4%, and 12.3%, respectively, while 7.8% contain more than 5 states. Thus, 92.2% of the trajectories represent compact iterative workflows with 3–5 states, whereas the remainder captures longer creation trajectory.

Root-Level Semantic Statistics. As shown in Figure 4, generation roots account for 17.8% of all structured records. Within the generation subset, people-related content dominates at 13.68%, followed by animals (2.33%), transportation (0.49%), scenery (0.43%), and other categories (0.86%) of the full dataset, respectively. Since the initial generations in multi-turn trajectories exhibit a similar semantic distribution, we report only the aggregate generation statistics.

Most videos are approximately 10 seconds long and contain around 300 frames at 30 FPS, while approximately 65.5% exhibit a multi-shot structure. For prompt length, 52.2% contain no more than 10 words, followed by 16.6% with 11–25 words, 10.3% with 26–50 words, 10.7% with 51–100 words, and 10.3% with more than 100 words.

Edge-Level Operation Statistics. Within the single-turn editing subset, object-level operations account for 38.68%, followed by style-level (20.09%), scene-level (6.70%), narrative and temporal (3.72%), and other operations (5.21%) of the full dataset, respectively. Generation roots and editing edges provide the initial states and state transitions that compose creation trajectories. Because the video-level attributes of the single-turn and multi-turn editing subsets closely match those of the generation subset, we omit their duplicate distributions.

## Natural Distribution and Annotation Reliability

The distributions in Sora100K are naturally imbalanced, with people-related generation samples and several common editing operations accounting for relatively large proportions. Rather than constructing a conventionally balanced benchmark, Sora100K preserves the natural creation processes observed on an AI-Native video platform (OpenAI 2024) . The semantic categories of generation roots and editing operations of edges therefore reflect the creation intents observed during initial generation and successive refinement. We retain the corresponding annotations and metadata, allowing researchers to resample the data or construct balanced subsets for specific downstream tasks.

![](images/94150284d4c4e1ce8d814320463ad62f879cba08becf58ffba3ca3f93e63d5c8.jpg)  
Text: A cinematic side-view shot of a mysterious …

![](images/3c3274d7e209838a732e6d2b0803427972369e6fa71b481597244bdf9c8a9079.jpg)  
Edit Prompt 1: ASMR man kicking a guardrail

![](images/e16bc0e26da4f9309857c453eae43fbc093da3495761a9d34d6d6ff43084ae8f.jpg)  
Edit Prompt 2: He says stupid guardrails tightening …  
Figure 5: Qualitative comparison on trajectory-level video editing. The Sora100K-adapted pipeline produces higher aesthetic quality while maintaining comparable subject consistency across successive editing turns.

To assess the reliability of the VLM-based annotations used to characterize creation trajectories, we randomly sample generation roots and editing edges and compare the labels produced by Qwen2.5-VL-7B (Bai et al. 2025) with human annotations under the same taxonomies. The automatic annotations achieve an accuracy of 98.19%, a Cohen’s κ of 0.959, and a weighted F1 score of 98.26%. These results support the reliability of the semantic and editing-operation annotations used to characterize creation trajectories.

## Experiment

We evaluate whether Sora100K provides efective supervision for trajectory-level video creation. Because current open-source models do not support direct training on complete multimodal trajectories, we adapt separate generation and editing components using trajectory roots and edges, then recombine them for successive-turn evaluation. We also evaluate each component independently to identify the sources of performance gains.

## Experiment Setting

We construct an iterative pipeline with LTX-2-T2AV (HaCohen et al. 2026) as the generator and LTX-2-T2AV-IC (Ha-Cohen et al. 2026) as the editor. Using LoRA (Hu et al. 2022), we adapt the generator on 3,000 generation roots and the editor on 3,000 editing edges. During evaluation, the generator produces the initial video from the root prompt, and the editor then applies the original instructions sequentially, using each output as the input to the next turn. We evaluate every resulting video state and compare the original and Sora100K-adapted pipelines. Detailed settings and more experiment samples are provided in the appendix.

<table><tr><td>Turn</td><td>Model</td><td> $S _ { \mathrm { e f f } }$  ↑  $S _ { \mathrm { S u b } }$ </td><td>↑</td><td> $S _ { \mathrm { A e s } }$ </td><td>↑ Avg.↑</td></tr><tr><td rowspan="2">Edit 1</td><td>LTX-2</td><td>52.0</td><td>97.5</td><td>41.5</td><td>63.7</td></tr><tr><td>Ours</td><td>54.0</td><td>97.0</td><td>45.4</td><td>65.5</td></tr><tr><td rowspan="2">Edit 2</td><td>LTX-2</td><td>34.0</td><td>97.0</td><td>41.5</td><td>57.5</td></tr><tr><td>Ours</td><td>32.0</td><td>97.7</td><td>48.4</td><td>59.4</td></tr><tr><td rowspan="2">Average</td><td>LTX-2</td><td>43.0</td><td>97.3</td><td>41.5</td><td>60.6</td></tr><tr><td>Ours</td><td>43.0</td><td>97.4</td><td>46.9</td><td>62.4</td></tr></table>

Table 2: Multi-turn video editing results over two successive editing turns. $S _ { \mathrm { e f f } }$ is a reference-based VLM score using the ground-truth edited video, while $S _ { \mathrm { S u b } }$ and $S _ { \mathrm { A e s } }$ are computed with VBench.

## Experiment Results

Trajectory-Level Video Creation Results. As shown in Figure 5, we create a trajectory, and we evaluate each edited state using a VLM-based evaluator (Bai et al. 2025) for editing efectiveness $( S _ { \mathrm { e f f } } )$ , subject consistency $( S _ { \mathrm { S u b } } )$ , and aesthetic quality $( S _ { \mathrm { A e s } } )$ , with their mean reported as the overall score. Table 2 shows that both pipelines degrade across turns. However, the Sora100K-adapted pipeline performs better, achieving a higher two-turn average of 62.4 versus 60.6.

The improvement primarily comes from aesthetic quality, whose two-turn average increases from 41.5 to 46.9. Subject consistency remains comparable, while editing efectiveness has the same two-turn average of 43.0 for both pipelines. These results show that Sora100K supervision improves the visual quality of video states throughout iterative creation, although successive editing remains challenging as errors accumulate across turns.

But the trajectory pipeline combines separate generation and editing models, the trajectory-level gains do not directly reveal which component benefits most from Sora100K supervision. We therefore conduct two controlled experiments that isolate the generation and editing stages, respectively, to identify the primary source of improvement.

Root-Level Video Generation Experiments. We compare the original LTX-2 generator with models adapted on Panda-70M (Chen et al. 2024) and Sora100K using the same 200 held-out prompts. As shown in Table 3, Sora100K achieves the best overall score of76.43 and leads on five ofthe six metrics. Compared with the original model, it improves aesthetic quality from 59.11 to 62.98, imaging quality from 47.84 to 57.22, and ArcFace (Deng et al. 2019) from 48.30 to 59.70. The substantial ArcFace gain indicates stronger identity consistency across shots, while motion smoothness remains nearly unchanged. Figure 6 further shows that the Sora100K-adapted model produces cleaner structures and more consistent subjects across shots. These results confirm that generation-root supervision provides a stronger initial state for trajectory construction.

<table><tr><td>Model</td><td>Subject Consistency↑</td><td>Background Consistency↑</td><td>Aesthetic Quality↑</td><td>Imaging Quality↑</td><td>Motion Smoothness↑</td><td>ArcFace↑</td><td>Avg.↑</td></tr><tr><td>LTX-2(Base)</td><td>86.07</td><td>92.07</td><td>59.11</td><td>47.84</td><td>99.60</td><td>48.30</td><td>72.17</td></tr><tr><td>Panda-70M</td><td>86.73</td><td>91.23</td><td>49.00</td><td>46.66</td><td>99.63</td><td>49.23</td><td>70.41</td></tr><tr><td>Sora100K</td><td>86.90</td><td>92.24</td><td>62.98</td><td>57.22</td><td>99.56</td><td>59.70</td><td>76.43</td></tr></table>

Table 3: Quantitative comparison of the base LTX-2 model and its variants fine-tuned on 3,000 samples from Panda-70M and Sora100K, respectively. All fine-tuned variants use the same training configuration. Bold indicates the best result.

![](images/b6ae60d91ef2be175d50bb96e34034837cd6b20f425158a699fa414a257bd6ac.jpg)  
Text: A man walking alone through a foggy forest at dawn. fog moving between trees. His face as he notices something ...

![](images/49ea9291503ba8a99487288d6e5f23a8f8862eec3f32b6cb0f4d0c985cd16d68.jpg)  
Text: A large desert landscape under a bright sun. A traveler slowly walking across the sand dunes ...

![](images/4af3b9d10fd222ec44017ff1fd16ec3991456f5ee15cf7e324380116ccce1bed.jpg)  
Text: An old library filled with tall wooden bookshelves and soft sunlight. A student slowly walking …

Figure 6: Qualitative comparison between LTX-2 (HaCohen et al. 2026) and our model on video generation. Our model yields more consistent subjects, and fewer structural artifacts across shots.
<table><tr><td>Model</td><td> $S _ { \mathrm { i n s t } }$  ↑</td><td> $S _ { \mathrm { e f f } }$  ↑</td><td> $S _ { \mathrm { S u b } }$  ↑</td><td> $S _ { \mathrm { A e s } }$ </td><td>↑ Avg.↑</td></tr><tr><td>LTX-2</td><td>43.4</td><td>38.7</td><td>92.0</td><td>49.8</td><td>56.0</td></tr><tr><td>Ours</td><td>47.9</td><td>44.2</td><td>91.7</td><td>53.8</td><td>59.4</td></tr></table>

Table 4: Single-turn video editing results. The average is the arithmetic mean of the four reported metrics.

Edge-Level Video Editing Experiments. We next evaluate whether Sora100K improves individual state transitions along a trajectory. A VLM-based evaluator (Bai et al. 2025) measures instruction following $( S _ { \mathrm { i n s t } } )$ , editing efectiveness $( S _ { \mathrm { e f f } } )$ , subject consistency $( S _ { \mathrm { S u b } } )$ , and aesthetic quality $( S _ { \mathrm { A e s } } )$ . As shown in Table 4, adaptation improves $S _ { \mathrm { i n s t } }$ from 43.4 $\mathrm { t o } 4 7 . 9 , S _ { \mathrm { e f f } }$ from 38.7 to 44.2, and $S _ { \mathrm { A e s } }$ from 49.8 to 53.8, increasing the average score from 56.0 to 59.4. Subject consistency remains nearly unchanged, decreasing slightly from 92.0 to 91.7. Figure 7 shows corresponding improvements in editing strength and visual quality. These results demonstrate that editing-edge supervision improves the transitions between successive video states.

![](images/e7f1b9eeded7095229103cb707362abf94e05b9602d2ce8002b850d82b4ceb2b.jpg)  
Edit Prompt: Enhance lighting  
Figure 7: Qualitative comparison between LTX-2 (HaCohen et al. 2026) and our model on video editing. Our model achieves better aesthetic quality, editing efectiveness and stronger instruction following.

Generation roots improve the initial state, while editing edges strengthen subsequent transitions; however, performance still declines across turns.

## Conclusion

We introduce Sora100K, a dataset that extends video data from isolated clips to structured audio-visual creation trajectories. It represents text-to-video generation, single-turn video editing, and multi-turn video editing subsets as roots, editing edges, and complete trajectories, while preserving source-to-edit lineage, intermediate video states. It also retains multi-shot structure and synchronized audio, more faithfully capturing key properties of modern AI-generated videos. Sora100K is constructed through systematic filtering, workflow reconstruction, structured annotation, and human validation, providing a curated resource for studying AI-Native video creation. Experiment results show that Sora100K provides efective supervision for video generation and single-turn editing, thereby enhancing performance in areas such as visual quality, cross-shot consistency, editing efectiveness and stronger instruction following.

Limitations. First, suitable open-source models remain limited; many (Team et al. 2026; Zhang et al. 2026) are derived from LTX-2 or incompatible with our input modalities. Second, creation trajectories remain beyond the capabilities of most open-source models, those are close-source.

## References

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Bain, M.; Nagrani, A.; Varol, G.; and Zisserman, A. 2021. Frozen in time: A joint video and image encoder for end-toend retrieval. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. (WebVid-10M dataset).

Brooks, T.; Holynski, A.; and Efros, A. A. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Caba Heilbron, F.; Escorcia, V.; Ghanem, B.; and Niebles, J. C. 2015. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Chen, T.-S.; Siarohin, A.; Menapace, W.; Deyneka, E.; Chao, H.-w.; Jeon, B. E.; Fang, Y.; Lee, H.-Y.; Ren, J.; Yang, M.-H.; et al. 2024. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13320–13331.

Deng, J.; Guo, J.; Xue, N.; and Zafeiriou, S. 2019. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4690–4699.

Geyer, M.; Bar-Tal, O.; Bagon, S.; and Dekel, T. 2023. Tokenflow: Consistent difusion features for consistent video editing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision.

Google. 2025. Generate videos with Veo 3.1 in Gemini API. Accessed: 2026-03-29.

HaCohen, Y.; Brazowski, B.; Chiprut, N.; Bitterman, Y.; Kvochko, A.; Berkowitz, A.; Shalem, D.; Lifschitz, D.; Moshe, D.; Porat, E.; et al. 2026. LTX-2: Eficient Joint Audio-Visual Foundation Model. arXiv preprint arXiv:2601.03233.

He, H.; Wang, J.; Zhang, J.; Xue, Z.; Bu, X.; Yang, Q.; Wen, S.; and Xie, L. 2025. OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing. arXiv preprint arXiv:2512.07826.

Ho, J.; Chan, T. S.; Fleet, D.; and Norouzi, M. 2022. Video difusion models. NeurIPS.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).

Hu, J.; Zhong, T.; Wang, X.; Jiang, B.; Tian, X.; Yang, F.; Wan, P.; and Zhang, D. 2024. Vivid-10m: A dataset and baseline for versatile and interactive video local editing. arXiv preprint arXiv:2411.15260.

Kling AI. 2026. All in One, One for All! Kling 3.0 Model Now Fully Rolled Out.

Lee, D.; Huang, C.-H. P.; Chen, X.; Ye, J. C.; Ceylan, D.; and Jeong, H. 2026. Memory-V2V: Memory-Augmented Video-to-Video Difusion for Consistent Multi-Turn Editing. arXiv:2601.16296.

OpenAI. 2024. Video generation models as world simulators. arXivpreprint. (Placeholder for Sora/Sora2 technical report).

Ouyang, H.; Wang, Q.; Xiao, Y.; Bai, Q.; Zhang, J.; Zheng, K.; Zhou, X.; Chen, Q.; and Shen, Y. 2023. CoDeF: Content Deformation Fields for Temporally Consistent Video Processing. arXiv preprint arXiv:2308.07926.

Qi, C.; Cun, X.; Zhang, Y.; Lei, C.; Wang, X.; Shan, Y.; and Chen, Q. 2023. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Qin, B.; Li, J.; Tang, S.; Chua, T.-S.; and Zhuang, Y. 2024. Instructvid2vid: Controllable video editing with natural language instructions. In 2024 IEEE International Conference on Multimedia and Expo (ICME), 1–6. IEEE.

Runway. 2025. Introducing Runway Gen-4.

Seedance, T.; Chen, D.; Chen, L.; Chen, X.; Chen, Y.; Chen, Z.; Chen, Z.; Cheng, F.; Cheng, T.; Cheng, Y.; et al. 2026. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148.

Singer, U.; Polyak, A.; Hayes, T.; Yin, X.; An, J.; Zhang, S.; Hu, Q.; Yang, H.; Ashual, O.; Gafni, O.; et al. 2023. Make-A-Video: Text-to-Video Generation without Text-Video Data. In ICLR.

Team, O.; Yu, D.; Chen, M.; Chen, Q.; Luo, Q.; Wu, Q.; Cheng, Q.; Li, R.; Liang, T.; Zhang, W.; et al. 2026. Mova: Towards scalable and synchronized video-audio generation. arXiv preprint arXiv:2602.08794.

Tulyakov, S.; Liu, M.-Y.; Yang, X.; and Kautz, J. 2018. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Vondrick, C.; Pirsiavash, H.; and Torralba, A. 2016. Generating videos with scene dynamics. In NeurIPS.

Wang, Q.; Shi, Y.; Ou, J.; Chen, R.; Lin, K.; Wang, J.; Jiang, B.; Yang, H.; Zheng, M.; Tao, X.; et al. 2025. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, 8428–8437.

Wu, J. Z.; Ge, Y.; Wang, X.; Lei, S. W.; Gu, Y.; Shi, Y.; Hsu, K.; Shan, Y.; Qie, X.; and Shou, M. Z. 2023. Tune-a-video: One-shot tuning of image difusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Xu, J.; Mei, T.; Yao, T.; and Rui, Y. 2016. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Xue, H.; Sun, Y.; Liu, B.; Fu, J.; Song, R.; Li, H.; and Luo, J. 2022. Clip-vip: Adapting pre-trained image-text model to video-language representation alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. (HD-VILA-100M dataset related).

Zhang, G.; Ma, X.; Huang, J.; Xu, H.; Yu, H.; Fu, S.; Li, Y.; Xue, Z.; Song, L.; Huang, H.; et al. 2026. Omninft: Modality-wise omni difusion reinforcement for joint audiovideo generation. arXiv preprint arXiv:2605.12480.

Zi, B.; Ruan, P.; Chen, M.; Qi, X.; Hao, S.; Zhao, S.; Huang, Y.; Liang, B.; Xiao, R.; and Wong, K.-F. 2025. Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists. In NeurIPS D&B.

## Appendix

## Motivation and Dataset Gap

As shown in Figure 8 – Figure 9, we present a comparison of the proposed dataset with existing representative video datasets. Existing public datasets for video generation and editing remain primarily organized around cliplevel, no-audio video samples, ofering very limited explicit support for multi-shot structures and audio-visual video creation. Consequently, these datasets are currently insuficient for investigating long-range temporal consistency, cross-shot composition, and audiovisual coherence.

![](images/cbd746604a8470e0343d5a91d16108abe125a2a37b81f807b96403ec7b5e5aab.jpg)  
Figure 8: Representative text-to-video samples from Panda-70M, Koala-36M, and Sora100K, illustrating their diferences in shot organization and visual content.

![](images/0f3adb528b29fbc2bd35496d454bf457d24af03e5bd45b79fe35aff6d5ac6021.jpg)

![](images/bce0b7aa191e3bdd52e0df1b1ebec4a9d34e86d9278eb56d57f5945bc8f38a44.jpg)  
Figure 9: Representative video editing samples from OpenVE-3M and Sora100K. The source and edited videos illustrate diferent types of content and style modifications.

Compared to existing video generation and editing datasets, Sora100K stands out for its more systematic coverage of key attributes such as long-term content, multishot structures, and audio-visual synchronization. These attributes are more aligned with current trends in video generation and the creative needs of real users. Therefore, this not only enhances the dataset’s relevance to real-world scenarios but also provides a more supportive resource foundation for research on long-range dependency modeling, cross-shot organization, and audio-visual collaborative generation.

## Dataset Samples

## Text-to-Video Generation Subset

In this section, we will present more samples from our dataset, with each sample displayed according to our classification categories. For Text-to-Video Generation, we organize all samples into eight semantic categories: People, Animal, Plant, Food & Drink, Transportation, Scenery, Architecture, and Science & Tech. Each generation video is assigned a primary label based on its main narrative theme. Each samples as shown in Figure 10 – Figure 11.

People. This category includes videos in which one or more human figures constitute the primary visual subject, covering portraits, social interactions, performances, sports, professional activities, and everyday behaviors. A video is assigned to this category when people remain the central focus, even if animals, vehicles, or environmental elements are also present. This category contains 14,169 samples.

Animal. This category includes videos in which animals constitute the primary visual subject, including pets, wildlife, birds, insects, marine animals, and anthropomorphic animals. This category contains 2,417 samples.

Transportation. This category includes videos primarily depicting vehicles or transportation-related activities, such as cars, trains, aircraft, ships, bicycles, motorcycles, and other forms of transportation. This category contains 511 samples. Scenery. This category includes videos primarily depicting natural environments or broad landscapes without a dominant human, animal, or human-made subject. Representative content includes mountains, forests, beaches, rivers, deserts, skies, and weather-related scenes. This category contains 446 samples.

Food & Drink. This category includes videos centered on food, beverages,ingredients, cooking processes, food preparation, plating, or eating and drinking scenes. This category contains 380 samples.

Architecture. This category includes videos in which buildings or human-made spaces constitute the primary visual subject, including houses, palaces, temples, bridges, urban structures, interior spaces, and other architectural environments. This category contains 237 samples.

Science & Technology. This category includes videos primarily depicting scientific or technological subjects, such as robots, electronic devices, computer interfaces, laboratories, spacecraft, industrial equipment, and futuristic technologies. This category contains 234 samples.

Plants. This category includes videos in which plants themselves constitute the primary visual subject, including flowers, trees, crops, potted plants, and plant-growth processes. Plants appearing only as part of a broader natural environment are instead categorized as Scenery. This category contains 44 samples.

![](images/e8917394dd33f895daa10c7979c7b2e0cbf9f2e4ebdb20ed83df318da0dadfab.jpg)  
Figure 10: Representative samples from the Text-to-Video Generation subset across eight semantic categories: People, Animals, Architecture, Food & Drink, Plants, Scenery, Science & Technology, and Transportation.

## Single-Turn Video Editing Subset

We further analyze the Single-Turn Video Editing subset along four dimensions: editing category, video duration, frame count, and number of shots. Based on the automatic annotations produced by Qwen2.5-VL, object-level editing accounts for the largest proportion at 52%, followed by stylelevel editing at 27%. Temporal and scene-level editing account for 9% and 5%, respectively, while the remaining 7% consists of operations that are not fully covered by the predefined taxonomy. These distributions show that Sora100K covers common video editing operations while retaining diverse and open-ended editing requests. Each samples as shown in Figure 12.

Object-Level Editing. This category includes localized modifications to foreground objects, people, animals, or other visible entities. Typical operations include adding, removing, replacing, duplicating, resizing, repositioning, recoloring, or modifying the attributes of an entity while largely preserving the surrounding environment and overall visual style. Object-level editing accounts for 52% of the Single-Turn Video Editing subset.

Style-Level Editing. This category includes modifications to the visual appearance of a video without fundamentally changing its primary subjects or scene structure. Typical operations include changing the artistic style, color palette, lighting, texture, rendering mode, or overall cinematic tone. Style-level editing accounts for 27% of the subset.

Temporal Editing. This category includes modifications to actions, motion dynamics, or the temporal organization of a video. Typical operations include changing an action, motion speed, movement direction, motion trajectory, event order, duration, or temporal continuity. Temporal editing accounts for 9% of the subset.

Scene-Level Editing. This category includes modifications to the broader environment or spatial composition of a video.

Food & Drink

Architecture

![](images/996057fb4097a357069d6f2733420e3f40bcd7ebd8105ff40b0dd442e5f402e8.jpg)  
Figure 11: Representative samples from the Text-to-Video Generation subset across eight semantic categories: People, Animals, Architecture, Food & Drink, Plants, Scenery, Science & Technology, and Transportation.

Typical operations include changing the background, location, weather, time of day, scene layout, viewpoint, or camera framing while retaining the main subject when appropriate. Scene-level editing accounts for 5% of the subset.

Other Editing. This category includes uncommon, compound, or ambiguous editing operations that cannot be reliably assigned to the predefined object-, style-, temporal-, or scene-level categories. It also covers instructions that simultaneously modify multiple aspects without a clearly dominant editing intent. This category accounts for the remaining 7% of the subset.

## Multi-Turn Video Editing Subset

The Multi-Turn Video Editing subset captures iterative creation processes in which each trajectory starts from an initial generated video and evolves through multiple successive editing turns. Unlike the Single-Turn Video Editing subset, which represents individual source–instruction–target transitions, this subset preserves the ordered sequence of intermediate video states and editing instructions. Each samples as shown in Figure 13.

## Dataset Annotation and Statistics

This section presents the annotation fields and detailed statistics of the three subsets in Sora100K: Text-to-Video Generation, Single-Turn Video Editing, and Multi-Turn Video Editing. We first describe the fields provided for each subset and then report their composition, semantic categories, editing categories, video attributes, prompt lengths, and editing chain lengths.

Table 5 summarizes the fields provided for the three subsets of Sora100K. The Text-to-Video Generation subset records the original prompt, supplementary description, video attributes, and semantic category. The Single-Turn Video Editing subset records each source–instruction–target transition together with its editing labels and video statistics. The Multi-Turn Video Editing subset further organizes successive video states into editing chains and records the length of each chain.

## Annotation Fields

Text-to-Video Generation Subset. Each sample in the Text-to-Video Generation subset contains seven fields. The sample\_id field provides a unique sample identifier, while prompt stores the original generation prompt. The description\_qwen field contains a supplementary video description generated by Qwen2.5-VL. The video attributes are recorded by duration\_sec, frame\_count, and scene\_count, which represent the video duration in seconds, the number of frames, and the number of scenes or shots, respectively. The category field stores the generation category label of the sample.

Scene-Level Editing  
Narrative and Temporal Editing  
![](images/64a89444e3085917530d5fda9d5d04506ddc52f43c9f9cfb2f0b14c80f71716d.jpg)  
Figure 12: Representative samples from the Single-Turn Video Editing subset across four editing categories. Each example shows the source video on the left and the edited result on the right, covering object-level, style-level, scene-level, and narrative and temporal editing operations.

Single-Turn Video Editing Subset. Each sample in the Single-Turn Video Editing subset represents one editing transition. The sample\_id field identifies the sample, while source\_video\_id and edited\_video\_id identify the source and edited videos, respectively. The edit\_instruction field stores the original editing instruction. The edit\_labels field records the corresponding editing category labels, such as object modification and style transfer. The subset also records the video duration, frame count, and scene or shot count of each sample.

Multi-Turn Video Editing Subset. The Multi-Turn Video

Editing subset organizes successively edited videos into ordered editing chains. Each chain is identified by the folder\_name field. The chain\_length field records the number of videos contained in the chain.

## Overall Dataset Composition

Sora100K contains 103,439 entries across its three subsets. The Text-to-Video Generation subset contains 18,451 entries and accounts for 17.8% of the dataset. The Single-Turn Video Editing subset contains 76,964 entries, representing the largest proportion at 74.4%. The remaining 8,024 entries belong to the Multi-Turn Video Editing subset, accounting for 7.8%.

## Text-to-Video Generation Statistics

Semantic Category Distribution. The semantic distribution of the Text-to-Video Generation subset is dominated by people-related content. The People category accounts for 76.85% of the subset, followed by Animal at 13.11%. Transportation and Scenery account for 2.77% and 2.42%, respectively. The remaining categories are grouped as Other, which accounts for 4.85%.

<table><tr><td>Field</td><td>Type</td><td>Description</td><td>Example</td></tr><tr><td colspan="4">Text-to-Video Generation</td></tr><tr><td>sample_id</td><td>string</td><td>Unique sample identifier</td><td>s_68dc...5177</td></tr><tr><td>prompt</td><td>string</td><td>Original generation prompt</td><td>Three guys getting lunch ... Three young men walk along a</td></tr><tr><td>description_qwen</td><td>string</td><td>Qwen2.5-VL-generated description</td><td>busy city street on a sunny day.</td></tr><tr><td>duration/frame/scene</td><td>tuple</td><td>Duration, frame count, and shot count</td><td>10/300/1</td></tr><tr><td>category</td><td>string</td><td>Generation category</td><td>Real_People</td></tr><tr><td colspan="4">Single-Turn Video Editing</td></tr><tr><td>sample/source/id</td><td>string</td><td>Sample, source-video, and edited-video identifiers</td><td>s_68de...3765/ s_68e1...ee3c</td></tr><tr><td>edit_instruction</td><td>string</td><td>Original editing instruction</td><td>Change to a girl.</td></tr><tr><td>edit_labels</td><td>list</td><td>Editing category labels</td><td>[object_modification, style_transfer]</td></tr><tr><td>duration/frame/scene</td><td>tuple</td><td>Duration, frame count, and shot count</td><td>10/300/1</td></tr><tr><td colspan="4">Multi-Turn Video Editing</td></tr><tr><td>folder_name</td><td>string</td><td>Unique creation-trajectory identifier</td><td>s_68de...1d28</td></tr><tr><td>chain_length</td><td>int</td><td>Number of videos in the trajectory 3</td><td></td></tr></table>

Table 5: Field definitions and representative examples for the three Sora100K subsets.

Prompt Length Distribution. Generation prompts are divided into five length ranges. Prompts containing no more than 10 words account for 52.20% of thesubset. Prompts containing 11–25 words account for 16.55%, while those containing 26–50 and 51–100 words account for 10.28% and 10.65%, respectively. Prompts longer than 100 words account for the remaining 10.32%.

Video Duration. Most videos have durations close to 10 seconds. Videos lasting between 9.5 and 10.5 seconds account for 73.48% of the dataset, while videos lasting between 10.5 and 15 seconds account for 23.84%. Videos no longer than 9.5 seconds account for 2.57%, and only 0.11% are longer than 15 seconds.

Scene Count. The scene-count distribution is reported over five ranges: 0, 1–3, 4–6, 7–10, and more than 10. Videos containing 4–6 or 7–10 scenes constitute the largest groups, while videos containing more than 10 scenes account for the smallest proportion.

## Single-Turn Video Editing Statistics

The Single-Turn Video Editing subset covers five major groups of editing operations. Object-level editing accounts for the largest proportion at 52%, followed by style-level editing at 27%. Scene-level editing accounts for 9%, while narrative and temporal editing accounts for 5%. The remaining 7% is categorized as Other.

Object-Level Editing. Object-level editing modifies visible entities or their attributes within a video. This group accounts for 52% of the Single-Turn Video Editing subset.

Style-Level Editing. Style-level editing changes the visual appearance or presentation style of a video. This group accounts for 27% of the subset.

Scene-Level Editing. Scene-level editing changes the scene or environmental content of a video. This group accounts for 9% of the subset.

Narrative and Temporal Editing. Narrative and temporal editing changes the narrative content or temporal development of a video. This group accounts for 5% of the subset.

## Multi-Turn Video Editing Statistics

The Multi-Turn Video Editing subset contains ordered video chains formed through successive editing operations. We use chain length, defined as the number of videos contained in an editing chain, to describe the structural distribution.

## Experiment

## Experiment Setting

Implementation Details. For the text-to-video generation experiment, we randomly sample 3,000 videos from the Textto-Video Generation subset and adapt the base LTX-2 model using LoRA. Each training sample retains its audio input and is resized to a resolution of 512 × 768 with 121 frames. We train the model for five epochs using a learning rate of $1 \times 1 0 ^ { - 4 }$ . The LoRA rank is set to 32, and LoRA adapters are applied to the $\mathrm { t o \_ k , \ t o \_ q , \ t o \_ v }$ , and to\_out.0 modules. Gradient checkpointing is enabled to reduce GPU memory consumption. The experiment is conducted on a single NVIDIA H100 GPU, with approximately 40 GB of peak GPU memory usage. The detailed configuration is summarized in Table 6. We follow the oficial LTX-2 training workflow and divide the adaptation process into data preprocessing and LoRA fine-tuning. For the video editing experiments, we randomly sample 3,000 samples from the Single-Turn Video Editing subset and fine-tune LTX-2-T2AV-IC. Each training sample contains the edited video, input audio, and

<table><tr><td>Configuration</td><td>Text-to-Video Generation</td><td>Single-Turn Video Editing</td></tr><tr><td>Base model</td><td>LTX-2</td><td>LTX-2-T2AV-IC</td></tr><tr><td>Training subset</td><td>Text-to-Video Generation</td><td>Single-Turn Video Editing</td></tr><tr><td>Training samples</td><td>3,000</td><td>3,000</td></tr><tr><td>Input</td><td>Video and audio</td><td>Source video, target video, and audio</td></tr><tr><td>Resolution</td><td> $5 1 2 \times 7 6 8$ </td><td> $5 1 2 \times 7 6 8$ </td></tr><tr><td>Number of frames</td><td>121</td><td>81</td></tr><tr><td>Fine-tuned component</td><td>DiT transformer</td><td>DiT transformer</td></tr><tr><td>Adaptation method</td><td>LoRA</td><td>LoRA</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs</td><td>5</td><td>5</td></tr><tr><td>LoRA rank</td><td>32</td><td>32</td></tr><tr><td>LoRA target modules</td><td> $\mathrm { t o \_ k , t o \_ q , t o \_ v , \bar { \Delta } v , }$  and  $\mathtt { t o \_ o u t \_ 0 }$ </td><td> $\mathrm { t o \_ k , t o \_ q , t o \_ v , \bar { \Delta } v , }$  and  $\mathtt { t o \_ o u t \_ 0 }$ </td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td><td>Enabled</td></tr><tr><td>Hardware</td><td>One NVIDIA H100 GPU</td><td>One NVIDIA H100 GPU</td></tr><tr><td>Peak GPU memory</td><td>Approximately 40 GB</td><td>Approximately 40 GB</td></tr></table>

Table 6: Training configurations for LoRA adaptation on the Text-to-Video Generation and Single-Turn Video Editing subsets.

![](images/476ab0341d6fa1af4f18e4fb07d6c174a4ae23b1f08a13d67b5e8c8beefb45b0.jpg)  
Text: # Dustlight Hymn — Music Video (Sora2 Video Prompt) ## Subject / Scene Settings - Audience: {locale="JP"}; Narrative tone: poetic…

![](images/f2e98621d746b1b69ed038296ffb1ea8f24c19d2bcdd58f3195bb64df174d059.jpg)  
Edit Prompt 1: Make her a fair skin woman with vampire fangs singing operatic metal wearing a red lace and beaded flowy dress …

![](images/1ed1e1ddf19554a8ee87f4b402a1acb3b08aac43fc83c065ffbb13cc6192efee.jpg)  
Edit Prompt 2:Same but another woman sings an English song. Song is about peace on the world…

Figure 13: Representative multi-turn video editing trajectory.

an in-context source video, together with the frame rate and context-downsampling information. All videos are resized to $5 1 2 \times 7 6 8$ and represented by 81 frames. During preprocessing, we load the text-encoder post-processing modules, video VAE encoder, audio VAE encoder, and Gemma-3-12B weights to precompute the training representations. We then fine-tune the DiT transformer for five epochs using a learning rate of $1 \times 1 0 ^ { - 4 }$ . LoRA is applied to the to\_ $\_ \mathrm { k , t o \_ q , t o \_ v , }$ and to\_out.0 modules with a rank of 32. Gradient checkpointing is enabled throughout training. All experiments are conducted on a single NVIDIA H100 GPU, with approximately 40 GB of peak GPU memory usage. The detailed configuration is summarized in Table 6.

## Experiment Result.

As shown in Figure 14 – Figure 17, we provide additional qualitative results of multi-turn video creation trajectories.

![](images/4dcdc4da7da2f4a27e1d7669801b31242d24c7f29dbb56613baf635722b727f4.jpg)

Text: A cinematic three-shot sequence.Wide shot of a snowy mountain landscape under a pale winter sky.Medium shot of a climber carefully walking along a narrow snowy ridge.Close-up of the climber’s determined face as cold wind blows across the mountain.Style:cinematic storytelling, smooth transitions between shots, realistic lighting, film quality.  
![](images/e751efe4ba77040ea25ab56cba783c107620f956f5a8f00572a56505e4dfb472.jpg)

Text: A cinematic three-shot sequence.Wide shot of a quiet countryside road surrounded by golden wheat fields.Medium shot of a young woman walking slowly along the road while the wind moves the tall grass.Close-up of her thoughtful expression as she looks toward the horizon.Style:cinematic storytelling, smooth transitions between shots, realistic lighting, film quality.  
![](images/60839e76c497c01dd4ea8360fad6bd8c421b833bf55a925fa90f4601e27707d6.jpg)  
Text: A cinematic three-shot sequence.Wide shot of a calm beach during sunset with waves gently touching the shore.Medium shot of a golden retriever running playfully along the sand.Close-up of the dog slowing down and looking toward the ocean.Style:cinematic storytelling, smooth transitions between shots, realistic lighting, film quality.  
Figure 14: Qualitative comparison between the Sora100K-adapted model and LTX-2 on text-to-video generation. Each example presents sampled frames from the two models under the same prompt, illustrating their diferences in visual quality, subject preservation, and consistency across shots. Red boxes highlight representative subject-consistency diferences.

![](images/3eca83673b9d23cf15daaa2a099f1bb71f37b9f6a958d8cf6b6fb9e7bfae82df.jpg)  
Edit Prompt: Remove an orange flower, merge two flowers into one…

![](images/5c5430a0cfe943c2ca5265b1b524c01957e9f8495461bc88391198ff3f404df6.jpg)  
Edit Prompt: Add a glowing blue light effect behind the woman and a glowing…

![](images/d987ce391abb7c3d80d7f6decc2de005b8159cd6ac798b037fa1b26c23519c0a.jpg)  
Edit Prompt: Add a ghost watermark…

Figure 15: Qualitative comparison between the Sora100Kadapted model and LTX-2 on single-turn video editing. The examples cover object removal and merging, lighting-efect addition, and watermark insertion.  
![](images/c2ab24c259ce7a6bddf33490ccac8cac424833ce966cd765826ebaf2d8fadf28.jpg)  
Edit Prompt4: And don't worry friends if your …

Figure 16: Qualitative comparison of multi-turn video editing between LTX-2 and our model. The models sequentially modify the source video according to four editing instructions. Our model better preserves character appearance and scene structure while accumulating changes across turns.

Source

![](images/72bb3d0325158184b14b7c90dfa1b751e9d974435c547e98de04ab4d61402f4e.jpg)  
Text: Hi detail high motion high definition Toei Animation anime style, high detail, slow motion, cinematic lighting…

![](images/e4768258e09e98c344e94a6b670aec03abca7bc67bb0b489a295736fd5538005.jpg)  
Edit Prompt1: High quality high motion anime style  
<sub>LT</sub><sup>X-</sup> Ours

![](images/0e03bf29fc20ec8168d6ebcfd502e20827dd645e9e40a440a66c0beda4744167.jpg)  
Edit Prompt2: aries grim has a halo.  
Figure 17: Qualitative comparison of multi-turn video editing between LTX-2 and our model. Starting from the same source video, the models sequentially apply a high-motion anime style and add a halo to the character. Our model better preserves the visual content and previous editing efects across successive turns.