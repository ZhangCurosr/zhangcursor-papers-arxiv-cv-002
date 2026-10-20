# MANGA2ANIMATE: AN AGENTIC FRAMEWORK FOR FAITHFUL MANGA ANIMATION WITH SCRIPT AND STORYBOARD PLANNING

Sheng-Wei Zhang<sup>∗</sup>, Hsi-An Chen<sup>∗</sup>, Yan-Ting Liou<sup>∗</sup>,   
Chih-Hsuan Chen, Shih-Chen Tseng, Yu-Lun Liu   
Department of Computer Science, National Yang Ming Chiao Tung University

![](images/e4ce9dcd03db4742716ae7bcce2376ecffce6e192202aa4f356914ccd92e8076.jpg)  
Figure 1: Manga2Animate turns existing manga pages into coherent animation that stays faithful to the original manga. Given a few manga pages (left), our multi-agent framework first builds an animation script and a visual storyboard as intermediate representations, iteratively refining both through creator-reviewer collaboration, and then generates the final animation from them (right). The resulting animation preserves the manga’s plot, character identity, and art style, while flowing coherently from clip to clip.

## ABSTRACT

Adapting existing manga into coherent animation that stays faithful to the source is a valuable yet challenging task. Existing methods either access the manga mainly through text, losing its characters and art style, or feed manga pages directly into video generation models, which misinterpret the manga’s structure and produce incorrect storylines. In this paper, we propose Manga2Animate, a multi-agent framework that grounds every stage of generation in the source manga. We decompose the task into two intermediate representations, an animation script and a visual storyboard. Each representation is produced by a creator agent and reviewed against the source panels by a paired reviewer agent. We tailor representation-specific selection and dual-level feedback to progressively refine both before synthesizing the final animation. Extensive experiments and a user study demonstrate that Manga2Animate produces animations more faithful to the source manga and more consistent across shots than state-of-the-art approaches. Project page: https://annchenn.github.io/Manga2Animate/

## 1 INTRODUCTION

Manga is one of the richest sources of stories for animation. Popular titles are routinely adapted into anime, and fans expect these adaptations to move smoothly and tell the story coherently while remaining faithful to the characters and art style they love. Yet producing such an adaptation remains slow and costly, often taking large teams of artists months of storyboarding, key animation, inbetweening, and coloring.

![](images/c8abdb613d29a6d74c87f490fa40141235c16432cb61a33e2e54ba0f811484ca.jpg)  
Figure 2: Limitations of existing methods to manga-to-animation generation. (a) Text-based storytelling methods (AniMaker, ViMax, StoryMem) access the manga mainly through text, losing its art style ( 1 ) or distorting characters ( 2 ). (b) Feeding manga pages into a video model (e.g., Seedance 2.0) tells the wrong story ( 3 ) and shows manga panels as video frames ( 4 ).

At the same time, a manga already specifies much of what an animation needs, including who the characters are and what they look like, how each shot is framed, and how the plot unfolds. With recent advances in video generation, it is natural to ask whether a manga can be brought to life automatically. However, as Fig. 2 shows, storytelling systems that first describe the manga in words discard its visual appearance, so the original art style is lost ( 1 ) and characters change appearance or drift between anime and photorealistic styles ( 2 ). Feeding manga pages directly into a video model fails for a different reason. The model knows neither the order in which the panels should be read nor what speech balloons mean, so it tells the wrong story ( 3 ) or even places manga panels, or entire pages, directly into the video as frames ( 4 ).

Our key observation is that today’s video generation models already know how to animate. What they cannot do is understand a manga. Given a manga page, they produce smooth and appealing motion, yet they follow the panels in the wrong order and misread the speech balloons, so the result looks convincing but tells a completely wrong story (Fig. 2b). We therefore do not replace these models. Instead, we keep their rich visual knowledge and give them what they are missing, a clear understanding of the manga. Much like an animation studio, Manga2Animate (Fig. 1) first turns the manga into an animation script, which describes what happens in each panel, and a storyboard, which shows how each shot should look. Both are built by AI agents and checked against the original panels, so the video model receives a faithful plan instead of raw manga pages. We find that this plan effectively constrains generation. With the script fixing the plot and the storyboard fixing the characters, composition, and art style, the video model no longer has to interpret the manga on its own and can focus on what it does best, bringing the story to life.

Experiments show that Manga2Animate produces animations that are more faithful to the original characters, art style, and plot than existing storytelling and video generation methods, even when these methods are given the same script as ours. Because our framework guides existing models rather than replacing them, it works with both open-source and commercial image and video gener ators, and outperforms prior methods with either. In a user study, participants selected our results as the best overall in over 90% of responses, for both storyboards and videos.

In summary, our contributions are threefold:

• We present Manga2Animate, to the best of our knowledge the first automated framework that turns existing manga into animation that is faithful to source manga.

• We show that planning through a script and a storyboard, iteratively refined with duallevel feedback from both and verified against the original manga, is key to preserving the characters, art style, and plot.

• We establish an evaluation protocol for manga-to-animation generation, under which Manga2Animate outperforms existing methods, and we further validate these results with a user study.

## 2 RELATED WORK

Manga understanding. Built on annotated corpora (Matsui et al., 2017; Aizawa et al., 2020; Vivoli et al., 2024a; Li et al., 2024a), manga understanding has progressed from detecting panels, characters, and text (Shen et al., 2023) to ordering panels and attributing dialogue (Sachdeva & Zisserman, 2024), naming characters across chapters (Sachdeva et al., 2024), and narrating chapters as prose (Sachdeva & Zisserman, 2025). Vision-language models are now benchmarked on manga (Baek et al., 2026; Ikuta et al., 2025; Vivoli et al., 2025), yet still misread cross-panel narrative (Baranwal et al., 2025) and can rely on memorized popular content rather than visual evidence (Szu-Tu et al., 2025); generation from comics remains largely unexplored (Vivoli et al., 2024b). We use Sachdeva & Zisserman (2024) for pre-processing and, given this fragility, review every intermediate output against the source panels.

Story visualization. Story visualization renders a narrative as a consistent image sequence, evolving from GANs (Li et al., 2019) to diffusion models with visual memory (Rahman et al., 2023), identity conditioning (Ye et al., 2023), shared attention (Tewel et al., 2024; Zhou et al., 2024), frame context (Liu et al., 2024a), multi-subject personalization (He et al., 2025b; Wang et al., 2025b; Wu et al., 2025b), iterative refinement (Mao et al., 2026), and language-model layout planning (He et al., 2025a; Wang et al., 2025a; Cheng et al., 2024), with dedicated benchmarks (Zhuang et al., 2026; Gao et al., 2025) and extensions to manga generation (Wu et al., 2025a; Wang et al., 2026a). These methods generate from text, whereas our storyboard is conditioned on and selected against the manga panels.

Multi-shot video generation. Video diffusion has advanced from short clips (Ho et al., 2022) to foundation models (Kong et al., 2024; Wan et al., 2025; Yang et al., 2025; Polyak et al., 2024; Seedance et al., 2026), whose adherence to explicit specifications is now benchmarked (Huang et al., 2026d). Cross-shot consistency is pursued through long-context tuning (Guo et al., 2025; Xiao et al., 2026; Meng et al., 2026), keyframe memory (Zhang et al., 2025a; Zhou et al., 2026), storyboard anchoring (Zhang et al., 2026), and reference-based identity injection (Jiang et al., 2025; Liu et al., 2025a; Hu et al., 2025). For animation, anime-oriented models and datasets (Jiang et al., 2024; Qiu et al., 2025) and in-betweening and colorization methods (Xing et al., 2024; Meng et al., 2025; Huang et al., 2024; Li et al., 2025; Liu et al., 2025b; Tang et al., 2025) create motion and color from sparse inputs. We use such models as interchangeable backends and, in the spirit of decoupling explicit state from generative rendering (Huang et al., 2026c), obtain consistency from the reviewed script and storyboard rather than from architecture (Sec. 4.4).

Agentic storytelling video generation. Language and vision-language agents increasingly orchestrate video production from script to storyboard, clips, and assembly (Lin et al., 2023; Li et al., 2024b; Xu et al., 2025a; Wu et al., 2025c; Hu et al., 2024; Xu et al., 2025b; Wang et al., 2026b; Huang et al., 2026b), some with critics that select clips by tree search (Shi et al., 2025), gate anime production stages (Zhang et al., 2025b; Yan et al., 2026), or self-improve at test time (Long et al., 2026), following self-feedback in language agents (Madaan et al., 2023; Shinn et al., 2023), a pattern also adopted to verify generated images and figures (Huang et al., 2026a; Tseng et al., 2026). Our creator–reviewer design is closest to these systems, but they invent characters, shots, and plot from a prompt and can only check outputs against their own script, whereas our reviewers check against the manga panels. Earlier manga-to-animation systems only moved a virtual camera over panels or added synthesized speech (Cao et al., 2016; Gupta et al., 2021); to our knowledge, ours is the first to generate faithful multi-shot animation from manga pages.

![](images/0485f68bcc36301adecefea02744f048fe820ef32a86bbdece55e2c8bce8bb56.jpg)  
Figure 3: Overview of the Manga2Animate pipeline. (a) Pre-processing extracts ordered panels, a transcript, and character descriptions. (b) Creator and reviewer agents refine an animation script and a visual storyboard over N rounds and select the best script and best storyboard, dual-level feedback also guides the next round. (c) Video generation runs in first-frame or reference-conditioned mode.

## 3 METHODOLOGY

## 3.1 TASK FORMULATION

Given a set of manga pages $M = \{ m _ { 1 } , . . . , m _ { p } \}$ in their original reading order, our goal is to generate a coherent animated video $V .$ . Each manga page consists of multiple manga panels, where a panel is an individual visual frame depicting a particular moment, action, or scene. From the input pages, we obtain an ordered sequence of $\bar { N _ { p } }$ target manga panels $P = \{ p _ { 1 } , \dots , p _ { N _ { p } } \} ( N _ { p } = 6 $ in our experiments) and a transcript T. For prominent characters appearing in the selected manga panels, we additionally use character reference images R. These references are taken from official color artwork when available; otherwise, clear appearances of the characters are cropped from the input manga.

Our pipeline transforms these inputs through two intermediate representations: an animation script $S = \{ s _ { 1 } , \ldots , s _ { N _ { p } } \}$ and a visual storyboard $B = \{ b _ { 1 } , \ldots , b _ { N _ { p } } \}$ , before producing the final video $V ,$ as illustrated in Fig. 3. We also derive per-character descriptions D from the reference images R.

The use of S and B is inspired by real-world animation production workflows, where the narrative is first organized into a script and subsequently translated into a storyboard before animation is produced. Following this workflow, S serves as the textual specification of the animation. Each script clip $s _ { i }$ describes the content corresponding to manga panel $p _ { i } .$ , including the scene, camera, character positions and actions, expressions, and dialogue. The storyboard B then converts this textual specification into a visual plan, where each storyboard panel $b _ { i }$ specifies the appearance and scene composition of the corresponding animated clip.

Throughout the pipeline, we maintain an explicit correspondence between the source manga and the generated representations as illustrated in Fig. A1: manga panel $p _ { i }$ , script clip $s _ { i } .$ , and storyboard panel $b _ { i }$ all describe the same narrative unit. This alignment allows each generated component to be directly traced back to its source manga panel during review and refinement.

## 3.2 AGENTS

Manga2Animate uses two types of agents at each stage of the pipeline (Fig. 3).

![](images/c68881c7f2d2eea035df0533b12cc8f205521ddf1262ba9de7da6cc7708d3d28.jpg)  
Figure 4: Compositional (clip-wise) selection by the Script Reviewer. The reviewer selects the best character and clip description at each panel position across the $N _ { s }$ candidate scripts and reassembles them into one best script (left). For one panel (right), it flags inaccurate details (red) in two candidates and selects the one matching the panel (green).

$\mathcal { S }$ Creators. A creator takes the inputs of its stage and produces the intermediate output of that stage.

Reviewers. A reviewer reviews the outputs of its creator, selects the best one, and gives feedback for refinement. Full review criteria and prompts are listed in Appendices G and H.

## 3.3 PRE-PROCESSING

Manga pages often exhibit non-trivial panel layouts, making reading order an important prerequisite for preserving the narrative flow. We therefore detect the panels on each page and determine their reading order using a manga-specific detection model and its associated ordering algorithm (Sachdeva & Zisserman, 2024). The resulting ordered panel sequence is then passed to the subsequent stages of the pipeline. In parallel, the same model also detects and transcribes the text on these pages, which we merge into a single transcript spanning all pages to help the Script Creator write each panel’s dialogue and plot (Sec. 3.4). Also, we use a Character Description Creator to establish a consistent representation of each character from the reference image R. It extracts relatively stable, identity-defining traits, such as facial features and body build, while excluding variable attributes such as clothing and accessories. These character descriptions are then provided to the downstream Script and Storyboard Creators to help maintain character consistency throughout the generation process.

## 3.4 SCRIPT AND STORYBOARD REFINEMENT

To improve generation reliability while controlling API costs, we combine a small number of candidates per stage with iterative refinement, in which reviewer feedback guides the next round of generation. We run N = 2 refinement rounds: the first round explores initial candidates, and the second round revises them using the first-round feedback.

In each refinement round, the Script Creator generates $N _ { s }$ script candidates $( N _ { s } = 3 )$ conditioned on the manga panels, character descriptions, transcript, and, from the second round onward, the feedback from the previous round. Errors in scripts are largely local, a candidate may describe one panel precisely yet misread another, and different candidates tend to make different mistakes. Selecting a single script as a whole would discard the correct descriptions contained in the other candidates. The Script Reviewer therefore performs compositional selection at the clip level (Fig. 4), preserving complementary strengths across candidates, and records the remaining weaknesses of each selected component as feedback for the next round.

The Storyboard Creator then generates $N _ { b }$ storyboard candidates $( N _ { b } = 3 )$ for the selected script. Unlike scripts, closed-source storyboards are selected as a complete image because their panels are generated jointly and share style, color, and lighting; recombining panels across candidates could introduce visual inconsistencies. The Storyboard Reviewer compares candidates pairwise in the first round and, in later rounds, compares each new candidate with the previous best, judging each storyboard against the script that produced it. If no new candidate wins, the previous best storyboard is retained with its script and feedback, keeping the final pair consistent. The open-source backend instead generates and selects panels individually before compositing them due to its constraints (more details in Appendix B).

Script-level and storyboard-level feedback are jointly provided to the next-round Script Creator, together with the best script from the previous round and the source manga panels. This duallevel feedback allows the pipeline to refine the textual specification using evidence from its visual realization. Issues flagged by the Storyboard Reviewer are treated as symptoms of ambiguous or inaccurate script descriptions: the Script Creator is instructed to identify the responsible part of the script, such as an ambiguous spatial description or a missing detail. To prevent unnecessary changes to already correct content, the Script Creator revises only the script clips or character descriptions explicitly identified as problematic by the reviewers, while leaving all other entries unchanged. When feedback from different reviewers conflicts, the corresponding source manga panel is used as the primary reference for deciding the revision. The script and storyboard selected in the final round are then passed to the video-generation stage.

## 3.5 VIDEO GENERATION PIPELINE

To accommodate different input formats of video generation models, our video generation pipeline supports two modes: a first-frame mode and a reference-conditioned mode. The former is used for models that take a single image as the first frame, while the latter is used for models that can take the complete storyboard as a reference and generate a multi-shot video at once. Our main experimental results use the reference-conditioned mode.

In first-frame mode, each storyboard panel serves as the first frame of its corresponding clip. For each clip, the Video Creator generates $N _ { v }$ candidates $( N _ { v } = 2 )$ conditioned on that panel and its script description, and the Video Reviewer selects the better one using the script, panel, and previously selected clip when available. This greedy and clip-by-clip process enforces cross-clip continuity at selection time, after which the selected clips are concatenated in story order. In referenceconditioned mode, the Video Creator instead uses the complete best script and full storyboard to generate $N _ { v }$ complete multi-shot candidates in one pass, from which the Video Reviewer selects the final result.

## 4 EXPERIMENTS

## 4.1 DATASETS

Our evaluation set contains eight stories of six ordered manga panels, drawn from Manga109 (Matsui et al., 2017; Aizawa et al., 2020) and additional cases covering diverse backgrounds, characters, interactions, and actions. All methods use the same stories, enabling evaluation of both panel-level fidelity and cross-clip consistency.

## 4.2 BASELINES

We use StoryDiffusion (Zhou et al., 2024), Story-Iter (Mao et al., 2026), and AnyStory (He et al., 2025b) as storyboard-generation baselines, and AniMaker (Shi et al., 2025), ViMax (Huang et al., 2026b), and StoryMem (Zhang et al., 2025a) as video-generation baselines. We further compare with direct manga-to-video generation by feeding the manga pages directly into Seedance 2.0 or MiniMax-H3. To compare generation ability rather than manga understanding, every script-based baseline receives the same refined script as Manga2Animate. We also provide the same manga character reference images whenever the baseline accepts reference images. Full adaptation details are in Appendix E.

## 4.3 EVALUATION METRICS

MangaEval adapts existing metrics to the manga input and multi-shot output, evaluating storyboard Visual Consistency, Content Alignment, Aesthetic Quality, and Cost, as well as video Manga Alignment, Video Quality, Aesthetic Quality, and Cost. Full definitions and objectives are in Appendix F (Tabs. A6 and A7).

Video Evaluation Metrics. At the video level, we evaluate I2V Subject (I2VS), I2V Background (I2VB), Subject Consistency (SC), Background Consistency (BC), Motion Smoothness (MS), and Dynamic Degree (DD) segment by segment against the source panels. We replace

![](images/faebdf270ae1ebee19213fca42786debb9e7c64cd94e4d4197796b6f8c07ff77.jpg)

Table 1: Quantitative comparison of multi-shot video generation. Our main setting (Gemini + Seedance 2.0) performs better than AniMaker, ViMax, and StoryMem in I2VS, I2VB, BC, Shadowv2, and Waifu-v3. For a fair comparison with ViMax, both methods use the same script, image model, and video model. Our FLUX.2 + MiniMax-H3 setting also performs well, showing that the improvement is not tied to one model combination. Direct generation has higher DD, but it may animate the page layout or produce the wrong actions (Fig. 2(b)); a higher DD means more movement, not more accurate movement. Red , orange , and yellow mark the top three results; ↑/↓ show whether higher or lower is better. Call counts are not ranked; – means no calls.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="2">Alignment</td><td colspan="3">Video Quality</td><td colspan="2">Aesthetic Quality</td><td colspan="3">Cost</td></tr><tr><td></td><td>I2VS↑ I2VB↑</td><td>SC↑ BC↑</td><td>MS↑</td><td>DD↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td><td>Calls (VLM)</td><td>Calls (Video Gen)</td><td>Time↓</td></tr><tr><td>AniMaker</td><td></td><td>0.775</td><td>0.749</td><td>0.922 0.950</td><td>0.979</td><td>7.680</td><td>0.410</td><td>2.694</td><td>82.500</td><td></td><td>8h9m</td></tr><tr><td>ViMax</td><td>Veo 3.1 Fast</td><td>0.774</td><td>0.752</td><td>0.881 0.932</td><td>0.978</td><td>1.639</td><td>0.368</td><td>4.394</td><td>15.125</td><td>6.000</td><td>12m18s</td></tr><tr><td>ViMax</td><td>Seedance 2.0</td><td>0.753</td><td>0.747</td><td>0.927 0.948</td><td>0.977</td><td>2.541</td><td>0.385</td><td>4.764</td><td>15.125</td><td>6.000</td><td>16m33s</td></tr><tr><td>ViMax</td><td>MiniMax-H3</td><td>0.771</td><td>0.767</td><td>0.938 0.957</td><td>0.977</td><td>0.977</td><td>0.326</td><td>4.505</td><td>15.500</td><td>1</td><td>28m58s</td></tr><tr><td>StoryMem</td><td></td><td>0.728</td><td>0.731</td><td>0.885 0.922</td><td>0.956</td><td>5.203</td><td>0.275</td><td>3.602</td><td>10.300</td><td>一</td><td>2h16m8s</td></tr><tr><td>Manga → Video (Direct)</td><td>MiniMax-H3</td><td>0.777</td><td>0.735</td><td>0.865 0.932</td><td>0.953</td><td>11.787</td><td>0.402</td><td>4.611</td><td>=</td><td>一</td><td>5m54s</td></tr><tr><td>Manga → Video (Direct)</td><td>Seedance 2.0</td><td>0.750</td><td>0.711</td><td>0.852 0.931</td><td>0.951</td><td>13.941</td><td>0.492</td><td>5.250</td><td>=</td><td>1.000</td><td>3m29s</td></tr><tr><td>Ours (FLUX.2)</td><td>MiniMax-H3</td><td>0.794</td><td>0.768</td><td>0.943 0.953</td><td>0.968</td><td>6.872</td><td>0.549</td><td>4.054</td><td>59.000</td><td></td><td>2h8m39s</td></tr><tr><td>Ours (FLUX.2)</td><td>Seedance 2.0</td><td>0.781</td><td>0.741</td><td>0.8920.942</td><td>0.958</td><td>9.377</td><td>0.595</td><td>4.429</td><td>59.000</td><td>2.000</td><td>2h5m46s</td></tr><tr><td>Ours (Gemini)</td><td>MiniMax-H3</td><td>0.792</td><td>0.791</td><td>0.940 0.953</td><td>0.969</td><td>3.715</td><td>0.413</td><td>4.352</td><td>29.500</td><td></td><td>27m5s</td></tr><tr><td>Ours (Gemini)</td><td>Seedance 2.0</td><td>0.791</td><td>0.768</td><td>0.931 0.958</td><td>0.960</td><td>9.457</td><td>0.518</td><td>5.090</td><td>29.500</td><td>2.000</td><td>24m49s</td></tr></table>

![](images/6e4357361e49d9163b9cd7bda9947404fb7ea6dbbf9cf0fc831ebcc914ce36ee.jpg)  
(a)

![](images/0e85c0a14470d5eee0f2b6815eafff7a8781dea23b2f17a704c70337b8935035.jpg)  
(b)  
Figure 5: Qualitative comparison of storyboard generation on two manga examples (source pages on the left). Manga2Animate keeps the source art style, character identity, and plot across all six panels. AniMaker and AnyStory sometimes leave panels uncolored, StoryDiffusion uses a generic illustration style and repeats content, and Story-Iter becomes photorealistic and changes character appearance. In (b), the baselines also show the falling winged creature incorrectly or leave it out. Red boxes mark example failures.

VBench++’s (Huang et al., 2025) DINO (Caron et al., 2021) with DINOv3 (Siméoni et al., 2025) for I2VS and its binary DD with a continuous motion-magnitude score; aesthetics are evaluated on sampled frames with two anime-specific models.

Detailed storyboard evaluation metrics are provided in Appendix F (Table A6).

## 4.4 QUANTITATIVE COMPARISONS

Overall, Table 1 shows that our pipeline better preserves the manga content and keeps the video clips visually consistent. The improvement remains when we change the image or video model, suggesting that it comes from the script, storyboard, and review process rather than one specific model. DD only measures how much movement a video contains, so a higher DD does not mean that the movement is correct. Storyboard results are reported in Appendix C.

## 4.5 QUALITATIVE ANALYSIS

Storyboard Qualitative Results. Figure 5 shows the benefit of generating and reviewing storyboards with the source panels instead of relying only on the shared refined script. Figure 4 shows how errors are corrected at the script stage.

Video Qualitative Results. As Fig. 6 shows, the reviewed storyboard provides appearance and composition that match the manga, while the script provides actions and dialogue. Together, they

![](images/6468ca483fb6dc23843d3379b0056a9aed6a5c255da8d1ce777c1f91688eeb2e.jpg)

![](images/a1537f700d89d2e64cb746dbaa6c661db5a312008675113ccd36ae9d2041935d.jpg)

(a)  
![](images/6e935514d8cfb7f9397e51a8e1afe828a6289fa1cf3dfd03fee3bc01592f4e5a.jpg)  
(b)  
Figure 6: Qualitative comparison of video generation on two manga examples (source pages on the left). Manga2Animate keeps the manga’s content, characters, and art style. AniMaker adds unrelated people or backgrounds, ViMax mixes anime and photorealistic styles and adds text, and StoryMem produces blurry frames and mistakes the basketball scene for tennis. Baselines may also switch to black-and-white frames or change characters within a video; red boxes mark example failures.

![](images/08007744b3faabfe86f10dfb3aa15a13c84a4c20575f1ef63019ff8c9330bf76.jpg)  
(a) Storyboard

![](images/b76fb7ebba920c87a9447c40f5b177c5a5cbe35b7c0a57ed7877ce0a9229c214.jpg)  
(b) Video  
Figure 7: User study on storyboard and video generation. Thirty-seven participants evaluated eight storyboard and eight video cases shown with the input manga; method names were hidden and outputs randomized. For each criterion, participants selected the best result. Manga2Animate achieves the highest preference on every storyboard criterion (Character Identity, Manga Plot, Absence of AI Hallucination, and Overall Performance) and every video criterion (the same four plus Visual Comfort and Consistency).

reduce incorrect content even though all methods receive the same refined script. Full videos are in the supplementary material.

Table 2: Effect of removing each stage-specific reviewer. When a reviewer is removed, that stage generates only one candidate and receives no review or feedback; all other settings stay the same. Removing any reviewer lowers the overall alignment and consistency scores. Removing the Script Reviewer causes the largest drop in I2VS. Some variants score higher on MS or Shadow-v2, but score lower on alignment, consistency, or DD. Red and orange mark the best and second-best results; higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">Alignment</td><td colspan="4">Video Quality</td><td colspan="2">Aesthetic Quality</td></tr><tr><td>I2VS↑</td><td>I2VB↑</td><td>SC↑</td><td>BC↑</td><td>MS↑</td><td>DD↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td></tr><tr><td>Without Script Reviewer</td><td>0.766</td><td>0.752</td><td>0.912</td><td>0.949</td><td>0.963</td><td>8.592</td><td>0.523</td><td>4.789</td></tr><tr><td>Without Storyboard Reviewer</td><td>0.783</td><td>0.756</td><td>0.923</td><td>0.948</td><td>0.971</td><td>5.616</td><td>0.479</td><td>4.540</td></tr><tr><td>Without Video Reviewer</td><td>0.768</td><td>0.749</td><td>0.902</td><td>0.941</td><td>0.957</td><td>8.330</td><td>0.517</td><td>4.996</td></tr><tr><td>Full Model</td><td>0.791</td><td>0.768</td><td>0.931</td><td>0.958</td><td>0.960</td><td>9.457</td><td>0.518</td><td>5.090</td></tr></table>

## 4.6 USER STUDY

As shown in Fig. 7, the strong preferences for Manga Plot and Character Identity agree in direction with MangaEval’s fidelity metrics (e.g., PA and CIDS), although the human and automatic criteria do not correspond one-to-one.

## 4.7 ABLATION STUDIES

We ablate the three stage-specific reviewers, compare clip-wise with whole-script selection, remove each level of feedback, ablate the refinement loop and storyboard generation, and evaluate video generation mode, using the main configuration as reference.

Stage-Specific Reviewers. Table 2 shows that all three reviewers contribute to the full model’s balanced performance.

Script Selection Strategy. Clip-wise selection improves alignment, consistency, and aesthetics, with only small decreases in I2VB and MS. Full results are in Appendix D (Table A2), and Fig. 4 shows an example.

Dual-Level Feedback. Removing either script- or storyboard-level feedback lowers CSD, CIDS, and PA, showing that the two are complementary. Full results are in Appendix D (Table A3).

Refinement Loop and Storyboard Generation. Skipping the feedback refinement or the storyboard generation step both lower video quality relative to the full model. Full results are in Appendix D (Table A4).

Video Generation Mode. These results show that Manga2Animate can use either mode. We use reference-conditioned generation for our main results; full results are in Appendix D (Table A5).

## 5 CONCLUSION

We present Manga2Animate, a multi-agent framework for faithfully adapting existing manga into coherent multi-shot animation. We decompose the task into two intermediate representations grounded in the source manga, an animation script and a visual storyboard. Creator and reviewer agents iteratively generate, review, and refine both. Since the two representations fail in different ways, we compose scripts clip by clip, select storyboards as a whole, and feed storyboard-level issues back to the script. Under MangaEval, an evaluation protocol adapted from ViStoryBench and VBench++, Manga2Animate outperforms state-of-the-art storyboard and video generation methods on manga-fidelity and consistency metrics, even when baselines receive the same refined script and character references, and is strongly preferred in a user study. Promising future directions include improving robustness to complex or roughly sketched art styles, developing source-grounded evaluation of story events, and extending narrative understanding across pages and chapters.

Limitations. Our method has several limitations. First, as an agentic pipeline built on stochastic generative models, its outputs may vary across runs. Second, API-based models may restrict generation of copyrighted or policy-sensitive content. Finally, when the source manga is highly sketchy, cluttered, or compositionally complex, or a scene requires context from elsewhere in the story, the pipeline may misinterpret panel content.

## REFERENCES

Kiyoharu Aizawa, Azuma Fujimoto, Atsushi Otsubo, Toru Ogawa, Yusuke Matsui, Koki Tsubota, and Hikaru Ikuta. Building a manga dataset" manga109" with annotations for multimedia applications. arXiv preprint arXiv:2005.04425, 2020.

Jeonghun Baek, Kazuki Egashira, Shota Onohara, Atsuyuki Miyai, Yuki Imajuku, Hikaru Ikuta, and Kiyoharu Aizawa. Mangavqa and mangalmm: A benchmark and specialized model for multimodal manga understanding. In Findings of the Association for Computational Linguistics: EACL 2026, pp. 5349–5370, 2026.

Aaditya Baranwal, Madhav Kataria, Naitik Agrawal, Yogesh S Rawat, and Shruti Vyas. Re: Versecan your vlm read a manga? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3820–3830, 2025.

Ying Cao, Xufang Pang, Antoni B Chan, and Rynson WH Lau. Dynamic manga: Animating still manga via camera movement. IEEE Transactions on Multimedia, 19(1):160–172, 2016.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 9630–9640. IEEE, 2021.

Junhao Cheng, Baiqiao Yin, Kaixin Cai, Minbin Huang, Hanhui Li, Yuxin He, Xi Lu, Yue Li, Yifei Li, Yuhao Cheng, et al. Theatergen: Character management with llm for consistent multi-turn image generation. arXiv preprint arXiv:2404.18919, 2024.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.

Silin Gao, Sheryl Mathew, Li Mi, Sepideh Mamooler, Mengjie Zhao, Hiromi Wakaki, Yuki Mitsufuji, Syrielle Montariol, and Antoine Bosselut. Vinabench: Benchmark for faithful and consistent visual narratives. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2870–2879. IEEE, 2025.

Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17281–17291, 2025.

Vaibhavi Gupta, Vinay Detani, Vivek Khokar, and Chiranjoy Chattopadhyay. C2vnet: A deep learning framework towards comic strip to audio-visual scene synthesis. In International Conference on Document Analysis and Recognition, pp. 160–175. Springer, 2021.

Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. Dreamstory: Open-domain story visualization by llmguided multi-subject consistent diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.

Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anystory: Towards unified single and multiple subject personalization in text-to-image generation. arXiv preprint arXiv:2501.09503, 2025b.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633– 8646, 2022.

Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. Storyagent: Customized storytelling video generation via multi-agent collaboration. arXiv preprint arXiv:2411.04925, 2024.

Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025.

Jing-En Huang, I Fang, Tzuhsuan Huang, Yu-Lun Liu, Chih-Yu Wang, Jun-Cheng Chen, et al. Genn-val: Agentic image data generation and validation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8786–8795, 2026a.

Lingxuan Huang, Sizhe He, Hengji Zhou, Liqiang Nie, Lianghao Xia, and Chao Huang. Vimax: Agentic video generation. arXiv preprint arXiv:2606.07649, 2026b.

Zheng-Hui Huang, Guixu Lin, Jiacheng Lin, Yi-Chuan Huang, Ruihan Yu, Muyao Niu, Siqi Yang, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang, et al. Programmable world model. arXiv preprint arXiv:2609.10540, 2026c.

Zheng-Hui Huang, Guixu Lin, Yu-Ju Tsai, Jian-Kai Zhu, Fengbo Lan, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang, and Zhixiang Wang. Prowbench: Do video models render what the program specifies? arXiv preprint arXiv:2610.02205, 2026d.

Zhitong Huang, Mohan Zhang, and Jing Liao. Lvcd: reference-based lineart video colorization with diffusion models. ACM Transactions on Graphics (TOG), 43(6):1–11, 2024.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Hikaru Ikuta, Leslie Wöhler, and Kiyoharu Aizawa. Mangaub: A manga understanding benchmark for large multimodal models. IEEE MultiMedia, 32(2):33–43, 2025.

Yudong Jiang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Bingwen Zhu, Xinwen Zhang, et al. Anisora: Exploring the frontiers of animation video generation in the sora era. arXiv preprint arXiv:2412.10255, 2024.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17191–17202. IEEE, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

Lingen Li, Guangzhi Wang, Zhaoyang Zhang, Yaowei Li, Xiaoyu Li, Qi Dou, Jinwei Gu, Tianfan Xue, and Ying Shan. Tooncomposer: Streamlining cartoon production with generative postkeyframing. arXiv preprint arXiv:2508.10881, 2025.

Yingxuan Li, Kiyoharu Aizawa, and Yusuke Matsui. Manga109dialog: A large-scale dialogue dataset for comics speaker detection. In 2024 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6. IEEE, 2024a.

Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6322– 6331. IEEE, 2019.

Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. Anim-director: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024b.

Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9801–9810. IEEE, 2023.

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023.

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6190–6200. IEEE, 2024a.

Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14951–14961. IEEE, 2025a.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Springer, 2024b.

Zhiheng Liu, Ka Leong Cheng, Xi Chen, Jie Xiao, Hao Ouyang, Kai Zhu, Yu Liu, Yujun Shen, Qifeng Chen, and Ping Luo. Manganinja: Line art colorization with precise reference following. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5666– 5677. IEEE, 2025b.

Do Xuan Long, Xingchen Wan, Hootan Nakhost, Chen-Yu Lee, Tomas Pfister, and Sercan Ö Arik. Vista: A test-time self-improving video generation agent. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6021–6032, 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

Jiawei Mao, Xiaoke Huang, Yunfei Xie, Yuanqi Chang, Mude Hui, Bingjie Xu, Zeyu Zheng, Zirui Wang, Cihang Xie, and Yuyin Zhou. Story-iter: A training-free iterative paradigm for long story visualization. In International Conference on Learning Representations, volume 2026, pp. 113143–113173, 2026.

Yusuke Matsui, Kota Ito, Yuji Aramaki, Azuma Fujimoto, Toru Ogawa, Toshihiko Yamasaki, and Kiyoharu Aizawa. Sketch-based manga retrieval using manga109 dataset. Multimedia tools and applications, 76(20):21811–21838, 2017.

Yihao Meng, Hao Ouyang, Hanlin Wang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Zhiheng Liu, Yujun Shen, and Huamin Qu. Anidoc: Animation creation made easier. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18187–18197. IEEE, 2025.

Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Shuailei Ma, Yixuan Li, Cheng Chen, et al. Holocine: Holistic generation of cinematic multishot long video narratives. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 461–471, 2026.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Lu Qiu, Yizhuo Li, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Animeshooter: A multi-shot animation dataset for reference-guided video generation. arXiv preprint arXiv:2506.03126, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2493–2502. IEEE, 2023.

Ragav Sachdeva and Andrew Zisserman. The manga whisperer: Automatically generating transcriptions for comics. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12967–12976. IEEE, 2024.

Ragav Sachdeva and Andrew Zisserman. From panels to prose: Generating literary narratives from comics. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21864– 21873. IEEE, 2025.

Ragav Sachdeva, Gyungin Shin, and Andrew Zisserman. Tails tell tales: Chapter-wide manga transcriptions with character names. In Asian Conference on Computer Vision, pp. 63–80. Springer, 2024.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

Conghao Tom Shen, Violet Yao, and Yixin Liu. Maru: A manga retrieval and understanding system connecting vision and language. arXiv preprint arXiv:2311.02083, 2023.

Haoyuan Shi, Yunxin Li, Xinyu Chen, Longyue Wang, Baotian Hu, and Min Zhang. Animaker: Multi-agent animated storytelling with mcts-driven clip generation. In Proceedings of the SIG-GRAPH Asia 2025 Conference Papers, pp. 1–11, 2025.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

Li-Zhong Szu-Tu, Ting-Lin Wu, Chia-Jui Chang, He Syu, and Yu-Lun Liu. Beyond memorization: A multi-modal ordinal regression benchmark to expose popularity bias in vision-language models. arXiv preprint arXiv:2512.21337, 2025.

Yunlong Tang, Junjia Guo, Pinxin Liu, Zhiyuan Wang, Hang Hua, Jia-Xing Zhong, Yunzhong Xiao, Chao Huang, Luchuan Song, Susan Liang, et al. Generative ai for cel-animation: A survey. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 3837– 3850. IEEE, 2025.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pp. 402–419. Springer, 2020.

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4): 1–18, 2024.

Shih-Chen Tseng, Chih-Hsuan Chen, Ryan Yang, Hsi-An Chen, Chun-Wei Tuan Mu, and Yu-Lun Liu. One figure, every canvas: Editable flowchart relayout via agentic pipeline. arXiv preprint arXiv:2610.06852, 2026.

Emanuele Vivoli, Marco Bertini, and Dimosthenis Karatzas. Comix: A comprehensive benchmark for multi-task comic understanding. Advances in Neural Information Processing Systems, 37: 140828–140846, 2024a.

Emanuele Vivoli, Mohamed Ali Souibgui, Andrey Barsky, Artemis LLabres, Marco Bertini, and Dimosthenis Karatzas. One missing piece in vision and language: A survey on comics understanding. arXiv preprint arXiv:2409.09502, 2024b.

Emanuele Vivoli, Artemis Llabrés, Mohamed Ali Souibgui, Marco Bertini, Ernest Valveny Llobet, and Dimosthenis Karatzas. Comicspap: understanding comic strips by picking the correct panel. In International Conference on Document Analysis and Recognition, pp. 337–350. Springer, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Muyao Wang, Zeke Xie, Yanhao Chen, Lixin Xiu, and Hideki Nakayama. Mangaflow: An end-to-end agentic framework for controllable story to manga generation. arXiv preprint arXiv:2605.28173, 2026a.

Qian Wang, Ziqi Huang, Ruoxi Jia, Paul Debevec, and Ning Yu. Mavis: A multi-agent framework for long-sequence video storytelling. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2273– 2295, 2026b.

Wen Wang, Canyu Zhao, Hao Chen, Zhekai Chen, Kecheng Zheng, and Chunhua Shen. Autostory: Generating diverse storytelling images with minimal human efforts. International Journal of Computer Vision, 133(6):3083–3104, 2025a.

Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. In International Conference on Learning Representations, volume 2025, pp. 95118–95146, 2025b.

Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong. Diffsensei: Bridging multi-modal llms and diffusion models for customized manga generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28684–28693. IEEE, 2025a.

Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18682–18692. IEEE, 2025b.

Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent cot planning. arXiv preprint arXiv:2503.07314, 2025c.

Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. In International Conference on Learning Representations, volume 2026, pp. 83349–83375, 2026.

Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. ACM Transactions on Graphics (TOG), 43 (6):1–11, 2024.

Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, and Mengyue Wu. Mm-storyagent: Immersive narrated storybook video generation with a multiagent paradigm across text, image and audio. arXiv preprint arXiv:2503.05242, 2025a.

Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, and Min Zhang. Filmagent: A multi-agent framework for end-to-end film automation in virtual 3d spaces. arXiv preprint arXiv:2501.12909, 2025b.

Hailong Yan, Shice Liu, Tao Wang, Xiangtao Zhang, Yijie Zhong, Jinwei Chen, Le Zhang, and Bo Li. Animeagent: Is the multi-agent via image-to-video models a good disney storytelling artist? arXiv preprint arXiv:2602.20664, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11941–11952. IEEE, 2023.

Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025a.

Lisai Zhang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Yuxin Hong, Zihao Zhang, et al. Anime: Adaptive multi-agent planning for long animation generation. In Proceedings ofthe SIGGRAPH Asia 2025 Posters, pp. 1–3. 2025b.

Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. Stage: Storyboardanchored generation for cinematic multi-shot narrative. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 659–669, 2026.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu, Bolan Su, and Ying-cong Chen. Videomemory: Toward consistent video generation via memory integration. arXiv preprint arXiv:2601.03655, 2026.

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024.

Cailin Zhuang, Ailin Huang, Yaoqi Hu, Jingwei Wu, Wei Cheng, Jiaqi Liao, Hongyuan Wang, Xinyao Liao, Weiwei Cai, Hengyuan Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9455–9467, 2026.

## APPENDIX

This appendix provides additional details that complement the main paper. Sec. A describes the implementation details, including model versions, hyperparameters, video specifications, hardware, and cost. Sec. B explains the panel-wise generation strategy of the open-source storyboard backend, Sec. C reports the storyboard quantitative evaluation, Sec. D gives the full script-selection-strategy ablation, the dual-level feedback ablation, an ablation on the refinement loop and storyboard generation, and a comparison of reference-conditioned and first-frame video generation, Sec. E describes how each baseline is adapted to manga inputs, Sec. F gives the full definitions of the MangaEval metrics and our modifications, Sec. G summarizes the criteria, scoring, and selection rules of every reviewer, and Sec. H lists the full prompts of all agents. The correspondence among panels, script clips, storyboard panels, and video clips (Sec. 3.1) is illustrated in Fig. A1. We further provide an HTML page in the supplementary material showcasing our generated videos.

## A IMPLEMENTATION DETAILS

![](images/6f8f89ab29a9322bd22af967ab847ef63e6474dad8ef9926d9d64128d37a476e.jpg)  
Figure A1: Correspondence among intermediate representations. Each manga panel (left) corresponds to one clip in the animation script (middle), which describes the environment, character positions, actions, and expressions of that panel in a structured JSON format. Each script clip is then rendered as one storyboard panel (right). In first-frame mode, that panel conditions the corresponding video clip directly; in reference-conditioned mode, the complete storyboard jointly conditions the multi-shot video.

Models. The Character Description Creator, Script Creator, Character Reviewer, Clip Reviewer, Storyboard Reviewer, and open-source Panel Reviewer use gemini-3.7-flash with the thinking level set to HIGH; the Video Reviewer uses gemini-3.1-pro-preview. The closed-source Storyboard Creator uses gemini-3.1-flash-image and outputs a 16:9 storyboard at 1376 × 768 in a 3 × 2 grid. We keep the default sampling parameters, including temperature, for all Gemini calls. Panel detection, reading-order estimation, and OCR use MAGI (Sachdeva & Zisserman, 2024). The open-source Storyboard Creator uses FLUX.2-dev with 28 inference steps and a guidance scale of 4.0. It generates three candidates per panel at 768 × 640, and its panel reviewer, also gemini-3.7-flash, scores each candidate independently and keeps the highest-scoring one per panel. All experiments use N = 2 refinement rounds, $N _ { s } = 3 ,$ ${ N _ { b } } ^ { - } = 3 ,$ , and $N _ { v } = 2$

Video generation. In the reference-conditioned mode, the video model receives the complete best script as a text prompt and the full storyboard grid as a single reference image. The closed-source pipeline uses Seedance 2.0 (dreamina-seedance-2-0-260128) in this mode and produces 15 s videos at 1280 × 720 and 24 fps, with generated audio. The open-source video-generation configuration supports two video backends. MiniMax-H3 (Ref2VA) runs in the same reference-conditioned mode and produces 15 s videos at 1344 × 768 and 24 fps with audio. Index-AniSora V3.2 (Jiang et al., 2024) runs in the first-frame mode: each cropped storyboard panel is placed as the first frame of its clip, and each clip has 81 frames at 16 fps and 1280 × 720. Cropped panels are passed to the video model without resizing or padding, and any aspect-ratio adaptation is handled by the video model’s own preprocessing. In all cases, two candidate videos are generated and the Video Reviewer selects the better one; when the reviewer judges the two candidates comparable, the first candidate is kept. All main quantitative comparisons in Table 1 use the reference-conditioned mode; Table A5 separately compares the reference-conditioned and first-frame modes.

Hardware and cost. Gemini and Seedance calls go through hosted APIs. MAGI runs locally on an NVIDIA RTX A6000 (48 GB). FLUX.2-dev runs on an NVIDIA A100 80GB PCIe, MiniMax-H3 on two NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) GPUs, and Index-AniSora V3.2 on a single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). For each story, the closed-source pipeline makes 34.5 calls on average in the script and storyboard stages (28.5 Gemini text calls and 6 image-generation calls) and 3 in the video stage (2 Seedance generations and 1 Gemini review), costing about US\$5.65 per story in total, of which US\$4.55 is Seedance. The costs in Table A1 cover the script and storyboard stages, whereas the VLM calls and time in Table 1 are counted end-to-end, covering script generation, storyboard generation, video generation, and all reviews, and Calls (Video Gen) counts video-generation calls only. For locally run image and video backends, no external generation API calls are required, although the creator and reviewer agents still use Gemini APIs. Their local computational cost is reflected in the reported wall-clock time. Wall-clock times depend on hardware, parallelism, and API latency and should be regarded as indicative.

## B OPEN-SOURCE STORYBOARD BACKEND

In our experiments, the open-source image model we tested (FLUX.2-dev) did not reliably generate a complete, high-quality multi-panel storyboard in a single pass. We therefore adopt a panel-wise generation strategy, producing multiple candidates for each panel and selecting the best one before composing the final storyboard. For each panel in the selected script, we construct a prompt that incorporates the character descriptions, the corresponding script content, and constraints derived from the original manga panel to preserve its composition, camera viewpoint, and character pose. When character reference crops are available, they are explicitly associated with the corresponding characters during generation.

FLUX.2-dev is conditioned on the source manga panel and the available character references to generate multiple candidates for each panel. A Gemini-based reviewer then scores each candidate independently on manga/script follow (the primary criterion), character identity consistency, art style and colorization quality, and background fidelity, each on a 0–4 scale, and the highest-scoring candidate is retained. The selected panels are then assembled into a $3 \times 2$ storyboard grid, following the same output format as the closed-source backend to enable evaluation under a unified protocol. Since every panel is conditioned on its source manga panel and the shared character references, these common conditions help maintain cross-panel consistency.

## C STORYBOARD QUANTITATIVE EVALUATION

Evaluation metrics. We evaluate storyboards with CSD, CIDS, PA, and OCCM from ViStory Bench (Zhuang et al., 2026), adapting three of them to manga: CIDS uses SigLIP (Zhai et al., 2023) instead of CLIP (Radford et al., 2021) for finer character-crop detail, OCCM’s expected character count comes from the source manga panel rather than the script’s cast list, and PA is judged by GPT-4.1 instead of Gemini (our pipeline’s reviewer) to reduce self-preference bias. Since all script-conditioned methods share the same refined script, PA measures content adherence under an identical specification, while CSD, CIDS, and OCCM (and the Manga Plot criterion of our user study) capture fidelity to the source manga itself; full definitions are in Appendix F. For Aesthetic Quality, we replace ViStoryBench’s general-purpose predictor with two anime-specific models, Aesthetic Shadow $\mathbf { v } 2 ^ { 1 }$ and $\mathrm { W a i f u } \mathrm { - } \mathrm { v } 3 ^ { 2 }$ , trained on anime/illustration imagery and used for training-data curation in production anime models.

Table A1: Quantitative comparison of storyboard generation. Red , orange , and yellow cells denote the best, second-best, and third-best results; ↑/↓ indicate that higher/lower is better. Call counts are not ranked; – denotes no calls of that type.
<table><tr><td rowspan="2">Method</td><td colspan="2">V.C.</td><td colspan="2">C.A.</td><td colspan="2">A.Q.</td><td colspan="3">Cost</td></tr><tr><td>CSD↑</td><td>CIDS↑</td><td>PA↑</td><td>OCCM↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td><td>Calls (VLM)</td><td>Calls (Img Gen)</td><td>Time↓</td></tr><tr><td>StoryDiffusion</td><td>0.509</td><td>0.713</td><td>2.05</td><td>57.5</td><td>0.697</td><td>5.789</td><td></td><td></td><td>49s</td></tr><tr><td>Story-Iter</td><td>0.486</td><td>0.652</td><td>1.85</td><td>66.3</td><td>0.615</td><td>6.308</td><td>60.000</td><td>11.000</td><td>13m20s</td></tr><tr><td>AnyStory</td><td>0.605</td><td>0.792</td><td>2.66</td><td>69.3</td><td>0.606</td><td>3.781</td><td>10.000</td><td>1.000</td><td>2m7s</td></tr><tr><td>AniMaker</td><td>0.508</td><td>0.797</td><td>3.34</td><td>67.5</td><td>0.448</td><td>4.522</td><td>82.500</td><td>18.000</td><td>8h9m</td></tr><tr><td>Ours (FLUX.2)</td><td>0.622</td><td>0.797</td><td>3.14</td><td>73.5</td><td>0.552</td><td>3.423</td><td>58.000</td><td></td><td>1h59m</td></tr><tr><td>Ours (Gemini 3.1 Image)</td><td>0.605</td><td>0.808</td><td>3.50</td><td>72.2</td><td>0.350</td><td>3.798</td><td>28.500</td><td>6.000</td><td>20m2s</td></tr></table>

Results. Table A1 compares storyboard generation across methods. The two Manga2Animate configurations jointly achieve the best results on all four manga-fidelity metrics: the open-source FLUX.2 backend reaches the highest CSD and OCCM, while the closed-source Gemini backend achieves the highest CIDS and PA, each improving over the best baseline. Since all baselines use the same refined script as our method, the PA gap reflects content adherence under an identical specification, and this advantage holds for both an open and a closed image backend rather than relying on a specific image model.

In contrast, StoryDiffusion and Story-Iter score higher on the anime-specific aesthetic metrics but considerably lower on manga fidelity (CIDS, PA, OCCM), since both freely generate imagery from text without being constrained by the original characters and composition. This indicates that singleimage aesthetics and adaptation fidelity are distinct aspects: an aesthetic predictor assesses the appeal of an individual illustration without considering whether it is consistent with the source. The strengths of Manga2Animate lie in character preservation, content alignment, and cross-panel consistency, which matter most for this task, and participants in our user study also clearly preferred our results overall (Sec. 4.6).

The two backends offer different cost trade-offs: FLUX.2 avoids external image-generation APIs but takes 1 h 59 min on local GPUs, while Gemini takes 20 min 2 s with 6 image-generation calls. Both are slower than StoryDiffusion and AnyStory but considerably cheaper than AniMaker, another multi-agent pipeline (8 h 9 min); users can choose a backend according to their compute resources and API budget.

## D ADDITIONAL ABLATION RESULTS

Script selection strategy. Sec. 4.7 compares clip-wise and script-wise script selection and reports the headline result; Table A2 gives the full metric breakdown. Clip-wise selection outperforms script-wise selection on I2VS, Subject Consistency, Background Consistency, and both aesthetic metrics, raising Waifu-v3 from 4.423 to 5.090 and Shadow-v2 from 0.473 to 0.518, while DD increases from 5.951 to 9.457. I2VB and Motion Smoothness are slightly lower than with script-wise selection (0.768 vs. 0.772 and 0.960 vs. 0.963), with both gaps within 0.004. We conjecture that clip-wise selection retains the more complete and specific clip descriptions among the candidates, enabling the video model to produce richer motion and visuals, whereas combining descriptions from different candidates may introduce slight variations in scene settings, causing the small drop in I2VB.

Dual-level feedback. To examine the contribution of each feedback source in the refinement loop (Sec. 3.4), we withhold either the script-level or the storyboard-level feedback from the next-round Script Creator, while the reviewers still select the best script and storyboard in every round. All variants use the closed-source configuration, i.e., gemini-3.7-flash for all creator and reviewer agents and gemini-3.1-flash-image as the Storyboard Creator, and are evaluated at the storyboard level, where the feedback takes effect directly. Withholding a feedback source changes only the content of the Script Creator’s input, not the number of generation or review calls, so every variant has the same computational cost as the full pipeline (Table A1); we therefore omit the cost columns. As shown in Table A3, removing either feedback source lowers CSD (0.605 to 0.597 and 0.588), CIDS (0.808 to 0.786 and 0.770), and PA (3.50 to 3.33 and 3.39) for the variants without script and without storyboard feedback, respectively. For OCCM, removing script feedback lowers the score from 72.2 to 70.8, whereas removing storyboard feedback yields a comparable 72.6. Removing storyboard feedback also lowers both aesthetic metrics more than removing script feedback (Shadow-v2: 0.249 vs. 0.334; Waifu-v3: 3.257 vs. 3.813), though the without-script-feedback variant slightly exceeds the full model on Waifu-v3 (3.813 vs. 3.798). These results indicate that script-level and storyboard level feedback are complementary, and combining them mainly improves character identity and content alignment.

Table A2: Ablation on script selection strategy. We compare Manga2Animate with clip-wise and script-wise selection. The full model selects the best clip description for each manga panel in a clipwise manner and composes them into the final script, whereas script-wise selection directly picks one complete candidate script. All other models and settings are unchanged. Red cells denote the better result in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2">Alignment</td><td colspan="4">Video Quality</td><td colspan="2">Aesthetic Quality</td></tr><tr><td>I2VS↑</td><td>I2VB↑</td><td>SC↑</td><td>BC↑</td><td>MS↑</td><td>DD↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td></tr><tr><td>Script-wise Selection</td><td>0.785</td><td>0.772</td><td>0.928 0.956</td><td></td><td>0.963</td><td>5.951</td><td>0.473</td><td>4.423</td></tr><tr><td>Clip-wise Selection (Full Model)</td><td>0.791</td><td>0.768</td><td>0.931 0.958</td><td></td><td>0.960</td><td>9.457</td><td>0.518</td><td>5.090</td></tr></table>

Table A3: Ablation on dual-level feedback. The reviewers still select the best script and storyboard, but the feedback of one reviewer is withheld from the next-round Script Creator. All variants use the closed-source (Gemini) configuration. The Full Model row is the run reported in Table A1. Cost is identical to the full pipeline and is therefore omitted. Red and orange mark the best and second-best results; higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">V.C.</td><td colspan="2">C.A.</td><td colspan="2">A.Q.</td></tr><tr><td>CSD↑</td><td>CIDS↑</td><td>PA↑</td><td>OCCM↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td></tr><tr><td rowspan="3">Without Script Feedback Without Storyboard Feedback</td><td>0.597</td><td>0.786</td><td>3.33</td><td>70.8</td><td>0.334</td><td>3.813</td></tr><tr><td>0.588</td><td>0.770</td><td>3.39</td><td>72.6</td><td>0.249</td><td>3.257</td></tr><tr><td>0.605</td><td>0.808</td><td>3.50</td><td>72.2</td><td>0.350</td><td>3.798</td></tr></table>

Refinement loop and storyboard generation. Skipping the second-round feedback (w/o Refinement) or replacing the generated storyboard with the original manga panels (w/o Storyboard Gen.) both lower I2VS, I2VB, SC, and BC relative to the full model, and removing storyboard generation causes the larger drop, confirming that both stages contribute to the final video quality.

Table A4: Ablation on the refinement loop and storyboard generation. w/o Refinement uses the first-pass script and storyboard directly, skipping the second-round feedback; w/o Storyboard Gen. keeps the same fully refined script but replaces the generated storyboard with a composite of the original manga panels. Removing either stage lowers I2VS, I2VB, SC, and BC relative to the full model, and removing storyboard generation causes the larger drop. Some variants score higher on MS or Shadow-v2, but lower on alignment, consistency, or DD. Red and orange mark the best and second-best results; higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">Alignment</td><td colspan="4">Video Quality</td><td colspan="2">Aesthetic Quality</td></tr><tr><td>I2VS↑</td><td>I2VB↑</td><td>SC↑</td><td>BC↑</td><td>MS↑</td><td>DD↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td></tr><tr><td>w/o Refinement</td><td>0.778</td><td>0.754</td><td>0.914</td><td>0.952</td><td>0.965</td><td>7.872</td><td>0.521</td><td>4.818</td></tr><tr><td>w/o Storyboard Gen.</td><td>0.766</td><td>0.745</td><td>0.896</td><td>0.947</td><td>0.961</td><td>8.289</td><td>0.515</td><td>5.050</td></tr><tr><td>Full Model</td><td>0.791</td><td>0.768</td><td>0.931</td><td>0.958</td><td>0.960</td><td>9.457</td><td>0.518</td><td>5.090</td></tr></table>

Video generation mode. Manga2Animate supports both reference-conditioned and first-frame video generation. In the former, the video model receives the complete storyboard as a reference image and generates a full video. In the latter, each storyboard panel serves as the first frame of its corresponding clip, and the selected clips are assembled in order.

Table A5: Comparison of Video Generation Modes. Both modes use Gemini for storyboard generation and Seedance 2.0 for video generation. Bold indicates the higher value for each metric. Reference-conditioned generation gives better subject alignment, background consistency, and aesthetic scores, while first-frame generation gives better background alignment and slightly better subject consistency and motion smoothness.
<table><tr><td rowspan="2">Mode</td><td colspan="2">Alignment</td><td colspan="4">Video Quality</td><td colspan="2">Aesthetic Quality</td></tr><tr><td>I2VS↑</td><td>I2VB↑</td><td>SC↑</td><td>BC↑</td><td>MS↑</td><td>DD↑</td><td>Shadow-v2↑</td><td>Waifu-v3↑</td></tr><tr><td>Reference-conditioned</td><td>0.791</td><td>0.768</td><td>0.931</td><td>0.958</td><td>0.960</td><td>9.457</td><td>0.518</td><td>5.090</td></tr><tr><td>First-frame</td><td>0.772</td><td>0.778</td><td>0.935</td><td>0.941</td><td>0.962</td><td>4.045</td><td>0.302</td><td>4.042</td></tr></table>

As shown in Table A5, reference-conditioned mode obtains higher I2VS, BC, Shadow-v2, and Waifu-v3 scores, while first-frame mode obtains higher I2VB and slightly higher SC and MS scores. The results demonstrate that the pipeline can also operate with first-frame video generation when full-storyboard reference conditioning is unavailable. We use reference-conditioned mode for the main results.

## E BASELINE ADAPTATION

StoryDiffusion (storyboard). StoryDiffusion’s official multi-character mechanism takes only plain-text, per-shot prompts that tag which characters appear, with no interface for imageconditioned identity. To keep the comparison fair, we take the same final best script produced by Manga2Animate’s own creator-reviewer refinement (Sec. 3.4) and reformat character descriptions and per-shot scene/action text into the per-shot prompts its official multi-character mechanism requires. Because StoryDiffusion has no reference-image input, the manga-cropped character images used elsewhere in our evaluation do not apply here; identity relies purely on text description, a limitation of the method when processing image information. Each story yields 6 panels, composed into the same 3 × 2 grid as the other backends for evaluation.

Story-Iter (storyboard). Story-Iter generates a sequence of story images from ordered text prompts and maintains cross-image consistency through Global Reference Cross-Attention (GRCA) over previously generated results. To unify the story content, we convert the final refined script of Manga2Animate (Sec. 3.4) into six prompts ordered by manga panel, preserving the characters, interactions, scenes, actions, and camera descriptions. Generation follows the official settings with the comic style. Since Story-Iter does not accept external character reference images, we do not add an identity-control module. The six generated images are composed into a 3 × 2 grid in order and scored with MangaEval.

AnyStory (storyboard). AnyStory supports personalized story image generation with character reference images. To unify the story content, we convert the final refined script of Manga2Animate (Sec. 3.4) into the structured script required by AnyStory, preserving the characters, scenes, actions, interactions, and camera descriptions of the six panels, and we use the same manga-cropped character images as character references. All other stages follow the official AnyStory implementation, without adding the Storyboard Creator or Reviewer of Manga2Animate. The six generated images are composed into a 3 × 2 grid in panel order and scored with MangaEval.

AniMaker (video). AniMaker natively takes a plain-text story and self-generates both the script (via an LLM) and the character portrait (via Hunyuan3D (Zhao et al., 2025)). To compare videogeneration ability rather than manga-understanding ability, we replace only these two front-end stages: the script is converted from Manga2Animate’s own final, refined script (Sec. 3.4) into Ani-Maker’s native script.json format, and the character portrait is replaced with the same mangacropped reference image used by our pipeline. All remaining stages (environment-image generation, scene-prompt synthesis, MCTS-based candidate clip generation and selection, narration, and final assembly) retain AniMaker’s original implementation and default hyperparameters, and outputs are scored under the same MangaEval protocol as all baselines.

StoryMem (video). StoryMem generates multi-shot videos shot by shot from a structured story script and a visual memory. To unify the story content, we convert the final refined script of Manga2Animate (Sec. 3.4) into the StoryMem format, preserving the characters, actions, scenes, and camera descriptions, and adopt the reference-initialized M2V setting, which uses the six manga panels as the initial visual references. Video generation uses the Wan2.2-I2V-A14B (Wan et al., 2025) backbone with the corresponding StoryMem LoRA checkpoint, and the visual-memory update and shot-by-shot generation follow the official settings. We keep its original mechanism without adding the storyboard, Video Reviewer, or candidate selection of Manga2Animate, and the generated shots are concatenated in script order for evaluation.

ViMax (video). ViMax supports a screenplay as input and automatically performs storyboarding, shot planning, character image generation, and video clip generation. To prevent it from rewriting the story from a short concept, we use its Script2Video workflow and convert the final refined script of Manga2Animate (Sec. 3.4) into a screenplay input that preserves the characters, scenes, actions, dialogue, camera descriptions, and narrative order. The manga-cropped character images serve as ViMax’s character references to maintain appearance consistency. We use gemini-3.6-flash and gemini-3.1-flash-image as the language and image generation models, and Seedance 2.0 (or Veo 3.1 Fast) as the video generation model through external APIs. All other storyboard, shot-planning, and video-generation stages follow the original Script2Video pipeline of ViMax, without adding the storyboard or reviewer modules of Manga2Animate.

## F EVALUATION DETAILS

Table A6: Storyboard Evaluation Metrics.  
Table A7: Video Evaluation Metrics.
<table><tr><td>Domain</td><td>Metric</td><td>Brief Description</td><td>Domain</td><td>Metric</td><td>Brief Description</td></tr><tr><td>Visual Consistency (V.C.)</td><td>CLIP Style Disentanglement (CSD)</td><td>Average source-to-generated and inter-panel style consistency</td><td>Alignment</td><td>I2V Subject Similarity (I2VS)</td><td>Subject similarity between the source manga and generated video frames using DINOv3 features</td></tr><tr><td></td><td>Character Identification Similarity (CIDS)</td><td>Average source-to-generated and inter-panel character identity consistency using SigLIP features</td><td></td><td>I2V Background Similarity (I2VB)</td><td>Background similarity between the source manga and generated video frames</td></tr><tr><td>Content Alignment (C.A.)</td><td>Prompt Alignment (PA)</td><td>GPT-4.1 judgment of scene, shot, character interaction, and individual action adherence</td><td>Video Quality (V.Q.)</td><td>Subject Consistency (SC)</td><td>Temporal consistency of subjects across generated video frames</td></tr><tr><td></td><td>Onstage Character Count Matching (OCCM)</td><td>Agreement between detected character counts in generated and source manga panels</td><td></td><td>Background Consistency (BC) Motion</td><td>Temporal consistency of backgrounds across generated video frames</td></tr><tr><td>Aesthetic Quality (A.Q.)</td><td>Aesthetic Shadow V2</td><td>General aesthetic quality of the generated storyboard</td><td></td><td>Smoothness (MS) Dynamic Degree</td><td>Temporal smoothness of motion across adjacent video frames Continuous</td></tr><tr><td></td><td>Waifu Scorer V3 (Waifu-v3)</td><td>Anime-oriented visual quality of the generated storyboard</td><td></td><td>(DD)</td><td>optical-flow-based motion intensity, normalized to a 256-pixel short side</td></tr><tr><td>Cost</td><td>Calls (VLM)</td><td>Number of VLM calls required for generation and review</td><td>Aesthetic Quality (A.Q.)</td><td>Aesthetic Shadow V2 (Shadow-v2)</td><td>General aesthetic quality of the generated video</td></tr><tr><td></td><td>Calls (Img Gen)</td><td>Number of external image-generation API calls</td><td></td><td>Waifu Scorer V3 (Waifu-v3)</td><td>Anime-oriented visual quality of the generated video</td></tr><tr><td></td><td>Time</td><td>Total storyboard generation time</td><td>Cost</td><td>Calls (VLM)</td><td>Number of VLM calls required for generation and</td></tr><tr><td></td><td></td><td></td><td></td><td>Calls (Video Gen)</td><td>review Number of external</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>video-generation API calls</td></tr></table>

We provide full definitions for the metrics summarized in Tables A6 and A7.

Storyboard level. Visual Consistency is measured by CSD and CIDS, each reporting the average of a cross score (similarity between generated panels and the character reference images) and a self score (similarity among generated panels). CSD (Somepalli et al., 2024) uses CSD-CLIP ViT-L style features and is unmodified. CIDS detects characters in the generated panels with GroundingDINO (Liu et al., 2024b), assigns them to reference characters by Hungarian matching (Kuhn, 1955), and compares identity features; we replace the original CLIP at 224px with SigLIP (so400m) at 384px, and the cross score is computed over successfully matched character crops. Content Alignment is measured by PA and OCCM. PA uses GPT-4.1 to score each panel on scene, character action, and camera, and each character crop on single-character action, on a 0–4 scale; each item is sampled 5 times at temperature 1.0 and averaged, and PA is the average over the four aspects. The scoring reference is the per-panel text converted from the refined script, which is shared by all script-conditioned methods. $\begin{array} { r } { \mathrm { O C C M } = 1 0 0 \cdot \exp \left( - \frac { \left| D - E \right| } { \epsilon + \operatorname* { m a x } \left( E , 1 \right) } \right) } \end{array}$ , where D is the number of characters GroundingDINO detects in the generated panel and E is the expected count, now obtained by applying the same detector to the source manga panel rather than taken from the script’s cast list. Aesthetic Quality is measured by two anime-specific models: Aesthetic Shadow v2 (a 1.1Bparameter high-/low-quality ViT classifier, score = P(high quality)) and Waifu-Scorer v3 (CLIP ViT-L/14 features with an MLP regression head, output in [0, 10]), averaged over the six generated panels. Computational Cost is the average number of VLM/LLM calls, image-generation calls, and wall-clock time per story.

Video level. Each video is split into six segments corresponding to the six manga panels, and metrics are computed per segment and averaged. Manga Alignment is measured by I2VS and I2VB. I2VS measures the subject similarity between the frames of each segment and the corresponding character reference image using VBench++’s original formula, which combines the maximum similarity to the reference with the mean and minimum similarity between consecutive frames, with its DINO ViT-B/16 backbone replaced by DINOv3 ViT-B/16; segments without characters are excluded. I2VB measures the background similarity between the frames of each segment and the corresponding source manga panel using DreamSim (Fu et al., 2023) features, with both formula and backbone unmodified. Video Quality is measured by Subject Consistency and Background Consistency (DINO- and CLIP-based frame-to-first-frame and frame-to-previous-frame similarity, respectively, both unmodified VBench++ metrics), Motion Smoothness (an unmodified VBench++ metric based on AMT frame interpolation (Li et al., 2023)), and Dynamic Degree. We change DD from VBench++’s binary decision based on the top-5% RAFT (Teed & Deng, 2020) flow magnitudes into a continuous motion-magnitude score: each video is resampled to about 8 fps, RAFT computes optical flow between consecutive frames at the native resolution, and the flow magnitude is averaged over all pixels and frame pairs and normalized by 256/ min(H, W). DD measures motion magnitude rather than motion correctness. Aesthetic Quality reuses Aesthetic Shadow v2 and Waifu-Scorer v3 from the storyboard level, applied to every frame per video and averaged. Computational Cost is the end-to-end average number of VLM calls, generation calls, and wall-clock time per story.

## G REVIEWER CRITERIA

All reviewers are prompted to first describe each candidate with respect to every criterion before reaching a verdict, and to output a structured JSON object containing per-criterion winners or scores, an overall winner, and improvement suggestions for the winner. The suggestions include only criteria on which the winner still shows observable weaknesses and serve as the feedback passed to the next refinement round. The full prompts are listed in Appendix H.

Character Reviewer. Inputs: the ordered manga panels, the character reference image(s), and the character descriptions of all $N _ { s }$ candidate scripts. Criteria (one winner each, one point per win): (1) clothing match, (2) physical appearance accuracy, including facial features, hairstyle, and body proportions, (3) color accuracy with respect to the character reference image, and (4) left/right specificity of asymmetric features such as scars, hair parts, and accessories. The manga panels serve as the ground truth except for colors, which follow the reference image. Ties in total score are broken in the priority order $( 2 ) > ( 1 ) > ( 4 ) > ( 3 )$ .

Clip Reviewer. Inputs: one manga panel, the character descriptions (used only to identify characters), and the $N _ { s }$ candidate descriptions of that panel. Criteria (one winner each): (1) small objects associated with each character, penalizing both missing and hallucinated objects, (2) facial expressions and emotions, (3) background accuracy, (4) object placement and orientation, (5) character body orientation and spatial relationships, and (6) dialogue speaker assignment. Ties are broken in the priority order $( 6 ) > ( 5 ) > ( 4 ) > ( 2 ) > ( 1 ) > ( 3 )$ . The reviewer runs independently for each panel position, and the winning descriptions are reassembled into the final script (Fig. 4).

Storyboard Reviewer. Inputs: the six manga panels, the character reference image(s), two candidate storyboards A and B, and the script used to generate them. Thirteen criteria, one point each: (1) character identity preservation, (2) overall art-style consistency with the manga, (3) color presence (not primarily black-and-white), (4) consistent panel size in the $3 \times 2$ layout, (5) only existing characters, (6) narrative order consistency, (7) script alignment, (8) rich background, (9) background consistency across panels, (10) character and object position consistency across panels, (11) panelinternal object layout match with the manga, (12) no dialogue boxes or captions, and (13) left/right consistency of asymmetric character features. Criteria (3), (4), (5), (6), and (12) are judged pass/- fail for each storyboard independently, so both storyboards may receive the point; the others are comparative. The storyboard with the higher total wins, and remaining ties are broken by a fixed criterion priority order. From the second round onward, a variant of the prompt compares each new candidate with the previous best storyboard, each judged against the script it was generated from.

Open-source panel reviewer. Inputs: the source manga panel, the character references for that panel (possibly none), the corresponding script clip, and one candidate panel. Each candidate is scored independently on a 0–4 scale for (1) manga/script follow, covering action, pose, composition, and framing (the primary criterion), (2) character identity consistency, (3) art style and colorization quality, penalizing black-and-white line art, screentone residue, and speech bubbles, and (4) background and environment fidelity. The candidate with the highest total is kept, with ties broken by criterion (1).

Video Reviewer. Inputs: the generation prompt and two candidate videos A and B, plus the previously selected clip for every clip after the first in the first-frame mode. For the first clip, and in the reference-conditioned mode, the criteria and weights are visual realism (0.35), physical commonsense (0.30), and video-prompt alignment (0.35), where alignment also checks left/right directional correctness. For subsequent clips in the first-frame mode, these weights become 0.2 each, and coherency with the previous clip is added with a weight of 0.4. Each criterion is scored 1 (A wins), 0 (B wins), or 0.5 (tie), and a penalty of 0.5 is subtracted for violating hard constraints, such as sudden appearance or disappearance of main objects, unnecessary scene transitions, or unnaturally fast or slow motion. If the weighted scores differ by less than 0.05, the two candidates are judged comparable and the first candidate is kept.

## H AGENT PROMPTS

We list the full prompts of all creator and reviewer agents in pipeline order. Text in braces or angle brackets (e.g., {Target Transcript}, <script>) is a placeholder filled in at run time.

\*\*Role & Objective\*\*   
You are an expert Vision AI and Character Designer. Your task is to analyze one or   
more reference images(it could be manga panel or colored reference image) and produce a precise, structured visual description of   
all   
visible characters.   
\*\*Instruction\*\*   
Extract all visible visual details of each character and map them strictly into   
the predefined template. The output must be a valid JSON object.   
\*\*Input Format\*\*   
- Image(s): One or more reference images, given in a fixed order, depicting one or   
more characters. If a single image is given, it may show one character or   
several characters together. If multiple images are given, each image shows one   
character’s clean design, and the images are given in the same order as that   
character’s id (the first image is Character 1, the second is Character 2, and   
so on).   
\*\*Output Format\*\*   
You must output ONLY a valid JSON object in the following structure:   
{   
"characters": [   
{   
"id": 1,   
"description": "[Character summary]; [Hair]; [Face & eyes];[Body build];" }   
\*\*Strict Rules\*\*   
1. \*\*Strict Template Order\*\*   
The description MUST follow EXACTLY this order:   
- Character summary (age, gender, body build, posture if visible)   
- Hair (color, length, style, bangs, special features)   
- Face & eyes (face shape, expression, eye color, notable features)   
- Body build (height, figure, color information)   
Each section MUST be separated by a semicolon ‘;‘   
2. \*\*Visibility Rule (IMPORTANT)\*\*   
If a part is NOT visible in the image:   
- Leave the section EMPTY (i.e., do not keep the position).   
- DO NOT write "not visible" or any similar phrase.   
Example:   
". .; Black hoodie; Backpack"   
3. \*\*Maximum Detail Requirement\*\*   
Each section MUST be as specific as possible:   
- Avoid vague terms like "casual body" or "light-color hair"   
- Specify color, texture, length, direction and style whenever visible   
- Include patterns, layers, and materials if detectable   
Bad example: "black hair"   
Good example: "Short straight black hair with blunt bangs"   
4. \*\*No Hallucination\*\*   
Only describe what is clearly visible.   
Do NOT guess hidden parts.   
5. \*\*Multiple Characters / Multiple Images\*\*   
- If a single image shows multiple characters, assign IDs sequentially (1, 2,   
3. .) in a consistent left-to-right / most-prominent-first order, and output   
each as a separate object.   
- If multiple images are given (one clean image per character), assign IDs in   
the SAME order the images were given: the first image is id 1, the second is   
id 2, and so on. Do not merge multiple images into one character entry.   
6. \*\*No Extra Text\*\*   
Output ONLY the JSON object.   
No explanations, no markdown, no extra text.

# Figure A2: Prompt for the Character Description Creator.

```markdown
You are an expert Animation Script Generator and Storyboard Analyst. Your task is to transform manga content into a structured
animation script by analyzing visual cues, provided character descriptions, and a transcript.
### Input Structure Context (CRITICAL)
All input images are manga **panel** images for the target manga**(the actual number depends on actual number of inputs, this is
critical)**
You will receive: a sequence of Target Individual Panel Images (e.g., panel1.png, panel2.png. .) + Target Character Description + Target
Transcript.
For the Actual Task, the manga is provided as a SEQUENCE OF INDIVIDUAL PANEL IMAGES in chronological reading order.
### Output Format (Strict JSON)
Generate the script in the following JSON format.
CRITICAL: Output ONLY a valid JSON object.
{
"characters": [
{
"id": 1,
"description": "[Character summary (age/gender/build)]; [Hair style & color]; [Facial features & eye color]; [Upper garment style
& color]; [Lower garment style & color]; [Footwear style & color]; CRITICAL: Base this primarily on the provided character
description, BUT if the manga visual shows different clothing or appearance, OVERRIDE the text and describe exactly what is in the
manga."
}
],
"clips": [
{
"id": 1,
"visual_analysis": "[CRITICAL STEP: Briefly describe exactly what is shown in Panel Image {id}. Explicitly state if there are speech
bubbles or if it is a silent reaction shot.]",
"environment": "[Time of day]. [Lighting, architecture, colors, atmosphere]. [CRITICAL: Detail the layout of the entire scene, the
EXACT QUANTITY of visible objects, and the spatial relationships between them (left/right).]",
"description": "[Camera shot type]. [CRITICAL: Detail character spatial positioning (e.g., ’Character[1] is standing at the left side of
the screen’), body orientation (e.g., ’facing screen right’), actions, and expressions]. Use ’Character[id]’ format. Include Dialogue: ’. .’
and Thoughts (if any). ONLY include dialogue that corresponds to the speech bubbles physically present in this specific panel.",
"characters_in_panel": [1, 2]
}
### Rules & Guidelines (STRICTLY ENFORCED)
1. **Visual Source of Truth (CRITICAL)**: The manga images are the ultimate source of truth. If the provided character descriptions
or transcripts contradict the visual details (e.g., different outfits, different speakers, missing speech bubbles), YOU MUST FOLLOW
THE MANGA VISUALS. Update the output character descriptions accordingly.
2. **Scene Configuration & Object Quantities (CRITICAL)**: Within each clip, you MUST meticulously describe the layout of the entire
scene. Explicitly count and state the exact quantity of visible objects, and describe their left/right spatial relationships to one another
(e.g., "Three boxes are stacked on the left side of the screen, with a desk to their right").
3. **Screen-Space Spatial Descriptions**: All spatial descriptions (e.g., left, right, front, back, near, far) MUST be explicitly defined
from the camera’s perspective. Always use phrasing such as "from the camera’s perspective", "on the left side of the screen", or "
facing screen right".
4. **Absolute Clip Correspondence**: The number of clips MUST exactly match the number of target panel images provided. Clip 1
MUST describe Panel Image 1, Clip 2 MUST describe Panel Image 2, and so on. Do not merge, split, or reorder clips based on the
transcript narrative.
5. **Visual-First Dialogue Mapping**: The provided transcript is a continuous flow of text. You must chunk and assign the dialogue
based STRICTLY on the visual presence of speech bubbles in each specific panel image.
6. **Silent Panels**: If a panel image shows a character reacting without any speech bubbles, DO NOT assign any leftover dialogue to
that clip. Describe it purely as a visual and silent moment.
7. **Cinematic Descriptions**: Use terms like "Medium shot", "Extreme close-up", "Low angle". Analyze the visuals to accurately
represent the events, composition, and intent.
8. **No Override Annotation**: When manga visuals override the provided character descriptions or transcript, apply the correction
silently in the output. Do NOT explicitly mention or annotate that an override has occurred.
9.**Accessory & Small Object Rule**: Any small objects or accessories (e.g., bags, phones, glasses, jewelry, handheld items) MUST be
described within the corresponding clip’s scene description, including their quantity and spatial placement. DO NOT include these
items in the character description fields.
10. **Do not add any description about speech bubbles and sound effect inside clip’s description**
11. **Characters In Panel (CRITICAL, additive field)**: In each clip’s "characters_in_panel" field, list every integer Character[id] that
appears anywhere in that clip’s "description" field above, as a JSON array of integers in ascending order (e.g. [1, 2]). Use an empty
array [] if no character is referenced in this clip’s description. Do NOT invent ids that are not present in the "characters" array.
## Actual Task
Now, analyze the provided sequence of Target Individual Panel Images, the Target Character Description, and the Target Transcript to
generate the final JSON script.
```

[Target Character Description]   
{Target Character Description}   
[Target Transcript]   
{Target Transcript}  
Figure A3: Prompt for the Script Creator (first round, without feedback).

![](images/5a705f338dcd5e9f6cfbf89f222159125aedec1f85a018bedcafecf4c449f0c5.jpg)

![](images/4804b0341ce8a9615a9fa832d3e47a18fd754d0d06f567e2b150ce406dc4fe28.jpg)  
Figure A4: Prompt for the Script Creator (refinement rounds, with feedback).

![](images/b5fe39befe64dabbe5e073f8a1cf28d5da2bec6fd42ed0154391e754055bee37.jpg)

![](images/cf4429146fcf9b07ad665fee9e3cff7504fe147be9eb358f19308ae1d22ee696.jpg)  
Figure A5: Prompt for the Character Reviewer.

![](images/16bae9bccd3b35f685ffc6e7d4589db4f9cb6fe7c47f272d5f16e54d7f142b8f.jpg)

2. Character Emotion and Expression   
\* Which script most accurately captures the facial expressions and emotional states of the characters as depicted in the panel?   
3. Background Accuracy   
\* Which script most accurately describes the background environment, setting, architectural elements, and atmosphere shown in   
the panel?   
4. Object Placement and Orientation   
\* Which script most accurately describes the relative positions, directions, and orientations of objects within the panel?   
5. Character Body Orientation and Spatial Relationship   
\* Which script most accurately describes each character’s body direction, pose orientation, and positional relationship to one   
another within the panel?   
6. Dialogue Speaker Assianment   
\* Which script most accurately assigns each line of dialogue to the correct speaker as shown in the panel?   
Scoring:   
\* For each criterion, only one script gets 1 point.   
\* All other scripts get 0 points.   
\* Total score per script = sum of points across all 6 criteria.   
Tie-breaking Rules for Highest Script:   
\* If two or more scripts have the same total score, break the tie by comparing the following criteria in this exact priority order:   
1. Dialogue Speaker Assignment   
2. Character Body Orientation and Spatial Relationship   
3. Object Placement and Orientation   
4. Character Emotion and Expression   
5. Character Small Object Accuracy   
6. Background Accuracy   
\* If scripts are still tied after applying all criteria above, choose the one that is overall more precise, less hallucinatory, and more   
faithful to the panel.   
\* The "Highest Script" field must contain exactly one script name.   
Script:   
{Script}   
Output your entire response as a single JSON object in the following format. Do not include any text outside the JSON.   
{   
"Character Small Object Accuracy": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
},   
"Character Emotion and Expression": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
}   
"Background Accuracy": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
},   
"Object Placement and Orientation": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
},   
"Character Body Orientation and Spatial Relationship": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
},   
"Dialogue Speaker Assignment": {   
"description\_for\_script1": ". .",   
"description\_for\_script2": ". .",   
"description\_for\_scriptN": ". .",   
"winner": "Script X",   
"explanation": "Explain why this script is better than all other provided scripts specifically for this criterion."   
},

![](images/ca1a114bc51ca54d50571400884d25406ca98c641225ee2d4d94b6417f9ea85a.jpg)  
Figure A6: Prompt for the Clip Reviewer.

![](images/072c6a7e8a1be1e22f4bf534fa0052f31e2fcfbf67ba584825806b709baef507.jpg)  
Figure A7: Prompt for the Storyboard Creator (closed-source backend).

You are a professional comic storyboard evaluator. The user will provide:

\* Image 1-6: Target manga panels given by order in a manga page using different names (panel1.png. .e.t.c.)

\* Image 7 onward, up to but not including the final two images: one or more character reference images (one per character, in Character[id] order; the exact count varies by target -- it may be a single composite reference sheet or several separate percharacter images)

\* The second-to-last image: Storyboard A

\* The last image: Storyboard B

\* Script JSON: A JSON file containing the original script used to generate the storyboards

Your task is to compare the two storyboard images and determine which storyboard performs better for EACH evaluation criterion.

Script:

{Script JSON}

## Evaluation Criteria:

1. Character Identity Preservation -- Which storyboard better preserves character identity and recognizable character-specific traits from the comic page and character reference sheet?

2. Overall Art Style Consistency -- Which storyboard better preserves the overall visual art style, rendering style, and aesthetic tone shown in the comic inner page?

3. Color Presence & Visual Variety -- Evaluate this criterion independently for each storyboard. A storyboard passes if it clearly contains visible non-monochrome color and is not primarily black-and-white, grayscale, or near-monochrome. If both storyboards are clearly not black-and-white, both must pass this criterion. Do not force a single winner for this criterion, and do not compare artistic quality, richness, or beauty of the colors.

4. Consistent Panel Size -- Does each storyboard maintain consistent panel sizes across panels? The storyboard must follow the 3\*2 format.

5. Only Existing Characters -- Which storyboard better avoids introducing unauthorized or extra characters not present in the comic page or Script JSON?

6. Narrative Order Consistency -- Which storyboard better follows the narrative order defined by the comic page and Script JSON? 7. Script Alignment Accuracy -- Which storyboard better matches the actions, scene transitions, emotions, and events described in the Script JSON?

8. Rich Background -- Which storyboard has richer and more detailed backgrounds?

9. Background Consistency -- Which storyboard maintains more consistent backgrounds across panels in style, palette, and setting? 10. Character & Object Position Consistency -- Which storyboard better preserves relative positions between characters and objects across panels?

11. Object Layout Match -- Which storyboard better matches the panel-internal object layout and spatial arrangement from the comic page? Judge this criterion based only on the composition inside each corresponding panel, including the relative positions, spacing, depth relationships, and arrangement of objects within the scene. Do not evaluate overall manga page layout, panel ordering , panel sizing, or storyboard-level framing structure.

12. No Dialogue Boxes -- Which storyboard better satisfies the requirement of having no dialogue boxes, speech bubbles, captions, subtitles, or narrative text? Except for the caption of panel x under the images.

13. Left/Right Feature Consistency -- Which storyboard more accurately preserves asymmetric character features on the correct side across panels, such as a scar over a specific eye, a one-sided hairstyle parting, asymmetric accessories, unique markings, or differently colored eyes, without unintentionally flipping them between shots?

## Scoring Rules:

\* Winner of a comparative criterion = 1 point

\* Loser of a comparative criterion = 0 points

\* For criteria 3, 4 ,5, 6 and 12:

\* Pass = 1 point

\* Fail = 0 points

\* Both storyboards may receive the same score

\* Total score = total number of points earned

After evaluating all criteria:

1. Calculate total scores for Storyboard A and Storyboard B.

2. Select the overall winner based on total score.

3. If scores are tied, choose the storyboard with better style consistency, narrative consistency, script alignment, and visual coherence

4. Provide detailed improvement suggestions for the winning storyboard so the model can further improve future generations.

Improvement Suggestions Rule (IMPORTANT):

\* ONLY include items that the model detects as problematic in the winning storyboard.

\* DO NOT include all categories by default.

\* OMIT any category that is already good or acceptable.

\* First describe storyboard A and storyboard B according to the corresponding rule, then give the final answer.

\* Each suggestion must be concrete, actionable, and based on observed weakness.

Output ONLY valid JSON. Do not include markdown.

Required JSON format:

"criteria\_results": {

"Character Identity Preservation": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." },

"Overall Art Style Consistency": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." },

![](images/eb028164c3c98a08d58c387b328ae04b9c630f9b7ee50305ad3bad847230ab0f.jpg)  
Figure A8: Prompt for the Storyboard Reviewer (first round, candidates from the same script).

![](images/af92ea80c3b0ae55a51ffc89adb16fcb7a03b1d5de03708bd0e5c7dbfed3e516.jpg)

## Script 2 JSON: {Script 2 JSON}

Evaluation Criteria:

1. Character Identity Preservation -- Which storyboard better preserves character identity and recognizable character-specific traits from the comic page and character reference sheet?

2. Overall Art Style Consistency -- Which storyboard better preserves the overall visual art style, rendering style, and aesthetic tone shown in the comic inner page?

3. Color Presence & Visual Variety -- Evaluate this criterion independently for each storyboard. A storyboard passes if it clearly contains visible non-monochrome color and is not primarily black-and-white, grayscale, or near-monochrome. If both storyboards are clearly not black-and-white, both must pass this criterion. Do not force a single winner for this criterion, and do not compare artistic quality, richness, or beauty of the colors

4. Consistent Panel Size -- Does each storyboard maintain consistent panel sizes across panels? The storyboard must follow the 3\*2 format.

5. Only Existing Characters -- Evaluate this criterion independently for each storyboard. A storyboard passes if it avoids introducing unauthorized or extra characters not present in the comic page or its corresponding script. Both storyboards may pass or fail independently.

6. Narrative Order Consistency -- Evaluate this criterion independently for each storyboard. A storyboard passes if it follows the narrative order defined by the comic page and its corresponding script. Both storyboards may pass or fail independently.

7. Script Alignment Accuracy -- Which storyboard better matches the actions, scene transitions, emotions, and events described in its corresponding script?

8. Rich Background -- Which storyboard has richer and more detailed backgrounds?

9. Background Consistency -- Which storyboard maintains more consistent backgrounds across panels in style, palette, and setting? 10. Character & Object Position Consistency -- Which storyboard better preserves relative positions between characters and objects across panels?

11. Object Layout Match -- Which storyboard better matches the panel-internal object layout and spatial arrangement from the comic page? Judge this criterion based only on the composition inside each corresponding panel, including the relative positions, spacing, depth relationships, and arrangement of objects within the scene. Do not evaluate overall manga page layout, panel ordering , panel sizing, or storyboard-level framing structure.

12. No Dialogue Boxes -- Evaluate this criterion independently for each storyboard. A storyboard passes if it contains no dialogue boxes, speech bubbles, captions, subtitles, or narrative text. Except for the caption of panel x under the images. Both storyboards may pass or fail independently

13. Left/Right Feature Consistency -- Which storyboard more accurately preserves asymmetric character features on the correct side across panels, such as a scar over a specific eye, a one-sided hairstyle parting, asymmetric accessories, unique markings, or differently colored eyes, without unintentionally flipping them between shots?

Scoring Rules:

\* Winner of a comparative criterion = 1 point

\* Loser of a comparative criterion = 0 points

\* For criteria 3, 4, 5, 6, and 12:

\* Pass = 1 point

\* Fail = 0 points

\* Both storyboards may receive the same score

\* Total score = total number of points earned

After evaluating all criteria:

1. Calculate total scores for Storyboard A and Storyboard B.

2. Select the overall winner based on total score.

3. If scores are tied, choose the storyboard with better style consistency, narrative consistency, script alignment, and visual coherence

4. Provide detailed improvement suggestions for the winning storyboard so the model can further improve future generations.

Improvement Suggestions Rule (IMPORTANT):

\* ONLY include items that the model detects as problematic in the winning storyboard.

\* DO NOT include all categories by default.

\* OMIT any category that is already good or acceptable.

\* First describe storyboard A and storyboard B according to the corresponding rule, then give the final answer.

\* Each suggestion must be concrete, actionable, and based on observed weakness.

Output ONLY valid JSON. Do not include markdown.

Required JSON format: Required JSON format:

}, "Consistent Panel Size": { "description\_for\_A": ". .", "description\_for\_B": ". .", "A": "pass or fail", "B": "pass or fail", "reason": ". ." }, "Only Existing Characters": { "description\_for\_A": ". .", "description\_for\_B": ". .", "A": "pass or fail", "B": "pass or fail", "reason": ". ." }, "Narrative Order Consistency": { "description\_for\_A": ". .", "description\_for\_B": ". .", "A": "pass or fail", "B": "pass or fail", "reason": ". ." }, "Script Alignment Accuracy": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." }, "Rich Background": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." }, "Background Consistency": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." }, "Character & Object Position Consistency": { "description\_for\_A": ". .", "description\_for\_B": ". .", "winner": "A or B", "reason": ". ." },

![](images/59534a30af7d7478b50541a27a187198b0af523c9fb9774b110d72ac57e57a10.jpg)  
Figure A9: Prompt for the Storyboard Reviewer (later rounds, new candidate vs. previous best).

![](images/206c3e06afc382c86b2327c75f32d702e53d4b1fd55ca4555e70359740eb4b42.jpg)

![](images/bd8f31431aad1339f1f619ff85a3803aff5d55b7903b3722bef8715e68dc6f0a.jpg)  
Figure A10: Prompt for the open-source panel reviewer.

![](images/73adcba5f6e277afd5435f1547f76c619c1ecbcc970746dc66bcf1508e581e51.jpg)  
Figure A11: Prompt for the Video Creator (reference-conditioned mode).

![](images/79848f9fe83101a6ffc64ad95fc3b002cb28f0d09ed83d1a5b246d0f18af15ab.jpg)

6. Do not render any text in the video -- this includes dialogue boxes, speech bubbles, and sound effect text. The storyboard may   
contain sound effect text (e.g. "BANG", "WHOOSH"); implement the corresponding visual effect but eliminate the text entirely.   
7. Don’t add any elements that are not described in the ’Script’ JSON. Strictly adhere to the provided character descriptions and   
environment settings without introducing new objects, characters, or backgrounds that are not mentioned in the script.   
8. Ensure that the characters’ movements and interactions are coherent and contextually appropriate based on the provided script,   
avoiding any actions that would be out of character or inconsistent with the described scenario   
9. Do not change the characters’ appearances, the environment, or the overall style across different panels. Each panel’s video   
should visually align with the others to create a cohesive narrative flow when viewed in sequence.   
Script:   
<script>  
Figure A12: Prompt for the Video Creator (first-frame mode).

![](images/5189bc351fb601d2a580384334b80b6e5ccd96af6e81ea8318e5e0c9f9bfd182.jpg)  
Figure A13: Prompt for the Video Reviewer (reference-conditioned mode and the first clip in the first-frame mode).

![](images/864f23f88a2f6b4a65b3c845460aed6dc0afedbce6ef8c5bccc194df7077b901.jpg)  
Figure A14: Prompt for the Video Reviewer (subsequent clips in the first-frame mode).