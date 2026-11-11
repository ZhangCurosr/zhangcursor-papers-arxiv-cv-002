# ZoomGain: Mining Global-to-Zoom Perception Gaps for Fine-Grained Visual Supervision

Haozhan Shen<sup>1,2∗</sup> Liangbo He<sup>1</sup> Jun Lan<sup>1†</sup> Huijia Zhu<sup>1</sup> Weiqiang Wang<sup>1</sup> Tiancheng Zhao<sup>3,4†</sup> Jianwei Yin<sup>1</sup>

<sup>1</sup>Ant Group <sup>2</sup>Zhejiang University <sup>3</sup>Om AI Research <sup>4</sup>Binjiang Institute of Zhejiang University

## Abstract

Fine-grained visual perception requires Vision-Language Models (VLMs) to locate and interpret subtle evidence within complex scenes. Constructing supervision for this ability is challenging: questions must have correct, unambiguous answers that depend on actual finegrained visual details rather than coarse cues or language priors. Verifying these properties makes manual annotation labor-intensive and automatic generation difficult for models that themselves struggle with fine-grained perception. Yet the same models can often recognize such details once the region is zoomed in. We exploit this global-to-zoom perception gap and introduce ZoomGain, a data construction pipeline that constructs questions about the full image from visual information gained only by zooming in. It verifies each answer on the zoomed view with multiple models, ensures that each question remains unambiguous in the full image and cannot be answered from language priors, and specifically generates high-quality distractors for multiple-choice questions. With this pipeline and human-only screening (no manual QA editing), we build ZoomGainBench, a 1,000-question bilingual benchmark on which the best of 28 VLMs achieves only 61.08. Failure analysis reveals that models favor a particular type of distractor when answering multiple-choice questions incorrectly. Beyond evaluation, training on just 9K ZoomGain samples substantially improves the visual perception capabilities of our 4B/9B models, enabling them to outperform several hundred-billion-parameter models across various perception benchmarks. The trained 9B model achieves SoTA performance on several benchmarks, such as 90.25 on HR-Bench 4K. These results establish global-to-zoom perception gaps as a practical basis for scaling fine-grained visual supervision for both evaluation and training. Our code is available at https://github.com/inclusionAI/ZoomGain.

## 1 Introduction

Fine-grained visual perception is essential in many practical applications, from detecting small vehicles in aerial imagery Xia et al. (2018) and identifying subtle defects in industrial inspection Bergmann et al. (2019) to reading product labels for people with visual impairments Gurari et al. (2018). Addressing such tasks with Vision-Language Models (VLMs) Team (2026); Bai et al. (2025); Zhu et al. (2025); Liu et al. (2023) requires locating and interpreting small details that are easily overlooked in the full image. To develop and benchmark such fine-grained capabilities, high-quality supervision is essential: answers must be correct and unambiguous, and questions must genuinely require fine-grained visual evidence.

Constructing such supervision at scale is challenging. Manual annotation is labor-intensive, as annotators must inspect small regions and verify each of these properties. Automatic generation Wang et al. (2025a); Wei et al. (2026) faces a related difficulty: the models used to generate questions may themselves struggle to perceive the relevant details in the full image. As a result, they may produce incorrect answers, ambiguous references, or questions that do not actually require fine-grained perception. Motivated by this, the key challenge is to obtain reliable supervision on visual details that models fail to perceive in full scenes.

(a) Turning Perception Gaps into Fine-Grained Supervision  
![](images/55af97479c0ca87a7f166d763f0d4ac78a4413dc7475f6b5ee55892ab2326a30.jpg)  
Figure 1: ZoomGain: turning global-to-zoom perception gaps into fine-grained visual supervision. (a) The runner’s shoe color is unclear in the global view but discernible in the zoomed view; ZoomGain asks about such details in the full image, using a global referring expression to identify the target. The paired views $V _ { g } ^ { i }$ and $V _ { z } ^ { i } ,$ and the referring e are defined in Section 3.2. (b) ZoomGainBench, obtained by human screening without QA editing, challenges all 28 evaluated VLMs (c) RLVR on 9K training QA pairs improves Qwen3.5-4B and Qwen3.5-9B over their base models and ZwZ-trained counterparts (ICML’26 Wei et al. (2026)).

Zooming in has long been used to recover such details, either at inference time through visual search Wu & Xie (2023); Lai et al. (2025); Shen et al. (2025) and agentic VLMs with cropping tools Zheng et al. (2026); Hong et al. (2026), or during training through region-to-image distillation Wei et al. (2026); Yuan et al. (2026), since details a model fails to resolve in the full image often become recognizable once the region is zoomed in. We specifically use this global-to-zoom perception gap to construct the supervision itself (Fig. 1 (a)): contrasting the two views locates details that are hard to perceive in the full image.

Based on this idea, we introduce ZoomGain, an automated data construction pipeline that converts visual information, which are gained only by zooming, into challenging questions about the full image (Contribution #1). ZoomGain (1) builds questions exclusively from details described in the zoomed view but unclear or absent in the full-image description, so that answering requires the fine detail itself; (2) keeps an answer only when multiple models agree on it in the zoomed view, and discards questions with ambiguous full-image references or answerable without images; and (3) generates various multiple-choice distractors that could mislead the queried VLMs.

Using ZoomGain, we curate ZoomGainBench, a bilingual benchmark of 1,000 questions on attribute recognition, counting, and OCR, whose questions and answers are entirely model-authored: two human reviewers only accept/reject candidates (Contribution #2). Among 28 evaluated VLMs, the best model, GPT-5.6-Sol, scores only 61.08, and only five exceed 50 (Fig. 1(b)). Error analysis further reveals that incorrect choices concentrate on distractors mined from full-image mispredictions during construction (Sec. 3.3). This indicates that when details are unresolved, models tend to fill the information gap with similar scene-plausible guesses, suggesting that the global-to-zoom perception gap is a common weakness across current VLMs, rather than an issue to specific models.

Beyond evaluation, ZoomGain data also serves as effective training supervision (Contribution #3). With identical source images and training settings, we train Qwen3.5-4B and Qwen3.5-9B Team (2026) on 9K ZoomGain and ZwZ (ICML’26 Wei et al. (2026)) QA pairs. ZoomGain-trained models consistently outperform ZwZ across all 14 evaluated benchmarks (Fig. 1(c)). The gains extend beyond fine-grained perception to general perception and difficult perception-and-reasoning tasks, e.g., 11.01 points boost on O3-Bench Li et al. (2026a) for the 9B model. Notably, our fine-tuned 9B model is on par with GLM-5.3-Flash GLM-5-Team et al. (2026) and 4B model surpasses Qwen3.5-122B-A10B. Ablations show that removing either incremental evidence or multi-source distractors substantially reduces these gains, confirming that both designs rooted in the global-to-zoom perception gap are essential.

## 2 Related Work

Fine-grained perception benchmarks. A variety of benchmarks specifically evaluate fine-grained perception in high-resolution images, including Wu & Xie (2023); Wang et al. (2025b); Zhang et al. (2024); Wang et al. (2025a); Gavrikov et al. (2026); Lai et al. (2025); Zhang et al. (2026b); Wei et al. (2026); O3-Bench Li et al. (2026a) further requires multi-step reasoning beyond perception. However, questions in these benchmarks are typically annotated manually or generated by models with extensive manual editing (Table 1), making scalable cruation costly. Moreover, existing benchmarks rarely filter out language-prior shortcuts Cadene et al. (2019); Chen et al. (2024); Yue et al. (2025) or design targeted distractors. In contrast, ZoomGain fully automates QA authoring with zoom-view verification, blind shortcut filtering, and distractor mining from model errors.

Improving fine-grained perception. Zooming has been widely adopted to enhance fine-grained perception, either at inference time via visual search Wu & Xie (2023); Shen et al. (2025); Lai et al. (2025); Li et al. (2026a); Wang et al. and agentic cropping tools Zheng et al. (2026); Hong et al. (2026); Su et al. (2026); Zhang et al. (2025), or during training through zoomed-to-global distillation Wei et al. (2026); Yuan et al. (2026); Li et al. (2026b); Zhang et al. (2026a). Unlike prior works that focus on inference procedures or training objectives, ZoomGain targets data construction itself: it leverages the global-to-zoom perception gap to identify what to ask (strictly on zoom-revealed details) and reliably verify ground truth, making it complementary to existing approaches.

## 3 ZoomGain Data Construction Pipeline

ZoomGain exploits the perception gap between global and zoomed-in views: fine-grained details that are difficult to discern in the full image may become clear once the target region is zoomed in. As illustrated in Figure 2, ZoomGain first discovers and grounds target anchors (Sec. 3.1), then constructs questions and answers from details newly revealed by zooming (Sec. 3.2), and finally verifies answer reliability and visual dependence and builds multi-source distractors that capture plausible model errors (Sec.n 3.3). The complete algorithm and additional implementation details are provided in Appendix A.

## 3.1 Anchor Discovery and Grounding

Given an input image, denoted as the global view $V _ { g } ,$ , we first use a VLM to generate a global scene caption $C _ { g } .$ . A text-only LLM then takes $C _ { g }$ as its sole input and proposes a set of candidate anchors. Each anchor i is represented by a referring expression $e _ { i }$ that describes the target’s appearance, spatial location, or relations to other objects. To ensure that $e _ { i }$ can identify a localizable instance unambiguously in the full image, three VLMs independently ground $e _ { i }$ in $V _ { g }$ . We retain $e _ { i }$ only if each model returns exactly one valid bounding box, since a missing box or multiple boxes indicate that the referent cannot be found or may be ambiguous, and all pairwise box IoUs are at least 0.5, which provides cross-model evidence that $e _ { i }$ refers to a consistent region. For each retained anchor, we take the coordinate-wise median of the predicted boxes, expand the resulting region to include more surrounding context, yielding the crop box $b _ { i , }$ , and crop $V _ { g }$ using $b _ { i }$ to obtain a zoomed view $V _ { z } ^ { i }$ . The quadruples $( V _ { g } , V _ { z } ^ { i } , b _ { i } , e _ { i } )$ serve as inputs to the subsequent question construction stage.

![](images/289aa6e7ada36191d589c7b5c107c28ac12fdbee45b72b1a0c87a3d298eb0b63.jpg)  
Figure 2: Overview of ZoomGain. (a) Discover anchors and obtain consensus regions through independent grounding. (b) Mine global-to-zoom incremental evidence from paired views to construct questions and reference answers. (c) Verify answers and visual dependence, then assemble and recheck multiple-choice questions with multi-source distractors.

## 3.2 Global-to-Zoom Evidence and QA Construction

For each retained anchor $i ,$ we use a strong VLM $M _ { \mathrm { { c a p } } }$ to describe the target region from both views. Specifically, we draw a red bounding box at $b _ { i }$ in $V _ { g }$ to obtain an annotated global view $V _ { g } ^ { i } .$ Figure 1(a) illustrates the paired views $( V _ { g } ^ { i } , V _ { z } ^ { i } )$ . We feed $V _ { g } ^ { i }$ into $M _ { \mathrm { { c a p } } }$ and prompt it to describe only the contents of the red box in as much detail as possible, yielding the global-view caption $C _ { g } ^ { i }$ . In a separate, independent call, we provide the zoomed view $V _ { z } ^ { i }$ and request a detailed caption $C _ { z } ^ { i } .$ . This produces two descriptions of the same region $b _ { i }$ at different viewing scales. We then use an LLM to compare $C _ { g } ^ { i }$ and $C _ { z } ^ { i } ,$ extracting details that are explicitly described in $C _ { z } ^ { i }$ but remain unclear or unmentioned in $C _ { g } ^ { i }$ . These details form a candidate fact set $\Delta _ { i } .$ . For example, in Figure 2(b), the runner’s shoe color is unclear in the global-view caption but identified as red in the zoomed-view caption, so it becomes a candidate fact in $\Delta _ { i }$

We generate three types of questions: attribute recognition, text recognition (OCR), and counting. An LLM selects one fact from $\Delta _ { i }$ that supports a definite answer and generates a local question $q _ { \mathrm { l o c a l } } ^ { i }$ and a reference answer $a _ { i }$ based on this fact. The local question is phrased for the zoomed view with a simple reference such as “the runner”, without specifying the target’s location in the full image. The LLM then replaces this reference with the global referring expression $e _ { i }$ and adjusts the wording to produce a global-view question $q _ { \mathrm { g l o b a l } } ^ { i } ,$ , keeping the queried detail and reference answer unchanged, so that questions about details revealed by zooming clearly identify the intended target in the full image. For example, “What color are the runner’s shoe uppers?” becomes “What color are the shoe uppers of the dark-shirted runner on the right yellow line?” Appendix H illustrates this construction process with attribute, counting, and OCR examples.

## 3.3 Quality Verification and Multi-Source Distractors

We verify each candidate QA pair in two ways. (1) For crop-based verification, we provide several VLMs with $q _ { \mathrm { l o c a l } } ^ { i }$ and the zoomed view $V _ { z } ^ { i } ,$ and sample multiple answers from each model. We retain the QA pair only if all models meet a predefined accuracy threshold on their sampled answers. Checking agreement among multiple models answering based on the zoomed-view helps ensure answer correctness, thereby avoiding sole reliance on a single model’s caption for answer annotation, which may contain hallucinations Rohrbach et al. (2018); Li et al. (2023); Jing et al. (2024). (2) For blind verification, models are asked to answer $q _ { \mathrm { g l o b a l } } ^ { i }$ without an image. We discard a QA if all blind models exceed a prescribed correct-answer threshold. This step effectively filters out questions that can be answered through language priors alone rather than actual image content Cadene et al. (2019); Chen et al. (2024); Yue et al. (2025).

For verified QA pairs suited to multiple-choice evaluation, we construct distractors from three complementary sources. (1) Visual evidence supplies plausible alternatives from image content, such as the neon-green shoes of another runner in Figure 2(c), which can expose errors in target selection or attribute binding. (2) Blind guesses provide incorrect answers produced without the image, capturing language priors. (3) Global-view errors provide incorrect answers produced from the full image, capturing confusions when local details are difficult to discern. We remove candidates equivalent to the reference answer or also valid for the question, merge semantically redundant candidates, and normalize their wording to form distinct answer options.

We combine the reference answer with the selected distractors and recheck the resulting MCQ using the zoomed view. This final check tests whether the reference answer remains reliably distinguishable after competing options are introduced. If the per-model agreement thresholds are not met, we replace or remove problematic distractors. OCR and counting questions retain an open-ended format and bypass distractor construction and MCQ rechecking. Appendix I provides examples of candidate QA pairs rejected by crop-based and blind verification.

## 4 Implementation Details and Data Preparation

Model Roles and Verification Settings. We use Qwen3.5-397B-A17B Team (2026) as $M _ { \mathrm { { c a p } } }$ and the initial global scene captioner. GLM-5.2 GLM-5-Team et al. (2026) performs incremental fact extraction and QA generation. Global-view error distractors are mined from incorrect answers produced by Qwen3.5-397B-A17B, Qwen3-VL-235B-A22B-Instruct Bai et al. (2025), and Kimi-K2.6 Team et al. (2026b) to the global question on the full image, with 10 answers sampled per model (30 in total). All verification stages likewise sample 10 answers per model and apply per-model accuracy thresholds. Complete model assignments, verification thresholds, and distractor repair settings are provided in Appendix B.

Data Generation and Benchmark Curation. We randomly sample 12K images from ZwZ Wei et al. (2026), which collects high-resolution images from several sources Kirillov et al. (2023); Chuang et al. (2025); Chen et al. (2022); Changpinyo et al. (2021); Krishna et al. (2017); Schuhmann et al. (2022), and generate a pool of 23,610 QA pairs with ZoomGain. GLM-5.2 jointly generates Chinese and English versions of both the local and global questions, and the two language versions of a question are counted as one QA pair. From this pool, two human reviewers select 1,000 questions for the benchmark without writing or editing any questions or answers.

Training Data Selection and Matching. Excluding the 1,000 benchmark items, the remaining QAs form the training pool. For a fair comparison with ZwZ, for each image, we take the smaller number of available unique questions between the two sides and sample that many from each side without replacement, yielding 8,746 image-matched QA pairs. We then duplicate 254 randomly chosen pairs on both sides to reach 9,000 samples per dataset. Both training sets thus use identical source images (ZwZ retains its bounding-box overlays) and differ solely in the questions.

## 5 Experiments on ZoomGain Bench

To examine whether questions mined from global-to-zoom perception gaps pose a genuine challenge to current VLMs, we evaluate 28 widely-used VLMs on ZoomGainBench, spanning closed-source frontier, agentic, fine-grained enhanced single-forward, and open-source general VLMs (Table 2).

Table 1: Comparison with existing fine-grained perception benchmarks. QA authoring: who writes the questions and answers; Model+Human denotes model-generated questions refined by human annotators, and Model denotes questions authored entirely by models, with humans only deciding which candidates to include. Blindfilter: questions answerable without the image are removed during construction. Local verif.: answers are automatically verified on the zoomed evidence region. Dual-view: both the full image and the evidence region are provided. Dedicated distractor generation: a separate stage for constructing distractor options (Section 3.3). In this column, ✗ indicates that no dedicated stage is reported, and N/A denotes open-ended evaluation. “–” denotes not provided or not specified.
<table><tr><td>Benchmark</td><td>#Questions</td><td>QA authoring</td><td>Language</td><td>Evidence</td><td>Blind filter</td><td></td><td>Local verif. Dual-view</td><td>Dedicated distractor generation</td></tr><tr><td>ZoomBench Wei et al. (2026)</td><td>845</td><td>Model+Human</td><td>EN</td><td>Auto</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>V* Bench Wu &amp; Xie (2023)</td><td>191</td><td>Human</td><td>EN</td><td>Human</td><td>x</td><td>x</td><td></td><td>x</td></tr><tr><td>HR-Bench Wang et al. (2025b)</td><td>200</td><td>Model+Human</td><td>EN</td><td>x</td><td></td><td></td><td></td><td>x</td></tr><tr><td>VisualProbe Lai et al. (2025)</td><td>500</td><td></td><td>EN</td><td>x</td><td></td><td></td><td></td><td>N/A</td></tr><tr><td>TreeBench Wang et al. (2025a)</td><td>405</td><td>Model+Human</td><td>EN</td><td>Human</td><td>×××</td><td>×××</td><td>××××</td><td>x</td></tr><tr><td>FINERS-4k Zhang et al. (2026b)</td><td>2,427</td><td>Human</td><td>EN</td><td></td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>ZoomGainBench (ours)</td><td>1,000</td><td>Model</td><td>EN/ZH</td><td>Auto</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 2: Evaluation results on ZoomGainBench. Scores are reported as percentages for English and Chinese questions. Each language-specific Overall is weighted by the number of questions in each category. Bold and underlined scores indicate the best and second-best results across all models in each column, respectively.
<table><tr><td></td><td colspan="4">English</td><td colspan="4">Chinese</td><td>Overall</td></tr><tr><td>Model</td><td>Attr</td><td>Count</td><td>OCR</td><td>Overall</td><td>Attr</td><td>Count</td><td>OCR</td><td>Overall</td><td></td></tr><tr><td colspan="10">Closed-source Frontier VLMs</td></tr><tr><td>GPT-5.5 OpenAI (b)</td><td>52.75</td><td>57.45</td><td>60.40</td><td>56.51</td><td>52.12</td><td>63.83</td><td>60.96</td><td>57.06</td><td>56.79</td></tr><tr><td>Gemini-3.6-Flash Google DeepMind (2026)</td><td>51.27</td><td>42.55</td><td>74.71</td><td>60.62</td><td>48.52</td><td>45.74</td><td>71.42</td><td>58.20</td><td>59.41</td></tr><tr><td>Claude-Opus-5 Anthropic (2026)</td><td>48.09</td><td>63.83</td><td>71.86</td><td>59.89</td><td>48.73</td><td>62.77</td><td>71.29</td><td>59.84</td><td>59.86</td></tr><tr><td>GPT-5.6-Sol OpenAI (c)</td><td>56.57</td><td>62.77</td><td>67.93</td><td>62.08</td><td>54.03</td><td>63.83</td><td>65.84</td><td>60.07</td><td>61.08</td></tr><tr><td colspan="10">Agentic VLMs</td></tr><tr><td>DeepEyes-7B Zheng et al. (2026)</td><td>24.79</td><td>20.21</td><td></td><td></td><td></td><td>22.34</td><td>34.46</td><td></td><td>29.99</td></tr><tr><td>Thyme-7B Zhang et al. (2025)</td><td>26.69</td><td>34.04</td><td>41.05 43.88</td><td>31.42 34.85</td><td>24.36 22.88</td><td>31.91</td><td>40.31</td><td>28.55 31.30</td><td>33.07</td></tr><tr><td>DeepEyesV2-7B Hong et al. (2026)</td><td>22.25</td><td>41.49</td><td>43.70</td><td>33.37</td><td>22.67</td><td>34.04</td><td>43.62</td><td>32.83</td><td>33.10</td></tr><tr><td>PixelReasoner-7B Su et al. (2026)</td><td>27.12</td><td>45.74</td><td>44.83</td><td>36.56</td><td>25.42</td><td>23.40</td><td>40.13</td><td>31.62</td><td>34.09</td></tr><tr><td colspan="10">Fine-grained Enhanced Single Forward VLMs</td></tr><tr><td>ZwZ-4B Wei et al. (2026)</td><td>30.08</td><td>39.36</td><td>29.68</td><td></td><td>33.47</td><td>50.00</td><td>34.45</td><td></td><td></td></tr><tr><td>Vision-OPD-4B Yuan et al. (2026)</td><td>35.59</td><td>59.57</td><td>45.69</td><td>30.78 42.23</td><td>39.62</td><td>59.57</td><td>43.80</td><td>35.45 43.31</td><td>33.12 42.77</td></tr><tr><td>Vision-OPD-9B Yuan et al. (2026)</td><td>46.40</td><td>55.32</td><td>47.18</td><td>47.58</td><td>47.67</td><td>60.64</td><td>37.97</td><td>44.68</td><td>46.13</td></tr><tr><td>ZwZ-8B Wei et al. (2026)</td><td>42.58</td><td>55.32</td><td>50.53</td><td>47.23</td><td>44.70</td><td>52.13</td><td>51.77</td><td>48.47</td><td>47.85</td></tr><tr><td colspan="10">Open-source General VLMs</td></tr><tr><td>MiMo-VL-7B-RL Xiaomi (2025)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-4.6V (106B) Team et al. (2025)</td><td>23.73 24.36</td><td>34.04</td><td>20.06</td><td>23.11</td><td>23.31</td><td>41.49</td><td>32.06</td><td>28.82</td><td>25.96</td></tr><tr><td>Qwen3-VL-4B Bai et al. (2025)</td><td>21.82</td><td>38.30 43.62</td><td>31.57</td><td>28.80</td><td>22.46</td><td>36.17 41.49</td><td>28.51 33.63</td><td>26.37</td><td>27.59</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025)</td><td>15.89</td><td>41.49</td><td>31.14 39.17</td><td>27.92</td><td>22.88 19.92</td><td>30.85</td><td>41.64</td><td>29.30 30.37</td><td>28.61</td></tr><tr><td>Qwen3.5-4B Team (2026)</td><td>21.40</td><td>51.06</td><td>34.40</td><td>28.40 29.83</td><td>22.67</td><td>46.81</td><td>34.87</td><td>30.23</td><td>29.38</td></tr><tr><td>Qwen3.5-9B Team (2026)</td><td>21.82</td><td>41.49</td><td>37.79</td><td>30.60</td><td>22.67</td><td>47.87</td><td>42.52</td><td>33.66</td><td>30.03 32.13</td></tr><tr><td>Qwen3.5-35B-A3B Team (2026)</td><td>28.18</td><td>44.68</td><td>46.10</td><td>37.51</td><td>27.33</td><td>51.06</td><td>47.95</td><td>38.51</td><td>38.01</td></tr><tr><td>Qwen3.6-35B-A3B Qwen Team (2026e)</td><td>26.91</td><td>54.26</td><td>50.21</td><td>39.59</td><td>26.69</td><td>52.13</td><td>46.17</td><td>37.54</td><td>38.57</td></tr><tr><td>Qwen3-VL-235B-A22B Bai et al. (2025)</td><td>29.45</td><td>52.13</td><td>48.01</td><td>39.64</td><td>30.93</td><td>50.00</td><td>50.90</td><td>41.39</td><td>40.51</td></tr><tr><td>Qwen3.5-27B Team (2026)</td><td>33.05</td><td>51.06</td><td>45.92</td><td>40.33</td><td>34.75</td><td>58.51</td><td>48.67</td><td>43.02</td><td>41.68</td></tr><tr><td>Qwen3.6-27B Qwen Team (2026b)</td><td>33.47</td><td>56.38</td><td>47.47</td><td>41.70</td><td>37.71</td><td>70.21</td><td>45.33</td><td>44.07</td><td>42.89</td></tr><tr><td>Qwen3.8-27B Qwen Team (2026c)</td><td>40.25</td><td>54.26</td><td>45.49</td><td>43.84</td><td>44.92</td><td>62.77</td><td>43.05</td><td>45.78</td><td>44.81</td></tr><tr><td>Qwen3.5-397B-A17B Team (2026)</td><td>33.47</td><td>65.96</td><td>51.95</td><td>44.55</td><td>38.14</td><td>62.77</td><td>50.41</td><td>45.78</td><td>45.16</td></tr><tr><td>GLM-5.3-Flash (320B) GLM-5-Team et al. (2026)</td><td>44.07</td><td>54.26</td><td>43.77</td><td>44.90</td><td>46.61</td><td>55.32</td><td>43.74</td><td>46.18</td><td>45.54</td></tr><tr><td>Kimi-K3 (2.8T) Team et al. (2026a)</td><td>49.15</td><td>64.89</td><td>45.48</td><td>49.04</td><td>46.82</td><td>64.89</td><td>46.22</td><td>48.26</td><td></td></tr><tr><td>Qwen3.8-Flash-Next (125B) Qwen Team (2026a;d)</td><td>47.46</td><td>67.02</td><td>56.97</td><td>53.42</td><td>52.97</td><td>67.02</td><td>58.45</td><td>56.67</td><td>48.65 55.05</td></tr></table>

Statistics. ZoomGainBench contains 1,000 questions over 941 images: 472 attribute recognition, 94 counting, and 434 OCR questions. Attribute questions have 4 to 6 answer options; counting and OCR questions are open-ended. Each question has English and Chinese versions (2,000 evaluation instances in total) and an automatically obtained evidence region specified by its crop box. Appendix C shows examples of all three question categories. Table 1 compares ZoomGainBench with existing benchmarks: it is the only one that combines fully model-authored QAs, automated verification, and dedicated distractor construction, and Section 5.2 shows how these distractors reveal errors shared across models.

Closed-source Open-source Fine-grained enhanced Agentic All models (c)

(a) Global-view errors attract choices

71.7% of errors vs. 54.5% expected

![](images/c93eb79d638a21f20a03243c5e12ce92a2d63313c488c5ea651f0ad22c5f7652.jpg)

![](images/b97eea85522002e9c1ec85a8727c51cdece5c919b1bfaae2a826cabeffb97a1b.jpg)  
(c) Small targets are hard Accuracy by anchor-area quartile

![](images/e3e6ee58bfc5916a79b4622466e2001300eeb26a9d698078a95928019171420e.jpg)  
Figure 3: Where models fail on ZoomGain Bench. (a) How often each model group chooses each type of distractor compared with the random baseline, considering only incorrect answers. Values above 1 indicate over-selection. Global-view error + blind guess denotes distractors produced both as incorrect answers when models view the full image and as guesses based on language priors without an image. (b) How often the 26 models not used in construction (all evaluated models except Qwen3.5-397B-A17B and Qwen3-VL-235B-A22B-Instruct) choose global-view errors, grouped by each error’s frequency in the 30 full-image answers collected during construction. (c) Accuracy by quartile of the anchor’s area in the full image.

Evaluation protocol. Multiple-choice questions are scored by exact match on the selected option, and counting questions by exact match on the predicted integer. OCR questions receive their ANLS score if it is at least 0.7, and zero otherwise. Prompts, decoding temperature, and thinking-mode settings for each model are provided in Appendix F.

## 5.1 Benchmark results.

ZoomGain challenges even frontier VLMs. The strongest model, GPT-5.6-Sol, reaches an overall score of only 61.08 (Table 2), and only five models exceed 50. All four closed-source models outperform every open-source general VLM: even GPT-5.5 (56.79) surpasses Qwen3.8-Flash-Next (55.05), the best open-source general model.

Fine-grained enhanced VLMs are highly competitive. ZwZ-8B and Vision-OPD-9B improve over their respective base models, Qwen3-VL-8B and Qwen3.5-9B, by 18.47 and 14.00 points. Despite its smaller size, ZwZ-8B (47.85) also surpasses Qwen3.5-397B-A17B (45.16).

Attribute recognition is the most difficult category. Attribute recognition yields the lowest score averaged over both languages for 23 of 28 models, despite being the only multiple-choice category, with a uniformguessing accuracy of about 21.5%. Qwen3-VL-8B, for example, falls below this level in both languages (15.89 in English and 19.92 in Chinese), motivating an analysis of which distractors attract incorrect choices.

## 5.2 Error analysis

Models are misled by global-view errors. Since every distractor retains its source from construction (Section 3.3), we can trace where incorrect choices go. Across all 28 models, global-view-error distractors receive 71.7% of incorrect choices, exceeding the random baseline of 54.5%,<sup>1</sup> whereas blind guesses and visual evidence receive only 0.54 and 0.59 times their random-baseline (Figure 3(a)). This over-selection holds for every model, including those not involved in construction Moreover, wrong answers that appear more often during construction are also chosen more often by other models. Among the 26 models not used in construction, errors that appeared 1–2 times in the 30 full-image answers are chosen 17.7% of the time, versus 46.0% for those that appeared more than 10 times (Figure 3(b)). That is, models reproduce the global-view confusions observed during construction, guessing a scene-plausible answer instead of resolving the queried detail, such as naming a hockey equipment brand for the logo on a hockey player’s pants (shown in Figure 8(a) in Appendix E). In other words, distractors mined from global-view errors capture confusions shared across models, rather than idiosyncrasies of the models used in construction.

Table 3: Results of RL training on ZoomGain and ZwZ data. $\Delta _ { \mathrm { Z o o m G a i n - Z w Z } }$ and $\Delta _ { \mathrm { Z o o m G a i n - B a s e } }$ denote score gains over the ZwZ-trained and base models, respectively; Avg. is computed over the 13 held-out benchmarks, excluding ZoomGain Bench.
<table><tr><td rowspan="2">Model</td><td>Held-in</td><td colspan="6">Fine-grained Perception</td><td colspan="3">General Perception</td><td colspan="3">|Difficult Percept &amp; Reason|</td><td rowspan="2">Avg.</td></tr><tr><td>|ZoomGain|</td><td>Zoom Bench</td><td>V* HR Bench Bench 4K</td><td>HR MME-RW 8K</td><td>Tree Lite</td><td>Visual Bench Probe</td><td>CV- Bench</td><td>MMStar MMVP</td><td></td><td>Vision</td><td>Baby Perception Bench</td><td>O3- Bench</td><td>Only Held-out</td></tr><tr><td colspan="10">Several Large Parameter VLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.1</td><td>23.22</td><td>47.22</td><td>70.1667.00 65.25</td><td></td><td>45.86</td><td>47.65</td><td>29.90</td><td>84.22</td><td>71.60</td><td>32.67 13.92</td><td></td><td>28.97</td><td>31.01</td><td>48.88</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>40.51</td><td>49.11</td><td>87.96 84.50 81.62</td><td></td><td>58.16</td><td>53.58</td><td>46.41</td><td>86.72</td><td>76.33</td><td>64.00</td><td>18.30</td><td>35.50</td><td>60.58</td><td>61.75</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>41.11</td><td>51.83</td><td>84.82 86.50 83.75</td><td></td><td>57.89</td><td>54.07</td><td>44.85</td><td>88.06</td><td>74.60</td><td>70.00</td><td>19.07</td><td>38.80</td><td>65.22</td><td>63.04</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>45.16</td><td>54.91</td><td>87.43 86.50 85.12</td><td></td><td>57.06</td><td>53.33</td><td>49.51</td><td>89.57</td><td>76.00</td><td>72.67</td><td>20.36</td><td>47.50</td><td>72.17</td><td>65.55</td></tr><tr><td>GLM-5.3-Flash (320B) Gemini-3-Flash</td><td>45.54 51.27</td><td>56.80</td><td>89.01</td><td>87.00 84.75</td><td>62.12</td><td>60.25</td><td>48.35</td><td>88.94</td><td>79.47</td><td>74.67</td><td>20.10</td><td>41.90</td><td>65.51</td><td>66.07</td></tr><tr><td></td><td></td><td>59.29</td><td>86.39</td><td>87.88 85.00</td><td>55.86</td><td>48.64</td><td>49.71</td><td>89.57</td><td>83.60</td><td>82.67</td><td>34.51</td><td>43.50</td><td>71.01</td><td>67.51</td></tr><tr><td colspan="10">Our Experimental Models</td><td colspan="7"></td></tr><tr><td>Qwen3.5-4B</td><td>30.03</td><td>48.88</td><td></td><td>81.68 81.25 77.63</td><td>57.89</td><td>49.38</td><td>43.30</td><td>85.99</td><td>66.80</td><td>57.33</td><td>14.69</td><td>33.10</td><td>54.20</td><td>57.86</td></tr><tr><td>w/ZwZ 9K</td><td>40.99</td><td>57.63</td><td></td><td>86.91 81.25 79.00</td><td>57.43</td><td>45.19</td><td>52.43</td><td>86.58</td><td>68.93</td><td>53.33</td><td>10.82</td><td>30.43</td><td>56.23</td><td>58.94</td></tr><tr><td>w/ ZoomGain 9K</td><td>55.81</td><td>59.88</td><td></td><td>90.05 87.25 83.88</td><td>61.59</td><td>49.14</td><td>57.28</td><td>88.02</td><td>72.53</td><td>61.33</td><td>21.13</td><td>36.23</td><td>63.19</td><td>63.96</td></tr><tr><td>∆ZoomGain-ZwZ</td><td>+14.83</td><td>+2.25</td><td>+3.14 +6.00 +4.88</td><td></td><td>+4.17</td><td>+3.95</td><td>+4.85</td><td>+1.44</td><td>+3.60</td><td>+8.00</td><td>+10.31</td><td>+5.80</td><td>+6.96</td><td>+5.03</td></tr><tr><td>∆ZoomGain-Base</td><td>+25.78</td><td></td><td></td><td>+11.00 +8.37 +6.00 +6.25</td><td>+3.70</td><td>-0.24</td><td>+13.98</td><td>+2.03</td><td>+5.73</td><td>+4.00</td><td>+6.44</td><td>+3.13</td><td>+8.99</td><td>+6.10</td></tr><tr><td>Qwen3.5-9B</td><td>32.13</td><td>50.30</td><td></td><td>86.91 82.13 79.63</td><td>58.52</td><td>49.88</td><td>44.66</td><td>86.27</td><td>71.73</td><td>60.00</td><td>15.98</td><td>34.20</td><td>56.52</td><td>59.75</td></tr><tr><td>w/ZwZ 9K</td><td>44.07</td><td>58.93</td><td>90.58</td><td>85.5082.38</td><td>61.70</td><td>47.90</td><td>52.62</td><td>88.16</td><td>73.27</td><td>63.33</td><td>17.27</td><td>34.13</td><td>57.10</td><td>62.53</td></tr><tr><td>w/ ZoomGain 9K</td><td>55.48</td><td>60.36</td><td>93.19 90.25 85.50</td><td></td><td>63.31</td><td>53.33</td><td>56.31</td><td>88.58</td><td>73.87</td><td>70.00</td><td>17.53</td><td>39.00</td><td>68.12</td><td>66.10</td></tr><tr><td>∆ZoomGain-ZwZ</td><td>+11.41</td><td>+1.43</td><td></td><td>+2.62 +4.75 +3.13</td><td>+1.62</td><td>+5.43</td><td>+3.69</td><td>+0.42</td><td>+0.60</td><td>+6.67</td><td>+0.26</td><td>+4.87</td><td>+11.01</td><td>+3.58</td></tr><tr><td>∆ZoomGain-Base</td><td>+23.35</td><td></td><td></td><td>+10.06 +6.28 +8.12 +5.87</td><td>+4.79</td><td>+3.45 +11.65</td><td></td><td>+2.31</td><td>+2.14</td><td>+10.00</td><td>+1.55</td><td>+4.80</td><td>+11.60</td><td>+6.35</td></tr></table>

Smaller targets are harder. Accuracy rises with the fraction of the image occupied by the anchor for every model group (Figure 3(c)): averaged over all models, it increases from 28.9% on the smallest quartile, where anchors occupy less than 0.4% of the image, to 55.2% on the largest.

Appendix D further reports the error analysis regarding open-ended questions, including how incorrect counts differ from the ground truth, as well as common OCR error patterns.

## 6 Experiments on ZoomGain Training Data

## 6.1 Main Results

Setup. We conduct the experiment on the image-matched ZoomGain and ZwZ sets described in Section 4, each with 9,000 samples. We select Qwen3.5-4B and Qwen3.5-9B Team (2026) as our base models, applying RL with GRPO Shao et al. (2024) using each set under identical settings (batch size 64, 8 rollouts per question; other details are shown in Appendix G). We evaluate on four groups of benchmarks. Held-in: ZoomGain Bench. Held-outfine-grained perception: ZoomBench Wei et al. (2026), V\* Bench Wu & Xie (2023), HR-Bench 4K and 8K Wang et al. (2025b), MME-RealWorld-Lite Zhang et al. (2024), TreeBench Wang et al. (2025a), and VisualProbe Lai et al. (2025). General perception: CV-Bench Tong et al. (2024a), MMStar Chen et al. (2024), and MMVP Tong et al. (2024b). Difficult perception and reasoning: BabyVision Chen et al. (2026), PerceptionBench Lin et al. (2026), and O3-Bench Li et al. (2026a). We also compare with some large parameter VLMs, including open-source and closed-source Google DeepMind (2025); OpenAI (a). We report the average score over the 13 held-out benchmarks, excluding ZoomGain Bench. Because ZoomGain Bench and the ZoomGain training data are generated by the same generation pipeline, including it could favor ZoomGain model.

Results. As shown in Table 3, models trained on ZoomGain data consistently outperform their ZwZ-trained counterparts on all 14 benchmarks. Since the two training sets differ solely in the questions, these gains can be attributed to the questions constructed by ZoomGain. Moreover, the improvements brought by ZoomGain extend across all benchmark groups. Compared with the base models, the gains of ZwZ are mainly concentrated on fine-grained perception. In contrast, ZoomGain improves all three groups. The advantage of ZoomGain over ZwZ is most pronounced on difficult perception and reasoning, where the 9B model gains 11.01 points on O3-Bench and the 4B model gains 10.31 points on BabyVision. These results indicate that ZoomGain data enables models to acquire transferable fine-grained perception capabilities.

Table 4: Ablation of incremental evidence and dedicated distractors. Avg. is computed over the 10 held-out benchmarks.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Incremental Dedicated Evidence</td><td rowspan="2">Distractors</td><td rowspan="2">Held-in</td><td colspan="6">Fine-grained Perception</td><td colspan="3">|Difficult Perception &amp; Reason</td><td rowspan="2">Avg. Only Held-out</td></tr><tr><td>ZoomGain |Zoom</td><td>V* Bench Bench</td><td>HR 4K</td><td>HR 8K</td><td>MME-RW</td><td>Tree Visual</td><td>Baby Vision</td><td>Perception Bench</td><td>O3- Bench</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td>Bench 30.03</td><td>48.88</td><td>81.68</td><td>81.25 77.63</td><td>Lite 57.89</td><td>Bench 49.38</td><td>Probe 43.30</td><td>14.69</td><td></td><td>54.20</td><td>54.20</td></tr><tr><td>+ZoomGain 9K</td><td>X</td><td></td><td>49.71</td><td>56.45</td><td>88.48</td><td>85.50 81.75</td><td>60.24</td><td>48.40</td><td>52.82</td><td>18.56</td><td>33.10 34.70</td><td>58.84</td><td>58.57</td></tr><tr><td>+ZoomGain 9K</td><td>√</td><td>x</td><td>46.85</td><td>57.04</td><td>90.05</td><td>85.75 80.25</td><td>62.69</td><td>48.89</td><td>54.76</td><td>17.53</td><td>34.97</td><td>59.42</td><td>59.14</td></tr><tr><td>+ZoomGain 9K</td><td></td><td></td><td>55.81</td><td>59.88</td><td>90.05</td><td>87.25 83.88</td><td>61.59</td><td>49.14</td><td>57.28</td><td>21.13</td><td>36.23</td><td>63.19</td><td>60.96</td></tr></table>

Remarkably, with only 9K training samples, the 9B model achieves an average score of 66.10, surpassing Qwen3.5-397B-A17B (65.55) and performing on par with GLM-5.3-Flash (66.07), while the 4B model (63.96) also surpasses Qwen3.5-122B-A10B and the ZwZ-trained 9B model. Moreover, ZoomGain-trained models achieve the best results among all compared models on 6 of 7 fine-grained perception benchmarks. However, we also notice that the strongest large models still lead on general perception and difficult perception and reasoning, suggesting that these tasks still rely heavily on model scale and reasoning capability.

## 6.2 Ablation Study

Setup. We ablate two core designs of ZoomGain on Qwen3.5-4B. All variants use 9K training samples, with the same RL settings and evaluation protocol as in Table 3. (1) Remove incremental evidence: instead of requiring questions to be based on incremental information that appears only in the zoomed-view caption, we generate questions directly from the zoomed-view caption. The source images, candidate QA pool size, and 9K sample selection procedure are otherwise identical to those of the full version. (2) Remove dedicated distractors: we keep each question, its answer, the correct option’s position, and the number of options unchanged, and replace only the distractors of each multiple-choice question with correct answers from other training questions in the same answer category (e.g., color or material). OCR and counting questions are open-ended and are unaffected by this ablation.

Results. As shown in Table 4, the results demonstrate that both designs contribute to training performance. The full ZoomGain pipeline achieves the highest average score across the 10 held-out benchmarks. Removing any component incude the performance losses on almost all benchmarks. Both ablations incur the largest losses in the difficult perception and reasoning group. The full version improves over the base model by an average of 6.19 points across these three benchmarks, while each ablation retains only about half of this gain. This indicates that the two designs mainly improve the capability of perception and reasoning that transfer beyond the training distribution.

The effect of incremental evidence is most apparent on fine-grained benchmarks: removing it reduces scores on ZoomBench and VisualProbe by 3.43 and 4.46 points, respectively. This indicates that restricting questions to evidence revealed only after zooming forces the model to learn to locate and recognize details hard to discern. Meanwhile, removing dedicated distractors causes the largest drop on the held-in ZoomGainBench, at 8.96 points. It is expected because the benchmark’s options are constructed using the same pipeline. Using dedicated distractors introduces harder confusions, forcing the model to attend more closely to visual details and thereby enhancing its perception capabilities under the same data budget.

## 7 Conclusion

To address the difficulty of constructing reliable fine-grained visual supervision at scale, we propose Zoom-Gain, a data construction pipeline that exploits the perception gap between global and zoomed-in views. ZoomGain turns details revealed only by zooming into questions about the full image, verifies their answers on the zoomed view with multiple models, and mines multi-source distractors that could mislead VLMs. With this pipeline, we construct ZoomGainBench, a challenging benchmark whose questions and answers are entirely model-authored. Through the analysis of failure cases, we find that different VLMs share similar confusions when fine details are unresolved in the full image. Experiments further show the effectiveness of ZoomGain data as training supervision, which consistently improves VLMs’ capabilities of fine-grained perception, general perception, and difficult perception and reasoning.

## References

Anthropic. Introducing Claude Opus 5. https://www.anthropic.com/news/claude-opus-5, July 2026.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad – a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

Remi Cadene, Corentin Dancette, Matthieu Cord, Devi Parikh, et al. Rubi: Reducing unimodal biases for visual question answering. Advances in Neural Information Processing Systems, 32:841–852, 2019.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3558–3568, 2021.

Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Yiping Bao, et al. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 27056–27087. Curran Associates, Inc., 2024. doi: 10.52202/079017-0850. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 2f8ee6a3d766b426d2618e555b5aeb39-Paper-Conference.pdf.

Meida Chen, Qingyong Hu, Zifan Yu, Hugues Thomas, Andrew Feng, Yu Hou, Kyle McCullough, Fengbo Ren, and Lucio Soibelman. Stpls3d: A large-scale synthetic and real aerial photogrammetry 3d point cloud dataset. arXiv preprint arXiv:2203.09065, 2022.

Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, et al. Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062, 2025.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

Paul Gavrikov, Wei Lin, M. Jehanzeb Mirza, Soumya Jahagirdar, Muhammad Huzaifa, Sivan Doveh, Serena Yeung-Levy, James Glass, and Hilde Kuehne. VisualOverload: Probing visual understanding of vlms in really dense scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026.

GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian

Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

Google DeepMind. Gemini 3 Flash Model Card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-Flash-Model-Card.pdf, December 2025.

Google DeepMind. Gemini 3.6 Flash Model Card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-6-Flash-Model-Card.pdf, July 2026.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, and Guohai Xu. Deepeyesv2: Toward agentic multimodal model. In International Conference on Learning Representations, volume 2026, pp. 114851–114872, 2026.

Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Associationfor Computational Linguistics: EMNLP 2024, pp. 5042–5063, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.290. URL https: //aclanthology.org/2024.findings-emnlp.290/.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal ofcomputer vision, 123(1):32–73, 2017.

Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv:2509.07969, 2025.

Kaican Li, Lewei Yao, Jiannan Wu, Tiezheng Yu, Jierun Chen, Haoli Bai, Lu Hou, Lanqing Hong, Wei Zhang, and Nevin L. Zhang. Insight-o3: Empowering multimodal foundation models with generalized visual search. In The Fourteenth International Conference on Learning Representations, 2026a.

Pengyu Li, Zhitao Gao, Lingling Zhang, Muye Huang, Yuanming Li, Fangzhi Xu, and Jun Liu. Visualopsd: Cross-modal on-policy self-distillation for efficient unified multimodal reasoning. arXiv preprint arXiv:2606.18974, 2026b.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 292–305, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.20. URL https: //aclanthology.org/2023.emnlp-main.20/.

Zichao Lin, Yifeng Xie, Bowen Qu, Haiming Wang, Jia Li, Haoning Wu, Yuhao Dong, Zuhao Yang, Jinguo Zhu, Haoyu Lu, et al. Perceptionbench: Evaluating atomic visual perception in multimodal large language models. arXiv preprint arXiv:2607.24957, 2026.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

OpenAI. GPT-5.1. https://developers.openai.com/api/docs/models/gpt-5.1, a. Accessed: 2026-09-26.

OpenAI. GPT-5.5. https://developers.openai.com/api/docs/models/gpt-5.5, b.

OpenAI. GPT-5.6 Sol. https://developers.openai.com/api/docs/models/gpt-5.6-sol, c.

Qwen Team. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and training stability. Technical report, Alibaba Group, August 2026a.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026b. URL https://qwen.ai/ blog?id=qwen3.6-27b.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026c. URL https://qwen.ai/blog? id=qwen3.8.

Qwen Team. Qwen3.8-Flash-Next: A new architecture, towards ultimate cost-efficiency, August 2026d. URL https://qwen.ai/blog?id=qwen3.8-flash-next.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026e. URL https://qwen.ai/ blog?id=qwen3.6-35b-a3b

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4035–4045, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1437. URL https://aclanthology.org/D18-1437/.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35: 25278–25294, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 6613–6629, 2025.

Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel space reasoning via curiosity-driven reinforcement learning. Advances in Neural Information Processing Systems, 38:8222–8251, 2026.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, Kexin Chen, Peng Chen, Ruijue Chen, Wentao Chen, Xin Chen, Yang Chen, Yanru Chen, Yifei Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Dazhi Cheng, Yean Cheng, Jialei Cui, Jingbing Cui, Anqi Dai, Jiaqi Deng, Hao Ding, Rui Ding, Shaofeng Ding, Mengfan Dong, Mengnan Dong, Yuhao Dong, Yuxin Dong, Angang Du, Chenzhuang Du, Dikang Du, Jusen Du, Yulun Du, Yu Fan, Jing Feng, Qiulin Feng, Yichen Feng, Kelin Fu, Qiang Fu, Fuxuan Gao, Hongcheng Gao, Jingyue Gao, Tong Gao, Weijia Gao, Shangyi Geng, Jie Gong, Linhu Gong, Shengao Gong, Xiaochen Gong, Qizheng Gu, Yicheng Gu, Shuhao Guan, Haiqing Guo, Shiqi Guo, Xiang Guo, Zhengyan Guo, Beixi Hao, Wenxin Hao, Xiaoru Hao, Dailan He, Haotian He, Lehan He, Qi He, Weiran He, Xinran He, Xinyi He, Yibo He, Yunjia He, Chao Hong, Tiange Hong, Hao Hu, Jiaxi Hu, Ruikun Hu, Weiming Hu, Yangyang Hu, Zhenxing Hu, Liang Hua, Jinbin Huang, Ke Huang, Ruiyuan Huang, Siying Huang, Weixiao Huang, Yan Huang, Zhengjie Huang, Zhiqi Huang, Yulong Hui, Chaobo Jia, Yutong Jiang, Zhejun Jiang, Zuoyou Jiang, Wenyi Jin, Xinyi Jin, Yu Jing, Huanjun Kong, Guokun Lai, Aidi Li, Cheng Li, Chengyuan Li, Cong Li, Fang Li, Guanyu Li, Haoyang Li, Jia Li, Junxiong Li, Lei Li, Letian Li, Lincan Li, Weihong Li, Wentao Li, Xintong Li, Yang Li, Yishen Li, Yiwei Li, Yuxiao Li, Zhaowei Li, Zhaoxi Li, Zheming Li, Zhengxiao Li, Zhiyuan Li, Jiawei Lin, Xiaohan Lin, Yibo Lin, Zichao Lin, Ziyan Lin, Bill Liu, Boxiao Liu, Chuan Liu, Liang Liu, Shaowei Liu, Shudong Liu, Shuran Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yanming Liu, Yibo Liu, Yipeng Liu, Zhengying Liu, Zhiheng Liu, Enzhe Lu, Haoyu Lu, Linqiang Lu, Tingzhan Lu, Zhiyuan Lu, Aotian Luo, G. Luo, Junyu Luo, Yifan Luo, B. Lyu, Wenzhou Lyu, Shaoguang Mao, Yuan Mei, Xin Men, Minqing Ni, Yixuan Niu, Siyuan Pan, Shujun Peng, Zhangyang Qi, Ruoyu Qin,

ZeChao Qin, Zeyu Qin, Haiquan Qiu, Jianxin Qiu, Jiezhong Qiu, Bowen Qu, Yuhao Qu, Zeyu Shang, Youbo Shao, Han Shen, Jincheng Shi, Juanfeng Shi, Lidong Shi, Shengyuan Shi, Wingchun Siu, Pengwei Song, Xiaoxi Song, Jianlin Su, Yunfeng Su, Zhaochen Su, Lin Sui, Jingsong Sun, Junyao Sun, Shaoning Sun, Shuzhe Sun, Tongyu Sun, Yujun Sun, Yunpeng Tai, Chuning Tang, Heyi Tang, Sirui Tang, Zecheng Tang, Chaoran Tian, Rongpeng Tian, Yu Tian, Wei Tu, Chensi Wang, Chuang Wang, Chunjie Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Hao Wang, Hao Wang, Huaqing Wang, Hui Wang, Jiayi Wang, Jinglong Wang, Jinhong Wang, Jiuzheng Wang, Linian Wang, Shaobo Wang, Shenzhi Wang, Shuyi Wang, Si Wang, Siyuan Wang, Tianfu Wang, Wenjue Wang, Xingran Wang, Xinmei Wang, Xinyuan Wang, Xusheng Wang, Yalin Wang, Yangkun Wang, Yao Wang, Yaoyu Wang, Yejie Wang, Yiqin Wang, Yucheng Wang, Yuzhi Wang, Zhaoji Wang, Zhaowei Wang, Zhengtao Wang, Zhenhao Wang, Zhongsheng Wang, Zifan Wang, Chu Wei, Ming Wei, Shouxin Wei, Zichen Wen, Fan Wu, Haoning Wu, Rucong Wu, Wenhao Wu, Xiaoxue Wu, Yingcong Wu, Yongqi Wu, Yuxin Wu, Zijian Wu, Xinglang Xian, Chenxuan Xiang, Yuye Xiang, Bocheng Xiao, Chenjun Xiao, Xin Xiao, Jin Xie, Xiaotong Xie, Yifeng Xie, Zhe Xie, Bowei Xing, Yiming Xiong, Baosheng Xu, Boyu Xu, Jiale Xu, Jianfan Xu, Jing Xu, Jinjing Xu, L. H. Xu, Qingtao Xu, Shuyao Xu, Suting Xu, Tiantian Xu, Tianxiang Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ye Xu, Yueni Xu, Ziyao Xu, Haonan Xue, Junjie Yan, Yaoyao Yan, Fan Yang, Guangyao Yang, Hao Yang, Junwei Yang, Ruoyu Yang, Wenjie Yang, Xiaofei Yang, Xinyu Yang, Yi Yang, Yiling Yang, Ying Yang, Yuchen Yang, Zhen Yang, Zhilin Yang, Zian Yang, Zuhao Yang, Haotian Yao, Dan Ye, Haoran Ye, Wenjie Ye, Zhanbo Ye, Bohong Yin, Haoxiang Yin, Xietong Yin, Chengzhen Yu, Haozhen Yu, Longhui Yu, Shengnan Yu, Shuying Yu, Tianxiang Yu, Enming Yuan, Mengjie Yuan, Tongtian Yue, Wei Yue, Yang Yue, Dunyuan Zha, Haobing Zhan, B. H. Zhang, Dehao Zhang, Fei Zhang, Hao Zhang, Haoyuan Zhang, Huanyu Zhang, Jiapei Zhang, Jiaxuan Zhang, Jin Zhang, Kaiyi Zhang, Miaozhen Zhang, Puqi Zhang, Qinglei Zhang, Rong Zhang, Rui Zhang, Shaoshuai Zhang, Shiyi Zhang, Xiaobin Zhang, Xiaoyun Zhang, Y. Zhang, Yangkun Zhang, Ye Zhang, Yichi Zhang, Yikun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Zijing Zhang, Bin Zhao, Chenguang Zhao, Feifan Zhao, Jinglun Zhao, Jinxiang Zhao, Shuai Zhao, Wenshuo Zhao, Xiangyu Zhao, Xuanle Zhao, Yikai Zhao, Zijia Zhao, Haozhi Zheng, Huabin Zheng, Ruihan Zheng, Shaojie Zheng, Tengyang Zheng, Haofeng Zhong, Lei Zhong, Longguang Zhong, M. Zhou, Qiankang Zhou, Runjie Zhou, Ruozhang Zhou, Xinyu Zhou, Yiqiao Zhou, Zaida Zhou, Jinguo Zhu, Liya Zhu, Xinhao Zhu, Yangjunfeng Zhu, Yuxuan Zhu, Zhen Zhu, Chen Zhuang, Weiyu Zhuang, and Xinxing Zu. Kimi k3: Open frontier intelligence, 2026a. URL https://arxiv.org/abs/2607.24653.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, Yanru Chen, Yanxu Chen, Yicun Chen, Yimin Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Ziwei Chen, Dazhi Cheng, Minghan Chu, Jialei Cui, Jiaqi Deng, Muxi Diao, Hao Ding, Mengfan Dong, Mengnan Dong, Yuxin Dong, Yuhao Dong, Angang Du, Chenzhuang Du, Dikang Du, Lingxiao Du, Yulun Du, Yu Fan, Shengjun Fang, Qiulin Feng, Yichen Feng, Garimugai Fu, Kelin Fu, Hongcheng Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Chengyang Gong, Xiaochen Gong, Zhuoma Gongque, Qizheng Gu, Xinran Gu, Yicheng Gu, Longyu Guan, Yuanying Guo, Xiaoru Hao, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Jiaxi Hu, Yangyang Hu, Zhenxing Hu, Ke Huang, Ruiyuan Huang, Weixiao Huang, Zhiqi Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yu Jing, Guokun Lai, Aidi Li, C. Li, Cheng Li, Fang Li, Guanghe Li, Guanyu Li, Haitao Li, Haoyang Li, Jia Li, Jingwei Li, Junxiong Li, Lincan Li, Mo Li, Weihong Li, Wentao Li, Xinhang Li, Xinhao Li, Yang Li, Yanhao Li, Yiwei Li, Yuxiao Li, Zhaowei Li, Zheming Li, Weilong Liao, Jiawei Lin, Xiaohan Lin, Zhishan Lin, Zichao Lin, Cheng Liu, Chenyu Liu, Hongzhang Liu, Liang Liu, Shaowei Liu, Shudong Liu, Shuran Liu, Tianwei Liu, Tianyu Liu, Weizhou Liu, Xiangyan Liu, Yangyang Liu, Yanming Liu, Yibo Liu, Yuanxin Liu, Yue Liu, Zhengying Liu, Zhongnuo Liu, Enzhe Lu, Haoyu Lu, Zhiyuan Lu, Junyu Luo, Tongxu Luo, Yashuo Luo, Long Ma, Yingwei Ma, Shaoguang Mao, Yuan Mei, Xin Men, Fanqing Meng, Zhiyong Meng, Yibo Miao, Minqing Ni, Kun Ouyang, Siyuan Pan, Bo Pang, Yuchao Qian, Ruoyu Qin, Zeyu Qin, Jiezhong Qiu, Bowen Qu, Zeyu Shang, Youbo Shao, Tianxiao Shen, Zhennan Shen, Juanfeng Shi, Lidong Shi, Shengyuan Shi, Feifan Song, Pengwei Song, Tianhui Song, Xiaoxi Song, Hongjin Su, Jianlin Su, Zhaochen Su, Lin Sui, Jinsong Sun, Junyao Sun, Tongyu Sun, Flood Sung, Yunpeng Tai, Chuning Tang, Heyi Tang, Xiaojuan Tang, Zhengyang Tang, Jiawen Tao, Shiyuan Teng, Chaoran Tian, Pengfei Tian, Ao Wang, Bowen Wang, Chensi Wang, Chuang Wang, Congcong Wang, Dingkun Wang, Dinglu Wang, Dongliang Wang, Feng Wang, Hailong Wang, Haiming Wang, Hengzhi Wang, Huaqing Wang, Hui Wang, Jiahao Wang, Jinhong Wang, Jiuzheng Wang, Kaixin Wang, Linian Wang, Qibin Wang, Shengjie Wang, Shuyi Wang, Si Wang, Wei Wang, Xiaochen Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yipu Wang, Yiqin Wang, Yucheng Wang, Yuzhi Wang, Zhaoji Wang, Zhaowei Wang, Zhengtao Wang, Zhexu Wang, Zihan Wang, Zizhe Wang, Chu Wei, Ming Wei, Chuan Wen, Zichen Wen, Chengjie Wu, Haoning Wu, Junyan Wu, Rucong Wu, Wenhao Wu,

Yuefeng Wu, Yuhao Wu, Yuxin Wu, Zijian Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Yuchong Xie, Yifei Xin, Bowei Xing, Boyu Xu, Jianfan Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinbo Xu, Xinran Xu, Yangchuan Xu, Yichang Xu, Yuemeng Xu, Zelai Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Guangyao Yang, Hao Yang, Junwei Yang, Kai Yang, Ningyuan Yang, Ruihan Yang, Xiaofei Yang, Xinlong Yang, Ying Yang, Yi Yang, Yi Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Dan Ye, Wenjie Ye, Zhuorui Ye, Bohong Yin, Chengzhen Yu, Longhui Yu, Tao Yu, Tianxiang Yu, Enming Yuan, Mengjie Yuan, Xiaokun Yuan, Yang Yue, Weihao Zeng, Dunyuan Zha, Haobing Zhan, Dehao Zhang, Hao Zhang, Jin Zhang, Puqi Zhang, Qiao Zhang, Rui Zhang, Xiaobin Zhang, Y. Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yushun Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Chenguang Zhao, Feifan Zhao, Jinxiang Zhao, Shuai Zhao, Xiangyu Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng, Ruihan Zheng, Shaojie Zheng, Tengyang Zheng, Junfeng Zhong, Longguang Zhong, Weiming Zhong, M. Zhou, Runjie Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Liya Zhu, Xinhao Zhu, Yuxuan Zhu, Zhen Zhu, Jingze Zhuang, Weiyu Zhuang, Ying Zou, and Xinxing Zu. Kimi k2.5: Visual agentic intelligence, 2026b. URL https://arxiv.org/abs/2602.02276.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv.org/abs/2507.01006.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024a.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024b.

Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, and Zhaoxiang Zhang. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv preprint arXiv:2507.07999, 2025a.

Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang, Li Shen, Yong Luo, Bo Du, and Dacheng Tao. Retrieval-augmented perception: High-resolution image perception meets visual rag. In Forty-second International Conference on Machine Learning. URL https://arxiv.org/abs/2503.01222.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025b.

Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, et al. Zooming without zooming: Region-to-image distillation for fine-grained multimodal perception. arXiv preprint arXiv:2602.11858, 2026.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. Dota: A large-scale dataset for object detection in aerial images. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. URL https://arxiv.org/abs/2506.03569.

Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-opd: Learning to see fine details for multimodal llms via on-policy self-distillation. arXiv preprint arXiv:2605.18740, 2026.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-pro: A more robust multi-discipline multimodal understanding benchmark. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 15134–15186, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.736. URL https://aclanthology.org/2025.acl-long.736/.

Kangning Zhang, Yixing Li, Shuai Shao, Qingyao Li, Zhengxi Lu, Zhiyuan Yao, Jianghao Lin, Wenxiang Jiao, Yuan Lu, Weiwen Liu, et al. Vad: Attributing visual evidence for target reconstruction in multimodal on-policy distillation. arXiv preprint arXiv:2607.28590, 2026a.

Lu Zhang, Jiazuo Yu, Haomiao Xiong, Ping Hu, Yunzhi Zhuge, Huchuan Lu, and You He. Finers: Finegrained reasoning and segmentation of small objects with reinforcement learning. Advances in Neural Information Processing Systems, 38:67403–67426, 2026b.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024.

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think beyond images, 2025. URL https://arxiv.org/abs/2508.11630.

Ziwei Zheng, Minghao Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, and Chao Shen. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. In International Conference on Learning Representations, volume 2026, pp. 126775–126798, 2026.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A Complete Algorithm and Additional Implementation Details

Section 3 describes the core pipeline. Here we add two implementation details, incorporated into Algorithm 1.   
Model assignments and thresholds are given in Appendix $\mathrm { B } ;$ the prompts are collected in Appendix J.

Anchor markers in the zoomed view. The expanded crop $V _ { z } ^ { i }$ may contain several objects of the anchor’s category. A VLM enumerates these instances, including partially visible ones, and matches the selected anchor to its grounded box. If multiple instances are present, we draw a red rectangle around the anchor itself; if only one is present, the crop remains unmarked. This rectangle is distinct from $b _ { i } ,$ which marks the entire crop region in the global view. The marked-view caption prompt uses it only to identify the anchor and otherwise follows the unmarked-view instructions. It must not be described as photographed content or used as an answer. Local questions may refer to the marker; global questions use natural referring expressions. Figure 4 shows both cases.

Joint crop/full-image audit and rewriting. After QA generation, EDGE\_REWRITER inspects the crop and the full photograph with $b _ { i }$ marked, together with both question forms and the reference answer. It checks whether answer-bearing text or objects continue beyond the crop, whether the global referent is unique, and whether both questions admit the same answer. It returns keep, rewrite, or drop. A bounded rewrite can add answer-free qualifiers valid in both views while preserving the non-OCR answer. For OCR, it may narrow the question and answer to a complete text block or specified prefix/suffix contained in the original answer, but cannot add text read from the full image. Counting questions whose requested set extends outside the crop are discarded. Revisions are synchronized across the local/global and English/Chinese questions, and target and answer constraints are checked again before crop-based verification (Section 3.3). Both anchor enumeration and this audit use Qwen3.5-397B-A17B.

(a) Multiple instances  
![](images/8d7622c5d7edb9d62323c1130ccf1fb0d82fba7e3524a8ef4c118e9c4d045e1d.jpg)  
Anchor marker added 2 people in crop

(b) Single instance  
![](images/9029c852c683b8021fdc51d95c00fb2e1529498c853720d600866979c59c35de.jpg)  
Figure 4: Conditional anchor markers in the zoomed view. (a) Two people are present, including the partially visible person on the left; the red rectangle selects the person wearing a red-and-black plaid shawl. (b) Only one person is present, so no marker is added. Both views are rendered from the original images and recorded crop/anchor coordinates using the pipeline’s marker and resizing functions. The rectangle identifies the anchor rather than the answer-bearing detail; it is used only in the zoomed view and is not part of the photographed scene.

Algorithm 1 ZoomGain Data Construction   
Require: Image pool $\scriptstyle { \mathcal { T } } ;$ configured models; $N = 1 0$ samples per model; R = 3 MCQ verification rounds   
Ensure: Verified QA dataset D with evidence regions and distractor provenance   
1: $\mathcal { D }  \emptyset$   
2: for each full image $V _ { g } \in \mathcal { T }$ do   
3: $C _ { g } \gets \mathrm { S C E N E } \bar { \mathrm { C A P T I O N } } ( V _ { g } ) ; \mathcal { E } \gets \mathrm { I }$ ROPOSEANCHORS $( C _ { g } )$   
4: for each referring expression $e _ { i } \in \mathcal { E }$ do   
// Anchor discovery and grounding   
5: $\mathcal { B } _ { i } \gets \mathrm { G R O U N D I N D E P E N D E N T L Y } ( V _ { g } , e _ { i } )$   
6: $\mathbf { i f \longrightarrow } \mathrm { V A L I D C R O U N D I N G } ( \mathcal { B } _ { i } , 0 . 5 )$ then continue   
7: $b _ { i } \gets \mathrm { E x P A N D } ( \operatorname { m e d i a n } _ { \operatorname { c o o r d } } ( \mathcal { B } _ { i } ) ) ; V _ { z } ^ { i } \gets \mathrm { C R O P } ( V _ { g } , b _ { i } )$   
8: $L _ { i }$ ← ENUMERATEANCHORINSTANCES $( V _ { z } ^ { i } , e _ { i } )$   
9: if no instance matches the grounded anchor then continue   
10: $\widetilde { V } _ { z } ^ { i } \gets \mathbf { M a R K A N C H O R I F M U L T I P L E } ( V _ { z } ^ { i } , L _ { i } )$   
// Global-to-zoom evidence and QA construction   
11: $V _ { g } ^ { i } \gets \sf D R A W B O \mathrm { x } ( V _ { g } , b _ { i } )$   
12: $C _ { g } ^ { \dot { i } }  M _ { \mathrm { c a p } } ( V _ { g } ^ { i } , b _ { i } ) ; C _ { z } ^ { i }  M _ { \mathrm { c a p } } ( \widetilde { V } _ { z } ^ { i } )$ $/ /$ Independent calls   
13: $\Delta _ { i }$ ← INCREMENTALFACTS $( C _ { g } ^ { i } , C _ { z } ^ { i } )$   
14: $( q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , a _ { i } , t _ { i } ) \gets \mathrm { G E N E R A T E Q A } ( \Delta _ { i } , e _ { i } , L _ { i } )$   
15: if no supported, unambiguous QA is generated then continue   
16: $( v , q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , a _ { i } )$ ← EDGEAUDITANDREWRITE $( \widetilde { V } _ { z } ^ { i } , V _ { g } ^ { i } , q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , a _ { i } )$   
17: if v = drop or post-rewrite constraints fail then continue   
// Quality verification and multi-source distractors   
18: $A _ { \mathrm { c r o p } } \gets \mathsf { S A M P L E } ( \mathrm { c r o p } , \widetilde { V } _ { z } ^ { i } , q _ { \mathrm { l o c a l } } ^ { i } , N )$   
19: if mi $\cap _ { m } { \mathrm { C O R R E C T } } \bigl ( A _ { \mathrm { c r o p } } ^ { m } , a _ { i } \bigr ) \ \stackrel { \_ } { < } \ 6$ then continue   
20: $A _ { \mathrm { b l i n d } }  \mathrm { S A M P L E } ( \mathrm { b l i n d } , \mathcal { D } , q _ { \mathrm { g l o b a l } } ^ { i } , N )$   
21: if mi $\boldsymbol { \mathbf { \lambda } } _ { ^ { 1 } m }$ CORREC $\Gamma ( A _ { \mathrm { b l i n d } } ^ { m } , a _ { i } ) \geq 3$ then continue   
22: $O _ { i }  \emptyset ; S _ { i }  \emptyset$   
23: if $t _ { i } =$ attribute then   
24: $A _ { \mathrm { g l o b a l } }  \mathrm { S A M P L E } ( \mathrm { g l o b a l } , V _ { g } , q _ { \mathrm { g l o b a l } } ^ { i } , N )$   
25: $D _ { v } \gets \mathrm { V I S U A L A L T E R N A T I V E S } ( V _ { g } , \widetilde { V } _ { z } ^ { i } , q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , a _ { i } )$   
26: $D _ { b } \gets$ INCORRECT $( A _ { \mathrm { b l i n d } } , a _ { i } ) ; D _ { g } $ INCORRECT $( A _ { \mathrm { g l o b a l } } , a _ { i } )$   
27: $( O _ { i } , S _ { i } ) \mathrel { \mathop {  } } \mathrm { C U R A T E O P T I O N S } ( a _ { i } , D _ { v } , D _ { b } , D _ { g } )$   
28: verified ← false   
29: for $r = 1 , \ldots , R$ do   
30: A<sub>mcq</sub> ← SAMPLE(mcq, $\widetilde { V } _ { z } ^ { i } , ( q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , O _ { i } ) , N )$   
31: if min CORRECT $\left( A _ { \mathrm { m c q } } ^ { m } , a _ { i } \right) \geq 6$ then   
32: verified ← true; break   
33: end if   
34: if $r < R$ then $( O _ { i } , S _ { i } ) $ REPAIRDISTRACTORS $( O _ { i } , S _ { i } )$   
35: end for   
36: if ¬verified then continue   
37: end if   
38: $\mathcal { D }  \mathcal { D } \cup \{ ( V _ { g } , V _ { z } ^ { i } , b _ { i } , q _ { \mathrm { l o c a l } } ^ { i } , q _ { \mathrm { g l o b a l } } ^ { i } , a _ { i } , t _ { i } , O _ { i } , S _ { i } ) \}$   
39: end for   
40: end for   
41: return D

Notation and operations. $L _ { i }$ records the anchor instances; $\widetilde { V } _ { z } ^ { i }$ is the optionally marked zoomed view. EDGEAUDITANDREWRITE returns a verdict v and the retained or revised QA under the constraints above. Both question forms have English and Chinese versions. The remaining operations follow Section 3, with verification settings in Appendix B; $S _ { i }$ preserves distractor provenance during repairs.

## B Construction Settings

Table 5 provides the complete model assignments for the ZoomGain pipeline described in Section 4.

Table 5: Model assignments in the ZoomGain construction pipeline.
<table><tr><td>Model</td><td>Roles</td></tr><tr><td>Qwen3.5-397B-A17B Team (2026)</td><td>Initial global scene captioning and paired-view captioning  $( M _ { \mathrm { { c a p } } } ) ;$  grounding; crop anchor enumeration; joint crop/full-image audit and rewriting; blind answering; global-view error collection; visual distractor generation; final MCQ verification.</td></tr><tr><td>GLM-5.2 GLM-5-Team et al. (2026)</td><td>Incremental fact extraction and QA generation.</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct Bai et al. (2025)</td><td>Grounding; crop-based verification; blind answering; global-view error col- lection; final MCQ verification.</td></tr><tr><td>Kimi-K2.6 Team et al. (2026b)</td><td>Grounding; crop-based verification; blind answering; global-view error col- lection; final MČQ verification.</td></tr><tr><td>Qwen3.5-122B-A10B Team (2026)</td><td>Answer-equivalence judging.</td></tr><tr><td>DeepSeek-V4-Pro DeepSeek-AI (2026)</td><td>Candidate distractor clustering and option-pool curation.</td></tr></table>

Verification Thresholds. All verification stages sample 10 answers per model. Crop-based verification requires at least 6 correct answers from each of its two models, Qwen3-VL-235B-A22B-Instruct and Kimi-K2.6. Blind verification rejects a question only when all three models each answer correctly at least 3 times. Final MCQ verification requires at least 6 correct selections from each of its three models.

Verification Rounds and Distractor Repair. Final MCQ verification uses at most three rounds: an initial check followed by up to two further checks after distractor repairs. Each round applies the per-model threshold described above. Candidates that still fail after the final round are discarded.

## C Benchmark Question Categories

The three question categories in ZoomGainBench cover the following fine-grained details. Figures 5–7 show three examples per category, with paired English and Chinese questions. The orange boxes mark the specific details needed to answer each question, with dashed lines connecting them to enlarged views. These boxes are selected for illustration and are distinct from the pipeline’s anchor boxes.

Attribute Recognition. Questions concern a target’s color, material, shape, object identity, actions, or relations. The examples below ask about a truck’s mirror housing, a glass tabletop, and a small logo on a golfer’s visor.

![](images/e6e65ca686be614fb4fdf17908d589323ffca8ae07db80883903fc63aa21ae57.jpg)  
Figure 5: Attribute recognition examples. (a) Mirror housing color; (b) tabletop material; (c) logo shape. Correct answers and their corresponding options are highlighted in green.

Counting. Questions ask for the number of entities, object parts, or repeated patterns. The examples below cover repeated images on a menu, people seated on a bench, and pockets on a shirt.

(a) Repeated images  
![](images/5d679fe50463826f446d994f004c80d75d4714f2b96f7ce82e162824d3f997e9.jpg)  
#000283

(b) People  
![](images/3a40c375d57f69533cd9cd2877f02fb209d20df4ab299f6876e3b264f09766a8.jpg)

How many burger images are displayed in the grid on the left portion of the digital menu boards in the background?

背景中数字菜单板左侧部分显示的网格中有多少张汉堡  
图片？  
ANSWER 8

#000400

How many individuals are seated on the bench inside the white shelter structure in the center-left background of the soccer field?

ANSWER 3

在足球场中左后方的白色遮棚结构内，有多少人坐在长凳上？

(c) Object parts  
#000862  
![](images/6e45b2b6197d484ee15a9eabdb80e6914ca5d11b72b21623e6759531e81b658b.jpg)  
Figure 6: Counting examples. (a) Eight burger images on a menu; (b) three people seated inside a shelter; (c) two chest pockets on a shirt. Each question requires an open-ended numerical answer.

How many chest pockets are on the blue polo shirt worn by the man in the center holding a champagne bottle?

画面中央手持香槟瓶的男子所穿的蓝色Polo衫上有几个胸前口袋？

Text Recognition (OCR). Questions require transcribing text on signs, vehicles, clothing, printed materials, and product labels. The examples below include a road sign, Japanese lettering on a truck, and Chinese text on a bottle label. Answers retain the original text visible in the image, regardless of the question language.

(a) Road sign  
![](images/2a14593766bb1d99c91ee52b201218b2654ab4bfe76cfac690c39b5e6e9de6a2.jpg)

(b) Vehicle lettering  
![](images/a92021075f612105e96ee253a2fb185136008b76370c08f21686ef40bb2c53d2.jpg)  
#000313

What text is displayed on the white supplementary plate mounted directly above the red circular no stopping sign on the right sidewalk in the mid-ground?

右侧人行道中景处红色圆形禁止停车标志正上方的白色  
辅助标牌上显示的文字是什么？  
ANSWER 100M

#000996

What Japanese text is printed in the largest bold black characters on the rear right door of the white box truck on the right side of the image?

图像右侧的白色厢式货车右后内上以最大的粗体黑色字  
符印有什么日文文字？  
ANSWER

(c) Product label  
#000384  
![](images/f644d3a0749245453a4dce2edd3058aa79644b3eba89e9987476d95856f4bca7.jpg)  
Figure 7: Text recognition examples. (a) A supplementary road sign; (b) lettering on a truck’s rear door; (c) the largest Chinese text on a bottle label. Reference transcriptions are shown in green.

What is the largest Chinese text printed on the dark blue label of the small plastic water bottle in the lower-right foreground?

右下角前景中的小塑料水瓶上深蓝色标签上印有的最大  
的中文字是什么？  
ANSWER

## D Additional Error Analysis

This section complements Section 5.2.

Per-model consistency. Every evaluated model over-selects global-view errors: they receive 65.0–78.2% of each model’s incorrect choices, against 52.3–57.8% expected under uniform choice among the distractors of each question. For the four closed-source models, which took no part in constructing the benchmark, the shares are 75.7% versus 57.4%. Their selection rate of a global-view error also rises with how consistently it arose during construction, from 13.4% when given by one or two of the 30 full-image answers to 35.2% when given more than ten times.

Controlling for option position. Models prefer some option positions over others. Normalizing by the average selection rate of each option position, global-view errors are still selected 1.34 times as often as expected, and 1.46 times when they also coincide with blind guesses, compared with 0.52 and 0.55 times for blind guesses and visual evidence.

Below-chance accuracy of Qwen3-VL-8B. Only 1.5% of Qwen3-VL-8B’s answers to attribute questions cannot be parsed, so its below-chance accuracy does not stem from formatting failures. Instead, 67.5% of its incorrect choices fall on global-view errors, against 54.0% expected. Its responses typically claim to inspect the target closely and then state a plausible but incorrect attribute, for example describing a dark blue backpack as black.

Errors on open-ended questions. Counting errors are mostly near misses: 71.6% of incorrect numeric answers are off by exactly one, and 67 of the 94 counting questions have a ground truth of at most three, so the difficulty lies in resolving a few small instances rather than in enumerating many objects. For OCR, among transcriptions without full credit, 29.1% remain close to the reference, with a character-level similarity of at least 0.5; 10.1% omit part of the target text or include adjacent text; and at least 13.4% transcribe other text in the scene instead of the target, as identified by matching predictions against the scene descriptions generated during construction. Most of the remaining transcriptions bear little resemblance to the reference.

## E Representative Failure Cases

Figure 8 shows representative failures discussed in Section 5.2.

## F ZoomGainBench Evaluation Settings

All non-agentic models use the same system prompt as ZwZ Wei et al. (2026), which asks the model to reason before placing its final answer within <answer> and </answer> tags. The user message contains the full image and the question with its options, if applicable. Counting questions additionally request Arabic numerals, while OCR questions request only the transcription within the answer tags, preserving capitalization, spacing, punctuation, and symbols. The Chinese version replaces the question stem while retaining the options and formatting instructions.

Qwen models, ZwZ, and Vision-OPD use a temperature of 0 and a maximum output length of 2,048 tokens, with native thinking disabled via enable\_thinking=false. MiMo-VL-7B-RL uses the same temperature and output limit, with /no\_think appended to the question. GLM-4.6V and GLM-5.3-Flash use a temperature of 0 and output limits of 4,096 and 8,192 tokens, respectively; their requests also set enable\_thinking=false. Kimi-K3 is evaluated with thinking enabled, a temperature of 0, and an output limit of 8,192 tokens. GPT-5.5, GPT-5.6-Sol, Gemini-3.6-Flash, and Claude-Opus-5 use an output limit of 4,096 tokens and retain the provider defaults for temperature and thinking mode.

The four agentic models use their native reasoning and tool-use prompts with multi-turn image interactions, all at a temperature of 0. The per-turn output limits are 10,240 tokens for DeepEyes and PixelReasoner, 20,480 for DeepEyesV2, and 2,048 for Thyme. The total output budget is 32,768 tokens per question. DeepEyes, DeepEyesV2, and PixelReasoner allow up to 11 turns, while Thyme allows 5; the tool-call limit is 10 for all except PixelReasoner, which allows 4. Generic answer-format instructions that conflict with the native agentic protocols are removed.

## G Training and Evaluation Settings

Training. Both datasets are trained with direct RL using GRPO, with a batch size of 64, 8 rollouts per question, and one epoch (140 steps), and the checkpoint at step 140 is evaluated. For both ZwZ and

![](images/bdb8e6a9721c3e8130820989262c9d3fc9d5b088d975cad7e6868fa686fad9a9.jpg)

![](images/cc40c8c5e35aca42fbaf1604aeea82f11141fe71b745c73aac65aa2db3df7541.jpg)  
What color is the rectangular object resting on the dashboard inside the windshield of the white truck with a canvas cargo cover parked on the street?

## (a) Logo recognition

![](images/0addec15656d4aeba4beb16b981a966371d5b4bf3f57aa2d56295e20da851ad7.jpg)

![](images/2964d539d7b09f8f72fd3b52d6ef27faae7e0a5d84a2f655dae8f40017939738.jpg)  
What logo appears on the thigh of the dark blue pants worn by the hockey player to the left of the yellow stretcher on the ice in the center foreground and closer to the camera?  
Distractor source: global-view error

(b) Bag type  
![](images/5a0ded46fb1c8f9c29af669859f7eb79950949441dde02f990e8c33c554f1ab4.jpg)  
What type of bag does the woman walking toward the camera in the lower-right foreground carry in her right hand?

![](images/887c90cfe316304b60565a6e15dbe0a729a015e5c7aa2bf63006caca20faa961.jpg)  
Distractor source: global-view error

## (c) Object color

## (d) Reflected structure

![](images/a12cb6de62ebf065e23f7a9e20f3e8ed6395b2842c59bad10925464514549c47.jpg)

![](images/e7437aa7b27209a677a16f18737074e172f3e863c2109965e3974111537bed7c.jpg)  
What structure is visible near the base of the mast of the reflected construction crane in the lower-left glass facade?  
Distractor source: global-view error

Counts refer to 28 models x 2 languages; other responses are not shown.

Figure 8: Representative failures on ZoomGainBench. Each case shows the full image with the queried detail outlined in red, an enlarged view of that region, and the responses of the 56 evaluations (28 models × 2 languages). (a) Nearly all evaluations name Bauer, a hockey equipment brand, instead of the NHL logo, while the blind guesses (Reebok, Nike) are never chosen. (b) The white tote bag is taken for a plastic bag; the visual-evidence distractors, drawn from the black bag on her shoulder and from another pedestrian’s purse, are never chosen. (c) Most evaluations (46/56) call the white object on the truck’s dashboard orange, likely the color of the vest worn by the worker in front of the truck. (d) The salient reflected silhouettes draw choices away from the smaller crane at the base of the mast.

ZoomGain, we use a learning rate of $1 0 ^ { - 6 }$ , no KL regularization, and a maximum response length of 2,046 tokens. ZwZ images retain their native bounding-box overlays, whereas ZoomGain uses the original images. For ZwZ, we follow the official reward definitions Wei et al. (2026). For ZoomGain, rewards follow the ZoomGainBench scoring rules: multiple-choice and counting questions receive a reward of 1 for an exact match on the selected option or predicted integer, respectively, and 0 otherwise. OCR questions receive their ANLS score as the reward if it is at least 0.7, and 0 otherwise.

Evaluation. For the training experiments, all models are evaluated with chain-of-thought prompting and a temperature of 0. Within each benchmark, base and trained models share the same prompt profile and output limit: 2,048 new tokens for all benchmarks except BabyVision, PerceptionBench, and O3-Bench, which use 8,192 tokens and are scored by GPT-5.5 as the answer judge. MMVP is scored by paired accuracy, CV-Bench by its official balanced score, and MME-RealWorld on 1,919 questions. VisualProbe is reported as the question-weighted average of its easy (141), medium (268), and hard (106) splits.

## H Global-to-Zoom Evidence and QA Construction Examples

Figures 9–11 illustrate the construction process in Section 3.2 for the three question types. Each example shows the paired views and caption excerpts, selected entries from the original incremental fact list, and how one fact leads to a local QA pair and then a global QA pair. Green text marks details newly described in the zoomed-view caption; purple text highlights the local and global target references. The zoomed views use the original pipeline crops.

## A small logo becomes a definite shape

![](images/752a0304c985cec9cff9db9e959e5f245168790a6765967c9eb5b0a9afcd49e4.jpg)  
Caption excerpts and selected facts from the generation record. Green: details added by zooming. Purple: target reference.

Figure 9: Attribute QA construction example. The global-view caption mentions logos on a visor without specifying their shapes. The zoomed-view caption identifies a small star-shaped logo. This fact supports a shape question, which is rewritten using the global referring expression while preserving the answer.

COUNTING Food images become an exact count  
![](images/9a66260593d48b4e0d03c860be43a761575bd64380c146562cb7baf186da4117.jpg)  
Caption excerpts and selected facts from the generation record. Green: details added by zooming. Purple: target reference.

Figure 10: Counting QA construction example. The global-view caption describes food images on digital menu boards without giving a count. The zoomed-view caption identifies eight burger images arranged in two rows of four. The selected fact yields a counting question with the same answer in both views.

OCR A blue label becomes readable text  
![](images/1e963d09688fc9f15b195f6769bad1007c7132552119ac9175e47456cf602340.jpg)  
Caption excerpts and selected facts from the generation record. Green: details added by zooming. Purple: target reference.

Figure 11: OCR QA construction example. The global-view caption describes a bottle with a blue label without transcribing its text. The zoomed-view caption reads the Chinese characters on the label. The selected text fact supports a local OCR question, which is rewritten to identify the bottle in the full image while retaining the original transcription as the answer.

## I Quality Verification Examples

Figures 12 and 13 show candidate QA pairs rejected by the two verification stages in Section 3.3. Crop-based verification checks whether the image supports the generated reference answer, while blind verification filters questions whose answers are predictable without an image. All questions, candidate answers, and vote counts are taken from the original generation records.

## Crop verification: rejecting incorrect reference answers

Models answer from the crop; agreement is checked against the generated reference.

![](images/9340f024b78a2932456cd87a544462ac29c105cce5fa8b9f1c605eaa66cdead9.jpg)  
Gate: retain only if each crop model matches the candidate reference in at least 6/10 samples. Both cases fail with 0/10 per model. Ref. matches are the original pipeline judge scores.  
Both original crop inputs have no red anchor marker. Orange detail boxes are added for display only.

Figure 12: Examples rejected by crop-based verification. The captions incorrectly describe (A) the race-car digit as “3” instead of “8” and (B) the door handle as silver instead of white. Both verification models give the visually supported answer in all 10 samples, yielding zero matches to the generated reference answer and failing the 6/10 threshold. The original crop inputs have no red anchor markers; orange boxes and enlarged details are added only for display.

Candidate reference: a drumstick

## QUALITY VERIFICATION

## Blind verification: rejecting answers predictable without images

Models receive the question only. The images below are shown for visual inspection.

A A familiar object-role association  
![](images/322d2fca47ca82f3e09f9d5c903d65efc03c38d1f791c24bf2cf581c55b83e34.jpg)  
Drumsticks are visible; the word "drummer" also makes the answer predictable.  
What is the drummer wearing a cap in the background on the right side of the image holding in each hand?

<table><tr><td colspan="3">Crop check: 10/10 for each of the two VLMs.</td></tr><tr><td>Blind model</td><td>Main response</td><td>Ref. matches</td></tr><tr><td>Qwen3.5</td><td>Drumsticks</td><td>10/10</td></tr><tr><td>Qwen3-VL</td><td>Drumsticks</td><td>10/10</td></tr><tr><td>Kimi-K2.6</td><td>Drumsticks</td><td>10/10</td></tr></table>

## B A familiar visual convention

![](images/a2e464e65ba50d070070478fa73489cc5dc1ed6004b74eab9ca2718ed736967e.jpg)  
Steam is drawn above the cup; the wording also strongly suggests steam.

## Question (no image supplied)

<table><tr><td colspan="3">Crop check: 10/10 for each of the two VLMs.</td></tr><tr><td>Blind model</td><td>Main response</td><td>Ref. matches</td></tr><tr><td>Qwen3.5</td><td>Steam</td><td>10/10</td></tr><tr><td>Qwen3-VL</td><td>Steam</td><td>10/10</td></tr><tr><td>Kimi-K2.6</td><td>Steam</td><td>9/10</td></tr></table>

Gate: reject if all three blind models match the reference in at least 3/10 samples. Case A: 30/30 total; case B: 29/30. Ref. matches are the original pipeline judge scores.

Blue boxes locate the displayed crops. Images were not supplied to the blind models.

Figure 13: Examples rejected by blind verification. The reference answers, (A) drumsticks and (B) steam, are supported by the images, but are also predictable from the questions alone. Without images, the three models obtain 30/30 and 29/30 reference matches in total, respectively, with every model meeting the rejection threshold of at least 3/10 matches. The images and blue crop boxes are shown only for the reader; no images are supplied to the blind models.

## J Prompts Used in Data Construction

This section provides the prompts for the roles described in Section 3, plus anchor enumeration, marked/ unmarked zoomed-view captioning, and EDGE\_REWRITER from Appendix A. Templates are taken from zoom\_- inc\_pipeline/prompts.py; view-dependent instruction blocks are taken from their callers in pipeline.py. Sample-specific placeholders such as {anchor}, {question}, and {reference} are filled at runtime. JSON braces are shown as literal braces rather than Python string escapes. A line of the form «<INSERT: NAME»> denotes exact insertion of the named block below, allowing shared instructions to be printed once without omitting their content. The insertion notation is for presentation only and is not sent to the models.

View-specific inputs. CAPTION1 receives the full image with the crop region marked. CAPTION2 receives an unmarked zoomed view, while CAPTION2\_ANCHOR\_MARKED prepends marker-specific instructions to the same caption template. The local anchor rule in QA generation is selected according to whether the zoomed view has an anchor marker. EDGE\_REWRITER receives the crop and the full image with its crop region marked; a candidate box on the crop can additionally locate the referent being audited. Crop solvers answer the local question; full-image and blind solvers answer the global question, with no image supplied for blind answering. The Kimi format instruction is appended to its open-answer requests. Visual distractor generation uses the appropriate crop/full-view rules. MCQ verification sees the crop, local question, global question as context, and assembled choices.

## J.1 Shared instructions

## SPATIAL\_FRAME\_RULES.

SPATIAL FRAME CONTRACT — apply to every direction word you read or write:

\- LEFT/RIGHT BETWEEN OBJECTS uses the viewer's frame of the image: 'the person to the left of the sign' means left as seen by the viewer. Never mirror-compute the reference object's own left/right when locating one object relative to another. When such a phrase is load-bearing, prefer explicit wording such as 'on the left side of the image'.

\- BODY PARTS are anatomical, never viewer-relative: 'right hand' always means the person's own right hand. Derive the side from the person's facing direction, never from raw image position: facing the camera -> the right hand appears on the viewer's LEFT; facing away from the camera -> the right hand appears on the viewer's RIGHT; facing image-left -> the left side of the body is toward the camera; facing image-right -> the right side of the body is toward the camera. If the facing direction is not clearly visible, never use left/right body-part wording; identify the part another way (by what it holds or touches, or with explicit viewer wording such as 'the hand on the left side of the image'). - FRONT/BEHIND is NEVER a depth word. Scene depth must be written only as 'closer to the camera (than X)' / 'farther from the camera (than X)' / 'beyond X from the camera's viewpoint'. The words front/behind are reserved exclusively for the intrinsic front/back sides of people, animals, and vehicles whose facing you have already stated: 'in front of the seated man' is the side his face and chest point toward, and 'behind him' is his back side, regardless of distance to the camera. - DEPTH-FLIP TRAP (the most common mistake — check every sentence): when the camera sees a person's BACK, that person faces AWAY, so objects deeper in the scene are IN FRONT of that person, and objects between the camera and the person are behind them. Writing 'behind the man' for a sign that lies beyond a man seen from the back is WRONG — write 'in front of the man (farther from the camera)' or 'farther from the camera than the man'. The same flip applies to seated crowds watching something away from the camera.

\- For objects with no inherent front (poles, boxes, trees, candles, signs), never use front/behind at all; use camera-distance or image-position wording.

\- VEHICLE AND BUILDING PARTS use part terms fixed to the object itself: 'the cab end', 'the rear of the trailer', 'the side facing the camera', 'the entrance side'. Never call a vehicle end 'front' unless it is the actual forward/cab end of that vehicle.

\- GROUPS and entities whose facing cannot be determined: use the viewer frame with explicit image wording ('in the upper-left part of the image', 'farther from the camera'), never intrinsic terms. - WHOSE FRAME 'THE IMAGE' MEANS. Image-frame wording — 'in the upper-left part of the image', 'near the left edge of the image', '图像左侧' — always refers to the FULL photograph, and to nothing else.

## CROP\_FRAME\_CAVEAT.

CROP FRAME CAVEAT:

\- Your view is a window cut out of the middle of a larger photograph, so its borders are not the photograph's borders and its left side is not the photograph's left side. A position read off this view may be stated only as local to it, never as image-frame wording.

\- A FRAME SIDE IS ONLY HONEST NEAR AN EDGE. 'on the left side of the image' is worth saying about something in the outer third of the photograph; for anything nearer the middle it reads as a claim the object is not where it is, and a solver scanning that side will not find it. When the target sits toward the centre, locate it against a named object instead — 'just right of the fire hydrant', 'in front of the bus shelter and a little to its left' — which stays true wherever the frame is drawn. You cannot see where this crop sits in the photograph, so when in doubt prefer the named object. - FRAME SUPERLATIVES REQUIRE THE WHOLE PHOTOGRAPH. leftmost, rightmost, topmost, bottommost, far left, f<sub>ar</sub> <sub>r</sub>i<sub>g</sub>h<sub>t,</sub> 最左<sub>,</sub> 最右<sub>,</sub> 最上<sub>,</sub> 最下<sub>,</sub> 最顶<sub>,</sub> 最底 <sub>assert</sub> <sub>t</sub>h<sub>at</sub> <sub>not</sub>hi<sub>ng</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> ki<sub>n</sub>d li<sub>es</sub> f<sub>urt</sub>h<sub>er</sub> <sub>out,</sub> <sub>w</sub>hi<sub>c</sub>h only a view of the entire photograph can establish. Without that view do not write one in either language. Bounded superlatives inside one named host ('the bottom word on the blue sign') are unaffected.

\- A HOST'S OWN CORNER NEEDS THAT CORNER IN VIEW. 'the bottom right corner of the window', 'the top of <sub>t</sub>h<sub>e s</sub>i<sub>gn</sub>'<sub>,</sub> '货架最下层' <sub>name a pos</sub>i<sub>t</sub>i<sub>on</sub> i<sub>ns</sub>id<sub>e anot</sub>h<sub>er o</sub>bj<sub>ect, an</sub>d <sub>t</sub>h<sub>ey are on</sub>l<sub>y true</sub> if <sub>THAT o</sub>bj<sub>ect</sub>'<sub>s</sub> edge on that side is visible here. When the host runs past the border, the corner you are looking at is the CROP's corner, not the host's, and the phrase points at something else entirely in the full photograph: a bookcase cropped to its top two shelves makes 'the book at the bottom of the bookcase' name a green spine on the second shelf, while the bookcase's real bottom shelf holds a red one. Use such a phrase only when you can see where the host ends on that side; otherwise identify the thing by what it is attached to, what is next to it, or its own visible attributes.

## KIMI\_FINAL\_ANSWER\_TAG.

Begin your reply with the words The answer is, then put only your final answer inside <answer></answer>. For example: The answer is <answer>answer here</answer>.

## J.2 Scene captioning and anchor grounding

## SCOUT\_CAPTION.

Describe this image thoroughly. Then list every distinct, individually identifiable object or structure you can see (buildings, boats, vehicles, people, signs, statues, gates, bridges, etc.). Do not list background materials such as sky, water, grass, foliage or pavement as objects. Direction wording: give every position in the image frame ('on the left side of the image', 'in the lower-right foreground'). Express depth only as foreground / background / closer to or farther from the camera. Never use 'in front of' or 'behind' a person, animal, or vehicle to mean nearer to or farther from the camera, and never name a person's own left/right body side.

## PHRASE\_REWRITE.

You are preparing referring expressions for a visual grounding model.

Below is a description of an image. Extract up to {max\_phrases} candidate landmark objects and rewrite each as a SELF-CONTAINED English referring phrase that would pick out exactly one object instance for someone looking at the image.

## Rules:

\- Each phrase must denote exactly ONE object instance. Add a position word (left / center / right, foreground / background) or a color/size attribute whenever the bare noun could match several instances.

\- Position words inside a phrase must be IMAGE-FRAME words only: left / center / right, upper / lower, foreground / background, closest to the camera. NEVER locate a candidate with 'in front of', 'behind', 'ahead of', or 'beyond' another object or person — depth intent is ambiguous and grounding models disagree on it. Never use a person's own left/right body side, and never locate one object through another object that the description places at a clearly different depth.

\- Only concrete, discrete objects: buildings, boats, vehicles, signs, towers, gates, statues, distinct groups. NEVER: sky, water, grass, trees, foliage, pavement, shadows, reflections, generic crowds. - Phrases must rely only on properties visible in the description; do not invent.

\- NEVER use written text, letters, logos or brand names read from the image as a qualifier (small text is unreliable at low resolution).

- Output strict JSON only:

```jsonl
{"phrases": [{"phrase": "...", "head_noun": "...", "qualifier_type": "position|color|size|none"}]}
```

{caption}

Return exactly one JSON object with these fields:   
- "count": a non-negative integer equal to the number of detected instances.   
- "instances": a JSON array containing exactly "count" objects.   
Each object must contain:   
- "bbox\_2d": four integers [x1, y1, x2, y2].   
- "description": a short, neutral, answer-free visual description.

## GROUND\_NORM1000.

Locate the object referred to by: "{phrase}"   
Interpret left/right/upper/lower/foreground/background in the phrase as the viewer's image frame, and foreground/background as distance from the camera.   
Measure x and y on an integer coordinate grid from 0 to 1000 relative to the full image width and height. Return only one tight bounding box in this form: [x1, y1, x2, y2]   
All four values must be integers between 0 and 1000. Include the complete visible extent of the referred object. If it is absent or ambiguous, return []. Do not output any other text.

## GROUND\_NORM01.

Interpret left/right/upper/lower/foreground/background in the phrase as the viewer's image frame, and foreground/background as distance from the camera.

Measure x and y as decimal coordinates from 0 to 1 relative to the full image width and height, where 0 is the left/top edge and 1 is the right/bottom edge. Return only one tight bounding box wrapped in tags, in this form:

All four values must be between 0 and 1. Include the complete visible extent of the referred object. If it is absent or ambiguous, return <bbox>[]</bbox>. Do not output any other text.

## J.3 Anchor enumeration and paired-view captioning

## ANCHOR\_INSTANCE\_CHECK.

Find EVERY distinct visible instance of the concrete object category "{noun}" in this image crop.   
Include partially visible instances.

For each instance, draw one tight bounding box around only its visible extent. Do not infer or include hidden parts. Use integer coordinates normalized from 0 to 1000 in [x1, y1, x2, y2] order.

The following consistency rule is mandatory: count == len(instances)

If no instance is visible, return exactly: {"count": 0, "instances": []}

Think carefully before answering. Output JSON only. Do not use Markdown fences or add explanations.

## CAPTION1.

You are given a full image with one highlighted red rectangular region.   
Describe \*\*only the visual content inside the highlighted region and never mention the rest of the image.\*\* Use the rest of the image only as context.

The pipeline has pre-selected one anchor object inside this region (localization metadata only): "{anchor}"

ANCHOR-CENTERED ORGANIZATION (same convention as the zoomed view): treat this anchor as the primary reference point of the description, not as its exclusive scope. First identify the anchor and state every attribute of it that is discernible at this scale, then cover its immediate neighborhood, then the rest of the region. Whenever visually supported, express other objects' positions relative to this anchor. The anchor phrase is localization metadata only: never copy an attribute, text, count, or relation from the phrase unless you can independently see it in the image. If several objects match the phrase, describe each of them and say the reference is ambiguous; if you cannot confidently locate the referent, say so and describe the region normally. The anchor never shrinks your coverage duty: everything else plainly visible in the region must still be described.

YOUR CAPTION DEFINES WHAT THE FULL-IMAGE VIEW CAN SEE. A later stage compares this caption against a zoomed-in description of exactly the same region and treats every attribute you omitted as "not visible at full-image scale". The zoomed description follows the same spatial and inclusion rules as this one; the only intended difference between the two descriptions is resolution. Omitting something that is in fact plainly visible therefore corrupts the comparison. Be exhaustive about everything you can genuinely discern at this scale, and explicit about what you cannot.

EXHAUSTIVENESS CONTRACT (work from large to small):

\- For every large or prominent object in the region — especially people — state ALL attributes that are plainly discernible at this scale: garment types and their colors (shirt, pants, dress, hat, shoes), object colors and materials, obvious pose or action, obviously identifiable held or attached items, and coarse spatial relations between the described objects. A large foreground person whose shirt color is obvious must have that shirt color stated.

\- Then cover mid-sized objects the same way.

\- For small objects and fine details (small text, small items, precise counts, subtle patterns, distant faces), report what is discernible and explicitly say "unclear" or "not readable at this scale" for the rest. Saying unclear is valuable signal; silently skipping is not.

\- Do not compress multiple distinct objects into one vague summary sentence when their individual attributes are visible.

\- When several instances of one category are present (price tags, signs, windows), bind each described instance to a stable qualifier — attachment or nearest-object relation, a part location on its host, or an explicit image-frame position — instead of the bare category noun.

<<<INSERT: SPATIAL\_FRAME\_RULES>>>

PERSON ORIENTATION: before naming a person's own left/right body side or a side-specific held item, state which way that person faces and derive the side per the contract above. When the facing is not discernible at this scale, use viewer wording ('the hand on the viewer's left') or say the side is unclear; never guess an anatomical side.

RECTANGLE-EDGE POLICY (aligned with the zoomed view): the rectangle boundary is an artificial viewport, not photographed scene content. For a discrete object that crosses the boundary, apply exactly one branch:

\- If it cannot be confidently identified, or no complete part of it lies inside the region, omit it entirely.

\- If it is clearly identifiable and at least one of its parts is completely visible inside the region, you MAY describe exactly those fully visible parts and their discernible attributes; say "unclear" for their small details. Never state whole-object attributes, whole-body actions, or totals for such an object, and never describe the part outside the region even though the full image shows it — it is outside this region's scope.

\- Never mention the clipping itself anywhere in the caption: no "partially visible", "cut off", or "at the edge" wording. Describe what is inside, say unclear for what is too small, and stay silent about the rest.

\- Large scene surfaces or structures may be described when a substantial, meaningful portion lies inside the region; do not describe their truncation.

Before returning the caption, perform a mandatory deletion check: remove every sentence that mentions the rectangle boundary or an object's clipped state.

If a small detail is not readable at this scale, explicitly say it is unclear. Do not guess counts, text, colors, or actions that are not visually clear.

## CAPTION2.

You are given an image crop constructed around one selected anchor object.

Selected anchor object (localization metadata only):

"{anchor}"

Write a detailed natural-language caption organized around this selected anchor and its visible neighborhood. Treat the anchor as the primary reference point, not as the exclusive scope of the caption.

Coverage and organization:

\- First describe the selected anchor itself in detail, including every clearly visible part, content, marking, readable text, color, attribute, condition, or action that belongs to it.

\- Then describe every other clearly visible object and detail in the image. Do not omit reliable visual information merely because it is not part of the anchor.

\- Whenever visually supported, express each other object's relationship to the anchor, preferring frame-free devices: attached to it, contained in it, on it, nearest to it, or beside it at the same depth. When a direction word is needed, it must follow the SPATIAL FRAME CONTRACT below: left/right between objects are the viewer's (write 'to the viewer's left of the anchor', never the anchor's own left/right), above/below are image-vertical, front/behind may only name the intrinsic sides of a person, animal, or vehicle whose facing you have stated, and any depth difference is written as closer to / farther from the camera.

\- If no reliable relationship to the anchor can be established, still describe the visible object with a neutral location within this crop. Never invent a relationship merely to connect it to the anchor. - DEPTH ZONES (mandatory): this crop is a 2D window that may span very different scene depths. For every non-anchor object you describe, make its depth relation to the anchor explicit: in the anchor's own depth zone, clearly closer to the camera than the anchor, or clearly farther from the camera. Never say an object is near, beside, or next to the anchor when it merely appears close in the 2D frame while lying at a clearly different depth (e.g. a foreground jogger inside a crop built around a distant lighthouse is 'much closer to the camera than the lighthouse', never 'near the lighthouse').

\- The anchor phrase above is only localization metadata. Do not copy or assume any attribute, text, identity, or full-scene relationship from that phrase unless it is independently visible in this image.

<<<INSERT: SPATIAL\_FRAME\_RULES>>><<<INSERT: CROP\_FRAME\_CAVEAT>>>

PERSON ORIENTATION AND HANDS (mandatory):

\- Before describing any person's pose, clothing sides, held objects, or body-part sides, state which way that person faces: toward the camera, away from the camera, toward the image-left, toward the image-right, or 'facing unclear' when it cannot be determined.

\- POSE BEFORE SIDE: when two limbs of one person must be told apart, prefer a pose or state descriptor that needs no laterality at all — the raised arm versus the arm hanging at the side, the hand gripping the strap, the gloved hand, the arm resting on the table. Such wording is unambiguous for any viewer and stays true regardless of facing.

\- Name an anatomical side only when a pose descriptor cannot separate the limbs AND the person's facing is unmistakable at this resolution (their face, or their back, is plainly readable — not inferred from a few pixels of hair or shoulder). Then derive the side from that stated facing per the frame contract and include the viewer-side derivation in parentheses so it is checkable, for example: 'He faces away from the camera; his right hand (on the viewer's right) holds a silver can.'

\- If the facing is not unmistakable, never write 'left hand' or 'right hand': write the pose descriptor, or 'one hand (laterality unclear)' plus a viewer-frame location. A small or distant person is exactly the case where a confident side claim is unsafe.

\- Whenever you describe what one hand holds, account for the other hand in the same pass: either describe what it visibly holds (or that it is visibly empty), or record it in the VISIBILITY AND REFERENCE NOTES section as out of view with unknown content.

INSTANCE LOCKING FOR REPEATED OBJECTS:

\- When the crop shows, or plausibly extends into, more than one instance of a category — price tags, labels, signs, screens, windows, wheels, hands, shoes, bags, and similar repeated items — never attach information to the bare category noun. Bind every described instance to a stable qualifier that would remain valid in the full image: an attachment or nearest-object relation ('the mug nearest the kettle'), or a part location on its host ('on the tail of the aircraft').

\- A position in this frame is NOT such a qualifier and never counts as one. This crop is a window cut out of a larger photograph and its borders are not the photograph's borders: whatever is leftmost, rightmost, or 'on the left side' HERE can sit dead centre, or on the opposite side, of the full image. Never write a frame superlative — leftmost, rightmost, topmost, bottommost, far left, far right — about anything in any wording: you cannot see what lies outside this crop, so you cannot know that anything in it is the outermost one.

\- When a rough position genuinely helps a reader picture the scene, word it as local to this crop ('in the left part of this crop'), never as a fact about the image. An instance with no attachment or part qualifier is described without one and recorded in the VISIBILITY AND REFERENCE NOTES section, not pinned by position.

Include useful details such as objects, readable text, colors, attributes, actions, and spatial relationships. There is no required schema, bullet format, or fixed wording; organize the description naturally.

Base every statement only on visual evidence in this crop:

\- HONEST EDGE REPORTING: the four outer crop edges are artificial viewport boundaries, not photographed scene content. When an object is cut off by an outer edge (the anchor included), simply say what you see: describe its clearly visible parts and their attributes, and never state whole-object attributes, whole-body actions, totals, or overall shape/size for it, and never guess what continues beyond the edge. Keep the main description positive — what IS visible — and put every cut-off / partially-visible / out-of-view remark in the VISIBILITY AND REFERENCE NOTES section at the end instead. A truncated object you cannot even identify belongs only in that section.

\- For text, use a deliberately conservative, character-by-character standard. A character is clearly legible only when its complete glyph shape and boundaries are visible and there is no reasonable alternative reading; prefer a shorter certain transcription over a longer plausible one, and never use language, brand, or scene context to complete a word. If the entire inscription meets this standard, transcribe it exactly. Otherwise never guess the uncertain characters: when a contiguous substring of at least three characters is clearly legible, report only that substring, say the rest is unclear, and state where it sits (its character count and whether it is the beginning, end, or middle). Call it a beginning or ending only when that text boundary is itself visible in the crop, and say whether the character touching the uncertain side has a complete isolated glyph — under cropping, occlusion, blur, or small scale such boundary characters are unreliable. If the sign, panel, or surface carrying an inscription is itself cut off by the crop edge at or near the text region, treat the inscription as partial with the cut side uncertain even when every visible glyph is complete — the crop is not a god's-eye view and the text may continue on the hidden part. Never attribute text to an object unless it is visibly printed on that object. TEXT COLOR CAUTION: outlined or two-tone glyphs are easily misread at this scale — dark characters with a light outline routinely look like white text on a dark panel. State a text color only when the stroke FILL color is unambiguous; when uncertain, write 'light-colored text on a dark panel' or 'dark text on a light panel', or omit the color entirely. A guessed text color must never appear.

FIGURE AND GROUND: a panel's color is never the glyphs' color. Name the color of the strokes and the color of the surface they sit on as two separate facts ('red characters on a yellow panel'), and never let one stand in for the other. Watch multi-tone signs especially: a fascia whose left panel is green and whose right panel is white has TWO panel colors and neither of them is a text color, so writing 'white letters' for black letters on its white panel is the same error as guessing. Illuminated night signage is where this goes wrong most.

\- COUNTING-SCOPE POLICY: do not state an exact number merely because N instances happen to be visible in this crop. An exact count is allowed only in one of these two cases:

(A) ANCHOR-OWNED SET: the counted items are parts of, attached to, worn by, held by, or contained in the selected anchor, and the complete anchor-side set relevant to the count is visible. (B) FULLY CLOSED ANCHOR-RELATIVE REGION: the counted items lie in a finite interval or region defined by the selected anchor and another specific visible object, and every boundary needed to delimit that region is fully visible in the crop. The crop edges must never serve as those boundaries. Phrases such as near, around, behind, in front of, to the left/right of, in the background, in the water, on the beach, or on the road merely locate visible instances; they do not close a counting set. Sea, water, beach, road, field, sky, crowd, and other surfaces or scene regions that continue beyond the crop are open sets, even when every instance currently visible in the crop is sharp and separable. For an open set, use non-numeric wording such as 'several are visible' when useful, and state that the exact scene total cannot be determined from the crop. Never estimate or give a range. - Do not infer hidden parts or use world knowledge to fill gaps. Describe uncertain details as unclear.

End the caption with a final section exactly titled 'VISIBILITY AND REFERENCE NOTES:'. It is a short bullet inventory whose only purpose is to warn later stages which references would be ambiguous or open; it is not a description and can never be quoted as visual evidence:

\- List each same-category counterpart whose content is unknown because it is cut off by a crop edge, out of view, or occluded: the other hand of a person when you described one hand, additional price tags/labels/signs beyond the ones you fully described, the unseen remainder of a vehicle or large object when you described one of its parts.

\- Each note records only: the entity class and its owner or host ('the man's other hand', 'additional price tags', 'the rest of the blue trailer'), its status (cut off at which edge, out of view, or occluded), and 'content unknown' when applicable.

\- Notes must NEVER contain attributes, colors, counts, actions, or transcribed text, and must never guess what the hidden content might be.

\- If nothing qualifies, end with exactly 'VISIBILITY AND REFERENCE NOTES: none.

## CAPTION2\_ANCHOR\_MARKED.

A thin red selection rectangle has been added to this crop because the crop contains multiple objects of the same category. It marks the one specific instance named as the selected anchor object in the instructions below. Use the rectangle silently to distinguish that photographed object, its parts, contents, markings, and related nearby objects from those belonging to other same-category objects. Then write the caption as if the artificial rectangle were absent. Never mention, describe, count, transcribe, or treat the rectangle, its color, shape, position, corners, or boundary as photographed content. Refer to the selected photographed object naturally, for example by its object category and visible scene position.

<<<INSERT: CAPTION2>>>

## J.4 Incremental evidence and bilingual QA generation INCREMENT.

Two descriptions of the SAME image region are given. caption1 was written while seeing the full image (the region was highlighted with a box). caption2 was written while seeing a high-resolution crop of exactly that region. Both captions were given the same selected anchor phrase and were asked to organize around that anchor, so 'the anchor' in either caption denotes the same photographed object; use that shared reference to align object identities between them.

Selected anchor object:

"{anchor}"

The phrases 'selected object', 'selected anchor', 'marked object', or 'outlined object' in caption2 refer to this exact photographed anchor, not to a newly appearing object. Use this metadata only to align object identity between the two captions. Do not merge a different same-category object with the selected anchor. If an object's identity cannot be aligned reliably, omit that candidate fact. Any selection rectangle, outline, box, marker, label, or annotation mentioned by caption2 is pipeline metadata, not photographed content, and must never become an incremental fact.

Extract the incremental facts: discrete, visually verifiable information that caption2 provides and caption1 either omits (absent), marks as unclear (unclear), or states differently (contradicted). Contradictions ARE increments: report them with relation "contradicted" and record caption1's claim.

MANDATORY CROP-VIEWPORT INVARIANCE CHECK (higher priority than treating contradictions as increments): every retained fact must describe the photographed scene or an object's intrinsic property, and must remain semantically true if the crop were expanded. The crop is not a privileged or god's-eye view of the scene; it merely hides content beyond its four artificial edges. A difference in how much of an object falls inside the crop is NEVER an absent, unclear, or contradicted scene fact. Reject any candidate based on an object being at a crop edge, cut off, cropped, only partially visible, or reduced to particular visible body parts by the crop. For example, if caption1 describes a person walking while caption2 says only the person's head and shoulders are visible at the bottom edge, these claims do NOT contradict each other: caption2 is reporting viewport truncation, so omit the candidate entirely. Apply this rule even if caption2 states the boundary observation confidently and even if it can be assigned one of the allowed fact types. Silently audit every proposed fact for crop-edge dependence before adding it to the JSON.

VISIBILITY AND REFERENCE NOTES handling: caption2 may end with a section titled 'VISIBILITY AND REFERENCE NOTES'. That section is disambiguation context ONLY. Never extract a fact from it, never quote it in whole or in part as source\_from\_caption2, and never let it add, upgrade, or contradict main-description content. Use it solely to stay aware that other same-category instances or hidden parts exist with unknown content.

REFERENT QUALIFIER PRESERVATION: when a caption2 evidence sentence pins an instance with a qualifier — an anatomical side ('in his right hand'), a stated facing direction, an attachment or nearest-object relation ('the mug nearest the kettle'), a part location on a host ('on the tail of the aircraft'), or a depth-zone tag ('much closer to the camera than the anchor') — the fact's statement MUST keep that qualifier. Dropping the qualifier and attributing the information to the bare category ('the man holds a can', 'the price tag reads 140') is forbidden, because other instances may exist beyond the crop. Never flip an anatomical side into a viewer side or vice versa when restating evidence; parenthetical viewer-side derivations from caption2 may be omitted from the statement, but the anatomical side itself must be preserved exactly.

SPATIAL-FRAME CONFLICTS ARE NOT INCREMENTS: both captions follow the same frame contract (anatomical body sides derived from stated facing; viewer frame between objects; no depth 'behind'). If the two captions disagree ONLY about which body side of the same person carries an item or pose (left vs right hand/arm/leg/shoulder), and caption2 states the person's facing with a derivation while caption1 does not, treat caption2's side as a corrected relabeling of the same observation: do NOT emit the side flip itself as a contradicted fact, and never create a fact whose main content is which body side something is on. Silently use caption2's side inside any other fact about that item.

CAPTION1 OMISSION GUARD (applies to relation 'absent' only): caption1 is written at full-image scale but may still simply forget an obvious attribute. When caption1 clearly describes an object as large or prominent within the region — for example a person occupying much of it — and the candidate fact states a coarse attribute any full-image viewer would plainly see (dominant clothing color or garment type, overall object color, coarse pose, obvious ongoing action), treat the missing attribute as a caption1 omission, NOT an increment, and omit the fact. Keep 'absent' facts only for genuinely fine-grained information: small text, small parts or accessories, precise counts, subtle patterns, small held items, or details whose visibility plausibly requires zooming in. This guard never applies to facts whose relation is 'unclear' (caption1 explicitly could not see it) or 'contradicted' (caption1 made a different claim).

Only keep facts of these types: count, OCR text, color, action, local attribute, spatial relation. Discard:

\- subjective wording and anything hedged with "possibly/maybe";

\- texture-only remarks (foliage, grass, sky, water, lighting);

- facts about the region boundary itself (what is inside/outside the box, objects cut by the crop edge);   
- range counts ("15-20") and estimates;

\- hue-level color differences (red vs orange, cyan vs blue) — keep a color contradiction only if the two claims are clearly different color classes.

COUNT FACT ELIGIBILITY (mandatory and higher priority than contradiction):

\- Retain an exact count only when it belongs to exactly one of two scope types: (A) anchor\_owned — items that are parts of, attached to, worn by, held by, or contained in the selected anchor; or (B) closed\_anchor\_relative\_region — items inside a finite region explicitly bounded by the selected anchor and another specific object, with all boundaries fully visible in caption2.

\- The artificial crop boundary never closes a set. Relative words such as near, around, behind, in front of, left/right of, in the background, in water, or on a beach/road/field are locators, not

complete boundaries. A visible subset of people or objects on a sea, beach, road, field, sky, crowd, or other region that may continue beyond the crop is not an exact scene count. Reject it even if caption2 gives a confident number and even if caption1 gives a different number.

\- For every retained count fact, add count\_scope\_type (anchor\_owned or closed\_anchor\_relative\_region), counted\_entity, scope\_boundary\_description, and scope\_fully\_visible=true. If you cannot truthfully fill all four fields from caption2, omit the count fact. For non-count facts, use null for these fields.

OCR evidence rules:

\- TYPE DISCIPLINE: any fact whose statement reports readable characters, letters, digits, words, logo text, or an inscription MUST use type 'OCR text' with all audit fields filled. Never type a text-reading fact as attribute, local attribute, or color — doing so would smuggle it past every OCR boundary rule. - Preserve caption2's exact certainty boundary. If caption2 says only a substring is legible, label the fact as partial OCR text and preserve the exact substring, its character count, and its stated position. - A partial OCR fact is eligible only when it contains at least three contiguous clearly legible characters and caption2 identifies a usable position for that substring.

\- Never fill in, repair, normalize, or guess unclear characters. Never turn a partial substring into a claim about the full inscription, word, brand, or name.

\- If caption2 transcribes a sequence but also says the inscription extends into a cropped or obscured side, treat it as partial OCR even when the visible sequence itself is described as clear. Preserve which side is uncertain and whether the boundary-adjacent glyph is fully isolated.

\- For every OCR fact, add these audit fields: ocr\_scope (full or partial), visible\_sequence (the exact transcribed characters), uncertain\_side (left, right, top, bottom, both, internal, none, or unknown), and boundary\_character\_isolated (a JSON boolean true or false, never a quoted string). If any portion of the same inscription, sign, label, or printed text is described as cropped, obscured, incomplete, or unclear, ocr\_scope MUST be partial. Set boundary\_character\_isolated to true only when caption2 explicitly confirms that the glyph touching the uncertain side is complete and visibly separated on both sides; otherwise use false.

\- For every OCR fact, also express the evidence relative to NORMAL READING ORDER rather than only to image geometry. Set substring\_position to full when the complete inscription is visible, prefix when the visible sequence is the beginning of a longer inscription, suffix when it is the ending, or unknown when caption2 does not support that decision. Set unsafe\_boundary to beginning or end when the corresponding boundary character of visible\_sequence touches an uncertain region; use both, internal, none, or unknown when appropriate. Do not trim visible\_sequence yourself. Physical top/bottom/left/right and sequence beginning/end are different concepts.

```jsonl
Output strict JSON only:
{"incremental_facts": [{"type": "...", "statement": "...", "source_from_caption2": "...", "ocr_scope":
"full|partial|null", "visible_sequence": "...|null", "uncertain_side":
"left|right|top|bottom|both|internal|none|unknown|null", "substring_position":
"full|prefix|suffix|unknown|null", "unsafe_boundary": "beginning|end|both|internal|none|unknown|null",
"boundary_character_isolated": false, "count_scope_type":
"anchor_owned|closed_anchor_relative_region|null", "counted_entity": "...|null",
"scope_boundary_description": "...|null", "scope_fully_visible": true, "relation_to_caption1":
"absent|unclear|contradicted", "caption1_claim": null}], "skip_reason": null}
If there is no reliable increment — in particular if every candidate fact concerns only vegetation,
sky, water, lighting, or generic texture — return an empty list with a short skip_reason.
caption1:
{caption1}
caption2:
{caption2}
```

## QA\_GENERATION.

You are generating fine-grained visual QA from incremental facts.

The anchor object (salient and uniquely locatable in the full image) is: "{anchor}"

Crop-local anchor rule:   
{local\_anchor\_rule}

Incremental facts (information visible only when zooming into the region around the anchor; each has an   
evidence sentence from the zoomed-in caption):   
{facts}   
Full zoomed-view description of the region (use it to check for duplicate objects of the same category   
— NOT as a source of new facts):   
{caption2}

Current generation target plus feedback from prior QA candidates for this SAME anchor: {retry\_context}

Generate up to {qa\_per\_anchor} questions.

Every incremental fact has a source\_fact\_index. Return that exact integer for the one fact used by each QA. Copy evidence\_sentence directly from that fact's source\_from\_caption2 when available; otherwise use its statement.

\- When generating more than one QA in the same response, every QA must use a different source\_fact\_index and evidence sentence; never make two questions from the same fact or paraphrase the same question. - Any question that asks the solver to read or transcribe text, letters, characters, digits, words, logo lettering, or a specified character span MUST use question\_type=ocr. Never label such a question as attribute, comparison, or another type.

- Read the ENTIRE OCR fact, especially statement and source\_from\_caption2; do not treat   
visible\_sequence as the only useful information. Visual details explicitly tied to the text—such as its background color, text color, position, orientation, containing panel, or relation to another label—are valuable referring cues. Prefer a question that uses those cues to identify one text region over a generic question that asks for every word on a whole sign or logo. Use a visual attribute as a locator, not as another requested answer.   
- A HEDGED IDENTITY IS NOT A REFERENT. When caption2 hedges what something is — 'what looks like a bi<sub>r</sub>d'<sub>,</sub> '<sub>poss</sub>ibl<sub>y</sub> <sub>a</sub> h<sub>e</sub>l<sub>met</sub>'<sub>,</sub> '某种容器' <sub>—</sub> i<sub>t</sub> i<sub>s</sub> <sub>te</sub>lli<sub>ng</sub> <sub>you</sub> i<sub>t</sub> <sub>cou</sub>ld <sub>not</sub> id<sub>ent</sub>if<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>o</sub>bj<sub>ect,</sub> <sub>an</sub>d <sub>a</sub> referring expression built on that guess matches nothing when the guess is wrong: a solver hunting for a bird will never find the plastic bag it actually is. Drop the hedged noun and locate the thing by what IS certain — where it sits, what it is inside or attached to, its color and size ('the small white object on the ledge above the second window'). If nothing certain identifies it, skip the fact. Never carry the hedge into the question either: 'the shoe or boot' tells a solver you are guessing. - LOCATOR ATTRIBUTE RELIABILITY: prefer host-level cues — panel/sign color, position, containing object, nearest landmark — over the text's OWN color. Small text color is frequently misperceived (dark outlined glyphs read as white), and a wrong locator attribute misleads solvers or matches a different text region. Include the text's color only when the fact states it without hedging AND it adds discrimination that the host cue does not already provide; when the panel alone is unique, leave the text color out. Never take a color from one region of a multi-tone sign and attach it to text sitting on another region; if you cannot say the stroke fill without hedging, use the host cue alone. - ONE HOST PER OCR ANSWER. The answer is text carried by a SINGLE bounded host — one sign, label, plate, panel, or printed line. Never assemble an answer out of characters sitting on separate objects, and never scope an OCR question by nearness to a GROUP ('the digits printed near the green bottles', 'the letters around the potted plants'): proximity to a set has no boundary a solver can reproduce, so which characters belong in the answer becomes arbitrary and a second reader will include a different set. If the text you want spans separate hosts, ask about one of them instead.   
- OCR facts may contain deterministic fields computed by code: safe\_sequence, safe\_character\_count, required\_span, and qa\_eligible. Never correct, repair, or invent characters beyond safe\_sequence. The answer must be one clearly supported CONTIGUOUS segment of safe\_sequence.   
- For a partial OCR fact (required\_span=first or last), write a PLAIN text-reading question about that inscription, located by host-level cues, with the answer equal to safe\_sequence verbatim. Do not add any span wording (no 'first/last k characters'), do not add totality words such as 'entire',   
'complete', or 'all the text', and never mention cropping or hidden characters. A dedicated   
edge-rewriting stage that sees the full photograph will rewrite such drafts into the final safe span wording; your draft only needs to pick the right inscription and copy safe\_sequence.   
- LOCALIZED FULL-OCR SEGMENT: when required\_span=full and the fact itself explicitly associates different complete words or consecutive word groups with independently identifiable visual regions, you may ask for one such region and use only its complete text as the answer. The chosen answer must be one or more WHOLE consecutive words appearing verbatim in safe\_sequence; never take characters from inside a word. Both questions must include the fact-supported visual locator that isolates that segment. Do not invent a color, position, container, or relation. If the fact does not support a unique localized segment, use the complete safe\_sequence instead.   
- Example: if a fact says that OPEN appears in white on an upper green panel and 24 HOURS appears on a separate lower black strip, prefer asking what text appears on the upper green panel (answer: OPEN), rather than what all text on the sign says. This example describes the policy only; never copy its objects, colors, positions, or text into another image's QA.   
- For required\_span=full, either ask for the complete clearly legible text and use safe\_sequence verbatim, or apply LOCALIZED FULL-OCR SEGMENT exactly as defined above.   
- Never write view-dependent instructions such as 'excluding the top character', 'ignore the cropped character', 'remove the first visible letter', or any reference to an image/crop boundary, cropping, truncation, or hidden characters. A fixed first/last k span must have the same answer in the full image and zoomed image.   
- Skip every OCR fact with qa\_eligible=false or without all required deterministic fields.   
- If the context contains prior-candidate entries in addition to the generation\_target\_state, choose a DIFFERENT incremental fact and answer property. Do not repeat or merely paraphrase any rejected or retained global/crop question, evidence sentence, or source fact.   
- Use rejection feedback only to choose a better-supported candidate. NEVER alter the reference answer to imitate verifier outputs, and never invent a fact absent from the incremental-facts list.   
- If a previous candidate failed crop verification, prefer a different fact whose answer is   
larger/clearer and directly visible in the crop. - The generation\_target\_state entry in the context is authoritative. If   
zoom\_required\_still\_needed=true and a previous accepted candidate was answerable from the full image, prefer a substantially smaller or finer detail marked absent/unclear/contradicted. If it is false, a retained full-image-answerable QA counts toward the accepted-question minimum; the next distinct accepted QA may be either full-image-answerable or zoom-required.

\- If retry feedback says a previous candidate remained ambiguous AFTER the dedicated image-aware referent repair failed, use a different uniquely referable target. Never add answer-leaking qualifiers. - DISAMBIGUATION: before writing each question, scan the full description above. If the answer object's category could match more than one object in the crop (e.g. a ring worn by a person vs a standalone ring product image), the crop-local referring phrase MUST include a qualifier that separates them (position, size, displayed-vs-worn, color). If no crop-visible qualifier can make it unique, skip that fact. - Write TWO semantically paired versions of every question:

1. question\_zh / question\_en are GLOBAL questions for the original full image. They MUST make the answer object locatable in the full image by a solver who has never seen the crop. Choose the locator by the target's REAL relation to the anchor (see GLOBAL LOCATOR POLICY below).

2. crop\_question\_zh / crop\_question\_en are LOCAL questions for the exact zoomed crop. Remove context that the crop itself cannot establish, such as "in the center of the plaza", "on the building in the background", or "to the right of the white building". Keep only qualifiers whose evidence is visible inside the crop and that are needed to identify the answer object there.

\- Follow the Crop-local anchor rule exactly. A selection rectangle may be used only as a referring-expression locator for the photographed anchor, its parts, contents, markings, or related nearby objects. Never ask about the rectangle itself, its color, shape, boundary, corners, or any artificial label. The GLOBAL questions must never mention a selection rectangle.

\- The GLOBAL and LOCAL versions MUST ask the same property of the same object and MUST have the same answer. The local version is not a new question.

\- The LOCAL question must be self-contained and its answer object must be uniquely identifiable inside the crop. If removing unavailable global context makes that impossible, skip the fact.

GLOBAL LOCATOR POLICY (the crop is a 2D window; objects inside it can lie at completely different scene depths, so 2D adjacency inside the crop is NOT scene proximity):

(a) If the answer object has a REAL, visually supported scene relation to the anchor — it is a part of it, attached to it, on the same supporting surface, held by it, or genuinely adjacent at the same depth — locate it through the anchor phrase as usual.

(b) If the answer object is clearly at a different depth than the anchor (the zoomed description says it is much closer to or farther from the camera, e.g. a foreground jogger inside a crop built around <sub>a</sub> b<sub>ac</sub>k<sub>groun</sub>d li<sub>g</sub>h<sub>t</sub>h<sub>ouse</sub>)<sub>, NEVER c</sub>l<sub>a</sub>i<sub>m prox</sub>i<sub>m</sub>i<sub>ty w</sub>i<sub>t</sub>h <sub>wor</sub>d<sub>s suc</sub>h <sub>as near,</sub> b<sub>es</sub>id<sub>e, next to,</sub> 附近<sub>,</sub> 旁边<sub>,</sub> and never use front/behind as a disguised depth word. That fabricates a relation that does not exist, misleads the solver into searching around the anchor, and reads unnaturally. PREFER KEEPING THE ANCHOR in the question through an HONEST relation — it is the verified, easiest-to-find landmark — choosing whichever of these reads naturally:

\- CAMERA-AXIS relation (works for ANY anchor, including faceless structures; preferred when the target lies on or near the anchor's line of sight, which crop co-membership usually guarantees): state the depth order along the viewing axis with the canonical wording — target closer to the <sub>camera:</sub> '<sub>t</sub>h<sub>e ye</sub>ll<sub>ow tax</sub>i b<sub>etween t</sub>h<sub>e c</sub>l<sub>oc</sub>k <sub>tower an</sub>d <sub>t</sub>h<sub>e camera</sub>' / 在钟楼与镜头之间的黄色出租车<sub>;</sub> <sub>target</sub> f<sub>art</sub>h<sub>er:</sub> '<sub>t</sub>h<sub>e</sub> b<sub>r</sub>i<sub>c</sub>k <sub>ware</sub>h<sub>ouse</sub> b<sub>eyon</sub>d <sub>t</sub>h<sub>e re</sub>d b<sub>arge,</sub> f<sub>art</sub>h<sub>er</sub> f<sub>rom t</sub>h<sub>e camera</sub>' / 越过红色驳 船更远处的砖砌仓库. When the two do not overlap in the image, add a small image-frame offset word ('below the clock tower and between it and the camera'). This wording is honest 3D geometry, never a proximity claim.

\- INTRINSIC ANCHOR-FRAME relation: when the anchor is a person, animal, or vehicle whose facing caption2 states, the target may be located in the anchor's own frame with each side derived mechanically per the contract. Example: a cyclist riding toward the viewer has the deeper scene BEHIND him and his left-hand side on the viewer's right, so 'the bright blue vertical sign behind the cyclist on his left-hand side' is honest and natural. SIDE-BLINDNESS GUARDRAIL: verification models cannot reliably resolve a person's left/right sides, so any anchor-frame directional clause is narrative scaffolding only — the question MUST still identify the target uniquely when that clause is ignored, through at least one anchor-independent discriminator (object type plus color, nearest fixed landmark, or viewer-frame position). If uniqueness would depend on the anchor-frame clause itself, do not use this option.

\- EXPLICIT IMAGE-FRAME relation to the anchor: honest 2D wording such as 'in the image, above the cyclist in the background' or 'below the lighthouse and much closer to the camera', never wording that implies physical proximity.

\- SELF-LOCATING referent (fallback when no anchor-linked wording reads naturally, e.g. the linkage would feel contrived across a large cluttered depth gap): drop the anchor and locate the target with full-image-frame position plus its own visible attributes, e.g. 'the jogger in the lower-center foreground of the image wearing a cap'. Depth words like foreground/closer to the camera stay true across views; horizontal words must be re-derived from the FULL image, never copied from the crop question.

(c) If none of these produces a referent you are confident is unique in the FULL image, skip the fact. Prefer facts about the anchor itself or objects in its own depth zone: a crop-foreground object much closer to the camera is usually large and legible in the full image, so its facts rarely survive the full-image difficulty gate; spend them only when nothing anchored remains.

\- Example pair:

GLOBAL: "What text is written on the sign in the center of the plaza?"

LOCAL: "What text is written on the sign?"

\- NEVER put the answer attribute itself into either referring expression (if the answer is a color, the referring phrase must not mention that color).

\- The answer must be a single definite value: an exact count, one color word, a short text string, or a short action. Skip facts with ranges, hedges, or subjective wording.

\- The answer itself must never be an anatomical body side ('left hand', 'right leg') NOR a part

location on an oriented object ('near the front left corner', 'at the rear end', 车头/车尾): all vision models share systematic mirror and orientation confusion, so such answers cannot be verified reliably. Sides and part locations serve only as referring qualifiers inside questions.

\- COUNTING SOURCE GATE: before using a count fact, classify its real-scene scope. It is eligible only as (A) anchor\_owned: the counted items are parts of, attached to, worn by, held by, or contained in the selected anchor, with the complete relevant anchor-side set visible; or (B)

closed\_anchor\_relative\_region: the items occupy a finite interval or region explicitly bounded by the anchor and another specific visible object, and all boundaries are fully visible. If neither classification is supported, skip the fact, even when it contains an exact number and is contradicted by caption1.

\- Crop edges never close a counting scope. Phrases such as near, around, behind, in front of, left/right of, in the background, in water, or on a beach/road/field identify a visible subset but do not define a complete set. Never generate a question asking for the total of such an open scene region. In particular, counting the instances currently visible in a cropped slice of sea, beach, road, crowd, field, or sky is forbidden.

\- For an eligible anchor\_owned count, the question must explicitly bind the counted items to the anchor. For an eligible closed\_anchor\_relative\_region count, the question must name both bounding objects and the finite interval/region between them. A generic phrase such as 'behind the anchor' is insufficient. - Prefer count facts carrying count\_scope\_type and scope\_fully\_visible=true. Independently audit the evidence and full crop description; metadata cannot rescue an actually open scope.

\- Skip facts about the region boundary and vegetation/sky texture.

\- Skip color facts that differ only at hue level between the two views.

## <<<INSERT: SPATIAL\_FRAME\_RULES>>><<<INSERT: CROP\_FRAME\_CAVEAT>>>

\- Apply the frame contract above to every direction word in questions and answers. Body-part sides stay anatomical exactly as the fact states them ('right hand' means his own right hand); never convert an anatomical side into an image side or vice versa. Keep parenthetical facing/viewer-side derivations OUT of the question text — questions carry only natural human phrasing.

\- BILINGUAL FRAME PARITY: the Chinese and English versions must encode the same frame with the same di<sub>sc</sub>i<sub>p</sub>li<sub>ne. In C</sub>hi<sub>nese, scene</sub> d<sub>ept</sub>h <sub>must</sub> b<sub>e wr</sub>i<sub>tten as</sub> 背景中 / 前景中 / 离镜头更远 / 更靠近镜头 <sub>—</sub> NEVER as 后方、身后、前方 or 前面 attached to a person or animal (write 跑者背景中更远处的车辆, not 跑 者后方的车辆). Use 身后/面前 only for a person's intrinsic sides and only when the question itself states that person's facing (e.g. 背对镜头的人身后). 左手/右手 are anatomical only, exactly like left/right hand in English.

\- LIMB QUALIFIER PREFERENCE: to single out one of a person's paired limbs, prefer the pose or state the fact records — the raised arm, the arm hanging at the side, the hand holding the strap — over an anatomical side. Pose wording is natural, needs no facing derivation, and is the only kind a solver can check; verification models do not reliably resolve a person's left from their right. Use 'left/right hand' only when the fact states the side with an explicit facing derivation AND no pose descriptor separates the limbs.

\- REFERENT CLOSURE (mandatory): if the fact's answer object belongs to a multi-instance category — hands, feet, shoes, price tags, labels, signs, screens, wheels, windows, bags, or similar repeated parts and products — and the zoomed description or its VISIBILITY AND REFERENCE NOTES indicate that other instances exist anywhere in the scene, are cut off, or are out of view with unknown content, then BOTH the global and crop questions MUST pin the exact instance using the qualifier carried by the fact and its evidence: the anatomical side ('in his right hand'), the attachment or nearest relation ('on the price tag closest to the kettle'), or the part location ('on the rear end of the large blue trailer'). A bare 'in his hand', 'on the price tag', 'on the sign', or 'on the vehicle' is forbidden in that situation. If the fact and evidence carry no such qualifier while the notes show counterparts exist, SKIP that fact.

\- Never mention the crop viewport, cropping, truncation, cut-off or out-of-view states, visibility limitations, or the VISIBILITY AND REFERENCE NOTES section in any question; the notes only tell you when an extra pinning qualifier is required. Positional image-frame references such as 'near the left edge of the image' are locators, not viewport talk, and remain allowed under the frame contract. - FRAME CLAIMS COME FROM CAPTION1 ONLY. caption2 saw a crop, not the photograph, so any left/right/side wording in it describes the crop's own frame and is worthless — often actively wrong — as a statement about the image: a crop taken from the right of the photo has its own left side. Never copy or paraphrase a horizontal position out of caption2 into a question. Use one only when caption1, which saw the whole photograph, places the target there, and word it as caption1 does.

\- NO FRAME SUPERLATIVES. Never write leftmost, rightmost, topmost, bottommost, far left, far right, <sup>最</sup> 左<sub>,</sub> 最右<sub>,</sub> 最上<sub>,</sub> 最下<sub>,</sub> 最顶<sub>,</sub> 最底 <sub>a</sub>b<sub>out</sub> <sub>t</sub>h<sub>e</sub> i<sub>mage</sub> i<sub>n</sub> <sub>e</sub>i<sub>t</sub>h<sub>er</sub> l<sub>anguage.</sub> <sub>Be</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>outermost</sub> <sub>o</sub>f <sub>a</sub> <sub>category</sub> is a claim about everything in the photograph, which neither caption can establish; use an attachment, nearest-object, or part qualifier instead. A superlative INSIDE one named host stays fine ('the bottom word on the blue sign'), because the host bounds it.

\- After eligible OCR facts, prefer counting / attribute facts, and prefer facts whose relation\_to\_caption1 is "contradicted" or "unclear".

{"qas": [{"question\_zh": "...", "question\_en": "...", "crop\_question\_zh": "...", "crop\_question\_en": ...", "answer": "...", "question\_type": "counting|ocr|attribute|action|spatial|comparison",   
"evidence\_sentence": "verbatim evidence", "source\_fact\_index": 0}]}   
If no fact qualifies, output {"qas": []}.

## LOCAL\_ANCHOR\_RULE\_UNMARKED.

No artificial selection rectangle is present. Neither LOCAL question may mention a box, outline, marker, or annotation.

## LOCAL\_ANCHOR\_RULE\_MARKED.

A thin red selection rectangle identifies the anchor in the crop. Both LOCAL questions MUST explicitly use the rectangle only as a locator, with wording such as 'the [object] inside the red selection <sub>rectang</sub>l<sub>e</sub>' / '红框标出的[物体]'<sub>.</sub> <sub>I</sub>f <sub>t</sub>h<sub>e</sub> f<sub>act</sub> <sub>concerns</sub> <sub>a</sub> <sub>part,</sub> <sub>content,</sub> <sub>mar</sub>ki<sub>ng,</sub> <sub>or</sub> <sub>near</sub>b<sub>y</sub> <sub>o</sub>bj<sub>ect,</sub> relate it explicitly to that red-boxed anchor. Never ask about the rectangle itself. GLOBAL questions must describe the photographed anchor naturally and must not mention it.

## J.5 Joint crop/full-image audit and rewriting

## EDGE\_REWRITER.

You are the crop-boundary auditor and question rewriter of a visual-QA data pipeline.

BACKGROUND — WHY THIS STAGE EXISTS:

The pipeline writes fine-grained QA about one anchor object. Every earlier stage saw ONLY a crop taken around that anchor: a describer captioned the crop, and a question writer turned caption facts into a crop-local question plus a full-image question. Neither could see past the crop's four edges, so two recurring failures escape them:

1. INCOMPLETE ANSWER CLAIM: the object or inscription carrying the answer continues beyond the crop boundary, and the recorded answer silently claims to be the complete text or a whole-object attribute when it is not (a sign whose top line sits above the crop; a hull that runs past the edge with more markings).

2. REFERENT AMBIGUOUS IN THE FULL IMAGE: the referring phrase picks exactly one thing inside the crop but matches several things in the full photograph (three white signs on the same wall; a second red inscription on the same hull).

You are the first stage that sees BOTH views, so these become direct observations for you. Downstream, independent solver models will answer the crop-local question seeing only the crop, and the full-image question seeing only the full photograph. Your goal: an honest, careful solver in EITHER view must arrive at exactly the recorded answer. Rejecting a question throws away everything already spent creating it, so PREFER RESCUING a question with a bounded rewrite over dropping it — but never rescue one at the cost of truthfulness.

## INPUTS:

Image 1 is the crop. A cyan box labeled C1, if present, marks the already-certified unique referent of the crop-local question; a thin red rectangle, if present, marks the anchor. Both are annotations, never scene content.

Image 2 is the FULL photograph with a red rectangle marking where the crop was taken. Use it to see what continues beyond the crop and what else in the scene matches the question's wording. Image 2 may be lower-resolution: use it to detect continuation and count referent matches, not to read new text.

THE QA UNDER AUDIT:   
anchor: "{anchor}"   
question\_type: {question\_type}   
global question (full-image solvers) EN: {question\_en}   
global question ZH: {question\_zh}   
crop-local question EN: {crop\_question\_en}   
crop-local question ZH: {crop\_question\_zh}   
recorded answer: {answer}   
mechanical OCR contract (null for non-OCR): {ocr\_constraint}   
caption evidence sentence: {evidence}

## AUDIT PROCEDURE:

A. CONTINUATION: in Image 2, does the object or inscription that carries the answer visibly continue beyond the crop rectangle in a way that could add to or change the answer as the question words it? For an OCR question, also check inside the crop whether the inscription's own host is cut or occluded near the text so the text may continue out of sight. A cut-off host is harmless when the question only asks about a bounded part that is fully visible.

THE BURDEN IS ON 'false'. Reaching the crop border IS continuation unless you can point to the reason it is not. Answer false only when you can name what bounds the host on that side — its own frame, edge, corner or end, visible INSIDE one of the two views — or when Image 2 shows the whole host with nothing further. If the host runs to the border and you cannot see where it stops, that is true, not false, however legible the part you can read: a street plate whose left edge is the crop's left edge gets read as 'STATION ROAD' when the plate says 'OLD STATION ROAD', and a menu board whose bottom is the crop's bottom shows two of its five lines. Both look complete. Being able to read what is visible is not evidence that nothing else is there.

Answering true does not condemn the QA — narrow\_semantic rescues it by re-scoping to a sub-block you CAN see whole ('the line nearest the arrow marker', 'the bottom word'). Prefer that rescue over a confirm you cannot justify.

B. UNIQUENESS: read each referring phrase of the GLOBAL question and count what it matches in Image 2; read the CROP question and count matches in Image 1. A locator difference that a careful solver would treat as decisive (unique attachment, position, or size wording) counts as disambiguation; a bare category plus color usually does not.

For an OCR question, ALSO count the text blocks: every separate inscription on the same host that a solver could take to match how the question describes the text (its color, case, or style) is a competing candidate. Judge like a strict solver, not a brand reader: a smaller second line counts; a darker shade of the same hue counts; digits and mixed letter-digit strings count as capital-glyph text; a registration or phone number printed under a brand name is the classic competing block. If more than one block could match, an honest solver cannot know which to read: the wording must single one out by row, size, or position (rewrite), or the QA must be dropped. Never assume the most prominent text is 'the' text.

WEAK TEXT REFERENT RULE: a text referent described only by color and/or letter case ('the red capital letters') with no size, row, or position qualifier is weak wording. Whenever the host carries ANY other marking — even one you judge too small or too faint to read — prefer a minimal tightening rewrite (add 'the largest', a row, or a position word) over keep. The tightening costs nothing and makes the question robust to markings at the edge of legibility.

Enumerate this audit in the output field other\_inscriptions\_on\_host: one entry per OTHER inscription, print, number, or marking on the same host (empty list only when the host truly carries no other markings). could\_match\_question\_wording=true means a strict solver could take that block as matching the question's text description. Keeping the question while any entry is true is a contradiction —

then you must rewrite (single the target out by row, size, or position) or drop; code enforces this. C. COUNT SCOPE (counting questions): the counted set must be closed by visible boundaries; if Image 2 shows the set continuing outside the crop while the question implies a total, the claim is broken.

## VERDICTS:

\- keep: both audits clean; the recorded answer is exactly what an honest solver would give in both views.

\- rewrite: a bounded edit repairs the problem — tighten a referring phrase so it is unique in both views, or narrow an OCR claim to a safe span (see OCR SPAN DECISION). Rewrite all four question fields consistently.

\- drop: the recorded answer is factually wrong or unverifiable in either view, no crop-visible qualifier can make the referent unique, or a count's scope is opened by the crop edge. Also drop when a rescue would need information a crop-only solver cannot see.

\- COUNT RESCUE BAN: a counting question whose set the crop opens can NEVER be rescued by wording.

Adding 'visible' / '可见' or similar makes the crop total and the full-image total differ, so the two views would have different correct answers. Such a question is unfixable: drop it.

## REWRITE RULES:

\- The recorded answer may NEVER be extended, corrected, or replaced with content you read in Image 2 — you sit downstream of evidence extraction and may only NARROW claims. For a non-OCR question the answer must stay exactly unchanged; you edit wording only. For an OCR question the answer may only shrink to a contiguous part of the recorded sequence.

\- CROSS-VIEW LOCATOR ATTRIBUTES: check every attribute the question uses to FIND the target — color, size, material, shape — against both images, including attributes the draft already contained. A sliver visible in the crop routinely misdescribes the whole object in the full photograph: a white sign whose only visible strip sits against an orange panel reads as 'the orange sign' in the crop and matches nothing (or something else) in the full image. Replace a misleading attribute with one true in both views, or drop when none exists.

\- Every qualifier you add must be verifiable inside the CROP (crop solvers cannot see past it) and stay true in the full image. Prefer attachment, nearest-object, part-position, row/line position, or

<sub>re</sub>l<sub>at</sub>i<sub>ve-s</sub>i<sub>ze</sub> <sub>cues.</sub> <sub>Never</sub> <sub>use</sub> <sub>v</sub>i<sub>ewport</sub> <sub>wor</sub>d<sub>s</sub> (<sub>crop,</sub> <sub>e</sub>d<sub>ge,</sub> <sub>truncate</sub>d<sub>,</sub> <sub>cut</sub> <sub>o</sub>ff<sub>,</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> <sub>port</sub>i<sub>on,</sub> 画面边 缘<sub>,</sub> 被截断)<sub>, an</sub>d <sub>never copy answer c</sub>h<sub>aracters or answer wor</sub>d<sub>s</sub> i<sub>nto any quest</sub>i<sub>on</sub> fi<sub>e</sub>ld<sub>.</sub>

\- Keep the two languages synchronized and keep the global/crop pairing: the global question locates the target for a full-image solver (usually via the anchor); the crop question drops context the crop cannot establish.

\- Spatial wording in rewrites must obey this contract:

<<<INSERT: SPATIAL\_FRAME\_RULES>>><<<INSERT: CROP\_FRAME\_CAVEAT>>>

OCR SPAN DECISION (REQUIRED for every ocr question, in ocr\_span\_decision; null for non-OCR):

You MAY change an OCR question's answer — this is usually the BEST rescue, not a last resort — provided the new answer is a contiguous part of the recorded one. Upstream drafts are often clumsy (a 20-character positional span, or a sequence that silently spans a truncation); your job is to re-scope onto a piece that is easy to refer to and true in both views: the bottom word, the largest line, the top row, or the first few letters. Work through the decisions in this order and pick the FIRST that yields a claim true in both views:

\- confirm: the inscription the question asks about is completely visible with no plausible continuation, and the recorded answer is its complete text (or the question already carries consistent span wording). Valid with verdict keep or with a rewrite that only fixed the referring phrase. confirm is FORBIDDEN whenever you observed the inscription's host cut or continuing at or near the text region — in the crop or in Image 2. Observing continuation and confirming anyway is a contradiction; choose narrow\_semantic, demote, or drop instead.

\- narrow\_semantic (PREFERRED rescue): re-scope the question to one fully-visible, visually isolable sub-block — the bottom word, the topmost row, the largest line, the group on one panel — whose complete text is a contiguous whole-word part of the recorded sequence; the answer becomes exactly that part ('what is the bottom word on the sign' -> 'Bakery' where the recorded answer was 'Old Town Bakery'). A solver reads such a unit naturally, with no character counting. Identify the sub-block ONLY by visual properties (position, size, row, color of its panel), never by quoting its text.

THE SUB-BLOCK MUST BE CLOSED ON THE CUT SIDE. A sub-block is only 'fully visible' if every one of its own boundaries is inside the crop, and the boundary that matters is the one facing the cut. When the host is cut on its LEFT or RIGHT, a row or line runs straight into that cut and is not closed, so 'the lower line', 'the word at the top of the sign', 'the largest word on the second row' and any other row- or band-based selector is invalid there — pick a unit bounded away from the cut instead (a word whose far side is clear space, the block on one whole panel). When the host is cut at the TOP or BOTTOM, rows away from the cut are fine and a column is not. A size superlative ('the largest word') over a unit that continues out of view is a claim about text you cannot see. If no closed sub-block exists, use demote\_first / demote\_last or drop; do not narrow onto an open one.

\- demote\_first / demote\_last: for the cases narrow\_semantic cannot express — no whole word is fully safe, but a clean sub-word boundary fragment is (only 'SUR' of a longer word is clearly visible). Ask for exactly the first/last k alphanumeric characters of the recorded sequence, with 3 <= k <= 7 — solvers must COUNT the characters, so a larger k turns reading into error-prone counting and is mechanically rejected. Use demote\_first when the text's beginning boundary is visibly real and continuation risk is at the end; demote\_last in the mirror case. CROSS-VIEW SPAN AGREEMENT (mandatory): before choosing k, read the same characters in Image 2 — every one of the k characters you keep must be identical in the crop and in the full photograph. A recorded sequence stitched together across a truncation (two cut lines read as one string) diverges from the real text at the stitch, and any k that reaches the stitch makes the answer wrong for full-image solvers: choose the smallest stable k (>= 3) that stays strictly before the divergence, and if even 3 agreeing characters do not exist, demote is unavailable. Both English question fields must contain the explicit k right beside first/last (e.g. 'the first 3 characters'), and the Chinese fields must mirror it (e.g. “前3个字母数字字符”). The answer is that exact span with its original spacing and punctuation preserved.

\- SHORT ANSWERS: an answer of fewer than 3 alphanumeric characters cannot be narrowed any further, so it survives only as confirm with no continuation anywhere at that text. If the inscription continues out of view (or its host is cut at the text), a 1-2 character recorded answer is a fragment of something longer: drop it rather than re-scoping the question onto that fragment — code rejects such a rescue either way.

\- Drop an OCR question only AFTER checking that no narrowing exists: neither a whole-word sub-block (narrow\_semantic) nor a 3-7 character boundary span (demote) yields a claim true in both views. A recorded answer being wrong AS A WHOLE does not justify drop when a narrowed part of it is still exactly right.

Do not transcribe scene text in reasons beyond what the recorded answer already contains. Return strict JSON only:

{"continuation\_beyond\_crop": true|false, "full\_image\_referent\_matches": <integer count for the global question's referent in Image 2>, "crop\_referent\_matches": <integer count in Image 1>, "other\_inscriptions\_on\_host": [{"description": "where/what kind of marking, WITHOUT transcribing it", "could\_match\_question\_wording": true|false}] (for non-OCR questions use []), "verdict": "keep|rewrite|drop", "ocr\_span\_decision": "confirm|narrow\_semantic|demote\_first|demote\_last"|null, "rewrite": null|{"question\_zh": "...", "question\_en": "...", "crop\_question\_zh": "..." "crop\_question\_en": "...", "answer": "..."}, "drop\_reason": null|"concise reason", "reason": "concise image-specific audit reason"}

## J.6 Crop, full-image, and blind answering CROP\_VQA.

Look at this image and answer the question with a single short value (a number, a color, a text string, or a few words). No explanation.   
Question: {question}

## CROP\_VQA\_ANCHOR\_MARKED.

The thin red selection rectangle is an artificial locator identifying the photographed anchor selected by the pipeline. A crop-local question may refer to the object inside the red rectangle, its parts, contents, markings, or directly related nearby objects. Use the rectangle only to resolve that reference rather than unmarked same-category objects. Never treat the rectangle itself as photographed content or as the requested answer.   
<<<INSERT: CROP\_VQA>>>

## CROP\_OCR\_VQA.

Look at this image and answer the OCR question by transcribing only the requested text or character span. Preserve character order and do not infer characters that are not clearly visible. If the requested text is not clear, answer exactly: unknown. If the question asks for exactly N characters or letters from a specified position, return exactly N alphanumeric characters in normal reading order: mechanically take the final N for a last/rightmost/ending-N question or the first N for a   
first/leftmost/beginning-N question. Spaces and punctuation do not count toward N. Do not return the complete visible word or any extra character. Check that the response contains exactly N characters before answering. Return the answer only, with no explanation.   
Question: {question}

## CROP\_OCR\_VQA\_ANCHOR\_MARKED.

The thin red selection rectangle is an artificial locator identifying the photographed anchor selected by the pipeline. A crop-local question may refer to the object inside the red rectangle, its parts, contents, markings, or directly related nearby objects. Use the rectangle only to resolve that reference rather than unmarked same-category objects. Never transcribe or return the rectangle itself as OCR content.

<<<INSERT: CROP\_OCR\_VQA>>>

## FULL\_IMAGE\_VQA.

Look at this image and answer the question with a single short value (a number, a color, a text string, or a few words). If you cannot see the answer clearly, answer exactly: unknown. No explanation. Question: {question}

## FULL\_IMAGE\_OCR\_VQA.

Look at this image and answer the OCR question by transcribing only the requested text or character span. Preserve character order and return your single best transcription from the visible image, even when some characters are difficult to read. If the question asks for exactly N characters or letters from a specified position, return exactly N alphanumeric characters in normal reading order: mechanically take the final N for a last/rightmost/ending-N question or the first N for a   
first/leftmost/beginning-N question. Spaces and punctuation do not count toward N. Do not return the complete visible word or any extra character. Check that the response contains exactly N characters before answering. Return the answer only, with no explanation.   
Question: {question}

## BLIND\_ANSWER.

You cannot see the photograph. Answer the question using only its wording and your learned language and world knowledge. Even if the image-dependent answer is uncertain, you must make your single most likely guess. Return one concrete, short answer of the requested type. Never answer unknown, cannot determine, not enough information, or any other abstention. Do not provide alternatives or an explanation. Question: {question}

## ANSWER\_EQUIVALENCE.

You are a careful, strict, and deterministic evaluator of short visual-question answers.

Inputs:

\- Question: {question}

\- Reference answer: {reference}

\- Candidate answer: {candidate}

Decide whether the candidate correctly fills the exact answer slot requested by the question.

Internally follow this order:

1. Infer the requested answer type and every explicit constraint in the question.

2. Identify the single answer asserted by the candidate. If it gives incompatible alternatives or adds an extra incorrect answer, reject it.

3. Compare that asserted answer with the reference in meaning.

Accept harmless grammatical context, articles, synonyms, paraphrases, and a more specific description when it preserves the reference meaning. Do not require the same wording. Reject a merely related object or attribute, a contradiction, the wrong answer type, an incomplete answer, or extra incorrect factual content.

For ordinary VQA, judge semantic equivalence in the context of the question; extra location or object context is harmless when it does not change the requested answer.

Think through the comparison internally. Output exactly one tag and nothing else:

<verdict>yes</verdict>

or

<verdict>no</verdict>

## OCR\_ANSWER\_EQUIVALENCE.

You are judging whether a candidate OCR response correctly answers the exact question. Judge visual content, not answer-only formatting.

Question: {question}

Reference answer: {reference}

Candidate response: {candidate}

Apply every positional and length constraint in the question. In particular, if it asks for the first/leftmost/beginning or last/rightmost/ending N characters and the candidate gives a longer transcription, mechanically take the first or final N alphanumeric characters from that transcription before comparing it with the reference. An explanation is allowed when its transcription and stated answer are mutually consistent. Do not say yes merely because the reference text appears somewhere in an unrelated explanation. If the response gives conflicting transcriptions or answers, or the derived requested span differs by any extra, missing, reordered, substituted, or accented character, say no. Ignore only letter case and superficial spacing or punctuation. Reply with exactly one word: yes or no.

## J.7 Multi-source distractors and MCQ verification

## FULL\_DISTRACTOR\_CLUSTERING.

You are a conservative semantic auditor constructing multiple-choice distractor candidates from repeated full-image VQA responses. You do not see the image and must not decide which answer is visually true. Do all semantic normalization and clustering yourself; the caller only validates IDs and counts support mechanically.

The caller has already removed responses judged equivalent to the reference and responses that are literally just 'unknown'. Some remaining responses may still be verbose abstentions, conflicted answers, or judge errors. Audit them conservatively rather than assuming every input is a usable distractor.

First infer the exact answer slot requested by the question: for example count, OCR string, time, color, material, object identity, action, or spatial relation. Then audit every response independently.

## Response rules:

1. committed means the response ultimately selects exactly one concrete answer. Extract the final selected answer even if it follows reasoning or an explicit self-correction.

2. abstain means the response ultimately says unknown, unclear, unreadable, cannot determine, or otherwise declines to answer. Classify it as abstain even if it mentions speculative possibilities. 3. conflicted means it leaves multiple incompatible alternatives without selecting one. Do not arbitrarily retain one alternative.

4. For abstain or conflicted, extracted\_answer and cluster\_id must be null.

Clustering rules:

5. Cluster committed answers only when they are interchangeable answers to this exact question. Normalize spelling, morphology, number words, abbreviations, and harmless surrounding wording semantically.

```jsonl
Question type: {question_type}
Question: {question}
Reference answer: {reference}
Responses:
{responses}
Output schema:
{"answer_slot": "short description of what kind of answer is requested", "responses": [{"id": "F001",
"disposition": "committed|abstain|conflicted", "extracted_answer": "answer or null", "cluster_id": "C1
or null"}], "clusters": [{"cluster_id": "C1", "canonical_answer": "concise answer", "member_ids":
["F001"], "representative_id": "F001", "relation_to_reference": "equivalent|different",
"answer_slot_fit": "valid|invalid", "reason": "short explanation"}]}
```

6. Do not merge merely related answers. Different counts, actions, times, directions, materials, object identities, or OCR strings are different answers.

7. relation\_to\_reference is equivalent only when the cluster would answer this question with the same meaning as the reference. This check remains required because the earlier per-response judge can make mistakes.

8. answer\_slot\_fit is valid only when the cluster supplies the requested kind of answer. A color is invalid for a material question; an explanation is acceptable only when one clean final answer can be extracted.

9. canonical\_answer must be a concise normalized form of the cluster's shared meaning.

representative\_id must name one actual member whose answer supports that canonical form.

## Completeness contract:

\- Process every supplied response ID exactly once.

\- Every committed response belongs to exactly one cluster.

\- Cluster member\_ids must exactly partition all committed responses.

\- Never invent, omit, or duplicate an ID.

\- Output one strict JSON object and no Markdown.

## BLIND\_DISTRACTOR\_CLUSTERING.

You are a conservative semantic auditor constructing multiple-choice distractor candidates from blind, text-only model responses. You do not see the image. Do all semantic normalization and clustering yourself; the caller will only validate IDs and count support mechanically.

First infer the exact answer slot requested by the question: for example count, OCR string, time, color, material, object identity, action, or spatial relation. Then audit every response independently.

## Response rules:

1. committed: the response ultimately selects exactly one identifiable answer. Extract that final answer even if it is surrounded by reasoning.

2. abstain: it says unknown, cannot determine, or refuses to guess.

3. conflicted: it leaves multiple incompatible alternatives without selecting one. A phrase such as 'red or blue' is conflicted when the question asks for one color. Do not arbitrarily keep one option. 4. For abstain or conflicted, extracted\_answer and cluster\_id must be null.

## Clustering rules:

5. Cluster committed answers only when they are interchangeable answers to this exact question. Normalize spelling, morphology, number words, abbreviations, and harmless extra wording semantically. 6. Do not merge merely related answers. Running, sitting, reading, and waving are different actions unless the response itself clearly commits to the same one. ALPHA, BETA, and GAMMA are different OCR strings. These are abstract boundary examples only; do not copy them into the result.

7. relation\_to\_reference is equivalent only when the cluster would answer the question with the same meaning as the reference.

8. answer\_slot\_fit is valid only when the cluster supplies the requested kind of answer. A color is invalid for a material question; an explanation is acceptable only when one clean final answer can be extracted.

9. canonical\_answer must be a concise normalized form of the cluster's shared meaning.   
representative\_id must name one actual member whose answer supports that canonical form.

\- Every committed response belongs to exactly one cluster.

\- Cluster member\_ids must exactly partition all committed responses.

\- Never invent, omit, or duplicate an ID.

\- Output one strict JSON object and no Markdown.

Question type: {question\_type}   
Question: {question}   
Reference answer: {reference}

Responses:

{responses}

Output schema:

{"answer\_slot": "short description of what kind of answer is requested", "responses": [{"id": "B001", "disposition": "committed|abstain|conflicted", "extracted\_answer": "answer or null", "cluster\_id": "C1 or null"}], "clusters": [{"cluster\_id": "C1", "canonical\_answer": "concise answer", "member\_ids": ["B001"], "representative\_id": "B001", "relation\_to\_reference": "equivalent|different", "answer\_slot\_fit": "valid|invalid", "reason": "short explanation"}]}

## VLM\_DISTRACTOR\_GENERATOR.

You are a conservative visual multiple-choice distractor generator. You see exactly one {view\_name} image. Generate zero to three short candidate answers that are visually motivated yet definitely incorrect for the question's intended target. The reference answer is immutable and correct. Returning no candidate is better than inventing unsupported content.

Infer the exact answer slot and the semantic core of the reference first. Every candidate must fill the same slot naturally and must differ in that core; changing an incidental adjective does not create a different answer. For object identity, use a genuinely different head noun, not another colored instance of the reference object. For actions, use a concrete visible action rather than clothing, possession, appearance, or posture. For counts, use only a defensible alternate inclusion boundary or nearby same-class group. For spatial/person identity, preserve the requested relation or entity type.

Preferred visual strategies, strongest first:

1. same\_property\_other\_instance: the requested property on another similar visible object.

2. analogous\_relation\_other\_object: a genuinely different object visible in an analogous relation or location.

3. target\_other\_attribute: only for scalar attributes such as color, material, or pattern, use another clearly visible value on the target.

4. nearby\_same\_slot: a clearly visible value of the exact requested type from a nearby object or region.   
5. count\_near\_miss: another visually defensible count, never an arbitrary number.

A different visible source is positive evidence: for a blue-hat color question, a yellow shirt or another person's white hat may supply yellow or white. Name the exact source and relation. Do not use world knowledge, generic alternatives, likely/perhaps/resembling evidence, unknown, the reference, or a synonym of it. MUTUAL EXCLUSIVITY: a candidate must be necessarily false whenever the reference is true, and an average viewer must be able to refute it by looking at the image. Never propose a broader or narrower rewording of the reference (stripes when the reference is white stripes, metal when it is stainless steel), a logically nested category (rectangular when the reference is square — every square is a rectangle), or a value so perceptually close to the reference that the image cannot distinguish them (square vs rectangle, circle vs near-circular oval). The thin red rectangle, if present, is an artificial locator and never image content. Rank strongest first, do not pad to three, and output strict JSON only.

View discipline: {view\_rules}

Question type: {question\_type}

## VISUAL\_DISTRACTOR\_CROP\_RULES.

Use only information visible in this expanded crop. Prefer a same-property value from another local instance, then another same-type attribute. Do not assume content from the unseen full image.

## VISUAL\_DISTRACTOR\_FULL\_RULES.

Use only information visible in the native full image. Borrow from a different object or text region and match the requested answer slot as closely as possible. You do not have access to the zoom crop.

## VLM\_DISTRACTOR\_VALIDATOR.

You are an independent conservative visual validator for proposed multiple-choice distractors. You see the same {view\_name} image used by the generator. The immutable reference is correct. Evaluate every candidate independently and do not generate replacements. A candidate should be wrong for the intended target but may deliberately borrow a value from a different visible object, attribute, action, count boundary, person, or text region. Do not reject merely because the source is elsewhere.

Infer the exact answer slot and reference semantic core. For scalar color, material, pattern, or orientation, a directly visible value from any source object is allowed. For object identity, reject any candidate sharing the reference head noun/category regardless of color, owner, or instance; accept only a genuinely different object in an analogous relation. For action, accept a concrete action visibly performed elsewhere but reject clothing, appearance, wearing, possession, and static attributes. For counts, require a defensible visible group/boundary. For spatial/person identity, preserve the requested relation/entity type.

For OCR, independently transcribe the claimed source instead of trusting the evidence description. The candidate must be an exact complete token or a question-licensed first/last-N span of that transcription; the full source may be longer, so PORTUGAL may be supported by LICOR DE PORTUGAL. A one-word question requires one word, and a name question rejects a bare route/count number, but another visible word need not share the target's real-world meaning. Source color and capitalization affect quality, not whether the answer string itself fits. If the text is too small to read independently, use reject or uncertain.

Accept only when the candidate has the same answer type/granularity, differs from the reference core, is not also correct for the target, has verified visual evidence, uses a valid strategy, and is concise/non-unknown. MUTUAL EXCLUSIVITY IS MANDATORY: 'not equivalent to the reference' is NOT enough. Reject any candidate that could be true together with the reference — broader/narrower rewordings (stripes vs white stripes), logically nested categories (rectangular when the reference is square), or values the image cannot refute because they are perceptually adjacent to the reference (square vs rectangle, circle vs near-circular oval; classic color pairs: silver vs white or grey, gold vs yellow, beige or cream vs white). The test: could a careful viewer point at the image and say 'definitely not this'? If not, reject. Reject generic, inferred, likely, or world-knowledge evidence. decision=accept requires quality high or medium. Process every ID exactly once and output strict JSON only.

## DISTRACTOR\_ONLY\_REWRITE.

You are editing multiple-choice distractor options. Rewrite ONLY the listed distractors. The reference answer is immutable context: never rewrite it, replace it, include it in the output list, or turn any distractor into the reference answer. You do not see the image.

Question type: {question\_type}

Question: {question}

Immutable reference answer: {reference}

Distractors to edit:

{distractors}

For each distractor:

1. Preserve exactly the answer it commits to. The rewrite must remain semantically interchangeable with that distractor for this question.

2. Make it a concise, natural, standalone answer to the question. Remove reasoning, hedging, redundant framing, and sentence-level filler.

3. Match the grammatical form and surface style of the immutable reference when useful, without changing the reference itself. For example, if the reference spells a count as a number word, spell distractor counts as number words too.

4. Never change a count value, time, OCR character sequence, proper name, direction, color, material, negation, object identity, action, or spatial relation. Formatting-only normalization is allowed when meaning is unchanged.

5. Do not add visual details, combine options, explain why an option is wrong, or mention the image or reference answer.

6. If a distractor is already concise, natural, and style-compatible, copy it unchanged.

7. Process every distractor ID exactly once. Do not add, drop, merge, or reorder IDs. Output strict JSON only.

Output schema:

{"distractors": [{"id": "D001", "rewritten\_answer": "concise answer", "changed": true, "reason": "brief editing reason"}]}

## MULTIPLE\_CHOICE\_POOL\_CURATOR.

You are the final text-only curator for a visual multiple-choice question. The immutable reference answer is correct. The entries come from several already-audited sources: jointly validated crop/full visual distractors, real wrong full-image model answers, qualified blind-prior answers, and possibly Qwen-only visual reserve candidates. You do not see the image. Do semantic organization only; do not invent, rewrite, improve, split, or add answer strings.

Treat provenance as evidence that an answer is a serious candidate, never as a reason to keep two semantically identical options. A borrowed scalar value remains allowed when it fits the requested slot: for a hat-color question, yellow may come from a visible yellow shirt. Infer the exact answer slot from the question and reference. Process every entry ID exactly once. Mark an entry eligible only when its answer string itself is a concise, committed answer in that exact slot and semantic granularity, semantically different from the reference, and not an abstention, explanation, or malformed fragment. Reject a related but different component, attribute, or category: for example, black rubber tires do not fit a wheel-style slot whose reference is silver alloy wheels, while spoked wheels do. Image correctness was checked upstream; this step judges only answer-slot fit, reference equivalence, mutual exclusivity with the reference, and cross-entry equivalence. A surviving distractor must be NECESSARILY FALSE when the reference is true: route to reference\_equivalent any broader, narrower, or logically nested description that can be true together with the reference (stripes when the reference is white stripes; rectangular when the reference is square — every square is a rectangle; metal when the reference is stainless steel), and route to invalid any entry an average viewer could not definitively refute because it is perceptually adjacent to the reference (square vs rectangle, circle vs near-circular oval). Use reference\_equivalent for paraphrases, aliases, broader/narrower wording that would still be accepted as the correct answer, numeric/number-word equivalents, and answers that merely add an attribute to the same requested object identity. Use invalid for wrong answer type, unknown/uncertain text, or content that does not stand alone as an option.

Cluster every eligible entry by the answer it commits to for this exact question. Merge semantic equivalents across all sources even when wording, articles, capitalization, inflection, number format, or specificity differ. Do not merge merely related alternatives that would be distinct choices. Each eligible ID must occur in exactly one cluster; rejected IDs occur in none. Choose representative\_id from the cluster itself, favoring the most concise natural existing answer. Never output a new canonical answer.

```jsonl
Question type: {question_type}
Question: {question}
Immutable reference answer: {reference}
Pool entries:
{entries}
Output strict JSON only:
{"answer_slot": "exact requested slot", "entries": [{"id": "P001", "disposition":
"eligible|reference_equivalent|invalid", "cluster_id": "C1 or null", "reason": "concise semantic
reason"}], "clusters": [{"cluster_id": "C1", "member_ids": ["P001"], "representative_id": "P001",
"relation_to_reference": "different", "answer_slot_fit": "valid", "reason": "shared committed
answer"}]}
```

## MULTIPLE\_CHOICE\_SOLVER.

Answer one multiple-choice visual question from the supplied zoom crop. The global question gives full-scene identity context; the crop question asks about the same target using wording appropriate for this crop. Use the crop question as the direct visual task. Inspect the image carefully and select exactly one displayed option. Do not explain, hedge, list multiple options, or use outside knowledge when the crop provides the answer.

Question type: {question\_type}   
Global question: {global\_question}   
Crop question: {crop\_question}   
Options:   
{options}   
Return strict JSON only, using one supplied choice label:   
{"choice": "A"}