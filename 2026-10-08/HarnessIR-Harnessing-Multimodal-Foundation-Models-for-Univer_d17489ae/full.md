POLYU VCLAB •PREPRINT 2026

# HarnessIR: Harnessing Multimodal Foundation Models for Universal Real-World Image Restoration

Xiangtao Kong<sup>1,2</sup> Shuaizheng Liu<sup>1,2</sup> Rongyuan Wu<sup>1,2</sup> Lingchen Sun<sup>1,2</sup> Zhengqiang Zhang<sup>1,2</sup> Jinxin Zhao<sup>1,2</sup> Yuhui Wu<sup>1,2</sup> Lei Zhang<sup>†1,2</sup>

<sup>1</sup> The Hong Kong Polytechnic University <sup>2</sup> OPPO Research Institute <sup>†</sup> Corresponding author (cslzhang@comp.polyu.edu.hk).

![](images/7011da1e19b6074aa62715a24053f806dc88fdf611be6ee0d94bf43b2268d246.jpg)  
Figure 1. Top and Bottom-left: Visual results of HarnessIR, which can restore image details without altering their content and texts. Bottom-right: Radar charts comparing diferent metrics on real-world test sets. HarnessIR achieves much better restoration performance than Nano Banana 2, GPT-Image-2.5, previous all-in-one and agentic IR methods.

Real-world low-quality images sufer from complex mixed degradations, including but not limited to noise, blur, atmospheric efects, etc. Recent agentic methods usually model real-world image restoration (Real-IR) as a sequential tool calling problem over task-specific single-degradation restoration models.This paradigm, however, is fundamentally limited because complex real-world degradations cannot be cleanly undone degradation by degradation, and the tool used for task-specific models caps the capability of the agent system. In this work, we present HarnessIR, an agentic framework for Real-IR by harnessing a multimodal foundation model (MFM) as the executor. HarnessIR consists of five stages: perception and diagnosis, on-demand tool invocation, prompt composition, execution, and verification-driven refinement. Unlike prior agentic Real-IR methods that rely on tool chains assembled from task-specific models, HarnessIR feeds the restoration requirements, the perceptual diagnosis, and the evidence into an MFM that performs restoration in a single pass, followed by verification stages to determine whether the result warrants further processing. Under our harness, of-the-shelf MFMs handle restoration tasks remarkably well, achieving state-of-the-art results on the widely used MiO100 synthetic benchmark. More importantly, by exploiting the strong generalization ability of MFMs, HarnessIR delivers compelling restoration quality on challenging real-world scenes where previous agentic IR systems often struggle. Codes is available at https://github.com/PolyU-VCLab/HarnessIR.

## 1 Introduction

Image restoration (IR) is a classical yet challenging research problem. Early IR research mainly focuses on a specific type of degradation, such as denoising [1, 2], deblurring [3, 4], super-resolution [5, 6], dehazing [7, 8], etc. Later, many all-in-one [9, 10] or multiple-in-one [11, 12] IR methods have been proposed, aiming to train a unified model to handle diferent types of image degradations. However, most of these methods still assume that the input image is corrupted by one single type of degradation, and they explicitly or implicitly recognize the degradation before restoring the desired image. In practice, however, captured images in real-world are often corrupted by complex, mixed degradations that cannot be modeled by a specific degradation type.

Inspired by the recent rapid development of agentic models in multimodal understanding and generation [13, 14], agentic IR methods [15, 16] have been proposed to address the issue of complex mixed degradations, where pretrained task-specific IR networks are exposed as restoration tools, and vision-language models (VLMs) are employed as the perceiver to assess image content and its degradations, map the identified degradations onto specialized IR models, and as the planner to schedule them into an execution sequence, frequently augmented with reflection, rollback, and rescheduling. However, this formulation is subject to fundamental limitations. First, it approximates coupled mixed degradations as a sequence of specialized restoration operations, whereas many real-world degradations interact and cannot be cleanly separated or reversed in sequence. Second, the planner converts fine-grained description of image content and degradation from the perceiver into a sequence of restoration tool calls, so much useful information is lost before restoration. Third, task-specified IR tools generalize poorly, limiting the overall generalization ability of the whole system. Last but not least, most agentic IR methods are developed and evaluated on synthetic mixed degradations, whereas real-world degradations often interact with each other, making them dificult to simulate. Therefore, although these paradigms have achieved great progress, their generalization to real-world scenes remains limited.

Recently, generative multimodal foundation models (MFMs) [17–19] have advanced rapidly. Having been trained for image editing, these models demonstrate great potential for IR tasks in recent works [20, 21]. Basically, MFMs treat IR as an editing problem: they accept the LQ image together with an instruction to accomplish the restoration task. Unfortunately, prompting MFMs naively often yields unsatisfying results, as illustrated in Fig. 1 and Fig. 3, these models often over-generate, embellishing detail and repainting texts and image contents. However, we argue that this does not mean that MFMs cannot perform complex real-world IR tasks; the key lies in how we instruct them, together with additional tools, to execute IR tasks in appropriate steps, harnessing the strong generative and generalization capabilities of MFMs for high-quality, universal real-world image restoration (Real-IR).

To this end, we present HarnessIR, an agentic framework for universal real-world IR by harnessing an MFM as the executor. HarnessIR is organized into five stages: perception and diagnosis, on-demand tool invocation, prompt composition, execution, and verification-driven refinement. One key distinction between our design and previous work is that HarnessIR schedules no restoration tools, and it performs restoration by MFM calls; however, it invokes auxiliary tools — OCR [22], face detection [23], depth estimation [24], semantic segmentation [25] — which act as evidence providers for prompt composition rather than links in a restoration chain. As shown in Fig. 2, the LQ image and the restoration requests are first passed to a VLM, which returns a diagnosis of the content and its degradations, along with a recommendation on which auxiliary tools to invoke. The system executes the plan and collects the resulting evidence, which is fed back to the VLM with the LQ input and the first-stage analysis. The VLM then composes a prompt that contains rich content and degradation information, as well as the gathered evidence rendered in language. The prompt and the LQ image are passed to the MFM executor, and the VLM then examines the output to determine whether the restoration is satisfactory and how to revise the prompt for re-execution. HarnessIR binds the MFM to what the given LQ image actually needs for restoration, eliciting the MFM’s restoration capability while regulating how far it can alter the input, so that degradations can be removed without changing the content.

Using advanced MFMs such as Nano Banana 2 [17] and GPT-Image-2.5-Sunburst [18] as executors, HarnessIR handles image restoration remarkably well. On MiO100 [11], a widely used synthetic mixeddegradation benchmark, HarnessIR surpasses existing all-in-one and agentic IR methods on almost every metric. On the more challenging real-world images captured with both single and mixed degradations, where many all-in-one and agentic IR methods fail outright, HarnessIR outperforms prior methods by a large margin. Our ablation studies attribute these restoration gains to the harness rather than the executor alone. Our contributions are summarized as follows:

• We first identify the limitations in the previous agentic IR paradigms and propose moving the agent action primitive from restoration tool scheduling to MFM harnessing.

• We then propose HarnessIR, an agentic framework that harnesses a single MFM as the restorer without using task-specific restoration models as tools.

• Experiments show that HarnessIR reaches leading performance on the widely used MiO100 benchmark; more importantly, it restores real-world images with complex degradations significantly better than existing agentic IR methods.

## 2 Related Work and Discussion

All-in-One and Agent-based Image Restoration. Early IR research usually addresses a specific type of degradation in isolation, such as denoising [1], deblurring [3], super-resolution [5], dehazing [8], etc. Subsequently, many all-in-one and multi-in-one methods have been proposed to address diferent degradations using a unified network, such as contrastive degradation embeddings [9], degradation-aware learnable prompts [10, 11], CLIP-based degradation cues [12, 26], and natural-language instructions [27]. However, most of these methods assume that the input LQ image still has a single identifiable degradation, which limits their performance in unconfined real-world scenes [28], where the captured images are often corrupted by complex and mixed degradations. To handle mixed degradations, agentic IR [15] takes task-specific restoration networks as tools and plans a sequence of calls over them. Existing agentic IR systems difer mainly in the design of the sequence. One line of research searches the space of tool orderings and uses intermediate results to update the plan [15, 16, 29]. Another line of work constrains the plan using degradation priors, staged decomposition, or complexity-based routing [30–32]. The third line learns the policy directly via reinforcement learning using perceptual or no-reference quality rewards [33–36]. In all these paradigms, the tools are task-specific restoration networks, which are also executors. Consequently, their generalization performance is bounded by these tools, which work poorly under complex, out-of-distribution real-world degradations.

Multimodal Foundation Models (MFMs). Generative MFMs [18, 19, 37] and their instruction-based editing interface allow a single model to perform an open set of pixel-level operations specified in language. Since restoration can be phrased as an editing instruction, researchers have begun to examine MFMs for IR. [20] evaluated Nano Banana 2 [17] as a unified restorer. While the perceptual results are appealing, it over-synthesizes many unfaithful details. Similarly, [21] prompted MFM with fixed instructions and found that the fidelity of the image is dificult to preserve. However, we argue that this does not mean that MFMs lack the capability for performing faithful Real-IR. The key lies in how we prompt them. With an appropriate harness to elicit and utilize their capabilities, the MFMs can significantly improve perceptual quality while preserving image content.

Harness: From Foundation Models to Agentic Systems. An MFM is not by itself an agent. What turns it into an agentic system is the harness [38], which includes the prompts and control flow that frame the task [39], the tools that collect evidence [40], the verification logic that decides when an output is acceptable [41], and the memory if needed [42]. With an appropriate harness, the base MFM can become a markedly strong agent depending on what is wrapped around it [38, 43]. Our proposed HarnessIR is a diferent yet much more efective instantiation for agentic IR: it diagnoses before acting, calls auxiliary tools for evidence, explicitly constrains what should be restored, and employs a verifier to decide whether to try again based on the current results.

## 3 HarnessIR

## 3.1 Problem Definition and Overview

Given an LQ image � and a user request �, we aim to produce a restored image �ˆ. The request is a short natural-language sentence provided by the user, expressing the restoration intent —e.g., improve the brightness of the photo (see the Appendix A.1). As shown in Fig. 2(a), prior agentic IR methods [15, 16] usually cast this as a composition problem over a library of task-specific restoration networks $\{ f _ { t } \} _ { t \in \mathcal { T } } \colon$ a perception step reads � into a description, a planner turns that description into a tool order $( t _ { 1 } , \dots , t _ { K } )$ , and the image is passed through the corresponding restoration networks one after another, with reflection between the steps. In contrast, our HarnessIR performs restoration with a call to a frozen MFM executor $\varepsilon .$ , driven by a prompt $p \mathrm { : }$

![](images/25884584ea26fad25d45b58098cb585f1845bc79e474c7fb1b9f428ff1b26978.jpg)  
Figure 2. Comparison between (a) prior agentic IR pipelines (top) and (b) HarnessIR (bottom). Prior methods sequentially apply task-specific restoration models planned by VLMs to produce the output. HarnessIR instead uses auxiliary tools for image-specific evidence, composes a restoration instruction for MFM, and invokes verification-driven re-execution when necessary.

$$
\mathrm { P r i o r ~ A g e n t i c ~ I R : \quad } \hat { y } ^ { ( k ) } = f _ { t _ { k } } \big ( \hat { y } ^ { ( k - 1 ) } \big ) , \qquad \mathrm { H a r n e s s I R : \quad } \hat { y } ^ { ( k ) } = \mathcal { E } \big ( x , p ^ { ( k ) } \big ) ,\tag{1}
$$

where $k = 1 , \ldots , K$ counts rounds and ${ \hat { y } } ^ { ( 0 ) } = x$ . Prior agentic IR methods vary the restoration executor to process intermediate outputs sequentially. Instead, HarnessIR keeps the executor fixed and varies only the prompt. In each round, it processes the original input � and therefore constitutes an alternative complete result. The verifier selects the returned candidate, $\hat { y } = \hat { y } ^ { ( k ^ { * } ) }$ , and requests another round only when necessary (Sec. 3.4).

We use five stages to achieve our goal, as illustrated in Fig. 2(b). We instantiate the diagnoser, composer, and verifier using the same frozen vision-language model (VLM) [44] with three role-specific briefs. The diagnoser $\mathcal { V } _ { d }$ produces a diagnosis � of image content and degradations, together with a plan $\pi \subseteq \mathcal { T } _ { \mathrm { a u x } }$ specifying what auxiliary tools to consult. After the planned tools return evidence �, the composer $\mathcal { V } _ { c }$ writes $p$ from $( x , r , d , e )$ the executor E produces ${ \hat { y } } ;$ and the verifier $\mathcal { V } _ { \nu }$ decides whether the result is accepted and which candidate is retained. Diagnosis and composition are separated because the final instruction can only be written after the diagnoser selects the tools and the requested evidence has been acquired. HarnessIR focuses on how to construct an image-specific instruction and how to verify and re-execute. In the following sections, we describe its evidence collection, prompt composition, and failure-triggered refinement in detail.

## 3.2 What is Planned: Evidence to Gather, Not Restorers to Run

HarnessIR plans over a diferent space from previous agentic IR methods [15, 30], which search over orderings of restoration networks to decide how the image will be processed. Instead, HarnessIR plans what the system should perceive before writing $p .$ The set of auxiliary tools — text recognition, face detection, depth estimation, and semantic segmentation — supplies observations that the executor would not otherwise seek on its own. For a capable MFM, the simpler the prompt it is given, the more freely it follows its own prior, which may result in hallucinations for restoration. The evidence and the instructions elicit what the executor should do to reduce the

possibility for hallucination.

The tools to consult are decided by the diagnoser that produces � in the same call, rather than by fixed rules, because the decision depends on the content of the image: text recognition is worth running when legible text is present, and depth estimation is worth taking when the degradation is spatially structured. For images with simple contents and degradation, the tools may not need to be called. The invoked tools return raw, image-specific observations, e.g., recognized strings, confidences, and bounding boxes for texts; bounding boxes for faces; a dense depth map; and a segmentation map with semantic labels. These outputs are supplied directly to the composer in their native structured or visual forms, and they are not passed to the executor. A complete example is provided in Appendix A.2.

## 3.3 What is Passed to Executor: Instruction Prompt, Not Execution Order

The diagnosis is expressed in language, while the evidence from the tool may be structured or visual. The composer interprets this evidence jointly with the LQ image rather than merely transcribing it, and converts the relevant observations into a single textual instruction. The evidence is folded into the image description rather than appended as a list, each piece tied to the content it describes and the degradation it bears witness to. For example, the haze is to be cleared more aggressively in the background than in the foreground, as the depth indicates. The composer decides whether to accept the evidence or weigh the evidence against the LQ input �.

In addition, the composer’s role-specific brief includes general restoration requirements that encode domain knowledge about each restoration operation. For example, deraining and desnowing should remove falling rain and snow without removing wet roads or snow cover in the scene, while brightness should remain unchanged when low-light enhancement is not requested. The composer jointly considers the provided information and these requirements, retains only the applicable requirements, and rewrites them into an image-specific prompt �. The executor receives only the LQ image � and the composed prompt �. Unlike prior agentic IR methods, which formulate their analysis as a list of task-specific restoration tools, HarnessIR embeds image-specific information into the prompt, including what degradations to treat, what content to preserve, and where these requirements apply. The complete brief of the composer is provided in the released codes.

## 3.4 When the Loop Fires: On Failure, Not by Design

HarnessIR is designed to succeed in one execution and invoke another round only when the verifier judges the result unsuccessful. The output is first measured using NR-IQA scores and inexpensive comparisons with �, including brightness, color statistics, and SSIM between the output and the LQ input. These measurements provide diagnostic evidence rather than fixed acceptance thresholds: they are passed to the verifier together with � and the output, and the verifier makes the final decision. The verifier carries the same restoration requirements as the composer and weighs the output against them, together with the request � and the measurements. The instruction to the executor is not shown to the verifier so that the result is judged against what the restoration task is supposed to do rather than against the instruction we ask the executor to follow.

HarnessIR starts over from �, not the output of the last round as in previous agentic IR methods. This is because if the output of the last round is verified to be unsuccessful, the content of the image can be altered, such as incorrect text, fabricated structure, etc. Such an output is the wrong place to start from, while � remains the safest input for the executor to start from. When the verifier calls for a redo, it writes down what should be corrected. The composer then writes a new instruction for the same �, and passes it to the executor. The verifier also compares the new restoration result with the best candidate obtained in previous rounds and retains the better one as �ˆ. We allow at most three execution rounds; if no candidate is accepted by then, the best candidate selected by the verifier is returned. The complete verifier prompts are provided in the released codes.

## 4 Experiments

## 4.1 Experimental Setup

Implementation of HarnessIR. HarnessIR is inference-only: no component is trained or fine-tuned. The diagnoser, composer, and verifier are all filled by Gemini-3.7-Flash [44], queried once per role under diferent briefs (the system prompts). We run HarnessIR with two MFMs: Nano Banana 2 (NB2) [17] and GPT-

![](images/433d27a3159213fabad11f4cf0b48431d8c439055c1b26e2d716cbd0d14829e8.jpg)  
Figure 3. Direct use of MFMs may yield higher NR-IQA scores than GT, even when the image content is largely altered. NR-IQA should not be used as an independent metric for IR performance.

Image-2.5-Sunburst (GPT-Image) [18], each accessed via its public interface. Four auxiliary tools are used: PP-OCRv6 [22] for text recognition, InsightFace Project [23] for face detection, Depth-Anything-V2 [24] for depth estimation, and SAM 3 [25] for semantic segmentation. We use a fixed request template for each degradation category to represent the usage of general users (see the Appendix A.1).

For each executor E, we report two variants under exactly the same LQ image and category-specific request �: Baseline-E, which directly receives the LQ image and �, representing direct MFM usage; HarnessIR-E, our proposed pipeline with at most three MFM execution rounds per image. The diference between the two variants lies in only the harness, rather than the user request.

Test Sets. We evaluate HarnessIR on three test sets. One is the synthetic test set MiO100 [11, 15], which has three degradation groups (A, B, and C). Since our main concern is the model performance on real-world LQ images, following SEAR [45], which aggregates 100 real-world pairs, we collect two larger real-world test sets from public datasets: Real-Paired-200 with paired GT and Real-NoGT-200 without GT. They cover unconstrained mixed degradations, as well as haze, rain, snow, low light and so on. The composition and sources are detailed in the Appendix B, and both sets will be made publicly available.

Compared Methods. We compare HarnessIR against six all-in-one models (AirNet [9], PromptIR [10], MiOIR [11], DA-CLIP [12], InstructIR [27], AutoDIR [26]) and six agentic methods (AgenticIR [15], MAIR [30], 4KAgent [16], TIR-Agent [34], SEAR [45], OPERA [36]). For those methods whose codes and weights are publicly available, we reproduce their results with the oficial implementations, and report more NR-IQA metrics beyond the ones reported in their original papers. For those methods whose codes are not released, we only quote the MiO100 results reported in their papers.

## 4.2 Evaluation Protocol and Metrics

Following prior works [15, 21, 45], we measure fidelity with PSNR and SSIM [46] in the Y channel, perceptual quality with LPIPS [47] and DISTS [48], and image quality with the no-reference image quality assessment (NR-IQA) metrics MANIQA [49], CLIP-IQA [50], MUSIQ [51], TOPIQ [52], and A-FINE [53]. However, it should be noted that a higher NR-IQA score does not, by itself, indicate better restoration performance because their purpose is to evaluate image quality without considering whether the restored content is correct or not. As a result, hallucination can be rewarded, even when the content of the restored image deviates significantly from the input LQ image. Fig. 3 shows some examples. We see that the restored images significantly outscore their GTs on NR-IQA metrics, while the image content and text have been largely repainted. Such cases are rare for earlier discriminative restoration networks, but can be common for difusion-based generative models. This is because the generative methods are more prone to altering image content and NR-IQA metrics only measure whether the given image is of good quality, no matter what content it has.

To solve this issue of unreasonable use of NR-IQA metrics in Real-IR method evaluation, we make two adjustments. First, when a GT exists, we report the absolute distance, denoted by Δ, between the NR-IQA scores of the restored image and the GT image. A value of Δ closer to zero means better quality since the GT is the target. Second, regardless of whether a GT exists, following recent IR and IQA works [54, 55], we use an independent VLM evaluator of the HarnessIR system to assess restoration performance directly. Given the LQ input and the restored output, it reports D-Score for degradation removal and F-Score for content preservation, <sub>both on a scale of 0–100. For each image, we compute their geometric mean, DF-Score =</sub> √<sub>��, which is</sub> high when both degradation removal and fidelity are good. The evaluator prompt and protocol, ablations, and comparisons with human judgments are provided in the Appendix C.

Table 1. Group-averaged results on MiO100. The best and second-best results are marked in red and blue, respectively. Results in which HarnessIR improves on its baseline are in bold. Δ means the absolute deviation from the GT score; the smaller, the better. The GT and LQ rows report the raw metric values. D-Score and F-Score are given by the VLM evaluator for degradation removal and content preservation, both on a scale of 0–100; DF-Score is their per-image geometric mean.
<table><tr><td rowspan="2">Method</td><td colspan="4">Full-reference Metrics</td><td colspan="6">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>∆MANIQA↓</td><td>∆CLIP-IQA↓</td><td>ΔMUSIQ↓</td><td>ΔTOPIQ↓</td><td>ΔAFINE-NR↓</td><td>D-Score↑</td><td>F-Score↑</td><td>DF-Score↑</td></tr><tr><td>AirNet</td><td>20.30</td><td>0.65</td><td>0.46</td><td>0.25</td><td>0.22</td><td>0.28</td><td>29.00</td><td>0.32</td><td>0.27</td><td>25.1</td><td>89.7</td><td>29.78</td></tr><tr><td>PromptIR</td><td>20.66</td><td>0.66</td><td>0.45</td><td>0.26</td><td>0.21</td><td>0.28</td><td>28.98</td><td>0.33</td><td>0.26</td><td>26.4</td><td>91.7</td><td>30.41</td></tr><tr><td>MiOIR</td><td>20.76</td><td>0.66</td><td>0.43</td><td>0.25</td><td>0.22</td><td>0.29</td><td>27.29</td><td>0.33</td><td>0.26</td><td>28.5</td><td>90.6</td><td>37.09</td></tr><tr><td>DA-CLIP</td><td>20.33</td><td>0.64</td><td>0.46</td><td>0.26</td><td>0.22</td><td>0.29</td><td>28.37</td><td>0.33</td><td>0.25</td><td>19.4</td><td>87.7</td><td>25.47</td></tr><tr><td>InstructIR</td><td>20.84</td><td>0.60</td><td>0.54</td><td>0.28</td><td>0.24</td><td>0.31</td><td>31.84</td><td>0.37</td><td>0.32</td><td>21.9</td><td>90.6</td><td>29.42</td></tr><tr><td>AutoDIR</td><td>20.59</td><td>0.65</td><td>0.39</td><td>0.23</td><td>0.14</td><td>0.25</td><td>16.92</td><td>0.25</td><td>0.15</td><td>38.9</td><td>68.0</td><td>39.05</td></tr><tr><td>AgenticIR</td><td>21.15</td><td>0.67</td><td>0.35</td><td>0.21</td><td>0.14</td><td>0.18</td><td>11.73</td><td>0.19</td><td>0.07</td><td>49.4</td><td>43.3</td><td>42.97</td></tr><tr><td>MAIR</td><td>20.45</td><td>0.64</td><td>0.33</td><td></td><td>0.32</td><td>0.18</td><td>11.61</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4KAgent</td><td>20.73</td><td>0.63</td><td>0.34</td><td>0.20</td><td>0.10</td><td>0.11</td><td>8.31</td><td>0.12</td><td>0.11</td><td>47.1</td><td>36.3</td><td>36.88</td></tr><tr><td>TIR-Agent</td><td>21.57</td><td>0.66</td><td>0.32</td><td></td><td>0.25</td><td>0.10</td><td>6.83</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEAR</td><td>21.51</td><td>0.67</td><td>0.34</td><td>-</td><td>0.32</td><td>0.17</td><td>10.56</td><td></td><td>-</td><td></td><td></td><td></td></tr><tr><td>OPERA</td><td>21.78</td><td>0.69</td><td>0.37</td><td>–</td><td>0.32</td><td>0.22</td><td>12.67</td><td>1</td><td>、</td><td></td><td>一</td><td></td></tr><tr><td>GPT-Image-2.5</td><td>17.95</td><td>0.54</td><td>0.36</td><td>0.20</td><td>0.02</td><td>0.10</td><td>4.64</td><td>0.06</td><td>0.01</td><td>48.8</td><td>20.9</td><td>30.38</td></tr><tr><td>HarnessIR-GPT</td><td>19.70</td><td>0.58</td><td>0.30</td><td>0.17</td><td>0.01</td><td>0.07</td><td>2.81</td><td>0.02</td><td>0.01</td><td>67.5</td><td>37.7</td><td>48.44</td></tr><tr><td>Nano Banana 2</td><td>20.81</td><td>0.63</td><td>0.24</td><td>0.13</td><td>0.00</td><td>0.03</td><td>0.70</td><td>0.04</td><td>0.00</td><td>76.7</td><td>44.9</td><td>55.87</td></tr><tr><td>HarnessIR-NB2</td><td>22.21</td><td>0.65</td><td>0.21</td><td>0.12</td><td>0.00</td><td>0.03</td><td>0.25</td><td>0.04</td><td>0.00</td><td>75.2</td><td>51.7</td><td>59.94</td></tr><tr><td>GT</td><td></td><td></td><td></td><td></td><td>0.64</td><td>0.65</td><td>68.79</td><td>0.64</td><td>-0.92</td><td></td><td></td><td></td></tr><tr><td>LQ</td><td>19.86</td><td>0.57</td><td>0.58</td><td>0.31</td><td>0.23</td><td>0.28</td><td>32.77</td><td>0.37</td><td>0.31</td><td></td><td></td><td></td></tr></table>

Table 2. Results on Real-Paired-200, which contains 200 real-world LQ images with paired GT from public real-world datasets. Marking and metrics follow Tab. 1.
<table><tr><td>Method</td><td colspan="4">Full-reference Metrics</td><td colspan="5">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>∆MANIQA↓</td><td>∆CLIP-IQA↓</td><td>ΔMUSIQ↓</td><td>∆TOPIQ↓</td><td>ΔAFINE-NR↓</td><td>D-Score↑</td><td>F-Score↑</td><td>DF-Score↑</td></tr><tr><td>AirNet</td><td>22.40</td><td>0.724</td><td>0.440</td><td>0.254</td><td>0.108</td><td>0.107</td><td>13.27</td><td>0.109</td><td>0.133</td><td>16.8</td><td>85.7</td><td>25.2</td></tr><tr><td>PromptIR</td><td>24.08</td><td>0.743</td><td>0.416</td><td>0.241</td><td>0.098</td><td>0.099</td><td>12.78</td><td>0.105</td><td>0.114</td><td>16.7</td><td>91.3</td><td>25.3</td></tr><tr><td>MiOIR</td><td>24.87</td><td>0.735</td><td>0.418</td><td>0.251</td><td>0.096</td><td>0.086</td><td>12.20</td><td>0.104</td><td>0.124</td><td>10.5</td><td>93.0</td><td>18.9</td></tr><tr><td>DA-CLIP</td><td>23.87</td><td>0.739</td><td>0.406</td><td>0.243</td><td>0.099</td><td>0.106</td><td>11.83</td><td>0.093</td><td>0.111</td><td>17.8</td><td>88.2</td><td>26.7</td></tr><tr><td>InstructIR</td><td>22.92</td><td>0.747</td><td>0.411</td><td>0.236</td><td>0.108</td><td>0.116</td><td>12.34</td><td>0.104</td><td>0.124</td><td>25.4</td><td>90.2</td><td>36.9</td></tr><tr><td>AutoDIR</td><td>23.80</td><td>0.735</td><td>0.372</td><td>0.233</td><td>0.079</td><td>0.093</td><td>4.19</td><td>0.052</td><td>0.046</td><td>29.0</td><td>74.9</td><td>32.5</td></tr><tr><td>AgenticIR</td><td>22.41</td><td>0.734</td><td>0.398</td><td>0.235</td><td>0.097</td><td>0.087</td><td>7.41</td><td>0.072</td><td>0.083</td><td>33.7</td><td>68.4</td><td>38.8</td></tr><tr><td>4KAgent</td><td>22.18</td><td>0.708</td><td>0.389</td><td>0.231</td><td>0.088</td><td>0.064</td><td>2.89</td><td>0.030</td><td>0.131</td><td>35.2</td><td>56.1</td><td>37.4</td></tr><tr><td>GPT-Image-2.5</td><td>20.87</td><td>0.660</td><td>0.361</td><td>0.222</td><td>0.095</td><td>0.201</td><td>15.78</td><td>0.212</td><td>0.180</td><td>69.0</td><td>27.5</td><td>40.1</td></tr><tr><td>HarnessIR-GPT</td><td>23.74</td><td>0.729</td><td>0.286</td><td>0.176</td><td>0.066</td><td>0.131</td><td>12.32</td><td>0.128</td><td>0.108</td><td>82.7</td><td>50.4</td><td>60.2</td></tr><tr><td>Nano Banana 2</td><td>23.94</td><td>0.729</td><td>0.309</td><td>0.181</td><td>0.058</td><td>0.077</td><td>11.91</td><td>0.118</td><td>0.074</td><td>67.9</td><td>47.8</td><td>52.9</td></tr><tr><td>HarnessIR-NB2</td><td>26.63</td><td>0.775</td><td>0.246</td><td>0.149</td><td>0.020</td><td>0.001</td><td>5.52</td><td>0.028</td><td>0.033</td><td>75.2</td><td>63.6</td><td>62.0</td></tr><tr><td>GT</td><td></td><td></td><td></td><td></td><td>0.560</td><td>0.451</td><td>54.36</td><td>0.417</td><td>-0.901</td><td></td><td></td><td></td></tr><tr><td>LQ</td><td>25.27</td><td>0.736</td><td>0.436</td><td>0.258</td><td>0.098</td><td>0.092</td><td>13.08</td><td>0.107</td><td>0.115</td><td></td><td></td><td></td></tr></table>

## 4.3 Main Results

Results on Synthetic MiO100 Test Sets. Tab. 1 summarizes the group-averaged results on MiO100 (detailed results of Group A, B and C are in Appendix D). Compared with the direct-use baselines (Nano Banana 2 and GPT-Image-2.5), HarnessIR-NB2 and HarnessIR-GPT gain 1.40 and 1.75 dB in PSNR, improve LPIPS by 0.03 and 0.06, and DF-Score by 4.1 and 18.1 points, respectively. These gains mainly reflect better content preservation. HarnessIR also substantially improves GPT-Image-2.5’s D-Score. For NB2, the D-Score changes only slightly because MiO100 is constructed by applying controlled synthetic degradations to curated GT images. NB2 can already remove these degradations efectively.

Against prior methods, HarnessIR-NB2 achieves the best results on most metrics except SSIM and F-Score. HarnessIR-GPT achieves the best result on ΔTOPIQ. Several all-in-one models achieve relatively high F-Scores but D-Scores below 30, suggesting that their fidelity comes partly from making only limited restoration changes.

Table 3. Results on Real-NoGT-200. Marking and metrics follow Tab. 1. Without GT, NR-IQA alone cannot correctly reflect the restoration performance, so these columns are grayed out.
<table><tr><td rowspan="2">Method</td><td colspan="5">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td>MANIQA↑</td><td>CLIP-IQA↑</td><td>MUSIQ↑</td><td>TOPIQ↑</td><td>AFINE-NR↓</td><td>D-Score↑</td><td>F-Score↑</td><td>DF-Score↑</td></tr><tr><td>AirNet</td><td>0.525</td><td>0.406</td><td>42.94</td><td>0.307</td><td>-0.689</td><td>13.4</td><td>86.5</td><td>21.6</td></tr><tr><td>PromptIR</td><td>0.531</td><td>0.413</td><td>43.17</td><td>0.310</td><td>-0.703</td><td>11.2</td><td>90.4</td><td>21.0</td></tr><tr><td>MiOIR</td><td>0.537</td><td>0.399</td><td>44.44</td><td>0.321</td><td>-0.695</td><td>11.4</td><td>90.2</td><td>22.5</td></tr><tr><td>DA-CLIP</td><td>0.529</td><td>0.395</td><td>43.50</td><td>0.311</td><td>-0.700</td><td>12.6</td><td>88.3</td><td>20.3</td></tr><tr><td>InstructIR</td><td>0.521</td><td>0.383</td><td>43.39</td><td>0.311</td><td>-0.689</td><td>21.4</td><td>86.0</td><td>32.0</td></tr><tr><td>AutoDIR</td><td>0.538</td><td>0.399</td><td>47.85</td><td>0.337</td><td>-0.732</td><td>17.9</td><td>83.7</td><td>27.0</td></tr><tr><td>AgenticIR</td><td>0.525</td><td>0.374</td><td>45.38</td><td>0.325</td><td>-0.697</td><td>19.1</td><td>78.1</td><td>26.9</td></tr><tr><td>4KAgent</td><td>0.516</td><td>0.396</td><td>48.29</td><td>0.343</td><td>-0.677</td><td>18.9</td><td>70.4</td><td>25.5</td></tr><tr><td>GPT-Image-2.5</td><td>0.676</td><td>0.693</td><td>72.92</td><td>0.658</td><td>-1.059</td><td>59.3</td><td>16.9</td><td>29.6</td></tr><tr><td>HarnessIR-GPT</td><td>0.660</td><td>0.665</td><td>70.28</td><td>0.610</td><td>-1.000</td><td>76.4</td><td>36.0</td><td>49.3</td></tr><tr><td>Nano Banana 2</td><td>0.623</td><td>0.534</td><td>66.18</td><td>0.531</td><td>-0.933</td><td>65.1</td><td>34.4</td><td>47.3</td></tr><tr><td>HarnessIR-NB2</td><td>0.605</td><td>0.484</td><td>62.08</td><td>0.483</td><td>-0.904</td><td>71.3</td><td>64.1</td><td>62.9</td></tr></table>

Although HarnessIR-GPT improves substantially over the direct use of GPT-Image-2.5, it is weak in PSNR and SSIM. This reflects the executor’s capabilities rather than a limitation of the harness: the harness helps to unleash the executor’s restoration potential but cannot raise its upper ceiling. GPT-Image-2.5 has a strong generative prior but tends to restore more aggressively, which makes its outputs less pixel-aligned with the GT.

Results on Real-Paired-200 and Real-NoGT-200 Test Sets. The advantage of HarnessIR is even more pronounced on real-world data. On Real-Paired-200 (Tab. 2), both HarnessIR variants substantially improve their direct-use baselines in fidelity and restoration quality. HarnessIR-NB2 achieves a 2.69 dB PSNR gain and reaches a DF-Score of 62.0, while HarnessIR-GPT improves PSNR by 2.87 dB and raises its DF-Score from 40.1 to 60.2. HarnessIR also outperforms all prior methods by a clear margin in DF-Score, with both variants exceeding the previous best of 38.8. HarnessIR-NB2 also achieves the smallest deviation from the GT on four of the five NR-IQA metrics. In particular, HarnessIR-NB2 achieves the highest PSNR, surpassing the best prior result by 1.76 dB. This is notable because the LQ input already reaches 25.27 dB; the high PSNR but low D-Scores of several all-in-one methods therefore likely reflect limited restoration, whereas HarnessIR-NB2 improves upon the input while maintaining strong degradation removal.

The results on Real-NoGT-200 (Tab. 3) lead to the same conclusion. HarnessIR substantially improves both D-Score and F-Score for the two executors, yielding DF-Scores of 62.9 for NB2 and 49.3 for GPT-Image-2.5. These scores are well above the best prior result of 32.0. The lower NR-IQA scores of HarnessIR relative to the direct use of MFM should therefore not be interpreted as a decline in performance. As discussed in Fig. 3, NR-IQA can reward hallucinated details even when the image content has been altered. We consequently use DF-Score for the no-GT setting, as it evaluates degradation removal and content preservation jointly.

Visual Results. Fig. 4 shows qualitative comparisons across the three test sets. The all-in-one methods make little visible improvement over the LQ inputs in these examples, with only PromptIR removing some of the snow in the third image. Previous agentic IR methods make the MiO100 image somewhat clearer, but produce only limited improvement on the two real-world images. In contrast, the MFM-based methods remove degradations more efectively. However, direct use of NB2 and GPT-Image-2.5 often alters image content. In the first example, HarnessIR better preserves the original color tone and scene content. In the second, both direct MFM baselines mistakenly remove the white cross already in the image, and NB2 also changes the text. In the third, both baselines turn the nighttime scene into daytime and remove the snow cover; GPT-Image-2.5 additionally makes the trees appear to have sprouted leaves. HarnessIR avoids these unwanted changes while still removing degradations, producing visually cleaner results that remain faithful to the original LQ images.

## 4.4 Ablation Studies

![](images/3adfa28c11644c0da3795f97144f36ebc061b1d395e5947fbcea68ef9bf333bd.jpg)  
Figure 4. Visual comparisons on three test sets. HarnessIR achieves better fidelity and visual quality.

## 4.4.1 Ablation ofExecutor Inputs

We first explore how information is prepared for and supplied to the MFM executor on Real-Paired-200. The settings vary depending on the user request, image-specific diagnosis, auxiliary-tool evidence, and whether that evidence is supplied as visual outputs or expressed in text. We use executor inputs to refer to this information; the LQ image is provided in every setting. To isolate their efects, all variants use a single execution round, without verification and re-execution. Tab. 4 compares five settings. In the baseline, the executor receives only the LQ image � and the user’s request �. Detailed User Request replaces the short request with a more explicit paragraph that a user could plausibly provide, but adds no image-specific analysis. Single-VLM Instruction uses one VLM to diagnose the image and compose the instruction without using any auxiliary tools. The final two settings follow HarnessIR’s use of separate VLM stages and auxiliary tools. In Visual Evidence, the executor also receives the tools’ original outputs, including the depth map and semantic segmentation map. In Text Evidence, the relevant evidence is composed into language by the composer, which matches the executor-input setting used by HarnessIR.

The results show a steady improvement as progressively more information is supplied to the executor. Expanding the user request alone raises PSNR from 23.94 to 24.75 dB and DF-Score from 52.9 to 54.5. Image-specific analysis by a single VLM further increases the DF-Score to 56.6. Adding auxiliary tools yields further gains, particularly in degradation removal: the visual-evidence setting achieves a D-Score of 74.4 and a DF-Score of 57.8. However, giving the executor visual maps directly can introduce artifacts, such as semantic-label colors appearing in the restored image. Expressing the relevant evidence in text avoids this risk and suits the executor’s language-based interface. This setting achieves the strongest one-round result, with a PSNR of 26.01 dB and a DF-Score of 59.6. Overall, the results suggest that richer requests, image-specific analysis, auxiliary evidence, and evidence-grounded instruction writing each contribute to better restoration and content preservation.

Table 4. Ablation of executor inputs on Real-Paired-200. All variants use one execution round, without re-execution to isolate the efect of how information is supplied to the executor. The bold row uses the executor-input setting adopted by HarnessIR. Metrics follow Tab. 1; the best and second-best results in each column are marked in red and blue.
<table><tr><td>Executor-input configuration</td><td colspan="4">Full-reference Metrics</td><td colspan="5">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td></td><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td>DISTS↓</td><td>∆MANIQA↓</td><td>∆CLIP-IQA↓</td><td>ΔMUSIQ↓</td><td>ΔTOPIQ↓</td><td>∆AFINE-NR↓</td><td>D-Score↑ F-Score↑</td><td></td><td>DF-Score↑</td></tr><tr><td>Direct MFM (Baseline)</td><td>23.94</td><td>0.729</td><td>0.309</td><td>0.181</td><td>0.058</td><td>0.077</td><td>11.91</td><td>0.118</td><td>0.074</td><td>67.9</td><td>47.8</td><td>52.9</td></tr><tr><td>Detailed User Request</td><td>24.75</td><td>0.733</td><td>0.298</td><td>0.177</td><td>0.054</td><td>0.069</td><td>10.88</td><td>0.105</td><td>0.065</td><td>71.4</td><td>50.9</td><td>54.5</td></tr><tr><td>Single-VLM Instruction</td><td>25.49</td><td>0.761</td><td>0.263</td><td>0.161</td><td>0.042</td><td>0.044</td><td>9.30</td><td>0.080</td><td>0.057</td><td>72.5</td><td>54.2</td><td>56.6</td></tr><tr><td>Two VLMs + Tools (Visual Evidence)</td><td>25.62</td><td>0.758</td><td>0.267</td><td>0.161</td><td>0.026</td><td>0.032</td><td>7.39</td><td>0.057</td><td>0.049</td><td>74.4</td><td>55.3</td><td>57.8</td></tr><tr><td>Two VLMs + Tools (Text Evidence)</td><td>26.01</td><td>0.763</td><td>0.261</td><td>0.158</td><td>0.032</td><td>0.024</td><td>7.72</td><td>0.062</td><td>0.045</td><td>73.0</td><td>58.5</td><td>59.6</td></tr><tr><td>GT</td><td></td><td></td><td></td><td></td><td>0.560</td><td>0.451</td><td>54.36</td><td>0.417</td><td>-0.901</td><td></td><td></td><td></td></tr></table>

Table 5. Ablation of refinement strategies on Real-Paired-200. We compare three settings: HarnessIR (Single-pass), which performs a single MFM execution without re-execution; HarnessIR (In-place refinement), which applies a revised instruction to the output of the previous round; and HarnessIR (Redo from LQ) which applies a revised instruction to original LQ input. Metrics follow Tab. 1, the best and second-best results are marked in red and blue.
<table><tr><td>Refinement strategy</td><td colspan="4">Full-reference Metrics</td><td colspan="5">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓ DISTS↓</td><td></td><td>∆MANIQA↓</td><td>∆CLIP-IQA↓</td><td>ΔMUSIQ↓</td><td>∆TOPIQ↓</td><td>ΔAFINE-NR↓</td><td>D-Score↑ F-Score↑</td><td></td><td>DF-Score↑</td></tr><tr><td>HarnessIR (Single-pass)</td><td>26.01</td><td>0.763</td><td>0.261</td><td>0.158</td><td>0.032</td><td>0.024</td><td>7.72</td><td>0.062</td><td>0.045</td><td>73.0</td><td>58.5</td><td>59.6</td></tr><tr><td>HarnessIR (In-place refinement)</td><td>25.64</td><td>0.760</td><td>0.265</td><td>0.159</td><td>0.020</td><td>0.016</td><td>6.79</td><td>0.052</td><td>0.032</td><td>71.8</td><td>55.6</td><td>57.2</td></tr><tr><td>HarnessIR (Redo from LQ)</td><td>26.63</td><td>0.775</td><td>0.246</td><td>0.149</td><td>0.020</td><td>0.001</td><td>5.52</td><td>0.028</td><td>0.033</td><td>75.2</td><td>63.6</td><td>62.0</td></tr><tr><td>GT</td><td></td><td></td><td></td><td></td><td>0.560</td><td>0.451</td><td>54.36</td><td>0.417</td><td>-0.901</td><td></td><td></td><td></td></tr></table>

These gains require only a modest increase in inference cost. As shown in Tab. 7, adding one VLM call raises the cost from \$0.070 for direct MFM usage to \$0.077. Using two VLM calls and auxiliary tools costs \$0.085. Thus, the improvements are achieved without a prohibitively expensive control loop, ofering a favorable trade-of between restoration quality and cost.

## 4.4.2 Redo vs. In-place Repair During Re-execution

We compare two refinement strategies with the single-pass version in Tab. 5. HarnessIR (Single-pass) performs one execution without re-execution and serves as the reference setting. HarnessIR (In-place refinement) feeds the output of the previous execution back to the executor for further correction, following the sequential refinement strategy of previous agent IR works [15, 30]. In contrast, HarnessIR (Redofrom LQ) returns to the original LQ image when refinement is triggered, composes a revised instruction, and generates a new candidate from scratch, which corresponds to the final HarnessIR design described in Sec. 3.4.

The results support restarting from the original input rather than refining an already altered image. In-place refinement is inferior even to the single-pass setting, with the DF-Score decreasing from 59.6 to 57.2. In contrast, redo from the LQ input improves PSNR from 26.01 to 26.63 dB and raises the D-Score/F-Score from 73.0/58.5 to 75.2/63.6, resulting in a DF-Score of 62.0. This suggests that an unsuccessful intermediate result may contain hallucinated or distorted content that should not be propagated into the next round. Restarting from the LQ input allows the revised instruction to correct the failure without accumulating errors. Importantly, this improvement is not uniform across all samples: only 37% of the 200 test images undergo re-execution, and the gains come from this subset. Images that already produce satisfactory results in the first round remain unchanged. Thus, redo primarily repairs dificult tail cases rather than improving every image uniformly. Detailed statistics on the re-execution frequency and its associated cost are provided in Sec. 4.5.

## 4.4.3 VLM Choice

We compare diferent VLMs for HarnessIR’s diagnoser and composer on Real-Paired-200. Verification-driven re-execution is disabled in all variants, allowing us to compare the VLMs without gains from additional execution rounds.

As shown in Tab. 6, the results suggest that HarnessIR does not depend on a specific VLM, provided the model is suficiently capable. The locally deployed Qwen3-VL-32B performs somewhat worse, reaching a DF-Score of 57.5. GPT-5.6-Luna and Gemini-3.7-Flash perform similarly, with DF-Scores of 59.1 and 59.6,

Table 6. Efect of the VLMs used for diagnoser and composer on Real-Paired-200. Verification-driven re-execution is disabled in all variants. Metrics follow Tab. 1; the best and second-best results in each column are marked in red and blue.
<table><tr><td>VLM</td><td colspan="4">Full-reference Metrics</td><td colspan="5">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td></td><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td>DISTS↓</td><td>∆MANIQA↓ ∆CLIP-IQA↓ ∆MUSIQ↓</td><td></td><td></td><td>ΔTOPIQ↓</td><td>∆AFINE-NR↓</td><td>D-Score↑ F-Score↑</td><td></td><td>DF-Score↑</td></tr><tr><td>HarnessIR (Qwen3-VL-32B [56])</td><td>25.68</td><td>0.760</td><td>0.267</td><td>0.161</td><td>0.034</td><td>0.024</td><td>7.82</td><td>0.067</td><td>0.050</td><td>71.2</td><td>56.1</td><td>57.5</td></tr><tr><td>HarnessIR (GPT-5.6-Sol [57])</td><td>26.18</td><td>0.768</td><td>0.253</td><td>0.151</td><td>0.029</td><td>0.017</td><td>6.48</td><td>0.058</td><td>0.048</td><td>74.4</td><td>59.7</td><td>60.5</td></tr><tr><td>HarnessIR (GPT-5.6-Luna [58])</td><td>26.05</td><td>0.762</td><td>0.259</td><td>0.155</td><td>0.034</td><td>0.027</td><td>7.78</td><td>0.066</td><td>0.059</td><td>72.7</td><td>58.7</td><td>59.1</td></tr><tr><td>HarnessIR (Gemini-3.7-Flash [44])</td><td>26.01</td><td>0.763</td><td>0.261</td><td>0.158</td><td>0.032</td><td>0.024</td><td>7.72</td><td>0.062</td><td>0.045</td><td>73.0</td><td>58.5</td><td>59.6</td></tr><tr><td>GT</td><td></td><td></td><td></td><td></td><td>0.560</td><td>0.451</td><td>54.36</td><td>0.417</td><td>-0.901</td><td></td><td></td><td></td></tr></table>

Table 7. Average per-image runtime, call counts, restoration performance, and cost on Real-Paired-200. VLM and MFM denote API calls. Tool / Perception: Tool reports local model used as restoration or auxiliary tools, perception reports local VLM model used for perception.
<table><tr><td>Method</td><td>VLM</td><td>MFM</td><td>Tool / Perception</td><td>Time (s)</td><td>PSNR↑</td><td>DF-Score↑</td><td>Cost ($)</td></tr><tr><td>AgenticIR</td><td>0.70</td><td>一</td><td>5.94 / 9.87</td><td>88.8</td><td>22.41</td><td>38.8</td><td>0.137</td></tr><tr><td>4KAgent</td><td>0.53</td><td>一</td><td>7.47 / 10.08</td><td>125.3</td><td>22.18</td><td>37.4</td><td>0.190</td></tr><tr><td>Baseline (MFM only)</td><td></td><td>1.00</td><td>一</td><td>14.7</td><td>23.94</td><td>52.9</td><td>0.070</td></tr><tr><td>HarnessIR (One VLM call)</td><td>1.00</td><td>1.00</td><td>一</td><td>27.2</td><td>25.49</td><td>56.6</td><td>0.077</td></tr><tr><td>HarnessIR (Two VLM calls + Tools)</td><td>2.00</td><td>1.00</td><td>2.40/ -</td><td>41.8</td><td>26.01</td><td>59.6</td><td>0.085</td></tr><tr><td>HarnessIR (Two VLM calls + Tools, GPT-5.6-Sol)</td><td>2.00</td><td>1.00</td><td>2.40 / -</td><td>45.5</td><td>26.18</td><td>60.5</td><td>0.183</td></tr><tr><td>HarnessIR (Final, Three VLM calls + Tools + redo)</td><td>3.90</td><td>1.45</td><td>2.40/ -</td><td>64.3</td><td>26.63</td><td>62.0</td><td>0.129</td></tr></table>

respectively. GPT-5.6-Sol achieves the highest PSNR and DF-scores, but its gains over Gemini-3.7-Flash are modest: 0.17 dB in PSNR and 0.9 points in DF-Score. We therefore use Gemini-3.7-Flash in the final system: its restoration performance is comparable, while GPT-5.6-Sol incurs substantially higher API cost (see Sec. 4.5).

## 4.5 Cost Analysis

As shown in Tab. 7, we report the per-image runtime, call counts, restoration performance, and cost of diferent HarnessIR variants and two previous agentic IR methods [15, 16] on Real-Paired-200. Runtime and VLM call counts come from actual runs. For local computation, including the prior methods’ functional modules and HarnessIR’s auxiliary tools, we convert measured runtime into cost using on-demand A100 GPU rental rates; API costs are estimated from observed token usage and public list prices. We exclude costs that are dificult to assign to individual images, such as parallel waiting, environment setup, and deployment overhead. Rental rates provide the fairest method-independent basis for comparison, whereas GPU purchase costs depend on assumptions about depreciation, utilization, service lifetime, and allocation, making per-image costs ill-defined and incomparable. The resulting values should therefore be viewed as idealized references for computational and monetary eficiency. We discuss the results first and detail the calculations afterward.

Results Analysis. Compared with the two previous agentic IR methods, the final HarnessIR system achieves a higher DF-Score on Real-Paired-200 at a lower estimated per-image cost: \$0.129, compared with \$0.137 for AgenticIR and \$0.190 for 4KAgent. For the HarnessIR variants, starting from direct MFM usage, adding a single VLM call raises the DF-Score from 52.9 to 56.6 and the PSNR from 23.94 to 25.49 dB, while increasing the cost only from \$0.070 to \$0.077 per image. This one-call variant requires no local GPU computation, making it a particularly attractive option when simplicity and deployment cost are priorities. Adding a second VLM call and local auxiliary tools raises the cost modestly to \$0.085, while further improving the DF-Score to 59.6 and the PSNR to 26.01 dB. Thus, the diagnosis-and-composition pipeline provides a favorable additional quality gain for a small increase in cost.

The complete HarnessIR system adds verification and selective re-execution, reaching a DF-Score of 62.0 and a PSNR of 26.63 dB at an average cost of \$0.129 per image. This average gain is driven by the subset of dificult cases that trigger re-execution rather than by uniformly improving every image: 37% of images undergo at least one redo, and the verifier retains the better candidate. The increased cost, therefore, supports targeted correction of cases that need it. Finally, replacing Gemini-3.7-Flash with GPT-5.6-Sol in the two-VLM configuration increases the cost from \$0.085 to \$0.183 per image. Although GPT-5.6-Sol yields some performance improvement, the gain is modest relative to the more than twofold cost increase. We therefore use Gemini-3.7-Flash in the final system to balance performance and cost.

![](images/5e3341b37fa2e2ec08412f5f690745d454c838ad74f298142fc4a423824e7749.jpg)  
Figure 5. Human selection rates among our methods and baselines.

Calculation Details. The GPU price is the mean on-demand A100 rate of the three major cloud providers, Google Cloud <sup>1</sup>, Microsoft Azure <sup>2</sup>, and Amazon Web Services<sup>3</sup>, normalized to a per-GPU hourly rate: \$2.671 per GPU-hour. All locally executed calls, including local restoration, perception, and auxiliary-tool calls, are converted to GPU-rental cost using this rate. Following the deployment setup of 4KAgent, both AgenticIR and 4KAgent use two GPUs: one GPU is dedicated to DepictQA, while the other runs the largest model so that it remains within the available GPU memory. For AgenticIR and 4KAgent, call counts and runtimes are averaged over Real-Paired-200.

We calculate API costs using public list prices, retrieved in September 2026, without cached-input or batch discounts.<sup>4</sup> The relevant prices are reported as input/output rates per million tokens: Nano Banana 2 uses in/out rates of \$0.50/\$60 for image output and \$0.50/\$3 for text output; Gemini-3.7-Flash uses an in/out rate of \$0.75/\$3.75; and GPT-5.6-Sol uses an in/out rate of \$4/\$20. Thus, GPT-5.6-Sol is more than five times as expensive as Gemini-3.7-Flash for both input and output tokens. For HarnessIR, API costs are calculated from the actual input and output usage of the VLM and MFM calls, while the cost of local auxiliary tools is calculated through GPU-rental time. Under HarnessIR-NB2, one Gemini-3.7-Flash VLM call costs approximately \$0.007 (diferent under diferent token counts), one MFM call costs approximately \$0.070, and one complete auxiliary-tool suite costs approximately \$0.0048. In the final system, every image undergoes diagnosis, composition, execution, and verification. About 29% of the 200 images trigger one redo and 8% trigger two redos; thus, 37% of the 200 images trigger at least one redo, corresponding to an average of 0.45 redo rounds per image. Each redo adds two VLM calls and one MFM call, yielding 3.90 VLM calls and 1.45 MFM calls per image on average. The costs reported in Tab. 7 are computed from unrounded per-image token usage and runtime measurements.

## 4.6 User Study

We conduct a user study to compare restoration quality on real images. For each test case, annotators see one low-quality (LQ) image together with the eight restorations of that image. The eight methods are the same set used in the DF-Score comparison: PromptIR [10], InstructIR [27], 4KAgent [16], AgenticIR [15], GPT-Image 2.5 [18], Nano Banana 2 [17], HarnessIR-GPT and HarnessIR-NB2. The eight results are shown in random order. Method names and metric values are withheld as well. Annotators are required to select the single restoration they judge to be the best. The instruction asks them to prefer a result that removes the target degradation, preserves the main scene content, and avoids obvious artifacts.

We draw 100 images at random from the Real-Paired-200 and 100 images at random from Real-NoGT-200, giving 200 groups in total. Each of the 25 participants annotates all 200 groups, so each group receives 25 independent votes and the study contains 25 × 200 = 5000 votes.

The selection rate of a method is the percentage of the 5000 votes in which annotators choose that method as the best restoration. Fig. 5 shows a clear preference for the harnessed restorations. HarnessIR-NB2 receives 48.3% of the votes and HarnessIR-GPT receives 28.1%, so the two HarnessIR variants together account for 76.4%. These human preferences are consistent with the quantitative results and qualitative comparisons presented above. They further confirm that HarnessIR substantially improves restoration quality while maintaining high perceptual quality and faithfully preserving the original content of the LQ images.

## 5 Conclusion

We presented HarnessIR, an agentic framework that used MFMs as the executor for universal real-world IR. By combining image-specific diagnosis, auxiliary tools, prompt composition, and verification-driven re-execution, HarnessIR harnessed the MFM to remove degradations without altering image content. Across synthetic and challenging real-world benchmarks, HarnessIR consistently improved restoration performance over the direct use of MFM, achieving particularly large gains on real-world images. Our ablations further showed that these improvements came from the coordinated design of the harness at an acceptable cost. Overall, HarnessIR demonstrated that a carefully designed harness can substantially unlock the restoration potential of proper MFMs and turn them into more efective and controllable universal real-world restorers.

Limitations. HarnessIR requires access to online VLM and MFM APIs, which is a common requirement by existing agentic IR methods such as AgenticIR [15] and 4KAgent [16] (using GPT-4o). Moreover, an agentic harness can only elicit and regulate the capabilities of its executor but cannot raise the executor’s performance ceiling. In the future, with this general IR agent, we can curate diverse and challenging restoration data and generate verification-guided training supervision to improve the capabilities of local and smaller models.

## References

[1] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017.

[2] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Eficient transformer for high-resolution image restoration. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022.

[3] Sung-Jin Cho, Seo-Won Ji, Jun-Pyo Hong, Seung-Won Jung, and Sung-Jea Ko. Rethinking coarse-to-fine approach in single image deblurring. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4641–4650, 2021.

[4] Liangyu Chen, Xin Lu, Jie Zhang, Xiaojie Chu, and Chengpeng Chen. Hinet: Half instance normalization network for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 182–192, 2021.

[5] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13, pages 184–199. Springer, 2014.

[6] Xiangyu Chen, Xintao Wang, Wenlong Zhang, Xiangtao Kong, Yu Qiao, Jiantao Zhou, and Chao Dong. Hat: Hybrid attention transformer for image restoration. arXiv preprint arXiv:2309.05239, 2023.

[7] Xiaohong Liu, Yongrui Ma, Zhihao Shi, and Jun Chen. Griddehazenet: Attention-based multi-scale network for image dehazing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7314–7323, 2019.

[8] Yuda Song, Zhuqing He, Hui Qian, and Xin Du. Vision transformers for single image dehazing. arXiv preprint arXiv:2204.03883, 2022.

[9] Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, and Xi Peng. All-In-One Image Restoration for Unknown Corruption. In IEEE Conference on Computer Vision and Pattern Recognition, New Orleans, LA, June 2022.

[10] Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan. Promptir: Prompting for all-in-one blind image restoration. Advances in Neural Information Processing Systems (NeurIPS), 2023.

[11] Xiangtao Kong, Chao Dong, and Lei Zhang. Towards efective multiple-in-one image restoration: A sequential and prompt learning strategy. arXiv preprint arXiv:2401.03379, 2024.

[12] Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjölund, and Thomas B Schön. Photo-realistic image restoration in the wild with controlled vision-language models. arXiv preprint arXiv:2404.09732, 2024.

[13] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[14] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

[15] Kaiwen Zhu, Jinjin Gu, Zhiyuan You, Yu Qiao, and Chao Dong. An intelligent agentic system for complex image restoration problems. In International Conference on Learning Representations, volume 2025, pages 57985–58013, 2025.

[16] Yushen Zuo, Qi Zheng, Mingyang Wu, Xinrui Jiang, Renjie Li, Jian Wang, Yide Zhang, Gengchen Mai, Lihong V Wang, James Zou, et al. 4kagent: agentic any image to 4k super-resolution. arXiv preprint arXiv:2507.07105, 2025.

[17] Google. Nano banana 2 (gemini-3.1-flash-image). https://ai.google.dev/gemini-api/docs/models/gemini-3. 1-flash-image, 2026.

[18] OpenAI. Gpt-image-2.5. https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst, 2026.

[19] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

[20] Weixiong Sun, Xiang Yin, and Chao Dong. Can nano banana 2 replace traditional image restoration models? an evaluation of its performance on image restoration tasks. arXiv preprint arXiv:2604.03061, 2026.

[21] Yufeng Yang, Xianfang Zeng, Zhangqi Jiang, Fukun Yin, Jianzhuang Liu, Wei Cheng, Shiyu Liu, Yuqi Peng, Gang YU, Shifeng Chen, et al. Realrestorer: Towards generalizable real-world image restoration with large-scale image editing models. arXiv preprint arXiv:2603.25502, 2026.

[22] Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou, et al. Pp-ocrv6: From 1.5 m to 34.5 m parameters, surpassing billion-scale vlms on ocr tasks. arXiv preprint arXiv:2606.13108, 2026.

[23] Guo Jia, Deng Jiankang, An Xiang, Yu Jack, and Gecer Baris. Insightface: 2d and 3d face analysis project. https://github.com/deepinsight/insightface, 2026.

[24] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in neural information processing systems, 37:21875–21911, 2024.

[25] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pages 138846–138923, 2026.

[26] Yitong Jiang, Zhaoyang Zhang, Tianfan Xue, and Jinwei Gu. Autodir: Automatic all-in-one image restoration with latent difusion. In European Conference on Computer Vision, pages 340–359. Springer, 2024.

[27] Marcos V Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In European Conference on Computer Vision, pages 1–21. Springer, 2024.

[28] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3086–3095, 2019.

[29] Haoyu Chen, Wenbo Li, Jinjin Gu, Jingjing Ren, Sixiang Chen, Tian Ye, Renjing Pei, Kaiwen Zhou, Fenglong Song, and Lei Zhu. Restoreagent: Autonomous image restoration agent via multimodal large language models. Advances in Neural Information Processing Systems, 37:110643–110666, 2024.

[30] Xu Jiang, Gehui Li, Bin Chen, and Jian Zhang. Multi-agent image restoration. International Journal ofComputer Vision, 134(5):205, 2026.

[31] Yingjie Zhou, Jiezhang Cao, Farong Wen, Zicheng Zhang, Yu Zhou, Yue Shi, Xiaohong Liu, Radu Timofte, Luc Van Gool, and Guangtao Zhai. Q-agent: Quality-driven chain-of-thought image restoration agent through robust multimodal large language model. arXiv preprint arXiv:2504.07148, 2025.

[32] Bingchen Li, Xin Li, Yiting Lu, and Zhibo Chen. Hybrid agents for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22636–22647, 2026.

[33] Jianglin Lu, Yuanwei Wu, Ziyi Zhao, Hongcheng Wang, Felix Jimenez, Abrar Majeedi, and Yun Fu. Restorer1: Eficient image restoration agents via reinforcement learning with multimodal llm perceptual feedback. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8629–8639, 2026.

[34] Guoli Jia, Yisheng Zhang, Haote Hu, Shanxu Zhao, Kaikai Zhao, Long Sun, Xinwei Long, Kai Tian, Che Jiang, Zhaoxiang Liu, et al. Tir-agent: Training an explorative and eficient agent for image restoration. In European Conference on Computer Vision, pages 523–540. Springer, 2026.

[35] Yunlong Lin, Zixu Lin, Haoyu Chen, Panwang Pan, Chenxin Li, Sixiang Chen, Kairun Wen, Yeying Jin, Wenbo Li, and Xinghao Ding. Jarvisir: Elevating autonomous driving perception with intelligent image restoration. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22369–22380. IEEE, 2025.

[36] Feng Zhu, Shuyang Xie, Yihan Zeng, Ming Liu, and Wangmeng Zuo. Opera: An agent for image restoration with end-to-end joint planning-execution optimization. arXiv preprint arXiv:2605.22104, 2026.

[37] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[38] John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

[39] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[40] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

[41] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

[42] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[43] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In International Conference on Learning Representations, volume 2025, pages 65882–65919, 2025.

[44] Google. Gemini 3.7 flash (gemini-3.7-flash). https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash, 2026.

[45] Shuang Cui, Fan Ji, Guanglong Sun, Yufei Guo, Xiongxin Tang, Jiangmeng Li, and Fanjiang Xu. Self-evolving agentic image restoration via deliberate planning and intuitive execution. arXiv preprint arXiv:2606.28971, 2026.

[46] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[48] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020.

[49] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1191–1200, 2022.

[50] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 2555–2563, 2023.

[51] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5148–5157, 2021.

[52] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2024.

[53] Du Chen, Tianhe Wu, Kede Ma, and Lei Zhang. Toward generalized image quality assessment: Relaxing the perfect reference quality assumption. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 12742–12752, 2025.

[54] Lingchen Sun, Rongyuan Wu, Xiangtao Kong, Jixin Zhao, Qiaosi Yi, Yujing Sun, Shuaizheng Liu, Zhengqiang Zhang, and Lei Zhang. Pixrestore: Unified image restoration via pixel difusion transformer. arXiv preprint arXiv:2608.16793, 2026.

[55] Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. Advances in Neural Information Processing Systems, 38:88167–88190, 2026.

[56] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[57] OpenAI. Gpt-5.6 sol. https://developers.openai.com/api/docs/models/gpt-5.6-sol, 2026.

[58] OpenAI. Gpt-5.6 luna. https://developers.openai.com/api/docs/models/gpt-5.6-luna, 2026.

[59] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pages 101–117. Springer, 2020.

[60] Cosmin Ancuti, Codruta O Ancuti, Radu Timofte, and Christophe De Vleeschouwer. I-haze: A dehazing benchmark with real hazy and haze-free indoor images. In International conference on advanced conceptsfor intelligent vision systems, pages 620–631. Springer, 2018.

[61] Codruta O Ancuti, Cosmin Ancuti, and Radu Timofte. Nh-haze: An image dehazing benchmark with nonhomogeneous hazy and haze-free images. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1798–1805. IEEE, 2020.

[62] Wei Li, Qiming Zhang, Jing Zhang, Zhen Huang, Xinmei Tian, and Dacheng Tao. Toward real-world single image deraining: A new benchmark and beyond. arXiv preprint arXiv:2206.05514, 2022.

[63] Ruijie Quan, Xin Yu, Yuanzhi Liang, and Yi Yang. Removing raindrops and rain streaks in one go. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9147–9156, 2021.

[64] Jaesung Rim, Haeyun Lee, Jucheol Won, and Sunghyun Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In European conference on computer vision, pages 184–201. Springer, 2020.

[65] Abdelrahman Abdelhamed, Stephen Lin, and Michael S Brown. A high-quality denoising dataset for smartphone cameras. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1692–1700, 2018.

[66] J Xu, H Li, Z Liang, D Zhang, and L Zhang. Real-world noisy image denoising: A new benchmark. arxiv 2018. arXiv preprint arXiv:1804.02603, 2018.

[67] Jie Liang, Radu Timofte, Qiaosi Yi, Zhengqiang Zhang, Shuaizheng Liu, Lingchen Sun, Rongyuan Wu, Xindong Zhang, Hui Zeng, and Lei Zhang. Ntire 2025 the 2nd restore any image model (raim) in the wild challenge, 2025. URL https://arxiv.org/abs/2506.01394.

[68] Junyang Chen, Jinshan Pan, and Jiangxin Dong. FaithDif: Unleashing difusion priors for faithful image super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, 2025.

[69] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[70] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Acdc: The adverse conditions dataset with correspondences for semantic driving scene understanding. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10745–10755. IEEE, 2021.

[71] Boyi Li, Wenqi Ren, Dengpan Fu, Dacheng Tao, Dan Feng, Wenjun Zeng, and Zhangyang Wang. Benchmarking single-image dehazing and beyond. IEEE Transactions on Image Processing, 28(1):492–505, 2019.

[72] Beibei Lin, Yeying Jin, Yan Wending, Wei Ye, Yuan Yuan, and Robby T Tan. Nighthaze: Nighttime image dehazing via self-prior learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 39, 2025.

[73] Qiyuan Guan, Xiang Chen, Guiyue Jin, Jiyu Jin, Shumin Fan, Tianyu Song, and Jinshan Pan. Rethinking nighttime image deraining via learnable color space transformation. Advances in Neural Information Processing Systems, 38: 3189–3225, 2026.

[74] Jie Chen, Cheen-Hau Tan, Junhui Hou, Lap-Pui Chau, and He Li. Robust video content alignment and compensation for rain removal in a cnn framework. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6341–6349, 2018. doi: 10.1109/CVPR.2018.00658.

[75] Yun-Fu Liu, Da-Wei Jaw, Shih-Chia Huang, and Jenq-Neng Hwang. Desnownet: Context-aware deep network for snow removal. IEEE Transactions on Image Processing, 27(6):3064–3073, 2018.

[76] Yuen Peng Loh and Chee Seng Chan. Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding, 178:30–42, 2019. doi: https://doi.org/10.1016/j.cviu.2018.10.010.

[77] Anthropic. Claude opus 5. https://www.anthropic.com/claude, 2026.

## Appendix

In this appendix, we provide the following materials:

• A. Input Request and A Complete Case Study: The fixed request template for each degradation category to represent the usage of general users and a real example processed by HarnessIR.

• B. Real Test Set Composition: Detailed construction and data sources of Real-Paired-200 and Real-NoGT-200.

• C. DF-Score Evaluation Protocol: The alignment with human judgments, and evaluator-VLM replacement experiments.

• D. Detailed Results on MiO100: Per-group results for MiO100 Groups A, B, and C.

• E. More Visual Results: Additional qualitative comparisons between HarnessIR and other representative methods.

## A Input Request and A Complete Case Study

## A.1 Input Request

We use short, fixed requests to represent the intent of a general user. The request for mix serves as the default. When one or more degradation types are specified, we prepend a removal instruction to the same default request. For mixed degradations, the names of all applicable degradations are joined with “and” or followed in the end. For example, an image containing both haze and rain needs to be brightened is assigned the request “Please remove the haze and rain from the image and restore this low-quality image to a clear, clean, and natural state. Meanwhile, recover its normal brightness”. The basic requests are as follows:

```markdown
### [[mix]] ###
Restore this low-quality image to a clear, clean and natural state.
### [[haze]] ###
Please remove the haze from the image and restore this low-quality image to a clear, clean and natural state.
### [[rain]] ###
Please remove the rain from the image and restore this low-quality image to a clear, clean and natural state.
### [[snow]] ###
Please remove the snow from the image and restore this low-quality image to a clear, clean and natural state.
### [[lowlight]] ###
Restore this low-quality image to a clear, clean and natural state and recover its normal brightness.
```

## A.2 A Complete Case Study

To make the workflow concrete, Fig. A.1 presents a real example processed by HarnessIR. Below, we provide the corresponding outputs at each stage: the initial diagnosis, the evidence returned by the selected auxiliary tools, the instruction composed for the MFM executor, and the verifier’s decision on whether to re-execute. This example illustrates how image-specific observations are translated into restoration requirements and how the resulting image is checked against the original input.

Output of Perception and Diagnosis. First, the input to HarnessIR contains the LQ image (left part of Fig. A.1) and the user request “Restore this low-quality image to a clear, clean and natural state." The output of the diagnoser is as follows:

"triage": {   
"content": "A bright blue bird holding a seed in its beak, standing on a dark feeder. The foreground   
↩→ contains scattered yellow corn kernels and a white textured block, both heavily out of focus. The   
↩→ background is a smooth, uniform green, indicating shallow depth of field in daylight.",   
"degradations": [   
{   
"type": "noise",

```jsonl
"severity": "moderate",
"spatial": "global",
"location": "whole image",
"worst_at": "green background and dark blue shadows on the bird",
"mildest_at": "bright white areas in foreground",
"evidence": "grainy texture visible across smooth surfaces"
},
{
"type": "softness",
"severity": "moderate",
"spatial": "local",
"location": "midground (bird)",
"worst_at": "bird’s body feathers",
"mildest_at": "bird’s eye and beak",
"evidence": "lack of fine feather detail despite being the focal point"
},
{
"type": "defocus-blur",
"severity": "severe",
"spatial": "local",
"location": "foreground and background",
"worst_at": "background green and foreground white block",
"mildest_at": "midground bird",
"evidence": "optical shallow depth of field blurring non-focal planes"
},
{
"type": "jpeg",
"severity": "mild",
"spatial": "global",
"location": "whole image",
"worst_at": "edges of the bird against the green background",
"mildest_at": "uniform background areas",
"evidence": "slight blockiness and ringing around high-contrast boundaries"
}
],
"sensitive": {
"has_text": true,
"has_faces": false,
"has_layered_depth": true,
"has_complex_semantics": true,
"notes": "Fine feather textures on the bird."
},
"tool_plan": [
{
"tool": "T1_OCR",
"reason": "The watermark text in the bottom right needs to be preserved and not distorted by
↩→ restoration."
},
{
"tool": "T3_DEPTH",
"reason": "The image relies heavily on shallow depth of field; a depth map will help ensure the
↩→ foreground and background remain soft while the midground bird is sharpened."
},
{
"tool": "T4_SEGMENTATION",
"reason": "Separating the bird from the background will allow for targeted detail enhancement on the
↩→ feathers without amplifying noise in the smooth green bokeh."
}
],
"difficulty_hint": "medium",
"difficulty_reason": "Recovering fine feather detail on the slightly soft bird without over-sharpening
↩→ the intentionally blurred foreground and background requires careful spatial control."
}
```

The diagnosis identifies text, layered depth, and semantically distinct regions in the image. Therefore, it requests OCR, depth estimation, and semantic segmentation tools to gather image-specific evidence for restoration.

Output of On-Demand Tool Call. The selected auxiliary tools return the OCR transcription, depth map, and semantic segmentation map shown in Fig. A.1. The OCR result is as follows:

"tool\_evidence": {   
"ocr": {   
"engine": "ppocr-v6",   
"n\_texts": 1,   
"texts": [

Round 2 Output

![](images/3fa5b9c91e4be1c775f6ff0d8ea3e28106ee7c7f387d78345f4bc29b195d3127.jpg)  
Round 1 Output

Figure A.1. A complete case study illustrating the workflow of HarnessIR, including the input image, depth map and semantic map from auxiliary tools, and the round1 and round2 output images.  
```json
{
"string": "© Sweet Dre",
"confidence": 0.9435150027275085,
"bbox_norm": [
0.839,
0.932,
1.0,
0.97]}]}
```  
In this example, there is no face, so there is no output related to face detection tool.

Output of Prompt Composition. The composer then converts the diagnosis and tool evidence into a restoration instruction for the MFM, as shown below. In particular, it uses the depth information to identify the bird and feeder as the intended focal plane, while recognizing the blur in the foreground and background as a natural shallow-depth-of-field efect that should be preserved rather than restored. The prompt also describes the scene and the bird’s recoverable details, and records the text in the lower-right corner so that it can be preserved unchanged.

You are an image restoration expert. Restore this photograph.\n\nThe image shows a bright blue bird holding a ↩→ seed in its beak, standing on a dark feeder. The foreground contains out-of-focus corn kernels and a ↩→ suet block, set against a smooth, out-of-focus green background.\n\nFidelity is your first priority. ↩→ Preserve the exact scene content, geometry, object positions, and structure. Do not add, remove, or ↩→ alter any objects. Maintain the current global color, tone, and white balance exactly as they are; do ↩→ not shift hues or recolor the image.\n\nCleanly remove sensor noise, grain, and JPEG compression ↩→ artifacts (such as blockiness and ringing around high-contrast edges) across the entire image. Ensure ↩→ no over-smoothing erases genuine textures, and avoid introducing over-sharpening halos, ghosting, or ↩→ visible seams.\n\nThe bird and the dark feeder it stands on represent the intended focal plane but ↩→ suffer from capture softness. Genuinely deblur this midground region. Recover fine, crisp detail in ↩→ the bird’s feathers, eye, beak, and the seed it holds. The bird must be clearly resolved and textured ↩→ without looking artificially sharpened.\n\nThe shallow depth of field is a physical property of this ↩→ photograph. The out-of-focus yellow corn and white block in the foreground, as well as the smooth ↩→ green background, must remain exactly as out of focus as they currently are. Do not attempt to ↩→ sharpen or deblur these areas; simply denoise them so the optical bokeh is clean and

↩→ smooth.\n\nFinally, in the bottom right corner, preserve the watermark text exactly as it appears. ↩→ The characters \"© Sweet Dre\" must remain legible and unchanged; do not invent, guess, or alter any ↩→ letters.

Output of Verification and Re-Execution. After execution, the generated image is passed to the verifier, whose response indicates that re-execution is necessary. As shown in the first-round result (see Fig. A.1), one readily identifiable issue is that the bird’s left claw should be occluded in the input image, but it is incorrectly generated in the restoration result. The Verifier detects this error, and the revised instruction explicitly emphasizes that the occluded claw should not be reconstructed. Correspondingly, the second-round output of Fig. A.1 no longer exhibits this problem. In the next round of verification, this result shows that there is no need to redo it.

{   
"reasoning": "The input suffers from moderate blur, noise, and compression artifacts, obscuring fine   
↩→ details like the bird’s feathers. The attempt aggressively denoises and sharpens the image, but goes   
↩→ too far in adding detail. Tracing back to Picture 1, the left claw of the bird in the picture should   
↩→ have been obscured, but now the error has occurred and it has been placed in front of the dark   
↩→ feeder. The crisp, distinct feathers on the bird’s back and breast in the attempt are not supported   
↩→ by the blurry input; they are fabricated. Additionally, there is an inconsistent, smooth, blurry   
↩→ patch on the bird’s lower left flank where the model failed to generate texture, contrasting sharply   
↩→ with the hallucinated details elsewhere. The decisive issue is the hallucination of high-frequency   
↩→ feather details.",   
"checklist": {   
"U1": "fail",   
"U2": "fail",   
"U3": "pass",   
"U4": "na",   
"U5": "pass",   
"U6": "pass",   
"U7": "fail",   
"U8": "pass",   
"U9": "fail",   
"U10": "fail",   
"T1": "pass",   
"T2": "pass",   
"T3": "fail",   
"T4": "fail",   
"T5": "na",   
"T6": "na",   
"T7": "pass",   
"T8": "fail"   
},   
"defects": [   
{   
"location": "bird’s left claw",   
"bbox\_norm": [   
0.45,   
0.67,   
0.69,   
0.90   
],   
"issue": "Compared with the input, it should be obscured.",   
"item": "U1",   
"severity": "moderate"   
},   
{   
"location": "bird’s back and breast",   
"bbox\_norm": [   
0.35,   
0.3,   
0.65,   
0.6   
],   
"issue": "Fabricated feather textures not supported by the blurry input.",   
"item": "U2",   
"severity": "severe"   
},   
{   
"location": "bird’s lower left flank",   
"bbox\_norm": [   
0.31,   
0.53,   
0.41,   
0.68   
],   
"issue": "Inconsistent smooth, blurry patch lacking texture.",

Table B.1. Source datasets of the two real-world test sets, with the number of images contributed by each in parentheses.
<table><tr><td>Degradation</td><td>Source datasets (images)</td><td>Total</td></tr><tr><td colspan="3">Real-Paired-200 200 LQ/GT pairs</td></tr><tr><td>Low resolution</td><td>DRealSR (20) [59], RealSR (20) [28]</td><td>40</td></tr><tr><td>Haze</td><td>I-HAZE (20) [60], NH-HAZE (20) [61]</td><td>40</td></tr><tr><td>Rain</td><td>RealRain-1k-H (20) [62], RainDS-real (20) [63]</td><td>40</td></tr><tr><td>Blur</td><td>RealBlur (20) [64]</td><td>20</td></tr><tr><td>Sensor noise</td><td>SIDD (20) [65], PolyU (20) [66]</td><td>40</td></tr><tr><td>Mixed, in the wild</td><td>RAIM (20) [67]</td><td>20</td></tr><tr><td colspan="3">Real-NoGT-200 — 200 LQ images, no reference</td></tr><tr><td>Mixed, no named weather</td><td>RealDeg (classic film (25),old photo(25) and social media(25)) [68], ImageNet-worst (25) [69]</td><td>100</td></tr><tr><td>Haze</td><td>ACDC-fog (7) [70], JarvisIR-fog (6) [35], RESIDE-RTTS (6) [71], NightHaze (6) [72]</td><td>25</td></tr><tr><td>Rain</td><td>ACDC-rain (7) [70], HQ-NightRain-Real (6) [73], JarvisIR-rain (6) [35], NTURain-real (6) [74]</td><td>25</td></tr><tr><td>Snow</td><td>ACDC-snow (9) [70], JarvisIR-snow (8) [35], Snow100K-realistic (8) [75]</td><td>25</td></tr><tr><td>Low light</td><td>ExDark (9) [76], ACDC-night (8) [70], JarvisIR-night (8) [35]</td><td>25</td></tr></table>

"item": "U9",   
"severity": "moderate"   
}   
],   
"scores": {   
"restoration\_sufficiency": 6,   
"fidelity": 3,   
"scope\_discipline": 3,   
"naturalness": 4   
},   
"overall": 3.0,   
"needs\_redo": true,   
"redo\_priority": 8.0,   
"best\_round": 0,   
"best\_round\_why": "only one attempt",   
},

The revised instruction from the composer is as follows:

"refine": {   
"mode": "global",   
"framing": "defect",   
"new\_instruction": "You are an image restoration expert. Restore this photograph.\n\nThe image shows   
↩→ a bright blue bird holding a seed in its beak, standing on a dark feeder. The foreground contains   
↩→ out-of-focus corn kernels and a suet block, set against a smooth, out-of-focus green   
↩→ background.\n\nFidelity is your first priority. Preserve the exact scene content, geometry, object   
↩→ positions, and structure. Do not add, remove, or alter any objects. Maintain the current global   
↩→ color, tone, and white balance exactly as they are; do not shift hues or recolor the   
↩→ image.\n\nCleanly remove sensor noise, grain, and JPEG compression artifacts (such as blockiness and   
↩→ ringing around high-contrast edges) across the entire image. Ensure no over-smoothing erases genuine   
↩→ textures, and avoid introducing over-sharpening halos, ghosting, or visible seams.\n\nThe bird and   
↩→ the dark feeder it stands on represent the intended focal plane but suffer from capture softness.   
↩→ Genuinely deblur this midground region to recover realistic detail in the bird’s eye, beak, and the   
↩→ seed it holds. However, the left claw of the bird in the picture should have been obscured, don’t try   
↩→ to generate it. You must be extremely careful with the bird’s plumage. Do not invent or hallucinate   
↩→ \"fabricated feather textures not supported by the blurry input\" on the bird’s back and breast. Any   
↩→ texture enhancement must be strictly derived from existing pixel data. Furthermore, the texture   
↩→ recovery must be consistent across the entire body of the bird. Do not leave an \"inconsistent   
↩→ smooth, blurry patch lacking texture\" on the bird’s lower left flank while sharpening the rest of   
↩→ the body. Ensure the level of detail is uniform, balanced, and physically plausible based only on the   
↩→ underlying image data.\n\nThe shallow depth of field is a physical property of this photograph. The   
↩→ out-of-focus yellow corn and white block in the foreground, as well as the smooth green background,   
↩→ must remain exactly as out of focus as they currently are. Do not attempt to sharpen or deblur these   
↩→ areas; simply denoise them so the optical bokeh is clean and smooth.\n\nFinally, in the bottom right   
↩→ corner, preserve the watermark text exactly as it appears. The characters \"© Sweet Dre\" must remain   
↩→ legible and unchanged; do not invent, guess, or alter any letters.",

## B Real Test Set Composition

We construct two real-world test sets from publicly available datasets: Real-Paired-200, which contains 200 paired low-quality and ground-truth images, and Real-NoGT-200, which contains 200 real-world low-quality images without ground-truth. The source datasets and the number of selected images are summarized in Tab. B.1. We randomly sample images from each source dataset; the number selected from each is given in parentheses in Tab. B.1. For consistency across datasets, images whose longer side is shorter than 1024 pixels are isotropically resized so that their longer side becomes 1024 pixels. Images whose longer side exceeds 1024 pixels are center-cropped along the longer dimension until it is 1024 pixels, while the shorter dimension is left unchanged; images whose longer side is already 1024 pixels are left unchanged. For paired samples, the corresponding ground-truth image undergoes the matching resize or center-crop operation so that it has the same final spatial resolution and remains pixel-aligned with the low-quality image.

## C DF-Score Evaluation Protocol

This section describes the evaluation protocol and reliability analysis of DF-Score. First, since the system prompt used by the independent VLM evaluator to assign D-Score and F-Score is several pages long, we provide it in released codes. We then assess the alignment of DF-Score and representative no-reference IQA metrics with human judgments through a pairwise study. Finally, we test whether D-Score, F-Score, and DF-Score remain consistent across VLM evaluators from diferent model families.

Alignment with Human Judgments. We evaluate how well DF-Score and commonly used NR-IQA metrics agree with human judgments of restoration quality. We construct 300 comparison pairs from the Real-NoGT-200 test set. Each pair contains one LQ image and two restorations of that image produced by distinct methods. The method pool includes PromptIR [10], InstructIR [27], 4KAgent [16], AgenticIR [15], GPT-Image 2.5 [18], Nano Banana 2 [17], HarnessIR-GPT, and HarnessIR-NB2. For each pair, two methods are sampled uniformly from this pool. The left–right order of the restorations is randomized, and method identities and metric scores are hidden from the annotators.

We recruit 30 annotators. For each pair, annotators are shown the LQ image together with the two restorations and asked to select the better result. The instructions emphasize that a preferable restoration should remove the target degradation, preserve the content and structure of the input image, and avoid visible artifacts. Each annotator provides one forced-choice judgment per pair.

We aggregate the annotations by majority vote to obtain the human preference for each pair. Pairs without a strict majority are discarded. For a metric �, we measure its agreement with human judgments using the alignment ratio:

$$
\mathrm { A l i g n } ( m ) = { \frac { \# \{ { \mathrm { v a l i d p a i r s ~ w h o s e ~ m e t r i c ~ o r d e r i n g ~ a g r e e s ~ w i t h ~ t h e ~ m a j o r i t y ~ v o t e } } \} } { \# \{ { \mathrm { v a l i d ~ p a i r s } } \} } } .
$$

A random ranking gives an expected alignment of 50%. We report the alignment of DF-Score and four representative NR-IQA metrics—MUSIQ, AFINE-NR, TOPIQ, and MANIQA—in Fig. C.2. DF-Score agrees with human preference in 87% of valid comparisons, whereas the NR-IQA metrics achieve lower alignment rates, ranging from 50.8% to 67.1%. This suggests that, when ground-truth images are unavailable, DF-Score may better reflect human preferences than the NR-IQA metrics.

Consistency Across Evaluator Families. We examine whether D-Score, F-Score, and their geometric mean, DF-Score, depend strongly on the choice of VLM evaluator. Using the restored outputs for the 400 images in Real-Paired-200 and Real-NoGT-200, we obtain scores from GPT-5.6-Sol and Claude-Opus-5 under the same evaluation protocol and compute the Pearson linear correlation coeficient (PLCC) between each evaluator’s scores and those of Gemini-3.7-Flash. Tab. C.2 reports the correlations for all three scores. Despite the presence of some outliers, the correlations are consistently high across all three measures, ranging from 0.85 to 0.94. This result suggests that DF-Score is robust to the choice of VLM evaluator. Assessing whether degradations have been removed and whether the original image content has been preserved is comparatively straightforward, and diferent VLMs appear capable of making these judgments consistently, even though their absolute scores may difer.

![](images/b7a39d7cac7592c44cf7b3b1ff5f1581a10447efb6db49e8cd9ebaa8d0a70d99.jpg)  
Figure C.2. Pairwise Align(�) with majority human preference on the real no-GT test set. Dashed line: chance (50%).

Table C.2. PLCC of D-Score, F-Score, and DF-Score between each alternative VLM evaluator and Gemini-3.7-Flash on restored outputs for the 400 images in Real-Paired-200 and Real-NoGT-200.
<table><tr><td rowspan="2">Evaluator</td><td colspan="3">PLCC vs. Gemini-3.7-Flash [44]</td></tr><tr><td>D-Score</td><td>F-Score</td><td>DF-Score</td></tr><tr><td>GPT-5.6-Sol [57]</td><td>0.85</td><td>0.87</td><td>0.94</td></tr><tr><td>Claude-Opus-5 [77]</td><td>0.87</td><td>0.89</td><td>0.92</td></tr></table>

## D Detailed Results on MiO100

Tab. D.3 shows the detailed quantitative results for each group of MiO100. Restoration becomes more challenging as the simulated degradations become more complex. Each of Groups A and B combines two degradations, while Group C combines three, resulting in more severe and interacting distortions. Accordingly, most methods show lower PSNR, SSIM, and DF-Score for Group C. For example, when using Nano Banana 2 directly, the DF-Score falls from 61.75 in Group A and 61.00 in Group B to 44.85 in Group C.

HarnessIR improves both executors in all three groups. For NB2, it raises the DF-Score to 65.47, 65.22, and 49.12 for Groups A, B, and C, respectively. The gains over direct GPT-Image-2.5 are larger for Groups A and B, at 27.29 and 20.85 points, and smaller for Group C, at 6.05 points. For all three groups, both HarnessIR variants achieve higher DF-Scores than the previous methods. Overall, the results show that HarnessIR’s advantage persists across diferent degradation levels.

## E More Visual Results

To further demonstrate the visual quality of HarnessIR, Fig. E.3 presents additional comparisons on the three test sets. Compared with direct MFM usage, HarnessIR restores facial details more faithfully and maintains a consistent color appearance, as illustrated by the first and fifth examples. It also recovers text while avoiding the spatial shifts and content changes observed in the direct MFM results, particularly in the second and third examples. For rainy and snowy scenes, HarnessIR removes the rain or snow that obstructs visibility while preserving the structure and appearance of other scene elements. For instance, it does not alter the shape of trees or hallucinate leaves in the fourth example of Fig. E.3.

The comparisons also reveal several limitations of previous methods on real-world images. Existing all-in-one and agentic methods often provide only a limited visible improvement on such inputs. InstructIR frequently brightens scenes even when low-light enhancement is not requested, as shown in the third and fifth examples. 4KAgent also sufers from a severe failure in the third example, producing a substantially corrupted result. This behavior illustrates a limitation of sequentially applying single-degradation restoration models: processing one degradation can change the distribution of the remaining degradations, causing the next model in the sequence to encounter an out-of-distribution input and fail. HarnessIR avoids this error-propagation problem by handling mixed degradations in a unified MFM execution while applying a redo refinement strategy.

Table D.3. Per-group results on MiO100, for which Tab. 1 reports the average. Marking and metrics follow Tab. 1.
<table><tr><td rowspan="2">Datasets Method</td><td rowspan="2"></td><td colspan="5">Full-reference Metrics</td><td colspan="3">No-reference Metrics</td><td colspan="3">VLM-based IQA</td></tr><tr><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓ DISTS↓</td><td>∆MANIQA↓</td><td></td><td>∆CLIP-IQA↓ΔMUSIQ↓</td><td>ΔTOPIQ↓</td><td>ΔAFINE-NR↓</td><td>D-Score↑</td><td></td><td>F-Score↑ DF-Score↑</td></tr><tr><td rowspan="10"></td><td>AirNet PromptIR</td><td>20.70</td><td>0.66 0.67</td><td>0.43 0.24 0.43 0.25</td><td>0.19 0.19</td><td>0.27 0.26</td><td>26.37 26.14</td><td>0.31 0.31</td><td>0.25 0.24</td><td>20.6 21.6</td><td>88.3 92.9</td><td>22.42 24.57</td></tr><tr><td></td><td>21.29 21.74</td><td></td><td>0.23</td><td>0.19</td><td>0.27</td><td>23.93</td><td>0.30</td><td></td><td></td><td></td><td></td></tr><tr><td>MiOIR DA-CLIP</td><td>20.87</td><td>0.69 0.65</td><td>0.39 0.42</td><td></td><td>0.26</td><td></td><td>0.30</td><td>0.23</td><td>27.5</td><td>89.1</td><td>35.04</td></tr><tr><td></td><td>21.52</td><td></td><td>0.24 0.25</td><td>0.19 0.21</td><td>0.31</td><td>24.94 27.86</td><td>0.34</td><td>0.24</td><td>20.4</td><td>92.9</td><td>25.87</td></tr><tr><td>InstructIR</td><td>21.04</td><td>0.64 0.67</td><td>0.47 0.37</td><td></td><td>0.24</td><td>15.18</td><td></td><td>0.27</td><td>24.4</td><td>87.9 66.2</td><td>31.96</td></tr><tr><td>AutoDIR</td><td></td><td></td><td>0.22</td><td>0.12</td><td></td><td></td><td>0.24</td><td>0.14</td><td>40.1</td><td></td><td>38.11</td></tr><tr><td>AgenticIR</td><td>21.53 21.02</td><td>0.70</td><td>0.31 0.19</td><td>0.11 0.31</td><td>0.17 0.17</td><td>8.80 9.47</td><td>0.17</td><td>0.07</td><td>54.8</td><td>47.1</td><td>48.09</td></tr><tr><td>MAIR</td><td>21.48</td><td>0.67 0.67</td><td>0.30 0.30</td><td>0.08</td><td>0.10</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Group A 4KAgent</td><td>22.07</td><td>0.69</td><td>0.29</td><td></td><td>0.10</td><td>5.47 4.97</td><td>0.11</td><td>0.08</td><td>49.4</td><td>32.7</td><td>36.78</td></tr><tr><td>TIR-Agent</td><td>21.80</td><td>0.70</td><td>0.30</td><td>0.25 0.30</td><td>0.15</td><td>7.37</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEAR OPERA</td><td>22.32</td><td>0.71</td><td>0.33</td><td></td><td>0.20</td><td>9.32</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>17.48</td><td>0.53</td><td>0.37</td><td>0.20</td><td></td><td>0.11</td><td>4.78</td><td>0.00</td><td></td><td>42.8 20.1</td><td>27.52</td></tr><tr><td>GPT-Image-2.5 HarnessIR-GPT</td><td>20.19</td><td>0.61</td><td>0.28</td><td>0.16</td><td>0.02 0.02</td><td>0.06</td><td>2.87</td><td>0.07 0.02</td><td>0.00</td><td>71.2 45.5</td><td>54.81</td></tr><tr><td>Nano Banana 2</td><td>21.38 22.45</td><td>0.66</td><td>0.22</td><td>0.13</td><td>0.00</td><td>0.04</td><td>0.35</td><td>0.04</td><td>0.01</td><td>80.2 51.4</td><td>61.75</td></tr><tr><td>HarnessIR-NB2 GT</td><td></td><td>0.67</td><td>0.20</td><td>0.12</td><td>0.00 0.64</td><td>0.03 0.65</td><td>0.31 68.66</td><td>0.04 0.00 0.64 -0.91</td><td></td><td>78.8 58.5</td><td>65.47</td></tr><tr><td>AirNet PromptIR MiOIR</td><td>19.97</td><td>0.61 0.70</td><td>0.50 0.38</td><td>0.28 0.22</td><td>0.21 0.16</td><td>0.28 0.23</td><td>29.06 21.46</td><td>0.34 0.25</td><td>0.27 0.20</td><td></td><td>83.1</td></tr><tr><td rowspan="10">Group B</td><td></td><td>20.38 0.71</td><td></td><td>0.36</td><td>0.15 0.17</td><td>0.23 0.22</td><td>21.42</td><td>0.25</td><td>0.19</td><td>28.8 31.6</td><td>88.1</td><td>33.91 35.52</td></tr><tr><td>DA-CLIP</td><td>21.19</td><td>0.72</td><td>0.34</td><td>0.22 0.21 0.23</td><td></td><td>20.66</td><td>0.27</td><td>0.21</td><td>39.0</td><td>90.5</td><td>48.97</td></tr><tr><td></td><td>20.76</td><td>0.69</td><td>0.38</td><td>0.16</td><td>0.26</td><td>20.91</td><td>0.27</td><td>0.16</td><td>23.8</td><td>80.2</td><td>29.74</td></tr><tr><td>InstructIR</td><td>20.37</td><td>0.62</td><td>0.51 0.27</td><td>0.19</td><td>0.27</td><td>26.93</td><td>0.32</td><td>0.27</td><td>13.2</td><td>90.4</td><td>22.54</td></tr><tr><td>AutoDIR</td><td>21.66</td><td>0.72</td><td>0.32 0.20</td><td>0.10</td><td>0.21</td><td>13.30</td><td>0.22</td><td>0.13</td><td>42.5</td><td>76.1</td><td>45.03</td></tr><tr><td></td><td>21.72</td><td>0.72</td><td>0.30 0.19</td><td>0.11</td><td>0.16</td><td>9.28</td><td></td><td></td><td></td><td>47.2</td><td>46.96</td></tr><tr><td>AgenticIR</td><td>20.92</td><td>0.70</td><td>0.28</td><td>0.29</td><td>0.14</td><td>7.87</td><td>0.16</td><td>0.05</td><td>53.4</td><td></td><td></td></tr><tr><td>MAIR</td><td>20.95</td><td>0.67</td><td>0.30</td><td>0.07</td><td>0.10</td><td>6.16</td><td>0.11</td><td>0.09</td><td>51.9</td><td>45.2</td><td>44.09</td></tr><tr><td>4KAgent TIR-Agent</td><td>22.80</td><td>0.73</td><td>0.27</td><td>0.24</td><td>0.10</td><td>3.61</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEAR</td><td>22.13</td><td>0.73</td><td>0.29</td><td>0.30</td><td>0.14</td><td>8.20</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OPERA</td><td>22.28</td><td>0.75</td><td>0.31</td><td>0.31</td><td>0.19</td><td>9.75</td><td></td><td></td><td></td><td></td><td>30.42</td></tr><tr><td>GPT-Image-2.5 HarnessIR-GPT</td><td>18.42 19.96</td><td>0.56 0.60</td><td>0.34 0.28</td><td>0.19 0.16</td><td>0.02 0.02</td><td>0.11 0.08</td><td>4.65 3.36</td><td>0.07 0.03</td><td>0.02 0.01</td><td>45.4 69.6</td><td>22.1 40.2 51.27</td></tr><tr><td>Nano Banana 2 HarnessIR-NB2</td><td>21.23 22.94</td><td>0.66 0.68</td><td>0.22 0.19</td><td>0.12</td><td>0.00 0.00</td><td>0.03 0.03</td><td>0.67 0.14</td><td>0.04 0.04</td><td>0.01</td><td>79.9 78.1</td><td>50.2 61.00 58.2 65.22</td></tr><tr><td>GT</td><td></td><td></td><td></td><td>0.11</td><td>0.64</td><td>0.65</td><td>68.85</td><td>0.64</td><td>0.00 -0.92</td><td></td><td></td></tr><tr><td>LQ</td><td>20.64</td><td>0.60</td><td>0.54</td><td>0.29</td><td>0.19</td><td>0.26</td><td>27.62</td><td>0.33</td><td>0.25</td><td></td><td></td></tr><tr><td>AirNet PromptIR MiOIR</td><td>20.22</td><td>0.59 0.59</td><td>0.57 0.57</td><td>0.30 0.30</td><td>0.30 0.30</td><td>0.35 0.35</td><td>39.18 39.39</td><td>0.41 0.42</td><td>0.35 0.35</td><td>26.0 97.6 26.0 94.1</td><td>33.00 31.15</td></tr><tr><td rowspan="9">DA-CLIP InstructIR AutoDIR AgenticIR MAIR Group C 4KAgent TIR-Agent SEAR OPERA</td><td></td><td>20.30</td></table>

![](images/71bbf9f8fb6cc43528d5039d083878bb1b1cbdb5da8488c13c7ae2355946b301.jpg)  
Figure E.3. Additional visual comparisons of diferent methods. HarnessIR more efectively removes degradations while preserving image content. Zoom in for a better view.