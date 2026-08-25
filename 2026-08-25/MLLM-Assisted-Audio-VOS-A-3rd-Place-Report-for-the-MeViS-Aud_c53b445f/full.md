# MLLM-Assisted Audio VOS: A 3rd Place Report for the MeViS-Audio Track, 8th LSVOS Challenge

Liangtao Shi<sup>1</sup>, Jinxia Xie<sup>2</sup>, Xiantao Hu<sup>2</sup>, and Ting Liu<sup>3</sup>

<sup>1</sup> Hefei University of Technology <sup>2</sup> Nanjing University of Science and Technology 3 Hunan Police College

Abstract. In this technical report, we present a training-free framework for audio-guided video object segmentation, which integrates Multimodal Large Language Models (MLLMs) with SAM-based segmentation models. We decompose the task into several stages and identify suitable foundation models for each stage. Without introducing additional model training or task-specific fine-tuning, our approach leverages the strong multimodal reasoning capabilities of MLLMs to model text-visual correspondence and employs SAM-based models for accurate object mask generation. The proposed framework demonstrates the efectiveness of leveraging foundation models for audio-guided video segmentation and achieves competitive performance in the MeViS-Audio Track of the 8th LSVOS Challenge.

Keywords: Audio-guided Video Object Segmentation · Segment Anything Model · Multimodal Large Language Model

## 1 Introduction

The 8th Large-scale Video Object Segmentation (LSVOS) Challenge provides a series of challenging benchmarks for advancing video object segmentation. It evaluates the capability of modern segmentation systems to understand complex visual content, accurately localize target objects, and maintain robust pixel-level segmentation across diverse video scenarios. With the rapid development of video understanding and multimodal learning, the challenge further explores diverse segmentation settings that incorporate visual, language, and audio cues through three tracks, covering complex video object segmentation, text-based referring video object segmentation, and audio-guided video object segmentation.

1) Complex Video Object Segmentation (MOSEv2 Track): As a classic track of the LSVOS Challenge, this track focuses on robust video object segmentation under challenging scenarios, including object disappearance and reappearance, severe occlusion, small objects, and crowded scenes. The evaluation is conducted on a test subset of MOSEv2 [5], an extension of MOSE [3] designed for complex VOS. MOSEv2 contains 5,024 videos with 10,074 objects and

701,976 masks across 200 categories, introducing more realistic challenges such as long-term disappearance, camouflage, adverse conditions, and non-physical targets.

2) Text-based Referring Video Object Segmentation (MeViS-Text Track): This track aims to segment video objects specified by natural-language motion expressions. The challenge evaluates models on a subset of MeViSv2 [4]. MeViSv2 extends the original MeViS [2] benchmark, which contains 2,006 videos, 8,171 objects, 28,570 motion expressions, and approximately 443K object masks. The new version preserves the original video collection while expanding the language annotations to 33,458 sentences and adding motion-reasoning and notarget expressions, as well as audio descriptions and trajectory annotations. The task requires jointly modeling language, object motion, and temporal context, making it a challenging testbed for motion-aware vision-language understanding and referring video object segmentation.

3) Audio-guided Video Object Segmentation (MeViS-Audio Track): The 8th LSVOS Challenge introduces this new track to extend referring expression video segmentation from language to audio. Given an audio clip associated with the target object, models are required to localize and segment the corresponding object throughout the video. The task involves associating acoustic cues with visual objects and maintaining the correspondence over time, introducing an additional challenge of audio-visual alignment beyond conventional referring video object segmentation. The evaluation is performed on a subset of MeViS-Audio [4].

Although the MeViS-Audio Track is newly introduced, previous solutions from the MOSEv2 and MeViS-Text Tracks [7] provide valuable insights. Meanwhile, recent studies on Audio-guided Video Object Segmentation, including the 5th PVUW Challenge [6], have explored efective solutions. A notable paradigm is to convert audio into text via ASR, allowing existing RVOS models and MLLMs to be applied. Following this paradigm, our framework first converts audio into textual descriptions using ASR models. The MLLM then jointly reasons over textual and visual inputs to refine referring prompts and select informative keyframes. RVOS/VOS models subsequently perform pixel-level mask prediction, followed by MLLM-based semantic verification between the predicted masks and textual descriptions. This framework achieves 3rd place in the MeViS-Audio Track of the 8th LSVOS Challenge.

## 2 Method

Our method develops a training-free pipeline for Audio-guided Video Object Segmentation, where MLLMs perform video-text understanding, while SAM-based models perform pixel-level segmentation. The overall pipeline comprises four stages: (1) audio-to-text conversion, (2) video-text joint analysis, (3) text-based video segmentation and mask-based tracking, and (4) mask-text consistency verification.

## 2.1 Stage 1: Audio-to-Text Conversion.

Given an audio clip, we first convert it into a textual representation using an automatic speech recognition (ASR) model. Specifically, we adopt Qwen3-ASR-1.7B [8] to transcribe the input audio A:

$$
q = \varPhi _ { \mathrm { A S R } } ( A ) ,
$$

where $q$ denotes the generated transcription. This modality conversion enables the audio-guided task to leverage existing text-conditioned video understanding and segmentation models without requiring additional task-specific training. However, the ASR transcription alone may provide only a coarse-grained description of the referred targets, which may be insuficient for existing RVOS methods when multiple targets, fine-grained attributes, or complex spatial and temporal cues are involved. Therefore, we employ an MLLM to perform joint analysis of the textual query and video content, refining the original query into a more informative and fine-grained referring prompt for subsequent video object segmentation.

## 2.2 Stage 2: Video-Text Joint Analysis.

Given the transcribed query q and the video ${ \mathrm { V } } ,$ we employ Gemini-3-Flash-Preview to jointly analyze the textual and visual information and identify the referred instances. Specifically, we determine the number of target instances and generate an instance-specific referring prompt and a representative keyframe for each target. For the i-th target, we obtain a structured tuple

$$
o _ { i } = ( d _ { i } , k _ { i } ) ,
$$

where $d _ { i }$ denotes a fine-grained referring prompt and $k _ { i }$ denotes the selected keyframe index. The prompt captures the visual characteristics and contextual cues that distinguish the target from other instances, while the keyframe indexed by $k _ { i }$ serves as the initialization frame for subsequent tracking.

## 2.3 Stage 3: Text-based Video Segmentation and Mask-based Object Tracking.

Given the target description $d _ { i }$ , we first apply MomentSeg [1] to perform referring video object segmentation on the entire video. For each target instance $o _ { i } .$ MomentSeg predicts a coarse mask sequence:

$$
\tilde { M } _ { i } = \{ \tilde { m } _ { i , t } \} _ { t = 1 } ^ { T } = \varPhi _ { \mathrm { M o m e n t S e g } } ( V , d _ { i } ) ,
$$

where $\tilde { m } _ { i , t }$ denotes the predicted mask at frame t. While MomentSeg provides coarse masks over the entire video, these masks may lack the pixel-level accuracy and temporal consistency required for high-quality segmentation. We therefore use the predicted mask at the MLLM-selected keyframe $k _ { i } .$ , denoted as $\tilde { m } _ { i , k _ { i } }$ , as the initial mask for subsequent bidirectional propagation and refinement with DAM4SAM [9]. Starting from the keyframe $k _ { i } ,$ , DAM4SAM propagates and refines the target mask bidirectionally across the video:

$$
\begin{array} { r } { \hat { M } _ { i } = \{ \hat { m } _ { i , t } \} _ { t = 1 } ^ { T } = \phi _ { \mathrm { D A M 4 S A M } } ( V , \tilde { m } _ { i , k _ { i } } , k _ { i } ) . } \end{array}
$$

For multiple target instances, the above procedure is performed independently for each instance, using its corresponding referring description and keyframe mask. The resulting per-instance mask sequences are subsequently aggregated to obtain the final segmentation result.

## 2.4 Stage 4: Mask-Text Consistency Verification.

Although DAM4SAM provides strong pixel-level segmentation and temporal tracking capabilities, its propagated masks may still drift toward visually similar distractors or fail under occlusion. To mitigate these errors, we introduce Gemini-3-Flash-Preview as a semantic verification module. Specifically, we merge the predicted masks of all instances in each frame and overlay the resulting mask on the corresponding video frame. The visualized masks, together with the transcribed text ${ \mathrm { q } } ,$ are then fed to Gemini-3-Flash-Preview to verify their semantic consistency. If the predicted mask is inconsistent with the query, it is discarded by setting the mask to empty; otherwise, it is retained.

## 3 Experiments

## 3.1 Evaluation Metrics

Following the oficial evaluation protocol of the 8th LSVOS MeViS Audio challenge, we adopt five metrics for evaluation, including region similarity $( \mathcal { I } )$ , boundary accuracy (F), N-acc., T-acc., and the final score. The region similarity metric $\mathcal { I }$ measures the overlap between the predicted segmentation regions and the ground-truth regions. The boundary accuracy metric $\mathcal { F }$ evaluates the alignment between the predicted object boundaries and the ground-truth boundaries. The combined metric $\mathcal { T } \& \mathcal { F }$ is computed as the average of $\mathcal { I }$ and F. N-acc. and T-acc. evaluate the model performance on no-target and target cases, respectively. The final score is computed by combining $\mathcal { T } \& \mathcal { F }$ , N-acc., and T-acc. for comprehensive evaluation.

## 3.2 Ablation Study

We evaluate the efectiveness of diferent components in our pipeline on the test set of the 8th LSVOS MeViS-Audio challenge. As shown in Table 1, MomentSeg achieves a $\mathcal { T } \& \mathcal { F }$ score of 53.09%, providing reliable initial segmentation results based on text-guided prompts. By incorporating DAM4SAM for mask propagation and refinement, the $\mathcal { I } , \mathcal { F }$ , and $\mathcal { T } \& \mathcal { F }$ scores improve to 53.81%, 59.63%, and 56.72%, respectively, demonstrating the efectiveness of temporal mask refinement.

Table 1: Comparison of diferent models on the 8th LSVOS MeViS-Audio test set.
<table><tr><td>Method</td><td>N-acc. T-acc.</td><td></td><td>J</td><td>F</td><td></td><td>J&amp;F Final Score</td></tr><tr><td>MomentSeg</td><td>41.38</td><td>92.53</td><td></td><td></td><td>49.72 56.47 53.09</td><td>62.33</td></tr><tr><td>+ DAM4SAM</td><td>41.38</td><td></td><td></td><td></td><td>92.5353.81 59.63 56.72</td><td>63.54</td></tr><tr><td>+ Consistency Verification 96.55</td><td></td><td></td><td></td><td></td><td>57.8344.34 47.7146.03</td><td>66.80</td></tr></table>

After introducing the consistency verification module, the final score further increases from 63.54% to 66.80%. Although the pixel-level segmentation metrics decrease due to stricter semantic filtering, N-acc. is significantly improved from 41.38% to 96.55%, indicating that the verification module efectively enhances the ability to distinguish target and no-target expressions. Overall, the results demonstrate that each component contributes diferently to the final system, where DAM4SAM improves mask quality and consistency verification enhances the robustness of audio-guided video object segmentation.

## 4 Conclusion

In this work, we present a training-free framework for audio-guided video object segmentation by integrating multimodal large language models with SAM-based segmentation models. By combining audio understanding, multimodal reasoning, mask propagation, and semantic verification, our framework efectively aligns audio cues with visual targets and achieves temporally consistent segmentation without task-specific training. Experiments on the MeViS-Audio Track of the 8th LSVOS Challenge demonstrate the efectiveness of our approach, achieving competitive performance and ranking 3rd in the challenge.

## References

1. Dai, M., Yang, S., Duan, B., Yang, W., Wang, J.: Momentseg: Moment-centric sampling for enhanced video pixel understanding. ECCV (2026)

2. Ding, H., Liu, C., He, S., Jiang, X., Loy, C.C.: Mevis: A large-scale benchmark for video segmentation with motion expressions. In: ICCV. pp. 2694–2703 (2023)

3. Ding, H., Liu, C., He, S., Jiang, X., Torr, P.H., Bai, S.: Mose: A new dataset for video object segmentation in complex scenes. In: ICCV. pp. 20167–20177 (2023)

4. Ding, H., Liu, C., He, S., Ying, K., Jiang, X., Loy, C.C., Jiang, Y.G.: Mevis: A multimodal dataset for referring motion expression video segmentation. IEEE TPAMI (2025)

5. Ding, H., Ying, K., Liu, C., He, S., Jiang, X., Jiang, Y.G., Torr, P.H., Bai, S.: Mosev2: A more challenging dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2508.05630 (2025)

6. Liu, C., Ding, H., Ravi, N., Wei, Y., He, S., Bai, S., Torr, P., Cao, L., Zhang, J., Miao, D., et al.: Report of the 5th pvuw challenge: Towards more diverse modalities in pixel-level understanding. arXiv preprint arXiv:2604.26031 (2026)

7. Liu, C., Ding, H., Ying, K., Hong, L., Xu, N., Yang, L., Fan, Y., Gao, M., Chen, J., Miao, Y., et al.: Lsvos 2025 challenge report: Recent advances in complex video object segmentation. arXiv preprint arXiv:2510.11063 (2025)

8. Shi, X., Wang, X., Guo, Z., Wang, Y., Zhang, P., Zhang, X., Guo, Z., Hao, H., Xi, Y., Yang, B., et al.: Qwen3-asr technical report. arXiv preprint arXiv:2601.21337 (2026)

9. Videnovic, J., Lukezic, A., Kristan, M.: A distractor-aware memory for visual object tracking with sam2. In: CVPR. pp. 24255–24264 (2025)