# Instruction Mining From Images: Constructing a Synthetic Dataset for Multimodal Learning ILSP DCVLR Competition Submission

Eleni E. Kasoura, Dimitrios Damianos, Georgios Paraskevopoulos and Vassilios Katsouros \*

Institute of Language and Speech Processing

Athena Research Center

Athens, GR 151 25

{eleni.kasoura, d.damianos, g.paraskevopoulos, vsk}@athenarc.gr

## Abstract

This work presents a compact and reproducible pipeline for synthesizing and curating image-based instruction-response pairs for DCVLR. We draw from the Cauldron collection and use Magpie-style instruction and response synthesis with Qwen2.5-VL-7B-Instruct to create concise, image-conditioned instruction pairs. The pipeline consists of instruction synthesis, conditioned on input images and controlled via a system prompt, sanitization, tagging, and filtering of the instructions based on the score obtained by a reward model (DeBERTa-v3). We retain samples with positive scores to maximize the quality of the generated instructions. We observe that score distributions have a median and mean reward score of  1.53 and  1.75 respectively, indicating that most generated instructions are of good quality. The resulting dataset contains diverse instructions in the math, spatial, and social reasoning domains.

## 1 Introduction

Multimodal systems increasingly rely on image-grounded instruction–response data to elicit robust visual reasoning. However, existing resources often prioritize scale over coverage, cleanliness, and reproducibility, which are equally critical for the DCVLR competition. We ask: Can we build a compact, high-quality, and fully traceable image–instruction corpus that reliably triggers visual reasoning without sacrificing eficiency?

Here, we present a single-pass, compute-eficient pipeline that synthesizes concise, visionaware instructions and high-fidelity responses with explicit traceability and tunable quality control.

Our approach adapts recent synthesis pipelines (i.e., Magpie Xu et al.) to vision–language models (VLMs). Starting from the Cauldron collection Laurençon et al. [2024], we condition the Magpie synthesis pipeline on the images to generate instructions and corresponding answers using Qwen2.5-VL-7B-Instruct Bai et al. [2025] under input formatting and sanitization. The instructions are then tagged, and scored using a reward model to keep only instructions with high reward scores, as a quality control step.

Concretely, the pipeline proceeds through five stages: (1) loading and optional subsampling; (2) instruction synthesis; (3) response generation with vision-aware decoding; (4) instruction sanitization; and (5) reward-model scoring. For multi-image conditioning, we introduce a Magpie-inspired formatting strategy that packs vision tokens to yield concise, grounded interactions without bloated context.

Contributions.

• Magpie-style pipeline: We use a system prompt and an image-only pre-query; the model first synthesizes the user instruction from the image, then generates the paired assistant response—yielding grounded instruction–response pairs.

• Vision-aware formatting for multi-image inputs. A Magpie-inspired scheme for conditioning on multiple images (via vision tokens) that preserves grounding while keeping interactions concise.

• Automated, tunable quality control. A reward-model scoring stage with a tunable threshold t. Stricter thresholds lead to higher rejection rates, but increase the final instruction quality

## 2 Related Work

Template-Driven Instruction Synthesis. Aligned language models can self-synthesize instructions when prompted with the left context of a chat template, eliminating seed questions and heavy prompt engineering. Magpie Xu et al. formalizes this idea and scales instruction creation with simple formatting and cleaning. Earlier works, i.e., Self-Instruct Wang et al. [2023], Alpaca Dubois et al. [2023], and Evol-Instruct/WizardLM Xu et al. [2024] have more limited scalability and diversity, regarding the number of synthesized instructions. We adapt Magpie’s self-synthesis to the VLM setting while enforcing strict formatting and sanitization.

Vision–Language Instruction Tuning and Formatting. Modern VLMs adopt concise, chat-oriented interfaces and lightweight post-training. Modern VLM examples tuned on visual instructions include LLaVA Liu et al. [2023], InstructBLIP Dai et al. [2023], MiniGPT-4 Zhu et al.. These systems exhibit visual reasoning capabilities and excellent multi-task performance, highlighting the benefits of multi-image conditioning, text-rich inputs, and instruction diversity.

Reward Models and Post-Hoc Filtering. Quality control in alignment often relies on reward models to filter or weigh data before training Ouyang et al. [2022], Bai et al. [2022]. We use a reward model based on DeBERTa-v3 He et al. for instruction scoring which provides good performance and low latency. Thresholding the scores yields a simple, reproducible keep/cut rule.

## 3 Methodology

Fig. 1 shows the proposed multimodal instruction synthesis pipeline. We adopt a Magpiestyle pipeline that turns images into concise, grounded prompt-response-image triplets, with lightweight cleaning and a reward-based filter. Concretely, we use an image-only pre-query; the model first autoregressively synthesizes the user instruction based on the image, and then, after inserting the assistant header, generates the paired assistant response. We apply light sanitization (token/whitespace normalization, template validation), rule-based guards for modality grounding, i.e., the instruction must reference visible content, and near-duplicate removal. Each example is conditioned by the image and a short system prompt that specifies the task family, e.g., “ask about spatial arrangement”. We use a compact chat format system turn followed by a user turn that carries vision tokens, so the model is primed to emit an instruction directly from visual context (self-synthesis).

![](images/f23a756d95f4106c8881ab609eb8c5b0f54303bd374088e6ac780b313359f716.jpg)  
Figure 1: The proposed pipeline for data synthesis and curation based on Magpie.

The model produces a candidate instruction conditioned on the image. A sanitization pass removes boilerplate, control markers, and residual formatting, yielding a minimal, modelready prompt. Given the cleaned instruction and the same image context, the model generates a response. A second sanitizationr pass standardizes punctuation and trims generation artifacts. We then perform tagging, e.g., dificulty labels, for downstream analysis.

All instruction–response pairs pass through a reward model that assigns a scalar score. A simple threshold selects retained items (higher scores) and discards low-quality pairs, enabling stricter or looser quality control. An optional reasoning check can be applied before final acceptance, followed by a brief final sanitization to ensure formatting consistency.

The outputs of this pipeline are compact, cleaned instructions, responses, and tags, together with reward scores. These artifacts are ready for post-hoc filtering, analysis, and integration into instruction-tuning or evaluation workflows.

## 4 Implementation details

For instruction and response synthesis we use Qwen2.5-VL-7B-Instruct. Images are sampled from the Cauldron collection, a unified collection of  50 vision–language training sets released with Idefics2 Laurençon et al. [2024]. Cauldron aggregates multidomain tasks (VQA, OCR/docs, charts/tables, math/logic, multi-image, screenshots) and is widely reused for instruction tuning. We sample 5000 images from each subset and generate instructions based on the sampled images. The reward model used for scoring is OpenAssistant/reward-model-deberta-v3-large-v2<sup>2</sup>. We set the selection threshold to t = 0, keeping samples with positive scores, and then select the top 10000 samples among those. All experiments are run on nodes from the Leonardo-Booster supercomputer. Each node contains 4 NVIDIA A100 GPUs with 64GB VRAM.

## 4.1 System Prompts

System prompts are organized into categories to steer generation for diferent capabilities. We maintain targeted categories for text extraction, spatial relations, and mathematics/quantitative reasoning, alongside a general category that applies across most subsets as seen in Appendix A.

![](images/c3c7a4fd3b436b2e668afc5dcc9bc71a00d2ee11f328fe3aa4f3cc5fd6345b2e.jpg)  
Figure 2: Reward score distribution of selected samples

Table 1: Subset counts for sampes included in the generated dataset. The dataset includes 10000 samples in total.
<table><tr><td>Subset</td><td>Count</td></tr><tr><td>cocoqa</td><td>1354</td></tr><tr><td>aokvqa</td><td>1311</td></tr><tr><td>vqav2</td><td>1289</td></tr><tr><td>clevr</td><td>998</td></tr><tr><td>ocrvqa ai2d</td><td>971</td></tr><tr><td></td><td>695</td></tr><tr><td>vsr</td><td>539</td></tr><tr><td>robut</td><td>516</td></tr><tr><td>mapqa</td><td>396</td></tr><tr><td>infographic</td><td>377</td></tr><tr><td>vistext</td><td>260</td></tr><tr><td>chart2text</td><td>198</td></tr><tr><td>figureqa</td><td>197</td></tr><tr><td>iconqa</td><td>192</td></tr><tr><td>chartqa</td><td>185</td></tr><tr><td>vqarad</td><td>103</td></tr><tr><td>diagram_image_to_text tat</td><td>82</td></tr><tr><td></td><td>81</td></tr><tr><td>hitab</td><td>73</td></tr><tr><td>hateful</td><td>70</td></tr><tr><td>finqa</td><td>61</td></tr><tr><td></td><td></td></tr><tr><td>dvqa</td><td>52</td></tr><tr><td>Total</td><td>10 000</td></tr></table>

## 5 Dataset Statistics

In Fig. 2 we see the reward score distribution of the selected samples. We see that the average and median reward scores are 1.7518 and 1.5341 respectively. with sample scores ranging from 0.5 to 6.5. Table 1 presents the subset counts for samples included in the final dataset. We observe that the dataset is skewed towards Visual Question Answering samples, while subsets containing math, diagrams and infographics are less represented. One explanation for this is that due to the higher complexity of these images, the model generates lower quality instructions and they are rejected by the reward model.

## 6 Conclusions

In this work we extend Magpie to generate visual instructions grounded on input images. Instruction synthesis is guided by system prompts and the quality of the resulting samples is automatically determined through reward model scores. In our implementation we have observed that proper sanitization of the generated instructions and the inclusion of system prompts for conditioning is paramount, since VLMs are biased towards producing captions and descriptions of the input images without further guidance.

In the future we plan to extend this pipeline to generate more diverse visual instruction sets, through the inclusion of more system prompts and diverse input images. Furthermore, we plan to run detailed ablations to guide our decisions on reward model, threshold and system prompt selection.

## Acknowledgements

We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC Development Access call.

## Limitations

Our work is still at the proof of concept stage and has several limitations. First, due to time constraints ablation studies were not systematically performed. The model used for synthesis is a relatively small (7 billion) parameter LLM, limiting the quality and diversity of the synthesized visual instructions. Finally, stratification of the selected samples during reward scoring and inclusion of more system prompts may be included to ensure higher diversity of the final dataset.

## Broader Impact

Our work on synthetic data generation has broad societal implications, ofering substantial benefits while also posing serious risks. On the positive side, this technology can significantly accelerate machine learning research, since scalable data synthesis pipelines are an economically viable way to obtain data, especially for areas and problems where data are relatively scarce. Furthermore, synthetic data ofer pathways for more private and fair models, since inherent biases can be balanced out and personal identifying information that exists in real datasets is not included. However, these capabilities are shadowed by the potential for malicious use, particularly in the generation of sophisticated disinformation and deepfakes. Furthermore, improperly trained models risk amplifying the societal biases they are intended to correct. Progress in this domain must be tightly coupled with a commitment to developing detection techniques, fairness audits, and clear ethical guidelines. Finally, there is an ongoing discussion about the decline in diversity of LLM outputs due the selfconsuming training loops in the research community, and synthetic data contribute to this issue. Further investigation should be performed about the limits of this approach.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36:30039–30069, 2023.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing. In The Eleventh International Conference on Learning Representations.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37:87874–87907, 2024.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, 2023.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, 2024.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In The Thirteenth International Conference on Learning Representations.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations.

A System prompt categories

<table><tr><td rowspan=1 colspan=1>Prompt type</td><td rowspan=1 colspan=1>Prompt examples (2 per type)</td></tr><tr><td rowspan=1 colspan=1>prompts</td><td rowspan=1 colspan=1>Try to ask an easy question for this image.Create an instruction that prompts the user to explain the se-quence of events taking place in the image.</td></tr><tr><td rowspan=1 colspan=1>math_prompts</td><td rowspan=1 colspan=1>Create a question that asks the user to describe the mathematicalconcept represented in the diagram.Write an instruction that asks the user to summarize the mainstatistical findings illustrated by this chart.</td></tr><tr><td rowspan=1 colspan=1>multi_image</td><td rowspan=1 colspan=1>Compare the 2 images and write a question that highlights thedifferences in the objects or scenes depicted.Look at the 2 images and create an instruction that asks the userto identify the similarities between the two.</td></tr><tr><td rowspan=1 colspan=1>general_prompts</td><td rowspan=1 colspan=1>Ask a question about one of the objects in this image.Create a question about what the people or animals are doing inthis picture.</td></tr><tr><td rowspan=1 colspan=1>spatial_prompts</td><td rowspan=1 colspan=1>Write a question that asks the user to describe where the mainobjects are located in relation to each other.Generate an instruction that asks the user to explain whether thisis an indoor or outdoor scene.</td></tr><tr><td rowspan=1 colspan=1>text_prompts</td><td rowspan=1 colspan=1>Generate a question that asks the user to read and summarize thevisible text in the image.Write an instruction that asks the user to transcribe the letters orwords shown in this picture.</td></tr></table>

Table 2: Prompt categories with two example prompts per type.