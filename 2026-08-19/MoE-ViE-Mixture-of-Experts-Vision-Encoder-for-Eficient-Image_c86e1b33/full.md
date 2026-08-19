# MoE-ViE: Mixture of Experts Vision Encoder for Eficient Image and Video Understanding

Bonan Zhang<sup>1</sup>, Shiyu Dong<sup>1</sup>, Quan Hung Tran<sup>1</sup>, Katharina Gschwind<sup>⋆</sup>, Shuqi Yang<sup>⋆</sup>, Sijia Chen<sup>⋆</sup>, Adel Ahmadyan<sup>1</sup>, Seungwhan Moon<sup>1</sup>, Lu Zhang<sup>1</sup>, Ahmed Kirmani<sup>1</sup>, Babak Damavandi<sup>1</sup>, and Anuj Kumar<sup>1</sup>

<sup>1</sup>Meta

<sup>⋆</sup>Work done while at Meta

{bonanz,shiyudong,quanhungtran,kgschwind,shuqiyang ,sijiac,ahmadyan,luzhang20,kirmani,babakd,anujk}@meta.com

Abstract. Vision encoders are a critical component of vision-language models, and scaling their capacity efectively improves performance. However, dense scaling increases compute cost and inference latency. Mixtureof-Experts (MoE) architectures ofer a compelling alternative, having enabled eficient scaling in LLMs, yet the MoE design space for CLIP-style vision encoders remains underexplored at State-of-the-Art (SOTA) levels. In this work, we systematically study MoE designs for vision encoder scaling and find that fine-grained MoE topologies yield substantial gains over both dense and standard MoE counterparts. We further propose an auxiliary-loss-free balancing variant for better expert utilization, and design a specialized MoE kernel to mitigate inference latency overhead. To enhance video capabilities while preserving image knowledge, we introduce frame-level distillation paired with a novel freezing mechanism. We pretrain a series of Mixture-of-Experts Vision Encoders (MoE-ViE) across a range of sizes, all consistently outperforming their dense counterparts. Our largest model matches the zero-shot performance of a SOTA encoder 1.7× its size at 76% of its latency. When aligned with an LLM, MoE-ViE surpasses all compared encoders on image and video benchmarks, including those with up to 5× more activated parameters. Code is available at https://github.com/facebookresearch/moe\_vie.

Keywords: Vision encoder · Mixture-of-Experts · CLIP

## 1 Introduction

Vision encoders have become a cornerstone in today’s vision-language models (VLMs). The pursuit of higher accuracy and broader applicability necessitates the continuous scaling up of vision models, which in turn has been a reliable route to stronger VLM capabilities [5, 12, 28, 70]. However, scaling dense vision encoders comes with system-level trade-ofs [11, 21]. Increasing their capacity typically induces growth in compute cost and inference-time latency. This issue becomes even more acute for high-resolution images or long videos, where moderate increases in encoder cost compound into substantial end-to-end overhead.

Mixture-of-Experts (MoE) ofers an alternative scaling mechanism by increasing the total model capacity while only activating a small subset of experts for any given input, thus decoupling model capacity from compute cost [63]. MoE has been widely used in large language models (LLMs) [1,36,47,72,85] and has carried over to VLMs [3, 18, 31, 45, 71]. In contrast, applying MoE to vision encoders, particularly to the prevailing contrastive language-image pretraining (CLIP) [59], remains less settled. While prior research provides several attempts on developing MoE vision encoders, none has closed the gap to State-of-the-Art (SOTA) performance on core vision benchmarks [43, 54, 61, 79, 92].

In this work, we present a systematic study of MoE architecture design in vision encoders across multiple scales. We observe that conventional MoE [63], i.e., treating MoE as a drop-in replacement for dense MLP blocks, provides only limited gains. In comparison, MoE vision encoders with fine-grained expert designs achieve larger, more consistent improvements. This aligns with insights from recent work on MoE-based LLMs [47]. MoE with fine-grained expert design is also well matched to the nature of encoding vision information, where a vision encoder is responsible for supporting a wide variety of visual concepts, styles, and domains. We further propose a variant of loss-free balancing [77] to balance expert utilization during training. One practical challenge is the system overhead introduced by memory-bound operations in MoE computations, which can hurt the latency benefits implied by sparsely activated parameters. We therefore develop an MoE kernel in Triton [74] to overcome this.

Beyond these architectural choices, we focus on refining our training strategy to emphasize building a unified image and video encoder. We first pretrain on a vast set of image-text pairs via contrastive training and then, following [5], we finetune with video data to enhance the model’s capability in video understanding. However, we observe that vanilla video finetuning can shift representations in ways that degrade our pretrained image capabilities. We mitigate this by introducing a frame-level distillation during video finetuning and by freezing MoE experts. Together, we are able to preserve the image representation learned during pretraining while allowing video supervision to strengthen video representation. Fig. 1 summarizes our MoE architecture and training pipeline.

Our investigation yields several key findings:

– We show that with appropriate architectural design, vision encoders with MoE consistently outperform their dense counterparts across multiple scales. This confirms the eficacy of sparse scaling for CLIP-style vision encoders.

– We demonstrate that the MoE design proposed in this work substantially improves over prior works on MoE vision encoders across model sizes.

– Our trained MoE-ViE achieves SOTA results at all scales on zero-shot image and video benchmarks. Our largest encoder achieves results comparable to the SOTA dense encoder $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G }$ , which is 1.7× larger.

– Our MoE kernel provides > 2.5× speedup in inference latency. It can be used as a drop-in implementation for diferent MoE topologies. – Aligned with an LLM, MoE-ViE yields strong downstream results, and matches or exceeds other SOTA encoders in comparable settings.

![](images/34bcc2625d0a9035099de469a0156c24cfded1cc3396123f3ab3784331c3719c.jpg)  
Fig. 1: Illustration of MoE-ViE architecture and training pipeline. MoE-ViE adopts a fine-grained MoE design with optimized MoE kernels for latency reduction. We first perform contrastive pretraining on large-scale image-text pairs. During video finetuning, we introduce frame-level distillation and expert freezing to preserve the pretrained image understanding capabilities.

## 2 Vision Encoder with MoE

In this section, we first review the MoE formulation, then present our studies on introducing MoE into CLIP vision encoders, including the architectural design and the control of expert utilization.

## 2.1 MoE Preliminaries

A typical MoE layer replaces the dense MLP block with N experts $\{ E _ { i } \} _ { i = 1 } ^ { N }$ Given an input token $x \in \mathbb { R } ^ { d }$ , a router produces logits

$$
s = W _ { r } x \in \mathbb { R } ^ { N }\tag{1}
$$

followed by selecting a subset of experts via top-k routing. The standard Softmaxbased gating computes

$$
g ( x ) = \mathrm { S o f t m a x } ( \mathrm { t o p } { - } k ( s ) )\tag{2}
$$

Denote $\mathcal { T } ( x ) \subset \{ 1 , \ldots , N \}$ to be indices of selected experts, the MoE output is

$$
y = \sum _ { e \in \mathcal { T } ( x ) } g _ { e } ( x ) E _ { e } ( x )\tag{3}
$$

MoE increases the parameter count with N while keeping the per-token compute proportional to k experts, providing a favorable capacity vs. compute trade-of when $k \ll N$

## 2.2 MoE for Vision Encoder

Adapting MoE to CLIP vision encoders raises three key design questions: (i) what expert topology is most appropriate for vision representations; (ii) how to ensure balanced expert utilization; (iii) how to realize the theoretical eficiency gains promised by MoE. We address these questions in the following sections.

Architecture Design Prior CLIP MoE works replace each dense MLP with N experts whose capacity matches the original MLP [43, 54, 79, 92]. We instead adopt a fine-grained expert design, i.e., each expert has reduced hidden width relative to the original dense MLP, enabling more experts under a comparable compute budget. This choice is motivated by the heterogeneous nature of visual features (e.g., color, shape, spatial layout) and by findings from LLMs that finer expert granularity yields more eficient scaling [20]. We apply MoE only to the vision tower and keep the text tower unchanged, since only the vision tower is used in any subsequent VLM alignment.

In addition to specialized experts, we include m shared experts that are always active, inspired by the idea of retaining a global thumbnail alongside high-resolution tiles in [48] as well as the success in LLMs [20]. The shared experts provide a persistent pathway for global context, while the routed experts capture conditional, token-specific transformations. We demonstrate expert specialization in more detail in Section 4.5.

We further replace the standard MoE Softmax gating with a Sigmoid function to avoid potential expert competition [19, 55]. Given logits s, we compute<sup>1</sup>

$$
g ( x ) = \mathrm { t o p } { - } k ( { \mathrm { S i g m o i d } } ( s ) )\tag{4}
$$

$$
g _ { e } ^ { \prime } ( x ) = \frac { g _ { e } ( x ) } { \sum _ { i \in \mathcal { T } ( x ) } g _ { i } ( x ) }\tag{5}
$$

where $g ^ { \prime } ( x )$ renormalizes the selected scores to keep the mixture scale stable.

Let there be cN experts in total, where each expert has hidden width reduced by c. We reserve the first m experts as shared and route over the remaining specialized experts, $\therefore e . , T ( x ) \subset \{ m + 1 , \ldots , c N \}$ . The layer output is

$$
y = \sum _ { e \in \mathcal { T } ( x ) } g _ { e } ^ { \prime } ( x ) E _ { e } ( x ) + \lambda \sum _ { e = 1 } ^ { m } E _ { e } ( x )\tag{6}
$$

where hyperparameter λ controls the weight of shared experts. We set $\lambda = 1$ in all experiments for simplicity, as our model is not sensitive to this choice. Additional details are provided in Appendix A.2.

Table 1 compares MoE designs trained on 12.8B samples from MetaCLIP [83]. We observe consistent improvements from the introduction of MoE, and the finegrained design leads to stronger performance.

Table 1: Comparison of diferent MoE designs in vision encoders.
<table><tr><td></td><td></td><td></td><td colspan="7">General Classification</td><td colspan="6">Image Retrieval</td></tr><tr><td></td><td>Actiited params Total</td><td>Params</td><td>As. Cas. ImmageNet</td><td>V1 [22]</td><td>IageNet [0] 60</td><td>b: oe 4]</td><td>Adv] 333 ImmageNet</td><td>Immaggeet</td><td>Ren nntoa]</td><td>Avygl Rvad</td><td>COO</td><td>[]1←] COO</td><td>[9]1←]</td><td>I30K [8] 1←]</td><td>I30K [88]←I</td></tr><tr><td>Model ViT-B/32 dense</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.1B</td><td>0.1B</td><td>55.8</td><td>67.5</td><td>59.3</td><td>50.6</td><td>27.8</td><td>73.7</td><td>58.4</td><td>36.3</td><td></td><td>54.4</td><td>63.3</td><td>79.6</td></tr><tr><td>ViT-B/32 conventional MoE</td><td>0.5B</td><td>0.1B</td><td>59.9</td><td>70.6</td><td>62.3</td><td>54.3</td><td>34.7</td><td>77.7</td><td></td><td>59.7</td><td>37.8</td><td>56.3</td><td>64.4</td><td>80.2</td></tr><tr><td>MoE-ViE-B/32</td><td>0.5B</td><td>0.1B</td><td>63.2</td><td>72.9</td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td></td><td>61.4</td><td>38.9</td><td>56.6</td><td>66.5</td><td>83.5</td></tr><tr><td>ViT-L/16</td><td>0.3B</td><td>0.3B</td><td>78.0</td><td>79.7</td><td>72.6</td><td>74.9</td><td>72.4</td><td>90.5</td><td>68.9</td><td></td><td>45.8</td><td>63.8</td><td>75.9</td><td>90.3</td></tr><tr><td>ViT-L/16 conventional MoE</td><td>1.7B</td><td>0.3B</td><td>78.5</td><td>80.2</td><td>72.9</td><td>73.5</td><td>74.9</td><td>90.8</td><td></td><td>69.0</td><td>45.5</td><td>63.4</td><td>76.1</td><td>90.8</td></tr><tr><td>MoE-ViE-L/16</td><td>1.7B</td><td>0.3B</td><td>79.6</td><td>80.7</td><td>74.1</td><td>76.5</td><td>75.4</td><td>91.5</td><td></td><td>69.3</td><td>46.0</td><td>63.8</td><td>76.3</td><td>91.0</td></tr></table>

Load Balancing Compared to prior works designing various auxiliary losses [54,61], optimized jointly with the main training objective, we propose a variant of the loss-free approach [77], which decouples the control of expert utilization from the training objective. In loss-free balancing, an extra bias term b is introduced to the router. Given routing logits s, the router now selects experts by

$$
g ( x ) = \mathrm { t o p } – k ( { \mathrm { S i g m o i d } } ( s ) + b )\tag{7}
$$

During training, the bias for expert e is updated using the observed token load. Let $t _ { e }$ be the number of tokens routed to expert e in the current iteration, and let $\mu _ { t }$ denote the mean token count across experts. The update rule is

$$
b _ { e } = b _ { e } - \alpha \ { \mathrm { s i g n } } ( t _ { e } - \mu _ { t } )\tag{8}
$$

with step size α. We notice, however, that the sign(·) update applies a constant magnitude correction (e.g., 1 or −1) regardless of deviation, which can cause oscillations when routing is close to balanced. Therefore, we propose to replace sign(·) with z-score, which scales the correction by the degree of imbalance and keeps it normalized for stability. The update of the bias term then becomes

$$
b _ { e } = b _ { e } - \alpha \frac { t _ { e } - \mu _ { t } } { \sigma _ { t } }\tag{9}
$$

where $\sigma _ { t }$ is the standard deviation of $\{ t _ { e } \}$ . We present ablations on diferent load balancing strategies in Section 4.4.

Scalability We then explore the scalability of MoE-ViE under contrastive pretraining. We train our models at multiple scales. At each scale, we train both the baseline dense model and MoE-ViE. All models are trained for 12.8B samples on MetaCLIP [83] at 224 × 224 resolution, and evaluated using ImageNet-1K classification accuracy [22]. Fig. 2 summarizes the results. Across all scales, MoE-ViE consistently outperforms its dense counterpart under the same compute budget, indicating a desired model capacity vs. compute trade-of. This motivates us to further scale up MoE-ViE as detailed in Section 4.1.

![](images/8e70bb125009c07c449413760fbbbfffa0054cc5a98c712da31694b0eb498b08.jpg)  
Fig. 2: MoE-ViE vs. dense models at multiple scales. MoE-ViE consistently outperforms their dense counterparts under the same compute budget.

## 2.3 Latency Optimization

Inherent MoE Eficiency Unlike dense networks where FLOPs scale linearly with total parameter count, MoE maintains constant active FLOPs per token by routing inputs to a small subset of experts. Thus, MoE theoretically ofers superior representation (via more parameters) without an inherent latency penalty, as its compute cost is governed only by the number of active parameters.

Optimized Kernel While MoE is theoretically compute-eficient, naive implementations (e.g., sequentially looping over experts in PyTorch) often underutilize GPUs, mainly due to three factors: (1) fragmenting compute into many small general matrix multiply (GEMM) operations reduces arithmetic intensity; (2) dynamic routing introduces CPU–GPU synchronization to determine per-expert workloads, creating idle hardware “bubbles”; (3) larger intermediate tensors amplify HBM trafic, making memory bandwidth the latency bottleneck.

To realize MoE’s theoretical eficiency in practice, we implement a specialized Triton [74] kernel with two key optimizations: First, Grouped GEMM aggregates all expert matrix multiplications (MatMuls) into a single, unified GPU task, enabling parallel processing across experts and eliminating repeated CPU–GPU handshakes. Second, Kernel Fusion combines MatMuls and non-linear activations (e.g., GeLU) into one pass, minimizing round-trips to HBM and reducing data movement. More details are included in Appendix B.

These optimizations are not auxiliary architectural tweaks. Instead, they are the necessary implementation mechanisms to faithfully translate MoE’s algorithmic eficiency into practical, hardware-eficient execution.

## 3 Robust Video Finetuning

Following [5], we first pretrain the model on image-text pairs and then finetune on video-text pairs to build a unified vision encoder for both image and video understanding. However, in our experiments, naively finetuning on video data substantially degrades image performance, indicating severe forgetting. Mixing image data into the video stage partially recovers image accuracy, but we find it also limits gains on video understanding (Section 4.4).

Recognizing that most VLMs use frame-based vision encoders to encode videos frame by frame [16,86], we regularize the video finetuning with frame-level distillation. We keep a copy of the image-pretrained model as a teacher. For a given video, we randomly sample a frame and feed it through the student and teacher vision towers to obtain logits S and T, and minimize the cosine distance

$$
\mathcal { L } _ { d } = 1 - \cos ( S , T )\tag{10}
$$

The final loss becomes

$$
\mathcal { L } = \mathcal { L } _ { c } + \beta \mathcal { L } _ { d }\tag{11}
$$

where $\mathcal { L } _ { c }$ denotes contrastive training loss, and $\beta$ is the weight of distillation.

We further reduce forgetting by freezing the MoE experts during video finetuning. This aligns with findings that MLP layers store models’ acquired knowledge, while attention layers are primarily responsible for feature interaction [24]. Since our vision encoder operates on individual frames, we do not expect video finetuning to require large changes to the per-frame visual vocabulary. Instead, the adaptation should mainly update how frame features are aggregated for video-specific alignment.

One issue is that $\mathcal { L } _ { c }$ updates both towers. So when finetuning on videos, the text tower may drift away from the alignment established during pretraining, while $\mathcal { L } _ { d }$ encourages the vision tower to stay close to its image-pretrained teacher, creating an optimization mismatch. We find that an eficient way to eliminate this is to freeze the MLP layers in the text tower as well. Our intuition is that the text tower has already trained on substantially larger and more diverse data, and our primary goal in this stage is to adapt the vision encoder to videos.

## 4 Experiments

We develop a series of MoE-ViE at various scales by building on top of the standard CLIP vision encoder architecture, where we replace the MLP modules in the vision tower with our proposed MoE blocks. We exclude the first transformer block from this replacement, as this first layer captures general, low-level, visual information, and we did not observe any gains by specializing it with MoE in our experiments.

## 4.1 Zero-Shot Benchmarks

Setup We conduct contrastive pretraining for MoE-ViE on 3.5B image-text pairs, comprised of 2B data from MetaCLIP [83], and 1.5B proprietary data. Unless otherwise specified, all of our MoE variants use 32 experts for each MoE layer to maintain a practical compute budget. Each expert is designed to be 1/4 of the corresponding dense MLP. We include more studies on the impact of the granularity in Appendix A.4 and on the number of experts in Appendix C.

We first train MoE-ViE-B and MoE-ViE-L models with 1/8 of experts activated, and then train an MoE-ViE-H with $1 / 4$ experts activated to explore further scaling. Following [5], we perform progressive resolution training, i.e., we increase image resolution as training progresses, for better training eficiency. After pretraining, we perform a short warmup using the pretraining data augmented with CC12M [8] and TreeOfLife [30]. For video finetuning, we uniformly sample 8 frames per video and use the curated video-text data from [5, 90]. Full training and evaluation details are provided in Appendices D.1 and D.2, respectively. Additional linear probing and LLM alignment results are reported in Appendix E.

Video Benchmarks We report the zero-shot video understanding capability of MoE-ViE in Table 2 on both classification and retrieval tasks. MoE-ViE achieves SOTA results at all evaluated scales, and even outperforms the larger $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G }$ model, which indicates that MoE scaling transfers efectively from image pretraining to our proposed video finetuning.

Table 2: Comparison of zero-shot video benchmarks between MoE-ViE and other previous SOTA vision encoders in the literature.
<table><tr><td></td><td>CIlas. Activted Params Total</td><td></td><td>Video Classification</td><td></td><td></td><td></td><td></td><td>Video Retrieval</td></tr><tr><td>Model SigLIP2-B/16 [75]</td><td>Params</td><td>Resion</td><td>A</td><td>400 38]</td><td>C68] [80] 003</td><td>HM 411]</td><td>Avgl Rvad</td><td>MS-TT [8]Λ←]</td><td>MSR-TT []L←8A4]</td></tr><tr><td> $\mathrm { P E _ { c o r e } B / i 6 \ [ 5 ] }$ </td><td>0.1B 0.1B</td><td>224</td><td>59.5</td><td>58.7</td><td>55.0</td><td>82.0 42.3</td><td>34.3</td><td>38.5</td><td>30.1</td></tr><tr><td>MoE-ViE-B/16</td><td>0.1B 0.1B 0.5B 0.1B</td><td>224 224</td><td>65.9 67.5</td><td>65.6 68.3</td><td>65.1 65.6</td><td>84.6 48.2 84.5 51.5</td><td>47.5 47.6</td><td>47.6 47.9</td><td>47.3 47.2</td></tr><tr><td>SigLIP2-L/16 [75]</td><td></td><td>384</td><td></td><td></td><td></td><td></td><td>36.5</td><td>41.5</td><td></td></tr><tr><td>PEcoreL/14 [5]</td><td>0.3B 0.3B 0.3B 0.3B</td><td>336</td><td>66.0 72.9</td><td>65.3 73.4</td><td>62.5 72.7</td><td>86.7 87.1</td><td>49.3 58.5 50.2</td><td>50.3</td><td>31.4 50.1</td></tr><tr><td>MoE-ViE-L/16</td><td>1.7B 0.3B</td><td>384</td><td>73.4</td><td>74.5</td><td>71.8</td><td>89.2</td><td>57.9 50.2</td><td>50.5</td><td>49.8</td></tr><tr><td>InternVL-C/16 [13]</td><td>5.5B 5.5B</td><td>224</td><td></td><td>69.1</td><td>68.9</td><td></td><td>42.5</td><td>44.7</td><td></td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B 1.1B</td><td>384</td><td>69.8</td><td>69.8</td><td>67.0</td><td>– 90.7</td><td>一 51.8 38.7</td><td>43.1</td><td>40.2 34.2</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B 1.9B</td><td>448</td><td>76.0</td><td>76.9</td><td>75.1</td><td>90.7</td><td>61.1 50.6</td><td>51.2</td><td>49.9</td></tr><tr><td>MoE-ViE-H/14</td><td>3.5B 1.1B</td><td>448</td><td>76.5</td><td>76.9</td><td>75.1</td><td>92.8</td><td>61.3 50.6</td><td>51.6</td><td>49.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

General Image Benchmarks Table 3 summarizes zero-shot performance on image classification and retrieval benchmarks. We compare MoE-ViE to both SOTA dense vision encoders and to prior CLIP MoE vision encoders. MoE-ViE achieves the best results at each scale among models with comparable sizes. In particular, MoE-ViE demonstrates consistently high accuracy on the challenging ImageNet-A [33] benchmark, with $+ 5 . 6 \%$ over $\mathrm { P E _ { c o r e } B , + 0 . 2 \% }$ over $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { L } .$

and even $+ 0 . 6 \%$ over $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G }$ which has 1.7× as many activated parameters as MoE-ViE-H.

Table 3: Comparison of zero-shot image classification and retrieval between MoE-ViE and other SOTA vision encoders in the literature.
<table><tr><td></td><td></td><td></td><td colspan="7">General Image Classification</td><td colspan="6">Image Retrieval</td></tr><tr><td></td><td>pParams Tottal</td><td>Ativated params</td><td>Resution</td><td>CIas. ImageNet A</td><td>VV] [2]</td><td>ImaggeNet [09] 6</td><td>bNetl</td><td>Adv] er333 IageNet</td><td>IageNet</td><td>Ren  2]</td><td>Avv. Rval COO</td><td>[] 1←6]</td><td>[7] L←6] COO</td><td>FI30K ] 1←]</td><td>FI30kK [88] L←1]</td></tr><tr><td>LIMOE-B/16 [54]</td><td>0.5B</td><td>0.1B</td><td>224</td><td></td><td>73.7</td><td>一</td><td>-</td><td>一</td><td>一</td><td></td><td>36.2</td><td>51.3</td><td></td><td></td><td></td></tr><tr><td>CLIP-UP-B/16 [79]</td><td>0.5B</td><td>0.2B</td><td>224</td><td>-</td><td>76.9</td><td>一</td><td></td><td>一</td><td>一</td><td>74.2</td><td>52.1</td><td></td><td>71.5</td><td>80.9</td><td>92.3</td></tr><tr><td>SigLIP2-B/16 [75]</td><td>0.1B</td><td>0.1B</td><td>224</td><td>74.0</td><td>78.2</td><td>71.4</td><td>73.6</td><td>55.0</td><td>91.7</td><td></td><td>73.7</td><td>52.1</td><td>68.9</td><td>80.7</td><td>93.0</td></tr><tr><td>PEcoreB/16 [5]</td><td>0.1B</td><td>0.1B</td><td>224</td><td>74.6</td><td>78.4</td><td>71.7</td><td>71.9</td><td>62.4</td><td>88.7</td><td></td><td>74.3</td><td>50.9</td><td>71.0</td><td>80.8</td><td>94.4</td></tr><tr><td>MoE-ViE-B/16</td><td>0.5B</td><td>0.1B</td><td>224</td><td>76.8</td><td>79.3</td><td>72.5</td><td>74.4</td><td>68.0</td><td>89.9</td><td></td><td>74.4</td><td>52.1</td><td>70.7</td><td>80.9</td><td>93.9</td></tr><tr><td>CLIP-MoE-L/14 [92]</td><td>0.9B</td><td>0.5B</td><td>336</td><td>-</td><td>74.6</td><td>68.5</td><td>33.5</td><td></td><td>-</td><td></td><td>53.6</td><td>46.8</td><td>65.0</td><td>42.1</td><td>60.5</td></tr><tr><td>LIMOE-L/16 [54]</td><td>1.7B</td><td>0.3B</td><td>224</td><td>-</td><td>78.6</td><td>1</td><td>-</td><td></td><td></td><td></td><td></td><td>39.6</td><td>55.7</td><td></td><td></td></tr><tr><td>CLIP-UP-L/14 [79]</td><td>1.7B</td><td>0.5B</td><td>224</td><td>I</td><td>81.2</td><td>-</td><td>-</td><td></td><td>- 1</td><td></td><td>75.5</td><td>53.9</td><td>73.8</td><td>82.0</td><td>92.4</td></tr><tr><td>SigLIP2-L/16 [75]</td><td>0.3B</td><td>0.3B</td><td>384</td><td>85.0</td><td>83.1</td><td>77.4</td><td>84.4</td><td>84.3</td><td>95.7</td><td></td><td>76.7</td><td>55.3</td><td>71.4</td><td>85.0</td><td>95.2</td></tr><tr><td>PEcoreL/14 [5]</td><td>0.3B</td><td>0.3B</td><td>336</td><td>86.1</td><td>83.5</td><td>77.9</td><td>84.7</td><td>89.0</td><td>95.2</td><td></td><td>78.8</td><td>57.1</td><td>75.9</td><td>85.5</td><td>96.6</td></tr><tr><td>MoE-ViE-L/16</td><td>1.7B</td><td>0.3B</td><td>384</td><td>86.2</td><td>83.6</td><td>77.9</td><td>85.0</td><td>89.0</td><td>95.4</td><td></td><td>78.8</td><td>57.2</td><td>75.9</td><td>85.7</td><td>96.5</td></tr><tr><td>EVA 18B/14 [70]</td><td>17.5B</td><td>17.5B</td><td>224</td><td>85.4</td><td>83.8</td><td>77.9</td><td>82.2</td><td>87.3</td><td>95.7</td><td></td><td>77.5</td><td>56.2</td><td>73.6</td><td>83.3</td><td>96.7</td></tr><tr><td>InternVL-C/16 [13]</td><td>5.5B</td><td>5.5B</td><td>224</td><td>84.1</td><td>83.2</td><td>77.3</td><td>80.6</td><td>83.8</td><td>95.7</td><td></td><td>78.6</td><td>58.6</td><td>74.9</td><td>85.0</td><td>95.7</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>384</td><td>88.0</td><td>85.0</td><td>79.8</td><td>88.0</td><td>90.5</td><td>96.6</td><td></td><td>77.6</td><td>56.1</td><td>72.8</td><td>86.0</td><td>95.4</td></tr><tr><td>PEcoreG/14 [5]</td><td>1.9B</td><td>1.9B</td><td>448</td><td>88.6</td><td>85.4</td><td>80.2</td><td>88.2</td><td>92.6</td><td>96.5</td><td></td><td>78.9</td><td>58.1</td><td>75.4</td><td>85.7</td><td>96.2</td></tr><tr><td>MoE-ViE-H/14</td><td>3.5B</td><td>1.1B</td><td>448</td><td>88.3</td><td>85.1</td><td>80.0</td><td>87.0</td><td>93.2</td><td>96.2</td><td></td><td>78.2</td><td>56.8</td><td>74.6</td><td>85.4</td><td>96.0</td></tr></table>

Fine-grained Image Benchmarks Table 4 presents the results for image classification on fine-grained categories. We see MoE-ViE excels in these tasks. Not only do our MoE-ViE models achieve SOTA results compared with similarly sized models, here, again, our largest model MoE-ViE-H outperforms $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G }$ whose activated size is 1.7× larger. Additionally, our model demonstrates high accuracy on OCR benchmarks, as evaluated on TextCaps [65]. We attribute these achievements to the nature of MoE, applied with the correct topology, enabling experts to specialize in diferent tasks.

## 4.2 VLM Alignment

Since VLMs represent the primary downstream application of vision encoders, evaluating alignment quality is essential to verify that improvements in visual representations translate to gains on end-user tasks such as visual question answering, video understanding, and captioning. Our alignment training follows a standard three-stage pipeline. In Stage 1 (warmup, 8k steps), we train only the two-layer MLP projectors while keeping all other parameters frozen. In Stage 2 (pre-alignment, 15k steps), we explore various training strategies, and find that freezing the encoder parameters for the first 1k iterations to stabilize the loss before transitioning to end-to-end training of all parameters yields the best downstream performance. In Stage 3 (supervised finetuning, 8k steps), we train all parameters of the model. Results are reported in Table 5 and Table 6 with Llama 3.1 Instruct 8B and Qwen 2.5 VL 7B, respectively. We use the tile-based approach [17, 78] to support dynamic resolutions.

Table 4: Comparison of zero-shot fine-grained image classification and OCR benchmarks between MoE-ViE and SOTA dense vision encoders. We re-evaluated part of the OCR results for SigLIP2 and PE if not reported in [75] and [5].
<table><tr><td>Model</td><td>Actiated Total Params A</td><td>Resltion paraams</td><td>Fine-grained Img Classification</td><td>CIas. 6 F101</td><td>[99] Iowers</td><td>Cuu] C773]</td><td>[64] Airccfts</td><td>[40] ar</td><td>OR AAveage</td><td>OCR</td><td>Cap [99] L←I</td><td>Cap [6] Tx) I←I )</td></tr><tr><td>SigLIP2-B/16 [75] PEcoreB/16 [5]</td><td>0.1B</td><td>0.1B</td><td>224</td><td>69.2</td><td>92.8</td><td>85.7</td><td></td><td>19.2</td><td>54.8</td><td>93.4</td><td>71.5 72.3</td><td>70.7</td></tr><tr><td>MoE-ViE-B/16</td><td>0.1B 0.5B</td><td>0.1B 0.1B</td><td>224</td><td>71.7 73.5</td><td>92.5</td><td>86.5</td><td>30.5</td><td>57.0</td><td>92.1</td><td></td><td>71.6 72.3</td><td>70.9</td></tr><tr><td>SigLIP2-L/16 [75]</td><td></td><td></td><td>224</td><td></td><td>94.2</td><td>87.3</td><td>34.9</td><td>58.2</td><td>93.1</td><td>72.5</td><td>72.7</td><td>72.3</td></tr><tr><td></td><td>0.3B</td><td>0.3B</td><td>384</td><td>76.1</td><td>96.1</td><td>90.0</td><td>31.6</td><td>67.0</td><td>95.8</td><td>79.2</td><td>80.2</td><td>78.2</td></tr><tr><td> $\mathrm { P E _ { c o r e } L / i 4 \ [ 5 ] }$ </td><td>0.3B</td><td>0.3B 0.3B</td><td>336</td><td>78.1 78.9</td><td>96.2</td><td>87.2</td><td>45.6</td><td>67.8</td><td>93.7</td><td></td><td>79.2</td><td>79.8 78.5</td></tr><tr><td> $\mathbf { M o E - V i E - L } / 1 6$ </td><td>1.7B</td><td></td><td>384</td><td></td><td>96.7</td><td>89.2</td><td>47.8</td><td>65.7</td><td>94.9</td><td></td><td>79.7 80.0</td><td>79.3</td></tr><tr><td>EVA 18B/14 [70]</td><td>17.5B</td><td>17.5B</td><td>224</td><td>75.9</td><td>95.8</td><td>86.0</td><td>43.1</td><td>59.7</td><td>94.9</td><td>、</td><td>–</td><td>一</td></tr><tr><td>InternVL-C/16 [13]</td><td>5.5B</td><td>5.5B</td><td>224</td><td>72.8</td><td>95.3</td><td>85.8</td><td>35.1</td><td>53.3</td><td>94.4</td><td></td><td></td><td>72.3</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>384</td><td>79.6</td><td>97.0</td><td>91.5</td><td>40.1</td><td>73.6</td><td>95.9</td><td></td><td>79.8 80.3</td><td>79.2</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>1.9B</td><td>448</td><td>83.8 83.9</td><td>96.9</td><td>91.4</td><td>57.6</td><td>78.2</td><td>94.7</td><td></td><td>79.1 79.3</td><td>78.8</td></tr><tr><td>MoE-ViE-H/14</td><td>3.5B</td><td>1.1B</td><td>448</td><td></td><td>97.1</td><td>92.4</td><td>54.3</td><td>80.1</td><td>95.5</td><td>80.4</td><td>80.6</td><td>80.2</td></tr></table>

Table 5: Alignment comparison across models. Columns are grouped by task type. Llama 3.1 Instruct 8B [29] is used as the base LLM, and we use a maximum of 4 tiles.
<table><tr><td></td><td></td><td colspan="6">Image</td><td colspan="4">Video</td><td colspan="3">Captioning</td></tr><tr><td>Model</td><td>Activivated params</td><td>Avega. Iges</td><td>[1] 99]</td><td>T] GL]</td><td>Ca] 51]</td><td>D 53]</td><td>I1 ° 52]</td><td>Av deo</td><td>Vi 2]</td><td></td><td>MV] 44</td><td>[] Eo50</td><td>AVv&amp; Ca.</td><td>[O0] C46]</td><td> a2]</td></tr><tr><td>MetaCLIP-G/14 [83] SigLIP2-g-opt/16 [75]</td><td>1.8B</td><td>61.3</td><td>72.8</td><td>65.4</td><td>68.1</td><td>61.3</td><td>39.1</td><td>45.4</td><td>46.5</td><td>44.7</td><td>45.0</td><td>125.5</td><td>134.4</td><td>116.5</td></tr><tr><td></td><td>1.1B</td><td>59.0</td><td>72.4</td><td>70.3</td><td>63.1</td><td>55.3</td><td>34.0</td><td>49.5</td><td>46.2</td><td>48.5</td><td>53.8</td><td>129.1</td><td>137.8</td><td>120.3</td></tr><tr><td>PEcoreG/14 [5]</td><td>1.9B</td><td>69.2</td><td>69.7</td><td>74.3</td><td>73.4</td><td>81.2</td><td>47.6</td><td>48.6</td><td>46.0</td><td>48.7</td><td>51.2</td><td>123.7</td><td>134.5</td><td>112.9</td></tr><tr><td>InternViT2.5/14 [12]</td><td>5.5B</td><td>68.1</td><td>72.9</td><td>71.3</td><td>74.6</td><td>74.3</td><td>47.6</td><td>48.7</td><td>46.0</td><td>49.6</td><td>50.6</td><td>123.0</td><td>132.5</td><td>113.5</td></tr><tr><td>AIMv2 3B/14 [26]</td><td>2.7B</td><td>69.8</td><td>72.2</td><td>79.2</td><td>73.0</td><td>78.2</td><td>46.5</td><td>49.7</td><td>49.6</td><td>49.9</td><td>49.6</td><td>130.6</td><td>139.7</td><td>121.5</td></tr><tr><td>MoE-ViE-H/14</td><td>1.1B</td><td>75.2</td><td>84.9</td><td>75.7</td><td>74.2</td><td>86.1</td><td>55.0</td><td>58.9</td><td>52.0</td><td>63.6</td><td>61.2</td><td>127.5</td><td>135.4</td><td>119.6</td></tr></table>

As shown in Table 5 and Table 6, MoE-ViE-H achieves strong performance across all evaluation axes. On both image and video benchmarks, our model outperforms all baselines within the same parameter class and even surpasses SOTA models with several times more activated parameters. In captioning, although we do not achieve the highest scores, our results remain highly competitive: among models with fewer than 2B activated parameters, MoE-ViE-H is second only to SigLIP2-g-opt. We provide more analysis on alignment in Appendix E.2.

Table 6: Alignment comparison across models. Columns are grouped by task type. Qwen 2.5 VL 7B is used as the base LLM, and we use a maximum of 4 tiles.
<table><tr><td></td><td></td><td colspan="6">Image</td><td colspan="4">Video</td><td colspan="3">Captioning</td></tr><tr><td></td><td>Activted pParams</td><td>Avgag. Iges</td><td>[6ε] AI2D</td><td>JL] VVGGL]</td><td>Cha C51]</td><td>D Λ65]</td><td>[n  52]</td><td>A. Vdeo</td><td>Vide Vd2]</td><td>MV] B44[]</td><td>o] Eo50</td><td>AVv Ca.</td><td>[0] 46]</td><td></td><td>[] a2]</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>62.7</td><td>75.2</td><td>70.3</td><td>71.0</td><td>60.4</td><td>36.7</td><td>55.6</td><td>52.0</td><td>52.8</td><td>62.0</td><td>130.1</td><td>139.0</td><td>121.1</td></tr><tr><td> $\mathrm { P \bar { E } _ { c o r e } G / 1 4 \ [ \bar { 5 } ] }$ </td><td>1.9B</td><td>65.4</td><td>72.9</td><td>67.9</td><td>75.9</td><td>68.8</td><td>41.6</td><td>54.1</td><td>48.7</td><td>52.9</td><td>60.8</td><td>123.8</td><td>135.2</td><td>112.3</td></tr><tr><td>InternViT2.5/14 [12]</td><td>5.5B</td><td>65.4</td><td>73.6</td><td>70.1</td><td>78.2</td><td>65.3</td><td>39.6</td><td>52.7</td><td>50.3</td><td>51.1</td><td>56.6</td><td>128.5</td><td>138.4</td><td>118.6</td></tr><tr><td>AIMv2 3B/14 [26]</td><td>2.7B</td><td>67.6</td><td>75.2</td><td>74.2</td><td>76.7</td><td>70.5</td><td>41.4</td><td>52.7</td><td>45.9</td><td>51.4</td><td>60.8</td><td>130.6</td><td>139.2</td><td>122.0</td></tr><tr><td>MoE-ViE-H/14</td><td>1.1B</td><td>74.0</td><td>86.5</td><td>72.3</td><td>73.0</td><td>84.1</td><td>54.0</td><td>59.2</td><td>53.4</td><td>61.5</td><td>62.8</td><td>129.7</td><td>138.6</td><td>120.8</td></tr></table>

## 4.3 Latency Comparison

With our kernel optimization presented in Section 2.3, we achieve inference latency comparable to a dense model (SigLIP2-g-opt) with a similar number of active parameters, despite possessing a significantly larger pool of trainable parameters. As shown in Table 7, the MoE kernel provides > 2.5× speedup across diferent batch sizes compared to the vanilla implementation. Our MoE-ViE runs at roughly 76% of latency compared to $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G } / 1 4$ , the 1.7× larger SOTA dense vision encoder, while demonstrating comparable performance on accuracy. This confirms that the representational gains of our proposed MoE-ViE are achieved without any unexpected latency regressions. Additionally, we further demonstrate that such eficiency gain is naturally translated to downstream VLMs, detailed in Appendix E.3.

Table 7: Performance comparison between MoE-ViE and dense vision encoders, across diferent inference batch sizes (bsz ). Due to diference in patch size, we report latency based on the same number of input tokens (# tokens = 576) for a fair comparison.
<table><tr><td></td><td>Activated Parameters Parameters</td><td>Total</td><td colspan="4">Latency (ms)</td></tr><tr><td>Model</td><td></td><td></td><td> $\pmb { b } \pmb { z } = \pmb { 1 6 }$ </td><td>32</td><td>64</td><td>128</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>76.39</td><td>141.78 275.94</td><td></td><td>549.62</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ \dot { 5 } ] }$ </td><td>1.9B</td><td>1.9B</td><td>101.21</td><td>188.76 366.23</td><td></td><td>708.62</td></tr><tr><td>MoE-ViE-H/14 (Vanilla Implementation)</td><td>1.1B</td><td>3.5B</td><td>318.76</td><td>448.93 740.84 1306.81</td><td></td><td></td></tr><tr><td>MoE-ViE-H/14 (Optimized Kernel)</td><td>1.1B</td><td>3.5B</td><td>82.59</td><td>145.82 276.87</td><td></td><td>544.96</td></tr></table>

## 4.4 Ablation Study

Auxiliary Loss Table 8 compares diferent expert balancing algorithms. We conduct these experiments with MoE-ViE-B with patch size 32, which is faster to train, but exhibits the same routing dynamics as other MoE-ViE variants. As seen, the loss-free approaches (i.e., [77] and ours) outperform other forms on all benchmarks. This aligns with our expectation, since expert utilization is achieved without directly perturbing optimization under the contrastive training objective. Our proposed magnitude-aware method further improves over [77], indicating a more stable and efective balancing control.

Table 8: Comparison of diferent auxiliary losses for training CLIP vision encoders.
<table><tr><td></td><td colspan="5">General Classification 4</td><td colspan="4">Image Retrieval</td></tr><tr><td>Auxiliary loss</td><td>As. Cs. ImmageNet</td><td>ImageNet [] [2]</td><td>cel [0] 60</td><td>Adve] err33 IagNet</td><td>Renns 22 ImageNet</td><td>Avgsv Rval</td><td>COCO [9]1←]</td><td>COCCO [9]L←]</td><td>FI30K [8] I←]</td><td>Ii30K [8] L←1]</td></tr><tr><td>Dense model</td><td>55.8</td><td>67.5</td><td>59.3</td><td>50.6</td><td>27.8</td><td>73.7</td><td>58.4</td><td>36.3</td><td>54.4 63.3</td><td>79.6</td></tr><tr><td>Importance and load loss [61]</td><td>59.9</td><td>70.6</td><td>62.3</td><td>54.3</td><td>34.7</td><td>77.7</td><td>59.7</td><td>37.8</td><td>56.3 64.4</td><td>80.2</td></tr><tr><td>Entropy loss [54]</td><td>59.9</td><td>70.7</td><td>62.2</td><td>53.9</td><td>35.5</td><td>77.1</td><td>60.0</td><td>38.5</td><td>56.4 65.0</td><td>80.1</td></tr><tr><td>Loss-free [77]</td><td>62.6</td><td>72.5</td><td>64.4</td><td>56.4</td><td>40.0</td><td>79.9</td><td>60.9</td><td>39.2</td><td>56.8 65.3</td><td>82.5</td></tr><tr><td>Ours</td><td>63.2</td><td>72.9</td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td>61.4</td><td>38.9</td><td>56.6 66.5</td><td>83.5</td></tr></table>

Video Finetuning We ablate the video finetuning components introduced in Section 3 in Fig. 3, using MoE-ViE-H. For clarity, we report ImageNet-1K as a representative image benchmark and K400 as a representative video benchmark.

As shown in Fig. 3a, while naive video finetuning on video data improves video performance, it causes a significant degradation on image benchmarks over time. Mixing image data into the video finetuning partially mitigates this issue, but fails to prevent degradation, and limits our gains on video understanding. We find that applying frame-level distillation without freezing the MLP layers in the text tower sharply hurts image performance. Finally, we see that, individually, both distillation and expert freezing help preserve image performance, and that their combination achieves the best results on both image and video benchmarks.

## 4.5 Expert Specialization

We conduct both qualitative and quantitative analysis of MoE-ViE’s expert activation patterns to better understand how fine-grained MoE captures complex, multi-faceted visual information. Across both tokens and tasks, we observe clear and meaningful expert specialization.

We first project each layer’s features to three principal components using PCA, and map the resulting 3D embeddings to RGB, as in [5, 66]. For each expert, we visualize the tokens routed to it, as shown in Fig. 4. Since we use topk routing, these subplots are not mutually exclusive, i.e., a token may appear in multiple expert-specific views. We also illustrate the subplot for the shared expert, which receives all tokens, preserving global context.

![](images/cb82ea84e453755c21be720cfa4954985ae11882ed20090911224097311750d0.jpg)

![](images/8d43beb82afc3cfb37132317a0d37fd6a0af67706da759887933f0328f0e4c21.jpg)

![](images/220c68db1a3100a52c23c4ebdcd7902ed4e2ff20922980c2a423b95cbe3c5f81.jpg)

![](images/46abd84cd33d8cbd5037b39085b72979b946c79b3e3a8d7f821bd5cc1717c34c.jpg)  
(a)

![](images/584008667012d11f950952653518e344008b33a7f1fa13dbc8d303103addfdae.jpg)  
(b)

![](images/10c3afbfaa22647075e5a1feac11cb8e93c59f956d9016e1a157da7eddcd089d.jpg)  
(c)  
Fig. 3: Ablations of diferent video finetuning methods. (a) Our proposed distillation and freezing techniques outperform finetuning with vanilla video data, or mixed image and video data; (b) We observe significant degradation if we apply frame-level distillation without freezing the text tower; (c) The combination of the proposed distillation and freezing methods yields the best results.

Such visualization reveals several compelling insights: (1) each expert extracts distinct, semantically meaningful aspects of the image; (2) these specializations are complementary, such that their combined outputs yield richer visual understanding. For example, in Fig. 4a, expert 11 predominantly activates on tokens corresponding to the dog’s ears, while expert 18 emphasizes the eyes and nose. In Fig. 4b, expert 17 highlights the plate, whereas expert 19 concentrates on the food. In both examples, expert 29 consistently activates on background regions.

We further quantify expert specialization through per-task expert utilization heatmaps (Fig. 5) and routing entropy (Table 9). As shown in Fig. 5, experts exhibit distinct activation profiles across tasks, consistent with the emergence of task-relevant specialization in deeper MoE layers. Interestingly, on domainspecific benchmarks (e.g., FGVC Aircraft, VTAB Flowers), activations are more imbalanced across experts than on more general benchmarks such as ImageNet or COCO. This concentration may help explain MoE-ViE’s stronger performance on fine-grained image benchmarks. To complement this analysis, we measure routing entropy across tasks and layers, summarized in Table 9. Entropy consistently decreases with depth, confirming higher specialization at later layers. Moreover, benchmarks spanning narrower visual domains (VTAB Flowers, FGVC Aircraft) exhibit lower entropy overall, indicating stronger semantic specialization.

![](images/b193d5777d8bee726b709c635a021193455791fa93b63bf02afb1ef0233d74d1.jpg)

![](images/127e14632256a30bec5793cfb3415bd18cb1ecab623c556d70a021046f267552.jpg)

![](images/5809d47a96d7d5e57d22105769e61e41f72317a6b13afa20d28a802c473034ec.jpg)

![](images/b4e64e0ec8b65bf0e01c9b5a7a7a9cb4c014fbb4735196d2a824897b7287da98.jpg)  
(a) From left to right: raw image, shared expert, experts 11, 18, 22, 29

![](images/a7e95b5601a615317d9e34b7b77025c539e189de473741b7cf7b314d729278b1.jpg)

![](images/3d86394e36799f4586cf02ecad7ab8a18fe210dd955fb3c4033af87bfd054e30.jpg)

![](images/3f8a43a424e8536690a11814f707e5f14a5c841bbdb296b63ced2f8f5d02ab84.jpg)

![](images/3e9923a27796bce3bb34d2b38679d349a0429d1826a49c58bc40a0fe5623c7ca.jpg)

![](images/17041a9b2a1c396ce2c289d440e52fb23463fd429fb0770692b3164b49947351.jpg)

![](images/d5fd308421c6e9cd122a8980fac945381e4ab5a94637a7bf9c8a3f703292e766.jpg)  
(b) From left to right: raw image, shared expert, experts 11, 17, 19, 29

![](images/7df5361a6ad82be3ed95cbd200f7095e2331a8723a934e074c82e5b8aa46514b.jpg)

![](images/12585f16d272de0c11759a1699088bc9720b09a60a349b105a09f70e791aed29.jpg)

Fig. 4: MoE Expert activations in layer 20 of MoE-ViE-H/14. Experts specialize for distinct, semantically meaningful, aspects of the image.  
![](images/ac4109485bb911fd4c5c52ee8937e45f24164d47bec906b2388416384b435dd6.jpg)  
(a) MoE layer 29

![](images/891497cedb409f24896d3fea1ea66bb61a589c923b7b6f1b40aa208162755a9c.jpg)  
(b) MoE layer 30

![](images/b7aa1644b5cb9f1b98ad11859fb6a7e1e3dc027c634c5e220918a3be7f05c91c.jpg)  
(c) Last MoE layer, layer 31  
Fig. 5: Expert activation across all data for a given task, final layers of MoE-ViE-H/14.

Table 9: Routing entropy averaged over layer ranges.
<table><tr><td>Benchmark</td><td>L0-5</td><td>L6-15</td><td>L16-25</td><td>L26-30</td><td>Uniform</td></tr><tr><td>VTAB Flowers [56]</td><td>3.20</td><td>3.17</td><td>2.83</td><td>2.30</td><td>3.47</td></tr><tr><td>FGVC Aircraft [49]</td><td>3.19</td><td>3.26</td><td>2.95</td><td>2.56</td><td>3.47</td></tr><tr><td>ImageNetV2 [60]</td><td>3.34</td><td>3.31</td><td>3.30</td><td>3.21</td><td>3.47</td></tr><tr><td>Flickr30k [88]</td><td>3.28</td><td>3.30</td><td>3.23</td><td>3.07</td><td>3.47</td></tr><tr><td>COCO [46]</td><td>3.30</td><td>3.32</td><td>3.30</td><td>3.18</td><td>3.5</td></tr></table>

## 5 Related Work

## 5.1 Contrastive Language-Image Pretraining

Contrastive vision-language pretraining (CLIP) has emerged as a foundational approach for building strong and transferable vision encoders [35, 59]. A number of extensions refine either its training objective or supervision signals. For instance, SigLIP [91] replaces the Softmax used to compute InfoNCE loss [57] with a Sigmoid function, enabling better scaling and eficiency. MaskCLIP [23] augments masked image modeling to train the vision model as a self-distillation. CoCa [89] incorporates caption loss in contrastive training, and LocCa [76] further introduces grounding supervision. SigLIP 2 [75] and PE [5] integrate several of these ingredients, achieving high accuracy on zero-shot image and video benchmarks.

## 5.2 Mixture-of-Experts (MoE)

MoE scales model capacity by replacing dense layers with an expert router and a set of expert layers. The learned expert router activates a small subset of experts per input, yielding sparse input-dependent computation. The idea traces back to conditional computation with expert gating formulations [9, 34, 37], and has since become a practical mechanism for scaling transformers via sparse top-k routing [63]. MoE has been extensively studied in natural language processing (NLP) [25,42], and has shown great success in many recent large language models [1,36,47,72,85]. Early MoE for vision eforts largely follow supervised training strategies [10,61,81] or introduce MoE during the post-training stages [64,80,82]. Later works have introduced MoE into CLIP pretraining [59]. LIMoE [54] pioneered contrastive learning with sparsely activated experts and demonstrated improved zero-shot image benchmarks. CLIP-MoE [92] and CLIP-UP [79] introduce upcycling techniques to convert pretrained dense CLIP models to MoE variants. Despite these advances, prior MoE vision encoders have not matched performance of SOTA dense models.

## 6 Conclusion

In this work, we present MoE-ViE, an eficient and performant vision encoder with fine-grained MoE, for image and video understanding. By combining our proposed architectural design, kernel optimization, magnitude-aware loss-free load balancing control, and refined video finetuning, MoE-ViE outperforms both prior dense and MoE vision encoders across image and video benchmarks at all scales. Our largest model achieves comparable results to a SOTA dense encoder that is 1.7× larger while running at 76% of its latency, demonstrating the significant potential for further efectively scaling up vision encoders.

## Acknowledgements

We would like to thank Giang Nguyen, Jiankun Liu, Zhuangqun Huang, Tuyen Tran, Xiao Zhang, Parisa Hassanzadeh, Lambert Mathias, Po-Yao Huang, Daniel Bolya for their contributions, support and discussions for this project.

## A Architecture Design Ablations

This section presents additional analyses of our MoE design, including our study on shared experts, activated experts, and the expert width. Unless otherwise noted, we train MoE-ViE-B with patch size 32 on 12.8B samples from MetaCLIP [83].

## A.1 Number of Shared Experts

Table 10 compares models with diferent numbers of shared experts. To keep the compute budget fixed, we hold the total number of experts constant (32 in this case) and trade of shared experts against sparsely routed experts. We see that introducing shared experts generally improves performance over a fully sparse routing configuration. However, increasing the number of shared experts beyond a small value yields no further gains and can even degrade results. Therefore, we use a single shared expert in all experiments for simplicity.

Table 10: Ablation on the number of shared experts.
<table><tr><td></td><td colspan="6">General Classification</td><td colspan="6">Image Retrieval</td></tr><tr><td># of shared experts</td><td>AIs. Cas. Imageet</td><td>[V]1 [2]</td><td>Iageet [09] 60</td><td>Ob::Gec  4]</td><td>Imageet</td><td>Adve rr33]</td><td>Renn  ] ImageNet</td><td>Avygvl Rval</td><td>[] 1←6] COO</td><td>[9]L←] COO</td><td>Ii30kK [88]1←</td><td>EI30K [88]←I</td></tr><tr><td>0</td><td>62.1</td><td>72.5</td><td>64.2</td><td>56.1</td><td>38.4</td><td>79.5</td><td>61.1</td><td>38.8</td><td>56.4</td><td>66.2</td><td>83.0</td></tr><tr><td>1</td><td>63.2</td><td>72.9</td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td>61.4</td><td>38.9</td><td>56.6</td><td>66.5</td><td>83.5</td></tr><tr><td>24</td><td>63.0</td><td>73.2</td><td>64.5</td><td>57.9</td><td>39.7</td><td>79.9</td><td>61.3</td><td>39.1</td><td>56.4</td><td>66.3</td><td>83.6</td></tr><tr><td></td><td>62.5</td><td>72.6</td><td>64.2</td><td>57.2</td><td>38.8</td><td>79.5</td><td>60.2</td><td>37.5</td><td>56.5</td><td>65.0</td><td>81.7</td></tr></table>

## A.2 Weight of Shared Experts

We sweep the scaling factor applied to the shared expert. Table 11 summarizes the results. Our model performance is largely insensitive to this weight across a broad range, suggesting that it is not a critical hyperparameter. We hypothesize that the scaling mainly afects the initial state. Further training can always adjust model parameters to reach a comparable solution.

Table 11: Ablation on the weight of shared experts.
<table><tr><td></td><td colspan="5">General Classification</td><td></td><td colspan="5">Image Retrieval</td></tr><tr><td>λ</td><td>AV. CIs.</td><td>Immageet</td><td>ImmageNet V] [2]</td><td>[09] 6</td><td>Ob::Gct  4]</td><td>AAdv] rerr33] Iaget</td><td>Rens a] ImaggeNet</td><td>Avv vadl</td><td>COO [9]1←]</td><td>COO [9]L←]</td><td>I30kK [88]I←]</td><td>FIi30K []←]</td></tr><tr><td>10</td><td>63.0</td><td>72.5</td><td>64.7</td><td></td><td>57.4</td><td>40.5</td><td>80.1</td><td>61.3</td><td>38.6</td><td>56.2</td><td>66.8</td><td>83.5</td></tr><tr><td>1</td><td>63.2</td><td>72.9</td><td></td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td>61.4</td><td>38.9</td><td>56.6</td><td>66.5</td><td>83.5</td></tr><tr><td>0.1</td><td>63.1</td><td>73.0</td><td>64.5</td><td></td><td>57.8</td><td>40.1</td><td>79.8</td><td>61.1</td><td>39.1</td><td>56.3</td><td>66.3</td><td>82.8</td></tr><tr><td>0.01</td><td>62.9</td><td>72.6</td><td>64.5</td><td></td><td>57.2</td><td>40.4</td><td>80.0</td><td>61.1</td><td>38.6</td><td>56.4</td><td>66.7</td><td>83.0</td></tr></table>

## A.3 Choice of Top-k

We study the efect of the number of activated experts $k ,$ summarized in Table 12, using a total of 32 fine-grained experts. Performance improves only marginally beyond $k = 8 ,$ indicating diminishing returns from additional expert activation. Based on this observation, we set $k \in \{ 4 , 8 \}$ depending on model size and compute budget, striking a favorable balance between capacity and eficiency.

Table 12: Ablations of top-k.
<table><tr><td colspan="4"></td><td colspan="3">General Classification</td><td></td><td colspan="3">Image Retrieval</td></tr><tr><td>k</td><td>AIas. Cas.</td><td>Immageet</td><td>Immageet V]1[2]</td><td>[0] 6</td><td>Ob:: Gct 44]</td><td>AAdv] r33] Iaget</td><td>Renm 32] Iaget</td><td>Avye Rval</td><td>COCO [9]1←]</td><td>COCO [9]L←]</td><td>Ii30k [88]I←] Ii30k [88]←</td></tr><tr><td>48</td><td>62.4</td><td>72.6</td><td>64.3</td><td>56.9</td><td>38.0</td><td>80.0</td><td>61.0</td><td>38.4</td><td>56.0</td><td>66.3</td><td>83.2</td></tr><tr><td></td><td>63.2</td><td>72.9</td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td>61.4</td><td></td><td>38.9</td><td>56.6 66.5</td><td>83.5</td></tr><tr><td>16</td><td>63.1</td><td>73.0</td><td>64.8</td><td>57.5</td><td>40.3</td><td>79.8</td><td>61.4</td><td>39.1</td><td>56.5</td><td>66.3</td><td>83.6</td></tr></table>

## A.4 Expert Width

We sweep the expert width factor $\textit { c } \left( i . e . \right.$ , each expert has $\textstyle { \frac { 1 } { c } } \times$ the dense MLP width) to study the efect of expert granularity while keeping the total expert capacity fixed at 8× the MLP size. Finer granularity consistently improves performance across all benchmarks, consistent with our qualitative and quantitative analysis in Section 4.5. However, gains saturate below $\textstyle { \frac { 1 } { 4 } }$ width, suggesting a practical lower bound on useful expert decomposition.

Table 13: Ablations of expert width factor c.
<table><tr><td colspan="6">General Classification</td><td colspan="4">Image Retrieval</td></tr><tr><td>C</td><td>A. Cs. Imageet</td><td>V] [2]</td><td>Imageet [] 60]</td><td>Ob::Gec 4]</td><td>Adve er3] IagNet</td><td>Rens t 2] IageNet</td><td>Avgva Rval COO</td><td>[]1←] COO</td><td>FIi30k []L←]</td><td>[88]I←] FIi30k [8]L←I]</td></tr><tr><td></td><td>59.9</td><td>70.6</td><td>62.3</td><td>54.3</td><td>34.7</td><td>77.7</td><td>59.7</td><td>37.8</td><td>56.3</td><td>64.4 80.2</td></tr><tr><td>1248</td><td>61.0</td><td>71.1</td><td>62.6</td><td>55.0</td><td>37.2</td><td>78.9</td><td>60.3</td><td>38.0</td><td>56.4 64.9</td><td>82.0</td></tr><tr><td></td><td>63.2</td><td>72.9</td><td>64.9</td><td>57.7</td><td>40.4</td><td>80.0</td><td>61.4</td><td>38.9</td><td>56.6 66.5</td><td>83.5</td></tr><tr><td></td><td>62.8</td><td>72.7</td><td>64.2</td><td>57.9</td><td>39.6</td><td>79.5</td><td>61.0</td><td>39.0</td><td>56.1 65.9</td><td>83.1</td></tr></table>

## B MoE Kernel

## B.1 Limitation of Vanilla MoE Implementation

The vanilla implementation of MoE processes one expert at a time in a Python for loop. While straightforward to implement, it has several performance limitations. First, processing experts sequentially fragments what could be a single large GEMM into many smaller, often highly imbalanced GEMMs. These small GEMMs exhibit low arithmetic intensity and insuficient parallelism, leading to poor GPU utilization and higher latency. Further, dynamic routing requires per-expert index selection to identify routed tokens, followed by separate gather/GEMM/activation/scatter launches for each expert. The resulting CPU-GPU synchronization and kernel-launch overhead can dominate latency when each expert processes only a small number of tokens. In addition, each iteration of the loop materializes multiple intermediate tensors, causing repeated reads and writes to HBM at each step. Moreover, the input is re-gathered per active expert (up to K times under top-K routing), creating redundant memory trafic that often makes bandwidth, instead of compute, the latency bottleneck.

## B.2 Optimized MoE Kernels

As described in Section 2.3, we implement specialized kernels to address each of the above bottlenecks. The optimized implementation (Algorithm 1) uses two custom GPU kernels written in Triton [74], a compiler for writing eficient GPU programs in Python. Algorithm 1 details the resulting procedure. The two optimizations mentioned in Section 2.3 map to the algorithm as follows:

1. Grouped GEMM. Instead of issuing multiple separate small GEMMs in a loop, the optimized kernel fuses all per-expert GEMMs into a single grouped operation executed in one GPU launch, using a 3D grid that parallelizes over tokens, output columns, and experts to improve arithmetic intensity and GPU utilization. A jagged layout handles variable per-expert token counts without padding, and GPU-side Sort/CumSum replaces per-expert Where queries, removing the loop and CPU-GPU synchronization so the forward pass completes in only two kernel launches.

2. Kernel Fusion. Each kernel fuses multiple steps to reduce HBM trafic. Kernel 1 computes the gate and up projections together with the SwiGLU activation while keeping intermediates in registers, materializing only the activation output. This roughly halves memory trafic compared to the vanilla implementation. It also folds token gathering into the GEMM via pre-sorted indices, and Kernel 2 similarly fuses the down projection with routing-weight scaling and the output scatter (via atomic adds), writing directly to Y without intermediate bufers.

We further utilize Grouped GEMM for training. Unlike the inference kernel, the training kernel does not fuse activation functions into the GEMM. Instead, the SwiGLU activation is applied as a separate $\mathrm { P y }$ Torch operator between the up projection and down projection GEMMs. This decoupled design allows us to flexibly experiment with diferent activation types $( e . g .$ , SwiGLU, GeLU, SiLU) without modifying or re-autotuning the Triton kernel.

Algorithm 1 Fused MoE FFN – Optimized Triton Kernel   
Require: Input tokens $\mathbf { X } \in \mathbb { R } ^ { B \times D } ;$ ; expert weights $\{ \mathbf { W } _ { 1 } ^ { ( e ) } , \mathbf { W } _ { 3 } ^ { ( e ) } , \mathbf { W } _ { 2 } ^ { ( e ) } \} _ { e = 1 } ^ { E } ;$ routing indices $\mathbf { T } _ { \mathrm { i d } }$ ∈   
$\mathbf { \bar { z } } ^ { B \times K } ;$ routi $\mathbf { \Psi } _ { \mathbf { Y } } ^ { \mathrm { l g } } \mathbf { w e i g h t s } ^ { - } \mathbf { \bar { T } } _ { w } \in \mathbb { R } ^ { \tilde { B } \times \tilde { K } }$   
Ensure: Output   
— Phase 1: Route tokens to experts (In-parallel routing) —   
1: (expert[n], pos[n]) ← Sort(T<sub>id</sub>.flatten()) \triangleright Sort by exp ert ID   
2: index $[ n ] \stackrel { . } {  } \mathrm { p o s } [ n ] / / \kappa$ \triangleright Map sorted position back to original token index   
3: ofsets[e] ← CumSum(BinCount(expert, E)) ${ \triangleright \ } [ E + 1 ] ;$ jagged segment boundaries   
4: max $\begin{array} { r } { \dot { \mathbf { \eta } } _ { \mathrm { - } } \dot { \mathbf { s e g } }  \operatorname* { m a x } _ { e } ( \operatorname { o f f s e t s } [ e + 1 ] - \operatorname { o f f s e t s } [ e ] ) \dot { \mathbf { \eta } } } \end{array}$ \triangleright Longest expert segment   
— Phase 2: Fused Gate+Up projection with SwiGLU (Kernel ${ \bf 1 } ) -$   
\triangleright All E experts execute in parallel as a single batched kernel launch   
5: for each GPU thread block $( m , n , e )$ in parallel do \triangleright 3D grid:   
⌈max $\underline { { \mathbf { s e g } } } / B _ { M } ] \times \left\lceil D ^ { \prime } / B _ { N } \right\rceil \times E$   
6: s ← ofsets[e]; l ← ofsets $[ e + 1 ] \textit { - s }$   
7: if m $\cdot \ B _ { M } \ \geq l$ then return \triangleright Skip blocks beyond this expert’s segment   
8: end if   
9: Load tile of input rows from X via index $[ s . s + l ]$ \triangleright Indirect gather, fused into GEMM   
10: $\mathbf { G } _ { \mathrm { t i l e } }  \mathbf { 0 } ; \ \mathbf { U } _ { \mathrm { t i l e } } ^ { \mathrm { - } }  \mathbf { 0 }$   
11: for $k = 0 , B _ { K } , 2 B _ { K } , \dots { \bf d o }$ \triangleright Tiled matrix multiply over dimension D   
12: $\mathbf { G } _ { \mathrm { t i l e } } \mathbf { \Lambda } + = \mathrm { t i l e } ( \mathbf { X } ) \cdot \mathrm { t i l e } ( \mathbf { W } _ { 1 } ^ { ( e ) } )$   
13: $\mathbf { U } _ { \mathrm { t i l e } } \mathbf { \Lambda } + = \mathrm { t i l e } ( \mathbf { X } ) \cdot \mathrm { t i l e } ( \mathbf { W } _ { 3 } ^ { ( e ) } )$   
14: end for   
15: $\mathbf { A } _ { \mathrm { t i l e } }  \mathrm { S i L U } ( \mathbf { G } _ { \mathrm { t i l e } } ) \odot \mathbf { U } _ { \mathrm { t i l e } }$ \triangleright SwiGLU fused in registers, no memory write for G, U   
16: Write $\mathbf { A } _ { \mathrm { t i l e } }$ to activation bufer   
17: end for   
Phase 3: Fused Down projection+Scatter-Add (Kernel ${ \bf 2 } ) -$   
18: $\mathbf { Y }  \mathbf { 0 } _ { B \times D }$   
19: for each GPU thread block $( m , n , e )$ in parallel do $\triangleright 3  { \mathrm { D } }$ grid over segments and output tiles   
20: s ← ofsets[e]; l ← ofsets[e+1] − s   
21: if block’s m-ofset $\geq l$ then return   
22: end if   
23: $\mathbf { O _ { \mathrm { t i l e } } }  \mathbf { O _ { \mathrm { t } } }$   
24: for $k = 0 , B _ { K } , 2 B _ { K } , . . .$ do   
25: $\mathbf { O } _ { \mathrm { t i l e } } \ + = \mathrm { t i l e } ( \mathbf { A } ) \cdot \mathrm { t i l e } ( \mathbf { W } _ { 2 } ^ { ( e ) } )$   
26: end for   
27: $\mathbf { O } _ { \mathrm { t i l e } }  \mathbf { O } _ { \mathrm { t i l e } } \cdot \mathbf { T } _ { u }$ [corresponding positions] \triangleright Multiply by routing weight   
28: AtomicAdd Y[index[·]], O<sub>tile</sub> \triangleright Scatter-add directly to output   
29: end for   
30: return Y

## B.3 Summary

Grouped GEMM and Kernel Fusion are our necessary implementation mechanisms to faithfully translate MoE’s algorithmic eficiency into practical, hardwareeficient execution. This is an important implementation piece for vision encoders, which process all image patches simultaneously. With high resolution and tiling, a single image can produce tens of thousands of patch tokens, and batched inference further multiplies this by the batch size. Our kernel is designed for this regime. The Grouped GEMM formulation maximizes compute utilization across the large token volume, while the jagged-segment layout with per-expert early-exit naturally accommodates the uneven routing distributions that arise from spatially correlated image patches.

By co-designing the MoE architecture with kernel-level optimizations that realize its eficiency gains, we reduce vision encoder latency while maintaining strong encoding quality, making the approach particularly efective for end-toend VLM applications.

## C Scaling Experts in MoE

We conduct additional experiments to study the efect of scaling the number of experts. We train MoE-ViE-B with patch size 32 on 10.9B samples from MetaCLIP [83]. Across all settings, we keep the number of activated experts fixed $( i . e . , k = 8 )$ and set each expert MLP to $1 / 4$ of the corresponding dense MLP width.

Fig. 6 shows zero-shot performance on four image classification benchmarks: ImageNet-1K [22], ImageNetV2 [60], ObjectNet [4], and ImageNet-A [33]. As seen, increasing the total number of experts consistently improves accuracy on all benchmarks. Given our resource constraints, we adopt 32 total experts when scaling to larger vision encoders, which provides a favorable balance between performance and training cost.

![](images/4616d0c870158f298b6ebd49b1ab96685d721d3e16d30563b7cddfb87a4d3128.jpg)  
(a) ImageNet 1K

![](images/ea19a0e5fe493c731de630b1f7a5777b8fd518c4cfc18fdb203d0e307f024b9a.jpg)  
(b) ImageNet A

![](images/8bb38a46c33839ca8eaf80ea06a6926c14ec7d73b14ecb0fe10a1150f8176cf3.jpg)  
(c) ImageNetv2

![](images/9db575e14cd03240d89fbcba077576d4b815c2e20af262df84d79aba47cbdbe9.jpg)  
(d) ObjectNet  
Fig. 6: Image classification accuracy by scaling the number of experts from 16 to 128, while keeping 8 activated experts.

## D Implementation Details

Table 14 summarizes the architectures of our MoE-ViE models. Our implementation is based on OpenCLIP [15]. Particularly, we remove the class token, as we did not observe any benefits out of it, and instead use the outputs from attention pooling. We adopt 2D RoPE [69] in the vision tower and absolute positional embedding in the text tower. We further replace standard FFN layers with SwiGLU [62] style, bringing the vision encoder design closer to that commonly used in LLM decoders. Below we describe the training and evaluation setup in detail.

Table 14: Configurations of MoE-ViE models.
<table><tr><td>Scale</td><td colspan="2">B</td><td colspan="2">L</td><td colspan="2">H</td></tr><tr><td>Encoder</td><td>vision tower</td><td>text tower</td><td>vision tower</td><td>text tower</td><td>vision tower</td><td>text tower</td></tr><tr><td>Params</td><td>0.46B (0.10B activated)</td><td>0.30B</td><td>1.67B (0.31B activated)</td><td>0.30B</td><td>3.50B (1.06B activated)</td><td>0.81B</td></tr><tr><td>Width</td><td>768</td><td>1024</td><td>1024</td><td>1024</td><td>1280</td><td>1408</td></tr><tr><td>Depth</td><td>12</td><td>24</td><td>24</td><td>24</td><td>32</td><td>34</td></tr><tr><td>MLP</td><td>512 × 32 (4 activated)</td><td>2730</td><td>684 × 32 (4 activated)</td><td>2736</td><td>854 × 32 (8 activated)</td><td>3754</td></tr><tr><td>Heads</td><td>12</td><td>16</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td>CLIP Dim</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td>Pooling</td><td>Attn Pool</td><td>EOS Token</td><td>Attn Pool</td><td>EOS Token</td><td>Attn Pool</td><td>EOS Token</td></tr><tr><td>Context length</td><td></td><td>144</td><td></td><td>144</td><td></td><td>144</td></tr><tr><td>Patch size</td><td>16</td><td></td><td>16</td><td></td><td>14</td><td></td></tr><tr><td>Pos. embedding</td><td>RoPE</td><td>Absolute</td><td>RoPE</td><td>Absolute</td><td>RoPE</td><td>Absolute</td></tr></table>

## D.1 Training

We perform contrastive pretraining of MoE-ViE models with progressively increasing image resolutions. Their schedules are as follows:

– MoE-ViE-B training includes image resolutions $1 1 2 \times 1 1 2 , 1 6 0 \times 1 6 0 , 2 2 4 \times$ 224, 48.1B samples seen in total.

– MoE-ViE-L training includes image resolutions $1 1 2 \times 1 1 2 , 1 6 0 \times 1 6 0 , 2 2 4 \times 2 2 4$ 336 × 336, 384 × 384, 48.2B samples seen in total.

– MoE-ViE-H training includes image resolutions $9 8 \times 9 8 , 1 5 4 \times 1 5 4 , 2 2 4 \times 2 2 4$ 336 × 336, 448 × 448, 60.9B samples seen in total.

Our training data contains 2B image-text pairs from MetaCLIP [83] and an additional 1.5B proprietary data. We set learning rate to $1 0 ^ { - 3 }$ with cosine decay. We use LAMB [87] optimizer with weight decay 0.05 to support the large batch size 262144 being used. $( \beta _ { 1 } , \beta _ { 2 } )$ is set to (0.9, 0.995). We reduce the batch size by 2× at the highest resolution to save the compute cost. We only use InfoNCE loss [57] during pretraining.

For video finetuning, we set the learning rate to $1 0 ^ { - 6 }$ , and train for 20.5M samples for all MoE-ViE sizes. The training data consists of data curated by [5,90]. We use the same optimizer as in pretraining, with batch size of 4096. We uniformly sample 8 frames for each video input. The weight β for distillation loss is set to 0.5. For alignment, Table 16 describes our training configurations in detail.

## D.2 Evaluation

To evaluate zero-shot classification and retrieval capability, we leverage CLIP benchmark framework [14]. We use the same prompt templates as provided in [5]. Following [70], we report the maximum score obtained under two data preprocessing methods, i.e., with and without center crop. We also perform reweighting [13] on the retrieval results.

We use the popular lmms-eval framework [93] to evaluate our alignment results. Greedy decoding is used in all cases.

## E Additional Results

## E.1 Linear Probing

Following recent works [5, 7, 13], we extend our classification evaluation to include linear probing on the ImageNet-1K [22]. We show the results in Table 15, alongside other vision encoders which we benchmark. We see MoE-ViE-H/14 outperforms all encoders tested, even those with significantly more active parameters.

Table 15: Linear probing results on ImageNet-1K.
<table><tr><td>Model</td><td>Active Params</td><td>ImageNet-1K</td></tr><tr><td>DINOv2-g [58]</td><td>1.1B</td><td>86.39</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>88.32</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>88.80</td></tr><tr><td>InternViT-6B [13]</td><td>5.5B</td><td>88.44</td></tr><tr><td>MoE-ViE-H/14</td><td>1.1B</td><td>88.85</td></tr></table>

## E.2 Alignment

To further investigate the potential of MoE-ViE in the alignment setting, we expand the training pipeline presented in Section 4.2. We use Llama 3.1 8B Instruct [29] as the LLM decoder. As detailed in Table 16, we extend both the pre-alignment and supervised fine-tuning stages, scaling the total training data to 42M samples. Table 17 reports the results. MoE-ViE-H exhibits strong data scaling behavior: with 42M samples, it surpasses $\mathrm { P E _ { l a n g } , }$ , which is mid-trained on 70M images and uses 1.7× more active parameters, across all three evaluation axes, achieving 81.3 average on image benchmarks, 63.1 on video, and 132.6 on captioning. We posit that, as observed in pretraining, data quality and scale are the decisive factors for absolute alignment performance, and that MoE-ViE’s sparse architecture is particularly well-suited to capitalize on these gains.

Table 16: Training configuration overview across stages. Stage 2a is our 1k sample VE freeze step described in Section 4.2.
<table><tr><td></td><td>S1</td><td>S2a</td><td>S2b</td><td>S3</td></tr><tr><td>Trainable Params</td><td>Projector</td><td>LLM + Proj.</td><td>All</td><td>All</td></tr><tr><td>Steps</td><td>8,000</td><td>1,000</td><td>34,000</td><td>15,000</td></tr><tr><td>Global Batch Size</td><td>2,048</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Total Samples</td><td>16.4M</td><td>0.5M</td><td>17.4M</td><td>7.7M</td></tr><tr><td>Learning Rate</td><td>10−4</td><td>4×10-5</td><td>4×10-5</td><td>10-5</td></tr><tr><td>Max Seq. Length</td><td>1,280</td><td>6,144</td><td>6,144</td><td>11,520</td></tr><tr><td>Max Video Frames</td><td>8</td><td>16</td><td>16</td><td>32</td></tr><tr><td>Max Num. Tiles</td><td>1</td><td>4</td><td>4</td><td>4</td></tr></table>

Table 17: Data scaling experiments with MoE-ViE. Columns are grouped by task type. We use a maximum of 4 tiles.
<table><tr><td></td><td></td><td></td><td colspan="6">Image</td><td colspan="4">Video</td><td colspan="3">Captioning</td></tr><tr><td></td><td>Trainng scale</td><td>Active Pparams</td><td>Ava. Iges</td><td>[1] 9]</td><td>[] TVVGL]</td><td>Cha 551]</td><td>D° LV5]</td><td>[n  52]</td><td>Avg. Vdeo</td><td>Vide 2]</td><td>MV] 44]</td><td></td><td>[50] BooScema</td><td>Av Ca.</td><td>[0] 46</td><td>[] da2]</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td></td><td>1.1B</td><td>59.0</td><td>72.4</td><td>70.3</td><td>63.1</td><td>55.3</td><td>34.0</td><td>49.5</td><td>46.2</td><td>48.5</td><td>53.8</td><td>129.1</td><td>137.8</td><td>120.3</td></tr><tr><td>PEcoreG/14 [5]</td><td></td><td>1.9B</td><td>69.2</td><td>69.7</td><td>74.3</td><td>73.4</td><td>81.2</td><td>47.6</td><td>48.6</td><td>46.0</td><td>48.7</td><td>51.2</td><td>123.7</td><td>134.5</td><td>112.9</td></tr><tr><td>InternViT2.5/14 [12]</td><td></td><td>5.5B</td><td>68.1</td><td>72.9</td><td>71.3</td><td>74.6</td><td>74.3</td><td>47.6</td><td>48.7</td><td>46.0</td><td>49.6</td><td>50.6</td><td>123.0</td><td>132.5</td><td>113.5</td></tr><tr><td>AIMv2 3B/14 [26]</td><td></td><td>2.7B</td><td>69.8</td><td>72.2</td><td>79.2</td><td>73.0</td><td>78.2</td><td>46.5</td><td>49.7</td><td>49.6</td><td>49.9</td><td>49.6</td><td>130.6</td><td>139.7</td><td>121.5</td></tr><tr><td>MoE-ViE-H/14</td><td>14M</td><td>1.1B</td><td>75.2</td><td>84.9</td><td>75.7</td><td>74.2</td><td>86.1</td><td>55.0</td><td>58.9</td><td>52.0</td><td>63.6</td><td>61.2</td><td>127.5</td><td>135.4</td><td>119.6</td></tr><tr><td>PElang G/14 [5]</td><td>&gt;70M</td><td>1.9B</td><td>79.3</td><td>75.0</td><td>82.3</td><td>81.8</td><td>89.8</td><td>67.8</td><td>54.1</td><td>49.6</td><td>52.6</td><td>60.0</td><td>131.9</td><td>140.3</td><td>123.4</td></tr><tr><td>MoE-ViE-H/14</td><td>42M</td><td>1.1B</td><td>81.3</td><td>90.5</td><td>79.9</td><td>80.8</td><td>90.4</td><td>64.8</td><td>63.1</td><td>53.0</td><td>71.5</td><td>64.8</td><td>132.6</td><td>141.5</td><td>123.6</td></tr></table>

## E.3 End-to-End VLM Latency

In addition to the latency for vision encoder itself reported in Table 7, we measure end-to-end inference latency after aligning the vision encoder to an LLM decoder (Llama 3.1 8B Instruct [29]). Table 18 compares diferent vision encoders in the resulting VLM. All of our benchmarking results on latency are conducted on a single H100 GPU. The overall trend matches Table 7 that MoE-ViE delivers comparable system-level latency while achieving substantially higher accuracy than the dense model with comparable size. Moreover, MoE-ViE attains significantly lower latency than the PE [5], with comparable accuracy achieved. These results indicate that the eficiency benefits of MoE-ViE extend beyond the vision encoder and translate directly into improved latency vs. accuracy trade-ofs in downstream VLM applications.

Table 18: End-to-end latency with diferent vision encoders.
<table><tr><td>Model</td><td>Total params</td><td>Active params</td><td>Batch size</td><td> $\# \mathrm { o f ~ t i l e s }$ </td><td>Latency (ms)</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>2</td><td>4</td><td>432.8</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>1.9B</td><td>2</td><td>4</td><td>493.2</td></tr><tr><td> $\mathbf { M o E - V i E - H } / \dot { 1 } \dot { 4 }$ </td><td>3.5B</td><td>1.1B</td><td>2</td><td>4</td><td>433.3</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>2</td><td>16</td><td>523.4</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>1.9B</td><td>2</td><td>16</td><td>587.4</td></tr><tr><td>MoE-ViE-H/14</td><td>3.5B</td><td>1.1B</td><td>2</td><td>16</td><td>520.7</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>2</td><td>36</td><td>654.7</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>1.9B</td><td>2</td><td>36</td><td>744.0</td></tr><tr><td>MoE-ViE-H/14</td><td>3.5B</td><td>1.1B</td><td>2</td><td>36</td><td>660.9</td></tr><tr><td>SigLIP2-g-opt/16 [75]</td><td>1.1B</td><td>1.1B</td><td>4</td><td>36</td><td>1297.9</td></tr><tr><td> $\mathrm { P E _ { c o r e } G / 1 4 \ [ 5 ] }$ </td><td>1.9B</td><td>1.9B</td><td>4</td><td>36</td><td>1416.8</td></tr><tr><td> $\mathbf { M o E - V i E - H } / \dot { 1 } \dot { 4 }$ </td><td>3.5B</td><td>1.1B</td><td>4</td><td>36</td><td>1300.9</td></tr></table>

## References

1. Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R.K., Bai, Y., Baker, B., Bao, H., et al.: gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 (2025)

2. Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., Anderson, P.: nocaps: novel object captioning at scale. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2019)

3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

4. Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., Katz, B.: Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems 32 (2019)

5. Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H.A., Wang, J., Monteiro, M., Xu, H., Dong, S., Ravi, N., Li, S.W., Dollar, P., Feichtenhofer, C.: Perception encoder: The best visual embeddings are not at the output of the network. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https:// openreview.net/forum?id=INqBOmwIpG

6. Bossard, L., Guillaumin, M., Van Gool, L.: Food-101–mining discriminative components with random forests. In: European conference on computer vision. pp. 446–461. Springer (2014)

7. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

8. Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3558–3568 (2021)

9. Chen, K., Xu, L., Chi, H.: Improved learning algorithms for mixture of experts in multiclass classification. Neural networks 12(9), 1229–1252 (1999)

10. Chen, T., Chen, X., Du, X., Rashwan, A., Yang, F., Chen, H., Wang, Z., Li, Y.: Adamv-moe: Adaptive multi-task vision mixture-of-experts. In: proceedings of the IEEE/CVF international conference on computer vision. pp. 17346–17357 (2023)

11. Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A.J., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al.: Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794 (2022)

12. Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024)

13. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 24185–24198 (2024)

14. Cherti, M., Beaumont, R.: Clip benchmark (Nov 2022). https://doi.org/10. 5281/zenodo.15403103, https://doi.org/10.5281/zenodo.15403103

15. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2818–2829 (2023)

16. Cho, J.H., Madotto, A., Mavroudi, E., Afouras, T., Nagarajan, T., Maaz, M., Song, Y., Ma, T., Hu, S., Jain, S., et al.: Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180 (2025)

17. Clark, C., Zhang, J., Ma, Z., Park, J.S., Tripathi, R., Lee, S., Salehi, M., Ren, J., Kim, C.D., Yang, Y., et al.: Molmo2: Open weights and data for vision-language models with video understanding and grounding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 28652–28668 (2026)

18. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

19. Csordás, R., Irie, K., Schmidhuber, J.: Approximating two-layer feedforward networks for eficient transformers. In: Findings of the Association for Computational Linguistics: EMNLP 2023. pp. 674–692 (2023)

20. Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., et al.: Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1280– 1297 (2024)

21. Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A.P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al.: Scaling vision transformers to 22 billion parameters. In: International conference on machine learning. pp. 7480–7512. PMLR (2023)

22. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)

23. Dong, X., Bao, J., Zheng, Y., Zhang, T., Chen, D., Yang, H., Zeng, M., Zhang, W., Yuan, L., Chen, D., et al.: Maskclip: Masked self-distillation advances contrastive language-image pretraining. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10995–11005 (2023)

24. Dong, Y., Noci, L., Khodak, M., Li, M.: Attention retrieves, mlp memorizes: Disentangling trainable components in the transformer. arXiv e-prints pp. arXiv–2506 (2025)

25. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research 23(120), 1–39 (2022)

26. Fini, E., Shukor, M., Li, X., Dufter, P., Klein, M., Haldimann, D., Aitharaju, S., da Costa, V.G.T., Béthune, L., Gan, Z., Toshev, A.T., Eichner, M., Nabi, M., Yang, Y., Susskind, J.M., El-Nouby, A.: Multimodal autoregressive pre-training of large vision encoders (2024), https://arxiv.org/abs/2411.14402

27. Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., Chen, P., Li, Y., Lin, S., Zhao, S., Li, K., Xu, T., Zheng, X., Chen, E., Shan, C., He, R., Sun, X.: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis (2025), https://arxiv.org/abs/ 2405.21075

28. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15180– 15190 (2023)

29. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., et al.: The llama 3 herd of models (2024), https://arxiv.org/abs/2407.21783

30. Gu, J., Stevens, S., Campolongo, E.G., Thompson, M.J., Zhang, N., Wu, J., Kopanev, A., Mai, Z., White, A.E., Balhof, J., et al.: Bioclip 2: Emergent properties from scaling hierarchical contrastive learning. arXiv preprint arXiv:2505.23883 (2025)

31. Guo, D., Wu, F., Zhu, F., Leng, F., Shi, G., Chen, H., Fan, H., Wang, J., Jiang, J., Wang, J., et al.: Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062 (2025)

32. Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8340–8349 (2021)

33. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15262–15271 (2021)

34. Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E.: Adaptive mixtures of local experts. Neural computation 3(1), 79–87 (1991)

35. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)

36. Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., et al.: Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024)

37. Jordan, M.I., Jacobs, R.A.: Hierarchical mixtures of experts and the em algorithm. Neural computation 6(2), 181–214 (1994)

38. Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., et al.: The kinetics human action video dataset. arXiv preprint arXiv:1705.06950 (2017)

39. Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., Farhadi, A.: A diagram is worth a dozen images. In: European conference on computer vision. pp. 235–251. Springer (2016)

40. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: Proceedings of the IEEE international conference on computer vision workshops. pp. 554–561 (2013)

41. Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., Serre, T.: Hmdb: a large video database for human motion recognition. In: 2011 International conference on computer vision. pp. 2556–2563. IEEE (2011)

42. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.: Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020)

43. Li, J., Wang, X., Zhu, S., Kuo, C.W., Xu, L., Chen, F., Jain, J., Shi, H., Wen, L.: Cumo: Scaling multimodal llm with co-upcycled mixture-of-experts. Advances in Neural Information Processing Systems 37, 131224–131246 (2024)

44. Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Lou, P., Wang, L., Qiao, Y.: Mvbench: A comprehensive multi-modal video understanding benchmark. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22195–22206 (2024)

45. Lin, B., Tang, Z., Ye, Y., Huang, J., Zhang, J., Pang, Y., Jin, P., Ning, M., Luo, J., Yuan, L.: Moe-llava: Mixture of experts for large vision-language models. IEEE Transactions on Multimedia (2026)

46. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

47. Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al.: Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024)

48. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26296–26306 (2024)

49. Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A.: Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151 (2013)

50. Mangalam, K., Akshulakov, R., Malik, J.: Egoschema: A diagnostic benchmark for very long-form video language understanding. In: Advances in Neural Information Processing Systems (2023)

51. Masry, A., Long, D.X., Tan, J.Q., Joty, S., Hoque, E.: ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In: Findings of the Association for Computational Linguistics: ACL 2022. pp. 2263–2279. Association for Computational Linguistics (2022)

52. Mathew, M., Bagal, V., Tito, R.P., Karatzas, D., Valveny, E., Jawahar, C.V.: Infographicvqa (2021), https://arxiv.org/abs/2104.12756

53. Mathew, M., Karatzas, D., Jawahar, C.V.: Docvqa: A dataset for vqa on document images. In: 2021 IEEE Winter Conference on Applications of Computer Vision (WACV). pp. 2199–2208 (2021)

54. Mustafa, B., Riquelme, C., Puigcerver, J., Jenatton, R., Houlsby, N.: Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems 35, 9564–9576 (2022)

55. Nguyen, H., Ho, N., Rinaldo, A.: Sigmoid gating is more sample eficient than softmax gating in mixture of experts. Advances in Neural Information Processing Systems 37, 118357–118388 (2024)

56. Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: 2008 Sixth Indian conference on computer vision, graphics & image processing. pp. 722–729. IEEE (2008)

57. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)

58. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R.: Dinov2: Learning robust visual features without supervision. CoRR abs/2304.07193 (2023), https://doi.org/10.48550/arXiv.2304.07193

59. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

60. Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do imagenet classifiers generalize to imagenet? In: International conference on machine learning. pp. 5389–5400. PMLR (2019)

61. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems 34, 8583–8595 (2021)

62. Shazeer, N.: Glu variants improve transformer. arXiv preprint arXiv:2002.05202 (2020)

63. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

64. Shen, L., Chen, G., Shao, R., Guan, W., Nie, L.: MoME: Mixture of multimodal experts for generalist multimodal large language models. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), https:// openreview.net/forum?id=Xskl7Da34U

65. Sidorov, O., Hu, R., Rohrbach, M., Singh, A.: Textcaps: a dataset for image captioning with reading comprehension. In: European conference on computer vision. pp. 742–758. Springer (2020)

66. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., et al.: Dinov3 (2025), https://arxiv.org/abs/2508. 10104

67. Singh, A., Pang, G., et al.: Textvqa: Towards reading and reasoning in visual question answering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019)

68. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)

69. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

70. Sun, Q., Wang, J., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, X.: Eva-clip-18b: Scaling clip to 18 billion parameters. arXiv preprint arXiv:2402.04252 (2024)

71. Team, K., Bai, T., Bai, Y., Bao, Y., Cai, S., Cao, Y., Charles, Y., Che, H., Chen, C., Chen, G., et al.: Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276 (2026)

72. Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al.: Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534 (2025)

73. Thomee, B., Shamma, D.A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., Li, L.J.: Yfcc100m: The new data in multimedia research. Communications of the ACM 59(2), 64–73 (2016)

74. Tillet, P., Kung, H.T., Cox, D.: Triton: an intermediate language and compiler for tiled neural network computations. In: Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. pp. 10–19 (2019)

75. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025)

76. Wan, B., Tschannen, M., Xian, Y., Pavetic, F., Alabdulmohsin, I.M., Wang, X., Susano Pinto, A., Steiner, A., Beyer, L., Zhai, X.: Locca: Visual pretraining with location-aware captioners. Advances in Neural Information Processing Systems 37, 116355–116387 (2024)

77. Wang, L., Gao, H., Zhao, C., Sun, X., Dai, D.: Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664 (2024)

78. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025)

79. Wang, X., Chen, C., Yang, Y., Chen, H.Y., Zhang, B., Pal, A., Zhu, X., Du, X.: Clip-up: A simple and eficient mixture-of-experts clip training recipe with sparse upcycling. arXiv preprint arXiv:2502.00965 (2025)

80. Wu, J., Hu, X., Wang, Y., Pang, B., Soricut, R.: Omni-smola: Boosting generalist multimodal models with soft mixture of low-rank experts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14205– 14215 (2024)

81. Wu, L., Liu, M., Chen, Y., Chen, D., Dai, X., Yuan, L.: Residual mixture of experts. arXiv preprint arXiv:2204.09636 (2022)

82. Wu, Y., Du, J., Yan, K., Ding, S., Li, X.: ToVE: Eficient vision-language learning via knowledge transfer from vision experts. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id= EMMnAd3apQ

83. Xu, H., Xie, S., Tan, X.E., Huang, P.Y., Howes, R., Sharma, V., Li, S.W., Ghosh, G., Zettlemoyer, L., Feichtenhofer, C.: Demystifying clip data. arXiv preprint arXiv:2309.16671 (2023)

84. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5288–5296 (2016)

85. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

86. Ye, H., Yang, C.H.H., Goel, A., Huang, W., Zhu, L., Su, Y., Lin, S., Cheng, A.C., Wan, Z., Tian, J., et al.: Omnivinci: Enhancing architecture and data for omnimodal understanding llm. arXiv preprint arXiv:2510.15870 (2025)

87. You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., Hsieh, C.J.: Large batch optimization for deep learning: Training bert in 76 minutes. In: International Conference on Learning Representations (2020), https://openreview.net/forum?id=Syx4wnEtvH

88. Young, P., Lai, A., Hodosh, M., Hockenmaier, J.: From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the association for computational linguistics 2, 67–78 (2014)

89. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917 (2022)

90. Zellers, R., Lu, X., Hessel, J., Yu, Y., Park, J.S., Cao, J., Farhadi, A., Choi, Y.: Merlot: Multimodal neural script knowledge models. Advances in neural information processing systems 34, 23634–23651 (2021)

91. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)

92. Zhang, J., Qu, X., Zhu, T., Cheng, Y.: Clip-moe: Towards building mixture of experts for clip with diversified multiplet upcycling. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 5406–5419 (2025)

93. Zhang, K., Li, B., Zhang, P., Pu, F., Cahyono, J.A., Hu, K., Liu, S., Zhang, Y., Yang, J., Li, C., Liu, Z.: Lmms-eval: Reality check on the evaluation of large multimodal models (2024), https://arxiv.org/abs/2407.12772