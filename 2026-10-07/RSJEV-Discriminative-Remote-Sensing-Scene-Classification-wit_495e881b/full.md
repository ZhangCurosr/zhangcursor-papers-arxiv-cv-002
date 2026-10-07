# RSJEV: Discriminative Remote Sensing Scene Classification with

Multimodal Large Language Models

Dongchen Si, Di Wang, Mingzhen Xu, Jing Zhang, Senior Member, IEEE, Bo Du, Senior Member, IEEE, and Liangpei Zhang, Fellow, IEEE

Abstract—Remote sensing scene classification is a fundamental task in Earth observation and geospatial analysis. Existing approaches mainly follow three paradigms: task-specific visual classification, vision-language similarity matching, and autoregressive multimodal generation. However, visual classifiers rely on predefined label spaces, CLIP-based methods perform recognition through static image-text alignment, and multimodal large language models (MLLMs) introduce unnecessary tokenlevel generation for classification tasks with explicit candidate categories. To address these limitations, we propose RSJEV, a one-pass multimodal decision framework for remote sensing scene classification. Unlike conventional MLLMs that formulate classification as autoregressive text generation, RSJEV reformulates scene classification as a candidate-conditioned multimodal discriminative decision process, where visual representations, task instructions, and candidate category semantics are jointly modeled. Specifically, we introduce a OnePass Decider that extracts multimodal decision states and directly estimates category probabilities within the candidate category space, eliminating autoregressive decoding while preserving vision-language interactions. Extensive experiments on three widely used remote sensing scene classification benchmarks, including UC Merced, AID, and NWPU-RESISC45, demonstrate that RSJEV achieves superior classification performance compared with representative CNN-, Transformer-, Mamba-, CLIP-, and MLLM-based methods. Moreover, RSJEV significantly reduces inference costs and achieves a better accuracy-efficiency trade-off with only a compact 0.8B-parameter model. These results demonstrate the effectiveness of state-conditioned multimodal decision making for efficient remote sensing image understanding. The code will be available at https://github.com/Dongtcs/RSJEV.

Index Terms—Multimodal large language models (MLLMs), remote sensing scene classification, discriminative decision making, one-pass inference, candidate-conditioned classification.

## I. INTRODUCTION

R <sup>Emote</sup> <sup>Sensing</sup> <sup>(RS)</sup> <sup>is</sup> <sup>an</sup> <sup>essential</sup> <sup>technology</sup> <sup>for</sup> <sup>Earth</sup>observation [1], environmental monitoring [2], disaster observation [1], environmental monitoring [2], disaster assessment [3], and geospatial intelligence [4]. As a fundamental task in RS image interpretation, scene classification provides scene-level semantic information to support various geospatial applications. However, substantial intra-class variability, inter-class visual similarity, and differences in spatial resolution pose challenges to accurate RS scene classification.

In recent years, convolutional neural networks (CNNs) [5] and vision transformers (ViTs) [6] have substantially improved

RS scene classification [7]–[10]. CNNs capture local spatial patterns through hierarchical feature extraction, whereas ViTs model long-range dependencies through self-attention. These models typically use task-specific classification heads that map visual representations to a predefined label space. Although effective for supervised classification, this formulation does not explicitly incorporate candidate category semantics expressed in natural language into the prediction process. Consequently, it lacks rich semantic priors to disambiguate visually confusing categories—such as differentiating visually subtle land covers (e.g., desert versus barren land) or geometrically akin infrastructures (e.g., railway versus freeway)—and strictly restricts the model to a fixed, non-transferable label space.

Vision-language models (VLMs) [11]–[13] introduce textual category semantics into RS scene classification. CLIP [11] aligns images and natural language in a shared embedding space, enabling recognition through textual descriptions of candidate categories. CLIP-based approaches, including prompt-learning variants [12], [13], typically compute category scores using similarities between image and text embeddings. This formulation incorporates category semantics through embedding similarity, but provides limited direct interaction among visual features, task instructions, and candidate categories during prediction. Moreover, classification performance can depend on how category prompts are formulated.

Multimodal large language models (MLLMs) [14]–[17] integrate visual representations with language modeling to support instruction-guided image interpretation. When applied to RS scene classification, generative MLLMs typically predict categories by producing textual responses autoregressively. Although this formulation supports flexible outputs, generating category names or explanatory answers can require multiple decoding steps, increasing inference latency and computational cost. For classification tasks with an explicit candidate category set, the required output is a discrete category decision, motivating prediction mechanisms that avoid iterative text generation.

To retain multimodal interactions while avoiding iterative text generation, we propose RSJEV, a one-pass multimodal decision framework for RS scene classification (Fig. 1). RSJEV reformulates classification as a candidate-conditioned discriminative decision problem, jointly modeling visual representations, task instructions, and candidate category semantics. Specifically, we introduce OnePass Decider, which extracts the final-layer hidden state at a designated answer slot as a multimodal decision representation. It then reuses the pretrained language modeling head to score candidate options and computes category probabilities through a softmax over the candidate space. This design enables direct category prediction in a single forward pass without autoregressive answer generation.

![](images/8f0720fc3074ffd0531bfebef65209392ade7e5abe5db0f054edfc321e65efc4.jpg)  
Fig. 1. Conceptual comparison of different classification paradigms: (left) conventional image-only visual classifiers, (middle) similarity-based or generative MLLMs, and (right) the proposed RSJEV. By eliminating iterative autoregressive decoding, RSJEV enables instruction-guided one-pass discriminative decisionmaking, achieving an effective balance between classification performance and inference efficiency.

The primary contributions of this paper are summarized as follows:

• We propose RSJEV, a one-pass multimodal decision framework for RS scene classification. It reformulates classification as a candidate-conditioned discriminative decision problem, enabling direct category prediction without autoregressive answer generation.

• We introduce OnePass Decider, which extracts an answerslot decision representation conditioned on visual content, task instructions, and candidate category semantics. It reuses the pretrained language modeling head to score candidate options and estimate probabilities within the specified candidate space.

• Extensive experiments on three benchmark datasets alongside comprehensive ablations demonstrate that RSJEV achieves competitive accuracy with significantly lower inference latency compared to generative MLLMs.

## II. RELATED WORK

Remote sensing image classification has been extensively studied with the development of deep learning. CNN-based models, such as ResNet [7], DenseNet [8], and EfficientNet [18], have achieved remarkable performance by learning hierarchical visual representations. Recently, Transformer-based architectures, including Vision Transformer [6] and Swin Transformer [19], have further improved scene classification by modeling global contextual relationships. Meanwhile, emerging state-space models, such as RSMamba [20], have explored efficient global feature modeling for remote sensing images. Despite their effectiveness, these methods generally rely on task-specific classification heads and predefined category spaces, which limits their adaptability to dynamic and open-vocabulary scenarios.

Contrastive vision-language models incorporate textual semantics into remote sensing image recognition. CLIP [11] aligns image and text representations in a shared embedding space, enabling classification using natural-language descriptions of candidate categories. Building on this framework, RemoteCLIP [21] expands remote sensing pretraining data by converting detection and segmentation annotations into imagecaption pairs. SkyScript [22] constructs remote sensing imagetext pairs using semantic information from OpenStreetMap and supports continual pretraining of SkyCLIP. These efforts improve image-text alignment for the remote sensing domain. In their standard zero-shot classification setting, category predictions are obtained by comparing image embeddings with text embeddings of candidate categories. This similaritybased formulation differs from constructing a joint decision representation conditioned on image content, task instructions, and the candidate set.

Multimodal large language models combine visual encoders with language models to support image-conditioned language tasks. General models, including BLIP-2 [23], LLaVA [24], Qwen3-VL [14], and InternVL3 [15], provide a foundation for multimodal image interpretation. In remote sensing, GeoChat [17] supports image- and region-level conversations and visual grounding. LHRS-Bot [25] leverages volunteered geographic information to construct training data and introduces a multilevel vision-language alignment strategy. EarthGPT [26] integrates interpretation tasks across optical, synthetic aperture radar, and infrared imagery through multisensor instruction tuning. These models primarily formulate task outputs as autoregressively generated responses. For scene classification with explicit candidate categories, generating category names or explanatory answers can require additional decoding steps. RSJEV instead extracts a candidate-conditioned decision representation at an answer slot and directly scores candidate options, enabling classification without iterative answer generation.

The reviewed methods primarily focus on visual representation learning, contrastive image-text alignment, or generative multimodal interpretation. RSJEV adopts a candidateconditioned discriminative formulation that extracts an answerslot representation from the multimodal backbone and reuses the pretrained language modeling head to score candidate options. By conditioning the decision representation on image content, task instructions, and candidate category semantics, RSJEV directly estimates category probabilities in a single forward pass without iterative text generation.

## III. METHODOLOGY

## A. Problem Formulation

Given a remote sensing image $I _ { i }$ and an ordered list of candidate categories $\mathcal { C } = \left( c _ { 1 } , c _ { 2 } , \ldots , c _ { K } \right)$ , we formulate scene classification as a candidate-conditioned multimodal discriminative decision problem. Each labeled sample is represented as

$$
X _ { i } = ( I _ { i } , P _ { i } , t _ { i } ) ,\tag{1}
$$

where $P _ { i }$ is the task prompt and $t _ { i } \in \{ 1 , \ldots , K \}$ denotes the index of the ground-truth category in C. The prompt comprises a classification question and its candidate options:

$$
P _ { i } = [ \mathrm { Q u e s t i o n } , \mathrm { O p t i o n s } ] .\tag{2}
$$

The candidate category descriptions are included in the prompt as semantic conditions and processed jointly with the image by the multimodal backbone. Rather than generating a category name through autoregressive decoding, RSJEV directly estimates a probability distribution over the K candidate options:

$$
p _ { i , k } = p _ { \theta } ( k \mid I _ { i } , P _ { i } , \mathcal { C } ) , \qquad k = 1 , \ldots , K ,\tag{3}
$$

where θ denotes the model parameters and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } p _ { i , k } = 1 } \end{array}$ The predicted category is obtained by selecting the option with the highest probability:

$$
\hat { t } _ { i } = \arg \operatorname* { m a x } _ { k \in \{ 1 , \dots , K \} } p _ { i , k } , \qquad \hat { y } _ { i } = c _ { \hat { t } _ { i } } .\tag{4}
$$

This formulation enables direct classification within the supplied candidate category space through a single forward pass, without autoregressive answer generation.

## B. RSJEV

In the original JEV [27], the same question needs to be expanded into multiple input paths according to the number of candidate options, and each path is separately fed into the model. For remote sensing image classification, this process requires repeated computation of image features and textual task prompts, introducing additional computational overhead. To address this issue, we propose RSJEV, which completes task decision through a single forward pass with one input path. The details are introduced as follows:

1) Overall Workflow: As illustrated in Fig. 2, RSJEV combines a pretrained multimodal backbone with OnePass Decider (OPD) for one-pass scene classification. The backbone jointly processes the image and a task prompt containing the instruction and candidate categories. OPD extracts the finallayer hidden state at the classification answer slot and reuses the pretrained language modeling head to compute candidateoption scores. Softmax converts these scores into category probabilities, and the highest-scoring option determines the prediction without autoregressive answer generation.

2) Model Components: MLLM Backbone: We adopt pretrained multimodal large language models as the multimodal backbone of RSJEV. The backbone jointly processes the remote sensing image and the task prompt containing the classification instruction and candidate category descriptions, producing contextualized multimodal hidden states. OPD then uses the final-layer hidden state at the classification answer slot for discriminative prediction.

OnePass Decider: To exploit multimodal contextual representations for direct classification while avoiding autoregressive decoding overhead, we introduce OPD for candidateconditioned discriminative decision-making. OPD uses the final-layer hidden state at the designated [RSSC] answer slot to directly estimate candidate category probabilities within a single forward pass, without iterative answer-token generation. Let the final-layer hidden states of the multimodal backbone be

$$
H _ { i } = \left[ h _ { i , 1 } , h _ { i , 2 } , \ldots , h _ { i , N _ { i } } \right] ^ { \top } \in \mathbb { R } ^ { N _ { i } \times d } ,\tag{5}
$$

where $N _ { i }$ is the input sequence length and d is the hidden dimension. OPD extracts the decision representation at the designated [RSSC] classification answer slot:

$$
h _ { i } ^ { \mathrm { a n s } } = H _ { i } [ s _ { i } , : ] ^ { \top } \in \mathbb { R } ^ { d \times 1 } ,\tag{6}
$$

where $s _ { i }$ denotes the position of [RSSC] in the input sequence. This representation is conditioned on the image, task instruction, and candidate category descriptions.

Each candidate option is associated with a token id $u _ { k }$ Then, OPD selects the corresponding weights from the pretrained language modeling head:

$$
W c = \left[ \begin{array} { c } { { W _ { \mathrm { L M } } [ u _ { 1 } , : ] } } \\ { { \vdots } } \\ { { W _ { \mathrm { L M } } [ u _ { K } , : ] } } \end{array} \right] , \qquad z _ { i } = W _ { \mathcal { C } } h _ { i } ^ { \mathrm { a n s } } .\tag{7}
$$

The candidate probabilities are obtained by normalizing only the selected scores:

$$
p _ { i , k } = \frac { \exp ( z _ { i , k } ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { i , j } ) } .\tag{8}
$$

where $W _ { \mathrm { L M } } ~ \in ~ \mathbb { R } ^ { L \times d }$ denotes the weight matrix of the pretrained language modeling head, and L is the vocabulary size.

The highest-probability option determines the predicted category, completing classification within a single forward pass without autoregressive answer generation.

![](images/36714676dd0793526e7f5e6bf9610c03680fe29c53a9d465af9c953755519068.jpg)  
Fig. 2. Overview of the proposed RSJEV framework for one-pass remote sensing scene classification.

## IV. EXPERIMENTAL RESULTS AND ANALYSIS

In this section, we evaluate RSJEV on three remote sensing scene classification benchmarks: UCM, AID, and NWPU-RESISC45. After describing the datasets and implementation details, we compare its classification performance with representative baselines. We then assess the applicability of OPD across different MLLM backbones. Finally, we compare RSJEV with generative MLLM baselines in terms of inference latency, GPU memory consumption, and throughput.

## A. Dataset Description

We evaluate RSJEV on three widely used remote sensing scene classification datasets: UC Merced Land Use (UCM) [28], Aerial Image Dataset (AID) [29], and NWPU-RESISC45 (NWPU) [30]. Following the split protocol in [9], we adopt the UCM-55, AID-28, and NWPU-28 settings, which use 50%, 20%, and 20% of the images for training, respectively, with the remaining images used for testing.

UCM [28]: The dataset contains 2,100 aerial images spanning 21 scene categories, with 100 images per category. The images have a spatial resolution of approximately 0.3 m per pixel and are typically 256 × 256 pixels in size.

AID [29]: The dataset comprises 10,000 aerial images collected from Google Earth, covering 30 scene categories with 220–420 images per category. Each image is 600 × 600 pixels in size, with spatial resolutions ranging from 0.5 to 8 m per pixel.

NWPU [30]: The dataset contains 31,500 RGB remote sensing images collected from Google Earth, spanning 45 scene categories with 700 images per category. Each image is 256 × 256 pixels in size.

## B. Implementation Details

Experimental Settings: The experiments are conducted based on the Qwen3.5-0.8B [14] multimodal model. During training, the model is optimized using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of 0.01, and a learning rate warm-up strategy with 50 warm-up steps. Following existing JEV-style implementations<sup>1</sup>, we use crossentropy loss together with Brier score regularization during training. The model is trained for 10 epochs on eight NVIDIA RTX A40 GPUs with a per-GPU batch size of 16, resulting in a total batch size of 128. The input image resolution is dynamically adjusted by the multimodal backbone, with the maximum token length is limited to 1536. Random image flipping is applied during training to improve the generalization ability of the model. The label smoothing coefficient is set to 0.05. All experiments are conducted using FP32 precision.

Evaluation Metrics: We use precision (P), recall (R), F1- score (F1), and overall accuracy (OA) as evaluation metrics.

## C. Comparison Experiments

We compare RSJEV with representative approaches from five categories: CNN-based Classification Models, Transformer-based Classification Models, Mamba-based Classification Models, Vision-Language Models, and Multimodal Large Language Models. Specifically, the CNN-based baselines include ResNet-101 [7], DenseNet-161 [8], and EfficientNet [18]. The Transformer-based baselines include DeiT III [31], Swin Transformer [19], and Vision Transformer [6]. The Mamba-based baselines include VMamba [32], Vision Mamba [33], and RSMamba [20]. The Vision-Language Models include Open-CLIP [11], MaPLe [12], and OSCLIP [13]. The Multimodal Large Language Models include Qwen3-VL [14], InternVL3 [15], LHRS-Bot-Nova [16], and GeoChat [17]. For CNN-, Transformer-, and Mamba-based models, we initialize the networks with ImageNet-1K pretrained weights provided by MMPreTrain [34]. For Vision-Language Models and Multimodal Large Language Models, we use their officially released pretrained checkpoints and further fine-tune them on the remote sensing scene classification task.

1) Quantitative Analyses: As reported in Table I, RSJEV achieves the highest precision, recall, F1-score, and overall accuracy across all three benchmarks. Specifically, its F1- scores reach 97.72%, 96.30%, and 94.58% on UCM, AID, and NWPU, respectively. Compared with the best-performing existing method on each dataset: namely OSCLIP on UCM, Qwen3-VL on AID, and VMamba on NWPU, RSJEV achieves improvements of 0.39, 0.78, and 0.41 percentage points, respectively. Furthermore, RSJEV consistently outperforms all evaluated contrastive vision-language models and generative MLLMs. Crucially, these gains are attained with a compact 0.8B-parameter backbone, outperforming scaled generative counterparts spanning 2B to 8B parameters. These empirical comparisons firmly demonstrate that RSJEV delivers superior classification performance while maintaining remarkable parameter efficiency.

TABLE I  
COMPARISON WITH STATE-OF-THE-ART METHODS ON THREE REMOTE SENSING SCENE CLASSIFICATION BENCHMARKS. THE BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Param</td><td colspan="4">UCM P</td><td colspan="4">AID</td><td colspan="4">NWPU</td></tr><tr><td></td><td>R</td><td>F1</td><td>OA</td><td>P</td><td>R</td><td>F1</td><td>OA</td><td>P</td><td>R</td><td>F1</td><td>OA</td></tr><tr><td>CNN-based Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-101 [7]</td><td>42.6 M</td><td>95.91</td><td>95.71</td><td>95.70</td><td>95.71</td><td>92.33</td><td>91.79</td><td>91.89</td><td>92.23</td><td>90.34</td><td>90.34</td><td>90.25</td><td>90.34</td></tr><tr><td>DenseNet-161 [8]</td><td>28.7 M</td><td>97.18</td><td>97.05</td><td>97.02</td><td>97.05</td><td>93.90</td><td>93.48</td><td>93.57</td><td>93.90</td><td>92.77</td><td>92.73</td><td>92.69</td><td>92.73</td></tr><tr><td>EfficientNet [18]]</td><td>87.4M</td><td>97.12</td><td>96.76</td><td>96.80</td><td>96.76</td><td>91.48</td><td>91.39</td><td>91.33</td><td>91.76</td><td>92.49</td><td>92.45</td><td>92.43</td><td>92.45</td></tr><tr><td colspan="10">Transformer-based Models</td><td colspan="3"></td><td></td></tr><tr><td>DeiT III [31]</td><td>86.9 M</td><td>94.46</td><td>94.28</td><td>94.19</td><td>94.28</td><td>88.78</td><td>88.67</td><td>88.51</td><td>89.10</td><td>87.43</td><td>87.42</td><td>87.23</td><td>87.42</td></tr><tr><td>Swin Transformer [19]</td><td>86.8 M</td><td>96.70</td><td>96.48</td><td>96.41</td><td>96.47</td><td>94.58</td><td>94.50</td><td>94.51</td><td>94.76</td><td>92.76</td><td>92.74</td><td>92.69</td><td>92.74</td></tr><tr><td>Vision Transformer [6]</td><td>88.3 M</td><td>86.48</td><td>85.81</td><td>85.87</td><td>85.81</td><td>81.50</td><td>80.53</td><td>80.69</td><td>81.19</td><td>78.14</td><td>77.93</td><td>77.79</td><td>77.93</td></tr><tr><td colspan="10">Mamba-based Models</td><td colspan="3"></td><td></td></tr><tr><td>VMamba [32]</td><td>89.0 M</td><td>97.39</td><td>97.14</td><td>97.14</td><td>97.14</td><td>95.67</td><td>95.43</td><td>95.50</td><td>95.79</td><td>94.22</td><td>94.20</td><td>94.17</td><td>94.20</td></tr><tr><td>Vision Mamba [33]</td><td>98.0 M</td><td>96.65</td><td>96.38</td><td>96.40</td><td>96.38</td><td>94.23</td><td>94.03</td><td>94.06</td><td>94.39</td><td>93.92</td><td>93.88</td><td>93.86</td><td>93.88</td></tr><tr><td>RSMamba [20]</td><td>33.1 M</td><td>94.78</td><td>94.57</td><td>94.59</td><td>94.48</td><td>82.28</td><td>81.63</td><td>81.70</td><td>82.37</td><td>80.98</td><td>80.99</td><td>80.87</td><td>80.99</td></tr><tr><td colspan="10">Vision-Language Models</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Open-CLIP [11]</td><td>151.0 M</td><td>96.57</td><td>96.38</td><td>96.40</td><td>96.38</td><td>95.35</td><td>95.21</td><td>95.23</td><td>95.46</td><td>91.14</td><td>90.83</td><td>90.70</td><td>91.22</td></tr><tr><td>MaPLe [12]</td><td>151.0 M</td><td>95.85</td><td>95.71</td><td>95.74</td><td>95.71</td><td>92.98</td><td>92.43</td><td>92.61</td><td>92.93</td><td>88.75</td><td>88.68</td><td>88.66</td><td>88.68</td></tr><tr><td>OSCLIP [13]</td><td>428.0 M</td><td>97.39</td><td>97.33</td><td>97.33</td><td>97.33</td><td>95.54</td><td>95.40</td><td>95.45</td><td>95.70</td><td>93.36</td><td>93.31</td><td>93.30</td><td>93.31</td></tr><tr><td colspan="10">Multimodal Large Language Models</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Qwen3-VL [14]</td><td>2.0 B</td><td>96.04</td><td>95.81</td><td>95.79</td><td>95.81</td><td>95.61</td><td>95.60</td><td>95.52</td><td>95.70</td><td>92.94</td><td>92.83</td><td>92.81</td><td>92.88</td></tr><tr><td>InternVL3 [15]</td><td>2.0 B</td><td>94.70</td><td>94.48</td><td>94.47</td><td>94.48</td><td>94.76</td><td>94.45</td><td>94.55</td><td>94.93</td><td>92.24</td><td>92.11</td><td>92.13</td><td>92.11</td></tr><tr><td>LHRS-Bot-Nova [16]</td><td>8.0 B</td><td>83.90</td><td>80.48</td><td>78.99</td><td>80.47</td><td>82.48</td><td>81.04</td><td>79.71</td><td>82.36</td><td>83.79</td><td>81.01</td><td>79.30</td><td>81.01</td></tr><tr><td>GeoChat [17]</td><td>7.0 B</td><td>86.17</td><td>78.10</td><td>77.21</td><td>78.10</td><td>79.14</td><td>73.98</td><td>71.47</td><td>73.80</td><td>91.26</td><td>90.23</td><td>90.23</td><td>90.23</td></tr><tr><td>RSJEV (Ours)</td><td>0.8 B</td><td>97.80</td><td>97.71</td><td>97.72</td><td>97.71</td><td>96.44</td><td>96.22</td><td>96.30</td><td>96.56</td><td>94.61</td><td>94.59</td><td>94.58</td><td>94.59</td></tr></table>

## D. Generalizability of OnePass Decider

We evaluate the effectiveness and generalizability of OPD across different MLLM backbones, with detailed results reported in Table II. Baseline denotes the generative classification configuration of each backbone, i.e., standard autoregressive next-token prediction of category answers, whereas Ours denotes the corresponding model equipped with OPD. On Qwen3.5-0.8B, OPD improves precision from 87.96%, 91.29%, and 88.24% to 97.80%, 96.44%, and 94.61% on UCM, AID, and NWPU, respectively, corresponding to gains of 9.84, 5.15, and 6.37 percentage points. To further assess whether these improvements generalize beyond the backbone used in the main experiments, we additionally adapt OPD to InternVL3.5-1B. OPD again consistently improves precision across all three benchmarks, yielding gains of 3.47, 4.01, and 1.39 percentage points on UCM, AID, and NWPU, respectively. These consistent improvements across both Qwen3.5- 0.8B and InternVL3.5-1B demonstrate that OPD is not tied to a specific MLLM backbone and exhibits strong generalizability across different architectures.

## E. Inference Efficiency Analysis

We evaluate the inference efficiency of RSJEV against six MLLM baselines: Qwen3.5-0.8B, Qwen3-VL-2B, InternVL3.5-1B, InternVL3-2B, LHRS-Bot-Nova-8B, and GeoChat-7B. All measurements are conducted on a single NVIDIA A40 GPU in bfloat16 precision with a batch size of 1, using 30 test images after a five-image warm-up. GPU memory consumption and inference latency are reported in Fig. 3, while throughput is presented in Fig. 4. RSJEV achieves the lowest GPU memory consumption and inference latency among the evaluated methods, requiring 1.68 GB of GPU memory and 102.18 ms per image. Its throughput reaches 9.79 FPS, exceeding all generative baselines, whose throughput ranges from 1.31 to 4.09 FPS. Specifically, RSJEV achieves 4.31 times the throughput of Qwen3.5 and 2.39 times that of InternVL3, the fastest generative baseline. These results highlight the efficiency of the compact RSJEV model, which directly estimates candidate category probabilities through a single forward pass without autoregressive answer generation. Together with the classification results in Table I, these findings demonstrate that RSJEV combines strong classification performance with lower memory requirements and faster inference than the evaluated MLLM baselines, offering a favorable performance-efficiency balance for remote sensing scene classification.

TABLE II  
GENERALIZABILITY ACROSS MLLM BACKBONES (P, %).
<table><tr><td>Backbone</td><td>Method</td><td>UCM</td><td>AID</td><td>NWPU</td></tr><tr><td rowspan="2">Qwen3.5-0.8B</td><td>Baseline</td><td>87.96</td><td>91.29</td><td>88.24</td></tr><tr><td>Ours</td><td>97.80</td><td>96.44</td><td>94.61</td></tr><tr><td rowspan="2">InternVL3.5-1B</td><td>Baseline</td><td>93.21</td><td>92.26</td><td>93.54</td></tr><tr><td>Ours</td><td>96.68</td><td>96.27</td><td>94.93</td></tr></table>

## V. CONCLUSION

In this paper, we propose RSJEV, a one-pass multimodal decision framework for remote sensing scene classification. RSJEV reformulates classification as a candidate-conditioned multimodal discriminative decision problem by jointly modeling visual representations, task instructions, and candidate category semantics. The proposed OnePass Decider directly estimates candidate category probabilities from the final-layer

Fig. 4. Inference throughput comparison among different MLLM-based approaches in terms of FPS.

![](images/ffa0851d134d1cf3c10e03fc7c9f55c7aea9403db8619be6734fc277b966e0bc.jpg)  
Fig. 3. Trade-off between GPU memory consumption and inference latency among different MLLM-based approaches.

hidden state at the classification answer slot, avoiding the additional decoding overhead of autoregressive answer generation. Experiments demonstrate that RSJEV achieves strong classification performance while substantially improving inference efficiency over generative MLLM baselines. Crossbackbone experiments further validate the effectiveness and generalizability of OnePass Decider across different MLLM architectures. Consequently, RSJEV highlights the considerable potential of JEV-style multimodal discriminative modeling as a new decision paradigm for remote sensing, offering a promising direction toward more efficient and scalable multimodal foundation models.

## REFERENCES

[1] B. Chintalapati, A. Precht, S. Hanra, R. Laufer, M. Liwicki, and J. Eickhoff, “Opportunities and challenges of on-board ai-based image recognition for small satellite earth observation missions,” Advances in space research, vol. 75, no. 9, pp. 6734–6751, 2025.

[2] P. Shu, R. W. Aslam, I. Naz, B. Ghaffar, D. E. Kucher, A. Quddoos, D. Raza, M. Abdullah-Al-Wadud, and R. M. Zulqarnain, “Deep learningbased super-resolution of remote sensing images for enhanced groundwater quality assessment and environmental monitoring in urban areas,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 7933–7949, 2025.

[3] S. Al Shafian and D. Hu, “Integrating machine learning and remote sensing in disaster management: A decadal review of post-disaster building damage assessment,” Buildings, vol. 14, no. 8, p. 2344, 2024.

[4] P. Liu, Y. Zhang, and F. Biljecki, “Explainable spatially explicit geospatial artificial intelligence in urban analytics,” Environment and Planning B: Urban Analytics and City Science, vol. 51, no. 5, pp. 1104–1123, 2024.

[5] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021.

[7] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[8] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR). Ieee, 2017, pp. 2261– 2269.

![](images/c9ca7b46b71afcd8ba6ca2d2ac2b6c4c76285b44f611e922eaa7d6665ea9950f.jpg)

[9] D. Wang, Q. Zhang, Y. Xu, J. Zhang, B. Du, D. Tao, and L. Zhang, “Advancing plain vision transformer toward remote sensing foundation model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[10] D. Wang, J. Zhang, B. Du, G.-S. Xia, and D. Tao, “An empirical study of remote sensing pretraining,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–20, 2023.

[11] A. Radford et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, 2021, pp. 8748–8763.

[12] M. U. khattak, H. Rasheed, M. Maaz, S. Khan, and F. S. Khan, “Maple: Multi-modal prompt learning,” in The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[13] D. Peng, X. Zhang, W. Wu, X. Ma, and W. Yu, “Osclip: Domain-adaptive prompt tuning of vision-language models for open-set remote sensing image classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 25 863–25 875, 2025.

[14] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[15] J. Zhu et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” 2025. [Online]. Available: https://arxiv.org/abs/2504.10479

[16] Z. Li, D. Muhtar, F. Gu, Y. He, X. Zhang, P. Xiao, G. He, and X. Zhu, “Lhrs-bot-nova: Improved multimodal large language model for remote sensing vision-language interpretation,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 227, pp. 539–550, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0924271625002230

[17] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “Geochat: Grounded large vision-language model for remote sensing,” The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[18] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 09–15 Jun 2019, pp. 6105–6114. [Online]. Available: https://proceedings.mlr.press/v97/tan19a.html

[19] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2021, pp. 10 012–10 022.

[20] K. Chen, B. Chen, C. Liu, W. Li, Z. Zou, and Z. Shi, “Rsmamba: Remote sensing image classification with state space model,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[21] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou, “Remoteclip: A vision language foundation model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 16, 2024.

[22] Z. Wang, R. Prabha, T. Huang, J. Wu, and R. Rajagopal, “Skyscript: A large and semantically diverse vision-language dataset for remote sensing,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 6, 2024, pp. 5805–5813.

[23] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PmLR, 2023, pp. 19 730–19 742.

[24] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023.

[25] D. Muhtar, Z. Li, F. Gu, X. Zhang, and P. Xiao, “Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model,” in European Conference on Computer Vision. Springer, 2024, pp. 440– 457.

[26] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, and X. Mao, “Earthgpt: A universal multimodal large language model for multisensor image comprehension in remote sensing domain,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–20, 2024.

[27] TypeSafe AI, “Typesafe api,” https://api.typesafe.ai/docs, 2026, version 0.2.0. Accessed: September 24, 2026.

[28] Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems. Association for Computing Machinery, 2010, pp. 270–279.

[29] G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu, “Aid: A benchmark data set for performance evaluation of aerial scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 7, pp. 3965–3981, 2017.

[30] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[31] H. Touvron, M. Cord, and H. Jegou, “DeiT III: Revenge of the ViT,”´ in Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV. Berlin, Heidelberg: Springer-Verlag, 2022, pp. 516–533. [Online]. Available: https://doi.org/10.1007/978-3-031-20053-3 30

[32] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, and Y. Liu, “Vmamba: Visual state space model,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 103 031–103 063. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf

[33] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision mamba: Efficient visual representation learning with bidirectional state space model,” in Forty-first International Conference on Machine Learning.

[34] M. Contributors, “Openmmlab’s pre-training toolbox and benchmark,” https://github.com/open-mmlab/mmpretrain, 2023.