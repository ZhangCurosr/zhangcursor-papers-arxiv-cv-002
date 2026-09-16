# TEDi: Temporal Memory-Enhanced and Denoising Transformer for Surgical Instrument Segmentation

Jiahong Yuan<sup>1\*</sup>, Weiming Mi<sup>2\*</sup>, Tao Zhang<sup>1\*\*</sup>, and Haoyin Zhou<sup>2\*\*</sup>

<sup>1</sup> Department of Automation, Tsinghua University, Beijing, China

yuanjh25@mails.tsinghua.edu.cn, taozhang@tsinghua.edu.cn

<sup>2</sup> Surgical Planning Laboratory, Brigham and Women’s Hospital, Harvard Medical School, United States

{wemi,zhouhaoyin}@bwh.harvard.edu

Abstract. Query-based segmentation methods have shown promising potential for surgical instrument segmentation and recognition, which is essential for scene understanding and downstream tasks in computerassisted surgery. However, most existing approaches predominantly rely on per-frame predictions and overlook cross-frame temporal priors as well as temporal-consistency constraints. This limitation often leads to unstable query representations and suboptimal category recognition. In this paper, we propose TEDi, a Temporal memory-Enhanced and Denoising transformer for surgical instrument segmentation that addresses these issues through Memory Search Enhancement and Temporal Consistency Denoising. The former introduces a query-level memory bank and a memory search enhancement encoder to retrieve discriminative representations from historical frames, enriching current-frame features. The latter constructs a temporally consistent reference as a cross-frame semantic anchor to suppress temporally unstable predictions and promote semantic coherence across frames. Extensive experiments on two benchmark datasets, EndoVis 2017 and EndoVis 2018, demonstrate that TEDi consistently outperforms state-of-the-art methods, highlighting its potential to further advance computer-assisted surgery. The code will be released after acceptance.

Keywords: Surgical Instrument Segmentation · Query-based Segmentation · Transformers · Deep Learning.

## 1 Introduction

Robot-assisted minimally invasive surgery (RAMIS) leverages dexterous articulated instruments and high-fidelity endoscopic imaging to enable precise manipulation while reducing surgical trauma and accelerating postoperative recovery [8, 19]. Accurate surgical instrument segmentation is fundamental to surgical scene understanding [12, 4], supporting downstream tasks such as instrument tracking [23], pose estimation [20], and trajectory prediction [25], and serving as a critical component for next-generation surgical robotic systems [13, 18, 17].

Early methods formulated instrument segmentation as pixel-wise classification, which often sufers from spatial class inconsistency [24, 27]. Recently, querybased segmentation (QBS) frameworks, such as Mask2Former [6], have gained increasing attention due to their unified set-prediction paradigm that jointly models instance masks and categories. Several studies have adapted QBS models to surgical scenarios. ISINet [10] builds upon Mask R-CNN [11] for instrument segmentation, while other approaches improve QBS-based models by enhancing the discriminability of query representations [5] and improving query initialization [9]. Beyond single-frame prediction, temporal cues have been introduced to further improve performance and cross-frame consistency, including leveraging historical queries for tracking [28] and extending Mask2Former with video Transformers [3] or spatio-temporal context modeling to strengthen temporal consistency and robustness [26].

However, empirical evidence suggests that, in surgical videos, QBS models typically achieve accurate localization while remaining prone to severe category prediction errors [3, 26]. These errors manifest as (i) misclassification among visually confusable instruments and (ii) inconsistent class predictions for the same instance across adjacent frames, resulting in temporally unstable semantics. This indicates that query representations learned primarily under single-frame supervision are insuficient to exploit temporal cues as stable priors for finegrained category discrimination.

To address these limitations, we propose a temporal-enhanced framework, TEDi, for surgical instrument segmentation, built upon two key principles: Memory Search Enhancement (MSE) and Temporal Consistency Denoising (TCD). By explicitly mining temporal cues, TEDi systematically reduces category prediction errors and improves temporal semantic stability.

Our main contributions are summarized as follows:

We propose query-level memory bank and memory search enhancement encoder that retrieve target representations from historical frames to provide cross-frame semantic priors for current-frame segmentation.

– We develop a window-based temporal-consistency denoising decoder that constructs a temporally stable reference and enforces identity-aligned temporal correction to suppress unreliable query representations.

Extensive experiments on two EndoVis benchmark datasets demonstrate that our method achieves competitive or superior performance compared with stateof-the-art approaches, with particularly notable improvements on confusable categories and temporal stability.

## 2 Method

Our framework follows a query-centric pipeline. For each frame, we first extract instance-aware query embeddings and refine them via Memory Search Enhancement (MSE), which retrieves cross-frame semantic priors from a query-level memory bank to strengthen the current-frame representations. We then apply Temporal Consistency Denoising (TCD), a local temporal-window mechanism that denoises unreliable queries, suppresses unstable predictions, and promotes semantic coherence across frames. Finally, the denoised queries are fed into the mask prediction head to produce the segmentation masks.

![](images/dc254158825b8617b2daad338715ae2766b1b0527be6264dd2482e4cb39ff04e.jpg)  
Fig. 1. The overview of TEDi. Panel (a) illustrates the overall pipeline of TEDi, while (b) and (c) depict the architectures of the MSE Encoder and the TCD Decoder.

## 2.1 Preliminaries

Baseline Segmentation Model. We adopt Mask2Former [6] as the baseline, a representative query-based segmentation framework that formulates segmentation as predicting categories and masks for a fixed set of learnable object queries. Given an input image $\overset { \smile } { I } \in \mathbb { R } ^ { H \times W }$ , the backbone and pixel decoder produce a feature map F, which interacts with N object queries $\mathbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N } , \mathbf { q } _ { i } \in \mathbb { R } ^ { d }$ through a transformer decoder. Each query outputs a mask prediction m<sub>i</sub> $\in [ 0 , 1 ] ^ { H \times \smile }$ and a classification score $\mathbf { c } _ { i } \in \mathbb { R } ^ { C }$ , forming a set of segmentation candidates.

## 2.2 Proposed Methodology

Memory Search Enhancement Encoder. Memory mechanisms are widely adopted in video segmentation to enhance current predictions using representations from previous frames [21, 15]. However, storing dense pixel-level features often introduces redundancy and weakens retrieval efectiveness [7, 29]. We instead propose a query-level memory bank with a memory search enhancement encoder, storing compact instance-aware queries. By encoding predicted masks and injecting mask-aware representations into the queries, the memory remains both eficient and spatially informative.

Firstly, we fine-tune the segmenter and then freeze its parameters to produces a set of query-level predictions pairs $\mathcal { H } _ { t } = \{ \mathbf { q } _ { t , i } , \mathbf { c } _ { t , i } , \mathbf { m } _ { t , i } \} _ { i = 1 } ^ { N }$ for frame t.

Secondly, we encode the mask $\mathbf { m } _ { t , i }$ into the query embedding space:

$$
\mathbf { z } _ { t , i } = \phi _ { m } ( \mathbf { m } _ { t , i } ) \in \mathbb { R } ^ { d } ,\tag{1}
$$

and fuse it with the original query $\mathbf { q } _ { t , i }$ using a learnable gating function to obtain the query-level memory:

$$
\mathbf { e } _ { t , i } = \mathbf { g } _ { t , i } \odot \mathbf { q } _ { t , i } + ( 1 - \mathbf { g } _ { t , i } ) \odot \mathbf { z } _ { t , i } , \quad \mathbf { g } _ { t , i } = \sigma ( \phi _ { g } ( \mathbf { q } _ { t , i } , \mathbf { z } _ { t , i } ) ) \in \mathbb { R } ^ { d } ,\tag{2}
$$

where $\phi _ { m } ( \cdot )$ and $\phi _ { g } ( \cdot )$ denote learnable projection functions, $\sigma ( \cdot )$ is the sigmoid function, and $\odot$ denotes element-wise multiplication. Hence, the memory of frame t is defined as $\mathbf { E } _ { t } = \{ \mathbf { e } _ { t , i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d }$

Thirdly, a fixed-capacity memory bank $\mathcal { M } _ { t }$ containing query embeddings from the recent K frames is maintained by updating online in a FIFO manner.

$$
\begin{array} { r } { \mathcal { M } _ { t } = \{ \mathbf { E } _ { \tau } \} _ { \tau \in \mathcal { T } _ { t } } , \quad \mathcal { T } _ { t } = \{ \tau \mid t - K \leq \tau \leq t - 1 \} . } \end{array}\tag{3}
$$

Finally, given the current-frame queries $\mathbf { Q } _ { t } ^ { C } \mathbf { \Psi } = \{ \mathbf { q } _ { t , i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d }$ and the concatenated stored memories along the query dimension $\mathbf { M } _ { t } = \mathrm { S t a c k } ( \mathcal { M } _ { t } ) \in$ $\mathbb { R } ^ { K N \times d }$ , we obtain enhanced queries by retrieving cross-frame semantic priors via stacked memory attention blocks:

$$
{ \mathbf { Q } } _ { t } ^ { E } = \operatorname { M e m A t t n } ( { \mathbf { Q } } _ { t } ^ { C } , { \mathbf { M } } _ { t } ) .\tag{4}
$$

The enhanced queries $\mathbf { Q } _ { t } ^ { E } = \{ \mathbf { q } _ { t , j } ^ { E } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { N \times d }$ are then used for subsequent temporal denoising and prediction.

Temporal Consistency Denoising Decoder. The baseline segmenter is trained in a frame-wise manner and therefore lacks explicit temporal constraints. When applied to videos, its query predictions may become temporally inconsistent or overly confident yet incorrect. Such instability can propagate across frames and impair segmentation reliability [26]. To address this limitation, we introduce a temporal consistency denoising decoder that refines query embeddings through identity-aligned temporal correction, enforcing coherence across frames and suppressing per-frame noisy query representations.

For frame t, we maintain a Temporal Consistency Reference $\mathbf { R } _ { t } = \{ \mathbf { r } _ { t , i } \} _ { i = 1 } ^ { N } \in$ $\mathbb { R } ^ { N \times d }$ , which is defined as the denoised queries from the previous frame and serves as a temporally stable semantic anchor:

$$
\begin{array} { r } { \mathbf { R } _ { t } = \left\{ \begin{array} { l l } { \mathbf { Q } _ { t } ^ { E } , } & { t = 0 , } \\ { \mathbf { Q } _ { t - 1 } ^ { D } , } & { t > 0 . } \end{array} \right. } \end{array}\tag{5}
$$

Then, we evaluate the similarity between the Reference and the enhanced queries, and obtain a one-to-one assignment $\pi _ { t }$ via the Hungarian algorithm:

$$
\pi _ { t } = \arg \operatorname* { m a x } _ { \pi } \sum _ { i = 1 } ^ { N } \mathbf { S } _ { i , \pi ( i ) } , \quad \mathbf { S } _ { i , j } = \frac { \langle \mathbf { r } _ { t , i } , \mathbf { q } _ { t , j } ^ { E } \rangle } { \| \mathbf { r } _ { t , i } \| _ { 2 } \| \mathbf { q } _ { t , j } ^ { E } \| _ { 2 } } .\tag{6}
$$

Hence, the enhanced queries can be subsequently reordered to align with $\mathbf { R } _ { t } \colon$

$$
\mathbf { Q } _ { t } ^ { \pi _ { t } } = \operatorname { R e o r d e r } ( \mathbf { Q } _ { t } ^ { E } , \pi _ { t } ) .\tag{7}
$$

Finally, we refine the queries with L cascaded denoising blocks using temporal consistency cues, where the core denoising cross-attention is defined as:

$$
\begin{array} { r } { \widetilde { \mathbf { Q } } _ { t } ^ { \ell } = \widetilde { \mathbf { Q } } _ { t } ^ { \ell - 1 } + \mathrm { C A } ( \mathbf { R } _ { t } , \mathbf { Q } _ { t } ^ { E } , \mathbf { Q } _ { t } ^ { E } ) , \qquad \widetilde { \mathbf { Q } } _ { t } ^ { 0 } = \mathbf { Q } _ { t } ^ { \pi _ { t } } , } \end{array}\tag{8}
$$

here, the residual term is applied on $\widetilde { \mathbf { Q } } _ { t } ^ { \ell - 1 }$ , ensuring that the decoder preserves the current-frame representation and only injects temporally consistent information. Within each block, we further apply self-attention and a FFN Network.

After L iterations, the denoised queries become the output of the decoder $\mathbf { Q } _ { t } ^ { D } = \widetilde { \mathbf { Q } } _ { t } ^ { L }$ , as well as the reference ${ \bf R } _ { t + 1 }$ for the next frame. To prevent long-term error accumulation, the temporal reference is reset every T frames.

Optimization. We train TEDi using a classification loss $\mathcal { L } _ { C l s }$ and a mask prediction loss $\mathcal { L } _ { M a s k }$ . Specifically, we adopt focal loss [14] for $\mathcal { L } _ { C l s } .$ , and combine cross-entropy (CE) loss and Tversky loss [22] for mask supervision. The overall loss is defined as:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { 1 } \mathcal { L } _ { C l s } + \lambda _ { 2 } \mathcal { L } _ { c e } + \lambda _ { 3 } \mathcal { L } _ { T v e r s k y } ,\tag{9}
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ , and $\lambda _ { 3 }$ denote the loss weights. In practice, we set $\lambda _ { 1 } = 2 , \lambda _ { 2 } = 5$ and $\lambda _ { 3 } = 5$

## 3 Experiments

All experiments were conducted using publicly available datasets. We evaluate TEDi on two public benchmarks for surgical instrument segmentation, EndoVis 2017(EV17) [2] and EndoVis 2018(EV18) [1]. To ensure fair comparisons with prior work, we strictly follow the commonly adopted dataset splits and label settings used in previous studies. For EV17, we follow [24] and report results under the 4-fold cross-validation protocol. For EV18, We adopt the validation splits and the instrument category annotations provided in [10]

We report performance using three commonly used metrics in this field: challenge IoU (Ch\_IoU), ISINet IoU(ISI\_IoU), and mean class IoU (mc\_IoU).

## 3.1 Implementation Details

We use Swin-Small [16] as the backbone and set the number of object queries to $N = 1 0$ to reduce redundancy. The memory bank capacity and denoising window length are set to $K = 3$ and $T = 3 .$ , respectively. Memory bank initialized from 3 preceding frames. Training is performed with Adam at a learning rate of $6 \times 1 0 ^ { - 5 }$ . Inference is restricted to the last frame of each window to maintain strict causality. All experiments are conducted on a single NVIDIA A100 GPU.

Table 1. Performance comparison on EV17/18 datasets. M2F refers to the baseline Mask2Former model, while N denotes the number of learnable object queries.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Ch IoU</td><td rowspan="2">ISI IoU</td><td rowspan="2">mc IoU</td><td colspan="7">Instrument Categories IoU</td></tr><tr><td>BF</td><td>PF</td><td>LND</td><td></td><td></td><td>VS/SI GR/CA MCS</td><td>UP</td></tr><tr><td colspan="10">Dataset EV17</td></tr><tr><td>TernausNet[24]</td><td>35.27</td><td>39.87</td><td>14.19</td><td>44.20</td><td>4.67</td><td>0.00</td><td>0.00</td><td>0.00</td><td>50.44</td><td>0.00</td></tr><tr><td>Dual-MF[27]</td><td>45.80</td><td></td><td>26.40</td><td>34.40</td><td>21.50</td><td>64.30</td><td>24.10</td><td>0.80</td><td>17.90</td><td>21.80</td></tr><tr><td>ISINet[10]</td><td>55.62</td><td>52.20</td><td>28.96</td><td>38.70</td><td>38.50</td><td>50.09</td><td>27.43</td><td>2.01</td><td>28.72</td><td>12.56</td></tr><tr><td>S3Net[5]</td><td>72.54</td><td>71.99</td><td>46.55</td><td>75.08</td><td>54.32</td><td>61.84</td><td>35.50</td><td>27.47</td><td>43.23</td><td>28.38</td></tr><tr><td>MATIS(Full)[3]</td><td>71.36</td><td>66.28</td><td>41.09</td><td>68.37</td><td>53.26</td><td>53.55</td><td>31.89</td><td>27.34</td><td>21.34</td><td>26.53</td></tr><tr><td>LACOSTE(S)[26]</td><td>76.32</td><td>72.37</td><td>48.22</td><td>73.24</td><td>52.04</td><td>60.41</td><td>38.73</td><td>0.00</td><td>54.53</td><td>67.88</td></tr><tr><td>QPD[9]</td><td>77.80</td><td>79.58</td><td>49.92</td><td>70.61</td><td>45.84</td><td>80.01</td><td>63.41</td><td>33.64</td><td>66.57</td><td>35.28</td></tr><tr><td>M2F(N = 100)</td><td>75.12</td><td>71.68</td><td>44.48</td><td>60.42</td><td>62.97</td><td>60.88</td><td>36.29</td><td>3.14</td><td>30.22</td><td>44.48</td></tr><tr><td>M2F(N = 10)</td><td>77.79</td><td>73.05</td><td>49.63</td><td>76.33</td><td>60.06</td><td>65.06</td><td>35.69</td><td>3.81</td><td>42.75</td><td>63.68</td></tr><tr><td>TEDi</td><td>80.71</td><td>78.30</td><td>52.11</td><td>75.55</td><td>58.55</td><td>65.78</td><td>39.32</td><td>15.76</td><td>40.64</td><td>69.13</td></tr><tr><td colspan="10">Dataset EV18</td></tr><tr><td>TernausNet[24]</td><td>35.27</td><td>12.67</td><td>10.17</td><td>13.45</td><td>12.39</td><td>20.51</td><td>5.97</td><td>1.08</td><td>1.00</td><td>16.76</td></tr><tr><td>Dual-MF[27]</td><td>70.40</td><td></td><td>35.09</td><td>74.10</td><td>6.80</td><td>46.00</td><td>30.10</td><td>7.60</td><td>80.90</td><td>0.10</td></tr><tr><td>ISINet[10]</td><td>73.03</td><td>70.97</td><td>38.73</td><td>73.83</td><td>40.35</td><td>30.98</td><td>37.68</td><td>0.00</td><td>88.16</td><td>0.16</td></tr><tr><td>S3Net[5]</td><td>75.81</td><td>74.02</td><td>42.58</td><td>77.22</td><td>50.87</td><td>19.83</td><td>50.59</td><td>0.00</td><td>92.12</td><td>7.44</td></tr><tr><td>MATIS(Full)[3]</td><td>84.26</td><td>79.12</td><td>54.04</td><td>83.52</td><td>41.90</td><td>66.18</td><td>70.57</td><td>0.00</td><td>92.96</td><td>23.12</td></tr><tr><td>LACOSTE(S)[26] 85.20</td><td></td><td>82.41</td><td>55.92</td><td>85.21</td><td>70.75</td><td>68.02</td><td>62.64</td><td>12.81</td><td>91.98</td><td>0.00</td></tr><tr><td>QPD[9]</td><td>77.77</td><td>78.43</td><td>43.84</td><td>82.80</td><td>60.94</td><td>19.96</td><td>49.70</td><td>0.00</td><td>93.93</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M2F(N = 100)</td><td>82.93</td><td>80.54</td><td>49.77</td><td>85.23</td><td>69.42</td><td>45.37</td><td>56.35</td><td>0.00</td><td>91.99</td><td>0.00</td></tr><tr><td>M2F(N = 10)</td><td>84.18</td><td>81.26</td><td>48.91</td><td>86.90</td><td>43.47</td><td>60.46</td><td>54.00</td><td>4.27</td><td>93.29</td><td>0.00</td></tr><tr><td>TEDi</td><td>86.69</td><td>84.82</td><td>57.41</td><td>87.45</td><td>56.22</td><td>69.91</td><td>69.96</td><td>14.82</td><td>93.94</td><td>9.59</td></tr></table>

## 3.2 Comparison with State-of-the-Art Methods

We compare TEDi with representative approaches, including single-frame models (QPD [9], S3Net [5]) and temporal models (MATIS [3], LACOSTE [26]). As shown in Table 1, TEDi consistently outperforms prior methods on EV18 across all three metrics. Compared with QPD, TEDi improves Ch\_IoU, ISI\_IoU, and mc\_IoU by 8.92, 6.39, and 13.57 points, respectively. Notably, for challenging classes such as Clip Applier and Large Needle Driver, TEDi achieves substantially higher class IoU, demonstrating the efectiveness of cross-frame memory retrieval and temporal-consistency denoising in mitigating category confusion. Compared with temporal approaches such as LACOSTE and MATIS, TEDi further achieves clear gains in Ch\_IoU and ISI\_IoU, highlighting the benefit of memory interaction and consistency-guided refinement in the query embedding space. On EV17, TEDi attains 80.71% Ch\_IoU and 78.30% ISI\_IoU, surpassing existing state-of-the-art methods on the main metrics. Improvements over LA-COSTE and MATIS remain consistent across evaluation protocols, confirming the robustness of the proposed framework.

Fig. 2 presents qualitative comparisons. In challenging scenarios with blurred boundaries, partial visibility, or temporal inconsistencies, existing methods often exhibit class flickering or semantic drift, whereas TEDi produces more stable predictions and improved category discrimination for confusable instruments.

![](images/0cfe85d46c68b4549be6c7e4f43d8cecca2c0fb01135eac8e7966728bf1cd2a3.jpg)  
Fig. 2. Qualitative comparison of TEDi with other SOTA methods. ✓ indicates that the instrument is classified and segmented correctly. O indicates missed instance, × shows the misclassified instance and A indicates ambiguous instance.

## 3.3 Query Visualization Analysis

We further analyze the baseline’s query predictions during inference and identify two major sources of category errors: (i) high-confidence but incorrectly classified queries that introduce noisy semantics, and (ii) correctly classified yet low-confidence queries whose scores are suppressed due to motion blur or occlusion. The latter are prone to being overlooked during decoding, reducing category separability.

To validate the efectiveness of the proposed modules, we visualize the accuracy of high-confidence queries on EV18 seq\_2, which are selected as those with class confidence score $s _ { C l s } > 0 . 8$ and mean class confidence score $s _ { M a s k } > 0 . 9$ As shown in $\mathrm { F i g . 3 } .$ , blue denotes correctly classified queries and yellow denotes mismatches. After incorporating MSE and TCD, the number of high-confidence correct queries increases substantially, while high-confidence incorrect queries are markedly reduced. These results suggest that MSE compensates for ambiguous frames via cross-frame semantic retrieval, and TCD suppresses noisy representations through temporal-consistency regularization.

## 3.4 Ablation Studies and Hyper-parameter Analysis

We conduct ablation studies on EV18 to evaluate the contribution of each component and analyze the impact of the memory length K and denoising window size T. The results are reported in Table 2.

![](images/318c9dc6d59565cdbaac614979900e49b098f1dc0d1515632c94ca047bd49a03.jpg)  
Fig. 3. The visualization analysis of query predictions. Panel (a) visualizes whether high-confidence query category predictions are correctly classified on EV18 seq\_2 (frames 000–149). Panel (b) reports the statistics of queries with correct category predictions. Panel (c) reports the statistics of queries with incorrect category predictions.

Overall, introducing the query-level memory bank with MSE consistently improves category discrimination, while the TCR-guided denoising decoder further enhances temporal semantic consistency and overall segmentation performance. In terms of hyper-parameters, the best performance is achieved with $K = 3$ and $T = 3$ . Overly long memory bank or denoising window tends to introduce more historical noise and causes error accumulation over time, whereas an overly short denoising window is insuficient to fully exploit temporal-consistency cues.

## 3.5 Computational Eficiency Analysis

To evaluate computational eficiency, we conducted experiments on a single NVIDIA A100 GPU using inputs at a resolution of $5 1 2 \times 5 1 2$ . TEDi contains 60.58M parameters and requires 78.92 GFLOPs, compared with 56.91 GFLOPs for the Mask2Former baseline. During temporal inference, TEDi reuses query representations from previously encoded frames, such that only the newly arriving frame requires backbone encoding. As a result, TEDi achieves an inference latency of 57.96 ms per clip, with the memory bank introducing only approximately 5.5 ms of additional overhead and a peak GPU memory footprint of 666.1 MB.

## 4 Conclusions

We propose TEDi, a temporally enhanced framework for query-based surgical instrument segmentation that integrates MSE and TCD to leverage cross-frame priors and enforce temporal coherence. Experiments on two benchmarks show consistent state-of-the-art performance, improving category prediction stability while remaining eficient and validating the benefits of query-level memory interaction and temporal correction for surgical scene understanding.

Table 2. Ablation analysis on EV18 dataset. For column 2-3, K denotes the length of the memory bank, while T denotes the denoising window size.
<table><tr><td rowspan="2">Model</td><td colspan="2">Params.</td><td rowspan="2">Ch</td><td rowspan="2">ISI IoU</td><td rowspan="2">mc</td><td colspan="8">Instrument Categories IoU</td></tr><tr><td>K T</td><td>IoU</td><td>IoU</td><td>BF</td><td>PF</td><td>LND</td><td>SI</td><td>CA</td><td>MCS</td><td>UP</td></tr><tr><td>M2F(N=10)</td><td></td><td></td><td></td><td>84.18</td><td>81.26</td><td>48.91</td><td>86.90</td><td>43.47</td><td>60.46</td><td>54.00</td><td>4.27</td><td>93.29</td><td>0.00</td></tr><tr><td>MSE</td><td>3</td><td>-</td><td></td><td>85.65</td><td>83.26</td><td>51.67</td><td>88.40</td><td>54.77</td><td>65.87</td><td>52.83</td><td>0.00</td><td>94.11</td><td>5.72</td></tr><tr><td>TCD</td><td>-</td><td>3</td><td></td><td>85.59</td><td>83.07</td><td>51.70</td><td>88.16</td><td>49.51</td><td>62.86</td><td>57.94</td><td>0.00</td><td>93.90</td><td>9.51</td></tr><tr><td>MSE+TCD 1</td><td></td><td>5</td><td></td><td>85.65</td><td>83.69</td><td>52.56</td><td>88.00</td><td>61.19</td><td>69.79</td><td>47.90</td><td>0.00</td><td>94.07</td><td>6.97</td></tr><tr><td>MSE+TCD 2</td><td></td><td>4</td><td></td><td>85.79</td><td>83.79</td><td>52.91</td><td>88.43</td><td>56.57</td><td>66.57</td><td>55.18</td><td>0.00</td><td>94.32</td><td>9.30</td></tr><tr><td>MSE+TCD 4</td><td></td><td>2</td><td></td><td>86.47</td><td>84.28</td><td>55.23</td><td>88.80</td><td>58.47</td><td>70.94</td><td>54.18</td><td>2.32</td><td>94.16</td><td>17.72</td></tr><tr><td>MSE+TCD 5</td><td></td><td>1</td><td></td><td>85.44</td><td>83.50</td><td>51.60</td><td>88.47</td><td>56.41</td><td>67.38</td><td>50.39</td><td>0.00</td><td>94.25</td><td>4.30</td></tr><tr><td>MSE+TCD 3</td><td></td><td></td><td>3</td><td>86.69</td><td>84.82</td><td>57.41</td><td>87.45</td><td>56.22</td><td>69.91</td><td>69.96</td><td>14.82</td><td>93.94</td><td>9.59</td></tr></table>

Acknowledgments. Jiahong Yuan and Tao Zhang received support from the 7th People’s Hospital of Zhengzhou. Weiming Mi and Haoyin Zhou received support from NIH grants R00EB027177 and R01EB036996.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Allan, M., Kondo, S., Bodenstedt, S., Leger, S., Kadkhodamohammadi, R., Luengo, I., Fuentes, F., Flouty, E., Mohammed, A., Pedersen, M., et al.: 2018 robotic scene segmentation challenge. arXiv preprint arXiv:2001.11190 (2020)

2. Allan, M., Shvets, A., Kurmann, T., Zhang, Z., Duggal, R., Su, Y.H., Rieke, N., Laina, I., Kalavakonda, N., Bodenstedt, S., et al.: 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 (2019)

3. Ayobi, N., Pérez-Rondón, A., Rodríguez, S., Arbeláez, P.: Matis: Masked-attention transformers for surgical instrument segmentation. In: 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2023)

4. Ayobi, N., Rodríguez, S., Pérez, A., Hernández, I., Aparicio, N., Dessevres, E., Peña, S., Santander, J., Caicedo, J.I., Fernández, N., et al.: Pixel-wise recognition for holistic surgical scene understanding. Medical Image Analysis p. 103726 (2025)

5. Baby, B., Thapar, D., Chasmai, M., Banerjee, T., Dargan, K., Suri, A., Banerjee, S., Arora, C.: From forks to forceps: A new framework for instance segmentation of surgical instruments. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 6191–6201 (2023)

6. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1290–1299 (2022)

7. Cheng, H.K., Schwing, A.G.: Xmem: Long-term video object segmentation with an atkinson-shifrin memory model. In: European conference on computer vision. pp. 640–658. Springer (2022)

8. Dagnino, G., Kundrat, D.: Robot-assistive minimally invasive surgery: trends and future directions. International Journal of Intelligent Robotics and Applications 8(4), 812–826 (2024)

9. Dhanakshirur, R.R., Shastry, K.A., Borgavi, K., Suri, A., Kalra, P.K., Arora, C.: Learnable query initialization for surgical instrument instance segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 728–738. Springer (2023)

10. González, C., Bravo-Sánchez, L., Arbelaez, P.: Isinet: an instance-based approach for surgical instrument segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 595–605. Springer (2020)

11. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE international conference on computer vision. pp. 2961–2969 (2017)

12. Huang, Y., Bai, L., Cui, B., Yuan, K., Wang, G., Hoque, M.I., Padoy, N., Navab, N., Ren, H.: Surgtpgs: Semantic 3d surgical scene understanding with text promptable gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 584–594. Springer (2025)

13. de Jong, R., Carolus, H., Franciscus, H., van Jaarsveld, R.C., van Hillegersberg, R., Josien, P., de With, P.H., al Khalil, Y., van Der Sommen, F., et al.: Scaling up self-supervised learning for improved surgical foundation models. Medical Image Analysis p. 103873 (2025)

14. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)

15. Liu, H., Zhang, E., Wu, J., Hong, M., Jin, Y.: Surgical sam 2: Real-time segment anything in surgical video by eficient frame pruning. arXiv preprint arXiv:2408.07931 (2024)

16. Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., et al.: Swin transformer v2: Scaling up capacity and resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12009–12019 (2022)

17. Lu, B., Li, B., Chen, W., Jin, Y., Zhao, Z., Dou, Q., Heng, P.A., Liu, Y.: Toward image-guided automated suture grasping under complex environments: A learning-enabled and optimization-based holistic framework. IEEE Transactions on Automation Science and Engineering 19(4), 3794–3808 (2021)

18. Nagy, T.D., Haidegger, T.: A dvrk-based framework for surgical subtask automation. Acta Polytechnica Hungarica pp. 61–78 (2019)

19. Westebring-van der Putten, E.P., Goossens, R.H., Jakimowicz, J.J., Dankelman, J.: Haptics in minimally invasive surgery–a review. Minimally Invasive Therapy & Allied Technologies 17(1), 3–16 (2008)

20. Rai, U., Xu, H., Giannarou, S.: Surgpose: Generalisable surgical instrument pose estimation using zero-shot learning and stereo vision. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 6875–6881. IEEE (2025)

21. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

22. Salehi, S.S.M., Erdogmus, D., Gholipour, A.: Tversky loss function for image segmentation using 3d fully convolutional deep networks. In: International workshop on machine learning in medical imaging. pp. 379–387. Springer (2017)

23. Sestini, L., Rosa, B., De Momi, E., Ferrigno, G., Padoy, N.: A kinematic bottleneck approach for pose regression of flexible surgical instruments directly from images. IEEE Robotics and Automation Letters 6(2), 2938–2945 (2021)

24. Shvets, A.A., Rakhlin, A., Kalinin, A.A., Iglovikov, V.I.: Automatic instrument segmentation in robot-assisted surgery using deep learning. In: 2018 17th IEEE international conference on machine learning and applications (ICMLA). pp. 624– 628. IEEE (2018)

25. Toussaint, M., Ha, J.S., Oguz, O.S.: Co-optimizing robot, environment, and tool design via joint manipulation planning. In: 2021 IEEE International Conference on Robotics and Automation (ICRA). pp. 6600–6606. IEEE (2021)

26. Wang, Q., Zhao, S., Xu, Z., Zhou, S.K.: Lacoste: Exploiting stereo and temporal contexts for surgical instrument segmentation. Medical Image Analysis 99, 103387 (2025)

27. Zhao, Z., Jin, Y., Gao, X., Dou, Q., Heng, P.A.: Learning motion flows for semisupervised instrument segmentation from robotic surgical video. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 679–689. Springer (2020)

28. Zhao, Z., Jin, Y., Heng, P.A.: Trasetr: track-to-segment transformer with contrastive query for instance-level instrument segmentation in robotic surgery. In: 2022 International conference on robotics and automation (ICRA). pp. 11186– 11193. IEEE (2022)

29. Zhou, J., Pang, Z., Wang, Y.X.: Rmem: Restricted memory banks improve video object segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18602–18611 (2024)