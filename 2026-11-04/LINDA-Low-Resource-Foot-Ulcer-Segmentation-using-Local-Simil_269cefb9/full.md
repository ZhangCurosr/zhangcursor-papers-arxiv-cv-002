# LINDA: Low Resource Foot Ulcer Segmentation using Local Similarity and Iterative Neural Dynamics

Trishaani Acharjee<sup>1,2⋆</sup>, Sudip Chakrabarty<sup>1,2⋆</sup>, Rajdeep Chatterjee<sup>1,2⋆</sup>(B), and Tathagata Bandyopadhyay<sup>3</sup>

<sup>1</sup> School of Computer Engineering, Kalinga Institute of Industrial Technology, Bhubaneswar, India

cse.rajdeep@gmail.com

<sup>2</sup> AmygdalaAI-India Lab, Bhubaneswar, India

Department of Informatics, Technical University of Munich, Munich, Germany

Abstract. Foot ulcers are a leading cause of lower limb amputation, and the highest burden often falls on low and middle income regions where specialists and computing hardware are both scarce. Wound monitoring tools are most useful there when they are accurate yet small enough to run on ordinary devices, but accuracy usually comes from heavy networks. We present LINDA, a compact model that grounds segmentation in an explicit structural prior instead of raw intensity. A Local Self-Similarity (LSS) map, computed directly from the image and stable under local lighting changes, is fused into a ResNet-34 encoder, and a small Liquid Time-Constant module refines the boundary over four iterative steps. With every model trained under one protocol on the FUSeg 2021 dataset, LINDA obtains the best Dice (86.68%), IoU (79.67%), and boundary accuracy (HD95 6.37 pixels) in our comparison while using only 25.70M parameters, fewer than the U-Net and transformer baselines. As the LSS map is a fixed function of the input, it exposes the structural cues the model follows in a readable form. Together this makes LINDA an accurate, inspectable, and lightweight option for wound assessment in resource-constrained care.

Keywords: Foot Ulcer Segmentation · Lightweight Segmentation · Local Self-Similarity · Resource-Constrained Settings, Liquid Neural Networks

## 1 Introduction

Foot ulcers are a primary cause of non-traumatic lower-limb amputation [1], and precise delineation of the wound bed is needed to track healing and plan treatment [2]. Manual planimetry, the clinical gold standard, is subjective, slow, and varies between observers, so automated segmentation has become a priority.

In many settings, and especially in low and middle income regions, this must run with limited specialist time and limited computing hardware, which favors models that are accurate yet small enough for modest devices.

Despite recent advances [3], automated foot ulcer segmentation remains a hard problem. Ulcers vary widely in scale, lighting, and texture. The transition zones between necrotic tissue, granulation tissue, and healthy skin are often ambiguous, with low contrast and difuse gradients [4]. Widely used architectures such as U-Net variants [5] and Vision Transformers (ViTs) [6] perform well on many medical segmentation tasks, yet wound boundaries can remain imprecise when ulcer shapes are irregular and contrast with healthy skin is low. Convolutional encoders have a limited receptive field, while the patch tokenization in ViTs can blur fine boundary detail and lead to oversmoothed masks [7]. Most of these models also predict the mask in a single forward pass, unlike the way a human reader traces a boundary, revisiting and adjusting it as local and global cues are weighed.

To address these limitations, we propose LINDA, a boundary aware architecture for foot ulcer segmentation. First, a Local Self-Similarity (LSS) map is computed directly from the input image and fused into a pretrained ResNet-34 backbone, giving the early layers explicit local texture awareness. Second, a Liquid Time-Constant (LTC) network [8] at the bottleneck acts as an iterative refinement loop that resolves boundary ambiguities. Finally, the structural prior stays visible: because the LSS map is computed before encoding and used as the target of the boundary loss, it ofers a concrete, readable view of the cues the network follows, in the spirit of explainable AI in clinical imaging [9–11].

Our contributions are as follows. (1) An additive LSS fusion module injects local patch-correlation statistics into the early encoder features, anchoring the network to tissue boundaries without disturbing the pretrained weights. (2) An LTC-driven refinement loop at the bottleneck uses iterative dynamics to refine a global spatial token that conditions the decoder. (3) A Boundary Alignment Loss ties the prediction to the LSS map, giving the lowest boundary error in our comparison together with strong overlap accuracy at a low parameter count. Overall, these components form a compact, interpretable foot ulcer segmentation model that matches the strongest baselines on region overlap and clearly improves boundary precision, while using far fewer parameters, a good fit for resource-constrained clinical settings.

## 2 Previous Work

Automated wound segmentation is built mostly on the U-Net family [5] and, more recently, on transformer encoders [6, 7]. Several methods target wounds directly. Wang et al. [12] trained fully convolutional networks for wound region extraction, and WSNet [13] added a wound-oriented pipeline with dedicated pre-processing. On the FUSeg 2021 benchmark, FUSegNet [14] reported the strongest published result and remains the reference point for this dataset. State-space (Mamba) models have also entered the field, from lightweight ulcer designs like DFU-MambaLiteUNet [15] to general backbones such as MambaVision [16] and MedMamba [17], which we include as baselines. A recent benchmark, WoundAmbit [18], finds that general-purpose segmentation models can match or beat task-specific designs once training and evaluation are standardized, and it argues for attention to out-of-distribution generalization, eficiency, and interpretability. This work takes that view as its starting point: rather than competing on overlap alone, it targets tight boundaries, a small parameter budget, and a structural prior a reader can check. Evaluation on further collections such as Medetec [19] is left for future work.

![](images/c853ba15bd5842ee0fe0b9199acd16f406759dd80682fb63f1dddc7433056e28.jpg)  
Fig. 1. The proposed LINDA, illustrating the integration of the ResNet-34 encoder, Additive LSS Fusion, LTC Refinement Loop, and Deep Supervision Decoder.

Local self-similarity was introduced by Shechtman and Irani [20] to match patterns across images and videos through internal patch correlations rather than raw intensity. We reuse this idea as a deterministic prior and fold it into the encoder. For interpretability, post-hoc attribution such as Grad-CAM and SHAP is widely used in medical imaging [11, 21], but these maps are produced after training and can be unstable. Our LSS maps are fixed functions of the input and serve as the target of the boundary loss, so they give a stable, readable view of the structural evidence the model is trained to follow.

## 3 Methodology

LINDA is an end-to-end, boundary-aware segmentation network for clinical wound images; Figure 1 gives an overview. It has four modules: a ResNet-34 encoder, an additive LSS fusion block, a Liquid Time-Constant (LTC) refinement loop, and a deep-supervision decoder. Diferent modules are discussed in the following subsections.

## 3.1 Local Self-Similarity (LSS) Fusion

To add an explicit structural prior, the LSS Extractor operates on the input image I before any deep encoding. Densely overlapping local patches p<sub>i</sub> ∈ R<sup>K×K×3</sup>

are extracted, and to stay robust to the uneven illumination common in wound photography, each is zero-centered and normalized (Eq. 1):

$$
\hat { \mathbf { p } } _ { i } = \frac { \mathbf { p } _ { i } - \mu ( \mathbf { p } _ { i } ) } { \| \mathbf { p } _ { i } - \mu ( \mathbf { p } _ { i } ) \| _ { 2 } + \epsilon }\tag{1}
$$

where $\mu ( \mathbf { p } _ { i } )$ is the patch mean and ϵ is a small stability constant. Since the patches are unit-normalized, the dot product between a central patch $\hat { \bf p } _ { c }$ and each of its N neighbors within radius $R \ ( { \mathrm { E q . ~ 2 } } )$ equals their cosine similarity, which compares patch structure independently of local brightness and contrast:

$$
S ( \hat { \mathbf { p } } _ { c } , \hat { \mathbf { p } } _ { n } ) = \hat { \mathbf { p } } _ { c } \cdot \hat { \mathbf { p } } _ { n }\tag{2}
$$

This reuses the local self-similarity descriptor of Shechtman and Irani [20] as a dense, deterministic prior. Aggregating the scores at each location gives a 3- channel LSS map $\mathbf { M } _ { L S S } \in \mathbb { R } ^ { 3 \times 5 1 2 \times 5 1 2 }$ , holding the mean $( \mu )$ , maximum (max), and standard deviation $( \sigma )$ of the local correlations. $\mathrm { ~ A ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ convolution with bilinear interpolation projects it to $\mathbf { F } _ { L S S } \in \mathbb { R } ^ { 6 4 \times 2 5 6 \times 2 5 6 }$ , which is added to the first ResNet feature map C1 to form the boundary-conditioned map $\mathbf { F 1 } =$ C1 ⊕ $\mathbf { F } _ { L S S }$ that guides the remaining encoder layers.

## 3.2 Latent Refinement (LTC)

To capture global dependencies, the bottleneck LTC loop ingests the high-level feature map $\mathbf { F 5 } ~ \in ~ \mathrm { \bar { \mathbb { R } } ^ { 5 1 2 \times 1 6 \times 1 6 } }$ together with a dual-mask input, the initial coarse mask and the evolving current mask. Features and masks are globally average-pooled (GAP) and concatenated into a 514-D vector $\mathbf { x } = [ \mathrm { G A P } ( \mathbf { F 5 } )$ $\mathrm { G A P } ( \mathbf { Y } _ { m a s k s } ) ]$ . The hidden state h then evolves by the ODE of Eq. 3:

$$
\frac { d \mathbf { h } } { d t } = \left[ - \mathbf { h } + \sigma _ { r e l } ( W _ { h } \mathbf { h } + W _ { i n } \mathbf { x } + b ) \right] \odot \frac { 1 } { \tau ( \mathbf { x } ) }\tag{3}
$$

where $\sigma _ { \mathit { r e l } }$ is ReLU and $\tau ( \mathbf { x } ) = \mathrm { s o f t p l u s } ( W _ { \tau } \mathbf { x } ) + \epsilon$ is an input-dependent time constant that modulates the cell’s memory. The system is solved by Euler discretization $\begin{array} { r } { ( \mathbf { h } _ { t + 1 } = \mathbf { h } _ { t } + \varDelta t \frac { d \mathbf { h } } { d t } ) } \end{array}$ over $T = 4$ steps, and the final state is projected into a Refinement Token, a single global vector that conditions the decoder and steers the predicted boundary.

## 3.3 Optimization and Deep Supervision

The network is trained with multi-scale deep supervision. As shown in $\mathrm { F i g . }$ . 1, it produces the final mask $\hat { \mathbf Y }$ and two auxiliary predictions, $\mathbf { A u x } _ { 1 } \in \mathbb { R } ^ { 3 2 \times \breve { 3 } 2 }$ and $\mathbf { A u x } _ { 2 } ~ \in ~ \mathbb { R } ^ { 6 4 \times 6 4 }$ . The total objective combines Binary Cross-Entropy (BCE), Dice, and a geometric Boundary Alignment Loss (BAL):

$$
\mathcal { L } _ { t o t a l } = \sum _ { k \in \{ m , a 1 , a 2 \} } \lambda _ { k } \left( \mathcal { L } _ { B C E } ^ { ( k ) } + \mathcal { L } _ { D i c e } ^ { ( k ) } \right) + \lambda _ { b } \mathcal { L } _ { B A L }\tag{4}
$$

with weights $\lambda _ { m } = 1 . 0 , \lambda _ { a 1 } = 0 . 4 , \lambda _ { a 2 } = 0 . 2$ , and $\lambda _ { b } = 0 . 5$ . The BAL grounds the prediction in tissue structure by matching the Sobel gradients $\nabla ( \cdot )$ of the predicted probability map to those of the deterministic LSS Mean channel $\mathbf { M } _ { L S S } ^ { \mu } \colon$

$$
\mathcal { L } _ { B A L } = \mathrm { M S E } ( \nabla ( \sigma ( \hat { \mathbf { Y } } ) ) , \nabla ( \mathbf { M } _ { L S S } ^ { \mu } ) )\tag{5}
$$

so it acts as a structural regularizer that penalizes predictions departing from the local tissue transitions the LSS extractor marks.

## 3.4 Interpretable Structural Priors

Because the LSS extractor is deterministic, its three channels carry a fixed, readable meaning rather than the shifting saliency of post-hoc attribution [21]. LSS Mean $( \mu )$ measures local homogeneity and highlights uniform regions such as granulation tissue in the wound bed. LSS Max keeps the strongest local correlation in each neighborhood, helping thin structures survive downsampling. LSS Std (σ) responds to high local variance and marks the sharp transition between wound and healthy skin. These maps are not a claim about what individual network weights compute. They are deterministic inputs to the encoder, and the Mean channel is also the target of the boundary loss $\left( \operatorname { E q . 5 } \right)$ , so this part of the pipeline can be inspected directly.

## 3.5 Compute, Data and Inference Time

For both training and evaluation, we have used single NVIDIA A100 GPU with 40 GB VRAM. We have performed all the experiments on MICCAI Foot Ulcer Segmentation (FUSeg) Challenge dataset [22], details of which are described in sec 4.1. Our model takes roughly 5 hours to train and ∼ 45 ms per image, on average, for inference.

## 4 Results

## 4.1 Dataset and Implementation Details

We use the MICCAI Foot Ulcer Segmentation (FUSeg) Challenge dataset [22], which has 1,210 images (810 train, 200 validation, 200 test) from 889 patients under varied lighting and backgrounds. The test labels are held private by the organizers, so all metrics are reported on the oficial validation set. Sample images are shown in Figure 2. Images and masks are processed at 512 × 512, the grid also used for prediction. LINDA is implemented in PyTorch 2.5.1 and trained on an NVIDIA A100. For a fair comparison, every model uses the same setup: 100 epochs, batch size 2, the Adam optimizer, and the checkpoint with the best validation Dice.

The ResNet-34 encoder uses ImageNet-pretrained weights, while the LSS Extractor and LTC loop are trained from scratch. The task is binary, wound against background; the necrotic and granulation regions noted earlier describe the visual dificulty inside the wound, not separate classes. We report Dice (DSC), IoU, and the 95th-percentile Hausdorf Distance (HD95), all on the wound class.

![](images/72a8b88b8b10c06dfa9875a45e2f28cf5364fc0e84c7bad24a5bb724bd5e290c.jpg)  
Fig. 2. Sample images and ground-truth masks from the FUSeg 2021 challenge dataset.

Table 1. Quantitative comparison of LINDA against representative segmentation methods under a unified training protocol.
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>Dice (%) IoU (%) (↑) (↑)</td><td>HD95 (↓)</td></tr><tr><td>VGG16-UNet [23]</td><td>69.31</td><td>42.19</td><td>84.88</td><td>77.29 14.72</td></tr><tr><td>TransUNet [6]</td><td>130.22</td><td>102.00</td><td>80.16 71.32</td><td>15.32</td></tr><tr><td>UNet [5]</td><td>31.78</td><td>374.34</td><td>84.04 76.23</td><td>13.31</td></tr><tr><td>Mask R-CNN [24]</td><td>90.44</td><td>67.13</td><td>83.97</td><td>76.42 12.89</td></tr><tr><td>ResNet101-UNet [25]</td><td>267.62</td><td>112.79</td><td>84.92 77.73</td><td>13.03</td></tr><tr><td>U2Seg [26]</td><td>86.47</td><td>88.46</td><td>75.85</td><td>70.13 16.07</td></tr><tr><td>SegNet [27]</td><td>69.21</td><td>42.19</td><td>85.32 78.11</td><td>12.71</td></tr><tr><td>FUSegNet [14]</td><td>67.22</td><td>25.82</td><td>86.43</td><td>79.41 16.06</td></tr><tr><td>MambaVision-UNet [16]</td><td>35.22</td><td>21.90</td><td>85.25</td><td>77.91 13.19</td></tr><tr><td>MedMamba-UNet [17]</td><td>18.36</td><td>43.30</td><td>84.16</td><td>76.30 15.74</td></tr><tr><td>LSS-Mamba</td><td>35.39</td><td>46.85</td><td>85.96</td><td>78.68 16.15</td></tr><tr><td>LINDA (Ours)</td><td>25.70</td><td>82.45</td><td>86.68</td><td>79.67 6.37</td></tr></table>

## 4.2 Comparison with State-of-the-art Methods

Table 1 compares LINDA with segmentation models re-trained under one protocol, which keeps the comparison fair. Under this setting LINDA gives the best Dice (86.68%), IoU (79.67%), and boundary accuracy (HD95 6.37 pixels) in the table. The boundary margin is the clearest: at 6.37 pixels the HD95 is roughly half that of the next-best model, which points to tighter agreement along the wound rim. LINDA reaches these numbers with 25.70 M parameters, far fewer than heavy encoders such as ResNet101-UNet [25], TransUNet [6], and Mask R-CNN [24], which supports its use where computing is limited. FUSegNet [14], the FUSeg 2021 challenge winner, is included as a domain-specific reference; under the same 100-epoch schedule it trails LINDA on all three metrics, while its originally published Dice, obtained with heavier training, is higher.

Table 2. Ablation study progressively adding LSS, LTC, and BAL to the baseline.
<table><tr><td>Model Variation</td><td>LSS LTC BAL</td><td>Dice (%)(↑)</td><td>IoU (%)(↑)</td><td>Params (M)</td></tr><tr><td>Baseline (ResNet-34 + U-Net)</td><td>√</td><td>85.22 83.51</td><td>77.56</td><td>24.87</td></tr><tr><td>Baseline + LSS Baseline + LTC</td><td>√</td><td>83.29</td><td>75.51 75.22</td><td>25.12 25.45</td></tr><tr><td>Baseline + LSS + BAL</td><td>V V</td><td>86.41</td><td>79.26</td><td>25.12</td></tr><tr><td>LSS + Decoder only + BAL</td><td>√ √</td><td>46.09</td><td>34.61</td><td>3.01</td></tr><tr><td>LINDA (w/o BAL)</td><td>V √</td><td>76.18</td><td>69.74</td><td>25.70</td></tr><tr><td>LINDA (Proposed)</td><td>√ √ √</td><td>86.68</td><td>79.67</td><td>25.70</td></tr></table>

## 4.3 Ablation Studies

Model Components: Table 2 reports the ablation on diferent components of the proposed model. Adding LSS and LTC to the ResNet-34 baseline without the Boundary Alignment Loss (BAL) drops Dice from 85.22% to 76.18%: pixel-wise losses alone do not bring the structural prior and the refinement together. BAL fixes this. With LSS and BAL the model reaches 86.41% Dice, and the LTC loop takes the full model to 86.68% Dice and 79.67% IoU. Two further rows isolate the rest: keeping only LSS with the decoder, without the pretrained encoder, collapses to 46.09% Dice, so the encoder is essential, while the gap from LSS+BAL to the full model is the LTC contribution. The full model adds only 0.83 M parameters over the baseline, and this pattern reinforces the interpretability point: the LSS prior is used well only once the boundary loss ties the prediction to it.

Number of refinement steps: Table 3 varies the number of LTC steps T. Performance is stable for T=1 to T=8, and T=4 gives the best balance of overlap (86.68% Dice, 79.67% IoU) and boundary accuracy (HD95 6.37) at a low step count. Beyond T=8 it collapses, with Dice below 73% and HD95 above 22 pixels, consistent with the Euler-discretized ODE becoming unstable over too many steps. We therefore use T=4.

## 4.4 A Closer Look at the LSS Maps

Figure 3 shows the three LSS channels for several wounds. The Std channel behaves like a local edge map: it stays quiet over uniform background and responds along the wound rim, which is exactly where boundary errors are most costly. These maps feed the encoder, and the Mean channel is the target of the boundary loss (Eq. 5), so they give a direct, image-level account of the structural information the model is pushed to respect. We therefore treat them as a way to inspect the model’s inputs and training signal, not as a measurement of how the deeper layers reach their final decision.

Table 3. Efect of the number of LTC refinement steps T on the FUSeg validation set. Dice/IoU in %, HD95 in pixels; T=4 (shaded) is used.
<table><tr><td>T</td><td>Dice (↑)</td><td>IoU (↑)</td><td>HD95 (4)</td><td>T</td><td>Dice (↑)</td><td>IoU (↑)</td><td>HD95 (4)</td></tr><tr><td>1</td><td>86.28</td><td>79.33</td><td>6.93</td><td>8</td><td>86.26</td><td>79.33</td><td>6.64</td></tr><tr><td>2</td><td>85.27</td><td>78.10</td><td>11.36</td><td>9</td><td>69.21</td><td>57.54</td><td>30.91</td></tr><tr><td>3</td><td>84.66</td><td>77.12</td><td>9.71</td><td>10</td><td>67.22</td><td>55.67</td><td>31.02</td></tr><tr><td>4</td><td>86.68</td><td>79.67</td><td>6.37</td><td>11</td><td>72.47</td><td>62.35</td><td>22.92</td></tr><tr><td>5</td><td>85.55</td><td>78.31</td><td>9.34</td><td>16</td><td>69.79</td><td>58.55</td><td>27.23</td></tr><tr><td>6</td><td>86.65</td><td>79.57</td><td>6.59</td><td>20</td><td>48.72</td><td>41.26</td><td>48.35</td></tr><tr><td>7</td><td>86.21</td><td>79.11</td><td>6.33</td><td>32</td><td>68.57</td><td>57.62</td><td>30.63</td></tr></table>

![](images/14759029d39b4a0de538337394378211e972bd7a3e690bf016be65823292ff1a.jpg)  
Fig. 3. The three LSS channels (Mean, Max, and Std) for sample wounds. The Std channel highlights the wound rim, while Mean and Max capture local homogeneity and strong local structure.

## 5 Discussion

We presented LINDA, a foot ulcer segmentation model that fuses a deterministic Local Self-Similarity prior into a ResNet-34 encoder and refines the prediction with a small Liquid Time-Constant module and a boundary alignment loss. It produces accurate masks with the tightest boundaries in our comparison at only 25.70 M parameters, and since the LSS maps are computed before encoding, they also give a readable view of the cues the model follows, with no separate explanation step, thus making it a faithful and explainable segmentation model.

## 6 Impact in RCS

In resource constrained setting, it is crucial to balance performance, model size and inference speed. Our proposed approach, achieves the best tradeof between model size (25.70 M parameters), training time (∼ 5 hrs) and per-image inference time (∼ 45 ms). LINDA achieves state-of-the-art scores in Dice and HD95 metrics with a reasonably low number of parameters and GFLOPS, as compared to its closest competitors, thus making it an excellent choice for a wound segmentation model in a resource constrained setting.

Acknowledgments. No acknowledgments or competing interests.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. David G Armstrong, Andrew JM Boulton, and Sicco A Bus. Diabetic foot ulcers and their recurrence. New England Journal of Medicine, 376(24):2367–2375, 2017.

2. Bill Cassidy, Neil D Reeves, Joseph M Pappachan, David Gillespie, Claire O’Shea, Sanjeev Rajbhandari, Arun G Maiya, Eibe Frank, Andrew JM Boulton, and Moi Hoon Yap. The dfuc 2020 dataset: Analysis towards diabetic foot ulcer detection. IEEE Journal of Biomedical and Health Informatics, 26(1):346–356, 2021.

3. Suhaylah Alkhalefah, Isra AlTuraiki, and Najwa Altwaijry. Advancing diabetic foot ulcer care: Ai and generative ai approaches for classification, prediction, segmentation, and detection. In Healthcare, volume 13, page 648. MDPI, 2025.

4. Manu Goyal, Neil D Reeves, Andrew K Davison, Sanjeev Rajbhandari, John Sprague, and Moi Hoon Yap. Recognition of ischemia and infection in diabetic foot ulcers: Dataset and techniques. Computers in Biology and Medicine, 117:103616, 2020.

5. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015, pages 234–241. Springer, 2015.

6. Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L Yuille, and Yuyin Zhou. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.

7. Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Tseng, Juan Idrees, Anwar Li, Holger Roth, and Daguang Xu. Unetr: Transformers for 3d medical image segmentation. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 574–584, 2022.

8. Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and Radu Grosu. Liquid time-constant networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 7657–7666, 2021.

9. Sudip Chakrabarty, Pappu Bishwas, Mainak Bandyopadhyay, and Jérémie Sublime. Can we trust ai with our ears? a cross-domain comparative analysis of explainability in audio intelligence. IEEE Access, 13:179733–179758, 2025.

10. Alican Akman and Björn W Schuller. Audio explainable artificial intelligence: A review. Intelligent Computing, 3:0074, 2024.

11. Bas HM Van der Velden, Hugo J Kuijf, Kenneth GA Gilhuijs, and Max A Viergever. Explainable artificial intelligence (xai) in deep learning-based medical image analysis. Medical image analysis, 79:102470, 2022.

12. Chuanbo Wang, D. M. Anisuzzaman, Victor Williamson, Mrinal Kanti Dhar, Behrouz Rostami, Jefrey Niezgoda, Sandeep Gopalakrishnan, and Zeyun Yu. Fully automatic wound segmentation with deep convolutional neural networks. Scientific Reports, 10(1):21897, 2020.

13. Subba Reddy Oota, Vijay Rowtula, Shahid Mohammed, Minghu Liu, and Manish Gupta. WSNet: Towards an efective method for wound image segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3234–3243, 2023.

14. Mrinal Kanti Dhar, Taiyu Zhang, Yash Patel, Sandeep Gopalakrishnan, and Zeyun Yu. FUSegNet: A deep convolutional neural network for foot ulcer segmentation. Biomedical Signal Processing and Control, 92:106057, 2024.

15. Yessica Nataliani, Hindriyanto Dwi Purnomo, Ivanna Kristianti Timotius, and Purwono Purwono. DFU-MambaLiteUNet: A lightweight and eficient model for diabetic foot ulcer segmentation. Expert Systems with Applications, 293:128689, 2025.

16. Ali Hatamizadeh and Jan Kautz. MambaVision: A hybrid mamba-transformer vision backbone. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

17. Yubiao Yue and Zhenzhang Li. MedMamba: Vision mamba for medical image classification. arXiv preprint arXiv:2403.03849, 2024.

18. Vanessa Borst, Timo Dittus, Tassilo Dege, Astrid Schmieder, and Samuel Kounev. WoundAmbit: Bridging state-of-the-art semantic segmentation and real-world wound care. In Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track (ECML PKDD). Springer, 2025.

19. Stephen Thomas. Medetec wound database, 2017. http://www.medetec.co.uk.

20. Eli Shechtman and Michal Irani. Matching local self-similarities across images and videos. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1–8. IEEE, 2007.

21. Usha Bhalla, Suraj Srinivas, and Himabindu Lakkaraju. Discriminative feature attributions: Bridging post hoc explainability and inherent interpretability. Advances in neural information processing systems, 36:44105–44122, 2023.

22. Chuanbo Wang, Amirreza Mahbod, Isabella Ellinger, Adrian Galdran, Sandeep Gopalakrishnan, Jefrey Niezgoda, and Zeyun Yu. Fuseg: The foot ulcer segmentation challenge. Information, 15(3):140, 2024.

23. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR), 2015.

24. Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2961–2969, 2017.

25. Foivos I Diakogiannis, François Waldner, Peter Caccetta, and Chen Wu. Resuneta: A deep learning framework for semantic segmentation of remote sensing data. volume 162, pages 94–114, 2020.

26. Dantong Niu, Xudong Wang, Xinyang Han, Long Lian, Roei Herzig, and Trevor Darrell. Unsupervised universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

27. Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE TPAMI, 39(12):2481–2495, 2017.