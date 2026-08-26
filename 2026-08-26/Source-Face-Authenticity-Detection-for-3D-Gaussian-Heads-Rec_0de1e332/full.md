# Source-Face Authenticity Detection for 3D Gaussian Heads Reconstructed from a Single Portrait: A Benchmark and Dedicated Detector

Yujie Gao<sup>1</sup>, Zijian Yu<sup>2</sup>, Yan Hong<sup>2</sup>, Jun Lan<sup>2</sup>, Jianfu Zhang<sup>1∗</sup>

<sup>1</sup>Shanghai Jiaotong University <sup>2</sup>Ant Group

GYJgyj123@sjtu.edu.cn; thss15\_yuzj@163.com; yanhong.sjtu@gmail.com; lanjun\_yelan@163.com; c.sis@sjtu.edu.cn

## Abstract

Recent advances in single-image 3D Gaussian head reconstruction have enabled highly realistic and freely renderable digital heads from a single portrait. However, reconstruction and rendering can weaken the forgery traces in the source portrait, making the resulting 3D face dificult to classify whether its underlying face is real or fake, and thereby posing risks to identity authentication and face privacy. To study this problem, we introduce the first large-scale benchmark for this task by collecting real portraits and fake portraits from multiple sources and evaluate representative existing detectors on this benchmark, revealing their lack of explicit mechanisms for retaining fine-grained information and maintaining feature consistency across rendered views. To directly address these two limitations, we propose a detector trained with a twostage strategy. In Stage I, masked autoencoding encourages the visual backbone to retain the fine-grained appearance information required for local reconstruction, while multi-view contrastive learning enforces feature consistency across rendered views of the same head. Since CLS tokens at diferent depths exhibit complementary spatial attention patterns, Stage II freezes the adapted backbone and concatenates low-, middle-, and high-level CLS tokens for classification. Experiments show that our method achieves the highest accuracy and ranks first across all reported metrics among the evaluated detectors.

## 1 Introduction

Recent advances in single-image 3D Gaussian head generation have made it increasingly practical to recover photorealistic and renderable 3D heads from only one portrait, enabling applications in digital humans, telepresence, content creation, and AR/VR. Yet a reconstructed 3D head is also an identity-bearing asset: when generated from a forged or manipulated portrait, a 3D Gaussian head may provide a persistent, freely renderable identity asset that can facilitate impersonation or attacks on identity-authentication systems. This makes authenticity detection essential for protecting both identity authentication and face privacy in 3D face applications. However, existing face forgery and fake image detectors mainly target 2D images or videos, and no dedicated method has yet been developed to determine, from a single rendered view of a 3D Gaussian head, whether its underlying source face is real or fake.

The evolution of single-image 3D head generation reflects a steady move toward more expressive and realistic representations. Early methods typically relied on fitting parametric face models, such as 3D morphable models (Blanz and Vetter 2023) and FLAME (Li et al. 2017), to estimate facial shape, expression, and pose from an input image. Later neural approaches introduced more flexible implicit 3D representations, including tri-plane features (An et al. 2023; Zhang et al. 2024; Li et al. 2024) and NeRF-based radiance fields (Gafni et al. 2021). More recently, 3D Gaussian Splatting (Kerbl et al. 2023) has become a promising representation for head reconstruction. By optimizing dense Gaussians from a single portrait, recent 3D Gaussian head methods (Taubner et al. 2025; Lyu et al. 2025; He et al. 2025; Gao et al. 2026) can generate increasingly photorealistic heads that faithfully preserve the identity, geometry, expression, and texture cues of the input image across novel views, which also poses new challenges for identity authentication and face privacy protection.

![](images/acff593b5aa832e5c345c1d9194dc0601031e7c9f57f2359c7d5c91f40b82f6b.jpg)  
Figure 1: Task definition. Given a 3D Gaussian head reconstructed from a single portrait image, our task is to determine whether its underlying face is real or fake. At inference time, the authenticity decision is made from a single rendered view of the reconstructed head.

In parallel, 2D visual authenticity detection has been extensively studied for both face forgery and general fake-image detection (Huang et al. 2023; Rössler et al. 2019; Cui et al. 2025; Tan et al. 2024a; Zhou et al. 2026; Yan et al. 2025b; Tan et al. 2024b). Existing methods primarily distinguish real and fake content using manipulation artifacts, texture inconsistencies, generator fingerprints, or statistical cues in 2D images and videos. However, directly applying these 2D detectors to authenticity detection for 3D Gaussian heads reconstructed from a single image presents substantial challenges: (1) 3D Gaussian reconstruction and re-rendering may compress or alter the original 2D forgery cues, leaving only subtle, localized evidence. Existing detectors lack explicit mechanisms to retain such fine-grained information, making these cues dificult to capture. (2) Changes in rendering viewpoint alter the visibility, scale, and projection of these cues, yielding diferent local evidence patterns across views of the same 3D Gaussian head. As a result, detectors trained only with standard 2D supervision may overfit to view-specific rendering shortcuts rather than learning view-robust authenticity cues. These challenges motivate a detector that preserves fine-grained appearance information in the 3DGS rendering domain and learns consistent authenticity representations across rendered views, even when only a single view is used at inference time.

![](images/a878de0de3f8a79938bef07bde06664c5f7df4aa61e29d22042d2174eb028c6c.jpg)  
Figure 2: Overview of our 3D Gaussian head dataset.

To systematically study this problem, we make three main contributions: (1) We introduce, to the best of our knowledge, the first large-scale dataset for real/fake 3D Gaussian head detection, where 2D forged faces from 9 methods across text-to-image generation, face swapping, and expression-driven synthesis, together with genuine portraits from 5 real-face datasets, are reconstructed by 5 single-image 3D head methods and rendered into approximately 360K multi-view 3DGS head samples; (2) We establish a benchmark on this dataset and evaluate representative baselines from both face forgery detection and general fake image detection, revealing their limitations in retaining fine-grained authenticity cues after Gaussian reconstruction and maintaining feature consistency across rendered views; (3) We show that reliable 3DGS head detection benefits from preserving fine-grained appearance information and enforcing cross-view feature consistency, while aggregating regional cues across network depths further improves classification.

Guided by these findings, we propose a two-stage detector: Stage I combines masked autoencoding (He et al. 2022) for detail-preserving representation learning with multi-view contrastive learning for cross-view consistency, while Stage II freezes the adapted backbone and fuses low-, middle-, and high-level CLS tokens for single-view classification.

In summary, our contributions are:

• We introduce the first large-scale dataset for real/fake 3D Gaussian head detection, comprising approximately 360K multi-view renderings.

• We establish a systematic benchmark with balanced, identity-disjoint splits and evaluate representative recent 2D authenticity detectors in the 3DGS head domain under a unified protocol.

• We propose a two-stage detector that combines masked autoencoding and multi-view contrastive learning for representation learning, followed by multi-level CLS token fusion for classification.

## 2 Related Works

3D Gaussian Head Reconstruction from Single Image: 3D Gaussian head reconstruction from a single portrait image has recently become an active research direction. Compared with earlier mesh- or implicit-field-based representations, 3DGS provides an explicit and eficiently renderable representation that can faithfully reproduce the input view while plausibly completing invisible regions of the head. Existing methods can be roughly grouped into several lines. Some approaches adopt a two-stage pipeline, first generating multi-view observations from the input portrait and then reconstructing a Gaussian point cloud from these synthesized views, as exemplified by FaceLift (Lyu et al. 2025). Another line incorporates parametric head priors, especially FLAME, into Gaussian representations to improve geometric stability, animation controllability, and reconstruction quality, including LAM (He et al. 2025), CAP4D (Taubner et al. 2025), and GAGAvatar (Chu and Harada 2024). More recent eforts further pursue one-step and eficient generation, aiming to reduce reconstruction latency while preserving high-quality full-head appearance, such as FastAvatar (Liang et al. 2025) and Any3DAvatar (Gao et al. 2026). These advances make single-image 3DGS head reconstruction increasingly practical, but they also raise the need to verify whether the reconstructed head originates from a real or fake face. Recent work such as Fake3DGS (Di Nucci et al. 2026; Han et al. 2026) has explored detecting manipulations of 3DGS representations, but it focuses on tampering with the Gaussian point cloud rather than the authenticity of the underlying source face and is therefore not directly applicable to our task.

AI-Generated Face Detection: Contemporary face forgeries mainly arise from three categories: text-to-image face generation (Wu et al. 2025; Z-Image Team et al. 2025; Black Forest Labs 2025; Google 2025; OpenAI 2025), expressiondriven face reenactment (Guo et al. 2024; Li et al. 2026; Zhao et al. 2025), and face swapping (Shiohara, Yang, and Taketomi 2023; Groshev et al. 2022; Wang et al. 2021; Chen et al. 2020). Accordingly, existing detection methods have developed along two major directions. The first is tailored specifically to facial content and exploits face-specific forensic evidence, such as identity inconsistency, blending boundaries, and localized manipulation artifacts, to distinguish forged faces from genuine ones (Huang et al. 2023; Rössler et al. 2019; Cui et al. 2025; Xu et al. 2023; Wang and Deng 2021; Zhao et al. 2021; Sun et al. 2022; Yan et al. 2024; Shiohara and Yamasaki 2022). The second treats forged faces as a special case of general AI-generated imagery and develops detectors over broader image domains (Yan et al. 2025a,b; Zhou et al. 2026; Tan et al. 2024b; Chen et al. 2025, 2024). By learning from diverse semantic content and generation pipelines, these general-purpose methods can capture a wider spectrum of forgery traces and often exhibit stronger crossgenerator generalization. Nevertheless, both directions are designed for native 2D images, leaving the authenticity detection of faces reconstructed and rendered as 3D Gaussian heads largely unexplored.

![](images/10d98097c29646bc771b015f537bb360f529a2c764bdf4c247c1dd690378def6.jpg)  
Figure 3: Overview of our two-stage training framework. In Stage 1, we train the visual backbone with two self-supervised objectives: masked autoencoding models local texture details and subtle Gaussian artifacts, while multi-view contrastive learning suppresses view-specific shortcuts. In Stage 2, low-, middle-, and high-level CLS tokens are normalized and concatenated, then fed into an MLP classification head for real/fake prediction.

## 3 Methodology

## 3.1 Dataset Construction

Recent single-image 3D Gaussian head reconstruction methods (Taubner et al. 2025; Lyu et al. 2025; He et al. 2025; Chu and Harada 2024; Gao et al. 2026; Liang et al. 2025; Gerogiannis et al. 2025) can generate high-fidelity 3D heads that nearly replicate the input portrait, particularly when rendered from viewpoints close to the input view. Consequently, forged or manipulated portraits can become persistent, freely renderable 3D identity assets, threatening identity authentication and intellectual property protection. To address this risk, we construct, to the best of our knowledge, the first large-scale dataset for real/fake 3D Gaussian head detection. Dataset Details: Our dataset contains 361,469 rendered images from 16,372 unique identities, comprising 170,060 rea and 191,409 fake samples. Fig. 2 presents representative examples from the resulting dataset. We collect real portraits from 5 real-face datasets: CelebV-HQ (Zhu et al. 2022), Ava-256 (Martinez et al. 2024), FFHQ (Karras, Laine, and Aila 2019), NeRSemble (Kirschstein et al. 2023), and VFHQ (Xie et al. 2022). We employ 9 2D face forgery methods to construct the fake portraits. Both real and fake portraits are then reconstructed using 5 single-image 3D Gaussian head methods. Each reconstructed head is rendered at a resolution of 512 × 512 from 5–10 randomly sampled viewpoints, with yaw and pitch ranging from −45<sup>◦</sup> to 45<sup>◦</sup> and the camera radius sampled from 0.8 to 1.2 relative to a normalized standard head scale.

Fake Head Generators: To construct fake 3D heads, we first generate forged 2D portraits spanning three categories: textto-image generation, face swapping, and expression-driven synthesis. For each category, we select a combination of recent and well-established forgery methods to cover both contemporary generation capabilities and representative classic pipelines. For text-to-image generation, we define 10 facial and appearance attributes, including gender, hairstyle, expression, and ethnicity, and prepare 50 candidate values for each attribute. We randomly sample one value from each attribute and combine the selected values into a prompt, which is then fed into Z-Image (Z-Image Team et al. 2025), Qwen-Image (Wu et al. 2025), GPT-Image-2 (OpenAI 2025), Nano Banana Pro (Google 2025), or FLUX.2 (Black Forest Labs 2025). For face swapping and expression-driven synthesis, we randomly sample two portraits from the same real face datasets used for the genuine subset. The two portraits serve as the source and target for face swapping with HyperSwapper (FaceFusion Team 2025) or BlendFace (Shiohara, Yang, and Taketomi 2023), and as the source and driving reference for expression transfer with LivePortrait (Guo et al. 2024) or PersonaLive (Li et al. 2026). Drawing these inputs from the real data sources helps reduce identity-distribution bias between real and fake samples. Each resulting forged portrait is subsequently reconstructed using one of five single-image 3D Gaussian head methods. We include both optimizationbased and feed-forward pipelines: CAP4D (Taubner et al. 2025), FaceLift (Lyu et al. 2025), LAM (He et al. 2025), GAGAvatar (Chu and Harada 2024), and Any3DAvatar (Gao et al. 2026). The reconstructed heads are finally rendered from multiple randomly sampled viewpoints using the protocol described above to produce the fake 3D head samples.

## 3.2 Two Stage Training

Existing 2D detectors face two challenges. First, 3D Gaussian reconstruction and re-rendering weaken forgery cues, leaving only subtle local evidence. Second, viewpoint changes cause this evidence to vary across views of the same head. As shown in Fig. 3, we address these challenges with a twostage framework that first learns representations that preserve fine-grained appearance details and remain consistent across views, and then freezes the visual backbone and fuses low-, middle-, and high-level CLS tokens for authenticity classification.

Self-Supervised Training: Given an image batch B, Each batch $\bar { B } \ = \ \{ { \mathcal G } _ { g } \} _ { g = 1 } ^ { G }$ contains G identity groups, where $\mathcal { G } _ { g } ~ = ~ \{ \mathbf { x } _ { g , k } \} _ { k = 1 } ^ { K }$ consists of K views rendered from the same 3D head, while diferent groups correspond to diferent identities. The total batch size is $B = G K$ . For the MAE branch, we flatten the grouped batch as $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { B }$ and divide each image into $P$ non-overlapping patches $\{ \mathbf { p } _ { i , j } \} _ { j = 1 } ^ { P } .$

MaskedAutoencoding. To adapt the backbone to the 3DGS rendering domain and encourage it to retain fine-grained local appearance information, we randomly mask a ratio $\rho$ of the patches in the MAE branch, forming a masked index set $\mathcal { M } _ { i }$ and a visible index set $\nu _ { i }$ , where $| \bar { \mathcal { M } } _ { i } | = \rho P$ and $\nu _ { i } =$ $\{ 1 , \ldots , P \} \backslash { \mathcal { M } } _ { i }$ . Only the visible patches are embedded and passed through a ViT-style visual backbone $f _ { \theta }$ to produce image tokens

$$
{ \bf Z } _ { i } = f _ { \theta } \left( \{ { \bf p } _ { i , j } \} _ { j \in \mathcal { V } _ { i } } \right) .\tag{1}
$$

We restore the image tokens to their original patch positions, insert a learned mask token at each masked position, and add positional embeddings. The complete token sequence is then processed by an MAE decoder $g _ { \phi }$ composed of N ViT blocks. A linear layer maps each decoder output token to the flattened RGB pixels of its corresponding patch, yielding the prediction $\hat { \mathbf { p } } _ { i , j }$ . We reconstruct only the unseen patches using

$$
\mathcal { L } _ { \mathrm { M A E } } = \frac { 1 } { B \rho P } \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { M } _ { i } } \left. \hat { \mathbf { p } } _ { i , j } - \mathbf { p } _ { i , j } \right. _ { 2 } ^ { 2 } .\tag{2}
$$

The reconstruction objective does not directly supervise authenticity labels or artifact locations. Instead, inferring missing content from visible patches provides target-domain selfsupervision and encourages the backbone to preserve structural and fine-grained appearance information. We hypothesize that this information-preserving representation provides a stronger basis for the downstream real/fake classifier, a hypothesis evaluated by the component ablation in Experiment Section.

Multi-View Contrastive Learning. Unlike the MAE branch, the contrastive branch feeds the complete patch sequence of each view into the visual backbone without masking. We extract and $\ell _ { 2 }$ -normalize its CLS token as the latent representation $\mathbf { z } _ { g , k }$ . Let ${ \mathcal { T } } = \{ ( g , k ) \}$ denote all views in the batch. For an anchor $\boldsymbol { a } ~ = ~ \left( \boldsymbol { g } , \boldsymbol { k } \right)$ , its positive set ${ \mathcal { P } } ( a ) = \{ ( g , q ) ~ | ~ q \neq k \}$ contains the other views in the same group, while ${ \mathcal { A } } ( a ) \bar { = } \mathcal { T } \backslash \{ a \}$ contains all non-anchor views. We first define the normalized similarity between anchor a and candidate $p$ as

$$
q ( a , p ) = \frac { \exp ( \mathbf { z } _ { a } ^ { \top } \mathbf { z } _ { p } / \tau ) } { \sum _ { n \in \mathcal { A } ( a ) } \exp ( \mathbf { z } _ { a } ^ { \top } \mathbf { z } _ { n } / \tau ) } ,\tag{3}
$$

where $\tau$ is the temperature. The multi-positive contrastive loss is

$$
\mathcal { L } _ { \mathrm { c o n } } = - \frac { 1 } { B } \sum _ { a \in \mathcal { T } } \frac { 1 } { | \mathscr { P } ( a ) | } \sum _ { p \in \mathcal { P } ( a ) } \log q ( a , p ) .\tag{4}
$$

This objective pulls views of the same head together in the latent space and pushes views from diferent identity groups apart, promoting viewpoint-invariant authenticity features.

Multi-Level Feature Fusion and Classification: Diferent network depths produce visual representations at diferent levels of abstraction. Low-, middle-, and high-level representations may each contain information useful for real/fake classification, and such information may not be fully retained in the final-layer CLS token. We therefore fuse CLS tokens from multiple depths, allowing the classifier to exploit complementary representations across the network. In Stage 2, we freeze the self-supervised visual backbone $f _ { \theta }$ and train only an MLP classifier $m _ { \psi }$ . Given a full, unmasked image $\mathbf { x } _ { i }$ , let the backbone contain L ViT blocks and let $\mathbf { c } _ { i } ^ { ( \ell ) }$ denote the CLS token output by block ℓ:

$$
\mathbf { c } _ { i } ^ { ( \ell ) } = \left[ f _ { \theta } ^ { ( \ell ) } ( \mathbf { x } _ { i } ) \right] _ { \mathrm { C L S } } , \quad \ell \in S = \{ \ell _ { \mathrm { l o w } } , \ell _ { \mathrm { m i d } } , \ell _ { \mathrm { h i g h } } \} .\tag{5}
$$

These layers provide low-, middle-, and high-level representations, respectively. We first $\ell _ { 2 }$ -normalize each CLS token:

$$
\bar { \mathbf { c } } _ { i } ^ { ( \ell ) } = \frac { \mathbf { c } _ { i } ^ { ( \ell ) } } { \| \mathbf { c } _ { i } ^ { ( \ell ) } \| _ { 2 } } , \quad \ell \in \mathcal { S } .\tag{6}
$$

The normalized tokens are then concatenated into a fused feature $\mathbf { h } _ { i } { \mathrm { : } }$

$$
\mathbf { h } _ { i } = [ \bar { \mathbf { c } } _ { i } ^ { ( \ell _ { \mathrm { l o w } } ) } ; \bar { \mathbf { c } } _ { i } ^ { ( \ell _ { \mathrm { m i d } } ) } ; \bar { \mathbf { c } } _ { i } ^ { ( \ell _ { \mathrm { h i g h } } ) } ] .\tag{7}
$$

The MLP produces the real/fake prediction

$$
\hat { \mathbf { y } } _ { i } = \mathrm { s o f t m a x } \left( m _ { \psi } \left( \mathbf { h } _ { i } \right) \right) ,\tag{8}
$$

and is optimized using the cross-entropy loss

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { c = 1 } ^ { 2 } y _ { i , c } \log \hat { y } _ { i , c } .\tag{9}
$$

<table><tr><td rowspan="2">Method</td><td colspan="4">2D Split</td><td colspan="5">3D Split</td><td rowspan="2">Total</td></tr><tr><td>T2I</td><td>ExpDriven</td><td>FaceSwap</td><td>Real</td><td>Any3DAvatar</td><td>Cap4D</td><td>FaceLift</td><td>GAGAvatar</td><td>LAM</td></tr><tr><td>IID</td><td>89.32</td><td>54.04</td><td>71.72</td><td>85.05</td><td>83.60</td><td>77.97</td><td>77.20</td><td>73.00</td><td>80.10</td><td>78.60</td></tr><tr><td>NPR</td><td>88.72</td><td>70.04</td><td>76.80</td><td>84.84</td><td>86.60</td><td>81.67</td><td>82.07</td><td>73.57</td><td>84.50</td><td>81.68</td></tr><tr><td>FreqNet</td><td>83.48</td><td>48.64</td><td>54.64</td><td>79.04</td><td>76.83</td><td>71.80</td><td>67.07</td><td>65.63</td><td>71.90</td><td>70.65</td></tr><tr><td>ForAda</td><td>95.76</td><td>58.28</td><td>79.20</td><td>89.15</td><td>89.37</td><td>79.80</td><td>84.13</td><td>78.20</td><td>85.73</td><td>82.49</td></tr><tr><td>AIDE</td><td>95.12</td><td>56.68</td><td>68.60</td><td>72.56</td><td>75.37</td><td>71.93</td><td>72.17</td><td>72.27</td><td>73.33</td><td>73.01</td></tr><tr><td>Effort</td><td>96.20</td><td>63.40</td><td>89.44</td><td>87.32</td><td>89.20</td><td>81.60</td><td>84.67</td><td>76.93</td><td>82.67</td><td>85.17</td></tr><tr><td>PGC</td><td>96.04</td><td>63.96</td><td>83.88</td><td>83.05</td><td>84.67</td><td>82.33</td><td>81.10</td><td>76.87</td><td>85.90</td><td>82.17</td></tr><tr><td>Ours</td><td>98.28</td><td>76.24</td><td>90.04</td><td>90.84</td><td>94.97</td><td>84.10</td><td>91.67</td><td>84.77</td><td>92.07</td><td>89.51</td></tr></table>

Table 1: Accuracy comparison across 2D forgery categories and 3D reconstruction methods.

<table><tr><td>Method</td><td>Acc</td><td>R.Acc</td><td>F.Acc</td><td>Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>IID</td><td>89.64</td><td>88.44</td><td>90.85</td><td>89.64</td><td>95.52</td><td>96.68</td></tr><tr><td>NPR</td><td>80.88</td><td>87.29</td><td>74.46</td><td>80.80</td><td>89.24</td><td>90.53</td></tr><tr><td>FreqNet</td><td>80.34</td><td>76.81</td><td>83.86</td><td>80.31</td><td>89.31</td><td>91.58</td></tr><tr><td>ForAda</td><td>84.34</td><td>76.37</td><td>92.31</td><td>84.24</td><td>94.50</td><td>95.70</td></tr><tr><td>AIDE</td><td>86.63</td><td>78.14</td><td>95.11</td><td>86.53</td><td>95.07</td><td>94.51</td></tr><tr><td>Effort</td><td>91.87</td><td>87.99</td><td>95.74</td><td>91.86</td><td>98.29</td><td>98.53</td></tr><tr><td>PGC</td><td>90.37</td><td>90.22</td><td>90.53</td><td>90.37</td><td>96.41</td><td>97.06</td></tr><tr><td>Ours</td><td>96.54</td><td>95.30</td><td>97.78</td><td>96.54</td><td>99.50</td><td>99.58</td></tr></table>

Table 2: OOD evaluation on the GPT-Image-2 subset.

<table><tr><td>Method</td><td>Acc</td><td>R.Acc</td><td>F.Acc</td><td>Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>IID</td><td>67.64</td><td>81.85</td><td>53.42</td><td>66.97</td><td>70.89</td><td>75.47</td></tr><tr><td>NPR</td><td>73.37</td><td>83.00</td><td>63.74</td><td>73.12</td><td>79.69</td><td>83.06</td></tr><tr><td>FreqNet</td><td>69.91</td><td>71.50</td><td>68.33</td><td>69.91</td><td>75.52</td><td>78.18</td></tr><tr><td>ForAda</td><td>76.59</td><td>72.47</td><td>80.71</td><td>76.55</td><td>85.19</td><td>86.61</td></tr><tr><td>AIDE</td><td>59.76</td><td>56.77</td><td>62.76</td><td>59.73</td><td>62.22</td><td>60.06</td></tr><tr><td>Effort</td><td>79.43</td><td>84.04</td><td>74.83</td><td>79.39</td><td>87.62</td><td>89.52</td></tr><tr><td>PGC</td><td>83.46</td><td>92.09</td><td>74.82</td><td>83.33</td><td>93.35</td><td>93.86</td></tr><tr><td>Ours</td><td>86.49</td><td>95.88</td><td>77.09</td><td>86.37</td><td>95.61</td><td>95.94</td></tr></table>

Table 4: OOD evaluation on the HyperSwapper subset.

<table><tr><td>Method</td><td>Acc</td><td>R.Acc</td><td>F.Acc</td><td>Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>IID</td><td>78.60</td><td>81.15</td><td>76.05</td><td>78.59</td><td>86.65</td><td>88.53</td></tr><tr><td>NPR</td><td>81.68</td><td>84.84</td><td>78.52</td><td>81.66</td><td>89.54</td><td>91.06</td></tr><tr><td>FreqNet</td><td>70.65</td><td>79.04</td><td>62.25</td><td>70.44</td><td>77.61</td><td>79.13</td></tr><tr><td>ForAda</td><td>82.49</td><td>82.44</td><td>82.55</td><td>82.49</td><td>90.50</td><td>91.87</td></tr><tr><td>AIDE</td><td>73.01</td><td>72.56</td><td>73.47</td><td>73.01</td><td>81.21</td><td>83.23</td></tr><tr><td>Effort</td><td>85.17</td><td>87.32</td><td>83.01</td><td>85.16</td><td>92.98</td><td>94.16</td></tr><tr><td>PGC</td><td>82.17</td><td>83.05</td><td>81.29</td><td>82.17</td><td>89.98</td><td>91.57</td></tr><tr><td>Ours</td><td>89.51</td><td>90.84</td><td>88.19</td><td>89.51</td><td>95.36</td><td>96.03</td></tr></table>

Table 3: Comparison with state-of-the-art detection baselines.

<table><tr><td>Method</td><td>Acc</td><td>R.Acc</td><td>F.Acc</td><td>Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>IID</td><td>59.42</td><td>98.15</td><td>20.69</td><td>52.26</td><td>60.93</td><td>68.73</td></tr><tr><td>NPR</td><td>64.65</td><td>85.43</td><td>43.88</td><td>63.06</td><td>69.54</td><td>72.63</td></tr><tr><td>FreqNet</td><td>60.42</td><td>60.16</td><td>60.68</td><td>60.42</td><td>63.73</td><td>61.34</td></tr><tr><td>ForAda</td><td>55.46</td><td>81.47</td><td>29.45</td><td>52.22</td><td>57.15</td><td>58.63</td></tr><tr><td>AIDE</td><td>55.85</td><td>51.90</td><td>59.79</td><td>55.78</td><td>56.17</td><td>52.76</td></tr><tr><td>Effort</td><td>60.65</td><td>88.98</td><td>32.31</td><td>57.21</td><td>63.83</td><td>67.90</td></tr><tr><td>PGC</td><td>64.67</td><td>94.21</td><td>35.13</td><td>61.29</td><td>76.69</td><td>78.22</td></tr><tr><td>Ours</td><td>80.93</td><td>95.44</td><td>66.42</td><td>80.52</td><td>91.94</td><td>92.69</td></tr></table>

Table 5: OOD evaluation on the PersonaLive subset.

## 4 Experiment

## 4.1 Experiment Settings

Training Setup: All experiments are conducted on 8 NVIDIA RTX A6000 GPUs. The MAE decoder comprises 8 ViT blocks. In Stage I, the model is trained for 70 epochs. Specifically, we freeze the visual backbone and optimize only the MAE decoder during the first 30 epochs. For the remaining 40 epochs, we unfreeze the visual backbone and jointly optimize it with the MAE decoder. In this stage, the learning rates are set to $1 \times 1 0 ^ { - 4 }$ for the MAE decoder and $1 \times 1 0 ^ { - 5 }$ for the visual backbone, with a batch size of 4 per GPU. For contrastive learning, we set $G = 2 , K = 2 ,$ and the temperature to $\tau = 0 . 1$ . During joint optimization, we optimize $\mathcal { L } _ { \mathrm { M A E } } + 0 . 1 \mathcal { L } _ { \mathrm { c o n } }$ . Following the original MAE configuration, we use a masking ratio of $\rho = 0 . 7 5$ , while the number of image patches $P$ and the hidden feature dimension follow the default configuration of each visual backbone. In Stage II, we freeze the visual backbone and train only the MLP classifier for 10 epochs, using a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 64. Unless otherwise specified, all input images are resized to 224 × 224 during training. For reproducibility, we set the random seed to 42 for all experiments.

Dataset: We construct an identity-disjoint benchmark of 147,900 images from 361,469 successfully rendered and decodable candidates, including 117,900 training, 15,000 validation, and 15,000 test images. To prevent identity leakage, samples linked by any shared person ID are grouped into a connected component and assigned entirely to one split. Within each split, real and fake samples follow an exact 1:1 ratio; fake samples are equally divided among text-toimage generation, expression-driven manipulation, and face swapping; and the five 3D reconstruction methods each account for 20% of the samples. The same class and forgerycategory proportions are maintained within each reconstruction method. Because only complete identity components satisfying these constraints are retained, the final benchmark is smaller than the candidate pool. It contains 13,981 unique identities, with no person IDs or identical rendered images shared across splits.

Baseline: We select recent detection models published at top-tier conferences that introduce substantive methodological innovations and can be retrained on our dataset. This selection enables all baselines to be evaluated under a unified training protocol, ensuring a fair comparison. Specifically, the selected baselines include four generalpurpose real-versus-fake image detectors: AIDE (Yan et al. 2025a) (ICLR 2025), Efort (Yan et al. 2025b) (ICML 2025), PGC (Zhou et al. 2026) (ICML 2026), and FreqNet (Tan et al. 2024b) (AAAI 2024). We also evaluate three detectors developed for face forgery and deepfake detection: IID (Huang et al. 2023) (CVPR 2023), ForAda (Cui et al. 2025) (CVPR 2025), and NPR (Tan et al. 2024a) (CVPR 2024). For all baselines, we standardize the input resolution to 224 × 224 and follow the best-performing training configuration provided by each method’s oficial implementation for all other hyperparameters. For the main comparison, our model uses DINOv3 ViT-H+/16 (Siméoni et al. 2026) and concatenates the CLS tokens from blocks 10, 18, and 23 (zero-based) for Stage II classification.

![](images/929caba06775288f2d45906bbe76b65464d7a2167bb7bd13f242bf6949302515.jpg)

Figure 4: Qualitative masked-reconstruction results. With 75% of the input patches masked, the MAE recovers coherent facial structure and fine-grained local details in both real and fake 3D Gaussian-rendered heads.
<table><tr><td>Method</td><td>Acc</td><td>R.Acc</td><td>F.Acc</td><td>Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>IID</td><td>82.55</td><td>81.91</td><td>83.19</td><td>82.55</td><td>90.16</td><td>92.06</td></tr><tr><td>NPR</td><td>81.83</td><td>88.14</td><td>75.53</td><td>81.76</td><td>87.66</td><td>90.70</td></tr><tr><td>FreqNet</td><td>68.87</td><td>72.92</td><td>64.82</td><td>68.82</td><td>76.08</td><td>77.85</td></tr><tr><td>ForAda</td><td>83.60</td><td>89.85</td><td>77.34</td><td>83.53</td><td>90.78</td><td>92.84</td></tr><tr><td>AIDE</td><td>81.40</td><td>89.11</td><td>73.69</td><td>81.29</td><td>87.20</td><td>90.19</td></tr><tr><td>Effort</td><td>85.46</td><td>89.57</td><td>81.37</td><td>85.44</td><td>92.16</td><td>93.95</td></tr><tr><td>PGC</td><td>83.33</td><td>87.64</td><td>79.03</td><td>83.30</td><td>90.26</td><td>92.59</td></tr><tr><td>Ours</td><td>89.24</td><td>91.61</td><td>86.87</td><td>89.24</td><td>95.62</td><td>96.53</td></tr></table>

Table 6: OOD evaluation on the Any3DAvatar subset.

Evaluation Metrics: We report overall accuracy (ACC), class-specific accuracies (Real-ACC and Fake-ACC), the macro-averaged F1 score (Macro-F1), the area under the receiver operating characteristic curve (AUC), and average precision with fake as the positive class $( \mathrm { A P _ { f a k e } } )$ . All results are reported as percentages(%).

## 4.2 Comparison with Baselines

Tab. 3 compares our method with recent general-purpose and face-oriented forgery detectors under the unified benchmark protocol. Our method achieves the best result across all evaluation metrics and consistently outperforms the strongest baseline. The gains in both Real-ACC and Fake-ACC further show that the improvement is balanced across the two classes rather than driven by a bias toward either real or fake samples. Tab. 1 provides a finer-grained comparison across the 2D source categories and 3D reconstruction methods. Our method ranks first in every reported category and reconstruction pipeline, demonstrating consistent performance across the balanced IID benchmark. For OOD evaluation, we exclude the newest method in each forgery category from training and reserve it for testing: GPT-Image-2 for text-to-image generation, PersonaLive for expression-driven manipulation, , HyperSwapper for face swapping and Any3DAvatar for 3D methods. As shown in Tab. 2, Tab. 4, Tab. 5, and Tab. $^ { 6 , }$ our method achieves the best results across all reported metrics on all four held-out subsets, demonstrating stronger generalization to unseen forgery methods and an unseen 3D reconstruction method.

<table><tr><td>Strategy</td><td>Acc</td><td>R.Acc</td><td></td><td>F.Acc Macro-F1</td><td>AUC</td><td>AP(fake)</td></tr><tr><td>w/o Stage I</td><td>85.79</td><td>86.39</td><td>85.20</td><td>85.79</td><td>93.18</td><td>94.14</td></tr><tr><td>w/o MAE</td><td>86.96</td><td>88.40</td><td>85.52</td><td>86.96</td><td>93.81</td><td>94.69</td></tr><tr><td>w/o CL</td><td>87.18</td><td>88.56</td><td>85.80</td><td>87.18</td><td>94.16</td><td>94.85</td></tr><tr><td>Ours</td><td>89.51</td><td>90.84</td><td>88.19</td><td>89.51</td><td>95.36</td><td>96.03</td></tr></table>

Table 7: Ablation study of Stage I training. We report results without masked autoencoding (w/o MAE), without multi-view contrastive learning (w/o CL), and without Stage I training (w/o Stage I). All results are reported as percentages.

<table><tr><td rowspan="2">Strategy</td><td colspan="2">Intra-Sim ↑</td><td colspan="2">Inter-Sim ↓</td><td colspan="2">Sep-Gap ↑</td></tr><tr><td>Real</td><td>Fake</td><td>Real</td><td>Fake</td><td>Real</td><td>Fake</td></tr><tr><td>w/o CL</td><td>0.7427</td><td>0.7334</td><td>0.3499</td><td>0.3678</td><td>0.3928</td><td>0.3656</td></tr><tr><td>w/ CL</td><td>0.8965</td><td>0.8926</td><td>0.3293</td><td>0.3470</td><td>0.5672</td><td>0.5456</td></tr></table>

Table 8: Cross-view representation analysis. Intra-Sim measures similarity across views of the same 3D head, Inter-Sim across diferent heads of the same authenticity class, and Sep-Gap is their diference.

## 4.3 Ablative Analyses

Masked Autoencoding: To assess the efectiveness of masked autoencoding, we remove the MAE decoder and $\mathcal { L } _ { \mathrm { M A E } }$ from Stage I while keeping all other components and training settings unchanged. As shown in Tab. 7, this removal consistently degrades all evaluation metrics, demonstrating that masked reconstruction provides complementary supervision and contributes to downstream authenticity classification. Fig. 4 shows that, even with 75% masking, MAE produces visually plausible reconstructions and recovers some personalized facial details in both real and fake samples. This observation suggests that the encoder attends to fine-grained local appearance information retained after 3D Gaussian reconstruction, supporting its usefulness for downstream authenticity classification together with the quantitative ablation.

![](images/73c8360d26ff58f6e840e492eba449903240099a81858111863b3e11a945df0a.jpg)  
Figure 5: t-SNE of multi-view representations before (left) and after (right) contrastive learning, showing tighter withinhead clusters and better separation across identities.

![](images/d46ac5ca59ccc61344b34ae937ee661808c956f92383965816558ac5fb8840c6.jpg)  
Figure 6: CLS-token attention across network depths. CLS-token attention maps, averaged over all attention heads, for the selected low-, middle-, and high-level blocks. Diferent depths exhibit distinct spatial response patterns.

Multi-View Contrastive Learning: To analyze the learned representation, we compute cosine similarity between finallayer CLS tokens from approximately 13,000 images in 3,800 balanced multi-view groups, forming about 52,000 positive and negative pairs. Tab. 8 shows that contrastive learning increases Sep-Gap mainly by improving intra-head similarity, confirming stronger cross-view consistency. Removing the contrastive branch and objective consistently degrades performance (Tab. 7). Consistently, the t-SNE visualization of 20 balanced identity groups (10 views each) in Fig. 5 shows more compact same-head clusters and clearer inter-group separation.

Multi-Level CLS Token Fusion: To examine how the classification utility of CLS tokens varies with network depth, we freeze the Stage I visual backbone and train an identical Stage II MLP classifier using the CLS token from each Transformer block separately. We group the blocks into low-, middle-, and high-depth ranges and report the mean performance of the corresponding single-layer probes, while reporting the final block separately. As shown in Tab. 9, CLS tokens across different depths all contain information useful for authenticity classification, while multi-level fusion consistently outperforms the single-layer averages and the last-layer classifier. To inspect how the spatial responses of CLS tokens vary across depth, we average the CLS-token attention weights over all attention heads for each visualized block. As shown in Fig. 6, diferent layers exhibit distinct attention regions, providing qualitative evidence that their spatial responses are not identical. Based on layer-wise probing and attention diversity, we select blocks 10, 18, and 23 (zero-based) as representative low-, middle-, and high-level layers and concatenate their normalized CLS tokens for Stage II classification.

<table><tr><td>Strategy</td><td>Acc</td><td colspan="4">R.Acc F.Acc Macro-F1 AUC AP(fake)</td></tr><tr><td>w/ Low-Layer 68.57 69.26 67.87</td><td></td><td></td><td>68.53</td><td>75.04</td><td>75.72</td></tr><tr><td>w/Mid-Layer 83.16</td><td></td><td>84.8 81.50</td><td>83.15</td><td>90.76</td><td>92.09</td></tr><tr><td>w/ High-Layer85.6487.4683.82</td><td></td><td></td><td>85.64</td><td>92.68</td><td>93.80</td></tr><tr><td>w/ Last-Layer84.31 85.93</td><td></td><td>82.69</td><td>84.31</td><td>91.62</td><td>92.95</td></tr><tr><td>w/ Multi-Layer 89.51 90.84 88.19</td><td></td><td></td><td>89.51</td><td>95.36</td><td>96.03</td></tr></table>

Table 9: Stage II ablation of CLS token selection. We compare the average results of single-layer classifiers in the low-, middle-, and high-depth ranges, the last-layer classifier, and low–middle–high CLS token fusion.
<table><tr><td></td><td></td><td colspan="4">Visual Backbone Acc R.Acc F.Acc Macro-F1 AUC AP(fake)</td></tr><tr><td>DINOv2 ViT-L/1486.92 88.20 85.64</td><td></td><td></td><td></td><td>86.92</td><td>93.35 94.40</td></tr><tr><td>DINOv3 ViT-B/1687.19 90.07 84.31</td><td></td><td></td><td>87.18</td><td>94.38</td><td>95.14</td></tr><tr><td>DINOv3 ViT-L/1688.49 90.13 86.85</td><td></td><td></td><td>88.49</td><td>95.05</td><td>95.80</td></tr><tr><td>DINOv3 ViT-H+/16 89.51 90.84 88.19</td><td></td><td></td><td>89.51</td><td>95.36</td><td>96.03</td></tr></table>

Table 10: Comparison of visual backbone versions and model scales for source-face authenticity detection. DI-NOv3 ViT-H+/16 achieves the best performance across all metrics.

Visual Backbone: Tab. 10 examines the efects of backbone generation and model capacity. At a comparable scale, DI-NOv3 outperforms DINOv2, indicating the benefit of newer pretrained representations. Within DINOv3, performance consistently improves as model capacity increases, with ViT-H+/16 achieving the best overall results and therefore serving as our default backbone. Importantly, our framework still outperforms the strongest baseline when using either the older DINOv2 backbone or the smallest DINOv3 variant. This shows that stronger backbones further improve performance, while the gains of our method cannot be attributed solely to backbone scale or recency.

## 5 Conclusion

Summary: We introduce the first large-scale dataset and systematic benchmark for real/fake 3D Gaussian head detection, together with a two-stage detector that learns detailpreserving, cross-view-consistent representations and fuses multi-level CLS tokens for classification. Our method consistently outperforms existing detectors.

Limitations: Expression-driven forgeries remain challenging, with an accuracy of 76.24%. Future work will expand this category for stronger detection.

## References

An, S.; Xu, H.; Shi, Y.; Song, G.; Ogras, Ü. Y.; and Luo, L. 2023. PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°. In CVPR 2023, 20950–20959. IEEE.

Black Forest Labs. 2025. FLUX.2. https://blackforestlabs.ai/. Text-to-image generation model.

Blanz, V.; and Vetter, T. 2023. A Morphable Model For The Synthesis Of 3D Faces. New York, NY, USA: Association for Computing Machinery, 1 edition. ISBN 9798400708978.

Chen, B.; Zeng, J.; Yang, J.; and Yang, R. 2024. DRCT: Difusion Reconstruction Contrastive Training towards Universal Detection of Difusion Generated Images. In Salakhutdinov, R.; Kolter, Z.; Heller, K. A.; Weller, A.; Oliver, N.; Scarlett, J.; and Berkenkamp, F., eds., Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, 7621–7639. PMLR / OpenReview.net.

Chen, R.; Chen, X.; Ni, B.; and Ge, Y. 2020. SimSwap: An Eficient Framework For High Fidelity Face Swapping. In Chen, C. W.; Cucchiara, R.; Hua, X.; Qi, G.; Ricci, E.; Zhang, Z.; and Zimmermann, R., eds., ACM MM 2020, 2003–2011. ACM.

Chen, R.; Xi, J.; Yan, Z.; Zhang, K.; Wu, S.; Xie, J.; Chen, X.; Xu, L.; Guan, I.; Yao, T.; and Ding, S. 2025. Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable. In Belgrave, D.; Zhang, C.; Montoya, L. N.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; Chen, N.; Ruíz, I. V. M.; and Loaiza-Bonilla, A., eds., NeurIPS 2025.

Chu, X.; and Harada, T. 2024. Generalizable and Animatable Gaussian Head Avatar. In Globersons, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J. M.; and Zhang, C., eds., NeurIPS 2024.

Cui, X.; Li, Y.; Luo, A.; Zhou, J.; and Dong, J. 2025. Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection. In CVPR 2025, 19207–19217. Computer Vision Foundation / IEEE.

Di Nucci, D.; Catalini, R.; Borghi, G.; and Vezzani, R. 2026. Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering. In ICPR 2026.

FaceFusion Team. 2025. FaceFusion: Next Generation Face Swapping Framework. https://github.com/facefusion/ facefusion. Accessed: 2026.

Gafni, G.; Thies, J.; Zollhöfer, M.; and Nießner, M. 2021. Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction. In CVPR 2021, 8649–8658. Computer Vision Foundation / IEEE.

Gao, Y.; Xiao, Y.; Zhu, X.; Li, Y.; Zhang, Y.; Zhang, L.; and Zhang, J. 2026. Any3DAvatar: Fast and High-Quality Full-Head 3D Avatar Reconstruction from Single Portrait Image. arXiv:2604.13856.

Gerogiannis, D.; Papantoniou, F. P.; Potamias, R. A.; Lattas, A.; and Zafeiriou, S. 2025. Arc2Avatar: Generating Expressive 3D Avatars from a Single Image via ID Guidance. In CVPR 2025, 10770–10782. Computer Vision Foundation / IEEE.

Google. 2025. Nano Banana Pro Image Generation Model. https://ai.google.

Groshev, A.; Maltseva, A.; Chesakov, D.; Kuznetsov, A.; and Dimitrov, D. 2022. GHOST—A New Face Swap Approach for Image and Video Domains. IEEE Access, 10: 83452– 83462.

Guo, J.; Zhang, D.; Liu, X.; Zhong, Z.; Zhang, Y.; Wan, P.; and Zhang, D. 2024. LivePortrait: Eficient Portrait Animation with Stitching and Retargeting Control. arXiv:2407.03168.

Han, H.; Luo, Z.; Qi, J.; Rocha, A.; and Wan, R. 2026. GS-Checker: Tampering Localization for 3D Gaussian Splatting. In AAAI 2026.

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. B. 2022. Masked Autoencoders Are Scalable Vision Learners. In CVPR 2022, 15979–15988. IEEE.

He, Y.; Gu, X.; Ye, X.; Xu, C.; Zhao, Z.; Dong, Y.; Yuan, W.; Dong, Z.; and Bo, L. 2025. LAM: Large Avatar Model for One-shot Animatable Gaussian Head. In SIGGRAPH 2025, 1–13.

Huang, B.; Wang, Z.; Yang, J.; Ai, J.; Zou, Q.; Wang, Q.; and Ye, D. 2023. Implicit Identity Driven Deepfake Face Swapping Detection. In CVPR 2023, 4490–4499. IEEE.

Karras, T.; Laine, S.; and Aila, T. 2019. A Style-Based Generator Architecture for Generative Adversarial Networks. In CVPR 2019, 4401–4410. Computer Vision Foundation / IEEE.

Kerbl, B.; Kopanas, G.; Leimkühler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4): 139:1–139:14.

Kirschstein, T.; Qian, S.; Giebenhain, S.; Walter, T.; and Nießner, M. 2023. NeRSemble: Multi-View Radiance Field Reconstruction of Human Heads. ACM Trans. Graph., 42(4).

Li, H.; Chen, C.; Shi, T.; Qiu, Y.; An, S.; Chen, G.; and Han, X. 2024. SphereHead: Stable 3D Full-Head Synthesis with Spherical Tri-Plane Representation. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., ECCV 2024, Lecture Notes in Computer Science, 324–341. Springer.

Li, T.; Bolkart, T.; Black, M. J.; Li, H.; and Romero, J. 2017. Learning a model of facial shape and expression from 4D scans. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 36(6).

Li, Z.; Pun, C.; Fang, C.; Wang, J.; and Cun, X. 2026. PersonaLive! Expressive Portrait Image Animation for Live Streaming. In CVPR 2026, 18118–18128. Computer Vision Foundation / IEEE.

Liang, H.; Ge, Z.; Majee, S.; Tiwari, A.; Godaliyadda, G. M. D.; Veeraraghavan, A.; and Balakrishnan, G. 2025. FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses. arXiv:2508.18389.

Lyu, W.; Zhou, Y.; Yang, M.; and Shu, Z. 2025. FaceLift: Single Image to 3D Head with View Generation and GS-LRM. In ICCV 2025.

Martinez, J.; Kim, E.; Romero, J.; Bagautdinov, T. M.; Saito, S.; Yu, S.; Anderson, S.; Zollhöfer, M.; Wang, T.; Bai, S.; Li,

D.; Fan, A.; Paquet, U.; Tomczak, J. M.; and Zhang, C., eds., NeurIPS 2024.

OpenAI. 2025. GPT Image Generation Models. https:// openai.com.

Rössler, A.; Cozzolino, D.; Verdoliva, L.; Riess, C.; Thies, J.; and Nießner, M. 2019. FaceForensics++: Learning to Detect Manipulated Facial Images. In ICCV 2019, 1–11. IEEE.

Shiohara, K.; and Yamasaki, T. 2022. Detecting Deepfakes with Self-Blended Images. In CVPR 2022, 18720–18729.

Shiohara, K.; Yang, X.; and Taketomi, T. 2023. BlendFace: Re-designing Identity Encoders for Face-Swapping. In ICCV 2023, 7600–7610. IEEE.

Siméoni, O.; Vo, H. V.; Seitzer, M.; Baldassarre, F.; Oquab, M.; Jose, C.; Khalidov, V.; Szafraniec, M.; Yi, S. E.; Ramamonjisoa, M.; Massa, F.; Haziza, D.; Wehrstedt, L.; Wang, J.; Darcet, T.; Moutakanni, T.; Sentana, L.; Roberts, C.; Vedaldi, A.; Tolan, J.; Brandt, J.; Couprie, C.; Mairal, J.; Jégou, H.; Labatut, P.; and Bojanowski, P. 2026. DINOv3. Trans. Mach. Learn. Res., 2026.

Sun, K.; Liu, H.; Yao, T.; Sun, X.; Chen, S.; Ding, S.; and Ji, R. 2022. An Information Theoretic Approach for Attention-Driven Face Forgery Detection. In ECCV 2022, 111–127. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3- 031-19780-2.

Tan, C.; Liu, H.; Zhao, Y.; Wei, S.; Gu, G.; Liu, P.; and Wei, Y. 2024a. Rethinking the Up-Sampling Operations in CNN-Based Generative Network for Generalizable Deepfake Detection. In CVPR 2024, 28130–28139. IEEE.

Tan, C.; Zhao, Y.; Wei, S.; Gu, G.; Liu, P.; and Wei, Y. 2024b. Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning. In Wooldridge, M. J.; Dy, J. G.; and Natarajan, S., eds., AAAI 2024, 5052–5060. AAAI Press.

Taubner, F.; Zhang, R.; Tuli, M.; and Lindell, D. B. 2025. CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Difusion Models. In CVPR 2025, 5318– 5330. Computer Vision Foundation / IEEE.

Wang, C.; and Deng, W. 2021. Representative Forgery Mining for Fake Face Detection. In CVPR 2021, 14918–14927.

Wang, Y.; Chen, X.; Zhu, J.; Chu, W.; Tai, Y.; Wang, C.; Li, J.; Wu, Y.; Huang, F.; and Ji, R. 2021. HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping. In Zhou, Z., ed., IJCAI 2021, 1136–1142. ijcai.org.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.; Bai, S.; Xu, X.; Chen, Y.; Chen, Y.; Tang, Z.; Zhang, Z.; Wang, Z.; Yang, A.; Yu, B.; Cheng, C.; Liu, D.; Li, D.; Zhang, H.; Meng, H.; Wei, H.; Ni, J.; Chen, K.; Cao, K.; Peng, L.; Qu, L.; Wu, M.; Wang, P.; Yu, S.; Wen, T.; Feng, W.; Xu, X.; Wang, Y.; Zhang, Y.; Zhu, Y.; Wu, Y.; Cai, Y.; and Liu, Z. 2025. Qwen-Image Technical Report. arXiv:2508.02324.

Xie, L.; Wang, X.; Zhang, H.; Dong, C.; and Shan, Y. 2022. VFHQ: A High-Quality Dataset and Benchmark for Video Face Super-Resolution. In CVPRW 2022, 656–665. IEEE.

Xu, Y.; Liang, J.; Jia, G.; Yang, Z.; Zhang, Y.; and He, R. 2023. TALL: Thumbnail Layout for Deepfake Video Detection. In ICCV 2023, 22601–22611. IEEE.

Yan, S.; Li, O.; Cai, J.; Hao, Y.; Jiang, X.; Hu, Y.; and Xie, W. 2025a. A Sanity Check for AI-generated Image Detection. In ICLR 2025. OpenReview.net.

Yan, Z.; Luo, Y.; Lyu, S.; Liu, Q.; and Wu, B. 2024. Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8984–8994.

Yan, Z.; Wang, J.; Jin, P.; Zhang, K.; Liu, C.; Chen, S.; Yao, T.; Ding, S.; Wu, B.; and Yuan, L. 2025b. Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection. In Singh, A.; Fazel, M.; Hsu, D.; Lacoste-Julien, S.; Berkenkamp, F.; Maharaj, T.; Wagstaf, K.; and Zhu, J., eds., ICML 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Z-Image Team; Cai, H.; Cao, S.; Du, R.; Gao, P.; Hao, A.; Hoi, S. C. H.; Hou, Z.; Huang, S.; Jiang, D.; Jiang, Y.; Jin, X.; Li, L.; Li, Z.; Li, Z.-Y.; Liu, D.; Liu, D.; Wu, Q.; Yu, F.; Zhan, Z.; Zhang, C.; Zhang, S.; Zhou, R.; and Zhou, S. 2025. Z-Image: An Eficient Image Generation Foundation Model with Single-Stream Difusion Transformer. arXiv:2511.22699.

Zhang, B.; Cheng, Y.; Wang, C.; Zhang, T.; Yang, J.; Tang, Y.; Zhao, F.; Chen, D.; and Guo, B. 2024. RodinHD: High-Fidelity 3D Avatar Generation with Difusion Models. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., ECCV 2024, Lecture Notes in Computer Science, 465–483. Springer.

Zhao, H.; Wei, T.; Zhou, W.; Zhang, W.; Chen, D.; and Yu, N. 2021. Multi-attentional Deepfake Detection. In CVPR 2021, 2185–2194.

Zhao, X.; Xu, H.; Song, G.; Xie, Y.; Zhang, C.; Li, X.; Luo, L.; Suo, J.; and Liu, Y. 2025. X-NeMo: Expressive Neural Motion Reenactment via Disentangled Latent Attention. In ICLR 2025. OpenReview.net.

Zhou, X.; Fei, J.; Yu, P.; Xie, J.; Cheng, C.; and Xia, Z. 2026. PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection. In ICML 2026.

Zhu, H.; Wu, W.; Zhu, W.; Jiang, L.; Tang, S.; Zhang, L.; Liu, Z.; and Loy, C. C. 2022. CelebV-HQ: A Large-Scale Video Facial Attributes Dataset. In Avidan, S.; Brostow, G. J.; Cissé, M.; Farinella, G. M.; and Hassner, T., eds., ECCV 2022, volume 13667 of Lecture Notes in Computer Science, 650–667. Springer.