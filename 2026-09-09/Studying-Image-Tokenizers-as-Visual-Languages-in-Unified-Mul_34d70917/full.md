# Studying Image Tokenizers as Visual Languages in Unified Multimodal Models

Siting Li12\* Zhengyang Wang2 Simon Shaolei Du1 Xi Chen2 Yang Liu² 1University of Washington 2Amazon FAR

## Abstract

Image tokenizers define the “visual language" of unified multimodal models, yet are commonly studied through isolated metrics or generation-/understanding-only evaluations. These evaluations do not fully capture how visual tokens behave when modeled jointly with text. We build a controlled pure-autoregressive testbed and track task-specific validation losses during multimodal continual pretraining across text, image, text-to-image (T2I), and image-to-text (I2T) prediction. We examine how these losses scale and relate to downstream performance, then use them to study multimodal learnability—how well image and text tokens are jointly modeled—and tokenizer design. We find that (1) losses should be analyzed by task, since they exhibit distinct scaling behavior and rank tokenizers differently. (2) The loss-performance relationship depends on the predicted token space: for a fixed tokenizer, T2I and I2T losses correlate with generation quality, but across tokenizers, the T2I loss-performance relationship shifts with the image-token space, whereas I2T loss, computed over a shared text vocabulary, provides a more consistent signal. I2T loss also correlates with both generation and visual understanding performance after supervised finetuning. Using losses as a lens, we show that (3) better reconstruction does not necessarily yield lower task-specific losses or stronger downstream performance, and that (4) image tokenizer choice can affect text modeling under joint optimization. As case studies, we revisit three tokenizer design axes—the discriminator, semantic supervision, and vocabulary size—to examine their effects on joint modeling and downstream performance. Together, our testbed offers a complementary perspective on image tokenizers as visual languages, highlighting their interplay with text in joint multimodal training.

## 1 Introduction

Large-scale unified multimodal models have advanced rapidly in both visual generation and understanding by modeling images and text within a single framework. Among these, autoregressive (AR) modeling is a popular choice due to its compatibility with strong off-the-shelf language models [52, 33, 59, 9]. In models that autoregressively predict both text and image tokens, a key component is the discrete image tokenizer, which converts raw pixels into discrete symbols that can be modeled alongside text [61, 46, 35, 34]. In this sense, an image tokenizer is not merely a preprocessing module; it defines the “visual language" that the language model must learn, align with text, and use for both visual generation and understanding.

However, existing tokenizer studies primarily examine this component outside the unified modeling context, relying on isolated reconstruction-based metrics (e.g., rFID [18]) or ImageNet classification accuracy [70, 71], which do not necessarily translate into better downstream performance [64]. Others study tokenizers through generation-only or understanding-only downstream pipelines and report benchmark scores [55, 67]. These end-to-end evaluations are valuable, but they still leave open how tokenizers affect joint modeling behavior. In unified pure-AR multimodal models, image and text tokens are modeled jointly under a shared model and training objective. Joint training with text may affect image-token modeling, while the choice of image tokenizer may in turn affect cross-modal alignment and text modeling. Whether these effects can be inferred from downstream-agnostic metrics or single-axis pipelines alone remains unclear. Consequently, tokenizer design choices, such as compression ratio and auxiliary losses, may be optimized without fully accounting for their effects on joint image-text modeling.

![](images/5db829e8f7320b09f242c769c0194553cd2ab77931bd022cd8c00aa145582f9e.jpg)  
Figure 1: Studying image tokenizers as visual languages in unified multimodal models. Prior tokenizer studies focus on downstream-agnostic metrics (reconstruction/probing) or single-axis generation-/understanding-only pipelines, which miss joint text-image modeling behavior. We study tokenizers in the context of unified AR multimodal training, using task-specific losses as a signal to reveal how the tokenizer shapes downstream joint modeling.

Motivated by this gap, we study how the image tokenizer shapes multimodal modeling behavior in unified AR multimodal training. In Section 4, we develop a controlled, pure-AR continual pretraining recipe using publicly available images and Qwen3 as the language backbone [63]. Building upon this testbed, we use task-specific validation losses during continual pretraining to characterize tokenizer effects on text, image, text-to-image (T2I), and image-to-text (I2T) modeling. Applying this perspective requires addressing two questions. First, it is unclear whether task-specific losses exhibit similar scaling behavior during mixed image-text AR continual pretraining, or whether they must be interpreted separately. Second, it is unclear how these losses relate to downstream benchmark performance. Different image tokenizers define different prediction spaces, which may alter the loss– performance relationship. Moreover, visual benchmarks emphasize semantic faithfulness, perceptual quality, and cross-modal alignment, which are not directly measured by next-token prediction loss.

We therefore first examine how pretraining losses should be interpreted in unified multimodal training (Section 5). Losses on text, image, T2I, and I2T all decrease with increasing data and model size, but exhibit distinct scaling behavior and rank tokenizers differently across tasks, making an averaged loss too coarse for this analysis. We then relate losses to generation and understanding benchmarks. When the tokenizer is fixed, T2I loss correlates with generation quality. I2T loss also tracks generation quality, suggesting that it captures aspects of multimodal training progress beyond caption prediction. Across tokenizers, however, the T2I loss-performance relationship shifts with the visual token space; vocabulary normalization reduces part of this shift, and the remaining gap is related to reconstruction fidelity. In contrast, I2T loss is computed over shared text tokens and remains a more consistent crosstokenizer signal. I2T loss measured before supervised finetuning also correlates with post-finetuning generation and general VQA performance across tokenizers, supporting its use as a diagnostic signal.

Using task-specific losses as a lens, we make several observations about how the image tokenizer affects multimodal modeling behavior (Section 6). First, reconstruction fidelity can diverge from multimodal learnability—how well image and text tokens are modeled under the shared AR objective— indicating that reconstruction metrics alone are insufficient for evaluating tokenizers in unified multimodal models. Second, tokenizer choice can affect text modeling through the image-token prediction objective under joint training, a cross-modal effect not captured by evaluating generation or understanding alone. We then revisit three tokenizer design axes through ablations and find that (1) a DINO-based discriminator improves reconstruction fidelity but does not consistently improve multimodal learnability or downstream performance, with VQAv2 performance declining; (2) semantic supervision encourages object-level image-token-word associations, improving multimodal learnability despite worse reconstruction fidelity; and (3) the relationship between vocabulary size and multimodal learnability is non-monotonic, though a larger vocabulary can still benefit downstream performance, potentially through higher reconstruction fidelity. Together, these results offer a complementary perspective on image tokenizers as visual languages, highlighting their interplay with text in joint multimodal training.

## 2 Related Work

Unified multimodal understanding and generation. There have been recent advancements in building unified multimodal models, especially models for both visual understanding (captioning and VQA) and generation (text-to-image), in the hope of their mutual benefit. Various architectures have been proposed for unified models to jointly model text and image, including pure-autoregressive (AR) models (e.g., Chameleon [52], Emu3.5 [9], Liquid [59], Janus-Pro [8], and LongCat-Next [53]) serial AR + diffusion models (e.g., BLIP3-o [7] and MetaQuery [40]), and hybrid AR + diffusion models (e.g., Transfusion [74] and BAGEL [10]). Among these models, some employ a unified architecture and tokenizer for understanding and generation (e.g., Chameleon and Liquid), while others use decoupled modules, Mixture-of-Experts (MoEs), or different inference modes for the two tasks, and some adopt separate image tokenizers, illustrating the difficulty in truly unifying the two tasks in both tokenization and downstream models [72]. We adopt a pure-AR setup to model both modalities through next-token prediction, providing a controlled setting for studying image tokenizer effects under a shared modeling objective.

Loss-based analysis of unified multimodal models. Loss-based metrics, such as Perplexity (PPL) or Bits-Per-Byte (BPB), are commonly tracked against compute to validate model efficiency: Kaplan et al. [23] established held-out cross-entropy as a predictable scaling signal, Magnusson et al. [36] adopt BPB because token-level perplexity is not comparable across tokenizers, and Gadre et al. [14] fit language-model loss directly to average downstream error. Aghajanyan et al. [1] pioneered this analysis for mixed-modal models by establishing foundational scaling laws for mixed-modal loss, demonstrating that joint cross-entropy follows predictable power laws. Chameleon uses compute-loss curves to prove the stability of its early-fusion architecture at scale [52]. More recently, Liquid [59] utilizes these loss-vs-compute trends to demonstrate that inter-modality interference within a unified token space diminishes as model capacity increases. BAGEL [10] and Shukor et al. [47] identify emerging reasoning properties and the advantage of early-fusion and MoEs when scaling up compute, respectively. Tong et al. [56] uses IsoFLOP analysis to uncover the scaling asymmetry between vision and language in unified models. The above works on multimodal training use loss to characterize scaling, optimization, or performance under a fixed token space; we instead ask how losses should be interpreted across image tokenizers and use losses to study image tokenizers' downstream effects.

Studying image tokenizer properties. (1) Reconstruction. PSNR and SSIM [58] are pixel-wise metrics for image reconstruction quality, but are sensitive to noise and poorly align with human perception. Feature-based metrics, including rFID [18], IS [44], and LPIPS [68] on the ImageNet-1K validation set [11], consider semantic and distributional quality of reconstructed images. Recent tokenizer benchmarks such as VTBench [31] and TokBench [60] measure how tokenizers preserve text, identity, and details, which are important for downstream tasks like OCR-VQA [37]. However, better reconstruction does not necessarily translate into better downstream performance, and could conflict with better generation [64]. ETT [57] shares this motivation but tunes the tokenizer end-toend with the downstream model, whereas we keep tokenizers frozen to compare fixed visual token spaces. (2) Generation. Tokenizers can be compared by training with the same generative model and measuring the generation quality by gFID on a fixed set [61, 67, 70, 32], but generation-only evaluation does not establish how tokenizer choice affects both generation and understanding under joint training. (3) Semantics for understanding. Zero-shot accuracy and linear probing accuracy on ImageNet-1K have been used for both continuous and discrete tokenizers [42, 39, 71, 70], but higher classification accuracy does not necessarily translate into better downstream visual understanding, as observed in UniTok [35]. GigaTok [62] suggests AR probing accuracy as a better proxy to predict downstream performance when training with a larger AR model. TA-Tok [17] directly measures downstream performance at varying data scales. Cambrian-1 [55] pioneered leveraging multimodal large language models as an interface for tokenizer evaluation, but is restricted to continuous tokenizers. Apart from the above, (4) codebook usage is often reported in earlier work, but recent tokenizers can achieve near 100% usage with techniques like entropy loss even when they have a large vocabulary [76].

## 3 Preliminaries

In this section, we review discrete image tokenizers and the key design axes along which they differ: model and codebook architecture, bitwise compression ratio, and training objectives. Table 1 summarizes the tokenizers used in our study.

The VQGAN-based image tokenizers studied here [12, 66] follow an encoder-quantizer-decoder architecture. The encoder maps an input RGB image $\boldsymbol { x } \in \mathbb { R } ^ { \tilde { H } \times W \times 3 }$ to a sequence of $K$ continuous vectors $f \in \mathbb { R } ^ { K \times D }$ , where $\bar { \boldsymbol { K } }$ is the number of image tokens and D is the feature dimension. The quantizer maintains a codebook $Z \in \mathbb { R } ^ { B \times D }$ of vocabulary size B and replaces each encoder vector with its nearest codebook entry, yielding quantized vectors $\boldsymbol { z } \in \mathbb { R } ^ { K \times D }$ . The corresponding codebook indices form the discrete image-token sequence modeled by the autoregressive model. The decoder reconstructs an RGB image $\overset { \vartriangle } { \hat { x } } \in \mathbb { R } ^ { H \times W \times \mathbf { \bar { 3 } } }$ from z. A standard VQGAN is trained with a vector-quantization (VQ) loss and an autoencoding (AE) loss:

$$
\mathcal { L } _ { V Q } = | | \mathbf { s g } ( \boldsymbol { f } ) - z | | _ { 2 } ^ { 2 } + \beta | | \boldsymbol { f } - \mathbf { s g } ( z ) | | _ { 2 } ^ { 2 } , \mathcal { L } _ { A E } = \mathcal { L } _ { 2 } ( \boldsymbol { x } , \hat { \boldsymbol { x } } ) + \mathcal { L } _ { \mathcal { P } } ( \boldsymbol { x } , \hat { \boldsymbol { x } } ) + \mathcal { L } _ { \mathcal { G } } ( \boldsymbol { x } , \hat { \boldsymbol { x } } ) ,\tag{1}
$$

where sg(·) denotes the stop-gradient operation, $\beta$ weights the commitment loss, and $\mathcal { L } _ { 2 } , \mathcal { L } _ { \mathcal { P } } , \mathcal { L } _ { \mathcal { G } }$ are the $L _ { 2 }$ reconstruction loss, the LPIPS perceptual loss [68], and the adversarial (GAN) loss from the discriminator, respectively.

Architecture. On the encoder/decoder choices, prior work adopts various architectures, including ConvNet [12, 51], ViT [66], and ViT-based models with learnable latent queries [67]. On the codebook design, some tokenizers use multiple sub-codebooks [41, 49, 35] to represent each spatial position with multiple indices. Modeling these indices introduces additional choices in sequence organization and prediction architecture, complicating controlled comparisons. We therefore focus on single-codebook tokenizers throughout this work.

Bitwise compression ratio. The bitwise compression ratio is determined by the input image resolution $H \times W$ , the number of image tokens

Table 1: Tokenizers studied in our framework: IBQ [46], GigaTok [62], and UniTok [35]. B: vocabulary size; $\mathcal { L } _ { s e m } \colon$ semantic loss; Disc.: discriminator type; Usage: codebook usage; rFID: reconstruction FID on ImageNet-1K; †: models trained by us.
<table><tr><td>Tokenizer</td><td>Size</td><td>B</td><td> $\mathcal { L } _ { s e m }$ </td><td>Disc.</td><td>Usage rFID↓</td></tr><tr><td>IBQ-1024</td><td>0.1B</td><td>1024</td><td></td><td>PatchGAN</td><td>99% 2.24</td></tr><tr><td>IBQ-8192</td><td>0.1B 8192</td><td></td><td></td><td>PatchGAN 98%</td><td>1.87</td></tr><tr><td>IBQ-16384</td><td>0.1B 16384</td><td></td><td>PatchGAN</td><td>96%</td><td>1.37</td></tr><tr><td>GigaTok</td><td>0.6B 16384</td><td></td><td></td><td>PatchGAN 100%</td><td>0.81</td></tr><tr><td>GigaTok-DINO 0.6B 16384</td><td></td><td></td><td></td><td>DINO</td><td>100% 0.51</td></tr><tr><td>UniTok†</td><td>0.8B 16384</td><td></td><td></td><td>DINOv2-S 100%</td><td>1.86</td></tr><tr><td>UniTok-sem†</td><td>0.8B 16384</td><td></td><td></td><td>DINOv2-S 100%</td><td>2.23</td></tr></table>

$K ,$ and the vocabulary size B. In this work, we fix the number of image tokens to $K = 1 6 \times 1 6$ and the input resolution to $H \times W = 2 5 6 \times 2 5 6$ , and study the vocabulary size B as the main compression-related axis using the IBQ tokenizer family [46] (Section 6.4). We defer to future work the study of varying $H \times W ~ ( { \bar { \mathbf { e } } } . \mathbf { g } .$ , any-resolution image tokenizers [34]) and varying K (e.g., highly compressed 1D tokenizers [2, 24]), as changing image resolution alters the visual detail available to the model, while changing K alters sequence length and the image-token budget during training, introducing additional factors into the comparison.

Training objectives. Beyond the standard losses in Eq. (1), we study two design choices in the training objective. On the supervision signal, recent tokenizers add semantic supervision, such as a CLIP contrastive loss [35], to encourage the tokens to capture high-level semantics; we study its effect on downstream joint modeling by training single-codebook UniTok variants without and with this loss, denoted UniTok and UniTok-sem, respectively [35], in Section 6.3. On the adversarial loss $\mathcal { L } _ { G }$ , some tokenizers replace the standard PatchGAN discriminator [22] with a DINO-based discriminator [4] to obtain lower rFID [54, 28]; we revisit whether this reconstruction gain carries over to downstream joint modeling using GigaTok with the two discriminators (denoted GigaTok and GigaTok-DINO) [62] in Section 6.1.

## 4 Framework Construction

In this section, we describe the framework for exploring the image tokenizer's effect on downstream unified multimodal training. In Section 4.1, we introduce the training recipe, which extends a pretrained text language model into a unified autoregressive multimodal model through continual pretraining and supervised finetuning. In Section 4.2, we define the validation loss we use to measure modeling quality across tasks. Further training details and tokenizer information are provided in Appendix A.

## 4.1 Training Recipe

Overall setup. We extend pretrained Qwen3 language models [63] into unified autoregressive multimodal models by expanding their vocabularies to include discrete image tokens. Our main experiments use three dense model sizes: Qwen3-0.6B, 1.7B, and 4B. Given an image tokenizer with vocabulary size B, we add B new learnable embeddings to the base model's vocabulary, initialized from a multivariate normal distribution matching the original embeddings’mean and covariance. We also extend the LM head to predict the newly added image tokens. We add special tokens (boi) and (eoi〉 to mark the start and end of image-token sequences. During training, we remove the text conditioning from 10% of T2I samples and use (unconditional) to mark unconditional image generation, enabling classifier-free guidance (CFG) at evaluation [19]. The resulting model accepts and produces both text tokens and flattened image-token IDs. We continually pretrain it on mixedmodal data and then apply supervised finetuning (SFT) on multimodal instruction-following data, as detailed below.

Continual pretraining stage. In this stage, we train the unified model on large-scale image-text and pure-text data.

(1) Data mixture and preprocessing. Our largest data scale is 60M samples, comprising 6.6M pure-text samples from DataComp-LM [27] and 53.3M image-text samples from three sources: (a) LAION-Aesthetics [45], filtered by aesthetic score ≥ 5.5 and recaptioned by InternVL3-1B [75]; (b) JourneyDB [50], recaptioned by GPT-3.5; and (c) BLIP3o-Pretrain-Short-Caption [7]. We resize all images to 256 × 256 and pre-tokenize them before training.

(2) Loss and token sequence formatting. For pure-text samples, we adopt the standard cross-entropy loss for next-token prediction in training. For image-text samples, we initially assign 80% to text-toimage (T2I) prediction and 20% to image-to-text (I2T) prediction. As described above, 10% of the T2I samples are converted to unconditional image generation. The conditional sequence formats are:

T2I: {text } {prompt } (boi〉 {image tokens } 〈eoi〉〈eos〉,

I2T: (boi〉 {image tokens } (eoi〉 {prompt} {text} (eos).

Following Liquid [59], we compute the cross-entropy loss only on the bold tokens (conditional next-token prediction), leaving the prompt and the conditioning modality unscored. The prompts are listed in Appendix A.2.

(3) Hyperparameters. We use a Warmup-Stable-Decay (WSD) schedule with a 0.03 warmup ratio and linear decay over the last 20% of steps. For the 0.6B model, we sweep the learning rate (lr) over {3e-5, 1e-4} and the batch size (bs) over {512, 1024, 2048}, forming a 2 × 3 grid. Because different tasks favor different hyperparameter settings (Appendix C.2), we adopt a balanced setting of (lr = 3e-5, bs = 512) for the main experiments and reuse it for the 1.7B and 4B models. We also experimented with larger learning rates (3e-4 for 0.6B and 1e-4 for 4B), but found that they caused unstable training with loss spikes.

Supervised finetuning (SFT) stage. We finetune the models for 2 epochs on 4.9M instructionfollowing samples. The mixture combines 1M LMSYS-Chat pure-text instructions [73], 2.9M multimodal instruction and captioning samples from Mini-Gemini [30], and 1M text-to-image samples (LAION-Aesthetics sampled from the continual-pretraining distribution, together with JourneyDB and BLIP3o-60K data [7]). We use a cosine schedule with a 0.03 warmup ratio, a peak learning rate of 5e-5, and batch size 1024.

Further details about data of both stages, including the subset proportions, can be found in Appendix A.1. All images used in the training are publicly accessible.

Training recipe validation. We verify our training recipe by training a Qwen3-8B model with the Chameleon tokenizer (Table 2). Compared with Liquid-7B, our model reaches similar performance on GenAI-Bench, WISE, and VQA (averaged over VQAv2, GQA, TextVQA, and POPE), while trailing on MJHQ-30K. Full benchmark scores and image generation examples are provided in Appendix A.3. We view this experiment

Table 2: Recipe validation. Our 8B model obtains similar GenAI-Bench [25], WISE [38], and VQA (Mean) scores to Liquid-7B [59], while trailing on MJHQ-30K [26]. “Data" refers to the number of samples seen in continual pretraining.

<table><tr><td>Model</td><td>Data GenAI↑ MJHQ-30K↓ WISE↑ VQA (Mean)↑</td><td></td><td></td></tr><tr><td>Liquid-7B 90M</td><td>0.72</td><td>5.47</td><td>0.41 61.40</td></tr><tr><td>Ours (8B) 60M</td><td>0.73</td><td>10.55 0.38</td><td>60.83</td></tr></table>

FLOPs (log10 scale)  
![](images/b3ebeeb346c665b3705e02fe6bbde002ea1792930364b92b2e9f37fe237b8f3a.jpg)  
Figure 2: Validation losses scale with data size differently across tasks for the 0.6B model. The y-axis is in log scale. The lines are linear fits to the dots per tokenizer.

as a validation that the recipe serves as a reliable testbed for studying tokenizers, rather than as evidence of achieving superior performance.

## 4.2 Validation Loss

We use validation loss, defined as the mean negative log-likelihood over supervised tokens, to measure how well a model fits held-out data, following common practice in language-model scaling studies [23, 20, 1]. Let N be the evaluation set. For a sequence $t \in N$ , let $\mathbf { \bar { \mathbf { } } } ( \bar { t } ) \subseteq \{ 1 , \dots , | t | \}$ denote its supervised positions: the tokens on which the loss is computed (the bold spans of the T2I/I2T formats in Section 4.1; all positions for pure-text samples). The model's log-likelihood on the supervised tokens is

$$
l = \sum _ { t \in N } \sum _ { i \in S ( t ) } \ln p ( t _ { i } \mid t _ { < i } ) ,\tag{2}
$$

where conditioning tokens (e.g., the text prompt or the conditioning image) are never scored but still enter through the context $t _ { < i } .$ Using $\begin{array} { r } { T ( \dot { N } ) = \sum _ { t \in N } | S ( t ) | } \end{array}$ for the number of supervised tokens, we report the mean per-token loss

$$
\mathcal { L } = - \frac { l } { T ( N ) } .\tag{3}
$$

Note on loss interpretation. Unlike the bits-per-byte loss common in language modeling, which upper-bounds a description length of the raw data [15, 36], our validation loss on image tokens is computed over the tokenizer's discrete latent codes. Because image tokenization is lossy, the model's likelihood on these codes cannot be converted into a description length for the original pixels. We therefore interpret L as a measure of latent-modeling quality for a fixed visual language. Comparisons across image tokenizers require accounting for differences in their token spaces, as examined in Section 5.2.

Evaluation protocol. We hold out 50,000 pure-text and 50,000 image-text samples, drawn from the training distribution, as the evaluation set. This yields validation losses for four tasks: text, unconditional image, text-to-image (T2I), and image-to-text (I2T). We apply the same loss-masking rules as in training: conditioning tokens provide context but do not contribute directly to the loss. Empirically, losses over different image sources behave similarly, so we report LAION-Aesthetics as our main proxy, denoted LAION-(Image, T2I, I2T); losses on the other sources are given in Appendix C.1. In loss-scaling plots, the x-axis FLOPs denote the compute used in continual mixedmodal pretraining.

## 5 Interpreting Loss in Unified Multimodal Training

Before using pretraining loss to study tokenizers, we first examine how it should be interpreted in unified multimodal training. We begin by asking whether a single unified loss metric can characterize tokenizer behavior in downstream modeling, or whether the task-specific losses must be analyzed separately (Section 5.1). We then examine whether these losses align with the performance on image generation benchmarks and how they can be calibrated across tokenizers (Section 5.2.1). Finally, we test how they relate to post-SFT performance (Section 5.2.2).

## 5.1 Task-Specific Loss Scaling

Our setting differs from standard language-only pretraining in three aspects: we continually train from a text-only pretrained model; the training is mixed-modal, spanning four distinct tasks; and the T2I and I2T losses are conditional. These differences make the loss harder to interpret: it is not obvious whether the four task-specific losses share a common scaling behavior or must be analyzed separately. We therefore first examine how each task's loss scales with data and model size, and then ask whether any single signal derived from the loss yields a tokenizer ranking that is consistent across tasks.

![](images/02a9935d56bf6b90981dbb1be0e12316b8a788b2733cecb91356812376913e4a.jpg)  
Figure 3: Validation losses scale with model size differently across tasks on 12M data. The y-axis is in log scale. The lines are linear fits to the dots per tokenizer.

All task-specific losses scale with data and model size. We plot validation loss against pretraining FLOPs for three tokenizers, varying data scale (Figure 2) and model size (Figure 3). Both figures show that the losses on all four tasks decrease smoothly and roughly follow a power law (a linear trend in the log-log plots) under both data and model scaling. For the data-scaling analysis, we use intermediate, pre-annealing checkpoints from a single training run for each tokenizer-model-size configuration, with lr = 3e-5 and bs = 512. Annealed checkpoints follow the same power-law trends, with annealing mainly reducing the text loss and leaving the image-side losses largely unchanged (Appendix C.3) Results across hyperparameters are deferred to Appendix C.2, and further continual-pretraining dynamics to Appendix B.

The scaling patterns differ across tasks. Although all four losses scale, they do so in qualitatively different ways, which makes an aggregated loss too coarse to serve as an analysis signal. (1) Text loss is largely governed by language-model initialization, while image-related losses are shaped more by multimodal continual pretraining. Under data scaling at a fixed model size (Figure 2), text loss changes only slightly, whereas image, T2I, and I2T losses decrease more substantially. This suggests that text-modeling quality is largely inherited from the pretrained language backbone, while image-related tasks benefit more from multimodal continual pretraining. Under model scaling (Figure 3), text loss also decreases, consistent with initializing larger models from stronger pretrained language backbones; this trend therefore cannot be attributed to multimodal continual pretraining alone [69]. (2) Image-token modeling drives T2I, while I2T reflects cross-modal alignment. The T2I loss closely tracks the unconditional image loss, indicating that image-token prediction accounts for much of the difficulty in T2I. The I2T loss, though also computed over text tokens, behaves differently from the text loss, indicating that conditioning on image tokens introduces a distinct cross-modal signal.

Tokenizer rankings differ across tasks under both data and model scaling. No single ordering of the three tokenizers holds across tasks. In Figure 2, UniTok has the highest text loss yet the lowest T2I loss, while GigaTok has the lowest text-related losses but the highest T2I loss. Consequently, no single task loss yields a tokenizer ranking that holds across tasks, and a simple average can obscure these task-dependent differences. These task-dependent rankings suggest trade-offs between text modeling and image-token prediction under the unified AR objective. Stronger semantic alignment, in turn, can help I2T even when a tokenizer is not optimal for pure-text modeling, as seen in the comparison between UniTok and IBQ-16384 in Figure 2.

Finding 1. Unified multimodal training loss should be analyzed per task: task-specific losses exhibit distinct scaling behavior that an averaged loss obscures, and no single tokenizer ranking holds across tasks.

## 5.2 Loss-Performance Relation

We next study how pretraining losses relate to benchmark performance. These validation losses measure next-token modeling quality, whereas downstream benchmarks emphasize semantic faithfulness, perceptual quality, and cross-modal alignment. We therefore ask the following questions:

• (Section 5.2.1) In pretraining, do the losses align with image-generation benchmark scores, and are these relationships consistent across tokenizers?

• (Section 5.2.2) Do the losses serve as signals of post-SFT benchmark performance?

![](images/b7e60d1d78e623fc532980c5f36c15b7567d1397f687eb603c4b5749c07a1acf.jpg)

![](images/bff73b80199eb16fc2ebd825c600b86fda3a3d70125262eca18f31e52e5aabfa.jpg)

![](images/36ce974bcd1d7eb553fdc4dcfc27cc71816e7838c04a3f938bd0a457c416ae0b.jpg)

![](images/5d4ce012edbe6f44633b3a84f11f902b9513a03c359405de998ec56595c3d911.jpg)

Figure 4: Both T2I loss and I2T loss align with text-to-image generation quality when tokenizer type is controlled. The lines are linear fits to all dots.  
![](images/32ada119e15d2344a9e62900df341bd654aee1184a7c186cff0ee55be07bd147.jpg)

![](images/f2dd29b24b2717665bc68622f916607c9a09e1842bb2a882515d2f2c3df7e72e.jpg)

![](images/f65d0456b42a177d4343b492d6233aaebf9c0e67cc67c0773243e41a7726ed33.jpg)

![](images/b72d596e0c5e46c5e3590a3bdb8ac2faef1441aa584cecf1c87dcc9c7e212666.jpg)  
Figure 5: T2I loss does not align with text-to-image generation quality across tokenizers (the first two plots), while I2T loss aligns with generation quality (the last two plots). The lines are linear fits to the dots per tokenizer and per hyperparameter.

Benchmarks. We evaluate the continually pretrained checkpoints under varying hyperparameters on GenAI-Bench [25] and MJHQ-30K [26], scored by VQAScore (↑) and $\mathrm { l o g _ { 1 0 } ^ { \cdot } \bar { g F I D } ( \downarrow ) }$ , respectively. We then apply our SFT recipe to the pretraining checkpoints after annealing and re-evaluate generation performance, together with general visual understanding on VQAv2 [16] and GQA [21]. Qualitative examples on GenAI-Bench and further benchmark details are in Appendix D.

## 5.2.1 Alignment between Losses and T2I Quality in Pretraining

We study the alignment between pretraining losses and T2I generation quality, which can be evaluated directly on the pretrained checkpoints without SFT.

Within a tokenizer, both T2I and I2T loss align with T2I quality. Since T2I generation predicts image tokens conditioned on text, one might expect the T2I validation loss to be the most relevant pretraining signal for generation quality. This holds when the tokenizer is fixed: as shown in Figure 4, T2I loss aligns strongly with GenAI-Bench VQAScore and MJHQ-30K gFID across data scales and hyperparameter settings. More surprisingly, I2T loss also correlates with generation quality, albeit more weakly, suggesting that it captures aspects of multimodal training progress beyond caption prediction.

Across tokenizers, the T2I loss-performance relation shifts with the image-token space. We next ask whether both alignments hold across tokenizers. I2T loss is computed over a shared text vocabulary, providing a common basis for comparison. Empirically, Figure 5 shows a more consistent relationship between I2T loss and generation quality across tokenizers. T2I loss, however, exhibits tokenizer-dependent shifts in its relationship with generation quality. Thus, T2I loss remains a useful modeling diagnostic, but the same loss need not correspond to the same image quality across tokenizers, since each defines a different image-token space.

Vocabulary normalization reduces the cross-tokenizer shift in the T2I loss-performance relation. We hypothesize that vocabulary size contributes to the numerical scale of T2I loss. A natural reference is log2 B, the entropy of a uniform distribution over B image tokens. We therefore define the vocabulary-normalized image loss as ${ \mathcal { L } } ^ { * } = { \mathcal { L } } / \log _ { 2 } B _ { \mathrm { { t } } }$ where B denotes the image vocabulary size. Evaluated on IBQ tokenizers with $B \in \{ 1 0 2 4 , \stackrel { \sim } { 8 } 1 9 2 , 1 6 3 8 4 \} , \mathcal { L } ^ { * }$ yields a more consistent T2I loss-performance relation across the IBQ variants (Figure 6). Further details are in Appendix C.5.

The residual cross-tokenizer gap is linked to reconstruction fidelity. Among tokenizers with the same vocabulary size (16384), Figure 7 shows that, at a fixed benchmark score, T2I loss follows the reverse order of the rFID ranking: a tokenizer with worse reconstruction fidelity reaches the same generation quality at a lower T2I loss. This association suggests that reconstruction fidelity may help explain the residual differences in the T2I loss-performance relationship at a fixed vocabulary size. These results motivate considering reconstruction fidelity alongside T2I loss when comparing generation performance across tokenizers.

![](images/2813c2aa2a8c5f5dc5fbf3f9bd5efac1576f05c2a468643ff31c07d7526a674a.jpg)  
Figure 6: After vocabulary normalization, a more consistent T2I loss-performance relation exists across IBQ tokenizers. The lines are linear fits to the dots per hyperparameter. Quantitative results are in Table 10.

![](images/1b0113691cc6ab3b7c00096c7176ce484c22b32e4cd54afade152b00ed9b57dc.jpg)  
Figure 7: The T2I loss-performance relation shift is related to differences in reconstruction fidelity. The rFID is negatively correlated with T2I loss when the GenAI benchmark score is controlled.

![](images/06456fa649e1afc97c4c77224cbf547319c867e57525b6e66e3479e05e26b0c0.jpg)

![](images/b9f431995d173030e97b77e08ff2faf5ba7bec89301f920df128a0083a065ec8.jpg)

![](images/d862c02fea65cdcc5a66d9fc81ed1acd8c76d527abbde1c344fbc562e6ef604d.jpg)

![](images/9235533276e075f966f96c6bce8338ba53d719903bd548bcc066a861bf945a18.jpg)  
Figure 8: Both I2T and T2I loss show strong correlation with post-SFT generation performance when tokenizer type is controlled. We use the checkpoints after annealing for loss evaluation.

Finding 2. Token space shapes the loss-performance relationship. I2T loss, computed over a shared text vocabulary, provides a more consistent cross-tokenizer signal of generation performance, whereas T2I loss exhibits tokenizer-dependent shifts. Vocabulary normalization reduces these shifts within the IBQ family, while residual differences at a fixed vocabulary size are associated with reconstruction fidelity.

## 5.2.2 Correlation between Losses and Post-SFT Performance

We ask whether pretraining losses remain a useful signal of benchmark performance after supervised finetuning, which introduces additional variation beyond pretraining.

Losses correlate with generation quality after SFT. The pretraining alignment of Section 5.2.1 carries over to post-SFT performance. When the tokenizer is fixed (GigaTok), both the T2I and I2T validation losses measured before SFT remain strongly correlated with post-SFT benchmark performance (Figure 8). Across tokenizers, the I2T loss again remains the more consistent signal, staying correlated with post-SFT generation performance (Figure 9).

I2T loss is informative about general visual understanding. Across tokenizers, hyperparameters, and training scales, the I2T loss shows a moderate correlation with post-SFT VQAv2 and GQA performance (Figure 9), indicating that it captures part of the capability that transfers to general VQA and can thus serve as a meaningful signal. The moderate correlation may partly reflect differences in both task and training stage: the loss measures caption prediction during pretraining, while the benchmarks measure question answering after SFT. Caption fit is therefore informative about VQA performance, but does not determine it. This imperfect correlation is unlikely to be explained solely by data-shuffling noise: in our reproducibility check of one GigaTok configuration, VQAv2 and GQA scores vary by less than 1% relative across two seeds (Table 11). The relationship is less consistent for specialized benchmarks such as TextVQA, which additionally require capabilities such as OCR (Appendix C.4).

Finding 3. Pretraining losses are informative about post-SFT performance. I2T loss correlates with both generation and general visual understanding across tokenizers, while the relationship between T2I loss and generation performance is tokenizer-dependent.

## 6 What Unified Training Reveals About Image Tokenizers

Building on our loss-based analysis of unified multimodal training (Section 5), we now use validation loss as a lens to reveal tokenizer properties that isolated metrics or single-axis evaluations miss:

![](images/0418035552a894c105cadba27806f6cd9945c1e30eec7080abd0c4e6e55da08c.jpg)  
Figure 9: I2T loss shows moderate correlation with both generation and understanding performance after SFT across tokenizers. We use the checkpoints after annealing for loss evaluation.

Table 3: Tokenizer comparison at 0.6B model scale. (rFID: reconstruction FID on ImageNet-1K; Text/I2T/T2I: last step loss; GenAI and VQAv2 are post-SFT; lr=1e-4 for GigaTok family, lr=3e-5 for UniTok family, batch size=512; †: tokenizers trained by us.)
<table><tr><td rowspan="2">Tokenizer</td><td>Fidelity</td><td colspan="3">Multimodal Learnability</td><td colspan="2">Performance</td></tr><tr><td>rFID↓</td><td>Text↓</td><td>I2T↓</td><td>T2I↓</td><td>GenAI↑</td><td>VQAv2↑</td></tr><tr><td>GigaTok-DINO</td><td>0.51</td><td>2.949</td><td>1.660</td><td>7.437</td><td>0.720</td><td>51.31</td></tr><tr><td>GigaTok</td><td>0.81</td><td>2.937</td><td>1.661</td><td>7.416</td><td>0.720</td><td>52.25</td></tr><tr><td>UniTok†</td><td>1.86</td><td>2.883</td><td>1.670</td><td>6.264</td><td>0.670</td><td>57.21</td></tr><tr><td>UniTok-sem†</td><td>2.23</td><td>2.873</td><td>1.655</td><td>5.848</td><td>0.690</td><td>61.28</td></tr></table>

reconstruction fidelity can diverge from multimodal learnability, and the image token space can even affect text modeling (Section 6.2). We study these effects through three tokenizer design axes as case studies: the discriminator (Section 6.1), semantic supervision (Section 6.3), and vocabulary size (Section 6.4).

## 6.1 Does better reconstruction imply better unified multimodal learning?

The premise behind reconstruction-based tokenizer selection is that higher-fidelity reconstruction yields better downstream results. Recent analyses question this premise, showing that tokenizer design trades off reconstruction fidelity, compression, and latent-space learnability, though largely in the visual-generation-only setting [3, 64, 43, 62]. We revisit this fidelity-learnability-performance relationship in unified AR multimodal training, where we measure fidelity by rFID and take multimodal learnability to be how well the resulting image and text tokens are modeled under the shared AR objective, as reflected by the task-specific validation losses of Section 5.

Better reconstruction does not always lead to better unified multimodal learning. We compare several tokenizer pairs on reconstruction fidelity, multimodal learnability, and downstream performance in Table 3. Within each tokenizer family, better reconstruction does not consistently correspond to better downstream generation or understanding performance. Replacing the standard PatchGAN discriminator with a DINO-based discriminator improves GigaTok's rFID from 0.81 to 0.51, yet this reconstruction gain does not translate into a better joint model: none of the validation losses improve significantly, GenAI is unchanged (0.720), and VQAv2 drops from 52.25 to 51.31. This complements the image-generation-only study of GigaTok [62], where the DINO-based discriminator is adopted to improve reconstruction. In contrast, semantic supervision in UniTok-sem worsens reconstruction fidelity yet improves all three reported validation losses and both reported downstream metrics (Section 6.3). Together, these results show that a reconstruction metric such as rFID alone is insufficient for tokenizer selection: we must also consider how the tokenizer affects multimodal learnability.

Finding 4. Reconstruction fidelity and multimodal learnability can diverge under unified AR training: lower rFID does not promise stronger downstream generation or understanding.

Finding 5. Improving rFID with a DINO-based discriminator does not improve multimodal learnability or downstream performance.

## 6.2 Does the image token space affect text modeling?

In unified AR training, text and image tokens are optimized under the same model and objective. This raises a question not addressed by evaluating generation or understanding alone: can the image tokenizer affect text modeling even when the text-only data, text tokenizer, and language backbone are all fixed?

Image token space affects text modeling difficulty under joint training. We observe this effect in two tokenizer comparisons (lr=3e-5, batch size=512) in Figure 10 (Joint). First, UniTok and UniTok-sem share the same text tokenizer and training recipe, yet UniTok-sem achieves lower text loss under joint text-image training. Second, among IBQ variants, IBQ-1024 obtains lower text loss than IBQ-8192 despite using the same text token space. These differences suggest that changes in the image token space can affect the difficulty of text modeling. In these comparisons, adding semantic supervision to UniTok or reducing the IBQ vocabulary from 8192 to 1024 lowers text loss under joint training.

![](images/1ed0b5364758535dbe52703f35da4ef9132cf4d186ce6ac3f7ca1abeacebdf8b.jpg)

![](images/20923fd880bcdd097e70d0bd79f52206f63d25d8dbec175ca7fa533aa2ecda9b.jpg)  
Figure 10: Changing the image token space can reduce interference between image-token and text-token modeling in joint training. In the joint setting, UniTok-sem and IBQ-1024 attain lower text loss than UniTok and IBQ-8192, respectively. When the image-generation objective is ablated (Text+I2T training), the text losses within each pair become similar (left), indicating that the interference stems from image-token modeling; the I2T gap, however, persists (right).

## The text-side effect comes from image token prediction rather than captioning (I2T). We

hypothesize that this text-side effect arises from interference between image token modeling and text token modeling. To test this, we remove the image token prediction objective by reformatting the T2I and image-generation samples into I2T order and train on only Text+I2T data with the same hyperparameter settings. Under this ablation, the text-loss gap largely disappears for both the UniTok pair and the IBQ pair (Figure 10, Text+I2T), indicating that the text-side difference is explained by the image token prediction objective in full unified training. Interestingly, the I2T gap in both pairs persists, suggesting that tokenizer choice affects how informative image tokens are for captioning through a pathway independent of image token prediction. In Appendix C.6, we further verify that the text-side effect is not caused by the I2T objective through a similar ablation on I2T data.

Finding 6. Image token space can affect text modeling difficulty under joint AR training. Our ablations attribute this effect to the image token prediction objective rather than to captioning (I2T).

## 6.3 Semantic supervision: How does semantic alignment change multimodal learnability?

Semantic supervision is commonly added in image tokenizer training to encourage visual features to encode higher-level semantics. In Section 6.1, we observed that it improves the multimodal learnability of the resulting visual language, which in turn benefits downstream performance. We now ask where this improvement comes from: does semantic supervision make local image-token sequences easier to predict (“better local visual grammar"), or does it make individual image tokens more informative for captioning (“better visual words")?

![](images/b1682607f3319962d536698c29ba2b99550b68ac8dda83f74e38bd95b8f5eedd.jpg)

![](images/a040f3b657bfea24046ade9d95291cd597bfa8ae131ba03da8e9ec0112bc911b.jpg)  
Figure 11: Semantic supervision strengthens objectlevel image-token-word alignment (“better visual words"). UniTok-sem yields larger I2T loss improvements on COCO object words than on stopwords, colors, or general content tokens (left), and higher PMI between image tokens and object words (right).

![](images/47a652ba73ae678ae947512c84292ae5af2c286b64af99020241222778ef1df9.jpg)  
Figure 12: Semantic supervision does not consistently reduce empirical image-token n-gram entropy. The empirical entropy of the tokens is calculated on tokenized images of the LAION-Aesthetics validation set.

The I2T gain is concentrated on visually grounded object words. In the Text+I2T setting of Section 6.2, where image token prediction is ablated, we track each tokenizer's I2T loss improvement over training and take the gap $( \bar { \Delta } _ { \mathrm { U n i T o k - s e m } } - { \Delta } _ { \mathrm { U n i T o k } } )$ , for which larger values mean greater relative improvement of UniTok-sem. This gain concentrates on visually grounded object words (Figure 11, left): COCO object-category tokens improve far more than stopwords, general content tokens (all tokens apart from stopwords), or color tokens (CSS color names), indicating that semantic supervision makes image tokens more informative for predicting object-level words.

Semantic supervision strengthens object-level image-token-word association. As further evidence, the pointwise mutual information (PMI) between image tokens and words, averaged within three word groups on the validation set, is higher for UniTok-sem on COCO object words (Figure 11, right), suggesting that semantic supervision associates image tokens more tightly with object-level words.

Semantic supervision does not consistently reduce empirical image-token n-gram entropy. An alternative explanation is that semantic supervision makes image-token sequences easier to model. Following Chan et al. [5], we compute the empirical 1-, 2-, 3-, and 7-gram entropy of the image tokens produced by UniTok and UniTok-sem on the validation set (Figure 12). While UniTok-sem tokens have slightly lower entropy at 1- and 2-grams, their 3- and 7-gram entropy is higher than for UniTok. These measurements do not show a consistent reduction in local n-gram entropy, providing no clear evidence for a simpler local visual grammar under this diagnostic.

Finding 7. Semantic supervision improves multimodal learnability and strengthens object-level imagetoken-word associations, without consistently reducing empirical n-gram entropy. These observations support a “better visual words" interpretation rather than a simpler local visual grammar.

## 6.4 Vocabulary size: Is a larger vocabulary better for unified multimodal learning?

Vocabulary size is a key compression axis of discrete image tokenizers. Prior work shows that scaling up the vocabulary improves reconstruction fidelity by letting each image token carry more visual information [46], but it may also change the modeling difficulty of image tokens. Since reconstruction fidelity and multimodal learnability can diverge, we ask whether a larger vocabulary brings better unified multimodal learning. Within the IBQ family, we vary $B \in \{ 1 0 2 4 $ , 8192, 16384} and compare image-side losses using the vocabulary-normalized image-token loss $\mathcal { L } ^ { \ast }$ (Section 5.2.1).

![](images/2fca16cc04bdc145d6f1e726b069eb945a112daec2419e05b375508297e97ddd.jpg)

![](images/b471f702575b0bba79cae0d65c99e2bb084dc392a2b3c4b887a4fc944518595d.jpg)

![](images/837458ad134d4c9ae7d920e3d1d29d0994c51a9340c8eb2e00d3014641a4dd8f.jpg)

![](images/1d91c6d61d09515a2beae94a78c589e4ba731c733ce3f19f44b4b50f4c81d53e.jpg)  
Vocabulary Size  
Figure 13: Vocabulary size changes task-specific validation losses with no simple monotonic trend. Among the three IBQ variants, IBQ-16384 achieves the best I2T loss, but IBQ-8192 attains the lowest T2I and image losses after normalization (lr=1e-4, batch size=512).

Vocabulary size affects task-specific losses non-monotonically. As shown in Figure 13, the relationship between vocabulary size B and loss is not simply larger-is-better. IBQ-16384 attains the best I2T loss, yet the intermediate size IBQ-8192 achieves the lowest normalized T2I and image losses among the three variants. Text loss, meanwhile, follows the reverse (also non-monotonic) trend relative to the image-side losses, again indicating that different tasks can prefer different visual token spaces.

A larger vocabulary can benefit downstream performance, possibly through higher reconstruction fidelity. Comparing the IBQ variants on generation and understanding benchmarks after SFT (Figure 14), IBQ-16384 achieves the best performance among the three, despite IBQ-8192 having lower T2I and image losses. As with the rFID-performance relation in Section 5.2.1, this is plausibly due to IBQ-16384's higher reconstruction fidelity. Beyond $B = 1 6 3 8 4$ , however,

![](images/3e82f054ebb1424821004c9dac90e2bd3f37d3a850870b0ae749e53551b90cbd.jpg)  
Figure 14: The largest vocabulary brings the best downstream performance among the three IBQ variants (lr=1e-4, batch size=512).

it remains unclear whether reconstruction fidelity or multimodal learnability is the dominant factor, which we leave to future work.

Finding 8. Vocabulary size affects multimodal learnability non-monotonically: an intermediate vocabulary gives the best T2I and image losses, yet a larger vocabulary can still benefit downstream performance, potentially through higher reconstruction fidelity.

## 7 Conclusion

We study image tokenizers in the context of unified AR multimodal models through a controlled. loss-based testbed. We first examine how validation loss relates to downstream performance and how the task-specific losses should be interpreted. Building on this, we show that reconstruction fidelity can diverge from multimodal learnability and that the image token space can affect text modeling under joint training. We also revisit discriminator choice, semantic supervision, and vocabulary size to examine their effects on unified multimodal learning. Our findings suggest that unified tokenizer design should account for both reconstruction fidelity and joint modeling difficulty across tasks, rather than optimizing any single proxy metric. These observations highlight the importance of studying image tokenizers as visual languages in interaction with text during joint multimodal training.

## Acknowledgments and Disclosure of Funding

SSD acknowledges the support of NSF IIS 2143493, NSF IIS 2229881, Sloan Fellowship, and the AI2050 program at Schmidt Sciences. This material is based on the Chameleon tokenizer supported by the Chameleon Research License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## References

[1] A. Aghajanyan, L. Yu, A. Conneau, W.-N. Hsu, K. Hambardzumyan, S. Zhang, S. Roller, N. Goyal, O. Levy, and L. Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pages 265–279. PMLR, 2023.

[2] R. Bachmann, J. Allardice, D. Mizrahi, E. Fini, O. F. Kar, E. Amirloo, A. El-Nouby, A. Zamir, and A. Dehghan. FlexTok: Resampling images into 1d token sequences of flexible length. In Forty-second International Conference on Machine Learning, 2025.

[3] Black Forest Labs. FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison, 2025. URL https://bfl.ai/research/representation-comparison.

[4] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[5] D. M. Chan, R. Corona, J. Park, C. J. Cho, Y. Bai, and T. Darrell. Analyzing the language of visual tokens. arXiv preprint arXiv:2411.05001, 2024.

[6] J. Chen, Q. Yu, X. Shen, A. Yuille, and L.-C. Chen. ViTamin: Designing scalable vision models in the vision-language era. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12954–12966, 2024.

[7] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. BLIP3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.

[8] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan. Janus-Pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,2025.

[9] Y. Cui, H. Chen, H. Deng, X. Huang, X. Li, J. Liu, Y. Liu, Z. Luo, J. Wang, W. Wang, et al. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

[10] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[11] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[12] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

[13] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, et al. DataComp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36:27092–27112, 2023.

[14] S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, et al. Language models scale reliably with over-training and on downstream tasks. arXiv preprint arXiv:2403.08540, 2024.

[15] L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

[16] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.

[17] J. Han, H. Chen, Y. Zhao, H. Wang, Q. Zhao, Z. Yang, H. He, X. Yue, and L. Jiang. Vision as a dialect: Unifying visual understanding and generation via text-aligned representations. arXiv preprint arXiv:2506.18898, 2025.

[18] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[19] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[20] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[21] D. A. Hudson and C. D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

[22] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.

[23] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[24] D. Kim, J. He, Q. Yu, C. Yang, X. Shen, S. Kwak, and L.-C. Chen. Democratizing text-to-image masked generative models with compact text-aware one-dimensional tokens. arXiv preprint arXiv:2501.07730, 2025.

[25] B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, T. Ling, X. Xia, P. Zhang, G. Neubig, et al. GenAI-Bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024.

[26] D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024.

[27] J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Y. Gadre, H. Bansal, E. Guha, S. S. Keh, K. Arora, et al. DataComp-LM: In search of the next generation of training sets for language models. Advances in Neural Information Processing Systems, 37:14200–14282, 2024.

[28] X. Li, K. Qiu, H. Chen, J. Kuen, J. Gu, B. Raj, and Z. Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.

[29] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

[30] Y. Li, Y. Zhang, C. Wang, Z. Zhong, Y. Chen, R. Chu, S. Liu, and J. Jia. Mini-Gemini: Mining the potential of multi-modality vision language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[31] H. Lin, T. Geng, Z. Xu, and W. Zhao. VTBench: Evaluating visual tokenizers for autoregressive image generation. arXiv preprint arXiv:2505.13439, 2025.

[32] H. Lin, T. Wang, Y. Ge, Y. Ge, Z. Lu, Y. Wei, Q. Zhang, Z. Sun, and Y. Shan. TokLIP: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422, 2025.

[33] H. Liu, W. Yan, M. Zaharia, and P. Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268, 2024.

[34] J. Lu, L. Song, M. Xu, B. Ahn, Y. Wang, C. Chen, A. Dehghan, and Y. Yang. AToken: A unified tokenizer for vision. arXiv preprint arXiv:2509.14476, 2025.

[35] C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi. UniTok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.

[36] I. Magnusson, A. Bhagia, V. Hofmann, L. Soldaini, A. H. Jha, O. Tafjord, D. Schwenk, E. Walsh, Y. Elazar, K. Lo, et al. Paloma: A benchmark for evaluating language model fit. Advances in Neural Information Processing Systems, 37:64338–64376, 2024.

[37] A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty. OCR-VQA: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE, 2019.

[38] Y. Niu, M. Ning, M. Zheng, W. Jin, B. Lin, P. Jin, J. Liao, C. Feng, K. Ning, B. Zhu, et al. WISE: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

[39] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[40] X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

[41] L. Qu, H. Zhang, Y. Liu, X. Wang, Y. Jiang, Y. Gao, H. Ye, D. K. Du, Z. Yuan, and X. Wu. TokenFlow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.

[42] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[43] V. Ramanujan, K. Tirumala, A. Aghajanyan, L. Zettlemoyer, and A. Farhadi. When worse is better: Navigating the compression-generation tradeoff in visual tokenization. arXiv preprint arXiv:2412.16326, 2024.

[44] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

[45] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes A. Katta, C. Mullis, M. Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35: 25278–25294, 2022.

[46] F. Shi, Z. Luo, Y. Ge, Y. Yang, Y. Shan, and L. Wang. Scalable image tokenization with index backpropagation quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16037–16046, 2025.

[47] M. Shukor, E. Fini, V. G. T. da Costa, M. Cord, J. Susskind, and A. El-Nouby. Scaling laws for native multimodal models. arXiv preprint arXiv:2504.07951, 2025.

[48] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

[49] W. Song, Y. Wang, Z. Song, Y. Li, H. Sun, W. Chen, Z. Zhou, J. Xu, J. Wang, and K. Yu. DualToken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025.

[50] K. Sun, J. Pan, Y. Ge, H. Li, H. Duan, X. Wu, R. Zhang, A. Zhou, Z. Qin, Y. Wang, et al. JourneyDB: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678, 2023.

[51] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

[52] C. Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

[53] M. L. Team, B. Xiao, C. Wang, C. Li, C. Zhang, C. Peng, H. Yu, H. Yang, H. Yan, H. Sun, et al. Longcat-next: Lexicalizing modalities as discrete tokens. arXiv preprint arXiv:2603.27538, 2026.

[54] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

[55] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan, A. Wang, R. Fergus, Y. LeCun, and S. Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024.

[56] S. Tong, D. Fan, J. Nguyen, E. Brown, G. Zhou, S. Qian, B. Zheng, T. Vallaeys, J. Han, R. Fergus, et al. Beyond language modeling: An exploration of multimodal pretraining. arXiv preprint arXiv:2603.03276, 2026.

[57] W. Wang, F. Zhang, Y. Cui, H. Diao, Z. Luo, H. Lu, J. Liu, and X. Wang. End-to-end vision tokenizer tuning. arXiv preprint arXiv:2505.10562, 2025.

[58] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[59] J. Wu, Y. Jiang, C. Ma, Y. Liu, H. Zhao, Z. Yuan, S. Bai, and X. Bai. Liquid: Language models are scalable and unified multi-modal generators. International Journal of Computer Vision, 2025.

[60] J. Wu, D. Luo, W. Zhao, Z. Xie, Y. Wang, J. Li, X. Xie, Y. Liu, and X. Bai. TokBench: Evaluating your visual tokenizer before visual generation. arXiv preprint arXiv:2505.18142, 2025.

[61] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. VILA-U: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

[62] T. Xiong, J. H. Liew, Z. Huang, J. Feng, and X. Liu. GigaTok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. arXiv preprint arXiv:2504.08736, 2025.

[63] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[64] J. Yao, B. Yang, and X. Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.

[65] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024.

[66] J. Yu, X. Li, J. Y. Koh, H. Zhang, R. Pang, J. Qin, A. Ku, Y. Xu, J. Baldridge, and Y. Wu. Vector-quantized image modeling with improved VQGAN. arXiv preprint arXiv:2110.04627, 2021.

[67] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37: 128940–128966, 2024.

[68] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[69] Y. Zhang, B. McKinzie, Z. Gan, V. Shankar, and A. T. Toshev. Pre-trained language models do not help auto-regressive text-to-image generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1281–1287, 2024.

[70] Y. Zhao, F. Xue, S. Reed, L. Fan, Y. Zhu, J. Kautz, Z. Yu, P. Krähenbühl, and D.-A. Huang. QLIP: Text-aligned visual tokenization unifies auto-regressive multimodal understanding and generation. arXiv preprint arXiv:2502.05178, 2025.

[71] A. Zheng, X. Wen, X. Zhang, C. Ma, T. Wang, G. Yu, X. Zhang, and X. Qi. Vision foundation models as effective visual tokenizers for autoregressive image generation. arXiv preprint arXiv:2507.08441, 2025.

[72] D. Zheng, M. Zhang, H. Li, K. Zou, H. Liu, Z. Guo, K. Feng, Y. Liu, Y. Luo, Y. Feng, et al. Architecture decoupling is not all you need for unified multimodal model. arXiv preprint arXiv:2511.22663, 2025.

[73] L. Zheng, W.-L. Chiang, Y. Sheng, T. Li, S. Zhuang, Z. Wu, Y. Zhuang, Z. Li, Z. Lin, E. P. Xing, et al. LMSYS-Chat-1M: A large-scale real-world LLM conversation dataset. arXiv preprint arXiv:2309.11998, 2023.

[74] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

[75] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[76] L. Zhu, F. Wei, Y. Lu, and D. Chen. Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%. Advances in Neural Information Processing Systems, 37:12612–12635, 2024.

Table 4: Proportion of subsets in continual pretraining data. \*We upsample JourneyDB from 4.2M to 6.5M.
<table><tr><td>Data Source</td><td># Data (M)</td><td>Filtering</td><td>Recaptioning</td></tr><tr><td>DataComp-LM</td><td>6.6</td><td>Random Sampling</td><td></td></tr><tr><td>LAION-Aesthetics</td><td>42</td><td>Aesthetic Score (5.5)</td><td>InternVL3-1B [75]</td></tr><tr><td>JourneyDB</td><td>6.5*</td><td></td><td>GPT3.5</td></tr><tr><td>BLIP3o-Pretrain-Short-Caption</td><td>4.8</td><td></td><td></td></tr></table>

Table 5: Proportion of subsets in supervised finetuning data. The LAION and JourneyDB textto-image data is randomly sampled from the same distribution of corresponding data in continual pretraining. \*We keep all 0.2M DVQA data without the filtering in Mini-Gemini.

<table><tr><td>Data Source</td><td># Data (M)</td></tr><tr><td>Mini-Gemini (VQA)</td><td>1.7*</td></tr><tr><td>Mini-Gemini (Caption)</td><td>1.2</td></tr><tr><td>LMSYS-Chat LAION-Aesthetics</td><td>1.0</td></tr><tr><td></td><td>0.86</td></tr><tr><td>JourneyDB</td><td>0.08</td></tr><tr><td>BLIP3o-60K</td><td>0.06</td></tr></table>

## A Training Details

## A.1 Data Proportion

We detail the data mixture in Table 4 and 5. The SFT data is similar to Liquid [59], but we substitute the text-to-image part with public data from pretraining and use more DVQA data.

We adopt 1:8 as the text:image-text ratio in continual pretraining. This choice is influenced by 1:2 in Liquid, while they use the Chameleon tokenizer with a sequence length of 1024. We observe that it is important to keep the total number of image tokens seen during training at the same level to achieve comparable performance, so we use 1:8 as the final ratio.

## A.2 Prompts

For visual generation, we uniformly sample the prompt from the same set of prompts as Liquid during pretraining: “Generate an image based on this description.", “Create an image that captures the provided description.", “Based on the previous text, produce a corresponding image.", “Please illustrate the above text with a picture.", “Translate the given description into a image.", “Construct a visual representation of the above description.", “Create a image that matches the text.", “Formulate a visual expression that reflects the narrative just provided.", “Give a visual depiction based on the above sentences.", “Create an image using the information mentioned above as guidance.". During loss calculation, we only use “Generate an image based on this description." Note that for unconditional visual generation, we use the format 〈unconditional〉 (boi〉 {image tokens}(eoi〉(eos〉 without any prompt.

For captioning, we use “The caption of this image is:" as the prompt.

## A.3 Model Performance

We list the model performance on visual generation and understanding benchmarks in Table 6 and provide qualitative image generation results on GenAI-Bench prompts in Figure 15. Our 8B model trained on public images with less pretraining data achieves similar performance to Liquid-7B on most benchmarks, which verifies that the framework serves as a reusable and reliable testbed for studying tokenizers' impact on downstream unified training.

## A.4 Tokenizer Information

IBQ (Index Backpropagation Quantization) [46] modifies standard vector quantization in VQGAN to enable scalable training of large codebooks. Instead of updating only the selected code entries, IBQ backpropagates gradients to all codebook embeddings jointly with the visual encoder, which helps maintain high codebook utilization and a more consistent latent space between encoded features and code vectors. A variant of IBQ is adopted in Emu3.5 [9] with semantic loss and 131072 as the vocabulary size. The authors release IBQ variants with vocabulary sizes of 1024, 8192, 16384, and 262144 without semantic loss. We adopt the first three since 262144 is much larger than the original vocabulary size in the base model, which might cause training difficulty.

× Failure “Three cookies on a plate."  
Table 6: Comparison of generation and understanding performance with Liquid-7B [59] after SFT. “Data" refers to the number of samples seen in continual pretraining.
<table><tr><td>Model</td><td>Data GenAI↑ MJHQ-30K↓ WISE↑ VQAv2↑ GQA↑ TextVQA↑ POPE↑ MME↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Liquid-7B 90M</td><td></td><td>0.72</td><td>5.47</td><td>0.41</td><td>68</td><td>56.1</td><td>40.4</td><td>81.1</td><td>1107.2</td></tr><tr><td>Ours (8B) 60M</td><td></td><td>0.73</td><td>10.55</td><td>0.38</td><td>67.3</td><td>54.44</td><td>43.46</td><td>78.1</td><td>1038.12</td></tr></table>

![](images/0968e3ef49e3eabb3428c8dd76da40154bc02752d4f08ee69bf5233b9dfc869e.jpg)  
√Success  
“A lone lighthouse standing guard on a rocky coastline."

![](images/3a9a87af9f5833f054ec1d4208514da58b2e67551889f89cff163ef7ba231ad2.jpg)  
√Success  
"A red rose in full bloom next to a pink rosebud in a garden."

![](images/2ecb55a21962552060a355799fc162805de495fbf7e3cf8e03ca91bbb343aef9.jpg)  
√Success “Two apples on a kitchen counter."

![](images/99bfa31ee0a65f56288eee6f76506cc83f348bbe1435ca19a96fec98f1a533b3.jpg)  
√Success  
“A painting where the mountain is depicted as taller than the trees in the foreground."

![](images/be9a9cee9b052b65e83a5230c1caf576637f3dc2b6a60ed2d36061d12d4818c2.jpg)  
× Failure

![](images/92cf6e418e5fb7838dfc8fb1954edb1aa0cbcdc79abc25294f115be157ca789e.jpg)  
“A knight with a feather plume helmet by a stone tower."  
Figure 15: Qualitative text-to-image generation of our 8B model on GenAI-Bench. All images are generated by the Qwen3-8B model with the Chameleon tokenizer after SFT, i.e., the same checkpoint reported in Table 6, using the prompt shown below each panel. Green marks cases with successful instruction following and red marks failure cases.

GigaTok [62] is proposed for autoregressive image generation and studies the effect of tokenizer scale and architectural design on discrete visual representations. It is based on VQGAN with large encoder-decoder networks and incorporates DINO-based semantic loss. It examines design choices such as asymmetric encoder-decoder scaling, different discriminator architectures, and alternative tokenization structures, and reports their impact on reconstruction behavior and downstream image generation. We revisit their conclusion on discriminator architectures using GigaTok-B-L under our framework.

UniTok [35] is a unified discrete image tokenizer designed to support both image generation and visual understanding. It is also based on a VQGAN and introduces multi-codebook quantization, which partitions latent features into multiple subspaces and discretizes each with independent codebooks to increase representational capacity while maintaining stable training. The tokenizer is trained with both reconstruction and semantic loss (contrastive loss with CLIP text encoder) so that the resulting tokens capture fine-grained visual details as well as high-level semantic information. We train a singlecodebook version of UniTok with vocabulary size 16384 and latent dimension 64, and ablate the usage of semantic loss, examining its effect on downstream scaling under our framework. Following the original paper, we initialize the model with ViTamin-L/16 [6]. Due to compute constraints, we train the model on DataComp-medium [13] for five epochs and use 4096 as the global batch size. We adopt a smaller learning rate (lr=2e-4 for tokenizer and 5e-6 for discriminator) with entropy loss weight 0.05 and semantic loss weight 0 or 0.05. The rFID, zero-shot accuracy, and linear probing accuracy on ImageNet-1K of the two UniToks are shown in Table 7: UniTok-sem is stronger than UniTok in image classification but is weaker in image reconstruction.

Table 7: Comparison of UniTok with and without semantic loss on rFID, top-1 zero-shot accuracy, and top-1 linear probing accuracy on ImageNet-1K. The linear probing training and evaluation follow the same practice as in the GigaTok paper.
<table><tr><td>Model</td><td colspan="3">rFID↓ ZS Accuracy↑ LP Accuracy↑</td></tr><tr><td>UniTok</td><td>1.86</td><td>5.34</td><td>15.36</td></tr><tr><td>UniTok-sem 2.23</td><td></td><td>46.79</td><td>59.42</td></tr></table>

Table 8: Most tokenizers have a linear or concave curve for text but a convex curve for image-related tasks. We list the quadratic coefficient a in quadratic fitting $( y = a x ^ { 2 } + b x + c )$ of the 0.6B model runs in Figure 2. Positive a indicates the convexity of the curves, while negative aindicates concavity.
<table><tr><td rowspan=1 colspan=3>Tokenizer        Text    Caption</td><td rowspan=1 colspan=1>T2I   Image</td></tr><tr><td rowspan=1 colspan=2>GigaTok       5.46e-04</td><td rowspan=1 colspan=1>1.96e-02</td><td rowspan=1 colspan=1>6.92e-035.15e-03</td></tr><tr><td rowspan=1 colspan=1>GigaTok-DINO</td><td rowspan=1 colspan=1>1.21e-04</td><td rowspan=1 colspan=1>-7.73e-03</td><td rowspan=1 colspan=1>3.66e-03 2.54e-03</td></tr><tr><td rowspan=1 colspan=1>IBQ-1024</td><td rowspan=1 colspan=1>-1.63e-03</td><td rowspan=1 colspan=1>1.11e-02</td><td rowspan=1 colspan=1>2.16e-02 1.57e-02</td></tr><tr><td rowspan=1 colspan=1>IBQ-8192</td><td rowspan=1 colspan=1>-2.24e-03</td><td rowspan=1 colspan=1>6.94e-03</td><td rowspan=1 colspan=1>1.21e-02 1.08e-02</td></tr><tr><td rowspan=1 colspan=1>IBQ-16384</td><td rowspan=1 colspan=1>-2.17e-03</td><td rowspan=1 colspan=1>8.35e-03</td><td rowspan=3 colspan=1>1.15e-02 9.01e-031.26e-02 1.15e-021.13e-021.22e-02</td></tr><tr><td rowspan=1 colspan=1>UniTok</td><td rowspan=1 colspan=1>-1.59e-03</td><td rowspan=1 colspan=1>1.01e-02</td></tr><tr><td rowspan=1 colspan=1>UniTok-sem</td><td rowspan=1 colspan=1>-2.01e-03</td><td rowspan=1 colspan=1>2.19e-02</td></tr></table>

Following the official code of UniTok, we adopt a zooming-then-cropping strategy for image preprocessing for UniTok and UniTok-sem, which is different from other tokenizers. This strategy is also applied in the pre-tokenization process in our training. Since we do not compare tokenizers from different families, this does not affect our conclusions.

## A.5 Compute Resources

The experiments were conducted on an internal cluster using 1, 2, 4, or 8 NVIDIA H200 GPUs with 141GB memory. The estimated compute for continual pretraining is listed in the result plots. The full research project also involved preliminary experiments for adjusting the training recipe and hyperparameters.

## B Supplementary Results on Training Dynamics

We try quadratic fitting for the loss-data relationship in Figure 2 for all tokenizers and record the quadratic coefficients in Table 8: Most data scaling curves show convexity for LAION-I2T, LAION-T2I, and LAION-Image loss, but a slight concavity for text loss. Considering that the data scaling dots are checkpoints from the same runs, this implies different training dynamics across tasks. In Figure 16, we present the loss curves of a 4B model training including earlier checkpoints, and find the curve shapes aligned with the finding on quadratic coefficients. We hypothesize that the image-related modeling might dominate optimization in the early phase but slowly saturate later, giving rise to the text loss decaying rate.

We look into the gradient norm during a single run and check how gradient norm per modality changes over time in continual pretraining with GigaTok and IBQ-16384. In Figure 17, we find that the T2I-related gradient is larger than the text gradient at the beginning, but gradually decays throughout the training. The text gradient decays at a slower rate and reaches a similar level to the T2I gradient at the end. This supports the hypothesis about the convexity and concavity we observe in data scaling curves.

![](images/cc83fab32935d497dd48226f0c959706c34d3d9f0981a243dffb89262f14c149.jpg)  
Figure 16: Quadratic fitting of UniTok-sem 4B training, including data points from earlier stage of continual training.

![](images/3d9af01d4cd25b321a5731011c2bd5e11eef77d1bacd0aa412d53e9e4357e7f7.jpg)  
Figure 17: Gradient norms over all parameters by modality across training steps of 0.6B model with GigaTok and IBQ-16384 (lr=3e-5, bs=512). For both runs, the T2I gradient slowly saturates at the end.

## C Supplementary Loss Results

## C.1 Loss over Different Image Sources

We observe very similar T2I loss and image loss trends across three data sources in Figure 18. The I2T loss trends are also similar between JourneyDB and BLIP3o-Short-Caption, but are slightly different from LAION-I2T, where (lr=3e-5, bs=512) surpasses other settings. We hypothesize that this is due to their different caption styles: recaptioned LAION-Aesthetics has longer and more detailed captions than the other two. Therefore, it would rely more on the models' language ability.

## C.2 Loss Scaling Results for Different Hyperparameters

To study the effect of hyperparameters on model fitting, we experiment with a 2×3 grid of (lr, bs) using GigaTok. In Figure 19, we observe that no hyperparameter setting dominates all tasks: Aggressive hyperparameter settings (high lr, small bs, represented by (1e-4,512)) are favored on LAION-I2T, LAION-T2I, and LAION-Image, but lags on text loss. Mild hyperparameters represented by (3e-5, 2048) maintain the property of the base model with better text fitting. Such conflict in hyperparameter preference echoes observations on the trade-off between maintaining the pure-text ability of the base model and accommodating it for image-related tasks [59].

![](images/24bf6027002d670ef340d78d41d04e49c9d56ef2c3cbef20baa4f115419694e8.jpg)  
Figure 18: Loss scales with data similarly for image-text data from different sources.

![](images/8046a1078fe3b37a68c94de2b625942222454850b9bc1db87a7c0adf1ea478eb.jpg)  
Figure 19: Loss scales with data for the 0.6B model with GigaTok differently under six hyperparameter settings. Text loss prefers a small learning rate and a large batch size, while image-related tasks favor the opposite settings.

## C.3 Annealed Loss Results

We plot the loss-FLOPs results after annealing along with the non-annealed results in Figure 20. On all tasks, the relative ranking among hyperparameters is barely changed after annealing compared with results before annealing in Figure 19. One notable exception is that lr=1e-4 has a greater loss decay during annealing: In LAION-I2T, (lr=3e-5, bs=512) reaches the lowest loss at the end of the constant learning rate stage, but is outperformed by (lr=1e-4, bs=512) after annealing. We observe that annealing mainly reduces the text loss but does not affect the image-related losses significantly. One hypothesis is that the loss on image tokens might lead to noisier or smaller gradients during the annealing stage than the loss on the text tokens.

![](images/84e8d38680446393b60f8ee24a6efd2afb5c223af85626a9d4da394e0b1a9303.jpg)  
Figure 20: Loss scales with data for annealed and non-annealed checkpoints.

## C.4 Loss-VQA Performance on TextVQA

We study the relationship between pretraining loss and a more specific VQA task: TextVQA. In Figure 21, I2T loss is negatively correlated for 0.6B models, but shows the opposite trend when we scale up the model size to 4B. As a side finding, we observe that optimal hyperparameters vary for different benchmarks: Aggressive hyperparameters win on VQAv2 and GQA, but mild hyperparameters lead to better TextVQA performance despite higher I2T loss. These results suggest that post-SFT understanding performance sometimes depends on additional capabilities be-

![](images/d3b582e08e7522581c32ff0050dfe5e3995fdd88b4189db42aa8c2ec824f1a6a.jpg)  
Figure 21: I2T loss has a noisy and inconsistent correlation with TextVQA performance.

yond pretraining caption fit, such as OCR. On the other hand, treating the average score on several VQA benchmarks as the visual understanding performance might omit their distinct properties and could be highly sensitive to benchmark choices.

## C.5 Details on Loss Normalization

For LAION-T2I-force and LAION-Image-force, we first set the output logits for non-image tokens to -inf between (boi〉 and (eoi〉. This ensures that the model can only choose from the image tokens. Then, we normalize the loss (log perplexity) by log vocabulary size. Empirically, we find that the effect of the first step (forcing image output) is negligible for the loss values, indicating that the model has learned to output image tokens between (boi〉 and (eoi) on the validation set even at the first checkpoint plotted in the results (trained with 12M data).

Note that we do not apply the same normalization to the text loss and I2T loss even though the total vocabulary sizes are not identical for the three IBQ variants. The reason is that we also find the first step (forcing text output) has a negligible effect on the loss values, which shows that the model only chooses from text tokens during text and I2T evaluation on the validation set. Hence, we do not normalize the losses by the total vocabulary size.

Comparison with empirical code entropy. Normalizing by $\log _ { 2 } B$ assumes the maximal entropy of the image vocabulary, whereas real codebooks are used unevenly. To test whether $\log _ { 2 }$ B is an adequate stand-in for the actual code distribution, we compare it with the empirical unigram entropy $H _ { 1 }$ measured on the validation image tokens of all seven tokenizers (Table 9). $H _ { 1 }$ lies within 1.3% of $\log _ { 2 } B$ for every tokenizer. Uneven code usage does occur: GigaTok's rank-frequency distribution is visibly skewed, with head codes used roughly 40 times as often as tail codes. This skew is nevertheless confined to a small fraction of code ranks and carries little probability mass, so GigaTok's entropy stays close to $\log _ { 2 } B _ { \mathrm { \ell } }$ The larger deficits of IBQ-16384 and UniTok-sem instead reflect lower utilization across a broader portion of the codebook.

Substituting $H _ { 1 }$ for $\log _ { 2 }$ B. We then relate the normalized T2I loss to GenAI-all at the 0.6B scale for the three IBQ variants under both normalizations (Table 10). The two choices give nearly identical results: both recover a strong loss-performance relation, whereas the unnormalized loss explains essentially none of the benchmark variance and tokenizer identity accounts for most of it.

Table 9: Maximal entropy log2 B versus empirical unigram entropy $H _ { 1 }$ of the image tokens on the validation set. $H _ { 1 }$ is within 1.3% of $\log _ { 2 }$ B for all seven tokenizers.
<table><tr><td>Tokenizer</td><td> $\log _ { 2 } B$ </td><td> $H _ { 1 }$ </td><td> $H _ { 1 } / \log _ { 2 } B$ </td></tr><tr><td>GigaTok</td><td>14.000</td><td>13.953</td><td>0.997</td></tr><tr><td>GigaTok-DINO</td><td>14.000</td><td>13.949</td><td>0.996</td></tr><tr><td>IBQ-1024</td><td>10.000</td><td>9.977</td><td>0.998</td></tr><tr><td>IBQ-8192</td><td>13.000</td><td>12.947</td><td>0.996</td></tr><tr><td>IBQ-16384</td><td>14.000</td><td>13.841</td><td>0.989</td></tr><tr><td>UniTok</td><td>14.000</td><td>13.859</td><td>0.990</td></tr><tr><td>UniTok-sem</td><td>14.000</td><td>13.821</td><td>0.987</td></tr></table>

Table 10: Relating normalized T2I loss to GenAI-all across the three IBQ variants (0.6B models). Normalizing by the empirical entropy $H _ { 1 }$ is nearly equivalent to normalizing by $\log _ { 2 } B ,$ while the unnormalized loss leaves the variance to be explained by tokenizer identity.
<table><tr><td>Normalization</td><td>Pearson r</td><td> $R ^ { 2 }$  from loss</td><td>Additional  $R ^ { 2 }$  from tokenizer identity</td></tr><tr><td>None</td><td>+0.016</td><td>0.000</td><td>0.972</td></tr><tr><td>Divide by  $\log _ { 2 } B$ </td><td>-0.957</td><td>0.915</td><td>0.048</td></tr><tr><td>Divide by  $H _ { 1 }$ </td><td>-0.953</td><td>0.909</td><td>0.055</td></tr></table>

Scope of the normalization. This normalization corrects the loss scale associated with vocabulary size; it does not calibrate losses across tokenizer architectures at a fixed $B ,$ which is consistent with the cross-tokenizer clustering in Figure 5. Our experiments also do not cover tokenizers with severe or complete codebook collapse; in such a regime the nominal vocabulary size would no longer represent the effective prediction space, and $\log _ { 2 }$ B normalization may not be applicable.

## C.6 Ablation on I2T Objective

To verify the effect of the I2T objective in joint training, we ablate it by reformatting the I2T samples into T2I order (with drop rate 10% which enables CFG) and train with only Text+T2I data with the same hyperparameter settings. This experiment setting resembles the ablation in Section 6.2 for studying the effect of the T2I objective.

![](images/e64fc350e9cddb874f6b586a132c5b514758868a2df465450743c2ae25588a35.jpg)

![](images/baf1a0d17d221b1ef76b60d3dbc3067fc1e6d37943ff9f30aaee88f74b29fb02.jpg)  
Figure 22: In the joint modeling setting, UniToksem has lower text loss than UniTok. In Text+T2I training ablating the I2T task, the gap is not reduced, suggesting that the interference does not come from the I2T task. As a side finding, the I2T task complements text modeling in joint training for both tokenizers.

In Figure 22, we observe that the gap between tokenizers on text and T2I is not reduced by ablating I2T data. Notably, the text loss increases for both UniTok and UniTok-sem when the I2T objective is ablated. This suggests that the I2T objective complements, rather than competes with, text modeling during joint training. As a comparison, the T2I loss slightly decreases when there is no I2T task.

## D Supplementary Benchmark Information and Results

(1) Text-to-image generation. Following Liquid [59], we report (1) VQAScore on GenAI-Bench [25], calculated by prompting CLIP-FlanT5-XXL with whether the image aligns with the text and recording the probability of answering “Yes" (hence ranging from 0 to 1), and (2) gFID on MJHQ-30K [26].During inference, we use unfiltered softmax sampling and CFG scale=7.0. We adopt a unified template “{text} Generate an image based on this description. $\langle { \mathrm { b o i } } \rangle ^ { \dag }$ for evaluation. Note that we force the model output to be an image token by setting the probability of text tokens to zero during sampling. Without this operation, we find that an under-optimized unified model could output mixed text tokens and image tokens, which even holds for larger public models like Liquid-7B. Some prompts and generated results are in Figure 23. The images are generated by the Qwen3-0.6B model trained with lr=1e-4, bs=512 on 60M data. The qualitative results align with the reported benchmark scores, as GigaTok and GigaTok-DINO produce images with the best quality and alignment, followed by UniTok and UniTok-sem with worse quality but good alignment.

Prompt (a) GigaTok (b) GigaTok-DINO (c) IBQ-16384 (d) IBQ-8192 (e) IBQ-1024 (f) UniTok (g) UniTok-sem  
![](images/12db7478e03a9fb463c0afa7f148237d01c3d40eed173637c8ca2208cc136b52.jpg)  
Figure 23: Qualitative comparison of generation quality on GenAI-Bench.

Table 11: Effect of the data-shuffling seed on post-SFT benchmark scores for one GigaTok configuration (0.6B, lr=1e-4, bs=1024). Relative difference is the absolute difference divided by the mean of the two runs. POPE and MME-P are markedly more sensitive to random seeds than the other benchmarks.
<table><tr><td>Benchmark</td><td>Seed 42</td><td>Seed 37</td><td>Abs. diff.</td><td>Rel. diff.</td></tr><tr><td>VQAv2↑</td><td>51.43</td><td>51.47</td><td>0.04</td><td>0.08%</td></tr><tr><td>GQA↑</td><td>43.11</td><td>42.71</td><td>0.40</td><td>0.93%</td></tr><tr><td>POPE↑</td><td>61.40</td><td>65.90</td><td>4.50</td><td>7.07%</td></tr><tr><td>MME-P↑</td><td>825.56</td><td>785.48</td><td>40.08</td><td>4.98%</td></tr><tr><td>GenAI-all↑</td><td>0.710</td><td>0.700</td><td>0.010</td><td>1.42%</td></tr><tr><td>MJHQ-30K gFID↓</td><td>9.2558</td><td>9.0812</td><td>0.1746</td><td>1.90%</td></tr></table>

(2) Visual understanding. We use VQAv2 [16] and GQA [21] as two general VQA benchmarks TextVQA [48] is included in Appendix C.4 as a specific VQA benchmark. We also test the models on POPE [29] and MME [65] following Liquid, but find large variance among runs with different shuffling seeds (see Table 11 and analysis below). Therefore, we exclude them from our analysis.

Benchmark sensitivity to random seeds. All central experiments in this work use a single datashuffling seed (42), which controls the ordering of training data while the language-model initialization is unchanged. To quantify how much of the benchmark spread this ordering accounts for, we additionally ran SFT on one GigaTok configuration (0.6B model, lr=1e-4, bs=1024) with datashuffling seed 37 and compared the two runs (Table 11). VQAv2, GQA, GenAI-all, and MJHQ-30K agree to within 1.9% relative difference, whereas POPE changes by 7.07% and MME-P by 4.98%. This quantitatively motivates excluding POPE and MME-P from the loss-correlation analysis: their data-order sensitivity is substantially larger than that of the benchmarks we do analyze. We note that this is a two-seed comparison for a single GigaTok configuration rather than a multi-seed replication of every tokenizer comparison due to high computational cost.

## E Limitations

First, our study is limited to seven discrete image tokenizers with fixed token length and singlecodebook designs in consideration of a controlled experiment setting. For example, on the sequence length (K) axis, we observe that the mixed-modal training is sensitive to the number of image tokens seen. Since it is mathematically impossible to simultaneously hold both the number of image tokens seen and the number of images seen constant when K varies, such a study would introduce confounding variables. We leave the study of tokenizers with sub-codebooks [34], larger input resolutions [52], and different sequence lengths [67] to future research.

Second, we use pretraining validation loss as a lens for studying joint modeling behavior, not as a replacement for downstream evaluation after SFT or for tokenizer selection. Although I2T loss provides a useful signal, it is less predictive for specialized tasks such as TextVQA.

Third, due to the high computational cost, we do not conduct a full scaling law study on unified multimodal AR training that starts from scratch with more model scales or derive compute-efficient frontiers. Future work could derive and compare image tokenizers’ effects on the compute-efficient frontier of multimodal AR models.

Fourth, analyses on the validation set, including PMI and n-gram entropy, are diagnostic rather than mechanistic, and high-order empirical entropy can suffer from finite-sample bias.

Fifth, our empirical findings are established for Qwen3-based pure-AR unified models. Holding the backbone family fixed is itself a control in our study, since it lets us vary the visual token space while keeping the downstream architecture and training recipe consistent. We therefore do not claim that every loss relationship or tokenizer ranking is unchanged under a different backbone family; validating them would require matched retraining and evaluation across tokenizers under that backbone. The analysis framework itself can be applied to another language-model family, and we view such cross-family validation as an important direction for future work.

Finally, our downstream evaluation covers a limited set of generation and VQA benchmarks; broader task, safety, and human-preference evaluations are needed to fully evaluate the models. In particular, our understanding evaluation covers general VQA (VQAv2 and GQA) and the OCR-oriented TextVQA, but includes no dedicated region-level grounding or fine-grained spatial-reasoning benchmark. We therefore do not claim that our findings extend to these specialized visual capabilities, and evaluating how the image token space affects them remains future work.

## F Third-Party Attribution Notices

Portions of this work use third-party software released under the MIT License: we train UniTok and UniTok-sem variants using the official UniTok code (https://github.com/FoundationVision/ UniTok), and we use the GigaTok and GigaTok-DINO tokenizers (https://github.com/ SilentView/GigaTok). The copyright and permission notices below are reproduced as required by that license.

Copyright (c) 2024 FoundationVision

Copyright (c) 2025 GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation Authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS", WITHOUT WARRANTY OF ANY KIND EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONIN-FRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## G Broader Impacts

This work studies image tokenizers in the context of unified pure-AR multimodal models. By showing how tokenizer design affects modeling, including cross-modal modeling and text-side behavior, our analysis may help researchers design and diagnose multimodal training beyond reconstruction or single-axis evaluation. Improved tokenizer analysis could contribute to stronger generation and understanding models, which may support creative, educational, and accessibility applications.

However, stronger multimodal models also raise risks such as biased outputs, misleading synthetic content, and misuse. Our experiments use publicly available images and do not introduce a deployed system, but future work using these insights should be paired with standard safety, bias, privacy, and responsible-release evaluations.