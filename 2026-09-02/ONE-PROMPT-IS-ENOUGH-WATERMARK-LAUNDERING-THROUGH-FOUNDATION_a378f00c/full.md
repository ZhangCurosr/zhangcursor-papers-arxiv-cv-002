# ONE PROMPT IS ENOUGH: WATERMARK LAUNDERING THROUGH FOUNDATION IMAGE MODELS

Jidong Yang<sup>1</sup>, Qi Li<sup>1</sup>, Wei Zong<sup>2</sup>, Yang-Wai Chow<sup>2</sup>, Willy Susilo<sup>2</sup>, Huaike Yu<sup>1</sup>, Chunpeng Wang<sup>1</sup>, and Suo Gao<sup>3</sup>

<sup>1</sup>Key Laboratory of Computing Power Network and Information Security, Ministry of Education; Shandong Computer Science Center; Shandong Provincial Key Laboratory of Industrial Network and Information System Security; Shandong Fundamental Research Center for Computer Science; Qilu University of Technology (Shandong Academy of Sciences), Jinan 250353, China

<sup>2</sup>Institute of Cybersecurity and Cryptology (iC2), University of Wollongong, Australia

<sup>3</sup>School of Information Science and Engineering, Dalian Polytechnic University, Dalian 116034, China

qluliqi@163.com; jidong\_yang\_paper@163.com; mpeng1122@163.com; huaikeyu@gmail.com wzong@uow.edu.au; caseyc@uow.edu.au; wsusilo@uow.edu.au; gaosuodlpu@163.com

## ABSTRACT

Invisible watermarks are typically evaluated against predefined perturbations, such as compression, blur, noise, cropping, and denoising. However, public foundation image models expose a distinct threat: an attacker can provide a watermarked image with a single reconstruction prompt and obtain a visually faithful output from which the invisible watermark can no longer be decoded reliably. We formalize this failure mode as watermark laundering and evaluate it by evaluate it through a joint payload-fidelity profile that measures bit error rate(BER) together with visual and semantic preservation. Across six OpenAI and Google image editing models, three representative watermarking schemes, and 1,800 reconstructed outputs, we observe two complementary laundering regimes: OpenAI models achieve the strongest attack ability across the evaluated schemes, whereas Nano Banana 2 reveals that DwtDct watermarks remain vulnerable even under high-fidelity reconstruction. Prompt ablations show that no single removal-oriented directive is necessary for attacking watermarks, indicating that the effect is primarily induced by the reconstruction pathway rather than explicit attack prompt. Comparisons with conventional attacks further indicate that prompt-conditioned reconstruction constitutes a distinct operational attack interface, motivating foundation-model reconstruction as a missing robustness condition for invisible watermark evaluation.

Keywords Invisible image watermarking · Watermark laundering · Foundation image models · Provenance security · Black-box attacks

## 1 Introduction

Invisible watermarks have become a crucial technology for image provenance, copyright protection, and AI-generated content identification. Invisible watermark robustness is usually evaluated using a fixed suite of compression, filtering, geometric transformation, or denoising attacks [1–3]. Recent studies have further extended this evaluation to learningbased removal and regeneration attacks. Such evaluations largely assume the attacker applies an explicit imageprocessing operator or a dedicated removal model. This assumption becomes increasingly incomplete as image editing capabilities are integrated into foundation image models. An attacker can request a content-preserving reconstruction of watermarked images through a request framed as noise or artifact removal. The output may remain visually and semantically faithful while making the embedded payload unreliable to the victim decoder.

We define this failure mode as watermark laundering (Figure 1), by analogy with information laundering, in which an intermediary preserves the usable content while weakening its association with the original input [4–8]. Here, the intermediary is a public foundation image model, and the weakened information is the victim payload. Given a watermarked image, watermark laundering occurs when prompt-conditioned reconstruction preserves its visible or semantic content while making the original watermark unreliable to the victim decoder. Although the reconstructed output may carry provenance metadata conforming to the Coalition for Content Provenance and Authenticity (C2PA) specification, including associated Content Credentials [9], such signals neither recover the victim payload nor constitute provenance substitution. Note that payload disruption alone is insufficient: decoder failure caused by severe image destruction is not meaningful laundering, which requires the output to remain visually or semantically usable.

![](images/5527fe918ae817db2af4e79f729d2aafc0a372c660a0b605fefe207c6da3139e.jpg)  
Figure 1: Threat model and evaluation for single-prompt watermark laundering through a public black-box foundation image editor. An attacker submits a watermarked image and a single reconstruction prompt without access to the victim decoder, embedding key, model parameters, gradients, or detector feedback. The reconstructed output is evaluated separately for victim-payload recoverability and content preservation.

Representative post-hoc watermarking schemes include HiDDeN, StegaStamp, RivaGAN, and MBRS, together with attacks based on signal processing, denoising, learned removal, and generative regeneration [10–16]. Unlike prior regeneration attacks that typically require access to a local or controllable generative pipeline, we study deployed public foundation image models invoked through a single natural-language prompt. Therefore, our contribution lies in characterizing a distinct operational threat rather than proposing a new internal removal method. Instruction-guided editing methods further demonstrate that image reconstruction can preserve scene content while rewriting the underlying pixel-level information [17–19].

We systematically evaluate this threat across six public foundation image models from two provider lineages, three representative watermarking schemes, and 1,800 reconstructed outputs. OpenAI models produce the strongest overall payload disruption across the evaluated schemes, while Nano Banana 2 demonstrates that DwtDct remains vulnerable even under high-fidelity reconstruction. Prompt ablations further show that explicit hidden-information-removal wording is not necessary for payload disruption. Requests framed as reconstruction, restoration, or artifact removal may therefore invoke generative processes that weaken invisible evidence, revealing a mismatch between natural-language task framing and low-level forensic preservation. Our experiments study this watermark-laundering effect rather than a bypass of any specific provider policy, and they do not imply any change in legal copyright ownership. Our contributions are summarized as follows:

• We establish a payload–fidelity evaluation framework for watermark laundering under a single-prompt black-box threat model. Rather than introducing a dedicated SOTA attack, we identify foundation-model reconstruction as a missing robustness condition in invisible watermark evaluation.

• We evaluate six publicly accessible foundation image models and three invisible watermarking schemes across 1,800 outputs generated from a stratified set of 100 images, revealing complementary high-disruption and high-fidelity laundering regimes.

• We characterize watermark laundering through prompt ablations, comparisons with conventional attacks, and high-frequency residual (HFR) analysis, showing that payload disruption is largely insensitive to the tested prompt components and only partially explained by high-frequency changes.

## 2 Threat Model and Theoretical Framework

Let $W _ { V } = ( \mathrm { E m b e d } _ { W _ { V } } , \mathrm { D e c o d e } _ { W _ { V } } )$ be a victim watermarking scheme. Given a clean image x and a k-bits payload $b \in \{ 0 , 1 \} ^ { k }$ , the corresponding watermarked image is $x _ { V } = \operatorname { E m b e d } _ { W _ { V } } ( x , b )$ . A black-box foundation model editor F takes a natural-language prompt p as input and returns the reconstructed output $y = F ( x _ { V } , p )$

Definition 1 (Watermark laundering). The transformation $x _ { V } \mapsto y$ is watermark laundering when $\hat { b } = \mathrm { D e c o d e } _ { W _ { V } } ( y )$ is an unreliable estimate of b while y remains visually faithful to x<sub>V</sub> and preserves its semantic content. For an evaluation set $D = \{ ( x _ { i } , \bar { b _ { i } } ) \} _ { i = 1 } ^ { N } , l e t x _ { V , i } = \mathrm { E m b e d } _ { W _ { V } } ( x _ { i } , \bar { b } _ { i } )$ and $y _ { i } = F ( x _ { V , i } , p )$ . We report the following joint profile:

$$
\begin{array} { r } { \mathcal { P } _ { D } ( W _ { V } , F , p ) = \big ( \overline { { \mathrm { B E R } } } _ { D } , \overline { { M } } _ { \mathrm { r e f } , D } , \overline { { M } } _ { \mathrm { c l e a n } , D } , } \\ { \overline { { M } } _ { \mathrm { s e m } , D } , \overline { { M } } _ { \mathrm { n r } , D } \big ) . \qquad } \end{array}\tag{1}
$$

where each overbar denotes an average over $D \colon M _ { \mathrm { r e f } }$ measuresfidelity to the watermarked input, $M _ { \mathrm { c l e a n } }$ measures fidelity to the clean image, $M _ { \mathrm { s e m } }$ measures semantic preservation, and $\dot { M } _ { \mathrm { n r } }$ measures no-reference image quality. We report raw BER. Random binary recovery corresponds to $B E R = 0 . 5 ,$ so movement toward 0.5from either direction indicates stronger disruption. A value above 0.5 is not treated as stronger than random recovery. Fidelity metrics are reported separately because their directions cannot be represented correctly by a single aggregate score.

Our formulation imposes three conditions. First, the adversary acts through a natural-language instruction rather than training, gradients, or optimization informed by the decoder. Second, the transformation is reconstructive: the editor returns newly rendered pixels rather than a tuned additive perturbation. Third, laundering has two requirements. Payload disruption without content preservation is ordinary destructive removal, whereas preservation without payload disruption is a benign edit. Define the bounded dataset-level removal score $R _ { \mathrm { B E R } } ( D ) = 1 - 2 | \overline { { \mathrm { B E R } } } _ { D } - 0 . 5 |$ , for which $\bar { R } _ { \mathrm { B E R } } ( D ) = 1$ denotes random bit recovery. For a prespecified application-specific similarity function Sim, oriented so that larger values indicate stronger content preservation, define $\begin{array} { r } { \overline { { \mathrm { S i m } } } _ { D } = | D | ^ { - 1 } \sum _ { i = 1 } ^ { N } \mathrm { S i m } ( y _ { i } , x _ { V , i } ) } \end{array}$ . A conceptual binary criterion can then be written as

$$
R _ { \mathrm { B E R } } ( D ) \geq \tau _ { \mathrm { R R } } \quad \mathrm { a n d } \quad \overline { { \mathrm { S i m } } } _ { D } \geq \tau _ { \mathrm { s i m } } .\tag{2}
$$

where $\tau _ { \mathrm { R R } }$ and $\tau _ { \mathrm { s i m } }$ are application-specific thresholds. The abstract function Sim is a conceptual criterion rather than an aggregate of the reported fidelity metrics. We do not instantiate universal values for these thresholds or report a thresholded laundering success rate. The experiments instead use the continuous joint profile because a single cutoff would obscure the BER–fidelity tradeoff.

The attacker has ordinary black-box API access to $F$ but no knowledge of the victim algorithm, payload, or key. The attacker receives no feedback from the victim decoder or detector and has no access to model training or gradients. The attacker submits a watermarked image and a single prompt to a public editing interface, which returns a reconstructed image rather than a tuned local perturbation. Success requires the returned image to remain usable and related to the source while making recovery of the victim payload unreliable.

Our primary outcome is victim watermark laundering. Let $L$ denote the conceptual conjunction in Equation (2) under application-specific thresholds, and let $E _ { \mathrm { p r o v } } \in \{ 0 , \tilde { 1 } \}$ indicate whether the returned output contains verified provider provenance evidence. We reserve provenance substitution for the stronger state

$$
S = L \wedge E _ { \mathrm { p r o v } } .\tag{3}
$$

Thus, provider metadata or a provider watermark cannot substitute for decoding evidence from the victim watermark. Conversely, L alone does not show that the image has acquired a new machine-readable origin. If verified provider evidence is present, an output satisfying state $S$ may present an apparent replacement of source or ownership evidence at the interface level, but it does not transfer legal copyright.

We use a behavioral bottleneck abstraction rather than claiming knowledge of proprietary implementations. Let $Z = E _ { F } ( x _ { V } )$ denote the representation used for reconstruction, and let $y = R _ { F , p } ( Z , \xi )$ denote the rendered output. Let $C = C ( \dot { x } )$ summarize objects, layout, scene structure, and visibly relevant appearance. Here, $I ( U ; V )$ denotes mutual information, and $I ( U ; V \mid T )$ denotes conditional mutual information, both measured in bits. The account makes three assumptions. First, semantic sufficiency requires $Z$ to retain enough information about $C$ to reconstruct the scene. Second, forensic insufficiency requires ${ \dot { I } } ( b ; Z \mid C )$ to be small, meaning that after conditioning on the visible content $C ,$ , the representation $Z$ retains little additional information about the watermark payload b. Third, the editor performs no adaptation informed by the decoder. With payloads sampled independently of $C _ { i }$ , the second condition implies that $I ( b ; Z )$ is small. These assumptions separate two quantities that robustness evaluations often conflate: a model can preserve information about depicted content while discarding information about the forensic payload embedded in the image [20–22].

Proposition 1 (Reconstruction bottleneck). For the Markov pathway $b  x _ { V }  Z  y ,$ data processing gives

$$
I ( b ; y \mid p ) \leq I ( b ; Z ) .\tag{4}
$$

Thus, a representation that retains little information about the victim payload cannot support reliable downstream recovery, even when it preserves visible content. The next proposition states the corresponding BER implication.

Proposition 2 (Implication for optimal recovery). If the payload bits $b _ { 1 } , \ldots , b _ { k }$ are independent and uniform, let $P _ { e , j } ^ { * }$ be the minimum error probability over all estimators of $\mathbf { \dot { \theta } } _ { j }$ from y. Then

$$
\mathrm { B E R } ^ { * } ( b \mid y ) : = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } P _ { e , j } ^ { * } \ge \frac { 1 } { k } \sum _ { j = 1 } ^ { k } h _ { 2 } ^ { - 1 } ( 1 - I ( b _ { j } ; y ) ) .\tag{5}
$$

Here, $h _ { 2 } ( q ) = - q \log _ { 2 } q - ( 1 - q ) \log _ { 2 } ( 1 - q )$ is the binary entropyfunction, and $h _ { 2 } ^ { - 1 }$ denotes its inverse restricted to $q \in [ 0 , 0 . 5 ]$ . For a uniform payload bit, $1 - I ( b _ { j } ; y )$ represents the residual uncertainty about $b _ { j }$ after observing y. Together with $P _ { e , j } ^ { * } \leq 0 . 5 ,$ , the bound implies that vanishing information about every bitforces optimal recovery toward the limit ofrandom decoding. This statement concerns recoverable information. The empirical tables instead report the raw BER of each fixed victim decoder and interpret it by proximity to 0.5. The result follows from the bitwise form of Fano’s inequality and the local expansion of binary entropy [23].

The bottleneck also predicts invariance to prompts. If prompts affect only the renderer and do not introduce a new information channel from b beyond $Z ,$ then changing prompt wording can alter color, texture, sharpness, or geometry but cannot recover payload information already discarded by the representation. Explicit “remove hidden information” language is therefore not necessary for BER to approach 0.5. This claim is conditional because prompts can still determine whether the result remains faithful enough to qualify as laundering.

For residual-sensitive schemes, we measure normalized high-frequency change

$$
\mathrm { H F R } ( x _ { V } , y ) = \frac { \| P _ { H } ( y ) - P _ { H } ( x _ { V } ) \| _ { 2 } } { \| P _ { H } ( x _ { V } ) \| _ { 2 } + \eta } ,\tag{6}
$$

where $P _ { H }$ selects high DCT bands. Large changes in the high-frequency residual can cause decoder margins to cross their decision boundaries. HFR summarizes the magnitude of these changes, but it need not determine BER because schemes differ in payload placement, redundancy, and decoder invariances.

The evaluation tests five linked expectations. Payload disruption can coexist with fidelity (H1), and schemes can respond asymmetrically because their carriers and decoders differ (H2). Prompt components should affect fidelity more than BER when payload loss occurs before rendering (H3). Later model versions can improve fidelity while preserving greater payload recoverability (H4). HFR should be positively associated with raw BER within the primary grid (H5). Because BER is not monotonic as an attack metric beyond 0.5, this association is interpreted as a relationship with decoder error rather than a monotonic disruption score. The theoretical account would be weakened if movement toward random recovery required explicit removal wording, occurred only when semantic fidelity collapsed, or showed no residual association for any evaluated scheme.

## 3 Experimental Setup

Models and watermarks. We evaluate GPT Image 1, GPT Image 1.5, and GPT Image 2, together with Nano Banana, Nano Banana Pro, and Nano Banana 2 [24–27]. The three victim watermarking schemes are DwtDct [28], DwtDctSvd [29], and RivaGAN [12]. All clean references are resized to 1024 × 1024 before watermark embedding, and all calls use the public image editing interfaces at that resolution. Before every API call, we retain only inputs whose payload is correctly decoded, preventing embedding failures from being counted as laundering.

Data and protocol. The stratified target list contains 100 MS-COCO images [30]: 40 scenes rich in texture, 30 scenes with large flat regions, and 30 scenes containing text. The full factorial grid has $3 \times 6 = 1 8$ watermark–model cells and 100 outputs per cell, for 1,800 primary API calls. A transient failure before output generation is retried once on the same image. A persistent failure triggers replacement with the next unused image from the same stratum, independently of decoder performance. Thus, every cell preserves the same stratum counts, although a rare replacement can break pairing of individual images across editing models. Every cell uses the same structured reconstruction prompt P<sub>struct</sub> without prompt search, repeated editing, or detector feedback. The full structured prompt and minimal prompt are reproduced in the supplementary material.

The structured prompt asks the editor to preserve objects, layout, canvas size, luminance, color statistics, and regional geometry while returning a single reconstructed image. It also includes preservation of visible content, grayscale luminance guidance, regional color constraints, removal of hidden information, and final output validation. These instructions define a constrained reconstruction request rather than an optimization loop: there is one input, one prompt, and one returned output. Failure handling never uses the performance of the victim decoder. Figure 2 visualizes the modular organization of $P _ { \mathrm { s t r u c t } }$

![](images/7395be484686291d6c5637a31ed9b7907f9bf6bd3f8bc7cca9402ce738f8fd1f.jpg)  
Figure 2: Structure of the single prompt $P _ { \mathrm { s t r u c t } }$ . The six gates represent prompt clauses for canvas, geometry, luminance, color, appearance, and hidden-signal handling before a black-box reconstruction call; they describe the request rather than observed internal model stages.

Ablations, metrics, and baselines. The main evaluation covers DwtDct, DwtDctSvd, and RivaGAN, whereas the focused prompt ablation is conducted on DwtDct only, using Nano Banana 2 and GPT Image 2. The tested variants separately remove preservation of visible content, joint luminance and color guidance, appearance and geometry constraints, and explicit language about hidden information. The evaluation also includes a preservation constraint and a minimal prompt. We report raw victim BER, for which proximity to 0.5 indicates stronger disruption. Reference fidelity is measured with PSNR, SSIM [31], LPIPS [32], and CLIP-Sim [33]. Table 2 reports clean-reference and no-reference image quality metrics. Conventional comparisons include Gaussian blur, Gaussian noise, contrast scaling, 30<sup>◦</sup> rotation, 50% cropping, and BM3D [14]. Means are reported over the stratified evaluation design. Formal uncertainty testing remains a limitation.

Reference and semantic metrics assess whether the output remains tied to the submitted image. No-reference metrics instead assess image quality without direct comparison to the source. We keep these metric families separate because a model may “repair” blur, darkness, or compression artifacts and obtain a better no-reference score while moving farther from the source under PSNR or SSIM. Likewise, BER is never interpreted alone: a value near the limit of random recovery does not provide strong laundering evidence when the output is unrecognizable under Definition 1.

## 4 Results

## 4.1 Joint BER–Fidelity Profile

Table 1 reports the main experimental results for all 18 watermark–model configurations under $P _ { \mathrm { s t r u c t } }$ , with N = 100 outputs per configuration. It jointly presents victim-payload disruption and reconstruction fidelity rather than reducing the two objectives to a single score. The GPT models occupy the regime with stronger disruption and lower reference fidelity. GPT Image 1 has the mean BER closest to random recovery (0.4808). GPT Image 2 improves mean PSNR over the earlier GPT models while retaining substantial disruption, with BER values of 0.4266, 0.4517, and 0.3620 for DwtDct, DwtDctSvd, and RivaGAN, respectively. The Google models yield progressively higher fidelity. Nano Banana 2 reaches a mean PSNR of 30.27 and a mean SSIM of 0.879 against the watermarked input, while DwtDct remains substantially disrupted at BER 0.4110. This high-fidelity laundering condition supports H1 within the evaluated setting because payload disruption persists without obvious image destruction.

The table does not support an explanation that is independent of the watermark scheme. DwtDct remains at a BER of approximately 0.41 across all six editing models, including the high-fidelity Google models. DwtDctSvd is similarly vulnerable to the earlier models but becomes more recoverable under Nano Banana Pro and Nano Banana 2. RivaGAN shows the largest model dependence: its learned decoder is strongly disrupted by the GPT models but retains substantially more payload under the later Google models. These differences are consistent with H2. The pattern across schemes is inconsistent with a single explanation based on frequency bands and is compatible with differences in payload placement, redundancy, and decoder invariances.

For both providers, later model versions improve source fidelity while mean BER moves farther from 0.5. Mean BER changes from 0.4808 to 0.4655 to 0.4134 across GPT Image versions and from 0.3282 to 0.2941 to 0.2667 across Nano Banana versions. This trend supports H4: later versions can improve fidelity while preserving greater payload recoverability rather than making laundering monotonically stronger. DwtDct is the exception, remaining at a BER of approximately 0.41 across all Google models despite the large fidelity gain.

Table 1: Core laundering results under $P _ { \mathrm { s t r u c t } }$ . Panel (a) reports BER for each victim watermark; values closer to 0.5 indicate stronger payload disruption. Panel (b) reports fidelity metrics averaged across the three watermarks, using the watermarked input as the reference. Each watermark–model cell has $N = \bar { 1 0 0 }$  
(a) Payload disruption (BER→ 0.5)
<table><tr><td rowspan=1 colspan=1>Editing model    DwtDctDwtDctSvdRivaGAN</td></tr><tr><td rowspan=1 colspan=1>GPT Image 1      0.4944      0.4907    0.4574GPT Image 1.5    0.4353     0.4963    0.4649</td></tr><tr><td rowspan=1 colspan=1>GPT Image 2      0.4266      0.4517    0.3620Nano Banana      0.4106     0.4167    0.1572Nano Banana Pro  0.4159      0.2947    0.1717Nano Banana 2    0.4110      0.2658    0.1233</td></tr><tr><td rowspan=1 colspan=1>(b) Reconstruction fidelity</td></tr><tr><td rowspan=1 colspan=1>Editing model    PSNR↑SSIM↑ CLIP-Sim↑</td></tr><tr><td rowspan=1 colspan=1>GPT Image 1     14.03600.4051     0.8939GPT Image 1.5   15.07320.4069     0.9056GPT Image 2     16.9877 0.5017     0.9424Nano Banana     20.4589 0.5738     0.9682Nano Banana Pro 25.8871 0.7497     0.9429Nano Banana 2   30.27260.8794     0.9837</td></tr></table>

Figure 3 presents the two objectives using clean-reference PSNR. Points near the dashed BER = 0.5 line indicate payload recovery approaching the random limit, while movement to the right indicates greater source fidelity. GPT models cluster in a region of high disruption and low fidelity. Nano Banana 2 moves substantially to the right, although its DwtDct point remains above BER 0.4. A ranking based only on BER would therefore obscure this high-fidelity condition.

Figure 4(a) illustrates the two regimes. GPT Image 2 preserves scene semantics but visibly changes exposure, contrast, texture, boundaries, and regions that resemble text, whereas Nano Banana 2 remains closely aligned with the source. The no-reference image quality metrics in Table 2 sometimes favor the more extensively altered outputs. This result reflects a mismatch between perceived naturalness and source fidelity rather than contradicting the lower fidelity at the pixel level.

## 4.2 Prompt Sensitivity of Watermark Laundering

Figure 4(b) summarizes the DwtDct ablation. On Nano Banana 2, removing any tested module keeps BER within 0.0023 of the value of 0.4110 under the full prompt. This result is consistent with payload disruption not requiring explicit language about hidden information. Fidelity is more sensitive. Removing the appearance and geometry constraints lowers PSNR by 0.302. Removing the joint luminance and color guidance increases SSIM by 0.0023 and decreases LPIPS, with essentially unchanged BER. The minimal prompt lowers Nano Banana 2 PSNR by 2.369. For GPT Image 2, it increases PSNR by 0.788 while changing BER by only −0.0051. The contrast between stable BER and larger fidelity changes supports H3. These descriptive differences are not treated as statistically resolved effects because formal uncertainty testing is absent.

The luminance and color guidance ablation is descriptive rather than a direct observation of model internals. On Nano Banana 2, omitting the grayscale luminance anchor and region-wise color constraint improves SSIM by 0.0023 and produces the lowest LPIPS among its ablations without changing BER. Because the editors are closed source, this result describes how the requested constraints affect the returned reconstruction. It does not establish that the models instantiate the requested intermediate representations. The stable BER is consistent with the editing pathway contributing to payload disruption without any single prompt module being necessary.

The comparison with the minimal prompt also constrains the policy interpretation. Explicit removal language is not required to disrupt the victim payload. The minimal request reaches BER 0.4167 for Nano Banana 2 and 0.4215 for GPT Image 2. The relevant security concern therefore extends beyond a single prohibited phrase that a lexical filter could block. General reconstruction capability can produce the transformation even when the request is phrased as ordinary image recreation. This result does not demonstrate a bypass of provider safeguards, but it indicates that lexical filtering alone would not address the observed reconstruction behavior.

![](images/a84c0732c0639af6def214db5d6ea3ded9d08262886f714c04629cb864dfcd58.jpg)  
GPT Image 1 O GPT Image 1.5 O GPT Image 2  Nano Banana O Nano Banana Pro O Nano Banana 2 Watermark O DwtDct□ DwtDctSvdΔ RivaGAN

Figure 3: Victim BER versus clean-reference PSNR. Marker shape denotes the watermark scheme. The dashed line marks random binary recovery.  
![](images/c6aadad907e95b2f5206277e62314a4878c91e44d7ed864fc55b9c78821fde61.jpg)

![](images/8056a840bd549e6dc769b1bbecb7c52fb5aad6bc736c0686384135e3e601c3bb.jpg)  
Figure 4: (a) Representative outputs generated with one prompt across editing models and victim watermark schemes. (b) Prompt-ablation changes relative to $P _ { \mathrm { s t r u c t } }$ for DwtDct; positive ∆BER moves toward random recovery because all evaluated BER values are below 0.5.

## 4.3 Comparison with Conventional Attacks

Table 2 combines payload disruption, clean-reference fidelity, and no-reference image quality for the retained conventional baselines and two laundering conditions produced by foundation models. For DwtDct, contrast scaling gives the BER closest to 0.5. The BER under Gaussian noise is 0.6225, which lies 0.1225 above the point of random recovery and coincides with low clean-reference fidelity and weaker no-reference quality. GPT Image 2 gives the BER closest to 0.5 for DwtDctSvd and RivaGAN. It also gives the best NIQE, BRISQUE, and CLIP-IQA values within every displayed watermark block, which is consistent with automatic repair behavior. Nano Banana 2 provides the laundering regime with higher fidelity, whereas Gaussian blur and BM3D preserve the clean reference most closely but are less consistently disruptive. The comparison does not exhaust regeneration methods. Its purpose is to distinguish the public reconstruction interface from fixed local operators.

Table 2: BER and image-quality comparison. Background colors distinguish victim watermark schemes. PSNR-clean and SSIM-clean use the clean reference; NIQE, BRISQUE, and CLIP-IQA are no-reference metrics. Bold marks the BER closest to 0.5 or the best quality value within each color block.
<table><tr><td>Attack condition</td><td>BER→ 0.5</td><td>PSNR-clean↑</td><td>SSIM-clean↑</td><td>NIQE↓</td><td>BRISQUE↓</td><td>CLIP-IQA↑</td></tr><tr><td colspan="7">DwtDct</td></tr><tr><td>Gaussian blur</td><td rowspan="4">0.3134 0.6225 0.4587</td><td rowspan="4">35.6690 20.4483 14.2076</td><td rowspan="4">0.9631 0.4112 0.6465</td><td rowspan="4">1.6203 1.9251 2.2275</td><td rowspan="4">50.4784 58.6475 58.5809</td><td>0.6505</td></tr><tr><td>Gaussian noise</td><td>0.4967</td></tr><tr><td>Contrast</td><td>0.4370</td></tr><tr><td>Rotation</td><td></td></tr><tr><td>Cropping</td><td>0.4094 0.0356</td><td>9.7087 10.0902</td><td>0.3037 0.2796</td><td>1.4469 1.4783</td><td>28.3639 45.3134</td><td>0.2276 0.4150</td></tr><tr><td>BM3D denoise</td><td>0.4097</td><td>36.7900</td><td>0.9423</td><td>1.6544</td><td>64.1586</td><td>0.6470</td></tr><tr><td>GPT Image 2</td><td>0.4266</td><td>16.3935</td><td>0.4758</td><td>1.4092</td><td>18.4072</td><td>0.7361</td></tr><tr><td>Nano Bañana 2</td><td>0.4110</td><td>30.4004</td><td>0.8769</td><td>1.4113</td><td>20.1624</td><td>0.5394</td></tr><tr><td colspan="7">DwtDctSvd</td></tr><tr><td>Gaussian blur</td><td>0.0000</td><td>35.1716</td><td>0.9632</td><td>1.6243</td><td>50.2400</td><td>0.5898</td></tr><tr><td>Gaussian noise</td><td>0.1331</td><td>20.3887</td><td>0.3921</td><td>1.9254</td><td>58.9234</td><td>0.4717</td></tr><tr><td>Contrast</td><td>0.3787</td><td>14.3713</td><td>0.6686</td><td>2.2672</td><td>58.6463</td><td>0.3656</td></tr><tr><td>Rotation</td><td>0.4062</td><td>9.8574</td><td>0.3104</td><td>1.4467</td><td>23.5672</td><td>0.1996</td></tr><tr><td>Cropping</td><td>0.0000</td><td>10.2587</td><td>0.2909</td><td>1.4771</td><td>44.1477</td><td></td></tr><tr><td>BM3D denoise</td><td>0.3591</td><td>36.2090</td><td>0.9412</td><td>1.6607</td><td>63.8982</td><td>0.3526</td></tr><tr><td>GPT Image 2</td><td>0.4517</td><td>16.9978</td><td>0.5135</td><td>1.3907</td><td>16.3967</td><td>0.6128</td></tr><tr><td>Nano Bañana 2</td><td>0.2658</td><td>30.7813</td><td>0.8847</td><td>1.3948</td><td>19.8320</td><td>0.7036</td></tr><tr><td>RivaGAN</td><td></td><td></td><td></td><td></td><td></td><td>0.4452</td></tr><tr><td></td><td></td><td>35.2932</td><td></td><td></td><td></td><td></td></tr><tr><td>Gaussian blur Gaussian noise</td><td>0.0000 0.0591</td><td></td><td>0.9631</td><td>1.6125 1.9186</td><td>50.7863</td><td>0.5548</td></tr><tr><td></td><td>0.1309</td><td>20.4930 14.3946</td><td>0.4015 0.6524</td><td></td><td>58.4579</td><td>0.4425</td></tr><tr><td>Contrast</td><td>0.3016</td><td>9.9658</td><td>0.3011</td><td>2.2565 1.4265</td><td>57.0378</td><td>0.3511</td></tr><tr><td>Rotation</td><td>0.0013</td><td>10.0768</td><td>0.2683</td><td>1.4234</td><td>26.4614</td><td>0.2225</td></tr><tr><td>Cropping</td><td></td><td></td><td></td><td></td><td>44.4159</td><td>0.3787</td></tr><tr><td>BM3D denoise</td><td>0.0191</td><td>36.2624</td><td>0.9415</td><td>1.6474</td><td>62.1148</td><td>0.6002</td></tr><tr><td>GPT Image 2</td><td>0.3620</td><td>17.6600</td><td>0.5171</td><td>1.3672</td><td>17.2455</td><td>0.6726</td></tr><tr><td>Nano Banana 2</td><td>0.1233</td><td>30.1518</td><td>0.8774</td><td>1.3760</td><td>18.5228</td><td>0.5033</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.4 High-Frequency Residual Analysis

Across 1,800 matched input–output pairs, HFR is positively associated with raw BER (Pearson $r = 0 . 3 1 5 8 .$ , Spearman $\rho \ : = \ : 0 . 3 6 0 7 )$ . This statistic describes decoder error and is not a monotonic measure of disruption for individual observations with BER above 0.5. The association is positive within DwtDct $( r = 0 . 3 3 6 1 )$ , DwtDctSvd $( r = 0 . 3 0 1 4 )$ and RivaGAN $( r = 0 . 4 6 6 2 )$ , but it is weak within several model aggregates. The aggregate association supports H5, while the weak within-model results limit HFR to one contributor to decoder errors rather than a complete scalar explanation. The measured change in high frequencies does not identify the coefficients that carry the payload or the learned invariances of each decoder.

The means for each model indicate a change in reconstruction behavior. Mean HFR decreases from 7.3947 for GPT Image 1 to 3.4389 for GPT Image 1.5 and 2.5204 for GPT Image 2, while mean BER changes from 0.4808 to 0.4655 to 0.4134. Later OpenAI models therefore produce smaller mean changes in the high-frequency residual while also moving farther from random bit recovery. Nano Banana Pro has the strongest association between HFR and BER within a model $( r = 0 . 3 5 8 5 )$ , whereas the correlations for the GPT models are near zero. HFR measures the magnitude of residual change but not whether that change aligns with the effective decision directions of a particular decoder.

GPT Image 2 also exhibits apparent repair behavior. Dark, blurry, compressed, or low-contrast inputs may be brightened, sharpened, or locally reconstructed. Such outputs can appear more natural and score better on no-reference quality metrics even when PSNR and SSIM decrease against the source. No-reference metrics assess perceived naturalness, whereas PSNR and SSIM assess source fidelity. The high-fidelity Nano Banana 2 results further indicate that laundering is not confined to visibly repaired or aggressively reconstructed images.

## 5 Security Implications for Watermark Laundering

Watermark laundering indicates a mismatch between model safety checks and provenance security. An editor may reject explicit requests to remove a watermark or alter ownership evidence while accepting a visually benign instruction to recreate, restore, or clean an image. If the reconstruction pathway discards the victim carrier, compliance with a policy defined at the level of wording does not guarantee provenance preservation in the output. An attacker can therefore use the prompt to repurpose a permitted editing operation for an effect that is relevant to provenance. The ablation results indicate that phrase blocking alone is insufficient because explicit language about removing hidden information is not necessary for the observed disruption.

The stronger state in Equation (3) further clarifies the risk. Our provider marker audit found C2PA/Content Credentials binary markers in all 100 sampled outputs from each provider [9, 34, 35]. A compatible signature verifier was unavailable, and SynthID was not tested. The audit therefore indicates a provider-side marker pathway but does not establish $E _ { \mathrm { p r o v } } = 1$ or contribute to the laundering success criterion. If recovery of the victim payload becomes unreliable and the editor attaches verified provenance evidence, the returned file can present an apparent machinereadable origin that differs from the provenance of the victim image. An attacker could also combine reconstruction with ordinary metadata or changes to visible labels to create an apparent copyright or attribution replacement. These operations constitute attacks on provenance and ownership evidence, but they do not change the underlying legal rights. Defenses should therefore bind authorization and provenance across transformations, preserve or cryptographically link upstream claims, and test outputs for loss of the victim signal rather than relying only on prompt moderation or the presence of a new provider mark.

## 6 Discussion and Conclusion

Within the evaluated setting, the results demonstrate that watermark laundering is a practical failure mode for victim watermarks: a single prompt submitted to a public black-box editor can disrupt invisible payloads while returning usable reconstructions. The OpenAI models provide the strongest disruption across schemes. GPT Image 1 is closest to random recovery on average, while GPT Image 2 improves fidelity and retains substantial disruption. The Nano Banana 2 results indicate that vulnerability in the transform domain can persist at substantially higher fidelity. The prompt ablations, conventional baselines, version comparisons, and HFR measurements are consistent with the reconstruction-bottleneck account, but they do not verify the internal causal pathways of proprietary editors. Watermark robustness is therefore not equivalent to provenance security. Future evaluations should include reconstruction with foundation models and prompts alongside classical perturbation, denoising, and generative reconstruction baselines. They should also report raw BER, interpret it by proximity to 0.5, and pair it with reference and semantic fidelity.

The evidence is limited to three watermark schemes, six evaluated editing models, a stratified design with 100 images, one prompt family, and a conventional baseline suite that does not include an additional diffusion reconstruction method. Formal confidence intervals and tests of effect size are also absent. We do not claim coverage of generative watermarks embedded within models, such as Stable Signature, Tree-Ring, or Gaussian Shading [36–38]. We also do not claim a bypass of provider policy, legal copyright transfer, or universal provenance substitution. Because model versions and interfaces evolve, the reported values constitute a temporal snapshot. The conclusion is limited to the evaluated interface: reconstruction accessed through prompts can decouple preservation of visible content from recovery of the victim payload. Watermark defenses and model safety designs should therefore evaluate that interface directly.

## Acknowledgments

This work was supported in part by Taishan Scholar under Grant tsqnz20250747; in part by the National Natural Science Foundation under Grant 62502250, Grant 62406051, Grant 62302249, Grant 62541206, and Grant 62272255; and in part by the Young Talent of Lifting Engineering for Science and Technology in Shandong under Grant SDAST2025QTB030.

## References

[1] Ingemar J. Cox, Joe Kilian, F. Thomson Leighton, and Talal Shamoon. Secure spread spectrum watermarking for multimedia. IEEE Transactions on Image Processing, 6(12):1673–1687, 1997.

[2] Frank Hartung and Martin Kutter. Multimedia watermarking techniques. Proceedings ofthe IEEE, 87(7):1079–1107, 1999.

[3] Ingemar J. Cox, Matthew L. Miller, Jeffrey A. Bloom, Jessica Fridrich, and Ton Kalker. Digital Watermarking and Steganography. Morgan Kaufmann, 2 edition, 2007.

[4] Adam Klein. Slipping racism into the mainstream: A theory of information laundering. Communication Theory, 22(4):427–448, 2012.

[5] David M. J. Lazer, Matthew A. Baum, Yochai Benkler, Adam J. Berinsky, Kelly M. Greenhill, Filippo Menczer, Miriam J. Metzger, Brendan Nyhan, Gordon Pennycook, David Rothschild, Michael Schudson, Steven A. Sloman, Cass R. Sunstein, Emily A. Thorson, Duncan J. Watts, and Jonathan L. Zittrain. The science of fake news. Science, 359(6380):1094–1096, 2018.

[6] Soroush Vosoughi, Deb Roy, and Sinan Aral. The spread of true and false news online. Science, 359(6380):1146–1151, 2018.

[7] Chengcheng Shao, Giovanni Luca Ciampaglia, Onur Varol, Kai-Cheng Yang, Alessandro Flammini, and Filippo Menczer. The spread of low-credibility content by social bots. Nature Communications, 9:4787, 2018.

[8] Kate Starbird. Examining the alternative media ecosystem through the production of alternative narratives of mass shooting events on twitter. In Proceedings ofthe International AAAI Conference on Web and Social Media (ICWSM), volume 11, pages 230–239, 2017.

[9] Coalition for Content Provenance and Authenticity. C2PA technical specification. https://spec.c2pa.org/specifications specifications/, 2026. Accessed 2026.

[10] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. HiDDeN: Hiding data with deep networks. In European Conference on Computer Vision (ECCV), pages 682–697, 2018.

[11] Matthew Tancik, Ben Mildenhall, and Ren Ng. Stegastamp: Invisible hyperlinks in physical photographs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2117–2126, 2020.

[12] Kevin Alex Zhang, Lei Xu, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Rivagan: Robust invisible video watermarking with attention, 2019.

[13] Zhaoyang Jia, Han Fang, and Weiming Zhang. MBRS: Enhancing robustness of DNN-based watermarking by mini-batch of real and simulated JPEG compression. In Proceedings ofthe 29th ACM International Conference on Multimedia (ACM MM), pages 41–49, 2021.

[14] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transform-domain collaborative filtering. IEEE Transactions on Image Processing, 16(8):2080–2095, 2007.

[15] Xuandong Zhao, Kexun Zhang, Zihao Su, Saastha Vasan, Ilya Grishchenko, Christopher Kruegel, Giovanni Vigna, Yu-Xiang Wang, and Lei Li. Invisible image watermarks are provably removable using generative ai. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[16] Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu. Image watermarks are removable using controllable regeneration from clean noise. In International Conference on Learning Representations (ICLR), 2025.

[17] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations (ICLR), 2022.

[18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In International Conference on Learning Representations (ICLR), 2023.

[19] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402, 2023.

[20] Pierre Moulin and Joseph A. O’Sullivan. Information-theoretic analysis of information hiding. IEEE Transactions on Information Theory, 49(3):563–593, 2003.

[21] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[23] Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley-Interscience, 2 edition, 2006.

[24] OpenAI. Introducing our latest image generation model in the api. https://openai.com/index/image-generation-api/, April 2025. Model identifier: gpt-image-1; accessed 2026.

[25] OpenAI. GPT Image 2 model. https://developers.openai.com/api/docs/models/gpt-image-2, April 2026. Model identifier: gpt-image-2; accessed 2026.

[26] Google. Introducing nano banana pro. https://blog.google/innovation-and-ai/products/nano-banana-pro/, November 2025. Gemini 3 Pro Image; accessed 2026.

[27] Google. Nano banana 2: Combining pro capabilities with lightning-fast speed. https://blog.google/innovation-and-ai/ technology/ai/nano-banana-2/, February 2026. Gemini 3.1 Flash Image; accessed 2026.

[28] Mauro Barni, Franco Bartolini, Vito Cappellini, and Alessandro Piva. A DCT-domain system for robust image watermarking. Signal Processing, 66(3):357–372, 1998.

[29] M. R. R. Kumari, V. V. Kumar, and K. R. Naidu. Digital image watermarking using DWT-SVD with enhanced tunicate swarm optimization algorithm. Multimedia Tools and Applications, 82(18):28259–28279, 2023.

[30] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), pages 740–755, 2014.

[31] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[32] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021.

[34] OpenAI. C2PA in chatgpt images. https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images, 2024. Accessed 2026.

[35] Google AI for Developers. Gemini api image generation documentation. https://ai.google.dev/gemini-api/docs/imagegeneration, 2026. Accessed 2026.

[36] Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[37] Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[38] Zijin Yang, Kai Zeng, Kejiang Chen, Han Fang, Weiming Zhang, and Nenghai Yu. Gaussian shading: Provable performancelossless image watermarking for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.