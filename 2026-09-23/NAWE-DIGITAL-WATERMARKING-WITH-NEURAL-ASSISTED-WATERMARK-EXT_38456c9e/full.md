# NAWE: DIGITAL WATERMARKING WITH NEURAL-ASSISTED WATERMARK EXTRACTION

Roman Chaban<sup>⋆</sup>, Vitaliy Kinakh<sup>⋆</sup>, Lilian Rouzaire<sup>†</sup> and Slava Voloshynovskiy<sup>⋆</sup>

<sup>⋆</sup>University of Geneva, Switzerland <sup>†</sup>Independent Researcher {roman.chaban, vitaliy.kinakh, svolos}@unige.ch, lilian.rouzaire@gmail.com

## ABSTRACT

NAWE (Neural-Assisted Watermark Extraction) combines an explicit signal-processing watermarking construction with a pretrained neural host predictor. A periodic, perceptually masked watermark carrier provides synchronization, Polar coding supplies redundancy, and denoising followed by subtraction extracts the embedded watermark. The denoiser remains frozen, without watermark-specific training. A one-factor-at-a-time study compares Wiener, BM3D, DRUNet, and GS-DRUNet host estimators. Comparisons with TrustMark, SSL Watermarking, PixelSeal, and WAM show NAWE’s lowest geometric and photometric class BER and strong message recovery, while filtering and noise remain limitations consistent with the non-adaptive selection of the watermark extractor. The comparison retains the systems’ different payloads and coding.

The code will be available at GitHub on paper acceptance.

Index Terms— Image watermarking, signal processing, synchronization, neural host suppression, Polar codes, robustness

## 1. INTRODUCTION

Invisible image watermarking must preserve visual quality while recovering a message after image processing and geometric distortion. Filtering and compression corrupt watermark evidence; cropping, resizing, and rotation also disturb carrier alignment. Robustness therefore requires synchronization, encoding, and suitable watermark extraction for efficient host-interference suppression.

Classical signal-processing designs include spread-spectrum modulation [1], perceptually masked transform-domain embedding [2], and stochastic content-adaptive allocation and host estimation [3]. Quantization index modulation [4], known-host data hiding [5], and improved spread spectrum [6] address host interference at embedding or detection. Geometric robustness has been pursued through transform invariants [7], registration templates [8], and repeated spatial patterns with autocorrelation-based acquisition [9]. Subsequent periodic and self-reference constructions combine this synchronization principle with multibit coding and adaptive embedding [10–12].

Learned systems extend this tradition through robust representations and distortion-aware embedder/extractor training [13–16]. NAWE instead uses a neural model for a specific receiver task: predicting the host so that subtraction extracts the watermark, thus providing the host interference cancellation. A coded, repeated spatial watermark carries out an encrypted message and supplies synchronization, and registered repetitions support soft decoding. The denoiser is pretrained and frozen; the watermarking system requires no task-specific network training.

Our contribution integrates explicit synchronization with replaceable neural host suppression, compares conventional and neural predictors, and evaluates bit and word error rates, and attack-family behavior.

## 2. RELATION TO LEARNED WATERMARKING

SSL Watermarking [13] freezes a self-supervised trained network and optimizes each image’s pixels by adversarial embedding: gradient-based updates make feature projections onto keyed directions match the signs specified by the message under a family of defined distortions. Extraction reads those signs without the host. The configurable message length $L _ { \mathrm { m s g } }$ counts direct bits without an outer channel code.

TrustMark [14] trains a residual embedder and a message extractor with image distortions; both operate feed-forward at deployment. The Q/BCH-5 interface uses 100 carrier bits: $L _ { \mathrm { m s g } } = 6 1 $ information bits, 35 BCH parity bits, and $L _ { \mathrm { s c h e m a } } = 4$ schema/reserved bits [17]. Channel decoding follows neural carrier extraction.

WAM [15] trains an embedder and an extractor that jointly localizes marked pixels and predicts a message of length $L _ { \mathrm { m s g } } = 3 2 $ bits at each location. Spatial aggregation recovers distinct messages from marked regions. Training includes localized embedding, splicing, and geometric/photometric augmentations.

PixelSeal [16] trains a feed-forward embedder/extractor with message length $L _ { \mathrm { m s g } } = 2 5 6$ bits, distortion augmentation, and highresolution adaptation. An adversarial discriminator supplies the imperceptibility objective instead of fixed perceptual losses, alongside the message-recovery objective. The adversarial optimization occurs during training.

NAWE embeds $L _ { \mathrm { m s g } }$ bits on the native image grid without learn able models and uses a frozen neural denoiser only at the extraction.

## 3. PERIODIC EMBEDDING AND BLIND EXTRACTION

## 3.1. Message coding and spatial construction

Figure 1 summarizes the signal path of NAWE. Let $\mathbf { x } \in [ 0 , 1 ] ^ { H \times W \times C }$ be the host and $\mathbf { b } _ { \mathrm { m s g } } \in \mathbf { \bar { \Gamma } } \{ 0 , \mathbf { \hat { \ l } } \} ^ { L }$ <sup>Lmsg</sup> an image-independent message, with $1 6 ~ \leq ~ L _ { \mathrm { m s g } } ~ \leq ~ 2 5 6$ bits. Appending cyclic redundancy check (CRC) bits of length $L _ { \mathrm { C R C } } = 1 6$ gives total length $L = L _ { \mathrm { m s g } } + L _ { \mathrm { C R C } }$ ; the CRC carries no user information (L<sub>CRC</sub> may be 0); it flags residual errors and steers list decoding (Sec. 3.3). The resulting word and the encoded codeword are

$$
\mathbf { u } = [ \mathbf { b } _ { \mathrm { m s g } } \Vert \mathbf { C } \mathrm { R C } _ { 1 6 } ( \mathbf { b } _ { \mathrm { m s g } } ) ] \in \{ 0 , 1 \} ^ { L } ,\tag{1}
$$

$$
\mathbf { c } = \mathrm { P o l a r E n c } _ { N , L } \left( \mathcal { E } _ { k _ { p } , \nu } ( \mathbf { u } ) \right) .\tag{2}
$$

Polar coding maps an L-bit input word to an N-bit codeword, where N is a power of two. The L input bits occupy positions corresponding to the most reliable polarized channels; the remaining $\bar { N } - L$ positions are fixed to known values, called frozen bits [18]. The code rate is $L / N$ . For example, choosing $L = 3 N / 4$ gives rate 3/4 while transmitting all N encoded bits, without puncturing (removing encoded bits).

![](images/4b69792f1d63f5fd49a034fb4906096a7ebba1b37b7af0a8cab1aa843fed140e.jpg)  
Fig. 1. NAWE signal path. Blue blocks perform analytical processing, green blocks carry image or message evidence, and orange denotes the frozen neural host predictor. The host determines the perceptual mask; the independent message and keyed synchronization bits determine the carrier. The receiver shares the key, nonce, layout, and decoder settings.

![](images/7dc2b020c63bace59f811c90d6398e2e320b21abb2f29546976b7835f9be7a86.jpg)  
Fig. 2. Composition of the 156-arm benchmark across 34 attack families. Inner sectors: geometry (44), photometric/color (26), compression/quantization (31), filtering/restoration/noise (25), and compound (30). Outer sectors show severity-arm counts.

The length-preserving encryption $\mathcal { E } _ { k _ { p } , \nu }$ uses the shared secret key $k _ { p }$ and a nonce ν, a public per-embedding value supplied to the receiver to diversify encryption.

The key/nonce pair $( k _ { p } , \nu )$ also generates the known pseudorandom synchronization vector $\mathbf { s } \in \{ 0 , 1 \} ^ { L _ { \mathbf { s } } }$ of length $L _ { \mathbf { s } }$ . These additional bits occupy dedicated cells alongside c; they are not user information or Polar parity bits. The receiver regenerates s to resolve orientation and translation, refine the affine estimate, and validate alignment (see Sec. 3.3). A shared layout maps the two bit sequences to a two-dimensional array, then bipolar modulation maps 0 to +1 and 1 to −1:

$$
\begin{array} { r } { { \bf w } _ { \mathrm { b l k } } = 1 - 2 \operatorname { L a y o u t } ( { \bf c } , { \bf s } ) . } \end{array}\tag{3}
$$

Expanding cells by integer scale s (distinct from s) and repeating the block produces the carrier w on the native image grid. Repetitions provide additional redundancy for the same word. The construction leaves $N , L _ { \mathrm { m s g } } , L _ { \mathrm { C R C } } , L _ { \mathbf { s } } ,$ , and s configurable.

## 3.2. Perceptual allocation and host suppression

The host determines a perceptual mask that is smaller in flat regions and larger in textured regions [3]:

$$
\mathcal { M } ( \mathbf { x } ) = \frac { \sigma ^ { 2 } + \gamma \beta ^ { 2 } } { \sigma ^ { 2 } + \beta ^ { 2 } } , \qquad \mathbf { m } = \frac { \mathcal { M } ( \mathbf { x } ) } { \mathrm { R M S } ( \mathcal { M } ( \mathbf { x } ) ) } .\tag{4}
$$

Here, $\sigma ^ { 2 }$ is the local image variance and $\gamma \in ( 0 , 1 ]$ is the flat-region weight before normalization. We set $\beta = 4 0$ when the variance is computed on the 8-bit scale, equivalently $\beta = 4 0 / 2 5 5$ on the [0, 1] scale. The mask m has unit root mean square (RMS), while α controls the overall embedding strength. Embedding is

$$
\begin{array} { r } { \mathbf { y } = Q \left( \mathrm { c l i p } _ { [ 0 , 1 ] } [ \mathbf { x } + \alpha \mathbf { m } \odot \mathbf { w } ] \right) . } \end{array}\tag{5}
$$

Here $Q$ denotes image quantization. The scalar α targets the embedding distortion of the saved image, with $\mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) \ =$ $- 1 0 \log _ { 1 0 } ( \Vert \mathbf { x } - \mathbf { y } \Vert _ { \mathrm { F } } ^ { 2 } / \mathrm { H W C } )$

For a received image z, the host estimate and watermark residual are

$$
\begin{array} { r } { \widehat { \mathbf { x } } = D _ { \theta _ { 0 } , \sigma } ( \mathbf { z } ) , \qquad \widehat { \mathbf { w } } = \mathbf { z } - \widehat { \mathbf { x } } . } \end{array}\tag{6}
$$

The denoiser has frozen weights $\theta _ { 0 }$ and noise-level setting σ. GS-DRUNet [19] is compared with DRUNet [20], BM3D [21], and Wiener filtering. Predictors are evaluated by downstream recovery.

## 3.3. Autocorrelation acquisition, aggregation, and decoding

The autocorrelation function (ACF) measures the similarity of the residual to its spatially shifted copies. It is computed by the Wiener– Khinchin relation

$$
R _ { \widehat { \mathbf { w } } } = { \mathcal { F } _ { 2 } ^ { - 1 } } \big ( | { \mathcal { F } _ { 2 } } \{ \widehat { \mathbf { w } } \} | ^ { 2 } \big ) .\tag{7}
$$

Here $\mathcal { F } _ { 2 }$ is the two-dimensional Fourier transform. The repeated carrier creates off-origin ACF peaks, following classical self-reference synchronization [9, 10, 12]. Under $\mathbf { r } \mapsto \mathbf { A r } + \mathbf { t }$ , their displacements depend on the four entries of $\mathbf { A } \in \mathbb { R } ^ { 2 \times 2 }$ , but not on the two translation components t. RANSAC fits candidate lattice bases and hence affine-matrix hypotheses. Inverse warping compensates the linear transform; folding and averaging registered tiles reinforces their common watermark.

The remaining orientation and translation are resolved by crosscorrelating the folded residual with the known bipolar synchronization cells 1−2s generated from the key/nonce pair $( k _ { p } , \nu )$ . Candidate cyclic shifts are ranked by this correlation, yielding translation in registered coordinates modulo the tile period. Correlation with the known synchronization cells also supports local refinement and validation of the affine hypotheses. The orientation/phase candidates are then decoded.

For candidate $j ,$ let $\ell _ { j }$ be the soft coded symbols after removal of synchronization cells. The receiver decodes and decrypts:

$$
\widehat { \mathbf { u } } _ { j } = \mathcal { E } _ { k _ { p } , \nu } ^ { - 1 } \left( \mathrm { P o l a r D e c } _ { N , L } ( \ell _ { j } ) \right) .\tag{8}
$$

![](images/60f1d02f416ef03e66a1ea1541da100851492348cd1554df60ba79384638dc66.jpg)  
Fig. 3. OFAT sensitivity as $\Delta \mathrm { B E R } = \mathrm { B E R } _ { \mathrm { a r m } } - \mathrm { B E R } _ { P _ { 0 } }$ (smaller is better). $\mathrm { B E R } _ { \mathrm { a r m } }$ is the BER of a construction with a single control parameter changed and $\mathrm { B E R } _ { P _ { 0 } }$ that of the reference $P _ { 0 }$ under the set of attacks that is comprised of 34 arms from 7 families (reduced form of Figure 2), among which are JPEG, Gaussian blur and noise, resize, rotation, crop and flip on 100 images. BER $_ { \cdot P _ { 0 } } = 0 . 0 7$ . Only one control parameter changes per row.

PolarDec is CRC-aided successive-cancellation list decoding, selecting the listed path whose CRC checks [22]. Writing $\widehat { \mathbf { u } } _ { j } \ =$ $[ \widehat { \mathbf { b } } _ { \mathrm { m s g } , j } \lVert \widehat { \mathbf { b } } _ { \mathrm { c r c } , j } ]$ , decoding stops and returns $\widehat { \mathbf { b } } _ { \mathrm { m s g } , j }$ when the recomputed and decoded CRC bits agree:

$$
\begin{array} { r } { \mathrm { C R C } _ { 1 6 } ( \widehat { \mathbf { b } } _ { \mathrm { m s g } , j } ) = \widehat { \mathbf { b } } _ { \mathrm { c r c } , j } . } \end{array}\tag{9}
$$

Candidates are tested until CRC agreement or budget J is exhausted. Blind extraction uses the key, nonce, layout, and decoder settings. It targets zero-bit-error recovery of u: every bit must match the transmitted word. At least one wrong bit or a decoding failure constitutes a word error. The word error rate (WER) is the fraction of trials with such an error, as defined in Sec. 4.1.

## 4. EVALUATION AND RESULTS

## 4.1. Comparison protocol and error rates

We compare native implementations of competing watermarking methods on identical images and attacks at matched embedding PSNR, retaining their payloads and available error correction. These are complete-system comparisons: equal PSNR does not equalize information rate, coding gain, or perceptual visibility. Images are COCO 2017 [23] resized to $5 1 2 ^ { 2 ^ { \cdot } }$ : 100 sources for the robustness benchmark and 100 for one-factor-at-a-time (OFAT) study. Each source draws an independent, domain-separated key and nonce, so key, nonce, and message never repeat.

For method q on arm a of attack family f, let ${ \bf b } _ { i } ^ { ( q ) }$ be a scored word of length $L _ { q }$ and $\widehat { \mathbf { b } } _ { i } ^ { ( q , f , a ) }$ its decoded output. With $\mathcal { T } _ { q , f , a }$ the set of returned words we define the bit error rate (BER) and word error rate (WER) as:

$$
\mathrm { B E R } _ { q , f , a } = \frac { \sum _ { i \in \mathcal { T } _ { q , f , a } } d _ { H } \big ( \widehat { \mathbf { b } } _ { i } ^ { ( q , f , a ) } , \mathbf { b } _ { i } ^ { ( q ) } \big ) } { | \mathcal { T } _ { q , f , a } | L _ { q } } ,\tag{10}
$$

$$
\mathrm { W E R } _ { q , f , a } = 1 - \frac { 1 } { T _ { f , a } } \sum _ { i \in \mathcal { T } _ { q , f , a } } \mathbf { 1 } \{ \widehat { \mathbf { b } } _ { i } ^ { ( q , f , a ) } = \mathbf { b } _ { i } ^ { ( q ) } \} .\tag{11}
$$

![](images/87703d466d0066ee30a77000941895419da59507107da28e0e10d2851e0d13c3.jpg)  
Fig. 4. Family-balanced WER versus BER over the robustness matrix in Figure 2, on logarithmic axes. Baseline curves sweep $\mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) \in [ 3 4 , 4 8 ]$ dB; NAWE markers show 48, 40, and 34 dB for protected-frame lengths $L \in \{ 3 2 , 6 4 , 2 5 6 \}$ , all at code rate $1 / 4$ Legend subscripts give the evaluated bit length.

Here $T _ { f , a }$ counts all trials and $d _ { H }$ is Hamming distance. BER is the fraction of wrong bits among returned words, a rate normalized by payload length, and is therefore comparable across different $L _ { q } .$ WER is the fraction of trials with at least one wrong bit or no decoded word. The indicator in (11) is one only when all $L _ { q }$ decoded bits match the original word. For NAWE, $L _ { q } = L = L _ { \mathrm { m s g } } + L _ { \mathrm { C R C } }$ includes message and CRC bits after decoding.

## 4.2. Parameter sensitivity and the host predictor

Figure 3 varies one control around the orange-diamond reference $P _ { 0 } \colon$ $L _ { \mathrm { m s g } } = 4 8 , L _ { \mathrm { C R C } } = 1 6$ and $L = L _ { \mathrm { m s g } } + L _ { \mathrm { C R C } } = 6 4$ is the Polar input length, which code rate $L / N = 1 / 4 , L _ { \mathbf { s } } = 6 8$ synchronization bits that occupy separate carrier cells and are excluded from Polar coding. Cell scale s = 2, GS-DRUNet at $\sigma = 1 8 , \mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) =$ 40 dB, and RGB image is $5 1 2 ^ { 2 }$ pixels. The $L _ { \mathbf { s } }$ keyed sync bits The panel reports $\Delta \mathrm { B E R } = \mathrm { B E R } _ { \mathrm { a r m } } - \mathrm { B E R } _ { P _ { 0 } }$ (smaller is better) against the reference $\mathrm { B E R } _ { P _ { 0 } } = 0 . 0 7$ , the operating point carried into the robustness benchmark (Figs. 2–5).

Two controls drive the largest reductions below $P _ { 0 } \colon$ the carrier scale s and the host suppressor $D _ { \theta _ { 0 } , \sigma }$ . A coarser cell $\left( s { = } 4 \right)$ and the neural denoiser push ∆BER most negative, while s=1 and $5 \times$ 5 Wiener are the worst arms (∆BER ≈ +0.08 and +0.07); GS-DRUNet, DRUNet (σ=10), and BM3D $( \sigma { = } 1 8 )$ lie within ≈ 0.01 of one another. The suppressor, the code rate $L / N { = } 1 / 4$ , and the sync length $L _ { \mathbf { s } } { = } 6 8$ are deliberate design choices near their best settings; too few sync cells $\left( L _ { \mathbf { s } } { = } 3 3 \right)$ ) or a higher rate cost accuracy at a redundancy trade we set once.

Embedding strength, image size, and payload are channel choices rather than design defaults, and they follow the expected monotone trend. At fixed code and cell geometry the averaged per-symbol SNR scales as $\mathrm { S N R } \propto \alpha ^ { 2 } H \bar { W } C / L _ { \mathrm { m s g } } ,$ , so stronger embedding (larger α, i.e. lower $\mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) )$ , a larger host $( H W C$ , more carrier repetitions), or a shorter payload each lower BER, roughly linearly across the swept range. The carrier is expanded by plain repetition on the native grid, so these operating points need no retraining, unlike the learned baselines, which require re-optimization for a new resolution or payload.

## 4.3. Benchmark coverage and distortion trade-off

Figure 2 groups 156 severity settings (arms) into 34 operation families $\mathcal { F }$ and five broad classes. Let $S _ { q , f , a }$ denote the selected endpoint (BER or WER) for method q on arm a of family $f ,$ whose arm set is

![](images/8472a34fef1fd66db8ca337614735d9c7e35e3a5c896a1c6a3ff04844715b7e5.jpg)  
Fig. 5. Per method BER at 40 dB across the 34 attack families from Figure 2 benchmark. Each marker is $\mu$ BER across all arms of a single attack from X axis and upward whisker is σ. Dashed horizontal lines represent $\mu$ of each distortion class. NAWE has the lowest geometric and photometric class averages.

$\boldsymbol { \mathcal { A } } _ { f }$ . We weight families equally:

$$
\overline { { S } } _ { q } = \frac { 1 } { \vert \mathcal { F } \vert } \sum _ { f \in \mathcal { F } } \frac { 1 } { \vert \mathcal { A } _ { f } \vert } \sum _ { a \in \mathcal { A } _ { f } } S _ { q , f , a } .\tag{12}
$$

Each family contributes $1 / | \mathcal { F } |$ regardless of its arm count, and its arms share that weight uniformly; the number of arms in a family therefore does not affect the aggregate. Class summaries apply the same rule to the families within a class. WAVES also emphasizes attack diversity [24]; geometric severity requires parameters beyond pixel-aligned PSNR.

Figure 4 plots WER against BER to evaluate the relation between zero-bit and multi-bit performance of NAWE and its competitors. Changing NAWE’s frame length L shifts its operating points along nearly linear log-log trend as expected when more coded symbols share a fixed distortion budget, without changing the construction or retraining a model. Across 34, 40, and 48 dB, NAWE has lower WER than every payload-matched baseline and competitive BER. At $\mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) < 4 0$ db NAWE starts to outperform other baselines in both WER and BER at competitive frame length $L .$ The most pronounced difference can be observed at $\mathrm { P S N R } ( \mathbf { x } , \mathbf { y } ) = 3 4$ db, where the competitors are saturated and cannot improve beyond theirs operational boundaries.

## 4.4. Attack-wise behavior and limits

Figure 5 compares the BER of NAWE and its competitive baselines using the benchmark composition in Figure 2.

Geometry: NAWE has the lowest class average, explained by the explicit synchronization mechanism. Flip and translation lie at the plotting floor, and rotation, shear, and resizing have low BER. SSL is stronger on some families, notably perspective: a general projective warp exceeds the acquisition model, which in a way can be tuned for perspective transforms as well.

Photometric and color: NAWE attains the lowest class summary and stays at the floor throughout this group, with ties on individual operations. Grayscale is particularly challenging for WAM. Altogether the learned baselines demonstrate inconcistency under somewhat trivial and not greatly destructive distortions.

Compression and quantization: NAWE remains competitive, especially for chroma JPEG, chroma subsampling, and palette quantization. Standard JPEG, posterization, and WebP produce more errors, and the other systems can achieve lower BER.

Filtering, restoration, and noise: NAWE underperforms the competing methods in this class, particularly TrustMark and WAM. Its off-the-shelf denoiser was pretrained for additive white Gaussian noise. Highly destructive attacks that primarily target high frequencies, like denoise and blurs, are excellent watermark suppressors. Nevertheless, severe blur removes commercially useful detail and resulting attacked image has comparatively low SSIM and PSNR. This training-task mismatch limits the current design; training or adapting the host predictor for extraction under these distortions is a future research direction.

Compound: The compound arms chain several operations to emulate real distribution pipelines, such as messaging and social re-encoding, display capture, and print. NAWE’s behavior on them tracks its per-class results: it stays robust on chains dominated by geometric and photometric steps, but degrades where a filtering or noise stage is present, inheriting the limitation above.

## 5. CONCLUSION

NAWE couples an explicit signal-processing construction and a periodic, perceptually masked carrier with autocorrelation-based synchronization and Polar coding – to a frozen neural denoiser used only for host suppression at extraction. This design keeps the receiver blind and free of watermark-specific training while a neural predictor still removes host interference. Across a broad multi-family benchmark NAWE is competitive overall and leads the geometric and photometric classes, and it attains the most reliable whole-word recovery among the compared systems, with a bit/word trade-off that stays predictable by design. Its weaknesses concentrate where the pretrained denoiser is mismatched to the attack, $\mathrm { i . e . , - s t r o n g }$ filtering, restoration, and noise, – rather than in the synchronization or coding stages.

Because the design is synchronization-based, modular, and not learned end to end, each stage can be improved in isolation without retraining the others, so the same construction can be pushed further toward any given attack, transmission channel, or payload regime. We see three directions. First, replacing or fine-tuning the host predictor for extraction under filtering, blur, and noise directly targets the main limitation while leaving the rest of the pipeline untouched. Second, extending the acquisition model from affine to general projective warps would close the remaining geometric gap. Third, since the carrier expands by plain repetition, its scale, code rate, and synchronization budget can be retuned per channel for a specific compression, display, or distribution channel trading capacity for robustness on demand.

## 6. REFERENCES

[1] Ingemar J. Cox, Joe Kilian, F. Thomson Leighton, and Talal Shamoon, “Secure spread spectrum watermarking for multimedia,” IEEE Transactions on Image Processing, vol. 6, no. 12, pp. 1673–1687, 1997.

[2] Mauro Barni, Franco Bartolini, Vito Cappellini, and Alessandro Piva, “A DCT-domain system for robust image watermarking,” Signal Processing, vol. 66, no. 3, pp. 357–372, 1998.

[3] Sviatoslav Voloshynovskiy, Alexander Herrigel, Nazanin Baumgaertner, and Thierry Pun, “A stochastic approach to content adaptive digital image watermarking,” in Information Hiding. 2000, vol. 1768 of Lecture Notes in Computer Science, pp. 211–236, Springer.

[4] Brian Chen and Gregory W. Wornell, “Quantization index modulation: A class of provably good methods for digital watermarking and information embedding,” IEEE Transactions on Information Theory, vol. 47, no. 4, pp. 1423–1443, 2001.

[5] Fernando Pérez-González, Félix Balado, and Juan R. Hernández, “Performance analysis of existing and new methods for data hiding with known-host information in additive channels,” IEEE Transactions on Signal Processing, vol. 51, no. 4, pp. 960–980, 2003.

[6] Henrique S. Malvar and D. A. F. Florencio, “Improved spread spectrum: A new modulation technique for robust watermarking,” IEEE Transactions on Signal Processing, vol. 51, no. 4, pp. 898–905, 2003.

[7] Joseph J. K. Ó Ruanaidh and Thierry Pun, “Rotation, scale and translation invariant spread spectrum digital image watermarking,” Signal Processing, vol. 66, no. 3, pp. 303–317, 1998.

[8] Shelby Pereira and Thierry Pun, “Robust template matching for affine resistant image watermarks,” IEEE Transactions on Image Processing, vol. 9, no. 6, pp. 1123–1129, 2000.

[9] Martin Kutter, “Watermarking resistance to translation, rotation, and scaling,” in Multimedia Systems and Applications, 1999, vol. 3528 of Proceedings ofSPIE, pp. 423–431.

[10] Sviatoslav Voloshynovskiy, Frédéric Deguillaume, and Thierry Pun, “Multibit digital watermarking robust against local nonlinear geometrical distortions,” in IEEE International Conference on Image Processing, 2001, vol. 3, pp. 999–1002.

[11] Sviatoslav Voloshynovskiy, Frédéric Deguillaume, Shelby Pereira, and Thierry Pun, “Optimal adaptive diversity watermarking with channel state estimation,” in Security and Watermarking of Multimedia Contents III, 2001, vol. 4314 of Proceedings ofSPIE.

[12] Frédéric Deguillaume, Sviatoslav Voloshynovskiy, and Thierry Pun, “Secure hybrid robust watermarking resistant against tampering and copy attack,” Signal Processing, vol. 83, no. 10, pp. 2133–2170, 2003.

[13] Pierre Fernandez, Alexandre Sablayrolles, Teddy Furon, Hervé Jégou, and Matthijs Douze, “Watermarking images in selfsupervised latent spaces,” in 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 3054–3058.

[14] Tu Bui, Shruti Agarwal, and John Collomosse, “TrustMark: Robust watermarking and watermark removal for arbitrary resolution images,” in 2025 IEEE/CVF International Conference on Computer Vision, 2025, pp. 18629–18639.

[15] Tom Sander, Pierre Fernandez, Alain Durmus, Teddy Furon, and Matthijs Douze, “Watermark anything with localized messages,” in The Thirteenth International Conference on Learning Representations, 2025.

[16] Tomáš Soucek, Pierre Fernandez, Hady Elsahar, Sylvestre-ˇ Alvise Rebuffi, Valeriu Lacatusu, Tuan Tran, Tom Sander, and Alexandre Mourachko, “Pixel Seal: Adversarial-only training for invisible image and video watermarking,” arXiv preprint arXiv:2512.16874, 2025.

[17] Adobe, “TrustMark: Universal watermarking for arbitrary resolution images,” https://github.com/adobe/trustmark, 2024.

[18] Erdal Arikan, “Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels,” IEEE Transactions on Information Theory, vol. 55, no. 7, pp. 3051–3073, 2009.

[19] Samuel Hurault, Arthur Leclaire, and Nicolas Papadakis, “Gradient step denoiser for convergent plug-and-play,” in International Conference on Learning Representations, 2022.

[20] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte, “Plug-and-play image restoration with deep denoiser prior,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11962–11972.

[21] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian, “Image denoising by sparse 3-D transformdomain collaborative filtering,” IEEE Transactions on Image Processing, vol. 16, no. 8, pp. 2080–2095, 2007.

[22] Ido Tal and Alexander Vardy, “List decoding of polar codes,” IEEE Transactions on Information Theory, vol. 61, no. 5, pp. 2213–2226, 2015.

[23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick, “Microsoft COCO: Common objects in context,” in European Conference on Computer Vision (ECCV), 2014, pp. 740–755.

[24] Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, and Furong Huang, “WAVES: Benchmarking the robustness of image watermarks,” in Proceedings ofthe 41st International Conference on Machine Learning, 2024, vol. 235 of Proceedings of Machine Learning Research, pp. 1456–1492.