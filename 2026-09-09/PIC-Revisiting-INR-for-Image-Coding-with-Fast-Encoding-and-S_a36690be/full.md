# PIC: Revisiting INR for Image Coding with Fast Encoding and Sub-Millisecond Decoding

Xiang Liu1,3\*0, Jinxiang Wang1\*, Bin Chen2\*\*0, Zimo Liu30, Mingyao Hong3, Jiawei Li40, Yaowei Wang2,30, and Shu-tao Xia1,30

1 Tsinghua University, China

2 Harbin Institute of Technology, Shenzhen, China 3 Peng Cheng Laboratory, China 4 Joy Future Academy, JD Group, China

Abstract. Implicit neural representation (INR) has achieved remarkable progress in novel view synthesis and image/video coding in recent years. Compared to conventional end-to-end image codecs, INR-based compressors demonstrate significant advantages in decoding complexity. However, their practical application has been hindered by the inferior encoding speed and underutilized decoding efficiency. In this work, we propose a feedforward INR image coding architecture, Practical INR Image Codec (PIC), that computes all the necessary information for INR network in a single forward pass, achieving an encoding speed of 20 FPS. Additionally, we implement a highly optimized decoder that reaches 2000 FPS decoding speed, significantly surpassing JPEG's performance at comparable rate-distortion (RD) performance. To the best of our knowledge, this work presents the first learning-based image codec that simultaneously outperforms or is comparable with JPEG in both RD performance and decoding speed while maintaining practical encoding speed. Code is available at https://github.com/actcwlf/PIC.

Keywords: Image Compression · Implicit Neural Representation · Feed Forward Network

## 1 Introduction

Image compression has long been a fundamental topic in signal processing and remains a critical technology underpinning more complex systems such as video coding. With the rapid expansion of Internet data, industrial data, AIGC data, and other forms of data, compression technology remains a critically important research area. Image compression technology has undergone significant evolution, encompassing traditional encoders such as JPEG [48], end-to-end models [1, 2], and recently emerged compression methods utilizing implicit neural representation (INR) [12, 13, 25] or Gaussian Splatting (GS) [21, 27, 54]. Each of these approaches demonstrates distinct strengths and limitations in different aspects of image compression tasks. Traditional image encoders typically leverage human understanding of signals and visual perception, employing predefined rules to discard visually insignificant information for compression. For instance, JPEG exploits the human eye's reduced sensitivity to certain highfrequency components by decomposing the image into different frequencies using Discrete Cosine Transform (DCT) and selectively retaining the most perceptually critical parts, thereby achieving highly efficient compression. Additionally, its codec design strikes a balance between rate-distortion (RD) performance and encoding/decoding speed. Subsequent traditional encoders further improved RD performance by adopting more sophisticated transformation techniques, such as wavelet transforms [42] and predictive coding [4], although at the cost of increased computational complexity and slower processing speeds.

Classic end-to-end image compression algorithms [1,2] conceptually follow the transform coding paradigm used in traditional image codecs. The key difference is that traditional compressors incorporate numerous manually designed transforming rules, whereas the end-to-end approach employs data-driven method to learn the transformation. Taking advantage of the powerful learning capability of neural networks, this data-driven nonlinear transformation can effectively model the image distribution prior to the data, thereby achieving RD performance that surpasses traditional image codecs [16, 20]. Furthermore, significant progress has been made based on generative models [34], particularly in low bit-rate compression scenarios oriented to perceptual metrics [51].

More recently, representation-based methods—such as INRs [12] and 3D Gaussian Splatting [21]—have emerged as promising alternatives. Their key advantage lies in extremely low decoding complexity [25, 31], often achieving orders-of-magnitude speedups over neural codecs [54]. Some variants also rival traditional codecs in rate-distortion performance [22]. However, these gains come at the cost of extremely slow encoding, often requiring minutes or hours to train a model per image, severely limiting real-world deployment.

In this paper, we propose a new image compression paradigm, Practical INR Image Codec (PIC), which bridges the gap between these paradigms. PIC directly produces a low-bit-rate neural representation in a single forward pass through an end-to-end trained network, combining fast encoding, ultra-fast decoding, and competitive rate-distortion performance. Tab. 1 demonstrates the main differences among three paradigms.

The primary contributions of this work are summarized below:

\- We propose a novel image coding paradigm PIC that directly generates low-bit-rate neural representation models through neural networks, simultaneously achieving fast encoding, ultra-fast decoding, and comparative RD performance.

\- We have implemented an optimized decoder that fully translates the lowcomplexity characteristics of neural representation models into practical high decoding speeds.

Table 1: Comparision of different paradigm. E2E and RM are abbreviation for end-toend and representation model respectively. Feedforward means the method is able to encode an image in one forward pass, instead of a full training process. RD represents rate distortion performance. We selected three representative methods, Factorized [1], Hyperprior [2], COIN [12], Cool-Chic v4.2 (fast) [22,25,26] and GaussianImage [54], for comparison. BD-Rate are calcualted relative to JPEG on Kodak dataset. Other results are representative value from the same experiment. Due to the lack of a fair FLOPs calculation method, the complexity of GaussianImage is left blank. Detailed numerical results of all metrics are shown in Fig. 3 and Fig. 4.
<table><tr><td colspan="2">Model</td><td colspan="4">Paradigm Feedforward BD-Rate↓ Enc.[ms]↓ Dec.[ms]↓ FLOPs/pixels↓</td></tr><tr><td>Factorized</td><td>E2E</td><td>√</td><td>-47.34%</td><td>24.3</td><td>35.7 42.175K</td></tr><tr><td>Hyperprior</td><td>E2E</td><td>√</td><td>-57.58%</td><td>141 169</td><td>45.546K</td></tr><tr><td>COIN</td><td>RM</td><td></td><td>25.53%  $1 . 4 0 \times { { 1 0 } ^ { 6 } }$ </td><td>3.18</td><td>29.057K</td></tr><tr><td>Cool-Chic v4.2</td><td>RM</td><td></td><td>-64.86%  $1 . 7 7 \times 1 0 ^ { 5 }$ </td><td>216</td><td>1.303K</td></tr><tr><td>GaussianImage</td><td>RM</td><td></td><td>36.55%  $6 . 9 9 \times 1 0 ^ { 5 }$ </td><td>0.439</td><td></td></tr><tr><td>PIC (Our)</td><td>E2E RM</td><td>√</td><td>-12.78%</td><td>28.6 0.418</td><td>1.074K</td></tr></table>

Through comprehensive experiments, we validate the performance of our proposed method, demonstrating significant improvements in both RD performance and encoding/decoding speed compared to prior representationbased image coding approaches. Notably, our method surpasses nvJPEG in terms of decoding speed.

## 2 Related Work

## 2.1 End-to-end Image Compression

Classic end-to-end image compression extends transform coding paradigm, which use neural network as both analysis transform and synthesis transform [1]. An important feature that distinguishes compression models from other models is the integer symbol constraints in entropy coding, which introduces nondifferentiable quantization in training [14]. Ballé et al. [1] pioneered a method to jointly optimize reconstruction loss and bit-rate constraints by maximizing the Evidence Lower Bound (ELBO) and formalizes the compression model optimization task as

$$
\mathcal { L } _ { \phi _ { g } , \theta _ { g } } = \lambda R + D ( g _ { s } ( Q ( g _ { a } ( \pmb { x } ; \phi _ { g } ) ) ; \theta _ { g } ) , \pmb { x } ) ,\tag{1}
$$

where $g _ { a } ( \cdot ; \phi _ { g } )$ and $g _ { s } ( \cdot ; \theta _ { g } )$ are analysis transformation and synthesis transformation respectively. $Q ( \cdot )$ is quantization operation. To achieve better task performance, several previous works have also explored many differentiable approximation of quantization [1, 14]. R is estimated bit rate. λ balances the reconstruction quality and bit-rate.

This fundamental architecture has been extensively developed in subsequent research. One important approach is to explore more efficient network architectures. Ballé et al. [2] introduced scale hyperpriors to better model latent distribution. More work investigate auto-regressive structure [17, 36, 37] or Transformer model [29,55]. Generative models represent another important category. Mentzer et al. [34] leveraged GAN architectures for high-fidelity reconstruction. With the rise of diffusion models, more research has begun exploring compression in extremely low bit-rate scenarios [51].

Although end-to-end methods are limited in practical applications due to computational constraints, their single forward encoding capability offers distinct advantages over INR training-based encoding, and provides valuable inspiration for exploring similar INR encoding schemes.

## 2.2 Representation-based Image Compression

These representation-based methods can be further categorized into several types, with the most prominent paradigm being the direct mapping of positional coordinates to target spaces. For instance, NeRF [35] maps ray angles to density and color along the ray, which has significantly impacted the fields of volume rendering and novel view synthesis, bringing widespread attention to INR technology [3]. Subsequently, this paradigm has expanded to other signal representation domains, including neural rendering [44], image representation, video representation, and further into image and video compression [8, 12].

Another approach involves using learnable parameters or grid as inputs instead of directly utilizing coordinates. Within the NeRF series, Instant-NGP [38] is a representative work that employs a multi-resolution spatial hash grid to store these learnable parameters. This same concept can be extended to other neural representation tasks. There are many workss have made remarkable progress in representing and compressing textures [45], BRDF [11], etc. Similarly, the COOL-CHIC [25] and its successors [22] have demonstrated impressive performance in image compression. This methodology also finds broad applications in video compression [24].

The 3D Gaussian Splatting (3DGS) [21] provides a different perspective of representation models. By directly organizing learnable parameters through specific structures, without relying on per-instance overfitted neural networks, 3DGS has demonstrated unique advantages in novel view synthesis tasks. Compared with INR, this representation method often exhibit more interpretable structure, providing unique advantages beyond quantity performance. Similarly, such methods have been successfully applied to tasks including 3D scene representation [19, 32], image compression [54], and video compression [30].

This class of methods has demonstrated unique advantages in compression, such as low decoding complexity, fast decoding speed, and high reconstruction quality. However, these strengths are difficult to achieve simultaneously in a single model. Moreover, the reliance on model training for encoding hinders their practical deployment. In this work, our proposed method explores how to balance multiple metrics to construct a practical encoder.

## 2.3 Hypernetwork for INR

Previous work has explored the concept of hypernetworks [15,23], which dynamically modulate neural network weights during inference. This idea can be traced back to even earlier mechanisms such as Fast Weight Programmers (FWP) [40] and several studies have also investigated its theoretical connections to linear Transformers [39]. Existing research on hypernetworks has predominantly focused on tasks such as continual learning |47], few-shot learning |41| and domain adaptation [46]. In the field of INR related research, the implementation of generalizable INR generation itself can be formulated as a hypernetwork model. This differs to some extent from the task of generalizable 3DGS generation [10]: 3DGS is, to a certain degree, an explicit representation model, whose distribution characteristics are closely coupled with the final target information. In contrast, the network parameter space of an INR lacks such a correlation with the signal it encodes, which poses a critical challenge for the design of generalizable INR frameworks. In terms of specific algorithm design, hypernetwork-based INR approaches have been explored in various tasks including content generation [43, 52], yet their investigation in compression tasks remains insufficient [7].

This paper addresses this gap by further exploring a generalizable/feedforward INR image encoder based on the hypernetwork mechanism. In particular, targeting the lack of research that simultaneously considers RD performance and coding efficiency, we attempt to design a codec that comprehensively considers performance across all dimensions through this approach, offering new insights into the field of learned image codecs.

## 3 Method

The primary reason for the slow encoding of representation models lies in the fact that the encoding process itself is the training process. Consequently, methods based on implicit neural representations—as well as similar approaches like Gaussian splatting—can only be applied in scenarios where encoding time is highly insensitive. A straightforward idea is to directly generate this representation model itself through a neural network. The proposed PIC follows the idea. Fig. 1 demonstrates the overall architecture of our method. Sec. 3.1 shows the details of transforming a representation to compression models. Sec. 3.2 introduces entropy estimation module. Sec. 3.3 describes the full pipeline and implementation details.

## 3.1 Repurposing Representation Models for Compression

To better introduce the proposed method, we begin with the image representation model $g _ { r }$ in Fig. 1. For a $H \times W$ image, Î is a set of pyramid-like multiresolution latents

$$
\begin{array} { r } { \pmb { \hat { y } } = \{ \hat { y } _ { i } \in \mathbb { R } ^ { H _ { i } \times W _ { i } } , i = 1 , 2 , \ldots , L \} , } \end{array}\tag{2}
$$

![](images/af1f2654e4da78369d67dfabf69bac42e649790d4d7ba52be12e2e9041d6d3c1.jpg)  
Fig. 1: The framework of PIC. The overall data flow is presented in an $\mathrm { S } \mathrm { - }$ shaped in the diagram, following $g _ { a }  \hat { { y } }  z  \hat { z }  g _ { r }  g _ { s }$ · ga and $g _ { m }$ are latent encoder and modulation net respectively. $\hat { \pmb { y } } = \{ y _ { 1 } , y _ { 2 } , . . . y _ { L } \}$ are latents generated by latent encoder $g _ { a }$ . All latents are divided into patches of size $8 \times 8$ and then concatenated together as z. Similar to other compression methods, z is quantized after DCT transformation. The ZpR module converts the quantized symbols into a more compact form, and we will discuss this module in detail in the Sec. 3.3. HE and HD are Huffman encoder and decoder. iZpR and iDCT are inverse transformation of DCT and ZpR respectively. $g _ { \tau }$ is part of the decoder and also acts as an image representation model. $g _ { s }$ is synthesis network. $\gamma$ and $\beta$ are mean and standard deviation of each channel of $g _ { m }$ output respectively.

where $\begin{array} { r } { H _ { i } = \frac { H } { 2 ^ { L - i } } , W _ { i } = \frac { W } { 2 ^ { L - i } } } \end{array}$ . L is the number of latents. To match the resolution of the output image, $\hat { \pmb { y } }$ is upsamlped to $\hat { \pmb { y } } _ { u } \in \mathbb { R } ^ { C \times H \times W }$ before being fed into the reconstruction network $g _ { s }$ . If we apply $g _ { r }$ in image representation task, image information will be stored in the weights of the network. All parameters, including $\hat { \boldsymbol y }$ and weights in $g _ { s }$ , are trainable and optimized jointly via gradient descent.

Obviously, achieving a fast encoding process through training is highly challenging. Therefore, our approach generates all weights of the representation model in a single step. Since the latents inherently lies in a space similar to the image domain, designing a simple yet functional encoder $g _ { a }$ is relatively straightforward, as shown in Fig. 1. The primary challenge is generating the network weights of $g _ { s }$ . While previous works have explored methods for network weight generation, their performance in compression tasks still leaves room for improvement [9]. In this paper, we adopt a simple strategy called channel-wise normalization and modulation.

We first divide $g _ { s }$ into shared part $\mathrm { M L P } ^ { s }$ and instance-dependent part ${ \mathrm { M L P } } ^ { i }$ Suppose N is batch size and $\pmb { f } \in \mathbb { R } ^ { N \times C \times H \times W }$ is the intermediate feature between ${ \mathrm { M L P } } ^ { s }$ and $\mathrm { M L P } ^ { i }$ , the instance-normalized feature is

$$
\bar { \pmb f } = \frac { \pmb f - \mathrm { m e a n } ( \pmb f ) } { \mathrm { s t d } ( \pmb f ) } .\tag{3}
$$

Then modulate the normalized feature $\bar { \pmb f }$

$$
\tilde { \pmb { f } } = \gamma \odot \bar { \pmb { f } } + \beta ,\tag{4}
$$

where $\gamma \in \mathbb { R } ^ { N \times C }$ and $\beta \in \mathbb { R } ^ { N \times C }$ are generated by modulation net $g _ { m }$ . Fraction and $\odot$ represent element-wise division and multiplication at C dimension respectively. Note $g _ { m }$ will generate corresponding γ and $\beta$ for each input x. One notable advantage of this transformation is that both the normalization and modulation parameters can ultimately be fused into the network weights of ${ \mathrm { M L P } } ^ { i }$ , thereby achieving the goal of generating corresponding network weights for each input. For detailed derivation, please refer to the Supplementary.

## 3.2 Entropy Estimation

In neural representation models, constrained by the overall model size, it is challenging to incorporate a large entropy estimation network. One solution is to employ a compact neural network to enhance RD performance via autoregressive methods [25]. However, the decoding speed of such approaches remains limited by their auto-regressive design. Inspired by Luo et al. [33], we estimate the distribution of DCT parameters to achieve rate estimation accordingly.

For latents generated by $g _ { a }$

$$
\pmb { y } = \{ y _ { i } \in \mathbb { R } ^ { H _ { i } \times W _ { i } } , i = 1 , 2 , \ldots , L \} ,\tag{5}
$$

we divide them into patches of size $8 \times 8$ , concatenate together as $z \in \mathbb { R } ^ { n \times 8 \times 8 }$ and transform to symbols through DCT

$$
z = { \mathrm { P a t c h i f y } } ( y _ { 1 } , \dots , y _ { L } ) ,\tag{6}
$$

$$
s = \operatorname { D C T } ( z ) .\tag{7}
$$

Let $s _ { i , k }$ is the k-th DCT component $( k \in \{ 0 , \ldots , 6 3 \} )$ of i-th block, the estimated entropy is

$$
\mathcal { L } _ { \mathrm { e n t r o p y } } = - \sum _ { i = 1 } ^ { n } \sum _ { k = 0 } ^ { 6 3 } \log _ { 2 } p _ { \theta _ { k } } ( s _ { i , k } + u ) , u \sim \mathrm { U n i f o r m } ( - 0 . 5 , 0 . 5 ) .\tag{8}
$$

![](images/8620175990c0df1783eb600aced1eb1b40dd2f9971918900907dcd87ce3b5828.jpg)  
Fig. 2: The pipeline of $\mathrm { Z p R }$ module. ZpR transform quantized DCT coefficients to compact symbols through three steps: zigzag reorder, trim trailing zeros and encode zeros using run length encode (partial RLE).

We use 64 channels entropy model, which means we model the distribution of DCT components independently. $\theta _ { k }$ is the corresponding parameters of the piecewise linear function for the k-th channel in entropy model [1, 6]. Similar to other compression framework , the entire pipeline is non-differentiable after quantization of s , so we use a simple uniform noise relaxation [1] to build an end-to-end trainable pipeline.

## 3.3 Full Pipeline and Hardware-affinity Implementation

JPEG reduces bit-rate through lossless compression techniques like the zigzag transform and run-length encoding (RLE). In our PIC, we similarly employ Zigzag reordering and partial Run-length encoding (ZpR) module to improve RD performance. As shown in Fig. 2, ZpR reduce zeros in zigzag reordered symbols. The main difference between ZpR and RLE in JPEG is we only reduce zeros in symbols. This strategy is more straightforward to implement while effectively reducing the bit-rate.

Unlike JPEG, our method employs image-specific Huffman coding tables, meaning the storage overhead of the Huffman tables also impacts the final bitrate performance. To ensure O(1) complexity for decoding one symbol, the coding table includs $2 ^ { p }$ entries (p is precision) which covers all possible bitstream pattern for decoding one symbol. For small pictures like those in Kodak dataset, the overhead of storing the code tables is non-negligible. Fortunately, the actual number of symbols used in encoding is typically fewer than 128, allowing for further compression of the code tables to reduce this overhead. Benefiting from the prefix code property of Huffman coding, we can apply RLE to compress the original code table, reducing the number of table entries to match the symbol count and significantly decreasing the storage overhead of the original code table.

As illustrated in Fig. 1, we integrated all of the above modules to achieve an trainable codec. The final loss function is

$$
\mathcal { L } = \lambda \mathcal { L } _ { \mathrm { e n t r o p y } } + D ( x , \hat { x } ) ,\tag{9}
$$

where D is distortion metric, λ balances the trade-off between reconstruction quality and bit-rate.

Another important topic is how to implement an optimized decoder. JPEG's concise design and long-term optimizations have made it one of the fastest encoders. However, due to its early introduction, JPEG has not fully leveraged the hardware advancements in recent years. Benefiting from recent research on parallel Huffman decoding [50] and the emergence of hardware features like Tensor Cores, it has become possible to develop a decoder that surpasses JPEG in performance.

In architecture design, we avoided using relatively large synthesis network like COOL-CHIC-family [22, 25] to comply with Tensor Core's matrix structure requirements. Each layer in our reconstruction network $g _ { s }$ maintains a width of 16, precisely matching the warp matrix multiplication of TF32 operations in Tensor Cores. This alignment maximizes hardware utilization and accelerates reconstruction speed. It should be noted that, although our current implementation relies on NVIDIA's Tensor Cores, other hardware also supports similar types of functionalities, such as AMD's rocWMMA. We believe such acceleration hardware will eventually be supported by more manufacturers' devices in the future as well.

Furthermore, since the precision of TF32 is lower than standard FP32, a gap would arise if training were conducted with FP32 while inference/decoding relied on Tensor Core-accelerated operations. To address this, we draw inspiration from neural rendering techniques and implement the training-phase code using the differentiable shading language Slang [18], which significantly reduces the complexity of integrating inline CUDA code in differentiable framework. This approach ensures consistency between training and inference while maintaining computational efficiency during training.

Qualitatively speaking, our decoder's complexity is actually higher than JPEG's. However, by fully leveraging hardware capabilities, our decoder has surpassed commercial decoders and provided new insights for the design of image codecs.

## 4 Experiments

## 4.1 Experiment Setup

Dataset. Since the proposed PIC is a generalizable method, it is necessary to train our model on a large dataset like end-to-end methods. We use LSDIR dataset [28] as training set, which includes 84,991 natural images. During training, these images were randomly cropped to a resolution of $5 1 2 \times 5 1 2$ . Additionally, Our evaluation is conducted on two popular datasets, $\mathrm { K o d a k } ^ { 5 }$ and CLIC⁶. Kodak dataset includes 24 images of size $7 6 8 \times 5 1 2$ . The CLIC dataset contains 41 high-resolution natural images.

Metrics. We use the peak signal-to-noise ratio (PSNR) in RGB 4:4:4 as distortion metric, which are the most widely used metric in compression. We also report and MS-SSIM [49] LPIPS [53] in main experiment, which is a popular perceptual metric to measure realism. Bit-per-pixel (BPP) is used as coding efficiency metric. To demonstrates the efficiency of PIC, we report encoding, decoding time and complexity as well.

![](images/88c570ed7c25a0768daab514d18d338b6427ce9f0dca50b752959d59b3804496.jpg)

![](images/883497a8b0b44193f8e64a2995ebe020893fd3d2ab0516ad951598cc3c551d85.jpg)  
(b) Speed and complexity comparison of all methods on Kodak dataset.  
Fig. 3: Results on Kodak dataset.

Implementation Details. Our complete framework is implemented in Py-Torch, with the key distinction that the synthesis network used in training is built upon Slang-torch 7 to simplify Tensor Core programming within the $\mathrm { P y } -$ Torch environment. The $\mathrm { Z p R }$ module and decoder are CUDA-accelerated, while the Huffman codec is adapted and modified from GPUHD [50]. Other auxiliary components such as bitstream $\mathrm { I / O }$ operations are implemented in C++. All experiments are performed on a single NVidia RTX 4090D. We jointly optimize the full set of parameters in $g _ { a } , g _ { m }$ and $g _ { s } . g _ { s }$ is a 4 layers MLP with ReLU activation. The final bitstream includes header information, Huffman code table, encoded latents, and parameters of ${ \mathrm { M L P } } ^ { i }$ in FP32 format.

Hyper-parameters settings. For the proposed PIC, we use total $L = 8$ latents. This means that the minimum size of images we can encode is $2 5 6 \times 2 5 6$ This also suggests that encoding smaller images requires reducing the value of L. During the training, we set batch size to 16 and randomly cropped the input image to a resolution of $5 1 2 \times 5 1 2$ . This optimization is performed separately for each $\lambda = \{ 0 . 0 1 , 0 . 0 0 5 , 0 . 0 0 2 , 0 . 0 0 1 \}$ using Adam optimizer and learning rate of 1e-4. All convolution layers have 16 channels. We trained our model for 10 epochs in all experiments. Mean squard error (MSE) is used as distortion metric in training.

![](images/662bc9892ae2651abe11c89938f92acd097a5f06e7183911b216941ab88d04ec.jpg)

![](images/409e486c10774b844c20b9b76dc3cb4ce6d4cf732386592621a08e455b9ddada.jpg)  
(b) Speed and complexity comparison of all methods on CLIC dataset.  
Fig. 4: Results on CLIC dataset.

Benchmarks. Our method is benchmarked against competitive representation based methods like COIN [12] GaussianImage [54] and GaussianImage++ [27]. We also compares with Cool-Chic v4.2 (fast) [22, 25, 26] which is the stateof-the-art method in representation based image codec. We reproduce all the results using the original source code. Another baselines include pretrained Factorized model [1] and Hyperprior [2] in CompressAI [6] and nvJPEG8, which is a CUDA-accelerated JPEG codec. We also include HEVC in comparison. Note we use MSE as distortion metrics in loss function for PIC in all experiments. For other methods, we follow their original loss function settings, even if they used a loss function different from MSE.

## 4.2 Quantity and Quality Results

Fig. 3 and Fig. 4 shows the main results of different methods on Kodak dataset and CLIC dataset repectively. Fig. 3a and Fig. 4a demonstrate the RD performance of all methods. All results are averaged on same corresponding hyperparameters setting for each method. PIC achieves better PSNR at high bit-rate region and comparable MS-SSIM and LPIPS performance with JPEG. PIC also outperforms other representation-based methods with similar decoding speed in all quality metrics at high bit rate region.

![](images/2247f050fe4698c77e1fb069d3f869936dd6748c0b605491345fe036edb7c075.jpg)  
Fig. 5: Visualizations on kodim17.png from Kodak dataset and juskteez-vu-1041.png from CLIC dataset.

Fig. 3b and and Fig. 4b present comprehensive comparisons of encoding/decoding speeds across all methods. PIC achieves comparable encoding speed to Factorized model [1] and Hyperprior model [2] but significantly faster decoding speed and lower decoding complexity. Compared with other representationbased approaches [12, 25, 54], PIC has an acceleration of exceeding three orders of magnitude. In terms of decoding speed, our approach ranks second only to GaussianImage [54] while outperforming all other alternatives.

This not only demonstrates the superiority of our proposed method, but also highlights the unique advantages of grid-based approaches in reconstruction quality. Our experiments further validate the performance advantages of COIN [12] and GaussianImage [54] in the low bit-rate region, while these methods exhibit weaker quality improvement compared to other approaches as the bit-rate increases.

![](images/f9935ffa4de9094e9cf89249897a8dabc13ccefa3adf50b63c47712b517944a2.jpg)  
(a) Ablation of synthesis net gs architecture. All components are necessary for performance.

![](images/ec1146c62342be4855f9b57ae3ddabec7dfc529fbe851db31d7859e046ef884b.jpg)  
(b) Ablation of entropy model. ρ is Pearson correlation coefficient.  
Fig. 6: Ablation results.

Fig. 5 is the visualization of all methods including JPEG [48], Factorized model [1], COIN [12], GaussianImage (abbreviated as GI) [54] and proposed PIC. Below each image are the corresponding PSNR/bpp results. On juskteezvu-1041.png, we select the model setting with best quality in original paper for COIN and GaussianImage. Our method demonstrates marginally superior performance compared to JPEG while significantly outperforming COIN and GaussianImage. Our approach also achieves comparable results to JPEG in terms of texture details and artifacts, which can be largely attributed to similar entropy modeling. In contrast, the Factorized model tends to produce overly smoothed outputs. COIN exhibits swirling artifacts, and the GaussiaImage displays Gaussian-like speckled artifacts.

Overall, although there is still a certain gap between our method and stateof-the-art approaches in terms of RD performance, PIC has surpassed the widely used JPEG and is a practically deployable approach. For other learning-based methods, the reality is that current architectures, whether based on end-to-end autoencoders or the overfitted Cool-Chic architecture, are constrained by their inherent limitations, making them difficult to deploy in general scenarios. Even the most lightweight Factorized model [1] is hampered by its high computational overhead. For Cool-chic-like models, their excessively slow encoding and decoding speeds render them impractical for most application. Meanwhile, the proposed PIC exhibits no significant weaknesses across key practical metrics. This offers a new perspective for future research on learning-based codec in realworld scenarios.

## 4.3 Ablation Study

Fig. 6a presents the architectural ablation results of the synthesis network $g _ { s } .$ "Shared $\operatorname { s y n } ^ { \dag }$ denotes using the same synthesis net $g _ { s }$ for all images, in which case our method degenerates into an end-to-end model. Evidently, the model performance under this configuration proves significantly inferior due to the limited capacity of decoder network. $^ { 6 0 } \mathrm { W / o }$ normalization" and $^ { 6 6 } \mathrm { W / o }$ modulation" respectively indicate the exclusion of normalization and modulation in $g _ { s } .$ Unsurprisingly, such configurations significantly degrade model performance. Only when incorporating all modules does our method achieve the desired performance.

![](images/964edc8a9d3a7092eff2ae2bb94aabce2fb9911909de831c81d83097112bf681.jpg)  
(a) Decoding speed for different width of synthesis network

![](images/34b2b8b58765167cc055f175061fad5fbe050588e172249bdc873d76bb8b57c8.jpg)  
(b) Detailed breakdown of decoding time  
Fig. 7: Ablation results of decoding speed.

Due to the inherent approximation nature of entropy networks, discrepancies with actual performance are unavoidable. Furthermore, post-processing in the $\mathrm { Z p R }$ module may potentially amplify these deviations. Fig. 6b illustrates the differences between the actual bit-rate and the estimated bit-rate. Although some discrepancies exist, overall consistency has been maintained. This finding aligns with prior studies [33]. Developing more accurate entropy estimation models remains an important direction for future improvements.

Fig. 7a is the ablation results of decoding speed for different width of synthesis network. Since the minimum matrix size supported by Tensor Cores for the TF32 data type is currently 16, a $1 6 \times 1 6$ matrix padded with zeros is used when the width is 8. Under this configuration, there is no significant difference in decoding speed between a width of 8 and a width of 16. Overall, the configuration with a width of 16 is the optimal choice, taking into account decoding speed, RD performance, and hardware compatibility. Fig. 7b is detailed breakdown of decoding time including host-to-device I/O time.

We further analyze the latent representation structure to better understand the model's performance. The detailed design of the pRLE module and the structural design of the synthesis network are also investigated. The visulization of latents and more results are included in the Supplementary.

## 5 Conclusion

In this work, we introduced PIC, an end-to-end INR image coding framework that generates all network parameters in a single forward pass, leading to an encoder substantially faster than prior representation-based approaches. We further designed a highly optimized decoder that outperforms JPEG in speed while delivering comparable rate-distortion performance. Together, these advances establish a new balance between efficiency and quality, setting a milestone for INR-based compression. While there is still headroom for improving RD performance, our method opens a novel paradigm for practical learning-based image codecs and provides a foundation for extending INR-based compression to other modalities.

## Acknowledgements

This work is supported in part by the National Science and Technology Major Project under grant 2025ZD1601300, National Natural Science Foundation of China under grant 62301189, 62576122,62571298, Guangdong Basic and Applied Basic Research Foundation under grant 2026A1515011139.

## References

1. Ballé, J., Laparra, V., Simoncelli, E.P.: End-to-end optimized image compression. In: International Conference on Learning Representations (2017)

2. Ballé, J., Minnen, D., Singh, S., Hwang, S.J., Johnston, N.: Variational image compression with a scale hyperprior. In: International Conference on Learning Representations (2018)

3. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5470–5479 (2022)

4. Bellard, F.: Bpg image format (2018), https://bellard.org/bpg/

5. Blard, T., Ladune, T., Philippe, P., Clare, G., Jiang, X., Déforges, O.: Overfitted image coding at reduced complexity. In: 2024 32nd European Signal Processing Conference (EUSIPCO). pp. 927–931. IEEE (2024)

6. Bégaint, J., Racapé, F., Feltman, S., Pushparaja, A.: Compressai: a pytorch library and evaluation platform for end-to-end compression research (2020)

7. Catania, L., Allegra, D.: Nif: A fast implicit image compression with bottleneck layers and modulated sinusoidal activations. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 9022–9031 (2023)

8. Chen, H., He, B., Wang, H., Ren, Y., Lim, S.N., Shrivastava, A.: Nerv: Neural representations for videos. Advances in Neural Information Processing Systems 34, 21557–21568 (2021)

9. Chen, H., Xie, S., Lim, S.N., Shrivastava, A.: Fast encoding and decoding for implicit video representation. In: European Conference on Computer Vision. pp. 402–418. Springer (2024)

10. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: European conference on computer vision. pp. 370–386. Springer (2024)

11. Dou, Y., Zheng, Z., Jin, Q., Ni, B., Chen, Y., Ke, J.: Real-time neural brdf with spherically distributed primitives. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4337–4346 (2024)

12. Dupont, E., Goliński, A., Alizadeh, M., Teh, Y.W., Doucet, A.: Coin: Compression with implicit neural representations. arXiv preprint arXiv:2103.03123 (2021)

13. Dupont, E., Loya, H., Alizadeh, M., Golinski, A., Teh, Y.W., Doucet, A.: Coin++: Data agnostic neural compression. arXiv preprint arXiv:2201.12904 1(2), 4 (2022)

14. Guo, Z., Zhang, Z., Feng, R., Chen, Z.: Soft then hard: Rethinking the quantization in neural image compression. In: International Conference on Machine Learning. pp. 3920–3929. PMLR (2021)

15. Ha, D., Dai, A., Le, Q.V.: Hypernetworks. arXiv e-prints pp. arXiv–1609 (2016)

16. He, D., Yang, Z., Peng, W., Ma, R., Qin, H., Wang, Y.: Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5718–5727 (2022)

17. He, D., Yang, Z., Peng, W., Ma, R., Qin, H., Wang, Y.: Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding. arXiv e-prints (2022)

18. He, Y., Fatahalian, K., Foley, T.: Slang: language mechanisms for extensible realtime shading systems. ACM Transactions on Graphics (TOG) 37(4), 1–13 (2018)

19. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: ACM SIGGRAPH 2024 conference papers. pp. 1–11 (2024)

20. Jiang, W., Yang, J., Zhai, Y., Ning, P., Gao, F., Wang, R.: Mlic: Multi-reference entropy model for learned image compression. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 7618–7627 (2023)

21. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)

22. Kim, H., Bauer, M., Theis, L., Schwarz, J.R., Dupont, E.: C3: High-performance and low-complexity neural compression from a single image or video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9347–9358 (2024)

23. Klocek, S., Maziarka, Ł., Wołczyk, M., Tabor, J., Nowak, J., Śmieja, M.: Hypernetwork functional image representation. In: International Conference on Artificial Neural Networks. pp. 496–510. Springer (2019)

24. Kwan, H.M., Gao, G., Zhang, F., Gower, A., Bull, D.: Hinerv: Video compression with hierarchical encoding-based neural representation. Advances in Neural Information Processing Systems 36, 72692–72704 (2023)

25. Ladune, T., Philippe, P., Henry, F., Clare, G., Leguay, T.: Cool-chic: Coordinatebased low complexity hierarchical image codec. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13515–13522 (2023)

26. Leguay, T., Ladune, T., Philippe, P., Clare, G., Henry, F.: Low-complexity overfitted neural image codec. arXiv preprint arXiv:2307.12706 (2023)

27. Li, T., Zhang, X., Ge, X., Xu, T., He, D., Zhang, J., Wang, Y.: Gaussianimage++: Boosted image representation and compression with 2d gaussian splatting. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 6442–6449 (2026)

28. Li, Y., Zhang, K., Liang, J., Cao, J., Liu, C., Gong, R., Zhang, Y., Tang, H., Liu, Y., Demandolx, D., et al.: Lsdir: A large scale dataset for image restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1775–1787 (2023)

29. Liu, J., Sun, H., Katto, J.: Learned image compression with mixed transformer-cnn architectures. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1–10 (2023)

30. Liu, X., Chen, B., Liu, Z., Wang, Y., Xia, S.T.: An exploration with entropy constrained 3d gaussians for 2d video compression. In: The Thirteenth International Conference on Learning Representations (2025)

31. Liu, X., Chen, J., Chen, B., Liu, Z., An, B., Xia, S.T., Wang, Z.: An efficient implicit neural representation image codec based on mixed autoregressive model for low-complexity decoding. arXiv preprint arXiv:2401.12587 (2024)

32. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20654–20664 (2024)

33. Luo, X., Talebi, H., Yang, F., Elad, M., Milanfar, P.: The rate-distortion-accuracy tradeoff: Jpeg case study. arXiv preprint arXiv:2008.00605 (2020)

34. Mentzer, F., Toderici, G.D., Tschannen, M., Agustsson, E.: High-fidelity generative image compression. Advances in neural information processing systems 33, 11913– 11924 (2020)

35. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis (2020)

36. Minnen, D., Ballé, J., Toderici, G.: Joint autoregressive and hierarchical priors for learned image compression (2018)

37. Minnen, D., Singh, S.: Channel-wise autoregressive entropy models for learned image compression (2020)

38. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022)

39. Schlag, I., Irie, K., Schmidhuber, J.: Linear transformers are secretly fast weight programmers. In: International conference on machine learning. pp. 9355–9366 PMLR (2021)

40. Schlag, I., Schmidhuber, J.: Gated fast weights for on-the-fly neural program generation. In: NIPS Metalearning Workshop (2017)

41. Sendera, M., Przewięźlikowski, M., Karanowski, K., Zięba, M., Tabor, J., Spurek, P.: Hypershot: Few-shot learning by kernel hypernetworks. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 2469–2478 (2023)

42. Skodras, A., Christopoulos, C., Ebrahimi, T.: The jpeg 2000 still image compression standard. IEEE Signal processing magazine 18(5), 36–58 (2001)

43. Skorokhodov, I., Ignatyev, S., Elhoseiny, M.: Adversarial generation of continuous images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10753–10764 (2021)

44. Sztrajman, A., Rainer, G., Ritschel, T., Weyrich, T.: Neural brdf representation and importance sampling. In: Computer Graphics Forum. vol. 40, pp. 332–346. Wiley Online Library (2021)

45. Vaidyanathan, K., Salvi, M., Wronski, B., Akenine-Möller, T., Ebelin, P., Lefohn, A.: Random-access neural compression of material textures. ACM Transactions on Graphics (2023)

46. Volk, T., Ben-David, E., Amosy, O., Chechik, G., Reichart, R.: Example-based hypernetworks for out-of-distribution generalization. arXiv preprint arXiv:2203.14276 (2022)

47. Von Oswald, J., Henning, C., Grewe, B.F., Sacramento, J.: Continual learning with hypernetworks. arXiv preprint arXiv:1906.00695 (2019)

48. Wallace, G.K.: The jpeg still picture compression standard. IEEE transactions on consumer electronics 38(1), xviii-xxxiv (1992)

49. Wang, Z., Simoncelli, E.P., Bovik, A.C.: Multiscale structural similarity for image quality assessment. In: The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003. vol. 2, pp. 1398–1402. Ieee (2003)

50. Weißenberger, A., Schmidt, B.: Massively parallel huffman decoding on gpus. In: Proceedings of the 47th International Conference on Parallel Processing. pp. 1-10 (2018)

51. Xia, Y., Zhou, Y., Wang, J., An, B., Wang, H., Wang, Y., Chen, B.: Diffpc: Diffusion-based high perceptual fidelity image compression with semantic refinement. In: The Thirteenth International Conference on Learning Representations (2025)

52. You, T., Kim, M., Kim, J., Han, B.: Generative neural fields by mixtures of neural implicit functions. Advances in Neural Information Processing Systems 36, 20352– 20370 (2023)

53. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

54. Zhang, X., Ge, X., Xu, T., He, D., Wang, Y., Qin, H., Lu, G., Geng, J., Zhang, J.: Gaussianimage: 1000 fps image representation and compression by 2d gaussian splatting. arXiv preprint arXiv:2403.08551 (2024)

55. Zou, R., Song, C., Zhang, Z.: The devil is in the details: Window-based attention for image compression. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17492–17501 (2022)

## A Fusing Modulations to Synthesis Net

In the encoding process, both the normalization and modulation parameters are fused into the parameters of ${ \mathrm { M L P } } ^ { i }$ . The modulated network parameters subsequently included as part of the bitstream. Suppose $\pmb { f } \in \mathbb { R } ^ { C \times \bar { H } \times W }$ is the output feature of ${ \mathrm { M L P } } ^ { s }$

$$
\begin{array} { r } { \pmb { f } = \mathrm { M L P } ^ { s } ( \hat { \pmb y } _ { u } ) , } \end{array}\tag{10}
$$

where $\hat { \mathbf { y } } _ { u }$ is upsampled latents shown in $\mathrm { F i g }$ 1 in main text. Note we omit batch dimensions for clarity. The modulated features is

$$
\bar { \pmb { f } } = \frac { \pmb { f } - \mu } { \sigma } ,\tag{11}
$$

$$
\tilde { \pmb { f } } = \gamma \odot \bar { \pmb { f } } + \beta ,\tag{12}
$$

where $\mu = \operatorname { m e a n } ( f ) , \sigma = \operatorname { s t d } ( f )$ . Here we normalize f at $H$ and $W$ channel. Fraction represents element-wise division at $C$ dimension. $\odot$ is element-wise multiplication at $C$ dimension. $\gamma \in \mathbb { R } ^ { C }$ and $\beta \in \mathbb { R } ^ { C }$ are modulation vectors. The first layer of $\mathrm { M L P } ^ { i }$ can be expanded as

$$
\pmb { f } ^ { \prime } = \pmb { w } ^ { T } \tilde { \pmb { f } } + b\tag{13}
$$

$$
\overline { { \mathbf { \psi } } } = w ^ { T } ( \gamma \odot \bar { \mathbf { f } } ) + w ^ { T } \beta + b\tag{14}
$$

$$
= \frac { w ^ { T } ( \gamma \odot f ) } { \sigma } - \frac { w ^ { T } ( \gamma \odot \mu ) } { \sigma } + w ^ { T } \beta + b .\tag{15}
$$

The net weight and bias included in bitstream are

$$
w ^ { \prime } = \frac { \gamma ^ { T } \odot w } { \sigma ^ { T } } ,\tag{16}
$$

$$
b ^ { \prime } = - \frac { w ^ { T } ( \gamma \odot \mu ) } { \sigma } + w ^ { T } \beta + b .\tag{17}
$$

In decoding phase, only $w ^ { \prime }$ and $b ^ { \prime }$ are required, with no need for normalization and modulation. In our experiments, we use first 3 layer as ${ \mathrm { M L P } } ^ { s }$ and last layer as ${ \mathrm { M L P } } ^ { i }$

## B More Implementation Detatils

## B.1 Fast Chunking Algorithms

Algo. 1 is a fast chunking algorithm that splits the symbol stream based on end-of-block markers (eob\_marker) and expands the zero symbols according to the pRLE code table. This function requires precomputing the offsets of all eob\_marker to enable parallel processing. We use thrust::copy\_if to efficiently obtain all offsets. For synthesis network $g _ { s } .$ , we fuse all layers in a single CUDA kernel to achieve efficient processing. All matrix multiplications in $g _ { s }$ are implemented using APIs provided by nvcuda::wmma.

## B.2 Details of Baseline Methods

For COIN [12] and GaussianImage [54], We reproduce all the results using the original source code. It should be noted that although our method only used MSE as the distortion metric during the training, we still adhered to the original loss function settings for the baseline methods, even if the original approaches employed loss functions other than simple MSE. Additionally, for all methods, we did not use model versions specifically trained for different test metrics. For instance, in the case of Factorized model [1], all experiments utilized a pre-trained version optimized with MSE as the loss function.

## B.3 Evaluation Protocol

Due to the fact that speed evaluation is susceptible to interference from various factors, including hardware occupancy and PyTorch's inherent asynchronous design, we have carefully designed the speed evaluation process and ensured exclusive access to the hardware during assessment. Although there are currently many implementations of JPEG, in order to achieve a fair comparision, it is necessary to use a GPU-optimized version. However, since nvNVJPE is not opensource, we implemented our binding based on previous implemntation 9. We have also meticulously optimized the input and output components of nvJPEG to minimize any additional overhead as much as possible. Besieds, we run encoding and decoding in same process without saving bitstream to disk for nvJPEG, which is intended to eliminate the I/O time. In experiment, we find running encoding and decoding in same process is sufficient to warmup nvJPEG. For all method, we report the result of serial decoding all images in corresponding dataset, following is part of benchmark code for nvJPEG

Algorithm 1 Expand Symbols   
1: procedure EXPAND SYMBOLS(symbols, offsets, output)   
2: offset start ← 0   
3: if gid ≠ 0 then gid is CUDA thread id   
4: offset start ← offsets[gid-1] + 1   
5: end if   
6: offset end ← offsets[gid]   
7: eob marker ← symbols[offset end]  Get eob marker from symbol stream   
8: output symbols ← output + gid × 64   
9: offset ← 0   
10: for i ← offset start to offset end-1 do   
11: s ← symbols[i]   
12: if s ≥ eob marker − 17 + 1 then  If s is a symbol in the pRLE code   
table   
13: for j ← 0 to s + 17− eob marker −1 do   
14: output symbols[ZIGZAG ORDER[offset]] ← 0   
15: offset ← offset + 1   
16: end for   
17: else   
18: output\_symbols[ZIGZAG\_ORDER[offset]] ← symbols[i]   
19: offset ← offset + 1   
20: end if   
21: end for   
22: end procedure

## C More Experiment Results

## C.1 Extended Results

Fig. 8 uses BD-rate as the vertical axis to compare the encoding/decoding speed and decoding complexity of different methods. While our method does not achieve optimal performance across all dimensions, it demonstrates balanced capabilities with no significant weaknesses in the three metrics. In contrast, autoencoder methods [1, 2] offer fast encoding capabilities but suffer from high complexity, whereas Cool-chic [25, 26] maintains low complexity at the cost of significantly slower encoding and decoding speeds.

Listing 1.1: Snippet used to mesure nvJPEG decoding speed.

```python
1 for idx in range(len(bs_pack)):
2 jpeg_bytes, img, f = bs_pack[idx]# encoded bitstream
3 torch.cuda.synchronize()
4 start_time = time.time()
5 img_decoded = coder.decode(jpeg_bytes) # run nvJPEG
decoder
6 torch.cuda.synchronize()
7 end_time = time.time()
8 total += end_time - start_time
9 print('decoding time', total / len(bs_pack))
```

![](images/8432ee5b9ef3c1f9a02107010cd5d0a4f84eb98b6bbf614edea6ddb5ca741d96.jpg)

![](images/52811e30383b8d42e2c9f40c12643dba551d70a54c2eaca1092c4720b2ae1c6f.jpg)

![](images/e6e5d132164062a20b4c7f47ec6d188c6c3a15c19a959f3c83b1143e6c250aea.jpg)  
Fig. 8: Results of comprehensive comparesion for both RD performance and practical performance on Kodak dataset.

## C.2 Latents Visualizations

To better demonstrate the model's performance under different bitrates, we visualized the latent representations when $\lambda = \{ 0 . 0 1 , 0 . 0 0 1 \}$ . Fig. 9 shows the visualizations without normalization. Overall, the latents at different resolutions reflect varying levels of detail from the original image. Fig. 10 presents the results after applying same normalization to all latents. Although these latents do not directly indicate the magnitude of the corresponding symbols, they indirectly reveal that at smaller λ, higher-resolution latents retain more details, while at bigger λ, more details are contained in relatively lower-resolution latents. This also reflects the bits allocation tendencies under different settings.

## C.3 Ablation study of model architecture

Given implementation complexity considerations, we opted for a simplified RLE. Because our symbol range is determined by training, we cannot fix an RLE encoding table like JPEG, but dynamically determine it through the transformed symbol stream. Tab. 2 presents a performance comparison of performing RLE encoding on more symbols (-1 and 1). From the results in the table, it can be seen that performing RLE only on 0 is not only easier to implement in engineering, but can also achieve better results in certain situations.

![](images/80d162704fba3fd717066e43495a6a08400ad01aac09db3e2eb320a9cc82c7da.jpg)

(a) λ = 0.001  
![](images/e100d6650102dc817b8bf34d64de2ffe06c03f3663ab9cd26c192196457508a4.jpg)  
(b) λ = 0.01  
Fig. 9: Visualizations of latents. These figures demonstrate unnormalized value for each latent.

Table 2: Ablation result of RLE. RLE on 0 achieves the best performance among four settings. BD-Rate is calculated relative to RLE on 0.
<table><tr><td>Setting</td><td>BD-Rate(↓)</td></tr><tr><td>No RLE</td><td>150.68%</td></tr><tr><td>RLE on 0</td><td>0%</td></tr><tr><td>RLE on 0, 1</td><td>1.18%</td></tr><tr><td>RLE on -1, 0, 1</td><td>2.70%</td></tr></table>

Another ablation study explored the position of MLPi. We investigated placing the MLPi at the input layer, intermediate layer, and output layer of the synthesis network, respectively. Tab. 3 presents the result. We found placing MLP at the end of synthesis network will obtain the best performance.

Furthermore, the width of the synthesis network is also a parameter that needs to be examined. Tab. 4 displays the performance under different widths. Overall, the configuration with a width of 16 is the optimal choice, taking into account decoding speed, RD performance, and hardware compatibility.

## C.4 Ablation of modulation

In addition to exploring the effect of the modulation mechanism on the model architecture proposed in this paper, we also validated the effectiveness of this mechanism on model architectures similar to N-O Cool-Chic. Fig. 11 illustrates the results of the ablation results. Since the last two layers of the N-O Cool-Chic's synthesis network are convolutional layers, we applied modulation to the features between the first two linear layers. The results show the effectiveness of the modulation mechanism in this architecture as well. Moreover, N-O Cool-Chic\* with modulation has surpassed the method proposed in this paper in terms of RD performance, highlighting the potential of the end-to-end INR paradigm for further advancements in its network architecture and training recipts.

![](images/df6027fee9b731bdaf8e4bef798ae96b3c6c850d493da5119b5e52197a453844.jpg)  
(a) $\lambda = 0 . 0 0 1$

![](images/c4f4de15da0277e204e26e0d04c9fa07b9b84b76a0d0744c988641ce50d9fc16.jpg)  
(b) $\lambda = 0 . 0 1$  
Fig. 10: Visualizations of latents. These figures demonstrate normalized value for each latent.

Table 3: Start means we place ${ \mathrm { M L P } } ^ { i }$ at the beginning of the synthesis network. Middle means we place an ${ \mathrm { M L P } } ^ { i }$ between two shared MLPs. BD-Rate is calculated relative to default configuration, in which ${ \mathrm { M L P } } ^ { i }$ are placed at the end of synthesis network.
<table><tr><td>Location</td><td>BD-Rate (↓)</td></tr><tr><td>Start</td><td>21.46%</td></tr><tr><td>Middle</td><td>2.43%</td></tr><tr><td>End (default for PIC)</td><td>0%</td></tr></table>

## C.5 Ablation of bitstream

The Fig. 12 shows the proportion of different components in the encoded bitstream on the Kodak dataset. The encoded latent variables account for the vast majority of the bitstream, and the codebook size increases with the bitstream but has little impact on the overall encoded size.

Table 4: Ablation result of the synthesis network width. BD-Rate is calculated relative to width 16.
<table><tr><td>Width BD-Rate(↓)</td></tr><tr><td>8 3.03%</td></tr><tr><td>16 0%</td></tr><tr><td>32 5.84%</td></tr></table>

![](images/8bbfad820edd04d2edf21c4ca1a80a7e64aa3e0cde9348ab1cf9f32a51ae1420.jpg)  
Fig. 11: Ablation results of modulation mechanism on N-O Cool-Chic-like (mark by \*) architecture [5]. It should also be noted that N-O Cool-Chic is not open-source, so the results presented here are only a preliminary reproduction without hyperparameters and training recipts tuning and may differ from those reported in the original paper.

![](images/19028916f12854c5b33e22d7982709589ad6609f36ae22f62d3bdc1b7c69e8f3.jpg)  
Fig. 12: Bitstream breakdown.