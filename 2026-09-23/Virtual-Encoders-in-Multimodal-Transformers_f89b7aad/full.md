# Virtual Encoders in Multimodal Transformers

Katsuya Ogata<sup>1</sup> and Yuta Nakashima<sup>1</sup>

The University of Osaka

{ogata, n-yuta}@im.sanken.osaka-u.ac.jp

Abstract. Multimodal language models traditionally rely on dedicated perceptual encoders to construct task-usable representations. More integrated architectures have recently emerged, which instead expose the shared transformer to lightly projected patches, audio frames, or discrete visual tokens. Where does this encoding happen when such representations are not provided? We find that the transformer can internalize this missing computation, constructing task-usable perceptual representations within its own early-to-middle layers before the downstream language model. We call this computational structure a Virtual Encoder. Across linear probing, similarities to perceptual encoders, and causal analyses, we identify signatures of this structure in models that receive perceptual tokens without continuous encoder-derived features. These analyses also suggest that the boundary between perception and language processing need not coincide within an architectural module. Instead, encoder-like computation can emerge as a functional regime within a shared transformer, providing a new perspective for understanding where and how multimodal models process perception.

Keywords: Multimodal transformers · Encoder-free MLLM · Subspaces

## 1 Introduction

The development of multimodal large language models (MLLMs) has been progressively bringing perception and language processing closer together. We organize current designs into three architectural families (Figure 2). Encoder-full MLLMs use a dedicated vision or audio encoder to construct continuous perceptual features before passing them to a language model through cross-attention, a resampler, or a projector [2,4,12,21,24]. Discrete-token MLLMs represent perceptual inputs as discrete codes in the same autoregressive stream as text, as in Chameleon and Emu3 [8, 37]. Encoder-free MLLMs omit a dedicated perceptual encoder and feed lightly projected image patches or audio frames into the shared Transformer, as in Fuyu, EVE, SOLO, and Gemma 4 [6,9,14,17]. Unlike encoder-full MLLMs, these two families thus shift multimodal perception from a dedicated external encoder into a part of the transformer-based language model itself.

This raises an intriguing question: How do perception and language processing coexist in a single Transformer-based model? Perceptual encoding and language processing may require qualitatively diferent computations. Moreover, although encoder-free and discrete-token MLLMs allow perceptual and language tokens to interact from the earliest layers, meaningful cross-modal interaction may require higher-level representations to be established first. That is, architectural unification does not necessarily imply computational unification. Perceptual encoding may be organized as a distinct functional stage within the shared Transformer. Although architectural boundaries disappear, deeper analysis of such a model may identify the functional boundaries that replace perceptual encoders.

![](images/8b917aa9849cde83a1097b7dfb6b672a053f4841b25c27a56196e412583b758a.jpg)  
Fig. 1: Overview. When a multimodal transformer receives perceptual tokens that have not already been organized by a continuous modality encoder, it must construct a useful representation internally. In Gemma 4 12B, an early band of layers, which we call the Virtual Encoder, performs this encoding for both modalities up to a mid-depth readout boundary. There the vision representation aligns with the model’s language subspace, whereas the audio representation, though formed just as early, is not maintained downstream.

We hypothesize that discrete-token and encoder-free MLLMs internalize part of the perceptual encoding normally performed by a dedicated encoder. Specifically, an early band of Transformer layers organizes perceptual tokens into a task-usable representation and thereby assumes an encoder-like computational role. We call this functional regime a Virtual Encoder (Figure 1).

Based on this hypothesis, we study encoder-free and discrete-token MLLMs and compare them with encoder-full MLLMs through linear probing [1], representational similarity [20], and causal analyses with perturbations over image and audio inputs. We set Gemma 4 12B [17] as our primary target model and consistently identify an encoder-like structure in it, forming the perceptual representation in early-to-middle layers, which are then read out.

Contributions. We introduce the Virtual Encoder, a functional regime in which a shared Transformer internally constructs perceptual representations when no continuous encoder-derived features are supplied. We show four properties. (1) Encoder-free and discrete-token MLLMs develop semantic decodability progressively over depth, whereas encoder-full MLLMs start with already organized representations. (2) Their early hidden states exhibit encoder-like geometry, and in Gemma 4 12B these states are directly sensitive to low-level perceptual structure. (3) The resulting representations are causally read out at architecture-specific depths. (4) Their subsequent routing is modality- and architecture-dependent, with vision in Gemma 4 12B transiently aligning with the language subspace near readout while audio follows a markedly diferent trajectory.

![](images/579660fcbed9389d3dbe30a7c41143233379e6dd06a107d3304615aee3f9d67a.jpg)  
Fig. 2: Three MLLM families distinguished by the representation supplied to the multimodal Transformer. Encoder-full MLLMs receive continuous features from a dedicated perceptual encoder. Discrete-token MLLMs receive discrete perceptual codes, while encoder-free MLLMs receive lightly projected patches or frames. We ask whether the latter two families form a Virtual Encoder inside the Transformer.

## 2 Related Work

## 2.1 Multimodal Large Language Models

Early MLLMs extended large language models with dedicated perceptual encoders and lightweight bridging modules. CLIP [32] and SigLIP [39] provided pretrained visual representations, while models such as Flamingo [2], BLIP-2 [21], LLaVA [24], and InstructBLIP [12] connected such features to language models through cross-attention, Q-Former-style modules, or learned projectors. More recent families, including Qwen-VL [4, 5, 36] and InternVL [10], further improve visual-language alignment while retaining a dedicated vision encoder. Across these architectures, perceptual representations are therefore constructed before entering the language model stack. Audio-language models have followed a similar trajectory, with models such as Qwen2-Audio relying on a dedicated audio encoder to produce continuous acoustic representations before languagemodel processing [11]. More recent unified models such as Gemma 4 instead process audio frames within the shared Transformer, further blurring the boundary between perception and language processing [17].

Discrete-token MLLMs weaken the boundary between perceptual and language processing by placing discrete perceptual codes in the same autoregressive stream as text. Chameleon [8] and Emu3 [37] follow this design. Encoder-free MLLMs remove the dedicated perceptual encoder more directly. Fuyu [6] feeds linearly projected image patches into the Transformer, while EVE [14, 15] and SOLO [9] train single-Transformer models without an external vision encoder. Mono-InternVL [25] incorporates visual experts within a monolithic language model, and Gemma 4 [17] extends encoder-free processing to image and audio inputs.

Discrete-token and encoder-free MLLMs difer in their input units and parameterization, but neither supplies the shared Transformer with continuous features from a dedicated perceptual encoder. We ask how their shared Transformers form perceptual representations and when those representations become available to language processing.

Prior work has primarily evaluated these architectures through downstream performance, leaving their internal perceptual computation largely unexplored. We instead ask how perceptual representations are constructed across Transformer layers when no continuous encoder-derived representation is provided.

## 2.2 Representation Analysis in Multimodal Language Models

Our analysis relies on two complementary views of latent representations: what information they encode and how they are geometrically organized. Linear probes have been widely used to test what is linearly decodable at a given layer [1]. Representation similarity is benefical to find encoder-like structure in a model. Centered kernel alignment (CKA) [20], singular vector canonical correlation analysis [33], and projection-weighted canonical correlation analysis [27] compare representations within and across networks. Intrinsic-dimension analyses further characterize how representation geometry changes across network depth [3,34]. A complementary line of work studies representations as linear subspaces, including the linear representation hypothesis [30]. Principal angles give a classical comparison between two subspaces [7], while cross-projected variance measures how much activity from one condition is captured by another condition’s PCA subspace [16].

Such tools have identified some characteristic structures in multimodal models. The modality gap [22] describes the separation of image and text representations in a shared space, while Nikankin et al. [29] identify modality-specific circuits and late cross-modal alignment within VLMs. Jain et al. [19] further show that intermediate LLM states contain visual perceptual representations whose quality correlates with downstream performance, and Venhof et al. [35] study how representations produced by a dedicated vision encoder progressively map into the language feature space.

Our focus difers in asking how perceptual representations are first constructed when no continuous modality-encoder representation is provided, and how the resulting Virtual Encoder output is subsequently routed among modalityspecific and language-aligned subspaces within the shared residual stream.

## 3 Paired Image-Audio Dataset

We compare image and audio representations using concepts that occur in both modalities. Lin et al. [23] introduced this pairing to study whether multimodal learning can improve a unimodal classifier. Following their protocol, we pair ImageNet-1k images [13] with ESC-50 sounds [31]. We retain their 27 concept pairs and add four unambiguous matches (cow, siren, car horn, and door knock), yielding a set C of 31 concepts. Each concept contains 40 images and 40 audio clips, resulting in 1,240 samples per modality. The shared concept labels let us compare layerwise probe and representation results across image and audio inputs. We split the samples into a training set D and a held-out set $\mathcal { D } _ { \mathrm { H } }$ with $| \mathcal { D } | : | \mathcal { D } _ { \mathrm { H } } | = 3 : 1$ , preserving the uniform concept distribution in both sets. Each element is a tuple $( \kappa _ { \mathrm { V } } , \kappa _ { \mathrm { A } } , c )$ , where $\kappa _ { m }$ for $m \in \{ \mathrm { V } , \mathrm { A } \}$ denotes an image or audio sample and $c \in { \mathcal { C } }$ is its shared concept label. For the causal experiment (Section 4.3), we randomly select four held-out samples per concept and denote this subset by $\mathcal { D } _ { \mathrm { C } } \subset \mathcal { D } _ { \mathrm { H } }$ . We additionally group the concepts into animated/nonanimated classes.

## 4 The Virtual Encoder

Encoder-full MLLMs delegate perception to a dedicated encoder. A Vision Transformer or audio spectrogram model transforms the raw signal into a continuous semantic representation before it reaches the shared Transformers. Encoder-free models omit this stage and take lightly projected patches or frames directly as input, while discrete-token MLLMs take discrete perceptual codes. Our central hypothesis is that, in the latter two cases, an early band of Transformer layers internalizes the missing computation and comes to play the functional role of a modality-specific encoder. We call this early-layer regime the Virtual Encoder.

To test this hypothesis, we ask three questions. (i) When do perceptual representations become semantically organized? (ii) Do intermediate representations exhibit geometry similar to that of a dedicated encoder? (iii) Until what depth are modality-token states causally required for the output? We investigate these questions through linear probing, representational similarity, and causal intervention. We primarily study the encoder-free Gemma 4 12B. The crossmodel analyses additionally include the encoder-free EVE [15] and Fuyu [6], the discrete-token Chameleon [8], and the encoder-full controls LLaVA-1.5 [24], Qwen3-VL [5], Qwen3-Omni [38], and Qwen2-Audio [11].

## 4.1 Semantic Decodability Emerges with Depth

We use linear probing to test whether encoder-free and discrete-token MLLMs progressively organize perceptual inputs into linearly decodable concepts, in contrast to encoder-full MLLMs whose Transformer inputs have already passed through a dedicated encoder.

![](images/1274c37b6192151b84cbba0e16fdc71772b5eb3597047bf0e730e78a32ddc15d.jpg)  
(a) Image

![](images/6a72203a07d8649ff813bb9cad8f34a627a4fdb0559d75f23df26871ef2cd5c2.jpg)  
(b) Audio  
Fig. 3: Layerwise |C|-way linear-probe accuracy. Encoder-free and discrete-token MLLMs develop concept decodability inside the Transformer, whereas encoder-full controls start with highly decodable representations at layer 0.

Experimental setup. We ask when concept identity becomes linearly decodable. For $( K _ { \mathrm { V } } , K _ { \mathrm { A } } , c ) \in \mathcal { D }$ and modality $m \in \{ \mathrm { V } , \mathrm { A } \}$ , feeding sample $\kappa _ { m }$ to a model produces $T _ { m }$ modality-token positions.<sup>1</sup> We index these positions by $t \in \{ 1 , \dots , T _ { m } \}$ and denote their hidden states at depth (or layer) l by $h _ { t } ^ { ( l ) } ( { \cal K } _ { m } ) \in$ $\mathbb { R } ^ { d }$ and represent the image or audio sample by mean-pooling its modality-token hidden states:

$$
z ^ { ( l ) } ( K _ { m } ) = \frac { 1 } { T _ { m } } \sum _ { t = 1 } ^ { T _ { m } } h _ { t } ^ { ( l ) } ( K _ { m } ) .\tag{1}
$$

For each l, we independently fit an ℓ -regularized linear concept classifier $f _ { l } ( z ^ { ( l ) } ( K _ { m } ) ) =$ $W _ { l } z ^ { ( l ) } ( K _ { m } ) + b _ { l } \in \mathbb { R } ^ { | \mathcal { C } | }$ using the representations and concept labels in D and evaluate classification accuracy on $\mathcal { D } _ { \mathrm { H } }$

Results. The layer-wise accuracy is shown in Fig. 3. In the encoder-free MLLMs, semantic information becomes increasingly linearly decodable with depth. For Gemma 4 12B on images, accuracy rises from 8.4% at the projection output at layer 0 to a peak of 95.5% at layer 21, and EVE (23.5%→92.6% at layer 0 to 23) and Fuyu (9.7% → 91.3% at layer 0 to 32) show similar trends. In contrast, the encoder-full Qwen3-VL and LLaVA-1.5 already exhibit highly decodable representations at layer 0, with Qwen3-VL reaching 94.2%. Their Transformer inputs have already been organized by a dedicated modality encoder. The discrete-token Chameleon also builds comparable decodability only after a substantial stretch of Transformer computation. Its decodability is low at layer 0, climbs to a peak at layer 8, and then declines in later layers.

The audio pathway of the encoder-free Gemma 4 12B exhibits an even more pronounced depth-dependent pattern. Probe accuracy climbs from 5.2% at layer 0 to 87.7% at layer 6, but then declines to 36.1% by the final layer. In contrast, the encoder-full Qwen2-Audio is 96.8% decodable at layer 0 and stays above 97% throughout. Gemma 4 12B thus forms a strongly decodable audio representation early but does not maintain that level of decodability downstream.

## Finding 1

Encoder-free and discrete-token MLLMs start with weak linear decodability and develop it over depth, whereas encoder-full MLLMs are already highly decodable at the Transformer input.

## 4.2 Early Layers Exhibit Encoder-like Representations

We test whether hidden states in discrete-token and encoder-free MLLMs share geometric structure with representations from a dedicated single-modal encoder. Encoder-full MLLMs provide controls whose Transformer inputs have already been organized by such an encoder.

Experimental setup. We compare each Transformer layer with DINOv2 for vision and Audio Spectrum Transformer (AST) [18] for audio using linear CKA [20]. For each $( \mathcal { K } _ { \mathrm { V } } , \mathcal { K } _ { \mathrm { A } } , c ) \in \mathcal { D } _ { \mathrm { H } }$ and modality $m \in \{ \mathrm { V } , \mathrm { A } \}$ , we use Eq. (1) to form a sample-level perceptual representation from $\kappa _ { m }$ . We define $X _ { l } \in \dot { \mathbb { R } } ^ { | \mathcal { D } _ { \mathrm { H } } | \times d }$ by vertically stacking and mean-centering these representations at depth l. We similarly define $Y _ { l ^ { \prime } } \in \mathbb { R } ^ { | \mathcal { D } _ { \mathrm { H } } | \times d ^ { \prime } }$ for reference-encoder depth $l ^ { \prime }$ , where $d ^ { \prime }$ is the dimensionality of the reference representation. CKA compares how the two models organize whole samples without requiring their token grids or feature dimensions to match, which is given by

$$
\mathrm { C K A } ( X _ { l } , Y _ { l ^ { \prime } } ) = \frac { \Vert X _ { l } ^ { \top } Y _ { l ^ { \prime } } \Vert _ { \mathrm { F } } ^ { 2 } } { \Vert X _ { l } ^ { \top } X _ { l } \Vert _ { \mathrm { F } } \Vert Y _ { l ^ { \prime } } ^ { \top } Y _ { l ^ { \prime } } \Vert _ { \mathrm { F } } } .\tag{2}
$$

We use the debiased variant of the linear CKA estimator [28]. We use the same image-level pooling for all six vision-capable models. We do the same for the audio modality using AST as our reference encoder.

Even if CKA against an early reference-encoder layer establishes geometric similarity, it does not by itself show that the model processes low-level properties such as edges or textures. We therefore complement CKA with a direct sensitivity test in Gemma 4 12B only for visual input. If its early states depend on low-level visual information carried by high spatial frequencies, removing high-frequency information should substantially change their geometry. For each RGB channel of a clean image I, we apply a two-dimensional Fourier transform, retain the centered circular region whose radius is 12.5% of the maximum frequency radius, and set all coeficients outside this region to zero. We then apply the inverse transform to obtain the low-pass image <sup>˜</sup>I and measure

$$
S _ { l } = 1 - \mathrm { C K A } ( X _ { l } , \tilde { X } _ { l } ) ,\tag{3}
$$

where $\tilde { X _ { l } }$ is the column-centered representations of $\tilde { I }$ computed in the same way as $X _ { l }$ over 60 images randomly sampled from $\mathcal { D } _ { \mathrm { H } }$ . Larger $S _ { l }$ means that removing high-frequency visual information more strongly reorganizes the representation geometry.

![](images/c9528789c1bfd9bfafff67febda015e677ef51ccc638cb4658c7207151532c83.jpg)  
Fig. 4: Image-level CKA. The horizontal axis is the target model’s hidden state; the vertical axis is DINOv2 hidden state for vision and AST hidden state for audio. The first three columns show vision against DINOv2; the last column shows audio against AST. Dashed cyan boxes mark the first 20% of MLLM states against DINOv2 layer 0 to 9 or the first 40% of AST states.

Results. Figure 4 compares six vision-capable models spanning the three families shown in Figure 2. The encoder-free MLLMs are Gemma 4 12B, EVE, and Fuyu, the discrete-token MLLM is Chameleon, and the encoder-full MLLMs are LLaVA-1.5 and Qwen3-VL. The encoder-free and discrete-token maps show a contiguous region of elevated CKA in the lower-left, whereas this block-like structure is weaker in the encoder-full controls. This qualitative pattern indicates that a range of early MLLM states, rather than a single isolated layer pair, organizes images similarly to early layers of a dedicated encoder.

To quantify this pattern, we summarize the lower-left region using the first 20% of MLLM states against DINOv2 layers 0 to 9, shown by the dashed boxes in Figure 4. The mean CKA within this region is 0.33 for Gemma 4 12B, 0.37 for EVE, 0.37 for Fuyu, and 0.61 for Chameleon. The corresponding values are 0.15 for LLaVA-1.5 and 0.21 for Qwen3-VL. Thus models that receive patches or discrete image tokens exhibit about twice the early-to-early similarity on average (0.42 versus 0.18) of models supplied with continuous encoder features.

The audio column shows a related contrast. The encoder-free Gemma 4 12B begins nearly unaligned with AST at layer 0 (CKA 0.00), develops strong ASTlike geometry (CKA 0.89 at layers 4 and 5), and declines to 0.23 by layer 48. The encoder-full Qwen2-Audio starts at 0.55 and remains high, reaching 0.64 at its final state. Thus AST-like geometry develops inside Gemma 4 12B, whereas Qwen2-Audio receives features already shaped by its audio encoder.

The perturbation test clarifies what the early layers of Gemma 4 12B process. At layer 1, high-frequency removal reduces clean-to-perturbed CKA to 0.48 $( S _ { 1 } = 0 . 5 2 )$ . The sensitivity falls to 0.27 at layer 48. Thus the early representation is especially sensitive to fine-scale image structure. Together with its image-level similarity to early DINOv2 layers, this provides evidence for low-level visual processing in Gemma 4 12B.

## Finding 2

Models given patches or discrete image tokens have stronger similarity between their early states and early DINOv2 states than models given continuous encoder features. In Gemma 4 12B, direct sensitivity to high-frequency removal further shows that the early layers process low-level visual structure. These similarities describe how the representations are organized.

## 4.3 The Virtual Encoder’s Output is Causally Read Out at Model-Specific Depth

In an encoder-full MLLM, perceptual information reaches the shared Transformer only after the dedicated encoder has completed its computation. In discrete-token and encoder-free MLLMs, perceptual tokens enter the multimodal Transformer directly and can transfer information to other token positions from the first block. We perturb modality-token states at individual depths to determine when later computation ceases to depend on those states.

Experimental setup. Controlled corruption of internal activations has been used to test whether particular states causally contribute to a model prediction [26]. We apply this intervention principle across depth to determine when the output for sample $\kappa _ { m }$ ceases to depend on its modality-token states. Specifically, we perturb its $T _ { m }$ modality-token positions after Transformer layer l as follows.

$$
\tilde { h } _ { t } ^ { ( l ) } ( K _ { m } ) = \left\{ \begin{array} { l l } { h _ { t } ^ { ( l ) } ( K _ { m } ) + \epsilon _ { t } ^ { ( l ) } , } & { 1 \leq t \leq T _ { m } , } \\ { h _ { t } ^ { ( l ) } ( K _ { m } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad \epsilon _ { t } ^ { ( l ) } \sim \mathcal { N } ( 0 , \sigma _ { l } ^ { 2 } \mathbb { I } _ { d } ) ,\tag{4}
$$

where $\mathbb { I } _ { d }$ is the d-dimensional identity matrix. The states of all other token positions and layers remain unchanged. We sweep one corrupted block at a time over the samples in $\mathcal { D } _ { \mathrm { C } }$ . For Gemma 4 12B, we use $\sigma _ { l } ~ = ~ 5 0 0$ for both the image and audio, sweeping all layers. For the images, we additionally repeat all layers with $\sigma \in \{ 2 5 0 , 1 0 0 0 \}$ to test sensitivity to corruption strength. For Chameleon, whose hidden states’ scale changes substantially with depth, we use $\sigma _ { l } = 2 0 \times \mathrm { R M S } ( \{ h _ { t } ^ { ( l ) } ( \mathcal { K } _ { m } ) \} _ { t = 1 } ^ { T _ { m } } )$ , where RMS is taken across all modality-token positions and hidden dimensions.

We relabel the samples in $\mathcal { D } _ { \mathrm { C } }$ as animated or non-animated based on their concept labels and use the prompt “Is this alive? Answer by either yes or no.” Besides accuracy, we measure the change in the yes/no logit margin. For the n-th selected sample, let $o _ { v } ^ { ( n ) } \in \mathbb { R }$ denote the next-token logit assigned to vocabulary item v. The margin is given by

$$
\delta _ { n } = o _ { \mathtt { y e s } } ^ { ( n ) } - o _ { \mathtt { n 0 } } ^ { ( n ) } .\tag{5}
$$

![](images/95b120c9a40b4f9b9b79d5ccda9027c4ebc756407a4f7fe809c9fe3938425a4f.jpg)  
(a) Image

![](images/dda463d798734f659c5bf804b03a60bd2f7b0cfe76bd9b3c2b3fc7ff7ae06ecd.jpg)  
(b) Audio  
Fig. 5: Causal readout of modality information measured by single-layer corruption of the modality-token residual across depth. The top row shows classification accuracy, and the bottom row shows the mean absolute shift in the yes–no logit margin. For the image modality (left), accuracy collapses to the majority-class floor when the modality-token residual is corrupted at any layer up to the readout boundary (near layer 23; dashed vertical line), but fully recovers when corruption is applied beyond this boundary. The three image curves, corresponding to σ ∈ {250, 500, 1000}, all identify the same transition. The audio modality (right; $\sigma = 5 0 0 )$ shows a similar boundary, although the efect of corruption is weaker.

We compute the same margin after perturbing the hidden states at depth l, denoted by $\tilde { \delta } _ { n } ^ { ( l ) }$ . We define the mean absolute diference between clean and perturbed logit margins as

$$
\varDelta _ { l } = \frac { 1 } { | \mathscr { D } _ { \mathrm { C } } | } \sum _ { n = 1 } ^ { | \mathscr { D } _ { \mathrm { C } } | } | \tilde { \delta } _ { n } ^ { ( l ) } - \delta _ { n } | ,\tag{6}
$$

A large $\varDelta _ { l }$ or an accuracy drop means the answer still depends on the modalitytoken residual at layer $l ;$ a near-zero efect means that information has already been read into other positions.

Results. Figure 5 shows the accuracy and $\varDelta _ { l }$ for perturbation at layer l. For the image modality, corrupting the image tokens at any layer up to layer 22 collapses accuracy to the majority-class floor (64.5%, mean of ∆ logit margin over layers 0 to 22 is 13.42, 13.50, and 13.38 for $\sigma = 2 5 0$ , 500, and 1000, respectively), whereas corruption from layer 24 onward leaves the prediction essentially unchanged (accuracy back to $\geq 9 5 \%$ , margin shift decaying to zero). The image representation carried by the image tokens is thus causally necessary only up to a readout boundary near layer 23, close to the depth at which the probe peaks (layer 21). This boundary is insensitive to a four-fold change in corruption strength. For $\sigma = 2 5 0 , 5 0 0$ , and 1000, accuracy is at or below the 64.5% majority floor through layer 20, recovers to 93.5–94.4% at layer 23, and reaches at least 95.2% at layer 24. The margin-shift curves also nearly coincide. The audio pathway shows the same late boundary (around layer 22 to 24) but a far weaker efect, consistent with an audio representation that is only weakly usable to begin with.

![](images/16b34cfd3b00165e35b84ce4d76ef53e5dc229add65bb1155b516b0799985e42.jpg)  
(a) Linear probe

![](images/30ead95e15b841db30a47841212a6a4b5eeea38996768c5065aa8bcd30a2d945.jpg)  
(b) Encoder similarity

![](images/e9d683377923c987a076512b29046fe49e23f5992c3c0ac867dec3ab14580fa3.jpg)  
(c) Causal readout

![](images/54189cd9874ecb2f3ae1914dce1ea5b57fc7368583460c4948554586af5ad378.jpg)  
(d) Image-text subspace analysis  
Fig. 6: Converging evidence in Chameleon. (a) Concept decodability peaks at hidden state layer 8. (b) Image-level CKA against DINOv2 is high around the same early-tomiddle transition. (c) Corrupting the VQ-image-token residual collapses the alive/notalive task to its majority floor through layer 7 and recovers across layers 8 to 11 (shaded). (d) Image-text subspace overlap follows a diferent trajectory from Gemma 4 12B, peaking earlier and decreasing across the causal readout window.

We repeat the image token intervention in Chameleon, whose Transformers receive 1, 024 discrete image tokens. On the same $| \mathcal { D } _ { \mathrm { C } } |$ held-out images, clean classification accuracy is 87.1%, and the majority baseline is 64.5%. Corrupting layers 0 to 7 reduces accuracy exactly to that floor. It then recovers through a compact transition, reaching 68.5% at layer 8, 75.0% at layer 9, 80.6% at layer 10, and 91.1% at layer 11. (Figure 6c); the absolute logit-margin efect decays toward zero over the remaining depth. Thus, the discrete-token model causally consumes its image-token representation around layers 8 to 11, close to its probe maximum at layer 8 (Section 4.1) and a region of high image-level encoder similarity (Section 4.2).

Together, these interventions localize when the Virtual Encoder’s output is consumed near the midpoint of Gemma 4 12B and substantially earlier in Chameleon. They establish necessity before the respective readout transitions. They do not pinpoint a single encoding layer, since corrupting any layer in the network propagates, and a suficiency test via activation patching remains for future work.

## Finding 3

The output depends strongly on image-token residuals through layer 22 in Gemma 4 12B, but Chameleon begins recovering across layers 8 to 11. The readout depth is therefore an empirical property of each model.

## 4.4 An Early-Layer Virtual Encoder Regime

Taken together, the probe, representational similarity, and causal readout identify an early-to-middle regime in Gemma 4 12B that behaves as an implicit modality encoder. We refer to this computational role as the Virtual Encoder. Related probe and representational signatures appear in other models without continuous encoder features, while Chameleon exhibits a similar but earlier causal readout transition. A Virtual Encoder is a functional description whose presence and depth must be established empirically.

## 5 Characterizing the Virtual Encoder through Modality Subspace Allocation

We next ask whether the diferent downstream behaviors of vision and audio are reflected in how their representations occupy the shared residual stream. We focus on Gemma 4 12B, where image, text, and audio tokens are processed within the same residual stream by the same set of transformer weights. This shared d-dimensional space allows us to directly compare the modality-specific subspaces formed at each depth and to ask where the Virtual Encoder places its output relative to the language subspace. In particular, because all modalities share a common coordinate system, their subspace geometry can be quantified using principal angles, unlike the cross-model comparison in Section 4.2, where we instead relied on the angle-invariant CKA measure.

Experimental setup. For modality $m \in \{ \mathrm { V } , \mathrm { A } , \mathrm { T } \}$ at depth l of a target model, where T means the text modality, we stack the pooled and centered representations $X _ { m } ^ { ( l ) } \in \mathbb { R } ^ { | \mathcal { D } _ { \mathrm { C } } | \times d }$ as in Eq. (1). We use the samples in $\mathcal { D } _ { \mathrm { C } }$ and write a description for each sample. For textual representation, we mean-pool over all textual tokens. Let $U _ { m } ^ { ( l ) } \in \mathbb { R } ^ { d \times k }$ contain the top k principal component directions of $X _ { m } ^ { ( l ) }$ . Following the classical principal-angle construction of Björck and Golub [7], the j-th singular value $\sigma _ { j }$ of $U _ { m } ^ { ( \bar { l } ) \top } U _ { m ^ { \prime } } ^ { ( l ) }$ for $m ^ { \prime } \in \{ \mathrm { V } , \mathrm { A } , \mathrm { T } \}$ equal the cosines of the j-th principal angle $\theta _ { j }$ . We define their mean as a metric for overlap summary, i.e.,

$$
{ \mathrm { O v e r l a p } } _ { m , m ^ { \prime } } ^ { ( l ) } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \sigma _ { j } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \cos \theta _ { j } .\tag{7}
$$

Identical subspaces for modalities m and $m ^ { \prime }$ give $\mathrm { O v e r l a p } _ { m , m ^ { \prime } } ^ { ( l ) }$ being 1, and orthogonal subspaces gives 0. To calibrate this scale, we sample 2, 000 pairs of independent random k-dimensional subspaces in the same d-dimensional residual space and report their mean overlap. We repeat the analysis for $k \in \{ 5 , 1 0 , 2 0 , 4 0 \}$ Following cross-projected variance analyses [16], we also report the fraction of variance in modality m captured by the subspace of modality m<sup>′</sup> as follows.

$$
\mathrm { V a r C a p } _ { m  m ^ { \prime } } ^ { ( l ) } = \frac { \Vert X _ { m } ^ { ( l ) } U _ { m ^ { \prime } } ^ { ( l ) } \Vert _ { \mathrm { F } } ^ { 2 } } { \Vert X _ { m } ^ { ( l ) } \Vert _ { \mathrm { F } } ^ { 2 } } .\tag{8}
$$

![](images/1f15c9f746fa07f526f30d3e75a3d1a18ea553119d1d34a1325b3709b5108a0e.jpg)  
(a) Gemma 4 12B modality overlap

![](images/f999d5c7485db222b2ea15f9044b456837df08ee799c3a6331b6332b733b2fe9.jpg)  
(b) Image-audio contro

![](images/206ab8e03ca33594ca7c53513daf4e479731047e6b05f551de4c80239e1eb8ee.jpg)  
(c) Sensitivity to k  
Fig. 7: Modality subspace allocation in the shared residual stream. (a) Top-20 overlap in Gemma 4 12B; the horizontal dashed line is the mean overlap of independent random 20-dimensional subspaces in the original d-dimensional space. Image-text subspace alignment is maximal near the readout transition (vertical dashed line), whereas the audio pairs remain much weaker but above chance. (b) Image-audio overlap for Gemma 4 12B and the encoder-full Qwen3-Omni control. (c) The image-text peak and layer-mean audio overlaps for $k \in \{ 5 , 1 0 , 2 0 , 4 0 \}$ , with the corresponding random baseline.

We use the encoder-full Qwen3-Omni as a control. We apply the same imagetext analysis to the discrete-token Chameleon using the |D | descriptions and compare hidden states.

Results. The Virtual Encoder rotates vision into the language subspace near the readout boundary. In Gemma 4 12B (Figure 7a), the image-text subspace overlap climbs from near-orthogonal at layer 0 (Overlap $_ { \mathrm { V , T } } ^ { ( 0 ) } = 0 . 0 6 )$ to a broad high-overlap band at layers 18 to 22, peaking at layer 19 (0.53; 34% of the image subspace’s variance lies inside the text subspace), and then falls away again. This band contains the probe peak at layer 21 and lies immediately before the causal readout transition near layer 23 (Section 4.3). The image subspace is therefore most aligned with the model’s language subspace around the depth where the prediction becomes independent of the image-token residual. The image-audio and text-audio overlaps are much weaker. At k = 20, their layer means are 0.15 and 0.14, respectively, which are closer to independence (the random baseline’s overlap is 0.061). They therefore share some structure and should not be described as strictly orthogonal, but remain far below the image-text peak of 0.53. This separation is robust to PCA rank as shown in Figure 7c). Thus, the audio modality does not undergo the pronounced, readout-aligned rotation toward the language subspace seen for vision.

The encoder-full control exhibits a diferent subspace trajectory. In the encoder-full control (Figure 7b), the image-audio overlap rises monotonically with depth, with no comparable peak. This trajectory is consistent with progressive cross-modal mixing and does not exhibit the transient geometry observed in Gemma 4 12B.

The computational role does not require one universal routing geometry. Chameleon provides a useful counterpoint (Figure 6d). Its image-text overlap rises from 0.059 at layer 0 to a modest early peak of 0.091 at layer 4, then decreases across the causal readout transition to 0.047 at layer 11. It therefore does not reproduce Gemma 4 12B’s readout-aligned rotation. Thus constructing and consuming a perceptual representation inside the Transformer also occurs in a discrete-token MLLM, while the route taken through the residual stream is architecture dependent. In Gemma 4 12B, modalities occupy only weakly overlapping subspaces, with vision rotating from an image-private subspace toward the language subspace around readout.

Taken together, Sections 4 and 5 separate three properties of the Virtual Encoder. They show where a perceptual representation is formed, when it is causally consumed, and how it is routed. Gemma and Chameleon support the first two properties, whereas their subspace trajectories difer. A Virtual Encoder ofers the computational role of internally constructing a usable perceptual representation before readout, rather than by a particular layer interval or routing geometry.

## 6 Discussion

The Virtual Encoder perspective separates when perceptual representations are formed, when they are read out from modality tokens, and how they are routed afterward. Its strength and depth may depend on how much perceptual processing remains after fusion, placing architectures on a continuum rather than in a fixed encoder-free category. This view makes readout depth and modalityto-language alignment potential design variables. Controlling them may help locate information loss, improve fusion, and preserve perceptual evidence for downstream reasoning.

Our evidence nevertheless leaves important gaps. The full image-text-audio subspace analysis is limited to Gemma 4 12B, while Chameleon provides only an image-text comparison. The language subspace is operationally defined by PCA of text representations. We do not intervene directly on subspace rotation, and audio’s weak language alignment is only correlated with its loss of decodability. Controlled front-end comparisons, direct routing interventions, and broader cross-architecture evaluation are therefore needed to establish when Virtual Encoders arise and whether their geometry causally afects model behavior.

## 7 Conclusion

We identify the Virtual Encoder as a regime in which an MLLM internally forms a usable perceptual representation before causal readout. Its depth and routing vary across architectures. Together, these findings provide a concrete way to identify and study encoder-like computation inside encoder-free multimodal models, rather than assuming it from the architecture alone.

## References

1. Alain, G., Bengio, Y.: Understanding intermediate layers using linear classifier probes. In: International Conference on Learning Representations (ICLR) Workshop (2017) 2, 4

2. Alayrac, J.B., et al.: Flamingo: A visual language model for few-shot learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2022) 1, 3

3. Ansuini, A., Laio, A., Macke, J.H., Zoccolan, D.: Intrinsic dimension of data representations in deep neural networks. In: Advances in Neural Information Processing Systems (NeurIPS) (2019) 4

4. Bai, J., et al.: Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 (2023) 1, 3

5. Bai, S., et al.: Qwen3-VL technical report. arXiv preprint arXiv:2511.21631 (2025) 3, 5

6. Bavishi, R., et al.: Fuyu-8b: A multimodal architecture for ai agents. Adept AI Blog (2023) 1, 4, 5

7. Björck, Å., Golub, G.H.: Numerical methods for computing angles between linear subspaces. Mathematics of Computation 27(123), 579–594 (1973). https://doi. org/10.1090/S0025-5718-1973-0348991-3 4, 12

8. Chameleon Team: Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 (2024) 1, 4, 5

9. Chen, Y., et al.: A single transformer for scalable vision-language modeling. Transactions on Machine Learning Research (TMLR) (2024) 1, 4

10. Chen, Z., et al.: InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 3

11. Chu, Y., et al.: Qwen2-Audio technical report. arXiv preprint arXiv:2407.10759 (2024) 3, 5

12. Dai, W., Li, J., Li, D., et al.: InstructBLIP: Towards general-purpose visionlanguage models with instruction tuning. In: Advances in Neural Information Processing Systems (NeurIPS) (2023) 1, 3

13. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A largescale hierarchical image database. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2009) 5

14. Diao, H., Cui, Y., Li, X., et al.: Unveiling encoder-free vision-language models. In: Advances in Neural Information Processing Systems (NeurIPS) (2024) 1, 4

15. Diao, H., et al.: EVEv2: Improved baselines for encoder-free vision-language models. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2025) 4, 5

16. Elsayed, G.F., Lara, A.H., Kaufman, M.T., Churchland, M.M., Cunningham, J.P.: Reorganization between preparatory and movement population responses in motor cortex. Nature Communications 7, 13239 (2016). https://doi.org/10.1038/ ncomms13239 4, 12

17. Gemma Team, Google DeepMind: Introducing Gemma 4. Google DeepMind (2026), https://deepmind.google/models/gemma/gemma-4/ 1, 2, 3, 4

18. Gong, Y., Chung, Y.A., Glass, J.: AST: Audio Spectrogram Transformer. In: Interspeech. pp. 571–575 (2021). https://doi.org/10.21437/Interspeech.2021-698 7

19. Jain, J., Yang, Z., Shi, H., Gao, J., Yang, J.: Elevating visual perception in multimodal llms with visual embedding distillation. In: NeurIPS. vol. 38, pp. 91092– 91123 (2025) 4

20. Kornblith, S., Norouzi, M., Lee, H., Hinton, G.: Similarity of neural network representations revisited. In: International Conference on Machine Learning (ICML) (2019) 2, 4, 7

21. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning (ICML) (2023) 1, 3

22. Liang, W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J.: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2022) 4

23. Lin, Z., Yu, S., Kuang, Z., Pathak, D., Ramanan, D.: Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 5

24. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Advances in Neural Information Processing Systems (NeurIPS) (2023) 1, 3, 5

25. Luo, G., et al.: Mono-InternVL: Pushing the boundaries of monolithic multimodal large language models. arXiv preprint arXiv:2410.08202 (2024) 4

26. Meng, K., Bau, D., Andonian, A., Belinkov, Y.: Locating and editing factual associations in GPT. In: NeurIPS. pp. 17359–17372 (2022) 9

27. Morcos, A.S., Raghu, M., Bengio, S.: Insights on representational similarity in neural networks with canonical correlation. In: Advances in Neural Information Processing Systems (NeurIPS) (2018) 4

28. Murphy, A., Zylberberg, J., Fyshe, A.: Correcting biased centered kernel alignment measures in biological and artificial neural networks. In: ICLR Workshop on Representational Alignment (2024) 7

29. Nikankin, Y., Arad, D., Gandelsman, Y., Belinkov, Y.: Same Task, Diferent Circuits: Disentangling modality-specific mechanisms in VLMs. In: NeurIPS (2025) 4

30. Park, K., Choe, Y.J., Veitch, V.: The linear representation hypothesis and the geometry of large language models. In: International Conference on Machine Learning (ICML) (2024) 4

31. Piczak, K.J.: ESC: Dataset for environmental sound classification. In: Proceedings of the 23rd ACM International Conference on Multimedia (2015) 5

32. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML) (2021) 3

33. Raghu, M., Gilmer, J., Yosinski, J., Sohl-Dickstein, J.: SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In: Advances in Neural Information Processing Systems (NeurIPS) (2017) 4

34. Valeriani, L., et al.: The geometry of hidden representations of large transformer models. In: Advances in Neural Information Processing Systems (NeurIPS) (2023) 4

35. Venhof, C., Khakzar, A., Joseph, S., Torr, P., Nanda, N.: How visual representations map to language feature space in multimodal llms. In: CVPRW (2025) 4

36. Wang, P., et al.: Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024) 3

37. Wang, X., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024) 1, 4

38. Xu, J., et al.: Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765 (2025) 5

39. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023) 3