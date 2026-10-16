# Not Just a Subject: Capturing Adaptive Scene-Specific Variance for Visual Storytelling

SeungJu Cha<sup>1</sup> , Ye-Chan Kim<sup>1</sup> , Kwanyoung Lee<sup>1</sup> , and Dong-Jin Kim<sup>1</sup>

Hanyang University, Seoul 04763, South Korea {sju9020, dpcksdl78, mobled37, djdkim}@hanyang.ac.kr <sup>§</sup> https://github.com/SeungJuCha/EOS-Story

Abstract. Recent advancements in difusion models have significantly improved text-to-image generation, enabling their application to visual storytelling, where coherent image sequences are generated from narrative prompts. However, existing methods primarily focus on identity preservation and often fail to adapt to diverse scene variations, resulting in semantic misalignment and structural inflexibility. We propose EOS-Story, a training-free framework that harmonizes identity consistency with scene-specific adaptation across text and vision spaces. In the text space, we decouple the narrative into identity and scene sub-prompts to prevent information dilution, but this leaves the scene without access to appearance cues. To bridge this gap, we introduce Cross-Contextual Bridging, which softly injects a phrase-level appearance residual into the scene’s ⟨EOS⟩ embedding. In the vision space, we find that scale mismatch between identity and scene subjects causes a few identity tokens to be repeatedly transferred, producing structural artifacts. We address this with Adaptive Identity Preservation, which anchors identity features at the scene-specific scale (Identity Anchoring) and integrates them via Identity-Aware Self-Attention. To evaluate performance under varied spatial compositions, we extend the Consistory+ benchmark with diverse framing scenarios (Consistory+\_+). Extensive experiments show that EOS-Story achieves competitive or superior performance on both benchmarks.

Keywords: Storytelling · Text-to-Image Generation

## 1 Introduction

Difusion models [23,26] have recently advanced the quality of text-to-image generation, enabling diverse applications in visual content creation. Among these, the visual storytelling task, applicable to digital comics and character-driven animation [22], is particularly challenging because it requires aligning with evolving narrative semantics while maintaining a coherent subject across scenes [30].

Recent approaches [17, 30, 40] explicitly design mechanisms to preserve subject identity by transferring subject-relevant features into each scene during generation. While these methods efectively maintain identity consistency, they often underrepresent scene-specific variance in both text and vision spaces—the diverse semantic and visual elements that vary across narrative contexts (Fig. 1). First, at the semantic level, current methods [17, 30, 40] often treat narrative prompts as monolithic inputs, where various semantics (e.g., appearance and scene descriptions) are encoded into a unified embedding. This can lead to information dilution, making it dificult for the model to uniformly attend to all details [14]. As a result, even when a prompt contains rich details (e.g., specific actions or environmental changes), the model often fails to selectively extract scene-specific cues from the prompt. This directly undermines narrative fidelity, producing scenes that ignore their intended semantics (Fig. 1a,b). Second, at the visual level, existing approaches typically rely on spatially blind feature injection, where identity features are transferred without accounting for the spatial constraints or framing of each scene. This oversight of structural scene variance (e.g., camera angles or subject scales) can lead to subject inflexibility, such as contextually mismatched poses or repetitive structural patterns across scenes (Fig. 1b,c). Consequently, subjects are rendered with rigid structural patterns that ignore each scene’s intended layout, further weakening alignment with scene-specific prompts. Therefore, achieving coherent storytelling requires scenespecific adaptation, where identity consistency is harmonized with the semantic and structural variance of each scene.

![](images/c602207351dec5de5e938c2a7d8fe1c403b56b8308c04b947b21cf3928118405.jpg)  
Fig. 1: Limitations in scene-level adaptation of previous methods. Existing approaches often fail to account for scene-specific variance: (a) semantic misalignment in simple cases, (b) loss of fine-grained prompt details in complex scenarios, and (c) inability to adapt to diverse viewpoints and scales. These issues often lead to repetitive subject structures and weakened alignment with scene-specific prompts. In contrast, our method achieves better scene-level fidelity by adaptively harmonizing identity consistency with the unique context of each scene.

To address this issue, we propose EOS-Story, a framework designed to adaptively account for scene-specific variance via two complementary modules, (1) Cross-Contextual Bridging: We hypothesize that semantic misalignment arises when long and complex prompts are encoded holistically, causing scenevarying attributes to be diluted [14,16]. Rather than encoding all information in a single prompt, we decouple the narrative into an identity prompt describing the subject and a scene prompt describing the scenario. For instance, “a woman with blond hair reading a book.” is split into an identity prompt (“a woman with blond hair.”) and a scene prompt (“a woman reading a book.”). However, this separation introduces a new challenge: since the scene prompt contains no appearance information, the subject generated within the scene has no way to reflect the actual target identity. To bridge this semantic gap without reverting to monolithic encoding, we preserve the scene sub-prompt as a semantic anchor and softly inject appearance information at the embedding level. Motivated by the strong generative influence and summarization ability of ⟨EOS embeddings [20, 37], we interpret it not merely as a terminal token, but as a residual-based semantic control point for scene-aware storytelling (Sec. 3.1). Unlike prior EOS-based methods that substitute the ⟨EOS⟩ embedding between prompts difering by a single word (e.g., an object class) [9, 20], we capture a phrase-level appearance diference and infuse it into the scene’s ⟨EOS⟩, reflecting rich multi-attribute cues without explicit word-level repetition (Sec. 3.2). This provides the scene image with the specified appearance cues at the semantic level, enabling more efective feature-level identity transfer.

(2) Adaptive Identity Preservation: To mitigate subject inflexibility induced by spatially blind feature transfer, we introduce a spatially adaptive identity injection mechanism that better aligns identity features with the scenespecific layout (Sec. 3.3). We observe that the scale mismatch between identity and scene subjects prevents various identity tokens from corresponding to the scene tokens. As a result, only a few identity tokens are repeatedly transferred to the scene and cause structural artifacts. Motivated by this, we apply Identity Anchoring, which adaptively resizes identity features to match the intended framing and scale of the scene. This enforces structural compatibility between identity and scene representations, reducing scale-induced distortions (Fig. 10b). We then refine the integration through Identity-Aware Self-Attention, which enables scene features to selectively attend to subject features, building upon [30]. Unlike existing spatially blind transfer strategies [17, 30, 40], our approach supports scene-conditioned variation in subject structure (e.g., viewpoint changes) while preserving core identity characteristics. To evaluate scene-specific structural variance, we introduce Consistory+\_+, extending Consistory+ [17] with diverse framing prompts [15] to simulate the realistic framing shifts of realworld storytelling. We validate our method on Consistory+ and Consistory+\_+, demonstrating consistently improved performance under framing variations. Our contributions are as follows:

We show that the ⟨EOS⟩ embedding can carry phrase-level, multi-attribute information, enabling appearance cues to be infused during generation.

– We propose EOS-Story, a training-free framework with two modules: Cross-Contextual Bridging (text-space semantic recomposition) and Adaptive Identity Preservation (vision-space scale-aware transfer).

– We introduce Consistory+\_+, an extended benchmark with diverse framing phrases, and demonstrate consistent improvements over existing approaches on both the original and extended benchmarks.

## 2 Related Work

Visual Storytelling. With the advancement of difusion models [23, 26], their applications have expanded into various domains. To meet users’ needs for generating images that preserve subject identity, numerous personalization methods have been proposed [1, 7, 11, 13, 28, 36]. However, these approaches primarily focus on preserving subject appearance [30], often overlooking how the subject should adapt across diferent visual contexts. To bridge this gap, visual storytelling [17, 19, 25, 33, 40] has emerged as a compelling paradigm for generating identity-consistent images across diverse and dynamic scenes [17]. Some methods fine-tune difusion models to explicitly capture subject identity [25,33], while [8,17,19,30,34,40] explore training-free strategies that leverage intermediate features from the attention mechanism. For example, Consistory [30] directly maps subject identity across diferent scenes to achieve identity consistency, while 1P1S [17] leverages the context consistency of CLIP [24] and transfers this information through the cross-attention mechanism. However, these methods prioritize identity consistency at the cost of scene adaptation, often producing rigid subject depictions and weak prompt alignment. While concurrent works [8, 34] attempt to alleviate subject rigidity, they still delegate prompt alignment to the underlying difusion model. In contrast, we find that faithful scene adaptation depends not only on the vision space but also on the text space, and design our framework to jointly address both—improving prompt alignment through semantic-level residual injection and adapting to structural variance through scale-aware feature transfer.

⟨EOS⟩ Embedding in Difusion Models. In CLIP [24], which is widely used to encode text in text-to-image generation models, the ⟨EOS⟩ token encapsulates a contextual summary of the entire prompt [37]. Recent mechanistic analyses [31] further show that padding tokens, including ⟨EOS⟩, carry non-trivial semantic influence in text-to-image generation. Built upon this, [37] demonstrates that ⟨EOS⟩ has dominant influence in image generation: replacing the ⟨EOS⟩ embedding of an original prompt (e.g., “A blue bird”) with that of a target prompt (e.g., “A blue chair”) can steer the generated image toward the target semantics. Motivated by this property, several works [9, 12, 20] have utilized ⟨EOS⟩ or text-embedding manipulation for downstream tasks. Specifically, [9] substitutes ⟨EOS⟩ to resolve semantic binding issues in object–attribute prompts, [20] injects the entire base prompt embedding along with ⟨EOS⟩ into the position of specific objects to perform image editing, and [12] manipulates text embeddings to suppress unwanted content. Unlike prior methods that either substitute ⟨EOS between prompts difering by a single word [9, 20] or manipulate embeddings for content removal [12], we use ⟨EOS⟩ as a scene-aware interface to transfer phrase-level appearance information into scene-specific prompt representations for visual storytelling.

## 3 Method

Overview. To achieve identity consistency while allowing subjects to adapt to diverse scenes, we propose EOS-Story (Fig. 3), a framework that models scene-specific variance in two complementary spaces. (1) Cross-Contextual Bridging (CCB) addresses semantic variance by transferring phrase-level appearance cues into the scene context through the ⟨EOS⟩ embedding, mitigating information dilution in long and complex prompts. (2) Adaptive Identity Preservation (AIP) addresses structural variance by aligning identity features to each scene’s subject scale (Identity Anchoring), followed by Identity-Aware Self-Attention refinement.

## 3.1 EOS Embedding as a Semantic Handle

In this section, we explore the generative functional role of the ⟨EOS⟩ embedding.

We first test whether the ⟨EOS⟩ embedding can serve as a global semantic control point. As shown in Fig. 2a, adding an attribute embedding (e.g., “black”) at diferent token positions of a base prompt (“A photo of a dog”) produces notably diferent outcomes. Directly adding the attribute embedding to the target word token (“dog”) fails to alter the output, and even a broad injection across all non-EOS tokens produces only minor changes. In contrast, injecting the attribute only into the ⟨EOS⟩ embedding induces a clear semantic shift, suggesting that the ⟨EOS⟩ embedding serves as an effective handle for embedding-level semantic editing during generation.

We further examine whether semantic changes can be composed at the ⟨EOS⟩ level through additive and subtractive operations (Fig. 2b). Simple subtraction often yields incomplete edits, likely because unchanged word-level tokens continue to preserve the original semantics (Fig. 2b2), whereas addition and replacement

![](images/243f636784c011b5b8ec493c6f5a13482f822506a22594b36c2091240f428140.jpg)  
Fig. 2: Efect of EOS modification. (a) Injecting “black” into the ⟨EOS⟩ token most efectively reflects the intended attribute. (b) Editing ⟨EOS⟩ embedding induces coherent semantic changes in images.

![](images/ca7795fcf7a7c4de8b09cb750fcb6097cccdf4456132d0239993e83e6d906f8b.jpg)  
Fig. 3: EOS-Story pipeline. Our framework contains two complementary modules: (1) Cross-Contextual Bridging for representation-level semantic recomposition (Sec. 3.2), and (2) Adaptive Identity Preservation (AIP) for spatially-aware structural alignment (Sec. 3.3). AIP consists of two subcomponents: Identity Anchoring (IA), which establishes scale- and context-aware correspondence, and Identity-Aware Self-Attention (IASA), which refines context-dependent feature integration. During the denoising process over difusion timesteps T, we switch from explicit structural anchoring (IA) to attention-based refinement (IASA) at timestep t within the unet.

more reliably induce coherent semantic shifts. Notably, our observation is that the ⟨EOS⟩ embedding can capture not only a single attribute [9, 20] but also multiple semantic factors, simultaneously afecting actions and environments through one EOS-level edit (Fig. 2b3). These edits occur without modifying the word-level tokens, indicating that semantic adjustments at the ⟨EOS⟩ level can override or complement persistent token-level semantics. These observations suggest that the ⟨EOS⟩ embedding provides a context-preserving handle that can encode rich, multi-attribute semantics through a residual operation. Motivated by this, we design Cross-Contextual Bridging, which transfers phrase-level appearance information into each scene’s ⟨EOS⟩ embedding, allowing subjectspecific appearance cues to be integrated while maintaining the scene context.

## 3.2 Cross-Contextual Bridging

To avoid treating a story prompt as a single monolithic description, we construct reference subprompts from structured storyboard inputs (e.g., Consistory+ benchmark [17]). Each input is decomposed into four semantic components: style, subject, appearance, and scene. Encoding all components as a single prompt (e.g., “A majestic winter-themed artwork of a regal snow queen in icy beauty watching over a frozen kingdom.”) can dilute fine-grained appearance cues in long and compositionally complex descriptions [14]. To recover such appearance cues without lengthening the scene prompt, we construct three reference subprompts around a minimal shared context: (1) Style Prompt: $\mathrm { S t y l e ~ + ~ }$ Subject, (2) Identity Prompt: $\mathrm { S t y l e + S u b j e c t + A }$ ppearance, and (3) Scene Prompt: Style + Subject + Scene. The shared Style + Subject prompt serves as the minimal anchor for measuring appearance-specific displacement before transferring it to each scene prompt. This anchor is scene-independent, while still preserving the subject slot and visual style, making it suitable for defining an appearance cue that can be shared across all scenes.

By keeping the scene prompt free of explicit appearance phrases, we preserve the scene description as the primary semantic context. Cross-Contextual Bridging then reintroduces the missing appearance cue at the embedding level. This allows the subject appearance to be recovered without modifying the wordlevel scene description. Building on our empirical observations (Sec. 3.1 and Appendix A), we utilize the ⟨EOS⟩ embedding as a controllable semantic handle, enabling phrase-level appearance transfer without increasing the textual length of the scene description. We obtain the corresponding ⟨EOS⟩ embeddings, $e o s _ { s t y } ,$ eos $i d \cdot$ and $e o s _ { s c } ,$ from the CLIP [24] text encoder. Although $e o s _ { i d }$ contains the desired appearance information, directly adding it to $e o s _ { s c }$ can redundantly reinforce the shared style and subject semantics, leading to unintended semantic drift. We therefore define the appearance residual as the local displacement from the shared style-subject anchor to the identity prompt, following the approximate linearity of CLIP embedding space [32]: $\Delta _ { a p p } = e o s _ { i d } - e o s _ { s t y }$ . We transport this residual to the scene prompt by constructing the infused embedding $e o s _ { i n f } \colon$

$$
e o s _ { i n f } = e o s _ { s c } + \varDelta _ { a p p } .\tag{1}
$$

We replace each scene ⟨EOS⟩ embedding $e o s _ { s c }$ with its infused counterpart $e o s _ { i n f }$ . The identity prompt embedding and the modified scene prompt embedding are then fed into the difusion model for the subsequent feature-level identity transfer. This formulation restores subject appearance cues within the scene prompt representation while preserving the original scene semantics. This appearance-aware representation further reduces the semantic gap between the scene subject and its identity counterpart before feature injection, providing a stronger basis for the subsequent feature-level identity transfer (Sec. 3.3).

## 3.3 Adaptive Identity Preservation

Unlike spatially blind identity injection, we focus on anchoring identity features according to each scene’s subject structure, allowing the subject to adapt in scale and viewpoint while preserving its core identity cues. To achieve natural subject alignment across scenes, we transfer visual features from the identity image $x _ { i d } \in$ $\mathbb { R } ^ { \breve { H } W \times C }$ to N scene images $\{ x _ { s c } ^ { n } \} _ { n = 1 } ^ { N } \in \mathbb { R } ^ { N \times H W \times C }$ . We use intermediate features from the first decoder block of the U-Net, where coarse spatial structure is known to emerge [29], to establish correspondence between identity and scene subject regions. Identity Anchoring is applied during early denoising stages, when global layout is primarily established, so that later steps can refine fine-grained details without disrupting the aligned structure. During generation, we obtain subject masks $\mathcal { M } _ { i d } \ \in \ \mathbb { R } ^ { \breve { H } W \times 1 }$ and $\{ \mathcal { M } _ { s c } ^ { n } \} _ { n = 1 } ^ { N } \in \mathbb { R } ^ { N \times \breve { H } \breve { W } \times 1 }$ from the cross-attention maps of the corresponding subject tokens.

Identity Anchoring. Because each scene is guided by a diferent narrative prompt, the subject’s scale and framing naturally vary across scenes. We empirically find that this scale variation directly degrades feature correspondence in Fig. 4. In the figure, as the scale mismatch between identity and scene subjects grows, the correspondence from scene tokens concentrates on a smaller subset

of identity tokens. Consequently, a few identity tokens are overused across the scene subject, causing structural artifacts and scene-inflexible depictions (Fig. 10b). This suggests that establishing reliable correspondence requires first aligning the two subjects to a comparable size.

![](images/f414c26ce0a50dbb05db7d687ad556f6e960dba1ee393e8942eb179ed751a41f.jpg)

We therefore rescale the identity features to align with the subject size in each scene, enabling adaptive feature matching across varying spatial contexts (Fig. 3 (2)). For each scene, we crop the identity subject region $\hat { x } _ { i d } ~ \in ~ \mathbb { R } ^ { h ^ { \prime } w ^ { \prime } \times C }$ and mask $\hat { \mathcal { M } } _ { i d } ^ { \mathrm { ~ ~ } } \in$ $\mathbb { R } ^ { h ^ { \prime } w ^ { \prime } \times 1 }$ using the bounding box of $\mathcal { M } _ { i d }$ . We similarly crop the scene subject region $\hat { x } _ { s c } ^ { n } = \mathrm { C r o p } ( \mathcal { M } _ { s c } ^ { n } \odot x _ { s c } ^ { n } ) \in$ $\mathbb { R } ^ { h w \times \check { C } }$ , where $h ^ { \prime } w ^ { \prime }$ and hw denote the

Fig. 4: Identity-token coverage. Unique identity-token coverage—the fraction of distinct identity tokens referenced by scene tokens—drops sharply as scale mismatch | log r| grows without size matching (orange), while size matching keeps it stable (blue). Low coverage means few tokens are repeatedly transferred.

identity and scene-specific subject sizes, respectively. We then resize the masked identity feature to match the scene subject’s resolution, obtaining the sizealigned identity feature: $\boldsymbol { \tilde { x } } _ { i d } ^ { n } = \operatorname { R e s i z e } ^ { ( n ) } \bigl ( \hat { \mathcal { M } } _ { i d } \odot \hat { \boldsymbol { x } } _ { i d } \bigr ) \in \mathbb { R } ^ { h w \times C }$ , where $\bar { \mathrm { R e s i z e } } ^ { ( n ) } ( \cdot )$ resizes the input to the spatial size of $\hat { x } _ { s c } ^ { n }$ . This size-aware alignment allows scene features to reference identity information at a comparable scale, reducing mismatched transfers from irrelevant identity regions. To transfer identity information into each scene, we compute a correspondence map between the size-aligned identity feature $\tilde { x } _ { i d } ^ { n } \in \mathbb { R } ^ { h w \times \tilde { C } }$ and the scene subject feature $\hat { x } _ { s c } ^ { n } \in \mathbb { R } ^ { h w \times C }$ . After computing the similarity matrix $S = \hat { x } _ { s c } ^ { n } \cdot ( \tilde { x } _ { i d } ^ { n } ) ^ { \top }$ , we identify the most similar position in the identity feature for each scene position $i \in [ 1 , h w ]$

$$
\varPhi _ { i } = \underset { j \in [ 1 , h w ] } { \mathrm { a r g m a x } } S _ { i j } .\tag{2}
$$

Since every scene position is matched to its most similar identity feature, positions without a genuine counterpart are still forced into a match, propagating spurious identity cues. To suppress unreliable transfers, we retain only matches whose similarity exceeds an adaptive confidence threshold derived from the similarity distribution. Specifically, we set the threshold to $\mu + \sigma _ { \mathrm { { ; } } }$ , where $\mu$ and $\sigma$ denote the mean and standard deviation of similarities across all scene positions. We then blend the scene subject and matched identity features:

$$
\hat { x } _ { s c } ^ { n } \gets \lambda \cdot \hat { x } _ { s c } ^ { n } + ( 1 - \lambda ) \cdot \tilde { x } _ { i d } ^ { n } [ \varPhi ] ,\tag{3}
$$

where $\lambda \in [ 0 , 1 ]$ balances scene context preservation and identity transfer. Finally, we replace the subject region in $x _ { s c } ^ { n }$ with the blended subject feature ${ \hat { x } } _ { s c } ^ { n } .$ This adaptive alignment harmonizes identity cues with scene-specific structure, preserving the intended composition of each scene.

Identity-Aware Self-Attention. Although the blended scene feature $x _ { s c } ^ { n }$ incorporates visual information from the identity image $x _ { i d } .$ explicit feature replacement may still fail to blend naturally into the surrounding scene context. Built upon [30], we enable each scene to dynamically attend to identity features during self-attention (Fig. 3 (3)). After extracting size-aligned key and value $\tilde { K } _ { i d } ^ { n }$ and $\tilde { V } _ { i d } ^ { n }$ from the identity image feature $x _ { i d }$ , we concatenate them with the corresponding key and value from the scene features $ { \boldsymbol { x } } _ { s c } ^ { n } \colon  { \boldsymbol { K } } _ { s c , i d } ^ { n } =  { \operatorname { C o n c a t } } (  { \boldsymbol { K } } _ { s c } ^ { n } , \tilde {  { \boldsymbol { K } } } _ { i d } ^ { n } )$ $V _ { s c , i d } ^ { n } = \mathrm { C o n c a t } ( V _ { s c } ^ { n } , \tilde { V } _ { i d } ^ { n } )$ . The self-attention is then computed:

$$
\mathrm { A t t n } ( Q _ { s c } ^ { n } , K _ { s c , i d } ^ { n } , V _ { s c , i d } ^ { n } ) = \mathrm { S o f t m a x } \left( \frac { Q _ { s c } ^ { n } K _ { s c , i d } ^ { n \top } } { \sqrt { d } } \right) V _ { s c , i d } ^ { n } .\tag{4}
$$

## 4 Experiments

Implementation Details. We use the SDXL-base-1.0 checkpoint for all experiments and set the number of denoising steps to 50. We extract binary mask through the Otsu algorithm [21], which is commonly used [30]. Adaptive Identity Preservation is applied only to the decoder layers of the unet, with a transition from Identity Anchoring to Identity-Aware Self-Attention at step t=9, motivated by the observation that global layout forms in early denoising steps while finegrained details emerge later [37]. The weighting coeficient λ is fixed at 0.55 in all experiments. All experiments are conducted on an NVIDIA A6000 GPU.

Benchmarks. Consistory+ [17] spans subjects across diverse categories (e.g., humans, animals, nature). To evaluate generation performance under more complex spatial contexts, we introduce an extended version, Consistory+\_+, by adding diverse framing phrases covering “shot size,” “camera angle,” and “composition.” We first collect framing-related descriptors from prior scene-structure studies [15], filtering out those unlikely to be visually reflected in generated images. To automatically assign appropriate descriptors, we instruct GPT-4o [10] to classify each subject and apply framing tags only to animate ones (e.g., humans, animals), for which such cues are perceptually more salient and semantically aligned [27]. Further details are provided in the supplementary material.

Metrics. We evaluate text-to-image alignment using CLIP-T [39]. For subject similarity, we employ CLIP-I [24] and DreamSim [6], following [8,17]. To measure aesthetic quality, we use a SigLIP-based predictor [4, 38], as each story involves diverse artistic styles and appearances. To assess fine-grained prompt details, we use ImageReward [35], a human-aligned metric capturing nuanced text-image alignment beyond superficial similarity. To evaluate framing fidelity, we employ ShotVL-7B [15] to classify whether generated images align with the framing phrases specified in the text prompt.

Table 1: Quantitative results on Consistory+ [17] and our extended Consistory+\_+. Our method achieves better text alignment without compromising image quality. For StoryDifusion [40], we generate images at a resolution of 768 × 768.
<table><tr><td rowspan="2"></td><td colspan="5">Consistory+</td><td colspan="5">Consistory+_+</td></tr><tr><td>CLIP-T↑ CLIP-I↑ DreamSim↓ Aesthetic↑ ImageReward↑</td><td></td><td></td><td></td><td></td><td>CLIP-T↑ CLIP-I↑ DreamSim↓ Aesthetic↑ ImageReward↑</td><td></td><td></td><td></td><td></td></tr><tr><td>SDXL (Base)</td><td>0.8971</td><td>0.8430</td><td>0.3124</td><td>6.1227</td><td>0.8289</td><td>0.9071</td><td>0.8376</td><td>0.3246</td><td>6.1508</td><td>1.1835</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Training-Based</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PhotoMaker [13] IP-Adapter [36]</td><td>0.8871 0.8771</td><td>0.8342 0.9398</td><td>0.3479</td><td>5.6051</td><td>0.8289</td><td>0.8776 0.8351</td><td>0.8300 0.9410</td><td>0.3594 0.1329</td><td>5.6222</td><td>0.8717</td></tr><tr><td colspan="9">0.1396 5.8172 0.8283</td><td>5.8367</td><td>0.5114</td></tr><tr><td>StoryDiffusion [40]</td><td></td><td></td><td></td><td></td><td>Training-Free</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.8985</td><td>0.8909</td><td>0.2284</td><td>6.0884</td><td>0.9579</td><td>0.8900</td><td>0.8960</td><td>0.2253</td><td>6.0926</td><td>1.1256</td></tr><tr><td>Consistory [30]</td><td>0.8868</td><td>0.8806</td><td>0.2355</td><td>6.1070</td><td>0.9814</td><td>0.8974</td><td>0.8730</td><td>0.2463</td><td>6.1588</td><td>1.1879</td></tr><tr><td>1P1S [17]</td><td>0.8871</td><td>0.8756</td><td>0.2132</td><td>6.0296</td><td>0.8289</td><td>0.8900</td><td>0.8679</td><td>0.2368</td><td>6.1106</td><td>0.9731</td></tr><tr><td>EOS-Story (Ours)</td><td>0.9042</td><td>0.8850</td><td>0.2005</td><td>6.1764</td><td>1.1680</td><td>0.8998</td><td>0.8787</td><td>0.2146</td><td>6.1791</td><td>1.3217</td></tr></table>

## 4.1 Main Experiments

Quantitative Result. We conduct comprehensive quantitative evaluations on Consistory+ and Consistory+\_+ in Tab. 1. Our method consistently achieves improved performance across nearly all metrics, especially showing notable gains in CLIP-T, verifying that Cross-Contextual Bridging efectively enhances text

alignment. While CLIP-I is slightly below the best, it remains competitive with the highest score and achieves the best DreamSim, demonstrating effective identity preservation. We also observe substantial improvements in image quality metrics (Aesthetic, ImageReward), confirming EOS-Story’s ability to maintain visual fidelity. For framing accuracy (Fig. 5), which evaluates whether the framing phrases are properly reflected, EOS-Story also

![](images/1705b1e81f391ca69fe93bd59a981aa845b557f79c167372bf12bdfbf07e6630.jpg)  
Fig. 5: Top-1 framing classification accuracy on Consistory+\_+. Evaluated with ShotVL-7B [15], EOS-Story efectively captures the intended framing phrases.

achieves the best results on Consistory+\_+. These findings further corroborate the efectiveness of our method for adaptively capturing scene variances.

Qualitative Result. We first compare the results in Fig. 6. Across diverse scenes, our model generates images that are both textually and visually wellaligned compared to prior methods. For instance, some models produce structurally fixed or repetitive outputs across varied scenes (Fig. 6a row 5), while others fail to reflect key attributes (e.g., underrepresenting subject details such as the “beard” of wizard in Fig. 6a). On Consistory+\_+, which includes various framing phrases (Fig. 7), our model successfully captures the intended framing while preserving both prompt and subject fidelity across scenes. Interestingly, under scenarios with extremely long prompts (Fig. 8), our model robustly adapts framing structures according to the various viewpoints, whereas prior methods often struggle to do so. These results demonstrate that EOS-Story efectively adapts to scene variance, ultimately producing enhanced visual storytelling outputs. Additional comparisons and results are in the supplementary materials.

![](images/d6a6595ee3404fc3c0ab66628518ba0534e533ff69733a65923ca159f8621d92.jpg)  
Fig. 6: Image comparison with previous methods. Our method faithfully reflects the given scenario while showing a flexible subject in each image.

## 4.2 Efectiveness of Cross-Contextual Bridging

Building upon the empirical observation in Sec. 3.1 that the ⟨EOS⟩ embedding acts as a global semantic control point, we conduct a quantitative evaluation in addition to providing qualitative samples. As shown in Tab. 2, injecting appearance information solely into the subject token fails to reliably encode identity cues. Distributing the same information across all non-EOS tokens partially improves alignment, yet the gains remain limited, suggesting that localized tokenlevel injection struggles to propagate appearance semantics consistently during generation. In contrast, modifying only the ⟨EOS⟩ embedding consistently achieves the highest text–image alignment and perceptual similarity. These results indicate that semantic control is more efectively exercised at the EOS level, where global context is integrated, rather than at individual word tokens that contribute only localized representations.

![](images/a86f59e1eab4a6b4ac72733d33a893c71e2b27dee4a969fc296f8793442ec571.jpg)

![](images/155faaee10ac33af2bd09db18a278f47424961edf630c417166bacb743baf91c.jpg)  
Fig. 7: Image diference with various framing phrases. We show a line of sto ryboard from our Consistory+\_+.  
Fig. 8: Viewpoint adaptation comparison. Various viewpoints are appended to the end of the same scene.

To further analyze the design choice of our residual formulation, we compare $e o s _ { i n f }$ with $e o s _ { o r g } .$ , where $e o s _ { o r g }$ is obtained by holistically encoding the full original prompt (style + subject + appearance + scene). We further consider two alternative constructions: (1) directly adding $e o s _ { i d }$ to $e o s _ { s c } .$ and (2) an orthogonal-based method following [2], where we construct an orthogonalized appearance component $\begin{array} { r } { \Delta _ { a p p } ^ { o r t h } = e o s _ { i d } - \frac { \mathsf { \bar { e } } o s _ { i d } \cdot e o s _ { s t y } } { | | e o s _ { s t y } | | ^ { 2 } } e o s _ { s t y } } \end{array}$ . We begin by constructing two semantic subspaces using PCA, based on the ⟨EOS⟩ embeddings obtained from appearance and scene phrases in the Consistory+ benchmark.

To ensure reliability, we exclude samples lacking explicit appearance descriptions (e.g., prompts specifying only a subject without appearance descriptors). Next, we project each EOS embedding onto these subspaces and compute the average cosine similarity to quantify alignment with each semantic axis. As shown in Tab. 3, directly adding $e o s _ { i d }$ degrades alignment, likely due to redundant semantic components that disrupt scene coherence. The orthogonal variant partially alleviates this issue, but still falls short of containing both appearance and scene semantics. In contrast, our residual formulation $( e o s _ { s c } + \varDelta _ { a p p } )$ achieves the highest alignment across both semantic subspaces and CLIP-T. Taken together,

Table 2: Various position for injecting information. Injecting solely on EOS token shows the best score.  
Table 3: Subspace alignment of $e o s _ { i n f } .$ Alignment scores on appearance/scene subspaces and CLIP-T for selected EOS variants.
<table><tr><td></td><td>CLIP-T↑</td><td>CLIP-I↑</td><td>DreamSim↓</td></tr><tr><td>Baseline (SDXL)</td><td>0.8971</td><td>0.8430</td><td>0.3124</td></tr><tr><td>Subject token</td><td>0.8909</td><td>0.8782</td><td>0.2341</td></tr><tr><td>All w/o EOS</td><td>0.8990</td><td>0.8823</td><td>0.2150</td></tr><tr><td>Only EOS (Ours)</td><td>0.9042</td><td>0.8850</td><td>0.2005</td></tr></table>

<table><tr><td></td><td>Appearance↑</td><td>Scene↑</td><td>CLIP-T↑</td></tr><tr><td> $\overline { { ( 1 ) ~ e o s _ { o r g } } }$ </td><td>0.555</td><td>0.692</td><td>0.8935</td></tr><tr><td>(2)  $e o s _ { s c } + e o s _ { i d }$ </td><td>0.519</td><td>0.655</td><td>0.8925</td></tr><tr><td>(3)  $e o s _ { s c } + \varDelta _ { a p p } ^ { o r t h }$ </td><td>0.552</td><td>0.694</td><td>0.9033</td></tr><tr><td> $\underline { { e o s _ { s c } + \varDelta _ { a p p } \ ( e o s _ { i n f } ) } }$ </td><td>0.564</td><td>0.706</td><td>0.9042</td></tr></table>

these results indicate that residual appearance injection at the EOS token helps efectively reflect the intended semantics during generation. To qualitatively support this analysis, Fig. 9 visualizes cross-attention maps. Compared to $e o s _ { o r g } .$ , which exhibits dispersed attention over irrelevant regions, $e o s _ { i n f }$ concentrates attention on semantically relevant areas, such as correctly attending to the specific details (e.g., “green knitted hat”). This focused attention

![](images/08a13a523abb732d9100775bf06f4814dd2aca30c9a907e2ca7d5fcfdc5112f1.jpg)  
Fig. 9: Attention sample visualization. $e o s _ { i n f }$ attends more strongly to semantically relevant regions.

pattern suggests that our EOS residual formulation helps mitigate semantic dilution, enabling the model to selectively attend to scene-specific cues during generation. Additional analyses are in the supplementary materials.

## 4.3 Ablation Studies

We conduct ablation studies on Consistory+. Additional experimental results are provided in the supplementary material.

Component Ablation. In Tab. 4, we report score variations across diferent component configurations. Applying Cross-Contextual Bridging (CCB) consistently improves CLIP-T, demonstrating enhanced text alignment. This is further supported by Fig. 10a, where intended actions, objects, or situations are more accurately depicted. Notably, improvements from CCB are primarily reflected in text-related metrics, whereas Adaptive Identity Preservation contributes more prominently to identity-related scores such as CLIP-I. This complementary trend indicates that each module efectively targets its intended objective, enabling more balanced control over textual fidelity and identity consistency. While configuration (7) achieves higher CLIP-I, our full method, EOS-Story, outperforms in both CLIP-T and DreamSim. We also compare results with and without subject size matching (Fig. 10b, Tab. 5). Without size aligning, scale discrepancies between identity and scene representations often lead to misaligned identity transfer, resulting in distorted subject proportions. In contrast, size-aware alignment ensures that identity features are referenced at an appropriate spatial scale, promoting more coherent subject integration.

![](images/9d5222a8b078989d643b19a0cf380bfbc27d4316e723a7fb7531ca7ec3a8c8cc.jpg)

Table 4: Score diference across components. Combining all components efectively balances subject and prompt coherence.
<table><tr><td></td><td>CCB IA</td><td>IASA</td><td>CLIP-T</td><td>CLIP-I</td><td>DreamSim</td></tr><tr><td>SDXL (2) (3) (4)</td><td rowspan="5">J</td><td rowspan="5"></td><td>0.8971 0.8974</td><td>0.8430 0.8290</td><td>0.3124 0.3067</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.8924</td><td>0.8729</td><td>0.2348</td></tr><tr><td></td><td>0.8924</td><td>0.8730</td><td>0.2348</td></tr><tr><td>(5)</td><td>0.9037</td><td>0.8743</td><td>0.2164</td></tr><tr><td>(6)</td><td>√</td><td>L</td><td>0.9037</td><td>0.8742</td><td>0.2167</td></tr><tr><td>(7)</td><td></td><td>√</td><td>0.8935</td><td>0.8924</td><td>0.2105</td></tr><tr><td>Ours</td><td>V</td><td>√</td><td></td><td>0.9042 0.8850</td><td>0.2005</td></tr></table>

Fig. 10: Component ablation results. CCB improves alignment, while removing size matching causes structural errors.  
Table 5: Efect of size matching. Size matching promotes structural coherence and improves image quality.
<table><tr><td></td><td>CLIP-I</td><td>Aesthetic</td><td>ImageReward</td></tr><tr><td>W o match</td><td>0.8842</td><td>6.1749</td><td>1.1618</td></tr><tr><td>W match</td><td>0.8850</td><td>6.1764</td><td>1.1680</td></tr></table>

## 5 Conclusion

In this work, we propose EOS-Story, a training-free framework that adaptively models scene-specific variance in visual storytelling across text and vision spaces. We introduce Cross-Contextual Bridging, which decouples the narrative into identity and scene sub-prompts and bridges the resulting semantic gap by injecting phrase-level appearance residuals into the scene’s ⟨EOS⟩ embedding. To complement this with structural adaptation, we further propose Adaptive Identity Preservation, which anchors subject features at the scene-specific scale and refines integration through Identity-Aware Self-Attention. To evaluate scene adaptability, we extend Consistory+ with diverse framing variations (Consistory+\_+). Extensive experiments show that EOS-Story achieves competitive or superior performance in both prompt fidelity and structural adaptability.

Acknowledgements. This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. RS-2020-II201373, Artificial Intelligence Graduate School Program (Hanyang University) and No. RS-2025- 02219062, Self-training framework for VLM-based defect detection and explanation model in manufacturing process).

## References

1. Avrahami, O., Hertz, A., Vinker, Y., Arar, M., Fruchter, S., Fried, O., Cohen-Or, D., Lischinski, D.: The chosen one: Consistent characters in text-to-image difusion models. In: ACM SIGGRAPH 2024 conference papers. pp. 1–12 (2024)

2. Dalva, Y., Venkatesh, K., Yanardag, P.: Fluxspace: Disentangled semantic editing in rectified flow transformers. arXiv preprint arXiv:2412.09611 (2024)

3. Ding, Y., Zhang, L.L., Zhang, C., Xu, Y., Shang, N., Xu, J., Yang, F., Yang, M.: Longrope: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753 (2024)

4. discus0434: Aesthetic predictor v2.5. https : / / github . com / discus0434 / aesthetic-predictor-v2-5 (2024), gitHub repository, accessed 2025-09-28

5. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)

6. Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., Isola, P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344 (2023)

7. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)

8. Gao, Z., Zhu, B., Yao, L., Yang, J., Tai, Y.: Subject-consistent and pose-diverse text-to-image generation. arXiv preprint arXiv:2507.08396 (2025)

9. Hu, T., Li, L., van de Weijer, J., Gao, H., Shahbaz Khan, F., Yang, J., Cheng, M.M., Wang, K., Wang, Y.: Token merging for training-free semantic binding in text-to-image synthesis. Advances in Neural Information Processing Systems 37, 137646–137672 (2024)

10. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)

11. Li, D., Li, J., Hoi, S.: Blip-difusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems 36, 30146–30166 (2023)

12. Li, S., Van de Weijer, J., Khan, F., Hou, Q., Wang, Y., et al.: Get what you want, not what you don’t: Image content suppression for text-to-image difusion models. In: International Conference on Learning Representations. vol. 2024, pp. 3182–3208 (2024)

13. Li, Z., Cao, M., Wang, X., Qi, Z., Cheng, M.M., Shan, Y.: Photomaker: Customizing realistic human photos via stacked id embedding. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8640–8650 (2024)

14. Lian, L., Li, B., Yala, A., Darrell, T.: Llm-grounded difusion: Enhancing prompt understanding of text-to-image difusion models with large language models. arXiv preprint arXiv:2305.13655 (2023)

15. Liu, H., He, J., Jin, Y., Zheng, D., Dong, Y., Zhang, F., Huang, Z., He, Y., Li, Y., Chen, W., et al.: Shotbench: Expert-level cinematic understanding in visionlanguage models. arXiv preprint arXiv:2506.21356 (2025)

16. Liu, L., Du, C., Pang, T., Wang, Z., Li, C., Xu, D.: Improving long-text alignment for text-to-image difusion models. arXiv preprint arXiv:2410.11817 (2024)

17. Liu, T., Wang, K., Li, S., van de Weijer, J., Khan, F.S., Yang, S., Wang, Y., Yang, J., Cheng, M.M.: One-prompt-one-story: Free-lunch consistent text-to-image generation using a single prompt. arXiv preprint arXiv:2501.13554 (2025)

18. Lu, Z., Wang, Z., Huang, D., Wu, C., Liu, X., Ouyang, W., Bai, L.: Fit: Flexible vision transformer for difusion model. arXiv preprint arXiv:2402.12376 (2024)

19. Mao, J., Huang, X., Xie, Y., Chang, Y., Hui, M., Xu, B., Zhou, Y.: Story-adapter: A training-free iterative framework for long story visualization. arXiv preprint arXiv:2410.06244 (2024)

20. Mun, S., Nam, J., Cho, S., Ok, J.: Addressing attribute leakages in difusion-based image editing without training. arXiv preprint arXiv:2412.04715 (2024)

21. Otsu, N., et al.: A threshold selection method from gray-level histograms. Automatica 11(285-296), 23–27 (1975)

22. Phung, Q., Mai, L., Heilbron, F.D.C., Liu, F., Huang, J.B., Ham, C.: Cineverse: Consistent keyframe synthesis for cinematic scene composition. arXiv preprint arXiv:2504.19894 (2025)

23. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)

24. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

25. Rahman, T., Lee, H.Y., Ren, J., Tulyakov, S., Mahajan, S., Sigal, L.: Make-astory: Visual memory conditioned consistent story generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2493– 2502 (2023)

26. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

27. Rooney, B., Bálint, K.E.: Watching more closely: Shot scale afects film viewers’ theory of mind tendency but not ability. Frontiers in Psychology 8 (2018), https: //api.semanticscholar.org/CorpusID:3754212

28. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22500–22510 (2023)

29. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image difusion. Advances in Neural Information Processing Systems 36, 1363–1389 (2023)

30. Tewel, Y., Kaduri, O., Gal, R., Kasten, Y., Wolf, L., Chechik, G., Atzmon, Y.: Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG) 43(4), 1–18 (2024)

31. Toker, M., Galil, I., Orgad, H., Gal, R., Tewel, Y., Chechik, G., Belinkov, Y.: Padding tone: A mechanistic analysis of padding tokens in t2i models. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 7618–7632 (2025)

32. Trager, M., Perera, P., Zancato, L., Achille, A., Bhatia, P., Soatto, S.: Linear spaces of meanings: compositional structures in vision-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15395–15404 (2023)

33. Wang, J., Yan, C., Lin, H., Zhang, W., Wang, M., Gong, T., Dai, G., Sun, H.: Oneactor: Consistent subject generation via cluster-conditioned guidance. Advances in Neural Information Processing Systems 37, 21502–21536 (2024)

34. Wang, M., Ding, H., Peng, J., Zhao, Y., Chen, Y., Wei, Y.: Characonsist: Finegrained consistent character generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16058–16067 (October 2025)

35. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)

36. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image difusion models. arXiv preprint arXiv:2308.06721 (2023)

37. Yi, M., Li, A., Xin, Y., Li, Z.: Towards understanding the working mechanism of text-to-image difusion model. Advances in Neural Information Processing Systems 37, 55342–55369 (2024)

38. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)

39. Zhengwentai, S.: clip-score: CLIP Score for PyTorch. https://github.com/ taited/clip-score (March 2023), version 0.2.1

40. Zhou, Y., Zhou, D., Cheng, M.M., Feng, J., Hou, Q.: Storydifusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems 37, 110315–110340 (2024)

## Supplementary Materials Not Just a Subject: Capturing Adaptive Scene-Specific Variance for Visual Storytelling

![](images/468e07dd277e59f0e2a6d5f030d3a5c2635726d7bb841504bc0aa6933b9e273b.jpg)  
Fig. 11: Empirical analysis of 〈EOS〉 linearity. (Parallelism) $\varDelta _ { a p p }$ from the same attribute stays aligned across subjects but separates across attributes, indicating a subject-agnostic direction. (Transferability) A transferred residual matches the explicit-prompt embedding (transfer target), unlike the base or a random attribute. (Bottom) Increasing λ in progressively strengthens the target appearance (round glasses), confirming controllable influence.

## A EOS Analysis

Empirical Analysis of EOS Linearity. Although prior work has shown that CLIP text embeddings often exhibit approximate linear compositionality over semantic factors [32], it is non-trivial whether the same property holds at the ⟨EOS⟩ hidden state, which aggregates contextual information through causal attention rather than being a simple sum of word embeddings. Rather than assuming exact linearity, we empirically examine whether the appearance residual $\Delta _ { a p p } = e o s _ { i d } - e o s _ { s t y }$ behaves as a transferable semantic direction. To this end, we construct a large prompt set spanning diverse subjects, styles, and scenes, with appearance attributes of varying complexity—from short phrases to long relational descriptions (Fig. 11).

![](images/e1cb52995c17ca62653441d1a4c2c6bb4df20dcd4febbd44e1295d7ee67dc918.jpg)

![](images/0fd5aa8d7e9693f072de86ee2d303914efd0d403a9ec3441e20e25b539968846.jpg)  
Fig. 12: Cumulative explained variance for scene and appearance subspaces.

![](images/d11c60bbcda697bc1ef1c9b20a6550cb3ec168b7bfc06c83db1a80e3a29708a3.jpg)  
Fig. 13: Additional cross-attention map comparison between ${ e o s } _ { o r g }$ and $e o s _ { i n f }$

Parallelism. We first test whether $\varDelta _ { a p p }$ encodes appearance independently of the subject it was extracted from. Residuals extracted from the same attribute but diferent subjects remain highly aligned, whereas residuals from diferent attributes are clearly separated. This indicates that $\varDelta _ { a p p }$ captures a subjectagnostic appearance direction rather than subject-specific content.

Transferability. We then add a $\varDelta _ { a p p }$ extracted from one subject to the ⟨EOS⟩ of a diferent prompt which only contains subject and scene descriptions, and compare it against the embedding of an explicit prompt (subject + appearance + scene) that states the attribute in words (transfer target). The two are highly similar, while both the appearance-free base and a mismatched random attribute are clearly distinguished. Notably, this holds even for long, compositional phrases, indicating that EOS-level residuals transfer rich appearance semantics—not merely single words—in a linear manner.

Generative efect. Finally, scaling the injected residual as $e o s _ { s c } + \lambda \cdot \varDelta _ { a p p }$ progressively strengthens the target appearance in the generated image (Fig. 11, bottom): as λ increases, the attribute (e.g., round glasses) emerges and intensifies, confirming that the residual exerts a controllable, monotonic influence on generation. Together, these results show that $\varDelta _ { a p p }$ functions as a subject-agnostic, transferable, and controllable appearance direction—justifying our EOS-level residual manipulation.

![](images/9b760e440af393f907cebb90c803ea9046256c27683c0464cf5e182531da39b0.jpg)  
Fig. 14: Cosine similarity distribution between ${ e o s } _ { o r g }$ and $e o s _ { i n f } .$

![](images/06308b780fc4b441422bffb8c7109105a411a0f26c390376c4d1725b1acdae48.jpg)  
Fig. 15: t-SNE plot of ${ e o s } _ { o r g }$ and $e o s _ { i n f } .$ Red and green arrows show the position of $e o s _ { i n f }$ from its corresponding $e o s _ { o r g } .$

Subspace Construction. To compute alignment with appearance and scene subspaces, we first collect all appearance-and scene-related phrases from the Consistory+ benchmark. To verify whether a meaningful subspace can be constructed, we examine the cumulative explained variance from the PCA of each set. As illustrated in Fig. 12, capturing 90% of the cumulative variance requires only 81 dimensions for appearance and 221 for scene, out of the total 768 dimensions in the CLIP text embedding space.

Additional Analysis on $e o s _ { i n f }$ . While $e o s _ { i n f }$ exhibits better alignment with semantic subspaces (Tab. 3), such alignment does not inherently guarantee semantic preservation—unintended deviations from the original meaning may degrade generation quality. To investigate this, we compute the pairwise cosine similarity between $e o s _ { o r g }$ and $e o s _ { i n f }$ across samples (Fig. 14), under the hypothesis that significant semantic shifts would result in low similarity. However, the similarity remains consistently high $( \mu = 0 . 9 2 6 0 )$ , suggesting that Cross-Contextual Bridging maintains core semantic content. We further support this with a t-SNE visualization (Fig. 15), where $e o s _ { i n f }$ embeddings consistently cluster near their $e o s _ { o r g }$ counterparts. Together with qualitative examples (Fig. 13) showing improved attention localization, these results suggest that the infused ⟨EOS⟩ embedding efectively combines essential semantics while remaining close to the original semantic configuration.

Text Encoder Robustness. SDXL employs two types of CLIP text encoders: one with a 768 dimensional output and another with 1024 dimensions. To examine whether the larger encoder also exhibits similar linear behavior, we measure the pairwise cosine similarity between $e o s _ { o r g }$ and $e o s _ { i n f }$ using the 1024 dimensional encoder. As shown in bottom of Fig. 14, the similarity remains consistently high $( \mu = 0 . 9 4 0 7 )$ , indicating that the Cross-Contextual Bridging mechanism can be used across diferent text encoder configurations.

Table 6: $e o s _ { i n f }$ construction ablation. We analyze text-alignment (CLIP-T) by varying (A) base EOS choice, (B) scaling factor of the injected direction, and (C) choice of appearance direction to inject.
<table><tr><td rowspan=1 colspan=1>CLIP-T</td></tr><tr><td rowspan=1 colspan=1>(A) Baselines /naive combinations</td></tr><tr><td rowspan=1 colspan=1> ${ e o s } _ { o r g }$                             0.8935 $e o s _ { s c } ^ { o r j } + e o s _ { a p p } ^ { o r g } + e o s _ { s t y } ^ { o r g }$         0.8625 $e o s _ { s c } + e o s _ { i d }$                      0.8925</td></tr><tr><td rowspan=1 colspan=1>(B) Mixing-weight ablation (hyperparameter)</td></tr><tr><td rowspan=1 colspan=1> $0 . 5 e o s _ { s c } + 0 . 5 e o s _ { i d }$              0.8864 $e o s _ { s c } + 0 . 5 \ : \varDelta _ { a p p }$                  0.8995 $e o s _ { s c } + 0 . 5 \ : \Delta _ { a p p } ^ { o r t h }$                  0.8992 $e o s _ { s c } + 1 . 5 \varDelta _ { a p p }$                  0.9040 $e o s _ { s c } + 1 . 5 \ : \Delta _ { a p p } ^ { o r t h }$                  0.9038</td></tr><tr><td rowspan=1 colspan=1>(C) Direction choice ablation</td></tr><tr><td rowspan=1 colspan=1> $\varDelta _ { a p p }$                               0.8726 $e o s _ { i d } + \Delta _ { s c }$                       0.8865 $e o s _ { s c } + \varDelta _ { a p p } ^ { o r t h }$                     0.9033</td></tr><tr><td rowspan=1 colspan=1> $\overline { { e o s _ { s c } + \Delta _ { a p p } \ ( \mathrm { O u r s } ) } }$             0.9042</td></tr></table>

Additional Variants of the EOS-Based Control. We further analyze variations of EOS-based semantic control in Tab. 6. Beyond the main variant presented in the primary experiments (Tab. 3), we systematically examine diferent EOS construction strategies by varying the base EOS components, injected directions, and mixing weights. Specifically, $\varDelta _ { a p p } \ : \left( \mathrm { T a b . 6 C } \right)$ corresponds to directly replacing the EOS embedding with the appearance residual, following [9]. The configuration $e o s _ { s c } ^ { o r g } + e o s _ { a p p } ^ { o r g } + e o s _ { s t y } ^ { o r g }$ naively concatenates EOS embeddings obtained from each phrase-level component in the benchmark. The relatively lower performance of $\varDelta _ { a p p }$ can be attributed to the absence of scene information in the resulting embedding. While another variant $e o s _ { i d } + \Delta _ { s c }$ is mathematically similar to our additive formulation, it exhibits slightly reduced alignment. We hypothesize that this variant introduces multiple semantic directions originating from overlapping bases, whereas our formulation injects a consistent appearance residual $\varDelta _ { a p p }$ into multiple scene-grounded ⟨EOS⟩ anchors $\{ e o s _ { s c } ^ { n } \} _ { n = 1 } ^ { N } . \mathrm { B y }$ shifting several scene embeddings along a consistent semantic direction, our approach maintains a structured integration of narrative context and appearance cues. This parallel infusion results in more stable semantic control and ultimately improves text–image alignment compared to subject-anchored alternatives. Overall, these results indicate that simply replacing or naively combining EOS components does not suficiently preserve semantic grounding. In contrast, controlled injection of the appearance direction into a scene-aware EOS consistently improves text alignment, highlighting the importance of structured and context-preserving semantic modulation.

![](images/be00be5cfbc1ca8a53ba3921e5c1bebc4a3d9c2ba78c59e8a9d7eddbe7bbc996.jpg)

Table 7: Framing phrases in ShotBench
<table><tr><td>Shot Size</td><td>Wide Close up Extreme wide Medium close up</td><td>Lens Size</td><td>Long lens Wide Ultra wide (fisheye) Medium</td><td>Shot Framing</td><td>Single Insert 2 shot</td><td>Camera Movement Push in Pull out Boom up</td></tr><tr><td></td><td>Medium wide Medium Extreme close up</td><td>Lighting Type Daylight</td><td>Artificial light Mixed light</td><td></td><td>Group shot Establishing shot Over the shoulder 3 shot</td><td>Pan left Pan right Tilt down Tilt up</td></tr><tr><td>Camera Angle Aerial</td><td></td><td></td><td></td><td></td><td></td><td>Boom down</td></tr><tr><td></td><td></td><td></td><td></td><td>Lighting Condition Side light</td><td></td><td></td></tr><tr><td></td><td>Overhead</td><td></td><td>Firelight</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Overcast</td><td></td><td>Back light</td><td>Zoom in</td></tr><tr><td></td><td>Low angle</td><td></td><td>Practical light</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Sunny</td><td>High contrast</td><td></td><td>Static</td></tr><tr><td></td><td>High angle</td><td></td><td></td><td>Silhouette</td><td></td><td>Move to right</td></tr><tr><td></td><td>Dutch angle</td><td></td><td>Moonlight</td><td>Edge light</td><td></td><td>Move to left</td></tr><tr><td>Composition</td><td>Center</td><td></td><td>Fluorescent</td><td>Underlight</td><td></td><td>Zoom out</td></tr><tr><td></td><td>Balanced</td><td></td><td>HMI</td><td>Top light</td><td></td><td>Arc</td></tr><tr><td></td><td>Symmetrical</td><td></td><td>Tungsten</td><td>Hard light</td><td></td><td>Camera roll</td></tr><tr><td></td><td>Right heavy</td><td></td><td></td><td>Soft light</td><td></td><td>Dolly zoom</td></tr><tr><td></td><td>Left heavy</td><td>LED</td><td></td><td>Low contrast</td><td></td><td></td></tr><tr><td></td><td>Short side</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Fig. 16: Statistical analysis across scene structure categories. Each graph presents the mean and standard deviation of top-1 and top-3 accuracies for each option within a category. Results are averaged over 10 random seeds.

## B Consistory+\_+ & Framing Evaluation Details

The ShotBench [15] includes a wide range of framing phrases across eight categories (Tab. 7). To ensure a focused and rigorous evaluation of scene-specific structural variance, we intentionally restricted the scope to three primary categories, excluding the remaining five based on the following methodological criteria: (1) Prompt Redundancy: Shot Framing descriptors (e.g., “single”, “2 shot”) primarily specify the number of characters, which is already inherently defined by the subject phrases in the storyboard prompts; (2) Redundancy Mitigation: Lens Size phrases (e.g., “wide lens”) are largely redundant as they are semantically encompassed by the Shot Size category (“wide shot”); (3) Domain Incompatibility: Camera Movement is inherently applicable to temporal video sequences and is thus irrelevant to static image generation; and (4) Task Relevance: Lightingrelated categories fall outside the scope of our work, as they primarily focus on stylistic or artistic variations rather than the structural and compositional elements that define scene-specific variance.

Framing Phrase Filtering. Although SDXL generally produces textually wellaligned images, it sometimes struggles to accurately reflect specific framing phrases. Before incorporating these phrases into our benchmark extension, we first evaluate how well they are captured by the base SDXL model using the

ShotBench evaluation mechanism. To quantitatively assess phrase reflection, we construct approximately 90 diverse scene prompts and append each framing phrase at the end (e.g., “A photo of a woman baking cookies, close-up shot”). We generate images for 10 random seeds per phrase and compute the top-1 and top-3 accuracy, along with standard deviations, as shown in Fig. 16. For example, in the “shot size” category with 7 phrase options, this results in a total of 6,300 images (90 scenes × 7 phrases × 10 seeds). We filter out framing phrases whose top-1 accuracy is zero (e.g., “extreme wide shot”), as these are not reliably reflected in the generated outputs from SDXL.

Consistory+\_+ With the remaining framing phrases, we extend the original Consistory+ benchmark using GPT-4o. Since these phrases are visually meaningful primarily for animate subjects, we first guide GPT-4o to determine whether the subject in each prompt is animate. Then, the model is instructed to append the most appropriate framing phrase for the given scene, resulting in Consistory+\_+.

Framing Accuracy Evaluation. Following the evaluation protocol in [15], we provide the generated image along with the corresponding framing category (e.g., shot size) and prompt the model with a multiple-choice question. For instance, for shot size evaluation, we prompt the model with a multiple-choice question such as “What is the shot size of this shot?” and provide predefined options (e.g., wide shot, medium shot, close-up shot). The predicted category is compared against the framing phrase specified in the text prompt, and accuracy is reported.

## C Additional Experiments

## C.1 Ablation Studies

Top-α and Blending Scale λ. Fig. 18 shows that as λ decreases (thereby injecting more subject-relevant features), the generated results better capture the identity characteristics (e.g., beard of a man). In Tab. 8, we report the performance variations with respect to diferent hyperparameter settings. We set $\alpha = \mu + \sigma$ and λ = 0.55 based on the best trade-of observed between the CLIP-T and CLIP-I score. Furthermore, Fig. 17 illustrates the efect of similarity filtering (Sec. 3.3). While a fixed proportion α can preserve identity, varying similarity distributions across scenes may cause noisy or discarded correspondences, leading to artifacts such as duplicated features. In contrast, our adaptive threshold derives scene-specific criteria from the similarity distribution, enabling more stable identity integration without such artifacts.

EOS Change Range. Since text embeddings include multiple ⟨EOS⟩ tokens after the sentence ends, we investigate the efect of the substitution range in Tab. 9. The results show that replacing all ⟨EOS⟩ tokens often degrades image quality and leads to unnatural scene rendering (Fig. 19), likely due to redundant perturbations introduced when modifying padded EOS positions.

Layer Application Variation. We apply Identity Anchoring (IA) and Identity-Aware Self-Attention (IASA) at diferent layers of the unet decoder in the difusion model. While prior works typically transfer identity-related features within the decoder (up-block) layers, we further investigate the impact of layer selection. As shown in Fig. 20, we vary the application layers and plot the resulting CLIP-T and CLIP-I scores. Interestingly, when IA and IASA are applied to configurations that include the up-block layers, the CLIP-I score notably improves (surpassing 0.880). These results empirically highlight the importance of the decoder layers (up-blocks) in efectively conveying identity features and enhancing the image quality during generation.

Identity  
� = 0.3  
� = 0.5  
� = 0.7  
Adaptive � (Ours)  
![](images/00fbc374c16ebb27a4831c830d416da75cd6544dfda335b12dcd646151efefa3.jpg)  
Fig. 17: Efectiveness of adaptive thresholding. While diferent scenes require distinct α values (red), our adaptive threshold (µ + σ) provides reliable results by dynamically adjusting to each scene (blue).

![](images/0a8eb93a34482640ad8b7db0cc55c4f450bdbe8d57d107927f6a42594e054955.jpg)

Table 8: Ablation of α and blending scale λ
<table><tr><td>α</td><td>CLIP-T CLIP-I</td><td>λ</td><td>CLIP-T</td><td>CLIP-I</td></tr><tr><td>Baseline</td><td>0.8971</td><td>0.8430</td><td>-</td><td></td></tr><tr><td>0.1</td><td>0.9033</td><td>0.8841</td><td>0.45</td><td>0.9027 0.8851</td></tr><tr><td>0.2</td><td>0.9036</td><td>0.8836 0.5</td><td>0.9033</td><td>0.8843</td></tr><tr><td>0.3</td><td>0.9039</td><td>0.8848</td><td>0.55 0.9042</td><td>0.8850</td></tr><tr><td>0.4</td><td>0.9040</td><td>0.8852</td><td>0.6</td><td>0.9044 0.8837</td></tr><tr><td>0.5</td><td>0.9033</td><td>0.8860</td><td>0.65 0.9034</td><td>0.8797</td></tr><tr><td>0.6</td><td>0.9026</td><td>0.8864</td><td>0.7 0.9035</td><td>0.8798</td></tr><tr><td>0.7</td><td>0.9022</td><td>0.8871</td><td>0.75</td><td>0.9038 0.8787</td></tr><tr><td>µ + σ</td><td>0.9042</td><td>0.8850</td><td>0.8</td><td>0.9044 0.8769</td></tr></table>

Fig. 18: Visualization of λ sensitivity. As λ increases, the degree of subject identity preservation decreases.

Table 9: EOS substitution range. We compare the result when change all the remaining EOS embeddings in text embeddings.
<table><tr><td></td><td>All First (Ours)</td></tr><tr><td>CLIP-I</td><td>0.8741 0.8850</td></tr><tr><td>Dreamsim</td><td>0.2185 0.2005</td></tr><tr><td>Aesthetic</td><td>6.1111 6.1764</td></tr></table>

![](images/3677bb7167c16bdac63787b05a8f97b80572a7a8cffb1e4cedf112f935dccc82.jpg)  
Fig. 19: Image comparison of EOS substitution scope. Full substitution often leads to misaligned generations, resulting in unnatural scene rendering.

![](images/2183dbf7d934c8b3c452d57da852491390f4576a8be55ddff6b11325cbb9c460.jpg)

![](images/2252f8822d43b2d793c8b6ada5a41380a3f4cb0e4828da485b97d7aeb0cba716.jpg)  
Fig. 20: Score diference across injection layers. We visualize the score changes when applying IA and IASA at diferent layers. Higher values toward the top-right indicate better overall performance.  
Fig. 21: Performance trade-of with switching step. Switching at t = 9 effectively balances CLIP-T and CLIP-I scores.

Switching Timestep in AIP. Prior research indicates that the global layout of generated images is primarily established during the initial denoising stages. Based on this observation, we evaluate the performance trade-ofs by varying the switching timestep between Identity Anchoring (IA) and Identity-Aware Self-Attention (IASA). As illustrated in Fig. 21, transitioning at t = 9 efectively balances prompt alignment (CLIP-T) and identity consistency (CLIP-I), ensuring both structural flexibility and identity preservation.

## C.2 Extended Experiments

Model Variation. To evaluate the generalizability of our method across different model variants based on the SDXL architecture, we apply EOS-Story to several pretrained weights: Juggernaut-X-v10, Playground-v2.5, and RealVisXL. We generate images both with and without framing phrases in Fig. 22. Although the influence of framing phrases may appear subtle in some cases—particularly when the original scene already shares similar structural layouts—the generated images consistently reflect the intended semantics.

A hyper-realistic digital painting of a teenage girl with blonde hai

![](images/1d1fc03d2b562a49fc26fffa7de3dbb26f13ef84d942b689205c0dcae11146e3.jpg)

![](images/252644b922e688de3443d1669ec4428b1f3617390bbb7a44ee6a7a80573d4e7d.jpg)  
Fig. 23: Image results with Control-Net. Our model can be integrated with various conditioning models, such as ControlNet.

Fig. 22: Results on diferent model weights. For each row in every model, framing phrases are added to the end of prompts.  
Table 10: Score on Consistory+ on SD3.
<table><tr><td></td><td>CLIP-T CLIP-I Dreamsim</td></tr><tr><td>SD3</td><td>0.8786 0.8517 0.3076</td></tr><tr><td>Ours (SD3)</td><td>0.8978 0.8728 0.2571</td></tr></table>

ControlNet Adaptation. Our model can also be adapted to work with various pretrained conditioning models such as ControlNet. As shown in Fig. 23, integrating our method with ControlNet enables the generation of subject-coherent image sequences with diverse poses.

Free-form Prompt. Following prior works, our model adopts the structured prompt format used in the Consistory+ benchmark. However, in practical

form natural language prompts instead of carefully formatted inputs. To bridge this gap, we additionally employ GPT-4o as a prompt parser that converts free-form user instructions into the intended prompt format for EOS-Story, thereby enabling our method to support more flexible user inputs, as shown in Fig. 24 and Tab. 11.

![](images/756fb7f728790d4f9ed1682e2f53cb2bc37f906c3cae6da79dd4ccaae6382af7.jpg)  
A dark enchanting artwork of a mysterious witch in flowing robes flying on a broomstick under a full moon. Next, she is reading incantations from an ancient book. After that, she is casting spells in a candlelit cabin. Later, she gathers herbs in a misty forest.  
Fig. 24: Free-form prompt generation results. Our model can handle free-form user prompts by parsing them into the intended EOS-Story format with an LLM.

SD3 Applicability. We use the Stable Difusion 3.5 Large weights with a CFG scale of 4.5 and 28 sampling steps. In SDXL, identity information is incorporated via extended attention, where the subject key and value embeddings are concatenated with those of the scene. However, in DiT-based models, fixed positional encodings and sequence-length constraints make such direct concatenation non-trivial [3,18]. To address this architectural diference, we compute attention along two parallel paths: (i) the original scene query–key–value path and (ii) a subject-referenced path using the scene query with subject key and value embeddings. The resulting attention outputs are then combined, enabling implicit identity referencing without modifying the positional encoding structure. Despite the architectural adaptation, the core semantic control mechanism remains intact. As shown in Fig. 25 and Tab. 10, our method consistently improves alignment, demonstrating its applicability to DiT-based difusion models.

Table 11: Quantitative results on free-form story generation.
<table><tr><td>Method</td><td colspan="5">CLIP-T↑ CLIP-I↑ DreamSim↓ Aesthetic↑ ImageReward↑</td></tr><tr><td>SDXL (Base)</td><td>0.8971</td><td>0.8430</td><td>0.3124</td><td>6.1227</td><td>0.8289</td></tr><tr><td>EOS-Story (Ours)</td><td>0.9042</td><td>0.8850</td><td>0.2005</td><td>6.1764</td><td>1.1680</td></tr><tr><td>EOS-Story (Free-form)</td><td>0.9040</td><td>0.8848</td><td>0.2001</td><td>6.1867</td><td>1.1712</td></tr></table>

![](images/557c77fde7c7d488e9b774f6dc196c5f458b79695a8d828f19894a37bb427dc9.jpg)  
Fig. 25: EOS-Story results with SD3 [5].

FLUX Applicability. Although SD3 also adopts MM-DiT blocks, we further extend our method to FLUX, as many recent models are built upon it. Since FLUX relies solely on a T5 encoder—unlike the CLIP encoder used in the SD series—the 〈EOS〉 token acts merely as an end-of-sentence marker with negligible influence on generation. We therefore apply only the AIP module to FLUX, which is model-agnostic by design.

<table><tr><td></td><td>CLIP-T↑</td><td>CLIP-I↑</td><td></td><td>DreamSim↓ ImageReward↑</td><td>Aesthetic↑</td></tr><tr><td>FLUX (baseline)</td><td>0.8696</td><td>0.8555</td><td>0.2445</td><td>1.2988</td><td>6.0466</td></tr><tr><td colspan="6">Large Pretrained</td></tr><tr><td>FLUX-Kontext</td><td>0.8544</td><td>0.9032</td><td>0.1602</td><td>1.2246</td><td>5.9489</td></tr><tr><td colspan="6">Training-Free</td></tr><tr><td>CharaConsist</td><td>0.8697</td><td>0.8636</td><td>0.2358</td><td>1.2774</td><td>6.0572</td></tr><tr><td>EOS-Story (FLUX)</td><td>0.8722</td><td>0.8712</td><td>0.2052</td><td>1.2673</td><td>6.0943</td></tr></table>

Table 12: Quantitative comparison under the FLUX backbone. Bold and underline denote the best and second-best.

![](images/c0c6fde29581a36011e1b84bff3799fdf481ae8f50a08f4e2cf8ee0b5780fef8.jpg)  
Fig. 26: EOS-Story on FLUX. Although the large pre-trained model (Flux-Kontext) better preserves the subject identity, it still struggles to reflect the given text (e.g., computer) and produces rigidly similar pose.

Recent large pretrained image editing model FLUX-Kontext exhibits strong identity-preservation capabilities and can be directly applied to visual storytelling. However, we observe that it remains anchored to the reference, producing copy-and-paste-style synthesis that fails to follow scene-specific prompts (Fig. 26, Tab. 12). As shown in Tab. 12, while it achieves strong identity scores (e.g., CLIP-I, DreamSim), its prompt alignment (CLIP-T) and overall quality remain limited, indicating that identity preservation alone is insuficient for storytelling—scene-specific adaptation is essential. We further compare with CharaConsist [34], a recent FLUX-based storytelling method. Since its evaluation benchmark is not publicly available, we compare on Consistory+ for a fair setting. EOS-Story achieves competitive or superior performance across both identity and prompt-alignment metrics, demonstrating that our scene-adaptive design transfers efectively to FLUX.

![](images/2e2883ec79955a775f39e3e03693780ed784bd5dc547185e73b857a3f8312fbe.jpg)  
Fig. 27: Image results on multi-characters.

Multi-Character Generation. By defining multiple characters, our model can be applied to generate images that preserve each subject’s identity. As shown in Fig. 27, it captures both the intended prompt and subject identities, resulting in coherent and visually consistent story sequences.

User Study & Computational Cost. We conducted a user study with 20 participants based on 40 example storyboards. For each storyboard, participants were asked to select the result that best balances identity consistency, prompt alignment, and scene adaptability among the four methods (Fig. 29). As shown in Fig. 28, EOS-Story was selected as the best in 52.8% of cases, outperforming the baselines. We report GPU memory usage and latency for generating 5 stories in Tab. 13. Note that StoryDifusion is measured at 768×768 resolution, and 1P1S processes images individually rather than in batches, resulting in lower GPU usage. While EOS-Story uses slightly more memory than the baselines, its latency remains comparable, indicating that the additional memory cost does not translate into a noticeable runtime overhead.

## C.3 Additional Qualitative Results

We present additional qualitative comparisons in Fig. 31 and Fig. 32. As shown, EOS-Story efectively preserves subject identity while achieving better alignment with the given prompts—even in complex scenes involving diverse framing phrases. Our method also supports generating storyboards of varying lengths (short, medium, long), as illustrated in Fig. 33 and Fig. 34, and Fig. 35. Furthermore, Fig. 36 demonstrates how our model adapts framing structures in response to diferent framing phrases.

## D Limitation & Future Work

Although our model efectively balances prompt and subject coherence, several limitations remain. While the cross-attention map generally captures subject regions accurately, it occasionally fails to attend to the correct area, leading to misaligned or partially misplaced generations—issues also observed in previous works. This can be handled by explicitly using the subject mask as an additional condition. In addition, there are cases where the model fails to generate accurate images at all (e.g., Fig. 30 “faun”), which appears to be an inherent limitation of the underlying SDXL backbone. Furthermore, since Cross-Contextual Bridging exploits the contextual summarization property of the CLIP ⟨EOS⟩ token, it does not directly transfer to models based on T5 or LLM-based text encoders, where the ⟨EOS⟩ token no longer serves a comparable role. Enabling analogous residual-based semantic composition on such encoders is a promising direction for extending our framework to a broader range of modern difusion models.

![](images/bfa856f27cff57d0ff003117d2cb037db805ae63052ef57c863ba1b31b696233.jpg)

Fig. 28: User study results. EOS-Story is most preferred (52.8%) for balancing identity, prompt alignment, and scene adaptability across 20 participants.  
![](images/9f78aab67a205825f5abd79872b096bb849525e75669429df5e0684862c88f6a.jpg)

<table><tr><td></td><td>GPU (MB)</td><td>Latency (s)</td></tr><tr><td>StoryDiffusion</td><td>46287</td><td>98.774</td></tr><tr><td>Consistory</td><td>43628</td><td>144.814</td></tr><tr><td>1P1S</td><td>17073</td><td>118.739</td></tr><tr><td>EOS-Story</td><td>47777</td><td>118.570</td></tr></table>

Table 13: Cost of generating 5 stories.

Fig. 29: User study example. Participants were shown four options (one per method) generated from the same storyboard, and asked to select the result that best balances identity, prompt alignment, and scene adaptability.  
![](images/80c81f5119677fc27dae8f14a60efcf5bf6de6aa35a606d84512a8a35b97dae5.jpg)  
Fig. 30: Failure cases. There are cases where the baseline model (SDXL) struggles to generate certain concepts (e.g., “faun”). Additionally, since our method—like prior works—leverages the cross-attention map as a mask, the resulting masks sometimes capture irrelevant regions.

Stomping through a Swinging a massive muddy swamp club in battle  
Served in a cozy mug by the fire  
Placed on a wooden tray with cookies  
![](images/79ee900afa017204cf6724ae3216b90ea4542f454288c220d5a0876f1b3b14c3.jpg)  
Fig. 31: Additional qualitative comparison.

![](images/819bead9d8819e4d62c893b618335a22f239e3ee9f208537e3c5e17363719fac.jpg)  
Fig. 32: Additional qualitative comparison on framing phrase adaptation.

![](images/d670e70488ffad01c3ed978c11b8b8613d38b0bb219026509d9b302b37252e87.jpg)  
Fig. 33: Medium length story generation.  
A heartwarming and cinematic scene of a space explorer in a sleek white space suit with glowing emblems

![](images/2ef1338e108b8b71fdf8d99891f9ee85bd618df9b07dcf72cc3a72128427b49e.jpg)  
A feminine Makoto Shinkai style anime illustration of a cute girl with blond hair and black glasses

Fig. 34: Short length story generation.

Making friendship bracelets at a table

“A photo of a teenage girl with blond hair”  
![](images/8a36cea3f353cd5ffa93bfd08c50cf6aaf7ca1df3eb4cb00984aa43384da4751.jpg)  
Fig. 35: Long story generation. Our model accurately reflects detailed scene descriptions while preserving subject identity, even in extended scenarios.

![](images/827978196cbc78e697640995caa8cad5c548a131a01867118b7d06eb962ba5d5.jpg)  
Fig. 36: Additional framing phrase adaptation results.