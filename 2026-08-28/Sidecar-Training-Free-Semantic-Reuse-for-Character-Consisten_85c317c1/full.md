# Sidecar: Training-Free Semantic Reuse for Character-Consistent Free-form Visual Storytelling

Sibo Dong Sarah Adel Bargal Department of Computer Science, Georgetown University, Washington, D.C., USA {sd1242, sarah.bargal}@georgetown.edu

## Abstract

Visual storytelling requires generating images that follow a narrative while preserving consistent character identities across frames. In free-form story generation, a character isfully described only when first introduced and is later referred to by a type-level mention or pronoun. Although this setting better reflects natural storytelling, laterprompts may omit important identity-related semantics, making character consistency more difficult to maintain. We propose Sidecar, a plug-and-play semantic augmentation module that preserves entity-level information from the initial description and injects the missing semantics into later prompt embeddings. Sidecar requires no additional training and does not modify the architecture of the base diffusion model. Experiments on FreeStoryBench show that Sidecar consistently improves prompt-image alignment and character consistency across multiple SDXL- and FLUX-based baselines, with negligible computational overhead.

## 1. Introduction

Visual storytelling aims to generate a sequence of images from a sequence of narrative prompts. Each image should accurately reflect its corresponding prompt, while the complete sequence should preserve consistent characters, objects, and visual appearance across frames. Compared with single-image text-to-image (T2I) generation, visual storytelling requires the model to satisfy both promptimage alignment and cross-frame consistency. However, pretrained T2I models [4, 10, 16, 19] typically generate each image independently and lack an explicit mechanism for preserving the cross-frame consistency.

Existing visual storytelling methods improve crossframe consistency by incorporating information from previous prompts or generated images [3, 5, 11, 14, 17, 20]. Several approaches introduce additional modules trained on story-level data to capture dependencies across frames [1, 7, 21, 22, 25, 27]. Although effective, these methods require task-specific training and additional computational resources. Training-free methods instead modify the inference process of pretrained T2I models [2, 12, 23, 24, 26] to preserve character consistency by reusing attention features or latent representations from previous generations.

Most training-free methods assume that the complete character description is repeated in every prompt. Repeated descriptions provide the text encoder with similar character semantics, which simplifies identity preservation. However, this prompt format differs from natural storytelling, where a character is generally described upon first appearance and later referred to by a name, a type, or a pronoun.

FreeStory [2] addresses this by introducing the freeform story generation setting. Under this formulation, the complete character description is included only in the first prompt, while subsequent prompts use shorter references such as a character type or a pronoun. For example, after introducing “a sleek red fox with a fluffy white-tipped tail and amber eyes,” later prompts may refer to the same character as “the fox” or “he.” This formulation more closely follows natural narrative structure and avoids repeatedly inserting the same description into every prompt. It also improves the readability and semantic continuity of the narrative. Moreover, shorter references reduce prompt length, which is particularly important for long or multi-character stories, where repeated descriptions may occupy a substantial portion of the text encoder’s maximum length.

Although FreeStory improves character consistency under free-form prompts, later references can still weaken the text condition. As shown in Figure 1, replacing a full character description with a type-level mention or pronoun removes explicit identity attributes, leaving later prompt embeddings semantically incomplete. Visual feature reuse [2, 12, 23, 24, 26] can transfer information from previous frames, but it does not directly restore the missing identity semantics in the current text condition.

To address this, we propose Sidecar, a plug-and-play semantic augmentation module for free-form visual storytelling. Sidecar preserves entity-level semantic information from the initial character description and injects the missing character semantics into the text representations of later prompts. Rather than replacing the original text embedding or appending the full description to every prompt, Sidecar provides an auxiliary semantic representation alongside the original text-encoding process. The original prompt specifies the action and scene, while Sidecar supplements the identity information omitted by the character reference. Sidecar requires no additional training and does not modify the architecture of the base diffusion model. It can therefore be integrated into different pretrained generation pipelines and combined with existing storytelling methods.

![](images/e0e844a20548ca53691e65ae9ebfb4a5910eca8bdc1302de9114bbd233ada39d.jpg)  
Figure 1. Identity drift under different character references in free-form story generation. Starting from the same initial character, we generate the next frame using a state-of-the-art free-form visual storytelling method, FreeStory [2], while keeping generation settings unchanged. Replacing the full description with a type or pronoun progressively removes explicit identity-related information from the prompt, leading to inconsistency. By incorporating our Sidecar method, improved character consistency is achieved under both settings.

We evaluate Sidecar on FreeStoryBench [2] using both SDXL- and FLUX-based models. Sidecar consistently improves vanilla T2I backbones and existing storytelling methods, including StoryDiffusion [26], ConsiStory [23], and FreeStory [2]. The results demonstrate improvements in both prompt-image alignment and character consistency across story frames. Sidecar also introduces negligible additional GPU memory usage and inference time.

In summary, Sidecar provides a plug-and-play mechanism for preserving entity-level semantics under free-form story prompts. It injects the character information omitted by later mentions into the text-conditioning process without changing the architecture of the base diffusion model. Experiments on FreeStoryBench show consistent improvements across multiple SDXL- and FLUX-based baselines, with negligible additional computational overhead.

## 2. Related Work

## 2.1. Visual Storytelling

Visual storytelling aims to generate a sequence of images that follows a textual narrative while preserving coherent characters, scenes, and visual appearance across frames. Early methods learn story-level dependencies by conditioning each generation on previous prompts or images [5, 14, 17, 25], or by introducing additional modules to aggregate multimodal history [1, 3, 7, 11, 20, 21, 27].

These approaches improve visual consistency through taskspecific training on story datasets. However, they require additional trainable components, story-level supervision, or modifications to the original generation architecture. These limitations motivate training-free methods that adapt pretrained T2I models directly during inference.

## 2.2. Training-Free Consistent Story Generation

A major line of training-free research preserves subject identity by reusing attention or intermediate visual features across generated images. ConsiStory [23] introduces subject-driven shared attention and correspondence-based feature injection to transfer subject information between images. StoryDiffusion [26] proposes Consistent Self-Attention, which establishes feature interactions across a sequence of generations. CharaConsist [24] further improves fine-grained character consistency through point-tracking attention, adaptive token merging, and decoupled control of foreground and background features.

Another line of work improves consistency by modifying, expanding, or reorganizing the textual conditions used for story generation. OnePromptOneStory [12] concatenates all story prompts into a single input and exploits the contextual consistency of the text encoder to preserve character identity. Infinite-Story [15] introduces Identity Prompt Replacement to align identity attributes across different prompts, together with attention guidance for appearance and style consistency. DreamStory [8] uses an LLM to construct detailed descriptions for the subjects and scenes. It then generates subject portraits and uses them together with the textual descriptions as multimodal references. These methods demonstrate the importance of textual and multimodal conditions to improve consistency. However, they commonly concatenate, rewrite, replace, or expand the original prompts. In contrast, Sidecar retains the original narrative prompts and supplements their text representations with entity-level semantics preserved from the initial description.

Most training-free consistency methods assume that a complete character description is repeated across prompts [12] or can be reconstructed through prompt rewriting [8]. FreeStory [2] introduces a free-form story setting in which the full character description is provided only in the first prompt, while later prompts refer to the same character through a type-level mention or pronoun. This setting more closely follows natural narrative structure and avoids repeatedly inserting detailed descriptions into every prompt. FreeStory maintains character consistency by reusing attention features from previous generations. Nevertheless, the text representations of later prompts remain semantically under-specified because type and pronouns omit the detailed identity information contained in the initial prompt. Sidecar addresses this complementary text-side limitation by preserving the entity semantics from the first prompt and injecting them into later text representations without rewriting the prompts or modifying the base diffusion model.

![](images/1036b379c75d52ed4d2ba5201fa74bcf2b60a83c6b1b7e10c742406d330ca733.jpg)  
Figure 2. Overview of Sidecar. Given the first story prompt, Sidecar extracts the hidden states corresponding to the complete character description at each text-encoder layer and stores them as layer-wise semantic representations. When encoding a later prompt, the corresponding representations are temporarily inserted after the begin-of-sentence token and interact with the current prompt tokens through self-attention. The temporary tokens are removed after each layer, preserving the original output length and interface of the text encoder. The resulting augmented text condition is then provided to the original image-generation pipeline, while any existing visual consistency mechanism remains unchanged

## 3. Method

## 3.1. Problem Formulation

Given a story consisting of N textual prompts, we denote the prompt sequence as

$$
\mathcal { P } = \{ P _ { 0 } , P _ { 1 } , \ldots , P _ { N - 1 } \} .\tag{1}
$$

Under the free-form setting, the character is introduced with a detailed description in the first prompt, while subsequent prompts refer to the same character using a shorter typelevel mention or pronoun. Let d denote the complete character description and $m _ { i }$ denote its reference in prompt $P _ { i }$ Although d and $m _ { i }$ refer to the same entity, they contain substantially different semantic information.

A pretrained text encoder $\mathcal { E }$ maps each prompt to a sequence of contextualized token representations:

$$
\mathbf { H } _ { i } = \mathcal { E } ( P _ { i } ) \in \mathbb { R } ^ { L \times D } ,\tag{2}
$$

where $L$ is the token sequence length and $D$ is the hidden dimension. When $P _ { i }$ contains only a type or pronoun, its representation may not retain the appearance attributes introduced by $d .$ Existing storytelling methods can transfer visual information from previous generations, but the current text condition remains semantically under-specified.

We address this by augmenting the text-encoding process with the hidden representations of the initial character description. The augmented text condition is denoted as

$$
\begin{array} { r } { \widetilde { \bf H } _ { i } = \mathcal { E } _ { \mathrm { S i d e c a r } } ( P _ { i } , d ) . } \end{array}\tag{3}
$$

Sidecar does not rewrite the current prompt or permanently increase the length of its output representation. Instead, it

introduces temporary semantic tokens within the text encoder, allowing the current prompt to retrieve the identity information omitted by the shorter character mentions.

## 3.2. Sidecar Semantic Augmentation

Figure 2 presents an overview of Sidecar. The method consists of two stages: semantic extraction and semantic injection. During semantic extraction, Sidecar identifies the detailed character description and extracts its layer-wise hidden representations. During semantic injection, these representations are inserted as temporary semantic tokens when encoding later prompts.

Semantic Extraction. We first identify the token span corresponding to the complete character description d. Let

$$
\mathcal { M } = \{ ( s _ { 1 } , e _ { 1 } ) , \dotsc , ( s _ { K } , e _ { K } ) \}\tag{4}
$$

denote the set of matched token spans, where each pair $( s _ { k } , e _ { k } )$ specifies the start and end indices of a description phrase. Consider a Transformer text encoder with L layers. Let

$$
\mathbf { H } _ { \mathrm { r e f } } ^ { \ell - 1 } \in \mathbb { R } ^ { L \times D }\tag{5}
$$

denote the input hidden states of layer ℓ when encoding the initial character description. At each layer, Sidecar extracts the hidden states corresponding to the identified character:

$$
\mathbf { S } ^ { \ell } = \mathrm { G a t h e r } \left( \mathbf { H } _ { \mathrm { r e f } } ^ { \ell - 1 } , \mathcal { M } \right) \in \mathbb { R } ^ { M \times D } ,\tag{6}
$$

where M is the total number of tokens contained in the matched spans.

The extracted representation $\mathbf { S } ^ { \ell }$ preserves the token-level semantics of the full character description at layer ℓ. Sidecar stores these representations separately for each encoder layer, retaining the contextualized character information formed at different representation depths.

Semantic Injection. When encoding a later prompt $P _ { i }$ Sidecar injects the extracted description representations into each Transformer layer. Let

$$
\mathbf { H } _ { i } ^ { \ell - 1 } = \left[ \mathbf { h } _ { i , \mathrm { B O S } } ^ { \ell - 1 } ; \mathbf { h } _ { i , 1 } ^ { \ell - 1 } ; \ldots ; \mathbf { h } _ { i , L - 1 } ^ { \ell - 1 } \right]\tag{7}
$$

denote the input hidden states of layer $\ell ,$ where $\mathbf { h } _ { i , \mathrm { B O S } } ^ { \ell - 1 }$ is the begin-of-sentence representation. Sidecar inserts $\mathbf { S } ^ { \ell }$ immediately after the begin-of-sentence token:

$$
\mathbf { X } _ { i } ^ { \ell } = \left[ \mathbf { h } _ { i , \mathrm { B O S } } ^ { \ell - 1 } ; \mathbf { S } ^ { \ell } ; \mathbf { h } _ { i , 1 } ^ { \ell - 1 } ; \ldots ; \mathbf { h } _ { i , L - 1 } ^ { \ell - 1 } \right] .\tag{8}
$$

The augmented sequence is then processed by the original Transformer layer:

$$
\overline { { \mathbf { X } } } _ { i } ^ { \ell } = \mathbf { X } _ { i } ^ { \ell } + \mathrm { M S A } ^ { \ell } \left( \mathrm { L N } _ { 1 } ^ { \ell } \left( \mathbf { X } _ { i } ^ { \ell } \right) \right) ,\tag{9}
$$

$$
\widehat { \mathbf { X } } _ { i } ^ { \ell } = \overline { { \mathbf { X } } } _ { i } ^ { \ell } + \mathrm { M L P } ^ { \ell } \left( \mathrm { L N } _ { 2 } ^ { \ell } \left( \overline { { \mathbf { X } } } _ { i } ^ { \ell } \right) \right) .\tag{10}
$$

The semantic tokens are visible to the tokens of the current prompt during self-attention. The prompt representation can therefore retrieve identity-related information that is absent from a type-level mention or pronoun. Unlike direct embedding replacement, this operation allows the text encoder to contextualize the initial character semantics together with the current action, scene, and other content.

After the Transformer layer is evaluated, the temporary semantic-token outputs are removed:

$$
\mathbf { H } _ { i } ^ { \ell } = \mathrm { R e m o v e S i d e c a r } \left( \widehat { \mathbf { X } } _ { i } ^ { \ell } \right) .\tag{11}
$$

The resulting representation has the same sequence length and hidden dimension as the original prompt representation. It can therefore be passed to the next encoder layer without changing the external interface of the text encoder.

This process is repeated at every text-encoder layer using the semantic representation extracted from the corresponding layer. In particular, $\mathbf { S } ^ { \ell }$ is inserted only during the computation of layer ℓ and is removed before the next layer. The semantic tokens are therefore temporary and do not accumulate across the encoder.

After the final layer, Sidecar produces the augmented prompt representation

$$
\widetilde { \bf H } _ { i } = \mathcal { E } _ { \mathrm { S i d e c a r } } \left( P _ { i } , \{ { \bf S } ^ { \ell } \} _ { \ell = 1 } ^ { L _ { \mathcal { E } } } \right) .\tag{12}
$$

The original narrative prompt remains unchanged. Sidecar does not replace mentions such as “the man” or “he” with the full description. Instead, the current prompt specifies the new action and scene, while the temporary semantic tokens provide the identity information omitted by the reference.

## 3.3. Encoder-Specific Integration

Sidecar operates within the text-conditioning stage and does not modify the image-generation network. Let $\mathcal { G }$ denote a pretrained image generator and let $\gamma _ { < i }$ represent any history visual information used by an existing visual storytelling method [2, 12, 23, 26]. The i-th story frame is generated as

$$
I _ { i } = \mathcal { G } \left( \widetilde { \mathbf { H } } _ { i } , \mathcal { V } _ { < i } \right) ,\tag{13}
$$

where $\gamma _ { < i }$ is omitted for a vanilla text-to-image model. When Sidecar is combined with existing methods, their original consistency mechanisms remain unchanged. Sidecar operates as a complementary text-side module that augments the semantic condition, while the underlying method continues to propagate visual features across story frames.

Because SDXL [16] and FLUX [10] use different text encoders, we adopt an encoder-specific integration strategy.

SDXL Integration. SDXL [16] contains two CLIP text encoders. We apply the Sidecar semantic extraction and injection independently to both encoders. Given the augmented token representations $\widetilde { \mathbf { H } } _ { i } ^ { ( 1 ) }$ and $\widetilde { \mathbf { H } } _ { i } ^ { ( 2 ) }$ , the final text condition follows the original SDXL formulation:

$$
\widetilde { \mathbf { H } } _ { i } ^ { \mathrm { S D X L } } = \mathrm { C o n c a t } \left( \widetilde { \mathbf { H } } _ { i } ^ { ( 1 ) } , \widetilde { \mathbf { H } } _ { i } ^ { ( 2 ) } \right) ,\tag{14}
$$

where concatenation is performed along the feature dimension. The pooled text representation is obtained following the standard SDXL pipeline. Sidecar is applied only to the positive text condition, while the negative prompt is encoded without semantic augmentation.

FLUX Integration. FLUX [10] uses a CLIP encoder and a T5 encoder to construct its text condition. We use different semantic augmentation strategies for the two encoders. For CLIP, we apply the layer-wise Sidecar mechanism described above. For T5, we concatenate the first story prompt $P _ { 0 }$ with the current prompt $P _ { i }$ before encoding:

$$
{ \bf Z } _ { i } ^ { + } = { \mathcal { E } } _ { \mathrm { T 5 } } \left( \left[ P _ { 0 } ; P _ { i } \right] \right) ,\tag{15}
$$

After joint contextual encoding, we retain only the hidden states corresponding to the current prompt:

$$
\widetilde { { \bf Z } } _ { i } = \mathrm { S l i c e } _ { P _ { i } } \left( { \bf Z } _ { i } ^ { + } \right) .\tag{16}
$$

Although the output corresponding to $P _ { 0 }$ is removed, the tokens from the first prompt contribute to the T5 selfattention computation. The retained states of $P _ { i }$ are therefore contextualized by both the initial character information and the current narrative. The resulting representation has the same sequence length as the standard T5 encoding of $P _ { i }$ preserving compatibility with the original FLUX pipeline.

We use this encoder-specific strategy based on the empirical results. Sidecar enables layer-wise semantic interaction in the CLIP pathway, while T5 jointly contextualizes the initial and current prompts through its native self-attention. The longer context supported by T5 makes prompt concatenation practical, while avoiding direct modification of its intermediate hidden states. The resulting CLIP and T5 representations provide complementary semantic conditioning.

Overall, Sidecar introduces no trainable parameters and requires no additional optimization. It preserves the inputoutput interface of the original text-conditioning pipeline, allowing it to be integrated with both vanilla generation backbones and existing training-free storytelling methods.

## 4. Experiments

## 4.1. Experimental Setup

Dataset. We evaluate our method on the single-character subset of FreeStoryBench [2], which contains 100 stories with six scenes per story. Each story can be rendered under six prompting settings that differ in how the character is referenced across scenes. We use the mixed setting, where the complete character description appears only in the first prompt, while subsequent prompts refer to the character using a mixture of type-level mentions and pronouns. This setting reflects natural storytelling and provides a challenging testbed for character consistency because later prompts omit the explicit character semantic information.

Baselines. We compare Sidecar with both SDXL-based and FLUX-based generation models. For the SDXLbased models, we evaluate vanilla SDXL [16], StoryDiffusion [26], and ConsiStory [23]. For the FLUX-based models, we evaluate vanilla FLUX [10] and FreeStory [2]. We additionally compare with OnePromptOneStory [12]. Since it concatenates all story prompts into a single input and directly modifies the text embeddings, its formulation differs from the targeted free-form setting and is not readily compatible with our proposed Sidecar. We therefore report it as an independent baseline without applying Sidecar.

Metrics. We use three metrics to evaluate the generated story images. CLIP [9] similarity measures the semantic alignment between each generated image and its corresponding text prompt. DINO [13] similarity evaluates character consistency across frames within the same story. DreamSim [6] measures perceptual similarity between generated character appearances, where a lower value indicates better consistency. To reduce the influence of background similarity, we use Grounded SAM [18] to segment the target character in each image, and replace the background region with random noise before computing DINO and DreamSim.

Implementation Details. Sidecar is applied at inference time without introducing trainable parameters or modifying pretrained image-generation models. It operates on all CLIP Transformer layers and only augments the positive text-conditioning path. For SDXL, Sidecar is applied to both CLIP encoders. For FLUX, it is applied to CLIP, while T5 jointly encodes $[ P _ { 0 } ; P _ { i } ]$ with a maximum length of 512 tokens and retains only the hidden states of $P _ { i }$ . We use the default inference settings from the released baseline implementations and keep the generation settings fixed when comparing each baseline with its Sidecar variant. All experiments are conducted on a single NVIDIA L40S GPU. Our code will be made publicly available upon acceptance.

## 4.2. Results on FreeStoryBench

Table 1 shows the quantitative results on FreeStoryBench. We report the performance of each baseline before and after adding Sidecar. The results show that Sidecar consistently improves all evaluated baselines across both SDXLbased and FLUX-based models. For the SDXL backbone, Sidecar improves vanilla SDXL from 0.776 to 0.859 in

![](images/cbcfa580cba99dcb4191bc3fbf17ee0d7f162bc55eb9b797454ffbb386249bd0.jpg)  
Figure 3. Qualitative comparison on FreeStoryBench. Each story provides a complete character description only in the first prompt, while later prompts use type or pronouns. Compared with the original baselines, Sidecar better preserves character consistency across different scenes. Sidecar also maintains text-image alignment, indicating that the improved consistency does not prevent the character or scene from changing according to the narrative. Additional qualitative results are provided in the supplementary material.

Table 1. Quantitative results on FreeStoryBench. We evaluate SDXL-based and FLUX-based methods with and without Sidecar. Highe CLIP and DINO scores are better, while lower DreamSim is better. For each Sidecar variant, the value in parentheses denotes the change relative to its corresponding baseline. Sidecar consistently improves character consistency and prompt-image alignment across all evaluated models, with particularly large gains on DINO and DreamSim.
<table><tr><td>Method</td><td></td><td>CLIP↑</td><td>DINO ↑</td><td>DreamSim↓</td></tr><tr><td rowspan="6">S-ed mooes</td><td>OnePromptOneStory [12]</td><td>0.791</td><td>0.361</td><td>0.495</td></tr><tr><td>SDXL</td><td>0.776</td><td>0.308</td><td>0.540</td></tr><tr><td>SDXL + Sidecar</td><td>0.859 (+0.083)</td><td>0.621 (+0.313)</td><td>0.333 (-0.207)</td></tr><tr><td>StoryDiffusion [26]</td><td>0.803</td><td>0.410</td><td>0.483</td></tr><tr><td>StoryDiffusion + Sidecar</td><td>0.886 (+0.083)</td><td>0.702 (+0.292)</td><td>0.272 (-0.211)</td></tr><tr><td>ConsiStory [23]</td><td>0.779</td><td>0.349</td><td>0.505</td></tr><tr><td rowspan="4">F-ed mooels</td><td>ConsiStory + Sidecar</td><td>0.880 (+0.101)</td><td>0.692 (+0.343)</td><td>0.264 (-0.241)</td></tr><tr><td>FLUX FLUX + Sidecar</td><td>0.776</td><td>0.323</td><td>0.523</td></tr><tr><td>FreeStory [2]</td><td>0.842 (+0.066)</td><td>0.575 (+0.252)</td><td>0.346 (-0.177)</td></tr><tr><td>FreeStory + Sidecar</td><td>0.801 0.869 (+0.068)</td><td>0.421 0.649 (+0.228)</td><td>0.448 0.291 (-0.157)</td></tr></table>

CLIP, from 0.308 to 0.621 in DINO, and reduces DreamSim from 0.540 to 0.333. Similar improvements are also observed when Sidecar is applied to StoryDiffusion and Consistory. For FLUX-based models, Sidecar also shows consistent improvement, indicating that Sidecar is not limited to a specific diffusion architecture, and can be applied to both UNet-based and transformer-based generation backbones.

The improvements are especially significant on the consistency metrics. This suggests that the proposed semantic sidecar effectively helps the model preserve character identity across different story frames. Moreover, Sidecar also improves CLIP similarity for every baseline. This shows that the improved consistency does not come from simply forcing visual similarity between frames, but also improves the semantic alignment between prompt and image.

We also measure the computational overhead introduced by Sidecar across different baseline methods. Sidecar introduces negligible computational overhead, as it only augments the text-encoding process while leaving the generation backbone unchanged. Across the evaluated baselines, inference time and peak GPU memory remain largely comparable after enabling Sidecar; for example, on SDXL, runtime increases by only 0.2% with unchanged peak memory, while on ConsiStory the increases are 0.9% and 0.5%, respectively. Full efficiency results are reported in Appendix.

Overall, these results show that Sidecar serves as a plugand-play semantic augmentation module for free-form story generation. Across different baselines and backbones, Sidecar consistently improves character consistency with negligible additional computational cost, while maintaining and in all cases improving prompt-image alignment.

## 4.3. Ablation Study

Effect of Reference Type. Table 2 analyzes the effect of Sidecar under type-level mentions and pronouns. We report results for SDXL and FreeStory to cover two different generation backbones while also including both a vanilla T2I model and a dedicated storytelling method. This provides a representative evaluation of whether the observed behavior generalizes across model families and generation settings. Sidecar improves all metrics for both FreeStory and SDXL, with consistently larger gains for pronoun-based prompts. On FreeStory, the DINO improvement increases from 0.077 for type-level mentions to 0.309 for pronouns, while the DreamSim improvement increases from 0.061 to 0.206. A similar trend is observed on SDXL. These results support our hypothesis that Sidecar is particularly beneficial when the prompt contains less explicit identity information.

Table 2. Effect of Sidecar under different character-reference types. ∆ reports the improvement introduced by Sidecar. For DreamSim, improvement is measured by the reduction in distance.
<table><tr><td colspan="4">Type-level Mention</td><td colspan="3">Pronoun</td></tr><tr><td>Metric</td><td></td><td>Baseline +Sidecar</td><td>∆</td><td>Baseline +Sidecar</td><td></td><td>∆</td></tr><tr><td>Freory CLIP↑</td><td>0.844</td><td>0.850</td><td>0.007</td><td>0.771</td><td>0.849</td><td>0.078</td></tr><tr><td>DINO↑</td><td>0.614</td><td>0.690</td><td>0.077</td><td>0.334</td><td>0.643</td><td>0.309</td></tr><tr><td>DreamSim ↓</td><td>0.331</td><td>0.270</td><td>0.061</td><td>0.506</td><td>0.301</td><td>0.206</td></tr><tr><td rowspan="3">CLIP↑ SAS DINO↑</td><td>0.810</td><td>0.862</td><td>0.052</td><td>0.712</td><td>0.864</td><td>0.151</td></tr><tr><td>0.503</td><td>0.636</td><td>0.133</td><td>0.201</td><td>0.602</td><td>0.401</td></tr><tr><td>DreamSim ↓ 0.438</td><td>0.322</td><td>0.116</td><td>0.614</td><td>0.341</td><td>0.273</td></tr></table>

Text-Encoder Integration Strategies. Table 3 compares different strategies for incorporating the character semantics into the CLIP and T5 encoders of FreeStory. Applying Sidecar only to CLIP improves all metrics over the original FreeStory, showing that layer-wise semantic augmentation is effective for CLIP. T5 concatenation alone yields only modest improvements, while directly applying Sidecar to T5 degrades performance, suggesting that the same injection strategy does not transfer uniformly across text encoders. Joint prompt concatenation for both encoders provides a strong alternative. However, with T5 concatenation fixed, replacing CLIP concatenation with Sidecar further improves all metrics from 0.860/0.594/0.324 to 0.869/0.649/0.291 for CLIP/DINO/DreamSim. This controlled comparison isolates the contribution of CLIP Sidecar and shows that the gains of the FLUX-based integration cannot be attributed to T5 concatenation alone.

Table 3. Ablation of text-encoder integration strategies on FreeStory. The gray row denotes the FreeStory setting, while the green row denotes the FreeStory+Sidecar setting.
<table><tr><td>CLIP</td><td>T5</td><td>CLIP↑</td><td>DINO ↑</td><td>DreamSim ↓</td></tr><tr><td>Original</td><td>Original</td><td>0.801</td><td>0.421</td><td>0.448</td></tr><tr><td>Sidecar</td><td>Original</td><td>0.830</td><td>0.481</td><td>0.413</td></tr><tr><td>Original</td><td>Concatenation</td><td>0.809</td><td>0.432</td><td>0.444</td></tr><tr><td>Original</td><td>Sidecar</td><td>0.750</td><td>0.285</td><td>0.548</td></tr><tr><td>Sidecar</td><td>Sidecar</td><td>0.794</td><td>0.514</td><td>0.399</td></tr><tr><td>Concatenation</td><td>Concatenation</td><td>0.860</td><td>0.594</td><td>0.324</td></tr><tr><td>Sidecar</td><td>Concatenation</td><td>0.869</td><td>0.649</td><td>0.291</td></tr></table>

The results suggest that CLIP and T5 benefit from different integration strategies. For CLIP, direct concatenation is constrained by its 77-token limit and may truncate scene information, especially for multi-character prompts. Sidecar avoids consuming this token budget by separately encoding the reference and injecting only character-description representations. In contrast, T5 supports a much longer context, making joint prompt concatenation more practical. Our ablation also shows that Sidecar is less effective than concatenation for T5. The best performance is therefore achieved by combining CLIP Sidecar with T5 concatenation.

Effect of Semantic Interaction Location. Table 4 compares Sidecar with a post-encoder semantic expansion baseline on SDXL. Specifically, it independently encodes the initial description and current prompt, then replaces the final CLIP representations of the current character mention with the final representations of the complete description. Although this strategy restores the omitted semantics and bypasses CLIP’s input-length constraint, the description and current-prompt representations do not interact through text-encoder self-attention.

Post-encoder expansion improves over the original SDXL baseline, showing that directly restoring the missing information is beneficial. Sidecar achieves stronger overall performance, suggesting that semantic availability alone is insufficient. Allowing the reused description representations to interact with the current action and scene within each CLIP layer provides more effective character conditioning than directly modifying the final text representation.

Table 4. Effect of semantic interaction location on SDXL. Postencoder replacement substitutes the final mention representations with those of the full description, whereas Sidecar enables layerwise interaction between the description and current prompt. Both are applied to both CLIP encoders.
<table><tr><td>Method</td><td>CLIP↑</td><td>DINO↑</td><td>DreamSim ↓</td></tr><tr><td>SDXL</td><td>0.776</td><td>0.308</td><td>0.540</td></tr><tr><td>SDXL + Post-Encoder Expansion</td><td>0.835</td><td>0.488</td><td>0.407</td></tr><tr><td>SDXL + Sidecar</td><td>0.859</td><td>0.621</td><td>0.333</td></tr></table>

## 4.4. Discussion

Sidecar complements visual consistency methods by addressing missing identity semantics in later free-form prompts. Its gains on both vanilla generators and storytelling baselines suggest that text-side semantic reuse remains useful even when visual features are already propagated across frames. Larger improvements for pronouns support this interpretation, while the comparison with concatenation and post-encoder replacement indicates that layer-wise interaction within CLIP is more effective than simply providing the initial description. The different behaviors of CLIP and T5 further motivate encoder-specific integration. The improvements in CLIP similarity suggest that consistency is not achieved at the expense of overall prompt-image alignment.

Limitations. Sidecar relies on the initial prompt to provide a clear and accurate character description; ambiguous or incomplete descriptions may therefore limit its effectiveness. Our current evaluation focuses on stories with a single main character, and extending the method to multiple interacting characters would require reliable entity-specific extraction and disambiguation. In addition, CLIP retains its 77-token context limit, so very long initial prompts may truncate relevant character information.

## 5. Conclusion

We introduced Sidecar, a training-free semantic augmentation module for free-form visual storytelling. Sidecar extracts identity-related representations from the initial character description and injects them into the encoding of later prompts, restoring information omitted by the reference mentions. Sidecar requires no additional training, introduces negligible computational overhead, and can be integrated with both vanilla diffusion models and existing storytelling methods. Experiments on FreeStoryBench show that Sidecar improves character consistency and prompt-image alignment across SDXL- and FLUX-based models. These results demonstrate that incomplete text semantics are an important source of identity drift and that text-side semantic augmentation provides a simple and complementary solution for more consistent story generation.

## References

[1] Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in textto-image diffusion models. In ACM SIGGRAPH 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 1, 2

[2] Sibo Dong, Ismail Shaheen, and Sarah Adel Bargal. Freestory: Training-free character consistency for free-form visual storytelling. arXiv preprint arXiv:2606.25079, 2026. 1, 2, 3, 4, 5, 7

[3] Sibo Dong, Ismail Shaheen, Maggie Shen, Rupayan Mallick, and Sarah Adel Bargal. ViSTA: Visual Storytelling using Multi-modal Adapters for Text-to-Image Diffusion Models . In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 12–21, Los Alamitos, CA, USA, 2026. IEEE Computer Society. 1, 2

[4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings ofthe 41st International Conference on Machine Learning. JMLR.org, 2024. 1

[5] Zhangyin Feng, Yuchen Ren, Xinmiao Yu, Xiaocheng Feng, Duyu Tang, Shuming Shi, and Bing Qin. Improved visual story generation with adaptive context modeling. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 4939–4955, Toronto, Canada, 2023. Association for Computational Linguistics. 1, 2

[6] Stephanie Fu, Netanel Y. Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: learning new dimensions of human visual similarity using synthetic data. In Proceedings of the 37th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2023. Curran Associates Inc. 5

[7] Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Yingqing He, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, and Yujiu Yang. Interactive story visualization with multiple characters. In SIGGRAPH Asia 2023 Conference Papers, New York, NY, USA, 2023. Association for Computing Machinery. 1, 2

[8] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(12):11874–11891, 2025. 2, 3

[9] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 5

[10] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1, 4, 5

[11] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yan feng Wang, and Weidi Xie. Intelligent grimm - open-ended visual storytelling via latent diffusion models. In Proceed ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6190–6200, 2024. 1, 2

[12] Tao Liu, Kai Wang, Senmao Li, Joost van de Weijer, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang, Jian Yang, and Ming-Ming Cheng. One-prompt-one-story: Free-lunch consistent text-to-image generation using a single prompt. In The Thirteenth International Conference on Learning Repre sentations, 2025. 1, 2, 3, 4, 5, 7

[13] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V.´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. Featured Certification. 5

[14] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. Synthesizing coherent story with auto-regressive latent diffusion models. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2908– 2918, 2024. 1, 2

[15] Jihun Park, Kyoungmin Lee, Jongmin Gim, Hyeonseo Jo, Minseok Oh, Wonhyeok Choi, Kyumin Hwang, Jaeyeul Kim, Minwoo Choi, and Sunghoon Im. Infinite-story: A training-free consistent text-to-image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 8278–8286, 2026. 2

[16] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 1, 4, 5

[17] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In 2023 IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 2493–2502, 2023. 1, 2

[18] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 5

[19] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2022. 1

[20] Xiaoqian Shen and Mohamed Elhoseiny. Storygpt-v: Large language models as consistent story visualizers. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13273–13283, 2025. 1, 2

[21] Tianyi Song, Jiuxin Cao, Kun Wang, Bo Liu, and Xiaofeng Zhang. Causal-story: Local causal attention utilizing parameter-efficient tuning for visual story synthesis. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3350–3354, 2024. 1, 2

[22] Ming Tao, Bing-Kun Bao, Hao Tang, Yaowei Wang, and Changsheng Xu. Storyimager: A unified and efficient framework for coherent story visualization and completion. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LVI, page 479–495, Berlin, Heidelberg, 2024. Springer-Verlag. 1

[23] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Trans. Graph., 43(4), 2024. 1, 2, 4, 5, 7

[24] Mengyu Wang, Henghui Ding, Jianing Peng, Yao Zhao, Yunpeng Chen, and Yunchao Wei. Characonsist: Finegrained consistent character generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16058–16067, 2025. 1, 2

[25] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 1871–1881, 2025. 1, 2

[26] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 1, 2, 4, 5, 7

[27] Zhongyang Zhu and Jie Tang. Cogcartoon: Towards practical story visualization. Int. J. Comput. Vision, 133(4): 1808–1833, 2024. 1, 2

Table 5. Multi-character evaluation on FreeStoryBench. Higher is better for CLIP and DINO, while lower is better for DreamSim.
<table><tr><td>Method</td><td>CLIP↑</td><td>DINO↑</td><td>DreamSim ↓</td></tr><tr><td>SDXL</td><td>0.808</td><td>0.327</td><td>0.496</td></tr><tr><td>SDXL + Sidecar</td><td></td><td>0.888 (+0.080) 0.506 (+0.179) 0.380 (−0.116)</td><td></td></tr><tr><td>StoryDiffusion StoryDiffusion + Sidecar 0.884 (+0.071) 0.570 (+0.182) 0.340 (−0.137)</td><td>0.813</td><td>0.388</td><td>0.477</td></tr><tr><td>ConsiStory</td><td>0.823</td><td>0.359</td><td>0.478</td></tr><tr><td>ConsiStory + Sidecar</td><td></td><td>0.893 (+0.070) 0.589 (+0.230) 0.307 (−0.171)</td><td></td></tr><tr><td>FLUX</td><td>0.799</td><td>0.360</td><td>0.471</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>FLUX + Sidecar</td><td></td><td>0.850 (+0.051) 0.485 (+0.125) 0.384 (−0.087)</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>FreeStory</td><td>0.845</td><td></td><td>0.392</td></tr><tr><td>FreeStory + Sidecar</td><td>0.867 (+0.022) 0.522 (+0.059) 0.354 (−0.038)</td><td>0.463</td><td></td></tr></table>

## A. Multi-Character Evaluation

We evaluate Sidecar on the multi-character subset of FreeStoryBench to examine whether entity-specific semantic augmentation remains effective when multiple characters appear in the same story. The subset contains 100 stories and 600 frames. For each character, we use its corresponding structured description span provided by FreeStoryBench and construct the Sidecar representation independently, allowing each entity to retrieve its own reference semantics without sharing description tokens.

Table 5 shows that Sidecar consistently improves performance across all evaluated baselines. DINO increases substantially for all compared baselines, with corresponding reductions in DreamSim. Importantly, CLIP alignment improves for every baseline, indicating that the gains in multicharacter consistency do not come at the expense of overall prompt fidelity. These results demonstrate that Sidecar extends effectively to multi-character stories and can preserve entity-specific semantics without evident interference between characters.

## B. Effect of Sidecar Placement Across Encoder Layers

We further study how the placement of Sidecar within the text encoder affects its performance. Using SDXL, we divide the CLIP text encoder into three equal depth ranges and apply Sidecar only to the first (Early), middle (Middle), or last (Late) one-third of the encoder layers. We compare these variants with the original SDXL baseline and our default configuration that applies Sidecar at all encoder layers.

As shown in Table 6, the effectiveness of Sidecar increases substantially with encoder depth. Applying Sidecar only to the early layers provides little improvement over the baseline, whereas the middle layers yield moderate gains and the late layers produce a much larger improvement in both character consistency and prompt alignment. This trend suggests that the higher-level contextual representations formed in deeper CLIP layers are particularly important for recovering character semantics. Nevertheless, applying Sidecar throughout the entire encoder consistently achieves the best performance.

Table 6. Ablation on Sidecar placement across CLIP encoder layers on SDXL. We only apply Sidecar to the first, middle, and last one-third of encoder layers, respectively.
<table><tr><td>Method</td><td>CLIP-T ↑</td><td>DINO ↑</td><td>DreamSim ↓</td></tr><tr><td>SDXL</td><td>0.776</td><td>0.308</td><td>0.540</td></tr><tr><td>+ Sidecar (Early 1/3)</td><td>0.759 (-0.017)</td><td>0.326 (+0.018)</td><td>0.526 (-0.014)</td></tr><tr><td>+ Sidecar (Middle 1/3)</td><td>0.785 (+0.009)</td><td>0.401 (+0.093)</td><td>0.500 (-0.040)</td></tr><tr><td>+ Sidecar (Late 1/3)</td><td>0.840 (+0.064)</td><td>0.544 (+0.236)</td><td>0.394 (-0.146)</td></tr><tr><td>+ Sidecar (All)</td><td>0.859 (+0.083)</td><td>0.621 (+0.313)</td><td>0.333 (-0.207)</td></tr></table>

Table 7. Computational efficiency on the full dataset evaluation. Runtime is averaged over all processed samples, and GPU memory reports the peak allocated memory during inference.
<table><tr><td>Method</td><td>Peak Mem. (GB) Avg. Time (s) ∆Mem. ∆Time</td><td></td><td></td></tr><tr><td>SDXL</td><td>10.49</td><td>35.68</td><td></td></tr><tr><td>SDXL + Sidecar</td><td>10.49</td><td>35.76</td><td>0.0% +0.2%</td></tr><tr><td>StoryDiffusion</td><td>36.67</td><td>61.13</td><td></td></tr><tr><td>StoryDiffusion + Sidecar</td><td>36.68</td><td>62.61</td><td>0.0% +2.4%</td></tr><tr><td>ConsiStory</td><td>21.66</td><td>151.41</td><td></td></tr><tr><td>ConsiStory + Sidecar</td><td>21.77</td><td>152.72</td><td>+0.5% +0.9%</td></tr><tr><td>FLUX</td><td>33.83</td><td>158.18</td><td></td></tr><tr><td>FLUX + Sidecar</td><td>34.05</td><td>158.71</td><td>+0.7% +0.3%</td></tr><tr><td>FreeStory</td><td>35.07</td><td>209.20</td><td></td></tr><tr><td>FreeStory + Sidecar</td><td>35.07</td><td>214.74</td><td>0.0% +2.6%</td></tr></table>

## C. Computational Efficiency

We further evaluate the computational cost of Sidecar by measuring peak allocated GPU memory and average inference time under the same generation settings. Table 7 reports the results for each baseline and its Sidecaraugmented counterpart. Overall, Sidecar introduces only marginal computational overhead across all evaluated methods. Peak GPU memory remains nearly unchanged after enabling Sidecar: SDXL, StoryDiffusion and FreeStory show no measurable increase, while ConsiStory and FLUX increase by only 0.5%, and 0.7%, respectively. The additional inference time is similarly small, ranging from 0.2% on SDXL to at most 2.6% on FreeStory. These results are consistent across both vanilla generators and existing consistency methods.

Overall, Sidecar is computationally lightweight, adding minimal memory and runtime overhead while substantially improving character consistency. This efficiency follows from its design, which augments only the textconditioning process while leaving the underlying imagegeneration backbone unchanged.

![](images/45522e56597af98962c63bf3386d1b8a234a5bb7e0653380b217788f3e317596.jpg)  
Figure 4. Qualitative comparison on FreeStoryBench.

![](images/600f5993cfd0486d9059eba8ef1a4a870182a44d970aecfb3a8728a8c9e351a7.jpg)  
Figure 5. Qualitative comparison on FreeStoryBench. Sidecar recovers identity semantics omitted from later prompts. Since vanilla T2I models do not contain an explicit cross-frame consistency mechanism, Sidecar improves identity preservation but does not guarantee complete visual consistency.

An elderly wizard   
wearing tattered blue   
robes and a pointed hat   
stands in the center of   
the library, reading a heavy grimoire.

SL

![](images/896646b5ec827783a87e325d1b84e4af12cc25906f2066b47894ac2b7002e14f.jpg)  
Figure 6. Qualitative comparison on FreeStoryBench. Sidecar recovers identity semantics omitted from later prompts. Since vanilla T2I models do not contain an explicit cross-frame consistency mechanism, Sidecar improves identity preservation but does not guarantee complete visual consistency.