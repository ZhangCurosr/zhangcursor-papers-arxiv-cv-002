# Towards Clinically Faithful Medical Image Captioning via Enhanced Vision–Language Alignment

Yunseo Lee, Hyun Jun Kim, Heeseung Shin, and Changwon Lim

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract— Medical image captioning is an important technique that can accelerate early-stage diagnostic workflows and enhance the interpretability of medical diagnostic AI systems. However, unlike general image captioning, generating clinically reliable captions remains challenging due to grayscale-based modalities, subtle anatomical cues, specialized medical phrasing, and variations in data quality. Despite recent advances in large vision–language models, fluent outputs do not necessarily guarantee sufficient alignment with clinical concept spaces or evaluation criteria. To address this issue, we propose a framework that strengthens clinical alignment by separating and enhancing training-time alignment and inference-time alignment. We build a medical image captioning pipeline that integrates single/dual vision encoders based on BioMedCLIP and SigLIP2, a Q-Former, and a LLaMA-based decoder, and examine the contribution of auxiliary learning for UMLS concept/type prediction. At inference, we apply singleembedding-based reranking to select the most appropriate caption among candidates, while at training we introduce MedPAIR-SCST, which combines clinically relevant rewards to shift the generative distribution toward improved clinical alignment. Our experiments show that leveraging complementary visual representations with a multi-encoder design and concept-level auxiliary learning can contribute to preserving clinically meaningful information. Furthermore, inference-time reranking provides a practical way to improve semantic and clinical alignment without additional training, whereas MedPAIR-SCST goes beyond selection by directly improving the model’s distribution to generate more consistent and clinically grounded captions. These findings suggest that jointly leveraging selection-based alignment and reinforcement-learning-based alignment can promote more trustworthy medical image captioning even in data-constrained settings.

IndexTerms— Clinical alignment, medical image caption-

ing, multimodal learning, reinforcement learning, visionlanguage model

## I. INTRODUCTION

EDICAL image captioning—automatically generating has the potential to accelerate report drafting, improve contentbased image retrieval, and increase the interpretability of diagnostic AI models. Compared with natural-image captioning, the task is complicated by grayscale modalities, subtle anatomical cues, and a highly specialized vocabulary, all of which demand fine-grained visual reasoning and domain knowledge [1]. While recent systems have made notable progress with encoder–decoder frameworks trained on paired image–text datasets, performance remains constrained by data quality, domain adaptability, and output reliability [2], [3], [4]. Public medical corpora frequently contain low-resolution images [5] and annotation-induced artifacts [6], degrading perception and propagating spurious patterns into generation. Moreover, generic vision encoders may miss subtle, domain-specific signals [7], and caption decoders can produce inconsistent or incomplete statements due to limited grounding in clinical semantics [8].

Despite advances in large vision–language models (VLMs), two structural gaps persist in practice. First, general-purpose pretraining induces a global-semantics bias that under-attends cues dispersed over small receptive fields, encouraging formulaic phrasing (e.g., “mild cardiomegaly,” “no acute process”). Second, there is no guarantee that generated text is automatically aligned with clinical concept spaces (terminologies/types) or with evaluation metrics (e.g., BERTScore, ROUGE, biomedical similarity). Outputs may read naturally yet remain weakly supported or terminologically inconsistent, motivating a shift from fluency-centric modeling to clinical alignment as a primary objective.

We address this gap by disentangling learning from inference and evaluating them separately. As a Stage-1, we train MedCap under six configurations to establish a reproducible baseline for comparison. This allows us to analyze how performance changes with encoder composition and auxiliary supervision, and to identify potentially complementary behaviors across models.

At inference time, we incorporate single-embedding based reranking as an alignment method (not a learning stage):

among candidate captions produced by the Stage-1 models, we select the sentence with the highest image–text compatibility under a pre-specified embedding model. This post-hoc selection requires no parameter updates or additional data, directly injects a vision–language alignment signal that tends to reduce hallucinations and improve relevance, and admits simple policy controls (e.g., thresholding, length normalization) for deployment.

However, post-hoc selection does not modify the model’s generative distribution. We therefore introduce Stage-2 (training-time alignment) via a modified self-critical sequence training procedure. For each input, multiple candidates are sampled and re-scored under teacher-forcing sequence loglikelihoods; optimization then shifts the distribution toward clinically aligned metrics. Our modification emphasizes practical stability without a reference model or KL regularization by combining group-level reward aggregation with reference-free pairwise comparison, mitigating the high-variance behavior commonly observed in standard SCST. In short, inference-time reranking selects better candidates, whereas Stage-2 improves the generator itself; the two are complementary.

Under identical decoding, we compare Stage-1 SFT models, inference-time alignment methods at test time (reranking), and Stage-2 training based on a modified self-critical objective. We report textual metrics (BERTScore, ROUGE, BLEURT) and biomedical similarity measures, and we monitor reward– metric alignment during training to verify that improvements follow the intended evaluation direction rather than superficial overfitting. Implementation details (ensemble and selection policies) are deferred to the Appendix; the main text focuses on the principles and effects of alignment.

The contributions of this work are as follows:

1) Constructed an end-to-end framework for medical image captioning by integrating multi-encoder visual grounding, a query-aggregation module, and a Llama-based decoder with supervised fine-tuning.

2) Strengthened clinical alignment along two orthogonal axes rather than a sequential pipeline: (a) inference-time selection via post-hoc reranking and (b) training-time distributional optimization via an enhanced self-critical objective; each axis is evaluated independently and can be applied on its own.

3) Proposed a reference/KL-free self-critical variant that combines group-level reward summarization with reference-free pairwise ordering, yielding clinical alignment gains without a reference model or KL regularization.

## II. RELATED WORK

Medical image captioning has evolved alongside advances in vision-language modeling, primarily following the encoderdecoder paradigm widely used in natural image captioning. Early works employed convolutional neural networks (CNNs) as visual encoders paired with recurrent neural networks (RNNs) or Transformer-based decoders to generate captions [1]. However, these approaches often lacked clinical specificity, as they relied on general-purpose image features and were trained on limited or noisy medical datasets.

Medical image encoders are now designed primarily within a vision–language pretraining paradigm that goes beyond single-modality feature extraction to jointly model images and their accompanying reports. BioViL-T [9] extends this approach with a hybrid CNN–Transformer encoder for longitudinal chest X-rays and associated reports. BiomedCLIP [10] builds a CLIP-style biomedical foundation model by pretraining a ViT vision encoder and PubMedBERT text encoder on PMC-15M, a dataset of over 15 million figure–caption pairs. By learning domain-specific image–text embeddings from large-scale medical corpora, these encoders provide clinically aware visual representations that substantially improve transfer to downstream classification, retrieval, and reporting tasks over generic natural-image encoders.

On the decoder side, medical-domain large language models (LLMs) have been proposed as specialized generators for biomedical and clinical text. BioGPT [11] is a generative Transformer pretrained on large-scale PubMed abstracts, achieving state-of-the-art performance on multiple biomedical NLP benchmarks. Med-PaLM [12] adapts a general-purpose LLM (Flan-PaLM) to the medical domain via instruction tuning on MultiMedQA, reaching near-clinician-level accuracy on exam-style questions. These medical-domain decoders provide a natural generator for captioning and report-generation systems that require both accurate biomedical terminology and clinically appropriate reasoning.

Beyond using medical encoders and decoders in isolation, several recent systems propose full medical vision–language models (VLMs) that jointly process images and text and can generate free-form clinical descriptions. LLaVA-Med [13] extends a general vision–language assistant (LLaVA) to the biomedical domain via two-stage multimodal instruction tuning on PMC-15M figure–caption pairs and GPT-4–generated biomedical conversations. XrayGPT [14] specializes this paradigm to chest radiographs by combining a MedCLIP visual encoder with a Vicuna-based medical LLM through a learnable projection layer to support radiologyreport summarization and interactive question answering over X-rays. These medical VLMs show that pairing domainspecific visual encoders with instruction-tuned medical (or general) LLMs can substantially improve clinical faithfulness and versatility in medical image captioning and reporting.

Across image, video, and audio captioning, a common posthoc strategy is to first produce multiple candidate captions and then rerank them using a compatibility score computed in a shared multimodal embedding space. Fang et al. (2015) [15] captioning system generates many candidate sentences with a maximum entropy language model and then applies a Deep Multimodal Similarity Model together with sentence-level features to rerank the hypotheses, thereby operationalizing selection in an image–text semantic space. In the video domain, BLIP4video [16] first generates multiple candidate captions and then reranks them using both a video–text alignment score and a CIDEr-based n-gram similarity. Subsequently, two lightweight post-hoc procedures apply these scores at intra- and inter-inference stages, and the final caption is selected from the reordered set. SLAM-AAC [17] introduces CLAP-Refine as a plug-and-play inference step that computes audio–text similarity in a CLAP embedding space and reranks beam-search outputs to pick the final caption. Complementary to reranking, an alternative post-hoc selection paradigm uses a large language model to summarize the set of candidate captions into a single, comprehensive description—for example, IC3 [18] samples diverse captions from a base captioner and prompts a vision-free LLM to synthesize them into one consolidated caption.

If reranking is a test-time post-hoc procedure that selects the best sentence from a pool of hypotheses, Self-Critical Sequence Training (SCST) [19] internalizes the same selection criteria into the training objective so that the generator learns to produce better sentences outright. Building on this classical line of work, recent work in reinforcement and preference learning broadens the selection signal. PPO [20] has been applied to image-to-text generation as a trust-region policy-gradient alternative that controls policy shift and KL divergence while stabilizing CIDEr-driven optimization. DPO [21] subsequently emerged as a reference-light preference optimizer; in the clinical setting, RRG-DPO [22] curates preferred/dispreferred/sub-preferred pairs using biomedical-CLIP neighborhoods and abnormality conflicts, aligning report generation on X-ray and CT toward clinically faithful preferences. Finally, GRPO [23] introduces group-wise relative optimization to captioning: the model samples multiple hypotheses per image and maximizes within-group relative advantages under a KL constraint, reducing update variance and jointly improving stability and diversity relative to SCST.

## III. METHOD

## A. Base Model Architecture

The overall architecture of our proposed medical image captioning model is illustrated in Figure 1. The model consists of dual vision encoders, a Query Transformer (Q-Former), and a domain-adapted LLaMA decoder, which are described in detail in the following subsections.

![](images/b282af1c427987e774141ac84c826d67a867260e1fdd4e6e9de49aea3331f2ad.jpg)  
Fig. 1. The figure illustrates the architecture of a medical image captioning model that generates a final caption by fusing outputs from two vision encoders, followed by a Q-Former and a LLaMA decoder.; CC BY-NC [Al Mulhim et al.] [24]

1) Dual Encoder: To derive robust and semantically rich visual representations from medical images, we adopt an ensemble of two vision encoders. Specifically, we utilize SigLIP2 [25], a general-purpose image encoder pretrained on large-scale natural image-text pairs, and BioMedCLIP [10], a medical-domain-specific encoder trained on 15 million imagecaption pairs mined from PubMed Central. To address the lack of medical image knowledge in the original SigLIP2 model, we perform domain-specific pre-adaptation by fine-tuning it on the dataset provided by the ImageCLEF2025 Caption Prediction Task [26], [27]. This improves the encoder’s ability to capture domain-relevant visual features while preserving generalization capacity.

Following standard practice, we remove the classification heads from both encoders and use the token-level hidden states as feature maps. Let the output token sequences of BioMedCLIP and SigLIP2 be $\mathbf { H } _ { \mathrm { b i o c l i p } } \in \mathbb { R } ^ { B \times T _ { \mathrm { b i o } } ^ { - } \times d _ { \mathrm { b i o } } }$ and $\mathbf { H } _ { \mathrm { s i g l i p } } \in \mathbb { R } ^ { B \times T _ { \mathrm { s i g } } \times d _ { \mathrm { s i g } } }$ , respectively. We then compare three strategies for fusing these two sequences before passing them to the Q-Former as encoder hidden states.

First, in the simple feature concatenation baseline, we aim to minimally process the encoder outputs before feeding them into the Q-Former. Concretely, we extract a global representation from each sequence, denoted $\mathbf { f } _ { \mathrm { b i o c l i p } } ~ \in ~ \mathbb { R } ^ { B \times \bar { d } _ { \mathrm { b i o } } }$ and $\mathbf { f } _ { \mathrm { s i g l i p } } ~ \in ~ \mathbb { R } ^ { B \times d _ { \mathrm { s i g } } }$ , for example by applying global average pooling or taking a special classification token. These vectors are concatenated along the feature dimension to obtain $\mathbf { f } ~ = ~ [ \mathbf { f } _ { \mathrm { b i o c l i p } } ; \mathbf { f } _ { \mathrm { s i g l i p } } ] ~ \in ~ \mathbb { R } ^ { B \times D }$ , where $D = d _ { \mathrm { b i o } } + d _ { \mathrm { s i g } }$ . The resulting f is treated as a length-1 token sequence and used as the encoder hidden state for Q-Former cross-attention. This design corresponds to a late-fusion scheme that preserves the pretrained encoders’ representations with minimal distortion.

Second, in Bi-Directional Self-Attention Fusion, we concatenate the token sequences from the two encoders along the sequence dimension and feed the result into a lightweight multi-layer Transformer encoder to obtain a jointly contextualized representation through global self-attention. The fused token sequence takes the form $\mathbf { \bar { H } } _ { \mathrm { f u s e } } \in \mathbb { R } ^ { B \times ( T _ { \mathrm { b i o } } + T _ { \mathrm { s i g } } ) \times d }$ , where d denotes the hidden dimension of the fusion module, and $\mathbf { H } _ { \mathrm { f u s e } }$ is directly used as the encoder hidden state for the Q-Former.

Third, in Dual Cross-Attention Fusion, we keep the SigLIP2 and BioMedCLIP token sequences as two separate streams and apply a stack of bi-directional cross-attention blocks so that the two representations explicitly exchange information. The outputs of the final block are then concatenated to form a fused token sequence, which is passed to the Q-Former as the encoder hidden state.

All three fusion schemes are implemented within an otherwise identical captioning architecture and training configuration, allowing analysis of fusion behavior independent of decoder or optimization changes.

2) Query Transformer (Q-Former): To reduce redundancy and computational burden, we apply a Q-Former [28] that compresses the encoder hidden states into a fixed set of informative latent tokens. Let $\mathbf { H } _ { \mathrm { e n c } } \in \mathbb { R } ^ { B \times T _ { e } \times d _ { e } }$ denote the encoder hidden states provided by the selected fusion scheme. Let $\mathbf { Q } _ { 0 } \in \mathbb { R } ^ { B \times N _ { q } \times d _ { q } }$ denote a learnable set of query tokens. The Q-Former maps $( \mathbf { Q } _ { 0 } , \mathbf { H } _ { \mathrm { e n c } } )$ to $\mathbf { Z } \in \mathbb { R } ^ { B \times N _ { q } ^ { - } \times d _ { q } ^ { - } }$ through stacked Transformer layers composed of self-attention over the query tokens and cross-attention to $\mathbf { H } _ { \mathrm { e n c } }$ . This output is used for both caption generation and auxiliary concept classification.

To enhance medical grounding, we incorporate a multitask classification objective [29]. The output Z is mean-pooled across the query dimension to produce a global representation $\bar { \mathbf { z } } \in \mathbb { R } ^ { B \times d _ { q } }$ . This representation is passed through two linear classifiers: one to predict concept presence among $C$ Concept Unique Identifiers (CUIs) and another to predict $T$ coarse concept types. The overall training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c a p t i o n } } + \lambda \mathcal { L } _ { \mathrm { c l s } } , } \end{array}\tag{1}
$$

where $\mathcal { L } _ { \mathrm { c a p t i o n } }$ denotes the cross-entropy loss over caption tokens, $\mathcal { L } _ { \mathrm { c l s } }$ denotes the auxiliary multi-label margin loss, and λ is the weighting factor for the auxiliary term.

3) Caption Decoder: For the caption generation task, we adopt Bio-Medical LLaMA-3-8B [30], a domain-specialized variant of Meta-Llama-3-8B-Instruct [31], as the language decoder. The model has been fine-tuned on BioMedData, a high-quality biomedical dataset containing over 500,000 entries. The dataset comprises a blend of synthetic and manually curated samples, enabling robust generalization across a wide range of biomedical contexts. During training, the Q-Former output tokens are inserted as prefix embeddings that condition each decoding step on visual evidence. To enable efficient finetuning, we incorporate LoRA [32] modules into the decoder. This allows the model to adapt to medical image captioning tasks with minimal parameter updates while preserving the core language modeling capabilities of LLaMA.

## B. Post-processing

To further refine the raw captions generated by six distinct captioning models, we applied post-processing strategies to improve both clinical consistency and factual relevance. This section presents two major post-processing components: (1) summarization-based refinement using GPT APIs and (2) candidate caption reranking based on semantic and domainspecific metrics.

1) Summarization-based Refinement: We employed two GPT-4-based strategies for summarization [33] to consolidate the six candidate captions—each produced by a different model—into a single, medically accurate sentence. Both approaches aimed to improve readability, reduce redundancy, and ensure consistency with structured medical knowledge. The exact prompts used for each summarization method are provided in Table I.

Prompt-guided summarization: Six caption candidates, one produced by each captioning model, were aggregated and provided to GPT-4 using a fixed prompt template. The prompt requested a concise and clinically coherent summary under the assumption that all captions corresponded to the same medical image. This helped reduce redundancy, resolve inconsistencies, and unify expression styles across captions.

Chain-of-Thought Summarization: In this variant, the prompt explicitly encouraged chain-of-thought reasoning [34] before concluding the final summary. The intent was to increase factual consistency by encouraging the model to align each summary point with the underlying clinical evidence extracted from input captions. Empirically, this strategy improved alignment with structured medical entities.

2) Caption Reranking: To select the most appropriate caption among the generated candidates, we implemented a reranking module based on three different metrics: BioMed-CLIP image–text alignment, BLEURT self-consensus, and

TABLE I  
PROMPT TEMPLATES USED FOR GPT-4-BASED MEDICAL IMAGE CAPTION SUMMARIZATION.
<table><tr><td>Chain-of-Thought Summarization Template You are a board-certified radiologist.</td><td>Prompt-guided Template</td><td>Summarization</td></tr><tr><td>Task: 1) Parse each caption and list seman- tic components such as modality, anatomic site, and pathologies. 2) Build a consensus table based on token frequency. 3) Resolve conflicts by majority vote or by selecting the more specific description. 4) Compose one radiology-style sen- tence (approximately 35–45 words). Output: FINAL_CAPTION: &lt;your summary&gt; Captions: caption 1: {caption1 } caption 2: {caption2} caption 3: {caption3} caption 4: {caption4} caption 5: {caption5} caption 6: {caption6}</td><td>You are a radiologist summarizing multiple captions of a medical image into one detailed sentence. Instructions: 1) Integrate the imaging modal- ity, anatomical location, pathological findings, and relevant clinical de- tails. 2) Use medically correct and extrac- tive phrasing with high token over- lap; avoid paraphrasing unless syn- onymous medical terminology im- proves clarity. 3) Use present continuous tense with a subject-predicate-object structure. 4) Keep the summary natural, clini- cally accurate, and around 40 words. 5) If captions are inconsistent, prior- itize findings with the highest diag- nostic or therapeutic relevance. Captions: caption 1: {caption1} caption 2: {caption2} caption 3: {caption3} caption 4: {caption4} caption 5: {caption5} caption 6: {caption6}</td><td></td></tr></table>

The Chain-of-Thought strategy decomposes candidate captions into semantic units and aggregates them through consensus, whereas the Promptguided strategy directly synthesizes multiple captions into a clinically coherent single-sentence summary. Both templates are designed to generate standardized radiology-style descriptions from diverse candidate captions.

BioBERT centroid proximity. The overall framework of these reranking strategies is illustrated in Figure 2.

![](images/dd44bf269682cf9f1f35aa914d52e9e5bd9f85795f93d6d69c671b5c22b7acb7.jpg)  
Fig. 2. Overview of reranking methods: (a) BioBERT centroid similarity, (b) BLEURT self-consensus, (c) BioMedCLIP image-text alignment

BioMedCLIP image–text alignment: Each caption $c _ { i }$ is embedded into $\mathbf { v } _ { i }$ using the BioMedCLIP [10] text encoder, following the BioMedCLIP-based reranking strategy in [35], and the corresponding image is embedded into w using the image encoder. We compute the cosine similarity

$$
\mathrm { s i m } _ { i } = \cos ( \mathbf { v } _ { i } , \mathbf { w } ) = \frac { \mathbf { v } _ { i } \cdot \mathbf { w } } { \| \mathbf { v } _ { i } \| \cdot \| \mathbf { w } \| } ,
$$

which measures image–text alignment in a biomedical semantic space. The selected caption is $\hat { c } = \mathrm { a r g } \operatorname* { m a x } _ { i }$ sim<sub>i</sub>.

BLEURT self-consensus: BLEURT [36] estimates sentence quality via a regression head over BERT-style embeddings. For caption $c _ { i }$ among n candidates, we compute the leave-one-out average

$$
{ \mathrm { s c o r e } } _ { i } = { \frac { 1 } { n - 1 } } \sum _ { j \neq i } { \mathrm { B L E U R T } } ( c _ { i } , c _ { j } ) ,
$$

which favors captions that are semantically central to the candidate set and therefore robust to outliers. The selected caption is cˆ = arg max score .

BioBERT centroid proximity: All captions are embedded via BioBERT [37] as vectors $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { n }$ . The centroid

$$
\mathbf { v } _ { c } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { v } _ { i }
$$

represents the consensus semantic position. Each caption is then ranked by its Euclidean distance to the centroid [38], and the selected caption is $\hat { c } = \arg \operatorname* { m i n } _ { i } \| \mathbf { v } _ { i } - \mathbf { v } _ { c } \|$

## C. Reinforcement Learning with MedPAIR-SCST

To further align the decoder with clinically meaningful semantics, we fine-tune the model using a self-critical sequence training (SCST) objective augmented with a clinically informed reward. For each image $x _ { b } ,$ the current decoder policy samples K candidate captions, $\{ c _ { b , 1 } , \hdots , c _ { b , K } \}$ , using temperature-scaled top-k sampling and, optionally, top-p sampling.

Given the ground-truth caption $y _ { b } ,$ each candidate is assigned a composite reward,

$$
\begin{array} { r } { R ( c , y ) = \displaystyle \frac { 1 } { 3 } \Big ( \mathrm { B E R T S c o r e } _ { \mathrm { F 1 } } ( c ^ { \prime } , y ^ { \prime } ) } \\ { + \mathrm { R O U G E - } \imath _ { \mathrm { F 1 } } ( c ^ { \prime } , y ^ { \prime } ) } \\ { + \mathrm { U M L S } \ – \mathrm { F 1 } ( c , y ) \Big ) . } \end{array}\tag{2}
$$

where $c ^ { \prime }$ and $y ^ { \prime }$ denote lightly normalized captions obtained by lowercasing and replacing digits with the token “number,” whereas c and y denote the corresponding raw texts. BERTScore is computed with IDF reweighting, ROUGE-1 is measured as unigram F1, and UMLS-F1 is defined as the F1 score between the sets of Concept Unique Identifiers (CUIs) extracted from the raw texts using MedCAT. This reward jointly encourages lexical and semantic similarity as well as accurate recovery of UMLS concepts.

To obtain sequence-level log-likelihoods for policy-gradient updates, all sampled captions are re-scored in a single teacherforcing pass. Specifically, the image-conditioned prefix representation is concatenated with the token embeddings of each sampled caption, and the decoder is run in teacher-forcing mode. For each caption $^ { C _ { b , k } }$ , we sum token log-probabilities up to the first EOS token, mask padding positions, and normalize by the effective sequence length:

$$
\bar { \ell } _ { b , k } = \frac { 1 } { T _ { b , k } } \sum _ { t = 1 } ^ { T _ { b , k } } \log p _ { \theta } \big ( c _ { b , k , t } \mid c _ { b , k , < t } , x _ { b } \big ) ,\tag{3}
$$

where $T _ { b , k }$ denotes the number of valid tokens before EOS. We use $\ell _ { b , k }$ as a length-normalized sequence score to reduce trivial length effects.

Within each image-specific group b, the sampled rewards $\{ R _ { b , 1 } , \ldots , R _ { b , K } \}$ are first centered and then converted into normalized weights through a softmax with temperature τ :

$$
w _ { b , k } = \frac { \exp \left( ( R _ { b , k } - \bar { R } _ { b } ) / \tau \right) } { \sum _ { j = 1 } ^ { K } \exp \left( ( R _ { b , j } - \bar { R } _ { b } ) / \tau \right) } ,\tag{4}
$$

where $\bar { R } _ { b }$ is the mean reward for image b. We then define group-wise advantages as

$$
a _ { b , k } = w _ { b , k } - \frac { 1 } { K } ,\tag{5}
$$

so that $\textstyle \sum _ { k = 1 } ^ { K } a _ { b , k } = 0$ and only relative reward differences within the group contribute to the update. The resulting groupwise SCST loss is

$$
\mathcal { L } _ { \mathrm { g r o u p } } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } a _ { b , k } \bar { \ell } _ { b , k } ,\tag{6}
$$

which increases the likelihood of captions with positive groupwise advantages and decreases that of captions with negative advantages.

To align model scores more explicitly with the clinically informed reward, we further introduce a reference-free pairwise ranking loss. For each image b, we consider all ordered index pairs $\mathcal { P } _ { b } = \{ ( i , j ) \mid i \neq j , R _ { b , i } > R _ { b , j } \}$ , and penalize inconsistencies between reward ordering and model scores using a softplus margin:

$$
\mathcal { L } _ { \mathrm { p a i r } } = \frac { 1 } { \sum _ { b } | \mathcal { P } _ { b } | } \sum _ { b = 1 } ^ { B } \sum _ { ( i , j ) \in \mathcal { P } _ { b } } w _ { b , i j } \mathrm { s o f t p l u s } \Big ( m - \big ( \bar { \ell } _ { b , i } - \bar { \ell } _ { b , j } \big ) \Big ) ,\tag{7}
$$

where m is a small margin and $w _ { b , i j }$ is a non-negative weight that increases with the reward gap $R _ { b , i } - R _ { b , j }$ . This loss can be interpreted as a logistic pairwise preference objective defined on differences in sequence log-likelihood, encouraging higherreward captions to receive higher model scores without requiring a separate frozen reference policy or KL regularization.

The main reinforcement learning objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M e d P A I R } } = \mathcal { L } _ { \mathrm { g r o u p } } + \lambda _ { \mathrm { p a i r } } \mathcal { L } _ { \mathrm { p a i r } } , } \end{array}\tag{8}
$$

where $\lambda _ { \mathrm { p a i r } }$ controls the strength of the ranking signal. In practice, we additionally apply small regularization terms to the sampling weights and token-level likelihoods, such as entropybased penalties and a mild fluency constraint, to improve optimization stability. For clarity, these auxiliary stabilizers are omitted from the main objective.

In summary, the proposed Medical Pairwise Aligned Imagecaptioning with Reinforcement SCST (MedPAIR-SCST) objective combines (i) SCST-style self-critical optimization with group-normalized advantages, (ii) a clinically informed reward that explicitly scores UMLS concept recovery in addition to lexical and semantic similarity, and (iii) a reference-free pairwise ranking term that aligns reward ordering with model likelihoods. Together with lightweight auxiliary regularizers for optimization stability, this design improves both linguistic quality and clinical factuality without maintaining a separate frozen reference model, thereby avoiding the additional memory and system complexity typically associated with KLregularized reinforcement learning fine-tuning.

## IV. EXPERIMENTS

## A. Experimental Setups

We conduct experiments on the extended version of the ROCOv2 dataset [26], [39], specifically curated for the Image-CLEFmedical 2025 Caption Prediction Task [26]. Unlike the original ROCOv2 [39], this updated release includes additional manual annotations as well as a newly introduced test set for the 2025 challenge. The dataset configuration differs from prior versions: the previous test set from ROCOv2 has been reassigned as the validation set, and the prior validation set has been merged into the training set. The newly collected 2025 test set contains unseen images to evaluate generalization performance under updated task conditions. However, because the ground-truth captions for the 2025 test set are not released even after the end of the challenge, we report all quantitative results on the validation split. The resulting splits comprise 80,091 training images and 17,277 validation images. Each image is associated with a manually curated caption and UMLS concepts, making it suitable for both generation and concept detection tasks.

Model performance is evaluated using four metrics that assess both relevance and factuality. Relevance is measured with BERTScore (Recall with IDF), ROUGE-1 (F1), and BLEURT. BERTScore is computed with the microsoft/deberta-xlargemnli model using IDF scores derived from the validation set, and BLEURT uses the recommended BLEURT-20 checkpoint. All relevance metrics are calculated on lowercase, punctuationfree captions with numbers replaced by the token “number.” For factuality, UMLS Concept F1 is computed using MedCAT [40] with semantic type filtering via QuickUMLS.

Our model employs either BioMedCLIP or SigLIP2 as standalone vision encoders, or an ensemble configuration obtained by channel-wise concatenation of their outputs. When using Bio-Medical LLaMA-3-8B as the language decoder, visual features are processed by a 6-layer Q-Former with 32 learnable query tokens. In this configuration, the Q-Former maps the concatenated 2,304-dimensional visual representation to 4,096-dimensional embeddings compatible with the 8B decoder. The model optionally includes auxiliary classification heads that jointly predict 2,478 UMLS concepts and 21 coarse semantic types. The overall training objective is defined as a weighted sum of the captioning loss and the concept classification loss, with the weighting factor λ fixed to 0.3. Training is performed with the AdamW optimizer; the learning rate is linearly increased to $1 \times 1 0 ^ { - 4 }$ during the first epoch and then annealed to $1 \times 1 0 ^ { - 6 }$ over a total of 10 epochs. Experiments with the 8B decoder are conducted on a single NVIDIA H100 GPU with a batch size of 16 and a gradient accumulation step of 2. At inference time, we use beam search with a beam width of 3, a repetition penalty of 2.5, a length penalty of 2.0, and a minimum and maximum generation length of 8 and 64 tokens, respectively.

TABLE II  
SUMMARIZES THE PERFORMANCE OF SIX BACKBONE CONFIGURATIONS THAT DIFFER IN (I) VISION-ENCODER COMPOSITION AND (II) THE PRESENCE OF AUXILIARY CONCEPT HEADS. ALL MODELS SHARE THE Q-FORMER AND A BIOMEDLLAMA-3-8B DECODING STACK AND ARE TRAINED UNDER THE IDENTICAL HYPER-PARAMETER SCHEDULE.
<table><tr><td>Encoder</td><td>Aux.</td><td>BERT-R</td><td>ROUGE-1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td>BioMedCLIP</td><td>X</td><td>0.5845</td><td>0.2261</td><td>0.3100</td><td>0.1405</td></tr><tr><td>SigLIP2</td><td>X</td><td>0.5796</td><td>0.2194</td><td>0.3047</td><td>0.1397</td></tr><tr><td>Dual Encoder</td><td>X</td><td>0.5826</td><td>0.2305</td><td>0.3133</td><td>0.1514</td></tr><tr><td>BioMedCLIP</td><td>0</td><td>0.5860</td><td>0.2252</td><td>0.3100</td><td>0.1411</td></tr><tr><td>SigLIP2</td><td>0</td><td>0.5837</td><td>0.2289</td><td>0.3090</td><td>0.1438</td></tr><tr><td>Dual Encoder</td><td>0</td><td>0.5863</td><td>0.2347</td><td>0.3150</td><td>0.1528</td></tr></table>

In the lightweight 1B configuration, we keep the same vision encoder choices and Q-Former design (6 layers and 32 learnable query tokens), but project a 1,536-dimensional concatenated visual representation into a 2,048-dimensional embedding space to match the hidden size of the 1B decoder. The auxiliary heads and loss formulation (including λ = 0.3) are identical to those of the 8B model. Experiments with the 1B decoder are run on a single NVIDIA A100 GPU, while all remaining training and decoding hyperparameters (optimizer, number of epochs, learning rate schedule, batch size, gradient accumulation, and beam search configuration) are kept the same as in the 8B setting.

Additionally, for the 1B decoder, we perform a secondstage fine-tuning experiment, denoted MedPAIR-SCST, on top of the dual-encoder + auxiliary-task model after completing the first-stage supervised training. In this phase, we initialize from the stage-1 checkpoint and apply Self-Critical Sequence Training (SCST) for 2 epochs, evaluating on the validation set every 0.1 epoch to closely track performance. During SCST, we sample $K \ = \ 4$ candidate captions per image with a maximum generation length of 40 tokens. Sampling hyperparameters are set to a temperature of 0.9, top-k = 40, and $\mathrm { t o p } \mathrm { - } p ~ = ~ 0 . 8 5 ;$ the pairwise ranking component uses a margin of 0.02 and a pairwise weight of 0.3. Optimization is carried out with AdamW and a cosine-style learning rate schedule bounded between a minimum of $5 \times 1 0 ^ { - 6 }$ and a maximum of $1 \times 1 0 ^ { - 5 }$ , while all other optimization settings follow those of the supervised stage.

## B. Base Model

Table II reports the results obtained with the 8B decoder as we vary the choice of vision encoder and the presence of an auxiliary UMLS concept classification head. Among single-encoder configurations without the auxiliary task, BioMedCLIP outper forms SigLIP2 in both BERTScore Recall (0.5845 vs. 0.5796) and BLEURT (0.3100 vs. 0.3047), while the dual-encoder model further improves BLEURT to 0.3133 and achieves the highest UMLS F1 (0.1514) within this group. Adding the UMLS concept and type prediction auxiliary head consistently improves factuality while preserving or slightly enhancing relevance across all vision encoders: UMLS F1 increases from 0.1405 to 0.1411 for BioMedCLIP and from 0.1397 to 0.1438 for SigLIP2. The dual-encoder with the auxiliary task attains the best overall performance, achieving the highest scores on all four metrics (BERTScore Recall 0.5863, ROUGE-1 0.2347, BLEURT 0.3150, and UMLS F1 0.1528), indicating that combining complementary visual features with an explicit UMLS concept classification head is most effective for jointly improving semantic relevance and clinical faithfulness in the 8B decoder setting.

TABLE III  
SUMMARIZES THE PERFORMANCE OF SIX BACKBONE CONFIGURATIONS THAT DIFFER IN (I) VISION-ENCODER COMPOSITION AND (II) THE PRESENCE OF AUXILIARY CONCEPT HEADS. ALL MODELS SHARE THE Q-FORMER AND A BIOMEDLLAMA-3-1B DECODING STACK AND ARE TRAINED UNDER THE IDENTICAL HYPER-PARAMETER SCHEDULE.
<table><tr><td>Encoder</td><td>Aux.</td><td>BERT-R</td><td>BERT-R w/o IDF</td><td>ROUGE-1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td rowspan="6">BioMedCLIP SigLIP2 Dual Encoder BioMedCLIP SigLIP2</td><td>X</td><td>0.5714</td><td>0.6284</td><td>0.2285</td><td>0.3025</td><td>0.1317</td></tr><tr><td>X</td><td>0.5632</td><td>0.6227</td><td>0.2183</td><td>0.2937</td><td>0.1197</td></tr><tr><td>X</td><td>0.5775</td><td>0.6246</td><td>0.2382</td><td>0.3098</td><td>0.1450</td></tr><tr><td>0</td><td>0.5625</td><td>0.6189</td><td>0.2167</td><td>0.3000</td><td>0.1229</td></tr><tr><td>0</td><td>0.5577</td><td>0.6157</td><td>0.2067</td><td>0.2921</td><td>0.1099</td></tr><tr><td>0</td><td>0.5734</td><td>0.6298</td><td>0.2334</td><td>0.3065</td><td>0.1463</td></tr></table>

TABLE IV

COMPARISON OF THREE DUAL-ENCODER FUSION STRATEGIES IN TERMS OF CAPTIONING AND CLINICAL CONCEPT METRICS.
<table><tr><td>Fusion method</td><td>BERT-R</td><td>BERT-R w/o IDF</td><td>ROUGE-1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td>Simple Concat SA Fusion CA Fusion</td><td>0.5734 0.5383 0.5570</td><td>0.6298 0.5394 1</td><td>0.2334 0.1839 0.2064</td><td>0.3065 0.2893 0.2893</td><td>0.1463 0.1096 0.1238</td></tr></table>

Table III reports the results obtained with the 1B decoder as we vary the choice of vision encoder and the presence of an auxiliary UMLS concept classification head. Among configurations without the auxiliary task, the dualencoder model clearly outperforms the single-encoder variants, achieving the highest BERTScore Recall (0.5775), ROUGE-1 (0.2382), BLEURT (0.3098), and UMLS F1 (0.1450). In contrast, adding the auxiliary UMLS head to the single-encoder models yields no benefit and even degrades performance: for BioMedCLIP, UMLS F1 drops from 0.1317 to 0.1229, and for SigLIP2 from 0.1197 to 0.1099, accompanied by small decreases in relevance metrics. For the dual-encoder, the auxiliary task leads to a modest improvement in UMLS F1 (from 0.1450 to 0.1463) and BERTScore (Recall, no IDF; from 0.6246 to 0.6298), while slightly reducing BERTScore Recall, ROUGE-1, and BLEURT. Overall, in the 1B decoder setting the dual-encoder architecture remains effective, but the gains from introducing the UMLS concept classification head are limited and substantially weaker than those observed with the 8B decoder.

## C. Fusion Method

In Section III-A.1, we introduced three strategies for combining the outputs of the two visual encoders: simple feature concatenation, Bi-Directional Self-Attention Fusion, and Dual Cross-Attention (CA) Fusion. In this subsection, we compare these alternatives under an otherwise identical captioning architecture and training setup. The quantitative results are summarized in Table IV, which reports BERTScore-Recall (with and without IDF), ROUGE-1, BLEURT, and UMLS concept F1.

TABLE V  
VALIDATION-ONLY COMPARISON OF THREE RERANKING STRATEGIES APPLIED TO SIX CANDIDATE CAPTIONS PER IMAGE GENERATED BY THE 8B DECODER. THE BEST SCORE FOR EACH METRIC IS SHOWN IN BOLD.  
(\* CANDIDATES INCLUDE GPT-4 CHAIN-OF-THOUGHT AND PROMPT-GUIDED SUMMARIES.)
<table><tr><td>Reranker</td><td>BERT-R</td><td>ROUGE-1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td>BioMedCLIP</td><td>0.5873</td><td>0.2338</td><td>0.3130</td><td>0.1499</td></tr><tr><td>BLEURT*</td><td>0.5880</td><td>0.2368</td><td>0.3178</td><td>0.1539</td></tr><tr><td>bioBERT</td><td>0.5922</td><td>0.2409</td><td>0.3179</td><td>0.1552</td></tr><tr><td>Base Model (8B)</td><td>0.5826</td><td>0.2440</td><td>0.3176</td><td>0.1547</td></tr></table>

Across all reported metrics, the simple concatenation baseline achieves the best overall performance. It attains the highest BERTScore-Recall (0.5734), ROUGE-1 (0.2334), BLEURT (0.3065), and UMLS F1 (0.1463). Both attention-based fusion variants perform worse. Bi-Directional Self-Attention Fusion shows the largest drop, with BERTScore-Recall decreasing to 0.5383, ROUGE-1 to 0.1839, BLEURT to 0.2893, and UMLS F1 to 0.1096. Dual Cross-Attention Fusion shows a more moderate decline relative to simple concatenation (BERTScore-Recall 0.5570, ROUGE-1 0.2064, BLEURT 0.2893, and UMLS F1 0.1238), but it still does not outperform the concatenation baseline.

These results suggest that, under the limited medical image– report data regime considered here, adding extra self-attention or cross-attention fusion blocks to more tightly couple the dual-encoder representations does not improve downstream performance and may instead make optimization less effective. In contrast, preserving the representations produced by each pretrained encoder and allowing the Q-Former to attend directly over the concatenated token sequence leads to stronger and more consistent results across both captioning and clinical concept metrics. This trend is broadly consistent with prior observations that simple late-fusion strategies can remain competitive in multimodal settings, including UniCat [41], where simple late-fusion strategies based on independently pretrained encoders and embedding concatenation were shown to be robust and effective. Taken together, our results suggest that, in data-constrained medical imaging settings, it is often preferable to preserve the strengths of each pretrained encoder and adopt minimal fusion, rather than introducing more complex fusion modules.

## D. Post-Processing

As summarized in Table V, reranking based on a single embedding-based score criterion consistently shifts the performance profile across metrics. The bioBERT selector achieves the best BERTScore Recall (0.5922), BLEURT (0.3179), and UMLS F1 (0.1552), while the BLEURT-based selector is a close second (0.5880 / 0.3178 / 0.1539). By comparison, the BioMedCLIP selector performs worse on all reported metrics (BERTScore Recall 0.5873, ROUGE-1 0.2338, BLEURT 0.3130, UMLS F1 0.1499). Notably, the non-reranked base model retains the highest ROUGE-1 (0.2440), suggesting that inference-time candidate selection tends to improve semantic and clinical alignment (BERTScore, BLEURT, and UMLS F1) at the expense of a small reduction in n-gram overlap.

TABLE VI  
COMPARISON OF OUR 1B BASE MODEL AND MEDPAIR-SCST AGAINST PRIOR METHODS ACROSS TEXT QUALITY AND CLINICAL CONCEPT FIDELITY METRICS ON THE MEDICAL IMAGE CAPTIONING BENCHMARK.
<table><tr><td>Method</td><td>BERT-R</td><td>BERT-R w/o IDF</td><td>ROUGE- 1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td>R2Gen [42]</td><td>0.5519</td><td>0.5967</td><td>0.2212</td><td>0.2710</td><td>0.1495</td></tr><tr><td>CvTdistilGPT2 [43]</td><td>0.5805</td><td>0.6207</td><td>0.2299</td><td>0.2867</td><td>0.1318</td></tr><tr><td>Base Model(1B)</td><td>0.5775</td><td>0.6246</td><td>0.2382</td><td>0.3098</td><td>0.1450</td></tr><tr><td>Base Model(1B) + MedPAIR-SCST</td><td>0.6000</td><td>0.6584</td><td>0.2755</td><td>0.3122</td><td>0.1821</td></tr></table>

In addition to reranking, we also evaluated GPT-4-guided summarization as an alternative post-processing strategy. Although aggregating multiple candidate captions into a single concise output is intuitively appealing, the summarizationbased approaches performed worse than reranking in our quantitative evaluations. In particular, both prompt-guided summarization and chain-of-thought prompting often produced less precise outputs and occasionally introduced clinically irrelevant or hallucinated content. These weaknesses were especially apparent in factual grounding metrics such as UMLS F1, suggesting that generative summarization may abstract away or omit clinically important concepts during compression. A more detailed comparison of these behaviors is provided in Table VII.

## E. Comparison with other models

We evaluated the effectiveness of our approach by comparing it with established medical image captioning methods. The quantitative results are reported in Table VI. Our 1B base model shows competitive, and in some cases improved, performance relative to R2Gen [42] and CvTdistilGPT2 [43]. In particular, the base model achieves higher ROUGE-1 (0.2382) and BLEURT (0.3098), indicating better surface-level alignment and overall generation quality.

Further improvements are observed when MedPAIR-SCST is applied to the base model, as shown in Table VI. BERTScore Recall increases from 0.5775 to 0.6000, and BERTScore Recall without IDF rises from 0.6246 to 0.6584. ROUGE-1 shows a substantial gain from 0.2382 to 0.2755, while BLEURT improves slightly from 0.3098 to 0.3122. Most notably, UMLS F1 increases from 0.1450 to 0.1821, suggesting that MedPAIR-SCST improves clinically grounded concept fidelity in addition to linguistic quality. Overall, these results support the effectiveness of our reinforcement-learning-based optimization in producing more clinically faithful captions.

## V. CONCLUSION

This study treats clinical alignment as the central objective of medical image captioning and proposes a two-axis alignment framework that separates training-time alignment from inference-time alignment, allowing their effects to be examined independently. We build a baseline captioning model consisting of dual visual encoders (BioMedCLIP and SigLIP2), a Q-Former with learnable query tokens, and a LLaMAbased decoder, and we systematically analyze, under matched training conditions, the contributions of (i) vision-encoder composition and (ii) auxiliary UMLS concept/type classification heads. Under the 8B decoder setting, the configuration combining dual encoders with auxiliary UMLS supervision achieves the most stable and strongest performance across BERTScore, ROUGE-1, BLEURT, and UMLS F1, suggesting that complementary visual representations and explicit concept learning can jointly improve semantic relevance and clinical factuality. In contrast, under the 1B decoder setting, although the benefits of dual encoders largely remain, the auxiliary heads provide limited gains and can even degrade performance, indicating that the effectiveness of auxiliary supervision depends on decoder capacity and the model’s ability to absorb additional training signals.

For inference-time alignment, we apply embedding-based reranking over candidates generated by six independently trained models. Selection strategies based on BioBERT centroid proximity and BLEURT self-consensus consistently improve BERTScore, BLEURT, and UMLS F1, demonstrating that even lightweight post-hoc selection can substantially enhance clinical relevance and concept fidelity. By contrast, GPT-4-based summarization refinement, despite its intuitive appeal, underperforms reranking in quantitative evaluation, plausibly because compressive rewriting may dilute salient findings or introduce extraneous content. Moreover, in the comparison of dual-encoder fusion designs, we find that simple token concatenation yields better overall results across metrics than more complex self-attention or cross-attention fusion modules, reinforcing the effectiveness of minimal fusion in data-constrained clinical settings.

For training-time alignment, we propose MedPAIR-SCST, a reinforcement-style objective that operates stably without a reference model or KL regularization. By combining group-level reward aggregation with a reference-free pairwise ranking term, MedPAIR-SCST shifts the model’s generation distribution toward clinically more accurate outputs and yields substantial gains, particularly in UMLS F1, while preserving linguistic quality. Comparisons with prior baselines such as R2Gen and CvTdistilGPT2 show that our 1B-based model is competitive, and that applying MedPAIR-SCST further improves alignment, supporting the practical value of distribution-level optimization for clinical faithfulness.

Nevertheless, this work has several limitations. First, because the ground-truth annotations for the ImageCLEFmedical 2025 test split are not publicly available, our quantitative analyses rely on the validation split, which limits our ability to fully assess generalization to hidden test data. Second, our reward design is based on a combination of BERTScore, ROUGE-1, and UMLS F1, which may still introduce biases toward particular surface forms or terminology distributions. Third, the limited effectiveness of auxiliary UMLS supervision in the 1B setting suggests that interactions among auxiliary task difficulty, loss weighting, label quality, and decoder capacity require more careful study.

Future directions include: (1) evaluation under hidden-test conditions and on external multi-institutional datasets to better quantify clinical generalization; (2) the design of more structured factuality rewards that encode modality-, anatomy-, and finding-level constraints while explicitly modeling boundary cases and contradictions between concepts; (3) improved auxiliary learning efficiency for lightweight decoders through lossweight scheduling and hierarchical target formulations; and (4) extension of MedPAIR-SCST to larger decoders, together with minimal-fusion principles, to achieve stronger clinical reliability under limited-data regimes.

TABLE VII  
QUANTITATIVE COMPARISON OF GPT-BASED SUMMARIZATION METHODS FOR MEDICAL CAPTION REFINEMENT USING CANDIDATE CAPTIONS GENERATED BY THE 8B DECODER.
<table><tr><td>Method</td><td>BERT-R</td><td>ROUGE-1</td><td>BLEURT</td><td>UMLS F1</td></tr><tr><td>CoT</td><td>0.5705</td><td>0.2032</td><td>0.3197</td><td>0.1236</td></tr><tr><td>Prompt-guided</td><td>0.5723</td><td>0.2016</td><td>0.3176</td><td>0.1242</td></tr></table>

In conclusion, this study provides a systematic treatment of selection-based inference alignment and distribution-based training alignment for medical image captioning. Through principled dual-encoder feature composition and the referencefree MedPAIR-SCST objective, we advance clinically more reliable caption generation and provide empirical evidence that, in resource- and data-constrained medical imaging settings, carefully designed alignment signals and their placement can be more effective than increasing architectural complexity.

## APPENDIX

## SUMMARIZATION-BASED REFINEMENT RESULTS

GPT-4-based summarization of candidate captions can produce more concise outputs; however, it did not lead to consistent improvements in clinical alignment. For instance, the prompt-guided summarization variant achieved a BERTScore Recall of 0.5723 and a UMLS concept F1 of 0.1242, both lower than those of the best direct-generation configuration in our study (Table II). Similarly, no systematic gains were observed in ROUGE-1 or BLEURT. We attribute this behavior to the compressive nature of summarization: during abstraction, medically salient local findings and uncertainty expressions present in the candidates may be omitted, which is detrimental in medical captioning. Moreover, summarization is a rephrasing operation rather than a selection mechanism; instead of correcting factual errors introduced during generation, it may introduce additional errors by altering phrasing and inadvertently changing clinical meaning. Detailed results are reported in Table VII.

## ACKNOWLEDGMENT

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)(RS-2024-00360176).

[1] D. R. Beddiar, M. Oussalah, and T. Seppanen, “Automatic captioning¨ for medical imaging (mic): a rapid review of literature,” Artificial intelligence review, vol. 56, no. 5, pp. 4019–4076, 2023.

[2] Z. Zhao, L. Alzubaidi, J. Zhang, Y. Duan, and Y. Gu, “A comparison review of transfer learning and self-supervised learning: Definitions, applications, advantages and limitations,” Expert Systems with Applications, vol. 242, p. 122807, 2024.

[3] M. Limbu and D. Banerjee, “Medblip: Fine-tuning blip for medical image captioning,” arXiv preprint arXiv:2505.14726, 2025.

[4] H. Guan and M. Liu, “Domain adaptation for medical image analysis: a survey,” IEEE Transactions on Biomedical Engineering, vol. 69, no. 3, pp. 1173–1185, 2021.

[5] S. Umirzakova, S. Ahmad, L. U. Khan, and T. Whangbo, “Medical image super-resolution for smart healthcare applications: A comprehensive survey,” Information Fusion, vol. 103, p. 102075, 2024.

[6] F. Perez-Garc ´ ´ıa, H. Sharma, S. Bond-Taylor, K. Bouzid, V. Salvatelli, M. Ilse, S. Bannur, D. C. Castro, A. Schwaighofer, M. P. Lungren, et al., “Exploring scalable medical image encoders beyond text supervision,” Nature Machine Intelligence, pp. 1–12, 2025.

[7] Z. Lu, H. Li, N. A. Parikh, J. R. Dillman, and L. He, “Radclip: Enhancing radiologic image analysis through contrastive language–image pretraining,” IEEE Transactions on Neural Networks and Learning Systems, 2025.

[8] P. Kaliosis, J. Pavlopoulos, F. Charalampakos, G. Moschovis, and I. Androutsopoulos, “A data-driven guided decoding mechanism for diagnostic captioning,” in Findings ofthe Associationfor Computational Linguistics: ACL 2024 (L.-W. Ku, A. Martins, and V. Srikumar, eds.), (Bangkok, Thailand), pp. 7450–7466, Association for Computational Linguistics, Aug. 2024.

[9] S. Bannur, S. Hyland, Q. Liu, F. Perez-Garcia, M. Ilse, D. C. Castro, B. Boecking, H. Sharma, K. Bouzid, A. Thieme, et al., “Learning to exploit temporal structure for biomedical vision-language processing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15016–15027, 2023.

[10] S. Zhang, Y. Xu, N. Usuyama, H. Xu, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri, et al., “Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs,” arXiv preprint arXiv:2303.00915, 2023.

[11] R. Luo, L. Sun, Y. Xia, T. Qin, S. Zhang, H. Poon, and T.-Y. Liu, “Biogpt: generative pre-trained transformer for biomedical text generation and mining,” Briefings in bioinformatics, vol. 23, no. 6, p. bbac409, 2022.

[12] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, et al., “Large language models encode clinical knowledge,” Nature, vol. 620, no. 7972, pp. 172– 180, 2023.

[13] C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T. Naumann, H. Poon, and J. Gao, “Llava-med: Training a large language-and-vision assistant for biomedicine in one day,” Advances in Neural Information Processing Systems, vol. 36, pp. 28541–28564, 2023.

[14] O. C. Thawakar, A. M. Shaker, S. S. Mullappilly, H. Cholakkal, R. M. Anwer, S. Khan, J. Laaksonen, and F. Khan, “Xraygpt: Chest radiographs summarization using large medical vision-language models,” in Proceedings of the 23rd workshop on biomedical natural language processing, pp. 440–448, 2024.

[15] H. Fang, S. Gupta, F. Iandola, R. Srivastava, L. Deng, P. Dollar, J. Gao,´ X. He, M. Mitchell, J. C. Platt, C. L. Zitnick, and G. Zweig, “From captions to visual concepts and back,” in Computer Vision and Pattern Recognition, pp. 1473 – 1482, 2014.

[16] Z. Yue, Y. Liu, L. Zhang, L. Yao, and Q. Jin, “Rucaim3-tencent at trecvid 2022: Video to text description,” in Proceedings of TRECVID, 2022.

[17] W. Chen, Z. Ma, X. Li, X. Xu, Y. Liang, Z. Zheng, K. Yu, and X. Chen, “SLAM-AAC: Enhancing audio captioning with paraphrasing augmentation and CLAP-refine through LLMs,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1 – 5, 2025.

[18] D. Chan, A. Myers, S. Vijayanarasimhan, D. Ross, and J. Canny, “IC3: Image captioning by committee consensus,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 8975 – 9003, 2023.

[19] S. J. Rennie, E. Marcheret, Y. Mroueh, J. Ross, and V. Goel, “Selfcritical sequence training for image captioning,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1179 – 1195, 2017.

[20] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[21] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is

secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53728–53741, 2023.

[22] H. Liu, D. Wei, Z. Xu, X. Wu, Y. Zheng, and L. Wang, “Rrg-dpo: Direct preference optimization for clinically accurate radiology report generation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 552–562, Springer, 2025.

[23] X. Liang, “Group relative policy optimization for image captioning,” arXiv preprint arXiv:2503.01333, 2025.

[24] O. N. Al Mulhim, “Huge thoracic aortic aneurysm presenting with jaundice: A case report,” Vascular Health and Risk Management, pp. 1– 4, 2022.

[25] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[26] H. Damm, T. M. G. Pakull, H. Becker, B. Bracke, B. Eryilmaz, L. Bloch, R. Brungel, C. S. Schmidt, J. R ¨ uckert, O. Pelka, H. Sch ¨ afer, A. Idrissi-¨ Yaghir, A. B. Abacha, A. G. S. de Herrera, H. Muller, and C. M.¨ Friedrich, “Overview of ImageCLEFmedical 2025 – medical concept detection and interpretable caption generation,” in CLEF 2025 Working Notes, CEUR Workshop Proceedings, (Madrid, Spain), CEUR-WS.org, September 9–12 2025.

[27] B. Ionescu, H. Muller, D.-C. Stanciu, A.-G. Andrei, A. Radzhabov,¨ Y. Prokopchuk, S¸ tefan, Liviu-Daniel, M.-G. Constantin, M. Dogariu, V. Kovalev, H. Damm, J. Ruckert, A. Ben Abacha, A. Garc¨ ´ıa Seco de Herrera, C. M. Friedrich, L. Bloch, R. Brungel, A. Idrissi-Yaghir,¨ H. Schafer, C. S. Schmidt, T. M. G. Pakull, B. Bracke, O. Pelka, B. Ery-¨ ilmaz, H. Becker, W.-W. Yim, N. Codella, R. A. Novoa, J. Malvehy, D. Dimitrov, R. J. Das, Z. Xie, H. M. Shan, P. Nakov, I. Koychev, S. A. Hicks, S. Gautam, M. A. Riegler, V. Thambawita, P. Halvorsen, D. Fabre, C. Macaire, B. Lecouteux, D. Schwab, M. Potthast, M. Heinrich, J. Kiesel, M. Wolter, and B. Stein, “Overview of ImageCLEF 2025: Multimedia retrieval in medical, social media and content recommendation applications,” in Experimental IR Meets Multilinguality, Multimodality, and Interaction, Proceedings of the 16th International Conference of the CLEF Association (CLEF 2025), (Madrid, Spain), Springer Lecture Notes in Computer Science LNCS, September 9-12 2025.

[28] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning, pp. 19730– 19742, PMLR, 2023.

[29] E. Hirsch, G. Dawidowicz, and A. Tal, “Medrat: Unpaired medical report generation via auxiliary tasks,” in European Conference on Computer Vision, pp. 18–35, Springer, 2024.

[30] ContactDoctor, “ContactDoctor-Bio-Medical: A High-Performance Biomedical Language Model.” https://huggingface.co/ ContactDoctor/Bio-Medical-Llama-3-8B, 2024. Accessed: 2025-06-16.

[31] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[32] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al., “Lora: Low-rank adaptation of large language models.,” ICLR, vol. 1, no. 2, p. 3, 2022.

[33] D. M. Chan, A. Myers, S. Vijayanarasimhan, D. A. Ross, and J. Canny, “Ic3: Image captioning by committee consensus,” arXiv preprint arXiv:2302.01328, 2023.

[34] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24824–24837, 2022.

[35] Z. Zhang, B. Wang, W. Liang, Y. Li, X. Guo, G. Wang, S. Li, and G. Wang, “Sam-guided enhanced fine-grained encoding with mixed semantic learning for medical image captioning,” in ICASSP 2024- 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1731–1735, IEEE, 2024.

[36] T. Sellam, D. Das, and A. P. Parikh, “Bleurt: Learning robust metrics for text generation,” arXiv preprint arXiv:2004.04696, 2020.

[37] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, “Biobert: a pre-trained biomedical language representation model for biomedical text mining,” Bioinformatics, vol. 36, no. 4, pp. 1234–1240, 2020.

[38] S. Lamsiyah, A. El Mahdaouy, B. Espinasse, and S. E. A. Ouatik, “An unsupervised method for extractive multi-document summarization based on centroid approach and sentence embeddings,” Expert Systems with Applications, vol. 167, p. 114152, 2021.

[39] J. Ruckert, L. Bloch, R. Br¨ ungel, A. Idrissi-Yaghir, H. Sch¨ afer, C. S.¨ Schmidt, S. Koitka, O. Pelka, A. B. Abacha, A. G. S. de Herrera, H. Muller, P. Horn, F. Nensa, and C. M. Friedrich, “ROCOv2: Radiology¨ objects in context version 2, an updated multimodal image dataset,” Scientific Data, vol. 11, no. 1, 2024.

[40] N. C. Codella, Y. Jin, S. Jain, Y. Gu, H. H. Lee, A. B. Abacha, A. Santamaria-Pang, W. Guyman, N. Sangani, S. Zhang, et al., “Medimageinsight: An open-source embedding model for general domain medical imaging,” arXiv preprint arXiv:2410.06542, 2024.

[41] J. Crawford, H. Yin, L. McDermott, and D. Cummings, “Unicat: Crafting a stronger fusion baseline for multimodal re-identification,” arXiv preprint arXiv:2310.18812, 2023.

[42] Z. Chen, Y. Song, T.-H. Chang, and X. Wan, “Generating radiology reports via memory-driven transformer,” arXiv preprint arXiv:2010.16056, 2020.

[43] A. Nicolson, J. Dowling, and B. Koopman, “Improving chest x-ray report generation by leveraging warm starting,” Artificial intelligence in medicine, vol. 144, p. 102633, 2023.