# Token-Oriented Semantic Communication with Pretrained Vision Transformers

Jiwoong Im, Minwoo Kim, Jaeho Lee, Member, IEEE, Yo-Seb Jeon, Member, IEEE, and Yongjune Kim, Member, IEEE

Abstract—Token communications realize the semantic communication principle at the granularity of transformer tokens, providing a promising direction for client–server collaborative inference in resource-constrained edge systems. However, directly transmitting token embeddings presents two practical challenges: substantial communication cost and limited interoperability across model-specific token embedding spaces. To address these challenges, we propose a token-oriented semantic communication framework. In this framework, token-level task relevance determines which compressed image latents are transmitted, enabling token-granular transmission without directly transmitting token embeddings. The framework is modular, coordinating three pretrained components—a lightweight client-side vision transformer (ViT), a learned image compression (LIC) model, and a large server-side ViT—without end-to-end training. The key enabler is the one-to-one spatial alignment between ViT patch tokens and the LIC latent vectors, which allows token-level task relevance to directly determine which latent vectors are transmitted. Building on this alignment, token-aligned LIC selectively transmits taskrelevant latents, layer-selective attention rollout estimates token relevance from a selected range of attention layers in a single forward pass, and surrogate token substitution adapts the frozen server model by optimizing a single learnable token. Experiments on ImageNet show that the proposed framework achieves a more favorable rate–accuracy trade-off than recent semantic communication schemes, hand-crafted codecs, and task-agnostic LIC models.

Index Terms—Semantic communications, token communications, edge computing, learned image compression, collaborative inference, vision transformer.

## I. INTRODUCTION

Recent advances in transformer architectures have fundamentally reshaped the internal representation and processing of modern artificial intelligence (AI) models [1]. Transformers process inputs as a sequence of tokens, each of which serves as a unified computational unit across multiple layers. This tokenized formulation has proven effective across a wide range of modalities, including language, images, audio, and video [1]–[6]. In particular, token representations offer a structured, modular interface that is naturally compatible with global context modeling via attention mechanism [1]. As a result, tokens are now widely regarded not only as an internal feature format for transformer models, but also as a candidate unit for computation and communication.

Meanwhile, with the emergence of AI-native wireless networks, communication paradigms are progressively shifting beyond conventional bit-level communication toward semantic communications [7]–[13]. Rather than optimizing humanperceived reconstruction fidelity quantified in terms of bitlevel distortion, semantic communications attempt to enhance communication efficiency by selectively transmitting information relevant to downstream task objectives. Tokens provide a natural unit for this shift, and recent studies have accordingly investigated token communications, which realize the semantic communication principle at the granularity of transformer tokens [14]–[19]. Operating at the token level reveals which tokens are relevant to the downstream task, so that the communication payload can be concentrated on inference-relevant tokens rather than the entire source, as in conventional bit-level communication.

Realizing this advantage by transmitting token representations themselves, however, entails two practical considerations. The first concern is communication cost: each token consists of a high-dimensional embedding vector whose dimensionality increases with model capacity, so transmitting even a reduced number of tokens may incur a substantial payload. The second concern is the limited interoperability of token embedding spaces across independently trained models, as these embedding spaces can differ substantially because of differences in architectures, pretraining procedures, and downstream objectives, even at an identical model dimensionality [20]. To address both considerations, several studies jointly train the transmitter and receiver in an end-to-end manner, typically with a shared codebook that converts token embeddings into codeword indices [14], [21]–[24]. Such fully integrated designs are effective within a given deployment, yet they reduce modularity, incur retraining costs, and couple the communication strategy to a specific downstream task.

To retain the benefits of token-level processing while addressing these considerations, we propose a token-oriented semantic communication framework, considering image classification as the downstream task. Rather than compressing or exchanging token embeddings, the proposed framework compresses the input image with learned image compression (LIC), which exploits the statistics of natural images to attain a content-adaptive rate–distortion performance competitive with conventional image compression [25]–[28]. The proposed framework integrates token-level task relevance into this compression. The key enabler is the spatial alignment between visual tokens and LIC latent vectors: in the widely used architectures, the vision transformer (ViT) patch size equals the LIC encoder’s spatial downsampling factor, so each visual token corresponds to exactly one latent vector [2], [26]– [30]. Task relevance is thereby incorporated into compression without modifying the LIC architecture or introducing additional training, and the communication payload is reduced by compressing in the image domain rather than in the modelspecific embedding space. We use the term token-oriented to emphasize that token-level task relevance determines which token-aligned LIC latent vectors are transmitted, enabling transmission at token granularity without directly transmitting token embeddings.

The resulting framework is fully modular, coordinating three separately pretrained models for client–server collaborative inference: a lightweight client-side ViT, an LIC-based neural compressor, and a large server-side ViT. Specifically, the client-side ViT estimates the task relevance of visual tokens, and this token-level relevance is used to guide the selection of latent representations within the compressor. The neural compressor then reduces communication payload by transmitting only the selected latent representations, while the server-side ViT performs downstream inference from the reconstructed image. Since no token embedding is exchanged across models, their embedding spaces need not be compatible, and each component can be replaced without retraining the others from scratch.

Prior semantic communication frameworks that use tokenlevel relevance to guide transmission without directly transmitting token embeddings can likewise be regarded as tokenoriented [31]–[34]. However, they differ in the transmitted representation, sending image patches in the pixel domain, either raw or scalar-quantized. Operating in the LIC latent domain thus distinguishes the proposed framework within this class and enhances the rate–accuracy trade-off.

Upon this framework, we introduce three complementary components, each acting on one of the pretrained models, to further optimize the rate–accuracy trade-off:

Token-aligned LIC: Token-level task relevance is integrated into an LIC model to reduce the transmission rate while preserving task-relevant visual information. Exploiting the structural alignment between the visual tokens of ViT and the latent representations of LIC on a shared spatial grid, the token-level relevance map directly indexes the latents to be transmitted. Our approach selectively transmits only the latents at task-relevant positions, reducing the transmission rate while retaining compatibility with the underlying LIC architecture.

• Layer-selective attention rollout: The client-side ViT estimates token-level task relevance precisely by aggregating attention weights over a selected range of transformer layers. This selective aggregation provides more accurate relevance estimates than the last-layer attention commonly adopted in prior semantic communication frameworks [31]–[34], the attention rollout accumulated over all layers [35], and even gradient-weighted attention [36]. Moreover, unlike gradient-weighted attention [36], it requires only a single forward pass, avoiding the memory and latency overhead of gradient computation that is demanding for resource-constrained clients.

• Learnable surrogate token substitution: Server-side inference with partial visual information is improved by substituting the token positions of the unselected regions with a single learnable surrogate token. Inference without any adaptation, whether using the directly reconstructed image or only the selected patches, degrades classification accuracy on the server, whereas retraining the large server backbone is costly and alters its behavior on complete inputs. Instead, we optimize only the single surrogate token with the backbone kept frozen, providing parameter-efficient input-level adaptation that improves inference with partial visual information while preserving the original complete-input behavior. It further improves the robustness of server-side inference over unreliable channels, where the tokens affected by channel impairments are likewise replaced by the surrogate token.

Built from three pretrained models without end-to-end training, the proposed framework enables modular and communication-efficient collaborative inference. Across the evaluated rate regime, it attains a more favorable rate– accuracy trade-off than both recent semantic communication schemes [33], [34], which exploit task relevance without efficient compression, and task-agnostic LIC models [26]– [28], which optimize reconstruction without task relevance.

The remainder of this paper is organized as follows. Section II reviews related work on token-level semantic communications, learned image compression, attention-guided importance estimations, and learnable input tokens. Section III presents an overview of our proposed token-oriented semantic communication framework. Section IV introduces the main technical contributions of this work, optimizing the rate– accuracy trade-off of the proposed framework: token-aligned LIC, layer-selective attention rollout, and surrogate token substitution. Section V reports the experimental results, and Section VI concludes the paper.

## II. RELATED WORK

## A. Token-Level Semantic Communications

Semantic communication frameworks operating at token granularity differ in what they transmit, how they identify task-relevant content, and how the receiver is adapted for inference. As summarized in Table I, VQ-enabled token communications transmit the token representations themselves as vector-quantized codebook indices, which requires end-to-end joint training to establish a codebook shared between the two ends [14], [22]–[24]. Token-oriented communications instead guide transmission by token-level relevance while operating on pretrained models, yet prior frameworks in this class transmit image patches in the pixel domain [31]–[34]. The proposed framework is likewise token-oriented but transmits LIC latents, thereby improving rate efficiency while retaining modularity, i.e., operation with pretrained models without end-to-end joint training.

## B. Learned Image Compression

With the advancement of machine learning, the learned image compression (LIC) paradigm [25]–[28] has achieved rate–distortion performance competitive with conventional hand-crafted image compression. Among the various LIC architectures, we build upon the nonlinear transform coding framework [25] and its extension to a hyperprior network [26], which underlies many widely adopted LIC models. The endto-end compression pipeline maps an input image X to a reconstruction $\widehat { \mathbf { X } } \in \mathbb { R } ^ { \widehat { H } \times W \times C }$ through the following sequential pipeline:

TABLE I  
COMPARISON OF TOKEN-LEVEL SEMANTIC COMMUNICATION FRAMEWORKS FOR COLLABORATIVE INFERENCE
<table><tr><td rowspan=1 colspan=1>Framework</td><td rowspan=1 colspan=1>TransmittedRepresentation</td><td rowspan=1 colspan=1>Token ImportanceEstimation</td><td rowspan=1 colspan=1>Server-SideAdaptation</td><td rowspan=1 colspan=1>Modularity</td></tr><tr><td rowspan=1 colspan=1>VQ-Based Token Comm. [14], [22]-[24]</td><td rowspan=1 colspan=1>VQ Codebook Indices</td><td rowspan=1 colspan=1>Auxiliary Network or None</td><td rowspan=1 colspan=1>Joint Training</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Prior Token-Oriented Comm. [31]–[34]</td><td rowspan=1 colspan=1>Raw or Quantized Pixels</td><td rowspan=1 colspan=1>Last-Layer Attention</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1> $\checkmark$ </td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>Token-AlignedLIC Latents</td><td rowspan=1 colspan=1>Layer-SelectiveAttention Rollout</td><td rowspan=1 colspan=1>Surrogate Token(Frozen Backbone)</td><td rowspan=1 colspan=1>√</td></tr></table>

$$
\mathbf { X } \ { \overset { g _ { a } } { \longrightarrow } } \ \mathbf { X } ^ { c } \ { \overset { Q } { \longrightarrow } } \ { \widehat { \mathbf { X } } } ^ { c } \ { \overset { g _ { s } } { \longrightarrow } } \ { \widehat { \mathbf { X } } } .\tag{1}
$$

Here, an encoder $g _ { a }$ extracts the latent representation ${ \bf X } ^ { c } \in  \large $ $\mathbb { R } ^ { N \times D _ { c } }$ , where $D _ { c }$ denotes the model dimension of the neural compressor. Q introduces discretization to obtain the quantized latent $\widehat { \mathbf { X } } ^ { c }$ , and a decoder $g _ { s }$ reconstructs the image.

To effectively remove the remaining spatial redundancy of latents ${ \bf X } ^ { c }$ , [26] introduced a hyperprior network that transmits side information. Specifically, a hyper encoder $h _ { a }$ extracts a hyper latent $\mathbf { Z } \ = \ h _ { a } ( \mathbf { X } ^ { c } )$ , which is quantized to Zb. A hyper decoder $h _ { \mathscr { s } }$ then decodes $\hat { \mathbf { Z } }$ to estimate the conditional distribution of $\widehat { \mathbf { X } } ^ { c }$ , modeled as a normal distribution with zero mean. Subsequently, [27] extended this architecture to jointly predict both the mean $\mu$ and variance σ of the elements of the latents. These predicted parameters define the precise probability mass function required by the arithmetic coder, allowing the entropy coding process to closely approach the theoretical Shannon entropy limit.

## C. Attention-Guided Importance Estimation

Vision transformer (ViT) is a transformer-based architecture for computer vision tasks [2]. To perform inference, an input image $\dot { \mathbf { X } } \in \mathbb { R } ^ { H \times W \times C }$ is reshaped into a sequence of flattened 2D patches $\mathbf { X } ^ { p } \in \mathbb { R } ^ { N \times ( P ^ { 2 } \cdot C ) }$ , where (H, W), C, and $( P , P )$ denote the resolution of the original image, the number of channels, and the resolution of each image patch, respectively. Here, $\begin{array} { r l } { N \ = \ } & { { } { \frac { H W } { P ^ { 2 } } } } \end{array}$ denotes the resulting number of image patches. Each patch $\mathbf { x } _ { i } ^ { p }$ is then tokenized into a visual token $\bf \bar { x } _ { i } ^ { t } \in \mathbb { R } ^ { 1 \times D }$ by linear projection, forming the visual token matrix $\mathbf { X } ^ { t } \in \dot { \mathbb { R } } ^ { N \times D }$ whose ith row is $\mathbf { x } _ { i } ^ { t }$ . Then, the class token $\mathbf { x } _ { \mathrm { c l s } }$ is prepended to the sequence of visual tokens. The input embedding of the ViT encoder $\widetilde { \mathbf { X } } ^ { t } \in \mathbb { R } ^ { ( N + 1 ) \times D }$ is given by

$$
\begin{array} { r } { \widetilde { \mathbf { X } } ^ { t } = \left[ \mathbf { x } _ { \mathrm { c l s } } ; \mathbf { X } ^ { t } + \mathbf { E } \right] , } \end{array}\tag{2}
$$

where $\mathbf { E } \in \mathbb { R } ^ { N \times D }$ denotes the standard learnable position embedding.

Since token interactions in ViT are primarily mediated through multi-head self-attention blocks, the attention weights can be interpreted as modeling global dependencies among tokens that contribute to the final inference. Accordingly, many recent studies have estimated token relevance ${ \textbf { R } } \in$ $\mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$ and token-level task relevance $\bar { \mathbf { r } } \in \mathbb { R } ^ { N }$ from the attention weights computed during downstream inference. Below, we briefly review representative attention-guided token relevance methods. For brevity, we denote the attention weights at layer l and head h by $\bar { \mathbf { A } } _ { l , h } \in \mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$

• Last-layer attention: The most direct approach estimates token relevance using the attention weights from the final transformer layer:

$$
{ \bf R } = \mathbb { E } _ { h } \left[ { \bf A } _ { L , h } \right] ,\tag{3}
$$

where $L$ denotes the number of transformer layers and $\mathbb { E } _ { h } [ \cdot ]$ represents averaging over attention heads. It is widely used in prior token-oriented communication studies owing to its simplicity [31]–[34].

• Attention rollout: Attention rollout aggregates attention weights across all transformer layers to approximate the overall token dependencies captured by the model [35]:

$$
\widetilde { \mathbf { A } } _ { l } = \mathbb { E } _ { h } \left[ \mathbf { A } _ { l , h } \right] + \mathbf { I } _ { N + 1 } , \quad \mathbf { R } = \widetilde { \mathbf { A } } _ { L } \widetilde { \mathbf { A } } _ { L - 1 } \cdot \cdot \cdot \widetilde { \mathbf { A } } _ { 1 } ,\tag{4}
$$

where $\mathbf { I } _ { N + 1 } \in \mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$ denotes the identity matrix that accounts for the residual connection in ViT.

• Gradient-weighted attention: Gradient-weighted attention additionally reweights the attention rollout based on the gradient of the top-1 predicted logit with respect to the attention weights [36]. Although it provides precise taskrelevance estimation, its reliance on gradient computation constitutes a practical limitation, which precludes its deployment on resource-constrained clients.

The above methods share a common final step: the tokenlevel task relevance ¯r is extracted from the first row of R, which quantifies the relevance of the class token, the input to the classification head, to the visual tokens. Excluding the class-token self-relevance $R _ { 1 , 1 }$ , we normalize the N entries corresponding to the visual tokens as

$$
\bar { r } _ { i } = \frac { R _ { 1 , i + 1 } } { \sum _ { j = 1 } ^ { N } R _ { 1 , j + 1 } } ,\tag{5}
$$

so that ¯r forms a distribution over the visual tokens.

## D. Learnable Input Tokens

Learnable input tokens, i.e., trainable vectors inserted into the input sequence, have been used to adapt or train vision transformers. Visual prompt tuning prepends a few learnable prompt tokens to the input and optimizes only them while freezing the backbone, enabling parameter-efficient task adaptation [37]. On the other hand, masked image modeling inserts a single shared mask token at masked positions, jointly optimized with the backbone for self-supervised reconstruction during pretraining [38], [39]. The proposed surrogate token substitution combines both aspects: a single learnable token substitutes missing positions and is optimized with the backbone frozen. It differs in purpose, being trained for the downstream classification objective to compensate for missing visual content at inference.

![](images/623f25aae2bbc1b7f2a27f8ca2a8e246aa399662d38998fb62c6758054d44cef.jpg)  
Fig. 1. Overview of the proposed token-oriented semantic communication framework based on pretrained ViT models. The circled numbers in the figure follow the same order as the line numbers in Algorithm 1.

## III. OVERVIEW OF THE PROPOSED FRAMEWORK

This section presents the client–server collaborative inference procedure of the proposed token-oriented semantic communication framework, deferring its technical components to Section IV. We consider a setting in which a resourceconstrained client cooperates with a computationally capable server. The client employs a lightweight ViT-based classifier (e.g., DeiT-Tiny [29]) together with a neural compressor, whereas the server employs a large-scale ViT classifier (e.g., DeiT-III-Large [30]) and the corresponding neural decompressor.

Algorithm 1 Inference Procedure of the Proposed Framework   
Input: Input image $\overline { { \mathbf { X } , } }$ token-selection threshold $\overline { { \delta , } }$ client   
tokenizer $f _ { \alpha } : \mathbf { X } \to \mathbf { X } ^ { t } .$ , server tokenizer $f _ { \beta } : \widehat { \mathbf { X } }  \widehat { \mathbf { X } } ^ { t } .$   
client classifier $f _ { \theta } : { \bf X } ^ { t }  \bar { \bf r } ,$ server classifier $f _ { \phi } : \widetilde { \mathbf { X } } ^ { t } \to y ,$   
neural compressor $f _ { \psi } : ( { \bf X } , { \bf s } )  { \bf b }$ , and neural decompressor   
$f _ { \xi } : ( \widehat { \mathbf { b } } , \widehat { \mathbf { s } } ) \to \widehat { \mathbf { X } } .$   
Output: Classification result y.   
1: $\mathbf { X } ^ { t } \gets f _ { \alpha } ( \mathbf { X } )$ ▷ Client-side visual token extraction   
2: $\bar { \mathbf { r } }  f _ { \theta } ( \mathbf { X } ^ { t } )$ ▷ Token-level task relevance   
3: Obtain token-selection map $\mathbf { s } ~ \in ~ \{ 0 , 1 \} ^ { N }$ from ¯r using   
threshold δ   
4: b $ f _ { \psi } ( { \bf X } , { \bf s } )$ ▷ Compressed bitstream   
5: Transmit $( \mathbf { b } , \mathbf { s } )$ over the communication channel   
6: Server receives $( \widehat { \mathbf { b } } , \widehat { \mathbf { s } } )$   
7: Xb $\gets f _ { \xi } ( \widehat { \mathbf { b } } , \widehat { \mathbf { s } } )$ ▷ Reconstructed image   
8: $\widehat { \mathbf { X } } ^ { t } \gets \dot { f } _ { \beta } ( \widehat { \mathbf { X } } )$ ▷ Server-side visual token extraction   
9: Construct $\widetilde { \mathbf { X } } ^ { t }$ from $\widehat { \mathbf { X } } ^ { t }$ via surrogate token substitution   
10: $y \gets f _ { \phi } ( \widetilde { \mathbf { X } } ^ { t } )$ ▷ Final inference result   
11: Return y to the client

Our overall framework is illustrated in Fig. 1, where the circled numbers follow the same order as the line numbers in Algorithm 1. First, the client extracts visual tokens $\mathbf { X } ^ { t }$ from the input image X and computes the token-level task relevance ¯r using the lightweight ViT. Based on these scores, the client selects the task-relevant visual tokens and obtains a tokenselection map $\mathbf { s } ~ \in ~ \{ 0 , 1 \} ^ { N }$ , where N denotes the number of visual tokens. The client then generates the compressed bitstream b, which contains the visual latent representations corresponding to the positions of the selected tokens and the hyperpriors, and transmits $( \mathbf { b } , \mathbf { s } )$ to the server. After transmission, the server receives (bb, bs) and reconstructs the image $\widehat { \mathbf { X } } ,$ which is then converted into server-side visual tokens $\widehat { \mathbf { X } } ^ { t }$ . From $\widehat { \mathbf { X } } ^ { t }$ , the input sequence $\widetilde { \mathbf { X } } ^ { t }$ of the server-side ViT is constructed via surrogate token substitution. Finally, the server-side predictor produces the classification result y and returns it to the client. The overall inference procedure is described in Algorithm 1.

We integrate token selection into LIC by retaining only the visual latent representations corresponding to the task-relevant token positions. Consequently, the latent payload is reduced according to the number of selected tokens. The transmitted bitstream also includes the hyperpriors and the selection map ${ \mathbf { s } } ,$ but the bits allocated to them are sufficiently small compared to those used for the visual latent representations. As a result, the bitstream size decreases approximately linearly with the number of selected tokens, as analyzed in Section IV-A.

## IV. MAIN TECHNICAL COMPONENTS

In this section, we present three technical components of the proposed framework: (1) token-aligned learned image compression, (2) layer-selective attention rollout, and (3) learnable surrogate token substitution. The client first estimates the token-level task relevance of visual tokens via layerselective attention rollout on a lightweight ViT model. It then performs token-wise on/off selection based on the estimated relevance, and the LIC model generates a bitstream containing the latent representations at the selected token positions and the hyperpriors. At the server, the reconstructed patches at unselected positions are unreliable because the corresponding latents are not transmitted. To prevent these unreliable patches from degrading downstream inference, the server replaces the tokens at those positions with a learnable surrogate token.

## A. Token-Aligned Learned Image Compression

Given the token-level task relevance ¯r estimated from the attention weights of the client-side ViT (Section IV-B), the client selects task-relevant tokens to reduce communication overhead. A straightforward approach is to transmit the selected image patches directly in the pixel domain [33] or after importance-aware scalar quantization [34]. However, both approaches encode each patch independently and thus cannot exploit the statistical redundancy that learned image compression captures, remaining far less rate-efficient than LIC models, as demonstrated in Section V. This motivates integrating token selection into the compression process itself.

We propose token-aligned $L I C ,$ which integrates attentionaware token selection with an LIC model by operating in the latent domain, as shown in Fig. 2. The key enabler is a structural alignment between the visual tokens of ViT and the latent representations of LIC: the encoder maps an input image X to a latent representation ${ \bf X } ^ { c }$ , which is quantized to $\hat { \hat { \mathbf { X } } } ^ { c } \in \mathbb { R } ^ { N \times D _ { c } }$ with spatial resolution $\textstyle \left( { \frac { H } { P } } , { \frac { W } { P } } \right)$ , yielding $\begin{array} { r } { N = \frac { H W } { P ^ { 2 } } } \end{array}$ spatial positions identical to the number of visual tokens. Note that both architectures conventionally reduce the input resolution by the same factor $P = 1 6 \colon$ ViT through a patch size of (16, 16) [2], [29], [30], and LIC through four stride-2 convolutional stages [26]–[28]. This one-to-one correspondence between the ith visual token $\mathbf { x } _ { i } ^ { t }$ and the ith latent vector $\widehat { \mathbf { x } } _ { i } ^ { c }$ , denoting the ith rows of $\mathbf { X } ^ { t }$ and $\widehat { \mathbf { X } } ^ { c }$ , respectively, allows the token-selection map $\mathbf { s } \in \{ 0 , 1 \} ^ { N }$ to directly index the latent representations, controlling which latent vectors are included in the transmitted bitstream without modifying the LIC architecture.

Given the task relevance $\bar { \mathbf { r } } \in \mathbb { R } ^ { N }$ , we construct the selection map s using the attention-sum threshold selection of [33], where $s _ { i } = 1$ indicates that the ith token is selected. Tokens are selected in descending order of relevance until the cumulative sum of relevance first exceeds a preset threshold $\delta ,$ so that the number of selected tokens varies across images according to semantic content, such as object size and inference difficulty. We denote the resulting sets of selected and unselected spatial indices as $S = \{ i \mid s _ { i } = 1 \}$ and $\mathcal { U } = \{ i \mid s _ { i } = 0 \}$ respectively.

1) Selective Latent Transmission: To selectively include only task-relevant latent representations in the transmitted bitstream, we exploit the structure of the hyperprior-based entropy model [26]. Specifically, the distribution of the dth element of the ith latent $\widehat { x } _ { i , d } ^ { c }$ is fully parameterized by the predicted mean $\mu _ { i , d }$ and scale $\sigma _ { i , d }$ , obtained from the hyper decoder $h _ { s }$ using the quantized hyperprior $\widehat { \textbf { Z } } [ 2 7 ]$ . Since the latent elements are modeled as conditionally independent given $\widehat { \mathbf { Z } } ,$ each element can be entropy-coded and decoded using only its own predicted parameters $\left( \mu _ { i , d } , \sigma _ { i , d } \right)$ , without access to any other latent element. Consequently, the client guides the arithmetic encoder to encode only the latent vectors at positions $\textit { i } \in \textit { S }$ , bypassing entropy coding for positions $i \in \mathcal { U }$ entirely.

![](images/f302f78406a478fa941d54f9258101ee31eff3af95c361e893111d9310af5ad9.jpg)  
Fig. 2. Proposed token-aligned learned image compression framework. $Q , A E ,$ and AD refer to the quantizer, arithmetic encoder, and arithmetic decoder, respectively. After latent encoding, only the latent representations at taskrelevant token positions are retained and compressed into a bitstream. At the server, missing latent positions are imputed with hyperprior-predicted means before image reconstruction by the neural decoder.

The transmitted bitstream comprises the tuple $\left( \widehat { \mathbf { X } } _ { S } ^ { c } , \widehat { \mathbf { Z } } , \mathbf { s } \right)$ where $\widehat { \mathbf { X } } _ { S } ^ { c } \in \mathbb { R } ^ { | S | \times D _ { c } }$ stacks the retained latent vectors $\widehat { \mathbf { x } } _ { i } ^ { c } , i \in$ $s ,$ in ascending order of spatial index. This selective encoding reduces the communication cost; the resulting rate reduction is quantified in the theoretical analysis below.

2) Theoretical Analysis of Communication Efficiency: To analyze the communication efficiency, we establish a theoretical bound demonstrating that an approximately linear reduction in bit rate is achievable. The analysis assumes an ideal entropy coder in which the code length of each symbol approaches its self-information.

The number of bits required to encode the retained latents is given by

$$
\boldsymbol { B } _ { \widehat { \mathbf { X } } _ { \mathcal { S } } ^ { c } } = \left\lceil - \sum _ { i \in \mathcal { S } } \sum _ { d } \log _ { 2 } p \Big ( \widehat { x } _ { i , d } ^ { c } \mid \widehat { \mathbf { Z } } \Big ) \right\rceil .\tag{6}
$$

Analogously, the number of bits required to encode the complete latent matrix $\widehat { \mathbf { X } } ^ { c }$ , i.e., over both selected and unselected positions, is

$$
B _ { \widehat { \mathbf { X } } ^ { c } } = \left\lceil - \sum _ { i = 1 } ^ { N } \sum _ { d } \log _ { 2 } p \Big ( \widehat { x } _ { i , d } ^ { c } \mid \widehat { \mathbf { Z } } \Big ) \right\rceil .\tag{7}
$$

Transmitting the complete latent representation together with the side information would then require

$$
\boldsymbol { B } _ { \mathrm { t o t } } = { B } _ { \widehat { \mathbf { X } } ^ { c } } + { B } _ { \widehat { \mathbf { Z } } } ,\tag{8}
$$

where $\boldsymbol { B } _ { \widehat { \mathbf { Z } } }$ denotes the number of bits allocated to the quantized hyperprior. As the proposed framework transmits only the retained latents, however, the actual number of transmitted bits is

$$
\boldsymbol { B } = B _ { \widehat { \mathbf { X } } _ { s } ^ { c } } + B _ { \widehat { \mathbf { Z } } } + B _ { \mathbf { s } } ,\tag{9}
$$

where $B _ { \mathrm { s } }$ denotes the number of bits allocated to the binary selection map.

Consequently, the ratio of the actual to the total bits admits the upper bound

$$
\frac { B } { B _ { \mathrm { t o t } } } \leq \eta + \frac { \rho } { \rho + ( 1 - \rho ) k } , \quad \mathrm { w h e r e ~ } k = \frac { \bar { H } _ { \mathcal { U } } } { \bar { H } _ { S } } ,\tag{10}
$$

where $\begin{array} { r } { \rho = \frac { | \boldsymbol { s } | } { | \boldsymbol { s } | + | \boldsymbol { u } | } } \end{array}$ denotes the token selection ratio, $\eta \ : = \ :$ $\frac { B _ { \hat { \mathbf { z } } } + B _ { \mathbf { s } } } { B _ { \mathrm { t o t } } }$ denotes the side-information overhead ratio including the hyperprior and the selection map, and $\bar { H } _ { S }$ and $\bar { H } _ { \mathcal { U } }$ denote the average self-information per latent over the selected and unselected positions, respectively.

In the regime $k \approx 1$ , where the selected and unselected regions exhibit comparable information density, the bound simplifies to

$$
\frac { B } { B _ { \mathrm { t o t } } } \lesssim \eta + \rho ,\tag{11}
$$

showing that the proposed selective transmission yields a rate reduction approximately linear in the selection ratio $\rho ,$ offset only by the side-information overhead η. The bound further characterizes the nonlinear regimes: the reduction is sublinear when the unselected region is information-sparse $( k < 1 )$ , and superlinear when it is information-dense relative to the selected region (k > 1).

3) Predicted-Mean Imputation: Before latent decoding, the decoder imputes the unselected latent positions with their predicted means, derived from the received hyperprior Zb following the mean-scale hyperprior architecture [27]. From a statistical perspective, the predicted mean is the conditional expectation of the local latent feature given the context captured by the hyperprior, and is thus a principled substitute for the missing latent. This choice matters because the LIC decoder uses convolutional layers with overlapping receptive fields: the reconstruction of each selected patch is influenced by the latents at neighboring positions, so the imputed values directly affect the quality of the neighboring selected patches. Substituting the unselected latents with their conditional expectations preserves the local statistical consistency of the latent map, improving downstream classification accuracy over zero padding, as shown in Section V-E.

Furthermore, since $\widehat { \pmb { \mu } } \in \mathbb { R } ^ { N \times D _ { c } }$ is already produced by the hyper decoder for entropy decoding, the imputation introduces negligible additional computational overhead. The ith row of the imputed latent representation $\widetilde { \mathbf { X } } ^ { c }$ at the decoder is

$$
\widetilde { \mathbf { x } } _ { i } ^ { c } = s _ { i } \widehat { \mathbf { x } } _ { i } ^ { c } + \left( 1 - s _ { i } \right) \widehat { \pmb { \mu } } _ { i } , \quad i = 1 , \ldots , N ,\tag{12}
$$

where $\widehat { \pmb { \mu } } _ { i }$ denotes the ith row of ${ \widehat { \mu } } .$ . Note that predicted-mean imputation is introduced solely to improve the reconstruction

![](images/0552917c28f0649f32caa4bd3d21f405a299c8d6170d21e9fc51ebc0dc70e984.jpg)  
Fig. 3. Visualization of token-level task relevance from DeiT-Tiny [29]. The first column shows example ImageNet images labeled as ‘great gray owl, ‘Pembroke,’ and ‘polecat,’ respectively. The remaining columns, from left to right, show task-relevance maps obtained using last-layer attention, attention rollout, and layer-selective attention rollout from the 7th to the 12th layers, respectively.

of the selected image patches: the reconstructed patches at unselected positions are not forwarded to the server-side predictor, as described in Section IV-C.

## B. Layer-Selective Attention Rollout

In transformer architectures, token interactions are primarily mediated by multi-head self-attention, whose attention weights capture global dependencies among tokens. For image classification with ViTs, the final prediction is produced from the class token, so the attention from the class token to the visual tokens naturally indicates how much each visual token contributes to the classification task.

Along this line, last-layer class-token attention has been widely used for task relevance estimation owing to its simplicity [31]–[34]. However, as shown in Fig. 3, last-layer attention often assigns high relevance to non-object regions, overestimating the task relevance of background patches. This limitation may arise because visual tokens are progressively contextualized across transformer layers [35], [36], [40]: by the final layer, each token aggregates information from other tokens and no longer represents only its own patch, so the class-token attention at a single layer does not faithfully reflect which patches originally contributed to the prediction.

Attention rollout was originally proposed to trace attention propagation among language tokens in natural language processing [35]. To account for the progressive contextualization described above, it multiplies the attention matrices across all layers, thereby explicitly tracking the information flow from the input to the final representation. As shown in Fig. 3, the resulting map is largely concentrated on the main object. However, when the selected patches are transmitted for serverside inference, attention rollout yields lower classification accuracy than last-layer attention unless only a small fraction of patches is retained [33], indicating that accurate object localization does not necessarily translate into accurate taskrelevance estimation.

![](images/d0cd7ef627c09d98036e5dd5a516fe9815aa7deac3957ba1a82d9d166630d13b.jpg)  
Fig. 4. Visualization of class-token attention weights across all layers of DeiT-Tiny. The original image is ‘polecat,’ same as Fig. 3. Note that the visualized attentions are averaged along the head dimension.

To understand why both methods fall short, we visualize the class-token attention weights across all layers of DeiT-Tiny [29] in Fig. 4. The attention patterns are noisy and weakly interpretable in the early layers, concentrate on the main object in the middle layers, and drift toward subsets of the background in the final layers. This layer-dependent behavior explains the preceding observations: last-layer attention inherits the background drift of the final layers, whereas attention rollout accumulates the noise of the early layers. These observations motivate aggregating attention over a selected range of layers that excludes the noisy early layers, rather than relying on a single layer or on all layers.

Therefore, we propose layer-selective attention rollout, which aggregates attention weights over a selected range of ViT layers. The layer-selective attention rollout from the $L _ { s }$ th layer to the $L _ { e } \mathrm { t h }$ layer, where $L _ { s } \leq L _ { e } ,$ , is computed as

$$
\widetilde { \mathbf { A } } _ { l } = \mathbb { E } _ { h } \left[ \mathbf { A } _ { l , h } \right] + \mathbf { I } _ { N + 1 } , \quad \mathbf { R } = \widetilde { \mathbf { A } } _ { L _ { e } } \widetilde { \mathbf { A } } _ { L _ { e } - 1 } \cdot \cdot \cdot \widetilde { \mathbf { A } } _ { L _ { s } } ,\tag{13}
$$

and the token-level relevance ¯r is then obtained via (5). For DeiT-Tiny with L = 12 layers, we set $( L _ { s } , L _ { e } ) = ( 7 , 1 2 )$ determined by the layer-range study in Section V-F. The last layer is included because it directly shapes the class token consumed by the classification head [2], while the noisy early layers are excluded. Examples of the resulting relevance maps are shown in Fig. 3, assigning relevance to the main object together with parts of its surrounding regions.

As shown in Section V-F, layer-selective attention rollout estimates task relevance more accurately than last-layer attention and attention rollout, and matches or exceeds gradientweighted attention [36] across our main operating region. Achieving this accuracy without gradients is what matters in practice: gradient-weighted attention requires a backward pass through the client-side ViT at every inference, and the additional computation and peak-memory footprint for storing intermediate activations are costly on resource-constrained clients. Layer-selective attention rollout attains this accuracy, exceeding gradient-weighted attention on our main client configuration, from a single forward pass, making precise taskrelevance estimation practical on such clients.

## C. Surrogate Token Substitution

Through token-aligned LIC, the server obtains a reconstructed image $\widehat { \mathbf { X } } \in \breve { \mathbb { R } } ^ { H \times W \times C }$ that retains the original spatial resolution, in which the patches at unselected positions are synthesized from the hyperprior-predicted means and thus reflect only a coarse statistical estimate of the original content at those positions. To prevent these patches from serving as misleading visual evidence for the server-side predictor, we introduce a single learnable token, termed the surrogate token, that replaces the tokens at unselected positions and is optimized while the pretrained backbone is kept frozen. The modified input token sequence $\widetilde { \mathbf { X } } ^ { t } \in \mathbb { R } ^ { ( N + 1 ) \times D }$ is constructed by prepending the class token $\mathbf { x } _ { \mathrm { c l s } }$ and assigning the surrogate token $\mathbf { x } _ { \mathrm { s u r } } \in \mathbb { R } ^ { 1 \times D }$ to the unselected positions:

$$
\begin{array} { r l } & { \quad \widetilde { \mathbf { x } } _ { 1 } ^ { t } = \mathbf { x } _ { \mathrm { c l s } } , } \\ & { \widetilde { \mathbf { x } } _ { i + 1 } ^ { t } = s _ { i } \widehat { \mathbf { x } } _ { i } ^ { t } + \left( 1 - s _ { i } \right) \mathbf { x } _ { \mathrm { s u r } } + \mathbf { e } _ { i } , \quad i = 1 , \ldots , N , } \end{array}\tag{14}
$$

(15)

where $\widetilde { \mathbf { x } } _ { i } ^ { t }$ and $\mathbf { e } _ { i }$ denote the ith row of the resulting token sequence $\widetilde { \mathbf { X } } ^ { t }$ and the position embedding E in (2), respectively. We optimize $\mathbf { x } _ { \mathrm { s u r } }$ for the downstream classification objective with the backbone frozen, so that the surrogate token learns to indicate the absence of information rather than supply misleading visual evidence.

Without any adaptation, the server can perform inference directly on the reconstructed image by replacing $\mathbf { X } ^ { t }$ in (2) with $\widehat { \mathbf { X } } ^ { t }$ (direct-reconstruction inference), or exploit the variablelength input property of ViTs to use only the reconstructed patches at the selected positions (selected-patch inference). While some accuracy reduction is unavoidable when inferring from partial visual information, both adaptation-free baselines leave the frozen predictor unadapted to the incomplete input and thus miss the accuracy that adaptation could recover. At the opposite extreme, retraining the server-side ViT on incomplete images recovers this margin but is computationally expensive [37], [41], [42], and updating the backbone weights can degrade performance on complete or near-complete inputs, where the pretrained DeiT-III-Large [30] already attains its peak accuracy of 86.81 % on ImageNet.

Surrogate token substitution occupies the middle ground between these extremes: the cost of adaptation is a single learnable vector of $D$ parameters (D = 1024 for DeiT-III-Large), and since the backbone remains intact, the predictor’s behavior on complete inputs is identical to that of the original pretrained model. Section V-G demonstrates that it maintains downstream accuracy more effectively than both adaptationfree baselines.

Surrogate token substitution further improves the robustness of server-side inference over unreliable channels. When the received bitstream differs from the transmitted one because of uncorrected channel impairments, a subset of the transmitted LIC latents may be erased. In this case, the server can replace the tokens at the affected positions with the surrogate token, reducing the impact of erasures on classification accuracy; this robustness gain is evaluated over a packet-erasure channel in Section V-C.

TABLE II  
MODEL COMPLEXITY AND CLASSIFICATION ACCURACY ON IMAGENET-1K [26], [29], [30], [43]
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=3>Parameters(million)</td><td rowspan=1 colspan=1>Memory(MB)</td><td rowspan=1 colspan=1>MACs(G)</td><td rowspan=1 colspan=1>ClassificationAccuracy (%)</td></tr><tr><td rowspan=2 colspan=1>DeiT-TinyDeiT-SmallDeiT-BaseDeiT-III-Large</td><td rowspan=2 colspan=3>5.722.186.6304.4</td><td rowspan=2 colspan=1>22.988.2346.31217.5</td><td rowspan=2 colspan=1>1.264.6117.5861.6</td><td rowspan=2 colspan=1>72.279.881.886.8</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Neural Comp.Neural Decomp.</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>5.5</td><td rowspan=2 colspan=1>22.118.0</td><td rowspan=2 colspan=1>3.212.76</td><td rowspan=2 colspan=1>==</td></tr><tr><td rowspan=1 colspan=3>4.5</td></tr></table>

## V. EXPERIMENTAL RESULTS

We evaluate the proposed token communication framework on ImageNet classification. After describing the experimental settings (Section V-A), we first present the main results: the rate–accuracy trade-off against prior token-oriented semantic communication baselines, hand-crafted codecs, and LIC frameworks (Section V-B), the robustness of surrogate token substitution under a packet-erasure channel (Section V-C), and the additional gain of entropy-aware image transmission (Section V-D). We then conduct ablation studies that isolate the contribution of each technical component: token-aligned LIC (Section V-E), layer-selective attention rollout (Section V-F), and surrogate token substitution (Section V-G).

## A. Experimental Settings

In the proposed system, we assume that the client deploys DeiT-Tiny [29], while the server employs DeiT-III-Large [30], reflecting the disparity in memory and computational capacity between the client and the server. As shown in Table II, the accuracy gap between the two models is substantial: DeiT-III-Large attains 86.8 %, whereas DeiT-Tiny reaches only 72.2 %. However, this accuracy improvement comes with substantially higher resource requirements, including approximately 50× larger memory consumption and computational complexity than those of DeiT-Tiny. In contrast, the hyperprior-based neural compressor [26], [27] deployed on the client side requires only 22.1 MB of memory, making it suitable for resourceconstrained clients. Overall, the client-side models, consisting of DeiT-Tiny and the neural compressor, require only 45 MB of memory and 4.47 GMACs, whereas the server-side models, consisting of DeiT-III-Large and the neural decompressor, require more than 1.2 GB of memory and approximately 64.36 GMACs.

We evaluate the framework on ImageNet [43] for image classification. Each image is center-cropped and resized to a resolution of (224, 224) pixels. We adopt publicly available pretrained ViT parameters, where DeiT-III-Large is additionally pretrained on ImageNet-21k [30]. The LIC model and the surrogate token are trained separately, with the settings described below.

• LIC training: We train a hyperprior model [26] with a mean hyperprior [27] for image reconstruction on ImageNet, setting the channel dimensions of the latent representation and the hyperprior to 192 and 128, respectively. A separate LIC model is trained for each considered value of λ, while the remaining optimization settings follow [26], [27].

• Surrogate token training: Following Section IV-C, we optimize a single surrogate token $\mathbf { x } _ { \mathrm { s u r } } \in \mathbb { R } ^ { 1 \times D }$ for the classification objective while keeping the pretrained backbones frozen. Surrogate token training emulates the proposed pipeline: the client selects task-relevant tokens via layer-selective attention rollout under threshold $\delta _ { \mathrm { s u r } }$ and transmits a bitstream compressed by an LIC model pretrained with $\lambda _ { \mathrm { s u r } }$ , after which the server replaces the unselected positions with the surrogate token before inference. The token is trained for 20 epochs, with the remaining settings following [39].

The selection map $\mathbf { s } \in \{ 0 , 1 \} ^ { N }$ accompanies the encoded image data $( \widehat { \mathbf { X } } _ { S } ^ { c } , \widehat { \mathbf { Z } } )$ so that the server can both reconstruct the image and identify the positions to be replaced by the surrogate token. At our resolution, the selection map occupies only N = 196 bits per image, corresponding to less than 0.004 bits per pixel (bpp), where the bit rate is computed as the average of $\frac { \mathrm { s e n t ~ b i t s } } { \mathrm { n u m b e r ~ o f ~ p i x e l s } }$ over the evaluation set. We therefore omit the cost of s when computing the overall bit rate.

## B. Rate–Accuracy Trade-Off

In Fig. 5, we evaluate the trade-off between bit rate and classification accuracy of the proposed framework against three baseline families: prior token-oriented semantic communication frameworks, hand-crafted codecs, and LIC models. For a fair comparison, all baselines decode the transmitted representation into an image that is classified by the same pretrained server model, DeiT-III-Large, fixing the upper-bound accuracy under lossless transmission to 86.81 %; the rate– accuracy curves thus differ only in how efficiently each method compresses and transmits the image. End-to-end frameworks, in contrast, jointly train a task-specific classifier with the transceiver, yielding a different accuracy ceiling and no standalone image, and thus are not directly comparable under this controlled protocol.

The rate–accuracy trade-off of the proposed framework is characterized in the token-aligned LIC module by varying the token-selection threshold δ and switching among LIC models trained with different values of λ, where δ controls the number of selected tokens and λ determines the operating point of the LIC model. We sweep δ from 0.5 to 0.98 and λ from 0.0075 to 2, where larger values of either generally yield higher bit rates and improved downstream performance. The remaining components are fixed during inference: $( L _ { s } , L _ { e } ) = ( 7 , 1 2 )$ for layer-selective attention rollout, and a surrogate token trained with $( \delta _ { \mathrm { s u r } } , \lambda _ { \mathrm { s u r } } ) = ( 0 . 3 5 , 0 . 0 3 )$ . The effects of $( L _ { s } , L _ { e } )$ and $\left( \delta _ { \mathrm { s u r } } , \lambda _ { \mathrm { s u r } } \right)$ are further discussed in Section V-F and Section V-G, respectively.

As shown in Fig. 5(a), the proposed framework achieves higher downstream accuracy at comparable bit rates to the token-oriented baselines, namely selected-patch transmission [33] and importance-aware quantization [34]. Since these baselines also exploit task relevance but transmit in the pixel domain, this gain comes mainly from the rate efficiency of token-aligned LIC. Against hand-crafted codecs (JPEG2000, WebP, and BPG), Fig. 5(b) shows a favorable trade-off below 2 bpp: the proposed framework attains

![](images/ce7193e13abce4fee0a8e64588685df7924ff8f4591584d35218418783e56d4f.jpg)  
(a) Token-oriented semantic communication frameworks

![](images/55866bd84790b1b4ee7e0f9bdae2bc14c7638a0ea74fa2d1af9ee446f8711dc6.jpg)  
(b) Hand-crafted codecs

![](images/16d519a551398111548a6149411df30c5b450b3894bbfdf7ffdfa32ddab86791.jpg)  
(c) LIC frameworks  
Fig. 5. Trade-off between bit rate and classification accuracy of the proposed framework against (a) prior token-oriented semantic communication frameworks [33], [34], (b) hand-crafted codecs, and (c) LIC frameworks [26]– [28]. Server accuracy corresponds to the original downstream performance of DeiT-III-Large, i.e., 86.81 %, attainable under lossless transmission at 24 bpp. Proposed denotes the achievable curve formed by connecting the Pareto-optimal operating points of the proposed framework.

![](images/b8632dde47617fd166dfc0da78ea24265632793a06bd0efb9c5faa9d9dc75195.jpg)  
Fig. 6. Classification accuracy over a packet-erasure channel. Original accuracy corresponds to the errorless downstream performance of the proposed framework, i.e., 85.92 %, where $p _ { e } = 0 .$ The reported accuracy is averaged over 20 independent trials.

85.83 % accuracy at 1.24 bpp, only 0.98 percentage points below the server accuracy, whereas WebP and BPG require about 1.57 bpp and 1.82 bpp to match it. A comparable advantage holds against task-agnostic LIC frameworks [26]– [28], where Fig. 5(c) shows higher accuracy at comparable bit rates below 2.5 bpp. Although the proposed selective transmission is not directly compatible with autoregressive context models such as ELIC [28], its implementation with the simpler mean-scale hyperprior still achieves a more favorable rate– accuracy trade-off than ELIC. This result highlights the benefit of task-relevance-guided latent selection over improving reconstruction-oriented entropy modeling alone. Overall, the gains against hand-crafted codecs and prior LIC frameworks confirm the benefit of incorporating task relevance, beyond optimizing reconstruction fidelity alone.

## C. Classification Accuracy over Erasure Channel

To evaluate the robustness of surrogate token substitution over unreliable channels, we measure the classification accuracy of the proposed framework on a packet-erasure channel. Because the LIC model uses arithmetic coding to compress both latent representations and hyperpriors, a single erasure may disrupt the synchronization of the entropy model between the encoder and decoder; we therefore packetize them and independently encode each packet to localize the impact of erasures.

The hyperpriors, which provide the side information for latent decoding, are placed in a single packet assumed to be received without error. The latents are then greedily packed into the remaining packets in descending order of their per-latent self-information $\begin{array} { r } { H _ { \widehat { \mathbf { x } } _ { i } ^ { c } } = - \sum _ { d } \log _ { 2 } p ( \widehat { x } _ { i , d } ^ { c } \mid \widehat { \mathbf { Z } } ) } \end{array}$ . Each packet, constrained to a maximum transmission unit of 1460 bytes, is then independently arithmetic-coded and transmitted over the erasure channel with erasure probability $p _ { e }$

We evaluate the classification accuracy of DeiT-III-Large on images compressed by the LIC model with $\begin{array} { r l } { \lambda } & { { } = } \end{array}$ 0.3 and transmitted under erasure probabilities $\mathit { p _ { e } } \in$ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}. The latents in erased packets are replaced by their predicted means, and the corresponding tokens are processed by one of three server-side inference strategies: direct-reconstruction inference, selectedpatch inference, and surrogate token substitution. To isolate the effect of erasures, selective latent transmission is not applied, fixing the bit rate at 2.12 bpp. For reference, an image-level coding without packetization yields 2.11 bpp, indicating that packetization introduces a negligible overhead. As shown in Fig. 6, surrogate token substitution is consistently the most robust of the three strategies across the erasure probabilities.

![](images/ef23d9ffc78ee84d71e54493c60c53d3b47ba535e56bcc7143f8a3275356179b.jpg)  
Fig. 7. Trade-off between bit rate and classification accuracy of the proposed framework with and without entropy-aware image transmission (EIT), obtained over the sweep of $( \tau , \delta , \lambda )$ . Both are the achievable curve formed by connecting the Pareto-optimal operating points.

## D. Entropy-Aware Image Transmission

Entropy-aware image transmission (EIT) further reduces the communication payload by allowing the client to classify sufficiently confident inputs locally without transmitting them to the server [33]. Specifically, EIT computes the min-entropy of the client-side predictive distribution, which is low for confident predictions, and invokes server-side inference only when the min-entropy is at least a threshold τ.

We sweep τ from 0.1 to 1, together with reduced sets of δ and λ within the ranges of Section V-B. As shown in Fig. 7, EIT shifts the achievable curve toward lower bit rates: the framework with EIT attains 85.89 % accuracy at 0.94 bpp, whereas approximately 1.24 bpp is required without EIT to attain comparable accuracy. Since the additional computation amounts to a single classification head, EIT provides a practical complement to the proposed framework, although it is not claimed as a contribution of this work.

## E. Token-Aligned Learned Image Compression

We first analyze the rate-control behavior of the proposed framework through its two hyperparameters δ and λ. Fig. 8 shows the operating points obtained by varying δ and switching among LIC models trained with different λ, whose connected achievable curve enables adaptive rate control under varying channel or latency conditions. Adjusting only δ at inference provides fine-grained control within a narrow bit-rate range without switching the LIC model, while five LIC models with $\lambda \in \{ 0 . 0 3 , 0 . 1 , 0 . 2 , 0 . 4 , 1 \}$ suffice to cover the range from 0.5 bpp to 2.5 bpp at near-optimal accuracy, keeping the memory cost of rate control low.

![](images/7f94054d7abe615a91e8fda69dcb8dd65f4fa18b5a717cd8c56df1f51a9cea70.jpg)

Fig. 8. Achieved rate–accuracy points of the proposed framework. Achievable accuracy is formed by connecting the Pareto-optimal operating points. Each solid line connects the operating points obtained with the same LIC model parameters while varying only δ.  
![](images/4022011adbb46c8aab97c1f0cc629fe660d6bef8a5d61d5cc8d66861ad6fb8f4.jpg)  
Fig. 9. Trade-off between bit rate and classification accuracy for LIC models without and with mean imputation. Without mean imputation refers to the hyperprior-based LIC model [26] without mean imputation, whereas with mean imputation denotes the model augmented with mean hyperpriors [27], where the missing latent vectors are imputed using the predicted means. Both models are trained with $\lambda = 0 . 1$

To assess the effect of mean imputation, we compare hyperprior-based LIC models without and with mean imputation [26], [27], using the same settings as the main experiments except that the surrogate token is trained in an LIC-agnostic manner for fairness. As shown in Fig. 9, the model with mean hyperpriors attains higher downstream accuracy. This gain does not come from the unsent regions, which are uniformly replaced by a surrogate token before inference, but from improved reconstruction of the selected regions: since LIC latents can influence neighboring patches, the selected patches are affected by adjacent unsent latents, and mean imputation alleviates this effect.

## F. Layer-Selective Attention Rollout

We first identify the most task-relevant layer range $( L _ { s } , L _ { e } )$ motivated by the noisy early-layer attentions observed in Fig. 4. Fixing $L _ { e } ~ = ~ L$ and ablating the LIC module, we measure selected-patch inference accuracy in which DeiT-Tiny selects the top-60, top-80, or top-100 tokens by layer-selective attention rollout and DeiT-III-Large classifies from the selected patches. As shown in Fig. 10, aggregating over the latter half of the layers, particularly with $L _ { s } = 7 .$ yields the best performance in our main operating region.

![](images/3b4d8f057b7a11dbb2e3ee6e3ce4084faf8e6a8033a00f39d165203dde6a8917.jpg)  
Fig. 10. Selected-patch inference accuracy under layer-selective attention rollout computed with different layer ranges $( L _ { s } , L _ { e } )$ . The end layer is fixed as $L _ { e } = L ,$ where $L = 1 2$ for DeiT-Tiny.

![](images/6c7fe58000c6db32dde85ead6085226f55f7dc0b096f80f55008bd474a27ae49.jpg)  
Fig. 11. Selected-patch inference accuracy using different attention-guided importance. Selected token rate is defined as the ratio of selected tokens to the total number of tokens.

We then compare layer-selective attention rollout with $( L _ { s } , L _ { e } ) = ( 7 , 1 2 )$ against other attention-guided importance methods under the same setting: last-layer attention [33], attention rollout [35], and gradient-weighted attention [36]. As shown in Fig. 11, layer-selective attention rollout attains the most favorable rate–accuracy trade-off in our main operating region, even without the additional gradient computation required by gradient-weighted attention.

Moreover, we verify that the effectiveness of layer-selective attention rollout is not specific to the choice of client-side ViT by repeating the comparison with DeiT-Tiny replaced by DeiT-Small and DeiT-Base. Both backbones comprise $L = 1 2$ transformer layers, for which we set $( L _ { s } , L _ { e } ) = ( 5 , 1 2 )$ and (7, 12), respectively, searched by the same procedure used for DeiT-Tiny. As shown in Fig. 12, layer-selective attention rollout consistently outperforms last-layer attention and attention rollout, and performs comparably to gradient-weighted attention on both backbones. Crucially, layer-selective attention rollout attains this accuracy with only a single forward pass, whereas the gradient computation required by gradientweighted attention becomes increasingly impractical for these larger backbones—making layer-selective attention rollout the more practical choice at comparable accuracy.

![](images/973c527486db181d04fd00dfb787742fbf23efdbcca9a0319b796d69692e9ab8.jpg)  
(a) DeiT-Small for client

![](images/233be6c7d2bbf2201327e6b3a0c8f0be29bed18e15aa3a36b06ea1241928b7a0.jpg)  
(b) DeiT-Base for client  
Fig. 12. Selected-patch inference accuracy using different attention-guided importance with the client-side ViT replaced by (a) DeiT-Small and (b) DeiT-Base, both having $L = 1 2$ transformer layers. Layer-selective attention rollout is computed over $( L _ { s } , L _ { e } ) = ( 5 , 1 2 )$ and (7, 12), respectively.

## G. Surrogate Token Substitution

The surrogate token is trained by emulating the proposed pipeline under training hyperparameters $\delta _ { \mathrm { s u r } }$ and $\lambda _ { \mathrm { s u r } }$ , which set the token-selection threshold and the LIC operating point during training, respectively; the detailed settings are given in Section V-A. Because these hyperparameters determine the quality of training-time inputs, they can affect both training stability and final accuracy.

We first determine $\delta _ { \mathrm { s u r } }$ by ablating the LIC module and training a separate surrogate token for each value of $\delta _ { \mathrm { s u r } }$ from 0.25 to 0.98. Fig. 13 evaluates these tokens at test-time thresholds $\delta \in \{ 0 . 6 , 0 . 7 , 0 . 7 9 \}$ in our main operating region, showing that $\delta _ { \mathrm { s u r } } ~ = ~ 0 . 3 5$ consistently attains near-optimal accuracy despite the mismatch between $\delta$ and $\delta _ { \mathrm { s u r } }$

Fixing $\delta _ { \mathrm { s u r } } ~ = ~ 0 . 3 5$ , we determine $\lambda _ { \mathrm { s u r } } ~ = ~ 0 . 0 3$ in the same manner. With these hyperparameters, Fig. 14(a) shows that surrogate token substitution outperforms both adaptationfree baselines. Furthermore, Fig. 14(b) shows that the LICaware surrogate token, trained with $( \delta _ { \mathrm { s u r } } , \lambda _ { \mathrm { s u r } } ) = ( 0 . 3 5 , 0 . 0 3 )$ consistently outperforms the LIC-agnostic counterpart trained with $\delta _ { \mathrm { s u r } } ~ = ~ 0 . 3 5$ alone, indicating that incorporating LIC awareness during training yields additional gains.

![](images/e03c317247ed156254c3b87258dfdd5b5394ddf1e141fb80129efd4b385808fd.jpg)  
Fig. 13. Evaluation accuracy for surrogate tokens trained under different token-selection thresholds $\delta _ { \mathrm { s u r } } .$ . We evaluate under the test-time thresholds $\delta \in \{ 0 . 6 , 0 . 7 , 0 . 7 9 \}$ }. The annotations indicate the corresponding values of $\delta _ { \mathrm { s u r } }$ for each training token rate.

## VI. CONCLUSION

We proposed a modular, token-oriented semantic communication framework that coordinates pretrained client-side ViT, LIC, and server-side ViT models for client–server collaborative inference. To improve the rate–accuracy trade-off, we developed three technical components: (1) token-aligned learned image compression, which uses token-level task relevance to selectively transmit spatially aligned LIC latents; (2) layerselective attention rollout, which efficiently estimates token relevance from a selected range of attention layers; and (3) learnable surrogate token substitution, which compensates for missing visual information at the server. Experimental results demonstrated that the proposed framework achieves a more favorable rate–accuracy trade-off than recent token-oriented semantic communication schemes, hand-crafted codecs, and learned image compression models. Overall, these results indicate that token-aligned compression of the input image, rather than direct compression of token representations, is a promising direction for modular token-level semantic communication in resource-constrained edge AI systems.

## REFERENCES

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Dec. 2017, pp. 5998–6008.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Representations (ICLR), Jun. 2021.

[3] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio spectrogram transformer,” in Proc. Interspeech, Aug. 2021, pp. 571–575.

[4] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lucic, and C. Schmid, “ViViT: A video vision transformer,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 6836–6846.

![](images/acf224655bd73d2a03f3bfea37d5ee87c3f0f7210788fcdedb4846038c386175.jpg)  
(a) Server-side inference strategies

![](images/586e5d34bd7ffc41037c748dc0d2091b27b761422dc5eb3a060fd688353cf83e.jpg)  
(b) LIC-aware vs. LIC-agnostic training

Fig. 14. Trade-off between bit rate and classification accuracy of surrogate token substitution. (a) Comparison of server-side inference strategies, where the surrogate token is trained with $( \delta _ { \mathrm { s u r } } , \lambda _ { \mathrm { s u r } } ) = ( 0 . 3 5 , 0 . 0 3 )$ ). (b) Comparison of surrogate tokens trained with and without the LIC module.

[5] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Proc. Int. Conf. Mach. Learn. (ICML), Jul. 2021, pp. 813–824.

[6] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), Jul. 2021, pp. 8748– 8763.

[7] C. Zhang, H. Zou, S. Lasaulce, W. Saad, M. Kountouris, and M. Bennis, “Goal-oriented communications for the IoT and application to data compression,” IEEE Internet Things Mag., vol. 5, no. 4, pp. 58–63, Dec. 2022.

[8] G. Shi, Y. Xiao, Y. Li, and X. Xie, “From semantic communication to semantic-aware networking: Model, architecture, and open problems,” IEEE Commun. Mag., vol. 59, no. 8, pp. 44–50, Aug. 2021.

[9] Q. Lan, D. Wen, Z. Zhang, Q. Zeng, X. Chen, P. Popovski, and K. Huang, “What is semantic communication? A view on conveying meaning in the era of machine intelligence,” J. Commun. Inf. Netw., vol. 6, no. 4, pp. 336–371, Dec. 2021.

[10] D. Gund¨ uz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener,¨ K. K. Wong, and C.-B. Chae, “Beyond transmitting bits: Context, semantics, and task-oriented communications,” IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 5–41, Jan. 2023.

[11] H. Xie and Z. Qin, “A lite distributed semantic communication system for internet of things,” IEEE J. Sel. Areas Commun., vol. 39, no. 1, pp. 142–153, Jan. 2021.

[12] Y. Kim, J. Shin, Y. Cassuto, and L. R. Varshney, “Distributed boosting classification over noisy communication channels,” IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 141–154, Jan. 2023.

[13] Y. Kim, Y. Cassuto, and L. R. Varshney, “Distributed boosting classifiers over noisy channels,” in Proc. Asilomar Conf. Signals, Syst. Comput.,

Nov. 2020, pp. 1491–1496.

[14] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, “Token communications: A large model-driven framework for cross-modal context-aware semantic communications,” IEEE Wireless Commun. Mag., vol. 32, no. 5, pp. 80–88, Oct. 2025.

[15] B. Liu, L. Qiao, Y. Wang, Z. Gao, Y. Ma, K. Ying, and T. Qin, “Textguided token communication for wireless image transmission,” in Proc IEEE/CIC Int. Conf. Commun. China (ICCC), Aug. 2025, pp. 1–6.

[16] J. Peng, H. Xing, Z. Xiao, L. Xu, and X. Lei, “Large model empowered multi-modal semantic communication with selective tokens for training,” IEEE Signal Process. Lett., vol. 32, pp. 2967–2971, Jul. 2025.

[17] A. Devoto, J. Pomponi, S. Petruzzi, P. Di Lorenzo, and S. Scardapane, “Adaptive semantic token selection for AI-native goal-oriented communications,” in Proc. IEEE Global Commun. Conf. Workshop (GC Wkshps), Dec. 2024, pp. 1–6.

[18] A. Devoto, J. Pomponi, M. Merluzzi, P. Di Lorenzo, and S. Scardapane, “Adaptive semantic token communication for transformer-based edge inference,” IEEE Trans. Mach. Learn. Commun. Netw., vol. 4, pp. 422– 437, Jan. 2026.

[19] H. Wei, W. Ni, W. Wang, W. Xu, D. Niyato, and P. Zhang, “Token communication in the era of large models: An information bottleneckbased approach,” IEEE Wireless Commun. Lett., vol. 15, pp. 186–190, Oct. 2026.

[20] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proc. Int. Conf. Mach. Learn. (ICML), Jun. 2019, pp. 3519–3529.

[21] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Nov. 2017, pp. 6306–6315.

[22] Q. Hu, G. Zhang, Z. Qin, Y. Cai, G. Yu, and G. Y. Li, “Robust semantic communications with masked VQ-VAE enabled codebook,” IEEE Trans. Wireless Commun., vol. 22, no. 12, pp. 8707–8722, Dec. 2023.

[23] G. Zhang, Q. Hu, Z. Qin, Y. Cai, G. Yu, and X. Tao, “A unified multitask semantic communication system for multimodal data,” IEEE Trans. Commun., vol. 72, no. 7, pp. 4101–4116, Jul. 2024.

[24] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen, “An image is worth 32 tokens for reconstruction and generation,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Jun. 2024, pp. 128 940– 128 966.

[25] J. Balle, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image´ compression,” in Proc. Int. Conf. Learn. Representations (ICLR), Feb. 2017.

[26] J. Balle, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational´ image compression with a scale hyperprior,” in Proc. Int. Conf. Learn. Representations (ICLR), Feb. 2018.

[27] D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and´ hierarchical priors for learned image compression,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Dec. 2018, pp. 10 771–10 780.

[28] D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image compression with unevenly grouped spacechannel contextual adaptive coding,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognition (CVPR), Jun. 2022, pp. 5708–5717.

[29] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in Proc. Int. Conf. Mach. Learn. (ICML), Jul. 2021, pp. 10 347–10 357.

[30] H. Touvron, M. Cord, and H. Jegou, “DeiT III: Revenge of the ViT,” in´ Proc. European Conf. Comput. Vis. (ECCV), Oct. 2022, pp. 516–533.

[31] T. Liu, P. Li, Y. Gu, and P. Liu, “Efficient transformer inference for extremely weak edge devices using masked autoencoders,” in Proc. IEEE Int. Conf. Commun. (ICC), May 2023, pp. 1718–1723.

[32] T. Liu, P. Li, Y. Gu, P. Liu, and H. Wang, “Adaptive offloading of transformer inference for weak edge devices with masked autoencoders,” ACM Trans. Sensor Netw., Jan. 2024.

[33] J. Im, N. Kwon, T. Park, J. Woo, J. Lee, and Y. Kim, “Attention-aware semantic communications for collaborative inference,” IEEE Internet Things J., vol. 11, no. 22, pp. 37 008–37 020, Nov. 2024.

[34] J. Park, Y. Oh, Y. Kim, and Y.-S. Jeon, “Vision transformer-based semantic communications with importance-aware quantization,” IEEE Internet Things J., vol. 12, no. 17, pp. 35 662–35 677, Sep. 2025.

[35] S. Abnar and W. Zuidema, “Quantifying attention flow in transformers,” in Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL), Jul. 2020, pp. 4190–4197.

[36] H. Chefer, S. Gur, and L. Wolf, “Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 397–406.

[37] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, “Visual prompt tuning,” in Proc. European Conf. Comput. Vis. (ECCV), Oct. 2022, pp. 709–727.

[38] H. Bao, L. Dong, S. Piao, and F. Wei, “BEiT: BERT pre-training of image transformers,” in Proc. Int. Conf. Learn. Representations (ICLR), Jan. 2022.

[39] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognition (CVPR), Jun. 2022, pp. 16 000–16 009.

[40] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognition (CVPR), Jun. 2021, pp. 782–791.

[41] J. E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Representations (ICLR), Apr. 2022.

[42] B. X. B. Yu, J. Chang, H. Wang, L. Liu, S. Wang, Z. Wang, J. Lin, L. Xie, H. Li, Z. Lin, Q. Tian, and C. W. Chen, “Visual tuning,” ACM Comput. Surv., vol. 56, no. 12, pp. 1–38, Dec. 2024.

[43] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognition (CVPR), Jun. 2009, pp. 248–255.