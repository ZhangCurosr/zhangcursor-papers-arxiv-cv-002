# Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning

Liangjian Wen, Jiangtao Zheng, Jiang Duan<sup>∗</sup>, Qun Dai, Yong Dai, Jianzhuang Liu, Jun Wang, Shu-Dong Huang, Xinwang Liu, Zenglin Xu, Zhao Kang<sup>∗</sup>

Abstract—Multimodal self-supervised learning is challenged by heterogeneity, unequal informativeness, and complex information interactions across multimodal data sources such as images, text, audio, and tabular data. While contrastive methods effectively align shared semantics, they typically rely on strong multi-view redundancy assumptions, overlooking modality-specific and synergistic information, thus leading to suboptimal representations. We propose the Multimodal Information Interactions Bottleneck (MIIB), a principled framework grounded in Partial Information Decomposition that explicitly encodes redundant, unique, and synergistic information within multimodal interactions. MIIB employs a dual-path model architecture for modality-specific encoding and multimodal fusion, optimized using surrogate tractable entropy-based objectives that enforce minimal sufficient representations without requiring explicit mutual information estimation. By leveraging variational bounds and score-based entropy estimation, MIIB efficiently captures complex crossmodal interactions while suppressing superfluous information. Experiments across diverse benchmarks demonstrate consistent improvements over state-of-the-art multimodal self-supervised methods. Code is available at https://github.com/wanglingzaishi/ MIIB.

Index Terms—Self-supervised Learning, Multimodal Interactions, Information Bottleneck

## I. INTRODUCTION

L. Wen, J. Zheng, J. Duan, and Q. Dai are with the School of Computing and Artificial Intelligence, Southwestern University of Finance and Economics, Chengdu, China. J. Wang is with the School of Management Science and Engineering, Southwestern University of Finance and Economics, Chengdu, China. L. Wen is also with the Artificial Intelligence and Digital Finance Key Laboratory of Sichuan Province (e-mail: wlj6816@gmail.com). J. Duan is also with Chengdu Everimaging Science and Technology Co., Ltd.

Y. Dai is with the X-Humanoid Research Institute, Beijing, China.

J. Liu is with the Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China (e-mail: jz.liu@siat.ac.cn).

S. Huang is with the School of Artificial Intelligence, Sichuan University, Chengdu, China.

X. Liu is with School of Computer, National University of Defense Technology, Changsha, 410073.

Z. Xu is with the Artificial Intelligence Innovation and Incubation Institute, Fudan University and also with the Shanghai Academy of AI for Science, Shanghai, China.

Z. Kang is with the University of Electronic Science and Technology of China, Chengdu, China.

<sup>∗</sup>Jiang Duan and Zhao Kang are the corresponding authors (e-mail: duanj t@swufe.edu.cn; zkang@uestc.edu.cn).

This work was supported by the Major Science and Technology Special Project of the Sichuan Provincial Department of Science and Technology (Grant No. 2024ZDZX0002), the National Natural Science Foundation of China (No. U24A20323), Fundamental Research Funds for the Central Universities (JBK202511081), the Sichuan Provincial Innovation Group Project (Grant No. 2024NSFTD0054), Natural Science Foundation of China under Grant No. 62502397, National Natural Science Foundation of China (NSFC) (72471197), and the Sichuan Provincial Philosophy and Social Science Fund (Grant No. SCJJ25ND091).

ULTIMODAL data, which encompasses diverse structures, has become ubiquitous in real-world applications, including healthcare diagnostics [1], [2], autonomous driving [3], [4], multimedia retrieval [5], [6], and humancomputer interaction [7], [8]. These data exhibit key challenges: (1) heterogeneity in distributions (e.g., differing statistical properties between sparse textual data and dense visual data); (2) varying levels of informativeness across modalities (e.g., one modality may provide rich semantic details while another offers only supplementary context); and (3) complex information interactions among modalities (e.g., a harmless cartoon image paired with a simple caption, which together subtly endorses extremist ideology, though neither does so alone). Although self-supervised learning methods, such as SimCLR [9] and MAE [10], have achieved remarkable success in unimodal domains, a critical question remains: how can multimodal self-supervised learning effectively overcome the inherent challenges of multimodal data?

Some works, such as CLIP [5] and ALIGN [11], try to align representations across modalities by treating them as multi-view augmentations, thereby capturing shared semantics. Although these methods are effective for tasks like zero-shot classification and cross-modal retrieval, they often depend on a restrictive assumption known as the multi-view redundancy assumption [12]–[14]. This assumption states that each modality is (approximately) sufficient on its own for predicting downstream tasks and contains the same task-relevant information as the others. However, this assumption originates from traditional multi-view learning and has limitations in realworld multimodal scenarios, where many tasks involve only minimal shared information across modalities. To address this limitation, FactorCL [15] explicitly decomposes representations into shared and unique components, allowing the model to learn the unique information from each modality beyond just the multi-view redundancy. In addition, [16] extends the multi-view redundancy paradigm from contrastive learning to better suit multimodal scenarios. However, multimodal representation learning involves multiple modalities, introducing complex information interactions that can undermine the efficacy of straightforward contrastive approaches, as these approaches mainly focus on learning invariant features through data augmentations across multiple views.

To understand multimodal interactions, we can use Partial Information Decomposition (PID) [17]–[19]. In the PID framework, these interactions can be broken down into three main components: Redundancy (R): This represents information that is shared across multiple modalities. For example, in an imagetext pair describing a scene, both modalities may convey the same core concepts (e.g., object identity), leading to overlapping signals. While redundancy can reinforce robustness in learning, excessive reliance on it in contrastive objectives may cause overfitting to shared artifacts, reducing the model’s ability to generalize. Uniqueness (U): This encompasses information specific to a single modality, which is not available in others. For instance, spatial details in images or syntactic nuances in text are unique contributions. Modality-specific noise, such as visual clutter or linguistic ambiguities, often falls under this category, potentially introducing distortions that misalign cross-modal representations if not properly mitigated. Synergy (S): This denotes complementary information that emerges only through the combination of modalities, providing insights beyond what individual sources offer. An example is inferring sentiment from audio tone combined with textual content, where the joint analysis reveals nuances absent in isolation.

![](images/6764ebb4c4ab1c19bbf6c283e2856da038973e61b63a6606ab089db31173f640.jpg)  
Fig. 1. The process framework for multimodal representation learning organizes modality-specific learning and fusion as a sequential yet interdependent process.

The information-theoretic PID framework aligns with the broader structure of multimodal representation learning. As shown in Fig. 1, multimodal representation learning comprises two primary components: individual modality-specific representation learning and modality fusion. The former focuses on extracting latent features from each modality independently, leveraging modality-tailored encoders to capture intrinsic patterns and reduce dimensionality while preserving modalityunique and redundant information. Due to the heterogeneity in distributions of multimodal data and varying levels of informativeness across modalities, it is challenging to learn effective individual modality-specific representations. The latter integrates these modality-specific representations into a unified multimodal embedding, often through mechanisms to exploit cross-modal synergies. Capturing these complex interactions effectively is essential for multimodal representation learning, enabling models to generalize across varied tasks.

The process of modality-specific representation learning and modality fusion in multimodal systems is inherently sequential and interdependent. Specifically, modality-specific representation learning acts as a foundational preprocessing stage, wherein each individual modality—such as text, images, or audio—is processed independently to extract, isolate, and refine its unique signals. This step mitigates noise, enhances feature relevance, and generates high-quality embeddings that serve as reliable inputs for subsequent fusion. In contrast, the modality fusion phase integrates these preprocessed representations by aligning them across modalities and combining complementary information, thereby enriching the overall representational quality. This integration enables the model to capture holistic multimodal patterns, leading to superior performance on downstream tasks that demand comprehensive cross-modal understanding. However, the effectiveness of the modality fusion process can, in turn, impose constraints on modality-specific representation learning; for instance, suboptimal fusion outcomes—such as misalignment or overemphasis on certain modalities—may necessitate iterative refinements in the individual modality encoders, creating a feedback loop that influences the entire learning pipeline.

In the multimodal representation learning process, if interactions between certain modalities are not adequately captured—due to factors like insufficient training data or architectural limitations—this deficiency propagates through the interdependent nature of modality-specific representation learning and modality fusion. These shortcomings can result in mutual degradation: suboptimal modality-specific features impede effective fusion, while imperfect fusion exacerbates errors in individual representations. This ultimately hinders the learning of effective multimodal representations. To address these challenges, building on this foundation, we propose the Multimodal Information Interactions Bottleneck (MIIB), a novel self-supervised framework that enforces minimal sufficient representations at both modality-specific and fusion levels. MIIB introduces a dual-path architecture with specialized Transformer blocks for integrated fusion, modalityspecific processing, and synergistic extraction, optimized via a simplified surrogate loss that balances minimality, sufficiency, and disentanglement without relying on intractable mutual information estimators. By incorporating variational bounds and score-based entropy estimation [14], MIIB efficiently captures complex interactions while reducing superfluous information, leading to more generalizable multimodal representations.

Our contributions are threefold: (1) We formalize the multimodal information interactions bottleneck from an information-theoretic perspective, highlighting how interdependent modality-specific learning and fusion exacerbate inefficiencies in existing methods. (2) We introduce MIIB, a principled framework that promotes effective multimodal integration through targeted losses for modality-specific, fusion, and synergistic representations. (3) Through extensive experiments on benchmarks, we demonstrate that MIIB outperforms state-of-the-art multimodal self-supervised models.

## II. RELATED WORK

a) Multimodal Learning: Multimodal learning aims to connect and integrate information from diverse data sources— such as text, images, audio, tactile, and tabular data—to enhance the understanding of cross-modal information [7], [20], [21]. Early research focused on training separate encoders and relying on conventional fusion techniques like feature concatenation [22] or modality-specific prediction averaging [23] to aggregate information across modalities. With the development of Transformers and ViTs, the introduction of dynamic crossmodal attention mechanisms [24], [25] has enabled models to more effectively capture and model complex inter-modal interactions. Contemporary methodologies typically adhere to a two-stage framework: initially utilizing specialized encoders to extract features from specific modalities, and subsequently projecting these representations into a unified embedding space [7], [16] for alignment or fusion. This paradigm has been widely adopted in multimodal representation learning [5], [26], cross-modal alignment [27], [28], multimodal clustering [29], [30], and generative modeling [31], [32].

b) Self-Supervised Multimodal Representation Learning: Self-supervised learning aims to generate supervision signals from the inherent structure of data [33], [34]. In the multimodal domain, existing methodologies are widely organized into two representative paradigms. The first is the generative paradigm, which utilizes proxy tasks—such as cross-modal reconstruction [32] or masked prediction [35]—to restore masked or absent inputs. Alternatively, the contrastive learning framework [5], [36] represents another extensively adopted strategy. Methods in this category leverage distinct modalities to formulate semantically complementary views, constructing positive pairs through data augmentation [9]. They then employ contrastive objectives (e.g., InfoNCE) to minimize the distance between positive pairs while maximizing the separation between negative pairs, thereby aligning the semantic features of diverse modalities within the representation space [37]. A parallel line of work approaches common information through an identifiability lens, seeking to provably recover and disentangle the latent factors that are shared across views/tasks from those that are task-unique [38]–[40]. MIIB is orthogonal: instead of disentangling shared/unique factors, it encodes task-relevant common information into one sufficiency–minimal bottleneck, with no identifiability assumptions.

c) Multimodal Contrastive Learning: While conventional contrastive learning frameworks [5], [11] excel in crossmodal alignment, they implicitly rely on the multiview redundancy assumption [41], [42], positing that shared information suffices for downstream tasks. Consequently, these methods predominantly emphasize shared representations, neglecting to explicitly model uniqueness and synergistic interactions. To circumvent this limitation, FactorCL [15] attempts to explicitly disentangle shared and unique information, yet it relies on restrictive augmentation assumptions and fails to model synergistic interactions. Subsequently, CoMM [16] employs multimodal data augmentation and information-theoretic loss functions, aiming to capture redundancy, uniqueness, and synergy within a shared representation space. However, while CoMM theoretically encompasses all three types of interactions, its mechanism for modeling complex synergistic interactions leaves room for further optimization. Consequently, designing effective mechanisms to enhance the capture of synergistic information remains a critical challenge in the field [43]. Beyond contrastive alignment, HRIL explores higher-order tensor modeling for multimodal synergy [44], while MSCR jointly addresses modality utilization balance and synergistic information discovery [45].

Positioning against FactorCL and CoMM. MIIB differs in three respects. (i) Instead of regularizing one fused space, it models multimodal learning as two interdependent processes: modality-specific encoding and fusion, and imposes a minimal-sufficiency bottleneck at both levels (Fig. 1), so the two stages constrain each other. (ii) It gives synergy a dedicated stream from a separate Synergy Transformer and supervises it with the fused cross-view representations: each synergy representation is trained to predict the other view’s fused representation, which can be satisfied only by information jointly carried across modalities — so synergy is learned, not merely encouraged to differ from the unimodal streams. (iii) The objective is reduced to a finite hypersphere entropy-and-alignment regularizer estimated by a scorebased von Mises–Fisher estimator, avoiding the direct mutualinformation estimation that both FactorCL and CoMM rely on.

## III. PRELIMINARY

## A. Minimal Sufficient Representation

Under the information bottleneck principle [46], [47], a representation z is considered sufficient for a target variable $Y$ if it preserves all relevant information in the input X necessary for predicting Y . Among all sufficient representations, a minimal sufficient representation is one that achieves this sufficiency while discarding as much superfluous (i.e., taskirrelevant) information from X as possible. This concept, originally defined for supervised tasks, has been extended to self-supervised multi-view learning.

Under the multi-view assumption, where two views $v _ { 1 }$ and $v _ { 2 }$ share task-relevant semantics, a representation $z _ { 1 }$ is sufficient if it captures the shared information $I ( v _ { 1 } ; v _ { 2 } )$ . It is minimal if it eliminates view-specific superfluous information by minimizing the conditional mutual information $I ( z _ { 1 } ; v _ { 1 } | v _ { 2 } )$ [48].

## IV. CONTRASTIVE MULTIMODAL LEARNING FROM AN INFORMATION PERSPECTIVE

Let $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ denote random variables, each embodying data from a unique modality (such as images, text, audio, or tabular structures), accompanied by a downstream objective represented by Y. The goal is to construct a robust multimodal latent representation $Z _ { \theta } ~ = ~ f _ { \theta } ( X )$ , where $X ~ = ~ ( X _ { 1 } , \ldots , X _ { n } )$ and θ signifies the parameters of the multimodal encoder, such that $Z _ { \theta }$ qualifies as effective by encapsulating task-relevant details from $Y$ while preserving mutual information: $I ( Z ; Y ) = I ( X ; Y )$

In self-supervised learning, the target variable Y is left undefined throughout the representation learning stage, thereby introducing a distinctive obstacle. To mitigate this issue, Dufumier et al. [16] generalize the multiview redundancy paradigm from contrastive learning to encompass multimodal scenarios:

Assumption 1 (Minimal Label-Preserving Multimodal Augmentations). There exists a collection $\tau ^ { * }$ of multimodal operators such that, for every $t \in \tau ^ { * }$ and the transformed instance $X ^ { \prime } = t ( X )$ , the equality $I ( X ; X ^ { \prime } ) = I ( X ; Y )$ is satisfied, thereby maintaining the informational relevance to the label $Y .$

Assumption 1 holds validity in the context of multimodal representation learning. This assumption facilitates an expanded spectrum of data augmentations, unconstrained by the specific ensemble ${ \cal T } _ { c } ^ { \star } = \{ t ( X ) =$ $( t _ { 1 } ( X _ { 1 } ) , t _ { 2 } ( X _ { 2 } ) , \ldots , t _ { n } ( X _ { n } ) ) \}$ . Within this framework, $X ^ { \prime }$ assumes a function comparable to supervisory signals in labeled training paradigms. We introduce $Z _ { \theta } ^ { \prime } = f _ { \theta } ( X ^ { \prime } )$ , with $X ^ { \prime } = t ( X )$ and $t \in \mathcal { T }$ denoting a multimodal transformation. Leveraging the data processing inequality applied to the Markov chain $X  X ^ { \prime }  Z _ { \theta } ^ { \prime }$ , we derive the ensuing bounds on mutual information:

$$
I ( Z _ { \theta } ; Z _ { \theta } ^ { \prime } ) \leq I ( Z _ { \theta } ; X ^ { \prime } ) \leq I ( X ; X ^ { \prime } ) .\tag{1}
$$

This inequality underpins the subsequent lemma:

Lemma 1. In the process of optimizing $f _ { \theta }$ to maximize $I ( Z _ { \theta } ; Z _ { \theta } ^ { \prime } )$ , assuming $f _ { \theta }$ exhibits adequate representational capacity, the optimal parameters $\theta ^ { \star }$ yield $I ( Z _ { \theta ^ { \star } } ; Z _ { \theta ^ { \star } } ^ { \prime } ) =$ $I ( Z _ { \theta ^ { \star } } ; X ^ { \prime } ) = I ( X ; X ^ { \prime } )$

From Lemma 1, we observe that maximizing the mutual information $I ( Z _ { \theta } ; Z _ { \theta } ^ { \prime } )$ ) effectively aligns the representations to capture the shared information between X and $X ^ { \prime } { \mathrm { . } }$ , which, under Assumption 1, preserves the task-relevant information for $Y$

Although contrastive multimodal learning can, in theory, yield effective representations, maximizing the mutual information $I ( Z _ { \theta } ; Z _ { \theta } ^ { \prime } )$ in practice often leads to suboptimal performance, particularly in real-world applications involving diverse datasets. This arises because contrastive multimodal learning primarily emphasizes learning invariant features under data augmentations from multiple views.

Multimodal representation learning involves integrating information from diverse sources such as images, text, and audio. This fundamental difference arises primarily from the intricate interactions among multiple modalities, where each modality may contribute unique yet complementary information that must be harmoniously fused to form a cohesive representation. As a result, multimodal approaches require sophisticated mechanisms to capture the complex interactions among multiple modalities.

## V. METHOD

## A. Multimodal Information Interactions Bottleneck

Multimodal Interactions. Multimodal interactions refer to the process of extracting and integrating information from diverse data sources, such as text, images, audio, or tabular data, to form a cohesive representation for downstream tasks. To understand these multimodal interactions, we can leverage Partial Information Decomposition (PID) [17], [18], a theoretical framework that dissects the mutual information among multiple variables into non-overlapping components. Under the PID framework, these interactions can be decomposed into three primary components: Redundancy (R): This represents information that is shared across multiple modalities. Uniqueness (U): This encompasses information specific to a single modality that is not available in others. Synergy (S): This denotes complementary information that emerges only through the combination of modalities, providing insights beyond what individual sources offer.

For analytical tractability, we focus on the case of $n = 2$ Consider two modalities $X _ { 1 }$ and $X _ { 2 } ,$ , along with a task Y. However, it can be directly extended to n modalities. The mutual information $I ( X _ { 1 } , X _ { 2 } ; Y )$ measures the total taskrelevant information from both modalities combined. Using partial information decomposition (PID), this breaks down as:

$$
I ( X _ { 1 } , X _ { 2 } ; Y ) = R + S + U _ { 1 } + U _ { 2 } ,\tag{2}
$$

where R is redundant information shared by $X _ { 1 }$ and $X _ { 2 } .$ , S is synergistic information arising only from their integration, and $U _ { 1 }$ and $U _ { 2 }$ are unique contributions from $X _ { 1 }$ and $X _ { 2 }$ , respectively. This breakdown aligns with consistency equations from the chain rule of mutual information:

$$
\begin{array} { l } { { I ( X _ { 1 } ; Y ) = R + U _ { 1 } , \quad I ( X _ { 2 } ; Y ) = R + U _ { 2 } , } } \\ { { \phantom { I } } } \\ { { I ( X _ { 1 } ; X _ { 2 } ; Y ) = R - S . } } \end{array}\tag{3}
$$

We propose the Multimodal Information Interactions Bottleneck (MIIB), which preserves modality independence (to maintain the integrity of unique signals) while fostering collaborative integration (to leverage synergies across modalities). The architecture of MIIB is depicted in Fig. 2, comprising two key phases: encoding latent features tailored to each modality and integrating multimodal features using a Transformer. Starting with an input collection of n modalities denoted as $X ~ = ~ ( X _ { 1 } , X _ { 2 } , \ldots , X _ { n } )$ , we derive augmented versions $X ^ { \prime }$ and $X ^ { \prime \prime }$ via data augmentation. These are then fed into encoders specific to each modality, allowing independent extraction of latent representations $e _ { 1 } ^ { \prime } , e _ { 2 } ^ { \prime } , \ldots , e _ { n } ^ { \prime }$ and $e _ { 1 } ^ { \prime \prime } , e _ { 2 } ^ { \prime \prime } , \ldots , e _ { n } ^ { \prime \prime }$ . As illustrated in Fig.2, the resulting features from these modalities undergo parallel processing in three separate paths: (1) Features from all modalities are combined via concatenation and passed through a Transformer block to generate integrated representations $Z ^ { \prime }$ and $Z ^ { \prime \prime } ;$ ; (2) Features from individual modalities are routed separately into Transformer blocks, resulting in modality-specific outputs $Z _ { 1 } ^ { \prime } , Z _ { 2 } ^ { \prime } , \ldots , Z _ { n } ^ { \prime }$ and $Z _ { 1 } ^ { \prime \prime } , Z _ { 2 } ^ { \prime \prime } , \ldots , Z _ { n } ^ { \prime \prime } ;$ (3) Features from all modalities are combined via concatenation and passed through another Transformer block to generate synergistic representations $Z _ { s } ^ { ' }$ and $Z _ { s } ^ { ' \prime }$

In the context of multimodal representation learning, we introduce the information bottleneck principle during the modality-specific representation learning phase to derive minimal sufficient representations for each modality, denoted as $Z _ { 1 } ^ { \prime } , Z _ { 2 } ^ { \prime } , \ldots , Z _ { n } ^ { \prime }$ and $Z _ { 1 } ^ { \prime \prime } , Z _ { 2 } ^ { \prime \prime } , \ldots , Z _ { n } ^ { \prime \prime }$ . Specifically, each modality feature $Z _ { i } ^ { \prime }$ incorporates a shared component R along with unique modality-specific information $U _ { i } ,$ , thereby ensuring representational sufficiency while minimizing superfluous details. This approach yields robust, high-quality embeddings that serve as reliable inputs for subsequent fusion processes. To this end, we derive the following optimization objective using the relaxed Lagrangian multiplier method:

![](images/a0bc1bec5ad92a7e5b5fca862fe84968393977d9483bc776d609ffb9025e3b47.jpg)  
Fig. 2. The overall architecture of MIIB. Given n input modalities $X = ( X _ { 1 } , \ldots , X _ { n } ) $ , we first augment them to generate two views, $X ^ { \prime }$ and $X ^ { \prime \prime }$ These views are encoded independently by modality-specific encoders and latent converters to extract latent features $e _ { 1 } ^ { \prime } \ldots e _ { n } ^ { \prime }$ and $e _ { 1 } ^ { \prime \prime } \ldots e _ { n } ^ { \prime \prime }$ , which provide unified embeddings for subsequent multimodal interaction. These features are then processed by two parallel Transformer streams. The Fusion Transformer processes concatenated multimodal features to produce fused representations $Z ^ { \prime } , Z ^ { \prime \prime }$ while processing individual modality features to yield modality-specific representations $Z _ { 1 } ^ { \prime } \ldots Z _ { n } ^ { \prime }$ and $Z _ { 1 } ^ { \prime \prime } \ldots Z _ { n } ^ { \prime \prime }$ . The Synergy Transformer operates on the concatenated multimodal features to model cross-modal interactions, producing synergistic representations $Z _ { s } ^ { \prime }$ and $Z _ { s } ^ { \prime \prime } .$ Finally, these representations are used to compute the modality-specific loss $( \mathcal { L } _ { \mathrm { M S } } )$ , modality fusion loss $\bar { ( \mathcal { L } _ { \mathrm { M F } } ) }$ , and synergistic loss $( \mathcal { L } _ { \mathrm { { S y n e r g y } } } )$

$$
\mathcal { L } _ { M S } ^ { \prime } \left( \theta \right) = \sum _ { i = 1 } ^ { n } \left[ \lambda _ { \mathrm { M S } } I _ { \theta } \left( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } \mid X ^ { \prime \prime } \right) - I _ { \theta } \left( Z _ { i } ^ { \prime } ; X ^ { \prime \prime } \right) \right] ,\tag{4}
$$

where θ denotes the parameters of the multimodal model, and n represents the number of modalities. Based on ${ \mathrm { A s } } -$ sumption 1, for $Z _ { i } ^ { \prime } ,$ , the input view is its own $X _ { i } ^ { \prime }$ and the supervisory (label-surrogate) view is the other view $X ^ { \prime \prime }$ under the Markov assumption $( X _ { i } ^ { \prime } \to Z _ { i } ^ { \prime }$ , target $X ^ { \prime \prime } )$ . Consequently, maximizing $I _ { \theta } ( Z _ { i } ^ { \prime } ; X ^ { \prime \prime } )$ facilitates the extraction of the relevant component $R$ alongside $U _ { i }$ , thereby ensuring sufficiency. Moreover, superfluous information can be identified through the conditional mutual information $I _ { \theta } \left( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } | X ^ { \prime \prime } \right)$ . Reducing this measure effectively removes such superfluous information. Considering the modality-specific representation $Z _ { i } ^ { \prime \prime }$ and the self-supervised signal $X ^ { \dot { \prime } }$ under the Markov assumption $( X _ { i } ^ { \prime \prime } \to Z _ { i } ^ { \prime \prime }$ , target $X ^ { \prime } )$ , we can derive a similar optimization objective as follows:

$$
\mathcal { L } _ { M S } ^ { \prime \prime } \left( \theta \right) = \sum _ { i = 1 } ^ { n } \left[ \lambda _ { \mathrm { M S } } I _ { \theta } \left( Z _ { i } ^ { \prime \prime } ; X _ { i } ^ { \prime \prime } \mid X ^ { \prime } \right) - I _ { \theta } \left( Z _ { i } ^ { \prime \prime } ; X ^ { \prime } \right) \right] .\tag{5}
$$

During the modality fusion process, we aim for the multimodal representations $Z ^ { \prime }$ and $Z ^ { \prime \prime }$ to capture the task-relevant information $\textstyle R + S + \sum _ { i = 1 } ^ { n } U _ { i }$ . To facilitate multimodal information interaction under Partial Information Decomposition (PID), we introduce three types of information losses to promote this interaction.

First, $Z ^ { \prime }$ and $Z ^ { \prime \prime }$ serve as mutual self-supervision signals. According to Lemma 1, maximizing the mutual information $I ( Z _ { \theta } ; Z _ { \theta } ^ { \prime } )$ enables the learning of task-relevant information $\textstyle R + S + \sum _ { i = 1 } ^ { n } U _ { i }$ for the target $Y$

Second, the individual modality representations $Z _ { 1 } ^ { \prime } , Z _ { 2 } ^ { \prime } , \ldots , Z _ { n } ^ { \prime }$ and $Z _ { 1 } ^ { \prime \prime } , Z _ { 2 } ^ { \prime \prime } , \ldots , Z _ { n } ^ { \prime \prime }$ act as self-supervision signals with respect to $Z ^ { \prime \prime }$ and $Z ^ { \prime }$ , respectively. Maximizing the mutual information $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \left[ I _ { \theta } \big ( \dot { Z } _ { i } ^ { \prime } ; Z ^ { \prime \prime } \big ) ^ { * } + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z ^ { \prime } ) \right] } \end{array}$ captures the shared information $R \ + \ U _ { i }$ between each modality feature and the fused multimodal representation.

Third, the synergistic representations $Z _ { s } ^ { \prime }$ and $Z _ { s } ^ { \prime \prime }$ serve as self-supervision signals with respect to $Z ^ { \prime \prime }$ and $Z ^ { \prime } .$ , respectively. Maximizing the mutual information $I ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } ) +$ $I ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } )$ learns the shared synergistic information $S .$

To enhance the effectiveness of the modality fusion process, we minimize the conditional mutual information $I _ { \theta } ( Z ^ { \prime } ; X ^ { \prime } \mid$ $X ^ { \prime \prime } )$ and $I _ { \theta } ( Z ^ { \prime \prime } ; X ^ { \prime \prime } | X ^ { \prime } )$ , thereby reducing superfluous information and facilitating minimal sufficient representation learning for $Z ^ { \prime \prime }$ and $Z ^ { \prime }$ . In summary, we derive the modality fusion loss as follows:

$$
\begin{array} { l } { { \displaystyle { \mathcal L } _ { M F } \left( \theta \right) = \lambda _ { \mathrm { M F } } [ I _ { \theta } ( Z ^ { \prime } ; X ^ { \prime } \mid X ^ { \prime \prime } ) + I _ { \theta } ( Z ^ { \prime \prime } ; X ^ { \prime \prime } \mid X ^ { \prime } ) ] } } \\ { { \displaystyle \qquad - I _ { \theta } ( Z ^ { \prime } ; Z ^ { \prime \prime } ) - \sum _ { i = 1 } ^ { n } [ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z ^ { \prime } ) ] } } \\ { { \displaystyle \qquad - I _ { \theta } ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } ) - I _ { \theta } ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } ) } } \end{array}\tag{6}
$$

To enhance synergistic interactions in our multimodal fusion framework, we aim to enforce minimal sufficient synergistic representations $Z _ { s } ^ { \prime }$ and $Z _ { s } ^ { \prime \prime }$ . These representations capture only the essential collaborative information shared between modalities, while minimizing redundancy. This approach ensures efficient integration without overwhelming the model with irrelevant details. First, to reduce superfluous information in these representations, we derive an optimization objective based on conditional mutual information. Specifically, we minimize the dependency of $Z _ { s } ^ { \prime }$ on $X ^ { \prime }$ given $X ^ { \prime \prime }$ , and vice versa, which encourages $Z _ { s } ^ { \prime }$ and $Z _ { s } ^ { \prime \prime }$ to focus solely on synergistic aspects:

$$
I _ { \theta } \left( Z _ { s } ^ { ' } ; X ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { s } ^ { ' \prime } ; X ^ { \prime \prime } | X ^ { \prime } \right) .\tag{7}
$$

Next, we introduce a disentangled information loss to further enable the Transformer to extract pure synergistic information. This loss promotes separation by penalizing mutual information between individual modality-specific features $( Z _ { i } ^ { \prime }$ and $Z _ { i } ^ { \prime \prime } )$ and the opposing synergistic representations $( Z _ { s } ^ { \prime \prime }$ and $Z _ { s } ^ { \prime } )$ across n elements

$$
\sum _ { i = 1 } ^ { n } \left[ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z _ { s } ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z _ { s } ^ { \prime } ) \right] .\tag{8}
$$

By combining these components, we formulate the overall optimization loss for the synergistic representations in multimodal fusion $( \mathcal { L } _ { \mathrm { s y n e r g y } } )$

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { s y n e r g y } } \left( \theta \right) = \lambda _ { \mathrm { s s } } [ I _ { \theta } \left( Z _ { s } ^ { ' } ; X ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { s } ^ { ' \prime } ; X ^ { \prime \prime } | X ^ { \prime } \right) ] } } \\ { { \displaystyle ~ + \sum _ { i = 1 } ^ { n } [ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z _ { s } ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z _ { s } ^ { \prime } ) ] . } } \end{array}\tag{9}
$$

Consider the term $- I ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } ) \ : - \ : I ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } )$ that appears in the mutual-fusion loss $\mathcal { L } _ { M F } .$ . Minimizing this term is mathematically equivalent to maximizing the sum of mutual informations $\bar { I } ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } ) + I ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } )$ . Under this objective, the synergistic representation $Z _ { s } ^ { \prime }$ and the modality-fused features $\dot { Z } ^ { \prime \prime }$ of the opposite view mutually serve as self-supervision signals (and symmetrically between $Z _ { s } ^ { \prime \prime }$ and $Z ^ { \prime } )$ . Consequently, each synergy stream is encouraged to align with the full fused representation produced by the other view. Because $Z ^ { \prime \prime }$ aggregates information from both modalities of the opposite view, a high value of $I ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } )$ can be attained only when $Z _ { s } ^ { \prime }$ encodes information that is jointly carried across the two views. This cross-view mutual-information term therefore supplies the precise learning signal that enables the model to capture the shared synergistic information S. Hence, by incorporating the terms $- \dot { I } ( Z _ { s } ^ { \bar { \prime } } ; Z ^ { \prime \prime } ) - I ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } )$ from $\mathcal { L } _ { M F }$ , this optimization objective promotes the learning of minimality, sufficiency, and disentanglement.

In summary, the MIIB framework is constructed by integrating three core optimization objectives to address the challenges in multimodal representation learning. The modalityspecific loss $\mathcal { L } _ { M S } = \mathcal { L } _ { M S } ^ { \prime } + \mathcal { L } _ { M S } ^ { \prime \prime }$ ensures minimal sufficient representations for each individual modality by minimizing superfluous information while maximizing task-relevant components. The modality fusion loss $\mathcal { L } _ { M F }$ promotes effective integration across modalities through mutual self-supervision, capturing shared, unique, and synergistic information while reducing superfluous information. Finally, the synergy loss $\mathcal { L } _ { \mathrm { s y n e r g y } }$ enforces disentangled and minimal sufficient synergistic representations, enhancing collaborative interactions without introducing irrelevant details. Together, these losses form the comprehensive MIIB objective, optimized as follows:

$$
\mathcal { L } _ { M I I B } ( \theta ) = \mathcal { L } _ { M S } ( \theta ) + \mathcal { L } _ { M F } ( \theta ) + \mathcal { L } _ { \mathrm { s y n e r g y } } ( \theta ) .\tag{10}
$$

## B. Simplifying Optimization Objectives

MIIB cannot be directly applied to multimodal representation learning because mutual information is generally intractable to compute. This difficulty is particularly pronounced when the encoders are deterministic: in this case, the relevant mutual information terms are not finite, a well-known obstacle for applying the information bottleneck principle to deterministic neural network classifiers [49]. Consequently, the mutual information objectives in $\operatorname { E q . }$ (10) become formally intractable, as they presuppose stochastic encoders capable of producing finite mutual information values. The variational formulation of the information bottleneck [37] typically requires the introduction of additional stochastic networks and has been observed to underperform in representation learning tasks. To address this limitation and enable MIIB for deterministic encoders, we exploit the relationship between mutual information and entropy to reformulate the objective as an entropy/alignment regularizer. This reformulation does not correspond to a rigorous stochastic information-compression bottleneck. Instead, it aligns more closely with the Deterministic Information Bottleneck (DIB) framework [50], in which the entropy of the representation serves as a surrogate for the compression term.

Consider that mutual information can be decomposed into entropy and conditional entropy as follows: $\begin{array} { r l } { I ( A ; B ) } & { { } = } \end{array}$ $H ( A ) - H ( A \mid B )$ . We can derive and simplify the mutual information term in MIIB. On the one hand, this approach converts the estimation of mutual information into the estimation of entropy and conditional entropy. On the other hand, it reduces computational overhead by merging similar terms.

a) Simplifying the modality-specific loss : The modality-specific loss is defined as

$$
\begin{array} { r l r } {  { \mathcal L _ { M S } ( \theta ) = \sum _ { i = 1 } ^ { n } \biggl \{ \lambda _ { \mathrm { M S } } [ I _ { \theta } ( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } | X ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; X _ { i } ^ { \prime \prime } \mid X ^ { \prime } ) ] } \\ & { } & { \mathrm { ~ - ~ } I _ { \theta } ( Z _ { i } ^ { \prime } ; X ^ { \prime \prime } ) - I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; X ^ { \prime } ) \biggr \} } \qquad ( 1 1 } \end{array}\tag{1}
$$

Let’s break it down step by step, focusing on the key terms inside the sum for each modality i. First, recall that conditional

mutual information $I ( A ; B | C ) \ = \ H ( A | C ) \ : - \ : H ( A | B , C )$ Applying this to the two terms:

$$
\begin{array} { r l } & { I _ { \theta } \left( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { i } ^ { \prime \prime } ; X _ { i } ^ { \prime \prime } \mid X ^ { \prime } \right) } \\ & { = H ( Z _ { i } ^ { \prime } | X ^ { \prime \prime } ) - H ( Z _ { i } ^ { \prime } | X _ { i } ^ { \prime } , X ^ { \prime \prime } ) } \\ & { + H ( Z _ { i } ^ { \prime \prime } \mid X ^ { \prime } ) - H ( Z _ { i } ^ { \prime \prime } \mid X _ { i } ^ { \prime \prime } , X ^ { \prime } ) . } \end{array}\tag{12}
$$

Here, $Z _ { i } ^ { \prime }$ is the representation from view $X _ { i } ^ { \prime }$ under the Markov assumption $( X _ { i } ^ { \prime } \  \ Z _ { i } ^ { \prime }$ , target $X ^ { \prime \prime } )$ , and $Z _ { i } ^ { \prime \prime }$ from view $X _ { i } ^ { \prime \prime }$ under the Markov assumption $( X _ { i } ^ { \prime \prime }  Z _ { i } ^ { \prime \prime }$ , target $X ^ { \prime } )$

In the deterministic encoder limit, the conditional distributions reduce to Dirac deltas, $p ( Z _ { i } ^ { \prime } \mid X _ { i } ^ { \prime } , X ^ { \prime \prime } ) = \delta ( Z _ { i } ^ { \prime } - f ( X _ { i } ^ { \prime } ) )$ and $p ( Z _ { i } ^ { \prime \prime } \mid X _ { i } ^ { \prime \prime } , X ^ { \prime } ) = \delta ( \bar { Z } _ { i } ^ { \prime \prime } - \bar { f } ( \bar { X } _ { i } ^ { \prime \prime } ) )$ . Consequently, the conditional differential entropies $H ( Z _ { i } ^ { \prime } \mid X _ { i } ^ { \prime } , X ^ { \prime \prime } )$ and $H ( Z _ { i } ^ { \prime \prime } \mid$ $X _ { i } ^ { \prime \prime } , X ^ { \prime } )$ diverge $\mathrm { ~ t o ~ } - \infty .$ , causing the corresponding mutual information terms $I _ { \theta } ( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } | X ^ { \prime \prime } )$ and $I _ { \theta } \bar { ( Z _ { i } ^ { \prime \prime } ; X _ { i } ^ { \prime \prime } \ | \ X ^ { \prime } ) }$ to diverge $_ { 1 0 } \ + \infty$ . Since the encoder is deterministic, the representation of each input view is fixed with no additional stochasticity. Following the approach of the Deterministic Information Bottleneck (DIB) [50], these conditional entropy terms are therefore dropped from the objective, as they contribute no uncertainty once the input view is observed. In place of the original mutual information terms, we substitute the marginal conditional entropies $H ( Z _ { i } ^ { \prime } \mid X ^ { \prime \prime } )$ and $H ( Z _ { i } ^ { \prime \prime } \mid X ^ { \prime } )$ This substitution can enforce information minimality (i.e., compression), yielding a tractable entropy-based objective. So, simplifying, we get:

$$
\begin{array} { r l } & { I _ { \theta } ( Z _ { i } ^ { \prime } ; X _ { i } ^ { \prime } | X ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; X _ { i } ^ { \prime \prime } \mid X ^ { \prime } ) } \\ & {  H ( Z _ { i } ^ { \prime } | X ^ { \prime \prime } ) + H ( Z _ { i } ^ { \prime \prime } \mid X ^ { \prime } ) , } \end{array}\tag{13}
$$

From a practical perspective, while the conditional distributions are nominally Dirac deltas (yielding infinite conditional entropy in the strict continuous case), real neural network implementations operate under finite floating-point precision, implicit stochasticity from Batch Normalization, weight decay, and quantization. These factors collectively act as an effective regularization scale $\epsilon > 0$ , rendering the conditional entropy a large but finite negative constant that diverges only logarithmically with ϵ [51]. This practical regularization ensures that the resulting entropy-based objectives remain well-defined and tractable.

Next, let’s handle the subtracted terms. Mutual information $I ( A ; B ) = H ( A ) - H ( A | B )$ , so:

$$
\begin{array} { r l } & { - I _ { \theta } \left( Z _ { i } ^ { \prime } ; X ^ { \prime \prime } \right) - I _ { \theta } \left( Z _ { i } ^ { \prime \prime } ; X ^ { \prime } \right) = } \\ & { - H ( Z _ { i } ^ { \prime } ) + H ( Z _ { i } ^ { \prime } | X ^ { \prime \prime } ) - H ( Z _ { i } ^ { \prime \prime } ) + H ( Z _ { i } ^ { \prime \prime } \mid X ^ { \prime } ) } \end{array}\tag{14}
$$

Now, putting it all together for the expression inside the loss

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { M S } ( \theta ) = \sum _ { i = 1 } ^ { n } \Bigl \{ - H ( Z _ { i } ^ { \prime } ) - H ( Z _ { i } ^ { \prime \prime } ) } } \\ { { \displaystyle ~ + ~ ( \lambda _ { \mathrm { M S } } + 1 ) [ H ( Z _ { i } ^ { \prime } | X ^ { \prime \prime } ) + H ( Z _ { i } ^ { \prime \prime } | X ^ { \prime } ) ] \Bigr \} . } } \end{array}\tag{15}
$$

b) Simplifying the modality fusion loss: The modality fusion loss function is defined as:

$$
\begin{array} { r l r } {  { \mathcal L _ { M F } ( \theta ) = \lambda _ { \mathrm { M F } } [ I _ { \theta } ( Z ^ { \prime } ; X ^ { \prime } \mid X ^ { \prime \prime } ) + I _ { \theta } ( Z ^ { \prime \prime } ; X ^ { \prime \prime } \mid X ^ { \prime } ) ] } } \\ & { } & { \quad - I _ { \theta } ( Z ^ { \prime } ; Z ^ { \prime \prime } ) - \sum _ { i = 1 } ^ { n } [ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z ^ { \prime } ) ] } \\ & { } & { \quad - I _ { \theta } ( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } ) - I _ { \theta } ( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } ) . } \end{array}\tag{6}
$$

Let’s break it down step by step, following a similar approach to the previous derivation, by expanding terms using entropy definitions and assuming deterministic encoders. First, recall that the conditional mutual information $I ( A ; B | C ) =$ $H ( A | C ) { - } H ( A | B , C )$ . Applying this to the two leading terms:

$$
\begin{array} { r l } & { I _ { \theta } \left( Z ^ { \prime } ; X ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z ^ { \prime \prime } ; X ^ { \prime \prime } \mid X ^ { \prime } \right) } \\ & { = H ( Z ^ { \prime } | X ^ { \prime \prime } ) - H ( Z ^ { \prime } | X ^ { \prime } , X ^ { \prime \prime } ) } \\ & { + H ( Z ^ { \prime \prime } \mid X ^ { \prime } ) - H ( Z ^ { \prime \prime } \mid X ^ { \prime \prime } , X ^ { \prime } ) , } \end{array}\tag{17}
$$

Since the encoders and fusion are deterministic, the conditional distributions of $Z ^ { \prime }$ given $( X ^ { \prime } , X ^ { \prime \prime } )$ and $Z ^ { \prime \prime }$ given $( X ^ { \prime \prime } , X ^ { \prime } )$ are degenerate. As in the modality-specific derivation, we drop the corresponding conditional differential entropy terms when constructing the surrogate objective, yielding:

$$
\begin{array} { r l } & { I _ { \theta } ( Z ^ { \prime } ; X ^ { \prime } | X ^ { \prime \prime } ) + I _ { \theta } ( Z ^ { \prime \prime } ; X ^ { \prime \prime } \mid X ^ { \prime } ) } \\ & {  H ( Z ^ { \prime } | X ^ { \prime \prime } ) + H ( Z ^ { \prime \prime } \mid X ^ { \prime } ) . } \end{array}\tag{18}
$$

Now, let’s handle the subtracted terms one by one. Mutual information $I ( A ; B ) = H ( A ) - H ( A | B )$ , so we expand each using this form to highlight the conditional entropies: First, $- I _ { \theta } \bar { ( } Z ^ { \prime } ; Z ^ { \prime \prime } ) = - H ( \bar { Z ^ { \prime } } ) \bar { + } H ( Z ^ { \prime } | Z ^ { \prime \prime } )$ (alternatively, it could be expressed as $- H ( Z ^ { \prime \prime } ) + H ( Z ^ { \prime \prime } | Z ^ { \prime } )$ , but we’ll use this version for consistency with cross terms).

Next, for the summed terms:

$$
\begin{array} { l } { { \displaystyle - \sum _ { i = 1 } ^ { n } \left[ I _ { \theta } \left( Z _ { i } ^ { \prime } ; Z ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { i } ^ { \prime \prime } ; Z ^ { \prime } \right) \right] } } \\ { { \displaystyle = \sum _ { i = 1 } ^ { n } [ - H ( Z ^ { \prime \prime } ) + H ( Z ^ { \prime \prime } | Z _ { i } ^ { \prime } ) - H ( Z ^ { \prime } ) + H ( Z ^ { \prime } \mid Z _ { i } ^ { \prime \prime } ) ] . } } \end{array}\tag{19}
$$

Similarly, for the final terms:

$$
\begin{array} { r l } & { - I _ { \theta } \left( Z _ { s } ^ { \prime } ; Z ^ { \prime \prime } \right) - I _ { \theta } \left( Z _ { s } ^ { \prime \prime } ; Z ^ { \prime } \right) = } \\ & { - H ( Z _ { s } ^ { \prime } ) + H ( Z _ { s } ^ { \prime } | Z ^ { \prime \prime } ) - H ( Z _ { s } ^ { \prime \prime } ) + H ( Z _ { s } ^ { \prime \prime } \mid Z ^ { \prime } ) . } \end{array}\tag{20}
$$

Hence, putting it all together, we obtain

$$
\begin{array} { r l } {  { \mathcal { L } _ { M F } ( \theta ) = \lambda _ { \mathrm { M F } } [ H ( Z ^ { \prime } | X ^ { \prime \prime } ) + H ( Z ^ { \prime \prime } \mid X ^ { \prime } ) ] } } \\ & { - \ H ( Z ^ { \prime } ) + H ( Z ^ { \prime } | Z ^ { \prime \prime } ) } \\ & { + \sum _ { i = 1 } ^ { n } [ - H ( Z ^ { \prime \prime } ) + H ( Z ^ { \prime \prime } | Z _ { i } ^ { \prime } ) - H ( Z ^ { \prime } ) + H ( Z ^ { \prime } \mid Z _ { i } ^ { \prime \prime } ) ] } \\ & { - \ H ( Z _ { s } ^ { \prime } ) - H ( Z _ { s } ^ { \prime \prime } ) + H ( Z _ { s } ^ { \prime } | Z ^ { \prime \prime } ) + H ( Z _ { s } ^ { \prime \prime } \mid Z ^ { \prime } ) . \qquad ( \mathcal { C } } \end{array}\tag{21}
$$

c) Simplifying the synergistic representation loss: The synergistic representation loss in multimodal fusion is defined as:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { s y n e r g y } } \left( \theta \right) = \lambda _ { \mathrm { s s } } [ I _ { \theta } \left( Z _ { s } ^ { ' } ; X ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { s } ^ { ' \prime } ; X ^ { \prime \prime } | X ^ { \prime } \right) ] } } \\ { { \displaystyle ~ + \sum _ { i = 1 } ^ { n } [ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z _ { s } ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z _ { s } ^ { \prime } ) ] . } } \end{array}\tag{22}
$$

This loss encourages conditional minimality of the synergistic representations and penalizes mutual information between individual modality representations and the opposite view’s synergistic representation. Here, $X ^ { \prime } ~ = ~ \{ X _ { 1 } ^ { \prime } , \ldots , X _ { n } ^ { \prime } \}$ and $\hat { X } ^ { \prime \prime } = \{ X _ { 1 } ^ { \prime \prime } , \ldots , X _ { n } ^ { \prime \prime } \}$ denote the multimodal inputs from two views, with n modalities; $Z _ { i } ^ { \prime }$ and $Z _ { i } ^ { \prime \prime }$ are the representations for modality i from each view; and $Z _ { s } ^ { \prime }$ and $Z _ { s } ^ { \prime \prime }$ are the fused synergistic representations from each view. Let’s derive a simplified form step by step, assuming deterministic encoders (no added randomness), similar to the previous multi-view self-supervised loss derivation. First, expand the conditional mutual information terms using the definition $I ( A ; B | C ) =$ $H ( A | C ) - H ( A | B , C )$

$$
\begin{array} { r l } & { I _ { \theta } \left( Z _ { s } ^ { ' } ; X ^ { \prime } | X ^ { \prime \prime } \right) + I _ { \theta } \left( Z _ { s } ^ { ' \prime } ; X ^ { \prime \prime } | X ^ { \prime } \right) } \\ & { = H ( Z _ { s } ^ { \prime } | X ^ { \prime \prime } ) - H ( Z _ { s } ^ { \prime } | X ^ { \prime } , X ^ { \prime \prime } ) } \\ & { + H ( Z _ { s } ^ { \prime \prime } | X ^ { \prime } ) - H ( Z _ { s } ^ { \prime \prime } | X ^ { \prime \prime } , X ^ { \prime } ) . } \end{array}\tag{23}
$$

Since the encoders and fusion are deterministic, the conditional distributions of $Z _ { s } ^ { \prime }$ given $( X ^ { \prime } , X ^ { \prime \prime } )$ and $Z _ { s } ^ { \prime \prime }$ given $( X ^ { \prime \prime } , X ^ { \prime } )$ are degenerate. As in the modality-specific derivation, we drop the corresponding conditional differential entropy terms when constructing the surrogate objective, yielding:

$$
\begin{array} { r l } & { I _ { \theta } ( Z _ { s } ^ { ' } ; X ^ { \prime } | X ^ { \prime \prime } ) + I _ { \theta } ( Z _ { s } ^ { ' \prime } ; X ^ { \prime \prime } | X ^ { \prime } ) } \\ & {  H ( Z _ { s } ^ { \prime } | X ^ { \prime \prime } ) + H ( Z _ { s } ^ { \prime \prime } | X ^ { \prime } ) . } \end{array}\tag{24}
$$

To enforce statistical independence in the disentangled representations, we require an effective estimator for the mutual information terms in the loss $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \left[ I _ { \theta } ( Z _ { i } ^ { \prime } ; Z _ { s } ^ { \prime \prime } ) + I _ { \theta } ( Z _ { i } ^ { \prime \prime } ; Z _ { s } ^ { \prime } ) \right] } \end{array}$ Minimizing these mutual information quantities is equivalent to promoting orthogonality between the paired latent vectors in the representation space, which helps prevent entanglement while avoiding collapse modes where private representations align excessively with shared ones. Drawing from geometric constraints, we propose a cosine-similarity-based hinge loss that penalizes cosine similarities exceeding a threshold $\gamma ,$ thereby encouraging angular separation between unimodal and synergistic representations. This formulation offers computational efficiency over direct mutual information estimators and ensures robust disentanglement. Specifically, the cosine disentanglement loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i s e n t a n g l e } } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \operatorname* { m a x } \left( 0 , \frac { \langle \mathbf { z } _ { i } ^ { \prime } , \mathbf { z } _ { s } ^ { \prime \prime } \rangle } { \| \mathbf { z } _ { i } ^ { \prime } \| _ { 2 } \cdot \| \mathbf { z } _ { s } ^ { \prime \prime } \| _ { 2 } } - \gamma \right) \right] } \\ { + \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \operatorname* { m a x } \left( 0 , \frac { \langle \mathbf { z } _ { i } ^ { \prime \prime } , \mathbf { z } _ { s } ^ { \prime } \rangle } { \| \mathbf { z } _ { i } ^ { \prime \prime } \| _ { 2 } \cdot \| \mathbf { z } _ { s } ^ { \prime } \| _ { 2 } } - \gamma \right) \right] , } \end{array}\tag{25}
$$

where $\gamma$ controls the margin of the cosine-similarity constraint between unimodal and synergy representations. This pairwise

Algorithm 1 MIIB Training Algorithm   
Require: Multimodal data augmentation $\overline { { \mathcal { T } ^ { * } } }$ , independent   
unimodal encoders $( f _ { i } ) _ { i \in [ 1 . . n ] } ,$ where n denotes the num  
ber of modalities, fusion transformer $g _ { f } ,$ synergy trans  
former $g _ { s } .$   
1: for sampling minibatch $X = ( X _ { 1 } , \ldots , X _ { n } )$ do   
2: Draw $t ^ { \prime } , \bar { t ^ { \prime \prime } } \sim \tau ^ { * }$   
3: $X ^ { \prime } , X ^ { \prime \prime } \gets t ^ { \prime } ( X ) , t ^ { \prime \prime } ( X )$   
4: for $i \in [ 1 , \ldots , n ]$ do   
5: $e _ { i } ^ { \prime } \gets f _ { i } ( X _ { i } ^ { \prime } )$   
6: $e _ { i } ^ { \prime \prime } \gets f _ { i } ( X _ { i } ^ { \prime \prime } )$   
7: end for   
8: $Z ^ { \prime }  g _ { f } ( e _ { 1 } ^ { \prime } , \ldots , e _ { n } ^ { \prime } )$   
9: $Z ^ { \prime \prime }  \bar { g } _ { f } ( e _ { 1 } ^ { \prime \prime } , \ldots , e _ { n } ^ { \prime \prime } )$   
10: for $i \in [ 1 , \ldots , n ]$ do   
11: $Z _ { i } ^ { \prime } \gets g _ { f } ( e _ { i } ^ { \prime } )$   
12: $Z _ { i } ^ { \prime \prime } \gets \dot { g } _ { f } ( e _ { i } ^ { \prime \prime } )$   
13: end for   
14: $Z _ { s } ^ { \prime }  g _ { s } ( e _ { 1 } ^ { \prime } , . . . , e _ { n } ^ { \prime } )$   
15: $Z _ { s } ^ { \prime \prime } \gets g _ { s } ( e _ { 1 } ^ { \prime \prime } , \dots , e _ { n } ^ { \prime \prime } )$   
16: Update parameters $( f _ { i } ) _ { i \in [ 1 . . n ] } , g _ { f } , g _ { s }$ to minimize   
$\hat { \mathcal { L } } _ { M I I B }$   
17: end for   
18: Return $( f _ { i } ) _ { i \in [ 1 \ldots n ] } , g _ { f } , g _ { s }$

approach mirrors the independence objective. It is also differentiable almost everywhere and scalable for optimization in neural networks. Combining all parts, the full loss becomes:

$$
\begin{array} { r l } { \mathrm { ~ } } & { { } \mathcal { L } _ { \mathrm { s y n e r g y } } \left( \theta \right) = \lambda _ { \mathrm { s s } } \left[ H ( Z _ { s } ^ { \prime } | X ^ { \prime \prime } ) + H ( Z _ { s } ^ { \prime \prime } | X ^ { \prime } ) \right] } \\ { ~ } & { { } + \mathcal { L } _ { \mathrm { d i s e n t a n g l e } } . } \end{array}\tag{26}
$$

This form emphasizes minimizing entropy in individual representations while conditioning on cross-view synergistic fusions for the synergistic representations themselves.

d) Final Simplified Optimization Objectives: We derive the variational upper bound for the conditional entropy in Appendix A. The minimization of the conditional entropy is transformed into alignment between representations. For $\hat { \mathcal { L } } _ { M I I B } ( \theta )$ , we obtain the final loss by merging the identical alignment and entropy terms.

$$
\begin{array} { r l r } {  { \hat { \mathcal { L } } _ { M I I B } ( \theta ) = \sum _ { i = 1 } ^ { n } [ - \lambda _ { \mathrm { s i n g l e } } [ H ( Z _ { i } ^ { \prime } ) + H ( Z _ { i } ^ { \prime \prime } ) ] ] } } \\ & { - \lambda _ { \mathrm { s i n s i o n } } [ H ( Z ^ { \prime } ) + H ( Z ^ { \prime \prime } ) ] - \lambda _ { \mathrm { s y n e r g y } } [ H ( Z _ { s } ^ { \prime } ) + H ( Z _ { s } ^ { \prime \prime } ) ] } \\ & { - \sum _ { i = 1 } ^ { n } [ \mathbb { E } _ { p ( Z _ { i } ^ { \prime } , Z ^ { \prime \prime } ) } [ Z _ { i } ^ { ' \top } Z ^ { \prime \prime } ] + \mathbb { E } _ { p ( Z _ { i } ^ { \prime \prime } , Z ^ { \prime } ) } [ Z _ { i } ^ { ' \top } Z ^ { \prime } ] ] } \\ & { - \mathbb { E } _ { p ( Z ^ { \prime } , Z ^ { \prime \prime } ) } [ Z ^ { ' \top } Z ^ { \prime \prime } ] - \mathbb { E } _ { p ( Z _ { s } ^ { \prime } , Z ^ { \prime \prime } ) } [ Z _ { s } ^ { ' \top } Z ^ { \prime \prime } ] } \\ & { - \mathbb { E } _ { p ( Z ^ { \prime \prime } , Z ^ { \prime } ) } [ Z _ { s } ^ { ' \top } Z ^ { \prime } ] + \mathcal { L } _ { \mathrm { d i s e n u n g l e } } . } & { \mathrm { ( f ~ } } \end{array}\tag{27}
$$

We employ the Score-Based Entropy Estimation with the von Mises-Fisher Kernel from [14] to estimate the entropy terms in $\hat { \mathcal { L } } _ { M I I B } ( \theta )$ , as described in detail in Appendix A. For the detailed MIIB Training Algorithm, please refer to Algorithm 1.

Relation to manifold-uniformity regularization. Our entropy terms are related to alignment–uniformity [52]: maximizing the marginal hypersphere entropy $H ( Z )$ is the uniformity objective (up to the bounded-support constant), and minimizing the conditional entropy $H ( Z \mid X ^ { \prime \prime } )$ is the alignment objective. Thus MIIB’s ”bottleneck” is, by construction, a structured multi-stream alignment–uniformity regularizer that is principled, bounded, and consistent with the multi-view IB of [48] and our prior single-stream work [14] rather than a literal channel-capacity constraint.

## VI. EXPERIMENTS

We designed different sets of experiments to evaluate the effectiveness of MIIB, encompassing both synthetic benchmarks and multiple large-scale real-world datasets. First, to evaluate the model’s capacity to capture three core multimodal interactions (i.e., uniqueness, redundancy, and synergy), we constructed a controlled environment based on the Trifeature dataset [53]. We followed the design in CoMM [16] to generate synthetic data and assessed performance through specifically designed tasks. Second, we conducted experiments on several widely used multimodal benchmark datasets spanning real-world scenarios in diverse domains (e.g., healthcare, robotics, etc.), aiming to evaluate the model’s generalizability under different modality combinations and demonstrate its capability to learn effective multimodal representations for complex tasks. Detailed experimental settings are provided in Appendix B. Unless otherwise specified, we adopted the linear probing protocol for evaluation. Specifically, we freeze the pre-trained feature extractors and train a corresponding linear classifier or regressor on top of the learned representations, depending on the specific downstream task.

## A. Controlled Experiments on Synthetic Trifeature

Following the experimental protocol of CoMM [16], we construct a synthetic bimodal dataset based on Trifeature to establish a controlled environment, evaluating MIIB’s capacity to capture uniqueness, redundancy, and synergy through two independent experimental settings.

TABLE I  
LINEAR PROBING ACCURACY (IN %) OF REDUNDANCY (SHAPE), UNIQUENESS (TEXTURE) AND SYNERGY (COLOR AND TEXTURE) ON TRIFEATURE DATASET. <sup>†</sup> DENOTES RESULTS ARE FROM [16].
<table><tr><td>Model</td><td>Redundancy↑</td><td>Uniqueness↑</td><td>Synergy↑</td></tr><tr><td>Cross† [5]</td><td>100.0</td><td>11.6</td><td>50.0</td></tr><tr><td>Cross+Self† [54]</td><td>99.7</td><td>86.9</td><td>50.0</td></tr><tr><td>FactorCL† [15]</td><td>99.8</td><td>62.5</td><td>46.5</td></tr><tr><td>CoMM [16]</td><td> $9 9 . 9 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $8 6 . 8 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $7 1 . 4 { \scriptstyle \pm 3 . 4 7 }$ </td></tr><tr><td>MIIB (ours)</td><td> $9 9 . 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> ${ \bf 8 9 . 8 _ { \pm 3 . 2 3 } }$ </td><td> ${ \bf 7 9 . 1 { \scriptstyle \pm 2 . 9 7 } }$ </td></tr></table>

For the assessment of uniqueness and redundancy, we define shape as the shared (redundant) feature and texture as the modality-specific (unique) feature. We generate the dataset by selecting image pairs that share an identical shape but possess independent textures. We then design two linear probing subtasks: (1) identifying the shared shape class (redundancy task); and (2) predicting the modality-specific unique texture class (uniqueness task). The chance-level accuracy for both tasks is 10%.

For the evaluation of synergy, we artificially construct a strong cross-modal correlation within the training set by defining a mapping M between the texture of Modality $1 \left( X _ { 1 } \right)$ and the color of Modality $2 \ ( X _ { 2 } )$ (e.g., stripes correspond to red). The model is trained exclusively on image pairs that satisfy this mapping, with the downstream task defined as determining whether a given image pair complies with the mapping rule, formulated as $Y = 1 \iff ( \operatorname { t e x t u r e } ( X _ { 1 } ) , \operatorname { c o l o r } ( X _ { 2 } ) ) \in { \mathcal { M } }$ The chance-level accuracy for this task is 50%.

The quantitative results are summarized in Table I. The “Cross” model with InfoNCE-based cross-modality constraints [5] achieves the best performance in capturing redundant information but struggles with uniqueness and synergy. FactorCL [15] and self-supervised constraints on each encoder (“Cross+Self”) [54] enhance uniqueness modeling but remain limited in synergy. CoMM [16] serves as a comprehensive baseline by addressing all three task-related interactions. However, the synergy accuracy of 71.4% suggests that there remains substantial room for more effectively capturing synergistic information. In contrast, MIIB demonstrates superior performance in encoding both uniqueness and synergy, yielding improvements of 3.0 and 7.7 percentage points over CoMM, respectively. This improvement can be attributed to the proposed synergy constraint, which further enhances the extraction of collaborative multimodal information, thereby better capturing complex cross-modal interactions.

## B. Experiments on MultiBench

Following [15], [16], we evaluate on a subset of real-world datasets from MultiBench [55] spanning diverse domains: (1) the MIMIC dataset [56] for healthcare, including static tabular data and ICU time-series; (2) MOSI [57], UR-FUNNY [58], and MUSTARD [59] for affective computing and multimedia analysis, including visual, audio, and textual modalities; and (3) the Vision & Touch dataset [60] for robotics manipulation, including image, force, and proprioception data. Further details regarding the datasets are provided in Appendix C.

a) Performance on Bimodal Benchmarks: For a fair comparison, we utilize identical encoders and modality configurations as CoMM. We evaluate MIIB under two settings: (1) self-supervised pretraining followed by linear probing and (2) supervised end-to-end fine-tuning after pretraining. Under the linear probing protocol, we compare MIIB against several strong baselines, including Cross, Cross+Self, FactorCL, and CoMM. In the fine-tuning setting, we focus on CoMM, the most closely related method, and conduct a controlled comparison using the same fine-tuning protocol.

As presented in Table II, under the linear probing setting, MIIB achieves the best performance across all evaluated datasets. In classification tasks, MIIB surpasses the CoMM baseline in average accuracy (66.73% vs. 64.45%). Specifically, we observe substantial improvements of 3.7, 2.1, 1.6, and 1.7 percentage points on the MOSI, UR-FUNNY, MUS-TARD, and MIMIC datasets. For the regression task, MIIB achieves an MSE of $5 . 7 4 ( \times 1 0 ^ { - 4 } )$ , outperforming the previous best result of 7.56. In the supervised full-parameter fine-tuning setting, MIIB consistently achieves better performance across all evaluated benchmarks, improving the average score over CoMM from 68.16 to 69.55.

TABLE II  
LINEAR PROBING $\mathrm { M S E } ( \times 1 0 ^ { - 4 } )$ FOR REGRESSION TASK AND TOP-1 ACCURACY (IN %) FOR CLASSIFICATION TASKS ON MULTIBENCH. <sup>†</sup> DENOTES RESULTS ARE FROM [16]. <sup>∗</sup> DENOTES AVERAGE IS ONLY SELECTED FROM THE RESULTS OF CLASSIFICATION TASKS. COLORED ROWS INDICATE FINE-TUNING RESULTS.
<table><tr><td rowspan="2">Model</td><td>Regression</td><td colspan="5">Classification</td></tr><tr><td>V&amp;T EE↓</td><td>MIMIC↑</td><td>MOSI↑</td><td>UR-FUNNY↑</td><td>MUSTARD↑</td><td> $_ \mathrm { A v e r a g e } ^ { \ast } \uparrow$ </td></tr><tr><td>Cross† [5]</td><td> $3 3 . 0 9 { \scriptstyle \pm 3 . 6 7 }$ </td><td> $6 6 . 7 { \scriptstyle \pm 0 . 1 }$ </td><td> $4 7 . 8 { \scriptstyle \pm 1 . 8 }$ </td><td> $5 0 . 1 { \pm } 1 . 9 $ </td><td> $5 3 . 5 { \pm } 2 . 9$ </td><td>54.52</td></tr><tr><td>Cross+Self† [54]</td><td> $7 . 5 6 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $6 5 . 4 9 { \scriptstyle \pm 0 . 0 }$ </td><td> $4 9 . 0 { \scriptstyle \pm 1 . 1 }$ </td><td> $5 9 . 9 { \scriptstyle \pm 0 . 9 }$ </td><td> $5 3 . 9 { \scriptstyle \pm 4 . 0 }$ </td><td>57.07</td></tr><tr><td>FactorCL† [15]</td><td> $1 0 . 8 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $6 7 . 3 { \scriptstyle \pm 0 . 0 }$ </td><td> $5 1 . 2 { \scriptstyle \pm 1 . 6 }$ </td><td> $6 0 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> $5 5 . 8 0 { \scriptstyle \pm 0 . 9 }$ </td><td>58.7</td></tr><tr><td>CoMM [16]</td><td> $7 . 9 6 { \scriptstyle \pm 2 . 1 3 }$ </td><td> $6 6 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td> $6 3 . 7 { \scriptstyle \pm 2 . 5 }$ </td><td> $6 3 . 3 { \scriptstyle \pm 0 . 5 }$ </td><td> $6 4 . 4 { \scriptstyle \pm 1 . 1 }$ </td><td>64.45</td></tr><tr><td>MIIB (ours)</td><td> ${ \bf 5 . 7 4 _ { \pm 0 . 6 9 } }$ </td><td> ${ \bf 6 8 . 1 { \scriptstyle \pm 0 . 3 } }$ </td><td> ${ \bf 6 7 . 4 { \scriptstyle \pm 1 . 8 } }$ </td><td> ${ \bf 6 5 . 4 { \scriptstyle \pm 0 . 9 } }$ </td><td> ${ \bf 6 6 . 0 { \scriptstyle \pm 2 . 0 } }$ </td><td>66.73</td></tr><tr><td>CoMM [16]</td><td> $1 . 6 1 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $6 7 . 5 7 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $7 1 . 8 4 { \scriptstyle \pm 2 . 7 }$ </td><td> $6 4 . 9 9 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $6 8 . 2 5 { \scriptstyle \pm 1 . 6 9 }$ </td><td>68.16</td></tr><tr><td>MIIB(fine-tuned)</td><td> ${ \bf 1 . 4 7 _ { \pm 0 . 0 6 } }$ </td><td> ${ \bf 6 9 . 0 _ { \pm 0 . 5 7 } }$ </td><td> ${ \bf 7 3 . 2 4 } _ { \pm 1 . 2 7 }$ </td><td> ${ \bf 6 6 . 2 4 } _ { \pm 0 . 1 8 }$ </td><td> $\mathbf { 6 9 . 7 3 { \scriptstyle \pm 2 . 3 0 } }$ </td><td>69.55</td></tr></table>

TABLE III

LINEAR PROBING TOP-1 ACCURACY (IN %) FOR CLASSIFICATION TASKS ON VISION&TOUCH AND UR-FUNNY. <sup>†</sup> DENOTES RESULTS ARE FROM [16]
<table><tr><td>Model</td><td>#Mod.</td><td> $V \mathcal { k } T \ C P \uparrow$ </td><td>UR-FUNNY↑</td></tr><tr><td>Cross [5]</td><td>2</td><td> $8 6 . 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 0 . 1 ^ { \dagger }$ </td></tr><tr><td>Cross+Self [54]</td><td>2</td><td> $8 7 . 6 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $5 9 . 9 ^ { \dagger }$ </td></tr><tr><td>CoMM [16]</td><td>2</td><td> $8 7 . 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $6 3 . 3 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td>MIIB (ours)</td><td>2</td><td> ${ \bf 8 8 . 4 _ { \pm 0 . 3 2 } }$ </td><td> ${ \bf 6 5 . 4 } _ { \pm 0 . 9 0 }$ </td></tr><tr><td>CMC† [61]</td><td>3</td><td>94.1</td><td>59.2</td></tr><tr><td>CoMM [16]</td><td>3</td><td> $9 4 . 1 _ { \pm 0 . 1 7 }$ </td><td> $6 4 . 8 _ { \pm 1 . 1 3 }$ </td></tr><tr><td>MIIB (ours)</td><td>3</td><td> ${ \bf 9 4 . 2 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> ${ \bf 6 6 . 2 } _ { \pm 0 . 8 5 }$ </td></tr></table>

b) Performance on Trimodal Benchmarks: To assess the generalization of MIIB to more than two modalities, we conduct experiments on two trimodal datasets from MultiBench: Vision&Touch (vision, force, and proprioception modalities for contact prediction) and UR-FUNNY (vision, text, and audio for humor detection). To better evaluate the information gain from the third modality, we include bimodal results for contrast, using visual and proprioceptive modalities for Vision&Touch and textual and visual modalities for UR-FUNNY. In the bimodal setting, we compare against Cross, Cross+Self, and CoMM; in the trimodal setting, we adopt CMC [61] and CoMM as baselines.

Overall, these results demonstrate the effectiveness of MIIB and highlight its robustness across diverse real-world benchmarks under bimodal settings.

Results are summarized in Table III. First, incorporating the additional modality yields consistent gains for MIIB: accuracy gains of 5.8 percentage points on Vision&Touch and 0.8 percentage points on UR-FUNNY. Second, MIIB achieves comparable or superior performance to all baselines in trimodal evaluations. Notably, on UR-FUNNY, it surpasses the leading baseline CoMM by 1.4 percentage points; on Vision&Touch, MIIB matches the state-of-the-art performance of CoMM. These results indicate that our model can be extended beyond bimodal settings, effectively utilizing additional modalities to further enhance performance.

## C. Experiments on Multimodal IMDb

MM-IMDb [62] is a real-world multi-label movie genre prediction dataset consisting of poster images and plot summaries. The task is challenging due to the highly imbalanced class distribution and the large semantic gap between the visual and textual modalities. Consistent with prior work [55], the visual (posters) and textual (plot summaries) modalities provide complementary cues—capturing aesthetic style and narrative content, respectively. Consequently, accurate genre prediction requires synergizing information from both modalities. We evaluate the proposed MIIB under two experimental settings: self-supervised linear probing and supervised fine-tuning. In the self-supervised setting, we compare against unimodal baselines (SimCLR [9], CLIP (Visual), and CLIP (Text) [5]) and multimodal baselines (SLIP [63], CLIP (Visual + Text), and CoMM). For a fair comparison, MIIB and CoMM are built on the same pretrained CLIP backbone. In the supervised finetuning setting, we compare MIIB against representative fully supervised baselines such as MFAS [64] and ReFNet [65]. In addition, we compare MIIB with CoMM under an identical full fine-tuning protocol.

As shown in Table IV, MIIB achieves the best performance across all evaluation metrics. Compared to the strong multimodal baseline CoMM, we obtain improvements of 1.51 percentage points in Weighted F1 and 2.41 percentage points in Macro F1. We also observe that CLIP with its original pretrained weights outperforms CLIP further trained on unlabeled MM-IMDb data (58.9% vs. 54.49% in Weighted F1). This observation suggests that relying solely on redundancy-focused alignment may be insufficient for movie genre prediction, which often benefits from complementary integration of visual and textual modalities [16]. In contrast, the superior performance of MIIB highlights its effectiveness on this imbalanced and semantically heterogeneous multi-label multimodal task. In the fine-tuning setting, MIIB consistently outperforms both existing fully supervised baselines and fine-tuned CoMM.

## D. Few-shot Transfer Performance on the MM-IMDb Dataset

To further evaluate the transferability and sample efficiency of the learned representations, we conduct few-shot experiments on the MM-IMDb dataset using only 1% and 10% of the labeled training data, together with a full fine-tuning setting for comparison. We restrict this few-shot evaluation to MM-IMDb, which contains 25,959 movies in total and is substantially larger than the affective-computing benchmarks considered here, such as MUSTARD (690 utterances) and MOSI (2,199 utterances). As shown in Table V, MIIB consistently outperforms CoMM across all supervision levels in terms of both Weighted-F1 and Macro-F1. Notably, under only 1% labeled data, MIIB outperforms CoMM by 2.54 in Weighted-F1 and 3.14 in Macro-F1, achieving scores of 63.22 and 56.80, respectively. The consistent improvements under the 10% and full-data settings further indicate that MIIB learns transferable multimodal representations that remain effective across different levels of supervision.

TABLE IV  
LINEAR PROBING F1-SCORE (WEIGHTED AND MACRO) (IN %) FOR MM-IMDB. △ INDICATES FURTHER TRAINING ON UNLABELED DATA. † DENOTES RESULTS ARE FROM [16]. FOR FAIR COMPARISON, BOTH COMM AND MIIB ARE IMPLEMENTED USING THE SAME PRE-TRAINED CLIP BACKBONE. COLORED ROWS INDICATE FINE-TUNING RESULTS.
<table><tr><td>Model</td><td>Modalities</td><td> $w e i g h t e d \ – f I \uparrow$ </td><td> $m a c r o { - } f I { \uparrow }$ </td></tr><tr><td>SimCLR† [9]</td><td>V</td><td> $4 0 . 3 5 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $2 7 . 9 9 { \scriptstyle \pm 0 . 3 3 }$ </td></tr><tr><td>CLIP† [5]</td><td>V</td><td> $^ \mathrm { 5 1 . 5 }$ </td><td>40.8</td></tr><tr><td>CLIP† [5]</td><td>L</td><td>51.0</td><td>43.0</td></tr><tr><td>CLIP† [5]</td><td>V+L</td><td>58.9</td><td>50.9</td></tr><tr><td>SLIPt△ [63]</td><td>V+L</td><td> $5 6 . 5 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 7 . 3 5 { \scriptstyle \pm 0 . 2 7 }$ </td></tr><tr><td> $\mathrm { C L I P ^ { \dag \triangle } \ [ 5 ] }$ </td><td>V+L</td><td> $5 4 . 4 9 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 4 . 9 4 { \scriptstyle \pm 0 . 3 0 }$ </td></tr><tr><td>CoMM [16]</td><td>V+L</td><td> $6 1 . 2 9 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 3 . 7 9 _ { \pm 0 . 2 2 }$ </td></tr><tr><td>MIIB (ours)</td><td>V+L</td><td> ${ \bf 6 2 . 8 0 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> ${ \bf 5 6 . 2 0 { \scriptstyle \pm 0 . 1 6 } }$ </td></tr><tr><td>MFAS† [64]</td><td>V+L</td><td>62.50</td><td>55.6</td></tr><tr><td>ReFNet† [65]</td><td>V+L</td><td></td><td>56.7</td></tr><tr><td>CoMM [16]</td><td>V+L</td><td> $6 3 . 9 2 _ { \pm 0 . 4 1 }$ </td><td> $5 7 . 9 0 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>MIIB (ours)</td><td>V+L</td><td> $\mathbf { 6 5 . 1 4 _ { \pm 0 . 1 1 } }$ </td><td> ${ \bf 5 9 . 3 2 _ { \pm 0 . 3 3 } }$ </td></tr></table>

TABLE V  
PERFORMANCE COMPARISON ON THE MM-IMDB DATASET UNDER DIFFERENT EVALUATION PROTOCOLS. RESULTS ARE REPORTED IN MACRO-F1 AND WEIGHTED-F1 ALONG WITH STANDARD DEVIATIONS. “LINEAR” DENOTES THE BEST SELF-SUPERVISED LINEAR PROBING RESULTS.
<table><tr><td>Setting</td><td>Model</td><td>weighted-f1↑</td><td>macro-f1↑</td></tr><tr><td rowspan="2">Linear</td><td>CoMM [16]</td><td> $6 1 . 2 9 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 3 . 7 9 _ { \pm 0 . 2 2 }$ </td></tr><tr><td>MIIB (ours)</td><td> ${ \bf 6 2 . 8 0 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> ${ \bf 5 6 . 2 0 { \scriptstyle \pm 0 . 1 6 } }$ </td></tr><tr><td rowspan="2">1%</td><td>CoMM [16]</td><td> $6 0 . 6 8 _ { \pm 0 . 3 1 }$ </td><td> $5 3 . 6 6 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>MIIB (ours)</td><td> ${ \bf 6 3 . 2 2 } _ { \pm 0 . 3 1 }$ </td><td> ${ \bf 5 6 . 8 0 { \scriptstyle \pm 0 . 4 7 } }$ </td></tr><tr><td rowspan="2">10%</td><td>CoMM [16]</td><td> $6 1 . 6 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 4 . 4 8 _ { \pm 0 . 7 2 }$ </td></tr><tr><td>MIIB (ours)</td><td> ${ \bf 6 3 . 6 0 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $\mathbf { 5 7 . 0 0 } _ { \pm 0 . 4 7 }$ </td></tr><tr><td rowspan="2">Full</td><td>CoMM [16]</td><td> $6 3 . 9 2 _ { \pm 0 . 4 1 }$ </td><td> $5 7 . 9 0 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>MIIB (ours)</td><td> ${ \bf 6 5 . 1 4 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> ${ \bf 5 9 . 3 2 } _ { \pm 0 . 3 3 }$ </td></tr></table>

## E. Quantitative Analysis of Cross-Modal Interactions via InterSHAP

To rigorously quantify the extent to which our proposed synergy constraint facilitates effective cross-modal interactions, we employ the InterSHAP metric [66]. Unlike traditional performance-based metrics (e.g., accuracy or F1 score) that only reflect the final outcome, InterSHAP is designed to explicitly disentangle and measure the contribution of interactions versus unimodal features to the model’s decision-making process. InterSHAP quantifies the proportion of a model’s behavior attributable to cross-modal interactions, leveraging the Shapley Interaction Index (SII) to decompose the prediction f(x) into individual modality contributions (main effects) and cross-modal interaction terms. Let M be the number of modalities and $\phi _ { i j }$ denote the SII value between modalities i and j (aggregated globally across the dataset or locally per sample). The InterSHAP score is defined as the ratio of the aggregated interaction effects to the total model behavior:

TABLE VI  
INTERSHAP ANALYSIS OVER 15 RANDOM SEEDS ON TWO BENCHMARK DATASETS. RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION TOGETHER WITH PAIRED t-TESTS.
<table><tr><td>Model Setting</td><td>UR-FUNNY</td><td>MIMIC</td></tr><tr><td>w/o Synergy</td><td> $5 . 7 8 { \scriptstyle \pm 2 . 0 5 }$   ${ \bf 7 . 3 2 } _ { \pm 1 . 9 8 }$ </td><td> $1 . 3 4 _ { \pm 0 . 8 0 }$   ${ \bf 2 . 5 4 { \scriptstyle \pm 1 . 2 6 } }$ </td></tr><tr><td>w/ Synergy</td><td>+1.54</td><td></td></tr><tr><td>Improvement t-test</td><td>2.70</td><td>+1.20 3.61</td></tr><tr><td></td><td></td><td></td></tr><tr><td>p-value</td><td>0.018</td><td>0.003</td></tr></table>

$$
\mathrm { I n t e r S H A P } = \frac { \sum _ { i , j = 1 } ^ { M } \left| \phi _ { i j } \right| } { \sum _ { i , j = 1 } ^ { M } \left| \phi _ { i j } \right| } ,\tag{28}
$$

where the numerator sums the absolute interaction effects $( i ~ \neq ~ j )$ , and the denominator represents the total absolute attributions, including unimodal main effects $( i = j )$ . A higher InterSHAP value indicates greater reliance on cross-modal interaction effects rather than unimodal main effects. This metric supports any M, provides both local (sample-level) and global explanations, and is performance-agnostic.

As shown in Table VI, we extend the InterSHAP analysis to 15 random seeds and conduct paired t-tests to assess statistical significance. The proposed synergy objective consistently increases InterSHAP values on both datasets, improving the interaction ratio from 5.78% to 7.32% on UR-FUNNY and from 1.34% to 2.54% on MIMIC. Moreover, the corresponding improvements are statistically significant $( p \ < \ 0 . 0 5 )$ , suggesting that the observed interaction gains are robust across different random initializations. Since InterSHAP is designed to quantify the relative contribution of cross-modal interaction effects, these results provide strong empirical evidence that the proposed synergy objective effectively encourages richer cross-modal complementary interactions. InterSHAP estimation variance is governed primarily by test-set size: on the two datasets above (UR-FUNNY, $N = 1 0 5 8 ;$ MIMIC, $N = 3 2 6 2 )$ , the per-seed variance is small enough for the synergy-induced gain to be clearly resolved. We additionally applied the same 15-seed analysis to MOSI $( N ~ = ~ 6 8 6 )$ and MUSTARD (N = 138); at this much smaller scale, the sampling variance of the Shapley interaction estimate itself becomes too large for any method comparison to be statistically reliable, independent of whether the underlying synergy effect is real. We therefore restrict our significance claim to the datasets on which InterSHAP can be reliably measured, and report the MOSI/MUSTARD results in full in Appendix B rather than omitting them.

## F. Ablation Studies

To rigorously evaluate the effectiveness of MIIB, we conduct ablation studies on two challenging real-world benchmarks: UR-FUNNY and MUSTARD. We select these datasets because, compared to synthetic benchmarks (e.g., Trifeature), affective computing tasks such as humor and sarcasm detection typically require more complex cross-modal interactions (e.g., semantic incongruity). Recent information-theoretic analysis [67] further shows that UR-FUNNY and MUSTARD exhibit higher levels of synergy under their proposed metric than many commonly used multimodal benchmarks. Specifically, our ablation study consists of two parts: loss function ablation and parameter sensitivity analysis.

TABLE VII  
LINEAR PROBING ACCURACY (IN %) ON UR-FUNNY AND MUSTARD DATASETS UNDER DIFFERENT COMBINATIONS OF LOSS FUNCTIONS. $\lambda _ { 1 } ,$ $\lambda _ { 2 } ,$ AND $\lambda _ { 3 }$ DENOTE THE WEIGHTS FOR L<sub>MF</sub> , L<sub>MS</sub> , AND L<sub>synergy</sub>
<table><tr><td colspan="3">Loss Weights</td><td rowspan="2"> $\mathrm { U R - F U N N \uparrow }$ </td><td rowspan="2">MUSTARD↑</td></tr><tr><td> $\lambda _ { 1 }$ </td><td> $\lambda _ { 2 }$ </td><td> $\lambda _ { 3 }$ </td></tr><tr><td>1</td><td>0</td><td>0</td><td> $6 3 . 5 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $6 2 . 4 { \scriptstyle \pm 3 . 2 2 }$ </td></tr><tr><td>1</td><td>1</td><td>0</td><td> $6 3 . 8 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $6 5 . 2 { \scriptstyle \pm 1 . 2 4 }$ </td></tr><tr><td>1</td><td>0</td><td>1</td><td> $6 3 . 9 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $6 2 . 8 { \scriptstyle \pm 1 . 6 1 }$ </td></tr><tr><td>1</td><td>1</td><td>1</td><td> ${ \bf 6 5 . 4 } _ { \pm 0 . 8 9 }$ </td><td> ${ \bf 6 6 . 0 { \scriptstyle \pm 2 . 0 } }$ </td></tr></table>

![](images/7f790bc5907cbf145ba88bb309bda66cce6f1553dfccb21033e219bfb00f50d2.jpg)  
Fig. 3. Parameter sensitivity analysis on the UR-FUNNY dataset with respect to the fusion penalty $( \lambda _ { \mathrm { f u s i o n } } ) .$ , unimodality penalty $( \lambda _ { \mathrm { { s i n g l e } } } ) .$ , and the synergy penalty $( \lambda _ { \mathrm { s y n e r g y } } )$

a) Loss Function: Table VII presents the ablation results for different loss components in MIIB. Using only $\lambda _ { 1 } ~ ( \mathrm { i . e . }$ only ${ \mathcal { L } } _ { M F } )$ yields the lowest performance on both datasets. Enabling the unimodal constraint $( \lambda _ { 2 } = 1 )$ consistently improves the results, with a particularly notable gain on MUS-TARD $( 6 2 . 4 \%  6 5 . 2 \% )$ . Similarly, enabling the synergy objective $( \lambda _ { 3 } = 1 )$ also leads to performance improvements over the baseline, demonstrating its effectiveness in modeling cross-modal interactions. When both objectives are jointly incorporated, MIIB achieves the best results of 65.4% on UR-FUNNY and 66.0% on MUSTARD, indicating that the unimodal constraint and the synergy objective provide complementary benefits for learning robust multimodal representations.

![](images/60119c5638ddea101f10694386fef30ea47e2834c930b57871a473f43f48e6f0.jpg)  
Fig. 4. Parameter sensitivity analysis of the hyperparameter $\gamma$ on the UR-FUNNY and MOSI datasets.

b) Parameter sensitivity analysis: We analyze the sensitivity of the key hyperparameters in MIIB. Figure 3 presents the results on the UR-FUNNY dataset for the loss weights $\lambda _ { \mathrm { f u s i o n } } , \lambda _ { \mathrm { s i n g l e } } ,$ and $\lambda _ { \mathrm { s y n e r g y } } ,$ which correspond to the entropy regularization coefficients for multimodal alignment, unimodal alignment, and synergistic representations, respectively. All three parameters exhibit non-monotonic trends, indicating that moderate values generally yield better performance than extreme settings. In particular, the unimodality penalty $\lambda _ { \mathrm { s i n g l e } }$ is the most sensitive: overly large values lead to a pronounced performance drop, indicating that excessively strong entropy regularization on unimodal representations can overly flatten the feature distribution and reduce discriminative power. In contrast, $\lambda _ { s y n e r g y }$ yields relatively stable performance across a broad range of values, suggesting robustness to its scaling. The fusion penalty $\lambda _ { \mathrm { f u s i o n } }$ reaches its best performance around $\lambda _ { \mathrm { f u s i o n } } = 0 . 1$ , with performance degrading when it is set too small or too large.

We further investigate the effect of the margin parameter $\gamma ,$ which controls the cosine-similarity constraint between unimodal and synergy representations. As shown in Fig. $^ { 4 , }$ both UR-FUNNY and MOSI exhibit non-monotonic trends with respect to γ, with the best performance obtained at moderate values $( \gamma \approx 0 . 2 – 0 . 3 )$ . Performance decreases when $\gamma$ is set either too small or too large, indicating that downstream performance is sensitive to the choice of the margin parameter.

## G. Computational Complexity and Runtime Analysis

As shown in Table VIII, MIIB maintains a computational complexity comparable to the CoMM baseline. The slight increase in complexity is primarily attributed to the synergy transformer, which adopts the same architectural design as the fusion transformer. Despite this structural addition, the resulting overhead remains modest relative to the CoMM baseline. For instance, on the large-scale MM-IMDb dataset, the parameter count and FLOPs increase by roughly 3–5% (e.g., params from 236.5M to 243.6M), with a corresponding increase in training time of roughly 12 minutes. We consider this slight computational cost to be a justifiable trade-off given the significant improvements in performance.

TABLE VIII  
COMPARATIVE ANALYSIS OF COMPUTATIONAL COMPLEXITY AND TRAINING EFFICIENCY. WE REPORT FLOPS, MACS, MODEL SIZE (#PARAMS), ANDTOTAL TRAINING TIME. ALL MEASUREMENTS ARE CONDUCTED ON A SINGLE NVIDIA 4090 GPU
<table><tr><td>Dataset</td><td>Model</td><td>FLOPs</td><td>MACs</td><td>#Params</td><td>Total Training Time</td></tr><tr><td rowspan="2">UR-FUNNY</td><td>CoMM</td><td>0.15G</td><td>0.08G</td><td>7.11M</td><td>0.48 h</td></tr><tr><td>MIIB</td><td>0.18G</td><td>0.09 G</td><td>7.13M</td><td>0.54 h</td></tr><tr><td rowspan="2">MMIMDb-clip</td><td>CoMM</td><td>36.27 G</td><td>18.14G</td><td>236.5 M</td><td>4.11 h</td></tr><tr><td>MIIB</td><td>38.08 G</td><td>19.04 G</td><td>243.6M</td><td>4.31 h</td></tr></table>

![](images/dd19709f76570ae3e937c1a5f2c132b4e138cfbc16db567a3f6362908e02da7c.jpg)

![](images/b1c337db14de927c88818bf5dab451efa597678efd93cd56d981ead58c5be7f5.jpg)  
Fig. 5. t-SNE visualization of unimodal and synergy representations on the MOSI (left) and MM-IMDb (right) datasets. The figure illustrates the distribution of unimodal representations (Modality 0, Modality 1) and the corresponding synergy representations in the learned embedding space.

## H. Visualization of Unimodal and Synergy Representational Disentanglement

To examine whether the proposed disentanglement loss encourages separation among different representations, we visualize unimodal and synergy representations using t-SNE on the MOSI and MM-IMDb datasets, as shown in Figure 5. As observed, unimodal representations from different modalities and the synergy representation occupy clearly separated regions in the embedding space on both datasets. This separation reflects the explicit constraint imposed by Eq. (25), which discourages excessive similarity between unimodal and synergy representations and promotes their separation. These visualizations provide qualitative evidence that the proposed objective encourages disentanglement between unimodal and synergy representations.

## VII. CONCLUSION

In this work, we formalize MIIB from an informationtheoretic perspective. We show that modality-specific learning and inter-modal fusion dependencies fundamentally limit existing multimodal methods. By employing PID to encode redundancy, uniqueness, and synergy, MIIB introduces a dualpath Transformer architecture with entropy-based objectives that produce minimal yet sufficient multimodal representations, consistently outperforming strong baselines across multiple benchmarks.

We acknowledge that MIIB introduces a modest computational overhead: on MM-IMDb, the dedicated Synergy Transformer increases parameters and FLOPs by roughly 3–5% over the CoMM backbone. The synergy objective is designed as an inductive bias toward synergistic cross-modal information rather than a strict or provable recovery of the theoretical PID synergy term S. We view this overhead as a favorable trade-off given the consistent gains in modeling cross-modal interactions. A current limitation of MIIB is its primary focus on static modalities and relatively compact backbones. Future work will extend MIIB to dynamic modalities such as video and explore its integration with large foundation models in multi-source settings, aiming to develop more robust and scalable multimodal AI systems.

## REFERENCES

[1] L. R. Soenksen, Y. Ma, C. Zeng, L. Boussioux, K. Villalobos Carballo, L. Na, H. M. Wiberg, M. L. Li, I. Fuentes, and D. Bertsimas, “Integrated multimodal artificial intelligence framework for healthcare applications,” npj Digit. Med., vol. 5, no. 1, p. 149, 2022.

[2] T. Tu, S. Azizi, D. Driess, M. Schaekermann, M. Amin, P.-C. Chang, A. Carroll, C. Lau, R. Tanno, I. Ktena, et al., “Towards generalist biomedical ai,” Nejm Ai, vol. 1, no. 3, p. AIoa2300138, 2024.

[3] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 11621–11631, 2020.

[4] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al., “Planning-oriented autonomous driving,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 17853–17862, 2023.

[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn., pp. 8748–8763, 2021.

[6] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al., “Flamingo: a visual language model for few-shot learning,” in Proc. Adv. Neural Inform. Process. Syst., vol. 35, pp. 23716–23736, 2022.

[7] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machineˇ learning: A survey and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, pp. 423–443, 2019.

[8] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Proc. Adv. Neural Inform. Process. Syst., vol. 36, pp. 34892–34916, 2023.

[9] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proc. Int. Conf. Mach. Learn., pp. 1597–1607, 2020.

[10] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 16000–16009, 2022.

[11] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.- H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in Proc. Int. Conf. Mach. Learn., pp. 4904–4916, 2021.

[12] K. Sridharan and S. M. Kakade, “An information theoretic framework for multi-view learning,” in Annu. Conf. Learn. Theory, 2008.

[13] Y. H. Tsai, Y. Wu, R. Salakhutdinov, and L. Morency, “Self-supervised learning from a multi-view perspective,” in Proc. Int. Conf. Learn. Representations, 2021.

[14] L. Wen, X. Wang, J. Liu, and Z. Xu, “MVEB: Self-supervised learning with multi-view entropy bottleneck,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 9, pp. 6097–6108, 2024.

[15] P. P. Liang, Z. Deng, M. Ma, J. Zou, L.-P. Morency, and R. Salakhutdinov, “Factorized contrastive learning: Going beyond multi-view redundancy,” in Proc. Adv. Neural Inform. Process. Syst., vol. 36, pp. 32971– 32998, 2023.

[16] B. Dufumier, J. Castillo-Navarro, D. Tuia, and J.-P. Thiran, “What to align in multimodal contrastive learning?,” in Proc. Int. Conf. Learn. Representations, 2025.

[17] P. L. Williams and R. D. Beer, “Nonnegative decomposition of multivariate information,” arXiv preprint arXiv:1004.2515, 2010.

[18] N. Bertschinger, J. Rauh, E. Olbrich, J. Jost, and N. Ay, “Quantifying unique information,” Entropy, vol. 16, no. 4, pp. 2161–2183, 2014.

[19] L. Wen, L. Li, J. Duan, Y. Dai, J. Liu, and Z. Kang, “Dependency, compression, and synergy: A unified information-theoretic view of multimodal learning,” arXiv preprint arXiv:2609.14421, 2026.

[20] C. Akkus, L. Chu, V. Djakovic, S. Jauch-Walser, P. Koch, G. Loss, C. Marquardt, M. Moldovan, N. Sauter, M. Schneider, et al., “Multimodal deep learning,” arXiv preprint arXiv:2301.04856, 2023.

[21] Y. Guo, C. Guo, A. Sun, H. He, X. Yang, Y. Lu, Y. Zhang, X. Guo, D. Zhang, J. Liu, J. Duan, Y. Xiao, L. Wen, H.-M. Xu, and Y. Dai, “Web-CogReasoner: Towards multimodal knowledge-induced cognitive reasoning for web agents,” in Proc. Int. Conf. Learn. Representations, pp. 36186–36210, 2026.

[22] S. K. D’mello and J. Kory, “A review and meta-analysis of multimodal affect detection systems,” ACM computing surveys (CSUR), vol. 47, no. 3, pp. 1–36, 2015.

[23] J. H. Friedman and B. E. Popescu, “Predictive learning via rule ensembles,” Ann. Appl. Statist., vol. 2, pp. 916–954, Sep. 2008.

[24] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proc. Annu. Meeting Assoc. Comput. Linguistics., pp. 6558–6569, 2019.

[25] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 10, pp. 12113–12132, 2023.

[26] G. Barnum, S. Talukder, and Y. Yue, “On the benefits of early fusion in multimodal representation learning,” arXiv preprint arXiv:2011.07191, 2020.

[27] M. Li, R. Xu, S. Wang, L. Zhou, X. Lin, C. Zhu, M. Zeng, H. Ji, and S.-F. Chang, “CLIP-Event: Connecting text and images with event structures,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 16420–16429, 2022.

[28] L. A. Hendricks, J. Mellor, R. Schneider, J.-B. Alayrac, and A. Nematzadeh, “Decoupling the role of data, attention, and losses in multimodal transformers,” Trans. Assoc. Comput. Linguistics, vol. 9, pp. 570– 585, 2021.

[29] Z. Lou, M. Lv, Y. Zhan, Y. Li, C. Zhang, and S. Hu, “Granular information bottleneck for deep multi-modal clustering,” IEEE Trans. Image Process., vol. 35, pp. 7307–7318, 2026.

[30] S. Hu, J. Fan, Y. Wu, J. Wang, X. Jiang, P. Lv, and M. Xu, “To the best of trust: Full-stage trusted multi-modal clustering,” IEEE Trans. Image Process., vol. 35, pp. 8635–8647, 2026.

[31] C. Ahuja, D. W. Lee, Y. I. Nakano, and L.-P. Morency, “Style transfer for co-speech gesture animation: A multi-speaker conditional-mixture approach,” in Proc. Eur. Conf. Comput. Vis., pp. 248–265, Springer, 2020.

[32] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in Proc. Int. Conf. Mach. Learn., pp. 8821–8831, 2021.

[33] R. Balestriero, M. Ibrahim, V. Sobal, A. Morcos, S. Shekhar, T. Goldstein, F. Bordes, A. Bardes, G. Mialon, Y. Tian, et al., “A cookbook of self-supervised learning,” arXiv preprint arXiv:2304.12210, 2023.

[34] L. Wen, Y. Zhou, L. He, M. Zhou, and Z. Xu, “Mutual information gradient estimation for representation learning,” in Proc. Int. Conf. Learn. Representations, April 26-30 2020.

[35] R. Bachmann, D. Mizrahi, A. Atanov, and A. Zamir, “MultiMAE: Multimodal multi-task masked autoencoders,” in Proc. Eur. Conf. Comput. Vis., pp. 348–367, 2022.

[36] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “ImageBind: One embedding space to bind them all,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 15180–15190, 2023.

[37] H. Wang, X. Guo, Z.-H. Deng, and Y. Lu, “Rethinking minimal sufficient representation in contrastive learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 16041–16050, 2022.

[38] Q. Zhang, M. Lu, S. Yu, J. Xin, and B. Chen, “Discovering common information in multi-view data,” Information Fusion, vol. 108, p. 102400, 2024.

[39] M. Kleinman, A. Achille, S. Soatto, and J. Kao, “Gacs-korner common information variational autoencoder,” Advances in Neural Information Processing Systems, vol. 36, pp. 66020–66043, 2023.

[40] A. Achille and S. Soatto, “Emergence of invariance and disentanglement in deep representations,” Journal of Machine Learning Research, vol. 19, no. 50, pp. 1–34, 2018.

[41] Y. Tian, C. Sun, B. Poole, D. Krishnan, C. Schmid, and P. Isola, “What makes for good views for contrastive learning?,” in Proc. Adv. Neural Inform. Process. Syst., vol. 33, pp. 6827–6839, 2020.

[42] Y. Lin, Y. Gou, X. Liu, J. Bai, J. Lv, and X. Peng, “Dual contrastive prediction for incomplete multi-view representation learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 4, pp. 4447–4461, 2023.

[43] L. Wen, Q. Dai, J. Liu, J. Zheng, Y. Dai, D. Wang, Z. Kang, J. Wang, Z. Xu, and J. Duan, “InfMasking: Unleashing synergistic information by contrastive multimodal interactions,” in Proc. Adv. Neural Inform. Process. Syst., 2025.

[44] Q. Dai, L. Wen, J. Duan, Y. Dai, D. Wang, M. Wang, M. Wang, J. Liu, H. Yan, and Z. Kang, “HRIL: Learning multimodal synergy via higherorder tensor modeling,” in Proc. Adv. Neural Inform. Process. Syst., 2026.

[45] X. Chen, L. Wen, J. Duan, D. Wang, Y. Dai, J. Bai, J. Liu, Z. Tian, G. Qiu, and Z. Kang, “MSCR: Jointly balancing utilization across modalities and discovering synergistic information,” in Proc. Adv. Neural Inform. Process. Syst., 2026.

[46] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” arXiv preprint physics/0004057, 2000.

[47] S. Hu, Z. Lou, X. Yan, and Y. Ye, “A survey on information bottleneck,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 8, pp. 5325–5344, 2024.

[48] M. Federici, A. Dutta, P. Forre, N. Kushman, and Z. Akata, “Learning´ robust representations via multi-view information bottleneck,” in Proc. Int. Conf. Learn. Representations, 2020.

[49] R. A. Amjad and B. C. Geiger, “Learning representations for neural network-based classification using the information bottleneck principle,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 9, pp. 2225–2239, 2020.

[50] D. J. Strouse and D. J. Schwab, “The deterministic information bottleneck,” Neural Comput., vol. 29, no. 6, pp. 1611–1630, 2017.

[51] E. Abdelaleem, I. Nemenman, and K. M. Martini, “Deep variational multivariate information bottleneck - a framework for variational losses,” Journal ofMachine Learning Research, vol. 26, no. 140, pp. 1–50, 2025.

[52] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in Proceedings of the 37th International Conference on Machine Learning, vol. 119 of Proceedings of Machine Learning Research, pp. 9929–9939, PMLR, 13–18 Jul 2020.

[53] K. Hermann and A. Lampinen, “What shapes feature representations? exploring datasets, architectures, and training,” in Proc. Adv. Neural Inform. Process. Syst., vol. 33, pp. 9995–10006, 2020.

[54] X. Yuan, Z. Lin, J. Kuen, J. Zhang, Y. Wang, M. Maire, A. Kale, and B. Faieta, “Multimodal contrastive training for visual representation learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 6995– 7004, 2021.

[55] P. P. Liang, Y. Lyu, X. Fan, Z. Wu, Y. Cheng, J. Wu, L. Chen, P. Wu, M. A. Lee, Y. Zhu, et al., “MultiBench: Multiscale benchmarks for multimodal representation learning,” in Proc. Adv. Neural Inform. Process. Syst. – Track on Datasets and Benchmarks, vol. 1, 2021.

[56] A. E. W. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi, B. Moody, P. Szolovits, L. A. Celi, and R. G. Mark, “MIMIC-III, a freely accessible critical care database,” Scientific data, vol. 3, no. 1, p. 160035, 2016.

[57] A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency, “Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages,” IEEE Intell. Syst., vol. 31, no. 6, pp. 82–88, 2016.

[58] M. K. Hasan, W. Rahman, A. Zadeh, J. Zhong, M. I. Tanveer, L.- P. Morency, et al., “UR-FUNNY: A multimodal language dataset for understanding humor,” in EMNLP-IJCNLP, pp. 2046–2056, 2019.

[59] S. Castro, D. Hazarika, V. Perez-Rosas, R. Zimmermann, R. Mihalcea,´ and S. Poria, “Towards multimodal sarcasm detection,” in Proc. 57th Annu. Meeting Assoc. Comput. Linguist., pp. 4619–4629, 2019.

[60] M. A. Lee, Y. Zhu, P. Zachares, M. Tan, K. Srinivasan, S. Savarese, F.-F. Li, A. Garg, and J. Bohg, “Making sense of vision and touch: Learning multimodal representations for contact-rich tasks,” IEEE Trans. Robot., vol. 36, no. 3, pp. 582–596, 2020.

[61] Y. Tian, D. Krishnan, and P. Isola, “Contrastive multiview coding,” in Proc. Eur. Conf. Comput. Vis., pp. 776–794, 2020.

[62] J. Arevalo, T. Solorio, M. Montes-y Gomez, and F. A. Gonz´ alez, “Gated´ multimodal units for information fusion,” in Proc. Int. Conf. Learn. Representations Workshops (ICLR-W), 2017.

[63] N. Mu, A. Kirillov, D. Wagner, and S. Xie, “SLIP: Self-supervision meets language-image pre-training,” in Proc. Eur. Conf. Comput. Vis., pp. 529–544, 2022.

[64] J.-M. Perez-R ´ ua, V. Vielzeuf, S. Pateux, M. Baccouche, and F. Jurie,´ “MFAS: Multimodal fusion architecture search,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 6966–6975, 2019.

[65] S. Sankaran, D. Yang, and S.-N. Lim, “Refining multimodal representations using a modality-centric self-supervised module,” 2022.

[66] L. Wenderoth, K. Hemker, N. Simidjievski, and M. Jamnik, “Measuring cross-modal interactions in multimodal models,” in Proc. AAAI Conf. Artif. Intell., pp. 21501–21509, 2025.

[67] P. P. Liang, Y. Cheng, X. Fan, C. K. Ling, S. Nie, R. Chen, Z. Deng, F. Mahmood, R. Salakhutdinov, and L.-P. Morency, “Quantifying & modeling feature interactions: An information decomposition framework,” in Proc. Adv. Neural Inform. Process. Syst., vol. 36, pp. 27351– 27393, 2023.

![](images/0bc82bc646e7d6b89db7f87a072571a1d303f89e82120ee8c8c2677c7c9c3206.jpg)

Liangjian Wen is currently an associate professor with the School of Computing and Artificial Intelligence, Southwestern University of Finance and Economics, Chengdu, China. The Ph.D. degree was obtained from the School of Computer Science and Engineering, University of Electronic Science and Technology of China (UESTC) in July 2021. His research interests include machine learning, deep learning, representation learning, and selfsupervised learning.

![](images/b59f7811f518721756609b93ed843f6da274c08bffdf54a382a8175295b1c842.jpg)

Jiangtao Zheng received the BE degree from Zhengzhou University of Aeronautics, Zhengzhou, China. He is currently pursuing the M.S. degree with the School of Computing and Artificial Intelligence, Southwestern University of Finance and Economics, Chengdu, China. His research interests include machine learning, deep learning, and representation learning.

![](images/a85ef54a37e5c61044001191fae6d0c8269051bcaa45517278e52abec2eb1add.jpg)

Jiang Duan received the BSc degree in mechanical engineering from Southwest Jiaotong University, Chengdu, China, in 2001, the MSc degree from the University of Derby, Derby, U.K., in 2002, and the PhD degree from the University of Nottingham, U.K., in 2006. He is a professor at the Southwestern University of Finance and Economics, China. He is a recognized national expert, the winner of the first Sichuan Outstanding Talent Award (The Highest Talent Award of Sichuan province),

the winner of the Sichuan Youth Science and Technology Award, and a standing committee member of the Sichuan Association for Science and Technology. He has published more than 30 academic papers, won more than 10 international and national patents, and his research has been funded by about 20 national and provincial funds.

![](images/fa08cda97db5eff95c64b52f7a67d4e5e564e3301a2505b985b9825dad21e16c.jpg)

Qun Dai received the BE degree from Southwestern University of Finance and Economics, Chengdu, China, in 2025. He is currently working toward the master’s degree with the School of Computing and Artificial Intelligence, Southwestern University of Finance and Economics, Chengdu, China. His research interests include machine learning, deep learning, and representation learning.

![](images/f37a26b8a40b7ca80d1cf4e0942e7db6846ae0398b3eaa885ccd57f18e766c91.jpg)

Yong Dai received the Ph.D. degree from the University of Electronic Science and Technology of China in 2021. He previously conducted research on large language models at Microsoft Research Asia and at Tencent AI Lab. Since May 2025, he has served as Head of Multimodal Large Models at the Beijing Humanoid Robot Innovation Center, leading research on embodied braincerebellum systems and world models.

![](images/064c70f6c08a13d2fa64c09b503de452e6fc71569546f18a9a7c27681b362339.jpg)

Jianzhuang Liu (Senior Member, IEEE) received the PhD degree in computer vision from The Chinese University of Hong Kong, in 1997. From 1998 to 2000, he was a research fellow with Nanyang Technological University, Singapore. From 2000 to 2012, he was a postdoctoral fellow, an assistant professor, and an adjunct associate professor with The Chinese University of Hong Kong, Hong Kong. In 2011, he joined the Shenzhen Institutes of Advanced Technology, Chinese Academy of

Sciences, Shenzhen, China, as a professor. He was a principal researcher in Huawei Company from 2012 to 2023. He has authored more than 200 papers in the areas of computer vision, image processing, machine learning, and multimodal AI.

![](images/2d5c5646f4e512bc022b722f7d15022d444c60a1d3f26b1f4fc21cc5eb5b2938.jpg)

Jun Wang , Professor, Doctoral Supervisor, Southwestern University of Finance and Economics, School of Management Science and Engineering. Jun Wang has long been committed to emerging interdisciplinary research in fintech, financial intelligence, and digitalintelligence management. From the perspective of big data analytics, he addresses significant issues in financial markets using advanced machine learning, data analytics, and artificial intelligence methods.

![](images/5920edfbc6149a1c14eefad435cedfded69992706ac59483478532272f76374a.jpg)

Shu-Dong Huang (Member, IEEE) received his PhD degree in computer science and engineering from the University of Electronic Science and Technology of China, Chengdu, China. He is currently an associate research professor with the School of Artificial Intelligence, Sichuan University, Chengdu. His research interests include multimodal representation learning, and multimodal Large Language Models.

![](images/ea33205178d2e7944d261ab6fa346951cb88ff0941d2c86753091c8ebb9e4a3d.jpg)

Xinwang Liu received his PhD degree from National University of Defense Technology (NUDT), China, in 2013. He is now Professor at School of Computer, NUDT. His current research interests include kernel learning, multi-view clustering and unsupervised feature learning. Dr. Liu has published 80+ peer-reviewed papers, including those in highly regarded journals and conferences such as IEEE TPAMI, IEEE T-KDE, IEEE T-IP, IEEE T-NNLS, IEEE T-MM, IEEE T-IFS, ICML,

NeurIPS, CVPR, ICCV, AAAI, IJCAI, etc. He is an Associate Editor of IEEE T-NNLS and Information Fusion Journal.

![](images/4a92333adb225aedd581b12d84b5f92d53f11e8deaf93f80631a98c42ecc31e9.jpg)

Zenglin Xu (SM’20) is a Distinguished Professor at the Artificial Intelligence Innovation and Incubation (AI³) Institute, Fudan University, and an AI Scientist at the Shanghai Academy of AI for Science. He earned his Ph.D. in Computer Science and Engineering from The Chinese University of Hong Kong and completed postdoctoral research at Purdue University and the Max Planck Institute for Informatics. Dr. Xu serves as Vice President of the International Neural Network Society

(INNS) and is the Editor-in-Chief of Academia AI and Applications. His research centers on trustworthy machine learning and agentic AI, AI for Science, and domain-specific applications in healthcare, finance and education.

![](images/d069d68e8234e0b165b19f96b9479d558649d1b4fdee3db523ed5a220a478e84.jpg)

Zhao Kang received the Ph.D. degree in computer science from the Southern Illinois University Carbondale, Carbondale, IL, USA, in 2017. He is currently a Professor in the School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, China. His research interests include graph machine learning and large language models. Dr. Kang has been an AC/SPC/PC Member for a number of top conferences, such as NeurIPS, ICML, ICLR, AAAI,

IJCAI, CVPR, and SIGKDD. He serves as an Associate Editor for Neural Networks and Pattern Recognition.