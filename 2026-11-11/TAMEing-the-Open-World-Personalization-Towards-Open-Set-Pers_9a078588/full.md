# TAMEing the Open-World Personalization: Towards Open-Set Personalized MLLM Assistant

Rongpei Hong<sup>1</sup> Jian Lang<sup>1</sup> Ting Zhong<sup>1∗</sup> Fan Zhou<sup>1,2</sup> <sup>1</sup>University of Electronic Science and Technology of China <sup>2</sup>Intelligent Digital Media Technology Key Laboratory of Sichuan Province

## Abstract

Personalizing Multimodal Large Language Models (MLLMs) as intelligent assistants has recently emerged as an active research topic, where an MLLM is expected to go beyond generic, one-size-fits-all replies, and instead generate responses grounded in user-specific objects, entities, and concepts encountered during interactions. Nevertheless, existing studies mainly focus on personalizing MLLMs on a static, predefined set of personalized concepts and require users to manually and frequently extend this set to accommodate newly emerging concepts, which is infeasible in dynamic-world scenarios. To address this limitation, we recast conventional MLLM personalization as an Open-Set problem, and propose TAME-O, the first long-context open-set personalized MLLM assistant. Specifically, TAME-O introduces a training-free, plug-and-play Concept Reconciliation Skill (CR Skill). It enables the assistant to automatically unlock novel concepts during prolonged human–MLLM interactions while preventing confusion with previously unlocked ones, allowing the MLLM to continually grasp more and more concepts and progressively refine the modeling of known ones. Extensive experiments under the open-set personalization setting underscore the efficacy of our method.

## 1 Introduction

Imagine a future intelligent assistant that, through prolonged interactions, can not only remember you, but also gradually uncover and internalize your personal entities that are never specified in advance: your blue cup, and your yellow puppy Timi. When the assistant “sees” these entities in later conversations, it can visually recognize them and generate “personalized responses” that are grounded in the appropriate contexts, rather than one-size-fits-all answers.

All these imaginations are closely tied to the research field of multimodal large language model (MLLM) personalization [1, 2, 3, 4], which aims to enable general-purpose MLLMs (e.g., GPT-5 [5], Qwen3.5 [6]) to understand user-specific concepts, also referred to as “personalized concepts”<sup>1</sup> (e.g., Bob’s dog), instead of treating them as generic objects. Most existing studies, such as MyVLM [1] and R2P [7], focus on the capability of MLLMs in visual identification of personalized concepts, followed by a simple substitution of the corresponding concept in the output generation. Nevertheless, such static applications are overly simplified and fail to cater to real-world demands, where user-MLLM interactions are prolonged and dynamic, and dialogues become more complex due to extended contexts. As a remedy, TAME [4] explored long-context personalized dialogues with MLLMs by modeling long- and short-term attribute shifts of each concept over time, enabling the personalization experience to progressively improve over extended human-MLLM interaction.

However, all existing methods, including TAME, can only operate under the closed-setpersonalization setting: the MLLM is personalized on a fixed and static set of user-specific concepts that are manually predefined by the user. In real-world scenarios, users frequently encounter new personalized concepts [8, 7], and this rigidity severely undermines the flexibility of existing methods, which require users to frequently register the “novel personalized concepts” for the MLLM by hand (cf. Figure 1(a)). In contrast, an ideal MLLM assistant should gradually unlock more and more novel personalized concepts automatically during continued human–MLLM interaction, while progressively refining the modeling of previously known concepts. We term this capability Open-Set MLLM Personalization, an appealing property for MLLM assistants in the dynamic world (cf. Figure 1(b)).

![](images/eb1d8cae0b1001ec2ecd33d049f593bb76c04702e15972c0a7660ef39b26a67f.jpg)  
Figure 1: Paradigm comparison between (a) Closed-Set Personalization, where the MLLM passively acquires novel concepts from user provision, and (b) Open-Set Personalization, where the MLLM actively grasps new concepts from everyday interactions.

To achieve open-set personalization, the core challenge lies in enabling the MLLM assistant to reliably resolve newly emerging personalized concepts with existing ones over extended, longcontext interactions. On one hand, when novel personalized concepts are misidentified as existing ones, the assistant may accumulate erroneous contexts for the existing concepts. On the other hand, misclassifying an existing user-specific concept as novel not only prevents the assistant from leveraging accumulated contexts during generation, but also leads to conceptfission, where an existing concept is erroneously forked into two or more concepts.

To address these two symmetric problems, we draw upon the insight that user-specific concepts (e.g., visual appearance) evolve over time. As a result, the past dialogues that are temporally proximal to the current interaction often provide more reliable cues for identification than distant history. Accordingly, when comparing a newly appearing concept with those grasped from past dialogues, more recent memories should be emphasized for robust concept maintenance.

To this end, we propose TAME-O, the first Open-set, long-context personalized MLLM assistant that can automatically TAME novel user-specific concepts as they appear during ongoing interactions, beyond a predefined closed-set of concepts. TAME-O endows the MLLM assistant with a novel training-free Concept Reconciliation Skill (CR Skill) that mediates between incoming concepts and stored knowledge. Specifically, the CR Skill equips the MLLM with a new Time-Aware Dynamic Memory (TAD Memory), which stores personalized concepts along with their associated attributes extracted from past interactions, where each attribute is associated with a temporal marker to reflect when it was observed. Subsequently, when a concept is involved in the current interaction, the CR Skill retrieves candidate relevant concepts from the TAD Memory by considering concept attributes in a temporally weighted manner, prioritizing memories closer to the current interaction. These retrieved candidates, together with the current user query and recent dialogue contexts, are jointly reasoned by the MLLM to determine whether the concept aligns with an existing one or represents a new concept, enabling the assistant to effectively grasp novel user-specific concepts while continuously refining unlocked ones. The CR Skill is designed to be flexible and can be readily integrated into existing personalized MLLM assistants (e.g., TAME) in a plug-and-play way, granting them open-set personalization capability in the dynamic world. We list our contributions as follows:

• We formalize the problem of Open-Set MLLM Personalization, moving beyond static concept dictionaries to enable continuous, autonomous concept acquisition in dynamic environments.

• We propose TAME-O, the first open-set, long-context personalized MLLM assistant. TAME-O introduces a novel plug-and-play CR Skill that can be integrated into existing MLLM assistants to activate their open-set capability, automatically managing the lifecycle of personalized concepts by detecting the new and refining the old.

• Experiments on both existing LCMP-H and LCMP-O, a dedicated benchmark for open-set MLLM personalization newly constructed in this study, showcase the efficacy of TAME-O. Code is provided at https://github.com/ronpay/TAME-O.

## 2 Related Work

MLLM Personalization. MLLMs have been widely adopted as assistants, while most mainstream models still operate under a one-size-fits-all paradigm, treating all users as interchangeable and failing to adapt to individual identities, preferences, or interaction histories [3, 5, 9, 10, 11, 12]. MyVLM [1] and Yo’LLaVA [2] pioneered the concept of MLLM personalization, which enables an MLLM with generic knowledge to respond to user visual queries in a personalized manner. They achieved personalization by training a set of parameters to visually memorize each concept [13, 14]. To alleviate the frequent retraining for each newly emerging concept, studies such as RAP [8], PVIT [15], R2P [7], and PLVM [16] explored memory-based or tuning-free mechanisms, where a set of user-predefined personalized concepts is first stored in memory, and an incoming concept is then treated as a query to visually match against these stored concepts for personalized response generation. However, all these methods primarily focus on the visual identification of personalized concepts, that is, recognizing an object in an image as user-specific and replacing a generic object reference with the corresponding personalized concept name for captioning or single-turn question-answering. Such overly simplified settings fail to meet real-world requirements, where human–MLLM interactions are persistent and highly contextualized, and personalization must account for complex dependencies from extended interaction history rather than isolated, static inputs. To remedy such limitation, TAME [4] proposed the first long-context MLLM personalized assistant, which fosters MLLM to capture the variations in both long- and short-term attributes of user-specific concepts. Nevertheless, existing methods (including TAME) are limited to only responding to a manually predefined set of user-specific concepts and fail to adapt to dynamic real-world settings. In contrast, we propose TAME-O, which activates Open-Set Personalization capability through a training-free, plug-and-play Concept Reconciliation Skill (CR Skill), which effectively and autonomously unlocks novel personalized concepts while avoiding potential conflicts with previously “memorized” ones. The additional literature reviews for memory-augmented LLM and open-set recognition are in Appendix G.

## 3 Methodology: TAME-O

Revisiting TAME. We begin by re-formalizing the pipeline of TAME [4], the first long-context personalized MLLM assistant. TAME considers a continuous stream of human–MLLM interactions: ${ \boldsymbol { S } } =$ $\langle \mathcal { Q } _ { 1 } , \mathcal { Q } _ { 2 } , \ldots , \mathcal { Q } _ { i } , \ldots \rangle$ , where each $\mathcal { Q } _ { i }$ denotes an atomic single-turn vision–language dialogue occurring at time step i, consisting of a user query $q _ { i }$ and the corresponding assistant response $a _ { i }$ . When responding to $q _ { i }$ , the MLLM assistant is required to leverage the potential contexts from the past dialogues tied to the personalized concept appearing in $\mathcal { Q } _ { i } \colon \bar { a } _ { i } = \bar { \Phi } _ { \mathrm { M L L M } } \big ( q _ { i } , \pi \big ( \langle \mathcal { Q } _ { 1 } , \mathcal { Q } _ { 2 } , \ldots , \bar { \mathcal { Q } } _ { i - 1 } \rangle \big ) \big )$ where $\pi ( \cdot )$ denotes a context extraction strategy that filters out relevant information for current concept from the interaction history. To achieve this, TAME endows the assistant with double memories, a static memory $\mathcal { M } _ { S }$ and a dynamic memory $\mathcal { M } _ { D }$ , which are designed to maintain the long-term and short-term attributes of personalized concepts from past dialogues, respectively. Each memory stores items as a tuple (Cid, Attr), where Cid is the concept identifier that is used to uniquely identify a personalized concept, while Attr denotes an attribute extracted from a specific dialogue turn. Subsequently, when facing an incoming personalized concept, TAME first identifies its corresponding concept identifier Cid, and then aggregates the memory items (attributes) that are marked with the same Cid from $\mathcal { M } _ { S }$ and $\mathcal { M } _ { D }$ to support personalized response generation.

However, TAME implicitly assumes a closed-set personalization setting, where a set of $N$ userspecific concepts is predefined and fixed throughout the interaction: $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , \cdots , c _ { N } \}$ , where C is the static concept set. When encountering a novel concept $c _ { N + 1 }$ , TAME requires the user to manually initialize this concept and provide the corresponding concept identifier for proper memory management. Consequently, such a setting is ill-suited to dynamic real-world scenarios, where new personalized concepts frequently arise, and requiring users to explicitly specify concept identifier introduces significant overhead and undermines usability.

Open-Set MLLM Personalization Formalization. We consider a more real-world setting, in which personalized concepts are not predefined by users but are instead automatically grasped by the MLLM assistant through extended interactions. Specifically, the concept set is defined as: $\mathcal { C } _ { i - 1 } = \{ c _ { 1 } , c _ { 2 } , \cdot \cdot \cdot , c _ { K _ { i - 1 } } \}$ , where $\mathcal { C } _ { i - 1 }$ is the dynamic concept set at time step $i - 1$ that contains $K _ { i - 1 }$ concepts unlocked by the assistant up to the current step (empty at start). At time step i, the concept referred to in the dialogue $\mathcal { Q } _ { i }$ is judged to be either novel or existing. If it is identified as a novel concept, the personalized concept set is updated by the assistant as: $\mathcal { C } _ { i } = \{ c _ { 1 } , c _ { 2 } , \cdot \cdot \cdot , c _ { K _ { i - 1 } } , c _ { K _ { i } } \}$ , where $c _ { K _ { i } }$ is the new concept appearing in $\mathcal { Q } _ { i }$

![](images/2359c2fe38d54143b6e5d36767134e0332ed536493c5bf77d2bb216147b18809.jpg)  
Figure 2: Paradigm of TAME-O. When receiving a new dialogue, the CR skill first retrieves candidate personalized concepts from the TAD Memory in a temporally weighted manner. The retrieved candidates are then provided to the MLLM, together with the user query and recent dialogue context, to determine whether the referenced concept is existing or novel, achieving open-set personalization.

Paradigm of TAME-O. To address this challenging setting, TAME-O provides existing personalized assistants with a novel training-free and plug-and-play Concept Reconciliation Skill (CR Skill), which handles novel concepts while mitigating potential conflicts with existing ones. Specifically, TAME-O first equips the MLLM assistant with a Time-Aware Dynamic Memory (TAD Memory), which captures variations of personalized concepts in a temporally annotated manner (cf. Section 3.1). Subsequently, when a new interaction occurs, the CR Skill evaluates the incoming concept against memory items using a temporally weighted matching strategy, and provides a set of candidate concepts to the MLLM for final resolution of this concept (cf. Section 3.2). We further provide an example of integrating the CR Skill into the existing personalized assistant TAME (cf. Section 3.3). Finally, we discuss potential concept fission issues and present corresponding mitigation solutions (cf. Section 3.4). The framework of TAME-O is in Figure 2.

## 3.1 Time-Aware Dynamic Memory

We consider that after time step i − 1, the personalized concepts that have been identified by the MLLM assistant are represented as a dynamic concept set $\mathcal { C } _ { i - 1 } \bar { = } \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { K _ { i - 1 } } \}$ , where $K _ { i - 1 }$ denotes the number of personalized concepts discovered up to time step $i - 1$ . Given a new dialogue $\mathcal { Q } _ { i } ,$ a simple approach to determining whether a novel personalized concept appears is to compare the information of the concept extracted from $a _ { i }$ against the grasped concepts in memory. However, in the long-context interactions, the information (also referred to as attributes) of each concept often varies over time [4]. Consequently, early attributes of a concept may become outdated and unfaithful for depicting such concept, whereas more recent attributes can better reflect its current state.

Inspired by this observation, we introduce a novel training-free Concept Reconciliation Skill (CR Skill), which first endows the MLLM assistant with a new Time-Aware Dynamic Memory (TAD Memory) to store each attribute in a temporally-annotated manner. Specifically, taking completing a dialogue $\mathcal { Q } _ { k }$ and storing the newly acquired information tied to the corresponding personalized concept as an example, the TAD Memory $\mathcal { M } _ { T }$ operates as follows:

$$
( \mathtt { C i d } _ { k } , \mathtt { A t t r } _ { k } , \delta _ { k } ) = \Phi ( \mathtt { C i d } _ { k } , \mathcal { Q } _ { k } , \mathcal { M } _ { T } , \mathcal { P } _ { u } ) ,\tag{1}
$$

$$
\mathcal { M } _ { T }  \Psi ( \mathcal { M } _ { T } , \mathtt { C i d } _ { k } , \mathtt { A t t r } _ { k } , \delta _ { k } , t _ { k } ) ,\tag{2}
$$

where ${ \tt C i d } _ { k }$ is the identifier of the concept referred to in $\mathcal { Q } _ { k }$ (its acquisition is discussed in the next section), $\mathtt { A t t r } _ { k }$ is the attribute information extracted from $\mathcal { Q } _ { k } , \delta _ { k }$ is the memory execution indicator determined by the MLLM $\Phi ( \cdot )$ for this attribute $\mathtt { A t t r } _ { k }$ that follows the update actions adopted in TAME (i.e., add or modify an attribute), $\mathcal { P } _ { u }$ is the prompt for memory update, and $\Psi ( \cdot )$ is a memory update operation that updates the TAD Memory according to the indicator $\delta _ { k } . ~ t _ { k }$ is the temporal marker for $\mathtt { A t t r } _ { k }$ , which is separately maintained for each concept and set to 0 for the first stored memory item (attribute). It is incremented by 1 and assigned to each new attribute that is added or updated. Consequently, TAD Memory stores each attribute in a triple tuple: $( \mathsf { C i d } _ { k } , \mathsf { A t t r } _ { k } , t _ { k } )$ . If Cid is determined to be novel, then the concept set is updated to $\mathcal { C } _ { i } = \{ c _ { 1 } , c _ { 2 } , \cdot \cdot \cdot , c _ { K _ { i - 1 } } , c _ { K _ { i } } \}$

## 3.2 Open-Set Concept Reconciliation

When starting a new dialogue $\mathcal { Q } _ { i }$ , the assistant is required to identify the concept appearing in user query $q _ { i }$ , resolving whether it is novel or existing and giving its appropriate identifier. Specifically, the CR Skill first retrieves a set of candidate personalized concepts by comparing the visual <sup>2</sup> and textual information extracted from $q _ { i }$ with the information of each concept $c _ { k }$ stored in the TAD Memory in a temporally weighted manner, emphasizing the memory items that are closer to the q<sub>i</sub>:

$$
S ( c _ { k } \mid q _ { i } ) = \frac { 1 } { Z _ { k } } \sum _ { j = 1 } ^ { N _ { k } } \exp \left( - \lambda \cdot \frac { t _ { k } ^ { \mathrm { m a x } } - t _ { k } ^ { ( j ) } } { N _ { k } } \right) \cdot s _ { k } ^ { ( j ) } ,\tag{3}
$$

$$
s _ { k } ^ { ( j ) } = \cos ( \phi ( \mathcal { T } _ { i } ) , \phi ( \mathcal { T } _ { k } ^ { ( j ) } ) ) + \cos ( \phi ( \mathcal { V } _ { i } ) , \phi ( \mathcal { V } _ { k } ^ { ( j ) } ) ) .\tag{4}
$$

Here, $S ( c _ { k } \mid q _ { i } )$ denotes the temporally weighted similarity score between the information extracted from $q _ { i }$ and the $N _ { k }$ historical memory items associated with the personalized concept $c _ { k }$ in $\mathcal { M } _ { T } \colon$ $\{ ( \mathcal { T } _ { k } ^ { ( j ) } , \mathcal { V } _ { k } ^ { ( j ) } , t _ { k } ^ { ( j ) } ) \} _ { j = 1 } ^ { N _ { k } } . ~ \mathcal { T } _ { i }$ and $\nu _ { i }$ represent the textual and visual information extracted from the incoming user query $q _ { i }$ , while $( T _ { k } ^ { ( j ) } , \mathcal { V } _ { k } ^ { ( j ) } )$ correspond to the textual and visual attributes stored in the $j \cdot$ -th memory item of concept $c _ { k } . t _ { k } ^ { ( j ) }$ is the temporal marker for that item, and $t _ { k } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j } t _ { k } ^ { ( j ) }$ denotes the most recent temporal marker of attributes for concept $c _ { k } . \ s _ { k } ^ { ( j ) }$ is the similarity score between the current query and the j-th memory item, computed using cosine similarity in the embedding space defined by ϕ(·). λ is the temporal decay coefficient, and $Z _ { k }$ is the normalization term ensuring a scaled concept-level similarity score, defined as:

$$
Z _ { k } = \sum _ { j = 1 } ^ { N _ { k } } \exp \left( - \lambda \cdot \frac { t _ { k } ^ { \operatorname * { m a x } } - t _ { k } ^ { ( j ) } } { N _ { k } } \right) .\tag{5}
$$

Subsequently, the CR skill selects the Top-K candidate personalized concepts $\mathcal { C } _ { K }$ with the highest similarity scores from the grasped concept set $\mathcal { C } _ { i - 1 }$ after time step $i - 1$ . For all candidates, the assistant feeds their most recent memory items, together with the current query $q _ { i }$ and the most recent W dialogue turns, into the MLLM to determine whether the referred concept corresponds to an existing personalized concept or a novel one:

$$
\mathsf { C i d } ^ { * } = \Phi \left( \{ \mathcal { M } _ { k } ^ { \mathrm { r } } \} _ { c _ { k } \in \mathcal { C } _ { K } } , q _ { i } , \langle \mathcal { Q } _ { i - W } , \ldots , \mathcal { Q } _ { i - 1 } \rangle , \mathcal { P } _ { r } \right) .\tag{6}
$$

Here, ${ \tt C i d } ^ { * }$ denotes the concept identifier predicted by the MLLM for the personalized concept referred to in the incoming dialogue $\mathcal { Q } _ { i }$ . The output $\boldsymbol { \mathrm { C i d } } ^ { * }$ can either correspond to an existing concept identifier in the candidate concept set $\mathcal { C } _ { K }$ or indicate a novel concept, which will be initialized with a new identifier. $\mathcal { M } _ { k } ^ { \mathrm { r } }$ represents the recent R memory items associated with concept $c _ { k } ,$ , which reflects its latest attribute state. $q _ { i }$ denotes the current user query in the dialogue turn, and $\langle \mathcal { Q } _ { i - W } , \ldots , \mathcal { Q } _ { i - 1 } \rangle$ denotes the most recent W dialogue turns serving as recent conversational context. $\mathcal { P } _ { r }$ is the prompt that instructs the MLLM to perform concept reconciliation and to output either an existing concept identifier or a novel one. Consequently, the CR skill completes the concept reconciliation.

## 3.3 Plug-and-Play Operation

We take the integration of TAME-O (i.e., the training-free CR Skill) into TAME as an example. Specifically, we retain TAME’s static memory $\mathcal { M } _ { S }$ while replacing its dynamic memory $\mathcal { M } _ { D }$ with TAD Memory $\mathcal { M } _ { T }$ . After obtaining the determined concept identifier ${ \tt C i d } ^ { * }$ in $\mathcal { Q } _ { i }$ in Eq. (6), if the concept is an existing one, the assistant leverages this identifier to locate relevant memory items from both memories, and aggregates the corresponding information for personalized response generation (following the original TAME pipeline). If the concept is determined to be novel, a new identifier will be assigned to this concept and stored in the TAD Memory, therefore unlocking new concept automatically. Notably, when integrated with TAME, we also incorporate information from the static memory for candidate concept retrieval. The final similarity score for each candidate concept is obtained by averaging the similarity scores from the static memory and the TAD Memory.

![](images/8f7f896dfc53e40d15fdeca38a593a92623002195d53bf98ffa30605930dc592.jpg)  
Figure 3: A concept fission example for concept Cat 1.

## 3.4 Discussion of Concept Fission and Its Mitigation

Although the CR Skill effectively resolves novel and existing concepts during ongoing interactions, it may still misidentify an existing concept as a novel one. In such cases, the original concept unit is unintentionally forked into multiple concept units with different identifiers, a phenomenon we refer to as concept fission. This leads to memory chaos, where no single concept unit retains complete contextual information, ultimately resulting in degraded personalization performance. To stably mitigate this issue, we introduce a Periodic Concept Fusion Strategy (PCF Strategy), which is performed every fixed interval of T interaction steps. Specifically, after each interval T, we compute pairwise similarities between all personalized concepts based on their stored memory items, and select the Top-L most similar concept pairs as candidates for fusion:

$$
S ( c _ { m } , c _ { n } ) = \frac { 1 } { \left| \mathcal { M } _ { m } \right| \left| \mathcal { M } _ { n } \right| } \sum _ { i = 1 } ^ { \left| \mathcal { M } _ { m } \right| \left| \mathcal { M } _ { n } \right| } s _ { m , n } ^ { ( i , j ) } ,\tag{7}
$$

$$
s _ { m , n } ^ { ( i , j ) } = \cos ( \phi ( \mathcal { T } _ { m } ^ { ( i ) } ) , \phi ( \mathcal { T } _ { n } ^ { ( j ) } ) ) + \cos ( \phi ( \mathcal { V } _ { m } ^ { ( i ) } ) , \phi ( \mathcal { V } _ { n } ^ { ( j ) } ) ) .\tag{8}
$$

Here, $S ( c _ { m } , c _ { n } )$ is the pairwise similarity between two concepts $c _ { m }$ and $c _ { n } . \ M _ { m }$ and $\mathcal { M } _ { n }$ represent the memory sets associated with concepts $c _ { m }$ and $c _ { n } ,$ respectively. $s _ { m , n } ^ { ( i , j ) }$ is the similarity between the i-th memory item of concept $c _ { m }$ and the j-th memory item of concept $c _ { n }$ , computed using cosine similarity in the embedding space defined by $\phi ( \cdot )$ . The overall concept-level similarity $S ( c _ { m } , c _ { n } )$ is obtained by averaging the similarities over all memory item pairs, yielding a robust measure of semantic closeness between the two concepts. Subsequently, the Top-L concept pairs are derived:

$$
\mathcal { F } = \mathrm { T o p } { - } L \Big \{ ( c _ { m } , c _ { n } ) \big | S ( c _ { m } , c _ { n } ) \Big \} _ { m \neq n } .\tag{9}
$$

Afterward, we further introduce a similarity threshold τ to identify the most potentially forked concept pairs in $\mathcal { F }$ , and leverage the MLLM assistant with prompt $\mathcal { P } _ { f }$ to evaluate each concept pair to perform the fusion and then update the memories. The prompts $\mathcal { P } _ { r }$ and $\mathcal { P } _ { f }$ are presented in Appendix J, while the pseudocode is provided in Appendix A. The complexity analysis of TAME-O is in Appendix B.

## 4 Experiments

We provide a brief version of the experimental settings, with a detailed version in Appendix C.

Benchmarks. Since open-set personalization is inherently intertwined with long-context dialogues, we first adopt an existing long-context personalized MLLM benchmark LCMP-H [4], which consists of 30 personalized concepts (Pet (10), Object (10), and Person (10)), 420 historical dialogue turns, 720 personalized images, and 180 hard questions. In addition, we introduce a new benchmark tailored for long-context open-set personalization, termed LCMP-O. Compared with LCMP-H, LCMP-O poses a more challenging open-set scenario by incorporating a larger number of personalized concepts (45 concepts) with more visually similar appearances. Specifically, its concepts are evenly split into Pet (15), Object (15), and Person (15), where pets and objects are organized into groups of three concepts sharing the same breed or object class, and persons are selected with similar ages and an approximately 1:1 gender ratio. Following LCMP [4], we generate 10 historical dialogue turns for each concept to reveal its long-term attributes and short-term events, as well as $^ { 5 }$ hard questions requiring reasoning over both types of attributes. Overall, LCMP-O contains 450 historical dialogue turns, 675 personalized images, and 225 hard questions, with more details in Section C.2.

Open-Set Personalization Setup. To reflect the open-set setting, we adopt a concept-incremental evaluation protocol. Specifically, personalized concepts are introduced to MLLMs sequentially rather than all at once. For each concept, the MLLM is first exposed to its associated historical dialogues in a temporally sequential manner, one dialogue turn at a time step <sup>3</sup>. After all dialogue turns for the concept have been processed, the MLLM is instantly evaluated on the corresponding QA set tied to that concept. Moreover, in contrast to [4], the concept identifiers of historical dialogues under the open-set setting are not provided to the assistant. Instead, the assistant is required to autonomously and incrementally unlock new concepts.

Baselines. We first compare TAME-O against several state-unaware MLLM personalization baselines, including Backbone (the native MLLM without personalization adaptation), MyVLM [1], Yo’LLaVA [2], RAP [8], PeKit [17], and R2P [7]. Notably, we do not apply the open-set setting to these baselines, as their state-unaware design prevents them from directly supporting unlocking novel concepts automatically across interactions. In addition, we construct a new open-set personalization baseline, TAME<sup>†</sup>, by extending the long-context baseline TAME [4] with a simple retrieval-based concept unlocking and resolution mechanism. Specifically, TAME<sup>†</sup> retrieves candidate concepts from the memory and treats the Top-1 retrieved concept as the matched existing concept, if its similarity exceeds a predefined threshold. Otherwise, the concept is considered novel. We further experiment with TAME<sup>†</sup> w/ MLLM decision (i.e., the w/o TAD Memory variant in Table 2).

Evaluation Metrics. Following LCMP, we employ evaluation metrics: (1) Accuracy of Choice questions (ACC-C), (2) Accuracy of Free-text questions (ACC-F), (3) Scoring Point Rate Long-term attributes (SPR-L), and (4) Scoring Point Rate Short-term attributes (SPR-S), which measure the proportion of responses that explicitly reference the relevant attribute variation of the personalized concept. Moreover, we propose (5) Concept Reconciliation Accuracy (CRA) as a new metric for evaluating the open-set personalization capability. Formally, CRA can be defined as:

$$
\mathrm { C R A } = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i = 1 } ^ { \vert \mathcal { H } \vert } \mathbb { I } \big ( \mathsf { C i d } _ { i } = \mathsf { C i d } _ { i } \big ) ,\tag{10}
$$

where H is the set of historical dialogues, I(·) is the indicator function that returns 1 if the prediction is correct and 0 otherwise, $\mathsf { C i d } _ { i }$ is the concept identifier determined by the MLLM assistant in i-th historical dialogue. $\mathrm { C i d } _ { i }$ denotes the ground-truth concept identifier for the i-th dialogue. If the referred concept is an existing one (already grasped), Cid<sub>i</sub> corresponds to the identifier of that concept. If it is novel, $\mathsf { C i d } _ { i }$ is expected to be a previously unused identifier. If an existing concept has been forked into multiple identifiers (concept fission), any predicted identifier $\mathsf { C i d } _ { i }$ that corresponds to one of the forked identifiers is treated as a correct match. Moreover, we propose the metric (6) Concept Fission Rate (CFR) to measure the fission rate of the concepts at different time steps t:

$$
\mathrm { C F R } ( t ) = \frac { \sum _ { c _ { i } \in \mathcal { C } _ { t } } \operatorname* { m a x } \Bigl ( 0 , | \hat { \mathcal { T } } _ { c _ { i } } ^ { ( t ) } | - 1 \Bigr ) } { \operatorname* { m a x } ( t - | \mathcal { C } _ { t } | , \epsilon ) } ,\tag{11}
$$

where $\mathcal { C } _ { t } = \{ c _ { 1 } , \ldots , c _ { N _ { t } } \}$ denotes the set of ground-truth personalized concepts that exist at time step $t , \hat { \mathcal { T } } _ { c _ { i } } ^ { ( t ) }$ is the set of concept identifiers derived by the MLLM assistant at time step t associated with the ground-truth concept $c _ { i }$ , which contains the correct identifier and may unintentionally include forked identifiers introduced due to concept fission. As a result, CFR(t) is designed to characterize the temporal evolution of concept fission degree during long-term interactions.

Implementation Details. We adopt the MLLM Qwen3.5-9B [6] as our main backbone, while also evaluating TAME-O on another MLLM InternVL3.5-8B [18] for generalizability. Since TAME-O is plug-and-play, we adopt TAME as the base personalized assistant for evaluation. In addition, we also evaluate its scalability on other MLLM personalization methods to unlock their open-set capability (cf. Appendix F). When visually grounding the concept in user’s query, we first prompt the MLLM to extract open-set category descriptions from the query, and then use the extracted categories to perform grounding on the query image with Grounding-DINO [19]. If no concept is grounded, we provide the whole image to the MLLMs for concept reconciliation. In addition, we adopt Qwen3-VL-Embedding-2B [20] to generate embeddings for both visual and textual inputs.

Table 1: Performance comparison on the LCMP-H and LCMP-O datasets. The best results are in black bold. Higher values of all metrics indicate better performance. TAME<sup>†</sup> is the modified TAME.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">MLLM Backbone</td><td rowspan="2">Training Free</td><td rowspan="2">State Aware</td><td rowspan="2">Open</td><td colspan="4">LCMP-H</td><td colspan="4">LCMP-0</td></tr><tr><td>Šet</td><td>ACC-C ACC-F SPR-L SPR-S CRA</td><td></td><td></td><td>ACC-C ACC-F SPR-L SPR-S CRA</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Backbone</td><td>InternVL3.5-8B</td><td rowspan="2">√</td><td rowspan="2">x</td><td rowspan="2">x</td><td rowspan="2">40.00</td><td>20.00 12.78</td><td>26.11</td><td></td><td>32.00 18.67</td><td>16.44</td><td>7.11</td><td></td></tr><tr><td></td><td>36.11 20.56</td><td>17.22 32.78</td><td>38.67</td><td>22.22</td><td>29.78</td><td>14.67</td><td></td></tr><tr><td>MyVLM</td><td>Qwen3.5-9B LLaVA-v1.6-7B</td><td rowspan="2">x</td><td rowspan="2">-x</td><td rowspan="2">x</td><td>40.00</td><td>13.33 8.89</td><td>12.22</td><td>46.67</td><td>21.33</td><td>18.22</td><td>6.67</td><td></td></tr><tr><td>LLaVA-v1.5-13B</td><td>x x</td><td>x</td><td>38.89</td><td>9.44 8.89</td><td>17.22</td><td>56.44 6.67</td><td>20.89</td><td>9.78</td><td></td></tr><tr><td>Yo&#x27;LLaVA</td><td>LLaVA-v1.5-13B</td><td rowspan="2">x</td><td rowspan="2">x</td><td rowspan="2">x</td><td>41.11</td><td>4.44</td><td>33.33 23.89</td><td>34.22</td><td>20.89</td><td>19.1ī</td><td>17.33</td><td></td></tr><tr><td>RAP</td><td></td><td></td><td>46.67</td><td>9.44 15.00</td><td>20.00</td><td>22.22</td><td>28.89</td><td></td><td></td></tr><tr><td rowspan="2">PeKit</td><td rowspan="2">Phi3-V-3.8B InternVL3.5-8B</td><td rowspan="2">√</td><td rowspan="2">x</td><td rowspan="2">x</td><td>61.11</td><td>49.44 21.11</td><td></td><td>51.56 52.89</td><td></td><td></td><td>15.11</td><td></td></tr><tr><td></td><td>68.33 56.67</td><td></td><td>35.00</td><td>40.89</td><td>29.78</td><td>20.00 10.22</td><td></td></tr><tr><td></td><td>Qwen3.5-9B InternVL3.5-8B</td><td rowspan="2">√</td><td rowspan="2">一 x</td><td rowspan="2">x</td><td>52.78</td><td>35.56</td><td>35.56 48.33</td><td></td><td>56.00 65.78 40.89</td><td>33.78</td><td>19.56</td><td></td></tr><tr><td></td><td></td><td>66.67</td><td>55.56</td><td>8.89</td><td>25.00</td><td></td><td>10.22</td><td>26.22</td><td></td></tr><tr><td>R2P</td><td>Qwen3.5-9B InternVL3.5-8B</td><td rowspan="2"></td><td rowspan="2">√</td><td rowspan="2"></td><td>53.89</td><td>39.44</td><td>45.00 48.33 22.78 42.78</td><td>33.08</td><td>60.89 41.78 58.22 19.11</td><td>20.00 34.67</td><td>36.44</td><td></td></tr><tr><td>TAME†</td><td>√</td><td>√ 65.56</td><td>34.44</td><td></td><td></td><td></td><td></td><td>15.11</td><td>28.73</td></tr><tr><td rowspan="2"></td><td rowspan="2">Qwen3.5-9B InternVL3.5-8B</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">√</td><td></td><td></td><td>22.22 42.22</td><td>34.59</td><td>49.33 32.44</td><td>50.67</td><td>27.11</td><td>31.63</td></tr><tr><td></td><td>75.00 61.67</td><td>51.11</td><td>57.78~69.17</td><td>70.67</td><td>38.22</td><td>56.44</td><td>51.11 66.37</td></tr><tr><td>TAME-O</td><td>Qwen3.5-9B</td><td></td><td>√</td><td></td><td>82.22</td><td>61.67</td><td>64.44</td><td>70.00 72.18</td><td>75.11 52.89</td><td>72.00</td><td></td><td>59.11 71.49</td></tr></table>

![](images/c3cfc64fc505b46157f5effcb8fe9f914c805fbb3747c9ebb88a4159daf20442.jpg)  
(a) LCMP-H.

![](images/fad49918e5e0e6d149be09638e905d0658e79b509c4359c4ace2bb5e2f4ce16c.jpg)  
(b) LCMP-O.  
Figure 4: Performance comparison between TAME<sup>†</sup> and TAME-O with respect to a continually increasing number of concepts. ■ denotes TAME<sup>†</sup>, while ▲ denotes TAME-O.

Table 2: Ablation study on core components in TAME-O with Qwen3.5-9B.
<table><tr><td rowspan="2">Module</td><td rowspan="2">Variant</td><td colspan="4">LCMP-H</td><td colspan="5">LCMP-O</td></tr><tr><td>ACC-C</td><td>ACC-F</td><td>SPR-L</td><td>SPR-S</td><td>CRA</td><td>ACC-C</td><td>ACC-F</td><td>SPR-L</td><td>SPR-S</td><td>CRA</td></tr><tr><td rowspan="2">CR Skill</td><td>w/o TAD Memory</td><td>77.78</td><td>57.22</td><td>60.56</td><td>68.89</td><td>69.67</td><td>70.67</td><td>48.44</td><td>68.44</td><td>53.33</td><td>66.59</td></tr><tr><td>w/o MLLM Decision</td><td>72.78</td><td>48.89</td><td>40.00</td><td>52.22</td><td>51.13</td><td>46.22</td><td>37.78</td><td>52.44</td><td>29.33</td><td>46.33</td></tr><tr><td rowspan="2">PCF Strategy</td><td>w/o CR Skill</td><td>66.11</td><td>38.33</td><td>24.44</td><td>45.56</td><td>36.84</td><td>52.44</td><td>34.67</td><td>50.67</td><td>29.33</td><td>33.41</td></tr><tr><td>w/o PCF Strategy</td><td>78.89</td><td>59.44</td><td>61.11</td><td>67.22</td><td>69.92</td><td>73.78</td><td>47.56</td><td>70.22</td><td>54.67</td><td>68.15</td></tr><tr><td>TAME-O</td><td>ALL</td><td>82.22</td><td>61.67</td><td>64.44</td><td>70.00</td><td>72.18</td><td>75.11</td><td>52.89</td><td>72.00</td><td>59.11</td><td>71.49</td></tr></table>

## 4.1 Open-Set Personalization Performance

From the results in Table 1, we observe that: (O1) Training-based methods, such as MyVLM and Yo’LLaVA, perform poorly on both benchmarks, as encoding long-context personalized information with a limited number of lightweight parameters is insufficient to capture extended interaction histories. In contrast, RAG-based, training-free methods such as R2P and PeKit achieve stronger performance by storing personalized concepts in external memory while keeping model parameters fixed. However, all these baselines operate under a closed-set assumption and lack the ability to automatically incorporate novel personalized concepts. (O2) TAME is the SOTA baseline for long-context personalization. However, under the open-set setting, TAME<sup>†</sup> exhibits suboptimal performance, since simply treating all historical attributes equally fails to identify appropriate matched concepts. Moreover, the concept fission problem further undermines the effectiveness of TAME<sup>†</sup>. In contrast, TAME-O achieves the best performance for two reasons: the CR Skill accurately reconciles incoming personalized concepts by treating the stored concept attributes in a temporally weighted manner, emphasizing the most recently observed concept information. Moreover, the PCF Strategy further stably fuses fissioned concepts, mitigating the concept fission problem and enabling the better localization of sufficient contextual information for each concept. Furthermore, Figure 4 shows that TAME-O maintains stable and effective open-set personalization performance as the total number of concepts continually increases.

![](images/3ed66afbe63a6a745cd6665069827017716dc6f2b95a911f1afa23185965c49b.jpg)  
(a) LCMP-H.

![](images/69bc239e4ea51a01067ec55f5a8f1317243018e50e8bd3c97b007bc5baa27746.jpg)  
(b) LCMP-O.

![](images/97020104d4f87d4e2d93b7212b5a11f8d1b5c4a52ffb7db4cb110b4dccfc5b5f.jpg)  
(c) LCMP-H.

![](images/4e56531828342f76612e3e29aba880aa3d556fcdbe0b985defb1672f412ce65f.jpg)  
(d) LCMP-O.

Figure 5: Comparison of CFR among TAME<sup>†</sup>, TAME-O w/o PCF Strategy, and TAME-O.  
![](images/9207b027866de3bf15379da32ecf24f571618a2732ee20ce951a0394f0c881e0.jpg)  
(a) Reconciliation accuracy.

![](images/5cc0951e87a3f9d02b810610bdfc808892d9787db61121f83056d9163c4d31cd.jpg)  
(b) QA accuracy.

![](images/34084f6e15a67f4c0b683bec2a80b6e20c2308be0493c8494faff63e2aa3b30c.jpg)  
(c) Scoring point rate.  
Figure 6: Performance on LCMP-O with concepts stratified by confusability. ACC and SPR average ACC-C/ACC-F and SPR-L/SPR-S, respectively. ■ denotes TAME<sup>†</sup>, while ▲ denotes TAME-O.

## 4.2 Further Analysis

Ablative Analysis. We conduct comprehensive ablations on the core components within TAME-O, with results in Table 2. For CR Skill, we evaluate the variants: (1) w/o TAD Memory, where the Time-Aware Dynamic (TAD) Memory is reduced to a static memory and all temporal markers associated with concept attributes are removed; (2) w/o MLLM Decision, where the Top-1 retrieved concept is directly treated as the matched concept if its similarity score exceeds a predefined threshold, and otherwise regarded as novel; (3) w/o CR Skill, where both the TAD Memory and the MLLMbased decision mechanism are removed, and a simple time-unaware retrieval strategy is applied, with the Top-1 retrieved concept always selected as the matched existing concept if it exceeds a predefined threshold. We observe that variants (1)–(3) exhibit clear performance drops, indicating that the accurate reconciliation of novel and existing user-specific concepts is vital for open-set personalization. For PCF Strategy, we entirely remove it to design a variant: w/o PCF Strategy. We observe that this variant incurs a clear performance drop, as the degree of concept fission becomes increasingly severe over time without any mitigation, leading to memory chaos.

Concept Fission and Fusion Analysis. We quantitatively evaluate the severity of the concept fission and how well TAME-O mitigates the fission problem. Specifically, we report the Concept Fission Rate (CFR), as defined in Eq. (11), over time steps on the historical dialogues of LCMP-H and LCMP-O. From Figure 5(a)–(b), we observe that the CFR of TAME<sup>†</sup> rapidly reaches 47.8% and 73.3% on LCMP-H and LCMP-O and then remains high, while that of TAME-O w/o PCF Strategy gradually increases. Nevertheless, TAME-O w/o PCF Strategy has a much lower CFR than TAME<sup>†</sup>, since the CR skill mitigates concept fission at its source by more accurately reconciling incoming open-set personalized concepts with existing ones. From Figure 5(c)–(d), we observe that with the PCF strategy equipped, the CFR never exceeds that without it and finally drops from 17.8% to 13.2% (LCMP-H) and from 22.0% to 13.8% (LCMP-O), indicating that the PCF strategy effectively fuses potentially forked concepts. Moreover, we empirically observe that the PCF Strategy performs fusion with relatively high accuracy, achieving a strong fusion success rate (cf. Appendix E for details).

Robustness to Concept Confusability. Since visually similar personalized concepts are where open-set reconciliation is most likely to fail, we further break down the performance on LCMP-O by concept confusability. Specifically, we measure the confusability of each concept by its nearestneighbour similarity to the other concepts, computed with the same encoder and text-plus-visual similarity score as the PCF Strategy (Eq. (7)), and split the 45 concepts into three equal-sized tertiles (Low, Mid, and High). As shown in Figure 6, TAME-O outperforms TAME<sup>†</sup> on every metric in every similarity bin, with CRA gains of 32.9–52.2 points. Moreover, TAME-O maintains relatively stable performance as concept confusability increases, keeping its CRA above 60 even in the most confusable bin. In contrast, the CRA of TAME<sup>†</sup> already drops sharply at medium similarity, as its threshold-based Top-1 matching is easily misled by similar concepts.

![](images/2be3d34bbbb03efda4df0001df4226535aa3f0cb797da923ae3dd50ba3800077.jpg)  
(a) LCMP-H.

![](images/4b89868da4d5f2f77ea335b510f793214a5dc768e53e9d73aacfe1e492addd43.jpg)  
(b) LCMP-O.

Figure 7: Performance (mean ± std) across five concept appearance orders. Numbers above the bars are the gains over TAME<sup>†</sup>. denotes TAME<sup>†</sup>, while denotes TAME-O.  
![](images/103d03b3852a0253be6dde6a96d91530db71d542bf7539f63ee550b21ee56047.jpg)  
Figure 8: Case study on resolving an existing concept (left) and a novel concept (right).

Stability across Concept Appearance Orders. Under the open-set setup, personalized concepts emerge sequentially. As a result, different appearance orders lead to different sets of previously acquired concepts in memory for the same incoming concept. This, in turn, affects the retrieved candidate concepts and the difficulty of determining whether the incoming concept is existing or novel. Here, we report the mean performance and variability of TAME-O and TAME<sup>†</sup> across five concept appearance orders in Figure 7. We observe that TAME-O maintains stable performance across different concept appearance orders and consistently outperforms TAME<sup>†</sup>.

Case Study: Old and Novel Concept Reconciliation. In Figure 8, we visualize two cases to showcase the procedure of TAME-O in tackling the open-set personalization problem. On the top, the incoming concept is correctly identified by the CR skill as an existing one (Dog 2), after which the relevant memory items are retrieved for the personalized response. On the bottom, the CR skill correctly identifies the incoming concept as novel, responds without retrieving existing memories, and unlocks this new concept with an initialized identifier Bunny 1 in the TAD Memory.

## 5 Conclusion

In this study, we pioneered the research in long-context MLLM personalization into the open-set setting. To this end, we proposed TAME-O, which automatically tames novel personalized concepts from everyday interactions while resolving potential conflicts with previously memorized ones. Extensive experiments in the open-set setting on LCMP-H and LCMP-O benchmarks demonstrate that TAME-O effectively integrates incoming personalized concepts even under visually similar concepts, mitigates concept fission, and remains robust to varying concept arrival orders.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Natural Science Foundation of China (Grant No. 62572097). This work was also supported by the Center for HPC, University of Electronic Science and Technology of China. The authors declare no competing interests.

## References

[1] Yuval Alaluf, Elad Richardson, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Myvlm: Personalizing vlms for user-specific queries. In European Conference on Computer Vision (ECCV), pages 73–91. Springer, 2024.

[2] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’LLaVA: Your Personalized Language and Vision Assistant. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[3] Jaeik Kim, Woojin Kim, Woohyeon Park, and Jaeyoung Do. Mmpb: It’s time for multi-modal personalization. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[4] Rongpei Hong, Jian Lang, Yong Wang, Ting Zhong, and Fan Zhou. Tameing long contexts in personalization: Towards training-free and state-aware mllm personalized assistant. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2026.

[5] OpenAI. Gpt-5 system card. https://openai.com/index/gpt-5-system-card, August 2025.

[6] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[7] Deepayan Das, Davide Talon, Yiming Wang, Massimiliano Mancini, and Elisa Ricci. Trainingfree personalization via retrieval and reasoning on fingerprints. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[8] Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, and Xiangyu Yue. RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025.

[9] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[10] Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, et al. UniCTokens: Boosting personalized understanding and generation via unified concept tokens. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[11] Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, and Yuheng Li. Yo’chameleon: Personalized vision and language generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR), pages 14438–14448, 2025.

[12] Chang Nie, Chaoyou Fu, Yifan Zhang, Haihua Yang, and Caifeng Shan. Personavlm: Long-term personalized multimodal llms. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR), 2026.

[13] Ruichuan An, Sihan Yang, Renrui Zhang, Ming Lu, Tianyi Jiang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, Shanghang Zhang, and Wentao Zhang. MC-LLaVA: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024.

[14] Ruichuan An, Kai Zeng, Ming Lu, Sihan Yang, Renrui Zhang, Huitong Ji, Hao Liang, and Wentao Zhang. Concept-as-Tree: A controllable synthetic data framework makes stronger personalized VLMs. In European Conference on Computer Vision (ECCV), 2026.

[15] Renjie Pi, Jianshu Zhang, Tianyang Han, Jipeng Zhang, Rui Pan, and Tong Zhang. Personalized Visual Instruction Tuning. In International Conference on Learning Representations (ICLR), 2025.

[16] Chau Pham, Hoang Phan, David Doermann, and Yunjie Tian. PLVM: A tuning-free approach for personalized large vision-language model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3671–3680, 2025.

[17] Soroush Seifi, Vaggelis Dorovatas, Matteo Cassinelli, Fabien Despinoy, Daniel Olmeda Reino, and Rahaf Aljundi. Personalization Toolkit: Training Free Personalization of Large Vision Language Models. Transactions on Machine Learning Research (TMLR), 2026.

[18] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, et al. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv preprint arXiv:2508.18265, 2025.

[19] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. In European Conference on Computer Vision (ECCV), pages 38–55, 2024.

[20] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[21] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 3498–3505, 2012.

[22] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In European Conference on Computer Vision (ECCV), pages 740–755, 2014.

[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS), 36:34892–34916, 2023.

[24] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm Agents Are Experiential Learners. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 19632–19642, 2024.

[25] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A Survey on the Memory Mechanism of Large Language Model-based Agents. ACM Transactions on Information Systems, 43(6):1–47, 2025.

[26] Zeyuan Liu, Jeonghye Kim, Xufang Luo, Dongsheng Li, and Yuqing Yang. Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization. In International Conference on Learning Representations (ICLR), 2026.

[27] A. Maharana, Dong-Ho Lee, S. Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 13851–13870, 2024.

[28] Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang. Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory. arXiv preprint arXiv:2311.08719, 2023.

[29] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing Large Language Models with Long-Term Memory. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 19724–19731, 2024.

[30] Gustavo Penha, Ali Vardasbi, Enrico Palumbo, Marco De Nadai, and Hugues Bouchard. Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other? In ACM Conference on Recommender Systems (RecSys), pages 340–349. ACM, 2024.

[31] Mufan Xu, Gewen Liang, Kehai Chen, Wei Wang, Xun Zhou, Muyun Yang, Tiejun Zhao, and Min Zhang. Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning. In Findings of the Association for Computational Linguistics (ACL), pages 24068– 24084, 2025.

[32] Yibo Wang, Yong Rui, and Min-Ling Zhang. Classifier-induced Reciprocal Points for Multilabel Open-set Recognition. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 26643–26651, 2026.

[33] Yimin Fu, Zhunga Liu, and Jialin Lyu. Reason and Discovery: A New Paradigm for Open Set Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 47(7):5586–5599, 2025.

[34] Walter J Scheirer, Anderson de Rezende Rocha, Archana Sapkota, and Terrance E Boult. Toward open set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 35(7):1757–1772, 2012.

[35] Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Learning placeholders for open-set recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 4401–4410, 2021.

[36] Baile Xu, Furao Shen, and Jian Zhao. Contrastive open set recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), volume 37, pages 10546–10556, 2023.

[37] Jun-Yi Hang and Min-Ling Zhang. Binary decomposition: A problem transformation perspective for open-set semi-supervised learning. In International Conference on Machine Learning (ICML), 2024.

[38] Poojan Oza and Vishal M Patel. C2ae: Class conditioned auto-encoder for open-set recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 2307–2316, 2019.

[39] Xin Sun, Zhenning Yang, Chi Zhang, Keck-Voon Ling, and Guohao Peng. Conditional gaussian distribution learning for open set recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 13480–13489, 2020.

[40] Yunrui Guo, Guglielmo Camporese, Wenjing Yang, Alessandro Sperduti, and Lamberto Ballan. Conditional variational capsule network for open set recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 103–111, 2021.

[41] Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. Taskbench: Benchmarking large language models for task automation. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 4540–4574, 2024.

[42] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology (TIST), 15(3):1–45, 2024.

[43] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning. In International Conference on Machine Learning (ICML), 2025.

[44] Zheng Lian, Haoyu Chen, Lan Chen, Haiyang Sun, Licai Sun, Yong Ren, Zebang Cheng, Bin Liu, Rui Liu, Xiaojiang Peng, Jiangyan Yi, and Jianhua Tao. AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models. In International Conference on Machine Learning (ICML), 2025.

[45] Mingi Jung, Saehyung Lee, Eunji Kim, and Sungroh Yoon. Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models. In International Conference on Machine Learning (ICML), 2025.

[46] Tzu-Tao Chang and Shivaram Venkataraman. LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models. In International Conference on Machine Learning (ICML), 2025.

[47] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models. In International Conference on Learning Representations (ICLR), 2025.

[48] Chia-Hao Kao, Cheng Chien, Yu-Jen Tseng, Yi-Hsin Chen, Alessandro Gnutti, Shao-Yuan Lo, Wen-Hsiao Peng, and Riccardo Leonardi. Bridging Compressed Image Latents and Multimodal Large Language Models. In International Conference on Learning Representations (ICLR), 2024.

[49] Tianle Gu, Zeyang Zhou, Kexin Huang, Dandan Liang, Yixu Wang, Haiquan Zhao, Yuanqi Yao, Xingge Qiao, Keqing Wang, Yujiu Yang, Yan Teng, Yu Qiao, and Yingchun Wang. Mllmguard: A Multi-dimensional Safety Evaluation Suite for Multimodal Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[50] Ziqiang Liu, Feiteng Fang, Xi Feng, Xeron Du, Chenhao Zhang, Noah Wang, Yuelin Bai, Qixuan Zhao, Liyang Fan, Chengguang Gan, Hongquan Lin, Jiaming Li, Yuansheng Ni, Haihong Wu, Yaswanth Narsupalli, Zhigang Zheng, Chengming Li, Xiping Hu, Ruifeng Xu, Xiaojun Chen, Min Yang, Jiaheng Liu, Ruibo Liu, Wenhao Huang, Ge Zhang, and Shiwen Ni. Ii-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[51] Jian Lang, Rongpei Hong, Ting Zhong, Leiting Chen, Qiang Gao, and Fan Zhou. From shallow humor to metaphor: Towards label-free harmful meme detection via lmm agent selfimprovement. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 588–599, 2026.

[52] FENGYUN WANG, Sicheng Yu, Jiawei Wu, Jinhui Tang, Hanwang Zhang, and Qianru Sun. 3d question answering via only 2d vision-language models. In International Conference on Machine Learning (ICML), 2025.

[53] Kangyu Zhu, Peng Xia, Yun Li, Hongtu Zhu, Sheng Wang, and Huaxiu Yao. Mmedpo: Aligning medical vision-language models with clinical-aware multimodal preference optimization. In International Conference on Machine Learning (ICML), 2025.

[54] Haochen Liu, Li Chen, Yu Qiao, Chen Lv, and Hongyang Li. Reasoning multi-agent behavioral topology for interactive autonomous driving. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 92605–92637, 2024.

[55] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. Lamp: When large language models meet personalization. In Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 7370–7392, 2024.

[56] Lucie Charlotte Magister, Katherine Metcalf, Yizhe Zhang, and Maartje ter Hoeve. On the Way to LLM Personalization: Learning to Remember User Conversations. In Proceedings of the First Workshop on Large Language Model Memorization (L2M2), pages 61–77. Association for Computational Linguistics, 2025.

[57] Saleh Afzoon, Usman Naseem, Amin Beheshti, and Zahra Jamali. PersoBench: Benchmarking Personalized Response Generation in Large Language Models. arXiv preprint arXiv:2410.03198, 2024.

[58] Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. Personalizing Dialogue Agents: I have a dog, do you have pets too? In Annual Meeting ofthe Association for Computational Linguistics (ACL), pages 2204–2213, 2018.

[59] Yoonna Jang, Jungwoo Lim, Yuna Hur, Dongsuk Oh, Suhyune Son, Yeonsoo Lee, Dong-Hoon Shin, Seungryong Kim, and Heuiseok Lim. Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), pages 10803–10812, 2022.

[60] Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, C. J. Taylor, and Dan Roth. Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. In Conference on Language Modeling (COLM), 2025.

[61] Dongshuo Liu, Zhijing Wu, Dandan Song, and Heyan Huang. A Persona-Aware LLM-Enhanced Framework for Multi-Session Personalized Dialogue Generation. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 103–123, 2025.

## Appendix

## Content

A Pseudocode of TAME-O 17   
B Complexity Analysis of TAME-O 17   
B.1 Complexity of CR Skill . 17   
B.2 Complexity of PCF Strategy 17   
C Detailed Experimental Settings 18   
C.1 Benchmark: LCMP-H 18   
C.2 Benchmark: LCMP-O 19   
C.3 Baselines 19   
C.4 Implementation Details . 20   
D Hyper-Parameter Analysis 20   
E Quantitative Analysis of the PCF Strategy 20   
F Scalability across Other MLLM Personalized Assistants 21   
G Additional Literature Review 21   
G.1 Memory-Augmented LLMs . 21   
G.2 Open-Set Recognition . 22   
G.3 Multimodal Large Language Models . 22   
G.4 Large Language Model Personalization 23   
H Additional Case Study 23   
H.1 Additional Cases of Concept Reconciliation 23   
H.2 Cases of Concept Fusion 23   
I Failure Case Analysis: Concept Fission 23   
J Prompt Presentation for TAME-O 24   
K Limitations 24   
L Broader Impacts 24   
M License of Assets 25

Algorithm 1 Python-like Pseudocode for TAME-O   
# Q\_i: dialogue at time step i (q\_i, a\_i)   
# q\_i: current user query (before generation)   
# M\_T: Time-Aware Dynamic Memory (Cid -> [(Attr, t)])   
# C: personalized concept set   
# K: number of candidate concepts   
# W: context window size   
# T: fusion interval   
# initialize memory and concept set   
M\_T = {}   
C = set()   
for i, Q\_i in enumerate(dialogue\_stream):   
Encode user query   
T\_i, V\_i = Encode(q\_i)   
Concept Reconcile (CR Skill)   
scores = {}   
for c\_k in C:   
# temporally weighted retrieval using timestamps t\_k^(j), each item   
is (Attr, t)   
scores[c\_k] = TemporalWeightedSimilarity(T\_i, V\_i, M\_T[c\_k])   
# select Top-K candidate concepts   
C\_K = TopK(scores, K)   
# MLLM-based concept reconciliation (Eq. 5)   
Cid\* = MLLM\_Reconcile(recent\_memory(C\_K, M\_T), q\_i, recent\_dialogues(W))   
# decide whether concept is novel or existing   
if Cid\* == "novel":   
Cid = NewConceptID()   
C.add(Cid)   
M\_T[Cid] = [] # initialize memory list   
else:   
Cid = Cid\*   
# Personalized Response Generation   
# retrieve relevant memory items using resolved Cid   
context = RetrieveMemory(M\_T, Cid)   
a\_i = MLLM\_Generate(q\_i, context)   
# - Update Time-Aware Dynamic Memory   
# delta indicates add / modify of an attribute   
Cid, Attr, delta = MLLM\_Update(Cid, Q\_i, M\_T)   
# update temporal marker for this concept   
t\_new = GetNextTimestamp(M\_T[Cid])   
M\_T = MemoryUpdate(M\_T, Cid, Attr, delta, t\_new)   
# - Periodic Concept Fusion (PCF Strategy)   
if i % T == 0:   
# compute pairwise concept similarity   
concept\_pairs = PairwiseSimilarity(M\_T)   
fusion\_pairs = TopL(concept\_pairs)   
for (c\_m, c\_n) in fusion\_pairs:   
M\_T = MLLM\_Fuse(c\_m, c\_n, M\_T, \tau)

## A Pseudocode of TAME-O

A concise version of Python-style pseudocode for TAME-O is offered in Algorithm 1.

## B Complexity Analysis of TAME-O

We delve into the complexity analysis of TAME-O. We first provide a definition of the complexity of each MLLM invocation:

Definition B.1 (Complexity of an MLLM Invocation) We consider a single invocation of an MLLM consisting of H multimodal Transformer layers, each with hidden dimension d, operating on an input sequence X with length l, the self-attention operation is defined as:

$$
{ \mathrm { A t t e n t i o n } } ( X ) = \operatorname { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V ,\tag{12}
$$

where $Q = X W _ { Q } , K = X W _ { K }$ , and $V = X W _ { V }$ , with $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { d \times d }$ . For simplicity, we assume standard multi-head self-attention with shared sequence length for queries, keys, and values. Under this setting, the computational complexity ofone MLLM invocation is: $O \big ( H \cdot ( l d ^ { 2 } + l ^ { 2 } d ) \big )$ ) , where $l d ^ { 2 }$ corresponds to the linear projectionsfor query, key, and value representations, and $l ^ { 2 } d$ arisesfrom the self-attention computation. However, H is often a small constant, and we therefore simplify the complexity to $O ( l d ^ { 2 } + l ^ { 2 } d )$ in thefollowing discussion.

We then analyze the computational complexity of TAME-O by decomposing it into two major components: (1) the Concept Reconciliation Skill (CR Skill), and (2) the Periodic Concept Fusion Strategy (PCF Strategy).

## B.1 Complexity of CR Skill

The CR Skill consists of two stages: temporally weighted candidate retrieval and MLLM-based concept resolution.

Temporally Weighted Retrieval. At time step $i ,$ the assistant maintains a dynamic concept set $\mathcal { C } _ { i - 1 }$ Each concept $c _ { k }$ is associated with $N _ { k }$ memory items stored in the TAD Memory. For an incoming query $q _ { i } ,$ , the CR Skill computes the temporally weighted similarity score $S ( c _ { k } \mid q _ { i } )$ by comparing the query representation with all memory items of $c _ { k }$ . The cost of computing a single cosine similarity in the CLIP embedding space is $O ( \bar { d } )$ . Therefore, the total cost of temporally weighted retrieval over all concepts is: $O \Big ( \sum _ { k = 1 } ^ { | \mathcal { C } _ { i - 1 } | } N _ { k } \cdot d \Big ) = O ( | \mathcal { M } _ { T } | \cdot d )$ , where $\vert \mathcal { M } _ { T } \vert$ denotes the total number of memory items in the TAD Memory. In practice, $N _ { k }$ is bounded to a constant, making this step linear in memory size.

MLLM-based Concept Reconciliation. After retrieval, only the Top-K candidate concepts are retained, where $K \ll | { \mathcal { C } } _ { i - 1 } |$ . The CR Skill then jointly feeds the most recent memory items of these candidate concepts, together with the current user query and a fixed window of recent dialogue turns, into the MLLM for final concept resolution. Importantly, this reconciliation step is performed via a single MLLM invocation, rather than multiple independent calls per candidate. Since the number of candidate concepts $K$ , the number of memory items per candidate, and the dialogue context window size are all strictly bounded, the total input sequence length to the MLLM is also bounded. Consequently, the MLLM-based concept resolution involves a bounded-length attention computation. According to the attention complexity, this step incurs a cost of $O ( l d ^ { 2 } + \ l ^ { 2 } d )$ , where l is the total token length of the fixed prompt, candidate memory items, and recent dialogue context. As l is independent of the total number of discovered concepts $| { \mathcal { C } } _ { i - 1 } |$ , this step introduces a bounded computational overhead.

## B.2 Complexity of PCF Strategy

The PCF Strategy is executed periodically every T interaction steps, rather than at every step. At each execution, the strategy computes pairwise similarities between personalized concepts to identify potentially forked ones.

![](images/bfbb96d3216a810b0bcf09bdf836fff7be63956c1b4724a3965ae461453000e4.jpg)  
Figure 9: Cases of personalized concepts from LCMP benchmark.

For each concept pair $\left( { { c _ { m } } , { c _ { n } } } \right)$ , the similarity $S ( c _ { m } , c _ { n } )$ is computed by averaging cosine similarities over all pairs of memory items from $\mathcal { M } _ { m }$ and $\mathcal { M } _ { n }$ . The cost of computing the similarity for one concept pair is: $O ( | { \mathcal { M } } _ { m } | \cdot | { \mathcal { M } } _ { n } | \cdot d )$ . Subsequently, computing similarities for all concept pairs yields: $\begin{array} { r } { O \Big ( \sum _ { ( c _ { m } , c _ { n } ) \in \mathcal { C } \times \mathcal { C } , m \neq n } | \mathcal { M } _ { m } | \cdot | \mathcal { M } _ { n } | \cdot d \Big ) } \end{array}$ . For each selected pair, the actual fusion decision is delegated to the MLLM. Similar to the CR Skill, this involves a bounded-length attention computation and incurs a constant overhead. Moreover, since we restrict the number of candidate pairs for concept fusion to the Top-L most similar ones that also exceed a predefined similarity threshold, the overall computational complexity of this step is strictly bounded. In practice, we observe that TAME-O incurs about 6.5 seconds runtime cost per user-MLLM interaction, which is acceptable. And baseline TAME<sup>†</sup> requires about 5 seconds per interaction.

## C Detailed Experimental Settings

## C.1 Benchmark: LCMP-H

Since open-set personalization is studied in long-context, real-world human–MLLM interactions, rather than in static and overly simplified settings (e.g., personalized image captioning), we first adopt the hard version of the LCMP benchmark (Long-Context MLLM Personalization) [4], a recent benchmark and the only available benchmark specifically designed for long-context MLLM personalization. Below we propose a detailed description of this benchmark.

LCMP is the first benchmark for evaluating personalized multimodal large language models in long-context interactions. It consists of 30 personalized concepts, including Pet (10), Object (10), and Person (10), along with approximately 420 historical dialogue turns, 240 easy questions, 180 hard questions, and 720 personalized images, and includes two evaluation sets: (1) LCMP-E, which contains easy VQA questions involving a single type of attribute (either short-term or long-term), and (2) LCMP-H, which contains hard questions requiring reasoning over both types of attributes. It measures an assistant’s ability to perceive and model variations in both short-term and long-term attributes of user-specific concepts and to generate contextually appropriate responses based on extended dialogue histories. LCMP is constructed through three stages: (1) personalized concept creation, where raw images are sampled and endowed with fine-grained temporal attributes; (2) multi turn vision–language dialogue generation, which simulates historical interactions that reveal concept attribute changes over time; and (3) evaluation question generation, which produces single-turn VQA tasks requiring reasoning over the preceding dialogue context. The benchmark thus explicitly tests long-context reasoning and personalization beyond static single-turn tasks. We adopt the hard version of the LCMP benchmark, namely LCMP-H, in this study for evaluation. As illustrated in Figure 9, we provide cases of personalized concepts from LCMP-H.

## C.2 Benchmark: LCMP-O

To comprehensively evaluate the open-set MLLM personalization capability of the MLLM assistant, we construct a new benchmark: Open-Set Long-Context MLLM Personalization (LCMP-O). The original LCMP benchmark exhibits two limitations in evaluating open-set personalization performance: (1) LCMP contains a small number of personalized concepts, which constrains the diversity of the open-set evaluation. (2) The personalized concepts in LCMP are characterized by highly distinctive attributes, making it relatively easy for MLLMs to differentiate new concepts from previously grasped ones, softening the challenges of open-set personalization.

Taking the above limitations into consideration, we construct LCMP-O with 45 personalized concepts, deliberately designed to present high intra-class visual similarity and thus pose stronger challenges for open-set personalization evaluation. Following the construction pipeline of LCMP [4], we generate 10 historical dialogue turns and 5 hard evaluation questions per concept, yielding a total of 450 dialogues and 225 evaluation questions.

The 45 concepts span three types, each chosen to maximize within-type visual ambiguity: (1) 15 Pets (cats, dogs, and rabbits), where every three concepts belong to the same breed, making fine-grained visual disambiguation particularly challenging. These cats and dogs concepts are sourced from the Oxford-IIIT-Pet dataset [21], which covers 37 breeds with substantial intra-breed variation in pose, lighting, and deformation. And the rabbits concepts are collected by authors. (2) 15 Persons, sampled from subjects of similar age with an approximately 1:1 gender ratio. Person images are drawn from MS-COCO [22], following the same protocol as TAME [4]. (3) 15 Objects, where every three concepts belong to the same object class. These concepts are sourced from PerVA [17], which contains object concepts that exhibit high visual and semantic similarity. In Figure 13, we provide cases of concepts from the LCMP-O benchmark.

## C.3 Baselines

Since open-set MLLM personalization is novel as a setting, we compare TAME-O with traditional closed-set MLLM personalization baselines and a specifically designed baseline for the open-set setting. For traditional baselines, we select MyVLM [1], Yo’LLaVA [2], RAP [8], PeKit [17], and R2P [7], with detailed descriptions as follows:

• MyVLM pioneers the personalization of general VLMs so they can understand and talk about concepts unique to a given user. It augments LLaVA [23] with external concept heads that first detect a target entity and then inject a learned concept embedding into the model’s mid-level features, guiding the decoder to mention the concept naturally.

• Yo’LLaVA frames subject personalization as learning a compact representation from only a few example images. It encodes the subject into a small set of latent prefix tokens that sit between the vision encoder and LLaVA’s language decoder, binding visual attributes to text without touching backbone weights.

• RAP turns a generic MLLM into a user-aware assistant by adding an external memory layer. A key–value store “remembers” user facts and portraits, a multimodal retriever selects the most relevant keys at inference, and the generator conditions on both the query and retrieved context to tailor its answer.

• PeKit argues that strong personalization can be achieved entirely at inference time. It pairs offthe-shelf visual descriptors with retrieval-augmented generation and lightweight visual prompts, injecting the correct identity or object instance into any MLLM decoder without gradient updates.

• R2P shows that an MLLM can personalize itself using nothing but its pre-trained weights and a clever retrieval stack. It distills each personalized concept into a sparse fingerprint of discriminative attributes, retrieves candidate fingerprints at run time, and employs chain-of-thought reasoning to verify them before generation.

For open-set personalization method, we additionally create a competitive baseline TAME<sup>†</sup>, a modified variant of the long-context MLLM personalized assistant TAME [4]. TAME is the first long-context MLLM personalization method that explicitly models both the long-term and short-term variations of personalized concepts and retrieves reliable contextual information for each concept to support personalized response generation. We unlock the open-set capability of TAME by retrieving the most relevant concept stored in memory based on the visual and textual information in the query.

Table 3: Performance of TAME-O under varying values of hyper-parameters.
<table><tr><td></td><td colspan="5">Parameter λ</td><td colspan="5">Parameter K</td></tr><tr><td>LCMP-H</td><td>0</td><td>0.1</td><td>1</td><td>2</td><td>5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>ACC-C (%)</td><td>78.33</td><td>80.56</td><td>82.22</td><td>82.44</td><td>78.67</td><td>74.44</td><td>82.78</td><td>82.22</td><td>81.11</td><td>80.00</td></tr><tr><td></td><td colspan="5">Parameter λ</td><td colspan="5">Parameter K</td></tr><tr><td>LCMP-O</td><td>0</td><td>0.1</td><td>1</td><td>2</td><td>5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>ACC-C (%)</td><td>72.89</td><td>75.89</td><td>75.11</td><td>75.44</td><td>76.11</td><td>71.44</td><td>74.17</td><td>75.11</td><td>75.67</td><td>75.18</td></tr></table>

Table 4: CFA of TAME-O over successive time steps on different datasets.
<table><tr><td></td><td colspan="5">LCMP-H</td><td colspan="5">LCMP-0</td></tr><tr><td>Time Step (t)</td><td>50</td><td>100</td><td>200</td><td>300</td><td>400</td><td>50</td><td>100</td><td>200</td><td>300</td><td>400</td></tr><tr><td>CFA (%)</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.74</td><td>95.65</td><td>96.67</td></tr></table>

A predefined similarity threshold is then applied: if the retrieved similarity exceeds the threshold, the concept is classified as existing; otherwise, it is treated as novel. We conduct a search over the threshold in the range [0, 1] with an interval of 0.1, and select the best-performing value of 0.6 for all experiments with TAME<sup>†</sup>.

## C.4 Implementation Details

We adopt the latest released MLLMs Qwen3.5-9B [6], and InternVL3.5-8B [18], as the MLLM backbone. Since TAME-O is a plug-and-play framework, we adopt TAME as the base personalized assistant for evaluation. When identifying the concept in user’s query, we first prompt the MLLM to extract open-set category descriptions from the query, and then use the extracted categories to perform grounding on the query image with Grounding-DINO [19]. In addition, we adopt the Qwen3-VL-Embedding-2B [20] to generate embeddings for both visual and textual inputs. For the CR Skill, we set the temporal decay coefficient λ to 1, the number of candidates K to 3, and the number of latest memory items for each concept R to 3. For PCF Strategy, we set the time interval T to 10, the number of fusion candidates L to 3, and the threshold τ to 0.6. For the CFR metric in Eq. (11), we set ϵ to 1. Since each forked identifier is created at one of the $t - | \mathcal { C } _ { t } |$ turns that revisit an already-seen concept, $\mathrm { C F R } ( t ) \in [ 0 , 1 ]$ . All the experiments are conducted on an NVIDIA RTX 5090 GPU.

Concept Identifiers. In our experiments, each novel concept is assigned an identifier named after its ground-truth concept in the benchmark (e.g., dog\_2), and each forked concept receives an additional numeric suffix (e.g., dog\_2\_1). These identifiers serve only as evaluation bookkeeping, allowing CRA and CFR to map every predicted identifier back to its ground-truth concept. They are never shown to the MLLM, since an identifier is assigned only after the CR Skill has decided that a concept is novel, and both $\mathcal { P } _ { r }$ and $\mathcal { P } _ { f }$ refer to candidate concepts through anonymous labels, describing each one only by its memory items and portrait (cf. Appendix J). Therefore, the identifiers leak no ground-truth information, and because all compared methods, including TAME<sup>†</sup>, share the same scheme, none of them benefits from it.

## D Hyper-Parameter Analysis

In this section, we evaluate the effect of important hyper-parameters within TAME-O: the temporal decay coefficient $\lambda ,$ and the number of candidates K. As shown in Table 3, our TAME-O exhibits relatively low sensitivity to hyperparameter variations, when $\lambda > 0$ or $K > 1$ . We set $K = 3$ and λ = 1 across all datasets.

## E Quantitative Analysis of the PCF Strategy

In the main paper, we offer the Concept Fission Rate (CFR) metric for TAME-O when equipped with the PCF Strategy. Here, we take a further step to empirically evaluate the fusion accuracy of the PCF

Table 5: Performance comparison on the LCMP-H and LCMP-O datasets. Higher values indicate better performance.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Open-Set Method</td><td colspan="5">LCMP-H</td><td colspan="5">LCMP-O</td></tr><tr><td>ACC-C</td><td>ACC-F</td><td>SPR-L</td><td>SPR-S</td><td>CRA</td><td>ACC-C</td><td>ACC-F</td><td>SPR-L</td><td>SPR-S</td><td>CRA</td></tr><tr><td rowspan="2">PeKit</td><td>TAME†</td><td>41.67</td><td>26.67</td><td>18.33</td><td>23.33</td><td>24.56</td><td>38.22</td><td>22.22</td><td>16.00</td><td>10.22</td><td>19.82</td></tr><tr><td>TAME-O</td><td>48.33</td><td>45.56</td><td>28.89</td><td>37.22</td><td>49.62</td><td>48.00</td><td>26.22</td><td>20.00</td><td>12.44</td><td>48.33</td></tr><tr><td rowspan="2">R2P</td><td>TAMEf</td><td>36.11</td><td>26.11</td><td>26.11</td><td>23.33</td><td>25.56</td><td>32.44</td><td>21.33</td><td>16.00</td><td>28.44</td><td>23.16</td></tr><tr><td>TAME-O</td><td>42.22</td><td>31.67</td><td>36.67</td><td>35.00</td><td>54.14</td><td>50.22</td><td>34.22</td><td>17.33</td><td>30.22</td><td>52.34</td></tr></table>

Strategy. Formally, each fusion operation is defined as an automatic merging process that combines two personalized concepts $c _ { m }$ and $c _ { n }$ stored in the TAD Memory into a unified one. When $c _ { m }$ and $c _ { n }$ correspond to the same ground-truth concept, the fusion operation is considered correct; otherwise, it is regarded as incorrect. Based on this criterion, we define the Concept Fusion Accuracy (CFA) as follows.

$$
\mathrm { C F A } ( t ) = \frac { 1 } { \lvert \mathscr { U } _ { \leq t } \rvert } \sum _ { ( c _ { m } , c _ { n } ) \in \mathscr { U } _ { \leq t } } \mathbb { I } \big ( \mathtt { G T } ( c _ { m } ) = \mathtt { G T } ( c _ { n } ) \big ) ,\tag{13}
$$

where $\mathcal { U } _ { < t }$ denotes the set of concept pairs $\left( { { c _ { m } } , { c _ { n } } } \right)$ that have been accepted by the PCF Strategy and fused into unified concepts up to time step $t , \mathtt { G T } ( c _ { m } )$ denotes the ground-truth concept identifier of concept $c _ { m }$ , which corresponds to the identifier assigned when the concept first appears, and $\mathbb { I } ( \cdot )$ is a binary indicator function that outputs 1 if the condition holds and 0 otherwise. In Table 4, we present the cumulative CFA at successive time steps while processing the historical dialogues in LCMP-H and LCMP-O. We observe that TAME-O consistently achieves high CFA values, suggesting that the fusion decisions made by the PCF Strategy are reliable.

Note on the CFR Computation. All CFR values in Figure 5 are computed strictly following Eq. (11) with ϵ = 1, recomputed from the logs of the same runs, and plotted without any per-method rescaling. This corrects two inconsistencies in the submitted version. First, the submitted Figure 6 normalized the total number of identifiers $\textstyle \sum _ { c _ { i } \in { \mathcal { C } } _ { t } } | \hat { \mathcal { T } } _ { c _ { i } } ^ { ( t ) } |$ , rather than the number of forked identifiers, by $t - | \mathcal { C } _ { t } |$ and rescaled the $\mathrm { T A M E } ^ { \dagger }$ curve by a factor of 0.55. Second, the logger in the released code normalized the number of forked identifiers by the total number of identifiers instead of by $t - | \mathcal { C } _ { t } |$ . Both the plotting script and the released code are now aligned with Eq. (11). Under all three normalizations, the ordering TAME<sup>†</sup> > TAME-O w/o PCF Strategy > TAME-O holds at every sampled time step.

## F Scalability across Other MLLM Personalized Assistants

Since TAME-O is designed as a training-free and plug-and-play framework, we further evaluate its scalability by integrating it with other personalized MLLM assistants beyond TAME [4] (reported in the main paper). Specifically, we integrate TAME-O with two representative training-free, RAGbased personalization methods, PeKit[17] and R2P[7], which differ in their memory organization and retrieval strategies. To enable open-set personalization, we augment both methods with our Concept Reconciliation (CR) Skill and a Time-Aware Dynamic (TAD) Memory that stores dialogue contexts with temporal markers. The CR Skill is used to determine whether an incoming personalized concept is novel or previously observed, while the original memory mechanisms of each method are retained for concept-specific storage and retrieval during response generation. As illustrated in Table 5, we compare the performance of these two baseline methods under the open-set setting by integrating TAME<sup>†</sup> and TAME-O, respectively. We observe that integrating TAME-O yields superior open-set personalization performance, demonstrating the strong scalability and generality of our method.

## G Additional Literature Review

## G.1 Memory-Augmented LLMs

Recent research on LLMs explores augmenting models with external memory to store and retrieve past interactions, enabling improved adaptability without parameter updates [24, 25, 26]. Existing studies typically organize memory as structured summaries, temporal events, or reasoning traces to highlight key information while reducing redundancy [27, 28]. To scale memory usage, retrieval mechanisms are introduced to efficiently locate relevant past experiences, often embedding memory entries as dense vectors and retrieving the most relevant ones via similarity search [29, 30, 31], with cosine similarity and vector indexing tools such as FAISS commonly used in practice. These studies demonstrate that external memory can effectively extend LLM capabilities without retraining by enabling models to accumulate and reuse past knowledge.

Building upon these advances, memory-based methods have also been studied in MLLM personalization, eliminating the need for frequent retraining for novel concepts [8, 4]. Unlike general memory-augmented LLM approaches that treat each memory entry independently, these methods explicitly maintain concept-level memory units to organize information around individual personalized concepts. Approaches such as RAP [8] and PeKit [17] store user-specific concepts in external memory and perform retrieval-based matching, while R2P [7] introduces structured “fingerprints” with reasoning for identification. More recently, TAME [4] extends the memory for concept storage to the modeling of long-context information for each user-specific concept. However, these methods generally assume a closed-set of predefined concepts and fail to extend the memory autonomously by the MLLMs for accommodating novel concepts. In contrast, our TAME-O introduces a time-aware dynamic memory with explicit concept reconciliation, enabling continuous concept acquisition and refinement under open-set, long-context interactions. Moreover, the PCF strategy further refines the memory to mitigate excessive concept fission that could otherwise lead to severe memory chaos.

## G.2 Open-Set Recognition

Open-set recognition (OSR) addresses the limitation of the closed-set assumption in traditional deep learning systems, enabling models to both classify known classes and detect inputs belonging to previously unseen classes [32, 33, 34]. The problem was first formally introduced by [34], who proposed a binary SVM-based framework for training and evaluation. Since then, extensive research has been devoted to this area. Existing approaches can be broadly divided into discriminative and generative methods. Discriminative approaches aim to learn more robust representations or decision boundaries to reduce overconfidence on unknown classes, such as placeholder-based representative points, supervised contrastive learning, and diverse representation regularization [35, 36, 37]. In contrast, generative approaches synthesize instances near the boundary between known and unknown classes using techniques such as GANs, variational autoencoders, or prototype-based modeling to better characterize open space risk [38, 39, 40].

Despite the name “open-set,” these methods are primarily designed to identify whether an input belongs to an unknown category, typically labeling it as unknown. In contrast, the open-set personalization setting considered in this work requires not only identifying novel user-specific concepts, but also assigning them new personalized concept identities, enabling the assistant to continually expand and refine the set of personalized concepts during interaction.

## G.3 Multimodal Large Language Models

Large Language Models (LLMs), built upon extensive world knowledge, have rapidly advanced across a wide range of domains [41, 42]. Building on this foundation, Multimodal Large Language Models (MLLMs) have recently emerged by integrating language decoders with multimodal encoders, enabling LLMs to perceive and reason over diverse data modalities [43, 44, 45, 46, 47, 48, 49, 50, 51]. Among MLLMs, vision–language models have been extensively studied [5, 18], and the fusion of visual and linguistic modalities has driven significant progress in applications such as Visual Question Answering (VQA) [52], multimodal medical analysis [53], and autonomous driving [54]. With these advances, MLLMs are increasingly regarded as reliable human assistants, supporting users in various aspects of daily life. Despite their success, mainstream MLLMs are primarily trained for generalpurpose scenarios and fail to adapt to individual users’ preferences and personalized contexts [1]. To bridge this gap, recent studies have begun to explore personalized MLLM assistants [2, 15, 4]. Among them, MC-LLaVA [13] extends personalization to multiple concepts via multi-concept instruction tuning, and CaT [14] synthesizes training data to strengthen personalized VLMs, while both still operate on a predefined set of concepts. In this work, we take a step further by addressing the open-set personalization setting and propose the first open-set personalized MLLM assistant TAME-O, which better aligns with users in open-world, long-term interactions.

![](images/3b60f968840bd0809bd2fddbb097af49cc3f49863428edc1a9fb10be78a6033b.jpg)

Figure 10: Case study: the procedure of TAME-O in resolving an existing old concept (left) and a novel concept (right).  
![](images/9a7f9f3dd5fc1567b0d839d9de4e85ff174bec62583f69a3ff324d28a755676e.jpg)  
Figure 11: Case study: two concepts that have similar attributes are selected to be fused into one concept.

## G.4 Large Language Model Personalization

Large Language Model (LLM) personalization [55, 56, 57, 58, 59] focuses on adapting an LLM’s responses to a single user’s preferences, interaction history, and persona characteristics (e.g., a preference for concise responses). Early efforts such as PersonaMem [60] introduce benchmarks with numerous simulated profiles to test whether models can leverage dialogue history to infer user preferences and deliver personalized responses. PersoBench [57] further explores personaaware dialogue generation in a zero-shot setting. Building upon these evaluations, methods like PLUM [56] employ dialogue-based data augmentation to fine-tune LLMs for more consistent useraware behavior. PALACE [61] advances this line with a topic-aware retrieval mechanism that extracts relevant cues from extended histories and a persona-driven prompt learning module for multi-session adaptation. Nevertheless, LLM personalization focuses on tailoring a model’s responses to a single user’s preferences, dialogue history, and persona traits (e.g., preferring concise answers). In contrast, MLLM personalization emphasizes enabling personalized conversations grounded in user-related multimodal concepts, such as the user’s pets, friends, and frequently mentioned objects.

## H Additional Case Study

## H.1 Additional Cases of Concept Reconciliation

In Figure 10, we provide additional concept reconciliation cases of TAME-O in resolving novel and existing concepts.

## H.2 Cases of Concept Fusion

As illustrated in Figure 11, we provide a case illustrating the concept fusion procedure of our proposed PCF Strategy.

## I Failure Case Analysis: Concept Fission

In this section, we present a failure case analysis of TAME-O, focusing on unintentional concept fission, as illustrated in Figure 12. Specifically, a concept originally associated with Dog 2 — a pet dog undergoing a temporary physical appearance change (e.g., its fur becoming dirty after playing in the mud) — is incorrectly identified as a novel concept, Dog 4. This misclassification triggers an unintended concept fission.

![](images/045d6a49c4d8097357217bddf97d86f19d439c54eaf1a6506ee120fbd8e7c92d.jpg)  
Figure 12: Case study: incorrect concept identification leads to the fission of an already grasped concept.

## J Prompt Presentation for TAME-O

In this section, we present the prompts $\mathcal { P } _ { r }$ and $\mathcal { P } _ { f }$ designed for performing concept reconciliation and fusion, respectively.

## K Limitations

Reliance on External Modules. TAME-O relies on off-the-shelf grounding and embedding models to localize and retrieve personalized concepts, and their errors may propagate to concept reconciliation. Nevertheless, these modules can be readily replaced by stronger ones as they become available.

Robustness under Noisy Interactions. Our benchmarks are constructed in a controlled manner, whereas real-world interactions can be noisier, e.g., with ambiguous references or low-quality images and descriptions. We leave a systematic evaluation under such noisy interactions to future work.

Indirect Recovery from Incorrect Fusion. Concept fusion may occasionally merge two distinct concepts. Although such a merge cannot be directly reverted, TAME-O can recover from it indirectly: benefiting from the attribute-level and time-aware memory organization, subsequent interactions progressively correct the affected concept memory with recent observations. A direct mechanism to revert incorrect fusions is left to future work.

Scalability and Privacy. The pairwise similarity computation in concept fusion grows quadratically with the number of concepts, although the number of MLLM invocations per fusion round is bounded by L. In large-scale deployment, this cost can be reduced by caching pairwise similarities and recomputing only the pairs involving recently updated concepts, or by approximate nearest-neighbor search. Moreover, long-term personalized memory contains private user information, which calls for local storage and user-controlled deletion in practical deployment.

## L Broader Impacts

Since this study focuses on designing open-set personalized MLLM assistant, which is closely related to everyone’s daily life, it has broader impacts in both positive and negative aspects. On the positive side, open-set MLLM personalization can make future assistants more adaptive, user-aligned, and convenient in daily life. By automatically discovering and maintaining user-specific concepts through long-term interactions, such assistants can provide responses that are more context-aware and personalized, reducing the burden on users to manually register or describe every personal entity. This capability may benefit various real-world scenarios, such as daily assistance, education, accessibility support, elderly care, and long-term human AI collaboration, where understanding user-specific objects, people, pets, or environments is important for providing useful and grounded assistance. On the negative side, such systems may introduce risks from both misuse and the inherent limitations of MLLMs. For misuse, the ability to continuously acquire personalized concepts could be abused for intrusive profiling, unauthorized surveillance, or manipulative personalized content generation. For model limitations, our method may inherit hallucination and bias issues from the underlying MLLMs. It may generate responses that are not fully grounded in the input image, dialogue context, or stored personalized memories, and such hallucinations can be harmful in high-stakes scenarios. Moreover,

<table><tr><td>Prompt: You are a concept identification assistant. Your goal is to determine if the user&#x27;s question refers to one of the EXISTING candidates or to a completely NEW concept. {Recent Conversations} # Question</td></tr><tr><td></td></tr><tr><td>{User Query}</td></tr><tr><td>Image: {present / absent} # Candidates (Existing Memories):</td></tr><tr><td>[For each candidate concept ck ∈ CK, k = 1, . . . , K:]</td></tr><tr><td>## Candidate: CANDIDATE_{k}</td></tr><tr><td>- Portrait: [IMAGE]</td></tr><tr><td>- Static Memory: {Static Memory Items}</td></tr><tr><td>- Recent Dynamic Memory: {Recent Dynamic Memory Items}</td></tr><tr><td># Instructions Compare the entity in the Question against each Candidate.</td></tr><tr><td>• Check for mismatches in Name, Species, Physical Appearance, or Core Attributes.</td></tr><tr><td>• If the Question describes an entity that is fundamentally different from a Candidate, they are NOT the same.</td></tr><tr><td>• If the Question explicitly mentions a candidate&#x27;s proper name, select that candidate unless</td></tr><tr><td>species or stable identity details contradict it. • Transient details (e.g., accessories, location, mood, pose, or activity) may differ from</td></tr><tr><td>memory and are not enough by themselves to make a new concept; compare stable identity details (e.g., name, species, breed, color, or markings).</td></tr><tr><td>• If the tone of the Question is to introduce a new concept (e.g., a person, pet, or belonging), select“NEW_CONCEPT&quot;.</td></tr><tr><td>• Context might be similar, but specific details must match.</td></tr><tr><td>• Use only the neutral candidate labels shown above. Do not infer identity from labels.</td></tr><tr><td>• If NO candidate is a strong match, select “NEW_CONCEPT&quot;.</td></tr><tr><td># Output Format</td></tr><tr><td>Output ONLY one of the following:</td></tr><tr><td>1. “EXISTING:CANDIDATE_N&quot; (if it matches a candidate) 2. “NEW_CONCEPT&quot;</td></tr></table>

## Prompt P<sub>r</sub> for Concept Reconciliation.

biases from the vision encoder and language model may lead to unfair or stereotypical interpretations of user-specific entities. Therefore, open-set personalized MLLM assistants should be deployed with appropriate safeguards, usage restrictions, and careful evaluation, especially in sensitive applications.

## M License of Assets

Datasets: LCMP (research-use), Oxford-IIIT Pet (CC BY-SA 4.0), MS-COCO (Flickr Terms of Use and image-specific licenses), and PerVA (research-use).

## Prompt P<sub>f</sub> for Concept Fusion.

Prompt: You are a concept similarity expert. Your task is to determine if two concepts represent the SAME entity that was incorrectly split (fissioned).

## # CONCEPT A

\## Portrait: {Portrait Image or “(No portrait available)”}

\## Static Memory (Permanent Facts): {Static Memory Items}

\## Dynamic Memory (Recent Events): {Dynamic Memory Items}

## # CONCEPT B

\## Portrait: {Portrait Image or “(No portrait available)”}

\## Static Memory (Permanent Facts): {Static Memory Items}

\## Dynamic Memory (Recent Events): {Dynamic Memory Items}

## # Instructions

Compare the two concepts carefully to determine if they are the SAME entity.

## CRITICAL RULES FOR “SAME” VERDICT:

1. Conclude “SAME” only with clear identity evidence: the portraits show the same individual/object, or the memories share a distinctive name, model, or owner-specific identifier.

2. Generic category overlap is not enough: two cats or two similar-looking people can still be different entities.

3. If both concepts have specific names, brands, models, colors, breeds, or personal details that differ, conclude “DIFFERENT”.

4. A generic reference (e.g., “the cat”) matches a specific concept only when the visual evidence or distinctive memories strongly connect them.

5. Merge an unnamed fragment with few memories into the richer concept when its later details are compatible and no identity evidence contradicts it.

## GUIDELINES:

• Assume memories accumulate over time.

• Do not merge based on broad class, scene, pose, or activity alone.

• Later memory fragments may omit the name.

• If the evidence is weak or ambiguous, favor “DIFFERENT”.

## # Output Format

1. Brief reasoning (1–2 sentences).

2. Final verdict: “VERDICT: SAME” or “VERDICT: DIFFERENT”.

![](images/f1f26caefc161d8779235d293f880cb0a1b79d90c4a4929321df2b7f7c4868eb.jpg)  
Figure 13: Cases of personalized concepts from LCMP-O benchmark.