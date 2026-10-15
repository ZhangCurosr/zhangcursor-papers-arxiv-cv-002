# MMKG-R1: Constructing Multimodal Knowledge Graphs via Reinforcement Learning

Zongyu Wu<sup>†</sup>, Yilong Wang<sup>†</sup>, Xiaochen Wang, Minhua Lin, Yuan Zhong,

Fenglong Ma, Xiang Zhang, Suhang Wang

The Pennsylvania State University

{zongyuwu,yvw5769,xcwang,mfl5681,yfz5556,fenglong,xzz89,szw494}@psu.edu

## Abstract

Multimodal knowledge graphs (MMKGs) organize information across different modalities to support multimodal Graph Retrieval-Augmented Generation (RAG), which aims to improve the performance of multimodal large language models (MLLMs). The effectiveness of these multimodal Graph RAG methods largely depends on the quality of the constructed MMKGs. However, most existing methods construct MMKGs without explicit guidance on graph quality, while reinforcement learning (RL)-based methods that rely on finalanswer correctness may provide only indirect feedback on MMKG quality. To address these limitations, we propose MMKG-R1, a novel MMKG-quality-driven RL framework for training an MLLM as a multimodal knowledge constructor. To be specific, MMKG-R1 introduces multiple quality-aware rewards to directly evaluate both the quality of information extracted from individual knowledge sources and the quality of the assembled MMKG. These rewards are then used to guide the RL training. Experiments across multiple datasets show that MMKG-R1 constructs MMKGs that improve downstream question-answering accuracy.

## 1 Introduction

Multimodal large language models (MLLMs) (Liu et al., 2023, 2024; Bai et al., 2025; Yin et al., 2024) have been widely adopted across various domains. Despite their effectiveness, they still suffer from outdated knowledge and hallucinations (Bai et al., 2024; Li et al., 2023), especially when answering questions that require recent or unseen information. Retrieval-Augmented Generation (RAG) (Lewis et al., 2020; Gao et al., 2023; Abootorabi et al., 2025) is an established approach for reducing hallucinations in (M)LLMs (Touvron et al., 2023; Wang et al., 2024) and enabling them to access and utilize external knowledge. As knowledge is often multimodal, including information like texts and images, recent works (Guo et al., 2025a; Dai et al., 2026) start constructing multimodal knowledge graphs (MMKGs) (Liang et al., 2024) to support multimodal Graph RAG such as MMGraphRAG (Wan and Yu, 2025). By incorporating both semantic information in each modality and cross-modal associations, these methods aim to further improve retrieval quality and downstream question answering performance.

![](images/c9b0279f973c94d2d8de5fa1166c70474a08d134a4e70e8eb92c3147930988cc.jpg)  
Figure 1: An example illustrating that a correct downstream answer for a certain question does not necessarily indicate that all content in the constructed MMKG is accurate or of high quality. MMKG-R1 introduces explicit quality-aware rewards to assess the constructed MMKG and utilize them to train a multimodal constructor.

Although MMKG-based Graph RAG methods have shown promising performance, most existing approaches (Guo et al., 2025a; Wan and Yu, 2025) follow a common pipeline in which MMKGs are constructed directly by pretrained models without explicitly considering their utility for downstream RAG tasks. Since the performance of these methods heavily rely on the quality of the constructed MMKGs, a promising direction is to train a model to generate high-quality MMKGs.

Some recent approaches (Tsang et al., 2026; Lin et al., 2026) have begun exploring graph optimization through reinforcement learning (RL). AutoGraph-R1 (Tsang et al., 2026) trains constructors for text-derived KGs using task-aware rewards based on deducibility and passage recall. EvoGraph-R1 (Lin et al., 2026) trains an agent to retrieve from and edit a multimodal knowledge hypergraph, primarily guided by final answer corretness. However, as noted in AutoGraph-R1, answerlevel feedback alone does not reliably reflect graph quality. A correct downstream answer does not necessarily imply that the MMKG correctly represents the source documents. For example, as shown in Figure 1, a model using multimodal Graph RAG may correctly answer a question about a company’s profit even when the constructed MMKG records an incorrect profit value. Other correctly extracted information, such as the company’s revenue, may still enable retrieval of the relevant source image, allowing the model to obtain the correct profit value directly from the image. However, such noise in the MMKG may result in reduced performance on other questions. For example, when asked "What is the revenue of the company with a profit of \$20M?", the system may fail to identify the relevant company and retrieve its source image due to the incorrect profit value in the MMKG, potentially leading to an incorrect answer.

Therefore, in this paper, we propose to study a novel problem of designing reward signals that assess both the quality of information extracted from individual knowledge sources and the quality of the MMKG assembled from the extracted information to better train the MMKG constructor through RL. This raises two key challenges: (i) How can we directly evaluate the quality of constructed MMKGs without relying on final answer correctness? (ii) How can we provide effective reward signals at both the knowledge-source and MMKG levels to guide constructor training? To address the challenges, we propose MMKG-R1, a novel graph-quality-driven RL framework that trains a multimodal constructor using complementary feedback at both MMKG and knowledgesource levels. At the MMKG level, we design Deducibility Reward to evaluate the evidence retrieved from the assembled MMKG, encouraging the constructor to preserve relevant facts and relations from the original knowledge sources. At the knowledge-source level, we design two reward signals. The Source-Grounded Quality Reward compares extracted information with its original knowledge sources to evaluate extraction accuracy and coverage of key information, independent of the current question. The Graph-Readiness Reward encourages well-structured extractions that provide usable entities and facts for MMKG assembly.

Our main contributions are: (i) We present MMKG-R1, a novel quality-driven RL framework for constructing high-quality MMKGs from multimodal knowledge sources; (ii) We move beyond conventional final-answer-based rewards by designing multiple quality-aware reward signals that directly assess construction quality at both the knowledge-source and MMKG levels, providing more targeted feedback to guide MMKG construction; and (iii) We conduct comprehensive experiments on multiple datasets, showing that MMKG-R1 consistently generates high-quality MMKGs that can effectively support multimodal Graph RAG.

## 2 Related Work

## 2.1 Graph RAG

Retrieval-augmented generation (RAG) (Lewis et al., 2020; Chen et al., 2022; Hu et al., 2023; Yu et al., 2025; Abootorabi et al., 2025; Gao et al., 2023) provides external evidence to mitigate the knowledge limitations and hallucinations of (multimodal) large language models. Graph RAG extends this paradigm by organizing knowledge into graph structures to support retrieval and generation (Edge et al., 2024; Sun et al., 2024; Guo et al., 2025b; Wang et al., 2026). For example, GraphRAG (Edge et al., 2024) uses hierarchical community summaries, while LightRAG (Guo et al., 2025b) combines graph-based indexing with dual-level retrieval. Recent methods construct multimodal knowledge graphs (MMKGs) to support multimodal Graph RAG (Wan and Yu, 2025; Yuan et al., 2026; Guo et al., 2025a; Dai et al., 2026; Wu et al., 2026b). For instance, MMGraphRAG (Wan and Yu, 2025) builds image-based entity-relation graphs and a text-based KG, respectively, then integrates them through cross-modal fusion. However, most existing methods rely on pretrained models without task-specific training of the MMKG construction process.

## 2.2 Reinforcement Learning for Graph Construction

Reinforcement learning (RL) methods (Schulman et al., 2017) like Group Relative Policy Optimization (GRPO) (Shao et al., 2024), has been used to train LLMs for retrieval decisions and graph reasoning (Jin et al., 2025b; Wu et al., 2026a; Luo et al., 2026; Xu et al., 2026; Zhang and Zhao, 2025). Recent studies further apply RL to graph construction. AutoGraph-R1 (Tsang et al., 2026) uses GRPO to train a textual graph constructor with graph utility rewards, while EvoGraph-R1 (Lin et al., 2026) learns to evolve query-conditioned multimodal hypergraphs using rewards for answer correctness, trajectory structure, and action cost. However, rewards in EvoGraph-R1 do not directly evaluate the quality of the constructed graph. MMKG-R1 instead learns to construct high-quality multimodal knowledge graphs through direct quality feedback. Our designed rewards assess both the information extracted from individual sources and the assembled MMKG, providing supervision for sourcegrounded extraction and graph readiness.

## 3 Preliminaries

In this section, we first formalize how MMKG-R1 constructs an MMKG and how the constructed graph supports question answering, and then define the learning problem.

Multimodal graph construction. Let X denote a collection of source instances from multimodal documents, each being a text passage, an image, or a table. MMKG-R1 trains an MLLM $\pi _ { \theta }$ as the multimodal constructor to generate a structured extraction $y _ { x }$ of typed entities and subject–relation–object triples from the visible content $v ( x )$ of each source $x \in \mathcal { X }$

$$
y _ { x } \sim \pi _ { \theta } ( \cdot \mid v ( x ) ) .\tag{1}
$$

Additional source information $\mu _ { x } .$ , such as provenance metadata and references to the original content, is maintained by the environment and is never shown to $\pi _ { \theta }$ . Two fixed, non-trainable modules then process the extractions: the parser $\mathcal { P }$ parses each $y _ { x } .$ , and the assembler $\boldsymbol { B }$ uses the parsed results and $\mu _ { x }$ to assign graph identifiers, attach provenance, link entities across sources, and assemble the graph as:

$$
G _ { \theta } ( \mathcal { X } ) = \mathcal { B } \big ( \{ ( \mathcal { P } ( y _ { x } ) , \mu _ { x } ) \} _ { x \in \mathcal { X } } \big ) .\tag{2}
$$

The resulting $G _ { \theta }$ is a fact-centric and sourcegrounded MMKG: each extracted triple becomes a fact node connected to its arguments and its supporting source, and fixed entity linking connects compatible mentions across sources. Since P and B are fixed, the quality of $G _ { \theta }$ is governed by the extractions of $\pi _ { \theta }$

Graph-based question answering. Given a question $q$ and a constructed MMKG $G ,$ a fixed retrieval and evidence-packing pipeline selects relevant evidence under predefined budgets:

$$
E ( q , G ) = ( { \mathcal { F } } _ { q } , { \mathcal { V } } _ { q } ) ,\tag{3}
$$

where $\mathcal { F } _ { q }$ contains retrieved structured facts with source provenance, and $\mathcal { V } _ { q }$ contains selected original visual inputs (images and image-backed tables) with their available source metadata. A downstream model answers $q$ using only $E ( q , G )$ , without access to the full set of sources.

Problem formulation. Each training example $\xi _ { q } = ( q , \mathcal { A } _ { q } , \mathcal { X } _ { q } ) \in \mathcal { D }$ consists of a question $q ,$ a set of accepted answers $\mathcal { A } _ { \boldsymbol { q } }$ , and the annotated supporting sources $\mathcal { X } _ { q }$ . Construction is questionblind: $\pi _ { \theta }$ processes each $x \in \mathcal { X } _ { q }$ independently from $v ( x )$ alone, without access to $q , { \mathcal { A } } _ { q } ,$ , or the other sources, and yields $G _ { \theta } ( \mathcal { X } _ { q } )$ via Eqs. (1)–(2), so its extractions can be reused across questions. The question and answers are used only by the training environment to assess the resulting graph. Therefore, we seek to optimize

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \xi _ { q } \sim \mathcal { D } } \mathbb { E } _ { \pi _ { \theta } } \Big [ \mathscr { Q } \big ( G _ { \theta } ( \mathcal { X } _ { q } ) ; \xi _ { q } \big ) \Big ] ,\tag{4}
$$

where the inner expectation is over the sampled extractions $\{ y _ { x } \} _ { x \in \mathcal { X } _ { q } }$ , and $\mathcal { Q }$ scores the quality of the graph itself rather than the correctness of a final answer. Section 4 specifies Q and how Equation. (4) is optimized.

## 4 Method

Our proposed MMKG-R1 trains a multimodal constructor to facilitate the contruction of MMKGs that effectively support downstream question answering. As shown in Figure 2, we introduce three complementary reward signals that jointly evaluate evidence sufficiency for downstream question answering, fidelity to the original knowledge source, and structural usability: (i) Deducibility Reward, which evaluates whether the retrieved evidence from a constructed MMKG is sufficient to support a correct answer; (ii) Source-Grounded Quality Reward, which assesses whether the knowledge extracted by the constructor faithfully preserves the central content of its original knowledge source; and (iii) Graph-Readiness Reward, which encourages structurally valid and usable outputs for graph construction. Evidence sufficiency depends on the question and on how extractions from different sources combine, so the Deducibility Reward is computed at the candidate-graph level. Fidelity and usability are properties of each extraction regardless of the question, so the Source-Grounded Quality and Graph-Readiness Rewards are computed at the source-extraction level.

![](images/3033826851c6d482d43ecdb299dc6ea923bcb7c9a93554c67f5534e5d3a7c42a.jpg)  
Figure 2: Overview of MMKG-R1. The constructor independently extracts structured knowledge from each knowledge source. The extracted knowledge is then used to assemble MMKGs. Three designed rewards guide constructor optimization: the graph-level Deducibility Reward and two source-level rewards, the Source-Grounded Quality Reward and the Graph-Readiness Reward.

## 4.1 Source-Level Construction and Graph Assembly

For each training question $q ,$ we consider its annotated supporting knowledge sources $\chi _ { q } .$ , as defined in Section 3. The constructor $\pi _ { \theta }$ processes each knowledge source $x _ { m } ^ { \left( q \right) }$ independently, using only its content $v ( x _ { m } ^ { ( q ) } )$ . Let $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ denote the rollout policy used to collect the current training batch. For each knowledge source $x _ { m } ^ { \left( q \right) }$ , we independently sample four responses $y _ { m , i } ^ { ( q ) } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot ~ | ~ v ( x _ { m } ^ { ( q ) } ) )$ , where $i \in \{ 1 , 2 , 3 , 4 \}$

Constructor outputs with the same sampling index i are then parsed and combined into a candidate multimodal knowledge graph $G _ { i } ^ { ( q ) }$ , while preserving source metadata. This produces four candidate MMKGs per question for group-relative comparison, without enumerating all possible combinations of outputs across knowledge sources. For each candidate graph $G _ { i } ^ { ( q ) }$ , we use the fixed retrieval and evidence-packing pipeline introduced in Section 3 to obtain $E _ { i } ^ { ( q ) } = E ( q , G _ { i } ^ { ( q ) } )$ , which includes structured facts retrieved from the graph and original visual evidence when available. We use a shared, fixed environment to perform entity linking, graph assembly, retrieval, and budgeted evidence packing for all candidates. Throughout training, only the constructor is updated, while all other components remain fixed.

## 4.2 Graph-Level Deducibility Reward

We first present the design of the Deducibility Reward, which is inspired by the Knowledge-Carrying Reward proposed in AutoGraph-R1 (Tsang et al., 2026). For each candidate MMKG $G _ { i } ^ { ( q ) }$ , we use a frozen MLLM as the Judge to evaluate whether the retrieved evidence $E _ { i } ^ { ( q ) }$ from $G _ { i } ^ { ( q ) }$ provides sufficient support for at least one accepted complete answer in $\mathcal { A } _ { \boldsymbol { q } }$ . Training graphs are constructed from the annotated supporting sources $\chi _ { q } .$ , which contain the evidence needed to answer $q .$

To be specific, the judge model is prompted to evaluate two types of evidence gaps in sequence: (i) Whole-bundle gap: Indicates that both the retrieved structured facts and the original visual information in knowledge sources are insufficient to support any complete accepted answer; and (ii) Structuredpath gap: This gap is evaluated only when the whole bundle is sufficient. It indicates that omissions or contradictions in answer-critical information prevent the structured facts from supporting a complete, correct, and unambiguous reasoning path. The two checks distinguish the usefulness of the retrieved evidence from the completeness and correctness of the structured reasoning path.

Each reported gap describes concrete missing or conflicting evidence. A reported whole-bundle gap results in a score of 0. When only a structuredpath gap exists, the evidence supports an answer, but the structured reasoning path is incomplete or incorrect. We assign a score of 0.5 to recognize the usefulness of the evidence while encouraging improvements to the structured facts. A candidate with neither gap receives a score of 1. These scores define three levels of evidence utility, formalized as follows:

$$
\begin{array} { r } { d _ { i } ^ { ( q ) } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { w h o l e - b u n d l e ~ g a p , } } \\ { 0 . 5 , } & { \mathrm { s t r u c t u r e d - p a t h ~ g a p ~ o n l y , } } \\ { 1 , } & { \mathrm { n e i t h e r ~ g a p . } } \end{array} \right. } \end{array}\tag{5}
$$

## 4.3 Knowledge Source-Level Feedback

While the Deducibility Reward described above evaluates whether retrieved graph evidence is sufficient to support downstream question answering, it does not fully capture the quality of individual knowledge source-level extractions. To address this limitation, we introduce two complementary knowledge source-level rewards: the Source-Grounded Quality Reward and the Graph-Readiness Reward.

Source-Grounded Quality Reward. The proposed Deducibility Reward evaluates evidence for the current question, but source extractions are intended for reuse across queries. An extraction may retain enough information to retrieve a relevant image or answer the current question while omitting other central content or introducing errors unrelated to that question. These deficiencies may leave the Deducibility Reward unchanged, yet limit the graph’s reliability and usefulness for other queries. We therefore introduce the Source-Grounded Quality Reward to encourage the constructor to accurately extract the knowledge source’s central knowledge. A frozen judge model is prompted to evaluate each selected extraction against its original knowledge source, without access to the question or accepted answers.

The judge model evaluates two complementary criteria: (i) reliability, requiring the extraction to contain at least one meaningful, source-supported assertion and no material factual or binding errors; and (ii) central coverage, requiring the extraction to capture the source’s key explicit information, including entities, relations, values, and necessary qualifiers, without requiring exhaustive coverage of minor details.

An extraction receives a binary quality reward $u _ { m , i } ^ { ( q ) } = 1$ only if both criteria are satisfied; otherwise, $u _ { m , i } ^ { ( q ) } = 0$ . Thus, an accurate extraction that omits central information, or a comprehensive extraction containing material errors, receives a reward of 0. Source-Grounded Quality Reward is considered only when all four candidate graphs have valid judge evaluations and receive the same Deducibility Reward of either 0.5 or 1. In these groups, the retrieved evidence is sufficient to support a complete answer, so the graph-level reward cannot distinguish among candidates. Source-level evaluation then provides additional feedback by assessing extraction accuracy and coverage of key information that the question-specific reward may overlook. This encourages the constructor to better preserve information from the original sources, even when candidates are equally answerable. For each group, we sample at most one eligible source to assess extraction quality, providing additional supervision while limiting evaluation costs.

Graph-Readiness Reward. To encourage the constructor to generate structurally valid and usable extractions, we introduce the Graph-Readiness Reward. It considers both the format validity of the original response, similar to format rewards used in prior work (Jin et al., 2025a), and the usability of the extracted content for MMKG construction.

For each response $y _ { m , i } ^ { ( q ) }$ , we define a binary Graph-Readiness reward $r _ { m , i } ^ { ( q ) } \in \{ 0 , 1 \}$ . We set $r _ { m , i } ^ { ( q ) } = 1$ only if the original output satisfies the required JSON schema without repair, the response terminates normally and passes the generationprotocol checks, and it produces at least one usable source-bound entity mention or fact. Otherwise, $r _ { m , i } ^ { ( q ) } = 0$ . Malformed responses receive a reward of 0 even if the parser can recover their content, as do structurally valid but empty responses.

## 4.4 GRPO-Based Optimization

We normalize Deducibility rewards $d _ { i } ^ { ( q ) }$ across valid candidate graphs for each question to compute the advantages $A _ { m , i } ^ { ( q ) , D }$ for their corresponding source responses. The source-level advantages $A _ { m , i } ^ { ( q ) , Q }$ and $A _ { m , i } ^ { ( q ) , R }$ are computed from separately mean-centered rewards $u _ { m , i } ^ { ( q ) }$ and $r _ { m , i } ^ { ( q ) }$ across responses generated for the same source. After applying the corresponding activation and source-credit masks, the combined advantage for each source response $y _ { m , i } ^ { ( q ) }$ is:

$$
A _ { m , i } ^ { ( q ) } = A _ { m , i } ^ { ( q ) , D } + \lambda _ { Q } A _ { m , i } ^ { ( q ) , Q } + \lambda _ { R } A _ { m , i } ^ { ( q ) , R } ,\tag{6}
$$

where $\lambda _ { Q }$ and $\lambda _ { R }$ weight the two source-level advantages. Finally, we optimize the constructor with

<table><tr><td>Method</td><td>MMQA</td><td>MMDocRAG</td><td>MMLongBench-Doc</td></tr><tr><td>Direct Inference</td><td>23.96</td><td>50.45</td><td>24.73</td></tr><tr><td>Naive RAG</td><td>29.79</td><td>59.46</td><td>25.38</td></tr><tr><td>Base GraphRAG (Dense)</td><td>33.75</td><td>67.12</td><td>40.65</td></tr><tr><td>F1-based RL (Dense)</td><td>35.83</td><td>71.17</td><td>40.43</td></tr><tr><td>MMKG-R1 (Dense)</td><td>38.13</td><td>72.07</td><td>41.29</td></tr><tr><td>Base GraphRAG (Subgraph)</td><td>29.58</td><td>63.96</td><td>40.65</td></tr><tr><td>F1-based RL (Subgraph)</td><td>29.79</td><td>64.41</td><td>41.08</td></tr><tr><td>MMKG-R1 (Subgraph)</td><td>33.96</td><td>65.32</td><td>41.51</td></tr><tr><td>Base GraphRAG (Hybrid)</td><td>40.83</td><td>66.67</td><td>41.94</td></tr><tr><td>F1-based RL (Hybrid)</td><td>38.54</td><td>68.02</td><td>42.15</td></tr><tr><td>MMKG-R1 (Hybrid)</td><td>42.50</td><td>68.92</td><td>42.37</td></tr></table>

Table 1: Answer accuracy (%) on the three evaluation subsets. Qwen3-VL-8B-Instruct is used as the answer model. Direct Inference and Naive RAG do not construct knowledge graphs. For MMLongBench-Doc, trained constructors are transferred from MMDocRAG without additional training. Bold indicates the best result within each dataset–graph-retrieval block.

GRPO by maximizing a dual-clipped surrogate objective (Ye et al., 2020).

## 5 Experiments

In this section, we conduct comprehensive experiments to answer these research questions: (i) RQ1: Can our proposed MMKG-R1 construct highquality multimodal knowledge graphs (MMKGs) that improve the performance of multimodal Graph RAG? (ii) RQ2: Are our proposed reward scores more effective than the F1 score in guiding the construction of high-quality MMKGs? and (iii) (RQ3): How do different reward designs affect the performance of MMKG-R1?

## 5.1 Experimental Setup

## 5.1.1 Datasets

We train two multimodal knowledge constructors separately on the training sets of MM-DocRAG (Dong et al., 2025) and MultiModalQA (MMQA) (Talmor et al., 2021). We then evaluate them on subsets of their respective test sets. We use subsets for two reasons: (i) Some test questions share gold passages with training questions, which may lead to data leakage and affect evaluation reliability; (ii) Some test questions can be answered using textual information alone, without requiring visual evidence, making them less suitable for evaluating multimodal knowledge construction. To assess the transferability of MMKG-R1, we further evaluate its performance on a subset of MMLongBench-Doc (Ma et al., 2024). The details of all used datasets are described in Appendix A.

<table><tr><td>Answer Model Method</td><td></td><td></td><td>Dense Subgraph Hybrid</td><td></td></tr><tr><td rowspan="3">32B-AWQ</td><td>Base</td><td>39.79</td><td>35.42</td><td>43.96</td></tr><tr><td>F1-based RL</td><td>40.00</td><td>36.46</td><td>46.46</td></tr><tr><td>MMKG-R1</td><td>40.63</td><td>38.54</td><td>48.96</td></tr></table>

Table 2: Performance of MMKG-R1 on MMQA with Qwen3-VL-32B-Instruct-AWQ as the answer model. Base denote Base Graph RAG.

## 5.1.2 Evaluation Protocol

We use a judge model to evaluate answer accuracy. To reduce potential evaluation bias, we use Qwen3.5-27B (Team, 2026) as an independent judge, distinct from the model used for calculating rewards during training. For each dataset, all methods are evaluated using the same prompt and evaluation criteria.

## 5.1.3 Baseline Methods

We compare MMKG-R1 with four baseline methods: (i) Direct Inference with Materials. We retain the supporting materials and add distractor within the model’s context budget. The frozen answer model directly answers from this expanded multimodal context. (ii) Naive RAG. We index raw text and table chunks together with visual materials in a corpus-wide pool using a multimodal embedding model. For each question, the top 20 units are retrieved by embedding similarity and supplied to the frozen answer model. (iii) Base GraphRAG. A constructor without reinforcement learning extracts knowledge to build an MMKG for downstream retrieval and question answering. (iv) F1-based RL. This baseline adopts a correctnessoriented RL objective used in representative related work (Lin et al., 2026), computing token-level F1 between the responses generated by a frozen answer model and the reference answers. It retains the Graph-Readiness Reward but excludes the Source-Grounded Quality Reward.

![](images/35a09772a583e5bf75a8ffb0970193d2a1ecda836ecc31ec549322df93c40da2.jpg)  
Figure 3: Reward ablations on MMQA subset. Metric is answer accuracy (%).

## 5.1.4 Implementation Details

Computational Devices. All training and evaluation experiments are conducted on machines equipped with four NVIDIA RTX A6000 GPUs, each with 48 GB of memory.

Backbone Models. We use Qwen3-VL-4B-Instruct (Bai et al., 2025) as the multimodal constructor. Both the Deducibility Reward and Source-Grounded Quality Reward use frozen Qwen3- VL-32B-Instruct-AWQ. The Graph-Readiness Reward is computed using deterministic rules. The downstream answer model is Qwen3-VL-8B-Instruct (Bai et al., 2025) for all three datasets. All methods use the same frozen answer model and answer-evaluation protocol within each dataset.

Retrieval Settings During training, we use a fixed retrieval and evidence-packing procedure with entity, fact, and relation entry points to compute the Deducibility Reward. At evaluation time, we assess the same trained constructor under three different retrieval methods: Dense Retrieval, Subgraph Retrieval, and Hybrid Retrieval.

More implementations are described in B.

## 5.2 Main Results

Table 1 reports answer accuracy across the three evaluation subsets. Direct Inference and Naive RAG generally achieve lower accuracy than graphbased retrieval methods. This highlights the benefits of structured multimodal knowledge in graphs. F1-based RL training improves downstream accuracy over Base GraphRAG in most settings, showing that final-answer correctness can provide useful feedback for MMKG construction. However, these improvements are not consistent across all settings. For example, on the MMQA subset with hybrid retrieval, accuracy drops from 40.83% for the untrained Base GraphRAG to 38.54% after F1- based RL training, suggesting that answer-based rewards may not always provide reliable guidance for MMKG construction. Our proposed MMKG-R1 consistently outperforms both Base GraphRAG and F1-based RL, For example, on the MMQA subset with dense retrieval, MMKG-R1 achieves an accuracy of 38.13%, outperforming both Base GraphRAG (33.75%) and F1-based RL (35.83%). This demonstrates that our quality-aware rewards provide effective guidance for MMKG construction

## 5.3 Transferability of MMKG-R1

Different Answer Models. On the MMQA subset, we additionally evaluate Qwen3-VL-32B-Instruct-AWQ as the answer model, alongside Qwen3-VL-8B-Instruct. The results are shown in Table 2. MMKG-R1 consistently constructs MMKGs that achieve higher answer accuracy across all three retrieval strategies, even when evaluated with a different answer model.

Cross-Dataset Transferability. To evaluate the transferability of MMKG-R1, We further evaluate the MMDocRAG-trained constructors on MMLongBench-Doc without further training. As shown in Table 1, MMKG-R1 consistently outperforms both Base GraphRAG and F1-based RL across all three retrieval strategies. These results suggest that our quality-aware rewards help the constructor learn knowledge construction capabilities that transfer to unseen datasets.

## 5.4 Ablation Studies

To examine the contributions of our reward design, we evaluate two variants on the MMQA subset: (i) MMKG-R1 w/o Source-Grounded Quality (SGQ) Reward. This variant removes the SGQ Reward while retaining the Deducibility and Graph-

![](images/ac4d46b31c93ccc1ae2ff23c184585692bbdc0370b688cf1f5747506beab754b.jpg)

![](images/d0e7cbbe02bf819ed795c77ef0949af7ca09f7ecb09b0af188c1ce2276dcd6bf.jpg)

![](images/f307f471d8e2cca8a050576926db75ea7a29d25197e062d7ac07a03e324d777d.jpg)  
Figure 4: Training dynamics on MMQA. (a) Proportion of discriminative question groups during training with our Deducibility Reward and its binary variant. (b,c) Mean number of triples per source output and output delivery failure rates for MMKG-R1 and MMKG-R1 w/o SGQ.

Readiness Rewards. (ii) Binary Deducibility. This variant removes the SGQ Reward and binarizes the Deducibility Reward by mapping both 0.5 and 1 to 1, while keeping 0 unchanged. It no longer distinguishes between incomplete and complete structured reasoning paths when the evidence supports the answer. We use Qwen3-VL-32B-Instruct-AWQ for answer generation and evaluation in this ablation study. As shown in Figure 3, full MMKG-R1 consistently outperforms both variants. Removing the SGQ Reward reduces accuracy, while binarizing the Deducibility Reward generally leads to further degradation. These results suggest that MMKG-level deducibility feedback alone does not fully substitute for source-grounded quality supervision. We further examine how our Deducibility Reward compares with its binary variant during training, and how adding SGQ affects the constructor’s outputs.

Deducibility Reward vs. Its Binary Variant. We first compare how our Deducibility Reward and its binary variant distinguish candidate graphs for the same question. A group is discriminative if at least three valid candidates remain and their task rewards are not all identical. Figure 4(a) reports the proportion of such groups among all sampled groups, with SGQ disabled in both variants. Our reward yields a higher proportion across all reported windows. Unlike the binary variant, it distinguishes incomplete from complete structured reasoning paths even when both support the answer, providing finer feedback for relative optimization.

Effect of SGQ. We next examine how adding SGQ affects the constructor’s outputs using two metrics: (1) Triples amount, defined as the average number of triples obtained from each source output after parsing and deduplication; and (2) Output delivery failure rate, defined as the proportion of source outputs that fail the structural, normal-termination, or nonempty-output requirements. Figure 4(b,c) compares these metrics throughout training between full MMKG-R1 and MMKG-R1 w/o SGQ. As training progresses, the full model retains more triples while exhibiting fewer output delivery failures. These observations complement the downstream accuracy gains, suggesting that SGQ helps preserve structured information and improve output reliability.

Together, these analyses support the complementary roles of the two rewards: our Deducibility Reward provides differentiated task feedback among candidates for the same question, while SGQ supplies additional quality supervision at the source level. Both contribute to training the constructor.

## 6 Conclusion

We introduce MMKG-R1, a novel reinforcement learning framework that trains a multimodal constructor to help build useful multimodal knowledge graphs (MMKGs) for downstream question answering. MMKG-R1 combines the graph-level Deducibility Reward with two knowledge-sourcelevel rewards, the Source-Grounded Quality Reward and the Graph-Readiness Reward, to provide complementary feedback on evidence sufficiency, source fidelity, and usability. Results across multiple datasets show that the multimodal constructor trained with MMKG-R1 can effectively extract knowledge to build useful MMKGs, while demonstrating strong generalization to unseen datasets.

## Limitations

The Deducibility and Source-Grounded Quality rewards rely on external multimodal large language models as judges, making their reliability dependent on the quality of the judge model’s assessments.

## Ethical considerations

This work aims to train a multimodal large language model to extract high-quality knowledge from multimodal sources for constructing multimodal knowledge graphs. All experiments are conducted based on publicly available datasets. We do not anticipate significant ethical or societal risks arising from this research.

## References

Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. In Findings ofthe Associationfor Computational Linguistics: ACL 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. Preprint, arXiv:2404.18930.

Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. 2022. MuRAG: Multimodal retrieval-augmented generator for open question answering over images and text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.

Sijun Dai, Qiang Huang, Xiaoxing You, and Jun Yu. 2026. Mg<sup>2</sup>-rag: Multi-granularity graph for multimodal retrieval-augmented generation. In European Conference on Computer Vision (ECCV).

Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, and Yong Liu. 2025. Benchmarking retrieval-augmented multimomal generation for document question answering. In Advances in Neural Information Processing Systems.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,

Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. Preprint, arXiv:2404.16130.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.

Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025a. Rag-anything: All-in-one rag framework. Preprint, arXiv:2510.12323.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025b. LightRAG: Simple and fast retrievalaugmented generation. In Findings ofthe Association for Computational Linguistics: EMNLP 2025.

Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A. Ross, and Alireza Fathi. 2023. Reveal: Retrievalaugmented visual-language pre-training with multisource multimodal knowledge memory. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, and Jiawei Han. 2025a. An empirical study on reinforcement learning for reasoning-search interleaved llm agents. Preprint, arXiv:2505.15117.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025b. Search-r1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing.

Wanying Liang, Pasquale De Meo, Yong Tang, and Jia Zhu. 2024. A survey of multi-modal knowledge graphs: Technologies and trends. ACM Computing Surveys, 56.

Jiashi Lin, Changhong Jiang, Xiangru Lin, Ruifei Zhang, Xinyi Zhu, Jiyao Liu, Cheng Tang, Ye Du, Shujian Gao, Junzhi Ning, Lihao Liu, Ziyan Huang, Tianbin Li, Jin Ye, and Junjun He. 2026. Evographr1: Self-evolving multimodal knowledge hypergraphs for agentic retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems.

Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, and Anh Tuan Luu. 2026. Graph-r1: Towards agentic graphRAG framework via end-to-end reinforcement learning. In Forty-third International Conference on Machine Learning.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. In Advances in Neural Information Processing Systems.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Preprint, arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In International Conference on Learning Representations.

Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021. Multimodalqa: complex question answering over text, tables and images. In International Conference on Learning Representations.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, and Yangqiu Song. 2026. Autograph-r1: End-to-end reinforcement learning for knowledge graph construction. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers).

Xueyao Wan and Hang Yu. 2025. Mmgraphrag: Bridging vision and language with interpretable multimodal knowledge graphs. Preprint, arXiv:2507.20804.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. Preprint, arXiv:2409.12191.

Xiaochen Wang, Zongyu Wu, Yuan Zhong, Xiang Zhang, Suhang Wang, and Fenglong Ma. 2026. Gpr: Empowering generation with graph-pretrained retriever. In Proceedings ofthe ACM Web Conference 2026.

Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun MA, and Ziwei Liu. 2026a. MMSearchr1: Incentivizing LMMs to search. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Zongyu Wu, Yilong Wang, Xiaochen Wang, Minhua Lin, Zhichao Xu, Fenglong Ma, Xiang Zhang, and Suhang Wang. 2026b. Multi-granularity contextenhanced rag over multimodal knowledge graphs. Preprint, arXiv:2608.25986.

Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng, Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, and Haibo Ding. 2026. Beyond correctness: Rewarding faithful reasoning in retrieval-augmented generation. Transactions on Machine Learning Research.

Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, et al. 2020. Mastering complex control in moba games with deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2024. A survey on multimodal large language models. National Science Review, 11(12).

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Vis-RAG: Vision-based retrieval-augmented generation on multi-modality documents. In The Thirteenth International Conference on Learning Representations.

Xu Yuan, Liangbo Ning, Qingqing Ye, Wenqi Fan, and Qing Li. 2026. mkg-rag: Leveraging multimodal knowledge graphs in retrieval-augmented generation for knowledge-intensive vqa. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Zhiqiang Zhang and Wen Zhao. 2025. A collaborative reasoning framework powered by reinforcement learning and large language models for complex questions answering over knowledge graph. In Proceedings ofthe 31st International Conference on Computational Linguistics.

## A Dataset Details

MMDocRAG. MMDocRAG (Dong et al., 2025) is designed to evaluate multimodal evidence retrieval and question answering over multimodal documents. MMDocRAG is licensed under the Apache License 2.0.

MultiModalQA. MultiModalQA (MMQA) (Talmor et al., 2021) is built from Wikipedia tables linked to text passages and images, and is designed to evaluate complex question answering involving cross-modal and multi-hop reasoning. In our experiments, text and images are preserved in their original forms, while structured tables are serialized into text. MMQA is publicly available, but no explicit license is specified.

MMLongBench-Doc. MMLongBench-Doc (Ma et al., 2024) evaluates long-document understanding involving information like text and images. We select answerable questions with valid supporting pages and evidence types that include figures or charts. This subset is used only for cross-dataset transfer evaluation with the constructor trained on MMDocRAG. MMLongBench-Doc is licensed under the CC BY-NC 4.0.

## B More Implementation Details

For constructor rollouts during the training phase, we set the sampling temperature to 1.0 and the maximum output length to 4,096 tokens. We build our

training pipeline using verl (Sheng et al., 2025) and use vLLM (Kwon et al., 2023) for answer model inference.

## C AI Usage Disclosure

The authors used generative AI tools to assist with writing and coding. All AI-assisted content and code were reviewed and verified by the authors