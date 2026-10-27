# Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives

Radha Krishna Deshpande, Ajoy Mondal, Akash Manna, and C V Jawahar

CVIT, International Institute of Information Technology, Hyderabad, India (radhakrishna.d,akash.manna)@research.iiit.ac.in , (ajoy.mondal,jawahar)@iiit.ac.in

Abstract. India’s historical public archives present severe computer vision and document image analysis challenges: decades of parliamentary records survive only as degraded, multi-column scanned pages featuring mixed scripts, complex layouts, and dense visual noise where OCR failure is the primary obstacle to information retrieval. We present an end-to-end framework that couples vision-based layout segmentation and multi-script OCR with an agentic retrieval-augmented generation (RAG) pipeline. Our approach specifically addresses how visual degradation and layout fragmentation propagate into text extraction errors, using an iterative agentic loop to navigate around OCR noise and layout ambiguities before generating answers. Evaluated on the 200,000-page Punjab Vidhan Sabha archive, our system demonstrates robust handling of historical document degradation, outperforming standard text-only baselines. The code is publicly available at GitHub Repo.

Keywords: Document Retrieval, Cross-lingual, Retrieval Augmented Generation, Large Language Models

## 1 Introduction

Historical legislative archives are dificult to process because the main challenge is often the document image itself. Faded text, irregular layouts, marginal notes, and mixed scripts can introduce errors during layout analysis and OCR, which then propagate to retrieval and question answering [21,22,29]. This is particularly challenging for large archives containing Hindi, Punjabi, and English documents with varying print quality and complex layouts.

Rather than treating OCR as a separate preprocessing step, we study the complete pipeline from document images to retrieval and answer generation. Our framework combines layout-aware OCR with an agentic retrieval loop that can check the retrieved evidence, reformulate the query, and search again when the available context is insuficient. This helps keep generated answers grounded in the original document evidence.

Retrieval-augmented generation (RAG) was introduced to combine language models with external knowledge sources [13] and has since been extended to iterative and agent-based retrieval [8, 20, 24]. Recent document retrieval methods have also moved toward vision-based representations, reducing reliance on

OCR for document retrieval [7, 18]. Multi-page document understanding further shows the importance of preserving visual information during retrieval and reasoning [6]. At the same time, OCR benchmarks have shown that multimodal language models still face dificulties with text recognition, particularly for multilingual and visually challenging documents [17], while recent work demonstrates that OCR errors can significantly afect downstream question answering [21]. Language-model-based OCR correction has therefore become an active research direction, especially for historical documents [3, 25].

These developments motivate an end-to-end approach for historical legislative archives in which visual processing, retrieval, and answer generation are evaluated together. We focus on the Punjab Vidhan Sabha proceedings, a large multilingual archive spanning 1950–2024. Our system uses open-weight models throughout the pipeline so that the complete workflow can be deployed locally and the retrieved evidence can be traced back to the source pages.

In this paper, we contribute:

– an agentic RAG framework that combines layout-aware OCR, hybrid retrieval, and iterative evidence checking for a multilingual legislative archive;   
– an evaluation of cross-lingual question answering on a large-scale historical document collection;

– an analysis of how OCR and layout errors afect retrieval and answer quality, together with an ablation of the agentic retrieval loop.

## 2 Related Work

Work on multilingual archive navigation sits at the intersection of several distinct lines of research: retrieval-augmented generation, agentic search, multilingual text embeddings, and multimodal document retrieval. None of these paradigms were originally optimized for a corpus precisely like ours, making it necessary to examine their contributions and limitations regarding noisy, multi-script legislative archives.

## 2.1 Agentic and Multilingual Retrieval-Augmented Generation

Retrieval-augmented generation originated from the premise that language models perform better when granted access to external knowledge rather than depending solely on parametric memory [13]. Gao et al. categorize RAG frameworks into naive, advanced, and modular designs, emphasizing that iterative query reformulation yields greater accuracy gains than merely scaling retriever capacity [8].

Agentic RAG advances this by treating retrieval as an adaptive, multi-step agent loop rather than a static query-response pipeline [20, 24]. Architectures that dynamically navigate between keyword indexing, dense semantic search, and document-level browsing ofer crucial flexibility for decades-old legislative records [1]. While agentic orchestration has been successfully demonstrated on historical collections like the CIA’s declassified archives, those implementations typically remained monolingual without addressing script-switching constraints [26].

Cross-lingual and multilingual RAG present distinct structural hurdles. Chirkova et al. demonstrated that maintaining output consistency requires robust prompt design rather than auxiliary translation steps [5], while Liu et al. observed performance degradation in cross-lingual retrieval tasks [15]. Conversely, Li et al. highlighted that incorporating multilingual evidence can stabilize reasoning in ambiguous contexts [14]. Influenced by these findings, our system indexes Hindi, Punjabi, and English sources concurrently. Foundation models such as Multilingual-E5-Large provide strong cross-lingual mappings, serving as an important precedent for our multi-script workflow [23, 28].

## 2.2 OCR and Historical Multilingual Document Processing

Historical document retrieval requires careful accounting for OCR degradation. Zhang et al. quantified how noise introduced during text extraction degrades downstream RAG performance [29], and Piryani et al. showed that these errors compound significantly in non-Latin and multi-script environments like Punjabi Gurmukhi and Hindi Devanagari [21].

Recent post-correction methodologies utilize LLMs to clean historical text and generalize better than traditional sequence models [25]. Bourne demonstrated that injecting domain-specific historical context further improves entity recognition for specialized legislative terminology [3]. Romein et al. confirm that structurally complex, multi-script documents remain a primary challenge in AIassisted archive digitization [22].

## 2.3 Multimodal and Vision-Language Document Retrieval

Traditional text-only pipelines often discard critical page layout structures such as tables, headers, and column boundaries, motivating a shift toward multimodal document retrieval. Ma et al. review this paradigm shift toward vision-language document encodings [19]. Frameworks such as ColPali [7], DSE [18], and advanced visual embedding architectures like NetraEmbed [12] encode document images directly into dense vector spaces, preserving layout semantics and layout structures across multiple languages. Cho et al. extended this to multi-page retrieval frameworks [6], while Liu et al. emphasize the integration of sparse lexical search with dense visual embeddings as an optimal trajectory for complex document repositories [16]. Our optimized pipeline leverages these insights by pairing robust multilingual textual dense representations with an adaptive agentic loop tailored for historical legislative document search.

## 3 Methodology

## 3.1 Query Management and Multi-Lingual Processing

The system initiates by receiving a user query, denoted as q, in its native script and language. Rather than relying on intermediate machine translation, our architecture processes the query natively against a multilingual semantic vector space, preserving domain-specific legislative terminology and regional context across Hindi, Punjabi, and English. Fig. 1 shows the pipeline: the query is embedded natively, the agent loop checks context suficiency and rewrites the query if needed, and the generator produces the grounded answer.

![](images/32f3f9a36c3007f5c2edaff3f58bc23ffadc75b25efb724fb141096afbfc158f.jpg)  
Fig. 1. Architecture of the proposed Agentic Retrieval-Augmented Generation (RAG) system. The framework features multi-lingual query processing, an iterative LLMdriven evaluation loop to ensure context suficiency, and a grounded response generation pipeline.

## 3.2 Ofline Processing and Real-Time Retrieval

The foundation of the retrieval system relies on a corpus of documents, $\mathcal { D } =$ $\{ d _ { 1 } , d _ { 2 } , \dots , d _ { n } \}$ . During the ofline phase, an embedding model maps these documents into a dense vector space alongside their associated metadata. Let $E ( \cdot )$ represent the multilingual embedding function. The resulting document embeddings are stored systematically within a Vector Database.

For real-time retrieval, the native query q is embedded directly to produce a query vector $v _ { q } = E ( q )$ . The system queries the Vector Database to retrieve the top-k most relevant context chunks, $C = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { k } \}$ , by calculating the similarity score between $v _ { q }$ and the document vectors $v _ { d _ { i } }$

$$
\mathrm { s i m } ( q , d _ { i } ) = \frac { v _ { q } ^ { \top } v _ { d _ { i } } } { \| v _ { q } \| _ { 2 } \| v _ { d _ { i } } \| _ { 2 } } .\tag{1}
$$

## 3.3 Iterative Agent Loop for Context Evaluation

A defining feature of this architecture is the autonomous Agent Loop, designed to mitigate hallucinations and ensure the retrieved context is adequately informative despite historical OCR degradation. Once the initial context C is retrieved,

a dedicated evaluation LLM analyzes its suficiency to answer q based on an information gain threshold function:

$$
\mathcal { G } ( C , q ) = \sum _ { j = 1 } ^ { | q | } \mathbb { I } ( w _ { j } \in \operatorname { T o k e n s } ( C ) )\tag{2}
$$

– If the context score satisfies the suficiency threshold, the workflow proceeds directly to the generation phase.

If the context is deemed insuficient, the agent triggers an adaptive query rewriting mechanism to generate an optimized query variant $q ^ { \prime } .$ . The optimization objective minimizes semantic ambiguity:

$$
q ^ { \prime } = \arg \operatorname* { m a x } _ { \tilde { q } } \mathrm { S i m } ( E ( \tilde { q } ) , E ( q ) ) - \lambda \mathcal { L } _ { \mathrm { O C R \_ n o i s e } } ( \tilde { q } )\tag{3}
$$

Here, λ balances semantic similarity to q against $\mathcal { L } _ { \mathrm { O C R \_ n o i s e } } ( \tilde { q } )$ , a conceptual penalty on query terms likely to be missed due to OCR errors in the indexed pages, which in practice is approximated by prompting the rewriting LLM to preserve the intent of q while rephrasing terms that failed to retrieve evidence. The system then loops back to the retrieval step using this refined query to fetch a more robust contextual foundation.

## 3.4 Grounded Response Generation

Upon validating context suficiency, the generation LLM synthesizes a grounded response directly in the user’s native language. This generation is strictly conditioned on both the native query and the validated context, maximizing factual grounding without translation intermediaries. The probability of generating the response sequence of tokens w can be modeled as:

$$
P ( R \mid q , C ) = \prod _ { t = 1 } ^ { T } P ( w _ { t } \mid w _ { < t } , q , C )\tag{4}
$$

## 3.5 Algorithm: Native-Language Agentic RAG Loop

Algorithm 1 Native-Language Multi-Step Agentic Retrieval   
Require: User query q, Vector DB collection , Maximum loops L   
Ensure: Grounded response R in native language   
1: current\_query  q   
2: for step = 1 to L do   
3: C  .query(current\_query, k = 5)   
4: is\_suff icient, eval\_log  LLM\_Evaluate(current\_query, C[0])   
5: if is\_sufficient or step == L then   
6: break   
7: end if   
8: current\_query  LLM\_Rewrite(current\_query, eval\_log)   
9: end for   
10: R LLM\_Generate(q, C)   
11: return R

## 4 Experiments and Results

## 4.1 Dataset

The dataset used in this research consists of Punjab Vidhan Sabha (legislative assembly of Punjab state) documents collected from archival records spanning from the 1950s to 2024. The collection includes assembly proceedings, debates, reports, notices, and other oficial legislative documents published across diferent time periods.

The records were digitized from bound volumes held by [https://pvs.neva. gov.in/], a process that was carried out incrementally over several decades rather than in a single digitization efort. This matters for the corpus’s characteristics: pages from the 1950s and 1960s were typically typeset on older printing equipment and later scanned at lower resolution, often with visible ink bleedthrough, faded typeface, and skewed page boundaries, while more recent decades (post-2020) are comparatively cleaner but still exhibit dense multi-column layouts and inconsistent formatting across sessions.

Documents in the corpus are predominantly in Hindi (Devanagari) and Punjabi (Gurmukhi), with English appearing throughout, particularly in earlier-era records and administrative headers; it is common for a single page to mix two or all three languages, for instance an English-language question header followed by a Punjabi response, or bilingual tabular data. The collection totals approximately 200,000 pages a covering assembly sessions, starred and unstarred questions, budget and financial statements, committee reports, and procedural notices. Fig. 2 shows representative pages illustrating the multi-column layouts, mixed scripts, and visual degradation typical of the corpus.

Given the scale of the archive, we did not attempt to hand-annotate the full collection. Instead, we constructed a stratified evaluation benchmark of 1,000 pages and 200 gold-annotated question-answer pairs, sampled to balance representation across decade, script, and document type, so that evaluation results reflect the corpus’s actual heterogeneity rather than only its cleaner, more recent portion. To isolate the contribution of the vision pipeline specifically, we additionally hand-transcribed a smaller subset of 20 pages, selected to span a range of degradation levels, as ground truth for OCR quality evaluation.

## 4.2 Experimental Setup

We ran our experiments on a single Nvidia RTX-4060 (16 GB VRAM). Our evaluation framework implemented a robust local storage backend using ChromaDB alongside an adaptive agentic retrieval architecture driven by BAAI/bge-m3 for dense and hybrid token-level multilingual embeddings (supporting English, Punjabi Gurmukhi, and Hindi Devanagari).

As described in Section 3.5, the retrieval index spans the full 200,000-page Punjab Vidhan Sabha archive. All quantitative results in this section are computed against our stratified evaluation benchmark of 1,000 pages and 200 goldannotated queries, drawn from this corpus and balanced across decade, script, and document type. Additionally, to isolate vision-side performance, we handtranscribed a subset of 20 pages and evaluated multiple layout segmentation and OCR model combinations against this ground truth.

![](images/23ecd811de9092bdeb064c22e5342b1659a0a18e9d18bcab4cdac9fbb19419fd.jpg)  
Fig. 2. Overview of dataset pages featuring tabular data, multi-scripts, and faded text characteristics.

## 4.3 Metrics

We evaluate the proposed system at three levels: OCR recognition, document retrieval, and answer quality.

OCR Recognition: We use Character Recognition Rate (CRR) and Word Recognition Rate (WRR) to evaluate OCR performance. Both measures are derived from the Levenshtein edit distance, which counts substitutions, deletions, and insertions required to transform the predicted sequence into the reference sequence [9, 10].

The Character Recognition Rate is defined as

$$
\mathrm { C R R } = 1 - \frac { S _ { c } + D _ { c } + I _ { c } } { N _ { c } }\tag{5}
$$

$$
= \frac { N _ { c } - S _ { c } - D _ { c } - I _ { c } } { N _ { c } } ,\tag{6}
$$

where $S _ { c } , D _ { c } .$ , and $I _ { c }$ denote the number of character substitutions, character deletions, and character insertions, respectively, and $N _ { c }$ denotes the number of characters in the reference text.

Similarly, the Word Recognition Rate is defined as

$$
\mathrm { W R R } = 1 - \frac { S _ { w } + D _ { w } + I _ { w } } { N _ { w } }\tag{7}
$$

$$
= \frac { N _ { w } - S _ { w } - D _ { w } - I _ { w } } { N _ { w } } ,\tag{8}
$$

where $S _ { w } , D _ { w } ,$ and $I _ { w }$ denote the number of word substitutions, word deletions, and word insertions, respectively, and $N _ { w }$ denotes the number of words in the reference text [9, 10].

Retrieval: We evaluate retrieval efectiveness using Mean Reciprocal Rank (MRR) and Hit Rate at k (Hit@k). MRR measures the reciprocal rank of the highest-ranked relevant document for each query and averages this value over all queries [27]:

$$
\mathrm { M R R @ 1 } = \frac { 1 } { | Q | } \sum _ { i = 1 } ^ { | Q | } \frac { 1 } { \mathrm { r a n k } _ { i } } .\tag{9}
$$

Hit@k measures the proportion of queries for which at least one relevant document is retrieved within the top-k ranked results:

$$
\mathrm { H i t @ k } = \frac { 1 } { | Q | } \sum _ { i = 1 } ^ { | Q | } \mathbb { I } ( \mathrm { r a n k } _ { i } \leq k ) .\tag{10}
$$

Here, $Q$ denotes the set of evaluation queries, |Q| denotes the number of queries, rank<sub>i</sub> denotes the rank of the first relevant document for query $i ,$ and I(·) is the indicator function that returns 1 when the condition is satisfied and 0 otherwise.

Answer Quality: We use the F1 score, which combines Precision and Recall through their harmonic mean:

$$
F 1 = \frac { 2 P R } { P + R } ,\tag{11}
$$

where P denotes Precision and R denotes Recall. Precision measures the proportion of predicted answer tokens that are correct, while Recall measures the proportion of reference answer tokens that are correctly identified. The F1 score therefore provides a balanced measure of answer quality [4].

## 4.4 Retrieval Performance Comparison

Our primary goal was to evaluate and compare the retrieval performance of various multilingual embedding models and our proposed optimization framework on the domain-specific Punjab Vidhan Sabha (PVS) benchmark.

As shown in Table 1, token-level and smaller cross-lingual models like MuRIL and LaBSE struggle heavily with the complex, multi-script historical documents, yielding near-zero or low MRR@1 scores. Larger dense retrievers such as Multilingual-E5-Large and NetraEmbed scale better, achieving Hit@3 scores above 0.81. However, BGE-M3 and our optimized pipeline establish the strongest baseline and performance standard, successfully navigating historical OCR degradation and mixed-script text to secure top-tier retrieval accuracy.

Fig. 3 shows the full pipeline from query to grounded answer , and Fig. 4 compares the pages retrieved by each model for one Punjabi query.

Table 1. PVS Retrieval Evaluation on the 1,000-page / 200-query stratified benchmark (indexed against the full 200,000-page archive).
<table><tr><td>Model &amp; Pipeline Setup</td><td>Eval. Benchmark</td><td></td><td>MRR@1 Hit@3 Hit@5 Prec@3</td><td></td><td></td></tr><tr><td>MuRIL</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.000</td><td>0.005</td><td>0.015 0.002</td></tr><tr><td>LaBSE</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.320</td><td>0.510 0.590</td><td>0.170</td></tr><tr><td>Embedding-gemma (300M)</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.525</td><td>0.735 0.775</td><td>0.245</td></tr><tr><td>NetraEmbed (dim=768)</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.580</td><td>0.816 0.890</td><td>0.272</td></tr><tr><td>NetraEmbed (dim=1536)</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.598</td><td>0.835 0.905</td><td>0.278</td></tr><tr><td>NetraEmbed (dim=2560)</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.605</td><td>0.843 0.912</td><td>0.281</td></tr><tr><td>Multilingual-E5-Large</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.620</td><td>0.855 0.915</td><td>0.285</td></tr><tr><td>Standard BGE-M3</td><td>1,000 Pages / 200 Queries</td><td></td><td>0.775</td><td>0.900 0.940</td><td>0.300</td></tr><tr><td>Ours (Optimized Pipeline) 1,000 Pages / 200 Queries</td><td></td><td></td><td>0.810</td><td>0.925 0.955</td><td>0.325</td></tr></table>

![](images/95ccab4cffc2bd974c5001f89e839d7f8b7f8f31387646cb73e73e85d156bed1.jpg)  
Fig. 3. End-to-end cross-lingual RAG pipeline for legislative documents, from the user query and document retrieval to grounded answer generation in Punjabi and translated to english for the reference.

<table><tr><td>Query</td><td>Embedding model</td><td>Retrived document</td><td></td><td>Actual Relavant document</td></tr><tr><td>T1948 E き すさi 3Hf ?</td><td>multilingual- e5-large</td><td>sasdyeg.mke [Miuister for Relief and Rehabilitation] Edd asedrawsa a ea r cl deca le i of  t s the toe o a    ta Eds d aa Fory o ihe Dep East Panjab. </td><td>Shri Biim Sen Sacbar : (间) + Dintrint. 1938 totot ca Himar Rohntak Gurgson</td><td>(a) to (d) : The required inforimation is given below : meumemo 1939 1946 1 1947 1948 1949 1950 è 12 30 15 17 12 11 10</td></tr><tr><td>T1948 E Tmき すさ f 可?</td><td>bge-m3</td><td>770 EAST PUNJAB LEGISLATIVE ASSEMBLY [30th March, 1948 The Hon. Shri Prithvi Singh Azad : (a) Yes. Shri Prabodh Chaadra : May I know why the license was renewed within Minister : I have alresdy given my reply to it. Shri Prabodh Chandra : May I know as to why action was not takem against those from whom unlicensed liquor was recovere Minister : Simply because they were refugees and they gave assurances that th wod nt  s things in future</td><td>Shri Bhim Sen Sacbar (a) + Dintriet. 1918 Haar Ilohitak Gupson</td><td>(a) to (d) : The required inforimation is given below : 1939 1946 1947 1948 1949 1 1950 20 12 30 15 17 2 11 10</td></tr><tr><td>T1948 西E Tす すせ 3f 可?</td><td>labse</td><td>(c) the target date by which training in the said institute is expected to start ; (d) the total number of trainees, trade-wise, who are expected to receive training in each of the said training institutes information is laid on the Table of the House. Shri Harbans Lal (Deputy Minister) : A statement giving the requisite geis Statement he S irecto 0.19 Place Distrier</td><td>Shri Biim Sen Sachar : (a) + Distriet. 1918 1939 Hiaste 20 Ioltak 12</td><td>()Thd   lw 1946 1947 1948 1949 1950 1 12 30 15 17 11 10</td></tr><tr><td>HT 1948 き すさi mf ね可話?</td><td>Netra embedding [2560 dim]</td><td>STARRED QUESTIONS AND ANSWERS   (  )          &#x27;  ,  a      z a f đe  fa 21 , 17s  ał  6/43 at e, 12 .          gra&#x27;s wüts far atat fap t    la    è wrent et faa er fa atar fan   fea iè  wrn st. . a.  a  a a    fe e r t fa few fe aara è n fé t fa fe</td><td>(18)1 Shri Biim Sen Sachar : (a) FOB-F • Ditriez. 1938 1939 Himer Roltsk Gugaon</td><td>(a) to (d) : The requirrd information is given below : Tad   d  a e 1946 1947 1948 1949 1950 11 . 4 12 30 15 17 12 11 10 16</td></tr><tr><td>T1948 すさ 3f 可話?</td><td>embedding gemma- 300m</td><td>762 EAST PUNJAB LBGISLATIVE ASSEMBLY [30th March, 1948 seaxg.mke [Minister for Relief and Rehabilitation] ch vehicies into bPdd r  P  to ad aseedranve ad b is 1948, Cos  De s</td><td>Shri Biim Sen Sachar (a) • Diatria. 1938 1939 3 of tb Hiwoer Roltsk Gurgn</td><td>(a) to (d) : The required inlormstion in given below : deumamen 1946 1947 1948 1949 1950 n Sa s2. 1951. 1952 20 30 15 17 12 11 10</td></tr><tr><td>T1948 西 Tき すせ m3f 可?</td><td>OUR PROPOSED SYSTEM</td><td>Shri Biim Sen Sachar : (a) to (d) : The required inforinstion is given below : (a) • DHatriut. 1938 1939 1946 1947 1948 1949 1950 1951. 1952 t M . Himer a0 12 15 11 Rohtk 12 11 10 Gurgaon 11 10 99</td><td>Shri Biim Sen Sachar : (a) • Diatrikez. 1938 Histr Roltak Gugasn</td><td>16 (a) to (d) : The required inforimstion in given belaw : deurmamves 1939 1946 1947 1948 1949 1950 1 20 12 15 17 12 11 10 11 16</td></tr></table>

Fig. 4. Qualitative comparison of cross-lingual embedding models for historical legislative document retrieval. The same Punjabi query is given to multilingual-e5-large, BGE-M3, LaBSE, Netra embedding (2560 dimensions), and EmbeddingGemma-300M. The retrieved document from each model is compared with the ground-truth relevant document shown in the rightmost column. The retrieved results demonstrate diferences in cross-lingual semantic matching, with several models returning visually and semantically unrelated legislative records. In contrast, our proposed system retrieves the relevant document containing the required information, demonstrating its efectiveness for retrieval from multilingual historical archives.

## 5 Ablation

## 5.1 OCR and Layout Pipeline Evaluation

Our production pipeline combines a fine-tuned YOLO [11] layout model with our CRNN-based OCR recognizer. We compare it with a CRAFT-based layout model [2] using the same CRNN recognizer, along with Google OCR and Surya OCR.

Table 2. OCR pipeline comparison on the 20-page ground-truth subset.
<table><tr><td>Layout Model</td><td>OCR Model</td><td>CRR (%) WRR (%)</td><td></td></tr><tr><td>Gemini-3.1</td><td>Gemini-3.1</td><td>99.7</td><td>89.6</td></tr><tr><td>Surya2</td><td>Surya2</td><td>99.6</td><td>88.9</td></tr><tr><td>Sarvam</td><td>Sarvam</td><td>99.5</td><td>89.4</td></tr><tr><td>YOLO (Fine-tuned) CRNN (Ours)</td><td></td><td>95.67</td><td>80.03</td></tr><tr><td>CRAFT</td><td>CRNN (Ours)</td><td>89.00</td><td>60.35</td></tr></table>

The fine-tuned YOLO + CRNN pipeline achieves 95.67% CRR and 80.03% WRR, compared with 89.00% CRR and 60.35% WRR for $\mathrm { C R A F T } + \mathrm { C R N N } .$ Since the same CRNN recognizer is used in both pipelines, the improvement shows the efect of the layout model on the quality of the extracted text. Google OCR and Surya OCR achieve higher recognition scores, while our pipeline remains fully local and provides greater control for processing the large archive.

Table 3. PVS Retrieval Evaluation using diferent OCR models on the 200-query benchmark.
<table><tr><td colspan="4">OCR Model MRR@1 Hit@3 Hit@5 Prec@3</td></tr><tr><td>Gemini-3.1 0.825</td><td>0.920</td><td>0.960</td><td>0.3533</td></tr><tr><td>Sarvam 0.820</td><td>0.935</td><td>0.955</td><td>0.325</td></tr><tr><td>Surya2 0.815</td><td>0.925</td><td>0.950</td><td>0.305</td></tr><tr><td>Ourocr 0.810</td><td>0.925</td><td>0.955</td><td>0.325</td></tr></table>

## 5.2 Ablation on Agent Loop Iterations

To quantify the contribution of the iterative agentic loop, we vary the maximum number of retrieval–rewrite iterations using a fixed 200-query evaluation dataset, $L \in \{ 1 , 2 , 3 \}$ , and measure retrieval quality (MRR@1, Precision@3) alongside end-to-end answer correctness (Answer F1-Score) on the same 200-query benchmark. L = 1 corresponds to a single-pass, non-agentic retrieval baseline with no query rewriting.Fig. 5 illustrates how iterative query rewriting corrects retrieval errors. In the first two iterations, the system retrieves irrelevant documents, but after further query refinement, the third iteration successfully retrieves the correct document containing the required information.

As shown in Fig. 6, increasing L from 1 to 3 improves MRR@1 from approximately 0.89 to 0.97 and Precision@3 from approximately 0.33 to 0.37, but the efect is most pronounced on Answer F1-Score, which rises from approximately 0.18 to 0.35 — nearly a 2× relative gain. This asymmetry indicates that the agentic loop’s primary benefit is not marginal re-ranking of already-retrieved passages, but recovering answer-relevant context that single-pass retrieval misses entirely, consistent with our claim that iterative query reformulation is essential for navigating OCR noise and layout ambiguity in this corpus.

To assess how well diferent instruction-tuned language models comprehend and adapt to the domain-specific historical text and OCR-induced artifacts within our legislative archive corpus, we evaluate the average perplexity scores across three prominent backbones: Mistral-7B, Phi-3, and Qwen2.5 as shown in Fig. 7.

![](images/4d6d73f7da0b2008e6c7e5737c7a998b066c733753911e67ecc97af18d30688d.jpg)  
Fig. 5. Iterative retrieval correction through query rewriting. The first two iterations retrieve irrelevant same documents, while the third iteration retrieves the correct document containing the relevant information. The example demonstrates a cross-lingua setting, where the query is in Punjabi and the retrieved document is in English.

![](images/93d09609dee21e41156afcc15dad2cb0540a86e66d854f01aa0a05b830cb1e2b.jpg)

![](images/3677c5fff61bdd48c448dee52eb5e61d0a7139d1dda3df8757723eed1392f109.jpg)  
Fig. 6. Performance metrics vs Iteration loops of the proposed system across diferent agent loop iterations. The results show how the system performance changes as the number of iterations increases.  
Fig. 7. Average Perplexity scores of the language models on multi-script documents. Qwen2.5 achieves the lowest perplexity (4.46), followed by Phi-3 (4.99) and Mistral-7B (7.81).

## 5.3 Limitations

Despite the strong performance of our multilingual agentic RAG framework, several limitations remain. Severe OCR errors in degraded pages can still afect dense retrieval and entity matching. The iterative agentic loop and LLM-based suficiency checks also increase inference latency compared with standard retrieval. In addition, complex financial tables and unstructured tabular data may lose hierarchical structure during text-to-chunk conversion. Future work will focus on improving OCR robustness, reducing inference latency, and preserving complex table structures.

## 6 Conclusion

In this work, we introduced an Agentic-RAG framework designed to bridge the gap between complex historical archives and user information needs. By blending iterative query rewriting, adaptive document retrieval, and vision-language generation, our approach efectively tackles the nuances of multilingual document search.

Evaluated on the Punjab Vidhan Sabha debates dataseta challenging collection spanning multiple decades in Punjabi, Hindi, and Englishthe system demonstrated robust resilience against real-world hurdles such as historical vocabulary shifts, degraded text, and severe OCR noise. Unlike traditional, rigid RAG pipelines, the iterative loop implemented here acts as a self-correcting mechanism, dynamically verifying context suficiency to minimize hallucinations and ground final answers directly in source material.

Ultimately, this study highlights that agentic workflows can significantly enhance information discovery in multilingual public records. Beyond historical text exploration, the principles demonstrated here pave the way for more dependable, transparent, and accessible digital archives for researchers, policymakers, and the public.

Future work will focus on multi modal retrieval, expanding the system to more Indian languages and government archives, and supporting conversational search with long-context reasoning. User studies with researchers and other archive users will further evaluate its usability and reliability.

Acknowledgment : This work is supported by MeitY, Government of India, through the NLTM-Bhashini project.We also sincerely thank Prof. Gurpreet Singh Lehal(gslehal@gmail.com) for his valuable support and assistance during the project evaluation.

## References

1. Ayanami, et al.: A-RAG: Scaling agentic retrieval-augmented generation via hierarchical retrieval interfaces. arXiv preprint arXiv:2602.03442 (2025)

2. Baek, Y., Lee, B., Han, D., Yun, S., Lee, H.: Character region awareness for text detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9365–9374 (2019)

3. Bourne, J.: Clocr-c: Context leveraging ocr correction with pre-trained language models (2025), https://arxiv.org/abs/2408.17428

4. Chinchor, N.: Muc-4 evaluation metrics. In: Proceedings of the Fourth Message Understanding Conference (1992)

5. Chirkova, N., et al.: Retrieval-augmented generation in multilingual settings. arXiv preprint arXiv:2407.01463 (2024)

6. Cho, H., et al.: M3DocRAG: Multi-modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952 (2024)

7. Faysse, M., et al.: ColPali: Eficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449 (2024)

8. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., Wang, H.: Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2024)

9. IIIT Hyderabad: Ihtr 2023: Indian handwritten text recognition. Indian Language OCR Competition (2023), https://ilocr.iiit.ac.in/ihtr/task.html

10. IIIT Hyderabad: Icdar 2024 rdtag: Evaluation metrics. ICDAR 2024 Robust Document Text and Graphics (2024), https://ilocr.iiit.ac.in/icdar\_2024\_rdtag/ task.html

11. Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics yolo (2023), https://github.com/ ultralytics/ultralytics

12. Kolavi, A.S., Jain, V.: M3dr: Towards universal multilingual multimodal document retrieval (2025), https://arxiv.org/abs/2512.03514

13. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented generation for knowledge-intensive nlp tasks. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 33, pp. 9459–9474 (2020)

14. Li, B., et al.: BordIRLines: A dataset for geopolitical border disputes in multilingual rag. arXiv preprint arXiv:2410.01171 (2024)

15. Liu, W., Trenous, S., Ribeiro, L.F.R., Byrne, B., Hieber, F.: XRAG: Cross-lingual retrieval-augmented generation. In: Findings of the Association for Computational Linguistics: EMNLP 2025 (2025)

16. Liu, Y., et al.: Roles of mllms in visually rich document retrieval for rag: A survey. In: Proceedings of IJCNLP 2025 (Long Papers) (2025)

17. Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin, L., Bai, X.: Ocrbench: On the hidden mystery of ocr in large multimodal models. Science China Information Sciences 67(12) (2024)

18. Ma, X., et al.: DSE: Document screenshot embeddings. arXiv preprint arXiv:2406.11251 (2024)

19. Ma, X., et al.: Scaling beyond context: A survey of multimodal retrieval-augmented generation for document understanding. arXiv preprint arXiv:2510.15253 (2025)

20. Mishra, S., Niroula, S., Yadav, U., Thakur, D., Gyawali, S., Gaire, S.: SoK: Agentic retrieval-augmented generation (RAG): Taxonomy, architectures, evaluation, and research directions. arXiv preprint arXiv:2603.07379 (2025)

21. Piryani, B., Mozafari, J., Jatowt, A.: MultiOCR-QA: A multilingual dataset for qa on ocr-processed texts. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2038–2048 (2025)

22. Romein, C.A., et al.: Transforming historical newspaper research and preservation through AI: A global perspective. Publications 7(1), 10 (2025)

23. Sener, S.: Advanced retrieval augmented generation: Multilingual semantic retrieval across document types by fine-tuning transformer based language models and OCR integration. Engineering and Technology Journal 9(7), 4404–4411 (2024)

24. Singh, A., et al.: Agentic retrieval-augmented generation: A survey on agentic RAG. arXiv preprint arXiv:2501.09136 (2025)

25. Thomas, A., Gaizauskas, R., Lu, H.: Leveraging llms for post-ocr correction of historical newspapers. In: Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024. pp. 116–121. ELRA and ICCL (2024)

26. Černý, J., Avramov, K., Pendse, L.: A multi-stage agentic AI system for extracting information from large digital archives: Case study on the czechoslovak year 1968 in CIA’s FOIA collection. The Electronic Library (2026). https://doi.org/10. 1108/EL-06-2025-0272

27. Voorhees, E.M.: The trec-8 question answering track report. In: Proceedings of the Eighth Text REtrieval Conference (TREC-8) (1999)

28. Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F.: Multilingual E5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 (2024)

29. Zhang, R., et al.: OHRBench: A benchmark for ocr-induced noise efects on rag pipelines. arXiv preprint (2024)