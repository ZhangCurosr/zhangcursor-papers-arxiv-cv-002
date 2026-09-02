# MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

Debanjan Mahata, Atharva Tendle, Daniel Preo¸tiuc-Pietro, Yong Zhuang, Ozan <sup>˙</sup>Irsoy Bloomberg, NYC, USA

{dmahata,atendle,dpreotiucpie,yzhuang52,oirsoy}@bloomberg.net

## Abstract

Retrieval over visually rich documents has a representation problem: important content often lives in tables, charts, figures, and layout relations that plain OCR linearizes, corrupts, or omits. ColPali-family visual retrievers address this with patch-level multi-vector indexes and late-interaction scoring, keeping imagederived retrieval on the query-time serving path. We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing that shifts multimodal reasoning to index time. During ingestion, a multimodal LLM converts rendered pages into verified textual fields that are indexed with BM25F and optionally fused with dense retrieval, enabling text-centric serving over multimodally grounded evidence. On Vi-DoRe V3, MIDR Hybrid achieves 0.6219 average nDCG@10 across five English domains, a 23.0% relative gain over BM25, remaining competitive with ColQwen2.5. On two Frenchdocument domains, enrichment bridges English queries and French page text, lifting BM25 from 0.1532 to 0.5448 nDCG@10 and outperforming ColQwen2.5. Across all seven domains, MIDR leads ColQwen2.5 on four while using ∼9× smaller index memory and approximately 2× lower query latency. These results establish index-time multimodal reasoning as a compelling accuracy–deployment alternative to serving-time visual late interaction.

## 1 Introduction

Visually rich documents often express important information outside ordinary running text. In enterprise reports, filings, manuals, slide decks, and scientific documents, key facts and claims may be encoded in tables, charts, figures, captions, visual grouping, or page layout (Loison et al., 2026; Dong et al., 2025): a table value depends on row and column headers, a chart trend on axes and legends, and a slide-level claim on visual grouping. Plain OCR linearizes these structures into a flat character stream, discarding the cues that make their content searchable (Zhang et al., 2025a).

This matters because Retrieval-Augmented Generation (RAG) is increasingly used for question answering over large enterprise document collections (Lewis et al., 2020; Fan et al., 2024; Ma et al., 2025), and its effectiveness depends on whether the retriever surfaces the right evidence in the first place. Even as context windows expand (Liu et al., 2025a), retrieval remains necessary to reduce inference cost (Izacard and Grave, 2021; Karpukhin et al., 2020) and to search collections that exceed practical context budgets (Qiu et al., 2025; Liu et al., 2025b).

The challenge is especially acute in specialized domains such as finance (Chen et al., 2021), legal analysis (Kulkarni et al., 2026), medicine (Singhal et al., 2025), and education (Alawwad et al., 2025), where critical evidence is frequently layoutdependent and visually structured (Loison et al., 2026; Cho et al., 2025; Dong et al., 2025). Effective retrieval over such documents therefore requires going beyond extracted text alone (Cho et al., 2024; Tanaka et al., 2025).

The state-of-the-art response has been to retrieve over rendered page images directly. Col-Pali (Faysse et al., 2025) adapts a PaliGemma backbone with a late-interaction retrieval head (Khattab and Zaharia, 2020), encoding each rendered page into patch-level multi-vector representations scored with MaxSim. ColQwen2.5 extends this ColPali-style recipe to a Qwen2.5-VL backbone and achieves strong open-weight performance on ViDoRe V3 (Loison et al., 2026). This line of work establishes an important lesson: visually rich document retrieval requires multimodal understanding. It also makes a particular design choice: because OCR text is insufficient, the retrieval stack should serve queries over rendered-page image representations using large visual multi-vector indexes and late-interaction scoring.

This design is powerful, but it keeps visual retrieval on the serving path. That distinction matters for practical RAG deployments, where indexing and querying have different cost profiles. Documents are typically indexed in offline ingestion pipelines, whereas the resulting index may be searched repeatedly by many users and increasingly by agentic workflows that issue multiple retrieval calls per request (Dong et al., 2026). Multimodal reasoning performed during indexing can therefore be amortized over future queries, while serving retrieval over visual multi-vector indexes carries recurring query-time costs: large image-derived indexes, a compatible multimodal query encoder, and late-interaction scoring over candidates.

Recent multimodal large language models (MLLMs) make it possible to revisit this design choice. MLLMs can follow structured instructions, condition jointly on rendered page images and extracted text, and interpret layout-dependent evidence such as tables, charts, figures, and visually grouped content (Zhang et al., 2024; Qwen Team, 2026). This paper explores an alternative use of that capability: rather than serving every query over rendered-page image representations, use MLLMs during ingestion to convert visual and layout-dependent evidence into retrieval-ready textual fields. Thefailure ofOCR-only retrieval motivates multimodal document understanding; it does not necessarily require serving-time retrieval over rendered-page image representations.

Much of the information a page provides to a retriever — entities, claims, quantities, table structure, chart encodings, layout relations, captions, and visually grounded textual context — is queryindependent and a property of the document itself. If these properties can be inferred during ingestion and materialized as structured textual evidence, then query-time retrieval can remain text-centric while still benefiting from multimodal document understanding.

We call this design enrichment-augmented indexing: use rendered pages for multimodal reasoning at index time, but serve retrieval through lexical, dense, and hybrid text-search infrastructure.

We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing. As shown in Figure 1, MIDR separates enrichment from indexing and retrieval: document- and page-level enrichments are produced offline, then indexed as BM25F fields and dense vectors for text-centric serving. A document-level pass first produces document enrichments, including document type, primary focus, and main entities. A page-level pass then uses these document enrichments together with each rendered page image and extracted page text to produce page-level enrichments: layout and quality signals, table and chart descriptions, domain tags, keyphrases, and coarse- and fine-grained QA pairs. Enrichment proceeds through an extract– verify–refine loop: page-level fields are first generated, then audited for grounding and consistency against the rendered page and extracted text, and finally revised only where verification identifies unsupported or inconsistent content. The verified fields are then served through lexical, dense, and hybrid text retrieval (Section 3), so that at query time retrieval operates over multimodally grounded textual evidence, without processing page images or maintaining a visual multi-vector index.

On ViDoRe V3 (Loison et al., 2026), MIDR Hybrid reaches 0.6219 average nDCG@10 across the five English-document domains, a 23.0% relative gain over the raw-BM25 baseline of 0.5057, while remaining competitive with our ColQwen2.5 reproduction, with almost the entire remaining gap concentrated in computer science (Table 2). On the two French-document domains, the same pipeline acts as a cross-lingual bridge: raw BM25 collapses to 0.1532 nDCG@10 because English queries share little surface vocabulary with French page text, while MIDR Hybrid reaches 0.5448 and leads ColQwen2.5 on both domains by generating English enrichments at index time (Section 5). It does so from an index roughly 9× smaller than ColQwen2.5’s visual multi-vector index, at 1.1– 2.6× lower query latency depending on domain (Table 4). We additionally quantify the one-time ingestion cost and evaluate sensitivity to the enrichment MLLM.

The ablations indicate that MIDR works by exposing distinct types of retrieval evidence rather than by adding undifferentiated generated text: QA pairs create lexical query–page bridges, keyphrases support dense semantic matching, and table summaries provide targeted gains on table pages, with the largest improvements where OCR loses the most structure (Tables 6 and 7). A QA-only configuration nearly matches the full system on the English domains (0.6200 vs. 0.6219 nDCG@10), while withholding the rendered page image reduces aggregate performance by 0.0113, with the effect concentrated in OCR-hard domains. The comparison with visual retrieval also clarifies the boundary of the approach: multi-vector retrieval over rendered pages remains stronger on equations, diagrams, code layout, and fine-grained visual disambiguation, and a per-query oracle over the two systems reaches 0.7042 nDCG@10 (Appendix M), so the two encode complementary rather than redundant evidence.

![](images/6f8c9833434318c775b14b4ca768f84f6efdd6bc9b7a03d8f55e8431ab316f97.jpg)  
Figure 1: MIDR end-to-end pipeline. (a) Enrichment pipeline: documents are decomposed into rendered page images and extracted page text; a document-level MLLM pass over the first five pages produces document enrichments, which condition page-level extract–verify–refine enrichment. (b) Indexing and retrieval: verified enrichment outputs are indexed as BM25F fields and dense vectors, then queried through lexical, dense, or hybrid retrieval with Reciprocal Rank Fusion, without serving-time visual multi-vector retrieval.

## Contributions and Key Findings

• We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing: a deployment alternative to ColPali-style visual retrieval that performs multimodal understanding over rendered pages at index time, then serves verified textual retrieval fields through BM25F, dense, and hybrid text-centric infrastructure.

• On ViDoRe V3, MIDR Hybrid reaches 0.6219 average nDCG@10 on the English domains, a 23.0% relative gain over raw BM25, remaining competitive with our ColQwen2.5 reproduction, from an index roughly 9× smaller than ColQwen2.5’s.

• On the two French-document ViDoRe V3 domains, index-time enrichment acts as a crosslingual bridge: MIDR Hybrid lifts BM25 from 0.1532 to 0.5448 nDCG@10 by translating layout-grounded evidence into English at ingestion time and leads ColQwen2.5 on both (Section 5).

• Field-, page-, query-, MLLM-, and oraclelevel analyses, including QA-only and pageimage ablations, show when enrichment helps and where visual retrieval remains stronger; a per-query oracle reaches 0.7042 nDCG@10, indicating that index-time enrichment and visual late interaction encode complementary evidence types.

## 2 Related Work

MIDR builds on standard text retrieval infrastructure while addressing a representation gap that text-only retrievers cannot close by themselves. BM25 and BM25F (Robertson et al., 1994, 2004) remain strong sparse baselines, dense biencoders (Karpukhin et al., 2020) improve semantic recall, and hybrid fusion such as Reciprocal Rank Fusion (Cormack et al., 2009) often combines their strengths. Learned sparse models such as SPLADE (Formal et al., 2021, 2022) further expand textual matching surfaces. These methods improve retrieval once evidence is present in the index, but they do not recover visual or layoutdependent evidence lost when OCR flattens tables, charts, and figures.

A related line of work enriches documents before indexing. doc2query and docTTTTTquery (Nogueira et al., 2019) generate synthetic queries, Doc2Query++ (Kuo et al., 2025) improves coverage and fusion, and recent LLMbased systems such as EnrichIndex (Chen et al., 2025) and IndexRAG (Bao and Shi, 2026) move query-independent reasoning offline through summaries, QA pairs, or bridging facts. Closest to MIDR, PREMIR (Choi et al., 2025) uses a multimodal LLM to generate cross-modal prequestions from documents before retrieval, and

MLDocRAG (Zhang and Wu, 2026) generates finegrained queries from heterogeneous multimodal chunks and links them across modalities and pages. We therefore do not claim index-time enrichment as new. MIDR differs in that enrichment is fielded and verified: each page yields a typed multi-field record whose fields play distinct retrieval roles (Table 1) rather than a single generated surface, and every field is audited against the rendered page and extracted text before indexing, with refinement firing on 9.6% of pages overall and 52% on the hardest domain. We do not reimplement these systems as page-level ViDoRe retrievers: their retrieval units, pipelines, and target tasks differ from ours, so a reimplementation would not be the controlled comparison. We instead report controlled simplifications of MIDR itself—a QA-only variant, which is a doc2query-style single-surface configuration, and an OCR-only variant that withholds the page image (Section 5).

The dominant alternative is visual multi-vector retrieval. ColBERT (Khattab and Zaharia, 2020) introduced late interaction over token-level vectors, and ColBERTv2 (Santhanam et al., 2022) improved the efficiency and quality of this paradigm. Col-Pali (Faysse et al., 2025) adapts late interaction to rendered document pages, and ColQwen2.5 extends the ColPali-style approach with a stronger vision-language backbone, setting the open source state of the art on ViDoRe V3 (Loison et al., 2026). These systems address OCR failure by retrieving over rendered page images directly. Optimized late-interaction kernels (Pony et al., 2026; Sharma, 2026) reduce the query-time cost of MaxSim scoring and would narrow the latency difference between the two designs; they do not reduce the memory needed to store patch-level page representations, which is where the designs differ structurally. MIDR accepts the need for multimodal understanding but separates it from query-time visual retrieval: rendered images are used during ingestion, while serving uses text-centric indexes.

Multimodal RAG systems such as M3DocRAG (Cho et al., 2024), VDocRAG (Tanaka et al., 2025), MDocAgent (Han et al., 2025), and ViDoRAG (Wang et al., 2025) combine document images, retrieval, agents, and generation, typically spending multimodal computation at query time. Benchmarks such as REAL-MM-RAG (Wasserman et al., 2025), MMDocIR (Dong et al., 2025), and ViDoRe V3 (Loison et al., 2026) establish the need to evaluate retrieval over complex layouts and visually grounded evidence. Query-side methods such as HyDE (Gao et al., 2023), Guided Query Refinement (Uzan et al., 2026), and multimodal reranking (Geigle et al., 2022) are complementary: they change how queries are processed, whereas MIDR changes what the index contains.

## 3 Enrichment-Augmented Indexing with MIDR

## 3.1 Problem Setup and Design Objective

Let D be a collection of visually rich documents, where each document d consists of pages $p _ { d , i }$ . Each page has a rendered image $I _ { d , i }$ and extracted text $x _ { d , i }$ , and the task is to rank pages for a text query $q .$ The design question is where multimodal computation should occur. Visual multi-vector retrievers encode rendered pages into image-derived representations that remain on the query-time serving path. In contrast, MIDR derives compact document context $c _ { d }$ and page-level enrichment fields $e _ { d , i }$ during ingestion, then serves retrieval over textual fields $\{ x _ { d , i } , c _ { d } , e _ { d , i } \}$ using lexical, dense, or hybrid retrieval. The objective is to amortize multimodal reasoning before queries arrive while preserving text-centric serving.

MIDR implements this design as a training-free enrichment framework. Given a visually rich document, it constructs a fielded textual representation of each page grounded in both the rendered page image and extracted page text. As shown in Figure 1, MIDR has three stages: document-level enrichment, verified page-level enrichment, and text-centric indexing.

## 3.2 Document and Page Enrichments

For each document, MIDR first produces document-level enrichments from the first five pages. These fields provide global context for page-level enrichment, helping disambiguate repeated entities, acronyms, and domain-specific references. Each page is then enriched using four inputs: the rendered page image, extracted page text, document-level enrichments, and page metadata. The image exposes layout, tables, charts, figures, and visual grouping that OCR may flatten; the extracted text preserves exact lexical evidence. Table 1 lists the resulting document- and page-level enrichments, separating indexed retrieval fields from routing and control fields.

<table><tr><td>Level</td><td>Enrichment</td><td>Role in MIDR</td></tr><tr><td colspan="3">Indexed retrieval fields</td></tr><tr><td>Document document_focus Document main_entities</td><td></td><td>Global topic or purpose. Salient organizations,</td></tr><tr><td></td><td></td><td>products, datasets, regulations, drugs, or systems. Compact domain descriptors for lexical</td></tr><tr><td>Page</td><td>topic_tags</td><td>and dense matching.</td></tr><tr><td>Page</td><td>keyphrases</td><td>Entity-metric-concept phrases for se- mantic matching.</td></tr><tr><td>Page</td><td>table_summary</td><td>Textualizes headers, units, rows, values, and comparisons.</td></tr><tr><td>Page</td><td>chart_summary</td><td>Textualizes axes, legends, trends, quan- tities, and visual relations.</td></tr><tr><td>Page</td><td>coarse_qa</td><td>Broad page-level QA pairs aligned with likely intents. Precise QA pairs for facts, values, defi-</td></tr><tr><td>Page</td><td>fine_qa</td><td>nitions, cells, and visual details.</td></tr><tr><td colspan="3">Routing and control fields</td></tr><tr><td></td><td>Document document_type</td><td>Genre or source type; conditions page interpretation.</td></tr><tr><td>Page</td><td>layout</td><td>Page structure and visual content type; routes table/chart handling.</td></tr><tr><td>Page</td><td>signal_quality</td><td>Marks low-signal or decorative pages; gates enrichment.</td></tr><tr><td>Page</td><td></td><td>verification_issuesUnsupported or inconsistent fields</td></tr><tr><td>Page</td><td>refinement_edits</td><td>found during verification. Fields revised after verification for trace- ability.</td></tr></table>

Table 1: MIDR enrichment schema. Indexed fields are used by BM25F and dense retrieval; routing and control fields guide enrichment, verification, and analysis.

## 3.3 Extract–Verify–Refine

Page enrichment follows an extract–verify–refine loop. Figure 2 illustrates the three-stage enrichment process. The extractor generates a structured page enrichment from the rendered page image, extracted text, document enrichments, and page metadata. A deterministic postprocessor normalizes tags and keyphrases, removes duplicate QA pairs, and enforces consistency between layout flags and summary fields. The verifier audits the draft against the rendered page and extracted text for grounding, layout consistency, internal consistency, answer quality, and completeness. If issues are found, the refiner revises only the flagged fields and postprocessing is applied again. This conservative loop matters because index-time errors can affect retrieval for many future queries.

## 3.4 Indexing and retrieval

The verified page enrichments and original page text are indexed as separate BM25F fields (Robertson et al., 2004). Document-level indexed fields are replicated across the pages of the corresponding document so that each page exposes both local evidence and global context. All BM25F fields are weighted uniformly; Appendix N reports an apriori role-based variant and shows that the schema is insensitive to this choice.

For dense retrieval, MIDR embeds the original page text and each enrichment field separately using EmbeddingGemma, then combines field embeddings with mean pooling. For hybrid retrieval, MIDR fuses BM25F and dense rankings with Reciprocal Rank Fusion (Cormack et al., 2009). Thus, multimodal reasoning is performed once during ingestion, while query-time retrieval operates over BM25F and dense text indexes containing multimodally grounded evidence, without page-image processing, visual multi-vector indexes, or lateinteraction scoring.

![](images/01ebb0f4040c8ae869313ed025f405a794ca50e2f64c3ee493d296a5f89068e6.jpg)  
Figure 2: Extract–verify–refine enrichment loop. An initial structured enrichment is extracted from the page, verified against the rendered page image and extracted text, and selectively refined when verification identifies issues.

## 4 Experimental Setup

## 4.1 Benchmark

We evaluate on ViDoRe V3 (Loison et al., 2026), a page-level benchmark for multimodal document retrieval. All experiments use the English-query configuration, and all MIDR enrichments are generated in English, regardless of source-document language. The primary evaluation covers five Englishdocument domains—computer science, finance, HR, industrial, and pharmaceuticals—with 1,489 queries over 12,968 pages from 101 documents. We additionally evaluate two French-document domains—energy and physics—as a cross-lingual stress test, with 610 English queries over 3,899 pages from 83 documents. Overall, the evaluation covers 2,099 queries over 16,867 pages from 184 documents; Appendix A gives the per-domain breakdown. Following ViDoRe, the retrieval unit is a page, the candidate pool is all pages within the corresponding domain, and we report nDCG@10 using the official qrels and ir-measures (MacAvaney et al., 2022).

## 4.2 Enrichment Model

MIDR uses the two-stage enrichment pipeline of Section 3, with GPT-5.1 as the enrichment MLLM for all main results. We measure sensitivity to that choice by re-running the full pipeline with alternative backends and reindexing from scratch, holding retrieval, fusion, embeddings, and field weights fixed. Frontier MLLMs cluster closely on the English evaluation, spanning 0.6120–0.6231 nDCG@10 with GPT-5.1 the strongest aggregate among them, while the open-weight backend we tested trails substantially and collapses on French (Appendix K). GPT-5.1 is selected once on the five-domain English evaluation; we do not switch enrichment models by domain.

Page-level prompts are domain-aware across the five English-document domains and the two French-document domains. The enrichment pipeline covers all 16,867 pages. For the Frenchdocument domains, generating enrichments in English lets us test whether index-time enrichment can bridge English queries to French page content.

## 4.3 Retrieval Systems

We evaluate four text-centric retrieval configurations: BM25 over raw page markdown, BM25F over enriched fields, dense retrieval with meanpooled EmbeddingGemma field embeddings, and MIDR Hybrid, which combines lexical and dense rankings. We compare MIDR against open-weight visual multi-vector retrievers: ColQwen2.5, the strongest ColPali-family model at comparable index size, and ColEmbed-3B-v2 (Moreira et al., 2026), a stronger and substantially larger lateinteraction retriever. All retrieval-effectiveness numbers are produced by our own runs against the public qrels using the same corpora and evaluation code. Dense and multi-vector experiments use FAISS (Douze et al., 2025) on a single NVIDIA L4. Appendix C documents the embedding-model and field-pooling choices behind the dense retriever and implementation details for the visual baselines.

## 4.4 Ablations

We evaluate group ablations, leave-one-out field ablations, and query-level stratifications. Group ablations test compact variants such as QA-only, no-QA, semantic-only, visual-only, and no-semantic. Leave-one-out ablations measure the contribution of individual enrichment fields. Stratified analyses group query results by visual content type and query type. The full ablation suite and run inventory are provided in the appendix.

## 5 Results and Analysis

We empirically evaluate the design claim behind MIDR: multimodal document understanding can be moved from query time to index time while preserving retrieval quality and reducing serving-time cost. We examine this claim step by step, moving from the main accuracy result and serving efficiency to cross-lingual behavior, enrichment mechanisms, and complementarity with visual multi-vector retrieval.

## How does index-time enrichment compare with visual multi-vector retrieval?

Table 2 gives the central result: MIDR Hybrid reaches 0.6219 nDCG@10, competitive with ColQwen2.5 aggregate of 0.6300 and leading on two of five English domains. Relative to raw BM25 (0.5057), this is a 23.0% gain, showing that much of the gap to visual retrieval can be closed through text-side representation alone, without serving-time visual matching. The gain is not only a fusion effect: enriched BM25F reaches 0.5592, while dense mean-pool retrieval reaches 0.5898 over the same fielded representation, showing that multimodally grounded fields improve both lexical and semantic retrieval.

The aggregate gap is smaller than it first appears. Computer science alone accounts for 0.045 of it (0.7170 vs. 0.7623), consistent with the advantage of visual encoders on formulaic notation, diagrams, and code layout. Across the other four English domains the two systems average 0.5981 and 0.5969 respectively: MIDR leads on HR (0.6043 vs. 0.6018) and pharmaceuticals (0.6424 vs. 0.6382), and trails by less than 0.002 on finance and industrial.

To broaden the comparison beyond ColQwen2.5, we additionally evaluate ColEmbed-3B-v2 (Moreira et al., 2026), a stronger visual late-interaction retriever. As shown in Table 3, ColEmbed-3Bv2 reaches 0.6730 average nDCG@10 on the five English domains, establishing a higher accuracy point among the visual retrievers we evaluate. This gain comes with a substantially larger visual multi-vector index: 11.07 MB per page compared with 0.038 MB per page for MIDR. The comparison therefore positions MIDR as a strong accuracy–deployment operating point rather than the maximum-accuracy retriever.

Does index-time enrichment change the servingtime cost profile?

Table 4 summarizes the deployment tradeoff. Normalized to BM25, ColQwen2.5 requires 27.9× query latency and 65.0× index memory, while MIDR Hybrid uses 14.0× latency and 7.5× memory. Thus, relative to ColQwen2.5, MIDR Hybrid is roughly 2× faster and 9× smaller at query time. ColEmbed-3B-v2 establishes a higher-accuracy operating point, but with a substantially larger visual multi-vector index (11.07 MB/page versus 0.038 MB/page for MIDR). Enriched BM25F alone provides an even cheaper operating point, at 3.4× latency and 2.5× memory. This is the intended amortization tradeoff: MIDR shifts multimodal computation into offline ingestion, whereas visual multi-vector retrieval keeps image-derived indexes, multimodal query encoding, and late-interaction scoring on the recurring query-time path.

<table><tr><td>System</td><td>CS</td><td>Finance</td><td>HR</td><td>Industrial</td><td>Pharma</td><td>Avg.</td></tr><tr><td>BM25 over markdown</td><td>0.5774</td><td>0.4920</td><td>0.4851</td><td>0.4398</td><td>0.5341</td><td>0.5057</td></tr><tr><td>Enriched BM25F</td><td>0.6497</td><td>0.5453</td><td>0.5292</td><td>0.4705</td><td>0.6012</td><td>0.5592</td></tr><tr><td>Dense mean-pool</td><td>0.7107</td><td>0.6030</td><td>0.5474</td><td>0.4541</td><td>0.6341</td><td>0.5898</td></tr><tr><td>ColQwen2.5</td><td>0.7623</td><td>0.6276</td><td>0.6018</td><td>0.5200</td><td>0.6382</td><td>0.6300</td></tr><tr><td>MIDR Hybrid (ours)</td><td>0.7170</td><td>0.6261</td><td>0.6043</td><td>0.5197</td><td>0.6424</td><td>0.6219</td></tr></table>

Table 2: ViDoRe V3 English-domain nDCG@10. Blue-shaded cells mark the best result within each domain; the orange-shaded cell marks the best aggregate. MIDR is competitive with ColQwen2.5. MIDR uses uniform field weights throughout.

<table><tr><td>System</td><td>EN-5 nDCG@10</td><td>MB/page</td></tr><tr><td>MIDR Hybrid</td><td>0.6219</td><td>0.038</td></tr><tr><td>ColQwen2.5</td><td>0.6300</td><td>0.37</td></tr><tr><td>ColEmbed-3B-v2</td><td>0.6730</td><td>11.07</td></tr></table>

Table 3: Accuracy–index-size comparison on the five English-document ViDoRe V3 domains. ColEmbed-3Bv2 achieves the highest retrieval accuracy, while MIDR provides a substantially smaller text-centric index.
<table><tr><td>Retrieval path</td><td>Latency Memory lower is better</td></tr><tr><td>BM25</td><td>1.0× 1.0×</td></tr><tr><td>Enriched BM25F</td><td>3.4× 2.5×</td></tr><tr><td>MIDR Hybrid</td><td>14.0× 7.5×</td></tr><tr><td>ColQwen2.5</td><td>27.9× 65.0×</td></tr></table>

Table 4: Average query latency and index memory normalized to BM25, measured with our retrieval implementations. Green-shaded rows mark MIDR; the orange-shaded row marks the visual multi-vector baseline. MIDR Hybrid is roughly 2× faster and 9× smaller than ColQwen2.5 at query time.

## Can enrichment bridge language mismatch before retrieval?

The French-document domains test this second consequence of enrichment-augmented indexing.

<table><tr><td>System</td><td>Energy</td><td>Physics</td><td>Avg.</td></tr><tr><td>BM25 over markdown</td><td>0.1577</td><td>0.1488</td><td>0.1532</td></tr><tr><td>Enriched BM25F</td><td>0.5132</td><td>0.4081</td><td>0.4606</td></tr><tr><td>Dense mean-pool</td><td>0.6042</td><td>0.4341</td><td>0.5192</td></tr><tr><td>ColQwen2.5</td><td>0.5967</td><td>0.4663</td><td>0.5315</td></tr><tr><td>MIDR Hybrid (ours)</td><td>0.6192</td><td>0.4704</td><td>0.5448</td></tr></table>

Table 5: French ViDoRe V3 nDCG@10 with English queries and French documents. Blue-shaded cells mark best domains; orange-shaded cells mark best averages. MIDR uses English index-time enrichments to exceed reproduced ColQwen2.5 on aggregate and on both domains.

Here the documents are French and the queries are English. Raw BM25 over French markdown collapses to 0.1532 average nDCG@10 (Table 5), since English queries share little surface vocabulary with French page text. Visual retrievers handle this mismatch implicitly by matching rendered page images at serving time. MIDR handles it differently: because enrichments are generated in English from French page content, cross-lingual retrieval becomes monolingual matching against English enrichment fields at query time. Enriched BM25F lifts the French average to 0.4606, and MIDR Hybrid reaches 0.5448, exceeding reproduced ColQwen2.5 both on aggregate (0.5448 vs. 0.5315) and on each domain—energy (0.6192 vs. 0.5967) and physics (0.4704 vs. 0.4663)—without serving-time visual matching. The language normalization happens once per document at ingestion, not once per query.

This result exposes a deployment property of text-mediated enrichment: the language of the index can be chosen during ingestion. A multilingual collection can therefore be searched through a target-language textual index, while visual retrievers achieve language bridging only implicitly through rendered-page representations. The result is specific to French documents with English queries; broader language-pair coverage remains

<table><tr><td>Configuration</td><td>BM25F</td><td>Dense</td><td>Hybrid</td></tr><tr><td>Markdown only</td><td>0.5057</td><td>0.5177</td><td>0.5737</td></tr><tr><td>Semantic only</td><td>0.5057</td><td>0.5472</td><td>0.5843</td></tr><tr><td>No QA</td><td></td><td>0.5606</td><td>0.5788</td></tr><tr><td>QA only</td><td>0.5602</td><td>0.5622</td><td>0.6200</td></tr><tr><td>Full MIDR (role-based)</td><td>0.5592</td><td>0.5898</td><td>0.6231</td></tr></table>

Table 6: Group ablations on the five English domains. Bold marks the best result per retrieval path; underline marks the compact QA-only hybrid result, which recovers most of the gain over markdown-only retrieval.

## future work.

## What has enrichment actually added to the index?

The 23.0% aggregate lift over raw BM25 comes from partially non-overlapping lexical and dense contributions: Enriched BM25F reaches 0.5592 and dense mean-pool field retrieval 0.5898, which their RRF hybrid combines into 0.6219 (Table 2). Group ablations in Table 6 show that QA pairs are the dominant low-cost enrichment surface: qa\_only reaches 0.6200, recovering 94% of the enrichment gain over a markdown-only hybrid baseline. Conversely, removing QA drops hybrid retrieval to 0.5788, close to the markdown-only setting. This supports the interpretation that generated QA pairs create the main query–page bridge, while the full schema adds targeted gains through table summaries, document focus, and other fields.

Field ablations point to distinct retrieval roles. QA pairs provide the strongest lexical bridge, while keyphrases and QA fields support dense matching; full leave-one-out results appear in Appendix E. Table summaries have little aggregate effect, but their value is concentrated on table pages, where removing them costs 0.028 nDCG@10.

## Where does enrichment help most?

Table 7 shows that the gains are largest where OCR loses structure. Mixed-visual pages improve by 38.6% relative nDCG@10, table pages by 31.8%, and numerical queries by 41.7%. Text-only pages and boolean queries improve least (+14.3% and +11.7%), where raw lexical evidence already gives BM25 a stronger starting point. This supports the central mechanism: MIDR helps most when retrieval depends on structure that OCR flattens.

## Does the result depend on field weighting?

ViDoRe V3 provides no development split, so there is no principled way to tune BM25F field weights without fitting to the evaluation queries. We therefore report uniform weights, with every field set to 1.0, as MIDR’s configuration throughout this paper, and we do not tune them. As a check that this costs us nothing, we also evaluated a role-based weighting assigned a priori from each field’s intended retrieval role (Appendix N): it scores 0.6231 on English, 0.0012 above uniform, and 0.5339 on French, 0.0109 below. The two configurations are within noise on English and uniform is better on French, which indicates that the multi-field schema rather than weight tuning drives the gains. Field boosts remain available as a per-deployment knob, but no headline result in this paper depends on setting them.

<table><tr><td>Stratum</td><td>BM25</td><td>MIDR Hyb.</td><td>∆</td></tr><tr><td colspan="4">Page visual content</td></tr><tr><td>Mixed visual</td><td>0.3957</td><td>0.5484</td><td>+38.6%</td></tr><tr><td>Table</td><td>0.4680</td><td>0.6170</td><td>+31.8%</td></tr><tr><td>Infographic</td><td>0.4867</td><td>0.6346</td><td>+30.4%</td></tr><tr><td>Chart</td><td>0.4993</td><td>0.6046</td><td>+21.1%</td></tr><tr><td>Text only</td><td>0.5618</td><td>0.6424</td><td>+14.3%</td></tr><tr><td colspan="4">Query type</td></tr><tr><td>Numerical</td><td>0.4525</td><td>0.6414</td><td>+41.7%</td></tr><tr><td>Open-ended</td><td>0.3716</td><td>0.5136</td><td>+38.2%</td></tr><tr><td>Compare</td><td>0.5015</td><td>0.6208</td><td>+23.8%</td></tr><tr><td>Extractive</td><td>0.5896</td><td>0.7135</td><td>+21.0%</td></tr><tr><td>Multi-hop</td><td>0.4996</td><td>0.5711</td><td>+14.3%</td></tr><tr><td>Boolean</td><td>0.6004</td><td>0.6707</td><td>+11.7%</td></tr></table>

Table 7: MIDR Hybrid vs. raw BM25 by page visual content and query type on the five English domains. ∆ is relative nDCG@10 gain; full breakdowns are in Appendix F.

## Are enriched text retrieval and visual multivector retrieval substitutes?

The paired analysis suggests they are not. Because per-query scores are not available in the released ColQwen2.5 aggregates from Loison et al. (2026), this analysis uses our local ColQwen2.5 reproduction. Across the 1,489 English-domain queries, roughly half have a decisive winner at the > 0.1 nDCG@10 threshold: MIDR wins 351 queries and ColQwen2.5 wins 362. MIDR tends to win when exact financial, pharmaceutical, or regulatory quantities have been verbalized into QA pairs and keyphrases, especially on pages whose tables would be flattened by raw OCR. ColQwen2.5 tends to win when page evidence depends on distinctions that textual enrichment can blur, such as code blocks, equations, diagrams, or multiple visually similar tables. A per-query oracle that selects the better of the two systems reaches 0.7042 nDCG@10, a 13.0% relative gain over MIDR and 11.8% over reproduced ColQwen2.5 (Appendix M). The strongest reading is therefore complementarity, not replacement: enriched text indexes and visual multi-vector indexes capture different signals. ColEmbed-3B-v2 further shows that stronger visual retrieval can raise the accuracy ceiling, but at substantially greater index cost.

## 6 Conclusion and Future Work

MIDR shows that multimodal reasoning for visually rich document retrieval can be moved from query time to index time. By converting rendered pages into verified textual retrieval fields, enrichment-augmented indexing reaches performance comparable to strong visual multivector retrievers on ViDoRe V3 while exceeding ColQwen2.5 on the French domains, from an index roughly 9× smaller and at a one-time ingestion cost paid before any query arrives. The same design enables index-time transformations such as generating English enrichment fields from French document pages, and ablations show that the gains are mechanistic: different fields support different retrieval paradigms, with improvements concentrated where OCR loses structure. More broadly, MIDR reframes multimodal retrieval as a systems-design question about where multimodal understanding should occur, rather than whether it is needed.

MIDR is a framework rather than a fixed configuration: the enrichment schema, the prompts, the MLLM backend, the embedding model, and the field boosts are all deployment choices, and an improvement in any of them carries over without changing the serving path. The broader conclusion is not that visual encoders are unnecessary, but that enriched text indexes and visual multi-vector indexes are complementary design points, and that index-time enrichment is worth reaching for when deployment constraints favor amortized ingestion, text-centric infrastructure, or auditable evidence. A natural next step is adaptive retrieval: agents can route queries across typed enrichment fields—QA pairs, table summaries, keyphrases, and document context—and fall back to visual retrieval when finegrained visual matching is required.

## 7 Limitations

## 7.1 Comparison Scope

Our evaluation covers ViDoRe V3, with detailed analysis on five English-document domains and a targeted cross-lingual study on two Frenchdocument domains. We compare against openweight visual retrievers reproduced locally under a single controlled pipeline, but ViDoRe V3 lists further systems that we do not evaluate, including API-only models for which controlled local measurement is not possible (Loison et al., 2026).

MIDR is also not the most accurate retriever in this space: ColEmbed-3B-v2 reaches 0.6730 average nDCG@10 on the English domains against MIDR’s 0.6219, while storing roughly 11 MB per page against MIDR’s 0.038 MB. Our claim therefore concerns the accuracy–deployment tradeoff rather than peak accuracy, and a sweep of the full leaderboard would characterize that tradeoff more completely than the subset of systems we were able to run.

## 7.2 Serving Cost

The latency comparison is made against unoptimized late-interaction scoring. Optimized MaxSim kernels (Pony et al., 2026; Sharma, 2026) reduce the query-time cost of visual multi-vector retrieval and would narrow this gap. They do not reduce the cost of storing patch-level page representations, so index memory rather than latency is where the two designs differ structurally.

## 7.3 Ingestion Cost

MIDR moves computation rather than removing it. Ingestion requires on average 2.1 MLLM calls and approximately 8k tokens per page. This is a one-time index-building cost that is amortized over future queries, but it scales with corpus size and must be incurred again when documents change or the enrichment schema is revised.

## 7.4 Enrichment MLLM

Retrieval quality depends on the model used at index time. Frontier backends cluster within 0.011 nDCG@10 on English, but the open-weight backend we tested trails by 0.046 on English and collapses on French (0.298 vs. 0.534; Appendix K). The cross-lingual result in particular should not be assumed to transfer to every backend, and deployments restricted to open-weight models should expect a gap.

## 7.5 Field and Language Coverage

Chart summaries are net-neutral to negative on aggregate and clearly harmful on some domains, and main-entity strings retain source-language forms that act as confounders on French; both need prompt-level redesign before they generalize across source languages. The cross-lingual study also covers one direction of one language pair. The extract– verify–refine loop reduces unsupported and inconsistent fields before indexing, but grounding and coverage remain open challenges.

## References

Hessa A Alawwad, Areej Alhothali, Usman Naseem, Ali Alkhathlan, and Amani Jamal. 2025. Enhancing textual textbook question answering with large language models and retrieval augmented generation. Pattern Recognition, 162:111332.

Zhenghua Bao and Yi Shi. 2026. Indexrag: Bridging facts for cross-document reasoning at index time. Preprint, arXiv:2603.16415.

Peter Baile Chen, Tomer Wolfson, Michael Cafarella, and Dan Roth. 2025. EnrichIndex: Using LLMs to enrich retrieval indices offline. arXiv preprint arXiv:2504.03598.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711. Association for Computational Linguistics.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multi-modal retrieval is what you need for multipage multi-document understanding. Preprint, arXiv:2411.04952.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2025. M3DocVQA: Multi-modal multi-page multi-document understanding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 6237– 6247.

Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, and Youngjae Yu. 2025. Zero-shot multimodal document retrieval via cross-modal question generation. arXiv preprint arXiv:2508.17079.

Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759.

Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. 2025. MM-DocIR: Benchmarking multimodal retrieval for long documents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 30971–31005. Association for Computational Linguistics.

Kuicai Dong, Shurui Huang, Fangda Ye, Wei Han, Zhi Zhang, Dexun Li, Wenjun Li, Qu Yang, Gang Wang, Yichao Wang, Chen Zhang, and Yong Liu. 2026. Doc-researcher: A unified system for multimodal document parsing and deep research. In Proceedings ofthe ACM Web Conference 2026, pages 2349–2360, New York, NY, USA. Association for Computing Machinery.

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025. The faiss library. Preprint, arXiv:2401.08281.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on RAG meeting LLMs: Towards retrieval-augmented large language models. arXiv preprint arXiv:2405.06211.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. ColPali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449.

Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2022. From distillation to hard negative sampling: Making sparse neural ir models more effective. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, pages 2353–2359. Association for Computing Machinery.

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. Splade: Sparse lexical and expansion model for first stage ranking. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, pages 2288–2292. Association for Computing Machinery.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777, Toronto, Canada. Association for Computational Linguistics.

Gregor Geigle, Jonas Pfeiffer, Nils Reimers, Ivan Vulic, and Iryna Gurevych. 2022. Retrieve fast, rerank smart: Cooperative and joint approaches for improved cross-modal retrieval. Transactions of the Associationfor Computational Linguistics, 10:503– 521.

Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A multi-modal multi-agent framework for document understanding. arXiv preprint arXiv:2503.13964.

Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Associationfor Computational Linguistics: Main Volume, pages 874–880.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781.

Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 39–48.

Anagha Kulkarni, Parin Rajesh Jhaveri, Prasha Shrestha, Yu Tong Han, Reza Amini, and Behrouz Madahian. 2026. Long-context long-form question answering for legal domain. arXiv preprint arXiv:2602.07190.

Tzu-Lin Kuo, Wei-Ning Chiu, Wei-Yun Ma, and Pu-Jen Cheng. 2025. Doc2query++: Topic-coverage based document expansion and its application to dense retrieval via dual-index fusion. Preprint, arXiv:2510.09557.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems.

Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. 2025a. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407.

Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, and Zhicheng Dou. 2025b. Sliding windows are not the end: Exploring full ranking with long-context large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 162–176.

António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, and Gautier Viaud. 2026. ViDoRe v3: A comprehensive evaluation of retrieval augmented generation in complex real-world scenarios. arXiv preprint arXiv:2601.08620.

Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, and Jimmy Lin. 2025. VISA: Retrieval augmented generation with visual source attribution. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30154– 30169.

Sean MacAvaney, Craig Macdonald, and Iadh Ounis. 2022. Streamlining evaluation with ir-measures. In Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022, Stavanger, Norway, April 10–14, 2022, Proceedings, Part II, pages 305–310. Springer-Verlag.

Gabriel de Souza P Moreira, Ronay Ak, Mengyao Xu, Oliver Holworthy, Benedikt Schifferer, Zhiding Yu, Yauhen Babakhin, Radek Osmulski, Jiarui

Cai, Ryan Chesler, et al. 2026. Nemotron colembed v2: Top-performing late interaction embedding models for visual document retrieval. arXiv preprint arXiv:2602.03992.

Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. arXiv preprint arXiv:1904.08375.

Roi Pony, Daniel Ezer, Adi Raz Goldfarb, Idan Friedman, Oshri Naparstek, and Udi Barzelay. 2026. Flash-MaxSim: IO-aware fused kernels for late-interaction retrieval. arXiv preprint arXiv:2605.29517.

Yifu Qiu, Varun R. Embar, Yizhe Zhang, Navdeep Jaitly, Shay B. Cohen, and Benjamin Han. 2025. Eliciting in-context retrieval and reasoning for long-context language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 3176– 3192.

Qwen Team. 2026. Qwen3.5-Omni technical report. arXiv preprint arXiv:2604.15804.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at TREC-3. In Proceedings of The Third Text REtrieval Conference (TREC 1994), volume 500-225, pages 109–126.

Stephen E. Robertson, Hugo Zaragoza, and Michael J. Taylor. 2004. Simple BM25 extension to multiple weighted fields. In Proceedings of the 2004 ACM CIKM International Conference on Information and Knowledge Management, pages 42–49.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715–3734.

Ashutosh Sharma. 2026. TileMaxSim: IO-aware GPU MaxSim scoring with dimension tiling and fused product quantization. arXiv preprint arXiv:2606.26439.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-level medical question answering with large language models. Nature Medicine, 31:943–950.

Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2025. VDocRAG: Retrieval-augmented generation over visually-rich documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Omri Uzan, Asaf Yehudai, Roi Pony, Eyal Shnarch, and Ariel Gera. 2026. Guided query refinement: Multimodal hybrid retrieval with test-time optimization. Preprint, arXiv:2510.05038.

Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, et al. 2025. Embedding Gemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354.

Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. 2025. ViDoRAG: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9113–9134. Association for Computational Linguistics.

Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, and Leonid Karlinsky. 2025. REAL-MM-RAG: A real-world multi-modal retrieval benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31660–31683. Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2023. C-Pack: Packed resources for general chinese embeddings. arXiv preprint arXiv:2309.07597.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. 2025. Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765.

Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. 2024. Mm-llms: Recent advances in multimodal large language models. Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 12401–12430.

Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2025a. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17443–17453.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025b. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Yongyue Zhang and Yaxiong Wu. 2026. ML-DocRAG: Multimodal long-context document retrieval augmented generation. arXiv preprint arXiv:2602.10271.

## A Dataset and Enrichment Coverage

<table><tr><td>Domain</td><td>Pages</td><td>Documents</td></tr><tr><td>Computer Science</td><td>1,360</td><td>2</td></tr><tr><td>Finance (EN)</td><td>2,942</td><td>6</td></tr><tr><td>HR</td><td>1,110</td><td>14</td></tr><tr><td>Industrial</td><td>5,244</td><td>27</td></tr><tr><td>Pharmaceuticals</td><td>2,312</td><td>52</td></tr><tr><td>Energy</td><td>2,225</td><td>41</td></tr><tr><td>Physics</td><td>1,674</td><td>42</td></tr><tr><td>Total</td><td>16,867</td><td>184</td></tr></table>

Table 8: ViDoRe V3 per-domain page and document counts processed by the GPT-5.1 extract–verify–refine pipeline; five English and two French domains.

## B Enrichment Schema and Field Weights

## C Retrieval-Side Design Choices

This section reports the embedding backbone and field-pooling decisions used in the main results. We treat both as fixed design choices selected by preliminary comparison rather than as a primary axis of investigation; MIDR’s contribution is the enrichment schema and the extract–verify–refine indexing path, not a new dense retriever.

Embedding backbones. We compared three open-weight text embedding models as the dense backbone for MIDR: BAAI/bge-large-en-v1.5 (Xiao et al., 2023), a 335M-parameter English BERT-based encoder; Qwen/Qwen3-Embedding-0.6B (Zhang et al., 2025b), a 0.6B-parameter multilingual encoder with instruction-aware embeddings; and google/embeddinggemma-300m (Vera et al., 2025), a 308M-parameter multilingual encoder based on Gemma 3. In preliminary English-5 comparisons, EmbeddingGemma produced the strongest dense-only nDCG@10 while matching or exceeding the other backbones on the French domains, where its multilingual training was a deciding factor. Because backbone selection is not the focus of this paper and the three models cluster within a narrow band on English, we report only the EmbeddingGemma configuration in the main results and leave a systematic embedding-model sweep to follow-up work.

Field-pooling variants. MIDR embeds the original page text and each enrichment field separately, then combines the per-field embeddings into a single dense ranking. We compared four pooling strategies for combining query similarities across fields i, with field weight $w _ { i }$ and per-field document embedding $e _ { d _ { i } }$ , and k and r are the RRF parameters:

<table><tr><td>Level</td><td>Enrichment</td><td>Role in MIDR</td></tr><tr><td colspan="3">Indexed fields</td></tr><tr><td></td><td>Document document_focus</td><td>Primary topic or purpose; provides global context.</td></tr><tr><td colspan="2">Document main_entities</td><td>Šalient organizations, products, datasets, regulations, drugs, or systems;</td></tr><tr><td>Page</td><td>topic_tags</td><td>supports disambiguation. Compact domain-specific descriptors for lexical and dense matching.</td></tr><tr><td>Page</td><td>keyphrases</td><td>Concise entity-metric-concept phrases for semantic matching.</td></tr><tr><td>Page</td><td>table_summary</td><td>Textualizes rows, columns, headers, units, values, and salient comparisons.</td></tr><tr><td>Page</td><td>chart_summary</td><td>Textualizes axes, legends, plotted quan- tities, trends, and visual relationships.</td></tr><tr><td>Page</td><td>coarse_qa</td><td>Broad page-level QA pairs aligned with likely user intents.</td></tr><tr><td>Page</td><td>fine_qa</td><td>Precise QA pairs for facts, values, defi- nitions, table cells, and visual details.</td></tr><tr><td colspan="3">Routing and control fields (not indexed)</td></tr><tr><td></td><td>Document document_type</td><td>Document genre or source type; condi- tions page-level interpretation.</td></tr><tr><td>Page</td><td>layout</td><td>Page structure and visual content type; routes table/chart handling.</td></tr><tr><td>Page</td><td>signal_quality</td><td>Indicates whether the page contains suf- ficient retrievable evidence or is low- signal/decorative; gates enrichment.</td></tr><tr><td>Page</td><td></td><td>verification_issuesUnsupported or inconsistent fields iden- tified during verification; used for refine-</td></tr><tr><td>Page</td><td>refinement_edits</td><td>ment and auditing. Records fields revised after verification for traceability.</td></tr></table>

Table 9: MIDR enrichment schema. Document-level enrichments provide global context for page interpretation; page-level enrichments materialize visual and layout-dependent evidence as structured fields. Indexed fields are matched against queries through BM25F and dense retrieval; routing and control fields support enrichment, verification, and analysis.

• weighted sum: $\begin{array} { r } { \sum _ { i } w _ { i } \cos ( e _ { q } , e _ { d _ { i } } ) ; } \end{array}$

• max: max<sub>i</sub> (w<sub>i</sub> $\cos ( e _ { q } , e _ { d _ { i } } ) ) ;$

• weighted RRF: $\textstyle \sum _ { i } w _ { i } / ( k + r _ { i } ) ;$

• mean pool: cos $\left( e _ { q } , \sum _ { i } w _ { i } e _ { d _ { i } } / \sum _ { i } w _ { i } \right)$

In preliminary comparisons, mean pooling dominated the other three variants across both English and French domains. We adopt mean pooling as the default MIDR dense representation throughout the main results and report no further pooling ablations, since the gap to the next-best variant was both small and consistent. As with the embedding backbone, we view pooling as a fixed design choice rather than a primary contribution.

## D Group Ablations

Tables 10 and 11 report group ablations on the English and French domains. The English-French asymmetry is sharp: on French, qa\_only matches the full schema and no\_semantic stays close to it, while on English the multifield schema yields larger targeted gains. Across all seven domains, qa\_only reaches 0.5982 vs. 0.5976 hybrid nDCG@10 for the full schema—non-QA fields contribute positively on English but add offsetting noise on French (Appendix G).

<table><tr><td>Configuration</td><td>BM25F</td><td>Dense</td><td>Hybrid</td></tr><tr><td>Full MIDR (role-based)</td><td>0.5592</td><td>0.5898</td><td>0.6231</td></tr><tr><td>Markdown only</td><td>0.5057</td><td>0.5177</td><td>0.5737</td></tr><tr><td>QA only</td><td>0.5602</td><td>0.5622</td><td>0.6200</td></tr><tr><td>No QA</td><td></td><td>0.5606</td><td>0.5788</td></tr><tr><td>Semantic only</td><td>0.5057</td><td>0.5472</td><td>0.5843</td></tr><tr><td>Visual only</td><td>0.4620</td><td>0.4115</td><td>0.4779</td></tr><tr><td>No semantic</td><td></td><td>0.5818</td><td>0.6156</td></tr></table>

Table 10: Group ablations (English, 5 domains). Full MIDR enrichment (0.6231 hybrid) gains +23.2% over markdown-only BM25 (0.5057); decomposed, enriched BM25F alone is +10.6% and dense mean-pool alone is +16.6% (cf. Table 2). QA-only captures most aggregate hybrid gains, while full enrichment gives the best overall score and better targeted coverage.
<table><tr><td>Configuration</td><td>BM25F</td><td>Dense</td><td>Hybrid</td></tr><tr><td>Full MIDR (role-based)</td><td>0.4606</td><td>0.5192</td><td>0.5339</td></tr><tr><td>Markdown only</td><td>0.1532</td><td>0.4538</td><td>0.3357</td></tr><tr><td>QA only</td><td>0.4755</td><td>0.5177</td><td>0.5438</td></tr><tr><td>No QA</td><td></td><td>0.4917</td><td>0.3842</td></tr><tr><td>Semantic only</td><td>0.1532</td><td>0.4854</td><td>0.3654</td></tr><tr><td>Visual only</td><td>0.2339</td><td>0.3752</td><td>0.3368</td></tr><tr><td>No semantic</td><td></td><td>0.5136</td><td>0.5244</td></tr></table>

Table 11: Group ablations (French, 2 domains). qa\_only slightly exceeds the full schema (0.5438 vs. 0.5339 hybrid), and no\_semantic stays close to the full schema—both consistent with the cross-lingual sign flips in Appendix G, where non-QA fields contribute positively on English but add offsetting noise on French.

## E Stratified Field-Level Ablations

Aggregate leave-one-out deltas (Table 12) hide strongly targeted gains. Field value is concentrated, not distributed. Table summaries illustrate this sharply: aggregate hybrid delta 0.004 nDCG@10, but on table pages specifically, removing them costs 0.028—a seven-fold concentration. Table 13 shows similar specialization: coarse QA dominates numerical queries, fine QA dominates extractive ones, and document focus helps open-ended and

<table><tr><td>Removed field</td><td>BM25F</td><td>Dense</td><td>Hybrid</td></tr><tr><td>fine_qa</td><td>-0.028</td><td>-0.009</td><td>-0.016</td></tr><tr><td>coarse_qa</td><td>-0.024</td><td>-0.011</td><td>-0.014</td></tr><tr><td>keyphrases</td><td>0.000</td><td>-0.010</td><td>-0.005</td></tr><tr><td>document_focus</td><td>-0.008</td><td>-0.004</td><td>-0.005</td></tr><tr><td>table_summary</td><td>+0.001</td><td>-0.003</td><td>-0.004</td></tr><tr><td>main_entities</td><td>0.000</td><td>+0.002</td><td>-0.003</td></tr><tr><td>topic_tags</td><td>0.000</td><td>-0.001</td><td>-0.001</td></tr><tr><td>chart_summary</td><td>+0.007</td><td>-0.004</td><td>+0.003</td></tr></table>

Table 12: Leave-one-out field ablations on English-5. Negative deltas mean removing the field hurts. Bold: largest drop per backend.

multi-hop queries where topical alignment matters more than surface overlap.
<table><tr><td>Field</td><td>Helps most on</td><td>Hurts most on</td></tr><tr><td>table_summary</td><td>table pages (-0.028)</td><td>chart pages (+0.013)</td></tr><tr><td>chart_summary</td><td>infographics (-0.011)</td><td>boolean queries (+0.021)</td></tr><tr><td>coarse_qa</td><td>numerical (-0.040)</td><td></td></tr><tr><td>fine_qa</td><td>extractive (-0.030)</td><td></td></tr><tr><td>document_focus</td><td>open-ended (-0.016)</td><td></td></tr><tr><td>keyphrases</td><td>extractive (-0.018)</td><td>chart pages (+0.010)</td></tr></table>

Table 13: Selected stratified leave-one-out findings on English-5. Deltas are hybrid nDCG@10 changes when the field is removed (positive means removing the field helps).

Negative findings for chart and entity fields. Chart summaries are the only field that hurts hybrid retrieval on aggregate (+0.003 nDCG@10 when removed); main\_entities shows a weaker version of the same pattern. We hypothesize that broad trend descriptions (“increased over the period”, “declining trajectory”) over-match boolean queries without anchoring the correct page; the effect amplifies on French (Appendix G). We treat both as deployment knobs to toggle for cost-sensitive or multilingual settings, and leave stricter value extraction and language-aware canonicalization to follow-up work.

## F Gains by Visual Content and Query Type

Per-stratum BM25 → BM25F → hybrid progressions backing the aggregate in Table 7. English strata in Tables 14–15; French in Tables 16–17. Counts are query-level; small French strata (e.g., numerical n = 17, image n = 13) should be read with sample size in mind.

## G French Field Behavior

The English leave-one-out picture (Table 12) does not transfer unchanged to French. Two shifts matter.

<table><tr><td>Visual content</td><td>n</td><td>BM25</td><td>BM25F</td><td>Hybrid</td><td>Total gain</td></tr><tr><td>Mixed visual</td><td>222</td><td>0.3957</td><td>0.4832</td><td>0.5484</td><td>+38.6%</td></tr><tr><td>Table</td><td>370</td><td>0.4680</td><td>0.5519</td><td>0.6170</td><td>+31.8%</td></tr><tr><td>Image</td><td>25</td><td>0.5065</td><td>0.5768</td><td>0.6866</td><td>+35.6%</td></tr><tr><td>Infographic</td><td>109</td><td>0.4867</td><td>0.5526</td><td>0.6346</td><td>+30.4%</td></tr><tr><td>Chart</td><td>95</td><td>0.4993</td><td>0.5651</td><td>0.6046</td><td>+21.1%</td></tr><tr><td>Text only</td><td>666</td><td>0.5618</td><td>0.5820</td><td>0.6424</td><td>+14.3%</td></tr></table>

Table 14: Stratified gains by visual content type (English, 5 domains). Enrichment helps most where raw BM25 is weakest.

<table><tr><td>Query type</td><td>n</td><td>BM25</td><td>BM25F</td><td>Hybrid</td><td>Total gain</td></tr><tr><td>Numerical</td><td>49</td><td>0.4525</td><td>0.5393</td><td>0.6414</td><td>+41.7%</td></tr><tr><td>Open-ended</td><td>309</td><td>0.3716</td><td>0.4272</td><td>0.5136</td><td>+38.2%</td></tr><tr><td>Compare-contrast</td><td>260</td><td>0.5015</td><td>0.5578</td><td>0.6208</td><td>+23.8%</td></tr><tr><td>Extractive</td><td>348</td><td>0.5896</td><td>0.6539</td><td>0.7135</td><td>+21.0%</td></tr><tr><td>Enumerative</td><td>241</td><td>0.4839</td><td>0.5320</td><td>0.5922</td><td>+22.4%</td></tr><tr><td>Multi-hop</td><td>91</td><td>0.4996</td><td>0.5459</td><td>0.5711</td><td>+14.3%</td></tr><tr><td>Boolean</td><td>191</td><td>0.6004</td><td>0.6259</td><td>0.6707</td><td>+11.7%</td></tr></table>

Table 15: Stratified gains by query type (English, 5 domains). Numerical and open-ended queries benefit most from index-time enrichment.

<table><tr><td>Visual content</td><td>n</td><td>BM25</td><td>BM25F</td><td>Hybrid</td><td>Total gain</td></tr><tr><td>Infographic</td><td>28</td><td>0.1450</td><td>0.5775</td><td>0.6271</td><td>+332.4%</td></tr><tr><td>Table</td><td>79</td><td>0.1626</td><td>0.5257</td><td>0.6116</td><td>+276.2%</td></tr><tr><td>Mixed visual</td><td>172</td><td>0.1283</td><td>0.3885</td><td>0.4586</td><td>+257.5%</td></tr><tr><td>Image</td><td>13</td><td>0.1378</td><td>0.5041</td><td>0.4758</td><td>+245.2%</td></tr><tr><td>Text only</td><td>213</td><td>0.1592</td><td>0.4554</td><td>0.5400</td><td>+239.2%</td></tr><tr><td>Other visual</td><td>72</td><td>0.1470</td><td>0.4183</td><td>0.4908</td><td>+233.8%</td></tr><tr><td>Chart</td><td>33</td><td>0.2498</td><td>0.6998</td><td>0.7522</td><td>+201.1%</td></tr></table>

Table 16: Stratified gains by visual content type (French, 2 domains). All strata show >200% relative gains because raw BM25 on French documents is near-floor (Section 5); absolute hybrid nDCG@10 values (0.46– 0.75) are comparable in magnitude to English (Table 14). The largest relative lifts are on infographics and tables, where layout-dependent evidence is most salient.

<table><tr><td>Query type</td><td>n</td><td>BM25</td><td>BM25F</td><td>Hybrid</td><td>Total gain</td></tr><tr><td>Numerical</td><td>17</td><td>0.1579</td><td>0.6392</td><td>0.7254</td><td>+359.5%</td></tr><tr><td>Open-ended</td><td>152</td><td>0.0971</td><td>0.3587</td><td>0.4415</td><td>+354.9%</td></tr><tr><td>Enumerative</td><td>84</td><td>0.1196</td><td>0.4171</td><td>0.4769</td><td>+298.9%</td></tr><tr><td>Multi-hop</td><td>40</td><td>0.1156</td><td>0.3121</td><td>0.3963</td><td>+242.9%</td></tr><tr><td>Compare-contrast</td><td>117</td><td>0.1744</td><td>0.5023</td><td>0.5663</td><td>+224.6%</td></tr><tr><td>Extractive</td><td>135</td><td>0.2069</td><td>0.5462</td><td>0.6317</td><td>+205.3%</td></tr><tr><td>Boolean</td><td>65</td><td>0.2010</td><td>0.5521</td><td>0.6039</td><td>+200.5%</td></tr></table>

Table 17: Stratified gains by query type (French, 2 domains). The English ordering of relative gains (numerical and open-ended at the top, boolean at the bottom; Table 15) carries over to French, but all magnitudes are inflated by the BM25 floor on French documents. Sample sizes are small for the rarest strata (numerical n = 17, multi-hop n = 40).

First, QA fields carry essentially all of the crosslingual bridge signal: removing fine\_qa costs 0.0315 nDCG@10 on French versus 0.0162 on English, and coarse\_qa costs 0.0279 versus 0.0141 (Table 18). Consequently, qa\_only matches the full schema across all seven domains (0.5982 vs. 0.5976 hybrid), exceeding the 94% English-only recovery.

Second, chart\_summary and main\_entities flip sign: removing chart\_summary helps French hybrid retrieval by 0.0104 nDCG@10, and removing main\_entities helps by 0.0075. Chart descriptions generated from French visual content produce noisy English representations that overmatch unrelated queries, and main-entity strings retain French forms or transliterations that act as confounders. For multilingual deployments, both fields likely need prompt-level redesign before they generalize across source languages. Per-domain LOO deltas appear in Table 18; the flat zeros for keyphrases, topic\_tags, and main\_entities under BM25F reflect the role-based weighting from Section 3, where these fields contribute essentially through dense retrieval.

## H Per-Domain Latency and Memory

Table 19 decomposes Table 4 by domain, reporting query latency and index memory normalized to BM25 on each domain. Both tables report ratios rather than absolute measurements. The aggregate column is the mean of the per-domain ratios, so it does not equal the ratio of the corresponding per-domain means.

## I Offline Ingestion Cost

Table 20 summarizes the offline cost of the extract– verify–refine pipeline over all 16,867 ViDoRe V3 pages, using GPT-5.1 as the enrichment MLLM; Table 21 decomposes it by domain. Figures are aggregated from the per-page enrichment traces. We report MLLM calls, tokens, wall-clock time, and refinement rate rather than a monetary total, because the dollar cost depends on the provider and the pricing agreement under which the pipeline is run. The cost is one-time and paid at index build, whereas visual multi-vector retrieval pays GPU cost on every query for the life of the index.

Amortized over all pages, the three pipeline stages cost 2,344 input and 1,441 output tokens for extraction, 3,489 and 290 for verification, and 424 and 183 for the conditional refinement call. Serial wall-clock time is 17.7 s, 3.9 s, and 1.7 s respectively; the stage figures are rounded independently, so they sum to slightly more than the measured 23.2 s per-page total in Table 20. Verification is therefore cheap in output tokens but expensive in input tokens, because it re-reads the draft enrichment alongside the page; refinement is the cheapest stage and runs on fewer than one page in ten.

<table><tr><td></td><td colspan="3">BM25F</td><td colspan="3">Dense</td><td colspan="3">Hybrid</td></tr><tr><td>Removed field</td><td>Energy</td><td>Physics</td><td> $\operatorname { A v g }$ </td><td>Energy</td><td>Physics</td><td> $\operatorname { A v g }$ </td><td>Energy</td><td>Physics</td><td> $\operatorname { A v g }$ </td></tr><tr><td>fine_qa</td><td>-0.0698</td><td>-0.0701</td><td>-0.0699</td><td>-0.0090</td><td>-0.0064</td><td>-0.0077</td><td>-0.0353</td><td>-0.0278</td><td>-0.0315</td></tr><tr><td>coarse_qa</td><td>-0.0707</td><td>-0.0697</td><td>-0.0702</td><td>-0.0140</td><td>-0.0042</td><td>-0.0091</td><td>-0.0343</td><td>-0.0215</td><td>-0.0279</td></tr><tr><td>keyphrases</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>-0.0151</td><td>-0.0032</td><td>-0.0092</td><td>-0.0027</td><td>+0.0002</td><td>-0.0012</td></tr><tr><td>document_focus</td><td>+0.0002</td><td>-0.0053</td><td>-0.0026</td><td>-0.0024</td><td>+0.0023</td><td>0.0000</td><td>-0.0006</td><td>+0.0054</td><td>+0.0024</td></tr><tr><td>table_summary</td><td>-0.0119</td><td>+0.0027</td><td>-0.0046</td><td>-0.0021</td><td>+0.0017</td><td>-0.0002</td><td>-0.0072</td><td>+0.0068</td><td>-0.0002</td></tr><tr><td>main_entities</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>+0.0082</td><td>+0.0180</td><td>+0.0131</td><td>+0.0072</td><td>+0.0078</td><td>+0.0075</td></tr><tr><td>topic_tags</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>-0.0095</td><td>-0.0131</td><td>-0.0113</td><td>-0.0043</td><td>-0.0029</td><td>-0.0036</td></tr><tr><td>chart_summary</td><td>-0.0072</td><td>+0.0259</td><td>+0.0093</td><td>-0.0050</td><td>+0.0017</td><td>-0.0017</td><td>+0.0090</td><td>+0.0118</td><td>+0.0104</td></tr></table>

Table 18: French leave-one-out field ablations. Negative deltas mean removing the field hurts performance; positive deltas mean removing the field helps. Bold marks the most consequential removal per backend. Sign convention matches Table 12.
<table><tr><td>Metric</td><td>Method</td><td>CS</td><td>Energy</td><td>Fin.</td><td>HR</td><td>Ind.</td><td>Phar.</td><td>Phys.</td><td>Avg.</td></tr><tr><td rowspan="6">Latency</td><td>BM25</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Enr. BM25F</td><td>1.8</td><td>3.5</td><td>2.4</td><td>2.2</td><td>6.2</td><td>1.9</td><td>5.6</td><td>3.4</td></tr><tr><td>Gemma</td><td>2.6</td><td>3.3</td><td>1.7</td><td>2.5</td><td>0.9</td><td>0.9</td><td>7.9</td><td>2.8</td></tr><tr><td>ColQwen2.5</td><td>17.1</td><td>32.3</td><td>22.2</td><td>13.7</td><td>23.0</td><td>26.7</td><td>60.5</td><td>27.9</td></tr><tr><td>Enr. BM25F+Gemma</td><td>7.5</td><td>11.4</td><td>7.6</td><td>7.7</td><td>4.9</td><td>8.2</td><td>21.4</td><td>9.8</td></tr><tr><td>MIDR Hybrid</td><td>8.8</td><td>14.1</td><td>9.2</td><td>8.9</td><td>20.2</td><td>10.3</td><td>26.4</td><td>14.0</td></tr><tr><td rowspan="6">Memory</td><td>BM25</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Enr. BM25F</td><td>2.1</td><td>2.5</td><td>2.1</td><td>2.4</td><td>2.3</td><td>1.8</td><td>4.0</td><td>2.5</td></tr><tr><td>Gemma</td><td>0.4</td><td>0.4</td><td>0.5</td><td>0.3</td><td>0.6</td><td>2.4</td><td>0.9</td><td>0.8</td></tr><tr><td>ColQwen2.5</td><td>56.0</td><td>43.6</td><td>59.0</td><td>42.7</td><td>76.7</td><td>72.4</td><td>105.0</td><td>65.0</td></tr><tr><td>Enr. BM25F+Gemma</td><td>3.0</td><td>2.8</td><td>2.9</td><td>2.8</td><td>3.6</td><td>3.3</td><td>4.9</td><td>3.3</td></tr><tr><td>MIDR Hybrid</td><td>6.6</td><td>5.6</td><td>6.7</td><td>5.5</td><td>8.4</td><td>7.9</td><td>11.7</td><td>7.5</td></tr></table>

Table 19: Per-domain query latency and index memory, normalized to BM25 on each domain. Aggregate values in the rightmost column match Table 4. Physics is the most expensive domain for every method; ColQwen2.5’s memory cost scales with page count and visual complexity, ranging from 42.7× (HR) to 105.0× (physics), while MIDR Hybrid stays within $5 . 5 { \times } { - } 1 1 . 7 { \times }$ across the same domains. “Enr. BM25F+Gemma” uses single-vector Gemma; MIDR Hybrid uses mean-pooled (MP) Gemma over enrichment fields.

<table><tr><td>Ingestion metric</td><td>Per page</td></tr><tr><td>MLLM calls</td><td>2.10</td></tr><tr><td>Input tokens</td><td>6,257</td></tr><tr><td>Output tokens</td><td>1,914</td></tr><tr><td>Wall-clock (serial)</td><td>23.2 s</td></tr><tr><td>Refinement rate</td><td>9.6%</td></tr></table>

Table 20: Offline ingestion cost of the extract–verify– refine pipeline, averaged over all 16,867 ViDoRe V3 pages with GPT-5.1 as the enrichment MLLM. This is a one-time index-build expense, amortized over all future queries.

To isolate what the rendered page image contributes over extracted text alone, we re-ran the entire en-

## J Contribution of the Page Image

<table><tr><td>Domain</td><td>Pages</td><td>Calls/pg</td><td>In tok/pg</td><td>Out tok/pg</td><td>Refine</td></tr><tr><td>Computer Science</td><td>1,360</td><td>2.07</td><td>5,474</td><td>1,443</td><td>6.6%</td></tr><tr><td>Finance (EN)</td><td>2,942</td><td>2.03</td><td>6,446</td><td>1,765</td><td>2.7%</td></tr><tr><td>HR</td><td>1,110</td><td>2.52</td><td>10,951</td><td>7,198</td><td>52.3%</td></tr><tr><td>Industrial</td><td>5,244</td><td>2.08</td><td>5,817</td><td>1,590</td><td>7.9%</td></tr><tr><td>Pharmaceuticals</td><td>2,312</td><td>2.06</td><td>5,335</td><td>1,243</td><td>5.8%</td></tr><tr><td>Energy (FR)</td><td>2,225</td><td>2.06</td><td>6,698</td><td>1,661</td><td>6.1%</td></tr><tr><td>Physics (FR)</td><td>1,674</td><td>2.11</td><td>5,516</td><td>1,333</td><td>11.2%</td></tr><tr><td>Total</td><td>16,867</td><td>2.10</td><td>6,257</td><td>1,914</td><td>9.6%</td></tr></table>

Table 21: Per-domain ingestion cost. “Refine” is the fraction of pages on which verification reported an issue and the refiner was invoked. HR is a strong outlier at 52.3%, which also makes it the most expensive domain per page; finance, with clean extracted text, refines on 2.7% of pages.

richment pipeline with the page image withheld from the extractor, verifier, and refiner, holding everything else fixed. The document-level enricher never reads page images—it operates on extracted page text only—so the resulting pipeline is end-toend text-only, and the difference between the two runs is attributable to the page image.

Because the paired runs were evaluated on the same machine, we also report the full-enrichment control from that machine alongside the cached values used elsewhere in the paper; the device difference is at most 0.0043 nDCG@10 on any domain, which is why cached results are used for the rest of the paper.

<table><tr><td>Domain</td><td>OCR-only</td><td>Full</td><td>Image ∆@10</td><td>∆@5</td></tr><tr><td>Pharmaceuticals</td><td>0.6137</td><td>0.6483</td><td>+0.0346</td><td>+0.0314</td></tr><tr><td>Computer Science</td><td>0.7061</td><td>0.7191</td><td>+0.0131</td><td>+0.0132</td></tr><tr><td>HR</td><td>0.5953</td><td>0.6009</td><td>+0.0055</td><td>+0.0033</td></tr><tr><td>Finance (EN)</td><td>0.6314</td><td>0.6338</td><td>+0.0024</td><td>+0.0059</td></tr><tr><td>Industrial</td><td>0.5199</td><td>0.5206</td><td>+0.0008</td><td>+0.0046</td></tr><tr><td>EN-5 avg</td><td>0.6133</td><td>0.6245</td><td>+0.0113</td><td></td></tr><tr><td>Energy (FR)</td><td>0.5943</td><td>0.6085</td><td>+0.0142</td><td>+0.0201</td></tr><tr><td>Physics (FR)</td><td>0.4590</td><td>0.4637</td><td>+0.0047</td><td>+0.0005</td></tr></table>

Table 22: Hybrid nDCG@10 with and without the page image at enrichment time, both measured on the same machine. The image contributes most where OCR mangles content—chemical and scientific notation on pharmaceuticals, code and garbled tables on computer science—and least on clean extracted text, including industrial, which has the highest proportion of table pages in the corpus. French domains are reported separately because the cross-lingual bridge confounds the comparison.

Withholding the image is also a poor cost saving. On finance it reduces input tokens by 18% but leaves output tokens unchanged, while the verifier’s refinement rate rises from 2.7% to 5.2%: without visual grounding the verifier finds more to correct, and the extra refinement calls claw back part of the saving.

## K Enrichment MLLM Sensitivity

Our main results use GPT-5.1 as the enrichment MLLM. To assess how much of MIDR’s gain depends on this specific choice, we re-ran the full extract–verify–refine pipeline with three alternative models—GPT-5.4, Claude Sonnet 4.5, and the open-source Qwen3-Omni-30B-A3B (Xu et al., 2025)—and reindexed each corpus from scratch. All other components (retrieval, fusion, embeddings, weights) are held fixed.

Frontier MLLMs cluster tightly. On English-5, the three frontier models span 0.6120–0.6231 nDCG@10 (2.5% relative best-to-worst), well inside the BM25 → MIDR gain of 23.2%. MIDR’s improvement therefore comes principally from the enrichment-augmented indexing design rather than from any specific MLLM. GPT-5.1 leads on computer science, HR, and industrial; Claude Sonnet 4.5 leads on finance and pharmaceuticals (the largest single-domain margin in the table, +0.013 nDCG@10 over GPT-5.1) and on both French domains. The aggregate ordering should be read with some caution: prompting, decoding temperature, and structured-output configuration can each shift any single model by amounts comparable to the deltas in Table 23, and GPT-5.4’s slight regression relative to GPT-5.1 is plausibly within that variance band rather than a stable ranking.

<table><tr><td>Domain</td><td>GPT-5.1</td><td>GPT-5.4</td><td>Claude S4.5</td><td>Qwen3-O.</td></tr><tr><td colspan="5">English domains</td></tr><tr><td>Computer Sci.</td><td>0.7178</td><td>0.7065</td><td>0.7078</td><td>0.693</td></tr><tr><td>Finance</td><td>0.6322</td><td>0.6224</td><td>0.6351</td><td>0.574</td></tr><tr><td>HR</td><td>0.5990</td><td>0.5921</td><td>0.5946</td><td>0.526</td></tr><tr><td>Industrial</td><td>0.5212</td><td>0.5087</td><td>0.5154</td><td>0.466</td></tr><tr><td>Pharma</td><td>0.6454</td><td>0.6305</td><td>0.6584</td><td>0.627</td></tr><tr><td>EN-5 avg</td><td>0.6231</td><td>0.6120</td><td>0.6223</td><td>0.5772</td></tr><tr><td colspan="5">French domains</td></tr><tr><td>Energy</td><td>0.6084</td><td>0.5997</td><td>0.6143</td><td>0.294</td></tr><tr><td>Physics</td><td>0.4594</td><td>0.4485</td><td>0.4660</td><td>0.302</td></tr><tr><td>FR-2 avg</td><td>0.5339</td><td>0.5241</td><td>0.5402</td><td>0.298</td></tr></table>

Table 23: MIDR Hybrid nDCG@10 when the enrichment MLLM is varied, holding the rest of the pipeline fixed. Bold marks the best MLLM per row. The three frontier models cluster within roughly 2.5% relative on English-5; Qwen3-Omni-30B-A3B trails by a wider margin and performs substantially worse on French.

Open-source MLLM: verifier calibration. Qwen3-Omni-30B-A3B trails the frontier cluster by 4–5 points on English-5 (0.5772 vs. 0.6120– 0.6231, −7.9% relative to GPT-5.1) and performs substantially worse on French at 0.298. Two factors compounded. Qwen3-Omni had lower structured-output reliability in our setup (a non-trivial fraction of calls returned malformed JSON, requiring retries). More interestingly, its verifier was systematically over-strict relative to its own extractor: borderline-but-supported fields were flagged as unsupported, and the refiner shrank or dropped them, producing indexes sparser than the unverified drafts. GPT-5.1 showed the opposite pattern, with verification typically expanding coverage where extraction was conservative. The verifier in extract–verify–refine therefore becomes a limiting factor on enrichment volume, with tightness determined by verifier calibration rather than by the loop itself. We treat the Qwen3-Omni numbers as suggestive, but the qualitative direction is consistent across our runs.

<table><tr><td>Region</td><td>Count</td><td>Interpretation</td></tr><tr><td>MIDR wins by &gt; 0.1 nDCG@10</td><td>351</td><td>Enriched facts/QA expose evidence.</td></tr><tr><td>ColQwen2.5 wins by &gt; 0.1 nDCG@10</td><td>362</td><td>Visual matching preserves structure.</td></tr><tr><td>Both succeed (&gt; 0.5 nDCG@10)</td><td>522</td><td>Shared easy/evident queries.</td></tr><tr><td>Both fail (&lt; 0.3 nDCG@ 10)</td><td>160</td><td>Multi-page or cross-document gaps.</td></tr><tr><td>Marginal gap (&lt; 0.1 nDCG@ 10)</td><td>94</td><td>Practically tied.</td></tr></table>

Table 24: Per-query partition of MIDR Hybrid vs. reproduced ColQwen2.5 across the 1,489 English-domain queries.

Implication. Extract–verify–refine does not require the same model in both roles. Pairing a frugal open-source extractor with a stronger verifier (or vice versa) is a natural way to trade ingestion cost against enrichment quality without retraining. A systematic study of mixed extractor/verifier configurations is left to future work.

## L Per-Query Complementarity

Table 24 partitions the 1,489 English-domain queries by per-query nDCG@10 gap between MIDR Hybrid and our local ColQwen2.5 reproduction. Roughly half of queries (48%) have a decisive winner at the > 0.1 nDCG@10 threshold, split nearly evenly between MIDR (351) and ColQwen2.5 (362). Only 6% fall within a marginal < 0.1 gap; the rest both succeed (35%) or both fail (11%).

## M Oracle Complementarity with ColQwen2.5

Per-domain oracle gains on English (Table 25) and French (Table 26). French oracle gains (+0.08 nDCG@10 over either system) match the English pattern, indicating complementarity extends to the cross-lingual setting.

## N Implementation Details

We document the package versions and configuration parameters used across the retrieval, embedding, and evaluation stack to support reproduction.

Lexical retrieval (BM25 and BM25F). BM25 over markdown and BM25F over enriched fields are both implemented using Whoosh’s <sup>1</sup> scoring framework. Tokenization uses Whoosh’s default analyzer (lowercasing, standard stopword removal, no stemming applied to enrichment fields to preserve named entities and domain-specific tokens).

BM25 parameters are left at their library defaults $( k _ { 1 } = 1 . 2 , b = 0 . 7 5 )$ , and the same $k _ { 1 }$ and b apply per field for BM25F.

Field weights. All results in the main paper use uniform BM25F field weights, with every field set to 1.0. ViDoRe V3 ships no development split, so tuning field weights would mean fitting the evaluation queries; we therefore do not tune them. As a robustness check we also evaluated a role-based weighting assigned a priori from each field’s intended retrieval role—page text 1.0, keyphrases and main entities 1.2, document focus 1.1, topic tags 0.9, table and chart summaries 0.8, coarse and fine QA fields 0.7. Table 27 compares the two. The role-based scheme is 0.0012 better on the English aggregate and 0.0109 worse on French, and no perdomain difference exceeds 0.011, so the schema is insensitive to this choice. Field boosts remain a per-deployment knob for practitioners who do have a validation set.

Dense retrieval. Page text and each enrichment field are embedded with EmbeddingGemma using the transformers library for model loading and inference. Field embeddings are combined by mean pooling and L2-normalized. The dense index is a FAISS IndexFlatIP (exact innerproduct search, no quantization or approximatesearch structures), built per domain over all pages in the domain’s candidate pool.

Multi-vector retrieval (ColQwen2.5 reproduction). For the local ColQwen2.5 reproduction used in the per-query complementarity analysis (Section 5), we load the released checkpoint via transformers and score candidates with the published late-interaction scoring routine. Patch-level multi-vector representations are stored uncompressed.

Fusion. Hybrid retrieval combines BM25F and dense rankings using Reciprocal Rank Fusion (Cormack et al., 2009) with the standard constant k = 60.

Evaluation. All retrieval metrics are computed with ir-measures (MacAvaney et al., 2022) against the official ViDoRe V3 qrels. We report nDCG@10 (the primary metric defined by ViDoRe V3), evaluated at cutoff 10 across all systems and configurations.

Enrichment generation. Multimodal LLM calls (GPT-5.1, GPT-5.4, GPT-5.4-mini, Claude Sonnet

<table><tr><td>Domain</td><td>MIDR</td><td>ColQwen2.5</td><td>Oracle</td><td>Gain over MIDR</td><td>Gain over ColQwen2.5</td></tr><tr><td>Computer Science</td><td>0.7178</td><td>0.7623</td><td>0.8119</td><td>+0.0941</td><td>+0.0497</td></tr><tr><td>Finance</td><td>0.6322</td><td>0.6276</td><td>0.7221</td><td>+0.0898</td><td>+0.0944</td></tr><tr><td>HR</td><td>0.5990</td><td>0.6018</td><td>0.6825</td><td>+0.0835</td><td>+0.0808</td></tr><tr><td>Industrial</td><td>0.5212</td><td>0.5200</td><td>0.5948</td><td>+0.0736</td><td>+0.0748</td></tr><tr><td>Pharmaceuticals</td><td>0.6454</td><td>0.6382</td><td>0.7095</td><td>+0.0641</td><td>+0.0713</td></tr><tr><td>Average</td><td>0.6231</td><td>0.6300</td><td>0.7042</td><td>+0.0810</td><td>+0.0742</td></tr></table>

Table 25: Oracle complementarity analysis on the five English ViDoRe V3 domains. A per-query oracle that selects the better of MIDR and ColQwen2.5 reaches 0.7042 average nDCG@10, +0.0810 over MIDR alone and +0.0742 over ColQwen2.5, indicating that the two systems succeed and fail on different queries. The MIDR column uses role-based field weights, which differ from the uniform configuration reported in Table 2 by 0.0012 on the English aggregate. Averages are simple means across the five domains.
<table><tr><td>Domain</td><td>MIDR</td><td>ColQwen2.5</td><td>Oracle</td><td>Gain over MIDR</td><td>Gain over ColQwen2.5</td></tr><tr><td>Energy</td><td>0.6084</td><td>0.5967</td><td>0.6891</td><td>+0.0807</td><td>+0.0924</td></tr><tr><td>Physics</td><td>0.4594</td><td>0.4663</td><td>0.5407</td><td>+0.0813</td><td>+0.0744</td></tr><tr><td>Average</td><td>0.5339</td><td>0.5315</td><td>0.6149</td><td>+0.0810</td><td>+0.0834</td></tr></table>

Table 26: Oracle complementarity analysis on the two French ViDoRe V3 domains. MIDR and ColQwen2.5 are within ±0.012 nDCG@10 of each other on both domains, and the oracle gains (+0.0810 over MIDR, +0.0834 over ColQwen2.5) are comparable to the English case in Table 25—confirming that the complementarity is not an artifact of either source language.

<table><tr><td>Domain</td><td>Uniform</td><td>Role-based</td><td>∆</td></tr><tr><td>Computer Science</td><td>0.7170</td><td>0.7178</td><td>+0.0008</td></tr><tr><td>Finance (EN)</td><td>0.6261</td><td>0.6322</td><td>+0.0061</td></tr><tr><td>HR</td><td>0.6043</td><td>0.5990</td><td>-0.0053</td></tr><tr><td>Industrial</td><td>0.5197</td><td>0.5212</td><td>+0.0015</td></tr><tr><td>Pharmaceuticals</td><td>0.6424</td><td>0.6454</td><td>+0.0030</td></tr><tr><td>EN-5 avg</td><td>0.6219</td><td>0.6231</td><td>+0.0012</td></tr><tr><td>Energy (FR)</td><td>0.6192</td><td>0.6084</td><td>-0.0108</td></tr><tr><td>Physics (FR)</td><td>0.4704</td><td>0.4594</td><td>-0.0110</td></tr><tr><td>FR-2 avg</td><td>0.5448</td><td>0.5339</td><td>-0.0109</td></tr></table>

Table 27: Hybrid nDCG@10 under uniform and a-priori role-based BM25F field weights. ∆ is role-based minus uniform; positive means role-based is better. Uniform is reported throughout the paper.

4.5, Qwen3-Omni-30B-A3B) use each provider’s structured-output mode where available, with low decoding temperature to favor deterministic extraction.

## O Enrichment Prompts

This appendix documents the prompts used by the extract–verify–refine pipeline that produces MIDR’s page-level enrichments (Section 3). The pipeline runs three stages per page: an extraction call that generates a draft enrichment from page text and image; a verification call that audits the draft against the page; and a conditional refinement call that fixes only the issues flagged by the verifier. Each stage uses a system prompt and a user-message template; the templates below show the prompts that produced all results reported in this paper.

## O.1 Extraction prompts

The extraction system prompt is composed of a domain-specific header followed by a shared rule block (Figure 3). We use eight domain headers: one per ViDoRe V3 domain (finance, computer science, energy, HR, industrial, pharmaceuticals, physics) plus a general fallback. Two representative headers are shown in Figures 4 and 5; the remaining six follow the same structure (one paragraph of domain-specific guidance covering terminology, units, table conventions, and topic-tag priorities) and are available in the accompanying release. The user-message template (Figure 6) is shared across all domains and is sent alongside the page image as a multimodal input.

Extraction - Shared rule block (appended to every   
domain header)   
CRITICAL RULES — follow these exactly:   
1. Use ONLY information explicitly present on THIS PAGE.   
2. Document-level context (type, focus, entities) is for   
grounding only — never invent facts from it.

![](images/b43bb54f87611435e7fd2a3f30d86f3835dce0321108f0ee5271ef96db86e0ee.jpg)

Figure 3: Shared rule block appended to every domainspecific header in the extraction system prompt. Defines layout categories, signal-quality labels, summary constraints, and QA structure.  
![](images/cbcb07cff038cbaeae286eb8fb7a5d4d5eeae933026e1218e834eea16a03cb85.jpg)

Figure 4: Domain header for finance, prepended to the shared rule block in Figure 3. Other English-domain headers (computer science, HR, industrial, pharmaceuticals) follow the same structure.  
![](images/88d647f0d70a1d1feb0d44dc9706c8807d05ec72bda35c530d9aebf6e47a8cd8.jpg)

Figure 5: Headers for French-source domains (physics, energy) and the general fallback include an explicit LANGUAGE clause forcing English output for crosslingual retrieval.  
Extraction - User-message template (shared across   
domains)   
DOCUMENT CONTEXT:   
- Document type: {document\_type}   
- Document focus: {document\_focus}   
- Key entities: {main\_entities}   
PAGE TEXT:   
{page\_text}   
The PAGE IMAGE is also provided. Analyze both the text and   
image to produce all enrichment fields.  
Figure 6: Extraction user-message template. The three document placeholders are filled from a separate document-level enrichment pass (Section 3); the page image is attached as a multimodal input alongside this text.

## O.2 Verification prompt

After extraction, every draft enrichment is checked against the page by a second call configured with the prompt in Figure 7. The verifier walks a fivepoint checklist (layout consistency, fact grounding, internal consistency, answer quality, completeness) and returns a structured list of issues with field names, issue types, descriptions, and suggested fixes. It sets is\_consistent=true only when zero issues are found, in which case refinement is skipped. The user-message template is in

Verification - System prompt   
You are a meticulous verification agent. Your job is to   
check a draft page enrichment against the original page   
content (text + image) and identify any issues.   
VERIFICATION CHECKLIST:   
1. LAYOUT CONSISTENCY   
- If has\_table=true, does the page actually contain a   
table (rows + columns)?   
- If has\_chart=true, does the page actually contain a   
chart/visualization?   
- If has\_table=false, is table\_summary set to "none"?   
- If has\_chart=false, is chart\_summary set to "none"?   
- Bullet points and lists are NOT tables.   
2. FACT GROUNDING   
- Is every claim in the enrichment supported by the page   
text or image?   
- Are numerical values, dates, and entity names accurate?   
- Are QA answers grounded in page content (not   
hallucinated)?   
3. INTERNAL CONSISTENCY   
- Do QA pairs contradict each other?   
- Are there exact duplicate questions (in coarse\_qa,   
fine\_qa, or across both)?   
- Do topic tags and keyphrases match the page’s actual   
content?   
4. ANSWER QUALITY   
- Are answers contextual (value + context) or bare values?   
- Bare values like "42%" with no context are an issue.   
5. COMPLETENESS   
- Are obvious facts on the page missing from fine\_qa?   
- For tables: is each cell accounted for?   
For each issue found, specify:   
- field: Which enrichment field has the issue   
- issue\_type: One of "hallucination", "inconsistency",   
"missing\_context", "duplicate"   
- description: What is wrong   
- suggested\_fix: How to fix it   
Set is\_consistent=true ONLY if zero issues are found.

Figure 7: Verification system prompt. The fivepoint checklist mirrors the constraints of the extraction prompt, letting the same model audit drafts against the rules it was meant to follow.  
Verification - User-message template   
DRAFT ENRICHMENT (JSON):   
{draft\_json}   
PAGE TEXT:   
{page\_text}   
The PAGE IMAGE is also provided. Verify the draft enrichment   
against the page content.  
Figure 8: Verification user-message template. The draft enrichment produced by extraction is passed back to the model verbatim, alongside the original page text and image.

## O.3 Refinement prompt

When verification flags one or more issues, a third call applies targeted fixes using the prompt in Figure 9. Refinement is conditional: on pages where verification returns no issues, this stage is skipped and the draft enrichment is indexed as-is. The refiner is instructed to fix only the listed issues and to log every change in a changes\_made field, which feeds the refinement\_edits routing field in Table 9. The user-message template is in Figure 10.

Refinement - System prompt (runs only when verifi  
cation flags issues)   
You are a refinement agent. You receive a draft enrichment   
and a list of verification issues. Your job is to produce   
a corrected enrichment that fixes all identified issues   
while preserving everything that was correct.   
RULES:   
1. Fix ONLY the issues listed — do not make unnecessary   
changes.   
2. Preserve correct content exactly as-is.   
3. When fixing QA answers, ensure they include value AND   
context.   
4. When fixing layout issues, ensure summary fields are   
consistent with layout flags.   
5. When removing duplicates, keep the more   
detailed/contextual version.   
6. Log every change you make in the changes\_made list.   
7. Generate all outputs in English, preserving exact values   
and technical terms.

Figure 9: Refinement system prompt. The "fix only the listed issues" constraint is what gives extract–verify– refine its surgical behavior: the refiner is not free to rewrite the draft, only to patch flagged fields.  
Refinement - User-message template   
DRAFT ENRICHMENT (JSON):   
{draft\_json}   
VERIFICATION ISSUES:   
{issues\_json}   
PAGE TEXT:   
{page\_text}   
The PAGE IMAGE is also provided. Fix the identified issues   
while preserving correct content.  
Figure 10: Refinement user-message template. The model receives both the original draft and the structured list of issues from verification, plus the page text and image for re-grounding.

## P Qualitative Analysis: How Enrichments Improve Retrieval

To illustrate the mechanism by which MIDR enrichments improve retrieval, we present two case studies drawn from the finance and computer science evaluation domains. In each case, we trace a query that the baseline BM25 system (operating on raw markdown text only) fails to retrieve correctly, and show how the enriched fields bridge the gap.

## P.1 Finance Domain: Vocabulary Mismatch on Restructuring Charges

Query. “What were the total restructuring charges for the year 2020?”

Gold page. Wells Fargo & Company 2021 Annual Report (NYSE\_WFC\_2021), page 204.

Retrieval results.

• Baseline BM25 (markdown only): target page not retrieved in top 100 (nDCG@10 = 0.0).

• Enriched BM25F (with MIDR fields): target page retrieved at rank 1 (nDCG@10 = 1.0).

Why the baseline fails. The page contains a detailed discussion of Wells Fargo’s restructuring initiatives and an accrual table (“Accruals for Restructuring Charges”), but the raw markdown text is dominated by descriptions of personnel costs, facility closures, and accounting methodology. The specific phrase “total restructuring charges for the year 2020” does not appear verbatim. Meanwhile, many other pages across the corpus mention “restructuring charges” in passing (e.g., Accenture, Texas Instruments, Nike), creating strong lexical competition from irrelevant documents.

How enrichments fix it. The MIDR pipeline generates several enrichment fields that directly address this query:

• Coarse QA: “What were Wells Fargo’s total restructuring charges and their components for the year ended December 31, 2020?” → “For the year ended December 31, 2020, Wells Fargo recorded total restructuring charges of \$726 million, including \$716 million of personnel costs. . . ,,

• Table summary: “Table 22.1, titled ‘Accrualsfor Restructuring Charges,’ presents Wells Fargo & Company’s restructuring-related accrual activity by category (Personnel costs, Facility closure costs, Other, and Total) for thefiscal years ended December 31, 2019 and 2020, in millions.. . . ”

• Document focus: “Wells Fargo & Company 2021 financial performance and CEO discussion ofstrategic, risk, and operational transformation.”

The coarse QA field is particularly effective here: it pre-generates a natural-language question that closely mirrors the user’s query, creating the lexical overlap that the raw text lacks. The table summary further reinforces relevance by explicitly mentioning the year 2020 and the “total” category.

## Extracted page text (abbreviated).

The Company began pursuing various initiatives to reduce expenses and create a more efficient and streamlined organization in third quarter 2020. Actions from these initiatives may include (i) reorganizing and simplifying business processes [...] (ii) reducing headcount, (iii) optimizing third-party spending [...]

Restructuring charges are recorded as a component of noninterest expense on our consolidated statement of income.

[...]

The following costs associated with these initiatives are included in restructuring charges:

\- Personnel costs: Severance costs associated with headcount reductions [...]

\- Facility closure costs: Write-downs and acceleration of depreciation [...]

## P.2 Computer Science Domain: Garbled Table Content

Query. “How do range(1, 5) and range(1, 5, 2) differ in output pattern?”

Gold pages. Introduction to Python Programming, pages 135–136 and 149 (Chapter 5: Loops, range() function reference table). Three pages are marked relevant in the ground truth.

## Retrieval results.

• Baseline BM25 (markdown only): no target page in top 10; first target at rank 21 (nDCG@10 = 0.0).

• Enriched BM25F (with MIDR fields): all three target pages at ranks 1, 2, and 3 (nDCG@10 = 1.0).

Why the baseline fails. The relevant pages contain a reference table (Table 5.1) that lists range() function call patterns with their outputs. However, the markdown extracted from the PDF via VLMbased conversion renders the table as garbled pipedelimited text with OCR artifacts. The specific calls range(1, 5) and range(1, 5, 2) do not appear in the markdown at all; the table shows different examples such as range(4), range(2, 6), and range(1, 7, 2). The word “differ” also does not appear. The baseline therefore cannot match this page to the query.

How enrichments fix it.

• Table summary: “The table titled ‘Using the range() function’ lists three Python range() function call patterns (range(end), range(start, end), and range(start, end, step)), each with a textual description, one or more example calls, and the resulting integer sequences. It emphasizes that sequences start at 0 or the given start value, end before the end value, and use a specified step size (default 1 or custom).” converts the garbled table into clean, searchable prose that introduces the key terms start, end, step, and pattern.

• Fine QA: “How does range(start, end, step) behave in terms of start, end, and step size according to the table?” → “The form range(start, end, step) generates a sequence beginning at start until end with a step size equal to step.” directly introduces the parametric framing that bridges to the user’s query about how two-argument and three-argument calls “differ.”

• Fine QA: “What output sequence is provided for the example call range(1, 7, 2)?” → “The example range(1, 7, 2) produces the sequence 1, 3, 5.” surfaces a concrete step-2 example with its output, providing lexical overlap with the query’s range(1, 5, 2).

The table summary and fine QA fields serve as a readable proxy for the tabular content that was lost during PDF-to-text conversion. The VLM that generates enrichments can interpret the table visually from the page image, recovering structured information that text extraction alone cannot.

Extracted page text (abbreviated).

```matlab
| Range() function in for loop | ...
| | A for loop can be used for iteration and
counting. The range() function is a common
approach for implementing counting for loop.
function of between the in a A range()
generates a sequence integers two numbers
given size. [...]
| Range function | Description | Example | Output |
| range (end) | Generates a sequence beginning
at 0 until end. Step size: 1 | range (4) |
0, 1, 2, 3 |
| | | range(0) 3) | 0, 1, 2 |
| | | range(2, 6) | 2, 3, 4, 5 |
| lange(start, end) | Generates a sequence
beginning at start until end. Step size: 1 |
range(-13, -9) | -13, -12, [...]
```

## P.3 Discussion

These examples illustrate two complementary mechanisms by which MIDR enrichments improve lexical retrieval:

1. Vocabulary bridging. The coarse and fine QA fields pre-generate natural-language questions that mirror how users formulate queries, bridging the gap between query vocabulary (e.g., “total restructuring charges,” “differ in output pattern”) and document vocabulary (e.g., accounting methodology prose, garbled table markup).

2. Structured content recovery. The table summary field converts tabular and visual content, which is often poorly captured by PDF-to-text extraction, into clean, searchable prose. In the finance example, the table summary surfaces year-specific totals; in the computer science example, it recovers the semantics of a reference table that was garbled during text extraction.

In both cases, the enrichments do not add new information; the answers are present on the original pages. Instead, they re-express the page content in the vocabulary and structure that users naturally employ when searching. The finance example demonstrates vocabulary mismatch across documents, while the computer science example demonstrates information loss during text extraction from visually structured content.