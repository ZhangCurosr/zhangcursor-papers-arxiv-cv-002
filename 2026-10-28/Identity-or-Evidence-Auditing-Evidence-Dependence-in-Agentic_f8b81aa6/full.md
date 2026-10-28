# Identity or Evidence? Auditing Evidence Dependence in Agentic Life-Science Benchmarks

Danai Brilli Zo Labs dbrilli@zolabs.bio

Maria Georgila Zo Labs mgeorgila@zolabs.bio

## Abstract

Agentic life-science benchmarks give an agent evidence about a named entity and score an answer against a recorded outcome. That score tests reasoning over evidence only if the evidence, rather than the name, determines the answer. We examine this distinction by varying identity and evidence independently on a clinical-outcome benchmark and published tasks from Biomni and LAB-Bench. Blinding identity matters little when a prompt contains a decisive sequence or passage. It matters a great deal when the prompt is mostly a name and candidate list (0.742 named versus 0.082 blinded; chance 0.093). Across 96 LAB-Bench LitQA2 questions, the gap increases under one fixed passage-redaction scheme (+0.000 to +0.135; per-instance slope +0.147, $p = 0 . 0 0 4 )$ . On the clinical benchmark, a noevidence recall probe matches prediction accuracy and recovers a database-specific programme count. So, we argue that benchmarks intended to measure evidencedependent reasoning should report identity-blinded results, retrieval telemetry, and positive controls.

![](images/31aef54a07d227eaf584d65982315e65975a11ddd8db527649196c5d01b1de4c.jpg)

![](images/d651a0ca3f5f5b4151e0f46a82cbbb9ba8acad8a9774505ad1b7666e0e004f04.jpg)  
Figure 1: The cost of identity blinding varies with available evidence. Left: conditions within a row differ only in entity mentions; numeric evidence is byte-identical. Right: on 96 LitQA2 questions, the supporting passage is redacted in four steps while entity mentions remain intact. The blinding gap grows with redaction (per-instance slope +0.147, $p \overset { ^ { \mathrm { ~ ~ } } } { = } 0 . 0 0 4 )$

Table 1: Interpretive status of the main analyses.
<table><tr><td>analysis</td><td>status</td><td>evidence</td><td>key limitation</td></tr><tr><td>LitQA2: full pas- primary sage</td><td></td><td>0.990 named and blinded at 0% redaction</td><td>96 questions; one model</td></tr><tr><td>LitQA2: redac- primary tion gradient</td><td></td><td>identity-gap slope +0.147, p 二 0.004</td><td>one seeded mask</td></tr><tr><td>Clinical dossier: blind</td><td>supporting</td><td>0.531 blinded versus 0.783 as written</td><td>semantic context also removed</td></tr><tr><td>Clinical dossier: recall</td><td>supporting</td><td>0.818 all-case accuracy; count ρ = 0.654</td><td>not a per-prediction mechanism test</td></tr><tr><td>Deletion controls</td><td>diagnostic</td><td>controls are inconclusive</td><td>no attribution is claimed</td></tr></table>

## 1 Introduction

Agentic biology benchmarks reconstruct evidence available at a scientific decision point, remove the conclusion, and score an agent against the eventual recorded outcome. This design spans molecular discovery, genomics, single-cell analysis, and laboratory tasks [3, 4, 5, 9, 6, 8, 7, 10, 13, 19, 11, 12, 14]. Its intended construct is reasoning from supplied evidence. But if a prompt retains a recognisable gene, drug, disease, or study, an agent may instead use prior knowledge or retrieve the recorded answer.

We ask a simple question: how much of the score survives when the name disappears and the evidence stays put? The answer differs across the tasks we audited. Identity is largely irrelevant when the prompt itself supplies the needed evidence, but it becomes important when it does not.

Contributions. The paper introduces paired prompt variants that separate identity, evidence, and record recall. It then uses LitQA2 to show that the cost of hiding identity rises as passage evidence is removed. Finally, it shows why deleting evidence alone is a poor test of content use: deletion changes the prompt in several ways at once.

## 2 Related work and protocol

Partial-input baselines have exposed hypothesis-only inference and passage-free comprehension [17, 20, 21], knowledge-conflict and contamination probes test related failures [23, 26]. Feng et al. [22] make the complementary point: changing an input does not, by itself, identify what caused a model’s response. We treat this as a discriminant-validity problem [15], drawing on work about benchmark scrutiny, confidence, irrelevant context, and statistical power [29, 30, 27, 28, 24, 25, 16].

From an instance with evidence E, identity I, and outcome, the audit creates paired variants. BLIND holds E fixed while removing I. DELETE holds I fixed while removing E; PLACEBO replaces E with another instance’s evidence. RECALL is run only after blinding bites, to ask whether identity gives access to the recorded label. Positive controls verify that the instrument can transmit a decisive change. Every contrast is paired over the same instances and reported with an exact McNemar test, Wilson interval, and minimum detectable effect (MDE) for nulls [16]. The LitQA2 full-passage comparison and redaction slope are primary analyses, while clinical blind and recall are supporting evidence. Full prompts, raw responses, and scripts are supplementary.

What each contrast estimates. Let $A ( E , I )$ denote paired accuracy under evidence E and identity I. The identity cost is $A ( E , I ) - A ( E , \emptyset ) \colon$ : it is positive when the named prompt outperforms an otherwise identical blinded prompt. The evidence contrast is $A ( E , I ) - A ( \emptyset , I )$ , but it is not a content-use estimand by itself: removing a block also changes length, formatting, and the option to use that block. A placebo comparison is more informative only when the replacement has known signal. Thus a blinded contrast can establish dependence on the disclosed identity under this intervention. It cannot decide whether that dependence reflects memorised records, biological knowledge, or the loss of semantic context. The recall probe and functional-description diagnostic help distinguish these explanations to some extent.

Table 2: Core audit conditions. E is the evidence block and I the entity identity. All contrasts are paired, while the synthetic positive control measures the instrument, not the benchmark.
<table><tr><td>condition</td><td>what it changes</td><td>what a movement means</td></tr><tr><td>AS-WRITTEN</td><td>nothing</td><td>the benchmark&#x27;s own number, under our harness</td></tr><tr><td>BLIND</td><td>I → opaque salted tokens</td><td>does the score survive without the name?</td></tr><tr><td>DELETE</td><td>E → a note that there is none</td><td>does the score use the evidence at all?</td></tr><tr><td>PLACEBO</td><td>E → another instance&#x27;s evidence</td><td>if this costs what deletion costs, not content</td></tr><tr><td>RECALL</td><td>E and the question cut; record asked for</td><td>does the agent already know the label?</td></tr><tr><td>POSITIVE</td><td>a decisive line, right 95% of the time</td><td>can the instrument detect evidence?</td></tr></table>

## 3 Experimental setup

Benchmark and label. From Minikel et al. [1, 2], we construct 286 target–indication instances (286 genes, 183 indications; prevalence 0.521), stratified by outcome, genetic support, and prominence. Each dossier contains association counts, MeSH similarity, provenance, years, locus-to-gene scores, allele frequencies, and twelve similar associations,while outcome-derived fields are excluded by assertion. The task is whether a programme completed Phase II advances beyond it. Genetically supported pairs are over-sampled, so we make no population-level claim. The label is a snapshotlevel, censored highest recorded phase, not a measure of scientific decision quality: it aggregates programmes, labels “not advanced” rather than failure, and has no per-programme date.

Field, signal, and execution checks. Four rendered labels are materially less precise than their source quantities: “AF” is ALT-allele frequency rather than minor-allele frequency; “fine-mapping share” is a normalised Open Targets L2G score rather than a posterior, exact MeSH identity is not biological relatedness and 174 of 801 counted associations bypass the displayed strict-support threshold. These issues do not alter paired contrasts with the same evidence block, but limit interpretation of deletion and placebo conditions.

The rendered Rare/orphan indication field is a programme-level regulatory designation, not an intrinsic property of an indication. It reproduces most of the simple supervised baseline, whereas genetics alone has little signal:
<table><tr><td>features (gene-grouped 5-fold CV)</td><td>population</td><td>286-pair subset</td></tr><tr><td>majority class</td><td>0.569</td><td>0.523</td></tr><tr><td>orphan flag only</td><td>0.600</td><td>0.582</td></tr><tr><td>17 genetics features, no orphan flag</td><td>0.579</td><td>0.495</td></tr><tr><td>genetics plus orphan flag</td><td>0.610</td><td>0.488</td></tr></table>

We use claude-opus-5 and repeat selected conditions on a smaller model from the same family. Grading is deterministic and offline: 5,475 decisions, 350 recall-probe tasks, and 81 externalreplication calls parse successfully. A prespecified 150-instance rerun used fresh agents and complete per-call logs: it found no retrieval or browser calls and reproduced the named, blinded, and no-evidence pattern (0.760, 0.560, and 0.807). This is a stronger execution check, not a sealed environment: the primary filesystem was not condition-sandboxed, serving and decoding settings were uncontrolled, and network use was prohibited by instruction in the original campaign. Named, blinded, and functional variants change only target and indication fields, keeping the evidence block is byte-identical.

The rerun validates execution only for the logged subset. It does not establish the serving model’s weights, rule out all information channels in the primary run, or turn instruction-based network restrictions into a sandbox. We report these constraints because a benchmark audit should make its own execution assumptions visible. Two conditions stopped early and remain exploratory and all other reported clinical conditions cover the full 286 instances.

Interpretation and reporting. We report accuracy because it is the benchmark’s score, but do not interpret it as calibrated probability or scientific utility. The clinical label is imbalanced, so a blinded score near 0.521 must be read against both the majority baseline and the paired named–blinded contrast. For the published tasks, chance values depend on their answer spaces. The key question is whether the above-chance margin persists after identity is removed. We designate the LitQA2 full-passage contrast and its redaction gradient as primary because they directly hold the passage and entity manipulation apart. Other contrasts are included to map failure modes, not to support a single pooled conclusion.

## 4 Results

## 4.1 Entity blinding on the clinical-outcome benchmark

BLIND replaces the gene, indication, trait, and source names with opaque per-instance tokens, keeping all numeric dossier fields unchanged. With the full dossier, accuracy falls from 0.783 as written to 0.531 [0.474, 0.589], near the 0.521 majority baseline. With identity retained but the dossier removed, it reaches 0.836 [0.788, 0.874]. Thus, in this clinical task, the score is not recovered by the blinded dossier (Figure 1).

This is a 25.2-point named–blinded gap on the same 286 decisions, not a comparison across separate samples. The result says that the particular rendered dossier does not preserve the performance of the named prompt under opaque-token blinding. It does not say that every numeric field is ignored: the blind transformation removes useful referential and semantic structure together with the entity name. The no-dossier condition also should not be read as a clean “name-only” baseline, because its shorter format changes the task. It establishes, however, that high accuracy remains possible without the supplied dossier.

Blinding also removes semantic context, so this result does not show that names simply substitute for evidence. The functional-description condition is therefore diagnostic rather than a clean identity control. That caution is warranted: an adversarial probe recovers the exact HGNC symbol in 114 of 120 descriptions, but in only 2 of 72 blinded prompts:

<table><tr><td>variant</td><td>probes</td><td>exact gene recovered</td><td>mean confidence</td></tr><tr><td>opaque tokens</td><td>72</td><td>2/72</td><td>0.044</td></tr><tr><td>functional description</td><td>120</td><td>114/120</td><td>≈ 0.70</td></tr></table>

## 4.2 Recall probe

The identity result has two explanations: identity may supply biological knowledge or access to the recorded outcome. A no-dossier recall probe asks for the record rather than a prediction and permits UNKNOWN. It collects recognition, recorded phase, and programme count for 286 benchmark pairs, 32 real non-clinical pairs, and 32 absent pairs. The outcome and task instruction are removed and opaque identifiers prevent an answer key in task files.

Recall reproduces prediction. The answered-only comparison is least score-dependent; coding UNKNOWN as “not advanced” is a sensitivity analysis:

<table><tr><td></td><td>accuracy</td><td>n</td></tr><tr><td>recall, substantive answers only</td><td>0.836 [0.784, 0.878]</td><td>238</td></tr><tr><td>no-dossier prediction, same pairs</td><td>0.849</td><td>238</td></tr><tr><td>paired difference</td><td>−0.013 (p = 0.61)</td><td></td></tr><tr><td>answer agreement</td><td>0.937</td><td></td></tr><tr><td>recall, all pairs (UNKNOWN = not advanced)</td><td>0.818 [0.769, 0.859]</td><td>286</td></tr></table>

![](images/40c69c777b50a9336065cc128611ec52871f4dda1601e1341895f8c1213186da.jpg)  
Figure 2: Recall tracks a database-specific count. Among 218 numeric recall responses, the recalled number of programmes rises with the number recorded in the source table $( \rho = 0 . 6 5 4 )$ . Medians and interquartile ranges are shown.

The recall task produces almost the same answers without supplied evidence. On 218 numeric answers, recalled programme count tracks the database count (Spearman $\rho = 0 . 6 5 4$ , permutation $p < 0 . 0 0 0 5 )$ , including within prominence strata. This is a commercial bookkeeping quantity, not something biological plausibility determines. The probe is consistent with access to recorded-outcome information, though it cannot establish what drove any one prediction.

Recognition is reported for 73.4% of benchmark pairs, 37.5% of real non-clinical pairs, and 6.3% of absent pairs, with parallel differences in recalled clinical status. The prespecified instance-level association between recognition and no-dossier prediction is null (Fisher $p = 0 . 3 7 )$ . A temporal or unpublished-outcome holdout would be preferable, but this dataset cannot support one because every outcome comes from one undated 2023 snapshot. Highest phase is partly public through trial registries, making the programme count the more specific discriminator.

## 4.3 Why evidence deletion is not interpretable

A synthetic line that agrees with the label 95% of the time yields 0.962 accuracy, while the same line with an independent value yields 0.479. The agent follows the line on 99.0% and 100% of tasks, respectively, at similar stated confidence. The positive control therefore detects a directive, not necessarily distributed dossier content. Deleting the dossier raises accuracy from 0.783 to 0.836 (difference −0.052, 95% CI [−0.085, −0.020], p = 0.0026), but it also changes prompt length, format, and the evidence slot. A swapped-dossier placebo and presentation controls do not identify which component drives the effect, so the placebo substitutes a low-signal genetics block.

## 4.4 Confidence and published benchmark audits

Across ten conditions spanning 63 points of accuracy, mean stated confidence moves only 12 points (slope +0.068). The agent is under-confident when it performs well and over-confident when blinded:

<table><tr><td>condition</td><td>accuracy</td><td>confidence</td><td>confidence minus accuracy</td></tr><tr><td>name only</td><td>0.836</td><td>0.722</td><td>-0.114</td></tr><tr><td>full dossier</td><td>0.783</td><td>0.725</td><td>-0.058</td></tr><tr><td>blinded</td><td>0.531</td><td>0.639</td><td>+0.107</td></tr><tr><td>false identity</td><td>0.202</td><td>0.711</td><td>+0.509</td></tr></table>

Because the own dossier has limited signal, we also audit published tasks that score well above chance. These audits are case studies, not a representative benchmark survey. The four rows below are Biomni Eval1 task families [12]; two are vendor LAB-Bench MCQs.

![](images/72caa908128a42fc7712b69f23171fc4ada5e56512222d4879ef941a789651a6.jpg)  
dossier shown / characters

![](images/765776f6246c78e06b4b6f04bba703c9e21a870b9b1d71f0e9e27e449c4bb1d7.jpg)  
Figure 3: Deletion does not isolate dossier content. The own dossier, a matched placebo dossier, and a numerically scrambled dossier all reduce accuracy relative to no dossier. Because the substituted blocks have little independently measured signal, these controls do not identify a content-use effect.

![](images/3f74bd6eb7260be8cfa1dd845f3ef65fe77621a9337a5724de2bccd44b7256cb.jpg)  
Figure 4: Stated confidence is weakly related to accuracy across conditions. Accuracy spans 63 points while mean confidence spans 12. The false-identity condition is particularly over-confident.

<table><tr><td>task family (instances)</td><td>prompt supplies</td><td>named</td><td>blinded</td></tr><tr><td>lab_bench_seqqa (21)</td><td>the sequences themselves</td><td>0.952</td><td>1.000</td></tr><tr><td>lab_bench_dbqa (35)</td><td>a database lookup key</td><td>0.629</td><td>0.171</td></tr><tr><td>screen_gene_retrieval (50)</td><td>written research context</td><td>0.660</td><td>0.120</td></tr><tr><td>gwas_causal_gene (147)</td><td>name and candidate list</td><td>0.742</td><td>0.082</td></tr></table>

A name-and-candidate-list prompt loses its entire above-chance margin (+66.0 points, $p = 3 . 6 \times$ $1 0 ^ { - 2 6 }$ ; chance 0.093), whereas a sequence task does not move detectably.

The contrast between these rows is useful precisely because their prompt roles differ. In lab\_bench\_seqqa, the sequence remains present after entity blinding and can still determine the answer. In lab\_bench\_dbqa, hiding the lookup key makes the requested record much harder to retrieve. The screen\_gene\_retrieval and gwas\_causal\_gene rows retain the question and candidate set but lose the entity that connects them to a known result. These examples do not identify a universal benchmark defect. Instead, they show why an identity-blinded score is best reported beside the standard score: a large gap is interpretable as a dependency exposed by the intervention, while a small gap is meaningful only when a positive control establishes that the audit could have detected a change.

The sequence result alone is not an equivalence claim: it has one discordant pair. The stronger evidence for a small gap is the 0% LitQA2 condition below, with 96 questions, no discordant pairs, and $| \mathrm { c o s t } | \le 0 . 0 3 1$ . Byte-identical prompts that name no entity give 0.818 versus 0.818, establishing a run-to-run floor. The poor blinded candidate-list performance is consistent with position-only policies, rather than a partial recovery of the named task.

LitQA2 evidence gradient. LAB-Bench LitQA2 [11] provides a key-passage field. We redact the passage in four steps, drawing positions on a token sequence shared across conditions. Entity mentions are never redacted, so the evidence manipulation preserves identity and the blinding manipulation preserves evidence.
<table><tr><td>words redacted</td><td>named</td><td>blinded</td><td>cost</td><td></td></tr><tr><td>0%</td><td>0.990</td><td>0.990</td><td>+0.000</td><td> $\mathrm { n o d i s c o r d a n t p a i r s , | c o s t | \leq 0 . 0 3 1 }$ </td></tr><tr><td>33%</td><td>0.958</td><td>0.948</td><td>+0.010</td><td> $[ - 0 . 0 3 5 , + 0 . { \bar { 0 } } 5 6 ]$ </td></tr><tr><td>66%</td><td>0.865</td><td>0.771</td><td>+0.094</td><td> $\tilde { [ + 0 . 0 2 9 , + 0 . 1 5 9 \tilde { ] } } , p = 0 . 0 1 2$ </td></tr><tr><td>100%</td><td>0.531</td><td>0.396</td><td>+0.135</td><td> $[ + 0 . 0 2 9 , + 0 . 2 4 2 ] , p = 0 . 0 2 4$ </td></tr></table>

Within this fixed masking scheme, the identity gap grows as evidence is removed (per-instance slope +0.147 per unit redaction, $p = 0 . 0 0 4 )$ . The 66% level matters because neither condition is near ceiling (0.865 versus 0.771): the vanishing gap at 0% is consistent with sufficient evidence, not merely saturation. The 66-point collapse reproduces on the second model at +62.6 points. Up to ten instances share an agent call, so we report call-clustered resampling beside paired McNemar inference [18].

## 4.5 Robustness and interpretation

For the named-versus-no-evidence contrast, the paired i.i.d. standard error is 0.0165 and the paired cluster-robust standard error is 0.0133; an unpaired clustered analysis gives 0.0309 because it discards within-instance pairing. We retain paired inference as primary, but do not make a global multipletesting claim over roughly 40 tests.

The narrowest claim supported by the primary analysis is an interaction within one fixed LitQA2 masking scheme: as more words in the supplied passage are redacted, the named–blinded gap increases. We need to note however, that it does not estimate the prevalence of record recall among agents or among life-science benchmarks, nor does it prove that every named response used identity. Those stronger causal questions would need new benchmark construction, like temporally held-out outcomes, independently verified aliases, or a sealed inference environment. We include the clinical and external case studies to make the design limitations concrete, rather than to substitute them for such experiments.

Four clinical conditions were repeated over all 286 instances on the smaller model with byteidentical prompts. Dispatch-mode checks on twelve blinded batches gave the same functional-versusblinded gap (+0.071) in one-call and grouped execution. These are useful sensitivity checks, but not independent replication across model families. Similarly, functional descriptions are not a clean blinding control, since they omit identifiers by instruction but can still reveal a gene through a distinctive biological description.

## 5 Discussion and conclusion

The audit measures what a score depends on under a specified intervention, not whether an agent can reason in general. In our clinical-outcome benchmark, the blinded dossier does not recover the named score, and the recall probe is consistent with access to recorded outcomes. In external case studies, the identity gap varies with how much decisive evidence the prompt supplies. These results motivate a discriminant-validity check for benchmarks intended to operationalise reasoning over evidence [15], as they do not estimate how common identity dependence is across the field. A bare evidence-deletion result cannot establish content use.

Recommendations. Benchmark builders should consider reporting an identity-blinded comparison, retrieval telemetry where tools are available, and confidence by condition. A positive control can make a null more interpretable, a placebo should replace signal rather than noise, and public outcome tables merit a direct recall probe.

## 5.1 Limitations

The identity result cannot separate recognition from the possibility that opaque tokens hurt the agent’s reasoning, so +25.9 would be an upper bound. Whether prediction used recall on a given instance has a null prespecified test (Fisher $p = 0 . 3 7 )$ and an exploratory rebuild that we do not treat as a claim. The deletion effect is bounded, not explained. Our claims cover audited entity-level tasks with snapshot-level, censored, undated labels. On execution: two models from one family, one harness, neither certifiable, and decoding uncontrolled.

## References

[1] Minikel, E.V., Painter, J.L., Dong, C.C., Nelson, M.R. Refining the impact of genetic evidence on clinical success. Nature 629(8012):624–629, 2024. doi:10.1038/s41586-024-07316-0.

[2] Nelson, M.R. et al. The support of human genetic evidence for approved drug indications. Nature Genetics 47:856–860, 2015. doi:10.1038/ng.3314.

[3] Jacko, M. et al. TxBench-Oligonucleotide Discovery: Benchmarking AI Agents on Experimentally Grounded Decisions in the Discovery of ASO/siRNA Therapeutics. LatchBio, 2026.

[4] Le, H. et al. TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology. arXiv:2606.19245, 2026.

[5] Ramasamy, R. et al. TxBench–Antibody Discovery: Benchmarking AI Agents on Experimentally Grounded Decisions in Therapeutic Antibody Discovery. LatchBio, 2026.

[6] Workman, K. et al. scBench: Evaluating AI Agents on Single-Cell RNA-seq Analysis. arXiv:2602.09063, 2026.

[7] Muralidharan, H. et al. EpiBench: Verifiable Evaluation of AI Agents on Epigenomics Analysis. arXiv:2606.13602, 2026.

[8] Workman, K. et al. SpatialBench: Can Agents Analyze Real-World Spatial Biology Data? arXiv:2512.21907, 2025.

[9] Bhowmick, A. et al. VariantBench: An Agentic Benchmark for Genetic Variant Discovery and Interpretation. LatchBio, 2026.

[10] Mitchener, L. et al. BixBench: A Comprehensive Benchmark for LLM-Based Agents in Computational Biology. arXiv:2503.00096, 2025.

[11] Laurent, J.M. et al. LAB-Bench: Measuring Capabilities of Language Models for Biology Research. arXiv:2407.10362, 2024.

[12] Huang, K. et al. Biomni: A General-Purpose Biomedical AI Agent. bioRxiv 2025.05.30.656746, 2025.doi:10.1101/2025.05.30.656746.

[13] Chen, Z. et al. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery. In Proc. ICLR, 2025.

[14] Roohani, Y. et al. BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments. In Proc. ICLR, 2025.

[15] Jacobs, A.Z., Wallach, H. Measurement and Fairness. In Proc. ACM FAccT, 375–385, 2021. doi:10.1145/3442188.3445901.

[16] Card, D. et al. With Little Power Comes Great Responsibility. In Proc. EMNLP, 9263–9274, 2020.

[17] Gururangan, S. et al. Annotation Artifacts in Natural Language Inference Data. In Proc. NAACL-HLT, 107–112, 2018.

[18] Cameron, A.C., Gelbach, J.B., Miller, D.L. Robust inference with multiway clustering. J. Business & Economic Statistics 29(2):238–249, 2011.

[19] Siegel, Z.S. et al. CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark. arXiv:2409.11363, 2024.

[20] Poliak, A. et al. Hypothesis Only Baselines in Natural Language Inference. In Proc. \*SEM, 180–191, 2018.

[21] Kaushik, D., Lipton, Z.C. How Much Reading Does Reading Comprehension Require? A Critical Investigation of Popular Benchmarks. In Proc. EMNLP, 5010–5015, 2018.

[22] Feng, S., Wallace, E., Boyd-Graber, J. Misleading Failures of Partial-Input Baselines. In Proc. ACL, 5533–5538, 2019.

[23] Longpre, S. et al. Entity-Based Knowledge Conflicts in Question Answering. In Proc. EMNLP, 7052–7063, 2021.

[24] Shi, F. et al. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Proc. ICML, 31210–31227, 2023.

[25] Liu, N.F. et al. Lost in the Middle: How Language Models Use Long Contexts. TACL 12:157–173, 2024.

[26] Sainz, O. et al. NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark. In Findings ofEMNLP, 10776–10787, 2023.

[27] Kadavath, S. et al. Language Models (Mostly) Know What They Know. arXiv:2207.05221, 2022.

[28] Xiong, M. et al. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In Proc. ICLR, 2024.

[29] Bowman, S.R., Dahl, G.E. What Will It Take to Fix Benchmarking in Natural Language Understanding? In Proc. NAACL-HLT, 4843–4855, 2021.

[30] Kapoor, S. et al. AI Agents That Matter. arXiv:2407.01502, 2024.