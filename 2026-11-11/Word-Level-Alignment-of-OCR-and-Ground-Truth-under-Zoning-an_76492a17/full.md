# Word-Level Alignment of OCR and Ground Truth under Zoning and Reading-Order Errors

Jorge Ram´on Fonseca Cacho<sup>1\*</sup>

<sup>1\*</sup>Department of Computer Science, University of Nevada, Las Vegas, Las Vegas, NV, USA. ORCID: 0000-0001-7917-5169.

Corresponding author(s). E-mail(s): jorge.fonsecacacho@unlv.edu;

## Abstract

Purpose OCR post-processing needs OCR text aligned word by word with its ground truth. Data sets rarely provide that mapping, and a monotone aligner can lose its place when zoning errors permute the reading order. We revisit this problem, the Zoning Problem of our 2019 paper, with a protocol that tests correctness under such permutations.

Methods We apply OCR-side reading-order permutations with construction-time word-level gold and score correct-match F1, cascade length and recovery. We compare the 2019 location-aware aligner, a re-anchoring extension of it, a RETAS-style baseline (RETAS-lite), a multi-region aligner that combines unique-word and unique n-gram anchors with chunk-to-window assignment, genalog’s RETAS and the original Java RETAS. Experiments use synthetic character noise with block moves, paragraph swaps and column interleaving, historical newspaper reading-order permutations, geometry-derived gold for Tesseract OCR, and a small word-box set.

Results Synthetic tests show where local tracking fails after large displacements and where uniqueanchor methods collapse when unique anchors are scarce. On our synthetic boilerplate page the 2019 method had the highest F1 among those tested. Historical newspaper and geometry-derived evaluations carry the limitations of approximate gold and the article-order convention. The first multi-region version has a quadratic scaling cost.

Conclusion Local tracking remains fragile under large reading-order changes; the tested anchorbased methods handle many of those cases but fail when unique anchors are scarce. Approximate gold, the article-order convention, missing residual annotation, and the lack of modern multi-column word-level ground truth remain limitations.

Keywords: OCR, ground-truth alignment, zoning errors, reading order, RETAS, post-OCR correction

## 1 Introduction

Testing an OCR post-processing method [1] on aligned word pairs requires knowing which ground-truth word, if any, corresponds to each OCR word. Data sets rarely come with this mapping. They usually give a page image, the OCR text, and a ground truth text, and the mapping has to be recovered by aligning the two texts. Alignment errors then become label errors: a corrector trained on misaligned pairs learns to turn one word into an unrelated one, and an evaluation run on misaligned pairs reports the wrong error rate [1–3].

For short or clean texts, established sequencealignment methods are available. Dynamic programming finds an optimal alignment under edit distance [4, 5], and anchor-based schemes such as RETAS make it fast enough for whole books [6]. The hard case is zoning. When an OCR engine segments a page into the wrong regions, or reads the regions in the wrong order, the OCR text is a permutation of the ground truth at the block level, on top of the usual character errors [7]. A monotone alignment cannot represent that permutation, and a method that follows its position in the ground truth can lose its place and fail for many words in a row.

In 2019 we described this as the Zoning Problem [8]. Our location-aware alignment algorithm, tested on the TREC-5 Confusion Track [9], reached an average alignment accuracy of 98.547% on documents without zoning problems and 81.07% on documents with them. We proposed ways to close that gap but left them as future work. This paper picks up from there. Seven years on, the question is still practical: digitized newspapers and other multi-column collections are exactly where reading-order errors are common, and they are also where many post-OCR training sets come from.

We started by re-testing the problem, and the first lesson was about measurement. An early version of our test harness moved an out-of-order block in both the OCR text and the ground truth, so there was no reading-order error to recover from, and it counted every word that received some match as a success, whether or not the match was correct. Under that harness every method looked perfect. We rebuilt the harness so that the reading order changes on the OCR side only, every OCR word has a known gold position, and we count correct matches, the length of each error cascade, and how quickly a method recovers. The old “matched” rate is still reported where it helps comparison with 2019, but never as a correctness score.

With the synthetic measurement corrected, we compare six aligners (counting the two frozen versions of the multi-region aligner as one) on synthetic zoning errors, on real newspaper reading order, on real Tesseract OCR [10] of historical newspapers from the KB IMPACT collection [11, 12], on a small set of pages with human word boxes, and on TREC-5. The contributions are:

• A measurement protocol for alignment under zoning errors: OCR-side reading-order permutation with word-level gold, precision, recall and F1 of correct matches, cascade length and recovery rate, with bootstrap confidence intervals and paired tests (Sect. 3).

• A gold alignment for real OCR derived from page geometry, which matches OCR word boxes to ground-truth word boxes by overlap and edit distance, with an explicit high-confidence subset and an error estimate (Sect. 5.1).

• A multi-region aligner that combines uniqueword and unique n-gram anchors with greedy assignment of OCR chunks to ground-truth windows. It is simple and its parameters were frozen before the real-OCR tests (Sect. 4.4). The comparison includes our RETAS-style baseline, genalog’s RETAS [13, 14] and the original Java RETAS [15] (Sect. 6.3).

• An account of where it fails: the first version has quadratic scaling and becomes slower than the baseline somewhere between 10,000 and 25,000 tokens, and like the other anchor-based methods tested it collapses on our synthetic boilerplate page with few distinct word types, where our 2019 location-aware method has the highest F1 among the tested methods on that fixture (Sects. 6.5 and 6.6). A second frozen version reduces the length problem.

We also report what we could not reproduce. TREC-5 does not label which documents have zoning problems, so the 2019 split into 98.5% and 81% cannot be rebuilt; we report TREC-5 results as a reproduction attempt only (Sect. 6.7).

Our comparison covers a limited set of implementations and settings. We do not infer a fieldwide ranking from it.

## 2 Related work

## Sequence alignment.

The classic tools for aligning two strings are dynamic programming over edit operations, introduced for protein sequences by Needleman and Wunsch [4] and for string correction by Wagner and Fischer [5]. Both run in time proportional to the product of the two lengths, which can be prohibitively slow for whole books, and both produce a monotone alignment: if word i is aligned before word j in one text, it is aligned before it in the other. A reading-order error breaks that assumption, so a monotone alignment cannot preserve all correct correspondences between a displaced block and the text whose order it crosses.

## Anchor-based alignment.

Heckel [16] observed that lines occurring exactly once in each of two files can serve as candidate anchors, and that the regions between anchors can then be compared separately. The same idea, at the word level, underlies the main OCR alignment methods for long texts. Feng and Manmatha aligned OCR output with book text through a hierarchical scheme based on hidden Markov models [17]. Yalniz and Manmatha’s Recursive Text Alignment Scheme (RETAS) [6] aligns words that are unique in both texts, recurses into the segments between matched unique words until the segments are small, and finishes with editdistance alignment. The original Java implementation is available [15], and Microsoft’s genalog package [13, 14] includes a Python reimplementation that uses Needleman–Wunsch on the final segments. RETAS was designed for books, where the order of the text is reliable, and it treats the text between two consecutive matched anchors as one segment.

## Location-aware alignment.

Our 2019 algorithm [8] takes a diferent route. It walks through the OCR text one word at a time, keeps a pointer into the ground truth, and looks for each OCR word in a small window around that pointer, first exactly and then with increasing edit distance. It does not need unique words, and it is fast, but a block that moves farther than the window sends the pointer astray. The 2019 paper reported this as the Zoning Problem and measured its cost on TREC-5.

## Zoning and reading order.

Kanai et al. [7] measured zoning quality by the number of edit operations, including block moves, needed to turn OCR output into the correct text. Clausner, Pletschacher and Antonacopoulos [18] proposed a character accuracy measure that does not depend on reading order. Both evaluate OCR output; neither produces a word-level alignment that can be used as training data. Ground truth formats such as PAGE [19] store the reading order explicitly, which is what makes our reading-order and geometry-gold tests possible.

## Data sets.

The TREC-5 Confusion Track [9, 20] provides 1994 Federal Register documents in three versions: the original typeset text and two OCR versions with roughly 5% and 20% character error. The IMPACT project produced ground truth for historical documents in PAGE format [11]; the Dutch national library (KB) released its IMPACT newspaper ground truth into the public domain [12]. Post-OCR competitions at ICDAR [2, 3] and recent surveys [1] show how much of the field depends on aligned OCR and ground truth pairs.

## Non-monotone reference collation.

ACDC [21] aligns noisy HTR with digital editions or other witnesses using a character-level HMM whose reference read head can jump backwards or forwards by an arbitrary distance. Charactersubsequence pruning limits the search; the model is implemented as seriatim in passim v2. ACDC selects aligned lines for recognizer retraining. Nonmonotone reference alignment and alignmentderived training supervision are therefore established prior work, not general novelty claims of this paper. Our study instead examines the word-link protocol of Sect. 3 and frozen implementations; ACDC has not been evaluated in our comparison.

## Text reuse and multi-token links.

Miller, Kuflik and Lavee [22] combine Smith– Waterman alignment with edit/fastText similarity, word-boundary split/merge handling, multitoken traceback and post-alignment transposition matching. They evaluate aligned token-position tuples, including many-to-one links, against expert Hebrew/Aramaic alignments. Thus neither position-link evaluation nor split/merge-aware alignment is new in general. Their text-reuse task also permits semantic, acronym and morphological correspondences, whereas our present single-index word mapping targets literal OCR occurrences; the scores are not interchangeable.

## Dutch parallel-corpus construction.

Debaene et al. [23] compare RETAS with sentence-embedding retrieval within each early modern Dutch play and with a newline-aware extension of that retrieval that generates concatenations of consecutive segments. Their triadic construction chooses the lowest-CER candidate across the three approaches for each ground-truth sentence. This is directly relevant to OCR/reference corpus construction in Dutch, but sentence-level CER selection is a diferent criterion from independently verified word-occurrence links. We have not tested their retrieval methods here.

## Granularity-adaptive evaluator matching.

MinerU2.5-Pro [24] introduces Multi-Granularity Adaptive Matching (MGAM) for documentparsing evaluation. Their Section 5.2 keeps ground-truth segmentation fixed and compares Hungarian matching at the original prediction granularity, after line-break splitting, and after consecutive split/merge partition enumeration, choosing the best aggregate candidate. Text elements use edit similarity, and predicted tables can be converted to text for matching. This is prior art for adaptive split/merge assignment, but its evaluator element matches are not independent word-occurrence gold. We do not infer comparative alignment accuracy from its parsing scores.

## Reference-grounded line annotation.

RefLAM [25] combines page-signature anchoring with greedy fuzzy line-to-reference-window matching, separate main/margin handling and human review of every included line. Its confidence-100 rule establishes normalized-string identity, not a unique occurrence or a pixel-perfect box (their Remark 1). It is a close pointer/window referencegrounding pipeline, but its line-level review task difers from our single-index word-link evaluation. We do not treat its annotation throughput as an aligner speed comparison.

## 3 Measuring alignment under zoning errors

## 3.1 Task

Let $G = g _ { 1 } \ldots g _ { m }$ be the ground truth words of a page and $O \ = \ o _ { 1 } \dots o _ { n }$ the OCR words. An aligner returns a mapping $a : \{ 1 . . n \} \to \{ 1 . . m \} \cup$ $\{ \emptyset \}$ that assigns each OCR word a ground-truth position or leaves it unmatched. The evaluation interface records at most one ground-truth position per OCR word; this does not itself guarantee an injective mapping across OCR words. In the gold alignment, merged and split words are left unmatched or represented by one of their parts. A gold alignment $a ^ { * }$ gives the correct position, or ∅ for OCR words with no source.

## 3.2 Metrics

We count a true positive when $a ( i ) = a ^ { * } ( i ) \neq \emptyset$ When the gold position is nonempty, a diferent nonempty prediction counts as both a false positive and a false negative, and an unmatched prediction counts as a false negative. When the gold position is $\mathcal { O } ,$ any match counts as a false positive. Precision, recall and F1 follow as usual. This difers from the success rate used in 2019 and in our first test harness, which counted the share of OCR words that received any match. We keep that share under the name matched where it helps comparison with earlier numbers, but it is not a measure of correctness: an aligner that matches every word to the wrong position scores 100% on $\operatorname { i t } .$

Zoning errors produce cascades. Let a break be an OCR position $i > 1$ for which both $a ^ { * } ( i )$ and $a ^ { * } ( i - 1 )$ are nonempty and $a ^ { * } ( i ) \neq a ^ { * } ( i - 1 ) + 1$ The first position and transitions involving an OCR word with gold ∅ are excluded. The cascade length at a break is the number of consecutive incorrectly aligned OCR words starting at the break, and $\bar { C }$ is its mean over breaks. The recovery rate $R _ { N }$ is the share of breaks followed by at least one correct alignment among the next N OCR words; we report $N = 1 0$

Confidence intervals are 95% percentile bootstrap intervals over seeds (synthetic data) or pages (real data), with 2,000 resamples [26]. To compare two aligners on the same items we use a paired bootstrap with $5 , 0 0 0$ resamples of the per-item difference and report its mean, interval, and $p ,$ twice the share of bootstrap means on the other side of zero (including zero), capped at one. A reported $p < 0 . 0 0 1$ means that none of the 5,000 bootstrap means fell on the opposite side of zero. These intervals describe the evaluated seeds or pages; they are not claims about a larger population.

## 3.3 A generator with known gold

The synthetic test starts from a fixed ground truth of 517 words of mostly Federal-Register-style prose with 327 distinct word types (case-folded). The OCR side is a permuted copy, and the gold alignment is the permutation itself, so every OCR word keeps the index of the ground-truth word it came from. We use three zoning patterns:

• Block move: a contiguous block (4%, 8% or 15% of the words, and at least eight words long) is cut from the first half of the page and inserted near the end, well beyond a local search window.

• Paragraph swap: the page is split into paragraphs of max(20, $\lfloor n / ( 2 k + 2 ) \rfloor )$ words, where n is the page length and k the number of swap steps (86 words at medium severity), and the paragraphs are reordered by a permutation that is guaranteed not to be the identity (1, 2 or 3 swap steps).

• Column interleave: the ground truth is read as a left column followed by a right column, and the OCR side alternates chunks of 8, 12 or 16 words from the two columns, as when lines of two columns are read across.

The three numbers in each pattern are the mild, medium and severe settings. For block moves a higher setting displaces more text; for paragraph swaps it adds swap steps over shorter paragraphs; for column interleave it only makes the chunks longer, which means fewer breaks, so “severe” interleave is not necessarily harder than “medium”. The chunk lengths 8, 12 and 16 are also the chunk sizes of the multi-region aligner (Sect. 4.4), so its column-interleave results may be helped by this match; the real-OCR tests do not have this property. After permuting, each word longer than two characters receives, with probability p, one random character substitution, insertion or deletion; p is 2%, 6% or 10% in the main sweep and up to 25% in the stress tests. The noisy word keeps its gold position. Unless stated otherwise we report medium severity with $p = 6 \%$ over seeds 0–9.

The base text was written to contain no repeated paragraphs. An earlier version built a longer page by repeating the base text six times; every column half then had an exact copy elsewhere on the page, and on paragraph swap every aligner locked onto the sequential copy and produced the same mapping. We report only results from the non-repeating generator.

## 4 Aligners

All aligners take the two word sequences, lowercased and split into runs of word characters, and return at most one ground-truth index per OCR token; global injectivity is not assumed for every baseline. Edit distance is character-level Levenshtein distance between words [5].

## 4.1 Location-aware alignment (2019)

We reimplemented the 2019 algorithm [8] from its published description. It keeps a pointer p to the last matched ground-truth position. For each OCR word in order it tries, and stops at the first success: an exact match at $p + 1 ;$ an exact match in the forward window $[ p + 1 , p + w ] ;$ ; an exact match in the backward window $[ p - w , p - 1 ] ;$ and then, for edit distance $d = 1 , 2 , \dots , d _ { \mathrm { m a x } } .$ , the first word within distance d in the forward window and then in the backward window. A match moves the pointer to the matched position. A failure leaves the word unmatched and advances the pointer by one, on the assumption that the OCR word was a garbled version of the next ground-truth word. The search uses $w = 2 0$ and $d _ { \operatorname* { m a x } } = 3$ and tries the forward window before the backward window, matching the 2019 settings. Where the published description is silent on whether confidence weights afect match selection, we record the weights, for example for dictionary words, but they do not change which match is chosen.

The weakness is visible in the description. When a block of the OCR text comes from a place farther than w words away, the correct positions can lie outside the search windows. Each failure advances the pointer by one word, which can cause drift and further failures in the text that follows.

## 4.2 Re-anchoring

The re-anchoring extension was our first attempt at the open problem. It runs the 2019 algorithm, then scans the result for streaks of at least four consecutive unmatched OCR words. For each streak it searches for a short sequence, four words and then three, that starts within the first 80 positions of the streak and agrees with some ground-truth sequence from 80 positions before to 200 positions after the last good match, in all but at most one word, where a word agrees if it is equal or within edit distance 1. If one is found, the next 40 OCR words are matched greedily forward from the anchor within the next 60 ground-truth positions, overwriting earlier matches. The search is still local to the last good match, so it repairs drift after short displacements but cannot find a block that came from far away.

## 4.3 RETAS-lite

As a fast stand-in for RETAS we use a singlelevel version that we call RETAS-lite. It takes as anchors all words that occur exactly once in the OCR text and exactly once in the ground truth, sorted by OCR position, and matches each anchor to its ground-truth twin. Between two consecutive anchors it matches each OCR word to the closest ground-truth word within edit distance 2, searching forward from the last match for at most 40 positions and not past the next anchor. Unlike RETAS it does not recurse, and it does not discard anchors that are out of order; the second diference is what lets it follow a moved block, because the unique words inside the block still point to the right place. RETAS-lite is our own reimplementation and should not be read as a measurement of RETAS itself; for that we run genalog and the original Java tool (Sect. 4.6).

## 4.4 Multi-region alignment

The multi-region aligner keeps the anchor idea of RETAS-lite and adds a second stage that is built for out-of-order material. It has four steps (Algorithm 1).

1. Anchors. Words that are unique on both sides are anchors, as in RETAS-lite. Then, for k = 4, 3, 2 in that order, every k-gram of words that occurs exactly once in the OCR text and exactly once in the ground truth, and does not overlap positions already anchored on either side, adds k anchors. The k-grams find anchors in text whose individual words are common.

2. Local fill. The anchors are sorted by OCR position. For each consecutive pair whose groundtruth positions are also increasing, the unmatched

OCR words between them are matched to the closest unused ground-truth word between the two anchors, within edit distance $e = 2$ and at most $L = 6 0$ positions ahead of the last match.

3. Chunk-to-window assignment. This step handles material that the anchors left unmatched, which after a zoning error is mostly the displaced text. For chunk sizes s = 16, 12, 8 in turn, the OCR text is cut into consecutive chunks of s words, and chunks with at least max(2, ⌊s/3⌋) unmatched words are kept. The ground truth is cut into windows of s words with a stride of $s / 2$ and windows that contain any matched position are dropped. Each chunk–window pair is scored by the size of the multiset intersection of their words, and pairs with a score below 2 are dropped. The pairs are accepted greedily from the highest score down, each chunk at most once and no two accepted windows overlapping, and each accepted pair is filled locally as in step 2. Because the windows can be anywhere in the ground truth, a chunk can be placed far from its neighbours, which is what a reading-order error requires.

4. Final sweep. Each OCR word still unmatched is matched to the first unused exact twin in the ground truth, searching forward from the position after the previous matched word and wrapping around, and otherwise to any unused word within edit distance 1.

The parameters are the anchor sizes (4, 3, 2), chunk sizes (16, 12, 8), e = 2, L = 60 and the minimum chunk score of 2. They were set during the synthetic and reading-order experiments of Sects. 6.1 and 6.2 and then frozen, under the tag multiregion-v1-frozen, before any of the stress, real-OCR, word-box or long-document tests. The v1 code freeze is e7813fc; the v2 code freeze is 83bb593. The tag commits 9fa4083 and ca17f76 record those freezes rather than changing the algorithms. No value was changed afterwards; every later change went into a separate version. The results of Sects. 6.1 and 6.2 are therefore in-sample for multi-region v1, and only the later tests are held out.

The aligner is not new in its parts. Uniqueelement anchors go back to Heckel [16] and RETAS [6], and window matching by bag-ofwords overlap is a common heuristic. We study this particular ordering of the steps and unconstrained chunk placement, rather than claiming non-monotone reference alignment as a new idea; ACDC already permits arbitrary reference jumps [21]. Step 3 compares every chunk with every window, so its cost grows with the product of the two lengths; we return to this in Sect. 6.6.

Algorithm 1 Multi-region alignment (v1). O:   
OCR words, G: ground truth words.   
1: A ← unique-word anchors ∪ non-overlapping   
unique k-gram anchors, $k = 4 , 3 , 2$   
2: $a ( i )  j$ for every anchor $( i , j ) \in A$   
3: for consecutive anchors $( i _ { 1 } , j _ { 1 } ) , ( i _ { 2 } , j _ { 2 } )$ with   
$j _ { 1 } < j _ { 2 }$ do   
4: $\mathrm { F I L L } ( O [ i _ { 1 } { + } 1 . . i _ { 2 } { - } 1 ] , G [ j _ { 1 } { + } 1 . . j _ { 2 } { - } 1 ] )$ ▷   
edit distance $\leq 2 ,$ look-ahead $\leq 6 0$   
5: end for   
6: for $s \in ( 1 6 , 1 2 , 8 )$ do   
7: $c \gets \mathrm { O C R }$ chunks of s words with $\geq$   
max $( 2 , \lfloor s / 3 \rfloor )$ unmatched words   
8: W ← GT windows of s words, stride $s / 2 ,$   
with no matched position   
9: $P  \{ ( c , w , | c \cap w | ) : c \in { \mathcal { C } } , w \in { \mathcal { W } } , | c \cap$   
w| $\geq 2 \}$ , sorted by score   
10: for $( c , w , \cdot ) \in P$ with c unassigned and w   
disjoint from assigned windows do   
11: assign c → w; Fill(c, w)   
12: end for   
13: end for   
14: match leftovers to the next unused exact twin,   
else to any unused word within distance 1

## 4.5 Multi-region v2

After the long-document and boilerplate results of Sect. 6 we built a second version with three changes, developed only on synthetic seeds 100– 129 and on IMPACT pages outside the 200-page test set, then frozen as multiregion-v2-frozen and evaluated once on the held-out data:

• Indexing. Chunk-to-window scoring uses an inverted index from words to candidate windows and precomputed word counts per window. On a development document of 10,413 tokens, with the long-document fallback left of, this was 3.9 times faster than v1 at the same F1.

• Long-document fallback. Documents of 8,000 OCR tokens or more are handed to RETAS-lite. On development data this beat both no fallback and a windowed variant in accuracy and runtime at 10,000 and 25,000 tokens.

• Scarcity merge. If fewer than 5% of tokens are unique on both sides and fewer than 2% of word 3-grams are, or if fewer than 35% of OCR words are matched after the v1 steps, the unmatched spans are aligned with the 2019 location-aware algorithm and merged in. The idea is to use the 2019 positional prior where anchors are scarce.

We wrote a success criterion into the proposal before the held-out run: on boilerplate pages, v2 should beat v1 by at least 0.15 F1 with paired $p < 0 . 0 1$ without losing more than 0.01 on column interleave. Sect. 6.6 shows that the first part was not met.

## 4.6 External baselines

genalog. We use the RETAS reimplementation in Microsoft’s genalog (MIT licence) [14], align w anchor, which aligns characters and returns, for each OCR word, the ground-truth words it overlaps. When there are several we keep the one with the smallest edit distance to the OCR word.

Java RETAS. We compiled the original Recursive Text Alignment Scheme (version 1.1, GPL-3.0) [15] from the Early Modern OCR Project mirror with JDK 17, ran it at word level, and parsed its aligned output into a word mapping. On one document pair, the tool’s reported accuracy (0.495) and the F1 of our parsed mapping (0.493) agreed to within 0.002, a check of the parsing rather than a general proof of it.

## 5 Data and protocol

## 5.1 Data

## Synthetic zoning.

We use the generator of Sect. 3.3 with seeds 0–9. For the stress tests we vary the noise rate from 10 to 25% under column interleave, split the page into 8, 15 or 30 regions, use short pages of about 96 words with a paragraph swap, and use a boilerplate page of 693 words containing 59 distinct word types (only 10 occur once), modelled on the repetitive notices of the Federal Register, with medium column interleave and 6% noise. For long documents we concatenate distinct paragraphs, with no repetition, to 1,000–50,000 tokens and apply medium column interleave with 6% noise (five seeds up to 10,000 tokens, three above).

## KB IMPACT reading order.

The KB IMPACT newspaper ground truth [12] contains 1,024 newspaper pages from 1618 to 1885 in PAGE format [19], with text regions and PAGE reading-order groups. Ordering between articles inside a top-level UnorderedGroup is not specified by that annotation. The KB released it as public domain (CC0). For a first real test without OCR noise the historical fixture takes as G the groundtruth text in the XML traversal order of region references (a linearization convention, not a complete annotated page order) and the same text in geometric order (top to bottom, using column bins when the page has columns) as O; the gold alignment is the index map within each region. We shufled the sorted page list with seed 0. Of the first 50 pages we checked, 40 have a geometric order that difers from the XML traversal order, and we use those 40.

## KB IMPACT with real OCR.

For the main test we ran Tesseract 5 [10] (nld+eng, page segmentation mode 6) on the KB page images of 200 pages drawn with seed 42, and used at most the first 600 OCR words of each page. Five of those pages also appear in the 40- page reading-order fixture; the frozen parameters were not chosen on either set. There is no wordlevel alignment between Tesseract output and the ground truth, so we derived one from geometry. The PAGE files give boxes for text regions only, so we split each region box into line strips and each line into word boxes in proportion to character counts. Each Tesseract word box is then matched greedily, one to one, to the ground-truth word box that maximizes IoU − 0.75 NED, where NED is the edit distance normalized by the longer word; a match is accepted if this score is positive and either IoU ≥ 0.05 or NED ≤ 0.2, with a fallback to the nearest word in the same region when NED ≤ 0.2. A gold match is high-confidence if the words are equal, or if NED ≤ 0.15 and IoU ≥ 0.05.

This gold is approximate. We automatically classified every matched pair on 20 pages as exact, close (NED ≤ 0.25) or mismatched, and estimated its error as the share of mismatches plus a quarter of the close pairs. The estimate was 0.238, with 62.6% of matched pairs exact and 64.7% of OCR words covered. Summed over the same 20 pages the stored type counts are 17,894 exact matches, 3,845 close (soft-edit) matches, 3,452 text mismatches and 14,882 unmatched OCR tokens (25,191 matched pairs among 40,073 OCR tokens). That quality check uses every OCR token on those 20 pages, not the 600-word cap used in the alignment evaluation. These are micro totals; the 0.238 error estimate and the 62.6% and 64.7% figures above remain page-mean summaries and are not the corresponding micro shares. The estimate counts disagreements between the OCR and ground-truth spellings of box-matched words, so it mixes gold errors with OCR errors, and it does not see wrong matches between similar words; it should be read as a rough indication rather than a bound. For these reasons we report F1 on all gold matches and on the high-confidence subset, with the approximate-gold limitations stated above.

## Word-box gold.

To check the geometry gold where real word boxes exist, we used three CC0 corpora of 16th- and 17th-century prints with human word coordinates in PAGE format (17 frak simple, 17 frak complex and 16 ant simple [27]), 14 pages in total, OCRed with Tesseract (deu+eng) and matched by the same IoU and NED rule against the true word boxes. Here the gold uses human-annotated word boxes, but the pages are blackletter and antiqua prints on which Tesseract is weak, and the same error estimate is 0.447 with 64.9% coverage, which may reflect OCR errors as well as matching errors. We found no open data set of modern multi-column pages with human word boxes under a CC0, CC-BY or public-domain licence.

## TREC-5.

The Confusion Track [9, 20] is the data of the 2019 paper. We matched 479 original documents with their degrade5 versions by file name and sampled 30 documents (seed 0), using at most the first 600 tokens of each for the frozen comparison. The track has no word-level alignment and no zoning labels. We therefore score agreement with genalog’s alignment as a soft reference and the matched rate, and we also build a synthetic-gold variant: 20 original documents (first 500 tokens) with medium column interleave and 10% noise from our generator. The TREC-5 tarball is served from the NIST track page without an agreement form; we used it for research only and do not redistribute it.

## 5.2 Protocol

The multi-region v1 parameters were frozen before the stress tests, and v2 was developed on separate seeds and pages and evaluated once. The 200 IMPACT test pages were held out from parameter development. Their geometry gold was not inspected while v2 was built, apart from the automatic 20-page quality classification described above, which uses a subset of these 200 pages. All aligners were run on an AMD EPYC 9175F CPU using Python 3.12, one document at a time; runtimes are wall-clock seconds per document and include only the alignment call. Synthetic results use seeds 0–9 unless noted; real-data intervals are over pages.

Use of AI assistants in preparing the manuscript and supporting code is documented in the Statements and Declarations.

## 6 Results

## 6.1 Synthetic zoning errors

Table 1 shows the ablation on the synthetic generator. The 2019 location-aware algorithm handles a single moved block reasonably well (F1 0.890), because the moved block and the text right after it are lost, but it fails on paragraph swaps (0.012) and on interleaved columns (0.255). Table 2 shows why: after a paragraph swap its cascade runs for 129 words on average, longer than one 86-word paragraph, and only one break in six is followed by a correct match within ten words. This is the Zoning Problem of 2019 in its cleanest form.

Re-anchoring helps, but not enough to recover from large reorderings. It raises F1 to 0.922, 0.339 and 0.353 and roughly halves the cascades on paragraph swap and column interleave. We checked that the gain comes from re-anchoring itself and not from noise in the harness: on seed 0 of a medium paragraph swap with 2% noise it fired nine times and changed 145 mappings, raising F1 from 0.003 to 0.338. Its search, however, stays near the last good match, and a paragraph that moved by a third of the page is out of reach.

The anchor-based methods are in a diferent range. RETAS-lite scores 0.998, 0.977 and 0.907, with mean cascades of 0.7 to 3.3 words, which confirms that unique words are good anchors even when the text is out of order, as long as the aligner does not insist on keeping the anchors in order. The multi-region aligner reaches 1.000, 1.000 and 0.983, and every break is followed by a correct match within ten words. genalog’s RETAS scores 0.920 on block moves but 0.181 on paragraph swaps and 0.500 on column interleave. It is also two to three orders of magnitude slower on these 517-word pages, 0.55–0.81 s against 0.001–0.002 s for RETAS-lite and multi-region.

Figure 1 is a token-index schematic of that paragraph-swap case. We draw blocks by groundtruth index rather than source text, so the figure does not quote corpus wording. After the swap, the first OCR paragraph should map to paragraph B and the second to paragraph A. The 2019 tracker keeps a local window of $w ~ = ~ 2 0$ near the last accepted match. It is then still looking near the start of A while the OCR stream is reading B, and the stored mean cascade on this condition is 129 tokens (Table 2). Unique-word and n-gram anchors can jump to the matching paragraph anywhere on the page.

Figure 2 plots those same stored rows. On paragraph swap the mean cascade is 129 tokens for the 2019 method, 66.5 after re-anchoring, 207 for genalog, 3.3 for RETAS-lite and 0 for multiregion. On column interleave the same order is 48.8, 23.6, 6.3, 1.3 and 0.1; on a single block move it is 25.3, 18.2, 13.7, 0.7 and 0.0. The figure is a companion to Table 2, not a new experiment.

## 6.2 Real reading order without OCR noise

Table 3 lists the stored Step-3d reading-order scores: multi-region 0.987, RETAS-lite 0.986, genalog 0.668, the 2019 method 0.573 and reanchoring 0.578. Java RETAS was rerun on the same Step-3d page list, giving F1 0.590 [0.557, 0.623] and recovery $\begin{array} { r l r } { R _ { 1 0 } } & { { } = } & { 0 . 8 7 0 . } \end{array}$ A diferent-draw Java-harness run is excluded from cross-method comparisons.

## 6.3 Real OCR with geometry gold

On the high-confidence gold, stored F1 is multiregion v1 0.650, RETAS-lite 0.504, genalog 0.371, Java RETAS 0.273 and the 2019 method 0.053. The stored all-gold diferences are +0.078 [+0.068, +0.087] for multi-region minus RETASlite, +0.256 [+0.235, +0.277] for multi-region minus Java RETAS, and +0.178 [+0.155, +0.201] for multi-region minus genalog. The first two were recomputed from stored per-page scores. The geometry-derived gold remains approximate.

Table 1 Synthetic zoning ablation: word-alignment F1 against gold, mean over 10 seeds with 95% bootstrap CI (medium severity, 6% character noise, 517-token base text). Reading order is permuted on the OCR side only.
<table><tr><td>Aligner</td><td>Block move</td><td>Paragraph swap</td><td></td><td>Column interleave</td></tr><tr><td>Location-aware (2019)</td><td>0.890 [0.865, 0.913]</td><td></td><td>0.012 [0.003, 0.029]</td><td></td><td>0.255 [0.242, 0.265]</td></tr><tr><td>+ re-anchoring</td><td>0.922 [0.907, 0.935]</td><td></td><td>0.339 [0.338, 0.340]</td><td></td><td>0.353 [0.348, 0.358]</td></tr><tr><td>RETAS-lite</td><td>0.998 [0.996, 0.999]</td><td></td><td>0.977 [0.975, 0.977]</td><td></td><td>0.907 [0.904, 0.909]</td></tr><tr><td>genalog (RETAS)</td><td>0.920 [0.920, 0.921]</td><td></td><td>0.181 [0.164, 0.215]</td><td></td><td>0.500 [0.499, 0.501]</td></tr><tr><td>Multi-region v1</td><td>1.000 [0.999, 1.000]</td><td></td><td>1.000 [1.000, 1.000]</td><td></td><td>0.983 [0.981, 0.985]</td></tr></table>

Table 2 Cascade behaviour on the synthetic ablation of Table 1: mean cascade length $\bar { C }$ (consecutive wrong words after a zoning break) and recovery rate R10 $R _ { 1 0 }$ (share of breaks followed by a correct match within 10 words). Means over 10 seeds.
<table><tr><td rowspan="3">Aligner</td><td colspan="2">Block move</td><td colspan="2">Paragraph swap</td><td colspan="2">Column interleave</td></tr><tr><td> $\bar { C }$ </td><td> $R _ { 1 0 }$ </td><td> $\bar { C }$ </td><td> $R _ { 1 0 }$ </td><td> $\bar { C }$ </td><td> $R _ { 1 0 }$ </td></tr><tr><td>Location-aware (2019)</td><td>25.3</td><td>0.33</td><td>129.0</td><td>0.17</td><td>48.8</td><td>0.23</td></tr><tr><td>+ re-anchoring</td><td>18.2</td><td>0.53</td><td>66.5</td><td>0.17</td><td>23.6</td><td>0.33</td></tr><tr><td>RETAS-lite</td><td>0.7</td><td>1.00</td><td>3.3</td><td>0.98</td><td>1.3</td><td>1.00</td></tr><tr><td>genalog (RETAS)</td><td>13.7</td><td>0.67</td><td>207.0</td><td>0.07</td><td>6.3</td><td>0.51</td></tr><tr><td>Multi-region v1</td><td>0.0</td><td>1.00</td><td>0.0</td><td>1.00</td><td>0.1</td><td>1.00</td></tr></table>

Those means hide a skewed page-level picture (Fig. 4). Recomputed from the stored all-gold F1 of the same 200 pages, multi-region has page-mean 0.432 and RETAS-lite 0.354 (diference +0.078), matching the stored paired gap; the 2019 method has page-mean 0.038 and falls below 0.10 on 191 of the 200 pages. multi-region exceeds RETAS-lite on 175 pages and is behind on 25, with no ties. The median page diference is +0.085. Of the 175 pages where multi-region is ahead, the 52 largest positive gaps account for at least half of the summed positive gap, and the 99 largest account for at least 80%: the headline +0.078 is not a uniform lift. multi-region reaches F1 of at least 0.50 on 80 pages, and on 75 of those the 2019 method is below 0.15.

Before constructing geometry gold, the same pages were scored for agreement with genalog: stored values are multi-region 0.185, RETASlite 0.172, Java RETAS 0.116 and the 2019 method 0.021, with multi-region minus RETASlite +0.012 [+0.008, +0.017]. These are agreement records, not correctness measurements.

## 6.4 Pages with human word boxes

Table 5 reports stored 14-page word-box scores: multi-region v1 F1 0.742 (all gold) and 0.948 (high-confidence), and RETAS-lite 0.698 and 0.910; the stored all-gold diference is +0.043 [+0.024, +0.065]. Stored all-gold F1 for the 2019 method, genalog and Java RETAS is 0.483, 0.492 and 0.378. These early modern prints are not a multi-column newspaper evaluation.

## 6.5 Stress tests

Table 6 and Fig. 5 vary one dificulty at a time with the v1 parameters frozen. Under character noise from 10% to 25%, multi-region stays between 0.957 and 0.973 while RETAS-lite drops to 0.870– 0.893; the paired gap is +0.079 to +0.093. Noise destroys some unique words and creates false ones; we think the n-gram anchors and chunk overlaps of multi-region are less sensitive to a single corrupted word than unique words alone. When the page is split into more regions, the gap grows from +0.017 at 8 regions to +0.046 at 30. On short pages of about 96 words both methods are at the ceiling and the diference is not significant $( p = 0 . 2 3 )$ In the noise and region conditions Java RETAS stays between 0.38 and 0.48 and the 2019 method between 0.10 and 0.27.

![](images/245675c45b91a358ddf8a5009c614b6c4e6041263bd40db01e0454989b304add.jpg)

Fig. 1 Token-index schematic of an OCR-side paragraph swap. Blocks are labelled by ground-truth index ranges, not by source text. Construction gold maps each OCR token to its original index. The 2019 tracker keeps a local window near the last accepted match (w = 20), so after the swap it continues searching near paragraph A while the OCR stream reads paragraph B; the stored mean cascade on the medium paragraph-swap condition is 129 tokens (Table 2). Unique-word and n-gram anchors are not bound to that window.  
![](images/3a33324e90a42f3b7b53fd44ce372c771e46a52e65e5488a636bbe5014e304c0.jpg)  
Fig. 2 Mean cascade length on the synthetic ablation at medium severity and 6% character noise, the same stored rows that fill Table 2. No new aligner runs.

The boilerplate page is diferent. With 59 distinct word types in 693 tokens, every anchorbased method collapses: multi-region 0.125, Java RETAS 0.069 and RETAS-lite 0.030. The 2019 location-aware method scores 0.456, more than three times the best anchor-based method. On this page, knowing roughly where one is in the ground truth is worth more than any anchor. It is a single synthetic text, so we do not know how far the result generalises, but we do not think it is an artefact of the generator, and it is the reason v2 tries to fall back on the 2019 method when anchors are scarce.

Table 3 KB IMPACT newspaper pages whose geometric order difers from the stored XML-traversal linearization of PAGE region references (40 of 50 scanned pages). The ground-truth text is used on both sides, so there is no OCR noise; only the reading order difers. F1 with 95% bootstrap CI over pages, recovery $R _ { 1 0 }$ and runtime per page.
<table><tr><td>Aligner</td><td>F1</td><td> $R _ { 1 0 }$ </td><td>Runtime (s)</td></tr><tr><td>Location-aware (2019)</td><td>0.573 [0.435, 0.707]</td><td>0.550</td><td>0.008</td></tr><tr><td>+ re-anchoring</td><td>0.578 [0.441, 0.712]</td><td>0.552</td><td>0.031</td></tr><tr><td>RETAS-lite</td><td>0.986 [0.972, 0.995]</td><td>0.987</td><td>0.005</td></tr><tr><td>Multi-region v1</td><td>0.987 [0.969, 0.999]</td><td>0.995</td><td>0.009</td></tr><tr><td>genalog (RETAS)</td><td>0.668 [0.549, 0.792]</td><td>0.643</td><td>2.544</td></tr><tr><td>Java RETAS</td><td>0.590 [0.557, 0.623]</td><td>0.870</td><td>0.102</td></tr></table>

Table 4 Real Tesseract OCR on 200 held-out KB IMPACT pages (seed 42; at most 600 OCR words per page), scored against geometry-derived gold. “All” scores every OCR word that received a gold match; “high-conf.” keeps only words whose gold match is exact or has $\mathrm { N E D } \leq 0 .$ 15 with IoU ≥ 0.05. The gold is approximate (Sect. 5.1), with the limitations described there. Mean F1 with 95% bootstrap CI over pages.
<table><tr><td>Aligner</td><td>F1 (all gold)</td><td>F1 (high-conf. gold)</td></tr><tr><td>Multi-region v1</td><td>0.432 [0.404, 0.460]</td><td>0.650 [0.623, 0.675]</td></tr><tr><td>Multi-region v2</td><td>0.403 [0.375, 0.432]</td><td>0.607 [0.579, 0.634]</td></tr><tr><td>RETAS-lite</td><td>0.354 [0.326, 0.381]</td><td>0.504 [0.477, 0.530]</td></tr><tr><td>genalog (RETAS)</td><td>0.254 [0.230, 0.277]</td><td>0.371 [0.342, 0.399]</td></tr><tr><td>Java RETAS</td><td>0.176 [0.164, 0.187]</td><td>0.273 [0.260, 0.284]</td></tr><tr><td>+ re-anchoring</td><td>0.042 [0.034, 0.051]</td><td>0.059 [0.050, 0.071]</td></tr><tr><td>Location-aware (2019)</td><td>0.038 [0.031, 0.047]</td><td>0.053 [0.044, 0.064]</td></tr></table>

![](images/cf135769af48d8cf99492ab2eb7c7b7e1c25486a146703bc11c04021c3b07f75.jpg)  
Fig. 3 F1 on the 200 KB IMPACT pages with Tesseract OCR (Table 4), on all and high-confidence geometry gold, with bootstrap intervals.

## 6.6 Long documents and the second version

The multi-region aligner was designed for pages, and Table 7 and Fig. 6 show what happens beyond them. All methods score lower at 50,000 tokens than at 1,000. We think this is because the same zoning pattern over a longer text creates more breaks and leaves a smaller share of words unique. Up to 5,000 tokens multi-region v1 is clearly ahead of RETAS-lite (0.771 against 0.504 at 1,000 tokens, 0.423 against 0.294 at 5,000). At 10,000 they are level, and at 25,000 and 50,000 RETASlite is better (0.341 against 0.256 at 50,000). The runtime of v1 grows from 0.021 s at 1,000 tokens to 2.53 s at 10,000 and 61.5 s at 50,000, roughly quadratic, while RETAS-lite stays under a second. Profiling attributes 4.46 of 5.29 s at about 10,000 tokens to the chunk-to-window scoring of step 3. An earlier long-document test built its documents by repeating the base text; there both methods collapsed (multi-region 0.424 at 1,000 tokens and 0.016 at 50,000, RETAS-lite 0.026 and 0.001) because repetition removes uniqueness, so that test measures repetition rather than length and we do not use it further.

The frozen v2 bypasses v1’s quadratic chunkto-window scoring on long documents. In the separate Step-8 side-by-side re-run on the same machine, v1 took 57.7 s at 50,000 tokens rather than the scaling study’s 61.5 s; this is run-to-run timing noise, not a change to v1. With its fallback, v2 is identical to RETAS-lite from 8,000 tokens on and runs in 0.51 s at 50,000 tokens instead of 57.7 s, with F1 0.341 instead of 0.256 (Table 8). Below the threshold it is less than 0.01 below v1, and its indexed chunk assignment is faster (0.22 s against 0.59 s at 5,000 tokens).

![](images/a4a71246891251c58af012f3cad51d91539db990081e8197c4280a6c626b7e5c.jpg)

![](images/3ec1e7c3f765002b6025d72d52f7e02299732898c2b819a6f0b8d4d7bbdc5f8b.jpg)  
page ΔF1 (multi-region v1 − RETAS-lite)  
Fig. 4 Page-level all-gold F1 on the 200 KB IMPACT pages with Tesseract OCR. Left: multi-region v1 against RETASlite (the diagonal is equality). Right: histogram of the per-page diference, with the mean marked. Recomputed from the stored per-page scores that underlie Table 4.

Table 5 Fourteen CC0 pages with human word boxes (blackletter and antiqua prints), Tesseract OCR, gold from word-box IoU and NED. Small sample; the gold-matching and domain limitations still apply.
<table><tr><td>Aligner</td><td>F1 (all gold)</td><td>F1 (high-conf. gold)</td></tr><tr><td>Multi-region v1</td><td>0.742 [0.687, 0.795]</td><td>0.948 [0.918, 0.975]</td></tr><tr><td>Multi-region v2</td><td>0.735 [0.677, 0.791]</td><td>0.947 [0.917, 0.973]</td></tr><tr><td>RETAS-lite</td><td>0.698 [0.629, 0.765]</td><td>0.910 [0.869, 0.949]</td></tr><tr><td>Location-aware (2019)</td><td>0.483 [0.373, 0.592]</td><td>0.694 [0.547, 0.835]</td></tr><tr><td>genalog (RETAS)</td><td>0.492 [0.341, 0.628]</td><td>0.647 [0.437, 0.833]</td></tr><tr><td>Java RETAS</td><td>0.378 [0.323, 0.436]</td><td>0.663 [0.609, 0.712]</td></tr></table>

![](images/0510f4a56269620a2306026b1ed2b42a9a4491bb248a25973f84902e5446e52b.jpg)

![](images/534e5f5e5a85541d28b1ff46b249486d4c13cf6fcf6b9ac5a0ef9e64133d298b.jpg)

![](images/5a02fa22101ef5d7b11209044c47f5c1e4ddfb70fc93cb4df260d94bdf03181b.jpg)  
Fig. 5 Stress tests of Table 6: F1 against the characternoise rate on column interleave (left) and against the number of text regions (right). Means over 10 seeds.  
Fig. 6 Long documents of distinct text (Table 7): mean F1 with the range over seeds (left) and runtime per document on a log scale (right).

In the IMPACT comparison in Table 8, highconfidence F1 is v2 0.607 versus v1 0.650, paired diference −0.043. Recorded fallback counts are scarcity 48/200 and long-document 0/200 under the 600-token cap. These counts do not establish the cause of a score diference. The word-box allgold diference is stored as −0.007. In the synthetic boilerplate test, v2 gains 0.034 over v1 (0.159 versus 0.125), below the pre-specified 0.15 target and the 2019 method’s 0.456. The scarcity merge applies the 2019 prior only to unmatched spans. We withhold an explanation of its residual errors pending independent annotation.

Table 6 Stress tests with frozen parameters, 10 seeds. Noise rows use column interleave with the given per-word characternoise rate; region rows split the page into k regions; short pages have about 96 words; the boilerplate page has 693 words containing 59 distinct word types. ∆ is the paired bootstrap mean of multi-region v1 minus RETAS-lite with 95% CI, and p twice the share of bootstrap means on the other side of zero, capped at one $( ^ { 6 4 } { < } 0 . 0 0 1 ^ { 9 }$ means none of the resamples).
<table><tr><td>Condition</td><td>MR v1</td><td>RETAS-lite</td><td>Java RETAS</td><td>2019</td><td>Δ</td><td>(MR v1 − lite)</td><td></td></tr><tr><td>Noise 0.10</td><td>0.972</td><td>0.893</td><td>0.476</td><td>0.264</td><td>+0.079</td><td>[+0.076, +0.083]</td><td>&lt;0.001</td></tr><tr><td>Noise 0.15</td><td>0.973</td><td>0.889</td><td>0.457</td><td>0.243</td><td>+0.084</td><td>[+0.078, +0.091]</td><td>&lt;0.001</td></tr><tr><td>Noise 0.20</td><td>0.957</td><td>0.873</td><td>0.436</td><td>0.228</td><td>+0.084</td><td>[+0.077, +0.094]</td><td>&lt;0.001</td></tr><tr><td>Noise 0.25</td><td>0.962</td><td>0.870</td><td>0.416</td><td>0.268</td><td>+0.093</td><td>[+0.086, +0.099]</td><td>&lt;0.001</td></tr><tr><td>8 regions</td><td>0.998</td><td>0.981</td><td>0.432</td><td>0.141</td><td>+0.017</td><td>[+0.014, +0.021]</td><td>&lt;0.001</td></tr><tr><td>15 regions</td><td>0.995</td><td>0.969</td><td>0.385</td><td>0.101</td><td>+0.027</td><td>[+0.022, +0.031]</td><td>&lt;0.001</td></tr><tr><td>30 regions</td><td>0.989</td><td>0.942</td><td>0.382</td><td>0.131</td><td>+0.046</td><td>[+0.039, +0.054]</td><td>&lt;0.001</td></tr><tr><td>Short pages</td><td>1.000</td><td>0.998</td><td>0.245</td><td>0.321</td><td>+0.002</td><td>[+0.000, +0.004]</td><td>0.23</td></tr><tr><td>Boilerplate</td><td>0.125</td><td>0.030</td><td>0.069</td><td>0.456</td><td>+0.094</td><td>[+0.087, +0.104]</td><td>&lt;0.001</td></tr></table>

Table 7 Long documents built from distinct text (no repetition), column interleave, medium severity, 6% noise. Mean F1 over seeds (number of seeds in parentheses; range over seeds in Fig. 6) and mean runtime per document in seconds. v1 and RETAS-lite are from the Step-7 scaling study; v2 is from the separate Step-8 re-run. At 8,000 tokens and above, v2 hands the document to RETAS-lite, so its F1 equals RETAS-lite by construction.
<table><tr><td></td><td colspan="2">Multi-region v1</td><td colspan="2">Multi-region v2</td><td colspan="2">RETAS-lite</td></tr><tr><td>Tokens (seeds)</td><td>F1</td><td>S</td><td>F1</td><td>S</td><td>F1</td><td>S</td></tr><tr><td>1,000 (5)</td><td>0.771</td><td>0.021</td><td>0.768</td><td>0.015</td><td>0.504</td><td>0.003</td></tr><tr><td>5,000 (5)</td><td>0.423</td><td>0.588</td><td>0.418</td><td>0.219</td><td>0.294</td><td>0.011</td></tr><tr><td>10,000 (5)</td><td>0.334</td><td>2.533</td><td>0.326</td><td>0.036</td><td>0.326</td><td>0.036</td></tr><tr><td>25,000 (3)</td><td>0.284</td><td>14.7</td><td>0.343</td><td>0.153</td><td>0.343</td><td>0.148</td></tr><tr><td>50,000 (3)</td><td>0.256</td><td>61.5</td><td>0.341</td><td>0.511</td><td>0.341</td><td>0.507</td></tr></table>

## 6.7 TREC-5: a reproduction attempt

Table 9 returns to the data of the 2019 paper. We cannot rebuild its split: TREC-5 does not mark which documents have zoning problems, and the 2019 document lists and scoring script are not part of the materials we had. On 30 random documents of degrade5, our reimplementation of the 2019 method matches 95.7% of OCR words, between the 98.5% and 81% reported in 2019 for the two groups; we cannot say more than that.

The table also shows why the matched rate is not a correctness score. On degrade20 the 2019 method still matches 74.0% of words, but its agreement with genalog’s alignment is only 0.102. Against genalog as the reference, multiregion agrees slightly more than RETAS-lite on degrade5 (0.944 against 0.934, paired +0.010 [+0.006, +0.015]) and much more on degrade20 (0.541 against 0.378, +0.163 [+0.145, +0.183]). Java RETAS agrees with genalog at 0.450 and 0.179, although both implement the same published scheme. Because genalog is the reference here, these numbers measure agreement with one aligner, not correctness.

With synthetic character noise and column interleaving on TREC originals (20 documents, medium column interleave, 10% noise), the canonical Step-4 comparison gives multiregion v1 F1 0.914 [0.880, 0.939], RETAS-lite 0.704 [0.667, 0.741] and Java RETAS 0.304 [0.280, 0.327]. These are correctness scores against the synthetic generator’s word-level gold, not evidence on real OCR and not agreement with genalog. A separate Step-8 synthetic-noise and column-interleaving re-run, used only for the v1– v2 comparison in Table 8, gives v1 0.945, v2 0.942 and RETAS-lite 0.736. The two runs use diferent draws and seeds and must not be pooled.

## 7 Discussion and limitations

## What changed since 2019.

The 2019 paper treated zoning as a failure of a tracker that loses its place, and proposed a location-aware tracker that searches a window around its last match. Our re-anchoring extension repairs short drifts but does not resolve large reading-order changes in these tests, because its search starts from the last good match. The methods that handle large displacements start from the other end. They first match what can be matched with high confidence anywhere on the page, and only then fill in the rest.

Table 8 Multi-region v1 and v2, both frozen, on the held-out evaluations (Step-8 re-runs unless noted). ∆ is the paired bootstrap mean of v2 minus v1 with 95% CI. <sup>a</sup>Three seeds only, so we give the mean and the range over seeds instead of a bootstrap interval. Synthetic and long-document entries are distinct from the newspaper and word-box evaluations. The TREC row uses synthetic character noise and column interleaving, not real OCR.
<table><tr><td>Evaluation</td><td>v1 F1</td><td>v2 F1</td><td>∆ (v2 − v1)</td></tr><tr><td>IMPACT test-200, all gold</td><td>0.432 [0.404, 0.460]</td><td>0.403 [0.375, 0.432]</td><td>-0.029 [-0.032, -0.026]</td></tr><tr><td>IMPACT test-200, high-conf.</td><td>0.650 [0.623, 0.675]</td><td>0.607 [0.579, 0.634]</td><td>-0.043 [−0.048, -0.038]</td></tr><tr><td>Word-box gold, all</td><td>0.742 [0.687, 0.795]</td><td>0.735 [0.677, 0.791]</td><td>-0.007 [−0.012, -0.002]</td></tr><tr><td>TREC synthetic (Step 8)</td><td>0.945 [0.935, 0.955]</td><td>0.942 [0.933, 0.952]</td><td>-0.002 [-0.004, -0.001]</td></tr><tr><td>Boilerplate stress</td><td>0.125 [0.118, 0.134]</td><td>0.159 [0.151, 0.168]</td><td>+0.034 [+0.029, +0.041]</td></tr><tr><td>Long document, 50k tokensa</td><td>0.256 (0.239 to 0.287)</td><td>0.341 (0.320 to 0.363)</td><td>+0.085 (+0.077 to +0.096)</td></tr></table>

Table 9 TREC-5 Confusion Track, 30 randomly sampled Federal Register documents. There is no word-level gold, so we report agreement F1 with genalog’s alignment as a soft reference (genalog scores 1 by definition) and the fraction of OCR words that receive any match (“matched”), the closest analogue we can compute to the 2019 success rate. Neither is a correctness score.
<table><tr><td></td><td colspan="3">degrade5 (≈5% CER)</td><td colspan="3">degrade20 (≈20% CER)</td></tr><tr><td>Aligner</td><td colspan="2">Agreement F1</td><td>Matched</td><td colspan="2">Agreement F1</td><td>Matched</td></tr><tr><td>Location-aware (2019)</td><td>0.888</td><td>[0.827, 0.933]</td><td>0.957</td><td>0.102</td><td>[0.076, 0.131]</td><td>0.740</td></tr><tr><td>+ re-anchoring</td><td>0.893</td><td>[0.834, 0.936]</td><td>0.959</td><td>0.107</td><td>[0.081, 0.137]</td><td>0.743</td></tr><tr><td>RETAS-lite</td><td>0.934</td><td>[0.921, 0.945]</td><td>0.912</td><td>0.378</td><td>[0.354, 0.404]</td><td>0.375</td></tr><tr><td>Multi-region v1</td><td>0.944</td><td>[0.934, 0.954]</td><td>0.933</td><td>0.541</td><td>[0.521, 0.561]</td><td>0.671</td></tr><tr><td>Java RETAS</td><td>0.450</td><td>[0.438, 0.463]</td><td>0.311</td><td>0.179</td><td>[0.170, 0.188]</td><td>0.145</td></tr><tr><td>genalog (RETAS) (reference)</td><td></td><td>1.000 [1.000, 1.000]</td><td>1.000</td><td></td><td>1.000 [1.000, 1.000]</td><td>1.000</td></tr></table>

## When to use which aligner.

The synthetic tests suggest that multi-region v1 is suited to page-sized inputs. The real-OCR comparison is subject to the approximate-gold limitations below. On the tested long synthetic documents, RETAS-lite and v2 (which falls back to it) had higher F1 and lower runtime than v1 at the largest lengths. On our synthetic boilerplate page with few distinct word types, the 2019 location-aware method had the highest F1 among the tested methods. We have not established whether this result holds for forms or repeated notices.

## Limitations.

The main real-OCR test uses gold derived from region boxes, with an estimated error in about a quarter of the matched pairs, and the only test with true word boxes has 14 pages. The gold and the article-order convention are not independently validated. All OCR came from one engine,

Tesseract 5. Our RETAS baseline is a simplified reimplementation, and our genalog and Java RETAS results depend on our conversion of their output to word mappings, which we checked for Java RETAS only. The real-OCR data is historical: 17th- to 19th-century Dutch newspapers and 16th- and 17th-century prints. We found no open data set of modern multi-column pages with wordlevel ground truth. The 2019 TREC-5 split could not be reproduced. The multi-region aligner is quadratic in document length, and the v2 attempt to combine it with the 2019 prior did not reach its pre-specified target. The synthetic boilerplate fixture does not establish that a location prior will help on real anchor-scarce pages. We have not independently annotated the remaining disagreements, so we do not classify their causes or propose a new aligner on that basis. In a small LoRA corrector pilot under a region-level evaluation protocol, alignment-derived training pairs did not improve CER over raw OCR; whether better alignment helps downstream correction remains open.

## Reproducibility.

All aligners are deterministic and run on CPU. Parameters are frozen and listed in Sect. 4. The data sets are obtainable: TREC-5 from

NIST, the KB IMPACT ground truth under CC0, and the word-box corpora under CC0. The alignment code will be released under the Apache License 2.0 at https://github.com/ sniperjf/zone-robust-ocr-align at submission.

## 8 Conclusion

We revisited the Zoning Problem from our 2019 alignment paper with OCR-side permutations and word-level construction gold. The multiregion aligner combines unique-word and n-gram anchors, local matching and chunk-to-window assignment; the synthetic experiments examine where local tracking and anchor-based matching fail. Re-anchoring gave a modest gain on the synthetic permutations; on the 200 real pages it was close to the 2019 baseline. The multi-region v1 implementation has a quadratic scaling cost. Approximate gold and the article-order convention remain limitations of the intrinsic evaluations. Independent residual-error annotation and modern multi-column word-level ground truth remain necessary.

Acknowledgements. We thank the Chambers-Grundy Center for Transformative Neuroscience and the Department of Brain Health at the University of Nevada, Las Vegas, for access to an NVIDIA H200 GPU used in a small LoRA corrector pilot (1.691 GPU-hours of device time), mentioned only as a limitation in Sect. 7 and not tabulated as a result. Alignment experiments were run on an AMD EPYC 9175F CPU using Python 3.12, as reported in Sect. 5.

## Statements and Declarations

## Funding.

No funding was received for conducting this study.   
Compute resources are acknowledged above.

## Competing interests.

The author has no competing interests to declare that are relevant to the content of this article.

Ethics approval and consent.

Not applicable.

## Data availability.

The TREC-5 Confusion Track data is available from NIST (https://trec.nist.gov/ data/t5 confusion.html). The KB IMPACT ground truth is available from the KB under CC0 (https://lab.kb.nl/dataset/ ground-truth-impact-project). The word-box corpora (17 frak simple, 17 frak complex and 16 ant simple) are available under CC0 from https://github.com/tboenig/17 frak simple,

https://github.com/tboenig/17 frak complex and https://github.com/tboenig/16 ant simple. We do not redistribute any of these data sets. Derived geometry gold and page lists will be released with the code if rights permit; they are held pending a rights check. The alignment code will be released under the Apache License 2.0 at https: //github.com/sniperjf/zone-robust-ocr-align at submission.

## Code availability.

The alignment code will be released under the Apache License 2.0 at https://github.com/ sniperjf/zone-robust-ocr-align at submission.

## Author contributions.

J.R.F.C. (ORCID 0000-0001-7917-5169) is the sole author and was responsible for conceptualization, methods, software, analysis and writing.

## Declaration of AI-assisted writing and tools.

During the preparation of this work, the author used AI assistants, namely Grok Bot (xAI), Claude via Claude Code (Anthropic), and the OpenAI Codex CLI, to help with language refinement, manuscript organization, drafting portions of the text from author-provided research materials, and implementing and running experiment code under the author’s direction. The OpenAI Codex CLI was also used to proofread manuscript wording against source files. The author designed the study, reviewed and edited the content as needed, and takes full responsibility for the content of this manuscript.

![](images/66c7537400a82f274db45f1cfa7b89ba32d7c1ba13c3ada5f79bbfded59be420.jpg)  
Jorge Ram´on Fonseca Cacho is an Assistant Professor (tenure track) of Computer Science at the University of Nevada, Las Vegas (UNLV), where he has

taught since 2018. He received his Ph.D. in Computer Science from UNLV in 2019 with a dissertation on improving OCR post-

processing with machine learning, and a B.A. in Film and Theater from UNLV. His research covers OCR postprocessing, NLP, reproducible research, and trustworthy, verifiable LLM systems that trace outputs to their sources. As Co-Investigator on Alzheimer’s clinical-trial observatory projects, he leads AI and data engineering for large-scale biomedical text extraction.

## References

[1] Nguyen, T.T.H., Jatowt, A., Coustaty, M., Doucet, A.: Survey of post-OCR processing approaches. ACM Computing Surveys 54(6), 1–37 (2021) https://doi.org/10.1145/ 3453476

[2] Chiron, G., Doucet, A., Coustaty, M., Moreux, J.-P.: ICDAR2017 competition on post-OCR text correction. In: 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), pp. 1423–1428. IEEE, Piscataway, NJ (2017). https://doi.org/10.1109/ICDAR.2017.232

[3] Rigaud, C., Doucet, A., Coustaty, M., Moreux, J.-P.: ICDAR 2019 competition on post-OCR text correction. In: 2019 International Conference on Document Analysis and Recognition (ICDAR), pp. 1588–1593. IEEE, Piscataway, NJ (2019). https://doi.org/10. 1109/ICDAR.2019.00255

[4] Needleman, S.B., Wunsch, C.D.: A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of Molecular Biology 48(3), 443–453 (1970) https://doi.org/ 10.1016/0022-2836(70)90057-4

[5] Wagner, R.A., Fischer, M.J.: The stringto-string correction problem. Journal of the ACM 21(1), 168–173 (1974) https://doi.org/ 10.1145/321796.321811

[6] Yalniz, I.Z., Manmatha, R.: A fast alignment scheme for automatic OCR evaluation of books. In: 2011 International Conference on Document Analysis and Recognition (ICDAR), pp. 754–758. IEEE, Piscataway,

NJ (2011). https://doi.org/10.1109/ICDAR. 2011.157

[7] Kanai, J., Rice, S.V., Nartker, T.A., Nagy, G.: Automated evaluation of OCR zoning. IEEE Transactions on Pattern Analysis and Machine Intelligence 17(1), 86–90 (1995) https://doi.org/10.1109/34.368146

[8] Fonseca Cacho, J.R., Taghva, K.: Aligning ground truth text with OCR degraded text. In: Arai, K., Bhatia, R., Kapoor, S. (eds.) Intelligent Computing (Comp-Com 2019). Advances in Intelligent Systems and Computing, vol. 997, pp. 815–833. Springer, Cham (2019). https://doi.org/10. 1007/978-3-030-22871-2 58

[9] Kantor, P.B., Voorhees, E.M.: The TREC-5 confusion track: Comparing retrieval methods for scanned text. Information Retrieval 2(2–3), 165–176 (2000) https://doi.org/10. 1023/A:1009902609570

[10] Smith, R.: An overview of the Tesseract OCR engine. In: Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), vol. 2, pp. 629–633. IEEE, Piscataway, NJ (2007). https://doi.org/10.1109/ICDAR. 2007.4376991

[11] Papadopoulos, C., Pletschacher, S., Clausner, C., Antonacopoulos, A.: The IMPACT dataset of historical document images. In: Proceedings of the 2nd International Workshop on Historical Document Imaging and Processing (HIP), pp. 123–130. ACM, New York, NY (2013). https://doi.org/10.1145/ 2501115.2501130

[12] IMPACT Project: IMPACT KB Groundtruth. KB Lab, The Hague. https://lab. kb.nl/dataset/ground-truth-impact-project. Public domain / CC0. Accessed 6 November 2026

[13] Gupte, A., Romanov, A., Mantravadi, S., Banda, D., Liu, J., Khan, R., Meenal, L.R., Han, B., Srinivasan, S.: Lights, camera, action! A framework to improve NLP accuracy over OCR documents. In: Document Intelligence Workshop at KDD 2021 (2021).

[14] Microsoft: genalog: Synthetic document generation and text alignment. Python package, MIT licence, https://github.com/microsoft/ genalog. Accessed 6 November 2026 (2021)

[15] Yalniz, I.Z.: RETAS: Recursive Text Alignment Tool, version 1.1. Java software, Center for Intelligent Information Retrieval, University of Massachusetts Amherst; GPL-3.0. Mirror used: https: //github.com/Early-Modern-OCR/RETAS. Accessed 6 November 2026 (2013)

[16] Heckel, P.: A technique for isolating diferences between files. Communications of the ACM 21(4), 264–268 (1978) https://doi.org/ 10.1145/359460.359467

[17] Feng, S., Manmatha, R.: A hierarchical, HMM-based automatic evaluation of OCR accuracy for a digital library of books. In: Proceedings of the 6th ACM/IEEE-CS Joint Conference on Digital Libraries (JCDL), pp. 109–118. ACM, New York, NY (2006). https: //doi.org/10.1145/1141753.1141776

[18] Clausner, C., Pletschacher, S., Antonacopoulos, A.: Flexible character accuracy measure for reading-order-independent evaluation. Pattern Recognition Letters 131, 390– 397 (2020) https://doi.org/10.1016/j.patrec. 2020.02.003

[19] Pletschacher, S., Antonacopoulos, A.: The PAGE (page analysis and ground-truth elements) format framework. In: 2010 20th International Conference on Pattern Recognition (ICPR), pp. 257–260. IEEE, Piscataway, NJ (2010). https://doi.org/10.1109/ ICPR.2010.72

[20] National Institute of Standards and Technology: TREC-5 Confusion Track data. https: //trec.nist.gov/data/t5 confusion.html. Accessed 6 November 2026

[21] Smith, D.A., Murel, J., Allen, J.P., Miller, M.T.: Automatic collation for diversifying corpora: Commonly copied texts as distant

supervision for handwritten text recognition. In: Computational Humanities Research Conference (CHR 2023). CEUR Workshop Proceedings, vol. 3558, pp. 206–221 (2023). https://ceur-ws.org/Vol-3558/paper1708.pdf

[22] Miller, H., Kuflik, T., Lavee, M.: Text alignment in the service of text reuse detection. Applied Sciences 15(6), 3395 (2025) https: //doi.org/10.3390/app15063395

[23] Debaene, F., Maladry, A., Lefever, E., Hoste, V.: Evaluating transformers for OCR post-correction in early modern dutch theatre. In: Proceedings of the 31st International Conference on Computational Linguistics, pp. 10367–10374. Association for Computational Linguistics, ??? (2025). https://aclanthology.org/2025.colingmain.690/

[24] Wang, B., He, T., Ouyang, L., Wu, F., Zhao, Z., Chu, T., Qu, Y., Jin, Z., Zeng, W., Miao, Z., Xu, B., Niu, J., Cai, M., Qiu, J., Zhang, Q., Ma, D., Sun, Y., Dong, H., Zhang, W., Xiao, J., Shi, J., Liao, P., Zhao, X., Zhong, H., Wei, L., Yu, J., Yang, J., Li, W., Wang, S., Wu, Q., Zhou, X., Li, W., Li, Z., Tu, Z., Wu, J., Wu, L., Xu, C., Chen, K., Zhang, W., Qiao, Y., Zhou, B., Lin, D., He, C.: MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale. arXiv preprint arXiv:2604.04771 (2026). https:// arxiv.org/html/2604.04771

[25] Guechaoui, M., Zellagui, M.D., Chaib, S., Dhelim, S.: RefLAM: A Reference-Grounded Line Annotation Pipeline for Historical Arabic Manuscripts. arXiv preprint arXiv:2608.25140 (2026). https://arxiv.org/ html/2608.25140

[26] Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman and Hall/CRC, New York (1994). https://doi.org/10.1201/ 9780429246593

[27] Boenig, M.: 17 frak simple, 17 frak complex and 16 ant simple: PAGE ground truth with word coordinates. CC0 1.0. https: //github.com/tboenig/17 frak simple, https: //github.com/tboenig/17 frak complex,

https://github.com/tboenig/16 ant simple.

Accessed 6 November 2026