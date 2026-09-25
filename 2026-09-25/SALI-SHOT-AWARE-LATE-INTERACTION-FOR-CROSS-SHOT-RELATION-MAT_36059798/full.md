# SALI: SHOT-AWARE LATE INTERACTION FOR CROSS-SHOT RELATION MATCHING IN TEXT-TO-VIDEO RETRIEVAL USING FILM-GRAMMAR KNOWLEDGE

Toya Oyama<sup>⋆†</sup> Rainer Lienhart<sup>‡</sup> Shin’ichi Satoh<sup>†⋆</sup>

<sup>⋆</sup>The University of Tokyo <sup>†</sup>National Institute of Informatics <sup>‡</sup>University of Augsburg

## ABSTRACT

Text-to-video retrieval usually represents a video clip by a single embedding. This embedding often loses important relations between people. E.g., an interaction “Anna confronts Mark” is regularly filmed as alternating shot and reverse shot of both (Fig. 1a). No single shot or averaged embedding over clip shots captures this relation. Thus, we propose SALI (Shot-Aware Late Interaction). It extracts the subject and object from a single-sentence query, and matches the query, its subject and object text embeddings against each visual shot embedding of a video clip. The matching operator is greedy max or optimal transport. A film-grammar penalty in fine-tuning adds a small, consistent shift. Built on CLIP4Clip-meanP, SALI keeps overall recall on par on Condensed Movies and ActivityNet while raising R@1 on multishot relation queries by 3 and 12 points, the most among all compared methods, and improves such queries on MSR-VTT at a cost of 1.4 R@1 overall.

Index Terms— text-to-video retrieval, late interaction, optimal transport, film grammar, compositionality

## 1. INTRODUCTION

Text-to-video retrieval (T2VR) ranks a gallery of videos by relevance to a text query. Most recent systems build on CLIP [1] and represent the whole clip by one vector, the mean over all frame embeddings [2]. Finer-grained alternatives compare the text embedding with each frame embedding [3, 4] or each query-word embedding with each frame embedding [5]. All treat a video as a bag or sequence of frames, but movies and dramas are organized by camera cuts into shots and scenes [6–9], with dozens of shots per twominute clip (Table 1). This setting, single-sentence queries against clips of tens of shots, is the one we address. A single video clip embedding ignores this structure. We call this gap between the clip’s structure and its single embedding the single-video-embedding gap. Because of it, retrieval on movies and dramas degrades.

![](images/baa26fcca1739892a20b7cc17072ec9ed0cee0b75e92e29558ae11a72028a5c4.jpg)  
Fig. 1. Film grammar breaks single-embedding and per-shot matching. (a) An interaction between two people is filmed as alternating shot and reverse shot: no single shot shows the relation. (b) Per-shot max gives the true video and a one-shot look-alike the same score. (c) Mean pooling averages the shots into one vector and dilutes the relation. (d) SALI matches the subject and the object to diferent shots.

The failure is not just that long videos dilute local evidence: film grammar [10] places the evidence in a specific way. An interaction between two people—“Anna confronts Mark”—is regularly filmed as alternating shot and reverse shot [11]: one shot shows Anna, the next Mark, and no single shot depicts the relation itself (Fig. 1a). Per-shot matching therefore cannot separate the true video from a look-alike with one similar shot (Fig. 1b), and mean pooling dilutes the relation (Fig. 1c). Sec. 2 shows that zero-shot CLIP degrades most on queries about relations between people when the video clip has many shots.

The closest idea is late interaction [12–14]: a video is represented by many vectors, one per frame or token, and each query token is matched to its most similar vector. The similarities are combined into one score by max, by learned or attention weights [5, 15], or by optimal transport (OT) [16, 17]. Such matcher finds “Anna” and “Mark” separately but never asks how they appear in the query.

We close the single-video-embedding gap with SALI (Shot-Aware Late Interaction), contributing: (1) a diagnosis on Condensed Movies [6] showing that a single video embedding fails specifically on queries about relations between people over many-shot videos; (2) SALI, which keeps one embedding per shot and matches the complete query, its subject and its object against the shots with a matching operator<sup>1</sup>, greedy max or optimal transport; (3) an analysis of which matching operator to use at training and at inference, showing that the gains on relational queries come from greedy max at inference while a film-grammar penalty in fine-tuning adds a small, consistent shift.

## 2. DIAGNOSIS: DOES THE PROBLEM EXIST?

Benchmarks. Existing T2VR benchmarks difer in what a video is: a short web clip (MSR-VTT [18]), an untrimmed activity video with a paragraph caption (ActivityNet Captions [19]), a movie segment of a few seconds (LSMDC [7], MAD [8]), or a two-minute movie scene with a caption about its characters (Condensed Movies, CM [6]). Table 1 shows their number of shots per video and the share of relational queries, i.e., queries with a verb from a list of 11 interpersonal verbs and two person nouns or pronouns. CM is our primary testbed.

Where T2VR fails. We test our hypothesis that a single video embedding fails specifically when the query describes a relation between two people and the video spreads that relation over many shots. Our diagnosis has four steps. (1) Pool: all 10,621 obtainable CM videos (Sec. 4), each with its caption as the query and the video as the only correct answer (5.0% of the queries are relational). (2) Retrieval: zero-shot CLIP ViT-B/16 [1] with one vector per video (the mean of its shot means), so that the result reflects CLIP itself. (3) Split: queries into relational vs. non-relational, videos into quartiles by shot count. (4) Measure: R@10 (%) per quartile of the matching video. For relational queries it falls from 28.3 in the fewestshot quartile to 14.8 in the most-shot quartile (−13.5). For non-relational queries it falls only from 23.5 to 18.2 (−5.3). A single video embedding thus fails specifically on relational queries over many-shot videos. We therefore evaluate on Rel-Multi, the relational queries whose videos have at least the median shot count (34 shots).

## 3. SALI: METHOD

Shot and query embeddings (Fig. 2). A video is split into S shots by an of-the-shelf shot-boundary detector (PySceneDetect [20]). Each frame is encoded by CLIP’s [1] image encoder. The mean of the frame embeddings of shot k is its shot embedding $v _ { k } .$ , and the S shot embeddings form the video feature tensor $V \in \mathbb { R } ^ { S \times d }$ . On the query side we use the same list of interpersonal verbs as in Sec. 2 (11 hand-selected patterns such as confronts, talks to, looks at). The verb splits the sentence: the words before it form the subject span and the words after it the object span. The full query and the two spans are encoded by the same CLIP text encoder into three text embeddings q<sub>full</sub>, $q _ { \mathrm { s u b j } }$ and $q _ { \mathrm { o b j } }$ . Queries without such a verb use $q _ { \mathrm { s u b j } } { = } q _ { \mathrm { o b j } } { = } q _ { \mathrm { f u l l } }$ . Text and shot embeddings pass through linear projections $p _ { q }$ (shared by the three text embeddings) and $p _ { v }$ , both initialized as the identity matrix.

Table 1. Datasets (oficial splits; CM: obtainable clips). Shots: median shot count per video; rel.: relational queries; Rel-Multi: rel. queries whose video has at least the median shot count (CM 34, ANet 4); queries: # of queries. ANet test: the val\_1 split.
<table><tr><td colspan="4">Shots rel. (#/ %) Rel-Multi (# / %) queries (#)</td></tr><tr><td>CM [6] train</td><td>33 410/4.8</td><td>222/2.6</td><td>8,499</td></tr><tr><td>CM [6] val</td><td>40 41/4.8</td><td>26/3.1</td><td>846</td></tr><tr><td>CM [6] test</td><td>36 82/6.4</td><td>48/3.8</td><td>1,276</td></tr><tr><td>ANet [19] test</td><td>3 854/17.4</td><td>419/8.5</td><td>4,917</td></tr><tr><td>MSR-VTT [18] test</td><td>3 61/6.1</td><td>一</td><td>1,000</td></tr></table>

Matching operators. A matching operator scores a query against a video clip from the similarities between the three projected text embeddings (full, subj, obj) and the $S$ projected shot embeddings, which we collect in $M ~ \in ~ \mathbb { R } ^ { 3 \times S }$ (cosine similarities). We use two operators. Greedy max lets each text embedding pick its most similar shot: $s _ { \mathrm { m a x } } =$ ${ \frac { 1 } { 3 } } \sum e$ max<sub>k</sub> $M _ { e k }$ . Optimal transport (OT) instead distributes each text embedding’s mass (1/3) over the shots through a transport plan $T \in \mathbf { \mathbb { R } } _ { > 0 } ^ { 3 \times S }$ subject to row sums of $1 / 3 ,$ and column sums to at most c, a learnable per-shot capacity in (0.1, 1), so that the three embeddings cannot all be placed on the same shot. We compute $T$ with five iterations of entropic Sinkhorn [16, 21] with a learnable temperature ε and score $s _ { \mathrm { O T } } = \langle T , M \rangle$ ⟩. Both operators are diferentiable. With c = 1 and $\varepsilon  0$ , the plan puts each row’s mass on one shot and s<sub>OT</sub> equals $s _ { \mathrm { m a x } }$

Film-grammar penalty. Ofline, faces are embedded [22] and clustered across shots (cosine similarity > 0.4). Each shot takes the identity of its largest face, which gives the sameperson matrix $\dot { P ^ { \mathrm { ~ \scriptsize ~ ( ~ 0 ~ , ~ 1 ~ ) ~ } } } ^ { \dot { S } \times S }$ with $P _ { k l } = 1$ if shots k and l show the same person. $P$ is used only in the training loss, so inference needs no face pipeline. The two penalties follow from shot/reverse-shot editing [11]. P1: Same-shot penalty. Each shot centers on one person, so the subject and the object being matched to the same shot indicates a look-alike rather than the relation. P2: Same-person penalty. The subject and the object are diferent characters, so the two being matched to shots that show the same person $( P _ { k l } { = } 1 )$ contradicts the query. With $T _ { s }$ and $T _ { o }$ the subject and object rows of the transport plan T, these become two diferentiable terms:

![](images/04684a2a42b364b41e03221a2098f44d322b371a85111f1e5da28b6317461cc8.jpg)  
Fig. 2. SALI overview. (a) Ofline: the same-person matrix $P$ from cross-shot face clustering (training videos only). Training: optionally, the two penalty terms (Eq. 1) are computed from the transport plan of the matching operator and shape the encoder (Table 3). Inference: the same parameters are used with either operator. (b) The two alternatives for the matching operator, greedy max and OT, and the penalty.

$$
\begin{array} { r } { s = s _ { \mathrm { o p } } - \underbrace { \mathrm { s o f t p l u s } ( w _ { 1 } ) T _ { s } T _ { o } ^ { \top } } _ { \mathrm { P l : ~ s a m e - s h o t p e n a l t y } } - \underbrace { \mathrm { s o f t p l u s } ( w _ { 2 } ) T _ { s } P T _ { o } ^ { \top } } _ { \mathrm { P 2 : ~ s a m e - p e r s o n ~ p e n a l t y } } , } \end{array}\tag{1}
$$

with $s _ { \mathrm { o p } } ~ \in ~ \{ s _ { \mathrm { m a x } } , s _ { \mathrm { O T } } \}$ the operator’s score and w<sub>1</sub>, w<sub>2</sub> global learnable scalars initialized near zero (softplus keeps the weights positive). Positive pairs can avoid the penalty by placing subject and object mass on diferent shots. Both terms are symmetric in subject and object. The information about their specific role enters only through $q _ { \mathrm { f u l l } }$ . The penalty is always computed on the transport plan T, also when the score uses greedy max.

Score and training. The clip-level score $s _ { \mathrm { c l i p } }$ is the cosine between the projected $q \mathrm { f u l l }$ and the projected mean of all frame embeddings, with its own identity-initialized projections. So it equals CLIP4Clip-meanP [2] at the start of fine-tuning. It is mixed with the shot-level score s of Eq. 1 by a learned gate g: $\hat { s } = g s + ( 1 - g ) s _ { \mathrm { c l i p } } . g$ is one scalar per query type (with or without an interpersonal verb), initialized near 0.5. The model is trained with symmetric InfoNCE [2] over sˆ while both CLIP towers are fine-tuned. The penalty is part of this objective, not a set of hard negatives [23]. At inference, the trained model is used unchanged; the matching operator is either greedy max or OT (Table 2).

## 4. EXPERIMENTS

Datasets and protocols. Of the 34,185 clips in the released catalog of CM [6], 10,621 (31.1%) are still obtainable: 8,499 from oficial train movies, 846 from val movies and 1,276 from test movies. Relational captions are only about 6% of CM, and the oficial test movies contain only 48 Rel-Multi queries (Table 1), so we build a larger evaluation gallery that shares no movie with the training set: all 2,122 val- and testmovie clips plus 140 train-movie clips selected by caption to add relational queries, 2,262 clips in total with 203 relational queries (113 in Rel-Multi). We train on 7,073 of the remaining train-movie clips (1,221 movies); 724 clips of 136 further train movies form our validation split (val) for epoch selection. The other 562, selected with the 140, share a movie with the training set and are used for neither training nor evaluation. ActivityNet Captions [19] (val\_1, the standard test split [2]) and MSR-VTT [18] (1k-A test [24]) are the generalization datasets (Table 4); on both, 10% of the training videos are used as val. Person nouns (Sec. 2) include capitalized non-initial tokens. Rel-Multi requires the median shot count of Table 1 (34 on CM, 4 on ActivityNet, which has fewer shots per video), and MSR-VTT keeps the relational subset only.

Training. All systems share CLIP ViT-B/16 [1] and the same oficial number of frames per video (128 on CM; 64 on ActivityNet and 12 on MSR-VTT, the released defaults). SALI regroups those frames by the shot their timestamp falls in (≈3.6 frames per shot on CM). SALI is warm-started from a CLIP4Clip-meanP [2] checkpoint trained with the oficial implementation on this split, and both towers are fine-tuned for 4 epochs (AdamW, $1 0 ^ { - 7 }$ backbone $/ 1 0 ^ { - 4 }$ head, batch 48, symmetric InfoNCE, 3 seeds). The baselines (CLIP4Clip [2], X-CLIP [5], TS2-Net [4]) are run with their oficial implementations on this split under each method’s own recipe (batch 24) and trained until their val curve saturates—5 epochs for CLIP4Clip and X-CLIP, 10 for TS2-Net. X-Pool [3] is excluded because its query-conditioned pooling admits no precomputed index. The OT temperature ε and per-shot capacity c are learnable and start at 0.07 and 0.89. The penalty weights $w _ { 1 } , w _ { 2 }$ start at −3 (softplus(−3)≈0.05).

Evaluation. We report R@K (%) on all evaluation queries and on their Rel-Multi subsets. For every method, the epoch is chosen by overall R@1 on the val splits above. The evaluation split is never used for selection. The performance of SALI in Tables 2–4 is averaged over three training runs with diferent seeds (seed standard deviation 0.2 R@1 overall, 0.4 on Rel-Multi). The baselines have no randomly initialized parameters and are single runs.

Main results (Table 2). Both SALI variants start from CLIP4Clip-meanP [2]. SALI+max raises overall R@5 and

Table 2. CM [6] R@K (%, higher is better) with the same 128 frames per video. SALI+max: greedy max at training and inference; SALI+OT: OT at training and inference. <sup>∗</sup>: position tables extended beyond the released frame cap. Shaded: ours; bold: best per column.
<table><tr><td rowspan="2">Method</td><td>All (n=2262)</td><td>Rel-Multi (n=113)</td></tr><tr><td>R@1 R@5 R@10</td><td>R@ R@5 R@10 1</td></tr><tr><td>CLIP4Clip-meanP [2] CLIP4Clip-seqTransf [2]</td><td>27.23 47.92 56.72 * 25.29 946.91 55.13</td><td>23.89 45.13 57.52 324.78 42.48 51.33</td></tr><tr><td>X-CLIP [5]* TS2-Net [4]*</td><td>25.99 49.20 20.34 41.73</td><td>57.74 22.12 46.90 52.21 51.4618.5833.63 46.90</td></tr><tr><td>SALI+max</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>26.94 49.19</td><td>57.26 26.84 48.08</td></tr><tr><td></td><td></td><td>56.93</td></tr><tr><td>SALI+OT</td><td>27.65 48.57</td><td></td></tr><tr><td></td><td></td><td>57.47 23.30 46.02 51.33</td></tr><tr><td></td><td></td><td></td></tr></table>

R@10 and Rel-Multi R@1 and R@5, while overall R@1 drops by 0.29. Its Rel-Multi R@1 and R@5 are the highest of all methods. SALI+OT moves the other way. It has the highest overall R@1 of all methods but a Rel-Multi R@1 below CLIP4Clip-meanP.

Ablation (Table 3). Each comparison changes one setting. (i) The penalty raises Rel-Multi R@1 in every configuration, by less than one query each, and never lowers overall R@1. The two highest Rel-Multi R@1 values of Table 3 are both penalty rows. The penalty’s Rel-Multi R@1 gain is largest with OT training and greedy max at inference, whereas the penalty trained with greedy max lowers Rel-Multi R@5. (ii) For the two OT-trained models, switching to greedy max at inference raises Rel-Multi R@1 by about four queries, far beyond the seed standard deviation, at a cost of 0.7–0.8 overall R@1. We interpret this as follows: at inference the transport plan spreads each text embedding’s mass over several shots, which helps ordinary queries but hurts relational ones, whose evidence is one shot per person.

Controls. SALI mixes two scores through the gate and continues training a CLIP4Clip-meanP checkpoint, so its gains could come from ensembling or from longer training. (a) Ensembling: we train three CLIP4Clip-meanP models with diferent seeds and average the similarity scores of each pair. Over the better model of the pair, the average gains at most 0.88 in overall R@1 and R@5, and at most 2.65 and 1.77 in Rel-Multi R@1 and R@5. (b) Longer training: four more epochs of CLIP4Clip-meanP lower R@1 and R@5. SALI+max’s gains in overall R@5 and Rel-Multi R@1 and R@5 exceed the gains of both experiments; in Rel-Multi R@1 the margin over the largest ensembling gain is only 0.30.

Mechanism. In all runs of Tables 2 and 3, the learned penalty weights stay near their start (0.05). To test a strong penalty, we trained max+pen with the weights started at 0.31, 0.67 and 2.1 (3 seeds each). With the penalty at inference, Rel-Multi R@10 falls as the weight grows, from 56.9 (SALI+max) to 56.3, 54.3 and 40.4: the penalty pushes the correct videos down rather than the distractors. Without the penalty at inference, all three models equal SALI+max within seed noise, and

Table 3. Training vs. inference configuration on CM (setting of Table 2, 3 seeds). Train/Infer: matching operator (max: greedy max; OT) and whether the film-grammar penalty (pen) is applied. Shaded: penalty in training only, penalty-free greedy max at inference; bold: best per column.
<table><tr><td colspan="2">Train</td><td colspan="2">Infer</td><td colspan="2">All</td><td colspan="2">Rel-Multi</td></tr><tr><td>max</td><td>OT pen</td><td>max</td><td>OT</td><td>pen</td><td>R@1</td><td>R@5</td><td>R@1 R@5</td></tr><tr><td></td><td> $\begin{array} { r l r l } { \surd } & { { } } & { - } & { { } - } \end{array}$ </td><td></td><td>√</td><td></td><td>26.94</td><td>49.19 26.84</td><td>48.08</td></tr><tr><td></td><td> $\begin{array} { r l r l } { \surd } & { { } } & { - } & { { } { \surd } } \end{array}$ </td><td></td><td>√</td><td>√</td><td>27.09</td><td>49.01</td><td>27.14 44.84</td></tr><tr><td>√</td><td></td><td>L</td><td>√ 一</td><td>一</td><td>27.09</td><td>48.872</td><td>26.55 45.13</td></tr><tr><td>一</td><td> $\checkmark$ </td><td></td><td></td><td>5 一</td><td>27.65</td><td>48.57</td><td>23.30 46.02</td></tr><tr><td></td><td> $\checkmark$ </td><td></td><td></td><td></td><td>26.86</td><td>48.73</td><td>26.55 47.49</td></tr><tr><td></td><td> $\checkmark$ </td><td>√</td><td></td><td>√  $\checkmark$ </td><td>27.69</td><td>48.82</td><td>23.89 45.72</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>27.0048.85</td><td>27.14 46.90</td></tr></table>

Table 4. Generality on ActivityNet [19] and MSR-VTT [18] (R@K, %). CLIP4Clip [2]: the meanP variant trained with the oficial implementation on each dataset. SALI+max and SALI+OT as in Table 2; 3 seeds. MSR-VTT relational subset: n = 61. Shaded: SALI (ours); bold: best per column.
<table><tr><td rowspan=1 colspan=1>ActivityNetAll     Rel-MultiR@1 R@5 R@1 R@5</td><td rowspan=1 colspan=1>MSR-VTTAll        Rel.R@1 R@5 R@1 R@5</td></tr><tr><td rowspan=1 colspan=1>CLIP4Clip40.25 71.77 40.10 74.22</td><td rowspan=1 colspan=1>46.2070.30 54.1078.69</td></tr><tr><td rowspan=1 colspan=1>SALI+max 41.12 71.23 52.19 78.52SALI+OT 40.70 71.50 50.68 77.88</td><td rowspan=1 colspan=1>44.83 71.40 55.19 80.8744.97 71.60 53.55 78.14</td></tr></table>

OT+pen models trained with the same weights behave alike. Even a strong penalty thus leaves the encoder unchanged. In short, the Rel-Multi gains come from greedy max at inference (Table 3).

Generality (Table 4). On ActivityNet [19], SALI+max keeps overall recall on par with CLIP4Clip-meanP while Rel-Multi R@1 rises by 12 points and R@5 by 4. On MSR-VTT [18], overall R@1 drops by 1.4 while overall R@5 and relational R@1 and R@5 rise. SALI+OT is on par with SALI+max overall and below it on relational queries on both datasets.

Limitation. Shot-level matching costs overall R@1 on short clips: on MSR-VTT (10–30 s, 12 frames per video) SALI+max is 1.4 R@1 below CLIP4Clip-meanP. On ActivityNet, whose videos have the same median shot count but run about two minutes with 64 frames, SALI+max raises overall R@1 and raises Rel-Multi recall by a wide margin. Our experiments do not tell whether the short clips or the small frame budget (12 frames) cause the drop on MSR-VTT.

## 5. CONCLUSION

SALI represents a clip by one embedding per shot and matches the query and its subject and object against the shots with greedy max or OT. The gains on relational queries come from greedy max at inference. A film-grammar penalty in finetuning adds a small, consistent shift. Built on CLIP4ClipmeanP, SALI+max achieves the highest Rel-Multi R@1 and R@5 on Condensed Movies with overall recall on par, and SALI+OT the highest overall R@1. SALI+max also raises Rel-Multi recall on ActivityNet by a wide margin and improves relational queries on MSR-VTT at a cost of 1.4 overall R@1.

## 6. REFERENCES

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021, pp. 8748–8763.

[2] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li, “CLIP4Clip: An empirical study ofCLIP for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293–304, 2022.

[3] Satya Krishna Gorti, Noël Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei Yu, “X-Pool: Cross-modal language-video attention for text-video retrieval,” in Proc. IEEE/CVF CVPR, 2022, pp. 4996–5005.

[4] Yuqi Liu, Pengfei Xiong, Luhui Xu, Shengming Cao, and Qin Jin, “TS2-Net: Token shift and selection transformer for textvideo retrieval,” in Proc. ECCV, 2022, pp. 319–335.

[5] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji, “X-CLIP: End-to-end multi-grained contrastive learning for video-text retrieval,” in Proc. ACM Multimedia, 2022, pp. 638–647.

[6] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman, “Condensed movies: Story based retrieval with contextual embeddings,” in Proc. ACCV, 2020, pp. 460–479.

[7] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele, “Movie description,” International Journal of Computer Vision, vol. 123, no. 1, pp. 94–120, 2017.

[8] Mattia Soldan, Alejandro Pardo, Juan León Alcázar, Fabian Caba Heilbron, Chen Zhao, Silvio Giancola, and Bernard Ghanem, “MAD: A scalable dataset for language grounding in videos from movie audio descriptions,” in Proc. IEEE/CVF CVPR, 2022, pp. 5016–5025.

[9] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin, “MovieNet: A holistic dataset for movie understanding,” in Proc. ECCV, 2020, pp. 709–727.

[10] Daniel Arijon, Grammar of the Film Language, Focal Press, London, 1976.

[11] David Bordwell, Kristin Thompson, and Jef Smith, Film Art: An Introduction, McGraw-Hill Education, New York, NY, twelfth edition, 2019, ISBN 978-1-260-05608-2.

[12] Omar Khattab and Matei Zaharia, “ColBERT: Eficient and efective passage search via contextualized late interaction over BERT,” in Proc. ACM SIGIR, 2020, pp. 39–48.

[13] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu, “FILIP: Fine-grained interactive language-image pretraining,” in Proc. ICLR, 2022.

[14] Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno Kriz, Celso M. de Melo, Benjamin Van Durme, and Rama Chellappa, “Video-ColBERT: Contextualized late interaction for text-to-video retrieval,” in Proc. IEEE/CVF CVPR, 2025, pp. 19691–19701.

[15] Qiang Wang, Yanhao Zhang, Yun Zheng, Pan Pan, and Xian-Sheng Hua, “Disentangled representation learning for textvideo retrieval,” arXiv preprint arXiv:2203.07111, 2022.

[16] Marco Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” in Proc. NeurIPS, 2013, pp. 2292–2300.

[17] Shraman Pramanick, Li Jing, Sayan Nag, Jiachen Zhu, Hardik Shah, Yann LeCun, and Rama Chellappa, “VoLTA: Visionlanguage transformer with weakly-supervised local-feature alignment,” Transactions on Machine Learning Research, 2023.

[18] Jun Xu, Tao Mei, Ting Yao, and Yong Rui, “MSR-VTT: A large video description dataset for bridging video and language,” in Proc. IEEE CVPR, 2016, pp. 5288–5296.

[19] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles, “Dense-captioning events in videos,” in Proc. IEEE ICCV, 2017, pp. 706–715.

[20] Brandon Castellano, “PySceneDetect: Video scene cut detection and analysis tool,” https://github.com/ Breakthrough/PySceneDetect, 2026, Version 0.7, released 2026-05-03.

[21] Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard, “Scaling algorithms for unbalanced optimal transport problems,” Mathematics of Computation, vol. 87, no. 314, pp. 2563–2609, 2018.

[22] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou, “ArcFace: Additive angular margin loss for deep face recognition,” in Proc. IEEE/CVF CVPR, 2019, pp. 4685–4694.

[23] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou, “When and why vision-language models behave like bags-of-words, and what to do about it?,” in Proc. ICLR, 2023.

[24] Youngjae Yu, Jongseok Kim, and Gunhee Kim, “A joint sequence fusion model for video question answering and retrieval,” in Proc. ECCV, 2018, pp. 487–503.