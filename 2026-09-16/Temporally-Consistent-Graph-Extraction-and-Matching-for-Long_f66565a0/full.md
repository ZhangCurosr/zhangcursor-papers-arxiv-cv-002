# Temporally Consistent Graph Extraction and Matching for Longitudinal Angiographic Images

Linus Kreitner<sup>1</sup> , Laurin Lux<sup>1,2</sup> , Carmen Baumann<sup>3</sup> , Daniel Rueckert<sup>1,2,4</sup> and Martin J. Menten<sup>1,2,4</sup>

<sup>1</sup> Chair for AI in Healthcare and Medicine, Technical University of Munich (TUM) and TUM University Hospital, Munich, Germany

<sup>2</sup> Munich Center for Machine Learning (MCML), Munich, Germany

3 Ophthalmology, Technical University of Munich, Munich, Germany 4 Department of Computing, Imperial College London, UK {linus.kreitner, martin.menten}@tum.de

Abstract. Recent advances in angiographic imaging have enabled longitudinal visualization of the microvasculature. Image processing pipelines based on vessel graphs are able to resolve subtle temporal changes at the level of individual blood vessels. However, current strategies for graph extraction, refinement, and matching are highly sensitive, with even minuscule diferences in the underlying segmentation map resulting in substantially diferent vessel graphs. These artifacts severely inhibit the ability to accurately match sequential vessel graphs of the same subject over time. To address this problem, we propose a strategy that matches graphs before jointly refining them. Specifically, we perform an early matching after basic graph extraction before removing spurious bulges and merging junctions in both graphs using joint information. In experiments with complex retinal vessel graphs, we demonstrate that this strategy results in a higher matched area without graph fragmentation compared to separate or no refinement, respectively.

Keywords: Graph extraction · Angiography · Longitudinal imaging · Graph matching · Optical coherence tomography angiography · Retina

## 1 Introduction

Advances in medical imaging technology have made detailed visualization of the microvasculature a clinical reality [11,24,29]. The advent of non-invasive, cost-efective angiographic imaging has resulted in the emergence of longitudinal datasets that monitor growth, regression, and remodeling of the vasculature [1,13,26]. Due to the complexity of the underlying anatomy, clinicians and researchers rely on automated image processing algorithms to extract quantitative vascular biomarkers. As a first step, these tools usually delineate the blood vessels [21,12,18,2]. The resulting segmentation map already facilitates calculation of basic biomarkers, such as vessel density, and their changes over time.

However, with improving imaging capabilities there is a growing interest in more granular biomarkers that resolve temporal changes at the level of individual blood vessels. Such fine-grained analysis requires advanced image processing pipelines that extract and process vessel graphs from angiographic images [14]. Vessel graphs are typically constructed by obtaining a topological representation of a segmentation mask and converting it to a graph that compactly encodes bifurcation points and connecting vessel segments as nodes and edges [19,23,30,8]. Additionally, nodes and edges can be imbued with additional features, such as vessel thickness, segment length, curvature, or branching angle [5,25]. Afterwards, this basic graph can be iteratively refined by removing superfluous nodes and edges. In the longitudinal setting, graph matching is required to establish correspondences between edges that represent the same anatomical structures at diferent time points [7,6]. However, current strategies for graph extraction, refinement, and matching are highly sensitive to small variations in the input segmentation mask. Even minuscule diferences can result in substantially diferent vessel graphs. These artifacts severely inhibit the ability to accurately match sequential graphs of the same subject, even if a clear correspondence between the vasculature at two time points exists. Currently, these matching errors greatly diminish the utility of vessel graphs for analysis of longitudinal angiographic imaging data.

Our work addresses this problem by making a series of contributions:

1. We show that highly similar angiographic images can yield vastly diferent vessel graphs. We establish that small variations in the underlying segmentation map have an outsized impact on the graph refinement stage.

2. To address this problem, we propose to refine longitudinal vessel graphs under consideration of temporally adjacent samples. Specifically, we perform an early matching after basic graph extraction before removing spurious bulges and merging junctions in both graphs using joint information (see Figure 1).

3. In experiments using a unique optical coherence tomography angiography (OCTA) dataset with repeated intra-subject scans, we demonstrate the utility of our method in a highly challenging setting that requires matching longitudinal vessel graphs with 2,000 edges.

## 2 Methods

We propose an early graph matching and joint refinement strategy for better graph correspondence between time points. We first introduce the simpler "separate refinement" graph processing pipeline that sequentially extracts (Section 2.1), refines (Section 2.2), and matches (Section 2.3) two time-adjacent graphs. The individual steps build on well-established principles such as skeleton extraction, topological simplification, and artifact removal, and we show that the pipeline performs comparably to widely adopted baseline methods[20,4,22]. We then show how to adapt this baseline for our proposed strategy of early graph matching and joint refinement (Section 2.4). Controlling the entire pipeline allows us to quantify the benefit of our joint refinement strategy without extraneous factors afecting the results.

![](images/30d11ae8c7f03078fac2a5f4b8eddd90547115be75554bd770399f4f0a12dcdb.jpg)  
Fig. 1. Pipeline to refine and match two vessel graphs from 2D segmentation masks of longitudinal angiographic images. We propose early matching of temporally adjacent graphs and refinement using joint information to promote convergence into a similar graph representation.

## 2.1 Graph extraction

We define a vessel graph $G = ( V , E )$ as a set of nodes V denoting bifurcation points as well as a set of edges E representing the connecting vessels.

Node extraction To construct the graph, we initially extract the skeleton of the 2D segmentation mask [17]. We define all centerline pixels with a single 8-connected neighbor as leaf nodes and pixels with more than two neighbors as junction nodes. All adjacent junction nodes are grouped into hubs before applying a breadth-first search to find all distinct paths between hubs.

Edge assignment We extract the distinct paths (edges) between hubs as the shortest 8-connected pixel walks between two hubs that do not visit another hub in between. All junction nodes within a hub are merged into a single merged node and placed at the centroid of the hub. We then remove redundant skeleton pixels around the merged node that no longer lie on the shortest path, or add pixels to reconnect the incoming edges if the centroid is not on the skeleton.

Node and edge features Finally, we compute the radius r of both nodes and edges as the exact distance transform (EDT) at their coordinates or the median EDT at their skeleton coordinates, respectively. Additionally, each edge carries its underlying skeleton and surrounding 2D segmentation mask pixels as features.

## 2.2 Graph refinement

The previously described graph extraction results in a raw graph that still contains many small spurious edges due to small variations in the segmentation mask. To separate true vessels from these artifacts, we conduct three additional refinement steps to simplify the vessel graph topology.

Bulge removal This first refinement step aims to remove spurious edges e leading from a base node $n _ { b }$ towards a leaf node $n _ { l }$ . We use the resolution-agnostic bulge size metric defined by Drees et al. [5],

$$
\beta ( e ) = \frac { \mathrm { l e n g t h } ( e ) - \mathrm { r a d i u s } ( n _ { b } ) + \mathrm { r a d i u s } ( n _ { l } ) } { \mathrm { r a d i u s } ( e ) } ,\tag{1}
$$

to remove any bulges with a size smaller than a threshold $T _ { \mathrm { b u l g e } }$

Adjacent node merging When two vessels cross approximately at $\mathrm { ~ a ~ } 9 0 ^ { \circ }$ angle, a degree-4 crossing node is created. However, if two thick vessels overlap at an acute angle, the medial axis often produces an H-shaped configuration consisting of two degree-3 junction nodes connected by a short bridging edge $e _ { H }$ . The second refinement step resolves this artifact by merging these two junction nodes into a single higher-degree crossing node. The cost of this merging operation consists of two terms. The first penalizes long, thin bridges and is defined as $\begin{array} { r } { \mathcal { C } _ { \mathrm { m e r g e } } ^ { e _ { H } } = \frac { \mathrm { l e n g t h } ( e _ { H } ) } { \mathrm { r a d i u s } ( e _ { H } ) } } \end{array}$ . A second term measures geometric plausibility by quantifying how well incoming edges can be paired with outgoing edges of similar orientation and radius. Crossings involving more than two junction nodes are resolved by iteratively merging node pairs starting with the lowest merge cost.

Skeleton cleanup Merging junction nodes and placing a new crossing node at the center of their connecting edge leads to locally suboptimal skeleton geometry. In particular, the incident branches may share pixel segments near crossings. This final refinement step rewires the skeleton paths around crossing nodes to minimize the overlap of their incoming edges. We remove all skeleton pixels within the node’s radius and then reconnect incoming edges by a straight line.

## 2.3 Graph matching

Graph matching aims to find a correspondence between the edges of two related graphs by establishing a bijective mapping $\mathcal { M } : E _ { A } \mapsto E _ { B }$ that explicitly assigns each edge in graph A to exactly one edge in graph B [15]. However, such a mapping is unlikely to exist for vessel graphs. For instance, intermediate bulges in one graph can splinter edges into multiple smaller edges, preventing a 1-to-1 correspondence. We therefore relax the strict matching objective in favor of a more flexible path mapping $\mathcal { M } ^ { \prime } : P _ { A } \mapsto P _ { B }$ , with $P = \{ ( e _ { 1 } , \ldots , e _ { n } ) : e \in E \}$ and $n \in  { \mathbb { N } } _ { 0 }$ (see Figure 2). A path is a concatenation of multiple edges with a single start and single endpoint, where each edge may only appear once. The arc length of a path is defined as the cumulative Euclidean distance between neighboring skeleton pixels.

Edge selection and path construction To find the optimal mapping between two graphs, we initially define a sparse cost matrix between paths in graph A and graph B. To avoid computing the match afinity for all possible $2 ^ { | E _ { A } | } \times \mathsf { \bar { 2 } } ^ { | E _ { B } | }$ path combinations, we only consider the most likely candidates. One of the most important factors for similarity is spatial distance. We therefore align both graphs by computing the global afine transformation that maps each pixel coordinate in graph B to the respective coordinate in graph A.

![](images/01e02140d9281d82f6bb171627a2d04d6b9dec0205001693746e922e1bbe1e70.jpg)

Fig. 2. Four-step graph matching process to establish a 1-to-1 pairing between edges.  
![](images/581c2cf8ab95cadf0c1717c06443050facaa389d985c87cfd294b88af83f60c7.jpg)  
Fig. 3. The six metrics used to calculate the match cost between candidate paths.

To build the candidate set for a given edge $e _ { A }$ in graph $\mathrm { A } ,$ we find all edges in graph B that are within a given search distance of $e _ { A }$ . We generate all possible paths by concatenating edges of the selected subset. Depth-first path exploration stops if we encounter unrealistic angles or radius changes between edges. We then analogously generate all candidate paths in graph A by finding all possible paths that include $e _ { A }$ and consist of edges that are within search distance.

Pairing and score computation We compute a match cost $s _ { c }$ for each candidate pairing $c = ( p _ { A } , p _ { B } )$ as the weighted sum of the following metrics (see Figure 3):

1. Distance: Euclidean distance between the path centroids.

2. Rotation: Acute rotation angle $| \phi | \in [ 0 , 9 0 ^ { \circ } ]$ to align path B with path A.

3. Shape: Pairwise Euclidean distance between the K points of the resampled paths that were aligned using the found centroids and rotation angles.

4. Area: Defined by $a _ { p } = \mathrm { a r c l e n g t h } _ { p } \times r _ { p }$

5. Endpoint: We compare the start and endpoints of two paths by computing the earth moving distance between the nodes’ angular densities $\rho ( n , \phi )$ . The density is given by counting non-zero mask pixels in a ring around the node with outer radius $r _ { \mathrm { o u t } } = 2 \cdot r _ { \mathrm { i n } } = 2 \cdot r _ { n }$ and dividing by the total pixel count.

6. Intermediate junction penalty: The sum of penalties for each node within a path, where each protruding edge adds to a node’s respective penalty proportional to its radius and length.

Path matching and decomposition After computing the match cost for each candidate pair $c \in C .$ , we define the following Boolean decision variables: $x _ { c }$ signals whether a candidate is selected, while $\begin{array} { r } { m _ { e } ^ { G } = \sum _ { c \in C : e \in E _ { c } ^ { G } } x _ { c } } \end{array}$ with $G \in \{ A , B \}$ signals whether a given edge in graph A or B is matched. Here $E _ { c } ^ { G }$ denotes the set of edges that are part of the candidate c in graph G. We then ensure at most one candidate for a given edge may be selected by enforcing $\textstyle \sum _ { c \in C : e \in E _ { c } } x _ { c } \leq 1$ Finally, we define the penalty for an edge e remaining unmatched by a linear function $\lambda a _ { e } + \epsilon$ of its area $a _ { e }$ . The matching objective is then given by

$$
\operatorname* { m i n } \underbrace { \sum _ { c \in { \cal C } } s _ { c } x _ { c } } _ { \mathrm { m a t c h ~ c o s t } } + \underbrace { \lambda \left( \sum _ { e \in { \cal E } _ { A } } a _ { e } \big ( 1 - m _ { e } ^ { A } \big ) + \sum _ { e \in { \cal E } _ { B } } a _ { e } \big ( 1 - m _ { e } ^ { B } \big ) \right) + \epsilon } _ { \mathrm { u n m a t c h e d ~ p e n a l t y } } .\tag{2}
$$

We solve this binary integer linear program using PuLP’s time-constrained COIN-OR Branch and Cut (CBC) solver [10].

Decomposition into edges Finally, the resulting path matches must be decomposed into 1-to-1 matches of individual edges. For this, we walk along both paths simultaneously and whenever we encounter a junction in one path, we insert a degree-2 node in the other path. These degree-2 nodes manually fragment edges to achieve direct one-to-one correspondence.

## 2.4 Joint refinement of matched graphs

When performing graph refinement independently on adjacent longitudinal graphs, small structures may be removed in one graph because they fall below the bulge removal threshold, even though the structure is clearly present at other time points. To improve matching correspondence across time points, we therefore propose to perform the graph refinement steps described in Section 2.2 using information from other graphs.

Specifically, we first perform 1) basic graph extraction separately, then 2) conduct the matching phase on these raw graphs, and finally perform 3.1) bulge removal and 3.2) junction merging using joint information. A bulge edge in graph A is removed if and only if both its bulge size and that of its matched counterpart in graph B are below the bulge removal threshold. Unmatched edges are treated as before, without additional constraints. Similarly, we only collapse an H-bridge edge in one graph if the matched edge in the other graph also satisfies the removal criterion. By coupling refinement decisions across time points, this strategy suppresses artifacts caused by imperfect segmentation errors while preserving anatomically consistent vessel structures.

## 3 Experiments and Results

Longitudinal OCTA dataset For our experiments, we use an in-house dataset of 160 retinal OCTA images. Two repeated scans of 80 healthy eyes were acquired on the same day. For each scan, we extract $1 . 5 \times 1 . 5 \ : \mathrm { m m ^ { 2 } }$ large en-face projections of the inner retinal vasculature. We obtain a detailed 608 ×608 pixel segmentation mask of all images using a public tool for OCTA image segmentation [16]. As the vascular structure of the retina is highly stable over this short time interval, we expect only minor structural changes due to varying light exposure and the circadian cycle. This idealized setting allows us to isolate the efect of small artifacts inherent to the graph construction pipeline.

![](images/6946bbe1756a0a99b60dbcaee69c1af8a53ec815bc8b0398ce73fbb72840720e.jpg)  
Fig. 4. Quantitative comparison of the introduced graph construction methods. Left: Graph extraction without refinement results in graphs with spurious edges that do not reflect the true underlying physiology. Graph refinement in our pipeline and established baselines reduces the number of spurious edges. Right: Joint refinement results in a higher matched area compared to separate refinement, while exhibiting a similar or better edge matching quality than separate refinement or the Moriconi et al. baseline.

Graph refinement to reduce spurious edges We compare the three introduced graph construction strategies: i) graph extraction without refinement, ii) extraction with separate refinement, and iii) extraction with joint refinement. Additionally, we include two established graph extraction tools, Voreen [20] and VesselVio [4] for comparison. First, we demonstrate the benefit of graph refinement when processing complex vessel graphs. Small segmentation variations along vessel boundaries can result in spurious bulges. Unrefined graphs, on average, contain about 1,300 edges, the majority of which are very small segments (see Figures 4 and 5). Our bulge removal and junction merging refinement reduces this fragmentation and yields a more coherent graph topology. We find that our graph extraction strategy yields similar graphs as established baseline methods with regard to number of edges and average vessel size.

Benefit of our joint graph refinement However, refining longitudinal scans separately can cause the resulting graphs to diverge topologically, ultimately degrading matching performance. To quantify this efect, we compare matched area and matching quality on separately refined graphs using both our method and the Moriconi et al. baseline [22], and contrast these results with our proposed joint refinement approach. We define matching quality $Q = 1 { - } \mathrm { s M A P E } ( i , j )$ , with sM $\begin{array} { r } { \mathrm { A P E } ( i , j ) = \frac { 1 } { | \mathcal { M } | } \sum _ { ( i , j ) \in \mathcal { M } } \frac { | b _ { i } - b _ { j } | } { | b _ { i } | + | b _ { j } } } \end{array}$ <sub>|</sub> [9], as the mean agreement of vessel-specific biomarkers b (arclength, radius, area, and straightness) of all matched vessels i and $j .$ . Thus, biomarker agreement guards against trivially increasing coverage though random pairs. We tune all methods on a manually annotated graph pair.

![](images/ca147a9edbb48d15d7367d3b95e89e052fb58808bc7f382ca9f084d31972a484.jpg)  
Fig. 5. Representative example of matching quality for the three diferent graph refinement strategies. No refinement fragments edges due to spurious branches (upper arrow), while refining graphs separately causes poor matching quality, indicated by an increased number of unmatched or wrongly paired edges (lower arrow). In contrast, joint refinement yields convincing edge-level matches.

Separate refinement decreases the total area of the graph that can be successfully matched (see Figure 4 and Figure 5). In contrast, performing early matching followed by joint refinement increases the total matched area, while maintaining stable match quality. The baseline produces a larger number of matches, however, the biomarker correspondence between matched vessels is substantially lower.

## 4 Discussion

Longitudinal angiographic imaging combined with vessel graphs has the potential to detect temporal changes at the level of individual blood vessels. However, this study has shown for the first time that current strategies for graph extraction and refinement are highly sensitive to small variations in input images. These artifacts severely inhibit the ability to successfully match sequential vessel graphs of the same subject, even if a clear correspondence between the vasculature exists. To address this problem, we have introduced a strategy that matches graphs before joint refinement. In experiments with complex retinal vessel graphs, we have demonstrated that this strategy results in a higher matched area without graph fragmentation compared to separate or no refinement, respectively.

These results were obtained in a carefully crafted setting with a specifically developed graph extraction and matching algorithm. Nonetheless, we postulate that our strategy of early matching before refinement is conceptually compatible with other methods, such as those that construct graphs by predicting links between a candidate set of nodes [27,30] or directly extract the graph from the image [3,28]. Similarly, our method can be applied to three-dimensional images or to the simultaneous extraction of more than two vessel graphs. As such, our work has the potential to support the extraction of advanced graph-based biomarkers from longitudinal angiographic images.

Acknowledgments. This work was partially funded by ERC Grant Deep4MI (Grant No. 884622). Martin J. Menten is funded by the German Research Foundation under projects 528297171 and 532139938.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. AI-READI Consortium: AI-READI: rethinking AI data collection, preparation and sharing in diabetes research and beyond. Nature metabolism 6(12), 2210–2212 (2024)

2. Arsalan, M., Haider, A., Lee, Y.W., Park, K.R.: Detecting retinal vasculature as a key biomarker for deep learning-based intelligent screening and analysis of diabetic and hypertensive retinopathy. Expert Systems with Applications 200, 117009 (2022)

3. Berger, A.H., Lux, L., Shit, S., Ezhov, I., Kaissis, G., Menten, M.J., Rueckert, D., Paetzold, J.C.: Cross-domain and cross-dimension learning for image-to-graph transformers. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 64–74. IEEE (2025)

4. Bumgarner, J.R., Nelson, R.J.: Open-source analysis and visualization of segmented vasculature datasets with vesselvio. Cell Reports Methods 2(4), 100189 (2022). https://doi.org/10.1016/j.crmeth.2022.100189

5. Drees, D., Scherzinger, A., Hägerling, R., Kiefer, F., Jiang, X.: Scalable robust graph and feature extraction for arbitrary vessel networks in large volumetric datasets. BMC bioinformatics 22(1), 346 (2021)

6. Fang, H., Zhu, J., Ai, D., Huang, Y., Jiang, Y., Song, H., Wang, Y., Yang, J.: Greedy soft matching for vascular tracking of coronary angiographic image sequences. IEEE Transactions on Circuits and Systems for Video Technology 30(5), 1466–1480 (2019)

7. Feuillâtre, H., Nunes, J.C., Toumoulin, C.: An improved graph matching algorithm for the spatio-temporal matching of a coronary artery 3d tree sequence. IRBM 36(6), 329–334 (2015)

8. Fhima, J., Eijgen, J.V., Stalmans, I., Men, Y., Freiman, M., Behar, J.A.: Pvbm: A python vasculature biomarker toolbox based on retinal blood vessel segmentation. In: European Conference on Computer Vision. pp. 296–312. Springer (2022)

9. Flores, B.E.: A pragmatic view of accuracy measurement in forecasting. Omega 14(2), 93–98 (1986). https://doi.org/10.1016/0305-0483(86)90013-7

10. Forrest, J., Ralphs, T., Vigerske, S., Santos, H.G., Forrest, J., Hafer, L., Kristjansson, B., jpfasano, EdwinStraver, Jan-Willem, Lubin, M., rlougee, a andre, jpgoncal1, Brito, S., h-i gassmann, Cristina, Saltzman, M., tosttost, Pitrus, B., MATSUSHIMA, F., Vossler, P., Ron @ SWGY, to-st: coin-or/Cbc: Release releases/2.10.12 (Aug 2024). https://doi.org/10.5281/zenodo.13347261

11. Hartung, M.P., Grist, T.M., François, C.J.: Magnetic resonance angiography: current status and future directions. Journal of Cardiovascular Magnetic Resonance 13(1), 19 (2011)

12. Ikram, M.K., Cheung, C.Y., Lorenzi, M., Klein, R., Jones, T.L., Wong, T.Y., et al.: Retinal vascular caliber as a biomarker for diabetes microvascular complications. Diabetes care 36(3), 750 (2013)

13. Kim, Y.S., Hwang, S., Kim, M.H., Kwon, B., Song, Y., Lee, K., Lee, D.H.: Longitudinal MR angiographic evaluation of circle of Willis morphologic remodeling and induced aneurysms in Hashimoto rat cerebral aneurysm model. Scientific Reports (2026)

14. Kirbas, C., Quek, F.K.: Vessel extraction techniques and algorithms: a survey. In: Third IEEE Symposium on Bioinformatics and Bioengineering, 2003. Proceedings. pp. 238–245. IEEE (2003)

15. Kong, T.Y., Rosenfeld, A.: Digital topology: Introduction and survey. Computer Vision, Graphics, and Image Processing 48(3), 357–393 (1989)

16. Kreitner, L., Paetzold, J.C., Rauch, N., Chen, C., Hagag, A.M., Fayed, A.E., Sivaprasad, S., Rausch, S., Weichsel, J., Menze, B.H., et al.: Synthetic optical coherence tomography angiographs for detailed retinal vessel segmentation without human annotations. IEEE Transactions on Medical Imaging 43(6), 2061–2073 (2024)

17. Lee, T., Kashyap, R., Chu, C.: Building skeleton models via 3-d medial surface axis thinning algorithms. CVGIP: Graphical Models and Image Processing 56(6), 462–478 (1994). https://doi.org/10.1006/cgip.1994.1042

18. Li, R., Hui, Y., Zhang, X., Zhang, S., Lv, B., Ni, Y., Li, X., Liang, X., Yang, L., Lv, H., et al.: Ocular biomarkers of cognitive decline based on deep-learning retinal vessel segmentation. BMC geriatrics 24(1), 28 (2024)

19. Li, R., Huang, Y.J., Chen, H., Liu, X., Yu, Y., Qian, D., Wang, L.: 3D graphconnectivity constrained network for hepatic vessel segmentation. IEEE Journal of Biomedical and Health Informatics 26(3), 1251–1262 (2021)

20. Meyer-Spradow, J., Ropinski, T., Mensmann, J., Hinrichs, K.: Voreen: A rapidprototyping environment for ray-casting-based volume visualizations. IEEE Computer Graphics and Applications 29(6), 6–13 (2009). https://doi.org/10.1109/MCG. 2009.130

21. Moccia, S., De Momi, E., El Hadji, S., Mattos, L.S.: Blood vessel segmentation algorithms—review of methods, datasets and evaluation metrics. Computer methods and programs in biomedicine 158, 71–91 (2018)

22. Moriconi, S., Zuluaga, M.A., Jäger, H.R., Nachev, P., Ourselin, S., Cardoso, M.J.: Elastic registration of geodesic vascular graphs. In: MICCAI (2018)

23. Müller, T.T., Starck, S., Dima, A., Wunderlich, S., Bintsi, K.M., Zaripova, K., Braren, R., Rueckert, D., Kazi, A., Kaissis, G.: A survey on graph construction for geometric deep learning in medicine: Methods and recommendations. Transactions on Machine Learning Research (2024)

24. Oglat, A.A., Matjafri, M., Suardi, N., Oqlat, M.A., Abdelrahman, M.A., Oqlat, A.A.: A review of medical doppler ultrasonography of blood flow in general and especially in common carotid artery. Journal of medical ultrasound 26(1), 3–13 (2018)

25. Paetzold, J.C., McGinnis, J., Shit, S., Ezhov, I., Büschl, P., Prabhakar, C., Todorov, M.I., Sekuboyina, A., Kaissis, G., Ertürk, A., et al.: Whole brain vessel graphs: a dataset and benchmark for graph learning and neuroscience (vesselgraph). arXiv preprint arXiv:2108.13233 (2021)

26. Riedel, E.O., de la Rosa, E., Baran, T.A., Petzsche, M.H., Baazaoui, H., Yang, K., Musio, F.A., Huang, H., Robben, D., Seia, J.O., et al.: ISLES’24–a real-world longitudinal multimodal stroke dataset. arXiv preprint arXiv:2408.11142 (2024)

27. Shin, S.Y., Lee, S., Yun, I.D., Lee, K.M.: Deep vessel segmentation by learning graphical connectivity. Medical image analysis 58, 101556 (2019)

28. Shit, S., Koner, R., Wittmann, B., Paetzold, J., Ezhov, I., Li, H., Pan, J., Sharifzadeh, S., Kaissis, G., Tresp, V., et al.: Relationformer: A unified framework for imageto-graph generation. In: European conference on computer vision. pp. 422–439. Springer (2022)

29. Spaide, R.F., Fujimoto, J.G., Waheed, N.K., Sadda, S.R., Staurenghi, G.: Optical coherence tomography angiography. Progress in retinal and eye research 64, 1–55 (2018)

30. Yu, H., Zhao, J., Zhang, L.: Vessel segmentation via link prediction of graph neural networks. In: International workshop on multiscale multimodal medical imaging. pp. 34–43. Springer (2022)