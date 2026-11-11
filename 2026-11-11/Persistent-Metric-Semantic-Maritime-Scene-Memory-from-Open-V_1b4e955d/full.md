# Persistent Metric-Semantic Maritime Scene Memory from Open-Vocabulary 3D Scene Graphs

Kim Alexander Christensen<sup>1</sup>, Dina Zimmermann<sup>1</sup>, Martin Steinert<sup>1,†</sup> and Eleni Kelasidi<sup>1,†</sup>

Abstract— Three-dimensional scene graphs (3DSGs) can provide robots with persistent metric-semantic representations for spatial reasoning, but the existing methods have primarily been developed for indoor and terrestrial environments. In this paper we investigate how a representative outdoor openvocabulary method transfers to maritime environments and which adaptations are needed. Our experiments show that zeroshot representative outdoor 3DSG descriptions agree with our labeled ground truth for a majority of geometry-supported objects on our data, but the resulting map exhibits considerable physical-entity fragmentation, and its terrestrial higher-level assumptions don’t capture maritime spatial organization needs. We therefore keep the underlying mapping method and postprocess its accumulated evidence. We show a metric-semantic consolidation method followed by geometry-grounded multiview vision-language model (VLM) reasoning that reconciles remaining fragments into physical entities and groups entities through observations into local places. In parallel, we construct maritime macro regions using map priors and onboard water and structure observations, and associate them with the local places. The resulting representation reduces physical-entity fragmentation beyond both the baseline and our geometrysemantic consolidation component alone, supports correspondence of fixed local places across repeated maritime traversals, and improves contextual retrieval on our reviewed queries over tested baselines. These results show how an outdoor openvocabulary map can be adapted into persistent maritime scene memory with our methods for object/place identity and spatial abstractions.

## I. INTRODUCTION

Autonomous surface vessels (ASVs) are being developed for transportation, inspection, monitoring, and other repeated operations in harbors, inland waterways, and coastal areas [1]–[3]. Such operations require more than just obstacle tracking and detection in the moment. A vessel revisiting the same area should retain a memory of persistent objects, distinguish them from temporary occupants, and reason about semantic observations in their spatial context. This motivates a metric-semantic scene memory of the maritime environment.

Three-dimensional scene graphs (3DSGs) can provide such representations by connecting metric geometry with semantic entities, relations, and higher-level spatial abstractions [4]–[7]. Recent open-vocabulary systems extend this capability beyond the previously fixed semantic taxonomies [8]–[10]. However, 3DSG hierarchies are strongly domain dependent. Indoor systems are concerned with rooms and floors, while outdoor systems use abstractions such as roads, lanes, terrain, or task-dependent regions [11]–[14]. An ASV instead operates through large scale open areas like canals, harbors and open water. Existing outdoor openvocabulary 3DSGs do not establish which parts of their representation transfer to maritime environments, and which parts require domain-specific adaptations.

![](images/9fa54fc9d273a4db3d9ca3ef1f468c242bfa752d54d7aec083503b7ca6f69616.jpg)  
Fig. 1: Overview of the maritime scene-memory pipeline. It first accumulates persistent map fragments and their observation histories, followed by a conservative geometricsemantic consolidation. Selected geometrically grounded multi-view observations are then processed by a visionlanguage model (VLM) at two abstraction levels: physicalentity reconciliation decides which fragments belong to the same object, while place formation groups persistent entities and observations into meaningful local places. In parallel, coarse maritime regions are derived from geographic topology and onboard water/structure evidence. Local places are associated with these regions and corresponding places are related across repeated traversals, forming a persistent scene memory for contextual retrieval.

We study this transfer using OpenGraph [10], a representative hierarchical open-vocabulary RGB–LiDAR 3DSG designed for large-scale outdoor mapping, as a baseline. We find that for discrete maritime objects with good geometric map support, accumulated OpenGraph descriptions agree with the ground-truth class in 99.1% of cases, while extended and partially hidden waterfront structures are less reliable. Despite this transfer, the resulting mapped object representation shows considerable fragmentation, while its terrestrial higher-level organization assumptions does not provide the spatial structure required for maritime scene understanding. We therefore use the released perception and incremental mapping substrate while contributing a maritime-adapted method for fragment consolidation followed by grounded multi-view vision-language reasoning at the physical-entity and local-place levels. Higher level maritime macro regions are constructed separately from geographic topology and onboard water/structure evidence. Our proposed method is summarized in Fig. 1.

The complete adaptation reduces physical-entity fragmentation while preserving vessel recovery, reducing nodes per recovered vessel from 1.957 to 1.298 and fragmented vessels from 25 to 11 in our evaluation. We show that independently constructed local places correspond across traversals despite differences in reconstruction and different object occupants, with 12 reciprocal correspondences across ten evaluated geographic contexts. The resulting contextual representation also improves paired-survey retrieval F1 from 0.378 to 0.810 compared to a local metric baseline.

Our contributions are as follows:

• A layer-wise maritime transfer evaluation of an outdoor open-vocabulary 3DSG. We show that semantic descriptions transfer well for geometry-supported discrete objects, while mapped physical-entity consistency and higher-level spatial organization require adaptation.

• A grounded maritime scene-memory construction. We keep the underlying open-vocabulary perception and incremental mapping pipeline, but adapt its accumulated representation through maritime-specific consolidation and grounded multi-view vision-language model (VLM) reasoning to form physical entities and local places. Higher-level maritime context is constructed separately by combining geographic topology with onboard water and structure observations.

• An evaluation of persistence and contextual use across repeated maritime surveys. We evaluate physical-entity consistency, cross-traversal local-place correspondence, macro-region repeatability, and contextual retrieval compared to simpler spatial baselines.

## II. RELATED WORK

3DSGs combine metric grounding with semantic entities, relations, and/or hierarchical spatial abstractions [4]–[7]. Further extensions include methods for task dependent abstractions, dynamic environments, and long-term memory [15]– [17]. In recent years, foundation models have enabled open vocabulary objects and relations [8], [9], [18], where OpenGraph is a relevant example as a hierarchical openvocabulary camera-LiDAR 3DSG designed for large-scale outdoor environments [10]. Incremental 3DSGs relevant for robotics typically lift frame-level observations into 3D and then associate them across viewpoints using geometric and semantic cues, which can yield duplicated, fragmented, merged, or missing object nodes because of partial visibility, ambiguous boundaries and geometry [7], [8], [10], [19]. These problems are often exacerbated outdoors [20]. To com bat this, ThinkGraphs selects geometrically and semantically plausible duplicate tracks, finds informative views, and uses a VLM to verify merges [21]. DAAAM [17] similarly performs later semantic annotation, fragment reconciliation, and hierarchical region construction, while relationship-aware 3DSGs prompt VLMs for hierarchical relationships [22].

Higher-level 3DSG graph structure is environment dependent, with indoor systems typically organizing objects through places, rooms, floors and buildings [5]–[7], [9], while outdoor systems need different abstractions [11], [12], [19]. Car-based approaches such as OpenGraph, CURB-SG, and CURB-OSG take advantage of trajectories, roads, lanes, intersections and curbs to form their graph [10], [11], [23]. Other approaches organize outdoor spaces using semantic composition [12], terrain and traversability [13], [24], or multiscale object parts [25]. The high-level abstractions can also depend on the task at hand [16], prior knowledge [14], abstractions derived from large language models (LLMs) or VLMs [26], [27], or specific missions [28]. No single outdoor hierarchy is universal, and existing formulations are largely organized around terrestrial cues like roads and terrain or task priors.

Maritime surface robotics already has domain-specific methods and benchmarks on visual obstacle and water perception [29], [30], while SGK-Net [31] applies graphstructured reasoning to maritime 2D navigational images. Vision-language models have also been used in maritime autonomy for open vocabulary scene understanding [32]. These methods establish maritime semantic and relational perception, but don’t construct a persistent semantic 3DSG. LiDAR SLAM has also been demonstrated for maritime mapping and localization in coastal areas [33], [34]. Further, there is existing work in persistent geographic structure representing inland waterways through semantic world models based on chart-derived static objects and runtime context [35], as well as Oceanscape which constructs coastal navigation graphs from geospatial information [36]. Existing near-shore mapping works also combines onboard perception with prior map geometry [37], and coastline geometry has been used as a structural constraint for ASV localization [38], the commonality is that map priors are used together with onboard observations, rather than reconstructing large-scale maritime environments from sensing alone. Many terrestrial surfaces can be represented directly by LiDAR or terrain semantics. Water provides negligible LiDAR surface returns [39], which is why visual water evidence can provide additional useful spatial information for ASV operations [29], [30]. In our work, we represent large scale geographic information with map topology, while onboard water/structure observations ground the currently observed scene.

Despite existing work in 3DSGs and maritime autonomy, persistent maritime 3DSGs that organize physical entities, local places, and geographic context across repeated surveys remain largely unexplored.

## III. MARITIME SCENE-MEMORY ADAPTATION

As will be shown, direct transfer of OpenGraph leaves two main representation limitations: fragmented physical identities and no appropriate higher-level maritime spatial organization. We therefore keep the released perception and incremental mapping substrate while adapting the representation at the levels where transfer is insufficient. Our conservative consolidation and grounded multi-view reasoning methods predict physical entities and local places, while geographic topology together with onboard water and structure observations independently provides region-level maritime context. Fixed local place infrastructure is subsequently used to relate corresponding places across repeated traversals.

## A. Problem Formulation

Given posed and calibrated RGB–LiDAR observations from traversal t, an outdoor open-vocabulary mapper produces a set of persistent map fragments $\mathcal { F } ^ { t }$ . Each fragment contains reconstructed geometry, semantic attributes, and its observation history, but a fragment does not necessarily correspond exactly with a physical object.

Our objective is to construct a maritime scene representation with increasingly abstract spatial entities,

$$
\mathcal { F } ^ { t }  \mathcal { E } ^ { t }  \mathcal { P } ^ { t } ,\tag{1}
$$

where $\mathcal { E } ^ { t }$ denotes physical entities and $\mathcal { P } ^ { t }$ local places. In parallel, maritime macro regions $\mathcal { R } ^ { t }$ are constructed in the world frame from map topology and traversal-dependent water and structure evidence. Local places are then spatially associated with these regions. Uncertain fragments or entities can remain unassigned.

For repeated surveys, a place is considered persistent when it is supported by the same underlying fixed infrastructure, even if only part of that structure is observed and its movable occupants differ between traversals. Thus, stable infrastructure defines place identity, while vessels and other movable objects describe traversal-dependent scene state. We focus on static structures and objects that are stationary during a traversal, including moored vessels. Dynamic-object reconstruction and tracking are outside the scope of this work.

## B. Outdoor Mapping and Conservative Consolidation

We use the released OpenGraph [10] perception and mapping pipeline through dataset-specific adapters. Openvocabulary image observations are lifted using LiDAR and incrementally associated into persistent fragments, retaining their geometry, semantic attributes, and observation histories. Details can be found in [10]. We keep the corresponding OpenGraph configuration fixed within each comparison.

The first adaptation is a conservative consolidation stage C050 to remove clear duplicate fragments before higher-level reasoning. For a smaller source point cloud X and candidate destination point cloud Y , define directed geometric support

$$
o _ { d } ( X , Y ) = { \frac { | \{ x \in X : \operatorname* { m i n } _ { y \in Y } \| x - y \| \leq d \} | } { | X | } } .\tag{2}
$$

where d is the Euclidean support tolerance.

A merge is considered when the fragments have a 3D bounding-box gap of at most 0.5 m, their semantic-feature cosine similarity is at least 0.8, $o _ { 0 . 5 } ( X , Y ) ~ \geq ~ 0 . 7$ and $o _ { 0 . 5 } ( Y , X ) \ge 0 . 1$ . For co-observed fragments, the sourcesupport requirement is increased to 0.9. We additionally reject merges that expand either horizontal extent beyond 1.2 times the larger input extent plus 0.25 m. Accepted merges subsequently use the released OpenGraph fusion operation while preserving fragment lineage for the later multi-view reasoning experiments.

## C. Grounded Multi-View Physical-Entity Reasoning

The previous stage removes clear duplicates, but one physical object can still be represented by several fragments. We therefore post-process the accumulated map by having a VLM reason over selected observations from its history. Candidate fragment groups contain a maximum of 18 elements, with a sampled support gap of at most 8 m and center separation of at most 45 m. The candidate geometry is projected back into recorded RGB observations, and views are kept when their mapped support is geometrically visible and consistent with the current LiDAR returns. Numbered overlays identify the candidate fragments in each view.

For physical-entity reconciliation, the VLM (GPT-6 Astra [40]) receives three complementary views together with geometry diagnostics for all candidate fragments. It first proposes which fragments belong to the same physical object, followed by a contrastive verification using the same evidence and proposed group. The verification considers whether the fragments instead represent separate physical objects. Only supported and geometrically consistent groups are joined into physical entities. Each accepted group forms a parent entity while retaining its constituent fragment IDs and lineage. Unsupported or conflicting groups remain separate. The essence of this method is that the VLM reasons over recorded observations geometrically grounded to candidate map fragments rather than over an RGB image or semantic descriptions alone.

## D. Grounded Local-Place Reasoning

At the next abstraction level, place reasoning uses broader multi-view context to determine which entities and observations then form one persistent local setting. The candidates are shown in overlapping spatial windows rather than proposed place boundaries. The VLM is asked which fixed setting remains when current vessels and activity are removed, which leads to stable infrastructure anchors giving the place an identity while movable objects characterize its current state but do not connect place proposals in isolation. This reasoning is performed independently for each traversal.

The place reasoning starts from six views, supplemented when necessary with targeted RGB image additions for qualifying structural anchors not represented in the initial selection. The supported proposals are reconciled through shared, uncontested stable anchors, meaning that the same mapped infrastructure elements support the proposals being joined. Conflicting assignments continue to stay separate. A place does not need to consist only of already reconciled physical entities. Instead, locally grounded unresolved observations can also contribute to its support.

## E. Maritime Macro-Regions

At a higher scale, we independently construct maritime regions in $\mathcal { R } ^ { t }$ from geographic map priors and onboard water and structure observations. Since LiDAR returns are negligible on water, we project camera-observed water masks from the released image perception onto a fixed nominal horizontal water plane using calibrated cameras and supplied poses to provide positive water evidence in the world frame. LiDAR observations provide complementary structure and obstacle evidence.

Let $W _ { o }$ and $\boldsymbol { W } _ { c } ^ { t }$ denote map-prior and camera-observed water, and $B _ { o }$ and $B _ { s } ^ { t }$ mapped and observed structural barriers. The represented water domain is

$$
W ^ { t } = ( W _ { o } \cup W _ { c } ^ { t } ) \setminus ( B _ { o } \cup B _ { s } ^ { t } ) .\tag{3}
$$

The onboard evidence is accumulated across observations and represented on a 2 m spatial grid, with at most one water vote and one structure vote per keyframe and cell. For accumulated water and structure counts w and s, cameraobserved water requires $w \geq 3$ and $w / ( w + s ) \geq 0 . 8 .$ . Cells with $s \geq 3$ outside the camera-observed water form structural barriers.

Mapped land remain a hard constraint on the large-scale topology, while onboard evidence grounds the water and structure observed during traversal t. We estimate local water widths at 8 m-spaced probes using 36 directions out to 300 m. Candidate cuts require at least 75% supported water and barrier contact within 8 m of their endpoints. Widening candidates additionally require a width ratio of at least 1.8 and a difference of at least 30 m. Map-mouth separators are prioritized, with 250 m separator suppression, 2 m separator buffers and 4 m connectivity closing. Components with at least $2 0 { , } 0 0 0 ~ \mathrm { m } ^ { 2 }$ of represented water form macro regions.

Place–region associations use the nearest assigned region cell within 10 m of fixed support, retaining weak or multiple associations and the unassigned fraction.

## F. Cross-Traversal Place Correspondence

Next, we want to determine if independently constructed places represent the same persistent setting across repeated completed surveys. We measure correspondence based on shared fixed physical support. Movable occupants, semantic descriptions, and macro-region IDs provide no positive correspondence evidence. Fixed place support is represented using 0.5 m voxelized geometry from the stable anchors, excluding disputed memberships and movable objects.

At 3 m distance, candidate correspondences require at least 50 close support voxels on each side and at least 20% directed coverage in one direction. Candidates are ranked using the harmonic mean of directed coverage. A reciprocal correspondence additionally requires at least 20% coverage in both directions, mutual best rank, and a score margin of at least 0.10 over the next candidate on each side. The comparison assumes both traversals are expressed in a common world frame.

## IV. EXPERIMENTAL SETUP

## A. Datasets and Evaluation Protocol

We use two main sources of data. The first is the singletarget MAPPING sequence from the Maritime Urban Tracking dataset (MUT) [41]. The sequence contains synchronized LiDAR, monocular RGB image frames, and RTK GNSSbased INS poses. We process every tenth LiDAR scan, giving 967 selected keyframes.

We manually annotate 222 object footprints, comprising 139 discrete objects (including 92 vessels) and 83 extended structures. We split the map across the canal and get 42 north-bank vessels for development and 50 south-bank vessels for evaluation. Ground-truth identities are never provided to the mapping or reconciliation methods.

The second source of data is the Pohang Canal Dataset [39], where we use repeated traversals P00 and P03. Each traversal covers approximately 7.5 km, with 2,008 and 2,045 accepted 1 Hz keyframes, respectively. The traversals use the same local coordinate system and origin. Entity and place construction use the forward stereo-left camera, four horizontal omni cameras, and the front LiDAR. The macro-region water layer additionally uses the remaining horizontal omni camera. Geographic context is provided by an OpenStreetMap [42] (OSM) prior.

We organize the evaluation around four questions:

RQ1 Which parts of an outdoor open-vocabulary 3DSG transfer directly to maritime scenes and which do not?

RQ2 Can improved consolidation and grounded multi-view VLM reasoning improve physical-entity consistency?

RQ3 Can the adapted representation form meaningful higher-level maritime macro regions and local places that persist across repeated traversals?

RQ4 Does the resulting hierarchy improve contextual retrieval across single and repeated surveys compared with simpler spatial representations?

## B. Evaluation

RQ1. We use the OpenGraph baseline B0 to evaluate what transfers directly to the maritime domain. On the MUT dataset, we associate mapped records with the annotations using the geometric support criterion described in the method and normalize their accumulated text to a fixed simple ontology using an LLM text-only normalizer without access to ground-truth labels. We report geometric coverage and semantic agreement among supported objects, separately for discrete objects and extended structures. We also evaluate physical-entity recovery and fragmentation against the vessel annotations. The higher-level transfer is evaluated qualitatively by whether the existing outdoor representation provides meaningful spatial organization for the maritime scenes considered here.

RQ2. We compare OpenGraph baseline B0, our conservative consolidation C050, and C050+VLM post-processing. A node supports the ground-truth reference vessel when at least 50 entity points lie within 0.25 m of its footprint and either 10% of the node or 200 points satisfy this condition. We report recovery, nodes per recovered vessel, fragmented vessels, and multi-vessel merge evidence, with sensitivity across three support criteria and 0 − 1 m tolerances. Small and uncertain objects were not labeled, so we don’t report whole-scene precision.

RQ3. For macro regions, we compare onboard water/structure, the same evidence with radar, OSM alone, and OSM plus onboard evidence. We measure cross-traversal agreement with ARI on common camera-water cells. We compare against four environmental areas digitized from the dataset authors’ illustration [39] and manually registered using five landmarks. We report the equal-weighted mean IoU over the four one-to-one-matched reference areas on assigned reference-covered water. This provides an approximate reference for geographic plausibility.

Local semantic place persistence is evaluated on ten matched contexts by independently constructing P00 and P03 and reporting reciprocal, partial, unmatched, and conflicting correspondences together with fixed-support overlap. Without exhaustive ground-truth place boundaries, this measures repeatability rather than boundary accuracy.

RQ4. We compare flat entity text, lexical- and suppliedlocation radius retrieval, our proposed hierarchy, and an equivalent flat representation of the same relations. All methods receive the same structured query material, and language parsing is not evaluated. The radius baselines use 50 m neighborhoods around either a text-selected (BM25) anchor or a supplied representative place location. For paired queries, the supplied location is the mean anchor-entity center in P00, carried unchanged into P03. The hierarchy retrieves place members and follows stored place correspondences for paired-survey queries. The primary evaluation contains 20 reviewed single-survey and four paired-survey queries. We additionally evaluate 18 macro-context queries comparing direct region assignment, fixed-radius geometry, and hierarchical place–region associations. These combine reviewed semantics with fixed spatial reference rules. We report mean per-query F1 and exact answer sets separately for the two primary cohorts.

## V. RESULTS

## A. RQ1: Zero-Shot Maritime Transfer

The obtained results show that OpenGraph ends up producing useful open-vocabulary object-level observations of the maritime scene without maritime-specific adaptations. On the MUT dataset, 112/139 discrete objects (80.6%) have geometric support, and accumulated map text agrees with the ground-truth class on 111/112 of these (99.1%). Extended structures are more difficult, with 46/83 (55.4%) supported and 15/46 (32.6%) agreeing with the overall class (these are often partially hidden behind the objects).

TABLE I: Physical-entity evaluation on MUT dataset. Each bank reports recovered vessels, nodes per recovered vessel, and fragmented recovered vessels.
<table><tr><td>Method</td><td>Recovered Nodes / rec. Fragmented</td><td></td><td></td></tr><tr><td rowspan="4">OpenGraph B0 C050</td><td colspan="3">North development (/42)</td></tr><tr><td>36</td><td>2.306</td><td>20</td></tr><tr><td>36</td><td>1.861</td><td>17</td></tr><tr><td>C050 + VLM 36</td><td>1.222</td><td>8</td></tr><tr><td colspan="3">South evaluation (/50)</td></tr><tr><td>OpenGraph B0</td><td>46</td><td>1.957</td><td>25</td></tr><tr><td>C050</td><td>46</td><td>1.413</td><td>15</td></tr><tr><td>C050 + VLM</td><td>47</td><td>1.298</td><td>11</td></tr></table>

However, the persistent object representation is considerably fragmented as seen in the examples in Fig. 1 and 2 and Table I. Checked against the MUT vessel annotations, B0 recovers 36/42 north-bank and 46/50 south-bank vessels under our primary support criterion, but these are represented by 2.306 and 1.957 nodes per recovered vessel, respectively. Of the recovered vessels, 20/36 in the north and 25/46 in the south are represented by several nodes.

The higher-level OpenGraph representation is organized around terrestrial road and lane structure and does not directly provide appropriate spatial organization for the maritime domain. Thus, the object-level observations transfer usefully, while physical-entity consistency and higher-level spatial organization require adaptation. We therefore keep the perception and object-mapping substrate and adapt the physical entity and higher-level representation in the following experiments.

## B. RQ2: Physical-Entity Reconciliation

Table I compares OpenGraph B0, our C050 consolidation, and our multi-view VLM reconciliation, while examples are shown in Fig. 1 and 2. C050 reduces fragmentation substantially while preserving the primary recovery of the baseline. The VLM stage further reduces nodes per recovered vessel from 1.861 to 1.222 on the north bank set and from 1.413 to 1.298 on the south bank set. The number of fragmented vessels decreases from 17 to 8 and from 15 to 11, respectively. Figure 2 illustrates object consolidation and place formation on the MUT north bank.

Relative to B0, the complete method reduces nodes per recovered vessel from 1.957 to 1.298 and fragmented vessels from 25 to 11 on the south bank. Every vessel recovered by C050 remains recovered across all twelve sensitivity settings, and no extra vessel pair appears in the multi-vessel merge diagnostic. At the primary setting, the VLM additionally recovers one southern vessel. This result indicates that selected multi-view scene evidence can resolve much of the remaining physical fragmentation beyond geometric-semantic consolidation without increasing the observed multi-vessel merge evidence. The final representation retains the same 111/112 semantic agreement on the supported discrete objects.

The successful object consolidation is also observed quali tatively on P00 and P03 where multi-camera lifting increases the number of B0 and C050 fragments considerably compared to the mono-camera lifting performed on MUT.

![](images/b55d64645b6bbebe4862b72ae5bf3200dae0eef9088528656d3bb763cc1ea428.jpg)  
Fig. 2: Representative view of the MUT north bank showing (a) B0 fragmentation, (b) C050+VLM objects, and (c) local places.

## C. RQ3: Persistent Maritime Spatial Hierarchy

a) Macro regions: The selected OSM plus onboard representation produces four regions in P00 and five in P03 with cross-traversal ARI 0.965 on common camerawater cells. The regions correspond closely with the four environmental areas identified independently in [39], where a comparison to our method on P00 and P03 is shown in Fig. 3. On assigned reference-covered water, equal-weighted mean IoU over the four one-to-one-matched reference areas is 0.812 for P00 and 0.810 for P03. Region assignments cover approximately 78.1% and 78.0% of the reference domain, respectively. The additional P03 region corresponds to a subdivision not present in the four-region reference.

For comparison, onboard water/structure, the radaraugmented variant, and OSM alone give ARI 0.907, 0.623, and 1.000, respectively. OSM alone repeats by construction.

The combined representation therefore recovers a similar coarse spatial organization across the two traversals while remaining broadly consistent with [39]. The regions are not map-independent, but prior-assisted. The shared OSM topology provides the large-scale structure, while onboard water and structure observations provide traversal-specific observable real evidence.

b) Local places: Local places are constructed independently in P00 and P03 before comparison. Across the ten matched geographical areas, we obtain 13 sample-associated places in P00 and 16 in P03. In total, 12 ordinary reciprocal correspondences are recovered, together with 3 partial alternatives and one type conflict among the partial relations. One P00 place and two P03 places remain unmatched.

![](images/c41dd2529c086c9efd7392bab958206b73e7e414895220e407c2338a37b4f14f.jpg)  
Fig. 3: Macro regions compared. The first is defined by [39] while P00 and P03 are from our method. The regions are largely consistent, with one small extra region in P03.

TABLE II: Contextual retrieval on Pohang.
<table><tr><td colspan="4">Single survey (/20)</td></tr><tr><td>Method</td><td>F1</td><td>Exact</td><td>Paired (/4) F1 Exact</td></tr><tr><td>Flat entity text</td><td>0.250</td><td>4</td><td>0.000 0</td></tr><tr><td>Lexical radius</td><td>0.387</td><td>4</td><td>0.150 0</td></tr><tr><td>Location radius</td><td>0.487</td><td>4</td><td>0.378 0</td></tr><tr><td>Hierarchy</td><td>0.787</td><td>11</td><td>0.810 2</td></tr></table>

The matched places include crossings, waterfronts, canopy-associated areas, among others. Reciprocal fixedsupport overlap ranges from 0.341 to 0.770, reflecting both repeated structures and uncertain place extents. These result support repeatable local settings across traversals, while also showing that place extent remains less stable than the underlying fixed anchors. A persistent place example is shown in Fig. 4, where a place is recognized on both traversals despite P03 being severely blocked by vessels not present in P00.

## D. RQ4: Contextual Retrieval

We finally evaluate if our inferred hierarchy improves retrieval of entities and observations in a spatial context. Table II reports the primary Pohang evaluation. Flat entity text retrieval performs poorly when queries require spatial context. Radius-based retrieval improves, but our hierarchy performs best on both single-survey and paired-survey queries.

An equivalent flat representation of the same relations gives the same answers. On the 16 context-conditioned single-survey queries, supplied-location radius F1 increases from 0.250 at 25 m to 0.579 at 100 m, but remains below the hierarchy. In one query (Fig. 4), for example, the hierarchy retrieves all six reviewed vessel records across the two surveys, despite very different spatial scenes, while the supplied-location radius retrieves three. The quay extends beyond the 50 m neighborhood around its P00 representative location.

Find vessels at P00's northern-port quay and its matching place in P03. Keep survey labels.  
![](images/53b48427dc8bb95b8c941ff4d49cd8834bee4713d3125c5a24baafe55f73ec88.jpg)  
Fig. 4: Place persistence and retrieval at the Northern port quay. The same fixed setting is matched despite different vessel occupancy and reconstructed geometry. Place relations retrieve $6 / 6$ reviewed vessel records across P00 and P03, compared with $3 / 6$ for the 50 m radius baseline centered at the P00 anchor location in both surveys.

The macro-context queries provide a test of the higher levels of the representation. Direct region queries are already handled well by geometric region assignment, with F1 0.967 for both geometry and hierarchy. For place-mediated macro queries, the hierarchy reaches F1 0.873 compared with 0.807 for a 50 m geometric support radius. Increasing the geometric radius to 100 m closes this gap. The hierarchy therefore provides an explicit way to propagate regional context through local-place relations, although sufficiently broad geometric lookup can recover the same information in these cases.

The hierarchy does improve contextual retrieval on the primary reviewed Pohang queries compared with entity-text and local radius-based representations, while the macro-context evaluation shows that some of the same context can be recovered using sufficiently broad geometric neighborhoods.

## VI. DISCUSSION

The results show that good semantics don’t always lead to coherent 3D scene representation. In the maritime domain, large objects, limited viewing angles, and poor visibility can all be contributing factors to the high level of fragmentation observed with OpenGraph. While a lot of the perception and mapping pipeline can be maintained, physical identities and spatial abstractions needs to be adapted.

We have seen that fixed infrastructure can support place correspondence even when reconstructed place extent and vessel occupancy differ considerably, as it can be seen in Fig. 4. Explicit place relations also help retrieval when a setting extends beyond a local metric neighborhood or must be related across traversals. The equivalent-flat control shows that this benefit comes from the inferred contextual relations rather than the graph data structure itself.

The resulting method can provide persistent context for different repeated maritime operations such as inspection, revisit planning and change detection. The current evaluation is limited to two datasets, prior-assisted macro regions and selected repeated-survey contexts, while dynamic, broader and longer-term evaluation remains future work.

## VII. CONCLUSION

Persistent scene memory is a prerequisite for autonomous maritime operations: a robot that repeatedly surveys the same waters should relate new observations to previously observed settings rather than treating each traversal independently. We show how an outdoor open-vocabulary 3DSG can be adapted into such a memory. Using OpenGraph as a representative baseline, our layer-wise transfer evaluation shows that openvocabulary semantics transfer well to geometry-supported discrete maritime objects, while persistent physical identity and higher-level spatial organization do not: the mapped representation remains considerably fragmented, and the terrestrial organization assumptions don’t provide the spatial structure required for maritime scene understanding.

We therefore keep the released perception and incremental mapping substrate and adapt its accumulated representation. Maritime-adapted consolidation and grounded multi-view vision-language reasoning form physical entities and local places, while maritime macro regions are built from geographic topology and onboard water and structure evidence. The complete adaptation substantially reduces physical-entity fragmentation while preserving vessel recovery, yields local places that correspond across independently constructed traversals despite different reconstruction and object occupancy, and improves paired-survey contextual retrieval over a local metric baseline. More broadly, the results show that 3DSGs can require domain adaptation even when the underlying semantic observations transfer well, providing persistent contextual memory for repeated maritime operations.

## ACKNOWLEDGMENT

OpenAI ChatGPT has assisted with brainstorming, critical discussion, and review, while Codex has supported software implementation, debugging, and experimental tooling. The authors performed and directed the research, developed and evaluated the ideas, validated any AI-assisted outputs, and retain full responsibility for the work.

## REFERENCES

[1] E. F. Brekke, E. Eide, B.-O. H. Eriksen, E. F. Wilthil, M. Breivik, E. Skjellaug, Ø. K. Helgesen, A. M. Lekkas, A. B. Martinsen, E. H. Thyri, T. Torben, E. Veitch, O. A. Alsos, and T. A. Johansen, “milliAmpere: An Autonomous Ferry Prototype,” Journal of Physics: Conference Series, vol. 2311, p. 012029, 2022.

[2] M. Blanke, P. N. Hansen, K. Dittmann, T. T. Enevoldsen, D. Dagdilelis, F. E. Schöller, M. K. Plenge-Feidenhans, J. Becktor, D. Papageorgiou, and R. Galeazzi, “GreenHopper: The danish spearhead towards autonomous waterborne mobility,” in Journal ofPhysics: Conference Series, vol. 2867, 2024, p. 012035.

[3] A. Akbar, A. K. Aasen, M. K. Msakni, K. Fagerholt, E. Lindstad, and F. Meisel, “An economic analysis of introducing autonomous ships in a short-sea liner shipping network,” International Transactions in Operational Research, vol. 28, pp. 1740–1764, 2021.

[4] I. Armeni, Z.-Y. He, J. Gwak, A. R. Zamir, M. Fischer, J. Malik, and S. Savarese, “3d scene graph: A structure for unified semantics, 3d space, and camera,” in IEEE/CVF international conference on computer vision, 2019, pp. 5664–5673.

[5] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, “3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans,” Robotics: Science and Systems, 2020.

[6] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y. Chang, J. Shi, A. Gupta, and L. Carlone, “Kimera: From slam to spatial perception with 3d dynamic scene graphs,” The International Journal of Robotics Research, vol. 40, pp. 1510–1546, 2021.

[7] N. Hughes, Y. Chang, and L. Carlone, “Hydra: A real-time spatial perception system for 3d scene graph construction and optimization,” Robotics: Science and Systems, 2022.

[8] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “Conceptgraphs: Open-vocabulary 3D scene graphs for perception and planning,” in IEEE International Conference on Robotics and Automation, 2024, pp. 5021–5028.

[9] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, “Hierarchical open-vocabulary 3D scene graphs for language-grounded robot navigation,” in Robotics: Science and Systems, 2024.

[10] Y. Deng, J. Wang, J. Zhao, X. Tian, G. Chen, Y. Yang, and Y. Yue, “Opengraph: Open-vocabulary hierarchical 3d graph representation in large-scale outdoor environments,” IEEE Robotics and Automation Letters, vol. 9, pp. 8402–8409, 2024.

[11] E. Greve, M. Büchner, N. Vödisch, W. Burgard, and A. Valada, “Collaborative dynamic 3d scene graphs for automated driving,” in IEEE International Conference on Robotics and Automation, 2024, pp. 11 118–11 124.

[12] J. Strader, N. Hughes, W. Chen, A. Speranzon, and L. Carlone, “Indoor and outdoor 3d scene graph generation via language-enabled spatial ontologies,” IEEE Robotics and Automation Letters, vol. 9, pp. 4886– 4893, 2024.

[13] C. R. Samuelson, A. Austin, S. Knoop, B. Romrell, G. R. Slade, T. W. McLain, and J. G. Mangelson, “Terra: Hierarchical terrain-aware 3d scene graph for task-agnostic outdoor mapping,” arXiv preprint arXiv:2509.19579, 2025.

[14] G. Tonetti, L. Kneip, A. Gawel, and M. Hutter, “Prior-SG: Task and prior driven region segmentation for scene graphs in arbitrarilystructured environments,” arXiv preprint arXiv:2606.06170, 2026.

[15] L. Schmid, M. Abate, Y. Chang, and L. Carlone, “Khronos: A unified approach for spatio-temporal metric-semantic slam in dynamic environments,” Robotics: Science and Systems, 2024.

[16] D. Maggio, Y. Chang, N. Hughes, M. Trang, J. D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone, “Clio: Realtime task-driven open-set 3D scene graphs,” IEEE Robotics and Automation Letters, vol. 9, pp. 8921–8928, 2024.

[17] N. Gorlo, L. Schmid, and L. Carlone, “Describe anything anywhere at any moment,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 35 002–35 013.

[18] S. Koch, N. Vaskevicius, M. Colosi, P. Hermosilla, and T. Ropinski, “Open3DSG: Open-vocabulary 3D scene graphs from point clouds with queryable objects and open-set relationships,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 183–14 193.

[19] D. Rotondi, F. Argenziano, S. Koch, N. Hughes, M. Büchner, J. Wald, L. Rosenberger Schmid, D. Nardi, A. Valada, L. Paull, F. Tombari, L. Carlone, and K. O. Arras, “3D scene graphs: Open challenges and future directions,” arXiv preprint arXiv:2606.19383, 2026.

[20] H. Lim, N. Hughes, X. Yu, R. Xu, Y. Chang, J. Shi, R. Talak, and L. Carlone, “Hydra++: Real-time hierarchical 3d scene graph construction with object-level shape estimation,” arXiv preprint arXiv:2607.09455, 2026.

[21] D. Bickici, M. Pabst, S. Mori, and D. Schmalstieg, “Think while you map: Asynchronous vision-language agents for incremental 3D scene graphs,” in European Conference on Computer Vision, 2026.

[22] A. Gassol Puigjaner, A. Zacharia, and K. Alexis, “Relationship-aware hierarchical 3D scene graph,” in IEEE International Conference on Robotics and Automation, 2026.

[23] T. Steinke, M. Büchner, N. Vödisch, and A. Valada, “Collaborative dynamic 3d scene graphs for open-vocabulary urban scene understanding,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025, pp. 6000–6007.

[24] C. R. Samuelson, T. W. McLain, and J. G. Mangelson, “Towards terrain-aware task-driven 3d scene graph generation in outdoor environments,” arXiv preprint arXiv:2506.06562, 2025.

[25] J. Nyffeler, F. Tombari, and D. Barath, “Hierarchical 3d scene graphs construction outdoors,” in IEEE/CVF International Conference on Computer Vision, 2025, pp. 26 817–26 826.

[26] M. Berg, G. Konidaris, and S. Tellex, “Using language to generate state abstractions for long-range planning in outdoor environments,” in International Conference on Robotics and Automation, 2022, pp. 1888–1895.

[27] T. Hao, H. Lan, A. Li, C. Chen, E. Li, J. Yang, Y. Su, P. Lin, and W. Chuang, “OVIP-SG: Open-vocabulary instance-preserving scene graphs for mapping and retrieval of small, fine-grained objects,” arXiv preprint arXiv:2608.17633, 2026.

[28] V. K. Viswanathan, M. A. V. Saucedo, S. G. Satpute, C. Kanellakis, and G. Nikolakopoulos, “xflie: Leveraging actionable hierarchical scene representation for autonomous semantic-aware inspection missions,” IEEE Transactions on Field Robotics, 2026.

[29] B. Bovcon, J. Muhovic, D. Vranac, D. Mozetiˇ c, J. Perš, and M. Kris-ˇ tan, “MODS–a USV-oriented object detection and obstacle segmentation benchmark,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, pp. 13 403–13 418, 2022.

[30] L. Žust, J. Perš, and M. Kristan, “LaRS: A diverse panoptic maritime obstacle detection dataset and benchmark,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 20 304–20 314.

[31] W. Yang, H. Qiu, X. Luo, and S. Xie, “SGK-Net: A novel navigation scene graph generation network,” Sensors, vol. 24, p. 4329, 2024.

[32] K. A. Christensen, A. G. Tufte, A. Gusev, R. Sinha, M. Ganai, O. A. Alsos, M. Pavone, and M. Steinert, “Foundation models on the bridge: Semantic hazard detection and safety maneuvers for maritime autonomy with vision-language models,” Ocean Engineering, vol. 359, p. 124646, 2026.

[33] R. Sawada and K. Hirata, “Mapping and localization for autonomous ship using LiDAR SLAM on the sea,” Journal of Marine Science and Technology, vol. 28, pp. 410–421, 2023.

[34] A. Engström, D. Geiseler, K. Blanch, O. Benderius, and I. García Daza, “A lidar-only SLAM algorithm for marine vessels and autonomous surface vehicles,” IFAC-PapersOnLine, vol. 55, pp. 229– 234, 2022.

[35] S. Van Baelen, G. Peeters, H. Bruyninckx, P. Pilozzi, and P. Slaets, “Dynamic semantic world models and increased situational awareness for highly automated inland waterway transport,” Frontiers in Robotics and AI, vol. 8, p. 739062, 2022.

[36] E. S. Fagerhaug, R. T. Bye, O. L. Osen, and L. I. Hatledal, “Oceanscape: A graph-based framework for autonomous coastal navigation,” Ocean Engineering, vol. 320, p. 120230, 2025.

[37] N. Dalhaug, A. Stahl, R. Mester, and E. F. Brekke, “Near-shore mapping for detection and tracking of vessels,” in International Conference on Information Fusion, 2025, pp. 1–8.

[38] D. R. Benham and J. G. Mangelson, “The coastline as a structural constraint: Harnessing scene geometry for autonomous surface vessel localization,” arXiv preprint arXiv:2608.21276, 2026.

[39] D. Chung, J. Kim, C. Lee, and J. Kim, “Pohang canal dataset: A multimodal maritime dataset for autonomous navigation in restricted waters,” The International Journal of Robotics Research, vol. 42, pp. 1104–1114, 2023.

[40] OpenAI, “GPT-6 Astra,” OpenAI API documentation, accessed: 2026-09-15. [Online]. Available: https://developers.openai.com/api/ docs/models/gpt-6-astra

[41] N. Dalhaug, T. A. Nygård, M. Hinostroza, H. D. Flemmen, E. Martens, A. Stahl, R. Mester, and E. F. Brekke, “Maritime urban tracking dataset in harbor environment,” Scientific Data, vol. 13, p. 818, 2026.

[42] OpenStreetMap contributors, “OpenStreetMap,” data available under the Open Database License. [Online]. Available: https://www. openstreetmap.org