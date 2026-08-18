# REMOTE-SENSING CITY LAYOUT EXTRACTION WITH MLLM

Zigan Zhou

Kai Li

Yupeng Deng

City University of Hong Kong

City University of Hong Kong

Aerospace Information Research Institute

Hong Kong SAR, China

Hong Kong SAR, China

Chinese Academy of Sciences

ziganzhou4-c@my.cityu.edu.hk

School of Electronic, Electrical and

Beijing, China

Communication Engineering

dengyp@aircas.ac.cn

University of Chinese Academy of Sciences

Beijing, China

likai211@mails.ucas.ac.cn

Abstract—Remote-sensing systems usually describe urban content with detection boxes, semantic masks, or vector boundaries. Such outputs locate classes and support image-plane scoring, yet they do not by themselves constitute an executable layout that retains object identities, typed relations, topology, and regeneration rules. Code-as-City instead casts urban-layout extraction from a single top-down image as constrained code generation with a multimodal large language model (MLLM). An image model first produces an aligned five-class semantic layout prior. Three ordered MLLM passes use the image and this prior to recover roads, land-cover regions and relations, and buildings. Deterministic normalization converts the accumulated records into a city graph and a restricted layout program. Executing the program creates a renderable 3D city layout and an orthographic semantic projection over shared geometry. The projection admits pixel-level comparison with remote-sensing masks, while named objects, relations, and editing operations remain available for synchronized regeneration of both views. Evaluated on the 100 scenes of CityLayout-100, the complete framework obtains 41.1% mean intersection-over-union and 48.3% global intersection-overunion. This result provides quantitative evidence that visual observations can be translated into inspectable, editable city code with coupled planar and 3D outputs.[

Index Terms—remote sensing, multimodal large language models, city layout, code generation, semantic projection, 3D city modeling.

## I. INTRODUCTION

Detection and segmentation turn remote-sensing imagery into boxes, masks, or vector boundaries that specify what is present and where it is located. These are effective imageplane interfaces, but the usual outputs do not jointly preserve persistent identity, typed relations, topology, and rules for regeneration. That distinction becomes consequential when image interpretation is expected to feed a maintainable city asset. Structured formats and procedural models require explicit objects and construction logic, neither of which is directly available from one overhead view. This work asks whether a multimodal large language model (MLLM) can organize that visual evidence as an executable city layout without giving up segmentation-style measurement.

Code-as-City maps a near-nadir RGB image to a normalized city graph and a restricted layout program. A controller schedules three dependent passes—roads, land-cover regions and relations, and buildings—and transfers validated records between them. It subsequently executes the generated program and invokes fixed checks or a predefined repair routine when an artifact fails. Every MLLM pass receives a generated semantic layout prior with aligned class and footprint evidence, together with the RGB image for cues about identity, function, relations, appearance, and approximate height. The same named objects and geometry produce an orthographic semantic projection and a renderable 3D layout (Fig. 1); the former is scored in the image plane, while the graph and program remain available for inspection, editing, and joint regeneration.

The work contributes an image-to-city-code task definition in which the primary prediction is an executable layout program rather than a terminal raster or vector. Named objects and typed relations survive execution, allowing planar and 3D outputs to be regenerated from one representation. It also introduces a prior-grounded MLLM workflow: decomposed spatial parsing is guided by a generated semantic layout prior, and validated intermediate records, constrained code, and predefined recovery paths leave a traceable generation record. For evaluation, CityLayout-100 couples standard pixel metrics on the orthographic projection with synchronized executable 3D examples over 100 scenes. The complete framework records 41.1% mean IoU and 48.3% global IoU, while Section IV-D isolates the contribution of the semantic layout prior to spatial grounding.

## II. RELATED WORK

Overhead scenes have been represented at several levels of geometric detail. Detection localizes oriented objects, instance segmentation separates individual masks, and semantic segmentation assigns dense classes [1–4]. IRSAMap carries this progression into vector land-cover objects and topology-aware mapping [5]. These task definitions serve localization and pixel evaluation, but they do not require a regenerable construction with persistent identities and typed relations. Code-as-City preserves image-plane measurement through projection while placing executable structure in the prediction space.

![](images/38ffeeab7e07b3edf182d00d44583f5694777148ba9ec9966c192df34a2c8790.jpg)  
Fig. 1. Code-as-City data flow. Its generated semantic layout prior conditions three dependent MLLM parses; the accumulated records form a normalized graph and restricted program. One execution yields a planar projection and 3D layout from shared objects.

Procedural and structured city representations approach the problem from the asset side. Context-sensitive shape grammars encode hierarchical construction rules for editable architecture [6], while CityJSON stores semantic objects, identifiers, attributes, and geometry in a machine-processable form [7]. In both cases, structure becomes useful once the rules or geometry are available. The present task begins earlier: it must recover the objects and relations needed to instantiate such a layout from one overhead image.

Executable visual reasoning offers a route between perception and structured assets. VISPROG and ViperGPT compose perception modules as inspectable programs [8, 9]. LayoutGPT expresses spatial arrangements in code-like form, whereas 3D-GPT and Holodeck connect language-guided planning to 3D construction [10–12]. Code-as-Room further generates executable indoor scenes from top-down images [13]. Their settings center on general visual reasoning, language-conditioned generation, or indoor layouts. Code-as-City instead addresses outdoor road networks and heterogeneous land cover, using one program for both a segmentationcompatible planar projection and a synchronized 3D scene.

## III. METHOD

## A. Task Contract and Semantic-Prior-Guided Parsing

For a near-nadir RGB image I, the predicted artifact is a constrained program that executes into a typed city graph and a renderer-facing layout. The same named objects and geometry then yield an orthographic semantic projection and an oblique 3D view. Evaluation covers building, road area, sport, vegetation, and water. Within the program, open\_space is a generic renderable ground-surface object whose semantic type distinguishes vegetation, water, or sport; buildings and roads have dedicated constructors. Rendered objects use normalized geometry and typed relations, including road connectivity and spatial adjacency. Because the target is planimetric organization rather than metric 3D recovery from one view, height and appearance attributes are approximate.

An image model derives a canonical five-color semantic layout prior M from I. Both inputs accompany every MLLM call: M supplies aligned category and footprint evidence, while I retains cues for function, relations, roof appearance, and approximate height. Parsing proceeds through three coordinated calls. The road pass recovers centerlines, widths, intersections, and connectivity. Given that road record, the topology pass extracts open-space regions and typed relations. The building pass then reads the accumulated topology to identify individual footprints and repeated groups. Coordinates remain normalized to the original image frame, and each later pass extends the shared object inventory rather than replacing it.

The prompts use M as the principal planimetric constraint. After graph construction, connected components in the prior are used to reconcile the geometry of buildings, road areas, sports surfaces, and water. Building height and appearance fields still come from the RGB image, whereas vegetation and the remaining geometry are graph-derived. The prior therefore grounds MLLM interpretation without acting as an independent final segmentation head or as the reported prediction. Section IV-D evaluates the same framework with and without this input.

## B. Typed Graph, Executable API, and Bounded Control

Deterministic normalization converts the three records into stable identifiers, canonical coordinates, and typed relations for connectivity, containment, adjacency, service, and alignment. A serializer expresses this graph as Python code over a closed layout API. The main operations are add\_road, add\_building, add\_open\_space, and relate; schema-specific constructors additionally cover intersections and repeated building groups. Execution occurs in a timeout-limited subprocess that emits JSON, followed by checks for malformed geometry, duplicate identifiers, and invalid relation endpoints.

Recovery depends on the observed failure. Malformed MLLM JSON is salvaged or repaired; missing critical roads, buildings, open spaces, or relations can initiate a topologyfocused retry before deterministic topology repair and validation. The controller consequently has a bounded operational role: scheduling specialized calls, retaining intermediate artifacts, executing the program, and sending failed checks to predefined recovery paths. The reported experiments do not use visual self-iteration.

TABLE I  
COMPLETE-SYSTEM AGGREGATE RESULTS $( N = 1 0 0 , \% )$
<table><tr><td>Method</td><td>mIoU</td><td>GIoU</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Code-as-City</td><td>41.1</td><td>48.3</td><td>65.8</td><td>64.6</td><td>65.2</td></tr></table>

## C. Shared Rendering and Evaluation

One conversion turns the validated graph into a shared layout. Road centerlines are buffered by estimated widths, building polygons are extruded to approximate heights, and semantic open spaces become low surfaces. A deterministic Three.js renderer creates the oblique view; an orthographic projector rasterizes the identical object geometry with a fixed palette. For class c,

$$
\mathrm { I o U } _ { c } = \frac { | \hat { Y } _ { c } \cap Y _ { c } | } { | \hat { Y } _ { c } \cup Y _ { c } | } , \qquad \mathrm { m I o U } = \frac { 1 } { 5 } \sum _ { c \in \mathcal { C } } \mathrm { I o U } _ { c } .\tag{1}
$$

IoU is computed after class counts have been pooled across the evaluated scenes, preventing absent class–scene pairs from contributing artificial perfect scores. Since both views consume the same layout, an edit to a named object or relation is reflected when the 2D and 3D outputs are regenerated.

## IV. EXPERIMENTS

## A. Dataset and Evaluation Protocol

The evaluation uses CityLayout-100, a fixed collection of 100 near-nadir 512 × 512 crops from IRSAMap V2 [5]. Difficulty is determined by the number of annotated landcover instances: Easy scenes contain at most 17, Medium scenes 18–23, and Hard scenes at least 24. Scored classes are building, road area, sport, vegetation, and water; road-line annotations serve only as diagnostics. For each reported group, true positives, false positives, and false negatives are pooled over scenes before class IoU is calculated. mIoU averages the five pooled class IoUs, while GIoU, precision, recall, and F1 are obtained from counts pooled across classes. Thus an absent class–scene pair cannot introduce an artificial perfect score.

In the complete pipeline, gpt-image-2 generates the semantic layout prior and gpt-5.5 performs the decomposed MLLM parsing. The accumulated records subsequently pass through graph normalization, topology repair, constrainedcode execution, Three.js construction, and orthographic evaluation.

## B. Overall and Class-wise Results

Once the program has executed and been projected, Codeas-City attains 41.1% mIoU and 48.3% GIoU (Table I). The executable representation therefore retains measurable agreement in the image plane. Water, vegetation, and buildings overlap more closely than road areas and sports surfaces (Table II). Road-area IoU responds strongly to errors in bufferednetwork width and boundaries, even though the underlying roads remain named and connected rather than becoming anonymous pixels. These scores serve to assess the executable layout rather than to position a task-trained segmentation model.

TABLE II  
CLASS-WISE IOU OF THE COMPLETE SYSTEM (%).
<table><tr><td>Class</td><td>Build.</td><td>Road</td><td>Sport</td><td>Vege.</td><td>Water</td></tr><tr><td>IoU</td><td>44.8</td><td>14.0</td><td>34.2</td><td>53.1</td><td>59.5</td></tr></table>

TABLE III

COMPLETE-SYSTEM RESULTS BY SCENE DIFFICULTY (%).
<table><tr><td>Difficulty</td><td>mIoU</td><td>GIoU</td></tr><tr><td>Easy</td><td>42.3</td><td>48.5</td></tr><tr><td>Medium</td><td>36.3</td><td>50.9</td></tr><tr><td>Hard</td><td>42.7</td><td>45.6</td></tr></table>

TABLE IV

EFFECT OF THE GENERATED SEMANTIC LAYOUT PRIOR $( N = 1 0 0 , \% )$
<table><tr><td>Variant</td><td>mIoU</td><td>GIoU</td><td>Prec.</td><td>Rec. F1</td></tr><tr><td>Without semantic layout prior</td><td>36.3</td><td>42.5</td><td>60.7 58.7</td><td>59.7</td></tr><tr><td>With semantic layout prior</td><td>41.1</td><td>48.3</td><td>65.8 64.6</td><td>65.2</td></tr></table>

## C. Complexity and Qualitative Analysis

Instance-count difficulty does not produce a monotonic trend in Table III. Medium scenes yield the highest GIoU but the lowest mIoU, consistent with good overlap on dominant areas and less even agreement across classes. Hard scenes still reach 42.7% mIoU, suggesting that boundary ambiguity and rare-class geometry influence the result independently of the number of instances.

Fig. 2 presents one complete, legible case from each difficulty level, with the panel collectively covering all five classes. In every row, the planar projection and near-nadir rendering are produced by the same layout, so a discrepancy can be traced to its named object and geometry. The examples are qualitative; all quantitative claims are based on the 100-scene set.

## D. Ablation of the Semantic Layout Prior

Table IV isolates the generated semantic layout prior by evaluating the same pipeline with and without that input. In its absence, the MLLM parses the RGB image alone; when present, aligned class and footprint evidence grounds each parsing pass. The difference is 4.8 points in mIoU, 5.8 points in GIoU, and 5.5 points in F1. Per-class IoU changes by +17.6 for building, +4.0 for road area, −8.3 for sport, +2.7 for vegetation, and +8.4 for water. The aggregate gain supports semantic grounding for MLLM layout reasoning, while the sport result shows that an inaccurate prior may also propagate into executable geometry.

![](images/cf9634d9956ced4fff050c491955736db70d78e678f3401e331d768fc45206f9.jpg)  
Fig. 2. Difficulty-stratified input, ground truth, generated semantic prior, planar projection, and near-nadir 3D layout. Separate keys distinguish ground truth/prior from rendered-layout colors. Left cards give case-level scores; mIoU<sup>∗</sup> averages present classes only.

## V. CONCLUSION

Code-as-City recovers a measurable urban layout by generating constrained code with an MLLM. Localization is retained in named objects, the orthographic projection permits segmentation-style evaluation, and the executable prediction keeps typed relations and editable construction operations together with a synchronized 3D output. Over 100 scenes, the generated semantic layout prior improves the alignment of executed layouts by grounding decomposed MLLM reasoning and selected footprints. In this form, remote-sensing detection and segmentation can feed an executable-layout representation whose code remains measurable, inspectable, editable, and renderable. Subject to human review, the structured objects and geometry could be translated into city-data formats and adapted for OSM2World-like map-to-3D asset maintenance.

## REFERENCES

[1] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “DOTA: A large-scale dataset for object detection in aerial images,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 3974– 3983.

[2] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask R-CNN,”´ in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2961– 2969.

[3] S. W. Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. S. Khan, F. Zhu, L. Shao, G.-S. Xia, and X. Bai, “iSAID: A large-scale dataset for instance segmentation in aerial images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2019, pp. 28–37.

[4] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for se-

mantic image segmentation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 801–818.

[5] Y. Meng, L. Deng, Z. Xi, J. Chen, J. Chen, A. Yue, D. Liu, K. Li, C. Wang, K. Li, Y. Deng, and X. Sun, “IRSAMap: Towards large-scale, high-resolution land cover map vectorization,” arXiv preprint arXiv:2508.16272, 2025.

[6] P. Muller, P. Wonka, S. Haegler, A. Ulmer, and L. V. Gool,¨ “Procedural modeling of buildings,” in ACM SIGGRAPH 2006 Papers, 2006, pp. 614–623.

[7] H. Ledoux, K. A. Ohori, K. Kumar, B. Dukai, A. Labetski, and S. Vitalis, “CityJSON: A compact and easy-to-use encoding of the CityGML data model,” Open Geospatial Data, Software and Standards, vol. 4, no. 1, p. 4, 2019.

[8] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 14 953– 14 962.

[9] D. Sur´ıs, S. Menon, and C. Vondrick, “ViperGPT: Visual inference via python execution for reasoning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 11 888–11 898.

[10] W. Feng, W. Zhu, T.-J. Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang, “LayoutGPT: Compositional visual planning and generation with large language models,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 18 225–18 250.

[11] C. Sun, J. Han, W. Deng, X. Wang, Z. Qin, and S. Gould, “3D-GPT: Procedural 3D modeling with large language models,” arXiv preprint arXiv:2310.12945, 2023.

[12] Y. Yang, F.-Y. Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han, J. Wu, N. Haber, R. Krishna, L. Liu, C. Callison-Burch, M. Yatskar, A. Kembhavi, and C. Clark, “Holodeck: Language guided generation of 3D embodied ai environments,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16 277–16 287.

[13] Y. Yang, Z. Luo, W. Gan, J. Hao, J. Lu, J. Yan, Z. Lyu, and X. Xu, “Code-as-room: Generating 3D rooms from topdown view images via agentic code synthesis,” arXiv preprint arXiv:2605.18451, 2026.