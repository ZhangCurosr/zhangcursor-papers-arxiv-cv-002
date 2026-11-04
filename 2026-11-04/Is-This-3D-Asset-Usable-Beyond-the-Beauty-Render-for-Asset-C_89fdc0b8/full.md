# Is This 3D Asset Usable? Beyond the Beauty Render for Asset Curation

Rushikesh Zawar Adobe

Vladimir Kim Adobe

## Abstract

Large 3D repositories make it easy to collect assets, but difficult to determine whether they can be reused reliably across scene assembly, editing, rendering, game production, and training-data construction. An asset can look plausible in RGB while hiding disconnected geometry, problematic UV organization, or materials that fail to resolve correctly after import. We introduce Diagnostic Rendered Evidence (DRE): aligned beauty, wireframe, and UV-seam-plus-wireframe views scored by pass-specific frozen ViT classifiers and combined through late fusion. On 1,234 human labeled TexVerse assets, nested grouped evaluation achieves 86.36% balanced accuracy and 88.32% bad-asset recall. On a separate, balanced 764 external reference test, DRE achieved 88.9% balanced reference-label accuracy. Our method requires only an imported asset and renderer-derived evidence, without repository identity, creator, popularity, text, or source metadata at inference. Our results support cross-corpus portability and show that purpose built diagnostic views expose defects that just the beauty views can miss. Webpage: https://rbz-99.github.io/is-this-3d-asset-usable/

## 1 Introduction

Consider a common workflow where we download a model from a large 3D repository. It can look perfectly fine and still be difficult to use. A few RGB views look convincing: the shape is recognizable, the texture appears detailed, and nothing is obviously broken. Yet once the asset enters a 3D workflow, different problems can emerge. The mesh may contain disconnected or irregular geometry, the UV layout may complicate texturing, or materials may fail to resolve correctly after import. None of these problems necessarily prevents the asset from producing an attractive preview.

Repository assets are increasingly reused as structured 3D objects: retrieved and arranged into scenes [25, 1, 17], edited [11], rigged for animation [15], or used in modern image-to-3D pipelines that explicitly represent geometry, UVs, textures, and PBR attributes [3, 19, 23]. Across these uses, plausible RGB appearance does not guarantee reliable reuse.

Large repositories such as ShapeNet [4], Objaverse and Objaverse-XL [8, 7], and TexVerse [26] make this a repository-scale curation problem: How can we determine whether a 3D asset isfitfor reuse before selecting it from a large repository?

This requires expert review, that considers appearance, geometry, UV structure, and material behavior. We therefore construct a 1,234-asset TexVerse set annotated by 3D researchers and introduce diagnostic rendered evidence: aligned beauty, wireframe, and UV-seam-plus-wireframe views that expose complementary properties of the same asset. Separate models score each evidence type before late fusion produces the final keep/reject decision. Our scope is operational fitness for reusable textured/PBR assets rather than universal aesthetic quality or asset provenance. On our human-labeled assets, our method reaches 86.36% balanced accuracy and 88.32% bad-asset recall. Controlled ablations show that the structural diagnostic passes provide information beyond a matched beauty-only representation. Our contributions are: 1: We introduce diagnostic rendered evidence for 3D asset curation using aligned beauty, wireframe, and UV-seam-plus-wireframe views. 2: We develop a robust rendering pipeline for heterogeneous repository assets, including guarded material fallbacks and explicit material-resolution provenance. 3: We provide a 1,234-asset expert-annotated benchmark for reusable textured/PBR assets, with a keep/reject rubric covering geometry, UV, material, render, and composition failures.

## 2 Related Work and Scope

3D assets as reusable components: Recent systems increasingly treat existing 3D assets as reusable components. Holodeck retrieves and arranges Objaverse assets for scene generation [25]; Open-Universe similarly retrieves meshes from uncurated databases and optimizes their placement [1]; and Lay-A-Scene studies object arrangement [17]. Existing assets are also edited [11] and automatically rigged for animation [15]. These settings motivate evaluating asset properties that are not fully captured by a beauty render.

3D curation and quality assessment: Objaverse++ and MeshFleet use filtering and human review to construct higher-quality 3D corpora [14, 2], while generative-3D benchmarks evaluate geometry, texture, and cross-view consistency [13, 20, 27, 10]. Perceptual mesh-quality studies likewise show that geometry and texture jointly affect human judgments [12, 16]. Full-reference metrics assume a known reference [16, 24]; repository curation instead requires judging a single asset without one.

Structured evidence for asset quality: Modern 3D systems explicitly represent geometry, UVs, and materials. Hunyuan3D separates mesh and texture generation [28], SF3D predicts UV-unwrapped geometry and material properties [3], and AssetGen reconstructs geometry and PBR appearance [19]. Related work further models UV and material structure explicitly [5, 22, 23], while UV-specific methods motivate interpreting parameterization in surface context [21, 18]. Our method uses rendered views of these structures as no-reference evidence for asset curation.

## 3 Methodology

Evidence Generation and Provenance: For each imported GLB, we render synchronized diagnostic passes using fixed camera poses, lighting, background, and framing. Four aligned views are generated per pass and scored independently before late fusion. The beauty pass captures overall appearance and visible rendering or material failures, but may hide structural defects through texture, shading, and lighting. The wireframe pass exposes mesh structure, including disconnected components, irregular triangulation, excessive density, and geometric clutter. These patterns are treated as evidence rather than fixed quality rules. The UV-seam-plus-wireframe pass overlays UV boundaries on the mesh, allowing seam organization to be interpreted in its 3D surface context rather than from a flattened atlas alone.

Repository-Agnostic Evidence: DRE is designed around information that can be obtained from an imported asset rather than from a particular repository. The renderer uses no asset ID, creator, source label, caption, popularity statistic, or repository-specific metadata. It first resolves standard Principled-BSDF base-color inputs, then follows supported graph-connected image textures and guarded fallbacks; when resolution is not possible, it records an explicit unresolved-material state rather than inventing a material interpretation. We also generated 2D UV-island layouts and measurements such as coverage, angular distortion, and texel density. These signals were reviewed by 3D experts and evaluated in additional classifier configurations, but provided little improvement over the surface-aligned diagnostic views. We therefore retain them for review and audit rather than classifier input, particularly since UV packing conventions vary across asset categories and authoring workflows. Figure 1 illustrates the synchronized evidence available to both the reviewer and the classifier. No individual diagnostic pass is treated as ground truth. Human review separates several kinds of concern: low overall curation value, geometry integrity or clutter, UV organization, material-resolution anomalies, missing evidence, and other import or composition problems.

Multi-view Scoring and Fusion: For pass m and view v, a frozen ImageNet ViT-B/16 encoder [9] produces $z _ { m , v } \in \mathbb { R } ^ { 7 6 8 }$ . We summarize the four aligned views with their feature-wise mean and standard deviation. A standardized, class-balanced logistic-regression model is then trained separately for beauty, wireframe, and UV-seam-plus-wireframe evidence. Each model produces a probability $p _ { m } ( \mathrm { g o o d } )$ describing how strongly that particular type of evidence supports keeping the asset.

![](images/88b64f05a30564c5b3e86c99fd2f0264f74401f8ab361d9f8500a53dd8f56269.jpg)  
Figure 1: Aligned beauty, wireframe, and UV-seam-plus-wireframe evidence. The external baseline assigns this asset Superior, but the diagnostic views expose its disconnected planar fragments.

<table><tr><td colspan="2">(aSeiecteu Tuslon</td></tr><tr><td>Metric</td><td>Value</td></tr><tr><td>Accuracy</td><td>85.58%</td></tr><tr><td>Balanced accuracy</td><td>86.36%</td></tr><tr><td>Bad-asset recall</td><td>88.32%</td></tr><tr><td>Good-asset recall</td><td>84.41%</td></tr><tr><td>Good-asset precision</td><td>94.44%</td></tr><tr><td>Bad predicted good</td><td>43</td></tr><tr><td>Good predicted bad</td><td>135</td></tr></table>

(b) Representation-controlled evidence ablation
<table><tr><td>Evidence configuration</td><td>Acc.</td><td>BA</td><td>BR</td></tr><tr><td>Beauty</td><td>73.26</td><td>71.26</td><td>66.30</td></tr><tr><td>Wireframe</td><td>81.20</td><td>80.35</td><td>78.26</td></tr><tr><td>UV seams + wire</td><td>84.60</td><td>84.97</td><td>85.87</td></tr><tr><td>Beauty + wireframe</td><td>81.28</td><td>80.65</td><td>79.08</td></tr><tr><td>Beauty + UV seams + wire</td><td>84.28</td><td>84.03</td><td>83.42</td></tr><tr><td>Wireframe + UV seams + wire</td><td>84.04</td><td>84.88</td><td>86.96</td></tr><tr><td>All three passes</td><td>82.90</td><td>84.15</td><td>87.23</td></tr></table>

Table 1: Grouped development results on 1,234 human-labeled assets. (a) Selected nested grouped fusion. (b) Ablations holding labels, groups, ViT representation, scoring family, and thresholdselection protocol fixed. Acc.: accuracy, BA: balanced accuracy, BR: bad-asset recall.

## 4 Learning and Evaluation

We use TexVerse [26] to create our set under CC BY, CC BY-SA, or CC0 licenses. Our supervised set contains 1,234 assets annotated by expert 3D researchers: 866 good and 368 bad. An additional 6,685 assets based on the deterministic render QA and their metadata are used as less weighted auxiliary training data. We use five-fold nested stratified group cross-validation, grouping by creator or source where available. Inner folds train the pass-specific classifiers, generate out-of-fold predictions for fusion, and select weak-label weights, regularization, pass combinations, and the decision threshold. Auxiliary samples sharing a group with the held-out fold are excluded. Metadata and strict full-PBR inventory provide weak-label sources, while terms such as Rodin, photogrammetry, or scan identify low-weight negative candidates rather than evaluation labels. Face count, texture resolution, creator, and source are retained only for audit and stratification. Table 1 reports outer-fold predictions.

Modality Ablation: Table 1 keeps the labels, grouped folds, ViT representation, classifier family, fusion method, and threshold-selection protocol fixed while changing only the render evidence. UV-seam-wire achieves the highest balanced accuracy, while the full three-pass model gives the highest bad recall and fewest false accepts. Thus, the benefit is not simply due to using more views: the structural passes provide information complementary to beauty renders. We exclude material-only views because their appearance is difficult to separate from renderer behavior. A gray render, for example, may indicate an intentional material, scalar input, vertex color, missing texture, or failed graph resolution. Material behavior is therefore audited separately rather than used as a learned visual modality.

Baselines: The matched beauty-only ViT in Table 1 is our primary representation baseline. It keeps the labels, groups, backbone, classifier, fusion, and threshold-selection protocol fixed, isolating the contribution of diagnostic evidence. We also evaluate the released Objaverse++ [14] on our dataset using its 40-view protocol. Our method uses no popularity, creator, source, or text features at

Table 2: Objaverse++ versus DRE (ours) on expert-labeled TexVerse assets and a cross-source reference set.
<table><tr><td>Set</td><td>Model</td><td> $_ \mathrm { A c c . / B A \uparrow }$ </td><td> $\mathrm { { G } - \mathrm { { R e c . } \uparrow } }$ </td><td> $\mathbf { B } { \cdot } \mathbf { R e c } . \ \uparrow$  1</td><td>BAR↓</td></tr><tr><td>TexVerse (1,234)</td><td>Objaverse++ DRE (ours)</td><td>44.65 / 55.72 85.58 / 86.36</td><td>28.29 84.41</td><td>83.15 88.32</td><td>16.85 11.68</td></tr><tr><td>Cross-source</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Objaverse++</td><td>46.99 / 46.99</td><td>88.74</td><td>5.24</td><td>94.76</td></tr><tr><td>(764)</td><td>DRE (ours)</td><td>88.87 / 88.87</td><td>80.37</td><td>97.38</td><td>2.62</td></tr></table>

![](images/21322ca1519e0f3930a8e076d0d24dafc3fb551063f62d313af70bab9b0a85a4.jpg)  
Figure 2: Three failure cases revealed by diagnostic rendered evidence. (A) An incomplete scan-like fragment is visibly unusable in beauty views. (B) A plausible-looking character shows unusually broad UV-seam coverage. (C) A regression failure rated Superior by the external baseline reveals disconnected and intersecting geometry in wireframe views. These examples illustrate diagnostic evidence rather than universal rejection rules.

inference and requires 12 diagnostic images per asset, compared with 40 RGB views for Objaverse++.   
Objaverse++ outputs are mapped as High/Superior = keep and Low/Medium = reject.

Our method retains 731 of 866 good assets and admits 43 of 368 bad assets. Objaverse++ retains 245 good assets and admits 62 bad assets. A paired exact McNemar test gives $p = \bar { 1 } . 3 1 \times 1 0 ^ { - 1 0 2 }$ , with 568 assets correct only for our method and 63 correct only for Objaverse++. Further we additionally evaluate cross-corpus portability on a fixed, balanced external reference set of 382 good and 382 bad assets from strict-PBR ABO [6] and Objaverse assets. DRE achieves 88.9% balanced reference-label accuracy, with 80.4% good-reference recall and 97.4% bad-reference recall. Objaverse got about 47.0% balanced reference-label accuracy: although it retains 88.7% of the strict-PBR reference assets, it rejects only 5.2% of the bad assets, admitting 94.8% as good.

Figure 2 shows how diagnostic passes make different failure modes inspectable. Some failures are visible in conventional appearance renders, while others become clear only when the same camera poses expose geometry or UV seams. The diagnostic representation does not turn mesh density or seam count into hard rules. It gives reviewers and the classifier evidence that can be interpreted in the context of the asset and its intended downstream use.

## 5 Conclusion

A beauty render can indicate whether an asset looks plausible, but not whether it is operationally fit for reliable reuse. Diagnostic evidence for appearance, topology, UV structure, and material resolution makes the relevant conditions inspectable from the asset itself. By separating these evidence types before fusion, DRE provides a repository-agnostic basis for quality-controlled asset selection for various workflows.

## References

[1] Aguina-Kang, R., Gumin, M., Han, D.H., et al.: Open-universe indoor scene generation using llm program synthesis and uncurated object databases (2024)

[2] Boborzi, D., Mueller, P., Emrich, J., Schmid, D., Mueller, S., Mikelsons, L.: Meshfleet: Filtered and annotated 3d vehicle dataset for domain specific generative modeling. arXiv preprint arXiv:2503.14002 (2025)

[3] Boss, M., Huang, Z., Vasishta, A., Jampani, V.: Sf3d: Stable fast 3d mesh reconstruction with uv-unwrapping and illumination disentanglement. arXiv preprint arXiv:2408.00653 (2024)

[4] Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., Xiao, J., Yi, L., Yu, F.: Shapenet: An information-rich 3d model repository. Tech. Rep. arXiv:1512.03012, Stanford University, Princeton University, and Toyota Technological Institute at Chicago (2015)

[5] Chen, Z., Wang, Y., Sun, W., Wang, F., Chen, Y., Liu, H.: Meshgen: Generating pbr textured mesh with render-enhanced auto-encoder and generative data augmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5835–5848 (2025)

[6] Collins, J., Goel, S., Deng, K., Luthra, A., Xu, L., Gundogdu, E., Zhang, X., Vicente, T.F.Y., Dideriksen, T., Arora, H., Guillaumin, M., Malik, J.: Abo: Dataset and benchmarks for realworld 3d object understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21126–21136 (2022)

[7] Deitke, M., Liu, R., Wallingford, M., Ngo, H., Michel, O., Kusupati, A., Fan, A., Laforte, C., Voleti, V., Gadre, S.Y., VanderBilt, E., Kembhavi, A., Vondrick, C., Gkioxari, G., Ehsani, K., Schmidt, L., Farhadi, A.: Objaverse-xl: A universe of 10m+ 3d objects. In: Advances in Neural Information Processing Systems. vol. 36 (2023)

[8] Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13142–13153 (2023)

[9] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

[10] Ebert, D., et al.: 3d arena: An open platform for generative 3d evaluation. arXiv preprint arXiv:2506.18787 (2025)

[11] Erkoc¸, Z., Gumeli, C., Wang, C., et al.: Preditor3d: Fast and precise 3d shape editing (2024)¨

[12] Guo, J., Vidal, V., Cheng, I., Basu, A., Baskurt, A., Lavoue, G.: Subjective and objective visual´ quality assessment of textured 3d meshes. arXiv preprint arXiv:2102.03982 (2021)

[13] He, Y., Bai, Y., Lin, M., Zhao, W., Hu, Y., Sheng, J., Yi, R., Li, J., Liu, Y.J.: T<sup>3</sup>bench: Benchmarking current progress in text-to-3d generation. arXiv preprint arXiv:2310.02977 (2023)

[14] Lin, C., Liu, H., Lin, Q., Bright, Z., Tang, S., He, Y., Liu, M., Zhu, L., Le, C.: Objaverse++: Curating a high-quality 3d dataset for training and evaluation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. pp. 6813–6822 (2025)

[15] Liu, I., Xu, Z., Yifan, W., et al.: Riganything: Template-free autoregressive rigging for diverse 3d assets (2025)

[16] Nehme, Y., Delanoy, J., Dupont, F., Farrugia, J.P., Le Callet, P., Lavoue, G.: Textured mesh ´ quality assessment: Large-scale dataset and deep learning-based quality metric. arXiv preprint arXiv:2202.02397 (2023)

[17] Rahamim, O., Segev, H., Achituve, I., et al.: Lay-a-scene: Personalized 3d object arrangement using text-to-image priors (2024)

[18] Richardson, E., Metzer, G., Alaluf, Y., Giryes, R., Cohen-Or, D.: Texture: Text-guided texturing of 3d shapes. arXiv preprint arXiv:2302.01721 (2023)

[19] Siddiqui, Y., Monnier, T., Kokkinos, F., Kariya, M., Kleiman, Y., Garreau, E., Gafni, O., Neverova, N., Vedaldi, A., Shapovalov, R., Novotny, D.: Meta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials. arXiv preprint arXiv:2407.02445 (2024)

[20] Su, S., Cai, X., Gao, L., Zeng, P., Du, Q., Li, M., Shen, H.T., Song, J.: Gt23d-bench: A comprehensive general text-to-3d generation benchmark. arXiv preprint arXiv:2412.09997 (2024)

[21] Wang, Z., Wei, X., Shi, R., Zhang, X., Su, H., Liu, M.: Partuv: Part-based uv unwrapping of 3d meshes. arXiv preprint arXiv:2511.16659 (2025)

[22] Wei, X., Zhang, B., Yang, X., Wang, Y., Guo, C., Zhao, X., Luximon, Y.: Pbr3dgen: A vlm-guided mesh generation with high-quality pbr texture. arXiv preprint arXiv:2503.11368 (2025)

[23] Xiang, J., Chen, X., Xu, S., Wang, R., Lv, Z., Deng, Y., Zhu, H., Dong, Y., Zhao, H., Yuan, N.J., Yang, J.: Native and compact structured latents for 3d generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14419–14429 (2026)

[24] Yang, K., Yang, Q., Li, Z., Xu, Y.: Textured mesh quality assessment using geometry and color field similarity. arXiv preprint arXiv:2505.10824 (2025)

[25] Yang, Y., Sun, F.Y., Weihs, L., et al.: Holodeck: Language guided generation of 3d embodied ai environments. CVPR (2024)

[26] Zhang, Y., Zhang, L., Ma, R., Cao, N.: Texverse: A universe of 3d objects with high-resolution textures. arXiv preprint arXiv:2508.10868 (2025)

[27] Zhang, Y., Cui, B., Yang, Q., Li, Z., Xu, Y.: Benchmarking and learning multi-dimensional quality evaluator for text-to-3d generation. arXiv preprint arXiv:2412.11170 (2024)

[28] Zhao, Z., Lai, Z., Lin, Q., Zhao, Y., Liu, H., et al.: Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 (2025)