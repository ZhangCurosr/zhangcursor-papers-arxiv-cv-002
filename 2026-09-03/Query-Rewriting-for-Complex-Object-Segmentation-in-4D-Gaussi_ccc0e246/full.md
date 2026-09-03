# Query Rewriting for Complex Object Segmentation in 4D Gaussian Representations

Thanh-Khoi Nguyen<sup>1,2</sup> Thien-Phuc Tran <sup>⋆1,2</sup> Minh-Triet Tran<sup>1,2</sup>

<sup>1</sup> University of Science, Ho Chi Minh City, Vietnam2 Viet Nam National University, Ho Chi Minh City, Vietnam23120009@student.hcmus.edu.vn, ttphuc@selab.hcmus.edu.vn,tmtriet@fit.hcmus.edu.vn

Abstract. Recent 4D Gaussian representation frameworks have demonstrated strong performance in language-guided dynamic scene understanding. However, these methods remain highly sensitive to verbose and narrative-style queries that contain noisy contextual information. In this paper, we investigate the impact of query rewriting for complex object segmentation in 4D Gaussian representations. Inspired by recent findings in retrieval-augmented language models and keyword-guided query reformulation, we propose a training-free reinterpretation strategy that transforms long descriptive queries into concise keyword-grounded forms. Our approach progressively reduces linguistic noise while preserving semantic anchors relevant to object-centric representations. Experiments on HyperNeRF and Neu3D demonstrate that concise rewritten queries significantly improve both temporal localization and spatial segmentation performance. In particular, our method improves average temporal accuracy from 60.92% to 92.21% and average vIoU from 20.08% to 76.94% without any additional fine-tuning. Extensive ablation studies further reveal that shorter, keyword-focused queries consistently yield stable video-feature similarity distributions and better alignment with object-centric Gaussian representations.

Keywords: 4D Gaussian Splatting · Query Rewriting · Temporal Segmentation · Video Understanding · Language-Guided Segmentation

## 1 Introduction

Language-guided retrieval and segmentation in dynamic scenes have recently attracted significant attention due to the emergence of 4D Gaussian representation frameworks [21]. Methods such as 4D LangSplat [12] enable languageconditioned querying over 4D Gaussians by associating semantic embeddings with spatiotemporal scene elements. Despite their efectiveness, these approaches exhibit substantial sensitivity to query formulation.

In practical settings, users rarely provide concise object-centric queries. Instead, queries are often verbose, narrative, ambiguous, or conversational. The core issue with long-form queries is that they often extend beyond describing the target object and incorporate substantial peripheral context. This introduces irrelevant signals into the input, efectively acting as noise without contributing meaningful discriminative information for retrieval.

According to Schema Theory in cognitive psychology, humans acquire new knowledge most efectively when incoming information is closely aligned with existing prior knowledge structures. When a concept is expressed using unfamiliar, verbose, or ambiguous language, it becomes significantly more dificult for learners to comprehend and retain . In contrast, when instructors employ familiar keywords and core concepts as anchoring cues, learners can more readily assimilate new information into their existing schemas, thereby substantially improving learning eficiency. Analogously, retrieval models - particularly those trained on concise and well-structured captions - struggle when processing long, complex queries that deviate from the training distribution. Existing studies suggest a semantic gap between natural user expressions and the representations used for retrieval. Furthermore, long and unfamiliar queries often introduce hallucinated or out-of-scope semantic concepts, reducing retrieval robustness.

Motivated by these observations, we investigate the role of query rewriting for language-guided segmentation in 4D Gaussian representations. Rather than expanding queries with additional context, we explore the opposite direction: progressively simplifying and grounding queries toward concise keyword-focused forms. Our approach is training-free and requires no modifications to the underlying 4D LangSplat architecture.

Specifically, we introduce a multi-level rewriting framework that transforms verbose narrative descriptions into compact keyword-grounded expressions. We demonstrate that shorter and more familiar queries significantly improve temporal localization and segmentation quality. Extensive experiments on Hyper-NeRF [13] and Neu3D [10] dataset show substantial improvements across all evaluated scenes.

Our contributions are summarized as follows:

We expose a critical structural vulnerability in current language-guided 4D Gaussian pipelines: the ’semantic gap’ between natural user formulations and the object-centric supervision paradigm used to train 4D Gaussians. We demonstrate that these models do not merely struggle with query length, but fail fundamentally due to the difusion of spatial activations caused by peripheral context

– We propose a simple yet efective training-free query-rewriting framework. By leveraging an LLM to dynamically extract attributes rather than finetuning per scene, our approach remains scene-independent while preserving open-vocabulary capabilities, thereby enhancing query interpretability, thus improving temporal localization and segmentation quality per query.

– We provide extensive ablation studies analyzing the efects of query length and keyword grounding on retrieval performance.

## 2 Related Work

4D Gaussian Splatting for Dynamic Scenes. Our work builds on recent advances in 4D Gaussian splatting for real-time dynamic scene rendering. Wu et al. introduced 4D Gaussian Splatting (4D-GS), which augments static 3D Gaussian splats with temporal components to enable eficient novel view synthesis of dynamic scenes while maintaining high visual fidelity [21]. Subsequent extensions have explored alternative parameterizations and eficiency improvements, such as 4D-Rotor Gaussian Splatting (4D-RotorGS) for anisotropic XYZT Gaussians [4] and orientation-anchored Gaussians for robust 4D reconstruction from monocular video [22]. These works primarily focus on geometric and photometric modeling, whereas we target robustness to language-guided segmentation on top of existing 4D Gaussian representations.

Language-guided 4D Gaussian Splatting. Several recent approaches have begun to enrich 4D Gaussian representations with semantic and linguistic cues. 4-LEGS extends this direction by embedding dense spatiotemporal language features directly into dynamic Gaussian representations, enabling finegrained text-conditioned localization and editing across both space and time in dynamic scenes [5]. Recently emerging as a generalist approach, 4D LangSplat learns “language fields” over 4D Gaussians via multimodal large language models (MLLMs), enabling both time-agnostic and time-sensitive open-vocabulary queries in dynamic scenes [12]. While recent systems have focused heavily on architectural and training strategies to align language with 4D geometry, they implicitly assume that users will provide optimal, object-centric text inputs. Our work investigates the query-side vulnerability of 4D Gaussian representations. By demonstrating that standard language-vision embeddings are highly unstable under natural, verbose formulations, we introduce a training-free structural intervention that maps natural language into the specific geometric and temporal parameters required by 4D Gaussians.

4D Segmentation and Semantic Scene Editing. In parallel with advances in representation and language conditioning, there has been growing interest in segmentation and editing in 4D Gaussian spaces. Ji et al. introduced Segment Any 4D Gaussians (SA4D), which generalizes “segment anything” capabilities to 4D worlds and supports operations such as object removal, recoloring, and composition on 4D Gaussians [7]. Wei’s thesis presents a Gaussian splatting-based 4D segmentation framework for volumetric video that enables semantic editing of dynamic scenes and ofers a comprehensive treatment of 4D segmentation pipelines, losses, and editing operations [20]. Compared to these approaches, our focus is not on proposing a new segmentation backbone, but on improving the robustness and stability of language-guided 4D segmentation by manipulating the query formulation itself.

Open-vocabulary 3D/4D Semantics. Our approach is intended to bridge the research gap in open-vocabulary 3D scene understanding, in which neural fields are supervised by language to support flexible semantic queries. A line of work on language fields for 3D scenes uses CLIP-like supervision to map spatial locations to semantic embeddings, enabling open-vocabulary querying and editing; we refer to a representative method in this space as open-vocabulary 3D scene understanding with neural fields and language supervision [11,15,24,9]. 4D LangSplat can be viewed as a 4D analogue of these language fields, extending them to dynamic scenes and 4D Gaussian representations [12]. In contrast to designing new language fields, our method assumes a pretrained languagevision space and instead studies how diferent query formulations - from narrative descriptions to concise keyword lists - afect retrieval and segmentation performance in that space.

Foundational Vision-Language Models. Finally, our method leverages foundational vision–language and segmentation models that have become standard components in open-vocabulary perception pipelines. Segment Anything (SAM) provides a high-capacity segmentation model that can generate object masks from 2D images with minimal supervision, and has been widely adopted as a geometric backbone in 3D and 4D pipelines [8,17,2]. CLIP [16] learns a joint image-text embedding space from natural language supervision at scale, enabling open-vocabulary image retrieval, classification, and segmentation. In our framework, SAM provides object-centric supervision in the image domain, while CLIP (or similar encoders) defines the language-vision embedding space, whose behavior we analyze under diferent query formulations; our query-rewriting strategy is designed to better exploit this space without any additional training.

## 3 Method

Our approach introduces a training-free, multi-level query-rewriting framework that improves language-guided segmentation in 4D Gaussian representations without requiring modifications to the underlying 4D LangSplat architecture. As discussed, user queries are often verbose and include not only descriptions of the target object but also substantial irrelevant peripheral context. Within the 4D LangSplat framework, 4D Gaussians are supervised using caption embeddings that are explicitly object-centric, capturing the deformation, motion, and state of a single object across frames. This design is particularly efective for object-centric descriptions (e.g., “a chicken” or “a cup”), but remains highly sensitive to extraneous linguistic artifacts (e.g., “sunlight” or “wind”).

Our key motivation is therefore to eliminate unnecessary contextual information and refine retrieval intent, reducing queries to their most concise and informative components. Empirical evidence further suggests that query expansion in LLM-based retrieval systems can lead to cumulative errors and degraded performance as query length increases [1]. Consequently, we focus on query compression, leveraging keywords as anchors during the rewriting process. This approach preserves semantic fidelity while constraining the generated captions to remain aligned with the target object and its associated factual attributes. The impact of keywords injection and query length is ablated in section 5. Our method is described in Figure 1.

![](images/f9b3dd4020d4dbb57f1082fb76a78bc0b4387d9c0f80f3c2f6d0977b679001fe.jpg)  
Fig. 1. Overall pipeline of our framework. We propose a simple training-free query reinterpretation module that removes linguistic noise, thereby improving the alignment of embeddings with object-centric 4D Gaussians and significantly boosting both temporal localization and spatial segmentation quality. Our method leverages of-the-shelf models, without any additional training.

## 3.1 Base Framework and Feature Extraction

We build on 4D LangSplat as our base framework for representing dynamic scenes and execute our method without any additional fine-tuning. In this framework, the dynamic 4D scene is parameterized by a set of deformable 3D Gaussians. To support semantic understanding, each Gaussian is augmented with dual semantic fields: a time-agnostic field and a time-varying field. For the timeagnostic semantic field, geometric features are first segmented into hierarchical regions using the Segment Anything Model (SAM) [8], after which CLIP [16] is employed to extract static feature representations. These features are subsequently passed through an autoencoder to reduce their dimensionality from 512 to 3, thereby improving memory eficiency. A parallel pipeline is used to construct the time-sensitive field: Gaussians are supervised by frame-level captions, and a status-deformable network predicts time-sensitive semantic features across diferent temporal instances.

## 3.2 Problem Formulation

Let a dynamic 4D scene be represented by a set of deformable Gaussians, where $F _ { x , y } ^ { \mathrm { a g n } } ~ \in ~ \mathbb { R } ^ { d _ { 1 } }$ and $F _ { x , y , t } ^ { \mathrm { s e n } } ~ \in ~ \mathbb { R } ^ { d _ { 2 } }$ denote the rendered time-agnostic and timesensitive semantic feature vectors at spatial pixel coordinates (x, y) and temporal step t, respectively, corresponding to the dual semantic fields embedded in each Gaussian. In the standard language-guided retrieval pipeline, a user provides a natural language query q, which is encoded into a text embedding using either

$E _ { T } ^ { \mathrm { a g n } } ( q ) \in \mathbb { R } ^ { d _ { 1 } }$ for the time-agnostic field, or $E _ { T } ^ { \mathrm { s e n } } ( q ) \in \mathbb { R } ^ { d _ { 2 } }$ for the time-sensitive field.

The retrieval pipeline proceeds in two sequential steps. First, $E _ { T } ^ { \mathrm { a g n } } ( q )$ is matched against $F _ { x , y } ^ { \mathrm { a g n } }$ via a relevance score to localize the target object and produce a spatial candidate mask $\mathcal { M } _ { x , y } .$ Second, the cosine similarity between $E _ { T } ^ { \mathrm { s e n } } ( q )$ and the time-sensitive features $F _ { x , y , i } ^ { \mathrm { s e n } }$ is computed exclusively within $\mathcal { M } _ { x , y } \mathtt { i }$

$$
\mathcal { S } ( q , F _ { x , y , t } ^ { \mathrm { s e n } } ) = \frac { E _ { T } ^ { \mathrm { s e n } } ( q ) \cdot F _ { x , y , t } ^ { \mathrm { s e n } } } { \Vert E _ { T } ^ { \mathrm { s e n } } ( q ) \Vert \Vert F _ { x , y , t } ^ { \mathrm { s e n } } \Vert } , \quad ( x , y ) \in \mathcal { M } _ { x , y }\tag{1}
$$

A frame t is classified as active if S exceeds a predefined threshold $\tau ,$ , and inactive otherwise.

The Semantic Gap. The core vulnerability of this formulation arises from the inherent nature of the input query q. In practical scenarios, users rarely input a concise, object-centric query q<sub>anchor</sub> $( \mathrm { { e . g . } , \tilde { \Omega } a \ c u p ^ { 3 } ) }$ . Instead, they typically provide a verbose, descriptive query q<sub>long</sub> $( \mathrm { e . g . , \ ^ { 6 } a }$ clear glass cup resting on a wooden table while sunlight hits it”).

We formulate $q _ { \mathrm { l o n g } }$ as a composition of the core semantic intent $q _ { \mathrm { a n c h o r } }$ and extraneous peripheral context $q _ { \mathrm { n o i s e } } .$

$$
q _ { \mathrm { l o n g } } = q _ { \mathrm { a n c h o r } } \cup q _ { \mathrm { n o i s e } }\tag{2}
$$

Because the foundation text encoder $E _ { T } ( \cdot )$ processes the sentence holistically, the presence of $q _ { \mathrm { n o i s e } }$ shifts the resulting embedding $E _ { T } ( q _ { \mathrm { l o n g } } )$ away from the optimal object-centric representation in the shared feature space. This semantic shift introduces irrelevant signals, leading to difused spatial activations and unstable temporal similarity scores $S ( q _ { \mathrm { l o n g } } , F _ { x , y , t } )$

Optimization Objective. To bridge this semantic gap without altering the underlying 4D LangSplat architecture or requiring additional fine-tuning, our objective is to define a training-free rewriting function $\mathcal { R } ( \cdot )$ that extracts the core semantic anchors from the verbose query:

$$
q _ { \mathrm { a n c h o r } } = \mathcal { R } ( q _ { \mathrm { l o n g } } )
$$

The ideal rewriting function minimizes the linguistic noise such that the similarity response of the rewritten query produces a precise spatial mask $\mathcal { M } _ { x , y }$ via $E _ { T } ^ { \mathrm { a g n } } ( \mathcal { R } ( q _ { \mathrm { l o n g } } ) )$ , and maximizes the cosine similarity S within that mask via $E _ { T } ^ { \mathrm { s e n } } ( \mathcal { R } ( q _ { \mathrm { l o n g } } ) )$ , jointly improving temporal localization and spatial segmentation quality.

## 3.3 Query Rewriting Strategy

Object-Level Caption Generation. Given a monocular video sequence, our primary objective is to construct robust textual descriptions that accurately define the ground-truth semantic target features, denoted as $F _ { t a r g e t }$ in our formulation. Following recent advancements in multimodal scene understanding [12,23], we employ an MLLM to perform instance-specific video captioning. Crucially, this MLLM-based visual prompting process is strictly a theoretical setup used exclusively to define the ideal semantic upper bound $( F _ { t a r g e t } )$ for our optimization objective. It is not executed during user inference. By establishing this mathematically clean, object-centric baseline ofline, we provide a strict target for our real-time rewriting function $\mathcal { R } ( q _ { l o n g } )$ to emulate.

Initially, spatial regions within the video are extracted and tracked using the Segment Anything Model (SAM) [8] and DEVA [3]. Because SAM generates hierarchical segmentations across multiple granularities, we explicitly filter the outputs to retain only object-level mask regions. During the captioning phase, we prompt the MLLM to generate per-object descriptions that comprehensively encapsulate four semantic dimensions: (1) the inherent identity of the object, (2) its visual appearance, (3) its current physical state, and (4) its temporal deformation or transformation trajectory.

A fundamental challenge in this process is efectively guiding the MLLM’s visual attention. Feeding isolated foreground masks strips away critical contextual cues, whereas providing raw video frames often causes the MLLM to hallucinate or conflate similar instances in complex dynamic scenes. To mitigate this, we adopt the visual prompting mechanism introduced in 4D LangSplat [12]. Specifically, we superimpose a distinct boundary contour around the target object and apply a grayscale filter to all background pixels outside this boundary. This strategy grounds the MLLM’s attention directly on the target object while preserving the necessary global scene context.

Keyword-Anchored Query Rewriting. As previously discussed, augmenting the verbose query $q _ { \mathrm { l o n g } }$ via query expansion introduces supplementary context, which frequently incurs the risk of semantic hallucinations and popularity bias, particularly for unfamiliar queries [1]. Consequently, we optimize the query formulation using a rigorous query-shortening mechanism. However, this compression must strictly avoid information loss by adhering to three core criteria: (1) absolute preservation of the target object’s identity, (2) retention of the core semantic intent of the original query, and (3) generation of a structural formulation that is highly condensed yet precisely accurate.

To meet these criteria, prompting an LLM to perform unconstrained shortening and simple summarization is insuficient. Without explicit constraints, LLMs generalize specific concepts into broader descriptions, leading to semantic drift and the omission of critical state-level details. To address this, we introduce a Keyword Injection mechanism to establish explicit semantic anchors within our rewriting function $\mathcal { R } ( \cdot ) \left[ 1 \right]$ . Directly incorporating these keywords into the rewrit ing process preserves the core semantic intent and guides our retrieval system’s attention to converge precisely on the target entity. Our rewriting mechanism serves as a structural translation layer mapping parts of the query into a strict, deterministic framework (Subject, State, Deformation).

Specifically, this pipeline uses a lightweight LLM to extract a constrained set of keywords from the object-level captions. These extracted, open-vocabulary keywords are deterministically categorized into essential attributes for 4D representation: Subject, State, and Deformation. This tripartite classification is explicitly designed to align with the dual semantic fields of the underlying 4D LangSplat architecture [12]. Following extraction, the LLM reconstructs the query by centering it on these semantic anchors, synthesizing them into a concise, grammatically coherent structure. This rewriting operation serves as a semantic filter, substantially mitigating the influence of the peripheral context q<sub>noise</sub> rather than explicitly masking it. Consequently, the resulting rewritten query q<sub>anchor</sub> yields a more stable semantic representation.

## 3.4 Integration and Inference Pipeline

Once the optimal, keyword-anchored query q<sub>anchor</sub> is extracted via our rewriting function R, it is mapped into the shared language-vision embedding space using pretrained text encoders to yield the semantic embeddings, $E _ { T } ^ { \mathrm { a g n } } ( q _ { \mathrm { a n c h o r } } )$ and $E _ { T } ^ { \mathrm { s e n } } ( q _ { \mathrm { a n c h o r } } )$ . Following 4D LangSplat, we employ a two-stage querying pipeline to isolate the target object and extract the final dynamic segmentation mask. First, $E _ { T } ^ { \mathrm { a g n } } ( q _ { \mathrm { a n c h o r } } )$ is matched against the time-agnostic features $F _ { x , y } ^ { \mathrm { a g n } }$ via a relevance score to produce the spatial candidate mask $\mathcal { M } _ { x , y } ,$ establishing the geometric presence of the target object irrespective of its temporal state. Subsequently, $E _ { T } ^ { \mathrm { s e n } } ( q _ { \mathrm { a n c h o r } } )$ is matched against the time-sensitive features $F _ { x , y , t } ^ { \mathrm { s e n } }$ via cosine similarity $s$ exclusively within $\mathcal { M } _ { x , y } .$ Frames whose score exceeds the threshold τ are classified as active, and $\mathcal { M } _ { x , y }$ is retained as the final segmentation mask for those frames.

## 4 Experiment

## 4.1 Experimental Setup

Datasets. We evaluate our method on four dynamic scenes from the HyperNeRF [13] and four from the Neu3D [10] dataset. These standard 3D/4D datasets primarily feature highly concise, object-centric captions that fail to capture realworld user interactions; in practice, users naturally employ conversational, narrative, and verbose phrasing that conveys substantial context. To rigorously simulate this well-documented human behavior, we utilized Gemini 3.0 [14] to generate complex queries that emulate realistic human verbosity and conversational ambiguity. This ensures the queries realistically mimic the ’semantic gap’ by including descriptive noise, without containing explicit keyword matches that directly refer to objects in the video, and fully reflect the four-level query complexity we proposed in Section 3.

Implementation Details. Following the 4D LangSplat pipeline, we first use the time-agnostic features to identify candidate masks for the referenced object, applying a cosine similarity threshold of 0.5. Subsequently, the same procedure is applied to the time-sensitive features to select the subset of masks that satisfy the query’s temporal constraints. We focus on rewriting and standardizing query formulations as described in section 3.3. Specifically, we employ Qwen2- VL-7B [19] as the multimodal backbone for instance-specific object captioning, e5-mistral-7b [18] for time-sensitive text decoder and utilize Llama3-8B [6] as the lightweight language model to execute the keyword extraction and rewriting pipeline.

To further experiment on the efectiveness of prompt length and keywords, we rewrite the original query at diferent levels:

– Level 0: Uses the original query without any modification.

– Level 1: Incorporates exactly one keyword corresponding to the referenced object, ensuring that this object does not appear at the beginning of the sentence; the query length is reduced to half of Level 0.

– Level 2: Utilizes all the generated keywords to construct the query, with the length reduced to half of Level 1.

– Level 3: Forms a semantically coherent sentence using only the generated keywords, with the shortest possible length.

All experiments are conducted on a single A100 GPU.

Baselines. We build upon 4D LangSplat as our base framework without any additional fine-tuning. For time-agnostic queries, geometric features are first segmented into hierarchical regions using SAM, and then CLIP is used to extract feature representations. These features are subsequently passed through an autoencoder to reduce their dimensionality from 512 to 3 for memory eficiency. Furthermore, we include 4DLangVGGT [23] as an additional baseline, representing a recent approach to 4D language-guided Gaussian Splatting for spatiotemporal scene understanding.

Metrics. Following [12], we adopt three primary metrics.

– Video IoU (vIoU) measures end-to-end pipeline performance. For each query, frames are classified as active or inactive based on whether their video feature similarity exceeds the mean similarity threshold, and vIoU is computed as the average spatial IoU between the predicted mask and ground-truth mask over correctly classified active frames.

– Temporal Accuracy measures the proportion of frames correctly classified as active or inactive, evaluating temporal localization independently of spatial grounding quality.

– Mean IoU (mIoU) evaluates the spatial segmentation quality for timeagnostic queries. It is calculated as the average Intersection over Union (IoU) between the predicted spatial mask and the ground-truth mask across all frames in the test set, independent of any temporal state constraints.

## 4.2 Main Results

Comparison with Baselines. Table 1 reports the vIoU and temporal accuracy across the four HyperNeRF scenes, where our method consistently outperforms both 4DLangVGGT and 4D LangSplat in time-sensitive querying. To prove the generalizability of our rewriting framework beyond a single dataset, we extend our evaluation to four additional distinct scenes from the Neu3D dataset (Table 2). As the results show, our training-free approach demonstrates strong robustness across complex, previously unseen conversational domains.

Table 1. Quantitative comparison on time-sensitive querying with complex queries on the HyperNeRF dataset [13].
<table><tr><td rowspan="2">Method</td><td colspan="2">americano</td><td colspan="2">chickchicken</td><td colspan="2">split-cookie</td><td colspan="2">espresso</td><td colspan="2">Average</td></tr><tr><td>Acc(%) vIoU(%)</td><td></td><td>)Acc(%) vIoU(%)</td><td></td><td></td><td></td><td>)Acc(%) vIoU(%) Acc(%) vIoU(%)</td><td></td><td>Acc(%) vIoU(%)</td><td></td></tr><tr><td>4DLangVGGT</td><td>53.70</td><td>13.40</td><td>58.53</td><td>6.25</td><td>56.39</td><td>0.00</td><td>48.14</td><td>10.12</td><td>54.19</td><td>7.44</td></tr><tr><td>4D LangSplat</td><td>55.05</td><td>34.07</td><td>44.02</td><td>8.93</td><td>71.70</td><td>23.95</td><td>72.92</td><td>13.38</td><td>60.92</td><td>20.08</td></tr><tr><td>Ours</td><td>95.20</td><td>83.74</td><td>95.65</td><td>89.02</td><td>97.17</td><td>85.38</td><td>80.83</td><td>49.60</td><td>92.21</td><td>76.94</td></tr></table>

Table 2. Time-agnostic query on Neu3D dataset[10] using descriptive queries.
<table><tr><td rowspan="2">Method</td><td colspan="4">Neu3D (mIoU)</td><td rowspan="2"></td></tr><tr><td>coffee-martini cook-spinach flame-salmon flame-steak Average</td><td></td><td></td><td></td></tr><tr><td>4D LangSplat</td><td>61.82</td><td>77.64</td><td>41.45</td><td>48.57</td><td>57.37</td></tr><tr><td>Ours</td><td>75.43</td><td>86.28</td><td>48.13</td><td>91.17</td><td>75.25</td></tr></table>

Table 3. Mean vIoU improvement over the descriptive query (∆vIoU, %) at each query familiarity level on the HyperNerf dataset [13].
<table><tr><td>Samples</td><td colspan="4">∆vIoU (L1) ∆vIoU (L2) ∆vIoU (Anchor) Improved Queries</td></tr><tr><td>americano</td><td>-0.73</td><td>+2.48</td><td>+49.67</td><td>6/8</td></tr><tr><td>espresso</td><td>-0.08</td><td>+1.63</td><td>+36.22</td><td>9/12</td></tr><tr><td>chickchicken</td><td>+41.47</td><td>+39.70</td><td>+80.08</td><td>8/8</td></tr><tr><td>split-cookie</td><td>+19.52</td><td>+31.07</td><td>+61.43</td><td>6/8</td></tr><tr><td>Overall</td><td>+13.37</td><td>+16.82</td><td>+54.56</td><td>29/36</td></tr></table>

Notably, our approach does not rely on test-time adaptation or fine-tuning over a predefined set of queries. Instead, it employs a training-free query rewriting mechanism, demonstrating strong robustness when handling complex queries in previously unseen domains.

## Impact of Query Familiarity Level.

A central contribution of this work is the observation that reformulating a verbose descriptive query into a more familiar, concise form consistently improves retrieval performance. To quantify this, we evaluate each query at four abstraction levels: descriptive, Level 1, Level 2, and anchor. Table 3 reports the mean vIoU improvement of each level relative to the descriptive baseline.

As shown in Table 3, reformulating queries into more familiar, keywordgrounded forms improves vIoU in 29 of 36 cases. The anchor query consistently achieves the highest performance across all datasets.

![](images/95d893f8f76ebb2e9c891c2595ce7701e34a69ece48aa21aad7ec0b01b3f6f89.jpg)  
Fig. 2. Video feature heatmaps for the same frame across four query abstraction levels. Warmer colors indicate higher cosine similarity between video features and the query embedding.

As observed, Levels 1 and 2 do not always exhibit strict sequential improvement. We attribute this variance to the holistic encoding mechanism of visionlanguage models like CLIP. Because these models evaluate text compositionally, retaining grammatical structure while injecting multiple keywords (Level 2) can alter the syntactic context, occasionally causing the encoded semantic attention to difuse. This instability demonstrates that intermediate query shortening is insuficient, therefore directly motivates our Level 3 (Anchor) formulation. By stripping away extraneous syntax and reducing the query strictly to its core keyword components, Level 3 minimizes compositional noise and consistently achieves the most robust retrieval performance.

Qualitative Results. Figure 2 presents video feature heatmaps for the same frame across all four query levels. The original descriptive queries produce highly difused and noisy activation patterns that often spread into background regions. In contrast, rewritten queries produce more concentrated activations around the target object. Notably, the anchor-level queries generate tightly localized heatmaps with minimal background activation, demonstrating substantially improved semantic alignment between the query embedding and the learned Gaussian representation.

![](images/7bb68f30928d4b522f01e84c087558a716431f98ea7fe1ca27bb52463c5a332f.jpg)  
Fig. 3. Visualization of temporal localization intervals. We illustrate the frames classified as semantically active using horizontal bars. The original descriptive queries (orange) struggle with temporal coherence, resulting in fragmented activation intervals characterized by frequent misdetections and wrong detections due to peripheral linguistic noise. In contrast, our rewritten keyword-anchored queries (blue) yield stable, continuous activation segments that precisely align with the target dynamic states.

Figure 3 visualizes the temporal retrieval performance by representing active frames as horizontal interval bars. The original descriptive queries produce highly fragmented and unstable temporal predictions, with noticeable misdetections and false alarms. Conversely, the concise rewritten queries yield continuous, stable activation intervals that accurately span the target temporal boundaries.

## 5 Ablation Study

## 5.1 Efect of Query Length on Retrieval Performance

To further validate that query conciseness drives performance, we analyze the relationship between query length and retrieval metrics across all datasets.

Figure 4 groups queries into three categories: short (< 5 words), medium (5–15 words), and long (> 15 words).

The results indicate that long queries yield the worst performance among all categories. This degradation is primarily caused by the introduction of noisy contextual information that biases the embedding process.

In contrast, concise queries focus on the most salient object-centric attributes, thereby aligning more closely with the supervision paradigm used in 4D LangSplat.

## 5.2 Impact of Keyword Anchoring

We further analyze the importance of explicit keyword anchoring in the rewriting pipeline in Table 4.

![](images/f6935ced9b7330a2e88b0fe2412cd7f3166dc69657d8ea09c52a0d12dad509d4.jpg)

![](images/5ccd1f9e811bf006440d97abdf8089ec92aad9ad4bd52aed76836db12991806a.jpg)  
Fig. 4. Efect of query length on retrieval performance. Longer queries consistently produce lower performance.

Table 4. Comparison between rewriting with and without keyword anchoring.
<table><tr><td rowspan="2">Setting</td><td colspan="2">americano</td><td colspan="2">chickchicken</td><td colspan="2">split-cookie</td><td colspan="2">Average</td></tr><tr><td>Acc(%) vIoU(%)</td><td></td><td>)Acc(%) vIoU(%)</td><td></td><td>Acc(%) vIoU(%)</td><td></td><td>Acc(%) vIoU(%)</td><td></td></tr><tr><td>Unconstrained summarization</td><td>51.38</td><td>30.99</td><td>61.96</td><td>29.53</td><td>74.29</td><td>37.71</td><td>62.54</td><td>32.74</td></tr><tr><td>Anchored summarization</td><td>95.20</td><td>83.74</td><td>95.65</td><td>89.02</td><td>97.17</td><td>85.38</td><td>96.01</td><td>86.05</td></tr></table>

The results demonstrate that keyword anchoring plays a critical role in preserving the core semantics of the original query during rewriting. Even without explicit keywords, shortening the query already improves performance over the original 4D LangSplat pipeline, highlighting the general negative impact of verbose contextual descriptions. However, this simplistic unconstrained approach falls short of our proposed anchored method, as we demonstrate that injecting anchor keywords produces substantially larger gains, suggesting that concise semantic grounding is essential for robust retrieval in 4D Gaussians.

## 6 Conclusion

In this work, we investigated the impact of query rewriting for language-guided segmentation in 4D Gaussian representations. We demonstrated that verbose, narrative-style queries significantly degrade retrieval performance by introducing noisy contextual information that is poorly aligned with object-centric supervision.

To address this limitation, we proposed a simple yet efective training-free rewriting framework that progressively transforms descriptive queries into concise keyword-grounded formulations. Extensive experiments on HyperNeRF and Neu3D showed substantial improvements in both temporal localization and segmentation quality without requiring any fine-tuning or architectural modifications.

Our analysis further revealed that shorter, semantically anchored queries yield more stable video-feature similarity distributions and stronger alignment with target objects. These findings suggest that query formulation itself is a critical factor in language-guided 4D scene understanding.

In future work, we plan to investigate automatic keyword discovery, adaptive rewriting strategies, and more principled modeling of the relationship between linguistic familiarity and visual retrieval performance.

## References

1. Abe, K., Takeoka, K., Kato, M.P., Oyamada, M.: Llm-based query expansion fails for unfamiliar and ambiguous queries. In: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) (2025). https://doi.org/10.48550/arXiv.2505.12694, https://arxiv. org/abs/2505.12694, short Paper Track

2. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: Sam 3: Segment anything with concepts. In: International Conference on Learning Representations (ICLR) (2026)

3. Cheng, H.K., Oh, S.W., Price, B., Schwing, A., Lee, J.Y.: Tracking anything with decoupled video segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1316–1326 (2023)

4. Duan, Y., Wei, F., Dai, Q., He, Y., Chen, W., Chen, B.: 4d-rotor gaussian splatting: Towards eficient novel-view synthesis for dynamic scenes. In: Proc. SIGGRAPH (July 2024)

5. Fiebelman, G., Cohen, T., Morgenstern, A., Hedman, P., Averbuch-Elor, H.: 4-legs: 4d language embedded gaussian splatting. Computer Graphics Forum p. e70085 (2025)

6. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., et al.: The llama 3 herd of models (2024)

7. Ji, S., Wu, G., Fang, J., Cen, J., Yi, T., Liu, W., Tian, Q., Wang, X.: Segment any 4d gaussians (2024), https://arxiv.org/abs/2407.04504

8. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollar, P., Girshick, R.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4015–4026 (October 2023)

9. Lee, H., Min, J., Park, J.: Cf3: Compact and fast 3d feature fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27906–27916 (2025)

10. Li, T., Srikantha, S., Mehta, O., Fyfe, M., Xu, W., Yu, H., Kumar, S., Ahuja, N., Seidel, H.P., Bimo, B., Golan, A., Bespalov, D., Ahuja, S., Wang, Z., Wang, S., Yeung, Y., Saito, S.: Neural 3d video synthesis from multi-view video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2022)

11. Li, W., Zhao, Y., Qin, M., Liu, Y., Cai, Y., Gan, C., Pfister, H.: Langsplatv2: Highdimensional 3d language gaussian splatting with 450+ fps. Advances in Neural Information Processing Systems (2025), https://arxiv.org/abs/2507.07136

12. Li, W., Zhou, R., Zhou, J., Song, Y., Herter, J., Qin, M., Huang, G., Pfister, H.: 4d langsplat: 4d language gaussian splatting via multimodal large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22001–22011 (2025)

13. Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Wu, V., et al.: Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021)

14. Pichai, S., Hassabis, D., Kavukcuoglu, K.: A new era of intelligence with gemini 3 (Nov 2025), https://blog.google/products-and-platforms/products/gemini/ gemini-3/#build-anything

15. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20051–20060 (June 2024)

16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning (ICML). pp. 8748–8763. PMLR (2021)

17. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: Sam 2: Segment anything in images and videos (2024), https://arxiv.org/abs/2408.00714

18. Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F.: Improving text embeddings with large language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 11897–11916 (2024)

19. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

20. Wei, B.: Towards Semantic Editing of Volumetric Video: A Gaussian Splatting-Based 4D Segmentation Framework. Ph.D. thesis, Université de Rennes (2025)

21. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20310–20320 (June 2024)

22. Wu, J., Tao, J., Wang, H., Liu, G., Kompella, R.R., Yan, Y.: Orientation-anchored hyper-gaussian for 4d reconstruction from casual videos. In: Advances in Neural Information Processing Systems (NeurIPS) (2025), arXiv:2509.23492

23. Wu, X., Bai, Y., Li, M., Wu, X., Zhao, X., Lai, Z., Liu, W., Wang, X.: 4dlangvggt: 4d language-visual geometry grounded transformer (2025), https://arxiv.org/ abs/2512.05060

24. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21676–21685 (2024)