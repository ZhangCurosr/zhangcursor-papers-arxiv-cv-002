# HimmBO: High-Dimensional Interactive Model-Merging Bayesian Optimization for Image Generation

Jinyuan Mao<sup>1</sup>, Zhihao Shen<sup>2</sup>, and Chenxi Liu<sup>3†</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>pix.ai, <sup>3</sup>University of Victoria

![](images/13939f7ca1419ef48b58815a07223c7cd619e07da32a99a96fd16c30914d3dc8.jpg)  
Figure 1: HimmBO lets a user explore merges of a whole adapter library (N=1080 here) through preference feedback. Screening narrows the library to a small set ofpromising adapters using only precomputed images, so it requires no image generation during interaction (left). Refinement then searches how to combine those adapters (middle) and returns the preferred merge and its coefficients (right).

## Abstract

Fine-tuning has become a common way to customize text-to-image models, leading to community collections of over 100K lightweight adapters. Adapters sharing a base model can be merged with continuous coefficients. This turns a large collection into a high-dimensional continuous design space, which recent work explores interactively using preferential Bayesian optimization (PBO). However, existing approaches are limited to a few dozen preselected adapters and require expensive image generation in every interaction round. We introduce HimmBO, a two-stage method that scales preference-driven adapter merg ing to much larger collections while reducing image-generation cost. Its screening stage recovers the active adapter set through level-set estimation over pre-generated images, requiring no image generation. The refinement stage then optimizes merge coefficients in the recovered low-dimensional subspace. Both stages use a style-aware kernel, allowing preference information to transfer between visually similar adapters. In simulated-user image-matching experiments, HimmBO achieves higherfinal similarity than all comparisons while reaching a given similarity with under one tenth of the image generations required by prior work. Its performance remains stable as the collection grows to 1080 adapters, while competing methods degrade substantially CCS Concepts

• Computing methodologies → Image manipulation; • Human-centered computing → Interactive systems and tools;

## 1. Introduction

Fine-tuning has become a common way to customize text-to-image diffusion models for particular subjects, appearances, and artistic styles [HSW<sup>∗</sup>22, RLJ<sup>∗</sup>23, ZSY<sup>∗</sup>24]. The lightweight nature of parameter-efficient adaptations such as LoRA has also encouraged online communities to build and share large collections of finetuned models, which we refer to broadly as adapters. Beyond using adapters individually, users can also perform operations over multiple adapters to create new models and visual effects. One such operation, and the focus of this work, is adapter merging, where multiple adapters are combined with user-specified coefficients to produce a hybrid model that reflects their respective contributions [IRW<sup>∗</sup>23]. This turns a collection of N adapters into an N-dimensional continuous design space (Fig. 2). Even interfaces exposing only a handful of coefficients become difficult to operate as the number of parameters grows, with prior studies reporting substantial degradation beyond roughly 5–10 sliders [DMB22]. As community adapter collections continue to grow, with over 100K adapters on Civitai alone [LWT<sup>∗</sup>24,SZY25], directly exploring the resulting space becomes increasingly impractical.

Selecting such coefficients can be viewed as optimizing a black-box userpreference function over the adaptermerging space. GimmBO [LLJ26] recently showed that preferential Bayesian optimization (BO) can solve this problem interactively for roughly 20–30 candidate adapters within 20 interaction rounds. However, two practical limitations remain: each iteration requires computationally expensive image generation, and the candidate set must already be reduced to only a few dozen adapters. Unlike conventional BO applications with near-instantaneous evaluations [KSSI17, KSG20], image generation makes even 20 rounds costly in wall-clock time, from minutes to hours depending on hardware or service (inset). Obtaining such a small candidate set requires either manual curation or an external retrieval system [LWT<sup>∗</sup>24,SZY25], adding user effort in the former case and requiring a sufficiently accurate language description in the latter. Our goal is thus to make preference-driven exploration practical over much larger adapter collections while reducing the image-generation cost.

![](images/f4ad3139766f4c0a5241ad9d7abb9659365759ca27b1f93cb7ced97920848b90.jpg)

We introduce HimmBO, a preference-driven optimization method that scales well with candidate adapter set size while remaining image-generation efficient. HimmBO builds on GimmBO’s two-stage strategy, which runs BO over all adapter coefficients inside a capped simplex that keeps merges sparse, reads the active set off the best merge found so far, then restarts BO in the lower-dimensional space those adapters define [LLJ26]. At higher dimensions, this strategy becomes error-prone because the second refinement stage relies on the sparsity pattern of a single tentative optimum. Our key insight is that screening need not identify the optimum itself, but only reliably recover the active adapter set (Sec. 4.3.2). We explore several alternative Bayesian formulations for screening, including active learning (AL) and level-set estimation (LSE), which consider a broader portion of the preference landscape rather than concentrating on a single optimum. We ultimately select LSE, which most accurately recovers the active adapters by drawing on multiple promising samples (Sec. 5.1.3). Crucially, LSE also admits a pool-based setting in which candidates are selected from a fixed sample library rather than generated dynamically during optimization. This allows us to pre-generate the corresponding images as a gallery, eliminating image-generation wait time during screening (Sec. 4.3.1). Finally, different adapters can produce similar visual results, so preference for one can inform preference for another similar adapter. We capture this relationship with a style kernel to further improve performance (Sec. 4.2).

![](images/e43911ea4a053edd8caf64cf3f87d7356c98b6d9011c4b3db553ab1dfcc59d26.jpg)  
Figure 2: Both the selected adapters and coefficients determine the appearance of the merged image. Each row shows two adapters used separately (left, middle) and a merge ofthe two (right).

We evaluate HimmBO through quantitative simulated-user experiments on image-matching tasks, following GimmBO’s evaluation protocol [LLJ26] (Sec. 5). In addition to final image similarity and ground-truth active-adapter recovery, we evaluate scalability and image-generation cost. To test scalability, we increase the adapter collection size from N = 40 to 120, 360, 720, and 1080. HimmBO degrades gracefully with library size, reaching 0.889 similarity at N = 1080, only 0.018 below its 0.907 at N = 40. In contrast, the best alternative drops sharply from 0.903 at N = 40 to 0.828 at N = 1080, and GimmBO itself to 0.762. HimmBO is also substantially more generation-efficient, reaching 0.80 similarity with 6.8% of the image generations GimmBO needs to reach the same level. Together, these results extend preference-driven model merging from a small, pre-curated adapter set to collections approaching the scale of contemporary community model libraries.

## 2. Related Work

Adapter libraries and merging. Diffusion models generate images by gradually denoising random noise [HJA20, RBL<sup>∗</sup>22], with large open models such as SDXL [PEL<sup>∗</sup>24] serving as bases for much of the surrounding community ecosystem. Adapters are weight updates trained on top of such a base to add a subject or style [RLJ<sup>∗</sup>23, ZWH<sup>∗</sup>25, ZSY<sup>∗</sup>24], including popular parameter-efficient forms such as LoRA [HSW<sup>∗</sup>22]. Because they can be trained from only a handful of images and are cheap to store and share, community collections now contain over 100K adapters [LWT<sup>∗</sup>24, SZY25]. Adapters from the same base can be combined by adding their weight updates, following a long history of weight averaging across related models [IPG<sup>∗</sup>18, WIG<sup>∗</sup>22]. This works because fine-tuned models often remain in the same low-loss region, allowing their updates to be added and scaled [FDRC20,IRW<sup>∗</sup>23]. When updates interfere, more elaborate rules can reweight or sparsify them [MR22, $\mathrm { Y T C ^ { * } } 2 3 , \mathrm { Y Y Y ^ { * } } 2 4 ] .$ but simple linear merging is what community tools commonly implement and what we use as well (Sec. 3). In image generation, merging is usually applied to a small, fixed set of adapters to compose several concepts or combine a subject with a style [GWW<sup>∗</sup>23, PYAW24, DYY25, SRC<sup>∗</sup>24, FVSCO24, OLH25]. These methods assume the participating adapters are given in advance, with coefficients either fixed by the method or fitted to a predefined target. GimmBO instead treats the coefficients as a search problem, using user preference comparisons in place of an explicit target [LLJ26]. Its candidate set, however, is still small and chosen beforehand. Our work extends this interactive search from a small preselected set to an entire adapter library.

![](images/e23329378170d051558e152a7810166a3ea8d8e1e55b399e2d7088ae7a1ec0f8.jpg)  
Figure 3: Retrieval depends on a query specified before search begins. Here, “hand-drawn” confines Stylus [LWT<sup>∗</sup>24] to a narrow sketch-like cluster, while HimmBO lets the user interactively explore broader related styles.

Selecting adapters from a large collection. When a collection contains hundreds or thousands of adapters, a common first step is to retrieve a small subset relevant to the user’s query. Stylus retrieves adapters from a text query [LWT 24], while LoRAverse additionally promotes coverage across the collection [SZY25]. Other work retrieves directly from model weights, either using them as style descriptors [LTJ24] or aligning them with text for prompt-based retrieval [LDC<sup>∗</sup>25]. Similar ideas apply to language adapters, using a task loss or a few examples instead of a prompt [HLL<sup>∗</sup>24, ZGW<sup>∗</sup>24, CPFD23]. These methods choose the adapter set once from a query specified before search begins, which may be difficult for users who cannot yet state their preference explicitly. We therefore make adapter selection part of the interactive loop and compare against retrieval for the same task (Sec. 5).

Human-in-the-loop design optimization. When quality is subjective, users must remain in the loop, and prior work has explored interfaces such as sliders and galleries. Direct slider control becomes difficult beyond roughly five to ten parameters [DMB22], while systems that learn subjective quality from human judgments still operate below the dimensionality we target [KSI14, DAM<sup>∗</sup>19, SFP<sup>∗</sup>20]. Galleries instead present outputs rather than parameters, either arranging samples from the design space for browsing $\scriptstyle [ \mathbf { M A B } ^ { * } 9 7$ , NDM06, SSCO09] or selecting from a fixed image database based on the user’s choices [CMM<sup>∗</sup>00]. They can remain responsive by precomputing expensive results and reusing them during interaction [SSM15, SXZ<sup>∗</sup>17].

BO provides a sample-efficient backend for human-in-the-loop design [BBdF10, KSSI17, KSG20] and has been applied to domains such as melodies, fonts, and game levels [ZKGI20, KKR21, $\mathrm { T S D } ^ { * } 2 5 , \mathrm { K R L } ^ { * } 1 6 ]$ , alongside work on how users and optimization systems cooperate [KG22, MDC<sup>∗</sup>24, NYKU25]. These methods typically propose new parameter settings each round, so their outputs must be generated during interaction rather than fully precomputed. GimmBO falls into this category [LLJ26], and our refinement stage similarly returns to continuous BO once the candidate adapters have been reduced. However, screening has a different goal: rather than finding a single optimum, it must identify a set of promising adapters across the preference landscape. We use pool-based level-set estimation (LSE) for screening to achieve better active-adapter recovery while allowing all screening images to be precomputed (Sec. 4.3.1&5.1.3).

Goal-driven learning. The human-in-the-loop methods above are examples of goal-driven learning, where each query is chosen from the current surrogate to serve a particular objective [DFNM24, PAS26]. BO seeks a single best point [SKKS10], AL an accurate surrogate through uncertainty- or informationbased queries [Mac92,SWGO00,HHGL11,BHGL19], and LSE the set of candidates above a level [BNG<sup>∗</sup>05,GCHK13,BSKC16]. LSE therefore best suits our screening stage, which recovers a set of promising adapters rather than a single optimum (Sec. 5.1.3). With preferential rather than scalar feedback, latent utility can be modeled with a probit likelihood [CG05], with preferential Bayesian optimization defining acquisition rules on this model [GDDL17, ALBF23]. For batched queries, diversity may be implicit or explicitly encouraged, for example with determinantal point processes [KT12, KDK16]. Large adapter libraries also introduce high dimensionality, addressed by prior work through linear embeddings $[ \mathrm { W H Z ^ { * } 1 6 , R W B ^ { * } 2 0 , C K L ^ { * } 2 0 } ]$ , trust regions [DPLRSDP23], or sparse coordinate selection [EJ21,LFE<sup>∗</sup>23]. These strategies can ease search in a high-dimensional coefficient space, but its dimension still grows linearly with the number of adapters. We instead use a fixed-dimensional style space where nearby adapters share feedback (Sec. 3).

## 3. From a curated set to a library

Adapters fine-tuned from the same base model can be merged. Given a library of N adapters and a coefficient vector $\pmb { \alpha } \in \mathbb { R } ^ { N }$ that assigns each adapter a strength, a merging procedure combines the adapters’ weight updates into a single model, and that model generates images that mix the adapters’ visual effects. Throughout the paper we use the standard linear rule

$$
\begin{array} { r } { \mathbf { W } ( \mathbf { a } ) = \mathbf { W } _ { 0 } + \sum _ { i } \alpha _ { i } \Delta \mathbf { W } _ { i } , } \end{array}\tag{1}
$$

where $\mathbf { W } _ { 0 }$ is the base model and $\Delta \mathbf { W } _ { i }$ the ith adapter’s weight update [HSW<sup>∗</sup>22,LLJ26]. It is simple, effective, and what community tools implement. However, our method does not depend on this particular merging rule and can be applied seamlessly to alternatives.

GimmBO [LLJ26] turns the choice of α into an interactive loop: each round it generates a batch of candidate merges, the user ranks the images, the ranking updates a surrogate function of the user’s preference over α, and an acquisition function picks the next batch. This works well when the candidate set is small and curated, typically 20–30 adapters chosen beforehand. Public collections, however, contain thousands of adapters, so users must solve a more fundamental problem before tuning coefficients: deciding which adapters to use in the first place. We build on two observations to extend the same interactive loop to answer both questions.

![](images/8578a1199a642db5d68fc458f96e435739fa6a7e8b64f8ac398106bde4f7032a.jpg)  
Figure 4: Adapter identities do not reflect visual distance: independently trained adapters can be unrelated, related, or nearly identical in appearance. Example pairs are shown at their measured style similarities (top), with the full pairwise distribution across the library below.

## Preference has several peaks.

A user who merges style adapters rarely has a single look in mind: the same user may find a woodblock print, a photograph, and an oil painting all close to what they want (inset), producing several distinct peaks of preference across the library. A search for a single best point can lock onto

![](images/c3afc0e581bbe183d9b1c8e8b42abfe88832b3c432bf1067e4e3c0db3508468e.jpg)

whichever peak is favored by its first few comparisons and keep querying there, especially in the high-dimensional setting we target, where only a small fraction of the space can be explored. Such a miss is silent—a user who never sees a style cannot ask for more of it—and can exclude adapters needed for a final merge spanning several peaks. Unlike GimmBO, whose first screening stage relies on a single tentative optimum and often captures only one peak, our search seeks broader coverage by identifying a set of promising points spanning multiple peaks (Sec. 4).

Preference depends on style, not coefficients. Users judge generated images by appearance, and visually similar styles occur throughout large adapter libraries. Independently trained adapters can produce nearly indistinguishable looks (Fig. 4, top). Yet coefficient space does not capture this structure because each adapter occupies its own axis, making all pairs equally distant regardless of visual similarity and preventing a preference model over α from transferring feedback to lookalikes. The feedback required therefore grows with the number of adapters. We instead model preference in a style space measured from generated images, allowing feedback on one adapter to inform others with similar styles.

## 4. Method

Given a library of N adapters and a merging rule, our goal is to find the merge that best matches the user’s preference:

$$
\operatorname* { m a x } _ { \mathbf { \alpha } \mathbf { \in } \mathcal { A } } f \bigl ( I ( \mathbf { \alpha } \mathbf { \alpha } ) \bigr ) ,\tag{2}
$$

where $\mathcal { A } = \mathbb { R } ^ { N }$ is the N-dimensional design space, I(α) is the image generated by the merged model, and f is the user’s preference function, assigning each image a scalar utility reflected in the user’s choices. The adapters with nonzero coefficients form the merge’s active set, $S = \operatorname { s u p p } ( \mathbf { \alpha } \mathbf { \mathbf { \alpha } } \mathbf { \mathbf { \alpha } } \mathbf { \mathbf { \alpha } } )$ . In practice, merges tend to have small active sets [LLJ26], so we follow this observation and retain m adapters for the refinement stage. GimmBO [LLJ26] extracts the active set from the support of the best merge found after several PBO iterations (10 by default [LLJ26]). At library scale, however, relying on a single tentative optimum becomes unreliable: it can miss adapters from other promising regions, making active-set recovery the central challenge.

Our answer is a single preference posterior driving two stages, shaped by the two observations in Sec. 3. The screening stage searches the full library for a small active set covering styles the user may still prefer, rather than committing to a single tentative optimum. The refinement stage then searches merges within that set. Two design choices make this practical at library scale. A gallery generated once per library contains every image shown during screening, eliminating image generation during screening (Sec. 4.3.1). A style kernel lets feedback on one adapter inform visually similar adapters, so each batch of q images informs the full library (Sec. 4.2). Both stages share and update the same posterior, so all feedback carries from screening into refinement.

## 4.1. Style representation

Users judge generated images by their appearance, not by the coefficients that produce them, so we model the user’s preference as $f ( \mathbf { z } )$ , where z represents visual style rather than the merging coefficients directly. We obtain z through a style map Φ applied to the generated image (Fig. 5):

$$
\mathbf { z } = \Phi \big ( I ( \mathbf { \boldsymbol { \mathsf { a } } } ) \big ) \in \mathbb { R } ^ { d } .\tag{3}
$$

Thus, the coefficients affect preference only through the appearance of the image they produce. We construct Φ from perceptual image features by centering features within each prompt to suppress shared content and composition, then normalizing and projecting with PCA (details in App. A.2). The perceptual encoder is interchangeable, and we use DreamSim [FTS 23] by default (alternative encoders are compared in App. C). Its output dimension is fixed with respect to library size, and PCA further compresses it using a fixed variance threshold, yielding a small dimension d that changes little as N grows. During screening, gallery images are pregenerated, so z is measured directly with Φ. During refinement, z is predicted from α for acquisition and then measured directly once the proposed merge is generated (Sec. 4.3.3).

![](images/96033859050e2b8f3568407f2814c0a230ad40df69c8d773a5666760bab5ba8a.jpg)  
Figure 5: Each adapter in the public library is generated alone on the same four prompts to form the gallery. The style map encodes these images into a style vector, and the resulting style kernel captures visual similarity across the library.

## 4.2. Preferential surrogate

We estimate the user’s preference with a latent Gaussian-process surrogate over an input representation x, $\hat { f } \sim \mathcal G P ( 0 , k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) )$ whose posterior given the comparisons D below has mean µ and standard deviation σ. Because users provide choices rather than numeric scores, all feedback is converted into pairwise comparisons and accumulated in a single data set D. During screening, the user simply marks the preferred images per batch, producing comparisons between marked and unmarked images. During refinement, where only a small candidate set remains, the user instead ranks their top choices, which similarly yields pairwise comparisons. We infer latent utilities from these comparisons through a standard probit likelihood and approximate the posterior with Laplace approximation [CG05]. Kernel hyperparameters are re-fit each round by maximum a posteriori estimation. See Sec. 4.4 for details.

What should x represent? The conventional choice, also used by GimmBO, is the coefficient vector itself, x = α. This works for a small candidate set but becomes a bottleneck at library scale. Each adapter occupies its own axis, so feedback on one adapter cannot inform visually similar others, and the amount of user feedback needed grows rapidly with N. We instead represent each observed candidate by both its style feature and merging coefficients, x = [ z∥α ], and use the additive kernel

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { s } ^ { 2 } e ^ { - \lVert \mathbf { z } - \mathbf { z } ^ { \prime } \rVert ^ { 2 } / 2 \ell _ { s } ^ { 2 } } + \sigma _ { I } ^ { 2 } e ^ { - \lVert \mathbf { \mathbf { a } } - \mathbf { \bf { a } } ^ { \prime } \rVert ^ { 2 } / 2 \ell _ { I } ^ { 2 } } ,\tag{4}
$$

with channel variances $\sigma _ { s } ^ { 2 } , \sigma _ { I } ^ { 2 }$ and length scales $\ell _ { s } , \ell _ { I }$ . The style channel transfers feedback between visually similar adapters, allowing each batch to inform the model across the library. The coefficient channel preserves distinctions between different merges that produce similar styles. We use a sum so that similarity in either channel can transfer preference information.

We quantify the effect of the style kernel in a simulated-user experiment, measuring how many feedback rounds are needed before we recover half of the active adapters in the ground-truth merge. With a coefficient-only kernel, this feedback cost grows rapidly with li-

![](images/5bb64cb6ffa2719a9e7d5f368a3a88865e8eb1ccb89f5a324c9ea2a1d056f945.jpg)

brary size. With the style kernel, it remains nearly flat and is about 130× lower at N=1080 (experiment details in App. D).

## 4.3. Two-stage optimization

HimmBO separates the search into two stages with distinct goals (inset). Screening searches the full library to identify an active set of m adapters from the N candidates, while refinement continuously op-

![](images/471248bc881efa921a39d15a9c226618c3c38a61a696b3e96ca0a245db26f633.jpg)

timizes the merge coefficients within that set. The active set produced by screening defines the reduced search space for refinement. Both stages share and update the same posterior (Sec. 4.2), so feedback collected during screening carries into refinement.

## 4.3.1. Precomputed gallery

Given a coefficient vector α for an adapter library, we define its gallery as a set of images generated by the corresponding merged model (Fig. 5). As described in Sec. 4.1, these images also serve as inputs to the style map Φ, yielding the corresponding style feature z. Before any user interaction, we precompute galleries for a set of coefficient samples using the same fixed prompt set (details in App. A.2), associating each α with its gallery and resulting style feature z. Each such sample therefore provides the full surrogate input x = [ z∥α ] (Sec. 4.2). If subsequent interaction selects only among these samples, their gallery images can be shown directly, requiring no image generation during interaction.

## 4.3.2. Screening

The goal of screening is to deliver an active set of m adapters for refinement. Because refinement cannot recover adapters omitted from this set, screening must cover promising regions rather than identify a single optimum. We thus consider active learning (AL) and level-set estimation (LSE), which target broader regions than a single maximizer. Although developed separately from BO, all three can be viewed as goal-driven learning over a shared surrogate with different acquisition goals [DFNM24, PAS26, GCHK13] (Fig. 6).

![](images/ee8a031a16f975f4ea76c826f34dab0bc4fa36a732cd2f4cf44b94f7d9e9ce71.jpg)  
Figure 6: With the same preference landscape (higher is more preferred) and query budget (dots), BO focuses on one peak, AL spreads queries broadly, and LSE targets uncertain boundaries around multiple promising peaks. Boundary evidence can establish a peak as above the level, retaining unqueried adapters (arrows).

The three formulations differ in what they seek from the same preference posterior. Bayesian optimization (BO) seeks a maximizer, $\mathbf { x } ^ { \star } = \arg \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { X } } f ( \mathbf { x } )$ . AL instead seeks an accurate surrogate over the domain, commonly framed as reducing integrated posterior variance, $\textstyle \int _ { \mathcal { X } } \sigma ^ { 2 } ( \mathbf { x } ) d \mathbf { x }$ [DFNM24, PAS26]. LSE seeks the set of candidates whose utility exceeds a level [GCHK13],

$$
H ^ { \star } = \{ i \colon f ( { \mathbf { z } } _ { i } ) \geq h ^ { \star } \} ,\tag{5}
$$

where $h ^ { \star }$ is a quantile of the utilities rather than a fixed threshold, since preference provides no absolute utility scale.

Both AL and LSE admit a pool-based setting, where queries are selected from a fixed candidate set. We instantiate the pool with one unit-strength sample per adapter, with its gallery and style feature $\mathbf { z } _ { i }$ precomputed as in Sec. 4.3.1. This is particularly practical because adapter platforms already maintain representative galleries for individual adapters. Our pool reuses images that would already be generated and stored, adding no extra image-generation or gallerystorage cost. Each round can then score the full pool and show selected samples directly from their galleries, requiring no image generation during screening.

These objectives produce different query patterns (Fig. 6). BO can concentrate around an early peak, leaving others unexplored. This is especially problematic in high dimensions, where a limited budget may miss peaks containing adapters needed by the final merge. AL covers the landscape broadly but spends queries in lowutility regions the user has rejected. LSE instead concentrates on uncertain level-set membership around multiple promising peaks, retaining candidates estimated above the level. Both AL and LSE recover the active set more accurately than BO as the library grows; LSE performs best and is used in our method (Sec. 5.1.3).

Screening proceeds in rounds: the current posterior scores the candidate pool, q samples are shown from their precomputed galleries, and the user’s selections update the posterior. We initialize the first batch with one random candidate and choose the remaining q−1 by farthest-first traversal in style space for broad coverage. Empirically, this initialization has little effect on the final result.

For subsequent rounds, we score each pool candidate by uncertainty in its level-set membership using the straddle rule $[ \mathrm { B N G } ^ { * } 0 5 ] ,$

$$
a ( i ) = 1 . 9 6 \sigma _ { i } - | \mu _ { i } - \hat { h } | , \qquad \hat { h } = \mathrm { Q u a n t i l e } _ { \theta } \left( \left\{ \mu _ { j } \right\} \right) ,\tag{6}
$$

where i indexes the pool candidates and $\mu _ { i } , \sigma _ { i }$ are their posterior mean and standard deviation. We re-estimate h<sup>ˆ</sup> each round so that a fraction 1 − θ of the candidates lies above it, while 1.96 follows the original straddle rule and corresponds to a 95% posterior interval. Straddle is a special case of LSE [GCHK13] that favors candidates with uncertain level-set membership. Following LSE, we partition candidates into those confidently above, confidently below, and still uncertain, and query from the uncertain set. We use only the LSE components compatible with preferential feedback, leaving broader adaptation of LSE to this setting as future work (Sec. 6).

Each round selects the $q$ highest-scoring candidates, subject to a mutual-exclusion radius in style space so that the batch does not spend multiple queries on lookalikes. We set this radius to the style length scale $\ell _ { s }$ of Eq. (4), within which the posterior already treats two candidates as near-copies of one query.

When screening ends, we form the estimated set of candidates above the level, $\hat { H } = \left\{ i : \mu _ { i } \geq \hat { h } \right\}$ , and select m adapters from it for refinement. Because H<sup>ˆ</sup> can span multiple preference peaks, the m slots are shared across them so that promising regions retain representation. Through the style kernel, an adapter need not be shown directly to enter $\hat { H } \colon$ comparisons on visually similar adapters can raise its posterior mean above the level. Details are in App. A.3.

## 4.3.3. Refinement

Refinement turns the screened active set into a final merge by optimizing how its m adapters are combined. We search over

$$
\begin{array} { r } { \Delta _ { B } = \big \{ { \pmb { \alpha } } \in \mathbb { R } ^ { m } : \ : \alpha _ { i } \in \{ 0 \} \cup [ { \tau } , 1 ] , \ : \sum _ { i } \alpha _ { i } \leq B \big \} , } \end{array}\tag{7}
$$

a sparse, capped simplex following the practical constraints used by GimmBO [LLJ26]. We set $\tau = 0 .$ 1 to exclude coefficients with negligible visual effect, while the cap B limits the total merge strength to avoid degraded generations.

Refinement follows GimmBO’s interaction protocol, with our own modifications. Starting from the posterior learned during screening, each round we obtain q merges from $\Delta _ { B }$ , generate their images, ask the user to rank their top choices, and update the posterior. Because refinement ultimately seeks a single best merge, we return to the BO formulation and optimize

$$
\operatorname* { m a x } _ { \{ \mathbf { \alpha } _ { 1 } , . . . , \mathbf { \alpha } _ { q } \} } \quad \mathbb { E } _ { \Phi } \left[ \operatorname* { m a x } _ { j = 1 , . . . , q } \Phi ( \mathbf { x } _ { j } ( \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } _ { j } ) ) \right] + \gamma \log \operatorname* { d e t } \left( K ^ { \operatorname { d i v } } + { \epsilon } I \right) ,
$$

$$
{ \mathrm { s u b j e c t ~ t o : } } \quad \mathbf { \alpha } \mathbf { _ { { j } } } \in \Delta _ { B } { \mathrm { ~ f o r ~ } } j = 1 , \ldots , q ,\tag{8}
$$

(9)

where the first term is the standard expected-maximum batch acquisition [WMHD17], with φ chosen as UCB, $\Phi ( { \bf x } ) = \mu ( { \bf x } ) + { }$ λσ(x) [SKKS10], and the second is a diversity term with weight $\gamma = 0 . 0 2$ and $\epsilon = 1 0 ^ { - 3 }$ in all experiments. Because a proposed merge has no image yet, we approximate its style feature from the gallery. Motivated by the approximately additive visual effects of LoRA merging, we predict the style feature as $\begin{array} { r } { \hat { \bf z } ( { \bf a } ) = \sum _ { i \in \mathcal { S } } \alpha _ { i } { \bf z } _ { i } , } \end{array}$ summing the active adapters’ style features weighted by their coefficients (App. A.4). This gives the surrogate input $\mathbf { x } = [ \hat { \mathbf { z } } ( \mathbf { \pmb { \alpha } } ) \lVert \mathbf { \alpha } \mathbf { \alpha } ]$ used for acquisition. Once the merge is generated, its measured style feature z replaces $\hat { \mathbf { z } } ( \pmb { \alpha } )$ in the posterior update. We construct the diversity term following determinantal point processes [KT12], with $K _ { j k } ^ { \mathrm { d i v } } = e ^ { - \| \bar { \mathbf { z } } _ { j } - \bar { \mathbf { z } } _ { k } \| ^ { 2 } / 2 \ell _ { \mathrm { d i v } } ^ { 2 } }$ measuring similarity between normalized predicted style features $\bar { \mathbf { z } } _ { j } = \hat { \mathbf { z } } ( \mathbf { \boldsymbol { \mathsf { a } } } _ { j } ) / \| \hat { \mathbf { z } } ( \mathbf { \boldsymbol { \mathsf { a } } } _ { j } ) \|$ . The standard batch term already encourages separation through posterior covariance, but can still yield visually redundant candidates. We add this log-determinant term to spread each batch across style rather than coefficient axes, increasing diversity while leaving the session’s final similarity essentially unchanged (Fig. 8, right; Sec. 5.1.3).

Finally, we modify GimmBO’s stickbreaking parameterization to remove its strong bias toward the cap B. GimmBO parameterizes

![](images/d7b9a4be9a53bd53ca2e1a02bbc6d1f4d653d21a2db5cf847617dd79ff1fed85.jpg)

∆ with variables in the unit cube so that L-BFGS-B can be applied directly. Conceptually, it starts with a stick of length B, lets each of the m adapter coordinates take a fraction of the remaining stick, and drops the final remainder as a slack coordinate. With uniform cube samples, the expected remainder shrinks to $2 ^ { - m } B ,$ , so the retained coefficients have total ${ \mathbb E } [ \Sigma _ { i } { \mathbb { a } } _ { i } ] = ( 1 - 2 ^ { - m } ) B . \mathrm { A t } m = 8 ,$ this is within 1% of B, concentrating samples near the cap even though real community merges span substantially lower totals (inset, left). We instead make the stick length $S \le B$ itself a variable. During sampling, S is drawn from the empirical distribution of coefficient totals in public merges; during optimization, it becomes an (m+1)-th decision variable, allowing the total budget to be searched directly (inset, right). The floor τ of Eq. (7) is enforced within the same map. We use the same public usage statistics to define this distribution and to construct our test cases, but the two uses are independent (see App. A.4).

## 4.4. Implementation details

We implement HimmBO in Python with BoTorch $\left[ \mathrm { B K J } ^ { * } 2 0 \right]$ and GPyTorch [GPB<sup>∗</sup>18]. Screening returns $m = 8$ active adapters for refinement. We re-fit the kernel hyperparameters of Eq. (4) each round by maximum a posteriori estimation under log-normal priors centered on offline-calibrated values, and use a smaller exploration weight than GimmBO in refinement. The remaining constants, readout details, and timings are given in App. A.

As in GimmBO, all images during refinement are generated from a shared anchor image using SDEdit [MHS<sup>∗</sup>22], keeping content fixed so the user primarily compares style. In practice, optimization overhead is small relative to image generation. A screening round, including posterior fitting and scoring the full candidate pool, takes about 0.3 s, while refinement requires about 1 s for posterior updating and acquisition optimization. Thus, search computation adds negligible interaction delay.

## 5. Evaluations

We evaluate with simulated users on a standard matching task (Sec. 5.1) and demonstrate three extensions (Sec. 5.2).

## 5.1. Quantitative experiments

Task and protocol. We evaluate HimmBO with a matching task, a standard protocol for human-in-the-loop optimization [BBdF10, KSSI17, KSG20, LLJ26]. We construct 24 test target images with different prompts, each generated from a hidden merge of multiple library adapters with strengths $\textstyle \sum _ { i } \mathbf { \alpha } _ { i }$ drawn from the empirical distribution observed in community merges. For each ground-truth (GT) target, the simulated user judges the similarity of candidate images generated with the same prompt to the GT and provides only preference feedback to the method. We evaluate on each target over five random seeds. The simulated user evaluates each shown image by its similarity to $I _ { \mathrm { G T } }$ , using DreamSim [FTS<sup>∗</sup>23] as the preference function, $f ( I ( \pmb { \alpha } ) , I _ { \mathrm { G T } } ) = \mathrm { D r e a m S i m } ( I ( \pmb { \alpha } ) , I _ { \mathrm { G T } } )$ , following GimmBO’s simulated-user protocol [LLJ26]. During screening, we run $T _ { 1 } = 1 2$ rounds with $q { = } 8$ precomputed gallery images per round, and the user selects the $c { = } 4$ most similar. Because screening operates entirely on precomputed galleries, these 12 rounds collect preference feedback without any new image generation. During refinement, we run $T _ { 2 } { = } 8$ rounds, each generating $q { = } 8$ merges and showing them alongside the previous round’s best, from which the user ranks the top five. We set these budgets taking user-facing cost into account, rather than performance saturation alone: screening mainly costs interaction effort, whereas refinement also incurs waiting time for image generation. This gives 64 inferences per session, versus 165 in GimmBO’s published configuration. Because image generation dominates latency and all other computation takes only seconds on the CPU, the lower inference budget directly reduces the user’s waiting time. A session then takes about 15 minutes versus 37 for GimmBO on a single GPU.

Library. We curate SDXL-Style-1K, a library of 1,080 SDXL LoRA style adapters. We crawl public SDXL adapters on Civitai, filter them by tags and usage statistics, use a language model to verify that each changes style rather than content, and manually inspect standardized generations (App. B.1). We generate each adapter’s gallery once before interaction, using four fixed subjects with one image per subject. For scalability experiments, we form nested libraries of N=40, 120, 360, 720, and 1080 adapters, each containing all smaller libraries. Our main performance analysis uses the N=360 library, while Sec. 5.1.2 evaluates performance across library sizes. We will release the library manifest, curation pipeline, and test inputs.

Comparisons. We compare HimmBO against GimmBO, retrieval-based alternatives, and a slider-style baseline. We evaluate GimmBO directly on the test library using its published configuration. For retrieval, we use Stylus $\scriptstyle [ \mathrm { L W T } ^ { * } 2 4 ] .$ , which retrieves adapters from text. To adapt it to our image-matching task, we caption the target GT with a vision-language model and use that caption as the query, giving Stylus additional GT information unavailable to other methods. At each round, we rerun Stylus with a new random seed to produce a new image batch, from which the user keeps the best. We also include Stylus+GimmBO, following GimmBO’s retrieval extension [LLJ26]: Stylus retrieves the top 40 adapters, then GimmBO runs its full method on this reduced set. For the slider-style baseline, we include random coordinate descent, which adjusts one coefficient at a time over eight evenly spaced values. We omit other slider-style baselines because they already performed poorly at N=20 [LLJ26] and degrade further as N grows. GimmBO and random coordinate descent share the same five initial images and thus start from the same point; the remaining methods start from candidates produced by screening or retrieval. Stylus and random coordinate descent evaluate eight candidates per round to match HimmBO’s batch size, while GimmBO retains its published interaction protocol.

Simulated user experiments  
![](images/0120a9a990cd49547bf2aa440ca253a4866bbb41b8616e590b3fde2b30b153cb.jpg)

![](images/68197d5a1fb42a4f8341b753ea1a8aedfd7ef137c78cc5bc04cfc290a35e0f8f.jpg)

![](images/6b7214ca104c5ed8bfb21b00eb20a54e497390beb1a5e093040745abca06f919.jpg)

![](images/b335be21fb9d6c8141883f43bcecead92d015e32e7438eeb20440b6cfb301015.jpg)

![](images/606f1d20eab4eff4aae62308f5997a0c536ce8d239844bee0e753afbc83df537.jpg)

![](images/34ce58376e826db47dc85a7eb126d96fd1049b486b89079b45162adc947d1ea2.jpg)  
Figure 7: Across 24 targets andfive seeds, HimmBO improves through screening and refinement when measured by either interaction rounds or inferences at N=360 (left). Hollow screening points are diagnostic readouts excludedfrom the inference budget, and the ring marks the start ofrefinement. The two smaller panels report the number ofinferences needed to reach 0.80 similarity and the accuracy oftarget-adapter recovery (F1). Across nested libraries, HimmBO maintains high similarity as N grows while most accurately recovering the target adapters at every scale (right).

Metrics. Similarity is measured by DreamSim [FTS<sup>∗</sup>23] between the final generated image and the GT, following GimmBO [LLJ26]. Scores of 0.9 and 0.95 correspond roughly to a match and a nearexact match, respectively. Sparsity recovery is measured by the F1 score between the adapters active in the final merge and the GT’s true components [LLJ26]. It penalizes both missed and redundant adapters, distinguishing whether a method recovers the true components or instead matches the GT using visually similar alternatives. The style map uses DreamSim by default, but HimmBO still outperforms GimmBO when both it and the simulated user use other encoders, such as TPIPS [WNH 26] (App. C).

## 5.1.1. Simulated-user comparison

We set up the simulated user to see the hidden GT image and judge candidate images by DreamSim similarity, while the method receives only the resulting preference feedback, following GimmBO’s matching protocol. We analyze each session in two ways (Fig. 7, left). First, we plot similarity against feedback rounds, measuring the user interaction budget. To visualize screening progress, after each round we read out the current active set, generate one batch of q=8 merges using the refinement acquisition on the feedback so far, and plot the best result (hollow dots). These diagnostic generations are excluded from HimmBO’s inference budget. Second, we plot the same progress against inferences, which captures the main source of user waiting time. GimmBO evaluates efficiency in rounds, capturing interaction effort but not inference-induced waiting time. HimmBO separates these costs:

screening advances the round count using precomputed images without spending inferences, so the two axes differ.

Against rounds, HimmBO improves steadily through screening, passes GimmBO before screening ends, and overtakes Stylus+GimmBO shortly after refinement begins, reaching the highest final similarity. Stylus+GimmBO is the strongest alternative and benefits from GT-derived auto-captioning for retrieval. Stylus alone starts strongly but quickly plateaus around 0.80, while GimmBO improves more gradually and random coordinate descent remains lowest. Against inferences, the ordering is similar, but HimmBO’s advantage is clearer because screening requires no image generation. HimmBO reaches 0.80 after only 8 inferences, compared with 53 for Stylus+GimmBO and 117 for GimmBO; Stylus and random coordinate descent never reach this level. A HimmBO session stops after 64 inferences, and the dashed continuation shows further improvement under the same inference budget as the comparisons. Sparsity recovery further explains these differences. HimmBO recovers the most GT adapters, followed by Stylus+GimmBO, whose subsequent search cannot recover components missed by Stylus. Stylus alone recovers some GT components, whereas GimmBO recovers almost none despite moderate image similarity. Instead, GimmBO boosts visually similar alternatives to approach the target, but at N=360 still stalls short of a match. This reflects the stylespace observation of Sec. 3: visually similar adapters can produce similar images even when the recovered active set is wrong.

## 5.1.2. Library scale

We next test how each method scales with library size using the nested libraries from Sec. 5.1 (Fig. 7, right). Using the nested libraries in Fig. 7 (right), we increase N from 40 to 1080 while keeping every target unchanged, so each larger library adds only distractor adapters, isolating sensitivity to library size from changes in the matching task. At N=40, the scale GimmBO was designed for, HimmBO and GimmBO perform similarly, while Stylus+GimmBO coincides with GimmBO because retrieval returns the full library. As N grows, HimmBO maintains similar similarity, while Stylus+GimmBO degrades but remains the strongest alternative and GimmBO drops sharply. Stylus stays nearly flat across library sizes, while random coordinate descent remains low. Sparsity recovery explains these trends. HimmBO recovers the most GT adapters at every size, although recovery declines for the two largest libraries as screening becomes harder. Stylus+GimmBO again lies in between because retrieval removes many distractors, but any GT component it misses remains unavailable to GimmBO. GimmBO’s recovery falls to near zero as the library grows because its coefficient-space surrogate cannot transfer feedback between visually similar adapters, so additional adapters act mainly as distractors. Despite HimmBO’s lower recovery at larger N, its similarity drops much less because larger libraries also contain closer visual substitutes for missed GT components. Refinement can exploit these substitutes, a property we develop further in Sec. 5.2.

![](images/07b74bd4418727bafe6c631511b701cdbaed504ff8cd029808c125eb00fc61e7.jpg)

![](images/950b2ecfc8d6a3d9a48c376569c880b8300888e339b48b18978087fb562937df.jpg)

![](images/81a4abdce26c0c210c8f799964a81de011cb30676e7dbf3fca8502704929f2f5.jpg)  
Figure 8: Ablation of HimmBO’s screening and refinement components. For screening, we compare UCB, ALM [Mac92, SWGO00], and straddle [BNG 05]; straddle overtakes the alternatives within twelve rounds at N=360 and leadsfrom N=360 onward (left). For refinement, all variants use the same active set: w/o style removes the style kernel, fixed stick fixes total strength to B, w/o DPP removes diversity, and w/o screening data discards screening comparisons (middle). Fixed stick causes the largest similarity loss, while removing DPP mainly yields visually similar proposals (right).

## 5.1.3. Ablations

We ablate using our simulated matching task with N = 360 for our screening and refinement stages respectively.

Screening. Following Sec. 4.3.2, we compare UCB, ALM [Mac92, SWGO00], and straddle as representative acquisition rules for BO, AL, and LSE, respectively (Fig. 8). At N=360, UCB improves fastest initially but plateaus as it concentrates on an early high-preference region. ALM explores more broadly and improves more steadily. Straddle spends its early rounds resolving uncertain level-set membership, then overtakes both before our twelve-round screening budget ends and achieves the highest recovery. The dashed continuation beyond twelve rounds shows how these trends continue past the session budget. At N=40, all three rules perform similarly because twelve rounds expose much of the library, but their differences grow with library size and straddle leads from N=360 onward. This supports our use of LSE for screening: it spends several early rounds identifying promising regions, but becomes increasingly valuable once the library is too large to explore directly.

Refinement. We ablate four refinement components while fixing the screening result so every variant starts from the same active set. Removing the style kernel reduces similarity because feedback can no longer transfer between visually similar adapters. Fixing the stick length to B, as in GimmBO, causes the largest loss by concentrating search near the cap rather than covering the broader range of total strengths seen in community merges. Removing the diversity term has almost no effect on similarity, as intended, because it diversifies each batch rather than improving the optimum directly. Without it, coefficient-space exploration can produce different merges with nearly identical styles, while the diversity term spreads them in style space (Fig. 8, right). Finally, discarding screening comparisons also reduces similarity, showing that refinement benefits from carrying earlier feedback into the reduced search space.

## 5.2. Extensions

We explore three extensions of HimmBO. Each builds on the default setup of Sec. 5.1 and the style map’s ability to encode arbitrary images, with all other session parameters unchanged.

Styles the library does not hold. The targets in Sec. 5.1 are built from adapters in the library, but a user’s desired style may have no exact match. To test this, we remove one GT component from the library for each target and rerun the same session. HimmBO still approximates the target by substituting a visually similar adapter for the missing component (Fig. 9). How well this works depends on the substitute: when sufficiently similar, the final merge still reaches a rough match (DreamSim above 0.9). As the library grows, screening faces more distractors, but denser style coverage also makes closer substitutes more likely.

![](images/143e015c387ecd26bd19edf98b075ba3d36d73c9868474724fad7bc7548c4bab.jpg)

what the target is made of  
target  
with the component  
with its nearest style  
no adapters  
![](images/0b2367b5d51c65c96247bf781158c34c02299f78b3ea85adde522e72f1bcdda6.jpg)  
Figure 9: When a target component is missing from the library, HimmBO substitutes the nearest available style. As the library grows, closer substitutes become available (left; style cos is the DreamSim cosine between the missing adapter and its nearest neighbor); for four additional targets, removing one GT component still lets HimmBO recover close approximations by substituting visually similar adapters (right).

Personalization. A user may already have reference images that indicate their preferred style before interaction begins. We use these images to initialize the preference model by averaging their representations into a single target for a simulated user and collecting three rounds of screening feedback. The subsequent interaction is otherwise unchanged, except that the user completes only T<sub>1</sub> = 9 screening rounds instead of the default 12. The references therefore guide early discovery while leaving the full adapter library available for exploration (Fig. 10). Screening then proceeds normally, without restricting the search to the neighborhood of the references.

A gallery that grows with use. We initialize the screening pool with single-adapter samples because they reliably produce valid, high-quality images. Generic space-filling sampling over multiadapter coefficients can instead produce many poor-quality combinations, as observed by previous work [LLJ26]. Refinement provides a better source of multi-adapter samples: every delivered merge has already produced a valid image and been preferred by a user. Because its coefficients and style feature are also known, we can add it directly to the screening pool without training a new adapter or generating another image (Fig. 11). In a preliminary experiment, adding such merges increases the mean final similarity from 0.923 to 0.944 (App. E). Over time, these user-vetted multiadapter samples could improve future screening with little additional acquisition overhead.

## 6. Limitations and Future Work

Our evaluation uses linear merging only (Eq. (1)). Screening itself is independent of the merging rule because its gallery contains only single-adapter images, so alternative rules [MR22, YTC 23, YYY<sup>∗</sup>24] would affect refinement alone and are straightforward to explore in future work.

Our evaluation relies on simulated preference feedback. GimmBO’s user study found DreamSim to be a good proxy for human judgments in the same matching task [LLJ26], supporting its use for controlled comparisons here. We additionally conducted a small preliminary test with real users while measuring session cost (App. A.6); across these multi-round sessions, HimmBO found better merges than GimmBO while completing the interaction faster. Our simulated user, however, provides noise-free and internally consistent feedback, whereas real users may be inconsistent or change their preferences during a session. A larger user study is needed to measure how often this occurs and whether it causes screening to select the wrong adapters or refinement to become less stable. These effects could motivate preference models that are more robust to noisy or drifting feedback.

Our adaptation uses only part of the standard LSE formulation. Standard LSE assumes scalar Gaussian observations and intersects confidence intervals across rounds to bound a fixed latent function on a common scale [GCHK13]. Our utilities are instead inferred from comparisons and the posterior is refit each round, so confidence intervals from different rounds are not directly comparable. We therefore use straddle and the per-round level-set partition, but omit interval intersection and related components. Extending these ideas to preferential feedback, including more recent levelset methods, is an interesting direction for future work.

![](images/7af1cb3e5a5b34f9871d3ff73d0ab14fe1d69c83a68ceb953fe797e9acee831b.jpg)  
Figure 10: Reference images let HimmBO personalize screening before interaction begins. They steer early screeningfrom a broad coverage of the library (left) toward styles matching the user’s preference, without restricting subsequent exploration (right); two examples are shown.

the gallery  
what a session delivers  
![](images/6e311b058645231514ba48d6e291858b12aac2afb65bc09077a7011eb4f1be04.jpg)  
Figure 11: Each completed session can expand the screening pool. Starting from single-adapter gallery samples (left), HimmBO delivers a preferred multi-adapter merge together with its coefficients and style feature (right), allowing it to be added directly without additional image generation.

The style map suppresses content shared across gallery images but does not remove it completely, so a reference image with different content may retrieve the wrong style neighbors. A representation that separates style and content more explicitly could make personalization more reliable across subjects. The transition from screening to refinement is also one-way, so an adapter omitted from the active set cannot be recovered later. Revisiting the full library as refinement progresses could correct such misses and make the two stages interact more tightly. Finally, a gallery that retains delivered merges (Sec. 5.2) grows indefinitely. Developing policies for which samples to retain or retire could preserve the benefits of gallery growth while keeping acquisition efficient.

Midjourney’s Style Creator<sup>†</sup> offers related capabilities, but represents the resulting style as a position in its proprietary style space. Our approach instead searches an extensible library of adapters, returning the adapters and coefficients that compose the result while allowing the library itself to grow over time.

Adapters also inherit and can amplify the harms of generative image models. A large part of what Civitai hosts exists to sexualize women, and a session can start from any photograph, so the same search machinery can be applied to a real person who never consented to it. Searching a whole library can also make it easier to find and combine adapters that imitate a living artist, which artists are actively trying to prevent [SCW<sup>∗</sup>23]. We plan to release our library as a manifest and curation pipeline rather than redistribute adapter weights. This does not resolve these broader problems, but safer curation and access mechanisms for large model collections are important directions for future work.

## References

[ALBF23] ASTUDILLO R., LIN Z. J., BAKSHY E., FRAZIER P. I.: qEUBO: A decision-theoretic acquisition function for preferential Bayesian optimization. In Proceedings ofthe 26th International Conference on Artificial Intelligence and Statistics (AISTATS) (2023), vol. 206 of Proceedings ofMachine Learning Research, pp. 1093–1114. 3

[BBdF10] BROCHU E., BROCHU T., DE FREITAS N.: A Bayesian interactive optimization approach to procedural animation design. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA) (2010), pp. 103–112. 3, 7

[BHGL19] BINOIS M., HUANG J., GRAMACY R. B., LUDKOVSKI M.: Replication or exploration? sequential design for stochastic simulation experiments. Technometrics 61, 1 (2019), 7–23. 3

[BKJ<sup>∗</sup>20] BALANDAT M., KARRER B., JIANG D. R., DAULTON S., LETHAM B., WILSON A. G., BAKSHY E.: BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. In Advances in Neural Information Processing Systems (2020), vol. 33, pp. 21524–21538. 7

[BNG<sup>∗</sup>05] BRYAN B., NICHOL R. C., GENOVESE C. R., SCHNEIDER J., MILLER C. J., WASSERMAN L.: Active learning for identifying function threshold boundaries. In Advances in Neural Information Processing Systems (2005), vol. 18. 3, 6, 9

[BSKC16] BOGUNOVIC I., SCARLETT J., KRAUSE A., CEVHER V.: Truncated variance reduction: A unified approach to Bayesian optimization and level-set estimation. In Advances in Neural Information Processing Systems (2016), vol. 29. 3

[CG05] CHU W., GHAHRAMANI Z.: Preference learning with Gaussian processes. In Proceedings ofthe 22nd International Conference on Machine Learning (ICML) (2005), pp. 137–144. 3, 5

[CKL<sup>∗</sup>20] CHIU C.-H., KOYAMA Y., LAI Y.-C., IGARASHI T., YUE Y.: Human-in-the-loop differential subspace search in high-dimensional latent space. ACM Transactions on Graphics 39, 4 (2020), 85:1–85:15. 3

[CMM<sup>∗</sup>00] COX I. J., MILLER M. L., MINKA T. P., PAPATHOMAS T. V., YIANILOS P. N.: The bayesian image retrieval system, pichunter: Theory, implementation, and psychophysical experiments. IEEE Transactions on Image Processing 9, 1 (2000), 20–37. 3

[CPFD23] CHRONOPOULOU A., PETERS M. E., FRASER A., DODGE J.: Adaptersoup: Weight averaging to improve generalization of pretrained language models. In Findings of the Association for Computational Linguistics: EACL 2023 (2023), pp. 2054–2063. 3

[DAM<sup>∗</sup>19] DESAI R., ANDERSON F., MATEJKA J., COROS S., MC-CANN J., FITZMAURICE G., GROSSMAN T.: Geppetto: Enabling semantic design of expressive robot behaviors. In CHI Conference on Human Factors in Computing Systems (2019), pp. 1–14. 3

[DFNM24] DI FIORE F., NARDELLI M., MAININI L.: Active learning and bayesian optimization: A unified perspective to learn with a goal. Archives of Computational Methods in Engineering 31, 5 (2024), 2985– 3013. 3, 6

[DMB22] DANG H., MECKE L., BUSCHEK D.: Ganslider: How users control generative models for images using multiple sliders with and without feedforward information. In CHI Conference on Human Factors in Computing Systems (2022), pp. 1–15. 2, 3

[DPLRSDP23] DIOUANE Y., PICHENY V., LE RICHE R., SCOTTO DI PERROTOLO A.: TREGO: A trust-region framework for efficient global optimization. Journal ofGlobal Optimization 86, 1 (2023), 1–23. 3

[DYY25] DALVA Y., YESILTEPE H., YANARDAG P.: Lorashop: Training-free multi-concept image generation and editing with rectified flow transformers. In Advances in Neural Information Processing Systems (2025). 3

[EJ21] ERIKSSON D., JANKOWIAK M.: High-dimensional bayesian optimization with sparse axis-aligned subspaces. In Uncertainty in Artificial Intelligence (UAI) (2021), pp. 493–503. 3, 15

[FDRC20] FRANKLE J., DZIUGAITE G. K., ROY D., CARBIN M.: Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning (ICML) (2020), pp. 3259–3269. 3

[FTS<sup>∗</sup>23] FU S., TAMIR N., SUNDARAM S., CHAI L., ZHANG R., DEKEL T., ISOLA P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems (2023), vol. 36. 4, 7, 8

[FVSCO24] FRENKEL Y., VINKER Y., SHAMIR A., COHEN-OR D.: Implicit style-content separation using b-lora. In European Conference on Computer Vision (ECCV) (2024), pp. 181–198. 3

[GCHK13] GOTOVOS A., CASATI N., HITZ G., KRAUSE A.: Active learning for level set estimation. In Proceedings ofthe 23rd International Joint Conference on Artificial Intelligence (IJCAI) (2013), pp. 1344– 1350. 3, 6, 10, 16

[GDDL17] GONZÁLEZ J., DAI Z., DAMIANOU A., LAWRENCE N. D.: Preferential Bayesian optimization. In International Conference on Machine Learning (ICML) (2017), pp. 1282–1291. 3

[GPB<sup>∗</sup>18] GARDNER J. R., PLEISS G., BINDEL D., WEINBERGER K. Q., WILSON A. G.: GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Advances in Neural Information Processing Systems (2018), vol. 31. 7

[GWW<sup>∗</sup>23] GU Y., WANG X., WU J. Z., SHI Y., CHEN Y., FAN Z., XIAO W., ZHAO R., CHANG S., WU W., ET AL.: Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. In Advances in Neural Information Processing Systems (2023), vol. 36. 3

[HHGL11] HOULSBY N., HUSZÁR F., GHAHRAMANI Z., LENGYEL M.: Bayesian active learning for classification and preference learning, 2011. arXiv:1112.5745. 3

[HJA20] HO J., JAIN A., ABBEEL P.: Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (2020), vol. 33, pp. 6840–6851. 2

[HLL<sup>∗</sup>24] HUANG C., LIU Q., LIN B. Y., PANG T., DU C., LIN M.: Lorahub: Efficient cross-task generalization via dynamic lora composition. In Conference on Language Modeling (COLM) (2024). 3

[HSW<sup>∗</sup>22] HU E. J., SHEN Y., WALLIS P., ALLEN-ZHU Z., LI Y., WANG S., WANG L., CHEN W.: LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR) (2022). 2, 3

[IPG<sup>∗</sup>18] IZMAILOV P., PODOPRIKHIN D., GARIPOV T., VETROV D., WILSON A. G.: Averaging weights leads to wider optima and better generalization. In Conference on Uncertainty in Artificial Intelligence (UAI) (2018), pp. 876–885. 2

[IRW<sup>∗</sup>23] ILHARCO G., RIBEIRO M. T., WORTSMAN M., GURURAN-GAN S., SCHMIDT L., HAJISHIRZI H., FARHADI A.: Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations (ICLR) (2023). 2, 3

[KDK16] KATHURIA T., DESHPANDE A., KOHLI P.: Batched gaussian process bandit optimization via determinantal point processes. In Advances in Neural Information Processing Systems (2016), vol. 29. 3

[KG22] KOYAMA Y., GOTO M.: Bo as assistant: Using bayesian optimization for asynchronously generating design suggestions. In ACM Symposium on User Interface Software and Technology (UIST) (2022), pp. 1–14. 3

[KKR21] KADNER F., KELLER Y., ROTHKOPF C.: Adaptifont: Increasing individuals’ reading speed with a generative font model and bayesian optimization. In CHI Conference on Human Factors in Computing Systems (2021), pp. 1–11. 3

[KRL<sup>∗</sup>16] KHAJAH M. M., ROADS B. D., LINDSEY R. V., LIU Y.-E., MOZER M. C.: Designing engaging games using bayesian optimization. In CHI Conference on Human Factors in Computing Systems (2016), pp. 5571–5582. 3

[KSG20] KOYAMA Y., SATO I., GOTO M.: Sequential gallery for interactive visual design optimization. ACM Transactions on Graphics 39, 4 (2020), 88:1–88:12. 2, 3, 7

[KSI14] KOYAMA Y., SAKAMOTO D., IGARASHI T.: Crowd-powered parameter analysis for visual design exploration. In ACM Symposium on User Interface Software and Technology (UIST) (2014), pp. 65–74. 3

[KSSI17] KOYAMA Y., SATO I., SAKAMOTO D., IGARASHI T.: Sequential line search for efficient visual design optimization by crowds. ACM Transactions on Graphics 36, 4 (2017), 48:1–48:11. 2, 3, 7

[KT12] KULESZA A., TASKAR B.: Determinantal point processes for machine learning. Foundations and Trends in Machine Learning 5, 2–3 (2012), 123–286. 3, 7

[LDC<sup>∗</sup>25] LI Z., DUAN Z., CHEN D., CHEN C., CHEN D., LI Y., CHEN Y.: Autolora: Automatic lora retrieval and fine-grained gated fusion for text-to-image generation, 2025. arXiv:2508.02107. 3

[LFE<sup>∗</sup>23] LIU S., FENG Q., ERIKSSON D., LETHAM B., BAKSHY E.: Sparse bayesian optimization. In International Conference on Artificial Intelligence and Statistics (AISTATS) (2023), pp. 3754–3774. 3

[LLJ26] LIU C., LING S., JACOBSON A.: Gimmbo: Interactive generative image model merging via bayesian optimization. In ACM SIG-GRAPH 2026 Conference Proceedings (2026). 2, 3, 4, 6, 7, 8, 10, 17, 18

[LTJ24] LIU C., TAKIKAWA T., JACOBSON A.: A lora is worth a thousand pictures, 2024. arXiv:2412.12048. 3

[LWT<sup>∗</sup>24] LUO M., WONG J., TRABUCCO B., HUANG Y., GONZALEZ J. E., CHEN Z., SALAKHUTDINOV R., STOICA I.: Stylus: Automatic adapter selection for diffusion models. In Advances in Neural Information Processing Systems (2024), vol. 37. 2, 3, 7, 18

[MAB<sup>∗</sup>97] MARKS J., ANDALMAN B., BEARDSLEY P. A., FREEMAN W., GIBSON S., HODGINS J., KANG T., MIRTICH B., PFISTER H., RUML W., RYALL K., SEIMS J., SHIEBER S.: Design galleries: A general approach to setting parameters for computer graphics and animation. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH) (1997), pp. 389–400. 3

[Mac92] MACKAY D. J. C.: Information-based objective functions for active data selection. Neural Computation 4, 4 (1992), 590–604. 3, 9, 19

[MDC<sup>∗</sup>24] MO G., DUDLEY J., CHAN L., LIAO Y.-C., OULASVIRTA A., KRISTENSSON P. O.: Cooperative multi-objective bayesian design optimization. ACM Transactions on Interactive Intelligent Systems 14, 2 (2024), 1–28. 3

[MHS<sup>∗</sup>22] MENG C., HE Y., SONG Y., SONG J., WU J., ZHU J.-Y., ERMON S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations (ICLR) (2022). 7, 17

[MR22] MATENA M., RAFFEL C.: Merging models with fisher-weighted averaging. In Advances in Neural Information Processing Systems (2022), vol. 35. 3, 10

[NDM06] NGAN A., DURAND F., MATUSIK W.: Image-driven navigation of analytical BRDF models. In Eurographics Symposium on Rendering (EGSR) (2006), pp. 399–407. 3

[NYKU25] NIWA R., YOSHIDA S., KOYAMA Y., USHIKU Y.: Cooperative design optimization through natural language interaction. In ACM Symposium on User Interface Software and Technology (UIST) (2025), pp. 1–25. 3

[OLH25] OUYANG Z., LI Z., HOU Q.: K-lora: Unlocking training-free fusion of any subject and style loras. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025), pp. 13041–13050. 3

[PAS26] POLKE D., AHLE E., SÖFFKER D.: Adaptive learning with gaussian process regression: A comprehensive review of methods and applications. Machine Learning and Knowledge Extraction 8, 4 (2026), 101. 3, 6

[PEL<sup>∗</sup>24] PODELL D., ENGLISH Z., LACEY K., BLATTMANN A., DOCKHORN T., MÜLLER J., PENNA J., ROMBACH R.: SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR) (2024). 2, 17

[PYAW24] PO R., YANG G., ABERMAN K., WETZSTEIN G.: Orthogonal adaptation for modular customization of diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), pp. 7964–7973. 3

[RBL<sup>∗</sup>22] ROMBACH R., BLATTMANN A., LORENZ D., ESSER P., OMMER B.: High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), pp. 10684–10695. 2

[RKH<sup>∗</sup>21] RADFORD A., KIM J. W., HALLACY C., RAMESH A., GOH G., AGARWAL S., SASTRY G., ASKELL A., MISHKIN P., CLARK J., KRUEGER G., SUTSKEVER I.: Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML) (2021). 18

[RLJ<sup>∗</sup>23] RUIZ N., LI Y., JAMPANI V., PRITCH Y., RUBINSTEIN M., ABERMAN K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), pp. 22500–22510. 2

[RWB<sup>∗</sup>20] RAPONI E., WANG H., BUJNY M., BORIA S., DOERR C.: High dimensional bayesian optimization assisted by principal component analysis. In International Conference on Parallel Problem Solving from Nature (PPSN) (2020), pp. 169–183. 3

[SCW<sup>∗</sup>23] SHAN S., CRYAN J., WENGER E., ZHENG H., HANOCKA R., ZHAO B. Y.: Glaze: Protecting artists from style mimicry by text-toimage models. In USENIX Security Symposium (2023), pp. 2187–2204. 11

[SFP<sup>∗</sup>20] SHIMIZU E., FISHER M., PARIS S., MCCANN J., FATA-HALIAN K.: Design adjectives: A framework for interactive modelguided exploration of parameterized design spaces. In ACM Symposium on User Interface Software and Technology (UIST) (2020), pp. 261–278. 3

[SGG<sup>∗</sup>24] SOMEPALLI G., GUPTA A., GUPTA K., PALTA S., GOLD-BLUM M., GEIPING J., SHRIVASTAVA A., GOLDSTEIN T.: Investigating style similarity in diffusion models. In European Conference on Computer Vision (ECCV) (2024), pp. 143–160. 18

[SKKS10] SRINIVAS N., KRAUSE A., KAKADE S. M., SEEGER M. W.: Gaussian process optimization in the bandit setting: No regret and experimental design. In International Conference on Machine Learning (ICML) (2010), pp. 1015–1022. 3, 6

[SRC<sup>∗</sup>24] SHAH V., RUIZ N., COLE F., LU E., LAZEBNIK S., LI Y., JAMPANI V.: Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision (ECCV) (2024), pp. 422–438. 3

[SSCO09] SHAPIRA L., SHAMIR A., COHEN-OR D.: Image appearance exploration by model-based navigation. Computer Graphics Forum 28, 2 (2009), 629–638. 3

[SSM15] SHUGRINA M., SHAMIR A., MATUSIK W.: Fab forms: Customizable objects for fabrication with validity and geometry caching. ACM Transactions on Graphics 34, 4 (2015), 100:1–100:12. 3

[SWGO00] SEO S., WALLAT M., GRAEPEL T., OBERMAYER K.: Gaussian process regression: Active data selection and test point rejection. In IEEE-INNS-ENNS International Joint Conference on Neural Networks (IJCNN) (2000), vol. 3, pp. 241–246. 3, 9

[SXZ<sup>∗</sup>17] SCHULZ A., XU J., ZHU B., ZHENG C., GRINSPUN E., MA-TUSIK W.: Interactive design space exploration and optimization for CAD models. ACM Transactions on Graphics 36, 4 (2017), 157:1– 157:14. 3

[SZY25] SONMEZER M., ZHENG M., YANARDAG P.: Loraverse: A submodular framework to retrieve diverse adapters for diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV) (2025), pp. 17879–17888. 2, 3

[TSD<sup>∗</sup>25] TATSUKAWA Y., SHEN I.-C., DOGAN M. D., QI A., KOYAMA Y., SHAMIR A., IGARASHI T.: Fontcraft: Multimodal font design using interactive bayesian optimization. In CHI Conference on Human Factors in Computing Systems (2025), pp. 1–14. 3

[WHZ<sup>∗</sup>16] WANG Z., HUTTER F., ZOGHI M., MATHESON D., DE FRE-ITAS N.: Bayesian optimization in a billion dimensions via random embeddings. Journal of Artificial Intelligence Research 55 (2016), 361– 387. 3

[WIG<sup>∗</sup>22] WORTSMAN M., ILHARCO G., GADRE S. Y., ROELOFS R., GONTIJO-LOPES R., MORCOS A. S., NAMKOONG H., FARHADI A., CARMON Y., KORNBLITH S., SCHMIDT L.: Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning (ICML) (2022), pp. 23965–23998. 2

[WMHD17] WILSON J. T., MORICONI R., HUTTER F., DEISENROTH M. P.: The reparameterization trick for acquisition functions. In NeurIPS Workshop on Bayesian Optimization (2017). 6

[WNH<sup>∗</sup>26] WANG S.-Y., NITZAN Y., HERTZMANN A., ZHU J.-Y., SHECHTMAN E., EFROS A. A., ZHANG R.: The many senses of visual similarity: A text-prompted image perceptual metric, 2026. arXiv: 2607.18237. 8, 18

[XLZ<sup>∗</sup>24] XIAO S., LIU Z., ZHANG P., MUENNIGHOFF N., LIAN D., NIE J.-Y.: C-pack: Packed resources for general chinese embeddings. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (2024), pp. 641–649. 18

[YTC<sup>∗</sup>23] YADAV P., TAM D., CHOSHEN L., RAFFEL C. A., BANSAL M.: TIES-merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems (2023), vol. 36. 3, 10

[YYY<sup>∗</sup>24] YU L., YU B., YU H., HUANG F., LI Y.: Language models are super mario: Absorbing abilities from homologous models as a free lunch. In International Conference on Machine Learning (ICML) (2024). 3, 10

[ZGW<sup>∗</sup>24] ZHAO Z., GAN L., WANG G., ZHOU W., YANG H., KUANG K., WU F.: Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild. In Findings ofthe Associationfor Computational Linguistics: ACL 2024 (2024), pp. 4447–4462. 3

[ZKGI20] ZHOU Y., KOYAMA Y., GOTO M., IGARASHI T.: Generative melody composition with human-in-the-loop bayesian optimization, 2020. arXiv:2010.03190. 3

[ZSY<sup>∗</sup>24] ZHENG M., SIMSAR E., YESILTEPE H., TOMBARI F., SI-MON J., YANARDAG P.: Stylebreeder: Exploring and democratizing artistic styles through text-to-image models. In Advances in Neural Information Processing Systems (2024), vol. 37. 2

[ZWH<sup>∗</sup>25] ZHANG X., WEI X., HU W., WU J., WU J., ZHANG W., ZHANG Z., LEI Z., LI Q.: A survey on personalized content synthesis with diffusion models. Machine Intelligence Research 22, 5 (2025), 817– 848. 2

## Supplemental Material

## Appendix A: Implementation details

## A.1. Surrogate hyperparameters and priors

The kernel in Eq. (4) has four hyperparameters: the channel variances $\sigma _ { s } ^ { 2 } , \sigma _ { I } ^ { 2 }$ and length scales $\ell _ { s } , \ell _ { I } ,$ with one length scale shared across each channel. We re-fit these hyperparameters each round by maximum a posteriori estimation using all comparisons collected so far. We fix the probit noise to one because the likelihood depends only on the ratio between utility differences and noise, and this scale is already controlled by the channel variances. Each hyperparameter has a log-normal prior centered at a value calibrated offline on simulated sessions separate from the evaluation, with standard deviation 0.5 in log space; nearby values perform similarly. This prior is important because, when comparisons are largely consistent, the likelihood continues to increase as the latent utilities are scaled up and therefore has no finite maximum. Without the prior, the hyperparameters drift to extreme values and search performance degrades, while fixing them at their calibrated values also performs worse. The prior center for $\sigma _ { s } ^ { 2 }$ is larger during screening than refinement (Table S1), likely because screening searches the full library rather than the m active adapters and therefore benefits from stronger style guidance.

Unlike GimmBO, we do not use a sparsity-inducing prior such as SAAS [EJ21], which assigns a separate length scale to each input dimension and shrinks most dimensions. Our style representation is already compact after PCA, and we instead use a single shared length scale per channel rather than selecting individual feature dimensions. This also reduces each hyperparameter re-fit from about ten seconds to under one second.

In refinement, we use a smaller exploration weight than GimmBO (λ = 1 versus 9) and add the diversity term with weight γ. Because screening has already explored the full library, refinement searches only merges of the m active adapters and requires less exploration. Empirically, the smaller exploration weight also improves performance. All constants are listed in Table S1.

## A.2. Offline precomputation

The gallery. Each adapter is generated alone at unit strength $( \alpha _ { i } = 1 )$ on four fixed prompts, covering a person, an animal, a still life, and a landscape, so that each style is observed across different subject types. Because all adapters use the same prompts, differences across their galleries primarily reflect the adapters’ effects. For our experiments, we use single adapters rather than merges because platforms already publish and preview adapters this way. Sec. 5.2 explores extending the gallery with user-vetted multiadapter merges collected during refinement.

The style map below maps each gallery image to a style coordinate. For a given session prompt, adapters are compared using the coordinates of their corresponding gallery images. We denote the coordinate of adapter i by $\mathbf { z } _ { i }$ . The gallery stores the screening images and, for each prompt, an N × d coordinate matrix used by the posterior. The images dominate storage, while all session-time computation uses only these matrices.

Building the gallery requires four inferences per adapter, paid once per library: about 4 GPU-hours at N=1080, shared by all sessions and users. Adding a new adapter requires only generating its gallery images and mapping them through the existing style map. The style kernel matrix then gains one row and column, and the median used to set $\ell _ { s }$ is recomputed in seconds.

Table S1: HimmBO’s hyperparameters. Kernel hyperparameters are re-fit every round (Sec. A.1), so the table reports their prior centers.
<table><tr><td>Symbol</td><td>Meaning</td><td>Value</td></tr><tr><td> $q$ </td><td>images per batch</td><td>8</td></tr><tr><td> $T _ { 1 }$ </td><td>screening rounds</td><td>12</td></tr><tr><td> $T _ { 2 }$ </td><td>refinement rounds</td><td>8</td></tr><tr><td> $m$ </td><td>size of the active set</td><td>8</td></tr><tr><td> $B$ </td><td>total-strength cap,  $\mathrm { E q . } ( 7 )$ </td><td>2</td></tr><tr><td> $\boldsymbol { \tau }$ </td><td>per-adapter floor, Eq. (7)</td><td>0.1</td></tr><tr><td> $\theta$ </td><td>quantile defining the level , Eq. (6)</td><td>0.5</td></tr><tr><td> $\rho _ { \ell }$ </td><td> $\ell _ { s }$  as a multiple of the median pairwise</td><td>0.5</td></tr><tr><td> $\sigma _ { s } ^ { 2 }$ </td><td>style distance style-channel variance, Eq. (4); screen- ing / refinement</td><td>16 /4</td></tr><tr><td> $\sigma _ { I } ^ { 2 }$ </td><td>coefficient-channel variance, Eq. (4)</td><td>0.5</td></tr><tr><td> $\ell _ { I }$ </td><td>coefficient-channel length scale, Eq. (4)</td><td>0.5</td></tr><tr><td> $\gimel$ </td><td>UCB exploration weight</td><td>1</td></tr><tr><td> $\gamma$ </td><td>diversity weight, Eq. (8)</td><td>0.02</td></tr><tr><td> $t _ { 0 }$ </td><td>SDEdit strength</td><td>0.8</td></tr><tr><td> $\ell _ { \mathrm { d i v } }$ </td><td>length scale of the diversity kernel</td><td>0.5</td></tr><tr><td> $\varepsilon$ </td><td>jitter in the log-determinant</td><td>10⁻³</td></tr></table>

Table S2: Final similarity under different PCA variance thresholds, in the main-experiment setting (N=360).
<table><tr><td>Explained variance</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td></tr><tr><td> $d$ </td><td>20</td><td>32</td><td>53</td><td>90</td><td>159</td></tr><tr><td>Similarity</td><td>0.893</td><td>0.910</td><td>0.901</td><td>0.925</td><td>0.917</td></tr></table>

The style map. Because all adapters share the same four prompts, images generated from a given prompt share much of their content. The style map Φ centers the features within each prompt, normalizes them, and reduces their dimension with PCA,

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { p } = \Phi \big ( I _ { i } ^ { p } \big ) = \mathrm { P C A } \Big ( \mathrm { n o r m a l i z e } \left( \Phi ( I _ { i } ^ { p } ) - \frac { 1 } { N } \sum _ { j } \Phi ( I _ { j } ^ { p } ) \right) \Big ) , } \end{array}\tag{S1}
$$

where ϕ is a perceptual image encoder, DreamSim by default (other encoders are compared in App. C). For each test case, we use the gallery prompt whose subject type best matches the test prompt (e.g., the animal prompt for an animal target) and use its style coordinates throughout the session.

Centering subtracts the mean feature for each prompt, which primarily captures shared subject and composition, emphasizing the adapters’ contributions. This step is essential: without it, nearestneighbor cosine similarities are all close to one, and the distancebased components of our method (the style kernel, exclusion radius, and peak readout) can no longer distinguish adapters. Normalization projects each feature onto the unit sphere so that distances reflect style rather than feature magnitude. PCA then retains enough components to explain 80% of the library variance, giving d=90 at N=360. Retaining more dimensions provides no gain, while retaining fewer reduces similarity (Table S2).

We center the prior for the style length scale relative to the library, $\ell _ { s } = \rho _ { \ell }$ median $i \neq j \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \|$ (Table S1), so it does not require retuning as the library changes.

## A.3. Screening

Adapting LSE. Following LSE [GCHK13], we partition adapters into three sets each round using their posterior intervals. Let $Q _ { t } ( i ) = [ \mu _ { t , i } - \beta \sigma _ { t , i } , \mu _ { t , i } + \beta \sigma _ { t , i } ]$ denote the interval for adapter i at round t, with $\beta = 1 . 9 6$ as in Eq. (6). Adapters whose intervals lie entirely above or below the level $\hat { h } _ { t }$ form

$$
H _ { t } = \{ i : \mu _ { t , i } - \beta \sigma _ { t , i } \geq \hat { h } _ { t } \} , \qquad L _ { t } = \{ i : \mu _ { t , i } + \beta \sigma _ { t , i } \leq \hat { h } _ { t } \} ,\tag{S2}
$$

and the remaining adapters, whose intervals contain $\hat { h } _ { t } .$ , form the uncertain set $U _ { t }$ . The straddle score in Eq. (6) implements this partition directly: $a _ { t } ( i ) > 0$ exactly when $\mathsf { \beta } \pmb { \sigma } _ { t , i } > \vert \mu _ { t , i } - \hat { h } _ { t } \vert$ , so we select each batch only from $U _ { t }$ . Early in a session, unexplored adapters have posterior means near the level and large uncertainty, placing them in $U _ { t }$ and causing them to be queried early. Because the three sets are rebuilt from the current posterior each round, an adapter can leave $H _ { t }$ or $L _ { t }$ as later comparisons shift its posterior mean, allowing earlier classifications to be revised as more feedback arrives.

We omit the remaining components of the standard LSE formulation [GCHK13], whose assumptions do not carry over to preferential feedback. Standard LSE observes noisy scalar values of a fixed function $f$ under a Gaussian likelihood with fixed kernel hyperparameters, yielding an exact posterior whose intervals across rounds bound the same $f$ on a common scale. Our posterior is instead inferred from comparisons through a Laplace approximation, with hyperparameters re-fit every round, so its utility scale can change across rounds. Standard LSE intersects intervals across rounds and schedules $\beta$ so that they jointly cover $f .$ Because our intervals are not comparable across rounds, we keep β fixed, do not intersect intervals, and allow classified adapters to be reconsidered. Its batch variant also spreads queries by updating the variance as if pending observations were already available, which relies on Gaussian observations. We instead separate each batch using the style radius of Sec. 4.3.2. Finally, standard LSE stops once the classification is ε-accurate, whereas we stop after a fixed budget of $T _ { 1 }$ rounds to bound user interaction time.

Determining the active set. After $T _ { 1 }$ rounds, a few hundred comparisons over N adapters leave substantial posterior uncertainty for many candidates. Because $H _ { T _ { 1 } }$ includes only adapters whose entire posterior interval lies above the level, it contains only those already classified with high confidence. We therefore read out the active set from the estimated level set H<sup>ˆ</sup> of Sec. 4.3.2, which also includes adapters whose posterior mean is above the level but whose membership remains uncertain. Because preference often has several peaks in style space (Sec. 3), we allocate the m slots across peaks so that the active set represents multiple promising regions rather than containing m lookalikes from the highest one (Fig. S1).

![](images/5c1b1dcef0376c48ed2e99f11de587e6a2b166d0d9a18c5699be3588a810e365.jpg)  
Figure S1: Plain top-m spends most ofthe $m { = } 8$ slots on the highest peak and leaves one peak without a slot (left), while the peak readout shares the slots across peaks (right). Contours show the posterior mean, and numbers give the order in which the slots are filled.

Table S3: F1 of the active set under different separation radii (10 seeds).
<table><tr><td>Radius  $/ \ell _ { s }$ </td><td>0.75</td><td>1.0</td><td>1.25</td><td>1.5</td><td>1.75</td><td>2.0</td><td>2.5</td></tr><tr><td>F1</td><td>0.369</td><td>0.370</td><td>0.370</td><td>0.374</td><td>0.372</td><td>0.358</td><td>0.329</td></tr></table>

We identify peaks greedily by visiting the adapters in $\hat { H }$ in decreasing order of posterior mean (Fig. S1). The first adapter initializes a peak, and each later adapter initializes a new peak if it lies more than $1 . 5 \ell _ { s }$ from all existing peak founders. Otherwise, it joins an existing peak whose founder lies within $\ell _ { s } .$ . We then fill the active set by taking the best unselected member from each peak in turn, following the order in which the peaks were created, until all m slots are filled. If the peaks run out of members, we fill the remaining slots from H<sup>ˆ</sup> in decreasing order of posterior mean, although H<sup>ˆ</sup> typically contains far more than m adapters. With only one peak, this procedure reduces to plain top-m selection.

The peak readout raises the F1 of the active set by about 0.04 over plain top-m. The separation radius of $1 . 5 \ell _ { s }$ gives the highest F1, smaller radii stay within 0.01 of it, and larger radii merge distinct peaks and fall toward plain top-m (Table S3).

## A.4. Refinement

Predicting the style of a merge. Because acquisition evaluates a proposed merge before its image is generated, its style feature is not yet available. We therefore predict it from the gallery as the coefficient-weighted sum of the active adapters’ style features,

$$
\hat { \mathbf { z } } ( \mathbf { \boldsymbol { \mathsf { a } } } ) = \sum _ { i \in \cal S } \alpha _ { i } \mathbf { z } _ { i } .\tag{S3}
$$

This prediction is used only for acquisition: once a merge is generated, it enters the posterior with its measured z, so prediction errors affect which merges are proposed but not what the model learns. Because zˆ is linear in α, Eq. (8) remains differentiable end to end. As in GimmBO, we optimize it with multi-start L-BFGS initialized from stick-breaking samples of the domain.

![](images/5cb22614dd8f67bdb1e61b513e4792c5f75247236c1ecc47dd44e71ae794b0ad.jpg)  
Figure S2: Nonzero per-adapter strengths: thresholding GimmBO’s map afterwards leaves sub-τ residuals (left), while enforcing τ inside the recursion does not (right).

Sampling the domain. GimmBO maps the unit cube onto the capped simplex using stick-breaking. Starting from $R _ { 0 } = B ,$ each coordinate takes a fraction of the remaining stick, $\alpha _ { i } =$ min $\left( x _ { i } R _ { i - 1 } , 1 \right)$ and $R _ { i } = R _ { i - 1 } - \alpha _ { i } .$ , with a random coordinate order to avoid favoring any axis. Every sample respects the cap, and most coordinates remain small, encouraging sparse merges. However, the total strength is strongly biased toward the cap: for uniform inputs, $\mathbb { E } [ \Sigma _ { i } \mathbf { a } _ { i } ] = B ( 1 - 2 ^ { - m } )$ , which is within 1% of B at $m = 8 ,$ whereas real community merges span a much broader range of lower totals (inset in Sec. 4.3.3).

We therefore replace the fixed $R _ { 0 } = B$ with a variable stick length $S \leq B .$ When drawing samples for acquisition optimization, we fit the distribution of S to the total strengths of over a thousand popular community merges, so the samples used to seed the L-BFGS restarts have realistic totals. During optimization, S becomes the (m+1)-th decision variable, allowing the search to optimize total strength directly rather than remain near the cap. The empirical distribution therefore affects only the restart samples, not the feasible domain, so totals rarely used by the community remain reachable during optimization.

Thresholding the stick-breaking output against τ after the fact either violates the cap or leaves sub-τ residuals, coefficients too small to matter but still nonzero (Fig. S2, left). We instead enforce τ directly within the recursion. With $R _ { 0 } = S$ and $\bar { u } _ { i } = \operatorname* { m i n } ( 1 , R _ { i - 1 } )$

$$
\alpha _ { i } = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } \bar { u } _ { i } < \tau , } \\ { \tau + x _ { i } \left( \bar { u } _ { i } - \tau \right) } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad R _ { i } = R _ { i - 1 } - \alpha _ { i } ,\tag{S4}
$$

so each break is either zero or at least $\tau ,$ and every sample lies exactly in $\Delta _ { B }$ (Fig. S2, right).

## A.5. Image generation

All images are generated with the SDXL 1.0 base model [PEL<sup>∗</sup>24] using the diffusers pipeline in half precision at a resolution of 1024 × 1280. Following GimmBO [LLJ26], we use the DPM++ 2M sampler with the Karras noise schedule, 30 denoising steps, and a classifier-free guidance scale of 7. During refinement, the m selected adapters are loaded once, and each merge fuses their weight updates into the base model using the coefficients α from Eq. (1) before image generation. Gallery images use seed 42 by default and are manually checked for quality, while all other images use the same five fixed seeds across methods, one per run.

![](images/5b3b746ddf3fa43445dab43137acf7835e3d3a5e16a84b05f1029ac5de8c5bff.jpg)  
Figure S3: Wall-clock time of one session, from the per-round times of Table S4. HimmBO requires less user interaction time because its screening rounds arefaster (Table S4).

Gallery images are generated from text alone, so that each adapter’s style is preserved without content constraints. During a session, we control content with SDEdit [MHS<sup>∗</sup>22]. Each session first generates an anchor image from its prompt with the base model alone, and every merge is then generated by noising this anchor to $t _ { 0 } = 0 . 8$ and denoising from there, which runs 24 of the 30 steps.

## A.6. Cost and engineering

Search computation remains small at every library size. Screening requires no GPU: each round consists of one posterior fit and one scan of the library, taking about 0.3 s. Gallery images that have not appeared in any comparison contribute nothing to the likelihood and can be integrated out exactly under the Gaussian process prior, so we fit the posterior only on compared images. During refinement, when only a few new images are added each round, this is about ten times cheaper than fitting on the entire gallery. The remaining per-round cost, scoring all adapters, scales linearly with N. The two quadratic quantities, the style kernel matrix and the median distance used to set $\ell _ { s } ,$ are computed once when the library is built and updated only when the library changes (Sec. A.2). Refinement always receives $m = 8$ adapters, which are loaded once at its start, so its session start-up cost is independent of N.

We report wall-clock time per round (Table S4) and per session (Fig. S3) on a single H100 GPU with the N=360 library, averaged over ten sessions done by two novice users. During refinement, each round generates q merges, while posterior updating and acquisition optimization take about 1 s compared with about 35 s for image generation. Thus, in both stages, user-facing latency is dominated by image generation or the user’s own judgment rather than search computation. On average, generation takes 4.4 s per image in HimmBO and 5.7 s in GimmBO because HimmBO refines only over the $m { = } 8$ selected adapters, which remain in memory, whereas GimmBO searches the full library and often needs to load adapters from storage. A HimmBO session takes about 15 minutes, compared with 37 minutes for GimmBO. Most of this difference comes from screening, whose 12 rounds require no image generation; the remainder comes from shorter user judgment per round and faster model fitting, which takes about ten seconds per round for GimmBO under its SAAS prior (Sec. A.1). Participants who used both methods reported that HimmBO performs better than GimmBO.

Table S4: Wall-clock seconds per round on a single H100 GPU with the N=360 library.
<table><tr><td></td><td>user</td><td>gen.</td><td>fit</td><td>acq.</td><td>total</td></tr><tr><td>HimmBO screening</td><td>24.4</td><td>0</td><td>0.28</td><td>0.02</td><td>24.7</td></tr><tr><td>HimmBO refinement</td><td>41.9</td><td>35.2</td><td>0.21</td><td>0.97</td><td>78.3</td></tr><tr><td>GimmBO</td><td>46.8</td><td>46.3</td><td>10.8</td><td>2.6</td><td>106.5</td></tr></table>

Table S5: Adapters left after each curation step.
<table><tr><td>Step</td><td>Adapters</td></tr><tr><td>Crawl: SDXL 1.0 adapters tagged style</td><td>9,909</td></tr><tr><td>Filter: downloads and tags</td><td>1,832</td></tr><tr><td>Language-model check</td><td>1,359</td></tr><tr><td>Style de-duplication</td><td>1,200</td></tr><tr><td>Generated successfully</td><td>1,172</td></tr><tr><td>Manual pass</td><td>1,080</td></tr></table>

## Appendix B: Experimental setup

## B.1. The library

Curation. On August 18, 2026, we queried the public Civitai API for LoRA adapters tagged style with SDXL 1.0 as the base model, obtaining 9,909 adapters. Table S5 lists how many adapters remain after each curation step of Sec. 5.1. Adapters for derived base models such as Pony and Illustrious rarely work on SDXL and are excluded. We keep adapters with at least 1,000 downloads and drop those whose tags or names indicate content other than style, such as characters, clothing, poses, detail enhancers, and sliders, as well as those lacking a downloadable SDXL version or a preview image. We also retain all adapters from GimmBO’s library that are still online. Because tags cannot distinguish a style from an adapter that adds content, a language model (Claude Haiku) reads each remaining adapter’s name, tags, trigger words, and description, keeping only those that change how an image looks rather than what it shows. Since independently trained adapters can produce nearly identical looks (Fig. 4), we repeatedly remove the less downloaded adapter of the most similar pair, measured by CSD features [SGG<sup>∗</sup>24] of their preview images, until 1,200 adapters remain. We then generate each adapter alone at unit strength from a fixed seed and a short prompt beginning with its trigger word. Of these, 1,172 produce an image, while the rest are no longer available or use unsupported formats. Finally, we inspect all generated images and remove adapters that do not act as styles, mostly those that replace the subject, such as building blocks or game icons, or add a generic look, such as cinematic.

Nested libraries. The full library, SDXL-Style-1K, contains 1,080 adapters with a median of 2,100 downloads, including GimmBO’s 40 adapters. We select the N=360 library to cover the style space broadly, then remove or add adapters to obtain the N=40, 120, and 720 libraries, so that all five are nested, $4 0 \subset 1 2 0 \subset 3 6 0 \subset 7 2 0 \subset$ 1080. We will release the manifest of all five libraries, including each adapter’s Civitai identifier and version.

## B.2. Targets and simulated user

The 24 ground-truth (GT) targets cover the four subject types used by the gallery prompts (a person, an animal, a still life, and a landscape) with six targets each. Their prompts will be released with the code. Half of the GT targets are generated from hidden merges of two library adapters and half from three, the most common merge sizes in community use and small enough for each component’s style to remain distinguishable. Their total strengths are drawn from the empirical distribution of community merges, using the same usage statistics as in Sec. A.4. To avoid low-quality targets, we manually inspect the sampled merges and exclude cases where a component’s style is barely visible or the generated image is low quality.

The simulated user follows GimmBO’s protocol (Sec. 5.1), with its choices converted into pairwise comparisons as follows. During screening, each of the c marked images is preferred over each of the q − c unmarked images, yielding 16 comparisons per round. During refinement, each of the top five images is preferred over every image ranked below it or left unranked.

## B.3. Baselines

GimmBO runs with its published implementation and configuration [LLJ26].

For Stylus [LWT<sup>∗</sup>24], we rebuild the retrieval index on our library rather than using Stylus’s own adapter index. A visionlanguage model (Gemini 3.7 Flash) reads each adapter’s model card and three of its images, one from our gallery, and writes a description of the adapter. We embed these descriptions with bgelarge-en-v1.5 [XLZ<sup>∗</sup>24] and retrieve the 150 adapters closest to the query by cosine similarity, where the query is a description of the target’s components written by the same model. Stylus’s composer and masking steps are kept as published, with the same model as the composer.

In our simulated-user experiments, across rounds, we rerun Stylus with a new seed, each round taking the top 8 adapters not taken in earlier rounds, then keep the running best solution. Stylus+GimmBO instead runs GimmBO on the top 40 of the 150 retrieved adapters after running one Stylus retrieval at the beginning.

Random coordinate descent follows GimmBO’s slider baseline. It visits the library’s adapters in random order and sweeps each one’s coefficient over eight evenly spaced values in [0, 1] while keeping the others fixed, and the simulated user keeps the most similar image as the new incumbent. The range includes 0 so that the baseline can switch adapters off, and we do not impose the cap B. Images already generated are reused rather than regenerated.

## Appendix C: Choice of perceptual encoder

The style map can use any perceptual encoder ϕ (Sec. 4.1). We compare DreamSim against three alternatives: the recent text-prompted perceptual metric TPIPS [WNH<sup>∗</sup>26], prompted with style; CLIP ViT-L/14 [RKH<sup>∗</sup>21]; and the style descriptor CSD [SGG<sup>∗</sup>24]. For each run, we replace the encoder in both the style map and the simulated user while keeping all other settings of the main N=360 experiments unchanged. Each row of Table S6 therefore evaluates a complete pipeline built on one encoder, rather than isolating the style kernel alone. We then score the final merge from each run against the GT using all four encoders.

Table S6: Similarity ofthefinal merge to the GT on the N=360 library. Rows give the encoder used throughout the search, including the simulated user, and columns the encoder used for scoring; the best entry ofeach column is in bold.
<table><tr><td></td><td colspan="4">Scored with</td></tr><tr><td>Searched with</td><td>DreamSim</td><td>TPIPS</td><td>CLIP</td><td>CSD</td></tr><tr><td>DreamSim</td><td>0.925</td><td>0.921</td><td>0.919</td><td>0.888</td></tr><tr><td>TPIPS</td><td>0.892</td><td>0.923</td><td>0.904</td><td>0.860</td></tr><tr><td>CLIP</td><td>0.875</td><td>0.901</td><td>0.911</td><td>0.853</td></tr><tr><td>CSD</td><td>0.873</td><td>0.898</td><td>0.908</td><td>0.867</td></tr></table>

The DreamSim pipeline achieves the highest score under three of the four evaluation encoders, while the TPIPS pipeline is slightly higher under TPIPS (0.923 versus 0.921). Notably, when scored with CLIP or CSD, the DreamSim pipeline also outperforms the pipelines that use CLIP or CSD during search. Thus, its advantage is not confined to evaluation with the same encoder used during search. In the DreamSim column, all three alternative pipelines also outperform GimmBO (0.808, Sec. 5.1), even though none uses DreamSim during search, whereas GimmBO’s simulated user does.

## Appendix D: How much feedback each kernel needs

The inset of Sec. 4.2 measures the number of feedback rounds required before the active set recovers at least half of the target’s components, using the nested libraries of Sec. 5.1. The two curves follow the same protocol and differ only in the kernel: the full kernel of Eq. (4) or its coefficient channel alone. We use pure posterior variance (ALM [Mac92]) for acquisition so that neither kernel is paired with an acquisition rule tailored to it. Each round shows q=8 gallery images, of which the simulated user marks the c=4 closest to the target (Sec. B.2), yielding 16 comparisons. Thus, the ratio between the curves is identical whether feedback cost is measured in rounds or comparisons. Adapters may be queried again in later rounds, and we impose no round limit because the coefficient-only kernel requires far more feedback than a normal session allows. We read out the active set as in Sec. A.3 with m=8 and evaluate four targets over five seeds.

## Appendix E: A gallery that grows with use

We test whether adding delivered merges to the gallery (Sec. 5.2, Fig. 11) improves later sessions. To mimic popular adapters that many users combine, we take 12 widely downloaded adapters with distinct styles from the N=360 library and sample 60 merges of two or three of them, following the target protocol of Sec. B.2. We run the default session (Sec. 5.1) with each merge as the target.

We then select six GT targets, three with two components and three with three, whose original sessions achieve a rough match but not a near-exact one. We add the remaining 54 merges to the gallery, together with their coefficients, images, and style features.

Each added merge shares at least one component with some target, but none uses exactly the same component set as any target, so the gallery contains related combinations rather than the answers themselves. An added merge enters the posterior like any other gallery sample: if selected by the procedure of Sec. A.3, it occupies one slot while introducing all of its components into the active set. We rerun the six targets with this expanded gallery and compare against their sessions on the original gallery averaged over the same five seeds.

The added merges improve active-set recovery for half of the targets and increase the mean final similarity from 0.923 to 0.944. In practice, popular community merges and merges delivered in earlier sessions could be added in the same way without additional image generation, allowing the gallery to become denser in regions that users frequently prefer. These results are preliminary, and confirming the effect at scale will require a larger study using real usage logs.

## Appendix F: Additional qualitative results

Fig. S4 shows eight adapters of our library through their gallery images.

Fig. S5 shows five open-ended sessions with no target to match. After screening the library as usual, participants write their own prompts and rank merges by preference alone.

landscape

landscape

animal

person

person

animal

still life  
still life  
![](images/cb74e6ec5dbe4b09eac5a51ae8f2ce91e5cf041e809cedc641a169e66bd6387d.jpg)  
Figure S4: Eight adapters of the library, each generated alone at unit strength on the four gallery prompts.

5  
7  
1  
2  
3  
4  
6  
![](images/a06103cdf3b7a2676b6d467d41192447f15e27a3b753645b26118ab6cdabf03e.jpg)  
Figure S5: The refinement stage of five open-ended sessions, one per row.