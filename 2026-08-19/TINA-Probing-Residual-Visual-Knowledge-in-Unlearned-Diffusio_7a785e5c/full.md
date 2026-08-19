# TINA+: Probing Residual Visual Knowledge in Unlearned Diffusion Models via Diffusion-Consistent Text-Free Inversion

Qianlong Xiang, Miao Zhang<sup>†</sup>, Member, IEEE, Kun Wang, Haoyu Zhang, Junhui Hou<sup>†</sup>, Senior Member, IEEE, Liqiang Nie, Senior Member, IEEE

Abstract—Although text-to-image diffusion models exhibit remarkable generative power, concept erasure techniques are essential for their safe deployment to prevent the creation of harmful content. To evaluate these erasure methods, a series of adversarial probes have been designed to test whether erased concepts can still be recovered, in turn driving the development of stronger erasure defenses. However, existing erasure and probe methods remain largely confined to a text-centric paradigm, focusing on whether the text-to-image mapping is severed while overlooking whether the corresponding visual knowledge still remains. To investigate this overlooked question, we adopt a visual perspective and leverage diffusion inversion to probe whether a generative trajectory can still be found to reconstruct visual instances of the erased concept. A natural starting point is standard diffusion inversion, where the text prompt facilitates faithful reconstruction. However, this textual condition is precisely what text-centric defenses suppress, and its involvement would also prevent a purely visual assessment of residual knowledge. Operating instead under a null-text condition removes this dependence and enables a purely visual probe. This benefit comes at a cost: the absence of textual conditioning amplifies the minor approximation errors inherent in standard diffusion inversion, hindering faithful trajectory recovery. To address these challenges, we introduce TINA+, a diffusion-consistent Text-free INversion Attack equipped with an optimization-based inversion procedure to improve null-text inversion accuracy. Beyond inversion accuracy, we find that unconstrained diffusion inversion may discover spurious trajectories, even allowing a randomly initialized diffusion model to reconstruct the target concept. Such trajectories are inconsistent with the diffusion process and may falsely indicate the presence of residual visual knowledge. TINA+ therefore introduces Diffusion-Consistent Trajectory Regularization to suppress this failure mode. By penalizing trajectories that fall far below the expected marginal energy evolution of diffusion, TINA+ avoids spurious inversion paths while preserving its ability to recover erased concepts through diffusion-consistent trajectories. Extensive experiments across twelve erasure methods, four concept-erasure tasks, and different model architectures demonstrate that TINA+ reliably probes residual visual knowledge through diffusion-consistent visual trajectories. These results provide stronger evidence that current methods often obscure concepts by severing text-image links, rathe than eliminating the underlying visual knowledge. The project page is https://qianlong0502.github.io/TINA-Plus-Homepage/.

Index Terms—Text-to-image diffusion models, concept erasure, machine unlearning, adversarial attacks.

## 1 INTRODUCTION

With the rapid development of deep learning [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], generative artificial intelligence, particularly Text-to-Image (T2I) diffusion models such as Stable Diffusion is fundamentally reshaping the landscape of digital content creation [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22]. Their remarkable capacity for creative synthesis has catalyzed countless applications, from art and design to entertainment and media. However, since these models are typically trained on massive, unfiltered datasets scraped from the Internet [23], their proliferation brings a turbulent undercurrent of ethical and safety challenges. The same models that generate breathtaking art can be misused to create deepfakes that violate personal privacy [24], imitate artistic styles that infringe on copyright [25], [26], [27], and produce harmful Not-Safe-For-Work (NSFW) imagery [28], [29], [30], posing risks to intellectual property and societal well-being.

To address these challenges, the field has converged on the paradigm of Concept Erasure, a specialized form of Machine Unlearning [31]. This paradigm aims to mitigate risks by directly modifying model parameters. For instance, approaches such as Erasing Stable Diffusion (ESD) and Ablating Concepts (AC) retrain specific model layers to remap a forbidden concept representation to that of a neutral or null concept, thereby severing the explicit link between a text prompt and its corresponding visual output [32], [33]. Other methods like Forget-Me-Not (FMN) minimize the attention maps corresponding to the target concept [34], while Unified Concept Editing (UCE) applies efficient, closed-form edits to the cross-attention weights [35]. Despite their distinct mechanisms, these approaches share a common, text-centric operational principle, equating erasure with merely severing its text-image mapping.

![](images/1aef6f3e524920842fd52bd60fb7437c599e42b547d9c2225eb9338d6c07f9ba.jpg)  
Fig. 1. Conceptual overview of text-centric erasure vulnerabilities and our TINA+ attack. Concept Erasure usually severs the link between a specific text condition and the undesired concept. Previous Attacks remain text-centric, finding adversarial text condition to reactivate the concept. Our TINA+ bypasses the text pathway entirely. Using an empty text condition, it finds a noise that regenerates the concept through a diffusion-consistent trajectory, providing stronger evidence that residual visual knowledge persists in existing erased models.

The text-centric paradigm, as conceptualized in Figure 1, has consequently guided adversarial attacks almost exclusively to the textual domain. Initial red-teaming efforts involve prompt-level manipulations, attempting to find alternative phrasings or complex prompts that reactivate the target concept [36], [28], [37]. Moreover, the more sophisticated embedding-space attacks, which move beyond discrete tokens, remain fundamentally dependent on the textual conditioning pathway [38]. Generally, these red-teaming methods aim to discover new token embeddings [38] or optimize an adversarial text prompt [28] that serves as a proxy for the erased concept. In response to this threat, an interplay between text-based attacks and defenses has driven the subsequent development of more resilient safeguards, such as AdvUnlearn [39] and STEREO [40]. This adversarial co-evolution has produced defenses that demonstrate increasing efficacy, thereby fostering an appearance of robust model security.

However, in this work, we contend that this competitive co-evolution of text-centric methods rests on a fundamental and fatal assumption. Specifically, existing paradigm mistakenly equates the erasure of a text-to-image link with the far more complex task of eliminating the underlying visual knowledge from the parameter space. Consequently, existing methods critically ignore the fact that even if textual inputs can no longer induce the target concept, the corresponding visual knowledge may still persist, untapped, within the model. To challenge this dominant paradigm and substantiate our claim, we introduce a core hypothesis:

If concept erasure primarily severs text-image mappings, the underlying visual knowledge may still persist within the erased model, enabling the erased concept to be regenerated through corresponding text-free generative trajectories.

To validate this hypothesis from a purely visual perspective, thereby challenging the efficacy of these text-centric defenses, we introduce TINA+, a diffusion-consistent Textfree INversion Attack. Unlike prior embedding-space attacks that seek a new textual proxy for the erased concept [38], TINA+ is designed to completely bypass the textual conditioning pathway. To achieve this, TINA+ leverages Denoising Diffusion Implicit Models (DDIM) [41] inversion as a visual probe (illustrated in Figure 1), allowing us to investigate the visual knowledge embedded within the model directly. However, identifying such a generative trajectory is challenging. Standard diffusion inversion typically relies on textual guidance to facilitate faithful reconstruction, but this guidance is precisely what text-centric defenses suppress and would also prevent a purely visual assessment of residual knowledge. Operating under the null-text condition necessary for a purely visual probe removes this dependence, but amplifies the minor approximation errors inherent in standard inversion. To address this imprecision, TINA+ incorporates an optimization procedure that corrects for these errors, allowing it to more accurately map a target image back to a latent noise vector. When used to initialize the subsequent generation process, the resulting latent vector can lead an erased model to regenerate the forbidden content through a text-free generative trajectory, entirely bypassing the text-conditioning mechanism.

However, accurate text-free inversion alone is not sufficient to provide a reliable probe of residual visual knowledge. An optimization-based inversion procedure may overfit the target image without necessarily reflecting a plausible diffusion generation process. In extreme cases, even a randomly initialized diffusion model, which contains no meaningful learned visual knowledge of the target concept, can be inverted to reconstruct the target concept through a spurious, diffusion-inconsistent trajectory. This observation reveals an important limitation: visual resemblance alone does not necessarily certify that the erased model truly preserves the corresponding concept knowledge. A reliable text-free probe should therefore not only regenerate the target concept, but also ensure that the discovered trajectory remains compatible with the expected marginal energy behavior of diffusion. To address this issue, TINA+ introduces Diffusion-Consistent Trajectory Regularization. Specifically, TINA+ penalizes trajectories whose latent energy falls far below the expected marginal energy evolution of diffusion, thereby suppressing severe energy collapse along spurious inversion paths. We further introduce a forward marginal initialization strategy to bypass the fragile low-noise region and stabilize trajectory discovery. With these designs, TINA+ serves not merely as a reconstruction-oriented attack, but as a diffusion-consistent visual probe that better distinguishes genuine residual visual knowledge from artifacts caused by unconstrained inversion.

Extensive experiments across four concept-erasure tasks evaluate TINA+ against twelve state-of-the-art erasure methods and six representative attacks. The results show that TINA+ remains effective even when existing textcentric attacks are substantially suppressed, supporting the persistence of text-independent visual trajectories in erased models. Negative-control experiments and trajectory diagnostics further demonstrate that the proposed diffusionconsistency constraints suppress spurious reconstructions, making TINA+ a more reliable probe of residual visual knowledge. Additional latent embedding and architecturegeneralization analyses show that the discovered trajectories activate concept-specific internal representations and that the exposed vulnerability is not limited to a particular diffusion architecture. Together, these findings provide compelling evidence that current erasure methods often obscure concepts by severing text-image links rather than fully removing the underlying visual knowledge, highlighting the need for more robust unlearning paradigms that operate directly on internal visual representations within T2I models.

To sum up, our contributions are as follows.

To our knowledge, we are the first to identify a fundamental limitation of the text-centric concept erasure paradigm: severing text-image mappings does not necessarily eliminate the underlying visual knowledge from diffusion models.

We propose TINA+, a text-free attack that performs optimization-based null-text inversion under diffusion-consistent trajectory regularization to reliably probe residual visual knowledge in erased models and assess the effectiveness of concept erasure.

Extensive experiments across twelve erasure methods, four concept-erasure tasks, and different model architectures, together with random-model controls and latent embedding analysis, demonstrate the effectiveness, reliability, and generalizability of TINA+.

The preliminary conference version of this work introduced TINA, an optimization-based text-free inversion attack, at CVPR 2026 [42]. This journal version develops it into TINA+ and offers three major improvements:

We provide a deeper analysis of text-free inversion and identify a previously overlooked failure mode: optimizing inversion accuracy alone can lead to spurious, diffusion-inconsistent trajectories, which may even produce visually recognizable target-like images on a randomly initialized diffusion model. This motivates a more reliable notion of trajectory validity beyond visual reconstruction.

We extend the original TINA framework to TINA+ by introducing diffusion-consistent trajectory regularization. Specifically, we derive a marginal energy constraint from the forward diffusion marginal to penalize severe latent energy collapse, and further introduce forward marginal initialization to stabilize trajectory discovery in the fragile low-noise region.

We substantially expand the experimental evaluation to validate the proposed diffusion-consistent probing framework. In addition to the original attack success evaluation, we include trajectory-level diagnostics, validity-aware analysis, ablation studies of the new components, and additional robustness studies to demonstrate that TINA+ can better distinguish genuine residual visual knowledge from artifacts caused by unconstrained inversion.

## 2 RELATED WORK

## 2.1 Text-to-Image Diffusion Models

The advent of Text-to-Image (T2I) diffusion models, such as Stable Diffusion [12], DALL-E 2 [43], and Imagen [15], has marked a paradigm shift in digital content generation. These models operate through a two-stage process: a forward diffusion process that incrementally adds noise to an image, and a learned reverse process that denoises a random vector back into a coherent image [44]. To achieve computational efficiency, models like Stable Diffusion perform this process in a lower-dimensional latent space [12]. Critically, their generative process is guided by textual prompts, typically encoded and injected into the denoising network via a cross-attention mechanism [1]. This tight coupling between text and image synthesis is the source of their remarkable controllability, but also the root of significant safety and ethical concerns, as the models can be prompted to produce NSFW imagery [28], [30], imitate copyrighted styles [25], [27], or violate personal privacy [24].

## 2.2 Concept Erasure in Diffusion Models

To address the misuse of T2I models, concept erasure has emerged as a critical field. While methods like SLD [30] focus on inference-time interventions, the dominant paradigm involves fine-tuning-based methods that modify model parameters. This paradigm originated with foundational works like ESD [32] and AC [33], which retrained models to break associations with specific text prompts. Subsequently, this was followed by more efficient and targeted techniques, such as FMN [34], UCE [35], MACE [45], and the adapter-based SPM [46], often focusing on cross-attention layers. Concurrently, general unlearning frameworks like SalUn [47], Scissorhands (SH) [48], and EraseDiff [49] were proposed for the concept erasure task. As text-based attacks began to bypass these initial erasure methods, advanced adversarially robust methods were developed, including RECE [50], AdvUnlearn [39], and STEREO [40]. These methods usually integrate attack considerations into their design to defend against adversarial prompts and embeddings. Ultimately, this co-evolution of text-based attacks and defenses has solidified a dominant, text-centric paradigm: most of them operate by identifying or severing the text-toimage mapping, rather than addressing the underlying visual knowledge.

## 2.3 Adversarial Attacks on Concept Erasure

The prevailing text-centric assumption of erasure methods has consequently shaped the landscape of adversarial attacks, which primarily seek to circumvent these textual defenses. These attacks can be classified by their operational domain. A broad class of attacks operates through textual prompts by searching for or optimizing discrete token sequences that trigger the erased concept, including PEZ [51], P4D [36], RAB [37], MMA [52], and UDA [28]. A more sophisticated class of attacks operates in the continuous embedding space. These methods aim to find a novel token embedding that acts as a proxy for the forbidden concept, effectively creating a new activation pathway to the visual knowledge. For instance, CCE [38] leverages textual inversion to discover surrogate embeddings for erased concepts.

Critically, despite their varying levels of sophistication, from simple phrasings to optimized vectors, these existing attacks share a fundamental limitation: most of them presuppose that the generative capabilities of the model must be accessed via its text-conditioning pathway. They focus on finding a new textual or embedding-space vector to bypass modified controls. In contrast, TINA+ targets a fundamentally different attack surface by probing whether visual knowledge persists independently of textual control. It bypasses the text-conditioning mechanism and discovers diffusion-consistent, text-free generative trajectories, enabling a reliable assessment of whether the erased model still retains the corresponding visual knowledge.

## 3 PRELIMINARIES

## 3.1 Latent Diffusion Models.

Latent Diffusion Models (LDMs) [12] generate images by reversing a T-step diffusion process in a compressed latent space. An image x is first encoded into its latent $z _ { 0 } = \mathcal { E } ( x )$ by a VAE encoder. The forward process creates a sequence of noised latents $\{ z _ { 1 } , \dots , z _ { T } \}$ . A denoising U-Net $\epsilon _ { \theta }$ learns to reverse this process. At each timestep $t ,$ it predicts the noise ϵ in $z _ { t } ,$ guided by a textual condition c. This condition is produced by a text encoder $\mathcal { T } _ { \mathrm { t e x t } }$ from a prompt y and integrated via cross-attention. The training objective minimizes the noise prediction error:

$$
\mathcal { L } = \mathbb { E } _ { z _ { 0 } , y , \epsilon , t } \left[ \| \epsilon - \epsilon _ { \theta } ( z _ { t } , t , c ) \| _ { 2 } ^ { 2 } \right] .\tag{1}
$$

This objective trains $\epsilon _ { \theta }$ to denoise $z _ { t }$ back towards $z _ { \mathrm { 0 } }$ in a manner faithful to the condition c. The final image is retrieved using the VAE decoder D. Following the DDPM [44] convention, we denote the per-step noise variance by $\beta _ { t , \cdot }$ , the corresponding signal coefficient by $\alpha _ { t } = 1 - \beta _ { t }$ , and the cumulative signal coefficient by

$$
\bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } .\tag{2}
$$

Throughout this paper, we use $\bar { \alpha } _ { t }$ for the cumulative coefficient that appears in the forward diffusion marginal.

## 3.2 DDIM Sampling.

Denoising Diffusion Implicit Models (DDIM) [41] enable deterministic generative sampling from LDMs by using a non-Markovian reverse process with zero sampling variance. Consequently, for a fixed model θ and condition $c ,$ any initial noise $z _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } )$ is deterministically mapped to a generated latent $z _ { 0 }$

This deterministic path is defined by an iterative update rule. Given the latent $z _ { t } ,$ the model first predicts the clean latent $\hat { z } _ { 0 } ( z _ { t } )$

$$
\hat { z } _ { 0 } ( z _ { t } ) = \frac { z _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( z _ { t } , t , c ) } { \sqrt { \bar { \alpha } _ { t } } } .\tag{3}
$$

The preceding latent $z _ { t - 1 }$ is then computed as:

$$
z _ { t - 1 } = \sqrt { \bar { \alpha } _ { t - 1 } } \hat { z } _ { 0 } ( z _ { t } ) + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \cdot \epsilon _ { \theta } ( z _ { t } , t , c ) ,\tag{4}
$$

where $\{ \bar { \alpha } _ { t } \} _ { t = 0 } ^ { T }$ are cumulative signal coefficients determined by the fixed noise schedule, with $\bar { \alpha } _ { 0 } = 1$

## 3.3 DDIM Inversion.

Building upon the deterministic nature of DDIM sampling, DDIM inversion provides a mechanism to reverse the generative process. Its objective is to map a given image, represented by its initial latent $z _ { \mathrm { 0 } } ,$ , back to a seed latent $z _ { T }$ from which the image can be reproduced through sampling. This capability is crucial for applications like real image editing and analysis.

Ideally, the inversion would be a perfect algebraic reversal of the sampling step in Eq. (4), which can be expressed as:

$$
z _ { t } = C _ { 1 } ( t ) z _ { t - 1 } + C _ { 2 } ( t ) \cdot \epsilon _ { \theta } ( z _ { t } , t , c ) ,\tag{5}
$$

where

$$
C _ { 1 } ( t ) = \frac { \sqrt { { \bar { \alpha } } _ { t } } } { \sqrt { { \bar { \alpha } } _ { t - 1 } } } ,\tag{6}
$$

$$
C _ { 2 } ( t ) = \sqrt { 1 - \bar { \alpha } _ { t } } - \sqrt { \frac { \bar { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } { \bar { \alpha } _ { t - 1 } } } .\tag{7}
$$

However, a practical challenge emerges: computing the latent $z _ { t }$ requires a noise prediction $\boldsymbol { \epsilon } _ { \theta } ( z _ { t } , t , c )$ , which itself depends on $z _ { t } ,$ as shown in Eq. (5). To break this circular dependency, standard DDIM inversion employs a key approximation: it estimates the required noise using the latent from the previous step, $z _ { t - 1 } ,$ , at timestep t−1. This results in the following iterative formula to approximate $z _ { t }$ from $z _ { t - 1 }         { : }$

$$
z _ { t } \approx f _ { \theta } ( z _ { t - 1 } , t , c ) ,\tag{8}
$$

where

$$
f _ { \theta } ( z _ { t - 1 } , t , c ) = C _ { 1 } ( t ) z _ { t - 1 } + C _ { 2 } ( t ) \cdot \epsilon _ { \theta } ( z _ { t - 1 } , t - 1 , c ) .\tag{9}
$$

Starting from the clean image latent $z _ { \mathrm { 0 } } ,$ , this equation is applied sequentially for $t = \check { 1 } , \dots , T$ to trace the trajectory back to an approximate seed latent zˆ<sub>T</sub>. This vector serves as an estimate of the desired seed in the high-noise latent space required for reconstructing the original image.

## 4 FRAMEWORK OVERVIEW

Figure 2 provides a high-level overview of our attack framework. Given a concept-erased diffusion model $\epsilon _ { \theta }$ and a target image x containing the erased concept, our goal is not to search for an adversarial textual prompt, but to examine the model from the visual latent space and determine whether it still admits a generative pathway for this concept. We first encode the target image into the latent space as $z _ { 0 } = \mathcal { E } ( x )$ The attack then identifies an inverted seed latent, denoted as $z _ { T } ^ { * } ,$ , by tracing the target latent toward the high-noise end of the diffusion process. This visual trajectory discovery stage provides a visual-space probe for examining whether the erased model still retains generative knowledge of the target concept. The seed latent $z _ { T } ^ { * }$ serves as the starting point for sampling: when the erased model starts from this seed, it is expected to traverse a corresponding reverse trajectory and synthesize content associated with the target concept.

Once the seed latent $z _ { T } ^ { * }$ is obtained, we feed it back into the same concept-erased model $\epsilon _ { \theta }$ and perform sampling to generate a new latent $z _ { \mathrm { 0 } } ^ { \prime } ,$ which is decoded into an image $\overline { { x } } ^ { \prime } = \mathcal { D } ( z _ { 0 } ^ { \prime } )$ . This regeneration stage closes the loop of our visual probe: it tests whether the seed latent discovered from the target concept can drive the erased model to synthesize the corresponding concept again. Finally, we evaluate the generated image using concept classifiers to determine whether the concept that should have been erased is regenerated. If the generated image is recognized as containing the erased concept, the attack is considered successful. Otherwise, no residual knowledge is detected under this visualspace probe.

![](images/db9fed07d1fa81fb10bb71b2fa9366aa9c418e8dbcc6e6329aea6c3127395621.jpg)  
(a) Target Concept  
(b) Generative Trajectory Discovery  
(c) Concept Regeneration & Evaluation

Fig. 2. Framework overview of our proposed attack. Given target images containing the erased concept, we identify generative trajectories and obtain the corresponding seed latents. These seed latents are then used to generate new images with the same concept-erased model. If the generated images still contain the concept that should have been erased, the model is considered to retain the corresponding visual knowledge.  
![](images/6483de819c16fcf7fb5202b8ab00ef979d1a4ba579a44529933d008e674ee53a.jpg)  
Fig. 3. Standard DDIM inversion is insufficient for reliable visual trajectory discovery. (a) Text-conditioned inversion is resisted by the concept erased model, since the corresponding text-to-image mapping has been suppressed. (b) Null-text inversion avoids the textual defense but suffers from accumulated inversion errors, failing to faithfully recover the target concept even on the unerased model. (c) Null-text inversion can even produce visually recognizable target-like images on a randomly initialized diffusion model through an unnatural, diffusion-inconsistent trajectory.

Generally, given a set of images representing a target concept, our framework discovers visual generative trajectories associated with these images and then tests whether the discovered seed latents can drive the erased model to generate the supposedly erased concept. However, successful regeneration alone is insufficient evidence of residual visual knowledge, because it may arise from a spurious trajectory that is inconsistent with the diffusion generation process. The following sections detail how such visual trajectories are identified, optimized, and regularized to reveal whether a concept-erased model still preserves a generative pathway to the erased concept.

## 5 DIFFUSION INVERSION FOR VISUAL TRAJEC-TORY DISCOVERY

The framework above reduces our attack to a visual trajectory discovery problem: given a target latent z associated with the erased concept, we aim to find a seed latent z<sup>∗</sup> that can drive the erased model to synthesize the target concept through sampling. Diffusion inversion provides a natural way to instantiate this trajectory discovery. In particular, DDIM inversion traces a given image latent back toward the high-noise seed latent space by reversing the sampling process, making it a representative tool for discovering the generative trajectory associated with a target image. If such an inversion succeeds on a concept-erased model, the resulting seed latent can be used to test whether the model still retains a visual pathway to the erased concept.

![](images/0e80ad52a10cddf39c2146ca1625868e10ccc0af2fa22428e453d50858e73ad3.jpg)  
(a) Optimization-Based Text-Free Inversion

![](images/097e40d8bc4536aefaa433eccd30f5df65db486bd54fbf0bbfe3b2e661bb99b1.jpg)

![](images/ce2b49587995d5905b88c0089896cd833584b09b88c8ac4902ca8131e69a3856.jpg)  
(b) Diffusion-Consistent Trajectory Regularization  
Fig. 4. Overview of TINA+. Given a target image, we encode it into a latent z and perform text-free inversion under the null-text condition. TINA+ first refines each intermediate latent through fixed-point consistency to correct accumulated inversion errors. Furthermore, TINA+ applies diffusionconsistent trajectory regularization, including forward marginal initialization and marginal energy loss, to avoid spurious diffusion-inconsistent trajectories. The resulting seed latent $z _ { T } ^ { * }$ is then used for generation with the same concept-erased model.

However, directly applying standard DDIM inversion to concept-erased models is fundamentally flawed and exposes several important limitations, as illustrated in Figure 3. First, if the inversion is attempted using the text condition c of the target concept (e.g., “Van Gogh style”), the concept-erased model $\epsilon _ { \theta }$ actively counteracts this prompt, leading to a failure in reconstructing the erased concept. This confirms that text-centric defenses are indeed working against text-guided processes. In other words, text-guided inversion still relies on the very text-to-image mapping that concept erasure aims to suppress, and therefore cannot serve as a reliable tool for probing text-independent visual knowledge. As shown in Figure 3(a), such a textconditioned inversion process is still trapped in the same text-centric interaction between erasure and attack, rather than providing a purely visual exploration of the erased model.

Second, we could attempt the inversion under a nulltext condition $\boldsymbol { \mathrm { \Sigma } } ( c \mathrm { \Sigma } = \mathrm { \Lambda } c _ { \mathrm { n u l l } } )$ to bypass this textual defense. This approach also fails, but in a more subtle way. As shown in Figure 3(b), the resulting image is visually closer to the original than the text-conditioned attempt, yet it still fails to faithfully regenerate the specific target concept. This partial failure stems from the standard inversion formula (Eq. (8)), which relies on a critical approximation: it uses the noise prediction from the previous step, $\epsilon _ { \theta } ( z _ { t - 1 } , t - 1 , c )$ to estimate the next latent $z _ { t } .$ This approximation is heavily dependent on a meaningful text condition c to guide the noise prediction. In the absence of such guidance $( c = c _ { \mathrm { n u l l } } )$ the predictions become unconstrained, and the small error introduced at each step rapidly compounds across the entire trajectory. Consequently, the final computed seed latent $\hat { z } _ { T }$ drifts sufficiently from the desired seed latent $z _ { T } ^ { * }$ to lose the high-fidelity details of the target concept, rendering it incapable of accurately generating the target concept again.

More importantly, we further observe a more severe limitation when applying null-text DDIM inversion to a completely randomly initialized diffusion model. As shown in Figure 3(c), even though the random model contains no meaningful generative knowledge of the target concept, standard inversion can still trace the target image toward a seed latent and then produce an image that preserves certain coarse visual structures of the input. However, the intermediate trajectory is highly unnatural from the perspective of diffusion dynamics: it is diffusion-inconsistent and deviates substantially from the typical progression of a diffusion generation process. This observation suggests that visual resemblance produced by unconstrained inversion does not necessarily indicate that the model truly retains the corresponding concept knowledge. In other words, if even a random diffusion model can yield an apparently conceptrelated trajectory, then standard DDIM inversion alone is insufficient for evaluating whether a concept-erased model actually preserves the erased visual knowledge.

These observations show that diffusion inversion is a natural starting point for visual trajectory discovery, but standard inversion is insufficient as a reliable trajectorydiscovery method for attacking concept-erased models. The key challenge is therefore not merely to invert a target image, but to identify a seed latent whose generation trajectory remains faithful to the target concept under the erased model while avoiding spurious, diffusion-inconsistent trajectories that can arise from unconstrained inversion. Motivated by this challenge, we introduce TINA+, an optimization-based text-free inversion method that refines the intermediate latents along the trajectory to correct the accumulated approximation errors of standard inversion. Furthermore, TINA+ regularizes the discovered trajectory with diffusion-consistency constraints, so that successful regeneration is supported by a trajectory that is more consistent with the diffusion generation process rather than by a spurious inversion path.

## 6 DIFFUSION-CONSISTENT TEXT-FREE INVER-SION

The previous section shows that standard DDIM inversion is insufficient for reliable visual trajectory discovery. In particular, text-conditioned inversion is resisted by the erased text-to-image mapping, while null-text inversion avoids the textual pathway but introduces two remaining challenges: inaccurate inversion caused by accumulated approximation errors, and unnatural inversion caused by spurious trajectories. As illustrated in Figure 4, TINA+ addresses these challenges through optimization-based text-free inversion and diffusion-consistent trajectory regularization. The former enforces fixed-point consistency to correct inversion errors, while the latter constrains latent energy evolution and stabilizes initialization in the low-noise region.

## 6.1 Optimization-Based Text-Free Inversion

To overcome the imprecision of approximate inversion, TINA+ reframes visual trajectory discovery as an optimization problem. We first revisit where the standard DDIM inversion approximation comes from. The approximate update in Eq. (8) is derived from the DDIM sampling equation in Eq. (4), but it replaces the unknown noise prediction at $z _ { t }$ with the noise predicted at the previous latent $z _ { t - 1 }$ . Before making this approximation, the DDIM sampling equation implies an exact implicit relation between two consecutive latents on a DDIM trajectory:

$$
z _ { t } = C _ { 1 } ( t ) z _ { t - 1 } + C _ { 2 } ( t ) \cdot \epsilon _ { \theta } ( z _ { t } , t , c ) .\tag{10}
$$

This relation characterizes the ideal inversion trajectory. Unlike the approximate DDIM inversion update, Eq. (10) is implicit because z<sub>t</sub> appears on both sides through $\boldsymbol { \epsilon } _ { \theta } \big ( \boldsymbol { z } _ { t } , t , \boldsymbol { c } \big )$ For compactness, we denote the right-hand side of Eq. (10) as

$$
f _ { \theta } ^ { * } ( z _ { t } , z _ { t - 1 } , t , c ) = C _ { 1 } ( t ) z _ { t - 1 } + C _ { 2 } ( t ) \cdot \epsilon _ { \theta } ( z _ { t } , t , c ) .\tag{11}
$$

With this notation, an exact latent $z _ { t }$ on the DDIM trajectory must satisfy the self-consistency relation $z _ { t } =$ $f _ { \theta } ^ { * } \left( z _ { t } , z _ { t - 1 } , t , c \right)$ . In other words, plugging $z _ { t }$ into the righthand side should return the same $z _ { t }$ . Thus, exact inversion at each timestep can be cast as a fixed-point problem, where $z _ { t }$ is a fixed point of the mapping $f _ { \theta } ^ { * }$ [53], [54].

Based on this fixed-point formulation, our key idea is to enforce the self-consistency constraint through optimization, rather than directly accepting the approximate closedform inversion update. For each optimized timestep $t ,$ given the previously computed $z _ { t - 1 , }$ , we seek a latent $z _ { t }$ whose denoising prediction satisfies the exact DDIM relation under the null-text condition. We formulate this as the following loss function:

$$
\mathcal { L } _ { \mathrm { f p } } ( z _ { t } ) = \left\| f _ { \theta } ^ { * } ( z _ { t } , z _ { t - 1 } , t , c _ { \mathrm { n u l l } } ) - z _ { t } \right\| _ { 2 } ^ { 2 } ,\tag{12}
$$

where $z _ { t }$ is initialized by $f _ { \theta } \big ( z _ { t - 1 } , t , c _ { \mathrm { n u l l } } \big )$ before the optimization.

Specifically, at each inversion step, we initialize $z _ { t }$ using the conventional DDIM inversion update in Eq. (8) under the null-text condition. Starting from this initialization, we iteratively refine $z _ { t }$ by gradient descent to minimize Eq. (12). In this sense, our optimization seeks a numerically selfconsistent point of the exact implicit DDIM relation, instead of directly trusting the approximate inversion formula [54]. This refinement makes each intermediate latent agree with the exact DDIM relation defined by the concept-erased model, without relying on the erased concept prompt.

By iteratively applying this optimization-based refinement during the inversion process, TINA+ identifies a seed latent z<sup>∗</sup> that can serve as a purely visual starting point for sampling. If the concept-erased model can regenerate the target concept from this latent under the null-text condition, it indicates that the model still admits a visual generative pathway to the erased concept.

## 6.2 Diffusion-Consistent Trajectory Regularization

The optimization above addresses inaccurate inversion, but it does not by itself guarantee that the discovered trajectory is compatible with the marginal behavior of diffusion. As discussed in Figure 3(c), visual resemblance alone is insufficient to certify that the trajectory reflects genuine concept knowledge retained by the model. We therefore introduce diffusion-consistent trajectory regularization. We first formulate a marginal energy constraint to detect and penalize severe energy collapse along diffusion-inconsistent trajectories. We then stabilize the ill-conditioned low-noise region with forward marginal initialization. The former defines a weak energy-based consistency criterion for the trajectory, while the latter avoids applying step-by-step inversion in the most fragile low-noise segment.

## 6.2.1 Marginal Energy Constraint

A reliable visual generative trajectory should not be an arbitrary path in latent space. Since diffusion models are trained through a forward noising process and generate by reversing its time evolution, a trajectory discovered by inversion should remain compatible with the forward diffusion marginal at each timestep, at least in a weak statistical sense. We therefore introduce a trace-level marginal energy constraint on each intermediate latent $z _ { t }$ . The goal is not to enforce the full Gaussian distribution of every intermediate point, but to preserve the basic marginal energy evolution implied by diffusion and to prevent severe latent energy collapse.

We start from the standard forward diffusion marginal conditioned on $z _ { \mathrm { 0 } } \mathrm { : }$

$$
\begin{array} { r } { q ( z _ { t } | z _ { 0 } ) = \mathcal { N } \left( \sqrt { \bar { \alpha } _ { t } } z _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) I \right) . } \end{array}\tag{13}
$$

Equivalently, a forward-marginal sample can be written as

$$
z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{14}
$$

For a discovered trajectory to be considered diffusionconsistent, its intermediate latent $z _ { t }$ should be compatible with this timestep-dependent marginal behavior. A stringent requirement would be to test whether every $z _ { t }$ fully follows the Gaussian distribution in Eq. (13). However, this is unnecessarily restrictive for inversion trajectories, which are not obtained by repeatedly sampling from the stochastic forward process. Instead, we use the forward marginal as a weaker moment-level reference and examine whether the overall scale of $z _ { t }$ follows the expected diffusion energy envelope. Specifically, we consider the squared latent norm

$$
S _ { t } = \| z _ { t } \| _ { 2 } ^ { 2 } .\tag{15}
$$

We call $S _ { t }$ the latent energy, since it is the sum of squared latent responses. It is also a trace-level second-order statistic,

because $\begin{array} { r } { \| z _ { t } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { D } z _ { t , i } ^ { 2 } } \end{array}$ summarizes the total secondorder magnitude of the latent vector. Let D denote the dimensionality of $z _ { t } .$ . Using Eq. (14), we expand

$$
\begin{array} { r l } & { S _ { t } = \left\| \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon \right\| _ { 2 } ^ { 2 } } \\ & { \quad = \bar { \alpha } _ { t } \| z _ { 0 } \| _ { 2 } ^ { 2 } + 2 \sqrt { \bar { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t } ) } z _ { 0 } ^ { \top } \epsilon + ( 1 - \bar { \alpha } _ { t } ) \| \epsilon \| _ { 2 } ^ { 2 } . } \end{array}\tag{16}
$$

Since $\mathbb { E } [ \epsilon _ { i } ] ~ = ~ 0 ,$ , the cross term has zero expectation: $\begin{array} { r } { \mathbb { E } [ z _ { 0 } ^ { \top } \epsilon ] \ = \ \dot { \sum } _ { i } z _ { 0 , i } \mathbb { E } [ \epsilon _ { i } ] \ = \ 0 , } \end{array}$ . Moreover, $\mathbb { E } [ \left. \epsilon \right. _ { 2 } ^ { 2 } ] \ = \ D$ for $\epsilon \sim \mathcal { N } ( 0 , I )$ . Therefore, the conditional expectation of the latent energy is

$$
m _ { t } = \mathbb { E } [ S _ { t } | z _ { 0 } ] = \bar { \alpha } _ { t } \| z _ { 0 } \| _ { 2 } ^ { 2 } + ( 1 - \bar { \alpha } _ { t } ) D .\tag{17}
$$

To measure how far an observed $S _ { t }$ deviates from this energy envelope, we further derive its conditional variance. From Eq. (16), the variance comes from the linear Gaussian term and the quadratic Gaussian term. Since $z _ { 0 } ^ { \top } \epsilon \ \sim \ { \mathcal { N } } ( 0 , \| z _ { 0 } \| _ { 2 } ^ { 2 } )$ , the variance of the linear term is $4 \bar { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t } ) \lVert z _ { 0 } \rVert _ { 2 } ^ { 2 } .$ . Since $\| \epsilon \| _ { 2 } ^ { 2 } \sim \chi _ { D } ^ { 2 }$ and $\mathrm { V a r } ( \chi _ { D } ^ { 2 } ) = 2 D / \chi$ , the variance of the quadratic term is $2 ( 1 - { \bar { \alpha } } _ { t } ) ^ { 2 } D$ . The covariance between the linear and centered quadratic terms is zero due to the symmetry of the standard Gaussian. Thus,

$$
v _ { t } = \mathrm { V a r } ( S _ { t } | z _ { 0 } ) = 2 ( 1 - \bar { \alpha } _ { t } ) ^ { 2 } D + 4 \bar { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t } ) \| z _ { 0 } \| _ { 2 } ^ { 2 } .\tag{18}
$$

We then define the marginal energy score as the standardized deviation of the observed latent energy:

$$
r _ { t } ^ { \mathrm { e n g } } = \frac { \| z _ { t } \| _ { 2 } ^ { 2 } - m _ { t } } { \sqrt { v _ { t } } } .\tag{19}
$$

Under the stochastic forward marginal, this score measures the deviation of $S _ { t }$ from its expected energy in standarddeviation units. Large negative values indicate that the latent energy falls far below the expected diffusion energy envelope at timestep t.

For visualization and diagnosis, we further define the observed energy transition. Let $S _ { 0 } = \| z _ { 0 } \| _ { 2 } ^ { 2 }$ . The observed transition of latent energy from the clean latent energy $S _ { 0 }$ toward the Gaussian prior energy $D$ is

$$
\gamma _ { t } = \frac { \| z _ { t } \| _ { 2 } ^ { 2 } - S _ { 0 } } { D - S _ { 0 } } .\tag{20}
$$

In practice, the Gaussian prior energy D is larger than the clean latent energy $S _ { 0 } ,$ so $\gamma _ { t }$ measures the normalized energy transition from the clean latent regime toward the prior regime. Substituting the conditional expectation $m _ { t }$ in Eq. (17) into this definition yields the marginal energy transition:

$$
\gamma _ { t } ^ { \mathrm { m a r g } } = 1 - \bar { \alpha } _ { t } .\tag{21}
$$

The marginal energy transition is induced by the forward diffusion marginal and depends only on the diffusion noise schedule. It describes the expected transition of latent energy from the clean latent regime toward the Gaussian prior regime. Comparing the observed energy transition $\gamma _ { t }$ with the marginal energy transition $\gamma _ { t } ^ { \mathrm { m a r g } }$ provides an intuitive diagnosis of whether a discovered trajectory follows a reasonable diffusion-like energy evolution.

Figure 5 illustrates this behavior on Stable Diffusion v1-4 and on a randomly initialized UNet. On Stable Diffusion v1- $^ { 4 , }$ the marginal energy score remains in a moderate range, and the observed energy transition follows the marginal energy transition reasonably well, indicating that the discovered inversion trajectory preserves a plausible diffusionlike energy evolution. In contrast, on the Random UNet, the marginal energy score rapidly drops to extremely negative values, while the observed energy transition deviates drastically from the marginal energy transition and even moves in the opposite direction. This means that the trajectory fails to undergo the expected energy transition prescribed by diffusion, revealing severe latent energy collapse and indicating a diffusion-inconsistent trajectory. These observations support the use of marginal energy to assess whether a discovered trajectory remains statistically compatible with the marginal energy behavior of diffusion. Rather than requiring exact distributional matching, it is sufficient to rule out trajectories whose latent energy evolution is grossly incompatible with the forward diffusion marginal.

![](images/ac50643725fff4928e09b411475673f18408e50b7fc94f8e09c428f2a2470b8f.jpg)  
Fig. 5. Trajectory energy diagnostics on Stable Diffusion v1-4 and a randomly initialized UNet. The top row shows the marginal energy score $\boldsymbol { r } _ { t } ^ { \mathrm { e n g } }$ , while the bottom row compares the observed energy transition $\gamma _ { t }$ with the marginal energy transition $\gamma _ { t } ^ { \mathrm { m a r g } } = 1 - \bar { \alpha } _ { t }$ . Stable Diffusion follows a reasonable diffusion energy transition, whereas the Random UNet exhibits severe energy collapse, indicating a diffusion-inconsistent trajectory.

For an exact forward-marginal sample, $\boldsymbol { r } _ { t } ^ { \mathrm { e n g } }$ would fluctuate around zero. However, an inversion trajectory need not be exactly centered at zero under this statistic. As shown in Figure $5 ,$ the main pathological behavior is not a mild two-sided deviation, but a severe deficit of latent energy along diffusion-inconsistent trajectories. Therefore, we do not minimize $( r _ { t } ^ { \mathrm { e n g } } ) ^ { 2 }$ and do not force $\boldsymbol { r } _ { t } ^ { \mathrm { e n g } }$ to vanish. Instead, we only penalize trajectories whose latent energy falls far below the expected diffusion energy envelope. The resulting marginal energy loss is

$$
\mathcal { L } _ { \mathrm { e n g } } ( z _ { t } ) = \mathrm { R e L U } \left( - r _ { t } ^ { \mathrm { e n g } } - \tau _ { \mathrm { e n g } } \right) ^ { 2 } ,\tag{22}
$$

where $\tau _ { \mathrm { e n g } }$ controls the tolerated energy deficit before the penalty is activated. This one-sided hinge loss allows plausible inversion trajectories to deviate from the stochastic marginal center, while preventing trajectories whose latent energy is far below the expected diffusion energy envelope. It avoids over-constraining plausible trajectories on the upper side, and focuses the regularization on the failure mode that is most strongly associated with diffusion-inconsistent inversion paths.

Algorithm 1 Diffusion-Consistent Text-Free Inversion   
1: Input: Concept-erased denoising model $\epsilon _ { \theta } ,$ target image   
latent $z _ { \mathrm { 0 } } ,$ DDIM steps $T ,$ null-text condition $c _ { \mathrm { n u l l } }$ , max   
optimization iterations $K ,$ , learning rate $\eta ,$ initialization   
fraction $\rho \in [ 0 , 1 )$ , regularization weight $\lambda _ { \mathrm { e n g } } ,$ energy   
tolerance $\tau _ { \mathrm { e n g } } .$   
2: Output: Seed latent $z _ { T } ^ { * } .$   
3: Set the initialization timestep $t _ { a } \gets \lfloor \rho T \rfloor$   
4: Sample $\epsilon \sim \mathcal { N } ( 0 , I )$   
5: Set $\hat { z } _ { t _ { a } } \gets z _ { t _ { a } } ^ { \mathrm { f m } } = \sqrt { \bar { \alpha } _ { t _ { a } } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t _ { a } } } \epsilon .$   
6: for $t \gets t _ { a } + 1$ to $T$ do   
7: Initialize $z _ { t } \gets f _ { \theta } ( z _ { t - 1 } , t , c _ { \mathrm { n u l l } } )$ using $\operatorname { E q . } ( 8 ) .$   
8: for $k \gets 1$ to $K$ do   
9: ${ \mathcal L } _ { \mathrm { f p } } \gets \| f _ { \theta } ^ { * } ( z _ { t } , z _ { t - 1 } , t , c _ { \mathrm { n u l l } } ) - z _ { t } \| _ { 2 } ^ { 2 }$   
10: Compute $\boldsymbol { r } _ { t } ^ { \mathrm { e n g } }$ using Eq. (19).   
11: $\mathcal { L } _ { \mathrm { e n g } } {  } \mathrm { R e L U } ( - r _ { t } ^ { \mathrm { e n g } } - \tau _ { \mathrm { e n g } } ) ^ { 2 }$   
12: $\mathcal { L } _ { t } ^ { \mathrm { T I N A + } }  \mathcal { L } _ { \mathrm { f p } } + \lambda _ { \mathrm { e n g } } \mathcal { L } _ { \mathrm { e n g } }$   
13: $z _ { t } \gets z _ { t } - \eta \nabla _ { z _ { t } } \mathcal { L } _ { t } ^ { 1 }$ INA+   
14: end for   
15: end for   
16: $z _ { T } ^ { * } \gets z _ { T }$   
17: return $z _ { T } ^ { * }$

## 6.2.2 Forward Marginal Initialization

The marginal energy constraint regularizes the overall scale of the discovered trajectory, but the very low-noise region near $z _ { \mathrm { 0 } }$ remains delicate. Let $\sigma _ { t } ^ { 2 } = 1 - \bar { \alpha } _ { t }$ . When t is close to zero, $\sigma _ { t } ^ { 2 }$ is small, so the forward marginal $q ( \boldsymbol { z } _ { t } | \boldsymbol { z } _ { 0 } )$ has very small variance and becomes highly concentrated around $z _ { 0 } .$ Accordingly, the conditional variance $v _ { t }$ in Eq. (18) also becomes small, making the standardized energy score overly sensitive to tiny numerical deviations. Moreover, the DDIM inversion update in this region is weakly informative, since adjacent noise levels are very close and the update behaves almost like an identity mapping. These properties make the earliest inversion steps fragile for trajectory discovery.

Motivated by these observations, we avoid performing step-by-step inversion in the earliest low-noise segment. Instead, we use forward marginal initialization to place the trajectory directly at a stable intermediate state. Specifically, we choose an initialization timestep $t _ { a }$ according to an initialization fraction $\rho \in [ 0 , 1 )$ . In practice, we set $\dot { t } _ { a } = \lfloor \rho T \rfloor$ where $\rho \in [ 0 , 1 )$ denotes the fraction of early inversion steps replaced by the forward marginal initialization. Given the target latent $z _ { 0 } ,$ , we sample $\epsilon \sim \mathcal { N } ( 0 , I )$ and construct

$$
z _ { t _ { a } } ^ { \mathrm { f m } } = \sqrt { \bar { \alpha } _ { t _ { a } } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t _ { a } } } \epsilon .\tag{23}
$$

This operation samples an intermediate latent from the forward marginal associated with $z _ { 0 } ,$ , bypassing the fragile low-noise inversion segment. Starting from $\hat { z } _ { t _ { a } } ^ { \mathrm { f m } } .$ , TINA+ continues the optimized text-free inversion toward the seed latent $z _ { T } ^ { * }$ using the fixed-point objective and the marginal energy constraint.

## 6.3 Overall Objective and Algorithm

We now summarize the overall diffusion-consistent trajectory discovery procedure. Given a target latent $z _ { \mathrm { 0 } } ,$ TINA+ first initializes the trajectory at a forward-marginal state $z _ { t _ { a } } ^ { \mathrm { f m } }$ through forward marginal initialization to bypass the fragile low-noise segment. It then proceeds from $t _ { a } + 1$ to $T$ and optimizes each intermediate latent $z _ { t }$ under two objectives: fixed-point consistency for accurate inversion and marginal energy regularization for diffusion consistency.

TABLE 1  
Summary of the unlearning and attack methods evaluated in this work, together with their venues and years.
<table><tr><td>Method</td><td>Venue Year</td></tr><tr><td>Unlearning Methods</td></tr><tr><td>ESD [32] ICCV 2023</td></tr><tr><td>AC [33] ICCV 2023</td></tr><tr><td>FMN [34] CVPRW 2024</td></tr><tr><td>UCE [35] WACV 2024</td></tr><tr><td>SH [48] ECCV 2024</td></tr><tr><td>MACE [45] CVPR 2024</td></tr><tr><td>SPM [46] CVPR 2024</td></tr><tr><td>RECE [50] ECCV 2024</td></tr><tr><td>AdvUnlearn [39] NeurIPS 2024</td></tr><tr><td>SalUn [47] ICLR 2024</td></tr><tr><td>EraseDiff [49] CVPR 2025</td></tr><tr><td>STEREO [40] CVPR 2025</td></tr><tr><td>Attack Methods</td></tr><tr><td>PEZ [51] NeurIPS 2023</td></tr><tr><td>MMA [52] CVPR 2024</td></tr><tr><td>ICML 2024</td></tr><tr><td>P4D [36]</td></tr><tr><td>UDA [28] ECCV 2024</td></tr><tr><td>RAB [37] ICLR 2024 2024</td></tr><tr><td>CCE [38] ICLR</td></tr></table>

The final per-timestep objective combines the fixed-point consistency loss and the marginal energy loss:

$$
\begin{array} { r } { \mathcal { L } _ { t } ^ { \mathrm { T I N A + } } = \mathcal { L } _ { \mathrm { f p } } ( z _ { t } ) + \lambda _ { \mathrm { e n g } } \mathcal { L } _ { \mathrm { e n g } } ( z _ { t } ) , } \end{array}\tag{24}
$$

where $\lambda _ { \mathrm { e n g } }$ controls the strength of the marginal energy regularization. When $\lambda _ { \mathrm { e n g } } ~ = ~ 0 ,$ the marginal energy loss is disabled. When $\rho = 0 ,$ , forward marginal initialization is disabled. Disabling both components yields the fixed-pointonly variant used in our ablation study.

After obtaining the seed latent $z _ { T } ^ { * } ,$ we feed it back into the same concept-erased model and perform sampling to generate an image. The generated image is then evaluated by concept classifiers to determine whether the concept that should have been erased is regenerated.

## 7 EXPERIMENTS

## 7.1 Experimental Setup

Table 1 summarizes the unlearning and attack methods evaluated in this work, together with their publication venues and years.

## 7.1.1 Unlearned DMs to be attacked.

The field of machine unlearning for diffusion models is advancing at a rapid pace. For our evaluation, we select target unlearning methods based on two criteria: the public availability of their source code and the reproducibility of their reported results. Specifically, our evaluation targets twelve prominent, open-source unlearning methods: (1)

TABLE 2  
Quantitative comparison of Attack Success Rates (ASR, in %) on the nudity erasure task. We evaluate our TINA+ against five baselines across eight prominent unlearning defenses. AdvUn. denotes AdvUnlearn. Bold indicates the best-performing attack.
<table><tr><td></td><td>ESD</td><td>FMN</td><td>UCE</td><td>MACE</td><td>RECE</td><td>AdvUn.</td><td>SalUn</td><td>STEREO</td><td>AVG.</td></tr><tr><td>PEZ</td><td>11.86</td><td>62.71</td><td>25.42</td><td>8.47</td><td>15.25</td><td>1.69</td><td>0.00</td><td>0.00</td><td>15.68</td></tr><tr><td>MMA</td><td>13.10</td><td>67.00</td><td>32.60</td><td>6.00</td><td>22.80</td><td>1.70</td><td>1.70</td><td>5.50</td><td>18.80</td></tr><tr><td>RAB</td><td>50.53</td><td>97.89</td><td>29.47</td><td>6.32</td><td>10.53</td><td>2.11</td><td>0.00</td><td>8.42</td><td>25.66</td></tr><tr><td>P4D</td><td>69.01</td><td>97.89</td><td>76.06</td><td>75.35</td><td>66.20</td><td>18.31</td><td>15.49</td><td>24.65</td><td>55.37</td></tr><tr><td>UDA</td><td>76.05</td><td>97.89</td><td>78.87</td><td>81.69</td><td>63.38</td><td>23.24</td><td>13.38</td><td>25.35</td><td>57.48</td></tr><tr><td>TINA+</td><td>86.44</td><td>100.00</td><td>97.46</td><td>93.22</td><td>93.22</td><td>92.37</td><td>73.73</td><td>97.46</td><td>91.74</td></tr></table>

TABLE 3

Comparison of Attack Success Rates (ASR, in %) on the artistic style erasure task. We report the performance of our TINA+ against five baselines across eight unlearning methods. AdvUn. denotes AdvUnlearn. Bold denotes the highest ASR.
<table><tr><td></td><td>ESD</td><td>FMN</td><td>AC</td><td>MACE</td><td>SPM</td><td>RECE</td><td>AdvUn.</td><td>STEREO</td><td>AVG.</td></tr><tr><td>PEZ</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.9</td></tr><tr><td>MMA</td><td>0.0</td><td>2.0</td><td>6.0</td><td>0.0</td><td>14.0</td><td>14.0</td><td>0.0</td><td>0.0</td><td>4.5</td></tr><tr><td>RAB</td><td>6.0</td><td>6.0</td><td>14.0</td><td>4.0</td><td>48.0</td><td>20.0</td><td>0.0</td><td>0.0</td><td>12.3</td></tr><tr><td>P4D</td><td>30.0</td><td>54.0</td><td>68.0</td><td>42.0</td><td>78.0</td><td>62.0</td><td>0.0</td><td>0.0</td><td>41.8</td></tr><tr><td>UDA</td><td>32.0</td><td>56.0</td><td>77.0</td><td>56.0</td><td>88.0</td><td>64.0</td><td>2.0</td><td>0.0</td><td>46.9</td></tr><tr><td>TINA+</td><td>60.0</td><td>70.0</td><td>72.0</td><td>72.0</td><td>74.0</td><td>72.0</td><td>68.0</td><td>46.0</td><td>66.8</td></tr></table>

ESD (erased stable diffusion) [32], (2) FMN (Forget-Me-Not) [34], (3) AC (ablating concepts) [33], (4) UCE (unified concept editing) [35], (5) EraseDiff [49], (6) SH [48], (7) MACE (Mass Concept Erasure) [45], (8) SPM (concept-SemiPermeable Membrane) [46], (9) RECE [50], (10) AdvUnlearn [39], (11) SalUn (saliency unlearning) [47], (12) STEREO [40]. We note that these unlearning methods are often specialized for specific domains (e.g., nudity) rather than being universal. Consequently, our evaluation is contextualized, assessing each method only on its intended unlearning tasks.

## 7.1.2 Baseline Attack Methods.

We evaluate the attack performance of TINA+ against five prevailing baseline attack methods: (1) PEZ [51], (2) MMA [52], (3) Prompting4Debugging (P4D) [36], (4) UnlearnDiffAtk (UDA) [28], and (5) Ring-A-Bell (RAB) [37]. The fundamental characteristic and shared limitation of all these baselines is their text-centric nature. They exclusively target the textual input pathway in an attempt to circumvent textual defenses. These approaches range from black-box jailbreaking techniques that discover alternative text prompts (e.g., PEZ, MMA, RAB) to white-box attacks that leverage model access to optimize for adversarial text conditions (e.g., P4D, UDA). Critically, all these existing attacks remain tethered to the text-conditioning mechanism, a pathway that TINA+ is designed to bypass entirely. For a unified comparison, we evaluate these baselines across the nudity, style, and object erasure tasks. We defer the comparison with textual-inversion attacks such as CCE [38] to a dedicated analysis that contrasts text inversion with our image-inversion approach.

## 7.1.3 Image Setup.

Our method requires a set of representative images, each exemplifying a concept targeted for erasure. We obtain these images by adopting the generation setup established in the UnlearnDiffAtk (UDA) study [28]. This process involves using the original Stable Diffusion v1.4 model, which is the pre-trained checkpoint before any unlearning methods were applied, to generate a collection of ground-truth images. The generation is guided by text prompts sourced from the same standard benchmarks used to evaluate the baseline attacks. Specifically, for nudity concepts, we use prompts from the I2P dataset [30]. For artistic styles, we focus our experiments on the Van Gogh style, using art-related prompts from the ESD evaluation setup [32]. For objects, we evaluate four concepts (Church, Garbage Truck, Parachute, and Tench), using prompts generated via GPT-4 corresponding to Imagenette classes, following the methodology in [28]. For celebrity identity erasure, we extend the data construction protocol of UnlearnDiffAtk [28] to the identity domain. Specifically, we use Stable Diffusion v1.4 to generate candidate images and retain 50 target cases for each of Taylor Swift, Elon Musk, and Adam Lambert using the open-source GIPHY Celebrity Detector (GCD)<sup>1</sup>.

## 7.1.4 Implementation Details.

All experiments are built upon the Stable Diffusion v1.4 checkpoint. TINA+ performs text-free inversion using a DDIMScheduler with 50 inversion steps, following the formulation in Section 6. Within the inner optimization loop at each inversion timestep t, we perform $K = 2 5$ refinement iterations to optimize the fixed-point and marginal energy objectives. We use AdamW [55] with a learning rate of $\eta = 0 . 0 0 1$ , and set the forward marginal initialization ratio to $\rho ~ = ~ 0 . 1$ , the marginal energy weight to $\lambda _ { \mathrm { e n g } } ~ = ~ 1 . 0 ,$ and the energy tolerance to $\tau _ { \mathrm { e n g } } ~ = ~ 1 0$ . Marginal energy regularization is applied over $t / \bar { T } \in [ 0 . 1 , 1 . 0 ]$ . After obtaining the optimized seed latent $z _ { T } ^ { * } ,$ , we feed it back into the same concept-erased model under the null-text condition to generate the evaluation image. This generation process uses the LMSDiscreteScheduler with 50 sampling steps and a Classifier-Free Guidance (CFG [56]) scale of 7.5. All experiments were conducted on a single NVIDIA A100 GPU.

![](images/b53a9eff88275f9d1a1c8b85d969f7dbdff9cc0db75ab4ad2c03d28c5d426146.jpg)  
Fig. 6. Qualitative comparison of PEZ, MMA, RAB, P4D, UDA, and TINA+ on the nudity erasure task. Each column corresponds to an unlearning defense. P4D, UDA, and TINA+ use the same I2P target, whereas PEZ, MMA, and RAB use representative cases from their native evaluation protocols. Sensitive content is redacted.

## 7.1.5 Evaluation Metrics.

Our evaluation methodology is primarily aligned with the protocol established by UnlearnDiffAtk (UDA) [28]. To quantitatively measure attack efficacy, we report the Attack Success Rate (ASR), defined as the percentage of generated images that are successfully identified by a post-generation classifier as containing the forbidden concept, thereby indicating a successful bypass of the unlearning safeguard. We employ a suite of specialized classifiers corresponding to each unlearning task. For harmful concept unlearning, we utilize NudeNet to detect nudity. For style unlearning, we adopt the 129-class ViT-base style classifier [57], [28], which was pre-trained on ImageNet, fine-tuned on the WikiArt dataset [58]. For object unlearning, we employ a standard ImageNet-pretrained ResNet-50 [59] for generated image classification. For celebrity identity erasure, we use GCD as the post-generation identity classifier. A generated image is counted as a successful recovery only when GCD detects a face and its top-1 predicted identity exactly matches the target celebrity. For the diffusion-consistency analysis, we additionally compare each generated image with its VAE-reconstructed target using LPIPS [60], DINOv2 feature similarity [61], and DreamSim [62]. DreamSim is a holistic perceptual distance trained to align with human similarity judgments. On the Random-UNet negative control, higher LPIPS and DreamSim, lower DINOv2 similarity, and lower ASR all indicate stronger suppression of spurious target reconstruction.

## 7.2 Experiment Results

We now empirically validate the efficacy of TINA+. Our experiments are designed to test the ability of TINA+ to bypass a wide spectrum of unlearning defenses, from foundational methods to the current state-of-the-art, and compare its performance directly against existing text-centric attacks. The results for nudity, style, object, and celebrity identity erasure are summarized in Tables 2, 3, 4, and 5, respectively.

TABLE 4  
Attack Success Rates (ASR, in %) of TINA+ and baselines on object erasure across four concepts (Church, Garbage Truck, Parachute, and Tench). AdvUn. denotes AdvUnlearn. Bold indicates the best-performing attack.
<table><tr><td></td><td>FMN</td><td>SPM</td><td>ESD</td><td>EraseDiff</td><td>SalUn</td><td>SH</td><td>AdvUn.</td><td>STEREO</td><td>AVG.</td></tr><tr><td colspan="10">Church</td></tr><tr><td>PEZ</td><td>32.00</td><td>34.00</td><td>2.00</td><td>2.00</td><td>12.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>10.25</td></tr><tr><td>MMA</td><td>32.00</td><td>26.00</td><td>6.00</td><td>10.00</td><td>8.00</td><td>0.00</td><td>0.00</td><td>2.00</td><td>10.50</td></tr><tr><td>RAB</td><td>58.00</td><td>68.00</td><td>14.00</td><td>28.00</td><td>4.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>21.50</td></tr><tr><td>P4D</td><td>94.00</td><td>94.00</td><td>58.00</td><td>42.00</td><td>62.00</td><td>0.00</td><td>2.00</td><td>8.00</td><td>45.00</td></tr><tr><td>UDA</td><td>98.00</td><td>94.00</td><td>66.00</td><td>50.00</td><td>68.00</td><td>4.00</td><td>8.00</td><td>10.00</td><td>49.75</td></tr><tr><td>TINA+</td><td>92.00</td><td>90.00</td><td>90.00</td><td>94.00</td><td>90.00</td><td>90.00</td><td>92.00</td><td>78.00</td><td>89.50</td></tr><tr><td colspan="10">Garbage Truck</td></tr><tr><td>PEZ</td><td>36.00</td><td>18.00</td><td>0.00</td><td>4.00</td><td>4.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>7.75</td></tr><tr><td>MMA</td><td>28.00</td><td>4.00</td><td>0.00</td><td>0.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>4.25</td></tr><tr><td>RAB</td><td>40.00</td><td>34.00</td><td>8.00</td><td>20.00</td><td>20.00</td><td>0.00</td><td>2.00</td><td>0.00</td><td>15.50</td></tr><tr><td>P4D UDA</td><td>98.00</td><td>76.00</td><td>24.00</td><td>44.00</td><td>38.00</td><td>8.00</td><td>4.00</td><td>2.00</td><td>36.75</td></tr><tr><td></td><td>100.00</td><td>80.00</td><td>32.00</td><td>46.00</td><td>36.00</td><td>6.00</td><td>8.00</td><td>6.00</td><td>39.25</td></tr><tr><td>TINA+</td><td>90.00</td><td>86.00</td><td>74.00</td><td>76.00</td><td>78.00</td><td>84.00</td><td>80.00</td><td>66.00</td><td>79.25</td></tr><tr><td colspan="10">Parachute</td></tr><tr><td colspan="10"></td></tr><tr><td>PEZ MMA</td><td>22.00</td><td>22.00</td><td>0.00</td><td>6.00</td><td>6.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>7.00</td></tr><tr><td>RAB</td><td>32.00 38.00</td><td>20.00</td><td>0.00 2.00</td><td>8.00</td><td>6.00 0.00</td><td>2.00 8.00</td><td>2.00</td><td>0.00</td><td>8.75</td></tr><tr><td>P4D</td><td>100.00</td><td>14.00 92.00</td><td>50.00</td><td>14.00 68.00</td><td>70.00</td><td>24.00</td><td>0.00 14.00</td><td>0.00 2.00</td><td>9.50 52.50</td></tr><tr><td>UDA</td><td>100.00</td><td>94.00</td><td>58.00</td><td>76.00</td><td>80.00</td><td>24.00</td><td>16.00</td><td>6.00</td><td>56.75</td></tr><tr><td>TINA+</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>94.00</td><td>92.00</td><td>84.00</td><td>90.00</td><td>90.00</td><td>90.00</td><td>88.00</td><td>80.00</td><td>88.50</td></tr><tr><td colspan="10">Tench</td></tr><tr><td>PEZ</td><td>6.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.25</td></tr><tr><td>MMA</td><td>24.00</td><td>16.00</td><td>2.00</td><td>0.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>5.50</td></tr><tr><td>RAB</td><td>12.00</td><td>24.00</td><td>6.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>5.25</td></tr><tr><td>P4D</td><td>94.00</td><td>82.00</td><td>32.00</td><td>8.00</td><td>18.00</td><td>6.00</td><td>8.00</td><td>0.00</td><td>31.00</td></tr><tr><td>UDA</td><td>100.00</td><td>88.00</td><td>46.00</td><td>2.00</td><td>12.00</td><td>6.00</td><td>2.00</td><td>2.00</td><td>32.25</td></tr><tr><td>TINA+</td><td>82.00</td><td>78.00</td><td>64.00</td><td>78.00</td><td>72.00</td><td>80.00</td><td>80.00</td><td>66.00</td><td>75.00</td></tr></table>

A focused comparison between text inversion and image inversion is reported in Table 6. A qualitative comparison for nudity erasure is provided in Figure 6.

## 7.2.1 Evaluation on Erased Models in Nudity Erasure.

Table 2 demonstrates the comprehensive success of TINA+ in the nudity erasure task. TINA+ achieves the highest ASR across all eight defenses. It reaches 86.44% against ESD and 93.22% against MACE, significantly outperforming all baselines. The most critical finding is the performance of TINA+ against defenses designed to be robust, such as AdvUnlearn, SalUn, and STEREO. Against these targets, text-based attacks are largely mitigated (e.g., UDA scores 23.24% on AdvUnlearn, RAB scores 0.00% on SalUn). In contrast, TINA+ maintains high ASRs (92.37%, 73.73%, and 97.46%), showing that it exploits a vulnerability that these defenses do not account for. Figure 6 provides complementary qualitative evidence. Under the shared I2P target used by P4D, UDA, and TINA+, the two text-based attacks become neutral or unrelated under stronger defenses such as AdvUnlearn, SalUn, and STEREO, whereas TINA+ consistently reconstructs the target content across all eight defenses. PEZ, MMA, and RAB are shown as representative results under their respective native evaluation protocols rather than as paired comparisons on the same target.

## 7.2.2 Evaluation on Erased Models in Style Erasure.

The results for artistic style erasure (Table 3) further reinforce this trend. TINA+ robustly bypasses the majority of defenses. Its 60.0% ASR against ESD starkly contrasts with the 32.0% from UDA. Furthermore, TINA+ shows dominant performance against robust methods like AdvUnlearn (68.0%) and STEREO (46.0%), where text-based attacks remain weak or ineffective. This finding demonstrates that even for abstract concepts like style, a text-free generative trajectory persists after erasure, which TINA+ successfully identifies and exploits. The Van Gogh example in Figure 7 further shows that, under STEREO, ordinary generation and UDA lose the target style and composition, whereas TINA+ closely recovers the target image.

## 7.2.3 Evaluation on Erased Models in Object Erasure.

The object erasure evaluation in Table 4 delivers the most compelling results. We evaluate four concepts (Church, Garbage Truck, Parachute, and Tench) against eight unlearning defenses. This task highlights the failure of text-centric unlearning against our text-free attack. Against modern defenses like SH and STEREO, text-based attacks (PEZ, MMA,

TABLE 5  
Attack Success Rates (ASR, in %) on celebrity identity erasure under ESD and STEREO. Bold indicates the best result for each identity under the same defense.
<table><tr><td>Attack</td><td>Taylor Swift</td><td>Elon Musk</td><td>Adam Lambert</td></tr><tr><td colspan="4">ESD</td></tr><tr><td>PEZ</td><td>2.00</td><td>0.00</td><td>4.00</td></tr><tr><td>MMA</td><td>2.00</td><td>0.00</td><td>2.00</td></tr><tr><td>RAB</td><td>12.00</td><td>6.00</td><td>4.00</td></tr><tr><td>P4D</td><td>98.00</td><td>100.00</td><td>90.00</td></tr><tr><td>UDA</td><td>98.00</td><td>100.00</td><td>92.00</td></tr><tr><td>TINA+</td><td>96.00</td><td>98.00</td><td>94.00</td></tr><tr><td colspan="4">STEREO</td></tr><tr><td>PEZ</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>MMA</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RAB</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>P4D</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>UDA</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>TINA+</td><td>94.00</td><td>46.00</td><td>64.00</td></tr></table>

TABLE 6

Comparison between text inversion (CCE) and image inversion (TINA+) across style, nudity, and object erasure. Results are ASRs (in %). Bold indicates the better inversion strategy.
<table><tr><td>Attack</td><td>Van Gogh</td><td>Nudity</td><td>Tench</td></tr><tr><td colspan="4">ESD</td></tr><tr><td>CCE</td><td>8.00</td><td>74.65</td><td>40.00</td></tr><tr><td>TINA+</td><td>60.00</td><td>86.44</td><td>64.00</td></tr><tr><td colspan="4">STEREO</td></tr><tr><td>CCE</td><td>4.00</td><td>16.90</td><td>2.00</td></tr><tr><td>TINA+</td><td>46.00</td><td>97.46</td><td>66.00</td></tr></table>

RAB, P4D, and UDA) largely fail, often with ASRs in the single digits. In contrast, TINA+ achieves uniformly high ASRs across concepts and defenses, with average ASRs of 89.50% (Church), 79.25% (Garbage Truck), 88.50% (Parachute), and 75.00% (Tench). Even on robust targets such as EraseDiff, SH, and STEREO, TINA+ remains effective (e.g., 78.00%, 80.00%, and 66.00% on Tench). This finding supports our central hypothesis: current SOTA methods merely sever text-image links, not the underlying visual knowledge, which TINA+ can access without any textual guidance. The Tench example in Figure 7 exhibits the same pattern for object erasure: the no-attack and UDA outputs deviate from the target object, while TINA+ restores its appearance and composition.

## 7.2.4 Evaluation on Erased Models in Celebrity Identity Erasure.

Table 5 extends the evaluation to three celebrity identities under ESD and STEREO. Under ESD, the black-box text attacks PEZ, MMA, and RAB remain weak (at most 12.00%), while the white-box P4D and UDA attacks recover identities effectively. TINA+ remains competitive, reaching 96.00%, 98.00%, and 94.00% on Taylor Swift, Elon Musk, and Adam Lambert, respectively. The distinction becomes much sharper under STEREO: all five text-centric attacks obtain 0.00% ASR for all identities, whereas TINA+ retains ASRs of 94.00%, 46.00%, and 64.00%. These results show that robust identity erasure can suppress attacks operating through textual conditions while leaving recoverable visual trajectories accessible to image inversion.

![](images/d4a3c8318a1b5ba89aadf20fcacb88c4e36459e41e496119a2f704e616cf6650.jpg)  
Fig. 7. Qualitative comparison on Van Gogh style and Tench object erasure under STEREO. Each row shows the VAE-reconstructed target, ordinary generation from the erased model, UDA, and TINA+. While the no-attack and UDA outputs deviate from the target concept or composition, TINA+ closely reconstructs the target-specific visual content.

![](images/8489bd1869a0e932329ece8678bd3f4e669900625551dbb2dff6f7ea763b8913.jpg)  
Fig. 8. Qualitative identity recovery on ESD and STEREO. Each row uses one target image for Taylor Swift, Elon Musk, or Adam Lambert. The no-attack outputs weaken or alter the target identity, whereas TINA+ reconstructs identity-specific facial characteristics under both defenses. For each celebrity, we select an index for which TINA+ is recognized as the target identity by GCD on both ESD and STEREO.

Figure 8 provides qualitative evidence complementary to Table 5. Across all three identities, the erased models ordinary outputs either drift toward a different face or lose the target-specific appearance altogether, with the effect particularly pronounced under STEREO. Starting from the corresponding target image, TINA+ nevertheless discovers a visual trajectory that restores recognizable identity cues under both defenses. This consistent recovery across identities supports the conclusion that the low text-attack ASR under robust erasure does not imply removal of the underlying visual identity knowledge.

![](images/ceeb1f80cf3061bcf0dc8af31036577b26c9f1234c9f25a1f3673c5c26faafef.jpg)  
Fig. 9. Roles of Forward Marginal Initialization (FMI) and marginal energy regularization in trajectory discovery on the Van Gogh benchmark. (a) On pretrained Stable Diffusion v1.4, FMI provides an initialization aligned with the expected marginal energy progression, and TINA+ follows the same plausible trajectory. (b) On the randomly initialized UNet negative control, FMI provides an aligned starting point but cannot prevent the subsequent energy collapse, whereas the full TINA+ objective substantially mitigates it. Curves report the mean over 50 target images.

## 7.2.5 Image Inversion versus Text Inversion.

Table 6 directly compares TINA+ with CCE [38], a representative textual-inversion attack, across style, nudity, and object concepts. TINA+ consistently outperforms CCE in all six settings. Under ESD, TINA+ improves the ASR from 8.00% to 60.00% on Van Gogh, from 74.65% to 86.44% on nudity, and from 40.00% to 64.00% on Tench. The advantage is even larger under STEREO, where CCE obtains only 4.00%, 16.90%, and 2.00%, while TINA+ reaches 46.00%, 97.46%, and 66.00%. This systematic gap supports our central argument: severing or hardening the textual pathway substantially weakens text inversion, but does not necessarily remove the diffusion-consistent visual trajectories exploited by TINA+.

## 7.3 Mechanism Analysis and Ablation Study

We next examine how Forward Marginal Initialization (FMI) and marginal energy regularization contribute to reliable trajectory discovery. For clarity, TINA denotes the fixedpoint-only method introduced in our conference version, while TINA w/ FMI additionally applies forward marginal initialization without marginal energy regularization. We contrast pretrained Stable Diffusion v1.4, which contains the target knowledge, with a randomly initialized UNet, which contains no learned target knowledge. Figure 9(a) shows the early trajectory on pretrained Stable Diffusion v1.4. Compared with DDIM inversion and TINA, adding FMI places the optimization at a substantially better aligned starting point, close to the expected marginal energy progression. TINA+ closely overlaps with TINA w/ FMI in this setting, indicating that the energy constraint remains inactive when the subsequent trajectory is already plausible rather than indiscriminately perturbing a valid inversion path.

The Random UNet negative control separates initialization quality from trajectory validity more clearly. As shown in Figure 9(b), FMI again starts near the marginal reference, but TINA w/ FMI subsequently develops a severe energy deficit. The full TINA+ objective instead intervenes during optimization and substantially mitigates this collapse.

![](images/ca630528068323b304a3c0830b2fb6c33a96c129cb1eba85803fbd13dee8c0d4.jpg)  
Fig. 10. Qualitative false-positive reconstruction on the Taylor Swift benchmark using a randomly initialized UNet. Although the model contains no learned target knowledge, DDIM inversion and unconstrained TINA closely reconstruct the targets, while FMI alone only partially disrupts these spurious results. Marginal energy regularization substantially degrades the diffusion-inconsistent reconstructions, causing all three TINA+ outputs to fail the identity detector.

This distinction also explains why energy validity cannot be reduced to a post-hoc rejection rule. A validity score could identify the collapsed Random-UNet trajectory after inversion, but the optimization would still return a visually recognizable false reconstruction, yielding poor qualitative evidence of failure. By incorporating the energy constraint into the optimization itself, TINA+ changes the discovered trajectory and turns the invalid reconstruction into an explicit generation failure.

Table 7 and Figure 10 provide complementary quantitative and qualitative evidence of false-positive suppression. Because the randomly initialized UNet contains no target knowledge, any recognizable target reconstruction constitutes a false positive rather than evidence of successful probing. As illustrated by the three Taylor Swift cases in Figure 10, DDIM inversion and TINA nevertheless reconstruct the target identity and scene surprisingly well, while TINA w/ FMI retains recognizable target characteristics despite introducing visible degradation. These examples show why visual reconstruction alone cannot establish that a model retains the target knowledge, and why an aligned initialization by itself does not guarantee a valid trajectory. In contrast, TINA+ substantially degrades the target-like reconstructions on the negative control, causing them to fail the identity detector. This qualitative transition is consistent with the 50-target results in Table 7. On the Taylor Swift benchmark, TINA+ increases LPIPS and DreamSim to 0.749 and 0.625, respectively, reduces DINO similarity to 0.400, and lowers ASR to 0.0%. The same pattern holds for Van Gogh, where TINA+ increases LPIPS and DreamSim to 0.651 and 0.493, respectively, reduces DINO similarity to 0.434, and lowers the false-positive ASR to 0.0%. Together with Figure 9, these results show that FMI supplies an aligned starting point, whereas marginal energy regularization is necessary to prevent subsequent optimization from exploiting diffusion-inconsistent paths.

![](images/5c067cac4af6c8715679a14f6a415a209cc675093964fc1a8e5a2fb12ed33f5d.jpg)

TABLE 7  
False-positive suppression throughout the progression from DDIM inversion to TINA+ on the Van Gogh and Taylor Swift benchmarks. The randomly initialized UNet contains no learned target knowledge and therefore serves as a negative control: higher LPIPS and DreamSim, lower DINO, and lower ASR indicate stronger suppression of spurious target reconstruction.
<table><tr><td colspan="5">Van Gogh</td><td colspan="4">Taylor Swift</td></tr><tr><td>Method</td><td>LPIPS ↑</td><td>DreamSim ↑</td><td>DINO↓</td><td>ASR↓</td><td>LPIPS ↑</td><td>DreamSim ↑</td><td>DINO↓</td><td>ASR↓</td></tr><tr><td>DDIM</td><td>0.396</td><td>0.212</td><td>0.725</td><td>48.0</td><td>0.286</td><td>0.164</td><td>0.695</td><td>86.0</td></tr><tr><td>TINA</td><td>0.361</td><td>0.181</td><td>0.769</td><td>60.0</td><td>0.245</td><td>0.140</td><td>0.730</td><td>88.0</td></tr><tr><td>TINA w/FMI</td><td>0.482</td><td>0.270</td><td>0.633</td><td>28.0</td><td>0.484</td><td>0.256</td><td>0.588</td><td>60.0</td></tr><tr><td>TINA+</td><td>0.651</td><td>0.493</td><td>0.434</td><td>0.0</td><td>0.749</td><td>0.625</td><td>0.400</td><td>0.0</td></tr></table>

(a) Optimized Noise z ¤

(b) ESD Mid-Block Features  
Fig. 11. t-SNE visualization of (a) the optimized initial noises $z _ { T } ^ { * }$ produced by TINA+ and (b) their corresponding deep ESD UNet activations, extracted from the last ResNet block of mid\_block, for four erased concepts. Each concept contains 50 target images.  
![](images/38f5f227688455e4e7d325b936fe742b7c6c394de9d68bdb688fada7dbcd3de4.jpg)  
Fig. 12. Generalization of TINA+ to a DiT-based architecture $( \mathtt { P i x A r t - X L - 2 - 5 1 } 2 \mathbf { x } 5 1 2 )$ . Each column shows a target image, the ordinary output of the ESD-erased model, and the corresponding TINA+ reconstruction. ESD removes the Parachute concept under normal sampling, whereas TINA+ recovers diverse target-specific parachutes.

## 7.4 Latent Embedding Analysis

Having examined the diffusion consistency of the discovered trajectories, we further investigate the internal responses that they elicit in the corresponding erased models. For four erased concepts (Tench, Church, Parachute, and Garbage Truck), we collect 50 initial noises z<sup>∗</sup> optimized by TINA+ against each corresponding ESD-sanitized model and visualize both the noises and their induced UNet activations using t-SNE. We then feed these optimized noises into their corresponding ESD UNets under a null-text condition and extract the intermediate representations through a forward inference. For visualization, we apply t-SNE (perplexity 30, with PCA initialization) to both the flattened noise vectors and the global-average-pooled features extracted from the last ResNet block of mid\_block. As shown in Figure 11, the optimized noises themselves exhibit little concept-wise separation in the latent space, consistent with their noise-like nature. However, their corresponding internal UNet activations become clearly separable by concept, forming distinct clusters. This result shows that although the optimized $z _ { T } ^ { * }$ remains unstructured in the input latent space, concept-wise structure emerges in the internal responses of the corresponding erased models. Together with the reconstruction results, this observation is consistent with TINA+ activating residual concept-related representations through text-free trajectories.

## 7.5 Generalization to DiT-Based Architectures

While the main experiments focus on UNet-based diffusion models (e.g., Stable Diffusion v1.4), an important question is whether TINA+ generalizes to fundamentally different model architectures. To investigate this, we evaluate TINA+ on PixArt-XL-2-512x512 [17], a text-to-image model built on the Diffusion Transformer (DiT) architecture [18] rather than a UNet backbone.

We conduct this evaluation on the “Parachute” concept. Additionally, as there are no publicly available erased versions of this DiT model, we first apply the ESD [32] method to erase the parachute concept from the model, and then perform our TINA+ attack on the resulting erased model.

Across 50 target images, the erased DiT model obtains a 0.0% ASR under normal prompt-based sampling, whereas TINA+ successfully recovers the Parachute concept in 42 cases, achieving an 84.0% ASR. As shown in Figure 12, the ordinary ESD outputs lose the target concept and scene content, while TINA+ reconstructs diverse target-specific parachutes from the same erased DiT model. These results demonstrate that the vulnerability exposed by TINA+ is not limited to UNet-based models, but also extends to DiTbased models.

## 8 CONCLUSION

We exposed a fundamental limitation of the text-centric concept erasure paradigm: severing text-image links is insufficient to truly remove the underlying visual knowledge. We introduced TINA+, a diffusion-consistent text-free attack that probes residual visual knowledge while bypassing textual controls entirely. TINA+ discovers text-free generative trajectories while suppressing spurious inversion paths that are inconsistent with the diffusion process. Extensive experiments across diverse concepts, erasure methods, and different model architectures demonstrate that TINA+ can reliably expose residual visual knowledge in concept-erased models. These findings indicate that existing methods may obscure visual knowledge by disrupting its textual associations rather than fully removing it, motivating future unlearning techniques that operate directly on internal visual representations.

## REFERENCES

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[2] H. Zhang, M. Liu, Y. Li, M. Yan, Z. Gao, X. Chang, and L. Nie, “Attribute-guided collaborative learning for partial person reidentification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 14 144–14 160, 2023.

[3] Z. Li, Y. Xie, R. Shao, G. Chen, W. Guan, D. Jiang, Y. Wang, and L. Nie, “Optimus-3: Dual-router aligned mixture-of-experts agent with dual-granularity reasoning-aware policy optimization,” arXiv preprint arXiv:2506.10357, 2025.

[4] X. Wang, T. Dai, H. Bai, Y. Zhao, and J. Xiao, “Unifying reconstruction and density estimation via invertible contraction mapping in one-class classification,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[5] X. Wang, X. Wang, H. Bai, E. G. Lim, and J. Xiao, “Decad: Decoupling anomalies in latent space for multi-class unsupervised anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 21 568–21 577.

[6] F. Liu, Y. Liu, H. Chen, Z. Cheng, L. Nie, and M. Kankanhalli, “Understanding before recommendation: Semantic aspect-aware review exploitation via large language models,” ACM Transactions on Information Systems, vol. 43, no. 2, pp. 1–26, 2025.

[7] K. Wang, Y. Hu, H. Liu, J. Shao, and L. Nie, “Cross-modal representation shift refinement for point-supervised video moment retrieval,” ACM Transactions on Information Systems, vol. 44, no. 3, pp. 1–30, 2026.

[8] K. Wang, H. Liu, L. Jie, Z. Li, Y. Hu, and L. Nie, “Explicit granularity and implicit scale correspondence learning for pointsupervised video moment localization,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 9214–9223.

[9] K. Wang, Y. Hu, H. Liu, L. Jie, and L. Nie, “Redundancy mitigation: Towards accurate and efficient image-text retrieval,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[10] K. Wang, Y. Hu, Z. Li, H. Liu, Q. Xiang, and L. Nie, “ ViSAGE @ NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results ,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2026.

[11] W. Guan, H. Zhang, M. Liu, Q. Xiang, Y. Wang, and L. Nie, “Spaceera++: A unified framework towards 3d spatial reasoning in video,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–15, 2026.

[12] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[13] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mc-Grew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021.

[14] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International conference on machine learning. Pmlr, 2021, pp. 8821– 8831.

[15] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in neural information processing systems, vol. 35, pp. 36 479–36 494, 2022.

[16] Q. Xiang, M. Zhang, Y. Shang, J. Wu, Y. Yan, and L. Nie, “Dkdm: Data-free knowledge distillation for diffusion models with any architecture,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 2955–2965.

[17] J. Chen, J. YU, C. GE, L. Yao, E. Xie, Z. Wang, J. Kwok, P. Luo, H. Lu, and Z. Li, “Pixart-\$\alpha\$: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id= eAKmQPe3m1

[18] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[19] X. Ju, A. Zeng, Y. Bian, S. Liu, and Q. Xu, “Pnp inversion: Boosting diffusion-based editing with 3 lines of code,” in The Twelfth International Conference on Learning Representations, 2023.

[20] D. Miyake, A. Iohara, Y. Saito, and T. Tanaka, “Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 2063–2072.

[21] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Null-text inversion for editing real images using guided diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6038–6047.

[22] Z. Zhang, M. Lin, S. YAN, and R. Ji, “Easyinv: Toward fast and better ddim inversion,” in Forty-second International Conference on Machine Learning, 2025.

[23] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25 278–25 294, 2022.

[24] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in 32nd USENIX security symposium (USENIX Security 23), 2023, pp. 5253–5270.

[25] H. H. Jiang, L. Brown, J. Cheng, M. Khan, A. Gupta, D. Workman, A. Hanna, J. Flowers, and T. Gebru, “Ai art and its impact on artists,” in Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society, 2023, pp. 363–374.

[26] K. Roose, “An ai-generated picture won an art prize. artists aren’t happy.” New York Times, vol. 16, no. 01, p. 2025, 2022.

[27] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Diffusion art or digital forgery? investigating data replication in diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6048–6058.

[28] Y. Zhang, J. Jia, X. Chen, A. Chen, Y. Zhang, J. Liu, K. Ding, and S. Liu, “To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now,” in European Conference on Computer Vision. Springer, 2024, pp. 385– 403.

[29] T. Hunter, “Ai porn is easy to make now. for women, that’s a nightmare.” The Washington Post, pp. NA–NA, 2023.

[30] P. Schramowski, M. Brack, B. Deiseroth, and K. Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 522–22 531.

[31] H. Xu, T. Zhu, L. Zhang, W. Zhou, and P. S. Yu, “Machine unlearning: A survey,” ACM Comput. Surv., vol. 56, no. 1, Aug. 2023. [Online]. Available: https://doi.org/10.1145/3603620

[32] R. Gandikota, J. Materzynska, J. Fiotto-Kaufman, and D. Bau, “Erasing concepts from diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 2426–2436.

[33] N. Kumari, B. Zhang, S.-Y. Wang, E. Shechtman, R. Zhang, and J.-Y. Zhu, “Ablating concepts in text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22 691–22 702.

[34] G. Zhang, K. Wang, X. Xu, Z. Wang, and H. Shi, “Forget-menot: Learning to forget in text-to-image diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 1755–1764.

[35] R. Gandikota, H. Orgad, Y. Belinkov, J. Materzynska, and D. Bau, ´ “Unified concept editing in diffusion models,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 5111–5120.

[36] Z.-Y. Chin, C.-M. Jiang, C.-C. Huang, P.-Y. Chen, and W.-C. Chiu, “Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 8468–8486.

[37] Y.-L. Tsai, C.-Y. Hsu, C. Xie, C.-H. Lin, J.-Y. Chen, B. Li, P.-Y. Chen, C.-M. Yu, and C.-Y. Huang, “Ring-a-bell! how reliable are concept removal methods for diffusion models?” in International Conference on Learning Representations, 2024.

[38] M. Pham, K. O. Marshall, N. Cohen, G. Mittal, and C. Hegde, “Circumventing concept erasure methods for text-to-image generative models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=ag3o2T51Ht

[39] Y. Zhang, X. Chen, J. Jia, Y. Zhang, C. Fan, J. Liu, M. Hong, K. Ding, and S. Liu, “Defensive unlearning with adversarial training for robust concept erasure in diffusion models,” Advances in neural information processing systems, vol. 37, pp. 36 748–36 776, 2024.

[40] K. Srivatsan, F. Shamshad, M. Naseer, V. M. Patel, and K. Nandakumar, “Stereo: A two-stage framework for adversarially robust concept erasing from text-to-image diffusion models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 23 765–23 774.

[41] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[42] Q. Xiang, M. Zhang, H. Zhang, K. Wang, J. Hou, and L. Nie, “Tina: Text-free inversion attack for unlearned text-to-image diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 30 076–30 086.

[43] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.

[44] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[45] S. Lu, Z. Wang, L. Li, Y. Liu, and A. W.-K. Kong, “Mace: Mass concept erasure in diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6430–6440.

[46] M. Lyu, Y. Yang, H. Hong, H. Chen, X. Jin, Y. He, H. Xue, J. Han, and G. Ding, “One-dimensional adapter to rule them all: Concepts diffusion models and erasing applications,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7559–7568.

[47] C. Fan, J. Liu, Y. Zhang, E. Wong, D. Wei, and S. Liu, “Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation,” in International Conference on Learning Representations, 2024.

[48] J. Wu and M. Harandi, “Scissorhands: Scrub data influence via connection sensitivity in networks,” in European Conference on Computer Vision. Springer, 2024, pp. 367–384.

[49] J. Wu, T. Le, M. Hayat, and M. Harandi, “Erasing undesirable influence in diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 28 263–28 273.

[50] C. Gong, K. Chen, Z. Wei, J. Chen, and Y.-G. Jiang, “Reliable and efficient concept erasure of text-to-image diffusion models,” in European Conference on Computer Vision. Springer, 2024, pp. 73–88.

[51] Y. Wen, N. Jain, J. Kirchenbauer, M. Goldblum, J. Geiping, and T. Goldstein, “Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery,” Advances in Neural Information Processing Systems, vol. 36, pp. 51 008–51 025, 2023.

[52] Y. Yang, R. Gao, X. Wang, T.-Y. Ho, N. Xu, and Q. Xu, “Mmadiffusion: Multimodal attack on diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7737–7746.

[53] S. Banach, “Sur les operations dans les ensembles abstraits et leur´ application aux equations int´ egrales,”´ Fundamenta mathematicae, vol. 3, no. 1, pp. 133–181, 1922.

[54] J. M. Ortega and W. C. Rheinboldt, Iterative solution of nonlinear equations in several variables. SIAM, 2000.

[55] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[56] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[57] B. Wu, C. Xu, X. Dai, A. Wan, P. Zhang, Z. Yan, M. Tomizuka, J. Gonzalez, K. Keutzer, and P. Vajda, “Visual transformers: Tokenbased image representation and processing for computer vision,” arXiv preprint arXiv:2006.03677, 2020.

[58] B. Saleh and A. Elgammal, “Large-scale classification of fine-art paintings: Learning the right metric on the right feature,” arXiv preprint arXiv:1505.00855, 2015.

[59] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[60] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[61] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou,´ J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[62] S. Fu, N. Tamir, S. Sundaram, L. Chai, R. Zhang, T. Dekel, and P. Isola, “Dreamsim: Learning new dimensions of human visual similarity using synthetic data,” in Advances in Neural Information Processing Systems, 2023, pp. 50 742–50 768.

![](images/646c95213e2f39c608a0b8660a20d19dcaabc5e5d9efca20dacb970c602a4ffc.jpg)  
Qianlong Xiang received the B.E. degree from the School of Computer Science and Technology, Harbin Institute of Technology, Weihai, China, in 2022. He is currently pursuing the Ph.D. degree at the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China. His research has been published in top-tier conferences including CVPR. He has served as a reviewer for various conferences and journals, such as IEEE TPAMI, ACM MM, and IEEE TCSVT. His main research

interests include model compression and generative AI.

![](images/94bd0ab2dc37440a1c29b77f9b3d0efeea6c5557e80b64f6a931d9fbb08a2b04.jpg)

Miao Zhang (Member, IEEE) received the Ph.D. degree from the University of Technology Sydney (UTS), Australia. He is currently a Professor with Harbin Institute of Technology (Shenzhen), Shenzhen, China. Before that, he was an Assistant Professor at Aalborg University, Aalborg, Denmark. His primary research interests include AutoML, model compression, efficient learning, and continual learning.

![](images/3d818bcabd2abcf8f0889fabd95a47dabaacdee416ae0479165d0940243e8ba2.jpg)

Kun Wang is a Research Fellow at the National University of Singapore. He received the Ph.D. degree from the School of Software, Shandong University, Jinan, China, in 2026, and the B.E. degree from the School of Computer Science and Technology, Shandong University, Qingdao, China, in 2022. His research has been published in top-tier conferences and journals, including ACM MM, IEEE TIP, and ACM TOIS. He also serves as a reviewer for IEEE TPAMI, IEEE TIP, IEEE TKDE, and IEEE TMM. His main research

interests include information retrieval and video analysis.

![](images/5c229957dc24c21ca536cc1c24ba94bdf3658e9187182bcf86d6dd084bce1c69.jpg)

Haoyu Zhang received the M.S. degree from Shandong University, in 2023. He is currently working toward the doctor’s degree with the School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen). His research interests include egocentric vision and spatial understanding. His work has been published in several top-tier conferences and journals, including IEEE TPAMI, CVPR, NeurIPS, ICML, AAAI, and ACM MM. He has served as a Reviewer for various conferences and journals, such as CVPR, ICCV, NeurIPS, ICML, ICLR, IEEE TPAMI, IEEE TKDE, and IEEE TMM.

![](images/10dbfa460d5a77c25cdc3faab4c054c1e982afe41a4a6e90e2f008adc82bd892.jpg)

Junhui Hou (Senior Member, IEEE) is a Professor with the Department of Computer Science, City University of Hong Kong (CityUHK). His research interests include multidimensional visual computing. He received the Early Career Award and Research Fellow from the Hong Kong Research Grants Council, the Excellent Young Scientists Fund from NSFC, the IEEE TIP Best Paper Award, and the CityUHK Presidential Research Excellence Award for Junior Faculty. He is serving as a Senior Area Editor for IEEE TIP

and an Associate Editor for IEEE TVCG and TMM. He served as an Associate Editor for IEEE TIP and TCSVT.

![](images/7f33a822d71a21ddafd1713a5880729e29f59955fe4a8fe15326c67b0bcf7589.jpg)

Liqiang Nie (Senior Member, IEEE) received the B.Eng. degree from Xi’an Jiaotong University, Xi’an, China, and the Ph.D. degree from the National University of Singapore, Singapore. Currently, he is a Professor with the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China, and a Fellow of the IAPR. Dr. Nie serves as an Associate Editor for IEEE TIP, IEEE TKDE, IEEE TMM, IEEE TCSVT, ACM ToMM, and Information Sciences. Meanwhile, he is the chair of ICMR 2025,

ICME 2025, and ACM MM 2027, and a member of the ICME steering committee. His main interests are multimedia content analysis and information retrieval. He has published over 150 papers and 5 books in these fields, with more than 30,000 Google Scholar citations. He has received many awards over the past four years, including SIGMM Rising Star in 2020, MIT TR35 China 2020, SIGIR Best Student Paper in 2021, IEEE AI’s 10 to Watch in 2022, ACM MM Best Paper Award in 2022, and the National Youth Science and Technology Award in 2024.