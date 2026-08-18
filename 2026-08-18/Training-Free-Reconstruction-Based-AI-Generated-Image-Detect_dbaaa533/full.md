# Training-Free Reconstruction-Based AI-Generated Image Detectors Are Inherently Vulnerable to Adversarial Examples

Roman Demchenko , Jonas Ricker , and Asja Fischer

Ruhr University Bochum, Bochum, Germany {roman.demchenko, jonas.ricker, asja.fischer}@rub.de

Abstract. The impressive visual quality and ubiquity of AI-generated images call for reliable and robust detection methods. Reconstructionbased detectors have emerged as a promising direction for transparent and training-free identification of synthetic images. However, due to their fundamentally diferent mode of operation (compared to standard, classifier-based methods), little is known about their adversarial robustness. In this work, we propose two novel attack methods targeted at detectors that leverage autoencoder reconstruction error. We find that by constructing imperceptible adversarial examples, the distance between original and reconstruction can be artificially increased, causing fake images to be wrongly classified as real. Our evaluation including images from three state-of-the-art generators and three detectors demonstrates that detection performance is significantly decreased, even if attacked images additionally undergo real-world degradations. Critically, our adversarial examples naturally transfer across detectors, as they all share the same principle, pointing towards an inherent vulnerability of reconstruction-based detectors.

Keywords: AI-Generated Image Detection · Adversarial Robustness

## 1 Introduction

Image generation models have advanced significantly in recent years. Modern latent difusion models (LDMs) [27] are capable of producing high-quality synthetic images. The spread of online services such as Midjourney [21] and DALL-E [23] allows for simple and fast production of synthetic images without any technical knowledge. Additionally, open-source models like Stable Difusion [29] and Flux [3] can be run on local machines with consumer-grade hardware, without limitations by third-party services or regulations.

The large number of easily accessible and highly capable image-generating models raises concerns about the associated risks and threats they may bring. For example, synthetic images can be used for creating fake identities on social media platforms [25], facilitating fraud [12], or spreading fake news [30]. This creates an urgent need for reliable detection methods.

Reconstruction-based detectors [6, 7, 26] have recently gained attention as a promising approach for detecting LDM-generated images. Unlike the majority of existing detectors, they do not train a neural network-based classifier on real and generated images. Instead, they are based on the assumption that the autoencoder (AE) used by a specific LDM can reconstruct generated images more accurately than real images. Consequently, images with low reconstruction error are classified as generated (by the LDM corresponding to the AE), while images with higher error are classified as real. By combining diferent AEs in an ensemble and using the minimum error for each respective image, detectors can generalize to a range of diferent LDMs. This detection approach provides several advantages. It requires no costly training, can be easily adapted to novel generative models (if the AE is available), and is more interpretable due to not relying on a black-box classifier.

Outside controlled settings, however, detection performance may be afected by both natural and adversarial image modifications. While most detectors are commonly evaluated for robustness against common image degradations, robustness against adversarial examples has not yet been extensively studied. Prior work shows that conventional data-driven detectors are highly susceptible to both white- and black-box attacks [4,19]. Given that reconstruction-based approaches lack a dedicated classifier, the question arises whether such detectors ofer increased adversarial robustness.

In this work, we provide the first comprehensive adversarial robustness analysis of reconstruction-based AI-generated image detectors. Unlike methods incorporating trained classifiers, computing adversarial examples against training-free detection methods is not straightforward due to the absence of the target model. We therefore develop two novel attacks. Our first attack operates in latent space, causing an image’s embedding to deviate from the original. Secondly, we directly target a diferentiable distance function in pixel space. It should be noted that both attacks do not require detailed knowledge of the detection methods, but instead directly target the underlying idea of fake images yielding lower reconstruction errors. Our empirical evaluation including three detectors and three datasets demonstrates that both attacks consistently degrade detection performance with minimal impact on visual quality. We also observe that attacks are transferable across AEs and remain efective when images undergo typical degradations like compression and blurring.

## In summary, we make the following contributions:

– We discover that training-free, reconstruction-based AI-generated image detectors are inherently vulnerable to adversarial examples. This vulnerability is not limited to specific detectors, but universally applies to the principle of training-free reconstruction-based detection.

– We explore two novel attack vectors, demonstrating their efectiveness across three detectors while remaining imperceptible to human observers.

– Our evaluation confirms that both attacks are robust to the choice of AEs and common image degradations, making them a considerable threat in realistic settings.

We release our code and data at https://github.com/romandemchenkox/t rainingfree-reconstruction-based-detectors-are-vulnerable-toadversarial-examples.

## 2 Related Work

## 2.1 AI-Generated Image Detection

Since generative models are able to create realistic-looking images, distinguishing real from generated images is an active area of research. Diferent detection approaches have emerged over time. While early methods exploit high-level artifacts or semantic inconsistencies [13, 14, 18], other approaches leverage low-level generation artifacts in the pixel [2,20] or frequency domain [11,35]. The dominant detection approach, however, is to train a neural network-based classifier on real and generated images, thereby automatically learning relevant features [17, 31]. It has recently been shown that using pretrained vision foundation models such as CLIP [24] as feature extractors can improve performance and generalization to unseen generative models [15, 22]

Reconstruction-based methods [6, 7, 26] work fundamentally diferent, in particular due to not requiring any training. They are based on the intuition that a generative model “remembers” the images it generated, allowing it to reconstruct them more accurately than real images. In its pure form, classification is performed by computing the distance between the original and reconstruction, with low errors suggesting that an image is AI-generated. AEROBLADE [26] applies this idea to LDMs by reconstructing images solely using the model’s AE. HFI [7] and RDD [6] further improve detection performance and generalization by reducing biases regarding image background and additionally leveraging latent space reconstruction errors, respectively. Another class of reconstruction-based methods, like FakeInversion [5] and DIRE [32], leverages the entire difusion and denoising processes for reconstruction. The diferences between the original image and its reconstruction are then used as training data for a neural network-based classifier.

We limit our evaluation to methods that directly use reconstruction errors for detection, without relying on a trained classifier.

## 2.2 Adversarial Robustness of AI-Generated Image Detectors

Compared to the number of works proposing novel detection methods, relatively little research is conducted on their adversarial robustness. The pioneering work by Carlini and Farid [4] demonstrates that neural network-based classifiers become inefective when adding imperceptible perturbations. While attacks are most efective in a white-box setting, the AUROC can be reduced from 0.95 to 0.22 in a black-box setting using a surrogate classifier. The general vulnerability is confirmed by subsequent works and extended to more recent detectors [9, 19]. However, for training-free reconstruction-based detectors, a detailed and targeted robustness analysis is still lacking. While AEROBLADE [26] was previously evaluated under an attack derived from a surrogate CNN model [10], it remains unclear whether the core rationale of reconstruction-based detection— the reconstruction error—can be attacked.

## 3 Preliminaries

In the following, we briefly recap the architecture of LDMs, introduce the three reconstruction-based detection methods we evaluate, and explain the concept of the chosen gradient-based method for computing adversarial examples.

## 3.1 Latent Difusion Models

Latent difusion models (LDMs) [27] are generative models capable of producing synthetic images conditioned on text prompts and, in some cases, additional inputs, such as images. LDMs typically consist of two main components: the U-Net [28] modeling the difusion process and an autoencoder (AE). In contrast to DMs, which operate in pixel space, an image processed by an LDM is first encoded into a latent representation by the AE. The U-Net is then trained on latents of real images, in which Gaussian noise is gradually applied to the input latent during the forward difusion process, allowing the model to learn to remove the added noise and restore the original latent. After training, the model is capable of producing fully synthetic images from Gaussian noise, guided by a text prompt. This process is called denoising. After denoising, the resulting latent is decoded back to the pixel space, forming the final image.

## 3.2 Detection Methods

Diferent reconstruction-based detection methods for LDM-generated images have been proposed [6, 7, 26]. All of them have in common that they allow identifying generated images stemming from a certain model based on an analysis of their AE reconstruction. In the following, we briefly introduce three state-ofthe-art, training-free approaches in more detail.

AEROBLADE [26] reconstructs the input image using the AE of an LDM and compares it to the original. The LPIPS [34] metric, which measures perceptual similarity of two images, is used to capture the diferences between the image and its reconstruction. The LPIPS distance between them is observed to be higher for real images than for fake images generated by the same LDM. This is because synthetic images from the LDM are generated from latents that lie within the AE’s learned latent distribution. As a result, they can be reconstructed accurately by the same AE. However, images that were not generated by the same

LDM are not guaranteed to be well represented by this latent distribution. Consequently, reconstructing them may introduce distortions, which are highlighted by higher LPIPS values. The reconstruction error computed by AEROBLADE can be formulated as follows:

$$
\Delta _ { \mathrm { A E R O B L A D E } } = d ( x , \mathrm { A E } _ { i } ( x ) ) \enspace ,\tag{1}
$$

where $\mathrm { A E } _ { i } ( \cdot ) \ : = \ : { \mathcal D } _ { i } ( { \mathcal E } _ { i } ( \cdot ) )$ denotes the AE consisting of an encoder $\mathcal { E } _ { i }$ and a decoder $\mathcal { D } _ { i }$ , and d is a distance metric, which is commonly chosen to be LPIPS.

HFI [7] operates similarly but computes the reconstruction error in a more robust way. The authors observe that high-frequency components of an image have the greatest influence on its reconstruction error. To isolate the contribution of high-frequency components to the final prediction, the reconstruction error is computed separately for the original image and a version where high-frequency components are removed. This is done with a low-pass filter function, such as Gaussian blur. The LPIPS distance between the low-pass filtered image and its reconstruction is subtracted from the distance of the original. The reconstruction error of HFI is defined as:

$$
\begin{array} { r } { \varDelta _ { \mathrm { H F I } } = d ( x , \mathrm { A E } _ { i } ( x ) ) - d ( \mathcal { F } ( x ) , \mathrm { A E } _ { i } ( \mathcal { F } ( x ) ) ) } \end{array}\tag{2}
$$

RDD [6] is the most recent detector and an extension of HFI. It combines debiased reconstruction errors in both pixel and latent space. The reconstruction error in the pixel space is measured by the $S _ { \mathrm { i m a g e } }$ score. This score is obtained by combining two debiasing strategies, based on the ideas applied in HFI. First, the reconstruction error of an image is compared to that of its rotated version. The intuition is that rotating real images does not significantly change their reconstruction error, while rotating fake images makes them more dificult to reconstruct. Second, a low-pass filter is applied to suppress the contribution of simple backgrounds, isolating only high-frequency components of the images. The reconstruction error in the latent space is represented by the $S _ { \mathrm { l a t e n t } }$ score. It measures the accuracy of the reverse difusion process of the U-Net of the chosen LDM on the input latent and compares it to that of a rotated latent. The intuition behind this approach is that rotating latents of images that are unknown to the difusion model cannot be denoised as accurately as latents of real images or images generated by that particular LDM.

The final score of RDD is the combination of $S _ { \mathrm { i m a g e } }$ and $S _ { \mathrm { l a t e n t } }$

$$
\varDelta _ { \mathrm { R D D } } = S _ { \mathrm { i m a g e } } ( x ) \times S _ { \mathrm { l a t e n t } } ( \mathcal { E } _ { i } ( x ) ) ^ { 2 } \ .\tag{3}
$$

All three detection methods can be configured with multiple AEs by taking the minimum reconstruction error across all AEs as the final detection score.

## 3.3 Adversarial Examples

Suppose a neural network $f _ { \theta }$ , given an input $x ,$ produces a prediction ${ \hat { y } } = f _ { \theta } ( x )$ which matches the ground truth label y. The core idea of an adversarial attack

is to find a sample $x ^ { \prime }$ that is reasonably similar to $x ,$ so that the target model misclassifies it. This condition can be defined as follows:

$$
f _ { \theta } ( x ^ { \prime } ) \neq y , \mathrm { s . t . } d ( x , x ^ { \prime } ) \leq \epsilon .\tag{4}
$$

In particular, this can be achieved by means of gradient-based attacks, such as projected gradient descent (PGD) [16]. PGD generally targets a neural network and computes gradients of the chosen loss with respect to the model input. Then it applies perturbations to the input, so that the chosen loss is maximized and the changes remain relatively imperceptible. The strength of the perturbation is controlled by the chosen perturbation budget ϵ.

## 4 Threat Model

Due to the distinct nature of training-free reconstruction-based detection methods, our threat model difers from the “classical” adversarial setting. Typically, the attacker’s goal is to create an adversarial example that causes a classifier to wrongly predict a label that difers from the ground-truth label. In a white-box setting, the attacker has access to the architecture and weights of the classifier. In contrast, we refer to a black-box setting if the attacker can only query the classifier or uses a surrogate model to craft adversarial perturbations.

However, training-free reconstruction-based AI-generated image detectors do not utilize a dedicated classifier. Instead, the autoencoder of a generative model (or multiple ones) essentially is the classifier. Classical definitions of white- and black-box attacks are therefore not suitable, since the attacker has inherent knowledge about the defender’s abilities (except for the specific set of AEs in the ensemble). In the defender’s best-case scenario, the AE that generated the image is part of the set of AEs used for detection, as it should yield the lowest reconstruction error. Since the attacker knows which AE generated the image (and has access to it), they naturally have white-box access and can craft an ideal adversarial perturbation. In case the defender’s set of AEs does not include the AE that created the attacker’s image, there are two outcomes in which the attacker succeeds. Either the image is not classified as fake (even without adversarial perturbations), since none of the candidate AEs yields a low enough reconstruction error. Alternatively, if the “next-best” AE in the ensemble can produce an accurate reconstruction, the adversarial example may generalize (as we show in Section 6.3) and still cause a misclassification.

## 5 Attacks on Reconstruction-Based Detectors

Gradient-based attacks, like PGD, generally require a diferentiable model to be able to compute a gradient with respect to the given input. Training-free detection methods do not incorporate a trained classifier dedicated to distinguishing between fake and real images. As a consequence, gradient-based attacks cannot be applied in the same manner as for conventional classifier-based methods.

However, their reconstruction process still involves diferentiable components. Training-free reconstruction-based detectors rely on the assumption that images produced by a particular LDM can be reconstructed more easily using the matching AE. Since an AE is diferentiable, it is a suitable target for a gradient-based attack.

The core idea of the proposed attack method is to disrupt the reconstruction process, making detection methods unable to distinguish between perturbed fake images and real images. This can be accomplished by adding an adversarial perturbation to the input image, causing a higher reconstruction error.

## 5.1 Latent Space Attack

One way to achieve this goal is to modify the target image in a way that maximizes the distance between the latent representations of the attacked and original image within the latent space of the respective AE. This corresponds to maximizing the following loss function w.r.t. η,

$$
L _ { \mathrm { l a t e n t } } ( x , \eta ) = \| \mathcal { E } ( x ) - \mathcal { E } ( x + \eta ) \| _ { 2 } \ ,\tag{5}
$$

where $\mathcal { E } ( \cdot )$ represents the encoder of the given $\operatorname { A E } , \ \| \cdot \| _ { 2 }$ the Euclidean norm, and $\eta$ is from a predefined ϵ-region to ensure the adversarial image is reasonably similar to the original image. Intuitively, the largest change in latent space can be achieved by leaving the data manifold, which makes the image harder to reconstruct by the decoder, leading to an increase in reconstruction error.

## 5.2 Pixel Space Attack

An alternative approach is to increase the distance between the image and its reconstruction directly. When measuring the distance in terms $\operatorname { o f } ,$ for instance, LPIPS, this is equivalent to maximizing $L _ { \mathrm { p i x e l } } ( x , \eta )$ , s.t. $\| \eta \| _ { \infty } < \epsilon .$ with

$$
L _ { \mathrm { p i x e l } } ( x , \eta ) = \mathrm { L P I P S } ( x + \eta , \mathcal { D } \circ \mathcal { E } ( x + \eta ) ) ,\tag{6}
$$

where $\mathcal { E } ( \cdot )$ and $\mathcal { D } ( \cdot )$ denote the encoder and decoder of the AE, respectively. Since both LPIPS and the AE are diferentiable, the gradients of the loss function can be computed via backpropagation. Note that LPIPS could be replaced by any other choice of distance measure, as long as it is diferentiable.

## 6 Experiments

We first describe how we conducted our experiments, after which we evaluate our proposed attacks. Lastly, we analyze the quality of our adversarial examples and perform diferent ablations.

## 6.1 Experimental Setup

We evaluate our attacks on a subset of the Synthbuster dataset [1], which contains synthetic images generated by several LDMs. The real images are taken from the RAISE-1k dataset [8]. This dataset contains high-quality photos of diferent scenes, based on which all images from the Synthbuster dataset were generated to ensure semantic alignment. Two additional sets of images were generated using Flux.1-schnell and Stable Difusion (SD) 3.5-medium, based on the prompts provided in the Synthbuster dataset and using default parameters for each LDM. To match the employed AEs, we only include images generated by Midjourney-v5, Flux.1-schnell, and SD3.5 in the experiments. According to prior work [7], using the SD2 AE tends to produce more accurate predictions in reconstruction-based detection methods on Midjourney than on the actual SD2 dataset. In total, we use 300 synthetic and 100 real images. All images are preprocessed to a fixed resolution of $5 1 2 \times 5 1 2$ pixels using center cropping and resizing. This provides a reasonable compromise between image quality, expected input resolution of the employed models, and computational cost.

The attack targets reconstruction-based training-free detection methods. Therefore, three state-of-the-art detectors are selected for evaluation. AEROBLADE and HFI are initialized with the distance function $\mathrm { L P I P S _ { 2 } }$ (referring to its second layer), as it has been shown to be the most efective for both methods [7]. For HFI, a Gaussian blur with a $3 \times 3$ kernel and a standard deviation of $\sigma = 0 . 8$ is used as the low-pass filter. RDD applies the same Gaussian blur settings and $\mathrm { ~ a ~ } 9 0 ^ { \circ }$ rotation for the respective transformations. All parameters are set in accordance with the original publication, namely $\lambda _ { R } = 0 . 5$ and $\lambda _ { L } = 1$ . For better comparability between detectors, $\mathrm { L P I P S _ { 2 } }$ is also used as the distance function. Since only LDM-generated images are used in the experiments, the $S _ { \mathrm { i m a g e } }$ -score of RDD is used as its final prediction. All detectors are evaluated using an ensemble of AEs corresponding to SD2, SD3.5, and Flux. Following prior work, the minimum reconstruction error across the selected AEs is used as the final prediction.

Unlike trained classifiers, training-free methods do not automatically provide a decision threshold. Their score distributions vary across diferent datasets and often overlap between real and fake images, making threshold-dependent metrics unreliable. The performance is therefore measured using the area under the ROC curve (AUROC), which evaluates the ability of a detector to distinguish between real and fake images across all decision thresholds. All variants of the proposed attack are run using a fixed number of iterations $n = 2 0$ and a step size $\textstyle \alpha = { \frac { 2 \epsilon } { n } }$ where ϵ is the $L _ { \infty }$ perturbation budget. The attack is evaluated with ϵ ranging from $1 / 2 5 5$ to $1 0 / 2 5 5$ . Furthermore, a random start within the $L _ { \infty } { \mathrm { - b a l l } }$ of radius ϵ around the original image is employed.

## 6.2 Evaluating Attack Strength

The detection performance is first measured on clean images $( \mathrm { i . e . , } \ \epsilon = 0 )$ and then under the proposed attack at diferent ϵ-budgets. To mimic the practically relevant setting, only fake images are perturbed. The attack is computed independently of the target detector, meaning the same images are evaluated by all detectors. The AUROC under the attack targeting the matching AEs on the three selected datasets is presented in Figure 1. The impact of each attack variant (denoted as $L _ { \mathrm { l a t e n t } }$ and $L _ { \mathrm { p i x e l } } )$ is shown on a grid of nine plots, where each plot corresponds to a detector-dataset pair.

![](images/fc75ab0b982ea1653d3cf043a56363acbe8dd86e5820c3847beeff83be4a28b7.jpg)  
(a) L<sub>latent</sub>

![](images/eacd96153812ef0693a1a3a95aa827a5a77e55effd347966dee019b3520b0888.jpg)  
(b) L<sub>pixel</sub>  
Fig. 1: AUROC of the detectors under the proposed attack on the three selected datasets.

As expected, an AE performs better on the matching dataset than on nonmatching datasets. For example, the SD2 AE detects fake images from Midjourney with high confidence, whereas the Flux AE is only reliable for images generated by Flux. The plots show that this trend is consistent across all AEs and detectors. Notably, the ensemble performance is not always as high as the performance of the best AE. This efect could be attributed to Flux and SD3.5 generally producing lower reconstruction errors than older AEs [3], making an ensemble approach more fragile.

The AUROC of all three detectors decreases substantially with increasing ϵ for all AEs on all datasets, indicating the efectiveness of the attack. The performance of AEROBLADE decreases monotonically for both attack variants. HFI behaves similarly under $L _ { \mathrm { p i x e l } }$ , but its performance with the SD2 AE stabilizes around 0.4 after $\epsilon = 4 / 2 5 5$ under $L _ { \mathrm { l a t e n t } }$ , slightly drifting upwards as ϵ increases. A similar pattern is observed for RDD as well under $L _ { \mathrm { l a t e n t } }$ . We observe that $L _ { \mathrm { p i x e l } }$ is significantly more efective than $L _ { \mathrm { l a t e n t } }$ . Additionally, for each dataset, an attack based on the matching AE is most efective against the respective AE used for detection. Generally, detection becomes unreliable $( \mathrm { A U R O C } \le 0 . 5 )$ at $\epsilon = 3 / 2 5 5$ or $\epsilon = 4 / 2 5 5$ for the three detectors using any of the AEs for estimating the adversarial example on all datasets. Among the considered detection methods, RDD appears slightly more robust, retaining the highest ensemble performance.

Based on our results, we conclude that the attack is highly efective against the selected detectors, with HFI being more robust than AEROBLADE, and RDD demonstrating the highest robustness among them.

![](images/bac51f7ae41299fd380654a49f4cf89cd93e68a6fd172d7c7b3872c96b6046bd.jpg)  
(a) L<sub>latent</sub>

![](images/c4a535b740f5acc636926f1815feb89d8ee507a3e2fbce91850c4c77480d7a36.jpg)  
(b) L<sub>pixel</sub>  
Fig. 2: AUROC of the detectors under the proposed attack targeting the selected AEs.

## 6.3 Transferability

To better understand the real-world threat of the attack (i.e., the threat in a scenario in which the attacker assumes a reconstruction-based detection method, but does not know which AE is used), it is important to assess the transferability of the attack between diferent AEs. For that, the attack is run using the AEs of SD2, SD3.5, and Flux and evaluated against the detectors initialized with all three of them. Performance is averaged across the selected datasets.

As observable in the results shown in Figure 2, the attack remains efective against all three detectors, even when targeting only a single AE. Interestingly, detectors using SD2 only tend to have the highest performance compared to other AEs, no matter which AE is targeted by the attacker. When SD3.5 or Flux are targeted, the respective AE consistently demonstrates the lowest performance.

Overall, the attack using $L _ { \mathrm { p i x e l } }$ is more efective than $L _ { \mathrm { l a t e n t } }$ in this setting.   
Furthermore, the attack exhibits strong transferability across diferent AEs.

## 6.4 Quality of Adversarial Examples

Clearly, larger perturbations lead to diminishing detector performance. However, too large perturbations may attract attention and expose the fact that the image has been tampered with. Therefore, it is important to consider the imperceptibility of the computed perturbations. To assess it, the adversarial perturbations are compared to regular transformations. Imperceptibility is evaluated using LPIPS and structural similarity index measure (SSIM) [33] by measuring the similarity between the original and perturbed images. Figure 3 shows the mean distance values of both variants of the attack targeting the SD2 AE and tested on the Midjourney dataset.

The attack generally yields a better AUROC-LPIPS tradeof than natural corruptions such as JPEG, Gaussian blur, and noise. JPEG demonstrates lower LPIPS values despite introducing significant artifacts. $L _ { \mathrm { l a t e n t } }$ drops below AU-ROC=0.5 for ϵ = 3 while staying at an LPIPS value of about 0.24. $L _ { \mathrm { p i x e l } }$ shows the same AUROC but a far larger LPIPS value of about 0.38 at the same value of ϵ. SSIM, on the other hand, is relatively high for both attack variants, staying around 0.96 at $\epsilon = 3$

![](images/b2d31a073e3245488f21a5223688ba581dbd8334e402857bb035923dfbe63b79.jpg)  
(a) LPIPS

![](images/940ed12e9a66e5cb8f2bae6898a27432fc0be3ac4662aece5494f6998e7cf7df.jpg)  
(b) SSIM  
Fig. 3: AUROC of diferent attack variants and distortions in relation to the image quality.

Overall, it can be said that the proposed attack remains reasonably imperceptible with $L _ { \mathrm { l a t e n t } }$ and becomes more apparent for $L _ { \mathrm { p i x e l } }$

## 6.5 Adversarial Examples in Real-World Settings

While adversarially forged images are efective when evaluated by the target detection methods directly, this may not be the case in a practical setting. Images are compressed, resized, or otherwise distorted when uploaded to the Internet. Studying how such transformations may afect the impact of the adversarial perturbations is essential to assessing their actual threat in practice.

The mean performance of the three selected detectors under the proposed attacks with $\epsilon = 2 , 4 , 6 , 8$ in the presence of JPEG compression and Gaussian blur is summarized in Figure 4. The attack using $L _ { \mathrm { l a t e n t } }$ is always represented by a solid line, whereas $L _ { \mathrm { p i x e l } }$ is represented by a dashed line.

As expected, applying distortions decreases the baseline performance of the detectors. Interestingly, the performance under all the attacks slightly improves when the final image is distorted only by a small amount. In particular, applying JPEG with $q = 9 0$ gives a small boost in performance due to the perturbations being distorted by the compression, reducing their impact. However, this protective efect is eventually outweighed as the compression level increases. A similar efect is observed with blur. In general, a distorted adversarially perturbed fake image is still misclassified when compared to a distorted real image.

## 6.6 Ablations

The following ablations investigate the influence of the distance function used for the $L _ { \mathrm { p i x e l } }$ attack and the $S _ { \mathrm { l a t e n t } }$ component of RDD.

![](images/bab85bed1f820559f5b53f429272e96a2fa7db48b0feb56f4bd08c1a22967b09.jpg)  
(a) JPEG

![](images/3e69bc8c08730a759a84ca458aca9af5771f982813418150868bbb230ece5d28.jpg)  
(b) Blur

Fig. 4: The mean performance of all three detectors measured in AUROC under the proposed attack with various ϵ-budgets under JPEG and Blur.  
![](images/5ba00904c2f49d2f60736e3e55829102df6513ea0f1d8be295a08a7c050b1698.jpg)  
Fig. 5: AUROC of the detectors under the proposed attack on the three selected datasets using MSE distance.

MSE as Pixel Distance The proposed $L _ { \mathrm { p i x e l } }$ attack uses LPIPS in its attack objective to match the distance used by the evaluated detectors. To investigate the versatility of the attack, we replace it with the mean squared error (MSE) as follows:

$$
L _ { \mathrm { M S E } } ( x , \eta ) = \mathrm { M S E } ( x + \eta , \mathcal { D } \circ \mathcal { E } ( x + \eta ) ) \ ,\tag{7}
$$

As shown in Figure $5 ,$ the adversarial efect is considerably weaker than with LPIPS. However, the attack remains efective for AEROBLADE and for all configurations of HFI and RDD except when only SD2 is used for detection. This demonstrates that the attack is not limited by the choice of the specific distance function used by the target detector.

Full RDD Although LDM-generated images are generally detected using the $S _ { \mathrm { i m a g e } }$ score of RDD, it can be of interest to observe if the $S _ { \mathrm { l a t e n t } }$ score can improve detection of perturbed images. The performance of RDD is evaluated on 100 images from the Midjourney dataset, using the SD2 AE and U-Net for detection and the SD2 AE as the attack target.

![](images/f97808a3f65b5de39c574f24ee8696f96d047372575fb427cf06d5944e16fc0d.jpg)  
(a) $L _ { \mathrm { l a t e n t } }$

![](images/6556aab285ef68c28bf8d820d1a737f65324287b9cccb8520c04c1d0f670d2b5.jpg)  
(b) $L _ { \mathrm { p i x e l } }$  
Fig. 6: AUROC of the detectors using the SD2 AE under the proposed attack on the Midjourney dataset.

As shown in Figure 6, the $S _ { \mathrm { l a t e n t } }$ score actually improves as perturbation strength increases. At low perturbation strengths, its AUROC is close to $0 . 5 ,$ as both real and Midjourney images can be reconstructed easily by the SD2 difusion model. Then, as the perturbations move the images away from the learned manifold of SD2, the reconstruction error increases, and $S _ { \mathrm { l a t e n t } }$ becomes a more reliable detection method for such perturbed images. The contribution of $S _ { \mathrm { l a t e n t } }$ improves the overall RDD score, reducing the strength of the attack. Interestingly, this efect is significant for $L _ { \mathrm { l a t e n t } }$ but is considerably weaker for $L _ { \mathrm { p i x e l } }$

## 7 Discussion and Limitations

Our work reveals an inherent vulnerability of training-free reconstruction-based detectors. Across diferent datasets, AEs, and detectors, we observe that adversarial perturbations significantly decrease detection performance. Moreover, our results confirm our assumptions described in Section 4. Regardless of whether the AE used to craft the adversarial example is part of the defender’s ensemble, the attack remains efective. In addition, attacks are not limited to a specific detector, but apply to all detectors leveraging AE-based reconstruction errors. Together, this means that attacks will likely be efective against future reconstruction-based detectors, as long as they employ the same core methodology.

Despite carefully choosing our experimental conditions, our work has several limitations. First and foremost, our attacks are directly targeted at reconstructionbased detectors and may not generalize to classifier-based detectors. Moreover, although we observe transferability, some settings we consider assume that the attacker has access to the AE that generated their image. While this is naturally given for open-source models, attacks using proprietary generative models may be less efective. Due to the computational complexity and the large number of possible combinations between datasets, AEs, and detectors, the number of images we analyzed per experiment is relatively small.

While our work uncovers a fundamental vulnerability of reconstruction-based detectors, there are several aspects left for future work. The most pressing one is the development of efective defenses. Another interesting research direction may be creating adversarial examples that are efective against both classifierand reconstruction-based detectors at the same time.

## 8 Conclusion

In contrast to conventional data-driven detection approaches, reconstructionbased detectors have several desirable properties. They do not require any training, can be easily extended to new generative models, and ofer interpretable detection and attribution. These properties are a consequence of leveraging the model’s own AE for calculating reconstruction errors. In this work, we demonstrate a fundamental downside of this approach: By adding imperceptible adversarial perturbations, we can intentionally increase the diference between original and reconstruction, rendering the detector inefective. Moreover, the fact that the AEs leveraged for detection are usually openly available benefits the attacker, mimicking a natural white-box setting. We call upon the research community to critically revisit the idea of reconstruction-based AI-generated image detection and consider its adversarial robustness as an important direction for future work.

## References

1. Bammey, Q.: Synthbuster: Towards detection of difusion model generated images. IEEE Open Journal of Signal Processing (2023)

2. Barni, M., Kallas, K., Nowroozi, E., Tondi, B.: CNN detection of GAN-generated face images based on cross-band co-occurrences analysis. In: IEEE International Workshop on Information Forensics and Security (WIFS) (2020)

3. Black Forest Labs, Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., Müller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., Smith, L.: FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space. arXiv Preprint (2025)

4. Carlini, N., Farid, H.: Evading deepfake-image detectors with white- and black-box attacks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2020)

5. Cazenavette, G., Sud, A., Leung, T., Usman, B.: FakeInversion: Learning to detect images from unseen text-to-image models by inverting stable difusion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

6. Choi, S., Lee, H., Lee, J., Kim, R., Choi, S.J., Lee, M.: A debiased reconstructionbased framework for training-free detection of ai-generated images. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026)

7. Choi, S., Lee, H., Lee, J., Kim, S., Choi, S.J., Lee, M.: HFI: A unified framework for training-free detection and implicit watermarking of latent difusion model generated images. arXiv Preprint (2026)

8. Dang-Nguyen, D.T., Pasquini, C., Conotter, V., Boato, G.: RAISE: A raw images dataset for digital image forensics. In: ACM Multimedia Systems Conference (MMSys) (2015)

9. De Rosa, V., Guillaro, F., Poggi, G., Cozzolino, D., Verdoliva, L.: Exploring the adversarial robustness of CLIP for AI-generated image detection. In: IEEE International Workshop on Information Forensics and Security (WIFS) (2024)

10. Diao, Y., Zhai, N., Miao, C., Yu, Z., Wei, X., Yang, X., Wang, M.: Vulnerabilities in AI-generated image detection: The challenge of adversarial attacks. arXiv Preprint (2025)

11. Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., Holz, T.: Leveraging frequency analysis for deep fake image recognition. In: International Conference on Machine Learning (ICML) (2020)

12. Gelbart, H.: Scammers profit from turkey-syria earthquake. https://www.bbc.co m/news/world-europe-64599553 (2023), BBC News. Accessed: 2025-11-30

13. Guo, H., Hu, S., Wang, X., Chang, M.C., Lyu, S.: Eyes tell all: Irregular pupil shapes reveal GAN-generated faces. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2022)

14. Hu, S., Li, Y., Lyu, S.: Exposing GAN-Generated faces using inconsistent corneal specular highlights. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2021)

15. Koutlis, C., Papadopoulos, S.: Leveraging representations from intermediate encoder-blocks for synthetic image detection. In: European Conference on Computer Vision (ECCV) (2025)

16. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.: Towards deep learning models resistant to adversarial attacks. arXiv Preprint (2019)

17. Mandelli, S., Bonettini, N., Bestagini, P., Tubaro, S.: Detecting GAN-generated images by orthogonal training of multiple CNNs. In: IEEE International Conference on Image Processing (ICIP) (2022)

18. Matern, F., Riess, C., Stamminger, M.: Exploiting visual artifacts to expose deepfakes and face manipulations. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2019)

19. Mavali, S., Ricker, J., Pape, D., Fischer, A., Schönherr, L.: Adversarial Robustness of AI-Generated Image Detectors in the Real World . In: Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA) (2026)

20. McCloskey, S., Albright, M.: Detecting GAN-generated imagery using saturation cues. In: IEEE International Conference on Image Processing (ICIP) (2019)

21. MidJourney Inc: Midjourney: AI image generation model. https://www.midjourn ey.com (2022), Accessed: 2025-11-30

22. Ojha, U., Li, Y., Lee, Y.J.: Towards universal fake image detectors that generalize across generative models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)

23. OpenAI: Dall·e: Creating images from text. https://openai.com/index/dall-e/ (2021), Accessed: 2026-06-08

24. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML) (2021)

25. Ricker, J., Assenmacher, D., Holz, T., Fischer, A., Quiring, E.: AI-generated faces in the real world: A large-scale case study of Twitter profile images. In: International Symposium on Research in Attacks, Intrusions and Defenses (RAID) (2024)

26. Ricker, J., Lukovnikov, D., Fischer, A.: Aeroblade: Training-free detection of latent difusion images using autoencoder reconstruction error. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

27. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

28. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI) (2015)

29. Stability AI: Stability AI Image Models, https://stability.ai/stable-image, Accessed: 2025-11-30

30. Vincent, J.: The swagged-out pope is an AI fake — and an early glimpse of a new reality. https://www.theverge.com/2023/3/27/23657927/ai-pope-imagefake-midjourney-computer-generated-aesthetic (2023), the Verge. Accessed: 2025-11-30

31. Wang, S.Y., Wang, O., Zhang, R., Owens, A., Efros, A.A.: CNN-generated images are surprisingly easy to spot... for now. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020)

32. Wang, Z., Bao, J., Zhou, W., Wang, W., Hu, H., Chen, H., Li, H.: DIRE for difusion-generated image detection. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023)

33. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing (2004)

34. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018)

35. Zhang, X., Karaman, S., Chang, S.F.: Detecting and simulating artifacts in GAN fake images. In: IEEE International Workshop on Information Forensics and Security (WIFS) (2019)