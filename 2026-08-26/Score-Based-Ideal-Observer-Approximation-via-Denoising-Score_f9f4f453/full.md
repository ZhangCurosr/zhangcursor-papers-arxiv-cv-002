# Score-Based Ideal Observer Approximation via Denoising Score Matching for Signal-Known-Exactly Detection Tasks

Weimin Zhou<sup>a,</sup> <sup>b</sup>

<sup>a</sup>Wyant College of Optical Sciences, University of Arizona, AZ 85721, USA

<sup>b</sup>Department of Radiology and Imaging Sciences, University of Arizona College of Medicine – Tucson, AZ 85721, USA

## ABSTRACT

The Bayesian Ideal Observer (IO) establishes the theoretical upper bound on task performance for binary detection tasks. However, analytical computation of the IO test statistic is generally intractable. Numerical approaches based on Markov-chain Monte Carlo (MCMC) methods, including their recent deep generative model-based extensions, typically require extensive posterior sampling for each test image. Supervised learning has also been investigated to approximate the IO performance. However, such methods are typically trained for a specific detection task and signal and may require retraining when the task or signal changes. The score function, defined as the gradient of the log probability density, encodes the local geometry of the data distribution and is a fundamental quantity in modern score-based generative modeling. This work reformulates the IO test statistic in terms of the score function and introduces a score-based ideal observer (SIO). The proposed SIO uses a denoising convolutional neural network trained exclusively on signal-absent images to estimate the signal-absent score function. Once trained, the resulting score model can be used to approximate the IO test statistic for detection tasks involving arbitrary additive signals, without per-image posterior sampling or signal-specific retraining. Numerical studies consider a signal-known-exactly (SKE) detection task with a stochastic lumpy-background model. The results demonstrate that the proposed SIO can closely approximate the IO performance.

Keywords: Bayesian Ideal Observer, signal detection, score function, denoising score matching

## 1. INTRODUCTION

The Bayesian ideal observer (IO) provides the optimal decision strategy for binary signal detection tasks and establishes a theoretical upper bound on task performance for assessing and optimizing imaging systems.<sup>1</sup> However, analytical computation of the IO test statistic is generally intractable for imaging problems involving complex stochastic backgrounds because the underlying probability density functions are typically unknown. Existing numerical approaches based on Markov-chain Monte Carlo (MCMC)<sup>2–4</sup> can approximate the IO test statistic but typically require extensive posterior sampling for each test image. Supervised learning methods can provide eficient alternatives,<sup>5–7</sup> but are typically trained for a specific detection task and signal and may require retraining when the task or signal changes.

Recent advances in score-based generative modeling provide an alternative means of characterizing complex data distributions through the score function, defined as the gradient of the log probability density.<sup>8,</sup> <sup>9</sup> The score function can be learned directly from data using denoising score matching without requiring explicit knowledge of the underlying probability density.<sup>10</sup>

In this work, we propose a score-based formulation of the IO test statistic and develop a score-based idea observer (SIO) for approximating the IO. The proposed formulation expresses the IO test statistic using the signal-absent score function integrated along a path defined by the known signal, followed by a nonprewhitening matched filter (NPWMF) operation. A denoising convolutional neural network is trained exclusively on signalabsent images to estimate the score function. Once trained, the same score model can be used to approximate the IO for detection tasks involving arbitrary additive signals without task-specific supervised retraining.

## 2. METHODS

## 2.1 Score and Denoising Score Matching

Recent advances in generative modeling have extensively utilized the score function, defined as the gradient of the log probability density with respect to the image data. The score function characterizes how the probability density of an image changes with respect to local perturbations in the image space and therefore serves as a powerful representation of underlying image statistics. For image data x sampled from the probability density function $\mathrm { p r } ( \mathbf { x } )$ , the score function is defined as:

$$
\psi _ { \mathbf { x } } ( \mathbf { x } ) = \nabla _ { \mathbf { x } } \log \mathrm { p r } ( \mathbf { x } ) .\tag{1}
$$

To approximate the score function $\psi ( \mathbf { x } )$ , a network $\phi _ { \theta } : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ parameterized by θ can be trained via score matching. However, directly optimizing the original score-matching objective requires computing the trace of the Jacobian of the network, which is not scalable to large networks and high-dimensional data.<sup>8</sup> Denoising score matching (DSM) avoids this dificulty by introducing a known perturbation distribution $q _ { \sigma } ( \tilde { \mathbf { x } } | \mathbf { x } )$ to produce the perturbed data x˜ from x, and the network parameters are optimized by minimizing:<sup>8</sup>

$$
\begin{array} { r } { \mathbb { E } _ { q _ { \sigma } ( \tilde { \mathbf { x } } | \mathbf { x } ) \mathrm { p r } ( \mathbf { x } ) } \Big [ \big \| \phi _ { \theta } ( \tilde { \mathbf { x } } ) - \nabla _ { \tilde { \mathbf { x } } } \log q _ { \sigma } ( \tilde { \mathbf { x } } \mid \mathbf { x } ) \big \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{2}
$$

A common choice for the perturbation distribution $q _ { \sigma } ( \tilde { \mathbf { x } } | \mathbf { x } )$ is an isotropic Gaussian distribution with standard deviation σ. The corresponding conditional score is:

$$
\nabla _ { \tilde { \mathbf { x } } } \log q _ { \sigma } ( \tilde { \mathbf { x } } \mid \mathbf { x } ) = - \frac { \tilde { \mathbf { x } } - \mathbf { x } } { \sigma ^ { 2 } } .\tag{3}
$$

The denoising score matching objective in $\operatorname { E q . } \ ( 2 )$ becomes a least-squares loss for predicting the scaled noise.

Below, we first reformulate the IO test statistic for signal-known-exactly (SKE) detection tasks using the score function and then present an IO approximation approach using denoising score matching.

## 2.2 IO Approximation via Denoising Score Matching

Consider an SKE signal detection task in which an observer classifies imaging data g as arising from either the signal-absent hypothesis $\left( H _ { 0 } \right)$ or the signal-present hypothesis $\left( H _ { 1 } \right)$ :

$$
\begin{array} { r l } & { H _ { 0 } : \mathbf { g } = \mathbf { H } \mathbf { f } _ { b } + \mathbf { n } \equiv \mathbf { b } + \mathbf { n } \equiv \mathbf { f } _ { 0 } + \mathbf { n } , } \\ & { H _ { 1 } : \mathbf { g } = \mathbf { H } ( \mathbf { f } _ { b } + \mathbf { f } _ { s } ) + \mathbf { n } \equiv \mathbf { b } + \mathbf { s } + \mathbf { n } \equiv \mathbf { f } _ { 1 } + \mathbf { n } . } \end{array}\tag{4}
$$

Assume that the measurement g is continuously valued, and denote its probability density under $H _ { j }$ by $\mathrm { p r } ( \mathbf { g } | H _ { j } )$ $( j ~ = ~ 0 , 1 )$ . For a SKE detection task in which the signal s is deterministic and additive, the signal-present distribution is a translated version of the signal-absent distribution:

$$
\mathrm { p r } ( \mathbf { g } | H _ { 1 } ) = \mathrm { p r } _ { \mathbf { f } _ { 1 } + \mathbf { n } } ( \mathbf { g } ) = \mathrm { p r } _ { \mathbf { f } _ { 0 } + \mathbf { s } + \mathbf { n } } ( \mathbf { g } ) = \mathrm { p r } _ { \mathbf { f } _ { 0 } + \mathbf { n } } ( \mathbf { g } - \mathbf { s } ) = \mathrm { p r } ( \mathbf { g } - \mathbf { s } | H _ { 0 } )\tag{5}
$$

The IO test statistic, defined by the log-likelihood ratio, can then be expressed as:

$$
\lambda _ { \mathrm { I O } } ( \mathbf { g } ) = \log \left[ { \frac { \mathrm { p r } ( \mathbf { g } | H _ { 1 } ) } { \mathrm { p r } ( \mathbf { g } | H _ { 0 } ) } } \right] = \log \left[ { \frac { \mathrm { p r } ( \mathbf { g } - \mathbf { s } | H _ { 0 } ) } { \mathrm { p r } ( \mathbf { g } | H _ { 0 } ) } } \right] = \log \left[ \mathrm { p r } ( \mathbf { g } - \mathbf { s } | H _ { 0 } ) \right] - \log \left[ \mathrm { p r } ( \mathbf { g } | H _ { 0 } ) \right] .\tag{6}
$$

Let ${ \bf g } _ { s } ( \alpha ) = { \bf g } - \alpha { \bf s } , \alpha \in [ 0 , 1 ]$ denote the straight-line path connecting g and $\mathbf { g } - \mathbf { s }$ . Then, by the fundamenta theorem of calculus,

$$
{ \begin{array} { r l } & { \lambda _ { \mathrm { I O } } ( \mathbf { g } ) = \log [ \operatorname { p r } ( \mathbf { g } _ { s } ( 1 ) \mid H _ { 0 } ) ] - \log [ \operatorname { p r } ( \mathbf { g } _ { s } ( 0 ) \mid H _ { 0 } ) ] } \\ & { \qquad = \int _ { 0 } ^ { 1 } { \frac { d } { d \alpha } } \log [ \operatorname { p r } ( \mathbf { g } _ { s } ( \alpha ) \mid H _ { 0 } ) ] d \alpha } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { 1 } \nabla _ { \mathbf { g } _ { s } ( \alpha ) } \log [ \operatorname { p r } ( \mathbf { g } _ { s } ( \alpha ) \mid H _ { 0 } ) ] ^ { T } { \frac { d \mathbf { g } _ { s } ( \alpha ) } { d \alpha } } d \alpha } \\ & { \qquad = - \mathbf { s } ^ { T } \displaystyle \int _ { 0 } ^ { 1 } \psi _ { \mathbf { g } \mid H _ { 0 } } ( \mathbf { g } _ { s } ( \alpha ) ) d \alpha } \end{array} }\tag{7}
$$

where $\psi _ { \mathbf { g } | H _ { 0 } } \left( \mathbf { g } _ { s } ( \alpha ) \right) \triangleq \nabla _ { \mathbf { g } _ { s } ( \alpha ) }$ log[pr $\left( \mathbf { g } _ { s } ( \alpha ) \mid H _ { 0 } \right) ]$ denotes the score function of the signal-absent measurement distribution evaluated at ${ \bf g } _ { s } ( \alpha )$ . Equation (7) shows that the IO test statistic can be computed by integrating the signal-absent score function along the path connecting g and $\mathbf { g } - \mathbf { s }$ , followed by an inner product with the negative signal image. The straight-line path is used here. In principle, any continuously diferentiable path connecting g and $\mathbf { g } - \mathbf { s }$ may also be used.

To estimate the score function $\psi _ { \mathbf { g } | H _ { 0 } } \left( \mathbf { g } \right)$ , we regard a signal-absent measurement g as a perturbed realization of the corresponding noise-free image $\mathbf { f } _ { 0 }$ . This is possible when evaluating imaging systems through virtual imaging trials, in which samples of $\mathbf { f } _ { 0 }$ can be generated using specified object and imaging models, and the corresponding measurements g can then be simulated by adding measurement noise drawn from a known distribution. Given a set of paired samples $\{ ( \mathbf { f } _ { 0 } ^ { ( i ) } , \mathbf { g } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ drawn from $\mathrm { p r } ( \mathbf { g } | \mathbf { f } _ { 0 } , H _ { 0 } ) \mathrm { p r } ( \mathbf { f } _ { 0 } )$ , the DSM formulation can be applied by identifying $\mathbf { f } _ { 0 }$ as the unperturbed data and g as the perturbed data. Accordingly, the score network $\phi _ { \pmb { \theta } } ( \mathbf { g } )$ is trained by minimizing

$$
\widehat { \mathcal { L } } _ { \mathrm { D S M } } ( \pmb { \theta } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \phi _ { \pmb { \theta } } \left( \mathbf { g } ^ { ( i ) } \right) - \nabla _ { \mathbf { g } ^ { ( i ) } } \log \operatorname { p r } \left( \mathbf { g } ^ { ( i ) } \mid \mathbf { f } _ { 0 } ^ { ( i ) } , H _ { 0 } \right) \right\| _ { 2 } ^ { 2 } .\tag{8}
$$

The population minimizer of this objective estimates the marginal score of the signal-absent measurement distribution. Once the model has been successfully trained, we have

$$
\phi _ { \pmb \theta } ( \mathbf { g } ) \approx \nabla _ { \mathbf { g } } \log \mathrm { p r } ( \mathbf { g } \mid H _ { 0 } ) = \psi _ { \mathbf { g } | H _ { 0 } } ( \mathbf { g } ) .\tag{9}
$$

When the measurement noise components are independent and identically distributed (i.i.d.) Gaussian random variables with zero mean and variance $\sigma _ { n } ^ { 2 }$ , the conditional score is

$$
\nabla _ { \mathbf { g } ^ { ( i ) } } \log \operatorname { p r } \left( \mathbf { g } ^ { ( i ) } \mid \mathbf { f } _ { 0 } ^ { ( i ) } , H _ { 0 } \right) = - \frac { \mathbf { g } ^ { ( i ) } - \mathbf { f } _ { 0 } ^ { ( i ) } } { \sigma _ { n } ^ { 2 } } = - \frac { 1 } { \sigma _ { n } ^ { 2 } } \mathbf { n } ^ { ( i ) } .\tag{10}
$$

In this case, we can train the network using the measurement residual. Specifically, let $\mathbf { r } _ { \theta } ( \mathbf { g } )$ denote a neural network parameterized by weight parameters θ. The network is trained to estimate the measurement noise by minimizing:

$$
\widehat { \mathcal { L } } _ { r } ( \pmb { \theta } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \mathbf { r } _ { \pmb { \theta } } \left( \mathbf { g } ^ { ( i ) } \right) - \left( \mathbf { g } ^ { ( i ) } - \mathbf { f } _ { 0 } ^ { ( i ) } \right) \right\| _ { 2 } ^ { 2 } .\tag{11}
$$

After training, the score function can be approximated as: $\begin{array} { r } { \psi _ { \mathbf { g } | H _ { 0 } } ( \mathbf { g } ) \approx - \frac { 1 } { \sigma _ { n } ^ { 2 } } \mathbf { r } _ { \pmb { \theta } } ( \mathbf { g } ) } \end{array}$ . Using the trained residual network, the IO test statistic in Eq. (7) can be approximated by numerical integration. In this work, the left Riemann sum is employed. Let K denote the number of integration points. We then have:

$$
\widehat { \lambda } _ { \mathrm { I O } } ( \mathbf { g } ) = \frac { 1 } { \sigma _ { n } ^ { 2 } } \mathbf { s } ^ { T } \Big [ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbf { r } _ { \pmb \theta } ( \mathbf { g } - \alpha _ { k } \mathbf { s } ) \Big ] = \frac { 1 } { \sigma _ { n } ^ { 2 } } \mathbf { s } ^ { T } \bar { \mathbf { r } } _ { \pmb \theta , K } ( \mathbf { g } ; \mathbf { s } ) ,\tag{12}
$$

where $\textstyle \alpha _ { k } = { \frac { k } { K } }$ and $\begin{array} { r } { \bar { \mathbf { r } } _ { \pmb { \theta } , K } \big ( \mathbf { g } ; \mathbf { s } \big ) \triangleq \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbf { r } _ { \pmb { \theta } } \big ( \mathbf { g } - \alpha _ { k } \mathbf { s } \big ) } \end{array}$ is referred to as the signal-path-averaged residual (SPAR). Remarkably, Eq. (12) reveals an elegant interpretation of the IO test statistic: once the SPAR has been estimated, the complex IO computation reduces to a simple non-prewhitening matched-filter operation applied to the SPAR. The overview of the proposed SIO framework is shown in Fig. 1.

![](images/619bf02977b61321d8b60bfa8f17e7d1c84b58632287e8db34023356512bd3ff.jpg)  
Figure 1: Overview of the proposed score-based ideal observer (SIO) framework. During training, a denois ing convolutional neural network is trained using noisy signal-absent images, where the corresponding noise realizations serve as the training targets. During inference, the trained network is evaluated at multiple signalinterpolated images to estimate the corresponding residuals (proportional to the negative score function). The predicted residuals are numerically integrated to produce the signal-path averaged residual (SPAR), which is subsequently combined with the known signal through an inner product to compute the SIO test statistic.

## 3. NUMERICAL STUDIES AND RESULTS

Computer-simulation studies were conducted to evaluate the proposed score-based ideal observer (SIO) approximation. A signal-known-exactly (SKE) binary signal detection task with a stochastic lumpy background<sup>11</sup> was considered. A residual denoising convolutional neural network was trained exclusively on signal-absent images to estimate the signal-absent score function. The trained network was subsequently used to approximate the IO test statistic using the left Riemann-sum approximation of the proposed line-integral formulation. Observer performance was quantified by the area under the receiver operating characteristic curve (AUC). The simulation setup and corresponding results are described below.

## 3.1 Simulation Setup

A Type-I lumpy background model<sup>11</sup> was employed to generate signal-absent object images. The background object was modeled as $\begin{array} { r } { f _ { b } ( \mathbf { r } ) = \sum _ { n = 1 } ^ { N _ { b } } l ( \mathbf { r } - \mathbf { r } _ { n } ) } \end{array}$ , where $N _ { b }$ is a Poisson random variable with mean ${ \bar { N } } = 5 ,$ and ${ \bf r } _ { n }$ denotes the center location of the nth lump, sampled from a uniform distribution over a $4 0 \times 4 0$ field of view (FOV). Each lump was modeled by a 2D Gaussian function with an amplitude of 1.2 and a width of 4.8. A signal-known-exactly (SKE) binary detection task was considered. The signal was modeled as a 2D Gaussian function centered in the field of view with an amplitude of 0.6 and a width of 2.0. The imaging system was modeled as a continuous-to-discrete Gaussian blur operator, $\begin{array} { r } { h _ { m } ( { \bf r } ) = \frac { h } { 2 \pi w ^ { 2 } } \exp \Bigl ( - \frac { ( { \bf r } - { \bf r } _ { m } ) ^ { T } ( { \bf r } - { \bf r } _ { m } ) } { 2 w ^ { 2 } } \Bigr ) } \end{array}$ , where $h = 1 . 5 , w = 0 . 8$ , and $\mathbf { r } _ { m }$ denotes the spatial location of the mth image pixel that uniformly samples the FOV. The resulting measurement images consisted of $4 0 \times 4 0$ pixels. Independent Gaussian noise having zero mean and a standard deviation of 1.3 was added to the noise-free images to produce the noisy image data. Examples of the signal-present images and the signal are shown in Fig. 2.

![](images/bbba82796a301e72af1317f4432ae69990ba7870d621ec99d99e6eafc2a4eca0.jpg)  
Figure 2: From left to right are four examples of signal-present noisy images and the signal image.

## 3.2 Network training

A 17-layer DnCNN<sup>12</sup> was employed to estimate the residual (noise) image required for score computation. The network architecture followed the original DnCNN design, consisting of an initial convolutional layer with ReLU activation, 15 convolutional layers each followed by batch normalization and ReLU activation, and a final convolutional layer for residual prediction. All convolutional layers used $3 \times 3$ kernels. The DnCNN was trained using 100,000 independently generated background (signal-absent) images. During training, independent measurement noise was added to each background image on-the-fly to generate the noisy input, while the corresponding noise realization was used as the training target. The network parameters were optimized by minimizing the mean squared error (MSE) between the predicted and true residual images using the Adam optimizer with an initia learning rate of $1 0 ^ { - 3 }$ . The training was performed on a single NVIDIA L40S GPU. The trained network was subsequently employed to approximate the IO test statistic using the proposed score-based formulation.

## 3.3 Results

To investigate the numerical approximation of the proposed line-integral formulation in Eq. (7), the SIO performance was evaluated as a function of the number of integration points K. Figure 3(a) shows the AUC achieved by the proposed SIO for diferent values of K. The observer performance rapidly converges as K increases and becomes nearly unchanged for $K \geq 5$ . Consequently, $K = 5$ was employed in all subsequent experiments.

To evaluate the ability of the proposed SIO to approximate the IO, the MCMC-based ideal observer (MCMC-IO) was employed as a numerical reference for the IO performance. In addition, the Hotelling observer (HO) implemented using covariance matrix decomposition was included as a conventional linear-observer benchmark. Figure 3(b) compares the ROC curves of the proposed SIO, MCMC-IO, and HO. The ROC curve produced by the proposed SIO closely matches that of the MCMC-IO and substantially outperforms the HO, indicating that the proposed score-based formulation can accurately approximate the IO performance. Unlike MCMC-based approaches that require extensive posterior sampling for each test image or supervised learning approaches that may require task-specific retraining, the proposed method requires only a single DnCNN trained on signal-absent images and a small number of score evaluations along the signal interpolation path.

![](images/71e81773800e2c500ede3c8c9e2218156dfff5a6c47507079db8eb966017844e.jpg)  
(a) AUC versus the number of integration points K.

![](images/128f7dad296d1ae4fe6112d4de92859a16d73de9976bdfd9d04fff28c04268d1.jpg)  
(b) ROC curves of the SIO, MCMC-IO, and HO.  
Figure 3: Performance evaluation of the proposed score-based ideal observer (SIO). (a) The SIO performance rapidly converges as the number of integration points increases, and $K = 5$ was used in subsequent experiments. (b) The proposed SIO closely approximates the MCMC-IO and substantially outperforms the HO.

## 4. CONCLUSION

A novel score-based ideal observer (SIO) is proposed by expressing the Bayesian IO test statistic as the negative inner product between the known signal and an integrated score function. The proposed approach enables eficient IO approximation using a single denoising network trained only on signal-absent images, thereby eliminating the need for task-specific supervised retraining and computationally intensive posterior sampling during inference. Preliminary results demonstrate that the proposed SIO closely approximates Bayesian IO performance while substantially outperforming the Hotelling observer for the considered binary detection task involving a lumpy background model. Future work will investigate more realistic stochastic object models and inference tasks relevant to medical imaging applications.

## ACKNOWLEDGMENTS

This work was supported by startup funds provided by the Wyant College of Optical Sciences and the Department of Radiology and Imaging Sciences at the University of Arizona.

## REFERENCES

[1] Barrett, H. H. and Myers, K. J., [Foundations of Image Science], John Wiley &amp; Sons (2013).

[2] Kupinski, M. A., Hoppin, J. W., Clarkson, E., and Barrett, H. H., “Ideal-observer computation in medical imaging with use of markov-chain monte carlo techniques,” Journal of the Optical Society of America A 20(3), 430–438 (2003).

[3] Zhou, W. and Anastasio, M. A., “Markov-Chain Monte Carlo approximation of the Ideal Observer using generative adversarial networks,” in [Medical Imaging 2020: Image Perception, Observer Performance, and Technology Assessment], 11316, 113160D, International Society for Optics and Photonics (2020).

[4] Li, D., Li, K., Zhou, W., and Anastasio, M. A., “Approximating the ideal observer for joint signal detection and estimation tasks by the use of markov-chain monte carlo with generative adversarial networks,” Journal of Medical Imaging 12(5), 051810–051810 (2025).

[5] Kupinski, M. A., Edwards, D. C., Giger, M. L., and Metz, C. E., “Ideal observer approximation using bayesian classification neural networks,” IEEE transactions on medical imaging 20(9), 886–899 (2001).

[6] Zhou, W., Li, H., and Anastasio, M. A., “Approximating the Ideal Observer and Hotelling Observer for binary signal detection tasks by use of supervised learning methods,” IEEE Transactions on Medical Imaging 38(10), 2456–2468 (2019).

[7] Zhou, W., Li, H., and Anastasio, M. A., “Approximating the Ideal Observer for joint signal detection and localization tasks by use of supervised learning methods,” IEEE Transactions on Medical Imaging 39(12), 3992–4000 (2020).

[8] Song, Y. and Ermon, S., “Generative modeling by estimating gradients of the data distribution,” Advances in neural information processing systems 32 (2019).

[9] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B., “Score-based generative modeling through stochastic diferential equations,” arXiv preprint arXiv:2011.13456 (2020).

[10] Vincent, P., “A connection between score matching and denoising autoencoders,” Neural computation 23(7), 1661–1674 (2011).

[11] Rolland, J. P. and Barrett, H. H., “Efect of random background inhomogeneity on observer detection performance,” Journal of the Optical Society of America A 9(5), 649–658 (1992).

[12] Zhang, K., Zuo, W., Chen, Y., Meng, D., and Zhang, L., “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,” IEEE transactions on image processing 26(7), 3142–3155 (2017).