# Multi-Person Human Motion Forecasting in Complex Scenes

Serdar Ozsoy<sup>1,2⋆</sup> , Lars Doorenbos<sup>1,2</sup> , and Juergen Gall<sup>1,2</sup>

<sup>1</sup> University of Bonn, Germany

2 Lamarr Institute for Machine Learning and Artificial Intelligence, Germany {soezsoy,doorenbos,gall}@iai.uni-bonn.de https://serdarozsoy.github.io/OCSD-project/

Abstract. Accurately forecasting the movement of people in complex scenes requires reasoning over the past and present state of the entire environment. In this context, efectively incorporating object information and social interactions into a unified framework remains particularly challenging. To address this, we propose Object-Conditioned Social Difusion (OCSD), a conditional difusion model that integrates motion history, multi-person interactions, and object cues into a single framework. OCSD uses an object-conditioning mechanism that modulates denoising at every timestep, enabling fine-grained human-object reasoning, and a social encoder that models the interactions between all humans in the scene. As a result, our model naturally handles varying group sizes, complex social interactions, and supports sampling multiple plausible futures. Extensive experiments show that OCSD achieves state-of-the-art results on the Humans in Kitchens (HiK) and HOI-M<sup>3</sup> benchmarks. It reduces the two-second path error by 121.5 mm (31.3%) on HiK and 130.5 mm (33.2%) on HOI-M<sup>3</sup> compared to prior work, and produces more realistic long-term forecasts.

Keywords: Human motion forecasting · Difusion models

## 1 Introduction

Anticipating how people will move in complex scenes is a crucial capability for autonomous systems to operate safely in various domains, including robotics [13,15,45] and autonomous driving [7,23,24,37]. However, achieving the accuracy desired to do so is a challenging task. A key part of this challenge lies in understanding the factors that influence human motion at diferent timescales. Even for short time horizons, accurate human motion forecasting requires more than extrapolating past poses: models must reason jointly about their own history, interactions with other people in the scene, as well as the objects. Longer time horizons introduce even more complexity for forecasting. Specifically, the mapping from observed motion to future behavior of people in complex scenes becomes an inherently ambiguous problem, where multiple motions can be correct given the same context.

![](images/f5a3e215872ada9a548fda3897fc9fa8d68bcd92f8ece73bf7b967057f12a971.jpg)  
Fig. 1: Multi-person human motion forecasts in complex scenes with OCSD. We show the last observed frame on the left. The next two panels display forecast poses after 1 second (middle) and 2 seconds (right), with grey skeletons denoting the ground truth and colored skeletons representing the predictions. The gray skeletons are well aligned with the forecast poses, which shows that our model can generate plausible poses in complex scenes by integrating context.

Short- and long-horizon forecasting are typically considered separately. For instance, short-term works tend to disregard the probabilistic characteristic of human motion and output only the most likely prediction (e.g., [39]). In contrast, works focusing on longer horizons produce stochastic predictions with generative models, but, as a result, tend to underperform deterministic models in terms of local realism [31]. Capturing these diferent factors within a unified framework remains an open problem.

To address this challenge, we propose a conditional denoising difusion model that holistically integrates the entire scene context. Our model uses an objectconditioning mechanism that modulates the denoising at each timestep, through which objects can impact the predicted future motion. Additionally, we use a social encoder that extracts the social context and integrates it into our model. As a result, our model can capture fine-grained human-object dependencies while also regarding social context, as shown in Fig. 1.

We show through extensive experiments that these modeling capabilities translate into strong empirical performance. We evaluate our method on two benchmarks, Humans in Kitchens (HiK) [42] and HOI-M<sup>3</sup> [58], on both shortand long-term human motion forecasting. For short horizons, the sequences our model generates outperform those from eight baselines in accuracy of predicted paths and poses, while for longer horizons our method reaches high realism and diversity. For example, on HiK, which has the most people in the scene, we reduce the state-of-the-art two-second path error by 121.5mm, representing an improvement of 31.3%. For longer forecasts up to ten seconds, our model achieves 25.9% higher realism than the previous state-of-the-art approaches. Similarly, our model reduces the two-second path error by 130.5 mm (33.2%) on HOI-M<sup>3</sup> compared to prior work. In short, our main contributions are as follows:

– We introduce OCSD, a conditional difusion framework that unifies past conditioning, social context, and object signals for multi-person motion forecasting in complex scenes.

We set a new state-of-the-art across various metrics on the HiK and HOI-M<sup>3</sup> datasets.

## 2 Related Works

Single-Person Human Motion Prediction. Early works in human motion prediction operate in single-person settings, with both deterministic models that predict a single future sequence and probabilistic models that predict multiple diverse future motions. Deterministic methods typically use Recurrent Neural Networks [14,19,30], Graph Convolutional Networks [25,27,29], Transformers [2,9], or Multilayer Perceptron-based [8,17,49] approaches, while Generative Adversarial Networks [6,22] and Variational Autoencoders [56] form the foundations for probabilistic human motion prediction. Subsequent work in probabilistic prediction mostly focuses on increasing the diversity of the predictions [3,4,12]. In addition, [28,52,54,57] not only diversify predictions but also ofer a way to control the diversity of predicted motions. Recently, difusion-based models [5,10,11,36,40] achieve high-fidelity forecasts while providing diversity, for instance, by re-framing the prediction as a masked motion completion task [10], or introducing latent difusion [5].

Multi-Person Human Motion Prediction. The setting of multi-person human motion prediction introduces an extra challenge of modeling interdependent social dynamics. Transformer-based architectures [20,32,46,47,50,53] became a common approach in this domain. For example, [47] uses a local-range encoder for individual motion and a global-range encoder for social interactions. Subsequent work refined this by modeling interactions at a finer granularity. For instance, [32] proposes modeling interactions at the body-part level, while [53] introduces an auxiliary loss to predict the future distance between joints. Besides transformer-based models, [60] proposes computationally eficient modeling of social interaction via an MLP-based framework. Generative models are also used in this context to capture social stochasticity: [55] introduces a duallevel generative modeling framework to separately model independent individual motion and global social interactions, and [44] leverages a difusion model with an order-invariant aggregation function to flexibly manage a varying number of people.

Human Motion Prediction with Object Context. Integrating object information is typically considered separate from social context and has its own set of approaches. As an early method, [26] proposes a two-stage pipeline to first predict future contact maps from a scene point cloud, and then forecasts the pose conditioned on these maps. [38] builds on this with a three-stage pipeline by predicting contact points, root trajectory, and full pose, respectively. While [51] proposes to constrain the whole-body motion by introducing the Signed Distance Function (SDF) volume as a dense global scene representation, [48] uses segmented object instances from 3D point clouds to predict diverse human motion.

Only recently, unified models attempt to merge both social and scene context. $[ 1 ]$ is an early attempt using GNNs with a pretrained object detector. [31] introduces a difusion-based model for multi-person settings that uses scene object information while targeting long-term (up to 10s) generation with an emphasis on diversity and plausibility. In contrast, [39] also ofered a unified hierarchical framework that operates on scene objects while focusing on more short-term (0.5 - 2s) prediction accuracy. Our model bridges these two directions: we achieve state-of-the-art short-term (0.5-2s) path and pose accuracy while also ensuring long-term (10s) plausibility and diversity with both object and social context.

## 3 Method

Forecasting the motion of multiple people in complex scenes requires modeling several factors: the past motion, the interactions between people in the scene, and the type and location of various objects. We denote the observed motion sequence of a group of P people for a duration of $T _ { i n }$ frames by a tensor of coordinates $\mathbf { X } _ { T _ { i n } } \in \bar { \mathbb { R } } ^ { T _ { i n } \times \bar { P ^ { \times } } J \times 3 }$ , where J represents the number of joints, and we refer to the pose of a single person p at frame t as $\mathbf { x } _ { t } ^ { ( p ) }$ . If there are fewer than $P$ people in the scene, the superfluous dimensions are filled with zeros. The object information is encoded as a set of M objects $\mathbf { O } = \{ o _ { 1 } , \cdot \cdot \cdot , o _ { M } \}$ , with each object o represented by N vertices $\mathbf { V } _ { o } \in \mathbb { R } ^ { N \times 3 }$ , fixed at the last frame of the observations, and a semantic type y. Given this context, the goal is to forecast the poses of all people in the future $T _ { o u t }$ frames, ${ \bf X } _ { T _ { o u t } } \in \mathbb { R } ^ { T _ { o u t } \times P \times J \times 3 }$ conditioned on the observed sequences ${ \bf X } _ { T _ { i n } }$ and the object information O. The full sequence thus lasts $T = T _ { i n } + T _ { o u t }$ frames.

Learning this mapping from observed motion to future behavior of people in complex scenes is an inherently ambiguous problem: for the same input, multiple plausible motions can be correct, especially for longer-term predictions. Therefore, we frame the motion forecasting task as learning a conditional distribution of future motion based on the observed motion and scene context, for which we rely on conditional difusion models [18]. The core of our model, OCSD, lies in how it represents and fuses information, which we describe in the sections below. We provide an overview of our method in Fig. 2.

## 3.1 Input Representations

Pose representation. The observed poses ${ \bf X } _ { T _ { i n } }$ contain the joint coordinates in the global coordinate system. However, directly operating in the global coordinates is suboptimal for pose forecasting [16]. Therefore, we use an alternative representation as in [44], and transform each per-person pose vector by $\mathbf { x } _ { t } ^ { ( p ) } = \left\lceil \mu _ { t } ^ { ( p ) } \in \mathbb { R } ^ { 3 } , r _ { t } ^ { ( p ) } \in \mathbb { R } ^ { 3 } , q _ { t } ^ { ( p ) } \in \mathbb { R } ^ { 3 J } \right\rceil$ . Here, $\mu _ { t } ^ { \left( p \right) }$ is the global translation given by the 3D world-coordinate position of the hip center, and $r _ { t } ^ { ( p ) }$ is the global rotation of the body’s 2D-plane facing direction represented as a 3D rotation. $q _ { t } ^ { ( p ) }$ are the local joint positions, representing the 3D positions of the J joints where the person is translated to the origin and rotated such that the hips are parallel to the x-axis. The total dimensionality of $\mathbf { x } _ { t } ^ { \left( p \right) }$ , therefore, is $D = 6 + 3 J$ We further standardize the local joints by the mean and standard deviation of each joint in the training set.

![](images/d5fad70f533340a57138779f3543e48d40dad73ecfc19b87243eb418f77d2669.jpg)

![](images/c84e61c9a6b9c8332b7d5473eed2512e6daed48fced6c6661ffaa20d5b0e5583.jpg)  
Fig. 2: Overview of OCSD. Top: Our method takes a sequence of poses for a dynamic number of people in the scene, along with object information. These inputs are split into the self-stream and two conditioning streams for the objects and social context. From this, OCSD predicts the poses of all people for the next $T _ { o u t }$ frames. Bottom: Details of our main model, following a U-Net architecture. We integrate the social context for each reference person into the bottleneck, while object information is added throughout the network via FiLM modules.

Social context representation. After this processing, all pose vectors are represented in their own local coordinate system. While this is important for stable learning [41], it destroys all information about the position of a person relative to the other people in the scene. To use this information in the model, we design a social representation that considers the relative position of other people in the scene from the perspective of each person. In this context, we refer to the person whose perspective is considered as the primary person $p ^ { \star }$

To obtain the social information for $p ^ { \star }$ , we establish a translation $\mathbf { T } ^ { ( { p } ^ { \star } ) }$ and rotation $\mathbf { R } ^ { \left( p ^ { \star } \right) }$ with regard to their pose in the last observed frame $t _ { r e f }$ Then, for every person $p ,$ we transform their entire pose vector $\mathbf { x } _ { t } ^ { \left( p \right) }$ into this coordinate system. Using these pose vectors, we construct a pairwise tensor $\pmb { \chi } \in \mathbb { R } ^ { T \times P \times P \times D }$ where each slice $\pmb { \chi } _ { t } ^ { ( p ^ { \star } , p ) }$ contains $\mathbf { x } _ { t } ^ { \left( p \right) }$ from the perspective of $p ^ { \star }$ . From $_ { x }$ , we extract two separate representations. These are the $s e l f -$ stream, representing the pose vector of all individual people in the scene $( \mathrm { i . e . } ,$ the diagonal of $x )$ , and the social-stream, which contains pose vectors of all other people in the scene relative to the reference person $p ^ { \star }$ . The self-stream is directly fed into the U-Net [35] of $f _ { \theta }$ . In contrast, the social-stream will be used to control the generation process, as described in Sec. 3.2.

Object context representation. To represent the objects present in the scene, we create personalized object information representations in a similar way to the social representation, encoding the relationship between a reference person and the objects in the scene. We canonicalize the vertices of an object $\mathbf { V } _ { o }$ into each person’s $p ^ { \star }$ reference frame using the same translation $\mathbf { T } ^ { ( { p } ^ { \star } ) }$ and rotation $\mathbf { R } ^ { \left( p ^ { \star } \right) }$ as in social context. Besides flattening N vertices to 3N points, we encode the semantics of the object into a one-hot vector. These are then combined into the object representations $\tilde { \mathbf { V } }$ by concatenation, leading to a final representation of $D _ { o } = 3 N + C$ , with $C$ the number of object types present.

## 3.2 Integrating Social and Object Context

We propose to use two mechanisms to integrate the social and object context representations into the pose forecasting model $f _ { \theta }$ to control the generation process, which we show schematically in Fig. 2 and describe in turn below.

Social integration. The social context is represented in the social-stream obtained from the pairwise relational tensor X. Similar to [44], we pass this information through a learned encoder and integrate the resulting embedding in the bottleneck of the U-Net. We utilize this Social Encoder to temporally downsample the social-stream through three successive blocks, each consisting of three strided 1D convolutions with a SiLU activation function and layer normalization. This encoder outputs a set of features $\mathbf { E } \in \mathbb { R } ^ { T / 8 \times P \times P \times d ^ { \prime } }$ , and $\mathbf { e } ^ { ( p ^ { \star } , p ) }$ denotes the encoded representation for person p relative to primary person $p ^ { \star }$

One challenge to consider is that there can be a dynamic number of people present in the scene. Therefore, to obtain a single and fixed-dimensional social representation for a primary person, $\mathbf { e } ^ { ( p ^ { \star } ) }$ , we mask out any zero-padded entries of E that are not related to any person. Then, we perform mean pooling across the second $P$ dimension. The final vector $\mathbf { e } ^ { ( p ^ { \star } ) }$ summarizes the entire social context from the perspective of $p ^ { \star }$ and is concatenated to the $p ^ { \star } \mathrm { \tilde { s } }$ self-stream features at the U-Net bottleneck, followed by a 1D Convolution, denoted as “Merge” in Fig. 2. Thereby, the model jointly processes the primary person’s own motion while integrating the surrounding social dynamics.

Object context integration. We design our object conditioning as a multiscale modulation mechanism applied at multiple stages $s ,$ which correspond to the features $\mathrm { H } \in [ d _ { 1 } , d _ { 2 } , d _ { 3 } , h _ { 2 } , u _ { 3 } , u _ { 2 } ]$ ] of the U-Net. This process is done for each person p and consists of three parts: First, the canonicalized object vertices are projected via an MLP to match the feature dimension of that stage. The resulting features form a set of M object tokens for each person $p ,$ which encode local scene information and can be integrated into the model. Second, the motion features $H _ { t } ^ { ( p , s ) }$ for person $p$ at layer s for each timeframe $t \in T$ are processed by cross-attention modules. In this setting, $H _ { t } ^ { ( p , s ) }$ acts as the query, and the M projected object tokens serve as keys and values. This way, the attention mechanism produces an embedding that summarizes the objects most relevant to the motion of the person $p$ in frame t at stage s. Finally, we integrate these features via Feature-wise Linear Modulation (FiLM) [33]. The object embedding is passed through a linear projection to predict a per-timeframe scale vector $\gamma _ { t } ^ { ( p , s ) }$ and a bias vector $\beta _ { t } ^ { ( p , s ) }$ to modulate the motion features:

$$
\widehat { H } _ { t } ^ { ( p , s ) } = H _ { t } ^ { ( p , s ) } \odot \big ( 1 + \gamma _ { t } ^ { ( p , s ) } \big ) + \beta _ { t } ^ { ( p , s ) } .\tag{1}
$$

In the first half of the U-Net, FiLM modulation is applied through the skip connections. In contrast, we apply it directly to the features in the second half. This process allows object context to precisely influence the motion generation.

## 3.3 Conditional difusion model

Training. Our difusion model $f _ { \theta }$ is trained to predict the clean motion $\mathbf { X } _ { 0 }$ from a noised sample X based on the past motion and scene context. Let $\mathbf { X } _ { 0 }$ be the full clean motion sequence, which includes the past ${ \bf X } _ { T _ { i n } }$ and future $\mathbf { X } _ { T _ { o u t } }$ . Given a difusion noise schedule $\{ \alpha _ { \tau } \} _ { \tau = 1 } ^ { K }$ with K denoising steps and its cumulative product $\begin{array} { r } { \bar { \alpha } _ { \tau } = \prod _ { s = 1 } ^ { \tau } \alpha _ { s } } \end{array}$ , we form the input tensor $\mathbf { X } _ { \tau }$ by combining the clean past and the noised future:

$$
{ \bf X } _ { \tau } = \big [ { \bf X } _ { T _ { i n } } ~ ; ~ \sqrt { \bar { \alpha } _ { \tau } } { \bf X } _ { T _ { o u t } } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \varepsilon \big ] ,\tag{2}
$$

where $\varepsilon \sim \mathcal { N } ( 0 , I )$ ). Our network $f _ { \theta }$ is trained to predict the full clean motion $\mathbf { X } _ { 0 }$ from $\mathbf { X } _ { \tau }$ , conditioned on the denoising step $\tau$ and object context O with a $\mathcal { L } _ { 1 }$ reconstruction objective:

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { x } _ { 0 } } = \mathbb { E } _ { \boldsymbol { \tau } } \left[ \left. \mathbf { X } _ { 0 } - f _ { \theta } ( \mathbf { X } _ { \boldsymbol { \tau } } , \boldsymbol { \tau } , \mathbf { O } ) \right. _ { 1 } \right] . } \end{array}\tag{3}
$$

We modify this base loss by applying a frame mask $\mathbf { M } _ { \mathrm { f r a } }$ to compute the loss only for frames $t > T _ { \mathrm { i n } }$ , and a presence mask $\mathbf { M } _ { \mathrm { p r e s } }$ to ignore absent individuals. Moreover, we decompose the masked loss into three components in our hybrid pose representation to ensure stable training: root translation $\mu ,$ , root orientation $r ,$ and local joint pose $q .$ . Each is normalized by its number of valid elements:

$$
\mathcal { L } = \frac { \| \varDelta _ { \mu } \| _ { 1 } } { N _ { \mu } } + \frac { \| \varDelta _ { \mathrm { r } } \| _ { 1 } } { N _ { \mathrm { r } } } + \frac { \| \varDelta _ { \mathrm { q } } \| _ { 1 } } { N _ { \mathrm { q } } } .\tag{4}
$$

Here, $\varDelta _ { \mu } , \varDelta _ { r }$ , and $\varDelta _ { q }$ denote the masked per-element errors for root translation, root orientation, and local joint pose, respectively, and $N _ { \mu } , N _ { r }$ , and $N _ { q }$ are the valid element count for each component.

Inference. We sample using the standard DDPM reverse process. To ensure the generated future is coherent with the given past ${ \bf X } _ { T _ { i n } }$ , we apply an inpainting strategy. A single noise vector $\mathbf { z } _ { \mathrm { p a s t } }$ is drawn for the past frames, and at each denoising step τ , a noised version of the past is concatenated with the predicted $\mathbf { X } _ { \tau - 1 }$ before passing it to the next step:

$$
{ \bf X } _ { \tau - 1 } ~ = ~ [ \sqrt { \bar { \alpha } _ { \tau } } { \bf X } _ { T _ { i n } } + \sqrt { 1 - \bar { \alpha } _ { \tau } } { \bf z } _ { \mathrm { p a s t } } ; { \bf X } _ { \tau - 1 } ^ { \mathrm { p r e d i c t e d } } ]\tag{5}
$$

which prevents the observed past from drifting during the sampling process and avoids discontinuities between observed and predicted poses.

## 4 Experiments

We describe datasets, metrics and implementation details briefly below and provide more details in the supplementary material.

Datasets. We evaluate our method on two recent multi-person human motion forecasting datasets. Humans in Kitchens [42] (HiK) is a motion-capture dataset of people interacting in four kitchen environments, containing up to 16 people and 50 annotated objects in a scene. Following the standard protocol [44,31,39], we train on three kitchens (A, B, and C) and evaluate on curated sequences from kitchen D, focusing on transitional moments. $\mathbf { H O I - M ^ { 3 } }$ [58] is a large-scale multi-human multi-object dataset captured in realistic indoor rooms. It contains 199 long sequences with up to 5 people and at least 5 objects per scene. We use the 49 living room sequences publicly available (1-51, except nonreleased 43 and 45), and set aside 20% as the test set following [39].

Metrics. We assess our model’s performance using four metrics. We evaluate short-term forecasts as in [39] with Path error, i.e., the error of root joint trajectory in millimeters (mm), and Pose error, which is the Mean Per Joint Position Error. Long-term predictions are scored with Normalized Directional Motion Similarity [43] (NDMS), which measures the realism of generated motion, and Unique Motion Word Ratio [31] (UMWR), which tests diversity.

Implementation Details. We train OCSD for 90 epochs using the Adam [21] optimizer with a learning rate of $3 \times 1 0 ^ { - 4 }$ . We define one epoch as 20,000 training samples. We follow the protocol from [31,39] and use the HiK dataset at its original 25 FPS. The HOI-M<sup>3</sup> dataset is downsampled from 60 FPS to 30 FPS. For short-term experiments, we provide 1 second of motion as input to predict the next 2 seconds. For long-term experiments, we use 2 seconds of input to predict 10 seconds. For evaluation, as our method is stochastic, we forecast four pose sequences per input motion, compute the corresponding metrics, and report the average performance.

Table 1: Short-term pose forecasting results. Baseline results are taken from [39]. Our approach consistently reaches the best results.
<table><tr><td colspan="2" rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Path Error (mm)</td><td colspan="6">Pose Error (mm)</td></tr><tr><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s</td><td>mean</td><td>0.5s</td><td>1.0s</td><td>1.5s</td><td></td><td>2.0s mean</td></tr><tr><td rowspan="9">HIK [42]</td><td colspan="2">ContAware [26]</td><td>138.3</td><td>251.3</td><td>352.4</td><td>430.8</td><td>239.3</td><td>87.8 117.1</td><td></td><td>136.1</td><td>147.8 106.8</td></tr><tr><td colspan="2">GIMO [59]</td><td>143.0</td><td>259.7</td><td>384.2</td><td>487.3</td><td>258.6</td><td>85.5 121.6</td><td></td><td>142.0</td><td>153.0 109.3</td></tr><tr><td colspan="2">STAG [25]</td><td>124.7</td><td>245.4</td><td>352.4</td><td>479.2</td><td>239.7</td><td>81.7 110.9</td><td></td><td>132.5</td><td>140.9100.6</td></tr><tr><td colspan="2">MutualDistance e [51]</td><td>128.7</td><td>253.2</td><td>372.5</td><td>479.2</td><td>246.0</td><td>82.9 117.2</td><td></td><td>138.5</td><td>148.2 105.9</td></tr><tr><td colspan="2">T2P [20]</td><td>88.6</td><td>199.6</td><td>318.8</td><td>447.1</td><td>208.7</td><td>74.2 108.6</td><td></td><td>127.5 142.6</td><td>96.9</td></tr><tr><td colspan="2">IAFormer [50]</td><td>83.9</td><td>195.0</td><td>311.1</td><td>434.9</td><td>200.1</td><td>71.5106.5</td><td>125.9</td><td>137.7</td><td>95.0</td></tr><tr><td colspan="2">SAST [31]</td><td>86.7</td><td>187.4</td><td>284.9</td><td>398.1</td><td>189.0</td><td>72.3 101.4</td><td>118.0</td><td>128.6</td><td>93.2</td></tr><tr><td colspan="2">HUMOF [39]</td><td>78.8</td><td>177.4</td><td>278.8</td><td>388.4</td><td>180.7</td><td>71.2 100.6</td><td></td><td>116.9 127.1</td><td>90.2</td></tr><tr><td colspan="2">OCSD</td><td></td><td>78.9 132.7</td><td>197.8</td><td>266.9</td><td>137.6</td><td>69.8</td><td>99.4</td><td>112.3</td><td>121.9</td></tr><tr><td rowspan="8">HOI-M3</td><td colspan="2">[58] ContAware [26]</td><td>125.6</td><td>239.9</td><td>285.4</td><td>432.9</td><td>236.9</td><td>106.2 152.8</td><td>174.3</td><td></td><td>88.4 197.1 137.5</td></tr><tr><td colspan="2">GIMO [59]</td><td>131.4</td><td>247.7</td><td>300.9</td><td>454.4</td><td>255.2</td><td>107.9 155.9</td><td>182.6</td><td></td><td>207.1 141.0</td></tr><tr><td colspan="2">STAG [25]</td><td>128.1</td><td>234.4</td><td>289.5</td><td>438.1</td><td>239.7</td><td>102.5 145.0</td><td></td><td>167.1</td><td>185.6131.2</td></tr><tr><td colspan="2">MutualDistance [51]</td><td>83.6</td><td>169.7</td><td>278.8</td><td>402.8</td><td>189.9</td><td>94.4 137.1</td><td></td><td>158.2</td><td>181.3 125.3</td></tr><tr><td colspan="2">T2P [20]</td><td>74.2</td><td>168.8</td><td>296.9</td><td>429.2</td><td>194.1</td><td>88.0 135.8</td><td>160.9</td><td></td><td>183.2 124.6</td></tr><tr><td colspan="2">IAFormer [50]</td><td>69.0</td><td>166.6</td><td>290.1</td><td>423.5</td><td>186.3</td><td>86.1 135.0</td><td>165.9</td><td></td><td></td></tr><tr><td colspan="2">SAST [31]</td><td>75.0</td><td>166.2</td><td>280.4</td><td>403.9</td><td>184.8</td><td>89.2 133.8</td><td>167.0</td><td></td><td>180.7 121.6</td></tr><tr><td colspan="2">HUMOF [39]</td><td>67.1</td><td>156.6</td><td>268.4</td><td>393.1</td><td>174.6</td><td>86.3 129.6</td><td>155.0</td><td></td><td>182.9 122.3</td></tr><tr><td colspan="2">OCSD</td><td></td><td>52.4 115.5</td><td></td><td>5 185.9 262.6</td><td> 123.5</td><td>63.8</td><td>93.1 112.5 127.9 87.3</td><td></td><td>172.1 117.9</td></tr></table>

Table 2: Long-term pose forecasting results on HiK. Baseline results are taken from [31]. Unlike previous approaches, our method performs well even for a 10-second horizon.
<table><tr><td rowspan="2"></td><td rowspan="2">NDMS ↑</td><td colspan="4">UMWR ↑</td></tr><tr><td>2s</td><td>4s</td><td>6s</td><td>8s 10s</td></tr><tr><td>MRT [47]</td><td>0.16</td><td>0.09</td><td>0.08</td><td>0.07 0.07</td><td></td></tr><tr><td>HisRep [27]</td><td>0.23</td><td>0.12</td><td>0.07</td><td>0.06</td><td>0.06 0.06</td></tr><tr><td>SiMLPe [17]</td><td>0.27</td><td>0.15</td><td>0.07</td><td>0.07</td><td>0.06 0.06</td></tr><tr><td>TriPod [i]</td><td>0.13</td><td>0.18</td><td>0.14</td><td>0.12</td><td>0.11 0.11</td></tr><tr><td>SAST [31]</td><td>0.17</td><td></td><td></td><td>0.41 0.21 0.15</td><td>0.14 0.15</td></tr><tr><td>OCSD</td><td>0.34</td><td></td><td></td><td></td><td>0.47 0.45 0.45 0.44 0.44</td></tr></table>

## 4.1 Main Results

We conduct a comprehensive quantitative evaluation against recent state-of-theart methods on the two datasets. We show our main results on HiK and HOI-M<sup>3</sup> for short-term forecasting in Tab. 1 and long-term forecasting in Tab. 2. Our model demonstrates superior performance in both short-term path and pose accuracy and long-term motion realism and diversity, outperforming specialized methods on their respective benchmarks. Specifically, our method achieves the best performance in all metrics for HiK in the short-term setting, except for the 0.5-second path error, where we trail HUMOF by 0.1 mm. Notably, the longer the time horizon considered, the larger the relative benefit of our method: after one second, we improve over HUMOF by 44.7 mm, while after two seconds this increases to 121.5 mm (31.3%). The pose error results are better for all four time-points considered; however, the margins are narrower. On HOI-M<sup>3</sup>, a similar trend is visible where OCSD outperforms the baselines in all aspects.

Table 3: Ablating object integration via cross-attention and FiLM. We report the path and pose error on HiK. Our combined mechanism is important to get the best results.
<table><tr><td rowspan="2">Method</td><td colspan="5">Path Error (mm)</td><td colspan="5">Pose Error (mm)</td></tr><tr><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s</td><td>mean</td><td>0.5s</td><td>1.0s</td><td>1.5s</td><td></td><td>2.0s mean</td></tr><tr><td>OCSD</td><td>78.9 132.7 197.8 266.9</td><td></td><td></td><td></td><td>137.6</td><td>69.8</td><td></td><td>99.4 112.3</td><td>121.9</td><td>88.4</td></tr><tr><td>only Cross-Att.</td><td>83.0</td><td>138.5</td><td>204.5</td><td>268.8</td><td>141.8</td><td>70.3 100.2</td><td></td><td>113.3</td><td>121.6</td><td>88.9</td></tr><tr><td>only FiLM</td><td>87.5</td><td>146.3</td><td>213.9</td><td>284.7</td><td>149.6</td><td>74.5 102.3</td><td></td><td>114.3</td><td>123.9</td><td>91.6</td></tr></table>

The comparison to long-term pose forecasting models in both realism and diversity further validates the benefits of our approach. We keep a high diversity (UMWR) across the entire ten seconds. This demonstrates that our model captures a much wider distribution of plausible human motion. In contrast, predictions by methods such as SAST tend to “freeze” after a certain amount of time, where the people stop moving entirely. We provide the related examples in the supplementary material. Furthermore, we achieve state-of-the-art results for the realism metric NDMS, improving over SiMLPe by 25.9%, from 0.27 to 0.34. This shows that our predicted motion segments are more similar to the actual motion.

## 4.2 Ablation studies

While we discuss the three main ablations regarding object conditioning, social and object context, and loss term below, we provide additional experiments in the supplementary material.

Object Conditioning. We validate the design of our object conditioning mechanism by comparing it with two other variants. First, we remove FiLM and use the underlying cross-attention directly to inject the object condition, denoted as “only Cross-Att”. Second, we remove the cross-attention and instead compute a masked mean-pooling over valid object tokens to obtain only a global object context per person and timeframe, denoted as “only FiLM”. From the results in Tab. 3, we find that FiLM itself provides the weakest form of conditioning. Vanilla cross-attention improves upon FiLM by providing more detailed context, as is visible in the 7.8 mm improvement in the mean path error. Nonetheless, our proposed object context integration module yields further improvements in the results. Specifically, we increase the mean path error by another 4.2 mm compared to cross-attention and 12.0 mm with respect to FiLM.

Social and Object Context. Next, we validate the benefits of incorporating social interaction and object information against three ablated versions: “self + social”, which removes the entire object-conditioning pathway, “self + object”, which removes the social encoder and merging layers, and “only self”, which removes both social and object context.

For short-term forecasting, Tab. 4 shows the efect of removing object and social information. For path error, both context are clearly beneficial. Incorporating social context reduces mean path error by 3.4 mm, while object information reduces it by 5.2 mm. By combining both forms, our full model achieves the best result overall, lowering the mean path error by 12.3 mm. This result demonstrates that the two sources of context are complementary rather than redundant, which is intuitive since the global trajectory is afected by where other people and objects are. In contrast, local pose in short horizons is primarily determined by the person’s own dynamics, so the “only self” model already achieves a low pose error. Therefore, the impact of context is not obvious in the mean over diferent types of activities. We provide more details on the context contribution in the supplementary material by reporting the metrics for representative activities.

Table 4: The importance of context for short-term forecasting. We show the path and pose error on HiK. Integrating all forms of context leads to the best performance.
<table><tr><td rowspan="2">Method</td><td colspan="4">Path Error (mm)</td><td colspan="4">Pose</td><td colspan="3">Error (mm)</td></tr><tr><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s</td><td>mean</td><td>0.5s</td><td></td><td>1.0s</td><td>1.5s</td><td></td><td>2.0s mean</td></tr><tr><td>OCSD</td><td></td><td></td><td>78.9 132.7 197.8</td><td>266.9</td><td></td><td>9137.6</td><td>69.8</td><td>99.4</td><td>112.3</td><td>121.9</td><td>88.4</td></tr><tr><td>self + social</td><td>81.6</td><td>141.9</td><td>213.8 </td><td>281.5</td><td></td><td>146.5</td><td>69.7 100.3</td><td></td><td>112.9</td><td>124.3</td><td>89.1</td></tr><tr><td>self + object</td><td>83.1</td><td>140.1</td><td>208.1</td><td>277.4</td><td></td><td>144.7</td><td>72.5</td><td>102.7</td><td>114.9</td><td>123.5</td><td>90.4</td></tr><tr><td>only self</td><td>84.5</td><td>146.2</td><td>217.7</td><td>289.8</td><td></td><td>149.9</td><td>70.8</td><td>99.2</td><td>112.2</td><td>122.9</td><td>88.5</td></tr></table>

Table 5: The impact of context for long-term forecasting. On the HiK dataset, realism score (NDMS) increases with the context while diversity score (UMWR) is lower because of the introduced constraints from environment and people.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NDMS ↑</td><td colspan="4">UMWR ↑</td></tr><tr><td>2s</td><td>4s</td><td>6s 8s</td><td>10s</td></tr><tr><td>OCSD</td><td>0.34</td><td></td><td></td><td>0.47 0.45 0.45 0.44 0.44</td></tr><tr><td>self + social</td><td>0.33</td><td></td><td></td><td>0.47 0.45 0.44 0.44 0.44</td></tr><tr><td>self + object</td><td>0.33</td><td></td><td>0.470.45 0.450.440.44</td><td></td></tr><tr><td>only self</td><td>0.32</td><td></td><td></td><td>0.48 0.46 0.45 0.45 0.45</td></tr></table>

For long-term forecasting, Tab. 5 shows that the NDMS realism metric also improves with more context. Interestingly, the diversity of samples as measured by the UMWR metric is slightly higher without any context, as more movements are plausible without any constraints from the environment. Nonetheless, the diversity of the diferent variants is similar.

Loss Aggregation. Finally, we ablate the way the difusion loss is aggregated over channels. We compare to a baseline where we remove the componentwise decomposition of Eq. (4). Instead, we compute a single scalar loss as the mean of the prediction error over all elements. The results in Tab. 6 show that our component-wise decomposition, which assigns more weight to the global positioning of the sequences, is important to reach a good performance.

Table 6: Ablating loss weighting. We report the average path and pose error over four samples on HiK. Using the loss weighting improves the results.
<table><tr><td rowspan="2">Method</td><td colspan="4">Path Error (mm)</td><td colspan="4">Pose Error (mm)</td></tr><tr><td>0.5s 1.0s</td><td>1.5s</td><td>2.0s</td><td>mean</td><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s mean</td></tr><tr><td>OCSD</td><td>78.9 132.7 197.8 266.9 137.6</td><td></td><td></td><td>69.8</td><td></td><td></td><td>99.4 112.3 121.9</td><td>88.4</td></tr><tr><td>w/o weighting 82.1 139.8 208.2 277.4 144.8</td><td></td><td></td><td></td><td></td><td>69.6 101.6 117.3 124.8</td><td></td><td></td><td>90.0</td></tr></table>

![](images/6a55cf8b22b04e7bbf0589bdd71d6f861824854ae20d46ed737b65073ab2c56d.jpg)  
Fig. 3: Qualitative examples of 10-second predictions for the HiK dataset. We visualize two generated samples for the same input. We show the last observed frame on the left, followed by the final predicted frame from two distinct generated samples. These examples demonstrate the model’s ability to generate diverse predictions, as shown by the purple/yellow person walking to diferent locations.

## 4.3 Qualitative Results

Beyond quantitative metrics, we provide qualitative visualizations to demonstrate the fidelity, realism, and diversity of our generated motions. Specifically, a key advantage of our difusion framework is the ability to model a distribution of plausible futures. Fig. 3 illustrates this by showing two distinct 10-second predictions sampled from the same input. This capability to generate a wide range of realistic behaviors directly explains our model’s significantly higher UMWR (diversity) score compared to all baselines, which tend to produce only a single, deterministic-looking outcome. In Fig. 4 from HOI-M<sup>3</sup>, the person sitting on the sofa stands up correctly and continues moving, the walking person continues walking smoothly, and the standing person achieves the same pose as the ground truth at a slightly diferent but reasonable location. We provide additional qualitative results and discuss limitations in the supplementary material.

## 5 Conclusion

We tackled the challenging problem of forecasting multi-person human motion in complex scenes. We presented OCSD, a model that integrates individual dynamics, social interactions, and object influence into a unified generative framework. By modulating the denoising process with object cues at every timestep and integrating per-person social context, OCSD captures fine-grained relations while maintaining flexibility in group size, interaction complexity, and the diversity of possible futures. Our experiments on the HiK and HOI-M<sup>3</sup> benchmarks demonstrate substantial improvements over existing methods. Overall, our results show that holistic approaches like OCSD, which integrate both social and object information, are necessary for accurate motion forecasting in complex scenes.

![](images/f3d5d785b140b407a699bb53d780affc902d552dd6c041dbfab7b3b6b173d0ed.jpg)  
Fig. 4: 2-second prediction for the HOI-M<sup>3</sup> dataset. We show the last observed frame on the left. The next two panels display forecast poses after 1 second (middle) and 2 seconds (right). Purple skeletons represent our predictions, while gray skeletons represent ground-truth poses. The model correctly continues the stand up motion from the sofa and the walk motion of the other person.

## Acknowledgements

The work has been supported by the ERC Consolidator Grant FORHUE (101044724).

## References

1. Adeli, V., Ehsanpour, M., Reid, I., Niebles, J.C., Savarese, S., Adeli, E., Rezatofighi, H.: Tripod: Human trajectory and pose dynamics forecasting in the wild. In: IEEE/CVF International Conference on Computer Vision. pp. 13390– 13400 (2021)

2. Aksan, E., Kaufmann, M., Cao, P., Hilliges, O.: A spatio-temporal transformer for 3d human motion prediction. In: 2021 International Conference on 3D Vision (3DV). pp. 565–574. IEEE (2021)

3. Aliakbarian, S., Saleh, F., Petersson, L., Gould, S., Salzmann, M.: Contextually plausible and diverse 3d human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 11333–11342 (2021)

4. Aliakbarian, S., Saleh, F.S., Salzmann, M., Petersson, L., Gould, S.: A stochastic conditioning scheme for diverse human motion prediction. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5223–5232 (2020)

5. Barquero, G., Escalera, S., Palmero, C.: Belfusion: Latent difusion for behaviordriven human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 2317–2327 (2023)

6. Barsoum, E., Kender, J., Liu, Z.: Hp-gan: Probabilistic 3d human motion prediction via gan. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops. pp. 1418–1427 (2018)

7. Batkovic, I., Zanon, M., Lubbe, N., Falcone, P.: A computationally eficient model for pedestrian motion prediction. In: 2018 European Control Conference (ECC). pp. 374–379 (2018)

8. Bouazizi, A., Holzbock, A., Kressel, U., Dietmayer, K., Belagiannis, V.: Motionmixer: Mlp-based 3d human body pose forecasting. In: Thirty-First International Joint Conference on Artificial Intelligence. pp. 791–798 (2022)

9. Cai, Y., Huang, L., Wang, Y., Cham, T.J., Cai, J., Yuan, J., Liu, J., Yang, X., Zhu, Y., Shen, X., et al.: Learning progressive joint propagation for human motion prediction. In: European Conference on Computer Vision. pp. 226–242. Springer (2020)

10. Chen, L.H., Zhang, J., Li, Y., Pang, Y., Xia, X., Liu, T.: Humanmac: Masked motion completion for human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 9544–9555 (2023)

11. Curreli, C., Muhle, D., Saroha, A., Ye, Z., Marin, R., Cremers, D.: Nonisotropic gaussian difusion for realistic 3d human motion prediction. In: Computer Vision and Pattern Recognition Conference. pp. 1871–1882 (2025)

12. Dang, L., Nie, Y., Long, C., Zhang, Q., Li, G.: Diverse human motion prediction via gumbel-softmax sampling from an auxiliary space. In: 30th ACM International Conference on Multimedia. pp. 5162–5171 (2022)

13. Foka, A.F., Trahanias, P.E.: Probabilistic autonomous robot navigation in dynamic environments with human motion prediction. International Journal of Social Robotics 2(1), 79–94 (2010)

14. Fragkiadaki, K., Levine, S., Felsen, P., Malik, J.: Recurrent network models for human dynamics. In: IEEE International Conference on Computer Vision. pp. 4346–4354 (2015)

15. Fridovich-Keil, D., Bajcsy, A., Fisac, J.F., Herbert, S.L., Wang, S., Dragan, A.D., Tomlin, C.J.: Confidence-aware motion prediction for real-time collision avoidance1. The International Journal of Robotics Research 39(2-3), 250–265 (2020)

16. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3d human motions from text. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5152–5161 (2022)

17. Guo, W., Du, Y., Shen, X., Lepetit, V., Alameda-Pineda, X., Moreno-Noguer, F.: Back to mlp: A simple baseline for human motion prediction. In: IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 4809–4819 (2023)

18. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Advances in Neural Information Processing Systems. vol. 33, pp. 6840–6851 (2020)

19. Jain, A., Zamir, A.R., Savarese, S., Saxena, A.: Structural-rnn: Deep learning on spatio-temporal graphs. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 5308–5317 (2016)

20. Jeong, J., Park, D., Yoon, K.J.: Multi-agent long-term 3d human pose forecasting via interaction-aware trajectory conditioning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1617–1628 (2024)

21. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: 3rd International Conference on Learning Representations (2015)

22. Kundu, J.N., Gor, M., Babu, R.V.: Bihmp-gan: Bidirectional 3d human motion prediction gan. In: AAAI Conference on Artificial Intelligence. vol. 33, pp. 8553– 8560 (2019)

23. Li, K., Eifert, S., Shan, M., Gomez-Donoso, F., Worrall, S., Nebot, E.: Attentionalgcnn: Adaptive pedestrian trajectory prediction towards generic autonomous vehicle use cases. In: 2021 IEEE International Conference on Robotics and Automation (ICRA). pp. 14241–14247. IEEE (2021)

24. Li, K., Shan, M., Narula, K., Worrall, S., Nebot, E.: Socially aware crowd navigation with multimodal pedestrian trajectory prediction for autonomous vehicles. In:

2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). pp. 1–8. IEEE (2020)

25. Ma, T., Nie, Y., Long, C., Zhang, Q., Li, G.: Progressively generating better initial guesses towards next stages for high-quality human motion prediction. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6437– 6446 (2022)

26. Mao, W., Hartley, R.I., Salzmann, M., et al.: Contact-aware human motion forecasting. In: Advances in Neural Information Processing Systems. vol. 35, pp. 7356– 7367 (2022)

27. Mao, W., Liu, M., Salzmann, M.: History repeats itself: Human motion prediction via motion attention. In: European Conference on Computer Vision. pp. 474–489. Springer (2020)

28. Mao, W., Liu, M., Salzmann, M.: Generating smooth pose sequences for diverse human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 13309–13318 (2021)

29. Mao, W., Liu, M., Salzmann, M., Li, H.: Learning trajectory dependencies for human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 9489–9497 (2019)

30. Martinez, J., Black, M.J., Romero, J.: On human motion prediction using recurrent neural networks. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 2891–2900 (2017)

31. Mueller, F.B., Tanke, J., Gall, J.: Massively multi-person 3d human motion forecasting with scene context. In: European Conference on Computer Vision Workshops. pp. 130–147. Springer (2024)

32. Peng, X., Mao, S., Wu, Z.: Trajectory-aware body interaction transformer for multiperson pose forecasting. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17121–17130 (2023)

33. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: AAAI Conference on Artificial Intelligence. vol. 32, pp. 3942–3951 (2018)

34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763. PMLR (2021)

35. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 234–241. Springer (2015)

36. Saadatnejad, S., Rasekh, A., Mofayezi, M., Medghalchi, Y., Rajabzadeh, S., Mordan, T., Alahi, A.: A generic difusion-based approach for 3d human pose prediction in the wild. In: International Conference on Robotics and Automation (ICRA) (2023)

37. Sch¨oller, C., Aravantinos, V., Lay, F., Knoll, A.: What the constant velocity model can teach us about pedestrian motion prediction. IEEE Robotics and Automation Letters 5(2), 1696–1703 (2020)

38. Scofano, L., Sampieri, A., Schiele, E., De Matteis, E., Leal-Taix´e, L., Galasso, F.: Staged contact-aware global human motion forecasting. In: British Machine Vision Conference. pp. 589–594 (2023)

39. Sun, C., Sun, Y., Han, X., Yang, Z., Liu, J., Zhu, X., Yiu, S.M., Ma, Y.: Humof: Human motion forecasting in interactive social scenes. arXiv preprint arXiv:2506.03753 (2025)

40. Sun, J., Chowdhary, G.: Comusion: Towards consistent stochastic human motion prediction via motion difusion. In: European Conference on Computer Vision. pp. 18–36. Springer (2024)

41. Sun, K., Lan, C., Xing, J., Zeng, W., Liu, D., Wang, J.: Human pose estimation using global and local normalization. In: IEEE International Conference on Computer Vision. pp. 5599–5607 (2017)

42. Tanke, J., Kwon, O.H., Mueller, F.B., Doering, A., Gall, J.: Humans in kitchens: a dataset for multi-person human motion forecasting with scene context. In: Advances in Neural Information Processing Systems. vol. 36, pp. 10184–10196 (2023)

43. Tanke, J., Zaveri, C., Gall, J.: Intention-based long-term human motion anticipation. In: 2021 International Conference on 3D Vision (3DV). pp. 596–605. IEEE (2021)

44. Tanke, J., Zhang, L., Zhao, A., Tang, C., Cai, Y., Wang, L., Wu, P.C., Gall, J., Keskin, C.: Social difusion: Long-term multiple human motion anticipation. In: IEEE/CVF International Conference on Computer Vision. pp. 9601–9611 (October 2023)

45. Unhelkar, V.V., Lasota, P.A., Tyroller, Q., Buhai, R.D., Marceau, L., Deml, B., Shah, J.A.: Human-aware robotic assistant for collaborative assembly: Integrating human motion prediction with planning in time. IEEE Robotics and Automation Letters 3(3), 2394–2401 (2018)

46. Vendrow, E., Kumar, S., Adeli, E., Rezatofighi, H.: Somoformer: Multi-person pose forecasting with transformers. arXiv preprint arXiv:2208.14023 (2022)

47. Wang, J., Xu, H., Narasimhan, M., Wang, X.: Multi-person 3d motion prediction with multi-range transformers. In: Advances in Neural Information Processing Systems. vol. 34, pp. 6036–6049 (2021)

48. Wang, T., Song, Z., Lou, Z., Cui, Q., Zhang, L., Cheng, C., Wang, H., Tang, X., Li, H., Zhou, H.: Harmonizing stochasticity and determinism: Scene-responsive diverse human motion prediction. In: Advances in Neural Information Processing Systems. vol. 37, pp. 39784–39811 (2024)

49. Wei, M., Miaomiao, L., Mathieu, S., Hongdong, L.: Learning trajectory dependencies for human motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 9488–9496 (2019)

50. Xiao, P., Xie, Y., Xu, X., Chen, W., Zhang, H.: Multi-person pose forecasting with individual interaction perceptron and prior learning. In: European Conference on Computer Vision. pp. 402–419. Springer (2024)

51. Xing, C., Mao, W., Liu, M.: Scene-aware human motion forecasting via mutual distance prediction. In: European Conference on Computer Vision. pp. 128–144. Springer (2024)

52. Xu, G., Tao, J., Li, W., Duan, L.: Learning semantic latent directions for accurate and controllable human motion prediction. In: European Conference on Computer Vision. pp. 56–73. Springer (2024)

53. Xu, Q., Mao, W., Gong, J., Xu, C., Chen, S., Xie, W., Zhang, Y., Wang, Y.: Joint-relation transformer for multi-person motion prediction. In: IEEE/CVF International Conference on Computer Vision. pp. 9816–9826 (2023)

54. Xu, S., Wang, Y.X., Gui, L.Y.: Diverse human motion prediction guided by multilevel spatial-temporal anchors. In: European Conference on Computer Vision. pp. 251–269. Springer (2022)

55. Xu, S., Wang, Y.X., Gui, L.: Stochastic multi-person 3d motion forecasting. In: The Eleventh International Conference on Learning Representations (2023)

56. Yan, X., Rastogi, A., Villegas, R., Sunkavalli, K., Shechtman, E., Hadap, S., Yumer, E., Lee, H.: Mt-vae: Learning motion transformations to generate multimodal human dynamics. In: European Conference on Computer Vision. pp. 265–281 (2018)

57. Yuan, Y., Kitani, K.: Dlow: Diversifying latent flows for diverse human motion prediction. In: European Conference on Computer Vision. pp. 346–364. Springer (2020)

58. Zhang, J., Zhang, J., Song, Z., Shi, Z., Zhao, C., Shi, Y., Yu, J., Xu, L., Wang, J.: Hoi-m<sup>3</sup>: Capture multiple humans and objects interaction within contextual environment. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 516–526 (2024)

59. Zheng, Y., Yang, Y., Mo, K., Li, J., Yu, T., Liu, Y., Liu, C.K., Guibas, L.J.: Gimo: Gaze-informed human motion prediction in context. In: European Conference on Computer Vision. pp. 676–694. Springer (2022)

60. Zheng, Y., Yu, R., Sun, J.: Eficient multi-person motion prediction by lightweight spatial and temporal interactions. In: IEEE/CVF International Conference on Computer Vision. pp. 10844–10853 (2025)

# Multi-Person Human Motion Forecasting in Complex Scenes Supplementary Material

## A Datasets

Here, we provide the full details on the datasets used in our paper:

Humans in Kitchens [42]. HIK is a motion-capture dataset of people interacting in four kitchen environments with minimal scripting and rich everyday activities. Each recording contains a dynamic number of people with a maximum of 16, and 50 annotated objects visible at the same time, providing complex social interactions and dense scene context. Following the standard protocol, we train on three kitchens (A, B, and C) and evaluate on curated sequences from kitchen D, with a focus on transitional moments.

– HOI-M<sup>3</sup> [58]. $\mathrm { H O I - M ^ { 3 } }$ is a large-scale multi-human multi-object dataset captured in realistic indoor rooms. It contains 199 long sequences with up to 5 people and at least 5 objects per scene, and all humans and objects are tracked in 3D. We use object vertices as object representation and 49 livingroom sequences publicly available (1-51, except 43 and 45). We set aside 20% as the test set following [39].

For the HiK dataset, we adhere to the standard evaluation protocol, utilizing the oficial test split and evaluation script released by [42,31]. For the HOI-M<sup>3</sup> dataset, we evaluated performance on a test set split by using the data processing pipeline of [39]. During inference, we predicted future motions for all frames in these sequences using a sliding window approach with a stride of 1 second.

## B Metrics

The metrics used in our work are as follows:

Path Error. This metric evaluates the accuracy of the predicted global trajectory. We define it as the Euclidean distance $( L _ { 2 }$ norm) between the predicted 3D position of the root joint $\left( \hat { p } _ { r o o t } \right)$ and its corresponding groundtruth position $\left( { p _ { r o o t } } \right)$ . For a future time step t, the path error is:

$$
\mathcal { E } _ { p a t h } ( t ) = | | \hat { p } _ { r o o t } ^ { t } - p _ { r o o t } ^ { t } | | _ { 2 }\tag{S1}
$$

– Pose Error. This metric evaluates the accuracy of the predicted local body configuration, independent of global translation. We use the standard Mean Per Joint Position Error (MPJPE), which is the mean Euclidean distance across all J − 1 non-root joints after aligning the predicted and ground-truth poses at the root. For a future time step t, the pose error is:

$$
\mathcal { E } _ { p o s e } ( t ) = \frac { 1 } { J - 1 } \sum _ { j = 1 } ^ { J - 1 } | | ( \hat { p } _ { j } ^ { t } - \hat { p } _ { r o o t } ^ { t } ) - ( p _ { j } ^ { t } - p _ { r o o t } ^ { t } ) | | _ { 2 }\tag{S2}
$$

where $\hat { p } _ { j } ^ { t }$ and $p _ { j } ^ { t }$ are the predicted and ground-truth 3D positions of the j-th joint.

Normalized Directional Motion Similarity (NDMS). For long-term and stochastic forecasting, per-frame accuracy metrics are insuficient due to the fact that they penalize plausible predictions that deviate from the single ground truth sequence. Therefore, we evaluate the realism of generated motion using the Normalized Directional Motion Similarity (NDMS) score [43], which compares generated motion against a distribution of real-world examples. To evaluate a predicted motion sequence $\chi ,$ NDMS divides it into overlapping short segments, referred to as motion words. For each motion word, which is 8 frames long, the metric calculates a similarity score by finding its Nearest Neighbor (NN) in a reference database (D) composed of real motion snippets. This use of a reference database and NN search allows the measure to account for the multi-modal nature of human movement. The core similarity function Ψ compares two motion words (x and y) by evaluating two key factors for each joint j at time t:

1. Directional Similarity: The alignment of the joint velocity vectors $( \dot { \mathbf { x } } _ { t , j }$ and $\dot { \mathbf { y } } _ { t , j } )$ .

2. Magnitude Ratio: The ratio of the velocity magnitudes, ensuring similar speeds.

The resulting NDMS score provides a plausibility measure between 0 and 1, where higher values indicate more realistic motion. NDMS penalizes discontinuities which are highly visible to human observers, such as abrupt transitions between the observed and forecast motion.

– Unique Motion Word Ratio (UMWR). A good stochastic model should not only produce realistic motions but also avoid generating the same plausible motion repeatedly. For this, we use the Unique Motion Word Ratio (UMWR) introduced in [31]. The UMWR calculation leverages the same underlying mechanism as NDMS. It uses the same reference database $D _ { : }$ composed of motion words of 8 frames. For a single-person predicted sequence $\chi ,$ the UMWR is calculated as the ratio of unique Nearest Neighbors (NN) found in the reference set $D$ to the total number of motion words in the sequence. Formally, the metric is defined in [31] as

$$
\mathrm { U M W R } ( \chi ) = \frac { \left| \{ \mathrm { N N } ( \chi _ { 1 : \kappa } ) , \mathrm { N N } ( \chi _ { 2 : \kappa + 1 } ) , \dots \mathrm { N N } ( \chi _ { \left| \chi \right| + 1 - \kappa : \left| \chi \right| ) } \} \right| } { \left| \chi \right| + 1 - \kappa }\tag{S3}
$$

where $\kappa = 8$ is the length of the motion words and χ predicted motion of the selected person with last observed frame. A score closer to 1.0 signifies high diversity, as the model generates a wide variety of movements that map to many diferent real motion words. A low score indicates repetitive or frozen motion, where the entire sequence maps to only a few unique motion words from D.

## C Additional Implementation Details

We trained all models on a single RTX-4090 GPU. To accommodate memory constraints during training, we adjusted the batch size according to the prediction horizon: we used a batch size of 64 for short-term (2-second) prediction models and a batch size of 16 for long-term (10-second) prediction models, where we also adjust the learning rate to $1 \times 1 0 ^ { - 4 }$ to accommodate the reduced batch size. Moreover, in the long-term setting, each temporal feature sequence at the bottleneck $\left( h _ { 1 } \right)$ is passed through multi-head self-attention with pre-norm and a residual connection to capture long-range temporal dependencies within each local context. Each residual block uses two successive dilated temporal convolutions (d=2 and d=4) instead of unit dilation to extend the temporal receptive field without increasing parameter count. We employ a dynamic sampling strategy. Rather than using fixed, strided subsequences, each training sample is a clip randomly selected from a valid start position within any sequence. We use 1000 denoising steps. For inference on the HOI-M<sup>3</sup> dataset, our approach requires as SAST 2s. To put this into context, HUMOF reports 43ms.

## D Additional Qualitative Results

In this section, we show further qualitative results of our model for both datasets. We use SAST [31] for comparison, as the code for HUMOF [39] was not publicly available at the time of preparing the manuscript.

## D.1 HiK dataset: 2-second prediction

In Fig. S1, we visualize the predicted motions against ground truth (GT) for a 2- second prediction window. Our model simultaneously predicts the future motions of all individuals in the scene. As demonstrated in the figure, our method generates highly accurate future poses. Even for the purple/yellow skeleton, which moves most in the scene, the predicted motion aligns closely with the ground truth (gray skeleton), making them dificult to distinguish even at frame 74. This validates the model’s ability to predict future motion precisely. In addition, examples for the whiteboard and armchair activities are provided in Fig. S2.

Comparison with SAST (2-second). In Fig. S3, we compare our method (OCSD) against the SAST baseline. Both models are trained to predict a 2- second horizon given a 1-second input. While our model correctly infers the walking trajectory of the purple/yellow person, SAST predicts that the person will stay close to the last observed position.

## D.2 HOI-M<sup>3</sup> dataset: 2-second prediction

In column (c) of Fig. S4, the central person continues to walk. The predicted motion follows with the correct poses but a little bit slower. In the mean time, the interaction of the left person with the chair continues closely to the real motion. The model correctly identifies the afordance for interaction and also generates a valid walking path.

![](images/c19f498cb2aeea547c4cc21deee814db1392ec4de13bf0b5f060dc5da634df4e.jpg)  
(a) Frame 24 (t = 1s)

![](images/23a76365dccc4f442846362399c1dd555ffb30469db477ff05e27c665fdfd61c.jpg)  
(b) Frame 74 (t = 3s)

![](images/16ad617943f5984ced58c12b8b061fe61640e5f5cb68ee4d38db8cc5b8ca62fb.jpg)  
(c) Bird’s-eye  
Fig. S1: Qualitative comparison on the HiK dataset (2-second prediction). We compare our predictions to the ground truth (GT). Red/Blue: Past GT poses. Green/Yellow: Predicted future poses. Gray: Future GT for comparison. Purple/Yellow: Predicted future and past GT poses of a person with large motion. The long gray and purple lines show the GT and predicted trajectory, respectively. OCSD’s predictions overlap significantly with the GT (Gray), indicating high accuracy.

## D.3 HiK dataset: 10-second prediction

In Fig. S5, we examine the stochastic nature of our model by generating multiple predictions for a single 10-second scenario. While column (a) shows the initial state, columns (b) and (c) display two diferent samples generated by our model for frame 299. As observed, the model generates diverse but plausible futures. The purple/yellow person walks to diferent spatial locations in sample 1 versus sample 2. This confirms that OCSD is not collapsing to a single deterministic mean, but is capable of modeling the natural variance in human motion over long horizons.

Comparison with SAST (10-second). We further compare long-term prediction capabilities in Fig. S6. The SAST model (row b) forecasts that the purple/yellow person sits down at a location where there is no chair. In contrast, OCSD (row a) maintains motion plausibility throughout the 10-second prediction window.

## E Limitations

The first limitation regards moving objects in the prediction interval. Our model uses the last observed positions of the objects for that interval. This avoids providing the model with oracle knowledge of future object positions, but it means that predictions cannot adapt to objects that move in the future. For example in Fig. S7, the positions of the bag, dumbbell and book change during the prediction interval, and the predicted motions remain anchored to their last observed positions. This causes predictions to deviate from the actual human-object interaction, and is observed in a subset of HOI-M<sup>3</sup>. HiK is largely unafected since their objects are mostly static. Addressing this would require jointly predicting the future positions of the objects alongside human motion, a promising direction for future work.

![](images/fbd11237c8aeede75273804bb25ba11061b0bac4edb88d61c7202c58759eedd2.jpg)  
Fig. S2: Object interactions on the HiK dataset (2-second prediction). Each row shows the last observed frame (Frame 24), 1-second prediction (Frame 49), and 2-second prediction (Frame 74). Visualization scheme follows Fig. S1. Row (a) shows the writing activity with interaction with the whiteboard, while row (b) shows the sitting down activity with interaction with the armchair.

A second limitation concerns the semantic object representation. Our onehot encoding for object category information is efective for the closed object vocabularies of HiK and HOI-M<sup>3</sup>, but it is not suitable for open-vocabulary generalization. Preliminary experiments with CLIP [34] text embeddings of object names did not improve our approach. As future work, vision-language embeddings could be integrated into the model for more generalizable object representations.

## F Additional Experiment Results

We perform a range of additional ablation studies to further justify the components of our method, report additional metrics focused on motion quality, evaluate the efect of the number of samples on the diversity, and report bestof-K results instead of the average score.

![](images/2981e13c1671ea40c8cbe1d7ad1932729faf448166255d4f13f0aa1d1a33fa48.jpg)  
Fig. S3: Qualitative comparison with SAST (HiK dataset, 2-second prediction). Each row shows the last observed frame (Frame 24), 1-second prediction (Frame 49), and 2-second prediction (Frame 74). Visualization scheme follows Fig. S1. SAST (row b) does not forecast that the purple person walks from the left group to the right group, while OCSD (row a) successfully predicts the walking motion.

## F.1 Social encoder design

We validate our social encoder design by showing that our mean pooling outperforms cross-attention for encoding social information in Tab. S1a, and that injecting this into the bottleneck gives better results than doing so throughout the network in Tab. S1b.

Table S1: Social Encoder ablation on HiK. Replacing mean pooling with cross-attention or integrating social context at multiple U-Net layers instead of the bottleneck increases mean path error (mm) and pose error (mm).  
(a) Mean pooling vs. Cross-attention
<table><tr><td colspan="3">Method Path Error↓ Pose Error↓</td></tr><tr><td>Mean pooling</td><td>137.6</td><td>88.4</td></tr><tr><td>Cross-attn</td><td>142.3</td><td>91.0</td></tr></table>

(b) Bottleneck vs. Multi-layer
<table><tr><td colspan="3">Social context Path Error↓ Pose Error↓</td></tr><tr><td>Bottleneck</td><td>137.6</td><td>88.4</td></tr><tr><td>Multiple layers</td><td>140.2</td><td>89.9</td></tr></table>

## F.2 Context contribution

While both social and object context have a clear positive efect on the path error, the averaged results in the main paper do not show the full picture for the pose error. To demonstrate this, we report the results of Tab. 4 for three representative activities in Tab. S2. We find that for the walking activity, where context is important for the path but not necessarily for the pose, including the context slightly increases the pose error. In contrast, for the more static, object-centered social activities, such as interacting with a cofee machine and the surrounding people, the inclusion of this context leads to better predictions. For the path error, context is always helpful.

![](images/c88ae148d661f6f93a9472afaaa1d91775b1613c2f8ca239249dab2ac4091793.jpg)  
(a) Frame 29 (t = 1)

![](images/8e627a8e3fdc0eb37d344e78c7e9a0de6acf55b2c7a8c2eab2abe7fbf259b4e9.jpg)  
(b) Frame 59 (t = 2s)

![](images/19c30a76a76774222eb2c706ea2a6c65116281f6a7dc9dc4dddb536c5e6e0b39.jpg)  
(c) Frame 89 (t = 3s)

Fig. S4: 2-second prediction on the HOI-M<sup>3</sup> dataset. Purple skeletons represent our predictions, while gray skeletons represent GT. The model accurately captures the pose dynamics over the 2-second horizon.  
![](images/7932b38e67bdbb8c9f24ab61aed18d42cd162c5c9ab940e9489d2f4ac150a40a.jpg)  
(a) Last Ground Truth

![](images/da260e4c05413a80178653dd46f837843ea30625a4555ad1422024ef435d9ae8.jpg)  
(b) Prediction Sample 1

![](images/18e3cace72d264801bfb74e7718815f780a8d7d5e61e6dc32573e3965b95da5b.jpg)  
(c) Prediction Sample 2  
Fig. S5: Qualitative comparison on the HiK dataset (10-second prediction). We visualize two stochastic samples for the same scene. (a) Last observed GT at t = 2 s (frame 49). (b - c) Final predicted frame at t = 12 s (frame 299) from two distinct generated samples. Visualization scheme follows Fig. S1. The purple/yellow person walks to diferent locations in (b) and (c). This demonstrates the model’s ability to generate diverse predictions.

Table S2: Context efect per activity on HiK. Context improves mean path error for all three representative activities. For pose error, benefits provided by context are only evident in the object-centered social activities sink and cofee.
<table><tr><td rowspan="2">Method</td><td colspan="2">Path Error (mm)</td><td colspan="2">Pose Error (mm) ↓</td></tr><tr><td>walking</td><td>sink coffee</td><td>walking sink coffee</td><td></td></tr><tr><td>OCSD</td><td>306.0</td><td>57.3 73.6</td><td>100.5</td><td>57.3 58.3</td></tr><tr><td>self + social</td><td>328.3</td><td>78.3 94.0</td><td>98.6</td><td>65.0 62.2</td></tr><tr><td>self + object</td><td>319.8</td><td>68.4 84.9</td><td>99.5</td><td>65.8 62.2</td></tr><tr><td>only self</td><td>334.1</td><td>81.4 96.0</td><td>98.3</td><td>63.8 62.4</td></tr></table>

![](images/f02d45ce0b31aa0bbad5027b4456ff5f87414274d706e691dd634c1dd3d727da.jpg)  
Fig. S6: Qualitative comparison with SAST (HiK dataset, 10-second prediction). Each row shows the last observed GT at Frame 49 (t = 2 sec), prediction at Frame 174 (t = 7 sec), and prediction at Frame 299 (t = 12 sec). Visualization scheme follows Fig. S1. For SAST, the purple/yellow person hallucinates a chair and sits down at a location where there is no chair (row b). Our model produces plausible motion for the persons (row a) over the full 10-second prediction horizon.

## F.3 U-Net vs. DiT

We exchange our U-Net backbone for a DiT, keeping the conditioning and training protocol identical. Tab. S3 shows that DiTs overfit on the relatively small multi-person motion forecasting datasets with object information and U-Nets remain the better choice.

Table S3: U-Net vs. DiT backbone on HiK. Difusion transformer performs worse than U-Net.
<table><tr><td rowspan="2">Backbone</td><td colspan="5">Path Error (mm) ↓</td><td colspan="5">Pose Error (mm)</td></tr><tr><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s</td><td>mean</td><td>0.5s</td><td>1.0s</td><td>1.5s</td><td></td><td>2.0s mean</td></tr><tr><td>U-Net</td><td></td><td>78.9 132.7 197.8 266.9 137.6</td><td></td><td></td><td></td><td></td><td>69.8 99.4 112.3 121.9 88.4</td><td></td><td></td><td></td></tr><tr><td>DiT</td><td></td><td>88.5 148.6 223.6 294.3 154.0</td><td></td><td></td><td></td><td></td><td>74.7 103.2 121.3 133.7</td><td></td><td></td><td>94.5</td></tr></table>

## F.4 Additional motion quality metrics

We report four additional metrics on HOI-M<sup>3</sup> in Table S4: Foot sliding (mm/s), Interpenetration depth (mm), Interpenetration rate, and the Mean Per Joint Velocity Error (MPJVE) (mm/s). We compare our results against ground truth (GT) motion. Foot sliding of predictions is within 5% of the GT. Interpenetration statistics are likewise close to GT. Note that ground-truth motion has penetrations as well due to approximations of the object surfaces. The results show that OCSD does not sufer from strong motion quality artifacts. This is also evident from the NDMS scores, which explicitly compares the velocity-magnitude ratios against real motion, reported in the paper.

![](images/22c4b66541540acc0c32f48e5dc535f7ac11f2633e1798b82f114920ccef7197.jpg)  
(a) Frame 29 (t = 1)

![](images/8ee1caa3f6faaee0da20ad1209994e26228c57b56ebab502010915e795f7f344.jpg)  
(b) Frame 59 (t = 2s)

![](images/7d563b4269751efd7abb6a86b9395995c9ed086c8b3ebe56727d730ef9b08f01.jpg)  
(c) Frame 89 (t = 3s)  
Fig. S7: Limitation regarding moving objects. Purple skeletons represent our predictions, while gray skeletons represent GT. Since the future position of the objects are unknown for the model during the prediction interval, predictions deviate from the actual human-object interaction. This is a 2-second prediction on the HOI-M<sup>3</sup> dataset.

Table S4: Additional motion quality metrics on HOI-M<sup>3</sup>. OCSD closely matches ground truth foot sliding and penetration statistics.
<table><tr><td></td><td>Method Foot sliding (mm/s) Interp.depth (mm)</td><td></td><td></td><td>Interp.rate MPJVE (mm/s)</td></tr><tr><td>GT</td><td>219.3</td><td>38.5</td><td>0.0338</td><td>N/A</td></tr><tr><td>OCSD</td><td>208.2</td><td>42.3</td><td>0.0343</td><td>330.2</td></tr></table>

## F.5 Number of evaluation samples

Tab. S5 shows that increasing the number of samples used for evaluation from 4 to 16 increases the UMWR, especially for 2-second predictions. This confirms that diversity is not saturated at four samples.

Table S5: Efect of sample count on diversity (UMWR) on HiK. Diversity scores increase with more samples.
<table><tr><td rowspan="2">Samples</td><td colspan="2">UMWR↑</td></tr><tr><td>2s 4s</td><td>6s 8s 10s</td></tr><tr><td>4</td><td>0.470.450.450.44 0.44</td></tr><tr><td>16</td><td>0.56 0.480.46 0.46 0.45</td></tr></table>

## F.6 Best-of-K analysis

Tab. S6 reports best-of-K path and pose errors on HiK for $\mathrm { ~ K ~ } \in \ \{ 1 , 5 , 1 0 , 2 0 \}$ Results are already good for K = 1 compared to the baselines, but they get even better for higher values. This supports the previous results, showing that OCSD produces diverse, yet consistently high-quality predictions.

<table><tr><td rowspan="2">K</td><td colspan="4">Path Error (mm)</td><td rowspan="2">↓</td><td colspan="4">Pose Error (mm) ↓</td></tr><tr><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s mean</td><td>0.5s</td><td>1.0s</td><td>1.5s</td><td>2.0s mean</td></tr><tr><td>1</td><td>79.0</td><td>131.0 200.8</td><td></td><td>270.9</td><td>139.0</td><td>69.5</td><td>101.8</td><td>115.0</td><td>121.8</td><td>89.0</td></tr><tr><td>5</td><td>57.5</td><td></td><td></td><td>90.7 134.8 181.4</td><td>95.5</td><td>54.2</td><td>75.0</td><td>87.5</td><td>93.1</td><td>68.3</td></tr><tr><td></td><td>10 51.8</td><td></td><td>79.2 116.7 156.1</td><td></td><td>83.5</td><td>50.1</td><td>68.3</td><td>79.8</td><td>85.1</td><td>62.5</td></tr><tr><td></td><td>20 46.5</td><td></td><td>69.3 103.0 137.4</td><td></td><td>74.0</td><td>47.1</td><td>63.3</td><td>74.0</td><td>79.4</td><td>58.3</td></tr></table>

Table S6: Best-of-K analysis on HiK. Accuracy improves steadily with K. OCSD produces diverse, yet consistently high-quality predictions.