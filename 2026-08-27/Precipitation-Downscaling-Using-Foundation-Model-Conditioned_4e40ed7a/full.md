# Precipitation Downscaling Using Foundation Model-Conditioned Difusion

Victor Nascimento Ribeiro<sup>1</sup>, Jorge Guevara<sup>1</sup>, Jorge Sebasti´an Moraga<sup>2</sup>, Chris Lucas<sup>2</sup>, Natalie Lord<sup>2</sup>, Andrew Taylor<sup>3</sup>, Edward Lockhart<sup>3</sup>, Will Trojak<sup>1</sup>, Johannes Schmude<sup>1</sup>, and Anne Jones<sup>∗1</sup>

<sup>1</sup>IBM Research <sup>2</sup>Fathom <sup>3</sup>STFC Hartree Centre

August 27, 2026

## Abstract

High-resolution precipitation fields are essential for hydrological impact assessment, yet global climate model outputs are too coarse and biased to be used directly. AI-based statistical downscaling with difusion models ofers a promising approach, but the mechanism by which largescale atmospheric predictors condition the generative process remains largely unexplored. We investigate three conditioning strategies for a denoising difusion probabilistic model applied to daily precipitation downscaling: channel concatenation of upsampled coarse predictors, crossattention conditioning with a learned convolutional encoder, and cross-attention conditioning with the frozen encoder of the Prithvi WxC weather foundation model pretrained on global reanalysis. All strategies are evaluated against an unconditioned baseline under identical training and inference conditions, using a comprehensive set of probabilistic, distributional, spectral, and extreme-event metrics for the Colorado River Basin.

Concatenation conditioning achieves the lowest pointwise CRPS and MSE, but tends to produce oversmoothed fields that suppress high-intensity events. In contrast, cross-attention conditioning provides substantially better precipitation distribution realism and modest improvements in spectral fidelity. Improvements are greatest for extremes: the Prithvi-WxC-conditioned model retains over half of > 100mm/day events, although estimates are uncertain due to limited samples. When trained on the full dataset, the learned convolutional model performs similarly to the foundation model-conditioned approach while requiring lower computational resources. However, the Prithvi-WxC-conditioned model achieves comparable performance with only five years of training data, demonstrating improved data eficiency. These results indicate that crossattention conditioning ofers advantages over simple concatenation for probabilistic precipitation downscaling, and that pretrained foundation model representations may ofer further advantages in data-limited settings.

## 1 Introduction

Future projections of precipitation time series are needed for risk assessment of climate change impacts such as flooding through simulation with hydrological models. Due to the coarse resolution and biases in the limited representation of small scale efects and extremes, precipitation outputs of GCMs are not suitable for flood modeling. Instead, downscaling methods are employed to derive high-resolution precipitation projections from low-resolution GCM inputs. Statistical downscaling, which is based on observed relationships between large scale climate variables and local climate, has long been established as a computationally eficient and efective downscaling approach, particularly when large ensembles are needed for impact modeling [Maraun et al., 2010].

Machine learning (ML) has a long history of being applied to this task [Vandal et al., 2019, Guti´errez et al., 2019]. More recently, a number of authors have explored Deep Learning (DL) applied to downscaling climate projections. Although these methods often ofer performance improvements over previous approaches, they are typically deterministic and tend to produce oversmoothed predictions, limiting their ability to represent extremes, uncertainty, and fine-scale spatial variability [Adewoyin et al., 2021]. Rampal et al. [2024] provide a review and highlight difusion as a promising new technique for generating high quality downscaling predictions.

Difusion models in AI emerged as an alternative to earlier image generation models such as Generative Adversarial Networks (GANs), which were successful but often dificult to train and prone to instability and ”mode collapse” [Gulrajani et al., 2017]. Difusion has been used in previous downscaling studies with promising results. Mardani et al. [2025] used residual corrective difusion to downscale ERA5 reanalysis to 2km radar-assimilated regional weather simulations. Tomasi et al. [2025] also used residual difusion to downscale ERA5 to a high-resolution reanalysis fields of 2m temperature and 10m horizontal wind for Italy and found the difusion approach outperformed baselines including a UNet and GAN. Brazidec et al. [2026] also used residual difusion along with a graph transformer encoder for global weather forecast downscaling from 100km to 30km. Lopez-Gomez et al. [2025] used a probabilistic difusion model to downscale 45km resolution regional climate projections to 9km for an area in the US, for multiple variables, including precipitation. Recent work has shown that difusion models consistently achieve better performance than traditional CNN and GAN approaches for downscaling precipitation, particularly in the capture of extremes [Saoulis et al., 2025].

Unlike classical unconditioned difusion models [Ho et al., 2020], climate downscaling requires learning a conditional distribution p(y|x), where y is the high-resolution target and x represents coarse-resolution atmospheric and physiographic predictors. The choice of conditioning mechanism and how the model incorporates predictor information can significantly impact downscaling performance. These mechanisms are well established in the broader machine learning literature; for example, Rombach et al. [2021] demonstrated the efectiveness of cross-attention for incorporating information such as text prompts into generative models, and Rozet and Louppe [2023] used conditioning via guidance applied at inference time. However, for climate downscaling, choice of conditioning mechanism is to date relatively unexplored.

Recent advances in deep learning have introduced AI models as emulators for weather forecasting. Koldunov et al. [2024] demonstrated that AI NWP models trained on ERA5 data could be used to reproduce the correct features as that scale when provided with lower resolution inputs, thereby acting as a downscaling and bias correction tool. Pretrained foundation models that unify multiple downstream tasks, including downscaling, have been proposed. Models such as ClimaX [Nguyen et al., 2023a] and Prithvi WxC [Schmude et al., 2024] demonstrate that transformer-based architectures pretrained on global reanalysis or climate projections can be efectively fine-tuned for high-resolution downscaling. Similar capabilities have been shown by ORBIT-2 [Wang et al., 2025] and AtmoRep [Lessig et al., 2023], which leverage specialized architectural adaptations and multimodal training, respectively. Beyond deterministic approaches, generative foundation models like Climate in a Bottle (cBottle) [Brenowitz et al., 2025] explicitly address probabilistic reconstruction and physical consistency at kilometer scales. These developments are supported by benchmarking frameworks such as ClimateLearn [Nguyen et al., 2023b], which facilitate standardized evaluation. Together, these works signal a shift toward general-purpose foundation models for downscaling, while highlighting ongoing challenges in physical consistency and multivariate coupling.

Key challenges remain for weather foundation models, particularly in downscaling applications.

Most models are still trained at coarse resolution, with high-resolution refinement handled externally and without standardized evaluation benchmarks. Physical inconsistencies, including violations of conservation laws and mismatches across variables, are frequently observed in super-resolved outputs. Extremes remain dificult to capture, and generalization across data sources (e.g., reanalysis to climate simulations) often requires substantial adaptation. High computational costs limit reproducibility, while limited interpretability, incomplete multi-modal integration, and restricted model availability continue to constrain broader operational use.

Despite recent progress in difusion and foundation model approaches for climate downscaling, several important gaps remain. First, most difusion-based downscaling studies rely on simple concatenation of conditioning variables, providing limited control over how large-scale atmospheric information influences fine-scale stochastic generation. Second, while weather foundation models have shown promise for deterministic downscaling, their potential as conditioning encoders for probabilistic generative models has not been well explored. Third, the computational and dataeficiency trade-ofs associated with diferent conditioning strategies remain poorly quantified.

In this work, we address these limitations by evaluating alternative conditioning mechanisms for difusion-based precipitation downscaling. Our contributions are threefold:

1. We provide a controlled comparison of conditioning strategies, contrasting channel concatenation with cross-attention conditioning to assess their impacts on spatial structure, distributional realism, and extreme event representation.

2. We introduce the use of a pretrained weather foundation model (Prithvi WxC) as a conditioned encoder within a probabilistic difusion framework, demonstrating how pretrained representations can enhance generative performance.

3. We quantitatively analyse how diferent conditioning approaches afect the amount of training data required to achieve comparable performance.

To enable this, we evaluate channel concatenation-based conditioning against cross-attention conditioning using both a task-specific learnable convolutional encoder and a pretrained weather foundation model (Prithvi WxC), under identical training and inference settings. Performance is assessed using metrics spanning distributional accuracy, multiscale spatial structure, probabilistic calibration, extreme-event behavior, and data eficiency.

## 2 Dataset

In this work, we focus on downscaling daily total precipitation in the Colorado River Basin, USA. In this context, difusion models are trained using paired datasets consisting of (1) a high-resolution target variable (e.g. precipitation) from observations or reanalysis, and (2) a set of low-resolution conditioning climate variables (typically large scale meteorological variables).

The target variable was obtained from ERA5-Land [Mu˜noz Sabater, 2019] at 0.1°, extracted from the Copernicus Data Store (CDS) [Copernicus Climate Change Service (C3S), 2019]. The dynamic conditioning variables used for downscaling were taken from the 0.25° ERA5 reanalysis [Hersbach et al., 2023], also extracted from the CDS [Copernicus Climate Change Service (C3S), 2023]. Consistent with established methods for statistical downscaling, precipitation was not included as a conditioning variable. Instead, we used large-scale atmospheric state variables associated with precipitation processes. Specifically, we used five vertical variables at three pressure levels and six surface variables, as summarized in Table 1. All variables were extracted for the domain of 30°N to 46°N and 116°W to 100°W and aggregated (mean or total, as appropriate for the variable) to daily values. Predictor variables were coarse-grained to a 1° grid to match the typical spatial resolution used in climate projections.

In addition to the dynamic variables we included five high-resolution static variables in the conditioning dataset. Three of these: cl, lsm and z were taken directly from ERA5-land at 0.1° resolution. The remaining variables, sdor and slor, which are not available in the ERA5-Land dataset, were derived from a 0.01° digital elevation model based on Copernicus GLO-90 DEM [European Space Agency, 2019] and interpolated to the same ERA5-Land grid.

Table 1 ERA5 atmospheric, surface, and static variables used to condition the difusion precipitation downscaling models. Vertical atmospheric variables are provided at three pressure levels (500, 700, and 850 hPa). Static physiographic variables are provided at their native resolution, while surface and vertical variables are coarse-resolution.
<table><tr><td>Variable</td><td>Description</td></tr><tr><td colspan="2">Vertical Atmospheric Variables at {500, 700, 850} hPa</td></tr><tr><td>q</td><td>Specific Humidity</td></tr><tr><td>t</td><td>Temperature</td></tr><tr><td>u</td><td>Zonal Wind Speed (West-East)</td></tr><tr><td>V</td><td>Meridional Wind Speed (South-North)</td></tr><tr><td>W</td><td>Vertical Velocity Geopotential</td></tr><tr><td colspan="2">Z Surface Variables</td></tr><tr><td>cape sp</td><td>Convective Available Potential Energy Surface Pressure</td></tr><tr><td>ssrd t2m</td><td>Surface Solar Radiation Downward 2 m Temperature</td></tr><tr><td>u10</td><td>10 m Zonal Wind Speed</td></tr><tr><td>v10</td><td>10 m Meridional Wind Speed</td></tr><tr><td colspan="2">Static Physiographic Variables</td></tr><tr><td>cl</td><td>Lake Cover</td></tr><tr><td>lsm</td><td>Land Sea Mask</td></tr><tr><td>sdor</td><td>Standard Deviation of Subgrid-Scale Orography</td></tr><tr><td>slor</td><td>Slope of Subgrid-Scale Orography</td></tr><tr><td>Z</td><td>Surface Geopotential</td></tr></table>

## 3 Methods

This section describes the stochastic precipitation downscaling framework used in this study. We first introduce the conditional difusion formulation and the base denoising architecture shared across all experiments. We then describe three alternative strategies for conditioning the difusion model on coarse-resolution atmospheric and high-resolution static physiographic predictors: (1) channel concatenation-based conditioning (CC); (2) cross-attention conditioning using a taskspecific learnable convolutional encoder (CA-CE); and (3) cross-attention conditioning with the frozen encoder of a pre-trained weather foundation model, Prithvi WxC (CA-PWC). These conditioning approaches represent the main experiments explored in this study. Finally, we describe the experimental setup used to evaluate the models.

## 3.1 Conditioning Difusion Models for Climate Downscaling

Classical difusion models define an unconditional generative process for a target high-resolution field y by progressively adding noise to training data and learning to reverse this process [Ho et al., 2020]. However, in climate downscaling, we aim to learn a conditional distribution p(y|x), where x represents coarse-resolution predictors such as atmospheric variables and static physiological information. The conditioning process transforms the task from unconditional synthesis to the generation of high-resolution states that are physically-plausible weather samples [Bassetti et al., 2024].

![](images/ea96b06c06278cf9ee22e0819b70f9f7e4168194e168609d9ab8b46633359626.jpg)  
Figure 1 Schematic of concatenation-based conditioning (CC) for the difusion precipitation downscaling model. Surface, vertical, and static physiographic predictor fields are concatenated with the noise input before being passed to the UNet denoising model, which outputs the downscaled total precipitation field. Surface and vertical coarse-resolution predictors must be interpolated to the high-resolution input grid.

We adopt the EDM formulation of difusion models [Karras et al., 2022], in which the forward process is defined by a continuous noise scale, σ rather than a fixed discrete timestep schedule. Training data are perturbed with noise sampled across a wide range of magnitudes, and the model is trained to recover the clean signal at each noise level. The reverse process generates the highresolution target variable: the model is initialized with noise along with the conditioning inputs and progressively denoises across a discretized sequence of noise levels to produce the target variable conditioned on the climate information.

We employ a UNet denoiser adapted for conditional generation, following the EDM preconditioning and loss-weighting scheme. This architecture integrates fine-scale local features with large-scale global context through its encoder–decoder structure and skip connections. Following Karras et al. [2022], the noise level, σ is supplied to the network as a continuous conditioning input rather than a discrete timestep index, allowing the model to modulate its denoising behavior smoothly across the full range of noise magnitudes used during training (see Section 3.5).

## 3.2 Concatenation-based Conditioning

Concatenation provides a straightforward and computationally eficient mechanism for incorporating predictor fields into difusion models, as illustrated in Fig. 1. In this approach, the conditioning variables are appended as additional channels to the noise input, a strategy widely used in conditional generative models and early difusion frameworks [Saharia et al., 2023, Ho et al., 2020] and widely used in climate downscaling [Addison et al., 2024]. This allows the model to directly access relevant atmospheric information and spatially detailed surface characteristics during the denoising process.

Although concatenation is often efective, it incorporates information implicitly by treating predictor fields as additional input channels, and the model must therefore learn how to extract and use this information from the raw inputs. As more variables are included, this approach can also become ineficient. Moreover, because the difusion network operates on high-resolution inputs, this strategy requires upsampling the low-resolution predictors (Surface and Vertical variables) using interpolation before concatenation [Srivastava et al., 2024].

To incorporate coarse-resolution atmospheric and high-resolution physiographic predictors into the difusion process, the model must be conditioned in a way that preserves large-scale context while maintaining fine-scale stochastic variability. Diferent conditioning mechanisms impose diferent inductive biases on how predictor information influences the denoising dynamics, with important implications for spatial structure, extremes, and uncertainty representations in downscaled precipitation.

## 3.3 Cross-Attention Conditioning

Cross-attention conditioning provides a more expressive mechanism for integrating coarse-resolution predictors into difusion models than simple concatenation [Yang et al., 2024]. This approach allows the denoising network to dynamically query conditioning fields and selectively attend to the most relevant features at each denoising step, providing an efective modeling of the spatial and dynamical relationships critical for downscaling [Zhao et al., 2023]. In contrast to concatenationbased conditioning, this approach removes the need for naive upsampling of coarse predictors. Although cross-attention has proven highly efective in computer vision applications such as textto-image generation [Po et al., 2024, Saharia et al., 2022], its use in climate downscaling remains relatively underexplored and is not yet a standard conditioning strategy.

In this conditioning approach, coarse-resolution predictors must first be mapped into a latent embedding space [Rombach et al., 2021]. The cross-attention mechanism computes a contextaware representation that injects conditioning information directly into the latent feature maps of the denoiser model. Specifically, in cross-attention conditioning intermediate UNet feature maps act as queries, while embeddings derived from coarse-resolution predictors (or other conditioning information) provide the keys and values.

Formally, let $F \in \mathbb { R } ^ { N \times d }$ denote a UNet feature map flattened over spatial dimensions, and let $C \in \mathbb { R } ^ { M \times d _ { c } }$ denote the conditioning embeddings. Linear projections are applied to obtain the queries, keys, and values:

$$
Q = F W _ { Q } , \quad K = C W _ { K } , \quad V = C W _ { V }\tag{1}
$$

where $W _ { Q } \in \mathbb { R } ^ { d \times d _ { k } } , \ W _ { K } \ \in \ \mathbb { R } ^ { d _ { c } \times d _ { k } }$ , and $W _ { V } \in \mathbb { R } ^ { d _ { c } \times d _ { v } }$ are learnable projection matrices. The cross-attention operation is then defined as

$$
\mathrm { C r o s s A t t e n t i o n } ( Q , K , V ) = \mathrm { S o f t m a x } \bigg ( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \bigg ) V\tag{2}
$$

When predictors are provided as gridded fields or images, the embeddings can be obtained using an encoder model that produces a compact representation suitable for attention interactions. The resulting predictor embeddings are supplied to cross-attention blocks throughout the UNet architecture after each convolutional block at diferent spatial resolutions, allowing the denoising model to use the conditioning information to guide the generation of both large-scale contextual information and fine-scale features during the denoising process.

Because the encoder model operates on low-resolution inputs, we therefore retain static physiographic variables at their native high resolution and incorporate them via concatenation-based conditioning. In order to apply the same cross-attention mechanism to these static variables, it would require spatial coarsening, leading to a substantial loss of fine-scale information that is critical for downscaling. A high-level schematic of this conditioning strategy is shown in Figure 2.

## 3.4 Conditioning using Prithvi WxC Weather Foundation Model

Recent advances in weather and climate AI led to the development of weather foundation models trained on massive Earth system datasets. Among these, the Prithvi WxC model [Schmude et al., 2024] provides a high-capacity spatiotemporal encoder specifically optimized for atmospheric fields. It was pretrained on Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2) dataset [Gelaro et al., 2017], a widely-used reanalysis dataset from NASA providing a broad range of global atmospheric and land surface spanning multiple decades.

![](images/92e260f34de1f53a4609759b8163d099830738a1818615e746338c79d86532b6.jpg)

Figure 2 Schematic of cross-attention conditioning (CA) integrated into the UNet difusion denoiser. Static physiographic variables are incorporated via channel concatenation at their native high resolution. Coarse-resolution surface and vertical predictors are encoded into a compact latent representation, which is supplied to cross-attention blocks at multiple spatial resolutions within the UNet encoder–decoder.  
![](images/c08ab8252ef780b6e0850d15e9496937e258da3af9a24f5082c4753ca61bc51c.jpg)  
Figure 3 Modifications for the integration of the Prithvi WxC weather foundation model as a conditioning encoder within the difusion framework. A lightweight convolutional preprocessing stage maps the coarse-resolution ERA5 predictor stack to the input format expected by the Prithv WxC encoder. The frozen foundation model encoder produces high-dimensional spatiotemporal embeddings, which are subsequently projected to a compact feature map via convolutional postprocessing layers and injected into the difusion UNet via cross-attention.

The Prithvi WxC model follows a vision transformer architecture composed of hierarchical Swin Transformer blocks operating on 3D patches (latitude, longitude, and time). During pretraining, the model is exposed to decades of global data through a masked autoencoder objective, with the goal of learning rich multiscale representations of atmospheric dynamics. As a result, Prithvi WxC produces latent embeddings that capture both local- and global-scale features relevant for climate predictions.

To incorporate Prithvi WxC as a conditioning module within a difusion downscaling model, we use only its encoder and add some extra convolutional layers, as illustrated in Figure 3.

First, the raw predictors (e.g., coarse-resolution ERA5 variables) must be mapped to the input channel configuration expected by the Prithvi WxC encoder. We implement a lightweight convolutional preprocessing stage that converts the predictor stack into the format expected by the foundation model: a feature map with 320 channels. This ensures compatibility with the pretrained patch-embedding layer and avoids retraining the early stages of the transformer from scratch, which in our testing led to worse results. All Prithvi WxC parameters remain frozen during training.

Second, because the native Prithvi WxC output embeddings are substantially higher-dimensional than those typically used in UNet denoisers, leading to a high computational cost, a post-processing projection is applied. A convolutional head reduces the embedding dimensionality to produce a compact feature map suitable for injection into the difusion denoiser. These reduced feature maps are then supplied to the cross-attention layers at multiple UNet spatial resolutions as described in the previous subsection.

In summary, the frozen component is the pretrained Prithvi WxC encoder, which acts purely as a fixed feature extractor. The trainable components are: the convolutional preprocessing adapter, the convolutional post-processing projection head, and the difusion UNet itself. This design conditions the difusion model on latent representations derived from a weather foundation model, encoding physically meaningful features, while restricting gradient updates to the lightweight adapters and the denoiser. In practice, this leads to a form of semantically informed conditioning: the Prithvi WxC encoder supplies context about the large-scale patterns, while the difusion model focuses on reconstructing high-resolution precipitation fields consistently with this context. The result is a conditioning mechanism designed to exploit foundation model representations in climate downscaling.

To compare with the Prithvi WxC encoder (CA-PWC), we employ a learnable convolutional encoder (CA-CE) trained from scratch to map the coarse-resolution predictor fields into a compact latent representation. This encoder consists of three convolutional blocks, each comprising two 2D convolutional layers with SiLU activations. In each block, the first and third convolution preserves spatial resolution and is followed by feature-wise layer normalization to stabilize training, while the second convolution applies a stride of two to progressively downsample the feature maps, which increases the receptive field as each unit in deeper layers corresponds to a larger region of the original input. The channel dimensionality is kept constant across blocks. The final feature map is subsequently used as conditioning (keys and values) for the cross-attention layers of the difusion UNet. This design provides a comparable and computationally eficient conditioning mechanism.

## 3.5 Experiments

As precipitation is a particularly challenging variable for statistical downscaling due to its high spatial heterogeneity and complex distributional tails, we employ a multi-faceted validation approach that evaluates the fields across several dimensions of quality.

We evaluate the proposed stochastic precipitation downscaling framework following established practice in climate downscaling. We employ non-overlapping temporal splits for training, validation, and testing: the training set covers 1985–2009, the validation set covers 2010–2012, and the test set covers 2013–2015. This partition prevents leakage from future years during model selection and ensures comparability across all experiments. All input variables are normalized using scale factors computed from the training set.

To address the three research questions posed in the Introduction, model performance is assessed across four complementary dimensions. First, to evaluate whether cross-attention conditioning reproduces more realistic fine-scale spatial properties than concatenation, we quantify spectral fidelity via the Radially Averaged Power Spectral Density (RAPSD) and log-spectral distance (RALSD), and spatial skill via the Fractions Skill Score (FSS) at multiple intensity thresholds. Second, to assess whether foundation model embeddings improve distributional accuracy and extreme-event representation, we compute distributional fidelity metrics (distribution bias, histogram CRPS), annual maximum precipitation statistics (CRPS, MSE, MAE, and distribution bias on annual maxima), and event frequency ratios across intensity bins. Probabilistic accuracy (pixel-wise and spatially aggregated CRPS) and bias metrics (absolute bias, relative bias, and their time-averaged counterparts) provide a broader baseline of downscaling skill across all conditioning strategies; ensemble calibration is evaluated via the rank histogram calibration error. Third, to quantify the extent to which foundation model conditioning reduces dependency on large historical training datasets, we compare both cross-attention encoders trained on progressively shorter historical periods spanning

5 to 25 years. Formal definitions of all metrics are provided in A.1.

All experiments employ a UNet denoising architecture with a symmetric encoder–decoder design consisting of four downsampling and four upsampling blocks. Skip connections between corresponding encoder and decoder blocks preserve information during the denoising process. The model is conditioned on difusion timestep information using a Fourier time embedding, which modulates the denoising behavior across noise levels. The experiments using Prithvi WxC maintain the pretrained backbone of the foundation model frozen.

As described in Section 3.2, the concatenation approach requires upscaling the coarse ERA5 predictors to the high-resolution predictand grid, for this we used nearest-neighbour interpolation. The static variables are already available at high-resolution, so no additional upscaling is necessary. We also evaluate an unconditioned difusion (UNC) model which serves as a baseline to isolate the architectural performance of the model from the gains provided by conditioning. Although it may capture broad statistical realism, it cannot accurately downscale specific weather events and is expected to fail on all conditional accuracy metrics.

During training, we apply log-uniform sampling of noise magnitudes following Brenowitz et al. [2025], with noise levels ranging from $\sigma _ { \mathrm { m i n } } = 0 . 0 2$ to $\sigma _ { \mathrm { m a x } } = 2 0 0 . 0$ . This range is chosen to span the full dynamic range between the lowest-variance and highest-variance modes of the precipitation fields. The authors show that sampling suficiently large noise levels during training is critical for difusion models to generate realistic predictions. The difusion model is trained for 100 epochs on a single NVIDIA A100 GPU using mixed-precision arithmetic with the bf16 floating-point format. To ensure fair comparisons between experiments, both training and inference are conducted using fixed random seeds, ensuring reproducibility across runs.

At inference time, stochastic high-resolution precipitation fields are generated using the noise discretization scheme introduced by Karras et al. [2022]. We employ $N = 2 0$ sampling steps with $\rho = 7$ , which controls the spacing of noise levels $\{ \sigma _ { \mathrm { m a x } } , \hdots , \sigma _ { \mathrm { m i n } } \}$ in the discretization schedule, followed by a final denoising step with $\sigma = 0$ . This sampling strategy enables a gradual transition from coarse stochastic structure to fine-scale spatial refinement. For each day in the test period, we generate five stochastic realizations for evaluation.

Overall, this experimental design, comprising standardized climate data, strict temporal splits, log-uniform noise sampling during training, a 20-step sampling schedule for inference, and 5-member ensemble predictions, provides a robust and reproducible setup for assessing stochastic difusionbased precipitation downscaling.

## 4 Results and Discussion

We evaluate the proposed difusion models for precipitation downscaling by comparing the four conditioning strategies under identical training, inference, and evaluation settings. All quantitative results are reported for daily total precipitation $\mathrm { ( m m / d a y ) }$ over the held-out test period (2013– 2015), with five stochastic ensemble members generated per test day, ensuring that performance diferences arise solely from the conditioning mechanism rather than data leakage or architectural variation.

## 4.1 Precipitation Evaluation

Here we present the results of the evaluation of pixel-level accuracy of the generated spatial rainfall time series, alongside distributional accuracy/realism, including the ability to reproduce extreme events. For daily total precipitation (Table 2), pointwise and spatially aggregated error metrics show that the CC model achieves the lowest raw CRPS (0.511) and MSE (4.282), indicating superior pixel-level accuracy. The CA-CE and CA-PWC models perform similarly to each other (raw CRPS of 0.581 and 0.557, respectively), while the UNC baseline exhibits substantially larger errors across all metrics as expected. However, when considering bias measures, CA-PWC achieves the lowest time-averaged absolute bias (0.095 mm/day) and absolute relative bias (0.097), outperforming both CC (0.185, 0.169) and CA-CE (0.107, 0.112).

Precipitation distribution and spatial spectral metrics (Table 3) indicate superior distributional performance of the cross-attention models. CA-PWC achieves the lowest distribution bias (0.065), nearly five times lower than CC (0.385) and half of the CA-CE (0.138). Despite slightly higher calibration error relative to the UNC model, both cross-attention models produce more accurate spatial power spectra, the CA-CE and CA-PWC achieve RAPSD CRPS values of 2.632 and 2.598, compared with 2.763 for CC and 7.011 for UNC. RALSD scores follow the same ordering, with CA-CE (5.390) and CA-PWC (5.539) outperforming CC (5.942), suggesting that cross-attention conditioning better preserves fine-scale spatial structure, producing realistic spatial rainfall distributions.

Table 2 Comparison of conditioning strategies for probabilistic accuracy and bias metrics for daily total precipitation (mm/day) over the test period (2013–2015). Bold values indicate best performance per metric. ”Raw” values correspond to mean of daily per-pixel precipitation metrics. ”x16” are calculated on 16x16 degree grid mean values. and ”Time $\mathrm { a v g } ^ { \dag }$ on mean values across all time steps in the test period.
<table><tr><td rowspan="3"></td><td colspan="2">CRPS</td><td colspan="2">MSE</td><td colspan="2">Abs. Bias</td><td colspan="2">Abs. Rel. Bias</td></tr><tr><td>Raw</td><td>(x16)</td><td>Raw</td><td>(x16)</td><td>Raw</td><td>Time avg.</td><td>Raw</td><td>Time avg.</td></tr><tr><td>UNC</td><td>1.092</td><td>1.034</td><td>15.993</td><td>11.893</td><td>1.617</td><td>0.402</td><td>317.22</td><td>0.380</td></tr><tr><td>CC</td><td>0.511</td><td>0.389</td><td>4.282</td><td>1.918</td><td>0.760</td><td>0.185</td><td>45.656</td><td>0.169</td></tr><tr><td>CA-CE</td><td>0.581</td><td>0.444</td><td>5.497</td><td>2.825</td><td>0.876</td><td>0.107</td><td>49.494</td><td>0.112</td></tr><tr><td>CA-PWC</td><td>0.557</td><td>0.415</td><td>5.934</td><td>2.548</td><td>0.827</td><td>0.095</td><td>35.710</td><td>0.097</td></tr></table>

Table 3 Comparison of conditioning strategies for precipitation distribution and spatial spectral metrics on the test period.
<table><tr><td rowspan="2"></td><td colspan="3">Precipitation Distribution</td><td colspan="2">Spatial Statistics</td></tr><tr><td>Distribution Bias</td><td>Calibration Error</td><td>Histogram CRPS</td><td>RALSD</td><td>RAPSD CRPS</td></tr><tr><td>UNC</td><td>0.358</td><td>0.028</td><td>0.150</td><td>13.107</td><td>7.011</td></tr><tr><td>CC</td><td>0.385</td><td>0.065</td><td>0.102</td><td>5.942</td><td>2.763</td></tr><tr><td>CA-CE</td><td>0.138</td><td>0.079</td><td>0.102</td><td>5.390</td><td>2.632</td></tr><tr><td>CA-PWC</td><td>0.065</td><td>0.050</td><td>0.102</td><td>5.539</td><td>2.598</td></tr></table>

![](images/e678224409cf1c27d065a21806b83dfea85be7cf947ea51359f4e04e26d5afee.jpg)  
Figure 4 Mean and maximum daily total precipitation (mm/day) over the test period (2013–2015) for the ERA5-Land reference dataset and each conditioning strategy. Top row (a): time-mean daily precipitation fields and corresponding marginal histograms. Bottom row (b): time-maximum daily precipitation fields and corresponding marginal histograms. Red dashed lines on histograms indicate the reference mean value.

![](images/5dff06abdd06a07b07bd410253243d1d0f10d53a9966fbdd906cc32cf251e2c9.jpg)  
Figure 5 Spatial maps of mean relative bias (%) in daily total precipitation (mm/day) for the test period (2013–2015). Top row: pixel-wise mean relative bias across all test days. Bottom row: mean relative bias computed after spatial maximum aggregation.

This contrast between pixel-level accuracy and distributional fidelity reflects the diference in how conditioning strategies represent precipitation. Spatial maps of mean and maximum daily precipitation (Figure 4) illustrate these diferences visually. All three conditioned models reproduce the broad spatial patterns of mean precipitation, but the CC model produces noticeably smoother fields. The marginal histograms confirm that CC underestimates the frequency of high-intensity events in the test dataset, whereas CA-CE and CA-PWC more faithfully reproduce the reference right tail. This efect is amplified in the maximum precipitation fields, where the cross-attention models capture peak values closer to the reference. Spatial bias maps (Figure 5) further show that CA-PWC exhibits the smallest mean relative bias across most of the domain for both daily and maximum precipitation. A notable shared feature of the CA-CE and CA-PWC models is a dry bias extending northward from the south-western corner of the domain. For maximum precipitation, the CA-PWC model tends to exhibit a dry bias to varying degrees across the domain, while the CA-CE also has a wet bias in some locations.

Table 4 Comparison of conditioning strategies for annual maximum daily precipitation (mm/day) over the test period (2013–2015). Bold values indicate best performance per metric.
<table><tr><td rowspan="2"></td><td colspan="4">Annual Maximum Precipitation</td></tr><tr><td>CRPS</td><td>MSE</td><td>MAE</td><td>Distribution Bias</td></tr><tr><td>UNC</td><td>9.234</td><td>288.551</td><td>11.869</td><td>0.359</td></tr><tr><td>CC</td><td>7.522</td><td>194.825</td><td>9.486</td><td>0.388</td></tr><tr><td>CA-CE</td><td>5.750</td><td>148.712</td><td>8.271</td><td>0.134</td></tr><tr><td>CA-PWC</td><td>5.730</td><td>175.959</td><td>8.854</td><td>0.062</td></tr></table>

The advantage of cross-attention conditioning is most pronounced for annual maximum precipitation (Table 4). CA-CE achieves the lowest MSE (148.712) and MAE (8.271), while CA-PWC achieves both the lowest CRPS (5.730) and the lowest distribution bias (0.062) for annual maxima. In contrast, the CC model performs worse on all four metrics, with a distribution bias of 0.388 comparable to the UNC baseline (0.359). These results are consistent with the hypothesis that cross-attention conditioning better captures precipitation extremes, although we note that the evaluation is based on a relatively limited sample of three years and so this result should be treated with caution.

Precipitation event frequency ratios (Figure 6) provide a direct diagnostic of extreme-event fidelity. For events in the 100–200 mm/day range, CA-PWC retains over 50% of reference event counts, CA-CE retains approximately 17%, and the CC model produces virtually no events at these intensities (ratio ≈ 0). All models perform comparably for moderate intensities (below 20 mm/day), where the choice of conditioning strategy has little efect.

Radially averaged power spectral density curves (Figure 7) corroborate these findings, where both cross-attention models more accurately reproduce the observed spatial power spectrum at scales below approximately 100 km, where the CC model shows a decay in spectral power consistent with its tendency to produce over-smoothed outputs.

![](images/516f6ed320335db6bff7146eba8297df573469286d9143964083198d1543147c.jpg)  
Figure 6 Event frequency ratios across precipitation intensity bins (mm/day) for each conditioning strategy over the test period (2013–2015). The ratio is computed as the number of predicted events divided by the number of reference events within each intensity bin, with a value of 1.0 (dashed red line) indicating a perfect match. Observation pixel-event counts: 0-0.1 mm/day (n=14.38M); 0.1-1 mm/day (n=4.68M); 1-10 mm/day (n=4.40M); 10-30 mm/day (n=569.6k); 30-50 mm/day $\mathrm { ( n { = } 3 7 . 9 k ) }$ ; 50-100 mm/day (n=6.7k); 100-200 mm/day (n=263). Lower counts in extreme bins indicate higher uncertainty in those ratios.

![](images/6472247fd18538e088db86ba865cd22959192881c07a3f61ad4cd2677a67b4a4.jpg)  
Figure 7 Radially averaged power spectral density (RAPSD) of daily total precipitation over the test period (2013–2015) for the ERA5-Land reference dataset and each conditioning strategy (top panel). The bottom panel shows the ratio of predicted to reference power spectra as a function of spatial wavelength (km). Values close to 1.0 indicate accurate reproduction of spatial variability at a given scale.

![](images/c84337bedc63c7b479ff0ec79c3e06e1a77255ee58a91cbacf20781b2de6b891.jpg)

![](images/3923bb9d43cb84c7b0e7068f9b005448d102a1fd0ffaeeb70ae13495480c6879.jpg)

![](images/58b8c0d221685682fab53844d8179ce9950d1666da03048abe448c81d959d057.jpg)

![](images/4a2dd7ef98e3a3f4071fdfd9522f87a3cb4d6acdaa96644fea7a7df1e4460158.jpg)  
Figure 8 Fractions Skill Score (FSS) for daily total precipitation at multiple intensity thresholds (mm/day) as a function of spatial scale (km) for each conditioning strategy over the test period (2013–2015). Higher FSS values indicate better spatial skill at a given scale and threshold.

Fractions Skill Score results (Figure 8) indicate moderately better spatial skill for CA-PWC and CA-CE at higher intensity thresholds and finer spatial scales. For larger thresholds, CA-PWC generally achieves higher FSS than other models, while diferences among models are small for low-intensity events.

In summary, these results reveal a clear trade-of between pixel-level accuracy, where concatenation held strong performance, and distributional and extreme-event fidelity, where cross-attention conditioning with either a convolutional or foundation-model encoder ofers substantial advantages across the distributional, spectral, and extreme event metrics considered here.

## 4.2 Data Eficiency and Computational Cost

To assess how well each cross-attention encoder exploits limited training data, we compare CA-CE and CA-PWC trained on five progressively shorter historical periods (5, 10, 15, 20, and 25 years), evaluating on the fixed 2013–2015 test set. Figure 9 shows CRPS, time-averaged absolute relative bias, and RALSD as a function of training volume.

(a)  
![](images/24fa5f59aa9d79ee685b965012701397e1c66fced8fbdb9f1f1623b7b7d9f52f.jpg)

(b)  
![](images/bce2d977d27f1cfc1ea31ffa5b8e29a0103d43dc2563683c7bbfebde7dd3d812.jpg)

(c)  
![](images/2cfc5ee182d794ceb34d1534c5c3e65bb747c69738e9a77ac8c6515221bf7132.jpg)  
Figure 9 Data eficiency of cross-attention conditioning strategies as a function of training data volume (5–25 years) for CA-CE and CA-PWC. (a): CRPS, (b): Absolute Relative Bias, (c): RALSD.

Across all three metrics CA-PWC consistently outperforms CA-CE for all experiments conducted in this study. For CRPS, both models improve with more data but remain above the Concatenation baseline (0.511) at all volumes, with CA-PWC degrading less steeply as data is reduced (16% vs. 20%). For absolute relative bias, both cross-attention models stay below the CC baseline (0.185) on high volumes of data. The clearest data-eficiency advantage emerges in RALSD, where

Prithvi WxC falls below the Concatenation threshold (5.94) already at 10 years and reaches 5.23 at 25 years, while CA-CE only crosses that threshold at the 20-year period, suggesting that the pretrained spatial representations in Prithvi WxC, when combined with cross-attention conditioning, are particularly efective for learning precipitation structure from fewer training samples.

Table 5 Model complexity across conditioning strategies. Trainable parameters refers to the number of parameters updated during training, while total parameters includes the frozen Prithvi WxC backbone. GPU memory usage includes loading both model and data. Inference time is reported per day per ensemble for the 1095 day × 5 ensemble test set on a single NVIDIA H100 80GB HBM3.
<table><tr><td></td><td>Trainable Params</td><td>Total Params</td><td>GPU Memory Usage</td><td>Inference Time</td><td>Training Time per Epoch</td></tr><tr><td>UNC</td><td>48.9 M</td><td>48.9 M</td><td>15.9 GB</td><td>0.210 s</td><td>138 s</td></tr><tr><td>CC</td><td>48.9 M</td><td>48.9 M</td><td>16.0 GB</td><td>0.211 s</td><td>141 s</td></tr><tr><td>CA-CE</td><td>52.9 M</td><td>52.9 M</td><td>27.8 GB</td><td>0.605 s</td><td>324 s</td></tr><tr><td>CA-PWC</td><td>85.0 M</td><td>301 M</td><td>29.4 GB</td><td>0.684 s</td><td>342 s</td></tr></table>

Table 5 summarises model complexity across all four conditioning strategies. The UNC and CC models share the same parameter count (48.9 M) since concatenation adds no learnable components beyond the base UNet denoiser. The CA-CE model adds 4 M parameters for the three-block convolutional encoder (52.9 M total). CA-PWC has 301 M total parameters, but only 85 M are trainable as the 216 M Swin-Transformer backbone is kept frozen, representing a 1.6× increase in trainable parameters relative to CA-CE. Training cost is therefore still relatively modest despite the large backbone, though inference cost is higher as the full frozen backbone must be evaluated on every sample.

## 5 Conclusions

This paper evaluated conditioning strategies for difusion-based statistical downscaling of precipitation over the Colorado River Basin, comparing an unconditioned baseline, a concatenation approach, and two cross-attention approaches using a learned convolutional encoder and the Prithvi WxC foundation model. All experiments were conducted under identical training and inference conditions, isolating the efect of the conditioning mechanism. The results reveal a consistent tradeof: concatenation conditioning achieved the lowest pixel-level CRPS and MSE, but cross-attention conditioning with either encoder produced more accurate precipitation distributions and improved representation of extreme events.

For daily precipitation, the CC model achieved the best raw probabilistic accuracy, outperforming the cross-attention models on CRPS and MSE at every grid point. However, this advantage is associated with a tendency of concatenation to produce oversmoothed fields and suppress highintensity events. The cross-attention models achieved distribution biases up to five times lower than Concatenation, reproduced the spatial power spectrum more faithfully at high scales, and matched observed annual maximum precipitation distributions far more accurately, although annual maximum statistics were derived from only three years of observations and are therefore subject to substantial sampling uncertainty. These diferences were most pronounced for extreme events: for rainfall intensities above 100 mm/day, CA-PWC retained over 50% of observed event counts while the CC model produced virtually none, although again these results are uncertain due to the limited number of observed tail events.

CA-PWC and the CA-CE performed broadly similarly across most daily metrics. CA-PWC achieved the lowest CRPS and distribution bias for annual maxima, the lowest time-averaged absolute relative bias, and marginally better spatial spectral scores, whereas CA-CE achieved lower MSE and MAE on annual maximum precipitation and slightly better spatial realism metrics. CA-PWC and CA-CE therefore provide two complementary alternatives to CC when distributional realism is important. CA-CE ofers a lightweight, operationally practical approach, while CE-PWC can additionally address limited training data regimes. Despite having 301 M total parameters and a larger memory footprint, only 85 M parameters were trained in the CE-PWC model, meaning it was able to ofer these data eficiency advantages with only a relatively small increase in training and inference times.

For this regional precipitation downscaling task, the results presented indicate that crossattention conditioning improves distributional realism, spectral fidelity, and extreme-event representation relative to simple channel concatenation. For applications in climate impact assessment, where the frequency and intensity of extreme precipitation events are of primary concern, this may be particularly advantageous.

This study is subject to several limitations. We considered only a single geographic domain, the Colorado River Basin, and a single variable, daily precipitation, therefore results may not generalise to other regions, climate regimes, or variables. In addition, all experiments rely on ERA5 reanalysis data for conditioning, which may not capture the biases and non-stationarities present in GCM climate projections. Similarly, we used a target of ERA5-Land, which does not directly integrate precipitation observations. Our evaluation period of three years means that metrics focused on the most extreme events, particularly annual maxima, are derived from a relatively small efective sample and are therefore associated with a greater uncertainty than those metrics derived from daily rainfall. While the training plus validation period of 28 years was suficient for the CA models to learn representations of extreme events, it is possible that a concatenation-based model could narrow the gap in extreme-event performance if trained on a larger dataset. This would need further work to evaluate. The transferability of these findings to future climate scenarios remains uncertain, and warrants further evaluation across multiple basins, variables, and data sources. Future work should also consider whether jointly fine-tuning the foundation model backbone on regional data can further narrow the gap in pixel-level accuracy without sacrificing distributional fidelity.

## Acknowledgments

This work was funded by the Hartree Centre for Digital Innovation, a collaboration between IBM and STFC. Generated using or contains modified Copernicus Climate Change Service information, 2019. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

## A Appendix

## A.1 Evaluation Metrics

Probabilistic Accuracy: We use the Continuous Ranked Probability Score (CRPS) to quantify the accuracy of the 5-member ensemble predictions. The CRPS measures the distance between the cumulative distribution function (CDF) of the predicted ensemble F(x) and the reference data y:

$$
C R P S ( F , y ) = \int _ { - \infty } ^ { \infty } ( F ( z ) - \mathbb { I } ( z \geq y ) ) ^ { 2 } d z\tag{3}
$$

where I is the indicator function. We report both the pixel-wise CRPS averaged over the domain and the CRPS of spatially aggregated fields (x16) to assess performance at diferent scales.

In addition, we compute a histogram CRPS to evaluate distributional consistency. Reference data and ensemble members are transformed to log-space and converted into normalized histograms over fixed bins, which define discrete distributions. The CRPS is then computed between the observed histogram and the ensemble of predicted histograms. It provides a measure of how well the ensemble reproduces the overall value distribution independent of spatial structure.

Distribution Bias: To assess how well the model reproduces the climatological distribution of precipitation, we compute the Distribution Bias, defined as the integrated absolute diference between the empirical CDF of the model $( F _ { m o d } )$ and the reference $\left( \boldsymbol { F _ { r e f } } \right)$ across all grid points and time steps:

$$
\mathrm { D i s t r i b u t i o n ~ B i a s } = \int | F _ { m o d } ( x ) - F _ { r e f } ( x ) | d x\tag{4}
$$

This metric captures systematic errors in the frequency of precipitation intensities, independent of temporal synchronization.

Bias Metrics: To assess systematic errors in the predicted precipitation fields, we compute several bias metrics operating at diferent levels of temporal aggregation. The absolute bias is defined as the mean absolute diference between predictions and reference across all ensemble members, time steps, and grid points:

$$
{ \mathrm { A b s o l u t e ~ B i a s } } = { \frac { 1 } { N } } \sum _ { t , i , j } { | \hat { y } _ { t , i , j } - y _ { t , i , j } | }\tag{5}
$$

The relative bias normalizes the absolute pointwise error by the observed value at each location, providing a scale-independent measure of error that is especially informative in regions of low mean precipitation where absolute errors are small but proportionally large. It is computed over all non-zero observed values (totaling $N ^ { * } )$ to avoid division by zero:

$$
{ \mathrm { A b s o l u t e ~ R e l a t i v e ~ B i a s } } = { \frac { 1 } { N ^ { * } } } \sum _ { \stackrel { t , i , j } { y _ { t , i , j } \neq 0 } } { \frac { | { \hat { y } } _ { t , i , j } - y _ { t , i , j } | } { y _ { t , i , j } } }\tag{6}
$$

The time-averaged bias collapses the temporal dimension by first computing the mean predicted and observed fields across all time steps, and then evaluating their diference at each spatial location. This metric therefore quantifies bias in the climatological mean field, rather than errors at individual time steps, and highlights spatial discrepancies between the predicted and observed long-term averages.

Calibration Error: We evaluate the reliability of the ensemble spread using the Calibration Error, which measures the consistency between predicted probabilities and observed frequencies. For an ensemble of size $M ,$ , if the reference falls into the $k ^ { t h }$ bin of the predictive distribution with frequency $p _ { k }$ , the error is:

$$
\mathrm { C a l i b r a t i o n \ E r r o r } = \sum _ { k = 1 } ^ { M + 1 } ( p _ { k } - \frac { 1 } { M + 1 } ) ^ { 2 }\tag{7}
$$

Ideally, the rank histogram should be uniform, corresponding to a calibration error of zero.

Spectral Fidelity: Spatial realism is evaluated using the Radially Averaged Power Spectral Density (RAPSD). The RAPSD S(k) at wavenumber k is computed by azimuthally averaging the 2D power spectrum $| \mathcal { F } ( I ) ( u , v ) | ^ { 2 }$

$$
S ( k ) = \frac { 1 } { N _ { k } } \sum _ { k \mathop { \leq } \sqrt { u ^ { 2 } + v ^ { 2 } } < k + 1 } | \mathcal { F } ( I ) ( u , v ) | ^ { 2 }\tag{8}
$$

where $\mathcal { F }$ denotes the Fourier transform and $N _ { k }$ is the number modes in the annulus. To compare the texture of generated fields against the ERA5-Land reference, we compute the log-spectral distance (RALSD) between the model and reference spectra.

Fractions Skill Score: Spatial accuracy is evaluated using the Fractions Skill Score (FSS), which evaluates agreement between predicted ensembles and observed precipitation within a local neighborhood with a exceedance threshold τ. For each member m, a binary exceedance field is constructed as $\mathbb { I } ( \hat { y } _ { t , i , j } ^ { ( m ) } \ge \tau )$ , and the ensemble mean fractional coverage $\bar { f } _ { s } ( t , i , j )$ is obtained by averaging these fields over a window of size s:

$$
\mathrm { F S S } ( s , \tau ) = 1 - \frac { \displaystyle \sum _ { t , i , j } \left( \bar { f } _ { s } ( t , i , j ) - f _ { s } ^ { \mathrm { o b s } } ( t , i , j ) \right) ^ { 2 } } { \displaystyle \sum _ { t , i , j } \bar { f } _ { s } ( t , i , j ) ^ { 2 } + \displaystyle \sum _ { t , i , j } \left( f _ { s } ^ { \mathrm { o b s } } ( t , i , j ) \right) ^ { 2 } }\tag{9}
$$

where $f _ { s } ^ { \mathrm { o b s } } ( t , i , j )$ is the corresponding fractional coverage of the binary reference field $\mathbb { I } ( y _ { t , i , j } \ge \tau )$ within the same window. The score ranges from 0 (no skill) to 1 (perfect agreement). We report the mean FSS averaged across all samples.

## References

Henry Addison, Elizabeth Kendon, Suman Ravuri, Laurence Aitchison, and Peter Watson. Machine learning emulation of precipitation from km-scale regional climate simulations using a difusion model, 07 2024.

Rilwan A. Adewoyin, Peter Dueben, Peter Watson, Yulan He, and Ritabrata Dutta. Tru-net: a deep learning approach to high resolution prediction of rainfall. Machine Learning, 110(8):2035–2062, 2021.

Seth Bassetti, Brian Hutchinson, Claudia Tebaldi, and Ben Kravitz. Difesm: Conditional emulation of temperature and precipitation in earth system models with 3d difusion models. Journal of Advances in Modeling Earth Systems, 16(10):e2023MS004194, 2024. doi: https://doi.org/10. 1029/2023MS004194. URL https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/ 2023MS004194. e2023MS004194 2023MS004194.

Jofrey Dumont Le Brazidec, Simon Lang, Martin Leutbecher, Baudouin Raoult, Gert Mertes, Florian Pinault, Aristofanis Tsiringakis, Pedro Maciel, Ana Prieto Nemesio, Jan Polster, Cathal O Brien, and Matthew Chantry. Downscaling weather forecasts from low- to high-resolution with difusion models, 2026. URL https://arxiv.org/abs/2604.03303.

Noah D. Brenowitz, Tao Ge, Akshay Subramaniam, Peter Manshausen, Aayush Gupta, David M. Hall, Morteza Mardani, Arash Vahdat, Karthik Kashinath, and Michael S. Pritchard. Climate in a bottle: Towards a generative foundation model for the kilometer-scale global atmosphere, 2025. URL https://arxiv.org/abs/2505.06474.

Copernicus Climate Change Service (C3S). ERA5-Land hourly data from 1950 to present. https:// cds.climate.copernicus.eu/cdsapp##!/dataset/10.24381/cds.e2161bac, 2019. Accessed: 2025-10-27.

Copernicus Climate Change Service (C3S). ERA5 hourly data on single levels from 1940 to present. https://cds.climate.copernicus.eu/cdsapp##!/dataset/10.24381/cds. adbb2d47, 2023. Accessed: 2025-10-27.

European Space Agency. Copernicus digital elevation model (dem). https://doi.org/10.5270/ ESA-c5d3d65, 2019. European Space Agency (ESA); Available at: https://doi.org/10.5270/ESAc5d3d65.

Ronald Gelaro, Will McCarty, Max J. Su´arez, Ricardo Todling, Andrea Molod, Lawrence Takacs, Cynthia A. Randles, Anton Darmenov, Michael G. Bosilovich, Rolf Reichle, Krzysztof Wargan, Lawrence Coy, Richard Cullather, Clara Draper, Santha Akella, Virginie Buchard, Austin Conaty, Arlindo M. da Silva, Wei Gu, Gi-Kong Kim, Randal Koster, Robert Lucchesi, Dagmar Merkova, Jon Eric Nielsen, Gary Partyka, Steven Pawson, William Putman, Michele Rienecker, Siegfried D. Schubert, Meta Sienkiewicz, and Bin Zhao. The modern-era retrospective analysis for research and applications, version 2 (merra-2). Journal of Climate, 30(14):5419 – 5454, 2017. doi: 10. 1175/JCLI-D-16-0758.1. URL https://journals.ametsoc.org/view/journals/clim/30/14/ jcli-d-16-0758.1.xml.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 5769–5779, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

J. M. Guti´errez, D. Maraun, M. Widmann, et al. An intercomparison of a large ensemble of statistical downscaling methods over europe: Results from the value perfect predictor crossvalidation experiment. International Journal of Climatology, 39(9):3750–3785, 2019. doi: https:// doi.org/10.1002/joc.5462. URL https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/ joc.5462.

H. Hersbach, B. Bell, P. Berrisford, G. Biavati, A. Hor´anyi, J. Mu˜noz Sabater, J. Nicolas, C. Peubey, R. Radu, I. Rozum, D. Schepers, A. Simmons, C. Soci, D. Dee, and J.-N. Th´epaut. ERA5 hourly data on single levels from 1940 to present. https://cds.climate.copernicus.eu/cdsapp##! /dataset/10.24381/cds.adbb2d47, 2023. Accessed: 2025-10-27.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. CoRR, abs/2006.11239, 2020. URL https://arxiv.org/abs/2006.11239.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of difusionbased generative models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 26565– 26577. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/ paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf.

Nikolay Koldunov, Thomas Rackow, Christian Lessig, Sergey Danilov, Suvarchal K. Cheedela, Dmitry Sidorenko, Irina Sandu, and Thomas Jung. Emerging ai-based weather prediction models as downscaling tools, 2024. URL https://arxiv.org/abs/2406.17977.

Christian Lessig, Ilaria Luise, Bing Gong, Michael Langguth, Scarlet Stadtler, and Martin Schultz. Atmorep: A stochastic model of atmosphere dynamics using large scale representation learning, 2023. URL https://arxiv.org/abs/2308.13280.

Ignacio Lopez-Gomez, Zhong Yi Wan, Leonardo Zepeda-N´u˜nez, Tapio Schneider, John Anderson, and Fei Sha. Dynamical-generative downscaling of climate model ensembles. Proceedings of the National Academy of Sciences, 122(17):e2420288122, 2025. doi: 10.1073/pnas.2420288122. URL https://www.pnas.org/doi/abs/10.1073/pnas.2420288122.

Douglas Maraun, Fredrik Wetterhall, Andrew M. Ireson, Richard E. Chandler, Michele Kendon, Martin Widmann, Sander Brienen, Henning W. Rust, Tobias Sauter, Michael Themassl, Victor Venema, Kyung-Ja Chun, Clare M. Goodess, Richard G. Jones, Christian Onof, Mathieu Vrac, and Martin Widmann. Precipitation downscaling under climate change: Recent developments to bridge the gap between dynamical models and the end user. Reviews of Geophysics, 48(3):

RG3003, 2010. doi: 10.1029/2009RG000314. URL https://agupubs.onlinelibrary.wiley. com/doi/full/10.1029/2009RG000314.

M. Mardani, N. Brenowitz, Y. Cohen, J. Pathak, C.-Y. Chen, C.-C. Liu, A. Vahdat, M. A. Nabian, T. Ge, A. Subramaniam, K. Kashinath, J. Kautz, and M. S. Pritchard. Residual corrective difusion modeling for km-scale atmospheric downscaling. Communications Earth & Environment, 6:124, 2025. doi: 10.1038/s43247-025-02042-5. URL https://doi.org/10.1038/ s43247-025-02042-5.

J. Mu˜noz Sabater. ERA5-Land hourly data from 1950 to present. https://cds.climate. copernicus.eu/cdsapp##!/dataset/10.24381/cds.e2161bac, 2019. Accessed: 2025-10-27.

Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K. Gupta, and Aditya Grover. Climax: A foundation model for weather and climate. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 25906–25933. PMLR, 2023a. URL https://proceedings. mlr.press/v202/nguyen23a.html.

Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, and Aditya Grover. Climatelearn: Benchmarking machine learning for weather and climate modeling. arXiv preprint arXiv:2307.01909, 2023b. URL https://arxiv.org/abs/2307.01909.

R. Po, W. Yifan, V. Golyanik, K. Aberman, J. T. Barron, A. Bermano, E. Chan, T. Dekel, A. Holynski, A. Kanazawa, C.K. Liu, L. Liu, B. Mildenhall, M. Nießner, B. Ommer, C. Theobalt, P. Wonka, and G. Wetzstein. State of the art on difusion models for visual computing. Computer Graphics Forum, 43(2):e15063, 2024. doi: https://doi.org/10.1111/cgf.15063. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15063.

Neelesh Rampal, Sanaa Hobeichi, Peter B. Gibson, Jorge Ba˜no-Medina, Gab Abramowitz, Tom Beucler, Jose Gonz´alez-Abad, William Chapman, Paula Harder, and Jos´e Manuel Guti´errez. Enhancing regional climate downscaling through advances in machine learning. Artificial Intelligence for the Earth Systems, 3(2):230066, 2024. doi: 10.1175/AIES-D-23-0066.1. URL https://journals.ametsoc.org/view/journals/aies/3/2/AIES-D-23-0066.1.xml.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent difusion models. CoRR, abs/2112.10752, 2021. URL https://arxiv.org/abs/2112.10752.

Fran¸cois Rozet and Gilles Louppe. Score-based data assimilation, 2023. URL https://arxiv.org/ abs/2306.10574.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David Fleet, and Mohammad Norouzi. Photorealistic text-to-image difusion models with deep language understanding. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 36479–36494. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713–4726, 2023. doi: 10.1109/TPAMI.2022.3204461.

Alexandros Saoulis, Chris Lucas, Natalie Lord, Nans Addor, and Jorge Moraga. Difusion models for climate data surpass alternative statistical downscaling techniques, 02 2025.

Johannes Schmude, Sujit Roy, Will Trojak, Johannes Jakubik, Daniel Salles Civitarese, Shraddha Singh, Julian Kuehnert, Kumar Ankur, Aman Gupta, Christopher E Phillips, Romeo Kienzler, Daniela Szwarcman, Vishal Gaur, Rajat Shinde, Rohit Lal, Arlindo Da Silva, Jorge Luis Guevara Diaz, Anne Jones, Simon Pfreundschuh, Amy Lin, Aditi Sheshadri, Udaysankar Nair, Valentine Anantharaj, Hendrik Hamann, Campbell Watson, Manil Maskey, Tsengdar J Lee, Juan Bernabe Moreno, and Rahul Ramachandran. Prithvi wxc: Foundation model for weather and climate, 2024. URL https://arxiv.org/abs/2409.13598.

Prakhar Srivastava, Ruihan Yang, Gavin Kerrigan, Gideon Dresdner, Jeremy J McGibbon, Christopher S. Bretherton, and Stephan Mandt. Precipitation downscaling with spatiotemporal video difusion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=hhnkH8ex5d.

E. Tomasi, G. Franch, and M. Cristoforetti. Can ai be enabled to perform dynamical downscaling? a latent difusion model to mimic kilometer-scale cosmo5.0 clm9 simulations. Geoscientific Model Development, 18(6):2051–2078, 2025. doi: 10.5194/gmd-18-2051-2025. URL https://gmd.copernicus.org/articles/18/2051/2025/.

Thomas Vandal, Evan Kodra, and Auroop R. Ganguly. Intercomparison of machine learning methods for statistical downscaling: the case of daily and extreme precipitation. Theoretical and Applied Climatology, 137(1-2):557–570, 2019. doi: 10.1007/s00704-018-2613-3. URL https://link.springer.com/article/10.1007/s00704-018-2613-3.

X. Wang, D. Lu, P. Thornton, P. Balaprakash, M. Ashfaq, et al. Orbit-2: Scaling exascale vision foundation models for weather and climate downscaling. arXiv preprint arXiv:2505.04802, 2025. URL https://arxiv.org/abs/2505.04802.

Tao Yang, Cuiling Lan, Yan Lu, and Nanning Zheng. Difusion model with cross attention as an inductive bias for disentanglement. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 82465–82492. Curran Associates, Inc., 2024. doi: 10.52202/079017-2622. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/9647157086adf5aa2c0217fb7f82bb19-Paper-Conference.pdf.

Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing textto-image difusion models for visual perception. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5706–5716, 2023. doi: 10.1109/ICCV51070.2023.00527.