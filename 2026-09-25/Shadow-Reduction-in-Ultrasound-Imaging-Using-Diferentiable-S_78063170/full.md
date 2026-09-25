# Shadow Reduction in Ultrasound Imaging Using Diferentiable Simulation and Radiance Field Decomposition

Valentin Bacher<sup>a</sup>, Pak Hei Yeung<sup>c,a</sup>, Bernhard Kainz<sup>d,e</sup>, Madeleine K Wyburd<sup>a,f</sup>, Nicola K Dinsdale<sup>a</sup>, Michael Gray<sup>b</sup>, Ana IL Namburete<sup>a</sup>

<sup>a</sup>Oxford Machine Learning in NeuroImaging Lab, Department of Computer Science, University of Oxford, OX1 3QD, United Kingdom

<sup>b</sup>Institute of Biomedical Engineering, University of Oxford, Marcela Botnar Wing, Oxford, OX3 7LD, United Kingdom

<sup>c</sup>Quantitative Healthcare Analysis (qurAI) Group, Informatics Institute, University of Amsterdam, Amsterdam, 1098 XH, The Netherlands

<sup>d</sup>Friedrich-Alexander-Universität Erlangen-Nürnberg, Werner-von-Siemens Str. 61, Erlangen, 91052, Germany

<sup>e</sup>Imperial College London, 180 Queen’s Gate, London, SW7 2AZ, United Kingdom <sup>f</sup>Department of Computer Science, University of Copenhagen, Universitetsparken 1, Copenhagen, 2100, Denmark

## Abstract

Acoustic shadows from bone and other highly attenuating tissues obscure clinically important structures in ultrasound. In fetal brain imaging, skullinduced artefacts disproportionately degrade the hemisphere closer to the transducer (proximal), limiting symmetric assessment of the two hemispheres. Existing correction methods require raw scanner data, impose restrictive assumptions on tissue properties, or rely on generative models that may hal lucinate anatomy. We present RFlash, a physics-informed post-processing method that decomposes beamformed ultrasound images into explicit atten uation and scatter-intensity maps using a diferentiable radiance-field formulation of image formation. Attenuation-adaptive re-rendering then removes the dependence of the signal at each depth on the intervening tissue, equivalent to virtually advancing the transducer into the tissue. Across 1,261 3D fetal brain volumes, 143 real 2D curvilinear abdominal scans, and 1,200 simulated 2D linear-probe liver scans, RFlash reduces shadow-related intensity diferences more efectively than classical Hughes–Duck attenuation correction

For a gestational-age model trained on the distal hemisphere (further from the transducer) and applied to the proximal hemisphere, prediction error decreases by 5.1 days (40%) relative to the original images. The estimated attenuation maps also yield shadow-confidence maps that improve random-forest bone-shadow segmentation over the image alone and receive greater SHAP importance than an existing neural confidence-map baseline, suggesting greater physical consistency. RFlash requires neither hardware modification nor access to raw scanner data and supports 2D and 3D acquisitions with linear and curvilinear probes, making it widely applicable allowing clinicians to use our method on their already acquired scanners and images.

Keywords: Ultrasound imaging, Shadow reduction, Diferentiable ultrasound simulation, Radiance fields, Fetal brain imaging, Explainable machine learning

## 1. Introduction

Ultrasound is a standard imaging modality in prenatal care [1]. It is cost-efective, portable, and low risk, which makes it suitable for repeated clinical examinations. Despite substantial improvements in image quality in recent decades, ultrasound images still sufer from low signal-to-noise ratio and physics-based artefacts, including acoustic shadows, reverberations, drop-out, and enhancement artefacts. These artefacts reduce local contrast and can obscure structures that are clinically relevant [2].

Shadow artefacts are particularly limiting when highly attenuating structures cannot be avoided by changing the probe position. This is common in fetal brain ultrasound, where the skull increasingly calcifies during gestation and produces stronger attenuation. As a result, the distal (further from the transducer) hemisphere is often better visible than the proximal (closer to the transducer) hemisphere, which limits assessment of bilateral anatomy and symmetry [3, 4]. This is clinically relevant as brain symmetry is an important biomarker in fetal neurodevelopmental assessment [5]. Representative examples of these artefacts are shown in Figure 1. Efective shadow reduction could, therefore, increase the amount of usable image information by improving visibility in the proximal hemisphere.

Although shadow reduction is important, it remains an unsolved problem [2, 6, 7]. Most ultrasound reconstruction pipelines assume similar soundenergy reduction across neighbouring scanlines. Heterogeneous tissue violates this assumption, which can lead to local under- or over-exposure [8]. Classical shadow-reduction methods compensate for such intensity variation using restrictive statistical or physical assumptions. For example, Hughes and Duck [9] and Knipp et al. [10] assume a linear relation between attenuation and scattering, which may break down in complex geometries and heterogeneous tissue. Frequency-based attenuation estimation has also been explored [11], but requires frequency information in their RF data, which almost all commercially available scanners don’t provide, limiting its use as a post-processing method. Learning-based approaches can reduce artefacts visually, but may lack physical grounding, may generalize poorly across probes, and can introduce hallucinated image content [12, 13].

![](images/6f062e4dcf291c87f1468d5c6a199b74aa0229977b826a9f63991d5f87f9e368.jpg)  
Figure 1: Acoustic artefacts in clinical B-mode ultrasound. Solid red boxes mark acoustic shadows behind strongly attenuating structures. The dashed blue box marks reverberation, the dashed green box marks enhancement artefacts, and the dashed orange box marks the contrast diference between the proximal (closer to the transducer) and distal (further away from the transducer) fetal brain hemispheres. RFlash targets enhancement and shadow artefacts and, to some extent, the contrast diference between the hemispheres.

In this work, we introduce Radiance Field for acoustic shadow reduction (RFlash), a physics-informed post-processing approach that explicitly models ultrasound propagation through tissue. RFlash uses a fully diferentiable ultrasound simulation pipeline to decompose ultrasound images into physically interpretable parameter maps. These parameters are then used to render attenuation-corrected, shadow-reduced images. Conceptually, the shadow-free rendering can be interpreted as virtually moving the transducer through the tissue, as illustrated in Figure 2. Each rendered depth is made less dependent on tissue closer to the original transducer position, preventing proximal structures from casting shadows onto deeper regions. As the method operates as post-processing and uses a canonical scanline-based imag ing model, it can be applied without hardware constraints and can support diferent acquisition geometries, including linear and curvilinear probes as well as 2D and 3D ultrasound.

We evaluate RFlash on fetal brain, real abdominal, and simulated liver ultrasound data spanning 3D acquisitions with a 3D probe, curvilinear 2D acquisitions, and simulated linear-probe data. The method improves image quality both visually and quantitatively, producing more uniform contrast across fetal brain hemispheres and reducing shadow-related intensity diferences in real images. We further show that the recovered proximalhemisphere information is relevant for a downstream fetal age-prediction task. Finally, we derive shadow confidence maps directly from the attenuation parameter and show that these maps are informative for random-forest shadow segmentation and perform well compared to existing methods.

The code, as well as links to most of the data, can be found in the GitHub repository: https://github.com/vbacher/RFlash

## 2. Related Work

## 2.1. Shadow Artefact Reduction and Detection

Earlier shadow reduction methods typically used single-step estimates, due to computational limitations. These approaches often estimate attenuation statistically using maximum-likelihood estimators [14], or derive closedform attenuation and scatter maps. Knipp et al. [10] and Hughes and Duck [9] assume a linear relation between attenuation and scattering, which allows closed-form solutions but over simplifies tissue properties. Related ideas have also been used more recently in optical coherence tomography, which has similar echo-based imaging principles but uses coherent light [15, 16]. To the best of our knowledge, these are the only shadow-reduction techniques that work as a post-processing step on B-mode images and does not require any data annotation for training and/or inference, which is why we chose Hughes shadow reduction as our baseline. Frequency-based methods estimate attenuation from attenuation-dependent shifts in the signal centre frequency [11, 17] and require frequency information only contained in Radio Frequency (RF) data, that most clinical systems do not provide.

More recent methods use iterative optimisation. Yu and Wang [18] solve a small partial diferential equation in each iteration, which increases computational cost. Machine-learning approaches often focus on shadow detection rather than shadow removal, commonly using generative adversarial networks (GANs) [12, 13]. These methods can produce visually plausible results, but GANs may hallucinate image content, which limits their reliability for medical use, as hallucinations may miss abnormalities afecting diagnosis.

![](images/4e50c73ef2661cbf3b684dae71d701b138182c5a2334b513cacc1fc1d8c46c3a.jpg)

![](images/b4bc82c3020088aecd2ec600cf6aa47b5c8ad39ef8826b45571c79b6b7759339.jpg)  
Figure 2: Principle of shadow-free rendering. Left: in a conventional acquisition, the intensity at a given depth (green ray) depends on all tissue between the transducer and that depth, so attenuating structures such as the skull cast shadows into everything below them. Right: RFlash renders each depth as though the transducer had been advanced to just above it (orange arcs, short green rays), making the rendered intensity independent of overlying tissue and suppressing the shadow.

RFlash operates on B-mode images and does not require frequency information or complex solvers at each iteration. It is trained on an individualimage basis, preventing knowledge transfer between subjects and reducing the risk of hallucinations.

## 2.2. Radiance Fields

Neural radiance fields (NeRFs) learn volumetric scene representations by mapping spatial coordinates and viewing directions to colour and opacity [19]. This implicit representation has been widely extended across domains [20]. In medical imaging, radiance fields have been used to reconstruct 3D ultrasound volumes from 2D slices [21, 22, 23, 24, 25, 26]. Hybrid and Gaussian-splatting variants further reduce training time [27, 28, 29].

We build on this work by adapting Radiance Fields to ultrasound physics. Instead of learning visual appearance, our model learns tissue parameters with a diferentiable, physics-informed renderer. Unlike conventional Radiance Fields trained from 2D projections, we use cross-sectional slice data and treat each ray as a direct observation of local acoustic interactions. For eficiency and interpretability, we use a fully explicit representation [30, 31], which supports fast training and transparent parameter estimates.

To our knowledge, this is the first use of radiance fields for shadow artefact reduction in ultrasound.

## 2.3. Ultrasound Simulation

Our scene representation is learned from an ultrasound slice or volume using a diferentiable ultrasound simulation. Eficient ultrasound simulators use convolutions or point-spread functions with tissue parameters or scatterers [32, 33, 34, 35, 36], and can be implemented eficiently on GPUs [37, 38]. These methods approximate interference patterns from point scatterers well, but do not explicitly model shadow formation along scanlines. Ray-tracing and scanline-tracing methods address this by modelling acoustic propagation at tissue interfaces [39, 40].

Extensions with convolutional scattering models [41, 42] or Monte Carlo sampling [43, 44] improve realism, but are computationally expensive and often non-diferentiable. This limits their use in learning-based optimisation. Recent diferentiable ray-based models improve optimisation compatibility [45], but still rely on restrictive sampling.

In this work, we use a simplified diferentiable ray-based model that ignores cross-scattering efects [46] and models signal intensities directly, which is in line with other physics models used for shadow reduction and 2D to 3D reconstruction [9, 22]. This avoids point-spread-function convolution and enables fast, physically consistent simulation for iterative optimisation and shadow correction.

## 3. Methods

Analogous to most NeRF-like models [19], our method comprises two main components: a scene representation learned as a set of physics-informed tissue parameters and a diferentiable ultrasound simulation that renders those parameters into ultrasound images. Both components are used in two steps: decomposition (a), which learns the tissue parameters, and shadowreduced rendering (b), which renders the shadow-free images, as shown in Figure 3.

## 3.1. Ultrasound image formation model

Ultrasound scanners difer in their proprietary acquisition and processing pipelines. Since RFlash is applied as a post-processing method to the final B-mode image, we assume that all images were generated using the standard pipeline shown in Figure 4. The exact signal processing pipeline of the image to be shadow removed does not need to be known for RFlash to work.

We assume that the input data have already been beamformed at the transducer, which is the process of focusing the ultrasound beam and forming the initial image. This allows each scanline to be treated independently along the propagation direction. The measured signal is corrected by time-gain compensation (TGC), logarithmically compressed, and envelope-detected using a Hilbert transform before display. It is then scan-converted from samples along scanlines into a Cartesian image grid. For linear probes, scan conversion reduces to resampling anisotropic scanline samples onto image pixels. For curvilinear probes, polar coordinates are additionally mapped to a Cartesian grid. For 3D probes, a mapping from the scanline–depth coordinate system to Cartesian volumes for storage and display is needed, which requires some knowledge of the 3D probe type. The 3D data we used were acquired with a mechanical probe that we model as a curvilinear 2D probe rotated around an internal probe axis, producing diferent angular coordinates in the two lateral directions, as illustrated in Figure 5. We do not model analogue-to-digital (A/D) conversion or binning, because the available inputs are already digitised and these lossy steps cannot be recovered from post-processed images.

![](images/14f086310a35b51d797db4e5b73f8e2fa424a90e5f3773b08edb54ffa2f270e7.jpg)  
Figure 3: The two phases of RFlash. (a) Decomposition: attenuation A and scatter intensity Φ are stored in an explicit volume, sampled along scanlines by trilinear interpolation, and passed through the diferentiable forward simulator $K : \mathrm { A } , \Phi \mapsto { \hat { \mathrm { Y } } }$ . The rendered slice Y<sup>ˆ</sup> is compared with the original slice Y by the loss $l ( \Upsilon , \hat { \Upsilon } )$ , and gradients (dashed arrows) update the parameter volumes. (b) Shadow-reduced rendering (bottom): the converged Φ from (a) is re-rendered through the shadow-avoiding operator $K ^ { \prime } : \Phi \mapsto \mathrm { Y } ^ { \prime }$ , which omits the attenuation term and so produces a shadow-free image. Solid arrows: forward simulation. Dashed arrows: gradient flow.

## 3.2. Explicit radiance-field representation

Our model follows the radiance-field view of diferentiable image formation, as summarised in Figure 3. Unlike the original NeRF formulation [19] and recent ultrasound radiance-field methods [21, 22], we do not represent the scene with a multi-layer perceptron (MLP). MLPs provide a continuous representation, but are slow to optimise and less interpretable because each weight can afect the scene globally. In our setting, the input volumes have a fixed discrete resolution; therefore, an implicit continuous representation

![](images/e1069ec1fcd0e8515ab7fe9aa3974dc2255abfb1d1a14526cb382175419171ab.jpg)

Figure 4: The canonical ultrasound imaging pipeline on which RFlash is built. Tissue response (left) is modulated by the propagating sound field, then processed in the scanner (centre: envelope detection, time-gain compensation, log compression) and converted to a display image (right: A/D conversion, binning, scan conversion). RFlash operates on the displayed, beamformed image and inverts the shaded operations. A/D conversion and binning are lossy and are not modelled. Compare with Figure 6, which shows the corresponding diferentiable pipeline for beamformed images.  
![](images/c0e0ec521d69826e09b5987f6a46a72c10183fa3a80ea873cde4cf38ffce2303.jpg)  
Figure 5: Geometry of the 3D acquisition. A mechanically swept probe is modelled as a curvilinear 2D array (yellow fan) rotated about an internal pivot axis (dashed), so that a voxel is addressed by depth r and two angular coordinates: φ within the fan and ϑ about the pivot. Each scanline is treated as an independent propagation path.

provides limited practical benefit.

We instead use a fully explicit volumetric representation. This representation is straightforward to initialise, inspect, and optimise, at the cost of a larger memory footprint. During training, each pixel in a reference ultrasound image is mapped into the representation space, which has the dimensions of the padded input image. The corresponding tissue parameters, such as attenuation and scatterer density, are sampled from the explicit volume using trilinear interpolation and passed through the diferentiable ultrasound simulator. The rendered 2D image is then compared with the reference image to update the volumetric parameters.

## 3.3. Diferentiable ultrasound simulation

The simulator uses two learned material parameters: attenuation A, which captures local efects that reduce image intensity, and scatter intensity Φ, which controls the local backscattered signal. For numerical stability, attenuation is scaled by a constant factor g, which avoids very small gradients during optimisation. We do not separately estimate reflection and refraction, because most structures are observed from a single acoustic view and these efects cannot be uniquely disentangled from the available images. We also assume a constant speed of sound c, yielding a linear relation between imaging depth and time of flight, $t \propto x$ , which is standard for B-mode ultrasound [47].

Following Hughes and Duck [9], we model the simulated signal before scan conversion as the product of scatter intensity and the remaining acoustic intensity after attenuation:

$$
\begin{array} { r } { \tilde { \mathrm { Y } } ( \pmb { x } ) = \Phi ( \pmb { x } ) \mathrm { I } ( \pmb { x } ) . } \end{array}\tag{1}
$$

The remaining intensity $\operatorname { I } ( { \pmb x } )$ follows an exponential decay based on the attenuation coeficient A along the propagation path, analogous to the Beer-Lambert law:

$$
\operatorname { I } ( \pmb { x } ) = \operatorname { I } _ { 0 } \exp \left( - \int _ { 0 } ^ { \pmb { x } } \mathrm { A } ( \pmb { u } ) g \mathrm { d } \pmb { u } \right) .\tag{2}
$$

Substitution gives the measured pre-display signal

$$
\tilde { \mathrm { Y } } ( \pmb { x } ) = \mathrm { I } _ { 0 } \Phi ( \pmb { x } ) \exp \left( - \int _ { 0 } ^ { \pmb { x } } \mathrm { A } ( \pmb { u } ) g \mathrm { d } \pmb { u } \right) .\tag{3}
$$

Here, $\tilde { \mathrm { Y } } ( { \pmb x } )$ denotes the envelope-detected backscattered intensity after acoustic propagation and attenuation, but before TGC and log compression. This formulation is consistent with classical analytic ultrasound models used for shadow reduction [9, 18].

Because the backscattered signal decays with depth, clinical ultrasound systems apply TGC before log compression to maintain approximately uniform contrast across the image. The scanner-specific TGC profile is typically not stored with the image data. We therefore model TGC as a depthdependent exponential gain,

$$
T G C ( { \pmb x } ) = \exp ( c ( { \pmb x } ) ) ,\tag{4}
$$

where $c ( { \pmb x } )$ is linear along the propagation direction. Equivalently, this can be written as

$$
c ( { \pmb x } ) = \int _ { 0 } ^ { \pmb x } { \mathrm { A d } \pmb u } .\tag{5}
$$

After applying TGC and log compression, the simulated B-mode intensity is

$$
\hat { \mathrm { Y } } ( \pmb { x } ) = \frac { \log ( 1 + \lambda \left[ T G C ( \pmb { x } ) \tilde { \mathrm { Y } } ( \pmb { x } ) \right] ) } { \log ( 1 + \lambda ) } ,\tag{6}
$$

where λ controls the dynamic range. Equations (1)–(6) define the forward simulation operator $K : { \mathrm { A } } , \Phi \mapsto { \hat { \mathrm { Y } } }$ , shown in Figure 3. The corresponding diferentiable processing pipeline is shown in Figure 6.

## 3.4. Shadow-reduced rendering

Shadow artefacts arise when a globally defined TGC profile does not compensate for spatially varying attenuation, especially behind strongly reflecting or absorbing structures such as bone. Shadow reduction can, therefore, be formulated as estimating a structure-aware compensation profile from the learned attenuation map. Instead of applying a global gain, we compute the adaptive gain

$$
T G C _ { s r } ( { \pmb x } ) = \exp \left( \int _ { 0 } ^ { { \pmb x } } { \mathrm { A } ( \pmb u ) g \mathrm { d } \pmb u } \right) .\tag{7}
$$

Applying this compensation cancels the attenuation term in Equation (3) and yields the shadow-reduced rendering

$$
\mathrm { Y } ^ { \prime } ( \pmb { x } ) = \frac { \log ( 1 + \lambda [ \mathrm { I _ { 0 } } \Phi ( \pmb { x } ) ] ) } { \log ( 1 + \lambda ) } ,\tag{8}
$$

which defines the shadow-reduced simulation operator $K ^ { \prime } : \Phi \mapsto \mathrm { Y } ^ { \prime }$

As shown in Figure 3, the attenuation and scatter maps are first optimised from the available training slices using the realistic forward operator K. Once these material maps have been estimated, the shadow-reduced operator $K ^ { \prime }$ renders the final shadow-reduced image.

![](images/f7d56c646b43e98a70b5d5e4145cbcf0bd13cc0b587328552f9d52f3a2925353.jpg)  
Figure 6: Visual representation of the diferentiable simulation pipeline as defined by eq. (1)–eq. (10). The first target (top right) is the simulated ultrasound image $\hat { \mathrm { Y } }$ , while the lower two targets are the shadow-reduced image $\mathrm { Y ^ { \prime } }$ and the shadow map M, which are only relevant after the tissue parameters have been estimated (b in Figure 3).

## 3.5. Shadow confidence maps

We derive a shadow confidence map by comparing the measured image with its shadow-reduced rendering in the linear intensity domain. Let $L ( \cdot )$ denote the log-compression operator in Equation (6). Assuming $\mathrm { I } _ { 0 } = \mathrm { I }$ , where I is the identity, the attenuation-related intensity ratio is

$$
\begin{array} { l } { \tilde { \mathrm { M } } ( r ) = T G C ( r ) \mathrm { I } ( r ) } \\ { = \displaystyle \frac { T G C ( r ) \Phi ( r ) \mathrm { I } ( r ) } { \Phi ( r ) } } \\ { = \displaystyle \frac { L ^ { - 1 } ( \mathrm { Y } ( r ) ) } { L ^ { - 1 } ( \mathrm { Y } ^ { \prime } ( r ) ) } . } \end{array}\tag{9}
$$

Here, $\mathrm { Y } ( r )$ is the measured image and $\mathrm { Y } ^ { \prime } ( r )$ is the corresponding shadowreduced rendering. The normalised shadow confidence map is then obtained as

$$
\mathrm { M } ( r ) = 1 - \exp \left[ - \tilde { \mathrm { M } } ( r ) \right] .\tag{10}
$$

## 4. Experimental Setup

## 4.1. Datasets

For evaluation, we used four diferent data sources spanning three diferent transducer geometries.

The main dataset used is a set of 3D fetal-head ultrasound volumes acquired as part of the INTERGROWTH-21<sup>st</sup> study [48], one of the largest fetal head ultrasound studies. Data were collected using a Philips HD9 (Philips Healthcare, Amsterdam, The Netherlands) scanner equipped with a curvilinear abdominal transducer (V7-3 3D). The dataset comprised 1261 volumes from fetuses between 15 and 31 gestational weeks (GW), each standardised to a spatial resolution of $1 6 0 \times 1 6 0 \times 1 6 0$ pixels. Further details on dataset acquisition, selection criteria, and preprocessing are provided in [5].

We also evaluated our method on a publicly available abdominal ultrasound dataset [49]. We used only the real scans from the test dataset, which contains 213 diferent 2D ultrasound scans of the liver, spleen, and kidneys. For 70 of these scans, no geometry could be determined by our geometry estimator, mostly because of coupling artefacts between the transducer and the skin. Without knowing a good estimation of the scanner geometry, we cannot accurately apply our shadow reduction method. As our method focuses on shadow reduction rather than geometry estimation, we omitted these scans.

To evaluate our method against the one used in Wysocki et al. [22], we used their simulated liver dataset containing 1200 simulated liver ultrasound scans from a linear probe.

To evaluate the shadow maps, we used 14 2D curvilinear fetal scans and their shadow maps for which the scanner geometry could be determined from Meng et al. [50].

Links to the publicly available data can be found in the GitHub repository: https://github.com/vbacher/RFlash

## 4.2. Baseline shadow reduction

As a baseline, we used the shadow reduction proposed by Hughes and Duck [9], as its approach can be adapted for use in post-processing and is the basis of most classical shadow-reduction techniques. In the original paper, shadow reduction was applied directly to radio-frequency (RF) data and, therefore, requires minor adaptation for use as a post-processing method.

The problem formulation used in their study is the same as Equation (3). Using their assumption that scattering intensity and attenuation are linearly dependent, $i . e . , \mathrm { ~ A ~ } = \textit { a } + b \Phi$ , with a and b as scalar factors, they solve Equation (3) for Φ, resulting in

$$
b \Phi _ { \varphi , \vartheta } ( r ) = \frac { \tilde { \mathrm { Y } } _ { \varphi , \vartheta } ( r ) \exp ( a r ) } { \int _ { r } ^ { \infty } \tilde { \mathrm { Y } } _ { \varphi , \vartheta } ( \nu ) \exp ( a \nu ) \mathrm { d } \nu } .\tag{11}
$$

Since a is fitted to the data, the inverse log compression

$$
L ^ { - 1 } ( \Upsilon _ { \varphi , \vartheta } ( r ) ) : = \frac { ( \lambda + 1 ) ^ { \Upsilon _ { \varphi , \vartheta } ( r ) } - 1 } { \lambda } = T G C ( r ) \tilde { \Upsilon } _ { \varphi , \vartheta } ( r ) ,\tag{12}
$$

can be used to find a such that $T G C ( r ) = \exp ( a r )$ . Using this equivalence, (11) can be reformulated as

$$
b \Phi _ { \varphi , \vartheta } ( r ) = \frac { L ^ { - 1 } ( \Upsilon _ { \varphi , \vartheta } ( r ) ) } { \int _ { r } ^ { \infty } L ^ { - 1 } ( \Upsilon _ { \varphi , \vartheta } ( \nu ) ) \mathrm { d } \nu } ,\tag{13}
$$

allowing us to directly use the sampled slice $\mathrm { Y } _ { \varphi , \vartheta } ( r )$ after undoing log compression. The shadow-reduced volumes using (13) are the benchmark we refer to as Hughes shadow reduction.

## 4.3. Implementation details

We initialised the parameter maps with a mean value of 1.2 for A and 4.25 for Φ. Further details on this choice are provided in Section Appendix B.2. For stable training, we normalised the input image to the same mean value as Φ, that is, $\mu = 4 . 2 5$ and a standard deviation of $\sigma = \mu / 3 \approx 1 . 4 1 6 7$ The constant ofset $g \ = \ 8 0$ was chosen to match the initialization of $\mathrm { A } ,$ which also defines the TGC as $\begin{array} { r } { \exp ( \int _ { 0 } ^ { x } \mathrm { A } _ { 0 } g \mathrm { d } \pmb { u } ) } \end{array}$ with $\mathrm { A } _ { 0 } = 1 . 2$ . For training, we used a weighted SSIM [51] and $L _ { 2 }$ loss, with $1 - S S I M ( \mathrm { Y } , \hat { \mathrm { Y } } )$ weighted by 0.95 and $\| \mathrm { Y } - \hat { \mathrm { Y } } \| _ { 2 } ^ { 2 }$ weighted by 0.05. The model was implemented in PyTorch [52] and optimised using Adam [53] with a learning rate of 0.03 for 300 epochs. Learning the representation took less than one minute on an NVIDIA GeForce RTX 4090 GPU with 24 GB of memory for each 3D fetal brain volume.

## 4.4. Evaluation: Measurement of shadows

We define an image Im as strongly afected by shadows if the mean grey value in shadowed regions difers from that in regions not afected by shadows. Let Ω denote the set of pixels in Im and let ${ \mathcal { M } } \subseteq \Omega$ denote the thresholded shadow map obtained from Hughes shadow reduction or RFlash, using the procedure described in Section 3.5. We denote the erosion operator by . Applying it to the thresholded shadow map and its complement defines the shadowed and non-shadowed regions as

$$
\begin{array} { r } { \boldsymbol { S } = \boldsymbol { \mathcal { E } } ( \boldsymbol { M } ) , } \end{array}\tag{14}
$$

and the corresponding non-shadowed region as

$$
{ \mathcal { N } } = { \mathcal { E } } ( \Omega \setminus { \mathcal { M } } ) .\tag{15}
$$

Once these regions are identified, we equalised the image histograms to make the absolute values comparable. The diference in mean grey values is then

$$
\Delta \mu ( I m ) = \mu _ { N } ( I m ) - \mu _ { S } ( I m ) ,\tag{16}
$$

so that positive values indicate higher mean intensity in the non-shadowed region than in the shadowed region. In contrast to [15], we calculate the mean over the entire image, not just single layers. For visual and numerical evaluation of the methods, we used all available datasets apart from the 14 samples provided by [50], as their statistical power was too small.

## 4.5. Evaluation: Age prediction as clinical downstream task

To assess the clinical relevance of the recovered structures in the proxi mal hemisphere, we evaluated whether a standard age-prediction model can make better use of this region after shadow reduction. Following [54, 55], we trained a 3D ResNet-10 [56] to predict fetal age from the better-visible distal hemisphere. The model was trained on the fetal brain volumes described in Section 4.1, aligned to a common space and size-normalised, which reduces the possibility that age is predicted from global size diferences rather than anatomical structure. In total, 1261 samples were used with a subject-level train, validation, and test split of [868, 133, 260]. Testing was then performed on both hemispheres to determine whether shadow reduction improves generalisation from the distal to the proximal hemisphere.

## 4.6. Evaluation: Consistency of shadow maps

To evaluate the physical consistency of the predicted shadow maps, and to compare them with the more widely studied shadow confidence maps, we compared our method with Ultra-NeRF [22], which addresses a related problem in 2D to 3D ultrasound reconstruction. We first compared the resulting shadow confidence maps visually on fetal brain examples from Meng et al. [57] and on the synthetic liver dataset of Wysocki et al. [22]. We then compared how informative the maps are for a downstream task by using a random forest to segment bone shadows, with the corresponding confidence map provided as an additional input. For a direct comparison, we followed the evaluation setup and, therefore, also the model choice and datasets of Yesilkaynak et al. [58]. Further design details can be found in that work. The random forest was trained either on the simulated ultrasound images alone or on the ultrasound images together with either the Ultra-NeRF confidence maps or our confidence maps. Performance was evaluated numerically and by analysing the feature importance assigned to the shadow maps.

## 5. Results

Visual results for three representative examples from each dataset are shown in Figure 7, with regions of interest highlighted by orange arrows. This figure also demonstrates shadow reduction on 3D, curvilinear 2D, and linear ultrasound imaging geometries. RFlash efectively reduced shadow artefacts across the evaluated datasets. In the fetal brain examples, structures in the proximal hemisphere, such as the Sylvian fissure, became more visible where suficient image information was available for reconstruction. The contrast between the proximal and distal hemispheres was also more uniform after shadow reduction. Compared with Hughes shadow reduction, RFlash handled enhancement artefacts more consistently and better equalised contrast across regions with diferent intensity levels, as highlighted in Figure 7.

## 5.1. Numerical Results

The numerical comparison in Table 1 shows that RFlash significantly outperformed Hughes shadow reduction on all real datasets. For each method, the measurement was computed using the corresponding shadow map. Additional results for crossed shadow maps are provided in Section Appendix A.1. On the simulated dataset, Hughes shadow reduction achieved the best numerical score, possibly because the assumed linear relation between attenuation and scattering is better satisfied in simulated images than in real ultrasound data. However, this metric does not capture enhancement artefacts, for which RFlash produced visually stronger results on the same dataset as marked in Figure 7.

Fetal Brain  
Abdominal  
Synthetic liver  
![](images/84d0f78def6bd4d4853c405ce5509317dd871eafb156364f8fd21131faf1380d.jpg)  
Figure 7: Qualitative comparison across acquisition geometries: fetal brain (3D me<sub>Figure 7: Qualitative comparison across acquisition geometries: fetal brain (3D mechan-</sub> ically swept), abdominal (curvilinear 2D) and simulated liver (linear 2D). Rows: or<sub>ically</sub> <sub>swept),</sub> <sub>abdominal</sub> <sub>(curvilinear</sub> <sub>2D)</sub> <sub>and</sub> <sub>simulated</sub> <sub>liver</sub> <sub>(linear</sub> <sub>2D).</sub> <sub>Rows:</sub> <sub>original</sub> image, Hughes shadow reduction, RFlash. Orange arrows mark regions of interest:<sub>image,</sub> <sub>Hughes</sub> <sub>shadow</sub> <sub>reduction,</sub> <sub>RFlash.</sub> <sub>Orange</sub> <sub>arrows</sub> <sub>mark</sub> <sub>regions</sub> <sub>of</sub> <sub>interest:</sub> <sub>in</sub> <sub>the</sub> fetal brain, recovery of the Sylvian fissure in the proximal hemisphere. In the abdo<sub>fetal brain, recovery of the Sylvian fissure in the proximal hemisphere. In the abdominal,</sub> over-brightening of shadowed regions by Hughes reduction, producing cloudy artover-brightening of shadowed regions by Hughes reduction, producing cloudy artefacts that RFlash avoids. In the liver dataset, enhancement artefacts, Hughes method dothat RFlash avoids. In the liver dataset, enhancement artefacts, Hughes method does not remoremove.

Table 1: Mean intensity diference $\Delta \mu$ across datasets for the original (O), Hughes shadowreduced (H), and RFlash (SR) volumes. Values are shown as mean standard deviation. The two right-most columns report the subject-wise diferences relative to RFlash. Diferences marked with <sup>∗</sup> are significantly diferent from 0 in a t-test with a p-value $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { < \frac { 0 . 0 1 } { 6 } = 0 . 0 0 1 6 6 7 } \end{array}$ for multiple-comparison correction.

$$
\Delta \mu _ { O } \downarrow
$$

$$
\Delta \mu _ { H } \downarrow
$$

$$
\Delta \mu _ { S R } \downarrow
$$

$$
( \Delta \mu _ { O } - \Delta \mu _ { S R } )
$$

$$
( \Delta \mu _ { H } - \Delta \mu _ { S R } )
$$

$$
6 4 . 1 2 \pm 2 7 . 2 1
$$

$$
{ \bf 1 3 . 6 1 \pm 7 . 3 7 }
$$

$$
1 6 . 9 3 \pm 1 7 . 4 7
$$

$$
4 7 . 1 9 \pm 2 0 . 0 6 ^ { \ast }
$$

$$
- 3 . 3 2 \pm 1 8 . 4 0 ^ { * }
$$

$$
8 7 . 6 5 \pm 1 6 . 8 4
$$

$$
1 3 . 4 1 \pm 1 0 . 1 3
$$

$$
1 2 9 . 0 4 \pm 7 . 5 9
$$

$$
{ \bf 8 . 0 4 \pm 6 . 4 1 }
$$

$$
4 8 . 2 6 \pm 2 2 . 8 4
$$

$$
7 9 . 6 1 \pm 1 8 . 4 1 ^ { \ast }
$$

$$
5 . 3 7 \pm 1 0 . 3 3 ^ { \ast }
$$

$$
{ \bf 4 1 . 9 4 \pm 1 1 . 7 3 }
$$

$$
8 7 . 1 0 \pm 1 2 . 5 7 ^ { \ast }
$$

$$
6 . 3 2 \pm 2 9 . 5 5 ^ { \ast }
$$

## 5.2. Age Prediction as Clinical Downstream Task

As described above, age prediction was trained on size-normalised distal hemispheres and evaluated separately on the distal and proximal hemispheres. The results are reported in Table 2. RFlash improved prediction accuracy for both hemispheres, including the distal hemisphere used during training. The largest benefit was observed in the proximal hemisphere, which was more strongly afected by shadows. Compared with the original images, RFlash reduced the prediction error in the proximal hemisphere by approximately 5 days relative to no shadow reduction and almost 2 days relative to Hughes shadow reduction, corresponding to relative improvements of 40% and 19%, respectively.

Table 2: Average prediction error ϵ in days for fetal age prediction. Training and validation were performed on the distal hemispheres, while testing was performed on the distal and proximal hemispheres. Prediction errors $\epsilon _ { O } , \epsilon _ { H }$ , and $\epsilon _ { S R }$ are reported as mean  standard deviation. Paired diferences are reported as mean [95% fetus-level cluster-bootstrap confidence interval], accounting for fetuses scanned at multiple gestational ages. Statistics were calculated over 263 scans from 252 fetuses. Statistical significance was assessed using a two-sided Wilcoxon signed-rank test, with diferences from multiple scans of the same fetus first averaged so that each fetus contributed one independent observation. Asterisks mark mean paired diferences significantly diferent from zero according to the Wilcoxon test after Bonferroni correction $\textstyle { \tilde { ( p < \frac { 0 . 0 1 } { 4 } ) } } = 0 . 0 0 2 5 )$
<table><tr><td>Hemisphere</td><td> $\epsilon _ { o }$ </td><td> $\epsilon _ { H }$ </td><td> $\epsilon _ { S R }$ </td><td> $\epsilon _ { O } - \epsilon _ { S R }$ </td><td> $\epsilon _ { H } - \epsilon _ { S R }$ </td></tr><tr><td>Distal</td><td> $4 . 7 0 \pm 3 . 9 6$ </td><td> $4 . 2 7 \pm 3 . 2 4$ </td><td> ${ \bf 3 . 9 6 \pm 2 . 8 9 }$ </td><td> $0 . 7 5 \left[ 0 . 2 7 , 1 . 2 2 \right]$ </td><td> $0 . 3 1 \left[ - 0 . 0 0 , 0 . 6 4 \right]$ </td></tr><tr><td>Proximal</td><td> $1 2 . 6 6 \pm 8 . 7 6$ </td><td> $9 . 4 4 \pm 7 . 8 1$ </td><td> ${ \bf 7 . 6 0 \pm 5 . 7 3 }$ </td><td> $5 . 0 8 [ 4 . 0 0 , 6 . 1 5 ] ^ { * }$ </td><td> $1 . 8 7 [ 0 . 7 6 , 2 . 9 6 ] ^ { * }$ </td></tr></table>

The individual samples showed considerable spread, reflecting diferences in image quality. Figure 8 shows the Bland–Altman plots for the original images, Hughes shadow reduction, and RFlash, plotting the diference between proximal and distal prediction errors against the mean error. As expected, the distal hemisphere was more predictive on average, while RFlash improved predictability in both hemispheres with a stronger efect in the proximal hemisphere. For readability, individual samples are shown only for RFlash, while the other methods are summarised by their convex hulls. The full distributions are provided in Section Appendix A.2. Representative cases where proximal-hemisphere age prediction performed well and less well are shown in Figure 9.

Bland-Altman Plot Comparing Both Hemispheres Across All Three Datasets  
![](images/829a803495e507d37149636722a15e52c282e223f4c58673baeb85477abcd212.jpg)  
Figure 8: Bland–Altman plots for age prediction on both hemispheres across the three datasets: the original images (orig), the images with Hughes shadow reduction (H), and the images with RFlash shadow reduction (SR). For readability, the first two datasets show only the convex hull, mean, and standard deviation. The full results are provided in Section Appendix A.2.

## 5.3. Shadow Confidence Maps

We evaluated the physical plausibility of the predicted shadow confidence maps using established confidence-map baselines. The visual comparison in Figure 10 suggests that the maps from our physics-informed model produce more realistic shadow segmentations.

![](images/a9e0b732e2ff9c8c0e19871709579418414520dd5972b8a3337ffc9e4697a2aa.jpg)  
Figure 9: Representative cases from the age-prediction experiment. Top row: a case where shadow reduction substantially increased contrast in the proximal hemisphere. The green arrow marks the cerebellum, which is visibly better resolved after shadow reduction. Bottom row: a case where a residual grey-value gradient across the hemispheres persists (red box) in the original and Hughes-reduced images but is largely removed by RFlash.

Compared with the physics-informed shadow maps from Wysocki et al. [22], random forests using our shadow maps performed better in all metrics except precision, as shown in Table 3. Figure 11 shows representative random-forest predictions. Although the predictions remain imperfect, those based on our shadow confidence maps are visually more physically plausible because they segment shadows until the lower edge of the frame. Analysis of SHAP [59] feature-importance maps further indicates that the maps from RFlash were more informative to the random forest than those from Yesilkaynak et al. [58], as shown in Figure 12. The colour maps are normalised column-wise to support comparison within each model configuration.

Meng  
![](images/5669eda7d20409b61d558b4ef64634d04da990af1d8780736009c22f7ed68b94.jpg)

Ours  
![](images/58d4525278d32b5d9ba44446c0983ad137706064009be77866f9c8de55fd3a87.jpg)  
Figure 10: Overlay comparison of the shadow confidence maps for two examples. The columns compare the overlays obtained from the method of Qingjie Meng et al. [57] and our method.

Table 3: Performance summary reported to three decimal places, averaged over 100 random forests. The baseline row used the ultrasound image as input, the second row used the shadow maps from Yesilkaynak [58], and the third row used the shadow maps from our method. The best value in each column is highlighted in bold.
<table><tr><td>Name</td><td>Accuracy ↑</td><td>Precision ↑</td><td>Dice ↑</td><td>Hausdorff dist. ↓</td></tr><tr><td>Baseline</td><td> $0 . 9 1 2 \pm 0 . 0 0 6$ </td><td> $0 . 5 8 5 \pm 0 . 0 0 1$ </td><td> $0 . 5 3 2 \pm 0 . 0 0 3$ </td><td> $8 . 1 9 6 \pm 0 . 0 3 4$ </td></tr><tr><td>Yesilkaynak</td><td> $0 . 9 2 2 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 6 9 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 5 1 9 \pm 0 . 0 0 5$ </td><td> $5 . 4 0 4 \pm 0 . 1 8 8$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 9 3 0 \pm 0 . 0 0 5 }$ </td><td> $0 . 6 9 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 5 8 8 \pm 0 . 0 0 4 }$ </td><td> ${ \bf 5 . 1 0 4 \pm 0 . 1 0 7 }$ </td></tr></table>

Original  
Baseline  
Yesil. pred.  
Our prediction  
Yesil. conf.map Our confidence  
![](images/f4dae61ca316b46897f33dfead64827ad82fc29beab8e9ce30e9d991d72d0658.jpg)  
Figure 11: Visual comparison for three selected ultrasound simulations of a liver phantom. The first column shows the ultrasound images with corresponding bone-shadow maps: the first row uses the ground-truth map, while the other rows use random-forest predictions. Yesil. refers to the methods introduced by [58]. The last two columns show confidence maps with the ground-truth outlines.

![](images/ad80da82dd9c3f4076c0738159b5c7931e431d5cf4d582c9c8fd9574d2b9631e.jpg)  
Figure 12: Explainability maps for a single random forest computed from 1000 randomly selected paths using TreeExplainer. The columns correspond to the baseline model using the ultrasound image only, the model using the Yesilkaynak confidence map [58], and our model. The first row shows SHAP feature-importance maps for the ultrasound images, the middle row shows the corresponding colour bars, and the final row shows the feature maps when available. The feature maps show the average importance of the 5  5-pixel patch around the classified centre pixel. The colours are matched within each random tree, that is, within each column. Higher SHAP magnitude on the RFlash confidence map (right) indicates that the random forest relied on it more than on the baseline map.

## 6. Discussion

We presented RFlash, an iterative shadow-reduction method for 2D and 3D ultrasound images. The method separates shadow reduction into two steps. First, the input image is decomposed into physics-informed parameter maps using a diferentiable ultrasound simulation. Second, these maps are used to render a shadow-reduced image. RFlash follows the radiance-field view of image formation [19], but uses a fully explicit representation instead of an MLP. This representation is combined with a physics-informed renderer that models the path of beamformed sound through tissue and estimates attenuation and scatter intensity. After decomposition, shadow-reduced images are rendered by reducing the dependence of each depth on tissue closer to the original transducer position, which can be interpreted as virtually moving the probe through the tissue.

This formulation avoids several limitations of previous shadow-reduction methods. Classical approaches often rely on strong assumptions about the relation between attenuation and scattering [9], while frequency-based techniques require specialised hardware or signal information [11]. In contrast, RFlash is applied as a post-processing method and does not depend on vendor-specific scanner access. This makes it applicable across diferent ultrasound geometries, including 2D and 3D acquisitions. As it is trained on individual images, there is no knowledge transfer between patients, eliminating the risk of hallucinations. Evaluation across these diferent probe geometries and acquisition types supports the robustness of RFlash to variation in the input ultrasound data.

The experiments show that RFlash can reduce shadows visually and quantitatively across diferent ultrasound image types. On real data, it outperformed the Hughes shadow-reduction baseline [9] in the proposed shadow metric and produced more uniform contrast. In fetal brain ultrasound, the method increased the visibility of structures in the proximal hemisphere, which is typically more afected by acoustic shadows. The downstream ageprediction experiment further suggests that the recovered information is clinically relevant: for a model trained on the distal hemisphere, shadow reduction improved prediction from the proximal hemisphere compared with both the original images and the baseline.

We also evaluated the physical consistency of the derived shadow confidence maps. Because these maps are computed directly from the attenuationrelated decomposition, they provide an interpretable estimate of shadow formation. Compared with related confidence maps from Wysocki et al. [22], our maps produced visually and quantitatively stronger guidance for randomforest shadow segmentation. The SHAP analysis further indicated that the random forest assigned more importance to our shadow maps, suggesting that they contain more task-relevant shadow information.

As the method is trained on an individual-image basis, it can only recover structures that are still present in the image. Although this allows for more post-processing, it limits the structures that can be recovered. Applying the method to channel or RF data, which contain more information, may increase its clinical relevance, but reduce its applicability scanners which allow for RF data access. Further it relies on accurate geometry estimation. Quite often, the scanner geometry can be found in the scanners spec sheets. If this is not the case, geometry estimation is needed with is a non-trivial task, as seen for the abdominal dataset. As an iterative method, RFlash is slower than closed-form solutions. Our code has not been optimised for speed, and runtime might be shortened with a hardware-optimised implementation.

## Acknowledgements

The first author was supported by a Clarendon Fund scholarship from the University of Oxford. The work of Ana Namburete was supported by the Gates Foundation. We thank the authors of Meng et al. [50] for providing us with the shadow maps of their fetal head scans.

## Declaration of Interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. A priority application has been filed at the UK Intellectual Property Ofice (UKIPO) for the method described in this paper.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.

During the preparation of this work, the authors used GitHub Copilot (with ChatGPT 5.4 backend) and OpenAI’s Codex (ChatGPT 5.5 and 5.6) to support code snipplets, generate figures from code-generated source files, and assist with the writing process. Generative AI was used exclusively to support implementation and improve the presentation of ideas, methods, and results. It was not used to generate the ideas, innovations, results, conclusions, or structure of the presented work. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] I. Pogledic, K. Mankad, M. Severino, T. Lerman-Sagie, A. Jakab, E. Hadi, A. C. Jansen, N. Bahi-Buisson, N. Di Donato, R. Oegema, C. Mitter, I. Capo, M. T. Whitehead, P. Haldipur, G. Mancini, T. A. G. M. Huisman, A. Righini, B. Dobyns, J. A. Barkovich, N. Jovanov Milosevic, G. Kasprian, M. Lequin, Prenatal assessment of brain malformations on neuroimaging: an expert panel review, Brain (2024) awae253. doi:10.1093/brain/awae253.

[2] T. R. Nelson, D. H. Pretorius, A. Hull, M. Riccabona, M. S. Sklansky, G. James, Sources and impact of artifacts on clinical three-dimensional ultrasound imaging, Ultrasound in Obstetrics & Gynecology 16 (2000) 374–383. doi:10.1046/j.1469-0705.2000.00180.x.

[3] G. Malinger, D. Paladini, K. K. Haratz, A. Monteagudo, G. L. Pilu, I. E. Timor-Tritsch, Isuog practice guidelines (updated): sonographic examination of the fetal central nervous system. part 1: performance of screening examination and indications for targeted neurosonography, Ultrasound in Obstetrics & Gynecology 56 (2020) 476–484. doi:10.1002/uog.22145.

[4] D. Paladini, G. Malinger, R. Birnbaum, A. Monteagudo, G. Pilu, L. J. Salomon, I. E. Timor-Tritsch, Isuog practice guidelines (updated): sonographic examination of the fetal central nervous system. part 2: performance of targeted neurosonography, Ultrasound in Obstetrics & Gynecology 57 (2021) 661–671. doi:10.1002/uog.23616.

[5] A. I. L. Namburete, B. W. Papież, M. Fernandes, M. K. Wyburd, L. S. Hesse, F. A. Moser, L. C. Ismail, R. B. Gunier, W. Squier, E. O. Ohuma, M. Carvalho, Y. Jafer, M. Gravett, Q. Wu, A. Lambert, A. Winsey, M. C. Restrepo-Méndez, E. Bertino, M. Purwar, F. C. Barros, A. Stein, J. A. Noble, Z. Molnár, M. Jenkinson, Z. A. Bhutta, A. T. Papageorghiou, J. Villar, S. H. Kennedy, Normative spatiotemporal fetal brain maturation with satisfactory development at 2 years, Nature 623 (2023) 106–114. doi:10.1038/s41586-023-06630-3.

[6] X. Xu, T. Sanford, B. Turkbey, S. Xu, B. J. Wood, P. Yan, Shadowconsistent semi-supervised learning for prostate ultrasound segmentation, IEEE Transactions on Medical Imaging 41 (2022) 1331–1345. doi:10.1109/TMI.2021.3139999.

[7] I. Hacihaliloglu, Enhancement of bone shadow region using local phasebased ultrasound transmission maps, International Journal of Computer Assisted Radiology and Surgery 12 (2017) 951–960. doi:10.1007/ s11548-017-1556-y.

[8] J. E. Aldrich, Basic physics of ultrasound imaging, Critical care medicine 35 (2007) S131–S137.

[9] D. I. Hughes, F. A. Duck, Automatic attenuation compensation for ultrasonic imaging, Ultrasound in Medicine & Biology 23 (1997) 651– 664. doi:10.1016/S0301-5629(97)00002-1.

[10] B. Knipp, J. Zagzebski, T. Wilson, F. Dong, E. Madsen, Attenuation and backscatter estimation using video signal analysis applied to b-mode images, Ultrasonic Imaging 19 (1997) 221–233. doi:10.1177/ 016173469701900305, pMID: 9447670.

[11] G. Treece, R. Prager, A. Gee, Ultrasound attenuation measurement in the presence of scatterer variation for reduction of shadowing and enhancement, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 52 (2005) 2346–2360. doi:10.1109/TUFFC.2005. 1563279.

[12] S. Yasutomi, T. Arakaki, R. Hamamoto, Shadow detection for ultrasound images using unlabeled data and synthetic shadows, 2019. URL: https://arxiv.org/abs/1908.01439. doi:10.48550/ ARXIV.1908.01439.

[13] S. Yasutomi, T. Arakaki, R. Matsuoka, A. Sakai, R. Komatsu, K. Shozu, A. Dozen, H. Machino, K. Asada, S. Kaneko, A. Sekizawa, R. Hamamoto, M. Komatsu, Shadow estimation for ultrasound images using auto-encoding structures and synthetic shadows, Applied Sciences 11 (2021). URL: https://www.mdpi.com/2076-3417/11/3/1127. doi:10.3390/app11031127.

[14] J. Hewlett E. Melton, J. David J. Skorton, Rational gain compensation for attenuation in cardiac ultrasonography, Ultrasonic Imaging 5 (1983) 214–228. doi:10.1177/016173468300500302, pMID: 6685367.

[15] M. J. A. Girard, N. G. Strouthidis, C. R. Ethier, J. M. Mari, Shadow Removal and Contrast Enhancement in Optical Coherence Tomography Images of the Human Optic Nerve Head, Investigative Ophthalmology & Visual Science 52 (2011) 7738–7748. doi:10.1167/iovs.10-6925.

[16] K. A. Vermeer, J. Mo, J. J. A. Weda, H. G. Lemij, J. F. de Boer, Depth-resolved model-based reconstruction of attenuation coeficients in optical coherence tomography, Biomed. Opt. Express 5 (2014) 322–337. doi:10.1364/BOE.5.000322.

[17] H. Kim, T. Varghese, Hybrid spectral domain method for attenuation slope estimation, Ultrasound in Medicine & Biology 34 (2008) 1808– 1819. doi:10.1016/j.ultrasmedbio.2008.04.011.

[18] Y. Yu, J. Wang, Backscatter-contour-attenuation joint estimation model for attenuation compensation in ultrasound imagery, IEEE Transactions on Image Processing 19 (2010) 2725–2736. doi:10.1109/TIP.2010. 2050636.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, Nerf: representing scenes as neural radiance fields for view synthesis, Communications of the ACM 65 (2021) 99–106.

[20] K. Gao, Y. Gao, H. He, D. Lu, L. Xu, J. Li, Neural radiance fields in 3d vision: A comprehensive review, Computational Visual Media (2026) 1–55. doi:10.26599/CVM.2026.9450535.

[21] P.-H. Yeung, L. S. Hesse, M. Aliasi, M. C. Haak, W. Xie, A. I. Namburete, Sensorless volumetric reconstruction of fetal brain freehand ultrasound scans with deep implicit representation, Medical Image Analysis 94 (2024) 103147. doi:10.1016/j.media.2024.103147.

[22] M. Wysocki, M. F. Azampour, C. Eilers, B. Busam, M. Salehi, N. Navab, Ultra-nerf: Neural radiance fields for ultrasound imaging, in: I. Oguz, J. Noble, X. Li, M. Styner, C. Baumgartner, M. Rusu, T. Heinmann, D. Kontos, B. Landman, B. Dawant (Eds.), Medical Imaging with Deep Learning, volume 227 of Proceedings of Machine Learning Research, PMLR, 2024, pp. 382–401. URL: https://proceedings.mlr.press/ v227/wysocki24a.html.

[23] R. Dagli, A. Hibi, R. Krishnan, P. N. Tyrrell, NeRF-US: Removing ultrasound imaging artifacts from neural radiance fields in the wild, in: K. Deshpande, M. Fiterau, S. Joshi, Z. Lipton, R. Ranganath, I. Urteaga (Eds.), Proceedings of the 9th Machine Learning for Healthcare Conference, volume 252 of Proceedings of Machine Learning Research, PMLR, 2024. URL: https://proceedings.mlr.press/v252/dagli24a.html.

[24] F. Gaits, N. Mellado, A. Basarab, Ultrasound volume reconstruction from 2D Freehand acquisitions using neural implicit representations, in: 21st IEEE International Symposium on Biomedical Imaging (ISBI 2024),

IEEE Signal Processing Society and IEEE Engineering in Medicine and Biology Society, Athènes, Greece, 2024, p. à paraître. URL: https: //hal.science/hal-04480668.

[25] X. Hu, B. Zou, Z. Xiao, Q. Chen, Neural radiance fields for ultrasound imaging with image-to-image refinements, in: 2024 4th International Conference on Neural Networks, Information and Communication Engineering (NNICE), 2024, pp. 355–358. doi:10.1109/NNICE61279.2024. 10498317.

[26] S. Zhang, C. Zhao, B. Ouyang, Hfus-nerf: Hybrid representation for fast ultrasound reconstruction in robotic ultrasound system, in: 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 4092–4098. doi:10.1109/ICRA55743.2025.11128742.

[27] M. C. Eid, P.-H. Yeung, M. K. Wyburd, J. F. Henriques, A. I. Namburete, Rapidvol: Rapid reconstruction of 3d ultrasound volumes from sensorless 2d scans, in: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), 2025, pp. 1–5. doi:10.1109/ISBI60581. 2025.10980994.

[28] Z. Guo, Z. Fang, Z. Fu, Ulre-nerf: 3d ultrasound imaging through neural rendering with ultrasound reflection direction parameterization, 2024. URL: https://arxiv.org/abs/2408.00860. arXiv:2408.00860.

[29] M. Eid, A. Namburete, J. F. Henriques, Ultragauss: Ultrafast gaussian reconstruction of 3d ultrasound volumes, in: C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, A. Faust (Eds.), International Conference on Learning Representations, volume 2026, 2026, pp. 93682–93716. URL: https://proceedings.iclr.cc/paper\_files/paper/2026/ file/9739fdfbecb84b2cab3ba06f3ee5498b-Paper-Conference.pdf.

[30] A. Chen, Z. Xu, A. Geiger, J. Yu, H. Su, Tensorf: Tensorial radiance fields, in: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXII, Springer-Verlag, Berlin, Heidelberg, 2022, pp. 333–350. doi:10.1007/ 978-3-031-19824-3\_20.

[31] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. de Mello, O. Gallo, L. Guibas, J. Tremblay, S. Khamis, T. Karras, G. Wetzstein,

Eficient geometry-aware 3d generative adversarial networks, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16102–16112. doi:10.1109/CVPR52688.2022.01565.

[32] J. C. Bamber, R. J. Dickinson, Ultrasonic b-scanning: a computer simulation, Physics in Medicine & Biology 25 (1980) 463. doi:10.1088/ 0031-9155/25/3/006.

[33] J. Meunier, M. Bertrand, Ultrasonic texture motion analysis: theory and simulation, IEEE Transactions on Medical Imaging 14 (1995) 293– 300. doi:10.1109/42.387711.

[34] B. Bürger, C. Abkai, J. Hesser, Simulation of dynamic ultrasound based on ct models for medical education, Studies in health technology and informatics 132 (2008) 56.

[35] T. Hergum, S. Langeland, E. W. Remme, H. Torp, Fast ultrasound imaging simulation in k-space, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 56 (2009) 1159–1167. doi:10.1109/ TUFFC.2009.1158.

[36] A. Marion, D. Vray, Toward a real-time simulation of ultrasound image sequences based on a 3-d set of moving scatterers, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 56 (2009) 2167–2179. doi:10.1109/TUFFC.2009.1299.

[37] O. Mattausch, O. Goksel, Image-based reconstruction of tissue scatterers using beam steering for ultrasound simulation, IEEE Transactions on Medical Imaging 37 (2018) 767–780. doi:10.1109/TMI.2017.2770118.

[38] S. U. Gjerald, R. Brekken, T. Hergum, J. D’hooge, Real-time ultrasound simulation using the gpu, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 59 (2012) 885–892. doi:10.1109/ TUFFC.2012.2273.

[39] W. Wein, S. Brunke, A. Khamene, M. R. Callstrom, N. Navab, Automatic ct-ultrasound registration for diagnostic imaging and imageguided intervention, Medical Image Analysis 12 (2008) 577–585. doi:10. 1016/j.media.2008.06.006, special issue on the 10th international conference on medical imaging and computer assisted intervention - MICCAI 2007.

[40] Y. C. Law, S. Ullrich, T. Knott, T. Kuhlen, S. Weg, Ultrasound image simulation with gpu-based ray tracing, Virtuelle und Erweiterte Realität 8 (2011) 183–194.

[41] B. Burger, S. Bettinghausen, M. Radle, J. Hesser, Real-time gpu-based ultrasound simulation using deformable mesh models, IEEE Transactions on Medical Imaging 32 (2013) 609–618. doi:10.1109/TMI.2012. 2234474.

[42] M. Salehi, S.-A. Ahmadi, R. Prevost, N. Navab, W. Wein, Patientspecific 3d ultrasound simulation based on convolutional ray-tracing and appearance optimization, in: N. Navab, J. Hornegger, W. M. Wells, A. Frangi (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Springer International Publishing, Cham, 2015, pp. 510–518.

[43] O. Mattausch, O. Goksel, Monte-Carlo Ray-Tracing for Realistic Interactive Ultrasound Simulation, in: S. Bruckner, B. Preim, A. Vilanova, H. Hauser, A. Hennemuth, A. Lundervold (Eds.), Eurographics Workshop on Visual Computing for Biology and Medicine, The Eurographics Association, 2016. doi:10.2312/vcbm.20161285.

[44] O. Mattausch, M. Makhinya, O. Goksel, Realistic ultrasound simulation of complex surface models using interactive monte-carlo path tracing, Computer Graphics Forum 37 (2018) 202–213. doi:10.1111/cgf.13260.

[45] F. Duelmer, M. F. Azampour, M. Wysocki, N. Navab, Ultraray: Introducing full-path ray tracing in physics-based ultrasound simulation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 653–662.

[46] N. Bertramo, G. Duguey, V. Gopalakrishnan, Difus: Diferentiable ultrasound rendering from volumetric imaging, in: D. Ni, A. Noble, R. Huang, W. Xue (Eds.), Simplifying Medical Ultrasound, Springer Nature Switzerland, Cham, 2026, pp. 23–32. doi:10.1007/ 978-3-032-06329-8\_3.

[47] M. Postema, Fundamentals of Medical Ultrasonics, 1 ed., CRC Press, Florence, 2011.

[48] J. Villar, A. T. Papageorghiou, R. Pang, E. O. Ohuma, L. C. Ismail, F. C. Barros, A. Lambert, M. Carvalho, Y. A. Jafer, E. Bertino, M. G. Gravett, D. G. Altman, M. Purwar, I. O. Frederick, J. A. Noble, C. G. Victora, Z. A. Bhutta, S. H. Kennedy, The likeness of fetal growth and newborn size across non-isolated populations in the intergrowth-21st project: the fetal growth longitudinal study and newborn crosssectional study, The Lancet Diabetes & Endocrinology 2 (2014) 781–792. doi:10.1016/S2213-8587(14)70121-4.

[49] J. I. Orlando, S. Vitale, US Simulation & Segmentation: Real and Synthetic Abdominal Ultrasound Scans with Manual Segmentations, 2022. URL: https://www.kaggle.com/datasets/ignaciorlando/ ussimandsegm, kaggle dataset. Accessed: 2026-07-17.

[50] Q. Meng, C. Baumgartner, M. Sinclair, J. Housden, M. Rajchl, A. Gomez, B. Hou, N. Toussaint, V. Zimmer, J. Tan, J. Matthew, D. Rueckert, J. Schnabel, B. Kainz, Automatic shadow detection in 2d ultrasound images, in: Data Driven Treatment Response Assessment and Preterm, Perinatal, and Paediatric Image Analysis, Springer International Publishing, Cham, 2018, pp. 66–75.

[51] Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, Image quality assessment: from error visibility to structural similarity, IEEE Transactions on Image Processing 13 (2004) 600–612. doi:10.1109/TIP.2003.819861.

[52] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch: An Imperative Style, High-Performance Deep Learning Library, in: H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc, E. Fox, R. Garnett (Eds.), Advances in Neural Information Processing Systems 32, Curran Associates, Inc., 2019, pp. 8024–8035.

[53] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, 2017. URL: https://arxiv.org/abs/1412.6980. arXiv:1412.6980.

[54] M. K. Wyburd, L. S. Hesse, M. Aliasi, M. Jenkinson, A. T. Papageorghiou, M. C. Haak, A. I. L. Namburete, Assessment of regional cortical development through fissure based gestational age estimation in 3d

fetal ultrasound, in: C. H. Sudre, R. Licandro, C. Baumgartner, A. Melbourne, A. Dalca, J. Hutter, R. Tanno, E. Abaci Turk, K. Van Leemput, J. Torrents Barrena, W. M. Wells, C. Macgowan (Eds.), Uncertainty for Safe Utilization of Machine Learning in Medical Imaging, and Perinatal Imaging, Placental and Preterm Image Analysis, Springer International Publishing, Cham, 2021, pp. 242–252.

[55] L. S. Hesse, N. K. Dinsdale, A. I. L. Namburete, Prototype learning for explainable brain age prediction, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 7903–7913.

[56] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[57] Q. Meng, M. Sinclair, V. Zimmer, B. Hou, M. Rajchl, N. Toussaint, O. Oktay, J. Schlemper, A. Gomez, J. Housden, J. Matthew, D. Rueckert, J. A. Schnabel, B. Kainz, Weakly supervised estimation of shadow confidence maps in fetal ultrasound imaging, IEEE Transactions on Medical Imaging 38 (2019) 2755–2767. doi:10.1109/TMI.2019.2913311.

[58] V. B. Yesilkaynak, V. G. Duque, M. Wysocki, Y. Velikova, D. Mateus, N. Navab, Ultrasound Confidence Maps with Neural Implicit Representation, Springer Nature Switzerland, 2024, pp. 89–100. URL: http://dx.doi.org/10.1007/978-3-031-66958-3\_7. doi:10. 1007/978-3-031-66958-3\_7.

[59] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, Curran Associates Inc., Red Hook, NY, USA, 2017, p. 4768–4777.

## Appendix A. Additional Results

## Appendix A.1. Additional Numerical Results

Table A.4: Mean intensity diference $\Delta \mu$ across datasets for the original (O), Hughes shadow-reduced (H), and RFlash (SR) volumes when all measures are computed with the RFlash shadow maps. Values are shown as mean standard deviation. The two rightmost columns report the subject-wise diferences relative to RFlash. Diferences marked with <sup>∗</sup> are significantly diferent from 0 in a t-test with a p-value $< \frac { 0 . 0 1 } { 6 } = 0 . 0 0 1 6 6 7$ for multiple-comparison correction.
<table><tr><td>Dataset</td><td> $\Delta \mu _ { O }$ </td><td> $\Delta \mu _ { H }$ </td><td> $\Delta \mu _ { S R }$ </td><td> $( \Delta \mu _ { O } - \Delta \mu _ { S R } )$ </td><td> $( \Delta \mu _ { H } - \Delta \mu _ { S R } )$ </td></tr><tr><td>Syn liver</td><td> $6 4 . 1 2 \pm 2 7 . 2 1$ </td><td> $1 3 . 4 4 \pm 2 4 . 4 6$ </td><td> $1 6 . 9 3 \pm 1 7 . 4 7$ </td><td> $4 7 . 1 9 \pm 2 0 . 0 6 ^ { \ast }$ </td><td> $- 3 . 4 9 \pm 8 . 7 5 ^ { \ast }$ </td></tr><tr><td>Abdomen</td><td> $8 7 . 6 5 \pm 1 6 . 8 4$ </td><td> $2 7 . 9 1 \pm 1 1 . 2 4$ </td><td> $8 . 0 4 \pm 6 . 4 1$ </td><td> $7 9 . 6 1 \pm 1 8 . 4 1 ^ { \ast }$ </td><td> $1 9 . 8 7 \pm 1 1 . 2 5 ^ { \ast }$ </td></tr><tr><td>Fetal Brain</td><td> $1 2 9 . 0 4 \pm 7 . 5 9$ </td><td> $5 8 . 9 0 \pm 1 8 . 1 6$ </td><td> $4 1 . 9 4 \pm 1 1 . 7 3$ </td><td> $8 7 . 1 0 \pm 1 2 . 5 7 ^ { \ast }$ </td><td> $1 6 . 9 5 \pm 1 9 . 3 3 ^ { \ast }$ </td></tr></table>

Table A.5: Mean intensity diference $\Delta \mu$ across datasets for the original (O), Hughes shadow-reduced (H), and RFlash (SR) volumes when all measures are computed with the Hughes shadow maps. Values are shown as mean standard deviation. The two rightmost columns report the subject-wise diferences relative to RFlash. Diferences marked with <sup>∗</sup> are significantly diferent from 0 in a t-test with a p-value $< \frac { 0 . 0 1 } { 6 } = 0 . 0 0 1 6 6 7$ for multiple-comparison correction.
<table><tr><td>Dataset</td><td> $\Delta \mu _ { O }$ </td><td> $\Delta \mu _ { H }$ </td><td> $\Delta \mu _ { S R }$ </td><td> $( \Delta \mu _ { O } - \Delta \mu _ { S R } )$ </td><td> $( \Delta \mu _ { H } - \Delta \mu _ { S R } )$ </td></tr><tr><td> $\operatorname { S y n }$  liver</td><td> $6 2 . 2 3 \pm 9 . 3 7$ </td><td> $1 3 . 6 1 \pm 7 . 3 7$ </td><td> $2 2 . 2 1 \pm 5 . 8 0$ </td><td> $4 0 . 0 2 \pm 6 . 3 5 ^ { \ast }$ </td><td> $- 8 . 5 9 \pm 8 . 8 1 ^ { \ast }$ </td></tr><tr><td>Abdomen</td><td> $7 0 . 2 6 \pm 2 2 . 0 4$ </td><td> $1 3 . 4 1 \pm 1 0 . 1 3$ </td><td> $1 1 . 4 8 \pm 7 . 9 0$ </td><td> $5 8 . 7 8 \pm 2 1 . 1 9 ^ { \ast }$ </td><td> $1 . 9 3 \pm 1 0 . 9 9$ </td></tr><tr><td>Fetal Brain</td><td> $8 8 . 0 1 \pm 1 2 . 2 9$ </td><td> $4 8 . 2 6 \pm 2 2 . 8 4$ </td><td> $1 5 . 2 5 \pm 1 1 . 9 2$ </td><td> $7 2 . 7 5 \pm 1 5 . 3 3 ^ { \ast }$ </td><td> $3 3 . 0 0 \pm 2 4 . 5 2 ^ { \ast }$ </td></tr></table>

![](images/781bd66c452a3373bfd48ffd47681837f76affab037a91bc6c2ed9ed7e73d822.jpg)  
(a) Bland–Altman plot for the original images

![](images/587ad0cbbcbdfb0dd8ff5ea843f7b6e85ab1706b83b04a689af94c75419dc235.jpg)  
(b) Bland–Altman plot for the Hughes-reduced images

![](images/5ceab08a403bfd15884dd3dc131cf8ff1c6f0dac26a479c97fef3d20bfe8821a.jpg)  
(c) Bland–Altman plot for the RFlash images  
Figure A.13: This figure complements Figure 8 by showing the individual data points underlying the convex hulls in that figure.

## Appendix B. Ablation Studies

Appendix B.1. Sensitivity with respect to sampling in simulation space

![](images/e30ad3951503daf12fb394af0832fe41cda486dc7bc3e6af02765477be6909e9.jpg)  
(a) Scaling of SSIM over n.

![](images/ab5a24e659ddbd3a604351eb8d43b06aad7a5fae276491cbd4405016ee3c9e84.jpg)  
(b) Scaling of L<sub>2</sub> over n.

![](images/2fd9732234f6def72ad3e612fbdee95d2fb14645ee8e08eee06abc2b5a10b367.jpg)  
(c) Scaling of runtime over n.  
Figure B.14: Similarity measures between simulated and original volumes as a function of n, where n is the edge length of the sampled cube in simulation space. The red line indicates the chosen value.

![](images/5721ffa77788a5320947f4041d67b368467b14c00df464ba551e421f94b4b839.jpg)  
(a) orig

![](images/fc6e5180c103400ccac4e367e54a78b8d2fd1648f5753c5f034cc794119302df.jpg)  
(b) n=51

![](images/5539c45ce6f6bf90023b1c48d5a8e280005c6be140074768ee2a8d5bab6217c0.jpg)  
(c) n=67

![](images/b1dd0c0d6964d5a5c9cb755542e702647e10d972dbce4974ed4eb9d87df49b4a.jpg)  
(d) n=90

![](images/b5b95ebe6e90e3b0d0c99b892ebae7edfbf0497423da772aa2f711a4d7bdf546.jpg)  
(e) n=120

![](images/74e72a7c9ba2d27a47e8979300873ccb7de1974ecf3578de33d262f938b4cb31.jpg)  
(f) n=160

![](images/b74524012987bda1d3e83732f5cb95e26d4846102ab4ea3b332906fca2cd0e21.jpg)  
(g) n=213

![](images/c1dc44f6d5624e293bd1b8ef0e929310472effdbd8282da1f45c0da91d5e2ce6.jpg)  
(h) n=284  
Figure B.15: This figure shows example volumes reconstructed using diferent densities of sampling in simulation space.

## Appendix B.2. Sensitivity to initial values

![](images/422bc245db48af1d2139082576349dd05c5226aedd8ce5aef3dce8a89039e80f.jpg)  
(a) Heat map $D _ { B } ( P , Q )$ over the tested initial values

![](images/4a02fc9064dbaf3e2e7328d003d8046e16038ad3e6eb28a4569188f1aa6a2092.jpg)  
(b) Heat map $D _ { K L } ( P , Q )$ over the tested initial values

![](images/2f122ebb691dab1df50bc0db88ceceb88cbe58bf7c6f6d26e02020c401d6a689.jpg)

![](images/c4a48fb16c2d4875424c158189a3679b469fe62ef471b89d21c3e32217ba7037.jpg)  
(c) Heat map $L _ { 2 }$ over the tested initial values  
(d) Heat map SSIM over the tested initial values  
Figure B.16: Heat maps of shadow-reduction measures for diferent combinations of initial values. The red box marks the selected combination.

![](images/5dcb97a0043629a3454d24f455acbb1572149d9ffdf65775fd71bdf3a2f38902.jpg)  
Figure B.17: SSIM and $L _ { 2 }$ after convergence for diferent weighting factors. The red line indicates the chosen value.

Appendix $B . 4$ . Convergence behaviour

![](images/15f718215e86d916b9e0053ca750b3bec1c42fe831c1bcba266baf86a87cce4e.jpg)  
Figure B.18: Convergence of the $L _ { 2 }$ and SSIM metrics over epochs. The red line indicates the chosen value.

Appendix B.5. Sensitivity of the shadow metric to the number of histogram buckets  
![](images/5bafcb25af929e425541029bdebd559988e9a2e1f33362eb091dcb40be7272dc.jpg)  
(a) Scaling of the three metrics as a function of the number of buckets on the original dataset.

![](images/07d06ce7f8cf379fde554c070aa4e68f5043d181839b24dd6409c8515f72d78d.jpg)  
(b) Scaling of the three metrics as a function of the number of buckets on the Hughes dataset.

![](images/0cf43ec092da8d8ff1c17eb6113d0a40a7cdf04b8c566d81611b52b50e4fbe65.jpg)  
(c) Scaling of the three metrics as a function of the number of buckets on the RFlash dataset.  
Figure B.19: Mean and confidence intervals for the three metrics described above as a function of the number of buckets used to calculate the histograms. The red line indicates the chosen value.