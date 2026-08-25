# Neural Operator based Multi-Field Reconstruction of Inner Solar Boundary State

Vignesh Kumar Pandian Sathia\*, Reza Mansouri\*, Dustin J. Kempton\*, Pete Riley†, Rafal A. Angryk\*

\*Department of Computer Science, Georgia State University

Atlanta, USA

{spandian1, rmansouri1, dkempton1, rangryk} @gsu.edu

†Predictive Sciences Inc.

San Diego, USA

pete@predsci.com

AbstractThe Solar wind is a continuous flow of charged particles emanating from the solar surface and governed by complex, interacting magnetohydrodynamic processes. Accurate specification of inner-boundary conditions is essential for heliospheric modeling and solar-wind prediction. In many practical applications, only a subset of interacting multi-field variables is directly available, but for a comprehensive view of solar wind prediction and downstream magnetohydrodynamic simulations, a more complete boundary state is required. In this work, we study the problem of learning the multi-field multi-scale solar magnetohydrodynamic state at 30 solar radii (R) using operator learning. Specifically, given the radial velocity and radial magnetic field, we aim to reconstruct the non-radial velocity and magnetic field components, radial and non-radial current density, thermodynamic density, and pressure components. This mapping is highly nonlinear, spatially coupled, and multi-scale, making it a challenging task for data-driven scientific machine learning.

To address this problem, we employ a Local Neural Operator (LocalNO) that learns mappings between input and output function spaces while retaining locality and resolution-awareness. Unlike conventional regression models and autoencoder models, neural operators are better suited for learning structured fieldto-field transformations arising from physical systems. The resulting predictions along with inputs are intended to serve as boundary condition variables for future inner-heliospheric modeling pipelines.

Index Terms—Neural operators, Local neural operator, Magnetohydrodynamics, Boundary-state reconstruction

## I. INTRODUCTION

The solar wind is a stream of charged particles emanating from the solar corona and propagating through the inner heliosphere (up to 5 AU) and outer heliosphere toward the heliopause. Understanding and predicting the solar wind is a fundamental problem in space physics and heliophysics because it governs the transport of solar plasma and magnetic fields through the heliosphere, drives space weather disturbances, and directly impacts planetary magnetospheres, satellite operations, communication systems, and human space exploration. Large-scale heliospheric dynamics are governed by coupled magnetohydrodynamic and plasma transport processes, where flow variables, magnetic fields, density, and pressure evolve jointly across radial distance and time. In practice, however, not all quantities required by these models are directly available, and recovering the full physical state from partial observations remains a challenging inverse problem.

In this work, we focus on the solar magnetohydrodynamic state at 30 $R _ { \odot } { \mathrm { : } }$ an inner boundary that is particularly relevant for heliospheric simulations. We assume that two physically meaningful quantities are available on this spherical surface: radial velocity and radial magnetic field. From these inputs, we seek to reconstruct the remaining boundary variables needed required for downstream heliospheric modeling, namely velocity in the $\theta$ and $\phi$ directions, magnetic field in the $\theta$ and $\phi$ directions, current density in the radial (r), $\theta ,$ and $\phi$ directions, as well as plasma density $\rho$ and pressure $p .$ This is a challenging task for several reasons. First, the target variables are strongly coupled through nonlinear physical structure. Second, the mapping is multi-field and high-dimensional, with each prediction corresponding to a structured field over the entire boundary surface rather than a single scalar. Third, the problem exhibits multi-scale spatial dependence.

Traditional supervised learning methods are not ideally matched to this setting. Convolutional architectures, such as CNNs, are effective at local image-like prediction tasks, but they may be sensitive to discretizations or resolution changes and may struggle to capture the spherical geometry and longitudinal wraparound effectively [1], [2]. Multilayer Perceptrons may be even more limited, since they are not translation invariant, and may even struggle to capture long-range interactions across the domain. More broadly, physical field reconstruction on a spherical boundary is not merely a pixel-to-pixel regression problem; it is an operator-learning problem in which the model must infer a functional transformation from partial boundary information as input to a complete multi-variable state.

Neural operators provide a natural framework for this class of problems. Rather than learning a finite-dimensional mapping tied to a specific discretization, neural operators aim to approximate operators between infinite-dimensional function spaces. This makes them particularly attractive for scientific machine learning, where the input and output are often fields governed by hidden physical processes. Among these methods, Local Neural Operators offer an appealing balance between expressive power and inductive bias. By emphasizing localized interactions while still learning function-space mappings, they are well suited for settings where the target variables depend on local structure but also emerge from broader physical coupling. For solar magnetohydrodynamic state reconstruction, this locality-aware formulation can help exploit the spatial organization present on the spherical boundary without reducing the task to purely pointwise estimation.

This paper formulates solar boundary-state reconstruction as a multi-output operator-learning problem and evaluates Local Neural Operators for this task at 30 $R _ { \odot }$ . Our central premise is that radial velocity and radial magnetic field contain sufficient latent information to reconstruct the missing boundary fields. This problem is closely aligned with the way operational solarwind prediction systems are constructed, like in WSA/ENLIL [3], [4], the near-Sun solar-wind state is specified from available magnetic-field information and then supplied as an innerboundary condition to a heliospheric MHD model. Beyond the immediate predictive task, this learned reconstruction can then serve as a complete boundary condition for downstream heliospheric models, providing a holistic surrogate modeling framework.

The contributions of this work are threefold. First, we formulate solar boundary reconstruction as a multi-output operator learning problem, where two observed radial components are used to predict a nine-channel magnetohydrodynamic target. Second, we focus on Local Neural Operators because they combine function-space learning with a locality bias suited to structured boundary fields. Third, the ablation studies explore the effectiveness and feasibility of various components that best suit the current problem setting of spherical boundary reconstruction. While the present work focuses on learning the boundary mapping itself, it is intended as a foundation for future integration with full inner heliospheric simulation frameworks.

The rest of the paper is organized as follows. Section II reviews prior work in solar wind prediction, scientific surrogate models, and neural operator methods. Section III explores the data transformation and preprocessing steps needed to handle the multi-scale components. Section IV discusses the problem setting, model architecture and training configuration, Section V describes the various experimental setup, evaluation strategy, baseline architectures, and proposed model, Section VI presents ablation studies used to identify suitable configurations for the proposed reconstruction setting. Section VII compares the results and their applicability in this study. To further the scientific research in this domain and share our findings, the code for presented work is available at [5].

## II. RELATED WORK

## A. Data-Driven Modeling in Solar and Space Physics

Physical systems are modeled using two broad paradigms: simulations and inversions. Forward simulations deals with known or assumed physical governing laws and evolving the system forward through numerical solvers or partial differential equations. Models in this paradigm include ballistic extrapolation methods [6], [7], Arge-Pizzo kinematic models [3], [8], HUX [9], [10], MAS [11], ENLIL [12], and other MHD-based heliospheric models. MAS stands for Magnetohydrodynamic Algorithm outside a Sphere and provides a physics based MHD model to simulate the solar corona by solving for the time-dependent resistive thermodynamic MHD equations in 3D spherical coordinates. It provides the training data used in this study because it contains the target variables described in Section III.

The latter paradigm, inversion, aims to infer guiding parameters of the system, from observed quantities. Inversion techniques have become quite useful for surrogate modeling, owing to the computational complexity, time complexity, and time sensitive nature of the problem of the simulation paradigm. Machine learning has become increasingly important for inversion modeling of solar physics and space weather forecasting, where complex nonlinear dynamics, expensive numerical simulations, and limited direct observations motivate surrogate and hybrid modeling approaches. Several inversion techniques have been employed for reconstructing fields from partial states, like Physics-informed Neural Networks [13]–[16], Physics-encoded Neural Networks [17]–[19], Physics-guided Neural Networks [20]–[23], all of which deal with coordinate grids as inputs and predict target variables. Another framework for the problem setting is directly learning the reconstruction maps from input fields using approaches like CNN-based architectures in [24]–[26], used for solar image denoising using CNNs, Autoencoder-based approaches [27] and heliophysics foundation models such as Surya [28] provide additional examples of representation learning for solar data, and U-Net-based architecture have also been used for detection and analysis of solar eruptive events [29] and [26] maps granular structures through semantic segmentation for a potential avenue for linking solar dynamics to surrogate models. In contrast to pixel-wise regression, neural operators can learn field-to-field mappings between functions, as discussed in Section II-C.

Much of this literature focuses on forecasting one or a small number of scalar or field-valued observables from historical measurements or simulation outputs. In contrast, the present problem concerns reconstruction of an entire multifield boundary state from only two observed components on a spherical surface. This distinction is important. Solar wind prediction tasks are often framed as time-series forecasting or regression from synoptic maps to scalar outputs at downstream locations. Our setting instead requires estimating missing physical variables that are distributed across the entire spherical boundary and are coupled through underlying hydrodynamic and magnetic structure. As a result, the task is closer to partial state reconstruction than conventional forecasting. This places it at the intersection of space physics, inverse problems, and structured scientific machine learning.

## B. Surrogate Modeling for Physical Systems

Surrogate models have long been used to accelerate numerical simulation and approximate expensive forward models in physics. Early approaches relied on reduced-order models [30] [31], Gaussian processes [32], or polynomial approximations. A major challenge in surrogate modeling for physical systems is preserving the physical and geometrical structure of the underlying problem. Many physical quantities are not independent they arise from coupled governing equations and conservation constraints. As a result, purely black-box regression may achieve low average error while still producing physically inconsistent fields. For boundary reconstruction problems such as ours, this issue is especially relevant because the predicted variables are intended to serve as inputs to downstream innerheliospheric models.

## C. Neural Operators

Neural operators were introduced to learn mappings between function spaces rather than between fixed-length vectors. This formulation makes them well suited for parametric PDE problems, where the goal is to map an input field, coefficient, or boundary condition to a solution field. A key advantage of operator learning is that it more naturally reflects the mathematical structure of physical modeling: both the inputs and outputs are functions defined on a domain. Compared with standard neural networks, operator learners can better capture global dependencies and may generalize better across discretizations.

Several neural operator architectures have been proposed in recent years. Among the most prominent are the Fourier Neural Operator (FNO) [33], [34], which uses spectral convolution in Fourier space to capture global interactions efficiently, and variants that adapt this idea to different geometries and domains like Spherical Fourier Neural Operator (SFNO) [35], Tensorized Fourier Neural Operator (TFNO) [36], and Geometry-informed Neural Operators [37]. These methods have demonstrated strong performance on a wide range of PDE benchmarks, including fluid flow, thermodynamic equations, and other spatiotemporal systems. However, purely global spectral methods may not always be ideal for problems where local structure is especially important or where the domain geometry introduces anisotropy and localized physical interactions.

Local Neural Operators (LocalNO) [38] was proposed to address some of these limitations by emphasizing localized kernel interactions using differential and integral kernels, while still retaining the operator learning perspective. Local operator formulations learn neighborhood-based interactions that can better capture fine-scale structure, sharp gradient regions, and localized correlations. This makes LocalNO a compelling candidate for solar boundary reconstruction.

## D. Operator Learning for Multi-Field and Partial-State Reconstruction

Most benchmark studies in neural operators consider mappings from one coefficient or forcing field to one solution field. In many scientific applications, however, the task is inherently multi-field: several coupled output variables must be predicted jointly. Joint prediction is often preferable to training separate models because the outputs are physically related and may share latent structure. In our case, the target variables include transverse velocity, transverse magnetic field, current density components, density, and pressure, all of which arise from the same underlying solar plasma state. A multi-output operator learner can exploit these dependencies and potentially produce more consistent reconstructions.

Another closely related theme is partial-state reconstruction, where only a subset of the full system variables is observed and the remainder must be inferred. This setting appears in data assimilation, inverse problems, and reduced sensing scenarios. Although not always framed explicitly as operator learning, many such tasks involve recovering hidden field variables from incomplete measurements. Our formulation can be understood in exactly this way: radial velocity and radial magnetic field are treated as partial observations of the boundary state, and the model learns an operator that reconstructs the missing components.

The present study differs from existing literature in three main ways. First, it focuses on heliospheric boundary reconstruction rather than direct forecasting or full forward simulation. Second, it formulates the task as a multi-field operator learning problem from sparse observed boundary variables to a richer magnetohydrodynamic state. Third, it investigates the use of a Local Neural Operator for this setting, motivated by the need to capture both field-to-field correlation and localized as well as long-range spatial dependence.

## III. DATA PREPARATION

Our dataset contains 616 Carrington rotations between 19 February 1975 and 21 December 2020, covering more than four complete solar cycles. The data come from MAS simulations that use boundary conditions at 30 $R _ { \odot }$ , derived from observations made by three instruments: the Kitt Peak Observatory (KPO), the SOHO/Michelson Doppler Imager (MDI), and the Solar Dynamics Observatory/Helioseismic and Magnetic Imager (HMI). The dataset contains a total of 907 samples from different instruments and Carrington rotations. Each instance consists of 11 solar wind components on the spherical surface grid (θ, φ) with a $( 1 0 9 \times 1 2 8 )$ resolution. These components are the velocity field (v), magnetic field (B), current density (J) in three spherical directions $( r , \theta , \phi )$ and scalar variables mass density (ρ) and the thermal pressure (p).

The target variables are distributed in different physical ranges, suggesting the need for careful data preprocessing and transformation steps for training. Training the models without these transformations may lead to weaker correlation between the input and target variables as discussed in Section VI-B. To address this issue, we apply a sign-preserving nonlinear transformation. The initial data distributions and the distributions after transformation is shown in Figure 1 emphasizing the effect of data transformation.

In particular, for variables whose values can be both positive and negative and exhibit narrow-tailed distributions with magnitude less than 1, we apply a signed square-root transform of the form

$$
\tilde { x } = \mathrm { s i g n } ( x ) \sqrt { | x | } .\tag{1}
$$

![](images/a2d38625bbad2471670c9792be000a767deb5dad31f59b69ccd3aacbe402fb16.jpg)  
Fig. 1: Data distribution. Top row (A) represents the original data, bottom row (B) represents the transformed data with signed square root transformation for variables $v _ { \theta } , v _ { \phi } , { \bf B } _ { \theta } , { \bf B } _ { \phi } , { \bf J } _ { r } , { \bf J } _ { \phi } , \rho ,$ and power transform $( \mathrm { p o w e r } = 1 / 4 )$ for pressure $p .$ No transformation is needed for $\mathbf { J } _ { \theta }$

This transformation is applied to $v _ { \theta } , v _ { \phi } , \mathbf { B } _ { \theta } , \mathbf { B } _ { \phi } , \mathbf { J } _ { r } , \ \mathbf { J } _ { \phi } ,$ and $\rho .$ This transformation expands the relative representation of small and medium-valued regions. Unlike absolute-value transformation, which removes the sign, the signed transform preserves the sign structure of the original field, which is essential for vector quantities such as transverse and longitudinal velocity, transverse and longitudinal magnetic field, and current density. Preserving sign is especially important in this setting because the directionality of the predicted quantities carries physical meaning and influences consistency across the reconstructed multi-field state. As an additional ablation, we also evaluate a signed logarithmic transformation (ln) as discussed in Section VI-B, where signed log transformation is applied to vθ, $v _ { \phi } , \mathbf { B } _ { \theta } , \mathbf { B } _ { \phi } , \mathbf { J } _ { r } ,$ and $\mathbf { J } _ { \phi } .$ , no transformation is applied to $\mathbf { J } _ { \theta } .$ square root transformation is applied to $\rho ,$ and power $( 1 / 4 )$ transformation is applied to $p .$ We use a 90/10 random train/test split grouped by Carrington rotation. All instrument-specific MAS products corresponding to the same Carrington rotation are assigned to the same split, preventing leakage across correlated samples and evaluates the model on genuinely unseen solar-rotation conditions.

After transformation, each output channel is normalized independently using min-max normalization, fitted on training data and transformed on testing data, to align the scale to [0, 1]. Component-wise normalization ensures that all predicted variables contribute more uniformly to the training objective. The normalization stabilizes optimization across multiscale target variables. The predicted normalized target variables are then de-normalized and an inverse transformation is applied and the metrics are scaled to reconstruct actual physical scales in CGS unit system $( c m / s , G , s t a t A / c m ^ { 2 } , g / c m ^ { 3 }$ , and $d y n / c m ^ { 2 }$ for $v , \mathbf { B } , \mathbf { J } , \rho ,$ and $p$ components respectively).

The resulting data pipeline therefore consists of five stages: first, extraction of the relevant boundary fields at 30 $R _ { \odot }$ second, sign-preserving nonlinear transformation of selected variables to improve distributional spread while maintaining directional information; third, per-component normalization to place all target channels on comparable numerical scales; fourth, per-component de-normalization of predicted target variables; fifth, applying inverse transformation and scaling to physical units. This preprocessing pipeline plays an important role in stabilizing optimization and improving the quality of multi-field reconstruction as discussed in Section VI-B.

![](images/67b9dd68098f3039a48755fe5fb5c536eb3a30c7bfc9a551b7de3246197ce55d.jpg)  
Fig. 2: Model Architecture of the proposed methodology employing a common Neural Operator that takes input channels and maps them into latent channels. The output of common NO is fed into task specific heads for correlated variables.

## IV. METHODOLOGY

## A. Problem Formulation and Proposed Architecture

We formulate solar boundary reconstruction as a supervised operator learning problem. Let $\begin{array} { r } { \boldsymbol { u } ~ = ~ \left[ v _ { r } , \mathbf { B } _ { r } \right] } \end{array}$ denote the observed input fields on the spherical boundary at 30 solar radii, where $v _ { r }$ is the radial velocity and $\scriptstyle \mathbf { B } _ { r }$ is the radial magnetic field. The objective is to map these partial observations to the full target state:

$$
y = \left[ v _ { \theta } , v _ { \phi } , { \bf B } _ { \theta } , { \bf B } _ { \phi } , { \bf J } _ { \theta } , { \bf J } _ { \phi } , { \bf J } _ { r } , \rho , p \right] .\tag{2}
$$

The model architecture is shown in Figure 2. The learning task is therefore not a scalar regression problem, but a structured field-to-field mapping in which the model must infer missing physical quantities from only a limited subset of observed variables.

## B. Model Families

To evaluate the suitability of operator learning for this problem, we study three model families: the Spherical Fourier Neural Operator (SFNO), the Tensorized Fourier Neural Operator (TFNO), and the Local Neural Operator (LocalNO).

The SFNO serves as a strong baseline for operator learning in scientific machine learning on spherical domains. It parameterizes the integral kernel in Fourier space and uses spherical convolution to propagate information across the spatial domain. This design allows SFNO to capture long-range interactions efficiently using spherical harmonics and has made it highly successful for learning PDE solution operators of data embedded in spherical space. In our setting, SFNO provides a global operator-learning baseline for mapping radial boundary observations to the missing multi-field magnetohydrodynamic state. The second baseline model for this study is the TFNO architecture that extends the FNO by factorizing the spectral weight tensors, thereby reducing the number of learnable parameters while retaining the overall spectral operator structure. This factorized design makes TFNO attractive when balancing model expressivity and parameter efficiency.

![](images/659111214b3bd1d5aeeb57664c317464811ae8153e4b188eb5d09726b1a3538f.jpg)  
Fig. 3: Component-wise distribution of evaluation metrics for rank ablation. Left is the Mean Square Error (MSE) of the 9 components to measure point-wise reconstruction error. Right is the Peak Signal-to-Noise Ratio (PSNR) to measure the signal normalized reconstruction fidelity.

Our target model class is the LocalNO. Unlike purely global spectral architectures, LocalNO introduces a stronger locality bias by combining localized differential and integral kernels with operator learning in addition to spectral/spherical convolution. This is particularly relevant for solar boundary reconstruction, where the missing target variables may depend strongly on local neighboring structure even though global coupling remains important. We investigate LocalNO variants that use spherical convolution without domain padding, instead of standard spectral convolution, which is designed to better align with the geometry of the solar boundary surface. Since the data live naturally on a spherical manifold rather than a planar Euclidean grid, spherical convolution offers an alternative inductive bias that may better respect the domain structure, making domain padding and spectral convolution less suitable for this setting.

## V. EXPERIMENTS

## A. Experimental Setup

We conduct experiments to evaluate the quality of reconstructing the nine-dimensional solar magnetohydrodynamic target state from $v _ { r }$ and $\mathbf { B } _ { r }$ at 30 $R _ { \odot }$ . The models compared in this study include SFNO, TFNO, and Local Neural Operator variants with spherical convolution without domain padding. SFNO serves as a global spherical baseline, TFNO as a parameter-efficient tensorized spectral baseline, and LocalNO as the locality-aware operator-learning family. We evaluated LocalNO variants with different convolutional modules during model selection. Because the final model uses the spherical convolution setting, which is better aligned with the angular solar boundary grid, we use this configuration for all reported LocalNO results.

All models are trained on the same preprocessed dataset and evaluated using the same protocol to ensure a fair comparison. All models are trained for 100 epochs using the same hyperparameters like n\_modes = (109, 128), Tucker Factorization, batch\_size=16, n\_1ayers=4, and 1r=1e −4. The focus of the experiments is on field reconstruction quality rather than computational speed alone, since the final objective is to obtain accurate boundary conditions for future heliospheric modeling. Because the outputs are multi-field and spatially structured, evaluation must capture both per-sample fidelity and per-component behavior. All models are trained on NVIDIA A40 GPUs with 48 GB of GPU memory, using PyTorch 2.11.0, and the neuraloperator v2.0.0 package for operator models.

## B. Evaluation Metrics

To comprehensively assess reconstruction quality, we report multiple complementary metrics, including Peak Signal-to-Noise Ratio (PSNR), Universal Quality Index (UQI) [39], Earth Mover's Distance (EMD) [40], Anomaly Correlation Coefficient (ACC), Normalized Nash-Sutcliffe Efficiency (NNSE) [41], and Mean Squared Error (MSE). These metrics are computed sample-wise and component-wise. The metrics measure the quality of reconstructing an entire example across all spatial locations, while revealing individual component-wise distribution of the predicted variables. This distinction is important because a model may achieve good average performance overall while failing on specific target components. By reporting both views, we obtain a more complete picture of model behavior.

PSNR measures reconstruction fidelity in terms of signal-tonoise ratio and is especially useful for quantifying pixel-level similarity. UQI captures structural similarity and perceptual consistency, which are important when assessing whether predicted fields preserve spatial organization rather than only numerical closeness. MSE measures the component-wise gridbased reconstruction error.

## VI. ABLATION STUDY

## A. Factorization Rank

To better understand the role of model capacity in tensorized and local operator learning, we conduct an ablation study over rank values. Specifically, we evaluate ranks $r \in$ {0.2, 0.3, 0.4, 0.5, 1.0}.

The purpose of this study is to analyze the trade-off between compression and expressivity. Lower ranks impose a stronger bottleneck on the operator representation, which improves parameter efficiency and regularization but can limit the model's ability to capture the complex coupling among solar hydrodynamic fields. Higher ranks provide greater expressive capacity, but they may reduce the benefits of structured factorization leading to diminished returns or over-parameterization.

Our experiments show that rank 0.4 yields the best overall trade-off across the evaluated metrics. This suggests that the solar boundary reconstruction problem, as shown in Figure 3 and Table I, benefits from a moderate amount of factorized capacity: too little rank underfits the nonlinear field relationships, while too much rank does not translate into corresponding gains in reconstruction quality. The result is particularly important because it indicates the need to balance representational flexibility with inductive regularization rather than simply maximizing capacity.

## B. Data Transformation

An additional experimental factor is the use of signed squareroot transformation and signed log transformation during preprocessing. Without this transformation, variables with highly concentrated or skewed distributions remain difficult to learn even after normalization, motivating nonlinear transformations that improve distributional spread, making it more difficult for the model to learn finer structure in low- to mediummagnitude regions. The signed square-root transform improves the spread of the data while preserving the sign of physically directional quantities. Empirically, this preprocessing improves stability and leads to better reconstruction across multiple target components as shown in Table II. We also compared the performance across the metrics averaged across components for both square-root and log transformation. This highlights the importance of physically informed preprocessing in multi-field scientific learning problems.

## VII. RESULTS AND DISCUSSION

Table III compares the proposed architecture with the SFNO and TFNO baselines. The LocalNO outperforms baselines across reported metrics averaged over each component. The spread is also smaller, suggesting more consistent performance across samples and output components. This supports the hypothesis that locality is a useful inductive bias for reconstructing coupled fields from partial radial observations.

Although SFNO representations are useful for global spherical-domain modeling, they may be less effective when reconstruction depends heavily on localized structures or when the variables have different scales and spatial patterns. The much larger standard deviations for SFNO indicate that its performance is unstable across components. The weaker performance of TFNO compared with LocalNO is consistent with the broader trend that a purely global or tensorized spectral operator may not be sufficient for this inherently spherical domain problem.

Table I and Figure 3 show the effect of factorization rank for the LocalNO. The increase in model performance by increasing rank values from 0.2 to 0.4 suggests limited expressivity and the loss of information from strong compression arising from lower rank values. The poorer performance as we increase the rank from 0.5 up to 1.0 can be attributed to model overparameterization and reduction of reconstruction flexibility

Table II show the effect of data transformation for more effective training. Signed square root transformation seems to perform better across the metrics over signed log transformation. This could be attributed to higher diverging nature of log transformation between positive and negative values resulting in poorer reconstruction of small-magnitude field values.

Although this work presents a promising approach, it has several limitations. First, the presented work is only evaluated on simulations, not real-world data. Second, it does not consider the radiative interactions of the field variables and is beyond the scope of this work. Finally, it does not consider high-resolution simulation due to differences in the spatial resolution and would require additional preprocessing.

## VIII. CONCLUSION

This work studied the problem of reconstructing the solar magnetohydrodynamic boundary state at 30 solar radii from only radial velocity and radial magnetic field. This reconstructed state is intended to serve as a boundary condition for future heliospheric simulations. Our findings indicate that model design choices strongly affect reconstruction quality. In particular, locality-aware operator learning provides a compelling framework for this problem. The rank ablation study further shows that moderate factorization is most effective, with rank 0.4 achieving the best balance between compression and expressive power. In addition, the signed square-root transformation used during preprocessing improves the spread of the data while preserving physically meaningful sign information, which contributes to more stable and effective learning.

TABLE I: Comparison of model performance when trained across factorization rank when using Tucker factorization. The metric values are aggregated over the components. For MSE and EMD [40] metrics, lower values are better, and for ACC, NNSE [41], PSNR, and UQI, higher values are better.
<table><tr><td rowspan="2">Rank</td><td colspan="2">ACC</td><td colspan="2">EMD</td><td colspan="2">MSE</td><td colspan="2">NNSE</td><td colspan="2">PSNR</td><td colspan="2">UQI</td></tr><tr><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td></tr><tr><td>0.2</td><td>0.9833</td><td>0.0241</td><td>0.0097</td><td>0.0194</td><td>0.0070</td><td>0.0192</td><td>0.9676</td><td>0.0430</td><td>37.3505</td><td>9.9128</td><td>0.8873</td><td>0.1108</td></tr><tr><td>0.3</td><td>0.9830</td><td>0.0256</td><td>0.0095</td><td>0.0203</td><td>0.0058</td><td>0.0169</td><td>0.9673</td><td>0.0445</td><td>37.5967</td><td>10.2752</td><td>0.8836</td><td>0.1149</td></tr><tr><td>0.4</td><td>0.9874</td><td>0.0205</td><td>0.0060</td><td>0.0121</td><td>0.0040</td><td>0.0114</td><td>0.9755</td><td>0.0364</td><td>38.6972</td><td>10.0915</td><td>0.9071</td><td>0.0929</td></tr><tr><td>0.5</td><td>0.9874</td><td>0.0203</td><td>0.0063</td><td>0.0127</td><td>0.0043</td><td>0.0126</td><td>0.9746</td><td>0.0381</td><td>38.0046</td><td>9.4680</td><td>0.9081</td><td>0.0914</td></tr><tr><td>1.0</td><td>0.9787</td><td>0.0338</td><td>0.0105</td><td>0.0215</td><td>0.0078</td><td>0.0218</td><td>0.9602</td><td>0.0549</td><td>37.0086</td><td>11.1103</td><td>0.8786</td><td>0.1204</td></tr></table>

TABLE II: Comparison of model performance for ablation study across data transformation strategies. The metric values are aggregated over the components. For MSE and EMD metrics, lower values are better, and for others higher values are better.
<table><tr><td rowspan="2">Transformation</td><td colspan="2">ACC</td><td colspan="2">EMD</td><td colspan="2">MSE</td><td colspan="2">NNSE</td><td colspan="2">PSNR</td><td colspan="2">UQI</td></tr><tr><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td></tr><tr><td>None</td><td>0.9823</td><td>0.0390</td><td>0.0072</td><td>0.0169</td><td>0.0052</td><td>0.0173</td><td>0.9686</td><td>0.0644</td><td>39.6712</td><td>10.6426</td><td>0.8607</td><td>0.1530</td></tr><tr><td>Square-root</td><td>0.9874</td><td>0.0205</td><td>0.0060</td><td>0.0121</td><td>0.0040</td><td>0.0114</td><td>0.9755</td><td>0.0364</td><td>38.6972</td><td>10.0915</td><td>0.9071</td><td>0.0929</td></tr><tr><td>Natural Log</td><td>0.9806</td><td>0.0407</td><td>0.0081</td><td>0.0193</td><td>0.0062</td><td>0.0207</td><td>0.9642</td><td>0.0728</td><td>39.1683</td><td>10.7892</td><td>0.8527</td><td>0.1603</td></tr></table>

![](images/865469112499259fc4aa299986f1b3c70f0e50bbdbc5e423061891b66e9df218.jpg)

(a) Carrington Rotation 1653 during low solar activity.  
![](images/221c4eb01bcf394287c87834cc466dc95626526fef435b0abd10b0d3ad7e4e54.jpg)  
(b) Carrington Rotation 2113 during high solar activity  
Fig. 4: Component-wise reconstruction of solar MHD state variables. In each subfigure, columns represent the target variables. The top row shows the ground truth (GT), the middle row shows the model prediction (P), and the bottom row shows the residual computed as $G T - P$

TABLE III: Comparison of model performance with respect to baseline models. The values are aggregated over components. lower values are preferred for MSE and EMD, higher value is preferred for the remaining metrics.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Params (in M)</td><td colspan="2">ACC</td><td colspan="2">EMD</td><td colspan="2">MSE</td><td colspan="2">NNSE</td><td colspan="2">PSNR</td><td colspan="2">UQI</td></tr><tr><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td></tr><tr><td>SFNO</td><td>31.65</td><td>0.6307</td><td>0.4542</td><td>0.0416</td><td>0.1106</td><td>0.0870</td><td>0.3197</td><td>0.7651</td><td>0.2274</td><td>28.6258</td><td>11.4363</td><td>0.5662</td><td>0.4028</td></tr><tr><td>TFNO</td><td>412.92</td><td>0.9488</td><td>0.0753</td><td>0.0150</td><td>0.0337</td><td>0.0194</td><td>0.0555</td><td>0.9200</td><td>0.0937</td><td>31.8203</td><td>8.4401</td><td>0.8035</td><td>0.1882</td></tr><tr><td>LocalNO</td><td>14.82</td><td>0.9874</td><td>0.0205</td><td>0.0060</td><td>0.0121</td><td>0.0040</td><td>0.0114</td><td>0.9755</td><td>0.0364</td><td>38.6972</td><td>10.0915</td><td>0.9071</td><td>0.0929</td></tr></table>

Overall, these results support the use of operator-learning surrogates for partial boundary state reconstruction in solar physics. In future work, the learned reconstructions can be integrated into downstream heliospheric simulation pipelines, and additional physics-aware constraints may be incorporated to further improve consistency and generalization.

## REFERENCES

[1] R. Mansouri, D. J. Kempton, P. Riley, and R. A. Angryk, "Toward data-driven surrogates of the solar wind with spherical fourier neural operator," in 2025 International Conference on Machine Learning and Applications (ICMLA), pp. 1070–1075, 2025.

[2] R. Mansouri, D. J. Kempton, P. Riley, and R. A. Angryk, "Autoregressive surrogate modeling of the solar wind with spherical fourier neural operator," in 2025 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 1–6, 2025.

[3] C. N. Arge and V. J. Pizzo, "Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates," Journal of Geophysical Research: Space Physics, vol. 105, no. A5, pp. 10465–10479, 2000.

[4] D. Odstrcil, "Modeling 3-d solar wind structure," Advances in Space Research, vol. 32, no. 4, pp. 497–506, 2003.

[5] V. K. P. Sathia, "solar\_wind\_prediction." https://github.com/vignesh-0510/ solar\_wind\_prediction, 2026. GitHub repository, accessed 20 July 2026.

[6] R. Schwenn, Large-Scale Structure of the Interplanetary Medium, pp. 99– 181. Berlin, Heidelberg: Springer Berlin Heidelberg, 1990.

[7] M. Neugebauer, R. J. Forsyth, A. B. Galvin, K. L. Harvey, J. T. Hoeksema, A. J. Lazarus, R. P. Lepping, J. A. Linker, Z. Mikic, J. T. Steinberg, R. von Steiger, Y.-M. Wang, and R. F. Wimmer-Schweingruber, “Spatial structure of the solar wind and comparisons with solar data and models," Journal of Geophysical Research: Space Physics, vol. 103, no. A7, pp. 14587–14599, 1998.

[8] C. N. Arge, D. Odstrcil, V. J. Pizzo, and L. R. Mayer, "Improved method for specifying solar wind speed near the sun," AIP Conference Proceedings, vol. 679, pp. 190–193, 09 2003.

[9] P. Riley and O. Issan, "Using a heliospheric upwinding extrapolation technique to magnetically connect different regions of the heliosphere,' Frontiers in Physics, vol. 9, p. 679497, 2021.

[10] O. Issan and P. Riley, "Theoretical refinements to the heliospheric upwind extrapolation technique and application to in-situ measurements," Frontiers in Astronomy and Space Sciences, vol. 8, p. 795323, 2022.

[11] P. Riley, J. Linker, and Z. Mikić, "An empirically-driven global mhd model of the solar corona and inner heliosphere," Journal of Geophysical Research: Space Physics, vol. 106, no. A8, pp. 15889–15901, 2001.

[12] D. Odstrcil, “Heliospheric 3-d mhd enlil simulations of multi-cme and multi-spacecraft events," Frontiers in Astronomy and Space Sciences, vol. Volume 10 - 2023, 2023.

[13] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational physics, vol. 378, pp. 686–707, 2019.

[14] S. Cuomo, V. S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, and F. Piccialli, “Scientific machine learning through physics-informed neural networks: Where we are and what's next," Journal of Scientific Computing, vol. 92, no. 3, p. 88, 2022.

[15] S. Cai, Z. Mao, Z. Wang, M. Yin, and G. E. Karniadakis, "Physicsinformed neural networks (pinns) for fluid mechanics: A review," Acta Mechanica Sinica, vol. 37, no. 12, pp. 1727–1738, 2021.

[16] S. Shaier, M. Raissi, and P. Seshaiyer, "Data-driven approaches for predicting spread of infectious diseases through dinns: Disease informed neural networks," 2022.

[17] C. Rao, H. Sun, and Y. Liu, "Hard encoding of physics for learning spatiotemporal dynamics," 2021.

[18] X. Guo, C. Hu, Y. Dai, H. Xu, and L. Zeng, “"Learning dense gassolids flows with physics-encoded neural network model," Chemical Engineering Journal, vol. 485, p. 150072, 2024.

[19] Y. Qin, H. Fu, F. Xu, and Y. Jin, “Emwp-rnn: A physics-encoded recurrent neural network for wave propagation in plasmas," IEEE Antennas and Wireless Propagation Letters, vol. 23, no. 1, pp. 219–223, 2023.

[20] S. Pawar, O. San, B. Aksoylu, A. Rasheed, and T. Kvamsdal, "Physics guided machine learning using simplified theories," Physics of Fluids, vol. 33, no. 1, 2021.

[21] H. Robinson, S. Pawar, A. Rasheed, and O. San, "Physics guided neural networks for modelling of non-linear dynamics," Neural Networks vol. 154, pp. 333–345, 2022.

[22] S. A. Faroughi, N. Pawar, C. Fernandes, M. Raissi, S. Das, N. K. Kalantari, and S. K. Mahjour, "Physics-guided, physics-informed, and physics-encoded neural networks in scientific computing," 2023.

[23] S. A. Faroughi, N. M. Pawar, C. Fernandes, M. Raissi, S. Das, N. K. Kalantari, and S. Kourosh Mahjour, "Physics-guided, physicsinformed, and physics-encoded neural networks and operators in scientific computing: Fluid and solid mechanics," Journal of Computing and Information Science in Engineering, vol. 24, p. 040802, 01 2024.

[24] Díaz Baso, C. J., de la Cruz Rodríguez, J., and Danilovic, S., “Solar image denoising with convolutional neural networks," A&A, vol. 629, p. A99, 2019.

[25] L. Xu, W.-Q. Sun, Y.-H. Yan, and W.-Q. Zhang, "Solar image deconvolution by generative adversarial network," Research in Astronomy and Astrophysics, vol. 20, no. 11, p. 170, 2020.

[26] R. Mansouri, R. Angryk, and K. Reardon, “Exploring solar granulation: from imax/sunrise to dkist," The International FLAIRS Conference Proceedings, vol. 38, May 2025.

[27] L. Pinheiro Cinelli, M. Araújo Marins, E. A. Barros da Silva, and S. Lima Netto, Variational Autoencoder, pp. 111–149. Cham: Springer International Publishing, 2021.

[28] S. Roy, J. Schmude, R. Lal, V. Gaur, M. Freitag, J. Kuehnert, T. van Kessel, D. V. Hegde, A. Muñoz-Jaramillo, J. Jakubik, E. Vos, K. Mandal, A. A. Asanjan, J. L. de Sousa Almeida, A. Lin, T. Singh, K. Yang, C. Pandey, J. Hong, B. Aydin, T. Kurth, R. McGranaghan, S. Kasapis, V. Upendran, S. Bahauddin, D. da Silva, N. V. Pogorelov, A. Spalding, C. Watson, M. Maskey, M. Guhathakurta, J. Bernabe-Moreno, and R. Ramachandran, "Surya: Foundation model for heliophysics," 2025.

[29] O. Stepanyuk and K. Kozarev, “Segmentation and tracking of eruptive solar phenomena with convolutional neural networks," Journal of Geophysical Research: Machine Learning and Computation, vol. 3, no. 1, p. e2025JH000728, 2026.

[30] M. Frangos, Y. Marzouk, K. Willcox, and B. van Bloemen Waanders, “Surrogate and reduced-order modeling: a comparison of approaches for large-scale statistical inverse problems," Large-Scale Inverse Problems and Quantification of Uncertainty, pp. 123–149, 2010.

[31] P. Benner, S. Gugercin, and K. Willcox, “A survey of projection-based model reduction methods for parametric dynamical systems," SIAM review, vol. 57, no. 4, pp. 483–531, 2015.

[32] R. B. Gramacy, Surrogates: Gaussian process modeling, design, and optimization for the applied sciences. Chapman and Hall/CRC, 2020.

[33] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, "Fourier neural operator for parametric partial differential equations," arXiv preprint arXiv:2010.08895, 2020.

[34] Y. Du, Q. Li, R. Gnanasambandam, M. Du, H. Wang, and B. Shen, “Global-local fourier neural operator for accelerating coronal magnetic field model," in 2024 IEEE International Conference on Big Data (BigData), pp. 1964–1971, IEEE, 2024.

[35] B. Bonev, T. Kurth, C. Hundt, J. Pathak, M. Baust, K. Kashinath, and A. Anandkumar, "Spherical fourier neural operators: Learning stable dynamics on the sphere," in International conference on machine learning, pp. 2806–2823, PMLR, 2023.

[36] J. Kossaifi, N. Kovachki, K. Azizzadenesheli, and A. Anandkumar, "Multigrid tensorized fourier neural operator for high-resolution pdes," arXiv preprint arXiv:2310.00120, 2023.

[37] Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, and A. Anandkumar, “Geometry-informed neural operator for large-scale 3d pdes," in Advances in Neural Information Processing Systems (A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, eds.), vol. 36, pp. 35836–35854, Curran Associates, Inc., 2023.

[38] M. Liu-Schiaffini, J. Berner, B. Bonev, T. Kurth, K. Azizzadenesheli, and A. Anandkumar, "Neural operators with localized integral and differential kernels," arXiv preprint arXiv:2402.16845, 2024.

[39] Z. Wang and A. Bovik, “A universal image quality index," IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81–84, 2002.

[40] Y. Rubner, C. Tomasi, and L. J. Guibas, “The earth mover's distance as a metric for image retrieval," International Journal of Computer Vision, vol. 40, pp. 99–121, Nov 2000.

[41] J. Nash and J. Sutcliffe, “River flow forecasting through conceptual models part i — a discussion of principles," Journal of Hydrology, vol. 10, no. 3, pp. 282–290, 1970.