# LAST-SR: LAPLACE-INSPIRED STEADY–TRANSIENTCOMPLEX-FREQUENCY DECOMPOSITION FOR SINGLE IMAGESUPER-RESOLUTION

A PREPRINT

Linhao Li Zhaojie Pan Langkun Chen School of Mathematics Southeast University Nanjing, China

September 3, 2026

## ABSTRACT

Single-image super-resolution (SISR) requires global context modeling for structurally consistent reconstruction. Fourier operators are increasingly adopted for global feature modeling. However, their periodic spectral bases constrain the representation of localized aperiodic variations, limiting the recovery of irregular structures and fine details. In dynamical systems, the Laplace neural operator extends Fourier modes to complex frequencies and decomposes the output signal into complementary steady-state and transient responses to jointly model periodic and aperiodic information. We derive, for the first time, an approximate steady–transient decomposition for two-dimensional feature maps, providing an analytical basis for the proposed complex-frequency decomposition. Accordingly, we propose LaST-SR, centered on a Complex-Frequency Decomposition module that couples a global full-spectrum Fourier branch for image-wide dependencies and long-range structural consistency with a window-conditioned local complex-frequency branch for localized, content-dependent aperiodic variations. To fuse the resulting features, we further design a Steady–Transient Collaborative Aggregation module for cross-branch interaction and joint aggregation. Experiments on five benchmarks show that LaST-SR achieves the best PSNR/SSIM among the compared methods for ×2 and ×4 SISR. Ablation studies further validate the effectiveness of the proposed architecture and its key modeling mechanisms.

## 1 Introduction

Single-image super-resolution (SISR) aims to reconstruct a high-resolution (HR) image from a degraded low-resolution (LR) observation, offering a computational alternative when direct acquisition of HR images is impractical. Since the degradation process discards or aliases high-frequency information, multiple HR images may correspond to the same LR observation, rendering SISR an inherently ill-posed inverse problem.

To constrain this ill-posed reconstruction problem, SISR models must infer missing high-frequency information by jointly exploiting local detail cues and long-range structural dependencies. CNN-based methods effectively extract local patterns, but their predominantly local operations make explicit image-wide structural modeling difficult Dong et al. [2015], Kim et al. [2016a,b], Lim et al. [2017], Zhang et al. [2018], Ahn et al. [2018]. Transformer-based methods strengthen contextual interaction through self-attention, yet the quadratic cost of global attention often restricts such interaction to local windows or regions Liang et al. [2021], Chen et al. [2022], Choi et al. [2023], Chen et al. [2023], Wang et al. [2023]. Against this background, Fourier-domain operators have been increasingly adopted for efficient image-wide feature interaction through spectral-domain modeling Zhang et al. [2022], Sinha et al. [2022], Xu et al. [2024], Liu and Tang [2025]. However, their globally supported periodic modes and spatially shared spectral modulation remain insufficiently adaptive to localized, content-dependent aperiodic variations.

Laplace-domain representation offers a principled direction for addressing this limitation. The Laplace transform generalizes the Fourier transform by extending the purely imaginary frequency iω to the complex frequency s = σ + iω, thereby augmenting oscillatory Fourier modes with exponential modulation. Building on this extension, the Laplace neural operator (LNO) employs a pole–residue formulation to decompose the dynamical output signal into complementary steady-state and transient responses, enabling the modeling of non-periodic signals and transient behavior beyond purely oscillatory Fourier modes Cao et al. [2024]. This motivates the use of complex-frequency modeling for SISR, where exponentially modulated modes provide additional flexibility for representing localized aperiodic variations. However, this decomposition is derived for time-dependent dynamical systems, and a corresponding formulation for static two-dimensional feature maps has not yet been established.

To address this gap, we derive an approximate steady–transient decomposition for static two-dimensional feature maps based on a Laplace-domain formulation. The two responses are then interpreted as having complementary spatial roles: the steady response models image-wide dependencies and long-range structural consistency, whereas the transient response captures localized, content-dependent aperiodic variations. Here, steady and transient denote complementary spatial responses rather than temporal dynamics. Guided by this formulation, we propose the Steady–Transient Super-Resolution Network (LaST-SR). Its core Complex-Frequency Decomposition (CFD) module implements the two responses using a global full-spectrum Fourier branch and a window-conditioned local complex-frequency branch, respectively, while the Steady–Transient Collaborative Aggregation (STCA) module enables cross-branch interaction and joint aggregation. The main contributions of this work are summarized as follows:

• To the best of our knowledge, we are the first to derive an approximate steady–transient decomposition for static two-dimensional feature maps, under certain assumptions. This formulation provides a spatial counterpart to the Laplace-domain pole–residue interpretation of time-dependent dynamical systems and a theoretical grounding for Laplace-inspired feature modeling in SISR.

• We propose LaST-SR, a framework centered on the Complex-Frequency Decomposition (CFD) module. CFD models the steady and transient responses via two complementary branches: a global full-spectrum Fourier branch for image-wide dependencies and long-range structural consistency, and a window-conditioned local complex-frequency branch for localized, content-dependent aperiodic variations. The Steady–Transient Collaborative Aggregation (STCA) module further enables cross-branch interaction and joint aggregation of the resulting steady and transient features.

• Extensive experiments on five standard benchmarks demonstrate that LaST-SR achieves the best PSNR/SSIM among the compared methods for both ×2 and ×4 SISR. Ablation studies further validate the proposed decomposition, the complementary roles of the two branches, and the effectiveness of STCA.

## 2 Related Work

## 2.1 Frequency-Domain SISR Methods

Frequency-domain modeling has been increasingly explored in SISR to capture image-wide dependencies efficiently. SwinFIR Zhang et al. [2022] and NL-FFC Sinha et al. [2022] incorporate Fourier convolution to enhance global feature interaction. FSR Li et al. [2023] combines transform-domain global features with local spatial features and processes different wavelet subbands using capacity-adaptive branches. FDSR Xu et al. [2024] decomposes frequency components for progressive reconstruction, whereas DiffFNO Liu and Tang [2025] introduces a weighted Fourier neural operator for arbitrary-scale SISR. Despite their different implementations, these methods largely rely on globally shared spectral modulation or predefined frequency decomposition, leaving localized and content-dependent aperiodic variations insufficiently modeled.

## 2.2 Complex-Frequency Modeling

Complex-exponential models extend purely oscillatory Fourier modes with exponential modulation. Classical methods, including Prony-type methods, Matrix Pencil, and ESPRIT, estimate poles and modal coefficients for signal decomposition and reconstruction Hu et al. [2013], Hua and Sarkar [1990], Roy and Kailath [1989]. In neural operator learning, LNO Cao et al. [2024] introduces a learnable pole–residue parameterization to represent Laplace-domain kernels and separates dynamical responses into steady-state and transient components. However, the steady-state–transient formulation in LNO is developed for temporal signals and time-dependent systems, while its adaptation to static two-dimensional feature modeling in SISR remains underexplored.

## 3 Theoretical Analysis

Let $\mathbf { X } \in \mathbb { R } ^ { B \times H \times W \times C _ { \mathrm { m i d } } }$ denote an intermediate feature tensor. For the b-th sample, we represent the discrete feature vectors ${ \bf X } _ { b , h , w , : }$ as samples of a continuous vector-valued function $\mathbf { v } _ { b } : \Omega  \mathbb { R } ^ { - _ { \mathrm { m i d } } } , \mathbf { v } _ { b } ( x _ { w } , y _ { h } ) = \mathbf { X } _ { b , h , w , : } \mathrm { ~ : ~ }$ , where the regular grid points $( x _ { w } , y _ { h } ) \in \Omega$ , and $\Omega = [ 0 , T _ { x } ) \times [ 0 , T _ { y } )$ . For notational simplicity, we fix one sample and one feature channel, denote the corresponding scalar component by $v : \Omega \to \mathbb { R }$

To characterize the global spectral structure of v, we adopt a truncated Fourier representation on the finite spatial domain under periodic extension:

$$
v ( x , y ) \approx \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \alpha _ { p q } \exp ( \mathrm { i } \omega _ { p } ^ { x } x ) \exp ( \mathrm { i } \omega _ { q } ^ { y } y ) ,\tag{1}
$$

where $\alpha _ { p q } \in \mathbb { C } , \omega _ { p } ^ { x } = 2 \pi p / T _ { x }$ , and $\omega _ { q } ^ { y } = 2 \pi q / T _ { y }$ . Since v is real-valued, the Fourier coefficients satisfy $\alpha _ { - p , - q } =$ $\overline { { \alpha _ { p q } } }$

Let u denote the response of a translation-invariant linear convolution operator with a learnable kernel κ acting on v:

$$
u ( x , y ) = ( \kappa * v ) ( x , y ) , \qquad ( x , y ) \in \Omega .\tag{2}
$$

Following the complex-exponential modeling paradigm of Prony-SS Hu et al. [2013], we parameterize the learnable kernel as

$$
\kappa ( x , y ) = \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \mathcal { R } _ { m n } \exp ( \mu _ { m } ^ { x } x + \mu _ { n } ^ { y } y ) ,\tag{3}
$$

where $\mu _ { m } ^ { x } , \mu _ { n } ^ { y } \in \mathbb { C }$ are learnable complex poles along the two spatial directions and $\mathcal { R } _ { m n } \in \mathbb { C }$ is the residue coefficient associated with each pole pair. Under a positive-quadrant extension of v and a first-quadrant-supported kernel $\kappa ,$ the two-dimensional one-sided Laplace convolution theorem gives $U ( s _ { x } , s _ { y } ) = K ( s _ { x } , \bar { s _ { y } } ) V ( s _ { x } , s _ { y } )$ . This product yields four pole families after partial-fraction expansion. Retaining the Fourier–Fourier and complex-pole–complex-pole families while collecting the two mixed families into an approximation residual gives

$$
\begin{array} { r } { u ( x , y ) \approx \displaystyle \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \lambda _ { p q } \exp ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y ) } \\ { + \displaystyle \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \gamma _ { m n } \exp ( \mu _ { m } ^ { x } x + \mu _ { n } ^ { y } y ) . } \end{array}\tag{4}
$$

The first term denotes the steady response $u _ { \mathrm { s s } } ( x , y )$ , and the second denotes the transient response $u _ { \mathrm { t r } } ( x , y )$ , where $\lambda _ { p q }$ and $\gamma _ { m n }$ are their corresponding effective coefficients.

Remark. The exact two-dimensional expansion also contains Fourier–complex-pole and complex-pole–Fourier families, which are not assumed to vanish. Equation (4) retains the two same-type families and collects the mixed families into a residual, yielding a task-oriented dual-response approximation. See Sec. A.2 of the technical supplement for the complete derivation.

In SISR, we assign complementary functional roles to the two retained responses. The steady response uses globally supported oscillatory modes to capture image-wide dependencies and long-range structural consistency. The transient response employs general complex-frequency modes to represent content-dependent aperiodic variations, while the windowed realization introduced below further provides local adaptation.

Under the periodic-extension convention, the periodic counterpart of Eq. (2) admits a mode-wise Fourier spectralmultiplier form with the same basis family as the steady response in Eq. (4). This correspondence motivates global Fourier-domain steady modeling; see Sec. A.3 of the technical supplement.

For noncoincident Fourier and learnable poles, residue evaluation yields the coefficient of the retained transient response as

$$
\gamma _ { m n } = \mathcal { R } _ { m n } \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \frac { \alpha _ { p q } } { ( \mu _ { m } ^ { x } - \mathrm { i } \omega _ { p } ^ { x } ) ( \mu _ { n } ^ { y } - \mathrm { i } \omega _ { q } ^ { y } ) } .\tag{5}
$$

As shown in Table 1, direct full-map evaluation materializes large complex-valued intermediate tensors and incurs substantial peak memory. Moreover, modal coefficients derived from the full-map spectrum are shared across all spatial locations, limiting their adaptation to region-specific variations. We therefore implement the transient response within normalized local windows, sharing poles and residues across windows while deriving window-specific modal coefficients from each local spectrum.

![](images/10688e6a79f567c3ce78fafacf9c4c09d5dc79feea4b75cde99d94f21b069a56.jpg)  
Figure 1: Overall architecture of LaST-SR. A shallow feature extractor is followed by D STAR blocks and a reconstruction module. Each STAR block comprises PDPM for branch-specific projection, CFD for global steady and windowed local transient modeling, and STCA for bidirectional cross-branch interaction and aggregation. The lower-left panel illustrates the Laplace-domain pole–residue interpretation: poles on the imaginary axis induce purely oscillatory steady responses, whereas poles with nonzero real parts yield transient responses with exponential envelopes.

<table><tr><td>Method</td><td>Peak Mem. (MB)</td><td>Latency (ms)</td></tr><tr><td>Global</td><td>≥ 4,529.85†</td><td>OOM</td></tr><tr><td>Windowed (Ours)</td><td>66.79</td><td>0.856</td></tr></table>

Table 1: Operator-level memory and latency comparison between the global and windowed transient implementations under the ×4 SR setting, using an LR feature resolution corresponding to a 720p (1280 × 720) output. Both implementations have identical parameter counts and leading-order arithmetic complexity. † The reported global memory is a conservative analytical lower bound determined by its largest complex-valued intermediate tensor, as the full-map forward pass did not complete.

## 4 LaST-SR Architecture

Guided by the approximate steady–transient decomposition, we develop LaST-SR, whose overall architecture is illustrated in Fig. 1. Its core CFD module models the two response families using a global full-spectrum Fourier branch and a window-conditioned local complex-frequency branch, respectively. PDPM provides branch-specific low-dimensional representations before CFD, while STCA enables cross-branch interaction and joint aggregation.

LaST-SR comprises shallow feature extraction, deep feature modeling with D sequential Steady–Transient Attention Representation (STAR) blocks, and high-resolution reconstruction:

$$
\begin{array} { r l } & { X _ { 0 } = H _ { \mathrm { S F } } ( I _ { \mathrm { L R } } ) , } \\ & { X _ { d } = H _ { \mathrm { S T A R } } ^ { ( d ) } ( X _ { d - 1 } ) + X _ { d - 1 } , \qquad d = 1 , \ldots , D , } \\ & { I _ { \mathrm { S R } } = H _ { \mathrm { R e c } } ( X _ { D } + X _ { 0 } ) , } \end{array}\tag{6}
$$

where $H _ { \mathrm { S F } }$ is a $3 \times 3$ convolution, $H _ { \mathrm { S T A R } } ^ { ( d ) }$ denotes the d-th STAR block, and $H _ { \mathrm { R e c } }$ consists of a $3 \times 3$ convolution followed by PixelShuffle. Each STAR block contains PDPM, CFD, and STCA: PDPM produces branch-specific low-dimensional features, CFD models the global steady and windowed local transient responses, and STCA enables their cross-branch interaction and joint aggregation.

## 4.1 PDPM: Pre-Decomposition Projection Module

PDPM generates low-dimensional feature representations for the steady and transient branches. Given $X _ { \mathrm { i n } } ~ \in$ $\mathbb { R } ^ { B \times H W \times C }$ , an independent two-layer pointwise projection is applied to each branch $\tau \in \{ \mathrm { s s } , \mathrm { t r } \}$

$$
Z _ { \tau } = \phi \Bigl ( \mathcal { P } _ { \tau } ^ { ( 2 ) } \left( \phi \Bigl ( \mathcal { P } _ { \tau } ^ { ( 1 ) } \left( \mathrm { L N } ( X _ { \mathrm { i n } } ) \right) \Bigr ) \right) \Bigr ) ,\tag{7}
$$

where LN denotes LayerNorm, ϕ denotes LeakyReLU, and $\mathcal { P } _ { \tau } ^ { ( i ) }$ denotes a pointwise channel projection implemented by $\mathbf { a \ 1 \times 1 }$ convolution. The two branches do not share parameters and produce $Z _ { \tau } \in \mathbb { R } ^ { B \times H W \times C _ { \mathrm { m i d } } ^ { \mathbf { \lambda } } }$ , where $\dot { C } _ { \mathrm { m i d } } < C$ The steady representation $Z _ { \mathrm { s s } }$ is reshaped into $X _ { \mathrm { s s } } \in \mathbb { R } ^ { B \times C _ { \mathrm { m i d } } \times H \times W }$ to enable global steady modeling. In parallel, $Z _ { \mathrm { t r } }$ is partitioned into non-overlapping $\bar { M } \times M$ windows, yielding $X _ { \mathrm { t r } } \in \mathbb { R } ^ { B N _ { w } \times C _ { \mathrm { m i d } } ^ { \smile } \times M \times M }$ for local transient modeling, where $N _ { w } = H W / M ^ { 2 }$

## 4.2 CFD: Complex-Frequency Decomposition

CFD instantiates the steady and transient responses in Eq. (4) using a global full-spectrum Fourier branch and a windowed local complex-frequency branch, respectively.

The lower-left panel of Fig. 1 visualizes a response slice by fixing $s _ { y } = s _ { y _ { 0 } } = 0 . 1 5 + 3 . 8 0 \mathrm { i }$ i and varying $s _ { x } = \sigma + \mathrm { i } \omega$ For a complex pole $\mu ,$ the inverse-transform basis is $e ^ { \mu x } = e ^ { \mathrm { R e } ( \mu ) x } e ^ { \mathrm { i } \mathrm { I m } ( \bar { \mu } ) x }$ . Fourier poles lie on the imaginary axis and yield purely oscillatory modes, whereas learnable poles may have nonzero real parts that introduce spatial growth or decay envelopes. The surface and scatter plots show the response magnitude and pole distribution, respectively.

## 4.2.1 Global Steady Branch

Given $X _ { \mathrm { s s } } .$ , the steady branch first computes $\widehat { X } _ { \mathrm { s s } } ( \xi ) = \mathcal { F } ( X _ { \mathrm { s s } } ) ( \xi )$ , where ξ denotes a two-dimensional discrete frequency. To preserve global spectral organization and fine-scale structural components, we retain and modulate the complete discrete spectrum. The modulated spectrum is given by

$$
\begin{array} { r } { \widehat { X } _ { s } ( \xi ) = P _ { \mathrm { s s } } \Big [ w ( \xi ) \widehat { X } _ { \mathrm { s s } } ( \xi ) \Big ] , } \end{array}\tag{8}
$$

where $P _ { \mathrm { s s } } \in \mathbb { C } ^ { C _ { \mathrm { m i d } } \times C _ { \mathrm { m i d } } }$ is a complex-valued channel-mixing matrix shared across frequencies, and $w ( \xi ) = 1 + \eta \| \xi \| ^ { \epsilon }$ is a frequency-dependent weight with learnable scalar η and fixed exponent ϵ. The steady feature is obtained by

$$
X _ { s } = \mathrm { R e } \left[ \mathcal { F } ^ { - 1 } ( \widehat { X } _ { s } ) \right] .\tag{9}
$$

## 4.2.2 Local Transient Branch

The local transient branch implements the transient response in $\operatorname { E q . } \left( 4 \right)$ by discretizing the coefficient relation in Eq. (5) within normalized local windows. Complex poles and residue parameters are shared across windows, while modal coefficients are dynamically generated from each local spectrum. For the r-th window, the local Fourier coefficients are computed as

$$
\alpha ^ { ( r ) } ( \omega ^ { x } , \omega ^ { y } ) = \mathscr { F } ( X _ { \mathrm { t r } } ^ { ( r ) } ) ( \omega ^ { x } , \omega ^ { y } ) ,\tag{10}
$$

where $\alpha ^ { ( r ) } \in \mathbb { C } ^ { C _ { \mathrm { m i d } } \times M \times M }$ , and $( \omega ^ { x } , \omega ^ { y } )$ denotes the discrete frequency grid within the window.

Learnable complex poles and residues are parameterized for each input–output channel pair:

$$
\begin{array} { r l } & { \mu ^ { x } \in \mathbb { C } ^ { C _ { \operatorname* { m i d } } \times C _ { \operatorname* { m i d } } \times K _ { x } } , \qquad \mu ^ { y } \in \mathbb { C } ^ { C _ { \operatorname* { m i d } } \times C _ { \operatorname* { m i d } } \times K _ { y } } , } \\ & { \mathcal { R } \in \mathbb { C } ^ { C _ { \operatorname* { m i d } } \times C _ { \operatorname* { m i d } } \times K _ { x } \times K _ { y } } , } \end{array}\tag{11}
$$

where $K _ { x }$ and $K _ { y }$ denote the numbers of complex-frequency modes along the two spatial directions. For channel pair $( c , d )$ and mode pair $( m , n )$ , the pole–residue spectral response is

$$
\Phi _ { c , d , m , n } ( \omega ^ { x } , \omega ^ { y } ) = \frac { \mathcal { R } _ { c , d , m , n } } { ( \mu _ { c , d , m } ^ { x } - \mathrm { i } \omega ^ { x } ) ( \mu _ { c , d , n } ^ { y } - \mathrm { i } \omega ^ { y } ) } .\tag{12}
$$

The modal coefficients for the r-th window are computed as

$$
\gamma _ { c , d , m , n } ^ { ( r ) } = \sum _ { \omega ^ { x } , \omega ^ { y } } \alpha _ { c } ^ { ( r ) } ( \omega ^ { x } , \omega ^ { y } ) \Phi _ { c , d , m , n } ( \omega ^ { x } , \omega ^ { y } ) .\tag{13}
$$

The transient response in the r-th window is reconstructed over normalized local coordinates $( \tilde { x } , \tilde { y } ) \in [ 0 , 1 ] ^ { 2 }$

$$
X _ { t , d } ^ { ( r ) } ( \tilde { x } , \tilde { y } ) = \mathrm { R e } \left[ \sum _ { c , m , n } \gamma _ { c , d , m , n } ^ { ( r ) } \exp \Bigl ( \mu _ { c , d , m } ^ { x } \tilde { x } + \mu _ { c , d , n } ^ { y } \tilde { y } \Bigr ) \right] .\tag{14}
$$

The window responses are reassembled into $X _ { t } \in \mathbb { R } ^ { B \times C _ { \mathrm { m i d } } \times H \times W }$ . The imaginary and real pole components govern local oscillation and spatial growth or decay, respectively. Since image coordinates are spatial rather than temporal, we do not impose a negative-real-part constraint. Instead, normalized local coordinates mitigate coordinate-range-induced exponential amplification.

## 4.3 STCA: Steady–Transient Collaborative Aggregation

STCA constructs shared guidance from the steady and transient features and symmetrically refines both branches. Given the CFD outputs $X _ { s }$ and $X _ { t }$ , each branch is first enhanced by window-based self-attention:

$$
\begin{array} { r } { S = X _ { s } + \mathrm { W M S A } _ { \delta } ( \mathrm { L N } ( X _ { s } ) ) , } \\ { T = X _ { t } + \mathrm { W M S A } _ { \delta } ( \mathrm { L N } ( X _ { t } ) ) . } \end{array}\tag{15}
$$

Here, δ alternates between 0 and $\lfloor M / 2 \rfloor$ across successive STAR blocks, corresponding to regular and shifted windows, respectively. The enhanced branches are fused to form a shared guidance feature:

$$
G = { \mathrm { W M S A } } _ { \delta } ( { \mathcal { C } } _ { 3 \times 3 } ( S + T ) ) .\tag{16}
$$

Using $S$ and T as branch-specific queries and G as the shared key and value, window-based cross-attention is performed as

$$
\begin{array} { r } { \widehat { S } = \mathrm { W C A } _ { \delta } ( S , G ) , } \\ { \widehat { T } = \mathrm { W C A } _ { \delta } ( T , G ) , } \end{array}\tag{17}
$$

where $\mathrm { W C A } _ { \delta } ( \mathbf { U } , \mathbf { Z } )$ uses U as the query and Z as both the key and value. After feed-forward refinement, the two branches are restored to the spatial layout and aggregated:

$$
\begin{array} { r l } & { S ^ { \prime } = \widehat { S } + \mathrm { F F N } \Big ( \mathrm { L N } ( \widehat { S } ) \Big ) , } \\ & { T ^ { \prime } = \widehat { T } + \mathrm { F F N } \Big ( \mathrm { L N } ( \widehat { T } ) \Big ) , } \\ & { F = \mathrm { W i n R e v } _ { \delta } ( S ^ { \prime } ) + \mathrm { W i n R e v } _ { \delta } ( T ^ { \prime } ) . } \end{array}\tag{18}
$$

The fused feature is projected back to the original channel dimension:

$$
X _ { \mathrm { f u s e } } = \phi \Big ( \mathcal { P } _ { f } ^ { ( 2 ) } \Big ( \phi \Big ( \mathcal { P } _ { f } ^ { ( 1 ) } ( F ) \Big ) \Big ) \Big ) ,\tag{19}
$$

where $\mathcal { P } _ { f } ^ { ( 1 ) }$ and $\mathcal { P } _ { f } ^ { ( 2 ) }$ are $1 \times 1$ convolutions that jointly restore the channel dimension from $C _ { \mathrm { m i d } }$ to $C$ . Finally, an overlapping cross-attention block (OCAB) Chen et al. [2023] alleviates window-boundary effects, followed by channel-attention recalibration:

$$
\begin{array} { r } { Y = \mathrm { O C A B } ( X _ { \mathrm { f u s e } } ) , } \\ { X _ { \mathrm { o u t } } = Y + \mathrm { C h A t t n } ( Y ) . } \end{array}\tag{20}
$$

Through shared guidance, STCA jointly refines image-wide structural information and localized aperiodic variations.

## 4.4 Training Objective

We combine spatial reconstruction and Fourier-magnitude consistency losses:

$$
\begin{array} { r l r } & { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s p a t i a l } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f r e q } } , } & \\ & { \mathcal { L } _ { \mathrm { s p a t i a l } } = \displaystyle \frac { 1 } { N } \left. I _ { \mathrm { S R } } - I _ { \mathrm { H R } } \right. _ { 1 } , } & \\ & { \mathcal { L } _ { \mathrm { f r e q } } = \displaystyle \frac { 1 } { N _ { f } } \left. | \mathcal { F } _ { 2 D } ( I _ { \mathrm { S R } } ) | - | \mathcal { F } _ { 2 D } ( I _ { \mathrm { H R } } ) | \right. _ { 1 } , } & \end{array}\tag{21}
$$

where $N$ and $N _ { f }$ are the numbers of spatial and frequency elements, and $\lambda _ { f }$ weights the frequency loss. The two terms provide complementary pixel- and spectral-domain supervision for the final reconstruction.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Scale Params. (K)</td><td>Set5</td><td>Set14</td><td>BSD100</td><td>Urban100</td><td>Manga109</td></tr><tr><td>PSNR/SSIM</td><td>PSNR/SSIM</td><td>PSNR/SSIM</td><td>PSNR/SSIM</td><td>PSNR/SSIM</td></tr><tr><td>CARN</td><td>1,592</td><td>37.76/0.9590</td><td>33.52/0.9166</td><td>32.09/0.8978</td><td>31.92/0.9256</td><td>38.36/0.9754</td></tr><tr><td>LAPAR-A</td><td>548</td><td>38.01/0.9605</td><td>33.62/0.9183</td><td>32.19/0.8999</td><td>32.10/0.9283</td><td>38.67/0.9772</td></tr><tr><td>DeFiANs</td><td>1,028</td><td>38.03/0.9605</td><td>33.63/0.9181</td><td>32.32/0.8999</td><td>32.20/0.9286</td><td>38.91/0.9775</td></tr><tr><td>SwinIR-light</td><td>910</td><td>38.14/0.9611</td><td>33.86/0.9206</td><td>32.31/0.9012</td><td>32.76/0.9340</td><td>39.12/0.9783</td></tr><tr><td>SwinIR-NG</td><td>1,181</td><td>38.17/0.9612</td><td>33.94/0.9205</td><td>32.31/0.9013</td><td>32.78/0.9340</td><td>39.20/0.9781</td></tr><tr><td>SwinFIR-T</td><td>872</td><td>38.22/0.9614</td><td>34.03/0.9215</td><td>32.37/0.9020</td><td>33.03/0.9362</td><td>39.30/0.9785</td></tr><tr><td>Omni-SR</td><td>×2 772</td><td>38.22/0.9613</td><td>33.98/0.9210</td><td>32.36/0.9020</td><td>33.05/0.9363</td><td>39.28/0.9784</td></tr><tr><td>FDSR</td><td>6,290</td><td>38.20/0.9613</td><td>33.90/0.9200</td><td>32.33/0.9011</td><td>32.72/0.9336</td><td>39.09/0.9781</td></tr><tr><td>SRConvNet-L</td><td>885</td><td>38.14/0.9610</td><td>33.81/0.9199</td><td>32.28/0.9010</td><td>32.59/0.9321</td><td>39.22/0.9779</td></tr><tr><td>CRAFT</td><td>737</td><td>38.23/0.9615</td><td>33.92/0.9211</td><td>32.33/0.9016</td><td>32.86/0.9343</td><td>39.39/ 0.9786</td></tr><tr><td>LAMNet-large</td><td>1,024</td><td>38.27/0.9615</td><td>34.07/0.9214</td><td>32.38/0.9023</td><td>33.16/0.9373</td><td>39.38/0.9785</td></tr><tr><td>MaIR-Small</td><td>1,355</td><td>38.20/0.9611</td><td>33.91/0.9209</td><td>32.34/0.9016</td><td>32.97/0.9359</td><td>39.32/0.9779</td></tr><tr><td>Ours</td><td>1,329</td><td>38.32 0.9617</td><td>34.13 0.9225</td><td>32.41 0.9026</td><td>33.25 0.9380</td><td>39.43 0.9786</td></tr><tr><td>CARN</td><td>1,592</td><td>32.13/0.8937</td><td>28.60/0.7806</td><td>27.58/0.7349</td><td>26.07/0.7837</td><td>30.45/0.9073</td></tr><tr><td>LAPAR-A</td><td>659</td><td>32.15/0.8944</td><td>28.61/0.7818</td><td>27.61/0.7366</td><td>26.14/0.7871</td><td>30.42/0.9074</td></tr><tr><td>DeFiANs</td><td>1,065</td><td>32.16/0.8942</td><td>28.63/0.7810</td><td>27.58/0.7363</td><td>26.10/0.7862</td><td>30.59/0.9084</td></tr><tr><td>SwinIR-light</td><td>930</td><td>32.44/0.8976</td><td>28.77/0.7858</td><td>27.69/0.7406</td><td>26.47/0.7980</td><td>30.92/0.9151</td></tr><tr><td>SwinIR-NG</td><td>1,201</td><td>32.44/0.8980</td><td>28.83/0.7870</td><td>27.73/0.7418</td><td>26.61/0.8010</td><td>31.09/0.9161</td></tr><tr><td>SwinFIR-T</td><td>967</td><td>32.49/0.8991</td><td>28.86/0.7879</td><td>27.74/0.7425</td><td>26.72/0.8076</td><td>31.23/0.9178</td></tr><tr><td>Omni-SR</td><td>×4 792</td><td>32.49/0.8988</td><td>28.78/0.7859</td><td>27.71/0.7415</td><td>26.64/0.8018</td><td>31.02/0.9151</td></tr><tr><td>FDSR</td><td>8,720</td><td>32.56/0.8995</td><td>28.85/0.7881</td><td>27.75/0.7415</td><td>26.68/0.8030</td><td>31.24/0.9174</td></tr><tr><td>SRConvNet-L</td><td>902</td><td>32.44/0.8976</td><td>28.77/0.7857</td><td>27.69/0.7402</td><td>26.47/0.7970</td><td>30.96/0.9139</td></tr><tr><td>CRAFT</td><td>753</td><td>32.52/0.8989</td><td>28.85/0.7872</td><td>27.72/0.7418</td><td>26.56/0.7995</td><td>31.18/0.9168</td></tr><tr><td>LAMNet-large</td><td>1,045</td><td>32.60/0.9000</td><td>28.87/0.7886</td><td>27.77/0.7434</td><td>26.82/0.8078</td><td>31.33/0.9183</td></tr><tr><td>MaIR-Small</td><td>1,374</td><td>32.62/0.8992</td><td>28.90/0.7882</td><td>27.77/0.7431</td><td>26.73/0.8049</td><td>31.34/0.9183</td></tr><tr><td>Ours</td><td>1,516</td><td>32.65 0.9000</td><td>28.93 0.7895</td><td>27.80 0.7440</td><td>26.87 0.8081</td><td>31.43 0.9189</td></tr></table>

Table 2: PSNR/SSIM comparison with existing SISR methods on five benchmarks. Best and second-best results are highlighted in bold and underlined, respectively.  
![](images/db53948c805e266afe743bdf6d827e06cede4051e3caec6f518f574a2149098b.jpg)  
Figure 2: Qualitative comparison for ×4 SISR on image 092 from Urban100 and YumeiroCooking from Manga109. LaST-SR better preserves line continuity, local junctions, and irregular stripe patterns. Best PSNR/SSIM scores are highlighted in red.

## 5 Experiments

We evaluate LaST-SR under the standard bicubic degradation setting, comparing it with state-of-the-art methods and analyzing its architectural components and key modeling mechanisms.

![](images/9027f1a1a50d854938d9f3318fc1d66ca7f4b4b7a0ce2637942d3935483f36ac.jpg)  
Figure 3: Visualization of intermediate representations in LaST-SR on Urban100 img\_005: (a) input image; (b,c) PDPM projections for the steady and transient branches, respectively; (d,e) the corresponding CFD responses; and (f) the STCA-fused representation.

Table 3: Ablation study of the main components for ×2 SISR on Urban100 and Manga109. In the corresponding variants, either the steady or transient operator is replaced with a convolutional operator, while cross-branch attention is replaced with independent self-attention. The best results are highlighted in bold.
<table><tr><td>Variant</td><td>Urban100 PSNR/SSIM</td><td>Manga109 PSNR/SSIM</td></tr><tr><td>Steady → Conv</td><td>28.12/0.8724</td><td>33.25/0.9533</td></tr><tr><td>Transient → Conv</td><td>33.14/0.9373</td><td>39.35/0.9784</td></tr><tr><td>Cross → Self-Attn</td><td>33.19/0.9376</td><td>39.41/0.9784</td></tr><tr><td>Full model</td><td>33.25 0.9380</td><td>39.43 0.9786</td></tr></table>

## 5.1 Experimental Settings

Datasets and metrics. We train LaST-SR on the 800 training images of DIV2K Timofte et al. [2017] with random flip ping and rotation. Evaluation is performed on Set5 Bevilacqua et al. [2012], Set14 Zeyde et al. [2012], BSD100 Martin et al. [2001], Urban100 Huang et al. [2015], and Manga109 Matsui et al. [2017]. LR images are generated by bicubic downsampling, and PSNR/SSIM are computed on the Y channel of YCbCr.

Implementation details. LaST-SR is implemented in PyTorch and trained on eight NVIDIA Tesla V100 GPUs with $B = 6 4 , C = 7 2$ , and $D = 6 { \mathrm { ~ S T A R } } $ blocks. PDPM projects each branch to $\bar { C } _ { \mathrm { m i d } } = 8$ channels. The steady branch uses the full discrete spectrum with $\epsilon = 0 . 7$ and learnable η, while the transient branch uses $K _ { x } = K _ { y } = 1 2$ complex-frequency modes. All window-based operations use $M = 1 6$

We train on randomly cropped 64 × 64 LR patches using Adam with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9$ . The initial learning rate $\mathrm { i s 2 \times 1 0 ^ { - 4 } }$ , and training lasts $5 \times 1 0 ^ { 5 }$ iterations. It is halved at $3 \times 1 0 ^ { 5 } , 4 \times 1 0 ^ { 5 } , 4 . 5 \times 1 0 ^ { 5 }$ , and $4 . 7 5 \times 1 0 ^ { 5 }$ iterations. We set the loss weight $\lambda _ { f } = 0 . 0 1$ . FLOPs are measured on an LR input corresponding to an HR output of $1 2 8 0 \times 7 2 0$

## 5.2 Comparison with State-of-the-Art Methods

We compare LaST-SR with representative CNN-, Transformer-, and frequency-domain SISR methods spanning comparable and larger parameter budgets, including CARN Ahn et al. [2018], LAPAR-A Li et al. [2020a], DeFiANs Huang et al. [2021], SwinIR-light Liang et al. [2021], SwinIR-NG Choi et al. [2023], SwinFIR-T Zhang et al. [2022], Omni-SR Wang et al. [2023], FDSR Xu et al. [2024], SRConvNet-L Li et al. [2025a], CRAFT Li et al. [2025b], LAMNet-large Hu and Sun [2025], and MaIR-Small Li et al. [2025c].

As shown in Table 2, LaST-SR achieves the best PSNR/SSIM on five benchmarks for both ×2 and ×4 SISR. On Urban100 and Manga109, LaST-SR improves upon the best competing results by 0.09 dB and 0.04 dB, respectively, for ×2, and by 0.05 dB and 0.09 dB, respectively, for ×4. Fig. 2 shows that existing methods often blur or disrupt local structures near bends, intersections, and directional changes. LaST-SR better preserves long-range structural continuity, coherent local arrangements, stripe orientations, and irregular textures across Urban100 and Manga109.

## 5.3 Ablation Studies

We conduct ablations under the $\times 2$ setting to evaluate the steady and transient operators, cross-branch interaction, Fourier-magnitude loss, and pole parameterization. All variants use the same training and evaluation settings.

Architectural components. We replace either the steady or transient operator with a convolutional operator and substitute independent self-attention for cross-branch attention. As shown in Table 3, replacing global Fourier modeling causes a dramatic PSNR drop of 5.13 dB on Urban100 and 6.18 dB on Manga109, demonstrating that it is indispensable for preserving image-wide dependencies and long-range structural consistency. Similarly, ablating local complexfrequency modeling yields drops of 0.11 dB and 0.08 dB, supporting its complementary contribution to modeling localized, content-dependent aperiodic variations. Replacing cross-branch attention decreases PSNR by 0.06 dB and 0.02 dB, confirming the efficacy of collaborative aggregation.

Key mechanisms. The removal of the Fourier-magnitude loss function causes a PSNR decline of 0.12 dB across both datasets, affirming that spectral consistency complements pixel-domain supervision. Constraining the learnable transient poles to $\mathrm { R e } ( \mu ) < 0$ lowers Urban100 PSNR by 0.10 dB. Allowing unconstrained real parts permits both increasing and decaying spatial envelopes within finite windows, thereby providing greater flexibility for modeling local aperiodic variations.

Feature responses. Fig. 3 illustrates the complementary behavior of the two responses: the steady response preserves image-wide structural consistency, whereas the transient response emphasizes localized, content-dependent aperiodic variations. Their fusion through STCA further enhances local edge responses while retaining the overall structural organization.

## 6 Conclusion

We have presented LaST-SR, a Laplace-inspired steady–transient framework for jointly modeling image-wide structural information and localized aperiodic variations in SISR. Under certain assumptions, we derive an approximate steady– transient decomposition for static two-dimensional feature maps, extending the pole–residue interpretation from dynamical systems to spatial feature modeling. The proposed CFD module models the two responses through global full-spectrum Fourier and window-conditioned local complex-frequency branches, while $\mathrm { { S T C A } }$ enables their collaborative aggregation. Extensive experiments and ablation studies demonstrate the effectiveness of LaST-SR. Future work will explore super-resolution under more complex degradations.

## A Steady–Transient Decomposition for Static Two-Dimensional Image Features

This section provides a detailed derivation of the approximate steady–transient decomposition introduced in the main paper. We first state the modeling conventions underlying the formulation and then derive the corresponding two-dimensional pole–residue representation.

## A.1 Continuous Feature Representation and Periodic-Extension Convention

Let $\mathbf { X } \in \mathbb { R } ^ { B \times H \times W \times C _ { \mathrm { m i d } } }$ denote an intermediate feature tensor. For the b-th sample, we choose a continuous interpolant $\mathbf { v } _ { b } \in C ( \overline { { \Omega } } ; \mathbb { R } ^ { C _ { \mathrm { m i d } } } ) \subset L ^ { 2 } ( \Omega ; \mathbb { R } ^ { C _ { \mathrm { m i d } } } )$ satisfying

$$
\mathbf { v } _ { b } ( x _ { w } , y _ { h } ) = \mathbf { X } _ { b , h , w , : } , \qquad 1 \leq h \leq H , \quad 1 \leq w \leq W ,\tag{22}
$$

where $\Omega = [ 0 , T _ { x } ) \times [ 0 , T _ { y } )$ and $( x _ { w } , y _ { h } ) \in \Omega$ denotes a point on the regular spatial grid. This interpolant is introduced solely for continuous-domain analysis and is not part of the network implementation. Consistent with the scalar derivation in the main paper, we subsequently fix one sample and one feature channel, denote the corresponding scalar component by $v : \Omega \stackrel { \cdot } { \to } \bar { \mathbb { R } }$ , and omit the batch and channel indices. The multi-channel formulation extends the scalar derivation by assigning complex-valued parameters to each input–output channel pair and aggregating the resulting responses over the input channels.

Assumption A.1 (Periodic-extension convention). We regard Ω as afundamental domain ofthe two-dimensional torus

$$
\mathbb { T } ^ { 2 } \cong \mathbb { R } ^ { 2 } / ( T _ { x } \mathbb { Z } \times T _ { y } \mathbb { Z } )\tag{23}
$$

and let $\widetilde { v } \in L ^ { 2 } (  { \mathbb { T } } ^ { 2 } )$ denote the resulting periodic representative. This construction is introduced only to define a global Fourier representation. It neither assumes that the original imagefeature is physically periodic nor requires matching values on opposite boundaries of Ω. Any boundary mismatch may introduce jump discontinuities at the periodic seams, which remain admissible in the $L ^ { 2 }$ setting.

For $p , q \in \mathbb { Z }$ , define the Fourier coefficients of $\widetilde { v }$ as

$$
\alpha _ { p q } = \frac { 1 } { T _ { x } T _ { y } } \int _ { 0 } ^ { T _ { x } } \int _ { 0 } ^ { T _ { y } } \widetilde { v } ( x , y ) \exp \left[ - \mathrm { i } \left( \omega _ { p } ^ { x } x + \omega _ { q } ^ { y } y \right) \right] d y d x ,\tag{24}
$$

where

$$
\omega _ { p } ^ { x } = \frac { 2 \pi p } { T _ { x } } , \qquad \omega _ { q } ^ { y } = \frac { 2 \pi q } { T _ { y } } .\tag{25}
$$

The corresponding rectangular partial Fourier sum is

$$
S _ { P , Q } \widetilde { v } ( x , y ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \alpha _ { p q } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) .\tag{26}
$$

By completeness of the Fourier basis in $L ^ { 2 } (  { \mathbb { T } } ^ { 2 } )$ ,

$$
\operatorname* { l i m } _ { P , Q \to \infty } \| \widetilde { v } - S _ { P , Q } \widetilde { v } \| _ { L ^ { 2 } (  { \mathbb { T } } ^ { 2 } ) } = 0 .\tag{27}
$$

Equivalently, for any $\varepsilon > 0$ , there exist $P _ { 0 } , Q _ { 0 } \in \mathbb { N }$ such that

$$
\lVert \widetilde { v } - S _ { P , Q } \widetilde { v } \rVert _ { L ^ { 2 } ( \mathbb { T } ^ { 2 } ) } < \varepsilon\tag{28}
$$

for all $P \geq P _ { 0 }$ and $Q \geq Q _ { 0 }$

For the subsequent derivation, let $v _ { P , Q }$ denote the restriction of $S _ { P , Q } \widetilde { v }$ to Ω. For notational simplicity, we write v in place of $v _ { P , Q }$ . The resulting truncated representation is

$$
v ( x , y ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \alpha _ { p q } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) , \qquad ( x , y ) \in \Omega .\tag{29}
$$

Because the considered feature channel is real-valued, its Fourier coefficients satisfy

$$
\alpha _ { - p , - q } = \overline { { \alpha _ { p q } } } .\tag{30}
$$

## A.2 Derivation of the Steady–Transient Decomposition

Under the truncated Fourier representation introduced above, we first consider a linear kernel integral mapping acting on the input feature v. For $( x , y ) \in \Omega$ , its general finite-domain form is

$$
u _ { \Omega } ( x , y ) = \iint _ { \Omega } \kappa \big ( ( x , y ) , ( \xi , \eta ) \big ) v ( \xi , \eta ) d \xi d \eta .\tag{31}
$$

Following the convolutional kernel mapping adopted in LNO Cao et al. [2024], we further restrict the kernel to depend only on the coordinate difference:

$$
\kappa \big ( ( x , y ) , ( \xi , \eta ) \big ) = \kappa ( x - \xi , y - \eta ) .\tag{32}
$$

Motivated by the complex-exponential representation of Prony-SS Hu et al. [2013] and the Laplace-domain pole–residue parameterization of LNO Cao et al. [2024], we adopt the following positive-quadrant extension and one-sided-support convention to establish a two-dimensional one-sided Laplace representation.

Assumption A.2 (Positive-quadrant extension and one-sided support). Let v be the truncated Fourier representation defined in the preceding subsection. Its positive-quadrant extension is

$$
v _ { + } ( x , y ) = \mathbf { 1 } _ { [ 0 , \infty ) ^ { 2 } } ( x , y ) \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \alpha _ { p q } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) .\tag{33}
$$

The translation-invariant kernel is represented by a first-quadrant-supported expansion of learnable complexexponential modes:

$$
\kappa _ { + } ( a , b ) = \mathbf { 1 } _ { [ 0 , \infty ) ^ { 2 } } ( a , b ) \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \mathcal { R } _ { m n } \exp ( \mu _ { m } ^ { x } a + \mu _ { n } ^ { y } b ) ,\tag{34}
$$

where $\mu _ { m } ^ { x } , \mu _ { n } ^ { y } \in \mathbb { C }$ are learnable complex poles along the two spatial directions and $\mathcal { R } _ { m n } \in \mathbb { C }$ is the residue coefficient associated with each pole pair.

Under Assumption A.2, the finite-domain response associated with the first-quadrant-supported kernel is

$$
u _ { \Omega } ( x , y ) = \iint _ { \Omega } \kappa _ { + } ( x - \xi , y - \eta ) v ( \xi , \eta ) d \xi d \eta .\tag{35}
$$

The positive-quadrant extension and one-sided support are introduced only to establish a two-dimensional one-sided Laplace representation and do not assign temporal causality to the spatial image coordinates. For $( x , y ) \in [ 0 , \infty ) ^ { 2 }$ define the one-sided convolution response on the positive quadrant as

$$
u _ { + } ( x , y ) = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } \kappa _ { + } ( x - \xi , y - \eta ) v _ { + } ( \xi , \eta ) d \eta d \xi .\tag{36}
$$

Because $\kappa _ { + } ( x - \xi , y - \eta ) = 0$ whenever $\xi > x \ \mathrm { o r } \ \eta > y ,$ , Eq. (36) reduces to

$$
u _ { + } ( x , y ) = \int _ { 0 } ^ { x } \int _ { 0 } ^ { y } \kappa _ { + } ( x - \xi , y - \eta ) v _ { + } ( \xi , \eta ) d \eta d \xi .\tag{37}
$$

For every $( x , y ) \in \Omega$ , the integration region $[ 0 , x ] \times [ 0 , y ]$ is contained in $\Omega ,$ and $v _ { + } = v$ on this region. Consequently,

$$
\boldsymbol { u } _ { \Omega } = \boldsymbol { u } _ { + } | _ { \Omega } .\tag{38}
$$

We analyze $u _ { + }$ on the positive quadrant below and omit the subscript $" + "$ for notational simplicity.

For a function $f$ defined on $[ 0 , \infty ) ^ { 2 }$ and satisfying the corresponding exponential-order conditions, its two-dimensional one-sided Laplace transform is defined as

$$
{ \mathcal { L } } _ { 2 } \{ f \} ( s _ { x } , s _ { y } ) = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } f ( x , y ) \exp [ - ( s _ { x } x + s _ { y } y ) ] d y d x .\tag{39}
$$

The two-dimensional one-sided convolution theorem gives

$$
U ( s _ { x } , s _ { y } ) = K ( s _ { x } , s _ { y } ) V ( s _ { x } , s _ { y } ) ,\tag{40}
$$

where

$$
V = \mathcal { L } _ { 2 } \{ v _ { + } \} , \qquad K = \mathcal { L } _ { 2 } \{ \kappa _ { + } \} , \qquad U = \mathcal { L } _ { 2 } \{ u _ { + } \} .\tag{41}
$$

Because $v _ { + }$ and $\kappa _ { + }$ are finite linear combinations of complex exponentials on the positive quadrant, their Laplace transforms are

$$
V ( s _ { x } , s _ { y } ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \frac { \alpha _ { p q } } { ( s _ { x } - \mathrm { i } \omega _ { p } ^ { x } ) ( s _ { y } - \mathrm { i } \omega _ { q } ^ { y } ) } ,\tag{42}
$$

$$
K ( s _ { x } , s _ { y } ) = \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \frac { \mathcal { R } _ { m n } } { ( s _ { x } - \mu _ { m } ^ { x } ) ( s _ { y } - \mu _ { n } ^ { y } ) } .\tag{43}
$$

These expressions hold in the common region of convergence

$$
\mathrm { R e } ( s _ { x } ) > \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { 1 \leq m \leq K _ { x } } \mathrm { R e } ( \mu _ { m } ^ { x } ) \right\} ,\tag{44}
$$

$$
\mathrm { R e } ( s _ { y } ) > \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { 1 \leq n \leq K _ { y } } \mathrm { R e } ( \mu _ { n } ^ { y } ) \right\} .\tag{45}
$$

Within this common region, the exponentially weighted absolute values of $v _ { + }$ and $\kappa _ { + }$ are integrable. Hence, the relevant multiple integrals are absolutely convergent, and the convolution theorem applies.

Assumption A.3 (Noncoincident Fourier and learnable poles). The learnable poles do not coincide with the retained Fourier poles:

$$
\mu _ { m } ^ { x } \neq \mathrm { i } \omega _ { p } ^ { x } , \qquad \mu _ { n } ^ { y } \neq \mathrm { i } \omega _ { q } ^ { y }\tag{46}
$$

for all valid indices. This condition ensures nonzero denominators in the subsequent partial-fraction expansion and excludes second-order poles caused by coincidences between the Fourier and learnable polefamilies. Pairwise distinctness among learnable poles is not required; repeated terms may be merged by summing their coefficients.

For compactness, define

$$
d _ { p m } ^ { x } = \mathrm { i } \omega _ { p } ^ { x } - \mu _ { m } ^ { x } , \qquad d _ { q n } ^ { y } = \mathrm { i } \omega _ { q } ^ { y } - \mu _ { n } ^ { y } .\tag{47}
$$

Proposition A.4 (Exact four-family expansion of the truncated one-sided model). Under Assumptions A.2 and A.3, the truncated one-sided convolution response admits, for $( x , y ) \in [ 0 , \infty ) ^ { 2 }$ , the exact expansion

$$
\begin{array} { r l r } {  { u ( x , y ) = \sum _ { p = - P \ q = - Q } ^ { P } \lambda _ { p q } \exp \big ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \big ) } } \\ & { } & { \quad + \sum _ { p = - P \ n = 1 } ^ { P } \int _ { P ^ { \mu } } ^ { K _ { y } } \exp \big ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { p } ^ { y } y \big ) } \\ & { } & { \quad + \sum _ { m = 1 } ^ { K _ { x } } \underbrace { q } _ { \mathrm { m } - Q } \rho _ { m q } ^ { \mu _ { P } } \exp \big ( \mu _ { m } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \big ) } \\ & { } & { \quad + \sum _ { m = 1 } ^ { K _ { x } } \underbrace { K _ { y } } _ { \mathrm { n } - 1 } \gamma _ { m n } \exp \big ( \mu _ { m } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \big ) , } \\ & { } & { \quad + \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \gamma _ { m n } \exp \big ( \mu _ { m } ^ { x } x + \mu _ { n } ^ { y } y \big ) , } \end{array}\tag{48}
$$

where

$$
\lambda _ { p q } = \alpha _ { p q } \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \frac { \mathcal { R } _ { m n } } { d _ { p m } ^ { x } d _ { q n } ^ { y } } ,\tag{49}
$$

$$
\rho _ { p n } ^ { F \mu } = - \sum _ { q = - Q } ^ { Q } \sum _ { m = 1 } ^ { K _ { x } } \frac { \alpha _ { p q } \mathcal { R } _ { m n } } { d _ { p m } ^ { x } d _ { q n } ^ { y } } ,\tag{50}
$$

$$
\rho _ { m q } ^ { \mu F } = - \sum _ { p = - P } ^ { P } \sum _ { n = 1 } ^ { K _ { y } } \frac { \alpha _ { p q } \mathcal { R } _ { m n } } { d _ { p m } ^ { x } d _ { q n } ^ { y } } ,\tag{51}
$$

$$
\gamma _ { m n } = \mathcal { R } _ { m n } \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \frac { \alpha _ { p q } } { d _ { p m } ^ { x } d _ { q n } ^ { y } } .\tag{52}
$$

Moreover, thefinite-domain response satisfies $u _ { \Omega } = u | _ { \Omega }$

Proof. Substituting Eqs. (42) and (43) into Eq. (40) gives

$$
\begin{array} { c } { { \displaystyle U ( s _ { x } , s _ { y } ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } } } \\ { { \displaystyle ~ \times \frac { \alpha _ { p q } \mathcal { R } _ { m n } } { ( s _ { x } - \mathrm { i } \omega _ { p } ^ { x } ) ( s _ { x } - \mu _ { m } ^ { x } ) } \frac { 1 } { ( s _ { y } - \mathrm { i } \omega _ { q } ^ { y } ) ( s _ { y } - \mu _ { n } ^ { y } ) } . } } \end{array}\tag{53}
$$

Assumption A.3 ensures that $d _ { p m } ^ { x } d _ { q n } ^ { y } \neq 0$ . Applying one-dimensional partial fractions along both spatial directions yields

$$
\begin{array} { r l } & { \frac { 1 } { ( s _ { x } - \mathrm { i } \omega _ { p } ^ { x } ) ( s _ { x } - \mu _ { m } ^ { x } ) ( s _ { y } - \mathrm { i } \omega _ { q } ^ { y } ) ( s _ { y } - \mu _ { n } ^ { y } ) } = } \\ & { \frac { 1 } { d _ { p m } ^ { x } d _ { q n } ^ { y } } \left( \frac { 1 } { s _ { x } - \mathrm { i } \omega _ { p } ^ { x } } - \frac { 1 } { s _ { x } - \mu _ { m } ^ { x } } \right) \left( \frac { 1 } { s _ { y } - \mathrm { i } \omega _ { q } ^ { y } } - \frac { 1 } { s _ { y } - \mu _ { n } ^ { y } } \right) . } \end{array}\tag{54}
$$

Expanding Eq. (54) and regrouping terms with identical two-dimensional pole pairs gives

$$
\begin{array} { r } { U ( s _ { x } , s _ { y } ) = \displaystyle \sum _ { p = - r } ^ { P } \displaystyle \sum _ { q = - Q } ^ { Q } \frac { \lambda _ { p q } } { \left( s _ { x } - \mathrm { i } \omega _ { p } ^ { 2 } \right) \left( s _ { y } - \mathrm { i } \omega _ { q } ^ { 3 } \right) } } \\ { + \displaystyle \sum _ { p = - P } ^ { P } \displaystyle \sum _ { n = 1 } ^ { K _ { y } } \frac { \rho _ { p n } ^ { F / \mu } } { \left( s _ { x } - \mathrm { i } \omega _ { p } ^ { 2 } \right) \left( s _ { y } - \mu _ { n } ^ { y } \right) } } \\ { + \displaystyle \sum _ { m = 1 } ^ { K _ { x } } \displaystyle \sum _ { q = - Q } ^ { Q } \frac { \rho _ { m q } ^ { \mu _ { F } ^ { F } } } { \left( s _ { x } - \mu _ { m } ^ { 2 } \right) \left( s _ { y } - \mathrm { i } \omega _ { q } ^ { y } \right) } } \\ { + \displaystyle \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \frac { \gamma _ { m n } } { \left( s _ { x } - \mu _ { m } ^ { x } \right) \left( s _ { y } - \mu _ { n } ^ { y } \right) } , } \end{array}\tag{55}
$$

where the coefficients are those in Eqs. (49)–(52). Since all sums are finite and the transforms share the common region of convergence in Eqs. (44) and (45), the inverse Laplace transform can be applied term by term. Using

$$
{ \mathcal { L } } _ { 2 } ^ { - 1 } \left\{ { \frac { 1 } { ( s _ { x } - a ) ( s _ { y } - b ) } } \right\} = \exp ( a x + b y )\tag{56}
$$

for each pole pair yields Eq. (48). The identity $u _ { \Omega } = u | _ { \Omega }$ follows from the preceding restriction argument, completing the proof. □

To obtain the dual-branch representation used in the main paper, we define the Fourier–Fourier pole family as the steady response, the complex-pole–complex-pole family as the transient response, and collect the two mixed pole families into an explicit residual:

$$
u ( x , y ) = u _ { \mathrm { s s } } ( x , y ) + u _ { \mathrm { t r } } ( x , y ) + r _ { \mathrm { m i x } } ( x , y ) ,\tag{57}
$$

where

$$
u _ { \mathrm { s s } } ( x , y ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } \lambda _ { p q } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) ,\tag{58}
$$

$$
u _ { \mathrm { t r } } ( x , y ) = \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \gamma _ { m n } \exp ( \mu _ { m } ^ { x } x + \mu _ { n } ^ { y } y ) ,\tag{59}
$$

and

$$
\begin{array} { r l } & { r _ { \mathrm { m i x } } ( x , y ) = \displaystyle \sum _ { p = - P } ^ { P } \sum _ { n = 1 } ^ { K _ { y } } \rho _ { p n } ^ { F \mu } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mu _ { n } ^ { y } y \bigr ) } \\ & { \qquad + \displaystyle \sum _ { m = 1 } ^ { K _ { x } } \sum _ { q = - Q } ^ { Q } \rho _ { m q } ^ { \mu F } \exp \bigl ( \mu _ { m } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) . } \end{array}\tag{60}
$$

The two mixed families employ different pole types along the two spatial directions and do not generally vanish. For the SISR setting considered in this work, we focus on two complementary response families: globally supported oscillatory modes for representing repetitive structures, long-range dependencies, and overall structural consistency, and general complex-frequency modes that allow spatially varying envelopes for representing content-dependent aperiodic variations, with local adaptivity further enhanced by the subsequent windowed implementation. The Fourier–Fourier and complex-pole–complex-pole responses correspond directly to these two functional roles, whereas the mixed families describe directionally hybrid responses that use heterogeneous frequency mechanisms along the two axes. To obtain a structurally symmetric and parameter-efficient dual-branch representation aligned with these modeling objectives, we explicitly retain the two same-type pole families and collect the mixed families into $r _ { \mathrm { m i x } }$ . Omitting this explicit residual yields the structured approximation adopted in the main paper:

$$
\boldsymbol { u } ( x , y ) \approx \boldsymbol { u } _ { \mathrm { s s } } ( x , y ) + \boldsymbol { u } _ { \mathrm { t r } } ( x , y ) .\tag{61}
$$

Here, “steady” and “transient” denote complementary spatial response families rather than temporal regimes. The derivation is carried out in the complexified feature space, and the real-valued network response is obtained by taking the real part of the resulting complex response.

## A.3 Fourier-Domain Form of the Convolution Response

The preceding subsection derives the pole–residue expansion of the convolution response through a two-dimensional one-sided Laplace transform. To explain why the retained steady response admits a global Fourier-domain representation, we now examine the corresponding periodic convolution operator under the periodic-extension convention. The purpose is not to rederive the steady–transient decomposition, but to show that convolution in the Fourier basis produces the same global mode family as the retained steady response.

Let $\widetilde v _ { P , Q } : = S _ { P , Q } \widetilde v \in L ^ { 2 } ( \mathbb { T } ^ { 2 } )$ denote the truncated periodic input, and let $\widetilde { \kappa } \in L ^ { 1 } (  { \mathbb { T } } ^ { 2 } )$ denote a periodic convolution kernel. Define the corresponding periodic convolution by

$$
u _ { \mathrm { F } } ( x , y ) = \int _ { 0 } ^ { T _ { x } } \int _ { 0 } ^ { T _ { y } } \widetilde { \kappa } ( x - \xi , y - \eta ) \widetilde { v } _ { P , Q } ( \xi , \eta ) d \eta d \xi ,\tag{62}
$$

where the coordinate differences in $\widetilde { \kappa }$ are interpreted periodically. Because $ { \widetilde { v } } _ { P , Q } \in L ^ { 2 } (  { \mathbb { T } } ^ { 2 } )$ and $\widetilde { \kappa } \in L ^ { 1 } ( \mathbb { T } ^ { 2 } )$ , the periodic convolution is well defined.

Let $\mathcal { F } _ { \mathbb { T } ^ { 2 } }$ denote the Fourier-series transform on the two-dimensional torus, with the normalization

$$
\mathcal { F } _ { \mathbb { T } ^ { 2 } } ( f ) [ p , q ] = \frac { 1 } { T _ { x } T _ { y } } \int _ { 0 } ^ { T _ { x } } \int _ { 0 } ^ { T _ { y } } f ( x , y ) ~\tag{63}
$$

The two-dimensional periodic convolution theorem gives

$$
\mathcal { F } _ { \mathbb { T } ^ { 2 } } ( u _ { \mathrm { F } } ) [ p , q ] = T _ { x } T _ { y } \mathcal { F } _ { \mathbb { T } ^ { 2 } } ( \widetilde { \kappa } ) [ p , q ] \mathcal { F } _ { \mathbb { T } ^ { 2 } } ( \widetilde { v } _ { P , Q } ) [ p , q ] .\tag{64}
$$

By the orthogonality of the Fourier basis,

$$
\mathcal { F } _ { \mathbb { T } ^ { 2 } } ( \widetilde v _ { P , Q } ) [ p , q ] = \left\{ \begin{array} { l l } { \alpha _ { p q } , } & { | p | \leq P , | q | \leq Q , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{65}
$$

Define the effective kernel multiplier at mode $( p , q )$ as

$$
H _ { p q } : = T _ { x } T _ { y } \mathcal { F } _ { \mathbb { T } ^ { 2 } } ( \widetilde { \kappa } ) [ p , q ] .\tag{66}
$$

For the retained modes, Eq. (64) becomes

$$
\mathcal { F } _ { \mathbb { T } ^ { 2 } } ( u _ { \mathrm { F } } ) [ p , q ] = H _ { p q } \alpha _ { p q } .\tag{67}
$$

Applying the inverse Fourier-series transform gives

$$
u _ { \mathrm { { F } } } ( x , y ) = \sum _ { p = - P } ^ { P } \sum _ { q = - Q } ^ { Q } H _ { p q } \alpha _ { p q } \exp \bigl ( \mathrm { i } \omega _ { p } ^ { x } x + \mathrm { i } \omega _ { q } ^ { y } y \bigr ) .\tag{68}
$$

Equation (68) shows that periodic convolution preserves the spatial Fourier basis functions and modulates only their coefficients through complex-valued mode multipliers. The retained steady response in Eq. (58) is spanned by exactly the same global Fourier-mode family. More specifically, Eq. (49) gives

$$
\lambda _ { p q } = M _ { p q } ^ { \mathrm { s s } } \alpha _ { p q } ,\tag{69}
$$

where

$$
M _ { p q } ^ { \mathrm { s s } } : = \sum _ { m = 1 } ^ { K _ { x } } \sum _ { n = 1 } ^ { K _ { y } } \frac { \mathcal { R } _ { m n } } { d _ { p m } ^ { x } d _ { q n } ^ { y } } .\tag{70}
$$

Hence, the steady response itself admits the Fourier spectral-multiplier form

$$
u _ { \mathrm { s s } } = \mathcal { F } _ { \mathbb { T } ^ { 2 } } ^ { - 1 } \left( \mathcal { M } _ { P , Q } ^ { \mathrm { s s } } \mathcal { F } _ { \mathbb { T } ^ { 2 } } ( \widetilde { v } _ { P , Q } ) \right) ,\tag{71}
$$

where $\mathcal { M } _ { P , Q } ^ { \mathrm { s s } }$ acts mode-wise as

$$
\left( \mathcal { M } _ { P , Q } ^ { \mathrm { s s } } \widehat { v } \right) [ p , q ] = M _ { p q } ^ { \mathrm { s s } } \widehat { v } [ p , q ] , \qquad | p | \leq P , \quad | q | \leq Q .\tag{72}
$$

Equation (71) has the same general form as the spectral convolution used in the Fourier neural operator Li et al. [2020b]: the input is transformed to the Fourier domain, each frequency mode is modulated by a complex-valued spectral multiplier, and the result is mapped back to the spatial domain through the inverse Fourier transform. Because Fourier modes have global spatial support, this form provides a direct analytical basis for implementing the steady response with a global Fourier-domain operator.

## References

Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2):295–307, 2015.

Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. Accurate image super-resolution using very deep convolutional networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1646–1654, 2016a.

Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. Deeply-recursive convolutional network for image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1637–1645, 2016b.

Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In Proceedings ofthe IEEE conference on computer vision and pattern recognition workshops, pages 136–144, 2017.

Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2472–2481, 2018.

Namhyuk Ahn, Byungkon Kang, and Kyung-Ah Sohn. Fast, accurate, and lightweight super-resolution with cascading residual network. In Proceedings ofthe European conference on computer vision (ECCV), pages 252–268, 2018.

Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021.

Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xin Yuan, et al. Cross aggregation transformer for image restoration. Advances in Neural Information Processing Systems, 35:25478–25490, 2022.

Haram Choi, Jeongmin Lee, and Jihoon Yang. N-gram in swin transformers for efficient lightweight image superresolution. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pages 2071–2081, 2023.

Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image super-resolution transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22367–22377, 2023.

Hang Wang, Xuanhong Chen, Bingbing Ni, Yutian Liu, and Jinfan Liu. Omni aggregation networks for lightweight image super-resolution. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 22378–22387, 2023.

Dafeng Zhang, Feiyu Huang, Shizhuo Liu, Xiaobing Wang, and Zhezhu Jin. Swinfir: Revisiting the swinir with fast fourier convolution and improved training for image super-resolution. arXiv preprint arXiv:2208.11247, 2022.

Abhishek Kumar Sinha, S Manthira Moorthi, and Debajyoti Dhar. Nl-ffc: Non-local fast fourier convolution for image super resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 467–476, 2022.

Pengcheng Xu, Qun Liu, Huanan Bao, Ruhui Zhang, Lihua Gu, and Guoyin Wang. Fdsr: An interpretable frequency division stepwise process based single-image super-resolution network. IEEE Transactions on Image Processing, 33: 1710–1725, 2024.

Xiaoyi Liu and Hao Tang. Difffno: Diffusion fourier neural operator. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 150–160, 2025.

Qianying Cao, Somdatta Goswami, and George Em Karniadakis. Laplace neural operator for solving differential equations. Nature Machine Intelligence, 6(6):631–640, 2024.

Jinmin Li, Tao Dai, Mingyan Zhu, Bin Chen, Zhi Wang, and Shu-Tao Xia. Fsr: A general frequency-oriented framework to accelerate image super-resolution networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1343–1350, 2023.

Sau-Lon James Hu, Wen-Long Yang, and Hua-Jun Li. Signal decomposition and reconstruction using complex exponential models. Mechanical Systems and Signal Processing, 40(2):421–438, 2013.

Y. Hua and T.K. Sarkar. Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise. IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5):814–824, 1990. doi:10.1109/29.56027.

R. Roy and T. Kailath. Esprit-estimation of signal parameters via rotational invariance techniques. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(7):984–995, 1989. doi:10.1109/29.32276.

Radu Timofte, Eirikur Agustsson, Luc Van Gool, Ming-Hsuan Yang, and Lei Zhang. Ntire 2017 challenge on single image super-resolution: Methods and results. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 114–125, 2017.

Marco Bevilacqua, Aline Roumy, Christine Guillemot, and Marie line Alberi Morel. Low-complexity single-image super-resolution based on nonnegative neighbor embedding. In Proceedings ofthe British Machine Vision Conference, pages 135.1–135.10. BMVA Press, 2012. ISBN 1-901725-46-4. doi:http://dx.doi.org/10.5244/C.26.135.

Roman Zeyde, Michael Elad, Matan Protter, Christian Gout, Jean-Daniel Boissonnat, Larry Schumaker, Tom Lyche, Albert Cohen, Patrick Chenin, and Marie-Laurence Mazure. On single image scale-up using sparse-representations. In Curves and Surfaces, volume 6920 of Lecture Notes in Computer Science, pages 711–730. Springer Berlin / Heidelberg, Germany, 2012. ISBN 9783642274121.

D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, volume 2, pages 416–423 vol.2, 2001. doi:10.1109/ICCV.2001.937655.

Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5197–5206, 2015.

Yusuke Matsui, Kota Ito, Yuji Aramaki, Azuma Fujimoto, Toru Ogawa, Toshihiko Yamasaki, and Kiyoharu Aizawa. Sketch-based manga retrieval using manga109 dataset. Multimedia tools and applications, 76(20):21811–21838, 2017.

Wenbo Li, Kun Zhou, Lu Qi, Nianjuan Jiang, Jiangbo Lu, and Jiaya Jia. Lapar: Linearly-assembled pixel-adaptive regression network for single image super-resolution and beyond. Advances in Neural Information Processing Systems, 33:20343–20355, 2020a.

Yuanfei Huang, Jie Li, Xinbo Gao, Yanting Hu, and Wen Lu. Interpretable detail-fidelity attention network for single image super-resolution. IEEE Transactions on Image Processing, 30:2325–2339, 2021.

Feng Li, Runmin Cong, Jingjing Wu, Huihui Bai, Meng Wang, and Yao Zhao. Srconvnet: A transformer-style convnet for lightweight image super-resolution. International Journal of Computer Vision, 133(1):173–189, 2025a.

Ao Li, Le Zhang, Yun Liu, and Ce Zhu. Exploring frequency-inspired optimization in transformer for efficient single image super-resolution. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):3141–3158, 2025b.

Zhenyu Hu and Wanjie Sun. Unifying dimensions: A linear adaptive mixer for lightweight image super-resolution. IEEE Transactions on Image Processing, 2025.

Boyun Li, Haiyu Zhao, Wenxin Wang, Peng Hu, Yuanbiao Gou, and Xi Peng. Mair: A locality-and continuity-preserving mamba for image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7491–7501, 2025c.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020b.