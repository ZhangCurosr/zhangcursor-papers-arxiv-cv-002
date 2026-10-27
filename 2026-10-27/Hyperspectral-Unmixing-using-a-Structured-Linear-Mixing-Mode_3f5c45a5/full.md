# Hyperspectral Unmixing using a Structured Linear Mixing Model with Additive Spectral Variability

Salah Eddine Brezini, Yannick Deville, Moussa Sofiane Karoui, Fatima Zohra Benhalouche and Abdelaziz Ouamri

Abstract— Linear hyperspectral unmixing aims at retrieving pure material spectra present in an observed scene and their associated abundance fractions in each pixel. In most practical situations, hyperspectral images are affected by spectral variability (SV) also known as intra-class variability, which is caused by different physical phenomena including atmospheric effects, intrinsic variability of the pure materials and illumination or topographic changes. Nevertheless, the standard hyperspectral unmixing methods that rely on the Linear Mixing Model (LMM) do not consider these phenomena. Thus, neglecting them will spread errors throughout the linear hyperspectral unmixing process. To avoid this problem, this paper proposes two hyperspectral unmixing algorithms dedicated to a parametric model called Structured Additive Variability LMM (SAV-LMM). This mixing model handles SV in an additive manner by incorporating two terms into the standard LMM. Moreover, the designed algorithms are based on Nonnegative Matrix Factorization (NMF) and optimize a cost function which integrates additional penalization accounting for SV. Furthermore, a novel improved initialization scheme, which takes SV into account, is proposed. Experiments conducted on both synthetic and real datasets show the effectiveness of the proposed algorithms for linearly unmixing hyperspectral images with SV compared to the tested state-of-theart methods.

Index Terms— Hyperspectral Imaging, Linear Spectral Unmixing, Spectral Variability, Nonnegative Matrix Factorization.

## I. INTRODUCTION

spectral resolution since they are acquired by sensorsR spectral resolution since they are acquired by sensors that collect light scattered by objects present in its field of view in hundreds of narrow and contiguous spectral bands covering the visible-infrared domain [1]. Thus, this rich spectral information enables an accurate identification of pure materials present in an observed hyperspectral image (HSI). This feature allows HSIs to be at the core of different space applications [2] including environmental monitoring [3], sustainable energy [4], and that undoubtedly improves our knowledge about the Earth surface and the physical phenomena surrounding us.

Nevertheless, since the acquisition of HSIs is achieved in narrow bandwidths, the remote hyperspectral sensors do not obtain enough photons to maintain a high signal-to-noise ratio (SNR) [5], [6]. This physical limitation compels the hyperspectral sensors to work by operating an optimal compromise between satisfactory SNR and spatial resolution [5]. This physical constraint inevitably provides an HSI with low spatial resolution and leads to the well-known problem of the existence of mixed pixels in HSIs [7]. Hence, it hinders their applications in many fields which need both high spectral and spatial resolutions. In this case, each observed pixel spectrum is represented by the mixture of spectral contributions from pure materials (also called endmembers), present in the considered HSI, and their associated abundance coefficients. The way the spectral combination is achieved depends directly on the observed scene [6]. To circumvent this physical limitation, the Spectral Unmixing (SU) remains relevant. SU techniques aim to estimate a collection of pure material spectra and their associated abundance coefficients. SU can be seen as a typical Blind Source Separation (BSS) problem [8], [9]. Nowadays, various methods, based on the Linear Mixture Model (LMM) [1], are being developed. Due to their simplicity, they are primarily known as Linear Spectral Unmixing (LSU) techniques. Indeed, LMM is the result of the basic assumption, which states that an endmember can be represented by only one spectrum in the whole HSI and the latter can be fully reconstructed from this assumption. However, this assumption does not hold anymore when some physical phenomena occur in the observed scene. Consequently, LMM appears rather limited by two main issues [10], [11], namely the spectral/intraclass variability [10], [12], [13] and the nonlinearity [11], [14]- [16].

The present paper aims to address the Spectral Variability (SV) issue. It is often caused by several factors such as: illumination and topographic changes, atmospheric conditions and/or material composition [10], [12], [13]. To tackle SV, the concept of endmembers is substituted by the notion of classes of endmembers. To this end, various methods have been developed. Techniques tackling SV can be grouped into three classes [10], [12]: (1) spectral library-based approaches, (2) methods using a particular statistical distribution to describe observed data, and (3) parametric model-based methods. A straightforward manner to tackle SV is to consider spectral libraries. Indeed, they can be obtained using two approaches: the first one is to gather spectra obtained from a field campaign [17] and the second one is to extract endmember spectra directly from the considered HSI [18]-[21] by running an Endmember Extraction Algorithm (EEA) several times on the considered HSI.

Recent approaches handle SV by representing an HSI as a statistical distribution [22]-[25]. Furthermore, parametric model-based techniques [10] integrate SV directly into LMM by considering variables representing SV. This class of methods is more flexible since neither a spectral library nor prior knowledge of the statistical distribution of the involved variables is needed to achieve LSU. Hence, different methods have recently been introduced [26]-[31]. In [26], the perturbed LMM (PLMM) has been proposed to handle SV by adding perturbation terms to endmembers. PLMM jointly estimates endmember spectra and abundance fractions. The extended LMM (ELMM) [27] represents SV arising from illumination and topography changes in each pixel by using scaling factors. Recently, a generalization of ELMM, termed as the generalized LMM (GLMM) [28], has been introduced. Among its advantages, GLMM allows the scaling factors to vary within each spectral band, providing GLMM with more flexibility with respect to ELMM. The augmented LMM (ALMM) [29] adds low-rank terms to ELMM to overcome its limitation by addressing more complex SV that cannot just be described by illumination changes. Lately, the focus of recent research has been on reducing the interdependencies among the abundance estimations [32]. In [33], the authors use spatial contextual information to improve the estimation of abundance fractions. In [34], a sparse representation is used to tackle SV.

The above methods generally perform well in the case of SV induced by illumination and topography changes, particularly the parametric model-based methods. In fact, they do not explicitly take into account the intrinsic variability caused by the chemical composition of a material, which is more complex to treat and cannot be just represented by scaling factors. Indeed, these methods deal with SV mainly in the spatial part of variables and do not consider it in the spectral side of the involved variables. To this end, new sophisticated LMMs have been developed that allow addressing SV in both spectral and spatial domains [35]. Furthermore, this extended version of LMM, which explicitly models SV not only in the spatial domain but also in the spectral domain, makes it possible to retrieve each pixel of an HSI the associated spectra for each class of endmembers. The first technique using this concept is known as Inertia-Constrained Pixel-by-Pixel NMF (IP-NMF) [35]. Furthermore, the assumption which states that SV is not only induced by topographic or/and illumination changes is supported by a data analysis in [35], which shows that in urban scenes, the intrinsic variability is present and cannot be neglected in the unmixing process. To overcome this shortcoming, IP-NMF penalizes the trace of the estimated spectra covariance matrix for each class to tune the inertia of each class of endmembers. IP-NMF proves to be very efficient in urban environments and for ocean remote sensing [3], [36]. In [37]-[39], a novel parametric model has been introduced, without any physical consideration, by adding two matrices to the LMM to address SV. Indeed, this variation can be defined as tunable additive shifts around an adaptive reference spectrum per endmember for each class of endmembers. Moreover, parametric models in [38] and [39] consider structured variables to deal more efficiently with SV.

In this paper, a complete flowchart is proposed by considering SV from the initialization stage to the estimation of endmember spectra and abundance fractions. To this end, two algorithms, based on Nonnegative Matrix Factorization (NMF) [40], [41], dedicated to a proposed parametric mixing model that deals with SV, are introduced and an improved initialization approach is proposed. More precisely, the designed algorithms are gradient descent-based ones. The first one, called Gradientbased Structured Additive Variability LMM (Grad-SAV-LMM), is based on the gradient descent algorithm [42]. The second one, termed Multiplicative SAV-LMM (Multi-SAV-LMM), considers the multiplicative gradient-based update rules [43].

As a matter of fact, the proposed methods and the considered state-of-the-art techniques address ill-posed optimization problems and the performance of the derived algorithms is directly affected by two important factors, which are noise present in the image and the initialization. To improve their performance, a new strategy is introduced by using a (pseudo or image-based) spectral library, obtained by means of the Automated Extraction of image-based Endmember Bundles (AEEB) algorithm [18], into the initialization stage.

1) Deal with SV by allowing more spectral flexibility with respect to some parametric models, since the structured variables permit the spectra to vary for each spectral band.

2) Address SV in very general variability patterns.

3) Introduce a novel manner to initialize the spectra of the

proposed algorithms and the tested state-of-the-art methods. The remainder of this paper is organized as follows. The data mixing model is described in Section II, in particular a great attention is paid to the adaptation of LMM to deal with SV by considering new variables. Section III presents the proposed algorithms derived from the introduced cost function and describes the initialization strategy adopted in the conducted experiments. Section IV presents the results obtained from the experiments conducted on synthetic and real data. Finally, Section V concludes this work.

## II. PROPOSED DATA MIXING MODEL

As stated above, within the framework of LMM, the observed nonnegative reflectance spectrum associated with a given pixel � can be written as follows:

$$
\begin{array} { r } { x _ { i } = \sum _ { j = 1 } ^ { M } r _ { j } c _ { j i } \qquad i = 1 , \ldots \ldots , P , } \end{array}\tag{1}
$$

where $r _ { j } \in R _ { + } ^ { L \times 1 }$ denotes the nonnegative reflectance spectrum of the $j ^ { t h }$ endmember. The coefficients $c _ { j i }$ stand for the nonnegative abundance fraction of the $j ^ { \mathrm { t h } }$ endmember in the considered pixel i. P and L are, respectively, the number of image pixels and the number of spectral bands, while M corresponds to the number of endmembers.

In the present study, SV is addressed by using an extended version of LMM introduced in [35] to model the data. Mainly, the extended model differs from LMM since it considers a different spectrum in each pixel of an HSI for each class of endmembers. For each class of endmembers, it allows one to derive different but close spectra in all the HSI pixels. Hence, this feature enables one to measure SV in spectral and spatial domains. Therefore, the model (1) becomes [35]:

$$
\begin{array} { r } { x _ { i } = \sum _ { j = 1 } ^ { M } r _ { i j } c _ { j i } \qquad i = 1 , \ldots \ldots , P . } \end{array}\tag{2}
$$

where $r _ { i j } \ \in R _ { + } ^ { L \times 1 }$ is the version associated with pixel i for the $j ^ { \mathrm { t h } }$ class of nonnegative endmember reflectance spectra.

SV is considered by adding two vector-form parameters into $r _ { i j }$ for each pixel. This permits to handle SV in the spectral part of the involved variables. According to this representation, $r _ { i j }$ can be formulated as [37]-[39]:

$$
r _ { i j } = s _ { i j } + a _ { i j } - b _ { i j } .\tag{3}
$$

Here, $s _ { i j } \in R _ { + } ^ { L \times 1 }$ denotes the nonnegative reflectance spectrum that is the reflectance spectrum for the $j ^ { \mathrm { t h } }$ class of endmembers. $a _ { i j }$ and $b _ { i j }$ are the nonnegative SV vectors for the version of the $j ^ { \mathrm { t h } }$ class of endmembers associated with pixel i. The vectors $a _ { i j }$ and $b _ { i j }$ allow the tuning of SV separately in each spectral band of the reference spectrum $s _ { i j }$ within a certain interval in each pixel of the considered HSI. This feature enables one to deal with SV in a general pattern and gives high flexibility to the proposed parametric model. The model described in Eq. (2) can be reformulated by inserting the expression of $r _ { i j }$ provided in Eq. (3). Therefore, Eq. (2) is formulated as follows:

$$
\begin{array} { r } { x _ { i } = \sum _ { j = 1 } ^ { M } ( s _ { i j } + a _ { i j } - b _ { i j } ) c _ { j i } . } \end{array}\tag{4}
$$

The proposed mixing model can be expressed in matrix form to describe the processed HSI by taking into consideration the particular structure of the involved variables. These structures allow one to address SV more efficiently. Thus, Eq. (4) can be written as:

$$
X = ( S _ { r } K + A - B ) \thinspace C ,\tag{5}
$$

where $X \in R _ { + } ^ { L \times P }$ is the matrix that contains the observed pixel spectra. Each row of this matrix is associated with one spectral band and each column with one observed pixel spectrum. For each pixel, to estimate a set of spectra corresponding to each class of endmembers present in the HSI, the matrix $\check { C } \check { \in } \mathbb { R } _ { + } ^ { M P \times P }$ also has to be adapted. Indeed, this matrix is block-diagonal and contains the abundance fractions [35]:

$$
C = \left[ \begin{array} { l l l l } { } & { 0 } & { } & { 0 } \\ { c _ { 1 } } & { \vdots } & { \cdots } & { \vdots } \\ { } & { 0 } & { } & { 0 } \\ { 0 } & { c _ { 2 } } & { \ddots } & { \vdots } \\ { } & { } & { } & { 0 } \\ { 0 } & { 0 } & { } & { } \\ { \vdots } & { \vdots } & { \cdots } & { c _ { P } } \\ { 0 } & { 0 } & { } & { } \end{array} \right] ,\tag{6}
$$

where $\boldsymbol { c } _ { i } = [ c _ { 1 i } , \ldots , c _ { M i } ] ^ { \mathrm { T } }$ is an M-element column vector that contains the abundance fractions associated with the $i ^ { \mathrm { { t h } } }$ observed pixel spectrum, and the notation $^ { 6 \ } ( 0 . . . 0 ) ^ { \mathrm { T } ^ { 5 } }$ corresponds to the M-zero-element column vector. $[ . ] ^ { \mathrm { T } }$ denotes the vector/matrix transpose. We can see clearly from the particular structure of the matrix � that this enables one to extract efficiently from each pixel spectra of each class of endmembers and to tackle SV in the spatial domain. The matrix $K \in \mathbb { R } _ { + } ^ { M \times M P }$ is the identity matrix of dimension $M \times M$ repeated P times. The introduction of this matrix allows one both to handle SV and to avoid any additional update rule and constraint related to the particular structure of the involved matrices including $S _ { r } , A$ and �. The matrix $S _ { r } \in \mathbb { R } _ { + } ^ { L \times M }$ contains the reference spectra. Unlike in [37]-[39], the reference spectra are not associated only with the first pixel but with the whole HSI. This undoubtedly enhances the robustness of the proposed methods, in particular when unmixing real data, which can contain, in the first pixel, an outlier and are contaminated by noise. The matrices $A \in \mathbb { R } _ { + } ^ { L \times M P }$ and $B \in \mathbb { R } _ { + } ^ { L \times M P }$ contain the upper and lower SV coefficients related to $a _ { i j }$ and $b _ { i j }$ respectively. These coefficients allow tuning the model of the whole observed spectra for each pixel of the processed HSI, with respect to the reference spectra matrix $S _ { r } ,$ within an additive manner. Moreover, these coefficients delimit the upper and lower variations with respect to these considered reference spectra. Thus, � and � can be formulated as:

$$
\begin{array} { r } { \boldsymbol { A } = [ A _ { 1 } | A _ { 2 } \ldots | A _ { P } ] , \qquad } \\ { \mathrm { w i t h } \boldsymbol { A } _ { t } \in \mathbb { R } _ { + } ^ { L \times M } ( t = 1 , . . . , P ) , } \end{array}\tag{7}
$$

$$
\begin{array} { r } { B = [ B _ { 1 } | B _ { 2 } \ldots | B _ { P } ] , \phantom { x x x x x x x x x x x x x x x x x x x x x } } \\ { \mathrm { w i t h } \ B _ { t } \in \ \mathbb { R } _ { + } ^ { L \times M } ( t = 1 , . . . , P ) . } \end{array}\tag{8}
$$

## III. PROPOSED UNMIXING ALGORITHMS

In this paper, two algorithms are proposed, which are pixel-bypixel extensions of NMF. In its standard form, NMF aims at decomposing the observed HSI into the product of two nonnegative matrices. This makes NMF attractive for linear hyperspectral unmixing since it tailors well the considered data and variables [43]. Moreover, the proposed algorithms use constraints accounting for SV and the particular structures of the involved variables in order to improve the unmixing performance. The introduced algorithms are gradient descentbased ones and aim at estimating variables by modelling the data mixing function defined by (5). The involved variables are �<sub>௥</sub><sup>෩</sup> , �<sup>ሚ</sup>, �<sup>෨</sup> and $\tilde { C }$ that respectively represent the estimate of variables $S _ { r } , A , B$ and $C .$ The proposed approaches consist in minimizing the following cost function by using iterative and gradient-based update rules:

$$
\begin{array} { c } { { J = \displaystyle \frac { 1 } { 2 } \left\| X - \big ( \widetilde { S } _ { r } K + \widetilde { A } - \widetilde { B } \big ) \widetilde { C } \right\| _ { F } ^ { 2 } + } } \\ { { \frac { \lambda _ { A } } { 2 } \left\| \widetilde { A } \right\| _ { F } ^ { 2 } + \frac { \lambda _ { B } } { 2 } \left\| \widetilde { B } \right\| _ { F } ^ { 2 } , } } \end{array}\tag{9}
$$

where ‖.‖<sub>F</sub> stands for the Frobenius norm.

The two penalization terms at the end of (9) limit the Frobenius norms of the matrices $\tilde { A }$ and $\tilde { B }$ . These penalizations have two purposes. The first one is related to the general framework of NMF. Besides the limitations cited above (initialization and noise), NMF is also limited by the fact it cannot guarantee that the estimated solution is a global minimum. Therefore, the two penalization terms permit to constrain the solution space and improve the unmixing performance of the introduced algorithms. The second objective is to control the amplitude of the SV coefficients. These penalization terms allow to obtain coherent SV around the reference spectra $\widetilde { S _ { r } }$ for each pixel of the HSI. $\lambda _ { A }$ and $\lambda _ { B }$ are positive balancing coefficients to manage the upper and the lower $\mathrm { S V } ,$ respectively. and do not necessarily have the same values.

Since the two algorithms are gradient descent-based methods, gradient expressions of the considered cost function are derived by making use of the properties provided in [44] and read:

$$
\begin{array} { r } { \frac { \partial J } { \partial \widetilde { S } _ { r } } = - X \tilde { C } ^ { T } K ^ { T } + \widetilde { S } _ { r } K \tilde { C } \tilde { C } ^ { T } K ^ { T } + \tilde { A } \tilde { C } \tilde { C } ^ { T } K ^ { T } - \tilde { B } \tilde { C } \tilde { C } ^ { T } K ^ { T } , } \end{array}\tag{10}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial \tilde { A } } = - X \tilde { C } ^ { T } + \widetilde { S _ { r } } K \tilde { C } \tilde { C } ^ { T } + \tilde { A } \tilde { C } \tilde { C } ^ { T } - \tilde { B } \tilde { C } \tilde { C } ^ { T } + \lambda _ { A } \tilde { A } , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial \tilde { B } } = \boldsymbol { X } \tilde { C } ^ { T } - \widetilde { S _ { r } } \boldsymbol { K } \tilde { C } \tilde { C } ^ { T } - \tilde { A } \tilde { C } \tilde { C } ^ { T } + \tilde { B } \tilde { C } \tilde { C } ^ { T } + \lambda _ { B } \tilde { B } , } \end{array}\tag{12}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial \tilde { c } } = - ( \widetilde { S _ { r } } K + \tilde { A } ) ^ { T } X + \tilde { B } ^ { T } X - \left( \tilde { B } ^ { T } \big ( \widetilde { S _ { r } } K + \tilde { A } \big ) + \right. } \end{array}
$$

$$
( \widetilde { S _ { r } } K + \tilde { A } ) ^ { T } \tilde { B } ) \tilde { C } + ( \widetilde { S _ { r } } K + \tilde { A } ) ^ { T } ( \widetilde { S _ { r } } K + \tilde { A } ) \tilde { C } + \tilde { B } ^ { T } \tilde { B } \tilde { C } .\tag{13}
$$

Since each element of the matrix ${ \tilde { S } } _ { r }$ has to be lower than or equal to one (i.e., when considering the relative reflectance measures), the following constraints should be considered:

$$
0 ^ { L \times M } \leq \tilde { A } _ { t } \leq \operatorname* { m i n } { ( u ^ { L \times M } , 1 ^ { L \times M } + \tilde { B } _ { t } - \widetilde { S _ { r } } ) } ,\tag{14}
$$

$$
\forall t = 1 \ldots P ,
$$

$$
0 ^ { L \times M } \leq \widetilde { B } _ { t } \leq \operatorname* { m i n } { \left( v ^ { L \times M } , \tilde { A } _ { t } + \widetilde { S } _ { r } \right) } ,\tag{15}
$$

$$
\forall t = 1 \ldots P ,
$$

where � (respectively �) corresponds to the maximum value that can be reached by any element of the considered submatrix of �ሚ (respectively �෨ ).

## A. Gradient-based SAV-LMM

The first proposed method, called Grad-SAV-LMM, applies the gradient descent approach with a fixed learning rate [42]. Thus, the following iterative update rule is applied to estimate the involved variables:

$$
\begin{array} { r } { \theta  \theta - \varphi _ { \theta } \frac { \partial J } { \partial \theta } , } \end{array}\tag{16}
$$

� stands for one of the four estimated matrices ${ \widetilde { \cal S } } _ { r } , { \tilde { \cal A } } ,$ �෨ and ${ \tilde { C } } ,$ and $\varphi _ { \theta }$ is a fixed scalar learning rate. According to (16), the update rules, for the four considered variables, are given by:

$$
\begin{array} { r } { \widetilde { S _ { r } }  \widetilde { S _ { r } } - \varphi _ { \widetilde { S _ { r } } } \frac { \partial J } { \partial \widetilde { S _ { r } } } , } \end{array}\tag{17}
$$

$$
\begin{array} { r } { \tilde { A }  \tilde { A } - \varphi _ { \tilde { A } } \frac { \partial J } { \partial \tilde { A } } , } \end{array}\tag{18}
$$

$$
\begin{array} { r } { \tilde { B } \gets \tilde { B } - \varphi _ { \tilde { B } } \frac { \partial J } { \partial \tilde { B } } , } \end{array}\tag{19}
$$

$$
\tilde { C } \gets \tilde { C } - \varphi _ { \tilde { C } } \frac { \partial J } { \partial \tilde { C } } .\tag{20}
$$

The convergence criterion is defined by:

$$
\Big \vert \frac { J ^ { ( t ) } - J ^ { ( t + 1 ) } } { J ^ { ( t ) } } \Big \vert \leq T r e s h ,\tag{21}
$$

where � is an iteration and ����ℎ is a predefined threshold. Moreover, to avoid an excessive number of iterations, a maximum number of iterations is also considered.

## B. Multiplicative-based SAV-LMM

The second proposed algorithm is based on the multiplicative gradient based-updated rules, which are obtained by the following procedure. The gradient expressions of � with respect to a matrix variable θ can here be formulated as the difference of two nonnegative functions [4], such that $\frac { \partial J } { \partial \theta } = { \frac { \partial J } { \partial \theta } } ^ { + } - { \frac { \partial J } { \partial \theta } } ^ { - }$ The functions ${ \frac { \partial J } { \partial \theta } } ^ { + }$ and $\frac { \partial J ^ { - } } { \partial \theta }$ correspond to the terms of (10)- (13) preceded by a plus and minus sign, respectively. The nonnegativity of θ throughout its adaptation can be fulfilled by initializing θ with a nonnegative value and by properly choosing the value of the learning rate matrix $\varphi _ { \theta }$ according to:

$$
\begin{array} { r } { \varphi _ { \theta } = \theta \oslash \frac { \partial J } { \partial \theta } ^ { + } , } \end{array}\tag{22}
$$

where $\oslash$ stands for the element-wise division. Therefore, the update rule (16) is here modified by using the element-wise multiplication by the matrix $\varphi _ { \theta }$ and thus becomes [4]

$$
\begin{array} { r } { \theta  \theta \odot \frac { \partial J ^ { - } } { \partial \theta } \oslash \frac { \partial J ^ { + } } { \partial \theta } . } \end{array}\tag{23}
$$

Thus, the proposed multiplicative update rules, for the four considered variables, are:

$$
\begin{array} { r } { \widetilde { S _ { r } }  \widetilde { S _ { r } } \odot ( X \tilde { C } ^ { T } K ^ { T } + \tilde { B } \tilde { C } \tilde { C } ^ { T } K ^ { T } ) \oslash } \\ { ( \widetilde { S _ { r } } K \tilde { C } \tilde { C } ^ { T } K ^ { T } + \tilde { A } \tilde { C } \tilde { C } ^ { T } K ^ { T } + \varepsilon ) , \qquad } \end{array}\tag{24}
$$

$$
\begin{array} { r } { \tilde { A } \gets \tilde { A } \odot ( X \tilde { C } ^ { T } + \tilde { B } \tilde { C } \tilde { C } ^ { T } ) \oslash ( \widetilde { S _ { r } } K \tilde { C } \tilde { C } ^ { T } + \tilde { A } \tilde { C } \tilde { C } ^ { T } + } \\ { \lambda _ { A } \tilde { A } + \varepsilon ) , \qquad } \end{array}\tag{25}
$$

$$
\begin{array} { r } { \tilde { B } \gets \tilde { B } \odot ( \widetilde { S _ { r } } K \tilde { C } \tilde { C } ^ { T } + \tilde { A } \tilde { C } \tilde { C } ^ { T } ) \oslash ( X \tilde { C } ^ { T } + \tilde { B } \tilde { C } \tilde { C } ^ { T } + } \\ { \lambda _ { B } \tilde { B } + \varepsilon ) , \qquad } \end{array}\tag{26}
$$

$$
\begin{array} { r l } & { \quad \tilde { C } \gets \tilde { C } \odot ( ( \widetilde { S _ { r } } K + \tilde { A } ) ^ { T } X + \left( \tilde { B } ^ { T } \big ( \widetilde { S _ { r } } K + \tilde { A } \big ) + \right. } \\ & { \left. \left( \widetilde { S _ { r } } K + \tilde { A } \right) ^ { T } \tilde { B } \right) \tilde { C } ) \oslash ( \tilde { B } ^ { T } X + ( \widetilde { S _ { r } } K + \tilde { A } ) ^ { T } ( \widetilde { S _ { r } } K + } \\ & { \quad \quad \quad \quad \tilde { A } ) \tilde { C } + \tilde { B } ^ { T } \tilde { B } \tilde { C } + \varepsilon ) , } \end{array}\tag{27}
$$

where � is a very small and positive value added to each denominator of update rules (24)-(27) to prevent any division by zero.

## C. Initialization by AEEB

As mentioned above, the performance of the proposed algorithms (and also of the considered state-of-art-techniques), which are nonconvex optimization problems, depend on their initialization. To improve this aspect, a straightforward strategy is adopted by using a (pseudo or image-based) spectral library.

To this end, such a spectral library is constructed by means of AEEB [18]. AEEB relies on a simple assumption that statistics of the considered HSI can be approximately recovered by using some small parts of it. Indeed, if there are a sufficient number of pure pixels (pixels containing only one endmember), they can be used to describe SV present in the observed scene by providing suitable candidates for each class of endmembers present in the observed HSI. One of the main motivations behind the use of such a spectral library is the fact that it can handle different types of SV [6]. The initialization proposed here is divided in three main stages. The first two stages construct the spectral library like in [18]. The last stage is devoted to the selection of the most suitable spectra to represent a class of endmembers.

More precisely, the first stage consists in building spectra bundles by means of several runs of an EEA on several randomly selected subsets of the HSI. In our experiments, the Vertex Component Analysis (VCA) algorithm [45] is used as an EEA. In the second stage, the extracted spectra are clustered into their corresponding class of endmembers. Such a clustering is achieved by using the k-means algorithm that considers the spectral angle distance or cosine distance [46]. The use of this distance is motivated by the fact that it is scale invariant [47]. The third and last stage consists in assigning the representative spectrum of each class endmember. For the best candidate for each class of pure materials, two strategies are considered: 1) Init-a: the candidate of each class is obtained by averaging the spectra composing each class of endmembers.

2) Init-b: the candidate of the associated class of endmembers is derived by selecting the median spectrum of each class of endmembers.

Moreover, the use of AEEB implies to set some parameters a priori including the number of pure materials present in each subset, the number of selected subsets and their corresponding size. For the sake of clarity, the performance of AEEB is directly related to the presence of a sufficient number of pure pixels in the processed HSI.

The Grad-SAV-LMM and Multi-SAV-LMM algorithms are described in Algorithm 1 and Algorithm 2, respectively.

Algorithm 1: Grad-SAV-LMM.   
Input: Hyperspectral image X   
1. Initialization stage   
1.1- Initialize $\widetilde { S _ { r } }$ from X by using the VCA method or   
by Init-a or Init-b.   
1.2- Set each element of $\tilde { C }$ by means of FCLS.   
2. Optimization stage (until reaching the predefined   
maximum number of iterations or satisfying (21))   
2.1- Optimize $\tilde { S } _ { r }$ by using (17).   
2.2- Optimize �ሚ by using (18) and (14).   
2.3- Optimize �෨ by using (19) and (15).   
2.4- Optimize �ሚ by using (20).   
Output: The estimated spectra of each class of   
endmembers and their associated abundance coefficients

Algorithm 2: Multi-SAV-LMM.   
Input: Hyperspectral image X   
1. Initialization stage   
1.1- Initialize $\widetilde { S _ { r } }$ from X by using the VCA method or by   
Init-a or Init-b.   
1.2- Set each element of $\tilde { C }$ by means of FCLS.   
2. Optimization stage (until reaching the predefined   
maximum number of iterations or satisfying (21))   
2.1- Optimize $\widetilde { S _ { r } }$ by using (24).   
2.2- Optimize $\tilde { A }$ by using (25) and (14).   
2.3- Optimize �෨ by using (26) and (15).   
2.4- Optimize $\tilde { C }$ by using (27).   
Output: The estimated spectra of each class of endmembers   
and their associated abundance coefficients.

## IV. TEST RESULTS

## A. Experimental setup and performance criteria

In this section, the performances of the designed algorithms are compared to some state-of-art methods. For the conducted experiments, we consider four methods, which consider ${ \mathrm { S V } } ,$ where their implementation details are briefly described in the following.

1) IP-NMF [35]: the endmembers are initialized by VCA, Inita and Init-b. As suggested by the authors in [35], the abundances are initialized by FCLS [48] and we fix the regularization parameter � to 30.

2) PLMM [26]: the abundances are initialized by FCLS, we set $\gamma = 1$ as suggested by the authors in [26]. � and $\beta$ are respectively set to 0.001 and 500. The endmember spectra are also initialized by VCA, Init -a and Init -b.

3) ELMM [27]: the endmember spectra are also initialized by VCA, Init-a and Init-b. The abundances are initialized by FCLS. We fix $\lambda _ { S } , \lambda _ { A }$ and $\lambda _ { \psi }$ to, respectively, 0.1, 0.01 and 0.001.

4) GLMM [28]: we set $\lambda _ { M } , \lambda _ { A }$ and $\lambda _ { \psi }$ to 0.1, 0.01 and 0.001 respectively. The endmember spectra are also initialized by VCA, Init-a and Init-b. The abundances are initialized by FCLS.

For the proposed techniques, we set the regularization parameters $\lambda _ { A }$ and $\lambda _ { B }$ in the following interval [0.1, 0.2, 0.4, 1, 2, 3, 4, 5, 10].

To assess the performances on the semi-synthetic dataset, the following measures are considered: Spectral Angle Mapper $( S A M _ { S } )$ Spectral Information Divergence $\left( S I D _ { S } \right)$ , and Normalized Mean Square Error $( N M S E _ { s } )$ , calculated between reference and estimated spectra, defined respectively by:

$$
\begin{array} { r } { { S A M _ { S } } = \frac { 1 } { { P M } } { \sum _ { i = 1 } ^ { P } } { \sum _ { j = 1 } ^ { M } } a r c c o s \ { \left( \frac { { \langle { { r _ { i j } } , { { \tilde { r } } _ { i j } } } \rangle } } { { { \left\| { { r _ { i j } } } \right\| _ { 2 } } \cdot \left\| { { { \tilde { r } } _ { i j } } } \right\| _ { 2 } } } \right) } , } \end{array}\tag{28}
$$

$$
\begin{array} { r l } & { { S I D } _ { S } = \frac { 1 } { P M } { \sum _ { i = 1 } ^ { P } } { \sum _ { j = 1 } ^ { M } } \bigl | r _ { i j } ^ { T } l n ( r _ { i j } \oslash \tilde { r } _ { i j } ) + } \\ & { \qquad \tilde { r } _ { i j } ^ { T } l n ( \tilde { r } _ { i j } \oslash r _ { i j } ) \bigr | , } \end{array}\tag{29}
$$

$$
\begin{array} { r } { N M S E _ { S } = \frac { 1 } { P M } \sum _ { i = 1 } ^ { P } \sum _ { j = 1 } ^ { M } \frac { \left. \boldsymbol { r } _ { i j } - \boldsymbol { \tilde { r } } _ { i j } \right. _ { 2 } } { \left. \boldsymbol { r } _ { i j } \right. _ { 2 } } , } \end{array}\tag{30}
$$

We also consider the abundance overall Root Mean Square Error $( R M S E _ { A } )$ , calculated between reference and estimated linear abundance fraction maps, defined by:

$$
\begin{array} { r } { R M S E _ { A } = \frac { 1 } { P } \sum _ { i = 1 } ^ { P } \sqrt { \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( c _ { j i } - \tilde { c } _ { j i } ) ^ { 2 } } , } \end{array}\tag{31}
$$

For the real data, since there is no ground truth, the following performance criteria are used: The reconstruction Spectral Angle Mapper $( S A M _ { X } ) _ { ; }$ , and the reconstruction RMSE $( R M S E _ { X } )$ , calculated between the considered original and reconstructed HSI images.

$$
\begin{array} { r } { { S A M } _ { X } = \frac { 1 } { P } { \sum } _ { i = 1 } ^ { P } a r c o s ( \ \frac { < x _ { . i } , \tilde { x } _ { . i } > } { \| x _ { . i } \| \| \tilde { x } _ { . i } \| } \ ) , } \end{array}\tag{32}
$$

$$
\begin{array} { r } { R M S E _ { X } = \frac { 1 } { P } \sum _ { i = 1 } ^ { P } \sqrt { \frac { 1 } { L } \sum _ { l = 1 } ^ { L } ( x _ { l i } - \tilde { x } _ { l i } ) ^ { 2 } } . } \end{array}\tag{33}
$$

For all the used criteria, the lower their values, the better the tested method.

## B. Test results with semi-synthetic data

For the conducted experiments, two spectral libraries are used. The first spectral library (SL-1) (see Fig. 1(a)) is composed of three classes of urban environment spectra obtained from ground measurements using a spectrometer during the French airborne UMBRA (Urban Material characterization in the sun and shade of Built-up structures and trees and their Retrieval from Airborne image) campaign [49]. This spectral library highlights well SV that can be present in real conditions particularly in urban scene. The three classes correspond to photovoltaic panels, tiles and grass. The pure material spectra contain 214 spectral bands after removing the atmospheric absorption bands [50] ranging from 400 to 2500 nm.

Five pure materials compose the second spectral library (SL-2) (see Fig. 1(b)), namely: railway, water, running track, healthy and synthetic grass. The considered hyperspectral endmember spectra with SV are derived from ground truth samples provided in [52]. These spectra, with 144 wavelengths, cover the spectral domain 0.36-1.04 μm.

These two spectral libraries enable one to generate synthetic data which are fairly representative of SV present in real HSIs. The first synthetic hyperspectral image (HI-1), generated using SL-1, contains 30×30 pixels. The abundances are obtained with the HYDRA toolbox [51].

The second synthetic hyperspectral image (HI-2) is generated following the same procedure as with HI-1, now using SL-2.

![](images/c9ff984ae93eeca5caf20dc143be7001abbe07f52686dcd6f7090bc7658d8355.jpg)  
(a)

![](images/3ca228b690394a78e3cddfef1a1f891cf9aed7f6d1001ff7d8f46dc92220835a.jpg)  
Fig. 1 Used spectral libraries in semi-synthetic data experiments: (a) SL-1, and (b) SL-2.

The considered semi-synthetic hyperspectral images are corrupted by white Gaussian noise with different signal-tonoise ratios (SNR) including 20, 25 and 30 dB.

The results for each configuration of noise level and used initialization for HI-1 are reported in Table I. The best results are shown in bold red and the second ones are in bold blue. First of all, for the classical initialization with VCA, we can see clearly that ELMM yields highest values of $S A M _ { S }$ in the configuration with a 20-dB SNR which demonstrates that SV is not only induced by topography and illumination changes, in particular in urban scenes like in our scenario. GLMM performs better since it has more flexibility. IP-NMF reaches even better values: this proves the effectiveness of the used extended version of LMM which considers SV in both spatial and spectral parts.

Table I demonstrates that the proposed initialization strategies (Init-a and Init-b) improve the performance of all tested methods, with substantial gains in both the spectral and spatial domains compared with the standard VCA-based initialization, particularly for HI-1 at SNR = 20 dB. These results confirm that considering SV during the initialization process reduces the impact of noise and decreases the sensitivity of the methods to their initialization.

Moreover, Table I shows that both proposed Grad-SAV-LMM and Multi-SAV-LMM methods outperform the state-of-the-art techniques in almost all tested configurations. Grad-SAV-LMM and Multi-SAV-LMM achieve the best results according to the considered spectral criteria, reaching the lowest values of $S A M _ { S }$ . Their significant performance improvements further highlight their robustness. In addition, Multi-SAV-LMM provides the best spatial performance, with the lowest values of $R M S E _ { A }$ across almost all SNR levels.

Indeed, the results in Table I show that the Grad-SAV-LMM and Multi-SAV-LMM methods generally offer the best performance for HI-1 hyperspectral unmixing. In particular, Grad-SAV-LMM generally achieves the lowest mean values for the considered evaluation criteria (example for Init-b and SNR=20 d $3 { \colon } S A M _ { S } = 5 . 6 1 ^ { \circ } , S I D _ { S } = 1 . 0 0 , N M S E _ { S } = 1 5 . 2 5 \%$ and $R M S E _ { A } = 0 . 0 9 4 3 )$ , while $\mathrm { M u l t i - S A V - L M M }$ often yields the best mean values in terms of the spatial metrics considered (example for Init-b and SNR=20: ��� = 6.11°, ��� = 1.15, $N M S E _ { S } = 1 6 . 2 3$ % and $R M S E _ { a } = 0 . 0 8 9 6 )$ . The improvement is particularly noticeable when the SNR increases from 20 to 30 dB, confirming the robustness of both approaches with respect to noise reduction.

Fig. 2 illustrates the obtained abundance maps for each method in the case of SNR=30 dB and VCA initialization. We can see that Multi-SAV-LMM retrieves better the abundances in particular for the first and third endmembers.

![](images/99fddbf2170825d968086786fe1bd05e0b654452fdff84121a43422f2c32b14b.jpg)  
Fig. 2 Obtained abundances maps for HI-1 using different techniques for SNR=30dB.

Table II reports the results obtained with the HI-2 dataset. Similar to what was observed for HI-1, the Init-a and Init-b strategies improve the performances of all considered state-ofthe-art methods, yielding better results in both spectral and spatial domains compared with the standard VCA-based initialization. These findings confirm that considering SV during the initialization process enhances the robustness of the tested approaches, particularly at high SNR levels.

Overall, the results reported in Table II show that the proposed Multi-SAV-LMM and Grad-SAV-LMM methods achieve the most competitive performance across all considered spectral and spatial criteria. For example, in the configuration with Initb and $S N R = 2 0 d B$ , the Grad-SAV-LMM method achieves the following performance: $S A M _ { S } = 4 . 2 5 ^ { \circ } , S I D _ { S } = 0 . 5 8 _ { ! }$ $N M S E _ { S } = 1 2 . 2 1$ % and $R M S E _ { a } = 0 . 1 0 0 1$ , the Multi-SAV-LMM method achieves the following performance: $S A M _ { S } =$ 4.31°, $S I D _ { S } = 0 . 6 0$ $N M S E _ { S } = 1 2 . 2 6 \%$ and $R M S E _ { a } =$ 0.0999. Besides, the IP-NMF and PLMM methods remain competitive in certain configurations. The consistent improvements achieved with Init-a and Init-b once again demonstrate the effectiveness of the proposed initialization strategies.

![](images/64b1151badeea9d38aba15662f0269177b43465f631020969b9d170deab47e6b.jpg)  
Fig. 3 Obtained abundances maps for HI-2 using different techniques for SNR=30dB.

Fig. 3 shows the estimated abundance maps for each method initialized by VCA in the case of SNR = 30 dB. As in the case of HI-1, PLMM fails to accurately retrieve the abundance maps. ELMM and GLMM provide more accurate estimations than PLMM. However, Fig. 3 illustrates that the proposed approaches produce superior results, particularly for Endmember 3, where the abundance maps generated by the proposed methods are closer to the ground truth compared to those produced by the other state-of-the-art techniques.

## C. Test results with real data

Two real hyperspectral images are considered in the conducted experiments. The first one (RHI-1) was acquired over the University of Houston, more precisely the Robertson stadium on the Houston Campus [30]. RHI-1 has 144 spectral bands in the 380 nm to 1050 nm spectral domain and is composed of 105×128 pixels. RHI-1 is constituted of four main classes of endmembers which are vegetation, concrete stands, asphalt and red metallic roofs [30]. The second considered real hyperspectral image (RHI-2) is the Jasper Ridge dataset. More precisely, it is a subset composed of 70×70 pixels and has 224 bands covering the wavelengths from 380 nm to 2500 nm. After removing channels 1–3, 108–112, 154–166, and 220–224 due to strong water vapor absorption and atmospheric effects, 198 spectral bands remain. This image is constituted of four main pure classes of materials, namely: water, dirt, tree and road.

TABLE I  
MEANS OF THE PERFORMANCE CRITERIA FOR THE HI-1 IMAGE
<table><tr><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>SNR</td><td rowspan=1 colspan=1>Initialization</td><td rowspan=1 colspan=1>Criterion</td><td rowspan=1 colspan=1>IP-NMF</td><td rowspan=1 colspan=1>PLMM</td><td rowspan=1 colspan=1>ELMM</td><td rowspan=1 colspan=1>GLMM</td><td rowspan=1 colspan=1>Grad-SAV-LMM</td><td rowspan=1 colspan=1>Multi-SAV-LMM</td></tr><tr><td rowspan=36 colspan=1>HI-1</td><td rowspan=12 colspan=1>20 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1> $\underline { { S A M _ { S } \left( ^ { \circ } \right) } }$ </td><td rowspan=1 colspan=1>7.55</td><td rowspan=1 colspan=1>7.91</td><td rowspan=1 colspan=1>8.67</td><td rowspan=1 colspan=1>7.93</td><td rowspan=1 colspan=1>6.54</td><td rowspan=1 colspan=1>6.60</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>3.73</td><td rowspan=1 colspan=1>2.10</td><td rowspan=1 colspan=1>2.56</td><td rowspan=1 colspan=1>1.68</td><td rowspan=1 colspan=1>2.80</td><td rowspan=1 colspan=1>1.40</td></tr><tr><td rowspan=1 colspan=1> $N M S E _ { S } \ ( \% )$ </td><td rowspan=1 colspan=1>23.58</td><td rowspan=1 colspan=1>22.94</td><td rowspan=1 colspan=1>20.69</td><td rowspan=1 colspan=1>18.09</td><td rowspan=1 colspan=1>17.36</td><td rowspan=1 colspan=1>17.45</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.116</td><td rowspan=1 colspan=1>0.3699</td><td rowspan=1 colspan=1>0.1027</td><td rowspan=1 colspan=1>0.1035</td><td rowspan=1 colspan=1>0.1133</td><td rowspan=1 colspan=1>0.0992</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1> $\underline { { S A M _ { S } \left( ^ { \circ } \right) } }$ </td><td rowspan=1 colspan=1>5.83</td><td rowspan=1 colspan=1>6.32</td><td rowspan=1 colspan=1>7.34</td><td rowspan=1 colspan=1>7.45</td><td rowspan=1 colspan=1>5.63</td><td rowspan=1 colspan=1>6.17</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=1>1.24</td><td rowspan=1 colspan=1>1.56</td><td rowspan=1 colspan=1>1.50</td><td rowspan=1 colspan=1>1.01</td><td rowspan=1 colspan=1>1.18</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } \left( \% \right) } }$ </td><td rowspan=1 colspan=1>18.28</td><td rowspan=1 colspan=1>17.68</td><td rowspan=1 colspan=1>17.78</td><td rowspan=1 colspan=1>17.37</td><td rowspan=1 colspan=1>15.29</td><td rowspan=1 colspan=1>16.37</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { A } } }$ </td><td rowspan=1 colspan=1>0.0990</td><td rowspan=1 colspan=1>0.3863</td><td rowspan=1 colspan=1>0.0975</td><td rowspan=1 colspan=1>0.0975</td><td rowspan=1 colspan=1>0.0977</td><td rowspan=1 colspan=1>0.0945</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>5.86</td><td rowspan=1 colspan=1>6.32</td><td rowspan=1 colspan=1>7.35</td><td rowspan=1 colspan=1>7.42</td><td rowspan=1 colspan=1>5.61</td><td rowspan=1 colspan=1>6.11</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>1.36</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.58</td><td rowspan=1 colspan=1>1.48</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.15</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } } } ~ ( \% )$ </td><td rowspan=1 colspan=1>17.95</td><td rowspan=1 colspan=1>17.42</td><td rowspan=1 colspan=1>17.74</td><td rowspan=1 colspan=1>17.28</td><td rowspan=1 colspan=1>15.25</td><td rowspan=1 colspan=1>16.23</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.0949</td><td rowspan=1 colspan=1>0.3848</td><td rowspan=1 colspan=1>0.0929</td><td rowspan=1 colspan=1>0.0925</td><td rowspan=1 colspan=1>0.0943</td><td rowspan=1 colspan=1>0.0896</td></tr><tr><td rowspan=12 colspan=1>25 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>7.06</td><td rowspan=1 colspan=1>7.22</td><td rowspan=1 colspan=1>7.43</td><td rowspan=1 colspan=1>7.14</td><td rowspan=1 colspan=1>6.53</td><td rowspan=1 colspan=1>6.56</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>2.16</td><td rowspan=1 colspan=1>1.90</td><td rowspan=1 colspan=1>2.24</td><td rowspan=1 colspan=1>1.74</td><td rowspan=1 colspan=1>1.90</td><td rowspan=1 colspan=1>1.68</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>18.59</td><td rowspan=1 colspan=1>19.22</td><td rowspan=1 colspan=1>17.68</td><td rowspan=1 colspan=1>17.14</td><td rowspan=1 colspan=1>17.39</td><td rowspan=1 colspan=1>17.79</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { A } } }$ </td><td rowspan=1 colspan=1>0.1084</td><td rowspan=1 colspan=1>0.3825</td><td rowspan=1 colspan=1>0.1062</td><td rowspan=1 colspan=1>0.1048</td><td rowspan=1 colspan=1>0.1050</td><td rowspan=1 colspan=1>0.0978</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1>SAM_S}(°)</td><td rowspan=1 colspan=1>5.69</td><td rowspan=1 colspan=1>5.95</td><td rowspan=1 colspan=1>6.08</td><td rowspan=1 colspan=1>6.13</td><td rowspan=1 colspan=1>5.43</td><td rowspan=1 colspan=1>5.61</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>1.36</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.33</td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>1.11</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>16.33</td><td rowspan=1 colspan=1>16.44</td><td rowspan=1 colspan=1>15.93</td><td rowspan=1 colspan=1>15.40</td><td rowspan=1 colspan=1>14.23</td><td rowspan=1 colspan=1>15.32</td></tr><tr><td rowspan=1 colspan=1>RMSEA</td><td rowspan=1 colspan=1>0.0888</td><td rowspan=1 colspan=1>0.3991</td><td rowspan=1 colspan=1>0.0879</td><td rowspan=1 colspan=1>0.0876</td><td rowspan=1 colspan=1>0.0871</td><td rowspan=1 colspan=1>0.0850</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1>SAMs} (°)</td><td rowspan=1 colspan=1>5.79</td><td rowspan=1 colspan=1>5.98</td><td rowspan=1 colspan=1>6.17</td><td rowspan=1 colspan=1>6.26</td><td rowspan=1 colspan=1>5.49</td><td rowspan=1 colspan=1>5.78</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>1.55</td><td rowspan=1 colspan=1>1.33</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>1.02</td><td rowspan=1 colspan=1>1.13</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } \left( \% \right) } }$ </td><td rowspan=1 colspan=1>17.28</td><td rowspan=1 colspan=1>16.60</td><td rowspan=1 colspan=1>16.42</td><td rowspan=1 colspan=1>15.84</td><td rowspan=1 colspan=1>14.31</td><td rowspan=1 colspan=1>15.71</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { A } } }$ </td><td rowspan=1 colspan=1>0.0922</td><td rowspan=1 colspan=1>0.4039</td><td rowspan=1 colspan=1>0.0908</td><td rowspan=1 colspan=1>0.0903</td><td rowspan=1 colspan=1>0.0892</td><td rowspan=1 colspan=1>0.0873</td></tr><tr><td rowspan=12 colspan=1>30 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1>SAM_s (°)</td><td rowspan=1 colspan=1>7.00</td><td rowspan=1 colspan=1>7.23</td><td rowspan=1 colspan=1>6.98</td><td rowspan=1 colspan=1>6.64</td><td rowspan=1 colspan=1>6.32</td><td rowspan=1 colspan=1>6.40</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>2.44</td><td rowspan=1 colspan=1>2.08</td><td rowspan=1 colspan=1>2.31</td><td rowspan=1 colspan=1>1.63</td><td rowspan=1 colspan=1>1.87</td><td rowspan=1 colspan=1>1.74</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>19.46</td><td rowspan=1 colspan=1>19.44</td><td rowspan=1 colspan=1>17.52</td><td rowspan=1 colspan=1>16.62</td><td rowspan=1 colspan=1>17.21</td><td rowspan=1 colspan=1>17.78</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { A } } }$ </td><td rowspan=1 colspan=1>0.1070</td><td rowspan=1 colspan=1>0.3706</td><td rowspan=1 colspan=1>0.1054</td><td rowspan=1 colspan=1>0.1013</td><td rowspan=1 colspan=1>0.1042</td><td rowspan=1 colspan=1>0.0989</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>5.67</td><td rowspan=1 colspan=1>6.38</td><td rowspan=1 colspan=1>5.48</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>5.27</td><td rowspan=1 colspan=1>5.43</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>1.46</td><td rowspan=1 colspan=1>1.33</td><td rowspan=1 colspan=1>1.26</td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>1.09</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>16.63</td><td rowspan=1 colspan=1>18.51</td><td rowspan=1 colspan=1>15.14</td><td rowspan=1 colspan=1>14.61</td><td rowspan=1 colspan=1>13.98</td><td rowspan=1 colspan=1>15.57</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.0885</td><td rowspan=1 colspan=1>0.3843</td><td rowspan=1 colspan=1>0.0875</td><td rowspan=1 colspan=1>0.0871</td><td rowspan=1 colspan=1>0.0866</td><td rowspan=1 colspan=1>0.0855</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1> $\underline { { S A M _ { S } \left( ^ { \circ } \right) } }$ </td><td rowspan=1 colspan=1>5.72</td><td rowspan=1 colspan=1>5.87</td><td rowspan=1 colspan=1>5.51</td><td rowspan=1 colspan=1>5.59</td><td rowspan=1 colspan=1>5.29</td><td rowspan=1 colspan=1>5.52</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>1.63</td><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=1>1.35</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>1.10</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>17.71</td><td rowspan=1 colspan=1>17.04</td><td rowspan=1 colspan=1>15.75</td><td rowspan=1 colspan=1>14.86</td><td rowspan=1 colspan=1>14.12</td><td rowspan=1 colspan=1>15.86</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.0889</td><td rowspan=1 colspan=1>0.4074</td><td rowspan=1 colspan=1>0.0879</td><td rowspan=1 colspan=1>0.0872</td><td rowspan=1 colspan=1>0.0863</td><td rowspan=1 colspan=1>0.0850</td></tr></table>

TABLE II  
MEANS OF THE PERFORMANCE CRITERIA FOR THE HI-2 IMAGE
<table><tr><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>SNR</td><td rowspan=1 colspan=1>Initialization</td><td rowspan=1 colspan=1>Criterion</td><td rowspan=1 colspan=1>IP-NMF</td><td rowspan=1 colspan=1>PLMM</td><td rowspan=1 colspan=1>ELMM</td><td rowspan=1 colspan=1>GLMM</td><td rowspan=1 colspan=1>Grad-SAV-LMM</td><td rowspan=1 colspan=1>Multi-SAV-LMM</td></tr><tr><td rowspan=36 colspan=1>HI-2</td><td rowspan=12 colspan=1>20 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.90</td><td rowspan=1 colspan=1>5.43</td><td rowspan=1 colspan=1>6.85</td><td rowspan=1 colspan=1>7.81</td><td rowspan=1 colspan=1>4.72</td><td rowspan=1 colspan=1>5.34</td></tr><tr><td rowspan=1 colspan=1> $\underline { { S I D _ { S } } }$ </td><td rowspan=1 colspan=1>1.19</td><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } } } ( \% )$ </td><td rowspan=1 colspan=1>18.07</td><td rowspan=1 colspan=1>17.78</td><td rowspan=1 colspan=1>17.83</td><td rowspan=1 colspan=1>18.74</td><td rowspan=1 colspan=1>17.45</td><td rowspan=1 colspan=1>14.21</td></tr><tr><td rowspan=1 colspan=1>RMSEA</td><td rowspan=1 colspan=1>0.1104</td><td rowspan=1 colspan=1>0.3249</td><td rowspan=1 colspan=1>0.1182</td><td rowspan=1 colspan=1>0.1170</td><td rowspan=1 colspan=1>0.1209</td><td rowspan=1 colspan=1>0.107</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.45</td><td rowspan=1 colspan=1>4.88</td><td rowspan=1 colspan=1>6.42</td><td rowspan=1 colspan=1>7.21</td><td rowspan=1 colspan=1>4.20</td><td rowspan=1 colspan=1>4.80</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>1.04</td><td rowspan=1 colspan=1>1.26</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.66</td></tr><tr><td rowspan=1 colspan=1>NMSEs (%)</td><td rowspan=1 colspan=1>13.29</td><td rowspan=1 colspan=1>13.13</td><td rowspan=1 colspan=1>17.39</td><td rowspan=1 colspan=1>17.74</td><td rowspan=1 colspan=1>12.52</td><td rowspan=1 colspan=1>12.09</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.1045</td><td rowspan=1 colspan=1>0.3486</td><td rowspan=1 colspan=1>0.1186</td><td rowspan=1 colspan=1>0.1176</td><td rowspan=1 colspan=1>0.1013</td><td rowspan=1 colspan=1>0.0946</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.73</td><td rowspan=1 colspan=1>5.09</td><td rowspan=1 colspan=1>6.59</td><td rowspan=1 colspan=1>7.26</td><td rowspan=1 colspan=1>4.25</td><td rowspan=1 colspan=1>4.31</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { s }$ </td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>13.16</td><td rowspan=1 colspan=1>12.94</td><td rowspan=1 colspan=1>16.72</td><td rowspan=1 colspan=1>17.13</td><td rowspan=1 colspan=1>12.21</td><td rowspan=1 colspan=1>12.26</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=1>0.3480</td><td rowspan=1 colspan=1>0.1171</td><td rowspan=1 colspan=1>0.1162</td><td rowspan=1 colspan=1>0.1001</td><td rowspan=1 colspan=1>0.0999</td></tr><tr><td rowspan=12 colspan=1>25 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.99</td><td rowspan=1 colspan=1>4.97</td><td rowspan=1 colspan=1>6.12</td><td rowspan=1 colspan=1>6.64</td><td rowspan=1 colspan=1>4.35</td><td rowspan=1 colspan=1>4.81</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } \left( \% \right) } }$ </td><td rowspan=1 colspan=1>14.43</td><td rowspan=1 colspan=1>13.04</td><td rowspan=1 colspan=1>16.31</td><td rowspan=1 colspan=1>18.30</td><td rowspan=1 colspan=1>12.19</td><td rowspan=1 colspan=1>12.91</td></tr><tr><td rowspan=1 colspan=1>RMSEA</td><td rowspan=1 colspan=1>0.1152</td><td rowspan=1 colspan=1>0.3261</td><td rowspan=1 colspan=1>0.1160</td><td rowspan=1 colspan=1>0.1149</td><td rowspan=1 colspan=1>0.0902</td><td rowspan=1 colspan=1>0.0942</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1>SAM_S}(°)</td><td rowspan=1 colspan=1>4.25</td><td rowspan=1 colspan=1>4.37</td><td rowspan=1 colspan=1>5.25</td><td rowspan=1 colspan=1>5.47</td><td rowspan=1 colspan=1>4.25</td><td rowspan=1 colspan=1>4.37</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.55</td></tr><tr><td rowspan=1 colspan=1>NMSES (%)</td><td rowspan=1 colspan=1>13.23</td><td rowspan=1 colspan=1>12.33</td><td rowspan=1 colspan=1>14.08</td><td rowspan=1 colspan=1>13.58</td><td rowspan=1 colspan=1>11.71</td><td rowspan=1 colspan=1>11.93</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.1007</td><td rowspan=1 colspan=1>0.3400</td><td rowspan=1 colspan=1>0.1035</td><td rowspan=1 colspan=1>0.1031</td><td rowspan=1 colspan=1>0.0897</td><td rowspan=1 colspan=1>0.0897</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1>SAM{S (°)</td><td rowspan=1 colspan=1>4.27</td><td rowspan=1 colspan=1>4.34</td><td rowspan=1 colspan=1>5.31</td><td rowspan=1 colspan=1>5.90</td><td rowspan=1 colspan=1>4.16</td><td rowspan=1 colspan=1>4.28</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.6765</td><td rowspan=1 colspan=1>0.5581</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.51</td></tr><tr><td rowspan=1 colspan=1> $N M S E _ { S } \left( \% \right)$ </td><td rowspan=1 colspan=1>13.58</td><td rowspan=1 colspan=1>12.35</td><td rowspan=1 colspan=1>14.15</td><td rowspan=1 colspan=1>14.19</td><td rowspan=1 colspan=1>11.60</td><td rowspan=1 colspan=1>11.68</td></tr><tr><td rowspan=1 colspan=1>RMSEA</td><td rowspan=1 colspan=1>0.1055</td><td rowspan=1 colspan=1>0.3404</td><td rowspan=1 colspan=1>0.1073</td><td rowspan=1 colspan=1>0.1068</td><td rowspan=1 colspan=1>0.0887</td><td rowspan=1 colspan=1>0.0893</td></tr><tr><td rowspan=12 colspan=1>30 dB</td><td rowspan=4 colspan=1>VCA</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>5.09</td><td rowspan=1 colspan=1>5.06</td><td rowspan=1 colspan=1>5.44</td><td rowspan=1 colspan=1>5.35</td><td rowspan=1 colspan=1>4.66</td><td rowspan=1 colspan=1>4.52</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>1.06</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>1.07</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.77</td></tr><tr><td rowspan=1 colspan=1>NMSEs (%)</td><td rowspan=1 colspan=1>15.30</td><td rowspan=1 colspan=1>13.10</td><td rowspan=1 colspan=1>16.06</td><td rowspan=1 colspan=1>15.64</td><td rowspan=1 colspan=1>12.878</td><td rowspan=1 colspan=1>13.00</td></tr><tr><td rowspan=1 colspan=1>RMSEA</td><td rowspan=1 colspan=1>0.1149</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.1194</td><td rowspan=1 colspan=1>0.1178</td><td rowspan=1 colspan=1>0.0923</td><td rowspan=1 colspan=1>0.0932</td></tr><tr><td rowspan=4 colspan=1>Init-a</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.15</td><td rowspan=1 colspan=1>4.21</td><td rowspan=1 colspan=1>4.60</td><td rowspan=1 colspan=1>4.79</td><td rowspan=1 colspan=1>4.23</td><td rowspan=1 colspan=1>4.06</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.51</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } \left( \% \right) } }$ </td><td rowspan=1 colspan=1>13.76</td><td rowspan=1 colspan=1>12.65</td><td rowspan=1 colspan=1>13.47</td><td rowspan=1 colspan=1>12.80</td><td rowspan=1 colspan=1>12.17</td><td rowspan=1 colspan=1>11.31</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.09723</td><td rowspan=1 colspan=1>0.3338</td><td rowspan=1 colspan=1>0.09946</td><td rowspan=1 colspan=1>0.09906</td><td rowspan=1 colspan=1>0.0850</td><td rowspan=1 colspan=1>0.0832</td></tr><tr><td rowspan=4 colspan=1>Init-b</td><td rowspan=1 colspan=1> $S A M _ { S } \left( ^ { \circ } \right)$ </td><td rowspan=1 colspan=1>4.31</td><td rowspan=1 colspan=1>4.34</td><td rowspan=1 colspan=1>4.75</td><td rowspan=1 colspan=1>4.92</td><td rowspan=1 colspan=1>4.35</td><td rowspan=1 colspan=1>4.10</td></tr><tr><td rowspan=1 colspan=1> $S I D _ { S }$ </td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.58</td></tr><tr><td rowspan=1 colspan=1> $\underline { { N M S E _ { S } } } ( \% )$ </td><td rowspan=1 colspan=1>14.77</td><td rowspan=1 colspan=1>13.42</td><td rowspan=1 colspan=1>13.71</td><td rowspan=1 colspan=1>13.45</td><td rowspan=1 colspan=1>12.78</td><td rowspan=1 colspan=1>11.89</td></tr><tr><td rowspan=1 colspan=1> $R M S E _ { A }$ </td><td rowspan=1 colspan=1>0.1006</td><td rowspan=1 colspan=1>0.3354</td><td rowspan=1 colspan=1>0.1023</td><td rowspan=1 colspan=1>0.1018</td><td rowspan=1 colspan=1>0.0841</td><td rowspan=1 colspan=1>0.0834</td></tr></table>

![](images/8b5b680b15d3339e56ea8b6f02acc65544c28e8f0c3f0dab7c29d98312c0e112.jpg)

![](images/af54da5a41c243f2325381845054919a5f930a8c83b2ecdc3d4bfea92ec38441.jpg)  
RGB representation RHI-1  
RGB representation RHI-2

Fig. 4 True color composite representation of the used real HSIs.

Moreover, since Init-a proves to be highly effective in terms of performances, it is used as the initialization for all the tested methods for the real data. Table III presents the results obtained by the tested methods on both real datasets.

The results show that GLMM consistently achieves the best performance for both the RHI-1 and RHI-2 real datasets. For ${ \mathrm { R H I - } } 1 .$ GLMM achieves the lowest average values, with a $S A M _ { X }$ equal to $0 . 1 0 ^ { \circ }$ and a $R M S E _ { X }$ equal to 0.0005. It is followed by ELMM, with a $S A M _ { X }$ equal to $0 . 2 1 ^ { \circ }$ and a $R M S E _ { X }$ equal to 0.0007.

For RHI-2, the same trend is observed: GLMM continues to perform best with a $S A M _ { X }$ equal to $0 . 2 9 ^ { \circ }$ and a $R M S E _ { X }$ equal to 0.0006, while ELMM ranks second a $S A M _ { X }$ equal to $0 . 5 1 ^ { \circ }$ and a $R M S E _ { X }$ equal to 0.0019. Conversely, IP-NMF yields the highest values for both datasets, indicating the poorest performance among the compared methods.

However, these considered criteria are not entirely representative of the overall performances. Indeed, it should be noted that reconstruction criteria are not always reliable indicators, as it is possible to inaccurately estimate the spectra and abundances, yet still obtain a good product of the two matrices.

Regarding the proposed methods, the following performance was obtained: for the RHI-1 real dataset, the Grad-SAV-LMM method achieved an average value of $S A M _ { X }$ equal to $1 . 7 0 ^ { \circ }$ and an average value of $R M S E _ { X }$ equal to $0 . 0 1 3 1$ , while the Multi-SAV-LMM method achieved a mean $S A M _ { X }$ equal to $1 . 4 8 ^ { \circ }$ and a mean $R M S E _ { X }$ equal to 0.0048. The Multi-SAV-LMM method thus performs better than the Grad-SAV-LMM for that image, particularly for the $R M S E _ { X }$ criterion. For ${ \mathrm { R H I - } } 2 ,$ the values increase for both methods: an average value $\operatorname { o f } S A M _ { X }$ equal to $3 . 7 9 ^ { \circ }$ and a $R M S E _ { X }$ equal to $0 . 0 2 2 3$ for Grad-SAV-LMM, compared to a $S A M _ { X }$ equal to $4 . 1 7 ^ { \circ }$ and a $R M S E _ { X }$ equal to 0.0240 for Multi-SAV-LMM.

TABLE III  
MEANS OF THE PERFORMANCE CRITERIA FOR RHI-1 AND RHI-2 IMAGES
<table><tr><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Initialization</td><td rowspan=1 colspan=1>Criterion</td><td rowspan=1 colspan=1>IP-NMF</td><td rowspan=1 colspan=1>PLMM</td><td rowspan=1 colspan=1>ELMM</td><td rowspan=1 colspan=1>GLMM</td><td rowspan=1 colspan=1>Grad-SAV-LMM</td><td rowspan=1 colspan=1>Multi-SAV-LMM</td></tr><tr><td rowspan=2 colspan=1>RHI-1</td><td rowspan=2 colspan=1>Init-a</td><td rowspan=1 colspan=1> $\underline { { S A M _ { X } \left( ^ { \circ } \right) } }$ </td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>1.70</td><td rowspan=1 colspan=1>1.48</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { X } } }$ </td><td rowspan=1 colspan=1>0.0163</td><td rowspan=1 colspan=1>0.0037</td><td rowspan=1 colspan=1>0.0007</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>0.0131</td><td rowspan=1 colspan=1>0.0048</td></tr><tr><td rowspan=2 colspan=1>RHI-2</td><td rowspan=2 colspan=1>Init-a</td><td rowspan=1 colspan=1> $\underline { { S A M _ { X } \left( ^ { \circ } \right) } }$ </td><td rowspan=1 colspan=1>5.65</td><td rowspan=1 colspan=1>2.23</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>3.79</td><td rowspan=1 colspan=1>4.17</td></tr><tr><td rowspan=1 colspan=1> $\underline { { R M S E _ { X } } }$ </td><td rowspan=1 colspan=1>0.0282</td><td rowspan=1 colspan=1>0.0110</td><td rowspan=1 colspan=1>0.0019</td><td rowspan=1 colspan=1>0.0006</td><td rowspan=1 colspan=1>0.0223</td><td rowspan=1 colspan=1>0.0240</td></tr></table>

![](images/e7c039e0e2b28a66df923bf00dfa5a2d31fe54bbf8988f54e100278d64b03a25.jpg)  
Fig. 5 Obtained abundance maps for RHI-1.

The abundance maps estimated by each method are shown in Fig. 5 and Fig. 6. For RHI-1, the estimated abundance maps demonstrate the effectiveness of the proposed approaches. In particular, the superiority of the introduced techniques is evident in the representation of concrete and asphalt, where both proposed methods produce results that closely resemble the RGB representation of RHI-1 provided in Fig. 4.

Indeed, asphalt and concrete are more difficult to distinguish when using GLMM and ELMM, on these specific processed real-world data, highlighting the advantage of the proposed approaches in capturing spectral differences. This confirms that the proposed model and associated methods are more able to handle SV and better adapted to the complexity of real urban scenes.

![](images/e469af5830cba3d7c553026fbffb8351161f760915579f0f18079dc94586c7a1.jpg)  
Fig. 6 Obtained abundance maps for RHI-2.

RHI-2 represents a more rural environment, allowing the evaluation of the proposed methods in a different context with distinct land cover types (water, dirt, tree, and road, as stated above). The proposed approaches achieve good results. The abundance maps estimated by the proposed methods are consistent with the RGB representation of the reference image (Fig. 4). These results confirm the adaptability and effectiveness of the proposed approaches across various real-world scenarios.

## V. CONCLUSION

In this paper, we have introduced two novel algorithms based on NMF that optimize an additive linear mixing model, namely Grad-SAV-LMM and Multi-SAV-LMM, for hyperspectral spectral unmixing explicitly addressing spectral variability. Extensive experiments conducted on synthetic data and two real datasets show that these methods achieve accurate estimation of abundance maps while obtaining good results in terms of spectral performances. Their ability to model spectral variability leads to more precise and reliable unmixing, especially in complex real-world environments. These methods also demonstrate adaptability across different types of scenes, making them useful for various remote sensing applications such as environmental monitoring and urban planning. Additionally, we introduce a novel initialization process that improves the results of the proposed algorithms and the tested state-of-the-art techniques. The results obtained in this study open up several promising areas of research in the future. Further work could aim to extend the proposed approaches to more complex and realistic scenarios, improve their robustness and computational efficiency, and evaluate their performance under various experimental conditions. Additionally, future studies could explore the integration of advanced optimization or machine learning techniques to improve the accuracy and generalization capabilities of the proposed methods.

## REFERENCES

[1] J.M. Bioucas-Dias, A. Plaza, N. Dobigeon, M. Parente, Q. Du, P. Gader, and J. Chanussot, “Hyperspectral Unmixing Overview: Geometrical, Statistical, and Sparse Regression-Based Approaches,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 5(2), pp. 354-379, 2012. doi: 10.1109/JSTARS.2012.2194696.

[2] J. M. Bioucas-Dias, A. Plaza, G. Camps-Valls, P. Scheunders, N. Nasrabadi, and J. Chanussot, “Hyperspectral Remote Sensing Data Analysis and Future Challenges,” IEEE Geosci Remote Sens Mag, vol. 1, no. 2, pp. 6–36, 2013, doi: 10.1109/MGRS.2013.2244672.

[3] Y. Deville; S.E. Brezini; F. Z. Benhalouche; M. S. Karoui; M. Guillaume; X. Lenot et al., “Modeling and Unsupervised Unmixing Based on Spectral Variability for Hyperspectral Oceanic Remote Sensing Data with Adjacency Effects,” Remote Sensing, vol. 15, no. 18, 2023, doi: 10.3390/rs15184583.

[4] M. S. Karoui, F Z Benhalouche, Yannick Deville, et al., “Partial linear NMF-based unmixing methods for detection and area estimation of photovoltaic panels in urban hyperspectral remote sensing data,” Remote Sensing, vol. 11, no. 18, 2019, doi: 10.3390/rs11182164.

[5] N. Akhtar, F. Shafait, and A. Mian, “Sparse spatio-spectral representation for hyperspectral image super-resolution,” in in Proc. European Conf. Computer Vision, 2014, pp. 63–78. doi: 10.1007/978-3-319-10584-0\_5.

[6] S. E. Brezini and Y. Deville, “Hyperspectral and Multispectral Image Fusion with Automated Extraction of Image-Based Endmember Bundles and Sparsity-Based Unmixing to Deal with Spectral Variability,” Sensors, vol. 23, no. 4, 2023, doi: 10.3390/s23042341.

[7] N. Keshava and J. F. Mustard, “Spectral unmixing,” IEEE Signal Process Mag, vol. 19, no. 1, pp. 44–57, 2002, doi: 10.1109/79.974727.

[8] P. Comon and C. Jutten, Handbook of Blind Source Separation: Independent Component Analysis and Applications. Academic Press, 2010.

[9] Y. Deville, “Blind Source Separation and Blind Mixture Identification Methods,” in Wiley Encyclopedia of Electrical and Electronics Engineering, 2016, pp. 1–3.

[10]R. Borsoi, T. Imbiriba, J. C. Bermudez, C. Richard, J. Chanussot, L. Drumetz, J.-Y. Tourneret, A. Zare, and C. Jutten, “Spectral Variability in Hyperspectral Data Unmixing: A Comprehensive Review,” IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 4, pp. 223–270, 2021, doi:10.1109/MGRS.2021.3071158.

[11] Y. Deville, G. Faury, V. Achard, and X. Briottet, “An NMF-based method for jointly handling mixture nonlinearity and intraclass variability in hyperspectral blind source separation,” Digit Signal Process, vol. 133, p. 103838, 2023, doi: https://doi.org/10.1016/j.dsp.2022.103838.

[12] A. Zare and K. C. Ho, “Endmember Variability in Hyperspectral Analysis: Addressing Spectral Variability During Spectral Unmixing,” IEEE Signal Process Mag, vol. 31, no. 1, pp. 95–104, 2014, doi: 10.1109/MSP.2013.2279177.

[13] B. Somers, G. P. Asner, L. Tits, and P. Coppin, “Endmember variability in Spectral Mixture Analysis: A review,” Remote Sens Environ, vol. 115, no. 7, pp. 1603–1616, 2011, doi: https://doi.org/10.1016/j.rse.2011.03.003.

[14] N. Dobigeon, J. Y. Tourneret, C. Richard, J. C. M. Bermudez, S. McLaughlin, and A. O. Hero, “Nonlinear unmixing of hyperspectral images: Models and algorithms,” IEEE Signal Process Mag, vol. 31, no. 1, pp. 82–94, 2014, doi: 10.1109/MSP.2013.2279274.

[15] I. Meganem, Y. Deville, S. Hosseini, P. Déliot, and X. Briottet, “Linearquadratic blind source separation using NMF to unmix urban hyperspectral images,” IEEE Transactions on Signal Processing, vol. 62, no. 7, pp. 1822–1833, 2014, doi: 10.1109/TSP.2014.2306181.

[16] F. Z. Benhalouche, Y. Deville, M. S. Karoui, and A. Ouamri, “Hyperspectral Unmixing Based on Constrained Bilinear or Linear-Quadratic Matrix Factorization,” Remote Sensing, vol. 13, no. 11, 2021, doi: https://doi.org/10.3390/rs13112132.

[17] D. A. Roberts, M. Gardner, R. Church, S. Ustin, G. Scheer, and R. O. Green, “Mapping chaparral in the Santa Monica Mountains using multiple endmember spectral mixture models,” Remote Sens Environ, vol. 65, no. 3, pp. 267–279, 1998, doi: 10.1016/S0034-4257(98)00037-6.

[18] B. Somers, M. Zortea, A. Plaza, and G. P. Asner, “Automated Extraction of Image-Based Endmember Bundles for Improved Spectral Unmixing,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 5, no. 2, pp. 396–408, 2012, doi: 10.1109/JSTARS.2011.2181340.

[19] M. Xu, L. Zhang, and B. Du, “An Image-Based Endmember Bundle Extraction Algorithm Using Both Spatial and Spectral Information,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 8, no. 6, pp. 2607–2617, 2015, doi: 10.1109/JSTARS.2014.2373491.

[20] T. Uezato, M. Fauvel, and N. Dobigeon, “Hyperspectral Unmixing with Spectral Variability Using Adaptive Bundles and Double Sparsity,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 6, pp. 3980–3992, 2019, doi: 10.1109/TGRS.2018.2889256.

[21] S. Ghanbari Azar, S. Meshgini, S. Beheshti, and T. Yousefi Rezaii, “Linear mixing model with scaled bundle dictionary for hyperspectral unmixing with spectral variability,” Signal Processing, vol. 188, p. 108214, 2021, doi: https://doi.org/10.1016/j.sigpro.2021.108214.

[22] A. Halimi, N. Dobigeon, and J.-Y. Tourneret, “Unsupervised Unmixing of Hyperspectral Images Accounting for Endmember Variability,” IEEE Transactions on Image Processing, vol. 24, no. 12, pp. 4904–4917, 2015, doi: 10.1109/TIP.2015.2471182.

[23] Y. Zhou, E. B. Wetherley, and P. D. Gader, “Unmixing urban hyperspectral imagery using probability distributions to represent endmember variability,” Remote Sens Environ, vol. 246, p. 111857, 2020, doi: https://doi.org/10.1016/j.rse.2020.111857.

[24] H. Liu, Y. Lu, Z. Wu, Q. Du, J. Chanussot, and Z. Wei, “Bayesian Unmixing of Hyperspectral Image Sequence with Composite Priors for Abundance and Endmember Variability,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022, doi: 10.1109/TGRS.2021.3064708.

[25] H. Liu, Y. Lu, Z. Wu, Q. Du, J. Chanussot, and Z. Wei, “Spectral variability Bayesian unmixing for hyperspectral sequence in wavelet domain,” IEEE Transactions on Geoscience and Remote Sensing, p. 1, 2023, doi: 10.1109/TGRS.2023.3310982.

[26] P. A. Thouvenin, N. Dobigeon, and J. Y. Tourneret, “Hyperspectral unmixing with spectral variability using a perturbed linear mixing model,” IEEE Transactions on Signal Processing, vol. 64, no. 2, pp. 525– 538, 2016, doi: 10.1109/TSP.2015.2486746.

[27] L. Drumetz, S. Henrot, M. A. Veganzones, J. Chanussot, and C. Jutten, “Blind hyperspectral unmixing using an extended linear mixing model to address spectral variability,” IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3890–3905, 2016, doi: 10.1109/TIP.2016.2579259.

[28] T. Imbiriba, R. A. Borsoi, and J. C. M. Bermudez, “Generalized Linear Mixing Model Accounting for Endmember Variability,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, Canada, 2018, pp. 1862–1866. doi: 10.1109/ICASSP.2018.8462214.

[29] D. Hong, N. Yokoya, J. Chanussot, and X. X. Zhu, “An Augmented Linear Mixing Model to Address Spectral Variability for Hyperspectral Unmixing,” IEEE Transactions on Image Processing, vol. 28, no. 4, pp. 1923–1938, 2019, doi: 10.1109/TIP.2018.2878958.

[30] F. Z. Benhalouche, M. S. Karoui, and Y. Deville, “Gradient-Based NMF Methods for Hyperspectral Unmixing Addressing Spectral Variability with a Multiplicative-Tuning Linear Mixing Model,” in IEEE International Geoscience and Remote Sensing Symposium IGARSS, 2021, pp. 3205–3208. doi: 10.1109/IGARSS47720.2021.9554445.

[31] M. S. Karoui, F. Z. Benhalouche, and Y. Deville, “A Gradient-Based Method for the Modified Augmented Linear Mixing Model Addressing Spectral Variability for Hyperspectral Unmixing,” in IEEE International Geoscience and Remote Sensing Symposium, IGARSS 2022, pp. 3279– 3282. doi: 10.1109/IGARSS46834.2022.9883849.

[32] T. Imbiriba, R. A. Borsoi, and J. C. M. Bermudez, “Low-Rank Tensor Modeling for Hyperspectral Unmixing Accounting for Spectral Variability,” IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 3, pp. 1833–1842, 2020, doi: 10.1109/TGRS.2019.2949543.

[33] R. A. Borsoi, T. Imbiriba, and J. C. M. Bermudez, “A Data Dependent Multiscale Model for Hyperspectral Unmixing with Spectral Variability,” IEEE Transactions on Image Processing, vol. 29, pp. 3638–3651, 2020, doi: 10.1109/TIP.2020.2963959.

[34] G. Zhang, S. Mei, B. Xie, M. Ma, Y. Zhang, Y. Feng et al., “Spectral Variability Augmented Sparse Unmixing of Hyperspectral Images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1– 13, 2022, doi:10.1109/TGRS.2022.3169228.

[35] C. Revel, Y. Deville, V. Achard, X. Briottet, and C. Weber, “Inertia-Constrained Pixel-by-Pixel Nonnegative Matrix Factorisation: A Hyperspectral Unmixing Method Dealing with Intra-Class Variability,” 2018. doi: 10.3390/rs10111706.

[36] Y. Deville; S.E. Brezini; F. Z. Benhalouche; M. S. Karoui; M. Guillaume; X. Lenot et al., “Hyperspectral Oceanic Remote Sensing with Adjacency Effects: From Spectral-Variability-Based Modeling to Performance of Associated Blind Unmixing Methods,” in IEEE International Geoscience and Remote Sensing Symposium, IGARSS 2019, pp. 82–285. doi: 10.1109/IGARSS.2019.8898430.

[37] S. E. Brezini, M. S. Karoui, F. Z. Benhalouche, Y. Deville, and A. Ouamri, “A pixel-by-pixel NMF-based method for hyperspectral unmixing using a new linear mixing model to address additively-tuned spectral variability,” in Image and Signal Processing for Remote Sensing XXV, International Society for Optics and Photonics, Strasbourg, France, 2019, p. 111550I. doi: /10.1117/12.2533159.

[38] S. E. Brezini, M. S. Karoui, F. Z. Benhalouche, Y. Deville, and A. Ouamri, “An NMF-Based Method for Hyperspectral Unmixing Using A Structured Additively-Tuned Linear Mixing Model to Address Spectral Variability,” in Mediterranean and Middle-East Geoscience and Remote Sensing Symposium, M2GARSS 2020 - Proceedings, Tunis, 2020, pp. 45– 48. doi: 10.1109/M2GARSS47143.2020.9105265.

[39] S.E. Brezini, Y. Deville, M.S. Karoui, F.Z. Benhalouche, and A. Ouamri, “A Penalization-Based NMF Approach for Hyperspectral Unmixing Addressing Spectral Variability with an Additively-Tuned Mixing Model,” in International Geoscience and Remote Sensing Symposium, IGARSS 2021, Brussels, 2021, pp. 3841–3844.

[40] A. Cichocki, R. Zdunek, A. H. Phan, and S. I. Amari, Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-Way Data Analysis and Blind Source Separation, John Wiley. Chichester, UK, 2009.

[41] X. R. Feng, H. C. Li, J. Li, Q. Du, A. Plaza, and W. J. Emery, “Hyperspectral unmixing using sparsity-constrained deep nonnegative matrix factorization with total variation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 10, pp. 6245–6257, 2018, doi: 10.1109/TGRS.2018.2834567.

[42] C. J. Lin, “Projected gradient methods for nonnegative matrix factorization,” Neural Comput, vol. 19, no. 10, pp. 2756–2779, 2007, doi: 10.1162/neco.2007.19.10.2756.

[43] D. Seung and L. Lee, “Algorithms for non-negative matrix factorization,” Adv Neural Inf Process Syst, vol. 13, no. 556–562, p. 3, 2001.

[44] K. B. Petersen and M. S. Pedersen, The matrix Cookbook. Technical Univ. Denmark, 2008. [Online]. Available: http://matrixcook-book.com/

[45] J. M. P. Nascimento and J. M. Bioucas-Dias, “Vertex component analysis: A fast algorithm to unmix hyperspectral data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898– 910, 2005, doi: 10.1109/TGRS.2005.844293.

[46] L. Qi, J. Li, Y. Wang, and X. Gao, “Region-Based Multiview Sparse Hyperspectral Unmixing Incorporating Spectral Library,” IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 7, pp. 1140–1144, 2019, doi: 10.1109/LGRS.2019.2891559.

[47] L. Drumetz, J. Chanussot, C. Jutten, W. K. Ma, and A. Iwasaki, “Spectral variability aware blind hyperspectral image unmixing based on convex geometry,” IEEE Transactions on Image Processing, vol. 29, pp. 4568– 4582, 2020, doi: 10.1109/TIP.2020.2974062.

[48] D. C. Heinz and C. I. Chang, “Fully Constrained Least Squares Linear Spectral Mixture Analysis Method for Material Quantification in Hyperspectral Imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 39, no. 3, pp. 529–545, 2001, doi: 10.1109/36.911111.

[49] K. R. M. Adeline, A. Le Bris, F. Coubard et al., “Description de la campagne aéroportée UMBRA: étude de l’impact anthropique sur les écosystèmes urbains et naturels avec des images THR multispectrales et hyperspectrales,” Revue française de photogrammétrie et de télédétection, no. 202, pp. 79–92, 2013.

[50] C. Miesch, L. Poutier, V. Achard, X. Briottet, X. Lenot, and Y. Boucher, “Direct and inverse radiative transfer solutions for visible and nearinfrared hyperspectral imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 7, pp. 1552–1562, 2005, doi: 10.1109/TGRS.2005.847793.

[51] http://www.ehu.es/ccwintco/index.php/HyperspectraI Imagery Synthesis tools for MATLAB.

[52] C. Debes, A. Merentitis, R. Heremans, J. Hahn, N. Frangiadakis, T. van Kasteren, W. Liao, R. Bellens, A. Pizurica, S. Gautama, W. Philips, S. Prasad, Q. Du, and F. Pacifici, “Hyperspectral and LiDAR Data Fusion: Outcome of the 2013 GRSS Data Fusion Contest,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 7(6), pp. 2405-2418, 2014.

[53] F. Zhu, Y. Wang, B. Fan, S. Xiang, G. Meng, and C. Pan, “Spectral unmixing via data-guided sparsity,” IEEE Transactions on Image Processing (TIP), vol. 23, no. 12, pp. 5412–5427, Dec 2014.