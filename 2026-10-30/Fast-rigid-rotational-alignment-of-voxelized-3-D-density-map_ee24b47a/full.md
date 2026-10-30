# Fast rigid rotational alignment of voxelized 3-D density maps in sliced Wasserstein distance

Oscar Mickelin<sup>∗1</sup>, Yunpeng Shi<sup>†2</sup>, Amit Singer<sup>‡3,4</sup>, and Eric J. Verbeke<sup>§5,6</sup>

<sup>1</sup> Yau Mathematical Sciences Center, Tsinghua University, Beijing 100084, China

<sup>2</sup>Department of Mathematics, University of California at Davis, Davis CA, USA

<sup>3</sup>Department of Mathematics, Princeton University, Princeton, NJ, USA

<sup>4</sup>Program in Applied and Computational Mathematics, Princeton University, Princeton, NJ, USA

<sup>5</sup>Center for Statistics and Machine Learning, Princeton University, Princeton, NJ, USA

<sup>6</sup>Premium Research Institute for Human Metaverse Medicine, The University of Osaka, Japan

## Abstract

We develop a fast method for the rigid alignment of voxelized three-dimensional density functions over SO(3) using sliced optimal transport. Our approach eficiently transforms each density map into a tensor of one-dimensional quantile functions, taken from central slices over the unit sphere, such that the Euclidean distance of the transformed volumes approximates the sliced 2-Wasserstein distance. Our formulation is therefore compatible with existing, fast, voxel-based Euclidean alignment algorithms. We provide theoretical analysis and numerical experiments demonstrating that the sliced 2-Wasserstein distance is a robust metric for comparing three-dimensional shapes and for computing their rigid alignment.

## 1 Introduction

We consider the problem of three-dimensional (3-D) rigid alignment between density functions f and g which are defined on $N \times N \times N$ Cartesian voxel grids, and are supported on the closed unit ball $\mathbb { B } = \{ x \in \mathbb { R } ^ { 3 }$ $\| x \| _ { 2 } \leq 1 \}$ . Additionally, the functions are assumed to be non-negative with $f \geq 0 , g \geq 0$ and normalized to $\begin{array} { r } { \int _ { \mathbb { R } } f ( x ) d x = \int _ { \mathbb { R } } g ( x ) = 1 } \end{array}$ , such that they may be thought of as probability density functions and are therefore compatible with optimal transport frameworks described later. Specifically, we seek to find:

$$
\operatorname * { a r g m i n } _ { R \in { \sf S O } ( 3 ) } d ( f , R g ) ,\tag{1}
$$

where $R \in { \mathsf { S O } } ( 3 )$ is a rotation and $d ( \cdot , \cdot ) \in [ 0 , \infty )$ denotes a generic distance function. In particular, we are motivated by the case where $f \neq R g$ for any $R \in { \mathsf { S O } } ( 3 )$ , which we refer to as heterogeneous functions (or heterogeneous 3-D volumes). This type of problem arises in many applications, most prominently in 3-D modeling [1] and biological imaging methods such as electron cryo-microscopy (cryo-EM) [2]. Typically, eq. (1) is solved using the Euclidean distance or standard correlation as the metric for comparison. However, it is known that the landscape with respect to these metrics can be highly irregular, especially for heterogeneous volumes [3].

We therefore focus on using metrics from optimal transport, as they are known to be more robust to changes in the geometry of the objects being compared [4, 5]. However, these metrics, in particular the Wasserstein distance (or Kantorovich-Rubenstein distance), come with a high computational cost. The naive computational complexity for the 2-Wasserstein distance between two $N \times N \times N$ functions using linear programming is $\mathcal { O } ( N ^ { 9 } )$ operations. For rotational alignment, where the distance must be computed over a grid of rotations (typically $N ^ { 3 }$ rotations, to scale with the size of the input functions), the computationa complexity increases to $\mathcal { O } ( N ^ { 1 2 } )$ operations. Although there exist fast algorithms for approximating the

Wasserstein distance [6], especially on geometric domains [7], they are still too slow for large-scale and high-dimensional 3-D alignments. This high computational cost drastically limits the application of the Wasserstein distance to rigid alignment tasks. Meanwhile, there exist eficient algorithms for computing the correlation over a grid of rotations in SO(3) with complexity O(N<sup>4</sup>) operations [8, 9]. Therefore, this work attempts to unify the two frameworks of fast rotational correlations in SO(3) with optimal transport metrics. Our contributions are twofold:

1. We detail an optimal transport-based alignment algorithm that performs a full grid optimization over SO(3) in $\mathcal { O } ( N ^ { 4 } )$ operations. Interestingly, we show how selecting d in eq. (1) as the sliced 2-Wasserstein distance leads to a problem of aligning two helper quantities in the Euclidean distance. Many alignment algorithms exist for the Euclidean distance and our approach therefore results in a modular and eficient framework where improved algorithms for Euclidean alignment can be leveraged to the setting of alignment in sliced 2-Wasserstein metrics.

As an additional advantage, our alignment algorithm works directly on voxel-representations of the data, unlike several approaches that require an initial conversion to point-cloud representations. Lastly, the algorithm is deterministic and does not require suficiently accurate initializations or suficiently tuned hyperparameters, whereas many related algorithms (discussed later) do.

2. We provide two theorems about the stability of three-dimensional sliced Wasserstein metrics between a structure and a deformation of itself. These results can be used to inform alignment of two heterogeneous functions, since the distance of the aligned structures results in a lower bound on the degree of the heterogeneity through the stability theorems.

This work builds on [10] which studied sliced Wasserstein-based alignment algorithms for two-dimensional functions. Compared to the 2-D case, the 3-D case poses additional challenges since the higher dimension adds a degree of freedom to the parametrization of the rotations. This means that we cannot simply apply a variation of the 2-D algorithm and care must be taken both theoretically and implementation-wise to obtain fast and accurate algorithms. We therefore phrase our results in terms of a more general and modular framework, as discussed in bullet point 1 above. We show numerical results that highlight the advantage of our algorithm over traditional correlation-based Euclidean alignment methods and a selection of other optimal transport-based alignment algorithms. Our code is available at [https://github.com/ EricVerbeke/fast\_volume\_alignment\_ot].

## 2 Related works

Three-dimensional rigid registration is a classical problem in computer vision. Existing methods can broadly be categorized according to the representation of the data and the optimization strategy. For example, two common approaches are point cloud registration algorithms [11–13], and algorithms that use 3-D feature descriptors for alignment [14]. Our approach focuses on using voxelized data directly. Voxel-based alignment methods are typically computed by evaluating rotational correlations using a spherical harmonic basis [8, 9]. While this approach is successful and widely used, it does not specifically handle deformations in the data. We note that there are methods that address finding correspondences between heterogeneous shapes [15], but they typically assume some kind of global registration of the input data. Therefore, the main contribution of our work is to combine the fast voxel-based alignment methods with optimal transport metrics to provide a more robust rigid alignment framework for heterogeneous data.

Specifically, we utilize a sliced Wasserstein approach in this work. The sliced Wasserstein distance was originally introduced for 2-D applications in [16] and has since been extended more generally in [17, 18]. It has been used for many applications, particularly for 2-D and 3-D classification tasks [19, 20]. We also note that there are already variants available for computing the sliced Wasserstein distance on the sphere. However, there are key diferences between our work and existing methods. In our work, we define slices as regularly spaced lines passing though the origin with data projected orthogonally onto these lines. In contrast, other related works take slices to be 2-D planes passing through the unit sphere [21], or consider the data to be projected onto great circles that are defined on the sphere [22] or into a Euclidean space using stereographic projections [23]. Another related approach uses lines through the origin as slices, together with a Monte Carlo approach to compute the sliced Wasserstein distance. However, for our purpose of alignment, there is no advantage to random sampling compared to equispaced projections [10].

A major application of 3-D alignment algorithms comes from the field of structural biology. In particular, cryo-EM relies heavily on alignment algorithms. Briefly, cryo-EM is a method used to obtain high-resolution 3-D reconstructions of biological molecules from large, high-dimensional and noisy 2-D tomographic projection images [2, 24]. Furthermore, cryo-EM experiments can contain extremely heterogeneous data; it can be both compositionally heterogeneous [25, 26] (the data contains diferent underlying 3-D structures), and conformationally heterogeneous [27, 28] (the data contains the same 3-D structure but with possible deformations). Both forms of heterogeneity challenge conventional alignment algorithms. To address these specific challenges, many novel algorithms have been developed [29–31]. Recently, several optimal transport based frameworks have also been introduced [3, 32, 33]. Importantly, several key diferences exist between our optimal transport based alignment algorithm and others; 1) we operate on the voxels directly, 2) we compute a full grid of rotations in optimal complexity and 3) we require no hyperparameters.

## 3 Preliminaries

This section details the definitions, conventions and observations used in the remainder of the paper, to make the exposition self-contained.

## 3.1 Rotation conventions

For any rotation $R \in { \mathsf { S O } } ( 3 )$ and function $f : \mathbb { R } ^ { 3 }  \mathbb { R }$ , we denote the action of R on f by $R f : \mathbb { R } ^ { 3 } \to \mathbb { R }$ 2 defined by

$$
( R f ) ( x , y , z ) = f ( R ^ { \top } ( x , y , z ) ) .
$$

Let $\theta \in [ 0 , \pi ] , \varphi \in [ 0 , 2 \pi ] , \omega \in [ 0 , 2 \pi ]$ be three angles. We will use the ZYZ Euler angle representation of rotations, and therefore write any rotation in ${ \mathsf { S O } } ( 3 )$ as

$$
R _ { \omega , \theta , \varphi } = R _ { \omega } R _ { \theta , \varphi } ,
$$

where $R _ { \omega }$ is a rotation around the z-axis by the angle $\omega \in ( 0 , 2 \pi ]$ , and $R _ { \theta , \varphi }$ a rigid rotation mapping the vector on the unit sphere with angular coordinates $( \theta , \varphi ) \in [ 0 , \theta ] \times ( 0 , 2 \pi ]$ to the positive z-axis. We can write $R _ { \omega } = Z ( \omega )$ and $R _ { \theta , \varphi } : = Y ( - \theta ) Z ( - \varphi )$ , where the two elemental rotations Y and Z are defined by

$$
Y ( \theta ) : = \left[ { \begin{array} { c c c } { \cos \theta } & { 0 } & { \sin \theta } \\ { 0 } & { 1 } & { 0 } \\ { - \sin \theta } & { 0 } & { \cos \theta } \end{array} } \right] , \quad Z ( \varphi ) : = \left[ { \begin{array} { c c c } { \cos \varphi } & { - \sin \varphi } & { 0 } \\ { \sin \varphi } & { \cos \varphi } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} } \right] .
$$

We will in the following deal with rotations of this form, followed by projections onto the z-axis, which are invariant to the action of the $R _ { \omega }$ (see Observation 2 below for a precise statement).

We will, by a slight abuse of notation, write expressions like $R ^ { \top } ( \theta , \varphi )$ to be the point on the unit sphere obtained by rotating the point with spherical angles $\theta , \varphi$

## 3.2 The Radon Transform

This section defines the Radon transform and recalls a number of its properties which will be useful in the remainder of the article. In three-dimensional space, the Radon transform is parametrized by the rotation angles $( \theta , \varphi )$ . It is computed by projecting the plane orthogonal to the viewing direction onto the viewing direction axis. In more detail, for a given function $f : \mathbb { R } ^ { 3 } \to \mathbb { C }$ , we denote the result of applying the Radon transform to f by $P _ { \theta , \varphi } f : \mathbb { R }  \mathbb { C } .$ , defined by

$$
\left( P _ { \theta , \varphi } f \right) ( z ) : = \iint _ { \mathbb { R } ^ { 2 } } ( R _ { \theta , \varphi } f ) ( x , y , z ) d x d y = \iint _ { \mathbb { R } ^ { 2 } } f ( R _ { \theta , \varphi } ^ { \top } ( x , y , z ) ) d x d y .\tag{2}
$$

By allowing the spherical angles $\theta , \varphi$ to vary, we will (when convenient) alternatively view $P _ { \theta , \varphi } f ( z )$ as a function defined on $\mathbb { R } ^ { 3 }$ written in terms of the spherical coordinates $( \theta , \varphi , z ) \in [ 0 , \theta ] \times ( 0 , 2 \pi ] \times \mathbb { R }$ . For convenience, we allow $z < 0$ in what follows and note that the point with spherical coordinates $( \theta , \varphi , - | z | )$ can also be written with the spherical coordinates $( \pi - \theta , \pi + \varphi , | z | )$

Observation 1. By the Fourier slice theorem, the projections can be evaluated using Fourier transforms

$$
P _ { \theta , \varphi } f ( z ) = \mathcal { F } _ { 1 } ^ { - 1 } \left( ( S _ { \theta , \varphi } \mathcal { F } _ { 3 } f ) ( \xi ) \right) = \int _ { \mathbb { R } } ( \mathcal { F } _ { 3 } f ) ( R _ { \theta , \varphi } ^ { \top } ( 0 , 0 , \xi ) ^ { \top } ) e ^ { 2 \pi i \xi z } d \xi ,\tag{3}
$$

where $\mathcal { F } _ { d }$ denotes the d-dimensional Fourier transform with the convention

$$
\mathcal { F } _ { d } ( f ) ( \boldsymbol { \xi } ) : = \int _ { \mathbb { R } ^ { d } } f ( \boldsymbol { x } ) e ^ { - 2 \pi i \boldsymbol { \xi } \cdot \boldsymbol { x } } d \boldsymbol { x } , \quad \boldsymbol { \xi } \in \mathbb { R } ^ { d } , \qquad \mathcal { F } _ { d } ^ { - 1 } ( f ) ( \boldsymbol { x } ) : = \int _ { \mathbb { R } ^ { d } } f ( \boldsymbol { \xi } ) e ^ { 2 \pi i \boldsymbol { \xi } \cdot \boldsymbol { x } } d \boldsymbol { \xi } , \quad \boldsymbol { x } \in \mathbb { R } ^ { d } ,
$$

and for any $g : \mathbb { R } ^ { 3 }  \mathbb { C }$ , we define the function $S _ { \theta , \varphi } ( g ) : \mathbb { R } \to \mathbb { C }$ by

$$
( S _ { \theta , \varphi } g ) ( k ) = g ( k \sin \theta \cos \varphi , k \sin \theta \sin \varphi , k \cos \theta )
$$

as the one-dimensional function obtained by restriction to given spherical angles $( \theta , \varphi )$

The next observation explains why our definition of $P _ { \theta , \varphi }$ does not include the additional angle ω.

Observation 2. For any $\omega \in ( 0 , 2 \pi ]$ and $f : \mathbb { R } ^ { 3 }  \mathbb { R }$

$$
\iint _ { \mathbb { R } ^ { 2 } } ( R _ { \omega } R _ { \theta , \varphi } f ) ( x , y , z ) d x d y = \iint _ { \mathbb { R } ^ { 2 } } ( R _ { \theta , \varphi } f ) ( x , y , z ) d x d y .
$$

Proof. We have

$$
\int \int _ { \mathbb { R } ^ { 2 } } ( R _ { \omega } R _ { \theta , \varphi } f ) ( x , y , z ) d x d y = \int \int _ { \mathbb { R } ^ { 2 } } ( R _ { \theta , \varphi } f ) ( R _ { \omega } ^ { \top } ( x , y , z ) ) d x d y = \int \int _ { \mathbb { R } ^ { 2 } } ( R _ { \theta , \varphi } f ) ( x ^ { \prime } , y ^ { \prime } , z ) d x ^ { \prime } d y ^ { \prime } ,
$$

after performing the change of variables $( x ^ { \prime } , y ^ { \prime } , z ) = R _ { \omega } ^ { \top } ( x , y , z )$ which has a Jacobian with determinant equal to 1. □

Any rotation $R \in S O ( 3 )$ defines an action on $P _ { \theta , \varphi } f$ by rotating the spherical angles $( \theta , \varphi )$ , i.e., by

$$
( R P _ { \theta , \varphi } f ) ( z ) = ( P _ { R ^ { T } ( \theta , \varphi ) } f ) ( z ) .
$$

Explicitly, we can describe this action with the following observation.

Observation 3. For any rotation $R \in S O ( 3 ) , ( R P _ { \theta , \varphi } f ) ( z ) = P _ { \theta , \varphi } ( R f ) ( z )$

Proof. For any $( \gamma , \alpha , \beta ) \in ( 0 , 2 \pi ] \times [ 0 , \pi ] \times ( 0 , 2 \pi ]$ , we have $R _ { \gamma , \alpha , \beta } P _ { \theta , \varphi } f ( z ) = P _ { s , t } f$ , where $( s , t , z )$ are the spherical coordinates of the image under $R _ { \gamma , \alpha , \beta } ^ { \top }$ of the point with spherical coordinates $( \theta , \varphi , z )$ , i.e., the spherical coordinates of $R _ { \gamma , \alpha , \beta } ^ { \top } R _ { \theta , \varphi } ^ { \top } e _ { z }$ . On the other hand, we have

$$
( P _ { \theta , \varphi } ( R _ { \gamma , \alpha , \beta } f ) ) \left( z \right) = \iint _ { \mathbb { R } ^ { 2 } } ( R _ { \theta , \varphi } R _ { \gamma , \alpha , \beta } f ) ( x , y , z ) d x d y\tag{4}
$$

By decomposing the rotation $R _ { \theta , \varphi } R _ { \gamma , \alpha , \beta }$ into Euler angles in the ZYZ convention, we can write

$$
R _ { \theta , \varphi } R _ { \gamma , \alpha , \beta } = R _ { u , v , w } ,
$$

for some Euler angles $u , v , w$ . By Observation 2, this implies that

$$
\left( P _ { \theta , \varphi } ( R _ { \gamma , \alpha , \beta } f ) \right) ( z ) = ( P _ { v , w } f ) ( z ) ,\tag{5}
$$

and we now claim that $( v , w ) = ( s , t )$ . To see this, note that the definition of $R _ { v , w }$ in Section 3.1 implies that $R _ { v , w } ( v , w ) = e _ { z }$ , so

$$
\begin{array} { r } { ( \boldsymbol { v } , \boldsymbol { w } ) = R _ { v , w } ^ { \top } e _ { z } = R _ { v , w } ^ { \top } R _ { u } ^ { \top } e _ { z } = R _ { u , v , w } ^ { \top } e _ { z } = ( R _ { \theta , \varphi } R _ { \gamma , \alpha , \beta } ) ^ { \top } \boldsymbol { e } _ { z } = R _ { \gamma , \alpha , \beta } ^ { \top } R _ { \theta , \varphi } ^ { \top } e _ { z } = ( s , t ) . } \end{array}
$$

## 4 Algorithm

This section presents the main result of the paper, i.e., an algorithm for aligning volumes using the sliced Wasserstein distance. For clarity, we separate the exposition into the following two subsections. The first section details the computations involved in computing and aligning in the sliced Wasserstein distance for continuous functions. The second section introduces the approximations involved when performing these computations on discrete data.

## 4.1 Continuous algorithm

The algorithm for alignment of continuous functions can be divided into three steps, presented below.

## 4.1.1 Step 1: computing Radon transforms

Let $f , g : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { + }$ be two probability measures which are normalized such that $\begin{array} { r l } { \int _ { \mathbb { R } ^ { 3 } } f ( x , y , z ) d x d y d z } & { { } = } \end{array}$ $\begin{array} { r } { \int _ { \mathbb { R } ^ { 3 } } g ( x , y , z ) d x d y d z = 1 } \end{array}$ . We compute the Radon transforms of $f$ and g, parametrized by a rotation axis $( \theta , \varphi )$ . For each axis $( \theta , \varphi ) \in \times [ 0 , \pi ] \times [ 0 , 2 \pi ]$ , we denote these one-dimensional projections by $\mu _ { \theta , \varphi } , \nu _ { \theta , \varphi } :$ $\mathbb { R } \to \mathbb { R } ^ { + }$ . They are defined by

$$
\mu _ { \theta , \varphi } ( x ) : = \left( P _ { \theta , \varphi } f \right) ( x ) , \qquad { \mathrm { a n d } } \qquad \nu _ { \theta , \varphi } ( x ) : = \left( P _ { \theta , \varphi } g \right) ( x ) .
$$

## 4.1.2 Step 2: computing Wasserstein distances

The continuous sliced 2-Wasserstein distance between f and $g$ is the mean squared 1-D 2-Wasserstein distance over all projection angles

$$
d _ { S W _ { 2 } } ^ { 2 } ( f , g ) = \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } d _ { W _ { 2 } } ^ { 2 } ( \mu _ { \theta , \varphi } , \nu _ { \theta , \varphi } ) \sin \theta d \varphi d \theta .\tag{6}
$$

Here, the squared 1-D 2-Wasserstein distance between two projections $\mu$ and ν is defined by

$$
d _ { W _ { 2 } } ^ { 2 } ( \mu , \nu ) = \int _ { 0 } ^ { 1 } \left| \left( \mathcal { V } \mu \right) ^ { - 1 } \left( t \right) - \left( \mathcal { V } \nu \right) ^ { - 1 } \left( t \right) \right| ^ { 2 } d t ,\tag{7}
$$

where $\begin{array} { r } { ( \mathcal { V } \mu ) ( s ) = \int _ { - \infty } ^ { s } \mu ( x ) d x } \end{array}$ denotes the cumulative distribution function and $( { \mathcal V } \mu ) ^ { - 1 } ( t ) = \operatorname* { i n f } \{ s : ( { \mathcal V } \mu ) ( s ) >$ $t \}$ is the associated quantile function (or functional inverse of the cumulative distribution function).

The integration performed in the Radon transform typically smooths out diferences between functions. One common approach in tomography to counteract this smoothing is to include a ramp-filter in Fourier space [34]. We therefore additionally define a ramp-filtered version of the 3-D sliced Wasserstein distance (referred to as ramp-filtered sliced Wasserstein distance), as introduced in [10]:

$$
\mu _ { \theta , \varphi } ^ { s } ( x ) = h \ast ( P _ { \theta , \varphi } ) f ( x ) = \mathcal { F } _ { 1 } ^ { - 1 } ( | \xi | ^ { 2 } ( \mathcal { F } _ { 3 } f ) ( R _ { \theta , \varphi } ^ { \top } \left( 0 , 0 , \xi \right) ^ { \top } ) ,\tag{8}
$$

where h is the ramp filter in the spatial domain, whose Fourier transform is $\mathcal { F } _ { 1 } ( h ) ( \xi ) = | \xi | ^ { 2 }$ . As previously described in [10], applying the ramp filter does not enforce positivity of the line projections, which is required for probability measures. We therefore split each ramp-filtered projection into its positive and negative parts

$$
\mu _ { \theta , \varphi } ^ { s } = \mu _ { \theta , \varphi } ^ { + } + \mu _ { \theta , \varphi } ^ { - } , \qquad \mu _ { \theta , \varphi } ^ { + } = \operatorname* { m a x } ( \mu _ { \theta , \varphi } , 0 ) , \quad \mu _ { \theta , \varphi } ^ { - } = \operatorname* { m i n } ( \mu _ { \theta , \varphi } , 0 )
$$

As noted in [10], although the line projections are all mean-zero, their norms can be diferent. Specifically

$$
\int \mu _ { \theta , \varphi } ^ { + } ( x ) + \mu _ { \theta , \varphi } ^ { - } ( x ) d x = \int \nu _ { \theta , \varphi } ^ { + } ( x ) + \nu _ { \theta , \varphi } ^ { - } ( x ) d x = 0 ,
$$

but it is not necessarily true that the norms of the signed projections are equal. Therefore, to follow the conditions required for the 1-D Wasserstein distance, the signed projections are additionally normalized to 1, as in:

$$
\widetilde { \mu } _ { \theta , \varphi } ^ { + } : = \frac { \mu _ { \theta , \varphi } ^ { + } } { \lVert \mu _ { \theta , \varphi } ^ { + } \rVert _ { 1 } } , \qquad \widetilde { \mu } _ { \theta , \varphi } ^ { - } : = \frac { \mu _ { \theta , \varphi } ^ { - } } { \lVert \mu _ { \theta , \varphi } ^ { - } \rVert _ { 1 } } .
$$

We then consider the 3-D ramp-filtered sliced 2-Wasserstein distance to be

$$
d _ { R F S W _ { 2 } } ^ { 2 } ( f , g ) = \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } d _ { W _ { 2 } } ^ { 2 } ( \tilde { \mu } _ { \theta , \varphi } ^ { + } , \tilde { \nu } _ { \theta , \varphi } ^ { + } ) \sin \theta d \varphi d \theta + \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } d _ { W _ { 2 } } ^ { 2 } ( \tilde { \mu } _ { \theta , \varphi } ^ { - } , \tilde { \nu } _ { \theta , \varphi } ^ { - } ) \sin \theta d \varphi d \theta .\tag{9}
$$

Even with the additional normalization of the filtered projections, eq. (9) provides a good measurement of the diference in the sense that $d _ { R F S W _ { 2 } } ^ { 2 } ( f , g ) = 0$ if and only if $f = g$ almost everywhere.

## 4.1.3 Step 3: computing alignment between inverse cumulative distribution functions

In the last step, the distance is minimized over rotations $R \in { \mathsf { S O } } ( 3 )$ . In more detail, we choose the alignment angles $( \alpha _ { \ast } , \beta _ { \ast } , \gamma _ { \ast } ) \in [ 0 , \pi ] \times [ 0 , 2 \pi ] ^ { 2 }$ by minimizing the sliced Wasserstein distance between f and $R _ { \theta , \varphi } g$ , i.e., by

$$
( \alpha _ { * } , \beta _ { * } , \gamma _ { * } ) = \underset { ( \alpha , \beta , \gamma ) \in [ 0 , \pi ] \times [ 0 , 2 \pi ] ^ { 2 } } { \arg \operatorname* { m i n } } d _ { S W _ { 2 } } ( f , R _ { \alpha , \beta , \gamma } g ) .\tag{10}
$$

Our alignment algorithm hinges on the following result.

Proposition 1. For any $( \alpha , \beta , \gamma ) \in [ 0 , \pi ] \times [ 0 , 2 \pi ] ^ { 2 }$ and functions $f , g : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { + }$ with $\begin{array} { r } { \int _ { \mathbb { R } ^ { 3 } } f d x = 1 = \int _ { \mathbb { R } ^ { 3 } } g d x _ { \colon } } \end{array}$ we have

$$
d _ { S W _ { 2 } } ( f , R _ { \alpha , \beta , \gamma } g ) = d _ { E } ( h _ { f } , R _ { \alpha , \beta , \gamma } h _ { g } ) ,\tag{11}
$$

where $d _ { E } ( \cdot , \cdot )$ denotes the Euclidean distance and

$$
\begin{array} { r } { h _ { f } ( t , \theta , \varphi ) = \frac { 1 } { t } \left( \mathcal { V } P _ { \theta , \varphi } f \right) ^ { - 1 } ( t ) , \qquad h _ { g } ( t , \theta , \varphi ) = \frac { 1 } { t } \left( \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } ( t ) . } \end{array}
$$

Similarly for the ramp-filtered sliced 2-Wasserstein distance

$$
d _ { R F S W _ { 2 } } ( f , R _ { \alpha , \beta , \gamma } g ) ^ { 2 } = d _ { E } ( h _ { f } ^ { + } , R _ { \alpha , \beta , \gamma } h _ { g } ^ { + } ) ^ { 2 } + d _ { E } ( h _ { f } ^ { - } , R _ { \alpha , \beta , \gamma } h _ { g } ^ { - } ) ^ { 2 } ,\tag{12}
$$

where

$$
\begin{array} { r } { h _ { f } ^ { \pm } ( t , \theta , \varphi ) = \frac { 1 } { t } \left( \mathcal { V } \tilde { \mu } _ { \theta , \varphi } ^ { \pm } \right) ^ { - 1 } ( t ) , \qquad h _ { g } ^ { \pm } ( t , \theta , \varphi ) = \frac { 1 } { t } \left( \mathcal { V } \tilde { \nu } _ { \theta , \varphi } ^ { \pm } \right) ^ { - 1 } ( t ) . } \end{array}
$$

Proof. The proof relies on commuting a number of the introduced operators. We first note that

$$
( \mathcal V { R } _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } g ) \left( t \right) = \int _ { - \infty } ^ { t } ( R _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } g ) ( z ) d z = R _ { \alpha , \beta , \gamma } \int _ { - \infty } ^ { t } ( P _ { \theta , \varphi } g ) ( z ) d z = ( R _ { \alpha , \beta , \gamma } \mathcal V _ { \theta , \varphi } g ) \left( t \right) .\tag{13}
$$

It then also follows that

$$
\left( \mathcal { V } R _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } g \right) ^ { - 1 } ( t ) = \left( R _ { \alpha , \beta , \gamma } \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } ( t ) = R _ { \alpha , \beta , \gamma } \left( \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } ( t ) .\tag{14}
$$

We conclude by making the following chain of observations

$$
d _ { S W _ { 2 } } ( f , R _ { \alpha , \beta , \gamma } g ) ^ { 2 } = \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } d _ { W _ { 2 } } ^ { 2 } ( P _ { \theta , \varphi } f , P _ { \theta , \varphi } R _ { \alpha , \beta , \gamma } g ) \sin { \theta } d \varphi d \theta\tag{by eq. (6)}
$$

$$
= \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } d _ { W _ { 2 } } ^ { 2 } ( P _ { \theta , \varphi } f , R _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } g ) \sin \theta d \varphi d \theta\tag{by Obs. 3}
$$

$$
= \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { 1 } \left| \left( \mathcal { V } P _ { \theta , \varphi } f \right) ^ { - 1 } \left( t \right) - \left( \mathcal { V } R _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } g \right) ^ { - 1 } \left( t \right) \right| ^ { 2 } d t \sin \theta d \varphi d \theta\tag{by eq. (7)}
$$

$$
= \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { 1 } \left| ( \mathcal { V } P _ { \theta , \varphi } f ) ^ { - 1 } \left( t \right) - \left( R _ { \alpha , \beta , \gamma } \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } \left( t \right) \right| ^ { 2 } d t \sin \theta d \varphi d \theta\tag{by eq. (14)}
$$

$$
= \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { 1 } \left| \left( \mathcal { V } P _ { \theta , \varphi } f \right) ^ { - 1 } \left( t \right) - R _ { \alpha , \beta , \gamma } \left( \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } \left( t \right) \right| ^ { 2 } d t \sin \theta d \varphi d \theta\tag{by eq. (14)}
$$

$$
= \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { 1 } \left| \frac { 1 } { t } \left( \mathcal { V } P _ { \theta , \varphi } f \right) ^ { - 1 } \left( t \right) - \frac { 1 } { t } R _ { \alpha , \beta , \gamma } \left( \mathcal { V } P _ { \theta , \varphi } g \right) ^ { - 1 } \left( t \right) \right| ^ { 2 } t ^ { 2 } \sin \theta d t d \varphi d \theta
$$

$$
= d _ { E } ( h _ { f } , R _ { \alpha , \beta , \gamma } h _ { g } ) ^ { 2 } .
$$

For the ramp-filtered sliced Wasserstein distance, we have

$$
\mathcal { V } \left( h * ( P _ { \theta , \varphi } R _ { \alpha , \beta , \gamma } f ) \right) ^ { \pm } = \mathcal { V } \left( h * ( R _ { \alpha , \beta , \gamma } P _ { \theta , \varphi } f ) \right) ^ { \pm } = \mathcal { V } R _ { \alpha , \beta , \gamma } \left( h * ( P _ { \theta , \varphi } f ) \right) ^ { \pm } = R _ { \alpha , \beta , \gamma } \mathcal { V } \left( h * ( P _ { \theta , \varphi } f ) \right) ^ { \pm } ,
$$

where the first equality is Observation 3, and the second comes from the fact that h is a radial function, so $R _ { \alpha , \beta , \gamma }$ commutes with this radial convolution. The action of $R _ { \alpha , \beta , \gamma }$ also commutes with extracting positive and negative parts, and the third equality is eq. (14). An analogous argument to the above chain of equalities then concludes the proof. □

In other words, we can compute the rotation minimizing the distance between f and g in the sliced 2-Wasserstein distance, by minimizing the Euclidean distance of the two helper functions $h _ { f }$ and $h _ { g }$ . This can be done using any existing 3-D Euclidean alignment algorithm. Similarly, for the ramp-filtered sliced 2-Wasserstein distance, we minimize the Euclidean distance between the two vector-valued helper functions

$$
\left[ h _ { f } ^ { + } \right] \quad \mathrm { ~ a n d ~ } \quad \left[ h _ { g } ^ { + } \right] .\tag{15}
$$

## 4.2 Discrete algorithm

For discretized volumes, the steps in Section 4.1 need to be performed on sampled data, given as voxel representations on discrete grids. This section explicitly writes out the discrete approximations of Section 4.1 and shows how the steps can be performed with low computational cost. A visualization of the transforms used to compute the discretized sliced Wasserstein distance can be found in Figure 1.

![](images/253342831a3246bbd81499ffb8ae7b23fcabd42328140f67548c62007c1ce664.jpg)  
Figure 1: Visualization of the transforms used to compute the 3-D sliced Wasserstein distance.

## 4.2.1 Step 1: transforms

We define a product grid of rotational angles in 3-D. Let the integers $n _ { \varphi }$ and n<sub>θ</sub> denote the number of grid-points in the two angular directions respectively and define a grid of rotational directions by

$$
\varphi _ { j } = { \frac { 2 \pi j } { n _ { \varphi } } } , \quad j = 1 , \ldots , n _ { \varphi } , \qquad { \mathrm { a n d } } \qquad \theta _ { k } = { \frac { \pi k } { n _ { \theta } } } , \quad k = 1 , \ldots , n _ { \theta } ,\tag{16}
$$

and a Cartesian grid (with $n _ { x } , n _ { y } , n _ { z }$ grid points in the $x , y , z$ directions, respectively), defined by

$$
x _ { p } = { \frac { 2 ( p - 1 ) } { n _ { x } } } - 1 , \quad y _ { q } = { \frac { 2 ( q - 1 ) } { n _ { y } } } - 1 , \quad z _ { i } = { \frac { 2 ( i - 1 ) } { n _ { z } } } - 1 , \quad p = 1 , \dots , n _ { x } , \quad q = 1 , \dots , n _ { y } , \quad i = 1 , \dots , n _ { z } .\tag{17}
$$

Let $f , g : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { + }$ be two probability measures which are normalized such that $\begin{array} { r l } { \int _ { \mathbb { R } ^ { 3 } } f ( x , y , z ) d x d y d z = } \end{array}$ $\begin{array} { r } { \int _ { \mathbb { R } ^ { 3 } } g ( x , y , z ) d x d y d z \ = \ 1 } \end{array}$ . Given samples of f and g on the grid defined by $x _ { p } , y _ { q } , z _ { i } ,$ we compute two tensors containing approximate projections of f and g. Denoting these by $P _ { f } , P _ { g } \in \dot { \mathbb { R } ^ { N \times n _ { \varphi } \times n _ { \theta } } }$ , their entries are defined as discrete approximations to the projection tensor at $z _ { i } , \theta _ { j } , \varphi _ { k }$ . The approximated projection operator is given by performing a non-uniform fast Fourier transform (NUFFT) [35, 36] in Observation 1, followed by a fast Fourier transform. In detail, the projection operator is given by the following sequence of definitions:

1. For $\ell = 1 , \ldots , n _ { z } , j = 1 , \ldots , n _ { \varphi } , k = 1 , \ldots , n _ { \theta }$ , define the NUFFT target points

$$
\xi _ { \ell j k } = \frac { n _ { z } } { 2 } ( z _ { \ell } \sin \theta _ { j } \cos \varphi _ { k } , z _ { \ell } \sin \theta _ { j } \sin \varphi _ { k } , z _ { \ell } \cos \theta _ { j } ) ,
$$

2. Compute the discrete Fourier transform using the NUFFT, as in

$$
a _ { \ell j k } = \frac { 8 } { n _ { x } n _ { y } n _ { z } } \sum _ { p = 1 } ^ { n _ { x } } \sum _ { q = 1 } ^ { n _ { y } } \sum _ { i = 1 } ^ { n _ { z } } f ( x _ { p } , y _ { q } , z _ { i } ) e ^ { - \frac { \pi } { 2 } ( x _ { p } , y _ { q } , z _ { i } ) \cdot \xi _ { \ell j k } } , \quad \ell = 1 , \dots , n _ { z } , \quad j = 1 , \dots , n _ { \theta } , \quad k = 1 , \dots , n _ { \varphi } .
$$

3. Compute the inverse 1-D discrete Fourier transform of the $a _ { \ell j k }$ , i.e., compute

$$
( P _ { f } ) _ { i , j , k } = \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { n _ { z } } a _ { \ell j k } e ^ { \frac { \pi } { 2 } n _ { z } z _ { \ell } z _ { i } } , \quad i = 1 , \ldots , n _ { z } , \quad j = 1 , \ldots , n _ { \theta } , \quad k = 1 , \ldots , n _ { \varphi } .
$$

This step can be performed using a standard FFT, after noting that

$$
\sum _ { \ell = 1 } ^ { n _ { z } } a _ { \ell j k } e ^ { \frac { \pi } { 2 } n _ { z } z _ { \ell } z _ { i } } = \sum _ { \ell = 1 } ^ { n _ { z } } a _ { \ell j k } e ^ { 2 \pi ( \ell - n _ { z } / 2 ) ( i - n _ { z } / 2 ) / n _ { z } } .
$$

The complexity of the NUFFT step performed with $\ell ^ { 1 } - \ell ^ { \infty }$ relative error ε is $\mathcal { O } ( n _ { x } n _ { y } n _ { z } \log ( n _ { x } n _ { y } n _ { z } ) +$ $n _ { z } n _ { \varphi } n _ { \theta } | \log \varepsilon | ^ { 3 } ) \ [ 3 5 , 3 6 ]$ . The complexity of the FFT step is $\mathcal { O } ( n _ { \theta } n _ { \varphi } n _ { z } \log n _ { z } )$

For the ramp-filtered Wasserstein distance, we define a version of $a _ { \ell j k }$ rescaled by the radial distance $| \xi _ { \ell j k } | ^ { 2 }$ , i.e.,

$$
a _ { \ell j k } ^ { s } = \frac { 8 } { n _ { x } n _ { y } n _ { z } } \sum _ { p = 1 } ^ { n _ { x } } \sum _ { q = 1 } ^ { n _ { y } } \sum _ { i = 1 } ^ { n _ { z } } | \xi _ { \ell j k } | ^ { 2 } f ( x _ { p } , y _ { q } , z _ { i } ) e ^ { - \frac { \pi } { 2 } ( x _ { p } , y _ { q } , z _ { i } ) \cdot \xi _ { \ell j k } } ,\tag{18}
$$

for each $\ell = 1 , \ldots , n _ { z } , \quad j = 1 , \ldots , n _ { \theta } , \quad k = 1 , \ldots , n _ { \varphi }$ . Subsequently, we define the signed and normalized Radon transforms

$$
( P _ { f } ^ { + } ) _ { i , j , k } = \operatorname* { m a x } \left( \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { n _ { z } } a _ { \ell j k } e ^ { \frac { \pi } { 2 } n _ { z } z _ { \ell } z _ { i } } , 0 \right) , \quad ( P _ { f } ^ { - } ) _ { i , j , k } = \operatorname* { m i n } \left( \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { n _ { z } } a _ { \ell j k } e ^ { \frac { \pi } { 2 } n _ { z } z _ { \ell } z _ { i } } , 0 \right)
$$

$$
( \tilde { P } _ { f } ^ { + } ) _ { i , j , k } = \frac { ( P _ { f } ^ { + } ) _ { i , j , k } } { \| P _ { f } ^ { + } \| _ { 1 } } , \quad ( \tilde { P } _ { f } ^ { - } ) _ { i , j , k } = \frac { ( P _ { f } ^ { - } ) _ { i , j , k } } { \| P _ { f } ^ { - } \| _ { 1 } }
$$

for all $i = 1 , \ldots , n _ { z } , \quad j = 1 , \ldots , n _ { \theta } , \quad k = 1 , \ldots , n _ { \varphi } .$ where we define

$$
\| P _ { f } ^ { \pm } \| _ { 1 } = \sum _ { i , j , k } w _ { i , j , k } | ( P _ { f } ^ { \pm } ) _ { i , j , k } | \cdot \sin ( \theta _ { k } ) ,
$$

for some quadrature weights $w _ { i , j , k }$ . We note that in our code implementation, the normalization is applied after transforming the signed projections into their respective discretized cdfs (described in the next section), which provides an equivalent normalization.

## 4.2.2 Step 2: distances

To compute the discretized 2-Wasserstein distance, we first convert the discretized projections $P _ { f }$ and $P _ { g }$ into their respective discretized cumulative distribution functions c $\mathrm { d f } _ { \theta _ { j } , \varphi _ { k } } f$ and $\mathrm { c d f } _ { \theta _ { j } , \varphi _ { k } } g _ { ; }$ which we now define. For fixed $j , k$ , we approximate the integral defining cumulative distribution functions by quadrature on the points $z _ { i } , \mathrm { i . e . } ,$ , for an integer m, define

$$
( \mathrm { c d f } _ { \theta _ { j } , \varphi _ { k } } f ) ( m ) = \sum _ { i = 0 } ^ { m } w _ { i } ( P _ { f } ) _ { i , j , k } ,\tag{19}
$$

for some choice of quadrature weights $w _ { i }$ for the nodes $z _ { i }$ . Our implementation uses a trapezoid rule, but other choices are possible.

We define $( \mathrm { c d f } _ { \theta _ { j } , \varphi _ { k } } f ) ( s )$ for non-integer s through piecewise linear interpolation. Then, the discretized quantile function is

$$
( { \mathrm { i c d f } } _ { \theta _ { j } , \varphi _ { k } } f ) ( t ) : = \operatorname* { m i n } \{ s : ( \mathrm { c d f } _ { \theta _ { j } , \varphi _ { k } } f ) ( s ) > t \} .\tag{20}
$$

The squared 2-Wasserstein distance $d _ { W _ { 2 } } ^ { 2 } ( P _ { \theta _ { j } , \varphi _ { k } } f , P _ { \theta _ { j } , \varphi _ { k } } g )$ is then approximated by

$$
d W _ { j , k } ^ { 2 } : = \sum _ { i = 1 } ^ { n _ { t } } w _ { i } \left| ( \mathrm { i c d f } _ { \theta _ { j } , \varphi _ { k } } f ) ( t _ { i } ) - ( \mathrm { i c d f } _ { \theta _ { j } , \varphi _ { k } } g ) ( t _ { i } ) \right| ^ { 2 } ,\tag{21}
$$

where $( t _ { i } , w _ { i } ) , ~ i ~ = ~ 1 , \ldots , n _ { t }$ are some choice of quadrature nodes and weights, respectively. Our code implements a choice of either Gauss-Legendre quarature or the composite Simpson’s 1/3-rule.

We lastly approximate $d _ { S W _ { 2 } } ^ { 2 } ( f , g )$ in eq. (6) using $d W _ { j , k } ^ { 2 }$ and quadrature, i.e., compute

$$
d S W ^ { 2 } : = \frac { 1 } { 4 \pi } \sum _ { j = 1 } ^ { n _ { \theta } } \sum _ { k = 1 } ^ { n _ { \varphi } } w _ { j , k } d W _ { j , k } ^ { 2 } \sin \theta _ { j } ,\tag{22}
$$

for some weights $w _ { j , k } .$

Specifically, our code implements composite Simpson’s $1 / 3 \AA$ -rule in both the θ and $\varphi$ variables, but other choices are possible as well. For the ramp-filtered Wasserstein distance, we define $d S W ^ { \pm }$ as approximations to the two terms in eq. (12) by replacing $P _ { f }$ by $P _ { f } ^ { \pm }$ in eq. (19) and repeating equations eq. (20) – eq. (22) verbatim. We remark that this implementation is used for computing the discrete sliced 2-Wasserstein distance, but for rotational distance, we compute the optimizing rotation in eq. (10) using only eq. (20) as described in the next section.

## 4.2.3 Step 3: alignment and local refinement

We use Proposition 1 to find an optimal rotational alignment in the sliced Wasserstein distance, by aligning the two discretized helper tensors $h _ { f } , h _ { g } \in \mathbb { R } ^ { n _ { t } \times n _ { \theta } \times n _ { \varphi } }$ defined by

$$
( h _ { f } ) _ { i , j , k } = \frac { 1 } { t _ { i } } ( \mathrm { i c d f } _ { \theta _ { j } , \varphi _ { k } } f ) ( t _ { i } ) , \quad ( h _ { g } ) _ { i , j , k } = \frac { 1 } { t _ { i } } ( \mathrm { i c d f } _ { \theta _ { j } , \varphi _ { k } } g ) ( t _ { i } ) , \quad i = 1 , \dots , n _ { t } , \quad j = 1 , \dots , n _ { \theta } , \quad k = 1 , \dots , n _ { \varphi } ,
$$

in the Euclidean distance. Any choice of existing Euclidean distance alignment algorithm can be employed, and we in particular use the algorithm from [8].

Briefly, this algorithm works by using a spherical harmonic representation to provide an eficient basis for rotational alignment. For a fixed radius $t _ { i } ,$ the tensors of quantile functions $h _ { f }$ and $h _ { g }$ are expanded into spherical harmonic basis functions. The spherical harmonics coeficients of rotations of $h _ { f }$ and $h _ { g }$ are sums of Wigner-D matrices multiplied by the spherical harmonics coeficients of $h _ { f }$ and $h _ { g }$ themselves. This allows the rotational dependence of the alignment problem to be evaluated analytically and without loss of accuracy. The correlation between $h _ { g }$ and all rotations of $h _ { f }$ can therefore be computed by comparing their spherical harmonic coeficients weighted by Wigner-D matrices, enabling fast determination of the optimal rotation. In full detail, this algorithm defines a maximum bandlimit L of the spherical harmonics and sets $n _ { \theta } , n _ { \varphi } = \mathcal { O } ( L )$ . This gives total complexity $\mathcal { O } \left( ( n _ { t } + \log L ) L ^ { 3 } \right)$ , which therefore sets the overall complexity of our approach when $n _ { t } , n _ { \theta } , n _ { \varphi } = \mathcal { O } ( N )$ . We refer the readers to [8] for more details.

For the case of the ramp-filtered sliced Wasserstein distance, we note that a small extension of the procedure of [8] allows for an alignment of the vector-valued functions in eq. (15). The only required change consists in computing the correlation between the vector-valued functions, after which the steps of the procedure can be repeated verbatim.

Lastly, we note that the approach of [8] computes a search of rotations on a regular grid of Euler angles. It therefore computes an approximation $( \alpha , \beta , \gamma )$ of the optimal Euler angles in eq. (10), with accuracy determined by the resolution of the grid. This in turn improves with increasing $n _ { z } , n _ { \theta } , n _ { \phi }$ and $n _ { t } ,$ so a more accurate approximation can be obtained by increasing these parameters. However, we note that improved accuracy can also be achieved through a local refinement algorithm around the initially obtained angles $( \alpha , \beta , \gamma )$

A simple local refinement algorithm consists in repeating the grid search of [8] on a grid localized around the obtained angles $( \alpha , \beta , \gamma )$ . This can be done with small computational cost by storing the correlation between $h _ { f }$ and $h _ { g }$ computed above and replacing the last FFT step in [8] by a non-uniform FFT onto the localized grid. If desired, this can subsequently be repeated on increasingly fine localized grids. This approach is implemented in our code.

Alternatively, gradient-based local refinement can be used by computing the gradients of eq. (11) and eq. (12) with respect to $\alpha , \beta$ and γ. This can be done in closed-form after expanding $h _ { f }$ and $h _ { g }$ in spherical harmonics, which ensures that the rotational action becomes linear combinations of the spherical harmonic expansion coeficients with Wigner D-matrices.

## 5 Stability estimates for the sliced Wasserstein distance in 3-D

This section presents two stability estimates for the sliced Wasserstein distance. Theorem 5.1 gives the exact value of the sliced Wasserstein distance between a function and a translation of itself. Theorem 5.2 gives an upper bound for the sliced Wasserstein between a function and a (general) deformation of itself. Moreover, we prove that the constant of this bound is sharp. In the context of alignment, after aligning two structures using the presented algorithm, these two results then give exact and lower bounds, respectively, on the degree of heterogeneity between the structures. For the sake of completeness, we present estimates valid for the sliced p-Wasserstein distance $d _ { S W _ { p } } ,$ for any $1 \leq p < \infty$ . This distance is defined replacing the exponent 2 with p throughout eq. (6) and eq. (7).

The following result is a three-dimensional counterpart of [10, Theorem 3.2].

Theorem 5.1. Let $1 \leq p < \infty$ . Let $f : \mathbb { R } ^ { 3 } \to [ 0 , \infty )$ be a probability density function. For a translation vector $s \in \mathbb { R } ^ { 3 }$ , define

$$
( T _ { s } f ) ( x ) = f ( x - s ) .
$$

Then

$$
d _ { S W _ { p } } ( f , T _ { s } f ) = ( p + 1 ) ^ { - 1 / p } \lVert s \rVert .
$$

In particular, for $p = 2$

$$
d _ { S W _ { 2 } } ( f , T _ { s } f ) = \frac { 1 } { \sqrt { 3 } } \lVert s \rVert .
$$

The proof is given in Section A.1. The next result extends the above translation analysis to general spatially varying, mass-preserving deformations.

Theorem 5.2. Let $1 \leq p < \infty .$ , and let $f : \mathbb { R } ^ { 3 } \to [ 0 , \infty )$ be a probability density function. Let $\Psi : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ be $a C ^ { 1 }$ difeomorphism, and let $f _ { \Psi }$ denote the density obtained by pushing f forward through Ψ, namely

$$
f _ { \Psi } ( y ) = f \bigl ( \Psi ^ { - 1 } ( y ) \bigr ) \left| \operatorname* { d e t } D \Psi ^ { - 1 } ( y ) \right| ,
$$

where $D \Psi ^ { - 1 }$ denotes the Jacobian matrix of $\Psi ^ { - 1 }$ . Then

$$
d _ { S W _ { p } } ( f , f _ { \Psi } ) \leq ( p + 1 ) ^ { - 1 / p } \left( \int _ { { \mathbb { R } } ^ { 3 } } \| \Psi ( x ) - x \| ^ { p } f ( x ) d x \right) ^ { 1 / p } .
$$

The constant $( p + 1 ) ^ { - 1 / p }$ is sharp for every $1 \leq p < \infty$ . In particular, for $p = 2$

$$
d _ { S W _ { 2 } } ( f , f _ { \Psi } ) \leq \frac { 1 } { \sqrt { 3 } } \left( \int _ { \mathbb { R } ^ { 3 } } \| \Psi ( x ) - x \| ^ { 2 } f ( x ) d x \right) ^ { 1 / 2 } .
$$

The proof is given in Section A.2. We note that this result can be compared to [37, Theorem 2.3], which proves the bound

$$
d _ { S W _ { p } } ( f , f _ { \Psi } ) \leq \operatorname* { s u p } _ { x \in \mathbb { R } ^ { 3 } } \| \Psi ( x ) - x \| .\tag{23}
$$

Since $\begin{array} { r } { \left( \int _ { \mathbb { R } ^ { 3 } } \| \Psi ( x ) - x \| ^ { p } f ( x ) d x \right) ^ { 1 / p } \leq \operatorname* { s u p } _ { x \in \mathbb { R } ^ { 3 } } \| \Psi ( x ) - x \| \left( \int _ { \mathbb { R } ^ { 3 } } f ( x ) d x \right) ^ { 1 / p } = \operatorname* { s u p } _ { x \in \mathbb { R } ^ { 3 } } \| \Psi ( x ) - x \| } \end{array}$ (because f is a probability density function so is non-negative and integrates to 1), Theorem 5.2 improves eq. (23).

In addition to our direct proof, we can also prove Theorem 5.2 using existing results. Indeed, for general p, we apply the comparison $d _ { S W _ { p } } ^ { p } \leq c _ { 3 , p } d _ { W _ { 1 } } ^ { p }$ from [38, Proposition 5.1.3]. We evaluate its constant in three dimensions as

$$
c _ { 3 , p } = { \frac { 1 } { 3 } } \int _ { S ^ { 2 } } \| \theta \| _ { p } ^ { p } d \sigma ( \theta ) = { \frac { 1 } { 2 } } \int _ { - 1 } ^ { 1 } | t | ^ { p } d t = { \frac { 1 } { p + 1 } } ,
$$

where σ is the uniform probability measure on $S ^ { 2 }$ . Combining this comparison with the coupling induced by $x \mapsto \left( x , \Psi ( x ) \right)$ , we obtain our deformation bound with exactly the same factor $( p + 1 ) ^ { - 1 / p }$

## 6 Numerical results

In this section, we provide numerical results that demonstrate the properties of the sliced 2-Wasserstein distance for comparing voxelized 3-D functions, and for its application to rigid alignment over rotations.

## 6.1 Timing results

First we show the computational runtime to transform the voxelized 3-D data into the quantile tensors used for the sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance. This step reflects the only precompute time required to transform the data prior to using other voxel-based alignment algorithms. As described in Section 4.2, the complexity of the NUFFT step performed with $\ell ^ { 1 } - \ell ^ { \infty }$ relative error ε is $\mathcal { O } ( n _ { x } n _ { y } n _ { z } \log ( n _ { x } n _ { y } n _ { z } ) + n _ { z } n _ { \varphi } n _ { \theta } | \log \varepsilon | ^ { 3 } )$ . Specifically, we use the Flatiron Institute NUFFT implementation in our code [35, 36]. The complexity of the subsequent FFT step is $\mathcal { O } ( n _ { \theta } n _ { \varphi } n _ { z } \log n _ { z } )$ . We show timing results for increasing number of voxels and report the average time for $n = 5$ trials in Table 1. All timing results are performed using a 2.6 GHz Intel Skylake processor and 32 GB of memory. These results show that the linear transforms required to compute the quantile tensor grow approximately as $\mathcal { O } ( N ^ { 3 } \log N )$ . We show timing results for the full alignment algorithm compared to other methods in Section 6.4.

Table 1: Quantile tensor transform runtime. Computational runtime for transforming voxel data into the quantile tensor used to compute the the sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance for increasing $N \times N \times N$ volumes. Values are reported in seconds as the mean and standard deviation of $n = 5$ trials.
<table><tr><td>Metric</td><td>32</td><td>64</td><td>128</td><td>256</td><td>512</td></tr><tr><td>Sliced 2-Wasserstein</td><td> $\overline { { 0 . 0 2 3 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 1 1 8 \pm 0 . 0 5 } }$ </td><td> $\overline { { 0 . 6 2 7 \pm 0 . 0 3 } }$ </td><td> $\overline { { 4 . 7 9 6 \pm 0 . 0 1 5 } }$ </td><td> $\overline { { 3 9 . 8 7 9 \pm 0 . 4 8 3 } }$ </td></tr><tr><td>Ramp-Filtered Sliced 2-Wasserstein</td><td> $0 . 0 2 8 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 1$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 3$ </td><td> $6 . 3 4 2 \pm 0 . 0 2 8$ </td><td> $5 2 . 2 4 9 \pm 0 . 6 1 9$ </td></tr></table>

## 6.2 Translation equivariance

Next, we demonstrate the translational equivariance property of the sliced Wasserstein distance in 3-D. As shown in Theorem 5.1, given a 3-D function $f ( x )$ and its shifted version $f ( x - s )$ , the sliced Wasserstein distance will change proportionally to the magnitude of the shift. To demonstrate this, we generate a simple 3-D Gaussian as $\exp ( - ( x ^ { 2 } + y ^ { 2 } + z ^ { 2 } ) / 2 \sigma ^ { 2 } )$ , with $\sigma = 5 .$ on a $N \times N \times N = 1 2 8 \times 1 2 8 \times 1 2 8$ voxel grid. We then shift the Gaussian in voxel increments along one axis, corresponding to a distance of 0.015625 on our defined scale of [−1, 1], and compute the squared Euclidean, sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance with the original (non-shifted) Gaussian. In Figure 2, we see a linear response from the optimal transport metrics but not for the Euclidean distance. This result demonstrates the favorable translational equivariance property of the sliced Wasserstein distance.

## 6.3 ModelNet10 data alignment

Next we test rotational alignment in sliced Wasserstein distance using the ModelNet10 [1] dataset. This dataset consists of computer-aided design models from 10 diferent categories (bathtub, bed, chair, desk, dresser, monitor, night stand, sofa, table, toilet). The diferent design of the models in each category provides a challenging dataset for 3-D alignment algorithms. Each model is represented by a mesh that is voxelized into a $N \times N \times N = 6 4 \times 6 4 \times 6 4$ binary occupancy grid using the voxelization implementation in the Trimesh $\mathrm { P y }$ thon library. We then apply a random rotation to each model.

![](images/371589913786f7c49cf999867eac93e9297e0804aa75926a0427381903f14f22.jpg)

![](images/d8e5f44d7086f671c8b6877f5e831f825f7dc7703f9fabe93e7f1fb1919d2e04.jpg)  
Figure 2: Translation equivariance of the sliced 2-Wasserstein distance in 3-D. A reference 3-D volume (standard 3-D Gaussian, with $\sigma = 5 )$ , is shifted in voxel increments and the distance is computed in Euclidean, sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance between the reference and shifted volume. Each metric is normalized by its maximum value for visualization.

For each model category, we compare the alignment results using the Euclidean, sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance. The models in each category are aligned to a reference model, which was selected as the model that minimizes the Euclidean distance to the average model of that category. Therefore, we expect the alignment results to favor alignment in Euclidean distance.

For a qualitative view of the alignment results, we display the first 11 models in the chair category aligned under each metric in Figure 3. We then summarize the alignment results for all models in the ModelNet10 test set, where the number of models per category ranges from $n = 5 0 \mathrm { t o } n = 1 0 0$ . To quantify the alignment results, we compute the geodesic distance between the ground truth rotation matrices and the estimated rotation matrices $R _ { 1 } , R _ { 2 } \in \mathsf { S O } ( 3 )$ , as defined by

$$
\small \Theta ( R _ { 1 } , R _ { 2 } ) = \cos ^ { - 1 } \left( \frac { t r ( R _ { 1 } ^ { \top } R _ { 2 } ) - 1 } { 2 } \right) .\tag{24}
$$

Specifically, because the models are considered to be aligned before the random rotation is applied, we measure how well the estimated alignment rotation matches the initial, random (ground truth) rotation. We then plot the cumulative number of models aligned within increasing radians from the ground truth rotation. We additionally note that some categories may contain symmetric models (e.g., table), which is not accounted for in the geodesic distance. Overall, these results show that the sliced Wasserstein metrics are either comparable, or greatly outperform the Euclidean distance for rotational alignment of the ModelNet10 data.

## 6.4 Alignment of 3-D cryo-EM data

A common task in cryo-EM pipelines is to align 3-D models [3, 29–32]. Cryo-EM data often contains two types of heterogeneity, which can challenge conventional alignment procedures. They are; 1) compositional heterogeneity, where the data contains diferent underlying structures, and 2) conformational heterogeneity, where the data contains the same underlying structure but with deformations. In Section 6.4.1, we first benchmark our alignment algorithm compared to other OT alignment methods using cryo-EM data without heterogeneity. Then, in Section 6.4.2, we demonstrate the results of our approach on heterogeneous cryo-EM data.

![](images/fce1b55e7b5e979461ee3e41698e99d4b1d7e2cc4845f9207a10e3725540cdfb.jpg)

![](images/1bbf763e4fe743b5f52cbb20c01d048dee9ce4b1263045163c184bb913fd1b8f.jpg)  
Euclidean sliced 2-Wasserstein ramp-filtered sliced 2-Wasserstein

Figure 3: Rotational alignment of the ModelNet10 data using diferent metrics. Models are discretized on a $N \times N \times N = 6 4 \times 6 4 \times 6 4$ voxel grid and are aligned to a reference model in Euclidean, sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance. (Top) Example alignment of the first 11 models from the chair category. The reference model is shown in purple and the aligned models are shown in gold. (Bottom) Summary of alignment results for 10 diferent categories. Each dataset contains n ≈ 100 models, and the plots show the number of correctly aligned models as a function of increasing radians under the geodesic distance in eq. (24).

## 6.4.1 Homogeneous Alignment

First we benchmark our alignment algorithm using 3-D models from published cryo-EM data without heterogeneity. We compare our algorithm to two other OT based methods including BOTalign [3] and AlignOT [32]. We test the alignment algorithms for 5 diferent cryo-EM volumes from the Electron Microscopy Data Bank (EMDB) [39] ranging in molecular weight from ≈ 170 − 2000 kDa, and resolution from ≈ 2.5 − 20<sup>˚</sup>A, as shown in Figure 4. In this section, we refer to our method as Fast Sliced Wasserstein 3-D (FSW3D), and specifically use the ramp-filtered sliced 2-Wasserstein version of the algorithm for our results.

In the first experiment, each volume is downsampled to $N \times N \times N = 6 4 \times 6 4 \times 6 4$ using the cryo-EM software ASPIRE [40]. To compare the diferent algorithms, we compute the geodesic distance eq. (24) between the estimated rotation alignment matrix and the ground truth rotation matrix for n = 50 trials, and report the mean and standard deviation in degrees, shown in Table 2. Euler angles for the ground truth rotations are selected uniformly and randomly as integers in {0, . . . , 359}. For BOTalign, we use the hyperparameter $\lbrace T = 1 5 0 \rbrace$ for the total number of iterations. For AlignOT, we use the cryo-EM volume threshold reported in the EMDB (required by AlignOT) and the hyperparameters $\{ n = 5 0 0 , \alpha = 0 . 1 , \epsilon = 1 0 0 , L = 1 0 0 \}$ which correspond to the number of points, the learning rate parameter, the regularization parameter and the maximum iterations, respectively. We additionally compare our method and BOTalign with and without their respective local refinement procedure. We also note that for AlignOT, alignment results tend to improve when the initialization is good [32] (i.e., the rotated volume is close to the original), however that is not the setting we are testing here. Our results show that, in general, FSW3D with local refinement is the best performing algorithm. Furthermore, we note that because our algorithm is deterministic and does not require hyperparameters or good initialization, the variance is lower than the other methods.

![](images/eb80c22b8c5ad39ba000b6e657688c57680913231444b047b8a10abdcf3cd2b1.jpg)  
Figure 4: Cryo-EM volumes from the EMDB [39] used for the homogeneous alignment experiments.

Table 2: Alignment accuracy results. The geodesic distance is computed between the estimated rotation and ground truth rotation for $n = 5 0$ random rotations of 5 diferent cryo-EM volumes from the EMDB [39], and is reported as the mean and standard deviation. The volumes are first downsampled to $N \times N \times N = 6 4 \times 6 4 \times 6 4$ Values are reported in degrees.
<table><tr><td></td><td>EMD-80134</td><td>EMD-2677</td><td>EMDB-8724</td><td>EMD-3342</td><td>EMD-1717</td></tr><tr><td>FSW3D</td><td> $\overline { { 2 . 1 2 \pm 0 . 7 7 } }$ </td><td> $\overline { { 2 . 1 2 \pm 0 . 7 7 } }$ </td><td> $\overline { { 2 . 1 4 \pm 0 . 7 8 } }$ </td><td> $\overline { { 2 . 1 3 \pm 0 . 7 6 } }$ </td><td> $\overline { { 2 . 1 2 \pm 0 . 7 7 } }$ </td></tr><tr><td>FSW3D + local</td><td> $0 . 0 9 \pm 0 . 0 4$ </td><td> $0 . 1 3 \pm 0 . 0 5$ </td><td> $0 . 2 1 \pm 0 . 0 9$ </td><td> $0 . 2 2 \pm 0 . 1$ </td><td> $0 . 0 8 \pm 0 . 0 4$ </td></tr><tr><td>BOTalign</td><td> $3 . 5 2 \pm 3 . 1 1$ </td><td> $9 . 5 4 \pm 1 9 . 1 8$ </td><td> $1 6 . 4 \pm 3 5 . 2 9$ </td><td> $2 . 6 5 \pm 1 . 1 9$ </td><td> $3 . 5 1 \pm 3 . 0 8$ </td></tr><tr><td> $\mathrm { B O T a l i g n + l o c a l }$ </td><td> $0 . 3 6 \pm 0 . 4 8$ </td><td> $0 . 2 1 \pm 0 . 0 9$ </td><td> $1 2 . 1 1 \pm 3 5 . 9 4$ </td><td> $0 . 2 2 \pm 0 . 1$ </td><td> $0 . 0 7 \pm 0 . 0 3$ </td></tr><tr><td> $\mathrm { { A l i g n O T } }$ </td><td> $7 8 . 0 4 \pm 6 7 . 8 1$ </td><td> $3 0 . 4 7 \pm 4 3 . 8 6$ </td><td> $1 2 4 . 6 7 \pm 4 4 . 9 2$ </td><td> $1 0 7 . 3 1 \pm 5 6 . 8 6$ </td><td> $1 1 9 . 7 7 \pm 3 6 . 9 7$ </td></tr></table>

Next we report the runtimes for each algorithm with volumes of size $N = \{ 3 2 , 6 4 , 1 2 8 \}$ in Table 3. The table shows the mean and standard deviation of $n = 5 0$ trials, reported in seconds. All results in Section 6.4.1 were done on a computer a 2.6 GHz Intel Skylake processor and 32 GB of memory. The reported computational complexity of each method is $\mathcal { O } ( N ^ { 4 } )$ for FSW3D, $\mathcal { O } ( T N ^ { 3 }$ log N) for BOTalign and $\mathcal { O } ( n ^ { 2 } L \log n \epsilon ^ { - 3 } )$ for AlignOT, where n is the number of points in the point cloud and L is the number of iterations. We note that for AlignOT, the complexity is dependent on the user defined number of points rather than the input number of voxels. However, because no guidance is given for selecting this parameter, we keep it fixed at $n = 5 0 0$ rather than scaling it linearly or cubically with the number of voxels. The more direct comparison between FSW3D and BOTalign shows that FSW3D has faster runtime at all sizes, even though the asymptotic complexity is worse. This section demonstrates that FSW3D has higher accuracy and faster runtimes among other OT based alignment algorithms.

Table 3: Timing results for EMDB-1717 at increasing number of voxels, N = {32, 64, 128}. Reported values are the mean and standard deviation of $n = 5 0$ trials, and are shown in seconds.
<table><tr><td></td><td>FSW3D</td><td>FSW3D + local BOTalign</td><td></td><td>BOTalign + local AlignOT</td><td></td></tr><tr><td> $\overline { { N = 3 2 } }$ </td><td> $\overline { { 0 . 2 3 \pm 0 . 0 1 } }$ </td><td> $\overline { { 0 . 3 8 \pm 0 . 0 7 } }$ </td><td> $\overline { { 6 . 5 7 \pm 0 . 1 4 } }$ </td><td> $\overline { { 1 2 . 2 5 \pm 1 . 9 } }$ </td><td> $\overline { { 4 . 8 3 \pm 0 . 0 4 } }$ </td></tr><tr><td> $N = 6 4$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $1 . 4 7 \pm 0 . 1 9$ </td><td> $3 7 . 3 5 \pm 0 . 5 5$ </td><td> $4 2 . 9 \pm 1 . 2 7$ </td><td> $4 . 8 1 \pm 0 . 0 5$ </td></tr><tr><td> $N = 1 2 8$ </td><td> $7 . 5 8 \pm 0 . 0 8$ </td><td> $1 0 . 9 4 \pm 1 . 3 8$ </td><td> $1 2 8 . 4 7 \pm 0 . 8 7$ </td><td> $1 4 2 . 2 \pm 2 . 9$ </td><td> $6 . 1 7 \pm 0 . 2 5$ </td></tr></table>

## 6.4.2 Heterogeneous Alignment

Next we test alignment results of our algorithm using heterogeneous cryo-EM data. Specifically, we focus on the case of conformational heterogeneity. We use a dataset from [27] that contains an $N \times N \times N =$ $1 2 8 \times 1 2 8 \times 1 2 8$ volume with simulated motion represented by n = 50 discrete states. In Figure 5a, the models are aligned to the initial (reference) volume to show the axis of motion in the data. We then compute the distance, not over rotations, between the initial volume and each of the subsequent volumes in motion using the Euclidean, sliced 2-Wasserstein and ramp-filtered sliced 2-Wasserstein distance. In this example of simple motion, the Wasserstein metrics demonstrate the favorable property of stability under deformation. However, this property will not always hold for more complicated motion. Nonetheless, this result suggests that, given a set of aligned volumes, the Wasserstein metrics can provide a more descriptive measure of similarity.

Next, in Figure 5b, we compare the rotational alignment between the initial and final volume using each metric. Notably, the alignment from all metrics yield diferent orientations. In particular, the Euclidean distance appears to align the ‘head’ of the volume to be aligned correctly, but the ‘leg’ no longer overlaps with the reference. Meanwhile, the Wasserstein metrics appear to keep all parts of the volumes closer together, but with some parts aligned in diferent orientations (e.g. the ‘head’ is rotated compared to the reference). This result demonstrates that the diferent metrics yield distinct alignment results for heterogeneous data and that, depending on the task, one metric might be preferred over the other.

## 7 Discussion

In this work, we develop a framework for fast alignment over SO(3) in the sliced 2-Wasserstein distance. Additionally, we provide theoretical results on the stability of sliced 2-Wasserstein distance in 3-D. This work extends the applicability of optimal transport and sliced probability metrics to the 3-D case, in particular for high-dimensional data that requires alignment over a large grid of rotations. The ability to rapidly compute optimal transport based metrics for large, voxelized data will allow for an alternative way to compare objects with distinct 3-D structures. We also note that this algorithm is readily implementable on GPUs.

Several important areas remain to be addressed to increase the applicability of this method. For example, this analysis assumes that the input volumes are strictly positive and have the same initial mass. However it is common to have mean-subtracted or noisy data. Another interesting topic to explore comes from observing that there is no true alignment between non-identical functions. To analyze this further, one could test the downstream efects of algorithms that first require 3-D alignment by comparing the results from both the Euclidean distance and sliced Wasserstein distance based alignments, similar to the work in [41]. Other algorithmic choices worth exploring include the choice of grid points, frequency marching schemes [42], multigrid schemes, numerical integration methods, interpolation schemes, and connections to related invariant metrics [20].

![](images/3da0e3dbfeed1f5bdc309682bd4ff40f17fc78589c8d2558748a7cea367ef145.jpg)

![](images/2dc26fc9d62e7f29550de8dabe6597b82cbf54085a2b10a10274bb316777102a.jpg)

![](images/f25c39c5a03125efce8b22f433aa7bc3ba8dbbb6d9a2ddb26cc2fc6e8dd22276.jpg)  
Figure 5: Distance and alignment of heterogeneous cryo-EM data. (a) Cryo-EM volumes of size $N { \times } N { \times } N =$ $1 2 8 \times 1 2 8 \times 1 2 8$ with simulated motion (see [27] for more details about the data). The distances are computed between the initial volume and the subsequent volumes with increasing state of motion. (b) Rotational alignment of the final volume to the initial volume under each metric. Orthogonal projection plots show the diference between the initial and final volume after rotational alignment, projected onto each orthogonal plane. Alignment in both sliced Wasserstein distances gives similar results which are distinct from the alignment in Euclidean distance.

## A Proofs

## A.1 Proof of Theorem 5.1

Proof. Define

$$
\begin{array} { r } { u _ { \theta , \varphi } : = R _ { \theta , \varphi } ^ { \top } e _ { z } = ( \sin \theta \cos \varphi , \sin \theta \sin \varphi , \cos \theta ) ^ { \top } . } \end{array}
$$

Thus $u _ { \theta , \varphi }$ is the projection direction corresponding to $P _ { \theta , \varphi }$ . Write $g ( x ) = f ( x - s )$ . For each $z \in \mathbb { R }$

$$
\begin{array} { r l } & { ( P _ { \theta , \varphi } g ) ( z ) = \displaystyle \int _ { \mathbb R ^ { 2 } } g \big ( R _ { \theta , \varphi } ^ { \top } ( x , y , z ) ^ { \top } \big ) \ d x d y } \\ & { \quad \quad \quad = \displaystyle \int _ { \mathbb R ^ { 2 } } f \big ( R _ { \theta , \varphi } ^ { \top } ( x , y , z ) ^ { \top } - s \big ) \ d x d y } \\ & { \quad \quad = \displaystyle \int _ { \mathbb R ^ { 2 } } f \big ( R _ { \theta , \varphi } ^ { \top } \big ( ( x , y , z ) ^ { \top } - R _ { \theta , \varphi } s \big ) \big ) \ d x d y . } \end{array}
$$

Writing $R _ { \theta , \varphi } s = ( ( R _ { \theta , \varphi } s ) _ { 1 } , ( R _ { \theta , \varphi } s ) _ { 2 } , ( R _ { \theta , \varphi } s ) _ { 3 } ) ^ { \intercal }$ and changing variables

$$
x ^ { \prime } = x - ( R _ { \theta , \varphi } s ) _ { 1 } , \qquad y ^ { \prime } = y - ( R _ { \theta , \varphi } s ) _ { 2 } ,
$$

we obtain

$$
\begin{array} { c l } { ( P _ { \theta , \varphi } g ) ( z ) = \displaystyle \int _ { \mathbb R ^ { 2 } } f \big ( R _ { \theta , \varphi } ^ { \top } ( x ^ { \prime } , y ^ { \prime } , z - ( R _ { \theta , \varphi } s ) _ { 3 } ) ^ { \top } \big ) \ d x ^ { \prime } d y ^ { \prime } } \\ { = ( P _ { \theta , \varphi } f ) ( z - ( R _ { \theta , \varphi } s ) _ { 3 } ) . } \end{array}
$$

Since

$$
\begin{array} { r } { ( R _ { \theta , \varphi } s ) _ { 3 } = e _ { z } ^ { \top } R _ { \theta , \varphi } s = ( R _ { \theta , \varphi } ^ { \top } e _ { z } ) ^ { \top } s = u _ { \theta , \varphi } \cdot s , } \end{array}
$$

we have

$$
P _ { \theta , \varphi } g = \tau _ { u _ { \theta , \varphi } \cdot s } P _ { \theta , \varphi } f ,
$$

where $\tau _ { a } \mu ( z ) = \mu ( z - a )$ denotes translation by a in one dimension.

For any one-dimensional probability density $\mu$ and any $a \in \mathbb { R }$ , the quantile function of $\tau _ { a } \mu$ is shifted by a:

$$
( { \mathcal V } \tau _ { a } \mu ) ^ { - 1 } ( t ) = ( { \mathcal V } \mu ) ^ { - 1 } ( t ) + a .
$$

Therefore,

$$
d _ { W _ { p } } ^ { p } ( \mu , \tau _ { a } \mu ) = \int _ { 0 } ^ { 1 } | a | ^ { p } d t = | a | ^ { p } .
$$

Applying this with $a = u _ { \theta , \varphi } \cdot s \mathrm { \ g i }$ ves

$$
d _ { W _ { p } } ^ { p } ( P _ { \theta , \varphi } f , P _ { \theta , \varphi } g ) = | u _ { \theta , \varphi } \cdot s | ^ { p } .
$$

Hence

$$
d _ { S W _ { p } } ^ { p } ( f , T _ { s } f ) = \frac { 1 } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } | u _ { \theta , \varphi } \cdot s | ^ { p } \sin \theta d \varphi d \theta .
$$

By rotational invariance of the surface measure on $S ^ { 2 }$ , the integral depends on s only through $\| s \|$ . Thus we may assume without loss of generality that $\boldsymbol { s } = \| \boldsymbol { s } \| e _ { z }$ . Then

$$
u _ { \theta , \varphi } \cdot s = \| s \| \cos \theta ,
$$

and so

$$
\begin{array} { l } { d _ { S W _ { p } } ^ { p } ( f , T _ { s } f ) = \displaystyle \frac { \| s \| ^ { p } } { 4 \pi } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } | \cos \theta | ^ { p } \sin \theta d \varphi d \theta } \\ { = \displaystyle \frac { \| s \| ^ { p } } { 2 } \int _ { 0 } ^ { \pi } | \cos \theta | ^ { p } \sin \theta d \theta } \\ { = \displaystyle \frac { \| s \| ^ { p } } { 2 } \int _ { - 1 } ^ { 1 } | t | ^ { p } d t } \\ { = \displaystyle \frac { \| s \| ^ { p } } { 2 } \cdot \displaystyle \frac { 2 } { p + 1 } = \displaystyle \frac { 1 } { p + 1 } \| s \| ^ { p } . } \end{array}
$$

Taking the p-th root completes the proof.

## A.2 Proof of Theorem 5.2

Proof. For $( \theta , \varphi ) \in [ 0 , \pi ] \times [ 0 , 2 \pi )$ , let

$$
\boldsymbol { u } _ { \theta , \varphi } = ( \sin \theta \cos \varphi , \sin \theta \sin \varphi , \cos \theta ) ^ { \top }
$$

be the corresponding projection direction. Let X be an $\mathbb { R } ^ { 3 } .$ -valued random vector with density $f ,$ and set $Y = \Psi ( X )$ , so that $\bar { Y }$ has density $f _ { \Psi }$ . Fix $( \theta , \varphi )$ and write $u = u _ { \theta , \varphi }$ . For every Borel set $A \subseteq \mathbb { R }$

$$
\mathbb { P } ( u \cdot X \in A ) = \int _ { \{ x : u \cdot x \in A \} } f ( x ) d x = \int _ { A } ( P _ { \theta , \varphi } f ) ( t ) d t ,
$$

where the last equality follows by integrating $f$ over the planes perpendicular to u. Thus, $u \cdot X$ has density $P _ { \theta , \varphi } f .$ The same argument shows that $u \cdot Y$ has density $P _ { \theta , \varphi } f _ { \Psi }$ . Consequently, the joint law of

$$
( u \cdot X , \ u \cdot Y ) = ( u \cdot X , \ u \cdot \Psi ( X ) )
$$

is a coupling of these two projected densities. Therefore, by the Kantorovich formulation of the Wasserstein distance [4, 5],

$$
\begin{array} { r l } & { d _ { W _ { p } } ^ { p } \left( P _ { \theta , \varphi } f , P _ { \theta , \varphi } f _ { \Psi } \right) \leq \mathbb { E } \left[ \left| u \cdot Y - u \cdot X \right| ^ { p } \right] } \\ & { \qquad = \displaystyle \int _ { \mathbb { R } ^ { 3 } } \left| u _ { \theta , \varphi } \cdot \left( \Psi ( x ) - x \right) \right| ^ { p } f ( x ) d x . } \end{array}
$$

Averaging this inequality over all projection directions and applying Tonelli’s theorem gives

$$
d _ { S W _ { p } } ^ { p } ( f , f _ { \Psi } ) \leq \int _ { \mathbb { R } ^ { 3 } } \left[ { \frac { 1 } { 4 \pi } } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } \left| u _ { \theta , \varphi } \cdot \left( \Psi ( x ) - x \right) \right| ^ { p } \sin \theta d \varphi d \theta \right] f ( x ) d x .
$$

For every $v \in \mathbb { R } ^ { 3 }$ , rotational invariance of the uniform surface measure on $S ^ { 2 }$ implies

$$
{ \frac { 1 } { 4 \pi } } \int _ { 0 } ^ { \pi } \int _ { 0 } ^ { 2 \pi } | u _ { \theta , \varphi } \cdot v | ^ { p } \sin \theta d \varphi d \theta = { \frac { \| v \| ^ { p } } { 2 } } \int _ { - 1 } ^ { 1 } | t | ^ { p } d t = { \frac { 1 } { p + 1 } } \| v \| ^ { p } .
$$

Applying this identity with $v = \Psi ( x ) - x$ yields

$$
d _ { S W _ { p } } ^ { p } ( f , f _ { \Psi } ) \leq \frac { 1 } { p + 1 } \int _ { { \mathbb R } ^ { 3 } } \| \Psi ( x ) - x \| ^ { p } f ( x ) d x .\tag{25}
$$

Taking the $p \mathrm { - }$ th root proves the stated bound.

To prove sharpness of the bound, we choose $\Psi _ { s } ( x ) = x + s$ . The push-forward density is

$$
f _ { \Psi _ { s } } ( x ) = f ( x - s ) = T _ { s } f ( x ) ,
$$

and its density-weighted displacement is exactly

$$
\left( \int _ { \mathbb { R } ^ { 3 } } \| \Psi _ { s } ( x ) - x \| ^ { p } f ( x ) d x \right) ^ { 1 / p } = \| s \| .
$$

eq. (25) therefore gives

$$
d _ { S W _ { p } } ( f , T _ { s } f ) \leq ( p + 1 ) ^ { - 1 / p } \Vert s \Vert .
$$

Theorem 5.1 shows that equality always holds. Thus, the translation theorem is the exact equality case of the general deformation bound and proves that its constant cannot in general be improved. For a general nonlinear deformation, the coupling used in the proof need not be optimal after every one-dimensional projection, so the inequality can be strict. □

## References

1. Zhirong Wu et al. 3D ShapeNets: A Deep Representation for Volumetric Shapes in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE, June 2015), 1912–1920. doi:10. 1109/CVPR.2015.7298801.

2. Singer, A. & Sigworth, F. J. Computational Methods for Single-Particle Electron Cryomicroscopy. Annual Review of Biomedical Data Science 3, 163–190. doi:10.1146/annurev-biodatasci-021020- 093826 (July 2020).

3. Singer, A. & Yang, R. Alignment of Density Maps in Wasserstein Distance. Biological Imaging 4, e5. doi:10.1017/S2633903X24000059 (2024).

4. Villani, C. Optimal Transport: Old and New doi:10.1007/978- 3- 540- 71050- 9 (Springer Berlin Heidelberg, 2009).

5. Peyr´e, G. & Cuturi, M. Computational Optimal Transport with Applications to Data Sciences. Foundations and Trends in Machine Learning 11, 355–607. doi:10.1561/2200000073 (Feb. 2019).

6. Cuturi, M. Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances. doi:10. 48550/ARXIV.1306.0895 (2013).

7. Solomon, J. et al. Convolutional Wasserstein Distances: Eficient Optimal Transportation on Geometric Domains. ACM Transactions on Graphics 34, 1–11. doi:10.1145/2766963 (July 2015).

8. Kovacs, J. A. & Wriggers, W. Fast Rotational Matching. Acta Crystallographica Section D Biological Crystallography 58, 1282–1286. doi:10.1107/S0907444902009794 (July 2002).

9. Kyatkin, A. B. & Chirikjian, G. S. Algorithms for Fast Convolutions on Motion Groups. Applied and Computational Harmonic Analysis 9, 220–241. doi:10.1006/acha.2000.0321 (Sept. 2000).

10. Shi, Y., Singer, A. & Verbeke, E. J. Fast Rigid Alignment of Heterogeneous Images in Sliced Wasserstein Distance. SIAM Journal on Imaging Sciences 19, 1469–1515. doi:10.1137/25M1742916 (Sept. 2026).

11. Besl, P. & McKay, N. D. A Method for Registration of 3-D Shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence 14, 239–256. doi:10.1109/34.121791 (Feb. 1992).

12. Bing Jian & Vemuri, B. C. Robust Point Set Registration Using Gaussian Mixture Models. IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1633–1645. doi:10.1109/TPAMI.2010. 223 (Aug. 2011).

13. Khoo, Y. & Kapoor, A. Non-Iterative Rigid 2D/3D Point-Set Registration Using Semidefinite Programming. IEEE Transactions on Image Processing 25, 2956–2970. doi:10.1109/TIP.2016.2540810 (July 2016).

14. Hutchison, D. et al. in Computer Vision – ECCV 2010 356–369 (Springer Berlin Heidelberg, 2010). doi:10.1007/978-3-642-15558-1\_26.

15. Ovsjanikov, M., Ben-Chen, M., Solomon, J., Butscher, A. & Guibas, L. Functional Maps: A Flexible Representation of Maps between Shapes. ACM Transactions on Graphics 31, 1–11. doi:10 . 1145 / 2185520.2185526 (Aug. 2012).

16. Rabin, J., Peyr´e, G., Delon, J. & Bernot, M. in Scale Space and Variational Methods in Computer Vision 435–446 (Springer Berlin Heidelberg, 2012). doi:10.1007/978-3-642-24785-9\_37.

17. Kolouri, S., Nadjahi, K., Simsekli, U., Badeau, R. & Rohde, G. K. Generalized Sliced Wasserstein Distances 2019. doi:10.48550/ARXIV.1902.00434.

18. Kolouri, S., Nadjahi, K., Shahrampour, S. & Simsekli, U. Generalized Sliced Probability Metrics in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (IEEE, May 2022), 4513–4517. doi:10.1109/ICASSP43922.2022.9746016.

19. Kolouri, S., Park, S. R. & Rohde, G. K. The Radon Cumulative Distribution Transform and Its Application to Image Classification. IEEE Transactions on Image Processing 25, 920–934. doi:10.1109/ TIP.2015.2509419 (Feb. 2016).

20. Beckmann, M., Beinert, R. & Bresch, J. Generalizations of the Normalized Radon Cumulative Distribution Transform for Limited Data Recognition. Journal of Mathematical Imaging and Vision 68, 45. doi:10.1007/s10851-026-01327-4 (Aug. 2026).

21. Quellmalz, M., Beinert, R. & Steidl, G. Sliced Optimal Transport on the Sphere. Inverse Problems 39, 105005. doi:10.1088/1361-6420/acf156 (Oct. 2023).

22. Bonet, C. et al. Spherical Sliced-Wasserstein 2022. doi:10.48550/ARXIV.2206.08780.

23. Tran, H. et al. Stereographic Spherical Sliced Wasserstein Distances 2024. doi:10.48550/ARXIV.2402. 02345.

24. Bendory, T., Bartesaghi, A. & Singer, A. Single-Particle Cryo-Electron Microscopy: Mathematical Theory, Computational Challenges, and Opportunities. IEEE Signal Processing Magazine 37, 58–76. doi:10.1109/MSP.2019.2957822 (Mar. 2020).

25. Kastritis, P. L. et al. Capturing Protein Communities by Structural Proteomics in a Thermophilic Eukaryote. Molecular Systems Biology 13, MSB167412. doi:10.15252/msb.20167412 (July 2017).

26. Verbeke, E. J., Mallam, A. L., Drew, K., Marcotte, E. M. & Taylor, D. W. Classification of Single Particles from Human Cell Extract Reveals Distinct Structures. Cell Reports 24, 259–268.e3. doi:10. 1016/j.celrep.2018.06.022 (July 2018).

27. Zhong, E. D., Bepler, T., Berger, B. & Davis, J. H. CryoDRGN: Reconstruction of Heterogeneous Cryo-EM Structures Using Neural Networks. Nature Methods 18, 176–185. doi:10.1038/s41592-020- 01049-4 (Feb. 2021).

28. Gilles, M. A. & Singer, A. Cryo-EM Heterogeneity Analysis Using Regularized Covariance Estimation and Kernel Regression. Proceedings of the National Academy of Sciences 122, e2419140122. doi:10. 1073/pnas.2419140122 (Mar. 2025).

29. Pettersen, E. F. et al. UCSF ChimeraX : Structure Visualization for Researchers, Educators, and Developers. Protein Science 30, 70–82. doi:10.1002/pro.3943 (Jan. 2021).

30. Harpaz, Y. & Shkolnisky, Y. Three-Dimensional Alignment of Density Maps in Cryo-Electron Microscopy. Biological Imaging 3, e8. doi:10.1017/S2633903X23000089 (2023).

31. He, B. et al. Accurate Global and Local 3D Alignment of Cryo-EM Density Maps Using Local Spatial Structural Features. Nature Communications 15, 1593. doi:10.1038/s41467- 024- 45861- 4 (Feb. 2024).

32. Riahi, A. T., Woollard, G., Poitevin, F., Condon, A. & Duc, K. D. AlignOT: An Optimal Transport Based Algorithm for Fast 3D Alignment With Applications to Cryogenic Electron Microscopy Density Maps. IEEE/ACM Transactions on Computational Biology and Bioinformatics 20, 3842–3850. doi:10. 1109/TCBB.2023.3327633 (Nov. 2023).

33. Tajmir Riahi, A., Zhang, C., Condon, A., Chen, J. & Dao Duc, K. Alignment of Partially Overlapping Cryo-EM Maps Using Unbalanced Gromov-Wasserstein Divergence. PRX Life 3, 023003. doi:10.1103/ PRXLife.3.023003 (Apr. 2025).

34. Kak, A. C. & Slaney, M. Principles ofComputerized Tomographic Imaging doi:10.1137/1.9780898719277 (Society for Industrial and Applied Mathematics, Jan. 2001).

35. Barnett, A. H., Magland, J. & Af Klinteberg, L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle” Kernel. SIAM Journal on Scientific Computing 41, C479– C504. doi:10.1137/18M120885X (Jan. 2019).

36. Barnett, A. H. Aliasing Error of the Exp ( β 1 - z 2 ) Kernel in the Nonuniform Fast Fourier Transform. Applied and Computational Harmonic Analysis 51, 1–16. doi:10.1016/j.acha.2020.10.002 (Mar. 2021).

37. Leeb, W. On Sliced Cram´er Metrics Aug. 2026. doi:10.48550/ARXIV.2508.02678.

38. Bonnotte, N. Unidimensional and Evolution Methods for Optimal Transportation PhD thesis (Dec. 2013). doi:10.70675/2833ed85z6cd3z4115z99bazba760e7494de.

39. The wwPDB Consortium et al. EMDB—the Electron Microscopy Data Bank. Nucleic Acids Research 52, D456–D465. doi:10.1093/nar/gkad1019 (Jan. 2024).

40. Wright, G. et al. ComputationalCryoEM/ASPIRE-Python: V0.14.3 Zenodo. Apr. 2026. doi:10.5281/ ZENODO.5657281.

41. Woollard, G., Herreros, D., Li, M., Cossio, P. & Duc, K. D. Improving Cryo-EM Optimization Robustness with an Optimal Transport Loss Function for Noisy Images Dec. 2025. doi:10.64898/2025.12. 23.696001.

42. Kruse, F., Debarnot, V., Kishore, V. & Dokmani´c, I. Fast Volume Alignment by Frequency-Marched Newton Method July 2026. doi:10.48550/ARXIV.2603.15285.

## Acknowledgments

A.S. was supported in part by AFOSR FA9550-23-1-0249, DARPA HR0011-25-3-E002, NSF DMS 2510039, Biohub FP00007575, and the Simons Foundation Math+X Investigator Award; Y.S. was supported by NSF DMS 2514152.