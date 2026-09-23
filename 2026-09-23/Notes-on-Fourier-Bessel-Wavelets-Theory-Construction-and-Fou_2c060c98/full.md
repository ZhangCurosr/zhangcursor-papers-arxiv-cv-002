# Notes on Fourier-Bessel Wavelets Theory, Construction, and Fourier-Domain Representation

Marcel Venturotti<sup>∗</sup>

Georgios Exarchakis<sup>†</sup>

September 23, 2026

## Abstract

These notes develop the mathematical foundations and construction of a Fourier-Bessel wavelet family inspired by the disk harmonics of Shaqfa et al.[9]. We begin with the relevant properties of Bessel and modified Bessel functions and introduce the wavelet properties required for the construction. We then derive the Fourier-Bessel disk harmonics as solutions to the Helmholtz equation on the unit disk subject to a Neumann boundary condition.

Building on this basis, we construct a wavelet family by applying a Gaussian spatial envelope and introducing a zero-mean correction for the zeroth angular order. We derive the corresponding normalisation constants for $L ^ { 2 } .$ -based applications and discuss $L ^ { 1 } .$ -based normalisation for frequency-domain peak consistency. Finally, we derive a closed-form Fourier-domain representation of the resulting wavelets.

The main motivation is the approximately linear spacing, which converges to π between consecutive radial eigenvalues. Rather than replacing the conventional dyadic organisation of wavelet families, this construction lays out the foundation to explore whether a more uniform radial frequency allocation can be useful for applications in which broad and balanced frequency coverage is desirable.

## Contents

1 Introduction 4   
2 Mathematical Background 5   
2.1 Bessel Functions . 5   
2.1.1 Bessel Functions of the First and Second Kind 5   
2.1.2 Modified Bessel Functions . 6   
2.1.3 Derivative Identities for Bessel Functions 7   
2.2 Wavelet Properties 10   
3 Fourier-Bessel Disk Harmonics 11   
3.1 The Helmholtz Equation . 11   
3.2 Neumann Boundary Condition 12   
3.3 Root Finding Using Muller’s Method . 13   
4 Construction of Fourier-Bessel Wavelets 14   
4.1 Low-Pass Filter . 15   
4.2 Zero-Mean Correction 15   
4.3 L<sup>2</sup> Normalisation . 16   
4.3.1 Angular orders m ≥ 1 16   
4.3.2 Zeroth angular order m = 0 . 17   
4.4 Fourier-Domain Peak Normalisation 17   
4.4.1 Angular orders m ≥ 1 18   
4.4.2 Zeroth angular order m = 0 18   
4.4.3 Asymptotic case 18   
5 Fourier-Domain Representation 19   
6 Frequency Tiling and Eigenvalue Spacing 21   
7 Conclusion 24

<table><tr><td>Symbol Meaning</td><td></td></tr><tr><td> $\rho , \varphi$ </td><td>Radial and angular polar coordinates in the spatial domain.</td></tr><tr><td> $x , y$ </td><td>Cartesian spatial coordinates, with  $x = \rho$  cos  $\varphi$  and  $y = \rho$  sin  $\varphi .$ </td></tr><tr><td> $q , \phi$ </td><td>Radial and angular polar coordinates in the frequency domain.</td></tr><tr><td> $m$ </td><td>Angular order of the Fourier-Bessel function.</td></tr><tr><td> $k$ </td><td>Root index, corresponding to the k-th positive root of  $J _ { m } ^ { \prime }$ </td></tr><tr><td> $\lambda _ { m , k }$ </td><td>The k-th positive Neumann eigenvalue for angular order  $m ,$  satis- fying  $J _ { m } ^ { \prime } ( \lambda _ { m , k } ) = 0$ </td></tr><tr><td> $J _ { m }$ </td><td>Bessel functions of the first kind.</td></tr><tr><td> $I _ { m }$ </td><td>Modified Bessel functions of the first kind.</td></tr><tr><td> $N _ { m , k }$ </td><td>Normalisation constant for the Fourier-Bessel disk basis.</td></tr><tr><td> $K _ { m , k }$   $\boldsymbol { \mathrm { { N } ^ { ( 2 ) } } }$ </td><td>Zero-mean correction term, non-zero only for  $m = 0 ,$ </td></tr><tr><td> $\mathbf { \Delta } ^ { \varDelta } m , k$   $N _ { \quad } ^ { ( 1 ) }$   $N _ { m , k } ^ { \setminus ( \cdot ) }$ </td><td> $L ^ { 2 }$  normalisation constant for the wavelet. Peak normalisation constant, obtained from the radial Fourier</td></tr><tr><td></td><td>response.</td></tr><tr><td> $\underset {  } { \psi _ { m , k } }$ </td><td>Fourier-Bessel wavelet in the spatial domain.</td></tr><tr><td> $\psi _ { m , k }$ </td><td>Fourier-Bessel wavelet in the frequency domain.</td></tr><tr><td> $q _ { m , k } ^ { * }$ </td><td>Frequency at which the radial Fourier response attains its maxi- mum.</td></tr></table>

Table 1: Notation used throughout the notes.

## 1 Introduction

Solid harmonic wavelets [5] provided an early example of constructing wavelets from solutions to diferential equations. By using harmonic functions, which are solutions to Laplace’s equation, the authors obtained wavelets whose Fourier representations form approximately ring-shaped structures rather than the more traditional Gaussian-shaped responses. This provides controlled coverage of frequency space with tunable overlap. Nevertheless, the resulting filter banks still rely on dyadic scaling and rotation, which produces a frequency organisation that places progressively greater separation between higher frequency bands.

Wavelets provide a natural framework for analysing signals at multiple scales, decomposing a signal into localised oscillatory components indexed jointly by position and scale [7, 4]. The multiresolution analysis introduced by Mallat [6] formalised this idea by organising signal information into a hierarchy of nested approximation spaces linked by dyadic dilations, giving wavelet decompositions their characteristic logarithmic frequency tiling. This dyadic organisation, in which each scale doubles the previous one, underlies much of classical wavelet theory, including results on regularity, sparsity, and stability. More recently, wavelet filter banks have been used as the basis for scattering networks [3, 10], which cascade wavelet transforms with pointwise nonlinearities and averaging operators to build signal representations that are stable to deformations while retaining high-frequency information that is otherwise lost under simple averaging. These constructions typically inherit the dyadic scale structure of classical wavelets, motivating interest in alternative filter constructions that depart from this scaling while retaining wavelet-like localisation properties.

In this work, we explore a diferent construction inspired by the disk harmonics introduced by Shaqfa et al. [9]. Rather than starting from solutions to Laplace’s equation, we use the Bessel-function solutions of the Helmholtz equation on the unit disk. These functions provide radial oscillations whose corresponding eigenvalues become approximately linearly spaced, with asymptotic spacing π. We use these eigenfunctions as the oscillatory component of a new wavelet family.

The resulting Fourier-Bessel wavelets share the ring-like frequency structure of solid harmonic wavelets, while replacing the traditional dyadic scale parameter with the eigenvalue associated with the radial Bessel function. This provides a natural mechanism for controlling the radial frequency location of the filters.

Importantly, the use of approximately linearly spaced radial frequencies is not proposed as a replacement for dyadic scaling. Dyadic scaling is fundamental to much of wavelet theory and underlies important theoretical properties such as Lipschitz continuity and stability under difeomorphisms. These properties have not been established for the present construction. Rather, the motivation here is to propose an alternative allowing to explore whether a diferent frequency organisation can be useful in settings where approximately uniform representation of radial frequencies is desirable. In particular, this may be relevant to reconstruction oriented tasks, where uniform frequency coverage can be preferable to deliberately allocating greater representation to lower frequencies.

Scope of these notes. The purpose of this manuscript is to provide a detailed and selfcontained derivation of the Fourier-Bessel wavelet construction introduced here. Rather than presenting a complete empirical evaluation, we focus on the mathematical motivation, construction, normalisation, and Fourier-domain representation of the proposed wavelets. We also provide a small number of numerical examples illustrating their frequency-domain behaviour and frame like coverage.

Contributions. The main novel mathematical components developed in these notes are:

1. the construction of a Gaussian-windowed Fourier-Bessel wavelet family together with the zero-mean correction required for the $m = 0$ mode.

2. closed-form expressions for the $L ^ { 1 }$ and $L ^ { 2 }$ normalisation constants and the corresponding frequency-domain peak normalisation.

3. a closed form Fourier domain representation of the resulting wavelets.

4. preliminary numerical evidence that the linear eigenvalue spacing can provide more uniform frequency coverage than Solid Harmonics.

To accompany the mathematical development, we have implemented the construction in the Python library (fbscatnet<sup>1</sup>). The library is intended to make the derivations and figures in these notes reproducible and provides functionality for using the wavelets within a scattering network, in a similar manner to Kymatio [2].

## 2 Mathematical Background

## 2.1 Bessel Functions

Bessel functions arise naturally when solving the radial component of the Helmholtz equation in polar coordinates. The Bessel equation of order m is

$$
x ^ { 2 } { \frac { d ^ { 2 } y } { d x ^ { 2 } } } + x { \frac { d y } { d x } } + ( x ^ { 2 } - m ^ { 2 } ) y = 0 .\tag{1}
$$

Because this is a second-order ordinary diferential equation, it has two linearly independent solutions. These are conventionally called the Bessel functions of the first and second kind.

## 2.1.1 Bessel Functions of the First and Second Kind

The Bessel functions of the first and second kind, $J _ { m }$ and $Y _ { m }$ , respectively, form the standard pair of solutions. We use $J _ { m }$ because it is regular at the origin for non-negative integer orders, whereas $Y _ { m }$ is singular there.

The first-kind Bessel function has the series representation

$$
J _ { m } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } } { n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m } .\tag{2}
$$

Here Γ denotes the Gamma function.

For suficiently large $x ,$ direct evaluation of the series can become numerically ineficient. We therefore also use the asymptotic form

$$
J _ { m } ( x ) = \sqrt { \frac { 2 } { \pi x } } \left[ \cos \left( x - \frac { m \pi } { 2 } - \frac { \pi } { 4 } \right) + \mathcal { O } ( | x | ^ { - 1 } ) \right] .\tag{3}
$$

For integer m we will also use

$$
J _ { - m } ( x ) = ( - 1 ) ^ { m } J _ { m } ( x ) .\tag{4}
$$

![](images/9efc7ddbbc448c8b1a13fb4ca0e5429ecb1b5e10b5cc9418533ac2e9165c8750.jpg)  
Figure 1: First-kind Bessel function $J _ { m }$ evaluated at orders $m = 0 , 1 , 2$ . The numerical implementation switches to the asymptotic approximation in Equation (3) for suficiently large x.

## 2.1.2 Modified Bessel Functions

Modified Bessel functions arise naturally in the Weber exponential integrals used later in the construction. The modified Bessel function of the first kind is

$$
I _ { m } ( x ) = i ^ { - m } J _ { m } ( i x ) = \sum _ { n = 0 } ^ { \infty } \frac { 1 } { n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m } .\tag{5}
$$

Remark 2.1. For higher orders, when combined with an exponential factor, $I _ { m }$ can grow large enough to cause numerical overflow. To avoid this we work with the exponentially scaled form.

$$
\widehat { I } _ { m } ( x ) = I _ { m } ( x ) \cdot e ^ { - | x | }\tag{6}
$$

$$
I _ { m } ( x ) = e ^ { x } \cdot \widehat { I } _ { m } ( x )\tag{7}
$$

![](images/18b56a9dfdf4731784c8b8a5f2e54536bf374a0f59fd723f6513391cc2c01911.jpg)  
Figure 2: Modified first-kind Bessel function $I _ { m }$ evaluated at orders $m = 0 , 1 , 2$

## 2.1.3 Derivative Identities for Bessel Functions

The derivative of the first-kind Bessel function is required when imposing the Neumann boundary condition.

Lemma 1 (Derivative identity for $J _ { m } )$ . For integer m,

$$
J _ { m } ^ { \prime } ( x ) = { \frac { 1 } { 2 } } \left[ J _ { m - 1 } ( x ) - J _ { m + 1 } ( x ) \right] .\tag{8}
$$

Proof. Starting from the series representation of the Bessel function of the first kind,

$$
J _ { m } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } } { n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m } ,\tag{9}
$$

we diferentiate term by term. Since only $\left( { \frac { x } { 2 } } \right) ^ { 2 n + m }$ depends on $x ,$ the chain rule gives

$$
{ \frac { d } { d x } } \left[ \left( { \frac { x } { 2 } } \right) ^ { 2 n + m } \right] = ( 2 n + m ) \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } { \frac { 1 } { 2 } } .\tag{10}
$$

Therefore,

$$
J _ { m } ^ { \prime } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } ( 2 n + m ) } { 2 n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m - 1 } .\tag{11}
$$

We now derive two equivalent expressions for $J _ { m } ^ { \prime } ( x )$ . The first is obtained by splitting the factor $( 2 n + m )$ as

$$
2 n + m = 2 ( n + m ) - m .
$$

Substituting this into (11) gives

$$
J _ { m } ^ { \prime } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } [ 2 ( n + m ) - m ] } { 2 n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m - 1 } .\tag{12}
$$

We distribute the sum into two separate series, $S _ { A }$ and $S _ { B }$

$$
J _ { m } ^ { \prime } ( x ) = S _ { A } ( x ) + S _ { B } ( x ) ,\tag{13}
$$

where

$$
S _ { A } ( x ) = \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } 2 ( n + m ) } { 2 n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } ,\tag{14}
$$

$$
S _ { B } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } ( - m ) } { 2 n ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m - 1 } .\tag{15}
$$

For $S _ { A }$ , the factor of 2 cancels:

$$
S _ { A } ( x ) = \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } ( n + m ) } { n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } .\tag{16}
$$

Using the Gamma-function identity

$$
\Gamma ( z + 1 ) = z \Gamma ( z ) ,
$$

with $z = n + m ,$ , we have

$$
\Gamma ( n + m + 1 ) = ( n + m ) \Gamma ( n + m ) .
$$

Hence,

$$
S _ { A } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } } { n ! \Gamma ( n + m ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m - 1 } .\tag{17}
$$

Comparing this expression with the series definition of $J _ { m - 1 } ( x )$ ，

$$
J _ { m - 1 } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( - 1 ) ^ { n } } { n ! \Gamma ( n + ( m - 1 ) + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + ( m - 1 ) } ,
$$

we see that

$$
S _ { A } ( x ) = J _ { m - 1 } ( x ) .\tag{18}
$$

We now evaluate $S _ { B }$ . First, we rewrite the power of $x / 2$ so that it has the same exponent as the series representation of $J _ { m } ( x )$ :

$$
\left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } = \left( { \frac { x } { 2 } } \right) ^ { 2 n + m } \left( { \frac { x } { 2 } } \right) ^ { - 1 } = \left( { \frac { x } { 2 } } \right) ^ { 2 n + m } { \frac { 2 } { x } } .\tag{19}
$$

Therefore,

$$
S _ { B } ( x ) = \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } ( - m ) } { 2 n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m } { \frac { 2 } { x } }\tag{20}
$$

$$
= - { \frac { m } { x } } \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } } { n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m }\tag{21}
$$

$$
= - \frac { m } { x } J _ { m } ( x ) .\tag{22}
$$

Consequently, the first derivative identity is

$$
J _ { m } ^ { \prime } ( x ) = J _ { m - 1 } ( x ) - { \frac { m } { x } } J _ { m } ( x ) .\tag{23}
$$

We now derive a second expression for $J _ { m } ^ { \prime } ( x )$ . Returning to (11), we instead retain $( 2 n + m )$ in its original form and split it as

$$
2 n + m = m + 2 n .
$$

This gives two new series, $S _ { C }$ and $S _ { D }$ , such that

$$
J _ { m } ^ { \prime } ( x ) = S _ { C } ( x ) + S _ { D } ( x ) ,\tag{24}
$$

where

$$
S _ { C } ( x ) = \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } m } { 2 n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } ,\tag{25}
$$

$$
S _ { D } ( x ) = \sum _ { n = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { n } 2 n } { 2 n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } .\tag{26}
$$

The first of these sums is the negative of $S _ { B } .$ so

$$
S _ { C } ( x ) = \frac { m } { x } J _ { m } ( x ) .\tag{27}
$$

For the second series, the factor of 2 cancels. Furthermore, the $n = 0$ term vanishes because of the factor n. We may therefore begin the sum at $n = 1$ :

$$
S _ { D } ( x ) = \sum _ { n = 1 } ^ { \infty } { \frac { ( - 1 ) ^ { n } n } { n ! \Gamma ( n + m + 1 ) } } \left( { \frac { x } { 2 } } \right) ^ { 2 n + m - 1 } .\tag{28}
$$

Using

$$
n ! = n ( n - 1 ) ! ,
$$

the factor of n cancels, giving

$$
S _ { D } ( x ) = \sum _ { n = 1 } ^ { \infty } \frac { ( - 1 ) ^ { n } } { ( n - 1 ) ! \Gamma ( n + m + 1 ) } \left( \frac { x } { 2 } \right) ^ { 2 n + m - 1 } .\tag{29}
$$

To express this in the standard series form for a Bessel function, we now shift the summation index. Let

$$
k = n - 1 , \qquad \mathrm { s o ~ t h a t } \qquad n = k + 1 .
$$

Since n begins at 1, the new index k begins at 0. We transform each component of the summand individually.

First, the alternating sign becomes

$$
( - 1 ) ^ { n } = ( - 1 ) ^ { k + 1 } = ( - 1 ) ^ { k } ( - 1 ) = - ( - 1 ) ^ { k } .\tag{30}
$$

Second, the factorial becomes

$$
( n - 1 ) ! = ( ( k + 1 ) - 1 ) ! = k ! .\tag{31}
$$

Third, the Gamma-function argument transforms as

$$
\Gamma ( n + m + 1 ) = \Gamma ( ( k + 1 ) + m + 1 )
$$

$$
= \Gamma ( k + m + 2 )\tag{32}
$$

$$
= \Gamma ( k + ( m + 1 ) + 1 ) .\tag{33}
$$

(34)

Finally, the exponent of $x / 2$ becomes

$$
2 n + m - 1 = 2 ( k + 1 ) + m - 1\tag{35}
$$

$$
= 2 k + m + 1\tag{36}
$$

$$
= 2 k + ( m + 1 ) .\tag{37}
$$

Substituting all of these transformations into $S _ { D } ( x )$ gives

$$
S _ { D } ( x ) = - \sum _ { k = 0 } ^ { \infty } \frac { ( - 1 ) ^ { k } } { k ! \Gamma { \left( k + ( m + 1 ) + 1 \right) } } \left( \frac { x } { 2 } \right) ^ { 2 k + ( m + 1 ) }\tag{38}
$$

$$
= - J _ { m + 1 } ( x ) .\tag{39}
$$

Therefore,

$$
J _ { m } ^ { \prime } ( x ) = { \frac { m } { x } } J _ { m } ( x ) - J _ { m + 1 } ( x ) .\tag{40}
$$

Combining the two identities, (23) and (40), allows the terms involving m $\phantom { } _ { ! J _ { m } } ( x ) / x$ to cancel:

$$
2 J _ { m } ^ { \prime } ( x ) = \Bigl ( J _ { m - 1 } ( x ) - { \frac { m } { x } } J _ { m } ( x ) \Bigr ) + \Bigl ( { \frac { m } { x } } J _ { m } ( x ) - J _ { m + 1 } ( x ) \Bigr )\tag{41}
$$

$$
= J _ { m - 1 } ( x ) - J _ { m + 1 } ( x ) .\tag{42}
$$

Dividing by 2 yields the desired derivative identity,

$$
J _ { m } ^ { \prime } ( x ) = { \frac { 1 } { 2 } } \left[ J _ { m - 1 } ( x ) - J _ { m + 1 } ( x ) \right] .\tag{43}
$$

![](images/2a73495eff670126791156b7942c0d611a84c30b2fd0e99aa13786e90838af5b.jpg)  
Figure 3: First-kind Bessel derivative $J _ { m } ^ { \prime }$ evaluated at orders $m = 0 , 1 , 2$

## 2.2 Wavelet Properties

Before constructing the Fourier-Bessel wavelets, we introduce the properties used throughout the remainder of the notes.

Definition 2.1 (Zero-mean condition). For the wavelet family considered here, we impose the zero-mean condition

$$
\int _ { \mathbb { R } ^ { d } } \psi ( x ) d x = 0 .\tag{44}
$$

This ensures that the wavelet has no response to a spatially constant component.

We additionally impose an $L ^ { p }$ normalisation according to the intended application. An $L ^ { 2 }$ normalisation is appropriate when the total energy of the wavelet should remain constant. For frequency-domain peak consistency, we instead use a normalisation based on the radial Fourier response. The $L ^ { 1 }$ norm provides the useful bound established below.

Theorem 2.1 (Plancherel’s theorem). Under the one-dimensional Fourier-transform convention

$$
{ \widehat { f } } ( \omega ) = \int _ { \mathbb { R } } f ( t ) e ^ { - i \omega t } d t ,\tag{45}
$$

we have

$$
\left\| f \right\| _ { 2 } ^ { 2 } = { \frac { 1 } { 2 \pi } } \left\| { \widehat { f } } \right\| _ { 2 } ^ { 2 } .\tag{46}
$$

Proof. Let $f , g \in L ^ { 1 } ( \mathbb { R } ) \cap L ^ { 2 } ( \mathbb { R } )$ . Their inner product is

$$
\langle f , g \rangle = \int _ { \mathbb { R } } f ( t ) { \overline { { g ( t ) } } } d t .\tag{47}
$$

Using the inverse Fourier transform,

$$
f ( t ) = { \frac { 1 } { 2 \pi } } \int _ { \mathbb { R } } { \hat { f } } ( \omega ) e ^ { i \omega t } d \omega ,\tag{48}
$$

we substitute into Equation (47) and, invoking Fubini’s theorem to exchange the order of integration (justified since $f , g \in L ^ { 1 } \cap L ^ { 2 } )$ , obtain

$$
\langle f , g \rangle = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } \hat { f } ( \omega ) \left( \int _ { \mathbb { R } } \overline { { g ( t ) } } e ^ { i \omega t } d t \right) d \omega .\tag{49}
$$

We now identify the bracketed integral. By definition Equation (45), the Fourier transform of $g$ is $\begin{array} { r } { \hat { g } ( \omega ) = \int _ { \mathbb { R } } g ( t ) e ^ { - i \omega t } d t } \end{array}$ , so its complex conjugate is

$$
\overline { { \hat { g } ( \omega ) } } = \overline { { \int _ { \mathbb { R } } g ( t ) e ^ { - i \omega t } d t } } = \int _ { \mathbb { R } } \overline { { g ( t ) } } e ^ { i \omega t } d t ,\tag{50}
$$

which is exactly the bracketed term in Equation (49). Substituting this identification gives

$$
\langle f , g \rangle = { \frac { 1 } { 2 \pi } } \int _ { \mathbb { R } } { \hat { f } } ( \omega ) { \overline { { { \hat { g } } ( \omega ) } } } d \omega .\tag{51}
$$

Setting $g = f$ gives $\langle f , f \rangle = \| f \| _ { 2 } ^ { 2 }$ on the left and $\textstyle { \frac { 1 } { 2 \pi } } \| { \hat { f } } \| _ { 2 } ^ { 2 }$ on the right, which is Equation (46).

Proposition 2.1 $( L ^ { 1 } – L ^ { \infty }$ Fourier bound). For $f \in L ^ { 1 } ( \mathbb { R } ^ { d } )$ ，

$$
\left\| { \widehat { f } } \right\| _ { \infty } \leq \left\| f \right\| _ { 1 } .\tag{52}
$$

Proof. For every frequency $\omega ,$

$$
\left| { \widehat { f } } ( \omega ) \right| = \left| \int _ { \mathbb { R } ^ { d } } f ( x ) e ^ { - i \omega \cdot x } d x \right|\tag{53}
$$

$$
\leq \int _ { \mathbb { R } ^ { d } } \left| f ( x ) \right| \left| e ^ { - i \omega \cdot x } \right| d x\tag{54}
$$

$$
= \int _ { \mathbb R ^ { d } } \left. f ( x ) \right. d x = \left\| f \right\| _ { 1 } .\tag{55}
$$

Taking the supremum over ω proves Equation (52).

## 3 Fourier-Bessel Disk Harmonics

We now construct the Fourier-Bessel basis on the planar unit disk

$$
\mathbb { D } = \left\{ ( \rho , \varphi ) : 0 \leq \rho \leq 1 , 0 \leq \varphi < 2 \pi \right\} .\tag{56}
$$

Fourier-Bessel basis functions arise as eigenfunctions of the Laplacian on the disk. Equivalently, they solve the Helmholtz equation subject to a boundary condition at $\rho = 1$

## 3.1 The Helmholtz Equation

We consider

$$
\frac { \partial ^ { 2 } f } { \partial \rho ^ { 2 } } + \frac { 1 } { \rho } \frac { \partial f } { \partial \rho } + \frac { 1 } { \rho ^ { 2 } } \frac { \partial ^ { 2 } f } { \partial \varphi ^ { 2 } } = - \lambda ^ { 2 } f .\tag{57}
$$

We seek separable solutions of the form

$$
f ( \rho , \varphi ) = R ( \rho ) \Phi ( \varphi ) .\tag{58}
$$

Substituting into Equation (57) gives

$$
\Phi \frac { d ^ { 2 } R } { d \rho ^ { 2 } } + \frac { \Phi } { \rho } \frac { d R } { d \rho } + \frac { R } { \rho ^ { 2 } } \frac { d ^ { 2 } \Phi } { d \varphi ^ { 2 } } = - \lambda ^ { 2 } R \Phi .\tag{59}
$$

Dividing both sides by RΦ isolates the radial and angular dependence:

$$
\frac { 1 } { R } \frac { d ^ { 2 } R } { d \rho ^ { 2 } } + \frac { 1 } { \rho R } \frac { d R } { d \rho } + \frac { 1 } { \rho ^ { 2 } \Phi } \frac { d ^ { 2 } \Phi } { d \varphi ^ { 2 } } = - \lambda ^ { 2 } .\tag{60}
$$

Multiplying through by $\rho ^ { 2 }$ and collecting the angular term on one side gives

$$
\rho ^ { 2 } \frac { R ^ { \prime \prime } } { R } + \rho \frac { R ^ { \prime } } { R } + \lambda ^ { 2 } \rho ^ { 2 } = - \frac { \Phi ^ { \prime \prime } } { \Phi } .\tag{61}
$$

The left-hand side depends only on ρ and the right-hand side only on $\varphi .$ . Both must therefore equal a common constant, which we write as $m ^ { 2 }$ :

$$
- { \frac { \Phi ^ { \prime \prime } } { \Phi } } = { \frac { \rho ^ { 2 } R ^ { \prime \prime } + \rho R ^ { \prime } + \lambda ^ { 2 } \rho ^ { 2 } R } { R } } = m ^ { 2 } .\tag{62}
$$

Remark 3.1. We fix the sign of the separation constant as $+ m ^ { 2 }$ rather than $- m ^ { 2 }$ . This is required for the angular equation to admit periodic (rather than exponentially growing/decaying) solutions, since 2π-periodicity of $\Phi$ is a physical requirement on the unit disk. The choice of sign is verified immediately below.

The angular equation is therefore

$$
\Phi ^ { \prime \prime } + m ^ { 2 } \Phi = 0 .\tag{63}
$$

Imposing 2π-periodicity gives

$$
\Phi _ { m } ( \varphi ) = e ^ { i m \varphi } , \qquad m \in \mathbb { Z } .\tag{64}
$$

The radial equation is

$$
\rho ^ { 2 } R ^ { \prime \prime } + \rho R ^ { \prime } + ( \lambda ^ { 2 } \rho ^ { 2 } - m ^ { 2 } ) R = 0 ,\tag{65}
$$

which is Bessel’s equation under the change of variable $x = \lambda \rho$ . The solution regular at the origin is therefore

$$
R _ { m , k } ( \rho ) = J _ { m } ( \lambda _ { m , k } \rho ) .\tag{66}
$$

## 3.2 Neumann Boundary Condition

To obtain the disk harmonics used here, we impose the Neumann condition

$$
\left. \frac { \partial R _ { m , k } } { \partial \rho } \right| _ { \rho = 1 } = 0 .\tag{67}
$$

Since

$$
\frac { \partial } { \partial \rho } J _ { m } ( \lambda _ { m , k } \rho ) = \lambda _ { m , k } J _ { m } ^ { \prime } ( \lambda _ { m , k } \rho ) ,\tag{68}
$$

the boundary condition is equivalent to

$$
J _ { m } ^ { \prime } ( \lambda _ { m , k } ) = 0 .\tag{69}
$$

Proposition 3.1 (Neumann eigenvalues). For each angular order $m _ {  }$ , the radial eigenvalues are given by the positive roots $\lambda _ { m , k }$ of Equation (69).

Definition 3.1 (Fourier-Bessel disk harmonic). For angular order m and root index $k ,$ define

$$
D _ { m , k } ( \rho , \varphi ) = N _ { m , k } J _ { m } ( \lambda _ { m , k } \rho ) e ^ { i m \varphi } ,\tag{70}
$$

where $N _ { m , k }$ is chosen according to the desired basis normalisation. For the orthonormal disk basis used in [9], the normalisation is

$$
N _ { m , k } = \frac { J _ { m } ( \lambda _ { m , k } ) ^ { - 1 } } { \sqrt { \pi ( 1 - \frac { m ^ { 2 } } { \lambda _ { m , k } ^ { 2 } } ) } }\tag{71}
$$

Remark 3.2. The distinction between the root index k and the eigenvalue $\lambda _ { m , k }$ is important throughout the construction. In particular, k is a discrete index, whereas $\lambda _ { m , k }$ determines the radial oscillation frequency.

## 3.3 Root Finding Using Muller’s Method

The eigenvalues are obtained by locating the roots of $J _ { m } ^ { \prime }$ . Following [9], we use Muller’s method. Given three starting estimates, the method iteratively fits a quadratic parabola and uses one of its roots as the next approximation. It can therefore be viewed as a higher-order extension of the secant method.

For initial estimates, we use McMahon’s asymptotic expansion [1, 8]. Keeping the first correction term gives

$$
\lambda _ { m , k } \approx \beta _ { m , k } - \frac { 4 m ^ { 2 } + 3 } { 8 \beta _ { m , k } } ,\tag{72}
$$

where

$$
\beta _ { m , k } = \left( k + \frac { m } { 2 } - \frac { 3 } { 4 } \right) \pi .\tag{73}
$$

The leading term in Equation (72) is linear in the root index k with slope $\pi .$ . This explains why the spacing between consecutive eigenvalues approaches π asymptotically. The correction term also shows why larger angular orders require larger k before this limiting spacing becomes apparent.

Remark 3.3. The asymptotic expansion is used here primarily to initialise numerical root finding and to interpret the frequency spacing. The numerical eigenvalues used in the wavelet construction are obtained from the roots of $J _ { m } ^ { \prime }$ rather than from the asymptotic approximation alone.

![](images/3edd50099ea569003c8bb987e7a63cef20f6ede3e54f35fe1409fd393d3fb9b1.jpg)  
Figure 4: Reproduction of [9]: eigenvalues associated with the roots of $J _ { m } ^ { \prime }$ for diferent angular orders and root indices.

## 4 Construction of Fourier-Bessel Wavelets

Having established the Fourier-Bessel basis on the unit disk, we now extend it to a wavelet family defined on the continuous plane. We use a Gaussian envelope to localise the oscillatory basis function in space. For notational simplicity, within this section we sometimes write $\lambda = \lambda _ { m , k }$ when the indices are unambiguous.

Definition 4.1 (Fourier-Bessel wavelet). The spatial Fourier-Bessel wavelet of angular order m and root index k is

$$
\psi _ { m , k } ( \rho , \varphi ) = N _ { m , k } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \left[ J _ { m } ( \lambda _ { m , k } \rho ) - K _ { m , k } \right] e ^ { i m \varphi } ,\tag{74}
$$

where $K _ { m , k }$ enforces the zero-mean condition and $N _ { m , k }$ denotes the application-specific normali sation.

Remark 4.1 (Radial and angular control). Unlike an afine wavelet construction in which a mother wavelet is repeatedly scaled and rotated, the present family keeps the Gaussian envelope fixed while using $\lambda _ { m , k }$ to control radial oscillation. Angular selectivity is provided directly by the factor $e ^ { i m \varphi }$

![](images/bb2480860da7024672f48b3276177918758f4a795ce6640cdcbe570b573a7f5c.jpg)  
Figure 5: Fourier-Bessel wavelet bank for $m , k = 5$ . Only the real components are shown. The imaginary components are obtained by $\mathrm { ~ a ~ } \pi / ( 2 m )$ angular phase shift. Here $\sigma = 1$

![](images/23410e146b39a8ae7fb024187ed522e0e973d0e893d215ad5a885b82baf07e4e.jpg)  
Figure 6: 3D rendering of the real and imaginary parts of the spatial Fourier-Bessel wavelet with $m = 1$ and $k = 2$

## 4.1 Low-Pass Filter

To cover the remaining low frequencies, we introduce a Gaussian low-pass filter,

$$
\phi ( \rho ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } } \exp \left( - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{75}
$$

Under the Fourier-transform convention used later, its frequency-domain representation is

$$
\widehat { \phi } ( q ) = \exp \left( - \frac { \sigma ^ { 2 } q ^ { 2 } } { 2 } \right) .\tag{76}
$$

## 4.2 Zero-Mean Correction

For $m \geq 1$ , the angular factor satisfies

$$
\int _ { 0 } ^ { 2 \pi } e ^ { i m \varphi } d \varphi = 0 ,\tag{77}
$$

so the zero-mean condition is automatically satisfied. The case $m = 0$ requires an explicit correction.

Proposition 4.1 (Zero-mean correction for $m = 0 )$ . For $m = 0$ , the unique constant $K _ { 0 , k }$ satisfying

$$
\int _ { \mathbb { R } ^ { 2 } } \psi _ { 0 , k } ( x ) d x = 0\tag{78}
$$

is

$$
K _ { 0 , k } = \exp \left( - \frac { \lambda _ { 0 , k } ^ { 2 } \sigma ^ { 2 } } { 2 } \right) .\tag{79}
$$

Proof. The zero-mean condition reduces to

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \left[ J _ { 0 } ( \lambda \rho ) - K \right] \rho d \rho = 0 .\tag{80}
$$

Separating the two terms gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } J _ { 0 } ( \lambda \rho ) \rho d \rho = K \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \rho d \rho .\tag{81}
$$

The first integral is Weber’s first exponential integral [11],

$$
\int _ { 0 } ^ { \infty } e ^ { - t ^ { 2 } \rho ^ { 2 } } J _ { 0 } ( \lambda \rho ) \rho d \rho = \frac { 1 } { 2 t ^ { 2 } } \exp \left( - \frac { \lambda ^ { 2 } } { 4 t ^ { 2 } } \right) .\tag{82}
$$

Setting $t ^ { 2 } = 1 / ( 2 \sigma ^ { 2 } )$ gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } J _ { 0 } ( \lambda \rho ) \rho d \rho = \sigma ^ { 2 } e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } .\tag{83}
$$

For the second integral, the substitution $u = \rho ^ { 2 } / ( 2 \sigma ^ { 2 } )$ gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \rho d \rho = \sigma ^ { 2 } .\tag{84}
$$

Therefore,

$$
\sigma ^ { 2 } e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } - K \sigma ^ { 2 } = 0 ,\tag{85}
$$

and hence Equation (79).

## 4.3 $L ^ { 2 }$ Normalisation

We now choose the normalisation constant so that

$$
\| \psi _ { m , k } \| _ { 2 } = 1 .\tag{86}
$$

The $m = 0$ and $m \geq 1$ cases difer because the zero-mean correction is present only for $m = 0$

## 4.3.1 Angular orders $m \geq 1$

For $m \geq 1$ , define

$$
N _ { m , k } ^ { ( 2 ) } = \left[ \pi \sigma ^ { 2 } e ^ { - \lambda _ { m , k } ^ { 2 } \sigma ^ { 2 } / 2 } I _ { m } \left( \frac { \lambda _ { m , k } ^ { 2 } \sigma ^ { 2 } } { 2 } \right) \right] ^ { - 1 / 2 } .\tag{87}
$$

Proposition 4.2 $( L ^ { 2 }$ normalisation for $m \geq 1 )$ . The constant in Equation (87) satisfies $\| \psi _ { m , k } \| _ { 2 } = 1$ for m $\geq 1$

Proof. For $m \geq 1 , K _ { m , k } = 0$ , so

$$
1 = 2 \pi \left( N _ { m , k } ^ { ( 2 ) } \right) ^ { 2 } \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } J _ { m } ( \lambda \rho ) ^ { 2 } \rho d \rho .\tag{88}
$$

We use Weber’s second exponential integral [11],

$$
\int _ { 0 } ^ { \infty } e ^ { - t ^ { 2 } \rho ^ { 2 } } J _ { m } ( \lambda _ { 1 } \rho ) J _ { m } ( \lambda _ { 2 } \rho ) \rho d \rho = { \frac { 1 } { 2 t ^ { 2 } } } e ^ { - ( \lambda _ { 1 } ^ { 2 } + \lambda _ { 2 } ^ { 2 } ) / ( 4 t ^ { 2 } ) } I _ { m } \left( { \frac { \lambda _ { 1 } \lambda _ { 2 } } { 2 t ^ { 2 } } } \right) .\tag{89}
$$

Setting $\lambda _ { 1 } = \lambda _ { 2 } = \lambda$ and $t ^ { 2 } = 1 / \sigma ^ { 2 }$ gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } J _ { m } ( \lambda \rho ) ^ { 2 } \rho d \rho = \frac { \sigma ^ { 2 } } 2 e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } I _ { m } \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } } 2 \right) .\tag{90}
$$

Substitution into the norm condition yields

$$
\left( N _ { m , k } ^ { ( 2 ) } \right) ^ { 2 } \pi \sigma ^ { 2 } e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } I _ { m } \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 } \right) = 1 ,\tag{91}
$$

which gives Equation (87).

## 4.3.2 Zeroth angular order $m = 0$

For $m = 0$ , the zero-mean correction contributes to the norm. Define

$$
N _ { 0 , k } ^ { ( 2 ) } = \left[ \pi \sigma ^ { 2 } \left( e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } I _ { 0 } \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 } \right) - 2 e ^ { - 3 \sigma ^ { 2 } \lambda ^ { 2 } / 4 } + e ^ { - \sigma ^ { 2 } \lambda ^ { 2 } } \right) \right] ^ { - 1 / 2 } .\tag{92}
$$

Proposition 4.3 $( L ^ { 2 }$ normalisation for $m = 0 )$ . The constant in Equation (92) satisfies $\| \psi _ { 0 , k } \| _ { 2 } = 1$

Proof. Expanding the squared correction gives

$$
1 = 2 \pi \left( N _ { 0 , k } ^ { ( 2 ) } \right) ^ { 2 } \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } \left[ J _ { 0 } ( \lambda \rho ) - K \right] ^ { 2 } \rho d \rho\tag{93}
$$

$$
= 2 \pi \left( N _ { 0 , k } ^ { \left( 2 \right) } \right) ^ { 2 } \left( I _ { A } + I _ { B } + I _ { C } \right) ,\tag{94}
$$

where

$$
I _ { A } = \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } J _ { 0 } ( \lambda \rho ) ^ { 2 } \rho d \rho ,\tag{95}
$$

$$
I _ { B } = - 2 K \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } J _ { 0 } ( \lambda \rho ) \rho d \rho ,\tag{96}
$$

$$
I _ { C } = K ^ { 2 } \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / \sigma ^ { 2 } } \rho d \rho .\tag{97}
$$

By Equation (89),

$$
I _ { A } = \frac { \sigma ^ { 2 } } { 2 } e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 2 } I _ { 0 } \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 } \right) .\tag{98}
$$

Using Equation (82) with $t ^ { 2 } = 1 / \sigma ^ { 2 }$

$$
I _ { B } = - K \sigma ^ { 2 } e ^ { - \lambda ^ { 2 } \sigma ^ { 2 } / 4 } = - \sigma ^ { 2 } e ^ { - 3 \sigma ^ { 2 } \lambda ^ { 2 } / 4 } ,\tag{99}
$$

where we used Equation (79). Finally,

$$
I _ { C } = \frac { K ^ { 2 } \sigma ^ { 2 } } { 2 } = \frac { \sigma ^ { 2 } } { 2 } e ^ { - \sigma ^ { 2 } \lambda ^ { 2 } } .\tag{100}
$$

Combining the three terms gives Equation (92).

## 4.4 Fourier-Domain Peak Normalisation

The $L ^ { 1 } – L ^ { \infty }$ bound in Proposition 2.1 motivates an $L ^ { 1 } { \mathrm { - b a s e d } }$ normalisation when frequencydomain amplitude consistency is desired. In the implementation, however, we directly normalise each wavelet by the maximum of its radial Fourier response.

Definition 4.2 (Peak normalisation). Let $R _ { m , k } ( q )$ denote the radial component of the Fourierdomain wavelet. We define

$$
N _ { m , k } ^ { ( 1 ) } = \frac { 1 } { \underset { q \geq 0 } { \operatorname* { m a x } } | R _ { m , k } ( q ) | } .\tag{101}
$$

The maximum can be found numerically on the finite frequency grid used in the implementation. We nevertheless derive the corresponding stationary-point equations below.

The modified Bessel function satisfies

$$
I _ { m } ^ { \prime } ( x ) = I _ { m - 1 } ( x ) - { \frac { m } { x } } I _ { m } ( x )\tag{102}
$$

$$
= \frac { m } { x } I _ { m } ( x ) + I _ { m + 1 } ( x ) ,\tag{103}
$$

and hence

$$
I _ { m } ^ { \prime } ( x ) = \frac { 1 } { 2 } \left[ I _ { m - 1 } ( x ) + I _ { m + 1 } ( x ) \right] .\tag{104}
$$

## 4.4.1 Angular orders m $\geq 1$

For $m \geq 1$ , set

$$
x = \sigma ^ { 2 } \lambda q .\tag{105}
$$

The radial response is

$$
R _ { m , k } ( q ) = \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } ( \lambda ^ { 2 } + q ^ { 2 } ) / 2 } I _ { m } ( x ) .\tag{106}
$$

Diferentiating and setting the derivative to zero gives

$$
0 = \frac { d R _ { m , k } } { d q }\tag{107}
$$

$$
= \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } ( \lambda ^ { 2 } + q ^ { 2 } ) / 2 } \left[ - \sigma ^ { 2 } q I _ { m } ( x ) + \sigma ^ { 2 } \lambda I _ { m } ^ { \prime } ( x ) \right] .\tag{108}
$$

Since the prefactors are non-zero,

$$
\lambda I _ { m } ^ { \prime } ( x ) = q I _ { m } ( x ) .\tag{109}
$$

Using the first derivative identity for $I _ { m }$ gives

$$
\lambda I _ { m - 1 } ( x ) = I _ { m } ( x ) \left( q + \frac { m } { \sigma ^ { 2 } q } \right) .\tag{110}
$$

In general, Equation (110) does not admit a closed-form solution for $q .$ The peak frequency $q _ { m , k } ^ { * }$ is therefore found numerically.

## 4.4.2 Zeroth angular order $m = 0$

For $m = 0$ , the correction term $\mathrm { g i }$ ves

$$
R _ { 0 , k } ( q ) = \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } ( \lambda ^ { 2 } + q ^ { 2 } ) / 2 } \left[ I _ { 0 } ( x ) - 1 \right] .\tag{111}
$$

Diferentiating gives the stationary-point equation

$$
\lambda I _ { 0 } ^ { \prime } ( x ) = q [ I _ { 0 } ( x ) - 1 ] .\tag{112}
$$

Since $I _ { 0 } ^ { \prime } ( x ) = I _ { 1 } ( x )$ 2

$$
\lambda I _ { 1 } ( x ) = q \left[ I _ { 0 } ( x ) - 1 \right] .\tag{113}
$$

Again, the peak frequency $q _ { 0 , k } ^ { * }$ is obtained numerically.

## 4.4.3 Asymptotic case

When $\sigma ^ { 2 } \lambda ^ { 2 } \gg 1$ , the large-argument asymptotic behaviour of the modified Bessel function gives

$$
I _ { m } ^ { \prime } ( x ) \approx I _ { m } ( x ) .\tag{114}
$$

Consequently, Equation (109) gives

$$
q _ { m , k } ^ { * } \approx \lambda _ { m , k } .\tag{115}
$$

Evaluating the radial response at this approximate maximum yields

$$
N _ { m , k } ^ { ( 1 ) } \approx \frac { 1 } { \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } \lambda _ { m , k } ^ { 2 } } I _ { m } ( \sigma ^ { 2 } \lambda _ { m , k } ^ { 2 } ) } , \qquad m \ge 1 ,\tag{116}
$$

$$
N _ { 0 , k } ^ { ( 1 ) } \approx \frac { 1 } { \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } \lambda _ { 0 , k } ^ { 2 } } \left[ I _ { 0 } ( \sigma ^ { 2 } \lambda _ { 0 , k } ^ { 2 } ) - 1 \right] } .\tag{117}
$$

## 5 Fourier-Domain Representation

The previous sections constructed the wavelets in the spatial domain. We now derive their closed-form Fourier representation. This form is useful both for analysing frequency coverage and for implementing the filters without explicitly computing a numerical Fourier transform.

We use the two-dimensional Fourier-transform convention

$$
\widehat { f } ( k _ { x } , k _ { y } ) = \int _ { \mathbb { R } ^ { 2 } } f ( x , y ) e ^ { - i ( k _ { x } x + k _ { y } y ) } d x d y .\tag{118}
$$

Writing the spatial and frequency coordinates in polar form,

$$
x = \rho \cos \varphi , \qquad y = \rho \sin \varphi ,\tag{119}
$$

and

$$
k _ { x } = q \cos \phi , \qquad k _ { y } = q \sin \phi ,\tag{120}
$$

we have

$$
k _ { x } x + k _ { y } y = q \rho \cos ( \varphi - \phi ) .\tag{121}
$$

Lemma 2 (Angular Fourier-Bessel integral). For integer m,

$$
\int _ { 0 } ^ { 2 \pi } e ^ { i m \varphi } e ^ { - i q \rho \cos ( \varphi - \phi ) } d \varphi = 2 \pi ( - i ) ^ { m } e ^ { i m \phi } J _ { m } ( q \rho ) .\tag{122}
$$

Proof. Set $\theta = \varphi - \phi$ . Then

$$
\int _ { 0 } ^ { 2 \pi } e ^ { i m \varphi } e ^ { - i q \rho \cos ( \varphi - \phi ) } d \varphi = e ^ { i m \phi } \int _ { 0 } ^ { 2 \pi } e ^ { i m \theta } e ^ { - i q \rho \cos \theta } d \theta .\tag{123}
$$

Using the Jacobi-Anger expansion

$$
e ^ { - i z \cos \theta } = \sum _ { n = - \infty } ^ { \infty } ( - i ) ^ { n } J _ { n } ( z ) e ^ { i n \theta } ,\tag{124}
$$

we obtain

$$
e ^ { i m \phi } \sum _ { n = - \infty } ^ { \infty } ( - i ) ^ { n } J _ { n } ( q \rho ) \int _ { 0 } ^ { 2 \pi } e ^ { i ( m + n ) \theta } d \theta .\tag{125}
$$

Only the term $n = - m$ survives. Using $J _ { - m } ( x ) = ( - 1 ) ^ { m } J _ { m } ( x )$ and the identity $\begin{array} { r l } { ( - i ) ^ { - m } ( - 1 ) ^ { m } = } \end{array}$ $( - i ) ^ { m }$ , the integral evaluates to

$$
2 \pi ( - i ) ^ { m } e ^ { i m \phi } J _ { m } ( q \rho ) .\tag{126}
$$

Theorem 5.1 (Fourier representation of the Fourier-Bessel wavelet). Under the convention in Equation (118), the Fourier transform of Equation (74) is

$$
\widehat { \psi } _ { m , k } ( q , \phi ) = ( - i ) ^ { m } e ^ { i m \phi } N _ { m , k } \left[ \sigma ^ { 2 } e ^ { - \frac { \sigma ^ { 2 } } { 2 } ( \lambda _ { m , k } ^ { 2 } + q ^ { 2 } ) } I _ { m } ( \sigma ^ { 2 } \lambda _ { m , k } q ) - K _ { m , k } \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } q ^ { 2 } / 2 } \right] .\tag{127}
$$

Proof. Substituting the polar coordinates into the Fourier transform and using the Jacobian $\rho$ gives

$$
\widehat { \psi } _ { m , k } ( q , \phi ) = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { 2 \pi } N _ { m , k } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \left[ J _ { m } ( \lambda _ { m , k } \rho ) - K _ { m , k } \right] e ^ { i m \varphi } \cdot e ^ { - i q \rho \cos ( \varphi - \phi ) } \rho d \varphi d \rho .\tag{128}
$$

Applying Lemma 2 gives

$$
\widehat { \psi } _ { m , k } ( q , \phi ) = ( - i ) ^ { m } e ^ { i m \phi } N _ { m , k } \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \left[ J _ { m } ( \lambda _ { m , k } \rho ) - K _ { m , k } \right] J _ { m } ( q \rho ) \rho d \rho .\tag{129}
$$

The first radial term is Weber’s second exponential integral,

$$
\begin{array} { r } { \displaystyle \int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } J _ { m } ( \lambda _ { m , k } \rho ) J _ { m } ( q \rho ) \rho d \rho } \\ { = \sigma ^ { 2 } e ^ { - \frac { \sigma ^ { 2 } } { 2 } ( \lambda _ { m , k } ^ { 2 } + q ^ { 2 } ) } I _ { m } ( \sigma ^ { 2 } \lambda _ { m , k } q ) . } \end{array}\tag{130}
$$

For the correction term, the remaining radial integral is

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } J _ { m } ( q \rho ) \rho d \rho .\tag{131}
$$

For the $m = 0$ correction used in the present construction, Weber’s first exponential integral gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \rho ^ { 2 } / ( 2 \sigma ^ { 2 } ) } J _ { 0 } ( q \rho ) \rho d \rho = \sigma ^ { 2 } e ^ { - \sigma ^ { 2 } q ^ { 2 } / 2 } .\tag{132}
$$

Substituting the radial integrals gives Equation (127).

Remark 5.1 (Separation of angular and radial structure). The Fourier representation separates naturally into the angular factor

$$
( - i ) ^ { m } e ^ { i m \phi }\tag{133}
$$

and a radial response depending only on $q .$ Thus m controls angular selectivity, while $\lambda _ { m , k }$ controls the radial frequency location.

![](images/3e987f804bc8740930f60a4205be423ce43b0f78603bd42d132996632edd7855.jpg)  
Figure 7: Equivalent Fourier-domain filter bank.

![](images/7347d20a11adcf321da7913a789be4b4c65672d08f26dce5f0aade5cae65def8.jpg)  
Figure 8: 3D rendering of the real and imaginary parts of the Fourier-domain wavelet with $m = 1$ and $k = 4$

## 6 Frequency Tiling and Eigenvalue Spacing

The use of the Neumann eigenvalues as the radial frequency parameter is motivated by their approximately uniform spacing. From Equation (72),

$$
\lambda _ { m , k } \approx \beta _ { m , k } - \frac { 4 m ^ { 2 } + 3 } { 8 \beta _ { m , k } } ,\tag{134}
$$

with

$$
\beta _ { m , k } = \left( k + \frac { m } { 2 } - \frac { 3 } { 4 } \right) \pi .\tag{135}
$$

Therefore, the leading-order spacing is

$$
\lambda _ { m , k + 1 } - \lambda _ { m , k } \longrightarrow \pi \qquad \mathrm { a s } \ k  \infty .\tag{136}
$$

Table 2 illustrates this convergence for the first few angular orders and root indices. For higher angular orders, convergence to the asymptotic spacing is slower. This is consistent with the m-dependent correction term in Equation (72), whose numerator grows quadratically with m.

Definition 6.1 (Frame). A filter bank consisting of a low-pass filter $\phi$ and a family of wavelet filters $\psi$ forms a frame for the signal space if there exist constants $0 < A \le B < \infty$ such that:

$$
A \leq | \widehat { \phi } ( \omega ) | ^ { 2 } + \sum _ { i = 1 } ^ { \infty } | \widehat { \psi } _ { i } ( \omega ) | ^ { 2 } \leq B\tag{137}
$$

If $A = B$ , the filter bank constitutes a tight frame, ensuring energy conservation (Theorem 2.1) and numerically stable, perfect reconstruction. In practice we aim to minimise the ratio with $A \approx B$

<table><tr><td>m</td><td>k</td><td> $k + 1$ </td><td> $\lambda _ { k }$ </td><td> $\lambda _ { k + 1 }$ </td><td>Spacing</td><td>Error  $( \Delta \lambda - \pi )$ </td><td>Rel. Error (%)</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3.8317</td><td>7.0156</td><td>3.1839</td><td>+0.0423</td><td>+1.35</td></tr><tr><td>0</td><td>2</td><td>3</td><td>7.0156</td><td>10.1735</td><td>3.1579</td><td>+0.0163</td><td>+0.52</td></tr><tr><td>0</td><td>3</td><td>4</td><td>10.1735</td><td>13.3237</td><td>3.1502</td><td>+0.0086</td><td>+0.27</td></tr><tr><td>0</td><td>4</td><td>5</td><td>13.3237</td><td>16.4706</td><td>3.1469</td><td>+0.0053</td><td>+0.17</td></tr><tr><td>0</td><td>5</td><td>6</td><td>16.4706</td><td>19.6159</td><td>3.1452</td><td>+0.0036</td><td>+0.12</td></tr><tr><td>1</td><td>1</td><td>2</td><td>1.8412</td><td>5.3314</td><td>3.4903</td><td>+0.3487</td><td>+11.10</td></tr><tr><td>1</td><td>2</td><td>3</td><td>5.3314</td><td>8.5363</td><td>3.2049</td><td>+0.0633</td><td>+2.01</td></tr><tr><td>1</td><td>3</td><td>4</td><td>8.5363</td><td>11.7060</td><td>3.1697</td><td>+0.0281</td><td>+0.89</td></tr><tr><td>1</td><td>4</td><td>5</td><td>11.7060</td><td>14.8636</td><td>3.1576</td><td>+0.0160</td><td>+0.51</td></tr><tr><td>1</td><td>5</td><td>6</td><td>14.8636</td><td>18.0155</td><td>3.1519</td><td>+0.0103</td><td>+0.33</td></tr><tr><td>2</td><td>1</td><td>2</td><td>3.0542</td><td>6.7061</td><td>3.6519</td><td>+0.5103</td><td>+16.24</td></tr><tr><td>2</td><td>2</td><td>3</td><td>6.7061</td><td>9.9695</td><td>3.2633</td><td>+0.1217</td><td>+3.88</td></tr><tr><td>2</td><td>3</td><td>4</td><td>9.9695</td><td>13.1704</td><td>3.2009</td><td>+0.0593</td><td>+1.89</td></tr><tr><td>2</td><td>4</td><td>5</td><td>13.1704</td><td>16.3475</td><td>3.1772</td><td>+0.0356</td><td>+1.13</td></tr><tr><td>2</td><td>5</td><td>6</td><td>16.3475</td><td>19.5129</td><td>3.1654</td><td>+0.0238</td><td>+0.76</td></tr><tr><td>3</td><td>1</td><td>2</td><td>4.2012</td><td>8.0152</td><td>3.8140</td><td>+0.6725</td><td>+21.40</td></tr><tr><td>3</td><td>2</td><td>3</td><td>8.0152</td><td>11.3459</td><td>3.3307</td><td>+0.1891</td><td>+6.02</td></tr><tr><td>3</td><td>3</td><td>4</td><td>11.3459</td><td>14.5858</td><td>3.2399</td><td>+0.0983</td><td>+3.13</td></tr><tr><td>3</td><td>4</td><td>5</td><td>14.5858</td><td>17.7887</td><td>3.2029</td><td>+0.0613</td><td>+1.95</td></tr><tr><td>3</td><td>5</td><td>6</td><td>17.7887</td><td>20.9725</td><td>3.1837</td><td>+0.0421</td><td>+1.34</td></tr><tr><td>4</td><td>1</td><td>2</td><td>5.3176</td><td>9.2824</td><td>3.9648</td><td>+0.8233</td><td>+26.20</td></tr><tr><td>4</td><td>2</td><td>3</td><td>9.2824</td><td>12.6819</td><td>3.3995</td><td>+0.2579</td><td>+8.21</td></tr><tr><td>4</td><td>3</td><td>4</td><td>12.6819</td><td>15.9641</td><td>3.2822</td><td>+0.1406</td><td>+4.48</td></tr><tr><td>4</td><td>4</td><td>5</td><td>15.9641</td><td>19.1960</td><td>3.2319</td><td>+0.0903</td><td>+2.88</td></tr><tr><td>4</td><td>5</td><td>6</td><td>19.1960</td><td>22.4010</td><td>3.2050</td><td>+0.0634</td><td>+2.02</td></tr></table>

Table 2: Convergence of consecutive Neumann eigenvalue spacings toward $\pi$ for angular orders $m = 0 , \ldots , 4$

In Figure 9, we explore the frame bounds ratio $B / A$ to evaluate the wavelets. This experiment is not intended as a proof of frame bounds. It is included only to illustrate why proposed frequency organisation may merit further investigation. Note that the lower bound is evaluated at 0.75π to avoid the dividing by 0 when discrete wavelets naturally decay at the edge. Both wavelet families were evaluated on the same resolution, image size, variance and peak normalisation. Furthermore, while Solid Harmonics produce $J \times L$ wavelets, Fourier-Bessel wavelets, due to the pyramidal constraint, create $\textstyle \sum _ { m = 0 } ^ { M } K - m$ filters.

Across the tested parameter range, the Fourier-Bessel banks exhibit lower coverage ripple than the corresponding Solid Harmonic banks. This behaviour is consistent with the near linear spacing of the radial frequencies, which distributes the filters more uniformly. By contrast, the dyadic organisation of the Solid Harmonic filters places greater emphasis on lower frequencies and progressively wider spacing at higher frequencies. This diference should not be interpreted as evidence that linear spacing is preferable: the frequency weighting induced by dyadic scaling is an important feature of conventional wavelet constructions and can be desirable for tasks such as image classification, where greater emphasis on lower frequencies may contribute to robustness to small perturbations. Rather, the experiment suggests that linear frequency organisation is an interesting alternative when more uniform frequency representation is desired.

![](images/b155a57ee09b6e1583c4d75f42b53c71b0bc5d37a5caf65934b0f61998845b4d.jpg)  
(a) Frequency coverage sum for the Fourier-Bessel bank $m = 3 , k = 8 , \sigma = 1$ . The linear spacing is clearly visible with little overlap, achieving a ratio of 2.2530.

![](images/e43fe27c94c18f708f780264991167894cfbbe8d13c3af3cb54abe94dc3efe8d.jpg)  
(b) Frequency coverage sum for the Solid Harmonic bank $J = 3 , L = 5 , \sigma = 1$ . Here, dyadic scaling causes wider coverage gaps and overlaps (ratio of 6.2377 in this example).

![](images/6f878531d8154a646f5b4d52fee1cce1b1317688a24bdd7cac03b701eba4c550.jpg)  
(c) Fourier-Bessel frequency coverage parameter search. Ratio values remain consistent, naturally beginning to increase for larger orders m as the bank begins to cover frequencies beyond the Nyquist limit, raising the lower bound at the cut of.

![](images/6bbf70b6fdaea4c559d0af89709fa848027fca2a77043f9c292447da60a87151.jpg)  
(d) Solid Harmonic frequency coverage parameter search. Values follow a similar pattern, increasing as the bank exceeds the boundary, but the dyadic scaling causes a more rapid increase of the ratio.  
Figure 9: Frequency coverage comparison between Fourier-Bessel wavelets and Solid Harmonics. Top row: Frequency coverage sums for a single representative bank of each family, showing the linear-spacing tiling of the Fourier-Bessel bank (a) against the dyadic-scaling gaps and overlaps of the Solid Harmonic bank (b). Bottom row: grid search over bank parameters, showing the frame ripple ratio $B / A$ for the Fourier-Bessel bank (c) and the Solid Harmonic bank (d); the Fourier-Bessel ratios stay consistently lower and grow more slowly than the Solid Harmonic ratios across the tested parameter range.

## 7 Conclusion

These notes have developed the mathematical foundations and construction of Fourier-Bessel wavelets. Starting from the Bessel diferential equation, we derived the Fourier-Bessel disk harmonics as solutions of the Helmholtz equation subject to a Neumann boundary condition. The resulting eigenvalues provide a natural radial frequency parameter whose asymptotic spacing approaches π.

We then constructed a wavelet family by applying a Gaussian spatial envelope to the Fourier-Bessel basis and introducing a zero-mean correction for the zeroth angular order. The corresponding $L ^ { 2 }$ normalisation constants were derived using Weber’s exponential integrals, while a peak normalisation based on the radial Fourier response was developed for applications requiring consistent frequency-domain amplitudes.

Finally, we derived a closed-form Fourier-domain representation of the wavelets. This representation separates naturally into angular and radial components and provides a direct description of the frequency response of each wavelet.

The preliminary frequency coverage provides evidence consistent with the motivation for the construction: in the configurations tested, the Fourier-Bessel bank exhibits a flatter frequency coverage profile relative to the corresponding Solid Harmonic bank. This experiment is intentionally limited and are not intended to establish general performance improvements.

The approximately linear radial frequency spacing should therefore be viewed as a complementary alternative to dyadic scaling rather than a replacement for it. Dyadic scaling remains central to wavelet theory and provides important theoretical and practical properties that have not been established for the present construction. The motivation for the Fourier-Bessel approach is instead to explore a diferent allocation of frequency resolution, which may be advantageous in reconstruction oriented settings where approximately uniform frequency representation is desirable. Determining the classes of tasks for which either frequency organisation is preferable is an open question.

The purpose of these notes is primarily theoretical and pedagogical. They are intended to provide a detailed mathematical reference for the construction rather than to constitute a comprehensive empirical evaluation of the resulting wavelet family.

The accompanying fbscatnet library implements the construction described throughout these notes and reproduces the figures presented here. A natural next step is to evaluate the resulting wavelets empirically within scattering networks and to compare their performance with existing wavelet constructions across relevant downstream applications.

## References

[1] Milton Abramowitz and Irene A. Stegun. Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables. Vol. 55. Applied Mathematics Series. Washington, D.C.: National Bureau of Standards, 1964.

[2] Mathieu Andreux et al. Kymatio: Scattering Transforms in Python. 2022. arXiv: 1812. 11214 [cs.LG]. url: https://arxiv.org/abs/1812.11214.

[3] Joan Bruna and St´ephane Mallat. Invariant Scattering Convolution Networks. 2012. arXiv: 1203.1513 [cs.CV]. url: https://arxiv.org/abs/1203.1513.

[4] Ingrid Daubechies. Ten lectures on wavelets. USA: Society for Industrial and Applied Mathematics, 1992. isbn: 0898712742.

[5] Michael Eickenberg et al. “Solid harmonic wavelet scattering for predictions of molecule properties”. In: The Journal of Chemical Physics 148.24 (May 2018). issn: 1089-7690. doi: 10.1063/1.5023798. url: http://dx.doi.org/10.1063/1.5023798.

[6] S.G. Mallat. “A theory for multiresolution signal decomposition: the wavelet representation”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 11.7 (1989), pp. 674– 693. doi: 10.1109/34.192463.

[7] Stphane Mallat. A Wavelet Tour of Signal Processing, Third Edition: The Sparse Way. 3rd. USA: Academic Press, Inc., 2008. isbn: 0123743702.

[8] F. W. J. Olver et al. Digital Library of Mathematical Functions. National Institute of Standards and Technology (NIST). url: https://dlmf.nist.gov/.

[9] Mahmoud Shaqfa et al. “Disk harmonics for analysing curved and flat self-afine rough surfaces and the topological reconstruction of open surfaces”. In: Journal of Computational Physics 522 (Feb. 2025), p. 113578. issn: 0021-9991. doi: 10.1016/j.jcp.2024.113578. url: http://dx.doi.org/10.1016/j.jcp.2024.113578.

[10] Laurent SIfre and St´ephane Mallat. Rigid-Motion Scattering for Texture Classification. 2014. arXiv: 1403.1687 [cs.CV]. url: https://arxiv.org/abs/1403.1687.

[11] G.N. Watson. A Treatise on the Theory of Bessel Functions. Cambridge Mathematical Library. Cambridge University Press, 1995. isbn: 9780521483919. url: https://books. google.fr/books?id=Mlk3FrNoEVoC.