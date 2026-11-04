# Hybrid Quantum–Classical Pauli-Correlator Architecture for Image Reconstruction

Sparsho Chakraborty<sup>1</sup> and Siddhartha Patra<sup>2\*</sup>

<sup>1</sup>School of Basic Sciences, Department of Physics, Indian Institute of Technology Bhubaneswar, Odisha, India.

<sup>2\*</sup>Centre for Quantum Engineering, Research and Education, TCG CREST, Kolkata, India.

\*Corresponding author(s). E-mail(s): siddhartha.patra@tcgcrest.org; Contributing authors: 25ph05023@iitbbs.ac.in;

## Abstract

We introduce the Pauli-Correlator Autoencoder (PCAE), a hybrid quantum–classical architecture for image reconstruction. PCAE splits an image into patches and compresses each patch as a matrix product state (MPS). A small quantum circuit turns each compressed patch into a few measured numbers, and a classical neural network rebuilds the patch from these numbers and the patch position. A learnable energy term, taken from physics, is added to training and tracked as training proceeds. Two versions place the six qubits on a line (PCAE1D) or on a grid (PCAE2D). An averaging step makes the PCAE2D features follow rotations and reflections of the image exactly. Trained on one image at a time, PCAE2D rebuilds images from six public datasets with a peak signal-to-noise ratio of 25.8 to 41.6 decibels. On CIFAR-10 images not used in training, a formula-based stand-in for the PCAE2D features performs within 1 decibel of several simpler classical methods. Five trained models also run on real IBM quantum computers without retraining, and rebuild images at 25.90 to 31.52 decibels. A short argument and a synthetic test show that, with a linear final layer, measuring qubits in pairs is needed for targets built from products of inputs. These results show that PCAE works as a hybrid model that rebuilds varied images and runs on today’s quantum hardware at the six-qubit scale.

Keywords: Quantum image processing, Tensor networks, Variational quantum algorithms, Pauli observables, Group-equivariant representations, Quantum hardware pilot

## 1 Introduction

Hybrid quantum–classical models are now often proposed for vision tasks. The aim is to obtain representations with a favourable inductive bias. Such models can use entanglement, physically motivated regularisers, or tensor-network structure, which standard convolutional networks do not have (Cerezo et al. 2021; Tilly et al. 2022; Preskill 2018). The variational quantum eigensolver and its later variants use the fact that physical Hamiltonians split into local terms. This allows them to build shallow circuits suited to the problem (Cerezo et al. 2021; Tilly et al. 2022; Bauer et al. 2020). Quantum image processing gives several ways to store pixel values directly in qubits. These range from amplitude-style encodings (Le et al. 2011) to basis-encoded pixel values (Zhang et al. 2013). Basic image operations on these encodings have also been developed (Chakraborty et al. 2022; Zhou et al. 2017; Yao et al. 2017). Tensor-network methods work well as compressed representations of structured classical data, such as images (Cirac et al. 2021; Latorre 2005; Han et al. 2018; Stoudenmire and Schwab 2016; Ran et al. 2020; Guala et al. 2023) and particle-physics events (Araz and Spannowsky 2022). Hybrid models have also been applied to related imaging tasks, such as tomographic reconstruction (Haga 2024; Nau et al. 2023) and ghost imaging (Zhai et al. 2025). These solve inverse problems from indirect measurements. PCAE instead encodes and decodes a known image. Quantum convolutional and capsule networks are further examples (Slabbert and Petruccione 2024; Cong et al. 2019; Hur et al. 2022; Liu et al. 2023). PCAE focuses on the combination of a Pauli-correlator latent space, enforced group-equivariant pooling, and a physically motivated energy term.

We introduce the Pauli-Correlator Autoencoder (PCAE1D/PCAE2D), which combines these ingredients into a single architecture with three properties studied jointly. The first is a Paulicorrelator latent space, in which measured VQC observables rather than a raw quantum state form the representation handed to a classical decoder, following quantum-enhanced feature maps (Havl´ıˇcek et al. 2019). The second is an exact, numerically verified $D _ { 4 } – e q u i v a r i a n t$ observablepooling construction, which group-averages the observable grid over the eight symmetries of the 4×4 patch grid rather than motivating symmetry through architecture choices alone. This follows groupequivariant classical vision (Cohen and Welling 2016; Bronstein et al. 2021) and the equivariantquantum-neural-network line that formalises symmetry-embedding constructions for variational circuits (Meyer et al. 2023; Nguyen et al. 2024; Schatzki et al. 2024; West et al. 2024, 2025). The third is a learnable Hamiltonian energy term, used as an auxiliary training objective and read out as a descriptive diagnostic, with full nearest-neighbour $X X / Y Y / Z Z$ sectors in PCAE1D and gridedge ZZ terms in PCAE2D. Table 10 contrasts these properties against classical, adversarial, and standard quantum autoencoder baselines.

## Novelty relative to the nearest hybrid-vision architectures

PCAE’s closest architectural relatives are three hybrid quantum-vision patterns already established individually in the literature.

Register-based quantum autoencoders (Romero et al. 2017; Wang et al. 2024; Asaoka and Kudo 2025) encode an image’s pixel amplitudes directly into a qubit register and train a trash-qubit fidelity objective. PCAE feeds the circuit only quantum-inspired, pre-compressed MPS patch features, not raw pixel amplitudes. It hands the decoder measured Pauli correlators rather than register amplitudes or a fidelity score. PCAE is an autoencoder in the usual encoder–decoder sense. It maps each image patch to a measured observable vector and decodes it back to pixels. It does not compress a quantum state. Closer to PCAE, a hybrid autoencoder has placed a circuit between a classical encoder and decoder and passed its per-qubit ⟨Z⟩ values to the decoder as the latent vector (Sakhnenko et al. 2022). However, it uses neither MPS input nor pairwise correlators.

Quantum-kernel vision models (Havl´ıˇcek et al. 2019; Beaulieu et al. 2022) embed an input into a quantum feature space and read out a scalar kernel value consumed by a classical kernel method. PCAE reads out an explicit vector of local and pairwise Pauli expectation values as a latent representation for a trained decoder, and adds enforced group-equivariant pooling and a Hamiltonian energy term that a kernel construction has no analogue for.

Tensor-network (TN) circuit classifiers (Guala et al. 2023; Stoudenmire and Schwab 2016; Han et al. 2018) use MPS structure either as a standalone classical model or as a template for a shallow quantum circuit trained for classification. PCAE uses MPS compression only as a quantum-inspired preprocessing step that sets a separately trained VQC’s rotation angles, and targets per-image reconstruction rather than classification. Further, an MPS feature extractor has been placed before a VQC and trained end to end with it for classification (Chen et al. 2021). The MPS in PCAE is instead a fixed SVD compression.

PCAE’s contribution is the assembly of MPS preprocessing and the three properties above in one pipeline (Table 10). Section 5.2 examines how the individual ingredients recur across this literature.

## Contributions

PCAE1D/PCAE2D combines MPS patch features, a six-qubit VQC, a 19–27-dimensional Paulicorrelator vector, 2D Fourier positional encoding, a graph-Hamiltonian energy term, and a classical decoder. An exact, numerically verified $D _ { 4 } .$ -equivariant observable-pooling construction makes the PCAE2D observable map equivariant to the eight symmetries of the square, to numerical precision (Section 4.7). The guarantee follows from Reynolds group averaging over the $D _ { 4 }$ orbit (Puny et al. 2022) rather than from anything quantum, and the same construction gives the same exactness for a purely classical feature map. We report it as a design property of the pooled observable map.

Identically configured per-image fitting experiments cover six real image datasets (Section 4.1). On held-out CIFAR-10, a closed-form surrogate of the PCAE2D observable map meets the ±1 dB equivalence criterion against five of seven resource-matched classical/ablated feature maps, under two one-sided tests (TOST) with Holm correction. It clearly outperforms the two random-feature null controls (Section 4.3). Five fitted checkpoints are replayed on IBM hardware without retraining and decoded from hardware-measured observables (Section 4.4). A noise/shot-budget sweep and repeated execution across backends support the pilot (Sections 4.5–4.6). In a leakage-audited protocol with a fixed linear readout, pair observables are necessary when the target depends explicitly on products of input pairs (Section 4.8).

The appendices give the training settings, the matched-budget ablation, further held-out results and the hardware methodology.

## 2 Pauli-Correlator Autoencoder: Architecture

PCAE treats each spatial patch as a finite spin state and compresses it as a matrix product state (MPS). A variational quantum circuit (VQC) then measures a fixed set of Pauli observables, which a classical multi-layer perceptron (MLP) decodes back into pixels. A learnable graph-Hamiltonian

energy term regularises training. Parameterised Hamiltonian learning approaches treat the Hamiltonian as the primary learned object (Shi et al. 2023). In PCAE it is instead an auxiliary objective alongside the reconstruction loss, and its learned energy is read out as a diagnostic.

## Problem statement.

Given an image x partitioned into non-overlapping patches $\{ x _ { p } \} _ { p = 1 } ^ { P } , \ \mathrm { P C A E }$ learns a per-patch reconstruction map. Each patch is compressed to a quantum-inspired MPS feature vector $\phi _ { \mathrm { M P S } } ( x _ { p } )$ a learned linear layer sets the $\mathrm { V Q C }$ rotation angles, and measuring a fixed set of Pauli observables yields the correlator feature map $\Phi ( x _ { p } ) \in \mathbb { R } ^ { d } \ ( d = 2 7$ for PCAE1D, 19 for PCAE2D). With $n = 6$ qubits, bond set $\mathcal { E } ,$ and c measured pair channels per bond,

$$
d = 2 n + c | \mathcal { E } | = \left\{ \begin{array} { l l } { 2 ( 6 ) + 3 ( 5 ) = 2 7 , } & { \mathrm { P C A E 1 D ~ ( \it Z Z , \it X X , Y Y ~ o n ~ a ~ 5 \mathrm { - } b o n d ~ c h a i n ) , } } \\ { 2 ( 6 ) + 1 ( 7 ) = 1 9 , } & { \mathrm { P C A E 2 D ~ ( \it Z Z ~ o n ~ 7 ~ g r i d ~ e d g e s ) , } } \\ { 2 ( 6 ) + 1 ( 5 ) = 1 7 , } & { \it { Z Z \mathrm { - } o n l y ~ a b l a t i o n ~ c o n t r o l . } } \end{array} \right.\tag{1}
$$

All PCAE1D and PCAE2D circuit results use $d \ = \ 2 7$ and $d \ = \ 1 9$ . A classical decoder $D _ { \theta }$ maps $\Phi ( x _ { p } )$ , together with a Fourier positional encoding $\pi ( p )$ , back to a reconstructed patch $\hat { x } _ { p } = D _ { \theta } \big ( \boldsymbol { \bar { \Phi } } ( \boldsymbol { x } _ { p } ) , \pi ( p ) \big )$ . Training minimises the pixel-reconstruction loss with the graph-Hamiltonian energy term $E ( \cdot )$ as an auxiliary regulariser,

$$
\mathcal { L } ( \theta ) = \frac { 1 } { P m } \sum _ { p = 1 } ^ { P } \left\| x _ { p } - \hat { x } _ { p } \right\| _ { 2 } ^ { 2 } + \mathrm { ~ } \lambda \frac { 1 } { P } \sum _ { p = 1 } ^ { P } E \big ( \Phi ( x _ { p } ) \big ) ,\tag{2}
$$

where m is the number of pixels per patch, so the first term is the mean squared pixel error, and the second term is the mean graph-Hamiltonian energy over patches and λ sets its weight (Section 5.1 reports its efect). The energy of patch p is

$$
E \big ( \Phi ( x _ { p } ) \big ) = \sum _ { ( u , v ) \in \mathcal { E } } \ : \sum _ { \alpha \in S } J _ { u v } ^ { \alpha } \ : \langle \sigma _ { u } ^ { \alpha } \sigma _ { v } ^ { \alpha } \rangle _ { p } ,\tag{3}
$$

where $\mathcal { E }$ is the bond set of Eq. $( 1 ) , S = \{ x , y , z \}$ for PCAE1D and $S = \{ z \}$ for PCAE2D, and the couplings $J _ { u v } ^ { \alpha }$ are learned. The shared pipeline is

$$
{ \begin{array} { r l } & { { \mathrm { I m a g e } }  \{ { \mathrm { P a t c h e s } } \}  { \mathrm { M P S } } } \\ & { \qquad 4 6 { \mathrm { - D } } / 2 2 { \mathrm { - D } } { \mathrm { ~ F e a t u r e } }  6 { \mathrm { - D ~ V e c t o r } } } \\ & { \qquad 2 7 { \mathrm { - D } } / 1 9 { \mathrm { - D } } { \mathrm { ~ L a t e n t } }  { \mathrm { D e c o d e r } } } \\ & { \qquad { \mathrm { R e c o n s t r u c t i o n } } . } \end{array} }\tag{4}
$$

## Why MPS features.

Treated as an amplitude-encoded quantum state, a patch of n sites has a coeficient tensor with $2 ^ { n }$ entries. Matrix-product-state factorisation instead represents it as a chain of small tensors bounded by a fixed bond dimension χ. This is a compact, singular-value-controlled compression whose entanglement-entropy profile is already known to track image compressibility (Latorre 2005;

Table 1 The two executed configurations. Both use six qubits and the same decoder. They difer in patch size and in the measured observable set.
<table><tr><td></td><td>PCAE1D</td><td>PCAE2D</td></tr><tr><td>Benchmark</td><td>Monalisa</td><td>multi-dataset</td></tr><tr><td>Image size</td><td> $2 5 6 \times 2 5 6$ </td><td> $3 2 \times 3 2$  (resized)</td></tr><tr><td>Patch size</td><td> $6 4 \times 6 4$ </td><td> $8 \times 8$ </td></tr><tr><td>MPS sites</td><td>12</td><td>6</td></tr><tr><td>Local  $X / Z$  expectations</td><td>24</td><td>12</td></tr><tr><td>Nearest-neighbour ZZ</td><td>11</td><td>5</td></tr><tr><td>Bipartite entropy features</td><td>11</td><td>5</td></tr><tr><td>MPS feature vector</td><td>46-D</td><td>22-D</td></tr><tr><td>Measured observables</td><td>27-D</td><td>19-D</td></tr><tr><td>local Z, local X</td><td>6, 6</td><td>6, 6</td></tr><tr><td>pair terms</td><td>5 each ZZ, XX, YY</td><td>7 grid-edge ZZ</td></tr><tr><td>Fourier positional vector</td><td>8-D</td><td>4-D</td></tr></table>

Stoudenmire and Schwab 2016), and whose local expectations and correlations are cheap features that do not require a full state-preparation circuit.

## Why a Pauli-correlator latent space rather than a raw quantum state.

Handing the classical decoder a fixed vector of measured Pauli expectation values, instead of full state tomography, keeps the quantum-to-classical interface at a size linear in the qubit count, on the bounded-degree graphs used here, rather than exponential. The execution cost also grows with the number of measurement settings and the shots per setting. It also follows the measured-observableas-feature pattern used by quantum-enhanced feature maps generally (Havl´ıˇcek et al. 2019) and by projected quantum kernels built from local reduced density matrices (Huang et al. 2021). This choice is what makes the energy diagnostics of Section 5.1 and the training-dynamics diagnostics of Section 4.9 possible, since a full state vector would not expose a comparably compact, physically meaningful summary.

## Why a grid rather than a chain interaction topology for PCAE2D.

Arranging the six qubits on a $2 \times 3$ grid, rather than a linear chain, gives the latent a twodimensional coupling pattern. Each patch is processed by its own circuit, so the grid couples the six circuit inputs of one patch, not neighbouring patches. The $D _ { 4 }$ pooling of Section 4.7 does not depend on this choice. It acts by permuting the $4 \times 4$ grid of patches in an image, and a $2 \times 3$ graph has no $9 0 °$ rotation symmetry of its own. PCAE1D’s chain topology is the control without a grid. It is used to measure what the grid adds (Section 5.3).

Two preprocessing configurations were executed, one per topology. Both follow the same three stages. Each image is first divided into sixteen non-overlapping patches, except in the topology comparison (Appendix A). Every patch is then $\ell _ { 2 }$ -normalised and sequentially SVD-compressed to a matrix product state of bond dimension $\chi = 4$ . Finally, the local expectations, nearest-neighbour correlations and bipartite entropy features of that state are read of as the quantum-inspired MPS feature vector. The two configurations difer only in the sizes involved, which Table 1 lists in full.

![](images/1a6345fc65295717bbe0feebe052e1d995e385900b16dbb714c2f418b8e39ee2.jpg)

![](images/03f6296151a7f5d3ce3fdcb08934f3e2ecce27fbdb95c8c47345a6dbdcee50aa.jpg)  
Fig. 1 The PCAE1D (top) and PCAE2D (bottom) variational ans¨atze, as built for execution, before transpilation (encoding $R _ { X }$ rotations, positional $R _ { Y }$ rotations, then two entangling layers of $R _ { Z ^ { - } } R _ { Y ^ { - } } R _ { Z }$ single-qubit rotations with a CNOT ring for PCAE1D or fixed grid-edge CNOTs for PCAE2D). Both circuits were rebuilt gate-for-gate in Qiskit and validated against the original PennyLane training circuit before use in the hardware pilot of Section 4.4.

The $2 ^ { n }$ pixel values of a patch (n = 6 for 8×8, n = 12 for $6 4 \times 6 4 )$ are flattened in row-major order and reshaped into an order-n tensor with binary indices, so each index is one site of dimension 2. The tensor is compressed with the compress method of quimb (Gray 2018) at maximum bond dimension $\chi .$ With its default settings, the compressed state is in right-canonical form, with the orthogonality centre on the first site. The features are expectation values and Schmidt spectra of the compressed state, so they do not depend on its canonical form. The entropy feature of each cut is computed from the Schmidt probabilities $p _ { k }$ of that cut, which are the squared singular values. The implementation squares them a second time and renormalises them, $q _ { k } = p _ { k } ^ { 2 } / \sum _ { j } p _ { j } ^ { 2 } ,$ and returns $- \sum q _ { k }$ ln $q _ { k }$ . This feature agrees with the von Neumann entropy $- \textstyle \sum _ { k } p _ { k } \ln p _ { k }$ for a product cut and for a flat Schmidt spectrum, but not in general. All models in this paper were trained with this feature. In the multi-dataset runs of Section 4.1, $1 0 ^ { - 4 }$ is added to every pixel before normalisation, so that all-zero patches, such as MNIST background, can be normalised. The ofset enters only the MPS input. The decoder is still trained to reconstruct the unmodified patch. The remaining stages are common to both. A learned linear layer compresses the MPS feature vector to six circuit angles, one per qubit. After the circuit is executed, a two-hidden-layer MLP (128 and 256 units) maps the measured observables together with the Fourier positional vector back to a single greyscale patch. The positional encoding follows established use as a spatial inductive bias in classical generative vision models (Xu et al. 2021). Fourier features of this kind also help coordinate-based networks fit fine image detail (Tancik et al. 2020). During training the circuit is simulated exactly. Unless otherwise stated, training used additive zero-mean Gaussian noise with standard deviation 0.01 on the observable vector. Artificial Gaussian noise was not added during evaluation or hardware replay. Finite-shot estimates enter only in the hardware replay and the noisy simulation of Sections 4.4–4.6. Figure 1 shows the exact PCAE1D and PCAE2D ansatz circuits. These were rebuilt in Qiskit for the hardware pilot of Section 4.4, and numerically verified against the training-time PennyLane circuit before use.

Table 2 PCAE variants tested in this paper.
<table><tr><td>Variant</td><td>Topology</td><td>Latent dim.</td><td>Symmetry</td></tr><tr><td>PCAE1D</td><td>6-qubit chain</td><td>27-D</td><td>Unpooled</td></tr><tr><td>PCAE2D</td><td>2 × 3 grid</td><td>19-D</td><td>Unpooled</td></tr><tr><td>D₄-pooled PCAE2D</td><td>2 × 3 grid, pooled</td><td>19-D</td><td>D₄-equivariant (exact)</td></tr></table>

Table 2 summarises the tested variants. PCAE1D uses a linear 6-qubit chain (5 nearestneighbour bonds, 27-D observable vector). PCAE2D arranges the same 6 qubits on a $2 \times 3$ lattice (19-D latent signature). A D -pooled PCAE2D variant (Section 4.7) group-averages the observable grid over the eight symmetries of the 4 × 4 patch grid, $\begin{array} { r } { \Phi _ { D _ { 4 } } ( x ) = \frac { 1 } { 8 } \sum _ { g \in D _ { 4 } } P _ { g } ^ { - 1 } \Phi ( g x ) } \end{array}$ , which is equivariant by construction. Each circuit has two entangling layers, with 12 CNOTs for PCAE1D and 14 for PCAE2D before transpilation (Figure 1).

## 3 Experimental Setup

The reconstruction results in Sections 4.1, 4.2 and 4.4–4.6 use single-image fitting, the protocol developed for PCAE on Monalisa. A fresh model is optimised directly on its target image, with no held-out split. These results therefore measure per-image fitting and reconstruction capability. The hardware pilot replays the resulting fitted checkpoints on real devices without retraining. The symmetry and restricted pair-observable diagnostics use closed-form feature maps and the separate protocols stated in their respective sections. The training-dynamics readouts of Section 4.9 are reported as a descriptive diagnostic, with their scope set by the negative control given there. Heldout generalisation and component necessity are evaluated under the resource-matched statistical protocol, using a closed-form surrogate of the PCAE2D observable map (Section 4.3). All settings report MSE and $\mathrm { P S N R } = 2 0 \log _ { 1 0 } ( \mathrm { \bar { 1 } / \sqrt { M S E } } )$ . Image settings additionally report SSIM. Gradienttrained PCAE models were optimised using full-batch Adam, with learning rates $3 \times 1 0 ^ { - 3 }$ for PCAE1D and $4 \times 1 0 ^ { - 3 }$ for PCAE2D, with Hamiltonian-loss weight λ = 0.01. We used 200 optimisation steps for the Monalisa and CIFAR-10 hardware-checkpoint fits, 200 steps for the six-dataset same-set fits, and 90 steps for the topology comparison. The Monalisa run used seed 42 and the topology comparison used seeds {0, 1, 2}. Circuit rotation parameters were initialised independently from $\mathcal { U } ( 0 , \pi )$ , and Hamiltonian couplings from 0.1 N (0, 1). The linear projection and decoder layers used PyTorch’s default initialisation. The software environment and the per-experiment settings are given in Appendix A.

## 4 Results

## 4.1 Reconstruction across six real image datasets

For each image, we trained a fresh PCAE2D model using non-overlapping 8 × 8 patches (16 patches/image, 200 optimisation steps). The sample covers CIFAR-10 (Krizhevsky 2009), MNIST (LeCun et al. 1998), Fashion-MNIST (Xiao et al. 2017), and the PathMNIST, BloodMNIST, and PneumoniaMNIST subsets of MedMNIST v2 (Yang et al. 2023). Table 3 reports mean PSNR between 25.8 and 41.6 dB (SSIM 0.91–1.00). The architecture trains end to end and fits varied image structures under a common per-image protocol. A position-only ablation of the same decoder also fits these sampled images (Table 3). It has a higher mean PSNR on three of the six datasets, and is more than 5 dB higher on CIFAR-10 and PneumoniaMNIST. Table 3 establishes fitting feasibility. Evaluation-time substitutions show that trained models are sensitive to their observable inputs (Section 4.4). The $D _ { 4 }$ construction and the restricted pair-observable diagnostic establish separate properties of the representation (Sections 4.7 and 4.8). This protocol is per-image fitting, in the same sense as implicit neural representations, which fit one network to one image or scene (Ulyanov et al. 2020; Sitzmann et al. 2020; Tancik et al. 2020; Mildenhall et al. 2020; Dupont et al. 2021; Str¨umpler et al. 2022). Quantum versions of this idea fit a single signal or image with a variational circuit (Zhao et al. 2024; Fujihashi and Koike-Akino 2026; Lizzio Bosco et al. 2026). The closest of these, QIREN, is a quantum coordinate network aimed at a fitting advantage over classical Fourier networks (Zhao et al. 2024). PCAE is instead a patch autoencoder with a measured-correlator bottleneck, and it aims at a practical hybrid pipeline whose quantum part is the measured observable vector. Generalisation to held-out images is tested in Section 4.3. A compression comparison would also need a bit rate (Dupont et al. 2021; Str¨umpler et al. 2022), so we report fitting quality here.

Table 3 PCAE2D same-set reconstruction across six real image datasets (per-image trained, 100 steps, mean ± std over sampled images). SSIM is computed globally over each image. The position-only ablation is the same decoder trained with all 19 observable inputs set to zero, on the same images and patch layout, with the same learning rate and 100-step budget. It uses seed 0 for every image. The PCAE2D seeds were not recorded, so the two are not paired by initialisation.
<table><tr><td></td><td></td><td colspan="2">PCAE2D</td><td colspan="2">Position-only ablation</td></tr><tr><td>Dataset</td><td>n images</td><td>s PSNR (dB)</td><td>SSIM</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>CIFAR-10</td><td>3</td><td> $3 7 . 7 5 \pm 2 . 3 8$ </td><td> $0 . 9 9 8 6 \pm 0 . 0 0 0 3$ </td><td> $4 2 . 9 8 \pm 8 . 5 4$ </td><td> $0 . 9 9 9 1 \pm 0 . 0 0 0 8$ </td></tr><tr><td>MNIST</td><td>3</td><td> $2 5 . 7 7 \pm 2 . 2 3$ </td><td> $0 . 9 7 8 3 \pm 0 . 0 1 1 6$ </td><td> $2 5 . 9 5 \pm 2 . 3 1$ </td><td> $0 . 9 7 9 6 \pm 0 . 0 1 0 1$ </td></tr><tr><td>Fashion-MNIST</td><td>3</td><td> $3 4 . 5 5 \pm 3 . 0 5$ </td><td> $0 . 9 9 8 0 \pm 0 . 0 0 0 6$ </td><td> $3 3 . 7 0 \pm 0 . 4 9$ </td><td> $0 . 9 9 6 8 \pm 0 . 0 0 2 6$ </td></tr><tr><td>PathMNIST</td><td>3</td><td> $3 6 . 8 6 \pm 2 . 1 1$ </td><td> $0 . 9 1 1 8 \pm 0 . 0 6 2 7$ </td><td> $3 5 . 4 9 \pm 1 . 9 2$ </td><td> $0 . 8 9 0 4 \pm 0 . 0 6 5 5$ </td></tr><tr><td>BloodMNIST</td><td>2</td><td> $3 6 . 5 7 \pm 2 . 3 6$ </td><td> $0 . 9 9 2 2 \pm 0 . 0 0 6 3$ </td><td> $3 4 . 6 7 \pm 3 . 9 8$ </td><td> $0 . 9 8 3 3 \pm 0 . 0 1 5 1$ </td></tr><tr><td>PneumoniaMNIST</td><td>2</td><td> $4 1 . 6 0 \pm 0 . 9 8$ </td><td> $0 . 9 9 8 5 \pm 0 . 0 0 0 6$ </td><td> $4 7 . 2 8 \pm 1 . 0 4$ </td><td> $0 . 9 9 9 6 \pm 0 . 0 0 0 2$ </td></tr></table>

## 4.2 Scope of per-image fitting

A model optimised on one image is specific to that image. In this single-seed test, PCAE2D with 8×8 patches is trained on Monalisa alone for 200 steps and then evaluated, without further training, on Michelangelo’s Hand of God. Zero-shot transfer to Hand of God reaches 8.31 dB PSNR, with SSIM 0.1044. Extending training to include Hand of God for another 200 steps recovers 28.63 dB, with SSIM 0.9858. Section 4.3 tests the held-out case directly.

## 4.3 Held-out comparison with a closed-form surrogate

The results above measure per-image fitting. A separate question is whether PCAE’s feature map improves reconstruction on real, held-out images under a resource-matched multi-seed protocol. This comparison uses a closed-form surrogate of the PCAE2D observable map, so that many seeds and controls can be run. The surrogate combines 18 local patch statistics, six products of local-feature pairs that stand in for the pairwise ZZ correlators, and eight positional features. It is specified by hand and computed without running the circuit. Thus, the term has a diferent meaning here from the classical surrogates of Schreiber et al. (2023), which are derived from a trained quantum model. The main results are summarised below, including the equivalence tests of Table 4. Each of five split seeds fits the readout on six CIFAR-10 images and evaluates it on four disjoint held-out images, so each mean covers 20 held-out reconstructions.

On held-out CIFAR-10, the raw/local linear control reaches $1 8 . 8 0 \pm 1 . 4 2$ dB, against the PCAE2D surrogate’s $1 8 . 3 0 \pm 1 . 6 2$ dB. Aggregating held-out images within each split gives a PCAE2D-minus-control diference of −0.50 dB, with bootstrap 95% $\mathrm { C I } \ [ - 0 . 7 6 , - 0 . 2 9 ]$ dB and Wilcoxon $p \ : = \ : 0 . 0 6 2 5$ at $n = 5$ seeds (Efron and Tibshirani 1993; Wilcoxon 1945). The same numerical direction holds on five further real datasets, which use an earlier feature allocation (Appendix C). The surrogate clearly beats the random-feature controls (Table 4), and its mean stays within 1 dB of the simpler maps. The raw/local linear control is 0.50 dB ahead of it.

All controls are matched at the readout. Each passes a 32-D feature vector to the same linear readout with 2112 parameters, on the same splits. This matches the readout resources. The controls difer in what their features can express. The no-entanglement and ZZ-only controls remove all or half of the six raw pair products, and the quadratic classical control adds second-order classical terms. The ZZ-only control keeps exactly three of them. The surrogate is a 32-D map, not the 19- D circuit map of Table 1. The quadratic classical control keeps the 18 local features, the first six of its nine products of adjacent local-feature pairs, and the eight positional features, which gives the same 32-D width. Neither the surrogate nor the quadratic control includes the sines of its products. The ZZ-only and raw/local maps are padded with three and six zero columns, respectively, to reach 32-D.

A TOST equivalence test (Schuirmann 1987; Lakens 2017) at a declared ±1 dB margin asks whether each gap is small enough to call two maps equivalent rather than merely “not shown to difer”. The estimand is the seed-level PCAE2D-minus-control PSNR diference, averaged over held-out images within each seed. Without correction, equivalence requires both one-sided tests to reject at $\alpha = 0 . 0 5$ . This is the same as the 90% confidence interval lying inside $[ - 1 , + 1 ]$ dB (Table 4). The margin is a declared choice. With five seeds, $p = 0 . 0 6 2 5$ is the smallest attainable nonzero two-sided Wilcoxon signed-rank p-value, so that test cannot reject at the 0.05 level. The t-based 90% interval is wider than the percentile-bootstrap 95% interval in these data. Bootstrap intervals can be unstable with only five seed-level observations. For scale, a 1 dB PSNR diference corresponds to a factor of approximately $1 0 ^ { 0 . 1 } \simeq 1 . 2 6$ in MSE. After Holm correction (Holm 1979) across the seven comparisons, five controls meet the equivalence criterion against PCAE2D within that margin. These are raw/local linear, no entanglement, ZZ-only observables, quadratic classical features, and shufled pair observables. The raw/local linear control has the largest of these gaps. It meets the criterion with TOST $p = 0 . 0 0 9$ and Holm-adjusted $p = 0 . 0 2 7$ . The two weakest controls, strict classical random features and random-VQC, are not equivalent because the surrogate substantially outperforms them. The trained circuit has been tested on held-out images in the topology comparison of Section 5.3. Trained on two images for 90 steps, PCAE2D reaches $1 1 . 5 7 \pm 1 . 5 3$ dB on three unseen CIFAR-10 images, about 1 dB below a matched random-latent control under the same protocol $( 1 2 . 6 4 \pm 2 . 1 2 \ \mathrm { d B }$ , Appendix C). Held-out generalisation of the trained circuit is therefore not established at this budget. In a larger end-to-end test with 150 training and 50 held-out images, a parameter-matched classical control reconstructs the held-out images better than the trained PCAE2D model (Appendix C).

Table 4 TOST equivalence test on held-out CIFAR-10 (±1 dB margin, $\alpha = 0 . 0 5 ,$ five seed-level paired diferences, closed-form PCAE2D surrogate minus control). The verdict uses the Holm-adjusted p. Without correction, equivalence is the same as the 90% CI lying inside [−1, +1] dB. “No entanglement” removes the pair products. “Random VQC” is a block of Gaussian random features.
<table><tr><td>Control</td><td>∆PSNR (dB)</td><td>90% CI (dB)</td><td>TOST p Holm p</td><td>Verdict</td></tr><tr><td>Raw/local linear</td><td>-0.50</td><td>[−0.78, -0.22]</td><td>0.009 0.027</td><td>Equivalent</td></tr><tr><td>No entanglement</td><td>-0.16</td><td>[-0.52, +0.20]</td><td>0.004 0.016</td><td>Equivalent</td></tr><tr><td>ZZ-only observables</td><td>-0.26</td><td>[−0.46, -0.07]</td><td>0.001 0.004</td><td>Equivalent</td></tr><tr><td>Quadratic classical</td><td></td><td>-0.14 [−0.24, -0.04]</td><td>&lt; 0.001 &lt; 0.001</td><td>Equivalent</td></tr><tr><td>Shuffled pair observables</td><td>-0.13</td><td>3[−0.38, +0.11]</td><td>0.001 0.004</td><td>Equivalent</td></tr><tr><td>Strict classical random features</td><td>+2.45</td><td>[+2.06, +2.84]</td><td>0.999 1.000</td><td>PCAE2D better</td></tr><tr><td>Random VQC</td><td></td><td>+7.15 [+5.73, +8.57]</td><td>1.000 1.000</td><td>PCAE2D better</td></tr></table>

A matched-budget, multi-seed component ablation gives a consistent picture on per-image reconstruction. The ablation uses 240 steps and 3 seeds per variant (4 for the baseline). It evaluates the stitched image after seam blending, which lowers the PSNR of well-fitted models and compresses the diferences between them (Section 5.1). The quantum baseline, the classical-replacement control and most other variants lie within a fraction of a decibel of one another, so this evaluation does not separate them. Apart from a freeze-classical sanity check, only a circuit-free random-VQC control performs clearly worse. Freeze-classical holds the decoder fixed, so it cannot learn to reconstruct and is expected to fail. The ablation uses a descriptive pooled-standard-deviation comparison rule at 3 seeds rather than a calibrated test. Full results are in Appendix B. Section 5.2 discusses why this result and the pair-observable necessity result of Section 4.8 are consistent.

## 4.4 A real hardware reconstruction pilot

The results above are noiseless-simulator results. We now report a small but complete imagereconstruction pilot executed on real IBM Quantum hardware (ibm fez, a 156-qubit Heron-class processor).

The pilot covers five images. These are the full 16-patch Monalisa benchmark and four nativeresolution CIFAR-10 images (cat, ship×2, airplane), each optimised directly on its own target image following the same per-image protocol as Sections 4.1 and 4.2. For each image we took an alreadytrained PCAE1D/PCAE2D checkpoint’s exact learned parameters and rebuilt its measurement circuits gate-for-gate in Qiskit. We verified the translation against the original PennyLane simulator to within numerical noise, at most $7 . 9 \times 1 0 ^ { - 4 }$ maximum absolute error per observable across every patch. We then executed the same circuits on real hardware with 256 shots per measurement basis. This is a fixed per-basis shot budget, rather than an adaptive shadow-tomography-style estimation scheme (Huang et al. 2020). The hardware-measured observables were decoded by the same trained classical decoder, with no retraining and no simulator involved anywhere in the reconstruction path.

Reconstructions from real, noisy hardware measurements retain recognisable image structure. Monalisa reaches 25.90 dB PSNR from hardware measurements, against a 37.99 dB noiselesssimulator replay of the identical circuit (Figure 2). The four CIFAR-10 images average 29.56 ± 2.03 dB on hardware against 42.69 ± 3.84 dB in simulation (Figure 3, Table 5). The observed gap is consistent with noise in these unmitigated device executions. Circuits were transpiled at optimisation level 1 and submitted through SamplerV2 with default options. Dynamical decoupling, Pauli twirling and readout-error mitigation were not enabled (see Temme et al. 2017; van den Berg et al. 2022; Kim et al. 2023). Each image used 256 shots per basis, totalling 12,288 shots for Monalisa and 8,192 for each CIFAR-10 image. The Monalisa hardware reconstruction (25.90 dB) remains well above the random-latent control. On the Monalisa hardware checkpoint, substituting random latents for the observables at evaluation gives 12.49 dB. This value comes from the trainingrun evaluation, which blends the seams of the stitched image (Appendix D). The pilot used 176 circuits. The contemporaneous account-meter reading increased by 16 quantum-processor seconds. A similar protocol, with training in simulation and inference on IBM hardware, has been used for MedMNIST classification with error suppression and mitigation (Singh et al. 2026). Here, it is applied to reconstruction without mitigation.

Table 5 Real IBM hardware reconstruction pilot (backend ibm fez, 256 shots/basis, no retraining). PSNR compares hardware-decoded and simulator-decoded reconstructions against the same ground-truth image.
<table><tr><td>Image</td><td>Patches</td><td>Sim. PSNR (dB)</td><td>HW PSNR (dB)</td></tr><tr><td>Monalisa (PCAE1D)</td><td>16</td><td>37.99</td><td>25.90</td></tr><tr><td>CIFAR cat (PCAE2D)</td><td>16</td><td>44.85</td><td>31.52</td></tr><tr><td>CIFAR ship, hydrofoil (PCAE2D)</td><td>16</td><td>36.56</td><td>26.44</td></tr><tr><td>CIFAR ship, sea boat (PCAE2D)</td><td>16</td><td>42.57</td><td>31.20</td></tr><tr><td>CIFAR airplane (PCAE2D)</td><td>16</td><td>46.79</td><td>29.10</td></tr><tr><td>CIFAR mean ± std (4 images)</td><td></td><td>42.69 ± 3.84</td><td>29.56 ± 2.03</td></tr></table>

Ground truth  
![](images/16ebfc409692d682102afa514ded13327fd9d9004bcc5d8909a75826c14296e7.jpg)

Simulator replay (37.99 dB)  
![](images/563f5f8d2db06ac17566e75aacb41defea01f6d471fdc6a068dbb6d34dea90ab.jpg)

Real IBM hardware (25.90 dB)  
![](images/25d35468654db4a0b01ef0e58754ce6e547fd74fdf705a0519bd270a4c2c1870.jpg)  
Fig. 2 Monalisa reconstructed from real IBM Quantum hardware measurements (right) against the same trained checkpoint’s noiseless-simulator replay (centre) and ground truth (left). The hardware image is visibly noisier and exhibits patch-boundary banding, while the subject remains recognisable.

The pilot establishes that the trained PCAE observable channel survives execution on presentday hardware. The learned parameters, the measurement circuits and the classical decoder all transfer unchanged, and the decoded images remain quantitatively informative at 256 shots per basis on an unmitigated device. This is the sense in which PCAE runs on real quantum hardware. The measurement construction and validation results are in Appendix D.

![](images/e057afbc35bd680ec4221d34a31cb19e76b1e8fcb9685d6825eae7f7670afbdf.jpg)  
Fig. 3 Four CIFAR-10 images reconstructed from real IBM Quantum hardware measurements (bottom row) against noiseless-simulator replays of the same trained PCAE2D checkpoints (middle row) and ground truth (top row).

## 4.5 Noise/shot trade-of: a calibrated-noise simulation

The five-image pilot above used a single shot budget (256/basis) and one execution per image, which bounds how much it can say about noise sensitivity on its own. To use IBM Quantum time well, we separate what a calibrated device-noise simulation can establish at zero hardware cost from what genuinely requires hardware. Using qiskit-aer with the stored calibration snapshot of ibm fez (FakeFez, no quantum-processor time consumed), we replay the same already-trained checkpoints’ circuits at four shot budgets (256/512/1024/4096 shots per basis, three repeated executions each) and compare against both the noiseless-simulator ceiling and the real-hardware point already reported above.

Two findings follow, averaged over the five checkpoints (Figure 4). First, shot noise accounts for only part of the noiseless-to-real gap. Moving from the noiseless ceiling to a calibrated-noise simulation at the same 256-shot budget already costs 8.68 dB on average, before any real device is involved. Second, more shots do not reliably recover it. Increasing to 4096 shots per basis changes PSNR by only −0.05 dB on average across the five checkpoints, with individual images ranging from a 1.33 dB gain to a 0.96 dB loss. At this circuit size and shot range, shot count is not a lever that reliably improves reconstruction quality.

![](images/05ea017bc82ca76360307f701e709dd786a1a0bcbb32760d8d8a135f6bb9bc00.jpg)  
Fig. 4 Calibrated device-noise simulation (FakeFez, mean ± std over three executions). Dotted lines show noiseless ceilings and stars show real-hardware measurements. Increasing 256 to 4096 shots changes mean PSNR by only −0.05 dB.

Comparing the 256-shot noisy simulation directly against the real-hardware point isolates what the calibration snapshot does not capture. That comparison leaves a further 4.24 dB average gap, ranging from 2.26 to 8.06 dB across images. This residual is consistent with efects that the calibration model does not capture, such as drift between calibration and execution, crosstalk, and correlated errors. The airplane image is the clear outlier, with an 8.06 dB residual gap, well above any other image. We report it rather than exclude it, since a single-execution, five-image pilot cannot distinguish an image-specific efect from ordinary job-to-job variation. This calibratednoise simulation is a free, repeatable complement to the pilot. It bounds how much of the observed hardware gap is attributable to modelled device noise rather than unmodelled efects, using zero additional quantum-processor time.

## 4.6 Real-hardware anchor points: repeated execution on two further backends

The simulated trade-of above is useful only if it reflects real devices. We checked this directly on hardware with four jobs from the free-tier IBM Quantum allowance. These four jobs used well under 1% of the monthly limit. Two checkpoints, Monalisa and CIFAR cat, were replayed with no retraining on ibm marrakesh and ibm kingston. Both backends are distinct from the original pilot’s ibm fez. We ran them at the original 256-shot budget and, separately, at 1024 shots (Table 6).

Monalisa reproduces almost exactly across backends at matched shots (25.94 vs. 25.90 dB, ibm marrakesh vs. ibm fez), and reaches 26.10 dB on ibm kingston at 1024 shots, consistent with the simulated finding above that shot count is not a reliable lever. The CIFAR cat point is the more informative one. On ibm kingston at the original 256-shot budget it reaches 28.93 dB, which is 2.59 dB below the ibm fez pilot point. Its own 1024-shot execution on the same backend recovers to 31.24 dB, within 0.28 dB of the original. This is real job-to-job and backend-to-backend variation of a magnitude comparable to the unmodelled residual identified by the simulator comparison (Section 4.5). It gives direct evidence that repeated execution matters at least as much as backend choice or shot count. Repeated runs therefore give more stable estimates of hardware performance.

Table 6 Real-hardware anchor points: same checkpoints, no retraining, on two further backends. The original pilot’s ibm fez point is shown for reference.
<table><tr><td>Image</td><td>Backend</td><td>Shots</td><td>PSNR (dB)</td></tr><tr><td>Monalisa</td><td>ibm_fez (original)</td><td>256</td><td>25.90</td></tr><tr><td>Monalisa</td><td>ibm_marrakesh</td><td>256</td><td>25.94</td></tr><tr><td>Monalisa</td><td>ibm_kingston</td><td>1024</td><td>26.10</td></tr><tr><td>CIFAR cat</td><td>ibm_fez (original)</td><td>256</td><td>31.52</td></tr><tr><td>CIFAR cat</td><td>ibm_kingston</td><td>256</td><td>28.93</td></tr><tr><td>CIFAR cat</td><td>ibm_kingston</td><td>1024</td><td>31.24</td></tr></table>

## 4.7 Exact D<sub>4</sub>-equivariant observable pooling

A D<sub>4</sub>-pooled PCAE2D observable-grid map is equivariant to the eight symmetries of the square by construction:

$$
\Phi _ { D _ { 4 } } ( x ) = { \frac { 1 } { 8 } } \sum _ { g \in D _ { 4 } } P _ { g } ^ { - 1 } \Phi ( g x ) , \qquad \Phi _ { D _ { 4 } } ( h x ) = P _ { h } \Phi _ { D _ { 4 } } ( x ) \forall h \in D _ { 4 } .\tag{5}
$$

Here $D _ { 4 }$ acts on an image x by rotating or reflecting its 4 × 4 grid of patches, and $P _ { g }$ permutes the per-patch observable vectors in the same way, so that $P _ { g h } = P _ { g } P _ { h }$ . The property then holds exactly for any base map Φ. This is the standard group-averaging construction (Puny et al. 2022). It needs one evaluation of the base map for each of the eight group elements. Substituting hx and re-indexing the sum with $g ^ { \prime } = g h$ gives $\hat { P } _ { g } ^ { - 1 } = P _ { h } P _ { g ^ { \prime } } ^ { - 1 }$ , and hence $\Phi _ { D _ { 4 } } ( h x ) = P _ { h } \Phi _ { D _ { 4 } } ( x )$ . The numerical check below therefore tests the implementation, not the property. We run it on the closed-form surrogate feature map over 7000 patch transforms of cached CIFAR images (Table 7, Figure 5). The pooled map’s equivariance error is $9 . 5 7 \times 1 0 ^ { - 1 7 } \pm 1 . 2 6 \times 1 0 ^ { - 1 7 }$ , which is floating-point noise. The unpooled positional map has an error of $0 . 8 1 5 \pm 0 . 1 2 9$ , and the local/raw control has $0 . 8 3 7 \pm 0 . 1 3 1$ . Both are of order one. The patch grid’s geometry therefore makes the unpooled map $D _ { 4 ^ { - } } m o t i v a t e d$ only, and the explicit pooling construction is what delivers the property. Applied to the trained circuit, the same pooling reduces the equivariance error to order $1 0 ^ { - 8 }$ . On the four trained PCAE2D hardware checkpoints, with an exact statevector pass over all eight transformed copies of each image, the mean error falls from 0.2777 without pooling to $5 . 3 9 \times 1 0 ^ { - 8 }$ with it. This construction operates at the feature-grid level and serves here as a numerical diagnostic, rather than as a component of the trainable reconstruction pipeline. Integrating it end-to-end (Section 5.4) is concrete future work. This route to symmetry difers from equivariant quantum neural networks, in which every circuit layer is built to commute with the group action (Nguyen et al. 2024). A recent image classifier of that kind builds $C _ { 4 }$ rotation symmetry into the circuit itself (San Sebastian Sein et al. 2025). In PCAE the symmetry comes from averaging the observable grid over the group after measurement. The circuit itself is not equivariant, and the same average applies equally to a classical feature map.

Table 7 $D _ { 4 }$ equivariance error for unpooled and pooled observable-grid maps. Lower is better. The pooled map is group-averaged over all eight square symmetries. Values are computed on the closed-form surrogate map.
<table><tr><td>Model</td><td>Mean error</td><td>Std. dev.</td></tr><tr><td>D₄-pooled PCAE2D</td><td> $9 . 5 7 \times 1 0 ^ { - 1 7 }$ </td><td> $1 . 2 6 \times 1 0 ^ { - 1 7 }$ </td></tr><tr><td>No positional encoding</td><td> $7 . 4 0 \times 1 0 ^ { - 1 }$ </td><td> $1 . 5 2 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>PCAE2D positional</td><td> $8 . 1 5 \times 1 0 ^ { - 1 }$ </td><td> $1 . 2 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Local/raw</td><td> $8 . 3 7 \times 1 0 ^ { - 1 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 1 }$ </td></tr></table>

![](images/4c47208109d0c3e580f0bbbed9497dee2cc1db773a31eeeffb94b33e9917056c.jpg)  
Fig. 5 $D _ { 4 }$ equivariance diagnostic. The group-pooled map is equivariant to numerical precision. The unpooled positional map is not.

## 4.8 Pair-observable necessity in the pairwise-product protocol

The diagnostic tests whether a pair-observable channel is needed to fit a target that depends on pair correlations. The target is a fixed synthetic function of simulated input variables, six per sample, and its pair terms are products of variables joined by edges of the $2 \times 3$ qubit grid. Every control receives the same raw inputs as PCAE2D. The protocol is audited for leakage. For each feature block, we check whether a linear or near-linear map reaches near-zero residual on the features used to build the target, since such a fit would copy the target from the input. The diagnostic reported here passes this check. All feature maps in this diagnostic are closed-form, and no circuit is evaluated. The pair-observable channel is represented by the analytic pair products that a pair measurement would supply. By construction, these products match the pair terms of the target.

Table 8 PCAE2D restricted pair-correlation diagnostic across five seeds. The task is deliberately favourable to pair-observable structure. The table gives evidence for pair-observable necessity under a matched linear readout. All features are closed-form. “Random $\{ \dot { \mathrm { V Q C } } ^ { \dag } \}$ is a block of Gaussian random features.
<table><tr><td>Model</td><td>Mean MSE</td><td>Mean PSNR (dB)</td><td>Mean  $R ^ { 2 }$ </td></tr><tr><td>PCAE2D pair observables</td><td> $8 . 5 9 6 9 \times 1 0 ^ { - 4 }$ </td><td>32.77</td><td>0.9735</td></tr><tr><td>Freeze quantum</td><td> $4 . 7 9 0 4 \times 1 0 ^ { - 3 }$ </td><td>27.34</td><td>0.8533</td></tr><tr><td>No pair observables</td><td> $\phantom { + } 3 . 1 4 6 1 \times 1 0 ^ { - 2 }$ </td><td>15.03</td><td>0.0191</td></tr><tr><td>Raw-linear classical</td><td> $3 . 1 6 5 4 \times 1 0 ^ { - 2 }$ </td><td>15.00</td><td>0.0133</td></tr><tr><td>Parameter-matched classical</td><td> $3 . 3 0 3 3 \times 1 0 ^ { - 2 }$ </td><td>14.82</td><td>-0.0297</td></tr><tr><td>Random VQC</td><td> $3 . 7 6 1 6 \times 1 0 ^ { - 2 }$ </td><td>14.25</td><td>-0.1732</td></tr><tr><td>Freeze classical</td><td> $6 . 0 9 7 3 \times 1 0 ^ { - 2 }$ </td><td>12.17</td><td>-0.9025</td></tr></table>

The target also has a term that is not a pair product, so no feature block reproduces it exactly. The experiment demonstrates representational capability under a deliberately aligned protocol, not a general performance advantage.

Table 8 and Figure 6 report the result across five seeds. PCAE2D’s pair-observable channel reaches mean $R ^ { 2 } = 0 . 9 7 3 5$ , or 32.77 dB PSNR. This is a 17.74 dB improvement over a no-pairobservables control $( R ^ { 2 } = 0 . 0 1 9 )$ and 18.52 dB over the Random VQC control (Gaussian random latents, $R ^ { 2 } = - 0 . 1 7 3 )$ . Randomly rescaling and masking the pair features (“freeze quantum”), with only the readout trained, still reaches $R ^ { 2 } = 0 . 8 5 3$ . A raw-linear classical control $( R ^ { 2 } = 0 . 0 1 3 3 )$ , a parameter-matched classical control $( R ^ { 2 } = - 0 . 0 3 0 )$ , and a frozen-classical control $( R ^ { 2 } = - 0 . 9 0 3 )$ all fail on this deliberately pair-dependent target. The no-pair-observables control retains only single-site nonlinear channels and removes every cross-site product feature. Under the fixed linear readout, it therefore cannot construct the cross terms on which the target depends.

The mechanism is explicit under the fixed linear readout. A linear map of single-qubit expectations $\{ \langle O _ { i } \rangle \}$ cannot form the cross-term $\langle O _ { i } \rangle \langle O _ { j } \rangle$ that the target depends on. The pair measurement supplies the joint correlator $\langle O _ { i } O _ { j } \rangle$ ⟩ directly as a feature. Formally, let $z _ { i } = \langle O _ { i } \rangle$ and let every admissible predictor be afine, $\begin{array} { r } { f ( \bar { z } ) = w ^ { \top } z + b . } \end{array}$ . Then $\partial ^ { 2 } f / \partial z _ { i } \partial z _ { j } = 0$ for all $i , j .$ A target with a term $c z _ { i } z _ { j } , c \neq 0 .$ , has this mixed derivative equal to c wherever $z _ { i }$ and $z _ { j }$ vary independently, so no afine $f$ can fit it. Adding $z _ { i } z _ { j }$ to the feature vector makes the term linearly representable. A pair-observable channel is therefore representationally necessary within the tested feature class and fixed linear-readout protocol to fit this target. A classical model that forms the same products, for example through quadratic features or a nonlinear readout, would also fit this target. For this reason we scope the claim to the linear protocol rather than to classical models in general.

## 4.9 A scoped training-dynamics diagnostic

Beyond the results above, we tracked two exploratory training-dynamics readouts. These are a magnetisation-style statistic $M _ { z } ( t )$ and a signed energy-to-entanglement ratio $R _ { E S } ( t )$ . Here $M _ { z } ( t )$ is the mean of the local ⟨Z⟩ observables over all qubits and patches at optimisation step $t . \ R _ { E S } ( t )$ is the learned Hamiltonian energy divided by the fixed sum of the MPS bond entropy features, so it tracks the energy up to a constant scale. Both drift smoothly over training with run-to-run fluctuations, and neither shows a sharp transition. Any apparent onset is a within-run medianplus-2σ threshold crossing, rather than a calibrated statistical test. The rule flags a run when the largest step-to-step change in $M _ { z }$ exceeds that run’s median change plus two standard deviations.

![](images/03088decabe302b16f099aad8eaf72a0d9f1af484dac1719755fcba3d18ed464.jpg)  
Fig. 6 Restricted pair-correlation diagnostic summary (mean ± std over five seeds). The pair-observable feature map dominates every control on this deliberately favourable task.

A negative control fixes the scope. We applied the identical protocol to a classical-replacement variant, in which the Hamiltonian energy is identically zero throughout training. That variant crosses the threshold in all $4 / 4$ tested cases, which is as often as the Hamiltonian-regularised model’s $4 / 4 .$ . The threshold therefore fires whether the Hamiltonian term is present or absent, and it does not track that term. We therefore report these traces as a descriptive readout on the learned energy and entanglement.

## 5 Discussion

## 5.1 Efect of the Hamiltonian energy term

In a single-seed comparison on Monalisa, the reconstruction loss alone gives the higher PSNR. Removing the signed energy term increases PSNR from 38.63 to 44.56 dB (Table 9). The legacy objective adds a signed mean energy to the reconstruction loss. The energy is linear in the learned couplings, which are not bounded, so it has no lower limit. The optimiser can therefore lower the total loss by driving the energy down without improving reconstruction. A contrastive variant, assigning lower energy to correct feature–position pairs than to shufled pairs, recovers part of the gap (42.92 dB) but stays below its no-energy counterpart. These single-seed values (Monalisa, 200 steps, seed 42) are computed from the reconstruction term of the training loss at the final step, over the decoded patches and without seam blending. The matched-budget ablation of Appendix B instead evaluates the stitched image after seam blending, which smooths a band around every patch boundary. Blending lowers the PSNR of a well-fitted model, as for the hardware checkpoint, which gives 37.99 dB without blending and 32.24 dB with it (Appendix D). Under that evaluation, the noenergy-loss variant and the baseline agree within their seed-to-seed spread. The two comparisons therefore use diferent evaluations and are not directly comparable. The efect of the signed energy term is in general agreement with an independent study, which finds no benefit beyond simple classical regularisers from a VQE-inspired energy term added to the loss of a generative adversarial network (Strnadel 2025).

Table 9 Hamiltonian-sector controls (single-seed, Monalisa, 200 steps, seed 42). PSNR is computed over the decoded patches, without seam blending.
<table><tr><td>Experiment</td><td>PSNR (dB)</td><td>Interpretation</td></tr><tr><td>Legacy baseline, signed energy</td><td>38.63</td><td>Standard energy-regularised run</td></tr><tr><td>No energy loss</td><td>44.56</td><td>Removing energy improves PSNR</td></tr><tr><td>Contrastive Hamiltonian core</td><td>42.92</td><td>Stronger objective helps modestly vs. baseline, still below no- energy-loss</td></tr></table>

The learned couplings do not enter the decoder directly, but the energy term contributes gradients during training and therefore changes the learned circuit representation. It is consequently an auxiliary training objective as well as a post-training diagnostic. A physically motivated regulariser has room to help when it resolves an otherwise underdetermined objective. Under this protocol, the reconstruction loss alone provides the stronger optimisation signal.

The training-dynamics readouts of Section 4.9 provide a descriptive window into how the learned energy and entanglement evolve.

## 5.2 Task-dependent representational scope

Whether a quantum feature map ofers any advantage over a classical model is a property of the specific data and target, not a fixed property of the model class. Dequantisation results have repeatedly shown that apparent quantum advantages on structured, low-entanglement classical data can be matched once the right classical feature is identified (Huang et al. 2021; Tang 2019). Where provable separations do exist, they are established for carefully constructed learning problems rather than for generic structured data (Gyurik and Dunjko 2023). We separate three questions. First, PCAE at six qubits can be simulated exactly on a classical computer, which says little about larger scales. Second, whether its model class admits an eficient classical replacement at scale is open. Tensor networks are one route to such a replacement, and they have been used to simulate quantum circuits (Patra et al. 2024, 2025) and variational quantum algorithms (Aizpurua et al. 2025). Tensor-network analyses also give conditions for such dequantisation of variational models (Shin et al. 2024), and a classical tie at small scale does not settle it. Random Fourier features can also replace some variational models eficiently, under stated conditions (Sweke et al. 2025). Quantum convolutional networks on common image benchmarks can be simulated eficiently by a classical method based on Pauli shadows (Bermejo et al. 2026). Third, on held-out CIFAR-10, a closedform surrogate of the PCAE2D map stays within 1 dB of matched classical feature maps. This matches large benchmarks in which simple classical models often match or beat small quantum models, and removing entanglement often costs little (Bowles et al. 2024). In PCAE the quantum processor evaluates only the circuit and returns the vector of measured Pauli expectation values. The MPS compression is quantum-inspired but runs on classical hardware, as do the Fourier positional encoding, the $D _ { 4 }$ pooling, the decoder and the optimisation. Any quantum contribution to PCAE must therefore enter through the measured observable vector.

The restricted diagnostic of Section 4.8 is constructed so that the relevant structure is a product of two input variables, which raw or local features cannot represent under a fixed linear readout.

That is the regime in which PCAE’s pair-observable channel separates. Natural image patches at the resolution and patch size used here are low-entanglement, locally structured data (Martyn et al. 2020; Lu et al. 2025), where a linear or shallow classical readout is already close to suficient. For the images of Table 3, the bipartite von Neumann entropy of the compressed patch states is at most 0.58 nats for CIFAR-10 and 0.24 nats for MedMNIST, over all five cuts of every patch. The largest value allowed at bond dimension four is ln 4. The distinction lies in the measured pairobservable channel rather than in the circuit. The two entangling layers do generate an entangled state, but whether the measured channel carries task-relevant information is a property of the target. On natural-image targets the pair observables add little beyond the local ones. On the constructed pairwise-product target they are necessary. Together with the held-out comparisons of Section 4.3, these results delineate where the tested entangling representation does and does not yield a measurable benefit.

Three of PCAE’s ingredients recur individually in the hybrid quantum-vision literature, covering MPS compression of structured input (Stoudenmire and Schwab 2016; Han et al. 2018; Cirac et al. 2021; Guala et al. 2023), a Pauli-correlator latent space handed to a classical decoder (Havl´ıˇcek et al. 2019), and a physically motivated auxiliary regulariser alongside a task loss (Shi et al. 2023). Its specific instantiation, a six-qubit VQC with a 19–27-D correlator vector and a graph-Hamiltonian term, sits within the parameter ranges typical of that literature. The measured behaviour is therefore evidence about the design pattern itself. This boundary is established for the datasets, patch sizes and qubit count tested here.

## 5.3 Qubit topology as an inductive bias

The choice of qubit interaction graph (PCAE2D vs. PCAE1D) is an architectural inductive bias, and we test it directly under matched held-out conditions. The real-circuit comparison (3 seeds, 3 held-out CIFAR-10 images) gives a PCAE1D-minus-PCAE2D mean PSNR diference of 0.16 dB. After averaging images within seed, the bootstrap 95% CI is [−0.38, 0.46] dB and the seed-level Wilcoxon test gives $p = 0 . 5 0$ . Both topologies fall about 1 dB below a matched random-latent control in this test (Appendix C), so the comparison does not show what the qubit topology contributes to the latent. PCAE2D completes the matched 90-step runs in roughly half the wallclock time. At the tested scale, topology is therefore primarily a resource-cost choice rather than a reliable accuracy advantage. These tests do not establish formal equivalence.

Table 10 contrasts PCAE’s structural properties against reference frameworks, including the three nearest hybrid-vision architecture families. Its distinguishing features are a Pauli-correlator latent space, a 2D Fourier positional bias with enforced $D _ { 4 }$ pooling, and a Hamiltonian energy term, each documented and scoped in the preceding sections.

## 5.4 Validated scope

Every circuit-based reconstruction result in this paper is per-image trained. The held-out comparison of Section 4.3 uses a closed-form surrogate of the PCAE2D map. On held-out images the trained circuit falls below a matched random-latent control at the tested budget, so its held-out generalisation is not established. The per-image results of Section 4.1 are not compared with a matched-size classical per-image network, as is usual for implicit neural representations. A singleseed position-only ablation of the same decoder has a higher mean PSNR than PCAE2D on three of the six datasets (Table 3).

Table 10 Architectural diferentiation from representative baseline designs, including the three nearest hybrid-vision architecture families (register-based QAE, quantum-kernel vision, TN-circuit classifier).
<table><tr><td>Axis</td><td>Conv. AE baseline</td><td>GAN base- line</td><td>Register- based QAE</td><td>Quantum- kernel vision</td><td>TN-circuit classifier</td><td>PCAE (ours)</td></tr><tr><td>Latent repre- sentation</td><td>Cęntinuous Rd</td><td>Cęntinuous Rd</td><td>Quantum register</td><td>Scalar kernel value</td><td>MPS/circuit state</td><td>Pauli- correlator vector</td></tr><tr><td>Spatial bias</td><td>Convolu- tional</td><td>Convolu- tional</td><td>Encoding- dependent</td><td>Encoding- dependent</td><td>MPS chain ordering</td><td>2D Fourier encoding</td></tr><tr><td>Auxiliary objective</td><td>Bottleneck reconstruc- tion</td><td>Adversarial loss</td><td>Fidelity / reconstruc- tion</td><td>None (kernel value ònly)</td><td>Classification loss</td><td>Hamiltonian energy term</td></tr><tr><td>Enforced group sym- metry</td><td>Not in base- line</td><td>Not in base- line</td><td>Not in refer- ence design</td><td>Not in refer- ence design</td><td>Not in refer- ence design</td><td>D₄ observ- able pooling</td></tr><tr><td>Evaluated role</td><td>Reconstruc- tion</td><td>Reconstruc- tion</td><td>Compression</td><td></td><td>Classification Classification</td><td>Recon- struction + diagnostics</td></tr></table>

The $D _ { 4 } .$ -pooled map (Section 4.7) is a feature-grid-level construction and numerical diagnostic, measured on the closed-form surrogate in Table 7. Integrating it into the trainable end-to-end pipeline is future work. The pair-observable necessity result (Section 4.8) is scoped to a synthetic task with closed-form features, constructed to require products of input pairs. Ordinary natural-image reconstruction lies outside that scope (see Section 5.2). The training-dynamics traces (Section 4.9) are a descriptive readout. The threshold fires whether the Hamiltonian term is present or absent, at 4/4 in both cases.

The hardware pilot (Section 4.4) is a five-image, unmitigated pilot with one run per image on the main backend rather than a statistically resolved hardware benchmark. The matched real-circuit topology comparison uses 3 seeds and 3 held-out images per topology. These results support a resource-cost interpretation rather than formal equivalence of the topologies. Every result here is at a six-qubit, two-layer scale. Trainability of variational models is known to degrade with width and depth through barren-plateau efects (McClean et al. 2018; Larocca et al. 2025). Further, models that provably avoid barren plateaus are often also eficiently simulable on a classical computer (Cerezo et al. 2025). Thus, whether the tested boundary moves at larger scale is an open question.

## 6 Conclusion

We presented the Pauli-Correlator Autoencoder (PCAE), a hybrid quantum–classical architecture for image reconstruction that assembles matrix-product-state patch features, a measured Paulicorrelator latent space and a learnable Hamiltonian energy term into a single pipeline (Section 2). Across six real image datasets, per-image-trained PCAE2D models reach 25.8–41.6 dB PSNR. On held-out CIFAR-10, a closed-form surrogate of the PCAE2D observable map is equivalent within 1 dB to five resource-matched feature maps (Section 4.3). A hardware pilot replayed trained checkpoints on IBM Quantum devices without retraining, establishing feasibility at a five-image scope with one run per image on the main backend (Section 4.4).

Pair observables are necessary on a synthetic task requiring products of input pairs, under a fixed linear readout (Section 4.8). The Hamiltonian energy term acts as an auxiliary training objective, and its learned energy is recorded as a descriptive readout during training (Section 5.1). The pooled observable map is also exactly D -equivariant by construction, which we report as a design-correctness property rather than a quantum capability, since the same Reynolds averaging applied to a classical feature map is equally exact. At the six-qubit scale tested here, PCAE fits varied real images and runs on real quantum hardware.

Future work should integrate the D -pooled map into an end-to-end trainable pipeline (Section 5.4), scale the hardware pilot toward a statistically resolved benchmark with error mitigation and repeated trials, test held-out generalisation of the trained circuit against the same resource-matched controls, search for further real-world tasks with pairwise-product structure where PCAE’s pair-observable separation translates into a reconstruction advantage, and extend the resource-matched, leakage-audited evaluation protocol to other hybrid quantum vision architectures.

Acknowledgements. We thank the maintainers of the open-source scientific computing libraries used in this work. We acknowledge support from ANRF, India (grant ANRF/ECRG/2025/003752).

## Statements and Declarations

Competing interests. The authors declare no competing interests.

Funding. This work was supported by the Anusandhan National Research Foundation (ANRF), India, under grant ANRF/ECRG/2025/003752.

Data availability. Dataset provenance and access routes are cited in the manuscript. All datasets used (CIFAR-10, MNIST, Fashion-MNIST, and the PathMNIST, BloodMNIST and PneumoniaMNIST subsets of MedMNIST v2) are publicly available.

Code availability. The code supporting this study is available from the corresponding author upon reasonable request.

Author contributions. S.C. and S.P. conceived the project. S.C. wrote the code and carried out the experiments and analysis. S.P. supervised the work. Both authors wrote the manuscript.

## Appendix A Training settings and evaluation protocol

All trained models use Adam with full-batch updates over the patches of their training images. The loss is the reconstruction mean squared error plus λ = 0.01 times the mean Hamiltonian energy, except where an ablation removes or replaces the energy term. The held-out comparison fits no network. Its linear readout is solved in closed form as a ridge regression with regularisation 10<sup>−6</sup>. Table A1 lists the settings of each experiment.

Where Table A1 lists seeds, the experiment drivers set the Python, NumPy and PyTorch random seeds explicitly. The drivers of the six-dataset fits and the CIFAR-10 hardware fits did not set or record a seed, so these are reported as not recorded. “Mean ± std” denotes the mean and the population standard deviation over the unit named in each caption. Bootstrap intervals are percentile 95% intervals, and Wilcoxon values are two-sided signed-rank tests. The runs used PyTorch 2.6.0 with CUDA 12.4, PennyLane 0.42.3 and quimb 1.14.0 on an NVIDIA GeForce GTX 1050.

Table A1 Training settings. Learning rates are for Adam. Patches are non-overlapping unless a stride is given.
<table><tr><td>Experiment</td><td>Model</td><td>Patches</td><td>Steps</td><td>Learning rate</td><td>Seeds</td></tr><tr><td>Monalisa hardware checkpointa</td><td>PCAE1D</td><td> $1 6 , 6 4 \times 6 4$ </td><td>200</td><td>0.003</td><td>42</td></tr><tr><td>240-step ablation</td><td>PCAE1D</td><td> $1 6 , 6 4 \times 6 4$ </td><td>240</td><td>0.003</td><td> $0 , 1 , 2 ^ { b }$ </td></tr><tr><td>Six-dataset fits</td><td>PCAE2D</td><td> $1 6 , 8 \times 8$ </td><td>100</td><td>0.004</td><td>not recorded</td></tr><tr><td>CIFAR-10 hardware fits</td><td>PCAE2D</td><td> $1 6 , 8 \times 8$ </td><td>200</td><td>0.004</td><td>not recorded</td></tr><tr><td>Topology comparison</td><td>1D / 2D</td><td>49, 8 × 8, stride 4</td><td>90</td><td> $0 . 0 0 3 \mathrm { ~ / ~ } 0 . 0 0 4$ </td><td>0, 1, 2</td></tr><tr><td>Held-out comparison</td><td>surrogate</td><td> $1 6 , 8 \times 8$ </td><td></td><td>n/a n/a (ridge)</td><td>0-4c</td></tr></table>

<sup>a</sup> The hardware pilot uses the legacy shared-baseline-seed-42 checkpoint. The Hamiltonian-control values in Table 9 are independent fresh reruns made with the same nominal 100-step, seed-42 settings; they are not the hardware checkpoint.  
<sup>b</sup> The reported baseline mean also includes seed 3.  
<sup>c</sup> Split seeds; each split fits the ridge readout on six CIFAR-10 images and holds out four disjoint images.

## Appendix B Matched-budget component ablation

All nine variants were trained on Monalisa with a matched 240-step budget and an identical learning rate and schedule, for three seeds each (seeds 0–2), with a fourth seed (3) for the baseline (Table B2). The first freeze-quantum attempts at seeds 0–2 failed with undefined (NaN) metrics because of an implementation error, and only the three successful reruns enter its mean. Freezeclassical trains only the circuit and the Hamiltonian couplings, with the decoder and projection layers fixed, and freeze-quantum does the reverse. No-entanglement removes the CNOT rings, and no-energy-loss removes the energy term. No-MPS replaces the MPS feature extractor with a 46- dimensional vector of direct patch statistics (mean, standard deviation, minimum, maximum, seven percentiles and a normalised intensity histogram), and keeps the projection, circuit, positional encoding and decoder. Random-VQC replaces the circuit observables at every forward pass with an independently resampled standard-Gaussian vector of the same dimension, sets the energy to zero, and trains the decoder normally. Classical-replacement replaces the circuit with a tanh map, and classical-matched limits that map to the circuit’s parameter count. The classical-replacement variant exceeds the quantum baseline by 0.081 dB, against a pooled standard deviation of 0.019 dB. The classical-matched variant falls 0.197 dB below the baseline. These diferences are measured on the stitched image after seam blending, which compresses the diferences between well-fitted variants (Section 5.1).

Table B2 Matched-budget (240-step) component ablation on Monalisa, three seeds per variant (four for the baseline), mean ± std. PSNR and SSIM are computed on the stitched image after seam blending.
<table><tr><td>Variant</td><td>PSNR (dB)</td><td>SSIM</td><td>Variant</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>Baseline (quantum)</td><td> $3 3 . 4 2 \pm 0 . 0 2$ </td><td> $0 . 9 9 3 8 5 \pm 0 . 0 0 0 0 3$ </td><td>Classical-replacement</td><td> $3 3 . 5 0 \pm 0 . 0 1$ </td><td> $0 . 9 9 3 9 5 \pm 0 . 0 0 0 0 2$ </td></tr><tr><td>Freeze-quantum</td><td> $3 3 . 3 9 \pm 0 . 0 6$ </td><td> $0 . 9 9 3 8 1 \pm 0 . 0 0 0 0 6$ </td><td>Classical-matched</td><td> $3 3 . 2 2 \pm 0 . 1 9$ </td><td> $0 . 9 9 3 5 6 \pm 0 . 0 0 0 2 7$ </td></tr><tr><td>No-entanglement</td><td> $3 3 . 4 8 \pm 0 . 0 2$ </td><td> $0 . 9 9 3 9 2 \pm 0 . 0 0 0 0 3$ </td><td>Random-VQC</td><td> $2 8 . 0 9 \pm 0 . 2 3$ </td><td> $0 . 9 7 9 0 7 \pm 0 . 0 0 1 3 5$ </td></tr><tr><td>No-MPS</td><td> $3 3 . 3 4 \pm 0 . 0 5$ </td><td> $0 . 9 9 3 7 4 \pm 0 . 0 0 0 0 8$ </td><td>Freeze-classical</td><td> $1 0 . 9 4 \pm 0 . 0 0$ </td><td> $0 . 0 2 1 9 \pm 0 . 0 0 0 1$ </td></tr><tr><td>No-energy-loss</td><td> $3 3 . 4 4 \pm 0 . 0 1$ </td><td> $0 . 9 9 3 8 7 \pm 0 . 0 0 0 0 1$ </td><td></td><td></td><td></td></tr></table>

## Appendix C Held-out reconstruction

The held-out protocol of Section 4.3 was repeated on cached subsets of six datasets, with three stratified split seeds (Table C3). From the first 400 images of each dataset, each split draws 20 training and 10 held-out images per class (one and one for CIFAR-10), and skips any class with fewer than 30 images among them. This gives 10/10, 200/100, 200/100, 160/80, 120/60 and 40/20 training/held-out images per split for CIFAR-10, MNIST, Fashion-MNIST, PathMNIST, Blood-MNIST and PneumoniaMNIST, so PathMNIST uses 8 of its 9 classes and BloodMNIST 6 of its 8. These results predate the correction of the positional channel in the CIFAR-10 comparison of Section 4.3. The earlier width selection retained only two of the eight positional features for the PCAE2D surrogate and none for the quadratic control. The no-entanglement and raw/local controls retained all eight. The feature allocation here therefore difers from that of the corrected comparison. The local/raw control has equal or higher mean PSNR on every dataset. After averaging within each split seed, all six seed-bootstrap intervals exclude zero, and each two-sided signed-rank test gives $p = 0 . 2 5$ , the smallest value attainable with three seeds.

Table C3 Held-out PSNR (dB) on cached subsets, mean ± std over held-out images, for the best local/raw control and the PCAE2D surrogate.
<table><tr><td>Dataset</td><td>Best local/raw</td><td>PCAE2D surrogate</td><td>Higher mean</td></tr><tr><td>CIFAR-10</td><td>21.17±1.98</td><td>19.99±2.02</td><td>Local/raw</td></tr><tr><td>MNIST</td><td>16.55±1.27</td><td>16.12±1.34</td><td>No-entanglement/local</td></tr><tr><td>Fashion-MNIST</td><td>17.67±1.71</td><td>17.33±1.80</td><td>Local/raw</td></tr><tr><td>PathMNIST</td><td>27.52±6.02</td><td>27.01±5.81</td><td>Local/raw</td></tr><tr><td>BloodMNIST</td><td>24.79±1.78</td><td>24.24±1.66</td><td>Local/raw</td></tr><tr><td>PneumoniaMNIST</td><td>26.49±2.48</td><td>25.73±2.29</td><td>Local/raw</td></tr></table>

In the held-out topology test of Section 5.3, PCAE1D reaches $1 1 . 7 3 \pm 1 . 9 9$ dB and PCAE2D reaches 11.57 ± 1.53 dB. A random-latent control under the identical protocol replaces the observable vector with resampled noise and trains the decoder normally. It reaches $1 2 . 7 4 \pm 2 . 2 1$ dB for PCAE1D and $1 2 . 6 4 \pm 2 . 1 2$ dB for PCAE2D. The trained circuits are about 1 dB below this floor, and the paired diference is negative in all six seed-level comparisons (two-sided Wilcoxon $p = 0 . 0 3 1 )$ . At 90 steps, only the PCAE1D-minus-PCAE2D diference in this test is informative. In a separate end-to-end test, one PCAE2D model is trained with full-batch Adam for 20 epochs on 150 CIFAR-10 images and evaluated, without further training, on 50 held-out images. Each of three seeds reshufles the split. The model reaches 14.39 ± 0.26 dB, against $1 5 . 6 1 \pm 0 . 2 9$ dB for a parameter-matched classical linear control trained the same way. The control is higher at each of the three seeds.

## Appendix D Hardware pilot methodology

The Monalisa circuit is the legacy PCAE1D checkpoint shared-baseline-seed-42, trained for 200 steps with seed 42 and learning rate 0.003. It is distinct from the fresh “Legacy baseline, signed energy” rerun of Table 9. Its noiseless reference of 37.99 dB (Table 5) is an exact statevector replay of the saved checkpoint through the hardware-reconstruction pipeline. Both the hardware and the simulator PSNR are computed over the decoded patches, without stitching them or blending the seams. The training-run evaluation of the same checkpoint blends the seams of the stitched image and records 32.24 dB. The four CIFAR-10 circuits are PCAE2D checkpoints, each trained on its own image with 16 non-overlapping 8 × 8 patches (Table A1). PCAE1D measures Z, X, ZZ, XX and YY (27 observables). Its pair correlators are joint parities of the same all-Z, all-X and all-Y basis measurements that give the single-qubit values, so three measurement circuits per patch sufice, or 48 circuits for Monalisa. PCAE2D measures Z, X and ZZ (19 observables) and needs two circuits per patch, or 128 circuits for the four CIFAR-10 images.

Before any hardware submission, every circuit was checked against the cached training-time observables by exact statevector simulation. The largest absolute error was $7 . 8 9 \times 1 0 ^ { - 4 }$ for Monalisa and at most $9 . 4 2 \times 1 0 ^ { - 6 }$ for the CIFAR-10 checkpoints. On hardware, the mean absolute deviation of the observables from the statevector values was 0.096–0.125, so gate translation is not the main source of the hardware–simulator PSNR gap. Hardware counts were converted to expectation values with the standard parity estimator and passed, with the training-time positional encoding, through the unmodified trained decoder.

## References

Aizpurua B, Patra S, Etxezarreta Martinez J, et al (2025) Hacking cryptographic protocols with tensor network attacks. Physica Scripta 100(8):085102. https://doi.org/10.1088/1402-4896/ adedd2

Araz JY, Spannowsky M (2022) Classical versus quantum: comparing tensor-network-based quantum circuits on Large Hadron Collider data. Physical Review A 106(6):062423. https://doi.org/ 10.1103/PhysRevA.106.062423

Asaoka H, Kudo K (2025) Quantum autoencoders for image classification. Quantum Machine Intelligence 7(2):71. https://doi.org/10.1007/s42484-025-00297-x

Bauer B, Bravyi S, Motta M, et al (2020) Quantum algorithms for quantum chemistry and quantum materials science. Chemical Reviews 120(22):12685–12717. https://doi.org/10.1021/acs.chemrev. 9b00829

Beaulieu D, Miracle D, Pham A, et al (2022) Quantum kernel for image classification of real world manufacturing defects. arXiv preprint arXiv:221208693 URL https://arxiv.org/abs/2212.08693, arXiv:2212.08693 [quant-ph]

van den Berg E, Minev ZK, Temme K (2022) Model-free readout-error mitigation for quantum expectation values. Physical Review A 105(3):032620. https://doi.org/10.1103/PhysRevA.105. 032620

Bermejo P, Braccia P, Rudolph MS, et al (2026) Quantum convolutional neural networks are efectively classically simulable. PRX Quantum 7(2):020304. https://doi.org/10.1103/8qt9-72ts, URL https://link.aps.org/doi/10.1103/8qt9-72ts

Bowles J, Ahmed S, Schuld M (2024) Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv preprint arXiv:2403.07059

Bronstein MM, Bruna J, Cohen T, et al (2021) Geometric deep learning: grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478 https://doi.org/10.48550/arXiv.2104. 13478

Cerezo M, Arrasmith A, Babbush R, et al (2021) Variational quantum algorithms. Nature Reviews Physics 3(9):625–644. https://doi.org/10.1038/s42254-021-00348-9

Cerezo M, Larocca M, Garc´ıa-Mart´ın D, et al (2025) Does provable absence of barren plateaus imply classical simulability? Nature Communications 16:7907. https://doi.org/10.1038/ s41467-025-63099-6

Chakraborty S, Mandal SB, Shaikh SH (2022) Quantum image processing: challenges and future research issues. International Journal of Information Technology 14(1):475–489. https://doi.org/ 10.1007/s41870-018-0227-8

Chen SYC, Huang CM, Hsing CW, et al (2021) An end-to-end trainable hybrid classical-quantum classifier. Machine Learning: Science and Technology 2(4):045021. https://doi.org/10.1088/ 2632-2153/ac104d

Cirac JI, P´erez-Garc´ıa D, Schuch N, et al (2021) Matrix product states and projected entangled pair states: concepts, symmetries, theorems. Reviews of Modern Physics 93(4):045003. https: //doi.org/10.1103/RevModPhys.93.045003

Cohen T, Welling M (2016) Group equivariant convolutional networks. In: Proceedings of the 33rd International Conference on Machine Learning, Proceedings of Machine Learning Research, vol 48. PMLR, pp 2990–2999

Cong I, Choi S, Lukin MD (2019) Quantum convolutional neural networks. Nature Physics 15(12):1273–1278. https://doi.org/10.1038/s41567-019-0648-8

Dupont E, Goli´nski A, Alizadeh M, et al (2021) COIN: COmpression with implicit neural representations. arXiv preprint arXiv:2103.03123

Efron B, Tibshirani RJ (1993) An Introduction to the Bootstrap. Chapman and Hall/CRC, New York, https://doi.org/https://doi.org/10.1201/9780429246593

Fujihashi T, Koike-Akino T (2026) Quantum implicit neural compression. In: Quantum Computing and Artificial Intelligence: First International Workshop, QC+AI 2025, Communications in Computer and Information Science, vol 2813. Springer Nature Switzerland, pp 60–69, https: //doi.org/10.1007/978-3-032-15931-1 5

Gray J (2018) quimb: A Python package for quantum information and many-body calculations. Journal of Open Source Software 3(29):819. https://doi.org/10.21105/joss.00819

Guala D, Zhang S, Cruz E, et al (2023) Practical overview of image classification with tensor-network quantum circuits. Scientific Reports 13(1):4427. https://doi.org/10.1038/ s41598-023-30258-y

Gyurik C, Dunjko V (2023) Exponential separations between classical and quantum learners. arXiv preprint arXiv:2306.16028 https://doi.org/10.48550/arXiv.2306.16028

Haga A (2024) Quantum annealing-based computed tomography using variational approach for a real-number image reconstruction. Physics in Medicine and Biology 69(4):04NT02. https://doi. org/10.1088/1361-6560/ad2155

Han ZY, Wang J, Fan H, et al (2018) Unsupervised generative modeling using matrix product states. Physical Review X 8(3):031012. https://doi.org/10.1103/PhysRevX.8.031012

Havl´ıˇcek V, C´orcoles AD, Temme K, et al (2019) Supervised learning with quantum-enhanced feature spaces. Nature 567(7747):209–212. https://doi.org/10.1038/s41586-019-0980-2

Holm S (1979) A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2):65–70

Huang HY, Kueng R, Preskill J (2020) Predicting many properties of a quantum system from very few measurements. Nature Physics 16(10):1050–1057. https://doi.org/10.1038/ s41567-020-0932-7

Huang HY, Broughton M, Mohseni M, et al (2021) Power of data in quantum machine learning. Nature Communications 12(1):2631. https://doi.org/10.1038/s41467-021-22539-9

Hur T, Kim L, Park DK (2022) Quantum convolutional neural network for classical data classification. Quantum Machine Intelligence 4(1):3. https://doi.org/10.1007/s42484-021-00061-x

Kim Y, Eddins A, Anand S, et al (2023) Evidence for the utility of quantum computing before fault tolerance. Nature 618:500–505. https://doi.org/10.1038/s41586-023-06096-3

Krizhevsky A (2009) Learning multiple layers of features from tiny images. Tech. rep., University of Toronto, Toronto, Ontario

Lakens D (2017) Equivalence tests: A practical primer for t tests, correlations, and metaanalyses. Social Psychological and Personality Science 8(4):355–362. https://doi.org/10.1177/ 1948550617697177

Larocca M, Thanasilp S, Wang S, et al (2025) Barren plateaus in variational quantum computing. Nature Reviews Physics 7:174–189. https://doi.org/10.1038/s42254-025-00813-9

Latorre JI (2005) Image compression and entanglement. arXiv preprint arXiv:quant-ph/0510031 https://doi.org/10.48550/arXiv.quant-ph/0510031

Le PQ, Dong F, Hirota K (2011) A flexible representation of quantum images for polynomial preparation, image compression, and processing operations. Quantum Information Processing 10:63–84. https://doi.org/10.1007/s11128-010-0177-y

LeCun Y, Bottou L, Bengio Y, et al (1998) Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11):2278–2324. https://doi.org/10.1109/5.726791

Liu Z, Shen PX, Li W, et al (2023) Quantum capsule networks. Quantum Science and Technology 8(1):015016. https://doi.org/10.1088/2058-9565/aca55d

Lizzio Bosco D, Wang S, Serra G, et al (2026) Quantum implicit neural representations for novel view synthesis. arXiv preprint arXiv:2601.05250

Lu S, Kan´asz-Nagy M, Kukuljan I, et al (2025) Tensor networks and eficient descriptions of classical data. Physical Review A 111(3):032409. https://doi.org/10.1103/PhysRevA.111.032409

Martyn J, Vidal G, Roberts C, et al (2020) Entanglement and tensor networks for supervised image classification. arXiv preprint arXiv:2007.06082

McClean JR, Boixo S, Smelyanskiy VN, et al (2018) Barren plateaus in quantum neural network training landscapes. Nature Communications 9(1):4812. https://doi.org/10.1038/ s41467-018-07090-4

Meyer JJ, Mularski M, Gil-Fuster E, et al (2023) Exploiting symmetry in variational quantum machine learning. PRX Quantum 4(1):010328. https://doi.org/10.1103/PRXQuantum.4.010328

Mildenhall B, Srinivasan PP, Tancik M, et al (2020) NeRF: Representing scenes as neural radiance fields for view synthesis. In: Computer Vision – ECCV 2020, Lecture Notes in Computer Science, vol 12346. Springer, pp 405–421, https://doi.org/10.1007/978-3-030-58452-8 24

Nau MA, Vija AH, Gohn W, et al (2023) Exploring the limitations of hybrid adiabatic quantum computing for emission tomography reconstruction. Journal of Imaging 9(10):221. https://doi. org/10.3390/jimaging9100221

Nguyen QT, Schatzki L, Braccia P, et al (2024) Theory for equivariant quantum neural networks. PRX Quantum 5(2):020328. https://doi.org/10.1103/PRXQuantum.5.020328

Patra S, Jahromi SS, Singh S, et al (2024) Eficient tensor network simulation of IBM’s largest quantum processors. Physical Review Research 6(1):013326. https://doi.org/10.1103/ PhysRevResearch.6.013326

Patra S, Singh S, Or´us R (2025) Projected entangled pair states with flexible geometry. Physical Review Research 7(1):L012002. https://doi.org/10.1103/PhysRevResearch.7.L012002

Preskill J (2018) Quantum computing in the NISQ era and beyond. Quantum 2:79. https://doi. org/10.22331/q-2018-08-06-79

Puny O, Atzmon M, Ben-Hamu H, et al (2022) Frame averaging for invariant and equivariant network design. In: International Conference on Learning Representations (ICLR)

Ran SJ, Sun ZZ, Fei SM, et al (2020) Tensor network compressed sensing with unsupervised machine learning. Physical Review Research 2(3):033293. https://doi.org/10.1103/ PhysRevResearch.2.033293

Romero J, Olson JP, Aspuru-Guzik A (2017) Quantum autoencoders for eficient compression of quantum data. Quantum Science and Technology 2(4):045001. https://doi.org/10.1088/ 2058-9565/aa8072

Sakhnenko A, O’Meara C, Ghosh KJB, et al (2022) Hybrid classical-quantum autoencoder for anomaly detection. Quantum Machine Intelligence 4(2):27. https://doi.org/10.1007/ s42484-022-00075-z

San Sebastian Sein P, Ca˜nizo M, Or´us R (2025) Image classification with rotation-invariant variational quantum circuits. Physical Review Research 7:013082. https://doi.org/10.1103/ PhysRevResearch.7.013082

Schatzki L, Larocca M, Nguyen QT, et al (2024) Theoretical guarantees for permutationequivariant quantum neural networks. npj Quantum Information 10(1):12. https://doi.org/10. 1038/s41534-024-00804-1

Schreiber FJ, Eisert J, Meyer JJ (2023) Classical surrogates for quantum learning models. Physical Review Letters 131(10):100803. https://doi.org/10.1103/PhysRevLett.131.100803

Schuirmann DJ (1987) A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. Journal of Pharmacokinetics and Biopharmaceutics 15(6):657–680. https://doi.org/10.1007/BF01068419

Shi J, Wang W, Lou X, et al (2023) Parameterized Hamiltonian learning with quantum circuit. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(5):6086–6095. https://doi. org/10.1109/TPAMI.2022.3203157

Shin S, Teo YS, Jeong H (2024) Dequantizing quantum machine learning models using tensor networks. Physical Review Research 6:023218. https://doi.org/10.1103/PhysRevResearch.6. 023218

Singh G, Jin H, Merz KM (2026) Benchmarking MedMNIST dataset on real quantum hardware. Scientific Reports 16:9017. https://doi.org/10.1038/s41598-026-35605-3

Sitzmann V, Martel JNP, Bergman AW, et al (2020) Implicit neural representations with periodic activation functions. In: Advances in Neural Information Processing Systems, vol 33. Curran Associates, Inc., pp 7462–7473, URL https://proceedings.neurips.cc/paper/2020/hash/ 53c04118df112c13a8c34b38343b9c10-Abstract.html

Slabbert D, Petruccione F (2024) Hybrid quantum-classical feature extraction approach for image classification using autoencoders and quantum SVMs. arXiv preprint arXiv:2410.18814 https: //doi.org/10.48550/arXiv.2410.18814

Stoudenmire EM, Schwab DJ (2016) Supervised learning with tensor networks. In: Advances in Neural Information Processing Systems, pp 4799–4807

Strnadel D (2025) Diferentiable energy-based regularization in GANs: a simulator-based explo ration of VQE-inspired auxiliary losses. arXiv preprint arXiv:2512.12581

Str¨umpler Y, Postels J, Yang R, et al (2022) Implicit neural representations for image compression. In: Computer Vision – ECCV 2022, Lecture Notes in Computer Science, vol 13686. Springer, Cham, pp 74–91, https://doi.org/10.1007/978-3-031-19809-0 5

Sweke R, Recio-Armengol E, Jerbi S, et al (2025) Potential and limitations of random Fourier features for dequantizing quantum machine learning. Quantum 9:1640. https://doi.org/10.22331/ q-2025-02-20-1640

Tancik M, Srinivasan PP, Mildenhall B, et al (2020) Fourier features let networks learn high frequency functions in low dimensional domains. In: Advances in Neural Information Processing Systems, vol 33. Curran Associates, Inc., pp 7537–7547, URL https://proceedings.neurips.cc/ paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html

Tang E (2019) A quantum-inspired classical algorithm for recommendation systems. In: Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing (STOC), pp 217–228, https://doi.org/10.1145/3313276.3316310

Temme K, Bravyi S, Gambetta JM (2017) Error mitigation for short-depth quantum circuits. Physical Review Letters 119(18):180509. https://doi.org/10.1103/PhysRevLett.119.180509

Tilly J, Chen H, Cao S, et al (2022) The variational quantum eigensolver: a review of methods and best practices. Physics Reports 986:1–128. https://doi.org/10.1016/j.physrep.2022.08.003

Ulyanov D, Vedaldi A, Lempitsky V (2020) Deep image prior. International Journal of Computer Vision 128:1867–1888. https://doi.org/10.1007/s11263-020-01303-4

Wang H, Tan J, Huang Y, et al (2024) Quantum image compression with autoencoders based on parameterized quantum circuits. Quantum Information Processing 23(2):41. https://doi.org/10. 1007/s11128-023-04243-3, URL https://doi.org/10.1007/s11128-023-04243-3

West MT, Heredge J, Sevior M, et al (2024) Provably trainable rotationally equivariant quantum machine learning. PRX Quantum 5(3):030320. https://doi.org/10.1103/PRXQuantum.5.030320

West MT, Heredge J, Sevior M, et al (2025) Erratum: provably trainable rotationally equivariant quantum machine learning [PRX Quantum 5, 030320 (2024)]. PRX Quantum 6:020902. https: //doi.org/10.1103/g4zk-79l9

Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bulletin 1(6):80–83. https://doi.org/10.2307/3001968

Xiao H, Rasul K, Vollgraf R (2017) Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747 https://doi.org/10.48550/arXiv. 1708.07747

Xu R, Wang X, Chen K, et al (2021) Positional encoding as spatial inductive bias in GANs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 13569–13578

Yang J, Shi R, Wei D, et al (2023) MedMNIST v2—a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data 10:41. https://doi.org/10.1038/ s41597-022-01721-8

Yao XW, Wang H, Liao Z, et al (2017) Quantum image processing and its application to edge detection: theory and experiment. Physical Review X 7(3):031041. https://doi.org/10.1103/ PhysRevX.7.031041

Zhai X, Xiao T, Huang J, et al (2025) Quantum neural compressive sensing for ghost imaging. Physical Review Applied 23(1):014018. https://doi.org/10.1103/PhysRevApplied.23.014018

Zhang Y, Lu K, Gao Y, et al (2013) NEQR: a novel enhanced quantum representation of digital images. Quantum Information Processing 12(8):2833–2860. https://doi.org/10.1007/ s11128-013-0567-z

Zhao J, Qiao W, Zhang P, et al (2024) Quantum implicit neural representations. In: Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol 235. PMLR, pp 60940–60956, URL https://proceedings.mlr.press/v235/zhao24l. html

Zhou RG, Hu W, Fan P, et al (2017) Quantum realization of the bilinear interpolation method for NEQR. Scientific Reports 7(1):2511. https://doi.org/10.1038/s41598-017-02575-6