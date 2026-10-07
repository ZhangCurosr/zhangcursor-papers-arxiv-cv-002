# Mu-DisCoCat: A Variational Pipeline for Compositional Generalization on Quantum Processors

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Accepted for publication at the IEEE 2nd International Conference on Quantum Artificial Intelligence (QAI 2026).

Mina Abbaszadeh   
Department of Computer Science   
University College London   
London, United Kingdom   
m.abbaszadeh@ucl.ac.uk

Martha Lewis Institute for Logic, Language and Computation University of Amsterdam Amsterdam, Netherlands m.a.f.lewis@uva.nl

Abstract—Achieving compositional concept generalization (CoCoGen), the ability to understand novel situations by recombining learned primitives, remains a fundamental challenge in artificial intelligence. Compositional semantic models such as Compositional Distributional Semantics (DisCoCat) offer solutions by generalising vectors to tensors, but suffer from scaling bottlenecks when learning the tensors. Mapping DisCoCat onto Variational Quantum Circuits (VQCs) resolves this limitation for text, yet the methodology has not been expanded to multimodal situations such as the ones involved in CoCoGen. This paper introduces Mu-DisCoCat: a multimodal variational quantum learning framework for DisCoCat that achieves CoCoGen. The framework first learns stable object representations from singleobject image–text pairs, then fixes these and uses them to learn the relations between them in multi-object situations. In classical simulations, the model used Uhlmann state fidelity to compute the overlap between the multimodal circuit representations and achieved higher relational OOD accuracy than the evaluated CLIP baseline. Its deployment was evaluated using the destructive SWAP test across noisy quantum emulators, including a range of IBM fake backends, IQM FakeAphrodite, and the IBM Marrakesh quantum processor. Despite real-world device noise, the hardware-executed models maintained a strong positive correlation with simulated fidelities, reliably distinguishing unseen similar and dissimilar pairs. Our work establishes a framework for executing CoCoGen on VQCs, demonstrating a viable use case for near-term quantum hardware.

## I. INTRODUCTION

Humans can understand novel situations by recombining previously learned concepts, a capability known as compositional concept generalization or CoCoGen. For example, a person who has learned that cars crossing roads are dangerous will still recognize the danger posed by a yellow car, even if they have never encountered a yellow car before. Achieving this capability in AI remains an open challenge, particularly

Matilda Karabina Moore   
Department of Computer Science   
University College London   
London, United Kingdom   
matilda.moore.21@ucl.ac.uk   
Raem Haq   
Department of Computer Science   
University College London   
London, United Kingdom   
raem.haq.23@ucl.ac.uk   
Mehrnoosh Sadrzadeh   
Department of Computer Science   
University College London   
London, United Kingdom   
m.sadrzadeh@ucl.ac.uk

for learning relational concepts [1]–[4]. Mainstream visionlanguage models such as OpenAI’s CLIP [5] achieve strong performance in image-text alignment but struggle with CoCo-Gen [6], [7]. The AI models based on compositional representations of meaning should exhibit a stronger performance. One such framework is the Distributional Compositional Categorical model (DisCoCat) [8]. Although theoretically well suited to compositional reasoning, DisCoCat has shown limited empirical success in CoCoGen, since it represents relations as tensors, learning which on classical computers has a high computational cost [9].

DisCoCat was inspired by Categorical Quantum Mechanics [10] and both frameworks are based on the theory of Compact Closed Categories. This enables us to represent its constructions as Variational Quantum Circuits (VQCs), allowing efficient training using hybrid quantum-classical optimization. VQCs only require a handful of parameters to learn the tensors, leading to a substantial improvement in the scalability and practicality of DisCoCat [11], [12]. Despite these advances, the original DisCoCat cannot be applied to CoCoGen, as its extensions to multimodal learning are limited, e.g. [13] considers audio-text representations only for adjective–noun phrases and [14], [15] only focuses on concepts seen during training. Initial VQC-based work on CoCoGen jointly learned object and relation representations within a single-stage relational training procedure, leading to suboptimal results [16]. In our subsequent work [17], we introduced a multistage training strategy in which object representations are first learned from single-object imagetext pairs and subsequently frozen while only the relation parameters are optimised. However, that implementation relied on post-selection for the DisCoCat grammatical contractions and used larger quantum representations, including 9-qubit angle-encoded CLIP image circuits, with evaluation restricted to classical quantum simulation. In the present work, we adapt this multistage framework for near-term quantum hardware by reducing the image representation to four qubits, replacing post-selection with discard, and estimating the resulting image-caption overlap using the destructive SWAP test. This enables evaluation on noisy quantum emulators and real quantum hardware. While Mu-DisCoCat provides an effective model for CoCoGen, executing it on current quantum hardware presents an additional challenge. During training, the similarities between image and caption representations are computed directly from quantum states using the Uhlmann fidelity. However, current quantum processors cannot directly access quantum states and instead provide only measurement outcomes. Consequently, deploying trained models requires a hardware compatible procedure that faithfully estimates the same similarity measure under realistic device noise. In this work, we employ the destructive SWAP test as a hardware compatible estimator of the quantum-state overlap, enabling direct experimental evaluation of the similarity learned during training. We also adapt the designed VQCs to a reduced hardware representation compatible with near-term quantum devices by reducing the depth and the number of qubits both for object and relation learning. The framework was validated on IBM Aer, multiple noisy IBM fake backends, the IBM Marrakesh quantum processor, and the IQM FakeAphrodite emulator. The results show that although hardware noise reduces the agreement with noiseless simulations, the learned similarity relationships remain positively correlated with the simulated fidelities and continue to distinguish similar from dissimilar image-caption pairs. These findings demonstrate the feasibility of achieving CoCoGen on current quantum hardware and provide a practical use case.

## II. BACKGROUND

DisCoCat encodes conceptual relations by mapping the pregroup semantics of text to finite-dimensional Hilbert spaces via a homomorphic mapping. A pregroup is a partially ordered monoid where each element has a left and a right adjoint

$$
\mathcal { P } = ( P , \cdot , 1 , \leq , ( - ) ^ { l } , ( - ) ^ { r } )
$$

Composition is enabled via the monoid multiplication. Whenever an element and its adjoint compose in the correct order, a cancellation happens, that is 1, the unit of our monoid multiplication. The following inequalities hold in the pregroup:

$$
p \cdot p ^ { r } \leq 1 \qquad p ^ { l } \cdot p \leq 1
$$

The pregroup semantics of an atomic entity is a generator of the pregroup; that of a relation, on the other hand, consists of multiplied adjoint types. For instance, given a pregroup generated over the set of generators $\{ n , s \}$ , the semantics of an atomic entity such as “cube” is $n ,$ whereas that of a relation such as “left $\mathrm { o f } ^ { \mathrm { * } }$ or “right $\mathrm { o f } ^ { \mathrm { * } }$ is $n ^ { r } \cdot s \cdot n ^ { l }$ . When a relation composes with two entities, the following cancellation happens:

$$
n \cdot ( n ^ { r } \cdot s \cdot n ^ { l } ) \cdot n \leq s
$$

A pregroup-Hilbert Space mapping sends atomic elements to finite dimensional Hilbert Spaces over the field of reals R, i.e.

$$
\mathcal { F } \colon \mathcal { P }  \mathbf { H i l b }
$$

Concretely, this mapping is defined by induction over the generators. For the above set of generators, the action of this map is defined as follows:

$$
{ \mathcal { F } } ( n ) = V \qquad { \mathcal { F } } ( s ) = W
$$

It assigns a finite dimensional Hilbert space to each generator. Pregroup elements with multiplication and adjoints are mapped to tensors of spaces with their duals as follows, the unit of multiplication is mapped to the field R. Formally, for $a , b \in P ;$

$$
{ \mathcal { F } } ( a ^ { r } ) = { \mathcal { F } } ( a ^ { l } ) = { \mathcal { F } } ( a ) ^ { * } , ~ { \mathcal { F } } ( a \cdot b ) = { \mathcal { F } } ( a ) \otimes { \mathcal { F } } ( b ) , ~ { \mathcal { F } } ( 1 ) = \mathbb { R }
$$

As an example, the type of the above relational entities are mapped to the following Hilbert space:

$$
{ \mathcal { F } } ( n ^ { r } \cdot s \cdot n ^ { l } ) = V ^ { * } \otimes W \otimes V ^ { * }
$$

The cancellations between pregroup types become as follows:

$$
\begin{array} { r l } & { \mathcal { F } ( a \cdot a ^ { r } \leq 1 ) = \mathcal { F } ( a ) \otimes \mathcal { F } ( a ) ^ { * } \to \mathcal { F } ( 1 ) = \mathbb { R } , } \\ & { \mathcal { F } ( a ^ { l } \cdot a \leq 1 ) = \mathcal { F } ( a ) ^ { * } \otimes \mathcal { F } ( a ) \to \mathcal { F } ( 1 ) = \mathbb { R } . } \end{array}\tag{1}
$$

Essentially, the above map takes the inner product of the elements of the tensor space, i.e. for $v _ { i } , v _ { j } \in \mathcal { F } ( a )$ , we have:

$$
\sum _ { i , j } c _ { i , j } ( v _ { i } \otimes v _ { j } ) \mapsto \sum _ { i , j } c _ { i , j } \langle v _ { i } \mid v _ { j } \rangle
$$

This inner product helps us define matrix-vector and matrixmatrix multiplication, generalising to tensor contraction in higher orders. For the calculations that show this see [8], [18]. Intuitively speaking, we are modeling semantic composition as tensor product and semantic interactions between words as tensor contraction. Due to space limitations we will not provide more details and refer the reader to existing work [8], [11], [16].

The Hilbert space representations assigned to entities and relations are compiled into variational quantum circuits (VQCs) using parameterized ansatz. We use the Instantaneous Quantum Polynomial (IQP) ansatz [19] for the image circuits and the Sim4 ansatz for the textual circuits [20]. For relational captions, Sim4 is applied to the individual word components, which are subsequently composed according to the corresponding DisCoCat grammatical reduction. The details of this assignment are given in Section III-B with an example provided in Figure 3. Other combinations were also considered but these resulted in the best performance, see Table I.

TABLE I: Stage 1 performance of different Sim4 and IQP ansatz assignments for the image and text circuits. Validation and test splits are out of distribution (OOD), and all results are averaged over five random seeds.
<table><tr><td>Structure</td><td>Train Valid</td><td></td><td>Test</td></tr><tr><td>Image_sim4 &amp; Text_sim4</td><td>24.76%</td><td>25%</td><td>33.33%</td></tr><tr><td>Image_sim4 &amp; Text_IQP</td><td>27.18%</td><td>25%</td><td>33.33%</td></tr><tr><td>Image_IQP &amp; Text_IQP</td><td>80.54%</td><td>45%</td><td>53.34%</td></tr><tr><td>Image_IQP &amp; Text_sim4</td><td>78.66%</td><td>75.00%</td><td>66.70%</td></tr></table>

## III. METHODS

## A. Multistage Training

Mu-DisCoCat consists of a two-stage training pipeline. In the first stage, object representations are learned from singleobject image-caption pairs. In the second stage, the learned object parameters are frozen and transferred to relational learning, where only the relation parameters are optimized. Figure 1 summarizes this pipeline.

Images are turned into VQCs using several encoding strategies. For a proof-of-concept validation, we employ one-hot (OHE) and multi-hot (MHE) encodings, using five qubits per object. For the single-object stage, we used OHE to encode an image given the object. While MHE is used, in the relational learning, to encode each of the two objects present in the image and their spatial relation. In both cases, each distinct type of image is represented by a binary vector and mapped onto a computational basis state, for more details see [17].

To test for scalability on real world data, we use frozen CLIP image embeddings. These are encoded into VQCs using two standard techniques: amplitude and angle encoding. For amplitude encoding, the entire 512-dimensional CLIP embeddings are encoded in 9 qubits using the formula $\textstyle | \psi \rangle = \sum _ { i = 1 } ^ { 5 1 2 } x _ { i } | i \rangle$ To keep the depth of the angle-encoded circuits low, we use a PCA-reduced version of the CLIP embeddings such that they can be encoded into the rotation angles of a quantum circuit using 9 qubits. We also use a Collage encoding, where relational image circuits are constructed manually by combining their single-object circuits using entanglement. In order to compare with a classical baseline, we use a CLIP textprojection (CTP) baseline, replacing the quantum text encoder with a CLIP text encoder while keeping the CLIP image encoder frozen. Thus, only the text-side CLIP parameters are updated during training, and the parameter counts reported for the CLIP baseline in Tables II and III refer to trainable parameters rather than the total number of parameters in the full CLIP model.

In Stage 1, the model is trained on single-object image– caption pairs using a supervised contrastive loss to learn grounded object representations shared across modalities. In Stage 2, the parameters learned for each shape in the previous stage are used as fixed angles for the rotation gates on the quantum circuits and only the relational parameters (left, right) are optimized using an image-caption alignment. Contrastive loss is implemented by taking positive (similar) samples to correspond to matching image-caption pairs, and negative (dissimilar) samples by pairing an image with an incorrect caption.

![](images/60130a3935040913f625c32494c9c6379792dcbba5c92de4effe81f0abac627b.jpg)  
Fig. 1: A multistage procedure for compositional image–caption alignment. After optimizing the noun parameters during single-object training, the learned noun parameters are transferred to relational training.

## B. From Simulation to Hardware

During Stage 1, both the caption and image circuits produce pure quantum states and the similarity between them is computed using the fidelity between two pure states, as follows:

$$
F ( | \psi _ { C } \rangle , | \psi _ { I } \rangle ) = | \langle \psi _ { C } | \psi _ { I } \rangle | ^ { 2 } .
$$

During Stage 2, the sentence meanings are represented by the output s-wires of the quantum circuit. To isolate this output subsystem, the remaining qubits may either be post-selected or discarded. Since post-selection is prohibitively expensive on current quantum hardware, we discard the remaining qubits. Mathematically, this corresponds to taking the partial trace over the discarded subsystem, yielding the following reduced density operator:

$$
\rho _ { C } = \mathrm { T r } _ { \mathrm { d i s c a r d e d } } \left( \vert \Psi _ { C } \rangle \langle \Psi _ { C } \vert \right) .
$$

In contrast, no qubits are discarded from the image circuit. Consequently, the image output remains a pure quantum state |ψ<sub>I</sub>⟩, or equivalently, $\rho _ { I } = | \psi _ { I } \rangle \langle \psi _ { I } |$ . The similarity between the caption and image representations is quantified during simulation using the Uhlmann fidelity. Since one argument is the pure state projector the Uhlmann fidelity reduces to the following formula:

$$
F ( \rho _ { C } , \rho _ { I } ) = \langle \psi _ { I } \vert \rho _ { C } \vert \psi _ { I } \rangle = \operatorname { T r } ( \rho _ { C } \rho _ { I } )
$$

Figure 2 presents the two 4-qubit circuits used in Stage 1: a Sim4 circuit for the caption representation and an IQP circuit for the image representation. Applying the same procedure to these circuits requires eight qubits in total. Figure 4 illustrates the hardware implementation of the destructive SWAP test for the Stage 2 configuration. Unlike the standard SWAP test, this ancilla-free variant performs Bell-basis measurements on corresponding output-qubit pairs, thereby reducing the required qubit count. In this stage, the image branch retains a 4-qubit IQP architecture, whereas the caption branch expands to a 20-qubit circuit obtained by composing the subject noun, relation, and object noun according to the DisCoCat grammatical structure, as illustrated in Figure 3. The four retained output qubits of the caption circuit are paired with the image qubits, resulting in a 24-qubit hardware circuit.

TABLE II: Results of single-object training
<table><tr><td>Models</td><td>Method</td><td>Train (avg)</td><td>OOD Valid (avg)</td><td>OOD Test (avg)</td><td>Best OOD test</td><td>Parameters</td></tr><tr><td>OHE</td><td></td><td>92.20%</td><td>95.00%</td><td>93.34%</td><td>100%</td><td>168</td></tr><tr><td rowspan="2">CLIP</td><td>Angle Enc.</td><td>78.66%</td><td>75.00%</td><td>66.70%</td><td>70.31%</td><td>312</td></tr><tr><td>Amplitude Enc.</td><td>87.37%</td><td>79.71%</td><td>80.97%</td><td>100%</td><td>312</td></tr><tr><td>(Classical Baseline) CLIP</td><td>CTP</td><td>95.63%</td><td>90.45%</td><td>91.00%</td><td>95.18%</td><td>63.4M</td></tr></table>

TABLE III: Results of relational learning
<table><tr><td>Models</td><td>Method</td><td>Train (avg)</td><td>OOD Valid (avg)</td><td>OOD Test (avg)</td><td>Best OOD Test</td><td>Parameters</td></tr><tr><td>MHE</td><td>Binary</td><td>99.00%</td><td>73.00%</td><td>85.88%</td><td>91.66%</td><td>279</td></tr><tr><td>Collage Encoding</td><td>Angle Enc.</td><td>82.20%</td><td>61.05%</td><td>72.32%</td><td>75.00%</td><td>426</td></tr><tr><td rowspan="2">CLIP Multistage</td><td>Amplitude Enc.</td><td>62.28%</td><td>40.00%</td><td>48.58%</td><td>50.00%</td><td>426</td></tr><tr><td>Angle Enc.</td><td>76.60%</td><td>61.25%</td><td>55.33%</td><td>60.16%</td><td>480</td></tr><tr><td rowspan="2">CLIP Non Multistage</td><td>Amplitude Enc.</td><td>59.02%</td><td>45.00%</td><td>50.00%</td><td>50.00%</td><td>480</td></tr><tr><td>Angle Enc.</td><td>55.26%</td><td>44.50%</td><td>53.09%</td><td>55.00%</td><td>792</td></tr><tr><td></td><td>Amplitude Enc.</td><td>50.64%</td><td>62.00%</td><td>42.93%</td><td>48.33%</td><td>792</td></tr><tr><td>(Classical) CLIP</td><td>CTP</td><td>75.94%</td><td>11.14%</td><td>50.00%</td><td>50.00%</td><td>151.5M</td></tr></table>

![](images/b808274080361d9fc7d90767f9bcfa0bac7525f9fbfb90b1a530ca3f7e814fe4.jpg)  
Fig. 2: Four-qubit quantum circuits used in Stage 1 for singleobject image–caption alignment. The caption circuit (left) is constructed using the Sim4 ansatz with trainable parameterized rotations, while the image circuit (right) uses the IQP ansatz to angle-encode PCA-reduced CLIP image features. Both circuits produce 4-qubit pure states, whose fidelity defines the image–caption similarity during training and is subsequently estimated using the destructive SWAP test.

![](images/d5d82f7af258558f226c904163598c556027c0ce9213b7fe052962e054327b29.jpg)  
Fig. 3: Variational Quantum Circuit for the relational caption obtained by applying the Sim4 ansatz to the corresponding DisCoCat diagram. The retained s-type output qubits represent the sentence meaning, while the remaining qubits are discarded to obtain the reduced caption density matrix $\rho _ { C }$

## IV. SIMULATION EVALUATION

Before validating Mu-DisCoCat on hardware, we evaluate its performance via classical simulation on an image–caption alignment task using the predefined single-object and relational splits of the CoBi2 benchmark [21]. The benchmark consists of controlled synthetic images generated using the CLEVR rendering framework and depicts geometric shapes and spatial relations. In Stage 1, the task is to align singleobject images with their corresponding object captions. In Stage 2, the task is relational image–caption alignment, where matching image–caption pairs are treated as positive samples and mismatched captions as negative samples. The benchmark is designed for out-of-distribution (OOD) compositional evaluation by assigning different combinations of the underlying concepts to the training, OOD validation, and OOD test splits. In the relational setting, the individual shapes and spatial relations are observed during training, whereas their particular combinations in the validation and test splits are unseen. The model must therefore generalize by recombining previously learned concepts.

![](images/0951050ca38e6583e831ca43c32118295764ec854d7b403592f2c671979d8e71.jpg)  
Fig. 4: Hardware implementation of the destructive SWAP test. The retained output qubits of the caption circuit are compared with the corresponding output qubits of the image circuit using Bell-basis measurements. The measured outcomes are combined to estimate the Hilbert–Schmidt overlap $\operatorname { T r } ( \rho _ { C } \rho _ { I } )$ which equals the Uhlmann fidelity for a pure image state. The figure depicts the Stage 2 hardware configuration; Stage 1 follows the same procedure using two 4-qubit circuits.

Tables II and III summarize the performance of the original multistage models, while Table IV reports the simulation performance of the 4-qubit angle encoding model used for the subsequent hardware experiments. The motivation for introducing this reduced hardware-compatible model is discussed in Section V. All models were trained using the Adam optimizer, and the reported results are averaged over five random seeds. For the 4-qubit model, we use a learning rate of 0.009 and a batch size of 64.

Overall, our framework consistently improves relational compositional generalization while using substantially fewer trainable parameters than both the non-multistage quantum models and the classical CLIP baseline. For singleobject training, OHE achieves the highest quantum accuracy (93.34%), while amplitude encoding performs best among the CLIP-based quantum models (80.97%), using only 312 trainable parameters compared with 63.4M for CLIP. For relational learning, the classical CLIP baseline performs at chance level (50%), whereas all multistage quantum models achieve substantially higher accuracies using only a few hundred trainable parameters versus 151.5M for CLIP, with MHE obtaining the best overall result (85.88%) and the Collage encoding following suit at 72.32%. Among the CLIP-based quantum models, amplitude encoding was more effective for single objects, whereas angle encoding consistently yielded better relational generalization. It is important to note that the quantum models and the CLIP baseline differ substantially in model capacity. The comparison is therefore not parameter-matched and should not be interpreted as evidence of computational or quantum advantage. Rather, the results show that the proposed structured quantum models can achieve competitive, and in the relational OOD setting higher, compositional generalization performance with a much smaller trainable parameter space. This suggests that the compositional structure and multistage training strategy may provide a useful inductive bias for this task. However, the smaller number of trainable parameters does not necessarily imply lower computational cost on current quantum hardware, where circuit execution, sampling, and noise introduce additional overheads.

![](images/4fbab15e21260b1dbf12596812117c570feb7ca8e38504926999a6b5c8efeae8.jpg)  
(a) Amplitude-Encoding

![](images/6751abbfd946f3aadca20ae8a89247ec306c3c140b967171d9836819108b0832.jpg)  
(b) Angle-Encoding  
Fig. 5: Amplitude vs. angle encoding state fidelity on images from the test dataset. The desired pattern is block diagonal, indicating high similarity within class and lower similarity between classes. Captions are abbreviated as: cube right cone (cuRco), cylinder left cube (cyLcu), cylinder right cone (cyRco), sphere left cylinder (sLcy), sphere right cone (sRco), and sphere right cylinder (sRcy).

We hypothesize that the nonlinear rotations of angle encoding produce more separable relational representations, contributing to an improved performance (see Figure 5). Furthermore, multistage training consistently outperforms the corresponding non multistage models while reducing the number of trainable parameters (480 vs. 792), supporting the hypothesis that learning stable object representations before relational reasoning improves compositional generalization. Fine-tuning the CLIP image encoder did not improve the relational performance, consistent with previous findings [6].

Table IV shows that the reduced 4-qubit angle-encoding model maintains strong simulation performance. In particular, the relational model achieves 72.69% OOD test accuracy using only 210 trainable parameters, demonstrating that the reduced hardware-compatible model provides a suitable basis for the hardware validation presented in Section V.

## V. HARDWARE VALIDATION

In order to run Mu-DisCoCat on hardware, we need to make amendments. For text, we use DisCoCat circuits with discard, as described in Section III. For images, we focus on angle encoding because it uses a PCA-reduced CLIP representation and exhibits a clearer class-separation structure in the imagestate fidelity analysis compared with amplitude encoding (see

TABLE IV: Simulation Performance of the 4-qubit angle encoding model used for hardware validation.
<table><tr><td>Models</td><td>Method</td><td>Train (avg)</td><td>Valid (avg)</td><td>OOD Test (avg)</td><td>Best OOD Test</td><td>Parameters</td></tr><tr><td>Single-object training</td><td>Angle Enc.</td><td>94.19%</td><td>93.75%</td><td>93.00%</td><td>100.00%</td><td>132</td></tr><tr><td>Relational training</td><td>Angle Enc.</td><td>78.91%</td><td>77.46%</td><td>72.69%</td><td>73.84%</td><td>210</td></tr></table>

Figure 5). To keep the number of resources low, we train a 4-qubit angle-encoding model using a PCA-based image representation (see Table IV). The corresponding Stage 1 and Stage 2 hardware circuit configurations are as described in Section III-B. Consistent with the multistage training framework, the noun parameters learned during single-object training remain fixed during the relational stage, while the relation parameters are optimized.

For hardware evaluation, state fidelity cannot be accessed directly. This is because current quantum hardware returns measurement outcomes and repeated circuit executions yield bit strings sampled from the Born rule probability distribution. These measurement statistics do not directly provide the quantum fidelity used during simulation. Instead, the two quantum states must be compared jointly, which is achieved here using the destructive SWAP test [22], [23], which estimates the Hilbert–Schmidt overlap $\operatorname { T r } ( \rho _ { C } \rho _ { I } )$ . We know

$$
\mathrm { T r } ( \rho _ { C } \rho _ { I } ) = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \prod _ { i = 1 } ^ { 4 } ( - 1 ) ^ { z _ { i } ^ { ( k ) } x _ { i } ^ { ( k ) } } ,
$$

where N is the number of shots and $z _ { i } ^ { ( k ) } , x _ { i } ^ { ( k ) }$ denote the Bellmeasurement outcomes of the ith output-qubit pair in shot k. Therefore, the destructive SWAP test directly estimates the same similarity measure optimized during simulation, without requiring quantum state tomography.

The setting was tested on hardware by executing the above described circuits, after training, on the held-out test set of the benchmark. To evaluate the performance, we compared the simulated test fidelities with the hardware estimated destructive SWAP test overlaps. This allowed us to assess how faithfully the learned similarity relationships are preserved after deployment.

Figure 6 shows that for the single-object model, the destructive SWAP test overlaps closely follow the simulation fidelities across both quantum emulators and real hardware. Although hardware noise slightly increases the scatter around the ideal diagonal, similar and dissimilar image–caption pairs remain well separated, indicating that the learned similarity relationships are faithfully preserved.

For relational learning, Figure 7 illustrates excellent agreement between the destructive SWAP test estimates and the exact fidelities in the noiseless IBM Aer and noisy IBM FakeMarrakesh emulators. In contrast, agreement decreases on IBM Marrakesh, illustrating the impact of hardware noise and device constraints.

In Stage 2, slightly stronger agreement was observed for IBM FakeMontreal than for IBM FakeMarrakesh. This is consistent with the differences in the respective backend calibration. Although both transpiled circuits use 24 active qubits, the average transpiled depth is marginally higher for IBM FakeMarrakesh (365 versus 346), which may contribute to the accumulation of gate errors during execution.

Consistent with the figures, see Table V for the exact correlation degrees between simulation fidelities in the singleobject model across both quantum emulators and real hardware. The reported strong agreement is expected because Stage 1 compares two 4-qubit quantum states using an 8-qubit destructive SWAP test circuit, resulting in a relatively shallow and compact hardware implementation that is less susceptible to accumulated noise. In line with this observation, Table V also shows relatively small mean absolute errors (MAE) and only a modest reduction in centroid distance across the noisy backends, indicating that the similarity structure learned during training remains largely intact despite hardware imperfections.

The hardware validation results are consistently stronger for single-object training than for relational training. As described in Section III-B, Stage 1 compares two 4-qubit quantum states using an 8-qubit test circuit, whereas Stage 2 requires a 24- qubit hardware circuit due to the larger relational caption representation. This increase in circuit complexity is reflected in the transpiled circuits, whose average depth increases from 16 in Stage 1 to 365 in Stage 2. Consequently, the relational circuits are substantially more susceptible to noise accumulation during execution. Table VI also shows that increasing the number of shots improves agreement with the classical fidelities.

The IQM FakeAphrodite backend produces lower overlap values and weaker correlation than the IBM emulators. Inspection of the backend calibration data suggests that the stronger degradation is consistent with shorter median coherence times $( T _ { 1 } \approx 4 8 . 9$ us and $T _ { 2 } \approx 9 . 6 0 ~ \mathrm { u s } )$ together with higher gate (1-qubit gates: ≈ 0.001, 2-qubit gates: ≈ 0.015) and readout error rates (≈ 0.05) compared to the IBM backends. Nevertheless, all backends retain a positive correlation (Pearson r) with the classical fidelities. Naturally, executions on the real hardware yielded the lowest correlations but as plot (c) of Figure 7 shows, similar and dissimilar image-caption pairs remain distinguishable. Together, the cluster centroid distance and the MAE indicate that, although agreement with the simulated fidelities gradually decreases from noiseless simulation to real hardware, the separation between similar and dissimilar image-caption pairs is preserved.

## VI. CONCLUSIONS AND FUTURE WORK

We presented a theoretical framework for a multimodal compositional semantic model (Mu-DisCoCat), where meanings of words are represented in Hilbert Space and encoded as VQCs. Meanings of sentences are obtained by composing these using tensor contraction, encoded as Bell measurements in VQCs. Meanings of images are also VQCs, obtained by either directly encoding the image vectors as VQCs (CLIP, One-Hot, Multi-Hot), or by concatenating the VQCs of each object within the image (Collage). We provided a hardwarecompatible implementation of Mu-DisCoCat by replacing post-selection with discard and state overlap fidelity with the destructive SWAP test. The model was trained and tested on a Compositional Concept Generalisation (CoCoGen) benchmark. Classical simulations show better results over mainstream AI, in particular OpenAI’s CLIP. Although hardware noise reduced the quantitative agreement with the simulated fidelities, the framework kept a strong correlation with the simulated fidelities. In particular, similar and dissimilar imagecaption pairs remain clearly distinguishable across all evaluated hardware platforms, demonstrating that the proposed framework preserves the learned similarity relationships after deployment on current quantum processors. These results show that Mu-DisCoCat can preserve compositional similarity relationships when deployed on current quantum processors. This supports its practical deployment on current quantum hardware for future use on other similar applications.

![](images/77afbbc655a0f8f8940e2b718b562b05431bef614380ae6b1db581482289b9b2.jpg)  
(a) IBM Aer (Noiseless)

![](images/7523f5fdae88943dd604da8add71ea5cc6d07884af41030935e9f544657e62dc.jpg)  
(b) IBM FakeMarrakesh (Noisy)

![](images/b3c1429a8f11f308aac05776b80482c4d8d9950dd6e4b7d64f20a644a675794d.jpg)  
(c) IBM Marrakesh (Real Hardware)  
Fig. 6: Stage 1 simulated fidelity versus destructive SWAP test overlap across quantum emulators and hardware. Simulated fidelity is evaluated on the held-out test pairs using the classically trained model. Red points denote similar (correct) image– caption pairs, while blue points denote dissimilar (incorrect) image–caption pairs.

![](images/a05296d552023687371d1388e90e13bd0a2516ec9866054a9695a106209b258f.jpg)  
(a) IBM Aer (Noiseless)

![](images/e90953e518505ff5e1532ce95bcbd1273aba35eab0faa0d4a8eab6a4cfec97b1.jpg)  
(b) IBM FakeMarrakesh (Noisy)

![](images/79a12939929e983519cd8ccdf8674f614b6e257ac0ea18ad2970afb53616213c.jpg)  
(c) IBM Marrakesh (Real Hardware)  
Fig. 7: Stage 2 simulated fidelity versus destructive SWAP test overlap across quantum emulators and hardware. Simulated fidelity is evaluated on the held-out test pairs using the classically trained model. Red points denote similar (correct) image–caption pairs, while blue points denote dissimilar (incorrect) image–caption pairs.

TABLE V: Comparison of destructive SWAP test overlap estimates and classical quantum-state fidelities across 580 image caption pairs for single objects. Mean overlaps, Pearson correlation, centroid distance and mean absolute error are reported.
<table><tr><td>Configuration</td><td>Dissimilar</td><td>Similar</td><td>r with classical</td><td>Centroid Distance</td><td>MAE</td></tr><tr><td>Single-Object Training Model (Classical)</td><td> $0 . 2 9 8 \pm 0 . 1 5 5$ </td><td> $0 . 5 3 3 \pm 0 . 1 0 3$ </td><td></td><td>0.234</td><td></td></tr><tr><td>IBM Aer (Noiseless), 1000 shots</td><td> $0 . 2 9 7 \pm 0 . 1 0 7$ </td><td> $0 . 5 3 2 \pm 0 . 1 0 7$ </td><td>0.987</td><td>0.234</td><td>0.023</td></tr><tr><td>IBM Aer (Noiseless), 100 shots</td><td> $0 . 3 0 0 \pm 0 . 1 7 9$ </td><td> $0 . 5 3 6 \pm 0 . 1 3 4$ </td><td>0.888</td><td>0.235</td><td>0.073</td></tr><tr><td>IQM FakeAphrodite (Noisy), 1000 shots</td><td> $0 . 0 9 3 \pm 0 . 0 3 4$ </td><td> $0 . 1 0 6 \pm 0 . 0 3 2$ </td><td>0.294</td><td>0.012</td><td>0.316</td></tr><tr><td>IBM FakeMarrakesh (Noisy), 1000 shots</td><td> $0 . 2 6 1 \pm 0 . 1 2 9$ </td><td> $0 . 4 4 6 \pm 0 . 0 8 3$ </td><td>0.977</td><td>0.185</td><td>0.067</td></tr><tr><td>IBM Marrakesh (Real Hardware), 1000 shots</td><td> $0 . 2 8 2 \pm 0 . 1 2 4$ </td><td> $0 . 4 4 3 \pm 0 . 0 8 9$ </td><td>0.949</td><td>0.161</td><td>0.069</td></tr></table>

TABLE VI: Comparison of destructive SWAP test overlap estimates and classical quantum-state fidelities across 260 image– caption pairs for relational circuits. Mean overlaps, Pearson correlation, centroid distance and mean absolute error are reported.
<table><tr><td>Configuration</td><td>Dissimilar</td><td>Similar</td><td>r with classical</td><td>Centroid Distance</td><td>MAE</td></tr><tr><td>Relational Training Model (Classical)</td><td> $0 . 4 1 3 \pm 0 . 0 9 1$ </td><td> $0 . 5 9 4 \pm 0 . 1 1 3$ </td><td></td><td>0.181</td><td></td></tr><tr><td>IBM Aer (Noiseless), 1000 shots</td><td> $0 . 4 1 4 \pm 0 . 0 9 8$ </td><td> $0 . 5 9 2 \pm 0 . 1 1 8$ </td><td>0.979</td><td>0.178</td><td>0.023</td></tr><tr><td>IBM Aer (Noiseless), 100 shots</td><td> $0 . 4 3 2 \pm 0 . 1 2 9$ </td><td> $0 . 6 0 2 \pm 0 . 1 3 6$ </td><td>0.803</td><td>0.170</td><td>0.063</td></tr><tr><td>IBM FakeMontreal (Noisy), 1000 shots</td><td> $0 . 3 6 4 \pm 0 . 0 7 4$ </td><td> $0 . 4 9 4 \pm 0 . 0 9 2$ </td><td>0.968</td><td>0.130</td><td>0.076</td></tr><tr><td>IQM FakeAphrodite (Noisy), 1000 shots</td><td> $0 . 1 7 5 \pm 0 . 0 3 5$ </td><td> $0 . 2 0 6 \pm 0 . 0 3 8$ </td><td>0.585</td><td>0.031</td><td>0.313</td></tr><tr><td>IBM FakeMarrakesh (Noisy), 1000 shots</td><td> $0 . 3 3 6 \pm 0 . 0 7 0$ </td><td> $0 . 4 7 1 \pm 0 . 0 8 8$ </td><td>0.938</td><td>0.135</td><td>0.099</td></tr><tr><td>IBM Marrakesh (Real Hardware), 1000 shots</td><td> $0 . 3 1 2 \pm 0 . 0 8 6$ </td><td> $0 . 3 9 5 \pm 0 . 0 8 6$ </td><td>0.732</td><td>0.083</td><td>0.159</td></tr></table>

One shortcoming is that we only trained the text component and kept the visual representations fixed. This is due to the lack of a compositional distributional theory for images. Future work will focus on constructing such a theory, translating it into VQCs, and evaluating compositional concept generalization on larger real-world datasets. Another important direction is to investigate quantum error mitigation techniques to improve the robustness of destructive SWAP test measurements on near-term quantum hardware, as well as evaluating the proposed framework on future fault-tolerant quantum devices.

## REFERENCES

[1] J. A. Fodor and Z. W. Pylyshyn, “Connectionism and cognitive architecture: A critical analysis,” Cognition, vol. 28, no. 1–2, pp. 3–71, 1988.

[2] B. H. Partee, “Lexical semantics and compositionality,” An Invitation to Cognitive Science, vol. 1, pp. 311–360, 1995.

[3] P. Smolensky, “Tensor product variable binding and the representation of symbolic structures in connectionist systems,” Artificial Intelligence, vol. 46, no. 1–2, pp. 159–216, 1990.

[4] E. Pavlick, “Semantic structure in deep learning,” Annual Review of Linguistics, vol. 8, no. 1, pp. 447–471, 2022.

[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning. PMLR, Jul. 2021, pp. 8748–8763, iSSN: 2640-3498.

[6] M. Lewis, N. Nayak, P. Yu, J. Merullo, Q. Yu, S. Bach, and E. Pavlick, “Does CLIP bind concepts? probing compositionality in large image models,” in Findings of the Association for Computational Linguistics: EACL 2024, 2024, pp. 1487–1500.

[7] B. Pearson, M. Wray, and M. Lewis, “Diffusion models for improved compositional generalisation in vlms,” in The 2nd Workshop on What is Next in Multimodal Foundation Models?, 2024.

[8] B. Coecke, M. Sadrzadeh, and S. Clark, “Mathematical foundations for a compositional distributional model of meaning,” arXiv preprint arXiv:1003.4394, Mar. 2010. [Online]. Available: https://arxiv.org/abs/ 1003.4394

[9] M. Lewis, P. Smolensky, and M. Sadrzadeh, “Clip-binding: A systematic study of compositionality in vision–language models,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), 2022.

[10] S. Abramsky and B. Coecke, “A categorical semantics of quantum protocols,” arXiv preprint arXiv:quant-ph/0402130, 2007. [Online]. Available: https://arxiv.org/abs/quant-ph/0402130

[11] R. Lorenz, A. Pearson, K. Meichanetzidis, D. Kartsaklis, and B. Coecke, “Qnlp in practice: Running compositional models of meaning on a quantum computer,” arXiv preprint arXiv:2102.12846, 2021. [Online]. Available: https://arxiv.org/abs/2102.12846

[12] K. Meichanetzidis, A. Toumi, G. de Felice, and B. Coecke, “Grammaraware sentence classification on quantum computers,” Quantum Machine Intelligence, vol. 5, no. 1, Feb. 2023.

[13] S. Nazir and M. Sadrzadeh, “How does an adjective sound like? exploring audio phrase composition with textual embeddings,” in Proceedings of the 2024 CLASP Conference on Multimodality and Interaction in Language Learning. Association for Computational Linguistics, 2024, pp. 13–18.

[14] K. I. Lo, H. Hawashin, M. Abbaszadeh, T. G. Limback-Stokin, H. Wazni,¨ and M. Sadrzadeh, “Discoclip: A distributional compositional tensor network encoder for vision-language understanding,” in Proceedings of the 14th Joint Conference on Lexical and Computational Semantics (\*SEM 2025), L. Frermann and M. Stevenson, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 316–327. [Online]. Available: https://aclanthology.org/2025.starsem-1.25/

[15] T. G. Limback-Stokin, T. Birdavade, K. I. Lo, and M. Sadrzadeh,¨ “Meaning representations as variational quantum circuits,” in Proceedings of the 7th International Workshop on Designing Meaning Representations (DMR). European Language Resources Association (ELRA), 2026, pp. 113–123. [Online]. Available: https://lrec.elra.info/conference/2026/workshop/dmr

[16] H. Hawashin, M. Abbaszadeh, N. Joseph, B. Pearson, M. Lewis, and M. Sadrzadeh, “Compositional concept generalization with variational quantum circuits,” in Proceedings of the Conference IEEE Quantum AI, 2025. [Online]. Available: https://arxiv.org/abs/2509.09541

[17] M. Abbaszadeh, M. K. Moore, M. Sadrzadeh, and M. Lewis, “Quantum models with multi-stage training for compositional concept generalization,” in 2026 IEEE International Conference on Quantum Computing and Engineering (QCE), 2026, accepted; arXiv:2608.15601.

[18] B. Coecke, E. Grefenstette, and M. Sadrzadeh, “Lambek vs. lambek: Functorial vector space semantics and string diagrams for lambek calculus,” Annals of Pure and Applied Logic, vol. 164, no. 11, pp. 1079– 1100, 2013.

[19] V. Havl´ıcek, A. D. C ˇ orcoles, K. Temme, A. W. Harrow, A. Kandala,´ J. M. Chow, and J. M. Gambetta, “Supervised learning with quantumenhanced feature spaces,” Nature, vol. 567, no. 7747, pp. 209–212, 2019.

[20] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, “Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms,” arXiv preprint arXiv:1905.10876, May 2019. [Online]. Available: https://arxiv.org/abs/1905.10876

[21] B. Pearson, B. Boulbarss, M. Wray, and M. Lewis, “Evaluating compositional generalisation in vlms and diffusion models,” arXiv preprint arXiv:2508.20783, 2025. [Online]. Available: https://arxiv.org/ abs/2508.20783

[22] H. Buhrman, R. Cleve, J. Watrous, and R. de Wolf, “Quantum fingerprinting,” Physical Review Letters, vol. 87, no. 16, p. 167902, 2001.

[23] L. Cincio, Y. Subas¸ı, A. T. Sornborger, and P. J. Coles, “Learning the quantum algorithm for state overlap,” New Journal of Physics, vol. 20, p. 113022, 2018.