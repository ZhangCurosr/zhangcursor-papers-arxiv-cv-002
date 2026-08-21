# Structured Afinity for Unsupervised Visual Class-Incremental Memory in Deep Artificial Immune Networks

Siphesihle Sithungu

University of Johannesburg Johannesburg, South Africa siphesihles@uj.ac.za

## Abstract

Artificial immune networks (AINs) are naturally memory-forming systems, but conventional visual AINs often rely on flattened vector afinity that ignores spatial structure. This paper studies whether structured, gradient-free immune afinity can make Deep AINs viable as replay-free visual class-incremental representation-memory learners. Visual B-cells are formalized as structured templates, including shifted-template afinity, zero-normalized cross-correlation (ZNCC) filters, and feature-map binding profiles. A repertoire is treated both as memory and as a representation-inducing basis, while depth is obtained by passing binding-profile response maps to subsequent immune layers. The resulting Deep AIN exhibits adaptive latent coordinate reorganization: as new classes arrive, the binding-profile space evolves while retaining recoverable structure for earlier classes. Experiments on sklearn digits, MNIST, Fashion-MNIST, and KMNIST show that preserving response maps is critical. Scalar binding-profile variants underperform, whereas feature-map Deep AINs learn class-discriminative visual memory without replay, label-driven immune updates, or backpropagation through the immune layers. On sklearn digits, downstream probes fitted on the learned binding profiles reach 0.939 final balanced accuracy with logistic regression and 0.902 with 1-nearest-neighbour after all ten classes are encountered, with initial-class retention of 0.978. Adaptive layer-wise scale calibration further improves the two-layer feature-map Deep AIN to 0.978 balanced accuracy. With the same calibration rule, Fashion-MNIST reaches 0.814 and KMNIST reaches 0.853. These probes are external validation tools, not components of the AIN. The results identify structured afinity, response-map preservation, adaptive latent reorganization, and layer-wise scale calibration as key mechanisms for replay-free visual immune memory.

Keywords: artificial immune networks; class-incremental learning; structured afinity; ZNCC; visual memory; binding profiles

## 1 Introduction

Artificial immune networks are naturally suited to memory-centred learning: they encounter antigens, adapt repertoires, retain useful responses, and discriminate among previously encountered patterns. However, visual data exposes a limitation of many conventional AIN formulations. If an image is treated as a flattened vector, the afinity between an antigen and a B-cell ignores spatial locality, translation, and local visual structure. This can make the AIN appear weak even when the underlying dificulty is the afinity operator rather than immune memory itself.

This paper studies visual AINs from the standpoint of online memory formation. A class arrives, the repertoire forms memory for that class, later classes arrive, and the system must retain and discriminate among all encountered classes. At stream step t, the model has encountered classes $\mathcal { C } _ { t } ,$ and evaluation is performed only over C<sub>t</sub>. This setting directly tests whether an immune repertoire can acquire new visual experience while preserving earlier memory.

The central hypothesis is that AINs become viable visual online learners when afinity is made compatible with image structure. Instead of representing a B-cell only as a point in a flattened pixel space, a visual B-cell can be a structured template or local filter. Its response to an image can form part of a binding profile, and stacked binding profiles can define a Deep AIN in which later layers learn over earlier immune responses. This makes the repertoire both a memory population and a representation-inducing basis.

The paper makes five contributions:

1. Structured visual immune afinity. The paper defines image-aware B-cell afinity operators that remain gradient-free, including shifted-template afinity, ZNCC filters, and featuremap response profiles.

2. A visual class-incremental memory protocol. The paper evaluates AINs on sequential class acquisition, old-class retention, balanced accuracy over encountered classes, memory growth, and update cost.

3. Deep visual binding profiles. The paper defines depth as the composition of visual response maps and binding-profile maps, so later immune layers learn memories over earlier immune responses.

4. Empirical evidence for online binding-profile memory. The paper shows that an online feature-map Deep AIN can learn binding-profile memory spaces whose class structure is recoverable by simple downstream probes, without using labels to update the immune layers.

5. A stability-plasticity diagnosis for visual AINs. The paper separates memory-slot growth from lower-representation adaptation, showing when adapting lower visual filters helps and when it destabilizes later memory.

## 2 Similar Works

## 2.1 Artificial Immune Systems and Immune Memory

Artificial immune systems are computational models inspired by immune recognition, clonal expansion, mutation, suppression, memory formation, and self/non-self discrimination [4, 5, 9]. Immune network theory is particularly relevant to this paper because it views immune behaviour as an adaptive population process rather than as the optimization of a single global parameter vector [12]. Artificial immune networks inherit this view by maintaining a repertoire of antibodies or B-cells whose afinities determine stimulation, adaptation, survival, and suppression.

Several AIS families are related to the present work. Negative selection algorithms construct detectors for self/non-self discrimination [10]. Clonal selection algorithms adapt high-afinity candidates through cloning and mutation [7]. Artificial immune recognition systems such as AIRS use immune-inspired memory cells for supervised classification [26]. aiNet-style artificial immune networks use immune interaction and suppression for clustering, compression, and data analysis [6, 24]. These methods establish immune memory as a computational learning mechanism, but they usually operate over vector features and are not primarily formulated as layered visual representation learners. The present paper keeps the immune-memory interpretation, but asks how afinity must be structured when (1) the antigen is an image, (2) learning occurs in a class-incremental stream, and (3) the network should not only learn the feature space but must also learn a latent space over binding profiles.

## 2.2 Prototype, Kernel, and Dissimilarity Representations

The binding-profile representation used here is closely related to prototype and similarity-based learning. RBF networks represent an input by its response to a set of centres [1, 2]. Learning vector quantization and self-organizing maps use learned reference vectors to organize data and support classification [14, 15]. Dissimilarity representations describe an object by its dissimilarities to a representation set rather than by its original coordinates [21]. Nystr¨om methods and random Fourier features similarly construct useful feature spaces from landmark or kernel-induced responses [22, 27].

This paper does not claim that representing an object by responses to reference patterns is new. The contribution is the immune realization of this idea for online visual memory. The reference set is an adaptive B-cell repertoire, binding profiles are the response coordinates, and the representation is updated by immune operations rather than by gradient descent. This distinction matters in the class-incremental setting because memory formation and representation formation are coupled: adding or adapting B-cells changes both what is remembered and the coordinates in which future antigens are represented.

## 2.3 Structured Visual Matching

Flattening an image into a vector discards spatial locality and makes small translations appear as large coordinate changes. Classical computer vision addressed this problem using structured local matching, local descriptors, and translation-tolerant template comparison. Normalized crosscorrelation and its eficient variants are standard tools for template matching because they compare local patterns after compensating for mean and contrast diferences [17]. Local descriptor methods such as Scale-Invariant Feature Transform (SIFT) and Histogram of Oriented Gradients (HOG) further show that robust visual recognition often depends on local spatial structure rather than raw pixel-vector distance [3, 19].

Convolutional neural networks also exploit locality and weight sharing, and have become the dominant learned approach for visual recognition [11, 16]. The structured afinity method proposed here is related to convolution in the operational sense that a local B-cell filter is evaluated across image patches. However, it is not a neural convolutional layer: filters are B-cells, responses are immunonological afinities, and adaptation occurs through immune repertoire dynamics rather than backpropagation. The purpose is therefore not to replace convolutional networks as supervised visual classifiers, but to test whether immune repertoires can form useful online visual memory when their afinity operator respects image structure.

## 2.4 Online, Incremental, and Continual Visual Learning

The experimental setting is also related to online and continual learning. Continual learning studies how a learner can acquire new information without catastrophically forgetting previous knowledge [8, 20, 25]. Neural continual-learning methods include regularization approaches such as elastic weight consolidation [13], exemplar or replay-based methods such as Incremental Classifier on Representation Learning (iCaRL) [23], and constrained update methods such as gradient episodic memory [18]. These approaches usually target supervised predictive performance and often rely on stored examples, replay bufers, or gradient-based parameter updates.

The present paper is positioned diferently. It evaluates whether an AIN can learn an online visual representation-memory space without replay and without backpropagation through the immune layers. Downstream classifiers are used only as probes of the learned binding-profile space, analogous to representation evaluation protocols. The immune model itself is not trained to minimize classification loss. This makes the comparison to replay MLPs and refit baselines informative but not defining: those baselines indicate how much supervised rehearsal or refitting can buy, whereas the AIN result tests whether structured immune memory can produce a useful visual representation under sequential exposure.

## 2.5 Positioning of This Paper

The closest conceptual neighbourhood of this work is the intersection of AIN memory, prototyperesponse representations, structured visual matching, and continual learning. The paper’s specific claim is narrower than general visual classification and broader than a single afinity ablation. It argues that visual AINs become viable online memory learners when the afinity mechanism is made compatible with the structure of the antigen. In this view, a B-cell is not merely a point in a feature space; it can be a local visual template whose response contributes to a spatial binding profile. A Deep AIN then learns over those response profiles, allowing later immune layers to model patterns of earlier immune responses.

This positioning separates the proposed model from four adjacent lines of work. Unlike conventional AIS classifiers, the immune layers are treated as representation-memory learners rather than as a direct supervised classifier. Unlike ordinary prototype or kernel methods, the representation basis is formed by immune adaptation and can grow or change online. Unlike convolutional neural networks, the visual filters are not optimized by backpropagation. Unlike standard continuallearning baselines, the central object of study is the learned immune binding-profile space rather than a replay-trained discriminative model.

## 3 Model

The model is built around a simple immune-representation idea: a B-cell repertoire is both a memory population and a coordinate system. When an antigen is presented to a repertoire, it does not only stimulate individual B-cells; it also produces a pattern of responses across the ful repertoire. That response pattern is the antigen’s binding profile. A Deep AIN is obtained by feeding one layer’s binding-profile response into the next layer as its antigenic input.

Figure 1 shows the high-level construction. The first layer receives an image. Its B-cells respond to local visual structure and produce response maps. The second layer receives the stack of response maps and learns response-pattern memory. Further layers, if used, repeat the same principle: each layer receives the previous layer’s immune response, not an externally engineered feature vector.

## 3.1 B-Cell Repertoires and Binding Profiles

Let a repertoire at layer ℓ be

$$
B ^ { ( \ell ) } = \{ b _ { 1 } ^ { ( \ell ) } , \dots , b _ { m _ { \ell } } ^ { ( \ell ) } \} ,
$$

where each $b _ { i } ^ { ( \ell ) }$ is a B-cell template in that layer’s antigen space. For an input $u ^ { ( \ell - 1 ) }$ , the layer computes afinities between the input and every B-cell. In the scalar case, the binding profile is

$$
\Phi _ { \mathcal { B } ^ { ( \ell ) } } ( u ^ { ( \ell - 1 ) } ) = \left[ K ( u ^ { ( \ell - 1 ) } , b _ { 1 } ^ { ( \ell ) } ) , \dots , K ( u ^ { ( \ell - 1 ) } , b _ { m _ { \ell } } ^ { ( \ell ) } ) \right] .
$$

![](images/ccec270f1616afc4eec4833d416368afa8e281365e79c898423c013f31f740fb.jpg)  
Figure 1: High-level Deep AIN construction. Each layer transforms its antigenic input into a repertoire response. Later layers learn over earlier response patterns.

This vector is a coordinate representation induced by the repertoire. A B-cell therefore has two roles: it is a memory element, and it defines one coordinate of the binding-profile representation.

For non-visual data, this construction can be implemented with a standard RBF afinity,

$$
K _ { \mathrm { f l a t } } ( x , b ) = \exp \left( - \frac { \| x - b \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .
$$

For images, however, this flattened afinity is a weak inductive bias because it treats an image as a coordinate vector rather than as a spatial object. The rest of the model replaces this flat afinity with structured visual afinity.

## 3.2 Structured Visual Afinity

The first structured variant keeps B-cells as full-image templates but allows small translations before computing afinity. For image X and template B,

$$
K _ { \mathrm { s h i f t } } ( X , B ) = \operatorname* { m a x } _ { \Delta \in \mathcal { T } _ { r } } K _ { \mathrm { f l a t } } ( X , T _ { \Delta } B ) ,
$$

where $\mathcal { T } _ { r }$ is a local translation window and $T _ { \Delta }$ applies a zero-padded shift. This gives the B-cell limited translation tolerance while keeping the immune update mechanism unchanged.

The main visual model uses convolution-style ZNCC afinity. A B-cell can be a local $k \times k$ visual filter. For an image patch P and B-cell filter b, the local normalized correlation is

$$
\mathrm { Z N C C } ( P , b ) = \frac { \langle P - \bar { P } , b - \bar { b } \rangle } { \| P - \bar { P } \| \| b - \bar { b } \| } .
$$

The response is mapped to [0, 1]:

$$
K ( P , b ) = { \frac { \mathrm { Z N C C } ( P , b ) + 1 } { 2 } } .
$$

Sliding the B-cell over all valid image patches produces a response map. If the response is collapsed to a scalar, the afinity of filter b to image X is

$$
K _ { \mathrm { f i l t e r } } ( X , b ) = \operatorname* { m a x } _ { P \in \mathcal { P } _ { k } ( X ) } K ( P , b ) .
$$

This operation is convolution-style in the computer-vision sense, but it is not a neural convolutional layer. Filters are B-cells, responses are immune afinities, and the filters are adapted by immune operations rather than backpropagation.

![](images/2adba5ae36d519858b611f0d20d7837430b666d9905e157a46adbbe7362da566.jpg)  
Figure 2: Convolution-style ZNCC binding response. A local B-cell filter is evaluated over image patches, producing a response map. The response map may be preserved for deeper layers or pooled into a scalar afinity.

## 3.3 Feature-Map Deep AIN

The feature-map Deep AIN preserves the spatial response maps instead of immediately collapsing each B-cell response to a scalar. If layer 1 has $m _ { 1 }$ local visual B-cells, it maps an image to

$$
R ^ { ( 1 ) } ( X ) \in \mathbb { R } ^ { m _ { 1 } \times h _ { 1 } \times w _ { 1 } } .
$$

This tensor is a stack of immune response maps, one map per layer-1 B-cell. Layer 2 receives this response-map stack as its antigenic input. A layer-2 B-cell is therefore not an image patch; it is a filter over lower-layer response maps. With $m _ { 2 }$ layer-2 B-cells, the second response is

$$
R ^ { ( 2 ) } ( X ) \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } \times w _ { 2 } } .
$$

The validation embedding used in the experiments is

$$
z ( X ) = \mathrm { v e c } ( R ^ { ( 2 ) } ( X ) ) .
$$

More generally, a Deep AIN composes immune response maps:

$$
R ^ { ( \ell ) } ( X ) = \Phi _ { \mathcal { B } ^ { ( \ell ) } } ( R ^ { ( \ell - 1 ) } ( X ) ) , \qquad R ^ { ( 0 ) } ( X ) = X .
$$

The important point is that depth is not obtained by attaching a classifier to an AIN. Depth is obtained by making the previous immune response the next immune layer’s antigenic input.

## 3.4 Online Immune Updating

Each repertoire is updated online using immune operations: afinity evaluation, stimulation of responsive B-cells, clonal expansion, mutation, and suppression. The immune update does not use class labels for the feature-map layers. Labels are used only by the external downstream validation probes after embeddings have been produced. This separation is essential: the AIN learns a visual memory representation, and the probe measures whether that representation contains recoverable class structure.

When a new class batch arrives, the lower visual repertoire may adapt, and the binding-profile coordinate system may change. This adaptive coordinate reorganization is a core property of the model: the current repertoire defines the current coordinate system. Earlier samples can be reembedded through the updated repertoire to determine whether the evolved immune memory still represents previous classes. Successful retention therefore does not require stale coordinates to remain fixed; it requires earlier classes to remain recoverable in the current binding-profile space.

## 3.5 End-to-End Pipeline

Figure 3 summarizes the full pipeline used in the experiments. Incoming visual batches update the immune repertoires. At evaluation time, encountered training and test samples are embedded through the current Deep AIN. External probes are then fitted on the current training embeddings and evaluated on current test embeddings. The probes validate the learned immune representation; they are not part of the immune update.

![](images/06a05858b2fc4424bed6b808a087f3f5c3f647d1149fc19785ce4a9d65dc7c84.jpg)  
Labels are used only by the validation probe after embeddings are produced; they do not update the immune layers.  
Figure 3: Experimental pipeline. The immune layers are updated online from incoming batches. External probes are fitted only after embeddings are produced, to evaluate the current bindingprofile space.

## 4 Experiment Setup

## 4.1 Datasets

The experiments use four grayscale visual benchmarks: sklearn digits, MNIST, Fashion-MNIST, and KMNIST. Sklearn digits provides $8 \times 8$ handwritten digit images and is used for the most detailed method comparison, timing analysis, trajectory analysis, and depth calibration. MNIST, Fashion-MNIST, and KMNIST provide $2 8 \times 2 8$ images and are used to test whether the same feature-map Deep AIN protocol transfers beyond the small sklearn digits setting. MNIST evaluates handwritten digit memory at a larger scale; Fashion-MNIST changes the visual domain to clothing categories; KMNIST introduces more complex character shapes and is used to diagnose the geometry of the learned binding-profile space.

All datasets are treated as class-incremental streams. The first class is used to initialize memory, and the remaining classes are introduced one at a time. The model is evaluated only on classes that have already been encountered. This focuses the experiment on the intended question: whether an immune repertoire can acquire, retain, and organize encountered visual classes as the stream grows.

## 4.2 Class-Incremental Protocol

Let $\mathcal { C } _ { t }$ denote the set of classes encountered by stream step t. The protocol is:

Algorithm 1 Class-incremental visual memory protocol   
Require: Initial class set $\mathcal { C } _ { 0 } .$ , stream classes $( c _ { 1 } , \ldots , c _ { T } )$ , initial samples $X _ { 0 } ,$ stream batches   
$( X _ { 1 } , \ldots , X _ { T } )$ , test set $X$ test   
1: Initialize immune memory $M _ { 0 }$ using $X _ { 0 }$   
2: Set encountered classes $\mathcal { C } _ { \mathrm { s e e n } }  \mathcal { C } _ { 0 }$   
3: Evaluate $M _ { 0 }$ on test samples with labels in $\mathcal { C } _ { \mathrm { s e e n } }$   
4: for $t = 1$ to $T$ do   
5: Receive new class batch $X _ { t }$ from class $c _ { t }$   
6: Update immune memory: $M _ { t } \gets$ ImmuneUpdate $( M _ { t - 1 } , X _ { t } )$   
7: Update encountered classes: $\mathcal { C } _ { \mathrm { s e e n } }  \mathcal { C } _ { \mathrm { s e e n } } \cup \{ c _ { t } \}$   
8: Embed encountered training samples through current memory $M _ { t }$   
9: Embed encountered test samples through current memory $M _ { t }$   
10: Fit external validation probe on current training embeddings   
11: Evaluate over test samples with labels in $\mathcal { C } _ { \mathrm { s e e n } }$   
12: end for

Unless otherwise stated, results are averaged over three random seeds. The sklearn digits, MNIST, Fashion-MNIST, and KMNIST class-incremental runs use the same stream structure: digit or class 0 initializes memory, and classes 1 through 9 arrive sequentially. The reported final-step results therefore evaluate performance after all ten classes have been encountered. The standard stream uses 80 initial samples from the first class, 60 samples per new class event, and 35 held-out test samples per class.

## 4.3 Feature-Map Deep AIN Configuration

The main proposed model is the feature-map Deep AIN. Layer 1 contains local visual B-cell filters. For a grayscale image X, each layer-1 B-cell produces a ZNCC response map by sliding over local image patches. Layer 2 receives the stack of layer-1 response maps as its antigenic input and learns response-map filters over that stack. The final representation used for validation is

$$
z ( X ) = \mathrm { v e c } ( R ^ { ( 2 ) } ( X ) ) .
$$

The default two-layer feature-map model uses 30 layer-1 filters and 30 layer-2 filters. For $2 8 \times 2 8$ datasets, the first-layer filter size is $5 \times 5$ and the second-layer response-map filter size is $2 \times 2 .$ , producing a final binding-profile embedding of dimension 15870. For sklearn digits, the same conceptual architecture is applied to $8 \times 8$ images, producing a smaller final representation. Immune updates use clonal selection style repertoire training with mutation and suppression; class labels are not used to update the immune layers.

Higher-layer afinity scales are important because response-map and binding-profile spaces do not have the same distance scale as the original image space. The main cross-dataset feature-map experiments therefore use an adaptive scale rule based on nearest-neighbour distances in the relevant response-profile space. The scale estimate is multiplied by a fixed factor and kept consistent across datasets in the main cross-dataset comparison. Manual fixed scales are used in the depth-calibration experiment to demonstrate the efect of scale mismatch.

## 4.4 Validation Probes

The feature-map Deep AIN is evaluated as an unsupervised online immune representation-memory learner. After each stream event, encountered training samples and encountered test samples are passed through the current immune layers to obtain binding-profile embeddings. External probes are then fitted on the encountered training embeddings and evaluated on the encountered test embeddings.

The primary probes are logistic regression and 1-nearest-neighbour. Logistic regression tests whether the learned immune representation is approximately linearly separable. The 1-nearestneighbour probe tests local class coherence in the binding-profile space. For KMNIST, additional probes are used to diagnose geometry: nearest-centroid probing tests compact global class means, and PCA+RBF-SVM tests whether a simple nonlinear boundary improves after dimensionality reduction. These probes are validation tools, not components of the AIN. They do not update the immune repertoire and should be interpreted as measurements of the representation produced by the AIN.

This validation protocol also clarifies how adaptive coordinate reorganization is handled. The immune coordinate system may change as new classes arrive. The experiments therefore do not reuse stale embeddings. Instead, all encountered samples are re-embedded through the current AIN at each evaluation step. This tests whether the current immune memory space still represents earlier classes after online adaptation.

## 4.5 Baselines

The baseline set separates non-updating, online-updating, refit, supervised-prototype, and neuralreplay behaviours:

• Static k-means: k-means fitted to the initial class only and never updated.

• Mini-batch k-means: k-means centroids updated online with incoming class batches.

• Full refit k-means: k-means refitted on all accumulated encountered data after each class event.

• Online prototype memory: a supervised nearest-prototype memory updated toward samples from the corresponding class.

• No-replay MLP: a neural classifier updated on the current batch only.

• Replay MLP: a neural classifier trained with a replay bufer of previous examples.

The comparison also includes scalar binding-profile AIN variants in the static diagnostic: flat RBF AIN, shifted-template RBF AIN, convolution-style ZNCC AIN, and multi-ZNCC AIN. These variants test whether changing the afinity operator in the first place improves the AIN response space before moving to feature-map response preservation.

## 4.6 Evaluation Metrics

The primary metric is balanced accuracy over encountered classes. This is preferred over ordinary accuracy because class-incremental streams can produce uneven dificulty across old and new classes. Initial retention measures accuracy on the initial class after later classes have arrived. Current-class accuracy measures acquisition of the most recently introduced class. These two metrics separate stability from plasticity: a method can learn the new class while forgetting the first class, or retain the first class while failing to acquire the new class.

The experiments also report representation dimension or memory size. For prototype and kmeans methods, this is the number of memory slots or centroids. For feature-map Deep AIN probes, it is the dimensionality of the final binding-profile embedding z(X) used by the external probe. Runtime is reported for the sklearn digits protocol as an implementation-level diagnostic, separating immune memory training, probe fitting, total training time, and inference time.

## 5 Results

## 5.1 Static Representation Diagnostic

Static multiclass classification is not the main task, but it verifies whether image-aware afinity improves the health of the AIN response space. Table 1 shows that structured afinity (all the methods shown in Table 1 except the Flat RBF AIN) substantially improves digit recognition on sklearn digits. The MNIST feature-map result is lower but confirms that the representation is not limited to the small 8 × 8 setting.

Table 1: Multiclass digit diagnostics. Sklearn digits rows report accuracy and balanced accuracy. The MNIST feature-map row reports multiclass accuracy over a 350-example-per-class subset.
<table><tr><td>Dataset</td><td>Method</td><td>Accuracy</td><td>Balanced accuracy</td></tr><tr><td>Sklearn digits</td><td>Flat RBF AIN</td><td>0.849</td><td>0.848</td></tr><tr><td>Sklearn digits</td><td>Shifted-template RBF AIN</td><td>0.917</td><td>0.917</td></tr><tr><td>Sklearn digits</td><td>Convolution-style ZNCC AIN</td><td>0.931</td><td>0.931</td></tr><tr><td>Sklearn digits</td><td>Feature-map Deep AIN</td><td>0.850</td><td></td></tr><tr><td>MNIST</td><td>Feature-map Deep AIN</td><td>0.731</td><td></td></tr></table>

## 5.2 Class-Incremental Visual Memory

Table 2 reports the final step of a sklearn digits class-incremental stream. The model starts from digit 0 and then receives digits 1 through 9 sequentially. Evaluation at the final step is over all ten encountered classes. The last two rows (Feature-map Deep AIN + logistic probe and Feature-map Deep AIN + 1NN probe) are downstream probes on the learned Deep AIN binding-profile space.

The online feature-map AIN learns a binding-profile memory space in which simple probes recover a strong class structure after sequential exposure. This is evidence about the learned immune memory itself. Full refit k-means and replay MLP remain relevant upper baselines because they reuse old data to reconstruct or rehearse the decision space. The probe result answers whether the immune layers themselves produce a useful online visual memory space. The answer is positive in this diagnostic, especially because both a linear probe and a non-parametric 1NN probe remain strong after all ten digit classes have arrived. For prototype-based rows, the final column reports the number of prototype or centroid memory slots. For feature-map Deep AIN rows, it reports the dimensionality of the final binding-profile embedding z(X) used by the external probe, not the total number of B-cells across layers.

Table 2: Final ten-class performance in the sklearn digits class-incremental visual-memory stream. Values are averaged over three seeds. Probe rows externally validate the learned Deep AIN bindingprofile memory.
<table><tr><td>Method</td><td>Evaluation mode</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td><td>Memory size / embedding dimension</td></tr><tr><td>Static k-means</td><td>no update</td><td>0.100</td><td>1.000</td><td>0.000</td><td>80</td></tr><tr><td>Mini-batch k-means</td><td>online centroid update</td><td>0.617</td><td>0.978</td><td>0.678</td><td>80</td></tr><tr><td>Full refit k-means</td><td>refit on seen data</td><td>0.928</td><td>1.000</td><td>0.867</td><td>120</td></tr><tr><td>Online prototype memory</td><td>supervised online prototypes</td><td>0.872</td><td>1.000</td><td>0.800</td><td>120</td></tr><tr><td>No-replay MLP</td><td>current batch only</td><td>0.827</td><td>0.922</td><td>0.767</td><td></td></tr><tr><td>Replay MLP</td><td>replay classifier</td><td>0.953</td><td>0.989</td><td>0.933</td><td></td></tr><tr><td>Feature-map Deep AIN + logistic probe</td><td>external probe on z(X)</td><td>0.939</td><td>0.978</td><td>0.933</td><td>270</td></tr><tr><td>Feature-map Deep AIN + 1NN probe</td><td>external probe on z(X)</td><td>0.902</td><td>0.978</td><td>0.833</td><td>270</td></tr></table>

Table 3 reports training and inference timing for the same sklearn digits protocol. These values are intended as implementation-level diagnostics rather than hardware-independent complexity claims. For ordinary methods, training time includes initial fitting and all online updates. For feature-map Deep AIN rows, memory-training time reports immune-layer initialization and updates, while probe-fitting time reports the external validation probe fitted after each event.

Table 3: Training and inference time for the sklearn digits Table 2 protocol. Values are seconds averaged over three seeds. Inference time is the total time across all ten evaluations.
<table><tr><td>Method</td><td>Memory train</td><td>Probe fit</td><td>Total train</td><td>Total inference</td><td>Train/event</td><td>Inference/eval</td></tr><tr><td>Online prototype memory</td><td>0.008</td><td>0.000</td><td>0.008</td><td>0.129</td><td>0.001</td><td>0.0129</td></tr><tr><td>Static k-means</td><td>0.081</td><td>0.000</td><td>0.081</td><td>0.088</td><td>0.008</td><td>0.0088</td></tr><tr><td>Mini-batch k-means</td><td>0.165</td><td>0.000</td><td>0.165</td><td>0.083</td><td>0.017</td><td>0.0083</td></tr><tr><td>No-replay MLP</td><td>0.554</td><td>0.000</td><td>0.554</td><td>0.007</td><td>0.055</td><td>0.0007</td></tr><tr><td>Replay MLP</td><td>0.954</td><td>0.000</td><td>0.954</td><td>0.005</td><td>0.095</td><td>0.0005</td></tr><tr><td>Full refit k-means</td><td>1.525</td><td>0.000</td><td>1.525</td><td>0.126</td><td>0.152</td><td>0.0126</td></tr><tr><td>Feature-map Deep AIN + 1NN probe</td><td>4.385</td><td>0.008</td><td>4.393</td><td>0.022</td><td>0.439</td><td>0.0022</td></tr><tr><td>Feature-map Deep AIN + logistic probe</td><td>4.385</td><td>1.111</td><td>5.496</td><td>0.008</td><td>0.550</td><td>0.0008</td></tr></table>

Table 4 repeats the same class-incremental protocol on a diferent dataset: MNIST. This is a more dificult visual setting because the images are larger and more variable. The feature-map Deep AIN again refers to the unsupervised immune layers; logistic regression and 1NN are externa probes fitted on the learned binding-profile memory.

The MNIST result sharpens the interpretation. The no-replay MLP achieves high current-class accuracy but loses the initial class almost completely, which is the expected forgetting pattern when no old data is rehearsed. Replay MLP avoids this collapse. The feature-map Deep AIN bindingprofile memory is competitive with replay MLP under external probing, with the 1NN probe slightly higher in balanced accuracy and initial retention in this diagnostic. This supports the claim that the immune layers are learning a stable and coherent visual memory space. As in Table 2, the final column reports memory slots for prototype rows and final binding-profile dimensionality for feature-map Deep AIN rows.

Table 5 repeats the protocol on Fashion-MNIST using the same feature-map architecture and the same adaptive layer-wise afinity scale rule. This is an important transfer test because the images are still grayscale $2 8 \times 2 8$ , but the visual categories are clothing classes rather than digit strokes.

Table 4: Final ten-class performance in the MNIST class-incremental visual-memory stream. Values are averaged over three seeds. Probe rows externally validate the learned Deep AIN binding-profile
<table><tr><td>Method</td><td>Evaluation mode</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td><td>Memory size / embedding dimension</td></tr><tr><td>Static k-means</td><td>no update</td><td>0.100</td><td>1.000</td><td>0.000</td><td>80</td></tr><tr><td>Mini-batch k-means</td><td>online centroid update</td><td>0.329</td><td>0.867</td><td>0.067</td><td>80</td></tr><tr><td>Full refit k-means</td><td>refit on seen data</td><td>0.699</td><td>0.933</td><td>0.756</td><td>120</td></tr><tr><td>Online prototype memory</td><td>supervised online prototypes</td><td>0.779</td><td>0.911</td><td>0.733</td><td>120</td></tr><tr><td>No-replay MLP</td><td>current batch only</td><td>0.444</td><td>0.100</td><td>0.967</td><td></td></tr><tr><td>Replay MLP</td><td>replay classifier</td><td>0.851</td><td>0.900</td><td>0.856</td><td></td></tr><tr><td>Feature-map Deep AIN + logistic probe</td><td>external probe on z(X)</td><td>0.851</td><td>0.900</td><td>0.911</td><td>15870</td></tr><tr><td>Feature-map Deep AIN + 1NN probe</td><td>external probe on z(X)</td><td>0.860</td><td>0.933</td><td>0.878</td><td>15870</td></tr></table>

Table 5: Final ten-class performance in the Fashion-MNIST class-incremental visual-memory stream. Values are averaged over three seeds. Feature-map rows use the same adaptive layerwise afinity scale rule.
<table><tr><td>Method</td><td>Evaluation mode</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td><td>Memory size / embedding dimension</td></tr><tr><td>Static k-means</td><td>no update</td><td>0.100</td><td>1.000</td><td>0.000</td><td>80</td></tr><tr><td>Mini-batch k-means</td><td>online centroid update</td><td>0.469</td><td>0.700</td><td>0.767</td><td>80</td></tr><tr><td>Full refit k-means</td><td>refit on seen data</td><td>0.649</td><td>0.733</td><td>0.900</td><td>120</td></tr><tr><td>Online prototype memory</td><td>supervised online prototypes</td><td>0.651</td><td>0.833</td><td>0.844</td><td>120</td></tr><tr><td>No-replay MLP</td><td>current batch only</td><td>0.298</td><td>0.000</td><td>1.000</td><td></td></tr><tr><td>Replay MLP</td><td>replay classifier</td><td>0.780</td><td>0.756</td><td>0.933</td><td></td></tr><tr><td>Feature-map Deep AIN + logistic probe</td><td>external probe on z(X)</td><td>0.814</td><td>0.811</td><td>0.922</td><td>15870</td></tr><tr><td>Feature-map Deep AIN + 1NN probe</td><td>external probe on z(X)</td><td>0.801</td><td>0.811</td><td>0.944</td><td>15870</td></tr></table>

The Fashion-MNIST result is important because it was not obtained by selecting a new datasetspecific RBF sigma. The same adaptive layer-wise afinity rule produces a binding-profile space that is competitive with replay MLP and stronger than the rest of the baselines in this stream. Therefore, the feature-map Deep AIN does not just demonstrate single-dataset superiority; instead, it demonstrates general competitiveness under a fixed operating protocol.

Table 6 makes a direct comparison against replay MLP across the current visual datasets using the fixed adaptive scale rule. The result supports an important claim: a replay-free, gradient-free feature-map Deep AIN learns binding-profile memory spaces that are consistently competitive with a replay-based neural classifier under the same class-incremental protocol. Replay MLP retains old information by storing and rehearsing previous samples while the Deep AIN instead retains information through immune repertoires and the response-profile representation induced by those repertoires. KMNIST also reveals a useful diagnostic distinction: the 1NN probe is much stronger than the logistic probe, suggesting that the learned memory space can be locally class-coherent even when it is not linearly organized.

The KMNIST discrepancy between logistic probing and 1NN probing motivates a more direct probe-geometry diagnostic. Table 7 compares probes that test diferent forms of structure in the same learned binding-profile space. Nearest-centroid probing tests whether each class is represented by a compact global mean; logistic probing tests linear separability; PCA+RBF-SVM tests a simple nonlinear boundary after dimensionality reduction; and 1NN tests local neighbourhood coherence.

This pattern supports the interpretation that the Deep AIN representation is meaningful but not necessarily globally linear. If the binding-profile space were simply failing, all probes would be weak. Instead, nonlinear and local probes substantially exceed linear and centroid probes. The evidence therefore suggests that feature-map Deep AINs can form locally coherent visual memory spaces whose quality may be underestimated by linear probes alone.

Table 6: Cross-dataset comparison of replay MLP and feature-map Deep AIN binding-profile probes. Values are final ten-class performance averaged over three seeds. Deep AIN rows use the same adaptive layer-wise afinity scale rule on every dataset.
<table><tr><td>Dataset</td><td>Method</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td></tr><tr><td>Sklearn digits</td><td>Replay MLP</td><td>0.946</td><td>0.989</td><td>0.944</td></tr><tr><td>Sklearn digits</td><td>Feature-map Deep AIN + logistic probe</td><td>0.943</td><td>0.978</td><td>0.933</td></tr><tr><td>Sklearn digits</td><td>Feature-map Deep AIN + 1NN probe</td><td>0.952</td><td>1.000</td><td>0.944</td></tr><tr><td>MNIST</td><td>Replay MLP</td><td>0.858</td><td>0.911</td><td>0.944</td></tr><tr><td>MNIST</td><td>Feature-map Deep AIN + logistic probe</td><td>0.862</td><td>0.900</td><td>0.856</td></tr><tr><td>MNIST</td><td>Feature-map Deep AIN + 1NN probe</td><td>0.857</td><td>0.922</td><td>0.889</td></tr><tr><td>Fashion-MNIST</td><td>Replay MLP</td><td>0.780</td><td>0.756</td><td>0.933</td></tr><tr><td>Fashion-MNIST</td><td>Feature-map Deep AIN + logistic probe</td><td>0.814</td><td>0.811</td><td>0.922</td></tr><tr><td>Fashion-MNIST</td><td>Feature-map Deep AIN + 1NN probe</td><td>0.801</td><td>0.811</td><td>0.944</td></tr><tr><td>KMNIST</td><td>Replay MLP</td><td>0.757</td><td>0.889</td><td>0.811</td></tr><tr><td>KMNIST</td><td>Feature-map Deep AIN + logistic probe</td><td>0.711</td><td>0.811</td><td>0.644</td></tr><tr><td>KMNIST</td><td>Feature-map Deep  $\mathrm { A I N } + 1 \mathrm { N N }$  probe</td><td>0.853</td><td>0.933</td><td>0.844</td></tr></table>

Table 7: KMNIST probe-geometry diagnostic on the same learned feature-map Deep AIN bindingprofile space. Values are final ten-class performance averaged over three seeds.
<table><tr><td>Probe</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td></tr><tr><td>Nearest centroid</td><td>0.612</td><td>0.811</td><td>0.589</td></tr><tr><td>Logistic regression</td><td>0.711</td><td>0.811</td><td>0.644</td></tr><tr><td> $\mathrm { P C A } + \mathrm { R B F } { \cdot } \mathrm { S V M }$ </td><td>0.818</td><td>0.911</td><td>0.778</td></tr><tr><td>1-nearest-neighbour</td><td>0.853</td><td>0.933</td><td>0.844</td></tr></table>

Table 8 shows the trajectory of the learned binding-profile space as classes are added.

Table 8: External probe trajectory on online feature-map Deep AIN binding profiles in the sklearn digits class-incremental stream. Values are averaged over three seeds.
<table><tr><td>Step</td><td>Encountered classes</td><td>LogReg balanced accuracy</td><td>1NN balanced accuracy</td><td>LogReg initial retention</td><td>1NN initial retention</td></tr><tr><td>0</td><td>{0}</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>1</td><td>{0,1}</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>2</td><td>{0, 1, 2}</td><td>0.981</td><td>0.985</td><td>1.000</td><td>1.000</td></tr><tr><td>3</td><td> $\{ 0 , 1 , 2 , 3 \}$ </td><td>0.953</td><td>0.944</td><td>0.989</td><td>1.000</td></tr><tr><td>4</td><td> $\{ 0 , 1 , 2 , 3 , 4 \}$ </td><td>0.971</td><td>0.962</td><td>0.989</td><td>0.978</td></tr><tr><td>5</td><td> $\{ 0 , \ldots , 5 \}$ </td><td>0.957</td><td>0.963</td><td>0.989</td><td>0.989</td></tr><tr><td>6</td><td> $\{ 0 , \ldots , 6 \}$ </td><td>0.951</td><td>0.944</td><td>0.989</td><td>0.989</td></tr><tr><td>7</td><td> $\{ 0 , \ldots , 7 \}$ </td><td>0.956</td><td>0.938</td><td>0.989</td><td>0.989</td></tr><tr><td>8</td><td> $\{ 0 , \ldots , 8 \}$ </td><td>0.943</td><td>0.905</td><td>0.989</td><td>0.989</td></tr><tr><td>9</td><td> $\{ 0 , \ldots , 9 \}$ </td><td>0.939</td><td>0.902</td><td>0.978</td><td>0.978</td></tr></table>

The decline in balanced accuracy as more classes arrive is expected: the representation task becomes harder with each new class. The key observation is that both probes remain strong and initial-class retention remains high, which indicates that the online immune layers have not collapsed the earlier class structure. The most dificult transition in this run occurs around digit 8, suggesting that class-specific visual overlap should be diagnosed per class rather than only by final average accuracy.

## 5.3 Depth and Layer-Wise Scale Calibration

Depth is useful only if later layers receive response spaces at a workable afinity scale. The featuremap AIN therefore requires a layer-wise scale choice for immune updates in binding-profile spaces. Table 9 summarizes the final ten-class sklearn digits result for one-, two-, and three-layer featuremap AINs. The depth-one model uses the first response maps directly as the validation embedding. Depth two and depth three learn over response-map spaces. The manually calibrated rows use fixed higher-layer training scales; the adaptive rows estimate the nearest-neighbour distance scale in the binding-profile space and multiply it by a scale factor.

Table 9: Depth and layer-wise scale calibration in the sklearn digits class-incremental stream. Values are final-step means over three seeds. Probe rows externally validate the learned bindingprofile memory.
<table><tr><td>Depth</td><td>Scale setting</td><td>Probe</td><td>Balanced accuracy</td><td>Initial retention</td><td>Current-class accuracy</td><td>Dimension</td><td>Interpretation</td></tr><tr><td>1</td><td>direct layer-1 maps</td><td>LogReg</td><td>0.977</td><td>1.000</td><td>0.989</td><td>480</td><td>strong shallow visual response space</td></tr><tr><td>1</td><td>direct layer-1 maps</td><td>1NN</td><td>0.982</td><td>1.000</td><td>0.978</td><td>480</td><td>strong shallow visual response space</td></tr><tr><td>2</td><td>fixed  $\sigma _ { 2 } = 1 . 0$ </td><td>LogReg</td><td>0.953</td><td>0.978</td><td>0.967</td><td>270</td><td>useful but under-scaled depth</td></tr><tr><td>2</td><td>fixed  $\sigma _ { 2 } = 2 . 0$ </td><td>LogReg</td><td>0.978</td><td>0.978</td><td>1.000</td><td>270</td><td>best calibrated two-layer result</td></tr><tr><td>2</td><td>adaptive NN scale ×6</td><td>LogReg</td><td>0.968</td><td>0.989</td><td>0.967</td><td>270</td><td>principled scale rule, near manual best</td></tr><tr><td>3</td><td>fixed σ2 = 2.0, σ3 = 2.0</td><td>LogReg</td><td>0.924</td><td>0.989</td><td>0.933</td><td>120</td><td>third layer improves with scale but compresses strongly</td></tr><tr><td>3</td><td>adaptive NN scales ×6, ×6</td><td>LogReg</td><td>0.927</td><td>0.989</td><td>0.944</td><td>120</td><td>best current three-layer result</td></tr></table>

The result is useful in two ways. First, it shows that poor higher-layer performance can be caused by scale mismatch rather than by depth itself. Increasing the second-layer training scale from 1.0 to 2.0 raises the two-layer logistic-probe result from 0.953 to 0.978 balanced accuracy. Second, it prevents an overclaim: deeper is not automatically better. A third layer retains the initial class well but compresses the representation from 270 to 120 dimensions and remains weaker than the calibrated two-layer model. This indicates that additional depth requires matched capacity and receptive-field choices rather than depth alone.

## 6 Discussion

The results support the central claim of the paper: visual AINs require structured afinity. The weakest results occur when images are treated as flattened vectors and compared by ordinary RBF afinity. This is not surprising, because a flattened representation makes the afinity score sensitive to pixel coordinate changes rather than to visual similarity. The improvement obtained by shiftedtemplate matching and convolution-style ZNCC therefore has a clear interpretation. It shows that the first obstacle for visual AINs is not necessarily the immune learning mechanism itself, but the mismatch between image structure and the afinity operator. Once the B-cell is allowed to behave as a structured visual template, immune memory becomes a much more plausible mechanism for visual data.

The strongest evidence comes from the feature-map Deep AIN. Scalar binding profiles collapse each B-cell response into one number immediately. That can be appropriate for low-dimensional tabular antigens, but it removes too much information from visual antigens. The feature-map model delays this collapse: a layer-1 B-cell produces a spatial response map, and the next layer learns over the stack of response maps. This is important because it gives a concrete computational meaning to depth in a visual AIN. A deeper immune layer is not simply another classifier placed after the first layer; it is a repertoire that learns memories of lower-layer response patterns. In immune terms, the higher layer is exposed to the system’s own response to the antigen, not only to the raw antigen itself.

The class-incremental results show that this response-pattern memory can remain useful under sequential exposure. The feature-map Deep AIN is not trained with labels and does not replay old samples. Nevertheless, its binding-profile space supports strong external probes after classes have arrived sequentially. This matters because the probes are not the proposed model; they are measurements of the representation produced by the AIN. The positive result therefore indicates that the immune repertoire has organized the visual stream into a memory space with recoverable class structure. This is the main empirical claim of the paper.

As new classes arrive, the lower immune layers adapt and the binding-profile coordinate system changes. The changing coordinate system is part of the online memory mechanism. The AIN is not replay-trained on old classes when new classes arrive. Instead, after immune adaptation, old and new samples can be passed through the current repertoire to obtain their current binding profiles. If earlier classes remain separable in this current representation space, then the system has reorganized its representation while retaining the ability to represent previous classes. The external readout is therefore used to test whether the current immune memory (representation) space still contains recoverable information about all encountered classes.

The comparison with replay MLP and full refit k-means should be interpreted carefully. Those methods are valid upper baselines for predictive performance because they reuse old information more directly: replay MLP rehearses stored examples, and full refit k-means reconstructs the clustering space from accumulated data. The feature-map Deep AIN is diferent in kind. Its immune layers adapt online, without replay, and form a binding-profile representation without backpropagation. Matching or exceeding replay MLP on some datasets is therefore significant, but the more important result is consistency: across sklearn digits, MNIST, Fashion-MNIST, and KMNIST, the learned immune representation remains competitive enough to be useful. This supports the claim that structured-afinity AINs are viable visual online memory learners, not merely small-dataset curiosities.

The probe results also reveal the geometry of the learned memory space. On KMNIST, 1- nearest-neighbour and PCA+RBF-SVM outperform logistic regression and nearest-centroid probes. This suggests that the binding-profile space can be locally coherent without being globally linearly separable. That observation is important for two reasons. First, it prevents an overly narrow evaluation of AIN representations using only linear probes. Second, it suggests that future immune readout mechanisms should not assume that useful memory must appear as a globally linear class structure. Local neighbourhoods, response-pattern similarity, and repertoire-level interactions may be more faithful to the immune interpretation.

The depth and scale-calibration results further qualify the role of depth. Depth is useful only when the higher layer operates at an appropriate afinity scale. A poorly scaled response-profile space can make a deeper AIN appear weak even when the underlying representation is meaningful. The performance improvement from fixed under-scaled settings to calibrated second-layer settings shows that higher-layer B-cells require scale selection appropriate to the response-profile space, not to the original image space. At the same time, the three-layer result shows that adding depth alone is not suficient. Additional layers compress and reorganize the representation, so their capacity and receptive fields must be matched to the structure of the previous layer’s response maps.

Overall, the paper establishes a specific claim. It does not show that AINs outperform modern deep visual classifiers, nor does it claim that ZNCC is the best possible immune afinity. Rather, it shows that replay-free visual online learning in AINs becomes credible when three conditions hold: B-cells use structured visual afinity, lower-layer responses preserve spatial response maps, and higher layers are calibrated to the binding-profile spaces they receive. Under those conditions, a Deep AIN can form a replay-free, gradient-free visual memory representation whose class structure is recoverable after sequential exposure.

## 7 Limitations

The results support structured-afinity Deep AINs as replay-free visual online memory learners, but they also identify several limitations of the current formulation.

Local-filter ambiguity. The feature-map model relies on local B-cell filters. Local responses are useful because they preserve spatial structure and make the AIN less dependent on flattened pixel distance. However, small local filters can also respond similarly to visually related classes. This is especially likely in digit and character data, where diferent classes may share strokes, corners, loops, and short edge fragments. In these cases, a lower layer may produce response maps that are locally meaningful but not suficiently distinctive for global class discrimination. The KMNIST probe diagnostic is consistent with this issue: the learned space remains locally coherent, but linear and centroid-based probes are weaker than local or nonlinear probes.

Repertoire growth and redundancy. Memory formation in an AIN can increase the number of B-cells or the dimensionality of the binding-profile representation. This is conceptually attractive because adding B-cells adds memory axes, but it also creates a practical tradeof. Larger repertoires increase training cost, inference cost, and the dimensionality of downstream binding profiles. They may also accumulate redundant or low-utility B-cells if suppression does not remove them efectively. The present paper reports runtime diagnostics, but it does not yet provide a full population-control mechanism for preserving compact repertoires while maintaining visual memory quality.

Higher-layer scale and capacity sensitivity. The depth experiments show that higherlayer performance depends strongly on the afinity scale and capacity of the response-profile space. A second layer improves substantially when its scale is calibrated, whereas a third layer remains weaker under the current configuration because it compresses the representation too aggressively. This does not undermine the definition of depth, but it shows that additional immune layers require matched receptive fields, repertoire sizes, and scale-selection rules. Depth is therefore a mechanism that must be configured with respect to the geometry of the previous layer’s response maps.

Scope of the current visual evidence. The strongest evidence in this paper is obtained on grayscale visual benchmarks: sklearn digits, MNIST, Fashion-MNIST, and KMNIST. These datasets are appropriate for establishing the core claim that structured afinity makes online visual AIN learning viable, but they do not exhaust the visual setting. Natural RGB images introduce colour channels, stronger background variation, texture diversity, and more complex object structure. The present results therefore establish a proof of concept for structured visual immune memory rather than a general solution to visual continual learning.

## 8 Conclusion

This paper provides evidence that AINs can learn visual data online when afinity respects image structure. Flattened vector afinity is a weak inductive bias for visual antigens, but structured immune afinity allows B-cells to act as local visual templates and response-map generators. In this setting, a Deep AIN is not a supervised classifier; it is a replay-free, gradient-free immune representation-memory learner whose binding-profile space can be evaluated by external probes.

The strongest evidence is that feature-map Deep AIN binding profiles form class-discriminative memory spaces under sequential visual exposure. On sklearn digits, a calibrated two-layer featuremap Deep AIN reaches 0.978 final balanced accuracy with a logistic external probe. Under the same adaptive scale rule, the model remains competitive across MNIST, Fashion-MNIST, and KMNIST, with the KMNIST diagnostic indicating that the learned memory space can be locally coherent even when it is not globally linearly separable. These results support the paper’s central claim: structured afinity makes visual online memory learning a viable and scientifically meaningful direction for artificial immune networks.

## References

[1] Christopher M. Bishop. Neural Networks for Pattern Recognition. Oxford University Press, 1995.

[2] David S. Broomhead and David Lowe. Multivariable functional interpolation and adaptive networks. Complex Systems, 2:321–355, 1988.

[3] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, volume 1, pages 886–893, 2005.

[4] Dipankar Dasgupta, editor. Artificial Immune Systems and Their Applications. Springer, 1999.

[5] Leandro N. de Castro and Jonathan Timmis. Artificial Immune Systems: A New Computational Intelligence Approach. Springer, 2002.

[6] Leandro N. de Castro and Fernando J. Von Zuben. An evolutionary immune network for data clustering. In Proceedings of the IEEE Brazilian Symposium on Artificial Neural Networks, pages 84–89, 2000.

[7] Leandro N. de Castro and Fernando J. Von Zuben. Learning and optimization using the clonal selection principle. IEEE Transactions on Evolutionary Computation, 6(3):239–251, 2002.

[8] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7): 3366–3385, 2021.

[9] J. Doyne Farmer, Norman H. Packard, and Alan S. Perelson. The immune system, adaptation, and machine learning. Physica D: Nonlinear Phenomena, 22(1–3):187–204, 1986.

[10] Stephanie Forrest, Alan S. Perelson, Lawrence Allen, and Rajesh Cherukuri. Self-nonself discrimination in a computer. In Proceedings of the 1994 IEEE Symposium on Research in Security and Privacy, pages 202–212, 1994.

[11] Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unafected by shift in position. Biological Cybernetics, 36(4):193–202, 1980.

[12] Niels K. Jerne. Towards a network theory of the immune system. Annales d’Immunologie, 125C(1–2):373–389, 1974.

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114 (13):3521–3526, 2017.

[14] Teuvo Kohonen. Self-organized formation of topologically correct feature maps. Biological Cybernetics, 43(1):59–69, 1982.

[15] Teuvo Kohonen. Improved versions of learning vector quantization. Proceedings of the International Joint Conference on Neural Networks, pages 545–550, 1990.

[16] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[17] J. P. Lewis. Fast normalized cross-correlation. In Vision Interface, pages 120–123, 1995.

[18] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems, 2017.

[19] David G. Lowe. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2):91–110, 2004.

[20] German I. Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54–71, 2019.

[21] Elzbieta Pekalska and Robert P. W. Duin. The Dissimilarity Representation for Pattern Recognition: Foundations and Applications. World Scientific, 2005.

[22] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, 2007.

[23] Sylvestre-Alvise Rebufi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. iCaRL: Incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[24] Jonathan Timmis, Mark Neal, and John Hunt. An artificial immune system for data analysis. Biosystems, 55(1–3):143–150, 2000.

[25] Gido M. van de Ven, Tinne Tuytelaars, and Andreas S. Tolias. Three types of incremental learning. Nature Machine Intelligence, 4:1185–1197, 2022.

[26] Andrew Watkins and Jonathan Timmis. Artificial immune recognition system (AIRS): An immune-inspired supervised learning algorithm. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 1390–1397, 2002.

[27] Christopher K. I. Williams and Matthias Seeger. Using the nystr¨om method to speed up kernel machines. In Advances in Neural Information Processing Systems, 2001.