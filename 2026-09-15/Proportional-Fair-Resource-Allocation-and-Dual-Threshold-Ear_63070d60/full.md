# Proportional-Fair Resource Allocation and Dual-Threshold Early-Exit Inference for Secure Cooperative Multi-Layer Edge Intelligence

Thai T. Vu, John Le, Tu N. Nguyen, Jun Shen, Quang Vinh Duong, Ha Nguyen

Abstract—This paper proposes FREDI (Fair Resource Allocation for Edge Dual-Threshold Inference), a secure wireless edgeintelligence framework for event-triggered inference in a cooperative user equipment (UE)–edge server (ES)–cloud system. Each UE performs early-exit convolutional neural network (CNN) screening using dual confidence thresholds, while critical events are securely offloaded to an edge server for detailed classification. We formulate a proportionally-fair utility maximization problem that jointly optimizes UE–ES association, wireless and processing resources, and confidence thresholds. FREDI decomposes the problem into proportional-fair resource allocation and dualthreshold inference optimization. We prove that the detectedcritical event set is set-monotone non-increasing in both thresholds, and exploit the finite empirical confidence domain for exact threshold optimization. An empirical resource–utility response envelope yields a computable global suboptimality bound and a sufficient condition for global optimality. By pre-eliminating infeasible UE–ES pairs and exactly projecting out bandwidth and transmit-power variables, the resource-allocation subproblem is reduced to a mixed-integer exponential-cone program solvable to the certified global optimality within a prescribed gap. Numerical results with early-exit MobileNetV2 and ShuffleNetV2 demonstrate near-perfect UE fairness with aggregate utility close to a Sum-Utility benchmark, reveal security-induced resource fragmentation, and demonstrate the Stage-A scalability from 6 to 144 UEs with median solving time below 0.1 s in the tested configurations.

Index Terms—Wireless edge intelligence, fair resource allocation, dual-threshold inference, early-exit CNN, secure offloading, mixed-integer conic optimization.

## I. INTRODUCTION

The growing demand for latency-sensitive intelligent services has accelerated the implementation of wireless edge intelligence, in which communication and nearby computing resources are jointly orchestrated for distributed inferences [1], [2]. In a three-layer deployment, user equipment (UEs) observes events close to their sources, edge servers (ESs) provide more capable inference, and a cloud server (CS) periodically updates the models. This hierarchy reduces reliance on remotecloud inference without requiring resource-constrained UEs to execute the complete learning pipeline.

Early-exit neural networks (EENNs) are well-established for adaptive inference, where intermediate classifiers allow sufficiently confident samples to terminate before the final layer, thereby reducing computation [3]–[5]. In particular, confidence-threshold exit policies are common in literature. More generally, two- or multiple-threshold decision rules that partition a confidence domain into decision and rejection/undecided regions have been widely used well before the emergence of edge AI [6]–[8]. In this paper, we study how early-exit convolutional neural networks (CNNs) and dualthreshold decision rules would interact with shared wireless and processing resources in a secure and cooperative multilayer edge system.

Recent studies have increasingly coupled deep neural network (DNN) inference with edge-resource management. Liu et al. [9] optimize multi-user communication and computation with batching and early exits in a single-ES system, while Kim and Lee [10] jointly optimize edge resources and DNN splitting in an edge–cloud setting. Zheng et al. [11] further consider multi-terminal, multi-base station (BS) semantic transmission with communication/computation allocation. Liu et al. [12] jointly optimize model partitioning, exit selection, association, bandwidth, and computation in multi-server mobile edge computing (MEC). Zhang et al. [13] and Yuan et al. [14] further couple service placement and model splitting with resource allocation. These studies demonstrate the value of co-designing DNN inference and edge-resource allocation, but none jointly considers proportional fairness, securityconstrained association, and per-UE dual-threshold screening.

Most closely related, Zhou et al. [15] use dual-threshold multi-exit inference for event-triggered offloading between a single device and an edge server. Our FREDI (Fair Resource Alloca- tion for Edge Dual-Threshold Inference) framework differs primarily at the network level: multiple UEs compete for communication and processing resources across multiple ESs, coupling threshold selection with UE–ES association, bandwidth, transmit power, ES capacity, security eligibility, and proportional fairness. This turns device-level inference and offloading control into network-wide resource orchestration with combinatorial association.

Fairness and security have largely been studied separately. Xu et al. [16] use proportional fairness for distributed DNNinference assignment and load balancing, while [17] studies energy-based proportional fairness for task offloading and resource allocation in cooperative edge computing, without considering early-exit inference. Security-aware offloading couples execution decisions with protection requirements and resource use [18], [19]. Hybrid quantum–classical models have also been explored for classification and distributed learning [20]–[23]. In FREDI, lightweight binary early-exit screening is performed at UEs, whereas heavier hybrid CN-NQNN (quantum neural network) inference [23] is performed at ESs.

Table I compares FREDI with representative studies. A checkmark denotes explicit treatment, ◦ for partial or related treatment, and “–” indicating an absence as a principal component. The comparison highlights differences in system scope and analytical treatment rather than a comparison of the novelty in standard early-exit or threshold mechanisms.

To address these gaps, we develop FREDI (Fair Resource Allocation for Edge Dual-Threshold Inference). Its novelty lies not in early-exit or dual-threshold inference alone, but in their network-scale formulation, structural analysis, and provably controlled optimization within a secure cooperative multi-layer edge-AI system. The main contributions are:

• Network-scale cooperative formulation: We formulate a proportional-fair multi-UE, multi-ES problem that jointly optimizes per-UE dual thresholds, security-constrained UE–ES association, wireless resources, and shared ES processing capacity, thereby capturing competition for heterogeneous communication and computing resources.

• Structural analysis and optimization guarantees: We prove that the detected-critical event set is set-monotone non-increasing in both thresholds, enabling exact finitedomain threshold optimization with directional pruning. We further eliminate infeasible UE–ES pairs and exactly project out bandwidth and power variables, yielding a mixed-integer exponential-cone resource-allocation problem with a certified optimality gap. A computable resource–utility response envelope bounds the end-to-end decomposition loss and gives a sufficient condition for global optimality.

• Fairness, security, and scalability evaluation: Experiments with early-exit MobileNetV2 and ShuffleNetV2 quantify the dual- versus coupled single-parameter threshold trade-off, fairness–utility trade-off, and securityinduced resource fragmentation, while testing scalability from � = 6 to � = 144 UEs. FREDI achieves nearuniform UE utility with practical computation over the tested network sizes.

The rest of this paper is organized as follows. Section II presents the system model, performance metrics, and joint optimization problem. Section III details FREDI, including its two-stage decomposition and optimality analysis. Section IV reports experimental results on thresholding, fairness, security constraints, and scalability. Section V concludes the paper.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

## A. System Model

We consider a cooperative three-layer Edge AI system comprising � local user devices (i.e., surveillance cameras), referred to as UEs and denoted by $\mathbb { N } = \{ 1 , 2 , \ldots , N \}$ ; � edge servers (ESs), denoted by $\mathbb { M } = \{ 1 , 2 , \dots , M \}$ ; and one cloud server (CS). The three-layer system entities are shown on the left of Fig. 1, while the right side summarizes the cooperative inference and periodic model-update workflow. During each scheduling period, UE � captures a sequence of independent events (or image), denoted by $\Phi _ { n } ~ = ~ \{ I _ { n 1 } , I _ { n 2 } , . . . , I _ { n | \Phi _ { n } | } \}$ where $I _ { n k }$ denotes the �th event and $| \Phi _ { n } |$ is the number of events. Each UE $n \in \mathbb { N }$ employs a lightweight CNN with dual-threshold early exits to classify locally observed events as normal or critical. Events detected as normal remain at the UE. For an event detected as critical, the UE securely offloads its extracted feature representation to one associated ES. The selected ES executes a hybrid CNN–QNN model for detailed multi-class classification and returns the resulting label or response action. The CS collects training information or model updates from the ESs, updates the classical CNN, featureadapter, and QNN parameters, and distributes the updated hybrid models back to the ESs. The model parameters are treated as fixed during each scheduling and inference period considered by the optimization.

![](images/9eb8c2801420b9a64f43361f0c4b871a93b3a217454057a32c2cf5f9e06eab55.jpg)  
Fig. 1: Three-layer Edge-AI system entities (left) and cooper ative inference and periodic model-update workflow (right).

Let $\mathbb { S } = \{ 1 , \ldots , S \}$ denote the set of security levels assigned to UEs, ESs, and the CS, where a larger index represents a stronger security level. Each UE � serves one application with a required security level $s _ { n } ^ { \mathrm { U E } } \in \mathbb { S } ;$ ; consequently, all events generated by that UE share the same security requirement. Their critical-event features can therefore be offloaded only to an ES � satisfying $s _ { i } ^ { \mathrm { E S } } \geq s _ { n } ^ { \mathrm { U E } }$ . We assume that the CS satisfies the required security and privacy protections.

1) Early-Exit Inference at UEs: The UE-side CNN consists of convolutional feature extractors, pooling or subsampling layers, nonlinear activations, and lightweight classification heads [24]. Because events vary in classification difficulty, we adopt early-exit inference [3]–[5] with a two-threshold confidence rule [6], [7]. At each exit, an event is classified as critical if its confidence reaches the upper threshold, as normal if it reaches the lower threshold, and otherwise proceeds to a deeper exit. The UE-specific thresholds are denoted by $( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ , with $0 \leq \alpha _ { n } ^ { l } \leq \alpha _ { n } ^ { u } \leq 1$ . Fig. 2 illustrates this policy. Our focus is the coupling of threshold-controlled inference loads with secure multi-UE/multi-ES resource orchestration, rather than the threshold rule itself.

Confidence Score. Let � be the number of layers of the lightweight CNN at UE �. Given the extracted feature of event $I _ { n k } \ \in \ \Phi _ { n }$ as input, the CNN generates a confidence score at each layer to assess whether the event is critical. The confidence score $C _ { n q } ^ { ( k ) }$ at layer � for event $I _ { n k }$ is defined as

TABLE I: Comparison with representative edge-inference and resource-allocation studies. Columns denote: confidence-based early exiting; two independently chosen thresholds; more than one device; more than one server; joint optimization of communication and server-side compute; an explicit proportional-fair objective; association restricted by a security or trust level; a proved structural property of the threshold rule; and a reported runtime study over increasing network size. Entries record the aspects each work treats as a principal component; a dash does not imply the aspect is unattainable within that framework.
<table><tr><td>Study</td><td>Early exit</td><td>Dual threshold</td><td>Multi- UE/device</td><td>Multi- ES/server</td><td>Joint network resource alloc. PF</td><td></td><td>Security</td><td>Threshold constraints structure/proof</td><td>Network scalability</td></tr><tr><td>Liu et al., JSAC’23 [9]</td><td>√</td><td></td><td>√</td><td>一</td><td>√</td><td>一</td><td></td><td></td><td></td></tr><tr><td>Xu et al., IoTJ&#x27;23 [16]</td><td>一</td><td></td><td>0</td><td>√</td><td>o</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Kim and Lee, IoTJ&#x27;24 [10]</td><td>√</td><td></td><td>0</td><td>0</td><td>√</td><td>一</td><td></td><td></td><td></td></tr><tr><td>Zheng et al., TWC&#x27;25 [11]</td><td>√</td><td>1</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Zhou et al., TCOM&#x27;26 [15]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>0</td><td>7</td><td></td><td>O</td><td></td></tr><tr><td>Liu et al., CJE’26 [12]</td><td>√</td><td>一</td><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>FREDI (this work)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

![](images/bdec742508dfea2219e5c75bcf5a2562f74ccc00e175a538ebd5816da0784abf.jpg)  
Fig. 2: CNN with dual-threshold early exits.

$$
C _ { n q } ^ { ( k ) } = \frac { e ^ { f _ { n q } ^ { ( k ) , \mathrm { c r i t i c a l } } } } { e ^ { f _ { n q } ^ { ( k ) , \mathrm { c r i t i c a l } } } + e ^ { f _ { n q } ^ { ( k ) , \mathrm { n o r m a l } } } } ,\tag{1}
$$

where $f _ { n q } ^ { ( k ) , \mathrm { c r i t i c a l } }$ and $f _ { n q } ^ { ( k ) , \mathrm { n o r m a l } }$ are the critical- and normalclass logits, respectively, at layer � for event $I _ { n k }$

Early-Exit Event Detection. The CNN classifies each event as critical (1) or normal (0) as early as possible. For $\alpha _ { n } ^ { l } < \alpha _ { n } ^ { u } ;$ processing stops at the first exit � whose confidence leaves the undecided interval ${ \cal T } _ { n } \triangleq ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ : event $I _ { n k }$ is declared normal if $C _ { n q } ^ { ( k ) } \ \leq \ \alpha _ { n } ^ { l }$ and critical if $C _ { n q } ^ { ( k ) } \ \geq \ \alpha _ { n } ^ { u }$ . If all exits remain within the interval, the event is conservatively declared normal at the final exit. When $\alpha _ { n } ^ { l } = \alpha _ { n } ^ { u }$ , the policy reduces to a single decision at the first exit, with equality assigned to the normal class.

Accordingly, the predicted label $\hat { y } _ { n k } \in \{ 1 , 0 \}$ is

$$
\hat { y } _ { n k } = \left\{ \begin{array} { l l } { 0 , \exists q \leq L : \ C _ { n q } ^ { ( k ) } \leq \alpha _ { n } ^ { l } , \ C _ { n t } ^ { ( k ) } \in \mathcal { I } _ { n } , \ \forall t < q , } \\ { 1 , \exists q \leq L : \ C _ { n q } ^ { ( k ) } \geq \alpha _ { n } ^ { u } > \alpha _ { n } ^ { l } , \ C _ { n t } ^ { ( k ) } \in \mathcal { I } _ { n } , \ \forall t < q , } \\ { 0 , \ C _ { n t } ^ { ( k ) } \in \mathcal { I } _ { n } , \ \forall t \leq L , } \\ { 1 \left\{ C _ { n 1 } ^ { ( k ) } > \alpha _ { n } ^ { l } \right\} , \alpha _ { n } ^ { l } = \alpha _ { n } ^ { u } . } \end{array} \right.\tag{2}
$$

where 1{·} is the indicator function. The four cases are mutually exclusive: when $\alpha _ { n } ^ { l } < \alpha _ { n } ^ { u }$ the last case is inactive, and when $\alpha _ { n } ^ { l } = \alpha _ { n } ^ { u }$ the interval ${ \cal { I } } _ { n }$ is empty, so only the first exit is examined and confidence exactly equal to the common threshold is assigned to the normal class.

2) Secure Communication Between UEs and ESs: We employ FDMA for UE–ES uplink communication. Let $\pmb { b } \ =$ $\{ b _ { n j } \} \in \mathbb { R } ^ { N \times M }$ and $\pmb { p } = \{ p _ { n j } \} \in \mathbb { R } ^ { N \times M }$ denote the bandwidth and transmit power allocated from UE � to $\operatorname { E S } j ,$ with perlink limits $b ^ { \mathrm { m a x } }$ and $p ^ { \mathrm { m a x } }$ . Each UE is associated with one ES, represented by $\mathbf { x } ^ { \prime } = \{ x _ { n j } \} \in \{ 0 , 1 \} ^ { N \times M }$ , where $x _ { n j } = 1$ denotes association between UE � and ES $j .$

Let $D _ { n }$ denote the representative size, in bits, of the feature data offloaded by UE � for one event. The uplink rate from UE � to ES � is

$$
r _ { n j } ( b _ { n j } , p _ { n j } ) = b _ { n j } \log _ { 2 } \left( 1 + \frac { g _ { n j } p _ { n j } } { \sigma _ { n j } ^ { 2 } b _ { n j } } \right) ,\tag{3}
$$

where $g _ { n j }$ and $\sigma _ { n j } ^ { 2 }$ are the channel gain and noise power spectral density of the legitimate link, respectively.

To model physical-layer confidentiality, we consider an eavesdropper that attempts to intercept transmissions from UE �. Its achievable rate on the bandwidth allocated to link (�<sub>,</sub> �) is

$$
r _ { n j } ^ { \mathrm { E V } } ( b _ { n j } , p _ { n j } ) = b _ { n j } \log _ { 2 } \left( 1 + \frac { g _ { n } ^ { \mathrm { E V } } p _ { n j } } { ( \sigma _ { n } ^ { \mathrm { E V } } ) ^ { 2 } b _ { n j } } \right) ,\tag{4}
$$

where $g _ { n } ^ { \mathrm { E V } }$ and $( \sigma _ { n } ^ { \mathrm { E V } } ) ^ { 2 }$ denote the corresponding eavesdropper-channel gain and noise power spectral density. The secure transmission rate [25] is

$$
r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) = \left[ r _ { n j } ( b _ { n j } , p _ { n j } ) - r _ { n j } ^ { \mathrm { E V } } ( b _ { n j } , p _ { n j } ) \right] ^ { + } ,\tag{5}
$$

where $[ z ] ^ { + } = \operatorname* { m a x } \{ 0 , z \}$ . Consistent with the FDMA model, transmissions are orthogonal in frequency, so (3) and (4) contain no inter-user or inter-cell interference term; the eavesdropper channel statistics $g _ { n } ^ { \mathrm { E V } }$ and $( \sigma _ { n } ^ { \mathrm { E V } } ) ^ { 2 }$ are assumed known at the scheduler, which is the standard worst-case assumption in secrecy-rate resource allocation and yields a conservative feasible set. The selected secure rate of UE � is

$$
r _ { n } ^ { \mathrm { s e } } ( \pmb { b } _ { n } , \pmb { p } _ { n } , \pmb { x } _ { n } ) = \sum _ { j = 1 } ^ { M } x _ { n j } r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) .\tag{6}
$$

For a positive selected secure rate, the secure uplink transmission delay is

$$
t _ { n } ^ { \mathrm { t x } } ( { \pmb { b } } _ { n } , { \pmb { p } } _ { n } , { \pmb { x } } _ { n } ) = \frac { D _ { n } } { r _ { n } ^ { \mathrm { s e } } ( { \pmb { b } } _ { n } , { \pmb { p } } _ { n } , { \pmb { x } } _ { n } ) } .\tag{7}
$$

Let $t _ { n } ^ { \mathrm { r } }$ denote the maximum allowable uplink transmission delay of UE �.

The real-time latency model considers only the secure uplink transmission delay from UEs to ESs. The execution and queueing delays associated with hybrid-model inference at the ESs, as well as the comparatively slower CS modelupdate delay, are beyond the scope of this work.

3) Hybrid CNN–QNN Multi-Class Inference at ESs: ES � is characterized by the tuple $( B _ { j } , W _ { j } , s _ { i } ^ { \mathrm { E S } } )$ , where $B _ { j }$ is its available UE-uplink bandwidth, $W _ { j }$ is its aggregate eventprocessing capacity over the considered scheduling period, and $s _ { i } ^ { \mathrm { E S } } \in \mathbb { S }$ is its security level. Each ES hosts a hybrid CNN– QNN classifier that processes the received representation and produces the final multi-class prediction. We measure $W _ { j }$ in event-processing units, with one unit corresponding to the workload required to process one offloaded event through the complete hybrid inference pipeline, assuming approximately homogeneous per-event workloads.

If UE � is associated with ES $j ,$ the system allocates uplink bandwidth $b _ { n j }$ and reserves event-processing capacity $w _ { n j }$ at ES �. Let $w ^ { \mathrm { m a x } }$ denote the maximum processing capacity that an ES can reserve for one UE during a scheduling period.

When an event is detected as critical at UE �, its feature representation is offloaded to the selected ES for multi-class classification and the corresponding response. Rather than explicitly modeling ES execution time, processing feasibility is captured by the aggregate capacity $W _ { j }$ and per-UE allocation $w _ { n j } ;$ the optimization is therefore agnostic to the ES classifier. We adopt the hybrid CNN–QNN instantiation as a forwardlooking design point for when variational quantum classifiers become practical, and it is not evaluated in this paper.

4) Periodic Model Aggregation and Updating at the CS: Periodically, the ESs transmit approved training information, model parameters, or feature summaries to the CS over secure backhaul links, as illustrated by the upper model-update path on the right of Fig. 1. The CS aggregates this information, updates the hybrid CNN–QNN model, and redistributes the updated model to the ESs for subsequent deployment. Cloudtraining delay and backhaul-resource allocation are outside the optimization scope.

## B. Performance Metrics

1) Categories ofInput Events and Output Results: For each event $I _ { n k } \in \Phi _ { n } ,$ , UE � uses its lightweight CNN to classify the event as critical or normal. Let $y _ { n k }$ and $\hat { y } _ { n k }$ denote the ground-truth and predicted labels of event $I _ { n k }$ , respectively. The event is correctly classified if $\begin{array} { r c l } { { \hat { y } _ { n k } } } & { { = } } & { { y _ { n k } ; } } \end{array}$ the four possible outcomes are the usual true/false positive/negative combinations, summarized as event sets in Table II.

Over a given time period, UE � captures the event set $\Phi _ { n } \ = \ \{ I _ { n 1 } , . . . , I _ { n | \Phi _ { n } | } \}$ . Let $\Phi _ { n } ^ { \mathrm { N } }$ and $\Phi _ { n } ^ { \bar { \mathrm { P } } }$ denote the sets of normal and critical events at UE �, respectively. Let $\hat { \Phi } _ { n } ^ { \mathrm { T P } }$ $\hat { \Phi } _ { n } ^ { \mathrm { F P } } , \ \hat { \Phi } _ { n } ^ { \mathrm { T N } }$ , and $\hat { \Phi } _ { n } ^ { \mathrm { F N } }$ denote the sets of true-positive (TP), false-positive (FP), true-negative (TN), and false-negative (FN) events, respectively, as summarized in Table II.

During the considered scheduling period, every event predicted as critical is offloaded to an ES. We therefore define the offloaded load of UE � as

$$
L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \triangleq | \hat { \Phi } _ { n } ^ { \mathrm { T P } } | + | \hat { \Phi } _ { n } ^ { \mathrm { F P } } | .\tag{8}
$$

TABLE II: Categories of events at UE �.
<table><tr><td>Set</td><td>Definition</td><td>Type</td></tr><tr><td> $\Phi _ { n }$ </td><td>All events at UE n</td><td>Input</td></tr><tr><td> $\Phi _ { n } ^ { \tilde { \mathrm { N } } }$ </td><td>Normal events at UE n</td><td>Input</td></tr><tr><td> $\Phi _ { n } ^ { \mathrm { P } }$   $\hat { \Phi } _ { \ldots } ^ { \mathrm { T P } }$ </td><td>Critical events at UE n</td><td>Input</td></tr><tr><td> $\hat { \Phi } _ { \ldots } ^ { n }$ </td><td>Correctly detected critical events</td><td>Output</td></tr><tr><td> $\hat { \Phi } _ { \ldots } ^ { n }$ </td><td>Normal events detected as critical</td><td>Output</td></tr><tr><td></td><td>Correctly detected normal events</td><td>Output</td></tr><tr><td> $\hat { \Phi } _ { n } ^ { \scriptscriptstyle n }$ </td><td>Critical events detected as normal</td><td>Output</td></tr></table>

Using the association and processing-allocation variables, define the event-processing capacity allocated to UE � as

$$
R _ { n } \triangleq \sum _ { j = 1 } ^ { M } x _ { n j } w _ { n j } .\tag{9}
$$

Because $w _ { n j }$ is measured in event-processing units over the same period, the allocated capacity must cover the offloaded load but need not exceed the number of events generated by the UE. These processing QoS requirements are specified in Problem ${ \bf P } _ { 0 }$

We then define the performance metrics that are influenced by the use of dual confidence thresholds.

2) User Utility Functions: Because correctly detecting critical events is the primary objective of the system, we define the utility of UE �, denoted by $\mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ , as the proportion of critical events correctly classified as critical. This quantity is the true-positive rate (TPR). From Table II we have

$$
\mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) = \frac { \vert \hat { \Phi } _ { n } ^ { \mathrm { T P } } \vert } { \vert \Phi _ { n } ^ { \mathrm { P } } \vert } .\tag{10}
$$

The communication and edge-processing resources required by UE � increase with its offloaded load $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ . We therefore optimize the UE utilities subject to the wireless and processing resources available at the ESs.

## C. Problem Formulation

Events arrive sequentially at the UEs, and only those predicted as critical are offloaded to ESs for hybrid CNN– QNN multi-class classification. Over one scheduling period, the joint design contains two coupled decision groups that motivate FREDI: (i) fair UE–ES association and wireless/edge resource allocation through $( \mathbf { x } , \pmb { b } , \pmb { p } , \pmb { w } )$ , and (ii) early-exit CNN inference control through the lower and upper confidence thresholds $( \alpha ^ { l } , \alpha ^ { u } )$ . The CNN and QNN model parameters distributed by the CS are fixed during this period; periodic cloud training is therefore outside ${ \bf P } _ { 0 }$

Maximizing aggregate utility alone may allocate few communication and processing resources to UEs whose utility gain per unit resource is comparatively small. FREDI therefore associates fairness with the resource-allocation layer: we adopt weighted proportional fairness across UEs, while the dual thresholds remain inference-control variables. We assume $\mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) ~ > ~ 0$ for every UE �. Weighted proportional fairness is induced by maximizing the standard network utility $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \rho _ { n } \ln ( \mathcal { U } _ { n } ) } \end{array}$ , where $\rho _ { n } \ > \ 0$ is the fairness weight of UE � [26].

The end-to-end problem nevertheless optimizes both groups jointly: the dual confidence thresholds $\alpha ^ { l }$ and $\alpha ^ { u }$ determine early-exit CNN decisions and offloaded load, whereas x, �, �, and � determine fair secure access to wireless and ES processing resources. The resulting problem is

$$
\mathbf { P } _ { 0 } : \quad \operatorname* { m a x i m i z e } _ { \alpha ^ { l } , \alpha ^ { u } , \mathbf { x } , b , p , w } \quad \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \Bigl ( \mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \Bigr )\tag{11}
$$

subject to

$$
0 \leq b _ { n j } \leq b ^ { \operatorname* { m a x } } ,
$$

$$
\forall ( n , j ) \in \mathbb { N } \times \mathbb { M } ,\tag{11a}
$$

$$
0 \leq p _ { n j } \leq p ^ { \operatorname* { m a x } } ,
$$

$$
\forall ( n , j ) \in \mathbb { N } \times \mathbb { M } ,\tag{11b}
$$

$$
0 \leq w _ { n j } \leq w ^ { \mathrm { m a x } } ,
$$

$$
\forall ( n , j ) \in \mathbb { N } \times \mathbb { M } ,\tag{11c}
$$

$$
\sum _ { n = 1 } ^ { N } x _ { n j } b _ { n j } \le B _ { j } ,
$$

$$
\forall j \in \mathbb { M } ,\tag{11d}
$$

$$
\sum _ { n = 1 } ^ { N } x _ { n j } w _ { n j } \le W _ { j } ,
$$

$$
\forall j \in \mathbb { M } ,\tag{11e}
$$

$$
\sum _ { j = 1 } ^ { M } x _ { n j } s _ { j } ^ { \mathrm { E S } } \geq s _ { n } ^ { \mathrm { U E } } ,
$$

$$
\forall n \in \mathbb { N } ,\tag{11f}
$$

$$
t _ { n } ^ { \mathrm { t x } } \left( b _ { n } , p _ { n } , \pmb { x } _ { n } \right) \leq t _ { n } ^ { \mathrm { r } } ,
$$

$$
\forall n \in \mathbb { N } ,\tag{11g}
$$

$$
R _ { n } \leq | \Phi _ { n } | ,
$$

$$
\forall n \in \mathbb { N } ,\tag{11h}
$$

$$
\sum _ { j = 1 } ^ { M } x _ { n j } = 1 ,
$$

$$
\forall n \in \mathbb { N } ,\tag{11i}
$$

$$
x _ { n j } \in \{ 0 , 1 \} ,
$$

$$
\forall ( n , j ) \in \mathbb { N } \times \mathbb { M } ,\tag{11j}
$$

$$
L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \le R _ { n } ,
$$

$$
\forall n \in \mathbb { N } ,\tag{11k}
$$

$$
0 \leq \alpha _ { n } ^ { l } \leq \alpha _ { n } ^ { u } \leq 1 ,
$$

$$
\forall n \in \mathbb { N } .\tag{11l}
$$

Constraints (11a)–(11c) specify the per-link bandwidth, transmission-power, and event-processing-reservation limits, respectively. Constraint (11i) associates each UE with exactly one ES. Constraints (11d) and (11e) enforce the aggregate bandwidth and event-processing capacities of each ES. Constraint (11f) ensures that a UE is associated only with an ES whose security level is sufficiently strong, while (11g) bounds its secure uplink delay. Finally, constraints (11k) and (11h) enforce the processing QoS chain $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \leq R _ { n } \leq | \Phi _ { n } |$

The coupling between the resource-allocation variables and the dual-threshold load constraint (11k) is the central structure exploited by FREDI in Section III.

## III. THE PROPOSED FREDI FRAMEWORK

In this section, we introduce the proposed method, Fair Resource Allocation for Edge Dual-Threshold Inference (FREDI). Its two stages deliberately assign proportional fairness to multi-UE wireless/edge resource allocation and dualthreshold optimization to early-exit CNN inference.

## A. Problem Characterization

Problem ${ \bf P } _ { 0 }$ is a nonlinear mixed-integer problem: it contains binary association variables $x _ { n j } .$ , empirical and generally non-differentiable utilities $\mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ , and products coupling the association decisions to the continuous allocations $( b _ { n j } , p _ { n j } , w _ { n j } )$ . FREDI exploits this structure through two complementary stages. Stage A performs proportionalfair UE–ES association and wireless/edge resource allocation and is transformed exactly into a mixed-integer exponentialcone problem. Stage B then performs dual-threshold inference optimization for each UE over the finite set of observed confidence scores under the processing capacity allocated by Stage A. Hence, fairness is governed by Stage A, whereas inference and offloading decisions are controlled by Stage B. The following analysis distinguishes the exactness of each stage from the end-to-end loss induced by their sequential composition.

Lemma 1 (Set-monotonicity of dual-threshold early exiting). Forfixed CNN confidence traces, let $\hat { \Phi } ^ { \mathrm { c r i t i c a l } } ( \alpha ^ { l } , \alpha ^ { u } )$ denote the events classified as critical under thresholds $0 \leq \alpha ^ { l } \leq \alpha ^ { u } \leq 1$ For any two admissible threshold pairs satisfying $\alpha ^ { l } { } ^ { \prime } \geq \alpha ^ { l }$ and $\alpha ^ { u \prime } \geq \alpha ^ { u }$

$$
\hat { \Phi } ^ { \mathrm { c r i t i c a l } } ( \alpha ^ { l \prime } , \alpha ^ { u \prime } ) \subseteq \hat { \Phi } ^ { \mathrm { c r i t i c a l } } ( \alpha ^ { l } , \alpha ^ { u } ) .\tag{12}
$$

Consequently, $| \hat { \Phi } ^ { \mathrm { c r i t i c a l } } |$ is monotonically non-increasing in either threshold over the admissible domain.

Proof: The proof is shown in Appendix A.

B. Fair Resource Allocation for Edge Dual-Threshold Inference

The empirical utility depends directly on the inference thresholds $( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ , whereas the wireless and processing allocations determine which threshold pairs are feasible through the offloaded-load constraint (11k). Recall from (9) that $R _ { n } =$ $\textstyle \sum _ { j = 1 } ^ { M } x _ { n j } w _ { n j }$ is the processing capacity allocated to UE $n .$

To expose the resource–inference coupling without assuming that the reserved capacity is fully utilized, let $\xi > 0$ denote a generic processing-capacity level and define the optimal Stage-B response of UE � as

$$
V _ { n } ( \xi ) \triangleq \operatorname* { m a x } _ { \begin{array} { c } { 0 \leq \alpha _ { n } ^ { l } \leq \alpha _ { n } ^ { u } \leq 1 } \\ { L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \leq \xi } \end{array} } \mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) .\tag{13}
$$

Only capacity levels for which the feasible set contains a positive-utility threshold pair are considered, consistently with the logarithmic domain assumed in ${ \bf P } _ { 0 }$ . Define the corresponding empirical response factor as

$$
\kappa _ { n } ( \xi ) \triangleq \frac { V _ { n } ( \xi ) } { \xi / | \Phi _ { n } | } .\tag{14}
$$

Consequently, the optimized end-to-end objective associated with a feasible resource allocation admits the exact decomposition

$$
\sum _ { n = 1 } ^ { N } \rho _ { n } \ln V _ { n } ( R _ { n } ) = \underbrace { \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \left( \frac { R _ { n } } { | \Phi _ { n } | } \right) } _ { A ( \mathbf { R } ) } + \underbrace { \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \kappa _ { n } ( R _ { n } ) } _ { K ( \mathbf { R } ) } .\tag{15}
$$

FREDI optimizes the processing-coverage term �(R) in Stage A and then realizes the best empirical inference response $V _ { n } ( R _ { n } )$ in Stage B. Hence, the only end-to-end approximation introduced by the decomposition is that Stage A does not optimize the response term �(R); this term characterizes the departure of the empirical inference response from exact proportionality to the normalized reserved capacity.

FREDI Stage A: Fair UE–ES association and resource allocation. The variables (x, �, �, �) are optimized to allocate secure communication resources and event-processing capacity proportionally across UEs. The Stage-A problem is

$$
\mathbf { P } _ { A } : \quad \underset { \mathbf { x } , b , p , w } { \mathrm { m a x i m i z e } } \quad \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \left( \frac { \sum _ { j = 1 } ^ { M } x _ { n j } w _ { n j } } { | \Phi _ { n } | } \right)\tag{16}
$$

subject to constraints (11a)–(11g) and (11h)–(11j).

Let $( \mathbf { x } ^ { * } , \pmb { b } ^ { * } , \pmb { p } ^ { * } , \pmb { w } ^ { * } )$ denote a globally optimal Stage-A allocation and $\begin{array} { r } { R _ { n } ^ { * } = \sum _ { j = 1 } ^ { M } x _ { n j } ^ { * } w _ { n j } ^ { * } . } \end{array}$ Since $| \Phi _ { n } |$ is fixed within an optimization instance, $\begin{array} { r } { \sum _ { n } ^ { \prime } \rho _ { n } \ln ( R _ { n } / | \Phi _ { n } | ) ~ = ~ \sum _ { n } \rho _ { n } \ln ( R _ { n } ) ~ - } \end{array}$ $\textstyle \sum _ { n } \rho _ { n }$ ln $| \Phi _ { n } | ;$ therefore, the normalized and unnormalized Stage-A objectives have identical maximizers. The normalization is retained to interpret $R _ { n } / | \Phi _ { n } |$ as the fraction of the UE workload for which processing capacity is reserved.

FREDI Stage B: Dual-threshold inference optimization. With the Stage-A allocation fixed, the thresholds are selected by

$$
\mathbf { P } _ { B } : \quad \underset { \alpha ^ { l } , \alpha ^ { u } } { \mathrm { m a x i m i z e } } \qquad \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \Bigl ( \mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \Bigr )\tag{17}
$$

subject to the threshold-domain constraint (11l) and the per-UE capacity constraints

$$
L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \leq R _ { n } ^ { * } , \qquad \forall n \in \mathbb { N } .\tag{18}
$$

Problem $\mathbf { P } _ { B }$ is separable across UEs and attains $V _ { n } ( R _ { n } ^ { * } )$ for every UE when each per-UE empirical search is solved exactly.

Let $F _ { 0 } ^ { \star }$ denote the globally optimal objective value of the empirical problem $\mathbf { P } _ { 0 } .$ , and define $\begin{array} { r } { F _ { 0 } ^ { \mathrm { F R E D I } } \triangleq \sum _ { n } \rho _ { n } } \end{array}$ ln $V _ { n } ( R _ { n } ^ { * } )$ as the objective obtained by globally solving $\mathbf { P } _ { A }$ followed by the exact Stage-B searches.

THEOREM 1 (End-to-end global performance bound). Suppose that the Stage-A solution admits a positive-utility Stage-B solution for every UE. Let $Q _ { n } ^ { + }$ denote the finite set ofpositiveutility threshold candidates for UE �, with $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) > 0 ,$ and define the computable response envelope

$$
\overline { { \kappa } } { _ n } \triangleq \operatorname* { m a x } _ { ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \in Q _ { n } ^ { + } } \frac { | \Phi _ { n } | { \mathcal { U } _ { n } } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) } { L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) } .\tag{19}
$$

Thus, ${ \overline { { \kappa } } } _ { n }$ is obtained during the same finite candidate enumeration used to solve Stage B and requires no solution of $\mathbf { P } _ { 0 } .$ Then

$$
0 \leq F _ { 0 } ^ { \star } - F _ { 0 } ^ { \mathrm { F R E D I } } \leq \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \left( \frac { \overline { { \kappa } } _ { n } } { \kappa _ { n } ( { \cal R } _ { n } ^ { * } ) } \right) .\tag{20}
$$

Proof: The proof is shown in Appendix C.

Corollary 1.1 (Conditional global optimality). If $\kappa _ { n } ( \xi )$ is constant over the positive-utility capacity domain of every UE �, then FREDI globally solves the empirical problem $\mathbf { P } _ { 0 } .$

Proof. Under the stated condition, $\overline { { \kappa } } _ { n } = \kappa _ { n } ( R _ { n } ^ { * } )$ for every UE, so the upper bound in (20) is zero. ■

The factor $\displaystyle \kappa _ { n } ( \xi ) ~ = ~ | \Phi _ { n } | V _ { n } ( \xi ) / \xi$ measures the optimized inference utility obtained per unit of normalized reserved capacity. It therefore reflects both the predictive yield of the selected threshold pair and any unused capacity caused by the discrete empirical operating points. If the Stage-B load is smaller than $\boldsymbol { R } _ { n } ^ { * }$ , the unused reservation can be released without changing the thresholds or the objective of $\mathbf { P } _ { 0 } ;$ this release does not retroactively change the Stage-A allocation decision.

![](images/1e859307fe89e2ce062467fe05b42ce96854e98b0058496cec45aa059e303142.jpg)  
Fig. 3: FREDI algorithm workflow.

Remark 1 (Solver-certified bound). If the mixed-integer solver terminates before proving the exact Stage-A optimum, let R denote its incumbent processing allocation, $A ^ { \mathrm { i n c } } = A ( \widehat { \mathbf { R } } )$ bits incumbent objective, and $A ^ { \mathrm { u b } }$ bits certified upper bound. Then

$$
F _ { 0 } ^ { \star } \leq A ^ { \mathrm { u b } } + \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \overline { { \kappa } } _ { n } .\tag{21}
$$

After exact Stage-B optimization under ${ \widehat { \mathbf { R } } } ,$ the resulting feasible objective $F _ { 0 } ^ { \mathrm { { F R E D I } } } ( \widehat { \mathbf { R } } )$ satisfies

$$
F _ { 0 } ^ { \star } - F _ { 0 } ^ { \mathrm { F R E D I } } ( \widehat { \mathbf { R } } ) \leq \underbrace { A ^ { \mathrm { u b } } - A ^ { \mathrm { i n c } } } _ { S t a g e - A s o l \nu e r g a p } + \underbrace { \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \left( \frac { \overline { { \kappa } } _ { n } } { \kappa _ { n } ( \widehat { R } _ { n } ) } \right) } _ { d e c o m p o s i t i o n b o u n d } .\tag{22}
$$

The second term is evaluated from the Stage-B candidate sets and the realized FREDI operating points; no iterative exchange between the two stages is required.

The FREDI workflow is illustrated in Fig. 3. Stage A allocates secure wireless and ES processing resources proportionally across UEs, whereas Stage B optimizes the dual inference thresholds under the resulting capacities. The � threshold-selection problems in Stage B are separable and can be solved in parallel using Algorithm 1. Theorem 1 bounds the end-to-end loss relative to the empirical global optimum, and Corollary 1.1 identifies the proportional-response condition under which the bound vanishes. The detailed solutions of the two stages are presented in Sections III-C and III-D, respectively.

## C. FREDI Stage A: Fair Resource Allocation

FREDI Stage A implements proportional fairness through the UE–ES association and wireless/edge resource-allocation problem $\mathbf { P } _ { A }$ . Problem $\mathbf { P } _ { A }$ contains products between binary association variables and continuous resource variables. Moreover, when there is no connection $( x _ { n j } = 0 )$ between UE � and ES $j ,$ the resource allocation variables can be set to 0. Thus, constraints (11a), (11b), and (11c) equivalently become

$$
0 \leq b _ { n j } \leq b ^ { \operatorname* { m a x } } x _ { n j } , \qquad \forall ( n , j ) \in \mathbb { N } \times \mathbb { M }\tag{23}
$$

$$
0 \leq p _ { n j } \leq p ^ { \operatorname* { m a x } } x _ { n j } , \qquad \forall ( n , j ) \in \mathbb { N } \times \mathbb { M }\tag{24}
$$

$$
0 \leq w _ { n j } \leq w ^ { \operatorname* { m a x } } x _ { n j } , \qquad \forall ( n , j ) \in \mathbb { N } \times \mathbb { M }\tag{25}
$$

Once these constraints are imposed, the products $x _ { n j } b _ { n j }$ and $x _ { n j } w _ { n j }$ are unnecessary in the aggregate capacity constraints. Thus, constraints (11d), (11e), and (11h) reduce, respectively, to

$$
\sum _ { n = 1 } ^ { N } b _ { n j } \leq B _ { j } , \qquad \forall j \in \mathbb { M } ,\tag{26}
$$

$$
\sum _ { n = 1 } ^ { N } w _ { n j } \leq W _ { j } , \qquad \forall j \in \mathbb { M } ,\tag{27}
$$

$$
\sum _ { j = 1 } ^ { M } w _ { n j } \leq | \Phi _ { n } | ,\tag{28}
$$

For the discrete security-level requirement, define

$$
a _ { n j } = \left\{ \begin{array} { l l } { 1 , } & { s _ { j } ^ { \mathrm { E S } } \geq s _ { n } ^ { \mathrm { U E } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad \forall ( n , j ) \in \mathbb { N } \times \mathbb { M } .\tag{29}
$$

The original security constraint (11f) can then be replaced by

$$
x _ { n j } \leq a _ { n j } , \qquad \forall ( n , j ) \in \mathbb { N } \times \mathbb { M } .\tag{30}
$$

1) Secure-Rate and Delay Reformulation: Define the normalized legitimate and eavesdropper channel gains as

$$
\gamma _ { n j } \triangleq { \frac { g _ { n j } } { \sigma _ { n j } ^ { 2 } } } , \qquad \gamma _ { n } ^ { \mathrm { E V } } \triangleq { \frac { g _ { n } ^ { \mathrm { E V } } } { ( \sigma _ { n } ^ { \mathrm { E V } } ) ^ { 2 } } } .\tag{31}
$$

For a link that can provide positive secrecy, we require $\gamma _ { n j } >$ $\gamma _ { n } ^ { \mathrm { E V } }$ . The secure rate can then be written without the positivepart operator as

$$
\begin{array} { c } { { r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) = b _ { n j } \log _ { 2 } \left( 1 + \displaystyle \frac { \gamma _ { n j } p _ { n j } } { b _ { n j } } \right) } } \\ { { - b _ { n j } \log _ { 2 } \left( 1 + \displaystyle \frac { \gamma _ { n } ^ { \mathrm { E V } } p _ { n j } } { b _ { n j } } \right) , } } \end{array}\tag{32}
$$

with the continuous extension $r _ { n j } ^ { \mathrm { s e } } ( 0 , p ) ~ = ~ 0$ for $p \ \geq \ 0 .$ . If $x _ { n j } = 1$ , the communication-delay requirement is

$$
{ \frac { D _ { n } } { r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) } } \leq t _ { n } ^ { \mathrm { r } } .\tag{33}
$$

Using the binary association variable, this condition is equivalently represented for every link by

$$
r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) \geq \bar { r } _ { n } x _ { n j } , \qquad \bar { r } _ { n } \triangleq \frac { D _ { n } } { t _ { n } ^ { \Gamma } } ,\tag{34}
$$

for all $( n , j ) \in \mathbb { N } \times \mathbb { M }$ . When $x _ { n j } = 0$ , the linking constraints set $b _ { n j } = p _ { n j } = 0$ , and (34) reduces to $0 \geq 0$ . When $x _ { n j } = 1$ it is equivalent to (33).

To characterize the curvature of the secure-rate constraint, define

$$
q _ { n j } ( z ) = \log _ { 2 } ( 1 + \gamma _ { n j } z ) - \log _ { 2 } ( 1 + \gamma _ { n } ^ { \mathrm { E V } } z ) .\tag{35}
$$

For $\gamma _ { n j } > \gamma _ { n } ^ { \mathrm { E V } } , q _ { n j } ( z )$ is concave for $z \geq 0 .$ This property is used below to establish the mixed-integer convex structure of $\mathbf { P } _ { A 1 }$

Because $| \Phi _ { n } |$ is constant for every UE within the scheduling period, maximizing $\begin{array} { r } { \sum _ { n } \rho _ { n } \ln ( R _ { n } / | \Phi _ { n } | ) } \end{array}$ is equivalent to maximizing $\textstyle \sum _ { n } \rho _ { n } \ln ( R _ { n } )$ . After applying the linking constraints, $\begin{array} { r } { R _ { n } = \sum _ { j } w _ { n j } } \end{array}$ , so we use the latter form to obtain a simpler conic representation.

The resulting resource-allocation problem is

$$
\mathbf { P } _ { A 1 } : \quad \underset { \mathbf { x } , \boldsymbol { b } , \boldsymbol { p } , \boldsymbol { w } } { \mathrm { m a x i m i z e } } \quad \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \left( \sum _ { j = 1 } ^ { M } w _ { n j } \right)\tag{36}
$$

« ¬subject to (23)–(25), (26)–(28), (30), (34), (11i), and (11j).

2) Feasible-Pair Preprocessing: A UE–ES pair is individually feasible only if it satisfies the security-level requirement and can support the required secure transmission rate under the per-link resource limits. Define

$$
\mathbb { M } _ { n } ^ { \underline { { \sf f } } } \triangleq \left\{ j \in \mathbb { M } \bigg \vert a _ { n j } = 1 , \ \gamma _ { n j } > \gamma _ { n } ^ { \mathrm { E V } } , \ r _ { n j } ^ { \mathrm { s e } } ( b ^ { \mathrm { m a x } } , p ^ { \mathrm { m a x } } ) \geq \bar { r } _ { n } \right\} .\tag{37}
$$

All infeasible association variables are fixed according to

$$
x _ { n j } = 0 , \qquad \forall n \in \mathbb { N } , \ j \notin \mathbb { M } _ { n } ^ { \mathrm { f } } .\tag{38}
$$

If $\mathbb { M } _ { n } ^ { \mathrm { f } } = \varnothing$ for any UE �, then $\mathbf { P } _ { A 1 }$ is infeasible.

After this fixing, $\mathbf { P } _ { A 1 }$ is a mixed-integer convex problem whose only source of nonconvexity is the binary association. Indeed, every remaining pair satisfies $\gamma _ { n j } > \gamma _ { n } ^ { \mathrm { E V } } ,$ , so $q _ { n j } ^ { \prime \prime } ( z ) =$ $\frac { \ d H _ { 1 } } { \ d t } [ ( \gamma _ { n } ^ { \mathrm { E V } } ) ^ { 2 } ( 1 + \gamma _ { n } ^ { \mathrm { E V } } z ) ^ { - 2 } - \gamma _ { n j } ^ { 2 } ( 1 + \gamma _ { n j } z ) ^ { - 2 } ] \ < \ 0$ for $z \geq 0$ and $q _ { n j }$ is concave; its closed perspective $b _ { n j } q _ { n j } ( p _ { n j } / b _ { n j } ) =$ $r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } )$ is therefore jointly concave in $( b _ { n j } , p _ { n j } ) \ [ 2 7 ,$ Sec. 3.2.6], making $\bar { r } _ { n } x _ { n j } - r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) \ \leq \ 0$ convex. All other constraints of the continuous relaxation are affine, and the objective is concave because $\rho _ { n } > 0$ and ln(·) is concave and increasing.

For every feasible pair, define the minimum bandwidth required at the maximum transmission power:

$$
\underline { { b } } _ { n j } \triangleq \operatorname* { m i n } _ { 0 \le b \le b ^ { \operatorname* { m a x } } } \left\{ b \Big | r _ { n j } ^ { \mathrm { s e } } ( b , p ^ { \operatorname* { m a x } } ) \ge \bar { r } _ { n } \right\} .\tag{39}
$$

The one-dimensional value $\underline { { b } } _ { n j }$ can be computed efficiently by bisection because $r _ { n j } ^ { \mathrm { s e } } ( b , p ^ { \mathrm { m a x } } )$ is continuous and nondecreasing in � over the feasible interval. It is also bounded: $b q _ { n j } ( p / \bar { b } )  p ( \gamma _ { n j } - \gamma _ { n } ^ { \mathrm { E V } } ) / \mathrm { l n } 2$ as $b  \infty .$ , so a link admits a feasible allocation only if $\bar { r } _ { n } < p ^ { \operatorname* { m a x } } ( \gamma _ { n j } - \gamma _ { n } ^ { \mathrm { E V } } ) / \ln 2$ , regardless of the available bandwidth. Security-induced infeasibility is therefore a property of the secrecy advantage rather than of spectrum scarcity alone.

3) Projection of Communication Variables: The power variables do not appear in the objective, and no aggregate power constraint is imposed. This structure permits the communication variables to be eliminated exactly rather than handled by a custom branch-and-bound procedure.

Lemma 2. Consider a binary association matrix x satisfying $\textstyle \sum _ { j = 1 } ^ { M } x _ { n j } \ = \ 1$ and $x _ { n j } ~ = ~ 0$ for all $j \notin \mathbb { M } _ { n } ^ { \mathrm { f } }$ . There exist bandwidth and power allocations (�, �) satisfying (23), (24), (26), and (34) if and only if

$$
\sum _ { n = 1 } ^ { N } \underline { { b } } _ { n j } x _ { n j } \leq B _ { j } , \qquad \forall j \in \mathbb { M } .\tag{40}
$$

Whenever (40) holds, one feasible recovery is

$$
b _ { n j } = \underline { { { b } } } _ { n j } x _ { n j } , \qquad p _ { n j } = p ^ { \mathrm { m a x } } x _ { n j } .\tag{41}
$$

Proof: See Appendix B.

The recovery in (41) selects one representative allocation among possibly many that share the same projected point; it excludes no feasible association or processing decision.

Remark 2. The exact projection relies on the absence of a transmission-energy objective or an aggregate power budget. If either feature is introduced, setting every selected link to $p ^ { \mathrm { m a x } }$ may no longer be optimal, and the variables $( b , p )$ must be retained in the optimization.

4) Mixed-Integer Exponential-Cone Reformulation: Introduce an auxiliary variable $\eta _ { n }$ for each UE and impose

$$
\eta _ { n } \leq \ln \left( \sum _ { j = 1 } ^ { M } w _ { n j } \right) .\tag{42}
$$

«Using the primal exponential cone

$$
\mathcal { K } _ { \exp } \triangleq \operatorname { c l } \left\{ \left( u , \nu , z \right) \in \mathbb { R } ^ { 3 } \ : \middle | \ : \nu > 0 , \ : u \geq \nu \exp ( z / \nu ) \right\} ,\tag{43}
$$

constraint (42) is equivalently represented as

$$
\left( \sum _ { j = 1 } ^ { M } w _ { n j } , \ 1 , \ \eta _ { n } \right) \in \mathcal { K } _ { \exp } .\tag{44}
$$

Because $\rho _ { n } ~ > ~ 0$ ¬and the objective maximizes $\begin{array} { r } { \sum _ { n } \rho _ { n } \eta _ { n } . } \end{array}$ (44) is active at the optimum, so introducing $\eta _ { n }$ linearizes the objective without changing it.

After applying Lemma 2, Subproblem A is reformulated as

$$
{ \bf P } _ { A 1 } ^ { \mathrm { p r o j } } : \quad \underset { { \bf x } , w , \eta } { \mathrm { m a x i m i z e } } \quad \sum _ { n = 1 } ^ { N } \rho _ { n } \eta _ { n }\tag{45}
$$

subject to (44), (11i), (40), (27), (25), (28), (38), and (11j).

Problem $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ is accordingly a mixed-integer exponentialcone problem: all constraints are affine except for the exponential-cone memberships, and the only discrete variables are the binary association variables.

Proposition 1. Problems $\mathbf { P } _ { A 1 }$ and $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ have the same optimal objective value. Furthermore, any global optimizer $( \mathbf { x } ^ { * } , w ^ { * } , \eta ^ { * } )$ of $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ , together with the communication recovery in (41), yields a global optimizer of $\mathbf { P } _ { A 1 }$ . Conversely, every feasible solution of $\mathbf { P } _ { A 1 }$ maps to a feasible solution of $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ with the same objective value by setting

$$
\eta _ { n } = \mathrm { l n } \Bigg ( \sum _ { j = 1 } ^ { M } w _ { n j } \Bigg ) .\tag{46}
$$

Proof. Lemma 2 and $\begin{array} { r } { \eta _ { n } ~ = ~ \ln ( \sum _ { j } w _ { n j } ) } \end{array}$ map every original feasible solution to an equally valued projected one, so $\mathrm { o p t } ( \mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } } ) \geq \mathrm { o p t } ( \mathbf { P } _ { A 1 } )$ . Conversely, the lemma recovers $( \pmb { b } , \pmb { p } )$ while $\rho _ { n } > 0$ makes (44) tight at optimum, yielding the reverse inequality and equivalence. ■

We solve $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ with the MOSEK mixed-integer conic solver [28] and recover $( \pmb { b } ^ { * } , \pmb { p } ^ { * } )$ via (41); the solver’s certified gap therefore applies to $\mathbf { P } _ { A 1 }$ itself.

## D. FREDI Stage B: Dual-Threshold Inference Optimization

FREDI Stage B acts on the inference layer: with $( \mathbf { x } ^ { * } , \pmb { b } ^ { * } , \pmb { p } ^ { * } , \pmb { w } ^ { * } )$ fixed, Subproblem $\mathbf { P } _ { B }$ separates across UEs. Because $\rho _ { n } > 0$ and ln(·) is strictly increasing, the threshold pair of UE � can be obtained from

$$
\mathbf { S P } _ { n } : \quad \operatorname* { m a x i m i z e } _ { \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } } \qquad \quad \mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )\tag{47}
$$

$$
\mathrm { s u b j e c t ~ t o } \qquad 0 \leq \alpha _ { n } ^ { l } \leq \alpha _ { n } ^ { u } \leq 1 ,\tag{48}
$$

$$
L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \le R _ { n } ^ { * } ,\tag{49}
$$

where $\begin{array} { r } { R _ { n } ^ { * } = \sum _ { j = 1 } ^ { M } x _ { n j } ^ { * } w _ { n j } ^ { * } . } \end{array}$ . By construction of Subproblem $\mathbf { A } ,$ $R _ { n } ^ { * } \ \leq \ | \Phi _ { n } | ;$ ; hence every feasible threshold solution satisfies the QoS chain $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \leq R _ { n } ^ { * } \leq | \Phi _ { n } |$

Algorithm 1: Dual-Threshold Selection for UE �   
Input: Observed confidence-score set $C _ { n } ;$ event labels;   
capacity $R _ { n } ^ { * }$   
Output: Optimal feasible thresholds $( \alpha _ { n } ^ { l , * } , \alpha _ { n } ^ { u , * } )$ and   
response envelope ${ \overline { { \kappa } } } _ { n }$   
1 Sort the unique values in $C _ { n } \cup \{ 0 , 1 \}$ into   
$c _ { 1 } < \dots < c _ { K } ;$   
2 Set $U ^ { \mathrm { b e s t } } \gets - \infty ;$   
3 Set $\overline { { { \kappa } } } _ { n } \gets 0 ;$   
4 for $i \gets 1$ to $K$ do   
5 for $j \gets i$ to � do   
6 Set $( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )  ( c _ { i } , c _ { j } ) ;$   
7 Evaluate $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ and $\mathcal { U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$   
8 if $L _ { n } > 0$ and $\mathcal { U } _ { n } > 0$ then   
9 $\overline { { \kappa } } _ { n } \gets \operatorname* { m a x } \{ \overline { { \kappa } } _ { n } , | \Phi _ { n } | \mathcal { U } _ { n } / L _ { n } \} ;$   
10 end   
11 if $L _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } ) \le R _ { n } ^ { * }$ and $\mathcal { U } _ { n } > U ^ { \mathrm { b e s t } }$ then   
12 Store the threshold pair and update $U ^ { \mathrm { b e s t . } }$   
13 end   
14 end   
15 end   
16 return $( \alpha _ { n } ^ { l , * } , \alpha _ { n } ^ { u , * } )$ and $\overline { { \kappa } } _ { n } ;$

Stage B therefore controls the detection–offloading behavior of the CNN through the label rule (2), rather than introducing fairness into the inference rule.

Role of monotonicity. Lemma 1 implies that the empirical utility and offloaded load are non-increasing in both thresholds. Hence, an optimal feasible pair lies on the lower boundary of the feasible threshold region, which permits directional pruning during an exact finite search.

Exact empirical search. The empirical utility and load are piecewise constant because they are obtained by counting samples. Their values can change only when a threshold crosses an observed confidence score. It is therefore sufficient to search threshold pairs drawn from the finite set of observed scores.

The exhaustive search is globally optimal over the empirical threshold domain. The additional envelope update evaluates (19) without changing the selected thresholds. Monotonicity can be used to stop exploring a row or column once all remaining pairs are dominated or infeasible, provided that any pruned candidates cannot increase either the feasible utility or the response envelope.

## IV. NUMERICAL RESULTS

We evaluate FREDI from three complementary perspectives. Scenario S1 isolates the dual-threshold (DT) inference component against a single-threshold (ST) policy; Scenario S2 evaluates the end-to-end fairness–utility trade-off and validates the two-stage decomposition; and Scenario S3 examines securityconstrained association and Stage-A scalability.

## A. Experimental Setup

1) Dataset, Models, and Implementation: We use a binary cats–dogs corpus with 4,001 cat and 4,006 dog images for training and 1,011 cat and 1,012 dog images for testing [29]. Images are resized to 224×224 and normalized using ImageNet statistics [30]. UE-side screening uses early-exit MobileNetV2 [31] and ShuffleNetV2 [32], with intermediate heads formed by global average pooling and a two-class linear projection. MobileNetV2 and ShuffleNetV2 contain 18 and 9 exits, respectively. These lightweight backbones are representative of confidence-based early-exit inference [3], [4].

Both models are trained on Google Cloud Platform using four vCPUs, 15 GB RAM, and one NVIDIA T4 GPU for 30 epochs, with batch size 64, learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and equally weighted cross-entropy losses across exits. Final-exit validation performance selects the checkpoint. For S1, exit-wise logits on the complete 2,023-image heldout test set are converted to positive-class softmax confidence traces; all threshold policies therefore operate on identical CNN outputs without retraining.

For S2 and S3, each UE generates $| \Phi _ { n } | = 1 0 0$ events per scheduling window and MobileNetV2 is used throughout. Unless otherwise stated, $M = 3 , B _ { j } = 2 0 ~ \mathrm { M H z } , D _ { n } = 1 2 8 ~ \mathrm { K i B }$ and $t _ { n } ^ { \mathrm { r } } ~ = ~ 1 0 0 ~ \mathrm { m s }$ . Random geometry generates physically consistent legitimate and eavesdropper channel gains but is not itself swept. The projected problem $\mathbf { P } _ { A 1 } ^ { \mathrm { p r o j } }$ is implemented using the MOSEK Fusion API [28]; minimum-bandwidth coefficients ${ \underline { { b } } } _ { n j }$ are computed by bisection and infeasible UE– ES pairs are removed before model construction. In S3-B, the relative MIP gap is $1 0 ^ { - 4 }$ and one warm-up solve is discarded before timing. Per-link caps are $b ^ { \mathrm { m a x } } = 2 0$ MHz, $p ^ { \mathrm { m a x } } = 0 . 2 \mathrm { W }$ (23 dBm) and $w ^ { \mathrm { m a x } } ~ = ~ 1 0 0$ events/window, the noise PSD is −174 dBm/Hz. UEs are uniformly distributed in a 70- m triangular serving region under distance based path loss (exponent 4) and Rayleigh fading. The threshold grid uses $\Delta _ { \alpha } = 0 . 0 0 2 5$

2) Scenario Design and Compared Methods: The following defines the compared policies, normalized loads, and statistical protocol.

Scenario S1: Threshold policies for early-exit screening. Scenario S1 isolates threshold selection from wireless and multi-UE resource-allocation effects. The allocated edgeprocessing budget is parameterized by the normalized ratio

$$
\eta _ { R } = \frac { R ^ { \star } } { | \Phi | } ,\tag{50}
$$

with $\eta _ { R } ~ \in ~ \{ 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$ . The proposed dual-threshold (DT) policy optimizes $( \alpha ^ { l } , \alpha ^ { u } )$ over $0 \leq \alpha ^ { l } \leq \alpha ^ { u } \leq 1$ , whereas the coupled single-parameter (ST) policy uses $\tau \in \ [ 0 . 5 , 1 ]$ with $\alpha ^ { l } = 1 - \tau$ and $\alpha ^ { u } = \tau .$ , the symmetric confidence rule used in conventional early-exit inference. Because $1 - \tau \leq \tau$ for every admissible �, the ST family is a one-dimensional subset of the DT family; S1 is therefore an ablation that isolates the value of decoupling the two thresholds, and DT is optimal for ST whenever the two are compared under the same processing budget. The comparison quantifies how much this additional degree of freedom is worth, and where the coupling becomes binding. Both use the same threshold-grid resolution $\Delta _ { \alpha } = 0 . 0 0 2 5$ . Refining to $\Delta _ { \alpha } = 0 . 0 0 1 2 5$ at the most sensitive MobileNetV2 point, $\eta _ { R } = 0 . 3 ,$ , leaves the selected optimum and reported metrics unchanged. TPR-optimal ties are resolved by lower offloaded load, lower normalized MACs, higher F1 score, and then threshold order.

Scenario S2: Utility and decomposition validation. Scenario S2 is the main system-level experiment. Each UE stream contains 50 positive and 50 negative held-out events sampled without replacement within a realization, with identical streams and network realization shared across methods. Sweeping $W _ { j } \in \{ 2 0 0 , 1 6 0 , 1 2 0 , 1 0 0 , 8 0 , 6 0 \}$ gives the normalized processing load

$$
\lambda _ { W } = \frac { \sum _ { n = 1 } ^ { N } \left| \Phi _ { n } \right| } { \sum _ { j = 1 } ^ { M } W _ { j } } = \frac { 2 0 0 } { W _ { j } } \in \{ 1 , 1 . 2 5 , 1 . 6 7 , 2 , 2 . 5 , 3 . 3 3 \} .\tag{51}
$$

FREDI is compared with a Sum-Utility benchmark over the same feasible associations and resources, maximizing $\textstyle \sum _ { n } { \mathcal { U } } _ { n }$ Since multiple optima may yield different fairness, we report the minimum and maximum Jain’s indices. We also validate the decomposition by exactly solving a finite empirical form of ${ \bf P } _ { 0 }$ at $\lambda _ { W } \in \{ 1 . 2 5 , 2 , 3 . 3 3 \}$ , maximizing $\begin{array} { r } { \sum _ { n } \rho _ { n } \ln ( \mathcal { U } _ { n } ) } \end{array}$ over the same feasible candidates.

Scenario S3: security constraints and scalability. Scenario S3 stress-tests FREDI Stage A through two complementary sub-experiments. In S3-A, FREDI is compared with an Unrestricted-PF reference under identical channels and resources. Unrestricted-PF removes only the UE–ES securityeligibility constraint while retaining the same proportionalfair resource allocation. In S3-B, FREDI is evaluated as the number of UEs increases.

In S3-A, seven UE security-demand profiles produce $\kappa _ { \mathrm { e l i g } } \in$ $\{ 1 , 0 . 8 9 , \hdots , 0 . 3 3 \}$ , each with 50 paired channel realizations sharing seeds across the two methods. Security-induced infeasibility is retained as an experimental outcome rather than resampled away.

In S3-B, � ∈ {6<sub>,</sub> 12<sub>,</sub> 24<sub>,</sub> 36<sub>,</sub> 60<sub>,</sub> 96<sub>,</sub> 144} with approximately balanced UE security tiers. To separate runtime growth from increasing resource scarcity, the per-ES capacities scale as

$$
W _ { j } ( N ) = \frac { 1 0 0 N } { 3 } , \qquad B _ { j } ( N ) = 2 0 \left( \frac { N } { 9 } \right) ~ \mathrm { M H z } .\tag{52}
$$

Fifty random realizations are used for each system size.

S1 is deterministic on the fixed test set. S2 and S3 use 50 realizations per operating point. Continuous quantities are reported as sample means with 95% confidence intervals unless otherwise stated; S3-B additionally reports median solve time, whereas the binary S3-A infeasibility rate uses 95% Wilson confidence intervals. For S2, the confidence intervals describe variability over the paired event-stream and network realizations drawn from the fixed held-out pool.

3) Performance Metrics: For S1, the primary detection metrics are true-positive rate (TPR), false-positive rate (FPR), and F1 score. The computation saving is

$$
G _ { \mathrm { c o m p } } = 1 - \frac { 1 } { | \Phi | } \sum _ { i \in \Phi } \frac { \mathrm { M A C } ( q _ { i } ) } { \mathrm { M A C } ( Q ) } ,\tag{53}
$$

where $q _ { i }$ is the exit used for event $i , \ \mathrm { M A C } ( q _ { i } )$ is the cumulative MAC count up to that exit, and � is the final exit.

For a nonnegative vector ${ \bf z } = \left( z _ { 1 } , \ldots , z _ { N } \right)$ , Jain’s index [33] is

$$
J ( \mathbf { z } ) = \frac { \left( \sum _ { n = 1 } ^ { N } \mathcal { Z } _ { n } \right) ^ { 2 } } { N \sum _ { n = 1 } ^ { N } \mathcal { Z } _ { n } ^ { 2 } } .\tag{54}
$$

For S2, let $U _ { n } \ \equiv \ { \mathcal U } _ { n } ( \alpha _ { n } ^ { l } , \alpha _ { n } ^ { u } )$ denote the final empirical

utility. We report the Jain’s indices

$$
J _ { R } = J ( \{ R _ { n } ^ { * } / | \Phi _ { n } | \} _ { n = 1 } ^ { N } ) , \qquad J _ { U } = J ( \{ U _ { n } \} _ { n = 1 } ^ { N } ) ,\tag{55}
$$

and the aggregate utility $\textstyle \sum _ { n } U _ { n }$ . To characterize the empirical response factor in (14) at the realized FREDI allocation, define

$$
\kappa _ { n } ^ { * } \equiv \kappa _ { n } ( R _ { n } ^ { * } ) = \frac { U _ { n } } { R _ { n } ^ { * } / | \Phi _ { n } | } , \qquad \mathrm { C V } _ { \kappa } = \frac { \mathrm { s t d } ( \{ \kappa _ { n } ^ { * } \} ) } { \mathrm { m e a n } ( \{ \kappa _ { n } ^ { * } \} ) } .\tag{56}
$$

Finally, the decomposition is compared with direct empirical optimization through

$$
F _ { 0 } ^ { \mathrm { F R E D I } } = \sum _ { n = 1 } ^ { N } \rho _ { n } \ln U _ { n } ^ { \mathrm { F R E D I } } , \qquad F _ { 0 } ^ { \mathrm { d i r e c t } } = \sum _ { n = 1 } ^ { N } \rho _ { n } \ln U _ { n } ^ { \mathrm { d i r e c t } } .\tag{57}
$$

The direct formulation excludes zero-utility empirical operating points because the logarithmic objective is undefined there.

The aggregate service ratio used in S3-A is

$$
S = \frac { \sum _ { n = 1 } ^ { N } R _ { n } } { \sum _ { n = 1 } ^ { N } | \Phi _ { n } | } .\tag{58}
$$

For ${ \mathrm { S } } 3 { \mathrm { - } } { \mathrm { A } } ,$ define $e _ { n j } = 1$ when UE � is security-eligible for ES $j ,$ and $e _ { n j } = 0$ otherwise. The eligible-association ratio is

$$
\kappa _ { \mathrm { e l i g } } = \frac { 1 } { N M } \sum _ { n = 1 } ^ { N } \sum _ { j = 1 } ^ { M } e _ { n j } ,\tag{59}
$$

and the infeasible-realization rate is

$$
P _ { \mathrm { i n f e a s } } = { \frac { K _ { \mathrm { i n f e a s } } } { K } } .\tag{60}
$$

We additionally report the actual feasible-pair ratio after preprocessing and ES processing and bandwidth utilization. For S3-B, we report MOSEK solve time and solver optimality status.

## B. Dual- versus Single-Threshold Screening Performance

Fig. 4 and Table III compare DT and ST under the same allocated edge-processing budget. Both policies use identical backbones and identical exit-wise confidence traces, so the differences isolate the effect of decoupling the two thresholds.

The coupling is most damaging under tight budgets. Because ST ties $\alpha ^ { l } = 1 - \tau { \mathrm { ~ t o ~ } } \alpha ^ { u } = \tau .$ , reducing the offloaded load forces $\tau \ \to \ 1$ , which raises the critical-exit threshold and widens the undecided interval I at the same time. For MobileNetV2 at $\eta _ { R } \in \{ 0 . 1 , 0 . 2 \}$ the load cap admits exactly one ST setting, $\tau \ = \ 1 \div$ then $( \alpha ^ { l } , \alpha ^ { u } ) = ( 0 , 1 )$ , I spans the whole confidence range, no sample crosses either threshold, and all are declared normal at the final exit by (2). Table III records the consequence — a true-positive rate of exactly zero, and no computation saving either, since nothing exits early. DT decouples the two decisions and attains TPRs of 0.197 and 0.396 with computation savings of 91.1% and 67.5% at the same budgets. ShuffleNetV2 does not collapse $( \tau = 0 . 9 9 7 5$ stays feasible), so the effect depends on how sharply a backbone concentrates its confidence traces; DT still leads it on every metric there, most visibly in computation saving (55.0% against 5.6% at $\eta _ { R } = 0 . 1 )$

At the intermediate budget $\eta _ { R } \ = \ 0 . 3$ , DT improves MobileNetV2 TPR from 0.576 to 0.593 and F1 from 0.729 to 0.742 while raising computation saving from 35.2% to 58.8%, at essentially the same offloaded load (29.0% to 29.9%). For ShuffleNetV2 the two policies select operating points with identical TPR, FPR, F1 and offloaded load, and DT gains only in computation saving (30.9% to 35.9%). At $\eta _ { R } = 0 . 5$ both backbones show small TPR and F1 gains, and the computation benefit becomes model-dependent.

![](images/0006e82de716145dbf1dc1e8cfbb5d0f4624773dfbdad6b68cb10aca8a04bef6.jpg)  
Fig. 4: True positive rate (top) and computation saving under the dual-threshold (DT) and coupled single-parameter (ST) policies as the allocated edge-processing budget $\eta _ { R }$ increases.

Above $\eta _ { R } = 0 . 5$ the comparison changes character, because ST cannot spend the budget it is given. Its offloaded set is {events whose confidence reaches � at some exit}, which on the balanced test corpus saturates near half the events as $\tau  0 . 5 \colon$ Table III shows the ST load rising only from 49.9% at $\eta _ { R } = 0 . 5$ to 50.5% at $\eta _ { R } = 0 . 8$ , while DT reaches 79.7%. The apparently unfavourable DT entries at $\eta _ { R } = 0 . 8$ — FPR 0.599 and F1 0.767 against 0.094 and 0.912 for ST — therefore compare a policy operating at its budget against one operating far below it. They also reflect the objective rather than the policy class: $\mathbf { S P } _ { n }$ maximizes TPR subject to the load cap and never penalizes false positives, so DT spends whatever capacity Stage A reserves. Where precision matters, the same machinery accepts an F1- or FPR-constrained utility without any change to Stage A.

The value of the second threshold is therefore not a uniform gain on every classification metric, but the removal of a structural constraint: it keeps detection alive where the coupled policy collapses, and it makes the full range of processing budgets reachable.

## C. Utility Performance and Decomposition Validation

The top panel of Fig. 5 compares the final-utility fairness of FREDI with the fairness envelope of Sum-Utility as processing contention increases. With sufficient processing capacity $( \lambda _ { W } ~ = ~ 1 )$ , both methods attain $J _ { U } ~ = ~ 1$ . FREDI remains essentially perfectly fair throughout the sweep, with mean $J _ { U } ~ \ge ~ 0 . 9 9 9 3$ . In contrast, the Sum-Utility envelope widens under heavier scarcity: at $\lambda _ { W } = 2 . 5 ,$ its mean lower and upper boundaries are approximately 0.9824 and 0.9953, and at $\lambda _ { W } ~ = ~ 3 . 3 3$ they become approximately 0.8561 and 0.9851. Hence, maximizing aggregate utility alone can admit substantially less fair optima even though an arbitrary solver tie-break may conceal this behavior.

TABLE III: Representative S1 operating points on the 2,023- image held-out test set.
<table><tr><td>CNN</td><td>Policy</td><td> $\eta _ { R }$ </td><td>TPR</td><td>FPR</td><td>F1</td><td>Offloaded load</td><td>Comp. saving</td></tr><tr><td>MobileNetV2</td><td>DT</td><td>0.1</td><td>0.197</td><td>0.003</td><td>0.328</td><td>10.0%</td><td>91.1%</td></tr><tr><td>MobileNetV2</td><td>ST</td><td>0.1</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.0%</td><td>0.0%</td></tr><tr><td>MobileNetV2</td><td>DT</td><td>0.2</td><td>0.396</td><td>0.003</td><td>0.566</td><td>20.0%</td><td>67.5%</td></tr><tr><td>MobileNetV2</td><td>ST</td><td>0.2</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.0%</td><td>0.0%</td></tr><tr><td>MobileNetV2</td><td>DT</td><td>0.3</td><td>0.593</td><td>0.005</td><td>0.742</td><td>29.9%</td><td>58.8%</td></tr><tr><td>MobileNetV2</td><td>ST</td><td>0.3</td><td>0.576</td><td>0.004</td><td>0.729</td><td>29.0%</td><td>35.2%</td></tr><tr><td>MobileNetV2</td><td>DT</td><td>0.8</td><td>0.994</td><td>0.599</td><td>0.767</td><td>79.7%</td><td>83.1%</td></tr><tr><td>MobileNetV2</td><td>ST</td><td>0.8</td><td>0.916</td><td>0.094</td><td>0.912</td><td>50.5%</td><td>67.8%</td></tr><tr><td>ShuffleNetV2</td><td>DT</td><td>0.3</td><td>0.579</td><td>0.020</td><td>0.724</td><td>30.0%</td><td>35.9%</td></tr><tr><td>ShuffleNetV2</td><td>ST</td><td>0.3</td><td>0.579</td><td>0.020</td><td>0.724</td><td>30.0%</td><td>30.9%</td></tr><tr><td>ShuffleNetV2</td><td>DT</td><td>0.8</td><td>0.975</td><td>0.615</td><td>0.753</td><td>79.5%</td><td>56.4%</td></tr><tr><td>ShuffleNetV2</td><td>ST</td><td>0.8</td><td>0.869</td><td>0.204</td><td>0.838</td><td>53.6%</td><td>56.6%</td></tr></table>

DT and ST are evaluated using identical exit-wise confidence traces. Bold entries highlight selected stronger detection or computation results at the same processing budget.

![](images/b999969eca64b6fd9d0981d7b18440cf23c13d2250fbf2c2ebb63d1066a28c46.jpg)  
Fig. 5: Jain’s index $J _ { U }$ (top) and aggregate utility (bottom) for FREDI and the Sum-Utility benchmark as the normalized processing load $\lambda _ { W }$ increases.

The fairness improvement is obtained with a small aggregate-utility cost. The bottom panel of Fig. 5 shows that Sum-Utility, by construction, gives the largest $\textstyle \sum _ { n } U _ { n }$ , while FREDI remains close over the full load range. $\mathrm { A t } \lambda _ { W } = 1$ , both attain an aggregate utility of 6. As contention increases, the largest mean difference over the six operating points is 0.0692 at $\lambda _ { W } = 2$ , where FREDI and Sum-Utility obtain 5.5252 and 5.5944, respectively. Under severe scarcity $( \lambda _ { W } = 3 . 3 3 )$ , the corresponding values are 3.5712 and 3.5936. Thus, FREDI maintains near-uniform user utility while sacrificing little aggregate detection utility.

Table IV assesses the decomposition from two complementary viewpoints. First, the normalized processing allocation is perfectly balanced at every tested load, $J _ { R } = 1$ , and the final utility remains extremely close to this allocation-level fairness, with $J _ { U } \ge 0 . 9 9 9 3$ . At the realized FREDI operating points, the mean coefficient of variation of $\kappa _ { n } ^ { * } = U _ { n } / ( R _ { n } ^ { * } / | \Phi _ { n } | )$ never exceeds 2.71%. The repeated small cross-UE dispersion values show that the realized response residual remains balanced across UEs over the load sweep; they do not by themselves establish the per-UE constancy required by Corollary 1.1 or evaluate the response envelope ${ \overline { { \kappa } } } _ { n }$ in Theorem 1.

TABLE IV: Empirical validation of the FREDI decomposition and direct finite empirical optimization of $\mathbf { P } _ { 0 } .$
<table><tr><td> $\lambda _ { W }$ </td><td> $J _ { R }$ </td><td> $J _ { U }$ </td><td> $\mathrm { C V } _ { \kappa }$  (%)</td><td> $F _ { 0 } ^ { \mathrm { F R E D I } }$ </td><td> $F _ { 0 } ^ { \mathrm { d i r e c t } }$ </td></tr><tr><td>1.00</td><td>1.0</td><td>1.0000</td><td>0.00</td><td></td><td></td></tr><tr><td>1.25</td><td>1.0</td><td>0.9999</td><td>1.06</td><td> $- 0 . 0 4 0 9 \pm 0 . 0 0 6 5$ </td><td> $- 0 . 0 0 4 0 \pm 0 . 0 0 2 3$ </td></tr><tr><td>1.67</td><td>1.0</td><td>0.9995</td><td>2.27</td><td>一</td><td></td></tr><tr><td>2.00</td><td>1.0</td><td>0.9993</td><td>2.71</td><td> $- 0 . 4 9 6 9 \pm 0 . 0 1 3 8$ </td><td> $- 0 . 4 2 2 8 \pm 0 . 0 1 2 8$ </td></tr><tr><td>2.50</td><td>1.0</td><td>0.9996</td><td>2.06</td><td></td><td></td></tr><tr><td>3.33</td><td>1.0</td><td>0.9998</td><td>1.34</td><td> $- 3 . 1 1 3 8 \pm 0 . 0 1 0 9$ </td><td> $- 3 . 1 0 0 8 \pm 0 . 0 0 8 4$ </td></tr></table>

Values are averaged over 50 paired realizations. $J _ { R } \ = \ J ( \{ R _ { n } ^ { * } / | \Phi _ { n } | \} )$ and $J _ { U } ~ = ~ J ( \{ \bar { U } _ { n } \} )$ . The $F _ { 0 }$ entries report mean ± 95% confidenceinterval half-width. Direct- ${ \bf P } _ { 0 }$ validation is evaluated at three representative processing loads. Bold entries mark the most stringent observed FREDI fairness-consistency point, where the minimum $J _ { U }$ and maximum $\operatorname { C V } _ { \kappa }$ occur.

Second, we compare FREDI with direct finite empirical optimization of the original logarithmic objective. All 150 direct- $\mathbf { \nabla } \cdot \mathbf { P } _ { 0 }$ validation instances (50 paired realizations at each of three representative loads) are solved to optimal integer status. As required, $F _ { 0 } ^ { \mathrm { d i r e c t } } ~ \ge ~ F _ { 0 } ^ { \mathrm { F R E D I } }$ in every case within numerical tolerance. The mean objectives remain close, particularly under severe scarcity, where they are −3.1138 and −3.1008 for FREDI and direct optimization, respectively. This direct comparison gives a realized empirical assessment of the end-to-end loss, complementing the general upper bound in Theorem 1. Together, the fairness consistency, small $\mathrm { C V } _ { \kappa } ,$ and direct-objective comparison support the FREDI two-stage decomposition over the evaluated operating regime without presuming exact end-to-end equivalence.

## D. Impact of Security Constraints and System Scalability

1) Security-Constrained Association: Fig. 6 jointly shows the service and feasibility effects of security-induced restrictions on UE–ES association. As shown in the top panel, when $K _ { \mathrm { e l i g } } \ge 0 . 7 7 8$ , FREDI serves essentially all offered workload and remains close to the eligibility-unconstrained PF resourceallocation upper bound (Unrestricted-PF). Below this region, the service ratio deteriorates rapidly, falling to about 0.661 at $\kappa _ { \mathrm { e l i g } } = 0 . 4 4 4$ and to 0.333 at $\kappa _ { \mathrm { e l i g } } = 0 . 3 3 3$

The bottom panel of Fig. 6 shows the corresponding infeasible-realization rate of the security-constrained FREDI formulation. All realizations remain feasible for $\kappa _ { \mathrm { e l i g } } \geq 0 . 7 7 8$ whereas $P _ { \mathrm { i n f e a s } }$ increases to 0.14 at $\begin{array} { r } { \kappa _ { \mathrm { e l i g } } \ = \ 0 . } \end{array}$ .444 and 0.22 at $\kappa _ { \mathrm { e l i g } } = 0 . 3 3 3$ . Hence, moderate security restrictions can be absorbed without loss of feasibility, whereas severe restrictions simultaneously reduce service capability and increase the probability of an infeasible association/resource-allocation instance.

The mechanism is resource fragmentation rather than a lack of aggregate capacity. The measured feasible-pair ratio closely tracks $K _ { \mathrm { e l i g } } .$ , confirming that security eligibility dominates the sweep, while the maximum ES bandwidth utilization rises from about 0.30 to 0.64. Under the most restrictive profile, mean processing utilization is only about 0.333 although one eligible ES is fully utilized. Capacity therefore remains stranded at ESs that some UEs are not permitted to access. Moreover, Jain fairness can still be high when service is poor (e.g., $J = 1$ at $\kappa _ { \mathrm { e l i g } } = 0 . 3 3 3 )$ , showing that fairness and service capability must be interpreted jointly.

![](images/b2dd7eb3d8766f773677d2c4e4ee1168d481ce5f1416ece5ccbd26fab3e890b1.jpg)  
Fig. 6: Service ratio (top) and infeasible-realization rate (bottom) of FREDI as the eligible association ratio $K _ { \mathrm { e l i g } }$ increases.

![](images/6a9fc1aa9d17ee24be0f73f425975a622d1b687dc7308b6d81d1c63f161b5957.jpg)  
Fig. 7: Optimization time of FREDI Stage A as the number of UEs � increases.

2) Solver Scalability: Fig. 7 reports the computational scalability of FREDI Stage A, with communication and processing resources scaled with � to maintain a comparable operating load. The median MOSEK solve time increases from about 2 ms at $N \ = \ 6$ to 59 ms at � = 144 and remains below 0.1 s even for the largest tested system. The wider interquartile range at large � reflects realization-dependent mixed-integer search induced by different feasible UE–ES association sets.

All 350 measured instances reach an optimal integer status within the prescribed relative-gap tolerance of $1 0 ^ { - 4 }$ . Together with preprocessing that removes infeasible UE–ES pairs before model construction, these results show that the FREDI security-constrained resource-allocation formulation remains computationally practical over the investigated system sizes without relaxing solution quality.

## V. CONCLUSION

This paper presented FREDI, a secure cooperative wireless edge-intelligence framework that jointly coordinates dualthreshold inference, UE–ES association, uplink bandwidth and power, and shared ES processing. Stage A uses proportional fairness, feasible-pair preprocessing, and an exact communication-variable projection to produce a mixed-integer exponential-cone formulation; Stage B exactly searches the empirical confidence-score domain for each UE. A computable response envelope bounds the loss relative to the empirical global optimum and yields a proportional-response condition for global optimality. Numerically, FREDI maintains nearperfect utility fairness $( J _ { U } ~ \ge ~ 0 . 9 9 9 3$ on average) while remaining close to the Sum-Utility benchmark; direct optimization of representative ${ \bf P } _ { 0 }$ instances further supports the twostage decomposition. The results also quantify the effects of security-constrained association and confirm practical Stage-A scalability over the tested system sizes.

## APPENDIX A PROOF OF LEMMA 1

Proof. Consider two feasible threshold pairs $( \alpha ^ { l } , \alpha ^ { u } )$ and $( \widetilde { \alpha } ^ { l } , \widetilde { \alpha } ^ { u } )$ such that $\widetilde { \alpha } ^ { l } \geq \alpha ^ { l }$ and $\widetilde { \alpha } ^ { u } \ \geq \ \alpha ^ { u }$ . Since the CNN e e eis fixed, the confidence trace ${ C _ { q } ^ { ( k ) } } _ { q = 1 } ^ { L }$ of each event $I _ { k }$ is independent of the thresholds.

Suppose that $I _ { k }$ is classified as critical under $( \widetilde { \alpha } ^ { l } , \widetilde { \alpha } ^ { u } )$ , first at layer $q ^ { \star }$ . For every $q < q ^ { \star }$ e e, the event has not terminated as normal, so $C _ { q } ^ { ( k ) } > \widetilde { \alpha } ^ { \bar { l } } \geq \alpha ^ { l } . \mathrm { { \bar { I } f } } C _ { q } ^ { ( k ) } \geq \alpha ^ { u }$ at any such layer, then $I _ { k }$ eis already classified as critical under $( \alpha ^ { l } , \alpha ^ { u } )$ ; otherwise it remains undecided. At layer $q ^ { \star }$ , critical classification under the larger thresholds gives $\check { C _ { q ^ { \star } } ^ { ( k ) } } \overset { ^ { * } } { \geq } \widetilde { \alpha } ^ { u } \geq \alpha ^ { u }$ , so the event is critical under $( \alpha ^ { l } , \alpha ^ { u } )$ eas well. The same argument applies at the final layer, while the equal-threshold convention in (2) preserves this implication when the undecided interval vanishes.

Hence, $I _ { k } \ \in \ { \widehat \Phi } ^ { \mathrm { c r i t i c a l } } ( { \widetilde \alpha } ^ { l } , { \widetilde \alpha } ^ { u } ) \quad \Longrightarrow \quad I _ { k } \ \in \ { \widehat \Phi } ^ { \mathrm { c r i t i c a l } } ( \alpha ^ { l } , \alpha ^ { u } )$ and therefore $\begin{array} { r l r } { \widehat { \Phi } ^ { \mathrm { c r i t i c a l } } ( \widetilde { \alpha } ^ { l } , \widetilde { \alpha } ^ { u } ) } & { { } \subseteq } & { \widehat { \Phi } ^ { \mathrm { c r i t i c a l } } ( \alpha ^ { l } , \alpha ^ { u } ) . } \end{array}$ , which eproves (12). Consequently, $| \widehat { \Phi } ^ { \mathrm { c r i t i c a l } } |$ bis non-increasing in either threshold. ■

## APPENDIX B PROOF OF LEMMA 2

Proof. For a feasible UE–ES pair satisfying $\gamma _ { n j } > \gamma _ { n } ^ { \mathrm { E V } }$ and fixed $b > 0 .$

$$
\frac { \partial r _ { n j } ^ { \mathrm { s e } } } { \partial p } = \frac { b } { \ln 2 } \left( \frac { \gamma _ { n j } } { b + \gamma _ { n j } p } - \frac { \gamma _ { n } ^ { \mathrm { E V } } } { b + \gamma _ { n } ^ { \mathrm { E V } } p } \right) > 0 ,\tag{61}
$$

so the rate increases in $p _ { \cdot }$ . For fixed $p > 0 ;$ write $r _ { n i } ^ { \mathrm { s e } } ( b , p ) =$ $b q _ { n j } ( p / b )$ , where $q _ { n j }$ is concave and $q _ { n j } ( 0 ) \ = \ \mathrm { \bar { 0 } } .$ . Since $q _ { n j } ( z ) / z$ is non-increasing, $r _ { n j } ^ { \mathrm { s e } } ( b , p ) = p q _ { n j } ( p / b ) / ( p / b )$ is non-decreasing in �.

For necessity, any feasible selected link satisfies $r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p ^ { \mathrm { m a x } } ) \ge r _ { n j } ^ { \mathrm { s e } } ( b _ { n j } , p _ { n j } ) \ge \bar { r } _ { n }$ and hence $\begin{array} { r } { b _ { n j } \ \geq \ \underline { { b } } _ { n j } . } \end{array}$ For an unselected link, $b _ { n j } \geq \underline { { b } } _ { n j } x _ { n j }$ is immediate. Summing over UEs gives

$$
\sum _ { n } \underline { { b } } _ { n j } x _ { n j } \leq \sum _ { n } b _ { n j } \leq B _ { j } ,\tag{62}
$$

which proves (40). Conversely, if this constraint holds, set $b _ { n j } ~ = ~ \underline { { { b } } } _ { n j } x _ { n j }$ and $p _ { n j } ~ = ~ p ^ { \mathrm { m a x } } x _ { n j }$ . Each selected feasible pair meets the secure-rate requirement by the definition of ${ \underline { { b } } } _ { n j } ;$ each unselected pair has zero rate and demand. The per-link bounds follow from $\underline { { b } } _ { n j } ~ \leq ~ b ^ { \mathrm { m a x } }$ on $\mathbb { M } _ { n } ^ { \mathrm { f } } .$ , and the aggregate constraint holds by construction. Thus, the recovered communication allocation is feasible. ■

## APPENDIX C

PROOF OF THEOREM 1

Proof. For any admissible capacity $\xi ,$ let $( \alpha _ { n } ^ { l , \xi } , \alpha _ { n } ^ { u , \xi } )$ attain $V _ { n } ( \xi )$ and let $L _ { n } ^ { \xi }$ be its load. Since $0 < L _ { n } ^ { \xi } \leq \xi .$

$$
\kappa _ { n } ( \xi ) = \frac { | \Phi _ { n } | V _ { n } ( \xi ) } { \xi } \leq \frac { | \Phi _ { n } | \mathcal { U } _ { n } ( \alpha _ { n } ^ { l , \xi } , \alpha _ { n } ^ { u , \xi } ) } { L _ { n } ^ { \xi } } \leq \overline { { \kappa } } _ { n } .
$$

Conversely, evaluating $V _ { n } ( \xi )$ at the load of a candidate attaining (19) gives the reverse inequality for the supremum; hence, (19) is exactly the upper envelope of $\kappa _ { n } ( \xi )$ over the positive-utility capacity domain. For the resource allocation of a global optimizer of $\mathbf { P } _ { 0 } .$ , replacing any threshold pair by a maximizer in (13) cannot decrease the objective. Its resource variables also satisfy the Stage-A constraints; hence, its processing-coverage term is no larger than the globally optimal Stage-A value $A ( \mathbf { R } ^ { * } )$ . Applying (15) and the preceding response bound therefore gives

$$
F _ { 0 } ^ { \star } \leq A ( \mathbf { R } ^ { \ast } ) + \sum _ { n = 1 } ^ { N } \rho _ { n } \ln \overline { { \kappa } } _ { n } .
$$

The exact Stage-B searches yield $\begin{array} { r l r } { F _ { 0 } ^ { \mathrm { F R E D I } } } & { { } = } & { A ( { \bf R } ^ { * } ) { \bf \Psi } + { \bf \Psi } } \end{array}$ $\textstyle \sum _ { n } \rho _ { n }$ ln $\kappa _ { n } ( R _ { n } ^ { * } )$ . Subtracting the latter identity from the upper bound proves (20); the lower inequality follows because the FREDI solution is feasible for ${ \bf P } _ { 0 }$ ■

## REFERENCES

[1] K. B. Letaief, Y. Shi, J. Lu, and J. Lu, “Edge artificial intelligence for 6g: Vision, enabling technologies, and applications,” IEEE journal on selected areas in communications, vol. 40, no. 1, pp. 5–36, 2021.

[2] J. Mendez, K. Bierzynski, M. P. Cuéllar, and D. P. Morales, “Edge intelligence: concepts, architectures, applications, and future directions,” ACM Transactions on Embedded Computing Systems (TECS), vol. 21, no. 5, pp. 1–41, 2022.

[3] S. Teerapittayanon, B. McDanel, and H.-T. Kung, “Branchynet: Fast inference via early exiting from deep neural networks,” in 2016 23rd international conference on pattern recognition (ICPR). IEEE, 2016, pp. 2464–2469.

[4] G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Weinberger, “Multi-scale dense networks for resource efficient image classification,” in International Conference on Learning Representations, 2018.

[5] H. Rahmath P, V. Srivastava, K. Chaurasia, R. G. Pacheco, and R. S. Couto, “Early-exit deep neural network-a comprehensive survey,” ACM Computing Surveys, vol. 57, no. 3, pp. 75:1–75:37, 2025.

[6] C. Chow, “On optimum recognition error and reject tradeoff,” IEEE Transactions on information theory, vol. 16, no. 1, pp. 41–46, 1970.

[7] G. Fumera, F. Roli, and G. Giacinto, “Reject option with multiple thresholds,” Pattern recognition, vol. 33, no. 12, pp. 2099–2101, 2000.

[8] P. L. Bartlett and M. H. Wegkamp, “Classification with a reject option using a hinge loss,” Journal ofMachine Learning Research, vol. 9, no. 8, 2008.

[9] Z. Liu, Q. Lan, and K. Huang, “Resource allocation for multiuser edge inference with batching and early exiting,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 4, pp. 1186–1200, 2023.

[10] J.-W. Kim and H.-S. Lee, “Early exiting-aware joint resource allocation and dnn splitting for multisensor digital twin in edge–cloud collaborative system,” IEEE Internet of Things Journal, vol. 11, no. 22, pp. 36 933– 36 949, 2024.

[11] Y. Zheng, T. Zhang, X. Mu, Y. Liu, and R. Huang, “Joint semantic transmission and resource allocation for intelligent computation task offloading in mec systems,” IEEE Transactions on Wireless Communications, vol. 24, no. 10, pp. 8756–8770, 2025.

[12] X. Liu, H. Zhai, X. Zhou, H. Zhang, and Q. Liu, “Joint resource allocation and computation offloading for dnn inference with model partition and early exit in mec networks,” Chinese Journal of Electronics, vol. 35, no. 1, pp. 215–232, 2026.

[13] Z. Zhang, T. Zhang, T. Shi, and Y. Wang, “Joint service placement and resource allocation for long-term dnn inference accuracy in dynamic mec networks,” IEEE Transactions on Vehicular Technology, 2025, early access.

[14] X. Yuan, N. Li, Q. Chen, W. Xu, and S. Guo, “Era: A qoe-aware collaborative inference algorithm for noma-based edge intelligence,” IEEE Transactions on Mobile Computing, 2025, early access.

[15] Y. Zhou, C. You, and K. Huang, “Communication efficient cooperative edge ai via event-triggered computation offloading,” IEEE Transactions on Communications, vol. 74, pp. 3190–3205, 2026.

[16] Y. Xu, T. Mohammed, M. Di Francesco, and C. Fischione, “Distributed assignment with load balancing for dnn inference at the edge,” IEEE Internet of Things Journal, vol. 10, no. 2, pp. 1053–1065, 2023.

[17] T. T. Vu, N. H. Chu, K. T. Phan, D. T. Hoang, D. N. Nguyen, and E. Dutkiewicz, “Energy-based proportional fairness in cooperative edge computing,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 12 229–12 246, 2024.

[18] H. Xiao, Q. Pei, X. Song, and W. Shi, “Authentication security level and resource optimization of computation offloading in edge computing systems,” IEEE Internet of Things Journal, vol. 9, no. 15, pp. 13 010– 13 023, 2022.

[19] K. Peng, P. Xiao, S. Wang, and V. C. Leung, “Scof: Security-aware computation offloading using federated reinforcement learning in industrial internet of things with edge computing,” IEEE Transactions on Services Computing, vol. 17, no. 4, pp. 1780–1792, 2024.

[20] A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of quantum neural networks,” Nature computational science, vol. 1, no. 6, pp. 403–409, 2021.

[21] C. Ren, R. Yan, H. Zhu, H. Yu, M. Xu, Y. Shen, Y. Xu, M. Xiao, Z. Y. Dong, M. Skoglund et al., “Toward quantum federated learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 9, pp. 15 580–15 600, 2025.

[22] C. Long, M. Huang, X. Ye, Y. Futamura, and T. Sakurai, “Hybrid quantum-classical-quantum convolutional neural networks,” Scientific Reports, vol. 15, no. 1, p. 31780, 2025.

[23] F. Fan, Y. Shi, T. Guggemos, and X. X. Zhu, “Hybrid quantum-classical convolutional neural network model for image classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 12, pp. 18 145–18 159, 2024.

[24] A. Patil and M. Rane, “Convolutional neural networks: an overview and its applications in pattern recognition,” in International Conference on Information and Communication Technology for Intelligent Systems. Springer, 2020, pp. 21–30.

[25] A. D. Wyner, “The wire-tap channel,” Bell system technical journal, vol. 54, no. 8, pp. 1355–1387, 1975.

[26] F. P. Kelly, A. K. Maulloo, and D. K. H. Tan, “Rate control for communication networks: shadow prices, proportional fairness and stability,” Journal of the Operational Research society, vol. 49, no. 3, pp. 237–252, 1998.

[27] S. Boyd and L. Vandenberghe, Convex Optimization. Cambridge, U.K.: Cambridge University Press, 2004.

[28] MOSEK ApS, MOSEK Fusion API for Python 11.2.2, 2026, [Online]. Available: https://docs.mosek.com.

[29] S. Schubert, “Cats and dogs dataset to train a dl model,” 2018, kaggle. [Online]. Available: https://www.kaggle.com/datasets/tongpython/catand-dog.

[30] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[31] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in 2018 IEEE/CVF conference on computer vision and pattern recognition. Ieee, 2018, pp. 4510–4520.

[32] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “Shufflenet v2: Practical guidelines for efficient cnn architecture design,” in European conference on computer vision. Springer, 2018, pp. 122–138.

[33] R. Jain, D.-M. Chiu, and W. Hawe, “A quantitative measure of fairness and discrimination,” Digital Equipment Corporation, Tech. Rep. TR-301, 1984.