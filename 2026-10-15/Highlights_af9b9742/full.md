## Highlights

CARD: Closing the Teacher-Student Loop for Adversarial Robustness Distillation Zhi Li, Haowei Liu, Hongchen Yang, Xiaoxuan Wang, Song Gao, Zhenli He, Wei Zhou, Shaowen Yao

• Adversarial distillation is extended from one-way transfer to closed-loop feedback.

• Reliability-enhanced student feedback guides targeted teacher refinement.

• Theory characterizes when weak-to-strong feedback preserves correct predictions.

• Refined-teacher gains improve students without redesigning the base objective.

# CARD: Closing the Teacher-Student Loop for Adversarial Robustness Distillation

Zhi Li<sup>a</sup>, Haowei Liu<sup>a</sup>, Hongchen Yang<sup>a</sup>, Xiaoxuan Wang<sup>b</sup>, Song Gao<sup>a,∗</sup>, Zhenli He<sup>a</sup>, Wei Zhou<sup>a</sup> and Shaowen Yao<sup>a</sup>

<sup>a</sup>Engineering Research Center ofCyberspace and School ofSoftware and AI, Yunnan University, Kunming, China <sup>b</sup>School ofInformation Science and Technology, Yunnan Normal University, Kunming, China

## A R T I C L E I N F O

Keywords: Adversarial Robustness Distillation Adversarial Robustness Knowledge Distillation Weak-to-Strong Learning Teacher-Student Feedback

## A BS T RA C T

Adversarial robustness distillation typically treats a robust teacher as a fixed knowledge source and transfers robustness to a compact student in a one-way manner. This paradigm terminates teacherstudent interaction once the student is distilled, thereby discarding potentially useful feedback from the distilled student. However, exploiting such feedback is nontrivial because indiscriminate weak-tostrong transfer may propagate student errors to the teacher. To address this issue, we propose Cyclic Adversarial Robustness Distillation (CARD), a closed-loop framework that turns the distilled student into a feedback source for refining the robust teacher and subsequently transfers the teacher gains back to the same student. CARD first performs strong-to-weak adversarial distillation and then uses reliability-enhanced student feedback to refine the robust teacher. The refinement combines inverse adversarial student feedback with teacher-student discrepancy maximization, after which the refined teacher is used to continue training the same student under the unchanged base distillation objective. We further establish a sample-level suficient condition for reliable weak-to-strong feedback transfer and derive a distribution-level upper bound on the refined teacher’s risk. Experiments on CIFAR-10, CIFAR-100, and TinyImageNet-10 across multiple teacher-student architectures and base distillation methods show that CARD improves adversarial robustness in most evaluated settings while generally preserving clean accuracy. These results show that the distilled student can provide useful feedback to the teacher, enabling robustness transfer to proceed beyond the conventional one-way teacher-tostudent paradigm.

## 1. Introduction

Lightweight neural networks are increasingly deployed on mobile, edge, and embedded devices because of their low computational and storage costs [41]. However, their limited capacity makes it dificult to maintain both clean accuracy and adversarial robustness [6, 8, 37]. Adversarial training improves robustness by optimizing models against adversarial perturbations [28], but its efectiveness often relies on substantial model capacity and computational cost [42, 21]. Adversarial robustness distillation provides a practical alternative by transferring robust knowledge from a highcapacity robust teacher to a compact student [13].

Existing adversarial-distillation methods mainly focus on improving the transfer of robust knowledge from teacher to student. Representative approaches employ robust soft labels [43], adaptive teacher guidance [16], feature or gradient alignment [1, 24], and adversarial trajectory modeling [9].

Despite these advances, most methods still follow a oneway teacher-to-student paradigm in which the teacher remains fixed throughout distillation. Once distillation is completed, the student becomes the endpoint of the transfer process. This one-way formulation may leave potentially useful teacher-student interactions underexploited. Although the distilled student remains weaker than the robust teacher overall, its predictions can still provide complementary feedback, while teacher-student disagreement can reveal inputs on which their behaviors difer substantially. This observation motivates us to investigate whether feedback from the distilled student can be used to refine a stronger robust teacher. However, weak-to-strong feedback can be risky, as aligning a strong teacher with predictions from a weaker student may propagate student errors and degrade the teacher. The central challenge is therefore to exploit student feedback while reducing the risk of erroneous knowledge transfer and focusing teacher refinement on informative regions of the input space.

In this paper, we propose Cyclic Adversarial Robustness Distillation (CARD), a closed-loop framework that extends conventional one-way robustness transfer into a teacherstudent feedback cycle. As illustrated in Figure 1, CARD consists of strong-to-weak adversarial distillation, weak-tostrong teacher refinement, and subsequent strong-to-weak gain transfer. CARD first trains a compact student under the guidance of a pretrained robust teacher using an existing adversarial-distillation objective. The distilled student is then frozen and used to refine the teacher through two complementary mechanisms. For clean-input refinement, CARD constructs inverse adversarial examples that reduce the student’s classification loss and uses the resulting student predictions to supervise the teacher on the corresponding clean inputs. For robust refinement, CARD identifies perturbed inputs by maximizing the discrepancy between the clean student prediction and the teacher prediction on the perturbed input, and then uses the clean student prediction to supervise the teacher on these inputs. After refinement, the teacher is frozen and used to continue training the same student from its previous checkpoint under the unchanged base distillation objective. In this way, CARD reorganizes the teacher-student training process into a closed feedback loop without requiring a redesign of the base distillation loss.

A key question is whether feedback from a weaker student can be used to refine a stronger teacher without compromising correct predictions. We therefore provide theoretical analysis of the weak-to-strong refinement process. At the sample level, we establish a suficient condition under which aligning the teacher with student feedback preserves the correct prediction. At the distribution level, we further derive an upper bound on the refined teacher’s classification risk. These results characterize when weak-to-strong feedback can be transferred reliably and provide theoretical support for the proposed closed-loop formulation.

The main contributions of this work are summarized as follows:

• We introduce CARD, a closed-loop adversarial robustness distillation framework in which a distilled student is used to refine the robust teacher, and the refined teacher then guides further training of the same student. CARD retains the original adversarialdistillation objective for student training, allowing it to augment existing methods without redesigning their base losses.

• We develop a student-feedback-based teacher-refinement mechanism that improves feedback quality and focuses refinement on regions of teacher-student disagreement. Inverse adversarial examples provide lowerloss student feedback for clean-input refinement, and discrepancy maximization identifies adversarial inputs for robust refinement. We further establish a sample-level suficient condition for preserving correct predictions under weak-to-strong feedback and derive a distribution-level upper bound on the refined teacher’s classification risk.

• We conduct extensive experiments on CIFAR-10, CIFAR-100, and TinyImageNet-10 across multiple teacher-student architectures and base adversarialdistillation methods. CARD improves adversarial robustness over the corresponding base methods in most evaluated settings while generally preserving clean accuracy, and ablation studies further verify the efectiveness of the proposed teacher-refinement and closed-loop mechanisms.

![](images/06bf67bfc63c3a9d078590cf565f700e07cf3f8a11e66d7b073d7ff8c5f0ab4e.jpg)  
Figure 1: From one-way to closed-loop adversarial robustness distillation. Conventional adversarial distillation treats the robust teacher as a fixed knowledge source and transfers robustness only from teacher to student. CARD closes the teacher-student loop by using the distilled student to refine the robust teacher and then using the refined teacher to continue training the same student. Snowflake and flame symbols denote frozen and trainable networks, respectively.

## 2. Related Work

In this section, we review the three lines of research most closely related to CARD, including adversarial training, adversarial robustness distillation, and teacher adaptation with weak-to-strong feedback.

## 2.1. Adversarial Training

Adversarial training improves model robustness by training against adversarial examples generated within a prescribed perturbation set. PGD-AT [28] formulated adversarial training as a min–max optimization problem and has become a widely adopted baseline for adversarial robustness. Subsequent studies have improved adversarial training from diferent perspectives, including adapting attack dificulty during training [3, 40, 35, 18], mitigating the trade-of between clean accuracy and adversarial robustness [39, 34, 36], and redesigning the training objective [29, 7]. Other approaches further enhance robust learning by exploiting informative training samples and local neighborhood structures [25, 26, 10, 30, 12]. Despite this progress, achieving strong adversarial robustness remains dificult for compact networks with limited representational capacity. This limitation motivates adversarial robustness distillation, which transfers robust knowledge from a high-capacity teacher to a lightweight student.

## 2.2. Adversarial Robustness Distillation

Adversarial robustness distillation transfers robust knowledge from a pretrained teacher to a compact student, aiming to improve student robustness without increasing its model capacity. ARD [13] incorporates robust teacher supervision into the adversarial training of the student. Subsequent methods improve teacher supervision through robust soft labels or adaptive guidance [43, 42], and enrich knowledge transfer through feature, gradient, or adversarial-trajectory alignment [1, 24, 9]. Other approaches further adapt the distillation process to the current student. PeerAiD [19] jointly trains an online peer tutor with the student, and AdaAD [16] exploits teacher-student discrepancy to generate adaptive adversarial examples for student training.

![](images/80d3ed59374563e98cdc227d05ccdb0444191b240b66e03c423118eaea467287.jpg)  
Figure 2: Overview of CARD. Stage 1 performs strong-to-weak adversarial distillation from a pretrained robust teacher to a compact student. Stage 2 freezes the first-round student and uses inverse adversarial examples and teacher adversarial examples to refine the pretrained teacher. Stage 3 replaces the initial teacher with the refined teacher and continues training from the first-round student checkpoint, transferring the teacher improvement back to the student.

Despite these advances, existing methods primarily focus on improving knowledge transfer from the teacher to the student. The distilled student is generally treated as the recipient of robust knowledge rather than as a source of feedback for refining the teacher after distillation. Although both AdaAD and CARD exploit teacher-student discrepancy, the discrepancy serves diferent roles in the two frameworks. AdaAD uses it to generate adaptive adversarial examples for student training, whereas CARD employs discrepancy maximization in a separate weak-to-strong stage to identify perturbed inputs where the frozen student’s predictions are used to refine the robust teacher.

## 2.3. Teacher Adaptation and Weak-to-Strong Learning

Several recent adversarial-distillation methods move beyond a strictly fixed supervisory model. PeerAiD [19] jointly trains an online peer tutor with the student, and CIARD [27] adopts a multi-teacher cyclic framework with clean and robust teachers and iteratively updates the robust teacher during student training. These methods adapt the supervisory model concurrently with student learning. CARD considers a diferent setting in which teacher refinement is performed after the student has completed an initial adversarialdistillation stage. Freezing the distilled student provides a stable feedback source and decouples weak-to-strong refinement from the evolving student-training dynamics, allowing post-distillation student feedback and teacher-student discrepancies to be explicitly utilized for teacher refinement. The resulting teacher is then used to provide updated robustness supervision to the same student, so that teacher refinement ultimately serves the further improvement of the compact student.

A related line of work, weak-to-strong learning, investigates whether supervision from a weaker model can improve a stronger model [2, 22, 11]. Although this direction is closely related to the teacher-refinement stage of CARD, existing weak-to-strong studies typically focus on improving the stronger model itself. CARD instead incorporates weakto-strong feedback into adversarial robustness distillation, where improving the teacher is not the final objective but an intermediate step toward further enhancing the compact student’s robustness.

## 3. Methodology

CARD extends conventional one-way adversarial robustness distillation into a closed teacher-student feedback loop. Given a training set $D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } , \mathbf { a }$ pretrained robust teacher $T ^ { ( 0 ) }$ , and an initialized compact student $S ^ { ( 0 ) }$ CARD proceeds through three sequential stages. The first strong-to-weak stage distills robust knowledge from $T ^ { ( 0 ) }$ to obtain a first-round student $S ^ { ( 1 ) }$ . The student is then frozen and used as a stable feedback source to refine the original teacher, producing $T ^ { ( 1 ) }$ . Finally, the refined teacher provides updated robustness supervision to the same student, yielding the final student $S ^ { ( 2 ) }$ . Figure 2 shows the complete workflow.

Algorithm 1 CARD   
Input: Pretrained robust teacher $T ^ { ( 0 ) }$ , initialized student   
$S ^ { ( 0 ) }$ , training set , base adversarial-distillation objective   
$\mathcal { L } _ { \mathrm { b a s e } } .$ , and perturbation set $B _ { \epsilon }$   
Parameters: Strong-to-weak distillation epoch $E _ { 1 }$ , strong  
to-weak gain transfer epoch $E _ { 2 } ,$ teacher-refinement epoch   
$E _ { T }$ , inverse and adversarial optimization steps $K _ { \mathrm { i n v } }$ and   
$K _ { \mathrm { a d v } } .$ , refinement weight $\lambda ,$ and teacher learning rate $\eta _ { T }$   
Output: Refined teacher $T ^ { ( 1 ) }$ and final student $\bar { S } ^ { ( 2 ) }$   
1: Stage 1: Strong-to-weak adversarial distillation   
2: Freeze $T ^ { ( 0 ) }$ and make $S ^ { ( 0 ) }$ trainable   
3: $S ^ { ( 1 ) } \gets \mathrm { A d v D i s t i l l } ( S ^ { ( 0 ) } , T ^ { ( 0 ) } , \mathcal { D } , E _ { 1 } ; \mathcal { L } _ { \mathrm { b a s e } } )$   
4: Stage 2: Weak-to-strong teacher refinement   
5: Freeze $S ^ { ( 1 ) }$ and make $T ^ { ( 0 ) }$ trainable   
6: for $e = 1$ to $E _ { T }$ do   
7: for each mini-batch $( x , y )$ sampled from  do   
8: Generate student inverse adversarial examples $x _ { \mathrm { i n v } }$   
using $K _ { \mathrm { i n v } }$ steps according to Eq. (4)   
9: Generate discrepancy-maximizing teacher adver  
sarial examples $x _ { \mathrm { a d v } }$ using $K _ { \mathrm { a d v } }$ steps according to   
Eq. (5)   
10: $\bar { \mathcal { L } _ { \mathrm { r o b } } }  D _ { \mathrm { K L } } \big ( S ^ { ( 1 ) } ( x ) \| T ^ { ( 1 ) } ( x _ { \mathrm { a d v } } ) \big )$   
11: ${ \mathcal { L } } _ { \mathrm { n a t } }  D _ { \mathrm { K L } } \big ( S ^ { ( 1 ) } ( x _ { \mathrm { i n v } } ) \| T ^ { ( 1 ) } ( x ) \big )$   
12: ${ \mathcal { L } } _ { \mathrm { r e f i n e } } \gets { \mathcal { L } } _ { \mathrm { r o b } } + \lambda { \mathcal { L } } _ { \mathrm { n a t } }$   
13: $T ^ { ( 1 ) } \gets T ^ { ( 1 ) } - \eta _ { T } \nabla _ { T ^ { ( 1 ) } } \mathcal { L } _ { \mathrm { r e f i n e } }$   
14: end for   
15: end for   
16: Stage 3: Strong-to-weak gain transfer   
17: Freeze $T ^ { ( 1 ) }$ and make $S ^ { ( 1 ) }$ trainable   
18: $S ^ { ( 2 ) } \gets \mathrm { A d v D i s t i l l } ( S ^ { ( 1 ) } , T ^ { ( 1 ) } , \mathcal { D } , E _ { 2 } ; \mathcal { L } _ { \mathrm { b a s e } } )$   
19: return $T ^ { ( 1 ) }$ and $S ^ { ( 2 ) }$

Throughout this section, $\mathcal { L } _ { \mathrm { b a s e } }$ denotes the studenttraining objective of the selected adversarial-distillation method. CARD retains the corresponding adversarial-example generation procedure and student-distillation objective in both strong-to-weak stages.

## 3.1. Strong-to-Weak Adversarial Distillation

CARD begins with conventional adversarial robustness distillation. The pretrained teacher $T ^ { ( 0 ) }$ is frozen, and the student is optimized using the objective and adversarialexample generation procedure of the selected base method:

$$
\begin{array} { r } { S ^ { ( 1 ) } \gets \mathrm { A d v D i s t i l l } \left( S ^ { ( 0 ) } , T ^ { ( 0 ) } , \mathcal { D } , E _ { 1 } ; \mathcal { L } _ { \mathrm { b a s e } } \right) , } \end{array}\tag{1}
$$

where AdvDistill $\left( S ^ { ( 0 ) } , T ^ { ( 0 ) } , D , E _ { 1 } ; \mathcal { L } _ { \mathrm { b a s e } } \right)$ denotes student training under the selected base adversarial-distillation method for $E _ { 1 }$ epochs. In our primary instantiation,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b a s e } } = \mathcal { L } _ { \mathrm { A d a A D } } . } \end{array}\tag{2}
$$

Other adversarial-distillation objectives can be substituted without changing the subsequent CARD workflow. Importantly, the purpose of this first stage is not only to obtain a compact robust student, but also to produce a taskadapted student that has already absorbed robust knowledge from the teacher and can subsequently serve as a meaningful feedback source.

## 3.2. Weak-to-Strong Teacher Refinement

After the initial distillation stage, we freeze $S ^ { ( 1 ) }$ and initialize the trainable teacher from $T ^ { ( 0 ) }$ . Although the student remains lower-capacity than the teacher, it has already acquired task-specific robust knowledge through the first strong-to-weak stage. Freezing the student provides a stable post-distillation reference and separates teacher refinement from the evolving dynamics of student training.

Weak-to-strong refinement must address two related issues. First, directly transferring arbitrary student predictions may propagate student errors to the teacher. Second, teacher refinement should focus on inputs where student feedback is more informative. CARD addresses these issues through inverse adversarial student feedback and teacher-student discrepancy maximization, respectively.

Let $\mathsf { \bar { S } } ^ { ( 1 ) } ( x ) , T ^ { ( 1 ) } ( x ) \in \bar { \Delta ^ { C - 1 } }$ denote the predictive distributions of the frozen student and the teacher being refined over � classes, with $\Delta ^ { C - 1 }$ denoting the $( C - 1 )$ -dimensional probability simplex. For an input $x \in [ 0 , 1 ] ^ { d }$ , the admissible $\ell _ { \infty }$ perturbation set is defined as

$$
\mathcal { B } _ { \epsilon } ( x ) = \left\{ \delta : \| \delta \| _ { \infty } \leq \epsilon , x + \delta \in [ 0 , 1 ] ^ { d } \right\} ,\tag{3}
$$

where $d$ is the input dimension and � is the perturbation budget.

## 3.2.1. Inverse adversarial studentfeedback

Directly using the student’s prediction on a clean input may propagate erroneous or poorly aligned supervision to the teacher. To obtain a more favorable student prediction with respect to the ground-truth label, we construct an inverse adversarial example by minimizing the student’s classification loss within the admissible perturbation set:

$$
\begin{array} { r l } & { \delta _ { \mathrm { i n v } } ^ { \star } \in \underset { \delta \in \mathcal { B } _ { \epsilon } ( x ) } { \arg \operatorname* { m i n } } \mathrm { C E } \left( S ^ { ( 1 ) } ( x + \delta ) , y \right) , } \\ & { x _ { \mathrm { i n v } } = x + \delta _ { \mathrm { i n v } } ^ { \star } . } \end{array}\tag{4}
$$

In contrast to standard adversarial-example generation, which typically seeks to increase the classification loss, Eq. (4) searches for a nearby input that reduces the student’s loss with respect to the ground-truth label. Because $\delta = 0$ is feasible, exact inner minimization guarantees

$$
\mathrm { C E } \left( S ^ { ( 1 ) } ( x _ { \mathrm { i n v } } ) , y \right) \leq \mathrm { C E } \left( S ^ { ( 1 ) } ( x ) , y \right) .
$$

The resulting prediction $S ^ { ( 1 ) } ( x _ { \mathrm { i n v } } )$ is then used to supervise the teacher on the corresponding clean input. This construction reduces the student’s ground-truth classification loss before feedback is transferred, but it does not guarantee that every resulting prediction is correct. The conditions under which such feedback preserves the ground-truth prediction after teacher alignment are analyzed in Sec. 3.4.

## 3.2.2. Teacher-student discrepancy maximization

Complementing the inverse adversarial feedback mechanism, CARD further identifies perturbed inputs where the teacher and the frozen student exhibit large predictive discrepancies. Specifically, we search for perturbations that maximize the discrepancy between the student’s clean prediction and the teacher’s prediction on the perturbed input:

$$
\begin{array} { r l } & { \delta _ { \mathrm { a d v } } ^ { \star } \in \underset { \delta \in \mathcal { B } _ { \epsilon } ( x ) } { \arg \operatorname* { m a x } } D _ { \mathrm { K L } } ( S ^ { ( 1 ) } ( x ) \| T ^ { ( 1 ) } ( x + \delta ) ) , } \\ & { x _ { \mathrm { a d v } } = x + \delta _ { \mathrm { a d v } } ^ { \star } . } \end{array}\tag{5}
$$

The resulting $x _ { \mathrm { a d v } }$ is a discrepancy-maximizing perturbed input relative to the frozen student’s clean prediction.

## 3.2.3. Teacher-refinement objective

The two branches provide complementary supervision for teacher refinement on perturbed and clean inputs. Their corresponding losses are defined as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r o b } } = D _ { \mathrm { K L } } ( S ^ { ( 1 ) } ( x )  T ^ { ( 1 ) } ( x _ { \mathrm { a d v } } ) ) , } \\ & { \mathcal { L } _ { \mathrm { n a t } } = D _ { \mathrm { K L } } ( S ^ { ( 1 ) } ( x _ { \mathrm { i n v } } )  T ^ { ( 1 ) } ( x ) ) . } \end{array}\tag{6}
$$

The robust branch aligns the teacher prediction at a discrepancy-maximizing perturbed input with the student’s clean prediction, whereas the natural branch aligns the teacher prediction on the clean input with the student’s inverse adversarial prediction. The complete refinement objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e f i n e } } = \mathcal { L } _ { \mathrm { r o b } } + \lambda \mathcal { L } _ { \mathrm { n a t } } , \qquad \lambda \geq 0 , } \end{array}\tag{7}
$$

where � controls the relative contribution of clean-input refinement.

Because $S ^ { ( 1 ) }$ remains frozen throughout this stage, its predictions serve as fixed supervision and only the teacher parameters are updated:

$$
\theta _ { T ^ { ( 1 ) } }  \theta _ { T ^ { ( 1 ) } } - \eta _ { T } \nabla _ { \theta _ { T ^ { ( 1 ) } } } \mathcal { L } _ { \mathrm { r e f i n e } } ,\tag{8}
$$

where $\eta _ { T }$ denotes the teacher learning rate.

Starting from $T ^ { ( 0 ) }$ , this optimization produces the refined teacher $T ^ { ( \bar { 1 } ) }$ . The natural and robust branches respectively refine the teacher on clean inputs and discrepancy-selected perturbed inputs, while the frozen student provides a stable source of feedback.

## 3.3. Strong-to-Weak Gain Transfer

Teacher refinement is not the final objective of CARD. Its purpose is to provide an updated robust teacher that can further supervise the compact student. After refinement, we therefore freeze $T ^ { ( 1 ) }$ and resume training from the firstround student $S ^ { ( 1 ) }$ , rather than initializing a new student. Continuing from $S ^ { ( 1 ) }$ preserves the knowledge acquired during the initial distillation stage and allows the refined teacher supervision to be incorporated into the existing student model.

CARD retains the student-training objective and adversarialexample generation procedure of the selected base adversarialdistillation method:

$$
S ^ { ( 2 ) } \gets \mathrm { A d v D i s t i l l } \left( S ^ { ( 1 ) } , T ^ { ( 1 ) } , { \cal D } , E _ { 2 } ; \mathcal { L } _ { \mathrm { b a s e } } \right) .\tag{9}
$$

Compared with the initial strong-to-weak stage, the original teacher $T ^ { ( 0 ) }$ is replaced by the refined teacher $T ^ { ( 1 ) }$ , and student optimization resumes from $S ^ { ( 1 ) }$ . The base student-distillation objective remains unchanged. The resulting $S ^ { ( 2 ) }$ constitutes the final student of CARD, completing the teacher–student feedback loop by transferring the refined teacher supervision back to the same student. The complete procedure is summarized in Algorithm 1.

## 3.4. Reliability Analysis of Weak-to-Strong Feedback

Having presented the complete CARD workflow, we now analyze the conditions under which weak-to-strong feedback preserves prediction correctness during teacher refinement. Let $q , p \in \Delta ^ { C - 1 }$ denote a frozen studentfeedback distribution and the corresponding teacher output after alignment, respectively. For the ground-truth class �, we define the student-feedback margin as

$$
m ( q , y ) = q _ { y } - \operatorname* { m a x } _ { c \neq y } q _ { c } .\tag{10}
$$

A positive margin $m ( q , y ) ~ > ~ 0$ indicates that the student feedback predicts the ground-truth class �.

Theorem 1 (Weak-to-strong prediction preservation). $I f m ( q , y ) > 0$ and

$$
D _ { \mathrm { K L } } ( q \| p ) < \frac { m ( q , y ) ^ { 2 } } { 2 } ,\tag{11}
$$

then

$$
p _ { y } > \operatorname* { m a x } _ { c \neq y } p _ { c } .\tag{12}
$$

PROOF. Define the total variation distance as

$$
D _ { \mathrm { T V } } ( p , q ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { C } | p _ { j } - q _ { j } | .
$$

For probability distributions � and $q ,$ each coordinate satisfies

$$
| p _ { j } - q _ { j } | \leq D _ { \mathrm { T V } } ( p , q ) .
$$

Part (i): Preservation under a positive margin. For any class $c \neq y ,$

$$
\begin{array} { r l } & { p _ { y } - p _ { c } = ( q _ { y } - q _ { c } ) + ( p _ { y } - q _ { y } ) - ( p _ { c } - q _ { c } ) } \\ & { \qquad \geq q _ { y } - q _ { c } - | p _ { y } - q _ { y } | - | p _ { c } - q _ { c } | } \\ & { \qquad \geq m ( q , y ) - 2 D _ { \mathrm { T V } } ( p , q ) . } \end{array}
$$

Therefore, if $D _ { \mathrm { T V } } ( p , q ) < m ( q , y ) / 2$ , then $p _ { y } - p _ { c } > 0$ for every $c \neq y .$

Part (ii): KL-based control of output deviation. Pinsker’s inequality gives

$$
D _ { \mathrm { T V } } ( p , q ) = D _ { \mathrm { T V } } ( q , p ) \leq \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( q \| p ) } .
$$

Using Eq. (11),

$$
D _ { \mathrm { T V } } ( p , q ) < \sqrt { \frac { 1 } { 2 } \cdot \frac { m ( q , y ) ^ { 2 } } { 2 } } = \frac { m ( q , y ) } { 2 } .
$$

Part (i) then gives $p _ { y } - p _ { c } > 0$ for every $c \neq y .$ Hence, $p _ { y } > \operatorname* { m a x } _ { c \neq y } p _ { c }$

Theorem 1 is independent of the relative capacities of the student and teacher. It does not imply that the student is globally superior to the teacher. Rather, it shows that on samples for which the student feedback predicts the ground-truth class with a positive margin, teacher alignment preserves that prediction whenever the remaining KL misalignment is suficiently small.

For the natural branch, the theorem applies with

$$
q = S ^ { ( 1 ) } ( x _ { \mathrm { i n v } } ) , \qquad p = T ^ { ( 1 ) } ( x ) ,
$$

where the inverse adversarial optimization reduces the student’s classification loss before its prediction is used to supervise the teacher on the clean input.

For the robust branch,

$$
q = S ^ { ( 1 ) } ( x ) , \qquad p = T ^ { ( 1 ) } ( x _ { \mathrm { a d v } } ) .
$$

Here, discrepancy maximization identifies the perturbed input, while outer minimization aligns the teacher prediction at that input with the clean student reference. We next derive a distribution-level risk bound for weak-to-strong teacher alignment.

Theorem 2 (Margin-dependent risk bound).

Let $Q , P \in \Delta ^ { C - 1 }$ and $Y ~ \in ~ \{ 1 , \ldots , C \}$ denote jointly distributed studentfeedback, teacher output after alignment, and ground-truth label, respectively. Define

$$
M _ { Q } = Q _ { Y } - \operatorname* { m a x } _ { c \neq Y } Q _ { c } ,\tag{13}
$$

and the student-feedback and teacher risks as

$$
R _ { Q } = { \mathrm { P r } } ( M _ { Q } \leq 0 ) , \qquad R _ { P } = { \mathrm { P r } } \left( P _ { Y } \leq \operatorname* { m a x } _ { c \neq Y } P _ { c } \right)\tag{14}
$$

Furthermore, define

$$
\psi ( Q , P , Y ) = \left\{ \begin{array} { l l } { \operatorname* { m i n } \left\{ 1 , \frac { 2 D _ { \mathrm { K L } } ( Q \| P ) } { M _ { Q } ^ { 2 } } \right\} , } & { M _ { Q } > 0 , } \\ { 0 , } & { M _ { Q } \leq 0 . } \end{array} \right.\tag{15}
$$

Then,

$$
\begin{array} { r l } & { R _ { P } \leq R _ { Q } + \mathrm { P r } \left( M _ { Q } > 0 , D _ { \mathrm { K L } } ( Q \| P ) \geq \frac { M _ { Q } ^ { 2 } } { 2 } \right) } \\ & { \qquad \leq R _ { Q } + \mathbb { E } \left[ \psi ( Q , P , Y ) \right] } \\ & { \qquad \leq 1 . } \end{array}\tag{16}
$$

PROOF. Let

$$
\mathcal { E } _ { P } = \left\{ P _ { Y } \leq \operatorname* { m a x } _ { c \neq Y } P _ { c } \right\}
$$

denote the teacher error event.

Part (i): Error decomposition. We partition the teacher error event according to whether the student feedback has a positive margin:

$$
\begin{array} { r l } & { R _ { P } = \operatorname* { P r } ( \mathcal { E } _ { P } ) } \\ & { \quad = \operatorname* { P r } \left( \mathcal { E } _ { P } \cap \{ M _ { Q } \leq 0 \} \right) + \operatorname* { P r } \left( \mathcal { E } _ { P } \cap \{ M _ { Q } > 0 \} \right) } \\ & { \quad \leq R _ { Q } + \operatorname* { P r } \left( \mathcal { E } _ { P } \cap \{ M _ { Q } > 0 \} \right) . } \end{array}
$$

Part (ii): Teacher errors require margin-dependent KL misalignment. For a sample satisfying $M _ { Q } > 0$ , if

$$
D _ { \mathrm { K L } } ( Q \Vert P ) < \frac { M _ { Q } ^ { 2 } } { 2 } ,
$$

Theorem 1 shows that, when the student feedback predicts the ground-truth class with a positive margin and the KL divergence after alignment is suficiently small, the teacher preserves the same ground-truth prediction. Therefore,

$$
\mathcal { E } _ { P } \cap \{ M _ { Q } > 0 \} \subseteq \left\{ M _ { Q } > 0 , D _ { \mathrm { K L } } ( Q \| P ) \geq \frac { M _ { Q } ^ { 2 } } { 2 } \right\} .
$$

It follows that

$$
R _ { P } \leq R _ { Q } + \mathrm { P r } \left( M _ { Q } > 0 , D _ { \mathrm { K L } } ( Q \| P ) \geq \frac { M _ { Q } ^ { 2 } } { 2 } \right) .
$$

Part (iii): Distribution-level control. Let

$$
\mathcal { A } = \left\{ M _ { Q } > 0 , D _ { \mathrm { K L } } ( Q \| P ) \geq \frac { M _ { Q } ^ { 2 } } { 2 } \right\} .
$$

By the definition of $\psi ( Q , P , Y )$ ,

$$
{ \mathcal { A } } = \{ \psi ( Q , P , Y ) = 1 \} .
$$

Because �(�, �, �) is non-negative,

$$
\begin{array} { r l } { \quad } & { \operatorname* { P r } ( \boldsymbol { \mathcal { A } } ) = \operatorname* { P r } \left( \psi ( \boldsymbol { \mathcal { Q } } , P , Y ) \ge 1 \right) } \\ & { \quad \le \operatorname { \mathbb { E } } \left[ \psi ( \boldsymbol { \mathcal { Q } } , P , Y ) \right] . } \end{array}
$$

Combining this result with Part (ii) gives

$$
R _ { P } \leq R _ { Q } + \mathbb { E } \left[ \psi ( Q , P , Y ) \right] .
$$

Moreover,

$$
\begin{array} { r } { 0 \leq \psi ( Q , P , Y ) \leq \mathbb { I } _ { \{ M _ { Q } > 0 \} } , } \end{array}
$$

and therefore

$$
\mathbb { E } \left[ \psi ( Q , P , Y ) \right] \leq \operatorname* { P r } ( M _ { Q } > 0 ) = 1 - R _ { Q } .
$$

Hence,

$$
R _ { P } \leq R _ { Q } + \mathbb { E } \left[ \psi ( Q , P , Y ) \right] \leq 1 .
$$

This proves Eq. (16).

For the natural branch, Theorem 2 is instantiated with

$$
Q = S ^ { ( 1 ) } ( X _ { \mathrm { i n v } } ) , \qquad P = T ^ { ( 1 ) } ( X ) ,
$$

whereas for the robust branch,

$$
Q = S ^ { ( 1 ) } ( X ) , \qquad P = T ^ { ( 1 ) } ( X _ { \mathrm { a d v } } ) .
$$

The bound shows that the teacher risk is controlled by two factors: errors already present in the student feedback and teacher-student misalignment relative to the feedback margin. For correctly classified student feedback, a larger positive margin or a smaller KL divergence reduces the margin-normalized misalignment term in the bound.

Theorem 2 does not, by itself, guarantee that teacher refinement always reduces classification risk. Rather, the bound certifies a strict improvement over the pre-refinement teacher whenever the resulting upper bound is lower than the pre-refinement teacher risk evaluated on the same input distribution. The analysis is therefore distribution-specific. For the robust branch, it characterizes the discrepancy-selected perturbed inputs considered during refinement, rather than worst-case robustness over the entire perturbation set.

## 4. Experiments

We evaluate CARD through white-box and transferbased black-box robustness experiments, followed by ablation studies, hyperparameter analysis, and qualitative visualization. The experiments are designed to examine the efectiveness and generalizability of CARD across diferent datasets, student architectures, and base adversarialdistillation methods.

## 4.1. Experimental Setup

This subsection describes the datasets, network architectures, compared methods, training configurations, and evaluation protocols used throughout the experiments. Unless otherwise stated, a baseline method and its corresponding CARD variant use the same dataset split, model architecture, and base adversarial-distillation configuration.

## 4.1.1. Datasets and preprocessing.

We evaluate CARD on CIFAR-10, CIFAR-100 [20], and a fixed ten-class subset of TinyImageNet [23], denoted as TinyImageNet-10. CIFAR-10 and CIFAR-100 each contain 50,000 training images and 10,000 test images at a spatial resolution of 32 × 32 pixels, with 10 and 100 classes, respectively. We use the original training and test partitions without introducing additional training data.

TinyImageNet-10 consists of ten fixed classes selected from TinyImageNet. For each class, all 500 training images and 50 validation images are retained, resulting in 5,000 training images and 500 evaluation images at a spatial resolution of 64 × 64 pixels. The original TinyImageNet training and validation splits are used for training and evaluation, respectively. The selected classes are Goldfish, King penguin, American alligator, Bullet train, School bus, Broom, Bucket, Hourglass, Snorkel, and Espresso. The same TinyImageNet-10 subset is used in all experiments.

For training, we apply random cropping with a padding of four pixels and random horizontal flipping, following commonly adopted preprocessing protocols in adversarial training [25] and adversarial distillation [24, 16]. No data augmentation is applied during evaluation.

## 4.1.2. Architectures and baselines.

We use WideResNet-28-10 (WRN-28-10) and WideResNet-34-10 (WRN-34-10) [38] as pretrained robust teachers. ResNet-18 [15] and MobileNet-V2 [31] are used as student architectures.

We compare CARD with representative adversarial training methods, including PGD-AT [28], TRADES [39], and ST [25], as well as adversarial robustness distillation methods, including ARD [13], RSLAD [43], IAD [42], AdaAD [16], PeerAiD [19], and AdaAD+IGDM [24]. The compared methods are reproduced following their published training protocols and hyperparameter settings whenever applicable.

For each CARD variant, the initial strong-to-weak stage follows the corresponding base adversarial-distillation method. CARD directly starts teacher refinement from the student checkpoint produced by this initial distillation stage, without introducing additional student training before teacher refinement.

In the main experiments, CARD is instantiated on AdaAD and AdaAD+IGDM. Once denotes one complete teacherrefinement and gain-transfer cycle. For variants marked Twice, the same refinement and gain-transfer procedure is applied for a second cycle, initialized from the teacher and student obtained after the preceding cycle.

## 4.1.3. Training and implementation details.

Teacher refinement is initialized from the pretrained robust teacher $T ^ { ( 0 ) }$ , while the student obtained from the initial strong-to-weak distillation stage, $S ^ { ( 1 ) }$ , is kept frozen throughout refinement. Thus, only the teacher parameters are updated during this stage.

The teacher is optimized using stochastic gradient descent (SGD) with a momentum of 0.9, a weight decay of

Clean and white-box robust accuracy (%) on CIFAR-10 with WRN-28-10 as the teacher and diferent student architectures. Within each student-architecture group, the best and second-best results in each column are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Architecture</td><td rowspan="2">Method</td><td colspan="5">CIFAR-10</td></tr><tr><td>Clean</td><td>FGSM</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>Teacher</td><td>WRN-28-10</td><td>ST [25]</td><td>84.57</td><td>62.17</td><td>57.67</td><td>54.99</td><td>53.77</td></tr><tr><td rowspan="17">Student</td><td rowspan="5"></td><td>PGD-AT [28]</td><td>82.50</td><td>57.52</td><td>51.53</td><td>50.57</td><td>48.04</td></tr><tr><td>TRADES [39]</td><td>82.17</td><td>58.20</td><td>52.21</td><td>50.48</td><td>49.18</td></tr><tr><td>ST [25]</td><td>83.43</td><td>59.91</td><td>53.72</td><td>51.66</td><td>50.01</td></tr><tr><td>ARD [13]</td><td>83.43</td><td>59.27</td><td>53.04</td><td>51.20</td><td>49.49</td></tr><tr><td>RSLAD [43]</td><td>82.44</td><td>60.99</td><td>55.66</td><td>53.19</td><td>51.90</td></tr><tr><td>IAD [42]</td><td>83.21</td><td>59.88</td><td>53.31</td><td>50.88</td><td>49.22</td></tr><tr><td rowspan="8">ResNet-18 AdaAD [16] PeerAiD [19]</td><td></td><td>84.05</td><td>60.51</td><td>55.82</td><td>53.01</td><td>51.90</td></tr><tr><td></td><td>82.75</td><td>59.68</td><td>53.78</td><td>51.64</td><td>49.56</td></tr><tr><td>AdaAD+IGDM [24]</td><td>82.81</td><td>61.04</td><td>57.01</td><td>53.83</td><td>52.75</td></tr><tr><td>Ours-AdaAD (Once)</td><td>84.15</td><td>61.24</td><td>55.89</td><td>53.59</td><td></td></tr><tr><td>Ours-AdaAD (Twice)</td><td>84.39</td><td>61.81</td><td>55.62</td><td>53.68</td><td>52.26 52.22</td></tr><tr><td>Ours-AdaAD+IGDM (Once)</td><td>82.92</td><td>61.95</td><td>57.03</td><td>54.37</td><td>52.92</td></tr><tr><td>PGD-AT [28]</td><td>79.99</td><td>54.94</td><td>49.23</td><td>47.70</td><td>45.11</td></tr><tr><td>TRADES [39]</td><td>80.98</td><td>55.52</td><td>49.97</td><td>47.76</td><td></td></tr><tr><td rowspan="11">MobileNet-V2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.51</td></tr><tr><td>ST [25]</td><td>80.05</td><td>56.75</td><td>51.97</td><td></td><td>48.14</td><td>47.12</td></tr><tr><td>ARD [13]</td><td>82.19</td><td>58.24</td><td>52.37</td><td></td><td>49.99</td><td>48.34</td></tr><tr><td>RSLAD [43]</td><td>82.65</td><td>59.74</td><td>54.57</td><td></td><td>51.95</td><td>50.49</td></tr><tr><td>IAD [42]</td><td>81.46</td><td>58.06</td><td>52.78</td><td></td><td>50.16</td><td>48.43</td></tr><tr><td>AdaAD [16]</td><td>83.34</td><td>59.21</td><td></td><td>54.03</td><td>50.96</td><td>49.57</td></tr><tr><td>PeerAiD [19]</td><td></td><td>79.98</td><td>57.48</td><td>53.89</td><td>50.85</td><td>49.04</td></tr><tr><td>AdaAD+IGDM [24]</td><td></td><td>81.39</td><td>58.99</td><td>54.60</td><td>51.45</td><td>50.20</td></tr><tr><td>Ours-AdaAD (Once)</td><td></td><td>83.37</td><td>59.59</td><td>53.89</td><td>51.85</td><td>50.36</td></tr><tr><td>Ours-AdaAD+IGD (Once)</td><td></td><td>79.71</td><td>59.04</td><td>54.70</td><td>52.00</td><td>50.69</td></tr></table>

$2 \times 1 0 ^ { - 4 }$ , and a batch size of 128. Teacher refinement is performed for 20 epochs. The learning rate is initialized to 0.02, reduced to 0.01 after the 10th epoch, and further reduced to 0.001 after the 18th epoch. The temperature used in the KL-divergence terms is fixed to $\tau = 1 . 0 $ . The selection of � is further analyzed in Section 4.5.

For both inverse adversarial feedback and teacher–student discrepancy maximization, perturbations are constrained within an $\ell _ { \infty }$ ball with radius $\epsilon ~ = ~ 8 / 2 5 5 .$ . Each inner optimization is performed for 10 steps with a step size of 2∕255. After each step, the perturbed input is projected back onto the corresponding $\ell _ { \infty }$ ball centered at the clean input and clipped to the valid input range.

After teacher refinement, the refined teacher $T ^ { ( 1 ) }$ is frozen, and student training resumes from the first-round checkpoint $S ^ { ( 1 ) }$ . The student is trained for an additional 30 epochs under the supervision of $T ^ { ( 1 ) }$ . The studentdistillation objective and adversarial-example generation procedure follow those of the corresponding base adversarialdistillation method, with method-specific hyperparameters retained unless otherwise stated.

## 4.1.4. Evaluation protocol.

For white-box evaluation, we consider FGSM [14], PGD-20 [28], a 20-step attack based on the CW margin loss [4], and AutoAttack [5]. All attacks are constrained by an $\ell _ { \infty }$ perturbation budget of $\epsilon \ : = \ : 8 / 2 5 5$ . For PGD-20 and the CW-20 attack, the step size is 2∕255. Each white-box attack is generated directly against the evaluated target model, and the same attack configuration is used for a baseline and its corresponding CARD variant. AutoAttack is evaluated in its standard configuration, comprising APGD-CE, APGD-DLR, FAB, and Square Attack, under the same $\ell _ { \infty }$ perturbation budget.

For transfer-based black-box evaluation, adversarial examples are first generated against independently trained surrogate models and then transferred directly to the target student without access to the target model parameters or gradients during attack generation. We consider PGD-100 and CW-100, both with $\epsilon = 8 / 2 5 5$ and a step size of 2∕255.

The robust surrogate models include ResNet-18, VGG-19 [33], and MobileNet-V2, whereas the naturally trained surrogate models include ResNet-18, MobileNet-V2, VGG-19, ResNet-50, DenseNet-169 [17], and WRN-34-10. For each surrogate model and attack configuration, the same transferred adversarial examples are used to evaluate all target methods, ensuring that the baseline methods and their corresponding CARD variants are compared under identical transfer attacks.

Table 2  
Clean and white-box robust accuracy (%) on CIFAR-10 with WRN-34-10 as the teacher and diferent student architectures. Within each student-architecture group, the best and second-best results in each column are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Architecture</td><td rowspan="2">Method</td><td colspan="5">CIFAR-10</td></tr><tr><td>Clean</td><td>FGSM</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>Teacher</td><td>WRN-34-10</td><td>ST [25]</td><td>84.75</td><td>62.40</td><td>57.83</td><td>55.49</td><td>54.33</td></tr><tr><td rowspan="17">Student</td><td rowspan="5"></td><td>PGD-AT [28]</td><td>82.50</td><td>57.52</td><td>51.53</td><td>50.57</td><td>48.04</td></tr><tr><td>TRADES [39]</td><td>82.17</td><td>58.20</td><td>52.21</td><td>50.48</td><td>49.18</td></tr><tr><td>ST [25]</td><td>83.43</td><td>59.91</td><td>53.72</td><td>51.66</td><td>50.01</td></tr><tr><td>ARD [13]</td><td>83.29</td><td>59.32</td><td>53.65</td><td>51.63</td><td>50.01</td></tr><tr><td>RSLAD [43]</td><td>82.98</td><td>60.71</td><td>55.61</td><td>53.00</td><td>51.49</td></tr><tr><td>IAD [42]</td><td>82.84</td><td>59.69</td><td>54.13</td><td>51.72</td><td>50.10</td></tr><tr><td rowspan="8">ResNet-18</td><td>AdaAD [16]</td><td>84.28</td><td>60.97</td><td>56.17</td><td>53.36</td><td>52.02</td></tr><tr><td>PeerAiD [19]</td><td>82.63</td><td>59.87</td><td>54.21</td><td>52.24</td><td>50.25</td></tr><tr><td>AdaAD+IGDM [24]</td><td>83.78</td><td>61.54</td><td>56.64</td><td>54.09</td><td>52.65</td></tr><tr><td>Ours-AdaAD (Once)</td><td>84.45</td><td>61.07</td><td>55.68</td><td>53.63</td><td></td></tr><tr><td>Ours-AdaAD (Twice)</td><td>84.49</td><td>61.49</td><td>55.39</td><td>53.95</td><td>52.22 52.34</td></tr><tr><td>Ours-AdaAD+IGDM (Once)</td><td>84.06</td><td>61.89</td><td>56.73</td><td>54.50</td><td>53.10</td></tr><tr><td>PGD-AT [28]</td><td>79.99</td><td>54.94</td><td>49.23</td><td>47.70</td><td>45.11</td></tr><tr><td></td><td>80.98</td><td>55.52</td><td>49.97</td><td>47.76</td><td>46.51</td></tr><tr><td rowspan="11">MobileNet-V2</td><td>TRADES [39]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ST [25]</td><td>80.05</td><td>56.75</td><td>51.97</td><td>48.14</td><td>47.12</td></tr><tr><td>ARD [13]</td><td>81.80</td><td>57.77</td><td>52.50</td><td>50.22</td><td>48.67</td></tr><tr><td>RSLAD [43]</td><td>83.11</td><td>59.66</td><td>54.24</td><td>51.68</td><td>50.28</td></tr><tr><td>IAD [42]</td><td>81.44</td><td>58.47</td><td>52.47</td><td>50.15</td><td>48.06</td></tr><tr><td>AdaAD [16]</td><td>83.86</td><td>59.32</td><td>54.06</td><td>51.41</td><td>49.95</td></tr><tr><td>PeerAiD [19]</td><td>82.12</td><td>59.25</td><td>53.70</td><td>50.76</td><td>48.76</td></tr><tr><td>AdaAD+IGDM [24]</td><td>81.48</td><td>58.83</td><td>54.72</td><td>51.52</td><td>50.34</td></tr><tr><td>Ours-AdaAD (Once)</td><td>83.90</td><td>59.55</td><td>53.71</td><td>51.57</td><td>50.02</td></tr><tr><td>Ours-AdaAD+IGD (Once)</td><td>81.73</td><td>59.58</td><td>54.54</td><td>51.84</td><td>50.37</td></tr></table>

## 4.2. White-Box Robustness

We first evaluate CARD under white-box attacks across diferent datasets, teacher architectures, student architectures, and base adversarial-distillation objectives.

## 4.2.1. Results on CIFAR-10.

Table 1 reports the CIFAR-10 results with WRN-28- 10 as the teacher. For ResNet-18, applying CARD once to AdaAD+IGDM improves clean, FGSM, PGD-20, CW-20, and AA accuracy over the original AdaAD+IGDM baseline by 0.11, 0.91, 0.02, 0.54, and 0.17 percentage points, respectively. Similarly, applying CARD once to AdaAD improves all five metrics over the original AdaAD baseline. A second CARD cycle further increases clean, FGSM, and CW-20 accuracy relative to the one-cycle AdaAD-based variant, while slightly reducing PGD-20 and AA accuracy.

For MobileNet-V2, the efect of CARD is more mixed. Compared with the original AdaAD+IGDM baseline, applying CARD once increases FGSM, PGD-20, CW-20, and AA accuracy by 0.05, 0.10, 0.55, and 0.49 percentage points, respectively, while reducing clean accuracy by 1.68 percentage points. Compared with the original AdaAD baseline, applying CARD once increases clean, FGSM, CW-20, and AA accuracy by 0.03, 0.38, 0.89, and 0.79 percentage points, respectively, whereas PGD-20 accuracy decreases by 0.14 percentage points. These results indicate that the efect of CARD on CIFAR-10 varies with both the student architecture and the base adversarial-distillation objective.

Table 2 further reports the CIFAR-10 results with WRN-34-10 as the teacher. For ResNet-18, compared with the original AdaAD baseline, applying CARD once increases clean, FGSM, CW-20, and AA accuracy by 0.17, 0.10, 0.27, and 0.20 percentage points, respectively, while PGD-20 accuracy decreases by 0.49 percentage points. Applying a second CARD cycle further changes the results relative to the one-cycle variant, increasing clean, FGSM, CW-20, and AA accuracy by 0.04, 0.42, 0.32, and 0.12 percentage points, respectively, while PGD-20 accuracy decreases by an additional 0.29 percentage points. Compared with the original AdaAD+IGDM baseline, applying CARD once improves clean, FGSM, PGD-20, CW-20, and AA accuracy by 0.28, 0.35, 0.09, 0.41, and 0.45 percentage points, respectively. This AdaAD+IGDM-based CARD variant achieves the highest FGSM, PGD-20, CW-20, and AA accuracy among the evaluated ResNet-18 student methods.

Table 3  
Clean and white-box robust accuracy (%) on CIFAR-100 with WRN-34-10 as the teacher and diferent student architectures. Within each student-architecture group, the best and second-best results in each column are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Architecture</td><td rowspan="2">Method</td><td colspan="5">CIFAR-100</td></tr><tr><td>Clean</td><td>FGSM</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>Teacher</td><td>WRN-34-10</td><td>ST [25]</td><td>59.40</td><td>35.92</td><td>33.71</td><td>30.30</td><td>29.19</td></tr><tr><td rowspan="17">Student</td><td rowspan="5"></td><td>PGD-AT [28]</td><td>57.52</td><td>32.12</td><td>28.50</td><td>27.12</td><td>24.68</td></tr><tr><td>TRADES [39]</td><td>56.32</td><td>29.83</td><td>26.95</td><td>23.77</td><td>22.69</td></tr><tr><td>ST [25]</td><td>55.75</td><td>33.00</td><td>30.92</td><td>26.89</td><td>26.04</td></tr><tr><td>ARD [13]</td><td>58.44</td><td>34.92</td><td>31.54</td><td>28.47</td><td>26.42</td></tr><tr><td>RSLAD [43]</td><td>56.55</td><td>35.08</td><td>32.85</td><td>29.34</td><td>28.00</td></tr><tr><td>IAD [42]</td><td>57.82</td><td>34.96</td><td>31.94</td><td>28.32</td><td>26.80</td></tr><tr><td rowspan="8">ResNet-18</td><td>AdaAD [16]</td><td>58.91</td><td>34.99</td><td>32.57</td><td>29.00</td><td>27.80</td></tr><tr><td>PeerAiD [19]</td><td>56.76</td><td>35.68</td><td>32.99</td><td>29.66</td><td>27.97</td></tr><tr><td>AdaAD+IGDM [24]</td><td>58.49</td><td>35.30</td><td>33.14</td><td>29.48</td><td>28.17</td></tr><tr><td>Ours-AdaAD (Once)</td><td>59.66</td><td>35.75</td><td>33.03</td><td>29.75</td><td></td></tr><tr><td>Ours-AdaAD (Twice)</td><td>59.68</td><td>35.77</td><td>32.95</td><td>29.62</td><td>28.19 28.19</td></tr><tr><td>Ours-AdaAD+IGDM (Once)</td><td>58.62</td><td>36.00</td><td>33.63</td><td>30.15</td><td>29.00</td></tr><tr><td>PGD-AT [28]</td><td>56.00</td><td>30.99</td><td>27.73</td><td>25.84</td><td>23.25</td></tr><tr><td></td><td></td><td>30.79</td><td>28.22</td><td>24.16</td><td></td></tr><tr><td rowspan="11">MobileNet-V2</td><td>TRADES [39]</td><td>56.82</td><td></td><td></td><td></td><td></td><td>23.18</td></tr><tr><td>ST [25]</td><td>54.53</td><td>31.91</td><td>30.10</td><td></td><td>25.41</td><td>24.64</td></tr><tr><td>ARD [13]</td><td>57.25</td><td>33.82</td><td>31.00</td><td></td><td>27.90</td><td>26.02</td></tr><tr><td>RSLAD [43]</td><td>56.78</td><td>34.50</td><td>32.11</td><td></td><td>28.43</td><td>26.94</td></tr><tr><td>IAD [42]</td><td>57.41</td><td>34.67</td><td>31.59</td><td></td><td>28.09</td><td>26.36</td></tr><tr><td>AdaAD [16]</td><td>57.95</td><td>33.41</td><td></td><td>30.92</td><td>27.00</td><td>25.55</td></tr><tr><td>PeerAiD [19]</td><td></td><td>56.67</td><td>34.59</td><td>31.47</td><td>28.82</td><td>26.92</td></tr><tr><td>AdaAD+IGDM [24]</td><td></td><td>57.12</td><td>34.34</td><td>32.33</td><td>28.47</td><td>27.44</td></tr><tr><td>Ours-AdaAD (Once)</td><td></td><td>58.24</td><td>33.98</td><td>31.30</td><td>28.11</td><td>26.40</td></tr><tr><td>Ours-AdaAD+IGD (Once)</td><td></td><td>57.14</td><td>34.82</td><td>32.59</td><td>28.88</td><td>27.74</td></tr></table>

For MobileNet-V2, the efect of CARD is again metric dependent. Compared with the original AdaAD baseline, applying CARD once increases clean, FGSM, CW-20, and AA accuracy by 0.04, 0.23, 0.16, and 0.07 percentage points, respectively, while PGD-20 accuracy decreases by 0.35 percentage points. Compared with the original AdaAD+IGDM baseline, applying CARD once increases clean, FGSM, CW-20, and AA accuracy by 0.25, 0.75, 0.32, and 0.03 percentage points, respectively, while PGD-20 accuracy decreases by 0.18 percentage points. Among the evaluated MobileNet-V2 student methods, the AdaAD+IGDM-based CARD variant achieves the highest CW-20 and AA accuracy.

## 4.2.2. Results on CIFAR-100.

Table 3 reports the results on CIFAR-100 with WRN-34-10 as the teacher. For ResNet-18, compared with the original AdaAD+IGDM baseline, applying CARD once improves clean, FGSM, PGD-20, CW-20, and AA accuracy by 0.13, 0.70, 0.49, 0.67, and 0.83 percentage points, respectively. For MobileNet-V2, compared with the corresponding AdaAD+IGDM baseline, the improvements are 0.02, 0.48, 0.26, 0.41, and 0.30 percentage points, respectively. CARD also consistently improves AdaAD after one cycle for both student architectures. For ResNet-18, compared with the original AdaAD baseline, applying CARD once increases clean, FGSM, PGD-20, CW-20, and AA accuracy by 0.75, 0.76, 0.46, 0.75, and 0.39 percentage points, respectively. For MobileNet-V2, the corresponding improvements over the original AdaAD baseline are 0.29, 0.57, 0.38, 1.11, and 0.85 percentage points. These results show that one CARD cycle consistently improves both clean and robust accuracy over the corresponding AdaAD and AdaAD+IGDM baselines across the two evaluated student architectures.

For ResNet-18 with AdaAD, applying a second CARD cycle yields only marginal changes relative to the one-cycle variant. Specifically, clean and FGSM accuracy increase by 0.02 percentage points each, PGD-20 and CW-20 accuracy decrease by 0.08 and 0.13 percentage points, respectively, and AA accuracy remains unchanged. This result indicates that additional refinement-transfer cycles do not necessarily provide uniform improvements across all evaluation metrics.

## 4.2.3. Results on TinyImageNet-10.

Table 4 reports the results on TinyImageNet-10 with WRN-34-10 as the teacher and ResNet-18 as the student.

Table 4  
Clean and white-box robust accuracy (%) on TinyImageNet-10 with WRN-34-10 as the teacher and ResNet-18 as the student. The best and second-best student results in each column are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Architecture</td><td rowspan="2">Method</td><td colspan="5">TinylmageNet-10</td></tr><tr><td>Clean</td><td>FGSM</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>Teacher</td><td>WRN-34-10</td><td>ST [25]</td><td>77.80</td><td>62.00</td><td>60.80</td><td>60.00</td><td>60.00</td></tr><tr><td rowspan="10">Student</td><td></td><td>PGD-AT [28]</td><td>79.60</td><td>61.00</td><td>57.20</td><td>57.20</td><td>55.20</td></tr><tr><td></td><td>TRADES [39]</td><td>77.80</td><td>58.80</td><td>56.80</td><td>55.60</td><td>55.40</td></tr><tr><td></td><td>ST [25]</td><td>76.60</td><td>62.20</td><td>61.00</td><td>59.00</td><td>58.20</td></tr><tr><td>ARD [13]</td><td></td><td>79.60</td><td>60.40</td><td>58.60</td><td>57.00</td><td>56.40</td></tr><tr><td></td><td>RSLAD [43]</td><td>76.80</td><td>62.60</td><td>61.80</td><td>59.20</td><td>59.20</td></tr><tr><td>IAD [42] ResNet-18</td><td></td><td>79.20</td><td>61.40</td><td>59.40</td><td>57.60</td><td>57.20</td></tr><tr><td>AdaAD [16]</td><td></td><td>78.40</td><td>62.80</td><td>62.40</td><td>59.60</td><td>59.40</td></tr><tr><td>PeerAiD [19]</td><td></td><td>82.60</td><td>62.40</td><td>58.80</td><td>58.20</td><td>57.20</td></tr><tr><td>AdaAD+IGDM [24]</td><td></td><td>77.40</td><td>62.80</td><td>62.20</td><td>60.20</td><td>59.80</td></tr><tr><td></td><td>Ours-AdaAD (Once)</td><td>79.20</td><td>61.60</td><td>61.00</td><td>60.00</td><td>59.40</td></tr><tr><td></td><td>Ours-AdaAD+IGDM (Once)</td><td>78.80</td><td>63.80</td><td>62.60</td><td>61.00</td><td>60.80</td></tr><tr><td></td><td>Ours-ARD (Once)</td><td>82.80</td><td>60.80</td><td>57.60</td><td>57.00</td><td>55.40</td></tr></table>

Table 5

Transfer-based black-box accuracy (%) on CIFAR-100 with WRN-34-10 as the teacher and ResNet-18 as the target student. PGD-100 and CW-100 adversarial examples are generated on independently trained robust ResNet-18, VGG-19, and MobileNet-V2 surrogate models. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">ResNet-18</td><td colspan="2">VGG-19</td><td colspan="2">MobileNet-V2</td></tr><tr><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td></tr><tr><td>PGD-AT [28]</td><td>33.69</td><td>33.41</td><td>41.64</td><td>42.03</td><td>38.36</td><td>38.27</td></tr><tr><td>TRADES [39]</td><td>35.71</td><td>36.49</td><td>40.75</td><td>41.68</td><td>37.55</td><td>37.91</td></tr><tr><td>ST [25]</td><td>37.67</td><td>38.17</td><td>42.00</td><td>42.68</td><td>39.13</td><td>39.65</td></tr><tr><td>ARD [13]</td><td>39.32</td><td>39.31</td><td>44.19</td><td>44.24</td><td>40.73</td><td>40.68</td></tr><tr><td>RSLAD [43]</td><td>39.18</td><td>39.72</td><td>43.23</td><td>43.86</td><td>40.58</td><td>40.81</td></tr><tr><td>IAD [42]</td><td>39.57</td><td>39.80</td><td>44.39</td><td>44.77</td><td>41.36</td><td>41.61</td></tr><tr><td>AdaAD [16]</td><td>39.64</td><td>39.76</td><td>44.76</td><td>44.85</td><td>41.50</td><td>41.11</td></tr><tr><td>PeerAiD [19]</td><td>39.53</td><td>39.97</td><td>43.70</td><td>43.39</td><td>41.42</td><td>41.00</td></tr><tr><td>AdaAD+IGDM [24]</td><td>39.76</td><td>40.05</td><td>44.86</td><td>45.31</td><td>41.51</td><td>41.40</td></tr><tr><td>Ours-AdaAD (Once)</td><td>40.14</td><td>40.23</td><td>45.32</td><td>45.53</td><td>41.94</td><td>41.78</td></tr><tr><td>Ours-AdaAD (Twice)</td><td>40.16</td><td>40.44</td><td>45.50</td><td>45.87</td><td>42.16</td><td>42.02</td></tr><tr><td>Ours-AdaAD+IGDM (Once)</td><td>40.20</td><td>40.28</td><td>45.14</td><td>45.48</td><td>41.98</td><td>41.82</td></tr></table>

Compared with the original AdaAD+IGDM baseline, applying CARD once improves clean, FGSM, PGD-20, CW-20, and AA accuracy by 1.40, 1.00, 0.40, 0.80, and 1.00 percentage points, respectively. The resulting AdaAD+IGDMbased CARD variant achieves the highest robust accuracy under all four attacks among the evaluated student methods, while also improving clean accuracy over its corresponding baseline.

The efects of CARD are less uniform when combined with the other distillation objectives. Compared with the original AdaAD baseline, applying CARD once increases clean and CW-20 accuracy by 0.80 and 0.40 percentage points, respectively, and leaves AA accuracy unchanged, while FGSM and PGD-20 accuracy decrease by 1.20 and 1.40 percentage points. Compared with the original ARD baseline, applying CARD once increases clean and FGSM accuracy by 3.20 and 0.40 percentage points, respectively, leaves CW-20 accuracy unchanged, and decreases PGD-20 and AA accuracy by 1.00 percentage point each. The ARDbased CARD variant achieves the highest clean accuracy among the evaluated student methods.

Overall, the efect of CARD on TinyImageNet-10 varies with the base adversarial-distillation objective. The AdaAD +IGDM-based variant improves both clean accuracy and robustness under all four evaluated attacks, whereas the AdaAD- and ARD-based variants exhibit metric-dependent trade-ofs between clean and robust performance.

## 4.3. Black-Box Transfer Robustness

We further evaluate the transfer-based black-box robustness of CARD on CIFAR-100 using independently trained robust and naturally trained surrogate models.

Table 7  
Table 6  
Transfer-based black-box accuracy (%) on CIFAR-100 with WRN-34-10 as the teacher and ResNet-18 as the target student. PGD-100 and CW-100 adversarial examples are generated on independently trained natural ResNet-18, MobileNet-V2, and VGG-19 surrogate models. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">ResNet-18</td><td colspan="2">MobileNet-V2</td><td colspan="2">VGG-19</td></tr><tr><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td></tr><tr><td>PGD-AT [28]</td><td>56.29</td><td>56.31</td><td>56.30</td><td>56.32</td><td>56.49</td><td>56.53</td></tr><tr><td>TRADES [39]</td><td>54.83</td><td>54.96</td><td>54.80</td><td>54.97</td><td>55.12</td><td>55.28</td></tr><tr><td>ST [25]</td><td>54.43</td><td>54.54</td><td>54.35</td><td>54.31</td><td>54.50</td><td>54.68</td></tr><tr><td>ARD [13]</td><td>56.99</td><td>57.20</td><td>57.12</td><td>57.04</td><td>57.38</td><td>57.15</td></tr><tr><td>RSLAD [43]</td><td>55.16</td><td>55.24</td><td>55.14</td><td>55.16</td><td>55.46</td><td>55.50</td></tr><tr><td>IAD [42]</td><td>56.81</td><td>56.98</td><td>56.82</td><td>56.91</td><td>57.04</td><td>57.09</td></tr><tr><td>AdaAD [16]</td><td>57.36</td><td>57.50</td><td>57.36</td><td>57.47</td><td>57.53</td><td>57.53</td></tr><tr><td>PeerAiD [19]</td><td>55.57</td><td>55.54</td><td>55.71</td><td>55.73</td><td>55.72</td><td>55.64</td></tr><tr><td>AdaAD+IGDM [24]</td><td>57.12</td><td>57.07</td><td>56.98</td><td>56.98</td><td>57.11</td><td>57.20</td></tr><tr><td>Ours-AdaAD (Once)</td><td>57.98</td><td>58.11</td><td>57.96</td><td>58.10</td><td>58.12</td><td>58.23</td></tr><tr><td>Ours-AdaAD (Twice)</td><td>57.99</td><td>58.18</td><td>58.01</td><td>58.15</td><td>58.37</td><td>58.44</td></tr></table>

Transfer-based black-box accuracy (%) on CIFAR-100 with WRN-34-10 as the teacher and ResNet-18 as the target student. PGD-100 and CW-100 adversarial examples are generated on independently trained natural ResNet-50, DenseNet-169, and WRN-34-10 surrogate models. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">ResNet-50</td><td colspan="2">DenseNet-169</td><td colspan="2">WRN-34-10</td></tr><tr><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td><td>PGD-100</td><td>CW-100</td></tr><tr><td>PGD-AT [28]</td><td>55.99</td><td>55.92</td><td>56.26</td><td>56.26</td><td>57.03</td><td>57.07</td></tr><tr><td>TRADES [39]</td><td>54.75</td><td>54.50</td><td>54.68</td><td>54.89</td><td>55.54</td><td>55.64</td></tr><tr><td>ST [25]</td><td>54.13</td><td>54.32</td><td>54.48</td><td>54.42</td><td>54.94</td><td>54.99</td></tr><tr><td>ARD [13]</td><td>56.85</td><td>56.93</td><td>57.18</td><td>56.96</td><td>57.53</td><td>57.60</td></tr><tr><td>RSLAD [43]</td><td>54.74</td><td>54.97</td><td>55.04</td><td>55.25</td><td>55.71</td><td>55.75</td></tr><tr><td>IAD [42]</td><td>56.70</td><td>56.81</td><td>56.75</td><td>56.93</td><td>57.32</td><td>57.30</td></tr><tr><td>AdaAD [16]</td><td>57.26</td><td>57.21</td><td>57.27</td><td>57.47</td><td>58.17</td><td>58.03</td></tr><tr><td>PeerAiD [19]</td><td>55.31</td><td>55.36</td><td>55.60</td><td>55.57</td><td>56.16</td><td>56.24</td></tr><tr><td>AdaAD+IGDM [24]</td><td>56.68</td><td>56.84</td><td>57.00</td><td>57.15</td><td>57.53</td><td>57.54</td></tr><tr><td>Ours-AdaAD (Once)</td><td>57.96</td><td>58.03</td><td>58.04</td><td>58.07</td><td>58.81</td><td>58.75</td></tr><tr><td>Ours-AdaAD (Twice)</td><td>58.05</td><td>57.96</td><td>58.30</td><td>58.04</td><td>58.82</td><td>58.82</td></tr></table>

## 4.3.1. Results with Robust Surrogates.

Table 5 reports the transfer-based black-box results with ResNet-18 as the target student and ResNet-18, VGG-19, and MobileNet-V2 as robust surrogate models. Compared with the original AdaAD baseline, applying CARD twice improves transferred-attack accuracy by 0.52–1.02 percentage points across all six surrogate-attack combinations.

Compared with the original AdaAD+IGDM baseline, applying CARD once also improves transferred-attack accuracy across all six surrogate-attack combinations, with gains ranging from 0.17 to 0.47 percentage points. These results indicate that CARD consistently improves transferbased black-box robustness against attacks generated from multiple robust surrogate architectures.

## 4.3.2. Results with Natural Surrogates.

Tables 6 and 7 report the transfer-based black-box results with ResNet-18 as the target student and ResNet-18, MobileNet-V2, VGG-19, ResNet-50, DenseNet-169, and

WRN-34-10 as naturally trained surrogate models. PGD-100 and CW-100 adversarial examples generated on each surrogate are directly transferred to the target student.

Compared with the original AdaAD baseline, applying CARD once improves transferred-attack accuracy by 0.59–0.82 percentage points across all 12 surrogate-attack combinations. Applying CARD twice also improves all 12 combinations over the original AdaAD baseline, with gains ranging from 0.57 to 1.03 percentage points.

Overall, these results demonstrate consistent improvements in transfer-based black-box robustness across attacks generated from multiple robust and naturally trained surrogate architectures.

## 4.4. Ablation Studies

We conduct ablation studies to examine three aspects of CARD: (1) whether weak-to-strong teacher refinement contributes beyond additional student training, (2) whether teacher refinement and student gain transfer should be optimized jointly or sequentially, and (3) how diferent supervision signals afect teacher refinement. Unless otherwise specified, the student is initialized from the checkpoint obtained after the initial strong-to-weak adversarial distillation stage, whereas the teacher is initialized from the pretrained robust teacher. All teacher-refinement variants use an $\ell _ { \infty }$ perturbation budget of $\epsilon = 8 / 2 5 5$ , with 10 inner optimization steps and a step size of 2∕255.

Table 9  
Table 8  
Component ablation of CARD on CIFAR-100 with WRN-34- 10 as the teacher and MobileNet-V2 as the student. W2S and S2W denote weak-to-strong teacher refinement and strong-toweak gain transfer, respectively.
<table><tr><td>Variant</td><td>W2S</td><td>S2W</td><td>Clean</td><td>CW-20</td><td>AA</td></tr><tr><td>AdaAD</td><td></td><td></td><td>57.95</td><td>27.00</td><td>25.55</td></tr><tr><td>S2W Only</td><td></td><td>√</td><td>57.09</td><td>27.55</td><td>26.15</td></tr><tr><td>CARD</td><td>√</td><td>√</td><td>58.24</td><td>28.11</td><td>26.40</td></tr></table>

## 4.4.1. Contribution of the Cyclic Components

We first examine whether the improvement of CARD can be attributed simply to additional student training. The baseline corresponds to the original AdaAD student. The S2W-only variant removes weak-to-strong (W2S) teacher refinement and directly continues strong-to-weak (S2W) student training under the guidance of the unchanged teacher using the original AdaAD objective. The complete CARD variant performs W2S teacher refinement before the subsequent S2W gain-transfer stage.

As shown in Table 8, compared with the original AdaAD baseline, S2W-only training improves CW-20 and AA accuracy by 0.55 and 0.60 percentage points, respectively, but reduces clean accuracy by 0.86 percentage points. Compared with S2W-only training, adding W2S teacher refinement further improves clean, CW-20, and AA accuracy by 1.15, 0.56, and 0.25 percentage points, respectively. Consequently, the complete CARD framework outperforms the original AdaAD baseline by 0.29 percentage points in clean accuracy and by 1.11 and 0.85 percentage points under CW-20 and AA, respectively. These results indicate that the improvement of CARD cannot be explained solely by additional student training and that W2S teacher refinement provides an additional contribution.

## 4.4.2. Separated versus Joint Optimization

We next examine whether W2S teacher refinement and S2W student gain transfer should be optimized jointly or sequentially. CARD adopts separated optimization, in which teacher refinement is completed before the refined teacher guides the subsequent student gain-transfer stage.

For comparison, we construct a Joint variant that updates the teacher and student simultaneously within a single training stage. The teacher is optimized using the CARD refinement objective, whereas the student is optimized using the original AdaAD objective:

Comparison of joint and separated optimization on CIFAR-100 with WRN-34-10 as the teacher and MobileNet-V2 as the student. Both variants contain the same W2S teacherrefinement and S2W gain-transfer components.
<table><tr><td>Method</td><td>Clean</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>AdaAD</td><td>57.95</td><td>30.92</td><td>27.00</td><td>25.55</td></tr><tr><td>Joint</td><td>56.93</td><td>31.23</td><td>28.00</td><td>26.30</td></tr><tr><td>Separated (Ours)</td><td>58.24</td><td>31.30</td><td>28.11</td><td>26.40</td></tr></table>

$$
\operatorname* { m i n } _ { \theta _ { T } } \mathcal { L } _ { \mathrm { { r e f i n e } } } , \qquad \operatorname* { m i n } _ { \theta _ { S } } \mathcal { L } _ { \mathrm { A d a A D } } ,\tag{17}
$$

where $\theta _ { T }$ and $\theta _ { S }$ denote the teacher and student parameters, respectively. During joint optimization, the current student provides feedback for teacher refinement while the current teacher simultaneously guides student optimization.

Both models are trained for 30 epochs. The teacher learning rate is set to 0.02, 0.01, 0.001, and 0.0001 during epochs 0–7, 8–15, 16–24, and 25–29, respectively, while the student learning rate is set to one-tenth of the corresponding teacher learning rate.

As shown in Table 9, compared with joint optimization, separated optimization improves clean accuracy by 1.31 percentage points and improves PGD-20, CW-20, and AA accuracy by 0.07, 0.11, and 0.10 percentage points, respectively. Although the robustness improvements are modest, they are consistent across all three attacks, while the improvement in clean accuracy is more pronounced. These results support the separated optimization strategy adopted in CARD.

## 4.4.3. Teacher-Refinement Signals

We further investigate how diferent supervision signals afect teacher refinement. During refinement, the student is frozen and only the teacher is updated. After each refinement variant, the resulting teacher is frozen and the same student undergoes the subsequent S2W gain-transfer stage using the AdaAD objective. Therefore, the variants difer only in the supervision used for teacher refinement. The w/o Refinement variant directly performs additional student training using the unchanged teacher. GT Only removes model-prediction supervision and relies only on groundtruth labels. Teacher Clean + GT uses the teacher’s clean prediction for robust refinement and the ground-truth label for clean-input refinement, thereby removing student feedback. Student Clean uses the clean student prediction for both refinement branches and therefore removes inverse adversarial feedback. Student Inverse instead uses the inverseadversarial student prediction for both branches. Teacher Inverse replaces student feedback with the corresponding inverse-adversarial teacher prediction. The complete CARD objective combines the clean student prediction for robust refinement with the inverse-adversarial student prediction for natural refinement.

Table 10  
Comparison of teacher-refinement supervision signals on CIFAR-100 with WRN-34-10 as the teacher and ResNet-18 as the student. The reported results correspond to the student obtained after gain transfer from the resulting teacher. GT denotes ground-truth hard-label supervision.
<table><tr><td>Refinement Signal</td><td>Clean</td><td>PGD-20</td><td>CW-20</td><td>AA</td></tr><tr><td>AdaAD</td><td>58.91</td><td>32.57</td><td>29.00</td><td>27.80</td></tr><tr><td>w/o Refinement</td><td>58.48</td><td>32.79</td><td>29.32</td><td>27.92</td></tr><tr><td>GT Only</td><td>61.85</td><td>26.30</td><td>26.09</td><td>23.64</td></tr><tr><td>Teacher Clean + GT</td><td>61.75</td><td>29.67</td><td>28.14</td><td>26.49</td></tr><tr><td>Student Clean</td><td>58.76</td><td>32.81</td><td>29.22</td><td>27.96</td></tr><tr><td>Student Inverse</td><td>60.98</td><td>32.51</td><td>29.29</td><td>27.73</td></tr><tr><td>Teacher Inverse</td><td>61.49</td><td>29.86</td><td>28.48</td><td>26.74</td></tr><tr><td>Ours-AdaAD (Once)</td><td>59.66</td><td>33.03</td><td>29.75</td><td>28.19</td></tr></table>

For the prediction-based variants, the perturbed teacher input is constructed by maximizing the discrepancy between the corresponding reference prediction and the teacher prediction on the perturbed input. For GT Only, the perturbed input is instead generated by maximizing the teacher’s crossentropy loss. The inverse variants construct the corresponding inverse adversarial input by minimizing the groundtruth cross-entropy loss of the student or teacher within the same perturbation set. These definitions isolate the efects of student feedback, inverse adversarial feedback, and the choice of supervision signal while keeping the subsequent student gain-transfer stage unchanged.

As shown in Table 10, none of the alternative refinementsignal variants improves all reported metrics over the original AdaAD baseline. Although GTOnly achieves the highest clean accuracy of 61.85%, its PGD-20 and AA accuracy decrease by 6.27 and 4.16 percentage points, respectively, relative to AdaAD. In contrast, the complete CARD objective achieves the highest robust accuracy under PGD-20, CW-20, and AA. Compared with the original AdaAD baseline, CARD improves clean, PGD-20, CW-20, and AA accuracy by 0.75, 0.46, 0.75, and 0.39 percentage points, respectively. Compared with the w/o Refinement variant, the corresponding improvements are 1.18, 0.24, 0.43, and 0.27 percentage points. These results support the use of clean student feedback for robust refinement together with inverseadversarial student feedback for natural refinement.

## 4.5. Sensitivity to the Refinement Weight �

The refinement weight � controls the contribution of the natural-refinement term relative to the robust-refinement term and is therefore the main method-specific hyperparameter in CARD. We analyze its sensitivity on CIFAR-10 using WRN-28-10 as the teacher and ResNet-18 as the student. All other training and refinement settings are kept unchanged. We vary � separately in the first- and secondcycle refinement stages and evaluate both the refined teacher and the student obtained after the subsequent gain-transfer stage.

As shown in Fig. 3(a), increasing � generally increases the natural accuracy of the first-cycle refined teacher while reducing its robust accuracy. A similar tendency is observed for the second-cycle teacher in Fig. 3(c). This behavior is consistent with the role of �: a larger value places greater emphasis on natural refinement, whereas a smaller value gives relatively greater importance to the robust-refinement term.

In contrast, the student performance in Figs. 3(b) and 3(d) varies less across the evaluated values of �. Although the refined teacher exhibits a clearer natural–robust trade-of, the student obtained after the subsequent gain-transfer stage is comparatively less sensitive to the precise refinement weight. This observation suggests that the efect of � on the intermediate teacher does not translate directly into an equally large variation in the final student performance.

Based on validation performance, we set � = 0.2 for the first-cycle refinement and � = 0.7 for the second-cycle refinement. Within the evaluated range, the final student performance remains relatively stable, indicating limited sensitivity to � in the considered setting.

## 4.6. Qualitative and Mechanistic Analysis

We provide complementary qualitative analyses to examine the attention patterns learned by CARD and the prediction behavior underlying weak-to-strong teacher refinement. Specifically, Grad-CAM visualizations are used to compare discriminative attention under adversarial perturbations, while confusion matrices are used to examine the class consistency of inverse-adversarial feedback and the prediction changes induced by teacher refinement and subsequent gain transfer.

## 4.6.1. Grad-CAM Visualizations

We use Grad-CAM [32] to visualize the discriminative regions emphasized by diferent models on adversarial CIFAR-100 inputs. As shown in Fig. 4, several baseline students exhibit relatively fragmented activation patterns or place substantial attention on background regions. In contrast, the CARD student shows more spatially concentrated activation on object-relevant regions in the visualized examples and exhibits greater qualitative similarity to the robust teacher.

Compared with the baselines, the CARD student attends more consistently to object-relevant regions and shows activation patterns closer to those of the robust teacher under adversarial perturbations. This qualitative similarity suggests that the student better retains the discriminative patterns of the robust teacher, consistent with the improved robustness observed in the quantitative results.

## 4.6.2. Confusion Matrix Analysis

To examine the feasibility of weak-to-strong teacher refinement, we analyze whether inverse adversarial student predictions provide suficiently reliable feedback to the teacher.

Figure 5 presents normalized confusion matrices on inverse adversarial, clean, and adversarial CIFAR-10 inputs, using WRN-28-10 as the teacher and ResNet-18 as the student. Both perturbed input types are generated using 10- step iterative optimization.

![](images/17faf31232df70daa3de6e13b668b690c31b0ce1a84b0d5169ebc23c28211025.jpg)  
(a) First-Cycle Teacher

![](images/35ceebc305841fe0f4b6d3609c511818ad8c76b05d76a609c140aa7bbc5ea1e7.jpg)  
(b) First-Cycle Student

![](images/890248e68ac0a5f3176a0e46b3e723965cd0fd9c138847932d22922c2e519bc5.jpg)  
(c) Second-Cycle Teacher

![](images/f36b46884a0f8108a513cd510e84ad2b542ccddef3f6bfc196076b6564fc2f65.jpg)  
(d) Second-Cycle Student

Figure 3: Sensitivity of CARD to the refinement weight � on CIFAR-10 with WRN-28-10 as the teacher and ResNet-18 as the student. Blue and red curves denote natural and robust accuracy, respectively. Panels (a) and (c) report the refined teacher after the first and second refinement cycles, while panels (b) and (d) report the corresponding students after the subsequent gain-transfer stage.  
![](images/8ca3f5a2274315739b4634a5fea927b45e15b76ec0b1cab1d0c3ecc89b4d9fca.jpg)  
Figure 4: Grad-CAM visualizations on adversarial CIFAR-100 inputs. The leftmost column shows the corresponding adversaria images, the middle columns show activation maps of diferent student methods, and the rightmost column shows those of the robust teacher. In the visualized examples, the CARD student exhibits more concentrated activation on object-relevant regions and greater qualitative similarity to the robust teacher than the baseline students.

All models achieve over 96% accuracy on inverse adversarial inputs, with predictions strongly concentrated along the main diagonal. This indicates that inverse adversarial inputs preserve reliable class-consistent information for teacher refinement. Refinement increases the teacher’s clean and adversarial accuracy from 84.57% and 58.32% to 84.63% and 59.10%, respectively. The final CARD student subsequently improves AdaAD from 84.05% to 84.15% on clean inputs and from 56.57% to 56.74% on adversarial inputs.

The refined teacher and final student show slight decreases on inverse adversarial inputs while remaining above 96%. Because these perturbations are optimized to facilitate correct classification, the reduced benefit from them, together with improved adversarial accuracy, suggests less sensitivity to input perturbations. Overall, consistent with Theorems 1 and 2, the results provide empirical support that reliable student feedback can be transferred to the teacher when it has a positive prediction margin and is suficiently aligned with the teacher output.

![](images/80cd988d3f73a41f8142f3a45213c6730581ff0ce592187f48b282fa9b21e8da.jpg)

![](images/3c595a18a7772f90f5522b9a09ebb6bf6b7dd42dc0544d4be7816435f97eb932.jpg)

![](images/f38b325f9e771d83f837102d3be6229d57ef2b20f9aad5110efd6a68c5225990.jpg)

![](images/2fa9146b49c8859a6fe1d6e7f954392093b21507c379fd8a03c29dea69b9aee4.jpg)

![](images/b847bc924c76afc854f7af176089755634dbb4218f320a45ecc293df505e628e.jpg)

![](images/219bb23ca261c2e95f49a0bcf70f85f5597a24ec232a07f30096182801675805.jpg)

![](images/0ba90eaed57eda1b490e1d784578d4a5be504e63c394e14a0f7e4c3f55616a28.jpg)

![](images/8d63ff7ef3b5cc98d58526287c10c0794b9e6355a0f43df7682828468c2f6ef0.jpg)

![](images/4a5de43773f2b2e34d5d596b0ad70eae19486af96be941fe9a8997bd43680a30.jpg)

![](images/968c75b1935439ceed11fee32a8ee9ab79ab3e128ad120538414be32e07c8e4d.jpg)  
Inverse Adversarial Samples

![](images/fa0bac2b0e4e92f5d33a095e2ac0288ee0423fd4d11909c8f6c930598270b086.jpg)  
Clean Samples

![](images/2760c817eb5b073cf0d9d77f94652904c73b30518820cc076f2a06fd2080beb9.jpg)  
Adversarial Samples  
Figure 5: Normalized confusion matrices on inverse adversarial, clean, and adversarial CIFAR-10 inputs with WRN-28-10 as the teacher and ResNet-18 as the student. The inverse adversarial and adversarial inputs are generated using 10-step iterative optimization. Columns correspond to the three input types, whereas rows show the initial teacher, AdaAD student, refined teacher, and final CARD student. The strong diagonal concentration on inverse adversarial inputs indicates that these inputs provide highly reliable class-consistent predictions. Teacher refinement improves clean and adversarial accuracy, and part of these gains is subsequently transferred to the final student.

## 5. Conclusion

In this work, we proposed Cyclic Adversarial Robust Distillation (CARD), which extends conventional one-way adversarial robustness distillation into a closed teacher– student feedback loop consisting of strong-to-weak distillation, weak-to-strong teacher refinement, and subsequent gain transfer. CARD uses inverse-adversarial student feedback and discrepancy-maximizing inputs to refine the teacher while retaining the base student-distillation objective. We further provided theoretical conditions for reliable weakto-strong feedback and showed experimentally that CARD improves adversarial robustness in most evaluated settings under both white-box and transfer-based black-box attacks.

CARD still has several limitations. Its current evaluation is restricted to relatively small-scale image classification benchmarks and primarily considers $\ell _ { \infty }$ -bounded attacks, while the additional refinement and gain-transfer stages introduce extra training cost. Moreover, the improvement is not uniform across all attacks and base distillation methods. Future work will investigate more eficient refinement strategies, larger-scale models and datasets, broader threat models, and adaptive feedback mechanisms that can better determine when and how student knowledge should be used to refine the teacher.

## CRediT authorship contribution statement

Zhi Li: Writing-Original Draft, Methodology, Visualization, Data Curation. Haowei Liu: Investigation, Data Curation. Hongchen Yang: Investigation, Validation. Xiaoxuan Wang: Conceptualization, Methodology. Song Gao: Conceptualization, Methodology, Writing-Original Draft. Zhenli He: Conceptualization, Methodology, Funding Acquisition. Wei Zhou: Funding Acquisition. Shaowen Yao: Funding Acquisition.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the author(s) used GPT-5.6 Sol for polishing the expression. The author(s) reviewed and edited the output as needed and take full responsibility for the content of the published article.

## Acknowledgments

This work is supported by the Yunnan Research Project (Grant Nos. 202505AF350053, 202503AG380006, 202401- AT070474, 202501AU070059, 202403AP140021), National Natural Science Foundation of China (Grant Nos. 62562061, 62502422 and 62462067), and Yunnan Provincial Department of Education Science Research Project (Grant Nos. 2025J0006, 2024J0010 and 2025J0007).

## Data availability

Data will be made available on request.

## References

[1] Bai, T., Zhao, J., Wen, B., 2024. Guided adversarial contrastive distillation for robust students. IEEE Transactions on Information Forensics and Security 19, 9643–9655.

[2] Burns, C., Izmailov, P., Kirchner, J.H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecofet, A., Joglekar, M., Leike, J., Sutskever, I., Wu, J., 2024. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision, in: International Conference on Machine Learning, PMLR. pp. 4971–5012.

[3] Cai, Q.Z., Liu, C., Song, D., 2018. Curriculum adversarial training, in: Proceedings of the 27th International Joint Conference on Artificial Intelligence, AAAI Press. pp. 3740–3747.

[4] Carlini, N., Wagner, D., 2017. Towards evaluating the robustness of neural networks, in: IEEE Symposium on Security and Privacy, pp. 39–57.

[5] Croce, F., Hein, M., 2020. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks, in: International Conference on Machine Learning, PMLR. pp. 2206– 2216.

[6] Cui, J., Gao, S., Lv, T., Ji, J., Yao, S., Zhou, W., 2026. Dual-label guided unrestricted target attack with difusion model. Neurocomputing 665, 132185.

[7] Cui, J., Tian, Z., Zhong, Z., Qi, X., Yu, B., Zhang, H., 2024. Decoupled kullback-leibler divergence loss, in: Advances in Neural Information Processing Systems, pp. 74461–74486.

[8] Dong, J., Koniusz, P., Chen, J., Ong, Y.S., 2024a. Adversarially robust distillation by reducing the student-teacher variance gap, in: European Conference on Computer Vision, Springer. pp. 92–111.

[9] Dong, J., Koniusz, P., Chen, J., Wang, Z.J., Ong, Y.S., 2024b. Robust distillation via untargeted and targeted intermediate adversarial samples, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28432–28442.

[10] Dong, J., Moosavi-Dezfooli, S.M., Lai, J., Xie, X., 2023. The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24678– 24687.

[11] Dong, J., Zhang, C., Qu, X., Ma, Z., Koniusz, P., Ong, Y.S., 2025. Robust superalignment: Weak-to-strong robustness generalization for vision-language models, in: Advances in Neural Information Processing Systems, pp. 18345–18377.

[12] Feng, Y., Fan, J., Sun, Y., 2025. Vulnerable data-aware adversarial training, in: Advances in Neural Information Processing Systems, pp. 328–356.

[13] Goldblum, M., Fowl, L., Feizi, S., Goldstein, T., 2020. Adversarially robust distillation, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3996–4003.

[14] Goodfellow, I., Shlens, J., Szegedy, C., 2015. Explaining and harnessing adversarial examples, in: International Conference on Learning Representations.

[15] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778.

[16] Huang, B., Chen, M., Wang, Y., Lu, J., Cheng, M., Wang, W., 2023. Boosting accuracy and robustness of student models via adaptive adversarial distillation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24668–24677.

[17] Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q., 2017. Densely connected convolutional networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4700–4708.

[18] Jia, X., Zhang, Y., Wu, B., Ma, K., Wang, J., Cao, X., 2022. Las-at: adversarial training with learnable attack strategy, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13398–13408.

[19] Jung, J., Jang, H., Song, J., Lee, J., 2024. Peeraid: Improving adversarial distillation from a specialized peer tutor, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24482–24491.

[20] Krizhevsky, A., 2009. Learning Multiple Layers of Features from Tiny Images. Technical Report. University of Toronto.

[21] Kuang, H., Liu, H., Wu, Y., Satoh, S., Ji, R., 2023. Improving adversarial robustness via information bottleneck distillation, in: Advances in Neural Information Processing Systems, pp. 10796–10813.

[22] Lang, H., Sontag, D., Vijayaraghavan, A., 2024. Theoretical analysis of weak-to-strong generalization, in: Advances in Neural Information Processing Systems, pp. 46837–46880.

[23] Le, Y., Yang, X., et al., 2015. Tiny imagenet visual recognition challenge. CS 231N 7, 3.

[24] Lee, H., Cho, S., Kim, C., 2025. Indirect gradient matching for adversarial robust distillation, in: International Conference on Learning Representations.

[25] Li, Q., Guo, Y., Zuo, W., Chen, H., 2023. Squeeze training for adversarial robustness, in: International Conference on Learning Representations.

[26] Li, Q., Hu, Y., Dong, Y., Zhang, D., Chen, Y., 2024. Focus on hiders: Exploring hidden threats for enhancing adversarial training, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24442–24451.

[27] Lu, L., Pang, S., Zheng, X., Gu, X., Du, A., Liu, Y., Zhou, Y., 2025. CIARD: Cyclic iterative adversarial robustness distillation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 350–359.

[28] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A., 2018. Towards deep learning models resistant to adversarial attacks, in: International Conference on Learning Representations.

[29] Pang, T., Lin, M., Yang, X., Zhu, J., Yan, S., 2022. Robustness and accuracy could be reconcilable by (Proper) definition, in: International Conference on Machine Learning, PMLR. pp. 17258–17277.

[30] Rade, R., Moosavi-Dezfooli, S.M., 2022. Reducing excessive margin to achieve a better accuracy vs. robustness trade-of, in: International Conference on Learning Representations.

[31] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C., 2018. Mobilenetv2: Inverted residuals and linear bottlenecks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4510–4520.

[32] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 618–626.

[33] Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks for large-scale image recognition, in: International Conference on Learning Representations.

[34] Suzuki, S., Yamaguchi, S., Takeda, S., Kanai, S., Makishima, N., Ando, A., Masumura, R., 2023. Adversarial finetuning with latent representation constraint to mitigate accuracy-robustness tradeof, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4367–4378.

[35] Wang, Y., Ma, X., Bailey, J., Yi, J., Zhou, B., Gu, Q., 2019. On the convergence and robustness of adversarial training, in: International Conference on Machine Learning, PMLR. pp. 6586–6595.

[36] Waseda, F.K., Chang, C.C., Echizen, I., 2025. Rethinking invariance regularization in adversarial training to improve robustness-accuracy trade-of, in: International Conference on Learning Representations.

[37] Yin, S., Xiao, Z., Song, M., Long, J., 2024. Adversarial distillation based on slack matching and attribution region alignment, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24605–24614.

[38] Zagoruyko, S., Komodakis, N., 2016. Wide residual networks, in: Proceedings of the British Machine Vision Conference, pp. 87.1– 87.12.

[39] Zhang, H., Yu, Y., Jiao, J., Xing, E., Ghaoui, L.E., Jordan, M., 2019. Theoretically principled trade-of between robustness and accuracy, in: International Conference on Machine Learning, PMLR. pp. 7472– 7482.

[40] Zhang, J., Xu, X., Han, B., Niu, G., Cui, L., Sugiyama, M., Kankanhalli, M., 2020. Attacks which do not kill training make adversarial learning stronger, in: International Conference on Machine Learning, PMLR. pp. 11278–11287.

[41] Zhao, S., Yu, J., Sun, Z., Zhang, B., Wei, X., 2022. Enhanced accuracy and robustness via multi-teacher adversarial distillation, in: European Conference on Computer Vision, Springer. pp. 585–602.

[42] Zhu, J., Yao, J., Han, B., Zhang, J., Liu, T., Niu, G., Zhou, J., Xu, J., Yang, H., 2022. Reliable adversarial distillation with unreliable teachers, in: International Conference on Learning Representations.

[43] Zi, B., Zhao, S., Ma, X., Jiang, Y.G., 2021. Revisiting adversarial robustness distillation: Robust soft labels make student better, in:

Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16443–16452.