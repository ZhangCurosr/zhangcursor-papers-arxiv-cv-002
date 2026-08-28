# Rethinking Image Processing for the Age of AI: A Problem-First Framework for Scientific Progress

Guoping Qiu

Abstract—Modern AI has greatly expanded the capabilities of image processing. However, the ready availability of powerful models, public datasets, and benchmark leaderboards has also encouraged a model-first research pattern: researchers increasingly begin with an available architecture and optimize it on a public benchmark, rather than beginning with the underlying real-world imaging problem. This can produce impressive benchmark results without necessarily improving our understanding or solution of the real problem. This paper argues for a problem-first approach that distinguishes the physical imaging problem, solution principle, statistical estimator, and computational implementation, while clarifying what modern AI can achieve and which fundamental problems remain unsolved. Through case studies of superresolution and low-light enhancement, we show how benchmark datasets may define tasks that differ substantially from the real-world problems they are intended to represent, and why performance improvements must be interpreted within the conditions under which they are obtained. We propose a six-stage workflow that places problem formulation, image acquisition, information-loss analysis, assumptions, ambiguity, and evaluation before model and dataset selection. The paper also proposes clearer standards for evidence, reproducibility, uncertainty, and claims of state-of-the-art performance. More fundamentally, it calls for a change in research culture and education so that future researchers learn to understand imaging problems deeply and use modern AI to achieve genuine scientific and technical advancement.

Index Terms—image processing, inverse problems, modern AI, scientific methodology, benchmarks, reproducibility, distribution mismatch

## I. INTRODUCTION

Classical image processing designed operators: convolutional filters, transform-domain rules, interpolation kernels, Wiener estimators, and regularized inverses. Learning by example moved part of this design into data, including early neural restoration, learned inter-resolution prediction, exemplars, dictionaries, and learned priors [1]–[5]. Deep networks now learn representations and mappings jointly and dominate restoration benchmarks [6]–[8].

This history is often presented as though modern learning methods have made the problems and principles of classical image processing less relevant. We argue for a more discriminating view. Four levels should not be conflated:

problem ̸= principle ̸= estimator ̸= implementation.

(1)

A convolutional neural network (CNN) or transformer is an implementation family. Conditional-mean estimation is a statistical estimator. Bayesian inference or empirical risk minimization is a solution principle. These concepts describe how a solution is formulated, estimated, or implemented; they do not define the imaging problem itself. The imaging problem must instead be defined by the desired output, the physical acquisition process, the information lost or corrupted, and the intended application.

The practical concern is no longer only whether a network improves PSNR, perceptual quality, or a challenge ranking. A model can be optimal for the distribution defined by its paired-data construction and still be systematically wrong for deployment. Conversely, no finite “real-world” dataset can represent every camera, illumination, processing pipeline, scene, user preference, and future domain. The field therefore needs both a clearer understanding of what benchmark results establish and a systematic approach for developing reliable methods when training data cannot fully represent real-world applications.

This need is especially important for researchers entering the field. Open-source implementations, pretrained models, and public benchmark datasets have greatly accelerated experimentation and widened participation. At the same time, publication and evaluation systems naturally favour results that are rapid, measurable, and directly comparable. It is therefore rational for researchers to begin with an available or fashionable architecture and a public benchmark dataset, optimize the model for that dataset, and only afterwards ask what physical problem was solved. In that sequence, a small numerical gain may reflect architecture, parameter count, training schedule, augmentation, random variation, test-set familiarity, or tuning effort rather than a new understanding of image formation or recovery. Calling every such gain “state of the art” can unintentionally turn a conditional leaderboard result into a general scientific claim. Moreover, when code, data construction, hyperparameters, selection criteria, computational budget, and repeated-run variation are incompletely reported, the ordering itself may not be independently reproducible. These observations do not diminish the value of careful benchmark research. They explain why current research practices have developed and identify the evidence needed so that later research can build reliably on the findings.

A deeper concern is what this research pattern teaches. When the normal entry point into the field is an existing codebase, a fashionable model family, and a benchmark table, students may learn to regard the architecture as the intellectual centre of the problem. They can become skilled at changing blocks, losses, schedules, and hyperparameters without developing an equally strong understanding of optics, sensing, sampling, noise, dynamic range, colour formation, identifiability, prior information, or decision theory. The result is not a lack of technical ability; it is an imbalance between computational proficiency and scientific understanding. Such an imbalance makes it harder to recognize when a benchmark represents the wrong conditional problem, when an apparent improvement comes from a stronger prior rather than better recovery, or when an entirely different formulation is needed.

This concern is not confined to one institution or research tradition. In informal discussions with peers at major international conferences, universities, national laboratories, and technology companies, a recurring concern is that students, researchers, and new recruits increasingly begin by tuning models and often have a weaker grasp of the physical and inferential foundations of the task. These discussions are not a formal survey and should not be read as quantified evidence. They nevertheless point to a widely recognized educational challenge: the field must preserve the speed and power of modern AI while restoring disciplined study of the problems to which it is applied.

The central educational and methodological proposition of this paper is therefore simple: study the imaging problem before selecting the model. A researcher should understand how the observation was formed, what information survives, what has been lost, which assumptions or priors can legitimately restore it, what ambiguities remain, and what output the application actually requires. Only then should a benchmark, loss, architecture, and training protocol be chosen. This ordering leaves ample room for powerful modern AI, but makes model novelty serve a defined scientific question rather than substitute for one.

This paper makes eight contributions.

1) It develops a first-principles decomposition of imaging research into problem, solution principle, estimator, and implementation, and a six-source account of the knowledge that resolves missing information. This provides a common language for designed filters, Bayesian/regularized reconstruction, example-based learning, deep networks, transformers, and generative priors.

2) It clarifies what deep learning and modern AI can achieve through representation learning, increased capacity, wider contextual modelling, optimization, and large empirical priors, while identifying fundamental problems that remain unresolved, including limits imposed by acquisition, sampling, non-identifiability, decision objectives, and dependence on the training distribution.

3) It formalizes benchmark–deployment mismatch and audits representative super-resolution (SR) and lowlight image enhancement (LLIE) datasets. DIV2K-style digitally downsampled SR, RealSR/DRealSR capture rigs, LOL paired exposures, SID short/long raw exposures, and SICE selected HDR renderings are shown to define related but non-equivalent conditional tasks. It distinguishes insufficient spatial capture in SR, and limited dynamic range and low SNR in LLIE, from the convenient artificial transformations used to construct many benchmarks.

It turns the analysis into a six-stage problem-first workflow and applies every stage to SR and low-light enhancement, contrasting the resulting research design with representative CNN, transformer, GAN, blinddegradation, Retinex, burst-acquisition, inverse-rendering, null-space, and diffusion approaches. The workflow is also intended as a practical guide for students and researchers entering image processing.

It proposes a programme for progress under irreducible mismatch between training data and real-world applications: physically meaningful forward-model families, robust and worst-group evaluation, cross-dataset and cross-sensor testing, measurement-constrained adaptation, calibrated uncertainty, benchmark portfolios, deployment feedback, and explicitly scoped claims.

It articulates standards for reproducible comparison and a ladder separating numerical, comparative, mechanistic, generalization, and problem-progress claims, then translates the framework into guidance for research training, authors, reviewers, and benchmark organizers.

7) It identifies researcher education as part of the scientific problem. It proposes a training sequence in which image formation, information loss, assumptions, ambiguity, and evaluation precede model selection, so that future researchers learn not only how to optimize systems but how to formulate and advance imaging problems.

8) It proposes a community-level cultural change that broadens what counts as progress and distributes responsibility across researchers, supervisors, reviewers, editors, benchmark organizers, institutions, and industry. The aim is to make problem formulation, diagnostic evidence, robustness, and reproducibility as visible and rewardable as leaderboard performance.

The claim is therefore not that “nothing has changed,” nor that modern AI, benchmarks, or current researchers are misguided. Deep learning can turn previously limited principles into powerful technologies, and benchmarks are indispensable for controlled comparison. The model-first pattern arises partly because the surrounding research system makes it efficient and rewardable. The contribution is to place models and benchmarks in the correct inferential hierarchy and to suggest how the community can broaden its incentives and evidence. By distinguishing improvement in architecture from improvement in problem formulation, data, estimator, or evidence base, the framework aims to help researchers design experiments that generalize more reliably, help reviewers and editors recognize diverse forms of rigorous contribution, and direct technical innovation toward the causes of imaging failure rather than benchmark-specific symptoms.

## II. WHY IMAGE PROCESSING NEEDS RETHINKING

## A. Speed of experimentation versus depth of understanding

Modern software has reduced the cost of entering imageprocessing research. A student can download a dataset, modify a network block, launch training, and obtain a complete table of standard metrics in a short time. This is an important democratization of research. It allows ideas to be tested rapidly and permits communities to build on shared implementations rather than repeatedly reconstructing basic infrastructure.

The same convenience, however, can invert the scientific order. A method may begin with the question “where can this model be applied?” rather than “what prevents this image from containing the desired information?” The benchmark then supplies the problem definition by default, the loss supplies the meaning of quality, and the architecture supplies the presumed novelty. Extensive training can make the resulting system highly optimized for this closed loop even when the loop omits important causes of the deployed failure.

Real scientific progress often moves more slowly. It may require characterizing a sensor, deriving a forward model, discovering which variables are confounded, constructing controlled data, deciding what constitutes a legitimate reference, and designing an evaluation that can falsify the central claim. These activities can yield less immediate numerical reward than replacing one backbone by another, yet they frequently determine whether the eventual technology works outside the laboratory. Patience is not an alternative to computational progress; it is what gives computation a meaningful target.

## B. Why the model-first pattern is rational

The model-first pattern should not be attributed to a deficiency of individual researchers. It is a rational response to the environment in which research is conducted. Reusable implementations and pretrained weights lower the cost of model experiments. Standard datasets and metrics make results easy to compare. Leaderboards make rank visible. Short publication cycles favour projects whose outcomes can be produced and communicated quickly. Reviewers reasonably ask for comparison with established baselines, while editors and programme committees need evidence that can be assessed consistently across many submissions.

By contrast, problem-first work often has higher initial cost and less predictable output. It may require access to sensors, expertise across optics and statistics, new data collection, careful target construction, or experiments that reveal that a convenient formulation is inadequate. A better forward model, a negative result, or a diagnosis of non-identifiability may be scientifically important without immediately producing the highest score in a familiar table. Under current incentives, choosing the more measurable path is understandable.

The required cultural change must therefore make rigorous problem work more visible and more rewardable. It should broaden, not replace, the community’s concept of contribution. Architectural advances and benchmark improvements remain valuable, but should sit alongside advances in problem formulation, acquisition modelling, dataset construction, uncertainty, evaluation, diagnosis, and deployment evidence. The goal is to encourage researchers to produce reliable advances that future work can build upon.

## C. The educational cost of a model-first paradigm

Research practice and researcher education cannot be separated. Students learn what a field considers important from the work they are encouraged to reproduce, the questions asked in group meetings, the criteria used in paper review, and the achievements rewarded by leaderboards. If success is repeatedly presented as selecting a recent architecture and obtaining a higher score on an established public benchmark dataset, this procedure gradually becomes an implicit curriculum.

That curriculum can produce researchers who are highly competent in software engineering, optimization, and empirical experimentation but less prepared to ask foundational questions. They may know how to train a restoration transformer without being able to explain which spatial frequencies were lost; how to apply a diffusion prior without distinguishing plausible completion from measurement-supported recovery; or how to optimize low-light enhancement without analysing exposure, photon noise, clipping, illumination, and display rendering. The weakness is not the use of modern models. It is the absence of the problem knowledge needed to decide when, why, and under what conditions those models are appropriate.

This has consequences for innovation. Without a firm understanding of the physical and statistical problem, novelty is likely to be sought within the space already defined by the benchmark: another backbone, module, loss term, or training recipe. Fundamental progress often requires stepping outside that space—redefining the target, modelling a neglected cause, collecting a different measurement, exposing non-identifiability, or changing the evaluation. Researchers can only make that step when they understand the problem independently of the current solution machinery.

The required change is therefore both methodological and educational. Problem-first research should be taught as a habit of scientific reasoning: begin with the world and its measurements; determine what is known, lost, and ambiguous; justify the information introduced by a solution; define the intended decision; and then select the most appropriate model. Modern AI belongs fully within this curriculum, but as a powerful means of inference rather than the starting definition of the problem.

## D. Three meanings of progress

The word progress is commonly used for three different achievements:

1) Benchmark progress: improvement of a metric under a fixed dataset and protocol.

2) Method progress: a better estimator, representation, optimizer, or implementation under stated conditions.

3) Problem progress: better understanding or solution of the intended physical and decision problem over a meaningful deployment region.

Benchmark progress can provide evidence for method progress, and method progress can contribute to problem progress. Neither implication is automatic. A 0.1 dB improvement may be valuable if achieved reproducibly under a demanding controlled comparison; it may be immaterial if obtained through additional supervision, compute, data, or test-specific tuning that is not acknowledged. Conversely, a method that sacrifices a fraction of a benchmark score may constitute greater problem progress if it handles unknown sensors, exposes uncertainty, preserves data consistency, or reveals a previously ignored failure mechanism.

## E. A hierarchy of scientific questions

Before asking which architecture performs best, an imaging study should answer a hierarchy of questions:

1) What physical quantity or rendered image is desired, and for what use?

2) How was the observation formed, and which variables influenced it?

3) What information about the target remains in the observation?

4) What information has been destroyed or confounded?

5) Which additional assumptions, data, or priors are scientifically legitimate?

6) What output or decision is most appropriate for the intended application, and which types of error matter most?

7) Which computational method implements that action effectively?

8) What evidence would show that the method works beyond the development benchmark?

This hierarchy does not demand a perfect physical model before learning can begin. It demands that approximation be explicit. A deliberately simplified model can be scientifically powerful because its domain of validity can be investigated; an implicit model hidden in a dataset is harder to interrogate.

## III. THE INFORMATION PROBLEM

A broad imaging model is

$$
\begin{array} { r } { Y = \mathcal { H } ( X ; \eta ) + N , } \end{array}\tag{2}
$$

where $X$ is the desired image, Y the observation, H the acquisition or degradation process, η its parameters, and N disturbance. The linear case is

$$
\begin{array} { r } { Y = H X + N . } \end{array}\tag{3}
$$

Restoration constructs

$$
{ \widehat { X } } = \delta ( Y ) .\tag{4}
$$

But if

$$
{ \mathcal { H } } ( X _ { 1 } ) = { \mathcal { H } } ( X _ { 2 } ) , \qquad X _ { 1 } \neq X _ { 2 } ,\tag{5}
$$

the observation cannot distinguish the two latent images. Depth, attention, or parameter count cannot restore uniqueness; an estimator must inject additional knowledge.

We distinguish six sources:

$$
{ \cal K } = \{ { \cal K } _ { m } , { \cal K } _ { p } , { \cal K } _ { a } , { \cal K } _ { i } , { \cal K } _ { e } , { \cal K } _ { r } \} ,\tag{6}
$$

denoting information retained in the measurement, acquisition physics, analytic assumptions, internal image statistics, external

examples, and representation/architecture bias, respectively. Conceptually,

$$
\begin{array} { r } { \widehat { X } = F ( Y ; K _ { p } , K _ { a } , K _ { i } , K _ { e } , K _ { r } ) . } \end{array}\tag{7}
$$

Classical interpolation emphasizes $K _ { a } ;$ model-based inversion combines $K _ { p }$ and $K _ { a } ;$ example-based methods use $K _ { e } ;$ zeroshot super-resolution uses $K _ { i } \ [ 9 ] ;$ and Deep Image Prior demonstrates the utility of $K _ { r }$ <sub>r</sub> without external training [10]. The useful question is therefore not simply whether a method learns, but which source resolves which ambiguity.

A method may use several of the information sources in (6) together. Considering each source separately, however, makes it easier to identify why the method may fail. Possible causes include an inaccurate model of the imaging process, insufficient information in the observed image, differences between the training and deployment data, and learned preferences for structures that are not present in the actual scene.

## A. Identifiability, stability, and support

Three questions should be separated. Identifiability asks whether distinct latent images can produce the same observation. Stability asks whether small perturbations in the observation or model can cause large changes in the reconstruction. Support asks whether the deployed observation lies in a region for which the estimator has relevant knowledge. Deep capacity can approximate a complicated inverse on the training support, but it cannot, by itself, recover information that was not preserved during image acquisition, make an unstable inverse stable without regularization, or guarantee behaviour outside that support.

If H is differentiable locally, small changes satisfy approximately

$$
\begin{array} { r } { \Delta Y \simeq J _ { \mathcal H } ( X ) \Delta X + \Delta N . } \end{array}\tag{8}
$$

Small singular values of $J _ { \mathcal { H } }$ identify directions in image space that are weakly observed; its null directions —changes in the underlying image that the acquisition process cannot detect — are not observed at all. A network may output visually convincing content in these directions because the training distribution makes some contents more probable than others. That is successful prior-based inference when the task permits plausibility, but it is not direct recovery from the measurement.

## B. Evidence-supported and prior-supported content

The distinction between evidence and prior is continuous rather than binary. For a reconstructed feature ϕ(X), one may ask how sensitive it is to the measurement, the forward model, and the prior. A feature that persists under plausible changes of prior but changes predictably with the observation is relatively evidence-supported. A feature that varies strongly across equally plausible priors or posterior samples is priorsupported. This suggests practical diagnostics: perturb the assumed degradation, vary the learned prior or sampling seed, enforce re-measurement consistency, and visualize uncertainty in weakly observed regions.

TABLE I  
FOUR LEVELS THAT SHOULD BE REPORTED SEPARATELY IN AN IMAGE-PROCESSING STUDY.
<table><tr><td>Level</td><td>Central question</td><td>Typical choices</td><td>Typical failure when con- flated</td></tr><tr><td>Problem</td><td>What latent quantity should be inferred from what observation, under which acquisition and use?</td><td>Sensor-to-radiance recovery, canonical rendering, perceptual enhancement, recognition- oriented preprocessing</td><td>A benchmark construction silently becomes the real- world problem</td></tr><tr><td>Principle</td><td>What knowledge makes the problem solvable?</td><td>Bayesian inference, regulariza- tion, empirical risk minimiza- tion, invariance, self-consistency</td><td>A generic learning prin- ciple is presented as a problem-specific explana- tion</td></tr><tr><td>Estimator</td><td>Which decision is optimal for the loss and uncertainty?</td><td>Conditional mean, median, MAP estimate, posterior sample, ro- bust decision</td><td>Different outputs are ranked as if they estimated the same target</td></tr><tr><td>Implementation</td><td>How is the estimator repre- sented and computed?</td><td>Filter, optimization, CNN, trans- former, diffusion sampler, hybrid system</td><td>Architectural novelty is mistaken for new informa- tion or new physics</td></tr></table>

The word hallucination is sometimes used for any generated detail. A more precise account distinguishes: (i) plausible completion consistent with the intended task; (ii) prior-dominated detail that is not identifiable but is honestly represented as uncertain; and (iii) confidently asserted content inconsistent with the measurement or application. The scientific concern is principally the third category, and the second category when it is presented as factual recovery.

## C. The target is part of the problem

Many image-processing tasks do not possess one natural ground truth. A long exposure is not the unique truth corresponding to a short exposure when objects move. A fused or tone-mapped HDR image is a rendering decision. A highresolution capture at another focal length is not the latent array that would have been recorded by an ideal high-resolution version of the original sensor. A normally exposed photograph changes exposure, noise, clipping, colour processing, and sometimes illumination relative to a low-light photograph.

Thus the target variable X must be defined, not merely supplied. Is it scene radiance, reflectance, a canonical camera rendering, a preferred photograph, or an input optimized for a downstream machine? Ambiguous target construction changes $p ( X | Y )$ and therefore changes the optimal estimator. Dataset documentation is consequently part of problem formulation rather than an administrative detail.

## IV. ONE STATISTICAL BRIDGE ACROSS THREE ERAS

Classical regularized inversion writes

$$
\widehat { X } = \arg \operatorname* { m i n } _ { X } D ( Y , \mathcal { H } ( X ) ) + \lambda R ( X ) ,\tag{9}
$$

while Bayesian inference uses

$$
p ( X | Y ) \propto p ( Y | X ) p ( X ) .\tag{10}
$$

Given paired examples $\mathcal { D } = \{ ( Y _ { i } , X _ { i } ) \} _ { i = 1 } ^ { n }$ , a learned implementation uses

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \sum _ { i } L ( F _ { \theta } ( Y _ { i } ) , X _ { i } ) .\tag{11}
$$

These forms relocate knowledge—from explicit R or $p ( X )$ into examples and parameters—but need not change the target decision rule. For loss $L ( X , a )$ , the Bayes action is

$$
\delta ^ { * } ( y ) = \arg \operatorname* { m i n } _ { a } \mathbb { E } [ L ( X , a ) | Y = y ] .\tag{12}
$$

Under squared error,

$$
\delta ^ { * } ( y ) = \mathbb { E } [ X | Y = y ] .\tag{13}
$$

For jointly Gaussian variables this is linear and underlies Wiener estimation [1]. A shallow network, CNN, or transformer may approximate progressively richer versions of the same conditional mean. Thus deep learning changed representation learning, capacity, receptive field, optimization, and the scale of empirical priors. It did not repeal sampling, information loss, identifiability, the dependence of an optimum on its loss, or the dependence of a learned estimator on its data distribution.

Generative methods do expand the available decision rules. Beyond the conditional mean, one may choose

$$
{ \widehat { X } } _ { \mathrm { M A P } } = \arg \operatorname* { m a x } _ { X } p ( X | Y )\tag{14}
$$

or sample

$$
{ \widetilde { X } } \sim p ( X | Y ) .\tag{15}
$$

The latter can represent multimodality, but plausible detail is not automatically measurement-supported detail [11], [12].

Table II shows why “classical versus learning” is incomplete. Dictionary methods learn representations but retain explicit inference; plug-and-play methods insert learned denoisers into model-based solvers; and diffusion restoration can combine a likelihood with a learned score prior [13]–[15]. A better taxonomy asks what is known, what is learned, what decision is made, and how it is computed.

## V. WHAT MODERN AI CHANGES AND WHAT IT DOES NOT A. Changes that are scientifically consequential

Modern AI has changed image processing in at least six substantive ways. First, learned hierarchical representations reduce dependence on manually designed features. Second, large receptive fields and attention allow nonlocal and semantic context to influence local reconstruction. Third, scale permits empirical priors to encode far more diverse image statistics than classical parametric models. Fourth, amortized inference moves computation from repeated per-image optimization into training, enabling fast deployment. Fifth, generative models can represent multimodal conditional distributions rather than a single point estimate. Sixth, pretraining and adaptation permit knowledge to be transferred across tasks and domains.

TABLE II  
A PROBLEM-FIRST COMPARISON OF THREE BROAD IMAGE-PROCESSING ERAS. THE BOUNDARIES ARE POROUS; THESE ARE DOMINANT TENDENCIESRATHER THAN EXCLUSIVE PROPERTIES.
<table><tr><td>Dimension</td><td>Designed/model-based process- Learning by example ing</td><td></td><td>Deep and generative learning</td></tr><tr><td>Representation</td><td>and regularizers</td><td>Usually designed; assumptions Designed features with learned Representation and processing visible in filters, transforms, predictors, dictionaries, exem- learned jointly at high capacity plars, or priors</td><td></td></tr><tr><td>edge</td><td>Primary knowl- Physics and analytic assump- Internal or external examples Large external priors, archi- tions  $( K _ { p } , K _ { a } )$ </td><td> $( K _ { i } , K _ { e } ) ,$  structure</td><td>often with explicit tecture bias, and increasingly physics/internal adaptation</td></tr><tr><td></td><td>Typical inference Direct filtering or explicit opti- Regression, lookup, retrieval, mization</td><td>or sparse inference</td><td>End-to-end regression, unfold- ing, adversarial inference, or posterior sampling</td></tr><tr><td>Main strength</td><td>predictable constraints</td><td>Transparency, data efficiency, Empirical flexibility with mod- erate interpretability</td><td>Capacity, context, transfer, and complex or multimodal condi- tionals</td></tr><tr><td>Characteristic risk</td><td>ors</td><td>Model bias and restrictive pri- Limited examples or represen- tation capacity</td><td>Distribution shift, opaque pri- ors, and convincing unsup- ported detail</td></tr><tr><td>What remains in- variant</td><td colspan="3">Observation formation, information loss, identifiability, loss-dependent decisions, and the need to justify prior information</td></tr></table>

These are not cosmetic implementation changes. They can alter which priors are available, which estimator can be approximated, and which problems are computationally feasible. A problem-first view therefore does not imply a return to handdesigned filters. It asks that these new capabilities be located accurately in the inference chain.

## B. Persistent constraints

Several constraints remain. The sensor still samples a physical field; saturation still discards magnitude; downsampling still removes or aliases frequencies; noise still reduces information; and multiple latent scenes can still explain one observation. The optimum remains conditional on a loss and distribution. A generative model can produce realistic textures, but realism is a property of a distribution or observer response, not proof that the particular texture occurred in the measured scene.

Likewise, scale does not make training data neutral. Large pretraining datasets reflect the capture devices and processing conventions used to create them, as well as uneven representation of different cultures and visual content, licensing restrictions, and data-selection processes.

## C. Hybridization is often the natural consequence

The problem-first framework frequently leads to hybrids rather than a choice between physics and learning. Known acquisition can enter through a likelihood or data-consistency term; unknown parameters can be estimated; a learned prior can supply image statistics; internal adaptation can use the test observation; and uncertainty can be represented through posterior samples or calibrated intervals. The resulting system may still use a transformer or diffusion model, but its role is explicit.

This also clarifies interpretability. A completely transparent small model is not automatically correct, and a large network is not automatically uninterpretable at the scientific level. Even when individual features are opaque, the system can remain scientifically interpretable if the target, acquisition family, information sources, loss, consistency constraints, and validated domain are explicit.

## VI. BENCHMARKS AS SCIENTIFIC INSTRUMENTS

## A. What a benchmark legitimately provides

A benchmark freezes some variables so that methods can be compared. It may provide common data, a reference construction, an evaluation metric, and a public protocol. These controls reduce experimental degrees of freedom and allow a community to accumulate results. Synthetic benchmarks additionally permit exact pairing and controlled perturbation; captured benchmarks incorporate effects that a simulator may miss. Both are indispensable.

But a benchmark is an instrument with a transfer function. Its selection of scenes, sensors, targets, degradations, preprocessing, and metrics determines what it is sensitive to and what it ignores. A score is evidence about performance under that instrument. Generalization to a broader claim requires additional argument and experiments.

Let B denote the benchmark construction, including its acquisition or synthesis process, sampling of scenes, target definition, and metric. A reported score is more precisely

$$
S ( F ; B , \Pi , T , C ) ,\tag{16}
$$

where Π is the evaluation protocol, T the model-selection and tuning procedure, and C the allowed resources. Omitting Π, T, or $C$ makes apparently identical comparisons potentially unequal.

## B. A taxonomy of benchmark mismatch

Benchmark–deployment mismatch has several distinct forms:

1) Forward-process mismatch: synthetic blur, noise, exposure, or compression differs from acquisition.

2) Scene and content mismatch: subjects, textures, motion, geography, or semantics differ.

3) Sensor and pipeline mismatch: optics, CFA, ISP, colour response, bit depth, and codec differ.

4) Target mismatch: the reference image represents a different exposure, rendering, viewpoint, or preference.

5) Loss and use mismatch: the optimized metric does not reflect human judgement, fidelity, safety, or downstream performance.

6) Protocol mismatch: preprocessing, crop conventions, colour spaces, test adaptation, or external data differ.

These mismatches should not be collapsed into a generic statement that “real data are different.” Each suggests different diagnostics and remedies.

## C. Benchmark saturation and adaptive overfitting

A test set may be nominally held out while influencing research indirectly through repeated publication, leaderboard feedback, visual inspection, and community-wide tuning. Over time, architectural and training choices become adapted to its idiosyncrasies. This is not necessarily misconduct; it is a predictable consequence of repeated selection. Studies using newly collected ImageNet-like test sets illustrate how performance can change under even carefully reconstructed data collection [16]. The scientific response is to maintain hidden or refreshed tests, use multiple independent benchmarks, report cross-dataset transfer, and resist interpreting tiny maturebenchmark gains as broad progress without corroboration.

## VII. STATE OF THE ART, REPRODUCIBILITY, AND SCIENTIFIC CLAIMS

## A. SOTA is a relation, not a property

No method is simply “state of the art.” It is state of the art with respect to a dataset, split, metric, preprocessing convention, resource envelope, and date. Different losses can reverse rankings; confidence intervals can overlap; and increased data or compute may account for the difference. The phrase is scientifically useful only when these conditions are visible.

We propose a claim ladder:

1) Numerical claim: the submitted outputs obtain a stated score.

2) Comparative claim: the method outperforms specified baselines under matched conditions.

3) Mechanistic claim: evidence attributes the gain to the proposed idea rather than confounds.

4) Generalization claim: the gain survives relevant shifts and independent tests.

5) Problem-progress claim: the method improves the intended real task or advances its scientific understanding. Each step requires new evidence. A leaderboard normally establishes at most the first two.

## B. Reproducibility has several levels

Computational reproducibility asks whether the same code and environment recover the result. Experimental reproducibility asks whether an independent implementation under the described protocol reaches the same conclusion. Inferential reproducibility asks whether reasonable changes of dataset, seed, metric, or analysis preserve the claimed ordering. A downloadable checkpoint can satisfy the first while leaving the latter two unanswered.

For mature benchmarks where differences are small, papers should report enough information to reconstruct data generation, preprocessing, training schedules, model selection, test-time processing, external data, and compute. Repeated runs should be reported when randomness is comparable with the claimed gain. Baselines should receive a fair tuning budget and the comparison should separate the proposed idea from extra parameters, pretraining, supervision, or inference cost. These requirements align with broader machine-learning reproducibility practice, which treats transparent experimental workflows as part of the mechanism by which findings become accessible and reliable [17].

## C. Negative and Diagnostic Results

The current research environment rewards a new maximum on a benchmark more visibly than a careful diagnosis of failure. This can make students and early-career researchers reluctant to report negative results, particularly when they believe that a paper without a state-of-the-art score is unlikely to be accepted. Consequently, unsuccessful experiments are often omitted, common failures are repeatedly rediscovered by different groups, and research effort remains concentrated on small improvements within established settings. This creates an incomplete scientific record and can delay recognition that the current formulation, data, or evaluation protocol is inadequate.

A negative result, however, is not simply the absence of an improvement. A scientifically valuable negative result tests a clearly stated hypothesis under appropriate and carefully controlled conditions. It verifies that the observed failure is not caused by an implementation error, inadequate optimization, an unfair baseline, or insufficient experimental effort. It then establishes where the failure occurs, whether it is reproducible across methods and datasets, and which assumptions are responsible. For example, showing that several substantially different architectures fail under the same change in sampling, sensor, illumination, or noise conditions can demonstrate that the limitation lies in the problem formulation or available information rather than in a particular network design.

TABLE III  
A PORTFOLIO OF EVIDENCE FOR CLAIMS ABOUT AN IMAGING METHOD.
<table><tr><td>Evidence component</td><td>What it establishes</td><td>What it does not establish alone</td></tr><tr><td>Matched synthetic benchmark</td><td>Controlled accuracy for a known degra- dation and reference</td><td>Robustness to unknown acquisition or real sensor pipelines</td></tr><tr><td>Captured paired bench- mark</td><td>Performance for a specific capture and registration protocol</td><td>Universal real-world generalization or unique ground truth</td></tr><tr><td>Cross-dataset/sensor transfer</td><td>Sensitivity to dataset and device shift</td><td>Coverage of all deployments</td></tr><tr><td>Forward-model stress test</td><td>Behaviour under controlled physical pa- Adequacy of the chosen model family rameter changes</td><td></td></tr><tr><td>Re-degradation consis- tency</td><td>under an assumed model</td><td>Compatibility of output with observation Truth of prior-dominated null-space detail</td></tr><tr><td>Unpaired in-the-wild testing</td><td>Qualitative failure modes and deployment Full-reference fidelity diversity</td><td></td></tr><tr><td>Human/downstream evaluation</td><td>Utility for a specified observer or task</td><td>Physical faithfulness for other uses</td></tr><tr><td>Repeated runs and abla- tions</td><td>ported protocol</td><td>Stability and attribution under the re- Generalization beyond that protocol</td></tr></table>

Diagnostic results can be especially valuable in image processing because they reveal the boundary between conditions under which a method is reliable and conditions under which it is not. Sensitivity analysis can determine how performance changes as blur, noise, exposure, sampling, compression, or distribution mismatch increases. Ablation and controlled interventions can identify which source of information supports a method and when that source becomes unreliable. Crossdataset and cross-sensor experiments can reveal whether an apparent advance is tied to one benchmark construction. Such findings map the validated operating region of a method and indicate where additional measurements, better physical models, different priors, or new evaluation protocols are needed.

Young researchers should therefore be encouraged to frame a well-supported negative result as a positive contribution to knowledge. A strong paper of this kind should state the original expectation, describe the controls used to test it, report the conditions under which it fails, provide evidence for the cause of failure, and explain the consequences for future research. Where possible, it should release the experimental protocol, failed cases, and relevant code or data so that other researchers can verify the conclusion and avoid repeating the same unsuccessful path. The contribution is not that “the method did not work,” but that the study has established why it did not work, under what conditions it fails, and what must change for further progress to become possible.

Reviewers and editors also have an important role. A paper should not be rejected merely because it does not report a new highest benchmark score. It should instead be judged according to whether the question is important, the experiments are rigorous, the negative conclusion is reproducible, plausible alternative explanations have been examined, and the findings change the community’s understanding or future research decisions. This does not lower the standard for publication.

On the contrary, a convincing negative result often requires stronger controls and more careful reasoning than a small positive improvement.

A problem-first research culture should therefore recognise calibrated negative results, sensitivity analyses, failure taxonomies, and mappings of validated operating conditions as legitimate scientific contributions. These studies reduce duplicated effort, prevent unsupported claims from becoming established assumptions, and identify where new measurements, formulations, or scientific ideas are required. Making such work visible and publishable would also allow young researchers to investigate difficult and uncertain questions without feeling that only a favourable numerical outcome can justify the effort.

## VIII. CASE STUDY I: SUPER-RESOLUTION AND THE SAMPLING DEFICIT

Super-resolution is fundamentally a problem of insufficient spatial measurement. A camera does not observe an HR array and subsequently apply the digital resize operation used by a benchmark. Scene radiance is filtered by the optics, integrated over finite pixel areas, sampled on a sensor lattice, corrupted by noise, and processed by a camera pipeline. A useful abstraction is

$$
\begin{array} { r } { Y = \mathcal { P } [ D _ { s } ( K X ) + N ] , } \end{array}\tag{17}
$$

where K represents optical blur and sensor integration, $D _ { s }$ represents spatial sampling, N is measurement noise, and $\mathcal { P }$ contains operations such as mosaicing, demosaicing, sharpening, quantization, and compression. When scene frequencies exceed the effective passband and sampling limit, they are attenuated, removed, or aliased. The observation therefore contains too few reliable independent measurements to determine the desired HR image.

Let A denote the resulting linearized measurement operator. Its null space

$$
\mathcal { N } ( A ) = \{ v : A v = 0 \}\tag{18}
$$

implies

$$
A ( X + v ) = A X , \qquad v \in { \mathcal { N } } ( A ) .\tag{19}
$$

This non-identifiability, rather than visible pixelation itself, is the root inference problem. Conceptually,

$$
X = A ^ { \dagger } Y + ( I - A ^ { \dagger } A ) Z ,\tag{20}
$$

where $A ^ { \dagger } Y$ is constrained by the measurement and $( I - A ^ { \dagger } A ) Z$ lies in the unobserved null space. A single-image algorithm can estimate the latter only by importing assumptions or prior information. It cannot establish that prior-supported detail is the uniquely recovered scene.

## A. Two physical causes obscured by one benchmark label

Practical SR couples two causes. The first is insufficient sampling: the optics–sensor system has not acquired enough spatial measurements to represent the desired bandwidth. The second is limited measurement quality: optical attenuation, pixel integration, noise, CFA sampling, demosaicing, compression, and camera processing reduce the reliability or accessibility of the samples that remain. These causes interact. Aliasing may preserve high-frequency energy in displaced form, while optical low-pass filtering can remove it; noise can make nominally observed frequencies unusable; and demosaicing can mix colour reconstruction with spatial-resolution loss.

A fixed bicubic reduction (or any other sub-sampling operation) does not reproduce this causal process. It begins with an already captured HR image and digitally constructs

$$
Y _ { b } = D _ { s } ( K _ { b } X ) ,
$$

where $K _ { b }$ and $D _ { s }$ are prescribed by the benchmark. This creates a clean and reproducible conditional task, but it should not be confused with the original problem of acquiring insufficient, noisy, possibly aliased samples through an unknown optical– sensor pipeline. Similar visual appearance does not imply identical missing information.

B. What the recent literature addresses—and what it leaves implicit

Classical interpolation fills missing samples using analytic smoothness; reconstruction methods impose explicit priors; external learning estimates LR–HR relations from datasets; and internal learning estimates them from recurrence within the test image. Early progressively predictive learning transferred a relation learned between known coarse levels to an unknown finer level [3]; example-based SR retained external LR–HR exemplars [4]; ZSSR trained an image-specific CNN from the test image [9]; and modern CNNs scale external conditional regression [6]. All address the ambiguity in (19), but obtain and encode the missing information differently.

Important recent exceptions begin closer to the root cause. Handheld multi-frame SR exploits natural subpixel motion to acquire complementary RAW samples, increasing both spatial information and SNR [18]; deep burst SR learns alignment and fusion of such multiple measurements [19]. Null-space learning explicitly preserves the measurement-determined component and learns only the unobserved component from a GAN prior [20]; unfolding methods derive network stages from data fidelity and an assumed degradation operator [21]. These works exemplify different problem-first responses: collect more evidence, separate evidence from prior, or make the forward operator govern inference.

The main benchmark-driven SISR stream, however, commonly begins one level later. An HR public dataset is selected, a fixed degradation generates paired LR inputs, a CNN, transformer, GAN, or diffusion architecture is chosen, and training optimizes reconstruction or perceptual scores. Richer blind-degradation simulations improve transfer, and stronger generative priors improve visual plausibility, but neither changes the central fact that one LR observation does not determine its null-space content. A method can therefore respond impressively to the appearance of low resolution without directly addressing why the relevant information was not acquired.

This is not a claim that learned SISR lacks value or that every recent paper is model-first. It is a distinction between three kinds of advance: a better estimator for a prescribed benchmark conditional; a better prior for plausible synthesis; and better acquisition or inference for the underlying sampling problem. These contributions should not be collapsed into one ranking. A physically meaningful SR study should say which component is supported by measurements, which is imposed by prior information, and whether its objective is faithful reconstruction, blind camera restoration, cross-device translation, or plausible generation.

The epistemic distinction can be summarized as

$$
\begin{array} { r } { \widehat { X } = X _ { \mathrm { e v i d e n c e } } + X _ { \mathrm { p r i o r } } , } \end{array}\tag{21}
$$

where the decomposition need not be computationally unique. In scientific or medical imaging, reporting which structures are prior-dominated may matter as much as visual sharpness.

Thus, higher PSNR under bicubic degradation demonstrates a better estimator for that benchmark, not necessarily recovery of real sensor detail; perceptual sharpness may also trade against fidelity. Evaluation should align the acquisition model, lossinduced estimator, and application, using realistic degradations, data consistency, and uncertainty rather than one average score.

a) Benchmark mismatch in SR.: The conventional SR benchmark is built from HR photographs rather than native LR acquisition. DIV2K provides 800/100/100 HR training/validation/test images, with LR inputs commonly generated by known bicubic downsampling [22]; Set5, Set14, BSD100, Urban100, and Manga109 are tested similarly. This gives exact registration and an unambiguous target, but defines $p _ { \mathrm { b e n c h } } ( Y | X )$ by the imposed kernel, omitting or simplifying optics, sensor sampling, demosaicing, signal-dependent noise, sharpening, compression, spatial variation, and processing order. It therefore benchmarks a matched simulator, not blind SR for an arbitrary camera.

Captured benchmarks reduce but do not remove the gap. RealSR changes focal length on two cameras and registers paired static scenes [23]; DRealSR broadens real scenes and degradations [24]. They contain genuine optical/camera effects, but the HR target is a second capture, so focal setting, alignment/resampling, exposure and ISP differences enter the pair. Static scenes and limited rigs narrow the distribution: these are valuable real-pair benchmarks, not universal ground truth for real-world SR.

## C. Four different tasks called super-resolution

At least four tasks are commonly grouped under one name. Known-degradation reconstruction estimates an HR image under a specified D, K, and noise model. Blind restoration must infer or marginalize unknown degradation. Cross-device enhancement learns a mapping defined by a particular paired capture rig. Generative upsampling seeks a plausible highresolution rendering conditioned on a low-resolution input. These tasks overlap, but they have different notions of correctness. PSNR against a bicubic-derived reference is appropriate for the first; perceptual preference may be central to the fourth; neither alone establishes faithful recovery from an unknown camera.

The distinction becomes more important as the prior becomes stronger. A diffusion or large generative prior can produce semantically convincing text, faces, vegetation, and masonry. If those patterns are weakly constrained by Y , the output may be useful for visual media while being unsuitable for evidence preservation, measurement, medicine, or remote sensing. The method should state which use it serves and expose prior dependence when factual fidelity matters.

## D. A stronger experimental design for SR

A problem-first SR experiment would begin by deciding whether the information deficit should be reduced through acquisition or managed through inference. Where possible, multiple shifted frames, alternative sensor layouts, optical coding, or other complementary measurements should be considered before asking a single network to synthesize all missing detail. For a single observation, the study would begin with a deployment envelope rather than one degradation. It would calibrate or bound optical modulation, pixel integration, sampling and aliasing, noise, mosaicing, compression, and processing order; reserve combinations outside the training support; and include native low-resolution images not generated from the target. The report would separate:

1) accuracy on the matched simulator;

2) robustness to forward-parameter changes;

3) transfer to independent sensors and pipelines;

4) data consistency after re-applying plausible degradations;

5) perceptual utility where the task permits generative detail; and

6) uncertainty, prior sensitivity, or solution diversity in the null space; and

7) gains due to genuinely additional measurements from gains due to a stronger learned prior.

This design can still identify a superior transformer or diffusion implementation, but its conclusion is stronger because the architecture is evaluated as one component of a defined inverse problem.

## IX. CASE STUDY II: LOW-LIGHT ENHANCEMENT AND THE PROXY PROBLEM

A common synthetic pair darkens a normally exposed image by a nonlinear curve and then adds noise, for example

$$
Y _ { s } = Q [ \mathcal { G } _ { \gamma } ( \alpha X ) + N _ { s } ] , \qquad 0 < \alpha < 1 ,
$$

where $\mathcal { G } _ { \gamma }$ is a gamma or related tone curve. This is more realistic than simple intensity scaling because it can imitate dark appearance, nonlinear camera response, quantization, and noise. Nevertheless, it still starts from a well-exposed X in which scene structure, colour, and shadow detail have already been captured. The transformation changes those recorded values; it does not reproduce why the values were difficult to record in the first place.

Real low-light formation is better described at the radiance and acquisition levels. Conceptually, let scene radiance be generated by reflectance R under spatially varying illumination L and recorded by a camera operator C:

$$
Y _ { r } = \mathcal { C } ( R \odot L ; t , \theta ) + N ,
$$

where t is exposure and θ collects aperture, sensor response, clipping, quantization, and in-camera processing. Photon scarcity and read noise reduce shadow SNR, while the coexistence of very dark and bright regions can exceed the usable dynamic range. Clipping at either end and coarse quantization are many-to-one operations: some information was never captured, rather than merely hidden by a gamma curve.

Low-light enhancement is therefore governed by two coupled root causes: insufficient dynamic range and insufficient signalto-noise ratio. Wu et al. formulate this distinction explicitly and address the former through a simplified inverse-rendering model [25]. Their method recovers a simplified scene representation and re-renders it under a canonical light source, shifting the objective from inverting an arbitrary darkening curve to correcting image formation from first principles. In conceptual form,

$$
Y _ { r } \longrightarrow ( \widehat { R } , \widehat { L } ) \longrightarrow \widehat { X } _ { c } = { \mathcal C } ( \widehat { R } \odot L _ { c } ; t _ { c } , \theta _ { c } ) ,
$$

where $L _ { c }$ denotes canonical illumination. This view separates scene reflectance from illumination and asks how the scene should be rendered, rather than assuming that a unique normallight image can be recovered by applying the inverse of the synthetic gamma function.

## A. Two physical causes obscured by one benchmark label

The HDR and low-SNR causes should be distinguished because they destroy information in different ways and therefore require different solution principles. Consider first an evenly illuminated scene recorded at a weak but spatially uniform light level. If the response is unsaturated, sufficiently sampled, and has adequate SNR, much of the visible darkness can in principle be corrected by exposure compensation and an appropriate rendering transform. Darkness by itself is not the difficult inverse problem. The difficulty begins when too few photons are collected: shot noise becomes large relative to signal, read noise and quantization become consequential, and amplification raises noise together with the desired signal. No tone curve or larger network can recreate measurement evidence that the sensor did not record.

The HDR cause is different. In many night, indoor, backlit, and unevenly illuminated scenes, bright light sources or well lit surfaces coexist with deep shadows. A single exposure cannot place the entire radiance range inside the sensor’s usable interval. Exposing for highlights drives shadows toward the noise floor; exposing for shadows clips highlights. The problem is therefore not adequately described as “make a dark image brighter.” It is to infer and render a scene whose spatial radiance range exceeds what the particular exposure and sensor can faithfully record. Illumination, reflectance, exposure, saturation, sensor noise, camera response, and the desired display rendering are coupled.

These causes also interact. A high-dynamic-range scene forces an exposure compromise that often creates low SNR in the dark regions; local brightening then exposes the noise produced by that compromise. Conversely, a globally photonlimited scene may have modest scene dynamic range but remain difficult because all measurements are uncertain. A scientifically defined LLIE task should therefore diagnose at least two variables,

$$
\underbrace { \rho _ { \mathrm { s c e n e } } / \rho _ { \mathrm { u s a b l e } } } _ { \mathrm { d y n a m i c - r a n g e ~ p r e s s u r e } } \quad \mathrm { a n d } \quad \underbrace { \mathrm { S N R } ( u ) } _ { \mathrm { s p a t i a l ~ m e a s u r e m e n t ~ r e l i a b i l i t y } } ,
$$

rather than treating “low light” as one scalar degradation strength. Here $\rho _ { \mathrm { s c e n e } }$ is the scene-radiance range, $\rho _ { \mathrm { u s a b l e } }$ is the reliably recordable sensor range under the selected exposure, and $\operatorname { S N R } ( u )$ varies spatially with the collected signal and sensor characteristics.

## B. What the recent literature addresses—and what it leaves implicit

Important exceptions address parts of this diagnosis. RAW short-to-long-exposure learning models genuine photon and read noise more faithfully than gamma-darkened RGB pairs [26]; SNR-aware enhancement explicitly changes processing according to spatial measurement reliability [27]; Retinexbased methods represent illumination variation [28], [29]; and simplified inverse rendering treats the illumination and dynamicrange cause at the image-formation level [25]. These works demonstrate that modern learning can be made substantially more effective when its role follows a physical diagnosis.

The broader literature, however, often begins one level later. A public dataset supplies low/normal pairs; its reference image is accepted as the target; a CNN, transformer, GAN, diffusion model, state-space model, or colour representation is selected; and the method is optimized for benchmark fidelity or perceptual quality. Noise suppression, illumination estimation, colour correction, and detail recovery are then introduced as network modules, frequently without first asking which failures arise from HDR exposure compromise, which arise from photon-limited measurement, which information is clipped or below the noise floor, and which output attributes are rendering preferences rather than recoverable scene properties. The method may respond successfully to visible symptoms while leaving the causal task under-specified.

It is not that all recent LLIE work ignores physics, nor that architectural innovation is unimportant. Rather, the main benchmark-driven stream rarely uses the two root causes to determine the target representation, required measurements, limits of recovery, training distribution, architecture, and evaluation protocol as one coherent chain. Consequently, methods addressing different conditionals—gamma inversion, paired-exposure translation, RAW photon-limited restoration, illumination correction, HDR reconstruction, or preferred tone rendering—are often compared in a common table as though they solved the same problem.

A problem-first study would reverse this order. It would first determine whether the deployment regime is dominated by dynamic-range pressure, low SNR, or both. It would then decide whether the scientifically appropriate intervention is improved acquisition (longer or multiple exposures, burst fusion, larger well capacity, RAW sensing, or HDR/event measurements), a calibrated noise estimator, inverse rendering, display tone mapping, or a combination. Only after those choices would it select the learned architecture. This diagnosis also changes evaluation: HDR-dominated cases require highlight/shadow information and rendering assessments, whereas SNR-dominated cases require signal-dependent noise, detail confidence, and crosssensor tests. A single average PSNR on a mixed benchmark cannot reveal which root cause has actually been addressed.

Consequently,

$$
p _ { s } ( Y | X ) \neq p _ { r } ( Y | X )
$$

is not merely a difference in noise level. The synthetic and real processes can differ in their latent variables, dynamic-range constraints, clipping, and the information preserved in shadows and highlights. They therefore define different conditional inference problems.

Let the MSE-optimal estimators under training and deployment distributions be

$$
F _ { \mathrm { t r a i n } } ^ { * } ( y ) = \mathbb { E } _ { \mathrm { t r a i n } } [ X | Y = y ] , \quad F _ { \mathrm { r e a l } } ^ { * } ( y ) = \mathbb { E } _ { \mathrm { r e a l } } [ X | Y = y ] .
$$

Then

$$
\boxed { p _ { \mathrm { t r a i n } } ( X | Y ) \neq p _ { \mathrm { r e a l } } ( X | Y ) \Rightarrow F _ { \mathrm { t r a i n } } ^ { * } \neq F _ { \mathrm { r e a l } } ^ { * } . }\tag{22}
$$

Equation (22) formalizes why unlimited capacity and perfect optimization cannot rescue a benchmark whose paired-data mechanism defines the wrong conditional task. A learner may become excellent at undoing gamma and injected noise while failing on illumination variation, shadow SNR, saturation, colour shifts, and real camera processing. Synthetic data remain valuable when their rendering and sensor model preserve the relevant physics; the issue is not simulation itself but whether it captures the causes of the deployed low-light problem [25], [28], [30].

The logic extends beyond low light: Gaussian noise, bicubic downsampling, and invariant blur are not universal acquisition models. Simplification is legitimate when scoped; validation should compare relevant conditional properties and use real pairs to test the simulator.

a) Benchmark mismatch in LLIE.: LOL-v1 contains 500 captured low/normal pairs (485 training, only 15 test) [28]; LOL-v2 adds real 689/100 and synthetic 900/100 splits [31]. They enable paired training, but define “normal” by their reference exposure and capture protocol. Limited scenes, alignment, cameras and exposure/ISO settings sample only part of illumination, dynamic range, motion, sensor, ISP and rendering diversity. LOL-v2-Synthetic further starts from an already informative normal image.

SID pairs 5,094 raw short exposures with long exposures from 424 static scenes and two cameras [26]. It captures genuine photon/read noise, but its static, sensor-specific raw-tolong-exposure task does not cover motion, processed JPEGs, other sensors, or illumination correction. SICE uses 589 multiexposure sequences and references selected from HDR/fusion results [32]; it captures dynamic-range issues, but its target is a preferred rendering, not unique physical truth. LOL exposure translation, SID raw short-to-long imaging, and SICE tone rendering are therefore related but different conditionals.

## C. Low light is not one degradation variable

The phrase “low-light image” conflates several causes. Global under-exposure can leave recoverable headroom; photon limited shadows can have fundamentally poor SNR; strong illumination variation produces simultaneous deep shadow and highlight clipping; an unsuitable white balance or ISP can introduce colour error; motion can make a long-exposure reference geometrically different; and a dark artistic rendering may contain adequate sensor information but intentionally low display luminance. These conditions should not share one unqualified target.

The HDR perspective is central when bright and dark scene regions exceed the usable capture range. Enhancement is then not simply monotonic brightening. Recovering or rendering the scene requires decisions about illumination, reflectance, noise, saturation, local contrast, colour appearance, and display range. Artificially darkening a normally exposed image changes its appearance but normally preserves a transformed version of information that has already been captured. It therefore risks teaching inversion of an artificial operation rather than recovery from limited exposure and dynamic range.

## D. Reference ambiguity and evaluation

For LLIE, two observers may legitimately prefer different normal renderings. Full-reference metrics penalize deviation from the chosen reference even when another output is visually or physically reasonable. Conversely, no-reference aesthetic metrics may reward brightness, contrast, or familiar colour while overlooking amplified noise or fabricated structure. Downstream recognition can establish utility for a particular machine task but not photographic fidelity.

A defensible evaluation therefore combines reference fidelity where a reference is meaningful, raw or forward consistency where acquisition is known, HDR and colour diagnostics, crosscamera transfer, human preference for rendering objectives, and task performance when the declared purpose is machine vision. Results should also be stratified by cause: uniform underexposure, spatially varying illumination, extreme dynamic range, saturation, noise floor, motion, and processed versus raw input.

## E. What a Problem-First LLIE Study Would Contribute

A problem-first LLIE study should begin by identifying which physical cause, or combination of causes, makes the observed image difficult to use. As discussed above, the two principal causes are insufficient dynamic range and low SNR. They are related but not identical. A scene containing bright and dark regions may exceed the usable dynamic range of the camera, forcing an exposure compromise between highlight saturation and unreliable shadows. A more uniformly dark scene may remain within the sensor’s dynamic range but contain too few detected photons to provide reliable measurements. The first problem concerns the range of radiance that can be captured simultaneously; the second concerns the reliability of the captured signal. A study that does not distinguish these causes may improve the appearance of benchmark images without establishing which real limitation it has addressed.

The strongest contribution therefore need not be a new network backbone. It may instead be a better formulation of the LLIE problem. For example, a study could develop and validate an inverse-rendering abstraction that separates illumination, reflectance, sensor response, and display rendering sufficiently well for the intended application. Its scientific contribution would lie in showing that these variables explain important failures of existing methods and that modelling them leads to more reliable enhancement. A learned model may still be essential, but its design would follow from the proposed imageformation model rather than serve as the starting point of the research.

A second contribution could be a calibrated family of acquisition and noise models. Rather than adding arbitrary Gaussian noise to an artificially darkened normal image, such a study could model photon-dependent noise, read noise, clipping, quantization, exposure, colour response, and incamera processing across a meaningful range of sensors. The important result would not merely be that training with more complicated degradation improves one score. It would be evidence that the model reproduces relevant properties of real low-light measurements, predicts where existing methods fail, and improves transfer to cameras or conditions not represented in the training data. A useful simulator should be judged by how faithfully it supports real-world inference, not simply by how visually convincing its synthetic images appear.

A third contribution could redefine the desired output. Many LLIE datasets treat one normally exposed photograph as the unique reference, even though enhancement may involve several different objectives. The desired result could be an estimate of scene radiance, a canonical rendering under specified illumination, an HDR representation, a visually preferred photograph, or an image optimized for a downstream vision task. These targets are not interchangeable. A representation that separates the estimated scene from its final display rendering would allow researchers to evaluate physical recovery and rendering preference separately. It would also prevent a method from being penalized simply because its valid rendering differs from one selected reference image.

Dataset design is another potentially fundamental contribution. A problem-first dataset could control variables that existing benchmarks confound, such as illumination distribution, scene dynamic range, exposure, sensor type, photon level, motion, saturation, colour processing, and target rendering. It could include separate subsets dominated by HDR limitations, low SNR, or their interaction. This would allow researchers to determine whether a method genuinely addresses a particular cause or merely performs well on the average composition of a benchmark. Carefully documenting how each input and reference was captured or constructed would make the conditional task represented by the dataset explicit.

A problem-first study may also contribute a better account of uncertainty and recoverability. Some shadow regions contain weak but recoverable measurements; others lie at or below the noise floor. Some highlights retain information in one colour channel; others are fully saturated. An enhancement method should ideally distinguish these situations. Instead of producing an equally confident output everywhere, it could estimate spatial confidence, indicate where detail is supported by the measurement, offer alternative plausible renderings, or decline to assert detail that cannot be justified. Such mechanisms are particularly important when enhanced images are used for scientific observation, medical analysis, surveillance, autonomous systems, or other applications in which plausible fabrication may be mistaken for evidence.

Evaluation should follow the claimed contribution. An inverse-rendering model should be tested for consistency with image formation and transfer across illumination conditions. A sensor/noise model should be validated against real measurements from independent devices. A proposed scene representation should be evaluated separately from the chosen display rendering. A new dataset should demonstrate which previously hidden failure modes it reveals. An uncertainty method should be assessed for calibration and its ability to identify irrecoverable regions. Conventional PSNR, SSIM, perceptual metrics, and user studies may remain useful, but no single metric can establish all these claims.

Architecture then becomes an enabling design choice. A CNN may be appropriate for efficient local processing, a transformer for modelling spatially varying illumination, a diffusion model for representing multiple plausible solutions, or a compact inverse-rendering network for interpretability and deployment efficiency. The architecture should be selected because its properties address a requirement identified by the problem analysis. Its contribution should be evaluated in terms of accuracy, robustness, computational cost, interpretability, and dependence on prior information, rather than only by whether it produces the highest score on one benchmark.

This broader allocation of scientific credit is important for the development of the field. It recognises advances in problem formulation, acquisition modelling, target definition, dataset construction, uncertainty analysis, and evaluation as substantial contributions in their own right. It also encourages researchers to investigate why low-light images are difficult rather than repeatedly optimizing models for the most convenient artificial representation of that difficulty. A problem-first LLIE study should ultimately leave the community with a clearer understanding of the problem, a more reliable way of solving or evaluating it, or new evidence about what can and cannot be recovered—not merely another entry at the top of a benchmark table.

## X. A PROBLEM-FIRST RESEARCH WORKFLOW

The framework suggests the following order:

$$
\left| \mathcal { H } \to \mathbb { Z } _ { \mathrm { l o s t / k e p t } } \to \mathcal { K } \to p ( X | Y ) \to \delta ^ { * } \to ( F _ { \theta } , \mathcal { D } ) . \right|\tag{23}
$$

That is: (1) model acquisition; (2) identify retained and lost information; (3) state legitimate knowledge sources; (4) characterize ambiguity and uncertainty; (5) choose the estimator and loss for the intended use; and only then (6) choose architecture and training data. A physics-aware learned formulation, for example, can combine these sources:

$$
\begin{array} { r l } & { \widehat { X } = \arg \underset { X } { \operatorname* { m i n } } D ( Y , \mathcal { H } ( X ) ) + \lambda _ { a } R _ { a } ( X ) } \\ & { ~ + \lambda _ { e } R _ { e } ( X ) + \lambda _ { i } R _ { i } ( X ; Y ) . } \end{array}\tag{24}
$$

This is not a prescription that every method must be modelbased. It is a requirement that omitted physics and imported priors be conscious, testable choices.

## A. Stage 1: define target, use, and acquisition

Specify X, Y, the intended user or downstream task, and a forward-process family. Distinguish controlled variables, nuisance variables, and unknown mechanisms. Where the exact camera pipeline is unavailable, give a justified abstraction and describe what it excludes. A useful first-stage output is not necessarily one equation; it can be a causal diagram, a calibrated simulator, or a bounded family of plausible processes.

## B. Stage 2: audit retained and lost information

Identify sampling limits, null spaces, clipping, quantization, noise floors, ambiguity between latent factors, and sensitivity to model error. This stage determines which desired outputs can be evidence-supported. It also reveals where collecting a better measurement may be more effective than constructing a larger estimator.

## C. Stage 3: inventory legitimate knowledge

List the information sources in K and justify their relevance. External data can provide strong natural-image or semantic priors; internal recurrence can adapt to the test image; physics can constrain consistency; analytic assumptions can improve stability; and architecture can bias the solution. The inventory should also identify prohibited information, such as contentspecific priors in a forensic setting or inaccessible metadata in deployment.

## D. Stage 4: characterize ambiguity and shift

Describe $p ( X | Y )$ conceptually even when it cannot be calculated. Ask whether it is narrow, multimodal, sensordependent, or preference-dependent. Define plausible shifts in $p ( Y | X ) , p ( X )$ , target construction, and user loss. Decide which uncertainty should be estimated and what out-of-domain behaviour is acceptable.

## E. Stage 5: choose the decision and evidence

Choose the loss or utility from the application. A mean estimate, perceptual mode, posterior sample, and task-optimized rendering answer different questions. At the same time, state the evidence required for the intended claim: matched accuracy, cross-domain robustness, human preference, physical consistency, safety, latency, or calibration. Designing evaluation at this point reduces the temptation to select favourable metrics after observing results.

## F. Stage 6: choose data, model, and optimization

Only after Stages 1–5 should the implementation be selected. Model capacity, pretraining, inductive bias, and compute should match the uncertainty and constraints of the problem. Training data should sample the validated forward family and relevant image distribution, while held-out tests should include purposeful shifts. This stage is where modern AI can be used most powerfully because its role is now specified.

## G. The workflow is iterative, not linear

Experiments will expose deficiencies in the acquisition family or target definition. The workflow should therefore loop: failures update the problem model; new measurements reduce ambiguity; and deployment evidence expands or contracts the validated region. What it prevents is not iteration but unexamined inheritance—the assumption that an established dataset, loss, and architecture collectively define the scientific problem.

## XI. FROM WORKFLOW TO RESEARCH PRACTICE

The preceding workflow is not only a conceptual ordering; it changes how the two case studies should be formulated, solved, and evaluated. The essential contrast is between starting with a model family and asking how well it performs on an established benchmark, and starting with the deployed information problem and asking which model, data, and objective it requires. The for mer sequence can encourage repeated substitution of the newest architecture into an inherited pipeline, followed by extensive tuning for a narrow score. The latter makes the scientific object, assumptions, and evidence explicit before optimization begins. Recent deep models contain major advances in representation and computation, but the workflow helps determine precisely what problem those advances solve and whether a reported gain survives outside the conditions under which it was selected.

## A. Super-Resolution Reconstructed from the Workflow

Stage 1: model acquisition. Rather than assume bicubic downsampling, specify a family $\mathcal { H } _ { \mathrm { S R } }$ containing plausible optics, pixel integration, sensor sampling, aliasing, mosaicing, demosaicing, noise, compression, sharpening, and spatially varying processing. Known parameters should be calibrated; unknown parameters should be estimated or represented by a distribution. The first design question is whether the intended resolution requires additional evidence, such as subpixelshifted burst frames, rather than a more powerful singleimage prior. Stage 2: identify information loss. Determine the passband, aliased components, null space, saturation, and noise-limited frequencies. This analysis states which detail is measurement-supported, which can become observable through complementary acquisition, and which cannot be uniquely recovered.

Stages 3–4: identify priors and ambiguity. State whether missing detail comes from analytic regularity, external HR images, internal recurrence, semantic foundation-model knowledge, or architecture bias. Evaluate how strongly the output changes when these priors or degradation assumptions change. Stage 5: choose the decision rule. MSE is appropriate for a conditional mean; perceptual or adversarial objectives select visually preferred solutions; diffusion posterior sampling represents multiple plausible HR images. The choice should follow the use case, not leaderboard convention. Stage 6: choose architecture and data. Only now decide whether a CNN, transformer, unfolding network, GAN, or diffusion model is the most effective implementation, and construct training pairs from the validated acquisition family.

This order differs from much benchmark SR, where a fixed degradation and loss are inherited before architecture design. SRCNN and SwinIR demonstrate progressively more capable mappings on synthetic benchmark distributions [6], [8]; Real-ESRGAN responds with a richer high-order synthetic degradation process [33]; RealSR and DRealSR replace synthetic LR with registered captures [23], [24]; and diffusion restoration adds powerful priors and posterior sampling [15]. By contrast, burst methods change the evidence by acquiring complementary samples [18], [19], while null-space and unfolding methods make the information decomposition or measurement model constrain the estimator [20], [21]. Each changes a different workflow component. A problem-first comparison should not pool their scores as though they estimate one $p ( X | Y )$ : it should state whether the advance changes acquisition, the assumed forward process, the observable reconstruction, the prior-supported null space, or only the implementation, and whether the target is bicubic SR, capture-rig transfer, blind camera SR, or plausible generative upsampling.

A convincing future SR study should therefore report: matched benchmark performance; robustness across calibrated degradation families; results on independently captured real LR/HR pairs where possible; data-consistency or re-degradation error; sensitivity to blur and noise assumptions; and either uncertainty or multiple solutions in strongly ambiguous regions.

Such evidence distinguishes improvement in the estimator from increased confidence in a learned prior.

## B. Low-Light Enhancement Reconstructed from the Workflow

Stage 1 should begin with illumination, reflectance, scene radiance, exposure, sensor noise, clipping, quantization, colour processing, and display rendering, rather than with a gammadarkened normal image. Stage 2 then identifies the distinct losses: photon-limited shadow information, clipped highlights, colour uncertainty, spatially nonuniform illumination, and compression or processing artifacts. This reveals why low light is simultaneously a denoising, inverse-rendering, colour, and high-dynamic-range problem.

At Stages 3–4, the method should declare whether it assumes Retinex-type decomposition, learned normal-light statistics, semantic plausibility, repeated structure, or a calibrated camera pipeline. The inverse-rendering view in [25] uses a simplified scene representation and canonical re-rendering, thereby changing the formulation from “invert an artificial darkening curve” to “estimate and correct the causes of image formation.” Ambiguity must still be acknowledged: saturated highlights and noise-floor shadows do not have unique inverses.

At Stage 5, the desired output must be defined. It may be a faithful estimate of a canonical rendering, a visually pleasing photograph, an HDR representation, or an image optimized for downstream recognition. These objectives imply different losses and acceptable uses of prior-supported detail. At Stage 6, architecture follows: a transformer may supply long-range illumination context, a diffusion model may represent multimodal uncertainty, and a compact inverse-rendering network may provide a more interpretable and data-efficient solution.

Recent LLIE research illustrates both progress and the remaining risk. Retinexformer combines an illumination model with a transformer [29]; HVI improves colour and intensity representation [34]; and challenge systems combine highcapacity models and optimize benchmark quality [35]. Yet a model trained and tested on LOL estimates the LOL paired exposure conditional, a SID model estimates a sensor-specific raw short-to-long exposure conditional, and SICE supplies a selected fusion rendering. High scores across rows of a benchmark table do not erase these distinctions. Under the workflow, researchers should report within-benchmark results, cross-dataset transfer (e.g., LOL-v1↔LOL-v2-Real and synthetic→real), cross-sensor raw testing for SID-type methods, and evaluation on dynamic scenes and independently captured real images. Physical or decomposition consistency and downstream vision should complement PSNR, SSIM and perceptual quality.

## C. Guidelines for Future Research

The two cases suggest a reusable protocol:

1) Define the deployed task before the benchmark. State the scene, sensor, processing pipeline, output representation, and intended use.

2) Write an observation model and its uncertainty. Distinguish known physics, estimated parameters, nuisance variables, and deliberately omitted effects.

3) Audit information loss. Identify null spaces, clipping, aliasing, quantization, noise floors, and other many-toone operations.

4) Declare every source of missing information. Separate measurement evidence from analytic, internal, external, semantic, and architectural priors.

5) Match the estimator to the use. Explain whether the output is a conditional mean, mode, perceptual decision, or posterior sample, and why.

6) Validate data and audit the benchmark. Document reference construction and changed latent variables; compare simulated and real conditionals; reserve independent real tests and state excluded conditions.

7) Evaluate beyond the matched benchmark. Include cross-dataset transfer, parameter shift, sensor or pipeline shift, re-degradation consistency, prior sensitivity, uncertainty, and task-relevant outcomes.

8) Locate novelty accurately. Distinguish advances in problem formulation, information source, objective, representation, inference, and architecture.

9) Make comparative claims reproducible. Release or fully specify preprocessing, training and selection procedures, hyperparameters, compute, and evaluation code; report repeated-run variation where it can affect the ranking; and compare under matched data, supervision, parameter, and computational conditions.

10) Treat SOTA as a scoped result, not a scientific conclusion. State the dataset, protocol, metric, resource budget, and statistical uncertainty under which superiority holds, then separately explain what new understanding or deployment capability the result establishes.

## D. Progress Under Irreducible Mismatch

Mismatch is not an anomaly that a sufficiently large dataset will finally remove. Cameras, optics, compression pipelines, illumination, content, user preference, and deployment objectives continue to change; even a captured “real-world” dataset is a finite sample from one protocol. The realistic goal is therefore not distribution identity, but controlled uncertainty about the gap and demonstrable performance over a useful region of possible worlds.

First, replace a single assumed forward model by a family $\{ \mathcal { H } _ { \eta } : \eta \in \Omega \}$ whose parameters describe plausible blur, noise, sampling, exposure, response, clipping, and processing. Training should cover physically meaningful combinations rather than arbitrary augmentation, and evaluation should include held-out combinations and deliberately shifted domains. Conceptually, a robust estimator minimizes risk over an uncertainty set Q around the available distributions,

$$
\operatorname* { m i n } _ { F } \operatorname* { s u p } _ { q \in { \mathcal { Q } } } \mathbb { E } _ { q } [ L ( F ( Y ) , X ) ] ,
$$

or reports both average and worst-group performance rather than only the mean on one benchmark. The purpose is not to guarantee every possible deployment, but to state the region for which evidence exists.

Second, design systems to diagnose and adapt. At test time, degradation estimation, metadata, self-supervision, internal image statistics, or a small amount of deployment data can update the forward model or estimator. Adaptation must be constrained by measurement consistency so that a model does not merely become more confident in its prior. When the input lies outside the validated region, the system should expose low confidence, multiple plausible outputs, or a refusal to assert unsupported detail.

Third, replace the single leaderboard by a benchmark portfolio: matched synthetic tests for controlled comparison; captured paired tests for specific devices and protocols; crossdataset and cross-sensor transfer; unpaired in-the-wild inputs; physical consistency tests; and downstream-task or human evaluation appropriate to the application. No component is sufficient alone. A leaderboard remains useful as a controlled instrument, but the best-tuned entry is not automatically the best explanation, the most reproducible method, or the most reliable deployed system. Progress is credible when a method improves the intended target without a hidden collapse under these complementary tests, and when the improvement remains meaningful under transparent, matched experimental conditions.

Finally, treat deployment as part of the scientific loop. Record failure modes, shifts in sensors or content, and calibration error; add representative cases to evaluation without discarding earlier tests; and distinguish a general method improvement from adaptation to a newly expanded benchmark. Claims should be scoped—for example, “robust across the tested camera/degradation family” rather than “solves real-world SR.” In this way, irreducible mismatch becomes manageable: not by pretending that train and real distributions coincide, but through explicit model uncertainty, stress testing, adaptation, calibrated outputs, and cumulative evidence.

## XII. IMPLICATIONS FOR RESEARCH TRAINING AND PUBLICATION

## A. How students should enter the field

A new researcher should not be required to master every classical method before using a neural network, nor should education delay engagement with current technology. The aim is integration rather than nostalgia. Researchers should learn the enduring questions alongside modern tools: image formation, sampling, convolution, noise, dynamic range, colour, ill-posedness, regularization, statistical estimation, uncertainty, decision objectives, and evaluation. Historical methods are valuable not as museum pieces but because their assumptions are visible. They demonstrate how a solution follows from a problem description and how failure follows when assumptions break.

We propose a problem-first training sequence:

1) Observe the physical process. Examine how optics, illumination, exposure, sensing, sampling, noise, and processing form the image used by the algorithm.

2) Analyse the inference problem. Identify retained information, lost information, confounded variables, nonidentifiability, and sensitivity to model error.

3) Reconstruct a transparent baseline. Implement or reproduce a classical or model-based approach whose assumptions can be inspected.

4) Interrogate the benchmark. Reconstruct how inputs and references were created, what conditional task they define, and which real conditions they exclude.

5) Interpret a modern model. Trace its output to measurement evidence, physics, analytic assumptions, internal statistics, external data, and architecture bias.

6) Create and diagnose mismatch. Change one acquisition, sensor, content, or target factor and study why performance changes rather than merely recording that it changes.

7) Design the solution from the diagnosis. Select a CNN, transformer, diffusion model, hybrid method, or different measurement only after identifying the limitation that the choice is intended to overcome.

8) Make a proportionate claim. Match the conclusion to the validated conditions and report uncertainty, failure, and prior dependence where relevant.

This sequence changes the meaning of technical competence. A capable researcher should be able not only to improve a model, but also to decide whether the model is solving the right problem; not only to use a dataset, but to explain the task its construction creates; and not only to report a metric, but to state what scientific conclusion that metric supports. Architecture research remains important, but becomes more original because it is directed toward a diagnosed limitation rather than a fashionable component.

## B. Responsibilities of supervisors and research leaders

The educational change cannot be delegated to students alone. Supervisors and research leaders determine whether projects begin with a problem description or a model name. They can require every proposal to state the acquisition process, target, ambiguity, prior sources, deployment conditions, and falsifiable claim before approving architecture development. Group discussions can examine failure cases and mismatches with the same seriousness as average benchmark improvements.

Recruitment and mentoring can likewise assess whether candidates understand why a method works, what information it uses, and when it should fail—not only whether they can reproduce a current model. Industry and national laboratories have a particular interest in this distinction because deployment rapidly exposes assumptions hidden by academic benchmarks. Strong fundamentals therefore improve employability and technology transfer as well as scholarship.

At community level, tutorials, doctoral courses, challenges, and review criteria should connect modern models back to acquisition, inverse problems, statistical decisions, and evidence. The objective is not to create a gatekeeping canon or to privilege one generation of methods. It is to ensure that the next generation inherits the accumulated scientific understanding of image processing while gaining the full benefit of modern AI.

## C. Questions for authors

Before submission, authors can ask:

1) Is the desired target defined independently of the dataset filename?

2) Which parts of the forward process are known, simulated, estimated, or ignored?

3) Which output details are measurement-supported and which are prior-supported?

4) Are baselines matched in data, supervision, compute, tuning, and test-time processing?

5) Is the claimed gain larger than relevant run-to-run or protocol variation?

6) What happens under an independently chosen dataset, sensor, or degradation shift?

7) Can another group reconstruct the data and evaluation pipeline from the report?

8) Does the conclusion state exactly what has improved and where it has been validated?

## D. Questions for reviewers, editors, and benchmark organizers

Reviewers should reward a well-supported problem or diagnostic contribution even when its headline metric is not the largest. They should distinguish absence of a leaderboard maximum from absence of novelty, and demand stronger evidence when a paper makes a broad real-world claim from a narrow benchmark. Conversely, a problem-first paper should not receive exemption from quantitative comparison; its evaluation should follow from its claim.

Editors, area chairs, and programme committees shape culture through reviewer guidance and selection criteria. They can state explicitly that novelty may reside in problem formulation, measurement, data, evaluation, diagnosis, or validated deployment—not only architecture. They can invite reviewers to assess whether evidence matches the claim rather than treating one benchmark maximum as a universal requirement. This does not lower standards; it makes the standards more scientifically discriminating.

Benchmark organizers can assist by documenting capture and target construction, preserving hidden tests, reporting uncertainty, separating tracks with different data and resource rules, and introducing shifted or refreshed evaluation. Leaderboards could show efficiency, calibration, worst-group performance, and cross-domain results alongside one average score. Such changes would make competition a better instrument of science.

## XIII. TOWARD A PROBLEM-FIRST RESEARCH CULTURE

## A. Culture changes through everyday decisions

Research culture is not changed by a manifesto alone. It changes through the questions supervisors ask, the projects institutions fund, the evidence reviewers request, the papers editors select, the dimensions leaderboards display, and the capabilities employers value. No single group can make the transition independently, and no group should be blamed for responding to existing incentives. The opportunity is collective: make the scientifically stronger path also a recognized and practical path to publication, recognition, and impact.

## B. Broadening novelty without lowering standards

A frequent concern is that broadening the definition of contribution could make evaluation subjective or excuse weak performance. The opposite should be the aim. A problemformulation paper must show that the existing formulation is inadequate and that the new one changes understanding or capability. A dataset paper must establish what new variation it measures and document its construction. A diagnostic paper must isolate causes and support reproducible conclusions. A robustness paper must define meaningful shifts. An architecture paper must demonstrate that its design, rather than unmatched resources, produces the claimed gain.

Thus, different contributions require different but equally rigorous evidence. The unifying standard is the alignment of problem, claim, and evidence. This is more demanding than requiring every paper to compete on the same table, because it asks reviewers to examine what has actually been learned.

## C. Changing the language of progress

Language helps create expectations. Instead of saying that a method “solves real-world super-resolution,” authors can state the tested camera and degradation family. Instead of calling a benchmark result simply “SOTA,” they can specify the dataset, protocol, resources, and uncertainty under which it is superior. Instead of describing generated detail as recovered, they can distinguish measurement-supported and prior-supported content. Such qualification does not weaken a paper. It makes its contribution more durable because later work can identify precisely what has been established and what remains open.

The community can likewise celebrate multiple forms of progress: a better score under matched conditions; a method that transfers across sensors; a dataset that exposes a hidden mismatch; a physical model that explains failure; a calibrated system that knows when it is uncertain; or a negative result that prevents repeated error. A mature science needs all of these.

## D. A gradual and inclusive transition

The proposed change should be evolutionary rather than adversarial. Existing benchmarks, challenges, and model families embody substantial collective effort and should be improved, not dismissed. Young researchers should not be disadvantaged for having trained within the prevailing system. Instead, courses, tutorials, review forms, benchmark tracks, and mentoring can progressively add problem-first questions and evidence. The transition succeeds if it expands researchers’ capabilities while preserving the openness, speed, and empirical energy that modern AI has brought to image processing.

## XIV. SCOPE AND LIMITATIONS

This paper is a methodological framework rather than an empirical meta-analysis of all image-processing leaderboards.

TABLE IV  
CONTRIBUTIONS TO A PROBLEM-FIRST IMAGE-PROCESSING CULTURE.
<table><tr><td>Community actor</td><td>Constructive practice</td><td>Cultural signal</td></tr><tr><td>Students and researchers</td><td>Begin with target, acquisition, information loss, assumptions, and use; treat architec- ture as a justified design decision</td><td>Understanding the problem is part of technical excellence</td></tr><tr><td>Supervisors and re- search leaders</td><td>Require a problem statement and falsifiable claim before model development; discuss failures and mismatch, not only average</td><td>Careful formulation and diagnosis are legitimate research outputs</td></tr><tr><td>Authors</td><td>scores Match claims to evidence; disclose re- sources and tuning; report robustness, un- certainty, and limitations</td><td>Credibility matters more than rhetorical breadth</td></tr><tr><td>Reviewers</td><td>Evaluate whether the method solves the stated problem and whether comparisons are fair; recognize non-architectural nov-</td><td>The highest familiar-benchmark score is not the only form of contribution</td></tr><tr><td>Editors and programme committees</td><td>elty Broaden scope and reviewer guidance; uation, and diagnostic papers</td><td>Publication rewards cumulative knowl- welcome problem, data, measurement, eval- edge, not only leaderboard movement</td></tr><tr><td>Benchmark organizers</td><td>Document construction; refresh tests; in- clude shifts, efficiency, uncertainty, and</td><td>Benchmarks are instruments for inquiry rather than definitions of reality</td></tr><tr><td>Institutions and fun- ders</td><td>worst-group results Support data collection, replication, long- horizon problem study, shared infrastruc-</td><td>Slow foundational work is a valued in- vestment</td></tr><tr><td>Industry and laborato- ries</td><td>ture, and negative findings Share deployment failure modes where possible; value physical and inferential understanding in recruitment</td><td>Real-world reliability requires more than benchmark fluency</td></tr></table>

SR and LLIE were selected because both show how an artificial benchmark construction can be mistaken for the real physical problem. In SR, digitally downsampling a high-quality image can obscure insufficient spatial acquisition, aliasing, limited measurement quality, and ambiguity about the missing detail. In LLIE, artificially darkening a normally exposed image can obscure dynamic-range limitations, photon-limited SNR, and inverse-rendering ambiguity. Other areas—denoising, deblurring, dehazing, compression restoration, computational photography, medical imaging, remote sensing, and inverse graphics—require domain-specific acquisition and decision analysis.

The discussion of research culture is similarly a reasoned position informed by recurring professional experience, not a sociological survey of the community. Practices vary across subfields, institutions, and laboratories, and many researchers already combine strong fundamentals with modern learning. The purpose is to articulate a recognizable structural risk and propose constructive changes that can be debated, tested, and refined.

Nor do explicit physics and broad testing eliminate mismatch. Forward models remain approximations, human preference changes, and deployment can reveal unforeseen variables. Uncertainty estimates may themselves be miscalibrated under shift. The framework therefore cannot guarantee universal validity. Its aim is to make the chain of inference inspectable, the evidence proportional to the claim, and future correction cumulative.

The framework also does not prescribe one balance between fidelity and generation. A creative application may legitimately prefer a plausible, aesthetically strong image to a faithful reconstruction. A scientific application may prohibit unsupported detail. Both can be excellent image processing when the target and evidence are stated; confusion arises when success under one objective is claimed as success under the other.

## XV. CONCLUSION

The protocol strengthens, rather than opposes, modern AI: architecture and scale follow a defined inferential target. Its broader message is that the future of the field depends on how future researchers are educated. They should learn image processing from the problem outward—image formation, information loss, assumptions, ambiguity, and decision—rather than from the currently fashionable model inward. This requires patience: modelling acquisition, constructing meaningful data, analysing failure, and validating transfer may take longer than fine-tuning another architecture, but these are often the activities through which durable scientific progress occurs.

Across SR and LLIE, the central lesson is the same. An artificial transformation can create an image with a similar appearance without reproducing the physical process that caused the real problem. Digitally downsampling a highquality image is not the same as failing to capture sufficient spatial detail through a real optical–sensor system. Artificially darkening a normally exposed image is not the same as recording a scene with insufficient dynamic range and low

SNR. Reliable advances therefore require validated acquisition models, clearly identified priors, appropriate objectives, reproducible comparisons, cross-domain evidence, adaptation, and honest reporting of uncertainty. A benchmark gain is valuable evidence within its stated conditions. Cumulative technical advancement requires the larger step of showing that the gain reflects better acquisition, better problem formulation, justified use of information, more reliable inference, or demonstrable capability on the intended imaging problem.

Rethinking image processing is therefore a scientific, educational, and cultural project. The community must continue to develop powerful models, but it must also develop researchers able to question the tasks those models inherit, understand the physical and statistical foundations of the problem, and design evidence capable of distinguishing benchmark optimization from real advancement. Achieving this does not require rejecting current practice or assigning blame. It requires researchers, supervisors, reviewers, editors, benchmark organizers, institutions, and industry to broaden what they recognize and reward as progress.

The desired culture combines the best qualities of modern AI research—openness, rapid experimentation, shared code, empirical testing, and powerful learned models—with the enduring disciplines of image science: careful observation, physical modelling, identifiability analysis, justified assumptions, uncertainty, falsifiable claims, and patient validation. The central question should not be only “which model wins?” but first “what is the problem, what information can solve it, and what evidence would show that we have genuinely advanced?”

## ACKNOWLEDGMENT

OpenAI’s ChatGPT was used during the preparation of this manuscript to assist with language revision, manuscript organisation, drafting and refining alternative formulations, and literature-search support throughout the paper. The author developed the central ideas, scientific arguments, interpretations, and conclusions; independently reviewed and revised all AIassisted text; verified the technical content and cited sources; and takes full responsibility for the final manuscript.

## REFERENCES

[1] N. Wiener, Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press, 1949.

[2] A. N. Tikhonov, “Solution of incorrectly formulated problems and the regularization method,” Soviet Mathematics Doklady, vol. 4, pp. 1035– 1038, 1963.

[3] G. Qiu, “A progressively predictive image pyramid for efficient lossless coding,” IEEE Transactions on Image Processing, vol. 8, no. 1, pp. 109–115, 1999.

[4] W. T. Freeman, T. R. Jones, and E. C. Pasztor, “Example-based superresolution,” IEEE Computer Graphics and Applications, vol. 22, no. 2, pp. 56–65, 2002.

[5] M. Aharon, M. Elad, and A. Bruckstein, “K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation,” IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4311–4322, 2006.

[6] C. Dong, C. C. Loy, K. He, and X. Tang, “Learning a deep convolutional network for image super-resolution,” in European Conference on Computer Vision, 2014, pp. 184–199.

[7] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: Residual learning of deep CNN for image denoising,” IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142–3155, 2017.

[8] J. Liang, J. Cao, G. Sun, K. Zhang, L. V. Gool, and R. Timofte, “SwinIR: Image restoration using swin transformer,” in IEEE/CVF International Conference on Computer Vision Workshops, 2021, pp. 1833–1844.

[9] A. Shocher, N. Cohen, and M. Irani, “Zero-shot super-resolution using deep internal learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 3118–3126.

[10] D. Ulyanov, A. Vedaldi, and V. Lempitsky, “Deep image prior,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 9446–9454.

[11] Y. Blau and T. Michaeli, “The perception-distortion tradeoff,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6228–6237.

[12] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.

[13] S. V. Venkatakrishnan, C. A. Bouman, and B. Wohlberg, “Plug-and-play priors for model based reconstruction,” in IEEE Global Conference on Signal and Information Processing, 2013, pp. 945–948.

[14] Y. Romano, M. Elad, and P. Milanfar, “The little engine that could: Regularization by denoising (RED),” SIAM Journal on Imaging Sciences, vol. 10, no. 4, pp. 1804–1844, 2017.

[15] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 23 593–23 606.

[16] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do ImageNet classifiers generalize to ImageNet?” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97, 2019, pp. 5389–5400.

[17] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Lariviere, A. Beygelzimer, F. d’Alche Buc, E. Fox, and H. Larochelle, “Improving reproducibility in machine learning research: A report from the NeurIPS 2019 repro ducibility program,” Journal of Machine Learning Research, vol. 22, no. 164, pp. 1–20, 2021.

[18] B. Wronski, I. Garcia-Dorado, M. Ernst, D. Kelly, M. Krainin, C.- K. Liang, M. Levoy, and P. Milanfar, “Handheld multi-frame superresolution,” ACM Transactions on Graphics, vol. 38, no. 4, 2019.

[19] G. Bhat, M. Danelljan, F. Yu, L. V. Gool, and R. Timofte, “Deep burst super-resolution,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 9209–9218.

[20] Y. Wang, Y. Hu, J. Yu, and J. Zhang, “GAN prior based null-space learning for consistent super-resolution,” in AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 2724–2732.

[21] K. Zhang, L. V. Gool, and R. Timofte, “Deep unfolding network for image super-resolution,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 3217–3226.

[22] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017, pp. 126–135.

[23] J. Cai, H. Zeng, H. Yong, Z. Cao, and L. Zhang, “Toward real-world single image super-resolution: A new benchmark and a new model,” in IEEE/CVF International Conference on Computer Vision, 2019, pp. 3086–3095.

[24] P. Wei, Z. Xie, H. Lu, Z. Zhan, Q. Ye, W. Zuo, and L. Lin, “Component divide-and-conquer for real-world image super-resolution,” in European Conference on Computer Vision, 2020, pp. 101–117.

[25] W. Wu, J. Pang, S. Gao, P. Zhang, Z. Chen, and G. Qiu, “Low-light image enhancement through learning a simplified inverse rendering model,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 10, pp. 9608–9622, Oct. 2025.

[26] C. Chen, Q. Chen, J. Xu, and V. Koltun, “Learning to see in the dark,” in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3291–3300.

[27] X. Xu, R. Wang, C.-W. Fu, and J. Jia, “SNR-aware low-light image enhancement,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17 714–17 724.

[28] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” in British Machine Vision Conference, 2018.

[29] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang, “Retinexformer: One-stage retinex-based transformer for low-light image enhancement,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 12 504–12 513.

[30] T. Plotz and S. Roth, “Benchmarking denoising algorithms with real ¨ photographs,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 1586–1595.

[31] W. Yang, W. Wang, H. Huang, S. Wang, and J. Liu, “Sparse gradient regularized deep retinex network for robust low-light image enhancement,” IEEE Transactions on Image Processing, vol. 30, pp. 2072–2086, 2021.

[32] J. Cai, S. Gu, and L. Zhang, “Learning a deep single image contrast enhancer from multi-exposure images,” IEEE Transactions on Image Processing, vol. 27, no. 4, pp. 2049–2062, 2018.

[33] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data,” in IEEE/CVF International Conference on Computer Vision Workshops, 2021, pp. 1905– 1914.

[34] Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Mei, Z. Wang, J. Sun, and Y. Zhang, “HVI: A new color space for low-light image enhancement,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5678–5687.

[35] X. Liu, Z. Wu, A. Li, F.-A. Vasluianu, Y. Zhang, S. Gu, L. Zhang, C. Zhu, and R. Timofte, “NTIRE 2024 challenge on low light image enhancement: Methods and results,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024.