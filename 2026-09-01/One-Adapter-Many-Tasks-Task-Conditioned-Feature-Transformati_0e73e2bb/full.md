# One Adapter, Many Tasks: Task-Conditioned Feature Transformations for Continual Learning

Yunxiang Fu Meng Lou Yizhou Yu School of Computing and Data Science The University of Hong Kong

## Abstract

Class-incremental learning (CIL) requires a model to incrementally learn tasks that contain new classes without accessing earlier training data while preserving the ability to recognize all seen classes. Recently, pretrained-model-based approaches have become prevalent by adapting a frozen backbone with additional lightweight trainable modules. Existing methods, however, exhibit limitations: task-specific adapters learn explicit per-task representations but are parameter- and computationinefficient, while LoRA-based merging methods combine per-task LoRA parameters into a single model whose static aggregated weights cause representation interference during inference. To address these problems, we present FACET: task-conditioned FeAture transformation with ConditionEd feature consisTency, achieving excellent parameter efficiency while producing highly discriminative features during inference. When continually trained on a task sequence, FACET learns a single shared adapter that employs a dynamic task-conditioned feature transformation, shaping the overall feature distribution of the adapter into a mixture of overlap-reduced task-specific components. On the other hand, we propose an efficient replay-free task-conditioned feature consistency loss, aiming to mitigate catastrophic forgetting of the learned mixture distribution in the adapter’s feature space. Even when maintaining only a single adapter, FACET demonstrates robust scalability. On both very long task sequences (e.g., 200 tasks) and standard short task sequences (e.g., 20 tasks), our method achieves superior performance while using significantly fewer trainable parameters and GFLOPs. The code will be made open source upon acceptance.

## 1 Introduction

In real-world applications, training data may arrive sequentially from continuously changing environments, which requires a learning system to acquire new concepts while retaining previous knowledge [15, 25]. However, standard deep neural networks generally suffer from catastrophic forgetting: when trained on new data, their performance on previously learned tasks can severely degrade [17, 13, 27]. Continual learning (CL) aims to address this problem by enabling models to learn from continuous data streams without forgetting past knowledge. Among the many CL settings, class-incremental learning (CIL) is one of the most challenging because the model must learn new classes over time and, at test time, discriminate among all classes seen so far [23].

Traditional continual learning methods address catastrophic forgetting through regularization [13, 17], replay [26, 19],architectural expansion [27, 38], and parameter isolation/sub-networks [2, 21]. More recently, pretrained-model (PTM) based approaches have become prevalent for CIL, as large pretrained backbones provide general-purpose, powerful representations, and require relatively few additional parameters when adapted to new tasks [35, 28, 42, 43, 33, 30, 12]. A common strategy attaches task-specific learnable adapters to a frozen vision backbone [42, 33, 12, 32], which produces explicit task-specific feature perturbations but requires a separate adapter per task, resulting in parameter and computation inefficiency. Another popular solution trains separate LoRA modules [11] by task and merges all learned modules before inference, exemplified by InfLoRA [18] and SD-LoRA [36]. Although such methods improve parameter efficiency at inference time, an input is passed to the aggregated weights, which give rise to representation interference and diluted features that are suboptimal for classification.

![](images/42cbf95dec67979f9955fc45cf6f412433ea9608feb87092936f3593b72c97a2.jpg)  
FACET (Ours)

![](images/c9996ca54b73b9a96934c4dbed8268b37ff2fa48eb65b98ee71afbab8c28b043.jpg)  
SD-LoRA

![](images/bd0947e38cd25b4dc453269a91f81451f6cfc07789d20125d5681d425605ea8a.jpg)  
Figure 1: Comparison of feature distributions. (a) t-SNE of block 1 features, colored based on task. FACET shows well-separated task-specific distributions, while SD-LoRA’s are mixed. (b) Class-averaged silhouette score (higher values mean better intra-class compactness and inter-class separation) across blocks. FACET achieves higher scores than representative LoRA-based (SD-LoRA) and per-task adapter based (TUNA) methods, producing more class-discriminative features that are beneficial for classification.

To address these issues, we present FACET, a novel PTM-based CIL method that achieves parameter efficiency by using a single continually optimized adapter with an explicit task-conditioning mechanism. The core idea is to obtain task-dependent features by applying a dynamic task-conditioned transformation to the adapter’s output. Specifically, we predict a task-dependent context vector for the input, and map it to a matrix that performs task-dependent channel-wise scaling and cross-channel mixing in the latent space of the adapter. This substantially improves the separability of task-specific feature distributions in the adapter’s feature space, thereby reducing representation interference.

However, training the shared adapter on a new task might distort the conditional feature distributions of previous tasks and make them less discriminative. To this end, the adapter is required to preserve the conditional feature distribution of any earlier task. That is, given any training sample from the new task, the adapter should be able to reproduce its feature representation as if conditioned on any earlier task. This goal is achieved via a simple task-conditioned feature consistency objective.

We compare the representation quality of FACET with representative methods in Fig. 1. In early adapter blocks, FACET’s task-specific feature distributions are widely separated, while those of SD-LoRA are completely mixed (Fig. 1 (a)). As CIL ultimately requires discrimination among classes, Fig. 1 (b) compares the block-wise class-averaged Silhouette score <sup>1</sup>. A higher score suggests more class-discriminative features. Driven by dynamic task conditioning, FACET achieves a significantly higher score in the last block, which indicates that features have transitioned from task-scaffolded to class-discriminative as depth increases.

We evaluate FACET on standard class-incremental learning benchmarks. FACET achieves the highest final accuracy on ImageNet-R, ImageNet-A, CIFAR-100, and ObjectNet. Compared with SD-LoRA [36], FACET improves the final accuracy from 75.26% to 79.97% on ImageNet-R with 20 tasks and from 55.96% to 64.87% on ImageNet-A. Surprisingly, on OmniBenchmark-1K [20] with 200 tasks, FACET significantly improves over MIN [12] and TUNA [33] by 3.45% and 6.81% in final accuracy, respectively. FACET is also 3.85× faster and reduces GFLOPs by 62.69% relative to MIN (Table 3). Note that MIN and TUNA learn a task-specific adapter for every task.

Our main contributions are as follows:

A novel single-adapter design with task-conditioned transformations. We introduce a single, continually trained adapter that incorporates a dynamic task-conditioned transformation. This design achieves parameter efficiency and improves the separability of task-specific feature distributions, thus reducing representation interference.

![](images/96d12a0237aeff44abf39214de8806f932ad191eb1fc66ea5c8d0b22e7b3a782.jpg)  
Figure 2: Illustration of FACET. (a) Task-conditioned feature transformation within a single adapter. We predict the task context c and use it to generate a transformation matrix $M _ { c }$ to obtain taskdependent features. Activation functions are not shown for simplicity. (b) Task-conditioned feature consistency objective when training on a new task t with dataset $\mathcal { D } _ { t }$

Conditioned feature consistency. We propose a conditioned feature consistency loss that aligns the feature representation of the latest adapter with its frozen snapshot under earlier task conditions, effectively mitigating forgetting.

State-of-the-art results. FACET achieves new state-of-the-art experimental results on class-incremental learning benchmarks with 10 to 200 tasks.

## 2 Related Work

Pre-trained model-based Continual Learning. Early continual learning methods combat catastrophic forgetting through regularization [13, 39], knowledge distillation [17], replay [26, 19], param eter isolation/sub-networks [2, 21, 24, 1], and architectural expansion [27, 38, 37].

Based on the powerful generalization ability of pretrained models (PTMs), recent class-incremental learning (CIL) methods attach a small number of learnable parameters for each task. For example, Prompt-tuning approaches such as L2P [35], DualPrompt [34], and CODA-Prompt [28] store a pool of task-specific tokens and retrieve them at inference. Recent methods provide stronger learning capacity by incrementally appending task-specific adapters, i.e., EASE [42] builds task-specific subspaces by incrementally tuning adapters: TUNA [33], MOS [30], and MIN [12] assign a dedicated adapter to each task to prevent forgetting. However, these methods are parameter inefficient and computationally heavy as they must activate or aggregate all stored task-specific adapters at test time. In contrast, we use a single shared adapter with a dynamic task-conditioned feature transformation, producing separable task-specific feature distributions without maintaining per-task adapters.

Parameter Efficient CIL. Some works strive for a single set of parameters in PTM-based CIL. SLCA [40] and FeCAM [6] focus on classifier alignment and covariance-aware classifiers, respectively. Closer to our design, LoRA-based merging methods (InfLoRA [18], SD-LoRA [36]) and model merging techniques (MagMax [22], OPCM [31]) first train separate LoRA/adapters per task and then combine them into a single model during inference. These methods apply a static merged model during inference, where the aggregated weights may cause representation interference and dilute the task-specificity of feature maps. In contrast, our method directly trains a single adapter for all tasks with explicit task conditioning, eliminating such interference without per-task adapters.

## 3 Method

## 3.1 Problem Formulation

In class-incremental learning, a model learns from a data stream $\mathcal { D } = \{ \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } , . . . , \mathcal { D } _ { T } \}$ consisting of $T$ tasks arriving sequentially. Each task t provides a training set $\mathcal { D } _ { t } = \{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = 1 } ^ { n _ { t } }$ , where $\ v { x } _ { i } ^ { \top }$ is an image, $y _ { i } ^ { t } \in \mathcal { V } _ { t }$ is its class label, and $n _ { t }$ is the number of training samples. The label sets of different tasks are disjoint: $\mathcal { V } _ { s } \cap \mathcal { V } _ { t } = \emptyset$ for all $s \neq t$ . We follow the exemplar-free setting [42, 33].

When task t is learned, only the latest training data $\mathcal { D } _ { t }$ are accessible. We denote $\mathcal { V } _ { \mathrm { n e w } } = \mathcal { V } _ { t }$ as the set of new classes and $\textstyle \mathcal { Y } _ { \mathrm { o l d } } = \bigcup _ { s = 1 } ^ { t - 1 } \mathcal { Y } _ { s }$ as the set of previously seen classes. After being trained on task t, the model is evaluated on all classes seen so far, $\mathcal { V } _ { \mathrm { o l d } } \cup \mathcal { V } _ { \mathrm { n e w } }$

We decompose the model into a pretrained ViT backbone $\mathcal { F }$ , an adapter with parameters θ [4] inserted in parallel to the FFN in every transformer block, and a classifier $\mathbf { W } _ { t }$ that grows by appending a new class vector for each task. The backbone $\mathcal { F }$ is frozen throughout training. Only θ and $\mathbf { W } _ { t }$ are updated. The goal is to learn all T tasks from the data stream and correctly classify any test sample into one of the classes in $\textstyle \bigcup _ { t = 1 } ^ { T } { \mathcal { Y } } _ { t }$ without access to past data.

## 3.2 Single Adapter with Dynamic Task-Conditioned Transformations

As shown in Fig. 2 (a), a frozen pretrained Vision Transformer (ViT) [5] backbone $\mathcal { F }$ is equipped with adapter modules placed in parallel to the feed-forward network (FFN) in every transformer block. For an input x to the FFN, the output of the block o is

$$
o = { \mathrm { F F N } } ( x ) + { \mathrm { A d a p t e r } } ( x , c ) + x ,\tag{1}
$$

where c denotes a context vector that reflects the task-related information of x. By conditioning the adapter’s output explicitly on $c ,$ we obtain task-dependent features during inference while using only a single shared set of adapter weights for all tasks, which is both parameter- and computation-efficient.

The hidden layer of a standard adapter [4] operates in a latent space, which often has much lower dimensionality than the ViT backbone. To increase learning capacity without substantially increasing trainable parameters, we extend the standard adapter with one additional non-linear hidden layer. The revised adapter first computes a latent representation $z \in \mathbb { R } ^ { r }$ as follows,

$$
z = \sigma \big ( \sigma ( x W _ { \mathrm { d o w n } } ) W _ { \mathrm { m i d } } \big ) , \qquad W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { d \times r } , W _ { \mathrm { m i d } } \in \mathbb { R } ^ { r \times r } ,\tag{2}
$$

where $\sigma$ is the GELU [7] activation, d is the ViT’s feature dimensionality, and $r < d$ is the latent dimensionality of the adapter. This is common computation for all tasks. Next, before projecting the latent representation back to the ViT feature space, we apply a task-conditioned linear transformation to $z ,$

$$
z _ { c } = z M _ { c } ,\tag{3}
$$

where $M _ { c } \in \mathbb { R } ^ { r \times r }$ is a transformation matrix dynamically generated from the context vector c and $z _ { c }$ is the resulting task-dependent latent representation. Context vectors related to different tasks give rise to different matrix transformations. As a result, the same latent representation z can be transformed to different task-dependent latent features. The adapter output is obtained by projecting $z _ { c }$ up to the ViT feature space through the shared up-projection $\mathbf { \bar { \boldsymbol { W } } } _ { u p } \in \mathbf { \bar { \mathbb { R } } } ^ { r \times d }$

$$
\mathrm { A d a p t e r } ( x , c ) = ( z _ { c } ) W _ { u p } .\tag{4}
$$

Since the up-projection is linear, applying the task-conditioned transformation in the latent space is sufficient to make the final adapter output task-dependent, while all adapter weights remain shared across tasks.

Let us now discuss more details about context vectors and task-conditioned transformations. The context vector reflects information on the task identity of the given input. A simple option is to set it to be the task identity itself, i.e., $c \in \{ 0 , 1 \} ^ { K _ { \operatorname* { m a x } } }$ , where $K _ { \mathrm { m a x } }$ is a sufficiently large upper bound on the total number of tasks (ablation of $K _ { \mathrm { m a x } }$ is given in Appendix Table 11). During training, the task identity is known by counting the number of trained tasks at present, while lightweight task-wise classifiers predict the context vector during inference (Sec. 3.4).

Given a context vector, we construct the task-conditioned transformation $M _ { c }$ in a low-rank decomposition form to maintain parameter efficiency.

$$
\begin{array} { r } { M _ { c } = \mathrm { D i a g } ( \mathbf { s } ) + U V ^ { \top } , } \end{array}\tag{5}
$$

where $\mathbf { s } \in \mathbb { R } ^ { r }$ determines task-conditioned scaling, which independently rescales each feature dimension in a task-dependent manner, and $U , V \in \mathbb { R } ^ { r \times k }$ with $k \ll r$ implement cross-dimension mixing, which redistributes activations across dimensions so that different tasks produce different linear combinations of the latent feature dimensions. The above matrix components are directly mapped from the context vector via linear projections,

$$
\begin{array} { r } { \mathbf { s } = W _ { s } c , \qquad U = \mathrm { r e s h a p e } ( W _ { U } c ) , \qquad V = \mathrm { r e s h a p e } ( W _ { V } c ) , } \end{array}
$$

where $W _ { s } \in \mathbb { R } ^ { r \times K _ { \operatorname* { m a x } } }$ and $W _ { U } , W _ { V } \in \mathbb { R } ^ { r k \times K _ { \operatorname* { m a x } } }$ are learnable projection matrices, and reshape(·) converts a vector of length rk into a $r \times k$ matrix. Thus, a single shared adapter, without any task-specific weights, produces task-conditioned feature transformations and, in turn, task-dependent final representations entirely on the basis of the context vector. This is the core difference between our method and other LoRA-based merging methods.

To improve training stability, we initialize $W _ { s } , W _ { U } , W _ { V }$ to zero so that the adapter output is zero during the first training step. We also add small Gaussian noise to the context vector during training, aiming to improve robustness against imperfect task identity predictions during inference and softly enable backward knowledge transfer during training.

Remark As a single adapter is shared among all learned tasks and each task has its own feature distribution, the overall distribution of the adapter’s final output features can be viewed as a mixture of task-specific components. Task-conditioned feature transformations are capable of reducing overlap among these task-specific components by altering their location and shape. Our experiments confirm that introducing task-conditioned transformations better separates task-specific components (Fig. 1).

We also considered adding explicit losses to improve the separability of the mixture components, such as maximizing their cosine distance or maximizing the orthogonality of task-conditioned transformations. Nevertheless, this reduces the learning capacity of the adapter and leads to suboptimal results, as shown in our ablation studies (Table 5).

Using context signals that reflect task identity as conditioning has been previously studied in continual learning. For instance, using them as learnable context tokens concatenated to the input of the model [35, 34, 28], as a gating mechanism to select subnetworks [2, 1, 24], or to route inputs to task-specific adapters [33, 30]. Distinct from previous works, FACET applies context signals to transform the adapter’s latent representation, making task-specific feature distributions more separable, which is a direction that has not been explored.

## 3.3 Task-Conditioned Feature Consistency

Since the adapter weights are tuned on every new task, there is an increasing risk of gradually shifting the adapter’s latent feature distribution. This may lead to forgetting of the learned mixture distribution because the task-specific mixture components are dependent on the adapter weights. To this end, we introduce a simple task-conditioned feature consistency objective that preserves the learned task-specific mixture components without resorting to any historical data. This allows our adapter to produce reliable task-dependent features for continual learning. Specifically, after learning task $t - 1$ , we make a copy of the latest adapter weights, $\theta ^ { - }  \theta .$ , and keep the copy frozen when tuning the adapter on task t. For each mini-batch, we uniformly sample the index of a previous task $j \sim \operatorname { \bar { U n i f o r m } } \{ 1 , \dots , t - 1 \}$ and construct the corresponding context vector $c ^ { j } \in \mathbb { R } ^ { K _ { m a x } }$ . Then the training samples for task t pass through the following two paths:

Main path: a training sample $x \in \mathcal { D } _ { t }$ is passed to the ViT backbone equipped with the adapter (θ) currently being trained on task t and use $c ^ { j }$ as the context vector. The [CLS] token from the last transformer block is the main feature, and is denoted as $\mathrm { C L S } _ { j } ^ { \theta }$

Reference path: x is passed to the ViT backbone equipped with the frozen adapter snapshot $( \theta ^ { - } )$ and use $c ^ { j }$ as the context vector again. The [CLS] token from the last transformer block is taken as the reference feature, and is denoted as $\mathrm { C L S } _ { j } ^ { \theta ^ { - } }$

Evaluating both paths on the same input under the same task conditioning enables us to directly measure how much the latest adapter has drifted from the learned feature distribution of task $j$ . This does not require x to be a sample from task $j$ as the context vector $c ^ { j }$ drives the adapter to process x as a sample from task $j$ regardless of its true task identity. Our experiments demonstrate that this simple strategy can preserve the feature distribution of historical tasks without using their original data (Appendix D) and mitigates forgetting substantially (Table 4).

The training loss is the squared Euclidean distance between conditioned [CLS] tokens averaged uniformly over all previous tasks:

$$
\mathcal { L } _ { \mathrm { c o n } } = \mathbb { E } _ { j \sim \mathrm { U n i f o r m } \left\{ 1 , \dots , t - 1 \right\} } \mathbb { E } _ { x \sim \mathcal { D } _ { t } } \left[ \left\| \mathbf { C L S } _ { j } ^ { \theta ^ { - } } - \mathbf { C L S } _ { j } ^ { \theta } \right\| _ { 2 } ^ { 2 } \right] .\tag{6}
$$

In practice, each mini-batch draws a single pair $( j , x )$ , yielding an unbiased stochastic estimate of this expectation and providing uniform coverage of all earlier tasks over the course of training. We sample only one previous task per iteration rather than considering all previous tasks simultaneously because the latter would over-constrain θ and reduce the adapter’s capacity to learn the latest task.

The overall training loss is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } , } \end{array}\tag{7}
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ is a weighted cross-entropy loss combined with classifier alignment as in [40, 33, 30] (applied to learn a new conditional feature distribution for task t), and $\lambda _ { \mathrm { c o n } } = 2$ in our experiments.

Note that our task-conditioned feature consistency differs from standard distillation methods such as LwF [17], which aligns global predictions or features [10, 16], but cannot preserve the task-specific feature distributions that are central to our design.

## 3.4 Oracle-Free Inference

Generating task-dependent features from FACET at each block requires a task context vector c. We therefore introduce a straightforward strategy to estimate c during inference. Specifically, we attach a linear classifier to the output of the attention layer within each transformer block (Fig. 2). This auxiliary classifier receives the same supervision alongside the final classification head during the continual training phase. During inference, this classifier outputs a probability distribution p<sub>t</sub> for all classes Y<sub>t</sub> for all learned tasks $t \in \{ 1 , . . . , T \}$ . We compute the entropy $\begin{array} { r } { H _ { t } = - \sum _ { y \in \mathcal { y } _ { t } } p _ { t } ( y ) } \end{array}$ log $p _ { t } ( y )$ for each task and select the task context with the highest prediction certainty (lowest entropy) $\hat { t } = \arg \operatorname* { m i n } _ { t } H _ { t } .$ . The final classifier predicts from the last block’s [CLS] feature. Notably, the auxiliary classifiers contain task-specific parameters with a total of $( d \times \textstyle \sum _ { t } | \mathcal { V } _ { t } | )$ parameters, where $\textstyle \sum _ { t } | { \mathcal { D } } _ { t } |$ is the total number of classes in the dataset. Table 3 shows that FACET significantly reduces the task-specific costs compared to existing task-specific adapter methods. Section 4.4 compares alternative context-prediction strategies.

## 4 Experiments

We evaluate FACET on standard CIL benchmarks covering short to very long task sequences. Despite employing a single adapter, FACET outperforms strong baselines, especially on long task sequences. Ablation studies validate the effectiveness of all components.

## 4.1 Experimental Setting

Datasets. We benchmark on four widely used datasets [35, 28, 42, 18, 12], including CIFAR-100 [14], ImageNet-R [8], ImageNet-A [9], and ObjectNet [3], plus long task sequences with up to 200 tasks in a recent benchmark, OmniBenchmark-1K [20] . Together, these benchmarks cover short, medium-length, and long task sequences in standard and distribution-shifted visual domains.

Baselines. We compare FACET against state-of-the-art prompt-based and adapter-based PTM CIL methods with task-specific parameters (L2P [35], DualPrompt [34], CODA-Prompt [28], EASE [42], APER [43], TUNA [33], MOS [30], and MIN [12]), classifier-based methods (SLCA [40], Fe-CAM [6]), and LoRA-based weight-merging methods (InfLoRA [18], SD-LoRA [36], MagMax [22], and OPCM [31]).

## Implementation Details.

We run all our experiments on a single NVIDIA L40S GPU. Following prior work [42, 43, 33, 12, 30], we employ the representative pre-trained vision backbone ViT-B/16-IN21K [5]. FACET adds a single adapter per transformer block (latent dimension 384, hidden depth 2) with k = 8 and $K _ { \mathrm { m a x } } = 2 0 0$ for the task-conditioned transformation matrix M . For training, we use SGD (momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 } )$ , cosine learning-rate decay with base learning rate 0.02 and batch size 16, and conditional feature consistency weight $\lambda _ { \mathrm { c o n } } = 2 . 0$ . The classification head follows [42, 33]. More details about datasets and settings are in the Appendix.

Evaluation Protocol and Performance Metrics. We follow standard settings [35, 33, 12] to partition a dataset into T tasks with the learning order determined by the random seed 1993. We denote Inc n as each task having n classes. For FACET, we report the average performance over 3 runs. We report

Table 1: Performance comparison with state-of-the-art methods on standard benchmarks.
<table><tr><td rowspan="3">Method</td><td colspan="4">ImageNet-R</td><td colspan="4">CIFAR-100</td><td rowspan="2" colspan="2">ImageNet-A 10 Tasks (Inc20)</td><td rowspan="2" colspan="2">ObjectNet 10 Tasks (Inc20)</td></tr><tr><td colspan="2">10 Tasks (Inc20)</td><td colspan="2">20 Tasks (Inc10)</td><td colspan="2">10 Tasks (Inc10)</td><td colspan="2">20 Tasks (Inc5)</td></tr><tr><td colspan="2">À AT</td><td colspan="2">A  ${ \bf A } _ { T }$ </td><td colspan="2">À  ${ \bf A } _ { T }$ </td><td colspan="2">A AT</td><td colspan="2">A  ${ \bf A } _ { T }$ </td><td colspan="2">AT</td></tr><tr><td>L2P(CVPR&#x27;22)</td><td>75.46</td><td>69.77</td><td>63.75</td><td>55.78</td><td>85.92</td><td>79.19</td><td>85.94</td><td>79.93</td><td>49.39</td><td>41.71</td><td>66.77</td><td>55.16</td></tr><tr><td>DualPrompt(ECCV&#x27;22)</td><td>73.10</td><td>67.18</td><td>66.52</td><td>61.77</td><td>89.65</td><td>84.89</td><td>87.87</td><td>81.15</td><td>53.71</td><td>41.67</td><td>64.31</td><td>52.99</td></tr><tr><td>CODA-Prompt(CVPR*23)</td><td>77.97</td><td>72.27</td><td>70.45</td><td>64.68</td><td>91.05</td><td>86.44</td><td>89.11</td><td>81.96</td><td>53.54</td><td>42.73</td><td>66.53</td><td>56.80</td></tr><tr><td>EASE(CVPR&#x27;24)</td><td>81.74</td><td>76.17</td><td>81.18</td><td>74.62</td><td>92.11</td><td>87.72</td><td>91.51</td><td>85.80</td><td>65.34</td><td>55.04</td><td>71.04</td><td>59.37</td></tr><tr><td>SEMA(CVPR&#x27;25)</td><td>81.39</td><td>77.84</td><td>77.84</td><td>69.60</td><td>91.60</td><td>86.75</td><td>92.23</td><td>87.84</td><td>63.83</td><td>52.21</td><td>67.95</td><td>54.92</td></tr><tr><td>MOS(AAAI&#x27;25)</td><td>81.74</td><td>76.17</td><td>82.96</td><td>77.93</td><td>93.83</td><td>90.19</td><td>93.30</td><td>89.25</td><td>69.13</td><td>59.12</td><td>74.75</td><td>65.10</td></tr><tr><td>TUNA(ICCV&#x27;25)</td><td>84.22</td><td>79.42</td><td>82.54</td><td>77.45</td><td>94.85</td><td>91.71</td><td>94.44</td><td>90.74</td><td>72.96</td><td>64.38</td><td>76.46</td><td>66.32</td></tr><tr><td>MIN(NeurIPS*25)</td><td>85.18</td><td>79.75</td><td>83.69</td><td>78.08</td><td>95.12</td><td>92.12</td><td>94.31</td><td>91.03</td><td>72.89</td><td>64.32</td><td>72.56</td><td>61.36</td></tr><tr><td>SLCA(CCV23)</td><td>81.17</td><td>77.00</td><td>81.85</td><td>76.63</td><td>92.67</td><td>89.30</td><td>93.52</td><td>90.07</td><td>68.66</td><td>58.74</td><td>74.12</td><td>63.23</td></tr><tr><td> $\mathrm { F e C A M _ { \mathrm { ( N e u r I P S ^ { \circ } 2 3 ) } } }$ </td><td>79.02</td><td>72.53</td><td>77.84</td><td>71.05</td><td>93.23</td><td>89.05</td><td>91.86</td><td>87.04</td><td>56.04</td><td>46.41</td><td>68.68</td><td>57.39</td></tr><tr><td>APER-Adapter(JCV&#x27;25)</td><td>75.82</td><td>67.95</td><td>72.35</td><td>64.33</td><td>92.22</td><td>87.45</td><td>90.65</td><td>85.15</td><td>60.53</td><td>49.57</td><td>69.24</td><td>57.41</td></tr><tr><td>MagMax-LoRA(ECCV24)</td><td>84.59</td><td>77.68</td><td>81.8</td><td>73.40</td><td>94.76</td><td>91.41</td><td>94.31</td><td>90.94</td><td>66.64</td><td>54.64</td><td>70.85</td><td>59.15</td></tr><tr><td>OPCM-LoRA(NeurlPS*25)</td><td>84.37</td><td>79.38</td><td>84.08</td><td>77.62</td><td>94.90</td><td>91.92</td><td>94.60</td><td>91.18</td><td>69.67</td><td>59.78</td><td>71.99</td><td>60.28</td></tr><tr><td>InfLoRA(CVPR&#x27;24)</td><td>80.82</td><td>75.65</td><td>77.28</td><td>71.01</td><td>91.70</td><td>86.51</td><td>89.13</td><td>81.46</td><td>58.50</td><td>46.28</td><td>70.67</td><td>58.04</td></tr><tr><td>SD-LoRA(ICLR&#x27;25)</td><td>82.04</td><td>77.34</td><td>80.22</td><td>75.26</td><td>92.54</td><td>88.01</td><td>90.90</td><td>85.18</td><td>64.95</td><td>55.96</td><td>70.37</td><td>58.54</td></tr><tr><td>FACET (Ours)</td><td>85.41</td><td>80.66</td><td>84.73</td><td>79.97</td><td>95.18</td><td>92.27</td><td>94.62</td><td>91.31</td><td>73.31</td><td>64.87</td><td>77.13</td><td>67.38</td></tr></table>

Table 2: Performance comparison on benchmarks with long task sequences. InfLoRA and SD-LoRA failed to converge on the 200 task setting.
<table><tr><td rowspan="3">Method</td><td colspan="2">ImageNet-R 50 Tasks (Inc4)</td><td colspan="2">ImageNet-A</td><td colspan="4">OmniBenchmark-1K</td></tr><tr><td colspan="2"></td><td colspan="2">50 Tasks (Inc4)</td><td colspan="2">100 Tasks (Inc10)</td><td colspan="2">200 Tasks (Inc5)</td></tr><tr><td>A</td><td>AT</td><td>A</td><td>AT</td><td>A</td><td>AT</td><td>A</td><td> $\mathbf { A } _ { T }$ </td></tr><tr><td>L2P(CVPR&#x27;22)</td><td>69.16</td><td>63.45</td><td>49.89</td><td>36.41</td><td>60.91</td><td>48.87</td><td>57.53</td><td>45.25</td></tr><tr><td>DualPrompt(ECCV&#x27;22)</td><td>64.00</td><td>56.33</td><td>43.85</td><td>29.95</td><td>62.18</td><td>49.45</td><td>59.13</td><td>45.62</td></tr><tr><td>CODA-Prompt(CVPR&#x27;23)</td><td>62.43</td><td>57.57</td><td>38.24</td><td>26.60</td><td>64.16</td><td>51.75</td><td>60.22</td><td>47.56</td></tr><tr><td>EASE(CVPR&#x27;24)</td><td>78.11</td><td>70.63</td><td>59.86</td><td>47.53</td><td>65.00</td><td>53.54</td><td>67.23</td><td>55.13</td></tr><tr><td>SEMA(CVPR*25)</td><td>67.80</td><td>59.32</td><td>52.99</td><td>40.68</td><td>56.55</td><td>33.96</td><td>45.06</td><td>19.95</td></tr><tr><td>MOS(AAAI&#x27;25)</td><td>75.15</td><td>66.80</td><td>63.72</td><td>51.22</td><td>76.80</td><td>64.27</td><td>75.92</td><td>63.51</td></tr><tr><td>TUNA(ICCV&#x27;25)</td><td>80.93</td><td>75.48</td><td>67.82</td><td>58.72</td><td>71.93</td><td>60.04</td><td>71.45</td><td>59.14</td></tr><tr><td>MIN(NeurIPS*25)</td><td>82.26</td><td>75.72</td><td>66.15</td><td>57.08</td><td>75.46</td><td>63.60</td><td>74.94</td><td>62.50</td></tr><tr><td>APER-Adapter(IJCV&#x27;25)</td><td>72.43</td><td>64.83</td><td>61.29</td><td>48.58</td><td>73.23</td><td>62.24</td><td>72.49</td><td>61.53</td></tr><tr><td>MagMax-LoRA(ECCV24)</td><td>81.87</td><td>74.97</td><td>65.40</td><td>52.58</td><td>69.93</td><td>41.85</td><td>61.30</td><td>30.27</td></tr><tr><td>OPCM-LoRA(NeurlPS*25)</td><td>82.49</td><td>73.70</td><td>66.58</td><td>54.02</td><td>60.19</td><td>33.73</td><td>58.32</td><td>28.43</td></tr><tr><td>InfLoRa(CVPR&#x27;24)</td><td>71.68</td><td>62.62</td><td>50.17</td><td>38.70</td><td>51.53</td><td>27.01</td><td></td><td></td></tr><tr><td>SD-LoRA(ICLR&#x27;25)</td><td>68.40</td><td>63.28</td><td>54.72</td><td>41.28</td><td>53.97</td><td>28.15</td><td></td><td></td></tr><tr><td>FACET (Öurs)</td><td>82.70</td><td>76.63</td><td>69.16</td><td>59.58</td><td>76.88</td><td>66.17</td><td>77.30</td><td>65.95</td></tr></table>

the average incremental accuracy $\textstyle { \bar { A } } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } A _ { t }$ and final accuracy $A _ { T }$ , where $A _ { t }$ is the average accuracy over all classes seen so far after the adapter has been trained on task t. Due to space limit, results on the ViT-B/16-IN1k backbone and other random seeds are provided in the Appendix B.1.

## 4.2 Main Results

Tables 1 and 2 compare FACET with SOTA PTM-based CIL methods. Our experimental results clearly demonstrate two key advantages. First, our single adapter yields superior performance on standard benchmarks in comparison to computationally heavy methods based on a set of task-specific adapters. Second, this design surprisingly achieves leading performance on long task sequences of up to 200 tasks when compared to other strong baselines. More details are provided below.

Comparison with State-of-the-Art Methods. Table 1 shows that FACET achieves the highest final accuracy A<sub>T</sub> on all four datasets that have 10 to 20 tasks. For instance, on ImageNet-R with 20 tasks, FACET improves $A _ { T }$ from 78.08% to 79.97% compared to MIN, one of the strongest baselines that stores one adapter-like module for every task. On the more challenging ImageNet-A benchmark, where adversarially filtered images make class discrimination more difficult, the strongest baselines with task-specific adapters, TUNA and MIN, substantially outperform LoRA-based merging methods, showing that explicit task-specific representations are important. Nevertheless, FACET achieves the highest $A _ { T }$ (64.87%) and competitive A<sup>¯</sup> (73.31%) with a single adapter, improving over the strongest LoRA-based merging method by 3.64% in A<sup>¯</sup> and 5.09% in $A _ { T }$ . OPCM-LoRA remains competitive on ImageNet-R and CIFAR-100, but FACET still improves A by 2.35% on ImageNet-R with 20 tasks.

Scaling to Long Task Sequences. Table 2 evaluates whether FACET remains effective as the number of tasks in a sequence increases to 100 or even 200, when storing, searching over, or merging many task-specific modules becomes increasingly difficult. On ImageNet-R with 50 tasks, FACET achieves the highest A<sup>¯</sup> and $A _ { T }$ . It improves $A _ { T }$ from 75.72% to 76.63% compared to MIN, and by an absolute

Table 3: Comparison of parameter, computation, and GFLOPs among SOTA methods. Latency is the median latency at batch size 32.  
ImageNet-R (50 tasks)
<table><tr><td>Method</td><td>Params(M)</td><td>Latency(ms)</td><td>GFLOPs</td><td>À</td><td> $A _ { T }$ </td></tr><tr><td>Ours</td><td>10.20</td><td>28.61</td><td>37.37</td><td>82.70</td><td>76.63</td></tr><tr><td>MoS</td><td>15.37</td><td>1431.30</td><td>1798.57</td><td>75.15</td><td>66.80</td></tr><tr><td>TUNA</td><td>15.37</td><td>1308.28</td><td>1798.05</td><td>80.93</td><td>75.48</td></tr><tr><td>MiN</td><td>135.15</td><td>44.02</td><td>55.39</td><td>82.26</td><td>75.72</td></tr></table>

OmniBenchmark-1K (200 tasks)
<table><tr><td>Method</td><td>Params(M)</td><td>Latency(ms)</td><td>GFLOPs</td><td>Ã</td><td> $A _ { T }$ </td></tr><tr><td>Ours</td><td>22.88</td><td>36.64</td><td>40.23</td><td>77.30</td><td>65.95</td></tr><tr><td>MoS</td><td>61.63</td><td>5717.77</td><td>7088.81</td><td>75.92</td><td>63.51</td></tr><tr><td>TUNA</td><td>61.63</td><td>6306.79</td><td>7086.94</td><td>71.45</td><td>59.14</td></tr><tr><td>MiN</td><td>1831.59</td><td>141.42</td><td>107.84</td><td>74.94</td><td>62.50</td></tr></table>

2.93% compared to OPCM-LoRA, the strongest LoRA-based merging method. On ImageNet-A with 50 tasks, FACET again achieves the highest A<sup>¯</sup> and $A _ { T }$ , improving over TUNA by 1.34% in A<sup>¯</sup> and 0.86% in $A _ { T }$ . This is significant because ImageNet-A is the setting where LoRA-based merging methods struggle the most: FACET improves $A _ { T }$ over OPCM-LoRA by 5.56% while preserving the single-adapter design. The scaling advantage becomes clearer for 100 and 200 tasks. On OmniBenchmark-1K with 100 tasks, FACET improves over the strongest baseline by 1.90% in $A _ { T } ,$ , while achieving the highest A<sup>¯</sup> of 76.88. When the number of tasks further increases to 200, FACET remains the top performer among all baselines, improving A<sup>¯</sup> from 75.92% to 77.30% and $A _ { T }$ from 63.51% to 65.95% over MOS. These results show that FACET is capable of scaling to a large number of tasks while maintaining a single adapter, and task-conditioned feature transformations remain effective even when the number of tasks reaches 200.

## 4.3 Parameter, Computation and GFLOPs

Table 3 compares our method with the best-performing baselines (i.e., MIN, MOS, and TUNA) in trainable parameter count, inference latency, and GFLOPs per image on long task sequences. Our method is among the most efficient and achieves a 156× speedup in inference latency and only use 0.57% GFLOPs compared to MoS when there are 200 tasks. Furthermore, FACET requires only 1.25% of the cumulative trainable parameters of MIN. These findings demonstrate that our proposed method maintains leading performance while delivering better parameter efficiency and substantially lower latency than per-task adapter methods such as TUNA/MOS.

## 4.4 Ablation Studies

We extensively verify the effectiveness of all components in FACET with ablation studies conducted on the ImageNet-R Inc10 (20 tasks) dataset, which provides a challenging setting.

The baseline is our full FACET method presented in Table 1.

Individual contributions of components. Table 4 decomposes the gains of FACET. A plain single adapter reaches

Table 4: Incremental component ablation on ImageNet-R (Inc10).
<table><tr><td>Variant</td><td> $M _ { c }$  cond</td><td>Consistency</td><td>Other designs</td><td>À  ${ \bf A } _ { T }$ </td></tr><tr><td>Plain Single Adapter</td><td>X</td><td>×</td><td>×</td><td>57.75 48.75</td></tr><tr><td> $+ M _ { c }$  Transformation</td><td>√</td><td>X</td><td>X</td><td>70.06 60.02</td></tr><tr><td>+ Cond. Feat. Consis.</td><td>√</td><td>√</td><td>X</td><td>83.90 78.23</td></tr><tr><td>FACET (Full)</td><td>√</td><td>√</td><td>√</td><td>84.70 79.97</td></tr><tr><td>Feat. Consis. Only</td><td>×</td><td></td><td>√</td><td>72.94 65.39</td></tr></table>

only 48.75% final accuracy, confirming that simply sharing weights across tasks is insufficient. Adding task-conditioned feature transformations $( M _ { c }$ transformation) improves final accuracy to 60.02%, demonstrating that explicit task conditioning and conditioned transformations yield a substantial benefit by making task-specific feature distributions more separable. The performance jump to 78.23% after adding task-conditioned feature consistency, showing that once FACET creates separable mixture components, they are fragile to forgetting and must be explicitly preserved. Note that non-conditioned feature consistency alone achieves an $A _ { T }$ of 65.39%, indicating that the interplay between conditioned feature transformation and consistency is crucial. Finally, other designs such as adding gaussian noise during training together contributes a further improvement to 79.97%.

Variants of conditioned feature transformation. Table 5 studies alternative forms of taskconditioned transformation and the effect of an explicit separability loss (Sec. 3.2), reporting ${ \bar { A } } ,$ A<sub>T</sub>, and the silhouette scores computed from last-block features as a measure of class discriminability.

Transforming the adapter input instead of its output causes a sharp drop of $A _ { T }$ to 69.22% and a negative silhouette score, suggesting that intra-class distances exceed inter-class distances. This confirms that conditioning the adapter output via Eq. (3) is essential. Diagonal-only and low-rank-only variants of $M _ { c } ,$ already raise performance clearly above the plain single adapter, showing that the primary gain comes from explicit task conditioning. Finally, adding explicit losses to further sepa-

Table 5: Ablation on conditioned feature transformation and separability loss.
<table><tr><td>Form</td><td>A</td><td> ${ \bf A } _ { T }$ </td><td>Silhouette</td></tr><tr><td>No transformation</td><td>72.49</td><td>65.39</td><td>-0.11</td></tr><tr><td>Concat c to adapter input</td><td>69.46</td><td>66.42</td><td>-0.086</td></tr><tr><td> $M _ { c }$  at adapter input</td><td>72.27</td><td>69.22</td><td>-0.057</td></tr><tr><td> $\bar { M _ { c } } = U \bar { V } ^ { \top }$ </td><td>84.50</td><td>78.58</td><td>0.026</td></tr><tr><td> $M _ { c } = \operatorname { D i a g } ( s )$ </td><td>84.46</td><td>79.60</td><td>0.040</td></tr><tr><td> $M _ { c } = \mathrm { D i a g } ( s ) \substack { + U V ^ { \top } }$ </td><td>84.70</td><td>79.97</td><td>0.042</td></tr><tr><td>+Orthogonal  $M _ { c }$  loss</td><td>82.19</td><td>78.63</td><td>0.035</td></tr><tr><td> $+ z _ { c }$  cos distance loss</td><td>81.83</td><td>78.02</td><td>0.014</td></tr></table>

rate task-specific feature distributions hurts performance since it reduces the learning plasticity for new tasks.

Design of conditioned feature consistency. Table 6 studies how to best preserve previously learned taskspecific feature distributions. Only enforcing consistency for the most recent task under-protects earlier task-specific distributions $( A _ { T } = 7 7 . 0 3 \% )$ , while considering all previous tasks simultaneously over-constrains the adapter and limits learning plasticity $( A _ { T } = 7 8 . 3 7 \% )$ . Uniformly sampling one previous task per mini-batch achieves the best trade-off, covering all prior task-specific distributions without hindering new task learning. A hard adapter snapshot outperforms an EMA teacher, indicating that the exact post-training adapter state is a more reliable reference. Alternative designs in computing the loss lead to suboptimal results, such as using CLS tokens from every block (rather than just the last block), all tokens from the last block, and using cosine distance (1-cosine similarity) in Eq. 6 instead of MSE.

Table 6: Ablation on conditioned feature consistency.
<table><tr><td>Strategy</td><td>A</td><td> ${ \bf A } _ { T }$ </td></tr><tr><td>Most recent (t — 1)</td><td>83.31</td><td>77.03</td></tr><tr><td>All tasks simultaneous</td><td>84.01</td><td>78.37</td></tr><tr><td>Uniform, EMA teacher Uniform, snapshot (ours)</td><td>83.96</td><td>78.85</td></tr><tr><td></td><td>84.70</td><td>79.97</td></tr><tr><td>All features (Last Block) CLS in every block</td><td>80.56 84.35</td><td>70.70</td></tr><tr><td>Cosine Distance</td><td>84.53</td><td>79.80 79.46</td></tr></table>

Task Context Prediction. Table 7 investigates alternative methods to predict task context during inference. Assembling the context vector using the highest logit in each task from the final classification head results in suboptimal performance, while a context with the highest logits from auxiliary classifiers sees a 1.34% drop in $A _ { T }$ demonstrating the advantage of entropy for task identity prediction. Directly filling the context using the entropy value for each task leads to a larger decline of $A _ { T }$ to

Table 7: Ablation on context prediction.
<table><tr><td>Predictor</td><td>Ã</td><td> ${ \bf A } _ { T }$ </td></tr><tr><td>Final classifier highest logits</td><td>84.07</td><td>79.33</td></tr><tr><td>Aux. classifier highest logits Entropy as task context</td><td>84.19 84.23</td><td>78.63 77.85</td></tr><tr><td>Min entropy</td><td>84.70 79.97</td><td></td></tr><tr><td>Oracle</td><td></td><td>81.62</td></tr></table>

77.85%, confirming that the use of sparse context vectors is beneficial. Notably, using ground truth task context during inference improves final accuracy to 81.62%, implying that FACET can benefit from more accurate task context predictors for short task sequences. Additional analysis on task context prediction and robustness of FACET to task context prediction errors are given in the Appendix.

## 5 Limitations

FACET is designed for exemplar-free class-incremental learning with a frozen pretrained vision backbone, disjoint task label sets, and known task boundaries during training. These assumptions make the method directly comparable to PTM-based CIL methods, but they also leave several directions open. While FACET substantially reduces the task-specific costs compared to state of the art methods for long task sequences with 200 tasks, task-specific cost is kept small in the auxiliary classifiers but does not fully disappear. Moreover, the single adapter capacity may saturate. Although we achieve strong performance up to 200 tasks, longer task sequences or highly heterogeneous tasks may eventually exceed what one adapter can represent. Extending our method to true lifelong continual learning over thousands of tasks is non-trivial. In addition, the replay-free objective preserves old task-specific feature distributions by matching a frozen snapshot on current-task inputs, which is efficient and avoids storing old data. However, it is not equivalent to matching the full old data distribution and may be weaker when the data distribution of the new task is highly different from earlier tasks. Promising directions include extending the method to large language models and studying alternative task context prediction methods.

## 6 Conclusion

This work has introduced FACET, a single-adapter approach for pretrained-model-based classincremental learning. By transforming the adapter’s latent representation in a task-conditioned manner while preserving previously learned task-specific feature distributions, FACET makes continual learning more efficient and scalable. Future work could extend the proposed method’s principles of sharing weights across tasks with dynamic conditioned transformation to extract task-specific features to other domains, such as large language models, vision-language models, and other scenarios where efficient continual learning is useful.

## References

[1] Davide Abati, Jakub Tomczak, Tijmen Blankevoort, Simone Calderara, Rita Cucchiara, and Babak Ehteshami Bejnordi. Conditional channel gated networks for task-aware continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3931–3940, 2020.

[2] Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3366–3375, 2017.

[3] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019.

[4] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems, 35:16664–16678, 2022.

[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[6] Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, and Joost Van De Weijer. Fecam: Exploiting the heterogeneity of class distributions in exemplar-free continual learning. Advances in Neural Information Processing Systems, 36:6582–6595, 2023.

[7] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

[8] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021.

[9] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021.

[10] Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi. A comprehensive overhaul of feature distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1921–1930, 2019.

[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 2022.

[12] Kai Jiang, Zhengyan Shi, Dell Zhang, Hongyuan Zhang, and Xuelong Li. Mixture of noise for pre-trained model-based class-incremental learning. Advances in neural information processing systems, 39, 2025.

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe national academy of sciences, 114(13):3521–3526, 2017.

[14] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.(2009), 2009.

[15] Timothée Lesort, Vincenzo Lomonaco, Andrei Stoian, Davide Maltoni, David Filliat, and Natalia Díaz-Rodríguez. Continual learning for robotics: Definition, framework, learning strategies, opportunities and challenges. Informationfusion, 58:52–68, 2020.

[16] Songze Li, Tonghua Su, Xu-Yao Zhang, and Zhongjie Wang. Continual learning with knowledge distillation: A survey. IEEE Transactions on Neural Networks and Learning Systems, 36(6): 9798–9818, 2024.

[17] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

[18] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23638–23647, 2024.

[19] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

[20] Meng Lou, Yunxiang Fu, and Yizhou Yu. Scaling continual learning with bi-level routing mixture-of-experts. arXiv preprint arXiv:2602.03473, 2026.

[21] Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 7765–7773, 2018.

[22] Daniel Marczak, Bartłomiej Twardowski, Tomasz Trzcinski, and Sebastian Cygert. Magmax:´ Leveraging model merging for seamless continual learning. In European Conference on Computer Vision, pages 379–395. Springer, 2024.

[23] Marc Masana, Xialei Liu, Bartłomiej Twardowski, Mikel Menta, Andrew D Bagdanov, and Joost Van De Weijer. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 5513–5533, 2022.

[24] Nicolas Y Masse, Gregory D Grant, and David J Freedman. Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. Proceedings ofthe National Academy ofSciences, 115(44):E10467–E10475, 2018.

[25] Yuan Meng, Zhenshan Bing, Xiangtong Yao, Kejia Chen, Kai Huang, Yang Gao, Fuchun Sun, and Alois Knoll. Preserving and combining knowledge in robotic lifelong reinforcement learning. Nature Machine Intelligence, 7(2):256–269, 2025.

[26] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings ofthe IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[27] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

[28] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11909–11919, 2023.

[29] Hai-Long Sun, Da-Wei Zhou, De-Chuan Zhan, and Han-Jia Ye. Pilot: A pre-trained modelbased continual learning toolbox. SCIENCE CHINA Information Sciences, 68(4):147101, 2025.

[30] Hai-Long Sun, Da-Wei Zhou, Hanbin Zhao, Le Gan, De-Chuan Zhan, and Han-Jia Ye. Mos: Model surgery for pre-trained model-based class-incremental learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 20699–20707, 2025.

[31] An Quang Tang, Enneng Yang, Li Shen, Yong Luo, Han Hu, Bo Du, and Dacheng Tao. Merging models on the fly without retraining: A sequential approach to scalable continual model merging. Advances in neural information processing systems, 39, 2025.

[32] Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. Self-expansion of pre-trained models with mixture of adapters for continual learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10087–10098, 2025.

[33] Yan Wang, Da-Wei Zhou, and Han-Jia Ye. Integrating task-specific and universal adapters for pre-trained model-based class-incremental learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 806–816, 2025.

[34] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European conference on computer vision, pages 631–648. Springer, 2022.

[35] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 139–149, 2022.

[36] Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, and Ying Wei. Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning. In International Conference on Learning Representations, 2025.

[37] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3014–3023, 2021.

[38] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017.

[39] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995. Pmlr, 2017.

[40] Gengwei Zhang, Liyuan Wang, Guoliang Kang, Ling Chen, and Yunchao Wei. Slca: Slow learner with classifier alignment for continual learning on a pre-trained model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19148–19158, 2023.

[41] Yuanhan Zhang, Zhenfei Yin, Jing Shao, and Ziwei Liu. Benchmarking omni-vision representation through the lens of visual realms. In European conference on computer vision, pages 594–611. Springer, 2022.

[42] Da-Wei Zhou, Hai-Long Sun, Han-Jia Ye, and De-Chuan Zhan. Expandable subspace ensemble for pre-trained model-based class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23554–23564, 2024.

[43] Da-Wei Zhou, Zi-Wen Cai, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Revisiting classincremental learning with pre-trained models: Generalizability and adaptivity are all you need. International Journal ofComputer Vision, 133(3):1012–1032, 2025.

## A Additional Dataset and Implementation Details

## A.1 Datasets

We evaluate FACET on five datasets, including ImageNet-R [8], ImageNet-A [9], CIFAR100 [14], ObjectNet [3, 43] , and OmniBenchmark-1K [41, 20]. The datasets used in the main comparison are summarized below:

ImageNet-R [8] has 200 classes with 30000 images. It consists of ImageNet-compatible classes rendered in artistic and non-standard visual styles.

ImageNet-A [9] contains 200 classes with 7475 images. The dataset consists of naturally adversarial images with ImageNet-compatible categories.

CIFAR100 [14] is a standard image classification benchmark that contains 100 classes and 60000 images.

Following [33, 30, 43], the ObjectNet [3, 43] dataset we refer to for CIL contains 200 classes with 33137 images. It a subset of the original ObjectNet [3] dataset that contains 50000 images of 313 distinct objects with different backgrounds, rotations, and imaging viewpoints.

OmniBenchmark-1K [41, 20] has 188569 images and 1000 classes with non-overlapping data with ImageNet-1k. It is a subset of the clean version of OmniBenchmark [41] with at least 100 images per class.

## A.2 Implementation Details

We use the PILOT [29] codebase to run our experiments. Following [35, 34, 42, 30, 43, 33, 12, 18, 36], our main experiments use the ViT-B/16 backbone pretrained on ImageNet-21k. FACET inserts one task-conditioned adapter shared across all tasks in each transformer block, with two 384-dimensional hidden layers. The linear projections for generating context vectors $( W _ { s } , W _ { U } , W _ { V } )$ are initialized to zeros for the first step of training. The auxiliary classifier is a linear layer that is trained to predict class labels directly. A zero-mean Gaussian noise with a standard deviation of 0.3 is added to the context vector during training to improve its robustness against imperfect context predictions during inference. Optimization uses SGD with a cosine learning-rate schedule, a weight decay of $5 \times 1 0 ^ { - \bar { 4 } }$ a batch size of 16, a learning rate of 0.04, and 20 training epochs. Classifier alignment following [33] is included during training. For the short task sequence benchmarks with 10 to 20 tasks, we ensemble the predictions when using task context and when disabling the adapter by setting task context to zero. For long task sequences, predictions use only the FACET probabilities from a single backbone pass.

## B Additional Experiments

Table 8: Performance comparison on ImageNet-R (Inc10) using ViT-B/16-IN21k backbone and 5 random seeds. Metric is $\bar { A } \bar { / } A _ { T }$
<table><tr><td>Method \ Seed</td><td>49</td><td>127</td><td>982</td><td>1523</td><td>2026</td><td>Mean ± Std</td></tr><tr><td>SD-LoRA</td><td>78.50 / 70.58</td><td>77.27 / 70.73</td><td>77.82 / 71.85</td><td>76.58 / 69.95</td><td>77.22 /70.77</td><td> $7 7 . 4 8 \pm 0 . 7 2 / 7 0 . 7 8 \pm 0 . 6 8$ </td></tr><tr><td>TUNA</td><td>82.51 / 77.78</td><td>81.55 / 76.97</td><td>82.37 / 77.73</td><td>82.44 / 77.68</td><td>82.44 / 77.15</td><td> $8 2 . 2 6 \pm 0 . 4 0 / 7 7 . 4 6 \pm 0 . 3 7$ </td></tr><tr><td>MIN</td><td>84.41 / 78.90</td><td>84.46 / 77.80</td><td>85.22 / 79.40</td><td>84.38 / 78.10</td><td>83.65 / 77.43</td><td> $8 4 . 4 2 \pm 0 . 5 6 / 7 8 . 3 3 \pm 0 . 8 1$ </td></tr><tr><td>FACET</td><td>85.63 / 80.80</td><td>85.14/ 79.32</td><td>85.89 / 80.98</td><td>84.66 / 79.52</td><td>84.77 / 79.28</td><td> $\mathbf { 8 5 . 2 2 \pm 0 . 5 3 / 7 9 . 9 8 \pm 0 . 8 4 }$ </td></tr></table>

Table 9: Performance comparison on ImageNet-R (Inc10) using ViT-B/16-IN1k backbone and 5 random seeds. Metric is $\bar { A } \bar { / } A _ { T }$
<table><tr><td>Method \ Seed</td><td>49</td><td>127</td><td>982</td><td>1523</td><td>2026</td><td>Mean ± Std</td></tr><tr><td>SD-LoRA</td><td>81.40 / 74.65</td><td>81.19 / 75.72</td><td>80.45 / 74.77</td><td>80.03 / 74.75</td><td>79.82 / 73.32</td><td> $8 0 . 5 8 \pm 0 . 7 0 / 7 4 . 6 4 \pm 0 . 8 6$ </td></tr><tr><td>TUNA</td><td>83.98 / 78.53</td><td>83.37 / 77.97</td><td>84.08 / 78.78</td><td>83.36 / 78.31</td><td>83.45 / 78.22</td><td> $8 3 . 6 5 \pm 0 . 3 5 / 7 8 . 3 6 \pm 0 . 3 1$ </td></tr><tr><td>MIN</td><td>85.01 / 79.35</td><td>84.26 / 77.95</td><td>85.75 / 80.03</td><td>83.73 / 78.47</td><td>83.86 / 78.28</td><td> $8 4 . 5 2 \pm 0 . 8 5 / 7 8 . 8 2 \pm 0 . 8 5$ </td></tr><tr><td>FACET</td><td>86.05 / 80.87</td><td>84.82 / 79.90</td><td>86.65 / 80.88</td><td>86.68 / 80.67</td><td>85.50 / 80.07</td><td> $8 5 . 9 4 \pm { \bf 0 . 7 9 } / 8 0 . 4 8 \pm { \bf 0 . 4 6 }$ </td></tr></table>

## B.1 Robustness to Task Order and Pretrained Backbone

To evaluate the robustness and generalizability of our approach, we shuffle the task order with 5 random seeds and experiment with two widely used pretrained backbones, ViT-B/16-IN21k and ViT-B/16-IN1k [5]. Tables 8-9 present our results. FACET outperforms strong baselines in all task orders on both backbones. This indicates that the advantage of FACET is consistent rather than an artifact of a particular task order.

## B.2 Detailed Ablations on Single Adapter Design

Random initialization starts each new task with a random transformation of the adapter feature space and reduces $A _ { T }$ to 76.68%. Zero initialization avoids this unstable start by making a newly activated task-conditioned mixture component initially fall back to the pretrained representation, improving $A _ { T }$ to 78.80%. Adding noise to the context vector further improves A to 79.97%, supporting its role in making the model robust to imperfect task identity prediction during inference.

Table 10: Impact of initialization and noise on ImageNet-R (Inc10).
<table><tr><td>Config.</td><td>A</td><td> ${ \bf A } _ { T }$ </td></tr><tr><td>Random, no noise</td><td>82.74</td><td>76.68</td></tr><tr><td>Zero, no noise</td><td>83.95</td><td>78.80</td></tr><tr><td>Zero, +noise</td><td>84.70</td><td>79.97</td></tr></table>

Finally, Table 12 varies the bottleneck dimension and adapter depth. The results are stable across a range of capacities, and deeper adapters do not consistently improve performance, indicating that FACET’s gains are not simply due to adding more adapter parameters.

As shown in Table 11, FACET is robust to different values of $K _ { m a x } ,$ with smaller values achieving better performance for ImageNet-R (Inc10) with 20 tasks. We select 200 since it is a sufficiently large upper bound on the number of tasks in CIL.

Table 11: Effect of $K _ { m a x }$ on ImageNet-R (Inc10).
<table><tr><td>Metric \ Dim</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td> $\mathbf { A } _ { \mathbf { T } }$ </td><td>80.20</td><td>80.14</td><td>79.97</td><td>79.64</td></tr><tr><td> $\bar { \bf A }$ </td><td>84.77</td><td>84.73</td><td>84.70</td><td>84.44</td></tr></table>

Table 12: Ablation of adapter capacity on ImageNet-R (Inc10). Metric is $( \mathbf { A } _ { T } )$
<table><tr><td>Hidden depth \ Dim</td><td>16</td><td>64</td><td>256</td><td>384</td></tr><tr><td>1</td><td>79.08</td><td>79.10</td><td>79.38</td><td>79.49</td></tr><tr><td>2</td><td>79.39</td><td>79.30</td><td>79.57</td><td>79.97</td></tr><tr><td>3</td><td>79.37</td><td>78.97</td><td>79.62</td><td>79.15</td></tr></table>

Table 13: Impact of the consistency weight $\lambda _ { \mathrm { c o n } }$ on ImageNet-R (Inc10).
<table><tr><td>Metric</td><td>0.5</td><td>1.0</td><td>2.0</td><td>4.0</td><td>8.0</td></tr><tr><td>A</td><td>83.94</td><td>84.75</td><td>84.70</td><td>84.34</td><td>84.13</td></tr><tr><td> ${ \bf A } _ { T }$ </td><td>77.12</td><td>79.28</td><td>79.97</td><td>78.72</td><td>78.50</td></tr></table>

## B.3 Impact of the Coefficient for

## Task-Dependent Feature Consistency Loss

Table 13 shows that $\lambda _ { \mathrm { c o n } } = 2 . 0$ achieves the highest final accuracy, achieving an optimal balance between mitigating forgetting of task-specific feature distributions and learning new task.

## C Analysis of Task Prediction Accuracy

We further evaluate the robustness of FACET to wrong task-context predictions during inference. Without retraining, we compare with a randomly assigned context and a deliberately incorrect context on the same checkpoint. Random and forced-wrong results are mean ± sample standard deviation over five tries.

Task prediction accuracy is 5.3 to 9.8 times above random guesses. More importantly, using an incorrect context reduces final accuracy by at most 4.48 (79.97 to 75.49 for ImageNet-R 20 tasks).

Table 14: Robustness of FACET to wrong task-context predictions at inference (same checkpoint, no retraining). Random and forced-wrong results are mean ± sample standard deviation over five tries.
<table><tr><td>Setting</td><td>Task-context pred. acc. (random choice)</td><td>Final acc. w/ predicted ctx.</td><td>Final acc. w/ oracle ctx.</td><td>Final acc. w/ random ctx.</td><td>Final acc. w/ forced-wrong ctx.</td></tr><tr><td>ImageNet-R Inc20</td><td>53.83% (10%)</td><td>80.66</td><td>86.38</td><td> $7 8 . 0 7 \pm 0 . 1 9$ </td><td> $7 7 . 5 4 \pm 0 . 1 5$ </td></tr><tr><td>ImageNet-R Inc10</td><td>49.05% (5%)</td><td>79.97</td><td>81.62</td><td> $7 5 . 8 0 \pm 0 . 2 4$ </td><td> $7 5 . 4 9 \pm 0 . 2 8$ </td></tr><tr><td>ImageNet-R Inc4</td><td>15.21% (2%)</td><td>76.63</td><td>78.72</td><td> $7 6 . 5 9 \pm 0 . 1 3$ </td><td> $7 6 . 4 7 \pm 0 . 0 9$ </td></tr></table>

We note that 75.49 is comparable to the performance of the strong baseline SD-LoRA (75.26). This shows that FACET is robust to wrong task-context predictions during inference. We note that providing ground-truth task context during inference is beneficial (ablation Table 7), and can improve final accuracy from 79.97 to 81.62 on ImageNet-R 20 tasks. Therefore, the performance of FACET is not only positively correlated with task-context prediction accuracy, but also robust against wrong task-context predictions. More advanced mechanisms to predict task context can further improve performance.

## D Additional Analysis of Task-Conditioned Feature Consistency

![](images/27378e079b421629df7a3ee90b593d96dcba121cbd50e51ba84c93641b5b2d82.jpg)

![](images/24c951e3a0d12e621d6bc2d7ae3712f2a8c2696b07e8b2a0cdf517f290680d5e.jpg)  
Figure 3: How well our single adapter preserves the feature distribution of the first task as it is sequentially trained on new tasks from task 2 to 100. Top: Mean square error as the distance metric. Bottom: Cosine distance as the distance metric. We compute the distance between the CLS feature produced by the model after it has learned the first task with the corresponding CLS feature after the model has sequentially learned $t \in \{ 2 , . . . , 1 0 0 \}$ } tasks. The original training data and context for the first task are used for a direct comparison.

Here we evaluate how well our conditioned feature consistency loss preserves the feature distributions of historical tasks. Specifically, we measure how well the sample features from the first task are preserved after learning new tasks. The CLS token from the last transformer block is selected as the feature representation since it is passed to the classifier for prediction. Note that the task-conditioned feature consistency loss is computed using data from task t $( x \in \mathcal { D } _ { t } )$ , which is the latest task the adapter is being trained on.

Results are shown in Fig. 3, where the solid line denotes the mean distance and the light shaded area illustrates the standard deviation. It can be seen that our conditioned feature consistency loss (blue line) effectively preserves the original features from the first task, even after the adapter has been trained on 99 subsequent tasks from the OmniBenchmark-1K dataset. Without our conditioned feature consistency loss (orange line), the MSE and Cosine distances are much larger, which indicates forgetting of the task-specific distribution. For instance, after task 100, the mean cosine distance and mean squared error when using the consistency loss are 0.0588 and 0.0866, respectively. Without the consistency loss, cosine distance and MSE increases to 0.249 (+323%) and 0.427 (+393%), respectively. These results confirm the effectiveness of the conditioned feature consistency loss in preserving task-specific feature distributions in continual learning even though it relies on data from the latest task instead of those from the original tasks.

We provide more details on how cosine distance and MSE are computed below: We first extract the CLS features of the data samples from the first task using the adapter trained on its dataset $\mathcal { D } _ { 1 }$ . This defines the original feature distribution of the first task. Then, after learning t tasks, we extract CLS features using the same data samples from $\mathcal { D } _ { 1 }$ and the same context vector for the first task, but using the adapter sequentially trained on all t tasks. We evaluate $t \in \{ 2 , . . . , 1 0 0 \}$ using the OmniBenchmark-1K dataset. Because the input images and context vectors are the same, any difference between corresponding CLS features directly measures how much an original feature from the first task has drifted after learning later tasks.

We quantify this drift using cosine distance and mean squared error (MSE). For each training image $x \in \mathcal { D } _ { 1 }$ , cosine distance is computed between the original CLS feature and the corresponding CLS feature extracted after learning t tasks:

$$
d _ { \mathrm { c o s } } = 1 - \frac { \langle \mathrm { C L S } _ { 1 } ^ { \theta _ { 1 } } , \mathrm { C L S } _ { 1 } ^ { \theta _ { t } } \rangle } { \| \mathrm { C L S } _ { 1 } ^ { \theta _ { 1 } } \| _ { 2 } \| \mathrm { C L S } _ { 1 } ^ { \theta _ { t } } \| _ { 2 } } ,
$$

where $\mathrm { C L S } _ { 1 } ^ { \theta _ { 1 } }$ is the original CLS feature from the model trained on the first task, and $\mathrm { C L S } _ { 1 } ^ { \theta _ { t } }$ is the CLS feature from the model after learning t tasks using the same context. We average the cosine distance over all data samples. We also compute the MSE distance:

$$
d _ { \mathrm { M S E } } = \frac { 1 } { N } \left. \mathrm { C L S } _ { 1 } ^ { \theta _ { 1 } } - \mathrm { C L S } _ { 1 } ^ { \theta _ { t } } \right. _ { 2 } ^ { 2 } ,
$$

where N is the number of images in the first task. These two metrics are complementary: cosine distance captures directional drift, while MSE captures overall feature-space displacement.

Table 15: Task-1 forgetting versus CLS feature drift under different $\lambda _ { \mathrm { { c o n } } }$ on ImageNet-R Inc10.
<table><tr><td> $\lambda _ { \mathrm { { c o n } } }$ </td><td>Task-1 forgetting</td><td>Cosine distance</td><td>MSE</td></tr><tr><td>0</td><td>35.96</td><td>0.2376</td><td>0.5072</td></tr><tr><td>0.5</td><td>36.59</td><td>0.0888</td><td>0.1578</td></tr><tr><td>1</td><td>31.23</td><td>0.0538</td><td>0.0961</td></tr><tr><td>2</td><td>18.61</td><td>0.0285</td><td>0.0502</td></tr><tr><td>4</td><td>15.46</td><td>0.0167</td><td>0.0296</td></tr><tr><td>8</td><td>13.32</td><td>0.0113</td><td>0.0206</td></tr></table>

We also directly test whether feature drift is related to old-task classification forgetting. On ImageNet-R Inc10, we train six models that differ only in $\lambda _ { \mathrm { { c o n } } } .$ After task 20, we compare the CLS features of the task-1 images with their features immediately after task 1, always using the task-1 context, and measure task-1 forgetting.

Across the six settings, cosine distance and MSE have Pearson correlations of 0.752 and 0.711 with task-1 forgetting. Greater drift is strongly associated with greater old-task classification forgetting.

## E Justification of Using Current-Task Data for Conditioned Feature Consistency

In Equation 6, we use the data from the current task $D _ { t }$ to preserve the old feature distributions. Here we provide a justification on why it works.

Current-task images are probe inputs used to detect whether new-task training changes the shared adapter under a previous context. They are not treated as approximations of old-task images in the consistency loss.

After task t−1, we freeze a copy of the adapter. While learning task $t ,$ a current image is processed by the frozen and updated adapters under the same sampled old context $c _ { j }$ . Because the input and context are identical, their feature difference isolates the effect of the adapter update. Minimizing this difference therefore constrains new-task training from changing the adapter’s behavior under $c _ { j }$

The backbone is frozen and all tasks use the same low-dimensional adapter. Consequently, current probes can protect old features when the same adapter weights affect both current and old features. In that case, a weight change that damages old features also changes the current probe features and is penalized by the consistency loss. This does not require the tasks to have identical images, labels, or feature distributions.

Local Justification. Let $\Delta \theta = \theta - \theta ^ { - }$ be one adapter update and let $h _ { \theta } ( x , c _ { j } )$ be the conditioned feature. For a small update,

$$
h _ { \theta } ( x , c _ { j } ) - h _ { \theta ^ { - } } ( x , c _ { j } ) \approx J _ { j } ( x ) \Delta \theta ,\tag{8}
$$

where $J _ { j } ( x )$ is the feature Jacobian with respect to the shared-adapter parameters. The current-input consistency loss and old-input feature drift are locally

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n } } ^ { j } \approx \Delta \boldsymbol { \theta } ^ { \top } G _ { t } ^ { j } \Delta \boldsymbol { \theta } , \qquad D _ { j } \approx \Delta \boldsymbol { \theta } ^ { \top } G _ { j } ^ { j } \Delta \boldsymbol { \theta } , } \end{array}\tag{9}
$$

where

$$
G _ { a } ^ { j } = \mathbb { E } _ { x \sim \mathcal { D } _ { a } } \left[ J _ { j } ( x ) ^ { \top } J _ { j } ( x ) \right] .\tag{10}
$$

If every small adapter change that affects old features is also detected by the current probes, up to a constant $\kappa ,$

$$
G _ { j } ^ { j } \preceq \kappa G _ { t } ^ { j } ,\tag{11}
$$

then

$$
\begin{array} { r } { D _ { j } \leq \kappa \mathcal { L } _ { \mathrm { c o n } } ^ { j } . } \end{array}\tag{12}
$$

Thus, minimizing consistency on current inputs also limits old-task feature drift when current and old inputs overlap and respond to the same adapter changes.

Empirical justification for sufficient overlap. Overlap means that changing the same adapter weights affects both sets of conditioned features in similar ways. It does not require current and old images, or even their frozen-backbone features, to be similar one by one. $J _ { j } ( x )$ records the effect of small adapter-weight changes for one input, and $G _ { t } ^ { j }$ and $G _ { j } ^ { j }$ summarize these effects over current and old inputs.

We test this overlap directly with feature-Jacobian products over all shared-adapter parameters. We use ImageNet-R (Inc10) and analyze a checkpoint obtained while training the second task. At this fixed checkpoint, we construct two input sets. The first contains old inputs, namely images from the first task. The second contains current probes, namely images from the second task. Both sets are processed using the old-task context $c _ { 1 }$ . Therefore, the only difference between $G _ { 1 } ^ { 0 }$ and $G _ { 2 } ^ { 0 }$ is the input distribution,

$$
\begin{array} { r } { G _ { 1 } ^ { 0 } = \mathbb { E } _ { { x } \sim \mathcal { D } _ { 1 } } \left[ J _ { 1 } ( x ) ^ { \top } J _ { 1 } ( x ) \right] , \qquad G _ { 2 } ^ { 0 } = \mathbb { E } _ { { x } \sim \mathcal { D } _ { 2 } } \left[ J _ { 2 } ( x ) ^ { \top } J _ { 2 } ( x ) \right] . } \end{array}\tag{13}
$$

For each image set, we measure how the 768-dimensional conditioned feature responds to all sharedadapter parameters. The auxiliary classification heads are excluded because they do not produce the conditioned feature matched by the consistency loss.

These measurements support the required coverage interpretation. Current-task images, when evaluated under the old-task context, activate most adapter directions that are important for first-task features. The experiment does not claim that old- and current-task images are semantically similar.

We verified that this behavior remains after the full 20-task sequence. At the final checkpoint, we use images from the last task as current probes and compare them with images from old tasks 0, 4, 8, 12, and 16. For each old task $j ,$ both image sets are evaluated under that task’s context $c _ { j }$ . Across these five comparisons, the average similarity between current-task directions and old-task directions is 0.869±0.049, and 99.58% of sampled old-active directions satisfy the directional relation with $\kappa { = } 5$

Table 16: Feature-Jacobian overlap between current probes and old inputs under the old-task context $c _ { 1 }$ on ImageNet-R (Inc10).
<table><tr><td>Measurement</td><td>Result</td><td>Interpretation</td></tr><tr><td>Similarity between changes detected by current and old inputs</td><td>adapter 0.809±0.041</td><td>Normalized similarity between  $G _ { 1 } ^ { 0 }$  and  $G _ { 2 } ^ { 0 }$  A value of 1 means identical sensitivity structure up to scale, and 0 means orthogonal structure. The result indicates substan- tial overlap.</td></tr><tr><td>Rank-16 old-gradient energy cap- tured by current directions</td><td> $8 5 . 2 { \pm } 3 . 5 \%$ </td><td>We learn the 16 strongest adapter directions from current probes and test them on held-out old inputs. They capture 85.2% as much old-input Jacobian variance as directions</td></tr><tr><td>Sampled old-active directions sat- isfying  $v ^ { \top } G _ { 1 } ^ { 0 } v \leq 2 v ^ { \top } G _ { 2 } ^ { 0 } v$ </td><td>96.88%</td><td>learned directly from old inputs. For 96.88% of sampled directions that affect old features, the old-input effect is no more than twice the effect de- tected by current probes.</td></tr><tr><td>Sampled old-active directions sat- isfying  $v ^ { \top } G _ { 1 } ^ { 0 } v \leq 5 v ^ { \top } G _ { 2 } ^ { 0 } v$ </td><td>100%</td><td>Every sampled old-active direction is detected by current probes within a factor of 5.</td></tr></table>