# Towards Zero-Shot Domain Generalization for ID Cards Presentation Attack Detection

<sub>Mario</sub> <sub>Nieto-Hidalgo</sub>1[0000-0003-0623-6455]<sub>,</sub> <sub>Juan</sub> <sub>M.</sub>

<sub>Espin</sub>1[0000-0001-6521-7890]<sub>,</sub> <sub>and</sub> <sub>Juan</sub> <sub>E.</sub> <sub>Tapia</sub>2[0000-0001-9159-4075]

<sup>1</sup> Facephi, R&D Department, Spain. {marionieto,jmespin}@facephi.com 2 da/sec-Biometrics and Internet Security Research Group. Hochschule Darmstadt, Germany. juan.tapia-farias@h-da.de

Abstract. Presentation-Attack Detection (PAD) for national ID cards is limited by the lack of publicly available genuine samples, making it dificult for systems to generalize across countries. This paper introduces two main innovations: (1) a Prototypical Network head using an EficientNet-V2-b0 backbone that requires only four genuine samples per class to create reliable prototypes; and (2) an episodic training regime that keeps PAD classes fixed while varying the card domain, allowing the network to learn universal attack cues. Evaluated on a large multi-country dataset and the public DLC-2021 benchmark, this method achieves an average Equal Error Rate of around 9%, outperforming conventional softmax and CLIP zero-shot baselines even with data from a single source country. This approach provides accurate, privacy-preserving PAD while minimizing data collection, facilitating scalable cross-jurisdictional remote onboarding.

Keywords: PAD ID cards · Zero-shot · Few-shot.

## 1 Introduction

Growing concerns about biometric companies in remote verification processes have increased the popularity of fully online onboarding systems. The popularity of these systems is mainly due to their user-friendly enrollment and the widespread adoption of smartphones. Users can enroll directly from home by taking a selfie or a photo of their ID card, without needing to go to a store or an ofice. These factors increase the number of users willing to enroll. However, this system carries a high risk of fraud, as users might attempt to exploit it. The easiest way to fool these systems is with Presentation Attacks (PAs), in which the fraudster impersonates another subject using a printed photo or a screen replay and presents it to the capture device. These systems prevent this type of fraud through PAD algorithms. Today, most solutions are based on algorithms, usually Deep Learning (DL) networks, trained on large amounts of images to distinguish high-frequency details in PAs, such as the Moiré patterns of screen-replay attacks, printed photo textures, etc.

The State-of-The-Art (SoTA) face PAD algorithms show good performance due to the large number of open-set databases available. However, they struggle somewhat with generalization to unseen capture devices or Presentation Attack Instruments (PAI). However, PAD systems for ID cards remain in their infancy, primarily due to the dificulty of obtaining suficient bona fide samples [12] and the lack of open-set databases. The ID cards contain sensitive information used to identify subjects; therefore, genuine bona fide samples cannot be published. This is why all publicly available datasets are used as bona fide samples of synthetic or simulated ID cards printed on Polyvinyl Chloride Plastic (PVC). The simulated bona fide samples (synthetic) are insuficient for research and for a production-ready PAD system.

The first competition on PAD on ID card was organized in the last International Joint Conference on Biometrics 2024 [12]. The results showed that the submitted algorithms lacked generalization capabilities. The 2025 edition of this challenge [13] showed improvements in the generalization of the algorithms tested, with one team surpassing the baseline. However, it remains challenging because the baseline and the best submitted algorithm difered only slightly when evaluated across four countries.

One of the main drawbacks of PAD algorithms for ID cards is their poor generalization to unseen document types or country versions [7]. Each country has its own ID card template, and multiple versions may coexist during certain periods. Each template varies significantly in its background, typography, layout, and other elements. This heterogeneity afects the performance of PAD systems, making generalization dificult. In Sanchez et al. [7], meta-learning techniques based on Few-shot learning were explored. It was suggested that at least 100 bona fide ID card users and their respective attacks were needed to obtain good performance in a new type of ID. However, even 100 bona fide users are dificult to obtain due to privacy concerns, and generating the corresponding attacks requires significant time and efort. Therefore, it is crucial to explore alternatives that could reduce this number.

Several open-set datasets are available on the SoTA. Some of the open-set datasets available are MIDV-500 [1], MIDV-2020 [2], MIDV-Holo [4], DLC-2021 [5], and SynIDPass [14], which usually contain a small number of synthetic users, printed on PVC, which is not enough to train a robust production-ready PAD system.

Motivated by the previous challenge, we proposed a new method toward Zero-shot based on Few-shot learning to predict on ID cards from a new country using pretrained features extracted from a PAD representing a diferent country. The main contributions of this work are:

– A Few-Shot approach is proposed to reduce the number of images needed to extend PAD to other countries.

– A comparison with a Zero-shot foundation model was explored to extend the generalization capabilities

– Our method was evaluated on private and open-set datasets using IDs from diferent countries.

The remainder of the article is organized as follows: Section 2 summarizes related works on PAD and FSL; Section 3 describes the proposed method; Section 4 describes the experimentation performed to test the proposed method; finally, Section 5 discusses the results and future work.

## 2 Related work

Few-shot learning (FSL) has emerged as an efective learning method with considerable potential [9]. Despite recent creative eforts to tackle FSL tasks, rapidly learning valid information from just a few or even zero samples remains a significant challenge.

Song et al. [9] propose and analyze, in detail, a survey of more than 200 works that have raised the open challenge of FSL in recent years, due to the restrictions and limited access to data imposed by privacy preservation issues.

The FSL approaches usually work with two sets of samples: a query set containing samples from unknown classes and a support set containing samples from known classes. The FSL approaches use the support set as a source of information to infer the class of each sample of the query set. Depending on the constitution of the support set, the nomenclature N-way-K-shot is used where N is the number of classes and K the number of samples of each class in the support set. When K = 0, it is called Zero-Shot learning; otherwise, the term K-Shot or Few-Shot is used.

In the literature, we found two main approaches based on FSL theory. One of them is the Relation Network (RL) [10], and the other is the Prototypical Network (PN) [8], which stands out for its simplicity.

Both networks are based on K-Nearest Neighbors (KNN), but with the advantage of being trainable end-to-end. The KNN can be considered as an FSL approach that computes the distance between the query sample and all support samples, returning the most common class among the K closest support samples. The problem with KNN is that its efectiveness depends on the discriminatory power of the embeddings used to compute the distance. The embeddings usually come from a previously trained backbone, and they are traditionally trained to optimize a Multi-Layer Perceptron (MLP) classifier, which produces an embedding space that may not be optimal for computing distances between samples of unknown class. In addition, traditional KNN requires lots of samples of each class to work correctly.

Both RL and PN try to solve these issues by providing an end-to-end trainable KNN. The RL replaces the MLP for a module that learns a distance function to optimize the discriminatory power of both the embeddings and the distance function.

On the other hand, PN replaces the MLP with a one-hot similarity score layer with a softmax as a loss function to optimize the distance between query samples and a prototype of each class. This prototype can be computed as the average embedding of the class. The ability of both networks to be trained endto-end, along with training that uses a diferent subset of classes per batch, enables them to perform well on unseen classes.

Sanchez et al. [7] used the Prototypical Network to improve generalization with a few samples. The Prototypical Network is a metric-based FS learner trained to assign samples to their class prototypes. This prototype is computed by averaging the embeddings of a few samples of the class. This average, along with the training process in which the prototypes are changed in each batch, allows the network to generalize to new prototypes. In that work, additional samples of new ID card types were incrementally added to the training set to establish a benchmark for the PN ability to learn from a few samples. That work concludes that at least 100 distinct bona fide users and their respective attacks were required to achieve a good performance.

In addition, architectures based on foundation models such as (Contrastive Language–Image Pre-training) CLIP [6] have revolutionized the Zero-Shot scene. The CLIP is a family of vision-language models trained on large-scale image-text pairs to learn joint representations of visual and textual data. The $C L I P  – B / 1 6$ model uses a $1 6 \times 1 6$ patch size, making it a medium-sized model that balances performance and eficiency. It is trained on 400 million image-text pairs from the internet and excels at zero-shot classification, image retrieval, and transfer learning. In contrast, $C L I P  – L / 1 4$ uses a $V i T – L / 1 4$ (Vision Transformer Large, 14 × 14 patches) architecture, which is significantly larger and more powerful. With over 1 billion parameters, it utilizes more complex visual and semantic relationships, achieving SoTA performance.

The CLIP is a Vision Transformer architecture (ViT) [3] based network that has been trained to compare the embedding generated by an image and its text description. This enables Zero-Shot by computing Cosine Similarity (CS) between the textual class descriptions and the images of those classes.

## 3 Proposed method: Few-Shot learning approach

In previous work, such as Sanchez et al. [7], the authors aim to expand into new countries or domains with reduced data use. In that work, the authors indicate that 100 bona fide samples and their attacks are needed to obtain competitive results. These 100 samples, in an ID card problem, aren’t a small number.

Our approach is based on the premise that identifying the ID card type is easier than the PAD itself, so we assume that there is an ID card type classifier that selects the appropriate support set. For example, we tested an approach that is trained on diferent types of ID cards, driving licenses, and passports. This classifier uses the first half of an EficientNetV2-b0 [11] backbone to generate embeddings. Those embeddings are then classified using a KNN that performs cosine distance to the embedding of the document template or specimen. This approach achieves an average accuracy of 98.5% across more than 150 diferent documents and 99.8% on the ID cards tested in this work. That classifier was trained with both real and synthetic samples. Although the classifier is out-ofscope for this work, it is essential to note that we could train one using only synthetic samples or specimens.

![](images/15cb8f775f6f5f2e7ff4a4ae80e2f0ba228cce802ee77ac02b21647d7107dd05.jpg)  
Fig. 1: Diagrams for baseline method (a), Prototypical Network train stage (b), and Prototypical Network inference stage (c).

A diagram of our method is shown in Figure 1. First of all, 1a, the baseline is shown, with a backbone (marked with an E in Figure), where an embedding is extracted by concatenating the Global Average Pooling (GAP) and Global Max Pooling (GMP) of the backbone output. At the head, a hidden layer and a Softmax Classifier Layer with four classes are present: bona fide, print, screen, and PVC, respectively.

To build our system, we modify the baseline by replacing the hidden layer with a Prototypical Layer. According to the N-Way-K-Shot nomenclature, our proposed method is a 4-Way-K-Shot because we have 4 classes. The K is defined in preliminary experiments, which couldn’t be exposed in this work due to length limitations. We defined a support set of 4 samples per class. Prototypes were built by averaging the embeddings of 4 samples per class of the target ID card type.

In the experiments, fewer samples (one, two, or three) showed high dependence on the selected samples, and with 4 samples, this is the lowest number at which the results are stable. We selected the lowest because we are in a scenario where bona fide samples are limited. In that case, our method is a $\it 4 - W a y - 4 - S h o t$ method that uses only 16 samples: 4 bona fide samples and their respective attacks. In our method, the model never uses any samples of the target ID card type during training; it only uses them during inference to build the prototypes. Unlike [7], where the samples are used to fine-tune the model and to build the prototypes together.

## 3.1 Training Stage

Our approach proposes improving the number of images used in the previously proposed method by leveraging Few-Shot training. We use episodic training. An episodic task is a single, self-contained training task with its own start and end, such as a game match or a specific classification problem. Train a model to recognize these particular classes using only the support set. To test, evaluate the model on new examples (query set) of those same classes, then create new episodes with diferent classes and examples, and repeat the process, allowing the model to learn how to learn across many other, small-scale challenges.

In our case, each batch comprises the same number of samples per class (bona fide, print, screen, and PVC) but includes only a single ID card type. This is similar to classic Few-shot episodic training; however, instead of changing the classes in each episode, we change the domain. This enhances the Prototypical Network’s ability to transfer across domains simply by changing the prototypes. The Prototypical head uses the Euclidean distance to compute the logits function. The Figure 1b shows the diagram of the PN training process.

## 3.2 Inference Stage

During inference, we rely on an ID card type classifier to select the prototypes for the current sample. To work with a new ID card type, we will need samples of each class to build the prototypes: 4 bona fide and their respective attacks. These prototypes can be precomputed and stored in a database to be fetched on demand. Figure 1c shows the inference process.

## 3.3 Configuration and hyperparameters

We used a backbone based on EficientNet-V2-b0 as a SoTA [7]. Therefore, we had an embedding of size 1 × 2, 560, as shown in Figure 1.

We used the unaligned card crop, resized to 224 × 224 as input. A batch size of 84 query samples (21 samples per class) and 16 support samples (4 samples per class). We used 4 class-output classifiers during training and validation with the softmax activation. However, we used the bona fide class as a binary classifier during inference. Regarding hyperparameters, we used AdamW as the optimizer with a learning rate of 5e − 4. We trained for up to 100 pseudo-epochs of 200 steps, with early stopping over the last 10 epochs.

## 3.4 Databases and Metrics

The databases used to evaluate and test the proposed methodology are described below. Two databases were used: a private database to enable extensibility to production systems and a public database to replicate the results of this work. For this purpose, to evaluate the performance of the proposed PAD method, the metrics described in $\mathrm { I S O / I E C ~ 3 0 1 0 7  – 3 ^ { 3 } }$ are to be used. The Equal Error Rate (EER), Bona fide Presentation Classification Error Rate (BPCER), and Attack Presentation Classification Error Rate (APCER). To further evaluate performance under fixed security constraints, the metric BPCER at fixed APCER $\left( \mathrm { B P C E R } _ { A P } \right)$ is employed.

The BPCER measures the proportion of bona fide presentations that are wrongly classified as attacks. The following expression defines it:

$$
B P C E R = \frac { 1 } { N _ { B F } } \sum _ { i = 1 } ^ { N _ { B F } } R E S _ { i } ,\tag{1}
$$

where $N _ { B F }$ is the number of bona fide presentation samples, and $R E S _ { i } = 1$ if the sample is incorrectly flagged as an attack.

The APCER is defined as:

$$
A P C E R _ { P A I S } = 1 - \frac { 1 } { N _ { P A I S } } \sum _ { i = 1 } ^ { N _ { P A I S } } R E S _ { i } ,\tag{2}
$$

where $N _ { P A I S }$ represents the total number of attack images for a given PAIS, and $R E S _ { i } = 1$ if the $i - t h$ image is correctly identified as an attack, or 0 if it is misclassified as bona fide.

Private Dataset A private database with diferent ID card types and bona fide, paper print, screen replay, and PVC print classes is used to test our approach. Table 1 shows the diferent ID card types and the number of users of each one. As seen there, the Chilean ID card type CHL-2 has significantly more users than the others.

![](images/b2724ff2b63ba82207188b2ca57c7d65cce0c5d1c03ae6d9180ce93de953a4b3.jpg)

(a) Example images from CHL-1, CHL-2 and Guatemala  
![](images/5718fcd42ed82ceb6fb28f60be95f130d5075b96ba617a70dee50d2d25d9fadf.jpg)  
(b) Example images from Mexico-1, Mexico-2 and Panama.  
Fig. 2: Example of images from a private dataset.

Public dataset DLC-2021 To ensure reproducibility of the proposed experiments, we have replicated the experiments using a publicly available dataset. The chosen dataset is DLC-2021 [5], which contains synthetic samples of different ID card types and passports with their corresponding screen and print attacks. We used only the ID card samples from Albania (ALB), Spain (ESP), Estonia (EST), Finland (FIN) and Slovakia (SVK).

Table 1: Number of samples per ID card type.
<table><tr><td></td><td>PAN COL CHL-1</td><td></td><td></td><td></td><td></td><td>CHL-2 MEX-1 MEX-2 GTM</td><td></td></tr><tr><td>Bona fide 1,006</td><td></td><td>80</td><td>696</td><td>13,000</td><td>353</td><td>660</td><td>1,001</td></tr><tr><td></td><td>Print 1,006</td><td>80</td><td>696</td><td>13,000</td><td>353</td><td>660</td><td>1,001</td></tr><tr><td>Screen 1,006</td><td></td><td>80</td><td>696</td><td>13,000</td><td>353</td><td>660</td><td>1,001</td></tr><tr><td>PVC</td><td>506</td><td>80</td><td>259</td><td>6,500</td><td>353</td><td>160</td><td>501</td></tr></table>

## 4 Experiments and Results

## 4.1 Experiment 0: CLIP Zero-Shot

Firstly, we evaluated CLIP Zero-Shot capabilities to compare them with our approach. We performed some evaluations using only the ID card for most samples, which is CHL-2, to adjust the text prompts and determine which CLIP backbone yields better results. The best performing configuration was used for comparison with our approach in both experiments.

We tested the backbone CLIP-B16 and CLIP-L14 from OpenAI, following these simple text prompts:

– Bona fide: “a photo of an ID card"

– Print: “a printed photo of an ID card"

– Screen: "a photo of an ID card shown on a screen device",

– PVC: "a photo of an ID card printed on a PVC card",

Because a PVC-printed ID card might be mistaken for the class-printed one, we also tested without that text class to see whether the results improved. Results are shown in the Table 2. We observe improvements in both B − 16 and L − 14 backbones when only 3 text classes are used. The backbone B − 16 also outperforms the backbone L − 14 on 3 classes. Therefore, we used B − 16 with only 3 text classes as the Zero-Shot approach.

Table 2: Experiment 0: CLIP test with CHL-2 in private dataset.
<table><tr><td>Model/Classes</td><td></td><td>EER% BPCER10 BPCER20 BPCER100</td><td></td></tr><tr><td>CLIP-B16 4C</td><td>30.53</td><td>66.66</td><td>81.63 95.24</td></tr><tr><td>CLIP-L14 4C</td><td>30.86</td><td>65.31</td><td>78.44 94.21</td></tr><tr><td>CLIP-B16 3C</td><td>20.71</td><td>41.52</td><td>60.14 88.06</td></tr><tr><td>CLIP-L14/ 3C</td><td>27.07</td><td>51.33</td><td>65.63 86.33</td></tr></table>

## 4.2 Experiment 1: Single domain training

In this experiment, we used the most common ID card type, CHL-2, with most samples to train both the baseline and prototypical networks. In this case, the episodic training consists of keeping the same classes and the same ID card type, but changing the samples of the support set in each episode. The goal of this experiment is to assess the generalization capabilities to unseen ID Card types when only a single ID card type is available during training. This is the most challenging case.

The results shown in Table 3 indicate that the baseline does not generalize to unseen ID card types. However, the Prototypical model shows better generalization. The best-performing ID card type for the baseline is PAN, which achieves an EER of 12.7% and a BPCER10 of 17.5% while the prototypical reaches 8.44% and 7.06%, respectively. On average, the baseline achieves an EER of 26.56% and a BPCER10 of 45.69%, while the prototypical achieves 9.07% and 10.15%, respectively, which is substantially better.

Comparing both to Zero-Shot CLIP-B16, we can observe that CLIP performs better than the baseline but worse than the prototypical approach.

Table 3: Experiment 1 on the private dataset in (%).
<table><tr><td colspan="5">EER% BPCER10 BPCER20 BPCER100</td></tr><tr><td colspan="5">Baseline</td></tr><tr><td>PAN</td><td>12.70</td><td>17.50</td><td>31.11</td><td>56.96</td></tr><tr><td>COL</td><td>38.54</td><td>66.25</td><td>75.00</td><td>92.5</td></tr><tr><td>CHL-1</td><td>21.35</td><td>36.64</td><td>40.80</td><td>71.84</td></tr><tr><td>MEX-1</td><td>56.80</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>MEX-2</td><td>56.80</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>GTM</td><td>16.73</td><td>33.87</td><td>48.55</td><td>65.83</td></tr><tr><td>AVG</td><td>26.56</td><td>45.69</td><td>53.79</td><td>75.96</td></tr><tr><td>STD</td><td>17.62</td><td>31.80</td><td>28.28</td><td>16.65</td></tr><tr><td colspan="5">Zero-Shot CLIP-B16</td></tr><tr><td>PAN</td><td>18.31</td><td>27.53</td><td>42.94</td><td>66.30</td></tr><tr><td>COL</td><td>22.29</td><td>38.75</td><td>47.5</td><td>78.75</td></tr><tr><td>CHL-1</td><td>14.79</td><td>22.27</td><td>36.21</td><td>65.80</td></tr><tr><td>MEX-1</td><td>8.17</td><td>7.08</td><td>13.31</td><td>34.28</td></tr><tr><td>MEX-2</td><td>17.37</td><td>28.33</td><td>44.70</td><td>77.58</td></tr><tr><td>GTM</td><td>11.04</td><td>12.19</td><td>23.88</td><td>65.03</td></tr><tr><td>AVG</td><td>15.33</td><td>22.69</td><td>34.76</td><td>64.62</td></tr><tr><td>STD</td><td>5.13</td><td>11.55</td><td>13.48</td><td>16.08</td></tr><tr><td colspan="5">Prototypical Network</td></tr><tr><td>PAN</td><td>8.44</td><td>7.06</td><td>14.21</td><td>36.18</td></tr><tr><td>COL</td><td>1.25</td><td>0.00</td><td>0.00</td><td>1.25</td></tr><tr><td>CHL-1</td><td>12.93</td><td>19.68</td><td>36.49</td><td>72.99</td></tr><tr><td>MEX-1</td><td>10.76</td><td>12.75</td><td>25.21</td><td>58.92</td></tr><tr><td>MEX-2</td><td>11.37</td><td>12.42</td><td>25.76</td><td>63.18</td></tr><tr><td>GTM</td><td>9.69</td><td>8.99</td><td>24.28</td><td>54.95</td></tr><tr><td>AVG</td><td>9.07</td><td>10.15</td><td>20.99</td><td>47.91</td></tr><tr><td>STD</td><td>4.12</td><td>6.58</td><td>12.48</td><td>25.88</td></tr></table>

## 4.3 Experiment 2: Multiple domain training

In this experiment, we performed Leave-One-Out (LOO) cross-validation on CHL-2, PAN, COL, and MEX-1, in which we trained both the baseline and the prototypical model using N − 1 ID card types and held out the remaining ID card type for testing. In this case, episodic training keeps the same classes while varying the ID Card type. The goal of this experiment is to assess the generalization capabilities when multiple ID card types are available during training.

Observing the results in Table 4, we can see that the advantage of the prototypical is reduced. In this experiment, the best-performing ID card type for the baseline is PAN, which achieves an EER of 3.06% and a BPCER10 of 0.6%, surpassing the prototypical, which achieves only 5.67% and 1.49%, respectively. However, on average, the prototypical outperforms the baseline: the baseline achieves an EER of 11.98% and a BPCER10 of 16.95%, while the prototypical achieves 7.41% and 5.88% respectively.

Comparing with the Zero-Shot CLIP-B16, we can observe that both the baseline and the prototypical now perform better. The only exception is in MEX-1, where CLIP outperforms both of them.

Table 4: Experiment 2 on the private dataset in (%).
<table><tr><td colspan="4">EER% BPCER10 BPCER20 BPCER100</td></tr><tr><td colspan="4">3.06</td></tr><tr><td>PAN COL</td><td>14.17</td><td>0.60 20.00</td><td>1.69 36.25</td></tr><tr><td>CHL-2</td><td>11.95</td><td>14.91</td><td>66.25</td></tr><tr><td>MEX-1</td><td>18.74</td><td>32.29</td><td>63.37 81.87</td></tr><tr><td></td><td></td><td>45.04 28.08</td><td></td></tr><tr><td>AVG STD</td><td>11.98 6.58</td><td>16.95 13.12</td><td>54.61 32.79</td></tr><tr><td colspan="4">Zero-Shot CLIP-B16</td></tr><tr><td>PAN 18.31</td><td>27.53</td><td>42.94</td><td>66.30</td></tr><tr><td>COL</td><td>22.29</td><td>47.5</td><td>78.75</td></tr><tr><td>CHL2</td><td>20.71 41.52 7.08</td><td>60.14 13.31</td><td>88.06</td></tr><tr><td>MEX1</td><td>8.17</td><td></td><td>34.28</td></tr><tr><td>AVG</td><td>17.37</td><td>28.72 15.64</td><td>66.85</td></tr><tr><td>STD</td><td>6.35</td><td>19.82</td><td>23.47</td></tr><tr><td colspan="4">Prototypical Network</td></tr><tr><td>PAN</td><td>5.67</td><td></td><td>25.65</td></tr><tr><td>COL</td><td>3.54</td><td>0.00 39.31</td><td>12.50</td></tr><tr><td>CHL-2</td><td>11.86</td><td>0.00 16.34</td><td>81.10</td></tr><tr><td>MEX-1</td><td>8.55</td><td>5.67</td><td>63.46</td></tr><tr><td>AVG</td><td>7.41</td><td>5.88</td><td>45.68</td></tr><tr><td>STD</td><td>3.61</td><td>7.38</td><td>16.33 17.34</td></tr></table>

## 4.4 Experiments with the public dataset - DLC-2021

Now, the same experiments are replicated using the DLC-2021 dataset.

Experiment 0 For CLIP, we tested the same backbones with ESP ID cards using only 3-class text prompts because, in DLC-2021, the bona fide is a PVCprinted ID card. We tested two sets of prompts: one with the bona fide prompt as the bona fide class, and another with the PVC prompt as the bona fide class. The results in Table 5 show that using the bona fide prompt (bf-prompt) produces the worst results compared to using the bona fide PVC-prompt, especially in BPCER100. The CLIP-B16 backbone is the best in BPCER100; however, in all cases, performance is quite low compared to the results obtained on the private dataset. This is because the bona fide samples of DLC-2021 are PVCprinted ID cards, which are often confused with the Print class text prompt. We chose the CLIP-B16 backbone using the PVC prompt to compare with our approach.

Table 5: CLIP tests (%) with ESP ID cards in DLC-2021.
<table><tr><td colspan="5">EER% BPCER10 BPCER20 BPCER100</td></tr><tr><td>B16-bf-Prompt</td><td>49.37</td><td>93.39</td><td>97.82</td><td>100.00</td></tr><tr><td>L14-bf-Prompt</td><td>38.05</td><td>89.99</td><td>95.96</td><td>99.42</td></tr><tr><td>B16-bf-PVC</td><td>41.81</td><td>81.33</td><td>90.64</td><td>96.92</td></tr><tr><td>L14-bf-PVC</td><td>42.67</td><td>89.16</td><td>96.47</td><td>99.17</td></tr></table>

Experiment 1 Replicating experiment 1 using the DLC-2021 database, the results are similar to those obtained with our private dataset, as shown in Table 6. In Experiment 1, the baseline does not generalize to unseen ID cards, with an average EER of 27.8%, while the prototypical achieves an average EER of 10.53%.

Table 6: Experiment 1 on the DLC-2021 dataset in (%).
<table><tr><td colspan="5">EER% BPCER10 BPCER20 BPCER100</td></tr><tr><td colspan="5">Baseline</td></tr><tr><td>ALB</td><td>26.91</td><td>53.43</td><td>70.32</td><td>85.92</td></tr><tr><td>EST</td><td>21.34</td><td>35.65</td><td>44.32</td><td>68.44</td></tr><tr><td>FIN</td><td>27.12</td><td>53.22</td><td>64.15</td><td>82.57</td></tr><tr><td>SVK</td><td>35.84</td><td>71.78</td><td>83.96</td><td>94.13</td></tr><tr><td>AVG</td><td>27.8</td><td>53.77</td><td>65.69</td><td>82.77</td></tr><tr><td>STD</td><td>5.99</td><td>15.17</td><td>16.48</td><td>10.71</td></tr><tr><td colspan="5">Zero-Shot CLIP-B16</td></tr><tr><td>ALB</td><td>38.13</td><td>76.65</td><td>88.54</td><td>98.35</td></tr><tr><td>EST</td><td>41.96</td><td>87.18</td><td>94.14</td><td>98.90</td></tr><tr><td>FIN</td><td>39.71</td><td>75.70</td><td>83.51</td><td>93.63</td></tr><tr><td>SVK</td><td>36.66</td><td>84.15</td><td>91.59</td><td>98.90</td></tr><tr><td>AVG</td><td>39.12</td><td>80.92</td><td>89.45</td><td>97.45</td></tr><tr><td>STD</td><td>2.27</td><td>5.63</td><td>4.57</td><td>2.56</td></tr><tr><td colspan="5">Prototypical Network</td></tr><tr><td>ALB</td><td>13.4</td><td>18.18</td><td>33.6</td><td>77.85</td></tr><tr><td>EST</td><td>5.98</td><td>0.92</td><td>8.24</td><td>35.59</td></tr><tr><td>FIN</td><td>9.99</td><td>9.74</td><td>25.8</td><td>51.34</td></tr><tr><td>SVK</td><td>12.73</td><td>17.35</td><td>31.77</td><td>89.7</td></tr><tr><td>AVG</td><td>10.53</td><td>11.55</td><td>24.85</td><td>63.62</td></tr><tr><td>STD</td><td>3.37</td><td>8.04</td><td>11.56</td><td>24.62</td></tr></table>

Experiment 2 In Experiment 2 and Table 7, the advantage of the Prototypical is reduced; the average EER is 10.27% for the baseline and 9.12% for the prototypical. In this case, the advantage of the Prototypical is lost when setting the threshold to BPCER100, where the baseline achieves an average of 45.15% while the prototypical reaches 61.6%.

Comparing results with CLIP-B16, we can observe that CLIP does not perform well with DLC-2021. Both the baseline and the prototypical models significantly outperform CLIP, which achieves an average EER of 39.65%.

Table 7: Experiment 2 on the DLC-2021 dataset in (%).
<table><tr><td colspan="5">EER% BPCER10 BPCER20 BPCER100</td></tr><tr><td colspan="5">Baseline</td></tr><tr><td>ALB</td><td>7.97</td><td>6.92</td><td>12.67</td><td>47.43</td></tr><tr><td>ESP</td><td>5.29</td><td>1.78</td><td>5.47</td><td>29.09</td></tr><tr><td>EST</td><td>2.95</td><td>0.31</td><td>0.67</td><td>7.88</td></tr><tr><td>FIN</td><td>20.10</td><td>36.41</td><td>49.34</td><td>82.26</td></tr><tr><td>SVK</td><td>15.05</td><td>21.16</td><td>33.52</td><td>59.11</td></tr><tr><td>AVG</td><td>10.27</td><td>13.32</td><td>20.34</td><td>45.15</td></tr><tr><td>STD</td><td>7.13</td><td>15.32</td><td>20.5</td><td>28.39</td></tr><tr><td colspan="5">Zero-Shot CLIP-B16</td></tr><tr><td>ALB</td><td>38.13</td><td>76.65</td><td>88.54</td><td>98.35</td></tr><tr><td>ESP</td><td>41.81</td><td>81.33</td><td>90.64</td><td>96.62</td></tr><tr><td>EST</td><td>41.96</td><td>87.18</td><td>94.14</td><td>98.90</td></tr><tr><td>FIN</td><td>39.71</td><td>75.70</td><td>83.51</td><td>93.63</td></tr><tr><td>SVK</td><td>36.66</td><td>84.15</td><td>91.59</td><td>98.90</td></tr><tr><td>AVG</td><td>39.65</td><td>81.00</td><td>89.68</td><td>97.34</td></tr><tr><td>STD</td><td>2.31</td><td>4.88</td><td>3.99</td><td>2.23</td></tr><tr><td colspan="5">Prototypical Network</td></tr><tr><td>ALB</td><td>10.28</td><td>10.34</td><td>16.95</td><td>84.33</td></tr><tr><td>ESP</td><td>4.55</td><td>0.43</td><td>3.26</td><td>74.05</td></tr><tr><td>EST</td><td>5.37</td><td>2.93</td><td>6.04</td><td>31.99</td></tr><tr><td>FIN</td><td>8.87</td><td>7.18</td><td>16.43</td><td>42.91</td></tr><tr><td>SVK</td><td>16.54</td><td>24.84</td><td>41.76</td><td>74.72</td></tr><tr><td>AVG</td><td>9.12</td><td>9.15</td><td>16.89</td><td>61.6</td></tr><tr><td>STD</td><td>4.78</td><td>9.57</td><td>15.18</td><td>22.75</td></tr></table>

## 5 Conclusion and future work

We have shown that it is possible to generalize to unseen ID card types using only four bona fide users and their respective attacks. The experiments show that our approach outperforms the baseline considerably when only one ID card type is available during training. However, in Experiment 2, where more ID card types are available during training, this advantage is reduced. Even though the Prototypical still outperforms the baseline. The need for an ID card type classifier can sometimes be problematic. Therefore, we recommend using our Prototypical approach when only a few ID card samples (four) are available during training; when there are suficient ID card types, the baseline’s generalization capabilities are adequate, so there is no clear advantage to using a more complex architecture such as the Prototypical.

One of the main drawbacks of this approach is the need for bona fide and attack samples. To address this issue, we will use an anomaly detection approach in which we train the prototypical model to determine whether a sample is similar to the bona fide prototype. This could reduce the samples required to only the bona fides of the target ID card. Furthermore, we will investigate the possibility of using only specimen samples as a prototype, so we will not even need any bona fide users.

Acknowledgements This research has been partially funded by Facephi, R&D department, and the European Union’s Horizon research and innovation program under grant agreements 101121280 (EINSTEIN) and the German Federal Ministry of Education and Research, and the Hessian Ministry of Higher Education, Research, Science, and the Arts, which jointly support the National Research Center for Applied Cybersecurity ATHENE.

## References

1. Arlazarov, V., Bulatov, K., Chernov, T., Arlazarov, V.: MIDV-500: a dataset for identity document analysis and recognition on mobile devices in video stream. Computer Optics 43(5) (Oct 2019). https://doi.org/10.18287/2412-6179-2019-43- 5-818-824, http://dx.doi.org/10.18287/2412-6179-2019-43-5-818-824

2. Bulatov, K.B., Emelianova, E.V., Tropin, D.V., Skoryukina, N.S., Chernyshova, Y.S., Sheshkus, A.V., Usilin, S.A., Ming, Z., Burie, J.C., Luqman, M.M., Arlazarov, V.V.: MIDV-2020: a comprehensive benchmark dataset for identity document analysis. Computer Optics 46(2), 252–270 (Apr 2022)

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 Words: Transformers for image recognition at scale. In: 9th Int. Conf. on Learning Representations, ICLR 2021, Austria, May 3-7. OpenReview.net (2021)

4. Koliaskina, L.I., Emelianova, E.V., Tropin, D.V., Popov, V.V., Bulatov, K.B., Nikolaev, D.P., Arlazarov, V.V.: MIDV-Holo: A dataset for id document hologram detection in a video stream. In: Fink, G.A., Jain, R., Kise, K., Zanibbi, R. (eds.) Document Analysis and Recognition - ICDAR 2023. pp. 486–503. Springer Nature Switzerland, Cham (2023)

5. Polevoy, D.V., Sigareva, I.V., Ershova, D.M., Arlazarov, V.V., Nikolaev, D.P., Ming, Z., Luqman, M.M., Burie, J.C.: Document liveness challenge dataset (dlc-2021). Journal of Imaging 8(7) (2022). https://doi.org/10.3390/jimaging8070181, https://www.mdpi.com/2313-433X/8/7/181

6. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/radford21a.html

7. Sanchez, A., Espín, J.M., Tapia, J.E.: Few-shot learning: Expanding ID cards presentation attack detection to unknown ID countries. In: 2024 IEEE International Joint Conference on Biometrics (IJCB). pp. 1–9 (2024). https://doi.org/10.1109/IJCB62174.2024.10744501

8. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 4080–4090. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017)

9. Song, Y., Wang, T., Cai, P., Mondal, S.K., Sahoo, J.P.: A comprehensive survey of few-shot learning: Evolution, applications, challenges, and opportunities. ACM Comput. Surv. 55(13s) (Jul 2023). https://doi.org/10.1145/3582688, https://doi.org/10.1145/3582688

10. Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P.H., Hospedales, T.M.: Learning to compare: Relation network for few-shot learning. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1199–1208 (2018). https://doi.org/10.1109/CVPR.2018.00131

11. Tan, M., Le, Q.: Eficientnetv2: Smaller models and faster training. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 10096–10106. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/tan21a.html

12. Tapia, J.E., Damer, N., Busch, C., Espin, J.M., Barrachina, J., Rocamora, A.S., Ocvirk, K., Alessio, L., Batagelj, B., Patwardhan, S., Ramachandra, R., Mudgalgundurao, R., Raja, K., Schulz, D., Aravena, C.: First competition on presentation attack detection on ID card. In: 2024 IEEE International Joint Conference on Biometrics (IJCB). pp. 1–10 (2024). https://doi.org/10.1109/IJCB62174.2024.10744475

13. Tapia, J.E., Nieto, M., Espin, J.M., Rocamora, A.S., Barrachina, J., Damer, N., Busch, C., Ivanovska, M., Todorov, L., Khizbullin, R., Lazarevich, L., Grishin, A., Schulz, D., Gonzalez, S., Mohammadi, A., Kotwal, K., Marcel, S., Mudgalgundurao, R., Raja, K., Schuch, P., Patwardhan, S., Ramachandra, R., Couto Pereira, P., Pinto, J.R., Xavier, M., Valenzuela, A., Lara, R., Batagelj, B., Peterlin, M., Peer, P., Muhammed, A., Nunes, D., Gonçalves, N.: Second competition on presentation attack detection on ID card (2025). https://doi.org/10.1109/IJCB65343.2025.11411073

14. Tapia, J.E., Stockhardt, F., González-Soler, L.J., Busch, C.: Syn-idpass: Passport synthetic dataset for presentation attack detection. In: 2025 IEEE International Joint Conference on Biometrics (IJCB). pp. 1–9 (2025). https://doi.org/10.1109/IJCB65343.2025.11411263