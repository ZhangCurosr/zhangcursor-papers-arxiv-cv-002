# VISION AND TEXT TRANSFORMER FOR PREDICTING ANSWERABILITY ON VISUAL QUESTION ANSWERING

PREPRINT

Tung Le\* <sup>†</sup> lttung@jaist.ac.jp

Huy Tien Nguyen<sup>†</sup> <sup>‡</sup> ntienhuy@fit.hcmus.edu.vn

Minh Le Nguyen\* nguyenml@jaist.ac.jp

September 16, 2026

## ABSTRACT

Answerability on Visual Question Answering is a novel and attractive task to predict answerable scores between images and questions in multi-modal data. Existing works often utilize a binary mapping from visual question answering systems into Answerability. It does not reflect the essence of this problem. Together with our consideration of Answerability in a regression task, we propose VT-Transformer, which exploits visual and textual features through Transformer architecture. Experimental results on VizWiz 2020 dataset show the effectiveness and robustness of VT-Transformer for Answerability on Visual Question Answering when comparing with competitive baselines.

Keywords Answerability · Visual Question Answering · VizWiz · Vision Transformer · Multi-head Attention

## 1 Introduction

In the rapid increasing of multi-modal information like videos, images, and texts, it has arisen a trend of interdisciplinary studies in Vision and Language. The goals of those researches are to reveal the relationship between visual and textual information to support multi-modal tasks such as Visual Question Answering [1, 2], Visual Commonsense Reasoning [3], Image Captioning [4] and so on.

In this work, we focus on an attractive and novel task, Answerability on Visual Question Answering (VQA), which is recently proposed in a contest of VQA for blind people [5, 6] in 2018. Generally, Answerability is to determine a score that reflects the answerable ability of samples. At first glance, we wonder whether this task is well worth our researches or not. It is useful and practical enough to help us in deciding to answer a question from users. If we can regard a sample as an unanswerable one, we can quickly respond to users instead of running a completed VQA model. It is beneficial to save computational cost in question answering phases.

Answerability belongs to multi-modal tasks for understanding hidden features between image and text. In traditional approaches, this task is often not considered as an individual task. Recent results in Answerability are derived from corresponding VQA systems. Firstly, a pre-defined answer’s vocabulary of VQA is appended by an “unanswerable” label. Then, a pre-defined binary mapping is to convert a predicted VQA label into an answerability score. Specifically, if the VQA model predicts a sample as an answerable one, its answerability score is 1, and otherwise. In those problem statements, Answerability is considered in a classification task. A fixed score of answerability in VQA systems is too far from the problem’s essence. Therefore, we propose to consider Answerability in a regression model instead of a binary mapping from VQA. Our proposal comes from two following reasons: (i) evaluation metrics (ii) optimization goal. Firstly, the popular evaluation metric in Answerability is an average precision (AP) scores

![](images/b7e47bfc88a0f9f51c4d1452c4c8097388c82111ceb3f083a010f9210d53af20.jpg)  
What website is this?

![](images/c72af43f477c9f4192b1fd104fb46ea684c9486d93e4f131523abe5418b54488.jpg)  
I cannot move the camera slightly closer to the monitor.

![](images/d7f06e8de8581637a675b00b8a52c5b64f6f3884b8287f75ae9b8e19bc252144.jpg)  
What will I do tomorrow

Figure 1: Examples of unanswerable sample in VizWiz dataset 2020 (Answerability score = 0)

determined by thresholds in a precision-recall curve. Secondly, VQA systems put their effort into predicting a suitable answer through answer’s distribution. However, Answerability reflects problems in samples and annotations. Typical examples of unanswerable samples are presented in Figure 1 such as poor-quality images, ambiguous questions, and inconsistent data of annotators. The fundamental challenges in Answerability are to extract visual-textual features and their relationship to determine answerability scores. Accordingly, our problem statement, regression Answerability, is not only novel but also powerful enough to overcome Answerability challenges.

Lately, BERT proves significant successes of Transformer [7] architectures in Natural Language Processing (NLP). Accordingly, recent researches in Computer Vision (CV) also gain much more interest in Transformer architectures and their components [8, 9]. Vision Transformer [8] is firstly introduced in an image recognition task when integrating Transformer architecture with the fewest mirror adjustments [8] into Image Embedding. Specifically, an image is divided into many regions called patches which are put into a Transformer architecture as an Image Encoder. Image patches play similar roles as tokens of sentences in NLP. Obviously, Transformer is useful enough to extract hidden features in images and texts. Advantages of Transformer in both CV and NLP inspire us to integrate them into our Answerability model.

Despite successes of Transformer in CV and NLP, an important question is how it works in the Answerability problem. Therefore, we propose a regression model combining the strength of Transformer in Text and Vision to overcome challenges in Answerability. Furthermore, we also take advantage of pre-trained models in our architecture to enhance its performance and robustness. In experiments, our models outperform previous baseline approaches in the VizWiz-2020 dataset. Our main contributions are as follows: (i) We introduce a novel problem statement in Answerability, which firstly introduces in the research. (ii) We propose a Vision-Text Transformer model in the Answerability task. (iii) Our proposed model proves its performance and robustness in a practical and novel dataset, VizWiz 2020.

## 2 Methodology

## 2.1 Question embedding

Recently, a proposal of BERT [10] marked a significant moment in linguistic representation. Particularly, BERT [10] can capture the meaning of words and sentences from their context. It allows one word to have multiple linguistic senses, which is unavailable in Glove [11], Word2Vec [12] and so on.

Specifically, in our architecture, a question is embedded by a pre-trained BERT model. The detail of our question embedding is shown in Figure 2. Each question is expanded by two special characters including [CLS] and [SEP] to mark the start and end of the sequence. A question representation is derived from a value of [CLS] vector.

## 2.2 Image embedding

In traditional approaches, people often process an image through Convolution Neural Networks. However, the successes of Transformer in NLP inspire researches to integrate it into vision area [13, 14]. Among previous approaches, Vision Transformer proves its efficiency in many image classification datasets. However, in our work, Vision Transformer is firstly combined with BERT [10] as Text Transformer.

Specifically, in Vision Transformer, a 2D image is split into a sequence of flattening regions called patches. The number of patches depends on the size of an input image and a predefined patch’s size. After, patches are connected with position embedding to retain positional information. Accordingly, Vision Transformer treats an image as a sentence whose tokens are similar to visual regions. In our model, we integrate a pre-trained Vision Transformer to extract visual features of a query image. The detail of our image embedding is presented in Figure 3. Specifically, we change the original classifier in Vision Transformer into a fully-connected layer to extract visual features.

![](images/0ce5673d145052d73e48f78d8682997d65a3e7c061ca9214608e95c4ecc0429e.jpg)  
Figure 2: Question Embedding: extracts textual features of words in question by BERT - Text Transformer.

![](images/d5ac262cfad5f88c85ade8ba7cca71e4bb1ad253b45e7c3304b46e2711532272.jpg)  
Figure 3: Image Embedding: splits an image into patches and convert it via Linear and Positional Embedding to extract the regional and visual features.

## 2.3 Vision-Text transformer answerability model

After extract visual and textual features in previous modules, we propose a framework to predict an answerability score. In this architecture, image and question features are normalized by a fully-connected layer into the same

![](images/d2f0d79cdfe4768ec1e7cc43ed710ba3eb765ec23f087533735194cba16d9cd9.jpg)  
Figure 4: VT-Transformer: combines Vision-Text Transformer features by a vector operation to predict an answerable score in the regression model

dimension space in Equation 1.

$$
\boldsymbol { f } _ { I } ^ { \prime } = \boldsymbol { W } _ { I } ^ { T } \boldsymbol { f } _ { I } + \boldsymbol { b } _ { I } ; \boldsymbol { f } _ { Q } ^ { \prime } = \boldsymbol { W } _ { Q } ^ { T } \boldsymbol { f } _ { Q } + \boldsymbol { b } _ { Q }\tag{1}
$$

After that, we use a vector operation ⊙ that includes either multiplication or concatenation to combine them into a meaningful representation. Finally, the answerability score is determined by a fully-connected layer and sigmoid

function via Equation 2.

$$
\hat { s _ { i } } = \sigma ( W _ { r } ^ { T } \left( f _ { I } ^ { ' } \odot f _ { Q } ^ { ' } \right) + b _ { r } )\tag{2}
$$

As we mentioned above, the most successful factor of this work is the novel problem statement. Specifically, we propose to consider this task in regression. Therefore, we use Mean Squared Error in Equation 3 as our loss function instead of cross-entropy in VQA.

$$
L o s s = \frac { 1 } { N } { \sum _ { i = 1 } ^ { n } { { { \left( { { s } _ { i } } - { { \hat { s } } _ { i } } \right) } ^ { 2 } } } }\tag{3}
$$

Where $s _ { i } = \{ 0 , 1 \}$ corresponds to unanswerable and answerable sample.

## 3 Experiment

## 3.1 Dataset and experimental settings

## 3.1.1 VizWiz dataset

VizWiz dataset [5, 6] is published in 2018 and updated until now. It was the first dataset that mentioned the Answerability problem in Visual Question Answering. In the VizWiz Competition, Answerability is regarded as an individual and novel task. In our work, all experiments are conducted and evaluated on the VizWiz dataset. The detail of this dataset is presented in Table 1. The distribution of unanswerable samples in VizWiz is quite high by 27% in train and 32% in the validation set. Respectively, an Answerability system is ideal and essential enough to support VQA models for filtering unanswerable samples that are too hard to predict an answer.

Table 1: Detail of VizWiz 2020 dataset
<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>No. Samples</td><td>20523</td><td>4319</td><td>8000</td></tr><tr><td rowspan="2">%Answerable</td><td>14981</td><td>2937</td><td rowspan="2"></td></tr><tr><td>73%</td><td>68%</td></tr><tr><td rowspan="2">%Unanswerable</td><td>5542</td><td>1382</td><td rowspan="2">一</td></tr><tr><td>27%</td><td>32%</td></tr></table>

## 3.1.2 Evaluation metric

The test set is confidential and not to disclose any details to researchers. All experimental results need evaluating by an online system in EvalAI<sup>4</sup>. Specifically, in VizWiz Challenge, the Answerability task is recommended to evaluate by average precision evaluation metric which computes the weighted mean of precisions under a precision-recall curve in Equation 4.

$$
A P = \sum _ { n } \left( R _ { n } - R _ { n - 1 } \right) P _ { n }\tag{4}
$$

where $R _ { n }$ and $P _ { n }$ are the precision and recall at the n-th threshold.

## 3.1.3 Experimental settings

In our architecture, we take advantage of pre-trained models in Vision and Text Transformer to extract visual and textual features. Besides, we also use fully-connected layers to normalize and reduce the dimension of features. All details of our implementation are presented in Table 2 to reproduce our model.

## 3.2 Results

However, after two years of this challenge, it exists no publication on this task despite its necessity and importance. Therefore, in these results, we mention two strong baselines that include VWTest<sup>5</sup> and BERT-RG-Regression [2]. Firstly, VWTest obtains the best performance in the VizWiz Contest 2020. Despite its no publication, VWTest i trustworthy enough to be considered as a reference. Secondly, BERT-RG [2] recently obtains state-of-the-art results in the Yes/No question type. The strength of BERT-RG is to combine both residual and global features from ResNet and VGG. Two of them are typical in Convolutional approaches remaining dominant in image understanding. This characteristic is suitable for our work to compare Vision Transformer against traditional methods. We reproduce BERT-RG [2] whose classifier is changed into a regression via linear layer as BERT-RG-Regression. Therefore, we consider BERT-RG-Regression as a competitive baseline in our comparison.

Table 2: Detail of experimental settings
<table><tr><td>Components</td><td>Value</td></tr><tr><td>Vision Transformer</td><td>B_16_imagenet1k.</td></tr><tr><td>BERT</td><td>bert-base-uncased</td></tr><tr><td>Full-connected Layer</td><td>cat: 768 - 512 - 1024 - 512 - 1 mul: 768 - 512 - 512 - 512 - 1</td></tr><tr><td>Vector operation</td><td>multiplication, concatenation</td></tr><tr><td>Optimizer</td><td>AdamW(lr = 3e-5, eps = 1e-8)</td></tr></table>

Table 3: The comparison results against strong baselines
<table><tr><td>Model</td><td>Average Precision</td><td>F1-score</td></tr><tr><td>VWTest</td><td>26.84</td><td>42.32</td></tr><tr><td>BERT-RG [2] Regression</td><td>52.22</td><td>41.85</td></tr><tr><td>VT-Transformer (Our model)</td><td>76.96</td><td>67.26</td></tr></table>

The detail of the comparison between our model and strong baselines is shown in Table 3. In both Average Precision and F1-score, our model outperforms the strong baselines. Apparently, our consideration in the regression task is more suitable than traditional approaches. In average precision, the enhancement of regression against classification in the problem statement proves clearly. In F1-score, although results depend on annotations of VQA task for unanswerable class, our model also obtains a significant improvement by 25%

## 3.3 Ablation Study

In this part, we also conduct ablation studies of components in our architecture. Firstly, we would like to reveal the strength of Vision Transformer into Image Embedding. In this component, we compare Vision Transformer against two powerful image classification models that consist of ResNet [15] and VGG [16]. In experiments, we only evaluate the pre-trained models among them. Specifically, we use pre-trained parameters of ResNet-152 and VGG-16 in the Pytorch library. Besides, we also compare the effect of vector operations consisting of multiplication and concatenation. The detail of our results in this ablation study is presented in Table 4. In this comparison, our model, VT-Transformer,

Table 4: Ablation studies on Image Embedding modules
<table><tr><td>Vector Operation</td><td>Model</td><td>Average Precision</td><td>F1-score</td></tr><tr><td rowspan="3">MUL</td><td>ResNet152</td><td>63.13</td><td>63.59</td></tr><tr><td>VGG16</td><td>70.75</td><td>66.82</td></tr><tr><td>VT-Transformer</td><td>76.96</td><td>67.26</td></tr><tr><td rowspan="3">CAT</td><td>ResNet152</td><td>67.74</td><td>65.16</td></tr><tr><td>VGG16</td><td>71.57</td><td>63.62</td></tr><tr><td>VT-Transformer</td><td>74.91</td><td>66.70</td></tr></table>

outperforms ResNet and VGG by approximately 10% on Average Precision and 3% on F1-score. It proves that Vision Transformer works well on Answerability instead of traditional image models. Transformer architecture brings the success of feature extraction in both vision and text. In somehow, consistent architectures in image and question em bedding lead to the effectiveness in optimization. Besides, we observe that multiplication is better than concatenation in our architecture.

Secondly, we also reveal the effects of pre-trained parameters in deploying our Answerability system. In this comparison, we only consider multiplication as a vector operation to combine visual and textual features. VT-Transformer inited by pre-trained parameters works well in Answerability. This evaluation is presented in Table 5. Successes of pre-trained systems with the fine-tuning mechanism come from their strong optimization in huge datasets.

Table 5: Effects of pre-trained parameters in VT-Transformer
<table><tr><td>Models</td><td>Average Precision</td><td>F1-score</td></tr><tr><td>w/o pretrained parameter</td><td>58.95</td><td>57.01</td></tr><tr><td>with pretrained parameter</td><td>76.96</td><td>67.26</td></tr></table>

Finally, we also present the impacts of different sizes on the Vision Transformer. The details of comparison and implementation are shown in Table 6. Settings of B-16 and VT-16 include Transformer blocks, hidden size, fullyconnected layer’s dim, and the number of heads. Furthermore, we only consider both of them without pre-trained parameters. The completed size of the VT-Transformer is approximately 2400MB and 460MB corresponding to B-16 and VT-16. The bigger model is, the more capacity it has. Although B-16 is better than VT-16, a smaller version, VT-16, also gets a high performance against two baselines in the VizWiz-2020 dataset.

Table 6: A comparison of Vision Transformer architectures
<table><tr><td>Model</td><td>AveragePrecision</td><td>F1-score</td></tr><tr><td>B-16 12 - 768 - 3072 - 12</td><td>58.95</td><td>57.01</td></tr><tr><td>VT-16 3 - 512 - 512 - 8</td><td>56.38</td><td>47.87</td></tr></table>

## 4 Conclusion

In this paper, we propose a Vision-Text Transformer model to overcome the challenges of the Answerability task. Our model takes advantage of pre-trained models and Transformer architecture. By formulating Answerability in the regression task, we propose a novel approach that integrates both Vision and Text Transformer to understand multi-modal data. Through specific experiments and ablation studies, our model outperforms competitive baselines in VizWiz 2020 dataset.

## Acknowledgment

This work was supported by JSPS Kakenhi Grant Number 20H04295, 20K2046, and 20K20625. The research also was supported in part by the Asian Office of Aerospace R&D (AOARD), Air Force Office of Scientific Research (Grant no. FA2386-19-1-4041)

## References

[1] Yanyuan Qiao, Zheng Yu, and Jing Liu, “VC-VQA: visual calibration mechanism for visual question answering,” in IEEE International Conference on Image Processing, ICIP 2020, Abu Dhabi, United Arab Emirates, October 25-28, 2020. 2020, pp. 1481–1485, IEEE.

[2] T. Le, N. Tien Huy, and N. Le Minh, “Integrating transformer into global and residual image feature extractor in visual question answering for blind people,” in 2020 12th International Conference on Knowledge and Systems Engineering (KSE), 2020, pp. 31–36.

[3] Tan Wang, Jianqiang Huang, Hanwang Zhang, and Qianru Sun, “Visual commonsense r-cnn,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[4] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo, “Cutmix: Regularization strategy to train strong classifiers with localizable features,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

[5] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham, “Vizwiz grand challenge: Answering visual questions from blind people,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[6] Danna Gurari, Qing Li, Chi Lin, Yinan Zhao, Anhong Guo, Abigale Stangl, and Jeffrey P. Bigham, “Vizwizpriv: A dataset for recognizing the presence and purpose of private visual information in images taken by blind people,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., pp. 5998– 6008. Curran Associates, Inc., 2017.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[9] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran, “Image transformer,” in Proceedings of the 35th International Conference on Machine Learning, Jennifer Dy and Andreas Krause, Eds., Stockholmsmässan, Stockholm Sweden, 10–15 Jul 2018, vol. 80 of Proceedings of Machine Learning Research, pp. 4055–4064, PMLR.

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota, June 2019, pp. 4171–4186, Association for Computational Linguistics.

[11] Jeffrey Pennington, Richard Socher, and Christopher D. Manning, “Glove: Global vectors for word representa tion,” in Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1532–1543.

[12] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean, “Efficient estimation of word representations in vector space,” in 1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings, Yoshua Bengio and Yann LeCun, Eds., 2013.

[13] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko, “End-to-end object detection with transformers,” in Computer Vision – ECCV 2020, Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, Eds., Cham, 2020, pp. 213–229, Springer International Publishing.

[14] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen, “Axial-deeplab: Stand-alone axial-attention for panoptic segmentation,” in Computer Vision – ECCV 2020, Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, Eds., Cham, 2020, pp. 108–126, Springer International Publishing.

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[16] Karen Simonyan and Andrew Zisserman, “Very deep convolutional networks for large-scale image recognition,” CoRR, vol. abs/1409.1556, 2014.