# V-REX: Eficient Specialist VLM Training for Veterinary X-Rays

Tim Elsner<sup>1</sup> , Nicole McNally<sup>2</sup> , Andre Dourson<sup>2</sup> , and Michael Fitzke<sup>2</sup>

<sup>1</sup> Vyyo AI

<sup>2</sup> Mars Petcare

nicole.mcnally@effem.com

Abstract. While generalist VLMs are expensive to train, creating domain experts is widely assumed to require fine-tuning increasingly large foundation models. We show that, in veterinary radiology, this assumption is misguided. By rethinking the entire VLM pipeline - from text tokenisation and pre-training to grounding and inference - we demonstrate that careful engineering can yield models that outperform much larger foundation models from scratch, without relying on any other data. Our approach introduces new strategies for generative pre-training and grounding that improve training eficiency, increasing data utilisation and downstream performance. Using only a fraction of the parameters, data, and compute of contemporary generalist models, we develop the first VLM capable of generating diagnostic reports for veterinary radiographs, surpassing open foundation models on this task by significant margin.

## 1 Introduction

Large Language Models [6,11,46] (LLMs) have demonstrated remarkable capabilities across a wide range of natural language tasks that have found their way into everyday life, including reasoning, question answering, and few-shot generalisation. In parallel, vision models have even surpassed human performance for tasks like visual recognition [30, 35] or detection [5, 50]. Vision-Language Models (VLMs) bring both domains together, producing a text output from visual input data through using an open vocabulary. These models typically extend an existing, pre-trained LLM backbone, fine-tuning it on how to shift the output distribution according to a visual input signal. While powerful, this design makes such systems highly expensive in terms of compute, limiting their possible training and application for many purposes: Such models have a much more vast pool of language and knowledge than what needed for a small, reliable expert model.

Such generalist or foundation model VLMs are jacks of all trades, suited for a wide range of daily tasks, but also masters of none: A model used for radiology that is also able to identify tropical plants while also being able to transcribe handwriting and labelling cars may not be able to outperform an experienced radiologist.

The general VLM pipeline can be broken down into the following parts that have a number of caveats with the currently dominating, jack of all trades approach:

Tokenisation A VLM doing radiology in English language does not need snippets from coding or Chinese characters. A general tokenisation inflates sequence length, meaning training and generation are more expensive and error-prone: More chances for mistakes exist if the output gets longer while attention becomes more costly and imprecise [38].

Pre-Training A pre-trained vision module that already knows how to extract features can be helpful, but is often too general for an expert task: If only 0.01% of the space is used to e.g. represent x-rays, building a dedicated unsupervised approach for x-ray can be the better choice. Similarly for language, we do not need taco recipes or French when writing English x-ray reports.

Training Even simply fine-tuning larger models can be too expensive. An already distilled (and hence, heavily maimed) model at e.g. 3B parameters is still expensive to train. Even when applying techniques like LoRA [24], we efectively try to find a low-rank adjustment of a space where only the tiniest part is relevant. Reacquiring that knowledge with a small model can be much faster than re-purposing a huge model where only a fragment of the learned knowledge is useful: Rather than combing through tons of hay, it’s more convenient to just get a new needle.

Inference Especially with very limited training data, the already produced text alone can be too strong of a prior. This can lead to the image input eventually being ignored. Produced outputs are also rarely explainable and strategies like multi-image VLMs need to be included in training, all leading to higher costs.

We use veterinary x-ray data as an exemplary task to show how to improve each step in the VLM pipeline, with medium-sized proprietary data (15m text-imagepairs) and medium-sized compute (an individual H100 for most experiments). Yet our approach produces model that beats generalist behemoths at a fraction of their costs, trained from scratch, with a lightweight codebase only consisting of Pytorch [37] and Lightning [17].

We aim for reproducibility, efective data and compute usage, and ease of implementation, delivering a framework that works well for specialised VLM tasks, referred to as V-REX. Our core contributions are as follows:

1. We ofer a best practise recipe to train a VLM from scratch, on a budget, with experiments to help estimate the required amounts of data and compute to beat large foundation models

2. We provide a new pre-training strategy that makes training cheaper, but also pushes the achievable boundaries by being more data eficient

3. We contribute strategies for inference that allow $e . g .$ multi-image processing and better image grounding, reducing language priors in favour of image content

## 2 Related Work

We follow the common definition of VLMs as any text-producing model that generates text according to a visual input [33], most commonly images. More precisely, captioning VLMs (like our proposed model) produce one text for a given visual input. Our work aims to make VLMs more eficient both in terms of compute and required data. Hence in our case, while training VLMs from scratch is possible for both CNN and Transformer-based architectures [31, 48], using a pre-trained vision encoder has proven to be more efective.

We thus discuss VLMs in general (Sec. 2.1), then focus on contrastive embeddings for improved performance (Sec. 2.2), and a specific contrastive embedding called RAPTOR used for our work as pre-trained vision encoder.

## 2.1 VLMs

Early work on vision-language modelling focused on directly mapping images to text via encoder–decoder architectures [49], e.g. combining a CNN image encoder with an LSTM [23] decoder to generate captions. This line of work established the now-standard paradigm of extracting a fixed visual representation and autoregressively generating text conditioned on it. This encoder-decoder design of captioning VLMs follows the transformer [47] structure of an encoder for the image and an autoregressive decoder for the text [1].

Dual-stream architectures such as [31] and [45] introduced separate visual and textual encoders with cross-attention layers to enable fine-grained alignment between modalities. These models were often pre-trained with multiple objectives, including masked language modelling, masked region modelling, and cross-modal matching.

More intricate VLM paradigms consist of two independent encoders, one for text and one for visuals, often also trained with an additional contrastive embedding. Approaches like the PaliGemma family [4, 42] follow this design. For both approaches, variants exist that utilise existing language models to ease the computational burden [10, 28, 55].

Note that for all the mentioned models, our proposed VLM is orders of magnitude smaller (800m parameters for the largest and uncompressed V-REX variant compared to e.g. 7B for the smallest and already distilled LLaVA [28]), while general VLMs also require orders of magnitudes more data than is available for most scenarios. To improve these the data- and computation cost hurdles of VLM training, we utilise tokenisation and pre-training in particular, as both ofer ways to significantly reduce costa while also improving performance.

Tokenisation Just as with pure LLMs, performance of text-producing networks directly depends on how text is captured. While some approaches skip tokenisation entirely [13, 36], most approaches still rely on Byte Pair Encoding (recursively replacing the most frequent pair of tokens with a single new token) [41], even for other domains like images [16], or apply related strategies like Word-Piece [14] or SentencePiece [20] that build a vocabulary according to heuristics. A domain-specific vocabulary can hence drastically reduce the average sequence length, reducing computational complexity while allowing less room for mistakes.

Pre-Training As VLMs often utilise already trained language models [10,28,55] or vision encoders [31, 45], this can be considered as a form of pre-training. However, for our purposes, we also consider approaches that do generative pretraining on visual data. Chen et al. [8] learn to autoregressively predict pixels to learn a rich representation that can be used for other tasks like classification. Similar in spirit, Xie et al. [51] does the same for a decoder-only transformer in GPT-style, while others [3] utilise BERT-Style [14] masked image modelling to learn semantic features. Our method shows the efectiveness of the same principals to VLMs, also showing efectiveness on already rich representations.

## 2.2 Contrastive Embeddings

Explicit contrastive embedding methods like CLIP [39] and related approaches [25,53,54] produce an output vector that describes a given input text or image, shaped such that the dot product between visual and text inputs measures the similarity. This is obtained through a contrastive loss that usually utilises similar (positive) and dissimilar (negative) inputs. Implicit contrastive embedding methods like DINO [7] and similar approaches [9, 21] forgo the contrastive embedding, instead e.g. relying on regularising the network enough to avoid the necessity of negative pairs and producing features in an unsupervised manner.

We can bring together the advantages of both worlds, using unsupervised and rich features that are steered towards more semantically meaningful results, using the text as a guidance for visual features towards what is relevant. We employ these RAPTOR embeddings as the backbone for our VLM, produced by a (currently) proprietary encoder trained on veterinary data. RAPTOR uses the DINO [7] architecture with a contrastive, CLIP-like [39] objective between embedded images and text, built upon Vet-DINO [15]. We use the feature tokens of the last layer of the transformer, right before the extraction of the CLS token. Essentially, we use RAPTOR to get an already rich visual representation of the image with features that are more relatable to text tasks. While our model also works with DINO features or just image projections, we observe better optima and faster convergence using RAPTOR tokens (see Fig. 1).

## 3 A Scaleable VLM Pipeline

Unlike most approaches [10, 28, 55] that train VLMs, we propose to approach each step of the "traditional" VLM pipeline from scratch for our target expert

![](images/7d7f3fc5cecbf522a040e204089e8c555d73d5e898a6a281ec7657dad7b7e82e.jpg)  
Fig. 1: Advantage of using RAPTOR: With the right pre-trained data embedding, our V-REX performs much better, however, still trains with just a linear embedding.

domain (here, veterinary data).

Commonly, VLMs rely on existing LLMs that are fine-tuned to also process images [1,4], hence keeping the general LLM tokeniser instead of using an expert vocabulary. We argue that while this eases the required computational burden by utilising existing language and expert knowledge, it also comes at the cost of a non-expert tokenisation. Our veterinary model does not need tokens for programming and mandarin. To see the efect of tokenisation, consider the error probability of a sequence:

$$
P ( e r r o r ) = 1 - ( 1 - e ) ^ { T }\tag{1}
$$

for $T$ tokens and an error probability per token of e. For an error rate per token of only $e = 0 . 0 2$ and a sequence length of $T = 2 0$ , reducing the sequence length to $T = 1 0$ increases the chance of generation without a faulty step from about 66% to 81%. We take this as an inspiration to revisit the entire VLM pipeline, in particular tokenisation, from scratch while keeping the used components simple and lean, avoiding e.g. upgrades to the transformer architecture [52] or flashy new tokenisers [13,36]. We use a schedule-free [12], rectified [29] version of Adam [27], forgoing any need for learning rate scheduling. Because there is a merit not only in explaining what worked, but also in what did not work, we list our failed experiments in Appendix B.

## 3.1 Text Preparation and Tokenisation

Tokenisation schemes like BPE [41] compress text according to heuristics, and even out the information density: Highly frequent pairs with low entropy of the input distribution are compacted together. This general strategy of applying generation to a more compact domain has been crucial for generative quality, also in other domains [32, 40]. However, when switching from a general purpose to an expert domain VLM, tokenisation is often left as is. We first homogenise text through converting to lower case and removing typos with an LLM. We then apply a simple approach of first using a word-level tokeniser, then apply BPE on the word-level until we reach a vocabulary of 32768 tokens. This gives us a high degree of compression with tokens that are still well interpretable (as the initial tokens are on word-level). For our data, this reduces the average text length from about 520 characters to 30.8 tokens (about 16.9 characters per token), compared to an average of 106 tokens (about 4.9 characters per token) for the GPT 5.x tokeniser (roughly 6 times longer sequences).

![](images/e3247193eb18c76133e70ce431f0a3d8cbdd3c12534d0387c6b73467cf53756c.jpg)  
Fig. 2: Our proposed pipeline follows a fully autoregressive and unified architecture: For the encoded, frozen embeddings of the image, autoregressively learn to predict the next (quantised) image token as a pre-training. We then autoregressively learn to produce text. All of our model is using the same space, with a simple sequence with image tokens row by row followed by the text. Extensions like multi-image support or IFG for better image adherence are applied during inference.

## 3.2 VLM Architecture

As the core of our model, we use a standard decoder-only transformer [47] that learns to autoregressively predict the text part of a sequence consisting of image tokens followed by the report (see Fig. 2). While flat projection of image patches did work, for our veterinary image tokens, we use RAPTOR-tokens (a DINO-CLIP hybrid trained on our same data, see Sec. 2.2).

As a pre-training step, we autoregressively train on the discretised image embeddings. For the text prediction training, we then allow full visibility within the image tokens, i.e. set the upper-left part of the mask that corresponds to the image tokens to 0. For the image tokens, we use a standard 2D sinusoidal encoding to annotate the position of the image embeddings, while applying the 1D sinusoidal encoding for the text position. We additionally add one embedding to all image tokens and one embedding to all text tokens, easing the distinction between image and text. We forgo more complicated (and powerful) encoding schemes [18,26,43], as our reports largely follow the same structure, are relatively short, and for simplicity. We also use the same hyper parameters for the diferent phases of training. While we do not discuss these changes here, our architecture can be also trained to work on multi-image inputs, simply concatenating the tokens of multiple images.

## 3.3 Optional Pre-Training

We apply generative pre-training on our images for two particular reasons: First, it enables the use of potentially unlabeled data, such as images without accompanying text. Second, the increased data utilisation allows for faster convergence to a better optimum. We argue that generative pre-training is the most wholistic form of learning, as learning to replicate an input data distribution requires understanding of the fabric of the data distribution. Training to autoregressively predict the next part of the image allows the transformer to already understand the structure and inner workings of the images.

In our case, we do all our experiments on already embedded 16−by−16 image patches of the RAPTOR framework.To enable autoregressive prediction of continuous values, we simply cluster the produced embeddings into 512 centres to obtain a discrete index representing groups of similar embeddings. As the assignment of a k-means cluster centre becomes memory intensive for high dimensional spaces, we obtain k diferent indices representing the diferent image patches by rounding the vectors representing the image patches into k diferent buckets. We do this eficiently by applying a binary quantisation on a projected subspace of the embedding vectors, with details on this given in Appendix A.

During pre-training, we only use the quantised versions of the embeddings to get a target for prediction, i.e. use the non-quantised version as input sequence to align with the later steps of using the unquantised image tokens.

## 3.4 Training

When continuing from the pre-training, we first train the head for 1500 steps, before gradually warming up the learning rate over the next 1500 steps. We randomly zero out the image embedding for 1 in 32 tokens in order to later identify and amplify the impact of the image on the text token prediction, achieving better grounding (see Sec. 3.5).

## 3.5 Inference-Time Improvements

While specifically training models to deliver more conceivable results or utilising multiple images is possible, smaller projects often are limited in terms of compute or data. We hence provide a small set of suitable extensions.

![](images/e2eab75e311f9929919ebe026511dd39114dfdc7a9645d4487f96dea2c83c783.jpg)  
Fig. 3: Visualisation of visual token impact: The model decides for a next token piece containing the word "cardiac", with the computed Integrated Gradients highlighting the heart.

Overcoming the language prior: Visual Grounding The already produced language can be a powerful prior for the next token predictor of VLMs, ignoring possible image content [2]. We adapt the idea of classifier-free guidance from Ho et al. [22] that propose to arithmetically increase the impact of the controlling signal. For this, they compute and amplify the diference in output with and without the controlling signal (in their case: text).

In our case, for each next token step, we call the model twice, with and without image, measure the probabilities for the next tokens, and then add the diference multiplied with a scalar: We define the next token probabilities given the previous n text tokens t and image I as $P ( t , I )$ , with $t = ( t _ { 0 } , t _ { 1 } , . . , t _ { n } )$ . We then define our image-free guidance (IFG), in analogy to classifier-free guidance in difusion models [22], with a magnitude factor s as:

$$
\begin{array} { r } { P _ { I F G } ( t , I ) = P ( t , I ) + s \cdot ( P ( t , I ) - p ( t , \emptyset ) ) } \end{array}\tag{2}
$$

This amplifies the contribution of the part of the signal that is specifically resulting from the image, while leaving tokens that result from the language prior alone unchanged. We then normalise the probabilities and sample the next token.

Multi-Image Prediction While we can train our model to use multiple images, we can just pool the results of predicting the next token from multiple images, always taking the most likely token over all images. This can increase accuracy for multi-image examples by up to 0.03 in top-1 accuracy without any changes to the model.

Explainability As the used RAPTOR image tokens (opposing to a vanilla DINO) have noisy attention maps through the additional grounding in text that can not be recognised directly anymore, we can apply the Integrated Gradients method [44] to show the visual attribution per text token (see Fig. 3).

## 4 Evaluation

We split the evaluation of our model in two parts. First, we discuss the performance on veterinary data compared to other methods. Second, we analyse the diferent components and strategies of our approach.

## 4.1 Performance Comparison

We provide evidence for the performance of our model by comparing it to powerful generalists (exemplary, OpenAI o3 [34]) and a fine-tuned VLMs (exemplary, PaliGemma [4]). To compare results, we let a model generate a report, then feed both the ground truth report written by a veterinarian and the generated report into o3. We prompt the model to first extract a list of existing findings according to a fixed catalogue from the ground truth, then do the same to the generated report, measuring the global F1 score. We exhibit results showing a strong advantage of our approach in Tab. 1, despite our model only having a fraction of the parameters. Note that fine-tuned versions of PaliGemma do not need to learn to speak first, giving our model a singificant advantage in the first days of training. After about 4 days, the losses of PaliGemma variants stopped improving.

We conclude that the existing expert knowledge, feature extraction, and the language knowledge in a generalist VLM is still below what we can achieve with a model orders of magnitudes smaller if it is crafted properly, and that even fine-tuning an existing model is not better than training an expert from scratch with out recipe. We provide a more detailed evaluation for scores on diferent conditions in Fig. 4.

![](images/a1dd5721fb923b35df86f15bd557ba6fd950e4434fe81ed70e15a4e33686df74.jpg)  
Fig. 4: F1 scores for the 30 most common conditions when using our V-REX model.

Table 1: We measure training time of using one H100 on our dataset, with PaliGemma being fine-tuned to our data. Note that these are micro F1 scores (globally across all classes of findings). Both LorA and Finetuning for Paligemma did not improve on their train loss anymore.
<table><tr><td>MODEL</td><td>DAYS ON H100 PARAMETERS SCORE</td><td>TRAINABLE F1 ↑</td><td></td></tr><tr><td>03</td><td>UNKNOWN</td><td>UNKNOWN</td><td>0.2</td></tr><tr><td>PALIGEMMA, LORA</td><td>1</td><td>3.3M</td><td>0.11</td></tr><tr><td>PALIGEMMA, LORA</td><td>5</td><td>3.3M</td><td>0.13</td></tr><tr><td>PALIGEMMA, FINETUNING</td><td>1</td><td>3B</td><td>0.34</td></tr><tr><td>PALIGEMMA, FINETUNING</td><td>5</td><td>3B</td><td>0.38</td></tr><tr><td>OURS, LIGHT</td><td>1</td><td>504M</td><td>0.34</td></tr><tr><td>OURS, LIGHT</td><td>5</td><td>504M</td><td>0.38</td></tr><tr><td>OURS, LIGHT</td><td>7</td><td>504M</td><td>0.39</td></tr><tr><td>OURS, BIG</td><td>600</td><td>925M</td><td>0.4</td></tr></table>

Table 2: Top-1 accuracy for next text token prediction on the test set after 30M samples seen, across diferent dataset sizes.

<table><tr><td>DATASET SIZE 0.1M 0.5 M</td><td>3M 15M</td></tr><tr><td>ACCURACY (TOP-1) ↑ 0.210 0.327 0.351 0.394</td></tr></table>

## 4.2 Analysis

Data Eficiency A known issue when training VLMs is data scarcity. Hence, for veterinary data, we evaluate how much data is actually needed to perform well. For simplicity, we hence measure performance on diferent dataset sizes for the same amount of steps Tab. 2. The diminishing results show that a fairly capable model can be trained with as little as 3m text-image pairs. We also measure the efects of using our existing data more efective through our proposed generative pre-training on images in Tab. 3.

## 4.3 Grounding Through IFG

We evaluate the efectiveness of our post-training Image Free Guidance in 5, seeing a noticeable increase in correct reports through amplifying image impact on token probabilities. Especially on smaller datasets, this technique can help amplify the models capabilities without more training, which is prone to overfitting on smaller datasets.

Table 3: Generative pre-training efect, measure as the best value occurring on the 3m subset. We pre-train for 15 epochs, giving the number of cluster centres used for discretisation to allow autoregressive pre-training. ’None’ shows without pre-training.
<table><tr><td>MEASURE</td><td>NONE</td><td>512</td><td>4096 32768</td></tr><tr><td>ACCURACY (TOP-1) ↑ .3769 .3898 .3806 .3753</td><td></td><td></td><td></td></tr><tr><td>ACCURACY (TOP-5)↑ .6739 .6857 .6741 .6694</td><td></td><td></td><td></td></tr><tr><td>PERPLEXITY ↓</td><td></td><td></td><td>14.8113.8515.4815.26</td></tr></table>

![](images/2990c1f762644040c4b94a6cdf9938af2208ffd1f098d59492c82cea06b937df.jpg)  
Fig. 5: We show the efectiveness of our additional grounding through comparing performance with and without IFG.

## 5 Conclusion

We introduce a recipe to train a small expert VLM from scratch. We conclude that better data usage through pre-training, additional grounding for better results, a strong pre-trained feature extractor like RAPTOR, and an expertlevel tokenisation are key components to obtain a cheap, lightweight model from scratch. Our results can help to refute the prejudice that fine-tuning a massive, general expert is the only viable approach: A small, lightweight model with the right design choices can be even more efective than the general purpose behemoths.

## References

1. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for

few-shot learning. Advances in neural information processing systems 35, 23716– 23736 (2022)

2. Asadi, M., O’Sullivan, J.W., Cao, F., Nedaee, T., Fardi, K., Li, F.F., Adeli, E., Ashley, E.: Mirage the illusion of visual understanding. arXiv preprint arXiv:2603.21687 (2026)

3. Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254 (2021)

4. Beyer, L., Steiner, A., Pinto, A.S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al.: Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726 (2024)

5. Browatzki, B., Sung, M., Wallraven, C.: Comparing humans and machines: the case of object detection (2024)

6. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D.: Language models are few-shot learners. CoRR abs/2005.14165 (2020), https://arxiv. org/abs/2005.14165

7. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. CoRR abs/2104.14294 (2021), https://arxiv.org/abs/2104.14294

8. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative pretraining from pixels. In: International conference on machine learning. pp. 1691–1703. PMLR (2020)

9. Chen, X., He, K.: Exploring simple siamese representation learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15750–15758 (2021)

10. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems 36, 49250–49267 (2023)

11. DeepSeek-AI, et al.: Deepseek-v3 technical report (2025), https://arxiv.org/ abs/2412.19437

12. Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A., Cutkosky, A.: The road less scheduled. Advances in Neural Information Processing Systems 37, 9974– 10007 (2024)

13. Deiseroth, B., Brack, M., Schramowski, P., Kersting, K., Weinbach, S.: T-free: Subword tokenizer-free generative llms via sparse representations for memory-eficient embeddings. arXiv preprint arXiv:2406.19223 (2024)

14. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). pp. 4171–4186 (2019)

15. Dourson, A., Taylor, K., Qiao, X., Fitzke, M.: Vet-dino: Learning anatomical understanding through multi-view distillation in veterinary imaging. arXiv preprint arXiv:2505.15248 (2025)

16. Elsner, T., Usinger, P., Nehring-Wirxel, J., Kobsik, G., Czech, V., He, Y., Lim, I., Kobbelt, L.: Multidimensional byte pair encoding: Shortened sequences for improved visual data generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 21331–21341 (2025)

17. Falcon, W., Team, T.P.L.: Pytorch lightning. GitHub. Note: https://github.com/Lightning-AI/pytorch-lightning (2019)

18. Gelberg, Y., Eguchi, K., Akiba, T., Cetin, E.: Extending the context of pretrained llms by dropping their positional embeddings. arXiv preprint arXiv:2512.12167 (2025)

19. Gloeckle, F., Idrissi, B.Y., Rozière, B., Lopez-Paz, D., Synnaeve, G.: Better & faster large language models via multi-token prediction (2024), https://arxiv. org/abs/2404.19737

20. Google: Sentencepiece: Unsupervised text tokenizer for neural network-based text generation. https://github.com/google/sentencepiece (2025), software

21. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems 33, 21271–21284 (2020)

22. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

23. Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural computation 9(8), 1735–1780 (1997)

24. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)

25. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)

26. Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., Reddy, S.: The impact of positional encoding on length generalization in transformers. Advances in Neural Information Processing Systems 36, 24892–24928 (2023)

27. Kingma, D.P.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

28. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)

29. Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., Han, J.: On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265 (2019)

30. Liu, M., Wei, J., Liu, Y., Davis, J.: Human and ai perceptual diferences in image classification errors. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 14318–14326 (2025)

31. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. CoRR abs/1908.02265 (2019), http://arxiv.org/abs/1908.02265

32. Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., Frey, B.: Adversarial autoencoders. arXiv preprint arXiv:1511.05644 (2015)

33. NVIDIA: What are vision language models (2025), https://www.nvidia.com/enus/glossary/vision-language-models/, accessed: 2026-01-27

34. OpenAI: Introducing openai o3 and o4-mini. https : / / openai . com / index / introducing-o3-and-o4-mini/ (Apr 2025), accessed: 2026-04-08

35. O’Toole, A.J., An, X., Dunlop, J., Natu, V., Phillips, P.J.: Comparing face recognition algorithms to humans on challenging tasks. ACM Transactions on Applied Perception (TAP) 9(4), 1–13 (2012)

36. Pagnoni, A., Pasunuru, R., Rodriguez, P., Nguyen, J., Muller, B., Li, M., Zhou, C., Yu, L., Weston, J.E., Zettlemoyer, L., et al.: Byte latent transformer: Patches scale

better than tokens. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 9238–9258 (2025)

37. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

38. Qin, Z., Han, X., Sun, W., Li, D., Kong, L., Barnes, N., Zhong, Y.: The devil in linear transformer. arXiv preprint arXiv:2210.10340 (2022)

39. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. CoRR abs/2103.00020 (2021), https://arxiv.org/abs/2103.00020

40. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

41. Sennrich, R., Haddow, B., Birch, A.: Neural machine translation of rare words with subword units. In: Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: long papers). pp. 1715–1725 (2016)

42. Steiner, A., Pinto, A.S., Tschannen, M., Keysers, D., Wang, X., Bitton, Y., Gritsenko, A., Minderer, M., Sherbondy, A., Long, S., et al.: Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555 (2024)

43. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

44. Sundararajan, M., Taly, A., Yan, Q.: Axiomatic attribution for deep networks. In: International conference on machine learning. pp. 3319–3328. PMLR (2017)

45. Tan, H., Bansal, M.: Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490 (2019)

46. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G.: Llama: Open and eficient foundation language models (2023), https://arxiv.org/abs/2302.13971

47. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

48. Vinyals, O., Toshev, A., Bengio, S., Erhan, D.: Show and tell: A neural image caption generator. CoRR abs/1411.4555 (2014), http://arxiv.org/abs/1411. 4555

49. Vinyals, O., Toshev, A., Bengio, S., Erhan, D.: Show and tell: A neural image caption generator. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3156–3164 (2015)

50. Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M.: Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7464–7475 (2023)

51. Xie, J., Ye, K., Li, Y., Li, Y., Lin, K.Q., Zheng, Y., Shen, L., Shou, M.Z.: Visorgpt: Learning visual prior via generative pre-training. arXiv preprint arXiv:2305.13777 (2023)

52. Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Zhao, L., et al.: mhc: Manifold-constrained hyper-connections. arXiv preprint arXiv:2512.24880 (2025)

53. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917 (2022)

54. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)

55. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)

## A Binary Subspace Quantisation

When applying k-means to divide an input space into k diferent clusters that we can use for generative autoregressive pre-training, memory use can skyrocket when assigning our continuos vectors of our input space of a vector x with d dimensions: For a number of k cluster centres, we need to compute a $d \times k -$ dimensional matrix per vector to find the closest cluster centre.

Our idea is to simply round each dimension of an input vector x in such a way that all possible value combinations of the dimensions form our k-many diferent indices. In the simplest form, we can just subtract the average x-vector from an input, then check for each dimension whether it is great or equal zero, or below. This gives us $2 ^ { d } .$ -many diferent quantised vectors. To have control over the number of quantised vectors, we project x to a dimensionality of n first using a projection A that we optimise to be as lossless as possible.

More formally, let $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { N } , x _ { i } \in \mathbb { R } ^ { d }$ , be a dataset of d-dimensional vectors. We seek a low-dimensional representation and a discrete clustering induced by binary codes. We learn an encoder matrix $A \in \mathbb { R } ^ { n \times d }$ and a decoder matrix $\overset { \cdot } { A ^ { \prime } } \in \mathbb { R } ^ { d \times n }$ by minimizing the reconstruction error

$$
\operatorname* { m i n } _ { A , A ^ { \prime } } \sum _ { i = 1 } ^ { N } \big \| A ^ { \prime } A x _ { i } - x _ { i } \big \| _ { 2 } ^ { 2 } .\tag{3}
$$

This corresponds to learning a rank-n linear subspace approximation of the data. Each data point is then projected into a lower dimensional latent space:

$$
z _ { i } = A x _ { i } \in \mathbb { R } ^ { n } .\tag{4}
$$

We compute the empirical mean of the latent representations

$$
\mu = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { i } ,\tag{5}
$$

and centre the latent vectors:

$$
\tilde { z } _ { i } = z _ { i } - \mu .\tag{6}
$$

We then assign a binary code $b _ { i } \in \{ - 1 , + 1 \} ^ { n }$ to each centred latent vector by applying an element-wise sign function:

$$
b _ { i } = \mathrm { s i g n } ( \tilde { z } _ { i } ) ,\tag{7}
$$

where

$$
\mathrm { s i g n } ( u ) _ { j } = \left\{ \begin{array} { l l } { + 1 , } & { u _ { j } \ge 0 , } \\ { - 1 , } & { u _ { j } < 0 . } \end{array} \right.
$$

Each binary vector $b _ { i }$ then corresponds to one of $2 ^ { n }$ possible binary codes. The cluster assignment function is therefore

$$
c _ { i } = b _ { i } \in \{ - 1 , + 1 \} ^ { n } ,\tag{8}
$$

implicitly partitioning the latent space into $2 ^ { n }$ axis-aligned regions in the projection space.

Complexity Example We consider clustering d = 128 dimensional vectors into $K = 4 0 9 6$ clusters. In standard K-means clustering, each data point is assigned to the nearest cluster center by computing the distance to all K centroids. This requires $K \times d = 4 0 9 6 \times 1 2 8 = 5 2 4 ,$ 288 scalar multiplications per data point, ignoring lower-order terms.

In our method, the number of clusters is determined by the number of latent dimensions n, with $2 ^ { n } = 4 0 9 6 \Rightarrow n = 1 2$ . Each data point is projected into a 12-dimensional latent space via a linear mapping $z = A x$ $A \in \mathbb { R } ^ { 1 2 \times 1 2 8 }$ , which only requires $n \times d = 1 2 \times 1 2 8 = 1$ ,536 scalar multiplications per data point. The cluster assignment is then obtained by applying an element-wise sign function to $z - \mu .$ , which incurs negligible additional cost.

The proposed method reduces the per-point cluster assignment cost by over two orders of magnitude while implicitly representing the same number of clusters, although at the cost of only allowing axis-aligned boundaries in the projection space.

## B Unsuccessful Approaches and Negative Results

As our sequences are somewhat structured and have a strong relation to each other, we explored Multi-Token Prediction [19]. We further hoped to obtain more training signal per token, as training this way not only gives a single signal (the next token), but the next few tokens. Our hopes were particularly on getting a higher data utilisation from this. However, in all our experiments, performance was always slightly worse or at best, roughly even. As multi-token prediction usually requires multiple orders of magnitude bigger models than ours to be efective, producing similar results does indicate that for a model that is an order of magnitude bigger, multi-token prediction could be beneficial.

Input and output embedding tying of the text does not help. From our experiments, we conclude that the model size (about 500M parameters), compared to the vocabulary size of about 32,000 tokens, sufers more from the restriction of using shared weights than it benefits from the reduced parameter count and improved alignment between input and output embeddings. We also speculate that, since the decoded tokens stem from a representation that has been enriched with image information, their modality changes enough to negatively impact performance when the embeddings are tied together.

We further experimented with random augmentations of the input image embeddings and diferent optimiser β-values, all to no avail.

Lastly, we made the important discovery that putting a model with dropout layers in evaluation mode to avoid dropping 10 percent of the information per layer actually helps performance (unsurprisingly...).