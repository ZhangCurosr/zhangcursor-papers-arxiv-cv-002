# NATIONAL-SCALE CROP TYPE MAPPING IN MOROCCO USING GOOGLEALPHAEARTH FOUNDATIONS EMBEDDINGS AND THE FIELDS OF THE WORLD FIELDBOUNDARY MAP

Walid Elbarz<sup>1</sup> , Mohamed Bourriz<sup>1,2,3</sup>, Vincent Simonneaux<sup>4</sup>, Hamd Ait Abdelali<sup>1</sup>, Franc¸ois Bourzeix<sup>1</sup>

<sup>1</sup>Analytics Lab (A-Lab), UM6P, Rabat 11103, Morocco

<sup>2</sup>Center for Remote Sensing Applications (CRSA), UM6P, Ben Guerir 43150, Morocco <sup>3</sup>Friedrich Schiller University Jena, Department of Geography, Jena 07743, Germany <sup>4</sup>Centre d’Etudes Spatiales de la Biosph <sup>´</sup> ere (CESBIO), Universit \` e de Toulouse, Toulouse, France´

## ABSTRACT

Recent geospatial foundation models have introduced a new paradigm for large-scale Earth observation by providing pretrained embedding representations that can be directly exploited for downstream tasks. In this work, we investigate the use of Google AlphaEarth Foundations embeddings for national-scale crop type mapping in Morocco. The proposed workflow operates directly on publicly available 64-dimensional pretrained embeddings using lightweight downstream classifiers, eliminating the need to process raw satellite imagery during model training and inference. The approach targets three crop categories, namely winter cereals, tree crops, and other crops. For nationwide inference, the Fields of the World global field boundary map is used to constrain predictions to agricultural parcels, enabling efficient large-scale mapping. Experimental evaluation on an independent test set over ten repeated runs achieved an average classification accuracy of (87.17 ± 0.42)%. Nationwide inference over Morocco was completed in approximately five hours, with the dominant computational cost arising from cloud-based retrieval of the embeddings rather than model execution. These results demonstrate the potential of pretrained embedding representations for computationally efficient national-scale crop type mapping.

Index Terms— AlphaEarth Foundations, Fields of the World, Crop mapping, Embeddings, Spatial transferability

## 1. INTRODUCTION

Remote sensing has become an indispensable technology for large-scale Earth observation, enabling continuous monitoring of environmental, agricultural, and ecological systems. Among its numerous applications, crop type mapping plays a central role in precision agriculture, agricultural statistics, food security assessment, and land-use management. Compared with traditional field surveys, satellite remote sensing provides a cost-effective and scalable alternative for producing crop maps over large geographic regions. Recent advances in artificial intelligence, machine learning, and big-data analytics have further accelerated this transition, enabling increasingly accurate crop classification from multitemporal satellite imagery [1].

Despite these advances, developing reliable crop mapping models remains challenging in regions where annotated agricultural data are scarce. Many countries, particularly in the Global South, lack the extensive ground-truth datasets required to train modern deep learning models, resulting in limited generalization and reduced mapping accuracy. Collecting field observations at national scale is both time-consuming and financially expensive, creating a major bottleneck for operational crop monitoring systems.

The emergence of geospatial foundation models offers a promising paradigm for addressing this limitation. Largescale models such as TerraMind [2], SpectralEarth-FM [3], and related foundation models leverage self-supervised pretraining on massive volumes of Earth observation data to learn generic representations that can be transferred to downstream remote sensing tasks with minimal labeled data. These models have demonstrated strong performance across a wide range of benchmarks, suggesting that pretrained representations can substantially reduce the dependence on extensive local annotations.

More recently, a complementary paradigm has emerged through the use of precomputed embedding representations [4]. Instead of processing raw satellite imagery during downstream inference, foundation models encode multi-temporal and multimodal observations into compact, high-dimensional feature vectors that capture both spectral and temporal information. This approach significantly reduces computational requirements while preserving the information necessary for classification. The concept was first demonstrated on a global scale by the Presto foundation model [5] in the

![](images/0587a940c4bc713ac3e75b7be62178b55a3fdfb4f45a50be5cbad4af78fa97f5.jpg)  
Fig. 1. Study area showing the agricultural regions considered in this work and the distribution of the reference data used for model development and evaluation.

WorldCereal project and has since been further advanced by Google’s AlphaEarth Foundations (AEF) embeddings [6] and the Tessera foundation modelc[7]. These embedding-based representations enable efficient large-scale analysis by replacing computationally intensive feature extraction with reusable latent representations. Recent studies have demonstrated that embedding-based workflows can not only improve classification performance but also substantially reduce storage, preprocessing, and inference costs, making national-scale mapping considerably more practical.

The availability of global foundation models, together with increasingly accessible planetary-scale Earth observation datasets, opens new opportunities for operational crop mapping in data-scarce regions. Rather than training complex deep learning architectures from scratch, downstream classifiers can be trained directly on foundation model embeddings, enabling efficient deployment while requiring only a limited amount of local training data.

In this context, this paper investigates the capability of AEF embeddings for large-scale crop type mapping across Morocco. Specifically, we evaluate the effectiveness of pretrained embedding representations for national-scale crop classification and demonstrate a computationally efficient workflow that substantially reduces the cost of feature extraction and model deployment while maintaining competitive mapping accuracy.

## 2. STUDY AREA

The study covers the major agricultural regions of Morocco, including Souss-Massa, Marrakech-Safi, Beni Mellal-´ Khenifra, Casablanca-Settat, Rabat-Sal´ e-K´ enitra, and Tanger-´ Tetouan-Al Hoce´ ¨ıma (Figure 1). These regions encompass a broad range of agro-climatic conditions, agricultural systems, and cropping practices, making them representative of Morocco’s agricultural landscape. The geographical diversity of the study area provides a suitable benchmark for evaluating the robustness and generalization capability of foundation model embeddings for national-scale crop type mapping.

Ground-truth data were compiled from the publicly available Moroccan agricultural database, which provides fieldlevel crop annotations across the selected regions [8]. To further evaluate the generalization capability of the proposed workflow, an independent validation dataset was assembled using reference data collected during the 2025 agricultural season together with field observations acquired in the Tensift region between 2018 and 2024. The complete reference dataset contains 13,844 agricultural polygons distributed among three crop classes: winter cereals, tree crops, and other crops. Figure 1 illustrates the geographical distribution of the study regions and the corresponding ground-truth data.

## 3. REMOTE SENSING DATA

This study relies exclusively on the Google AlphaEarth Foundations (AEF) embedding dataset [6]. The AEF is a geospatial foundation model that learns compact latent representations from large volumes of multi-temporal and multimodal satellite observations through self-supervised learning. Instead of operating directly on raw satellite imagery, the model encodes each observation into a fixed-length embedding vector that captures its spectral, temporal, and contextual characteristics, allowing the learned representations to be transferred efficiently to downstream Earth observation tasks.

The publicly available AEF embeddings, distributed through an open AWS S3 bucket, are provided at a spatial resolution of 10 m. Each observation is represented by a 64-dimensional quantized embedding vector, which is dequantized prior to model training. Throughout this work, these embeddings constitute the only input features used for crop classification.

The proposed workflow aims to classify agricultural observations into three general crop categories: winter cereals, tree crops (arboriculture), and other crops. To facilitate nationwide crop mapping, the Fields of the World (FTW) global field boundary map [9] is employed during inference. The FTW parcel layer is used as a spatial mask restricting inference to agricultural areas, thereby avoiding predictions over non-agricultural land cover. For every agricultural parcel, the corresponding AEF embedding is retrieved and classified into one of the three target crop categories. This strategy substantially reduces the computational cost of nationwide inference while preserving complete coverage of cultivated land. Although the FTW map may require local refinement, it provides an excellent starting point for large-scale operational crop mapping at a fraction of the computational cost.

![](images/6054c928c163f7e90393b7f87d729a722239f55f36f13765f249824561119e5b.jpg)  
Fig. 2. Overview of the proposed workflow. Publicly available AEF embeddings are first dequantized and combined with ground-truth reference polygons to construct the training dataset. Lightweight downstream classifiers are then trained and evaluated on an independent test set. During nationwide inference, the selected classifier is applied to the AEF embeddings within the Fields of the World field boundary map to produce the final nationwide crop type map.

## 4. TRAINING AND INFERENCE METHODOLOGY

The complete workflow adopted in this study is illustrated in Figure 2. The proposed methodology consists of four main stages: embedding preprocessing, training dataset generation, downstream model training and evaluation, and national-scale crop mapping.

## 4.1. Embedding Preprocessing

The AEF embeddings are distributed as quantized integer values. Prior to model training, each embedding vector is converted to floating-point values using the dequantization procedure proposed by the AEF authors:

$$
v _ { d } = \Big ( \frac { v _ { q } } { 1 2 7 . 5 } \Big ) ^ { 2 } \times \mathrm { s i g n } ( v _ { q } ) ,\tag{1}
$$

where $v _ { q }$ denotes the quantized embedding value and $v _ { d }$ its corresponding dequantized representation.

Following dequantization, every sample is represented by a 64-dimensional embedding vector with values approximately ranging from −0.4764 to 0.4764.

## 4.2. Training Dataset Generation

Reference crop polygons were spatially overlaid with the AEF embedding dataset in order to associate each labeled observation with its corresponding embedding vector. For every reference sample, the extracted information includes the embedding vector, crop label, acquisition year, and additional metadata used during model development.

After preprocessing, the final training pool contains 71,062 labeled samples distributed among the three crop classes. This increase relative to the number of reference polygons results from sampling multiple embedding pixels within individual agricultural polygons. An additional independent test dataset containing 34,565 samples was isolated prior to model development and remained completely untouched during training, model selection, and hyperparameter optimization.

## 4.3. Model Training and Spatial Validation

The proposed workflow adopts an embedding-based downstream classification paradigm.

Four downstream classifiers are evaluated throughout this study:

• Linear Probe;

• Multi-Layer Perceptron (MLP);

• Random Forest;

• Support Vector Machine (SVM).

To obtain a robust estimate of model performance, each classifier is trained and evaluated over ten independent random seeds. The best-performing classifier is subsequently evaluated on the frozen independent test dataset, which was isolated prior to model development and remained completely untouched throughout training and hyperparameter optimization, in order to assess transferability to previously unseen geographical regions.

![](images/22395ed37fee57fd0d8bfd93639962dd2d6244e4592ef7572cdf2ab5dc03e69f.jpg)  
Fig. 3. Examples of the national crop type map produced using the proposed workflow. (On the left) Agricultural landscape in the Gharb region. (On the right) Agricultural landscape in the El Hajeb region.

## 4.4. National-Scale Inference

Following model selection, the best-performing classifier is deployed to generate a national crop type map over Morocco.

Instead of performing exhaustive pixel-wise prediction across the entire country, inference is restricted to agricultural parcels extracted from the 2024 Fields of the World (FTW) dataset. For each agricultural parcel, the corresponding AEF embedding is retrieved and processed by the trained classifier to predict one of the three target crop classes.

This parcel-guided inference strategy significantly reduces computational requirements while enabling efficient nationwide crop mapping using only pretrained embedding representations. Figure 2 summarizes the complete processing pipeline from AEF embeddings to the final crop type map.

## 5. RESULTS

## 5.1. Classification Performance

The classification performance of the four evaluated downstream models on the independent test set is summarized in Table 1. Each model was trained and evaluated over ten independent random seeds; the reported accuracy corresponds to the mean and standard deviation computed across runs.

The results demonstrate that AEF contain highly discriminative information for agricultural crop classification. Despite relying exclusively on lightweight downstream classifiers, the learned representations achieve competitive and consistent performance across the three target crop categories. The MLP achieves the highest test accuracy at $( 8 7 . 1 7 \pm 0 . 4 2 ) \% ,$ closely followed by Random Forest at $( 8 6 . 8 8 \pm 0 . 0 8 ) \%$ and Linear Probe at $( 8 6 . 6 6 \pm 0 . 0 4 ) \%$ . All models yield test accuracies in the range of 86–87% on the independent test set, which consists exclusively of samples from regions excluded from model development, confirming the transferability of the AEF embeddings to previously unseen agricultural areas. Furthermore, the relatively small performance differences between the evaluated classifiers indicate that the pretrained embeddings already provide highly discriminative representations.

Table 1. Test accuracy (%) on the independent test set for each downstream classifier (mean ± standard deviation over 10 runs).
<table><tr><td>Model</td><td>Test Accuracy (%)</td></tr><tr><td>Linear Probe</td><td> $8 6 . 6 6 \pm 0 . 0 4$ </td></tr><tr><td>MLP</td><td> ${ \bf 8 7 . 1 7 \pm 0 . 4 2 }$ </td></tr><tr><td>Random Forest</td><td> $8 6 . 8 8 \pm 0 . 0 8$ </td></tr><tr><td>SVM</td><td> $8 6 . 0 8 \pm 0 . 2 4$ </td></tr></table>

## 5.2. National-Scale Crop Mapping

The selected MLP classifier is subsequently applied to the complete 2024 Fields of the World agricultural parcel layer to generate a national crop type map of Morocco. Nationwide inference was completed in approximately five hours, with the dominant computational cost arising from cloud-based retrieval of the AEF embeddings rather than model execution itself. This result highlights the practical scalability of the proposed embedding-based approach for operational crop monitoring at national scale.

Figure 3 illustrates representative examples of the produced national crop type map at regional scale.

## 6. CONCLUSION

This paper presented a computationally efficient workflow for national-scale crop type mapping in Morocco using Google AlphaEarth Foundations (AEF) pretrained embeddings. By operating exclusively on publicly available 64-dimensional embedding representations and lightweight downstream classifiers, the proposed approach eliminates the need for raw satellite imagery processing during both training and inference, substantially reducing the computational burden associated with large-scale crop mapping.

Experimental results on an independent test set demonstrated that the MLP classifier achieves the best performance, with a test accuracy of (87.17 ± 0.42)% over ten random seeds, confirming the spatial transferability of the AEF embeddings to previously unseen agricultural regions. Nationwide inference over Morocco, guided by the Fields of the World field boundary map, was completed in approximately five hours, with cloud-based embedding retrieval identified as the dominant computational bottleneck.

These findings demonstrate that foundation model embeddings offer a viable and scalable solution for crop type mapping in data-scarce regions, where collecting extensive ground-truth annotations is impractical. The combination of globally pretrained representations with the FTW agricultural parcel layer provides a powerful and cost-effective framework for operational crop monitoring at a national scale.

Future work will focus on incorporating additional crop categories, evaluating the generalization of the workflow across different agro-climatic contexts, and exploring multitemporal embedding aggregation strategies to further improve classification accuracy.

## Acknowledgements

The authors would like to thank Taylor Geospatial for publishing the Google AlphaEarth Foundations embeddings on Source Cooperative, and Source Cooperative, a non-profit open data publishing utility created by Radiant Earth, for providing the infrastructure that made these embeddings openly accessible to the research community. This work was also conducted within the Yield Gap project, supported by the OCP Foundation–UM6P agreement.

## 7. REFERENCES

[1] Mohamed Bourriz, Hicham Hajji, Ahmed Laamrani, Nadir Elbouanani, Hamd Ait Abdelali, Franc¸ois Bourzeix, Ali El-Battay, Abdelhakim Amazirh, and Abdelghani Chehbouni, “Integration of hyperspectral imaging and ai techniques for crop type mapping: Present status, trends, and challenges,” Remote Sensing, vol. 17, no. 9, pp. 1574, Apr. 2025.

[2] Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, Rahul Ramachandran, Paolo Fraccaro, Thomas Brunschwiler, Gabriele Cavallaro, Juan

Bernabe-Moreno, and Nicolas Longep´ e, “Terramind:´ Large-scale generative multimodality for earth observation,” 2025.

[3] Nassim Ait Ali Braham, Aaron Banze, Conrad M. Albrecht, Julien Mairal, Jocelyn Chanussot, and Xiao Xiang Zhu, “Spectralearth-fm: Bringing hyperspectral imagery into multimodal earth observation pretraining,” 2026.

[4] Isaac Corley, Caleb Robinson, Inbal Becker-Reshef, and Juan M. Lavista Ferres, “From pixels to patches: Pooling strategies for earth embeddings,” 2026.

[5] Gabriel Tseng, Ruben Cartuyvels, Ivan Zvonkov, Mirali Purohit, David Rolnick, and Hannah Kerner, “Lightweight, pre-trained transformers for remote sensing timeseries,” 2023.

[6] Christopher F. Brown, Michal R. Kazmierski, Valerie J. Pasquarella, William J. Rucklidge, Masha Samsikova, Chenhui Zhang, Evan Shelhamer, Estefania Lahera, Olivia Wiles, Simon Ilyushchenko, Noel Gorelick, Lihui Lydia Zhang, Sophia Alj, Emily Schechter, Sean Askay, Oliver Guinan, Rebecca Moore, Alexis Boukouvalas, and Pushmeet Kohli, “Alphaearth foundations: An embedding field model for accurate and efficient global mapping from sparse label data,” 2025.

[7] Zhengpeng Feng, Clement Atzberger, Sadiq Jaffer, Jovana Knezevic, Silja Sormunen, Robin Young, Madeline C. Lisaius, Markus Immitzer, Toby Jackson, James Ball, David A. Coomes, Anil Madhavapeddy, Andrew Blake, and Srinivasan Keshav, “Tessera: Temporal embeddings of surface spectra for earth representation and analysis,” 2025.

[8] Youness OUASSANOUAN, Jamal Elfarkh, Said Grich, Abderrahman Liblab, and Abdelghani Chehbouni, “Crop and irrigation type dataset for moroccan agricultural regions: A ground-truth resource for earth observation validation,” 2025.

[9] Caleb Robinson, Gedeon Muhawenayo, Subash Khanal, Zhanpei Fang, Isaac Corley, Ana M. Tarano, Lyndon ´ Estes, Jennifer Marcus, Nathan Jacobs, Hannah Kerner, Inbal Becker-Reshef, and Juan M. Lavista Ferres, “The first global agricultural field boundary map at 10m resolution,” 2026.