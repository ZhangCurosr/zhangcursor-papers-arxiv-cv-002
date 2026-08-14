# TabSOM: A tabular-to-image encoding method based on self-organizing maps

David Chushig-Muzo, Mar´ıa Angeles Rodr <sup>´</sup> ´ıguez de Cara, Eva Milara, Francisco J. Lara-Abelenda, Luis Zhinin-Vera, Diego H. Peluffo-Ordo´nez˜

Abstract—Tabular-to-image methods have emerged as novel approaches to leverage the high predictive performance of convolutional neural networks and vision transformers. They convert tabular data into image representations, mapping each feature at a fixed pixel location derived from a dimensionality-reduction method (e.g., t-SNE, UMAP, PCA). However, they encode only the marginal value of each feature and discard information about feature relationships. We propose TabSOM, a tabular-to-image encoding built on the Self-Organizing Map (SOM), which provides: (i) a spatial layout in which every input feature occupies a fixed canvas position derived from its component plane via collision-free Hungarian assignment; and (ii) a graph that captures pairwise feature relationships derived from the SOM component planes. The resulting image stacks two multi-scale node channels: one encodes feature values at fixed scales, while the other encodes pairwise feature interactions as spatial connections between related features. Two SOM-derived interpretability approaches are introduced: a prototype-inspired partial dependence plot and a class-separation importance score. Benchmarked against twelve existing tabular-to-image methods across public binary-classification datasets, TabSOM ranks first or second on every dataset and achieves the lowest variance of any method evaluated. Interpretability obtained with TabSOM was validated against Random Forest, XGBoost, and SHAP, the class-separation score shows reasonable agreement with established baselines on the top-ranked features while capturing complementary structural information from input data. These results demonstrate that TabSOM provides an effective and interpretable approach for applying deep learning architectures to tabular data, bridging the performance–interpretability gap in this domain.

Index Terms—Tabular-to-image methods, spatial encoding methods, self-organizing maps, component planes, convolutional neura networks, GradCAM

## 1 INTRODUCTION

characterized by a table-like format, with rows corresponding to samples and columns to features. While Deep Learning (DL) models, such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), have achieved remarkable results on data characterized by high spatial or temporal correlations (e.g., images and audio) [2], tabular data generally exhibit weak interdependencies among features. As a result, DL architectures often show reduced effectiveness on predictive tasks involving tabular data.

This has fostered recent research into encoding methods, which convert tabular data into images to leverage the predictive performance of CNNs and pretrained ViTs [1]. Several tabular-to-image methods have been proposed in the literature, ranging from parametric to non-parametric approaches [3]. Most of these methods, such as DeepInsight [4], TINTO [5], REFINED [6], project features into twodimensional through dimensionality reduction methods. They generate images in which spatial proximity reflects feature similarity, with neighboring pixels representing features with some similarity. Tabular-to-image methods have demonstrated competitive performance in several domains such as healthcare [7], [8], [9], indoor localization [10], [11], [12], and the internet of things [13] among others.

The performance of tabular-to-image encoding methods directly depends on how features are spatially arranged and how their values are encoded. Existing methods have addressed this mainly through dimensionality reduction methods [4], [5], including Principal Component Analysis (PCA), kernel PCA, t-distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP). These project the feature space into two-dimensional space, assigning each feature to its nearest pixel on the image canvas. Although they produce effective spatial layouts, the embedding is used solely to determine feature locations. As a result, each feature occupies a fixed position derived from the feature distribution, without explicitly encoding relationships among feature values.

The Self-Organizing Map (SOM), an artificial neural network based on unsupervised and competitive learning, performs a topology-preserving mapping from the input space to a two-dimensional grid of nodes [14]. The topologypreserving property enables that similar inputs are mapped to neighbor nodes on the grid. Then, local relationships in the input space are reflected in spatial proximity on the grid. As a result, SOM has been extensively used for visualization, clustering, and pattern recognition in different domains such as signal processing, healthcare among others [15], [16], [17]. Although SOM has shown excellent results in unsupervised tasks, its application within tabularto-image methods has been partially explored [18]. Unlike other dimensionality reduction techniques, the SOM preserves an interpretable structure. Each node is associated with a prototype, the grid of prototypes provides a densityweighted summary of the joint feature distribution, and the component planes capture how each feature varies across the map. Although tabular-to-image encoding methods have shown promising predictive performance, their interpretability has received limited attention. The graph structure of the SOM captures the topology of the data while providing interpretability.

In this paper, we propose a tabular-to-image encoding method named TabSOM, which includes pairwise feature interactions as part of the image. We train a SOM on the feature space and extract, for each feature, its component plane, i.e., the spatial distribution of that feature’s values across the trained SOM grid. We derive a canvas position (named anchor) for each feature from its component plane, and then optimize anchor collisions via a Hungarian assignment to grid cells. The SOM’s prototype grid serves as a density-weighted, topologically organized sample of the joint feature distribution, and the CNN’s predictions on those prototypes define a spatial prediction map over the grid. One of the main advantages of TabSOM over other tabular-to-image methods is that the SOM is not used solely to fit the layout but also provides two interpretability methods derived from the grid: (i) global feature ranking; and (ii) prototype-inspired partial dependence curves.

The main contributions of this paper are the following:

A novel tabular-to-image encoding method in which feature placement is derived from SOM component planes via anchor estimation and collision-free (Hungarian) assignment.

A relational channel that renders pairwise feature interactions derived from the SOM prototype geometry that determines placement.

A multi-scale rendering scheme in which the same feature layout is rendered at multiple Gaussian bandwidths as aligned channels, providing a CNN with both precise localization and regional coherence without requiring a single bandwidth choice.

An SOM-based interpretability framework that leverages the learned grid structure to provide two complementary explanations: (i) global feature ranking; and (ii) prototype-inspired partial dependence curves.

Feature maps that allow the encoding’s classseparability and per-feature spatial assignment to be inspected independently of any downstream classifier.

The remainder of this paper is organized as follows. Section 2 reviews tabular-to-image encoding methods of the literature. Section 3 presents each component of TabSOM and its derived interpretability tools. Section 4 describes the datasets used, experimental setup, and main results, including classification benchmark and interpretability analysis. Section 5 discusses limitations and directions for future work, and Section 6 presents the main conclusions.

## 2 RELATED WORK

A range of tabular-to-image methods have been proposed in the literature, which are categorized into two main categories [3]: non-parametric and parametric methods.

Non-parametric methods use heuristic algorithms to map features into a predefined image-grid representation without explicitly optimizing the spatial arrangement of features [3]. Representative approaches include the following. BarGraph encodes each tabular sample as a vertical bar chart, where each feature is assigned to a fixed column and the bar height corresponds to the normalized feature value [19]. Binary Image Encoding (BIE) and correlated BIE convert each numerical value to binary strings, then stack these bit-rows to form a two-dimensional matrix where 0 and 1 denote dark and light pixels, respectively [20]. SuperTML encodes feature values as text on an image, using CNNs as character/word-level feature extractors [21]. Tab2Visual encodes each feature as a vertical bar whose width is proportional to the normalized feature value, arranged in rows and columns of bars [22].

Parametric methods perform the spatial arrangement of features through an optimization-based preprocessing stage. Linear and nonlinear dimensionality reduction techniques (such as PCA, t-SNE, and UMAP) project features from the input space into a low-dimensional coordinate space, with the resulting coordinates defining the feature layout from which the image is built. Among the most popular methods, we find DeepInsight [4] and REFINED [6], which use t-SNE, kernel PCA and Multidimensional Scaling to obtain a 2D position for each feature, and then map feature values to pixel intensities. TINTO uses PCA and t-SNE to determine feature locations and the resulting coordinates are then transposed, scaled, and rounded to integer values [5]. It incorporates a blurring mechanism to produce smoother image representations, thereby enhancing compatibility with convolutional filters [5].

Fotomics applies the Fourier Transform independently to each feature column and represents the resulting complex coefficients through their real and imaginary components as coordinates in a two-dimensional Cartesian plane [23]. FC-Viz aggregates highly correlated features into clusters and evaluates the relationships between representative features from each cluster [24]. Ant colony optimization determines the spatial organization of the features, while dimensionality reduction methods are employed to compute pixel intensities [24]. Other methods in this category are IGTD [25] and LM-IGTD [26]. They perform a feature-topixel assignment as a distance-preservation optimization problem, minimizing the difference feature similarities and distances between samples. LM-IGTD [26] extends IGTD to handle low-dimensional and mixed-type data through an unsupervised stochastic noise generation.

Table 1 presents a summary of tabular-to-image methods, including category and technique used for the image layout.

TABLE 1: Summary of tabular-to-image encoding methods, including their category and the technique used to construct the image layout. Methods are shown in alphabetical order.
<table><tr><td>Method</td><td>Category</td><td>Technique for image layout</td></tr><tr><td>DeepInsight [4]</td><td>Parametric</td><td>t-SNE</td></tr><tr><td>Fotomics [23]</td><td>Parametric</td><td>Fourier Transform</td></tr><tr><td>FC-Viz [24]</td><td>Parametric</td><td>t-SNE, KPCA, UMAP and colony optimization</td></tr><tr><td>IGTD [25]</td><td>Parametric</td><td>distance-preservation optimization</td></tr><tr><td>LM-IGTD [26]</td><td>Parametric</td><td>distance-preservation optimization</td></tr><tr><td>REFINED [6]</td><td>Parametric</td><td>Bayesian Multidimensional Scaling</td></tr><tr><td>TINTO [5]</td><td>Parametric</td><td> $\mathrm { P C A } , \underline { { { \sf t } } } { \sf S N E }$ </td></tr><tr><td>BarGraph [19]</td><td>Non-parametric</td><td></td></tr><tr><td>BIE [27]</td><td>Non-parametric</td><td></td></tr><tr><td>SuperTML [21]</td><td>Non-parametric</td><td></td></tr><tr><td>Tab2Visual [22]</td><td>Non-parametric</td><td></td></tr></table>

## 3 METHODS

## 3.1 TabSOM

TabSOM consists of five stages: (1) SOM training and component-plane extraction, (2) feature anchor estimation, (3) collision-free spatial placement, (4) feature relationship graph construction, and (5) multi-channel image rendering. Each stage is described in detail below.

## 3.1.1 Self-organizing map training

Let $X \in \mathbb { R } ^ { N \times F }$ be a tabular dataset with N samples and F features, and let $y \in \{ 0 , 1 \} ^ { N }$ denote the corresponding vector labels. Each feature is first scaled using min-max or z-score normalization. A rectangular SOM with $H \times W$ nodes is trained on the normalized data using the standard online SOM update rule. Each node $( r , c )$ has associated a prototype vector $\mathbf { w } _ { r , c } ~ \in ~ \mathbb { R } ^ { F } .$ , initialized by sampling samples from the training data. At each training step $t ,$ a sample x is drawn at random, and its best matching unit (BMU) is identified as

$$
b = \arg \operatorname* { m i n } _ { ( r , c ) } \| \mathbf { w } _ { r , c } - \mathbf { x } \| ^ { 2 } ,
$$

and every node’s prototype is updated according to the rule:

$$
\mathbf { w } _ { r , c } \gets \mathbf { w } _ { r , c } + \eta ( t ) h _ { \sigma ( t ) } ( r , c , b ) \ ( \mathbf { x } - \mathbf { w } _ { r , c } ) ,
$$

where the learning rate $\eta ( t )$ and neighborhood radius $\sigma ( t )$ decay exponentially from initial values $\eta _ { 0 } , \sigma _ { 0 }$ to final values $\eta _ { 1 } , \sigma _ { 1 }$ over the course of training, and $h _ { \sigma } ( r , c , b ) =$ exp $\left( - \frac { \stackrel { . } { ( r - r _ { b } ) } ^ { 2 } + ( c - c _ { b } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right)$ is a Gaussian neighborhood function over grid coordinates. The grid size $H \times W$ is chosen such that the number of cells ${ \check { K } } = H \cdot W$ is at least $F ,$ ensuring sufficient placement capacity for features.

## 3.1.2 Component planes and feature anchors

After training, the j-th component plane $\Phi _ { j } ~ \in ~ \mathbb { R } ^ { H \times W }$ is defined as the j-th coordinate of every node’s prototype vector, $\Phi _ { j } ( r , c ) = [ \mathbf { w } _ { r , c } ] _ { i } ,$ a smooth spatial field over the grid whose value at $( r , c )$ indicates how high feature $j$ tends to be among samples mapped near that node.

Because the SOM update rule pulls neighboring nodes toward similar prototypes, $\Phi _ { j }$ varies gradually across the grid rather than at random. Nodes that are close together on the map represent similar combinations of feature values, then $\Phi _ { j }$ exhibits a coherent region of high intensity rather than several disconnected peaks. This topology preservation is what allows a single scalar feature to be associated with a location on the grid in the first place, rather than only with a value at each node independently.

We exploit this property to derive, for every feature, a canvas position that reflects where that feature is characteristically expressed across the training distribution. Each feature $j$ is first assigned an anchor position $( \mathbf { a } _ { j } \in [ 0 , 1 ] ^ { 2 } )$ representing its preferred location on the image canvas before collision resolution.

Component planes form the basis for both feature placement and the relational graph. Each feature $j$ is assigned an anchor position $\mathbf { a } _ { j } \in [ 0 , 1 ] ^ { 2 }$ , its preferred location on the image canvas, computed from $\Phi _ { j }$ by one of two methods:

Centroid anchor. The intensity-weighted center of mass of the (optionally sharpened) component plane:

$$
\mathbf { a } _ { j } = \frac { \sum _ { ( r , c ) } \mathbf { g } ( r , c ) \cdot \tilde { \Phi } _ { j } ( r , c ) ^ { \gamma } } { \sum _ { ( r , c ) } \tilde { \Phi } _ { j } ( r , c ) ^ { \gamma } } ,
$$

where $\tilde { \Phi } _ { j }$ is $\Phi _ { j }$ normalized to [0, 1], $\mathbf { g } ( r , c ) \in [ 0 , 1 ] ^ { 2 }$ is the normalized grid coordinate of cell $( r , c )$ , and $\gamma \geq 1$ is a sharpening exponent (default $\gamma = 2 )$ that pulls the anchor toward the plane’s peak by down-weighting low-intensity regions before averaging.

Mode anchor. The location of the plane’s maximum, arg $\operatorname* { m a x } _ { ( r , c ) } \Phi _ { j } ( r , c )$ , is optionally refined to sub-pixel precision via parabolic interpolation over the peak’s immediate neighborhood along each axis. Mode anchors better separate features whose component planes have similar overall mass but different peak locations, at the cost of anchors clustering toward canvas edges/corners for sharply peaked planes.

In practice, the centroid anchor is preferred when component planes are broad and overlapping (typical of lowto-moderate $F ,$ where the SOM has ample grid capacity per feature), while the mode anchor is preferred when F is large and many planes compete for similar grid regions, since a small shift in peak location is then more informative for separating features than a shift in their overall mass. Both anchor methods operate purely on $\Phi _ { j }$ and therefore on a fixed, N-independent $H \times W$ grid, so the cost of computing all $F$ anchors does not grow with the number of samples once the SOM itself has been trained.

## 3.1.3 Collision-free feature placement

Anchors computed may coincide or be placed closely, which would cause multiple features to overlap on the rendered image canvas. To solve this, we apply a discrete optimalassignment step. Let $\{ g _ { 1 } , \dotsc , g _ { K } \}$ denote the centers of the $K \stackrel { \smile } { = } H \times W$ canvas cells (in normalized $[ 0 , 1 ] ^ { 2 }$ coordinates). The cost of placing feature j in cell c is

$$
C _ { j c } = \alpha \lVert g _ { c } - \mathbf { a } _ { j } \rVert - \beta \tilde { \Phi } _ { j } ( g _ { c } ) ,
$$

where the first term penalizes displacement from the feature’s anchor and the second (optional, $\beta \geq 0 )$ rewards placing the feature where its own component plane is locally strong. The final placement $\sigma : \{ 1 , \dotsc , F \}  \{ 1 , \dotsc , K \}$ is obtained by solving the linear assignment problem

$$
\boldsymbol { \sigma } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \sigma } } \sum _ { j = 1 } ^ { F } C _ { j , \boldsymbol { \sigma } ( j ) }
$$

via the Hungarian algorithm, guaranteeing that no two features share a cell (zero overlap) while minimizing total displacement from the SOM-derived anchors. Feature $j ^ { \prime } \mathbf { s }$ final canvas position is the center of cell $\sigma ^ { * } ( j )$

## 3.1.4 Feature relationship graph

To capture pairwise feature interactions, we construct a feature relationship graph directly from the component planes. Each component plane $\Phi _ { j }$ is flattened to a vector in $\mathbb { R } ^ { H W }$ , and the relationship weight between features i and j is computed as either their Pearson correlation or cosine similarity across the flattened planes:

$$
W _ { i j } = \mathrm { c o r r } ( \mathrm { v e c } ( \Phi _ { i } ) , \mathrm { v e c } ( \Phi _ { j } ) )
$$

$$
W _ { i j } = \frac { \mathrm { v e c } ( \Phi _ { i } ) ^ { \top } \mathrm { v e c } ( \Phi _ { j } ) } { \lVert \mathrm { v e c } ( \Phi _ { i } ) \rVert \lVert \mathrm { v e c } ( \Phi _ { j } ) \rVert } .
$$

Negative weights are clipped to zero (retaining only positive co-variation), and the diagonal is set to zero. The edge set E is obtained by thresholding: $E = \{ ( i , j ) : W _ { i j } \geq \tau \}$ for a threshold $\tau \ \mathrm { ( d e f a u l t \ 0 . 5  – 0 . \bar { 6 } ) }$ . Because $W$ is derived from the same prototype geometry that determines feature placement, features connected by an edge are typically placed near each other on the canvas.

## 3.1.5 Multi-channel image rendering

Given the placement $\{ \mathbf { p } _ { j } \} _ { j = 1 } ^ { F } \subset [ 0 , 1 ] ^ { 2 }$ (canvas positions) and the edge set $E$ with weights $\{ W _ { i j } \} _ { ( i , j ) \in E } , \mathrm { ~ a ~ }$ tabular sample $\mathbf { x } \in \mathbb { R } ^ { F }$ is rendered into an image $\boldsymbol { I } \in \mathbb { R } ^ { H _ { \mathrm { i m g } } \times W _ { \mathrm { i m g } } \times C }$ with $C$ channels. We selected $C = 3 ,$ with two first node channels and a relational edge channel.

Node channels. For each of S Gaussian bandwidths $\sigma _ { 1 } <$ $\sigma _ { 2 } < \cdots < \sigma _ { S }$ (default $S = 3 , \mathbf { e . g . } \sigma \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 8 \}$ sharp to smooth), one channel is rendered as

$$
I _ { s } ( u , v ) = \sum _ { j = 1 } ^ { F } x _ { j } \cdot \exp \left( - \frac { \| ( u , v ) - \mathbf { p } _ { j } \| ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } \right) , \qquad s = 1 , \dots , S ,
$$

where $( u , v )$ ranges over pixel centers in $[ 0 , 1 ] ^ { 2 }$ . Each channel is the same spatial layout rendered at a different bandwidth: the sharp channel localizes each feature precisely (supporting fine-grained discrimination), while the smooth channel provides regional coherence over a neighborhood the size of a convolutional kernel’s receptive field (supporting the local-pattern assumptions of CNNs). Because all S channels share the same feature positions, they are spatially aligned, allowing a CNN to jointly exploit multiple scales — analogous to a feature pyramid. We use $S = \hat { 2 } ( \sigma \in \{ 0 . 0 5 , 0 . 0 8 \} )$ ), reserving the third channel for the relational edge channel.

Relational edge channel. For each edge $( i , j ) \in E ,$ , define the segment field as the Gaussian distance from $( u , v )$ to the line segment connecting $\mathbf { p } _ { i }$ and $\mathbf { p } _ { j } \colon$

$$
\begin{array} { l }  \displaystyle { \mathrm { s e g } _ { i j } ( u , v ) = \mathrm { e x p } \left( - \frac { d \big ( ( u , v ) , \overline { { \mathbf { p } _ { i } \mathbf { p } _ { j } } } \big ) ^ { 2 } } { 2 \sigma _ { e } ^ { 2 } } \right) , } \end{array}
$$

where $d ( \cdot , \cdot )$ is the perpendicular distance to the segment (clamped to the segment’s endpoints) and $\sigma _ { e }$ is a fixed edge bandwidth. The edge channel is then

$$
I _ { \mathrm { e d g e } } ( u , v ) = \sum _ { ( i , j ) \in E } W _ { i j } \cdot \sqrt { \operatorname* { m a x } ( x _ { i } , 0 ) \cdot \operatorname* { m a x } ( x _ { j } , 0 ) } \cdot g _ { i j } ( \mathbf { x } ) \cdot \mathbf { s e g } _ { i j } ( u , v ) ,
$$

where $\sqrt { x _ { i } x _ { j } }$ is the interaction intensity — non-zero only when both endpoint features are active for this sample — scaled by the relationship weight $W _ { i j }$ . This channel is therefore a spatial map of which pairwise feature interactions are active for the specific row being encoded, distinct from the node channels, which encode only marginal feature values.

The final image stacks all channels, I = $[ I _ { 1 } , \ldots , I _ { S } , I _ { \mathrm { e d g e } } \left( , I _ { \mathrm { g r a p h } } \right) ]$ , with $C ~ = ~ S + 1 ~ ( \mathrm { o r } ~ S + 2 $ with the graph channel); in our primary configuration $( S = 2 ,$ , edge channel included, no static graph channel) this gives $C = 3 .$

## 3.2 TabSOM-derived interpretability analysis

To provide interpretability, we derive two methods from the trained SOM, which provide a two-dimensional representation of the joint feature distribution.

Let the trained SOM consist of a grid of $K = H \times W$ nodes. Each node $( r , c )$ has a prototype weight vector $w _ { r , c } \in [ 0 , 1 ] ^ { F } .$ , where $F$ is the number of input features. The component plane of feature $j$ is $\Phi _ { j } ( r , { \bar { c } } ) ~ = ~ w _ { r , c , j } ,$ i.e. the j-th coordinate of the prototype at node $( r , c )$ . For each training sample $x _ { i } ,$ its best-matching unit $\mathrm { B M U } ( x _ { i } ) \in$ $\{ 1 , \ldots , H \} \stackrel { \smile } { \times } \{ 1 , \dot { \ldots } \cdot \cdot , W \}$ is the node whose prototype is closest in Euclidean distance.

## 3.2.1 Self-organizing map class-separation importance

For binary classification, we define class-conditional activation densities over the SOM grid from the BMU assignments of the training set. For class $\breve { k } \in \{ 0 , 1 \}$ ,

$$
A _ { k } ( r , c ) = { \frac { | \{ i : y _ { i } = k , { \mathrm { ~ B M U } } ( x _ { i } ) = ( r , c ) \} | } { \left| \{ i : y _ { i } = k \} \right| } } ,
$$

$A _ { 1 }$ and $A _ { 0 }$ are normalized satisfying $\begin{array} { r } { \sum _ { r , c } A _ { k } ( r , c ) = 1 } \end{array}$ The class-separation importance of feature $j$ is the absolute difference between the $\bar { A } _ { 1 } -$ and $A _ { 0 }$ -weighted averages of its component plane,

$$
\mathrm { I m p } _ { j } ^ { \mathrm { c l a s s - s e p } } = \left| \sum _ { r , c } A _ { 1 } ( r , c ) \Phi _ { j } ( r , c ) - \sum _ { r , c } A _ { 0 } ( r , c ) \Phi _ { j } ( r , c ) \right| .
$$

This quantity is large when feature $j$ takes systematically different typical values in the map regions favored by each class. It is computed from the SOM together with the (binary) training labels, independent of downstream predictive models. We use $\mathrm { I m p } _ { j } ^ { \mathrm { c l a s s - s e p } }$ as the primary SOM-derived global feature-importance ranking.

## 3.2.2 Prototype-based partial dependence plot

Each SOM prototype $w _ { r , c }$ represents a density-weighted combination of feature values drawn from the trained SOM’s organization of the input space. In contrast to the synthetic, independence-assuming Cartesian grids used by conventional Partial Dependence Plots (PDPs). We exploit this by encoding every prototype with the fitted encoder and passing the resulting image through the trained CNN f<sub>θ</sub>, yielding a prediction map

$$
\Psi ( r , c ) = f _ { \theta } \big ( \mathrm { E n c o d e } ( w _ { r , c } ) \big ) \in [ 0 , 1 ] ,
$$

interpreted as the predicted probability $P ( y = 1 )$ for the (synthetic but data-consistent) sample represented by node $( r , c )$ . Ψ has the same spatial shape (H, W) as every component plane $\Phi _ { j }$

For a given feature $j ,$ we construct a prototype-based PDP by plotting the pairs $\big ( \Phi _ { j } ( r , c ) , \Psi ( r , c ) \big )$ over all $K$ nodes, where $\Phi _ { j } ( r , c )$ is mapped back to the feature’s original units via the inverse min-max transform. Each point is weighted by its hit count

$$
n ( r , c ) = \big | \{ i : \mathrm { B M U } ( x _ { i } ) = ( r , c ) \} \big | ,
$$

the number of training samples for which node $( r , c )$ is the best-matching unit. A density-weighted mean curve is obtained by partitioning the nodes into B bins of approximately equal cumulative hit count, ordered by $\Phi _ { j } ,$ and computing the hit-count-weighted mean of Ψ within each bin. This curve is read analogously to a standard PDP — the expected model output as a function of feature $j -$ but is restricted to regions of feature space the SOM actually populates with prototypes, so that sparsely-supported (low hit-count) segments are visually distinguishable from well-supported ones. Where a prototype-based PDP is nonmonotonic, the underlying spatial maps $\Phi _ { j }$ and $\Psi$ can be inspected directly on the shared (H, W) grid to identify whether the non-monotonicity arises from an interaction with another feature (i.e., a region of the grid where some other component plane, rather than $\Phi _ { j }$ , aligns with Ψ).

## 4 RESULTS

## 4.1 Datasets

We evaluate TabSOM on four real-world datasets: Oxford Parkinson’s Disease (PAR), Pima Indians Diabetes (PID), QSAR Biodegradation (QSA), and Wisconsin Diagnostic Breast Cancer (WBC). All datasets are obtained from the public repository UCI Machine Learning Repository <sup>1</sup> and present two classes (binary classification). Table 2 summarizes the datasets, including the total number of samples and features.

TABLE 2: Datasets considered in this study.
<table><tr><td>Identifier</td><td>Dataset</td><td>Samples</td><td>Features</td></tr><tr><td>PID</td><td>Pima Indians Diabetes</td><td>768</td><td>8</td></tr><tr><td>PAR</td><td>Oxford Parkinson&#x27;s Disease</td><td>195</td><td>22</td></tr><tr><td>QSA</td><td>QSAR Biodegradation</td><td>1055</td><td>41</td></tr><tr><td>WBC</td><td>Wisconsin Breast Cancer</td><td>569</td><td>30</td></tr></table>

## 4.2 Experimental setup

We split each dataset into two independent subsets, a training subset (80% samples) and test subset (20% samples). To evaluate the generalization of predictive models, all methods are evaluated under 5-fold stratified crossvalidation. Class imbalance was addressed through random undersampling, and it is applied only for training subset to prevent data leakage. All features are min-max normalized to [0, 1]. We evaluate the predictive performance using the Area Under the Receiver Operating Characteristic Curve (AUROC). Results are reported as the mean and standard deviation across five random seeds.

TabSOM is configured as follows. The SOM grid side is set automatically to $H = W = \operatorname* { m a x } ( 8 , \lceil \sqrt { 1 . 3 F } \rceil )$ . Feature placement uses Hungarian assignment with centroid anchors, and the relational graph uses Pearson correlation between component planes with threshold $\tau \ : = \ : 0 . 5$ . The primary rendering configuration uses $S = 2$ node channels at $\sigma \in \left. 0 . 0 5 , 0 . 0 8 \right.$ and the relational edge channel.

Encoded images are classified using a CNN with three convolutional blocks (16-32-64 channels), each followed by batch normalization and ReLU, with a max-pooling step after the second block — followed by global average pooling and a single linear output unit). The network is trained with Adam, the learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size of 32 and a class-balanced binary cross-entropy loss. The same architecture and optimizer are used for every tabularto-image encoding method.

## 4.3 Classification results and benchmark

Table 3 compares TabSOM against twelve existing tabularto-image encoding methods across six binary datasets. Across the four datasets, TabSOM ranks first in AUCROC on Pima (0.8236) and WDBC (0.9911), and third on PAR (0.8852) and QSAR (0.9098). These results yield the second-highest overall mean AUCROC (0.9024) and the second-best average rank, with only the Combination method performing better overall (mean AUCROC: 0.9114). The performance gap between TabSOM and the strongest competing method is small across all datasets. TabSOM outperforms the nextbest method (Combination) by 0.0037 AUCROC on Pima and 0.0014 on WDBC, while underperforming Combination by 0.0327 AUCROC on Parkinsons and 0.0081 on QSAR. This places TabSOM and Combination as the two clearly strongest methods in the benchmark, consistently separated from the rest of the field by a substantial margin — the third-place method on most datasets (BarGraph or DistanceMatrix) trails TabSOM by roughly 0.01–0.04 AUCROC, while methods based on generic dimensionality-reduction embeddings (TINTO (tSNE), DeepInsight (tSNE), DeepInsight (UMAP), Fotomics) trail by considerably more, often falling 0.15–0.3 AUC below TabSOM and showing markedly higher variance (standard deviations frequently exceeding 0.10, compared to TabSOM’s 0.018–0.039 across all four datasets).

Additionally, TabSOM’s standard deviation is among the lowest of any method on every dataset, indicating more stable performance across different seeds than most tabularto-image approaches. Several methods (TINTO (tSNE and blur) on PAR, DeepInsight (tSNE) on PAR, Fotomics on WDBC) show standard deviations an order of magnitude larger, suggesting these embeddings are less reliable across different seeds. Finally, the gap between TabSOM and the weaker methods widens on datasets with fewer features (Pima, 8 features), where structured encodings (Tab-SOM, BarGraph, Combination, DistanceMatrix) outperform embedding-based methods by the largest margin (e.g., Tab-SOM exceeds TINTO (tSNE) by 0.30 AUCROC on Pima), while on higher-dimensional or more separable datasets (WDBC) most methods cluster more closely near the best performance.

TABLE 3: Benchmark of tabular-to-image methods using mean and standard deviation AUCROC values across five random seeds.
<table><tr><td>Method</td><td>Pima</td><td>Parkinsons</td><td>WDBC</td><td>QSAR</td></tr><tr><td>BarGraph</td><td> $0 . 8 1 0 0 \pm 0 . 0 3 0 1$ </td><td> $0 . 8 7 5 3 \pm 0 . 0 2 9 7$ </td><td> $0 . 9 8 5 9 \pm 0 . 0 0 5 5$ </td><td> $0 . 8 8 7 6 \pm 0 . 0 1 6 2$ </td></tr><tr><td>BIE</td><td> $0 . 7 5 8 5 \pm 0 . 0 5 1 3$ </td><td> $0 . 7 3 2 1 \pm 0 . 0 8 1 0$ </td><td> $0 . 9 7 6 5 \pm 0 . 0 1 4 0$ </td><td> $0 . 8 4 5 5 \pm 0 . 0 2 5 9$ </td></tr><tr><td>Combination</td><td> $0 . 8 1 9 9 \pm 0 . 0 2 8 5$ </td><td> $\mathbf { 0 . 9 1 7 9 \pm 0 . 0 3 1 2 }$ </td><td> $0 . 9 8 9 7 \pm 0 . 0 0 4 6$ </td><td> $0 . 9 1 7 9 \pm 0 . 0 1 6 3$ </td></tr><tr><td>DeepInsight (tSNE)</td><td> $0 . 5 8 9 8 \pm 0 . 0 8 3 5$ </td><td> $0 . 7 1 3 4 \pm 0 . 2 5 9 3$ </td><td> $0 . 7 5 7 6 \pm 0 . 1 4 3 3$ </td><td> $0 . 5 6 3 6 \pm 0 . 1 2 0 7$ </td></tr><tr><td>DeepInsight (UMAP)</td><td> $0 . 5 9 2 5 \pm 0 . 1 0 6 7$ </td><td> $0 . 6 2 3 4 \pm 0 . 1 9 4 8$ </td><td> $0 . 6 9 2 7 \pm 0 . 2 3 2 7$ </td><td> $0 . 8 2 3 1 \pm 0 . 0 5 2 6$ </td></tr><tr><td>DistanceMatrix</td><td> $0 . 7 1 5 0 \pm 0 . 0 5 3 2$ </td><td> $0 . 8 9 9 3 \pm 0 . 0 4 8 0$ </td><td> $0 . 9 6 2 4 \pm 0 . 0 1 7 0$ </td><td> $\mathbf { 0 . 9 1 4 0 \pm 0 . 0 1 3 6 }$ </td></tr><tr><td>FeatureWrap</td><td> $0 . 7 8 3 9 \pm 0 . 0 4 2 5$ </td><td> $0 . 7 7 7 6 \pm 0 . 1 2 5 6$ </td><td> $0 . 9 6 1 8 \pm 0 . 0 1 3 5$ </td><td> $0 . 8 3 9 2 \pm 0 . 0 4 3 3$ </td></tr><tr><td>Fotomics</td><td> $0 . 5 6 1 5 \pm 0 . 0 9 9 4$ </td><td> $0 . 6 8 4 2 \pm 0 . 2 6 4 3$ </td><td> $0 . 6 3 4 9 \pm 0 . 3 2 9 6$ </td><td> $0 . 7 6 1 2 \pm 0 . 0 8 2 9$ </td></tr><tr><td>IGTD</td><td> $0 . 5 5 4 8 \pm 0 . 1 1 9 0$ </td><td> $0 . 8 5 6 5 \pm 0 . 0 3 2 1$ </td><td> $0 . 7 8 2 6 \pm 0 . 1 7 5 9$ </td><td> $0 . 5 8 8 7 \pm 0 . 0 7 0 1$ </td></tr><tr><td>TINTO (PCA)</td><td> $0 . 6 0 0 1 \pm 0 . 0 7 5 5$ </td><td> $0 . 8 4 7 2 \pm 0 . 0 8 1 5$ </td><td> $0 . 7 8 8 1 \pm 0 . 0 9 7 1$ </td><td> $0 . 7 9 5 5 \pm 0 . 0 4 3 8$ </td></tr><tr><td>TINTO (tSNE)</td><td> $0 . 5 1 9 9 \pm 0 . 0 9 5 7$ </td><td> $0 . 7 7 6 1 \pm 0 . 1 2 5 1$ </td><td> $0 . 6 7 7 0 \pm 0 . 1 3 5 3$ </td><td> $0 . 7 0 7 6 \pm 0 . 1 1 0 9$ </td></tr><tr><td>TINTO (tSNE and blur)</td><td> $0 . 5 2 6 5 \pm 0 . 0 6 9 4$ </td><td> $0 . 6 7 3 0 \pm 0 . 2 4 0 1$ </td><td> $0 . 7 3 4 1 \pm 0 . 1 1 2 0$ </td><td> $0 . 6 5 4 0 \pm 0 . 0 9 5 7$ </td></tr><tr><td>TabSOM</td><td> $\mathbf { 0 . 8 2 3 6 \pm 0 . 0 2 4 0 }$ </td><td> $0 . 8 8 5 2 \pm 0 . 0 3 9 0$ </td><td> $\mathbf { 0 . 9 9 1 1 \pm 0 . 0 0 8 2 }$ </td><td> $0 . 9 0 9 8 \pm 0 . 0 1 8 0$ </td></tr></table>

## 4.4 Channel decomposition and feature maps

Figure 1 shows channel decomposition and final image resulting of the TabSOM encoding for samples from the PID. The sharp and mid node channels preserve each feature’s individual contribution at a stable canvas position across samples, with intensity tracking the feature’s normalized value. The edge channel shows the pairwise relationships captured by the correlation-based relational graph. A comparison across samples indicates that the overall structure of these channels is consistent among samples, whereas localized intensity differences indicate patient-specific feature values. Regarding feature relationships, for instance, sample 3 exhibits substantially stronger activation along the blood pressure–age and skin thickness–blood pressure edges than the other samples. This pattern is evident both in the edge channel and in the composite image, where it appears as a pronounced cyan–white region.

## 4.5 SOM-derived feature importance and dependence analysis

Figure 2 compares the feature ranking produced by SOM class separation against three established baseline importance measures: RF, XGB and SHAP. As shown, all four methods agree that glucose is the most important feature, and broadly agree in ranking BMI, age, and diabetes pedigree in the upper-middle tier while ranking blood pressure and skin thickness lowest. SOM class separation deviates most from the tree-based methods on age, for which it assigns the second-highest importance of any feature (0.54) versus a comparatively lower ranking from RF, XGB, and SHAP, and on pregnancies, for which it assigns higher relative importance than RF or SHAP. Despite these individual differences in ranking order, the overall agreement across methods derived from entirely different supports SOM class separation as a meaningful importance measure rather than an artifact of the encoding or grid placement procedure

Figure 3 shows the prototype-based partial dependence curve for features of the PID. Four features, glucose, age, blood pressure, and insulin, exhibit increasing dependence, with predicted probability spanning roughly the full observed range (0.0 to 0.7–0.8). Glucose in particular shows an almost linear relationship, consistent with its established role as the most direct marker of glycaemic control in this domain. The remaining four features, pregnancies, bmi, skin thickness, and diabetes pedigree, show nonmonotonic curves with a localized dip or spike in an overall increasing trend, and a visibly narrower range of predicted probability than the first group. diabetes pedigree shows the flattest and narrowest curve of all features (predicted probability varying only between approximately 0.12 and 0.52 in its observed range), consistent with its role as a comparatively weak indirect risk indicator relative to direct physiological measurements.

## 5 DISCUSSION

In this paper, we proposed TabSOM, a tabular-to-image encoding method based on the SOM, and compared it against twelve state-of-the-art tabular-to-image methods across four benchmark datasets. TabSOM achieves competitive classification performance in all evaluated datasets, ranking first or second on every dataset while exhibiting the lowest variance of any method in the comparison. The benchmark comparison reveals two insights. Methods based on deterministic and spatially consistent encodings (BarGraph, Combination, DistanceMatrix, FeatureWrap, and BIE) perform similarly to TabSOM because they produce stable image representations. Methods based on stochastic dimensionality reduction, such as TINTO (t-SNE and PCA variants), DeepInsight (t-SNE and UMAP), IGTD, and Fotomics, perform substantially worse, particularly on smaller datasets such as Pima and QSAR, where the instability of the embedding across folds is reflected in larger standard deviations. TabSOM’s component-plane placement avoids this instability by deriving feature positions from the SOM geometry rather than from a dataset-level embedding, producing a layout that is fixed after a single SOM fit.

The interpretability analysis shows that TabSOM provides global explanations through the class-separation importance score and feature interactions via the prototypebased partial dependence. The feature ranking based on class-separation importance identifies features whose component planes are spatially separated by class label, features whose high or low values map systematically to different regions of the SOM grid depending on the outcome. In PID, glucose, age and BMI rank highest according to classseparation importance, consistent with their relevance as relevant factors associated with diabetes. The feature ranking comparison performed with Random Forest, XGBoost, SHAP, and SOM class-separation showed that the SOMderived ranking agrees moderately with the model-based rankings, with the strongest agreement on the top-ranked features. This suggests that the SOM grid structure captures complementary information, providing interpretability of which features are globally important.

![](images/c46fc3fe230e41cf24cce976b7a1aec583f54a9396264df7f06477c0d333e02e.jpg)  
Fig. 1: Channel decomposition of the TabSOM encoding for four Pima Indians Diabetes samples. Three first columns show the individual channels: the sharp node channel (R, σ=0.05) and mid node channel (G, σ=0.08) render each feature’s value as a Gaussian field centered at its SOM-assigned position, while the edge channel (B) renders the relational graph’s pairwise feature interactions. Fourth channel shows the composite RGB image, whereas the fifth column presents the feature map.

Additionally, the prototype-based partial dependence curves reveal the feature’s effect on the classifier’s output. For the PID, glucose showed a near-linear positive relationship across its observed range, consistent with its role as a glycaemic marker. Age exhibits a steep rise through the 30s and 40s before plateauing and diabetes pedigree produces the flattest and narrowest curve of all eight features, consistent with its status as a comparatively weak risk indicator. Several features (pregnancies, BMI, and skin thickness) showed localized non-monotonicities that the spatial overlay diagnostic traces to low-density regions of the SOM grid or to co-location with higher-ranked features, rather than to independent physiological effects.

In the literature, a single work has explored SOM in tabular-to-image methods. It encodes each sample as a proximity activation map over the SOM prototype grid, measuring Gaussian-weighted distances between the sample and every prototype in RBF kernel space, which produce images where pixel intensities reflect manifold position rather than individual feature values. TabSOM differs in three main dimensions: (i) it derives explicit perfeature canvas positions from SOM component planes via Hungarian assignment, producing spatially interpretable images in which each feature occupies a fixed region; (ii) it adds a relational edge channel encoding pairwise feature interactions not representable in any marginalvalue image; and (iii) it provides interpretability tools (class-separation importance, prototype-based partial dependence) validated against established baselines, rather than qualitative activation-pattern inspection.

![](images/d195ee0c128d88b9dc7c6c0a9792ce625242ddd2ea7d8b21642bd637930bed33.jpg)  
(a)

Fig. 2: Feature importance for the PID considering: Random Forest impurity importance, XGBoost gain importance, SHAP values (computed on Random Forest), and the SOM-derived class-separation importance. Feature importance values are normalized to a maximum of 1 to visually compare feature rankings.  
![](images/a58734fd1485648eb7881035ccfcee83e3654b2c1a035588209012eb17256048.jpg)

![](images/eda14a7eb706cfea6f7357418c43951b785b3c26f7d151653af444464e563d2f.jpg)

![](images/6711efbdb8d5e1c325b27020eaa18ba749a987e1a4cf9880a8b49bcf4f205a03.jpg)

![](images/aaba687c2a6880e4015c018ca708a8dbb99bad549f654329d95537d497562a7e.jpg)

![](images/af0366d7e82d7515a5b12b59351870ad45745f9cb20658fde23bdea92bdb6cfa.jpg)

![](images/da1824cd964d7c94c0ebd2baf03299f9d0461f2b01f64908e2e3485f70124a6a.jpg)

![](images/7c4e060cd0a822cf30fd7cd7ac8b6340a488b21b287f10c3d242e4be7fcb2002.jpg)  
(a)

![](images/342a4412d42d73e597d39bcf285881f80aff2886fe5e8b9187694ec11a01aa2f.jpg)  
Fig. 3: Prototype-based partial dependence for all features of the PID. Each panel plots the SOM prototype grid’s predicted probability against the corresponding feature’s component plane value.

Beyond classification performance, interpretability analysis shows that the features identified by TabSOM align with established clinical knowledge, suggesting that TabSOM captures meaningful domain-relevant patterns. This property is especially valuable in high-stakes domains where transparency is as important as predictive performance. Future work will explore integration with multimodal data sources and systematic comparisons with established tabular interpretability methods. Future work will explore comparisons between Grad-CAM and established post-hoc methods for tabular data, such as SHAP, to better understand their strengths and limitations. Finally, future research may investigate the use of alternative architectures, such as vision transformers, to further enhance the representation learning capabilities of TabSOM-generated images.

## 6 CONCLUSIONS

This paper proposed TabSOM, a tabular-to-image encoding that uses the SOM to provide both a topology-based feature placement and a relational graph over features, rendered as a multi-channel image that separates marginal feature values from pairwise interactions. We benchmarked against twelve existing tabular-to-image methods on four public datasets. TabSOM ranked first and second on every dataset and exhibited the lowest standard deviation across all methods. TabSOM provides two interpretability tools, including a prototype-inspired partial dependence plot, and the classseparation importance score, to provide feature importance and ranking, and the dependence analysis. A comparative analysis of feature importance with SHAP, and RF showed agreement on the top-ranked features, consistent with the SOM grid capturing structural patterns beyond those encoded by tree-based impurity measures.

## REFERENCES

[1] Jiang JP, Liu SY, Cai HR, Zhou QL, Ye HJ. Representation learning for tabular data: A comprehensive survey. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2026.

[2] Mamdouh A, El-Melegy M, Ali S, Kikinis R. Tab2Visual: Deep Learning for Limited Tabular Data via Visual Representations and Augmentation. Pattern Recognition. 2026:113173.

[3] Liu J, Castillo-Cara M, Garc´ıa-Castro R. A comprehensive benchmark of spatial encoding methods for tabular data with deep neural networks. Information Fusion. 2025:104088.

[4] Sharma A, Vans E, Shigemizu D, Boroevich KA, Tsunoda T. DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture. Scientific reports. 2019;9(1):11399.

[5] Castillo-Cara M, Talla-Chumpitaz R, Garc´ıa-Castro R, Orozco-Barbosa L. TINTO: converting tidy data into image for classification with 2-dimensional convolutional neural networks. SoftwareX. 2023;22:101391.

[6] Bazgir O, Zhang R, Dhruba SR, Rahman R, Ghosh S, Pal R. Representation of features as images with neighborhood dependencies for compatibility with convolutional neural networks. Nature communications. 2020;11(1):4391.

[7] Lara-Abelenda FJ, Chushig-Muzo D, Peiro-Corbacho P, Gomez-´ Mart´ınez V, Wagner AM, Granja C, et al. Transfer learning¨ for a tabular-to-image approach: A case study for cardiovascular disease prediction. Journal of Biomedical Informatics. 2025;165:104821.

[8] Singh KR, Dash S, Liu H, Wang Z. Enhanced diabetes prediction using pre-trained CNNs, LSTM, and conditional GAN on transformed numerical data. Scientific Reports. 2026;16(1):8081.

[9] Gomez-Mart´ ´ınez V, Chushig-Muzo D, Soguero-Ruiz C. Tabularto-Image Encoding Methods for Melanoma Detection: A Proof-of-Concept. Applied Sciences. 2026;16(5):2459.

[10] Liu J, Castillo-Cara M, Garc´ıa-Castro R, Orozco-Barbosa L. Interpretable Hybrid Vision Transformer Architectures for MIMO-Based Indoor Localization using Synthetic Spatial Representations. IEEE Internet of Things Journal. 2026.

[11] Castillo-Cara M, Mart´ınez-Gomez J, Ballesteros-Jerez J, Garc ´ ´ıa-Varea I, Garc´ıa-Castro R, Orozco-Barbosa L. MIMO-Based Indoor Localisation with Hybrid Neural Networks: Leveraging Synthetic Images from Tidy Data for Enhanced Deep Learning. IEEE Journal of Selected Topics in Signal Processing. 2025.

[12] Talla-Chumpitaz R, Castillo-Cara M, Orozco-Barbosa L, Garc´ıa-Castro R. A novel deep learning approach using blurring image techniques for Bluetooth-based indoor localisation. Information Fusion. 2023;91:173-86.

[13] Babayigit B, Abubaker M. Image-based vulnerability detection based on a hybrid deep learning model in the Industrial Internet of Things using convolution neural network and transformer architectures. Engineering Applications of Artificial Intelligence. 2026;176:114803.

[14] Kohonen T, Oja E, Simula O, Visa A, Kangas J. Engineering applications of the self-organizing map. Proceedings of the IEEE. 1996;84(10):1358-84.

[15] Chushig-Muzo D, Soguero-Ruiz C, Engelbrecht AP, Bohoyo PDM, Mora-Jimenez I. Data-driven visual characterization of patient ´ health-status using electronic health records and self-organizing maps. IEEE Access. 2020;8:137019-31.

[16] Chon TS. Self-organizing maps applied to ecological sciences. Ecological informatics. 2011;6(1):50-61.

[17] Astudillo CA, Oommen BJ. Topology-oriented self-organizing maps: a survey. Pattern analysis and applications. 2014;17(2):223- 48.

[18] Achutha M, Das B. Topological Activation Maps for Visual Representation Learning from Tabular Data. In: International Conference on Multi-disciplinary Trends in Artificial Intelligence. Springer; 2025. p. 1-12.

[19] Sharma A, Kumar D. Classification with 2-D convolutional neural networks for breast cancer diagnosis. Scientific Reports. 2022;12(1):21857.

[20] Briner N, Cullen D, Halladay J, Miller D, Primeau R, Avila A, et al. Tabular-to-image transformations for the classification of anonymous network traffic using deep residual networks. IEEE Access. 2023;11:113100-13.

[21] Sun B, Yang L, Zhang W, Lin M, Dong P, Young C, et al. Supertml: Two-dimensional word embedding for the precognition on structured tabular data. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops; 2019. p. 0-0.

[22] Mamdouh A, El-Melegy M, Ali S, Kikinis R. Tab2visual: overcoming limited data in tabular data classification using deep learning with visual representations. arXiv preprint arXiv:250207181. 2025.

[23] Alenizy HA, Berri J. Transforming tabular data into images via enhanced spatial relationships for CNN processing. Scientific Reports. 2025;15(1):17004.

[24] Damri A, Last M, Cohen N. Towards efficient image-based representation of tabular data. Neural Computing and Applications. 2024;36(2):1023-43.

[25] Zhu Y, Brettin T, Xia F, Partin A, Shukla M, Yoo H, et al. Converting tabular data into images for deep learning with convolutional neural networks. Scientific reports. 2021;11(1):11325.

[26] Gomez-Mart ´ ´ınez V, Lara-Abelenda FJ, Peiro-Corbacho P, Chushig-Muzo D, Granja C, Soguero-Ruiz C. LM-IGTD: a 2D image generator for low-dimensional and mixed-type tabular data to leverage the potential of convolutional neural networks. arXiv preprint arXiv:240614566. 2024.

[27] Halladay J, Cullen D, Briner N, Miller D, Primeau R, Avila A, et al. Bie: binary image encoding for the classification of tabular data. Journal of Data Science. 2025;23(1):109-29.