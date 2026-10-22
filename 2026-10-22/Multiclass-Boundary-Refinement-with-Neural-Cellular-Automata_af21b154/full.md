# Multiclass Boundary Refinement with Neural Cellular Automata for Remote Sensing Land Cover Segmentation

Pascoassis Souza Santos Meira<sup>1</sup> Gilson Antonio Giraldi<sup>1</sup>

Rafael Duarte Coelho dos Santos<sup>2</sup>

<sup>1</sup>Laborat´orio Nacional de Computa¸c˜ao Cient´ıfica (LNCC), Petr´opolis, RJ, Brazil <sup>2</sup>Instituto Nacional de Pesquisas Espaciais (INPE), S˜ao Jos´e dos Campos, SP, Brazil

October 2026

## Abstract

Semantic segmentation of remote sensing imagery often places class boundaries a few pixels away from where they truly lie, and this position error is concentrated in the thin band measured by Boundary IoU (BIoU). We treat boundary refinement as the iterated application of a small learned local rule: the Boundary Refinement Neural Cellular Automaton (BRNCA), that starts from the class probabilities of a frozen, pre-trained segmenter and corrects them cell by cell. We generalise the binary rNCA to the multiclass case by keeping the visible channels on the probability simplex through an Aitchison (log-space) update and by adding free hidden channels. The best configuration improves the mean BIoU over the base segmenter by 2.36 points (from 46.63 to 48.99). The gain is concentrated on the irregular woodland class (+8.5 points BIoU); this suggests that position error is correlated with the fractal dimension of the class boundaries.

## 1 Introduction

Land use and land cover (LULC) maps underpin the monitoring of territorial change and evidence-based environmental policy. In remote sensing, producing them is a multiclass semantic segmentation problem: every pixel receives exactly one of several categories, so every region is delimited by the boundaries it shares with its neighbours, and the quality of the map depends directly on where those boundaries are placed. Modern segmenters, from convolutional networks [7, 15] to Vision Transformers [4] and foundation-model encoders [11, 14], are good at deciding what a region is and less reliable at deciding its boundary.

We distinguish two kinds of error (Figure 1). A semantic error assigns a whole region to the wrong class. A position error occurs when the interiors of two neighbouring regions are classified correctly but the decision boundary separating them is displaced with respect to the true one. Global metrics such as IoU mix both efects and are dominated by interiors. Boundary IoU (BIoU) [3] restricts the computation to a narrow band around the annotated boundary and therefore isolates position error; its complement, Interior IoU (IIoU), measures the remainder of each region. State-of-the-art segmenters typically score a much lower BIoU than IoU [3], demonstrating that boundary refinement remains an open challenge.

Two further obstacles are specific to remote sensing. First, the contours of natural classes (vegetation, rock, water) are highly irregular and can be characterised by fractal dimension [9, 19]. Second, a segmenter sees the scene through a finite window (the raster crop, the backbone patch, the convolution kernel or attention window) that is much smaller than the continuous scene it represents. We quantify the first obstacle through the fractal dimension of the class boundaries (Section 4.1) and address the second with local, iterated refinement.

![](images/0ef139daa18ccd89722b4b1f0a4529fe989c60c044ca9008532fa7668ac637dd.jpg)  
Figure 1: Position and semantic errors in a real segmentation. Left: input image; middle: error regions overlaid on it; right: error regions alone. Position errors are thin slivers along region contours; the large light-green blob is a semantic error (sparse vegetation predicted over an area of dense vegetation).

Cellular automata (CA) [17, 21] apply one simple rule uniformly to the immediate neighbourhood of every cell. Such a rule works at the native image resolution, and can be directed to the boundary, leaving interior decisions that are already correct untouched. A Neural Cellular Automaton (NCA) [10] parametrises the rule with a small neural network and trains it by back-propagation through time (BPTT) [20]; unrolling the automaton’s time steps, it can be seen as a recurrent residual convolutional network. rNCA [16] showed that an NCA can repair binary segmentation masks. Here we generalise the idea to multiclass segmentation with the Boundary Refinement Neural Cellular Automaton (BRNCA) and ask a deliberately narrow question: what is the smallest NCA that moves boundaries to the right place?

## Contributions.

1. A multiclass boundary refinement NCA (BRNCA) for a frozen base segmenter: the cell state holds the class distribution on the probability simplex, updated through an Aitchison (log-space) step, plus free hidden channels (Section 3).

2. A minimal design without sample pool, alive mask or stochastic updates, with the rationale for each removal in the setting of boundary position (Section 3.3). Also, we conduct a brief categorisation of the dynamic behaviour of BRNCA.

3. A controlled core ablation over the number of hidden channels $( m \in \{ 1 0 , 1 5 , 2 0 \} )$ ) and the number of neurons in the hidden layer of the update MLP $( d _ { h } \in \{ 6 4 , 1 2 8 , 2 5 6 \} )$ , with the original identity+Sobel perception bank, and a per-class BIoU/IIoU analysis related to the fractal dimension of the class boundaries (Sections 4.3 and 4.4).

## 2 Related Work

Neural cellular automata. Mordvintsev et al. [10] introduced NCAs to grow and regenerate a 64 × 64 RGBA image. Every cell holds a state vector that is updated, in parallel and with the same rule, by two modules: perception, which aggregates the neighbours with a fixed bank of depthwise kernels (identity and Sobel), and update, a shallow network that maps the perception vector to a state increment. Later works prune the machinery of the original NCA: Variational NCA [13] dispenses with the alive mask, and NoiseNCA [12] additionally removes the stochastic asynchronous update, injecting noise in the perception stage instead.

NCAs for segmentation. rNCA [16] is the closest work to our proposal. It applies an NCA to the output of a dense segmenter to repair binary masks of medical images, conditioning the rule on the input image; it targets topological errors (fragmented vessels, broken myocardium), measured with Dice, clDice and Betti errors. We instead target boundary position in the multiclass setting of land cover mapping.

![](images/2ef680fabd62c636e3f73cd52f7264092a40c00f7ea9c56dcf6065e32307c9ae.jpg)  
Figure 2: One BRNCA step. The frozen segmenter provides the initial state once; the loop (solid arrows) perceives the state, computes an increment $\delta = \left( \delta ^ { p } , \delta ^ { h } \right)$ with a shallow MLP conditioned on the image, and updates every cell synchronously (Equation (2)).

Boundary refinement. The dense CRF [6] is the classic multiclass refiner: a unary term from the base prediction and a pairwise term that penalises label disagreement between pixels close in space and appearance. CRF-as-RNN [23] parametrises the CRF as a diferentiable recurrent network, and ClassWise-CRF [24] adapts it to remote sensing. Model-agnostic refiners post-process the output of any segmenter: SegFix [22] replaces the labels of boundary pixels with those of interior pixels, guided by predicted boundary and direction maps, and SegRefiner [18] refines coarse masks through a discrete difusion process.

## 3 Boundary Refinement Neural Cellular Automaton (BRNCA)

## 3.1 Setting and state

Let $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ be an RGB image and $g$ a segmenter with $n _ { c }$ classes, pre-trained and frozen, whose softmax output $p _ { 0 } = g ( x )$ lies, at every pixel, on the probability simplex $\Delta ^ { n _ { c } - 1 }$ . BRNCA is an NCA whose cells are the pixels. The state of a cell at step t is

$$
u _ { t } = ( p _ { t } , h _ { t } ) , \qquad p _ { t } \in \Delta ^ { n _ { c } - 1 } , \quad t = 0 , 1 , \ldots , T ,\tag{1}
$$

where $p _ { t }$ represents the probability classification tensor of $n _ { c }$ visible channels (initialised with the base prediction) and the m hidden channels $h _ { t }$ are initialised at zero and evolve freely as short-term memory, without direct supervision. The refined segmentation is $p _ { T }$

## 3.2 Update rule

Figure 2 shows one step of the automaton; dashed arrows are computed once, solid arrows form the iterated loop. Three modules act on the state.

Perception. The fixed bank ${ \cal K } = \{ \mathrm { I d } , S _ { x } , S _ { y } \}$ of $3 \times 3$ depthwise kernels (identity and the two Sobel kernels, as in the original NCA [10]) is applied to every channel of $u _ { t }$ , giving the perception vector $z _ { t } = ( K * u _ { t } ) _ { K \in \mathcal { K } }$

Conditioning. A small trainable convolutional encoder maps the RGB image to a context vector $E ( x )$ , computed once. It gives the rule access to appearance information that is absent from the class probabilities.

Update. A shallow MLP $f _ { \theta }$ , shared by all cells (a 1 × 1 convolution), maps $[ z _ { t } , E ( x ) ]$ to an increment $\delta = ( \delta ^ { p } , \delta ^ { h } )$ . Its hidden layer has $d _ { h }$ neurons. A plain additive step would leave the simplex, so the visible channels are updated by an Aitchison perturbation [1] in log-probability space, and the hidden channels are updated additively:

$$
p _ { t + 1 } = { \mathrm { s o f t m a x } } { \bigl ( } \log p _ { t } + \delta ^ { p } { \bigr ) } , \qquad h _ { t + 1 } = h _ { t } + \delta ^ { h } , \qquad \delta = f _ { \theta } { \bigl ( } z _ { t } , E ( x ) { \bigr ) } .\tag{2}
$$

Therefore, the rule re-weights the current class evidence multiplicatively, so a class can be suppressed or promoted without ever leaving the simplex.

## 3.3 Design choices: a minimal BRNCA

Earlier NCAs for growth and repair carry extra machinery: an alive mask, a sample pool and stochastic asynchronous updates [10, 16]. We drop all three components. These are design decisions justified by our setting; they are not part of the core ablation of Section 4.3.

No alive mask. In Growing NCA the alive mask separates cells that take part in growth from empty background, and rNCA reuses its visible probability channel for that role. Variational NCA [13] and NoiseNCA [12] dispense with it. In our setting the initial state is the dense output of a segmenter, not a sparse seed, so every cell carries a meaningful class distribution from $t = 0$ and there is no growth front to gate. There is a second argument specific to boundary position. When class boundaries are misaligned, the incorrectly included pixels typically belong to regions classified with high confidence, meaning the model is sure of the category but blind to its precise limits. A mask driven by the state, whether by confidence or by margin, would tend to freeze exactly the cells that must change class for the boundary to move, and would tie the region where the boundary may move to the region where the base model already placed it. Leaving every cell free lets $f _ { \theta }$ decide, from perception and conditioning, where to act.

No sample pool. The pool maintains previously evolved state tensors to stabilise training across long trajectories and promote pattern persistence, feeding back a mixed batch comprising states sampled from the pool alongside fresh predictions generated directly by the base segmenter. However, in our boundary refinement setting, every training iteration starts directly from the base segmenter’s output $p _ { 0 }$ and a zero hidden state, matching the inference procedure while eliminating additional RAM–VRAM transfer overhead. Without the pool, the rule is never trained on states beyond the maximum training step $( T = 6 4 )$ ; we measure the impact of this choice in Section 4.6.

Synchronous updates. Unlike Growing NCA, which updates cells asynchronously, governed by a stochastic update probability and gated by an alive mask, in BRNCA every cell updates synchronously at every time step $( p _ { \mathrm { s t o c h } } = 1 )$ without pixelwise dropout or spatial gating.

![](images/b43fd15ce18cfe6298a6db388b12adca0146f66d5e467427b4d6576b79faf969.jpg)  
Figure 3: Cropped example of a LandCover.ai aerial image and its annotation mask, with colour legend for the five segmentation classes.

## 3.4 Training and inference

The automaton runs for T steps and is trained by BPTT with the cross-entropy between the final visible channels $p _ { T }$ and the ground truth, averaged over pixels (computed as the negative log-likelihood of $p _ { T } .$ which already lies on the simplex). During training, the number of automaton steps T is set between 32 and 64 (so $T = 6 4$ is the maximum training step); at inference, $T = 6 4$ The base segmenter stays frozen throughout.

## 4 Experiments

## 4.1 LandCover.ai Dataset

LandCover.ai [2] consists of aerial orthophotos of Poland (41 images, 25–50 cm/pixel; 8 images of 4200 × 4700 and 33 of 9000 × 9500 pixels) with five classes: background, building, woodland, water and road. It is the benchmark on which the core ablation is performed in Section 4.3. Figure 3 illustrates the pixelwise annotation.

Boundary complexity. To quantify how intricate the annotated boundaries are, we estimate the fractal dimension D of each class boundary by box counting on the contour of the binary classversus-rest mask (extracted with a morphological gradient), fitting log $N ( r ) = D \log ( 1 / r ) + C$ by least squares, where $N ( r )$ is the number of boxes of side r needed to cover the contour. For the LandCover.ai dataset, the per-class values range from D = 1.13 (water) to $D = 1 . 3 3$ (background), so the classes difer moderately, but enough to be ordered: water < building < road < woodland < background.

Boundary complexity is computed on the annotated masks at their native resolution, serving as a key factor for appropriate boundary refinement. The high D value for the background class occurs because it represents the complement of the other four classes; consequently, its contour consists of the interfaces with all of them, meaning D does not reflect the property of a natural land-cover class.

## 4.2 Setup

The base segmenter is a Dense Prediction Transformer [14] with a DINOv2 encoder [11] (DPT– DINOv2), kept frozen. It operates with a input size of $5 1 8 \times 5 1 8$ , whereas BRNCA processes four $2 5 9 \times 2 5 9$ sub-windows that are subsequently reconstructed for direct comparison. On LandCover.ai, the base segmenter achieves 83.69% mIoU and 46.63% BmIoU (Table 1), establishing the floor that BRNCA must improve. We report per-class IoU, BIoU and IIoU along with their class averages (mIoU, BmIoU, mIIoU); gains ∆ represent diferences relative to the base segmenter, expressed in percentage points (pp). BIoU is computed on a band of width $d = 5$ pixels around the annotated contour. We use AdamW [8] with learning rate $1 0 ^ { - 4 }$ . Reported numbers correspond to the epoch with the best BmIoU, with early stopping (patience of 10 epochs) and a maximum of 60 epochs. Experiments ran on single nodes of the SDumont supercomputer (LNCC), each with 6 CPU cores per task, 64 GB of RAM, and four 32 GB Tesla V100 GPUs.

<table><tr><td colspan="5">mIoU base segmenter = 83.69</td><td colspan="4">BmIoU base segmenter = 46.63</td></tr><tr><td>mIoU</td><td></td><td> $d _ { h } = 6 4$ </td><td> $d _ { h } = 1 2 8$ </td><td> $d _ { h } = 2 5 6$ </td><td>BmIoU</td><td> $d _ { h } = 6 4$ </td><td> $d _ { h } = 1 2 8$ </td><td> $d _ { h } = 2 5 6$ </td></tr><tr><td> $m = 1 0$ </td><td></td><td>84.04</td><td>83.98</td><td>84.23</td><td> $m = 1 0$ </td><td>48.41</td><td>48.11</td><td>48.70</td></tr><tr><td> $m = 1 5$ </td><td></td><td>83.86</td><td>84.08</td><td>84.23</td><td> $m = 1 5$ </td><td>47.56</td><td>48.43</td><td>48.99</td></tr><tr><td> $m = 2 0$ </td><td></td><td>84.05</td><td>84.23</td><td>84.31</td><td> $m = 2 0$ </td><td>48.33</td><td>48.82</td><td>48.89</td></tr></table>

Table 1: Core ablation on LandCover.ai: mIoU (left) and BmIoU (right), in $\% .$ , for m hidden channels in the cell state and $d _ { h }$ neurons in the hidden layer of the update MLP. Bold: best cell of each metric.

## 4.3 Core ablation

The minimal BRNCA of Section 3.3 has two free structural choices, which define its core: the number of hidden channels m (memory carried by each cell) and the number of neurons $d _ { h }$ in the hidden layer of the update MLP (capacity of the rule). The perception bank is fixed to identity+Sobel. We evaluate all pairs $( m , d _ { h } )$ with

$$
m \in \{ 1 0 , 1 5 , 2 0 \} , \qquad d _ { h } \in \{ 6 4 , 1 2 8 , 2 5 6 \} ,
$$

that is, nine configurations, with everything else fixed (synchronous updates, no alive mask, no pool, optimiser and number of steps as in Section 3). The experiment addresses two questions: (i) whether every core configuration improves BIoU over the base segmenter; and (ii) how sensitive the gain is to memory (m) and to the capacity of the MLP $\left( d _ { h } \right)$

According to Table 1, every one of the nine configurations improves on the base segmenter in both metrics. BmIoU rises by 0.93 to 2.36 pp (from 46.63 to between 47.56 and 48.99), while mIoU changes by +0.17 to +0.62 pp (from 83.69 to between 83.86 and 84.31). The gain is thus about four times larger on the boundary band than on the whole region, as expected from BRNCA, which moves contours and leaves the semantics of the interiors to the base model. The best BmIoU (48.99) is obtained with $m = 1 5 , d _ { h } = 2 5 6$ and the best mIoU (84.31) with $m = 2 0 .$ $d _ { h } = 2 5 6 ;$ the per-class analysis below uses the former. In absolute terms the position error remains large.

Sensitivity to m and $d _ { h }$ . From Table 1, averaged over m, BmIoU increases monotonically with $d _ { h }$ (48.10, 48.45 and 48.86 for $d _ { h } = 6 4 , 1 2 8 , 2 5 6 )$ , while m has a weaker, non-monotonic efect (48.41, 48.33 and 48.68 for $m = 1 0 , 1 5 , 2 0 )$ . This indicates that contour refinement is limited more by local rule capacity than by auxiliary memory. Even the smallest configuration $( m = 1 0 , d _ { h } = 6 4 )$ captures 75% of the maximum gain. Overall, the ablation robustly shows positive gains across all cells, driven mainly by $d _ { h }$ . This shows that performance depends more on the number of neurons in the hidden layer $\left( d _ { h } \right)$ than on the number of hidden channels (m).

Table 2: Per-class change with respect to the base segmenter, LandCover.ai validation, best epoch, for the best configuration of the ablation by BmIoU $( m = 1 5 , d _ { h } = 2 5 6$ ; Table 1), in pp. D is the fractal dimension of the class boundary (Section 4.1). Classes are sorted by increasing D; the last row is the unweighted mean over classes (∆mIoU, ∆BmIoU, ∆mIIoU).
<table><tr><td>Class</td><td>D</td><td>∆IoU</td><td>∆BIoU</td><td> $\Delta \mathrm { I I o U }$ </td></tr><tr><td>water</td><td>1.129</td><td>+0.01</td><td>+0.21</td><td>0.00</td></tr><tr><td>building</td><td>1.192</td><td>+0.04</td><td>+0.39</td><td>-0.02</td></tr><tr><td>road</td><td>1.260</td><td>-0.05</td><td>+0.82</td><td>-0.04</td></tr><tr><td>woodland</td><td>1.264</td><td>+2.15</td><td>+8.49</td><td>-0.79</td></tr><tr><td>background</td><td>1.330</td><td>+0.55</td><td>+1.89</td><td>-0.32</td></tr><tr><td>Mean</td><td>1.235</td><td>+0.54</td><td>+2.36</td><td>-0.23</td></tr></table>

## 4.4 Where does the gain come from?

Table 2 decomposes the best configuration by class and relates the change in BIoU to the fractal dimension D of each class boundary. The gain is far from uniform. Woodland alone gains 8.49 pp of BIoU; background follows with 1.89 pp, then road (0.82), building (0.39) and water (0.21). Without woodland the mean gain falls from 2.36 to 0.83 pp, so the headline BmIoU is largely coming from the natural irregular class woodland. Global IoU follows the same ordering at a smaller scale (woodland +2.15, background +0.55; building, water and road within ±0.05) and does not decrease appreciably for any class.

Relation to the fractal dimension. The classes are ordered by BIoU gain almost as they are ordered by D: water, building and road come first in the same order in both rankings, and the two classes with the highest D come last, with woodland $( D = 1 . 2 6 4 )$ gaining more than background $( D = 1 . 3 3 0 )$ . We notice that the Spearman coeficient is $\rho = 0 . 9 0$ , and $\rho = 1$ if woodland is left out. However, the Spearman coeficient is merely ordinal, in contrast with the magnitude captured by the Pearson coeficient $r = 0 . 3 8$ . So D is not directly correlated to how large the woodland BIoU gain is, since road and woodland have almost the same dimension (1.260 against 1.264), but gains that difer by a factor of ten (0.82 against 8.49). We conjecture that the pixels that fall inside the BIoU band help explain this discrepancy.

Road is thin and tubular, meaning a large share of its area lies within the band, keeping the band metric close to the global one; furthermore, its remaining contour is regular, which the base segmenter already localises well. By contrast, woodland forms massive, grouped patches, so only a small share of its pixels resides in the band, concentrating the base model’s errors there. Background is a special case, since an improvement of the contours of building, road or woodland is also an improvement of the contour of the background.

Boundary gain and interior degradation. The interior is not free. IIoU does not increase for any class and drops most where the boundary gain is largest: woodland −0.79, background −0.32, road −0.04, building −0.02, water 0.00. The ranking by interior loss is exactly the ranking by BIoU gain (Spearman $\rho = - 1$ between ∆BIoU and ∆IIoU, Pearson $r = - 0 . 9 8 )$ : the classes that benefit most from repositioning the contour are those whose interiors lose most. The loss is small against the gain (on average −0.23 pp against $+ 2 . 3 6 \mathrm { p p } )$ and BIoU and IIoU are measured on diferent pixel populations, so they cannot simply be subtracted; the commensurable evidence is that the global mIoU rises $\left( + 0 . 5 4 \mathrm { p p } \right)$ , so the net efect over all pixels is positive. One plausible reading is that a rule shared by all cells, when it moves a contour, also perturbs interior cells, producing isolated misclassified pixels or small spurious regions within previously correct areas.

## 4.5 How the automaton evolves

Figure 4 follows one crop through the 64 steps used at inference. The crop focuses on anthropogenic classes (buildings, a road and the background around them). After one step $( t = 1$ , when the gain over the base is still only 0.03 pp, so the panel is essentially the base prediction) the building is a smooth, rounded blob with a wide, difuse halo of uncertainty. The largest changes occur between $t = 8$ and $t = 3 2 \colon$ the building acquires straight sides and sharper corners, closer to the polygonal footprint of the label, the edge of the road straightens, and the uncertainty collapses from a broad halo into a narrow line that follows the contour. After t = 32 the changes are subtle at this scale, although the BmIoU gain keeps growing until about t = 62 (Figure 5), so the later steps refine details that are hard to see in a single crop. The figure also shows some drawbacks. Along the north-east side of the building the uncertainty stays broader and the contour ragged, and from about $t = 4 0$ small groups of isolated pixels appear near some contours (top of the crop, small blob at the left of the building).

## 4.6 Behaviour beyond the maximum training step

The automaton is trained and evaluated for a fixed number of steps $( T = 6 4$ at inference). Figure 5 runs the best configuration for twice that number of steps and plots the BmIoU gain at every step. The gain grows monotonically up to a plateau between roughly t = 60 and $t = 7 2$ where it reaches the 2.36 pp of Table 1, and then declines: about 1.96 pp at t = 96 and about 0.60 pp at $t = 1 2 8$ , a quarter of the peak, still positive but heading toward deterioration. Three things follow. First, the maximum training step of $T = 6 4$ , fixed beforehand, falls within the plateau that is best a posteriori. Second, the dynamics is not a convergence to a fixed point that preserves the refined contour; the automaton behaves as a transient that is good at the time it was trained for and that keeps changing, in the wrong direction, afterwards. Third, the decay is gradual compared to the rise in Figure 5, meaning BRNCA does not diverge abruptly; nevertheless, running inference past the maximum training step $( T = 6 4 )$ departs from the training objective and results in a performance loss.

This is the price of the minimal design. Without the sample pool the rule is never trained on states beyond the maximum training step, so those it meets at $t > 6 4$ are outside the training distribution and its behaviour is unconstrained. The same efect motivated the pool in Growing NCA, where it was introduced to make the pattern persist [10].

## 5 Discussion

What the evidence supports. A small, minimal BRNCA applied on top of a frozen segmenter raises BmIoU on LandCover.ai by 0.9 to 2.4 pp while mIoU changes by 0.2 to 0.6 pp, in all nine configurations of the ablation. The efect is therefore robust to the choice of m and $d _ { h }$ . However, the structural choice that clearly matters is the capacity of the rule. The gain is selective: it is concentrated on woodland and is small (below 1 pp) for building, water and road, and the ranking of the classes by gain broadly follows the ranking by the fractal dimension of their boundaries. Because the interior metric falls in the classes where the boundary metric rises, BRNCA behaves as a re-allocation of accuracy toward the boundary band rather than a free improvement, although the net efect over all pixels is positive. The scale of the efect is modest: BmIoU stays below 50%, and BRNCA removes only a small part of the position error of the base segmenter.

What the dynamics tells us about the design. The qualitative view and the behaviour past $T = 6 4$ shed light on two key aspects of the design choices in Section 3.3. First, since every cell updates without an alive mask or pixelwise dropout, the rule is free to shift wrong contours, which is precisely what boundary correction requires; the same choice generates an interior perturbation, causing IIoU degradation by modifying non-boundary regions. Second, dropping the sample pool makes training computationally eficient, and the automaton still achieves its peak performance at the maximum training step $( T = 6 4 )$ , but it leaves the dynamics beyond that step uncontrolled (Section 4.6).

![](images/0e292ecf4f8f05dc1b1d82b1b4b4dc03e552f98f80edeaf28ebd25c005230509.jpg)  
Figure 4: Evolution of the automaton over its 64 steps on LandCover.ai. Panels are read column by column, left column first. Top left: input image and ground-truth label (yellow: background; blue: building; grey: road). The remaining panels show, for $t \in \{ 1 , 8 , 1 6 , 2 4 , 3 2 , 4 0 , 4 8 , 5 6 , 6 4 \}$ , the predicted label (left) and the uncertainty map (right; dark: low, bright: high; defined by the entropy of $p _ { t } )$ .

![](images/6768b0ea080f253337477337fba6f0a177ee3434b0bfb3c741933553813c32ec.jpg)  
Figure 5: Gain of mean BIoU over the base segmenter as a function of the number of automaton steps, for the best configuration $( m = 1 5 , d _ { h } = 2 5 6 )$ , run for twice the 64 steps used at inference. The gain peaks at 2.36 pp, in the plateau between roughly $t = 6 0$ and $t = 7 2$ , and falls to about 0.6 pp at $t = 1 2 8$

Scope and limitations. This study relies on a single frozen base segmenter (DPT–DINOv2) evaluated on one benchmark (LandCover.ai), using three values per axis (m and $d _ { h } )$ in the core ablation. The conditioning encoder and the number of automaton steps are held fixed. The alive mask, the sample pool and the asynchronous update are removed by design and not ablated, so their efect on boundary position is argued but not measured here, apart from the indirect evidence of Section 4.6. The per-class conclusions rest on five classes and a single configuration, and the explanations ofered for them (the share of pixels inside the band, small-scale changes along fractal contours, perturbation of interior cells) are conjectures that this study did not test. BIoU is computed for a single band width. Finally, the mean gains are of the order of one to two BIoU points, and diferences of a few tenths of a point between cells are not interpreted. Comparisons with other boundary refiners and the computational cost of the refinement are not reported in this version.

Future work. Three directions follow. First, restoring the sample pool and studying the convergence and the attractors of the learned dynamics, so as to obtain a stopping criterion more robust than the present number of steps $( T = 6 4 )$ and to let BRNCA refine the boundaries beyond this value. Second, a comparison with the boundary refiners reviewed in Section 2 (for instance dense CRF, SegFix and SegRefiner) on at least two base segmenters, a convolutional U-Net with an EficientNet-B7 encoder and DPT–DINOv2, reporting ∆IoU, ∆BIoU and computational cost. Third, evaluation on further land-cover data, including a new dataset of the APA Petr´opolis protected area built in partnership with ICMBio [5], which has 8 classes without a background, and high annotation precision in boundary position.

## 6 Conclusion

We introduced BRNCA, which casts multiclass boundary refinement as the iteration of a small learned local rule on top of a frozen segmenter and reduces the NCA to its core: class probabilities kept on the simplex by an Aitchison update, free hidden channels, a fixed perception bank and synchronous updates, with no sample pool or alive mask. On LandCover.ai all nine configurations of the $m \times d _ { h }$ ablation improve BmIoU over a DPT–DINOv2 base, by 0.9 to 2.4 pp (46.63 to 48.99 for the best, $m = 1 5 , d _ { h } = 2 5 6 )$ , with mIoU changing by $+ 0 . 2 \ \mathrm { t o } \ + 0 . 6 \mathrm { p p }$ . The capacity of the update rule matters more than the number of hidden channels, and the smallest configuration already keeps three quarters of the best BIoU gain. The gain is concentrated on woodland (+8.5 pp BIoU, 72% of the total over classes); the classes are ordered by gain almost as they are ordered by the fractal dimension of their boundaries, and the gain is accompanied by a small loss of interior IoU that is larger where the boundary gain is larger. The gain peaks at the number of steps used in training (T = 64 in this case) and decays when the automaton runs longer.

## Acknowledgements

We thank the National Laboratory for Scientific Computing (LNCC) for access to the SDumont supercomputer, and the agencies CAPES and FACC for financial support.

## References

[1] John Aitchison. The statistical analysis of compositional data. Journal of the Royal Statistical Society: Series B (Methodological), 44(2):139–160, 1982.

[2] Adrian Boguszewski, Dominik Batorski, Natalia Ziemba-Jankowska, Tomasz Dziedzic, and Anna Zambrzycka. LandCover.ai: Dataset for automatic mapping of buildings, woodlands, water and roads from aerial imagery. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1102–1110, 2021. doi: 10.1109/ CVPRW53098.2021.00121.

[3] Bowen Cheng, Ross Girshick, Piotr Doll´ar, Alexander Kirillov, and Kaiming He. Boundary IoU: Improving object-centric image segmentation evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15334– 15343, 2021.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https://arxiv.org/abs/2010.11929.

[5] Instituto Chico Mendes de Conserva¸c˜ao da Biodiversidade. APA da Regi˜ao Serrana de Petr´opolis, 2026. URL https://www.gov.br/icmbio/pt-br/assuntos/ unidade-de-conservacao/unidades-de-biomas/mata-atlantica/lista-de-ucs/ apa-da-regiao-serrana-de-petropolis.

[6] Philipp Kr¨ahenb¨uhl and Vladlen Koltun. Eficient inference in fully connected CRFs with gaussian edge potentials. In Advances in Neural Information Processing Systems (NeurIPS), volume 24, 2011.

[7] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In 2015 IEEE Conference on Computer Vision and Pattern Recog-

nition (CVPR), pages 3431–3440, Los Alamitos, CA, USA, June 2015. IEEE Computer Society. doi: 10.1109/CVPR.2015.7298965.

[8] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. URL https://arxiv.org/abs/ 1711.05101.

[9] Benoit Mandelbrot. How long is the coast of Britain? statistical self-similarity and fractional dimension. Science, 156(3775):636–638, 1967. doi: 10.1126/science.156.3775.636.

[10] Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. Growing neural cellular automata. Distill, 2020. doi: 10.23915/distill.00023.

[11] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.

[12] Ehsan Pajouheshgar, Yitao Xu, and Sabine S¨usstrunk. NoiseNCA: Noisy seed improves spatio-temporal continuity of neural cellular automata. In Proceedings of the ALIFE 2024: The 2024 Conference on Artificial Life, page 57. MIT Press, 2024.

[13] Rasmus Berg Palm, Miguel Gonz´alez-Duque, Shyam Sudhakaran, and Sebastian Risi. Variational neural cellular automata, 2022. URL https://arxiv.org/abs/2201.12360.

[14] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12179–12188, 2021.

[15] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 234–241. Springer, 2015.

[16] Malte Silbernagel, Albert Alonso, Jens Petersen, Bulat Ibragimov, Marleen de Bruijne, and Madeleine K. Wyburd. RNCA: Self-repairing segmentation masks. In Yuankai Huo, Mingchen Gao, Chang-Fu Kuo, Yueming Jin, and Ruining Deng, editors, Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, volume 315 of Proceedings of Machine Learning Research, pages 2474–2495. PMLR, 08-10 Jul 2026. URL https://proceedings.mlr.press/v315/silbernagel26a.html.

[17] John von Neumann and Arthur W. Burks. Theory of self-reproducing automata. University of Illinois press Urbana, 1966.

[18] Mengyu Wang, Henghui Ding, Jun Hao Liew, Jiajun Liu, Yao Zhao, and Yunchao Wei. SegRefiner: Towards model-agnostic segmentation refinement with discrete difusion process. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https: //arxiv.org/abs/2312.12425.

[19] Xinhua Wang, Botao Yuan, Zhuang Li, and Heqi Wang. A fractal curve-inspired framework for enhanced semantic segmentation of remote sensing images. Sensors, 24(22):7159, 2024. ISSN 1424-8220. doi: 10.3390/s24227159.

[20] Paul J Werbos. Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10):1550–1560, 1990.

[21] Stephen Wolfram. A New Kind of Science. Wolfram Media, Champaign, IL, 2002. ISBN 978-1579550080.

[22] Yuhui Yuan, Jingyi Xie, Xilin Chen, and Jingdong Wang. SegFix: Model-agnostic boundary refinement for segmentation. In European Conference on Computer Vision (ECCV), 2020. URL https://arxiv.org/abs/2007.04269.

[23] Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip H.S. Torr. Conditional random fields as recurrent neural networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1529–1537, 2015.

[24] Qinfeng Zhu, Yunxi Jiang, and Lei Fan. ClassWise-CRF: Category-specific fusion for enhanced semantic segmentation of remote sensing imagery. Neural Networks, page 108485, 2025.