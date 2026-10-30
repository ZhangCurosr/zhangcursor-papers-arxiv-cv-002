# HairDeck: Animation-Ready Stylized Hair Cards from a Single Image

Yuze He<sup>1,∗</sup> Yixiang Wei<sup>1,∗</sup> Wang Zhao<sup>1</sup> Chuhao Chen<sup>2</sup> Yanning Zhou<sup>3,†</sup> Yong-Jin Liu<sup>1,†</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Carnegie Mellon University <sup>3</sup>The Chinese University of Hong Kong

![](images/c3248149fef036a231c159b3b3dfc2d9b23f1fd0998bf529c66e3cb3908b5e9f.jpg)  
Figure 1. HairDeck turns a single image into a hairstyle of stylized hair cards, together with the spring-bone chains that animate it. Center: the generated spring-bone chains drawn over the hair, fainter where they run inside it. Hair colors are taken from the input images fo display. Videos are on the project page.

## Abstract

The hair ofstylized characters in games and avatars is built from hair cards: artist-placed ribbons whose geometry is the visible hair, animated by spring bones. Existing imagebased methods produce realistic hair or a singlefused mesh, while CHARM, the only generator of stylized hair cards, decodes a hairstyle autoregressively as a long sequence of control points. Observing that a card hairstyle is an unordered set of artist-defined primitives, we propose HairDeck, which generates it as one: each card is a single token, and a set diffusion transformer denoises all cards together. From a single image, HairDeck generates the cards and the spring-bone chains that drive them, yielding an animatable

asset that exports to ready-to-use downstream formats. On CHARM’s own test set, it lowers Chamfer distance by 44% and card-orientation error by more than half while running 33 times faster, andfrom single images it outperforms CHARM, image-to-3D pipelines and strand-based methods. Project page: https://hyzcluster.github.io/ HairDeck

## 1. Introduction

3D generators now produce convincing shapes from a single image [71, 72, 77, 83], but not the structure that artists build assets from. The hair of stylized characters in games and avatars is a clear example. It is made of hair cards: a few dozen to over a hundred ribbons, each placed by an artist, whose geometry is the visible hair. Unlike the alphatextured hair cards of realistic hair, flat strips whose texture stands in for many strands [20, 60, 88], these cards have a rhombic or triangular cross-section whose width and thickness vary along the card. Such hair is commonly animated with spring bones: short chains of bones hung from the head whose tips are simulated every frame, so that the hair sways and settles. Similar simulations ship as Dynamic Bone in Unity [18], PhysBones in VRChat [61], Kawaii Physics in Unreal Engine [36] and a node in Godot [10]. Artists place every card and rig every chain by hand.

Existing methods do not produce this asset. Hair generators output realistic strands [13, 42, 49], including recent ones that accept stylized images [28, 32, 44, 65], and card extraction turns such strands into alpha-textured cards for realistic hair [60, 88]. Image-to-3D generators [72, 77] output a single fused mesh with no cards to animate. CHARM [15] is the only generator of stylized hair cards and introduced the benchmark for them. It generates the control points of a hairstyle one at a time, as a sequence of hundreds to thousands of discretized tokens.

We observe that a card hairstyle is an unordered set of artist-defined primitives, and we generate it as one. Like CHARM, we describe each card by its control points (position, width and thickness), and we add its cross-section type and its root normal, the width direction at the root, so each card keeps the cross-section and the orientation the artist gave it. What we change is the unit of generation, with two designs. First, each card becomes one continuous token: a card VAE encodes its variable-length control points into a latent, and the token holds this latent with the card’s root position, root normal and cross-section type, so a hairstyle has one token per card, about 70 on average. Second, all tokens of a hairstyle are generated together: a set diffusion transformer without positional embedding denoises them with flow matching [25, 27], and an existence flag on each token decides how many cards appear. Each design makes generation more accurate and faster. Together they amount to a simple rule for assets made of artist-defined primitives: one token per primitive, and no order among them.

A single image shows the hair relative to the head, so we predict the cards in a head frame anchored to the face, which improves where the hair sits without changing its shape. Images also come in many drawing styles, so we widen the appearance of our training renders in two ways: synthetic style augmentation, which improves accuracy on stylized inputs, and renders restyled by an image-editing model [68], which keep the card structure stable across styles.

Cards alone do not move. We generate the spring-bone chains with the same recipe: each chain becomes one token holding its path, joint count and stiffness, and a second set diffusion transformer generates all chains of a hairstyle together from its card tokens. With a fixed geometric skinning, the cards and chains form an animatable asset that exports to ready-to-use downstream formats.

In summary, our contributions are:

• A card token for stylized hair: a representation that keeps the cross-section type and the root normal, and a card VAE that encodes the variable-length control points into one continuous latent and decodes the card by arc length.

• A set formulation of hair-card generation: a set diffusion transformer without positional embedding generates all card tokens of a hairstyle together, and an existence flag on each token decides how many cards appear.

• Single-image generation with a face-anchored head frame, synthetic style augmentation and restyled renders.

• Spring-bone chains generated as a second set conditioned on the card tokens, which makes the output animatable.

We evaluate on CHARM’s test set with its protocol and also measure the orientation of every card against the original artist meshes. HairDeck lowers Chamfer distance by 44% and card-orientation error by more than half while running 33 times faster. From a single image, it outperforms CHARM, image-to-3D pipelines and strand generators.

## 2. Related Work

Strand-based hair modeling. Hair reconstruction methods can recover strands from multi-view images or videos [31, 35, 41, 48, 57, 70, 80] and from single images [50, 69, 86, 87]. Generative models synthesize strands from latent codes [13, 90], text [49] and images [42]. Recent ones accept stylized or anime images [28, 32, 44, 65], but their output is still strands, the representation of realistic hair. The stylized hair we generate uses cards instead.

Hair cards. Real-time engines render realistic hair with textured strips [20], the alpha-textured hair cards whose texture stands in for many strands. Zheng et al. [88] and Strands2Cards [60] convert strand grooms into such cards, HairLRM [44] applies the former to its generated strands, CGHair [33] reconstructs card-based hair from multi-view images, Lee et al. [22] inpaint strips from the sparse strands of a single portrait with a diffusion model on the scalp UV map, HairCS [30] turns cards back into strands, and production tools ship card generators for grooms [8, 46]. Stylized hair uses a different card, whose geometry is the visible hair. CHARM [15] introduced the generation of such cards with a control-point representation, a dataset of anime hairstyles (the benchmark we use) and an autoregressive transformer that emits the control points of a hairstyle one at a time. We add the cross-section type and the root normal to its description of a card and change the unit of generation.

Stylized 3D characters. Image-based methods for anime characters reconstruct stylized heads [5] or full characters [38, 64], and others separate the body, clothing and hair into components [14, 16, 40]. In all of them the hair is a continuous surface, fused with the character or as a separate component, rather than a set of cards. Earlier work modeled anime and cartoon hair from sketches, skeletons or parametric clumps [34, 43, 76] or as 2.5D layers [79].

![](images/cf6c784c2cf71226ac3d8969055a19be14b65f0a6891656c9ac6790a3c503f85.jpg)  
Figure 2. Overview. Left: a hairstyle is a set of hair cards. Each card, a polyline of control points with width and thickness, becomes one token holding its root position, root normal, card VAE latent z, cross-section type and existence flag. Middle: the set DiT denoises the token set with flow matching, attending across cards and to the condition, a point cloud or an image. Empty slots let the model decide how many cards to generate. Right: the card VAE decodes the tokens into cards, and the chain DiT generates their spring-bone chains.

Structured and set generation. Native 3D generators encode shapes as latent sets [4, 81, 83, 85] or structured latents [71, 72] and output a single surface. Assets built from discrete elements are mostly generated autoregressively: meshes as sequences of face or vertex tokens [6, 12, 45, 66, 84], ordered along triangle strips and UV charts in Strips as Tokens [73], and shapes as sequences of primitives [78]. Other work generates the elements of an asset together. 3DTopia-XL [7] gives each primitive one token and denoises the tokens with a diffusion transformer without positional encoding. StrokeFusion [89] encodes each sketch stroke with a VAE and diffuses the unordered set of stroke latents with presence flags. DiffuScene [59], SALAD [21], MeshFlow [56], BrepGen [74] and PartCrafter [24] jointly generate the objects of a scene, part latents, the triangles of a mesh, the faces of CAD models and the parts of objects. We build on this set formulation and design it for hair cards: a card VAE turns each card into one token, the set is conditioned on a point cloud or a single image, and a second set of spring-bone chains is conditioned on the cards. On the same tokens, set generation is both more accurate and faster than autoregressive generation.

Rigging and secondary motion. Automatic rigging predicts a skeleton and skinning weights for a given mesh [11, 26, 51, 52, 55, 75]. UniRig [82] also predicts spring bones and their physical attributes on VRoid characters [39], and DRiVE [54] generates skeletons that include hair and skirt bones for anime characters represented as 3D Gaussians. Both $\mathrm { r i g }$ an existing asset. We generate the spring-bone chains together with the hair they drive, conditioned on the generated cards, and simulate them with the spring-bone update of the VRM specification [62, 63].

## 3. Method

We propose HairDeck, which generates the cards of a hairstyle and the spring-bone chains to animate them (Fig. 2). Each card is encoded into one continuous token (Sec. 3.1), and a set diffusion transformer (set DiT) generates all card tokens together, conditioned on a single image (Sec. 3.3) or on a point cloud (Sec. 3.2), the input of CHARM’s benchmark. The card VAE decodes each token back into control points, from which we build the mesh. A second set DiT, the chain DiT, generates the spring-bone chains conditioned on the card tokens, and a fixed geometric skinning attaches the cards to them (Sec. 3.4).

## 3.1. Hair cards as tokens

Cards. Like CHARM [15], we describe a hair card by a polyline of n control points $\left( \mathbf { p } _ { i } , w _ { i } , h _ { i } \right)$ , each with a position $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ , a width $w _ { i }$ and a thickness $h _ { i } .$ . We add two attributes that this description leaves out: the cross-section type s, rhombic or triangular, instead of one rhombic crosssection for all cards, and the root normal ${ \bf d } _ { 1 }$ , the width direction of the card at its root, read from the artist mesh instead of being estimated from the first three control points. To build a card mesh, we propagate ${ \bf d } _ { 1 }$ along the polyline with a rotation-minimizing (Bishop) frame [1], giving the width direction $\mathbf { d } _ { i }$ at every control point, and sweep the cross-section along it.

Card VAE. Cards differ in their number of control points, while the generator needs a token of fixed size. A card VAE encodes the control points of a card, relative to its root position $\mathbf { p } _ { 1 } .$ , into a latent $\mathbf { z } \in \mathbb { R } ^ { 3 2 }$ (Fig. 3, top). Its encoder is a transformer over the control points, each tagged with its arc-length fraction $u _ { i } \in [ 0 , 1 ]$ ], together with a learned summary token whose output gives the mean and log-variance of z. The decoder predicts the number of control points n from z with a classification head and evaluates the card at n uniformly spaced arc-length fractions, embedded with lowfrequency Fourier features [58], with a transformer modulated by z through adaLN-Zero [37]. The loss is a squared error on positions, widths and thicknesses in world units, plus a cross-entropy on n and a small KL term [19].

![](images/b61fd1903f57413945e489c76d9989575ebf93331446ab5bb18ba12c01159590.jpg)  
Figure 3. Network design. Top: the card VAE. The encoder reads the control points of one card with their arc-length fractions $u _ { i } ,$ , plus a learned summary token [CLS], whose output alone gives the mean and log-variance of the 32-d latent z. The decoder predicts the number of points n and evaluates the card at n uniformly spaced arc-length fractions, modulated by z (adaLN-Zero). Bottom: both DiTs use the same block design (left), modulated by the timestep and the pooled condition. The set DiT predicts the velocity of 192 card token slots without positional embedding, conditioned on a point cloud or an image. The chain DiT does the same for 48 spring-bone chain slots conditioned on the card tokens sampled by the set DiT (dashed arrow). Empty slots (dashed boxes) have existence −1, filled ones +1.

Card tokens. Each card becomes a token

$$
\mathbf { x } = ( \mathbf { p } _ { 1 } , \mathbf { d } _ { 1 } , \mathbf { z } , s , e ) \in \mathbb { R } ^ { 4 0 }\tag{1}
$$

that holds its root position and root normal, its latent, its cross-section type s (−1 for rhombic, +1 for triangular) and an existence flag $e = + 1$ . A hairstyle is a set of N = 192 token slots, and an empty slot is all zeros except e = −1. After sampling, we read s and e by their sign.

## 3.2. Set diffusion of hair cards

Set diffusion transformer. The set DiT, a diffusion transformer [37], denoises the N token slots of a hairstyle together (Fig. 3, bottom). It uses no positional embedding, so it is equivariant to permutations of the slots, and no order of the cards is ever defined. Each block applies self-attention across the slots, cross-attention to the condition tokens C and an MLP, all modulated by the timestep and the pooled condition through adaLN-Zero. For a point cloud, C is the output of a frozen Dora-VAE encoder [4]; for an image, it comes from an image encoder (Sec. 3.3).

Flow matching. We train the generator with flow matching [25, 27]. For the token set $\mathbf { X } _ { 0 }$ of a training hairstyle and Gaussian noise $\epsilon ,$ the network $\mathbf { v } _ { \theta }$ predicts the velocity along the straight path between them:

$$
\mathbf { X } _ { t } = \left( 1 - t \right) \mathbf { X } _ { 0 } + t \mathbf { \epsilon } ,\tag{2}
$$

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { \mathbf { X } _ { 0 } , \epsilon , t } \left. \mathbf { v } _ { \theta } ( \mathbf { X } _ { t } , t , \mathbf { C } ) - ( \epsilon - \mathbf { X } _ { 0 } ) \right. _ { \mathbf { w } } ^ { 2 } , } \end{array}\tag{3}
$$

where $\left\| \cdot \right\| _ { \mathbf { w } }$ weights the root position and the existence flag by 4 and the other components by 1. We drop the condition in training to allow classifier-free guidance [17]. At test time we integrate with the Euler method and guidance and decode every slot with $e > 0$ . The existence flags let the card count settle during denoising together with the shapes.

## 3.3. Generation from a single image

Image condition. The image model is a set DiT of the same architecture, conditioned on the output tokens of a frozen DINOv3 ViT-L/16 [47] applied to a square crop around the hair and the face. For in-the-wild images, SAM 3 [3] provides the hair and face masks.

Head frame. The size and position of a hairstyle follow its head, whereas the dataset frame scales every character to unit height. We therefore predict the cards in a head frame anchored to the face (Fig. 5). Each character is mapped by $\mathbf { p } \mapsto k \mathbf { p } + \mathbf { b }$ , where $k = W / w$ scales its face width w to the median face width W of the training set and b moves the top center of its face to a common point $O ;$ widths and thicknesses scale with k. On rendered test characters, we map the output back with each character’s own k and b, which places the hair on its real head.

![](images/06cf9a0b0f95ec4c9c989de5f3ac2a27fe5be7825b29604c5fda125c93d04021.jpg)  
Figure 4. Point clouds from CHARM’s test set. CHARM and HairDeck output hair cards, colored per card like the artist ground truth; MeshAnything V2, DeepMesh and BPT output meshes. All results are shown on a fixed head at each character’s placement.

![](images/7b05d723dee3b9a82968b1006a7199d9a2fbce1000e1dca349cf6e1f40c7e3b3.jpg)  
Figure 5. Head frame. (a) The dataset frame scales every character to height 1, so heads differ in size and height; the right character’s ears count toward its height and lower its head. (b) In the head frame, all characters share the face width W and face top O.

Training images. We render every training character from 8 views in Blender [2], with random body poses, hair motion from spring-bone simulation, and cameras from the front to the back of the head; the target is always the restpose card set. We widen the appearance of the renders with synthetic style augmentation, such as halftone, XDoG [67] line drawings, flat colors and painterly filters (Sec. A), and we mix in renders restyled into three drawing styles, gacha (mobile-game) art, thick paint and semi-realistic painting, by an image-editing model [68].

## 3.4. Spring-bone chains

Cards alone are static, but the hair of game and avatar characters must move with them. Artists achieve this with spring-bone chains, sequences of joints hung from the head whose simulated motion the cards follow, and rig them by hand for every hairstyle; we generate the chains so that the output can be animated directly.

Chain tokens. A chain becomes a 27-dimensional token of its path (8 points by arc length), joint count, stiffness and existence flag, and a hairstyle a set of up to 48 chains.

Chain generation. The chain DiT has the same block design as the set DiT and generates the chain tokens of a hairstyle in the head frame with flow matching, crossattending to its card tokens (Fig. 3). It is trained on the ground-truth card tokens and, at test time, reads the generated ones; cards from a point cloud are first mapped into the head frame (Sec. A).

Skinning and simulation. After generation, a fixed geometric skinning binds the cards to the chains. We place the joints uniformly along each generated path. A card follows the chain closest to the lower 60% of its centerline if their mean distance is below 0.4 face widths, and otherwise moves rigidly with the head; each of its points follows the nearest segment of that chain, blended linearly with the neighboring segment within half a segment of each joint, with a weight that rises from 0 at the chain’s root to 1 over 0.05 face widths of arc length. We tune these two thresholds against the artists’ skin weights in the training data.

![](images/dce7a7bffdbaa07a89af6ca1bfc97822ec69d85f00c43dcff8fb41e2a3c9109a.jpg)  
Figure 6. Single images: two in-the-wild illustrations (top) and an AI-generated photorealistic portrait (bottom). Insets show the outlined regions with mesh edges. TRELLIS.2 and Hunyuan3D-2.1 take the hair region as input, and TRELLIS.2 outputs a flat plate facing the camera for both illustrations. Each card has its own color.

We simulate the chains with the VRM spring-bone update of UniVRM [62, 63], using the predicted stiffness and the drag and gravity of most VRoid chains.

## 3.5. Implementation details

The card VAE has four encoder and four decoder layers of width 256, the set DiT 12 layers of width 768 (181.3M parameters) and the chain DiT 8 layers of width 384 (30.5M). We train the point-cloud model for 80K steps with batch size 64 and the image model for 100K steps with batch size 48, both with AdamW [29] at learning rate 10<sup>−4</sup>; the image model starts from the point-cloud model. We sample with 50 Euler steps and a guidance scale of 2; Sec. A lists the remaining settings.

## 4. Experiments

## 4.1. Setup

Data. We use the dataset of CHARM [15], built from VRoid characters [39]: we take 36,440 of its training hairstyles and test on its 100 test hairstyles, with the point clouds CHARM released. For single images, we render the front views of the 76 test characters whose complete models are available, and we collect 40 in-the-wild images, anime illustrations sampled at random and stratified by hairstyle.

Metrics. On point clouds we follow CHARM’s protocol: it compares the generated cards with its ground truth, rebuilt from its card templates, by Chamfer distance (CD), earth mover’s distance (EMD), Hausdorff distance (HD) and voxel IoU. As the templates estimate card orientations, we also evaluate against the original artist meshes. Our orientation metric matches every generated card to the artist card closest in CD and averages the angle between their width directions over 32 positions along the card. Root precision and recall are the shares of generated and groundtruth cards whose root position lies within 2% of the character height of a root position in the other set. For single images, we align every output to the ground truth (Sec. D) and report the aligned CD and IoU on surface samples. The failure rate is the share of results that have fewer than 25% of the ground-truth cards (16 cards for in-the-wild images) or leave the scalp bare (hair covers less than 80% of the top or the back of the head); for the image-to-3D generators, whose output is one fused surface without cards, only the scalp criterion applies. We time every method on one NVIDIA RTX A6000 GPU (48 GB) with batch size 1, from the input to the cards, before meshing.

![](images/7552668697a595524e010ad6132d986dcae509120cc732e49aca572e18e41a91.jpg)  
Figure 7. Hairstyles generated by HairDeck from in-the-wild images. Hair colors are taken from the images for display.

Table 1. Generation from point clouds on CHARM’s 100 test hairstyles, with its protocol; orientation on the original meshes (artist cards meshed from root normals: 7.5°); time per hairstyle. Means leave out 1 DeepMesh and 3 BPT outputs that fail to scan.
<table><tr><td>Method</td><td>CD↓</td><td>EMD↓</td><td>HD↓</td><td>IoU↑</td><td>Orient.↓</td><td>Time↓</td></tr><tr><td>MeshAnything V2 [6]</td><td>0.0166</td><td>0.0182</td><td>0.0630</td><td>0.671</td><td></td><td>176 s</td></tr><tr><td>DeepMesh [84]</td><td>0.0357</td><td>0.0381</td><td>0.1001</td><td>0.530</td><td></td><td>1050 s</td></tr><tr><td>BPT [66]</td><td>0.0446</td><td>0.0507</td><td>0.1063</td><td>0.451</td><td></td><td>351 s</td></tr><tr><td>CHARM [15]</td><td>0.0117</td><td>0.0128</td><td>0.0498</td><td>0.753</td><td>40.0°</td><td>33.7 s</td></tr><tr><td>Retrieval</td><td>0.0119</td><td>0.0119</td><td>0.0505</td><td>0.732</td><td>24.8°</td><td></td></tr><tr><td>Card-level AR</td><td>0.0078</td><td>0.0093</td><td>0.0356</td><td>0.823</td><td>16.6°</td><td>6.91 s</td></tr><tr><td>HairDeck</td><td>0.0066</td><td>0.0084</td><td>0.0242</td><td>0.867</td><td>16.8°</td><td>1.01 s</td></tr></table>

## 4.2. Generation from point clouds

Table 1 and Fig. 4 compare HairDeck with CHARM and autoregressive mesh generators on CHARM’s test set. HairDeck lowers CHARM’s CD from 0.0117 to 0.0066 and the orientation error from 40.0° to 16.8°, while taking 1.01 s instead of 33.7 s. Retrieving the training hairstyle with the closest condition reaches a CD of 0.0119, so CHARM is about as accurate as recalling a training hairstyle. The mesh generators output a few large shells (MeshAnything V2, BPT) or triangle fragments (DeepMesh).

The card-level autoregressive model in Tab. 1 generates our card tokens one after another, with about as many parameters as our set DiT. One token per card already lowers CHARM’s CD to 0.0078 and the orientation error to 16.6°. Generating the same tokens as a set lowers CD further to 0.0066 and runs 6.8 times faster than the autoregressive model, which also stops early, with 60.8 cards on average against 68.9 for the set DiT and 67.4 in the ground truth.

Table 2. Generation from single images: aligned CD and IoU on 76 test characters, failure rates on them and on 40 in-the-wild images (Sec. 4.1), and time per image.
<table><tr><td></td><td colspan="2">Test characters</td><td colspan="2">Failure rate↓</td></tr><tr><td>Method</td><td>CD↓</td><td>IoU↑</td><td>Test</td><td>In the wild Time↓</td></tr><tr><td>CHARM [15]</td><td>0.0186</td><td>0.540</td><td>34.2%</td><td>50.0% 67.7 s</td></tr><tr><td>TRELLIS.2 [72]</td><td>0.0139</td><td>0.638</td><td>13.2%</td><td>77.5% 83.0s</td></tr><tr><td>Hunyuan3D-2.1 [77]</td><td>0.0148</td><td>0.638</td><td>30.3%</td><td>75.0% 60.8s</td></tr><tr><td>DiffLocks [42] + cards</td><td>0.0174</td><td>0.592</td><td>17.1% 17.5%</td><td>15.6s</td></tr><tr><td>HairDeck</td><td>0.0089</td><td>0.725</td><td>3.5% 10.0%</td><td>1.33 s</td></tr></table>

## 4.3. Generation from single images

Table 2 compares HairDeck with CHARM’s image model, the image-to-3D generators TRELLIS.2 [72] and Hunyuan3D-2.1 [77], and DiffLocks [42], whose strands we convert into cards. On the rendered test characters, HairDeck has the lowest aligned CD (0.0089, against 0.0139 to 0.0186); it fails on 3.5% of them and on 10% of the in-the-wild images, against 13.2% to 34.2% and 17.5% to 77.5% for the other methods, and takes 1.33 s per image, against 15.6 s to 83.0 s. Cascades that pass a point cloud sampled from an image-to-3D output to CHARM or our point-cloud model reach aligned CDs of 0.0138 to 0.0170 and fail on 22.4% to 32.9% of the test characters (Tab. 5). Figure 6 covers anime illustrations and a photorealistic portrait: the image-to-3D generators produce one fused surface, and the converted DiffLocks strands leave the crown bare and end in blocky clumps on the portrait. Figure 7 shows hairstyles generated from in-the-wild images in many drawing styles.

![](images/9b7570fc1ec8f98aaf03c6d8362c0a6b28195afa9c6b92297a05f01460eef530.jpg)  
Figure 8. Generated spring bones in motion. Left: the generated cards (transparent) with their generated spring-bone chains. Right: a 50° head turn; top row with the chains, bottom row without them. The orange outline marks the hair without spring bones in the same frame.

Table 3. Ablations on point clouds; each row changes one part of HairDeck (all rows in Tab. 6). Raw control points: the card resampled to 64 points by arc length. Turn: largest angle between consecutive segments of a card, median per hairstyle (artist cards: 14.1°).
<table><tr><td>Variant</td><td>CD↓</td><td>IoU↑</td><td>Orient.↓</td><td>Turn</td></tr><tr><td>HairDeck</td><td>0.0066</td><td>0.867</td><td>16.8°</td><td>14.2°</td></tr><tr><td>raw control points as tokens</td><td>0.0068</td><td>0.856</td><td>17.1°</td><td>21.9°</td></tr><tr><td>VAE with index decoding</td><td>0.0079</td><td>0.823</td><td>18.1°</td><td>25.3°</td></tr><tr><td>w/o root normal</td><td>0.0070</td><td>0.850</td><td>26.4°</td><td>13.9°</td></tr><tr><td>w/o existence (card-count head)</td><td>0.0068</td><td>0.859</td><td>16.7°</td><td>14.5°</td></tr><tr><td>CHARM&#x27;s card templates</td><td>0.0068</td><td>0.856</td><td>26.7°</td><td>15.8°</td></tr><tr><td>CHARM&#x27;s encoder (Michelangelo)</td><td>0.0068</td><td>0.853</td><td>17.2°</td><td>14.0°</td></tr><tr><td>small model (54.6M)</td><td>0.0072</td><td>0.847</td><td>17.5°</td><td>14.2°</td></tr></table>

## 4.4. Ablations

Card token. The card VAE latent gives a lower CD than raw control points (Tab. 3; 0.0066 against 0.0068), and its output is as smooth as the artists’ cards (turn 14.2° against 14.1°), whereas raw control points give jagged cards (21.9°). A VAE that decodes the control points by index, as StrokeFusion [89] decodes the points of a stroke, reaches a CD of 0.0079 with jagged cards (25.3°); our arc-length decoding, adaLN modulation and world-unit loss make the difference (Tab. 8). Without the root normal, the orientation error rises from 16.8° to 26.4°. Without the existence flag, a head that predicts the card count first generates too few cards (62.8 against 68.9).

Data, encoder and size. Training on CHARM’s card templates raises the orientation error to 26.7°. With CHARM’s encoder (Michelangelo [85]) or a generator of 54.6M parameters, fewer than CHARM’s 61.2M, CD stays at 0.0068 and 0.0072, far below CHARM’s 0.0117.

Image model. We test the image model on the front views of the 76 test characters and on their restyles into gacha art, thick paint and semi-realistic painting (Fig. 20). Without the head frame, the model must infer the head’s size and position: its CD before alignment rises from 0.0136 to 0.0189 and the overlap (IoU) of the reprojected hair with the input’s hair drops from 0.829 to 0.798, while its aligned CD stays at 0.0089. On the restyles, synthetic style augmentation lowers the aligned CD from 0.0100 to 0.0097, and restyled training images raise root precision from 0.781 to 0.849 at a similar root recall: without them, the model adds cards that start where no artist card does.

Table 4. Spring-bone chains on the 58 test hairstyles with chains, against the artists’ chains (metrics in Sec. 4.5; Ampl.: mean perpoint motion, 0.0157 W for the artists’ chains). Bottom rows: generated cards; –: not measured.
<table><tr><td>Chains</td><td>CD↓ Rec.↑</td><td>Prec.↑</td><td>Motion↑</td><td>Ampl.</td><td>Fit↑</td></tr><tr><td>Procedural</td><td>0.504 0.397</td><td>0.110</td><td>0.538</td><td>0.0473</td><td>0.832</td></tr><tr><td>Retrieval</td><td>0.431 0.289</td><td>0.285</td><td>0.622</td><td>0.0157</td><td>0.811</td></tr><tr><td>Ours, artist cards</td><td>0.279 0.625</td><td>0.585</td><td>0.702</td><td>0.0167</td><td>0.953</td></tr><tr><td>Ours, point cloud</td><td>0.319 0.542</td><td>0.543</td><td>一</td><td>I</td><td>0.952</td></tr><tr><td>Ours, single image</td><td>0.407 0.390</td><td>0.402</td><td>一</td><td></td><td>0.949</td></tr></table>

## 4.5. Spring bones

We evaluate chains on the 58 test hairstyles that have them (Tab. 4 and Fig. 8) against the artists’ chains: by position (CD in face widths W, recall and precision at 0.1 W), by the hair motion under a head turn, nod, jump and sway, and by the fit to the cards (Sec. A). On the artists’ cards, the chain DiT recovers the artists’ chains better than retrieval (recall 0.625 against 0.289, precision 0.585 against 0.285) and moves the hair more like the artists’ chains do: the correlation of the per-point motion is 0.702, against 0.622 for retrieval and 0.538 for a procedural rig that places chains by rules (Sec. D). On generated cards, recall is 0.542 from point clouds and 0.390 from single images, and the fit to the cards stays at 0.95.

![](images/e560914875ccc22311dd9aac1f7e1ae1fe44ff7bcba9fad9498ada223862bb9d.jpg)  
Figure 9. Interpolation between two test point clouds, inputs A and B: the condition tokens of B are matched to those of A and blended with the weight α under each column, and all samples share the initial noise. Hair colors and head placements are blended for display.

## 4.6. Applications

Interpolation. The Dora encoder computes the condition tokens of a point cloud at query points that farthest point sampling picks from it, so the tokens of two point clouds have no correspondence. To interpolate between two conditions, we match the tokens of the second to those of the first by Hungarian matching on the positions of their query points and blend the matched tokens linearly (Fig. 9).

Rigging existing hairstyles. The chain DiT reads only card tokens, so it can also rig card hairstyles that an artist modeled without spring bones; its evaluation on the artists cards in Sec. 4.5 measures this use.

## 5. Limitations and Conclusion

Limitations. HairDeck learns from stylized VRoid hairstyles, so very short hair and the alpha-textured cards of realistic hair lie outside its training data. It generates the geometry and UV coordinates of the cards but no textures. Some VRoid characters model bows and hats with hair cards, so HairDeck generates them as cards when the input shows them; filtering such cards from the training data would avoid this. The chain DiT follows the chain layouts common among VRoid characters and generates fewer chains for unusual ones, such as many chains radiating from the crown. Figure 23 shows failure cases.

Conclusion. HairDeck generates the stylized hair cards of a character and the spring-bone chains that animate them from a single image. Treating a hairstyle as an unordered set of artist-defined primitives, with one token per card and all tokens generated together, makes generation more accurate and faster than autoregressive decoding, and the same recipe generates the chains. Assets built from other artist-defined primitives, such as garments sewn from panels or drawings made of strokes, can be generated in the same way.

## References

[1] Richard L. Bishop. There is more than one way to frame a curve. The American Mathematical Monthly, 82(3):246– 251, 1975. 3, 15

[2] Blender Foundation. Blender 4.2 LTS. https://www. blender.org/download/releases/4- 2/, 2024. 5, 14

[3] Nicolas Carion et al. SAM 3: Segment anything with con cepts. In ICLR, 2026. 4, 14

[4] Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, and Ping Tan. Dora: Sampling and benchmarking for 3D shape variational auto-encoders. In CVPR, 2025. 3, 4, 13

[5] Shuhong Chen, Kevin Zhang, Yichun Shi, Heng Wang, Yiheng Zhu, Guoxian Song, Sizhe An, Janus Kristjansson, Xiao Yang, and Matthias Zwicker. PAniC-3D: Stylized single-view 3D reconstruction from portraits of anime characters. In CVPR, 2023. 2

[6] Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. MeshAnything V2: Artist-created mesh generation with adjacent mesh tokenization. In ICCV, 2025. 3, 7, 21

[7] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie,

Tong Wu, Shunsuke Saito, Liang Pan, Dahua Lin, and Ziwei Liu. 3DTopia-XL: Scaling high-quality 3D asset generation via primitive diffusion. In CVPR, 2025. 3

[8] Epic Games. Hair card generator for grooms in Unreal Engine (Unreal Engine 5.8 documentation). https:// dev.epicgames.com/documentation/en- us/ unreal- engine/hair- card- generator- forgrooms-in-unreal-engine, 2026. 2

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 13

[10] Godot Engine contributors. SpringBoneSimulator3D (Godot Engine 4.4 documentation). https://docs. godotengine.org/en/4.4/classes/class\_ springbonesimulator3d.html, 2025. 2

[11] Zhiyang Guo, Jinxu Xiang, Kai Ma, Wengang Zhou, Houqiang Li, and Ran Zhang. Make-It-Animatable: An efficient framework for authoring animation-ready 3D characters. In CVPR, 2025. 3

[12] Zekun Hao, David W. Romero, Tsung-Yi Lin, and Ming-Yu Liu. Meshtron: High-fidelity, artist-like 3D mesh generation at scale. arXiv preprint arXiv:2412.09548, 2024. 3

[13] Chengan He, Xin Sun, Zhixin Shu, Fujun Luan, Soren Pirk,¨ Jorge Alejandro Amador Herrera, Dominik L. Michels, Tuanfeng Y. Wang, Meng Zhang, Holly Rushmeier, and Yi Zhou. Perm: A parametric representation for multi-style 3D hair modeling. In ICLR, 2025. 2

[14] Yuze He, Yanning Zhou, Wang Zhao, Zhongkai Wu, Kaiwen Xiao, Wei Yang, Yong-Jin Liu, and Xiao Han. StdGEN: Semantic-decomposed 3D character generation from single images. In CVPR, 2025. 3

[15] Yuze He, Yanning Zhou, Wang Zhao, Jingwen Ye, Yushi Bai, Kaiwen Xiao, Yong-Jin Liu, Zhongqian Sun, and Wei Yang. CHARM: Control-point-based 3D anime hairstyle auto-regressive modeling. In SIGGRAPH Asia, 2025. 2, 3, 6, 7, 15, 16

[16] Yuze He, Yanning Zhou, Wang Zhao, Jingwen Ye, Zhongkai Wu, Ran Yi, and Yong-Jin Liu. StdGEN++: A comprehensive system for semantic-decomposed 3D character generation. IEEE TPAMI, 2026. 3

[17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4, 13

[18] Will Hong. Dynamic Bone, version 1.3.4 (Unity Asset Store). https://assetstore.unity.com/ packages/tools/animation/dynamic- bone-16743, 2024. 2

[19] Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In ICLR, 2014. 4, 13

[20] Chuan Koon Koh and Zhiyong Huang. A simple physics model to animate human hair modeled in 2D strips in real time. In Eurographics Workshop on Computer Animation and Simulation, 2001. 2

[21] Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. SALAD: Part-level latent diffusion for 3D shape generation and manipulation. In ICCV, 2023. 3

[22] Gyeongmin Lee, Wonjong Jang, and Seungyong Lee. Generating 3D hair strips from partial strands using diffusion model. In Pacific Graphics Conference Papers, Posters, and Demos, 2025. 2

[23] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 21

[24] Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. PartCrafter: Structured 3D mesh generation via compositional latent diffusion transformers. In NeurIPS, 2025. 3

[25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 2, 4

[26] Isabella Liu, Zhan Xu, Wang Yifan, Hao Tan, Zexiang Xu, Xiaolong Wang, Hao Su, and Zifan Shi. RigAnything: Template-free autoregressive rigging for diverse 3D assets. ACM TOG, 44(4), 2025. 3

[27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 2, 4

[28] Pengyu Long, Zijun Zhao, Min Ouyang, Qingcheng Zhao, Wei Yang, Lan Xu, and Jingyi Yu. Generating 3D hair strands from images with diverse styles and viewpoints. In ACM MM, 2025. 2

[29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 6, 13

[30] Zixuan Lu, Tongtong Wang, Yuefan Shen, Zhongtian Zheng, Chenfanfu Jiang, Yin Yang, and Kui Wu. HairCS: Recon structing strand-based hair from hair cards. arXiv preprint arXiv:2609.16465, 2026. 2

[31] Haimin Luo, Min Ouyang, Zijun Zhao, Suyi Jiang, Longwen Zhang, Qixuan Zhang, Wei Yang, Lan Xu, and Jingyi Yu. GaussianHair: Hair modeling and rendering with light-aware Gaussians. arXiv preprint arXiv:2402.10483, 2024. 2

[32] Haimin Luo, Min Ouyang, Lan Xu, and Jingyi Yu. HairGPT: Strand-as-language autoregressive modeling for realistic 3D hairstyle synthesis. ACM TOG, 45(4), 2026. 2

[33] Haimin Luo, Srinjay Sarkar, Albert Mosella-Montoro, Francisco Vicente Carrasco, and Fernando De la Torre. CGHair: Compact Gaussian hair reconstruction with card clustering. In CVPR, 2026. 2

[34] Xiaoyang Mao, Hiroki Kato, Atsumi Imamiya, and Ken Anjyo. Sketch interface based expressive hairstyle modelling and rendering. In Computer Graphics International, 2004. 3

[35] Giljoo Nam, Chenglei Wu, Min H. Kim, and Yaser Sheikh. Strand-accurate multi-view hair capture. In CVPR, 2019. 2

[36] Okazu (pafuhana1213). KawaiiPhysics: Simple bone physics for Unreal Engine 4 & 5. https://github. com/pafuhana1213/KawaiiPhysics, 2026. 2

[37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 4, 15

[38] Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. CharacterGen: Efficient 3D character generation from single images with multi-view pose canonicalization. ACM TOG, 43(4), 2024. 2

[39] pixiv. VRoid Hub. https://hub.vroid.com/. Accessed 2026-10-19. 3, 6

[40] Jiachen Qian, Hongye Yang, Youtian Lin, Tianhao Zhao, Feihu Zhang, Yao Yao, and Hengshuang Zhao. Anime-Ready: Controllable 3D anime character generation with body-aligned component-wise garment modeling. In ICLR, 2026. 3

[41] Radu Alexandru Rosu, Shunsuke Saito, Ziyan Wang, Chenglei Wu, Sven Behnke, and Giljoo Nam. Neural Strands: Learning hair geometry and appearance from multi-view images. In ECCV, 2022. 2

[42] Radu Alexandru Rosu, Keyu Wu, Yao Feng, Youyi Zheng, and Michael J. Black. DiffLocks: Generating 3D hair from a single image using diffusion models. In CVPR, 2025. 2, 7, 21

[43] Takeyuki Sakai and Vladimir Savchenko. Skeleton-based anime hair modeling and visualization. In International Conference on Cyberworlds, 2013. 3

[44] Yuefan Shen, Yican Dong, Xiufeng Huang, Zhongtian Zheng, Youyi Zheng, and Kui Wu. HairLRM: Strand-based hair modeling via large reconstruction models. In SIG-GRAPH, 2026. 2

[45] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. MeshGPT: Generating triangle meshes with decoder-only transformers. In CVPR, 2024. 3

[46] SideFX. Generating hair cards for games (Houdini 22.0 documentation). https://www.sidefx.com/docs/ houdini22.0/fur/haircards.html, 2026. 2

[47] Oriane Simeoni et al. DINOv3.´ TMLR, 2026. 4, 14

[48] Vanessa Sklyarova, Jenya Chelishev, Andreea Dogaru, Igor Medvedev, Victor Lempitsky, and Egor Zakharov. Neural Haircut: Prior-guided strand-based hair reconstruction. In ICCV, 2023. 2

[49] Vanessa Sklyarova, Egor Zakharov, Otmar Hilliges, Michael J. Black, and Justus Thies. Text-conditioned generative model of 3D strand-based human hairstyles. In CVPR, 2024. 2

[50] Vanessa Sklyarova, Egor Zakharov, Malte Prinzler, Giorgio Becherini, Michael J. Black, and Justus Thies. Im2Haircut: Single-view strand-based hair reconstruction for human avatars. In ICCV, 2025. 2

[51] Chaoyue Song, Xiu Li, Fan Yang, Zhongcong Xu, Jiacheng Wei, Fayao Liu, Jiashi Feng, Guosheng Lin, and Jianfeng Zhang. Puppeteer: Rig and animate your 3D models. In NeurIPS, 2025. 3

[52] Chaoyue Song, Jianfeng Zhang, Xiu Li, Fan Yang, Yiwen Chen, Zhongcong Xu, Jun Hao Liew, Xiaoyang Guo, Fayao Liu, Jiashi Feng, and Guosheng Lin. MagicArticulate: Make your 3D models articulation-ready. In CVPR, 2025. 3

[53] Masataka Sumi. MToon: Toon shader with Unity global illumination. https://github.com/Santarh/MToon, 2018. 14

[54] Mingze Sun, Junhao Chen, Junting Dong, Yurun Chen, Xinyu Jiang, Shiwei Mao, Puhua Jiang, Jingbo Wang, Bo Dai, and Ruqi Huang. DRiVE: Diffusion-based rigging empowers generation of versatile and expressive characters. In CVPR, 2025. 3

[55] Mingze Sun, Cheng Zeng, Jiansong Pei, Junhao Chen, Chaoyue Song, Shaohui Wang, Tianyuan Chang, Bin Huang, Zijiao Zeng, and Ruqi Huang. Animator-centric skeleton generation on objects with fine-grained details. In CVPR, 2026. 3

[56] Qi Sun, Kiyohiro Nakayama, Jing Nathan Yan, Qixing Huang, Alexander Rush, Leonidas Guibas, Gordon Wetzstein, Jing Liao, and Guandao Yang. MeshFlow: Mesh generation with equivariant flow matching. In SIGGRAPH, 2026. 3

[57] Yusuke Takimoto, Hikari Takehara, Hiroyuki Sato, Zihao Zhu, and Bo Zheng. Dr.Hair: Reconstructing scalp connected hair strands without pre-training via differentiable rendering of line segments. In CVPR, 2024. 2

[58] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In NeurIPS, 2020. 4, 13

[59] Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. DiffuScene: Denoising diffu sion models for generative indoor scene synthesis. In CVPR, 2024. 3

[60] Kenji Tojo, Liwen Hu, Nobuyuki Umetani, and Hao Li. Strands2Cards: Automatic generation of hair cards from strands. In SIGGRAPH Asia, 2025. 2

[61] VRChat Inc. PhysBones (VRChat creator documentation). https://creators.vrchat.com/commoncomponents/physbones, 2026. 2

[62] VRM Consortium. VRM specification, version 0.0. https: //github.com/vrm- c/vrm- specification/ tree/master/specification/0.0, 2018. 3, 6, 14

[63] VRM Consortium. UniVRM: VRM format implementation for Unity. https://github.com/vrm-c/UniVRM, 2026. 3, 6, 14

[64] Hongsheng Wang, Xinrui Zhou, and Feng Lin. NOVA-3D: Non-overlapped views for 3D anime character reconstruction. In ACM International Conference on Multimedia in Asia Workshops, 2024. 2

[65] Yating Wang, Kui Wu, Ran Yi, Zixuan Lu, Tongtong Wang, Zhongtian Zheng, Lizhuang Ma, and Yuefan Shen. Hair-Flow: Modeling 3D hair strands with multi-modal condi tioned diffusion. arXiv preprint arXiv:2610.17348, 2026. 2

[66] Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, Tong Zhang, Shenghua Gao, and C. L. Philip Chen. Scaling mesh generation via compressive tokeniza tion. In CVPR, 2025. 3, 7, 15

[67] Holger Winnemoller, Jan Eric Kyprianidis, and Sven C.¨ Olsen. XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization. Computers & Graphics, 36(6):740–753, 2012. 5, 14

[68] Chenfei Wu et al. Qwen-Image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 5, 14, 26

[69] Keyu Wu, Yifan Ye, Lingchen Yang, Hongbo Fu, Kun Zhou, and Youyi Zheng. NeuralHDHair: Automatic high-fidelity hair modeling from a single image using implicit neural rep resentations. In CVPR, 2022. 2

[70] Keyu Wu, Lingchen Yang, Zhiyi Kuang, Yao Feng, Xutao Han, Yuefan Shen, Hongbo Fu, Kun Zhou, and Youyi Zheng. MonoHair: High-fidelity hair modeling from a monocular video. In CVPR, 2024. 2

[71] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3D latents for scalable and versatile 3D generation. In CVPR, 2025. 1, 3

[72] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. Native and compact structured latents for 3D generation. In CVPR, 2026. 1, 2, 3, 7, 15, 18

[73] Rui Xu, Dafei Qin, Kaichun Qiao, Qiujie Dong, Huaijin Pi, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu, Wenping Wang, and Taku Komura. Strips as Tokens: Artist mesh generation with native UV segmentation. ACM TOG, 45(4), 2026. 3

[74] Xiang Xu, Joseph G. Lambourne, Pradeep Kumar Jayaraman, Zhengqing Wang, Karl D. D. Willis, and Yasutaka Furukawa. BrepGen: A B-rep generative diffusion model with structured latent geometry. ACM TOG, 43(4), 2024. 3

[75] Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. RigNet: Neural rigging for articulated characters. ACM TOG, 39(4), 2020. 3

[76] Kunio Yamamoto, Toma Shimosoyama, and Tsukasa Noma. NPR hair modeling with parametric clumps. In International Conference on Cyberworlds, 2017. 3

[77] Shuhui Yang et al. Hunyuan3D 2.1: From images to highfidelity 3D assets with production-ready PBR material. arXiv preprint arXiv:2506.15442, 2025. 1, 2, 7, 15

[78] Jingwen Ye, Yuze He, Yanning Zhou, Yiqin Zhu, Kaiwen Xiao, Yong-Jin Liu, Wei Yang, and Xiao Han. PrimitiveAnything: Human-crafted 3D primitive assembly generation with auto-regressive transformer. In SIGGRAPH, 2025. 3

[79] Chih-Kuo Yeh, Pradeep Kumar Jayaraman, Xiaopei Liu, Chi-Wing Fu, and Tong-Yee Lee. 2.5D cartoon hair modeling and manipulation. IEEE TVCG, 21(3):304–314, 2015. 3

[80] Egor Zakharov, Vanessa Sklyarova, Michael Black, Giljoo Nam, Justus Thies, and Otmar Hilliges. Human hair reconstruction with strand-aligned 3D Gaussians. In ECCV, 2024. 2

[81] Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3DShape2VecSet: A 3D shape representation for neural fields and generative diffusion models. ACM TOG, 42 (4), 2023. 3

[82] Jia-Peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. One model to rig them all: Diverse skeleton rigging with UniRig. ACM TOG, 44(4), 2025. 3

[83] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A controllable large-scale generative model for creating high-quality 3D assets. ACM TOG, 43(4), 2024. 1, 3

[84] Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. DeepMesh: Autoregressive artist-mesh creation with reinforcement learning. In ICCV, 2025. 3, 7, 15

[85] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3D shape generation based on shape-image-text aligned latent representation. In NeurIPS, 2023. 3, 8, 15

[86] Yujian Zheng, Zirong Jin, Moran Li, Haibin Huang, Chongyang Ma, Shuguang Cui, and Xiaoguang Han. HairStep: Transfer synthetic to real using strand and depth maps for single-view 3D hair modeling. In CVPR, 2023. 2

[87] Yujian Zheng, Yuda Qiu, Leyang Jin, Chongyang Ma, Haibin Huang, Di Zhang, Pengfei Wan, and Xiaoguang Han. Towards unified 3D hair reconstruction from single-view portraits. In SIGGRAPH Asia, 2024. 2

[88] Zhongtian Zheng, Tao Huang, Haozhe Su, Xueqi Ma, Yuefan Shen, Tongtong Wang, Yin Yang, Xifeng Gao, Zherong Pan, and Kui Wu. Auto hair card extraction for smooth hair with differentiable rendering. ACM TOG, 44(6), 2025. 2

[89] Jin Zhou, Yi Zhou, Hongliang Yang, Pengfei Xu, and Hui Huang. StrokeFusion: Vector sketch generation via joint stroke-UDF encoding and latent sequence diffusion. In AAAI, 2026. 3, 8, 15

[90] Yuxiao Zhou, Menglei Chai, Alessandro Pepe, Markus Gross, and Thabo Beeler. GroomGen: A high-quality generative hair model using hierarchical latent representations. ACM TOG, 42(6), 2023. 2

![](images/0243c2ac975c9948f1bd5b205246bd85e39a71002e794e11b2ddc80260bfd2ee.jpg)  
Figure 10. In-the-wild illustrations, hair-only references and test point clouds (blue), as labeled on the platforms, each to the left of its generated hairstyle. Hair colors are taken from the images for display; point-cloud results show one color per card.

## A. Implementation details

Card VAE. Cards are padded to 128 control points, with the padding masked in attention. The decoder embeds each arc-length fraction with Fourier features [58] of 6 frequencies. The cross-entropy on the number of control points is weighted by 0.1 and the KL term [19] by $1 0 ^ { - 6 }$ . We train it for 80K steps with batch size 1,024, drawing each batch from cards of similar length, with AdamW [29] (β = (0.9, 0.99), no weight decay) at learning rate $3 \times 1 0 ^ { - 4 }$ , a linear warmup over 500 steps, a cosine decay to 10% of the peak and gradient clipping at norm 1. It has 8.07M parameters.

Card tokens. Root positions and latents are standardized with their mean and standard deviation over the training set; root normals are unit vectors. Hairstyles with more than 192 cards (0.6% of the training set) are left out of training.

Set DiT. Timesteps are sampled as t = sigmoid(ξ) with $\xi \sim \mathcal { N } ( 0 , 1 )$ [9]. Both models use AdamW $( \beta \ =$ (0.9, 0.99), no weight decay) with gradient clipping at norm 1; the learning rate warms up linearly over 2,000 steps for the point-cloud model and 1,000 for the image model and then stays constant; we sample with an exponential moving average (decay 0.9995). With probability 0.1, learned null tokens replace the condition tokens. Classifier-free guidance [17] combines the two velocities as $\mathbf { v } _ { \alpha } + \gamma \left( \mathbf { v } _ { \mathbf { C } } - \mathbf { v } _ { \alpha } \right)$ with guidance scale γ = 2, and the Euler method takes 50 uniform steps from t = 1 to t = 0.

Point-cloud condition. We use the released Dora-VAE 1.1 encoder [4] and cache, for every training point cloud, its 512 output tokens of width 768, taken after the final layer norm and before the KL bottleneck.

Image condition. The crop is the square around the union of the hair and face masks, with an 8% margin on each side. During training we jitter it: its scale by a log-normal factor $( \sigma = 0 . 0 8 )$ , its center by $\mathcal { N } ( 0 , 0 . 0 4 ^ { 2 } )$ times the side length, and with probability 0.15 we cut off its bottom, keeping 45% to 80% of the hair height as in half-body images. DI-

NOv3 [47] runs at 448 pixels and returns one class token, four register tokens and 784 patch tokens; a layer norm, a linear projection to the model width and a learned positional embedding turn them into the condition. For in-the-wild images, SAM 3 [3] segments with the prompts “person”, “anime character”, “hair” and “face”; the union of the four masks, cleaned with a morphological closing and hole filling, is the matte that cuts the character out onto a white background, and the hair and face masks give the crop.

Head frame. Figure 5 illustrates the head frame of Sec. 3.3. $W = 0 . 1 2 7 6$ and $O \ = \ ( 0 , 0 . 4 7 9 1 , 0 . 0 2 5 8 )$ are medians over the training set in the dataset frame. The face width w of a character is the width of its face bounding box, and its anchor is the top center of that box (the center in x and z, the top in y). Over the training set, k ranges from 0.70 to 1.24 (5th to 95th percentile).

Training renders. Each training character is rendered from 8 views with Blender 4.2 [2] (EEVEE) at 768 pixels. For each view we draw a body pose (natural 70%, hands near the head 25%, T or A pose 5%, with jittered IK targets, randomly mirrored one-hand poses, random spine, neck and head rotations and finger curls), a hair state from the VRM 0.x spring-bone simulation [62] (at rest in 35% of the views, otherwise swinging or settling after a head turn, nod, body turn, jump or sway; in 30% of the views the hair gravity is set to 0.1 to 0.6), and a camera direction relative to the head (front and three-quarter views 80%, side 10%, back 10%; elevation drawn from $\mathcal { N } ( 5 ^ { \circ } , ( 1 2 ^ { \circ } ) ^ { 2 } )$ and truncated to $[ - 2 5 ^ { \circ } , 4 0 ^ { \circ } ] )$ . Lighting is a preset or a random sun, and 80% of the views draw MToon [53] outlines.

Appearance augmentation. The synthetic style augmentation draws from grayscale, halftone, XDoG [67] line drawings, full hue rotation, channel permutation, flat colors, painterly filters, thick ink lines, rim light, low resolution, blur, sharpening, noise and JPEG compression. 40% of the samples keep a light augmentation: brightness, contrast, saturation and hue jitter, blur and JPEG compression with probability 0.3 each, and occasional grayscale.

Restyled images. We restyle training renders with Qwen-Image-Edit-2511 [68] (20 steps) into gacha (mobile-game) art, thick paint and semi-realistic painting, and keep a restyled image only if its hair and character silhouettes (SAM 3) match the original (IoU at least 0.85 for the hair and 0.9 for the character). The pool holds 21,336 images, and a training sample is drawn from it with probability 0.33.

Spring-bone chains. For the point-cloud model, whose cards are in the dataset frame, we map the generated cards into the head frame with the test character’s k and b and re-encode them with the card VAE before running the chain DiT. Chains come from the VRM files of the training characters: 32,979 of the 36,440 training hairstyles have hair chains, and leaving out those with more than 48 chains or more than 192 cards leaves 32,599 hairstyles with 438,672 chains. The chain DiT has 6 attention heads. It is trained for 80K steps with batch size 128, AdamW at learning rate $2 \times 1 0 ^ { - 4 }$ after a linear warmup over 1,000 steps, condition dropout 0.1 and an exponential moving average with decay 0.999; all loss weights are 1 except 4 for the existence flag, and the chain slots are randomly permuted for every training sample. We tune the two skinning thresholds of Sec. 3.4, the assignment distance and the ramp length, on 200 random training characters with chains. For each character, its own chains with their own physics parameters drive 16 points per card under four head motions (a turn, a nod, a jump and a sideways sway), once with the artists’ skin weights, where each point takes the weights of the nearest vertex of the hair mesh, and once with our skinning. We choose the thresholds that maximize the motion correlation of Tab. 4: sweeping the assignment distance over 0.15, 0.25, 0.4 and 0.6 face widths and the ramp length over 0.01 to 0.3 face widths gives 0.4 and 0.05 (correlation 0.84), and the ramp length barely matters. On the 58 test hairstyles with chains, our skinning reaches a motion correlation of 0.77 with the artists’ skin weights. The simulation follows the reference spring-bone update of UniVRM [63]: at every frame, the tip of each bone keeps (1 − drag) of its previous displacement, moves toward its rest direction, rotated by the parent, by stiffness times the time step and along gravity by gravity times the time step, and is then pulled back to the bone length. We use no colliders and run at 60 fps. Generated chains use their predicted stiffness, drag 0.4 and no gravity.

Chain metrics. In Tab. 4, lengths are in face widths W, and each chain is represented by 8 points along its path. CD is the Chamfer distance between the path points of the generated and the artists’ chains. Recall and precision are the shares of artist and generated chains whose points lie, on average, within 0.1 W of the path of a chain in the other set. For motion, the artists’ cards are skinned to each set of chains as in Sec. 3.4, the artists’ chains keep their own physics parameters, and each rig is simulated under four head motions of 2.5 s each: a turn (a yaw of $5 0 ^ { \circ }$ in 0.35 s, then held), a nod (a pitch of $2 5 ^ { \circ }$ down and back in 0.5 s), a jump (0.15 m up and down in 0.6 s) and a sideways sway (6 cm at 1.5 Hz for two cycles). The displacement of each of 16 points per card is taken relative to the moving head, minus its displacement with the head at rest, so that sagging under gravity does not count. Motion is the correlation over card points between the mean displacement of each point and that under the artists’ chains, averaged over the four motions, and Ampl. is the mean displacement. Fit is the share of chain path points within 0.1 W of the card meshes.

Table 5. Cascades of image-to-3D generators (TRELLIS.2 [72] and Hunyuan3D-2.1 [77]) and point-cloud generators, against HairDeck. $\ddot { \cdots }  \ '$ feeds an image-to-3D output, sampled as a point cloud, to a point-cloud generator; PC model: our point-cloud model. Columns as in Tab. 2: aligned CD and IoU, failure rates (Sec. 4.1) and time per image.
<table><tr><td></td><td colspan="2">Test characters</td><td colspan="2">Failure rate↓</td><td></td></tr><tr><td>Method</td><td>CD↓</td><td>IoU↑</td><td>Test</td><td>In the wild Time↓</td><td></td></tr><tr><td>TRELLIS.2 → CHARM</td><td>0.0170</td><td>0.554</td><td>32.9%</td><td>80.0%</td><td>119s</td></tr><tr><td>Hunyuan3D-2.1 → CHARM</td><td>0.0163</td><td>0.573</td><td>28.9%</td><td>80.0%</td><td>96.8s</td></tr><tr><td>TRELLIS.2 → PC model</td><td>0.0138</td><td>0.621</td><td>22.4%</td><td>76.7%</td><td>85.8s</td></tr><tr><td>Hunyuan3D-2.1 → PC model</td><td>0.0143</td><td>0.615</td><td>29.4%</td><td>80.0%</td><td>64.2 s</td></tr><tr><td>HairDeck</td><td>0.0089</td><td>0.725</td><td>3.5%</td><td>10.0%</td><td>1.33 s</td></tr></table>

Export. Every card mesh carries UV coordinates: the first runs across the width, so the two faces of a card mirror each other and need no seam, and the second is the arc-length fraction, from the root (0) to the tip (1). To export to VRM, we replace the hair of a character model, its hair bones and its hair spring-bone groups with the generated cards, one spring-bone group per generated chain, and use the geometric skinning of Sec. 3.4 as skin weights. Results from in-the-wild images are exported the same way, on the body of a test character with the hair color taken from the image.

Compute. Each model trains on one NVIDIA RTX A6000 GPU (48 GB): the card VAE in 4 hours, the pointcloud model in about 10 hours, the chain DiT in about 3 hours and the image model, starting from the point-cloud model, in about 27 hours.

## B. Additional comparisons and ablations

Figure 11 shows eight more hairstyles generated from the point clouds of CHARM’s test set, and Figs. 12 and 13 compare the single-image methods on four test characters and nine in-the-wild illustrations. Figure 14 adds back views on two more point clouds: CHARM [15] leaves part of the back of the head bare on the first, and DeepMesh [84] and BPT [66] cover only parts of the back. Table 5 adds cascades of image-to-3D and point-cloud generators to Tab. 2, Tab. 6 lists all point-cloud ablations, and Tab. 7 the ablations of the image model. For the head-frame ablation of Sec. 4.4, we also render every test character from the side, with the pose, hair motion and lighting of its front view and the camera turned by 90° around the head.

## C. Card representation and orientation

Figure 15 compares the card tokens of Tab. 6.

Table 6. All ablations on point clouds; each row changes one part of HairDeck (rows of Tab. 3: Sec. 4.4; card tokens: Sec. C). CD and IoU on CHARM’s ground truth and Orient. on the original meshes, as in Tab. 1; turn: the largest angle between consecutive segments of a card, median per hairstyle (artist cards: 14.1°). Root position inside the latent: the card VAE encodes absolute positions, and the token has no separate root position. Bold: best in each column with an arrow.
<table><tr><td>Variant</td><td>CD↓</td><td>IoU↑</td><td>Orient.↓</td><td>Turn</td></tr><tr><td>HairDeck</td><td>0.0066</td><td>0.867</td><td>16.8°</td><td>14.2°</td></tr><tr><td>Card token</td><td></td><td></td><td></td><td></td></tr><tr><td>raw control points</td><td>0.0068</td><td>0.856</td><td>17.1°</td><td>21.9°</td></tr><tr><td>PCA, 64 coefficients</td><td>0.0078</td><td>0.832</td><td>17.8°</td><td>11.1°</td></tr><tr><td>VAE with index decoding</td><td>0.0079</td><td>0.823</td><td>18.1°</td><td>25.3°</td></tr><tr><td>latent z  $\in \mathbb { R } ^ { 6 4 }$ </td><td>0.0071</td><td>0.843</td><td>17.2°</td><td>13.4°</td></tr><tr><td>Token composition</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o root normal</td><td>0.0070</td><td>0.850</td><td>26.4°</td><td>13.9°</td></tr><tr><td>w/o existence (card-count head)</td><td>0.0068</td><td>0.859</td><td>16.7°</td><td>14.5°</td></tr><tr><td>root position inside the latent</td><td>0.0067</td><td>0.859</td><td>17.2°</td><td>14.6°</td></tr><tr><td>Data, encoder and size</td><td></td><td></td><td></td><td></td></tr><tr><td>CHARM&#x27;s card templates</td><td>0.0068</td><td>0.856</td><td>26.7°</td><td>15.8°</td></tr><tr><td>CHARM&#x27;s encoder [85]</td><td>0.0068</td><td>0.853</td><td>17.2°</td><td>14.0°</td></tr><tr><td>small model (54.6M)</td><td>0.0072</td><td>0.847</td><td>17.5°</td><td>14.2°</td></tr></table>

Card VAE design. Table 8 starts from a VAE that decodes the control points by index, as StrokeFusion [89] decodes the points of a stroke. It takes the same inputs as our card VAE; its encoder sees the control points without their arc-length fractions, and its decoder adds z to 128 learned slot embeddings and predicts in each slot a control point and whether the slot holds one, with a squared error on standardized features. Decoding at arc-length fractions instead cuts the reconstruction error by a factor of 4.3, and adaLN modulation [37] and the world-unit loss of Sec. 3.1, in place of a squared error on standardized features, together lower it by another 29%. The raw token resamples a card to 64 points by arc length and holds their positions relative to the root position, widths and thicknesses (317 values) with the number of control points; the PCA token holds the first 64 principal components of these 317 values, fitted on 300,000 training cards and scaled to unit variance.

Width direction of the data. Our cards keep the control point positions of CHARM’s templates; they differ in the width direction and in the widths and thicknesses, which we measure along the Bishop frame [1]. Table 9 rebuilds the artist cards with each choice of control points, root direction and propagation, and measures both the width direction and the shape. CHARM’s templates take the root direction from the first three control points and solve the rest of the card by least squares, which keeps each width direction perpendicular to the card and close to its neighbors. On the contro points quantized into CHARM’s tokens (512 levels per position axis), least squares rebuilds the shape more closely than a Bishop frame from the same start, as CHARM found for Bishop propagation from estimated normals [15], but both leave the width direction far off $( 4 0 . 6 ^ { \circ }$ and 35.3°), because the quantization turns the root direction from the first three points by $2 3 . 6 ^ { \circ }$ (median). Without the quantization, the Bishop frame is closer in both direction and shape, and with the root direction of the original mesh it is closer still, on artist and quantized points alike $( 7 . 5 ^ { \circ }$ and $8 . 2 ^ { \circ }$ , against $1 9 . 3 ^ { \circ }$ and $1 9 . 6 ^ { \circ }$ for least squares): given the right root direction, the quantization costs less than $1 ^ { \circ } .$ . Our cards take this route, with the root normal of the original mesh propagated by a Bishop frame; the card VAE adds $0 . 8 ^ { \circ }$ , and keeping the root direction constant along the card gives 11.9°. Our data cards are also closer to the original card meshes, with a two-sided point-to-surface distance of $0 . 9 4 5 \times 1 0 ^ { - 3 }$ against $1 . 3 8 8 \times 1 0 ^ { - 3 }$ for CHARM’s templates.

Table 7. Ablations of the image model on the 76 test characters. Front view: CD before alignment (CHARM’s protocol, so it includes where the hair sits), overlap (IoU) of the reprojected hair with the input’s hair, and aligned CD as in Tab. 2. Restyled: the front views restyled as in Fig. 20, with aligned CD, root precision P and recall R as in Sec. 4.1, and cards per hairstyle (ground truth: 65.1). Style aug.: the synthetic style augmentation of Sec. A; without it, every sample gets only the light augmentation. Bold: best in each column with an arrow.
<table><tr><td></td><td colspan="3">Front view</td><td colspan="4">Restyled</td></tr><tr><td>Variant</td><td>CD↓</td><td></td><td>Overlap↑ Aligned CD↓</td><td>Aligned CD↓</td><td>P↑</td><td>R</td><td>Cards</td></tr><tr><td>HairDeck</td><td>0.0136</td><td>0.829</td><td>0.0089</td><td>0.0096</td><td>0.849</td><td>0.833</td><td>68.9</td></tr><tr><td>w/o head frame</td><td>0.0189</td><td>0.798</td><td>0.0089</td><td>0.0095</td><td>0.736</td><td>0.756</td><td>69.8</td></tr><tr><td>w/o restyled images</td><td>0.0135</td><td>0.827</td><td>0.0090</td><td>0.0097</td><td>0.781</td><td>0.826</td><td>86.9</td></tr><tr><td>w/o restyled images and style aug.</td><td>0.0137</td><td>0.827</td><td>0.0091</td><td>0.0100</td><td>0.760</td><td>0.790</td><td>79.9</td></tr></table>

Table 8. Card VAE design: reconstruction of the cards of CHARM’s 100 test hairstyles, with $\textbf { z } \in \mathbb { R } ^ { 3 2 }$ . Per card, CD is the two-sided point-to-surface distance between the reconstructed and the input card, both meshed with the input card’s root normal, in $1 0 ^ { \hat { - 3 } }$ of the character height, and F1 is the F-score of their surface samples at distance 0.001; exact n: share of cards with the right number of control points; turn as in Tab. 6 (artist cards: $1 4 . 1 ^ { \circ } )$ . +: added to arc-length decoding, where z is added to every decoder query; uniform targets: training queries the decoder at n uniformly spaced arc-length fractions, as at test time. Bold: best in each column with an arrow.
<table><tr><td>Card code</td><td> $\mathrm { C D } ( 1 0 ^ { - 3 } ) \downarrow$ </td><td>F1↑</td><td>Exact n↑ Turn</td><td></td></tr><tr><td>index decoding</td><td>4.17</td><td>0.411</td><td>0.917</td><td> $2 5 . 3 ^ { \circ }$ </td></tr><tr><td>arc-length decoding</td><td>0.98</td><td>0.893</td><td>0.984</td><td> $1 5 . 6 ^ { \circ }$ </td></tr><tr><td>+ adaLN</td><td>0.91</td><td>0.904</td><td>0.984</td><td> $1 5 . 8 ^ { \circ }$ </td></tr><tr><td>+ world-unit loss</td><td>0.84</td><td>0.920</td><td>0.984</td><td> $1 3 . 1 ^ { \circ }$ </td></tr><tr><td>+ both</td><td>0.70</td><td>0.942</td><td>0.984</td><td> $1 3 . 7 ^ { \circ }$ </td></tr><tr><td>+ both + uniform targets (ours)</td><td>0.70</td><td>0.943</td><td>0.984</td><td> $1 4 . 3 ^ { \circ }$ </td></tr></table>

Width direction of the outputs. Table 10 shows the same on generated cards. On CHARM’s outputs, a Bishop frame from the first three points lowers the orientation error only from $4 0 . 0 ^ { \circ }$ to $3 7 . 5 ^ { \circ }$ and raises the CD slightly (0.0117 to

Table 9. Width direction and shape of artist cards rebuilt from their control points, on CHARM’s 100 test hairstyles: control points as placed by the artists or quantized by CHARM, the root direction from the first 3 points (their smallest principal axis, as in CHARM’s templates) or from the original mesh, and CHARM’s least squares, a Bishop frame or a constant direction along the card. Mean: orientation error as in Tab. 1; median: the same with the median over each hairstyle’s cards; P2S: two-sided point-tosurface distance to the original card mesh, with CHARM’s widths and thicknesses in every row so that only the width direction differs (card VAE: its own). Bold: best above the line.

<table><tr><td>Root direction Along the card Mean↓ Median↓</td><td></td><td></td><td></td><td> $\mathrm { P } 2 \mathrm { S } ( 1 0 ^ { - 3 } ) \downarrow$ </td></tr><tr><td colspan="5">Control points placed by the artists</td></tr><tr><td>first 3 points</td><td>least squares</td><td> $2 4 . 7 ^ { \circ }$ </td><td> $1 9 . 7 ^ { \circ }$ </td><td>1.39</td></tr><tr><td>first 3 points</td><td>Bishop</td><td>16.3°</td><td> $9 . 4 ^ { \circ }$ </td><td>1.24</td></tr><tr><td>original mesh</td><td>least squares</td><td>19.3°</td><td> $1 6 . 9 ^ { \circ }$ </td><td>1.36</td></tr><tr><td>original mesh</td><td>constant</td><td>11.9°</td><td> $9 . 7 ^ { \circ }$ </td><td>1.28</td></tr><tr><td>original mesh</td><td>Bishop</td><td>7.5°</td><td>5.6°</td><td>1.10</td></tr><tr><td colspan="5">Control points quantized by CHARM</td></tr><tr><td>first 3 points</td><td>least squares</td><td>40.6°</td><td>37.4°</td><td>2.20</td></tr><tr><td>first 3 points</td><td>Bishop</td><td>35.3°</td><td>30.7°</td><td>2.39</td></tr><tr><td>original mesh</td><td>least squares</td><td>19.6°</td><td>17.2°</td><td>1.43</td></tr><tr><td>original mesh</td><td>Bishop</td><td> $8 . 2 ^ { \circ }$ </td><td> $6 . 3 ^ { \circ }$ </td><td>1.19</td></tr><tr><td colspan="5">Card VAE reconstruction</td></tr><tr><td>original mesh Bishop</td><td></td><td>8.3°</td><td> $6 . 5 ^ { \circ }$ </td><td>1.29</td></tr></table>

Table 10. Width direction of generated cards on CHARM’s 100 test point clouds. Only the width direction used for meshing changes. Orient. and CD: as in Tab. 1; root: angle at the root, median over the cards of each hairstyle.
<table><tr><td>Method</td><td>Width direction</td><td>Orient.↓</td><td>Root↓</td><td>CD↓</td></tr><tr><td>CHARM</td><td>first 3 points + least squares</td><td> $4 0 . 0 ^ { \circ }$ </td><td>33.4°</td><td>0.0117</td></tr><tr><td>CHARM</td><td>first 3 points + Bishop</td><td> $3 7 . 5 ^ { \circ }$ </td><td>33.1°</td><td>0.0119</td></tr><tr><td>HairDeck</td><td>first 3 points + least squares</td><td> $3 0 . 2 ^ { \circ }$ </td><td>19.4°</td><td>0.0067</td></tr><tr><td>HairDeck</td><td>root normal + Bishop</td><td>16.8°</td><td>8.9°</td><td>0.0066</td></tr></table>

0.0119), because the root direction stays $3 3 ^ { \circ }$ off. On our outputs, the generated root normal with a Bishop frame is better in both than least squares from the first three points $( 1 6 . 8 ^ { \circ }$ against 30.2°, CD 0.0066 against 0.0067).

![](images/bf7d5dedfa070f5e5fa0d78ba34eb298d5b64d9d52c1879afecf2f0b1238df05.jpg)  
Figure 11. More point clouds from CHARM’s test set, shown as in Fig. 4.

![](images/b70805e28d670f5df5528da41db41d8f1d54076493aad96d20d7f8d092ab4c1b.jpg)  
Figure 12. Single images, shown as in Fig. 6 with a ground-truth column: four test characters (top) and three in-the-wild illustrations (bottom). TRELLIS.2 [72] outputs a flat plate facing the camera for the last one.

![](images/39e300f47a2fc4fecee96e3e920f763ea7299104fe829b48500b27eef0b3d743.jpg)  
Figure 13. Six more in-the-wild illustrations, shown as in Fig. 6. TRELLIS.2 outputs a flat plate facing the camera for five of them.

![](images/38a2e747f6f5c097e7ab720670d2914ef9539f29ca82cdde633451cf3f117d67.jpg)  
Figure 14. Two more point clouds from CHARM’s test set, shown as in Fig. 4, each with a row of back views below it.

![](images/298daf2e0e006e6acb6cced22063c75e7833a016b8446e3f7fc3a727b4a56018.jpg)  
Figure 15. Card tokens. A set DiT of the same architecture, trained with each card token of Tab. 6, generates two test hairstyles from point clouds; insets magnify the same window in each row. Raw control points and the VAE with index decoding give jagged cards, PCA with 64 coefficients gives straighter cards than the artists’, and our latent gives cards as smooth as theirs. Each card has its own color.

![](images/bb6151d7be5230e26f070e79cad200639f3c6aa24ddd48ec2e0714d82b2eb617.jpg)  
raw control points

![](images/85f040120a1d6821e901546fcbcaf505e226194938ff97585f4e5eafba899ba6.jpg)  
PCA, 64 coefficients

![](images/5ebc4cc2f367e0ed248a0835a5954ca4c6720ab9ed772b0e0ed8084681420736.jpg)  
VAE with index decoding

![](images/4151348c21cc1eb21012d2af1c079d0560162ea8c5879dff3a831857048c91b8.jpg)  
Artist cards

## D. Baseline details

Alignment. For the aligned CD and IoU of Tabs. 2 and 5, every output is normalized to the bounding box of the ground truth and then aligned to it by ICP with scale, started from 24 axis-aligned rotations.

CHARM. We run the authors’ checkpoints with greedy decoding: the point-cloud model on the point clouds of CHARM’s test set, and the image model on a line drawing of the hair made with CHARM’s pipeline.

Card-level autoregression. The card-level autoregressive model of Tab. 1, with 182.1M parameters, uses our card tokens, card VAE, data, condition and training steps. A 17-layer transformer of width 768 generates the cards in CHARM’s order, by the azimuth of each card’s mean point around the vertical axis, and a 6-layer per-token diffusion head, as in MAR [23], samples each token with 50 steps and guidance scale 2; before each card, a stop head decides whether the sequence ends.

Retrieval. Retrieval in Tab. 1 compares the mean of the input’s condition tokens with those of the 36,440 training hairstyles by cosine similarity and returns the cards of the closest. For the chains in Tab. 4, it returns the chains of the closest training hairstyle that has chains.

Procedural rig. The procedural rig of Tab. 4 places chains on cards by rules, with parameters from the spring bones of 300 training characters and lengths in face widths W. Cards whose centerlines lie close (mean distance below 0.2 W) and point the same way form a unit, which gets one chain per 0.23 W of its width. A chain follows the mean centerline of its cards, fitted by a quadratic Bezier curve,´ and starts 15% of the card length from the root. Longer chains get more joints and a lower stiffness, as in the training characters; all chains use drag 0.4 and no gravity.

Mesh generators. MeshAnything V2 [6], DeepMesh and BPT generate a mesh from a point cloud of the ground-truth hair mesh, with their official checkpoints and settings. MeshAnything V2 and BPT sample it from the marching-cubes surface of the hair, and DeepMesh from the mesh itself. The official sampler of DeepMesh mirrors its output, which we undo. We sample 10K points from depth renders of each output over 100 views, as CHARM scans its data, and evaluate them with CHARM’s own scripts.

Image-to-3D generators and cascades. TRELLIS.2 and Hunyuan3D-2.1 generate geometry only, from the image with all but the hair masked out. For the cascades of Tab. 5, the mesh is placed in the median hair bounding box of the training set, scanned into 10K points as above and passed to CHARM or our point-cloud model.

Table 11. Drawing styles on the 76 test characters: the front view (original) and its restyles of Fig. 20. Aligned CD as in Tab. 2 and root precision as in Sec. 4.1. w/o restyle: without restyled images; w/o both: also without synthetic style augmentation (Tab. 7). Bold: best model per style.
<table><tr><td rowspan="2"></td><td colspan="3">Aligned CD↓</td><td colspan="3">Root precision↑</td></tr><tr><td>HairDeck restyle</td><td>w/o</td><td>w/o both</td><td>HairDeck restyle</td><td>w/o</td><td>w/o both</td></tr><tr><td>Style original</td><td>0.0089</td><td>0.0090</td><td>0.0091</td><td>0.904</td><td>0.895</td><td>0.898</td></tr><tr><td>gacha</td><td>0.0092</td><td>0.0092</td><td>0.0093</td><td>0.859</td><td>0.830</td><td>0.821</td></tr><tr><td>thick paint</td><td>0.0098</td><td>0.0098</td><td>0.0103</td><td>0.818</td><td>0.700</td><td>0.676</td></tr><tr><td>semi-real</td><td>0.0096</td><td>0.0100 0.0102</td><td></td><td>0.869</td><td>0.812</td><td>0.784</td></tr></table>

DiffLocks. We run the official checkpoints of DiffLocks [42]. Its pipeline crops the image around a face box, which we take from our face mask on every input. Each output holds about 95K strands, which we convert into K = 60 cards: every strand is resampled to 16 points by arc length, k-means groups the strands into K clusters, and each cluster becomes one card whose centerline is the mean of its strands and whose width, thickness and width direction come from the spread of the strands around it. Strands laid inside the ground-truth cards and converted back reach an aligned CD of 0.0050, and K = 40, 100 and 150 give 0.0172, 0.0174 and 0.0172, against 0.0174 for K = 60.

## E. Additional results

More results. Figures 16 to 19 show 112 more hairstyles generated from in-the-wild images, and Fig. 10 places results from illustrations, hair-only references (no character) and point clouds next to their inputs.

Drawing styles. Figure 20 shows two test characters restyled into gacha art, thick paint and semi-realistic painting, with the hairstyle generated from each image. Table 11 evaluates every style on all 76 test characters for the three models of the style ablations in Tab. 7.

Denoising. Figure 21 follows the set DiT through denoising from a point cloud, and Fig. 22 from an illustration. The clean set predicted at the first steps holds only a few cards, and cards appear as their existence flags settle.

## F. Failure cases

Figure 23 shows failure cases on in-the-wild illustrations, where objects inside the matte (Sec. A) turn into cards.

![](images/2db8c1a9bd9b2a8e8f1359c6b6096f9baec31d6a8274e572fcdb7e2daf4f4531.jpg)  
Figure 16. More hairstyles generated from in-the-wild images. Each cell shows the input (top left), the back view (bottom left) and the front view of the generated hairstyle. Hair colors are taken from the images for display.

![](images/6c7b1d6e7be99e3dc5b4db43c1706c2ae9fb58fecb6bc274ea32708daf600831.jpg)  
Figure 17. More hairstyles generated from in-the-wild images, shown as in Fig. 16.

![](images/a3f1b757621b951c17ee74cc4cec4dfaefe9ca8f0f9fda5f9907ea1e3cebdaac.jpg)  
Figure 18. More hairstyles generated from in-the-wild images, shown as in Fig. 16.

![](images/62592e2943e339efcd4407c41dcf01ae4e8c444920d7ed1e127e7b6a3ec46c1a.jpg)  
Figure 19. More hairstyles generated from in-the-wild images, shown as in Fig. 16.

![](images/75082004c29f27f9e00930ce82506f8b19bb9b07e0a9687a325fa4d6c3b9bc05.jpg)

![](images/b3108f1607fb056f9813cb0e5c06b6e6bb743b87518c474cbe850eb2e5008455.jpg)  
original

![](images/c38c6faca6582abfbf020c0cadf2930cb5deb6ba65c67bb2b8ffd107582cbe7a.jpg)  
gacha

![](images/5f07cfc8d6a6fc0f62f8f261bb7adda5f6bd00891bf0f3ee6a2a852e31597e68.jpg)  
thick paint

![](images/3dbbb2e9e1ab8cb0a30df7c3b3e85275b43f15b4b83312d652cdf40f552f6224.jpg)  
semi-real

![](images/eb437abfbf16e88b730bddf7f87b4e49e69571c59ccbdaca73a27e95f909a354.jpg)  
original

![](images/7ae35442e307d9714f3953b074f58eb4c98524c65e9a1231fe9309058712dfaf.jpg)  
gacha

![](images/07eecc35ce59cfd8922ae185d65a1c3b0bb6fb7d4758cd30289af8c0b7469b24.jpg)  
thick paint

![](images/20543eb5a14aec86726482e976d2412b123382cc5cf2a21042892cf845658ba6.jpg)  
semi-real

Figure 20. Drawing styles. Two test characters, original and restyled by an image-editing model [68] into the gacha, thick paint and semi-real styles (top row of each), and the hairstyle generated from each image (bottom row). Each character’s hair is shown in one color.

![](images/fa99c8615405b277779195ed1152f7138f003711b8a9d81f048b90938688f10e.jpg)

![](images/37fd3b22544d69d9c6f2f60a009c794beb34636184fb4ce7bfeb21a4348acba9.jpg)

![](images/6f17201c55824183dfbdfc091ef15445871107197f8d11c6121e2962f6526e9a.jpg)

![](images/c5046cb16356d1fcd2bf66ce634f828cfb7d021b4422b3b3391e35c362ab707a.jpg)

![](images/847ac1a1a30caa04b6d63e9b4549ad0bfa16b5b5295b7119dea937ac20fc7b6e.jpg)

![](images/55cc4648e40a37d0ac66ff7455ffd13e55a9ebbc745bf9b2ef409d73d64f55b3.jpg)

![](images/b99a2412641e5062e88506bfc3cbc3f53f38c55ff2087fee20e90714d58666d3.jpg)

![](images/80603256f215bbf564dd15bc50a51f95def2f878808197becbb347fc75b9c61a.jpg)

![](images/3f83a24c175e4863494af000ff1aba489b9c49eef1b2ad2277124b339090b719.jpg)

![](images/a8fe58aaa23b761ccd4884e4c3431508f9149be7a4c3f1f3cc791a03dd94cb0d.jpg)

![](images/c57500cc5d988ab785c510a5ea0e68481ab8806011c49730d7cc842e9d24b7b7.jpg)

![](images/5c2405df242e330804b47d7460a527b9cd577454cba87e15e8cc30cd57e55b4b.jpg)

![](images/175af5a0ec3d66c39efaad35ee84e99d5454f8188e9d37984a3fe6d385c3e754.jpg)

![](images/9db0e686a142c056cde139c4dd127e3f026793a6ad031af8141a00f8f5aba3e9.jpg)

![](images/5288ba0aca21630df477752f55e5e6abcc23cbf34376da518e773360127c4886.jpg)

![](images/2b30ef81622ad3ee5e5c27d53e753b3646563aded13c2ab21d58fe69b832c10a.jpg)

![](images/4260a0bd4ff5f5afab8fcd2c027b7ec44eb7fe57852a96e3bc8124b0f229a38f.jpg)

![](images/745d68abad4fa2b9399557c1cae263e44c106d00e366d9ba42dacccccfbb9048.jpg)

![](images/9a803c4e00733c0267ae962d1dc43652ee9867b29902dad7223af376d7315230.jpg)

![](images/3a9ef8db1e9b932cdae64c7a7f8376aa4d21a15237bd6c734a878f17b2816ae7.jpg)

![](images/9c2ad781ed592ff1b0f105b5c4181d5bb8c8d87e05e647c30f0053b9d02900d4.jpg)

![](images/99c41d8ef059850709df19a57026ae7ada4414a77a21de64a90622231b72170b.jpg)  
input

![](images/6eb21ae39d9ff9ede85f463ef5e1465dbe00c14c34df2b2ad1100e758c686f39.jpg)  
t = 1.0

![](images/18934573da4a581c0a1ce0235fd653e0bb074308cc59830346f35ac1139acf14.jpg)  
t = 0.8

![](images/7b055c3186c1a607885e4b4d9f2d0330994c68e5fd11ea1dea7eecaab2d36037.jpg)  
t = 0.6

![](images/b3439033c5dfa338a54c344515a5bd5df1867e0e3dfdfe1833de9e1da5a9e99b.jpg)  
t = 0.4

![](images/ad3912e57f19588919bc8669e2f657564b64394e54d8084eef8e6f2ad43b7b56.jpg)  
t = 0.2

![](images/bbfc5922e47529041ebf9bf4128ffd9bcb5d5d80fe402aeca69fc43a7ca43562.jpg)  
t = 0.0

Figure 21. Denoising from a point cloud. For each test point cloud (left), top: the cards decoded from the noisy token set X ; bottom: the clean set predicted at that step. Each card keeps its color along the trajectory.

![](images/4a93ad12f9684f660f1967d38b799f1d910b5be454ac0f73d3283e32e863567a.jpg)  
Figure 22. Denoising from an illustration. On each platform, from left to right: the input, the cards decoded from the noisy token set X<sub>t</sub> at t = 1.0, 0.7 and 0.4, and the generated hairstyle. Each card keeps its color along the trajectory.

![](images/3e49abfccd3748b8a022d7bbcadc7eeab0f43bfb8e592223114fe70079a43857.jpg)  
props kept  
background kept  
tails kept  
effects kept  
hat kept  
horns kept

Figure 23. Failure cases on in-the-wild illustrations: the input (top), the matted input (middle) and our output (bottom). The matte keeps props, background, tails, effects, a hat or horns, and they turn into cards. Hair colors are taken from the images for display