# MCUFlowNet: Hardware-Aware Neural Architecture Search for Optical Flow on Microcontrollers

Enmin Lin, Jona Beysens

Department of Electrical Engineering (ESAT)

KU Leuven, Leuven, Belgium

{enmin.lin, jona.beysens}@kuleuven.be

Abstract—Dense optical flow allows small drones to avoid obstacles. However, few dense optical flow networks are designed for microcontrollers (MCUs) with neural accelerators, whose onchip SRAM limits the input resolution. We present PACE-NAS, a macro-to-micro neural architecture search framework that decouples memory from latency: a macro design fixes the backbone and its peak SRAM, and a micro design searches operators and depths to trade accuracy against compiler-estimated latency. We apply it to EdgeFlowNet on a Grove Vision AI V2 board (Ethos-U55), where profiling locates the peak SRAM at the fullresolution output head. Replacing transposed convolutions with bilinear upsampling and convolutions lowers the peak SRAM by 18%, and two lightweight context modules lower the endpoint error (EPE) without raising the peak. All searched subnets share this peak, so the search needs no memory constraint. We name the Pareto-optimal models MCUFlowNet; the fastest variant, MCUFlowNet-S, runs 1.76× faster than a retrained EdgeFlowNet at a 1.6% higher EPE and, in simulated forest flights at 2 m/s, reaches the goal without collision in 51.5% of trials, against 41.8%. PACE-NAS also applies to semantic segmentation, where a searched PIDNet-S variant runs 1.17× faster at similar accuracy. Code is available at https://gitlab.kuleuven. be/emedia-research-lab/esat-stadius/mcuflownet/mcuflownet.

## I. INTRODUCTION

Small aerial robots rely on timely dense optical flow, the per-pixel displacement between consecutive frames, for collision avoidance [1], and on optical flow for autonomous landing and self-motion estimation [2]. Perception delay shortens the time available to avoid obstacles [3], which motivates onboard processing [4], and strict weight and power budgets favor microcontrollers (MCUs) [5].

EdgeFlowNet, a compact multi-scale network of standard convolutions, brings dense optical flow to the Coral Edge TPU [6], but on MCUs its full-resolution output head sets the peak SRAM and thus bounds the input resolution. Hardwareaware neural architecture search (NAS) adapts networks to a target device [7]; on MCUs, it treats peak SRAM as a constraint and can be co-designed with the memory scheduling of the inference engine [8], [9]. For optical flow, however, NAS has typically targeted GPU-scale models [10], and few dense flow networks are designed for MCUs with neural accelerators.

This paper proposes PACE-NAS, a macro-to-micro framework for hardware-aware NAS that adapts existing dense flow networks to MCUs by decoupling memory from latency (Fig. 1). Its macro design profiles the target platform, restructures the layers that set the peak SRAM, and adds two lightweight context modules for image-wide context where SRAM is spare; these steps fix the backbone and its peak SRAM. Its micro design trains a weight-sharing supernet over operators and depths that leave this peak unchanged, then runs an evolutionary Pareto search over endpoint error (EPE) and the latency estimated by Arm’s Vela compiler [11]. To the best of our knowledge, this is the first work to use hardware-aware NAS to design dense optical flow networks for MCUs.

![](images/c7c1f45cfc1eac1bd05e686d307aaf345e081d714ab2f3e8155ee1a3ea2cd781.jpg)  
Fig. 1. PACE-NAS overview. Training cost is in estimated H200 GPU-hours, relative to one full training run of EdgeFlowNet (1×); S and L mark the two retrained subnets.

On Grove Vision AI V2, a board with an Arm Cortex-M55 MCU and an Ethos-U55 accelerator, we apply PACE-NAS to EdgeFlowNet and retrain two Pareto-optimal subnets, which we call MCUFlowNet-S and MCUFlowNet-L. Both our models need less peak SRAM than a retrained EdgeFlowNet; MCUFlowNet-S is 1.76× faster at a 1.6% higher EPE, while MCUFlowNet-L matches EdgeFlowNet’s latency at a similar EPE. Mainly because of its lower latency, MCUFlowNet-S also raises the success rate in simulated obstacle avoidance from 41.8% to 51.5%. On semantic segmentation, PACE-NAS finds a faster PIDNet-S at similar accuracy.

## II. RELATED WORK

Accurate optical flow networks typically run on GPUs: RAFT refines flow recurrently from all-pairs correlations [12], and NeuFlow v2 reaches real-time operation on embedded GPUs with coarse global matching and lightweight recurrent updates [13]. Lighter pyramid networks estimate flow from coarse to fine: SPyNet warps one frame toward the other at each level and predicts a residual flow [14], and Fast-FlowNet adds a compact cost volume, which stores matching costs between features of the two frames [15]. On MCUs, classical block matching has been accelerated with multicore parallelism [16], and NanoFlowNet runs on GAP8 with depthwise separable convolutions [1]. EdgeFlowNet instead uses an encoder–decoder of standard convolutions, without cost volumes or warping, and accumulates flow increments predicted at three scales [6]; all its operators run on the Ethos-U55, so we build on EdgeFlowNet.

![](images/b5f9ddc2d4ce3699942612358a6d6e4c27c373de6d801c5b071ddf4e3c69d1ea.jpg)  
Fig. 2. MCUFlowNet backbone and configuration space. ×1, ×2, or ×3 gives the number of residual blocks, and 3 × 3-dil combines a stride-2 3 × 3 convolution with a dilation-2 3 × 3 convolution. Insets (a) and (b) detail ECA and the global broadcast gate.

NAS has not yet been applied to dense optical flow on MCUs: FlowNAS searches optical flow encoders for GPUscale models under a parameter budget [10], and multiobjective zero-shot NAS returns a Pareto set for image classification [17]. NAS on MCUs has targeted classification and detection, where peak SRAM also depends on how the inference engine schedules layers in memory. MCUNet first fits its search space to the memory budget, then searches within it and co-designs the inference engine [8]; MCUNetV2 jointly searches networks and patch schedules to lower peak SRAM, at the cost of recomputing overlapping features [9]. Like MCUNet, PACE-NAS designs before it searches, but it first lowers peak SRAM by restructuring the backbone and then searches only choices that leave this peak unchanged, so the search needs no memory constraint.

## III. PACE-NAS: A MACRO-TO-MICRO DESIGN FRAMEWORK

Fig. 1 outlines the four steps of PACE-NAS: platform profiling (P) and architectural restructuring (A) form the macro design, which fixes the backbone and its peak SRAM; supernet training over a configuration space (C) and evolutionary Pareto search (E) form the micro design. Fig. 2 shows the resulting backbone for optical flow; the search changes only its 11 searchable boxes.

## A. Platform Profiling and Bottleneck Diagnostics (P)

On Grove Vision AI V2, firmware and frame buffers leave 1434 KiB of SRAM available for the tensor arena holding intermediate tensors. Fig. 3 presents per-stage Vela profiles of

EdgeFlowNet in 8-bit integer (INT8) precision at 160 × 208, before and after restructuring: before, the peak SRAM reaches 1430 KiB at the full-resolution output head (top), only 4 KiB below the available budget. This peak arises because the head must hold its upsampled input and its output simultaneously. Together, this head and the transposed convolution feeding it also account for about 40% of the estimated cycles, so we target the transposed convolutions for restructuring.

## B. Architectural Restructuring (A)

Vela 4.5.0 can cascade compatible layers (e.g., convolutions), computing them stripe by stripe so that the feature maps between them are never stored whole, but it does not cascade transposed convolutions [11]. We therefore replace all 11 transposed convolutions with standard convolutions of the same kernel size and insert a bilinear ×2 resize before each of the four former stride-2 layers (↑2 in Fig. 2), but keep Edge-FlowNet’s coarse-to-fine accumulation of flow increments. The peak then moves to final flow upsampling (Flow accum. in Fig. 3), and the restructured backbone fits a 172 × 224 input, at which EdgeFlowNet exceeds the tensor arena.

Because each convolution sees only a local window [18], we use the spare SRAM in the early stages (Fig. 3) to add image-wide context before flow prediction. Efficient channel attention (ECA) [19] reweights the channels of the lowestresolution feature, at 1/16 of the input resolution (DB0 in Fig. 2), where features carry broad spatial context and the tensor is small. A global broadcast gate adapted from squeezeand-excitation [18] then uses this context to modulate the 1/4 decoder feature, which feeds all three flow heads (Fig. 2).

## C. Configuration Space and Supernet Training (C)

The configuration space of PACE-NAS (Fig. 2) covers convolutional operators and residual-block depths and fixes connections, channel counts, upsamplers, ECA, and the gate. It contains no depthwise convolutions: on the Ethos-U55, 3×3 depthwise convolutions reach at most about 10% of peak throughput, whereas standard 3 × 3 convolutions exceed 90% once channel counts reach 32 [20]. Unlike channel counts, its choices leave tensor sizes unchanged and change only layers that need less SRAM than final flow upsampling, where the peak lies; no subnet can raise the peak, so the search needs no memory constraint (Section IV-B).

![](images/fe0bffc7b5ac63506634022dee486a0ba736178b784ca9e8102dd502bedfa65e.jpg)  
Fig. 3. Per-stage Vela profiles of the seed-42 INT8 baseline and restructured models (Table I) at $1 6 0 \times 2 0 8 \colon$ peak SRAM (top) and estimated cycles including memory access (bottom). Restructuring lowers the peak by 18% and moves it from the output head to final flow upsampling, leaving spare SRAM in the early stages. Stage names follow Fig. 2.

E0 and E1 each choose among three convolutional operators, which set the spatial context passed downstream; EB0, EB1, DB0, and DB1 each hold one to three residual blocks, which set capacity across scales; and each of the five costly high-resolution head convolutions uses a $3 \times 3$ or $5 \times 5$ kernel for local context (Fig. 2). Together, these choices give $3 ^ { 6 } \times 2 ^ { 5 } = 2 3 . 3 2 8$ subnets.

We train a weight-sharing supernet on FlyingChairs2 (FC2) [21] by sampling single paths, as in Single Path One-Shot (SPOS) [22]. Because subnets are scored with these shared weights instead of being retrained, we use FairNASinspired balanced sampling [23] to give competing operators comparable training exposure. Because the paths share weights, each sampled path pulls the shared layers toward its own features during supernet training [10]. We therefore add FlowNAS’s feature alignment distillation to the multi-scale flow and uncertainty losses [6]: at corresponding layers, every path learns to match a frozen EdgeFlowNet teacher, so all paths move toward the same features.

## D. Evolutionary Pareto Search and Model Selection (E)

As in SPOS, we score each subnet by its FC2 validation EPE with inherited supernet weights (supernet EPE) and by the latency that Vela estimates from its exported INT8 graph. NSGA-II [24] then searches for the EPE–latency Pareto front. From this front, we retrain two subnets from scratch and deploy them in INT8 (Section IV): MCUFlowNet-S, the fastest, and MCUFlowNet-L, the closest to EdgeFlowNet in Vela-estimated latency at $1 7 2 \times 2 2 4$ , for an equal-latency comparison.

## IV. EXPERIMENTAL EVALUATION

## A. Experimental Setup

The supernet is trained on FC2 at 172×224 for 200 epochs, after which NSGA-II evaluates 800 of the 23,328 subnets;

TABLE I  
DECODER ABLATION ON EDGEFLOWNET TRAINED ON FC2 AND FT3D AT 352 × 480: FP32 SINTEL FINAL EPE $( \mathrm { M E A N } \pm \mathrm { S D }$ , THREE SEEDS) AND VELA ESTIMATES AT $1 7 2 \times 2 2 4 .$
<table><tr><td>Decoder variant</td><td>EPE (px)↓</td><td>SRAM (KiB)↓</td><td>Latency (ms)↓</td></tr><tr><td>Transposed convolution</td><td> $6 . 9 6 \pm 0 . 0 3$ </td><td>1694*</td><td>195.2</td></tr><tr><td>Bilinear resize + convolution</td><td> $6 . 9 8 \pm 0 . 0 6$ </td><td>1386</td><td>165.0</td></tr><tr><td>+ ECA</td><td> ${ \bf 6 . 7 6 \pm 0 . 0 5 }$ </td><td>1386</td><td>165.3</td></tr><tr><td> $+ \mathrm { \ g a t e }$ </td><td> $6 . 8 3 \pm 0 . 0 6$ </td><td>1386</td><td>166.4</td></tr><tr><td>+  $\mathbf { \partial } \cdot \operatorname { E C A } + \mathbf { g a t e }$ </td><td> $6 . 7 8 \pm 0 . 0 4$ </td><td>1386</td><td>166.5</td></tr></table>

Exceeds the tensor arena.

together, these take an estimated 17 H200 GPU-hours, about half of one full EdgeFlowNet training run (Fig. 1).

For Table III, we train MCUFlowNet-S, MCUFlowNet-L, and EdgeFlowNet from scratch at $1 6 0 \times 2 0 8$ , the largest input at which EdgeFlowNet fits the tensor arena, with the same recipe: 400 epochs on FC2, then 10k steps on random FC2 crops resized to the input and 80k steps on a 3:1 mix of FC2 crops and FlyingThings3D (FT3D) frames [25]. For every model trained on FT3D, we keep the checkpoint with the lowest FT3D validation EPE. Test EPE is computed on all pairs of the Final pass of the MPI-Sintel training set [26] at its central $4 1 6 \times 1 0 2 4$ crop, without clipping large groundtruth displacements; deployed models see resized inputs, and their outputs are rescaled to the crop. The deployed models are quantized post-training to INT8 with 64 FC2 calibration pairs and compiled with Vela 4.5.0 for an Ethos-U55 with 64 multiply–accumulate (MAC) units at 400 MHz. Latencies in Tables III and IV are measured on the board with TensorFlow Lite Micro [27], excluding image capture and pre- and postprocessing.

## B. Macro Design Sets Memory, Search Allocates Latency

Table I ablates the macro-design components on Edge-FlowNet: the restructured decoder, ECA, and the gate. At $1 7 2 \times 2 2 4 .$ , restructuring reduces the Vela-estimated peak SRAM by 18%, to fit within the tensor arena, and latency by 15%, while the mean EPE changes by less than one standard deviation. ECA and the gate each lower EPE for every seed. Together they add 0.9% Vela-estimated latency without raising peak SRAM, and MCUFlowNet keeps both.

All 800 evaluated subnets share the same peak SRAM but span 2.19× in Vela-estimated latency, so the macro design alone sets the peak, and the search only decides where to spend latency. On the Ethos-U55, latency is better spent on depth at low resolution than on large high-resolution kernels such as EdgeFlowNet’s. An extra residual block at 1/8 or 1/16 resolution adds 9–24 ms and lowers the supernet EPE by 0.08– 0.21 px, whereas a $5 \times 5$ instead of a $3 \times 3$ kernel at H1 or H2 adds about 12 ms but lowers it by only 0.001–0.002 px. Accordingly, 50 of the 51 Pareto-optimal subnets use $3 \times 3$ kernels at H1 and H2, and MCUFlowNet-L spends 95% of its extra latency over MCUFlowNet-S on these low-resolution residual blocks. Vela’s latency estimates are reliable for the search: on the board, all deployed models run within 2.1% of their estimates.

TABLE II  
FP32 OPTICAL FLOW MODELS ON THE FINAL PASS OF THE MPI-SINTEL TRAINING SET AT 416 × 1024. ITERATION COUNTS ARE IN PARENTHESES.
<table><tr><td>Type</td><td>Model</td><td>Params (M)</td><td>MACs (G)</td><td>EPE (px)↓</td></tr><tr><td rowspan="4">Cost volume or warping</td><td>RAFT (32) [12]</td><td>5.26</td><td>762.4</td><td>2.77</td></tr><tr><td>NeuFlow v2 (9) [13]</td><td>9.03</td><td>109.1</td><td>2.59</td></tr><tr><td>FastFlowNet [15]</td><td>1.37</td><td>12.7*</td><td>4.13</td></tr><tr><td>SPyNet [14]</td><td>1.44</td><td>136.2</td><td>5.59</td></tr><tr><td>Encoder-decoder,</td><td>EdgeFlowNet [6]</td><td>2.74</td><td>18.6</td><td>8.20</td></tr><tr><td>no cost volume</td><td>MCUFlowNet-S†</td><td>2.65</td><td>17.8</td><td>6.80</td></tr><tr><td>or warping</td><td>MCUFlowNet-L†</td><td>5.90</td><td>28.4</td><td>5.79</td></tr></table>

<sup>∗</sup>Input resized to 448 × 1024, as FastFlowNet needs sides divisible by 64. <sup>†</sup>Trained with EdgeFlowNet’s two-stage scheme (FC2 at 352 × 480, then FT3D at 480 × 640), adding cosine learning-rate decay, gradient clipping, and FT3D augmentation.

## C. MCUFlowNet Runs Faster or as Fast at Similar EPE

Table II reports MCUFlowNet and published FP32 optical flow models with public weights. Models with cost volumes or warping reach the lowest EPEs, but in our tests none of them, not even FastFlowNet and SPyNet with the fewest parameters, could be deployed on the MCU. Compared with EdgeFlowNet, MCUFlowNet-S lowers the EPE by 17% at a similar MAC count, and MCUFlowNet-L by 29% with 1.5× the MACs.

Table III reports the deployed INT8 models on the board. Our recipe reduces the EPE of EdgeFlowNet by 20% relative to its public weights, so we compare against this retrained baseline. At the same input, MCUFlowNet-S runs 1.76× faster at a 1.6% higher EPE, and MCUFlowNet-L matches the baseline’s latency at a similar EPE. Both MCUFlowNet models need the same peak SRAM, 18% less than EdgeFlowNet, so choosing between them trades only speed against accuracy.

## D. Lower Latency Improves Obstacle Avoidance

We fly a simulated quadrotor through the AvoidBench forest [28] at 2 m/s; a trial succeeds if the quadrotor reaches the goal without collision. As in EdgeFlowNet [6], the controller steers toward free space near the goal direction. All flow models share this controller, and each steering update waits for the model’s Vela-estimated latency plus link and frame delays. MCUFlowNet-S succeeds more often than the retrained Edge-FlowNet (51.5% against 41.8%; Table III). When slowed to the steering interval of the retrained EdgeFlowNet, it succeeds in only 43.8% of trials and collides about as often as the baseline (0.91 against 0.92 per trial), so most of this gain comes from its lower latency, which lets the quadrotor react to obstacles earlier. At EdgeFlowNet’s latency, MCUFlowNet-L and the two EdgeFlowNet models succeed in 41.8% to 45.4% of trials whatever their EPE (6.36 to 8.03 px), so in this task latency matters more than EPE.

TABLE III  
INT8 MODELS ON GROVE VISION AI V2 AT 160 × 208. SRAM:  
VELA-ESTIMATED PEAK. SR AND COLL.: SUCCESS RATE AND COLLISIONS PER TRIAL OVER THE SAME 194 SIMULATED TRIALS FOR EVERY MODEL, AT 2 m/s.
<table><tr><td>Model</td><td></td><td>EPE (px)↓ SRAM (KiB)↓ Latency (ms)↓ SR (%)↑ Coll.↓</td><td></td><td></td><td></td></tr><tr><td>EdgeFlowNet (public)</td><td>8.03</td><td>1430</td><td>164.7</td><td>45.4</td><td>0.88</td></tr><tr><td>EdgeFlowNet (retrained)</td><td>6.40</td><td>1430</td><td>164.3</td><td>41.8</td><td>0.92</td></tr><tr><td>MCUFlowNet-S</td><td>6.50</td><td>1170</td><td>93.2</td><td>51.5</td><td>0.77</td></tr><tr><td>MCUFlowNet-L</td><td>6.36</td><td>1170</td><td>163.6</td><td>45.4</td><td>0.87</td></tr><tr><td>MCUFlowNet-S, slowed*</td><td></td><td></td><td>164.3</td><td>43.8</td><td>0.91</td></tr></table>

<sup>∗</sup>Steering interval set to the latency of the retrained EdgeFlowNet.

## E. Generalization to Segmentation

Beyond optical flow, we apply PACE-NAS to PIDNet-S [29] on Cityscapes [30] at a 160 × 320 input, at which PIDNet-S already peaks at 1350 KiB, 84 KiB below the tensor arena, so the macro design skips restructuring and adds only ECA and the gate. The configuration space covers the number of residual blocks in each of the seven stages and, in six of them, the second-convolution kernel size $( 3 \times 3$ or $5 \times 5 )$ of every block after the first. NSGA-II evaluates 960 of the 1.41 million subnets by INT8 mean intersection over union (mIoU) on the validation images and by Vela-estimated latency. All models start from the same ImageNet weights and use the official recipe with training views cropped from the full-resolution image, as for optical flow; we train three seeds per model and quantize them post-training with 256 training images. From the Pareto front, we retrain an accurate subnet and the subnets faster than PIDNet-S; Table IV reports three of them: a fast, a balanced, and an accurate subnet.

TABLE IV  
INT8 SEGMENTATION ON GROVE VISION AI V2 AT 160 × 320 (500 CITYSCAPES VALIDATION IMAGES; MEAN OVER THREE SEEDS, ± SD FOR MIOU); ALL MODELS PEAK AT 1350 KiB.
<table><tr><td>Model</td><td>mIoU (%)↑</td><td>Latency (ms)↓</td></tr><tr><td>PIDNet-S (retrained)</td><td> $5 5 . 2 3 \pm 0 . 5 5$ </td><td>188.6</td></tr><tr><td>PIDNet-S + ECA + gate</td><td> $5 5 . 4 4 \pm 0 . 7 3$ </td><td>191.4</td></tr><tr><td>Searched (fast)</td><td> $5 4 . 2 2 \pm 0 . 6 8$ </td><td>138.9</td></tr><tr><td>Searched (balanced)</td><td> $5 5 . 2 7 \pm 0 . 6 0$ </td><td>161.0</td></tr><tr><td>Searched (accurate)</td><td> ${ \pm } 5 5 . 8 1 \pm 0 . 5 4$ </td><td>200.4</td></tr></table>

Adding ECA and the gate raises the mean mIoU of the retrained PIDNet-S slightly, by 0.2 points (Table IV). Relative to the retrained PIDNet-S, the fast subnet trades 1.0 point of mIoU for a 1.36× speedup, the balanced subnet matches its mIoU (+0.04 points on average) with a 1.17× speedup, and the accurate subnet gains only 0.6 points, about one standard deviation, at 6% higher latency. The search can thus trade off accuracy against latency on segmentation as well.

## V. CONCLUSION

PACE-NAS treats peak SRAM as a design problem and latency as a search problem. Profiling (P) locates the layer that sets the peak SRAM, restructuring (A) lowers this peak once, for all subnets, and a configuration space (C) that leaves tensor sizes unchanged keeps it there, so the Pareto search (E) handles only accuracy and latency. PACE-NAS turns EdgeFlowNet, an existing network designed for the Edge TPU, into MCUFlowNet for the Ethos-U55. We build two variants, both with 18% less peak SRAM than EdgeFlowNet: against a retrained EdgeFlowNet, MCUFlowNet-S runs 1.76× faster at a 1.6% higher EPE and raises the simulated success rate at 2 m/s from 41.8% to 51.5%, while MCUFlowNet-L matches its latency at a similar EPE. The search also speeds up PIDNet-S for semantic segmentation by 1.17× at similar mIoU.

[1] R. J. Bouwmeester, F. Paredes-Vallés, and G. C. H. E. de Croon, “NanoFlowNet: Real-time dense optical flow on a nano quadcopter,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 1996–2003.

[2] F. Paredes-Vallés, J. J. Hagenaars, J. Dupeyroux, S. Stroobants, Y. Xu, and G. C. H. E. de Croon, “Fully neuromorphic vision and control for autonomous drone flight,” Sci. Robot., vol. 9, no. 90, 2024.

[3] D. Falanga, S. Kim, and D. Scaramuzza, “How fast is too fast? The role of perception latency in high-speed sense and avoid,” IEEE Robot. Autom. Lett., vol. 4, no. 2, pp. 1884–1891, 2019.

[4] D. Palossi, N. Zimmerman, A. Burrello, F. Conti, H. Müller, L. M. Gambardella, L. Benini, A. Giusti, and J. Guzzi, “Fully onboard AIpowered human-drone pose estimation on ultralow-power autonomous flying nano-UAVs,” IEEE Internet Things J., vol. 9, no. 3, pp. 1913– 1929, 2022.

[5] T. van Dijk, C. De Wagter, and G. C. H. E. de Croon, “Visual route following for tiny autonomous robots,” Sci. Robot., vol. 9, no. 92, 2024.

[6] S. R. K. P. Raju, R. Singh, M. Velmurugan, and N. J. Sanket, “Edge-FlowNet: 100FPS@1W dense optical flow for tiny mobile robots,” IEEE Robot. Autom. Lett., vol. 10, no. 1, pp. 128–135, 2025.

[7] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, and Q. V. Le, “MnasNet: Platform-aware neural architecture search for mobile,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 2815–2823.

[8] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, “MCUNet: Tiny deep learning on IoT devices,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 11 711–11 722.

[9] J. Lin, W.-M. Chen, H. Cai, C. Gan, and S. Han, “Memory-efficient patch-based inference for tiny deep learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 2346–2358.

[10] Z. Lin, T. Liang, T. Xiao, Y. Wang, and M.-H. Yang, “FlowNAS: Neural architecture search for optical flow estimation,” Int. J. Comput. Vis., vol. 132, no. 4, pp. 1055–1074, 2024.

[11] Arm Ltd, “Ethos-U Vela,” Software, version 4.5.0, 2025, accessed: Oct. 9, 2026. [Online]. Available: https://gitlab.arm.com/ artificial-intelligence/ethos-u/ethos-u-vela/-/tree/4.5.0

[12] Z. Teed and J. Deng, “RAFT: Recurrent all-pairs field transforms for optical flow,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 402– 419.

[13] Z. Zhang, A. Gupta, H. Jiang, and H. Singh, “NeuFlow-V2: Push highefficiency optical flow to the limit,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025, pp. 2479–2485.

[14] A. Ranjan and M. J. Black, “Optical flow estimation using a spatial pyramid network,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2720–2729.

[15] L. Kong, C. Shen, and J. Yang, “FastFlowNet: A lightweight network for fast optical flow estimation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2021, pp. 10 310–10 316.

[16] J. Kühne, M. Magno, and L. Benini, “Parallelizing optical flow estimation on an ultra-low power RISC-V cluster for Nano-UAV navigation,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), 2022, pp. 301–305.

[17] S. Narduzzi, R. Vuagniaux, K. Sharma, S.-C. Liu, and L. A. Dunbar, “Steerable zero-shot neural architecture search for efficient edge inference,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), 2025, pp. 1–5.

[18] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132–7141.

[19] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, “ECA-Net: Efficient channel attention for deep convolutional neural networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 534–11 542.

[20] Y. Zhao, J. Chen, S. Q. Zhang, S. S. Sarwar, K. H. Stangherlin, J. T. Gomez, J.-S. Seo, B. De Salvo, C. Liu, P. B. Gibbons, and Z. Li, “H4H: Hybrid convolution-transformer architecture search for NPU-CIM heterogeneous systems for AR/VR applications,” in Proc. Asia South Pacific Design Autom. Conf. (ASP-DAC), 2025, pp. 1133–1141.

[21] E. Ilg, T. Saikia, M. Keuper, and T. Brox, “Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 626–643.

[22] Z. Guo, X. Zhang, H. Mu, W. Heng, Z. Liu, Y. Wei, and J. Sun, “Single path one-shot neural architecture search with uniform sampling,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 544–560.

[23] X. Chu, B. Zhang, and R. Xu, “FairNAS: Rethinking evaluation fairness of weight sharing neural architecture search,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 12 239–12 248.

[24] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, 2002.

[25] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox, “A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 4040–4048.

[26] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black, “A naturalistic open source movie for optical flow evaluation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2012, pp. 611–625.

[27] R. David, J. Duke, A. Jain, V. Janapa Reddi, N. Jeffries, J. Li, N. Kreeger, I. Nappier, M. Natraj, T. Wang, P. Warden, and R. Rhodes, “TensorFlow Lite Micro: Embedded machine learning for TinyML systems,” in Proc Mach. Learn. Syst. (MLSys), vol. 3, 2021, pp. 800–811.

[28] H. Yu, G. C. H. E. de Croon, and C. De Wagter, “AvoidBench: A highfidelity vision-based obstacle avoidance benchmarking suite for multirotors,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 9183– 9189.

[29] J. Xu, Z. Xiong, and S. P. Bhattacharyya, “PIDNet: A real-time semantic segmentation network inspired by PID controllers,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 19 529–19 539.

[30] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes dataset for semantic urban scene understanding,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 3213–3223.