# Grid or Nodes? Probing Object Memory in World Models Under Partial Observability

Justin McMillen University of South Florida Tampa, FL, USA jmcmillen@usf.edu

Yasin Yilmaz University of South Florida Tampa, FL, USA yasiny@usf.edu

Abstract: A robot’s world model must remember objects that leave the camera’s view. FloWM, a learned world model, keeps them in a spatial grid of features that it moves with five built-in velocities. A node memory instead keeps each object’s position and velocity as numbers. We compare them in FloWM’s own block world, with an oracle-fed Kalman tracker as the node memory. How the grid is read changes its score: a single-cell probe finds 66% of out-of-view blocks, while FloWM’s own probe 89%. FloWM also scores 8 points higher in view although no frame is shown, which we attribute to its write-back of its own predictions. Out of view, the tracker reaches FloWM’s range in 14KB against 5.6MB, with no probe. On diagonal motion, outside its five velocities, FloWM finds 11% of blocks even after fine-tuning, while the unchanged tracker keeps 93%. Faster blocks defeat both until one more allowed speed takes the tracker to 94%, where FloWM’s built-in shifts would need 25 copies of its grid. A learned node memory must still replace the oracles and handle velocity under noise, long gaps, and the bounce.

Keywords: world models, partial observability, object memory, probing

## 1 Introduction

A robot that turns away from a moving cart must still know where the cart has gone. Robots increasingly plan with learned world models [1–3], so these models need a memory for objects that leave the camera’s view. Where should that memory keep a moving object? FloWM [4] keeps a top-down grid of learned features and moves it with a fixed set of built-in velocities. Its motion prior, what it assumes about motion, is therefore part of the memory’s layout. Our node-based memory platform instead stores one node per object, with its position and velocity as numbers.

Other video world models add a memory for consistency when the camera returns [5, 6], or keep state for subjects out of view [7, 8]. Related to a node memory are tracking by detection [9], models of object permanence [10, 11] and object-centric models with one representation per object [12–16]. Memories are usually judged by probes, networks trained to read out a property [17–19], whose scores reflect the probe as well as the memory [20, 21]. Benchmarks test what world models remember [22–26], and others question what replay quality or next-step accuracy shows [27–29].

We compare the two designs on FloWM’s own task, a room of sliding blocks. Each memory receives 70 steps of context and then predicts 70 steps from the agent’s actions alone. FloWM works from pixels, and the node memory is a Kalman tracker with oracle inputs: the simulator gives it block masks and slightly noisy positions. The tracker as-is is not a deployable system.

Does the grid hold a block’s position, or does the probe compute it (Sec. 3.1)? On FloWM’s own setting, how close does a node memory come (Sec. 3.2)? What happens outside the built-in velocities (Sec. 3.3)? What stands between the tracker and a full platform? (Sec. 3.4)?

![](images/08251148a3bb706a374d5e68187e5a61d58b0f202cc8c1d7c68e7699d280f81d.jpg)  
Figure 1: (a) The room from above. Solid arrows show motion in the standard setting, and dashed arrows the diagonal and speed-2 (double head) motion that the varied setting adds. (b) One prediction step of FloWM’s grid memory. (c) The tracker, a node memory.

## 2 Two memories

## 2.1 Environment and protocol

Dynamic Block World [4] is a Miniworld [30] room (Fig. 1a) whose floor is a grid of cells. Each evaluation video holds six blocks, one per color, which slide across the floor and bounce off the walls. An agent moves one cell or turns 90<sup>◦</sup> per step and sees through a 60<sup>◦</sup> camera. Each video has a context phase of 70 frames with actions and a prediction phase of 70 actions without frames.

At every predicted step, each memory names, for each block, one cell of a grid centered on the agent. Only the block’s exact cell is correct, so chance is below 0.1%. A sighting is a context frame in which the block is visible. No frame is shown during prediction. We call a block in view at a step when it lies inside the field of view the camera would have then, and out ofview otherwise.

In the standard setting, which follows FloWM’s published data, every block moves one cell per step along one floor axis. In our varied setting, each block draws a random nonzero velocity with whole-number components from −2 to 2 cells per step. Only 17% of blocks then move as in the standard setting (built-in), and 16% move one cell per step on both axes (diagonal). The other 67% move two cells per step on some axis (speed 2). We tested on 256 standard and 1,024 varied videos, and fitted probes and tracker settings on separate tuning and training videos (Apps. A and C).

## 2.2 FloWM’s grid memory

FloWM is trained end to end to predict camera frames. Its memory is a top-down grid around the agent, 33 × 33 cells that each hold 256 learned features. The memory holds five copies of this grid, one per built-in velocity: standing still, and one cell per step in each of the four axis directions (Fig. 1b). During each time-step, FLoWM’s memory does four operations: Move: every copy shifts by its own velocity, and all copies shift or rotate against the agent’s motion. These built-in shifts carry a block’s content while it is out of view. Mix: a learned 3 × 3 convolution combines the copies, so content can change copies when a block bounces. Render: in the prediction phase, a learned decoder renders the predicted frame from the cells in view. Write: a learned transformer writes a frame into the cells in view. It writes the camera frame during context and its own rendered frame during prediction, which we call the write-back.

The memory never stores a position as a number, so a probe must read it. The memory can carry only motion that the five copies and the mix can express. Its size grows with the area of the room times the number of velocities, not with the number of objects.

## 2.3 A node memory with oracle perception

A node memory stores one node per object (Fig. 1c). Each node holds a position and a velocity as numbers, their uncertainties, and appearance features. A node moves by its own velocity, so no velocity is built into the memory’s layout. The tracker keeps one Kalman filter [31] per block as its node, called a track. It matches detections to tracks by distance and color, in the style of SORT [9]. It has no trained networks. The simulator provides what a learned node memory must produce. These oracles are block masks, positions with one pixel of noise and their uncertainty, the walls, and at readout each detection’s identity, linking tracks to blocks (App. C). A minimal working node memory therefore still needs four learned parts: a detector that finds blocks and estimates their floor positions from pixels, a readout that links nodes to objects without true identities, a bounce model that does not know the walls, and velocity estimates that hold up without hand-set allowed speeds.

Table 1: Accuracy (%). Top: standard test videos and state size. Bottom: varied test videos, out of view. <sup>a</sup>Convolutional probe trained on the standard setting. <sup>b</sup>Probes trained on varied videos.
<table><tr><td>Standard setting</td><td>In view</td><td>Out of view</td><td>State</td></tr><tr><td>FloWM, convolutional probe</td><td>96.5</td><td>88.6</td><td>5.6MB</td></tr><tr><td>FloWM, single-cell probe</td><td>95.7</td><td>66.1</td><td>5.6MB</td></tr><tr><td>Tracker</td><td>90.9</td><td>90.8</td><td>14KB</td></tr><tr><td>Varied setting, out of view</td><td>Built-in</td><td>Diagonal</td><td>Speed 2</td></tr><tr><td>FloWM, releasedª</td><td>72.7</td><td>3.5</td><td>4.2</td></tr><tr><td>FloWM, fine-tunedb</td><td>39.1</td><td>11.0</td><td>13.0</td></tr><tr><td>FloWM, retrainedb</td><td>16.8</td><td>8.2</td><td>9.4</td></tr><tr><td>Tracker, speeds 0 and 1</td><td>89.9</td><td>93.1</td><td>4.9</td></tr><tr><td>Tracker, speeds 0, 1 and 2</td><td>91.0</td><td>94.1</td><td>93.9</td></tr></table>

![](images/c9c09a0a0a477ea7cb9a83472c8823e628b27676778e27aeaa338802ccebbdcf.jpg)  
Steps since last context sighting  
Figure 2: Accuracy by steps since a block’s last sighting, on the standard test videos, with FloWM read by the convolutional probe. Solid lines are blocks in view and dashed lines blocks out of view. The tracker’s two lines stay close from 5 to 99 steps, and the annotation marks FloWM’s in-view lead. Bars are 95% intervals, and hollow markers mark sparse bins.

Like FloWM, the tracker has a prior over speeds. A small error in a velocity estimated from noisy positions grows into a large position error over a long gap without sightings. The tracker therefore snaps each axis of a track’s velocity to an allowed speed, which we call speed snapping. On the standard setting the allowed speeds are 0 and 1 cell per step, fitted on training videos.

## 3 Results

## 3.1 Does the grid hold a position, or does the probe compute it?

We used two probes to answer this question. A convolutional probe, rebuilt from FloWM’s appendix, sees the whole grid and has 7.06M parameters per head, more than half of FloWM’s own 12.8M. A single-cell probe applies one linear map of 1,282 parameters to each cell’s own features. It finds a block only where the memory stores it in the block’s own cell. We trained one probe head per block color on frozen memory states (App. B). For blocks in view the single-cell probe finds the position, and for blocks out of view the convolutional probe computes part of it. The two probes agree on blocks in view and disagree out of view (Table 1, top). There the single-cell probe finds 66% of blocks and the convolutional probe 89%. The extra hits need neighboring cells or a nonlinear combination of features.

No block is observed during prediction, yet FloWM, read by the convolutional probe, scores 8 points higher in view than out of view. The difference holds at every time since the last sighting from 5 steps on (Fig. 2)<sup>†</sup>, while the tracker shows none. FloWM’s score also rises by 5 points when a block enters view<sup>†</sup>(App. D). We attribute both to the write-back, which rewrites the cells in view from FloWM’s own rendered frames. The write-back adds no observation, but in-view scores depend on rendering as well as memory, so we compare memories out of view for fairness.

## 3.2 On FloWM’s own setting, how close does a node memory come?

With oracle perception, a node memory reaches FloWM’s range. Out of view the tracker finds 91% of blocks, and FloWM, read by the convolutional probe, finds 89% (Table 1, top). We call this the same range and not a win, because our rebuilt probe scores 90% overall where FloWM’s authors report 96%. In view FloWM leads by about 6 points. The inputs also differ, since FloWM perceives from pixels and the tracker receives oracle detections. The tracker’s state takes 14KB against 5.6MB for FloWM’s five copies of the grid (App. H), from which FloWM also renders frames. The tracker’s positions are numbers, so no probe is needed to read them once tracks are linked to objects.

## 3.3 What happens outside the built-in velocities?

Each memory loses blocks that move outside its motion prior, and the known fixes differ in cost. On varied test videos, FloWM’s released model finds 73% of built-in blocks but 4% of diagonal and speed-2 blocks (Table 1, bottom). Using MSE, its prediction error on frames is 23 times higher than on the standard setting. Fine-tuning or retraining on varied videos lowers it, yet probes trained on those videos find at most 13% of diagonal and speed-2 blocks.

The tracker loses speed-2 blocks in the same way. Unchanged, with allowed speeds 0 and 1, it keeps diagonal blocks, which per-axis speeds can express, and finds 5% of speed-2 blocks. Without speed snapping it finds about 60% of them, so for the tracker a wrong prior costs more than none (App. F). With the allowed speeds refitted to 0, 1 and 2 on varied training videos, it finds 94% of diagonal and speed-2 blocks. With exact positions the tracker needs no allowed speeds and finds 97% of all blocks. Covering the varied setting’s 24 velocities with built-in shifts would take 25 copies of the grid, instead of five [4, their Apps. F and G.1]. Training that model is out of scope for this paper.

## 3.4 What stands between the tracker and the full platform?

A full node-based memory platform must replace every oracle in Sec. 2.3, and solve three more problems. The first is velocity under noise. Without speed snapping the tracker finds 47% of out-ofview blocks on the standard setting, not 91% (App. C), and snapping assumes few, discrete speeds.

The second is long gaps. Out of view, the tracker leads FloWM by 6 points at 10 to 19 steps since the last sighting and trails by 10 points after 100<sup>†</sup> (Fig. 2). Two causes explain the late deficit<sup>†</sup>. One sighting gives the tracker a position but no velocity. At long gaps FloWM finds 30% of blocks sighted once, the tracker almost none. Tuning videos suggest FloWM infers their direction (App. E). Other tracks pick up error after the last sighting, as when another block’s detection joins them.

The third is the bounce. We replaced the exact reflection off the true walls with a bounce trained on the track’s position error at each new sighting<sup>†</sup>. We ran this on training videos with up to ten blocks and repeated colors. The share of matches that join a detection to another block’s track rises from 16% to between 19% and 26%, against 17.5% without a bounce model (App. G).

## 4 Conclusion

On the standard setting, where both priors cover the motion, a node memory with oracle perception scores in the same range as FloWM, which works from pixels. When the prior is wrong, both memories struggle to accurately track blocks. Our evidence qualifies the workshop’s Motion 2: explicit dynamics generalize only as far as their motion prior reaches for both memories. We tested new dynamics, not new goals, and no implicit model. It supports Motion 6: the same memory scores differently with a different probe, and FloWM’s in-view scores rise through its own write-back, no through memory. Fair comparisons should report the probe and score objects out of view.

Limitations. We used one synthetic environment and one released FloWM model. The tracker’s inputs are oracles (Sec. 2.3). The two probes differ in both reach and capacity, so the single-cell probe isolates neither. We froze the tracker’s settings for the standard setting after seeing some test results. App. I lists the choices marked † and gives the intervals we have.

## References

[1] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025. doi:10.1038/s41586-025-08744-2.

[2] G. Zhou, H. Pan, Y. LeCun, and L. Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In International Conference on Machine Learning (ICML), 2025.

[3] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv:2506.09985, 2025.

[4] H. J. Lillemark, B. Huang, F. Zhan, Y. Du, and T. A. Keller. Flow equivariant world models: Memory for partially observed dynamic environments. In International Conference on Machine Learning (ICML), 2026.

[5] Z. Xiao, Y. Lan, Y. Zhou, W. Ouyang, S. Yang, Y. Zeng, and X. Pan. WorldMem: Longterm consistent world simulation with memory. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[6] J. Huang, X. Hu, B. Han, S. Shi, Z. Tian, T. He, and L. Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on Minecraft. arXiv:2510.03198, 2025.

[7] Z. Duan, J. Xia, Z. Zhang, W. Zhang, G. Zhou, C. Gou, Y. He, F. Chen, X. Zhang, and L. Liu. LiveWorld: Simulating out-of-sight dynamics in generative video world models. arXiv:2603.07145, 2026.

[8] K. Chen, D. Liang, X. Zhou, Y. Ding, X. Liu, P. Wan, and X. Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv:2603.25716, 2026.

[9] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft. Simple online and realtime tracking. In IEEE International Conference on Image Processing (ICIP), 2016.

[10] A. Shamsian, O. Kleinfeld, A. Globerson, and G. Chechik. Learning object permanence from video. In European Conference on Computer Vision (ECCV), 2020.

[11] P. Tokmakov, J. Li, W. Burgard, and A. Gaidon. Learning to track with object permanence. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[12] F. Locatello, D. Weissenborn, T. Unterthiner, A. Mahendran, G. Heigold, J. Uszkoreit, A. Dosovitskiy, and T. Kipf. Object-centric learning with slot attention. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[13] T. Kipf, E. van der Pol, and M. Welling. Contrastive learning of structured world models. In International Conference on Learning Representations (ICLR), 2020.

[14] A. Creswell, R. Kabra, C. Burgess, and M. Shanahan. Unsupervised object-based transition models for 3D partially observable environments. arXiv:2103.04693, 2021.

[15] T. Daniel, C. Qi, D. Haramati, A. Zadeh, C. Li, A. Tamar, D. Pathak, and D. Held. Latent particle world models: Self-supervised object-centric stochastic dynamics modeling. In International Conference on Learning Representations (ICLR), 2026.

[16] Y. Nishimoto and T. Matsubara. Object-centric world models for causality-aware reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2026.

[17] K. Li, A. K. Hopkins, D. Bau, F. Viégas, H. Pfister, and M. Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In International Conference on Learning Representations (ICLR), 2023.

[18] K. Gregor, D. J. Rezende, F. Besse, Y. Wu, H. Merzic, and A. van den Oord. Shaping belief states with generative environment models for RL. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

[19] E. Wijmans, M. Savva, I. Essa, S. Lee, A. S. Morcos, and D. Batra. Emergence of maps in the memories of blind navigation agents. In International Conference on Learning Representations (ICLR), 2023.

[20] J. Hewitt and P. Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743, 2019. doi:10.18653/v1/D19-1275.

[21] Y. Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022. doi:10.1162/coli\_a\_00422.

[22] J. Lu, D. Zhu, H. Shi, L. Cai, G. Tang, Y. Chen, J. Cao, D. Tang, Y. Zhang, Y. Dai, and X. Ju. Current world models lack a persistent state core. arXiv:2606.20545, 2026.

[23] Z. Ma, M. Liufu, and G. Gkioxari. Out of sight, out of mind? evaluating state evolution in video world models. arXiv:2603.13215, 2026.

[24] H. Chen, K. Zhou, H. Hua, K. Zhang, J. Qian, W. Ma, H. Chen, C. Liu, Y. Zhao, X. Wang, W. Li, A. Yuille, P. P. Liang, and Y. Du. MemoBench: Benchmarking world modeling in dynamically changing environments. arXiv:2606.27537, 2026.

[25] Q. Gu, B. Gao, R. Chen, G. Li, J. Li, Q. Wen, L. Niu, J. Tang, X. Chu, and J. Zhao. R2M-Bench: Evaluating revisit memory via relative consistency in interactive video world models. arXiv:2608.27328, 2026.

[26] K. Lian, S. Cai, Y. Liang, and A. Liu. LoopNav: Benchmarking spatial consistency in world models. arXiv:2505.22976, 2025.

[27] W. King, Z. Xue, Y. Bian, J. Huang, H. Li, Y. Li, Y. Su, Y. Li, H. Wang, S. Zhang, S. Zhang, Y. Niu, S. Xu, J. Zhuang, H. Huang, and N. Duan. Echo-Memory: A controlled study of memory in action world models. arXiv:2606.09803, 2026.

[28] K. Vafa, J. Y. Chen, A. Rambachan, J. Kleinberg, and S. Mullainathan. Evaluating the world model implicit in a generative model. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2024.

[29] J. Zhang, M. Jiang, N. Dai, T. Lu, A. Uzunoglu, S. Zhang, Y. Wei, J. Wang, V. M. Patel, P. P. Liang, D. Khashabi, C. Peng, R. Chellappa, T. Shu, A. Yuille, Y. Du, and J. Chen. Worldin-world: World models in a closed-loop world. In International Conference on Learning Representations (ICLR), 2026.

[30] M. Chevalier-Boisvert, B. Dai, M. Towers, R. de Lazcano, L. Willems, S. Lahlou, S. Pal, P. S. Castro, and J. Terry. Minigrid & Miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.

[31] R. E. Kalman. A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1):35–45, 1960.

[32] E. Ristani, F. Solera, R. S. Zou, R. Cucchiara, and C. Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European Conference on Computer Vision (ECCV) Workshops, 2016.

## A Environment, components and protocol

## A.1 Environment

Dynamic Block World [4] is a room built in Miniworld [30] (Fig. 1a). The floor is a grid of cells, and the walls are 16 cells apart. Colored blocks, each 0.7 cells wide, slide across the floor and bounce off the walls. An agent moves through the room in steps of one whole cell and turns in steps of 90<sup>◦</sup>. It sees the room through a camera with a $6 0 ^ { \circ }$ field of view, as 128 × 128 color frames.

Each video has two phases. In the context phase, a model sees 70 frames together with the agent’s actions. In the prediction phase, the model receives only the agent’s next 70 actions and sees no frames. We use two settings that differ only in how the blocks move.

In the standard setting, which follows the generator settings of FloWM’s published data, every block moves one cell per step along one of the two floor axes. We created a varied setting, in which each block gets a random nonzero velocity whose components are whole numbers of cells per step between −2 and 2. Only 16.9% of blocks in the varied setting move as they would in the standard setting. Another 15.8% move diagonally, one cell per step on each axis. The remaining 67.3% move two cells per step on at least one axis. We generated both settings with FloWM’s own version of Miniworld. In the varied setting, blocks faster than one cell per step move in one-cell sub-steps, so that they bounce off the walls instead of passing through them.

## A.2 FloWM’s memory in detail

FloWM keeps its memory as a top-down map of the floor around the agent. The map is a $3 3 \times 3 3$ grid of cells centered on the agent, and one cell of the map covers one cell of the floor. Each cell holds a vector of 256 learned features, and content about a block is meant to sit in the cell where the block is. The memory holds five copies of this grid. Each copy is tied to one velocity from a fixed set, the built-in velocities. The five velocities are standing still and moving one cell per step in each of the four axis directions (Fig. 1b). A block that moves in the +x direction should therefore be stored in the +x copy.

Every step, FloWM first moves and mixes the copies. In the context phase, it then writes the camera frame into the memory. In the prediction phase, it then renders a frame from the memory and writes that frame back into the memory.

• Move. FloWM shifts each copy by one cell in that copy’s direction and leaves the still copy in place. It also moves all copies against the agent’s own motion, so that the grid stays centered on the agent: a step forward shifts the grid by one cell, and a turn rotates the grid by 90<sup>◦</sup> and turns each copy’s direction with it. These moves are built in and are not learned. They carry a block’s content along with the block, even while the block is out of view.

• Mix. A learned $3 \times 3$ convolution combines the five copies, separately for each of the 256 features (57.6k parameters). This step can move content from one copy to another, which a bounce requires, since a block that hits a wall must move from the +x copy to the −x copy. Because the convolution spans neighboring cells, it can also move content by up to one cell. The mix applies the same weights at every cell, however, so it cannot tell where a wall is, and the content that turns a block at a wall must come from what the write stored.

• Render. During the prediction phase, a learned decoder renders the predicted camera frame from the cells in view, after the move and the mix and before that step’s write.

• Write. A learned transformer writes a frame into the memory. It updates only a fixed set of 177 cells, a 60<sup>◦</sup> wedge in front of the agent that covers the camera’s view, in all five copies. During the context phase it writes the real camera frame. During the prediction phase there is no camera frame, so FloWM writes the frame it has just rendered at the same step. We call this the write-back.

FloWM is trained end to end to predict frames, with the mean squared error (MSE) between predicted and true frames as its loss. It has 12.8M parameters, and its memory holds 33×33×5×256 ≈ 1.4M numbers. The velocity copies matter, since removing them raises the prediction error 12.6- fold [4, Table 2]. We use the released checkpoint, which reproduces the published MSE of 0.000603 on the release’s own validation clips.

Three properties of this memory matter for our study. First, it never stores a block’s position as a number. The position is implied by which cells hold the block’s content, so a probe is needed to read it. Second, the memory can only carry motion that its five copies, together with the mix, can express. Third, its size grows with the area of the room and the number of velocities, not with the number of objects.

## A.3 Where each part of the two memories comes from

Table 2 lists which parts of the tracker the simulator provides (oracle), which we wrote by hand, and which we fitted on oracle data, and it compares each part with FloWM.

Table 2: Where each part of the two memories comes from. Oracle parts are given by the simulator. Hand-set parts are written by us, with no training. Fitted parts are estimated from noisy oracle positions on training videos of the standard setting, with thresholds chosen on the tuning videos. Built in parts are fixed by FloWM’s design. Learned parts are trained with FloWM’s frame-prediction loss, except the probe, which is trained to read positions. The table describes the tracker with allowed speeds 0 and 1.

<table><tr><td>Part</td><td>Tracker</td><td>FloWM</td></tr><tr><td>Finding blocks in a frame</td><td>Oracle: true masks</td><td>Learned: write transformer</td></tr><tr><td>Block position in a frame</td><td>Oracle: true position plus one-pixel noise</td><td>Learned: write transformer</td></tr><tr><td>Block appearance</td><td>Oracle: mean color inside the true mask</td><td>Learned: 256 features per cell</td></tr><tr><td>Agent&#x27;s own motion</td><td>Hand-set: from actions</td><td>Built in: shift and rotation by actions</td></tr><tr><td>Motion between steps</td><td>Hand-set: Kalman predict step</td><td>Built in: five velocity copies</td></tr><tr><td>Allowed speeds</td><td>Fitted: snap each axis to 0 or 1</td><td>Built in: the five velocities</td></tr><tr><td>Bounces</td><td>Oracle: true walls, exact reflection</td><td>Learned: write transformer and mix layer</td></tr><tr><td>Linking a sighting to memory Reading out a position</td><td>Hand-set: SORT-style matching Oracle: track last matched by the block</td><td>Learned: write transformer Learned: probe per color</td></tr></table>

## A.4 Task and scoring

![](images/87991957c80c53226507bb8384618d4b2edda5c772e0884ec0df1311acc9d85f.jpg)

Figure 3: The protocol for one block. Both memories receive the same context phase. FloWM receives the frames, and the tracker receives oracle detections from the same frames. At every predicted step, each memory names one grid cell for the block, and the answer is correct only if it is the block’s exact cell. A sighting is a context frame in which the block covers at least 12 pixels. We split answers by whether the block lies inside the camera’s view at the scored step and by the steps since the block’s last sighting.

Figure 3 shows the protocol. Both memories receive the same videos. During the context phase, FloWM receives the 70 frames and actions, and the tracker receives oracle detections from the same

70 frames together with the same actions. During the prediction phase, both receive only the next   
70 actions.

At every one of the 70 predicted steps, each memory names one cell of the agent-centered 33 × 33 grid for each block, or states that the block is off the grid. This gives 1,090 possible answers. An answer is correct only if it is the cell that contains the block’s center at that step, counted in cells forward and to the right of the agent, so an answer one cell away counts as wrong. The room is smaller than the grid, so the correct answer is never off the grid. Accuracy is the fraction of correct answers over all blocks and all predicted steps. Guessing at random would score below 0.1%.

Each video that we score with probes or the tracker contains six blocks, one of each color (red, blue, green, purple, gray and yellow), so each block can be identified by its color. We use 978 videos from the standard setting as a tuning set for training probes and for choosing the tracker’s settings. The tracker’s allowed speeds and the learned bounce use separate training videos (Apps. C and G). We report results on 256 fresh videos from the standard setting, which we call the test set, and on 1,024 fresh videos from the varied setting. Confidence intervals are 95% intervals from resampling whole videos (a video bootstrap).

## A.5 Splitting the score

We split scores on the standard setting in three ways.

In view or out ofview. A block is in view at a predicted step if it lies inside the camera’s 60<sup>◦</sup> field of view at that step, even if another block hides it. During the prediction phase no frames are shown, so a block in view is no more observed than a block behind the agent. About four in five answers on the test set are out of view (85,602 of 107,520).

Steps since the last sighting. A sighting is a context frame in which the block’s true mask covers at least 12 pixels. For each answer, we count the steps from the block’s last sighting to the scored step (Fig. 3). On the test set this count runs from 1 to 137. Sixteen of the 1,536 test blocks are never sighted during the context phase, and they give about 1% of answers, both in view and out of view. These answers have no count, so they appear in neither this split nor the split by number of sightings, but they stay in every other score. The tracker has no track for such a block, so all its answers for it are wrong, while FloWM finds 5.0% of them in view and 2.2% out of view.

## B Probes

FloWM’s memory holds features rather than positions, so we need a probe to read a position out of it. We train one probe head per block color. Each head looks at the frozen memory at a predicted step, after that step’s write-back, and outputs one of the 1,090 answers for its color’s block. Heads are trained on memory states saved from the tuning videos and are then scored, unchanged, on the test videos. We use two probes.

The convolutional probe is the probe described in FloWM’s appendix. It is a convolutional network that sees the whole grid, with 7.06M parameters per head, which is more than half the size of FloWM itself (12.8M). Its code was not released, so we rebuilt it. Our rebuild scores 90.2% on the test set, while FloWM’s paper reports 96%.

The single-cell probe scores each cell using only that cell’s own features in the five copies. It applies one small linear map, shared by all cells, and has 1,282 parameters per head. It cannot gather evidence from neighboring cells, and it cannot combine features in a nonlinear way. It can therefore find a block only where the memory stores that block in the correct cell.

The tracker needs no probe. Its answer for a block is the grid cell that contains the position of the block’s last matched track, measured from the agent’s position and heading as computed from its actions, and rounded as the labels are. A block that was never sighted has no track, so its answers count as wrong.

## B.1 The probe changes the score

The two probes agree on blocks in view, where the single-cell probe scores 95.7% and the convolutional probe scores 96.5% (Table 1). They disagree on blocks out of view. The single-cell probe finds 66.1% of out-of-view blocks, while the convolutional probe finds 88.6%. About a quarter of the convolutional probe’s out-of-view hits therefore need information from neighboring cells or a nonlinear combination of features. For these blocks, the memory does not store the block in its own cell in a form that a simple read can find, and the probe computes part of the answer.

This matters when we compare memories. Out of view, the tracker scores 90.8%. It is 2.1 points above FloWM read by the convolutional probe (CI 0.5 to 3.7), and about 25 points above FloWM read by the single-cell probe. We do not read the first gap as a win for the tracker, because our rebuilt probe scores below the published one. Which memory looks better therefore depends on the probe. Counting all answers in view and out of view, the convolutional probe scores 90.2%, the single-cel probe 72.1% and the tracker 90.8%.

## C The tracker with oracle perception

A node memory takes the opposite approach to FloWM’s grid. It stores one node per object. Each node holds the object’s position and velocity as explicit numbers, an uncertainty for each, and features that describe the object’s appearance. Each node moves by its own velocity, so the memory itself does not limit which velocities it can carry, and its size grows with the number of objects rather than the area of the room. A learned node memory must solve several core problems. It must find objects in the image, estimate their floor positions from the image, learn how they move and bounce, and decide which node each new sighting belongs to.

To separate the bookkeeping from these perception problems, we built the simplest node memory and gave it perception for free. This is a Kalman tracker that receives true block positions and masks from the simulator. It contains no trained networks.

## C.1 The Kalman filter

A Kalman filter [31] keeps a best guess of an object’s state together with a measure of how uncertain that guess is. In the tracker, the state of each block is its floor position and velocity, $( x , y , v _ { x } , v _ { y } )$ , and the uncertainty is a 4 × 4 covariance matrix. The filter alternates two steps. The predict step moves the guess forward by one time step using the current velocity, and it increases the uncertainty to allow for motion that the filter cannot foresee. The update step receives a measured position and moves the guess toward it. The size of this move depends on the two uncertainties. A confident prediction moves only a little, while a confident measurement moves it a lot. The filter never measures velocity directly. It infers velocity from how the measured position changes across sightings. A track that receives only one detection therefore keeps its starting velocity of zero, and it stays where the block was seen.

## C.2 Matching and tracks

A tracker runs one filter per object and must decide which detection belongs to which filter. Our matching is in the style of SORT [9], and the tracker repeats four steps at every time step. First, it predicts every track forward, and when a track’s center would cross a line one cell inside a wall, where a real block’s center turns around, it mirrors the track back and reverses its velocity on that axis. Second, it matches the detections in the current frame to tracks. The cost of a pairing grows with the distance between the detection and the track’s predicted position, measured against their combined uncertainty, and with their difference in mean color, which counts more the longer the track has gone unseen. Pairs farther apart than the 99% range of that uncertainty are not allowed, and the tracker repeatedly takes the cheapest allowed pair, so it assigns at most one detection to each track. A leftover detection may still join a track that has gone unseen for several steps, under a wider distance limit, if their colors nearly match. Third, it updates each matched track with its detection. Fourth, it starts a new track for each detection that matched no track. A track whose block is out of view receives no detection of that block, so it only predicts, unless a detection of another block is matched to it. Like FloWM, the tracker computes the agent’s own position and heading from its actions, which are exact in this environment.

## C.3 Speed snapping

A velocity estimated from noisy positions is itself noisy, and a small velocity error grows into a large position error over a long gap. In the standard setting, every block moves at a speed of either 0 or 1 cell per step on each axis. The tracker uses this fact and snaps each axis of a track’s velocity to a speed of 0 or 1. During the context phase, it snaps an axis once its velocity estimate makes one of these speeds at least 99% probable, and it then holds that speed, reversing its sign at each bounce. When the prediction phase starts, it rounds every axis that has not yet snapped to the nearer of the two speeds. We found the two speeds from the velocities that the tracker itself estimated on 1,024 training videos of the standard setting, without true identities, and we chose the snapping thresholds on our tuning videos (App. A). Speed snapping is knowledge about the standard setting, in the same way that FloWM’s built-in velocities are. For the varied setting we refitted the allowed speeds without labels on 1,024 varied training videos, which gives 0, 1 and 2 cells per step (App. F.2).

## C.4 What the simulator gives the tracker

For every visible block in every context frame, the tracker receives the block’s true floor position with added random noise. The noise shifts the block’s bearing and its distance independently, each with a standard deviation of one image pixel, so the error grows with distance as it would for a real camera. The tracker also receives the uncertainty of each noisy position. It also receives the block’s true mask, and it uses the mean color inside the mask as the block’s appearance. It reflects tracks off the true wall positions. At scoring time, we must also know which track describes which block. We use the true identity of each detection for this purpose, and for each block we report the track that the block’s detections last matched. This readout is also an oracle step. FloWM’s probe receives similar help in another form, since it is trained with the true position of each block.

## C.5 How good can the tracker be?

Table 3: Out-of-view accuracy (%) of tracker variants on the standard test videos. The row marked “the tracker” is the tracker of the main text. The upper rows score all 85,602 out-of-view answers. The lower rows score only the 83,202 out-of-view answers for blocks sighted in at least two context frames. A restart predicts a block forward from the tracker’s estimate at its last sighting, with no further matching.

<table><tr><td>Tracker variant</td><td>Out of view</td></tr><tr><td>All blocks</td><td></td></tr><tr><td>Exact positions, no speed snapping</td><td>93.7</td></tr><tr><td>One-pixel noise, no speed snapping</td><td>47.1</td></tr><tr><td>One-pixel noise, speed snapping (the tracker) Exact positions, speed snapping</td><td>90.8</td></tr><tr><td></td><td>94.3</td></tr><tr><td>Blocks sighted at least twice†</td><td></td></tr><tr><td>The tracker</td><td>93.4</td></tr><tr><td>Restart from the tracker&#x27;s estimate at the last sighting</td><td>94.3</td></tr><tr><td>Restart with the true velocity</td><td>97.8</td></tr></table>

Because the tracker receives oracle inputs, its score is a ceiling for this tracker design and a reference point for a learned node memory. We first check that it is a strong reference. It keeps identities well, with an IDF1 score of 95.6% [32]. IDF1 measures how many detections a tracker assigns to the correct identity, balancing missed and wrong assignments. Table 3 shows how the tracker’s out-ofview accuracy depends on its inputs and on speed snapping.

Speed snapping carries most of the tracker’s strength under noise. With exact positions, speed snapping changes little (94.3% against 93.7%). Noise of a single pixel cuts this to 47.1%, because noisy positions give noisy speeds and the errors grow over long gaps. Speed snapping restores the score to 90.8%. These allowed speeds match the standard setting. For the varied setting we refitted them and ran the tracker once on the varied test videos (App. F.2).

The lower rows of Table 3 show where the remaining error comes from. For each block sighted at least twice, we restart the block from the tracker’s own estimate at its last sighting and predict it forward with no further matching. This restart scores 94.3%, slightly above the tracker’s 93.4% on the same answers. The difference is error that enters after the last sighting, for example when a detection of another block is matched to the track. Giving the restart the true velocity raises it to 97.8%, so velocity error at the last sighting costs about 3.5 points. We froze the tracker’s settings on the tuning set, but only after we had seen some test-set results.

## D Write-back

During the prediction phase no block is observed, whether it is in view or not. Yet the convolutional probe scores 8 points higher on blocks in view than on blocks out of view (Table 1). One explanation would be that blocks in view were seen more recently. To test this, we compare blocks in view and out of view that have the same number of steps since their last sighting. On the test videos, FloWM keeps an in-view advantage of 7 to 10 points in every bin from 5 steps on, and every confidence interval lies above zero<sup>†</sup>. The tracker has no such advantage. Weighted to the same mix of steps since the last sighting, its in-view score minus its out-of-view score is −0.1 points (CI −1.2 to 0.9), while FloWM’s is +7.5 points (CI 6.7 to 8.4).

The likely cause is FloWM’s write-back. At each predicted step, FloWM renders a frame and writes it back into the cells in view. When a block enters the camera’s view during the prediction phase, its probe accuracy rises from 84.8% to 92.4% on the tuning videos (read by heads trained on the other half of the tuning videos), although nothing new is observed<sup>†</sup>. On the test videos, read by the convolutional probe, it rises from 89.6% to 94.2% (4.6 points, CI 3.7 to 5.5), while the tracker stays at 90.5%. We also read the memory just before each write on the tuning videos, but the probe was trained on memory read after the write and scores lower before it even on blocks that stay in view (84.1% against 96.5%), so this read cannot say how much of the in-view advantage the write-back causes. During prediction the written frame is computed from the memory itself, so the write adds no new information about where blocks are. In-view scores therefore mix memory with the model’s own predictions, and we compare the two memories on blocks out of view.

## E Time since the last sighting

Figure 2 shows how accuracy on out-of-view blocks changes with the steps since the last sighting. FloWM falls from 96% at 2 to 4 steps to between 89% and 90% from 10 to 99 steps, and it ends at 82%. The tracker starts higher, at 97%, and falls further, to 72%. The ranking therefore flips. Hollow markers in Fig. 2 mark bins with fewer than 1,000 answers. At 10 to 19 steps, the tracker leads by 6 points (CI 4 to 8). At 100 to 139 steps, it trails by 10 points (CI 4 to 16)<sup>†</sup>.

This flip does not by itself show that FloWM’s memory fades more slowly. The late bins hold different kinds of blocks from the early bins, and two causes explain the tracker’s late deficit. We found both after seeing the results<sup>†</sup>.

The first cause is blocks sighted only once. Such a block gives the tracker a position but no velocity. Its track either stays where the block was seen or, for 13 of the 15 such blocks at 100 to 139 steps, is also updated by detections of other blocks, and in either case the tracker finds almost none of them. These blocks make up 14% of out-of-view answers at 100 to 139 steps but only 0.6% at 10 to 19 steps. The tracker finds 0.6% of them, while FloWM finds 30.2%. FloWM appears to infer a block’s direction from a single sighting. To check this, we took answers for blocks sighted only once in which the block’s last-seen cell and the cells it would reach in each of the four directions all differ.

Among FloWM’s answers that land on one of the four direction cells, 89% pick the true direction, where guessing would give 25% (tuning videos). We have not identified the cue that FloWM uses.

The second cause is error that enters a track during the context phase after the block’s last sighting, such as a match to another block’s detection. For blocks sighted at least twice, we restart each block from the tracker’s estimate at its last sighting, as in App. C. The restart beats the tracker by 6.7 points at 100 to 139 steps and by nothing at 10 to 19 steps. A long gap since the last sighting means that the context phase continued for many steps after it. During those steps, a detection of another block can be matched to the track and pull it away. Such errors do not show until the track is scored.

Splitting the gap by how often each block was sighted makes this precise. Blocks sighted once account for 4.1 of the 9.8 points (CI 0.9 to 7.9), and blocks sighted at least twice account for the other 5.7 (CI −0.2 to 11.7). For the latter, error that enters after the last sighting accounts for 5.8 points (CI 1.7 to 10.4), which is the 6.7-point loss against the restart weighted by these blocks’ share of the bin, and the restart itself scores 0.1 points above FloWM (CI −4.8 to 4.6). We made this split after we had seen the results<sup>†</sup>.

## F Varied setting

## F.1 FloWM on velocities outside its built-in set

The probe scores in this subsection are overall accuracy, over answers in view and out of view.   
Appendix F.2 gives out-of-view values by motion type (Table 6).

The mix layer could, in principle, let FloWM carry more velocities than its five copies. Since the mix can move content by one cell, a block moving two cells per step could be carried by a one-cell shift in a copy plus a one-cell move in the mix. Training on the varied setting tests whether FloWM learns to do this. We compare four models. The first is the released checkpoint, which was trained on the standard setting. The second is the released checkpoint fine-tuned on the varied setting for 20k steps. The third and fourth are trained from scratch for 150k steps with FloWM’s recipe, one on the varied setting and one on the standard setting. The from-scratch standard model checks how well our training run reproduces the released model.

We measure each model in two ways. The first is the MSE of predicted frames on 1,000 clips per setting, over 70 predicted frames, which needs no probe. The second is probe accuracy on the varied test videos, split by motion type into built-in, diagonal, and speed 2. For the released model, we read the varied videos with the convolutional probe trained on the standard setting, used unchanged. We also trained probe heads on the varied setting while the model generated new rollouts, rather than from saved memory states. We use such heads to read the fine-tuned and retrained models, whose memories the standard heads have never seen. Heads trained this way on the standard setting are scored on a separate set of 1,024 fresh standard videos, which include the 256 test videos.

Table 4: FloWM on the varied setting, as overall accuracy (%) over answers in view and out of view. The standard-video column is on 1,024 fresh standard videos, which include the 256 test videos, and every score in it comes from heads trained on the standard setting while the model generated rollouts. The varied-video columns are on the 1,024 varied test videos and use the heads named in the second column. MSE is on 1,000 clips with 70 predicted frames. Built-in means one cell per step along one axis. Heads marked <sup>∗</sup> were trained while the model generated rollouts, and the other heads were trained on saved memory states. Probe scores carry no interval.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Probe trained on</td><td rowspan="2">Standard videos</td><td colspan="4">Varied videos</td><td colspan="2">MSE</td></tr><tr><td>All</td><td>Built-in</td><td>Diagonal</td><td>Speed 2</td><td>Standard</td><td>Varied</td></tr><tr><td>Released</td><td>Standard</td><td>86.9</td><td>16.3</td><td>75.8</td><td>3.6</td><td>4.3</td><td>0.000649</td><td>0.0148</td></tr><tr><td>Released</td><td>Varied*</td><td></td><td>6.2</td><td>22.3</td><td>2.2</td><td>3.1</td><td></td><td></td></tr><tr><td>Fine-tuned on varied, 20k steps</td><td>Varied*</td><td>84.5</td><td>20.1</td><td>46.8</td><td>13.4</td><td>15.0</td><td>0.00135</td><td>0.00747</td></tr><tr><td>From scratch on standard, 150k</td><td>Standard*</td><td>84.8</td><td></td><td></td><td></td><td></td><td>0.000824</td><td>0.0158</td></tr><tr><td>From scratch on varied, 150k</td><td>Varied*</td><td>30.5</td><td>13.1</td><td>20.7</td><td>10.5</td><td>11.8</td><td>0.00592</td><td>0.00790</td></tr></table>

For the released model, the convolutional probe trained on the standard setting, used unchanged on varied videos, finds 75.8% of blocks that move at a built-in velocity (Table 4). It finds only 3.6% of diagonal blocks and 4.3% of speed-2 blocks. On built-in-velocity blocks it scores 14 points below the 90.2% it scores on the standard test videos (CI 13 to 16). This drop may come from the memory itself, or from memory states that the probe never saw in training. Probe heads trained on the varied setting fit their own training videos at 93% but find only 6.2% of blocks in fresh varied videos, so we rely on the probe trained on the standard setting for the released model. The MSE, which needs no probe, is 23 times higher on the varied setting than on the standard setting.

Fine-tuning on the varied setting for 20k steps halves the MSE on the varied setting. It also doubles the MSE on the standard setting. Its probe finds only 13% to 15% of diagonal and speed-2 blocks.

Training from scratch on the varied setting for 150k steps gives a varied-setting MSE only 1.33 times its own standard-setting MSE (CI 1.28 to 1.39). Its probe, however, finds only 10.5% to 11.8% of diagonal and speed-2 blocks, against 20.7% of built-in-velocity blocks. This model is also weak overall. Its MSE on the varied setting is 12.2 times the released model’s MSE on the standard setting. On standard clips, its MSE is 9.1 times the released model’s, and its probe finds 30.5% of blocks, against 86.9% for the released model read by the same kind of probe. Our from-scratch run on the standard setting missed the released MSE by more than 10% (0.000820 against 0.000603 on the release’s validation clips), so our training recipe may be weaker than the original. With one seed per run, these results neither show nor rule out that the mix layer can extend the built-in velocities.

Taken together, FloWM’s memory carries blocks whose motion matches its five velocities and largely loses the rest. Covering the varied setting’s 24 velocities with built-in shifts would take 25 copies of the grid, one of them still, or 7.0M numbers instead of 1.4M [4, their Apps. F and G.1]. Training that model is out of scope for this work.

## F.2 The tracker on the varied test videos

We ran the tracker once on the 1,024 varied test videos, the videos on which FloWM is scored above. Before any test video was read, we froze the code and settings and registered the run and its headline number. The headline is out-of-view accuracy on diagonal and speed-2 blocks for the tracker with allowed speeds 0, 1 and 2. Each of five configurations ran once, and Table 5 reports every number as it came out. Each configuration scores 430,080 answers, of which 342,917 are out of view, and 19 blocks were never sighted in context. In-view and overall accuracy are within about half a point of the out-of-view values.

With allowed speeds 0, 1 and 2, the tracker finds 93.9% of out-of-view diagonal and speed-2 blocks. On the same answers, FloWM’s released model, read by the convolutional probe trained on the standard setting, finds 4.1%. The paired difference is 89.8 points (CI 89.1 to 90.5). Table 6 gives FloWM’s out-of-view accuracy by motion type for all four rows. The other three rows saved no per-answer outputs, so their values are unpaired point estimates from accuracies logged per motion type. Pooled over diagonal and speed-2 blocks they score 2.9%, 12.6% and 9.1%.

The tracker of the main text, with allowed speeds 0 and 1, keeps diagonal blocks (93.1%) and loses speed-2 blocks (4.9%). Per-axis speeds of 0 and 1 can express one cell per step on each axis, but not two. On speed-2 blocks this tracker does worse than the tracker with no speed snapping (59.5%), so snapping to a wrong speed hurts more than not snapping. With exact positions, speed snapping makes no difference to the score over all blocks (97.0% with and without it).

Three caveats apply to the headline number. The tracker receives oracle detections, and it knows the true walls. Its allowed speeds were fitted without labels on 1,024 varied training videos, but its other settings did not pass our own selection rule. Before tuning, we fixed that a setting qualifies only if, with one-pixel noise, IDF1 reaches 95% with at most 1.1 nodes per block, and if, with exact positions, it does at least as well as no speed snapping. On 256 varied validation videos no setting passed (best IDF1 86.5%). We therefore kept the settings chosen for the standard setting, after the varied probe-training videos had already been run with these allowed speeds.

Table 5: The tracker on the 1,024 varied test videos: out-of-view accuracy (%) by motion type, and IDF1 (%). Each configuration ran once, after the code and settings were frozen and the run was registered.
<table><tr><td>Positions</td><td>Allowed speeds</td><td>Built-in</td><td></td><td></td><td></td><td>Diagonal and</td><td>IDF1</td></tr><tr><td>One-pixel noise</td><td>0,1 and 2</td><td>All 93.4</td><td>91.0</td><td>Diagonal 94.1</td><td>Speed 2 93.9</td><td>speed 2 93.9</td><td>93.6</td></tr><tr><td></td><td>0 and 1 (the tracker)</td><td>33.3</td><td>89.9</td><td>93.1</td><td>4.9</td><td>21.7</td><td>89.4</td></tr><tr><td>One-pixel noise One-pixel noise</td><td>no speed snapping</td><td>56.5</td><td>47.1</td><td>54.2</td><td>59.5</td><td>58.5</td><td>88.3</td></tr><tr><td>Exact</td><td>no speed snapping</td><td>97.0</td><td>95.1</td><td>97.1</td><td>97.5</td><td>97.4</td><td>94.2</td></tr><tr><td>Exact</td><td>0, 1 and 2</td><td>97.0</td><td>95.4</td><td>97.2</td><td>97.3</td><td>97.3</td><td>95.6</td></tr></table>

Table 6: FloWM on the 1,024 varied test videos: out-of-view accuracy (%) by motion type, on the same answers as Table 5. The first row comes from per-answer outputs. The other rows come from accuracies logged per motion type, and their last column assumes the first row’s answer counts.
<table><tr><td>Model</td><td>Probe trained on Built-in</td><td></td><td>Diagonal</td><td>Speed 2</td><td>Diagonal and speed 2</td></tr><tr><td>Released</td><td>Standard</td><td>72.7</td><td>3.5</td><td>4.2</td><td>4.1</td></tr><tr><td>Released</td><td>Varied</td><td>18.6</td><td>2.1</td><td>3.1</td><td>2.9</td></tr><tr><td>Fine-tuned on varied, 20k steps</td><td>Varied</td><td>39.1</td><td>11.0</td><td>13.0</td><td>12.6</td></tr><tr><td>From scratch on varied, 150k</td><td>Varied</td><td>16.8</td><td>8.2</td><td>9.4</td><td>9.1</td></tr></table>

## G Learned bounce

The tracker’s bounce is an exact reflection off the true walls, an oracle that a learned node memory will not have. We therefore tried to learn the bounce instead, while still telling the tracker where the walls are. This experiment used the tracker with the exact reflection replaced by a learned gate, described below, and it ran on training videos of the standard setting that we generated rather than on our evaluation videos. These videos hold six to ten blocks in the same six colors, so most repeat a color, and color alone cannot tell two blocks apart. Before training, the gate leaves every velocity unchanged, so each track moves at constant velocity and unseen tracks pass through walls. On the tuning videos, this untrained variant keeps identities far worse than the tracker, with an IDF1 of 67.9% against 95.2%.

We added a gate that acts when a track is near a wall. The gate multiplies the track’s velocity on that axis by a learned factor m. A factor of m = 1 continues, $m = 0$ stops, and $m = - 1$ reverses. We trained m to reduce the track’s position error each time a block is seen again. We measure matching quality as the share of the tracker’s matches that join a detection to a track started by another block. We changed the training objective after seeing early results, so we treat thi experiment as a diagnostic<sup>†</sup>.

A tracker’s motion model does two jobs. It predicts where an unseen block has gone, and it decides which track a new detection joins. Training it only to predict can therefore harm the second job. With no bounce model, the tracker matches 17.5% of detections to the wrong block. With exact reflection off the true walls, it matches 16.3%. With the learned bounce, it matches between 19% and 26%, across three learning rates and training steps 100 to 500. Learning the bounce made matching worse than having no bounce model at all.

Two errors explain this result, which we examine at a learning rate of $1 0 ^ { - 3 }$ and step 500. First, tracks whose speed was fixed by speed snapping learn to stop at the wall rather than bounce. the median learned factor is $m = - 0 . 0 0 6$ on the 6,375 steps where a block reaches the wall and keeps its velocity, and none of the 3,533 separate steps where it reverses gets a negative factor (median +0.545). When a track cannot tell whether its block will continue or reverse, stopping halfway lowers the average squared position error. However, it leaves the track behind the block, where it can capture another block’s detection. Second, tracks that have drifted, and whose speed is not fixed, reverse on 89% of the steps where the block in fact keeps going. Together, these errors raise the number of wrong-block matches from 3,928 to 5,738.

## H Cost

Given oracle positions and masks, the tracker updates its memory with about $1 0 ^ { 5 }$ to $1 0 ^ { 6 }$ arithmetic operations per context step, by our hand count, and its state takes 14 KB. FloWM’s write of one frame takes $1 . 5 \times 1 0 ^ { 1 0 }$ operations over a 5.6 MB grid. Of this cost, 96% grows with the 885 memory entries it updates, which are the 177 cells in view times the five copies. This comparison flatters the tracker, because FloWM also perceives, and a deployed tracker would need a detector and a position estimator whose cost would dominate. The tracker’s state also has a practical benefit. It stores each block’s position and velocity as numbers, so no trained probe is needed to read them once tracks are linked to objects.

## I Intervals and post-hoc analyses

Table 7: The main-text comparisons that have an interval, in points of accuracy. Each difference is the first quantity minus the second, so a negative value means the first is lower. Intervals are 95% bootstrap intervals.
<table><tr><td>Comparison</td><td>Difference</td><td>95% interval</td></tr><tr><td>Standard test videos, out of view: the tracker minus FloWM read by the convolutional probe</td><td>+2.1</td><td>0.5 to 3.7</td></tr><tr><td>FloWM, in view minus out of view, matched on steps since the last sighting</td><td>+7.5</td><td>6.7 to 8.4</td></tr><tr><td>The tracker, in view minus out of view, same matching</td><td>-0.1</td><td>-1.2 to 0.9</td></tr><tr><td>FloWM, a block after it enters view during prediction minus before, test videos</td><td>+4.6</td><td>3.7 to 5.5</td></tr><tr><td>The tracker minus FloWM, out of view, 10 to 19 steps since the last sighting</td><td>+6</td><td>4 to 8</td></tr><tr><td>The tracker minus FloWM, out of view, 100 to 139 steps since the last sighting</td><td>-10</td><td>-16 to -4</td></tr><tr><td>Varied test videos, out-of-view diagonal and speed-2 blocks: the tracker with allowed speeds 0, 1 and 2 minus FloWM&#x27;s released model</td><td>+89.8</td><td>89.1 to 90.5</td></tr></table>

Table 7 gives the intervals that exist for comparisons in the main text. The intervals for the two causes of the tracker’s late deficit are in App. E. The other comparisons, such as the two probes, the tracker without speed snapping and the learned bounce, have none.

We made the choices marked † after we had seen results. The first is the split by number of sightings and what builds on it: the two causes of the tracker’s late deficit, the split of that deficit by number of sightings (App. E) and the restart rows of Table 3. The second is the training objective of the learned bounce, which we changed after seeing early results (App. G). The third is the comparison of in-view and out-of-view accuracy at equal steps since the last sighting, which we proposed after seeing FloWM’s in-view lead, together with the analysis of blocks entering view that followed it (App. D). The fourth is the pair of bins quoted for the flip, 10 to 19 and 100 to 139 steps, since we have no record that we chose this pair in advance. The bin edges themselves were fixed before any result by steps since the last sighting.

Four further disclosures apply. No setting of the tracker with allowed speeds 0, 1 and 2 passed our pre-set selection rule, so we kept the standard setting’s settings after seeing results on varied videos outside the test set, and its run on the varied test videos was registered in advance (App. F.2). We froze the tracker’s settings for the standard setting on the tuning set, but only after we had seen some test-set results. A few numbers come from tuning or training videos and not from test videos: the read before the write, FloWM’s choice of direction after a single sighting, the first of the two measurements of a block entering view, and the learned bounce. The FloWM values for the finetuned and retrained models in the lower half of Table 1 are pooled from logged aggregates and carry no interval.