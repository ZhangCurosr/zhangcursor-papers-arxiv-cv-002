# Towards Surgical World-Action Modeling: A Preliminary Joint Visual-Trajectory Forecasting for Surgical Motion Planning

Weiliang Huang<sup>1</sup>, Huanrong Liu<sup>1,3</sup>, Bob Zhang<sup>1,†</sup>, Qi Dou<sup>4</sup>, Zhen Chen<sup>5</sup>, Yun Gu<sup>6,7</sup>, Guy Rosman<sup>8</sup>, and Qingbiao Li<sup>1,2,3,†</sup>

<sup>1</sup> Faculty of Information Science and Computing, University of Macau, Macau, China <sup>2</sup> Faculty of Engineering, University of Macau, Macau, China

<sup>3</sup> University of Macau Advanced Research Institute in Hengqin, Zhuhai, China

4 Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China

5 Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University, Hong Kong, China

Shanghai Key Laboratory of Flexible Medical Robotics, Tongren Hospital, Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai, China

<sup>7</sup> School of Automation and Intelligent Sensing, Shanghai Jiao Tong University, Shanghai, China

<sup>8</sup> School of Medicine, Duke University, Durham, North Carolina, USA qingbiaoli@um.edu.mo

<sup>†</sup> Corresponding author

Abstract. Reliable surgical planning requires models to anticipate not only how instruments will move, but also how the operative visual state will evolve together with such motion. Existing approaches typically treat future scene generation and instrument trajectory prediction as two separate tasks. Scene-only models cannot directly evaluate the accuracy of future instrument motion at the trajectory level, while trajectoryonly models fail to capture the visual consequences of instrument movement, leaving the consistency between predicted trajectories and future scene evolution unaddressed. Jointly forecasting both provides a more complete account of surgical action–scene dynamics by enabling explicit trajectory-level evaluation while simultaneously modeling the corresponding visual evolution. To bridge this gap, we present a preliminary joint visual-trajectory world-action model that simultaneously forecasts future visual states and instrument trajectories from historical surgical observations. Specifically, we encode historical video frames and tool trajectories into latent representations, which are processed by a temporal-spatial encoder and subsequently decoded through separate visual-state and trajectory prediction heads. Based on this preliminary architecture, a chunked autoregressive rollout is repeatedly applied to predict fifteen future steps. The chunked strategy consistently outperforms direct one-shot prediction across all evaluated horizons, improving first-segment PSNR from 18.86 to 23.11 dB and reducing ADE from 45.77 to 22.22 pixels. These results demonstrate the initial feasibility of joint visual-motion forecasting. However, we observe progressive visual degra-

dation and accumulated trajectory errors over longer prediction horizons, which remain important challenges for future surgical world-action modeling.

Keywords: Surgical world model · Surgical motion planning · Future visual-state prediction · Instrument trajectory prediction

## 1 Introduction

Reliable surgical planning [1, 15] requires more than estimating where an instrument will move next. Surgical actions continuously reshape the operative field through instrument–tissue interactions, tissue deformation, occlusion, bleeding, and changes in camera visibility. Consequently, a predicted trajectory may be geometrically close to an expert demonstration while still leading to an implausible or unsafe future surgical state. A planning model should therefore anticipate not only the future location of the instrument, but also how the local scene will change. However, current surgical trajectory prediction and future visual state prediction have largely developed as separate research directions.

Trajectory methods [16, 7, 12] typically forecast two-dimensional coordinates, robotic states, or motion commands without explicitly representing their visual consequences. Conversely, surgical video generation methods [3–5] model future appearance and scene dynamics but often provide limited evidence that the generated motion is geometrically accurate. This separation limits the ability of existing methods to jointly reason about instrument motion and the corresponding evolution of the operative environment, which is a core requirement for surgical world modeling and closed-loop planning.

Recent surgical world models [10, 4, 14, 8] have demonstrated promising capabilities in future video generation. Their evaluation, however, remains predominantly focused on generation-oriented criteria such as visual realism, temporal consistency, Fréchet Video Distance [17], PSNR, and SSIM. Although these metrics are useful for assessing appearance similarity, visually plausible predictions do not necessarily imply accurate instrument motion or stable long-horizon dynamics. A generated sequence may preserve the overall appearance of the surgical scene while exhibiting endpoint deviation, incorrect local motion, or progressively increasing drift during autoregressive rollout. These limitations motivate a world-action formulation that extends conventional future-state prediction by explicitly coupling world evolution with action-relevant motion. In our surgical setting, the future visual representation describes how the operative scene evolves, while the instrument trajectory provides an explicit representation of tool motion. Jointly predicting these complementary outputs enables visual-state evolution and instrument motion to be modeled and evaluated within a unified predictive framework, providing a more complete characterization of surgical dynamics.

In this work, we preliminarily investigate whether a predictive model can jointly forecast future visual representations and instrument trajectories from the same historical surgical information. Our model first encodes the observed surgical frames and historical tool trajectories into latent representations. These representations are then processed by a temporal-spatial encoder and fed into separate heads for future visual-state prediction and two-dimensional instrument trajectory prediction. To support long-horizon prediction, the model generates three future steps at each stage and recursively feeds the predicted visual and motion representations back into the model, producing a fifteen-step rollout through five successive prediction chunks. Scheduled sampling [2] is further employed during training to gradually expose the model to its own predictions and partially reduce the discrepancy between training and autoregressive inference. We evaluate the proposed baseline on SurgWMBench [11], a benchmark for surgical motion planning using endoscopic video sequences and annotated two-dimensional instrument trajectories. Our contributions are threefold:

1. We introduce a joint predictive model that simultaneously forecasts future visual latent representations and two-dimensional surgical instrument trajectories.

2. We develop a chunked autoregressive strategy that decomposes the fifteenstep prediction horizon into repeated three-step rollouts, while scheduled sampling partially reduces the train–inference discrepancy and improves long-horizon visual-motion stability.

3. We provide a preliminary evaluation on SurgWMBench, demonstrating the feasibility of joint visual-trajectory forecasting while highlighting remaining challenges in long-horizon visual fidelity and trajectory stability.

## 2 Method

## 2.1 Problem Formulation

Given a sequence of K observed surgical frames,

$$
\mathcal { T } _ { 1 : K } = \left\{ I _ { 1 } , \ldots , I _ { K } \right\} ,\tag{1}
$$

and the corresponding historical two-dimensional instrument trajectory,

$$
\mathcal { P } _ { 1 : K } = \left\{ \mathbf { p } _ { 1 } , \ldots , \mathbf { p } _ { K } \right\} , \qquad \mathbf { p } _ { t } \in \mathbb { R } ^ { 2 } ,\tag{2}
$$

our objective is to jointly predict future visual states and instrument trajectories over a horizon of H steps.

Instead of directly predicting future frames in the pixel space, we represent each future visual state using a latent feature representation. The joint prediction task is formulated as

$$
\left( \hat { \mathcal { Z } } _ { K + 1 : K + H } , \hat { \mathcal { P } } _ { K + 1 : K + H } \right) = F _ { \theta } \left( \mathcal { T } _ { 1 : K } , \mathcal { P } _ { 1 : K } \right) ,\tag{3}
$$

where $\hat { \mathcal { Z } } _ { K + 1 : K + H }$ denotes the predicted future visual representations and $\hat { \mathcal { P } } _ { K + 1 : K + H }$ denotes the predicted instrument trajectory. A frozen visual decoder D maps

![](images/8a0336428539dd61b58361c6576ea81599096269bf39c35e2e75dd2ce0a29553.jpg)  
Fig. 1. Overview of the proposed method. (a) Causal history encoding: five observed RGB frames are processed by the frozen SurgMotion ViT-L encoder to obtain causal token grids. (b) Closed-loop chunked future prediction: the predictor jointly forecasts future token grids and normalized 2-D tool positions in 3-step chunks; each predicted chunk updates the 5-step history for the next rollout, producing a total 15-step forecast.

each predicted latent representation back to an RGB frame for visualization and image-level evaluation:

$$
\hat { I } _ { t } = D \left( \hat { Z } _ { t } \right) .\tag{4}
$$

In this preliminary formulation, the future instrument trajectory is treated as an action-oriented planning representation rather than an explicit low-level robotic control command.

## 2.2 Joint Visual-Motion Representation

The historical surgical frames are first processed by a frozen SurgMotion [19] encoder as shown in Figure 1. We extract intermediate-layer visual tokens that retain local appearance and spatial details, together with final-layer visual tokens that capture higher-level semantic and motion context. The two visual feature streams are independently projected into a shared hidden dimension. For each historical step, the corresponding two-dimensional trajectory coordinate $\mathbf { p } _ { t }$ is embedded using a multilayer perceptron. The resulting trajectory embedding is spatially broadcast and fused with the visual representations:

$$
X _ { t } = \phi _ { \mathrm { m i d } } \left( Z _ { t } ^ { \mathrm { m i d } } \right) + \phi _ { \mathrm { f n a l } } \left( Z _ { t } ^ { \mathrm { f i n a l } } \right) + \phi _ { \mathrm { t r a j } } \left( \mathbf { p } _ { t } \right) ,\tag{5}
$$

where $\phi _ { \mathrm { m i d } } , \phi _ { \mathrm { f i n a l } }$ , and $\phi _ { \mathrm { t r a j } }$ denote the corresponding projection modules. This fusion constructs a shared representation containing both operative-scene information and historical instrument motion.

A temporal-spatial encoder is subsequently employed to model the evolution of the fused visual-motion representations. Temporal attention captures the dependencies among historical states, while spatial processing preserves the local structure of the operative field. The encoded feature corresponding to the latest historical state is used as the shared context representation C. Learned future-step embeddings are then combined with $C$ to generate step-specific representations for future prediction.

Two separate prediction heads operate on the shared context. The visualstate prediction head estimates a residual change with respect to the latest observed visual representation:

$$
\begin{array} { r } { \hat { Z } _ { K + j } = Z _ { K } + \varDelta Z _ { K + j } , \qquad j = 1 , \ldots , c , } \end{array}\tag{6}
$$

where $c = 3$ denotes the prediction chunk length. Predicting residual visual changes encourages the model to preserve the observed scene structure while modeling future state transitions.

For trajectory prediction, the shared context is spatially pooled and combined with the latest instrument position and motion information. We first construct a constant-velocity estimate:

$$
{ \tilde { \bf p } } _ { K + j } = { \bf p } _ { K } + j \left( { \bf p } _ { K } - { \bf p } _ { K - 1 } \right) .\tag{7}
$$

The trajectory prediction head then estimates a residual correction $\mathbf { r } _ { K + j } \colon$

$$
\hat { \mathbf { p } } _ { K + j } = \tilde { \mathbf { p } } _ { K + j } + \mathbf { r } _ { K + j } .\tag{8}
$$

This formulation introduces a simple motion prior while allowing the learned visual-motion representation to correct the trajectory direction, magnitude, and endpoint.

## 2.3 Chunked Autoregressive Rollout

Directly predicting all fifteen future steps in a single forward may produce temporally averaged visual states or inaccurate long-range trajectories. We therefore decompose the full prediction horizon into several short prediction chunks.

Given a five-step visual-motion history, the model predicts the next $c = 3$ visual representations and trajectory points:

$$
\left( \hat { \mathcal { Z } } _ { t + 1 : t + 3 } , \hat { \mathcal { P } } _ { t + 1 : t + 3 } \right) = F _ { \boldsymbol { \theta } } \left( \mathcal { Z } _ { t - 4 : t } , \mathcal { P } _ { t - 4 : t } \right) .\tag{9}
$$

After each prediction stage, the oldest three historical states are removed, and the three predicted visual representations and trajectory points are appended to construct a new five-step history window. The updated history is then used as the input to the next prediction stage.

During training, scheduled sampling is employed to reduce the discrepancy between teacher-forced training and fully autoregressive inference. At each rollout stage, the history input is constructed using either a ground-truth state or a model prediction:

$$
\bar { x } _ { t } = m _ { t } \hat { x } _ { t } + \left( 1 - m _ { t } \right) x _ { t } , \qquad m _ { t } \sim \mathrm { B e r n o u l l i } \left( \rho _ { e } \right) ,\tag{10}
$$

where $x _ { t }$ and $\hat { x } _ { t }$ denote the ground truth and predicted states, respectively. The variable $\rho _ { e }$ denotes the probability of selecting a predicted state at training epoch $e .$ This probability gradually increases during training, exposing the model to its own prediction errors while maintaining stable supervision during the early training stages. During inference, all prediction chunks are generated entirely from the model’s previous predictions.

## 3 Experiments

## 3.1 Dataset and Evaluation Protocol

We conduct experiments on SurgWMBench [11], a vision-based surgical motion planning benchmark constructed from SAR-RARP50 [13], which contains real clinical videos of Robot-Assisted Radical Prostatectomy (RARP) performed using the da Vinci surgical system. SurgWMBench focuses on suturing motions and extracts efective needle insertion and extraction actions as individual motion segments. The benchmark contains 1,637 valid segments, corresponding to 32,740 annotated surgical frames. For each frame, one task-relevant two-dimensional instrument anchor point is manually annotated, yielding a 20-point trajectory for each segment. Following the short-horizon prediction protocol of SurgWM-Bench [11], we use the first five frames and their corresponding trajectory points as historical observations and predict the remaining fifteen future visual states and trajectory points. Trajectory errors are evaluated in the original image pixel space.

We compare two prediction settings under the same strict history-only protocol. In the direct one-shot setting, the model predicts all fifteen future visual representations and trajectory points in a single forward pass. In the chunked rollout setting, the model predicts three future steps at each stage and recursively uses the predicted visual-motion states to construct the input for the next stage. This process is repeated five times to obtain the complete fifteen-step prediction.

For visual-state evaluation, the predicted latent representations are mapped to RGB frames using the frozen decoder. We report Peak Signal-to-Noise Ratio (PSNR) [9], Structural Similarity Index Measure (SSIM) [18], and Learned Perceptual Image Patch Similarity (LPIPS) [20]. Higher PSNR and SSIM indicate better visual reconstruction, whereas lower LPIPS indicates stronger perceptual similarity. For trajectory evaluation, we report Average Displacement Error (ADE) and Final Displacement Error (FDE) [6] in the original image pixel space. ADE measures the average point-wise trajectory error, while FDE measures the displacement error at the final point of that segment.

All metrics are reported separately over five non-overlapping future segments: $t + 1 { : } 3 , t + 4 { : } 6 , t + 7 { : } 9 , t + 1 0 { : } 1 2$ , and t + 13:15, where t denotes the current frame and all prediction ofsets are specified in frames. This segment-wise evaluation allows us to examine how visual quality and trajectory accuracy change as the prediction horizon increases.

## 3.2 Implementation Details

We train with AdamW $( 7 \times 1 0 ^ { - 5 }$ learning rate and $1 0 ^ { - 4 }$ weight decay), a fiveepoch warm-up, cosine decay to 5% of the initial rate, batch size one, and early stopping with patience 14 (maximum 70 epochs). The loss combines token MSE/cosine, RGB, Sobel-edge, temporal-diference, trajectory-position, velocity, and endpoint losses with weights 1, 0.05, 0.25, 0.10, 0.10, 5, 2, and 2, respectively; step weights increase from 1.0 to 1.8 over the future horizon. Scheduled sampling starts at epoch 8 and reaches a 0.75 predicted-state probability at epoch 45; validation and test are fully closed loop. All training experiments were performed on a single NVIDIA RTX 5090 GPU.

## 3.3 Quantitative Results

Table 1 compares the direct one-shot prediction and chunked autoregressive rollout settings. The chunked rollout consistently outperforms direct prediction across all five future segments and all evaluated metrics.

The largest performance diference is observed in the first prediction segment. Compared with direct one-shot prediction, chunked rollout improves PSNR from 18.859 to 23.105 dB and SSIM from 0.4713 to 0.7971. It also reduces LPIPS by 70.5%, from 0.6045 to 0.1783. For trajectory prediction, the ADE and FDE are reduced from 45.77 and 49.47 pixels to 22.22 and 31.64 pixels, respectively.

The advantage of chunked prediction remains observable at longer horizons. For the final t+13:15 segment, chunked rollout achieves a PSNR of 17.982 dB and an SSIM of 0.6343, compared with 16.872 dB and 0.4264 for direct prediction. Its LPIPS is also substantially lower, at 0.3047 compared with 0.6244. In terms of motion prediction, chunked rollout reduces ADE from 195.27 to 158.52 pixels and FDE from 208.04 to 172.12 pixels, corresponding to relative reductions of approximately 18.8% and 17.3%, respectively.

These results suggest that decomposing a long prediction horizon into shorter local transitions facilitates both visual-state modeling and trajectory prediction. Nevertheless, both visual quality and trajectory accuracy gradually deteriorate as the rollout horizon increases. Therefore, the chunked strategy reduces, but does not eliminate, long-horizon visual degradation and trajectory error accumulation.

Table 1. Segment-wise comparison between chunked autoregressive rollout and direct one-shot prediction. Higher values are better for PSNR and SSIM, while lower values are better for LPIPS, ADE, and FDE. Trajectory errors are measured in pixels.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Segment</td><td colspan="3">Visual Metrics</td><td colspan="2">Trajectory Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>ADE↓</td><td>FDE↓</td></tr><tr><td rowspan="5">Chunked rollout</td><td>t + 1:3</td><td>23.105</td><td>0.7971</td><td>0.1783</td><td>22.22</td><td>31.64</td></tr><tr><td>t + 4:6</td><td>20.395</td><td>0.7187</td><td>0.2380</td><td>52.78</td><td>62.64</td></tr><tr><td>t + 7:9</td><td>19.254</td><td>0.6799</td><td>0.2700</td><td>84.13</td><td>94.27</td></tr><tr><td>t + 10:12</td><td>18.495</td><td>0.6527</td><td>0.2906</td><td>119.38</td><td>133.06</td></tr><tr><td>t + 13:15</td><td>17.982</td><td>0.6343</td><td>0.3047</td><td>158.52</td><td>172.12</td></tr><tr><td rowspan="5">Direct one-shot prediction</td><td>t + 1:3</td><td>18.859</td><td>0.4713</td><td>0.6045</td><td>45.77</td><td>49.47</td></tr><tr><td>t + 4:6</td><td>17.963</td><td>0.4482</td><td>0.6138</td><td>72.58</td><td>85.47</td></tr><tr><td>t + 7:9</td><td>17.489</td><td>0.4385</td><td>0.6182</td><td>119.01</td><td>132.94</td></tr><tr><td>t + 10:12</td><td>17.132</td><td>0.4314</td><td>0.6221</td><td>156.11</td><td>169.57</td></tr><tr><td>t + 13:15</td><td>16.872</td><td>0.4264</td><td>0.6244</td><td>195.27</td><td>208.04</td></tr></table>

## 3.4 Qualitative Results

Figure 2 presents qualitative comparisons between direct one-shot prediction and chunked autoregressive rollout. We visualize the predictions at t + 3, t + 6, t+9, t+12, and t+15. Ground-truth future frames and trajectories are included as references. More qualitative results are available in the anonymized repository https://github.com/anonymous-authors-code/MICCAI2026\_Workshop\_ EndoLINA\_TSWAM\_Qualitative\_Results.git.

Consistent with the quantitative results, chunked rollout generally preserves the operative-scene structure more efectively and produces instrument trajectories that remain closer to the ground truth, particularly at early and intermediate prediction horizons. In comparison, direct one-shot prediction exhibits earlier visual degradation and larger trajectory deviations as the prediction horizon increases. At t + 12 and t + 15, however, both settings show reduced visual fidelity and noticeable endpoint errors. These examples further illustrate that short-chunk prediction improves rollout stability, while accurate long-horizon joint visual-motion forecasting remains challenging.

## 4 Conclusion

We present a preliminary surgical world-action model for joint future visualstate and instrument-trajectory prediction. A chunked 3→3 autoregressive rollout improves visual quality and trajectory accuracy over one-shot prediction on SurgWMBench, enabling 15-step forecasts. Performance still degrades over long horizons due to accumulated errors; future work will explore action conditioning, uncertainty modeling, and stronger visual-motion consistency.

![](images/950fbc604977de533b46aaca277f0ac23969f230358eef2c5bf248c17e2494c2.jpg)

![](images/93a0609c607f29c9eb8560ad6313427d53ebf3fa57d78656684df7ce257e56de.jpg)

![](images/58ec9bfaa4ad67c170c0a93fcff047a35e731e6cbc92a74306d8af060fe2959e.jpg)

![](images/17171b9811dc712dbf365b3cae04f4a718c12122abc8afb94e1fba8b3c5450c2.jpg)

![](images/50a288d90a65d8ba1e829aaf5e82aaf8482e619a9f23b07d103a6cbbc46cf478.jpg)  
Fig. 2. Qualitative comparison of direct one-shot prediction and chunked autoregressive rollout. Columns show the future states at t + 3, t + 6, t + 9, t + 12, and t + 15. Rows show the (a) direct one-shot prediction, (b) chunked autoregressive rollout and (c) ground truth. Instrument trajectories are overlaid on the corresponding visual states. For visualizaton purpose, we cropped the region of interest for comparision.

Acknowledgments. This work was supported by the University of Macau under Grants SRG2024-00056-FST, 0078/2024/RIB2, and FST/SP01/2024, the Dr. Stanley Ho Medical Development Foundation under Grant SHMDF-AI/2026 8 002 and the Science and Technology Development Fund of Macao S.A.R (FDCT) 0028/2023/RIA1.

Disclosure of Interests. The authors have no competing interests.

## References

1. Attanasio, A., Scaglioni, B., De Momi, E., Fiorini, P., Valdastri, P.: Autonomy in surgical robotics. Annual Review of Control, Robotics, and Autonomous Systems 4(1), 651–679 (2021)

2. Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N.: Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems 28 (2015)

3. Biagini, D., Navab, N., Farshad, A.: Hierasurg: Hierarchy-aware difusion model for surgical video generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 310–319. Springer (2025)

4. Chen, T., Yang, S., Wang, J., Bai, L., Ren, H., Zhou, L.: Surgsora: Object-aware difusion model for controllable surgical video generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 521–531. Springer (2025)

5. Cho, J., Schmidgall, S., Zakka, C., Mathur, M., Kaur, D., Shad, R., Hiesinger, W.: Surgen: Text-guided difusion model for surgical video generation. arXiv preprint arXiv:2408.14028 (2024)

6. Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi, A.: Social gan: Socially acceptable trajectories with generative adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2255–2264 (2018)

7. Hansen, P., Kim, J.W.B., Goldenberg, A., Chen, J.T., Li, Y.A., Deguet, A., et al.: Imitatecholec: A multimodal dataset for long-horizon imitation learning in robotic cholecystectomy. Scientific Data (2026)

8. He, Y., Guo, P., Xu, M., Li, Z., Myronenko, A., Imans, D., Liu, B., Yang, D., Gu, M., Ji, Y., et al.: Surgworld: Learning surgical robot policies from videos via world modeling. arXiv preprint arXiv:2512.23162 (2025)

9. Huynh-Thu, Q., Ghanbari, M.: Scope of validity of psnr in image/video quality assessment. Electronics letters 44(13), 800–801 (2008)

10. Koju, S., Bastola, S., Shrestha, P., Amgain, S., Shrestha, Y.R., Poudel, R.P.K., Bhattarai, B.: Surgical vision world model. In: MICCAI Workshop / Springer Proceedings (2025)

11. Liu, H., Huang, W., Zhang, B., Cai, W., Tian, C., Li, Q.: Surgwmbench: A visionbased benchmark for world-modeling surgical instrument motion planning. arXiv preprint arXiv:2608.08070 (2026)

12. Liu, H., Tian, C., Jia, T., Zhou, T., Liu, Q., Gao, Y., Ban, Y., Gu, Y., Rosman, G., Ma, X., et al.: Sutureagent: Learning surgical trajectories via goal-conditioned ofline rl in pixel space. arXiv preprint arXiv:2603.26720 (2026)

13. Psychogyios, D., Colleoni, E., Van Amsterdam, B., Li, C.Y., Huang, S.Y., Li, Y., Jia, F., Zou, B., Wang, G., Liu, Y., et al.: Sar-rarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496 (2023)

14. Rapuri, S., Seenivasan, L., Schneider, D., Soberanis-Mukul, R., He, Y., Ding, H., et al.: Saw: Toward a surgical action world model via controllable and scalable video generation. arXiv preprint arXiv:2603.13024 (2026)

15. Saeidi, H., Opfermann, J.D., Kam, M., Wei, S., Léonard, S., Hsieh, M.H., Kang, J.U., Krieger, A.: Autonomous robotic laparoscopic surgery for intestinal anastomosis. Science robotics 7(62), eabj2908 (2022)

16. Shi, C., Zheng, Y., Fey, A.M.: Recognition and prediction of surgical gestures and trajectories using transformer models in robot-assisted surgery. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 8017– 8024. IEEE (2022)

17. Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018)

18. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

19. Wu, J., Holm, F., Chen, C., Wang, A., Hu, Y., Ye, X., Zang, Z., Xu, M., Zhou, L., Liao, H., et al.: Unisurg: A video-native foundation model for universal understanding of surgical videos. arXiv preprint arXiv:2602.05638 (2026)

20. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)