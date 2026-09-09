# Rethinking Learned Occupancy in Autonomous Active Mapping with Observation-Gated Filtering

Jiahui Zhang<sup>1</sup>, Bonian Han<sup>1</sup>, Gongbo Liang<sup>2</sup>, and Yu Zhang<sup>1B</sup>

Abstract— Autonomous 3D active mapping requires a space robot to choose where to sense while building the geometry needed for navigation. Learned occupancy completion extends spatial context beyond the current field of view, but one predicted map often serves two planning roles: it scores expected surface gain and constrains collision-free motion. Unsupported occupancy can therefore distort both where the robot looks and where it believes it can travel. We study this coupled interface in a controlled closed-loop benchmark by holding the active-mapping system fixed and varying only its planner-facing occupancy across observation-only, learned, oracle-corrected, and ground-truth conditions. Improving occupancy accuracy does not monotonically improve closed-loop coverage: across 25 starts, planning with ground-truth occupancy reaches 70% of the learned baseline’s final coverage 12.7 steps earlier on average, while increasing final coverage by only 0.031. Guided by this diagnosis, we introduce an observation-gated filter that retains completion in insufficiently observed regions and suppresses predictions only after repeated frustum exposure without nearby RGB-D support. The filter improves both targeted failureprone starts without retraining or ground truth. These results motivate online revision of planner-facing geometry during autonomous intervals between communication windows. The current study assumes benchmark RGB-D observations and sufficiently accurate pose estimates; planetary sensing conditions and accumulated localization drift remain to be evaluated.

## I. INTRODUCTION

Autonomous 3D active mapping couples reconstruction with action selection: at each step, a robot must update an incomplete map and choose the observation that will improve it next. This closed loop is especially consequential for planetary surface or subsurface reconnaissance, where supervision is delayed, motion is costly, and reliable external localization may be unavailable [1] [2]. Learned geometric completion is attractive in this setting. An occupancy model can use partial observations to predict unobserved structure and guide sensing before a complete map exists [3] [4].

The same prior creates a coupled risk. In recent activemapping systems, one occupancy map contributes to the expected new surface visible from a candidate viewpoint and also constrains collision-free motion [5] [6]. A hallucinated surface may therefore attract the robot as an apparently informative target, block a route through free space, or do both. When the mapper must operate between communication windows, planner-facing geometry must remain revisable as onboard evidence accumulates.

![](images/cbb7fbd5052d176eb3f0a0cf2115c64a54421ea039ac167e7d450151cf4b4210.jpg)  
Fig. 1. Closed-loop occupancy-guided active mapping. Each rover observation updates a learned occupancy map used for both gain estimation and collision checking. Repeated exposure without nearby RGB-D support triggers observation-gated filtering before planning. The selected motion produces the next observation.

We therefore ask: when should an active mapper retain a learned geometric prior, and when should accumulated observations override it? We use a controlled terrestrial 3D benchmark to isolate this planner-facing mechanism. The experiments do not claim validation under planetary appearance, degraded sensing, or long-horizon localization drift. They instead diagnose the shared occupancy interface and test an online revision rule for the closed perception, mapping, and planning loop in Fig. 1.

## II. RELATED WORK

Space-robot autonomy couples onboard state estimation, guidance, hazard avoidance, and local motion planning so that a vehicle can continue operating when external localization or frequent ground intervention is unavailable [1]. Exploration determines where sensing effort should be spent. Coordinated rover–copter planning treats mapping location as a mission decision rather than a passive by-product of navigation [2]. Predictive perception has been studied for planetary exploration [7], while large-scale mapping addresses perceptually degraded subterranean environments [8]. Receding-horizon exploration and uncertainty-guided reconstruction similarly couple view utility with reachable motion [9] [10]. These systems establish the importance of onboard spatial reasoning, but do not isolate failures caused when learned completion is shared by gain estimation and collision checking.

Learned occupancy provides a complementary capability. Occupancy Networks represent 3D geometry as a continuous inside–outside function [3], while occupancy anticipation predicts free and occupied space for navigation before the environment is fully observed [4]. For active reconstruction, SCONE and MACARONS use learned occupancy and coverage anticipation [11] [5]; NARUTO plans from reconstruction uncertainty [12]; and NextBestPath searches beyond a single view [13]. MAGICIAN converts pretrained occupancy into imagined Gaussian primitives for long-horizon planning [6]. These approaches demonstrate the utility of geometric prediction, while aggregate mapping performance alone does not reveal which decisions are caused by unsupported completion.

We focus on this perception–planning interface. Rather than changing the occupancy architecture, we freeze a planner, intervene on its geometry, and trace changes in viewpoint utility and coverage. The proposed online correction layer preserves predictions in sparsely observed space and suppresses those that remain unsupported after repeated nominal exposure. This interface matters when learned completion guides action during long periods of onboard autonomy.

## III. LEARNED OCCUPANCY IN TWO PLANNING ROLES

## A. Planner-facing occupancy

At planning step $t ,$ the agent has accumulated an RGB-D surface point cloud $S _ { t }$ and visited poses $\mathcal { C } _ { t } .$ A pretrained network predicts an occupancy probability $\hat { \sigma } _ { t } ( \mathbf { x } )$ for a 3D query point; we threshold it at 0.5 to obtain $\widehat { \mathcal { O } } _ { t }$ . We instantiate the study with MAGICIAN [6]. It converts predicted occupied points to Gaussian primitives and renders their novelty from candidate views to estimate coverage gain. The same occupancy is used for collision checking during long-horizon trajectory search. Thus, occupancy changes both a trajectory’s score and whether the planner regards it as feasible.

Let $\widetilde { \mathcal { O } } _ { t }$ denote the occupied set exposed to the planner and $\mathcal { T } _ { t }$ its candidate trajectories. The shared interface is schematically

$$
\pmb { \tau } _ { t } ^ { * } = \arg \operatorname* { m a x } _ { \pmb { \tau } \in \mathcal { T } _ { t } } \sum _ { \mathbf { c } \in \pmb { \tau } } G _ { t } ( \mathbf { c } ; \widetilde { O } _ { t } ) \quad \mathrm { s . t . } \quad \mathrm { F r e e } ( \pmb { \tau } ; \widetilde { O } _ { t } ) = 1 ,\tag{1}
$$

where $G _ { t }$ is rendered predicted surface gain and Free is the collision-feasibility test. A single occupied prediction can therefore change the objective, the feasible set, or both. A false positive may look like an uncovered target and may also reject an otherwise traversable route; a false negative may remove a useful target or change inferred connectivity.

These effects unfold in closed loop. The selected trajectory determines the next RGB-D measurements, those measurements update $S _ { t }$ and the learned prediction, and the new planner-facing geometry changes the next planning problem. Pointwise occupancy accuracy at one step cannot fully characterize this downstream effect. Our interventions measure the net result after the feedback has unfolded.

## B. Controlled interventions

We hold the occupancy network, candidate views, beam search, collision rule, motion budget, and reconstruction pipeline fixed. Only the occupancy supplied to the planner changes. We compare: (i) accumulated observations only; (ii) learned occupancy with oracle false positives removed; (iii) the learned occupancy baseline; (iv) learned occupancy with oracle false negatives restored; and (v) ground-truth occupancy.

These conditions form a diagnostic decomposition, rather than five separately trained models. With $\mathcal { O } ^ { * }$ denoting reference occupied samples and $\mathrm { F P } _ { t } , \mathrm { F N } _ { t }$ the oracle-tagged error sets, the planner inputs are

$$
\begin{array} { r l r } & { \tilde { \mathcal { O } } _ { t } ^ { \mathrm { o b s } } = S _ { t } , } & \\ & { \tilde { \mathcal { O } } _ { t } ^ { - \mathrm { F P } } = S _ { t } \cup ( \widehat { \mathcal { O } } _ { t } \setminus \mathrm { F P } _ { t } ) , } & \\ & { \tilde { \mathcal { O } } _ { t } ^ { \mathrm { b a s e } } = S _ { t } \cup \widehat { \mathcal { O } } _ { t } , } & \\ & { \tilde { \mathcal { O } } _ { t } ^ { + \mathrm { F N } } = S _ { t } \cup \widehat { \mathcal { O } } _ { t } \cup \mathrm { F N } _ { t } , } & \\ & { \tilde { \mathcal { O } } _ { t } ^ { \mathrm { G T } } = \mathcal { O } ^ { * } . } & \end{array}\tag{2}
$$

Observation-only planning measures the net value of completion. The two oracle conditions ask whether commission or omission errors dominate while leaving the other error type intact. Ground-truth occupancy removes both and reveals what geometric correctness can achieve when view generation, trajectory search, motion budget, and reconstruction remain fixed.

For oracle tagging, a predicted point is a false positive if it is farther than $\delta = 0 . 0 3 d _ { \mathrm { s c e n e } }$ from the reference surface; a reference point is a false negative if no prediction lies within the same tolerance. Ground truth is used only for these diagnostic interventions and evaluation, never by the proposed online filter.

Experiments use Macarons++ [6]: five scenes, five fixed starts per scene, 100 planning actions and 101 camera poses per trajectory. Each condition runs closed loop, so a changed decision produces different future observations and worldmodel updates. We measure final surface coverage $C _ { T }$ normalized area under the coverage trajectory (AUC), and $S _ { 7 0 }$ , the first step reaching 70% of the learned baseline’s final coverage for the same start. AUC and $S _ { 7 0 }$ quantify coverageacquisition efficiency under the fixed motion budget; they do not measure energy directly.

We rerun the complete trajectory for every intervention rather than rescoring a shared log. After the first changed action, conditions receive different observations and no longer share the same map history. Paired differences thus measure the total system-level consequence of changing planner geometry at the same scene and start, including future sensing; they do not attribute every later trajectory difference to one isolated point.

## C. Findings

First, offline ground-truth tags reveal that phantom occupancy contributes 53.5% and 35.0% of the rendered gain in two viewpoints selected by the learned planner. The error therefore enters the decision signal, rather than remaining only a map-quality defect.

Second, completion has start-dependent value. Removing it changes pooled final coverage by only −0.007, but this average combines opposite behaviors. In the five starts where the baseline ends below 0.75 coverage, observation-only planning changes $C _ { T } / \mathrm { A U C } / S _ { 7 0 }$ by $+ 0 . 1 0 0 / + 0 . 0 5 5 / - 8 . 6 ;$ in the sixteen starts above 0.85, the changes are $- 0 . 0 5 0 / -$ $0 . 0 7 3 / + 9 . 4$ . These post-hoc strata describe heterogeneity and do not imply that failure is predictable from the initial pose.

TABLE I  
CONTROLLED OCCUPANCY INTERVENTIONS. THE LEARNED ROW REPORTS ABSOLUTE VALUES; OTHER ROWS ARE MEAN PAIRED CHANGES FROM THAT BASELINE. LOWER $\Delta S _ { 7 0 }$ IS FASTER. ORACLE ROWS USE GROUND TRUTH ONLY FOR DIAGNOSIS.
<table><tr><td>Planner occupancy Group</td><td></td><td> $\Delta C _ { T }$ </td><td>∆AUC</td><td> $\Delta S _ { 7 0 }$ </td></tr><tr><td>Obs. only</td><td> $\mathrm { A l l ~ } ( n = 2 5 )$ </td><td> $- 0 . 0 0 7 \ - 0 . 0 3 8$ </td><td></td><td>+4.6</td></tr><tr><td rowspan="5">Learned – FP†</td><td> $\mathrm { L o w } \ ( n = 5 )$ </td><td> $+ 0 . 1 0 0 ~ + 0 . 0 5 5$ </td><td></td><td>-8.6</td></tr><tr><td>High (n = 16)</td><td> $- 0 . 0 5 0 \ - 0 . 0 7 3$ </td><td></td><td>+9.4</td></tr><tr><td>All</td><td>-0.004 -0.006</td><td></td><td>+2.5</td></tr><tr><td>Low</td><td>+0.020+0.019</td><td></td><td>-0.6</td></tr><tr><td>High</td><td>-0.029 -0.019</td><td></td><td>+4.2</td></tr><tr><td rowspan="3">Learned (abs.)</td><td>All</td><td>0.846</td><td>0.644</td><td>31.6</td></tr><tr><td>Low</td><td>0.676</td><td>0.485</td><td>39.6</td></tr><tr><td>High</td><td>0.908</td><td>0.706</td><td>27.4</td></tr><tr><td rowspan="3">Learned + FN†</td><td>All</td><td></td><td>-0.026 +0.031</td><td>-3.0</td></tr><tr><td>Low</td><td>-0.044 +0.035</td><td></td><td>-4.0</td></tr><tr><td>High</td><td>-0.023+0.030</td><td></td><td>-1.9</td></tr><tr><td rowspan="3">Ground truth†</td><td>All</td><td>+0.031 +0.086</td><td></td><td>-12.7</td></tr><tr><td>Low</td><td>+0.095 +0.178</td><td></td><td>-28.4</td></tr><tr><td>High</td><td>+0.004 +0.055</td><td></td><td>-7.4</td></tr></table>

<sup>†</sup>Ground-truth oracle. Low: baseline $\overline { { C _ { T } < 0 . 7 5 ; } }$ high: $C _ { T } > 0 . 8 5 .$

Third, more accurate occupancy is not monotonically better for closed-loop mapping (Table I). Oracle false-positive removal fails to improve pooled performance. Restoring false negatives accelerates coverage $( \Delta \mathrm { A U C } ~ = ~ + 0 . 0 3 1$ $\Delta S _ { 7 0 } ~ = ~ - 3 . 0 )$ while reducing final coverage by 0.026. Even complete ground-truth occupancy primarily improves efficiency: ∆AUC is +0.086 and $S _ { 7 0 }$ is 12.7 steps earlier, whereas final coverage rises only 0.031. Ground truth is consequently a diagnostic reference for this fixed planner, not an upper bound on endpoint coverage.

This non-monotonicity is consistent with sequential exploration. A local correction can reorder candidate trajectories, after which the robot observes a different subset of the scene. It may accelerate early coverage without improving the final reachable surface, or remove a misleading target that had incidentally carried the robot toward useful nearby geometry. The gap between ground-truth improvements in AUC and final coverage further indicates that candidate-view placement, reachability, and trajectory search remain limiting even after geometry is corrected. World models therefore require decision-level evaluation alongside geometric accuracy.

## IV. OBSERVATION-GATED FILTERING

Global removal of completion would discard useful guidance in genuinely unobserved space. We instead classify a predicted point as unsupported only after repeated frustum inclusion without nearby accumulated surface support. Let $n _ { t } ( \mathbf { x } )$ count the acquired camera frustums containing prediction x. The filtered set is

$$
\mathcal { U } _ { t } = \left\{ \mathbf { x } \in \widehat { \mathcal { O } } _ { t } \ \bigg | \ n _ { t } ( \mathbf { x } ) \geq K , \ \operatorname* { m i n } _ { \mathbf { p } \in S _ { t } } \| \mathbf { x } - \mathbf { p } \| _ { 2 } > \epsilon \right\} ,\tag{3}
$$

with $K = 2$ and $\epsilon = 0 . 0 3 d _ { \mathrm { s c e n e } }$ . The planner receives $\textstyle S _ { t } \cup$ $( \widehat { \mathcal { O } } _ { t } \setminus \mathcal { U } _ { t } )$ . Unsupported points contribute neither rendered gain nor collision constraints. The set is recomputed after each observation, so later nearby surface support restores a prediction. The filter requires no ground truth, retraining, or extra rendering pass. Frustum inclusion is a simple exposure proxy; a filtered point is unsupported, not proven false.

TABLE II  
TARGETED FINAL COVERAGE (MEAN±STD, THREE REPETITIONS).
<table><tr><td>Start</td><td>Learned baseline</td><td>Obs.-gated filter</td><td> $\Delta C _ { T }$ </td></tr><tr><td>Pantheon/2</td><td> $0 . 3 0 9 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 4 7 2 \pm 0 . 0 7 8 }$ </td><td>+0.163</td></tr><tr><td>Sestino/2</td><td> $0 . 8 3 6 \pm 0 . 1 0 8$ </td><td> $\mathbf { 0 . 9 5 8 \pm 0 . 0 2 1 }$ </td><td>+0.122</td></tr></table>

TABLE III

TRANSLATION FROM THE CONTROLLED STUDY TO SPACE-ORIENTED VALIDATION. THE FINAL COLUMN LISTS REQUIRED TESTS, NOT DEMONSTRATED CAPABILITIES.
<table><tr><td>Interface</td><td>Current evidence</td><td>Required validation layer</td></tr><tr><td>Sensing</td><td>Benchmark RGB-D</td><td>Low light, dust, noise, and structured dropout</td></tr><tr><td></td><td>State estimation Benchmark poses</td><td>Drift, covariance, and relocalization failure</td></tr><tr><td>Exposure</td><td>tance</td><td>Frustum count plus dis- Occlusion- and ray-consistent visibil- ity</td></tr><tr><td>Mobility</td><td>Free 6-DoF motion</td><td>Rover kinematics, traversability, and clearance</td></tr><tr><td>Planner use</td><td>One shared filter</td><td>Channel-specific safety thresholds</td></tr><tr><td>Resources</td><td></td><td>No extra rendering pass Runtime, memory, energy, and fault response</td></tr></table>

Predictions with $n _ { t } ( { \bf x } ) < K$ are deliberately retained, even when they lack nearby support, because the robot may not yet have tested them. Repeated nominal exposure converts the continuing absence of support into a correction signal. Recomputing U<sub>t</sub> rather than permanently deleting points makes the update reversible: later surface evidence can return a prediction to both planner channels. The filter thus preserves completion in unexplored space while allowing accumulated observations to revise the planner-facing map.

On the two targeted failure-prone starts selected by the diagnosis, observation-gated filtering raises mean final coverage by 0.163 and 0.122 (Table II). This is a targeted proof of concept; the 25-start study supports the diagnosis, while broader filter evaluation remains necessary.

## V. IMPLICATIONS FOR ACTIVE PERCEPTION IN SPACE ROBOTICS

The closest deployment setting is rover-based surface or subsurface reconnaissance, where limited viewpoints, delayed supervision, and drift in onboard localization make planning depend on spatial predictions that remain revisable. The filter preserves learned structure in untested terrain, then revises it when repeated observations fail to provide nearby surface support. The same interface may also arise in inspection of habitats or orbital structures.

The filter sits after each world-model update and before the planner queries gain or feasibility. Because it uses accumulated observations rather than ground-truth labels or retraining, this interface could operate onboard between communication windows. That placement is attractive for delayed supervision, but the present experiments do not measure latency, memory, energy, or fault recovery. Table III therefore separates the mechanism tested here from the validation needed for a mission-oriented implementation. The role of onboard autonomy in current Mars rover operations further raises the standard for reliability and operational evidence [14].

Several gaps remain before such a filter can enter a flight-like navigation stack. Under degraded sensing, nominal frustum inclusion is weak evidence. A space-oriented implementation should count exposure only when a prediction is within usable sensor range, is not occluded by supported foreground structure, and receives a valid measurement capable of testing it. A dropped or saturated range return should provide no negative evidence. A valid ray that passes through the predicted location and terminates farther away can instead supply explicit free-space evidence. This distinction matters under low illumination, dust, reflective materials, or range-dependent dropout.

When reliable external localization is unavailable, both frustum membership and nearest-surface distance depend on drifting onboard pose estimates. Pose error can displace a correct prediction from its corresponding measurement, causing the filter to treat it as unsupported. A conservative extension should propagate pose uncertainty into both exposure and support tests, count an exposure only when the point is likely to lie in a valid visible region, and search for supporting observations within an uncertainty-expanded neighborhood. These are deployment requirements; the present study uses accurate benchmark poses and does not evaluate localization drift or failure.

The two planner channels also warrant different evidentiary thresholds. Removing a point from gain estimation changes exploration preference. Removing it from collision checking can admit a new trajectory and carries a greater safety burden. A flight-like system could down-weight gain after repeated valid contradiction while retaining a collision constraint until ray-consistent free space or redundant multi-view confirmation is available. The current shared filter operates on the same coupled planner interface examined in our diagnosis.

Validation should first inject structured depth dropout, range noise, and drift in onboard pose estimates while preserving the same scenes, starts, and planner. Comparisons should separate the present frustum-only gate from depth-aware variants and variants that incorporate pose uncertainty, and report incorrect suppression, collision events, path clearance, latency, and memory in addition to coverage. Planetary-analog trials should then use stereo or depth sensing with onboard visual–inertial or SLAM pose estimates. The present evidence establishes a planner-facing failure and a targeted correction; it does not establish flight readiness.

## VI. CONCLUSION

Our results motivate rethinking learned occupancy in autonomous active mapping as a revisable planning prior. Because learned occupancy enters both gain estimation and collision checking, its errors can alter where an active mapper looks and which motions it considers feasible. Controlled interventions show that occupancy accuracy alone does not predict closed-loop coverage. Observation-gated filtering retains useful completion while suppressing repeatedly unsupported predictions, improving two targeted failure cases without retraining or ground truth.

Although evaluated here for 3D active mapping, keeping planner-facing predictions revisable also matters in active search. Search-TTA [15] updates its image encoder during search with uncertainty-weighted online gradients, refining potentially inaccurate target-presence maps. Its parameter adaptation differs from our deterministic, reversible 3D occupancy gate, yet both use accumulated task evidence to revise predictions before subsequent planning decisions. We consider active search as a future work.

## REFERENCES

[1] M. Azkarate, L. Gerdes, L. Joudrier, and C. J. Perez-del Pulgar, “A´ GNC architecture for planetary rovers with autonomous navigation,” in Proceedings of the IEEE International Conference on Robotics and Automation, pp. 3003–3009, 2020.

[2] T. Sasaki, K. Otsu, R. Thakker, S. Haesaert, and A. Agha-Mohammadi, “Where to map? iterative rover-copter path planning for Mars exploration,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2123– 2130, 2020.

[3] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger, “Occupancy networks: Learning 3d reconstruction in function space,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4455–4465, IEEE, 2019.

[4] S. K. Ramakrishnan, Z. Al-Halah, and K. Grauman, “Occupancy anticipation for efficient exploration and navigation,” in European conference on computer vision, pp. 400–418, Springer, 2020.

[5] A. Guedon, T. Monnier, P. Monasse, and V. Lepetit, “Macarons:´ Mapping and coverage anticipation with rgb online self-supervision,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 940–951, IEEE, 2023.

[6] S. Li, A. Guedon, S. Chen, and V. Lepetit, “Magician: Efficient´ long-term planning with imagined gaussians for active mapping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21606–21615, 2026.

[7] K. Otsu, A.-A. Agha-Mohammadi, and M. Paton, “Where to look? predictive perception with applications to planetary exploration,” IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 635–642, 2017.

[8] K. Ebadi et al., “LAMP: Large-scale autonomous mapping and positioning for exploration of perceptually-degraded subterranean environments,” in Proceedings of the IEEE International Conference on Robotics and Automation, pp. 80–86, 2020.

[9] A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and R. Siegwart, “Receding horizon “next-best-view” planner for 3D exploration,” in Proceedings of the IEEE International Conference on Robotics and Automation, pp. 1462–1468, 2016.

[10] S. Lee, L. Chen, J. Wang, A. Liniger, S. Kumar, and F. Yu, “Uncertainty guided policy for active robotic 3D reconstruction using neural radiance fields,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 12070– 12077, 2022.

[11] A. Guedon, P. Monasse, and V. Lepetit, “Scone: Surface coverage ´ optimization in unknown environments by volumetric integration,” Advances in Neural Information Processing Systems, vol. 35, pp. 20731– 20743, 2022.

[12] Z. Feng, H. Zhan, Z. Chen, Q. Yan, X. Xu, C. Cai, B. Li, Q. Zhu, and Y. Xu, “NARUTO: Neural active reconstruction from uncertain target observations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21572–21583, 2024.

[13] S. Li, A. Guedon, C. Boittiaux, S. Chen, and V. Lepetit, “NextBestPath:´ Efficient 3D mapping of unseen environments,” in International Conference on Learning Representations, pp. 49464–49482, 2025.

[14] V. Verma et al., “Autonomous robotics is driving Perseverance rover’s progress on Mars,” Science Robotics, vol. 8, no. 80, p. eadi3099, 2023.

[15] D. M. S. Tan et al., “Search-TTA: A multi-modal test-time adaptation framework for visual search in the wild,” in Proceedings of the 9th Conference on Robot Learning, vol. 305 of Proceedings of Machine Learning Research, pp. 2093–2120, 2025.