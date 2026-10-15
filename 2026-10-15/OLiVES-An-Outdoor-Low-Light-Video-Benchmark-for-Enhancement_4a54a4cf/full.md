# OLiVES: An Outdoor Low-Light Video Benchmark for Enhancement and Segmentation

Yini Li<sup>∗</sup>   
Visual Information Laboratory   
University of Bristol   
Bristol, United Kingdom   
ub24017@bristol.ac.uk   
Joanne Lin<sup>∗</sup>   
Visual Information Laboratory   
University of Bristol   
Bristol, United Kingdom   
joanne.lin@bristol.ac.uk

David Blackham daveblackham@mac.com

David Bull   
Visual Information Laboratory   
University of Bristol   
Bristol, United Kingdom   
Dave.Bull@bristol.ac.uk

Nantheera Anantrasirichai Visual Information Laboratory University of Bristol Bristol, United Kingdom n.anantrasirichai@bristol.ac.uk

## Abstract

Low-light video processing remains an underexplored problem in computer vision, primarily due to the scarcity of high-quality ground-truth data. Existing public datasets exhibit several limitations, including restricted motion diversity, limited scene coverage, weak degradations, reliance on auxiliary modalities such as event data, and misalignment between low-light inputs and ground truth. Furthermore, current datasets typically address either low-light enhancement or scene understanding, whereas these tasks should be considered jointly. We introduce OLiVES, a benchmark dataset for outdoor low-light videos with dynamic scenes, designed to support both low-light video enhancement (LLVE) and video object segmentation (VOS). The dataset addresses key content gaps by incorporating a diverse range of natural textures, animals, occlusions, dynamic activities, and close-up scenes. To the best of our knowledge, OLiVES is the first outdoor low-light video dataset captured using professional cinema-grade cameras. It provides over 25,000 aligned normal- and low-light frame pairs, along with over 46,000 VOS annotations. We benchmark state-of-the-art LLVE and VOS methods on OLiVES and show that training on it improves performance on both tasks. The OLiVES project page is available at: https://anonymous-OLiVES.github.io/

## 1 Introduction

Enhancing the visibility and understanding of scenes captured under low-light conditions is highly desired due to its applicability to autonomous vehicles, surveillance, wildlife filmmaking and conservation, etc. While many computer vision methods show strong performance in well-lit scenes, their performance degrades significantly in low-light conditions [3, 40, 31, 30]. There are many reasons for this reduction in performance including noise, blur, low brightness/contrast and computational resources required to process videos.

Low-light video enhancement (LLVE) has emerged as a key task to address these challenges, aiming to improve perceptual quality or serve as a pre-processing step for downstream tasks. Low-light video enhancement provides an added challenge when compared to its image counterpart, low-light image enhancement (LLIE), due to the importance of temporal consistency. One major challenge in constructing LLVE dataset is the acquisition of paired low-light frames and corresponding groundtruth (GT). Most existing datasets adopt physical alignment methods to overcome this challenge, e.g. dual-camera systems [4] or repeated shooting of the same scene [31]. However, due to the constraints of the capture setup, the recorded videos are typically limited to either static scenes or scenes containing no moving objects. As a result, these datasets fail to adequately represent the complex motion patterns commonly encountered in real-world low-light environments. Furthermore, we observe that several existing datasets exhibit low degradation levels, e.g. DID [13] which simply reduces ISO to capture low-light videos.

Video object segmentation (VOS) is a further challenging downstream task whose objective is to segment and track target objects across frames. While numerous VOS benchmarks have been released [35, 36, 42, 46, 10, 11], the vast majority focus on well-lit conditions; to our knowledge, only Li et al. [23] has explored VOS in low-light through their LLE-VOS dataset. One of the main limitations of this dataset is that it is designed to leverage event data, which is not captured by most cameras, thus restricting its real-world applicability. More broadly, downstream tasks in low-light conditions remain under-studied, in large part because of the scarcity of training data. A common workaround is to apply a low-light enhancement step before feeding frames into models trained on predominantly well-lit datasets. However, despite recent progress in enhancement techniques [16, 2, 21, 26], this two-stage pipeline often hinders downstream performance. Enhancement methods improve perceptual quality but degrade the semantic cues that downstream models rely on [5, 23, 29, 30]. Moreover, existing studies indicate that jointly optimizing low-level enhancement and high-level semantics is beneficial to both tasks, improving restoration quality while simultaneously enhancing downstream performance [39, 52, 34, 15, 19, 30]. To the best of our knowledge, the only low-light video benchmark for both enhancement and a downstream task (semantic segmentation) is SDE [4]. Howevever, SDE has notable limitations: object motion is restricted (due to its setup) and the annotations are generated using off-the-shelf deep learning models without manual verification.

To address these challenges, we propose a new low-light benchmark, Outdoor Low-Light Videos for Enhancement and Segmentation (OLiVES), supporting joint LLVE and VOS tasks on real-world lowlight videos. Example videos are shown in Fig. 1. OLiVES provides 25,254 registered normal/lowlight frames for low-light video enhancement and 46,366 pixel-wise segmentation annotations for the low-light VOS task. Unlike existing datasets that treat these tasks independently, our benchmark enables a unified evaluation of restoration and downstream understanding, encouraging research for improving both perceptual and semantic quality. To capture diverse lighting conditions, including extreme low-light scenarios, we employ three Neutral Density (ND) filters of varying transmittance while keeping ISO and exposure time fixed. Crucially, OLiVES is captured with professional cinema grade cameras, yielding high-fidelity footage under genuine outdoor lighting. The resulting scenes feature animals, natural textures, foliage, and dynamic outdoor activity. This opens up benchmarking and model development for application areas underserved by existing datasets, such as natural history filmmaking, wildlife monitoring, and nighttime surveillance.

## 2 Related work

Low-light video enhancement. Low-light video enhancement is a restoration task that aims to improve perceptual fidelity. It involves several steps such as removing noise and blur degradations, increasing brightness and contrast, and enhancing colors. Several works [40, 13, 31, 4] have proposed benchmarks which provide low-light video frame inputs, and aligned normal-light video frames for ground truth. Table 1 summarizes existing datasets and their characteristics. However, each has notable limitations:

![](images/45545f1b08dafd30daaf6e236ad3c20b6ecaa608a5895c0b98ca37a3889dedfa.jpg)  
(b)  
Figure 1: Examples of the OLiVES dataset. (a) Low-light videos with their corresponding normallight ground truth and object segmentation masks. Each scene is provided at multiple light levels, (b) illustrating the noise levels exhibited in the low-light videos when using different ND filters (histogram-matched to GT for comparison).

Table 1: Comparison of low-light video enhancement datasets.
<table><tr><td>Dataset</td><td>Modality</td><td>Frames</td><td>Cam. Motion</td><td>Obj. Motion</td><td>Resolution</td><td>VOS ann.</td><td>Device</td></tr><tr><td>DRV [3]</td><td>Raw</td><td>22,220</td><td>Static</td><td>Static</td><td> $5 4 9 6 \times 3 6 7 2$ </td><td>x</td><td>Sony RX100 VI</td></tr><tr><td>SDSD [40]</td><td>RGB</td><td>37,500</td><td>Dynamic</td><td>Static</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>x</td><td>Canon 6D Mark II</td></tr><tr><td>LLRVD [14]</td><td>Raw</td><td>4,086</td><td>Dynamic</td><td>Dynamic</td><td> $2 6 0 0 \times 1 4 0 0$ </td><td>x</td><td>Sony α7R IV</td></tr><tr><td>BVI-RLV [31]</td><td>RGB</td><td>31,800</td><td>Dynamic</td><td>Static</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>x</td><td>Sony α7S II</td></tr><tr><td>SDE [4]</td><td>RGB/Event</td><td>31,477</td><td>Dynamic</td><td>Static</td><td> $3 4 6 \times 2 6 0$ </td><td>x</td><td>DAVIS 346</td></tr><tr><td>DID [13]</td><td>RGB</td><td>41,038</td><td>Dynamic</td><td>Static</td><td> $2 5 6 0 \times 1 4 4 0$ </td><td>x</td><td>5 Camera Models</td></tr><tr><td>OLiVES (Ours)</td><td>RGB</td><td>25,254</td><td>Dynamic</td><td>Dynamic</td><td> $1 9 2 0 \times 1 0 8 0$ </td><td>√</td><td>RED V-RAPTOR</td></tr></table>

Table 2: Comparison of video object segmentation (VOS) datasets.
<table><tr><td>Dataset</td><td>Light</td><td>Frames</td><td>Categories</td><td>Objects</td><td>Annotations</td><td>New objects</td></tr><tr><td>DAVIS 2016 [35]</td><td>Normal</td><td>3,440</td><td></td><td>50</td><td>3,455</td><td>x</td></tr><tr><td>DAVIS 2017 [36]</td><td>Normal</td><td>6,208</td><td></td><td>205</td><td>13,543</td><td>x</td></tr><tr><td>YouTube-VOS [42]</td><td>Normal</td><td>120,532</td><td>94</td><td>7755</td><td>197,272</td><td>√</td></tr><tr><td>MOSEv1 [10]</td><td>Normal</td><td>130,149</td><td>36</td><td>5,200</td><td>431,725</td><td>x</td></tr><tr><td>MOSEv2 [11]</td><td>Mixed</td><td>468,251</td><td>200</td><td>10,074</td><td>701,976</td><td>x</td></tr><tr><td>LLE-VOS [23]</td><td>Low</td><td>5,600</td><td></td><td>120</td><td>9,606</td><td>x</td></tr><tr><td>OLiVES (Ours)</td><td>Low</td><td>22,893</td><td>10</td><td>715</td><td>46,366</td><td>√</td></tr></table>

DRV [3] (also known as SMID) is a RAW low-light video dataset with aligned short/long-exposure pairs from a single device with ground truth restricted to static scenes. SDSD [40] is a public low-light video dataset captured with a mechatronic system, but suffers from spatial misalignment between normal/low-light pairs and inconsistent scenarios. DID [13] is the largest LLVE dataset by frame count, using five cameras to capture diverse degradations. However, it synthesizes low-light by reducing ISO under well-lit conditions, limiting realism and noise modelling, and its gimbal setup restricts motion to linear trajectories. LLRVD [14] is a low-light RAW dataset with richer motion, obtained by recording videos displayed on a Dell U2720QM monitor; however, the lighting is not physically realistic, and the dataset contains only 4,086 pairs. BVI-RLV [31] introduces non-uniform motion via a programmable dolly and mitigates misalignment through repeated capture and frame selection, but is limited to controlled indoor scenes with static objects, lacking outdoor scenarios and real motion. SDE [4] uses a robotic arm to capture more diverse motion than prior methods and records scenes with a DAVIS 346 event camera before applying an ND8 filter for low-light sequences. However, its low resolution (346×260) limits effectiveness for detail restoration.

These benchmarks all share a common limitation: their data capture process requires re-recording scenes under different lighting conditions, thus restricting the scene motion, due to the need for accurate frame alignment. The consequence of this is that these datasets do not exhibit heavy motion blur from moving objects; a key degradation found in low-light videos which compromises perceptual fidelity and performance in computer vision tasks. Table 1 shows that OLiVES is the only one to combine dynamic camera and object motion with VOS annotations.

Video object segmentation. Video object segmentation (VOS) is a high-level computer vision downstream task which extends the object tracking task by producing segmentation masks of objects of interest rather than bounding boxes. This provides a fine-grained understanding of the shape of the objects. Semi-supervised VOS is typical, requiring additional data to guide the method to correctly segment the object; this is most often a mask of the first frame where an object appears. Popular VOS benchmarks such as DAVIS [35, 36], YouTube-VOS [42, 46], MOSEv1 [10] all provide annotations for the mask-initialization semi-supervised setting. Most of these benchmarks only consider objects located in the first frame except for YouTube-VOS which adds an additional difficulty to the task by having videos where new objects appear in later frames.

While these datasets are valuable for real-world applications, they provide normal-light videos. Only MOSEv2 [11] includes low-light scenes, but does not explicitly separate them, grouping them with other challenging conditions such as underwater scenes and adverse weather. Li et al. [23] introduce LLE-VOS, a low-light VOS benchmark with aligned normal/low-light RGB frames, event data, and annotations. While it addresses a gap in the low-light domain, it has several limitations. First, performance relies heavily on event data, which is unavailable in standard cameras, limiting generalisability to RGB-only settings. Second, the low resolution (DAVIS 346) restricts fine-grained segmentation. In addition, the frames exhibit an orange colour cast from the capture process, potentially biasing model behaviour. The statistics of these VOS datasets are summarized in Table 2. Only our dataset and YouTube-VOS support objects that emerge after the first frame.

## 3 The OLiVES Dataset

## 3.1 Dataset Collection

Dual-exposure camera system. Fig. 2 shows our capture setup, where a Genlock timecode system enables temporally synchronized low-light and ground-truth videos in a dual-exposure camera system. Two cameras are mounted on a Harrier beam-splitter rig and a GFM camera dolly and jib to enable smooth, controlled motion without affecting camera alignment. The beam-splitter rig ensures precise physical alignment and divides incoming light equally into two optical paths via a high-quality 50/50 half-silvered mirror, allowing both cameras to capture identical images. Neutral density (ND) filters (Formatt Hitech Firecrest 1.2, 1.5, and 1.8), corresponding to transmittance ratios of 1/16, 1/32, and 1/64, are inserted into one path to simulate low-light conditions. These filters provide near-perfect light attenuation without noticeable color shift. The other path records the corresponding normal-light reference. A monitor connected to the cameras is used for real-time footage inspection.

![](images/34554e37975104cdfdde617e577b17d350ca7238e5aacfa0b323303613136b53.jpg)

![](images/bd629abe1e296c48bc72db93dd179315cc322f70b3419a66d8fb7d904afc79e7.jpg)

Figure 2: Capture setup. (a) Our beam-splitter rig with two genlocked cinema cameras mounted on a shared tripod. (b) Schematic of the optical path: incoming light is split between Camera 1, which records the normal-light ground truth $I _ { g t }$ , and Camera 2, which records the low-light frame $I _ { l o w }$ through one of three ND filters (1/16, 1/32, 1/64). Genlock ensures temporal synchronization between the two streams.  
![](images/ab2fdf273018aaf35e37988fae9e84804435d12bfa0dd206c795179509bcd932.jpg)  
Figure 3: Registration pipeline of the OLiVES dataset.

For high-fidelity image acquisition, we use two professional RED V-RAPTOR cameras paired with Zeiss Prime lenses. Following standard cinematography practice, we fix the ISO at 800, the frame rate at 25 fps, the aperture at F8 to maintain sufficient depth of field in outdoor scenes, and the shutter speed at 1/100 s to reduce motion blur. All other parameters are kept at their default settings. This setup represents a practical upper bound achievable with current imaging technology.

Video scenes. Existing low-light video datasets [40, 13, 31, 4] predominantly contain scenes with limited motion and close-up fine textures, due to constraints in their capture setups. To address this limitation, we deliberately capture scenes with richer and more diverse dynamics. In particular, to record natural wildlife behavior, we film animals, including lemur, gelada, meerkat, giraffe, and zebra in open zoo (Bristol Zoo Project by Bristol Zoological Society, UK) enclosures. We further include outdoor scenes featuring building textures and dynamic elements such as flags, pedestrians, and vehicles, with complex scenarios including occlusions and object interactions. With the support of professional photographers, our dataset spans a wide range of dynamic outdoor scenarios, including humans, animals, and less commonly studied objects such as flags and balls. The captured highresolution RAW data are first converted to 4096×2160 RGB images using the official REDCINE-X PRO platform with default image signal processing settings. We then crop 1920×1080 regions of interest and divide long recordings into short clips of 3 to 10 seconds.

## 3.2 Dataset Registration

Although Genlock ensures temporal synchronization between the two cameras, small spatial misalignments may remain between paired frames due to imperfections in beam-splitter geometry and lens calibration. To address this, we estimate dense pixel-wise correspondences using finetuned RoMa [12], followed by a warping procedure to align each normal-light frame to its corresponding low-light frame, as illustrated in Fig. 3. The low-light frames are kept unchanged, preserving their original distortion characteristics. The resulting transformations are limited to minor geometric corrections and do not modify photometric or structural content, thereby maintaining the validity of the normal-light frames as reference data.

![](images/2e5f0a78d812dd433b7c9c8017eae2103eae8d4b38c3fa68773476568545f212.jpg)  
Figure 4: Finetuning the fine feature encoder of RoMa with synthetically degraded low-light pairs and a contrastive objective to improve robustness to low-light degradations.

RoMa finetuning. RoMa [12] produces high-quality dense correspondences along with confidence maps under standard conditions, but directly applying the pre-trained model to our dataset leads to noticeable artifacts. This mainly arises from (i) severe domain shift between normal and low-light images, and (ii) noise and detail loss in low-light input. To improve robustness, we finetune RoMa using synthetically degraded data that mimics real low-light conditions as shown in Fig. 4. We leverage the synthetic low-light pipeline proposed by Lin et al. [30] to create synthetic low-light pairs with degradations from our captured low-light frames, and use these synthetic frames for finetuning RoMA. In addition to this, we use contrastive learning to enforce degradation invariant feature representations. We provide further details of the finetuning process in Appendix A.

Image Alignment. As low-light videos from different ND filters exhibit distinct statistical characteristics, we tailor registration strategies as in Fig. 3. For ND1.2, where details are largely preserved, we use histogram matching (HM) to reduce the brightness gap, followed by RoMa-based dense warping for alignment. We bypass enhancement for ND1.2 to keep the detail matching as accurate as possible. For ND1.5, which exhibits stronger noise, we first apply TempRetinex [26] for enhancement and denoising, followed by HM and dense warping. For ND1.8, where severe noise and color distortion compromise dense matching, we instead select reliable matched feature points using the confidence maps, and estimate a homography by RANSAC for global alignment, effectively avoiding local artifacts while maintaining alignment stability.

## 3.3 VOS Benchmark Setup

Video clip preparation. In order to provide similar lengths of videos to existing VOS benchmarks, we first randomly split the videos into smaller video clips between 40 and 100 frames. Next, we filtered the clips that were unsuitable for the VOS task (e.g. clips with no objects to track). Unlike LLE-VOS [23], we include clips with emerging/disappearing objects in our benchmark for additional difficulty. This also allows for a more practical applicability as objects cannot be guaranteed to remain within the frame. For the train/validation/test splits, we used the ratios 40%/30%/30%. We ensured that the test split remained the same as the test split for the LLVE task, while for the train/validation splits, we randomly selected clips from the LLVE train split for the respective VOS splits.

Annotations. We provide annotations for 10 categories: person, giraffe, zebra, monkey, meerkat, lemur, boat, car, flag, ball. For the training set, we first created pseudo ground truth masks by prompting SAM2 [38] using bounding boxes of the objects of interest onto the aligned normal-light videos. However, we found that the annotations generated with SAM2 are imperfect and thus we decided to refine a subset of the train set manually. As refining over 35K annotations is extremely time-consuming, especially for our videos where we capture difficult texture like animal fur, we randomly select 33% of the train videos for refinement; balancing the trade-off between manual-labor and effectiveness. The refinement process was case-by-case and involved using several procedures including: applying SAM2 again on later frames to reduce the mask drift in later frames, prompting SAM2 with a mask manually drawn for better guidance, and manually editing the generated masks. We implemented the same refinement approach for the validation set to reduce the inaccuracies from SAM2. For the test set, we hired annotators to manually annotate the objects in the videos, referring to the matched normal-light frame for better visibility. To guide the annotators, we provided the first segmentation mask of every object in the test set. During this manual annotation process, we employed an iterative feedback loop by regularly reviewing and refining the annotations, to ensure high-quality masks. See Appendix C for details on the annotation process.

（c）Optical flow of VOS  
![](images/d80483f66c5848c761fb7fe40e8f1b6b4da57c332866ba7aabe1324c25dfdf04.jpg)

![](images/2d6e4229edb9b731c285f322902d92ca0c3cc512da6f66f320bae2040a00fee1.jpg)

![](images/c80c59074857b1961143d6776f2ea9a1c979e1a7eb5326c2d3183ee6bc42a64e.jpg)

![](images/3716ddff61af4e33ea007eae680f688b275278ba5c24be668ecf0c3ca8e438cc.jpg)

![](images/12fa0e5189674e539c76ffab351c097bc394b5f8c5da196021ad8481b526cce8.jpg)

![](images/68c734371ac55060fd8190fdd8a43c2dae72295f7b8ce47899b57e469c680912.jpg)  
Figure 5: Dataset statistics overview. (a) Normalized pixel intensities for normal/low-light pairs, (b) ND filter ratios, (c) optical flow of VOS, (d) noise level distributions, (e) object category and size distributions.

## 3.4 Dataset Statistics

Composition. OLiVES comprises 137 video sequences (3–10 seconds each), totalling 25,254 frames. We split the dataset by scene into a training set (96 sequences, 17,890 frames) and a test set (41 sequences, 7,364 frames) to avoid overlap while maintaining similar content distributions across splits.

Intensity distribution. Figure 5(a) illustrates the normalized pixel intensity statistics for low-light frames under three ND filters (ND1.2, ND1.5, ND1.8) and their normal-light counterparts. The low-light intensity mainly covers a range of 0.01-0.27, while the normal-light one demonstrates a broad range of brightness ranging from 0.2 to 0.8. As shown in Fig. 5(b), the brightness distributions are also consistent between the training and testing sets. Specifically, the ratios of ND1.2, ND1.5, and ND1.8 are 28.7%:32.1%:39.2% in the training set and 28.8%:28.1%:43.1% in the test set.

Noise characteristics. To quantify the noise level, we compute the absolute difference mean $\big | \big | I _ { h m } - I _ { g t } \big | \big | _ { 1 } \big )$ of each paired frames, where the $I _ { h m }$ is the HM output of $I _ { l o w }$ . We fit the noise levels by normal distribution for better visualization in Fig. 5(d). As the ND stop increased, the noisy degree increased (noise level mean from 5.25 to 8.35, standard deviation from 1.68 to 3.02), indicating more challenging conditions in extreme darkness of OLiVES.

Motion characteristics. To analyze the motion characteristics of the dataset, we calculate the optical flow for all video frame pairs and statistically analyze the motion direction distribution of the objects. As shown in Fig. 5(c), objects exhibit motion in all directions, with horizontal motion being dominant, while also covering a rich variety of non-dominant directional motions. Regarding motion amplitude, the average inter-frame displacement of the objects is 8.3 pixels, with a variance of 139.6; while the average displacement of entire image is 7.7 pixels, with a variance of 275.7. The variance indicates that the dataset contains both small and large motions, reflecting diversity in motion scale.

Table 4: Benchmarking on OLiVES dataset across recent LLVE and LLIE SOTA methods. Bold and underline denote the best and second-best performances among all methods respectively.
<table><tr><td>Method</td><td>Venue</td><td>Type</td><td>Data</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>tLPIPS↓  $\times 1 0 ^ { - 2 }$ </td><td>WE↓  $\times 1 0 ^ { - 2 }$ </td><td>VMAF↑</td></tr><tr><td>SNR-aware [43]</td><td>CVPR&#x27;22</td><td>S</td><td>I</td><td>26.20</td><td>0.835</td><td>0.268</td><td>6.47</td><td>5.97</td><td>73.51</td></tr><tr><td>LLFlow [41]</td><td>AAAI&#x27;22</td><td>S</td><td>I</td><td>21.65</td><td>0.862</td><td>0.219</td><td>2.82</td><td>3.81</td><td>69.50</td></tr><tr><td>Retinexformer [2]</td><td>ICCV’23</td><td>S</td><td>I</td><td>27.65</td><td>0.834</td><td>0.241</td><td>6.35</td><td>5.41</td><td>76.93</td></tr><tr><td>LightenDiffusion [21]</td><td>ECCV’24</td><td>UP</td><td>I</td><td>20.32</td><td>0.741</td><td>0.380</td><td>8.70</td><td>5.56</td><td>54.36</td></tr><tr><td>Wakeup-darkness [51]</td><td>ACM TOMM&#x27;25</td><td>US</td><td>I</td><td>17.46</td><td>0.534</td><td>0.630</td><td>8.82</td><td>9.26</td><td>71.39</td></tr><tr><td>SGZSL [52]</td><td>WACV&#x27;22</td><td>US</td><td>V</td><td>16.61</td><td>0.527</td><td>0.693</td><td>10.70</td><td>11.80</td><td>74.74</td></tr><tr><td>Zero-TIG [25]</td><td>EUSIPCO&#x27;25</td><td>US</td><td>V</td><td>19.25</td><td>0.767</td><td>0.392</td><td>4.66</td><td>6.06</td><td>63.88</td></tr><tr><td>TempRetinex [26]</td><td>ICME&#x27;26</td><td>US</td><td>V</td><td>19.66</td><td>0.723</td><td>0.402</td><td>5.21</td><td>2.68</td><td>62.63</td></tr><tr><td>SDSD-Net [40]</td><td>ICCV’21</td><td>S</td><td>V</td><td>22.59</td><td>0.716</td><td>0.322</td><td>6.00</td><td>7.58</td><td>65.15</td></tr><tr><td>StableLLVE [49]</td><td>CVPR&#x27;21</td><td>S</td><td>V</td><td>22.60</td><td>0.851</td><td>0.300</td><td>2.80</td><td>4.48</td><td>49.35</td></tr><tr><td>DP3DF [44]</td><td>AAAI&#x27;23</td><td>S</td><td>V</td><td>23.22</td><td>0.781</td><td>0.325</td><td>6.22</td><td>6.57</td><td>37.37</td></tr><tr><td>FastLLVE [24]</td><td>ACM MM&#x27;23</td><td>S</td><td>V</td><td>25.87</td><td>0.800</td><td>0.309</td><td>4.97</td><td>6.88</td><td>73.01</td></tr><tr><td>BVI-CDM [32]</td><td>CVMP&#x27;24</td><td>S</td><td>V</td><td>26.40</td><td>0.854</td><td>0.326</td><td>1.67</td><td>3.90</td><td>56.03</td></tr><tr><td>BVI-Mamba [20]</td><td>MLCD&#x27;25</td><td>S</td><td>V</td><td>25.66</td><td>0.884</td><td>0.228</td><td>1.92</td><td>3.97</td><td>71.14</td></tr><tr><td>LLVE-STCD [45]</td><td>IJCAI&#x27;25</td><td>S</td><td>V</td><td>27.00</td><td>0.827</td><td>0.255</td><td>5.75</td><td>5.34</td><td>74.70</td></tr></table>

S: Supervised, US: Unsupervised, UP: Unpaired. I: Image, V: Video. WE: Warp Error

Table 5: Benchmark of SOTA VOS methods trained on the OLiVES dataset and evaluated on OLiVES validation and test set, and LLE-VOS validation set. Bold and underline denote the best and secondbest performances among all methods respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">OLiVES val</td><td colspan="4">OLiVES test</td><td colspan="3">LLE-VOS [23] val</td></tr><tr><td>J&amp;F</td><td>J</td><td>F</td><td>B</td><td>J&amp;F</td><td>J</td><td>F</td><td>B</td><td>J&amp;F</td><td> $\mathcal { I }$ </td><td>F</td></tr><tr><td>STCN [7]</td><td>NeurIPS&#x27;21</td><td>89.4</td><td>87.2</td><td>91.5</td><td>83.5</td><td>78.3</td><td>73.8</td><td>82.9</td><td>70.8</td><td>32.3</td><td>37.9</td><td>26.7</td></tr><tr><td>AOT [48]</td><td>NeurIPS&#x27;21</td><td>91.5</td><td>88.5</td><td>94.6</td><td>85.1</td><td>80.5</td><td>75.8</td><td>85.1</td><td>73.4</td><td>54.1</td><td>57.5</td><td>50.7</td></tr><tr><td>De-AOT [47]</td><td>NeurIPS’22</td><td>91.5</td><td>88.4</td><td>94.5</td><td>85.0</td><td>86.9</td><td>82.0</td><td>91.7</td><td>80.0</td><td>62.6</td><td>65.6</td><td>59.7</td></tr><tr><td>XMem [6]</td><td>ECCV’22</td><td>86.9</td><td>84.4</td><td>89.5</td><td>79.5</td><td>74.4</td><td>68.9</td><td>79.8</td><td>65.3</td><td>53.0</td><td>56.0</td><td>50.1</td></tr><tr><td>Cutie [8]</td><td>CVPR&#x27;24</td><td>93.1</td><td>91.3</td><td>94.8</td><td>88.0</td><td>88.3</td><td>84.2</td><td>92.4</td><td>82.0</td><td>59.2</td><td>62.8</td><td>55.7</td></tr><tr><td>OASIS [37]</td><td>ICCV’25</td><td>92.6</td><td>90.8</td><td>94.4</td><td>87.7</td><td>88.4</td><td>83.8</td><td>92.9</td><td>81.7</td><td>59.1</td><td>62.5</td><td>55.8</td></tr></table>

VOS annotation statistics. Table 3 summarizes the number of objects and annotations in our dataset. We provide statistics for number of videos, frames, objects and annotations. The distributions of object classes and sizes are shown in Fig. 5(e) and (f). We categorize object sizes based on the pixel area of their segmentation masks, following the thresholds defined by Lin et al. [33].

Table 3: Summary of OLiVES VOS benchmark for the train, validation and test set.
<table><tr><td>Split</td><td>Videos</td><td>Frames</td><td>Objects</td><td>Anns.</td></tr><tr><td>Train</td><td>129</td><td>8,826</td><td>275</td><td>17,901</td></tr><tr><td>Validation</td><td>99</td><td>7,012</td><td>196</td><td>12,820</td></tr><tr><td>Test</td><td>102</td><td>7,055</td><td>245</td><td>15,646</td></tr></table>

## 3.5 Limitations

While OLiVES addresses the challenges of other low-light video datasets, OLiVES has several limitations that we acknowledge: (1) the dataset is captured exclusively in outdoor daytime conditions with ND-filter-simulated low light rather than true low-light illumination; while this enables paired ground-truth capture, the noise and colour characteristics may differ from real low-light footage. (2) The scenes are restricted to a single geographic region, which may constrain generalisation. (3) The beam-splitter setup limits capture to tripod-based scenarios, excluding handheld and aerial footage. (4) While a subset of annotations is manually refined, the remaining training masks are generated by SAM2 without human verification and may contain errors.

## 4 Experiments

## 4.1 Implementation Details

Compute resources. We conducted all of our experiments on a shared cluster where each node utilizes 4 NVIDIA GH200s superchips. One NVIDIA GH200 superchip is comprised of a 72-core NVIDIA Grace CPU (with 120GB CPU memory) and a NVIDIA H100 GPU (with 96GB GPU memory). Each of the experiments we conducted varied greatly in runtime (due to different training configurations, architectures, etc.), but we estimate the total GPU hours is ∼500 hours.

LLVE training setup. We establish a comprehensive benchmark on OLiVES using a wide range of state-of-the-art (SOTA) methods for both LLIE and LLVE. For a fair comparison, supervised methods [43, 41, 2, 40, 49, 44, 24, 32, 20, 45], and unsupervised methods [51, 52, 25, 26] are trained on OLiVES with the settings from the original works. Notably, GT is incorporated as input priors in [21] while excluded from the loss function during training; thus, we categorize it as an unpaired method.

VOS training setup. We employed several state-of-the-art semi-supervised VOS methods for our low-light VOS task, including STCN [7], AOT [48], De-AOT [47], XMem [6], and Cutie [8]. For fair comparison, we considered only architectures using a ResNet-50 [17] feature extractor. All methods were initialized using the authors’ publicly released weights pretrained on DAVIS 2017 [36] and YouTube-VOS [42], and then finetuned on our training set for 30,000 iterations using a batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 4 }$ The only exception was OASIS [37], for which only ResNet-18 pre-trained weights were publicly available. To ensure consistency, we first re-trained OASIS following its original training protocol on DAVIS 2017 and YouTube-VOS, before finetuning it on our dataset using the same settings as the other methods.

## 4.2 Low-Light Video Enhancement

For the evaluation metrics, we adopt PSNR, SSIM, LPIPS [50] to measure the image restoration quality. To further assess video quality and temporal consistency, we employ tLPIPS [9], Warping Error (WE) [22], and VMAF [28]. The evaluate results on our test set is reported in Table 4.

We observe a clear performance gap between supervised and unsupervised approaches, and between image-based and video-based methods: (i) Image-based methods generally achieve higher PSNR and VMAF scores than video-based methods (e.g., Retinexformer [2]), but often exhibit weaker temporal consistency, as reflected by higher tLPIPS and WE values. In contrast, video-based methods produce more temporally stable results; for example, BVI-CDM [32] and BVI-Mamba [20] achieve lower tLPIPS and competitive WE scores, albeit at the cost of reduced contrast. (ii) Unsupervised and unpaired methods exhibit a substantial performance drop compared to supervised approaches across all metrics. For example, [52] and [51] achieve lower PSNR and higher LPIPS, highlighting the difficulty of learning robust enhancement without paired supervision on our dataset.

Overall, methods trained on OLiVES struggle under severe degradations. While some approaches perform well on individual metrics, no single method consistently achieves strong results across both quality and temporal measures. This indicates the value of OLiVES as a challenging benchmark and impresses the need for joint optimization of restoration quality and temporal consistency. Further qualitative analysis is provided in Appendix B.

## 4.3 Video Object Segmentation

We provide a quantitative evaluation of several state-of-the-art VOS methods trained on our dataset, and assess their performance on our validation and test sets, as well as the LLE-VOS [23] validation set, to demonstrate the generalization of our training data across low-light benchmarks. We use standard VOS evaluation metrics [35, 36]: Jaccard Index (J), F-measure (F) and J&F. Results are reported in Table 5. We report that [8] and [37] show the strongest performance on OLiVES and LLE-VOS, while [47] achieves the second-strongest performance on LLE-VOS. STCN [7] performs poorly on LLE-VOS, suggesting that its earlier architecture struggles to generalize across domains.

We further evaluate the VOS performance of four LLVE methods [2, 20, 45] trained on OLiVES within a two-stage pipeline, where enhancement is applied as a pre-processing step before inference with a pre-trained VOS model. For this experiment, we use Cutie with off-theshelf weights for segmentation after enhancement. Evaluation on the LLE-VOS validation set assesses the effectiveness of our training data for LLVE; results are reported in Table 6.

Table 6: Quantitative results of two-stage methods on LLE-VOS validation set, using LLVE methods trained on OLiVES and pre-trained Cutie [8].
<table><tr><td>LLVE Method</td><td>LLE-VOS [23] val  $\mathcal { I } \& \mathcal { F }$ </td><td>J</td><td> $\mathcal { F }$ </td></tr><tr><td>Cutie (off-the-shelf) [8]</td><td>53.2</td><td>58.5</td><td>48.0</td></tr><tr><td>LLVE-STCD [45] + Cutie</td><td>57.9</td><td>62.1</td><td>53.8</td></tr><tr><td>BVI-Mamba  $[ 2 0 ] + \mathrm { C u t i e }$ </td><td>58.1</td><td>62.0</td><td>53.8</td></tr><tr><td>RetinexFormer  $[ 2 ] + \mathrm { C u t i e }$ </td><td>58.3</td><td>61.9</td><td></td></tr><tr><td></td><td></td><td></td><td>54.6</td></tr></table>

We find that all VOS methods perform well even without fine-tuning on our dataset (see Appendix D), indicating strong object detectability. However, segmentation quality remains limited, with inaccurate boundaries and missing fine details (e.g., animal fur). This suggests that, while current methods are suitable for detection-oriented applications such as surveillance, they fall short for tasks requiring fine-grained detail, such as film post-production, augmented reality, and robotics manipulation, where accurate boundaries and fine structures are critical. We further analyze this behavior in Appendix E.

## 5 Conclusion

We present OLiVES, a new low-light video dataset supporting both low-light video enhancement (LLVE) and video object segmentation (VOS). OLiVES introduces diverse challenges, including complex motion, rich natural textures (e.g., textures of animal furs), and a wide range of illumination conditions that are widely absent by existing low-light datasets. In particular, it provides new data for low-light VOS, featuring occlusions, emerging objects, and degraded appearances due to motion blur, without reliance on auxiliary modalities such as event data. By establishing a unified benchmark for both LLVE and VOS, OLiVES facilitates research that jointly advances perceptual quality and semantic understanding. We hope this dataset will encourage further work in low-light video analysis and inspire more robust methods for challenging real-world conditions.

## References

[1] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR, 2022.

[2] Yuanhao Cai, Hao Bian, Jing Lin, Haoqing Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In ICCV, 2023.

[3] Chen Chen, Qifeng Chen, Minh N Do, and Vladlen Koltun. Seeing motion in the dark. In Proceedings of the IEEE/CVF International conference on computer vision, pages 3185–3194, 2019.

[4] Kanghao Chen, Guoqiang Liang, Yunfan Lu, Hangyu Li, and Lin Wang. Evlight++: Low-light video enhancement with an event camera: A large-scale real-world dataset, novel method, and more. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(2):1608–1625, 2026. doi: 10.1109/TPAMI. 2025.3617801.

[5] Linwei Chen, Ying Fu, Kaixuan Wei, Dezhi Zheng, and Felix Heide. Instance segmentation in the dark. International Journal ofComputer Vision, 131(8):2198–2218, 2023.

[6] Ho Kei Cheng and Alexander G. Schwing. Xmem: Long-term video object segmentation with an atkinsonshiffrin memory model. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 640–658, Cham, 2022. Springer Nature Switzerland. ISBN 978-3-031-19815-1.

[7] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems, 2021. ISBN 9781713845393.

[8] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3151–3161, June 2024.

[9] Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taixé, and Nils Thuerey. Learning temporal coherence via self-supervision for gan-based video generation. ACM Transactions on Graphics (TOG), 39(4):75–1, 2020.

[10] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. MOSE: A new dataset for video object segmentation in complex scenes. In ICCV, 2023.

[11] Henghui Ding, Kaining Ying, Chang Liu, Shuting He, Xudong Jiang, Yu-Gang Jiang, Philip HS Torr, and Song Bai. MOSEv2: A more challenging dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2508.05630, 2025.

[12] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, and Michael Felsberg. RoMa: Robust dense feature matching. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19790–19800, June 2024.

[13] Huiyuan Fu, Wenkai Zheng, Xicong Wang, Jiaxuan Wang, Heng Zhang, and Huadong Ma. Dancing in the dark: A benchmark towards general low-light video enhancement. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 12877–12886, October 2023.

[14] Ying Fu, Zichun Wang, Tao Zhang, and Jun Zhang. Low-light raw video denoising with a high-quality realistic motion dataset. IEEE Transactions on Multimedia, pages 1–13, 2022. doi: 10.1109/TMM.2022. 3233247.

[15] Yuxuan Gu, Haoxuan Wang, Pengyang Ling, Zhixiang Wei, Huaian Chen, Yi Jin, and Enhong Chen. Improving visual and downstream performance of low-light enhancer with vision foundation models collaboration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16071–16080, June 2025.

[16] Chunle Guo Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proceedings ofthe IEEE conference on computer vision and pattern recognition (CVPR), pages 1780–1789, June 2020.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.

[18] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[19] Paul Hill, Zhiming Liu, Alin Achim, Dave Bull, and Nantheera Anantrasirichai. DMAT: An end-to-end framework for joint atmospheric turbulence mitigation and object detection. In IEEE/CVF WACV, pages 2690–2699, 2026.

[20] G. Huang, R. Lin, Y. Li, D. Bull, and N. Anantrasirichai. Bvi-mamba: Video enhancement using a visual state-space model for low-light and underwater environments. In Machine Learning from Challenging Data 2025, 2025.

[21] Hai Jiang, Ao Luo, Xiaohong Liu, Songchen Han, and Shuaicheng Liu. Lightendiffusion: Unsupervised low-light image enhancement with latent-retinex diffusion models. In European Conference on Computer Vision, 2024.

[22] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang. Learning blind video temporal consistency. In European Conference on Computer Vision, 2018.

[23] Hebei Li, Jin Wang, Jiahui Yuan, Yue Li, Wenming Weng, Yansong Peng, Yueyi Zhang, Zhiwei Xiong, and Xiaoyan Sun. Event-assisted low-light video object segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3250–3259, June 2024.

[24] Wenhao Li, Guangyang Wu, Wenyi Wang, Peiran Ren, and Xiaohong Liu. Fastllve: Real-time low-light video enhancement with intensity-aware lookup table. In Proceedings of the 31th ACM International Conference on Multimedia, 2023.

[25] Y. Li and N. Anantrasirichai. Zero-tig: Temporal consistency-aware zero-shot illumination-guided low-light video enhancement. In EUSIPCO, 2025.

[26] Yini Li, Louis Forster, David Bull, and Nantheera Anantrasirichai. Tempretinex: Retinex-based unsupervised enhancement for low-light video under diverse lighting conditions. In 2026 ICME, 2026. URL https://arxiv.org/abs/2511.09609.

[27] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2041–2050, 2018. doi: 10.1109/CVPR.2018.00218.

[28] Zhi Li, Anne Aaron, Ioannis Katsavounidis, Anush Krishna Moorthy, and Mehdi Manohara. Toward a practical perceptual video quality metric. In Netflix Tech Blog, volume 6, 2016.

[29] Joanne Lin, Nantheera Anantrasirichai, and David Bull. Multi-scale denoising in the feature space for low-light instance segmentation. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025. doi: 10.1109/ICASSP49660.2025.10889336.

[30] Joanne Lin, Ruirui Lin, Yini Li, David Bull, and Nantheera Anantrasirichai. ELVIS: Enhance low-light for video instance segmentation in the dark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026.

[31] Ruirui Lin, Nantheera Anantrasirichai, Guoxi Huang, Joanne Lin, Qi Sun, Alex Malyugina, and David Bull. BVI-RLV: A fully registered dataset and benchmarks for low-light video enhancement. arXiv preprint arXiv:2407.03535, 2024.

[32] Ruirui Lin, Qi Sun, and Nantheera Anantrasirichai. Low-light video enhancement with conditional diffusion models and wavelet interscale attentions. In Proceedings of21st ACM SIGGRAPH Conference on Visual Media Production, 2024.

[33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014, pages 740–755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.

[34] Zhiming Liu, Paul Hill, and Nantheera Anantrasirichai. JDATT: A joint distillation framework for atmospheric turbulence mitigation and target detection. In Proceedings of the British Machine Vision Conference (BMVC), 2025.

[35] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition, 2016.

[36] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv:1704.00675, 2017.

[37] Guanyi Qin, Ziyue Wang, Daiyun Shen, Haofeng Liu, Hantao Zhou, Junde Wu, Runze Hu, and Yueming Jin. Structure matters: Revisiting boundary refinement in video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14431–14442, October 2025.

[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Ha6RTeWMd0.

[39] Shangquan Sun, Wenqi Ren, Tao Wang, and Xiaochun Cao. Rethinking image restoration for object detection. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 4461–4474. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 1cac8326ce3fbe79171db9754211530c-Paper-Conference.pdf.

[40] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. Seeing dynamic scene in the dark: High-quality video dataset with mechatronic alignment. In ICCV, 2021.

[41] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex C Kot. Low-light image enhancement with normalizing flow. In AAAI Conference on Artificial Intelligence, 2022.

[42] Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. YouTube-VOS: Sequence-to-sequence video object segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

[43] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In CVPR, 2022.

[44] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Deep parametric 3d filters for joint video denoising and illumination enhancement in video super resolution. In AAAI, 2023.

[45] Xiaogang Xu, Kun Zhou, Tao Hu, Jiafei Wu, Ruixing Wang, Hao Peng, and Bei Yu. Low-light video enhancement via spatial-temporal consistent decomposition. In IJCAI, 2025.

[46] Linjie Yang, Yuchen Fan, and Ning Xu. The 2nd large-scale video object segmentation challenge - video object segmentation track, October 2019.

[47] Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

[48] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

[49] Fan Zhang, Yu Li, Shaodi You, and Ying Fu. Learning temporal consistency for low light video enhancement from single images. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4967–4976, June 2021.

[50] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[51] Xiaofeng Zhang, Zishan Xu, Hao Tang, Chaochen Gu, and Wei Chen. Wakeup-darkness: When multimodal meets unsupervised low-light image enhancement. ACM Transactions on Multimedia Computing, Communications and Applications, 2025.

[52] S. Zheng and G. Gupta. Semantic-guided zero-shot learning for low-light image/video enhancement. In IEEE/CVF WACVW, pages 581–590, 2022.

## A Details of RoMA finetuning

Synthetic low-light data. Following Lin et al. [30], we model degradation of one clean normal-light image $I _ { g t }$ as:

$$
I _ { l o w } ^ { ' } = D e g ( I _ { g t } , \phi ) = H * ( 2 ^ { \epsilon } I _ { g t } ) + N ,\tag{1}
$$

where ∗ is the 2-D convolution, ϕ denotes degradation parameters including exposure adjustment $\epsilon ,$ a point spread function kernel $H$ , and physics-based noise map N. Here we denote Φ as a set of ϕ estimated from multiple real low-light video frames using VDPNet [30]. Since different ND filters produce distinct brightness and noise distributions, we further construct degradation parameter sets:

$$
\Phi = \{ \Phi _ { 0 } , \Phi _ { 1 . 2 } , \Phi _ { 1 . 5 } , \Phi _ { 1 . 8 } \} ,\tag{2}
$$

while $\Phi _ { 0 }$ corresponds to the identity mapping.

Degradation-augmented training. RoMa consists of a frozen coarse encoder $\mathcal { F } _ { \mathrm { c o a r s e } }$ and a VGG19- based fine encoder ${ \mathcal { F } } _ { \mathrm { f i n e } }$ , where ${ \mathcal { F } } _ { \mathrm { c o a r s e } }$ extracts global correspondences and $\mathcal { F } _ { \mathrm { f i n e } }$ refines local matches. We observe that low-light degradation mainly affects ${ \mathcal { F } } _ { \mathrm { f i n e } } .$ To validate this, we conduct an ablation study on MegaDepth-1500 [27] using the pretrained RoMa model, where we replace either the coarse or fine features with those extracted from degraded images. As shown in Table 7, replacing fine features results in a substantial drop across all metrics.

Therefore, we only finetune ${ \mathcal { F } } _ { \mathrm { f i n e } }$ and the following refinement module $\mathcal { R }$ . During training, given an image pair $( I _ { A } , I _ { B } )$ , we randomly apply degradation:

$$
I _ { B } ^ { ' } = D e g ( I _ { B } , \phi _ { 1 } ) , \phi _ { 1 } \sim \{ \Phi _ { 1 . 2 } , \Phi _ { 1 . 5 } , \Phi _ { 1 . 8 } \} ,\tag{3}
$$

and use $( I _ { A } , I _ { B } ^ { ' } )$ as input for training. Following [12], we use MegaDepth [27] as training set.

Table 7: Ablation study on the impact of coarse and fine features in RoMa. We replace the corresponding features in the pretrained model with those extracted from degraded inputs.
<table><tr><td>Method</td><td>AUC@5</td><td>AUC@10</td><td>AUC@20</td></tr><tr><td>Pretrained</td><td>62.28</td><td>76.45</td><td>86.25</td></tr><tr><td>Replace Coarse Features</td><td>62.19</td><td>76.35</td><td>86.15</td></tr><tr><td>Replace Fine Features</td><td>52.33</td><td>68.68</td><td>80.79</td></tr></table>

Degradation-invariant feature learning. To further enhance robustness, we enforce degradation invariant feature representations. Specifically, we generate an additional degraded sample and extract features as follow:

$$
I _ { B } ^ { ' \prime } = D e g ( I _ { B } , \phi _ { 2 } ) , \phi _ { 2 } \sim \Phi , \phi _ { 2 } \neq \phi _ { 1 } ,\tag{4}
$$

$$
\psi _ { B } ^ { ' \prime } = \mathcal { F } _ { \mathrm { f i n e } } ( I _ { B } ^ { ' \prime } ) , \quad \psi _ { B } ^ { ' } = \mathcal { F } _ { \mathrm { f i n e } } ( I _ { B } ^ { ' } ) .\tag{5}
$$

We encourage consistency between $\psi _ { B } ^ { ' }$ and $\psi _ { B } ^ { \prime \prime }$ using contrastive learning. Inspired by VICReg [1], we introduce an MLP predictor P to obtain projected feature embeddings in Eq. (6). We then apply the VICReg loss to enforce feature consistency and prevent representational collapse:

$$
\begin{array} { r } { \mathbf { z } _ { B } ^ { ' } = \mathcal { P } ( \psi _ { B } ^ { ' } ) , \quad \mathbf { z } _ { B } ^ { ' \prime } = \mathcal { P } ( \psi _ { B } ^ { ' \prime } ) . } \end{array}\tag{6}
$$

$$
\mathcal { L } _ { \mathrm { V I C R e g } } = \| \mathbf { z } _ { B } ^ { ' } - \mathbf { z } _ { B } ^ { ' \prime } \| _ { 2 } ^ { 2 } + \sum _ { j } \operatorname* { m a x } ( 0 , \gamma - \sigma ( \mathbf { z } _ { B } ^ { ' } ) _ { j } ) + \sum _ { i \neq j } \mathrm { C o v } ( \mathbf { z } _ { B } ^ { ' } ) _ { i j } ^ { 2 } ,\tag{7}
$$

where $\boldsymbol { \sigma } ( \mathbf { z } ) _ { j }$ denotes the standard deviation of the j-th feature dimension over the batch, γ denotes a minimum standard deviation, and Cov $( \mathbf { z } ) _ { i j }$ denotes the (i, j) entry of the covariance matrix of the projected features.

Following [18], we adopt an InfoNCE to improve instance-level discriminability in a batch of size K:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { \exp ( ( z _ { B } ^ { ' T } \cdot z _ { B } ^ { ' \prime } ) / 0 . 0 7 ) } { \sum _ { k = 0 } ^ { K } \exp ( ( z _ { B } ^ { ' T } \cdot z _ { B , k } ^ { ' \prime } ) / 0 . 0 7 ) } ,\tag{8}
$$

Both InfoNCE and VICReg losses are added to the original model loss. Notably, $I _ { B } ^ { \prime \prime }$ is only used to improve feature robustness during training and not used for the final matching.

Results. All the training and testing experiments are conducted on a single NVIDIA H100 GPU on the compute platform described in Section 4.1. Training duration is approximately 22 hours. We analysis finetuned performances on MegaDepth-1500 [27] where the area under the cumulative curve (AUC) of Essential matrix error at thresholds (5<sup>◦</sup>, 10<sup>◦</sup>, 20<sup>◦</sup>) are reported, as shown in Table 8. Degradation augmentation (I) improves performance over all metrics. Adding VICReg (II) further boosts performance under degradation, while maintaining comparable results on non-degraded data Incorporating InfoNCE (III) achieves the best results across all settings, improving both degraded and non-degraded performance. Finally, we select model III for our dataset registration. More implemented details and code can be referred to https://github.com/anonymous-OLiVES/ RoMa\_Finetune4OLiVES.

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \mathbf { z } _ { i } ^ { \prime \top } \mathbf { z } _ { i } ^ { \prime \prime } / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( \mathbf { z } _ { i } ^ { \prime \top } \mathbf { z } _ { j } ^ { \prime \prime } / \tau ) }\tag{9}
$$

## B LLVE Qualitative Results

We visualize the testing results of OLiVES on three representative SOTA methods (Retinexformer [2], BVI-Mamba [20], LLVE-STCD [45]). To further evaluate the effectiveness of our dataset, we assess the cross-dataset generalization by testing on DID [13], SDSD [40], BVI-RLV [31]. As shown in Fig. 6, models trained on OLiVES produce visually pleasing results with improved brightness and detail restoration. These results demonstrate that OLiVES enables models to learn more robust and transferable representations, leading to strong cross-dataset generalization and highlighting the effectiveness of our dataset.

Table 8: Finetuning performance of RoMa on non-degraded and degraded MegaDepth-1500. All values (except the first row) indicate absolute improvements over the pretrained baseline.
<table><tr><td rowspan="2">Method</td><td colspan="4">Non-degraded</td><td rowspan="2">Degraded AUC@10</td><td rowspan="2">AUC@20</td></tr><tr><td>AUC@5</td><td>AUC@10</td><td>AUC@20</td><td>AUC@5</td></tr><tr><td>Pretrained</td><td>62.28</td><td>76.45</td><td>86.25</td><td>54.75</td><td>70.42</td><td>82.04</td></tr><tr><td>I: Deg. Aug.</td><td>+0.34</td><td>+0.11</td><td>+0.04</td><td>+2.43</td><td>+2.06</td><td>+1.46</td></tr><tr><td>II: I+VICReg</td><td>+0.10</td><td>-0.03</td><td>-0.06</td><td>+2.74</td><td>+2.17</td><td>+1.49</td></tr><tr><td>III: II+InfoNCE</td><td>+0.56</td><td>+0.38</td><td>+0.17</td><td>+3.12</td><td>+2.44</td><td>+1.66</td></tr></table>

![](images/527c5e1733a1f21a0f45a5b391f335661c7332ad6a21135ebf2d3a9e7de2f12e.jpg)  
Figure 6: Qualitative results of LLVE methods trained on OLiVES.

## C Details of VOS Annotation Process

We recruited 9 annotators to manually produce ground-truth labels for the OLiVES test set by using our custom annotation software. We provided annotators with both the normal and low-light frames; the normal-light frames were for initial annotations and the low-light frames were to verify that they are correct. To ensure consistency and reduce ambiguity, annotators were provided with the first-frame object masks indicating when each instance appears in the sequence. We also provided them with custom annotation software, alongside detailed training materials (video tutorial and written instructions). The software included sliders to adjust brightness, contrast and saturation to help with verifying masks in low-light. We provide a screenshot of the tool in Fig. 7.

![](images/e9c7f390466011e2574c03cad870329c3a7a37a81c651597d70560f6e69ebf9f.jpg)  
[Space: Toggle Mode |M: Toggle Mask| Ctrl+Z/Y: Undo/Redo |Mouse Wheel: Zoom |Middle/Right-click+Drag: Pan|←/→: Navigate]

Figure 7: Custom annotation software to create manual annotations for OLiVES VOS benchmark.  
![](images/e01551b28c379542277d6be5c69a8e416ed8d23788f11a91399a85771349f21f.jpg)

![](images/155b9e5d697eef55de335eaaa8345125b6db46542ec314eec0cdcdd67662cd5d.jpg)  
Figure 8: Qualitative comparison of Cutie on an example video from MOSEv2 [11] (reduced frame brightness for better segmentation visibility). (Left) the first frame with the input masks. (Right) Crops of segmentation outputs for Cutie [8] with various training data.

Annotations were collected on a weekly basis, allowing us to perform regular quality assessments and provide targeted feedback. Annotators who consistently failed to meet quality standards were replaced, and their assigned videos were re-annotated by other workers. Following this initial stage, we introduced a second stage in which annotators reviewed and refined sequences they had not originally labeled, helping to improve consistency across the dataset. Finally, we conducted an additional round of review to perform minor corrections and ensure a uniformly high standard of annotation quality. All annotators were compensated at rates above the local minimum wage.

## D Further VOS Experiments

Cross-domain comparison. Furthermore, to show the cross-domain capabilities of our training data for low-light VOS tasks, we compare Cutie [8] with its off-the-shelf weights (DAVIS 2017 [36] and YouTube-VOS [42]), finetuned on LLE-VOS [23] and finetuned on OLiVES. We run inference on an example low-light video from the MOSEv2 [11] dataset and visualize our results in Fig. 8.

Table 9: Quantitative results on OLiVES test set, grouped by size of the objects. Count is the total count of the object sizes.
<table><tr><td rowspan="2">Size</td><td rowspan="2">Count</td><td colspan="3">OLiVES test</td></tr><tr><td>J&amp;F</td><td>JF</td><td></td></tr><tr><td>Small</td><td>5</td><td></td><td></td><td></td></tr><tr><td>Medium</td><td>40</td><td></td><td></td><td></td></tr><tr><td>Large</td><td>109</td><td></td><td></td><td></td></tr></table>

![](images/2e22854693dc354b3104c171e445e097d0278da74f6e3615639d389100096d55.jpg)

![](images/ed0bb257be6679d4caa54da9bf9c3ccff8b9425eb48ace9bfb19ee2e62d3986b.jpg)  
Figure 9: Examples of difficult segmentation scenarios in OLiVES.

Here, we find using the pre-trained weights, Cutie incorrectly produces segmentation masks for the people in the distance despite being occluded by a person in the foreground, while the finetuned model on LLE-VOS performed better with the occlusion but still produced masks in frame 5 despite the person not reappearing in the scene. Meanwhile, the model finetuned on OLiVES was able to correctly produce no segmentation masks when occluded.

## E Investigation of High VOS performance

To determine the cause of the high VOS performance, we group the objects by size and calculate the common VOS metrics: J &F, J and F. We collect these results using the pre-trained Cutie [8] (to avoid bias to the dataset) and present them in Table 9. We find that performances vary for small and medium objects but large objects have significantly higher scores. We can determine that the variance in performance for small and medium objects is due to the low count of both sizes.

However, closer inspection reveals that even when trained on our dataset, Cutie fails to capture fine-grained textures (see Fig. 9). In these examples, we can see that Cutie struggled to segment the details of the fur of the animals, as well as handling occlusions. While such performance may be sufficient for surveillance applications, it falls short of the precision required for post-production tasks, where high-quality, finely detailed masks are essential.

We conclude that due to the unbalanced size distribution of our dataset (refer to Fig. 5 (d)) and the bias of models on large objects, this results in numerically high VOS performances.