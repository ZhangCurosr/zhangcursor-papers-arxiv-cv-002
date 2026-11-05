# Navigational Safety of Marine Surface Vehicles in Restricted Visibility: From Intelligent Vision to Vision-Centered Multi-Sensor Fusion

Ryan Wen Liu<sup>a,b,c,d</sup>, Fengchuan Song<sup>b,d</sup>, Tianxiang Wang<sup>b,d</sup>, Guobei Peng<sup>b,d</sup>, Youngrong Kim<sup>e</sup>, Xian Zhong<sup>a,d,f</sup>, Hooi-Siang Kang<sup>g</sup>, Maohan Liang<sup>a,b,d,∗</sup>, Junzhi Yu<sup>h,∗</sup> and Shanliang Yao<sup>b,i,∗</sup>

<sup>a</sup>Sanya Science and Education Innovation Park ofWuhan University ofTechnology, Sanya 572025, China

<sup>b</sup>School ofNavigation, Wuhan University ofTechnology, Wuhan 430063, China

<sup>c</sup>School ofMarine Science and Technology, Wuhan University ofTechnology, Sanya 572025, China

<sup>dState</sup> <sup>Key</sup> <sup>Laboratory</sup> <sup>of</sup> <sup>Maritime</sup> <sup>Technology</sup> <sup>and</sup> <sup>Safety,</sup> <sup>Wuhan</sup> <sup>430063,</sup> <sup>China</sup>6

<sup>e</sup>School ofCivil and Environmental Engineering, Nanyang Technological University, 639798, Singapore2

f<sub>Hu</sub>b<sub>ei</sub> <sub>Key</sub> <sub>La</sub>b<sub>oratory</sub> <sub>o</sub>f <sub>Transportation</sub> <sub>Internet</sub> <sub>o</sub>f <sub>T</sub>h<sub>ings,</sub> <sub>Sc</sub>h<sub>oo</sub>l <sub>o</sub>f <sub>Arti</sub>fi<sub>cia</sub>l <sub>Inte</sub>ll<sub>igence,</sub> <sub>Wu</sub>h<sub>an</sub> <sub>University</sub> <sub>o</sub>f <sub>Tec</sub>h<sub>no</sub>l<sub>ogy,</sub> <sub>Wu</sub>h<sub>an</sub> 4300<sub>7</sub>0<sub>,</sub> <sub>C</sub>h<sub>ina</sub>0   
g<sub>Faculty</sub> <sub>of</sub> <sub>Mechanical</sub> <sub>Engineering,</sub> <sub>Universiti</sub> <sub>Teknologi</sub> <sub>Malaysia,</sub> <sub>Johor</sub> <sub>Bahru,</sub> <sub>Malaysia</sub>2

<sup>h</sup>Department of Advanced Manufacturing and Robotics, College of Engineering, Peking University, Beijing 100871, Chinav <sup>i</sup>School ofInformation Engineering, Yancheng Institute ofTechnology, Yancheng 224051, Chinao

## A R T I C L E I N F O

Keywords: Marine surface vehicles Navigational safety Intelligent vision Multi-sensor fusion<sup>.</sup> Autonomous navigation

## A BS T RA C T

Marine surface vehicles (MSVs) have attracted considerable attention for their commercial, scientific, and military applications. Owing to their favorable performance-to-cost ratio, shipboard cameras are widely used to detect and recognize potentially hazardous situations across diverse operational scenarios. However, visual perception is inevitably degraded by restricted conditions, including irregular motion of the imaging platform, adverse weather, and long imaging distances. To improve situational awareness and navigation capabilities, complementary sensors, such as the automatic identification system (AIS), radar, and light detection and ranging (LiDAR), etc., have become essential shipboard equipment. Multi-sensor fusion can provide richer perceptual information and help ensure navigational safety of MSVs in restricted visibility. This article reviews recent progress from intelligent vision to vision-centered multi-sensor fusion for newcomers and researchers in intelligent ship navigation and intelligent waterborne transportation system (IWTS). After introducing the generation of quality-degraded images in restricted visibility, we will review intelligent vision methods for four key issues, i.e., shipboard video stabilization, visual perception enhancement, object detection and tracking, and navigable area segmentation. We then examine vision-centered multi-sensor fusion methods, including infrared-visible image fusion, radar-vision fusion, LiDAR-vision fusion, AIS-vision fusion, and other multi-sensor fusion scenarios. Open-source datasets and safety-critical applications are also summarized and discussed. Finally, we highlight open challenges and future research directions. The collected maritime datasets are available at https://github.com/MAPGroup/Datasets and will be regularly updated.

## 1. Introduction

## 1.1. Background

Marine surface vehicles (MSVs), including unmanned surface vehicles (USVs) and autonomous surface vehicles (ASVs), have attracted considerable attention from both academia and industry, as illustrated in Fig. 1. Rapid advances in perception, communication, control, and artificial intelligence (AI) have enabled higher levels of safety, security, and eficiency in complex maritime environments (Forti et al., 2022). Consequently, MSVs can perform dangerous, dull, and dirty tasks with minimal or no human intervention, supporting applications such as environmental monitoring, hydrologic surveys, search and rescue, military reconnaissance, and defense. However, the corresponding operational efectiveness will be severely afected by adverse weather conditions (e.g., haze, rain, snow, and low-lightness, etc.), which significantly reduce the perception range, accuracy, and reliability for MSVs in practical applications. To guarantee reliable and safe navigation in restricted visibility, many eforts have focused on how to optimally select the shipborne sensors, which enable MSVs to increase the situational awareness capacity. In addition, how to develop the suitable and efective computing methods, related to shipborne sensors, is another long-standing challenge which urgently needs to be handled. The corresponding computing methods should exploit the full advantages of diferent sensors, and then provide high-level intelligence, high-precision perception, and high-reliability decision-making for MSVs in dynamic maritime conditions.

![](images/f9de22090b23c38dba8b1972c6111efc86acecd9cbb629b737148679a125af04.jpg)  
Figure 1: Fundamental architecture of a typical MSV. Visual illustration of an MSV equipped with multiple sensors, a ground station, GNC (guidance, navigation and control) system, communication system, propulsion and power system. Please refer to Ref. (Liu et al., 2016) for more details on the typical elements of MSV, i.e., sensors (a.k.a., sensing data collection equipment), hull and auxiliary structural elements, propulsion and power system, GNC system, communication system, ground station, etc.

In unknown and complex environments, the operational capabilities of MSVs, including situational awareness, decision-making, and control, depend heavily on shipborne sensors, such as visible-light and thermal infrared cameras, automatic identification system (AIS), radar, and light detection and ranging (LiDAR), etc. Visible-light camera, which operates similarly to human vision, provides detailed red-green-blue (RGB) images. However, the imaging quality will be significantly degraded under low-visibility weathers. In contrast, thermal infrared camera could capture the shapes of objects of interest through their emitted infrared radiation. Owing to the capacity of nighttime target detection, it has played a significant role in enhancing the navigational safety of MSVs. AIS, an automated vessel self-reporting system, receives both static information (e.g., the maritime mobile service identity (MMSI), vessel name, length, and breadth) and dynamic information (e.g., position, course, and speed) from surrounding vessels through very-high-frequency (VHF) radio signals (Liu et al., 2021c). It has been widely applied to trafic-risk analysis, collision avoidance, and route planning (Liu et al., 2022b). Radar detects surrounding surface vehicles and provides their bearings and distances for collision avoidance and navigation. In particular, S-band radar is able to robustly detect targets during poor weathers (e.g., hazy, snow and rain, etc.) since it has a longer wavelength. However, it ofers lower resolution for detecting small targets. X-band radar performs well in detecting small targets (e.g., small ships and buoys, etc.) but will fail in poor visibility conditions. In contrast, LiDAR uses laser rather than radio waves for short-range three-dimensional environmental perception, resulting in short-range high-precision surface measurements and 3D mapping capabilities. 4D millimeter wave (4D mmWave) radar has the capacity of measuring the range, azimuth, elevation, and velocity of targets (Yao et al., 2024b). It has attracted considerable attention in autonomous driving because of its all-weather adaptability and velocity-measuring capability. More details on the comparison of diferent shipborne sensors could be found in Table 1.

<table><tr><td colspan="8">Table 1: Comparisons of several shipboard sensors for MSVs. The visible-light camera, thermal infrared camera, AIS (automatic identification system), S-band radar, X-band radar, 3D LiDAR (light detection and ranging), and 4D mmWave (millimeter wave) radar are compared and analyzed in terms of data dimension spatial resolution, temporal resolution, detection range, cost, interference sources, strong points, and application scenarios. Note that the visible-light camera thermal infrared camera, and AIS belong to passive sensors. In contrast, the S/X-band radar, 3D LiDAR, and 4D mmWave radar are active sensors, which perceive their environment through the actively emitted electromagnetic waves.</td></tr><tr><td>Sensors</td><td>Data Dimension</td><td>Spatial Resolution</td><td>Temporal Resolution Range</td><td>Detection</td><td>Cost Sources</td><td>Interference</td><td>Strong Points</td><td>Application Scenarios</td></tr><tr><td>Visible-Light Camera</td><td>2D planar image</td><td>(1K/2K/4K)</td><td>High</td><td>Medium</td><td>Low- Medium</td><td>attenuation, flare/glare, etc.</td><td></td><td>High-resolution images Daytime target detection, short-range channel monitoring, berthing/unberthing visual assistance, target identity verification</td></tr><tr><td>Thermal Infrared 2D thermal Camera</td><td>radiation image</td><td>Pixel-level (640×512)</td><td>High</td><td>Medium</td><td>High</td><td>Medium- High-temperature background Almost all-weather interference, heavy rainfall attenuation, etc.</td><td>detection capability</td><td>Nighttime target detection, auxiliary observation in adverse weather, maritime search and rescue</td></tr><tr><td>AIS</td><td>1D attribute data</td><td>~10 m (depends on GNSS)</td><td>Low</td><td>Long</td><td>Low</td><td>VHF electromagnetic interference, AIS disabled by vessels, false signals, etc.</td><td>largely unaffected by adverse weather</td><td>Collision avoidance and Vessel identity recognition, situational awareness, collision avoidance, etc.</td></tr><tr><td>S-band Radar (2-4 GHz)</td><td>2D planar plots/tracks (range, azimuth)</td><td>Range resolution 10–30 m; azimuth resolution 1°-2.3°</td><td>High</td><td>Long</td><td>High</td><td>Medium— Sea clutter, rain attenuation, same-frequency interference, etc.</td><td>Robust long-range target detection in adverse weather</td><td>Medium- to long-range wide-area detection, ocean-going collision avoidance, maritime surveillance, etc.</td></tr><tr><td>X-band Radar (8-12 GHz)</td><td>2D planar plots/tracks (range, azimuth)</td><td>Range resolution 1–15 m; azimuth resolution 0.5°–1°</td><td>High</td><td>Long</td><td>Medium</td><td>Sea clutter, rain/snow attenuation, same-frequency interference, etc.</td><td>Smaller antenna and higher target resolution high-precision detection</td><td>Short- to medium-range small-target identification, collision avoidance in narrow channels, etc.</td></tr><tr><td>3D LiDAR</td><td>3D point cloud (range, azimuth, elevation, reflection intensity)</td><td>Range resolution ~2 High cm; angular resolution 0.01°-0.05°</td><td></td><td>Short- Medium</td><td>High</td><td>Rain/snow/haze attenuation, strong light interference, water surface reflection, etc.</td><td>measurements and 3D mapping</td><td>Highly accurate surfaceShort-range high-precision 3D modeling, berthing/unberthing assistance, etc.</td></tr><tr><td>4D mmWave Radar</td><td>4D point cloud (range, azimuth, elevation, radial velocity)</td><td>Range resolution 0.1–2 m; azimuth resolution 1°-3°</td><td>High</td><td>Short</td><td>High</td><td>Medium— Same-band millimeter-wave interference, heavy rainfall attenuation, etc.</td><td>Almost all-weather operation and robust perception</td><td>Short- to medium-range multi-target detection, collision avoidance in adverse weather, etc.</td></tr></table>

![](images/d72746ba2335df71285757c0028745347c2a03e4cf06acc5eac8bd5ff4cbe4e5.jpg)  
Figure 2: Organizational structure of our work. The image generation, intelligent vision, and vision-centered multi-sensor fusion methods, as well as the maritime datasets and safety-critical practical applications are firstly discussed. We then analyze the open challenges and future research directions in this work.

Owing to the favorable performance-to-cost ratio and installation convenience, visible-light camera has been widely exploited to support safe navigation. However, its imaging quality is readily degraded by adverse weather and illumination conditions, including low illumination, haze, snow, and rain (Cao et al., 2026). Accordingly, visual perception enhancement methods have been extensively studied to improve navigational safety of MSVs in restricted visibility. With the rapid development of artificial intelligence (AI) and computer vision (CV) techniques, many advanced Visual computing methods, including maritime object detection, recognition, tracking, and segmentation, further help MSVs understand their surroundings in diverse imaging and weather conditions. More recently, multi-sensor data fusion has received increasing attention for exploiting complementary sensing capabilities. Because cameras provide rich color and semantic information, this survey mainly focuses on vision-centered multi-sensor data fusion, including infrared-visible image fusion, radar-vision fusion, LiDAR-vision fusion, and AIS-vision fusion, etc.

## 1.2. Motivation and Contributions

By jointly exploiting heterogeneous sensor data to form a more reliable representation of the navigational environment, multi-sensor data fusion has become a key enabling technology for reliable assisted and autonomous navigation. Substantial eforts have surveyed data fusion for autonomous driving, intelligent vehicles, and intelligent transportation systems. Existing studies have demonstrated that fusion strategies can improve perceptual accuracy and decision reliability, thereby enhancing system reliability, robustness, and safety. To the best of our knowledge, no systematic survey has jointly reviewed intelligent vision and vision-centered data fusion for ensuring the navigational safety of MSVs in complex maritime environments.

Compared with road trafic, maritime trafic is more complex and vulnerable because of adverse weather and illumination conditions, high vessel-trafic density, and the unstable waterborne operating environment. Despite rapid advances in intelligent shipping technologies, maritime accidents can cause environmental pollution, property damage, injury, and loss of life. To improve navigational safety, numerous AI-based computational methods have been developed to enhance intelligent visual perception. We therefore first review representative intelligent vision methods, including shipboard video stabilization, visual perception enhancement, object detection and tracking, and navigable area segmentation, shown in Fig. 2. These methods enhance visual perception in restricted visibility, particularly in narrow waterways, ports, and congested waters. We then survey vision-centered multi-sensor data fusion methods, including infrared-visible image fusion, radar-vision fusion, LiDAR-vision fusion, and AIS-vision fusion. Their reliability and applicability have been evaluated through real-world experiments reported in the literature.

Table 2  
Abbreviations used in this article.
<table><tr><td>Abbreviation</td><td>Full term</td><td>Abbreviation</td><td>Full term</td></tr><tr><td>AI</td><td>Artificial intelligence</td><td>CV</td><td>Computer vision</td></tr><tr><td>AIS</td><td>Automatic identification system</td><td>MMSI</td><td>Maritime mobile service identity</td></tr><tr><td>ASVs</td><td>Autonomous surface vehicles</td><td>USVs</td><td>Unmanned surface vehicles</td></tr><tr><td>MSVs</td><td>Marine surface vehicles</td><td>MASS</td><td>Maritime autonomous surface ship</td></tr><tr><td>IMU</td><td>Inertial measurement unit</td><td>VHF</td><td>Very-high-frequency</td></tr><tr><td>UHD</td><td>Ultra-high definition</td><td>mmWave</td><td>Millimeter-wave</td></tr><tr><td>FOV</td><td>Field of View</td><td>DOF</td><td>Degree-of-freedom</td></tr><tr><td>BEV</td><td>Bird&#x27;s-eye view</td><td>ROS</td><td>Robot operating system</td></tr><tr><td>LiDAR</td><td>Light detection and ranging</td><td>AR</td><td>Augmented-reality</td></tr><tr><td>GCN</td><td>Graph convolutional network</td><td>GAN</td><td>Generative adversarial network</td></tr><tr><td>HADAR</td><td>Heat-assisted detection and ranging</td><td>MLLM</td><td>Multimodal large language model</td></tr><tr><td>DCPA</td><td>Distance at the closest point of approach</td><td>TCPA</td><td>Time to closest point of approach</td></tr><tr><td>IWTS</td><td>Intelligent waterborne transportation system</td><td>mloT</td><td>Maritime Internet of Things</td></tr></table>

The main contributions of this work are summarized as follows:

• We present a structured survey that connects intelligent vision with vision-centered multi-sensor data fusion, providing a unified perspective on perception techniques for safe MSV navigation in complex maritime and restricted-visibility conditions.

• We systematically review maritime perception and multi-sensor fusion datasets and synthesize applications in behavior prediction, collision avoidance, and autonomous navigation, highlighting the connection between environmental perception and safety-critical navigation functions.

• We identify key open challenges and future research directions, including restricted-visibility benchmarks, robust imaging under severe degradation, small-object detection, uncertainty-aware multi-sensor fusion, and vision-language scene understanding.

This work is intended to provide a comprehensive reference for newcomers and researchers studying the navigational safety of unmanned and autonomous surface vehicles in restricted visibility.

## 1.3. Organization

The remainder of this paper is organized as follows. Section 2 introduces physical modeling and data-driven methods for generating visually degraded images in restricted visibility. Section 3 reviews recent studies on intelligent vision, and Section 4 analyzes vision-centered multi-sensor data fusion methods. Section 5 discusses the multi-modal datasets and safety-critical maritime applications. We further discuss the open challenges and future research directions in Section 6. Section 7 concludes the paper by summarizing its main contributions. Table 2 lists the principal abbreviations and acronyms used throughout this paper.

## 2. Image Generation in Restricted Visibility

Before reviewing intelligent vision and vision-centered multi-sensor data fusion, we introduce physics-based and data-driven methods for generating visually degraded images in restricted visibility. These methods can increase the volume and diversity of training data, thereby improving the robustness and generalization of intelligent vision and data fusion methods. As illustrated in Fig. 3, existing approaches can be broadly divided into physics-based and data-driven methods. Physics-based methods synthesize degradations according to imaging mechanisms, whereas data-driven methods learn clean-to-degraded mappings from large-scale data using generative models.

![](images/f97efb2871e4bedd16dba8df526a5ff224be3fe0ec34a6619107e274c55cd12f.jpg)  
Figure 3: Overview of visually degraded image generation methods. Existing methods are mainly divided into two categories, i.e., physical modeling and data-driven methods.

## 2.1. Physics-Based Computational Methods

Visually degraded images can be synthesized using the following physical imaging models.

## 2.1.1. Hazy Imaging Model

According to Koschmieder’s law (Galdran et al., 2015), a synthetic hazy image � can be formulated as

$$
g \left( \mathrm { x } \right) = f \left( \mathrm { x } \right) t \left( \mathrm { x } \right) + A \left( 1 - t \left( \mathrm { x } \right) \right) ,\tag{1}
$$

where $\mathbf { x } \in \Omega .$ , with Ω denoting the image domain; � is the latent haze-free image; and $A \in ( 0 , 1 ]$ and $t \in ( 0 ,$ , 1] denote the atmospheric light and transmission map, respectively. A lower transmission indicates more severe visual degradation, which can impair subsequent detection and tracking.

## 2.1.2. Low-Light Imaging Model

According to Retinex theory (Land, 1977), a synthetic low-light image � can be modeled as

$$
g \left( \mathrm { x } \right) = r \left( \mathrm { x } \right) \circ l \left( \mathrm { x } \right) ,\tag{2}
$$

where ◦ denotes pixel-wise multiplication, and � and � are the reflectance and illumination components, respectively. During training, images can be randomly transformed into low-light versions before each epoch using Eq. (2) with $l \in ( 0 , 1 )$

Gamma-correction-based degradation is another widely-used strategy for synthesizing low-light images, i.e.,

$$
g \left( \mathbf { x } \right) = f \left( \mathbf { x } \right) ^ { \gamma } + n \left( \mathbf { x } \right) ,\tag{3}
$$

where $\gamma$ is the gamma-correction coeficient and � denotes sensor noise.

## 2.1.3. Rainy or Snowy Imaging Model

Rain and snow degradations are commonly synthesized by adding weather-related degradation layers. A rainy image � can be formulated as

$$
g \left( \mathrm { x } \right) = f \left( \mathrm { x } \right) + \mathcal { R } \left( \mathrm { x } \right) ,\tag{4}
$$

where  denotes the rain-streak layer. This model assumes that rain streaks primarily introduce high-frequency texture interference through linear superposition.

Because snow particles produce stronger occlusion and translucency efects, a snowy image � can be modeled as

$$
g \left( \mathrm { x } \right) = f \left( \mathrm { x } \right) \left( 1 - S \left( \mathrm { x } \right) \right) + M \left( \mathrm { x } \right) S \left( \mathrm { x } \right) ,\tag{5}
$$

where  denotes the snow mask and � represents the snow chromatic mapping component.

## 2.1.4. Mixed Degradation Model

In real-world environments, visual degradation is often caused by coupled adverse factors, including haze scattering, low illumination, rain, and snow. Such complex conditions can be synthesized by cascading the corresponding elementary degradation operators into a unified physical model, i.e.,

$$
g \left( \mathrm { x } \right) = { { \mathcal { P } } _ { h } } \left( { { \mathcal { P } } _ { r s } } \left( { { \mathcal { P } } _ { l } } \left( f \left( \mathrm { x } \right) \right) \right) \right) ,\tag{6}
$$

where $\mathcal { P } _ { h } , \mathcal { P } _ { r s } ,$ and $\mathcal { P } _ { l }$ denote the haze, rain-or-snow, and low-light degradation operators, respectively. Cascading diferent degradation operators enables this formulation to represent complex adverse-weather imaging conditions more efectively.

## 2.2. Data-Driven Learning Methods

Unlike physics-based approaches, data-driven methods use large-scale datasets to generate visually realistic degradations without explicitly modeling the physical imaging process. Although most existing studies focus on road transportation, these weather-generation techniques can also inform maritime vision tasks because of their similar atmospheric degradation characteristics. Generative adversarial networks (GANs) and denoising difusion probabilistic models (DDPMs) are two mainstream paradigms for adverse-weather scene generation.

## 2.2.1. GAN-Based Methods

GANs learn the distribution of real images through adversarial training between a generator and a discriminator, enabling visually realistic image synthesis and providing a methodological foundation for degraded image generation (Goodfellow et al., 2014). GAN-based methods can be broadly divided into paired clear-to-degraded image translation and unpaired domain translation.

Paired clear-to-degraded translation learns mappings using paired source and target images. A representative method is pix2pix, which learns a source-to-target domain mapping through paired supervision (Isola et al., 2017). For degraded image generation, clear images are used as inputs and images of the same scenes in restricted visibility serve as targets. This paradigm supports data augmentation and annotation reuse while largely preserving scene structures. For example, Shao et al. (2022a) developed a multi-scale pix2pix-based model to learn mappings between clear and rainy driving images, thereby generating high-quality rainy scenes. Li et al. (2024) used the pix2pix framework for controllable augmentation of low-light roadside images, synthesizing nighttime or low-visibility trafic scenes to improve object detection. Paired translation provides explicit mappings and strong structure preservation, but its practical use is constrained by the dificulty of collecting strictly paired clear and degraded images.

Unpaired domain translation learns mappings without paired samples. CycleGAN, a representative framework, uses adversarial and cycle-consistency constraints to learn mappings between source and target domains (Zhu et al., 2017). Because independently collecting clear and degraded images is generally easier than obtaining strict pairs, this paradigm is more suitable for real restricted-visibility scenarios. Weather GAN supports translation among sunny, cloudy, foggy, rainy, and snowy domains and reduces structural distortion in non-degraded regions through weather-cue segmentation and attention modules (Li et al., 2021). SUSTechGAN targets rainy and nighttime image generation for autonomous driving by incorporating dual-attention modules, a multi-scale generator, and task-related losses; the generated images are further used to improve object detection in adverse conditions (Lan et al., 2025). Although unpaired methods avoid reliance on strictly paired data, they may provide imprecise control over degradation intensity and cause object-boundar deformation or semantic-structure drift.

## 2.2.2. DDPM-Based Methods

Difusion models (Ho et al., 2020) have been widely applied to visual generation because of their training stability and generation quality. With advances in vision-language representations, text conditioning has been incorporated into difusion models to control image content, scene attributes, and degradation conditions, such as weather, noise, and blur (Ramesh et al., 2022; Rombach et al., 2022). This provides a flexible basis for semantically controlled degraded image generation. According to whether additional non-textual conditions, such as images, degradation intensity, and layouts, are incorporated, these methods can be divided into text-only conditioned and multi-conditioned approaches.

![](images/1f8657faa819c37a570cfcdc5eb45b4a11b386718f7ab2e359a2a91cc02f74c5.jpg)  
Figure 4: Overview of intelligent vision for marine surface vehicles. The intelligent vision pipeline integrates shipboard video stabilization, visual perception enhancement (including hazy, low-light, and multi-scene enhancement), object detection and tracking, and navigable area segmentation, providing robust situational awareness for MSVs in restricted visibility.

Text-only conditioned methods use manually specified prompts or prompts generated by large language models to control degradation types and scene attributes. For example, Rothmeier et al. (2024) propose a text-to-image model for synthesizing driving scenes under snow, haze, and rain. The generated samples are subsequently used to fine-tune object detectors, consistently improving detection accuracy across adverse weather conditions. WeatherDG combines Stable Difusion with a large-language-model-based procedural prompt-generation strategy to synthesize driving scenes under snow, rain, haze, and low-light conditions, thereby improving the adverse-weather domain generalization of semantic segmentation models (Qian et al., 2025). Text-only conditioning improves semantic control over degradation types and scene attributes, but may still produce inaccurate degradation regions, structural drift, and images unsuitable for annotation reuse.

Beyond text conditioning, multi-conditioned methods incorporate image content, degradation attributes, or scene structures to provide finer control over degradation and scene generation. GenDeg introduces a conditional difusion model that accounts for both degradation type and intensity and can synthesize large-scale images covering six degradation types (Rajagopalan et al., 2025). GenDS, constructed using GenDeg, further demonstrates that all-in-one restoration models trained on the generated dataset achieve improved generalization. Guo et al. (2025b) present X-to-Maritime, a multimodally conditioned generative model for maritime object detection that integrates text, object layouts, water-surface environments, and imaging attributes to generate diverse maritime scenes and improve detection generalization. Compared with text-only conditioned methods, multi-conditioned methods provide stronger control over degradation attributes, scene structures, and task-oriented requirements, but incur higher computational complexity because of the integration of multiple conditional inputs.

## 3. Intelligent Vision

Adverse weather and illumination conditions can substantially degrade the visual perception of MSVs, increasing collision risks in complex and high-density waters. Integrating advanced intelligent vision modules into assisted and autonomous navigation systems is therefore important for robust situational awareness, as shown in Fig. 4. This section

reviews recent advances in shipboard video stabilization, visual perception enhancement, object detection and tracking, and navigable area segmentation.

## 3.1. Shipboard Video Stabilization

Video stabilization has been extensively studied across intelligent transportation systems. Traditional approaches can generally be divided into mechanical, optical, and electronic stabilization. For shipboard optical systems, many existing methods rely on accurate detection of the sea-sky horizon (Zardoua et al., 2020). To facilitate experimentation and autonomous maritime navigation, the first public annotation software for horizon-line detection was developed and released in (Zardoua et al., 2023). Schwendeman and Thomson proposed a method for reconfiguring and repairing shipboard videos by automatically tracking the dynamic horizon (Schwendeman and Thomson, 2015), together with an open-source Horizon Stabilization package covering horizon detection, image stabilization, rectification, and error analysis<sup>1</sup>. Wen et al. (2019) combined optical-flow-based motion estimation, Fourier-series-based motion filtering, and motion compensation to suppress random camera shake. To achieve real-time stabilization in heavy-sea environments, Xin et al. (2022) combined mechanical and electronic stabilization: a three-degree-of-freedom (DOF) gimbal with a robust controller provides primary motion compensation, followed by sea-sky-line-based electronic motion estimation and compensation. More advanced techniques developed for other stabilization tasks, including optical-flow-driven neural networks (Yu and Ramamoorthi, 2020), meta-learning (Ali et al., 2024), GANs (Xu et al., 2018), and Transformers (Wang et al., 2024e), may also be extended to shipboard video stabilization. Related limitations and emerging directions have been discussed in recent surveys (Guilluy et al., 2021; Roberto e Souza et al., 2022; Wang et al., 2023c).

## 3.2. Visual Perception Enhancement

Shipboard visible-light cameras are frequently afected by adverse weather and imaging conditions, including haze, low illumination, rain, and snow. The resulting image degradation can impair downstream tasks such as object detection, recognition, tracking, and multi-sensor data fusion. This subsection reviews representative enhancement techniques for improving visual perception in these conditions.

## 3.2.1. Hazy Image Enhancement

Haze reduces the accuracy and robustness of maritime object detection and tracking and consequently weakens MSV situational awareness. Maritime scenes difer substantially from conventional natural scenes, and dehazing methods developed for non-maritime imagery may sufer from transmission-estimation errors or atmospheric-light misestimation. Because shipboard surveillance imagery is dominated by visually similar sea and sky regions, accurate sea-sky separation is also important. In (Hu et al., 2023a), mean-shift segmentation and adaptive threshold constraints were used to estimate the sea-sky separation and transmission map. Linear transformation and improved high-boost filtering were combined for maritime dehazing in (Ma et al., 2016), while a comprehensive scattering model integrating illumination decomposition, transmission estimation, and illumination compensation was further developed for haze removal under glow-shaped environmental illumination (Hu et al., 2019).

Deep learning has also been increasingly adopted for maritime image dehazing. Fan et al. (2022) proposed an end-to-end multi-branch gated fusion network to handle high-frequency details, thick-haze regions, and contrast enhancement. Guo et al. (2023a) developed a dual-attention and dual-frequency-guided network to preserve both structural and low-frequency information while supporting real-time dehazing. To address non-uniform haze, Liu et al. (2023a) introduced a three-stage network that jointly estimates haze distribution, learns haze-related features, and performs multi-feature fusion. To alleviate the lack of paired hazy and clean ground truths, He and Ji (2023) developed an end-to-end GAN under unpaired supervision and constructed the realistic unpaired REMIDE dataset for maritime image dehazing.

## 3.2.2. Low-Light Image Enhancement

In low-light conditions, visible-light cameras receive limited illumination, resulting in low contrast and poorly resolved details that degrade both visual information and safety-related vision tasks. Low-light image enhancement aims to recover hidden information while suppressing unwanted noise and artifacts (Li et al., 2022a). It has been widely studied for intelligent ground vehicles (Lu et al., 2025), unmanned aerial vehicles (UAVs) (Wang et al., 2021b; Xing et al., 2023), and autonomous underwater vehicles (AUVs) (Im et al., 2025).

![](images/45e4606eb186d60e971e41582002d16ef39e2e2063ee152c886d8f69c990fc26.jpg)  
Figure 5: Illustrations of specific-purpose, general-purpose, and multi-scene image enhancement. Each specific-purpose model is tailored to handle a particular type of degradation (e.g., haze, snow, rain, and low-lightness). The general-purpose model is proposed to tackle a variety of image enhancement tasks using a shared or scalable network (Cui et al., 2026). In contrast, the multi-scene framework is capable of tackling various degradations using one single model (a.k.a., all-in-one).

In maritime environments, high-value vessels may additionally employ cooled or uncooled thermal cameras. However, uncooled thermal cameras provide limited geometric texture, whereas cooled thermal cameras are considerably more expensive than conventional visible-light and uncooled thermal cameras. Owing to their low cost and convenient deployment, visible-light cameras therefore remain widely exploited for MSVs. Deep learning has shown strong performance for low-light image enhancement (Liu et al., 2021b). For maritime imagery, Liu et al. (2022d) proposed a lightweight GAN (LGAN) with an attention mechanism to suppress overexposure. To support object detection under low illumination, Guo et al. (2022) developed a Retinex-driven lightweight network using depthwise separable convolution for low-light image enhancement. A robust variational regularized low-light enhancement method was presented by simultaneously refining the illumination and reflection maps (Liu et al., 2024a). In addition, a complexity-aware expert mechanism was developed to mitigate the negative efects of low-light images on maritime safety (He et al., 2025). As the spatial resolution of captured images increases (e.g., from standard definition of 480p to ultra-high definition (UHD) of 4K), the computational eficiency will be obviously decreased, leading to non-real-time image enhancement and unfriendly human-computer interaction. To obtain competitive enhancement quality and computational eficiency, a double domain guided real-time image enhancement network (DDNet) was proposed to handle UHD images under low-light conditions (Qu et al., 2024).

## 3.2.3. Multi-Scene Image Enhancement

Most enhancement methods discussed above are specific-purpose, targeting a single degradation type. To address multiple adverse conditions within a unified framework, multi-scene enhancement framework has attracted increasing attention. Liu et al. (2024c) proposed an all-in-one low-visibility enhancement network (AiOENet) for haze, rain, snow, and low-lightness, although its large parameter count limits computational eficiency. To reduce computational cost for MSVs deployment in IWTS, Liu et al. (2024b) developed an edge-reparameterization- and attention-guided network (ERANet), shown in Fig. 5, improving visual quality while maintaining low computational cost and supporting downstream object detection and scene segmentation under haze, rain, and low-light conditions. To promote vessel trafic safety in both hazy and low-light conditions, a dual-branch encoder–decoder architecture (TCE) was proposed to decompose the objective of multi-scene enhancement into two synergistic sub-tasks, i.e., texture reconstruction and color restoration (Zhang et al., 2026a). The optimized results of these two sub-tasks jointly contribute to the final enhancement scenario. Other all-in-one restoration approaches, including PromptIR (Potlapalli et al., 2023), ADMS (Park et al., 2023), and AIRFormer (Gao et al., 2024), have been developed for diferent application scenarios. Pre-trained vision-language models have also demonstrated the ability to restore images afected by multiple adverse-weather degradations (Yang et al., 2024). These developments provide promising techniques for improving shipboard imaging quality across diverse degradation types and levels.

Table 3  
Representative maritime object detection methods.
<table><tr><td>Method</td><td>Year</td><td>Modality</td><td>Framework</td></tr><tr><td>Complex-scene-aware methods</td><td></td><td></td><td></td></tr><tr><td>RetinaMFANet (Ma et al., 2022)</td><td>2022</td><td>VIS</td><td>RetinaNet</td></tr><tr><td>Weather-OD (Chen et al., 2024)</td><td>2024</td><td>VIS</td><td>Multiple detectors</td></tr><tr><td>MDD-ShipNet (Wang et al., 2024c)</td><td>2024</td><td>VIS</td><td>CNN</td></tr><tr><td>FREGNet (Tian et al., 2024)</td><td>2024</td><td>VIS</td><td>CNN-GCN</td></tr><tr><td>CFDBNet (Tian and Meng, 2025)</td><td>2025</td><td>VIS</td><td>Dual-branch CNN</td></tr><tr><td>TSPNet (Zhang et al., 2024c)</td><td>2024</td><td>IR video</td><td>Temporal pyramid</td></tr><tr><td>IS-MLDA (Feng et al., 2024b)</td><td>2024</td><td>IR</td><td>YOLO-based</td></tr><tr><td>FEAFNet (Zhang et al., 2024b)</td><td>2024</td><td>IR</td><td>CNN-based</td></tr><tr><td>YOLO-IRS (Guo et al., 2025a)</td><td>2024</td><td>IR</td><td>YOLOv10</td></tr><tr><td>CFLFE (Ma et al., 2026a)</td><td>2026</td><td>IR</td><td>Cascade Swin</td></tr><tr><td>Scale-aware methods</td><td></td><td></td><td></td></tr><tr><td>GAN-CNN detector (Chen et al., 2020)</td><td>2020 2022</td><td>VIS</td><td>GAN-CNN</td></tr><tr><td>VarifocalNet (Shao et al., 2022b)</td><td>2023</td><td>VIS VIS</td><td>CNN-based</td></tr><tr><td>Multi-scale detector (Zhang et al., 2023)</td><td>2023</td><td>VIS</td><td>YOLOv7</td></tr><tr><td>THFE (Guo et al., 2023c) YOLO-MRS (Yu et al., 2024)</td><td></td><td></td><td>CNN-based</td></tr><tr><td></td><td>2024</td><td>VIS</td><td>YOLOv8</td></tr><tr><td>RTDETR-MARD (Sun et al., 2025)</td><td>2025</td><td>VIS</td><td>RT-DETR</td></tr><tr><td>USVRT-DETR (Zhang et al., 2025b)</td><td>2025</td><td>VIS</td><td>RT-DETR</td></tr><tr><td>SARFB (Wu et al., 2023)</td><td>2023</td><td>IR</td><td>YOLOv5</td></tr><tr><td>GLAF-DETR (Zhang et al., 2025c)</td><td>2025</td><td>IR</td><td>RT-DETR</td></tr><tr><td>IRSD-Net (Sun and Lian, 2025) FGOM-RTDETR (Wang et al., 2026)</td><td>2025 2026</td><td>IR</td><td>YOLOv11</td></tr><tr><td></td><td></td><td>IR</td><td>RT-DETR</td></tr><tr><td>Lightweight detection methods</td><td></td><td></td><td></td></tr><tr><td>Saliency-aware CNN (Shao et al., 2020)</td><td>2019</td><td>VIS</td><td>CNN-based</td></tr><tr><td>Lightweight YOLOv5 (Yang et al., 2022)</td><td>2022</td><td>VIS</td><td>YOLOv5</td></tr><tr><td>DP-SSD (Wang et al., 2022)</td><td>2022</td><td>VIS</td><td>SSD</td></tr><tr><td>CG-YOLOv8 (Gong et al., 2025)</td><td>2025</td><td>VIS</td><td>YOLOv8</td></tr><tr><td>IRS-DETR (Zheng et al., 2026)</td><td>2026</td><td>IR</td><td>RT-DETR</td></tr></table>

## 3.3. Object Detection and Tracking

Shipboard object detection and tracking face several recurring challenges, including substantial environmental variations, large target-scale diferences, limited shipboard computational resources, and frequent occlusions. Recent advances in object detection and tracking are reviewed separately below.

## 3.3.1. Object Detection

Visible-light imagery provides rich texture, color, and structural cues for object recognition and localization, whereas infrared imaging captures thermal radiation independently of ambient illumination and therefore ofers a complementary modality for maritime navigation. Nevertheless, both modalities remain challenging in complex maritime conditions. Existing studies can broadly be grouped into complex-scene-aware, scale-aware, and lightweight methods. Representative approaches are summarized in Table 3. Currently, most maritime object detection methods are developed based on the YOLO and DETR architectures, whose basic frameworks are illustrated in Fig. 6.

Complex-scene-aware methods. Maritime object detection is performed in highly variable conditions involving adverse weather, sea-surface interference, dense target distributions, and occlusions. Existing methods address these challenges through data modeling, feature representation, and detection-framework design. For visible-light imagery, Weather-OD trains specialized detectors for diferent weather conditions and dynamically selects the appropriate model according to the current environment (Chen et al., 2024). For foggy scenes, Wang et al. (2024c) combined defogging, exposure adjustment, tone mapping, contrast enhancement, and sharpening with CNN-based parameter learning to improve image quality and detection robustness. Tian et al. (2024) introduced a Feature Representation Enhancement module and a graph convolutional network (GCN)-based Combiner to fuse global semantic and local detailed key points under illumination noise and cloud interference. RetinaMFANet redesigns RetinaNet with multi-scale feature aggregation to improve representation under low illumination (Ma et al., 2022), while Tian and Meng (2025) developed a coarse-to-fine dual-branch network with peer-to-peer communication to progressively localize candidate ship regions and refine contour representations.

![](images/777ef086277a8ebf096621fb68f2f331289c201c6dc50e4b0906db4e11c88f06.jpg)  
Figure 6: Representative YOLO- and DETR-based frameworks for maritime object detection. The upper branch illustrates multi-scale feature extraction and aggregation followed by YOLO detection heads. The lower branch illustrates a DETR-based architecture with a hybrid encoder, uncertainty-minimal query selection, and a transformer decoder for object classification and localization.

Infrared imaging is particularly useful in low-light and nighttime conditions, but limited texture and structural cues make maritime infrared detection dificult in dynamic sea environments. Zhang et al. (2024c) exploited temporal diferences between targets and wave clutter using accumulated energy and motion-diference maps. Feng et al. (2024b) combined meta-learning and domain adaptation to learn semantically meaningful domain-invariant features, while Zhang et al. (2024b) incorporated dilated convolution into an attention module to enlarge the receptive field and strengthen semantic representation. To reduce missed detections and false alarms caused by densely distributed targets and occlusion, Guo et al. (2025a) incorporated a Kolmogorov-Arnold Network (KAN) (Liu et al., 2024d) and shifted-window multi-head self-attention into YOLOv10 (Wang et al., 2024a), forming YOLO-IRS. Ma et al. (2026a) further proposed a Spatio-Semantic Enhanced Cascade Swin Detection Network that combines structural-tensor-based spatiotemporal enhancement with dynamic context-aware semantic attention.

Scale-aware methods. Diferences in observation distance, sensor altitude, and viewing perspective produce substantial scale variations in maritime imagery, while distant small targets can be easily obscured by sea clutter. Feature pyramid architectures (Lin et al., 2017; Ghiasi et al., 2019; Tan et al., 2020) are therefore widely used to integrate multi-level features. Zhang et al. (2023) combined multi-scale feature extraction, coordinate convolution, and a dynamic head to support multi-scale detection while maintaining a lightweight design. Yu et al. (2024) introduced multi-scale cross-axis attention to capture long-range contextual dependencies, whereas Shao et al. (2022b) used deformable convolutions together with DIoU loss and SNMS to accommodate geometric variations and improve candidate-box selection in dense scenes.

With the success of DETR-based frameworks (Zhu et al., 2021; Zhang et al., 2022a; Zhao et al., 2024), Transformer architectures have increasingly been adapted to maritime multi-scale detection. RTDETR-MARD combines hierarchical multi-scale fusion and feature aggregation for floating-debris detection by USVs (Sun et al., 2025). USVRT-DETR integrates multi-kernel cascading, channel progressive compression, and a small-object-oriented feature pyramid with a P2 layer, SPDConv, and COK-Fusion (Zhang et al., 2025b). For tiny boats, Guo et al. (2023c) proposed Triple-Hierarchy Feature Enhancement (THFE) to reinforce semantic representations across network layers, while Chen et al. (2020)

Table 4  
Representative maritime object tracking methods.
<table><tr><td>Method</td><td>Year</td><td>Tracking paradigm</td><td>Base framework</td></tr><tr><td colspan="4">Tracking-by-detection methods</td></tr><tr><td>DeepSORT (Wojke et al., 2017)</td><td>2017</td><td>Motion-appearance association</td><td>SORT and deep Re-ID</td></tr><tr><td>YOLOv4-UKF (Hong et al., 2022)</td><td>2022</td><td>Detection and motion</td><td>YOLOv4 and UKF</td></tr><tr><td>ORB-and-size tracker (Park et al., 2022)</td><td>2022</td><td>Spatial-feature association</td><td>loU and ORB</td></tr><tr><td>MaritimeMOT (Shao et al., 2024)</td><td>2024</td><td>Maritime target association</td><td>YOLOv7-based MOT</td></tr><tr><td>Sea-loUTracker (Guo et al., 2024) VTM-YS (Hao et al., 2025)</td><td>2024</td><td>Buffered-loU association</td><td>SORT</td></tr><tr><td>TIM-DETR MOT (Ma et al., 2026b)</td><td>2025 2026</td><td>Motion-appearance association</td><td>StrongSORT and OSNet</td></tr><tr><td></td><td></td><td>Detection, association, and Re-ID</td><td>TIM-DETR and Siamese network</td></tr><tr><td colspan="4">Detection-free tracking methods</td></tr><tr><td>SiamFPN (Shan et al., 2021b)</td><td>2021</td><td>Siamese tracking</td><td>Siamese network and FPN</td></tr><tr><td>MGPN (Yang et al., 2023)</td><td>2023</td><td>Memory-guided tracking</td><td>Siamese network and memory mechanism</td></tr><tr><td>Online Siamese-RPN (Hu et al., 2023b)</td><td>2023</td><td>Online Siamese tracking</td><td>Siamese-RPN and UpdateNet</td></tr></table>

combined a Gaussian Mixture Wasserstein GAN with Gradient Penalty and a CNN detector to generate additional small-ship samples and enrich training data.

Multi-scale detection is particularly challenging in infrared imagery because wave clutter, thermal reflections, and dynamic backgrounds further reduce target contrast and boundary clarity. GLAF-DETR introduces Global Local Adaptive Fusion and dynamic attention-guided multi-scale fusion into RT-DETR to enhance sparse and small targets (Zhang et al., 2025c). Wu et al. (2023) used cascaded asymmetric convolutions to enlarge the receptive field with fewer parameters. Sun and Lian (2025) combined a Hierarchical Multi-Kernel Convolution Network with a Dynamic Cross-Scale Feature Pyramid Network to integrate low-level details and high-level semantics. Wang et al. (2026) further proposed an infrared far-shore object-focusing framework incorporating Feature Re-parameterization, the Coordinate Attention Golden Feature Pyramid Network (Wang et al., 2023a), and a Multi-Scale Stacked Network to improve multi-scale feature extraction and long-range modeling.

Lightweight methods. MSV platforms are constrained by shipboard computation, power consumption, and real-time requirements, making lightweight detection important for practical deployment. Common strategies include eficient architectures and operators such as depthwise separable convolutions (Howard et al., 2017; Han et al., 2020b; Ding et al., 2024a) and lightweight backbones (Howard et al., 2019; Tan and Le, 2019; Chen et al., 2023), as well as mode compression through pruning (Ye et al., 2023; Cheng et al., 2017) and knowledge distillation (Chen et al., 2017a; Peng et al., 2025; Chang et al., 2023).

For maritime detection, Gong et al. (2025) introduced depthwise separable convolutions, a P6 detection layer, and CARAFE-based lightweight upsampling into YOLOv8 to balance eficiency and multi-scale perception. Yang et al. (2022) combined an improved ShufleNetV2 backbone with attention mechanisms and depthwise separable convolutions and redesigned the neck for eficient feature extraction. Wang et al. (2022) developed a lightweight detector evaluated in terms of computational time, frame rate, and recognition accuracy. Shao et al. (2020) combined deep features, saliency maps, and coastline priors to improve ship detection in complex maritime scenes. For infrared imagery, Zheng et al. (2026) proposed a lightweight Transformer detector with a dual-branch backbone, Pinwheel-Shaped Convolution for downsampling and small-target preservation, and Matchability-Aware Loss (MAL) for dense occlusions.

## 3.3.2. Object Tracking

Object tracking continuously localizes targets across consecutive frames by modeling their appearance and motion, thereby estimating temporal trajectories. Reliable maritime tracking remains challenging because of appearance variations, dynamic backgrounds, platform motion, and frequent occlusions. Event-based sensing has also been explored for object tracking by exploiting its high temporal resolution and complementary spatio-temporal information (Deng et al., 2026). Depending on whether an independent detector is required during inference, existing methods can generally be divided into tracking-by-detection and detection-free tracking. Tracking-by-detection first detects ships in individual frames and then associates instances using motion prediction, appearance similarity, or spatial matching. Detection-free methods generally initialize a target in the first frame and directly estimate its subsequent location through template matching, feature interaction, or temporal modeling.

To assess tracker applicability in maritime environments, Din et al. (2025) developed a closed-loop framework integrating visual tracking with USV motion control and evaluated Siamese- and Transformer-based trackers in simulated and real scenarios, demonstrating the robustness of SeqTrack under challenging conditions such as low visibility. Bakht et al. (2025) introduced MVTD, a large-scale maritime tracking benchmark containing 182 high-resolution videos and approximately 150,000 annotated frames, and evaluated 14 state-of-the-art trackers, revealing performance degradation of general-purpose trackers in maritime scenes and the benefits of domain-specific fine-tuning. Representative methods are summarized in Table 4.

Tracking-by-detection methods. DeepSORT (Wojke et al., 2017) is a representative tracking-by-detection framework that extends SORT (i.e., simple online and realtime tracking) with deep appearance descriptors. It combines Kalmanfilter-based motion prediction with deep appearance similarity to improve association across frames and has also been widely applied to ship tracking. Hong et al. (2022) combined YOLOv4 (Bochkovskiy et al., 2020) with an Unscented Kalman Filter (UKF) for real-time multi-ship tracking. Park et al. (2022) combined IoU matching with ORB-and-size matching to handle rapid changes in target position and appearance.

For more complex maritime scenarios, Shao et al. (2024) combined YOLOv7 with shipboard-camera motion compensation, a maritime dynamic-spatial-information-based IoU metric, and progressive refinement cascade matching. Sea-IoUTracker (Guo et al., 2024) used SORT with Bufered-IoU at two bufer scales and motion prediction to recover associations when direct adjacent-frame matching fails. To reduce computational complexity and identity switches under occlusion, Hao et al. (2025) integrated OSNet (Zhou et al., 2019a) into StrongSORT for Re-ID feature extraction and used an NSA Kalman filter for trajectory prediction. For prolonged occlusions, Ma et al. (2026b) combined TIM-DETR detection and temporal feature preservation with Siamese appearance similarity, IoU-based motion constraints, and historical-feature backtracking to re-identify vessels and reduce trajectory fragmentation.

Detection-free methods. Detection-free tracking does not invoke an independent detector in every frame. Siamese tracking is a representative paradigm that reformulates tracking as similarity matching: shared-weight branches extract features from template and search images, and cross-correlation produces target responses. This design provides eficient sample utilization and a favorable accuracy-speed trade-of, although it can remain sensitive to severe inter-object occlusion.

To address occlusion, scale variation, and motion blur, Yang et al. (2023) proposed the Memory-Guided Perception Network (MGPN), in which a Dynamic Memory Mechanism stores historical target features and a Hierarchical Context-Aware Module integrates global and local multi-scale information. Hu et al. (2023b) combined a Siamese network with a region proposal network, fused ofline Siamese and online classifier scores, and introduced an occlusion-aware strategy that suspends template updates and activates global search when occlusion occurs; UpdateNet further alleviates template degradation. Shan et al. (2021b) proposed SiamFPN, combining a ResNet-50 feature pyramid, a Siamese architecture, multiple region proposal networks, and historical-information-guided proposal selection for accurate and real-time maritime tracking.

## 3.4. Navigable Area Segmentation

Unlike object detection, which localizes and classifies discrete objects, shipboard semantic segmentation assigns pixel-level labels to the visual scene. It can identify navigable regions and obstacles and thereby provide environmental information for visual navigation and path planning. According to the target categories, maritime segmentation can broadly be divided into navigable-water segmentation and obstacle segmentation. The former focuses on the navigable region, whereas the latter jointly segments obstacles (e.g., ships, buoys, piers, and land), sky, and navigable water.

## 3.4.1. Navigable Waters Segmentation

Navigable-waters segmentation directly identifies navigable regions from perceptual data, providing scene information for path planning and external navigation constraints (Kim et al., 2023). A major challenge is the visual similarity between water reflections and surrounding objects. To address this issue, Zhou et al. (2022) constructed an ideal afinity matrix from manually annotated water and background regions and trained a contextual-prior module to model intra-region similarity and inter-region diferences, improving discrimination between shoreline scenes and their reflections. Ding et al. (2024b) introduced explicit water-boundary prediction as an auxiliary task and constraint on navigable-area segmentation to reduce confusion between water and land. Zhou et al. (2024) further exploited the correlation between image-row position and the spatial distribution of sky and navigable water, incorporating sine-encoded row-position embeddings together with water-boundary constraints to reduce sky-water confusion.

Water reflections strongly afect cameras but have a substantially diferent impact on LiDAR. Accordingly, several studies have fused LiDAR point clouds with visual information (Gao et al., 2022; Grini et al., 2024) or used LiDAR alone (Shan et al., 2021a) for navigable-area segmentation. Because most laser energy incident on the water surface is absorbed or specularly reflected, the resulting point clouds are dominated by returns from land and other solid structures. Cross-modal methods can therefore use the three-dimensional geometric structure provided by LiDAR as auxiliary information to reduce reflection-induced visual ambiguity, while 360<sup>◦</sup> LiDAR-based approaches can also infer navigable regions from this geometric characteristic (Shan et al., 2021a).

## 3.4.2. Obstacle Segmentation

Perceiving only the navigable region is insuficient for higher-level functions such as navigation decision-making, which require a more complete representation of the surrounding environment. Obstacle segmentation therefore jointly distinguishes obstacles, sky, and navigable water. Using the MaSTr1325 (Bovcon et al., 2019) and MODS (Bovcon et al., 2022) datasets, Bovcon et al. evaluated general semantic segmentation models including U-Net (Ronneberger et al., 2015), PSPNet (Zhao et al., 2017), and DeepLab v2 (Chen et al., 2018), showing that illumination variations and water reflections can cause segmentation errors and inaccurate object contours.

To improve robustness to vessel motion, Bovcon et al. (2018) incorporated inertial measurement unit (IMU) data into probabilistic semantic segmentation. Roll and pitch measurements are used to estimate the horizon and adapt the conditional prior distribution and Gaussian hyperprior, improving segmentation under severe vessel sway. Because handcrafted graphical models remain limited under reflections, highlights, spray, and wakes, Bovcon and Kristan (2020, 2022) developed WaSR, which combines deep visual features with shipboard IMU information through an asymmetric encoder-decoder architecture and introduces semantic separation loss to improve feature-space separation among semantic components. Žust and Kristan (2022) further proposed WaSR-T by incorporating visual observations from the current and previous � frames and extracting temporal context to reduce ambiguity from temporally unstable reflections and highlights.

Beyond reflection interference, small-object segmentation and false positives near the water-sky boundary remain challenging. Chen et al. (2021a) proposed WODIS, an encoder-decoder network that extracts multi-rate high-level features, uses an attention-refinement module based on global average and max pooling for sea-sky-line regions, and integrates multidimensional features through a feature-fusion module. Feng et al. (2024a) developed a dual-branch architecture consisting of a dilated-convolution semantic branch (Yu and Koltun, 2015) and a detail branch centered on an embedded feature-refinement module. Directional feature refinement and asymmetric convolution emphasize stable structural components while suppressing high-frequency dynamic interference such as water reflections, and the two branches are progressively fused during inference.

Boundary accuracy constitutes another major challenge. Zhang et al. (2024a) introduced separate boundary-extraction and boundary-enhancement modules to explicitly fuse boundary cues with global semantic features, improving pixel classification around object contours while preserving regional segmentation. Zhang et al. (2026b) proposed a multi-level boundary-semantic reinforcement mechanism that combines explicit boundary supervision, Coordinate Attention for spatial localization, and a global-local boundary-aware discriminator based on shape context to adaptively emphasize geometrically inconsistent regions. Camera-based surround-view bird’s-eye view (BEV) segmentation has attracted attention in autonomous driving, but its development for water-surface environments remains limited by the scarcity of corresponding BEV datasets. Maritime BEV segmentation also faces vessel sway and reflection interference. To support this direction, Xu et al. (2024a) introduced the WSBEV dataset using a hybrid configuration of pinhole and fisheye cameras with diferent focal lengths. The Hybrid-BEV was then proposed, which was essentially a water-surface BEV segmentation network supporting heterogeneous camera inputs.

Overall, recent work has improved the accuracy and stability of maritime scene segmentation. However, environmental interference, small-object segmentation, and boundary accuracy remain important challenges. The attention mechanisms, global-local feature modeling, temporal modeling, and multi-modal data fusion provide potential directions for improving robustness, while lightweight and high-performance models remain important for practical MSV deployment.

## 4. Vision-Centered Multi-Sensor Fusion

To further improve maritime situational awareness, MSVs can exploit complementary information from multiple shipboard sensors. Vision-centered multi-sensor fusion integrates visual observations with other sensing modalities

to improve environmental perception in non-ideal navigation conditions. This section reviews representative fusion strategies, including infrared-visible image fusion and the fusion of vision with radar, LiDAR, AIS, and other shipboard sensors, with particular emphasis on navigational safety in restricted visibility.

## 4.1. Infrared and Visible Image Fusion

Both infrared and visible-light cameras have been widely used for surface-vessel perception (Bloisi et al., 2017). Visible imagery provides rich appearance information but is vulnerable to adverse illumination and weather, whereas infrared imagery captures thermal radiation but generally has lower spatial resolution and a more limited field of view (FOV). The substantial modality gap between visible and infrared imagery has also motivated cross-modal representation learning to improve their correspondence (Zhong et al., 2022). Infrared-visible fusion therefore seeks to preserve complementary thermal and appearance information, improving the robustness and accuracy of vessel recognition (El Mahdi, 2024). For example, Lu et al. (2023) proposed an attention-driven framework consisting of feature extraction, fusion, and reconstruction modules to improve navigational-environment perception. Calibrated and aligned infrared and visible images have also been used to enhance shipboard visual perception (Zhang et al., 2022b). Farahnakian et al. (2019) incorporated fused imagery into a CNN-based RetinaNet detector to improve MSV detection in diferent weather and illumination conditions, and further investigated pixel-, feature-, and decision-level fusion strategies for maritime object detection (Farahnakian and Heikkonen, 2020). The VAIS benchmark, containing more than 1000 paired infrared and visible images across six categories, was constructed to evaluate vessel recognition (Zhang et al., 2015). More comprehensive discussions of infrared-visible image fusion can be found in (Ma et al., 2019; Zhang and Demiris, 2023).

## 4.2. Radar and Vision Fusion

Visual sensors provide rich semantic and categorical information but have limited capability to directly measure relative distance, velocity, and course between a target and the own ship. Navigation radar on large MSVs and 4D millimeter-wave (mmWave) radar on smaller MSVs can provide complementary geometric and kinematic information, while maintaining all-day and all-weather sensing capabilities under low illumination and adverse weather (Yao et al., 2024a, 2025c). Radar-vision fusion can therefore jointly exploit visual semantics and target motion information for more comprehensive maritime situational awareness. Existing studies mainly consider navigation-radar-vision fusion and 4D-mmWave-radar-vision fusion.

## 4.2.1. Navigation Radar and Vision Fusion

Navigation radar and cameras difer substantially in sensing geometry and data representation, creating a pronounced cross-modal gap. Navigation radar measures range, velocity, and course in polar coordinates and commonly represents observations in BEV, whereas cameras map the three-dimensional environment onto a two-dimensional perspective image (Shi et al., 2026). Cross-modal matching and alignment are therefore central to navigation-radar-vision fusion.

Early methods primarily relied on static geometric alignment. Liu et al. (2021d) and Clunie et al. (2021) used calibrated extrinsic parameters and coordinate transformations to project radar detections onto the image plane. Although straightforward, such approaches are sensitive to calibration drift caused by vessel pitching and long-term vibration. Adaptive and probabilistic matching strategies have subsequently been developed to reduce this dependence. Farahnakian et al. (2018) introduced cross-modal consistency checks to filter redundant visual proposals using radar-derived spatial constraints. Cormack et al. (2020) proposed a joint registration and fusion framework using probabilistic data association to accommodate measurement uncertainty and dynamic sensor ofsets. Kim et al. (2021) improved matching robustness by optimizing geometric parameters, reducing dependence on precise initial calibration. AnytimeFusion (Shin et al., 2024) further employed particle swarm optimization for fully parameterized automatic alignment in dynamic maritime environments.

More recent studies have shifted toward learned cross-modal association. Xu et al. (2025) introduced a Transformerbased fusion module that uses cross-attention to associate radar observations with visual semantics in a latent feature space, reducing representational discrepancies between the two modalities. For small water-surface targets, where radar echoes are sparse and strongly afected by multipath clutter, Wei et al. (2024) proposed multi-frame radar feature accumulation to improve temporal density and continuity, followed by image-guided radar feature enhancement that uses high-resolution visual semantics to refine sparse radar representations.

![](images/08b6f4e375d3d6db061555788c706c22f7476941e0964ee1ac79beebd6569520.jpg)  
Figure 7: Schematic overview of 4D mmWave radar-vision fusion. Vision and 4D mmWave radar provide complementary semantic, geometric, and kinematic information. Cross-modal alignment establishes temporal, spatial, and regional correspondences between the two modalities, followed by radar-vision feature fusion using simple, attention-based, graphbased, or asymmetric fusion strategies. The fused features support downstream maritime perception tasks, including unified perception, multi-object tracking, and language-guided perception.

## 4.2.2. 4D mmWave Radar and Vision Fusion

Compared with navigation radar, 4D mmWave radar additionally measures elevation, extending perception from a two-dimensional plane to three-dimensional space, as illustrated in Fig. 7. Benefiting from this enhanced spatial sensing capability, 4D-mmWave-radar-vision fusion has been increasingly explored in autonomous driving for tasks such as 3D object detection and tracking, simultaneous localization and mapping, and BEV perception under diverse weather and illumination conditions (Fan et al., 2024). In contrast, the application of 4D-mmWave-radar-vision perception to maritime environments has emerged more recently and remains less extensively studied. Even so, recent advances in dedicated datasets, perception frameworks, and task-specific methods have begun to establish a solid foundation for further research on 4D-mmWave-radar-vision perception in maritime scenarios.

Representative datasets, such as WaterScenes (Yao et al., 2024b) and USVInland (Cheng et al., 2021a), have been constructed for 4D-mmWave-radar-vision perception under diverse illumination and weather conditions. These datasets have supported studies on challenging tasks, e.g., small maritime object detection (Cheng et al., 2021b; He et al., 2024).

Several studies have subsequently explored unified maritime perception. Achelous and Achelous++ (Guan et al., 2023, 2026b) introduced multi-task panoptic perception frameworks that jointly address heterogeneous tasks including object segmentation, navigable-area segmentation, object detection, waterline segmentation, and radar point-cloud segmentation. To improve cross-modal alignment and robustness against water-surface interference, Mask-VRDet (Guan et al., 2024a) introduced a dual-graph fusion framework to model non-Euclidean relationships between visual and radar features, together with a clutter-point-removal strategy. To accommodate diferent modality requirements across tasks, Guan et al. (2024b) further proposed an asymmetric fair fusion module that represents image and radar observations as irregular superpixel sets and interacts with them in a shared cross-modal feature space. Aiming at the issues of asynchronous modality degradation and the resulting decline in perception performance that can arise when using 4D mmWave radar and vision sensors in dynamic maritime environments, Song et al. (2026) proposed an uncertainty-aware radar-vision fusion method. By explicitly modeling the uncertainty inherent in visual and radar features, this method enables modality reliability allocation and the isolation of degradation noise at the feature level, thereby enhancing the robustness of fused perception in complex, dynamic environments.

Beyond detection and segmentation, radar-vision fusion has also been extended to tracking. Yao et al. (2025b) constructed USVTrack, a comprehensive 4D-mmWave-radar-vision tracking dataset for inland waterways, and proposed

![](images/701712642421a91cc9a2ca9a93c54118b6a95f165a5632a4d84e42c0ffc94036.jpg)  
Figure 8: Representative LiDAR–camera fusion paradigms for maritime 3D object detection. (a) Image detections define viewing frustums for selecting LiDAR points before 3D detection. (b) Image and LiDAR features are aligned and fused in a bird’s-eye-view representation. (c) Independently generated image detections and LiDAR proposals are associated across modalities and jointly refined to produce 3D bounding boxes.

Radar-Camera Matching (RCM). RCM uses radar-derived distance and velocity as kinematic constraints for cross-frame target association, reducing identity ambiguity during visual occlusion.

Recent studies have further extended radar-vision perception toward language-guided interaction. Guan et al. (2025a) introduced WaterVG, a waterway visual-grounding dataset containing more than 11,000 samples and 35,000 referred targets, with textual descriptions covering semantic information and numerical kinematic attributes such as distance and speed. The associated Potamoi model performs referring-expression comprehension and segmentation, enabling language-guided target localization and segmentation. NanoMVG (Guan et al., 2025b) subsequently introduced a low-power multi-task framework optimized for edge deployment while maintaining localization and segmentation capability. For autonomous navigation, occupancy modeling provides an important bridge between perception and path planning. Xu et al. (2024b) proposed a real-time 4D-mmWave-radar-camera fusion network for BEV occupancy prediction. A BEV spatial Transformer projects perspective-view image features into a radar-aligned BEV space, enabling two-dimensional occupancy mapping while suppressing water-surface clutter and noise.

These studies demonstrate the complementary roles of vision and 4D mmWave radar in maritime perception. Vision provides rich appearance and semantic information, whereas 4D mmWave radar contributes direct range, radial-velocity, and elevation measurements that provide additional geometric and kinematic constraints, particularly when visual observations are degraded. Their fusion therefore improves spatial localization, target association, tracking, and scene understanding under dynamic maritime conditions. However, current studies have predominantly focused on short-range to medium-range MSV perception, largely because the sensing range of existing 4D mmWave radar constrains long-range maritime operation. The existing radar-vision fusion methods extend to longer-range perception once radar sensing delivers reliable measurements at the required distances.

## 4.3. LiDAR and Vision Fusion

LiDAR provides accurate geometric and depth information, whereas cameras capture rich appearance and semantic information, making the two modalities complementary for three-dimensional environmental perception. In autonomous driving, LiDAR-vision fusion is commonly categorized as Early, Middle, and Late fusion according to the stage at which cross-modal interaction occurs, shown in Fig. 8. These paradigms also provide useful methodological references for maritime perception.

• Early fusion combines multimodal observations during preprocessing or at the input stage, typically through spatial alignment. This strategy preserves raw cross-modal information and facilitates direct modeling of geometricappearance relationships. PointPainting (Vora et al., 2020) and PointAugmenting (Wang et al., 2021a), for example, project point clouds onto the image plane or construct unified three-dimensional representations to enrich point-cloud features with visual information.

• Late fusion combines independently generated detection or recognition outputs. Frustum-PointNets (Qi et al., 2018) first obtain image-based 2D detections and project the corresponding regions of interest (RoIs) into 3D frustums, within which PointNet-like networks estimate 3D bounding boxes. RoarNet (Shin et al., 2019) first estimates geometrically feasible 3D candidate regions from monocular images and subsequently refines them using a 3D detector. Frustum-PointPillars (Paigwar et al., 2021) combines visual detections with PointPillars (Lang et al., 2019) representations to support localization under sparse point clouds.

• Middle fusion has become a major paradigm because it enables intermediate semantic features from diferent modalities to interact directly. Early approaches often projected point clouds into multiple 2D views. MV3D (Chen et al., 2017b), for example, projects point clouds into BEV and front-view representations and fuses image, BEV, and front-view features through RoI pooling.

More recent methods seek to preserve three-dimensional spatial information and learn adaptive cross-modal alignment. MMF (Liang et al., 2019) uses continuous fusion layers to map image features into BEV representations; EPNet (Huang et al., 2020a) establishes point-wise correspondences between point clouds and image features; and 3D-CVF (Yoo et al., 2020) performs cross-view mapping between voxel and image features. VPFNet (Zhu et al., 2023) fuses point-cloud voxels and stereo-image features through virtual grid points, while SFD (Wu et al., 2022) combines voxel and pseudo-point-cloud representations. AutoAlign (Chen et al., 2022) introduces learnable alignment and cross-attention for adaptive pixel-to-voxel aggregation. BEVFusion (Liu et al., 2023b) unifies multiple modalities in a shared BEV representation, whereas TransFusion (Bai et al., 2022) uses Transformer decoders to initialize object queries from LiDAR features and fuse them with image features through attention. DeepFusion (Li et al., 2022b) similarly converts modality-specific features into a common BEV representation for subsequent fusion.

Compared with autonomous driving, LiDAR-camera fusion for MSVs remains relatively underexplored. Existing studies have first investigated LiDAR-only maritime perception. Eustache et al. (2025) released a USV point-cloud dataset augmented through semi-physical simulation. For busy maritime environments, Xie et al. (2024) proposed a modular LiDAR-based detection framework together with Kalman-filter-based multi-object tracking to compensate for missed detections caused by motion and occlusion. Guo et al. (2026) combined average and attention pooling to aggregate local geometry and point intensity within pillars. Ponzini et al. (2025) integrated lightweight supervised classification into an unsupervised maritime detection pipeline, while Xie et al. (2026) combined attention-based pillar feature extraction with a hybrid MobileViT-CNN backbone for global representation learning.

LiDAR-only perception nevertheless remains sensitive to waves, reflections, and other environmental noise and provides limited appearance information for object recognition and classification. LiDAR-vision fusion therefore ofers a natural means of combining precise geometry with visual semantics for MSV navigation. Zhang et al. (2021b) developed an integrated WAM-V-USV perception and obstacle-avoidance framework combining LiDAR and vision with water-reflection removal, image denoising, sea-sky detection, and obstacle avoidance. Chen et al. (2021b) proposed a Three-Dimensional Perception Module (PMTD) that combines camera-LiDAR perception with GPS and IMU localization to support avoidance of static and dynamic obstacles.

For water-surface target detection and localization, Wang et al. (2023b) first detects candidate targets using CornerNet-Lite, projects LiDAR points onto the image plane, and combines camera and LiDAR confidence scores to determine targets and estimate their 3D positions. An image-dominant fusion framework was further proposed (Zhang et al., 2024d). In particular, the visual observations served as the primary modality and LiDAR provided complementary information. The tracking-assisted detection and confidence-based association were then exploited to compensate for missed detections. To alleviate the shortage of maritime multimodal data, a multimodal dataset (Choi et al., 2025) with image- and point-level annotations was introduced for obstacle detection and tracking, including detailed annotations for small objects. Yao et al. (2025a) developed a two-stage point-cloud clustering strategy using adaptive thresholds and elliptical heat zones to suppress mixed target-background false detections, followed by saliency-based segmentation for target-contour extraction. For autonomous docking, Xu et al. (2026) proposed a markerless berth-detection framework that combines scene semantics with camera-LiDAR fusion. It integrates a LiDAR-assisted view Transformer and fusion-attention module to generate semantic 3D point clouds and identify available docking spaces.

![](images/47cc92843d2c665974f40a80726984462d4059a56e2dc09d6f1904846d329fb9.jpg)  
Figure 9: Flowchart of AIS and vision data fusion. The AIS and vision data fusion contributes to an augmented reality (AR) navigation system for intelligent surface vehicles. It improves situational awareness and decision-making capacities for safe maritime navigation under complex conditions.

## 4.4. AIS and Vision Fusion

AIS provides vessel identity, position, and motion information, whereas cameras ofer complementary appearance and semantic cues. Their fusion can achieve the synchronized display of the vessel’s appearance and identity information, thus improving the maritime situational awareness, shown in Fig. 9. Liu et al. (2022a) fused YOLOX-based visual detections with synchronized AIS messages using distance and angular constraints, supporting an augmented-reality (AR)-assisted intelligent navigation system. Huang et al. (2023) combined image observations with AIS messages to estimate ship dimensions in real time and reduce potential collision risk. Ding and Weng (2024) jointly employed multiregion mapping and multi-feature similarity measurement to improve AIS-vision association and vessel collision-risk assessment in inland waterways, including in low-light conditions. The pioneering AR navigation system based on AIS and vision fusion was developed by Furuno Electric Co., Ltd. More details on representative companies and their navigation systems could be found in Section 5.4.

To address spatial and temporal discrepancies between AIS and camera observations, Xiao et al. (2025) introduced regional-division projection and graph-based multi-feature fusion framework. The AIS-vision fusion has also been investigated for maritime trafic situational awareness in diverse environments (Guo et al., 2023b; Qu et al., 2023; Lu et al., 2026). To improve maritime trafic safety, this fusion strategy has been successfully extended to shore-based maritime surveillance (Holst et al., 2026; Teixeira et al., 2026), and UAV-based surface-vessel identification and tracking (Zhao et al., 2025, 2026b).

## 4.5. Other Multi-Sensor Fusion Scenarios

Because individual sensors exhibit complementary strengths and limitations, integrating more than two sensing modalities can further improve maritime situational awareness in dynamic and complex environments (Zhang et al., 2025a). The Maritime RobotX Challenges (Stanislas and Dunbabin, 2019) have promoted the development and validation of such heterogeneous sensor-fusion systems. For example, probabilistic fusion of radar, LiDAR, and cameras has been investigated for robust target detection and classification in uncertain conditions, supporting autonomous navigation and collision avoidance.

Several studies have developed integrated multi-sensor navigation systems. Han et al. (2020a) presented both hardware and software components of an autonomous collision-avoidance system, while subsequent field experiments demonstrated fully unmanned operation in complex coastal trafic environments (Han et al., 2023). Clunie et al. (2021) combined monocular cameras with radar and LiDAR to estimate the position, dimensions, and heading of static and dynamic targets. For real-time detection and tracking, Haghbayan et al. (2018) proposed probabilistic data association using independent detections from radar, LiDAR, RGB cameras, and infrared cameras. Joint integrated probabilistic data association (JIPDA) and its multi-sensor extension were further developed and experimentally validated using these four sensing modalities (Helgesen et al., 2022). Mousazadeh et al. (2018) fused range-finder LiDAR, stereo vision, and ultrasonic sensors through an extended Kalman filter (EKF), demonstrating robustness under obstacles and environmental disturbances including wind, waves, and currents.

![](images/2872468b9e56df4674234f229f0647f6b22b283af5a9cb836300b42d07552991.jpg)  
Figure 10: Framework of vision-centered multi-sensor fusion. In particular, gyro denotes the gyrocompass device. MCGA, DAWF/TDSF, and VARF, respectively, denote the AIS–radar data association, AIS–radar data fusion, and vision–AIS–radar data fusion methods. The selection strategy seselects the appropriate fusion method based on available observation sources. Please refer to Ref. (Liu et al., 2026) for more details on vision-centered multi-sensor fusion and its application in situational awareness for safe navigation.

More recently, Liu et al. (2026) developed a NavEYE system to promote maritime situational awareness and safety navigation in complex environments, shown in Fig. 10. It is a vision-centered multi-sensor fusion system by integrating radar, RGB cameras, AIS, global navigation satellite system (GNSS), gyroscopes, and other shipboard sensors. To extend the perception range of MSVs, a new concept, i.e., beyond-visual-range (BVR) situational awareness (Zhao et al., 2026a), was proposed using the maritime Internet of Things (mIoT)-based collaborative framework. It has the capacity of detecting targets beyond the shipboard sensing coverage.

## 5. Datasets and Safety-Critical Applications

Deep learning methods for intelligent vision and multi-sensor data fusion depend strongly on the scale and diversity of training data. This section first reviews representative public datasets for maritime perception and then discusses safety-critical applications of intelligent vision and vision-centered data fusion, including vessel behavior prediction, intelligent collision avoidance, and vision-centered navigation systems.

## 5.1. Datasets

From the perspective of navigational safety, the development of maritime perception datasets can be broadly viewed in three stages, as summarized in Table 5. The first is dominated by monocular and stereo cameras, represented by MODD (Kristan et al., 2016), MODD2 (Bovcon et al., 2018), MID (Liu et al., 2021a), SMD (Moosbauer et al., 2019), and MaSTr1325 (Bovcon et al., 2019). These datasets established fundamental maritime vision tasks, including horizon and waterline estimation, obstacle detection, tracking, and navigable-area segmentation. Their relatively low acquisition and annotation costs facilitated early benchmarking. However, visible imagery is readily degraded by glare, fog/haze, nighttime illumination, specular reflections, wave disturbances, and long-range small targets, while monocular and stereo vision provide limited robustness in geometric and motion perception.

Table5:OverviewofpubliclyavailabledatasetsforMSVs.Tasks:OD:objectdetection;LS:waterlinesegmentation;OT:objecttracking;SS:semanticsegmentation;FS: free-spacesegmentation;PS:panopticsegmentation;IS:instancesegmentation;PP:panopticperception.
<table><tr><td>Dataset</td><td>Year</td><td>Sensors</td><td>Tasks</td><td>Annotations</td><td>Classes</td><td>Frames</td></tr><tr><td>MODD (Kristan et al., 2016)</td><td>2015</td><td>Monocular Camera</td><td>OD, LS</td><td>2D Box, 2D Line</td><td>2</td><td>4,454</td></tr><tr><td>MODD2 (Bovcon et al., 2018)</td><td>2018</td><td>Stereo Camera</td><td>OD, LS</td><td>2D Box, 2D Line</td><td>2</td><td>11,675</td></tr><tr><td>SMD (Moosbauer et al., 2019)</td><td>2019</td><td>Monocular Camera</td><td>OD, OT</td><td>2D Box</td><td>10</td><td>31,653</td></tr><tr><td>MaSTr1325 (Bovcon et al., 2019)</td><td>2019</td><td>Monocular Camera</td><td>SS</td><td>2D Pixel</td><td>4</td><td>1,325</td></tr><tr><td>MODS (Bovcon et al., 2022)</td><td>2021</td><td>Stereo Camera</td><td>OD, SS</td><td>2D Box, 2D Line</td><td>3</td><td>24,090</td></tr><tr><td>MID (Liu et al., 2021a)</td><td>2021</td><td>Monocular Camera</td><td>OD</td><td>2D Box</td><td>2</td><td>2,655</td></tr><tr><td>USVInland (Cheng et al., 2021a)</td><td>2021</td><td>Stereo Camera</td><td>SS, FS</td><td>2D Line</td><td>1</td><td>700</td></tr><tr><td>FloW (Cheng et al., 2021c)</td><td>2021</td><td>Monocular Camera, 3D Radar</td><td>OD</td><td>2D Box</td><td>1</td><td>2,000</td></tr><tr><td>SeaSAw (Kaur et al., 2022)</td><td>2022</td><td>Monocular Camera</td><td>OD</td><td>2D Box</td><td>12</td><td>1.9M</td></tr><tr><td>LaRS (Žust et al., 2023)</td><td>2023</td><td>Monocular Camera</td><td>SS, PS</td><td>2D Line</td><td>11</td><td>4,006</td></tr><tr><td>MVDD13 (Wang et al., 2024d)</td><td>2024</td><td>Monocular Camera</td><td>OD</td><td>2D Box</td><td>13</td><td>35,474</td></tr><tr><td>WaterScenes (Yao et al., 2024b)</td><td>2024</td><td>Monocular Camera, 4D Radar</td><td>OD, IS, SS, PP</td><td>2D Box, 2D Pixel, 3D Point</td><td>7</td><td>54,120</td></tr><tr><td>OASIs (Kim et al., 2024b)</td><td>2024</td><td>Monocular Camera</td><td>SS</td><td>2D Pixel</td><td>4</td><td></td></tr><tr><td>MVTD (Bakht et al., 2025)</td><td>2025</td><td>Monocular Camera</td><td>OT</td><td>2D Box, Track ID</td><td>4</td><td>~150k</td></tr><tr><td>PoLaRIS (Choi et al., 2025)</td><td>2025</td><td>Monocular/TIR Camera, LiDAR, Radar</td><td>OD, OT</td><td>2D Box, 3D Point, Track ID</td><td>2</td><td>~360k</td></tr><tr><td>USVTrack (Yao et al., 2025b)</td><td>2025</td><td>Monocular Camera, 4D Radar</td><td>OD, OT</td><td>2D Box, 3D Point, Track ID</td><td>3</td><td>68,822</td></tr><tr><td>MULTIAQUA (Muhovič and Perš, 2025)</td><td>2025</td><td>Stereo/Thermal/NIR Camera, LiDAR, Radar</td><td>SS</td><td>2D Pixel</td><td>4</td><td>3,293</td></tr><tr><td>SeaClips (Denk et al., 2026)</td><td>2026</td><td>Monocular Camera</td><td>OD</td><td>2D Box</td><td>7</td><td>31,606</td></tr><tr><td>WUTDet (Liang et al., 2026)</td><td>2026</td><td>Monocular Camera</td><td>OD</td><td>2D Box</td><td>1</td><td>100,576</td></tr></table>

The second stage is characterized by larger and more diverse RGB benchmarks, such as LaRS (Žust et al., 2023), MVDD13 (Wang et al., 2024d), SeaClips (Denk et al., 2026), and WUTDet (Liang et al., 2026). Progress is reflected not only in dataset scale but also in scene diversity, category complexity, temporal structure, and evaluation protocols. LaRS (Žust et al., 2023) covers lakes, rivers, and seas with 19 global scene attributes and panoptic annotations containing three stuf and eight thing classes, extending maritime perception toward panoptic scene understanding. SeaClips (Denk et al., 2026) emphasizes video-based frame-level detection, whereas WUTDet (Liang et al., 2026), with more than 100k images, provides a large-scale benchmark for dense small-ship detection in complex port and navigation environments. These datasets increasingly emphasize reliable generalization under diverse scene distributions and substantial target-scale variations.

The third stage moves toward multimodal, safety-oriented, and visibility-aware datasets. Representative examples include USVInland (Cheng et al., 2021a), FloW (Cheng et al., 2021c), WaterScenes (Yao et al., 2024b), PoLaRIS (Cho et al., 2025), USVTrack (Yao et al., 2025b), and MULTIAQUA (Muhovič and Perš, 2025). These datasets introduce complementary modalities such as millimeter-wave/4D radar, LiDAR, thermal infrared (TIR), and near-infrared (NIR), while emphasizing synchronization, extrinsic calibration, cross-modal projection, and temporal ID consistency (Yao et al., 2026). They reflect a transition from vision-only perception toward vision-centered data fusion, where cameras provide rich semantic and boundary information while radar, thermal infrared, and LiDAR contribute complementary geometric, structural, and sensing capabilities in restricted visibility. Accordingly, these benchmarks provide an increasingly important foundation for studying cross-modal perception and safety-relevant maritime navigation.

## 5.2. Vessel Behavior Prediction

Vessel behavior refers to the movements and operational actions of a vessel, including changes in course and speed, route following, berthing, anchoring, and collision avoidance. These behaviors reflect its navigational goals and responses to surrounding trafic and waterway conditions (Zhou et al., 2019b). Predicting such behavior is important for the navigational safety of marine surface vehicles in restricted visibility, where degraded visual observations can limit the timely recognition of developing encounters. Intelligent vision and vision-centered multi-sensor fusion provide complementary information on surrounding vessels, including their locations, motion histories, and contextual cues, which can support forecasts of their future behavior. These forecasts extend situational awareness from the current trafic state to potential future conflicts, supporting anticipatory risk assessment and collision avoidance (Zhang et al., 2022c). Vessel behavior prediction therefore forms an important link between perception and safe navigation. As summarized in Fig. 11, its principal tasks include motion and trajectory prediction, route and destination prediction, maneuvering intention prediction, and interaction-aware behavior prediction. These complementary tasks address where surrounding vessels may move, which navigational goals they may pursue, and how they may respond to an evolving encounter, with uncertainty considered across the diferent prediction tasks.

Motion and trajectory prediction estimates a vessel’s future positions and, where explicitly modeled, its course and speed over a specified prediction horizon. Zhang et al. (2022c) provides the spatial and temporal basis for trafic monitoring and anticipatory collision-risk assessment. Recurrent neural networks learn temporal dependencies from historical AIS observations (Capobianco et al., 2021), while recurrent encoder–decoder architectures predict future trajectory sequences together with uncertainty estimates (Capobianco et al., 2023). Gaussian-process models and hybrid predictors also account for uncertainty in future vessel motion (Rong et al., 2019; Wang et al., 2024b). These approaches extend motion prediction beyond a single extrapolated path by characterizing the range of possible future positions.

Route and destination prediction concerns the broader spatial goals of vessel movement. Route prediction estimates the likely navigational path or sequence of route segments, whereas destination prediction identifies the port or area toward which a vessel is heading. Historical AIS data provide recurring movement patterns that support these tasks (Tu et al., 2018). Route reconstruction from incomplete trajectories (Dobrkovic et al., 2018) and trajectory similarity learning (Liang et al., 2021) help establish historical route patterns and behavioral references, although neither task alone constitutes future prediction. For destination prediction, AIS-based random forests have been used to infer likely destinations from vessel observations (Zhang et al., 2020). Route and destination information can, in turn, constrain trajectory forecasts by distinguishing movements toward diferent navigational goals.

Maneuvering intention prediction addresses the actions a vessel is likely to undertake, such as changing course, accelerating, decelerating, stopping, or initiating an avoidance maneuver. It concerns the intended action underlying future movement, rather than only the resulting trajectory. Relevant evidence includes recent motion changes, vessel maneuverability, waterway geometry, and the surrounding encounter situation. Regional ship behavior prediction provides a maritime example of learning behavior from AIS observations (Murray and Perera, 2021). Related transportation studies have explored semantic information for intention prediction (Xiao et al., 2026) and visual intention cues for trajectory forecasting (Zhong et al., 2023), ofering methodological references for incorporating contextual evidence. Their application to maritime navigation nevertheless requires adaptation to vessel dynamics and maritime operating conditions.

Navigational Safety of Marine Surface Vehicles  
![](images/42f99aae1056ef7019ce43ae999184aeeb92808097f98b542907e2e9c83aa02c.jpg)  
Figure 11: Overview of vessel behavior prediction for navigational safety in restricted visibility. Behavior prediction involves estimating the future movements, trajectories, speeds, and destinations of vessels of interest. It is able to enhance maritime safety, strengthen maritime security, optimize port and trafic management, etc. In current literature, vessel behavior prediction mainly includes motion and trajectory prediction, route and destination prediction, maneuvering intention prediction, and interaction-aware behavior prediction in this work.

Interaction-aware behavior prediction estimates how a vessel’s future movement depends on nearby trafic participants. During crossing, overtaking, or head-on encounters, a vessel may alter its course or speed in response to the actions of other vessels. Spatio-temporal graph networks represent dependencies among vessel trajectories (Liu et al., 2022c), while interaction-aware deep generative models forecast future motion conditioned on surrounding trafic (Han et al., 2024). Related forecasting research has developed mechanisms for social interaction modeling (Alahi et al., 2016; Gupta et al., 2018), vectorized map and agent representations (Gao et al., 2020; Liang et al., 2020), and attention-based multi-agent prediction (Yuan et al., 2021; Shi et al., 2022). These mechanisms ofer useful references, but maritime applications must account for sparse and noisy AIS observations, waterway structure, and vessel-specific maneuverability. The International Regulations for Preventing Collisions at Sea (COLREGs) also provide a normative context for encounter behavior, although translating their provisions into computational models remains challenging (García Maza and Poo Argüelles, 2022; Chang et al., 2024).

These prediction tasks are complementary: destinations and routes provide navigational goals, maneuvering intentions indicate likely actions, and interaction models describe responses to surrounding vessels. Together, they support trajectory forecasts that account for both individual movement and the evolving trafic situation. Uncertainty should be considered across all these tasks because incomplete observations and alternative behavioral choices can lead to multiple plausible futures. Vessel behavior prediction therefore extends beyond forecasting where a vessel will move to estimating what it is likely to do and how its behavior may evolve within the maritime environment.

![](images/620bb324e9d8752ddf2388d95feab208e615f61eea13765eb2fd9f0b6f02b19e.jpg)  
Figure 12: Conceptual framework for intelligent collision avoidance in restricted visibility. Fused observations, behavior forecasts, and operating context support encounter evaluation and collision-risk assessment. Rule- and knowledge-based, model-based, and learning-based methods can be combined to generate avoidance decisions under safety, COLREGs, and maneuverability constraints. Guidance and motion control execute the resulting maneuvers, while updated observations close the feedback loop.

## 5.3. Intelligent Collision Avoidance

Intelligent collision avoidance enables marine surface vehicles to assess potential conflicts and select avoidance maneuvers while considering navigational safety, trafic rules, and vessel maneuverability (Huang et al., 2020b; Zhang et al., 2021c). Its importance becomes particularly evident in restricted visibility, where degraded visual observations can delay the detection of surrounding vessels and obscure the development of hazardous encounters. Within the scope of this review, intelligent collision avoidance is a downstream application through which intelligent vision and vision-centered multi-sensor fusion contribute to navigational safety. By combining visual information with complementary sensor observations, the perception system can provide the trafic-state estimates needed to assess collision risk and determine appropriate responses.

Fig. 12 illustrates the closed-loop process connecting upstream perception and behavior prediction with encounter evaluation, avoidance decision-making, and downstream guidance and motion control. Object detection, tracking, and multi-sensor fusion estimate the positions and movements of surrounding vessels, while behavior prediction provides possible future trajectories and associated uncertainties (Han et al., 2024; Capobianco et al., 2023; Wang et al., 2024b). These outputs support encounter assessment and avoidance decisions, which specify changes in course or speed, target waypoints, or a planned trajectory. Guidance and control then translate the selected maneuver into executable vessel motion (Tao et al., 2024). Updated observations allow the system to reassess the encounter as the maneuver proceeds.

The efectiveness of this process therefore depends on both the reliability of upstream information and the feasibility of downstream execution.

Collision-risk assessment provides a common foundation for diferent avoidance methods. Conventional indicators, including distance to closest point of approach (DCPA) and time to closest point of approach (TCPA), characterize the proximity and timing of a predicted encounter. Ship-domain-based approaches further evaluate whether and when another vessel may enter a protected region around the own ship (Szlapczynski and Szlapczynska, 2016). Probabilistic ship domains represent safety boundaries using data-driven probability distributions (Zhang and Meng, 2019), while environmental risk models incorporate hydrometeorological conditions into ship–ship collision-risk assessment (Zhang et al., 2021a). These developments broaden risk assessment beyond relative geometry, which is relevant when restricted visibility and uncertain observations afect the interpretation of an encounter.

Rule- and knowledge-based methods determine avoidance actions through explicit representations of navigational rules and seamanship. Quantitative modeling of the International Regulations for Preventing Collisions at Sea (COLREGs) supports encounter classification and the selection of rule-consistent maneuvers (He et al., 2017). This approach makes the reasoning behind an avoidance decision relatively transparent. However, converting contextual rule provisions into computational conditions remains dificult, particularly when several vessels interact or the available observations are incomplete. Critical analyses and recent reviews identify COLREGs interpretation and algorithmic formalization as continuing challenges for autonomous collision avoidance (García Maza and Poo Argüelles, 2022; Chang et al., 2024).

Model-based methods generate avoidance maneuvers using explicit representations of vessel motion, encounter geometry, and navigational constraints. Generalized velocity obstacles identify collision-threatening regions in velocity space and support the selection of alternative vessel velocities (Huang et al., 2019). Simulation-based control selection evaluates candidate behaviors through predictive hazard assessment while considering COLREGs compliance (Johansen et al., 2016). Multi-ship approaches further address avoidance under simultaneous encounter constraints (Cho et al., 2022). These methods provide a structured basis for evaluating candidate actions, but their predictions and decisions depend on the accuracy of vessel-state estimates, motion models, and assumptions about surrounding trafic. Uncertainty from upstream perception and behavior prediction consequently needs to be considered when assessing maneuver safety.

Learning-based methods infer avoidance policies from data or simulated interactions. Deep reinforcement learning has been used to learn avoidance actions for unmanned surface vehicles (Woo and Kim, 2020), with subsequent work incorporating COLREGs-related objectives into policy learning (Xu et al., 2020). More recent studies address the balance between collision avoidance and navigation eficiency in complex environments (Lou et al., 2024). These methods can represent nonlinear relationships between trafic observations and vessel actions, although performance depends on the coverage of training scenarios and the transferability of learned policies. For restricted-visibility applications, robustness to degraded or incomplete perception is therefore an important consideration.

Hybrid approaches combine complementary elements of these methods to connect rule interpretation, predictive planning, and executable control. COLREGs-compliant hybrid frameworks integrate avoidance mechanisms for diferent navigational situations (Eriksen et al., 2020), while integrated guidance and motion-control methods link anticipatory decisions with vessel execution (Tao et al., 2024). Recent research also considers uncertain target-ship intentions in avoidance decision-making (Li et al., 2026). Collectively, these developments highlight the need to carry uncertainty from perception and behavior prediction into risk assessment and maneuver selection. For navigational safety in restricted visibility, the central requirement is therefore an integrated process in which vision-centered multi-sensor fusion supports reliable trafic understanding, prediction identifies potential conflicts, and avoidance decisions remain feasible under vessel dynamics and applicable navigational rules.

## 5.4. Vision-Centered Navigation Systems

Vision-centered data fusion has increasingly been incorporated into assisted and autonomous maritime navigation systems, as illustrated in Fig. 13. Furuno Envision is an augmented-reality (AR)-based maritime situational-awareness system developed by Furuno Electric Co., Ltd. It combines visible-light cameras, AIS, and radar/chart information to display surrounding vessels and obstacles and support navigation under adverse weather and visibility conditions<sup>2</sup>. Orca AI incorporates thermal cameras to complement visual perception under dense haze, nighttime, and reduced visibility<sup>3</sup>. Satellite communication services such as Starlink can further improve connectivity in remote waters and support maritime operations.

![](images/bc0d82a94bacd3c11b8f7fa9a15cc2d5df87cc36bc150b70b8685ba1ce549229.jpg)  
(a) Furuno Envision

![](images/343e7b618bb80ee27d012942cece73ec28ea6633aec1318c0801bc6b49b8f905.jpg)  
(b) Orca Al

![](images/2c2847696b87ad0b4c8c197fba23b3c7edbbfc4006d2cd7f9866c418d7a8a3ac.jpg)  
(c) Avikus HiNAS

![](images/2468ab54abe46d1e31b90178a4234b6f4c80e0f6b54dfd026fb451a8eab9a8b9.jpg)  
(d) Marautec i-EYE  
Figure 13: Examples of vision-centered navigation systems. From top-left to bottom-right: (a) Furuno Envision AR-based navigation system; (b) marine situational awareness system developed by Orca AI Ltd.; (c) Hyundai intelligent navigation assistant system (HiNAS) with assisted navigation (top) and berthing/unberthing assistance (bottom); and (d) Marautec i-EYE with several practical capabilities.

The Hyundai Intelligent Navigation Assistant System (HiNAS)<sup>4</sup> integrates AI, AR, and multi-sensor data fusion and has been deployed on large vessels, including very large ore carriers (VLOCs), container ships, and liquefied natural gas (LNG) carriers. Its functions include obstacle detection, route following, collision avoidance, and docking assistance. Marautec i-EYE is another representative vision-centered navigation system<sup>5</sup>, integrating intelligent vision and data fusion to support target perception and early warning in restricted visibility.

The Maritime RobotX Challenge has also promoted the development of intelligent navigation systems for USVs. Team Angry-Nerds from the Korea Advanced Institute of Science and Technology (KAIST) developed a USV integrating monocular cameras, LiDAR, and hydrophones and completed five tasks involving navigation and control, underwater search and reporting, autonomous docking, buoy search and observation, and obstacle detection and avoidance (Park et al., 2017). Clunie et al. (2021) developed a Robot Operating System (ROS)-based ASV perception system integrating monocular visible-light cameras, LiDAR, and radar for object detection and tracking, semantic segmentation, and localization.

By integrating wall detection, object detection, path planning, and control, Kim et al. (2024a) converted a passenger cruise boat into an ASV and evaluated autonomous navigation in the Pohang Canal. The experiments also highlighted practical challenges, including degraded GPS localization beneath steel bridges and motion disturbances caused by wind and waves. For restricted-visibility navigation, Liu et al. (2022a) combined lightweight YOLOX-based visual detection with AIS messages to construct an AR-assisted intelligent navigation system. To improve AR-based maritime interaction, Nordby et al. (2024) introduced a user-interface design concept linking digital navigation information with the physical environment. In addition, Din et al. (2023) developed the catamaran USV marine, whose navigation system relies on vision-based target detection and tracking to support operation in complex navigation conditions.

Navigational Safety of Marine Surface Vehicles  
![](images/fa9eb70a24394478bbf7c68ad3aa2608193bc59d511631a9c477bbfdbea0e4b2.jpg)  
Figure 14: Overview of the open challenges and future research directions for navigational safety in restricted visibility. The five central topics, i.e., datasets and benchmarks, robust imaging in severely restricted visibility, object detection in complex conditions, multi-sensor fusion in dynamic environments, and scene understanding and reasoning, are organized from their key challenges to promising future research directions.

## 6. Open Challenges and Future Directions

Despite the substantial progress reviewed above, guaranteeing navigational safety for marine surface vehicles in restricted visibility still confronts several open challenges and motivates promising research directions. In this section, we organize the discussion around five central topics, as shown in Fig. 14: datasets and benchmarks, imaging and perception enhancement, object detection, multi-sensor data fusion, and scene understanding and reasoning. For each topic, we first analyze the key challenges and then outline the corresponding future directions.

## 6.1. Datasets and Benchmarks for Restricted Visibility

Challenges. Dense fog, low illumination, rain, and atmospheric scattering can substantially degrade maritime visual data, resulting in poor visibility, blurred object boundaries, and unstable feature representations (Yao et al., 2024b). Under such restricted-visibility conditions, optical sensors often sufer from texture loss, low contrast, and disappearance of long-range targets, reducing the reliability of detection, segmentation, and tracking. Maritime environments further introduce dynamic water reflections, irregular target motion, and rapidly changing weather, making perception models sensitive to domain shifts and performance degradation in real-world deployment. Existing datasets, however, remain insuficient for robust perception research in restricted visibility. Most public maritime datasets are collected primarily under clear-weather conditions and rely predominantly on RGB imagery, while relatively few provide adverse-weather scenarios or visibility-aware annotations. Current benchmarks also rarely characterize degradation severity, sensor reliability, or long-range observability in extreme conditions, and the limited availability of large-scale multimodal datasets covering dense fog and other restricted-visibility scenarios therefore remains a major obstacle to reliable all-weather intelligent waterway perception.

Future directions. Future research should emphasize robust multimodal perception in restricted visibility. Compared with conventional cameras, complementary modalities such as 4D millimeter-wave radar, thermal imaging, and LiDAR can compensate for information loss when optical sensing is severely degraded. Adaptive multimodal fusion, uncertaintyaware perception, and cross-modal feature enhancement are therefore promising directions for improving perception stability in adverse maritime environments. Integrating physics-informed degradation modeling and domain adaptation may further reduce the gap between synthetic and real adverse-weather conditions and improve model generalization.

![](images/6b4969a1f0c9cd516e80831d1ab5f80252dec6cb7892439e1c2468b1db83e5b4.jpg)

(a) HADAR Perception  
![](images/7d7a89c1c92c4b27d78d4c94a5f3e72811a36f0cc67f6cdfc5e30a351332b96d.jpg)  
(b) Tesla Photon-Count Reconstruction  
Figure 15: Emerging imaging technologies for enhanced visibility. Illustrations of two advanced imaging technologies, i.e., (a) HADAR and (b) Tesla’s AI photon count reconstruction, which can improve MSV perception in dense-haze scenarios.

Another important direction is the construction of comprehensive restricted-visibility maritime datasets and standardized robustness benchmarks. Future datasets should contain synchronized multimodal observations under diverse fog densities, illumination conditions, and navigation scenarios, together with fine-grained annotations of visibility levels, sensor degradation states, and environmental attributes. Beyond conventional accuracy metrics, evaluation protocols should assess robustness, temporal stability, and failure characteristics under severe environmental interference. Automatic annotation, self-supervised learning, and simulation-to-real transfer may further reduce annotation costs and improve dataset scalability, supporting the development of reliable all-weather intelligent waterway transportation.

## 6.2. Robust Imaging in Severely Restricted Visibility

Challenges. Dense haze is common in maritime environments and can contribute to severe navigational risks, economic losses, and casualties. Such conditions induce strong atmospheric scattering and light attenuation that sharply reduce image contrast and visible range, causing distant targets to fade or even vanish, while low illumination, rain, and nighttime glare impose similarly severe yet distinct degradations. Despite extensive eforts in both hardware- and software-based perception, satisfactory imaging under such conditions remains dificult because advanced sensing hardware can be costly, while conventional imaging methods may provide limited visual quality in severely restricted visibility. Consequently, reliable visual perception in such degraded conditions remains a fundamental obstacle, motivating the exploration of emerging imaging technologies.

Future directions. Emerging imaging technologies originally developed for other domains may provide useful directions for improving MSV situational awareness under adverse weather and imaging conditions. For example, Bao et al. (2023) presented the heat-assisted detection and ranging (HADAR), which combines thermal imaging, physical modeling, and machine learning for low-visibility perception, shown in Fig. 15(a). In particular, the HADAR extracts physical information from infrared radiation, capable of recovering textural and depth information under low-visibility conditions, providing a potential sensing approach for MSVs operating in dense haze or low-light environments.

Tesla’s AI photon-count reconstruction represents another potential direction for recovering scene information when conventional RGB imagery is severely degraded by challenging and dynamic illumination, shown in Fig. 15(b). Its ability to reconstruct visual information under low-light or glare conditions extends naturally to maritime perception in harsh lighting, dense haze, and other restricted-visibility scenarios. Deploying this approach in real maritime environments is a promising direction for robust perception under harsh illumination.

## 6.3. Object Detection in Complex Conditions

Challenges. Reliable object detection in restricted visibility is challenged by both image degradation and the intrinsic complexity of maritime scenes. Vessels, buoys, and other obstacles exhibit substantial variations in scale, shape, texture, and boundary appearance across distances, viewpoints, and weather conditions. Distant small objects are particularly dificult because they occupy few pixels and often appear as low-contrast, weak-response regions with blurred contours and limited texture. Dense fog, low illumination, rain, and strong water-surface reflections further suppress these cues and can cause confusion with waves, wakes, shoreline structures, and background highlights, leading to missed detections, false alarms, and unstable localization. Partial occlusion, dense target distributions, ship-shore background overlap, and viewpoint changes further reduce detection stability. Existing maritime detectors also face limitations in feature representation, background suppression, and cross-scale detection. Most rely on local appearance cues and conventional multi-scale feature pyramids, which may be insuficient for distant weak targets, low-texture objects, and severe background interference. Deep downsampling can weaken or remove small-object information, whereas shallow features are readily disturbed by water textures, reflection highlights, rain-fog noise, and shoreline clutter, and boundary blurring, dynamic water-surface interference, occlusion, and scale variation can also cause localization errors and unreliable confidence estimates.

Future directions. All-weather and all-time object detection is important for reliable maritime perception. Because visible-light sensors are vulnerable to adverse maritime conditions, future studies should further integrate infrared, radar, LiDAR, and other sensing modalities to construct multimodal detection frameworks for complex environments. Particular attention should be paid to small, distant, and weak targets in restricted visibility, which often exhibit small spatial extent, weak textures, blurred boundaries, and low target-background contrast. Detectors should therefore improve shallow-detail preservation, high-resolution representation learning, and cross-scale feature interaction.

Lightweight design and edge deployment are also important for practical MSV applications. Shipboard platforms typically have limited computational resources and strict real-time requirements, whereas many existing detectors remain computationally demanding. Future work could investigate maritime-oriented lightweight architectures, pruning, quantization, knowledge distillation, and dynamic inference to reduce latency and computational complexity while maintaining detection accuracy and robustness.

## 6.4. Multi-Sensor Fusion in Dynamic Environments

Challenges. Multi-sensor fusion in restricted visibility is further complicated by the pronounced discrepancies among heterogeneous modalities. Navigation radar measures range and velocity in polar coordinates and is commonly represented in a BEV framework, whereas cameras map the environment onto a two-dimensional perspective image, creating a large cross-modal gap in sensing geometry and data representation. Calibration-based alignment is sensitive to drift caused by vessel pitching and long-term vibration, and adaptive matching remains dificult in highly dynamic sea states. For water-surface targets, radar echoes are often sparse and strongly contaminated by multipath clutter, while LiDAR points sufer from wave-induced noise and provide limited appearance information for recognition. Under dense hazy, heavy rain, or nighttime glare, individual sensors may also fail or degrade asynchronously, producing inconsistent, low-confidence, or missing observations. Reasoning over such heterogeneous and partially unreliable evidence while maintaining temporal consistency therefore remains a key challenge for robust fusion.

Future directions. Future research should develop adaptive and uncertainty-aware fusion frameworks that dynamically reweight modalities according to their instantaneous reliability. Learnable cross-modal association, automatic online calibration, and probabilistic data association can reduce dependence on precise initial extrinsic parameters and accommodate dynamic sensor ofsets. Integrating fusion with occupancy modeling and unified BEV representations may further bridge perception and path planning in restricted visibility.

![](images/451859b18f6027cb5a3d2ac29a871f9acc5b45e81bc2a68f0cbf3b8b6f98504c.jpg)  
Figure 16: Conceptual framework for maritime scene understanding and reasoning under restricted visibility. Multimodal observations support scene understanding, maritime reasoning, and verified decision support, with dashed arrows indicating evidence refinement and uncertainty feedback.

Beyond pairwise fusion, heterogeneous fusion of radar, LiDAR, cameras, infrared, AIS, and other shipboard sensors deserves further exploration, since complementary modalities can compensate for individual failures. Lightweight and edge-deployable fusion architectures are also important for shipboard MSV platforms with limited computational resources. Uncertainty-aware fusion should further be combined with rule-based and probabilistic safety checks to support reliable collision avoidance in highly dynamic maritime environments.

## 6.5. Scene Understanding and Reasoning

Challenges. Beyond object-level perception, guaranteeing navigational safety further requires scene-level understanding and reasoning, which becomes substantially more dificult in restricted visibility. A safe maneuver depends not only on detecting vessels, buoys, and obstacles, but also on interpreting their spatial relations, encounter geometries, motion intentions, and potential collision risks. Dense fog, low illumination, rain, and atmospheric scattering degrade the very cues on which such reasoning relies: distant targets lose texture and contrast, inter-object spatial relationships become ambiguous, and temporal information is corrupted by motion blur, sensor noise, and intermittent occlusions. Systems must therefore reason over uncertain, incomplete, and sometimes contradictory observations, which greatly aggravates the dificulty of semantic scene comprehension and risk assessment. Existing scene understanding and reasoning methods, especially those built upon vision-language models and multimodal agents, remain insuficient under such conditions. Although recent studies have demonstrated the feasibility of vision-language maneuvering, structured prompt-based risk assessment, and multi-agent waterway question answering (Kim and Choi, 2025; Kim and Yoo, 2025; Guan et al., 2026a; Salgado et al., 2026), these methods are mostly evaluated on clear-weather or simulated scenes and may generalize poorly to degraded maritime environments. Current models also exhibit a limited grasp of maritime-specific semantics, weak spatial and geometric reasoning, insuficient temporal consistency, and a tendency to hallucinate when visual evidence is ambiguous, while their outputs rarely quantify uncertainty or provide verifiable evidence.

Future directions. Multimodal agents provide a potential framework for connecting low-level perception with highlevel navigational understanding. For MSVs, navigational safety requires not only detecting ships, buoys, waterlines, and obstacles, but also interpreting encounter relationships, potential risks, and possible responses. Multimodal large language model (MLLM)-powered agents can therefore serve as a semantic reasoning layer above conventiona perception modules, translating heterogeneous sensor observations into scene descriptions, risk interpretations, and decision-support information. Early maritime studies have already explored such vision-language reasoning. Vision-LLM-based maneuvering has been investigated in simulated MSV navigation scenarios, demonstrating the feasibility of vision-language reasoning for maneuver selection (Kim and Choi, 2025). Structured prompt-based vision-language reasoning further integrates object detection, image captioning, risk assessment, and maneuver recommendation for maritime decision support (Kim and Yoo, 2025). More recently, WaterVideoQA introduced a video question-answering benchmark for waterway scenes and proposed NaviMind, a multi-agent reasoning system comprising routing, captioning, reasoning, grading, and summary agents (Guan et al., 2026a). In parallel, USV-3.0 explores cognitive maritime navigation using vision-language models, human feedback, and spatio-temporal navigation memory (Salgado et al., 2026). These studies indicate an emerging transition from passive perception toward interactive maritime cognition.

Fig. 16 illustrates a conceptual framework connecting multimodal observations, structured scene understanding, maritime reasoning, and verified decision support, with feedback for evidence refinement and uncertainty assessment. Future research on multimodal maritime agents could proceed along several directions. Domain-specific maritime MLLMs could serve as reasoning backbones and be trained or adapted using synchronized multimodal data and structured representations, including visible, thermal, radar, LiDAR, AIS, electronic-chart, weather, and vessel-motion information, and should capture not only object semantics but also encounter geometry, sensor reliability, environmental degradation, COLREGs constraints, and long-horizon temporal evolution. Maritime agents should also be uncertainty-aware and verification-oriented, explicitly characterizing missing observations, inconsistent sensor evidence, low-confidence detections, and potential hallucinations, with safety-critical decisions further checked against rule-based risk models, probabilistic prediction, and formally constrained planning. Future benchmarks should evaluate multimodal agents beyond visual question-answering accuracy, considering factual consistency, temporal stability, COLREGs compliance, risk-ranking correctness, explanation quality, latency, and failure modes under dense haze, heavy rain, nighttime glare, sensor dropout, and challenging trafic encounters. Finally, human-in-the-loop design remains important, with multimodal agents serving as explainable assistants whose interfaces support transparent evidence retrieval, traceable outputs, and conservative escalation when visibility or sensor confidence deteriorates.

## 7. Conclusions

This work has provided a comprehensive survey of the techniques for guaranteeing the navigational safety of MSVs in restricted visibility, spanning from intelligent vision to vision-centered multi-sensor data fusion. We first reviewed intelligent vision methods that enhance shipboard perception under adverse weather and illumination, including shipboard video stabilization, visual perception enhancement for hazy, low-light, and multi-scene degradation, object detection and tracking, and navigable area segmentation. We then examined vision-centered multi-sensor data fusion, covering infrared-visible image fusion, radar-vision fusion, LiDAR-vision fusion, AIS-vision fusion, and heterogeneous multi-sensor fusion, and discussed how complementary modalities compensate for the limitations of visible-light cameras in restricted visibility. To ground these techniques in practical scenarios, we further summarized publicly available maritime datasets and analyzed safety-critical applications, including vessel behavior prediction, intelligent collision avoidance, and vision-centered navigation systems.

Despite the substantial progress, guaranteeing navigational safety in restricted visibility still confronts several open challenges, including the scarcity of visibility-aware multimodal benchmarks, robust imaging in severely degraded conditions, small-object detection in complex scenes, adaptive and uncertainty-aware fusion in dynamic environments, and scene understanding and reasoning with vision-language agents. We believe that future research should move from vision-only perception toward uncertainty-aware, rule-aware, and human-in-the-loop navigation. This survey is intended to serve as a structured reference for newcomers and researchers working on the navigational safety of intelligent surface vehicles in IWTS.

## CRediT authorship contribution statement

Ryan Wen Liu: Writing – review & editing, Writing – original draft, Supervision, Resources, Funding acquisition. Fengchuan Song: Writing – review & editing, Writing – original draft, Validation, Methodology. Tianxiang Wang: Writing – review & editing, Writing – original draft, Validation, Methodology. Guobei Peng: Writing – review & editing, Writing – original draft, Validation, Methodology. Youngrong Kim: Writing – original draft, Methodology. Xian Zhong: Writing – original draft, Methodology. Hooi-Siang Kang: Writing – original draft, Methodology. Maohan Liang: Writing – review & editing, Writing – original draft, Validation, Methodology. Junzhi Yu: : Writing – review & editing, Writing – original draft,Supervision. Shanliang Yao: Writing – review & editing, Writing – original draft, Funding acquisition.

## Declaration of competing interest

This manuscript has not been published or presented elsewhere in part or in entirety and is not under consideration by another journal. We have read and understood your journal’s policies, and we believe that neither the manuscript nor the study violates any of these. There are no conflicts of interest to declare.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 52422111, 52271365, and 62271361; the Excellent Youth Foundation of Hubei Scientific Committee under Grant 2024AFA042; the Key Research and Development Program of Hainan Province under Grant ZDYF2026GXJS024; the Key Research and Development Program of Hubei Province under Grant 2024BAB039; the Fundamental Research Funds for the Central Universities under Grant 104972026KFYzxk0029; and Basic Research Program of Jiangsu under Grant BK20251091.

## References

Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Li, F., Savarese, S., 2016. Social LSTM: Human trajectory prediction in crowded spaces, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 961–971.

Ali, M.K., Im, E.W., Kim, D., Kim, T.H., 2024. Harnessing meta-learning for improving full-frame video stabilization, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 12605–12614.

Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., Tai, C., 2022. TransFusion: Robust LiDAR-camera fusion for 3D object detection with transformers, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 1090–1099.

Bakht, A.B., Din, M.U., Javed, S., Hussain, I., 2025. MVTD: A benchmark dataset for maritime visual object tracking. arXiv preprint arXiv:2506.02866

Bao, F., Wang, X., Sureshbabu, S.H., Sreekumar, G., Yang, L., Aggarwal, V., Boddeti, V.N., Jacob, Z., 2023. Heat-assisted detection and ranging. Nature 619, 743–748.

Bloisi, D.D., Previtali, F., Pennisi, A., Nardi, D., Fiorini, M., 2017. Enhancing automatic maritime surveillance systems with visual information. IEEE Trans. Intell. Transp. Syst. (TITS) 18, 824–833.

Bochkovskiy, A., Wang, C., Liao, H.M., 2020. YOLOv4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934 .

Bovcon, B., Kristan, M., 2020. A water-obstacle separation and refinement network for unmanned surface vehicles, in: Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pp. 9470–9476.

Bovcon, B., Kristan, M., 2022. WaSR—a water segmentation and refinement maritime obstacle detection network. IEEE Trans. Cybern. (TCYB) 52, 12661–12674.

Bovcon, B., Mandeljc, R., Perš, J., Kristan, M., 2018. Stereo obstacle detection for unmanned surface vehicles by IMU-assisted semantic segmentation. Robot. Auton. Syst. (RAS) 104, 1–13.

Bovcon, B., Muhovič, J., Perš, J., Kristan, M., 2019. The MaSTr1325 dataset for training deep USV obstacle detection models, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 3431–3438.

Bovcon, B., Muhovič, J., Vranac, D., Mozetič, D., Perš, J., Kristan, M., 2022. MODS—a USV-oriented object detection and obstacle segmentation benchmark. IEEE Trans. Intell. Transp. Syst. (TITS) 23, 13403–13418.

Cao, K., Xu, S., Zhu, Z., Shi, X., Chong, S., Ding, W., Liang, M., Yao, S., Liu, R.W., 2026. Visual perception for autonomous surface vehicles in complex waterway environments: A review. Appl. Ocean Res. 175, 105222.

Capobianco, S., Forti, N., Millefiori, L.M., Braca, P., Willett, P., 2023. Recurrent encoder–decoder networks for vessel trajectory prediction with uncertainty estimation. IEEE Trans. Aerosp. Electron. Syst. (TAES) 59, 2554–2565.

Capobianco, S., Millefiori, L.M., Forti, N., Braca, P., Willett, P., 2021. Deep learning methods for vessel trajectory prediction based on recurrent neural networks. IEEE Trans. Aerosp. Electron. Syst. (TAES) 57, 4329–4346.

Chang, C., Wijeratne, I.B., Kontovas, C., Yang, Z., 2024. COLREG and MASS: Analytical review to identify research trends and gaps in the development of autonomous collision avoidance. Ocean Eng. 302, 117652.

Chang, J., Wang, S., Xu, H., Chen, Z., Yang, C., Zhao, F., 2023. DETRDistill: A universal knowledge distillation framework for DETR-families, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 6898–6908.

Chen, G., Choi, W., Yu, X., Han, T., Chandraker, M., 2017a. Learning eficient object detection models with knowledge distillation, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 742–751.

Chen, J., Kao, S., He, H., Zhuo, W., Wen, S., Lee, C., Chan, S.G., 2023. Run, don’t walk: Chasing higher FLOPs for faster neural networks, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 12021–12031.

Chen, L., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L., 2018. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 40, 834–848.

Chen, M., Sun, J., Aida, K., Takefusa, A., 2024. Weather-aware object detection method for maritime surveillance systems. Future Gener. Comput. Syst. (FGCS) 151, 111–123.

Chen, X., Liu, Y., Achuthan, K., 2021a. WODIS: Water obstacle detection network based on image segmentation for autonomous surface vehicles in maritime environments. IEEE Trans. Instrum. Meas. (TIM) 70, 7503213.

Chen, X., Ma, H., Wan, J., Li, B., Xia, T., 2017b. Multi-view 3D object detection network for autonomous driving, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 1907–1915.

Chen, Z., Chen, D., Zhang, Y., Cheng, X., Zhang, M., Wu, C., 2020. Deep learning for autonomous ship-oriented small ship detection. Saf. Sci. 130, 104812.

Chen, Z., Huang, T., Xue, Z., Zhu, Z., Xu, J., Liu, Y., 2021b. A novel unmanned surface vehicle with 2D-3D fused perception and obstacle avoidance module, in: Proc. IEEE Int. Conf. Robot. Biomimetics (ROBIO), pp. 1804–1809.

Chen, Z., Li, Z., Zhang, S., Fang, L., Jiang, Q., Zhao, F., Zhou, B., Zhao, H., 2022. AutoAlign: Pixel-instance feature aggregation for multi-modal 3D object detection. arXiv preprint arXiv:2201.06493 .

Cheng, Y., Jiang, M., Zhu, J., Liu, Y., 2021a. Are we ready for unmanned surface vehicles in inland waterways? the USVInland multisensor dataset and benchmark. IEEE Robot. Autom. Lett. (RA-L) 6, 3964–3970.

Cheng, Y., Wang, D., Zhou, P., Zhang, T., 2017. A survey of model compression and acceleration for deep neural networks. arXiv preprint arXiv:1710.09282 .

Cheng, Y., Xu, H., Liu, Y., 2021b. Robust small object detection on the water surface through fusion of camera and millimeter wave radar, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 15263–15272.

Cheng, Y., Zhu, J., Jiang, M., Fu, J., Pang, C., Wang, P., Sankaran, K., Onabola, O., Liu, Y., Liu, D., Bengio, Y., 2021c. FloW: A dataset and benchmark for floating waste detection in inland waters, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 10953–10962.

Cho, Y., Han, J., Kim, J., 2022. Eficient COLREG-compliant collision avoidance in multi-ship encounter situations. IEEE Trans. Intell. Transp. Syst. (TITS) 23, 1899–1911.

Choi, J., Cho, D., Lee, G., Kim, H., Yang, G., Kim, J., Cho, Y., 2025. Polaris dataset: A maritime object detection and tracking dataset in Pohang canal, in: Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pp. 13626–13632.

Clunie, T., DeFilippo, M., Sacarny, M., Robinette, P., 2021. Development of a perception system for an autonomous surface vehicle using monocular camera, LiDAR, and marine radar, in: Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pp. 14112–14119.

Cormack, D., Schlangen, I., Hopgood, J.R., Clark, D.E., 2020. Joint registration and fusion of an infrared camera and scanning radar in a maritime context. IEEE Trans. Aerosp. Electron. Syst. (TAES) 56, 1357–1369.

Cui, Y., Ren, W., Knoll, A., 2026. Bio-inspired image restoration, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 80452–80481.

Deng, Y., Zhong, X., Liu, W., Yu, Z., Yuan, J., Huang, T., 2026. HAD: Hierarchical asymmetric distillation to bridge spatio-temporal gaps in event-based object tracking. IEEE Trans. Neural Netw. Learn. Syst. (TNNLS) .

Denk, F., Rankl, C., Almouahed, S., Moser, D., Sablatnig, R., 2026. SeaClips: A video dataset for maritime object detection, in: Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 4599–4610.

Din, M.U., Bakht, A.B., Akram, W., Dong, Y., Seneviratne, L., Hussain, I., 2025. Benchmarking vision-based object tracking for USVs in complex maritime environments. IEEE Access 13, 15014–15027.

Din, M.U., Humais, A., Akram, W., Alblooshi, M., Saoud, L.S., Alblooshi, A., Seneviratne, L., Hussain, I., 2023. Marine: Design and implementation of unmanned surface vessel for vision guided navigation, in: Proc. Int. Conf. Adv. Robot. (ICAR), pp. 226–231.

Ding, H., Weng, J., 2024. A robust assessment of inland waterway collision risk based on AIS and visual data fusion. Ocean Eng. 307, 118242.

Ding, X., Zhang, Y., Ge, Y., Zhao, S., Song, L., Yue, X., Shan, Y., 2024a. UniRepLKNet: A universal perception large-kernel ConvNet for audio video point cloud time-series and image recognition, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5513–5524.

Ding, Y., Xu, Y., Liu, Q., Sun, H., Chen, F., 2024b. Research on boundary-aware waters segmentation network for unmanned surface vehicles in complex inland waters. Int. J. Mach. Learn. Cybern. 15, 2297–2308.

Dobrkovic, A., Iacob, M., van Hillegersberg, J., 2018. Maritime pattern extraction and route reconstruction from incomplete AIS data. Int. J. Data Sci. Anal. 5, 111–136.

El Mahdi, B.M., 2024. A novel multispectral vessel recognition based on RGB-to-thermal image translation. Unmanned Syst. 12, 627–640.

Eriksen, B.H., Bitar, G., Breivik, M., Lekkas, A.M., 2020. Hybrid collision avoidance for ASVs compliant with COLREGs rules 8 and 13–17. Front. Robot. AI 7, 11.

Eustache, Y., Seguin, C., Pecout, A., Foucher, A., Laurent, J., Heller, D., 2025. Marine object detection using LiDAR on an unmanned surface vehicle. IEEE Access 13, 121658–121669.

Fan, L., Wang, J., Chang, Y., Li, Y., Wang, Y., Cao, D., 2024. 4D mmWave radar for autonomous driving perception: A comprehensive survey. IEEE Trans. Intell. Veh. (TIV) 9, 4606–4620.

Fan, Y., Niu, L., Liu, T., 2022. Multi-branch gated fusion network: A method that provides higher-quality images for the USV perception system in maritime hazy condition. J. Mar. Sci. Eng. (JMSE) 10, 1839.

Farahnakian, F., Haghbayan, M., Poikonen, J., Laurinen, M., Nevalainen, P., Heikkonen, J., 2018. Object detection based on multi-sensor proposal fusion in maritime environment, in: Proc. IEEE Int. Conf. Mach. Learn. Appl. (ICMLA), pp. 971–976.

Farahnakian, F., Heikkonen, J., 2020. Deep learning based multi-modal fusion architectures for maritime vessel detection. Remote Sens. 12, 2509.

Farahnakian, F., Poikonen, J., Laurinen, M., Makris, D., Heikkonen, J., 2019. Visible and infrared image fusion framework based on RetinaNet for marine environment, in: Proc. Int. Conf. Inf. Fusion (FUSION), pp. 1–7.

Feng, H., Liu, W., Xu, H., He, J., 2024a. A lightweight dual-branch semantic segmentation network for enhanced obstacle detection in ship navigation. Eng. Appl. Artif. Intell. (EAAI) 136, 108982.

Feng, H., Tang, W., Xu, H., Jiang, C., Ge, S.S., He, J., 2024b. Meta-learning based infrared ship object detection model for generalization to unknown domains. Appl. Soft Comput. (ASC) 159, 111633.

Forti, N., d’Aflisio, E., Braca, P., Millefiori, L.M., Carniel, S., Willett, P., 2022. Next-gen intelligent situational awareness systems for maritime surveillance and autonomous navigation [point of view]. Proc. IEEE 110, 1532–1537.

Galdran, A., Vazquez-Corral, J., Pardo, D., Bertalmio, M., 2015. Enhanced variational image dehazing. SIAM J. Imaging Sci. (SIIMS) 8, 1519–1546.

Gao, J., Sun, C., Zhao, H., Shen, Y., Anguelov, D., Li, C., Schmid, C., 2020. VectorNet: Encoding HD maps and agent dynamics from vectorized representation, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 11525–11533.

Gao, J., Zhang, J., Liu, C., Li, X., Peng, Y., 2022. Camera-LiDAR cross-modality fusion water segmentation for unmanned surface vehicles. J. Mar. Sci. Eng. (JMSE) 10, 744.

Gao, T., Wen, Y., Zhang, K., Zhang, J., Chen, T., Liu, L., Luo, W., 2024. Frequency-oriented eficient transformer for all-in-one weather-degraded image restoration. IEEE Trans. Circuits Syst. Video Technol. (TCSVT) 34, 1886–1899.

García Maza, J.A., Poo Argüelles, R., 2022. COLREGs and their application in collision avoidance algorithms: A critical analysis. Ocean Eng. 261, 112029.

Ghiasi, G., Lin, T., Le, Q.V., 2019. NAS-FPN: Learning scalable feature pyramid architecture for object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 7036–7045.

Gong, Y., Jin, J., Liu, D., Ren, P., 2025. A real-time lightweight object detection algorithm based on improved you only look once version 8 for unmanned surface vehicle. Eng. Appl. Artif. Intell. (EAAI) 152, 110798.

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2014. Generative adversarial nets, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS).

Grini, J.T., Mester, R., Nygård, T.A., Dalhaug, N., Brekke, E.F., Stahl, A., 2024. FusedWSS: Water surface segmentation fusing machine learning and geometric cues, in: Proc. Int. Conf. Inf. Fusion (FUSION), pp. 1–8.

Guan, R., Jia, L., Yao, S., Yang, F., Xu, S., Purwanto, E., Zhu, X., Man, K.L., Lim, E.G., Smith, J., Hu, X., Yue, Y., 2025a. WaterVG: Waterway visual grounding based on text-guided vision and mmWave radar. IEEE Trans. Intell. Transp. Syst. (TITS) 26, 7275–7291.

Guan, R., Liang, S., Ouyang, N., Fei, W., Yao, S., Dai, W., Ge, C., Sun, P., Zhu, X., Huang, T., Liu, R.W., Xiong, H., 2026a. WaterVideoQA: ASV-centric perception and rule-compliant reasoning via multi-modal agents. arXiv preprint arXiv:2602.22923 .

Guan, R., Liu, J., Jia, L., Zhao, H., Yao, S., Zhu, X., Man, K.L., Lim, E.G., Smith, J., Yue, Y., 2025b. NanoMVG: USV-centric low-power multi-task visual grounding based on prompt-guided camera and 4D mmWave radar, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 11789–11796.

Guan, R., Yao, S., Liu, L., Zhu, X., Man, K.L., Yue, Y., Smith, J., Lim, E.G., Yue, Y., 2024a. Mask-VRDet: A robust riverway panoptic perception model based on dual graph fusion of vision and 4D mmWave radar. Robot. Auton. Syst. (RAS) 171, 104572.

Guan, R., Yao, S., Man, K.L., Zhu, X., Yue, Y., Smith, J., Lim, E.G., Yue, Y., 2024b. ASY-VRNet: Waterway panoptic driving perception model based on asymmetric fair fusion of vision and 4D mmWave radar, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 12479–12486.

Guan, R., Yao, S., Zhu, X., Man, K.L., Lim, E.G., Smith, J., Yue, Y., Yue, Y., 2023. Achelous: A fast unified water-surface panoptic perception framework based on fusion of monocular camera and 4D mmWave radar, in: Proc. IEEE Intell. Transp. Syst. Conf. (ITSC), pp. 182–188.

Guan, R., Zhao, H., Yao, S., Yu, L., Zhu, X., Liu, R.W., Lim, E.G., Ding, W., Yue, Y., Xiong, H., 2026b. Achelous++: Power-oriented multi-task panoptic waterway perception framework based on vision-radar fusion. Appl. Soft Comput. (ASC) 192, 114787.

Guilluy, W., Oudre, L., Beghdadi, A., 2021. Video stabilization: Overview, challenges and perspectives. Signal Process. Image Commun. 90, 116015.

Guo, D., Yin, Y., Jing, Q., Shao, Z., Xu, H., 2026. A LiDAR intensity-enhanced 3D object detection method for maritime autonomous navigation. Ocean Eng. 343, 123586.

Guo, L., Wang, Y., Guo, M., Zhou, X., 2025a. YOLO-IRS: Infrared ship detection algorithm based on self-attention mechanism and KAN in complex marine background. Remote Sens. 17, 20.

Guo, Y., He, S., Lu, Y., An, H., Tao, Y., Zhu, H., Liu, J., Fang, Y., 2025b. Neptune-X: Active X-to-maritime generation for universal maritime object detection, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 146587–146614.

Guo, Y., Liu, R.W., Lu, Y., Nie, J., Lyu, L., Xiong, Z., Kang, J., Yu, H., Niyato, D., 2023a. Haze visibility enhancement for promoting trafic situational awareness in vision-enabled intelligent transportation. IEEE Trans. Veh. Technol. (TVT) 72, 15421–15435.

i h b h j hi b d l i d l i i d f i f l fi surveillance in inland waterways. IEEE Trans. Intell. Transp. Syst. (TITS) 24, 12779–12792.

Guo, Y., Lu, Y., Liu, R.W., 2022. Lightweight deep network-enabled real-time low-visibility enhancement for promoting vessel detection in maritime video surveillance. J. Navig. 75, 230–250.

Guo, Y., Shen, Q., Ai, D., Wang, H., Zhang, S., Wang, X., 2024. Sea-IoUTracker: A more stable and reliable maritime target tracking scheme for unmanned vessel platforms. Ocean Eng. 299, 117243.

Guo, Y., Yu, H., Ma, L., Zeng, L., Luo, X., 2023c. THFE: A triple-hierarchy feature enhancement method for tiny boat detection. Eng. Appl. Artif. Intell. (EAAI) 123, 106271.

Gupta, A., Johnson, J., Li, F., Savarese, S., Alahi, A., 2018. Social GAN: Socially acceptable trajectories with generative adversarial networks, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 2255–2264.

Haghbayan, M., Farahnakian, F., Poikonen, J., Laurinen, M., Nevalainen, P., Plosila, J., Heikkonen, J., 2018. An eficient multi-sensor fusion approach for object detection in maritime environments, in: Proc. IEEE Intell. Transp. Syst. Conf. (ITSC), pp. 2163–2170.

Han, J., Cho, Y., Kim, J., Kim, J., Son, N., Kim, S., 2020a. Autonomous collision detection and avoidance for ARAGON USV: Development and field tests. J. Field Robot. (JFR) 37, 987–1002.

Han, J., Park, H., Park, K., Kim, S., Lee, J.Y., Pyo, C., 2023. Field demonstration of advanced autonomous navigation technique for a fully unmanned surface vehicle in complex coastal trafic areas. J. Field Robot. (JFR) 40, 1887–1905.

Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., Xu, C., 2020b. GhostNet: More features from cheap operations, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 1580–1589.

Han, P., Zhu, M., Zhang, H., 2024. Interaction-aware short-term marine vessel trajectory prediction with deep generative models. IEEE Trans. Ind. Inform. (TII) 20, 3188–3196.

Hao, K., Deng, Z., Wang, B., Jin, Z., Li, Z., Zhao, X., 2025. Lightweight multiobject ship tracking algorithm based on trajectory association and improved YOLOv7tiny. Expert Syst. Appl. (ESWA) 259, 125129.

He, X., Ji, W., 2023. Single maritime image dehazing using unpaired adversarial learning. Signal Image Video Process. (SIVP) 17, 593–600.

He, X., Li, J., Wang, C., Li, G., 2025. Complexity expert-guided visual perception enhancement for boosting maritime navigational safety under low-visibility scenes. Ocean Eng. 340, 122325.

He, X., Wu, D., Wu, D., You, Z., Zhong, S., Liu, Q., 2024. Millimeter-wave radar and camera fusion for multiscenario object detection on USVs. IEEE Sens. J. (JSEN) 24, 31562–31572.

He, Y., Jin, Y., Huang, L., Xiong, Y., Chen, P., Mou, J., 2017. Quantitative analysis of COLREG rules and seamanship for autonomous collision avoidance at open sea. Ocean Eng. 140, 281–291.

Helgesen, Ø.K., Vasstein, K., Brekke, E.F., Stahl, A., 2022. Heterogeneous multi-sensor tracking for an autonomous surface vehicle in a littoral environment. Ocean Eng. 252, 111168.

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 6840–6851.

Holst, F., Gülsoylu, E., Frintrop, S., 2026. Fusing monocular RGB images with AIS data to create a 3-D bounding box estimation data set for marine vessels. IEEE J. Oceanic Eng. (JOE) 51, 1676–1688.

Hong, X., Cui, B., Chen, W., Rao, Y., Chen, Y., 2022. Research on multi-ship target detection and tracking method based on camera in complex scenes. J. Mar. Sci. Eng. (JMSE) 10, 978.

Howard, A., Sandler, M., Chen, B., Wang, W., Chen, L., Tan, M., Chu, G., Vasudevan, V., Zhu, Y., Pang, R., Adam, H., Le, Q., 2019. Searching for MobileNetV3, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 1314–1324.

Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H., 2017. MobileNets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 .

Hu, H., Guo, Q., Zheng, J., Wang, H., Li, B., 2019. Single image defogging based on illumination decomposition for visual maritime surveillance. IEEE Trans. Image Process. (TIP) 28, 2882–2897.

Hu, Q., Zhang, Y., Zhu, Y., Jiang, Y., Song, M., 2023a. Single image dehazing algorithm based on sky segmentation and optimal transmission maps. Vis. Comput. (TVC) 39, 997–1013.

Hu, Z., Shao, J., Nie, F., Luo, Z., Chen, C., Xiao, L., 2023b. Robust online learning based on siamese network for ship tracking. Sci. Rep. 13, 7358.

Huang, T., Liu, Z., Chen, X., Bai, X., 2020a. EPNet: Enhancing point features with image semantics for 3D object detection, in: Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 35–52.

Huang, Y., Chen, L., Chen, P., Negenborn, R.R., van Gelder, P.H.A.J.M., 2020b. Ship collision avoidance methods: State-of-the-art. Saf. Sci. 121, 451–473.

Huang, Y., Chen, L., van Gelder, P.H.A.J.M., 2019. Generalized velocity obstacle algorithm for preventing ship collisions at sea. Ocean Eng. 173, 142–156.

Huang, Z., Hu, Q., Lu, L., Mei, Q., Yang, C., 2023. Online estimation of ship dimensions by combining images with AIS reports. J. Mar. Sci. Eng. (JMSE) 11, 1700.

Im, S.J., Yun, C., Lee, S.J., Park, K.R., 2025. Artificial intelligence-based low-light marine image enhancement for semantic segmentation in edge intelligence empowered internet of things environment. IEEE Internet Things J. (IoTJ) 12, 4086–4114.

Isola, P., Zhu, J., Zhou, T., Efros, A.A., 2017. Image-to-image translation with conditional adversarial networks, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 1125–1134.

Johansen, T.A., Perez, T., Cristofaro, A., 2016. Ship collision avoidance and COLREGs compliance using simulation-based control behavior selection with predictive hazard assessment. IEEE Trans. Intell. Transp. Syst. (TITS) 17, 3407–3422.

Kaur, P., Aziz, A., Jain, D., Patel, H., Hirokawa, J., Townsend, L., Reimers, C., Hua, F., 2022. Sea situational awareness (SEASAW) dataset, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 2579–2587.

Kim, D., Yoo, J., 2025. Structured prompt-based vision–language reasoning for risk assessment and navigation decisions in maritime navigation. J. Mar. Sci. Eng. (JMSE) 13, 2339.

Kim, J., Lee, C., Chung, D., Cho, Y., Kim, J., Jang, W., Park, S., 2024a. Field experiment of autonomous ship navigation in canal and surrounding nearshore environments. J. Field Robot. (JFR) 41, 470–489.

Kim, J., Lee, C., Chung, D., Kim, J., 2023. Navigable area detection and perception-guided model predictive control for autonomous navigation in narrow waterways. IEEE Robot. Autom. Lett. (RA-L) 8, 5456–5463.

Kim, K., Kim, J., Kim, J., 2021. Robust data association for multi-object detection in maritime environments using camera and radar measurements. IEEE Robot. Autom. Lett. (RA-L) 6, 5865–5872.

Kim, T., Choi, W., 2025. Autonomous vehicle maneuvering using vision–LLM models for marine surface vehicles. J. Mar. Sci. Eng. (JMSE) 13, 1553.

Kim, Y., Park, J., Kang, S., Kim, H., 2024b. Introducing VaDA: Novel image segmentation model for maritime object segmentation using new dataset. arXiv preprint arXiv:2407.09005 .

Kristan, M., Kenk, V.S., Kovačič, S., Perš, J., 2016. Fast image-based obstacle detection from unmanned surface vehicles. IEEE Trans. Cybern. (TCYB) 46, 641–654.

Lan, G., Peng, Y., Hao, Q., Xu, C., 2025. SUSTechGAN: Image generation for object detection in adverse conditions of autonomous driving. IEEE Trans. Intell. Veh. (TIV) 10, 4854–4864.

Land, E.H., 1977. The Retinex theory of color vision. Sci. Am. 237, 108–129.

Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O., 2019. PointPillars: Fast encoders for object detection from point clouds, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 12697–12705.

Li, C., Guo, C., Han, L., Jiang, J., Cheng, M., Gu, J., Loy, C.C., 2022a. Low-light image and video enhancement using deep learning: A survey. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 44, 9396–9416.

Li, K., Dai, Z., Wang, X., Song, Y., Jeon, G., 2024. GAN-based controllable image data augmentation in low-visibility conditions for improved roadside trafic perception. IEEE Trans. Consum. Electron. (TCE) 70, 6174–6188.

Li, X., Kou, K., Zhao, B., 2021. Weather GAN: Multi-domain weather translation using generative adversarial networks. arXiv preprint arXiv:2103.05422 .

Li, Y., Wu, D., Li, G., Fan, Z., Yuan, Z., 2026. Collision avoidance decision-making method for unmanned surface vehicles considering uncertain navigation intent of target ships. IEEE Trans. Intell. Transp. Syst. (TITS) 27, 4750–4765.

Li, Y., Yu, A.W., Meng, T., Caine, B., Ngiam, J., Peng, D., Shen, J., Lu, Y., Zhou, D., Le, Q.V., Yuille, A., Tan, M., 2022b. DeepFusion: LiDAR-camera deep fusion for multi-modal 3D object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 17182–17191.

Liang, J., Bao, M., Wang, T., Wang, X., Liu, A., Liu, R.W., 2026. WUTDet: A 100K-scale ship detection dataset and benchmarks with dense small objects. arXiv preprint arXiv:2604.07759 .

Liang, M., Liu, R.W., Li, S., Xiao, Z., Liu, X., Lu, F., 2021. An unsupervised learning method with convolutional auto-encoder for vessel trajectory similarity computation. Ocean Eng. 225, 108803.

Liang, M., Yang, B., Chen, Y., Hu, R., Urtasun, R., 2019. Multi-task multi-sensor fusion for 3D object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 7345–7353.

Liang, M., Yang, B., Hu, R., Chen, Y., Liao, R., Feng, S., Urtasun, R., 2020. Learning lane graph representations for motion forecasting, in: Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 541–556.

Lin, T., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S., 2017. Feature pyramid networks for object detection, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 2117–2125.

Liu, J., Li, H., Luo, J., Xie, S., Sun, Y., 2021a. Eficient obstacle detection based on prior estimation network and spatially constrained mixture model for unmanned surface vehicles. J. Field Robot. (JFR) 38, 212–228.

Liu, J., Xu, D., Yang, W., Fan, M., Huang, H., 2021b. Benchmarking low-light image enhancement and beyond. Int. J. Comput. Vis. (IJCV) 129, 1153–1184.

Liu, R.W., Guo, Y., Lu, Y., Chui, K.T., Gupta, B.B., 2023a. Deep network-enabled haze visibility enhancement for visual IoT-driven intelligent transportation systems. IEEE Trans. Ind. Inform. (TII) 19, 1581–1591.

Liu, R.W., Guo, Y., Nie, J., Hu, Q., Xiong, Z., Yu, H., Guizani, M., 2022a. Intelligent edge-enabled eficient multi-source data fusion for autonomous surface vehicles in maritime internet of things. IEEE Trans. Green Commun. Networking (TGCN) 6, 1574–1587.

Liu, R.W., Han, C., Huang, Y., 2024a. Low-light visibility enhancement for improving visual surveillance in intelligent waterborne transportation systems. IET Intell. Transp. Syst. (IET ITS) 18, 1632–1651.

Liu, R.W., Huo, X., Liang, M., Wang, K., 2022b. Ship collision risk analysis: Modeling, visualization and prediction. Ocean Eng. 266, 112895.

Liu, R.W., Liang, J., Wang, H., Bao, M., 2026. NavEYE: Vision–centered multi–sensor fusion–based situational awareness system for intelligent surface vehicles. Ocean Eng. 363, 126781.

Liu, R.W., Liang, M., Nie, J., Yuan, Y., Xiong, Z., Yu, H., Guizani, N., 2022c. STMGCN: Mobile edge computing-empowered vessel trajectory prediction using spatio-temporal multigraph convolutional network. IEEE Trans. Ind. Inform. (TII) 18, 7977–7987.

Liu, R.W., Liu, N., Huang, Y., Guo, Y., 2022d. Attention-guided lightweight generative adversarial network for low-light image enhancement in maritime video surveillance. J. Navig. 75, 1100–1117.

Liu, R.W., Lu, Y., Gao, Y., Guo, Y., Ren, W., Zhu, F., Wang, F., 2024b. Real-time multi-scene visibility enhancement for promoting navigational safety of vessels under complex weather conditions. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 19979–19994.

Liu, R.W., Lu, Y., Guo, Y., Ren, W., Zhu, F., Lv, Y., 2024c. AiOENet: All-in-one low-visibility enhancement to improve visual perception for intelligent marine vehicles under severe weather conditions. IEEE Trans. Intell. Veh. (TIV) 9, 3811–3826.

Liu, R.W., Nie, J., Garg, S., Xiong, Z., Zhang, Y., Hossain, M.S., 2021c. Data-driven trajectory quality improvement for promoting intelligent vessel trafic services in 6G-enabled maritime IoT systems. IEEE Internet Things J. (IoTJ) 8, 5374–5385.

Liu, X., Li, Y., Wu, Y., Wang, Z., He, W., Li, Z., 2021d. A hybrid method for inland ship recognition using marine radar and closed-circuit television. J. Mar. Sci. Eng. (JMSE) 9, 1199.

Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D.L., Han, S., 2023b. BEVFusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation, in: Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pp. 2774–2781.

Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y., Tegmark, M., 2024d. KAN: Kolmogorov-Arnold networks. arXiv preprint arXiv:2404.19756 .

Liu, Z., Zhang, Y., Yu, X., Yuan, C., 2016. Unmanned surface vehicles: An overview of developments and challenges. Annu. Rev. Control 41, 71–93.

Lou, M., Yang, X., Hu, J., Zhu, Z., Shen, H., Xiang, Z., Zhang, B., 2024. A balanced collision avoidance algorithm for USVs in complex environment: A deep reinforcement learning approach. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 21404–21415.

Lu, C., Qin, H., Deng, Z., Zhu, Z., 2023. Fusion2Fusion: An infrared–visible image fusion algorithm for surface water environments. J. Mar. Sci. Eng. (JMSE) 11, 902.

Lu, Y., Tian, J., Su, Y., Liu, J., Hao, C., Yue, C., Yang, X., 2025. Polarization-aware low-light image enhancement for nighttime intelligent vehicles. IEEE Trans. Intell. Veh. (TIV) 10, 1944–1958.

Lu, Y., Yang, K., Yang, D., Ding, H., Weng, J., Liu, R.W., 2026. Graph learning-driven multi-vessel association: Fusing multimodal data for maritime intelligence. IEEE Trans. Intell. Transp. Syst. (TITS) 27, 5739–5754.

Ma, D., Chen, S., Dong, L., Chen, T., Xu, W., 2026a. Spatio-semantic enhanced cascade swin detection network for infrared maritime targets. IEEE Trans. Geosci. Remote Sens. (TGRS) 64, 1–13.

Ma, J., Ma, Y., Li, C., 2019. Infrared and visible image fusion methods and applications: A survey. Inf. Fusion 45, 153–178.

Ma, J., Zheng, Y., Lyu, Y., Jia, C., Su, Y., 2026b. A multi-object tracking framework with feature backtracking under severe occlusion. Ocean Eng. 351, 124175.

Ma, R., Bao, K., Yin, Y., 2022. Improved ship object detection in low-illumination environments using RetinaMFANet. J. Mar. Sci. Eng. (JMSE) 10, 1996.

Ma, Z., Wen, J., Zhang, C., Liu, Q., Yan, D., 2016. An efective fusion defogging approach for single sea fog image. Neurocomputing 173, 1257–1267.

Moosbauer, S., Konig, D., Jakel, J., Teutsch, M., 2019. A benchmark for deep learning based object detection in maritime environments, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), pp. 916–925.

Mousazadeh, H., Jafarbiglu, H., Abdolmaleki, H., Omrani, E., Monhaseri, F., Abdollahzadeh, M., Mohammadi-Aghdam, A., Kiapei, A., Salmani Zakaria, Y., Makhsoos, A., 2018. Developing a navigation, guidance and obstacle avoidance algorithm for an unmanned surface vehicle (USV) by algorithms fusion. Ocean Eng. 159, 56–65.

Muhovič, J., Perš, J., 2025. MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation. arXiv preprint arXiv:2512.17450 .

Murray, B., Perera, L.P., 2021. An AIS-based deep learning framework for regional ship behavior prediction. Reliab. Eng. Syst. Saf. (RESS) 215, 107819.

Nordby, K., Fauske, J.E., Gernez, E., Mallam, S., 2024. A user interface design framework for augmented-reality-supported maritime navigation. J. Mar. Sci. Eng. (JMSE) 12, 505.

Paigwar, A., Sierra-Gonzalez, D., Erkent, Ö., Laugier, C., 2021. Frustum-PointPillars: A multi-stage approach for 3D object detection using RGB camera and LiDAR, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 2926–2933.

Park, D., Lee, B.H., Chun, S.Y., 2023. All-in-one image restoration for unknown degradations using adaptive discriminative filters for specific degradations, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5815–5824.

Park, H., Ham, S., Kim, T., An, D., 2022. Object recognition and tracking in moving videos for maritime autonomous surface ships. J. Mar. Sci. Eng. (JMSE) 10, 841.

Park, J., Kang, M., Kim, T., Kwon, S., Han, J., Wang, J., Yoon, S., Yoo, B., Hong, S., Shim, Y., Park, J., Kim, J., 2017. Development of an unmanned surface vehicle system for the 2014 Maritime RobotX Challenge. J. Field Robot. (JFR) 34, 644–665.

Peng, Y., Li, H., Wu, P., Zhang, Y., Sun, X., Wu, F., 2025. D-FINE: Redefine regression task of DETRs as fine-grained distribution refinement, in: Proc. Int. Conf. Learn. Represent. (ICLR).

Ponzini, F., Zaccone, R., Martelli, M., 2025. LiDAR target detection and classification for ship situational awareness: A hybrid learning approach. Appl. Ocean Res. 158, 104552.

Potlapalli, V., Zamir, S.W., Khan, S.H., Khan, F.S., 2023. PromptIR: Prompting for all-in-one image restoration, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 71275–71293.

Qi, C.R., Liu, W., Wu, C., Su, H., Guibas, L.J., 2018. Frustum PointNets for 3D object detection from RGB-D data, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 918–927.

Qian, C., Guo, Y., Mo, Y., Li, W., 2025. WeatherDG: LLM-assisted procedural weather generation for domain-generalized semantic segmentation. IEEE Robot. Autom. Lett. (RA-L) 10, 5919–5926.

Qu, J., Liu, R.W., Gao, Y., Guo, Y., Zhu, F., Wang, F., 2024. Double domain guided real-time low-light image enhancement for ultra-high-definition transportation surveillance. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 9550–9562.

Qu, J., Liu, R.W., Guo, Y., Lu, Y., Su, J., Li, P., 2023. Improving maritime trafic surveillance in inland waterways using the robust fusion of AIS and visual data. Ocean Eng. 275, 114198.

Rajagopalan, S., Nair, N.G., Paranjape, J.N., Patel, V.M., 2025. GenDeg: Difusion-based degradation synthesis for generalizable all-in-one image restoration, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 28144–28154.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M., 2022. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125 .

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 10684–10695.

Rong, H., Teixeira, A.P., Guedes Soares, C., 2019. Ship trajectory uncertainty prediction based on a Gaussian process model. Ocean Eng. 182, 499–511.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional networks for biomedical image segmentation, in: Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Interv. (MICCAI), pp. 234–241.

Rothmeier, T., Huber, W., Knoll, A.C., 2024. Time to shine: Fine-tuning object detection models with synthetic adverse weather images, in: Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 4447–4456.

Salgado, A., Vasconcellos, E.C., Guerra, R., Gonçalves, L.M.G., Clua, E.W.G., 2026. USV-3.0: Cognitive maritime navigation through vision-language models, human-in-the-loop learning, and spatio-temporal memory. Ocean Eng. 355, 125010.

Schwendeman, M., Thomson, J., 2015. A horizon-tracking method for shipboard video stabilization and rectification. J. Atmos. Ocean. Technol. (JTECH) 32, 164–176.

Shan, Y., Yao, X., Lin, H., Zou, X., Huang, K., 2021a. LiDAR-based stable navigable region detection for unmanned surface vehicles. IEEE Trans. Instrum. Meas. (TIM) 70, 8501613.

Shan, Y., Zhou, X., Liu, S., Zhang, Y., Huang, K., 2021b. SiamFPN: A deep learning method for accurate and real-time maritime ship tracking. IEEE Trans. Circuits Syst. Video Technol. (TCSVT) 31, 315–325.

Shao, H., Yang, J., Chen, G., Xie, Y., Qin, H., Huang, L., 2022a. High-quality rainy image generation method for autonomous driving based on few-shot learning, in: Proc. Int. Conf. Algorithms Comput. Artif. Intell. (ACAI), pp. 1–7.

Shao, Z., Lyu, H., Yin, Y., Cheng, T., Gao, X., Zhang, W., Jing, Q., Zhao, Y., Zhang, L., 2022b. Multi-scale object detection model for autonomous ship navigation in maritime environment. J. Mar. Sci. Eng. (JMSE) 10, 1783.

Shao, Z., Wang, L., Wang, Z., Du, W., Wu, W., 2020. Saliency-aware convolution neural network for ship detection in surveillance video. IEEE Trans. Circuits Syst. Video Technol. (TCSVT) 30, 781–794.

Shao, Z., Yin, Y., Lyu, H., Soares, C.G., 2024. A robust method for multi object tracking in autonomous ship navigation systems. Ocean Eng. 311, 118560.

Shi, K., He, S., Shi, Z., Chen, A., Xiong, Z., Chen, J., Luo, J., 2026. Radar and camera fusion for object detection and tracking: A comprehensive survey. IEEE Commun. Surv. Tutor. (COMST) 28, 3478–3520.

Shi, S., Jiang, L., Dai, D., Schiele, B., 2022. Motion transformer with global intention localization and local movement refinement, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 6531–6543.

Shin, K., Kwon, Y.P., Tomizuka, M., 2019. RoarNet: A robust 3D object detection based on region approximation refinement, in: Proc. IEEE Intell. Veh. Symp. (IV), pp. 2510–2515.

Shin, Y., Kim, H., Kim, J., 2024. AnytimeFusion: Parameter-free RGB camera-radar sensor fusion algorithm in complex maritime situations, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 12493–12500.

Song, F., Yao, S., Liang, M., Kim, Y., Liu, R.W., 2026. An uncertainty-aware 4D millimeter wave radar–camera fusion method for robust environmental perception under dynamic maritime conditions. Ocean Eng. 367, 127925.

Roberto e Souza, M., Maia, H.d.A., Pedrini, H., 2022. Survey on digital video stabilization: Concepts, methods, and challenges. ACM Comput. Surv. (CSUR) 55, 1–37.

Stanislas, L., Dunbabin, M., 2019. Multimodal sensor fusion for robust obstacle detection and classification in the Maritime RobotX Challenge. IEEE J. Oceanic Eng. (JOE) 44, 343–351.

Sun, B., Tang, H., Gao, L., Bi, K., Wen, J., 2025. RTDETR-MARD: A multi-scale adaptive real-time framework for floating waste detection in aquatic environments. J. Mar. Sci. Eng. (JMSE) 13, 996.

Sun, Y., Lian, J., 2025. IRSD-Net: An adaptive infrared ship detection network for small targets in complex maritime environments. Remote Sens. 17, 2643.

Szlapczynski, R., Szlapczynska, J., 2016. An analysis of domain-based ship collision risk parameters. Ocean Eng. 126, 47–56.

Tan, M., Le, Q., 2019. EficientNet: Rethinking model scaling for convolutional neural networks, in: Proc. Int. Conf. Mach. Learn. (ICML), pp. 6105–6114.

Tan, M., Pang, R., Le, Q.V., 2020. EficientDet: Scalable and eficient object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 10781–10790.

Tao, Y., Du, J., Lewis, F.L., 2024. Integrated intelligent guidance and motion control of USVs with anticipatory collision avoidance decision-making. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 17810–17820.

Teixeira, E.H., Coutinho, D.A., Mafra, S.B., De Figueiredo, F.A.P., 2026. Computer vision and machine learning approach for correlating optical coordinates with AIS data in maritime surveillance. IEEE Geosci. Remote Sens. Lett. (GRSL) .

Tian, Y., Meng, H., 2025. Coarse-to-fine dual-branch network for ship target recognition in complex environments. Eng. Appl. Artif. Intell. (EAAI) 161, 112120.

Tian, Y., Meng, H., Yuan, F., 2024. FREGNet: Ship recognition based on feature representation enhancement and GCN combiner in complex environment. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 15641–15653.

Tu, E., Zhang, G., Rachmawati, L., Rajabally, E., Huang, G., 2018. Exploiting AIS data for intelligent maritime navigation: A comprehensive survey from data to methodology. IEEE Trans. Intell. Transp. Syst. (TITS) 19, 1559–1582.

Vora, S., Lang, A.H., Helou, B., Beijbom, O., 2020. PointPainting: Sequential fusion for 3D object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4604–4612.

Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., Ding, G., 2024a. YOLOv10: Real-time end-to-end object detection, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 107984–108011.

Wang, C., Han, P., Zhu, M., Osen, O., Zhang, H., Li, G., 2024b. AIS data-based hybrid predictor for short-term ship trajectory prediction considering uncertainties. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 20268–20281.

Wang, C., He, W., Nie, Y., Guo, J., Liu, C., Wang, Y., Han, K., 2023a. Gold-YOLO: Eficient object detector via gather-and-distribute mechanism, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), pp. 51094–51112.

Wang, C., Ma, C., Zhu, M., Yang, X., 2021a. PointAugmenting: Cross-modal augmentation for 3D object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 11794–11803.

Wang, H., Tang, B., Cai, F., Li, M., Zhang, Z., Fan, D., 2026. FGOM-RTDETR: Far-shore guided object-focusing multiscale network with real-time detection transformer for infrared ship target detection. IEEE Trans. Geosci. Remote Sens. (TGRS) 64, 1–16.

Wang, L., Xiao, Y., Zhang, B., Liu, R., Zhao, B., 2023b. Water surface targets detection based on the fusion of vision and LiDAR. Sensors 23, 1768.

Wang, N., Wang, Y., Feng, Y., Wei, Y., 2024c. MDD-ShipNet: Math-data integrated defogging for fog-occlusion ship detection. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 15040–15052.

Wang, N., Wang, Y., Wei, Y., Han, B., Feng, Y., 2024d. Marine vessel detection dataset and benchmark for unmanned surface vehicles. Appl. Ocean Res. 142, 103835.

Wang, N., Zhou, C., Zhu, R., Zhang, B., Wang, Y., Liu, H., 2024e. SOFT: Self-supervised sparse optical flow transformer for video stabilization via quaternion. Eng. Appl. Artif. Intell. (EAAI) 130, 107725.

Wang, W., Peng, Y., Cao, G., Guo, X., Kwok, N., 2021b. Low-illumination image enhancement for night-time UAV pedestrian detection. IEEE Trans. Ind. Inform. (TII) 17, 5208–5217.

Wang, X., Liu, J., Liu, X., Liu, Z., Khalaf, O.I., Ji, J., Ouyang, Q., 2022. Ship feature recognition methods for deep learning in complex marine environments. Complex Intell. Syst. 8, 3881–3897.

Wang, Y., Huang, Q., Jiang, C., Liu, J., Shang, M., Miao, Z., 2023c. Video stabilization: A comprehensive survey. Neurocomputing 516, 205–230.

Wei, Q., Jiang, X., Liu, Y., Su, Q., Yu, M., 2024. Small object detection on the water surface based on radar and camera fusion, in: Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), pp. 8381–8385.

Wen, L., Yingjun, Z., Xuefeng, Y., 2019. A feature-based method for shipboard video stabilization, in: Proc. IEEE Int. Conf. Electron. Inf. Commun. Technol. (ICEIC), pp. 315–322.

Wojke, N., Bewley, A., Paulus, D., 2017. Simple online and realtime tracking with a deep association metric, in: Proc. IEEE Int. Conf. Image Process. (ICIP), pp. 3645–3649.

Woo, J., Kim, N., 2020. Collision avoidance for an unmanned surface vehicle using deep reinforcement learning. Ocean Eng. 199, 107001.

Wu, P., Su, S., Tong, X., Guo, R., Sun, B., Zuo, Z., Zhang, J., 2023. SARFB: Strengthened asymmetric receptive field block for accurate infrared ship detection. IEEE Sens. J. (JSEN) 23, 5028–5044.

Wu, X., Peng, L., Yang, H., Xie, L., Huang, C., Deng, C., Liu, H., Cai, D., 2022. Sparse fuse dense: Towards high quality 3D detection with depth completion, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 5418–5427.

Xiao, J., Lu, J., Li, J., Wei, M., Zhu, Z., Han, X., 2025. A novel and robust AIS-camera fusion method for situational awareness in autonomous ships, in: Proc. Chin. Control Decis. Conf. (CCDC), pp. 2764–2769.

Xiao, Y., Liu, W., Huang, W., Ma, J., Liu, R.W., Zhong, X., 2026. TCP: Text-guided cascade network for pedestrian crossing intention prediction. IEEE Trans. Intell. Transp. Syst. (TITS) 27, 831–841.

Xie, Y., Nanlal, C., Liu, Y., 2024. Reliable LiDAR-based ship detection and tracking for autonomous surface vehicles in busy maritime environments. Ocean Eng. 312, 119288.

Xie, Y., Wu, P., Englot, B., Nanlal, C., Liu, Y., 2026. Uncertainty-aware maritime point cloud detector (u-MPCD) for autonomous surface vehicles. IEEE J. Oceanic Eng. (JOE) 51, 220–242.

Xin, Z., Kong, S., Wu, Y., Zhan, G., Yu, J., 2022. A hierarchical stabilization control method for a three-axis gimbal based on sea–sky-line detection. Sensors 22, 2587.

Xing, L., Qu, H., Xu, S., Tian, Y., 2023. CLEGAN: Toward low-light image enhancement for UAVs via self-similarity exploitation. IEEE Trans. Geosci. Remote Sens. (TGRS) 61, 1–14.

Xu, H., Geng, Z., He, J., Zhang, X., Yu, Y., 2026. Markless boat berth detection for autonomous surface ships through fusion of LiDAR and camera. IEEE J. Oceanic Eng. (JOE) 51, 1238–1254.

Xu, H., Yu, Y., He, J., Pang, C., Zhang, X., 2025. Multi-stage fusion object detection with marine radar and camera in complex maritime context. IEEE Trans. Instrum. Meas. (TIM) 74, 2532414.

Xu, H., Zhang, X., He, J., Geng, Z., Pang, C., Yu, Y., 2024a. Surround-view water surface BEV segmentation for autonomous surface vehicles: Dataset, baseline and hybrid-BEV network. IEEE Trans. Intell. Veh. (TIV) 9, 7264–7278.

Xu, H., Zhang, X., He, J., Yu, Y., Cheng, Y., 2024b. Real-time volumetric perception for unmanned surface vehicles through fusion of radar and camera. IEEE Trans. Instrum. Meas. (TIM) 73, 5015912.

Xu, S., Hu, J., Wang, M., Mu, T., Hu, S., 2018. Deep video stabilization using adversarial networks. Comput. Graph. Forum (CGF) 37, 267–276.

Xu, X., Lu, Y., Liu, X., Zhang, W., 2020. Intelligent collision avoidance algorithms for USVs via deep reinforcement learning under COLREGs. Ocean Eng. 217, 107704.

Yang, H., Pan, L., Yang, Y., Liang, W., 2024. Language-driven all-in-one adverse weather removal, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 24902–24912.

Yang, X., Zhu, H., Zhao, H., Yang, D., 2023. Coastal ship tracking with memory-guided perceptual network. Remote Sens. 15, 3150.

Yang, Z., Li, Y., Wang, B., Ding, S., Jiang, P., 2022. A lightweight sea surface object detection network for unmanned surface vehicles. J. Mar. Sci. Eng. (JMSE) 10, 965.

Yao, R., Wang, H., Guo, Y., Xie, Z., 2025a. A real-time fusion of two-stage point cloud clustering and saliency image for water surface object detection. Remote Sens. 17, 3708.

Yao, S., Guan, R., Huang, X., Li, Z., Sha, X., Yue, Y., Lim, E.G., Seo, H., Man, K.L., Zhu, X., Yue, Y., 2024a. Radar-camera fusion for object detection and semantic segmentation in autonomous driving: A comprehensive review. IEEE Trans. Intell. Veh. (TIV) 9, 2094–2128.

Yao, S., Guan, R., Ni, Y., Xu, S., Yue, Y., Zhu, X., Liu, R.W., 2025b. USVTrack: USV-based 4D radar-camera tracking dataset for autonomous driving in inland waterways, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 19016–19023.

Yao, S., Guan, R., Peng, Z., Xu, C., Shi, Y., Yue, Y., Ding, W., Lim, E.G., Seo, H., Man, K.L., Ma, J., Zhu, X., Yue, Y., 2025c. Exploring radar data representations in autonomous driving: A comprehensive review. IEEE Trans. Intell. Transp. Syst. (TITS) 26, 7401–7425.

Yao, S., Guan, R., Wu, Z., Ni, Y., Huang, Z., Liu, R.W., Yue, Y., Ding, W., Lim, E.G., Seo, H., Man, K.L., Ma, J., Zhu, X., Yue, Y., 2024b. WaterScenes: A multi-task 4D radar-camera fusion dataset and benchmarks for autonomous driving on water surfaces. IEEE Trans. Intell. Transp. Syst. (TITS) 25, 16584–16598.

Yao, S., Li, Z., Guan, R., Xia, M., Hu, F., Cao, K., Xu, S., Zhu, X., Liu, R.W., Ding, W., 2026. GIF-Calib: Geometry-intensity fusion for 4D radar-camera calibration in autonomous driving vehicles. Inf. Fusion 136, 104510.

Ye, H., Zhang, B., Chen, T., Fan, J., Wang, B., 2023. Performance-aware approximation of global channel pruning for multitask CNNs. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 45, 10267–10284.

Yoo, J.H., Kim, Y., Kim, J., Choi, J.W., 2020. 3D-CVF: Generating joint camera and LiDAR features using cross-view spatial feature fusion for 3D object detection, in: Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 720–736.

Yu, C., Yin, H., Rong, C., Zhao, J., Liang, X., Li, R., Mo, X., 2024. YOLO-MRS: An eficient deep learning-based maritime object detection method for unmanned surface vehicles. Appl. Ocean Res. 153, 104240.

Yu, F., Koltun, V., 2015. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122 .

Yu, J., Ramamoorthi, R., 2020. Learning video stabilization using optical flow, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 8159–8167.

Yuan, Y., Weng, X., Ou, Y., Kitani, K.M., 2021. AgentFormer: Agent-aware transformers for socio-temporal multi-agent forecasting, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 9813–9823.

Zardoua, Y., Astito, A., Boulaala, M., Dokkali, Y., 2020. A short overview of horizon detection methods applied to maritime video stabilization, in: Proc. Int. Conf. Adv. Intell. Syst. Sustain. Dev. (AI2SD), pp. 857–864.

Zardoua, Y., Bergor, B.B., El Wahabi, A., El Mourabit, A., Chbeine, M., Astito, A., 2023. A horizon line annotation software for streamlining autonomous sea navigation experiments. Adv. Artif. Intell. Mach. Learn. 3, 1768–1786.

Zhang, B., Liu, J., Liang, J., Liu, R.W., 2026a. Low-visibility scene enhancement for promoting vessel trafic safety in intelligent maritime surveillance. Ocean Eng. 366, 127519.

Zhang, B., Liu, R.W., Liu, J., Chui, K.T., Gupta, B.B., 2025a. Multi-sensor data fusion meets edge computing for intelligent surface vehicles. IEEE Internet Things Mag. (IoTM) 8, 127–135.

Zhang, C., Bin, J., Wang, W., Peng, X., Wang, R., Halldearn, R., Liu, Z., 2020. AIS data driven general vessel destination prediction: A random forest based approach. Transp. Res. C Emerg. Technol. 118, 102729.

Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H., 2022a. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 .

Zhang, L., Du, X., Zhang, R., Zhang, J., 2023. A lightweight detection algorithm for unmanned surface vehicles based on multi-scale feature fusion. J. Mar. Sci. Eng. (JMSE) 11, 1392.

Zhang, L., Meng, Q., 2019. Probabilistic ship domain with applications to ship collision risk assessment. Ocean Eng. 186, 106130.

Zhang, L., Sun, X., Li, Z., Kong, D., Liu, J., Ni, P., 2024a. Boundary enhancement-driven accurate semantic segmentation networks for unmanned surface vessels in complex marine environments. IEEE Sens. J. (JSEN) 24, 24972–24987.

Zhang, L., Sun, X., Li, Z., Zhang, S., Liu, J., Dong, H., Kong, D., 2026b. Scene-adaptive semantic segmentation guided by multi-level boundary semantic-reinforcement for unmanned surface vessels. Measurement 257, 118655.

Zhang, M., Dong, L., Gao, Y., Wang, Y., 2024b. Infrared maritime object detection network with feature enhancement and adjacent fusion. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. (JSTARS) 17, 5750–5760.

Zhang, M., Dong, L., Huang, Z., Flierl, M., 2024c. TSPNet: Temporal-spatial pyramid network for infrared maritime object detection. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. (JSTARS) 17, 18058–18069.

Zhang, M., Montewka, J., Manderbacka, T., Kujala, P., Hirdaris, S., 2021a. A big data analytics method for the evaluation of ship–ship collision risk reflecting hydrometeorological conditions. Reliab. Eng. Syst. Saf. (RESS) 213, 107674.

Zhang, M.M., Choi, J., Daniilidis, K., Wolf, M.T., Kanan, C., 2015. VAIS: A dataset for recognizing maritime imagery in the visible and infrared spectrums, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), pp. 10–16.

Zhang, Q., Liu, X., Shao, M., Zhang, T., Sun, Y., Yuan, H., 2025b. USVRT-DETR: A transformer-based detector with multi-scale feature fusion and intra-scale interaction for USV surface scenarios. Ocean Eng. 342, 122926.

Zhang, Q., Shan, Y., Zhang, Z., Lin, H., Zhang, Y., Huang, K., 2024d. Multisensor fusion-based maritime ship object detection method for autonomous surface vehicles. J. Field Robot. (JFR) 41, 493–510.

Zhang, R., Su, Y., Li, Y., Zhang, L., Feng, J., 2022b. Infrared and visible image fusion methods for unmanned surface vessels with marine applications. J. Mar. Sci. Eng. (JMSE) 10, 588

Zhang, W., Guo, D., Shang, Y., Zhang, W., Hu, Z., 2025c. GLAF-DETR: Detection transformer with global-local adaptive fusion attention for infrared maritime object detection. IEEE Internet Things J. (IoTJ) 12, 44927–44940.

Zhang, W., Jiang, F., Yang, C., Wang, Z., Zhao, T., 2021b. Research on unmanned surface vehicles environment perception based on the fusion of vision and LiDAR. IEEE Access 9, 63107–63121.

Zhang, X., Demiris, Y., 2023. Visible and infrared image fusion using deep learning. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 45, 10535–10554.

Zhang, X., Fu, X., Xiao, Z., Xu, H., Qin, Z., 2022c. Vessel trajectory prediction in maritime transportation: Current approaches and beyond. IEEE Trans. Intell. Transp. Syst. (TITS) 23, 19980–19998.

Zhang, X., Wang, C., Jiang, L., An, L., Yang, R., 2021c. Collision-avoidance navigation systems for maritime autonomous surface ships: A state of the art survey. Ocean Eng. 235, 109380.

Zhao, C., Bao, M., Liang, J., Liang, M., Liu, R.W., Pang, G., 2025. Multi-sensor fusion-driven surface vessel identification and tracking using unmanned aerial vehicles for maritime surveillance. IEEE Trans. Consum. Electron. (TCE) 71, 6280–6293.

Zhao, C., Liang, M., Kang, J., Kim, Y., Liu, R.W., 2026a. Beyond-visual-range situational awareness for intelligent surface vehicles: A maritime IoT-based collaborative framework. IEEE Internet Things Mag. (IoTM) .

Zhao, C., Nie, J., Kang, J., Lu, Y., Liu, R.W., 2026b. Multi-modal graph neural network-inspired vessel identification and tracking for UAV-based maritime surveillance. IEEE Internet Things J. (IoTJ) .

Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J., 2017. Pyramid scene parsing network, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 2881–2890.

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J., 2024. DETRs beat YOLOs on real-time object detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 16965–16974.

Zheng, Y., Ao, X., Xie, W., Lv, X., Gao, Z., Liu, X., Fang, Z., 2026. IRS-DETR: Lightweight real-time transformer for infrared ship detection in maritime surveillance. Ocean Eng. 343, 123319.

Zhong, X., Lu, T., Huang, W., Ye, M., Jia, X., Lin, C., 2022. Grayscale enhancement colorization network for visible-infrared person re-identification. IEEE Trans. Circuits Syst. Video Technol. (TCSVT) 32, 1418–1430.

Zhong, X., Yan, X., Yang, Z., Huang, W., Jiang, K., Liu, R.W., Wang, Z., 2023. Visual exposes you: Pedestrian trajectory prediction meets visual intention. IEEE Trans. Intell. Transp. Syst. (TITS) 24, 9390–9400.

Zhou, G., Cheng, C., Chen, Y., 2024. Eficient multi-branch segmentation network for situation awareness in autonomous navigation. Ocean Eng. 302, 117741.

Zhou, K., Yang, Y., Cavallaro, A., Xiang, T., 2019a. Omni-scale feature learning for person re-identification, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 3702–3712.

Zhou, R., Gao, Y., Wu, P., Zhao, X., Dou, W., Sun, C., Zhong, Y., Wang, Y., 2022. Collision-free waterway segmentation for inland unmanned surface vehicles. IEEE Trans. Instrum. Meas. (TIM) 71, 5013416.

Zhou, Y., Daamen, W., Vellinga, T., Hoogendoorn, S., 2019b. Review of maritime trafic models from vessel behavior modeling perspective. Transp. Res. C Emerg. Technol. 105, 323–345.

Zhu, H., Deng, J., Zhang, Y., Ji, J., Mao, Q., Li, H., Zhang, Y., 2023. VPFNet: Improving 3D object detection with virtual point based LiDAR and stereo data fusion. IEEE Trans. Multimedia (TMM) 25, 5291–5304.

Zhu, J., Park, T., Isola, P., Efros, A.A., 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks, in: Proc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 2223–2232.

Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J., 2021. Deformable DETR: Deformable transformers for end-to-end object detection, in: Proc. Int. Conf. Learn. Represent. (ICLR).

Žust, L., Kristan, M., 2022. Temporal context for robust maritime obstacle detection, in: Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pp. 6340–6346.

Žust, L., Perš, J., Kristan, M., 2023. LaRS: A diverse panoptic maritime obstacle detection dataset and benchmark, in: Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 20304–20314.