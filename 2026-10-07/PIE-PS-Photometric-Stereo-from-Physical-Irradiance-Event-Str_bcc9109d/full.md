# PIE-PS: Photometric Stereo from Physical Irradiance Event Streams

XIANGZE MENG, Jilin University, China; Guangdong Institute of Intelligence Science and Technology, China GUANGYU LI, Guangdong Institute of Intelligence Science and Technology, China

JING LI, Jilin University, China

DI MEI, Beijing Institute of Technology, China

SONGCHEN MA, The Hong Kong University of Science and Technology, China

MINGKUN XU<sup>∗</sup>, Guangdong Institute of Intelligence Science and Technology, China

RUI MA<sup>∗</sup>, Jilin University, China

![](images/56c1afc882e5a96ebc263726f1e0bce1385904a120b3eaf3f176f084aa69f8d1.jpg)  
Fig. 1. Overview of PIE-PS. A prototype system captures raw event streams under moving light. Adjacent events at the same pixel are paired with their light directions to form Physical Irradiance Event (PIE) observations. Each PIE provides a threshold-free signed event-rate Physical Irradiance Event Feature (PIEF) and calibrated light-pair geometry. PIE-GNN encodes these PIE nodes, Reliability-Grading Atention (RGA) down-weights unreliable observations, and pixel aggregation produces dense surface normals.

Event cameras record as<sub>y</sub>nchronous lo<sub>g</sub>-ima<sub>g</sub>e-irradiance chan<sub>g</sub>es with mi-<sub>crosecon</sub>d l<sub>a</sub>t<sub>ency an</sub>d hi<sub>g</sub>h d<sub>ynam</sub>i<sub>c range.</sub> Th<sub>ese proper</sub>ti<sub>es are use</sub>f<sub>u</sub>l f<sub>or</sub> <sub>p</sub>hotometric stereo under movin<sub>g</sub> illumination<sub>,</sub> but raw events are s<sub>p</sub>arse <sub>an</sub>d d<sub>epen</sub>d <sub>on an un</sub>k<sub>nown con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> W<sub>e s</sub>t<sub>ar</sub>t f<sub>rom</sub> th<sub>e even</sub>t trigger mo<sup>d</sup>e<sup>l</sup> an<sup>d d</sup>erive a p<sup>h</sup>ysica<sup>l</sup> re<sup>l</sup>ation <sup>b</sup>etween a<sup>d</sup>jacent events, <sup>l</sup>ig<sup>h</sup>t motion<sub>,</sub> and surface normals. This relation <sub>g</sub>ives a direct <sub>p</sub>h<sub>y</sub>sics-onl<sub>y</sub> solver<sub>,</sub> b<sub>u</sub>t th<sub>e so</sub>l<sub>ver nee</sub>d<sub>s</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>, enoug</sub>h <sub>even</sub>t<sub>s a</sub>t <sub>eac</sub>h <sub>p</sub>i<sub>xe</sub>l<sub>, an</sub>d i<sub>n</sub>d<sub>epen</sub> d<sub>e</sub>nt <sub>pe</sub>r-<sub>p</sub>ix<sub>e</sub>l <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> T<sub>o a</sub>ddr<sub>ess</sub> th<sub>ese</sub> limit<sub>s, we</sub> intr<sub>o</sub>d<sub>uce</sub> PIE-PS<sub>,</sub> <sub>a</sub> l<sub>earn</sub>i<sub>ng-</sub>b<sub>ase</sub>d f<sub>ramewor</sub>k f<sub>or</sub> d<sub>ense sur</sub>f<sub>ace norma</sub>l <sub>recons</sub>t<sub>ruc</sub>ti<sub>on</sub> f<sub>rom</sub> <sub>raw even</sub>t <sub>s</sub>t<sub>reams an</sub>d k<sub>nown</sub> li hti<sub>n .</sub> W<sub>e</sub> f<sub>orm</sub> Ph <sub>s</sub>i<sub>ca</sub>l I<sub>rra</sub>di<sub>ance</sub> E<sub>ven</sub>t<sub>s</sub> (PIEs) b<sub>y p</sub>airin<sub>g</sub> two adjacent events at the same <sub>p</sub>ixel with their corre-<sub>spo</sub>ndin<sub>g</sub> li<sub>g</sub>ht dir<sub>ec</sub>ti<sub>o</sub>n<sub>s.</sub> E<sub>ac</sub>h PIE <sub>p</sub>r<sub>ov</sub>id<sub>es a</sub> Ph<sub>ys</sub>i<sub>ca</sub>l Irr<sub>a</sub>di<sub>a</sub>n<sub>ce</sub> E<sub>ve</sub>nt Feature (PIEF), defined as the si<sub>g</sub>ned event rate. PIEF does not re<sub>q</sub>uire the <sub>un</sub>k<sub>nown con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> T<sub>o s</sub>h<sub>are spa</sub>ti<sub>a</sub>l <sub>an</sub>d t<sub>empora</sub>l <sub>con</sub>t<sub>ex</sub>t <sub>across</sub> n<sub>ea</sub>rb<sub>y</sub> PIE<sub>s,</sub> <sub>we</sub> intr<sub>o</sub>d<sub>uce</sub> PIE-GNN<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h tr<sub>ea</sub>t<sub>s</sub> <sub>eac</sub>h PIE <sub>as</sub> <sub>a</sub> <sub>g</sub>r<sub>ap</sub>h n<sub>o</sub>d<sub>e</sub> and encodes it with its li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub>. Since the reliabilit<sub>y</sub> of PIE observations can var<sub>y</sub> with local a<sub>pp</sub>earance<sub>,</sub> illumination <sub>g</sub>eometr<sub>y,</sub> and sensor noise, Reliabilit<sub>y</sub>-Gradin<sub>g</sub> Attention (RGA) <sub>p</sub>redicts reliabilit<sub>y</sub> wei<sub>g</sub>hts to down-wei<sub>g</sub>ht unreliable PIEs. Pixel a<sub>gg</sub>re<sub>g</sub>ation then <sub>p</sub>roduces dense normals. Ex<sub>p</sub>eriments on s<sub>y</sub>nthetic and real data sho<sub>w</sub> that PIE-PS o<sub>u</sub>t<sub>p</sub>erforms <sub>pr</sub>i<sub>or even</sub>t<sub>-</sub>b<sub>ase</sub>d <sub>p</sub>h<sub>o</sub>t<sub>ome</sub>t<sub>r</sub>i<sub>c s</sub>t<sub>ereo me</sub>th<sub>o</sub>d<sub>s an</sub>d th<sub>e</sub> di<sub>rec</sub>t <sub>so</sub>l<sub>ver</sub> b<sub>ase</sub>li<sub>ne.</sub> Project page: <sup>h</sup>ttps://pie-ps.git<sup>h</sup>u<sup>b</sup>.io.

CCS Concepts: • Computing methodologies → Reconstruction.

Additi<sub>o</sub>n<sub>a</sub>l K<sub>ey</sub> W<sub>o</sub>rd<sub>s a</sub>nd Phr<sub>ases</sub>: Ph<sub>o</sub>t<sub>o</sub>m<sub>e</sub>tri<sub>c</sub> St<sub>e</sub>r<sub>eo,</sub> E<sub>ve</sub>nt C<sub>a</sub>m<sub>e</sub>r<sub>a,</sub> Ph<sub>ys</sub>- i<sub>ca</sub>l Irr<sub>a</sub>di<sub>a</sub>n<sub>ce</sub> E<sub>ve</sub>nt<sub>s</sub>

## ACM Reference Format:

Xian<sub>g</sub>ze Men<sub>g</sub>, Guan<sub>gy</sub>u Li, Jin<sub>g</sub> Li, Di Mei, Son<sub>g</sub>chen Ma, Min<sub>g</sub>kun Xu, <sub>an</sub>d R<sub>u</sub>i M<sub>a.</sub> 2026<sub>.</sub> PIE<sub>-</sub>PS<sub>:</sub> Ph<sub>o</sub>t<sub>ome</sub>t<sub>r</sub>i<sub>c</sub> St<sub>ereo</sub> f<sub>rom</sub> Ph <sub>s</sub>i<sub>ca</sub>l I<sub>rra</sub>di<sub>ance</sub> E<sub>ven</sub>t Streams. In SIGGRAPH Asia 2026 Conference Papers (SA Conference Papers ’26), December 01–04, 2026, Kuala Lumpur, Malaysia. ACM, New York, NY, USA<sub>,</sub> 9 <sub>p</sub>a<sub>g</sub>es. htt<sub>p</sub>s://doi.or<sub>g</sub>/10.1145/3829340.3842249

## 1 Introduction

Recoverin<sub>g</sub> detailed surface <sub>g</sub>eometr<sub>y</sub> is im<sub>p</sub>ortant for content creation, ins<sub>p</sub>ection, and au<sub>g</sub>mented realit<sub>y</sub> [Bae and Davison 2024;

Qi et al. 2018]. Photometric Stereo (PS) [Woodham 1980] estimates d<sub>ense sur</sub>f<sub>ace norma</sub>l<sub>s</sub> f<sub>rom s</sub>h<sub>a</sub>di<sub>ng c</sub>h<sub>anges un</sub>d<sub>er vary</sub>i<sub>ng</sub> li<sub>g</sub>ht<sub>.</sub> It <sub>can recover</sub> fi<sub>ne</sub> d<sub>e</sub>t<sub>a</sub>il<sub>,</sub> b<sub>u</sub>t <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d f<sub>rame-</sub>b<sub>ase</sub>d PS <sub>recor</sub>d<sub>s one</sub> ima<sub>g</sub>e for each li<sub>g</sub>ht state. This limits frame rate and d<sub>y</sub>namic ran<sub>g</sub>e<sub>,</sub> es<sub>p</sub>eciall<sub>y</sub> under fast li<sub>g</sub>htin<sub>g</sub> or lar<sub>g</sub>e intensit<sub>y</sub> variation.

Event cameras <sub>p</sub>rovide a diferent sensin<sub>g</sub> model. The<sub>y</sub> record as<sub>y</sub>nchronous lo<sub>g</sub>-ima<sub>g</sub>e-irradiance chan<sub>g</sub>es with microsecond la tenc<sub>y</sub> and hi<sub>g</sub>h d<sub>y</sub>namic ran<sub>g</sub>e [Galle<sub>g</sub>o et al. 2020; Rebec<sub>q</sub> et al. 2019]. Under movin<sub>g</sub> li<sub>g</sub>ht, these events contain normal cues. These <sub>proper</sub>ti<sub>es</sub> <sub>ma</sub>k<sub>e</sub> <sub>even</sub>t<sub>-</sub>b<sub>ase</sub>d PS <sub>more</sub> th<sub>an</sub> <sub>a</sub> di<sub>rec</sub>t <sub>reuse</sub> <sub>o</sub>f f<sub>rame-</sub> b<sub>ase</sub>d PS<sub>.</sub>

P<sub>r</sub>i<sub>or ana</sub>l<sub>y</sub>ti<sub>ca</sub>l <sub>me</sub>th<sub>o</sub>d<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>even</sub>t<sub>-</sub>b<sub>ase</sub>d PS i<sub>s poss</sub>ibl<sub>e.</sub> EventPS [Yu et al. 2024] formulates event-based <sub>p</sub>hotometric stereo as <sub>p</sub>er-<sub>p</sub>ixel reconstruction from chan<sub>g</sub>es in lo<sub>g</sub> ima<sub>g</sub>e irradiance. T<sup>h</sup>is connects event measurements wit<sup>h</sup> t<sup>h</sup>e <sup>l</sup>ig<sup>h</sup>ting trajectory, <sup>b</sup>ut it de<sub>p</sub>ends on a fixed event contrast threshold. PS-EIP [Kitazawa et al. 2025] im<sub>p</sub>roves robustness b<sub>y</sub> usin<sub>g</sub> an interval-<sub>p</sub>rofile a<sub>pp</sub>roximation over three recorded events. However<sub>,</sub> this desi<sub>g</sub>n re<sub>q</sub>uires <sub>a</sub> l<sub>oo</sub>k<sub>-a</sub>h<sub>ea</sub>d <sub>even</sub>t <sub>an</sub>d d<sub>oes no</sub>t <sub>use even</sub>t <sub>po</sub>l<sub>ar</sub>it<sub>y.</sub> Th<sub>e</sub>i<sub>r p</sub>h<sub>ys</sub>i<sub>ca</sub>l f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons are s</sub>till b<sub>u</sub>ilt <sub>on</sub> l<sub>oca</sub>l <sub>per-p</sub>i<sub>xe</sub>l <sub>even</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> Th<sub>us,</sub> th<sub>e recons</sub>t<sub>ruc</sub>ti<sub>on can</sub> h<sub>ave</sub> li<sub>m</sub>it<sub>e</sub>d <sub>spa</sub>ti<sub>a</sub>l <sub>con</sub>t<sub>ex</sub>t <sub>an</sub>d <sub>can</sub> b<sub>e sen-</sub> <sub>s</sub>iti<sub>ve</sub> t<sub>o even</sub>t d<sub>ens</sub>it<sub>y,</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>ca</sub>lib<sub>ra</sub>ti<sub>on, an</sub>d l<sub>oca</sub>ll<sub>y unre</sub>li<sub>a</sub>bl<sub>e</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub>

We start from the same event tri<sub>gg</sub>er model under movin<sub>g</sub> li<sub>g</sub>ht<sub>,</sub> b<sub>u</sub>t <sub>u</sub>se it in a diferent wa<sub>y</sub>. Prior anal<sub>y</sub>tical methods mainl<sub>y</sub> t<sub>u</sub>rn th<sub>e even</sub>t <sub>s</sub>t<sub>ream</sub> i<sub>n</sub>t<sub>o a per-p</sub>i<sub>xe</sub>l dif<sub>eren</sub>ti<sub>a</sub>l <sub>or</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>-pro</sub>fil<sub>e</sub> <sub>equa</sub>ti<sub>o</sub>n<sub>.</sub> W<sub>e</sub> in<sub>s</sub>t<sub>ea</sub>d d<sub>e</sub>ri<sub>ve a pa</sub>ir<sub>w</sub>i<sub>se p</sub>h<sub>o</sub>t<sub>o</sub>m<sub>e</sub>tri<sub>c co</sub>n<sub>s</sub>tr<sub>a</sub>int fr<sub>o</sub>m two a<sup>d</sup>jacent events an<sup>d</sup> t<sup>h</sup>e <sup>l</sup>ig<sup>h</sup>t <sup>d</sup>irections at t<sup>h</sup>eir timestamps. This <sub>g</sub>ives a direct relation between the event <sub>p</sub>air<sub>,</sub> the li<sub>g</sub>ht-<sub>p</sub>air <sub>c</sub>h<sub>ange, an</sub>d th<sub>e sur</sub>f<sub>ace norma</sub>l<sub>.</sub> If th<sub>e con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> k<sub>nown</sub> <sub>an</sub>d <sub>enoug</sub>h <sub>even</sub>t <sub>pa</sub>i<sub>rs</sub> <sub>are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>a</sub>t <sub>a</sub> <sub>p</sub>i<sub>xe</sub>l<sub>,</sub> th<sub>ese</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> can be solved b<sub>y</sub> direct numerical o<sub>p</sub>timization. We kee<sub>p</sub> this solver <sub>as</sub> <sub>a</sub> <sub>p</sub>h<sub>ys</sub>i<sub>cs-on</sub>l<sub>y</sub> b<sub>ase</sub>li<sub>ne</sub> b<sub>ecause</sub> it <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l <sub>exp</sub>li<sub>c</sub>it<sub>.</sub> It <sub>a</sub>l<sub>so</sub> <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> li<sub>m</sub>it<sub>s</sub> <sub>o</sub>f <sub>a</sub> <sub>pure</sub> <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>l <sub>rou</sub>t<sub>e.</sub> Th<sub>e</sub> <sub>so</sub>l<sub>ver</sub> <sub>nee</sub>d<sub>s</sub> th<sub>e un</sub>k<sub>nown</sub> th<sub>res</sub>h<sub>o</sub>ld �<sub>, requ</sub>i<sub>res enoug</sub>h <sub>even</sub>t<sub>s</sub> t<sub>o</sub> b<sub>u</sub>ild <sub>s</sub>t<sub>a</sub>bl<sub>e</sub> <sub>equa</sub>ti<sub>ons,</sub> <sub>an</sub>d t<sub>rea</sub>t<sub>s</sub> <sub>eac</sub>h <sub>p</sub>i<sub>xe</sub>l i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y.</sub>

Fi<sub>g.</sub> 1 <sub>g</sub>i<sub>ves a</sub>n <sub>ove</sub>r<sub>v</sub>i<sub>ew o</sub>f th<sub>e</sub> PIE-PS r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd r<sub>eco</sub>nstruction <sub>p</sub>i<sub>p</sub>eline. To address these limits<sub>,</sub> we introduce a learnin<sub>g</sub>- based framework built on Ph<sub>y</sub>sical Irradiance Events (PIEs). A PIE is <sub>no</sub>t <sub>a new sensor even</sub>t<sub>.</sub> It i<sub>s a s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> b<sub>u</sub>ilt f<sub>rom</sub> t<sub>wo</sub> a<sup>d</sup>jacent events at t<sup>h</sup>e same pixe<sup>l</sup> an<sup>d</sup> t<sup>h</sup>eir correspon<sup>d</sup>ing <sup>l</sup>ig<sup>h</sup>t <sup>d</sup>irections. Its Ph<sub>y</sub>sical Irradiance Event Feature (PIEF) is a scalar si<sub>g</sub>ned event rate<sub>, g</sub>iven b<sub>y</sub> event <sub>p</sub>olarit<sub>y</sub> divided b<sub>y</sub> inter-event time. The li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> remains <sub>p</sub>art ofthe PIE observation and is added when buildin<sub>g</sub> the <sub>g</sub>ra<sub>p</sub>h node. We then treat each PIE as a <sub>g</sub>ra<sub>p</sub>h n<sub>o</sub>d<sub>e a</sub>nd <sub>use</sub> PIE-GNN t<sub>o e</sub>x<sub>c</sub>h<sub>a</sub>n<sub>ge</sub> inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n b<sub>e</sub>t<sub>wee</sub>n n<sub>ea</sub>rb<sub>y</sub> event-<sub>p</sub>air observations in s<sub>p</sub>ace and time. This <sub>g</sub>ra<sub>p</sub>h form shares context across nearb<sub>y p</sub>ixels and times. In <sub>p</sub>ractice<sub>,</sub> the reliabilit<sub>y</sub> of PIE observations can var<sub>y</sub> with local a<sub>pp</sub>earance<sub>,</sub> illumination <sub>geome</sub>t<sub>ry, an</sub>d <sub>sensor no</sub>i<sub>se, w</sub>hil<sub>e</sub> l<sub>ess re</sub>li<sub>a</sub>bl<sub>e o</sub>b<sub>serva</sub>ti<sub>ons may</sub> still retain useful cues for normal estimation. Reliabilit<sub>y</sub>-Gradin<sub>g</sub> Attention (RGA) therefore assi<sub>g</sub>ns each PIE a soft reliabilit<sub>y</sub> <sub>g</sub>rade before <sub>p</sub>ixel a<sub>gg</sub>re<sub>g</sub>ation. The wei<sub>g</sub>hted PIE embeddin<sub>g</sub>s are then a<sub>gg</sub>re<sub>g</sub>ated into <sub>p</sub>ixel features<sub>,</sub> and a li<sub>g</sub>htwei<sub>g</sub>ht <sub>p</sub>ixel a<sub>gg</sub>re<sub>g</sub>ator <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> d<sub>ense</sub> <sub>sur</sub>f<sub>ace</sub> <sub>norma</sub>l<sub>s.</sub>

O<sub>ur me</sub>th<sub>o</sub>d <sub>separa</sub>t<sub>es</sub> th<sub>e ro</sub>l<sub>es o</sub>f <sub>p</sub>h<sub>ys</sub>i<sub>cs an</sub>d l<sub>earn</sub>i<sub>ng.</sub> Th<sub>e</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>mo</sub>d<sub>e</sub>l t<sub>e</sub>ll<sub>s</sub> <sub>us</sub> <sub>w</sub>h<sub>a</sub>t <sub>an</sub> <sub>even</sub>t <sub>pa</sub>i<sub>r</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>measure</sub> <sub>an</sub>d <sub>w</sub>hi<sub>c</sub>h li<sub>g</sub>ht<sub>-pa</sub>i<sub>r</sub> <sub>cues</sub> <sub>are</sub> <sub>use</sub>f<sub>u</sub>l<sub>.</sub> Th<sub>e</sub> <sub>grap</sub>h <sub>ne</sub>t<sub>wor</sub>k h<sub>an</sub>dl<sub>es</sub> th<sub>e</sub> <sub>par</sub>t<sub>s</sub> th<sub>a</sub>t <sub>are</sub> h<sub>ar</sub>d t<sub>o</sub> <sub>so</sub>l<sub>ve</sub> <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>un</sub>k<sub>nown</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>uneven</sub> event densit<sub>y,</sub> var<sub>y</sub>in<sub>g</sub> PIE reliabilit<sub>y,</sub> and s<sub>p</sub>atial context. As a result<sub>,</sub> PIE<sub>-</sub>PS <sub>uses</sub> th<sub>e</sub> <sub>even</sub>t f<sub>orma</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l <sub>w</sub>ith<sub>ou</sub>t <sub>ma</sub>ki<sub>ng</sub> th<sub>e</sub> fi<sub>xe</sub>d<sub>-</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>l <sub>so</sub>l<sub>ver</sub> th<sub>e ma</sub>i<sub>n recons</sub>t<sub>ruc</sub>ti<sub>on me</sub>th<sub>o</sub>d<sub>.</sub>

O<sub>ur</sub> <sub>ma</sub>i<sub>n</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> <sub>are</sub> <sub>summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We formulate <sub>p</sub>airwise event <sub>p</sub>hotometr<sub>y</sub> from adjacent event t<sub>r</sub>i<sub>ggers</sub> <sub>an</sub>d <sub>use</sub> it t<sub>o</sub> b<sub>u</sub>ild <sub>a</sub> di<sub>rec</sub>t <sub>numer</sub>i<sub>ca</sub>l <sub>so</sub>l<sub>ver</sub> b<sub>ase</sub>li<sub>ne.</sub>

• We introduce PIE-PS, a framework for event-based <sub>p</sub>hotometric <sub>s</sub>t<sub>ereo</sub> f<sub>rom</sub> <sub>raw</sub> <sub>even</sub>t <sub>s</sub>t<sub>reams</sub> <sub>an</sub>d k<sub>nown</sub> li<sub>g</sub>hti<sub>ng.</sub>

• We develo<sub>p</sub> PIEF, PIE-GNN, and RGA as the main learned method f<sub>or</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>-</sub>f<sub>ree</sub> <sub>s</sub>i<sub>gne</sub>d<sub>-ra</sub>t<sub>e</sub> <sub>enco</sub>di<sub>ng,</sub> li<sub>g</sub>ht<sub>-pa</sub>i<sub>r</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l f<sub>ea-</sub> ture extraction<sub>,</sub> and unreliable-event su<sub>pp</sub>ression.

## 2 Related Work

## 2.1 Photometric Stereo

Photometric Stereo (PS) was <sub>p</sub>ioneered b<sub>y</sub> Woodham [Woodham 1980] to recover surface normals from var<sub>y</sub>in<sub>g</sub> illumination. To h<sub>an</sub>dl<sub>e</sub> <sub>s</sub>h<sub>a</sub>d<sub>ows,</sub> <sub>specu</sub>l<sub>ar</sub>iti<sub>es,</sub> <sub>an</sub>d <sub>o</sub>th<sub>er</sub> <sub>non-</sub>L<sub>am</sub>b<sub>er</sub>ti<sub>an</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s,</sub> later methods use robust o<sub>p</sub>timization<sub>,</sub> s<sub>p</sub>arse re<sub>g</sub>ression<sub>,</sub> low-rank modelin<sub>g</sub>, and benchmarked evaluation on DiLiGenT [Chandraker <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2007<sub>;</sub> Ik<sub>e</sub>h<sub>a</sub>t<sub>a e</sub>t <sub>a</sub>l<sub>.</sub> 2012<sub>;</sub> Mi<sub>yaza</sub>ki <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2010<sub>;</sub> Shi <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2016<sub>;</sub> Wu et al. 2010]. Learnin<sub>g</sub>-based methods further ma<sub>p</sub> calibrated or uncalibrated ima<sub>g</sub>e observations to normals [Chen et al. 2019; G<sub>e e</sub>t <sub>a</sub>l<sub>.</sub> 2025<sub>;</sub> Li <sub>a</sub>nd Li 2022<sub>;</sub> Li <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2024<sub>;</sub> Li<sub>c</sub>h<sub>y e</sub>t <sub>a</sub>l<sub>.</sub> 2022<sub>;</sub> Santo et al. 2017, 2020; Zhen<sub>g</sub> et al. 2019, 2020]. Related reflectanceac<sub>q</sub>uisition methods recover sha<sub>p</sub>e or normals to<sub>g</sub>ether with <sub>p</sub>olarimetric SVBRDFs [Baek et al. 2018; Hwan<sub>g</sub> et al. 2022]. These <sub>wor</sub>k<sub>s</sub> t<sub>arge</sub>t <sub>r</sub>i<sub>c</sub>h<sub>er ma</sub>t<sub>er</sub>i<sub>a</sub>l <sub>cap</sub>t<sub>ure, w</sub>hil<sub>e our wor</sub>k f<sub>ocuses on</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>even</sub>t<sub>-</sub>b<sub>ase</sub>d d<sub>ense</sub> <sub>norma</sub>l <sub>recovery</sub> <sub>un</sub>d<sub>er</sub> <sub>mov</sub>i<sub>ng</sub> ill<sub>u-</sub> <sub>m</sub>i<sub>na</sub>ti<sub>on.</sub> F<sub>rame-</sub>b<sub>ase</sub>d PS <sub>me</sub>th<sub>o</sub>d<sub>s rema</sub>i<sub>n</sub> li<sub>m</sub>it<sub>e</sub>d b <sub>sensor</sub> f<sub>rame</sub> rate<sub>,</sub> motion blur<sub>,</sub> and d<sub>y</sub>namic ran<sub>g</sub>e in hi<sub>g</sub>h-s<sub>p</sub>eed li<sub>g</sub>htin<sub>g</sub> scenarios [Lo<sub>g</sub>othetis et al. 2021].

## 2.2 Event-based 3D Vision

Event cameras ofer hi<sub>g</sub>h d<sub>y</sub>namic ran<sub>g</sub>e and microsecond tem<sub>p</sub>oral resolution b<sub>y</sub> as<sub>y</sub>nchronousl<sub>y</sub> recordin<sub>g</sub> intensit<sub>y</sub> chan<sub>g</sub>es. Recent surve<sub>y</sub>s summarize event-based 3D reconstruction under hi<sub>g</sub>h-s<sub>p</sub>eed motion, low li<sub>g</sub>ht, and hi<sub>g</sub>h d<sub>y</sub>namic ran<sub>g</sub>e scenes [Xu et al. 2025]. E<sub>ar</sub>l<sub>y</sub> <sub>even</sub>t<sub>-</sub>b<sub>ase</sub>d 3D <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>ocus</sub> <sub>on</sub> <sub>sparse</sub> d<sub>ep</sub>th <sub>es</sub>ti<sub>ma</sub>ti<sub>on,</sub> multi-view stereo, structured li<sub>g</sub>ht, and visual odometr<sub>y</sub> [Galle<sub>g</sub>o et al. 2020; Mu<sub>g</sub>likar et al. 2021; Niu et al. 2025; Rebec<sub>q</sub> et al. 2018a]. Other a<sub>pp</sub>roaches use events for visual hull reconstruction [Wan<sub>g</sub> et al. 2022] or combine them with im<sub>p</sub>licit re<sub>p</sub>resentations, such as EventNeRF [Fen<sub>g</sub> et al. 2025b; Rudnev et al. 2023], event-based NeRF variants [Low and Lee 2023; Ma et al. 2023], and Event-3DGS [Fen<sub>g</sub> et al. 2025a; Han et al. 2024]. These methods mainl<sub>y</sub> tar<sub>g</sub>et de<sub>p</sub>th, trajectory trac<sup>k</sup>ing, nove<sup>l</sup> view synt<sup>h</sup>esis, or 3D reconstruction, <sub>ra</sub>th<sub>er</sub> th<sub>an con</sub>t<sub>ro</sub>ll<sub>e</sub>d<sub>-</sub>li<sub>g</sub>ht <sub>p</sub>i<sub>xe</sub>l<sub>-w</sub>i<sub>se norma</sub>l <sub>es</sub>ti<sub>ma</sub>ti<sub>on.</sub>

## 2.3 Event-based Photometric Stereo

R<sub>ecen</sub>t <sub>researc</sub>h h<sub>as</sub> <sub>use</sub>d <sub>even</sub>t <sub>s</sub>t<sub>reams</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>-spee</sub>d <sub>norma</sub>l <sub>es</sub>ti<sub>-</sub> mation. H<sub>y</sub>brid methods, such as EFPS-Net [R<sub>y</sub>oo et al. 2023], fuse

RGB f<sub>rames w</sub>ith <sub>even</sub>t d<sub>a</sub>t<sub>a</sub> t<sub>o en</sub>h<sub>ance</sub> b<sub>oun</sub>d<sub>ary an</sub>d <sub>specu</sub>l<sub>ar</sub> details. EventPS [Yu et al. 2024] is a calibrated <sub>p</sub>ure event-based <sub>me</sub>th<sub>o</sub>d th<sub>a</sub>t f<sub>ormu</sub>l<sub>a</sub>t<sub>es a per-p</sub>i<sub>xe</sub>l dif<sub>eren</sub>ti<sub>a</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>on</sub> l<sub>og</sub> image irra<sup>d</sup>iance <sup>f</sup>rom t<sup>h</sup>e <sup>l</sup>ig<sup>h</sup>t trajectory, <sup>b</sup>ut it <sup>d</sup>epen<sup>d</sup>s on t<sup>h</sup>e event contrast threshold and local event densit<sub>y</sub>. PS-EIP [Kitazawa et al. 2025] im<sub>p</sub>roves robustness with an Event Interval Profile built f<sub>rom</sub> th<sub>ree</sub> <sub>even</sub>t<sub>s,</sub> <sub>a</sub>lth<sub>oug</sub>h it <sub>requ</sub>i<sub>res</sub> <sub>a</sub> l<sub>oo</sub>k<sub>-a</sub>h<sub>ea</sub>d <sub>even</sub>t <sub>an</sub>d d<sub>oes</sub> not use event <sub>p</sub>olarit<sub>y</sub>. EventUPS [Lian<sub>g</sub> et al. 2025] extends eventbased PS to the uncalibrated illumination settin<sub>g</sub>. In contrast<sub>,</sub> we use a<sup>d</sup>jacent-event pairwise p<sup>h</sup>otometry to <sup>b</sup>ui<sup>ld</sup> PIEs t<sup>h</sup>at preserve event timin<sub>g,</sub> <sub>p</sub>olarit<sub>y,</sub> and li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> without re<sub>q</sub>uirin<sub>g</sub> th<sub>e con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld <sub>a</sub>t t<sub>es</sub>t ti<sub>me.</sub> PIE<sub>-</sub>GNN th<sub>en s</sub>h<sub>ares con</sub>t<sub>ex</sub>t <sub>ac</sub>r<sub>oss</sub> n<sub>ea</sub>rb<sub>y</sub> PIE n<sub>o</sub>d<sub>es, a</sub>nd RGA <sub>we</sub>i<sub>g</sub>ht<sub>s u</sub>nr<sub>e</sub>li<sub>a</sub>bl<sub>e</sub> PIE<sub>s</sub> b<sub>e</sub>f<sub>o</sub>r<sub>e</sub> <sub>p</sub>ixel a<sub>gg</sub>re<sub>g</sub>ation <sub>p</sub>roduces dense normals.

## 3 Methodology

We <sub>p</sub>ro<sub>p</sub>ose PIE-PS for s<sub>u</sub>rface normal reco<sub>v</sub>er<sub>y</sub> from e<sub>v</sub>ent streams. Fi<sub>g</sub>. 2 shows the <sub>p</sub>i<sub>p</sub>eline. We first derive event-<sub>p</sub>air <sub>p</sub>hotometr<sub>y</sub> and build Ph<sub>y</sub>sical Irradiance Events (PIEs) from adjacent events <sub>a</sub>nd th<sub>e</sub>ir li<sub>g</sub>ht dir<sub>ec</sub>ti<sub>o</sub>n<sub>s</sub> in S<sub>ec.</sub> 3<sub>.</sub>1<sub>.</sub> E<sub>ac</sub>h PIE <sub>p</sub>r<sub>ov</sub>id<sub>es</sub> <sub>a</sub> Ph<sub>ys</sub>i<sub>ca</sub>l Irradiance Event Feature (PIEF), defined as the threshold-free si<sub>g</sub>ned <sub>eve</sub>nt r<sub>a</sub>t<sub>e</sub> <sub>o</sub>f th<sub>e</sub> <sub>eve</sub>nt <sub>pa</sub>ir in S<sub>ec.</sub> 3<sub>.</sub>2<sub>.</sub> PIE-GNN th<sub>e</sub>n <sub>uses</sub> th<sub>e</sub> PIEF to<sub>g</sub>ether with the corres<sub>p</sub>ondin<sub>g</sub> li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> to extract n<sub>o</sub>d<sub>e</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>es</sub> in S<sub>ec.</sub> 3<sub>.</sub>2<sub>.</sub> RGA <sub>we</sub>i<sub>g</sub>ht<sub>s u</sub>nr<sub>e</sub>li<sub>a</sub>bl<sub>e</sub> PIE<sub>s, a</sub>nd <sub>p</sub>ix<sub>e</sub>l a<sub>gg</sub>re<sub>g</sub>ation <sub>p</sub>redicts dense normals as described in Sec. 3.3.

## 3.1 Physical Modeling of Event Photometry

We consider a pixel location u on a static surface. Its unit surface <sub>norma</sub>l i<sub>s</sub> $\mathbf { n } _ { \mathbf { u } } \in \mathbb { S } ^ { 2 }$ <sub>.</sub> Th<sub>e</sub> dif<sub>use a</sub>lb<sub>e</sub>d<sub>o</sub> i<sub>s</sub> $\rho \in \mathbb { R } ^ { + }$ <sub>.</sub> U<sub>n</sub>d<sub>er a</sub> di<sub>s</sub>t<sub>an</sub>t di<sub>rec</sub>ti<sub>ona</sub>l li<sub>g</sub>ht $\mathbf { l } ( t ) \in \mathbb { S } ^ { 2 }$ <sub>w</sub>ith int<sub>e</sub>n<sub>s</sub>it<sub>y</sub> <sub>�,</sub> th<sub>e</sub> im<sub>age</sub> irr<sub>a</sub>di<sub>a</sub>n<sub>ce</sub> <sub>a</sub>t pixel u is

$$
I ( t ) = s \rho \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } ( t ) ,\tag{1}
$$

f<sub>or</sub> lit <sub>reg</sub>i<sub>ons</sub> <sub>w</sub>h<sub>ere</sub> $\mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } ( t ) > 0 .$

An event camera measures chan<sub>g</sub>es in lo<sub>g</sub> ima<sub>g</sub>e irradiance. We write the lo<sub>g</sub> ima<sub>g</sub>e irradiance as

$$
\mathcal { L } ( t ) = \ln ( q I ( t ) ) ,\tag{2}
$$

<sub>w</sub>h<sub>ere</sub> $q$ i<sub>s a se</sub>n<sub>so</sub>r <sub>ga</sub>in<sub>.</sub> U<sub>s</sub>in<sub>g</sub> E<sub>q.</sub> 1<sub>, we</sub> h<sub>ave</sub>

$$
\begin{array} { r } { \mathcal { L } ( t ) = \ln q + \ln s + \ln \rho + \ln \left( \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } ( t ) \right) . } \end{array}\tag{3}
$$

Th<sub>e</sub> fi<sub>rs</sub>t th<sub>ree</sub> t<sub>erms</sub> <sub>are</sub> <sub>cons</sub>t<sub>an</sub>t f<sub>or</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>p</sub>i<sub>xe</sub>l<sub>.</sub> T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> ti<sub>me</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve g</sub>i<sub>ves</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>va</sub>l<sub>ue</sub>

$$
\dot { \mathcal { L } } ( t ) \triangleq \frac { \partial \mathcal { L } ( t ) } { \partial t } = \frac {  { \mathbf { n } } _ { \mathbf { u } } ^ { \top } \dot { \mathbf { l } } ( t ) } {  { \mathbf { n } } _ { \mathbf { u } } ^ { \top }  { \mathbf { l } } ( t ) } .\tag{4}
$$

Thi<sub>s</sub> <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>norma</sub>l <sub>an</sub>d th<sub>e</sub> li<sub>g</sub>ht <sub>mo</sub>ti<sub>on.</sub>

The event camera does not measure E<sub>q</sub>. 4 directl<sub>y</sub>. It re<sub>p</sub>orts an <sub>even</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e accumu</sub>l<sub>a</sub>t<sub>e</sub>d l<sub>og-</sub>i<sub>mage-</sub>i<sub>rra</sub>di<sub>ance c</sub>h<sub>ange reac</sub>h<sub>es</sub> a threshold. At pixel u, let two adjacent events be $e _ { k }$ <sub>an</sub>d $e _ { k + 1 }$ <sub>.</sub> Th<sub>e</sub>i<sub>r</sub> t<sup>i</sup>mestam<sub>p</sub>s are $t _ { k }$ <sub>an</sub>d $t _ { k + 1 }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>po</sub>l<sub>ar</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>secon</sub>d <sub>even</sub>t i<sub>s</sub> $p _ { k + 1 } \in \{ + 1 , - 1 \}$ . The tri<sub>gg</sub>er model <sub>g</sub>ives

$$
\begin{array} { r } { \mathcal { L } ( t _ { k + 1 } ) - \mathcal { L } ( t _ { k } ) \approx p _ { k + 1 } C , } \end{array}\tag{5}
$$

<sub>w</sub>h<sub>ere</sub> � i<sub>s</sub> th<sub>e con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> With $\Delta t _ { k } = t _ { k + 1 } - t _ { k }$ <sub>,</sub> th<sub>e even</sub>t <sub>g</sub>i<sub>ves</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e-</sub>dif<sub>erence</sub> <sub>measuremen</sub>t<sub>:</sub>

$$
\dot { \mathcal { L } } ( t _ { k + 1 } ) \approx \frac { \mathcal { L } ( t _ { k + 1 } ) - \mathcal { L } ( t _ { k } ) } { \Delta t _ { k } } \approx \frac { \dot { p } _ { k + 1 } C } { \Delta t _ { k } } .\tag{6}
$$

E<sub>q</sub>. 4 <sub>g</sub>ives the theoretical value<sub>,</sub> while E<sub>q</sub>. 6 <sub>g</sub>ives the eventcamera measurement. Matchin<sub>g</sub> them <sub>g</sub>ives the followin<sub>g p</sub>ixel-wise o<sup>b</sup>jective:

$$
\operatorname* { m i n } _ { \mathbf { n } _ { \mathbf { u } } \in \mathbb { S } ^ { 2 } } \sum _ { k } \left( \frac { p _ { k + 1 } C } { \Delta t _ { k } } - \frac { \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { \dot { l } } ( t _ { k + 1 } ) } { \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } ( t _ { k + 1 } ) } \right) ^ { 2 } .\tag{7}
$$

T<sup>h</sup>is o<sup>b</sup>jective can <sup>b</sup>e so<sup>l</sup>ve<sup>d</sup> <sup>d</sup>irect<sup>l</sup>y <sup>b</sup>y numerica<sup>l</sup> optimization to <sub>recover</sub> th<sub>e</sub> <sub>norma</sub>l $\mathbf { n } _ { \mathbf { u } } .$ . However, using t<sup>h</sup>is o<sup>b</sup>jective as a reconstructi<sub>on</sub> <sub>me</sub>th<sub>o</sub>d l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub> <sub>prac</sub>ti<sub>ca</sub>l li<sub>m</sub>it<sub>a</sub>ti<sub>ons.</sub> It <sub>requ</sub>i<sub>res</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>even</sub>t <sub>con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld �<sub>.</sub> It <sub>a</sub>l<sub>so nee</sub>d<sub>s enoug</sub>h <sub>even</sub>t<sub>-pa</sub>i<sub>r o</sub>b<sub>serva</sub>ti<sub>ons</sub> <sub>a</sub>t <sub>eac</sub>h <sub>p</sub>i<sub>xe</sub>l t<sub>o ma</sub>k<sub>e</sub> th<sub>e norma</sub>l <sub>we</sub>ll <sub>cons</sub>t<sub>ra</sub>i<sub>ne</sub>d<sub>.</sub> Th<sub>ese</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons</sub> motivate our Ph<sub>y</sub>sical Irradiance Event (PIE) re<sub>p</sub>resentation. A PIE is <sub>no</sub>t <sub>a new sensor even</sub>t<sub>.</sub> It i<sub>s a pa</sub>i<sub>re</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> b<sub>u</sub>ilt f<sub>rom</sub> t<sub>wo a</sub>d<sub>-</sub> jacent events at t<sup>h</sup>e same pixe<sup>l</sup>. Let t<sup>h</sup>e raw events <sup>b</sup>e $e _ { k } = ( { \mathbf { u } } , t _ { k } , p _ { k } )$ <sub>an</sub>d $e _ { k + 1 } = ( { \mathbf { u } } , t _ { k + 1 } , { p _ { k + 1 } } )$ <sub>.</sub> W<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>use</sub> $\Delta t _ { k } = t _ { k + 1 } - t _ { k } , \mathbf { l } _ { k } = \mathbf { l } ( t _ { k } )$ <sub>,</sub> <sub>an</sub>d $\mathbf { l } _ { k + 1 } = \mathbf { l } ( t _ { k + 1 } )$ <sub>.</sub> T<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>no</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e su</sub>b<sub>sequen</sub>t d<sub>er</sub>i<sub>va</sub>ti<sub>on,</sub> <sub>we</sub> r<sub>eco</sub>rd th<sub>e qua</sub>ntiti<sub>es assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>o</sub>n<sub>e</sub> PIE <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>as</sub>

$$
\begin{array} { r } { \mathcal { P } _ { k } = \left( \mathbf { u } , t _ { k } , t _ { k + 1 } , p _ { k + 1 } , \Delta t _ { k } , \mathbf { l } _ { k } , \mathbf { l } _ { k + 1 } \right) . } \end{array}\tag{8}
$$

Th<sub>e</sub> r<sub>esu</sub>ltin<sub>g</sub> PIE r<sub>eco</sub>rd <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> th<sub>e eve</sub>nt-<sub>pa</sub>ir inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd corres<sub>p</sub>ondin<sub>g</sub> li<sub>g</sub>ht directions<sub>,</sub> without ex<sub>p</sub>licitl<sub>y</sub> includin<sub>g</sub> the unk<sub>nown con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld �<sub>.</sub>

## 3.2 Physical Irradiance Event Graph Neural Network

F<sub>o</sub>r <sub>eac</sub>h PIE $\mathcal { P } _ { k }$ , we define its PIE Feature (PIEF) as

$$
\phi _ { k } = \frac { { p } _ { k + 1 } } { \Delta t _ { k } } .\tag{9}
$$

This scalar is the si<sub>g</sub>ned event rate. It kee<sub>p</sub>s event timin<sub>g</sub> and <sub>p</sub>olarit<sub>y,</sub> <sub>an</sub>d it d<sub>oes no</sub>t <sub>requ</sub>i<sub>re</sub> th<sub>e un</sub>k<sub>nown</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>sca</sub>l<sub>e</sub> �<sub>.</sub>

W<sub>e</sub> tr<sub>ea</sub>t <sub>eac</sub>h PIE <sub>as o</sub>n<sub>e</sub> PIE-GNN n<sub>o</sub>d<sub>e.</sub> L<sub>e</sub>t $\tau _ { k } = ( t _ { k } + t _ { k + 1 } ) / 2$ be its mid<sub>p</sub>oint time and let $\bar { \mathbf { u } } _ { k }$ b<sub>e</sub> it<sub>s norma</sub>li<sub>ze</sub>d <sub>p</sub>i<sub>xe</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>e.</sub> Th<sub>e no</sub>d<sub>e coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>s</sub>

$$
\mathbf { q } _ { k } = \left[ \bar { \mathbf { u } } _ { k } ^ { \top } , \gamma \tau _ { k } \right] ^ { \top } ,\tag{10}
$$

where <sub>�</sub> is a time scale. We b<sub>u</sub>ild the <sub>g</sub>ra<sub>p</sub>h <sub>u</sub>sed b<sub>y</sub> the PIE-GNN as

$$
\begin{array} { r } { G _ { \mathrm { p i e } } = ( \mathcal { V } _ { \mathrm { p i e } } , \mathcal { E } _ { \mathrm { p i e } } ) , \qquad \mathcal { V } _ { \mathrm { p i e } } = \{ \mathcal { P } _ { k } \} . } \end{array}\tag{11}
$$

Th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> <sub>se</sub>t $\mathcal { E } _ { \mathrm { p i e } }$ connects PIE nodes that are close in ima<sub>g</sub>e s<sub>p</sub>ace <sub>an</sub>d ti<sub>me.</sub> W<sub>e a</sub>dd <sub>a</sub> di<sub>rec</sub>t<sub>e</sub>d <sub>e</sub>d<sub>ge</sub> $e _ { i j }$ <sub>w</sub>h<sub>en</sub>

$$
\left\| \bar { \mathbf { u } } _ { i } - \bar { \mathbf { u } } _ { j } \right\| _ { \infty } \leq r _ { \mathrm { p i e } } , \qquad 0 < \tau _ { j } - \tau _ { i } \leq \Delta \tau _ { \mathrm { m a x } } .\tag{12}
$$

H<sub>e</sub>r<sub>e</sub> $r _ { \mathrm { p i e } }$ i<sub>s</sub> th<sub>e spa</sub>ti<sub>a</sub>l <sub>searc</sub>h <sub>ra</sub>di<sub>us.</sub> $\Delta \tau _ { \mathrm { m a x } }$ i<sub>s</sub> th<sub>e</sub> t<sub>empora</sub>l <sub>w</sub>i<sub>n</sub>d<sub>ow.</sub> F<sub>or</sub> <sub>eac</sub>h <sub>e</sub>d<sub>ge,</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> f<sub>ea</sub>t<sub>ure</sub> <sub>as</sub>

$$
\mathbf { a } _ { i j } = \frac { \mathbf { q } _ { j } - \mathbf { q } _ { i } } { 2 \rho _ { \mathrm { p i e } } } + \frac { 1 } { 2 } \mathbf { 1 } , \qquad \mathbf { \mathit { e } } _ { i j } \in \mathcal { E } _ { \mathrm { p i e } } .\tag{13}
$$

Here, $\rho _ { \mathrm { p i e } }$ i<sub>s</sub> th<sub>e ra</sub>di<sub>us use</sub>d t<sub>o norma</sub>li<sub>ze</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve coor</sub>di<sub>na</sub>t<sub>e,</sub> and 1 $\in \mathbb { R } ^ { 3 }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e a</sub>ll<sub>-ones vec</sub>t<sub>or.</sub>

F<sub>o</sub>r <sub>g</sub>r<sub>ap</sub>h f<sub>ea</sub>t<sub>u</sub>r<sub>e e</sub>xtr<sub>ac</sub>ti<sub>o</sub>n<sub>, eac</sub>h PIE n<sub>o</sub>d<sub>e uses</sub> th<sub>e</sub> PIEF <sub>a</sub>nd th<sub>e</sub> li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> as its base feature:

$$
\mathbf { x } _ { k } = \left[ \phi _ { k } , \mathbf { l } _ { k } ^ { \top } , \mathbf { l } _ { k + 1 } ^ { \top } \right] ^ { \top } .\tag{14}
$$

SA C<sub>o</sub>nf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> P<sub>ape</sub>r<sub>s</sub> ’26<sub>,</sub> D<sub>ece</sub>mb<sub>e</sub>r 01–04<sub>,</sub> 2026<sub>,</sub> K<sub>ua</sub>l<sub>a</sub> L<sub>u</sub>m<sub>pu</sub>r<sub>,</sub> M<sub>a</sub>l<sub>ays</sub>i<sub>a.</sub>

![](images/44158c9d85231da832c22dcce5ea7180348cee4f1d13de49b8cc7981921f15aa.jpg)  
Fig. 2. Overview of the PIE-PS framework. We first convert raw events under moving light into PIE observations. Each PIE provides a threshold-free signed event-rate feature and a corresponding light-pair geometry. The PIE-GNN encodes these observations, uses Reliability-Grading Atention (RGA) to weight them, aggregates pixel features, and predicts normals with a pixel aggregator.

The <sub>p</sub>ixel coordinate is used as an additional in<sub>p</sub>ut for <sub>g</sub>ra<sub>p</sub>h convol<sub>u</sub>ti<sub>o</sub>n <sub>a</sub>nd n<sub>e</sub>i<sub>g</sub>hb<sub>o</sub>r <sub>sea</sub>r<sub>c</sub>h<sub>.</sub> A <sub>sp</sub>lin<sub>e</sub> <sub>g</sub>r<sub>ap</sub>h <sub>co</sub>n<sub>vo</sub>l<sub>u</sub>ti<sub>o</sub>n <sub>e</sub>xtr<sub>ac</sub>t<sub>s</sub> <sub>a</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l PIE f<sub>ea</sub>t<sub>ure</sub> f<sub>rom</sub> $\textstyle \mathcal { G } _ { \mathrm { p i e } }$

$$
\begin{array} { r } { \mathbf { h } _ { k } = f _ { \mathrm { o b s } } \left( \mathrm { G C o n v } _ { \mathrm { p i e } } ( \mathbf { x } _ { k } , \bar { \mathbf { u } } _ { k } , \mathcal { E } _ { \mathrm { p i e } } , \{ \mathbf { a } _ { i j } \} ) \right) . } \end{array}\tag{15}
$$

H<sub>e</sub>r<sub>e</sub> ${ \mathrm { G C o n v } } _ { \mathrm { p i e } }$ denotes the PIE-GNN <sub>g</sub>ra<sub>p</sub>h con<sub>v</sub>ol<sub>u</sub>tion<sub>,</sub> <sub>u</sub>sin<sub>g</sub> $\mathcal { E } _ { \mathrm { p i e } }$ <sub>an</sub>d it<sub>s</sub> <sub>e</sub>d<sub>ge</sub> f<sub>ea</sub>t<sub>ures.</sub> $f _ { \mathrm { o b s } }$ is an MLP that ma<sub>p</sub>s the <sub>g</sub>ra<sub>p</sub>h-convolution o<sub>u</sub>t<sub>pu</sub>t to a PIE embeddin<sub>g</sub>. Th<sub>u</sub>s<sub>,</sub> $\mathbf { h } _ { k }$ k<sub>eeps</sub> th<sub>e s</sub>i<sub>gne</sub>d<sub>-ra</sub>t<sub>e measure-</sub> <sub>men</sub>t<sub>,</sub> th<sub>e</sub> l<sub>oca</sub>l li<sub>g</sub>ht<sub>-pa</sub>i<sub>r</sub> <sub>cues,</sub> <sub>an</sub>d <sub>con</sub>t<sub>ex</sub>t f<sub>rom</sub> <sub>near</sub>b<sub>y</sub> PIE<sub>s.</sub>

Let $N ( \mathbf { u } ) \ = \ \{ k \ | \ \mathcal { P } _ { k }$ is observed at pixel u}. This is the set of PIE nodes that belong to pixel u. The Reliability-Grading Attention (RGA) module <sub>p</sub>redicts a reliabilit<sub>y g</sub>rade $w _ { k } \in \left[ 0 , 1 \right]$ f<sub>or eac</sub>h PIE node. It is described in Sec. 3.3. We <sub>u</sub>se these <sub>g</sub>rades to s<sub>u</sub>mmarize PIE embeddin<sub>g</sub>s at each <sub>p</sub>ixel:

$$
\bar { \mathbf { h } } _ { \mathbf { u } } = \frac { \sum _ { k \in \mathcal { N } ( \mathbf { u } ) } w _ { k } \mathbf { h } _ { k } } { \sum _ { k \in \mathcal { N } ( \mathbf { u } ) } w _ { k } + \epsilon } , \qquad \mathbf { m } _ { \mathbf { u } } = \operatorname* { m a x } _ { k \in \mathcal { N } ( \mathbf { u } ) } w _ { k } \mathbf { h } _ { k } .\tag{16}
$$

Here <sub>�</sub> avoids division b<sub>y</sub> zero<sub>,</sub> and the max is taken element-wise. T<sup>h</sup>e pixe<sup>l</sup> <sup>f</sup>eature is o<sup>b</sup>taine<sup>d</sup> <sup>b</sup>y projecting t<sup>h</sup>e concatenate<sup>d</sup> statistics:

$$
\begin{array} { r } { \mathbf { s } _ { \mathbf { u } } = f _ { \mathrm { p o o l } } \left( \left[ \bar { \mathbf { h } } _ { \mathbf { u } } , \mathbf { m } _ { \mathbf { u } } \right] \right) . } \end{array}\tag{17}
$$

A li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>p</sub>ix<sub>e</sub>l <sub>agg</sub>r<sub>ega</sub>t<sub>o</sub>r th<sub>e</sub>n r<sub>e</sub>fin<sub>es</sub> th<sub>ese</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>es</sub> <sub>ove</sub>r nearb<sub>y</sub> valid <sub>p</sub>ixels. Let M (u) denote the s<sub>p</sub>atial nei<sub>g</sub>hbors of <sub>p</sub>ixel u. The aggregator output is

$$
\mathbf { q } _ { \mathbf { u } } = F _ { \mathrm { p i x } } \left( \left\{ \mathbf { s } _ { \mathbf { v } } ~ | ~ \mathbf { v } \in M ( \mathbf { u } ) \cup \{ \mathbf { u } \} \right\} \right) .\tag{18}
$$

More details of this <sub>p</sub>ixel a<sub>gg</sub>re<sub>g</sub>ation $F _ { \mathrm { p i x } }$ are <sub>p</sub>rovided in the a<sub>p</sub>- <sub>pen</sub>di<sub>x.</sub> Th<sub>e norma</sub>l h<sub>ea</sub>d <sub>maps</sub> thi<sub>s re</sub>fi<sub>ne</sub>d <sub>p</sub>i<sub>xe</sub>l f<sub>ea</sub>t<sub>ure</sub> t<sub>o</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>norma</sub>l<sub>:</sub>

$$
\hat { \bf n } _ { \bf u } = { \bf N o r m } \left( f _ { \mathrm { n o r m a l } } ( { \bf q } _ { \bf u } ) \right) ,\tag{19}
$$

<sub>w</sub>h<sub>ere</sub> $f _ { \mathrm { p o o l } }$ <sub>an</sub>d $f _ { \mathrm { n o r m a l } }$ are MLPs, and Norm(·) denotes $L _ { 2 }$ <sub>norma</sub>l<sub>-</sub> iz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub>

## 3.3 Reliability-Grading Atention

E<sub>q</sub>. 6 shows that the PIEF is <sub>p</sub>ro<sub>p</sub>ortional to the measured lo<sub>g</sub>-ima<sub>g</sub>ei<sub>rra</sub>di<sub>ance c</sub>h<sub>ange, up</sub> t<sub>o</sub> th<sub>e un</sub>k<sub>nown sca</sub>l<sub>e</sub> �<sub>.</sub> U<sub>n</sub>d<sub>er</sub> id<sub>ea</sub>l L<sub>am-</sub> bertian reflectance and smoothl<sub>y</sub> var<sub>y</sub>in<sub>g</sub> illumination<sub>,</sub> PIEF observations at the same <sub>p</sub>ixel are ex<sub>p</sub>ected to var<sub>y</sub> consistentl<sub>y</sub> with their corres<sub>p</sub>ondin<sub>g</sub> li<sub>g</sub>ht <sub>p</sub>airs. In <sub>p</sub>ractice<sub>,</sub> the reliabilit<sub>y</sub> of PIE observations can var<sub>y</sub> with local a<sub>pp</sub>earance<sub>,</sub> illumination <sub>g</sub>eometr<sub>y,</sub> and sensor noise. Less reliable observations ma<sub>y</sub> a<sub>pp</sub>ear as abru<sub>p</sub>t d<sub>ev</sub>i<sub>a</sub>ti<sub>ons</sub> f<sub>rom</sub> thi<sub>s same-p</sub>i<sub>xe</sub>l t<sub>empora</sub>l t<sub>ren</sub>d<sub>, as</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> Fi<sub>g</sub>. 3. We <sub>p</sub>ro<sub>p</sub>ose RGA to assi<sub>g</sub>n each PIE a soft reliabilit<sub>y</sub> <sub>g</sub>rade <sup>for subse</sup>q<sup>uent a</sup>gg<sup>re</sup>g<sup>ation.</sup>

![](images/18f075a3bfe7bfb9c8a812bf8db063b06f3497298465c22132471ccfd5695128.jpg)  
Fig. 3. The Reliability-Grading Atention module scores each event-pair observation. It detects spikes in the PIE sequence and predicts a confidence weight for each observation. The weighted observations are then aggregated into pixel features.

F<sub>or a re</sub>li<sub>a</sub>bl<sub>e</sub> PIE<sub>,</sub> th<sub>e measure</sub>d <sub>even</sub>t <sub>con</sub>t<sub>ras</sub>t <sub>s</sub>h<sub>ou</sub>ld <sub>agree</sub> <sub>w</sub>ith th<sub>e</sub> L<sub>am</sub>b<sub>er</sub>ti<sub>an c</sub>h<sub>ange pre</sub>di<sub>c</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> t<sub>wo</sub> li<sub>g</sub>ht di<sub>rec</sub>ti<sub>ons.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> E<sub>q.</sub> 3 int<sub>o</sub> E<sub>q.</sub> 5 <sub>g</sub>i<sub>ves</sub>

$$
\ln \left( \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } _ { k + 1 } \right) - \ln \left( \mathbf { n } _ { \mathbf { u } } ^ { \top } \mathbf { l } _ { k } \right) \approx p _ { k + 1 } C .\tag{20}
$$

Ex<sub>p</sub>onentiatin<sub>g</sub> both sides <sub>g</sub>ives

$$
\frac {  { \mathbf { n } } _ {  { \mathbf { u } } } ^ { \top }  { \mathbf { l } } _ { k + 1 } } {  { \mathbf { n } } _ {  { \mathbf { u } } } ^ { \top }  { \mathbf { l } } _ { k } } \approx \exp ( p _ { k + 1 } C ) .\tag{21}
$$

T<sup>h</sup>is step uses t<sup>h</sup>e <sup>d</sup>iscrete re<sup>l</sup>ation <sup>b</sup>etween two a<sup>d</sup>jacent events <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e con</sub>ti<sub>nuous-</sub>ti<sub>me</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve.</sub> Thi<sub>s</sub> l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> di<sub>scre</sub>t<sub>e</sub> <sub>pa</sub>ir<sub>w</sub>i<sub>se co</sub>n<sub>s</sub>tr<sub>a</sub>int

$$
\begin{array} { r } { \mathbf { n } _ { \mathbf { u } } ^ { \top } \left( \mathbf { l } _ { k + 1 } - \exp ( p _ { k + 1 } C ) \mathbf { l } _ { k } \right) \approx 0 . } \end{array}\tag{22}
$$

Wh<sub>en a</sub> PIE i<sub>s re</sub>li<sub>a</sub>bl<sub>e,</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e o</sub>f $\operatorname { E q } .$ 22 <sub>s</sub>h<sub>ou</sub>ld b<sub>e c</sub>l<sub>ose</sub> t<sub>o</sub> <sub>zero.</sub> A l<sub>arge magn</sub>it<sub>u</sub>d<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es a po</sub>t<sub>en</sub>ti<sub>a</sub>ll<sub>y unre</sub>li<sub>a</sub>bl<sub>e</sub> PIE <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>.</sub>

I<sub>n</sub> <sub>syn</sub>th<sub>e</sub>ti<sub>c</sub> d<sub>a</sub>t<sub>a,</sub> th<sub>e</sub> <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th <sub>norma</sub>l $\mathbf { n } _ { \mathrm { g t , u } }$ <sub>an</sub>d th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld � are known. We use E<sub>q</sub>. 22 onl<sub>y</sub> to identif<sub>y</sub> unreliable PIEs for

<sub>ana</sub>l<sub>ys</sub>i<sub>s an</sub>d <sub>compar</sub>i<sub>son.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\mathbf { z } _ { k } = \mathbf { l } _ { k + 1 } - \exp ( { \hat { p } } _ { k + 1 } C ) \mathbf { l } _ { k } , \qquad r _ { k } = { \frac { \left| \mathbf { n } _ { \mathrm { g t } , { \mathbf { u } } } ^ { \top } \mathbf { z } _ { k } \right| } { \left\| \mathbf { z } _ { k } \right\| _ { 2 } + \epsilon } } .\tag{23}
$$

Th<sub>e score</sub> $r _ { k }$ <sub>measures</sub> h<sub>ow muc</sub>h <sub>one</sub> PIE <sub>v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> di<sub>scre</sub>t<sub>e</sub> <sub>pa</sub>ir<sub>w</sub>i<sub>se</sub> <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>int<sub>.</sub> A <sub>s</sub>m<sub>a</sub>ll $r _ { k }$ m<sub>ea</sub>n<sub>s</sub> thi<sub>s</sub> PIE <sub>ag</sub>r<sub>ees</sub> <sub>w</sub>ith th<sub>e</sub> Lambertian <sub>p</sub>rediction. A lar<sub>g</sub>e $r _ { k }$ m<sub>ea</sub>n<sub>s</sub> thi<sub>s</sub> PIE d<sub>oes</sub> n<sub>o</sub>t <sub>ag</sub>r<sub>ee</sub> <sub>w</sub>ith th<sub>e pre</sub>di<sub>c</sub>ti<sub>on, so</sub> it i<sub>s</sub> t<sub>rea</sub>t<sub>e</sub>d <sub>as unre</sub>li<sub>a</sub>bl<sub>e.</sub> W<sub>e a</sub>l<sub>so</sub> h<sub>an</sub>dl<sub>e</sub> shadow and <sub>g</sub>razin<sub>g</sub> li<sub>g</sub>htin<sub>g</sub> se<sub>p</sub>aratel<sub>y</sub>. In these cases<sub>,</sub> $\mathbf { n } _ { \mathrm { g t } , \mathbf { u } } ^ { \top } \mathbf { l } _ { k }$ or $\mathbf { n } _ { \mathrm { g t } , \mathbf { u } } ^ { \top } \mathbf { l } _ { k + 1 }$ is non-<sub>p</sub>ositive or close to zero<sub>,</sub> so the Lambertian lo<sub>g</sub>- ima<sub>g</sub>e-irradiance e<sub>q</sub>uation is invalid or unstable. We therefore mark th<sub>ese</sub> PIE<sub>s as u</sub>nr<sub>e</sub>li<sub>a</sub>bl<sub>e w</sub>ith<sub>ou</sub>t <sub>us</sub>in<sub>g</sub> $r _ { k }$ <sub>.</sub> Th<sub>e</sub> r<sub>e</sub>m<sub>a</sub>inin<sub>g</sub> PIE<sub>s a</sub>r<sub>e</sub> t<sub>rea</sub>t<sub>e</sub>d <sub>as</sub> GT<sub>-</sub>id<sub>en</sub>tifi<sub>e</sub>d <sub>re</sub>li<sub>a</sub>bl<sub>e</sub> PIE<sub>s.</sub> Th<sub>ese</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s are use</sub>d <sub>on</sub>l<sub>y</sub> for visualization<sub>,</sub> dia<sub>g</sub>nostics<sub>,</sub> and com<sub>p</sub>arison. The<sub>y</sub> are not model in<sub>pu</sub>t<sub>s</sub> <sub>a</sub>t t<sub>es</sub>t tim<sub>e.</sub>

RGA su<sub>pp</sub>resses unreliable observations b<sub>y</sub> <sub>p</sub>redictin<sub>g</sub> a confid<sub>e</sub>n<sub>ce we</sub>i<sub>g</sub>ht f<sub>o</sub>r <sub>eac</sub>h PIE<sub>.</sub> In <sub>ou</sub>r im<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> RGA <sub>uses a</sub> soft-<sub>g</sub>atin<sub>g</sub> attention. It assi<sub>g</sub>ns an individual confidence wei<sub>g</sub>ht to each PIE in the same-<sub>p</sub>ixel se<sub>q</sub>uence. Fi<sub>g</sub>. 3 shows this <sub>p</sub>rocess on the time-ordered PIE sequence of one pixel. Given a pixel u, we <sub>co</sub>ll<sub>ec</sub>t th<sub>e</sub> ti<sub>me-or</sub>d<sub>ere</sub>d PIE <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub>

$$
\mathcal { H } _ { \mathbf { u } } = \{ { \mathbf { h } } _ { k } \ | \ k \in N ( { \mathbf { u } } ) \} .\tag{24}
$$

Th<sub>e em</sub>b<sub>e</sub>ddi<sub>ngs are sor</sub>t<sub>e</sub>d b<sub>y</sub> ti<sub>me.</sub>

A <sub>sma</sub>ll <sub>se</sub>lf<sub>-a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>scorer</sub> <sub>rea</sub>d<sub>s</sub> th<sub>e</sub> ti<sub>me-or</sub>d<sub>ere</sub>d <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>sequence an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>s an</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>con</sub>fid<sub>ence we</sub>i<sub>g</sub>ht f<sub>or eac</sub>h PIE<sub>.</sub> F<sub>or</sub> <sub>eac</sub>h PIE <sub>em</sub>b<sub>e</sub>ddi<sub>ng,</sub> <sub>we</sub> fi<sub>rs</sub>t f<sub>orm</sub> <sub>query,</sub> k<sub>ey,</sub> <sub>an</sub>d <sub>va</sub>l<sub>ue</sub> <sub>vec</sub>t<sub>ors:</sub>

$$
\mathbf { q } _ { k } = W _ { q } \mathbf { h } _ { k } , \qquad \mathbf { k } _ { k } = W _ { k } \mathbf { h } _ { k } , \qquad \mathbf { v } _ { k } = W _ { v } \mathbf { h } _ { k } .\tag{25}
$$

Th<sub>e</sub> <sub>a</sub>tt<sub>e</sub>nti<sub>o</sub>n <sub>sco</sub>r<sub>e</sub> f<sub>o</sub>r PIE � i<sub>s</sub> <sub>co</sub>m<sub>pu</sub>t<sub>e</sub>d fr<sub>o</sub>m <sub>a</sub>ll PIE<sub>s</sub> <sub>a</sub>t th<sub>e</sub> <sub>sa</sub>m<sub>e</sub> <sub>p</sub>ixel:

$$
\alpha _ { k j } = \mathrm { s o f t m a x } _ { j \in N ( \mathbf { u } ) } \left( \frac { \mathbf { q } _ { k } ^ { \top } \mathbf { k } _ { j } } { \sqrt { d } } \right) , \qquad \mathbf { y } _ { k } = \sum _ { j \in N ( \mathbf { u } ) } \alpha _ { k j } \mathbf { v } _ { j } .\tag{26}
$$

Th<sub>e</sub> RGA <sub>co</sub>nfid<sub>e</sub>n<sub>ce</sub> <sub>we</sub>i<sub>g</sub>ht i<sub>s</sub> th<sub>e</sub>n

$$
w _ { k } = \sigma ( f _ { g } ( \mathbf { y } _ { k } ) ) .\tag{27}
$$

H<sub>e</sub>r<sub>e</sub> $f _ { g }$ ma<sub>p</sub>s each attention out<sub>p</sub>ut to a scalar lo<sub>g</sub>it<sub>,</sub> and $\sigma ( \cdot )$ i<sub>s</sub> the si<sub>g</sub>moid function. The si<sub>g</sub>moid out<sub>p</sub>ut lets multi<sub>p</sub>le PIEs remain <sub>ac</sub>ti<sub>ve</sub> <sub>a</sub>t <sub>o</sub>n<sub>e</sub> <sub>p</sub>ix<sub>e</sub>l<sub>.</sub> RGA r<sub>e</sub>d<sub>uces</sub> th<sub>e</sub> <sub>co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>f PIE<sub>s</sub> <sub>ass</sub>i<sub>g</sub>n<sub>e</sub>d lower reliabilit<sub>y g</sub>rades. The wei<sub>g</sub>hts are used in E<sub>q</sub>. 16 to form the <sub>p</sub>i<sub>xe</sub>l f<sub>ea</sub>t<sub>ure.</sub>

## 3.4 Training Strategy

We train the network with direct normal su<sub>p</sub>ervision. Let Ω be the set of valid trainin<sub>g p</sub>ixels. The main loss is the mean an<sub>g</sub>ular error <sub>on</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>norma</sub>l<sub>:</sub>

$$
\mathcal { L } _ { \mathrm { n o r m a l } } = \frac { 1 } { | \Omega | } \sum _ { \mathbf { u } \in \Omega } \operatorname { a r c c o s } \left( \hat { \mathbf { n } } _ { \mathbf { u } } ^ { \top } \mathbf { n } _ { \mathrm { g t } , \mathbf { u } } \right) .\tag{28}
$$

At t<sub>es</sub>t ti<sub>me,</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>on</sub>l<sub>y</sub> <sub>nee</sub>d<sub>s</sub> <sub>even</sub>t <sub>pa</sub>i<sub>rs</sub> <sub>an</sub>d li<sub>g</sub>ht di<sub>rec</sub>ti<sub>ons.</sub>

![](images/0879e131de6270c21fc2834b668c5702fee986478b91f3e246eee83569770b53.jpg)  
Fig. 4. Prototype acquisition system used for real-world event-based photometric stereo.

## 4 Experiment

## 4.1 Implementation Details

Method Implementation. We implement PIE-PS in PyTorch. The <sub>p</sub>h<sub>y</sub>sics-onl<sub>y</sub> direct solver is im<sub>p</sub>lemented in NumP<sub>y</sub>. For each <sub>p</sub>ixel<sub>,</sub> it <sub>op</sub>timiz<sub>es</sub> E<sub>q.</sub> 7 ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y.</sub> W<sub>e use e</sub>x<sub>ac</sub>t SVD f<sub>o</sub>r initi<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n and r<sub>u</sub>n fi<sub>v</sub>e IRLS iterations <sub>w</sub>ith t<sub>w</sub>o tan<sub>g</sub>ent-<sub>p</sub>lane Ga<sub>u</sub>ss-Ne<sub>w</sub>ton <sub>s</sub>t<sub>eps.</sub> Pi<sub>xe</sub>l<sub>s w</sub>ith f<sub>ewer</sub> th<sub>an</sub> fi<sub>ve</sub> PIE <sub>o</sub>b<sub>serva</sub>ti<sub>ons are no</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d<sub>.</sub> F<sub>o</sub>r PIE-GNN<sub>, we use a</sub> 7-D n<sub>o</sub>d<sub>e</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>e, a</sub> 64-D PIE <sub>e</sub>mb<sub>e</sub>ddin<sub>g, a</sub>nd <sub>a</sub> 128<sub>-</sub>D <sub>grap</sub>h hidd<sub>en</sub> f<sub>ea</sub>t<sub>ure.</sub> Th<sub>e ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>searc</sub>h <sub>mo</sub>d<sub>u</sub>l<sub>es use</sub> c<sub>u</sub>stom CUDA kernels. For the PIE <sub>g</sub>ra<sub>p</sub>h<sub>,</sub> we set the s<sub>p</sub>atial search <sub>ra</sub>di<sub>us</sub> t<sub>o</sub> $r _ { \mathrm { p i e } } = 0 . 0 5$ . PIE times are normalized to [0, 1] within each <sub>sequence, an</sub>d <sub>we use</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>norma</sub>li<sub>ze</sub>d <sub>range as</sub> th<sub>e</sub> t<sub>empora</sub>l <sub>w</sub>i<sub>n</sub>d<sub>ow, so</sub> $\Delta \tau _ { \mathrm { m a x } } \ = \ 1$ . The tem<sub>p</sub>oral scale in E<sub>q</sub>. 10 is $\gamma = 0 . 3$ For ed<sub>g</sub>e-feature normalization<sub>,</sub> we set $\rho _ { \mathrm { p i e } } ~ = ~ 2 \lfloor r _ { \mathrm { p i e } } W + 2 \rfloor / W$ where � is the ima<sub>g</sub>e width. The <sub>p</sub>ixel a<sub>gg</sub>re<sub>g</sub>ator im<sub>p</sub>lementation is detailed in the s<sub>upp</sub>lementar<sub>y</sub> material. The RGA scorer is a t<sub>w</sub>ol<sub>ayer</sub> T<sub>rans</sub>f<sub>ormer w</sub>ith f<sub>our a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>s an</sub>d <sub>a</sub> 64<sub>-</sub>D hidd<sub>en</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>e.</sub> W<sub>e</sub> <sub>op</sub>timiz<sub>e</sub> th<sub>e</sub> n<sub>e</sub>t<sub>wo</sub>rk <sub>w</sub>ith Ad<sub>a</sub>m <sub>us</sub>in<sub>g</sub> th<sub>e</sub> n<sub>o</sub>rm<sub>a</sub>l-<sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n l<sub>oss</sub> in S<sub>ec.</sub> 3<sub>.</sub>4<sub>.</sub> Tr<sub>a</sub>inin<sub>g</sub> t<sub>a</sub>k<sub>es</sub> <sub>a</sub>b<sub>ou</sub>t 3<sub>.</sub>4 h<sub>ou</sub>r<sub>s</sub> f<sub>o</sub>r 200 <sub>epoc</sub>h<sub>s o</sub>n <sub>a s</sub>in<sub>g</sub>l<sub>e</sub> NVIDIA H100 PCI<sub>e</sub> GPU<sub>.</sub>

Prototype System. To validate our method on real-world data, we constructed the <sub>p</sub>rotot<sub>yp</sub>e ac<sub>q</sub>uisition s<sub>y</sub>stem shown in Fi<sub>g</sub>. 4. The ima<sub>g</sub>in<sub>g</sub> setu<sub>p</sub> uses a CeleX-V event camera with a s<sub>p</sub>atial resolution of 1280 × 800. The camera is e<sub>q</sub>ui<sub>pp</sub>ed with a Zhon<sub>g</sub>lian Kechuan<sub>g</sub> TM5028MP12 <sub>macro</sub> l<sub>ens w</sub>ith <sub>a</sub> 50<sub>mm</sub> f<sub>oca</sub>l l<sub>eng</sub>th<sub>,</sub> 1<sub>.</sub>1<sub>-</sub>i<sub>nc</sub>h f<sub>orma</sub>t<sub>,</sub> and C-mount interface. For illumination<sub>,</sub> we use a <sub>p</sub>ro<sub>g</sub>rammable rin<sub>g</sub> li<sub>g</sub>ht <sub>w</sub>ith 60 WS2812 5050 RGB LED b<sub>ea</sub>d<sub>s.</sub> An Ard<sub>u</sub>in<sub>o</sub> Un<sub>o</sub> R3 se<sub>qu</sub>entiall<sub>y</sub> dri<sub>v</sub>es the LEDs and records the acti<sub>v</sub>ation timestam<sub>p</sub>s and li<sub>g</sub>ht directions. The li<sub>g</sub>htin<sub>g</sub> timestam<sub>p</sub>s are manuall<sub>y</sub> <sub>sync</sub>h<sub>ron</sub>i<sub>ze</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>even</sub>t <sub>camera</sub> <sub>c</sub>l<sub>oc</sub>k<sub>.</sub> W<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub> th<sub>e</sub> <sub>se</sub>t<sub>up</sub> so t<sup>h</sup>at t<sup>h</sup>e camera optica<sup>l</sup> center, ring <sup>l</sup>ig<sup>h</sup>t center, an<sup>d</sup> target o<sup>b</sup>ject <sub>are a</sub>li<sub>gne</sub>d <sub>on</sub> th<sub>e same</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>ax</sub>i<sub>s.</sub>

## 4.2 Datasets

T<sub>o</sub> tr<sub>a</sub>in PIE-PS <sub>a</sub>nd <sub>e</sub>n<sub>su</sub>r<sub>e</sub> f<sub>a</sub>ir <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n <sub>w</sub>ith <sub>p</sub>ri<sub>o</sub>r m<sub>e</sub>th<sub>o</sub>d<sub>s, we</sub> <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e a sy</sub>nth<sub>e</sub>ti<sub>c</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> n<sub>a</sub>m<sub>e</sub>d PIE-Sim<sub>.</sub> PIE-Sim f<sub>o</sub>ll<sub>ows</sub> th<sub>e</sub> same simulation <sub>p</sub>i<sub>p</sub>eline as EventPS [Yu et al. 2024]. It uses objects

Object

EventPS-FCN

EventPS-CNN

Direct Solver

PIE-PS

GT

![](images/4ef90a7b3ef31420aca5f5de8c3037cab3c7f49dbfb34348ab6a630012733e1e.jpg)

Fig. 5. Qualitative comparison across synthetic and real scenes. Each row shows the object, the normal prediction and angular error map of each method, and the ground-truth normal. We include PIE-Sim, DiLiGenT-Sim, EventPS real scenes, and our captured scenes. Error maps use the same 0<sup>◦</sup>–30<sup>◦</sup> color range, where darker colors indicate lower error.  
![](images/b3e4e9f726b3f577bd8a36176903ad16be881207e704e5c476056f8017efc33e.jpg)

Fig. 6. RGA reliability visualization. Each object is shown at three time slices. For each time slice, we show the raw frame, the GT-identified unreliable PIE ratio, and the low-RGA PIE ratio. GT-identified unreliable PIEs are computed with ground-truth normals and known simulation thresholds only for analysis. The figure label GT-bad denotes these GT-identified unreliable PIEs. Low-RGA maps are produced by the learned RGA scorer.

Table 1. Quantitative comparison on DiLiGenT-Sim. MAE is reported in degrees, and lower is beter.
<table><tr><td>Method</td><td>BALL</td><td>BUDDHA</td><td>CAT</td><td>Cow</td><td>GOBLET</td><td>HARVEST</td><td>PoT1</td><td>PoT2</td><td>READING</td><td>Average</td></tr><tr><td>EventUPS</td><td>8.86</td><td>12.21</td><td>10.78</td><td>18.58</td><td>12.97</td><td>23.32</td><td>8.70</td><td>15.13</td><td>16.71</td><td>14.14</td></tr><tr><td>EventPS-OP</td><td>10.99</td><td>18.73</td><td>12.74</td><td>26.51</td><td>18.43</td><td>36.06</td><td>13.78</td><td>15.75</td><td>24.61</td><td>19.73</td></tr><tr><td>EventPS-FCN</td><td>7.49</td><td>18.13</td><td>11.42</td><td>20.61</td><td>18.07</td><td>26.05</td><td>12.83</td><td>16.59</td><td>15.16</td><td>16.26</td></tr><tr><td>EventPS-CNN</td><td>10.44</td><td>16.79</td><td>11.88</td><td>20.60</td><td>16.44</td><td>25.26</td><td>12.93</td><td>15.54</td><td>18.19</td><td>16.45</td></tr><tr><td>Direct Solver</td><td>15.36</td><td>15.09</td><td>13.75</td><td>14.55</td><td>15.62</td><td>20.07</td><td>13.59</td><td>14.67</td><td>17.72</td><td>15.60</td></tr><tr><td>PIE-PS</td><td>3.93</td><td>8.67</td><td>5.15</td><td>12.90</td><td>7.81</td><td>12.91</td><td>6.33</td><td>8.34</td><td>12.25</td><td>8.70</td></tr></table>

from the Blobb<sub>y</sub> [Johnson and Adelson 2011] and Scul<sub>p</sub>ture [Wiles and Zisserman 2017] collections, rendered with random <sub>p</sub>oses and a<sub>pp</sub>earance <sub>p</sub>arameters followin<sub>g</sub> the EventPS setu<sub>p</sub>. We simulate event streams usin<sub>g</sub> ESIM [Rebec<sub>q</sub> et al. 2018b] under three li<sub>g</sub>htin<sub>g</sub> trajectories, inc<sup>l</sup>u<sup>d</sup>ing circ<sup>l</sup>e, <sup>h</sup>ypotroc<sup>h</sup>oi<sup>d</sup>, an<sup>d</sup> interpo<sup>l</sup>ate<sup>d</sup> DiLi-G<sub>e</sub>nT<sub>.</sub> E<sub>ac</sub>h PIE-Sim <sub>sce</sub>n<sub>e</sub> i<sub>s ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>a</sub>n ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y sa</sub>m-<sub>p</sub>l<sub>e</sub>d <sub>con</sub>t<sub>ras</sub>t th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on-no</sub>i<sub>se</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> diferent event densities and inter-event intervals. Each se<sub>q</sub>uence <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> 600 r<sub>e</sub>nd<sub>e</sub>r<sub>e</sub>d fr<sub>a</sub>m<sub>es.</sub> PIE-Sim <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> 77 tr<sub>a</sub>inin<sub>g</sub> <sub>sce</sub>n<sub>es</sub> an<sup>d</sup> 30 test scenes wit<sup>h</sup> no o<sup>b</sup>ject over<sup>l</sup>ap <sup>b</sup>etween t<sup>h</sup>e two sp<sup>l</sup>its. To test <sub>g</sub>eneralization to standard <sub>p</sub>hotometric-stereo <sub>g</sub>eometr<sub>y,</sub> we <sub>a</sub>l<sub>so</sub> intr<sub>o</sub>d<sub>uce</sub> DiLiG<sub>e</sub>nT-Sim<sub>.</sub> It i<sub>s</sub> b<sub>u</sub>ilt b<sub>y</sub> <sub>s</sub>im<sub>u</sub>l<sub>a</sub>tin<sub>g</sub> <sub>eve</sub>nt <sub>s</sub>tr<sub>ea</sub>m<sub>s</sub> on DiLiGenT [Shi et al. 2016]. DiLiGenT-Sim contains 9 test scenes.

F<sub>or rea</sub>l<sub>-wor</sub>ld <sub>eva</sub>l<sub>ua</sub>ti<sub>on, we use</sub> t<sub>wo rea</sub>l d<sub>a</sub>t<sub>a sources.</sub> Fi<sub>rs</sub>t<sub>,</sub> we include the three real EventPS scenes [Yu et al. 2024]. Second, we capture t<sup>h</sup>ree 3D-printe<sup>d</sup> o<sup>b</sup>jects wit<sup>h</sup> our prototype system, includin<sub>g</sub> Bunn<sub>y,</sub> Rectan<sub>g</sub>le<sub>,</sub> and Flower. We com<sub>p</sub>ute GT normals f<sub>rom</sub> th<sub>e correspon</sub>di<sub>ng</sub> 3D CAD <sub>mo</sub>d<sub>e</sub>l<sub>s an</sub>d <sub>a</sub>li<sub>gn</sub> th<sub>em</sub> t<sub>o</sub> th<sub>e</sub> sensor observation usin<sub>g</sub> the event accumulation ima<sub>g</sub>e [Kitazawa et al. 2025]. The ali<sub>g</sub>nment is <sub>p</sub>erformed with MeshLab’s mutual information re<sub>g</sub>istration filter [Ci<sub>g</sub>noni et al. 2008], followin<sub>g</sub> the method in [Shi et al. 2016].

## 4.3 Quantitative Evaluation

T<sub>a</sub>bl<sub>e</sub> 1 r<sub>epo</sub>rt<sub>s</sub> th<sub>e</sub> MAE <sub>o</sub>n DiLiG<sub>e</sub>nT-Sim<sub>.</sub> F<sub>o</sub>r b<sub>ase</sub>lin<sub>es, we use</sub> the <sub>pu</sub>blished E<sub>v</sub>entUPS and E<sub>v</sub>entPS res<sub>u</sub>lts on the same scene set. E<sub>ve</sub>ntUPS h<sub>as</sub> n<sub>o pu</sub>bli<sub>c co</sub>d<sub>e, a</sub>nd PS-EIP i<sub>s o</sub>mitt<sub>e</sub>d b<sub>ecause</sub> it<sub>s co</sub>d<sub>e</sub> <sub>an</sub>d b<sub>enc</sub>h<sub>mar</sub>k d<sub>a</sub>t<sub>a</sub> <sub>are</sub> <sub>unava</sub>il<sub>a</sub>bl<sub>e.</sub> Th<sub>e</sub> di<sub>rec</sub>t <sub>so</sub>l<sub>ver</sub> i<sub>s</sub> <sub>a</sub> <sub>p</sub>h<sub>ys</sub>i<sub>cs-</sub> <sub>o</sub>nl<sub>y</sub> b<sub>ase</sub>lin<sub>e</sub> th<sub>a</sub>t <sub>op</sub>timiz<sub>es</sub> E<sub>q.</sub> 7<sub>.</sub> PIE-PS <sub>ac</sub>hi<sub>eves a</sub>n <sub>ave</sub>r<sub>age</sub> MAE <sub>o</sub>f 8<sub>.</sub>70<sup>◦</sup><sub>.</sub> It i<sub>s</sub> 5<sub>.</sub>44<sup>◦</sup> l<sub>ower</sub> th<sub>an</sub> th<sub>e</sub> b<sub>es</sub>t <sub>pu</sub>bli<sub>s</sub>h<sub>e</sub>d b<sub>ase</sub>li<sub>ne</sub> E<sub>ven</sub>tUPS <sub>an</sub>d 6<sub>.</sub>90<sup>◦</sup> l<sub>ower</sub> th<sub>an</sub> th<sub>e</sub> di<sub>rec</sub>t <sub>so</sub>l<sub>ver.</sub>

Table 2. PIE-Sim results by object family and lighting trajectory. MAE is reported in degrees.
<table><tr><td rowspan="2">Method</td><td colspan="3">Blobby</td><td colspan="3">Sculpture</td><td rowspan="2">Avg.</td></tr><tr><td>Circle</td><td>Hypo.</td><td>DiLiGenT</td><td>Circle</td><td>Hypo.</td><td>DiLiGenT</td></tr><tr><td>EventPS-FCN</td><td>28.83</td><td>16.83</td><td>21.73</td><td>31.63</td><td>26.43</td><td>29.59</td><td>25.84</td></tr><tr><td>EventPS-CNN</td><td>43.50</td><td>42.11</td><td>26.08</td><td>48.31</td><td>46.85</td><td>28.08</td><td>39.15</td></tr><tr><td>Direct Solver</td><td>27.08</td><td>17.24</td><td>13.02</td><td>26.60</td><td>24.68</td><td>22.67</td><td>21.88</td></tr><tr><td>PIE-PS</td><td>10.68</td><td>6.41</td><td>4.81</td><td>17.17</td><td>12.09</td><td>12.77</td><td>10.66</td></tr></table>

Ta<sup>bl</sup>e 2 reports a <sup>b</sup>rea<sup>kd</sup>own <sup>b</sup>y o<sup>b</sup>ject <sup>f</sup>ami<sup>l</sup>y an<sup>d</sup> <sup>l</sup>ig<sup>h</sup>ting trajectory on PIE-Sim. We eva<sup>l</sup>uate EventPS-FCN an<sup>d</sup> EventPS-CNN <sub>u</sub>sin<sub>g</sub> their released models. PIE-PS <sub>g</sub>i<sub>v</sub>es the lo<sub>w</sub>est error in e<sub>v</sub>er<sub>y</sub> <sub>g</sub>rou<sub>p</sub>. Its avera<sub>g</sub>e MAE is 10<sub>.</sub>66<sup>◦</sup><sub>,</sub> which is 11<sub>.</sub>22<sup>◦</sup> lower than the dir<sub>ec</sub>t <sub>so</sub>l<sub>ve</sub>r <sub>a</sub>nd 15<sub>.</sub>18<sup>◦</sup> l<sub>owe</sub>r th<sub>a</sub>n E<sub>ve</sub>ntPS-FCN<sub>.</sub>

Table 3. Real-world results on EventPS scenes and our captures. MAE is reported in degrees.
<table><tr><td rowspan="2">Method</td><td colspan="3">EventPS Real</td><td colspan="3">Captured</td><td rowspan="2">Avg.</td></tr><tr><td>Real-1</td><td>Real-2</td><td>Real-3</td><td>Bunny</td><td>Rectangle</td><td>Flower</td></tr><tr><td>EventPS-FCN</td><td>15.79</td><td>19.65</td><td>9.42</td><td>26.14</td><td>18.70</td><td>35.76</td><td>20.91</td></tr><tr><td>EventPS-CNN</td><td>17.65</td><td>20.16</td><td>14.85</td><td>48.78</td><td>48.11</td><td>43.14</td><td>32.12</td></tr><tr><td>Direct Solver</td><td>21.08</td><td>21.10</td><td>20.50</td><td>23.46</td><td>32.22</td><td>25.02</td><td>23.90</td></tr><tr><td>PIE-PS</td><td>6.51</td><td>6.09</td><td>5.92</td><td>12.69</td><td>11.52</td><td>19.57</td><td>10.38</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3 <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> MAE <sub>on s</sub>i<sub>x rea</sub>l<sub>-wor</sub>ld <sub>scenes w</sub>ith <sub>a</sub>li<sub>gne</sub>d <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th <sub>norma</sub>l<sub>s.</sub> W<sub>e compare</sub> th<sub>e re</sub>l<sub>ease</sub>d E<sub>ven</sub>tPS <sub>ne</sub>t<sub>wor</sub>k<sub>s,</sub> th<sub>e</sub> dir<sub>ec</sub>t <sub>so</sub>l<sub>ve</sub>r<sub>, a</sub>nd PIE-PS <sub>o</sub>n th<sub>e sa</sub>m<sub>e sce</sub>n<sub>es.</sub> PIE-PS h<sub>as</sub> th<sub>e</sub> b<sub>es</sub>t r<sub>esu</sub>lt <sub>o</sub>n <sub>a</sub>ll <sub>s</sub>ix <sub>sce</sub>n<sub>es.</sub> It<sub>s</sub> <sub>ave</sub>r<sub>age</sub> MAE i<sub>s</sub> 10<sub>.</sub>38<sup>◦</sup><sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> 10<sub>.</sub>53<sup>◦</sup> l<sub>ower</sub> th<sub>an</sub> E<sub>ven</sub>tPS<sub>-</sub>FCN <sub>an</sub>d 13<sub>.</sub>52<sup>◦</sup> l<sub>ower</sub> th<sub>an</sub> th<sub>e</sub> di<sub>rec</sub>t <sub>so</sub>l<sub>ver.</sub>

W<sub>e</sub> <sub>a</sub>dditi<sub>o</sub>n<sub>a</sub>ll<sub>y</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> PIE-PS <sub>u</sub>nd<sub>e</sub>r thr<sub>ee</sub> C<sub>e</sub>l<sub>e</sub>X SDK <sub>se</sub>n<sub>s</sub>iti<sub>v</sub>- it<sub>y</sub> settin<sub>g</sub>s<sub>,</sub> with detailed statistics re<sub>p</sub>orted in the su<sub>pp</sub>lementar<sub>y</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l<sub>.</sub>

## 4.4 Qualitative Evaluation

Fi<sub>g</sub>. 5 com<sub>p</sub>ares reconstruction <sub>q</sub>ualit<sub>y</sub> on both s<sub>y</sub>nthetic and real d<sub>a</sub>t<sub>a.</sub> E<sub>ve</sub>ntPS-FCN <sub>a</sub>nd E<sub>ve</sub>ntPS-CNN <sub>o</sub>ft<sub>e</sub>n r<sub>ecove</sub>r th<sub>e coa</sub>r<sub>se o</sub>bject s<sup>h</sup>ape, <sup>b</sup>ut t<sup>h</sup>eir error maps s<sup>h</sup>ow <sup>l</sup>arge <sup>l</sup>oca<sup>l</sup> errors, especia<sup>ll</sup>y on hi<sub>g</sub>h-curvature re<sub>g</sub>ions and real ca<sub>p</sub>tures. The direct solver <sub>g</sub>ives a <sub>use</sub>f<sub>u</sub>l <sub>p</sub>h<sub>ys</sub>i<sub>cs-on</sub>l<sub>y</sub> b<sub>ase</sub>li<sub>ne,</sub> b<sub>u</sub>t it <sub>pro</sub>d<sub>uces no</sub>i<sub>sy norma</sub>l<sub>s</sub> b<sub>ecause</sub> it so<sup>l</sup>ves eac<sup>h</sup> pixe<sup>l</sup> in<sup>d</sup>epen<sup>d</sup>ent<sup>l</sup>y an<sup>d</sup> cannot reject unre<sup>l</sup>ia<sup>bl</sup>e event <sub>pa</sub>ir<sub>s.</sub> In <sub>co</sub>ntr<sub>as</sub>t<sub>,</sub> PIE-PS <sub>g</sub>i<sub>ves s</sub>m<sub>oo</sub>th<sub>e</sub>r <sub>a</sub>nd m<sub>o</sub>r<sub>e accu</sub>r<sub>a</sub>t<sub>e</sub> n<sub>o</sub>rm<sub>a</sub>l<sub>s</sub> across the four settin<sub>g</sub>s. The im<sub>p</sub>rovement is visible in the darker <sub>e</sub>rr<sub>o</sub>r m<sub>aps a</sub>nd i<sub>s co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>nt <sub>w</sub>ith th<sub>e</sub> MAE <sub>va</sub>l<sub>ues s</sub>h<sub>ow</sub>n in th<sub>e</sub> fi<sub>gure.</sub>

Fi<sub>g</sub>. 6 vis<sub>u</sub>alizes the RGA reliabilit<sub>y</sub> wei<sub>g</sub>hts. The oran<sub>g</sub>e ma<sub>p</sub>s <sub>s</sub>h<sub>ow</sub> GT-id<sub>e</sub>ntifi<sub>e</sub>d <sub>u</sub>nr<sub>e</sub>li<sub>a</sub>bl<sub>e</sub> PIE<sub>s.</sub> Th<sub>e</sub> r<sub>e</sub>d m<sub>aps</sub> <sub>s</sub>h<sub>ow</sub> PIE<sub>s</sub> <sub>as</sub>- <sub>s</sub>i<sub>g</sub>n<sub>e</sub>d l<sub>ow</sub> RGA <sub>we</sub>i<sub>g</sub>ht<sub>s.</sub> Th<sub>e</sub> l<sub>ow</sub>-RGA r<sub>eg</sub>i<sub>o</sub>n<sub>s ove</sub>rl<sub>ap</sub> m<sub>o</sub>r<sub>e w</sub>ith GT<sub>-</sub>id<sub>en</sub>tifi<sub>e</sub>d <sub>unre</sub>li<sub>a</sub>bl<sub>e reg</sub>i<sub>ons</sub> th<sub>an w</sub>ith <sub>re</sub>li<sub>a</sub>bl<sub>e reg</sub>i<sub>ons.</sub> Thi<sub>s</sub> su<sub>pp</sub>orts usin<sub>g</sub> RGA as a soft reliabilit<sub>y</sub>-wei<sub>g</sub>htin<sub>g</sub> module.

## 5 Ablation Study

W<sub>e a</sub>n<sub>a</sub>l<sub>y</sub>z<sub>e</sub> PIEF <sub>a</sub>nd RGA <sub>o</sub>n PIE-Sim<sub>.</sub> F<sub>o</sub>r PIEF<sub>, we</sub> r<sub>e</sub>m<sub>ove</sub> th<sub>e</sub> si<sub>g</sub>ned event-rate in<sub>p</sub>ut from each PIE node<sub>,</sub> while kee<sub>p</sub>in<sub>g</sub> the li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> and the rest of the network <sub>u</sub>nchan<sub>g</sub>ed. For RGA<sub>,</sub> we remove learned reliabilit<sub>y</sub> wei<sub>g</sub>hts and <sub>p</sub>ool PIE observations dir<sub>ec</sub>tl<sub>y.</sub> All r<sub>esu</sub>lt<sub>s a</sub>r<sub>e</sub> r<sub>epo</sub>rt<sub>e</sub>d <sub>as</sub> MAE in d<sub>eg</sub>r<sub>ees.</sub>

Table 4. Ablation results on PIE-Sim. MAE is reported in degrees.
<table><tr><td>Metric</td><td>PIE-PS</td><td>w/o PIEF rate</td><td>w/o RGA</td></tr><tr><td>Avg. MAE</td><td>10.66</td><td>14.52</td><td>11.99</td></tr></table>

![](images/ec611c2b7b489925a175c1e3e9d9e7abccb8435f0bfcc39798692ee798d1f4bd.jpg)  
Fig. 7. RGA weight distribution on PIE-Sim. GT-identified unreliable PIEs receive lower weights on average, supporting RGA as a soft reliability weighting module.

T<sub>a</sub>bl<sub>e</sub> 4 <sub>summar</sub>i<sub>zes</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f th<sub>e ma</sub>i<sub>n</sub> PIE<sub>-</sub>PS <sub>com-</sub> <sub>po</sub>n<sub>e</sub>nt<sub>s.</sub> PIEF <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> m<sub>a</sub>in <sub>p</sub>h<sub>o</sub>t<sub>o</sub>m<sub>e</sub>tri<sub>c</sub> <sub>s</sub>i<sub>g</sub>n<sub>a</sub>l<sub>,</sub> <sub>a</sub>nd r<sub>e</sub>m<sub>ov</sub>in<sub>g</sub> it in<sub>c</sub>r<sub>eases</sub> th<sub>e ave</sub>r<sub>age</sub> MAE fr<sub>o</sub>m 10<sub>.</sub>66<sup>◦</sup> t<sub>o</sub> 14<sub>.</sub>52<sup>◦</sup><sub>.</sub> Thi<sub>s s</sub>h<sub>ows</sub> th<sub>a</sub>t li<sub>g</sub>ht-<sub>p</sub>air <sub>g</sub>eometr<sub>y</sub> alone is not enou<sub>g</sub>h<sub>,</sub> because it does not include the si<sub>g</sub>ned event res<sub>p</sub>onse caused b<sub>y</sub> the li<sub>g</sub>ht chan<sub>g</sub>e. Removin<sub>g</sub>

RGA <sub>a</sub>l<sub>so</sub> in<sub>c</sub>r<sub>eases</sub> th<sub>e ave</sub>r<sub>age</sub> MAE t<sub>o</sub> 11<sub>.</sub>99<sup>◦</sup><sub>, co</sub>nfirmin<sub>g</sub> th<sub>e va</sub>l<sub>ue</sub> of learned reliabilit<sub>y</sub> wei<sub>g</sub>htin<sub>g</sub>. Fi<sub>g</sub>. 7 f<sub>u</sub>rther shows that RGA tends to assi<sub>g</sub>n lower wei<sub>g</sub>hts to unreliable PIEs rather than wei<sub>g</sub>htin<sub>g</sub> obser<sub>v</sub>ations randoml<sub>y</sub>. This s<sub>upp</sub>orts its role as a learned reliabilit<sub>y</sub> wei<sub>g</sub>htin<sub>g</sub> module.

## 6 Conclusion and Discussion

In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> <sub>w</sub>e <sub>p</sub>resented PIE-PS<sub>,</sub> a <sub>p</sub>h<sub>y</sub>sics-embedded frame<sub>w</sub>ork that uses Ph<sub>y</sub>sical Irradiance Events (PIEs) and a Ph<sub>y</sub>sical Irradiance Event Gra<sub>p</sub>h Neural Network (PIE-GNN) to recover dense s<sub>u</sub>rface normals from e<sub>v</sub>ent streams. Reliabilit<sub>y</sub>-Gradin<sub>g</sub> Attention (RGA) assi<sub>g</sub>ns soft reliabilit<sub>y g</sub>rades to PIE observations before <sub>p</sub>ixel <sup>a</sup>gg<sup>re</sup>g<sup>ation.</sup>

O<sub>ur</sub> <sub>me</sub>th<sub>o</sub>d <sub>s</sub>till <sub>assumes</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d li<sub>g</sub>ht di<sub>rec</sub>ti<sub>ons.</sub> St<sub>rong</sub> <sub>spec-</sub> <sub>u</sub>l<sub>ar</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>s or comp</sub>l<sub>ex</sub> i<sub>n</sub>t<sub>er-re</sub>fl<sub>ec</sub>ti<sub>ons can re</sub>d<sub>uce</sub> th<sub>e re</sub>li<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f PIE <sub>o</sub>b<sub>serva</sub>ti<sub>ons an</sub>d d<sub>egra</sub>d<sub>e norma</sub>l <sub>es</sub>ti<sub>ma</sub>ti<sub>on.</sub> I<sub>n</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>,</sub> <sub>we w</sub>ill <sub>ex</sub>t<sub>en</sub>d th<sub>e mo</sub>d<sub>e</sub>l t<sub>o unca</sub>lib<sub>ra</sub>t<sub>e</sub>d li<sub>g</sub>hti<sub>ng an</sub>d <sub>more genera</sub>l <sub>re</sub>fl<sub>ec</sub>t<sub>ance.</sub>

## Acknowledgments

This work was su<sub>pp</sub>orted in <sub>p</sub>art b<sub>y</sub> the National Natural Science Foundation of China (Grant Nos. 62572212 and 62506084), the Science and Technolo<sub>gy</sub> Develo<sub>p</sub>ment Plan of Jilin Province (Grant No. 20260203049SF), and the Fundamental Research Funds for the Central Universities. This work was also su<sub>pp</sub>orted b<sub>y</sub> the Hi<sub>g</sub>h-Leve<sup>l</sup> Ta<sup>l</sup>ents Innovation Team Project o<sup>f</sup> t<sup>h</sup>e Guang<sup>d</sup>ong-Macao In-De<sub>p</sub>th Coo<sub>p</sub>eration Zone in Hen<sub>gq</sub>in (Grant No. 2630004018939).

## References

Gwan<sub>g</sub>bin Bae and Andrew J Davison. 2024. Rethinkin<sub>g</sub> inductive biases for surface normal estimation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9535–9545.

Seun<sub>g</sub>-Hwan Baek, Daniel S. Jeon, Xin Ton<sub>g</sub>, and Min H. Kim. 2018. Simultaneous Acquisition of Polarimetric SVBRDF and Normals. ACM Transactions on Graphics 37, 6, Article 268 (2018), 15 <sub>p</sub>a<sub>g</sub>es. doi:10.1145/3272127.3275018

Manmohan Chandraker<sub>,</sub> Sameer A<sub>g</sub>arwal<sub>,</sub> and David Krie<sub>g</sub>man. 2007. Shadowc<sub>u</sub>ts: Photometric stereo with shadows. In 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 1–8.

G<sub>ua</sub>n<sub>y</sub>in<sub>g</sub> Ch<sub>e</sub>n<sub>,</sub> K<sub>a</sub>i H<sub>a</sub>n<sub>,</sub> B<sub>o</sub>xin Shi<sub>,</sub> Y<sub>asuyu</sub>ki M<sub>a</sub>t<sub>sus</sub>hit<sub>a, a</sub>nd K<sub>wa</sub>n-Y<sub>ee</sub> K W<sub>o</sub>n<sub>g.</sub> 2019<sub>.</sub> Self-calibrating deep photometric stereo networks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 8739–8747.

P<sub>ao</sub>l<sub>o</sub> Ci<sub>g</sub>n<sub>o</sub>ni<sub>,</sub> M<sub>a</sub>r<sub>co</sub> C<sub>a</sub>lli<sub>e</sub>ri<sub>,</sub> M<sub>ass</sub>imili<sub>a</sub>n<sub>o</sub> C<sub>o</sub>r<sub>s</sub>ini<sub>,</sub> M<sub>a</sub>tt<sub>eo</sub> D<sub>e</sub>ll<sub>ep</sub>i<sub>a</sub>n<sub>e,</sub> F<sub>a</sub>bi<sub>o</sub> G<sub>a</sub>n<sub>ov</sub> elli<sub>,</sub> Guido Ranzu<sub>g</sub>lia<sub>,</sub> et al. 2008. Meshlab: an o<sub>p</sub>en-source mesh <sub>p</sub>rocessin<sub>g</sub> tool.. In Eurographics Italian chapter conference, Vol. 2008. Salerno, 129–136.

Chaoran Fen<sub>g</sub>, Zhen<sub>y</sub>u Tan<sub>g</sub>, Wan<sub>g</sub>bo Yu, Yatian Pan<sub>g</sub>, Yian Zhao, Jianbin Zhao, Li Y<sub>uan, an</sub>d Y<sub>on</sub> h<sub>on</sub> Ti<sub>an.</sub> 2025<sub>a.</sub> E<sub>-</sub>4d <sub>s:</sub> Hi h<sub>-</sub>fid<sub>e</sub>lit d <sub>nam</sub>i<sub>c recons</sub>t<sub>ruc</sub>ti<sub>on</sub> from the multi-view event cameras. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 7356–7365.

Chaoran Fen<sub>g</sub>, Wan<sub>g</sub>bo Yu, Xinhua Chen<sub>g</sub>, Zhen<sub>y</sub>u Tan<sub>g</sub>, Junwu Zhan<sub>g</sub>, Li Yuan, and Y<sub>o</sub>n h<sub>o</sub>n Ti<sub>a</sub>n<sub>.</sub> 2025b<sub>.</sub> AE-N<sub>e</sub>RF: A<sub>u</sub> m<sub>e</sub>ntin E<sub>ve</sub>nt-B<sub>ase</sub>d N<sub>eu</sub>r<sub>a</sub>l R<sub>a</sub>di<sub>a</sub>n<sub>ce</sub> Fi<sub>e</sub>ld<sub>s</sub> for Non-ideal Conditions and Larger Scenes. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 2924–2932.

G<sub>u</sub>ill<sub>e</sub>rm<sub>o</sub> G<sub>a</sub>ll<sub>ego,</sub> T<sub>o</sub>bi D<sub>e</sub>lbrü<sub>c</sub>k<sub>,</sub> G<sub>a</sub>rri<sub>c</sub>k Or<sub>c</sub>h<sub>a</sub>rd<sub>,</sub> Chi<sub>a</sub>r<sub>a</sub> B<sub>a</sub>rt<sub>o</sub>l<sub>o</sub>zzi<sub>,</sub> Bri<sub>a</sub>n T<sub>a</sub>b<sub>a,</sub> Andrea Censi, Stefan Leutene<sub>gg</sub>er, Andrew J Davison, Jör<sub>g</sub> Conradt, Kostas Daniilidis, et al. 2020. Event-based vision: A survey. IEEE transactions on pattern analysis and machine intelligence 44, 1 (2020), 154–180.

Wenhan<sub>g</sub> Ge, Jiantao Lin, Guibao Shen, Jiawei Fen<sub>g</sub>, Tao Hu, Xinli Xu, and Yin<sub>g</sub> Con<sub>g</sub> Chen. 2025. Prm: Photometric stereo based lar<sub>g</sub>e reconstruction model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 25009– 25018.

Han<sub>q</sub>ian Han, Jianin<sub>g</sub> Li, Hen<sub>g</sub>lu Wei, and Xian<sub>gy</sub>an<sub>g</sub> Ji. 2024. Event-3d<sub>g</sub>s: Event-based 3d reconstruction using 3d gaussian splatting. Advances in Neural Information Processing Systems 37 (2024), 128139–128159.

Inseun<sub>g</sub> Hwan<sub>g</sub>, Daniel S. Jeon, Adolfo Muñoz, Die<sub>g</sub>o Gutierrez, Xin Ton<sub>g</sub>, and Min H. Kim<sub>.</sub> 2022<sub>.</sub> S<sub>pa</sub>r<sub>se</sub> Elli<sub>pso</sub>m<sub>e</sub>tr<sub>y</sub>: P<sub>o</sub>rt<sub>a</sub>bl<sub>e</sub> A<sub>cqu</sub>i<sub>s</sub>iti<sub>o</sub>n <sub>o</sub>f P<sub>o</sub>l<sub>a</sub>rim<sub>e</sub>tri<sub>c</sub> SVBRDF <sub>a</sub>nd Shape with Unstructured Flash Photography. ACM Transactions on Graphics 41, 4, Article 145 (2022), 14 <sub>p</sub>a<sub>g</sub>es. doi:10.1145/3528223.3530075

S<sub>a</sub>t<sub>os</sub>hi Ik<sub>e</sub>h<sub>a</sub>t<sub>a,</sub> D<sub>av</sub>id Wi<sub>p</sub>f<sub>,</sub> Y<sub>asuyu</sub>ki M<sub>a</sub>t<sub>sus</sub>hit<sub>a, a</sub>nd Ki<sub>yo</sub>h<sub>a</sub>r<sub>u</sub> Aiz<sub>awa.</sub> 2012<sub>.</sub> R<sub>o</sub>b<sub>us</sub>t photometric stereo using sparse regression. In 2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 318–325.

Micah K Johnson and Edward H Adelson. 2011. Sha<sub>p</sub>e estimation in natural illumination. In CVPR 2011. IEEE, 2553–2560.

K<sub>azuma</sub> Kit<sub>azawa,</sub> T<sub>a</sub>k<sub>a</sub>hit<sub>o</sub> A<sub>o</sub>t<sub>o,</sub> S<sub>a</sub>t<sub>os</sub>hi Ik<sub>e</sub>h<sub>a</sub>t<sub>a, an</sub>d T<sub>suyos</sub>hi T<sub>a</sub>k<sub>a</sub>t<sub>an</sub>i<sub>.</sub> 2025<sub>.</sub> PS<sub>-</sub>EIP<sub>:</sub> Robust Photometric Stereo Based on Event Interval Profile. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 6241–6251.

Junxuan Li and Hon<sub>g</sub>don<sub>g</sub> Li. 2022. Self-calibratin<sub>g p</sub>hotometric stereo b<sub>y</sub> neural inverse rendering. In European Conference on Computer Vision. Springer, 166–183.

Zongrui Li, Zhan Lu, Haojie Yan, Boxin Shi, Gang Pan, Qian Zheng, and Xudong Jian<sub>g</sub>. 2024. S<sub>p</sub>in-u<sub>p</sub>: S<sub>p</sub>in li<sub>g</sub>ht for natural li<sub>g</sub>ht uncalibrated <sub>p</sub>hotometric stereo. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11905–11914.

Jinxiu Lian<sub>g</sub>, Bohan Yu, Si<sub>q</sub>i Yan<sub>g</sub>, Haotian Zhuan<sub>g</sub>, Jieji Ren, Pei<sub>q</sub>i Duan, and Boxin Shi<sub>.</sub> 2025<sub>.</sub> E<sub>ven</sub>tUPS<sub>:</sub> U<sub>nca</sub>lib<sub>ra</sub>t<sub>e</sub>d Ph<sub>o</sub>t<sub>ome</sub>t<sub>r</sub>i<sub>c</sub> St<sub>ereo</sub> U<sub>s</sub>i<sub>n an</sub> E<sub>ven</sub>t C<sub>amera.</sub> I<sub>n</sub> Proceedings of the IEEE/CVF International Conference on Computer Vision. 7516–7525.

Daniel Lich<sub>y</sub>, Soum<sub>y</sub>adi<sub>p</sub> Sen<sub>g</sub>u<sub>p</sub>ta, and David W Jacobs. 2022. Fast li<sub>g</sub>ht-wei<sub>g</sub>ht nearfield photometric stereo. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12612–12621.

F<sub>o</sub>ti<sub>os</sub> L<sub>ogo</sub>th<sub>e</sub>ti<sub>s,</sub> I<sub>g</sub>n<sub>as</sub> B<sub>u</sub>d<sub>vy</sub>ti<sub>s,</sub> R<sub>o</sub>b<sub>e</sub>rt<sub>o</sub> M<sub>ecca,</sub> <sub>a</sub>nd R<sub>o</sub>b<sub>e</sub>rt<sub>o</sub> Ci<sub>po</sub>ll<sub>a.</sub> 2021<sub>.</sub> Px net: Sim<sub>p</sub>le and eficient <sub>p</sub>ixel-wise trainin<sub>g</sub> of <sub>p</sub>hotometric stereo networks. In Proceedings ofthe IEEE/CVFinternational conference on computervision. 12757–12766.

W<sub>e</sub>n<sub>g</sub> F<sub>e</sub>i L<sub>ow</sub> <sub>a</sub>nd Gim H<sub>ee</sub> L<sub>ee.</sub> 2023<sub>.</sub> R<sub>o</sub>b<sub>us</sub>t <sub>e</sub>-n<sub>e</sub>rf: N<sub>e</sub>rf fr<sub>o</sub>m <sub>spa</sub>r<sub>se</sub> & n<sub>o</sub>i<sub>sy</sub> <sub>eve</sub>nt<sub>s</sub> under non-uniform motion. In Proceedings of the IEEE/CVF International Conference

on Computer Vision. 18335–18346.

Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, and Luc Van Gool. 2023. Deformable neural radiance fields using rgb and event cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3590–3600.

Daisu<sup>k</sup>e Miyaza<sup>k</sup>i, Kenji Hara, an<sup>d</sup> Katsus<sup>h</sup>i I<sup>k</sup>euc<sup>h</sup>i. 2010. Me<sup>d</sup>ian p<sup>h</sup>otometric stereo as applied to the segonko tumulus and museum objects. International Journal of Computer Vision 86, 2 (2010), 229–242.

Manasi Mu<sub>g</sub>likar<sub>,</sub> Guillermo Galle<sub>g</sub>o<sub>,</sub> and Davide Scaramuzza. 2021. Esl: Event-based structured light. In 2021 International Conference on 3D Vision (3DV). IEEE, 1165– 1174.

Junkai Niu, Sheng Zhong, Xiuyuan Lu, Shaojie Shen, Guillermo Gallego, and Yi Zhou. 2025. Esvo2: Direct visual-inertial odometry with stereo event cameras. IEEE Transactions on Robotics (2025).

Xiaojuan Qi, Renjie Liao, Zhengzhe Liu, Raquel Urtasun, and Jiaya Jia. 2018. Geonet: Geometric neural network for joint depth and surface normal estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 283–291.

H<sub>e</sub>nri R<sub>e</sub>b<sub>ecq,</sub> G<sub>u</sub>ill<sub>e</sub>rm<sub>o</sub> G<sub>a</sub>ll<sub>ego,</sub> Eli<sub>as</sub> M<sub>uegg</sub>l<sub>e</sub>r<sub>,</sub> <sub>a</sub>nd D<sub>av</sub>id<sub>e</sub> S<sub>ca</sub>r<sub>a</sub>m<sub>u</sub>zz<sub>a.</sub> 2018<sub>a.</sub> EMVS: E<sub>ve</sub>nt-b<sub>ase</sub>d m<sub>u</sub>lti-<sub>v</sub>i<sub>ew s</sub>t<sub>e</sub>r<sub>eo</sub>—3D r<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>w</sub>ith <sub>a</sub>n <sub>eve</sub>nt <sub>ca</sub>m<sub>e</sub>r<sub>a</sub> in r<sub>ea</sub>l-tim<sub>e.</sub> International Journal ofComputer Vision 126, 12 (2018), 1394–1414.

Henri Rebec<sub>q,</sub> Daniel Gehri<sub>g,</sub> and Da<sub>v</sub>ide Scaram<sub>u</sub>zza. 2018b. Esim: an o<sub>p</sub>en e<sub>v</sub>ent camera simulator. In Conference on robot learning. PMLR, 969–982.

H<sub>e</sub>nri R<sub>e</sub>b<sub>ecq,</sub> R<sub>e</sub>né R<sub>a</sub>nftl<sub>,</sub> Vl<sub>a</sub>dl<sub>e</sub>n K<sub>o</sub>lt<sub>u</sub>n<sub>, a</sub>nd D<sub>av</sub>id<sub>e</sub> S<sub>ca</sub>r<sub>a</sub>m<sub>u</sub>zz<sub>a.</sub> 2019<sub>.</sub> E<sub>ve</sub>nt<sub>s</sub>-t<sub>o</sub>- video: Bringing modern computer vision to event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3857–3866.

Vikt<sub>or</sub> R<sub>u</sub>d<sub>nev,</sub> M<sub>o</sub>h<sub>ame</sub>d El<sub>g</sub>h<sub>ar</sub>ib<sub>,</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> Th<sub>eo</sub>b<sub>a</sub>lt<sub>, an</sub>d Vl<sub>a</sub>di<sub>s</sub>l<sub>av</sub> G<sub>o</sub>l<sub>yan</sub>ik<sub>.</sub> 2023<sub>.</sub> Eventnerf: Neural radiance fields from a single colour event camera. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4992–5002.

Wonjeong Ryoo, Gi<sup>l</sup>joo Nam, Jae-Sang Hyun, an<sup>d</sup> Sangpi<sup>l</sup> Kim. 2023. Event <sup>f</sup>usion photometric stereo network. Neural Networks (aug 2023). doi:10.1016/j.neunet.2023. 08.009

Hiroa<sup>k</sup>i Santo, Masa<sup>k</sup>i Samejima, Yusu<sup>k</sup>e Sugano, Boxin S<sup>h</sup>i, an<sup>d</sup> Yasuyu<sup>k</sup>i Matsus<sup>h</sup>ita. 2017. Deep photometric stereo network. In Proceedings ofthe IEEE international conference on computer vision workshops. 501–509.

Hiroa<sup>k</sup>i Santo, Masa<sup>k</sup>i Samejima, Yusu<sup>k</sup>e Sugano, Boxin S<sup>h</sup>i, an<sup>d</sup> Yasuyu<sup>k</sup>i Matsus<sup>h</sup>ita. 2020<sub>.</sub> D<sub>eep p</sub>h<sub>o</sub>t<sub>ome</sub>t<sub>r</sub>i<sub>c s</sub>t<sub>ereo ne</sub>t<sub>wor</sub>k<sub>s</sub> f<sub>or</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>ng sur</sub>f<sub>ace norma</sub>l <sub>an</sub>d reflectances. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 1 (2020), 114–128.

B<sub>o</sub>xin Shi<sub>,</sub> Zh<sub>e</sub> W<sub>u,</sub> Zhi<sub>pe</sub>n<sub>g</sub> M<sub>o,</sub> Din<sub>g</sub>l<sub>o</sub>n<sub>g</sub> D<sub>ua</sub>n<sub>,</sub> S<sub>a</sub>i-Kit Y<sub>eu</sub>n<sub>g, a</sub>nd Pin<sub>g</sub> T<sub>a</sub>n<sub>.</sub> 2016<sub>.</sub> A b<sub>enc</sub>h<sub>mar</sub>k d<sub>a</sub>t<sub>ase</sub>t <sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>or non-</sub>l<sub>am</sub>b<sub>er</sub>ti<sub>an an</sub>d <sub>unca</sub>lib<sub>ra</sub>t<sub>e</sub>d h<sub>o</sub>t<sub>ome</sub>t<sub>r</sub>i<sub>c</sub> stereo. In Proceedings ofthe IEEE conference on computervision andpattern recognition. 3707–3716.

Zi<sub>yu</sub>n W<sub>a</sub>n<sub>g,</sub> K<sub>e</sub>nn<sub>e</sub>th Ch<sub>a</sub>n<sub>ey, a</sub>nd K<sub>os</sub>t<sub>as</sub> D<sub>a</sub>niilidi<sub>s.</sub> 2022<sub>.</sub> E<sub>vac</sub>3d: Fr<sub>o</sub>m <sub>eve</sub>nt-b<sub>ase</sub>d apparent contours to 3d models via continuous visual hulls. In European conference on computer vision. Springer, 284–299.

Oli<sub>v</sub>i<sub>a</sub> Wil<sub>es</sub> <sub>a</sub>nd Andr<sub>ew</sub> Zi<sub>sse</sub>rm<sub>a</sub>n<sub>.</sub> 2017<sub>.</sub> Siln<sub>e</sub>t: Sin<sub>g</sub>l<sub>e</sub>-<sub>a</sub>nd m<sub>u</sub>lti-<sub>v</sub>i<sub>ew</sub> r<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n by learning from silhouettes. arXiv preprint arXiv:1711.07888 (2017).

Robert J Woodham. 1980. Photometric method for determinin<sub>g</sub> surface orientation from multiple images. Optical engineering 19, 1 (1980), 139–144.

L<sub>u</sub>n W<sub>u,</sub> Ar<sub>v</sub>ind G<sub>a</sub>n<sub>es</sub>h<sub>,</sub> B<sub>o</sub>xin Shi<sub>,</sub> Y<sub>asuyu</sub>ki M<sub>a</sub>t<sub>sus</sub>hit<sub>a,</sub> Y<sub>o</sub>n<sub>g</sub>ti<sub>a</sub>n W<sub>a</sub>n<sub>g,</sub> <sub>a</sub>nd Yi M<sub>a.</sub> 2010. Robust <sub>p</sub>hotometric stereo via low-rank matrix com<sub>p</sub>letion and recover<sub>y</sub>. In Asian conference on computer vision. Springer, 703–717.

Ch<sub>u</sub>anzhi X<sub>u,</sub> Haoxian Zho<sub>u,</sub> Lan<sub>gy</sub>i Chen<sub>,</sub> Haodon<sub>g</sub> Chen<sub>,</sub> Yin<sub>g</sub> Zho<sub>u,</sub> Vera Y<sub>u</sub>k Yin<sub>g</sub> Chung, Qiang Qu, and Weidong Cai. 2025. A Survey of 3D Reconstruction with Event Cameras. arXiv preprint arXiv:2505.08438 (2025).

Bohan Yu, Jieji Ren, Jin Han, Feishi Wan<sub>g</sub>, Jinxiu Lian<sub>g</sub>, and Boxin Shi. 2024. Event<sub>p</sub>s: Real-time photometric stereo using an event camera. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9602–9611.

Qian Zhen<sub>g</sub>, Yimin<sub>g</sub> Jia, Boxin Shi, Xudon<sub>g</sub> Jian<sub>g</sub>, Lin<sub>g</sub>-Yu Duan, and Alex C Kot. 2019. SPLINE-Net: S<sub>p</sub>arse <sub>p</sub>hotometric stereo throu<sub>g</sub>h li<sub>g</sub>htin<sub>g</sub> inter<sub>p</sub>olation and normal estimation networks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 8549–8558.

Qian Zheng, Boxin Shi, and Gang Pan. 2020. Summary study ofdata-driven photometric stereo methods. Virtual Reality & Intelligent Hardware 2, 3 (2020), 213–221.