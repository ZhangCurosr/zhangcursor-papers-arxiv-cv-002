# LEAP: Accelerating ODE-based Difusion Sampling with Optimal Linear Leaps

H<sub>aowe</sub>i Zh<sub>e</sub>n<sub>g</sub><sup>∗†</sup> z<sup>h</sup>wei7107@zju.e<sup>d</sup>u.cn Z<sup>h</sup>ejiang University Han<sub>g</sub>zho<sub>u,</sub> China

Zh<sub>enyu</sub> Zh<sub>ou</sub><sup>∗†</sup> z<sup>h</sup>yz<sup>h</sup>ou@zju.e<sup>d</sup>u.cn Z<sup>h</sup>ejiang University Han<sub>g</sub>zho<sub>u,</sub> China

Can Wan<sub>g</sub><sup>∗†</sup> wcan@zju.e<sup>d</sup>u.cn Z<sup>h</sup>ejiang University Han<sub>g</sub>zho<sub>u,</sub> China

Jiawei Chen<sup>∗†</sup>   
s<sup>l</sup>eepy<sup>h</sup>unt@zju.e<sup>d</sup>u.cn   
Z<sup>h</sup>ejiang University   
Han<sub>g</sub>zho<sub>u,</sub> China

D<sub>e</sub>f<sub>ang</sub> Ch<sub>en</sub><sup>‡</sup> d<sub>e</sub>f<sub>c</sub>h<sub>ern@</sub>b<sub>er</sub>k<sub>e</sub>l<sub>ey.e</sub>d<sub>u</sub> Universit<sub>y</sub> of California<sub>,</sub> Berkele<sub>y</sub> C<sub>a</sub>lif<sub>o</sub>rni<sub>a,</sub> Unit<sub>e</sub>d St<sub>a</sub>t<sub>es</sub>

## Abstract

Difusion models <sub>p</sub>rimaril<sub>y</sub> rel<sub>y</sub> on <sub>g</sub>eneral-<sub>p</sub>ur<sub>p</sub>ose ODE-based <sub>numer</sub>i<sub>ca</sub>l <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng-</sub>f<sub>ree</sub> <sub>acce</sub>l<sub>era</sub>t<sub>e</sub>d <sub>samp</sub>li<sub>ng,</sub> b<sub>u</sub>t <sub>per-</sub> formance de<sub>g</sub>rades with coarse time discretization. In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we propose optimal linear leaps (LEAP), a specialized method grounded in t<sup>h</sup>eoretica<sup>l</sup> an<sup>d</sup> empirica<sup>l</sup> investigations o<sup>f</sup> samp<sup>l</sup>ing trajectory geometry. Bui<sup>l</sup>t upon <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity an<sup>d</sup> cross-trajectory con-<sub>s</sub>i<sub>s</sub>t<sub>e</sub>n<sub>cy,</sub> LEAP f<sub>o</sub>rm<sub>u</sub>l<sub>a</sub>t<sub>es</sub> <sub>a</sub> <sub>supe</sub>r<sub>v</sub>i<sub>se</sub>d l<sub>eas</sub>t-<sub>squa</sub>r<sub>es</sub> <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n pro<sup>bl</sup>em wit<sup>h</sup>in t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup> trajectory su<sup>b</sup>space, ac<sup>h</sup>ieving a precise ap-<sub>p</sub>roximation of future states without model evaluations. Extensive ex<sub>p</sub>eriments demonstrate that LEAP o<sub>u</sub>t<sub>p</sub>erforms <sub>p</sub>revio<sub>u</sub>s methods across diverse modalities, includin<sub>g</sub> text-to-ima<sub>g</sub>e (Stable Difusion 3.5, FLUX.1-dev) and ima<sub>g</sub>e-to-3D (TRELLIS.2) <sub>g</sub>eneration. Furthermore<sub>,</sub> LEAP can be seamlessl<sub>y</sub> inte<sub>g</sub>rated into existin<sub>g</sub> solver-based <sub>me</sub>th<sub>o</sub>d<sub>s, o</sub>f<sub>er</sub>i<sub>ng a p</sub>l<sub>ug-an</sub>d<sub>-p</sub>l<sub>ay en</sub>h<sub>ancemen</sub>t f<sub>or mo</sub>d<sub>ern</sub> dif<sub>u-</sub> sion mo<sup>d</sup>e<sup>l</sup>s. Co<sup>d</sup>e is avai<sup>l</sup>a<sup>bl</sup>e at <sup>h</sup>ttps://git<sup>h</sup>u<sup>b</sup>.com/zju-pi/LEAP.

## CCS Concepts

• Computing methodologies → Computer vision.

## Keywords

Dif<sub>us</sub>i<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>l<sub>s,</sub> Tr<sub>a</sub>inin<sub>g</sub>-fr<sub>ee</sub> A<sub>cce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> S<sub>o</sub>l<sub>ve</sub>r-b<sub>ase</sub>d

## ACM Reference Format:

Haowei Zhen<sub>g</sub>, Zhen<sub>y</sub>u Zhou, Jiawei Chen, Can Wan<sub>g</sub>, and Defan<sub>g</sub> Chen. 2026<sub>.</sub> LEAP: A<sub>cce</sub>l<sub>e</sub>r<sub>a</sub>tin<sub>g</sub> ODE-b<sub>ase</sub>d Dif<sub>us</sub>i<sub>o</sub>n S<sub>a</sub>m<sub>p</sub>lin<sub>g</sub> <sub>w</sub>ith O<sub>p</sub>tim<sub>a</sub>l Linear Leaps. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 18 <sub>p</sub>a<sub>g</sub>es. htt<sub>p</sub>s://doi.or<sub>g</sub>/10.1145/3767308.3836106

## 1 Introduction

Difusion-based <sub>g</sub>enerative models [9, 29, 32, 33] have achieved un<sub>p</sub>recedented success in multimedia content s<sub>y</sub>nthesis<sub>,</sub> ran<sub>g</sub>in<sub>g</sub> from ima<sub>g</sub>e [24, 25, 27], video [3, 10], audio [6, 13], and 3D [11, 41]. Des<sub>p</sub>ite their remarkable <sub>g</sub>eneration <sub>q</sub>ualit<sub>y,</sub> the slow sam<sub>p</sub>lin<sub>g</sub> <sub>p</sub>rocess im<sub>p</sub>oses a critical bottleneck for latenc<sub>y</sub>-sensitive a<sub>pp</sub>lications.

T<sub>o a</sub>dd<sub>ress</sub> thi<sub>s</sub> i<sub>ssue, numerous acce</sub>l<sub>era</sub>ti<sub>on</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ques</sub> h<sub>ave</sub> been <sub>p</sub>ro<sub>p</sub>osed. Trainin<sub>g</sub>-based a<sub>pp</sub>roaches distill a <sub>p</sub>re-trained multi-ste<sub>p</sub> model into an extremel<sub>y</sub> few-ste<sub>p</sub> student model [12, 20, 28, 31]. However, distillin<sub>g</sub> modern difusion models with billions of <sub>p</sub>arameters (e.<sub>g</sub>., 8B Stable Difusion 3.5 [8] and 12B FLUX [2]) is <sub>c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng</sub> <sub>an</sub>d i<sub>n</sub>fl<sub>ex</sub>ibl<sub>e</sub> t<sub>o</sub> <sub>accommo</sub>d<sub>a</sub>t<sub>e</sub> <sub>commun</sub>it<sub>y-</sub>t<sub>une</sub>d <sub>var</sub>i<sub>-</sub> ants and LoRA modules. In contrast<sub>,</sub> trainin<sub>g</sub>-free methods re<sub>q</sub>uire minimal <sub>p</sub>arameter u<sub>p</sub>dates<sub>,</sub> oferin<sub>g</sub> su<sub>p</sub>erior <sub>p</sub>lu<sub>g</sub>-and-<sub>p</sub>la<sub>y</sub> flexibility. Recent progress includes (i) solver-based method for calculating ordinary diferential equations (ODEs) [16, 18, 30, 46, 49], (ii) optimized timestep scheduling [4, 26, 43], and (iii) feature caching [14, 21, 39, 44]. Existin<sub>g</sub> hi<sub>g</sub>h-order multi-ste<sub>p</sub> solvers [16, 18, 46] estim<sub>a</sub>t<sub>e</sub> f<sub>u</sub>t<sub>u</sub>r<sub>e s</sub>t<sub>a</sub>t<sub>es v</sub>i<sub>a a</sub> lin<sub>ea</sub>r <sub>co</sub>mbin<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f hi<sub>s</sub>t<sub>o</sub>ri<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s are</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom numer</sub>i<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s, assum-</sub> in<sub>g</sub> suficientl<sub>y</sub> small and near-uniform ste<sub>p</sub> sizes. However<sub>,</sub> <sub>p</sub>ractical acceleration t<sub>yp</sub>icall<sub>y</sub> em<sub>p</sub>lo<sub>y</sub>s lar<sub>g</sub>e and non-uniform ste<sub>p</sub> sizes<sub>,</sub> makin<sub>g</sub> their assum<sub>p</sub>tions and associated conver<sub>g</sub>ence <sub>g</sub>uarantees less reliable. Concurrentl<sub>y,</sub> emer<sub>g</sub>in<sub>g</sub> research on the <sub>g</sub>eometric re<sub>g</sub>ularit<sub>y</sub> of difusion trajectories [4, 5, 36, 38] has uncovered intriguing empirica<sup>l</sup> patterns, suc<sup>h</sup> as cross-trajectory consistency <sub>an</sub>d l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>concen</sub>t<sub>ra</sub>ti<sub>on.</sub> Whil<sub>e prev</sub>i<sub>ous s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ave</sub> <sub>q</sub>ualitativel<sub>y</sub> observed these re<sub>g</sub>ularities<sub>,</sub> a ri<sub>g</sub>orous mathematical f<sub>oun</sub>d<sub>a</sub>ti<sub>on rema</sub>i<sub>ns</sub> l<sub>ar e</sub>l <sub>a</sub>b<sub>sen</sub>t<sub>.</sub> T<sub>o move</sub> b<sub>e on</sub>d h<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub> d<sub>e-</sub> si<sub>g</sub>ns toward a <sub>p</sub>rinci<sub>p</sub>led a<sub>pp</sub>roach<sub>,</sub> a dee<sub>p</sub>er and more ri<sub>g</sub>orous in<sub>v</sub>esti<sub>g</sub>ation into the <sub>u</sub>nderl<sub>y</sub>in<sub>g</sub> mechanics is essential.

To brid<sub>g</sub>e this <sub>g</sub>a<sub>p,</sub> we conduct a com<sub>p</sub>rehensive theoretical analysis o<sup>f</sup> t<sup>h</sup>e geometric properties in<sup>h</sup>erent in <sup>d</sup>i<sup>f</sup>usion trajectories, <sub>y</sub>ieldin<sub>g</sub> the followin<sub>g</sub> two theoretical insi<sub>g</sub>hts. Firstl<sub>y,</sub> we establish a t<sup>h</sup>eoretica<sup>l b</sup>oun<sup>d</sup> on t<sup>h</sup>e a<sup>l</sup>ignment error <sup>b</sup>etween trajectories a<sup>f</sup>- ter ort<sup>h</sup>ogona<sup>l</sup> rotationa<sup>l</sup> a<sup>l</sup>ignment, revea<sup>l</sup>ing t<sup>h</sup>e cross-trajectory <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> <sub>o</sub>f dif<sub>us</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> S<sub>econ</sub>dl<sub>y,</sub> <sub>we</sub> <sub>r</sub>i<sub>gorous</sub>l<sub>y</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er-</sub> ize t<sup>h</sup>e intrinsic <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity o<sup>f d</sup>i<sup>f</sup>usion samp<sup>l</sup>ing trajecto-<sub>r</sub>i<sub>es,</sub> <sub>prov</sub>i<sub>ng</sub> th<sub>a</sub>t i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>pa</sub>th<sub>s</sub> <sub>are</sub> <sub>con</sub>fi<sub>ne</sub>d t<sub>o</sub> <sub>a</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l subs<sub>p</sub>ace. Our anal<sub>y</sub>sis reveals that des<sub>p</sub>ite diferent noise initializations<sub>,</sub> these <sub>g</sub>enerative <sub>p</sub>aths share consistent low-dimensional d<sub>y</sub>namics.

Levera<sub>g</sub>in<sub>g</sub> these two mathematicall<sub>y g</sub>rounded <sub>p</sub>riors<sub>,</sub> we <sub>p</sub>ropose optimal linear leaps (LEAP), a training-free, solver-based ac-<sub>ce</sub>l<sub>era</sub>ti<sub>on</sub> f<sub>ramewor</sub>k<sub>.</sub> I<sub>n con</sub>t<sub>ras</sub>t t<sub>o</sub> t<sub>ra</sub>diti<sub>ona</sub>l <sub>numer</sub>i<sub>ca</sub>l <sub>so</sub>l<sub>vers</sub>

<sup>∗</sup>State Key La<sup>b</sup>oratory o<sup>f</sup> B<sup>l</sup>oc<sup>k</sup>c<sup>h</sup>ain an<sup>d</sup> Data Security, Z<sup>h</sup>ejiang University.   
<sup>†</sup>Han<sub>g</sub>zhou Hi<sub>g</sub>h-Tech Zone (Binjian<sub>g</sub>) Institute of Blockchain and Data Securit<sub>y</sub>.   
<sup>‡</sup>Corres<sub>p</sub>ondin<sub>g</sub> author.

![](images/57d00a81428678e1e5793cd5b5613fe072d3dac7eb617703fedf1ddd94261e03.jpg)  
Figure 1: Overview of the proposed LEAP. A linear predictor (blue boxes) extrapolates the subsequent score from a sliding history bufer of size � (dashed boxes), yielding an approximated state xˆ (grey circles) without model evaluations. To prevent error drift, a model evaluation subsequently refines this approximation into a corrected state x˜ (yellow circles).

th<sub>a</sub>t <sub>re</sub>l<sub>y on r</sub>i<sub>g</sub>id T<sub>ay</sub>l<sub>or coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s, our me</sub>th<sub>o</sub>d i<sub>n</sub>t<sub>ro</sub>d<sub>uces a</sub> d<sub>a</sub>t<sub>a-</sub> driven multi-ste<sub>p</sub> strate<sub>gy p</sub>owered b<sub>y</sub> least-s<sub>q</sub>uares o<sub>p</sub>timization. T<sup>h</sup>e intrinsic <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity o<sup>f</sup> <sup>d</sup>i<sup>f</sup>usion trajectories suggests th<sub>a</sub>t l<sub>oca</sub>l <sub>pa</sub>th <sub>segmen</sub>t<sub>s can</sub> b<sub>e accura</sub>t<sub>e</sub>l<sub>y c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub>d b<sub>y a</sub> li<sub>near</sub> subs<sub>p</sub>ace. Levera<sub>g</sub>in<sub>g</sub> this <sub>g</sub>eometric re<sub>g</sub>ularit<sub>y,</sub> we formulate the <sub>p</sub>rediction as a least-s<sub>q</sub>uares o<sub>p</sub>timization <sub>p</sub>roblem<sub>,</sub> which identifies the o<sub>p</sub>timal linear coeficients b<sub>y</sub> minimizin<sub>g</sub> the discre<sub>p</sub>- <sub>ancy</sub> b<sub>e</sub>t<sub>ween</sub> li<sub>near</sub>l<sub>y</sub> <sub>ex</sub>t<sub>rapo</sub>l<sub>a</sub>t<sub>e</sub>d hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>an</sub>d <sub>mu</sub>lti<sub>-s</sub>t<sub>ep</sub> groun<sup>d</sup>-trut<sup>h</sup> trajectory samp<sup>l</sup>es. Motivate<sup>d b</sup>y t<sup>h</sup>e cross-trajectory consistenc<sub>y,</sub> we demonstrate that the o<sub>p</sub>timal linear extra<sub>p</sub>olation coeficients are sam<sub>p</sub>le-a<sub>g</sub>nostic. This <sub>p</sub>ro<sub>p</sub>ert<sub>y g</sub>uarantees the transf<sub>era</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom</sub> <sub>a</sub> li<sub>m</sub>it<sub>e</sub>d <sub>se</sub>t <sub>o</sub>f <sub>re</sub>f<sub>erence</sub> t<sub>ra-</sub> jectories <sup>b</sup>e<sup>f</sup>ore in<sup>f</sup>erence, <sup>b</sup>ypassing t<sup>h</sup>e over<sup>h</sup>ea<sup>d</sup> <sup>f</sup>or per-samp<sup>l</sup>e o<sub>p</sub>timization. LEAP si<sub>g</sub>nificantl<sub>y</sub> enhances the sam<sub>p</sub>lin<sub>g</sub> eficienc<sub>y</sub> while <sub>p</sub>reservin<sub>g</sub> hi<sub>g</sub>h <sub>g</sub>eneration <sub>q</sub>ualit<sub>y</sub>. Cruciall<sub>y,</sub> as it o<sub>p</sub>erates entire<sup>l</sup>y at t<sup>h</sup>e macroscopic trajectory <sup>l</sup>eve<sup>l</sup>, our approac<sup>h</sup> <sup>f</sup>uncti<sub>ons as a mo</sub>d<sub>e</sub>l<sub>-agnos</sub>ti<sub>c an</sub>d <sub>p</sub>l<sub>ug-an</sub>d<sub>-p</sub>l<sub>ay mo</sub>d<sub>u</sub>l<sub>e.</sub> Thi<sub>s a</sub>ll<sub>ows</sub> for seamless inte<sub>g</sub>ration into diverse difusion models without an<sub>y</sub> <sub>a</sub>r<sub>c</sub>hit<sub>ec</sub>t<sub>u</sub>r<sub>a</sub>l m<sub>o</sub>difi<sub>ca</sub>ti<sub>o</sub>n<sub>s.</sub> Ext<sub>e</sub>n<sub>s</sub>i<sub>ve e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s ac</sub>r<sub>oss</sub> di<sub>ve</sub>r<sub>se</sub> modalities<sub>,</sub> includin<sub>g</sub> text-to-ima<sub>g</sub>e <sub>g</sub>eneration and text-to-3D <sub>g</sub>eneration demonstrate the su<sub>p</sub>erior <sub>p</sub>erformance and versatilit<sub>y</sub> of <sub>our me</sub>th<sub>o</sub>d<sub>.</sub>

In summar<sub>y,</sub> our main contributions are as follows:

• We <sub>p</sub>rove the intrinsic low-dimensionalit<sub>y</sub> and <sub>g</sub>eometric isomorp<sup>h</sup>ism o<sup>f</sup> generative trajectories across <sup>d</sup>iverse noise initia<sup>l</sup>- iz<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub>

• Motivated b<sub>y</sub> these theoretical insi<sub>g</sub>hts, we <sub>p</sub>ro<sub>p</sub>ose LEAP, a novel multi-ste<sub>p</sub> ODE solver that levera<sub>g</sub>es least-s<sub>q</sub>uares ali<sub>g</sub>nment to learn sam<sub>p</sub>le-a<sub>g</sub>nostic extra<sub>p</sub>olation coeficients<sub>,</sub> enablin<sub>g</sub> hi<sub>g</sub>hl<sub>y</sub> accurate linear lea<sub>p</sub>s with minimal com<sub>p</sub>utational <sub>over</sub>h<sub>ea</sub>d<sub>.</sub>

• Extensive ex<sub>p</sub>eriments across diverse <sub>g</sub>enerative tasks that s<sub>p</sub>annin<sub>g</sub> both text-to-tma<sub>g</sub>e s<sub>y</sub>nthesis and com<sub>p</sub>lex ima<sub>g</sub>e-to-3D <sub>genera</sub>ti<sub>on</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f <sub>our me</sub>th<sub>o</sub>d <sub>com-</sub> <sub>pare</sub>d t<sub>o</sub> <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>ng-</sub>f<sub>ree</sub> b<sub>ase</sub>li<sub>nes.</sub>

## 2 Preliminary

Difusion Models. Continuous-time difusion models progressively corru<sub>p</sub>t <sup>d</sup>ata $\mathbf { x } _ { 0 } \sim p _ { d a t a } ( \mathbf { x } )$ int<sub>o a</sub> tr<sub>ac</sub>t<sub>a</sub>bl<sub>e</sub> n<sub>o</sub>i<sub>se</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $\mathbf { x } _ { T } \sim$

$p _ { p r i o r } ( \mathbf { x } )$ via a forward Stochastic Diferential E<sub>q</sub>uation (SDE) [33]:

$$
\mathrm { d } \mathbf { x } _ { t } = f ( t ) \mathbf { x } _ { t } \mathrm { d } t + g ( t ) \mathrm { d } \mathbf { w } _ { t } , \qquad t \in [ 0 , T ] ,\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { w } _ { t }$ is the standard Wiener <sub>p</sub>rocess [22], while $f ( t )$ <sub>an</sub>d $g ( t )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> d<sub>r</sub>ift <sub>an</sub>d dif<sub>us</sub>i<sub>on coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> th<sub>e genera-</sub> ti<sub>ve process</sub> th<sub>a</sub>t <sub>reverses</sub> th<sub>e</sub> f<sub>orwar</sub>d SDE t<sub>o recover</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s-</sub> trib<sub>u</sub>tion is <sub>g</sub>o<sub>v</sub>erned b<sub>y</sub> the re<sub>v</sub>erse-time SDE $\mathrm { d } { \mathbf { x } } _ { t } = [ f ( t ) { \mathbf { x } } _ { t } -$ $g ( t ) ^ { 2 } \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) ] { \mathrm { d } } t + g ( t ) { \mathrm { d } } \bar { \mathbf { w } } _ { t }$ <sub>, w</sub>h<sub>ere</sub> $\bar { \mathbf { w } } _ { t }$ d<sub>eno</sub>t<sub>es a s</sub>t<sub>an</sub>d<sub>ar</sub>d Wi<sub>e</sub>n<sub>e</sub>r <sub>p</sub>r<sub>ocess</sub> in th<sub>e</sub> r<sub>eve</sub>r<sub>se</sub>-tim<sub>e</sub> dir<sub>ec</sub>ti<sub>o</sub>n<sub>.</sub> T<sub>o</sub> im<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt th<sub>e</sub> generative process, difusion models estimate the intractable score function $\nabla _ { \mathbf { x } _ { t } }$ <sup>l</sup>o<sub>g</sub> $p _ { t } ( \mathbf { x } _ { t } )$ <sub>us</sub>i<sub>ng a neura</sub>l <sub>ne</sub>t<sub>wor</sub>k $s _ { \theta } ( \mathbf { x } _ { t } , t )$ t<sub>ra</sub>i<sub>ne</sub>d b<sub>y</sub> score matc<sup>h</sup>ing o<sup>b</sup>jectives. For any <sup>d</sup>i<sup>f</sup>usion SDE, t<sup>h</sup>ere exists an Ordinar<sub>y</sub> Diferential E<sub>q</sub>uation (ODE) [33] that shares the identical mar<sub>g</sub>inal distrib<sub>u</sub>tions $p _ { t } ( \mathbf { x } )$ f<sub>or</sub> <sub>a</sub>ll $t \in [ 0 , T ]$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> k<sub>nown as</sub> the Probabilit<sub>y</sub> Flow ODE (PF-ODE):

$$
\mathrm { d } \mathbf { x } _ { t } = \left[ f ( t ) \mathbf { x } _ { t } - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla _ { \mathbf { x } _ { t } } \log \mathcal { p } _ { t } ( \mathbf { x } _ { t } ) \right] \mathrm { d } t .\tag{2}
$$

B<sub>y</sub> castin<sub>g</sub> the <sub>g</sub>enerative <sub>p</sub>rocess as a numerical initial value <sub>p</sub>robl<sub>e</sub>m<sub>,</sub> PF-ODE <sub>e</sub>n<sub>a</sub>bl<sub>es</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> <sub>sa</sub>m<sub>p</sub>lin<sub>g</sub> <sub>a</sub>nd <sub>u</sub>nd<sub>e</sub>r<sub>p</sub>in<sub>s</sub> <sub>va</sub>ri<sub>ous</sub> trainin<sub>g</sub>-free solver-based acceleration methods [18, 46].

Flow-based Models. Instead of explicitly defining a difusion <sub>process</sub> th<sub>roug</sub>h <sub>an</sub> SDE<sub>,</sub> fl<sub>ow-</sub>b<sub>ase</sub>d <sub>genera</sub>ti<sub>ve mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e a</sub> continuous-time Normalizin<sub>g</sub> Flow [1, 15, 17]. These models learn <sub>a</sub> ti<sub>me-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>vec</sub>t<sub>or</sub> fi<sub>e</sub>ld $v ( \mathbf { x } _ { t } , t )$ th<sub>a</sub>t t<sub>ranspor</sub>t<sub>s no</sub>i<sub>se</sub> t<sub>o</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>v</sub>i<sub>a a</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> ODE:

$$
\mathrm { d } \mathbf { x } _ { t } = v ( \mathbf { x } _ { t } , t ) \mathrm { d } t .\tag{3}
$$

To circ<sub>u</sub>m<sub>v</sub>ent the direct estimation of intractable mar<sub>g</sub>inal <sub>v</sub>ector <sup>fi</sup>e<sup>ld</sup>s, F<sup>l</sup>ow Matc<sup>h</sup>ing intro<sup>d</sup>uces a simu<sup>l</sup>ation-<sup>f</sup>ree o<sup>b</sup>jective w<sup>h</sup>ic<sup>h</sup> <sub>regresses a neura</sub>l <sub>vec</sub>t<sub>or</sub> fi<sub>e</sub>ld $v _ { \theta } ( \mathbf { x } _ { t } , t )$ t<sub>owar</sub>d<sub>s a con</sub>diti<sub>ona</sub>l <sub>vec</sub>t<sub>or</sub> fi<sub>e</sub>ld $\mathbf { u } _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ that defines the tar<sub>g</sub>et <sub>p</sub>robabilit<sub>y p</sub>ath �<sub>�</sub> (x):

$$
\mathbb { E } _ { t , \rho _ { 0 } ( \mathbf { x } _ { 0 } ) , \rho _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) } \left[ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] .\tag{4}
$$

S<sub>p</sub>ecificall<sub>y,</sub> b<sub>y</sub> settin<sub>g</sub> $T = 1 ,$ <sub>,</sub> anal<sub>y</sub>ticall<sub>y</sub> solvin<sub>g</sub> E<sub>q</sub>. 1 <sub>y</sub>ields a re<sub>p</sub>arameterization of $\mathbf { X } _ { t }$ as

$$
\begin{array} { r } { \mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { 0 } + \sigma _ { t } \mathbf { x } _ { 1 } , \qquad \mathbf { x } _ { 1 } \sim N ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}\tag{5}
$$

<sub>w</sub>h<sub>ere</sub>

$$
\alpha _ { t } = \exp \left( \int _ { 0 } ^ { t } f ( u ) \mathrm { d } u \right) , \quad \sigma _ { t } = \left( \int _ { 0 } ^ { t } g ( u ) ^ { 2 } \exp \left( - 2 \int _ { 0 } ^ { u } f ( s ) \mathrm { d } s \right) \mathrm { d } u \right) ^ { 1 / 2 } .\tag{6}
$$

![](images/333ec9898d1ec9141c2cf3553896a58d62ebfe81b83543e8137e44b855983d2b.jpg)  
Figure 2: Visualizing the trajectory regularity of difusion models. Following the visualization protocol in [5], we project 100 sampling trajectories generated with distinct noise and prompt initializations using SD3.5-Medium [8].

Th<sub>e con</sub>diti<sub>ona</sub>l <sub>ve</sub>l<sub>oc</sub>it<sub>y</sub> fi<sub>e</sub>ld i<sub>s</sub> th<sub>us g</sub>i<sub>ven</sub> b<sub>y</sub>

$$
\mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = { \frac { { \dot { \sigma } } _ { t } } { \sigma _ { t } } } \mathbf { x } _ { t } + \left( { \dot { \alpha } } _ { t } - \alpha _ { t } { \frac { { \dot { \sigma } } _ { t } } { \sigma _ { t } } } \right) \mathbf { x } _ { 0 } .\tag{7}
$$

Contem<sub>p</sub>orar<sub>y</sub> cuttin<sub>g</sub>-ed<sub>g</sub>e text-to-ima<sub>g</sub>e models increasin<sub>g</sub>l<sub>y</sub> favor the Flow Matchin<sub>g</sub> framework due to its conce<sub>p</sub>tual sim<sub>p</sub>licit<sub>y</sub> and flexible <sub>p</sub>ath desi<sub>g</sub>n [2, 8]. Since the<sub>y</sub> inherit the architecture <sub>an</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng conven</sub>ti<sub>ons o</sub>f dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s, we s</sub>till <sub>re</sub>f<sub>er</sub> t<sub>o</sub> th<sub>e</sub> fl<sub>ow-</sub>b<sub>ase</sub>d f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>as</sub> “dif<sub>us</sub>i<sub>on</sub>” <sub>w</sub>h<sub>en</sub> <sub>no</sub> <sub>am</sub>bi<sub>gu</sub>it<sub>y</sub> <sub>ar</sub>i<sub>ses.</sub>

Trajectory Regularity. While difusion models operate in an immensel<sub>y</sub> hi<sub>g</sub>h-dimensional ambient s<sub>p</sub>ace<sub>,</sub> recent em<sub>p</sub>irical observations revea<sup>l</sup> t<sup>h</sup>at t<sup>h</sup>eir samp<sup>l</sup>ing trajectories ex<sup>h</sup>i<sup>b</sup>it remar<sup>k</sup>- able <sub>g</sub>eometric re<sub>g</sub>ularit<sub>y</sub> [4, 5, 37]. Princi<sub>p</sub>al Com<sub>p</sub>onent Anal<sub>y</sub>sis (PCA) on individual sam<sub>p</sub>lin<sub>g</sub> trajector<sub>y</sub> exhibits that the <sub>g</sub>enerative dynamics are inherently low-dimensional where the explained variance is dominated b<sub>y</sub> the first few <sub>p</sub>rinci<sub>p</sub>al com<sub>p</sub>onents. Furt<sup>h</sup>ermore, as i<sup>ll</sup>ustrate<sup>d</sup> in Fig. 2, w<sup>h</sup>en projecting mu<sup>l</sup>tip<sup>l</sup>e trajectori<sub>es</sub> <sub>w</sub>ith di<sub>s</sub>tin<sub>c</sub>t n<sub>o</sub>i<sub>se</sub> <sub>a</sub>nd <sub>p</sub>r<sub>o</sub>m<sub>p</sub>t initi<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> int<sub>o</sub> <sub>a</sub> <sub>s</sub>h<sub>a</sub>r<sub>e</sub>d 3D <sub>space,</sub> th<sub>ey</sub> b<sub>un</sub>dl<sub>e</sub> i<sub>n</sub>t<sub>o a smoo</sub>th ‘t<sub>u</sub>b<sub>e</sub>’<sub>.</sub> Thi<sub>s v</sub>i<sub>sua</sub>l <sub>a</sub>li<sub>gnmen</sub>t hi<sub>g</sub>h lights a striking cross-trajectory consistency, indicating that diferent difusion <sub>p</sub>aths share nearl<sub>y</sub> isomor<sub>p</sub>hic macrosco<sub>p</sub>ic d<sub>y</sub>namics. Toget<sup>h</sup>er, t<sup>h</sup>is inter-trajectory <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity an<sup>d</sup> inter-trajectory consistency constitute t<sup>h</sup>e trajectory regu<sup>l</sup>arity, serving as t<sup>h</sup>e <sup>f</sup>ound<sub>a</sub>ti<sub>ona</sub>l i<sub>n</sub>t<sub>u</sub>iti<sub>on</sub> f<sub>or our su</sub>b<sub>sequen</sub>t th<sub>eore</sub>ti<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s an</sub>d th<sub>e</sub> d<sub>es</sub>i<sub>gn</sub> <sub>o</sub>f <sub>our</sub> <sub>acce</sub>l<sub>era</sub>ti<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m.</sub>

## 3 Methodology

## 3.1 The Intrinsic Geometry of Generative Trajectories

While <sub>p</sub>revious works [4, 5] have observed stron<sub>g g</sub>eometric re<sub>g</sub>u-<sup>l</sup>arity o<sup>f</sup> ODE samp<sup>l</sup>ing trajectories to justi<sup>f</sup>y existing acce<sup>l</sup>erate<sup>d</sup> <sub>samp</sub>li<sub>ng me</sub>th<sub>o</sub>d<sub>s, a</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng rema</sub>i<sub>ns unc</sub>l<sub>ear.</sub> In t<sup>h</sup>is section, we prove t<sup>h</sup>at <sup>d</sup>i<sup>f</sup>usion samp<sup>l</sup>ing trajectories can b<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y a</sub>li<sub>gne</sub>d th<sub>roug</sub>h <sub>or</sub>th<sub>ogona</sub>l t<sub>rans</sub>f<sub>orma</sub>ti<sub>on an</sub>d <sub>are</sub> <sub>em</sub>b<sub>e</sub>dd<sub>e</sub>d i<sub>n</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>su</sub>b<sub>spaces.</sub> T<sub>o</sub> thi<sub>s</sub> <sub>en</sub>d<sub>,</sub> <sub>we</sub> <sub>per</sub>f<sub>orm</sub> an anal<sub>y</sub>sis within the Gaussian mixture model (GMM) framework <sub>w</sub>ith <sub>s</sub>h<sub>are</sub>d <sub>sp</sub>h<sub>er</sub>i<sub>ca</sub>l <sub>covar</sub>i<sub>ance</sub> t<sub>o ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> dif<sub>u-</sub> sion d<sub>y</sub>namics. Since GMMs are <sub>u</sub>niversal a<sub>pp</sub>roximators ca<sub>p</sub>able of <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> hi<sub>g</sub>hl<sub>y</sub> <sub>comp</sub>l<sub>ex</sub> d<sub>a</sub>t<sub>a</sub> <sub>man</sub>if<sub>o</sub>ld<sub>s,</sub> thi<sub>s</sub> <sub>assump</sub>ti<sub>on</sub> <sub>no</sub>t <sub>on</sub>l<sub>y</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>es c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l d<sub>er</sub>i<sub>va</sub>ti<sub>ons</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so ensures</sub> th<sub>e</sub> b<sub>roa</sub>d <sub>app</sub>li<sub>ca</sub>bilit<sub>y o</sub>f <sub>our</sub> th<sub>eore</sub>ti<sub>ca</sub>l fi<sub>n</sub>di<sub>ngs.</sub> U<sub>n</sub>d<sub>er</sub> thi<sub>s m</sub>ild assum<sub>p</sub>tion<sub>,</sub> the data distribution is <sub>g</sub>iven b<sub>y</sub>

$$
{ p _ { \mathrm { d a t a } } } ( \mathbf { x } ) = \sum _ { s = 1 } ^ { S } { { \pi } _ { s } } N ( \mu _ { s } , { \sigma _ { 0 } ^ { 2 } } { \mathbf { I } } _ { d } ) , \quad { \pi } _ { s } > 0 , \quad \sum _ { r = 1 } ^ { S } { { \pi } _ { s } } = 1 ,\tag{8}
$$

<sub>w</sub>h<sub>ere</sub> � i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>componen</sub>t<sub>s,</sub> $\pi _ { s }$ are the mixin<sub>g</sub> wei<sub>g</sub>hts <sub>an</sub>d $\pmb { \mu } _ { s }$ <sub>are</sub> th<sub>e c</sub>l<sub>us</sub>t<sub>er means.</sub> L<sub>e</sub>t

$$
U : = \mathrm { s p a n } \{ \mu _ { 1 } , \dots , \mu _ { S } \} , \quad r : = \mathrm { d i m } ( U ) , \quad R : = \operatorname* { m a x } _ { 1 \leq s \leq S } \| \mu _ { s } \| .
$$

W<sub>e co</sub>n<sub>s</sub>id<sub>e</sub>r th<sub>e e</sub>x<sub>ac</sub>t PF-ODE <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e va</sub>ri<sub>a</sub>n<sub>ce</sub>- <sub>p</sub>reservin<sub>g</sub> forward difusion. Let $t \in [ 0 , T ]$ d<sub>eno</sub>t<sub>e</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d f<sub>orwar</sub>d dif<sub>us</sub>i<sub>on</sub> ti<sub>me.</sub> D<sub>eno</sub>t<sub>e</sub> $\mathbf { x } _ { t } ^ { ( m ) }$ t<sup>h</sup>e �-t<sup>h</sup> samp<sup>l</sup>ing trajectory evolvin<sub>g</sub> backward in time which is initialized from a <sub>p</sub>ure noise state $\mathbf { x } _ { T } ^ { ( m ) } \sim p _ { T }$ . To ana<sup>l</sup>yze t<sup>h</sup>e trajectory at <sup>d</sup>iscrete interva<sup>l</sup>s, we <sub>cons</sub>id<sub>er a</sub> b<sub>ac</sub>k<sub>war</sub>d ti<sub>me gr</sub>id $T = t _ { N } > t _ { N - 1 } > \cdot \cdot \cdot > t _ { 0 } = 0$ <sub>an</sub>d <sup>d</sup>e<sup>fi</sup>ne t<sup>h</sup>e trajectory states as

$$
\mathbf { D } _ { n } ^ { ( m ) } : = \mathbf { x } _ { t _ { n } } ^ { ( m ) } - \mathbf { x } _ { T } ^ { ( m ) } ,\tag{9}
$$

which re<sub>p</sub>resents the dis<sub>p</sub>lacement from the initial noise. Let

$$
\alpha _ { T } : = \exp \Big ( - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \beta ( s ) d s \Big ) , \qquad \nu _ { T } ^ { 2 } : = \alpha _ { T } ^ { 2 } \sigma _ { 0 } ^ { 2 } + ( 1 - \alpha _ { T } ^ { 2 } ) .\tag{10}
$$

F<sub>or</sub> b<sub>rev</sub>it<sub>y, we presen</sub>t <sub>a s</sub>i<sub>mp</sub>lifi<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>w</sub>ith <sub>a proo</sub>f<sub>s</sub>k<sub>e</sub>t<sub>c</sub>h b<sub>e</sub>l<sub>ow.</sub> Th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>eorem</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>a</sub>ll <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> <sub>an</sub>d <sub>r</sub>i<sub>gorous</sub> <sub>p</sub>r<sub>oo</sub>f i<sub>s</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d in th<sub>e</sub> A<sub>ppe</sub>ndix A<sub>.</sub>

Theorem 3.1 (Cross-trajectory consistency). Suppose that $\mathbf { x } _ { T } ^ { ( 1 ) } , \ldots , \mathbf { x } _ { T } ^ { ( M ) }$ are i.i.d. samples from the exact noised marginal ${ \mathit { p } } _ { T }$ Then for every $\delta \in ( 0 , 1 )$ ), with probability at least $1 - \delta ,$

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { 1 \leq m _ { 1 } , m _ { 2 } \leq M } \operatorname* { i n f } _ { \mathbf { \delta H } \in O ( d ) } \operatorname* { m a x } _ { 0 \leq n \leq N } \| \mathbf { D } _ { n } ^ { ( m _ { 1 } ) } - \mathbf { H D } _ { n } ^ { ( m _ { 2 } ) } \| } } \\ & { } & \\ & { } & { \leq C _ { T , 1 } \nu _ { T } ( \sqrt { r } + \sqrt { \log \frac { M } { \delta } } ) + C _ { T , 2 } R , } \end{array}\tag{11}
$$

where $C _ { T , 1 } , C _ { T , 2 } > 0$ depend only on the difusion schedule and not on �, �, or the mixture parameters.

Corollary 3.2 (Asymptotic consistency). $I f r + \log M = o ( d )$ and $R = o \left( \nu _ { T } \sqrt { d } \right)$ , then for every fixed $\delta \in ( 0 , 1 )$

$$
\frac { 1 } { \nu _ { T } \sqrt { d } } \operatorname* { m a x } _ { 1 \leq m _ { 1 } , m _ { 2 } \leq M } \operatorname* { i n f } _ { \mathbf { H } \in O ( d ) } \operatorname* { m a x } _ { 0 \leq n \leq N } \Bigl \| \mathbf { D } _ { n } ^ { ( m _ { 1 } ) } - \mathbf { H D } _ { n } ^ { ( m _ { 2 } ) } \Bigr \| \xrightarrow [ d \to \infty ] \mathbb { P } 0 .\tag{12}
$$

Proof sketch<sub>.</sub> L<sub>e</sub>t $P$ <sub>an</sub>d $Q = \mathbf { I } - P$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>or</sub>th<sub>ogona</sub>l <sub>pro-</sub> jections onto � an<sup>d</sup> $U ^ { \perp }$ <sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. In the A<sub>pp</sub>endix A<sub>,</sub> we show th<sub>a</sub>t th<sub>e reverse-</sub>ti<sub>me</sub> PF<sub>-</sub>ODE <sub>can</sub> b<sub>e wr</sub>itt<sub>en as</sub>

$$
\mathrm { d } \mathbf { x } _ { t } = \left[ a _ { t } \mathbf { x } _ { t } + b _ { t } \mathbf { m } _ { t } ( P \mathbf { x } _ { t } ) \right] \mathrm { d } t ,\tag{13}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { m } _ { t } ( \cdot ) \in U$ is the <sub>p</sub>osterior mean of the com<sub>p</sub>onent centers at time �. Conse<sub>q</sub>uentl<sub>y,</sub> se<sub>p</sub>aratin<sub>g</sub> the d<sub>y</sub>namics <sub>y</sub>ields

$$
Q \mathrm { d } \mathbf { x } _ { t } = a _ { t } Q \mathbf { x } _ { t } \mathrm { d } t , \qquad P \mathrm { d } \mathbf { x } _ { t } = \left[ a _ { t } P \mathbf { x } _ { t } + b _ { t } \mathbf { m } _ { t } ( P \mathbf { x } _ { t } ) \right] \mathrm { d } t .\tag{14}
$$

Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e</sub> $U ^ { \perp }$ <sub>-componen</sub>t <sub>evo</sub>l<sub>ves</sub> b<sub>y</sub> <sub>a</sub> <sub>common</sub> <sub>sca</sub>l<sub>ar</sub> <sub>resca</sub>li<sub>ng,</sub> <sub>w</sub>hil<sub>e</sub> <sub>a</sub>ll <sub>m</sub>i<sub>x</sub>t<sub>ure-spec</sub>ifi<sub>c</sub> <sub>non-</sub>li<sub>near</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s</sub> <sub>are</sub> <sub>con</sub>fi<sub>ne</sub>d t<sub>o</sub> th<sub>e</sub> $r \mathrm { - }$ dimensional subs<sub>p</sub>ace � .

Geometricall<sub>y,</sub> this im<sub>p</sub>lies that after a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the ortho<sub>g</sub>onal a<sup>l</sup>ignment, t<sup>h</sup>e maximum <sup>d</sup>iscrepancy <sup>b</sup>etween any pair o<sup>f</sup> traject<sub>or</sub>i<sub>es</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> t<sub>wo</sub> f<sub>ac</sub>t<sub>ors:</sub> th<sub>e norm</sub> dif<sub>erence o</sub>f th<sub>e</sub> initial noise within the ortho<sub>g</sub>onal com<sub>p</sub>lement $( \mathrm { i . e . , } \big | \| Q \mathbf { x } _ { T } ^ { ( m _ { 1 } ) } \big | \big | -$ $\big \| Q \mathbf { x } _ { T } ^ { ( m _ { 2 } ) } \big \| \big \| \big )$ <sub>, an</sub>d th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e magn</sub>it<sub>u</sub>d<sub>e o</sub>f th<sub>e s</sub>i<sub>gna</sub>l <sub>componen</sub>t <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a su</sub>b<sub>space</sub> $( \mathrm { i . e . , } \| P \mathbf { x } _ { T } ^ { ( m ) } \| )$ <sub>.</sub> Sin<sub>ce</sub> $Q \mathbf { x } _ { T } ^ { ( m ) }$ i<sub>s a</sub> G<sub>auss</sub>i<sub>a</sub>n <sub>vec</sub>t<sub>o</sub>r in dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n $d - r ,$ its norm concentrates uniforml<sub>y</sub> over <sub>�,</sub> <sub>suc</sub>h th<sub>a</sub>t th<sub>e norm</sub> dif<sub>erence</sub> i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b $O \big ( \nu _ { T } \sqrt { \log ( M / \delta ) } \big )$ <sub>.</sub> On th<sub>e o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>,</sub> ${ \cal P } { \bf x } _ { T } ^ { ( m ) }$ lives in an <sub>�</sub>-dimensional subs<sub>p</sub>ace and its <sub>norm sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\| P \mathbf { x } _ { T } ^ { ( m ) } \| = O \left( \alpha _ { T } R + \nu _ { T } \Big ( \sqrt { r } + \sqrt { \log ( M / \delta ) } \Big ) \right)\tag{15}
$$

uniforml<sub>y</sub> over <sub>�</sub> with hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub>. Combinin<sub>g</sub> these bounds <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> Th<sub>m.</sub> 3<sub>.</sub>1<sub>, an</sub>d C<sub>or.</sub> 3<sub>.</sub>2 f<sub>o</sub>ll<sub>ows</sub> b<sub>y norma</sub>li<sub>za</sub>ti<sub>on w</sub>ith $\nu _ { T } { \sqrt { d } }$ □

Remark. Cor. 3.2 indicates that in high-dimensional settings $( d  \infty )$ <sub>w</sub>hi<sub>c</sub>h <sub>ma</sub>t<sub>c</sub>h<sub>es se</sub>tti<sub>ng o</sub>f dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s,</sub> th<sub>e error</sub> b<sub>oun</sub>d i<sub>n</sub> Th<sub>m.</sub> 3<sub>.</sub>1 b<sub>ecomes</sub> <sub>prac</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>van</sub>i<sub>s</sub>h<sub>es.</sub> Thi<sub>s</sub> provi<sup>d</sup>es a strong t<sup>h</sup>eoretica<sup>l</sup> guarantee t<sup>h</sup>at our cross-trajectory ali<sub>g</sub>nment is hi<sub>g</sub>hl<sub>y</sub> accurate in realistic <sub>g</sub>eneration scenarios.

Corollary 3.3 (Intrinsic low-dimensionality of individual trajectories). Underthe assumptions ofThm. 3.1,foreach sampling trajectory $\mathbf { x } _ { t } ^ { ( m ) }$ , there exists a linear subspace

$$
S _ { m } : = U \oplus \operatorname { s p a n } \{ Q \mathbf { x } _ { T } ^ { ( m ) } \}\tag{16}
$$

with dim $( S _ { m } ) \leq r + 1 ;$ , such that

$$
\mathbf { x } _ { t _ { n } } ^ { ( m ) } \in S _ { m } , \qquad 0 \leq n \leq N .\tag{17}
$$

Equivalently, $i f \mathbf { X } ^ { ( m ) } \in \mathbb { R } ^ { ( N + 1 ) \times d }$ denotes the trajectory matrix whose �-th row is $( \mathbf { x } _ { t _ { n } } ^ { ( m ) } ) ^ { \top }$ , then

$$
\operatorname { r a n k } ( \mathbf { X } ^ { ( m ) } ) \leq r + 1 .\tag{18}
$$

Remark. Cor. 3.3 reveals that all trajectory variance is captured b<sub>y</sub> at most � + 1 <sub>p</sub>rinci<sub>p</sub>al com<sub>p</sub>onents. However, this $( r + 1 )$ di<sub>mens</sub>i<sub>ona</sub>l <sub>su</sub>b<sub>space</sub> $S _ { m }$ is sam<sub>p</sub>le-de<sub>p</sub>endent<sub>,</sub> as it is s<sub>p</sub>anned b<sub>y</sub> th<sub>e</sub> initi<sub>a</sub>l n<sub>o</sub>i<sub>se vec</sub>t<sub>o</sub>r $Q \mathbf { x } _ { T } ^ { ( m ) }$ <sub>.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng coro</sub>ll<sub>ary es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es a</sub> remarkable <sub>g</sub>eometric invariance: b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> a suitable ortho<sub>g</sub>onal trans<sup>f</sup>ormation to eac<sup>h</sup> trajectory, a<sup>ll</sup> trajectories can <sup>b</sup>e per<sup>f</sup>ect<sup>l</sup>y aligned into a single, shared low-dimensional subspace.

Corollary 3.4 (Universal low-dimensional alignment of diffusion trajectories). For any fixed unit vector e ∈ $U ^ { \perp }$ , there exist orthogonal matrices $\mathbf { H } _ { 1 } , \dots , \mathbf { H } _ { M } \in O ( d )$ satisfying ${ \bf H } _ { m } | _ { U } = { \bf I }$ and

$$
\mathbf { H } _ { m } { \cal Q } \mathbf { x } _ { T } ^ { ( m ) } = \| { \cal Q } \mathbf { x } _ { T } ^ { ( m ) } \| \mathbf { e } , \qquad \forall m \in \{ 1 , \ldots , M \} .\tag{19}
$$

Under such transformations, the aligned trajectories defined by

$$
\bar { \mathbf { x } } _ { t _ { n } } ^ { ( m ) } : = \mathbf { H } _ { m } \mathbf { x } _ { t _ { n } } ^ { ( m ) }\tag{20}
$$

all reside in the same $( r + 1 )$ -dimensional subspace

$$
{ \bar { S } } : = U \oplus { \mathrm { s p a n } } \{ \mathbf { e } \} .\tag{21}
$$

PF-ODE exhibits t<sub>w</sub>o com<sub>p</sub>lementar<sub>y</sub> str<sub>u</sub>ct<sub>u</sub>ral <sub>p</sub>ro<sub>p</sub>erties: crosstrajectory consistency an<sup>d</sup> intrinsic in<sup>d</sup>ivi<sup>d</sup>ua<sup>l</sup> <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity. T<sup>h</sup>m. 3.1 esta<sup>bl</sup>is<sup>h</sup>es t<sup>h</sup>at, <sup>d</sup>i<sup>f</sup>erent samp<sup>l</sup>ing trajectories can <sup>b</sup>e hi<sub>g</sub>hl<sub>y</sub> <sub>a</sub>li<sub>gne</sub>d th<sub>roug</sub>h <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> <sub>or</sub>th<sub>ogona</sub>l t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> C<sub>or.</sub> 3<sub>.</sub>3 <sup>f</sup>urt<sup>h</sup>er <sup>d</sup>emonstrates t<sup>h</sup>at eac<sup>h</sup> trajectory is strict<sup>l</sup>y con<sup>fi</sup>ne<sup>d</sup> to an $( r + 1 )$ -<sup>d</sup>imensiona<sup>l</sup> su<sup>b</sup>space. Nota<sup>bl</sup>y, a<sup>f</sup>ter a simp<sup>l</sup>e per-trajectory a<sup>l</sup>ignment intro<sup>d</sup>uce<sup>d</sup> <sup>b</sup>y Cor. 3.4, t<sup>h</sup>e entire co<sup>ll</sup>ection o<sup>f</sup> trajectories resides within a universal (� + 1)-dimensional subs<sub>p</sub>ace. Therefore, the hi<sub>g</sub>h-dimensional sam<sub>p</sub>lin<sub>g</sub> d<sub>y</sub>namics are efectivel<sub>y</sub> <sub>g</sub>overned b<sub>y on</sub>l<sub>y a</sub> f<sub>ew</sub> l<sub>a</sub>t<sub>en</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es: one ra</sub>di<sub>a</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>n</sub> $U ^ { \perp }$ <sub>an</sub>d at most <sub>�</sub> coordinates within the mean subs<sub>p</sub>ace �.

![](images/326443f63d6363ba9667cef85798cc936b7ed42ffef5914b17fe2d7844fb9361.jpg)  
Figure 3: Relative prediction error vs. noise level $\sigma _ { t }$ . We quantify the projection error of each score onto the subspace spanned by its corresponding historical bufer. The relative error is strictly bounded, using a small window size of $r = 3 .$

More im<sub>p</sub>ortantl<sub>y,</sub> the intrinsic low-dimensionalit<sub>y</sub> established in Cor. 3.3 im<sub>p</sub>lies that<sub>,</sub> within an<sub>y</sub> local window of a <sub>g</sub>iven trajectory, <sup>b</sup>ot<sup>h</sup> t<sup>h</sup>e <sup>h</sup>istorica<sup>l</sup> scores an<sup>d</sup> t<sup>h</sup>e target output resi<sup>d</sup>e in a subs<sub>p</sub>ace. While this <sub>g</sub>eometric re<sub>g</sub>ularit<sub>y</sub> is ri<sub>g</sub>orousl<sub>y</sub> establi<sub>s</sub>h<sub>e</sub>d <sub>un</sub>d<sub>er m</sub>ild <sub>assump</sub>ti<sub>on,</sub> it<sub>s va</sub>lidit<sub>y</sub> i<sub>n rea</sub>l<sub>-wor</sub>ld <sub>genera</sub>ti<sub>ve</sub> scenarios warrants em<sub>p</sub>irical verification. To brid<sub>g</sub>e this <sub>g</sub>a<sub>p,</sub> we investigate t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup> score <sup>d</sup>ynamics using samp<sup>l</sup>ing trajectories <sup>f</sup>rom Sta<sup>bl</sup>e Di<sup>f</sup>usion 3.5. Speci<sup>fi</sup>ca<sup>ll</sup>y, we quanti<sup>f</sup>y t<sup>h</sup>e ort<sup>h</sup>ogona<sup>l</sup> projecti<sub>on error o</sub>f th<sub>e curren</sub>t <sub>score</sub> $\mathbf { s } _ { t }$ <sub>on</sub>t<sub>o</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>su</sub>b<sub>space spanne</sub>d b<sub>y</sub> <sub>a s</sub>h<sub>or</sub>t hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l b<sub>u</sub>f<sub>er,</sub> d<sub>eno</sub>t<sub>e</sub>d <sub>as span</sub> $\begin{array} { r l } {  { \mathrm { 1 } \big ( \mathbf { s } _ { t + r } , \mathbf { s } _ { t + r - 1 } , \ldots \mathbf { s } _ { t + r } , \mathbf { s } _ { t + r + 1 } \big ) } \qquad } & { { } } \end{array}$ <sub>.</sub> A<sub>s</sub> i<sup>ll</sup>ustrate<sup>d</sup> in Fig. 3, t<sup>h</sup>e re<sup>l</sup>ative projection error is strict<sup>l</sup>y <sup>b</sup>oun<sup>d</sup>e<sup>d</sup>, dro<sub>pp</sub>in<sub>g</sub> to a mere 4%–5% even with a small window size $r = 3$ Thi<sub>s emp</sub>i<sub>r</sub>i<sub>ca</sub>l fi<sub>n</sub>di<sub>ng con</sub>fi<sub>rms</sub> th<sub>a</sub>t <sub>even</sub> f<sub>or</sub> hi<sub>g</sub>hl<sub>y comp</sub>l<sub>ex</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons,</sub> th<sub>e</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub> f<sub>u</sub>t<sub>ure score rema</sub>i<sub>ns s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y con</sub>fi<sub>ne</sub>d within the linear s<sub>p</sub>an of its recent histor<sub>y</sub>. This em<sub>p</sub>irical evident f<sub>or</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>lit<sub>y</sub> <sub>seam</sub>l<sub>ess</sub>l<sub>y</sub> t<sub>rans</sub>f<sub>orms</sub> <sub>our</sub> th<sub>eore</sub>ti<sub>ca</sub>l i<sub>n-</sub> si<sub>g</sub>hts into a <sub>p</sub>ractical al<sub>g</sub>orithm<sub>,</sub> leadin<sub>g</sub> to our linear extra<sub>p</sub>olation f<sub>ramewor</sub>k i<sub>n</sub> th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>sec</sub>ti<sub>on.</sub>

## 3.2 Optimal Linear Leaps with Aligned Trajectories

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we presen</sub>t <sub>our me</sub>th<sub>o</sub>d <sub>groun</sub>d<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> th<sub>e</sub> l<sub>oca</sub>l li<sub>near</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub> Thi<sub>s</sub> <sub>geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>regu</sub>l<sub>ar</sub>it<sub>y</sub> <sub>un</sub>d<sub>erp</sub>i<sub>ns</sub> th<sub>e use o</sub>f <sub>a</sub> li<sub>near ex</sub>t<sub>rapo</sub>l<sub>a</sub>t<sub>or</sub> t<sub>o pre</sub>di<sub>c</sub>t <sub>nex</sub>t <sub>score</sub> function. The overall <sub>p</sub>i<sub>p</sub>eline of our a<sub>pp</sub>roach is illustrated in Fi<sub>g</sub>. 1. T<sub>o</sub> f<sub>orma</sub>li<sub>ze</sub> thi<sub>s</sub> <sub>process,</sub> <sub>suppose</sub> <sub>we</sub> <sub>co</sub>ll<sub>ec</sub>t <sub>an</sub> <sub>ensem</sub>bl<sub>e</sub> <sub>o</sub>f � <sup>fi</sup>ne-graine<sup>d</sup> re<sup>f</sup>erence samp<sup>l</sup>ing trajectories. For t<sup>h</sup>e �-t<sup>h</sup> trajectory $( m = 1 , \ldots , M )$ <sub>,</sub> d<sub>eno</sub>t<sub>e</sub> $\tilde { \mathbf { s } } _ { n } ^ { ( m ) }$ the model out<sub>p</sub>ut (e.<sub>g</sub>., estimated score function) at timeste<sub>p</sub> $t _ { n } .$ <sub>, an</sub>d l<sub>e</sub>t $\tilde { \mathbf { x } } _ { n - 1 } ^ { ( m ) }$ b<sub>e</sub> th<sub>e nex</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> $\tilde { \mathbf { s } } _ { n } ^ { ( m ) }$

Given a win<sup>d</sup>ow size �, our o<sup>b</sup>jective at timestep $t _ { n }$ is to <sub>p</sub>redict th<sub>e</sub> <sub>nex</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>ou</sub>t<sub>pu</sub>t $\tilde { \bf s } _ { n - 1 }$ b<sub>y</sub> utilizin<sub>g</sub> <sub>�</sub> stored <sub>p</sub>revious model outputs a<sup>l</sup>ong t<sup>h</sup>e samp<sup>l</sup>ing trajectory. Denote t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup> <sup>h</sup>istory matrix of scores and the corres<sub>p</sub>ondin<sub>g</sub> tar<sub>g</sub>et b<sub>y</sub>

Algorithm 1 Coeficient Bank Construction   
Require: Reference score trajectories $\{ \mathcal { T } ^ { ( m ) } \} _ { m = 1 } ^ { M } ,$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\mathcal { T } ^ { ( m ) } =$   
$\{ \tilde { \mathbf { s } } _ { N } ^ { ( m ) } , \tilde { \mathbf { s } } _ { N - 1 } ^ { ( m ) } , . . . , \tilde { \mathbf { s } } _ { 1 } ^ { ( m ) } \}$ <sub>; w</sub>i<sub>n</sub>d<sub>ow s</sub>i<sub>ze �; r</sub>id<sub>ge coe</sub>fi<sub>c</sub>i<sub>en</sub>t �   
Ensure: Coeficient bank $C = \{ \mathbf { c } _ { N - r + 1 } , \mathbf { c } _ { N - r } , \ldots , \mathbf { c } _ { 1 } \}$   
1: for $n = N - r + 1 , N - r , . . . , 1$ do   
2: for $m = 1 , 2 , \ldots , M$ do   
3: for $j = 1 , \ldots , r - 1$ do   
4: $\Delta _ { n , j } ^ { ( m ) }  \tilde { \mathbf { s } } _ { n + j } ^ { ( m ) } - \tilde { \mathbf { s } } _ { n + j - 1 } ^ { ( m ) }$   
5: end for   
6: $\mathbf { B } _ { n } ^ { ( m ) } \gets \left[ \Delta _ { n , r - 1 } ^ { ( m ) } , \dots , \Delta _ { n , 1 } ^ { ( m ) } , \tilde { \mathbf { s } } _ { n } ^ { ( m ) } \right]$   
7: end for   
8: $\begin{array} { r } { \mathbf { c } _ { n } ^ { \star } \gets \arg \operatorname* { m i n } _ { \mathbf { c } \in \mathbb { R } ^ { r } } \sum _ { m = 1 } ^ { M } \left\| \tilde { \mathbf { s } } _ { n - 1 } ^ { ( m ) } - \mathbf { B } _ { n } ^ { ( m ) } \mathbf { c } \right\| _ { 2 } ^ { 2 } + \lambda \| \mathbf { c } \| _ { 2 } ^ { 2 } } \end{array}$   
9: end for   
10: return $C = \{ \mathbf { c } _ { N - r + 1 } ^ { \star } , \mathbf { c } _ { N - r } ^ { \star } , \ldots , \mathbf { c } _ { 1 } ^ { \star } \}$

Algorithm 2 LEAP Sampling (Predict-Correct)   
Require: Initial state $\tilde { \mathbf { x } } _ { N } ;$ coeficient bank C; window size �; score   
<sub>ne</sub>t<sub>wor</sub>k <sub>��</sub>   
Ensure: Final sample $\tilde { \mathbf { x } } _ { 0 }$   
1<sub>:</sub> Warm-u<sub>p</sub>: run base sam<sub>p</sub>ler for <sub>�</sub> ste<sub>p</sub>s   
2: $\mathcal { B } \gets \{ \tilde { \mathbf { s } } _ { N } , \tilde { \mathbf { s } } _ { N - 1 } , . . . , \tilde { \mathbf { s } } _ { N - r + 1 } \}$   
3: Update current state to x˜ <sub>�−�</sub>   
4: for $n = N - r + 1$ down to 3 step 2 do   
5: Predict:   
6: Com<sub>pu</sub>te diferences $\Delta _ { n , j }$ from B for $j = 1 , \ldots , r - 1$   
7: $\begin{array} { r } { \hat { \mathbf { s } } _ { n - 1 } \gets \mathbf { c } _ { n , r - 1 } \tilde { \mathbf { s } } _ { n } + \sum _ { j = 0 } ^ { r - 2 } \mathbf { c } _ { n , j } \Delta _ { n , r - j - 1 } } \end{array}$   
8: E<sub>vo</sub>l<sub>ve</sub> t<sub>o</sub> $\hat { \mathbf { x } } _ { n - 2 }$ via n<sub>u</sub>merical solver <sub>u</sub>sin<sub>g</sub> $\hat { \bf s } _ { n - 1 }$   
9: <sup>A</sup><sub>pp</sub>en<sup>d</sup> $\hat { \bf s } _ { n - 1 }$ to B and remove the oldest score   
10: Correct:   
11: E<sub>va</sub>l<sub>ua</sub>t<sub>e exac</sub>t <sub>score</sub> $\tilde { \mathbf { s } } _ { n - 2 } \gets s _ { \theta } ( \hat { \mathbf { x } } _ { n - 2 } )$   
12: E<sub>vo</sub>l<sub>ve</sub> t<sub>o</sub> $\tilde { \mathbf { x } } _ { n - 3 }$ via n<sub>u</sub>merical solver <sub>u</sub>sin<sub>g</sub> $\tilde { \bf s } _ { n - 2 }$   
13: <sup>A</sup><sub>pp</sub>en<sup>d</sup> $\tilde { \bf s } _ { n - 2 }$ to $\mathcal { B }$ <sub>an</sub>d <sub>remove</sub> th<sub>e o</sub>ld<sub>es</sub>t <sub>score</sub>   
14: end for   
15: return $\tilde { \mathbf { x } } _ { 0 }$

$$
\mathbf { H } _ { n } ^ { ( m ) } = \left[ \tilde { \mathbf { s } } _ { n + r - 1 } ^ { ( m ) } , \tilde { \mathbf { s } } _ { n + r - 2 } ^ { ( m ) } , \hdots , \tilde { \mathbf { s } } _ { n } ^ { ( m ) } \right] \in \mathbb { R } ^ { d \times r } .\tag{22}
$$

W<sub>e see</sub>k <sub>an op</sub>ti<sub>ma</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or</sub> ${ \mathbf w } _ { n } \in \mathbb { R } ^ { r }$ th<sub>a</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es:</sub>

$$
\tilde { \mathbf { s } } _ { n - 1 } ^ { ( m ) } \approx \mathbf { H } _ { n } ^ { ( m ) } \mathbf { w } _ { n } .\tag{23}
$$

T<sup>h</sup>e cross-trajectory consistency presente<sup>d</sup> in T<sup>h</sup>m. 3.1 <sup>d</sup>emonstrates t<sup>h</sup>at, a<sup>f</sup>ter proper a<sup>l</sup>ignment, <sup>d</sup>i<sup>f</sup>erent trajectories <sup>f</sup>o<sup>ll</sup>ow nearly isomorphic low-dimensional dynamics. Consequently, a single coeficient vector ${ \bf w } _ { n }$ <sub>can e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e score evo</sub>l<sub>u-</sub> tion o<sup>f</sup> <sup>d</sup>i<sup>f</sup>erent re<sup>f</sup>erence trajectories at a given samp<sup>l</sup>ing timestep, <sub>ena</sub>bli<sub>ng</sub> th<sub>e</sub> <sub>ex</sub>t<sub>rac</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub> <sub>un</sub>i<sub>versa</sub>l l<sub>oca</sub>l <sub>ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on</sub> l<sub>aw.</sub> F<sub>or</sub> mall<sub>y,</sub> we derive the ste<sub>p</sub>-wise o<sub>p</sub>timal <sub>p</sub>redictor within the famil<sub>y</sub> o<sup>f</sup> <sup>l</sup>inear extrapo<sup>l</sup>ators <sup>b</sup>y so<sup>l</sup>ving t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing mu<sup>l</sup>ti-trajectory l<sub>eas</sub>t<sub>-squares</sub> <sub>pro</sub>bl<sub>em:</sub>

$$
\mathbf { w } _ { n } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { r } } \sum _ { m = 1 } ^ { M } \left\| \tilde { \mathbf { s } } _ { n - 1 } ^ { ( m ) } - \mathbf { H } _ { n } ^ { ( m ) } \mathbf { w } \right\| _ { 2 } ^ { 2 } .\tag{24}
$$

Whil<sub>e</sub> th<sub>e</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on a</sub>b<sub>ove</sub> i<sub>s concep</sub>t<sub>ua</sub>ll<sub>y s</sub>t<sub>ra</sub>i<sub>g</sub>htf<sub>orwar</sub>d<sub>,</sub> directl<sub>y</sub> re<sub>g</sub>ressin<sub>g</sub> on $\left[ \tilde { \mathbf { s } } _ { n + r - 1 } , \tilde { \mathbf { s } } _ { n + r - 2 } , \ldots , \tilde { \mathbf { s } } _ { n } \right]$ i<sub>s o</sub>ft<sub>en numer</sub>i<sub>ca</sub>ll<sub>y</sub> ill-<sub>p</sub>osed. Since consecutive model out<sub>p</sub>uts alon<sub>g</sub> smooth se<sub>g</sub>ments <sub>o</sub>f th<sub>e</sub> dif<sub>us</sub>i<sub>on pa</sub>th <sub>are</sub> hi<sub>g</sub>hl<sub>y corre</sub>l<sub>a</sub>t<sub>e</sub>d<sub>,</sub> th<sub>e co</sub>l<sub>umns o</sub>f th<sub>e</sub> hi<sub>s</sub>t<sub>ory ma</sub>t<sub>r</sub>i<sub>x</sub> t<sub>en</sub>d t<sub>o</sub> b<sub>e near</sub>l<sub>y co</sub>lli<sub>near.</sub> A<sub>s a resu</sub>lt<sub>,</sub> th<sub>e assoc</sub>i<sub>-</sub> <sub>a</sub>t<sub>e</sub>d l<sub>eas</sub>t<sub>-squares</sub> <sub>sys</sub>t<sub>em</sub> b<sub>ecomes</sub> <sub>poor</sub>l<sub>y</sub> <sub>con</sub>diti<sub>one</sub>d<sub>,</sub> <sub>ma</sub>ki<sub>ng</sub> th<sub>e</sub> fitt<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>sens</sub>iti<sub>ve</sub> t<sub>o</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> <sub>error,</sub> i<sub>mper</sub>f<sub>ec</sub>t <sub>cross-</sub> trajectory a<sup>l</sup>ignment, an<sup>d</sup> numerica<sup>l</sup> insta<sup>b</sup>i<sup>l</sup>ities. To mitigate t<sup>h</sup>is issue, we reparameterize the local subspace using a diference basis rather than the raw basis. Specifically, for each trajectory window, <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> dif<sub>erences o</sub>f th<sub>e scores as:</sub>

$$
\Delta _ { n , j } ^ { ( m ) } : = \tilde { \mathbf { s } } _ { n + j } ^ { ( m ) } - \tilde { \mathbf { s } } _ { n + j - 1 } ^ { ( m ) } , \qquad j = 1 , \dotsc , r - 1 ,\tag{25}
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e mo</sub>difi<sub>e</sub>d hi<sub>s</sub>t<sub>ory ma</sub>t<sub>r</sub>i<sub>x as:</sub>

$$
\mathbf { B } _ { n } ^ { ( m ) } = \left[ \Delta _ { n , r - 1 } ^ { ( m ) } , \ldots , \Delta _ { n , 1 } ^ { ( m ) } , \tilde { \mathbf { s } } _ { n } ^ { ( m ) } \right] \in \mathbb { R } ^ { d \times r } .\tag{26}
$$

Rather than directl<sub>y</sub> estimatin<sub>g</sub> $\mathbf { w } _ { n } ,$ <sub>we</sub> fit <sub>an</sub> <sub>aux</sub>ili<sub>ary</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t vector ${ \mathbf c } _ { n } \in \mathbb { R } ^ { r }$ <sub>suc</sub>h th<sub>a</sub>t

$$
\tilde { \mathbf { s } } _ { n - 1 } ^ { ( m ) } \approx \mathbf { B } _ { n } ^ { ( m ) } \mathbf { c } _ { n } .\tag{27}
$$

In practice, we aggregate samp<sup>l</sup>es <sup>f</sup>rom a<sup>ll</sup> re<sup>f</sup>erence trajectories <sub>an</sub>d <sub>so</sub>l<sub>ve</sub> <sub>a</sub> <sub>r</sub>id<sub>ge</sub> <sub>regress</sub>i<sub>on</sub> <sub>pro</sub>bl<sub>em</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> dif<sub>erence</sub> b<sub>as</sub>i<sub>s</sub> f<sub>or</sub> each timeste<sub>p</sub> $n = N - 1 , N - 2 , . . . , r ;$

$$
\mathbf { c } _ { n } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { c } \in \mathbb { R } ^ { r } } \sum _ { m = 1 } ^ { M } \left\| \widetilde { \mathbf { s } } _ { n - 1 } ^ { ( m ) } - \mathbf { B } _ { n } ^ { ( m ) } \mathbf { c } \right\| _ { 2 } ^ { 2 } + \lambda \| \mathbf { c } \| _ { 2 } ^ { 2 } .\tag{28}
$$

O<sub>nce</sub> th<sub>e s</sub>t<sub>ep-w</sub>i<sub>se coe</sub>fi<sub>c</sub>i<sub>en</sub>t b<sub>an</sub>k $C = \{ \mathbf { c } _ { N - r + 1 } , \mathbf { c } _ { N - r } , \ldots , \mathbf { c } _ { 1 } \}$ h<sub>as</sub> b<sub>een so</sub>l<sub>ve</sub>d<sub>,</sub> i<sub>n</sub>f<sub>erence procee</sub>d<sub>s</sub> i<sub>n</sub> t<sub>wo p</sub>h<sub>ases.</sub> Fi<sub>rs</sub>t<sub>, we execu</sub>t<sub>e</sub> a warm-u<sub>p</sub> <sub>p</sub>hase b<sub>y</sub> runnin<sub>g</sub> the initial <sub>�</sub> denoisin<sub>g</sub> ste<sub>p</sub>s in a standard wa<sub>y</sub>. This initializes the histor<sub>y</sub> b<sub>u</sub>fer with the initial <sub>�</sub> <sub>mo</sub>d<sub>e</sub>l <sub>ou</sub>t<sub>pu</sub>t<sub>s an</sub>d th<sub>e</sub>i<sub>r correspon</sub>di<sub>ng s</sub>t<sub>a</sub>t<sub>es:</sub>

$$
\bigl \{ \tilde { \mathbf { s } } _ { N } , \tilde { \mathbf { s } } _ { N - 2 } , \ldots , \tilde { \mathbf { s } } _ { N - r + 1 } \bigr \} , \qquad \bigl \{ \tilde { \mathbf { x } } _ { N } , \tilde { \mathbf { x } } _ { N - 2 } , \ldots , \tilde { \mathbf { x } } _ { N - r } \bigr \} .\tag{29}
$$

This warm-u<sub>p</sub> <sub>p</sub>hase <sub>p</sub>rovides the minimum context re<sub>q</sub>uired b<sub>y</sub> the extra<sub>p</sub>olator. Movin<sub>g</sub> forward<sub>,</sub> we em<sub>p</sub>lo<sub>y</sub> an interleaved sam<sub>p</sub>lin<sub>g</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> th<sub>a</sub>t <sub>a</sub>lt<sub>erna</sub>t<sub>es</sub> b<sub>e</sub>t<sub>ween a ne</sub>t<sub>wor</sub>k<sub>-</sub>f<sub>ree pre</sub>di<sub>c</sub>ti<sub>on s</sub>t<sub>ep an</sub>d <sub>a</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>-</sub>b<sub>ase</sub>d <sub>correc</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ep.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> <sub>suppose</sub> th<sub>e</sub> <sub>curren</sub>t hi<sub>s</sub>t<sub>ory</sub> b<sub>u</sub>f<sub>er</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> <sub>�</sub> <sub>mos</sub>t <sub>recen</sub>t <sub>scores:</sub>

$$
\big \{ \tilde { \mathbf { s } } _ { n + r - 1 } , \tilde { \mathbf { s } } _ { n + r - 2 } , \hdots , \tilde { \mathbf { s } } _ { n } \big \} .\tag{30}
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>pre</sub>di<sub>c</sub>t th<sub>e</sub> <sub>nex</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>ou</sub>t<sub>pu</sub>t <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>earne</sub>d <sub>s</sub>t<sub>ep-w</sub>i<sub>se</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s:</sub>

$$
\hat { \mathbf { s } } _ { n - 1 } : = \mathbf { c } _ { n , r - 1 } \tilde { \mathbf { s } } _ { n } + \sum _ { j = 0 } ^ { r - 2 } \mathbf { c } _ { n , j } \Delta _ { n , r - j - 1 } .\tag{31}
$$

Th<sub>e ex</sub>t<sub>rapo</sub>l<sub>a</sub>t<sub>e</sub>d <sub>score</sub> $\hat { \bf s } _ { n - 1 }$ is t<sup>h</sup>en uti<sup>l</sup>ize<sup>d</sup> to evo<sup>l</sup>ve t<sup>h</sup>e trajectory to $\hat { \mathbf { x } } _ { n - 2 }$ via numerical solvers<sub>,</sub> eliminatin<sub>g</sub> the need for an ex<sub>p</sub>licit <sub>ne</sub>t<sub>wor</sub>k <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Th<sub>e ex</sub>t<sub>rapo</sub>l<sub>a</sub>t<sub>e</sub>d <sub>score</sub> $\hat { \mathsf { \pmb { s } } } _ { n - 1 }$ th<sub>en serves as a</sub> p<sup>rox</sup>y <sup>for</sup> $\tilde { \bf s } _ { n - 1 }$ <sub>an</sub>d i<sub>s pus</sub>h<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> b<sub>u</sub>f<sub>er,</sub> di<sub>sp</sub>l<sub>ac</sub>i<sub>ng</sub> th<sub>e</sub> <sub>o</sub>ld<sub>es</sub>t <sub>en</sub>t<sub>ry</sub> t<sub>o</sub> <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>s</sub>i<sub>ze</sub> <sub>o</sub>f <sub>�.</sub>

E<sub>x</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on a</sub>l<sub>one</sub> i<sub>nev</sub>it<sub>a</sub>bl<sub>y accumu</sub>l<sub>a</sub>t<sub>es numer</sub>i<sub>ca</sub>l d<sub>r</sub>ift <sub>over</sub> time. To ensure trajectory sta<sup>b</sup>i<sup>l</sup>ity, we imp<sup>l</sup>ement an inter<sup>l</sup>eave<sup>d</sup> Predict-Correct scheme. Followin<sub>g</sub> the short warm-u<sub>p</sub> sta<sub>g</sub>e<sub>,</sub> the <sub>so</sub>l<sub>ver</sub> <sub>a</sub>lt<sub>erna</sub>t<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>-</sub>f<sub>ree</sub> <sub>ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d <sub>exac</sub>t <sub>ne</sub>t<sub>wor</sub>k <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> b<sub>a</sub>l<sub>anc</sub>i<sub>ng compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>e</sub>fi<sub>c</sub>i<sub>ency w</sub>ith <sub>nu-</sub> merical robustness<sub>,</sub> <sub>p</sub>rovidin<sub>g</sub> a <sub>p</sub>ractical acceleration framework for difusion sam<sub>p</sub>lin<sub>g</sub>. Given the ste<sub>p</sub>-wise o<sub>p</sub>timalit<sub>y</sub> of our linear <sub>ex</sub>t<sub>rapo</sub>l<sub>a</sub>t<sub>ors an</sub>d th<sub>e</sub>i<sub>r capa</sub>bilit<sub>y</sub> t<sub>o</sub> b<sub>ypass re</sub>d<sub>un</sub>d<sub>an</sub>t <sub>ne</sub>t<sub>wor</sub>k evaluations, we name our method as Optimal Linear Leaps (LEAP). Th<sub>e</sub> f<sub>u</sub>ll <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>o</sub>f <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t b<sub>an</sub>k <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>an</sub>d LEAP <sub>sam-</sub> <sub>p</sub>lin<sub>g</sub> are <sub>p</sub>rovided in Al<sub>g</sub>orithm 1 and 2.

## 4 Experiments

We evaluate the <sub>p</sub>ro<sub>p</sub>osed LEAP on three cuttin<sub>g</sub>-ed<sub>g</sub>e difusion models: SD3.5-Medium [8] and FLUX.1-Dev [2] for text-ima<sub>g</sub>e task and TRELLIS.2 [41] for ima<sub>g</sub>e-3D task. The <sub>q</sub>ualit<sub>y</sub> of <sub>g</sub>enerated ima<sub>g</sub>es is <sub>q</sub>uantitativel<sub>y</sub> evaluated usin<sub>g</sub> three metrics<sub>,</sub> includin<sub>g</sub> Ima<sub>g</sub>eReward [42], CLIP Score [23], and HPSv2 [40]. The <sub>p</sub>rom<sub>p</sub>ts are drawn from the DrawBench [27] dataset. For ima<sub>g</sub>e-3D task, we eva<sup>l</sup>uate 3D o<sup>b</sup>ject on a test set o<sup>f</sup> 100 ran<sup>d</sup>om<sup>l</sup>y samp<sup>l</sup>e<sup>d</sup> images from the Goo<sub>g</sub>le Scanned Objects (GSO) dataset [7] b<sub>y</sub> PSNR and CLIP Score. B default, we fix a window size of � = 5 and set the rid<sub>g</sub>e re<sub>g</sub>ularization coeficient to � = 0.0001 across all ex<sub>p</sub>eriments. Unl<sub>ess o</sub>th<sub>e</sub>r<sub>w</sub>i<sub>se spec</sub>ifi<sub>e</sub>d<sub>,</sub> LEAP i<sub>s co</sub>m<sub>pa</sub>r<sub>e</sub>d <sub>w</sub>ith <sub>e</sub>xi<sub>s</sub>tin<sub>g s</sub>t<sub>a</sub>t<sub>e</sub>- <sub>o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>so</sub>l<sub>vers un</sub>d<sub>er</sub> id<sub>en</sub>ti<sub>ca</sub>l i<sub>n</sub>f<sub>erence se</sub>tti<sub>ngs.</sub>

## 4.1 Main Results

Text-to-image Generation. We evaluate LEAP on text-to-image <sub>g</sub>eneration<sub>,</sub> benchmarkin<sub>g</sub> it a<sub>g</sub>ainst re<sub>p</sub>resentative solvers includin<sub>g</sub> Euler method [30], DPM++ [19], UniPC [46], iPNDM [35], LTC-Accel [50] and STORK-4 [34]. As shown in Tab. 1, LEAP consistentl<sub>y</sub> achieves to<sub>p</sub>-tier <sub>p</sub>erformance across diverse sam<sub>p</sub>lin<sub>g</sub> bud<sub>g</sub>ets<sub>,</sub> efectivel<sub>y p</sub>reservin<sub>g</sub> both text-ima<sub>g</sub>e ali<sub>g</sub>nment and <sub>p</sub>erce<sub>p</sub>tual <sub>q</sub>ualit<sub>y</sub> for accelerated sam<sub>p</sub>lin<sub>g</sub>. Moreover<sub>,</sub> LEAP consistentl<sub>y</sub> enh<sub>ances</sub> th<sub>e per</sub>f<sub>ormance o</sub>fb<sub>o</sub>th fi<sub>rs</sub>t<sub>-or</sub>d<sub>er an</sub>d hi<sub>g</sub>h<sub>er-or</sub>d<sub>er so</sub>l<sub>vers,</sub> demonstratin<sub>g</sub> its versatile <sub>p</sub>lu<sub>g</sub>-and-<sub>p</sub>la<sub>y</sub> ca<sub>p</sub>abilit<sub>y</sub>. B<sub>y</sub> levera<sub>g</sub>- in<sub>g</sub> a <sub>p</sub>re-com<sub>p</sub>uted<sub>,</sub> ste<sub>p</sub>-wise coeficient bank<sub>,</sub> LEAP seamlessl<sub>y</sub> ada<sub>p</sub>ts an<sub>y</sub> solver without an<sub>y</sub> solver-s<sub>p</sub>ecific h<sub>yp</sub>er<sub>p</sub>arameter tunin<sub>g</sub> or model fine-tunin<sub>g,</sub> makin<sub>g</sub> it a solver-a<sub>g</sub>nostic enhancement m<sub>o</sub>d<sub>u</sub>l<sub>e.</sub> Fi<sub>g.</sub> 4 <sub>p</sub>r<sub>ov</sub>id<sub>es a qua</sub>lit<sub>a</sub>ti<sub>ve co</sub>m<sub>pa</sub>ri<sub>so</sub>n <sub>o</sub>f LEAP <sub>aga</sub>in<sub>s</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub>-<sub>o</sub>f-th<sub>e</sub>-<sub>a</sub>rt <sub>so</sub>l<sub>ve</sub>r<sub>s o</sub>n SD3<sub>.</sub>5-M<sub>e</sub>di<sub>u</sub>m<sub>.</sub> O<sub>u</sub>r LEAP <sub>e</sub>xhibit<sub>s su</sub>- <sub>p</sub>erior <sub>p</sub>erformance across four challen<sub>g</sub>in<sub>g</sub> scenarios includin<sub>g</sub> (1) text renderin<sub>g</sub>: LEAP accuratel<sub>y</sub> s<sub>y</sub>nthesizes le<sub>g</sub>ible text<sub>,</sub> whereas baselines sufer from t<sub>yp</sub>o<sub>g</sub>ra<sub>p</sub>hic errors; (2) <sub>p</sub>h<sub>y</sub>sical li<sub>g</sub>htin<sub>g</sub>: LEAP ca<sub>p</sub>tures realistic trans<sub>p</sub>arenc<sub>y</sub> and intricate li<sub>g</sub>ht refraction; (3) at trib<sub>u</sub>te bindin<sub>g</sub>: LEAP <sub>p</sub>recisel<sub>y</sub> binds the localized attrib<sub>u</sub>tes to their res<sub>p</sub>ective <sub>p</sub>artitions without semantic leaka<sub>g</sub>e.

Image-to-3D Generation. We further validate LEAP on imageto-3D <sub>g</sub>eneration usin<sub>g</sub> the TRELLIS.2 [41] architecture. As <sub>p</sub>resented in Tab. 2<sub>,</sub> o<sub>u</sub>r method achieves s<sub>y</sub>stematic im<sub>p</sub>rovements in b<sub>o</sub>th PSNR <sub>a</sub>nd CLIP S<sub>co</sub>r<sub>e ove</sub>r th<sub>e</sub> b<sub>ase</sub>lin<sub>e o</sub>n GSO d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> Th<sub>e</sub> <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub> r<sub>esu</sub>lt<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>g.</sub> 5 d<sub>e</sub>m<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>t<sub>e</sub> LEAP’<sub>s</sub> <sub>supe</sub>- rior ca<sub>p</sub>abilit<sub>y</sub> in (1) to<sub>p</sub>olo<sub>g</sub>ical correctness, maintainin<sub>g</sub> <sub>p</sub>recise connectivit<sub>y</sub> in com<sub>p</sub>lex jointed structures; (2) surface fidelit<sub>y</sub>; (3) <sub>g</sub>eometric com<sub>p</sub>leteness<sub>,</sub> efectivel<sub>y</sub> <sub>g</sub>eneratin<sub>g</sub> intricate and hi<sub>g</sub>hfre<sub>q</sub>uenc<sub>y</sub> <sub>g</sub>eometries<sub>,</sub> <sub>y</sub>ieldin<sub>g</sub> shar<sub>p</sub>er textural details and su<sub>p</sub>erior co<sup>l</sup>or accurac<sub>y</sub>.

## 4.2 Ablation Study

T<sub>o sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y va</sub>lid<sub>a</sub>t<sub>e</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>eac</sub>h k<sub>ey</sub> d<sub>es</sub>i<sub>gn</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>n</sub> th<sub>e</sub> <sub>propose</sub>d <sub>me</sub>th<sub>o</sub>d<sub>,</sub> <sub>we</sub> <sub>con</sub>d<sub>uc</sub>t <sub>compre</sub>h<sub>ens</sub>i<sub>ve</sub> <sub>a</sub>b<sub>-</sub> l<sub>a</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>u</sub>di<sub>es</sub> <sub>o</sub>n SD3<sub>.</sub>5-M<sub>e</sub>di<sub>u</sub>m<sub>.</sub> T<sub>o</sub> <sub>e</sub>n<sub>su</sub>r<sub>e</sub> <sub>a</sub> ri<sub>go</sub>r<sub>ous</sub> <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n<sub>,</sub> we isolate the im<sub>p</sub>act of each com<sub>p</sub>onent b<sub>y</sub> maintainin<sub>g</sub> a fixed �<sub>-s</sub>t<sub>ep</sub> <sub>samp</sub>li<sub>ng</sub> <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e,</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>ran</sub>d<sub>om</sub> <sub>see</sub>d<sub>s</sub> <sub>an</sub>d <sub>a</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t ensemble of reference trajectory (default � = 32). (a) Window Size �. As shown in Fig 6a, we adopt � = 5 as the default window size<sub>,</sub> as it delivers <sub>p</sub>eak or near-<sub>p</sub>eak <sub>p</sub>erformance in most scenarios. (b) Penalty Strength �. Fig. 6b demonstrates that LEAP is robust t<sub>o</sub> th<sub>e</sub> <sub>pena</sub>lt<sub>y</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t � <sub>across</sub> <sub>a</sub> <sub>w</sub>id<sub>e</sub> <sub>range</sub> <sub>o</sub>f <sub>va</sub>l<sub>ues.</sub> W<sub>e</sub> <sub>re-</sub> tain this <sub>p</sub>enalt<sub>y</sub> term across all ex<sub>p</sub>eriments to ensure theoretical consistency and numerical stability. (c) Number of Reference Trajectories �. As shown in Fig 6c, LEAP exhibits remarkable data eficienc<sub>y,</sub> maintainin<sub>g</sub> hi<sub>g</sub>h <sub>p</sub>erformance even with minimal reference data (� ≤ 2). This observation <sub>p</sub>erfectl<sub>y</sub> corroborates our t<sup>h</sup>eoretica<sup>l</sup> ana<sup>l</sup>ysis regar<sup>d</sup>ing cross-trajectory consistency. We <sup>d</sup>efault to � = 32 to ensure statisticall<sub>y</sub> robust covera<sub>g</sub>e across diverse prompts. (d) Basis Construction. Fig. 7 underscores the critical <sub>ro</sub>l<sub>e</sub> <sub>o</sub>f <sub>our</sub> dif<sub>erence-</sub>b<sub>as</sub>i<sub>s</sub> <sub>reparame</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on.</sub> Th<sub>e</sub> dif<sub>erence</sub> b<sub>as</sub>i<sub>s</sub> efectivel<sub>y</sub> rectifies the ill-conditionin<sub>g,</sub> ensurin<sub>g</sub> numerical stabilit<sub>y</sub> and enablin<sub>g</sub> consistent <sub>q</sub>ualit<sub>y</sub> im<sub>p</sub>rovements across all NFE sett<sup>i</sup>n<sub>g</sub>s.

![](images/503d2b170e89498b7e089d63944a3f3d0c5b71bd2b335f903a01b4a9911a64d4.jpg)

Figure 4: Text-to-image generation on SD3.5-Medium using 10 inference steps across diverse prompts.  
![](images/890baca4fd0f99585223fd18ab685f94861dd6ca90290e46aa64237503339186.jpg)  
Figure 5: Image-to-3D generation on TRELLIS.2 using 10 inference steps across diferent image prompts.

Table 1: Quantitative results on SD3.5-Medium and FLUX.1-dev. The best and second-best results are marked in bold and underlined, respectively. Our proposed method (LEAP) is highlighted with a gray background.
<table><tr><td rowspan="2">Steps</td><td rowspan="2">Method</td><td colspan="3">SD3.5-Medium</td><td colspan="3">FLUX.1-dev</td></tr><tr><td>HPSv2 ↑</td><td>ImageReward ↑</td><td>CLIPScore ↑</td><td>HPSv2 ↑</td><td>ImageReward ↑</td><td>CLIPScore ↑</td></tr><tr><td rowspan="7">10</td><td>Euler</td><td>25.8866</td><td>0.6653</td><td>32.7920</td><td>28.3538</td><td>0.8110</td><td>30.6939</td></tr><tr><td>DPM++</td><td>26.4565</td><td>0.7178</td><td>32.9027</td><td>28.6365</td><td>0.8331</td><td>30.7257</td></tr><tr><td>iPNDM</td><td>26.4041</td><td>0.7063</td><td>32.7579</td><td>28.8272</td><td>0.8956</td><td>30.9178</td></tr><tr><td>UniPC</td><td>26.4450</td><td>0.7122</td><td>32.6428</td><td>28.8193</td><td>0.8839</td><td>31.0634</td></tr><tr><td>LTC-Accel</td><td>26.2546</td><td>0.7324</td><td>33.1578</td><td>28.9543</td><td>0.9193</td><td>31.1217</td></tr><tr><td>STORK-4</td><td>26.0567</td><td>0.6619</td><td>32.8704</td><td>29.2846</td><td>0.9729</td><td>31.2895</td></tr><tr><td>LEAP (Euler)</td><td>26.8419</td><td>0.7692</td><td>33.2982</td><td>29.0536</td><td>0.9204</td><td>30.9412</td></tr><tr><td></td><td>LEAP (DPM++)</td><td>27.0237</td><td>0.8177</td><td>33.2850</td><td>29.4004</td><td>0.9816</td><td>31.4320</td></tr><tr><td rowspan="7">15</td><td>Euler</td><td>27.4725</td><td>0.8416</td><td>33.4644</td><td>29.2580</td><td>0.9470</td><td>31.0488</td></tr><tr><td>DPM++</td><td>27.8642</td><td>0.8438</td><td>33.4704</td><td>29.5030</td><td>0.9588</td><td>31.2782</td></tr><tr><td>iPNDM</td><td>27.9220</td><td>0.8770</td><td>33.4416</td><td>29.5056</td><td>0.9491</td><td>31.3836</td></tr><tr><td>UniPC</td><td>27.8680</td><td>0.8788</td><td>33.4703</td><td>29.4668</td><td>0.9414</td><td>31.2339</td></tr><tr><td>LTC-Accel</td><td>27.8174</td><td>0.9196</td><td>33.7184</td><td>29.5874</td><td>0.9576</td><td>31.1906</td></tr><tr><td>STORK-4</td><td>27.4224</td><td>0.8253</td><td>33.5179</td><td>29.9693</td><td>1.0308</td><td>31.3041</td></tr><tr><td>LEAP (Euler)</td><td>28.1826</td><td>0.9522</td><td>33.6817</td><td>29.6785</td><td>0.9719</td><td>31.1046</td></tr><tr><td></td><td>LEAP (DPM++)</td><td>28.2152</td><td>0.9806</td><td>33.5617</td><td>29.9024</td><td>1.0101</td><td>31.4981</td></tr><tr><td rowspan="7">20</td><td>Euler</td><td>28.2691</td><td>0.9201</td><td>33.6945</td><td>29.4556</td><td>0.9398</td><td>30.9759</td></tr><tr><td>DPM++</td><td>28.4878</td><td>0.9560</td><td>33.7391</td><td>29.6523</td><td>0.9460</td><td>31.0277</td></tr><tr><td>iPNDM</td><td>28.4252</td><td>0.9317</td><td>33.9325</td><td>29.6963</td><td>0.9886</td><td>31.2203</td></tr><tr><td>UniPC</td><td>28.4756</td><td>0.9364</td><td>33.7757</td><td>29.6773</td><td>0.9840</td><td>31.1858</td></tr><tr><td>LTC-Accel</td><td>28.3403</td><td>0.9547</td><td>33.7901</td><td>29.7326</td><td>0.9884</td><td>31.1142</td></tr><tr><td>STORK-4</td><td>28.1835</td><td>0.9035</td><td>33.8360</td><td>29.7927</td><td>1.0066</td><td>31.2650</td></tr><tr><td>LEAP (Euler)</td><td>28.5727</td><td>0.9701</td><td>33.7192</td><td>29.9370</td><td>1.0320</td><td>31.3433</td></tr><tr><td></td><td>LEAP (DPM++)</td><td>28.6259</td><td>0.9915</td><td>33.5356</td><td>30.1514</td><td>1.0514</td><td>31.6484</td></tr></table>

Table 2: Quantitative comparison of image-to-3D generation on the GSO dataset via TRELLIS.2.
<table><tr><td rowspan="2">Method</td><td colspan="5">PSNR ↑</td><td colspan="5">CLIP Score ↑</td></tr><tr><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td></tr><tr><td>Euler</td><td>7.250</td><td>7.396</td><td>7.235</td><td>7.361</td><td>7.405</td><td>0.8638</td><td>0.8654</td><td>0.8682</td><td>0.8650</td><td>0.8658</td></tr><tr><td>LEAP (Euler)</td><td>7.414</td><td>7.537</td><td>7.337</td><td>7.410</td><td>7.408</td><td>0.8677</td><td>0.8674</td><td>0.8693</td><td>0.8678</td><td>0.8709</td></tr></table>

## 5 Related Works

Fast numerical solvers. A predominant branch of training-free <sub>acce</sub>l<sub>era</sub>ti<sub>on</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>ocuses</sub> <sub>on</sub> i<sub>mprov</sub>i<sub>ng</sub> th<sub>e</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>so</sub>l<sub>ver</sub> em<sub>p</sub>lo<sub>y</sub>ed in reverse-time sam<sub>p</sub>lin<sub>g</sub>. Earl<sub>y</sub> work such as DDIM [30] <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d th<sub>e</sub> ODE<sub>-</sub>b<sub>ase</sub>d <sub>para</sub>di<sub>gm</sub> f<sub>or</sub> f<sub>as</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c gener-</sub> ation. Subse<sub>q</sub>uent research has develo<sub>p</sub>ed increasin<sub>g</sub>l<sub>y</sub> efective hi<sub>g</sub>h-order and multi-ste<sub>p</sub> solvers. PNDM [16] introduced multiste<sub>p</sub> <sub>p</sub>seudo numerical methods<sub>,</sub> showin<sub>g</sub> that levera<sub>g</sub>in<sub>g</sub> historical sam<sub>p</sub>les can si<sub>g</sub>nificantl<sub>y</sub> im<sub>p</sub>rove the <sub>q</sub>ualit<sub>y</sub>-s<sub>p</sub>eed trade-of over first-order sam<sub>p</sub>lers. Its im<sub>p</sub>roved im<sub>p</sub>lementation, iPNDM [35], is fre<sub>q</sub>uentl<sub>y</sub> utilized in modern difusion sam<sub>p</sub>lin<sub>g</sub>. DEIS [45] further <sub>exp</sub>l<sub>o</sub>it<sub>e</sub>d th<sub>e sem</sub>i<sub>-</sub>li<sub>near s</sub>t<sub>ruc</sub>t<sub>ure o</sub>f dif<sub>us</sub>i<sub>on</sub> d<sub>ynam</sub>i<sub>cs</sub> th<sub>roug</sub>h an ex<sub>p</sub>onential-inte<sub>g</sub>rator formulation. Similarl<sub>y</sub>, DPM-Solver [18] and DPM++ [19] derived dedicated hi<sub>g</sub>h-order solvers b<sub>y</sub> anal<sub>y</sub>ticall<sub>y</sub> isolatin<sub>g</sub> the linear com<sub>p</sub>onents<sub>,</sub> si<sub>g</sub>nificantl<sub>y</sub> reducin<sub>g</sub> dis-<sub>cre</sub>ti<sub>za</sub>ti<sub>on errors, an</sub>d h<sub>ave emerge</sub>d <sub>as a ro</sub>b<sub>us</sub>t <sub>prac</sub>ti<sub>ca</sub>l <sub>s</sub>t<sub>an-</sub> dard, <sub>p</sub>articularl<sub>y</sub> for <sub>g</sub>uided sam<sub>p</sub>lin<sub>g</sub> scenarios. UniPC [46] unified th<sub>e pre</sub>di<sub>c</sub>t<sub>or-correc</sub>t<sub>or para</sub>di<sub>gm.</sub> S<sub>u</sub>b<sub>sequen</sub>t <sub>re</sub>fi<sub>nemen</sub>t<sub>s suc</sub>h <sub>as</sub> DPM-Solver-v3 [48] and DC-Solver [47], further im<sub>p</sub>roved solver <sub>p</sub>arameterization and correction stabilit<sub>y</sub>. LTC-Accel [50] levera<sub>g</sub>es t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup> transition co<sup>h</sup>erence o<sup>f</sup> generative trajectories to <sup>d</sup>ynamica<sup>ll</sup>y approximate a<sup>d</sup>jacent networ<sup>k</sup> outputs, ac<sup>h</sup>ieving trainingfree acceleration. STORK [34] addressed numerical stifness and removed the reliance on semi-linear assum<sub>p</sub>tions and <sub>p</sub>ro<sub>p</sub>osed a <sub>s</sub>t<sub>a</sub>bili<sub>ze</sub>d <sub>so</sub>l<sub>ver.</sub> D<sub>esp</sub>it<sub>e</sub> th<sub>e</sub>i<sub>r</sub> dif<sub>erences,</sub> th<sub>ese me</sub>th<sub>o</sub>d<sub>s s</sub>till <sub>con</sub> struct u<sub>p</sub>date rules <sub>p</sub>rimaril<sub>y</sub> from numerical anal<sub>y</sub>sis or ste<sub>p</sub>-wise trend heuristics. In contrast<sub>,</sub> our method revisits multi-ste<sub>p</sub> solvin<sub>g</sub> <sup>f</sup>rom a trajectory-geometric perspective an<sup>d</sup> see<sup>k</sup>s to exp<sup>l</sup>oit t<sup>h</sup>e intrinsic <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup> an<sup>d</sup> cross-trajectory consistent structure o<sup>f</sup> generative trajectories more <sup>d</sup>irect<sup>l</sup>y.

![](images/e75ebd1549e5e3818ee8ddf720dedbf8051b97ad9ad518827ce479c77a598b11.jpg)  
(a) Window Size (�)

![](images/703178789c0a6972628c15b100fcd61ef80fcf88b2a149451612f3b51e4cda5d.jpg)  
(b) Penalty Strength (�)

![](images/b1c1f905570d74ac09d7f724ad40f955d0e00671acb84d9c1a37b7fc5ab1e2cf.jpg)  
(c) Number of Reference Trajectories (�)

Figure 6: Ablation studies on key hyperparameters in Stable Difusion 3.5-Medium. We evaluate the impact of (a) Window Size, (b) Penalty Strength, and (c) Number of Tokens on the HPSv2 score across diferent NFEs. The default configuration is indicated by the dashed vertical line.  
![](images/f706f87a08a1c90a02e47c643bc5ad53154138a7aea5dc88292be8c8d7c358de.jpg)  
Figure 7: Ablation study on basis construction in Stable Diffusion 3.5-Medium. Our proposed diferent basis efectively addresses the issue of ill-conditioning.

Difusion sampling dynamics. Prior research on difusion samp<sup>l</sup>ing <sup>d</sup>ynamics <sup>h</sup>as main<sup>l</sup>y esta<sup>bl</sup>is<sup>h</sup>e<sup>d</sup> trajectory regu<sup>l</sup>arity throu<sub>g</sub>h em<sub>p</sub>irical observations<sub>,</sub> su<sub>pp</sub>orted b<sub>y</sub> <sub>p</sub>reliminar<sub>y</sub> theoretical insi<sub>g</sub>hts under idealized settin<sub>g</sub>s. Notabl<sub>y</sub>, Wan<sub>g</sub> et al. [37] o<sup>b</sup>serve<sup>d</sup> t<sup>h</sup>at in<sup>d</sup>ivi<sup>d</sup>ua<sup>l</sup> reverse-time samp<sup>l</sup>ing trajectories are <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l<sub>, o</sub>ft<sub>en resem</sub>bli<sub>ng</sub> t<sub>wo-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>ro-</sub> tations. Chen et al. [5] studied deterministic PF-ODE trajectories, <sup>fi</sup>n<sup>d</sup>ing t<sup>h</sup>at eac<sup>h</sup> trajectory c<sup>l</sup>ose<sup>l</sup>y <sup>f</sup>o<sup>ll</sup>ows t<sup>h</sup>e <sup>d</sup>isp<sup>l</sup>acement <sup>b</sup>et<sub>ween</sub> it<sub>s</sub> <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s,</sub> <sub>w</sub>ith <sub>res</sub>id<sub>ua</sub>l<sub>s</sub> <sub>nea</sub>tl<sub>y</sub> <sub>cap</sub>t<sub>ure</sub>d b<sub>y</sub> t<sub>wo</sub> <sub>or</sub> th<sub>ree</sub> ortho<sub>g</sub>onal directions<sub>,</sub> <sub>y</sub>ieldin<sub>g</sub> a characteristic boomeran<sub>g</sub>-like sha<sub>p</sub>e. Their subse<sub>q</sub>uent work [4] stren<sub>g</sub>thened the cross-trajector<sub>y</sub> consistency c<sup>l</sup>aim <sup>b</sup>y a<sup>l</sup>igning projecte<sup>d</sup> trajectories wit<sup>h</sup> ort<sup>h</sup>ogona<sup>l</sup> Procrustes, revea<sup>l</sup>ing a near<sup>l</sup>y canonica<sup>l</sup> trajectory s<sup>h</sup>ape a<sup>f</sup>- t<sub>er samp</sub>l<sub>e-spec</sub>ifi<sub>c ro</sub>t<sub>a</sub>ti<sub>ons an</sub>d <sub>re</sub>fl<sub>ec</sub>ti<sub>ons.</sub> Whil<sub>e</sub> th<sub>ese wor</sub>k<sub>s</sub> <sup>h</sup>ave empirica<sup>ll</sup>y uncovere<sup>d</sup> t<sup>h</sup>e trajectory regu<sup>l</sup>arity o<sup>f d</sup>i<sup>f</sup>usion sam<sub>p</sub>lin<sub>g</sub> d<sub>y</sub>namics<sub>,</sub> their theoretical <sub>g</sub>roundin<sub>g</sub> relies lar<sub>g</sub>el<sub>y</sub> on KDE-based surro<sub>g</sub>ates of the em<sub>p</sub>irical data distribution<sub>,</sub> anal<sub>y</sub>ticall<sub>y</sub> tractable Ga<sub>u</sub>ssian models<sub>,</sub> and s<sub>upp</sub>ortive Ga<sub>u</sub>ssian-mixt<sub>u</sub>re <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons.</sub> T<sub>o</sub> th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>f <sub>our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge,</sub> <sub>pr</sub>i<sub>or</sub> lit<sub>era</sub>t<sub>ure</sub> l<sub>ac</sub>k<sub>s</sub> <sub>a un</sub>ifi<sub>e</sub>d th<sub>eorem w</sub>ithi<sub>n a</sub> Mi<sub>x</sub>t<sub>ure-o</sub>f<sub>-</sub>G<sub>auss</sub>i<sub>an</sub> f<sub>ramewor</sub>k th<sub>a</sub>t joint<sup>l</sup>y <sup>f</sup>orma<sup>l</sup>izes <sup>b</sup>ot<sup>h</sup> trajectory <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity an<sup>d</sup> crosstrajectory isomorp<sup>h</sup>ism. In contrast, our wor<sup>k</sup> provi<sup>d</sup>es a princip<sup>l</sup>e<sup>d</sup> d<sub>er</sub>i<sub>va</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>ese</sub> t<sub>wo proper</sub>ti<sub>es.</sub>

## 6 Conclusion

In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we introd<sub>u</sub>ced LEAP<sub>,</sub> a novel<sub>,</sub> trainin<sub>g</sub>-free m<sub>u</sub>lti-ste<sub>p</sub> ODE <sub>so</sub>l<sub>ver</sub> th<sub>a</sub>t <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y a</sub>dd<sub>resses</sub> th<sub>e per</sub>f<sub>ormance</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> of difusion models in low-NFE re<sub>g</sub>imes. Our a<sub>pp</sub>roach is built u<sub>p</sub>on a ri<sub>g</sub>orous investi<sub>g</sub>ation of the <sub>g</sub>eometric <sub>p</sub>ro<sub>p</sub>erties of difusion samp<sup>l</sup>ing trajectories. By proving t<sup>h</sup>e intrinsic <sup>l</sup>ow-<sup>d</sup>imensiona<sup>l</sup>ity an<sup>d</sup> cross-trajectory consistency o<sup>f</sup> generative pat<sup>h</sup>s, we <sup>f</sup>ormu<sup>l</sup>ate<sup>d</sup> a supervise<sup>d l</sup>east-squares optimization wit<sup>h</sup>in t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup> trajectory s<sub>u</sub>bs<sub>p</sub>ace. This allows LEAP to derive o<sub>p</sub>timal<sub>,</sub> sam<sub>p</sub>le-a<sub>g</sub>nostic extra<sub>p</sub>olation coeficients<sub>,</sub> achievin<sub>g</sub> accurate future state a<sub>pp</sub>roximati<sub>o</sub>n<sub>s w</sub>ith minim<sub>a</sub>l <sub>co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>ove</sub>rh<sub>ea</sub>d<sub>.</sub> Ext<sub>e</sub>n<sub>s</sub>i<sub>ve eva</sub>l<sub>ua</sub>ti<sub>o</sub>n<sub>s</sub> d<sub>e</sub>m<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>t<sub>e</sub> th<sub>a</sub>t LEAP <sub>se</sub>r<sub>ves as a ve</sub>r<sub>sa</sub>til<sub>e, p</sub>l<sub>ug</sub>-<sub>a</sub>nd-<sub>p</sub>l<sub>ay e</sub>nh<sub>a</sub>n<sub>ce</sub>- <sub>men</sub>t f<sub>or</sub> <sub>mo</sub>d<sub>ern</sub> l<sub>arge-sca</sub>l<sub>e</sub> dif<sub>us</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> LEAP <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orms s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t b<sub>ase</sub>li<sub>nes across</sub> di<sub>verse mo</sub>d<sub>a</sub>liti<sub>es,</sub> includin<sub>g</sub> text-to-ima<sub>g</sub>e <sub>g</sub>eneration usin<sub>g</sub> Stable Difusion 3.5 and FLUX<sub>.</sub>1-D<sub>ev</sub> <sub>a</sub>nd im<sub>age</sub>-t<sub>o</sub>-3D <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>us</sub>in<sub>g</sub> TRELLIS<sub>.</sub>2<sub>.</sub> F<sub>u</sub>t<sub>u</sub>r<sub>e</sub> <sub>wor</sub>k <sub>w</sub>ill d<sub>e</sub>l<sub>ve</sub> d<sub>eeper</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>proper</sub>ti<sub>es</sub> <sub>o</sub>f dif<sub>us</sub>i<sub>on</sub> samp<sup>l</sup>ing trajectories un<sup>d</sup>er practica<sup>l</sup> settings to <sup>f</sup>urt<sup>h</sup>er a<sup>l</sup>ign our theoretical framework with realistic<sub>,</sub> finite-ste<sub>p</sub> sam<sub>p</sub>lin<sub>g</sub> scenarios.

## Acknowledgments

This work is su<sub>pp</sub>orted b<sub>y</sub> the National Natural Science Foundation of China (No. 62476244), the Starr<sub>y</sub> Ni<sub>g</sub>ht Science Fund of Zhejian<sub>g</sub> Universit<sub>y</sub> Shan<sub>g</sub>hai Institute for Advanced Stud<sub>y</sub>, China (Grant No: SN-ZJU-SIAS-001) and the advanced com<sub>p</sub>utin<sub>g</sub> resources <sub>p</sub>rovided b<sub>y</sub> the Su<sub>p</sub>ercom<sub>p</sub>utin<sub>g</sub> Center of Han<sub>g</sub>zhou Cit<sub>y</sub> Universit<sub>y</sub>.

## References

[1] Michael S Alber<sub>g</sub>o and Eric Vanden-Eijnden. 2022. Buildin<sub>g</sub> normalizin<sub>g</sub> flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 (2022).

[2] Black Forest Labs. 2024. FLUX.1 [dev]. htt<sub>p</sub>s://<sub>g</sub>ithub.com/black-forest-labs/flux.

[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kili<sub>a</sub>n<sub>,</sub> D<sub>o</sub>minik L<sub>o</sub>r<sub>e</sub>nz<sub>,</sub> Y<sub>a</sub>m L<sub>ev</sub>i<sub>,</sub> Zi<sub>o</sub>n En<sub>g</sub>li<sub>s</sub>h<sub>,</sub> Vikr<sub>a</sub>m V<sub>o</sub>l<sub>e</sub>ti<sub>,</sub> Ad<sub>a</sub>m L<sub>e</sub>tt<sub>s,</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2023<sub>.</sub> St<sub>a</sub>bl<sub>e v</sub>id<sub>eo</sub> dif<sub>us</sub>i<sub>on:</sub> S<sub>ca</sub>li<sub>ng</sub> l<sub>a</sub>t<sub>en</sub>t <sub>v</sub>id<sub>eo</sub> dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>o</sub> l<sub>arge</sub> datasets. arXiv preprint arXiv:2311.15127 (2023).

[4] Defan<sub>g</sub> Chen, Zhen<sub>y</sub>u Zhou, Can Wan<sub>g</sub>, and Siwei L<sub>y</sub>u. 2025. Geometric re<sub>g</sub>ularit<sub>y</sub> in deterministic sam<sub>p</sub>lin<sub>g</sub> d<sub>y</sub>namics of difusion-based <sub>g</sub>enerative models. Journal ofStatistical Mechanics: Theory and Experiment 2025, 12 (2025), 124002.

[5] Defan<sub>g</sub> Chen, Zhen<sub>y</sub>u Zhou, Can Wan<sub>g</sub>, Chunhua Shen, and Siwei L<sub>y</sub>u. 2024. On the Trajector Re ularit of ODE-based Difusion Sam lin . In International Conference on Machine Learning. 7905–7934.

[6] Nanxin Chen, Yu Zhan<sub>g</sub>, Hei<sub>g</sub>a Zen, Ron J Weiss, Mohammad Norouzi, and Willi<sub>a</sub>m Ch<sub>a</sub>n<sub>.</sub> 2020<sub>.</sub> W<sub>ave</sub> r<sub>a</sub>d: E<sub>s</sub>tim<sub>a</sub>tin r<sub>a</sub>di<sub>e</sub>nt<sub>s</sub> f<sub>o</sub>r <sub>wave</sub>f<sub>o</sub>rm <sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> arXiv preprint arXiv:2009.00713 (2020).

[7] Laura Downs, Anthon<sub>y</sub> Francis, Nate Koeni<sub>g</sub>, Brandon Kinman, R<sub>y</sub>an Hickman, Kri<sub>s</sub>t<sub>a</sub> R<sub>ey</sub>m<sub>a</sub>nn<sub>,</sub> Th<sub>o</sub>m<sub>as</sub> B M<sub>c</sub>H<sub>ug</sub>h<sub>, a</sub>nd Vin<sub>ce</sub>nt V<sub>a</sub>nh<sub>ouc</sub>k<sub>e.</sub> 2022<sub>.</sub> G<sub>oog</sub>l<sub>e</sub> scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA). Ieee, 2553–2560.

[8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, H<sub>a</sub>rr<sub>y</sub> S<sub>a</sub>ini<sub>,</sub> Y<sub>a</sub>m L<sub>ev</sub>i<sub>,</sub> D<sub>o</sub>minik L<sub>o</sub>r<sub>e</sub>nz<sub>,</sub> Ax<sub>e</sub>l S<sub>aue</sub>r<sub>,</sub> Fr<sub>e</sub>d<sub>e</sub>ri<sub>c</sub> B<sub>oese</sub>l<sub>,</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2024<sub>.</sub> Scaling rectified flow transformers for high-resolution image synthesis. In Fortyfirst international conference on machine learning.

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

[10] Jonathan Ho, Tim Salimans, Alexe<sub>y</sub> Gritsenko, William Chan, Mohammad Norouzi and David J Fleet. 2022. Video difusion models. Advances in neural information processing systems 35 (2022), 8633–8646.

[11] Team Hun<sub>y</sub>uan3D, Shuhui Yan<sub>g</sub>, Min<sub>g</sub>xin Yan<sub>g</sub>, Yifei Fen<sub>g</sub>, Xin Huan<sub>g</sub>, Shen<sub>g</sub> Zh<sub>a</sub>n<sub>g,</sub> Z<sub>e</sub>bin H<sub>e,</sub> Di L<sub>uo,</sub> H<sub>ao</sub>lin Li<sub>u,</sub> Y<sub>u</sub>nf<sub>e</sub>i Zh<sub>ao, e</sub>t <sub>a</sub>l<sub>.</sub> 2025<sub>.</sub> H<sub>u</sub>n<sub>yua</sub>n3d 2<sub>.</sub>1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442 (2025).

[12] Don<sub>g</sub>jun Kim, Chieh-Hsin Lai, Wei-Hsian<sub>g</sub> Liao, Naoki Murata, Yuhta Takida, Tos<sup>h</sup>imitsu Uesa<sup>k</sup>a, Yutong He, Yu<sup>k</sup>i Mitsu<sup>f</sup>uji, an<sup>d</sup> Ste<sup>f</sup>ano Ermon. 2023. Consis tency trajectory mo<sup>d</sup>e<sup>l</sup>s: Learning pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity <sup>fl</sup>ow o<sup>d</sup>e trajectory o<sup>f d</sup>i<sup>f</sup>usion. arXiv preprint arXiv:2310.02279 (2023).

[13] Zhifen<sub>g</sub> Kon<sub>g</sub>, Wei Pin<sub>g</sub>, Jiaji Huan<sub>g</sub>, Kexin Zhao, and Br<sub>y</sub>an Catanzaro. 2020. Difwave: A versatile difusion model for audio synthesis. arXiv preprint arXiv:2009.09761 (2020).

[14] Senmao Li, Taihan<sub>g</sub> Hu, Joost van de Weijer, Fahad S Khan, Tao Liu, Linxuan Li, Shi<sub>q</sub>i Yan<sub>g</sub>, Yaxin<sub>g</sub> Wan<sub>g</sub>, Min<sub>g</sub>-Min<sub>g</sub> Chen<sub>g</sub>, andJian Yan<sub>g</sub>. 2024. Faster difusion: Rethinkin the role of the encoder for difusion model inference. Advances in Neural Information Processing Systems 37 (2024), 85203–85240.

[15] Yaron Li<sub>p</sub>man, Rick<sub>y</sub> TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022).

[16] Lu in Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. 2022. Pseudo numerical methods for difusion models on manifolds. arXiv preprint arXiv:2202.09778 (2022).

[17] Xin<sub>g</sub>chao Liu, Chen<sub>gy</sub>ue Gon<sub>g</sub>, and Qian<sub>g</sub> Liu. 2022. Flow strai<sub>g</sub>ht and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022).

[18] Chen<sub>g</sub> Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chon<sub>g</sub>xuan Li, and Jun Zhu. 2022<sub>.</sub> D<sub>pm-so</sub>l<sub>ver:</sub> A f<sub>as</sub>t <sub>o</sub>d<sub>e</sub> <sub>so</sub>l<sub>ver</sub> f<sub>or</sub> dif<sub>us</sub>i<sub>on</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> <sub>mo</sub>d<sub>e</sub>l <sub>samp</sub>li<sub>ng</sub> in around 10 steps. Advances in neural information processing systems 35 (2022), 5775–5787.

[19] Chen<sub>g</sub> Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chon<sub>g</sub>xuan Li, and Jun Zhu. 2025. D<sub>pm-so</sub>l<sub>ver</sub>++<sub>:</sub> F<sub>as</sub>t <sub>so</sub>l<sub>ver</sub> f<sub>or gu</sub>id<sub>e</sub>d <sub>samp</sub>li<sub>ng o</sub>f dif<sub>us</sub>i<sub>on pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c mo</sub>d<sub>e</sub>l<sub>s.</sub> Machine Intelligence Research 22, 4 (2025), 730–751.

[20] Simian Luo, Yi<sub>q</sub>in Tan, Lon<sub>g</sub>bo Huan<sub>g</sub>, Jian Li, and Han<sub>g</sub> Zhao. 2023. Latent consistenc<sub>y</sub> models: S<sub>y</sub>nthesizin<sub>g</sub> hi<sub>g</sub>h-resolution ima<sub>g</sub>es with few-ste<sub>p</sub> inference. arXiv preprint arXiv:2310.04378 (2023).

[21] Xin<sub>y</sub>in Ma, Gon<sub>g</sub>fan Fan<sub>g</sub>, and Xinchao Wan<sub>g</sub>. 2024. Dee<sub>p</sub>cache: Acceleratin<sub>g</sub> difusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 15762–15772.

[22] Bernt Oksendal. 2013. Stochastic diferential equations: an introduction with applications. Springer Science & Business Media.

[23] Alec Radford, Jon<sub>g</sub> Wook Kim, Chris Hallac<sub>y</sub>, Adit<sub>y</sub>a Ramesh, Gabriel Goh, Sandhini A<sub>g</sub>arwal, Girish Sastr<sub>y</sub>, Amanda Askell, Pamela Mishkin, Jack Clark, <sub>e</sub>t <sub>a</sub>l<sub>.</sub> 2021<sub>.</sub> L<sub>earn</sub>i<sub>ng</sub> t<sub>rans</sub>f<sub>era</sub>bl<sub>e v</sub>i<sub>sua</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>rom na</sub>t<sub>ura</sub>l l<sub>anguage superv</sub>i<sub>s</sub>i<sub>on.</sub> In International conference on machine learning. PmLR, 8748–8763.

[24] Adit<sub>y</sub>a Ramesh, Prafulla Dhariwal, Alex Nichol, Case<sub>y</sub> Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.

[25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Hi<sub>g</sub>h-resolution ima<sub>g</sub>e s<sub>y</sub>nthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

[26] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. 2024. Align your steps: Optimizing sampling schedules in difusion models. arXiv preprint arXiv:2404.14507 (2024).

[27] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Ja<sub>y</sub> Whan<sub>g</sub>, Emil<sub>y</sub> L Denton, Kamyar G<sup>h</sup>asemipour, Rap<sup>h</sup>ae<sup>l</sup> Gontijo Lopes, Burcu Karago<sup>l</sup> Ayan, Tim Salimans<sub>,</sub> et al. 2022. Photorealistic text-to-ima<sub>g</sub>e difusion models with dee<sub>p</sub> language understanding. Advances in neural information processing systems 35 (2022), 36479–36494.

[28] Tim Salimans and Jonathan Ho. 2022. Pro<sub>g</sub>ressive distillation for fast sam<sub>p</sub>lin<sub>g</sub> of difusion models. arXiv preprint arXiv:2202.00512 (2022).

[29] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Sur<sub>y</sub>a Gan<sub>g</sub>uli. 2015. Dee<sub>p</sub> unsu<sub>p</sub>ervised learnin<sub>g</sub> usin<sub>g</sub> none<sub>q</sub>uilibrium thermod<sub>y</sub>namics. In International conference on machine learning. pmlr, 2256–2265.

[30] Jiamin<sub>g</sub> Son<sub>g</sub>, Chenlin Men<sub>g</sub>, and Stefano Ermon. 2020. Denoisin<sub>g</sub> difusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

[31] Yan<sub>g</sub> Son<sub>g</sub>, Prafulla Dhariwal, Mark Chen, and Il<sub>y</sub>a Sutskever. 2023. Consistenc<sub>y</sub> models. (2023).

[32] Yan<sub>g</sub> Son<sub>g</sub> and Stefano Ermon. 2020. Im<sub>p</sub>roved techni<sub>q</sub>ues for trainin<sub>g</sub> scorebased generative models. Advances in neural information processing systems 33 (2020), 12438–12448.

[33] Yan<sub>g</sub> Son<sub>g</sub>, Jascha Sohl-Dickstein, Diederik P Kin<sub>g</sub>ma, Abhishek Kumar, Stefano Ermon<sub>,</sub> and Ben Poole. 2020. Score-based <sub>g</sub>enerative modelin<sub>g</sub> throu<sub>g</sub>h stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020).

[34] Zhen<sub>g</sub> Tan, Weizhen Wan<sub>g</sub>, Andrea L Bertozzi, and Ernest K R<sub>y</sub>u. 2025. STORK: Faster Dif<sub>u</sub>sion And Flo<sub>w</sub> Matchin<sub>g</sub> Sam<sub>p</sub>lin<sub>g</sub> B<sub>y</sub> Resol<sub>v</sub>in<sub>g</sub> Both Stifness And Structure-Dependence. arXiv preprint arXiv:2505.24210 (2025).

[35] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kas<sup>h</sup>i<sup>f</sup> Rasu<sup>l</sup>, Mis<sup>h</sup>ig Davaa<sup>d</sup>orj, an<sup>d</sup> T<sup>h</sup>omas Wo<sup>lf</sup>. 2022. Di<sup>f</sup>users: State-o<sup>f</sup>-t<sup>h</sup>eart difusion models. htt<sub>p</sub>s://<sub>g</sub>ithub.com/hu<sub>gg</sub>in<sub>g</sub>face/difusers.

[36] Zhen<sub>g</sub>chao Wan, Qin<sub>g</sub>son<sub>g</sub> Wan<sub>g</sub>, Gal Mishne, and Yusu Wan<sub>g</sub>. 2025. Elucidatin<sub>g</sub> flow matching ODE dynamics via data geometry and denoisers. In Forty-second International Conference on Machine Learning.

[37] Binxu Wan<sub>g</sub> and John J Vastola. 2023. Difusion models <sub>g</sub>enerate ima<sub>g</sub>es like painters: an analytical theory of outline first, details later. arXiv preprint arXiv:2303.02490 (2023).

[38] Guan<sub>gy</sub>i Wan<sub>g</sub>, Wei Pen<sub>g</sub>, Lijian<sub>g</sub> Li, Wen<sub>y</sub>u Chen, Yuren Cai, and Son<sub>g</sub>zhi Su. 2024. Difusion Sampling Correction via Approximately 10 Parameters. arXiv preprint arXiv:2411.06503 (2024).

[39] Felix Wimbauer, Bichen Wu, Ed<sub>g</sub>ar Schoenfeld, Xiaolian<sub>g</sub> Dai, Ji Hou, Zijian He, Artsiom Sanako<sub>y</sub>eu, Peizhao Zhan<sub>g</sub>, Sam Tsai, Jonas Kohler, et al. 2024. Cache me if you can: Accelerating difusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6211–6220.

[40] Xiaoshi Wu, Yimin<sub>g</sub> Hao, Ke<sub>q</sub>ian<sub>g</sub> Sun, Yixion<sub>g</sub> Chen, Fen<sub>g</sub> Zhu, Rui Zhao, and H<sub>ongs</sub>h<sub>eng</sub> Li<sub>.</sub> 2023<sub>.</sub> H<sub>uman pre</sub>f<sub>erence score v</sub>2<sub>:</sub> A <sub>so</sub>lid b<sub>enc</sub>h<sub>mar</sub>k f<sub>or eva</sub>l<sub>ua</sub>t<sub>-</sub> ing human preferences oftext-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023).

[41] Jianfen<sub>g</sub> Xian<sub>g</sub>, Xiaoxue Chen, Sichen<sub>g</sub> Xu, Ruichen<sub>g</sub> Wan<sub>g</sub>, Zelon<sub>g</sub> Lv, Yu Den<sub>g</sub>, Hon<sub>gy</sub>uan Zhu, Yue Don<sub>g</sub>, Hao Zhao, Nicholas Jin<sub>g</sub> Yuan, et al. 2025. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692 (2025).

[42] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Don<sub>g</sub>. 2023. Ima<sub>g</sub>ereward: Learnin<sub>g</sub> and evaluatin<sub>g</sub> human <sub>p</sub>references for text-to-image generation. Advances in Neural Information Processing Systems 36 (2023), 15903–15935.

[43] Shuchen Xue, Zhao<sub>q</sub>ian<sub>g</sub> Liu, Fei Chen, Shifen<sub>g</sub> Zhan<sub>g</sub>, Tian<sub>y</sub>an<sub>g</sub> Hu, Enze Xie, and Zhen<sub>g</sub>uo Li. 2024. Acceleratin<sub>g</sub> difusion sam<sub>p</sub>lin<sub>g</sub> with o<sub>p</sub>timized time steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8292–8301.

[44] Zichao Yu, Zhen Zou, Guojian<sub>g</sub> Shao, Chenwei Zhan<sub>g</sub>, Shen<sub>g</sub>ze Xu, Jie Huan<sub>g</sub>, F<sub>e</sub>n<sub>g</sub> Zh<sub>ao,</sub> Xi<sub>ao</sub>d<sub>o</sub>n<sub>g</sub> C<sub>u</sub>n<sub>, a</sub>nd W<sub>e</sub>n<sub>y</sub>i Zh<sub>a</sub>n<sub>g.</sub> 2025<sub>.</sub> Ab-<sub>cac</sub>h<sub>e</sub>: Tr<sub>a</sub>inin<sub>g</sub>-fr<sub>ee</sub> <sub>acce</sub>l<sub>era</sub>ti<sub>on o</sub>f dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s v</sub>i<sub>a a</sub>d<sub>ams-</sub>b<sub>as</sub>hf<sub>or</sub>th <sub>cac</sub>h<sub>e</sub>d f<sub>ea</sub>t<sub>ure reuse.</sub> I<sub>n</sub> Proceedings ofthe 33rd ACMInternational Conference on Multimedia. 10408–10417.

[45] Qinshen<sub>g</sub> Zhan<sub>g</sub> and Yon<sub>g</sub>xin Chen. 2022. Fast sam<sub>p</sub>lin<sub>g</sub> of difusion models with exponential integrator. arXiv preprint arXiv:2204.13902 (2022).

[46] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. 2023. Unipc: A <sub>un</sub>ifi<sub>e</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>or-correc</sub>t<sub>or</sub> f<sub>ramewor</sub>k f<sub>or</sub> f<sub>as</sub>t <sub>samp</sub>li<sub>ng o</sub>f dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s.</sub> Advances in Neural Information Processing Systems 36 (2023), 49842–49869.

[47] Wenlian<sub>g</sub> Zhao, Haolin Wan<sub>g</sub>, Jie Zhou, and Jiwen Lu. 2024. Dc-solver: Im<sub>p</sub>rovin<sub>g</sub> predictor-corrector difusion sampler via dynamic compensation. In European Conference on Computer Vision. Springer, 450–466.

[48] Kaiwen Zhen<sub>g</sub>, Chen<sub>g</sub> Lu, Jianfei Chen, and Jun Zhu. 2023. D<sub>p</sub>m-solver-v3: Improved difusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems 36 (2023), 55502–55542.

[49] Zhen<sub>y</sub>u Zhou, Defan<sub>g</sub> Chen, Can Wan<sub>g</sub>, and Chun Chen. 2024. Fast ode-based sampling for difusion models in around 5 steps. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7777–7786.

[50] Shangwen Zhu, Han Zhang, Zhantao Yang, Qianyu Peng, Zhao Pu, Huangji Wan<sub>g,</sub> and Fan Chen<sub>g</sub>. 2025. Acceleratin<sub>g</sub> Dif<sub>u</sub>sion Sam<sub>p</sub>lin<sub>g</sub> <sub>v</sub>ia Ex<sub>p</sub>loitin<sub>g</sub> Local Transition Coherence. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 18284–18293.

## A Proofs for Section 3.1

This a<sub>pp</sub>endix section <sub>p</sub>ro<sub>v</sub>es Theorems and Corollaries from Secti<sub>o</sub>n 3<sub>.</sub>1<sub>.</sub> Fir<sub>s</sub>t<sub>,</sub> <sub>we</sub> <sub>p</sub>r<sub>ov</sub>id<sub>e</sub> <sub>a</sub>n <sub>e</sub>x<sub>ac</sub>t d<sub>eco</sub>m<sub>pos</sub>iti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> PF-ODE under a mild assum<sub>p</sub>tion. Next<sub>,</sub> we <sub>p</sub>rove the deterministic <sub>p</sub>airwise ali<sub>g</sub>nment bound. Finall<sub>y,</sub> we a<sub>pp</sub>l<sub>y</sub> the Gaussian concentration th<sub>eorem</sub> t<sub>o</sub> <sub>ge</sub>t <sub>a</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y</sub> b<sub>oun</sub>d<sub>.</sub>

## A.1 Notations

Consider the variance-<sub>p</sub>reservin<sub>g</sub> forward difusion situation:

$$
d \mathbf { X } _ { t } = - \frac { 1 } { 2 } \beta ( t ) \mathbf { X } _ { t } d t + \sqrt { \beta ( t ) } d \mathbf { W } _ { t } , \qquad t \in [ 0 , T ] ,\tag{32}
$$

<sub>w</sub>h<sub>ere</sub> $\beta : [ 0 , T ]  ( 0 , \infty )$ i<sub>s</sub> i<sub>n</sub>t<sub>egra</sub>bl<sub>e.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\alpha _ { t } : = \exp \Big ( - \frac { 1 } { 2 } \int _ { 0 } ^ { t } \beta ( s ) d s \Big ) , \qquad \nu _ { t } ^ { 2 } : = \alpha _ { t } ^ { 2 } \sigma _ { 0 } ^ { 2 } + ( 1 - \alpha _ { t } ^ { 2 } ) .\tag{33}
$$

S<sub>upp</sub>ose data distrib<sub>u</sub>tion satisfies GMM ass<sub>u</sub>m<sub>p</sub>tion:

$$
p _ { 0 } ( { \bf x } ) = \sum _ { m = 1 } ^ { M } \pi _ { m } { \cal N } ( \mu _ { m } , \sigma _ { 0 } ^ { 2 } { \bf I } _ { d } ) , \qquad \pi _ { m } > 0 , \qquad \sum _ { m = 1 } ^ { M } \pi _ { m } = 1 .\tag{34}
$$

T<sub>o</sub> d<sub>escr</sub>ib<sub>e</sub> th<sub>e</sub> l<sub>ayou</sub>t <sub>o</sub>f th<sub>e m</sub>i<sub>xe</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on cen</sub>t<sub>er, we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e space</sub> f<sub>orme</sub>d b<sub>y</sub> th<sub>e</sub>i<sub>r means as</sub>

$$
U : = \mathrm { s p a n } \{ \pmb { \mu } _ { 1 } , . . . , \pmb { \mu } _ { M } \} , \qquad r : = \mathrm { d i m } ( U ) , \qquad R : = \operatorname* { m a x } _ { 1 \leq m \leq M } \| \pmb { \mu } _ { m } \| .\tag{35}
$$

We a<sup>l</sup>so <sup>l</sup>et � to <sup>b</sup>e t<sup>h</sup>e ort<sup>h</sup>ogona<sup>l</sup> projection onto �, wit<sup>h</sup> $\boldsymbol { Q } : = \mathbf { I } - \boldsymbol { P } ,$

## A.2 Structural lemmas

We now be<sub>g</sub>in to derive the closed-form ex<sub>p</sub>ressions of PF-ODE d<sub>y</sub>- namics. We start our <sub>p</sub>roofs b<sub>y</sub> formulatin<sub>g</sub> the mar<sub>g</sub>inal <sub>p</sub>robabilit<sub>y</sub> d<sub>ens</sub>it<sub>y o</sub>f th<sub>e</sub> f<sub>orwar</sub>d dif<sub>us</sub>i<sub>on.</sub>

Lemma A.1 (Closed Form of Marginal Probability Function). For every $t \in [ 0 , T ]$ , we have

$$
{ p } _ { t } ( { \bf x } ) = \sum _ { s = 1 } ^ { S } \pi _ { s } N ( \alpha _ { t } { \pmb \mu } _ { s } , \nu _ { t } ^ { 2 } { \bf I } _ { d } ) .\tag{36}
$$

<sup>P</sup>roof. <sup>S</sup>u<sub>pp</sub>ose $C \in \{ 1 , . . . , S \}$ satisf<sub>y</sub> condition $\mathbb { P } ( C = s ) = \pi _ { s }$ and su<sub>pp</sub>ose the conditional <sub>p</sub>robabilit<sub>y</sub> densit<sub>y</sub> of the initial state i<sub>s</sub>

$$
p ( \mathbf { x } _ { 0 } \mid C = s ) = N ( \mathbf { x } _ { 0 } ; { \pmb \mu } _ { s } , \sigma _ { 0 } ^ { 2 } \mathbf { I } _ { d } ) .\tag{37}
$$

B<sub>ecause</sub> f<sub>orwar</sub>d dif<sub>us</sub>i<sub>on</sub> <sub>process</sub> d<sub>e</sub>fi<sub>nes</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> densit<sub>y</sub>:

$$
p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = N ( \mathbf { x } _ { t } ; \alpha _ { t } \mathbf { x } _ { 0 } , ( 1 - \alpha _ { t } ^ { 2 } ) \mathbf { I } _ { d } ) .\tag{38}
$$

H<sub>e</sub>n<sub>ce,</sub> <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>o</sub>n $C = s ,$ the probability density ofx<sub>�</sub> is obtained b<sub>y</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng equa</sub>ti<sub>on:</sub>

$$
\begin{array} { l } { p ( \mathbf { x } _ { t } \mid C = s ) = \displaystyle \int p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } \mid C = s ) \mathrm { d } \mathbf { x } _ { 0 } } \\ { \displaystyle = N \big ( \mathbf { x } _ { t } ; \alpha _ { t } \pmb { \mu } _ { s } , ( \alpha _ { t } ^ { 2 } \sigma _ { 0 } ^ { 2 } + 1 - \alpha _ { t } ^ { 2 } ) \mathbf { I } _ { d } \big ) } \\ { \displaystyle = N ( \mathbf { x } _ { t } ; \alpha _ { t } \pmb { \mu } _ { s } , \nu _ { t } ^ { 2 } \mathbf { I } _ { d } ) . } \end{array}\tag{39}
$$

Avera<sub>g</sub>in<sub>g</sub> over <sub>�</sub> <sub>g</sub>ives the claim.

Lemma A.2 (Closed-form of score function). Define

$$
w _ { s } ( \mathbf { x } , t ) : = \frac { \pi _ { s } \exp \Big ( \frac { \alpha _ { t } } { \nu _ { t } ^ { 2 } } \langle \mathbf { x } , \pmb { \mu } _ { s } \rangle - \frac { \alpha _ { t } ^ { 2 } } { 2 \nu _ { t } ^ { 2 } } \| \pmb { \mu } _ { s } \| ^ { 2 } \Big ) } { \sum _ { j = 1 } ^ { S } \pi _ { j } \exp \Big ( \frac { \alpha _ { t } } { \nu _ { t } ^ { 2 } } \langle \mathbf { x } , \pmb { \mu } _ { j } \rangle - \frac { \alpha _ { t } ^ { 2 } } { 2 \nu _ { t } ^ { 2 } } \| \pmb { \mu } _ { j } \| ^ { 2 } \Big ) } ,\tag{40}
$$

and

$$
\mathbf { m } _ { t } ( \mathbf { x } ) : = \sum _ { s = 1 } ^ { S } w _ { s } ( \mathbf { x } , t ) \pmb { \mu } _ { s } .
$$

Then

(41)

$$
\nabla \log { p _ { t } ( \mathbf { x } ) } = - \frac { 1 } { \nu _ { t } ^ { 2 } } \big ( \mathbf { x } - \alpha _ { t } \mathbf { m } _ { t } ( \mathbf { x } ) \big ) .\tag{42}
$$

Proof<sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>1<sub>,</sub>

$$
p _ { t } ( \mathbf { x } ) = \sum _ { s = 1 } ^ { S } \pi _ { s } \phi _ { \nu _ { t } ^ { 2 } \mathbf { I } _ { d } } ( \mathbf { x } - \alpha _ { t } \pmb { \mu } _ { s } ) ,\tag{43}
$$

<sub>w</sub>h<sub>ere</sub> $\phi _ { \Sigma }$ denotes the Gaussian density with covariance Σ. Theref<sub>ore,</sub>

$$
\nabla p _ { t } ( \mathbf { x } ) = - \frac { 1 } { \nu _ { t } ^ { 2 } } \sum _ { s = 1 } ^ { S } \pi _ { s } ( \mathbf { x } - \alpha _ { t } \pmb { \mu } _ { s } ) \phi _ { \nu _ { t } ^ { 2 } \mathbf { I } _ { d } } ( \mathbf { x } - \alpha _ { t } \pmb { \mu } _ { s } ) .\tag{44}
$$

Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm o</sub>f <sub>score</sub> f<sub>unc</sub>ti<sub>on can</sub> b<sub>e</sub> d<sub>er</sub>i<sub>ve</sub>d b<sub>y</sub> dividin<sub>g</sub> E<sub>q</sub>. 44 b<sub>y</sub> $p _ { t } ( \mathbf { x } )$ :

$$
\nabla \log \mathcal { P } _ { t } ( \mathbf { x } ) = - \frac { 1 } { \nu _ { t } ^ { 2 } } \Big ( \mathbf { x } - \alpha _ { t } \sum _ { s = 1 } ^ { S } w _ { s } ( \mathbf { x } , t ) \mu _ { s } \Big ) = - \frac { 1 } { \nu _ { t } ^ { 2 } } \big ( \mathbf { x } - \alpha _ { t } \mathbf { m } _ { t } ( \mathbf { x } ) \big ) .\tag{45}
$$

Lemma A.3 (Sufficiency of Mean Subspace). For every x ∈ R<sup>�</sup> and $t \in [ 0 , T ]$ ，

$$
w _ { s } ( \mathbf { x } , t ) = w _ { s } ( P \mathbf { x } , t ) , \qquad \mathbf { m } _ { t } ( \mathbf { x } ) = \mathbf { m } _ { t } ( P \mathbf { x } ) \in U .\tag{46}
$$

Furthermore, let $\mathbf { X } _ { t \in [ 0 , T ] }$ be a PF-ODE trajectory. Its derivative with respect to � is given by

$$
\frac { d { \mathbf x } _ { t } } { d t } = a _ { t } { \mathbf x } _ { t } + b _ { t } \mathbf m _ { t } ( P { \mathbf x } _ { t } ) ,\tag{47}
$$

where

$$
a _ { t } : = \frac { 1 } { 2 } \beta ( t ) \Big ( 1 - \nu _ { t } ^ { - 2 } \Big ) , \qquad b _ { t } : = \frac { 1 } { 2 } \beta ( t ) \alpha _ { t } \nu _ { t } ^ { - 2 } .\tag{48}
$$

Proof. Note that the ex<sub>p</sub>ression for $w _ { s } ( \mathbf { x } , t )$ only depends on x th<sub>roug</sub>h th<sub>e</sub> i<sub>nner pro</sub>d<sub>uc</sub>t<sub>s</sub> $\langle \mathbf { x } , \pmb { \mu } _ { s } \rangle$ <sub>.</sub> Sin<sub>ce</sub> f<sub>o</sub>r $\pmb { \mu } _ { s } \in U _ { : }$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r } { \langle \mathbf { x } , \pmb { \mu } _ { s } \rangle = \langle P \mathbf { x } , \pmb { \mu } _ { s } \rangle , } \end{array}\tag{49}
$$

<sub>w</sub>hi<sub>c</sub>h l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub> $w _ { s } ( \mathbf { x } , t ) = w _ { s } ( P \mathbf { x } , t )$ <sub>.</sub> F<sub>u</sub>rth<sub>e</sub>rm<sub>o</sub>r<sub>e,</sub> it i<sub>s o</sub>b<sub>v</sub>i<sub>ous</sub> th<sub>a</sub>t $\mathbf { m } _ { t } ( x ) \in U$ because it is sim<sub>p</sub>l<sub>y</sub> a sum of vectors $\mu _ { s } \in U$

$$
\mathbf { m } _ { t } ( \mathbf { x } ) = \sum _ { s = 1 } ^ { S } w _ { s } ( \mathbf { x } , t ) \pmb { \mu } _ { s } .\tag{50}
$$

In f<sub>o</sub>r<sub>wa</sub>rd <sub>p</sub>r<sub>ocess,</sub> th<sub>e</sub> PF-ODE i<sub>s</sub>

$$
\begin{array} { r l } & { \displaystyle \frac { d \mathbf { x } _ { t } } { d t } = - \frac { 1 } { 2 } \beta ( t ) \mathbf { x } _ { t } - \frac { 1 } { 2 } \beta ( t ) \nabla \log p _ { t } ( \mathbf { x } _ { t } ) } \\ & { \displaystyle \qquad = \frac { 1 } { 2 } \beta ( t ) \Big [ \mathbf { x } _ { t } - \nu _ { t } ^ { - 2 } \mathbf { x } _ { t } + \alpha _ { t } \nu _ { t } ^ { - 2 } \mathbf { m } _ { t } ( \mathbf { x } _ { t } ) \Big ] . } \end{array}\tag{51}
$$

Usin<sub>g</sub> E<sub>q</sub>. 46 immediatel<sub>y</sub> <sub>y</sub>ields E<sub>q</sub>.47.

Lemma A.4 (Exact � ⊕ $U ^ { \perp }$ decomposition). Let

$$
\Phi \big ( t _ { 2 } , t _ { 1 } \big ) : = \exp \Big ( \int _ { t _ { 1 } } ^ { t _ { 2 } } a _ { v } d v \Big ) , \qquad \rho _ { t } : = \Phi ( t , T ) .\tag{52}
$$

Then we obtain the following equation by integrating from � to �:

$$
Q { \bf x } _ { t } = \rho _ { t } Q { \bf x } _ { T } ,\tag{53}
$$

and

$$
P \mathbf { x } _ { t } = \rho _ { t } P \mathbf { x } _ { T } - \int _ { t } ^ { T } \Phi ( t , \tau ) b _ { \tau } \mathbf { m } _ { \tau } ( P \mathbf { x } _ { \tau } ) d \tau .\tag{54}
$$

As a result,

$$
\begin{array} { r } { Q \mathbf { D } _ { t } = ( \rho _ { t } - 1 ) Q \mathbf { x } _ { T } . } \end{array}\tag{55}
$$

Proof. Apply � to Eq. 47. Since m $( P \mathbf { x } _ { t } ) \in U$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
Q \frac { d { \bf x } _ { t } } { d t } = a _ { t } Q { \bf x } _ { t } ,\tag{56}
$$

<sub>w</sub>h<sub>ose exac</sub>t <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> i<sub>s</sub>

$$
Q { \bf x } _ { t } = \exp \Big ( \int _ { T } ^ { t } a _ { v } d v \Big ) Q { \bf x } _ { T } = \rho _ { t } Q { \bf x } _ { T } .\tag{57}
$$

Simi<sup>l</sup>ar<sup>l</sup>y, <sup>f</sup>or t<sup>h</sup>e projection $P ,$ <sub>we</sub> <sub>can</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>a</sub>t

$$
P \frac { d \mathbf { x } _ { t } } { d t } = a _ { t } P \mathbf { x } _ { t } + b _ { t } \mathbf { m } _ { t } ( P \mathbf { x } _ { t } ) ,\tag{58}
$$

and the variation-of-constants formula (inte<sub>g</sub>ratin<sub>g</sub> from � to �) l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub>

$$
P \mathbf { x } _ { t } = \rho _ { t } P \mathbf { x } _ { T } + \int _ { T } ^ { t } \Phi ( t , \tau ) b _ { \tau } \mathbf { m } _ { \tau } ( P \mathbf { x } _ { \tau } ) d \tau .\tag{59}
$$

Subtracting �x<sub>�</sub> immediately gives the Eq. 55.

Lemma A.5 (Mean-Subspace Displacement Bound). Define

$$
A _ { T } : = \operatorname* { s u p } _ { 0 \leq t \leq T } | \rho _ { t } - 1 | , \qquad B _ { T } : = \operatorname* { s u p } _ { 0 \leq t \leq T } \int _ { t } ^ { T } \Phi ( t , \tau ) | b _ { \tau } | d \tau .\tag{60}
$$

Then every trajectory satisfies

$$
\operatorname* { s u p } _ { 0 \leq t \leq T } \| P \mathbf { D } _ { t } \| \leq A _ { T } \| P \mathbf { x } _ { T } \| + B _ { T } R .\tag{61}
$$

Proof<sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>4<sub>,</sub>

$$
P \mathbf { D } _ { t } = ( \rho _ { t } - 1 ) P \mathbf { x } _ { T } - \int _ { t } ^ { T } \Phi ( t , \tau ) b _ { \tau } \mathbf { m } _ { \tau } ( P \mathbf { x } _ { \tau } ) d \tau .\tag{62}
$$

Sin<sub>ce</sub> $\mathbf { m } _ { \tau } ( P \mathbf { x } _ { \tau } )$ i<sub>s a co</sub>n<sub>ve</sub>x <sub>co</sub>mbin<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f $\pmb { \mu } _ { 1 } , \ldots , \pmb { \mu } _ { M }$ <sub>,</sub> it’<sub>s o</sub>b<sub>v</sub>i<sub>ous</sub> th<sub>a</sub>t

$$
\left\| \mathbf { m } _ { \tau } ( P \mathbf { x } _ { \tau } ) \right\| \leq R .\tag{63}
$$

Hence b<sub>y</sub> usin<sub>g</sub> the trian<sub>g</sub>le ine<sub>q</sub>ualit<sub>y,</sub> we have

$$
\| P \mathbf { D } _ { t } \| \leq | \rho _ { t } - 1 | \| P \mathbf { x } _ { T } \| + \int _ { t } ^ { T } \Phi ( t , \tau ) | b _ { \tau } | R d \tau .\tag{64}
$$

Takin<sub>g</sub> the su<sub>p</sub>remum over $t \in [ 0 , T ]$ <sub>proves</sub> th<sub>e</sub> l<sub>emma.</sub>

Lemma A.6 (Minimum orthogonal alignment error). For any ${ \mathbf { u } } , { \mathbf { v } } \in \mathbb { R } ^ { m }$ ,

$$
\operatorname* { m i n } _ { \mathbf { H } \in O ( m ) } \| \mathbf { u } - \mathbf { H } \mathbf { v } \| = { \Big | } \| \mathbf { u } \| - \| \mathbf { v } \| { \Big | } .\tag{65}
$$

Proof. For any H $\in O ( m )$ <sub>,</sub> the ine<sub>q</sub>ualit<sub>y</sub>

$$
\| \mathbf { u } - \mathbf { H } \mathbf { v } \| \geq \left| \| \mathbf { u } \| - \| \mathbf { H } \mathbf { v } \| \right| = \left| \| \mathbf { u } \| - \| \mathbf { v } \| \right|\tag{66}
$$

<sub>a</sub>l<sub>ways</sub> h<sub>o</sub>ld<sub>s.</sub> C<sub>onverse</sub>l<sub>y,</sub> if $\mathbf { u } , \mathbf { v } \neq 0 ,$ choose H $\in O ( m )$ <sub>suc</sub>h th<sub>a</sub>t

$$
\mathbf { H } { \frac { \mathbf { v } } { \lVert \mathbf { v } \rVert } } = { \frac { \mathbf { u } } { \lVert \mathbf { u } \rVert } } .\tag{67}
$$

Th<sub>en</sub> $\mathbf { H } \mathbf { v } = ( \lVert \mathbf { v } \rVert / \lVert \mathbf { u } \rVert ) \mathbf { u } ,$ so

$$
\left\| \mathbf { u } - \mathbf { H } \mathbf { v } \right\| = \left\| \mathbf { u } - { \frac { \left\| \mathbf { v } \right\| } { \left\| \mathbf { u } \right\| } } \mathbf { u } \right\| = \left| \left\| \mathbf { u } \right\| - \left\| \mathbf { v } \right\| \right| .\tag{68}
$$

Finall<sub>y,</sub> if $\mathbf { u } = 0 \ \mathrm { o r } \ \mathbf { v } = 0 $ <sub>,</sub> th<sub>e</sub> <sub>resu</sub>lt h<sub>o</sub>ld<sub>s</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>ll<sub>y.</sub>

## A.3 A uniform high-probability bound

Bui<sup>ld</sup>ing upon t<sup>h</sup>e <sup>d</sup>eterministic trajectory <sup>d</sup>ecomposition provi<sup>d</sup>e<sup>d</sup> in the <sub>p</sub>revious subsection<sub>,</sub> this <sub>p</sub>art <sub>g</sub>ives a uniform hi<sub>g</sub>h-<sub>p</sub>robabilit<sub>y</sub> <sup>b</sup>oun<sup>d</sup> <sup>f</sup>or pairwise trajectory a<sup>l</sup>ignment proo<sup>f</sup>.

Propositio<sub>n</sub> $\mathsf { A . 7 }$ (Deterministic pairwise alignment). Suppose $\mathbf { x } _ { t \in [ 0 , T ] } ^ { ( 1 ) }$ and $\mathbf { x } _ { t \in [ 0 , T ] } ^ { ( 2 ) }$ are two exact PF-ODE trajectories with starting noise $\mathbf { x } _ { T } ^ { ( 1 ) }$ and $\mathbf { x } _ { T } ^ { ( 2 ) }$ . Then there exists H $\in O ( d )$ such that

$$
\operatorname* { s u p } _ { 0 \leq t \leq T } \| \mathbf { D } _ { t } ^ { ( 1 ) } - \mathbf { H D } _ { t } ^ { ( 2 ) } \| \leq A _ { T } \big | \| Q \mathbf { x } _ { T } ^ { ( 1 ) } \| - \| Q \mathbf { x } _ { T } ^ { ( 2 ) } \| \big |
$$

$$
+ A _ { T } \left( \lVert P \mathbf { x } _ { T } ^ { ( 1 ) } \rVert + \lVert P \mathbf { x } _ { T } ^ { ( 2 ) } \rVert \right) + 2 B _ { T } R .\tag{69}
$$

Consequently,

$$
\begin{array} { r l } { \displaystyle \underset { 0 \leq n \leq N } { \operatorname* { m a x } } \ : \| \mathbf { D } _ { n } ^ { ( 1 ) } - \mathbf { H D } _ { n } ^ { ( 2 ) } \| \leq A _ { T } \Big | \| Q \mathbf { x } _ { T } ^ { ( 1 ) } \| - \| Q \mathbf { x } _ { T } ^ { ( 2 ) } \| \Big | } & { } \\ { \displaystyle + A _ { T } \big ( \| P \mathbf { x } _ { T } ^ { ( 1 ) } \| + \| P \mathbf { x } _ { T } ^ { ( 2 ) } \| \big ) + 2 B _ { T } R . } \end{array}\tag{70}
$$

Proof<sub>.</sub> D<sub>eco</sub>m<sub>pose</sub> $\mathbb { R } ^ { d } = U \oplus U ^ { \perp }$ <sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>6<sub>,</sub> th<sub>e</sub>r<sub>e</sub> <sub>e</sub>xi<sub>s</sub>t<sub>s</sub> $\mathbf { H } _ { Q } \in O ( U ^ { \perp } )$ <sub>suc</sub>h th<sub>a</sub>t

$$
\| Q \mathbf { x } _ { T } ^ { ( 1 ) } - \mathbf { H } _ { Q } Q \mathbf { x } _ { T } ^ { ( 2 ) } \| = \big | \| Q \mathbf { x } _ { T } ^ { ( 1 ) } \| - \| Q \mathbf { x } _ { T } ^ { ( 2 ) } \| \big | .\tag{71}
$$

Choose an<sub>y</sub> $\mathbf { H } _ { P } \in O ( U )$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
\mathbf { H } : = \mathbf { H } _ { P } \oplus \mathbf { H } _ { Q } \in O ( d ) .\tag{72}
$$

B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>4<sub>,</sub>

$$
Q \mathbf { D } _ { t } ^ { ( k ) } = ( \rho _ { t } - 1 ) Q \mathbf { x } _ { T } ^ { ( k ) } .\tag{73}
$$

H<sub>e</sub>n<sub>ce</sub>

$$
Q \big ( \mathbf { D } _ { t } ^ { ( 1 ) } - \mathbf { H } \mathbf { D } _ { t } ^ { ( 2 ) } \big ) = ( \rho _ { t } - 1 ) \big ( Q \mathbf { x } _ { T } ^ { ( 1 ) } - \mathbf { H } _ { Q } Q \mathbf { x } _ { T } ^ { ( 2 ) } \big ) ,
$$

so

(74)

$$
\operatorname* { s u p } _ { 0 \leq t \leq T } \| Q ( \mathbf { D } _ { t } ^ { ( 1 ) } - \mathbf { H D } _ { t } ^ { ( 2 ) } ) \| \leq A _ { T } \big | \| Q \mathbf { x } _ { T } ^ { ( 1 ) } \| - \| Q \mathbf { x } _ { T } ^ { ( 2 ) } \| \big | .\tag{75}
$$

F<sub>o</sub>r th<sub>e</sub> �-<sub>co</sub>m<sub>po</sub>n<sub>e</sub>nt<sub>,</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>5 <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } { \| P ( \mathbf { D } _ { t } ^ { ( 1 ) } - \mathbf { H D } _ { t } ^ { ( 2 ) } ) \| \leq \| P \mathbf { D } _ { t } ^ { ( 1 ) } \| + \| P \mathbf { D } _ { t } ^ { ( 2 ) } \| } & { } \\ { \leq A _ { T } \| P \mathbf { x } _ { T } ^ { ( 1 ) } \| + B _ { T } R + A _ { T } \| P \mathbf { x } _ { T } ^ { ( 2 ) } \| + B _ { T } R . } \end{array}\tag{76}
$$

Combinin<sub>g</sub> the � and $U ^ { \perp }$ <sub>p</sub>arts via the trian<sub>g</sub>le ine<sub>q</sub>ualit<sub>y</sub> <sub>p</sub>roves th<sub>e con</sub>ti<sub>nuous-</sub>ti<sub>me</sub> b<sub>oun</sub>d<sub>.</sub> Th<sub>e</sub> di<sub>scre</sub>t<sub>e-</sub>ti<sub>me</sub> b<sub>oun</sub>d f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub>

$$
\operatorname* { m a x } _ { 0 \leq n \leq N } \| \mathbf { D } _ { n } ^ { ( 1 ) } - \mathbf { H D } _ { n } ^ { ( 2 ) } \| \leq \operatorname* { s u p } _ { 0 \leq t \leq T } \| \mathbf { D } _ { t } ^ { ( 1 ) } - \mathbf { H D } _ { t } ^ { ( 2 ) } \| .\tag{77}
$$

Lemma A.8 (Gaussian norm concentration). $I f \mathbf { g } \sim N ( 0 , \mathbf { I } _ { m } )$ then for every $t > 0 _ { : }$

$$
\mathbb { P } \big ( | \| \mathbf { g } \| - \mathbb { E } \| \mathbf { g } \| | \geq t \big ) \leq 2 e ^ { - t ^ { 2 } / 2 } , \qquad \mathbb { E } \| \mathbf { g } \| \leq \sqrt { m } .\tag{78}
$$

Proof. The ma<sub>p</sub> g ↦→ ∥g∥ is 1-Li<sub>p</sub>schitz, so the first ine<sub>q</sub>ualit<sub>y</sub> is th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an concen</sub>t<sub>ra</sub>ti<sub>on</sub> i<sub>nequa</sub>lit<sub>y.</sub> Th<sub>e secon</sub>d f<sub>o</sub>ll<sub>ows</sub> from Jensen’s ine<sub>q</sub>ualit<sub>y</sub>:

$$
\mathbb { E } \| \mathbf { g } \| \leq \sqrt { \mathbb { E } \| \mathbf { g } \| ^ { 2 } } = \sqrt { m } .\tag{79}
$$

Theorem A.9 (Explicit version of Theorem 3.1). Suppose that $\mathbf { x } _ { T } ^ { ( 1 ) } , \dots , \mathbf { x } _ { T } ^ { ( M ) }$ are i.i.d. samples from the exact noised marginal ${ \mathit { p } } _ { T }$ Then for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\begin{array} { r l r } { \displaystyle \operatorname* { m a x } _ { 1 \leq a , b \leq M } \displaystyle \operatorname* { i n f } _ { \mathbf { H } \in O ( d ) } \displaystyle \operatorname* { m a x } _ { 0 \leq n \leq N } \| { \mathbf D } _ { n } ^ { ( a ) } - \mathbf { H D } _ { n } ^ { ( b ) } \| } & \\ { \displaystyle 1 \leq a \sqrt { 2 } A _ { T } \nu _ { T } \sqrt { \log \frac { 4 M } { \delta } } } & \\ { \displaystyle } & { \qquad + \ 2 A _ { T } \Big ( \alpha _ { T } R + \nu _ { T } \Big [ \sqrt { r } + \sqrt { 2 \log \frac { 2 M } { \delta } } \Big ] \Big ) + 2 B _ { T } R . } \end{array}\tag{80}
$$

Proof<sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>1<sub>,</sub> th<sub>e</sub> n<sub>o</sub>i<sub>se</sub>d m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l <sub>a</sub>t tim<sub>e</sub> � i<sub>s</sub>

$$
{ \displaystyle p _ { T } ( { \bf x } ) = \sum _ { m = 1 } ^ { M } \pi _ { m } N ( \alpha _ { T } \pmb { \mu } _ { m } , \nu _ { T } ^ { 2 } { \bf I } _ { d } ) } .\tag{81}
$$

H<sub>ence eac</sub>h <sub>samp</sub>l<sub>e can</sub> b<sub>e represen</sub>t<sub>e</sub>d <sub>as</sub>

$$
\begin{array} { r } { \mathbf { x } _ { T } ^ { ( k ) } = \alpha _ { T } \pmb { \mu } _ { C _ { k } } + \nu _ { T } \mathbf { g } ^ { ( k ) } , } \end{array}\tag{82}
$$

<sub>w</sub>h<sub>ere</sub> $C _ { k } \ \in \ \{ 1 , . . . , M \} , \mathbb { P } ( C _ { k } \ = \ m ) \ = \ \pi _ { m }$ <sub>,</sub> <sub>an</sub>d $\mathbf { g } ^ { ( k ) } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ Sin<sub>ce</sub> $\pmb { \mu } _ { C _ { k } } \in U$

$$
Q \mathbf { x } _ { T } ^ { ( k ) } = \nu _ { T } Q \mathbf { g } ^ { ( k ) } \sim N ( 0 , \nu _ { T } ^ { 2 } \mathbf { I } _ { d - r } ) ,\tag{83}
$$

<sub>an</sub>d

$$
P \mathbf { x } _ { T } ^ { ( k ) } = \alpha _ { T } \pmb { \mu } _ { C _ { k } } + \nu _ { T } P \mathbf { g } ^ { ( k ) } .\tag{84}
$$

We first control the �-component. By Lemma A.8 and a union b<sub>oun</sub>d<sub>, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta / 2$

$$
\operatorname* { m a x } _ { 1 \leq k \leq M } \Bigl | \| Q \mathbf { x } _ { T } ^ { ( k ) } \| - \mathbb { E } \| Q \mathbf { x } _ { T } ^ { ( k ) } \| \Bigr | \leq \nu _ { T } \sqrt { 2 \log \frac { 4 M } { \delta } } .\tag{85}
$$

Th<sub>ere</sub>f<sub>ore,</sub> <sub>on</sub> thi<sub>s</sub> <sub>even</sub>t<sub>,</sub>

$$
\operatorname* { m a x } _ { 1 \leq a , b \leq M } \Bigl | \| Q \mathbf { x } _ { T } ^ { ( a ) } \| - \| Q \mathbf { x } _ { T } ^ { ( b ) } \| \Bigr | \leq 2 \nu _ { T } \sqrt { 2 \log \frac { 4 M } { \delta } } .\tag{86}
$$

Next <sub>w</sub>e control the �-com<sub>p</sub>onent. Since $P \mathbf { g } ^ { ( k ) } \sim N ( 0 , \mathbf { I } _ { r } )$ on $U _ { \textrm { \scriptsize : } }$ L<sub>emma</sub> A<sub>.</sub>8 <sub>an</sub>d $\| \mu _ { C _ { k } } \| \le R$ im<sub>p</sub>l<sub>y</sub> that<sub>,</sub> with <sub>p</sub>robabilit<sub>y</sub> at least $1 - \delta / 2$

$$
\operatorname* { m a x } _ { 1 \leq k \leq M } \| P \mathbf { x } _ { T } ^ { ( k ) } \| \leq \alpha _ { T } R + \nu _ { T } \bigg ( \sqrt { r } + \sqrt { 2 \log \frac { 2 M } { \delta } } \bigg ) .\tag{87}
$$

On the intersection of these two events<sub>,</sub> whose <sub>p</sub>robabilit<sub>y</sub> is <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$ Pr<sub>opos</sub>iti<sub>o</sub>n $\mathsf { A . 7 }$ <sub>g</sub>ives, for ever<sub>y</sub> <sub>p</sub>air (�, �), some <sub>or</sub>th<sub>ogona</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\mathbf { H } _ { a , b } \in O ( d )$ <sub>suc</sub>h th<sub>a</sub>t

$$
\begin{array} { l } { { \displaystyle \operatorname* { m a x } _ { 0 \le n \le N } \| { \bf D } _ { n } ^ { ( a ) } - { \bf H } _ { a , b } { \bf D } _ { n } ^ { ( b ) } \| } } \\ { ~ } \\ { { \displaystyle \le A _ { T } \cdot 2 \nu _ { T } \sqrt { 2 \log \frac { 4 M } { \delta } } } } \\ { { \displaystyle ~ + A _ { T } \cdot 2 \Big ( \alpha _ { T } R + \nu _ { T } \Big [ \sqrt { r } + \sqrt { 2 \log \frac { 2 M } { \delta } } \Big ] \Big ) + 2 B _ { T } R } . } \end{array}\tag{88}
$$

Takin<sub>g</sub> the maximum over (�, �) and then the infimum over $\mathbf { H } \in$ �(�) <sub>y</sub>ields the stated ine<sub>q</sub>ualit<sub>y</sub>. □

Proof of Corollary 3<sub>.</sub>2<sub>.</sub> Di<sub>v</sub>id<sub>e</sub> th<sub>e</sub> b<sub>ou</sub>nd in Th<sub>eo</sub>r<sub>e</sub>m A<sub>.</sub>9 b<sub>y</sub> $\nu _ { T } { \sqrt { d } } .$ <sub>.</sub> Thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { \nu _ { T } \sqrt { d } } \displaystyle \operatorname* { m a x } _ { 1 \leq a , b \leq n } \operatorname* { i n f } _ { \mathbf { H } \in O ( d ) } \operatorname* { m a x } _ { \mathbf { \ell } \in { \cal L } } \| { \mathbf { D } } _ { \ell } ^ { ( a ) } - \mathbf { H D } _ { \ell } ^ { ( b ) } \| } \\ & { \leq 2 \sqrt { 2 } A _ { T } \sqrt { \displaystyle \frac { \log ( 4 M / \delta ) } { d } } + 2 A _ { T } \sqrt { \displaystyle \frac { r } { d } } } \\ & { \quad \quad + 2 A _ { T } \sqrt { \displaystyle \frac { 2 \log ( 2 M / \delta ) } { d } } + 2 A _ { T } \displaystyle \frac { \alpha _ { T } R } { { \nu _ { T } \sqrt { d } } } + 2 B _ { T } \displaystyle \frac { R } { { \nu _ { T } \sqrt { d } } } . } \end{array}\tag{89}
$$

Since � + lo<sub>g</sub> $n = o ( d )$ <sub>an</sub>d $R = o \left( \nu _ { T } \sqrt { d } \right)$ <sub>,</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> <sub>con-</sub> <sub>verges</sub> t<sub>o</sub> 0 f<sub>or</sub> <sub>every</sub> fi<sub>xe</sub>d $\delta \in ( 0 , 1 )$ . This <sub>p</sub>ro<sub>v</sub>es the claim. □

## A.4 Proofs of the low-dimensionality Corollary

We now s<sup>h</sup>ow t<sup>h</sup>at un<sup>d</sup>er mi<sup>ld</sup> assumptions, eac<sup>h</sup> PF-ODE trajectory i<sub>s</sub> <sub>con</sub>fi<sub>ne</sub>d t<sub>o</sub> <sub>a</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>su</sub>b<sub>space.</sub> M<sub>oreover,</sub> <sub>a</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> a<sup>l</sup>ignment <sup>b</sup>rings a<sup>ll</sup> trajectories into a sing<sup>l</sup>e s<sup>h</sup>are<sup>d</sup> su<sup>b</sup>space.

Corollary A.10 (Explicit version of Corollary 3.3). Suppose $\mathbf { x } _ { t \in [ 0 , T ] } ^ { ( k ) }$ is an exact PF-ODE trajectory. Then

$$
\mathbf { x } _ { t } ^ { ( k ) } \in S _ { k } : = U \oplus \mathsf { s p a n } \{ Q \mathbf { x } _ { T } ^ { ( k ) } \} , \qquad \forall t \in [ 0 , T ] .\tag{90}
$$

In particular,

$$
\dim ( S _ { k } ) \leq r + 1 .\tag{91}
$$

Hence, on any discrete reverse-time grid $T = t _ { 0 } > t _ { 1 } > \cdot \cdot \cdot > t _ { L } = 0 _ { : }$ the trajectory matrix

$$
\mathbf { X } ^ { ( k ) } \in \mathbb { R } ^ { ( L + 1 ) \times d } , \qquad r o w _ { \ell } ( \mathbf { X } ^ { ( k ) } ) = ( \mathbf { x } _ { \ell } ^ { ( k ) } ) ^ { \top } ,\tag{92}
$$

satisfies

$$
\operatorname { r a n k } ( \mathbf { X } ^ { ( k ) } ) \leq r + 1 .\tag{93}
$$

Equivalently, the empirical trajectory covariance

$$
\Sigma _ { \mathrm { t r a j } } ^ { ( k ) } : = \frac { 1 } { M + 1 } \sum _ { m = 0 } ^ { M } \bigl ( \mathbf { x } _ { \ell } ^ { ( k ) } - \bar { \mathbf { x } } ^ { ( k ) } \bigr ) \bigl ( \mathbf { x } _ { \ell } ^ { ( k ) } - \bar { \mathbf { x } } ^ { ( k ) } \bigr ) ^ { \intercal }\tag{94}
$$

has rank at most $r + 1 .$ , and therefore

$$
\lambda _ { j } \left( \Sigma _ { \mathrm { t r a j } } ^ { ( k ) } \right) = 0 , \qquad j \geq r + 2 .\tag{95}
$$

Proof<sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>4<sub>,</sub>

$$
Q \mathbf { x } _ { t } ^ { ( k ) } = \rho _ { t } Q \mathbf { x } _ { T } ^ { ( k ) } ,\tag{96}
$$

<sub>w</sub>hi<sub>c</sub>h li<sub>es</sub> i<sub>n</sub> th<sub>e</sub> <sub>one-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>su</sub>b<sub>space</sub> <sub>span</sub> $\{ Q \mathbf { x } _ { T } ^ { ( k ) } \}$ <sub>.</sub> O<sub>n</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>,</sub>

$$
P \mathbf { x } _ { t } ^ { ( k ) } \in U\tag{97}
$$

f<sub>or a</sub>ll $t \in [ 0 , T ]$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\mathbf { x } _ { t } ^ { ( k ) } = { P } \mathbf { x } _ { t } ^ { ( k ) } + { Q } \mathbf { x } _ { t } ^ { ( k ) } \in { U } \oplus \operatorname { s p a n } \{ { Q } \mathbf { x } _ { T } ^ { ( k ) } \} .\tag{98}
$$

T<sup>h</sup>is proves t<sup>h</sup>at t<sup>h</sup>e w<sup>h</sup>o<sup>l</sup>e trajectory <sup>l</sup>ies in t<sup>h</sup>e <sup>l</sup>inear su<sup>b</sup>space $S _ { k }$ <sub>w</sub>ith di<sub>m</sub> $( S _ { k } ) \le r + 1$ . Since e<sub>v</sub>er<sub>y</sub> ro<sub>w</sub> of $\mathbf { \partial } \cdot \mathbf { \partial } _ { \mathbf { X } } ( k )$ b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $S _ { k }$ , we h<sub>ave</sub>

$$
\mathrm { r a n k } ( \mathbf { X } ^ { ( k ) } ) \leq \dim ( S _ { k } ) \leq r + 1 .\tag{99}
$$

Sin<sub>ce</sub> $S _ { k }$ is a <sup>l</sup>inear su<sup>b</sup>space, it contains t<sup>h</sup>e centere<sup>d</sup> trajectory vectors. Thus<sub>,</sub> the ima<sub>g</sub>e of the covariance matrix $\Sigma _ { \mathrm { t r a j } } ^ { ( k ) }$ li<sub>es w</sub>ithin $s _ { k } ,$ <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub><sup>i</sup>n<sub>g</sub>

$$
\mathrm { r a n k } \Big ( \Sigma _ { \mathrm { t r a j } } ^ { ( k ) } \Big ) \leq r + 1 .\tag{100}
$$

Thi<sub>s</sub> di<sub>rec</sub>tl<sub>y</sub> l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> <sub>e</sub>i<sub>genva</sub>l<sub>ues.</sub> □

Corollary A.11 (Explicit version of Corollary 3.4). Fix any unit vector e ∈ $U ^ { \perp }$ . For each trajectory �, choose an orthogonal matrix $\mathbf { H } _ { k } \in O ( d )$ such that

$$
\mathbf { H } _ { k } | _ { U } = \mathbf { I } , \qquad \mathbf { H } _ { k } Q \mathbf { x } _ { T } ^ { ( k ) } = \| Q \mathbf { x } _ { T } ^ { ( k ) } \| \mathbf { e } .\tag{101}
$$

Define the aligned trajectory

$$
\tilde { \mathbf { x } } _ { t } ^ { ( k ) } : = \mathbf { H } _ { k } \mathbf { x } _ { t } ^ { ( k ) } .\tag{102}
$$

Then

$$
\tilde { \mathbf { x } } _ { t } ^ { ( k ) } \in \widetilde { S } : = U \oplus \mathrm { s p a n } \{ \mathbf { e } \} , \qquad \forall k , \forall t \in [ 0 , T ] .\tag{103}
$$

In particular,

$$
\mathrm { d i m } ( \widetilde { S } ) \leq r + 1 .\tag{104}
$$

Hence, $i f \widetilde { \mathbf { X } } \in \mathbb { R } ^ { n ( L + 1 ) \times d }$ denotes the pooled aligned trajectory matrix obtained by stacking all aligned trajectory states, then

$$
\operatorname { r a n k } ( { \widetilde { \mathbf { X } } } ) \leq r + 1 .\tag{105}
$$

Equivalently, the pooled covariance of the aligned trajectories has rank at most � + 1.

Proof<sub>.</sub> B<sub>y co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>,</sub> $\mathrm { H } _ { k }$ <sub>ac</sub>t<sub>s as</sub> th<sub>e</sub> id<sub>e</sub>ntit<sub>y o</sub>n �<sub>.</sub> Th<sub>e</sub>r<sub>e</sub>f<sub>o</sub>r<sub>e,</sub> <sub>s</sub>in<sub>ce</sub> $P \mathbf { x } _ { t } ^ { ( k ) } \in U$

$$
\mathbf { H } _ { k } P \mathbf { x } _ { t } ^ { ( k ) } = P \mathbf { x } _ { t } ^ { ( k ) } \in U .\tag{106}
$$

M<sub>o</sub>r<sub>eove</sub>r<sub>,</sub> <sub>us</sub>in<sub>g</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>4<sub>,</sub>

$$
Q \mathbf { x } _ { t } ^ { ( k ) } = \rho _ { t } Q \mathbf { x } _ { T } ^ { ( k ) } ,\tag{107}
$$

<sub>an</sub>d th<sub>us</sub>

$$
\mathbf { H } _ { k } Q \mathbf { x } _ { t } ^ { ( k ) } = \rho _ { t } \mathbf { H } _ { k } Q \mathbf { x } _ { T } ^ { ( k ) } = \rho _ { t } \| Q \mathbf { x } _ { T } ^ { ( k ) } \| \mathbf { e } \in \mathrm { s p a n } \{ \mathbf { e } \} .\tag{108}
$$

Combinin<sub>g</sub> the � -<sub>p</sub>art and the $U ^ { \perp }$ <sup>-</sup>p<sup>art</sup> g<sup>ives</sup>

$$
\widetilde { \mathbf { x } } _ { t } ^ { ( k ) } = \mathbf { H } _ { k } \mathbf { x } _ { t } ^ { ( k ) } \in U \oplus \mathrm { s p a n } \{ \mathbf { e } \} = \widetilde { S } .\tag{109}
$$

T<sup>h</sup>ere<sup>f</sup>ore a<sup>ll</sup> a<sup>l</sup>igne<sup>d</sup> trajectories s<sup>h</sup>are a sing<sup>l</sup>e su<sup>b</sup>space $\widetilde s$ <sub>o</sub>fdi<sub>men-</sub> sion at most $r + 1$ <sub>, w</sub>hi<sub>c</sub>h i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e ran</sub>k <sub>resu</sub>lt<sub>s.</sub> □

Remark 1 (Connection to least-squares reconstruction). The above <sub>coro</sub>ll<sub>ar</sub>i<sub>es prov</sub>id<sub>e a</sub> di<sub>rec</sub>t <sub>exp</sub>l<sub>ana</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> l<sub>eas</sub>t<sub>-squares res</sub>id<sub>-</sub> <sub>u</sub>al observed in <sub>p</sub>ractice. S<sub>p</sub>ecificall<sub>y,</sub> ever<sub>y</sub> state $\mathbf { x } _ { n } ^ { ( k ) }$ on a <sub>g</sub><sup>i</sup>ven trajectory <sup>�</sup> <sup>b</sup>e<sup>l</sup>ongs to t<sup>h</sup>e same <sup>l</sup>inear su<sup>b</sup>space $S _ { k }$ <sub>w</sub>ith

$$
\dim ( S _ { k } ) \leq r + 1 .\tag{110}
$$

Th<sub>ere</sub>f<sub>ore,</sub> if <sub>a</sub> hi<sub>s</sub>t<sub>ory</sub> b<sub>u</sub>f<sub>er</sub>

$$
\{ \mathbf { x } _ { n - B } ^ { ( k ) } , \mathbf { x } _ { n - B + 1 } ^ { ( k ) } , . . . , \mathbf { x } _ { n - 1 } ^ { ( k ) } \}\tag{111}
$$

<sup>s</sup>p<sup>ans</sup> $S _ { k }$ <sub>,</sub> th<sub>en</sub> th<sub>e</sub> l<sub>eas</sub>t<sub>-squares recons</sub>t<sub>ruc</sub>ti<sub>on error o</sub>f $\mathbf { x } _ { n } ^ { ( k ) }$ f<sub>rom</sub> thi<sub>s</sub> b<sub>u</sub>f<sub>er</sub> i<sub>s exac</sub>tl<sub>y zero:</sub>

$$
\operatorname* { m i n } _ { \mathbf { c } \in \mathbb { R } ^ { B } } \left\| \mathbf { x } _ { n } ^ { ( k ) } - \sum _ { j = 1 } ^ { B } c _ { j } \mathbf { x } _ { n - j } ^ { ( k ) } \right\| = 0 .\tag{112}
$$

Th<sub>u</sub>s<sub>,</sub> in the exact s<sub>p</sub>herical GMM settin<sub>g,</sub> a s<sub>u</sub>ficientl<sub>y</sub> rich short b<sub>u</sub>f<sub>er a</sub>l<sub>rea</sub>d<sub>y</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e</sub> f<sub>u</sub>t<sub>ure po</sub>i<sub>n</sub>t b<sub>y</sub> li<sub>near regress</sub>i<sub>on.</sub> I<sub>n</sub> <sub>rea</sub>li<sub>s</sub>ti<sub>c</sub> dif<sub>us</sub>i<sub>on mo</sub>d<sub>e</sub>l<sub>s we</sub> d<sub>o no</sub>t <sub>expec</sub>t thi<sub>s</sub> id<sub>en</sub>tit<sub>y</sub> t<sub>o</sub> h<sub>o</sub>ld <sub>exac</sub>tl<sub>y,</sub> b<sub>u</sub>t <sub>sma</sub>ll l<sub>eas</sub>t<sub>-squares</sub> <sub>res</sub>id<sub>ua</sub>l<sub>s</sub> <sub>s</sub>t<sub>rong</sub>l<sub>y</sub> <sub>sugges</sub>t th<sub>a</sub>t th<sub>e</sub> denoisin<sub>g</sub> d<sub>y</sub>namics remain a<sub>pp</sub>roximatel<sub>y</sub> low-dimensional.

Table 3: LEAP combined with diferent base solvers in SD3.5- Medium. All results use the same 15-step schedule and � = 5 in LEAP’s setting.
<table><tr><td>Solver + Method</td><td>HPSv2↑</td><td>ImageReward↑</td><td>CLIPScore↑</td></tr><tr><td>Euler</td><td>27.4725</td><td>0.8416</td><td>33.4644</td></tr><tr><td>LEAP (Euler)</td><td>28.1826</td><td>0.9522</td><td>33.6817</td></tr><tr><td>DPM++</td><td>27.8642</td><td>0.8438</td><td>33.4704</td></tr><tr><td>LEAP (DPM++)</td><td>28.2152</td><td>0.9806</td><td>33.5617</td></tr><tr><td>iPNDM</td><td>27.9220</td><td>0.8770</td><td>33.4416</td></tr><tr><td>LEAP (iPNDM)</td><td>28.2204</td><td>0.9642</td><td>33.6601</td></tr><tr><td>UniPC</td><td>27.8680</td><td>0.8788</td><td>33.4703</td></tr><tr><td>LEAP (UniPC)</td><td>28.2309</td><td>0.9800</td><td>33.6077</td></tr></table>

Table 4: Ablation on the prediction-to-correction (P:C) schedule on SD3.5-Medium.
<table><tr><td>NFE</td><td>P:C</td><td>HPSv2↑</td><td>ImageReward↑</td><td>CLIPScore↑</td></tr><tr><td rowspan="2">10</td><td>1:1</td><td>26.8419</td><td>0.7692</td><td>33.2982</td></tr><tr><td>1:2</td><td>26.3182</td><td>0.7224</td><td>33.1515</td></tr><tr><td rowspan="2">15</td><td>1:1</td><td>28.1826</td><td>0.9522</td><td>33.6817</td></tr><tr><td>1:2</td><td>27.9594</td><td>0.9004</td><td>33.7109</td></tr><tr><td rowspan="2">20</td><td>1:1</td><td>28.5727</td><td>0.9701</td><td>33.7192</td></tr><tr><td>1:2</td><td>28.3575</td><td>0.9499</td><td>33.8125</td></tr></table>

## B Experimental Results

## B.1 Ablation Study

B.1.1 Compatibility with Diferent Base Solvers. To further evalu-<sub>a</sub>t<sub>e</sub> th<sub>e</sub> r<sub>o</sub>b<sub>us</sub>tn<sub>ess a</sub>nd <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>bilit<sub>y o</sub>f LEAP<sub>, we co</sub>nd<sub>uc</sub>t an additional ablation ex<sub>p</sub>eriment b<sub>y</sub> inte<sub>g</sub>ratin<sub>g</sub> LEAP <sub>w</sub>ith <sub>v</sub>ari<sub>ous</sub> <sub>w</sub>id<sub>e</sub>l<sub>y</sub> <sub>use</sub>d ODE <sub>so</sub>l<sub>ve</sub>r<sub>s,</sub> in<sub>c</sub>l<sub>u</sub>din<sub>g</sub> E<sub>u</sub>l<sub>e</sub>r<sub>,</sub> DPM++<sub>,</sub> iPNDM<sub>,</sub> <sub>a</sub>nd UniPC<sub>.</sub> Thi<sub>s e</sub>x<sub>pe</sub>rim<sub>e</sub>nt i<sub>s co</sub>nd<sub>uc</sub>t<sub>e</sub>d <sub>o</sub>n SD3<sub>.</sub>5 <sub>w</sub>ith NFE=15 and � = 5. As shown in Table 3, LEAP im<sub>p</sub>roves human <sub>p</sub>reference and text-ima<sub>g</sub>e ali<sub>g</sub>nment metrics (HPSv2, Ima<sub>g</sub>eReward, and CLIPScore) across all tested baseline solvers.

B.1.2 Predict-correct schedule. We further test the ratio between <sub>p</sub>rediction and correction ste<sub>p</sub>s. Consec<sub>u</sub>ti<sub>v</sub>e linear <sub>p</sub>redictions <sub>are</sub> th<sub>eore</sub>ti<sub>ca</sub>ll<sub>y</sub> d<sub>egra</sub>d<sub>a</sub>t<sub>e</sub> t<sub>o our curren</sub>t i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on as</sub> th<sub>e</sub> <sub>same</sub> li<sub>near su</sub>b<sub>space</sub> i<sub>s use</sub>d<sub>.</sub> W<sub>e</sub> th<sub>us a</sub>dd <sub>resu</sub>lt<sub>s regar</sub>di<sub>ng</sub> 2<sub>-s</sub>t<sub>ep</sub> <sub>consecu</sub>ti<sub>ve correc</sub>ti<sub>ons</sub> b<sub>e</sub>l<sub>ow, w</sub>hi<sub>c</sub>h <sub>are</sub> i<sub>n</sub>f<sub>er</sub>i<sub>or</sub> t<sub>o our</sub> d<sub>e</sub>f<sub>au</sub>lt 1<sub>:</sub>1 settin<sub>g,</sub> as shown in Table 4.

## B.2 Robustness, Transferability, and Generalization

B.2.1 Prompt-distribution robustness. The coeficient bank is con-<sub>s</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng re</sub>f<sub>erence promp</sub>t<sub>s samp</sub>l<sub>e</sub>d f<sub>rom</sub> D<sub>raw</sub>B<sub>enc</sub>h<sub>, w</sub>hil<sub>e</sub> eval<sub>u</sub>ation is cond<sub>u</sub>cted on fo<sub>u</sub>r <sub>p</sub>rom<sub>p</sub>t cate<sub>g</sub>ories from the HPSv2 b<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk<sub>.</sub> A<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in T<sub>a</sub>bl<sub>e</sub> 5<sub>,</sub> LEAP im<sub>p</sub>r<sub>oves</sub> E<sub>u</sub>l<sub>e</sub>r <sub>ac</sub>r<sub>oss</sub> <sub>eve</sub>r<sub>y</sub> <sub>ca</sub>t<sub>ego</sub>r<sub>y</sub> f<sub>o</sub>r HPS<sub>v</sub>2<sub>,</sub> Im<sub>age</sub>R<sub>ewa</sub>rd<sub>,</sub> <sub>a</sub>nd CLIPS<sub>co</sub>r<sub>e.</sub> Th<sub>ese</sub> r<sub>esu</sub>lt<sub>s</sub> indi<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t LEAP r<sub>e</sub>m<sub>a</sub>in<sub>s e</sub>f<sub>ec</sub>ti<sub>ve u</sub>nd<sub>e</sub>r <sub>a p</sub>r<sub>o</sub>m<sub>p</sub>t di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>s</sub>hift<sub>.</sub>

Table 5: Ablation study on prompt distributions (NFE= 15). Reference prompts are sampled from DrawBench, while eval uation prompts are sampled from the HPSv2 benchmark.
<table><tr><td>Metric</td><td>Method</td><td>Anime</td><td>ConceptArt</td><td>Paintings</td><td>Photo</td></tr><tr><td>HPSv2</td><td>Euler</td><td>30.14</td><td>28.63</td><td>28.99</td><td>26.56</td></tr><tr><td></td><td>LEAP (Euler)</td><td>31.22</td><td>30.15</td><td>30.14</td><td>27.37</td></tr><tr><td>ImageReward</td><td>Euler</td><td>1.10</td><td>1.02</td><td>1.06</td><td>0.80</td></tr><tr><td></td><td>LEAP (Euler)</td><td>1.18</td><td>1.15</td><td>1.16</td><td>0.96</td></tr><tr><td>CLIPScore</td><td>Euler</td><td>35.18</td><td>33.61</td><td>33.41</td><td>32.28</td></tr><tr><td></td><td>LEAP (Euler)</td><td>35.19</td><td>33.96</td><td>33.69</td><td>32.79</td></tr></table>

B.2.2 Coeficient transferability. We evaluate whether learned co-<sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s can</sub> b<sub>e reuse</sub>d b<sub>eyon</sub>d th<sub>e so</sub>l<sub>ver, mo</sub>d<sub>e</sub>l<sub>, or samp</sub>li<sub>ng</sub> <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> f<sub>rom w</sub>hi<sub>c</sub>h it <sub>was cons</sub>t<sub>ruc</sub>t<sub>e</sub>d<sub>.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y, we s</sub>t<sub>u</sub>d<sub>y</sub> <sub>cross-mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>cross-so</sub>l<sub>ver,</sub> <sub>an</sub>d <sub>cross-s</sub>t<sub>ep</sub> t<sub>rans</sub>f<sub>er.</sub> A<sub>s</sub> <sub>repor</sub>t<sub>e</sub>d i<sub>n</sub> Table 6<sub>,</sub> ever<sub>y</sub> transferred confi<sub>g</sub>uration im<sub>p</sub>roves its corres<sub>p</sub>ondin<sub>g</sub> b<sub>ase</sub>li<sub>ne on a</sub>ll th<sub>ree me</sub>t<sub>r</sub>i<sub>cs.</sub>

Table 6: Transferability of the coeficient bank. Cross-model and cross-solver results use 10 $\mathbf { N F E ; }$ cross-step results use 15 NFE.
<table><tr><td>Transfer setting</td><td>Method</td><td>HPSv2↑</td><td>ImageReward↑</td><td>CLIPScore↑</td></tr><tr><td rowspan="2">Cross-model</td><td>Euler</td><td>25.89</td><td>0.67</td><td>32.79</td></tr><tr><td>LEAP (FLUX coeffs.)</td><td>26.97</td><td>0.75</td><td>33.26</td></tr><tr><td rowspan="2">Cross-solver</td><td>DPM++</td><td>26.46</td><td>0.72</td><td>32.90</td></tr><tr><td>LEAP (Euler coeffs.)</td><td>27.01</td><td>0.80</td><td>33.48</td></tr><tr><td rowspan="2">Cross-step</td><td>Euler</td><td>27.47</td><td>0.84</td><td>33.46</td></tr><tr><td>LEAP (10-NFE coeffs.)</td><td>28.17</td><td>0.91</td><td>33.69</td></tr></table>

## B.3 Additional Qalitative Results

W<sub>e</sub> fi<sub>rs</sub>t <sub>prov</sub>id<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub> <sub>resu</sub>lt<sub>s</sub> f<sub>or</sub> b<sub>o</sub>th t<sub>ex</sub>t<sub>-</sub>i<sub>mage</sub> task and ima<sub>g</sub>e-3D task under the same settin<sub>g</sub> in Sec.4.1. For textima<sub>g</sub>e task<sub>,</sub> the <sub>qu</sub>alitati<sub>v</sub>e res<sub>u</sub>lts are sho<sub>w</sub>n in Fi<sub>g</sub>. 9. For ima<sub>g</sub>e-3D task<sub>,</sub> the <sub>q</sub>ualitative results are shown in Fi<sub>g</sub>. 10.

## B.4 Experimental Details

B.4.1 Hyperparameters. In this subsection, we discuss our experiment setu<sub>p</sub> in <sub>g</sub>reater detail. In Sec. 4<sub>,</sub> we evaluate the <sub>p</sub>ro<sub>p</sub>osed LEAP on three difusion models: SD3.5-Medium [8] and FLUX.1- Dev [2] for text-ima<sub>g</sub>e task and TRELLIS.2 [41] for ima<sub>g</sub>e-3D task.

W<sub>e</sub> t<sub>es</sub>t LEAP <sub>aga</sub>in<sub>s</sub>t <sub>p</sub>r<sub>ev</sub>i<sub>ous so</sub>l<sub>ve</sub>r<sub>s</sub> in t<sub>e</sub>xt-t<sub>o</sub>-im<sub>age</sub> t<sub>as</sub>k<sub>s,</sub> includin<sub>g</sub> DPM++ [19], STORK [34], and LTC-Accel [50]. In our eval <sub>ua</sub>ti<sub>o</sub>n<sub>,</sub> DPM++ i<sub>s co</sub>nfi<sub>gu</sub>r<sub>e</sub>d t<sub>o</sub> it<sub>s seco</sub>nd-<sub>o</sub>rd<sub>e</sub>r <sub>ve</sub>r<sub>s</sub>i<sub>o</sub>n<sub>,</sub> STORK <sub>u</sub>tiliz<sub>es</sub> $k = 4 , n = 1$ <sub>, an</sub>d $s \ = \ 9 ,$ <sub>, w</sub>hil<sub>e</sub> LTC-A<sub>cce</sub>l <sub>a</sub>d<sub>op</sub>t<sub>s</sub> $r \ = \ 2$ and � = 0.1 for both <sub>q</sub>uantitative and <sub>q</sub>ualitative assessments. We use a fixed window size of � = 5 and set the rid<sub>g</sub>e re<sub>g</sub>ularization coeficient to � = 0.0001 as the default ex<sub>p</sub>eriments settin<sub>g</sub>s of LEAP<sub>.</sub>

I<sub>n</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t <sub>o</sub>f i<sub>mage-</sub>t<sub>o-</sub>3D t<sub>as</sub>k<sub>s,</sub> <sub>we</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>our</sub> <sub>approac</sub>h a<sub>g</sub>ainst the Euler baseline. For these ex<sub>p</sub>eriments<sub>,</sub> we maintain the exact same confi<sub>g</sub>uration as mentioned in the text-to-ima<sub>g</sub>e tasks d<sub>escr</sub>ib<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub>

B.4.2 Prompts. In the text-to-image qualitative experiments, we ado<sub>p</sub>t the followin<sub>g</sub> text <sub>p</sub>rom<sub>p</sub>ts to <sub>g</sub>enerate ima<sub>g</sub>es:

• "A close-up ofa futuristic barista robot pouring latte art in the shape of a perfect hexagon. The robot’s metallic arm has the word ’SPEED’ etched into it. High contrast, cinematic lighting, photorealistic."

• "A transparent glass teapot filled with blooming jasmine flowers. Steam is rising, and sunlight is refracting through the water and glass, casting complex caustics onto a white tablecloth.",

• "A swarm of tiny, glowing robotic drones weaving an intricate, glowinglaser web between two giant, ancient redwood trees in a foggy forest. The web perfectly forms the exact shape of a rigid, symmetrical pentagram. Volumetric fog. "

In th<sub>e</sub> im<sub>age</sub>-t<sub>o</sub>-3D <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub> <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s,</sub> <sub>we</sub> <sub>use</sub> N<sub>a</sub>n<sub>o</sub> B<sub>a</sub>- nana2 to <sub>g</sub>enerate reference ima<sub>g</sub>es<sub>,</sub> as shown in Fi<sub>g</sub>. 8.

![](images/2b95d0e17f400a83213e92d0f3d32f12e1be420428a7dd85a6298aba10662da3.jpg)

![](images/3c1257870389e2876d1266ee2fd26bc3596e99d090fe8491da18da355fb54f4e.jpg)

Figure 8: Reference images adopted in image-3D qualitative experiments.  
DPM-Solver++  
LTC-Accel  
STORK  
LEAP (Ours)  
"An IT-guy trying to fix hardware of a PC tower is being tangled by the PC cables like Laokoon. Marble, copy after Hellenistic original from ca. 200 BC. Found in the Baths of Trajan, 1506."  
![](images/2ed86c4eb5579a8da3a40db46ea6b5c20b22f0249b1de25568db3f4fe2bb4a64.jpg)  
"A sign that says ‘Hello World’."

Figure 9: Additional qualitative comparison on the FLUX model (Part 1).

DPM-Solver++  
LTC-Accel  
STORK4  
LEAP (Ours)  
![](images/dbd9e9c5603d7f8ad4b26c1f05a34a358b2f30d5b775cc5038cbab737ec7196f.jpg)  
"A mechanical or electrical device for measuring time."

Figure 9: Additional qualitative comparison on the FLUX model (Continued).

![](images/25ba9486f20784f38fd9266c2d671c035e09f0f23d9def06df7a3dc1a6a10f7f.jpg)  
Figure 10: Additional image-3D qualitative results. Columns from left to right show the input conditions, generation results from the baseline solver (NFE=10), and results from our proposed LEAP method (NFE=10).