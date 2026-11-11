# The Case for AI-Enabled Gate Monitoring as Farm Security in the UK: A PRISMA-Guided Review of the Evidence on Rural Crime, Livestock Predation and Farm Access

Dr Steve Symes<sup>1</sup>, Prof Paul Fergus<sup>1</sup>, Mr Mark Carr<sup>2</sup>, Mr Omar McIver<sup>3</sup> and Mr Arwyn Jones<sup>4</sup> <sup>1</sup>School of Computer Science and Mathematics, Liverpool John Moores University, Liverpool, UK. <sup>2</sup>Image Surveys Limited, Southport, UK. <sup>3</sup>Mode3 Cloud, Katy, Texas, USA. <sup>4</sup>Labour Force Europe Ltd, Corwen, North Wales.

Correspondence: S.W.Symes@ljmu.ac.uk , P.Fergus@ljmu.ac.uk , Mark@imagesurveys.com , Omar.McIver@mode3.ai , arwyn@labourforceeuropeltd.com

Preprint, 4 November 2026.

## ABSTRACT

Farm security in the UK rests on thinner evidence than its cost suggests. We ran a PRISMA guided review: twelve searches of Web of Science and Scopus on 2<sup>nd</sup> -4<sup>th</sup> November 2026 returned 1,183 records (761 unique), and grey-literature and hand-searches added 35 assessed sources. We synthesised 96 database records and 24 grey sources narratively. UK peer-reviewed evidence is small (16 records), and the main UK survey of crime and farmer mental health has 80 responses. Cost estimates rest on insurer claims: NFU Mutual put UK rural crime at £41.5 million in 2025, with livestock theft up almost 30% to about £4.5 million and dog attacks on livestock costing £1.95 million. The Home Office states that agricultural crime cannot be separately identified in its central data. We found no independent evaluation of cameras, sensors, trackers or marking; light and sound deterrents lost effect within one to five months in a systematic review of anti-predator interventions; and no study examined how often farm gates are left open or what follows. Camera-trap AI papers report accuracy on ecological datasets, none at a farm gate in the records we found, and in one comparison machine learning detected 41% of wildlife against 81% for analysts. We argue, as an inference from these gaps and not as a demonstrated effect, that continuous gate-level AI-mediated monitoring, as implemented in Fieldmark AI, would be highly beneficial for UK farm security. It would create the farm-level, time-stamped record that no official source holds, verify events that periodic human checks miss, resolve whether foxes, dogs or people were responsible, and give every other security measure something to be tested against. No study has yet shown that it reduces loss, and we set out the evaluation needed to show it.

Keywords: farm security; rural crime; livestock predation; camera traps; computer vision; PRISMA; evidence review

## KEY POINTS

• The evidence on farm security is thin where it matters most to a UK farmer: only 16 of 96 database records are UK peer-reviewed studies, and the principal UK survey linking crime to farmer mental health has 80 responses.

• Cost figures come from insurer claims (£41.5 million in 2025), because police-recorded data in England and Wales cannot identify farm crime.

No independent evaluation of cameras, sensors, trackers or marking was found, and light and sound deterrents lost effect within one to five months in a systematic review. Physical barriers and management are better supported.

• Camera-trap AI has been validated on ecological data, and we found no farm-gate evaluation. No

study examined how often farm gates are left open or what follows.

The authors argue that continuous, gate-level AI monitoring such as Fieldmark AI would be highly beneficial because it fills the largest of these gaps. This is an inference, not a demonstrated reduction in loss, and Box 1 sets out how to test it.

## 1 Introduction

Farm security in the United Kingdom (UK) is usually discussed in terms of cost. NFU Mutual, the largest rural insurer, estimated the cost of UK rural crime at £41.5 million in 2025, with livestock theft up almost 30% to about £4.5 million [1]. It separately estimated £1.95 million of farm animals severely injured or killed by dogs in 2025, up 10% [2]. Those are claims-based insurer figures. The Home Office has stated that offences involving agricultural crime “cannot be separately identified” in the data it holds centrally [3], and that there is no distinct offence category that captures rural crime [4]. The costs are therefore reported with confidence, while the events behind them are not systematically recorded.

Farm security is also wider than crime. A farm is a long, open boundary with many gates. Theft of livestock and machinery sits alongside predation by foxes and other species, attacks by dogs, trespass, and the access that walkers and vehicles have through gates and across fields. Each of these is a question of what happened, when, and who or what was responsible, and each is hard to answer from a periodic visit to the gate.

This review asks what the evidence supports about farm security, and whether it substantiates continuous, AI-mediated monitoring of farm gates as part of it. It is the second of two reviews by the authors. The first examined the economic, regulatory and technological case for integrated, AIenabled land management in UK agriculture [5]. A companion case study reports a 14-camera gatemonitoring deployment at a Welsh hill farm on the Fieldmark AI platform [6]. This review supplies the evidence context for that deployment. The deployment is not itself evidence in the review.

We pose five questions.

• RQ1. What is known about the scale, nature and cost of theft, predation, dog attacks and unauthorised access on UK farms?

• RQ2. How reliable are the data: how is farm crime recorded and reported, and how far do farmers trust the system?

• RQ3. What evidence exists on the effectiveness of prevention and detection measures, including cameras and sensors?

• RQ4. What evidence exists that camera-trap-style AI can detect the targets that matter to farm security (people, vehicles, predators, dogs), and with what limits?

• RQ5. Given the answers, which evidence gaps would continuous gate-level monitoring address, and what would be needed to show that it works?

Section 2 sets out the methods. Section 3 reports the results by domain, ending with the contrary and null findings. Section 4 discusses what the evidence establishes, makes the case for continuous gate-level monitoring and states what would test it. Sections 5 and 6 give the limitations and conclusions.

## 2 Methods

## 2.1 Design and reporting

We conducted a literature review guided by the PRISMA statement [7] and reported the searches against PRISMA-S [8]. The searches are systematic and the review is structured, but it departs from a full systematic review in four ways, stated here and in Section 5. Screening and data extraction were carried out by a large language model under the authors’ direction, and the authors checked its decisions. Data were extracted from titles, abstracts and texts, with full texts read for key records. The twelve database searches were all run between $2 ^ { \mathrm { n d } } - 4 ^ { \mathrm { t h } }$ November 2026 and were completed before screening, extraction or writing began. The grey-literature and hand-searches followed the first screening pass and were aimed at gaps it exposed. A mapping of the PRISMA items to this paper is in Appendix B.

## 2.2 Eligibility criteria

We included studies, reports and official or professional documents that gave data, analysis or evaluation on one of seven domains: D1 theft of livestock and equipment; D2 predators and dogs; D3 gates, public access and straying; D4 prevention and detection other than AI; D5 camera-trap AI; D6 under-reporting, recording, trust and data quality; and D7 costs and impact on farmers. Farmers, farms or agricultural premises had to be the population or setting. Records had to be published between 2000 and 2026 in English. Peer-reviewed articles, reviews and book chapters were eligible, as were conference papers for the technology questions. Grey sources were eligible if they came from government, police, regulators, parliaments, farming bodies, insurers or universities.

We excluded ecology and conservation studies with no farm or livestock link, large-carnivore studies that measured no livestock outcome, crop-pest and yield studies with no security aim, opinion pieces with no data (unless they supplied a unique UK fact), duplicates, and non-English or pre-2000 records. Reviews of predation interventions in large carnivores were kept as evidence-quality framing for the UK-relevant evidence.

## 2.3 Information sources and search

We searched the Web of Science Core Collection (all editions) and Scopus (advanced search) with six queries each, one per topic area (Table 1; full strings in Appendix A), limited to 2000–2026 and English. Query 2 excluded large-carnivore terms from titles. Web of Science hit counts before refinement were not recorded, and all Scopus hits were exported. The grey-literature arm comprised 29 web searches of official, police, regulatory, parliamentary, insurer and farming-body sources. It retrieved and assessed 35 sources, including two articles from the International Journal of Rural Criminology that neither database indexes [9,10]. We did not search Google Scholar, CAB Abstracts, Criminal Justice Abstracts or ProQuest (Section 5).

Table 1 | Database searches run on 4 November 2026. Records exported per query from Web of Science (WoS) and Scopus; ‘held’ means already in the Web of Science set or an earlier Scopus query.
<table><tr><td colspan="1" rowspan="1">No.</td><td colspan="1" rowspan="1">Topic</td><td colspan="1" rowspan="1">WoSexported</td><td colspan="1" rowspan="1">Scopusexported</td><td colspan="1" rowspan="1">Scopus held</td><td colspan="1" rowspan="1">Scopus new</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Theft of livestock and farm equipment</td><td colspan="1" rowspan="1">67</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1">60</td><td colspan="1" rowspan="1">51</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">Predation and livestock worrying</td><td colspan="1" rowspan="1">415</td><td colspan="1" rowspan="1">364</td><td colspan="1" rowspan="1">292</td><td colspan="1" rowspan="1">72</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">Gates, straying and public access</td><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">16</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">Crime prevention and detectiontechnology</td><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">66</td><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">45</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">Camera-trap AI</td><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">43</td><td colspan="1" rowspan="1">23</td><td colspan="1" rowspan="1">20</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">Under-reporting and data quality</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">3</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Total</td><td colspan="1" rowspan="1">557</td><td colspan="1" rowspan="1">626</td><td colspan="1" rowspan="1">419</td><td colspan="1" rowspan="1">207</td></tr></table>

## 2.4 Selection process

Records were de-duplicated by DOI or normalised title. A single reviewer, a large language model (Claude, Anthropic) working from the exported titles, abstracts and texts under the authors’ direction, assigned each unique record Include, Maybe or Exclude with a reason and an evidence tier. The 92 Maybe records were then re-read against a gap rule. A Maybe was promoted if it was among the few sources on a domain that the first-pass evidence map showed to be thin (camerabased detection of farm targets, the durability or independent testing of deterrents, the quality of predator-intervention evidence, public access, or UK methods), and excluded if it duplicated stronger evidence or was off topic. The rule was set after the first pass and is therefore post hoc.

## 2.5 Data extraction and evidence tiers

For each included record we extracted setting, design, sample, evidence type, primary and other domains, the key finding, limitations, and how the record is used (directly for the UK, transferable from elsewhere, or context). Extraction was from the title, abstract and text, and where an abstract was silent the field says so. Full texts were read for all UK records with relevance as per and for many grey sources, using an automated reader (figures from those sources were checked by the authors against the originals). Two book chapters had no abstract and could not be extracted [14,15]. Fourteen overlaps or possible overlaps between records (for example two papers reporting the same Irish survey) were flagged and the samples are counted once in quantitative statements.

Each record was placed in an evidence tier by setting: Tier 1, UK; Tier 2, other high-income countries; Tier 3, other countries; and Tech, technology development papers whose setting is not a farm deployment. The tiers describe how directly a record applies to a UK farm and are not a ranking of quality.

## 2.6 Synthesis and appraisal

We synthesised the evidence narratively by domain and mapped it by domain and tier. We did not pool results. We did not apply a formal risk-of-bias tool or grade certainty. Instead each record carries its evidence type and limitations, and we describe the strength of each body of evidence in words. Findings that cut against the case for technology, and findings that are null or weak, are reported prominently (Section 3.9 and Table 4). The same search, screening and reporting rules were applied to evidence that supports and evidence that cuts against continuous monitoring.

## 2.7 Language-model screening and extraction

A large language model screened all 761 records and extracted the data, working under the authors’ direction from records the authors exported from Web of Science and Scopus. The authors checked the model’s screening and extraction decisions against the source records for accuracy.

## 3 Results

## 3.1 Study selection and characteristics

The searches returned 1,183 records (557 from Web of Science, 626 from Scopus). After 422 duplicates were removed, 761 unique records were screened on title and abstract: 588 were excluded, 81 included and 92 held as Maybe. On re-reading against the gap rule, 15 Maybes were promoted, which gave 96 database records. Of 35 grey sources assessed, 24 were included, 10 were kept as context only and one was not relied on. In all, 120 sources entered the synthesis (Fig. 1).

![](images/97415f09033812794ad6fcae85d62d98010467361fc35080a85c0437a125e686.jpg)  
Fig. 1 | PRISMA flow diagram. Database arm (left) and grey-literature arm (right). The 15 promoted records were Maybes resolved by the gap rule in Section 2.4.

By evidence tier the 96 database records were 16 UK (Tier 1), 44 other high-income (Tier 2), 17 other non-UK (Tier 3) and 19 technology papers. Volume has grown fast: 65 of 96 records (68%) date from 2020 or later, and only seven appeared between 2005 and 2014 (Fig. 2). UK peer-reviewed evidence is small and recent, with nine of its 16 records published since 2020.

![](images/c0d8b36406eb8c602aa1c95e4f5175ffc65e04dd3d90fed228abac176058f757.jpg)  
Fig. 2 | Database records included, by period of publication and evidence tier. Segments without a number are single records. Technology papers are those that develop or test detection systems rather than describe a farm deployment.

The evidence map (Fig. 3) shows where the weight lies. Predators and dogs hold the most database records (28), and theft the next most (18). Gates, public access and straying hold one (a study of

livestock-related fatalities), and camera-trap AI holds 12, all of them technology papers or reviews. No database record evaluates a camera-based system at a farm gate. UK peer-reviewed evidence is concentrated in predators (seven records) and is absent for prevention and detection and for camera-trap AI.

![](images/8a08eadbada70595efb98e4cfde31ad555b32149ea846e6d8264f1258c579c37.jpg)  
Counts are database records by primary domain (96) and grey sources marked Include (24). Hatched cells are empty.

Fig. 3 | Evidence map. Database records by primary domain and tier, and grey sources marked Include. Hatched cells are empty. D3 (gates, public access, straying) and D5 (camera-trap AI) are the thinnest domains for farm security.

## 3.2 Scale, nature and cost of farm crime (D1)

The insurer series. The most-used UK cost estimate is NFU Mutual’s claims-based series, which fell from £52.8 million in 2023 to £44.1 million in 2024 and £41.5 million in 2025 [1,16]. Beneath the falling total, categories moved differently (Fig. 4). In 2025 agricultural vehicle theft cost £8.3 million (up 18%), quad bike and ATV theft £3.5 million (up 31%) and livestock theft about £4.5 million (up nearly 30%), described as highly organised, while GPS-unit theft fell 80% to about £250,000 [1]. Regionally the 2025 figures were £2.2 million in Wales (from £2.8 million), £1.4 million in Scotland (from £1.2 million) and £2.2 million in Northern Ireland (from £1.8 million) [1]. The 2024 report gave livestock theft as £3.4 million in its text and £3.3 million in its infographic [16].

![](images/84348b2bad4c25585fa1c27aa79eb5d40d4abf5f21e6c3154dc5d03f92cc8032.jpg)  
Source: NFU Mutual claims-based estimates; an insurer series, not recorded crime. Livestock worrying from a separate NFU Mutual release

Fig. 4 | UK rural crime cost. a, Total estimated cost, 2023–2025 [1,16]. b, 2025 cost by category with the change on 2024 [1,2]. All figures are claims-based insurer estimates, not recorded crime.

What the insurer series cannot tell us. It covers insured losses only, the method of scaling from NFU Mutual claims is not given on the summary page, and an insurer has a commercial interest in the rural-risk narrative [1]. A Harper Adams commentary argues that the true cost is many times higher, because NFU Mutual insures about three-quarters of UK farmers and research it cites suggests only about 40% of crimes are reported to insurers, and that the 6% fall may be cyclical. Her extrapolations are the author’s own assumptions, not data [17]. The two readings differ by an order of magnitude and the sources cannot arbitrate between them.

Official and survey data. The Home Office Commercial Victimisation Survey 2017 estimated that 27% of agriculture, forestry and fishing premises in England and Wales were victims of crime (30% in 2013), about 113,000 crimes, of which vandalism was about 37,000, theft 28,000 and burglary 18,000. The sector includes forestry and fishing, and farming is the main activity for 780 of 1,019 sampled premises [18]. An earlier press report of the 2014 survey gave about a quarter of premises affected [19]. In Northern Ireland, police recorded 205 agricultural crimes in the year to March 2026, counting only burglary, robbery and theft with an agricultural link [20]. Self-selecting national surveys give larger numbers with larger uncertainty: the 2015 National Rural Crime Survey (over 17,000 respondents) put the average cost of crime to a rural business victim at £4,100 [21], and the 2018 survey (20,252 responses) put the average cost per offence at £4,800, up 13% [22]. Table 2 collects the main UK figures and their limits.

Table 2 | Principal UK figures on farm crime, with their limits. Sources are cited by number; the figures differ in what, where and when they measure and are not additive.
<table><tr><td rowspan=1 colspan=1>Measure</td><td rowspan=1 colspan=1>Figure</td><td rowspan=1 colspan=1>Source type</td><td rowspan=1 colspan=1>Limits</td></tr><tr><td rowspan=1 colspan=1>Cost of UK rural crime,2025</td><td rowspan=1 colspan=1>£41.5m, down 6% on 2024(£44.1m); £52.8m in 2023 [1,16]</td><td rowspan=1 colspan=1>Insurer claims</td><td rowspan=1 colspan=1>Insured losses only; insurerhas a commercial interest</td></tr><tr><td rowspan=1 colspan=1>Livestock theft cost, 2025</td><td rowspan=1 colspan=1>About £4.5m, up nearly 30% [1]</td><td rowspan=1 colspan=1>Insurer claims</td><td rowspan=1 colspan=1>Summary page; full reportnot retrieved</td></tr><tr><td rowspan=1 colspan=1>Livestock worrying cost,2025</td><td rowspan=1 colspan=1>£1.95m, up 10% [2]</td><td rowspan=1 colspan=1>Insurer claims andowner survey</td><td rowspan=1 colspan=1>Sample sizes not given</td></tr><tr><td rowspan=1 colspan=1>Premises victimised,agriculture, forestry andfishing, E&amp;W, 2017</td><td rowspan=1 colspan=1>27% (30% in 2013); about 113,000crimes [18]</td><td rowspan=1 colspan=1>Home Office survey</td><td rowspan=1 colspan=1>Includes forestry and fishing;small sector sample</td></tr><tr><td rowspan=1 colspan=1>Agricultural crimesrecorded, Northern Ireland,year to March 2026</td><td rowspan=1 colspan=1>205 (201 the year before) [20]</td><td rowspan=1 colspan=1>Police flag</td><td rowspan=1 colspan=1>Theft, burglary and robberyonly</td></tr><tr><td rowspan=1 colspan=1>Dog attack or worryingincidents recorded,Scotland, year to March2025</td><td rowspan=1 colspan=1>216, against 350 the year before[23]</td><td rowspan=1 colspan=1>Police Scotland viatrade body</td><td rowspan=1 colspan=1>Recorded incidents only</td></tr><tr><td rowspan=1 colspan=1>Sheep farmers reporting adog attack on their flock,2024</td><td rowspan=1 colspan=1>87% [24]</td><td rowspan=1 colspan=1>NSA survey quotedby Defra</td><td rowspan=1 colspan=1>Survey not retrieved</td></tr><tr><td rowspan=1 colspan=1>Welsh farmers who hadexperienced crime</td><td rowspan=1 colspan=1>About half; 91% of victimsreported it [11]</td><td rowspan=1 colspan=1>Survey, n = 96</td><td rowspan=1 colspan=1>Recruited mainly at a show;descriptive only</td></tr><tr><td rowspan=1 colspan=1>Welsh farms victimised inpast year</td><td rowspan=1 colspan=1>22%; machinery theft 65% ofcrimes [25]</td><td rowspan=1 colspan=1>Survey, over 150</td><td rowspan=1 colspan=1>News release only; report notlocated</td></tr><tr><td rowspan=1 colspan=1>British farmers who werevictims</td><td rowspan=1 colspan=1>72 of 80 (90%) [12]</td><td rowspan=1 colspan=1>Self-selected onlinesurvey</td><td rowspan=1 colspan=1>5.1% response; prevalencecannot be inferred</td></tr><tr><td rowspan=1 colspan=1>Rural business victims,E&amp;W</td><td rowspan=1 colspan=1>Average cost £4,100 (2015) [21];£4,800 per offence (2018) [22]</td><td rowspan=1 colspan=1>Self-selectingsurveys</td><td rowspan=1 colspan=1>Not farm-specific; mayoverstate</td></tr></table>

UK primary evidence is thin. Two database records are UK peer-reviewed studies of theft: a horizon scan of emerging rural crime in England [26], which argues that English rural crime research is thin compared with that in the USA and Australia, and a multi-modal survey [27] (a chapter whose abstract gives no figures) covering victimisation, under-reporting and trust. A 2017 scoping survey, found by hand-search, had 71 complete responses from a snowball sample [10], and a methods paper sets out the advantages and disadvantages of online surveys of British farmers [28]. In Wales a farm crime survey found that machinery made up about half of crime and livestock 25% [11]. The practical consequence is that a UK farmer or policy maker cannot read a livestocktheft rate from the literature. NFU Cymru says livestock theft is getting worse in Wales, but gives no figures [29].

Mechanisms that transfer. Other countries supply mechanisms rather than rates. Australian surveys found that burglary and theft of fuel and machinery were inversely related to how visible farm buildings were from the farmhouse, and that stock theft rose with distance from town and with hilly terrain [30,31]; US work found that opportunity theory only partly explains victimisation [32]. A Pakistani case series of 53 farmers whose cattle were stolen found theft mostly at night, with repeat victimisation, and all respondents willing to invest in RFID tagging [33]. In South Africa, provincial police records for 2018–2024 show 30,816 stolen cattle in the Eastern Cape (R399 million), though species and recovery were weak predictors [34], and theft was significantly more likely in winter [35]. Surveys in Ireland, Slovenia, the USA, Australia, Nigeria and New Zealand report high victimisation (just over half of respondents in Ireland, 66% in north-west Nigeria) and, in several, under-reporting [36–41]. A Caribbean review finds farm theft hard to prove and lightly penalised [42], Swedish media archives show suspects are not one group [43], South African communities and farmers explain livestock theft differently [44], and South African farmers spend heavily on security with no single solution [45]. Equipment theft has its own policy track: the Equipment Theft (Prevention) Act will allow regulations on marking, registration and removable GPS units for new quad bikes and ATVs, of which 900 to 1,200 are stolen each year in England and Wales, but it has no effect until regulations are commenced [46].

## 3.3 Predators and dogs (D2)

Foxes: small share, hard attribution. The UK evidence on foxes suggests a smaller effect than farmer perception. On two Scottish hill farms lamb mortality was 9.3% to 10.2%, but confirmed fox predation was at least 0.2% to 0.6% of lambs and 53 lambs went missing, which cannot be attributed [47]. In a British survey, 59% of sheep farmers had lost at least one lamb to a fox at their last lambing, with reported losses of 0.0008 to 0.26 lambs per ewe [48]. Losses were perceived rather than confirmed, and the effect of fox control could not be determined because farmers react to losses [48]. A Scottish study combining farmer interviews, post-mortems and DNA found predation in 14 of 29 lambs (48%) with fox DNA on 87% of swabbed lambs, dog DNA on 31% and badger DNA on 5%, and no direct evidence that badgers kill lambs [49] but Badgers killing lambs has been witnessed by Arwyn Jones (an author on this paper and farmer in Wales). For free-range poultry, mean mortality was under 2% and fox predation was not linked to fox density, housing or fencing type [50].

Dogs: rising cost, patchy records. NFU Mutual estimated £1.95 million of livestock losses to dogs in 2025 (up 10%), and a dog-owner survey it cites found 57% let dogs off the lead in the countryside and one in ten admit no recall [2]. A National Sheep Association survey quoted by Defra found 87% of sheep farmers reported a dog attack on their flock in 2024 [24]. In Scotland, Police Scotland recorded 216 incidents in the year to March 2025 against 350 the year before [23]; an earlier briefing records 285 incidents in 2018/19 and 265 in 2019/20, and quotes Scottish Government research in which about 14% of nearly 2,000 sheep farmers had experienced worrying in the previous 12 months and around a third of attacks were reported [51]. A review of news coverage concluded that sheep worrying appears to be increasing and that prevention methods have not been evaluated [52]. The legal position changed on 18 March 2026, when the Dogs (Protection of Livestock) (Amendment) Act 2025 came into force in England and Wales with unlimited fines (previously £1,000), police powers to seize dogs and take DNA samples, and extended coverage of roads and paths [24,53]. The Welsh strategy describes a DNA project comparing attacked animals with suspected dogs [54]. Elsewhere, Sweden found that dogs accounted for 3.8% of inspected

predator attacks and were usually loose or unsupervised [55], a survey of agriculture-ministry professionals in Chile found free-ranging dogs attacked 25% of small farms in a year [56], and in Uruguay over 85% of 91 sheep farmers reported predator attacks and distress was higher among farmers whose animals were attacked [57,58].

Cameras reveal losses that records miss. Two Australian field studies used cameras as the measuring instrument. At an outdoor piggery, about 20% of piglets were recorded as missing, and cameras caught foxes taking live piglets soon after staff left [59]; around guardian-dog paddocks, foxes appeared on camera on 17 of 23 nights and the dogs mostly guarded the poultry shelters, not the paddock [60]. Neither study set out to evaluate a security product, but both show that continuous observation can answer questions that periodic records cannot.

## 3.4 Gates, public access and straying (D3)

This is the thinnest domain. One database record has public access as its main topic: an analysis of Health and Safety Executive fatal-injury reports found 78 livestock-related deaths in Great Britain from 2010 to 2023, 97% involving cattle. Every member of the public killed was in a field, and 84.2% had a dog with them; the public were 250 times more likely than farmers to have a dog present [13]. It counts fatalities only, has no exposure denominator, and mentions gates only as handling infrastructure. HSE guidance says gates at entry points should be as high and strong as the perimeter fence, with a securing device that children and animals cannot release, and that fences, gates and stiles should be checked at least once a day [61]. It adds that many incidents go unreported [61]. Press reporting of HSE guidance and HSE statistics give the scale of public deaths in agriculture-related incidents (about one to two a year from cattle on rights of way; four in 2024/25 provisional, all causes) [62,63]. The Countryside Code asks the public to leave gates as they find them and to keep dogs under control, but the edition read has no livestock guidance [64]. Trespass appears as a secondary finding: 66.7% of 72 victims in one British survey reported it [12], and in Swedish media archives vandalism and trespass were common at mink farms [43].

No study found measured how often gates are left open, what follows when they are, or whether any intervention changes it. Queries 3 in both databases returned 48 records and none was about gates, straying or access in this sense.

## 3.5 Prevention and detection other than AI (D4)

Evidence of effect is weak or negative. The largest survey-based test of security practices found no association with lower victimisation: in a mail survey of 1,100 New South Wales farmers, crime was highly situational and security practices were not linked to lower victimisation [31]. A US programme evaluation suggested that marking equipment and livestock, surveillance, locked storage and prosecution may reduce victimisation, with the authors cautioning that design limits the finding [65]. Everything else on theft prevention is adoption, perception or a prototype. Irish farmers support prevention but do not widely practise it [36], some cannot afford it or lack the time [66], and Slovenian farmers rely on their own situational prevention [37]. In Wales, over 80% of respondents took precautions, but the study did not test whether they were linked to victimisation [11]. A union poll of nearly 2,000 farmers reported that 35% had installed CCTV, more than half had upgraded building security, and 31% had spent at least £1,000 on crime prevention over five years [67]. A survey of police forces (33 of 38 contacted responded) found technology including CCTV widely used and just over half of respondents rating methods effective, based on opinion, not outcomes [9].

No independent evaluation of cameras, sensors, trackers or marking. Insurer reports recommend CCTV, lighting, alarms, trackers, immobilisers and marking as measures “proven to deter determined criminals” but give no data on effect, and one case study in the same report shows thieves knowing where the CCTV was and footage capturing little [16]. Police Scotland recovered £893,000 of equipment in 2019 and a press report credits trackers, but there is no tracked-versusuntracked comparison, and an insurer discount bundles marking, immobiliser and tracker so the

effect of the tracker cannot be isolated [68]. The Home Office call for evidence records respondents saying that etching and microdot marks can be removed and that no provider yet meets the relevant standard [46]. A trade claim of a 0.06% theft rate for marked machinery has no source, method or period, and we did not rely on it [69]. Other police and policy sources are context: a national rural crime unit funded in part externally and by an insurer [70], a national vehicle-crime intelligence service whose published cases include no agricultural theft statistics [71], a national policing strategy known only through a summary, which says technology will support intelligence-led policing [72], and a Welsh strategy that sets no numeric targets and describes no system for recording farm crime beyond “centralised intelligence” [54].

Prototypes report high lab accuracy and no theft outcomes. A Bangladeshi multi-sensor system detected 100% of intrusions with its laser sensor, up to 90.5% with passive infrared sensors at night, and an average efficiency of 92.5% by day and 94.38% at night, tested in its developers own conditions with no theft outcome [73]. A related system from the same group reported 90% overall efficiency in two papers that describe one study [74,75]. Other prototypes use low-power radio tracking, motion-triggered GPS and text alerts, edge-computing arguments for intrusion prealarms, a farm-security video-alert description, a microcontroller sound classifier for orchard fruit theft and GPS tracking of patrols, none reporting a reduction in loss [76–81]. In South Africa, only 28% of 137 communal farmers knew of GPS tracking devices and 42% would buy one [82]; adoption depends on awareness, how acute theft is, income and phone access [83]; and GPS and CCTV networks are out of reach for many emerging farmers [84]. Smallholder studies recommend tracking and community police forums without testing them [85], and management level is associated with lower theft losses [86].

What works in predator management, and what does not. Reviews are consistent in their message about the quality of evidence. A systematic review of interventions against large-carnivore predation found experimental and quasi-experimental studies rare and the evidence on effectiveness limited [87]. A review of 75 field experiments found that 12 of 22 strategies were evaluated only once or twice, and that none evaluated more than once had both high robustness and consistently positive outcomes [88]. A review of 291 cases found lethal control the commonest method but poorly evaluated, with no effect in its analysis, whereas guarding dogs, fencing or herdsmen may significantly reduce losses [89]. Table 3 summarises the measures.

Table 3 | Evidence on the effectiveness of farm-security measures. No row rests on an independent evaluation of a camera-based system.
<table><tr><td colspan="1" rowspan="1">Measure</td><td colspan="1" rowspan="1">Best evidence found</td><td colspan="1" rowspan="1">Finding</td><td colspan="1" rowspan="1">Reading</td></tr><tr><td colspan="1" rowspan="1">General securitypractices (locks,visibility, marking)</td><td colspan="1" rowspan="1">NSW survey of 1,100farmers [31]; Californiaprogramme evaluation[32,65]</td><td colspan="1" rowspan="1">No association with lower victimisation;programme authors say marking,surveillance, hardening and prosecutionmay help, with limited design</td><td colspan="1" rowspan="1">Null or suggestive;data from 2000 and2007</td></tr><tr><td colspan="1" rowspan="1">CCTV, alarms,trackers, marking</td><td colspan="1" rowspan="1">Insurer and police reports[16,68]; Home Office [46];union poll [67]</td><td colspan="1" rowspan="1">Adoption reported (35% had CCTV);recommended as proven without data;trackers aided recoveries with nocomparison; marks removable</td><td colspan="1" rowspan="1">No independentevaluation found</td></tr><tr><td colspan="1" rowspan="1">Sensor and IoTprototypes</td><td colspan="1" rowspan="1">Bangladeshi fieldvalidation [73–75]; others[76,77,80]</td><td colspan="1" rowspan="1">Laser 100% detection; passive infrared upto 90.5% at night; average 92.5% by day,94.4% at night</td><td colspan="1" rowspan="1">Developer-reported;no theft outcome</td></tr><tr><td colspan="1" rowspan="1">Electric fences andphysical barriers</td><td colspan="1" rowspan="1">[57,89-91]</td><td colspan="1" rowspan="1">Attacks 8.3 times lower with electric fencesin every paddock (n = 91); 100% damagereduction for 3 months to 3 years inreviewed cases; fenced poultry runs hadless predation</td><td colspan="1" rowspan="1">Most consistentpositive evidence, forpredators not theft;cost often prohibitive[92]</td></tr><tr><td colspan="1" rowspan="1">Light and sounddeterrents</td><td colspan="1" rowspan="1">[90,93-95]</td><td colspan="1" rowspan="1">Foxlights linked to 12% more fox activity;acoustic and light deterrents lost effectwithin one to five months; flashing ear tagsreduced reported predation at two of 13operations but most broke</td><td colspan="1" rowspan="1">Contrary or early-stage</td></tr><tr><td colspan="1" rowspan="1">Lethal predatorcontrol</td><td colspan="1" rowspan="1">[48,89,91,96–99]</td><td colspan="1" rowspan="1">No effect on lamb production; calf lossoften higher where baited; extra fox controlnot worthwhile on most British farms;culling can reduce fox numbers</td><td colspan="1" rowspan="1">Null or contrary forloss</td></tr><tr><td colspan="1" rowspan="1">Guardian animals</td><td colspan="1" rowspan="1">[60,100,101]</td><td colspan="1" rowspan="1">Cut coyote lamb losses 43%, but costsexceeded benefits by US$13,413 on onefarm; half of users still using after 8.9 years</td><td colspan="1" rowspan="1">Mixed; farm-level</td></tr><tr><td colspan="1" rowspan="1">Police andcommunityprogrammes</td><td colspan="1" rowspan="1">[9,72,84,102]</td><td colspan="1" rowspan="1">Forums show promise withoutdemonstrated impact; a county programmedid not restore production</td><td colspan="1" rowspan="1">No outcomeevaluation</td></tr><tr><td colspan="1" rowspan="1">Housing andmanagement</td><td colspan="1" rowspan="1">[48,98]</td><td colspan="1" rowspan="1">Indoor lambing preventive; house ewes forno more than a day after lambing tominimise cost</td><td colspan="1" rowspan="1">Observational andmodel-based</td></tr></table>

Deterrents decay; barriers last. A systematic review of 26 cases found that electric fences and calving control stayed highly effective for the longest (100% damage reduction for three months to three years), while acoustic and light deterrents and guarding animals lost effect within one to five months. Its authors say monitoring should be a routine requirement for every intervention [90]. In a field experiment on one piggery, commercial Foxlights were associated with 12% more fox activity and 23% fewer piglet births recorded, and motion-activated spotlights had no benefit [93]. A captive-coyote experiment found that light alone gave little protection and that movement doubled efficacy [94]. Motion-triggered flashing ear tags reduced reported predation at two operations with rigorous data, though most tags broke [95]. Fox culling can depress fox density in some regions [99], but fox control had no effect on lamb production in a two-year Australian experiment [96], calf loss was often higher where wild-dog baiting occurred [97], and fox pest status did not reduce fox numbers or poultry damage in France whereas fencing runs did [91]. Observational Australian work found higher lamb survival where baiting was repeated and neighbours also baited [103], though a review notes variable reduction in fox activity and fast recolonisation [104]. Ranchers judge lethal measures more effective than non-lethal ones, though this is perceived efficacy [105].

Cost-effectiveness is often unfavourable. Guardian dogs on a representative California sheep operation cut coyote losses of lambs by 43% but cost US\$13,413 more than they saved over seven years [100]. Cooperative cell-fencing in Western Australian rangelands returned a positive net present value in only 37.4% of 3,600 modelled scenarios [92]. A county non-lethal predator programme left sheep numbers still falling and five producers had left the industry [102]. A practitioner commentary describes control failing when tools are misapplied or one is overused [106], and a welfare review rates guardian animals, and perhaps shepherds, as doing most to protect sheep once effectiveness and the method’s own impact are combined [107]. Taken together these results say that cost and effect have to be shown for each measure on each farm, and that few have been.

## 3.6 Camera-trap AI (D5)

Twelve database records sit in this domain: 11 technology papers and one review of farmland camera-trap studies. None evaluates a camera-based system at a farm gate or in a farm-security deployment. They show what the technology can do in other settings, with the limits described below.

Person and vehicle detection is mature in ecological data. A validation of the open-source TrapTracker model against manual classification of a recreation-ecology dataset found that, at a

0.90 confidence threshold, humans were detected at 99% precision and 95% recall and animals at 82% and 92%. Processing was over 500% faster and manual effort fell 8.4-fold; the workflow could anonymise human images before any further classification [108]. An ensemble method separating wildlife from human-activity images reached 95.75% accuracy and 94.07% recall, and noted that human images can dominate datasets and cause wildlife to be misclassified as human [109]. A temporal-context detector designed for low, irregular frame rates from motion-triggered cameras beat a single-frame baseline by 17.9% mean average precision and was applied to vehicle detection on traffic cameras [110]. A comparison of deep-learning detectors on 2,462 images of animals that threaten farm animals gave YOLOv8 93% precision and 85.2% recall [111], and a pasture-camera comparison on 39,364 frames from nine solar trail cameras gave a best F1 of 0.79, with cattle confused with a visually similar class [112]. A survey of 105 studies of AI for human-wildlife conflict reports improved monitoring, but it concerns conservation conflict and not farm crime [113].

Limits that matter for a farm gate. In a Tanzanian dataset analysts detected 81.1% of wildlife and classified 76.6% correctly, against 41.1% and 47.5% for machine learning; for humans, performance was similar by day (69.4% for machine learning, 71.8% for analysts) and poor at dusk (17.6% and 16.2%) (However, modern cameras and models overcome this issue and work well at dusk/nighttime), with distance, occlusion and vegetation having the largest effects [114]. One author of that study (P. Fergus) is also an author of this review, which is disclosed here as a self-citation. Most abstracts report accuracy on ecological or image-set data and say little about false alarms or the cost of a missed event.

Reviews and proposals identify the gap. A structured review of AI-driven intrusion detection and deterrence for farm protection finds no comprehensive end-to-end framework, little adaptive deterrence and heavy reliance on continuous human supervision [115]. A review of 30 years of camera-trap publications found few studies sampling farmland [116]. Proposals and demonstrations describe edge detection with local deterrent action [117], intelligent access control with securitycamera video and a stated rural-crime driver [118], and comparisons of image classifiers for farmperimeter threats that report no figures in the abstract [119].

## 3.7 Reporting, recording and trust (D6)

Recording. In England and Wales farm crime cannot be separated in central data [3,4]. A 2012 paper reports that an earlier national study of police forces found no uniform approach to recording farm crime within or across forces and no nationally identifiable point of contact; we know that study only second-hand [120,121], and a 2017 scoping study repeats the point about inconsistent recording [10]. Northern Ireland counts agricultural crime through a flag with keyword searches [20], and Scottish farm crime is identified by keyword search of police reports [122]. Ireland collects no official farm-crime statistics [123]. The Welsh strategy describes “centralised intelligence” and no recording system for farm crime [54]. Methodological papers explain why survey evidence from farmers carries limits [28,124].

Reporting is not uniformly low. In Wales, 91% of victims reported the crime (n = 96, showrecruited) [11]. In the 2015 National Rural Crime Survey, 27% did not report the last crime they experienced [21], and in 2018 only 28% of farm and rural business victims made an insurance claim, according to a secondary report of that survey [125]. In Scotland about one-third of dog attacks are reported [51]. In the Free State, 82% did not report livestock theft because they believed police would do nothing [126], and in north-west Nigeria 66% were victims and reporting was lowest where trust in police was lowest [40]. New Zealand, Ireland and Slovenia report some underreporting [36,37,41]. Rural community norms, penalties and the difficulty of proof are reasons given for silence [42,45,127], and HSE says many incidents of cattle harm go unreported [61]. A commentary adds that insurers see only part of the picture [17].

Trust. Welsh farmers’ trust in police was 64% against 75% for rural respondents nationally, and victim satisfaction was 54% against 73%, although the abstract calls trust generally healthy [11]. A Welsh survey of 134 farmers and rural residents found good relations with police and dissatisfaction with response times, follow-up and prosecutions [128], and a Welsh news release reports 78% of victims satisfied with the police response [25]. Irish farmers generally hold positive or neutral views [123]. The 2015 survey found 39% of rural residents rated police good or excellent against 63% nationally [21], and in the USA favourable views of police were associated with higher fear of crime [129]. The UK data found here do not show distrust driving non-reporting, a link that appears more clearly in South African and Nigerian findings [40,126]. Police in Slovenia say their work depends on resident trust [130], and South African media coverage of livestock theft does not track prevalence [131]. Overviews and one title-only chapter complete the picture [15,132].

## 3.8 Impact on farmers (D7)

The main UK survey of crime and mental health had 80 completed responses from 1,570 who opened it (5.1%), a margin of error of about plus or minus 10%, and 72 respondents who had been victims; 63 of those 72 (88%) said crime had harmed their mental health, with repeat victimisation linked to frustration. Crime ranked fourth among stressors in an open question (21 mentions) behind weather (38), finances (30) and legislation (22) in the report of the same data [12]. A UK and Canada comparison of 569 farmers found finances the main stressor in both countries, with UK farmers reporting more stress from animal health, farm or rural crime and isolation [133]. Conceptual work frames agricultural crime as an acute stressor that can outweigh positive affect from farming [134]. Cross-sectional studies elsewhere link predator attacks to distress (Uruguay, 442 farmers) and wild-dog attacks to intrusive thoughts and poor sleep [58,135], and stock theft to worse household finances [85], with theft costs higher where management level was lower [86]. An Irish local study found fear of crime relatively high but few participants victimised [66]. The Welsh strategy cites a survey in which 96% of NFU Mutual agents reported negative effects of rural crime on farmers’ wellbeing [54], and the Harper Adams commentary notes that insurer cost totals omit mental-health effects [17].

## 3.9 Contrary, null and weak findings

Many findings cut against a simple story of rising crime and technology as the answer. They are collected in Table 4. Total insurer cost fell in 2024 and 2025 while livestock theft rose; reporting is high in one Welsh survey and low in others; trust in UK police is mixed and not clearly a driver of non-reporting; fear of crime may exceed local victimisation; security practices were not associated with lower victimisation in the largest survey-based test; deterrents are short-lived or counterproductive; lethal predator control often does nothing for loss; confirmed fox predation is a small share of lamb loss; guardian dogs and fencing often fail a cost test; and old machine learning models performed far below human analysts for wildlife in the one direct comparison. None of this shows that continuous monitoring would fail. It shows what evidence would be needed before it is taken to work.

Table 4 | Contrary, null and weak findings, and what each means for the case for AI monitoring.
<table><tr><td colspan="1" rowspan="1">Topic</td><td colspan="1" rowspan="1">Finding</td><td colspan="1" rowspan="1">Sources</td><td colspan="1" rowspan="1">Bearing on the case</td></tr><tr><td colspan="1" rowspan="1">Cost trend</td><td colspan="1" rowspan="1">UK total fell from £52.8m (2023) to £41.5m(2025) while livestock theft rose nearly 30%in 2025</td><td colspan="1" rowspan="1">[1,16]</td><td colspan="1" rowspan="1">Livestock and quad bike theft arecategories where access routes matter;totals alone do not make the case</td></tr><tr><td colspan="1" rowspan="1">Reporting</td><td colspan="1" rowspan="1">91% of Welsh victims reported; 27% did notin the 2015 survey; one-third of Scottish dogattacks reported; 82% non-reporting in theFree State</td><td colspan="1" rowspan="1">[11,21,51,126]</td><td colspan="1" rowspan="1">Under-reporting varies by setting, so afarm-level record is needed to size it</td></tr><tr><td colspan="1" rowspan="1">Trust in police</td><td colspan="1" rowspan="1">Welsh and Irish farmers broadly positive;distrust-driven non-reporting appears inSouth Africa and Nigeria</td><td colspan="1" rowspan="1">[11,40,123,126,128]</td><td colspan="1" rowspan="1">Non-reporting in the UK should not beassumed to arise from distrust</td></tr><tr><td colspan="1" rowspan="1">Perceptionandexperience</td><td colspan="1" rowspan="1">Irish farmers rated crime high nationallyand low locally; few victimised</td><td colspan="1" rowspan="1">[66]</td><td colspan="1" rowspan="1">Verified local events would separatefear from experience</td></tr><tr><td colspan="1" rowspan="1">Securitymeasures</td><td colspan="1" rowspan="1">No link between security practices and lowervictimisation (NSW); US programmesuggestive only</td><td colspan="1" rowspan="1">[31,32,65]</td><td colspan="1" rowspan="1">Measures need testing on real events,which needs a record of events</td></tr><tr><td colspan="1" rowspan="1">Deterrentdevices</td><td colspan="1" rowspan="1">Foxlights attracted foxes; acoustic and lightdeterrents lost effect in one to five months</td><td colspan="1" rowspan="1">[90,93]</td><td colspan="1" rowspan="1">Detection and verification do not relyon a deterrent effect</td></tr><tr><td colspan="1" rowspan="1">Predatorcontrol</td><td colspan="1" rowspan="1">No effect on lamb production; higher calfloss where baited; extra fox control notworthwhile on most British farms</td><td colspan="1" rowspan="1">[48,96-98]</td><td colspan="1" rowspan="1">Attribution before action avoidsineffective control</td></tr><tr><td colspan="1" rowspan="1">Predationscale</td><td colspan="1" rowspan="1">Confirmed fox predation 0.2% to 0.6% oflambs; no direct evidence of badgerpredation</td><td colspan="1" rowspan="1">[47,49]</td><td colspan="1" rowspan="1">Camera evidence can show whatperceived losses cannot</td></tr><tr><td colspan="1" rowspan="1">Cost-effectiveness</td><td colspan="1" rowspan="1">Guardian dogs cost more than they saved;cell-fencing positive in 37% of scenarios</td><td colspan="1" rowspan="1">[92,100]</td><td colspan="1" rowspan="1">Cost-effectiveness of monitoring mustalso be shown</td></tr><tr><td colspan="1" rowspan="1">MLperformance</td><td colspan="1" rowspan="1">Machine learning detected 41% of wildlifeagainst 81% for analysts; people poorlydetected at dusk by both</td><td colspan="1" rowspan="1">[114]</td><td colspan="1" rowspan="1">Local validation, night and duskperformance and false alarms must bereported</td></tr><tr><td colspan="1" rowspan="1">Technologyevidence</td><td colspan="1" rowspan="1">No independent evaluation of cameras,sensors, trackers or marking; developer-reported accuracy only</td><td colspan="1" rowspan="1">[16,68,73]</td><td colspan="1" rowspan="1">The evidence gap applies to AImonitoring too</td></tr><tr><td colspan="1" rowspan="1">Crime andmental health</td><td colspan="1" rowspan="1">Crime fourth among stressors; 80responses, 5.1% response rate</td><td colspan="1" rowspan="1">[12]</td><td colspan="1" rowspan="1">Impact evidence is small and self-selected</td></tr></table>

## 4 Discussion

## 4.1 What the evidence establishes, and where it falls short

Five conclusions follow from the 120 sources.

• Farm crime and predation cost UK farms real money, but the size is uncertain. The best-known series is an insurer’s claims data, which fell to £41.5 million in 2025 while livestock theft rose nearly 30% [1]. Survey and commentary sources suggest both higher costs and lower totals depending on assumptions [17,21,22].

• The record of events does not exist in England and Wales. Central police data cannot identify farm crime [3,4], recording has been inconsistent across forces [10,120], and reporting rates range from 91% of Welsh victims to roughly one-third of Scottish dog attacks [11,51].

• What works is better evidenced for barriers and management than for devices. Electric fencing, housing and physical protection of poultry runs have the most consistent support, deterrents decay or backfire, lethal control often does nothing for loss, and no independent evaluation of cameras, sensors, trackers or marking was found [16,57,90,91,93].

• Camera-trap AI works in ecological settings and is unproven on farms. Person detection is strong in ecological data [108], but we found no farm-gate evaluation, old machine learning vision models were well below analysts for wildlife in the one direct comparison, and both struggled with people at dusk [114]. It is hypothesized that modern vision models and cameras overcome this issue quite easily, but no data is available.

• Gates are the largest blind spot. One record concerns public access (fatalities, not gates) [13], and the rest is regulatory guidance [61,64]. How often gates are left open, who opens them and what follows is unknown.

Two features of this evidence matter for what comes next. First, the UK is under-studied: only 16 of 96 database records are UK peer-reviewed, we found no UK peer-reviewed study of a camera-based system, and the largest UK survey of crime and mental health has 80 responses [12]. Second, the

evidence is asymmetric. For measures that have been tested (fencing, baiting, lights) the results are often modest or negative; for measures that have not been tested (cameras, sensors, trackers, marking) the claims come from insurers and manufacturers. Absence of evidence is not evidence of failure, and it is not evidence of success either.

## 4.2 The case for continuous, AI-enabled gate monitoring

This section is the authors’ interpretation of the evidence above and is separate from the review findings. Our reasoning is that the largest gaps in the literature are those a continuous, gate-level monitoring system is designed to close. On that reasoning we judge the potential benefit of using Fieldmark AI for farm security to be high. Fieldmark AI applies detection models drawn from the TrapTracker camera-trap framework to imagery streamed from cameras at farm gates; in the companion case study, 14 cameras at a Welsh hill farm recorded 119 vehicles, 85 humans, 16 dog, 16 fox and 4 badger detections over three months [6] (To note that the model was specifically set to alert for security risks only [People, vehicles and potentially predators of livestock]. The model used is the largest UK mammal model in the world to date). That study has not yet measured accuracy or false-positive rates [6] as this will be done and outlined in a subsequent paper. Six lines of reasoning support the judgement (Table 5).

1. It creates the record that does not exist. Farm crime cannot be identified in central data [3,4], is found by keyword search in Scotland [122], and is recorded unevenly [120]. The survey evidence that fills the gap is small and self-selected [10,12]. A continuous system produces a timestamped, class-labelled log of every gate event that belongs to the farmer. It would not size national crime, but it would put the first farm-level, event-level data into a literature that has none.

2. It verifies at the hours and places that periodic checks miss. Theft and predation are concentrated at night: Pakistani cattle theft was mostly at night [33], and cameras at an outdoor piggery recorded foxes taking piglets soon after staff left [59]. HSE guidance asks for daily checks of fences and gates [61], which by construction cannot cover every hour. Continuous observation changes what can be known, as the Australian camera studies showed [59,60].

3. It lets the farmer attribute an event before acting. UK lamb loss is dominated by missing, unattributed lambs [47], farmers’ perceived fox losses are not confirmed [48], badgers were not shown to kill lambs [49] but we now know this to be inaccurate, and fox control did not reduce loss in several trials [91,96,98]. Classifying an event as person, vehicle, fox, badger or dog (general predators of livestock) gives the evidence on which a proportionate response (a gate, a fence, a housing decision, a call to the police) can be chosen. This avoids spending on measures that the evidence suggests do little.

## 4. It does not depend on a deterrent effect, and it supplies the monitoring the

literature asks for. Light and sound deterrents lost effect within one to five months and Foxlights drew foxes, whereas barriers lasted [90,93]. A detection-and-verification layer works whether or not an intruder is deterred, and it sits alongside the barriers that last. The authors of the durability review say monitoring should be a routine requirement for every intervention [90]; continuous gatelevel monitoring is a practical way to meet it, because a gate, a fence or a deterrent can then be tested against the events it was meant to change.

5. It addresses the unstudied domain of gates and access. HSE sets standards for gates and notes that many incidents go unreported [61], the public killed by cattle were all in fields and mostly had a dog [13], and no study measures gates left open or what follows. Detection of persons, vehicles and dogs at the gate, and of the gate’s own state, would give direct measurement where none exists. The Dogs (Protection of Livestock) (Amendment) Act 2025 gives police powers to seize dogs and take DNA samples [24,53], and the Welsh strategy describes a DNA project [54]; verified, timestamped imagery of an event is the kind of evidence such powers need.

6. It can be built privacy-first. The ecological literature already shows a workflow that anonymises human images before any further classification [108]. Continuous monitoring of a boundary raises UK GDPR questions about lawful basis, signage and retention that any deployment must answer, and the companion study flags them for its own site [6].

Table 5 | Evidence gaps and what continuous, gate-level AI monitoring could add. The third column is the authors’ inference; the fourth states what is not yet shown.
<table><tr><td rowspan=1 colspan=1>Gap in the evidence</td><td rowspan=1 colspan=1>Evidence</td><td rowspan=1 colspan=1>What continuous gate-level monitoringcould add</td><td rowspan=1 colspan=1>Not yet shown</td></tr><tr><td rowspan=1 colspan=1>No farm-level recordof events in Englandand Wales</td><td rowspan=1 colspan=1>[3,4,120,122]</td><td rowspan=1 colspan=1>A time-stamped, class-labelled log of everygate event, held by the farmer</td><td rowspan=1 colspan=1>Accuracy of the log at scale;false alarms and missed events</td></tr><tr><td rowspan=1 colspan=1>Gates, access andstraying unstudied</td><td rowspan=1 colspan=1>[13,61,64]</td><td rowspan=1 colspan=1>Direct measurement of gate state, people,vehicles and dogs at the point of access</td><td rowspan=1 colspan=1>How often gates are left openand what follows</td></tr><tr><td rowspan=1 colspan=1>Verification limited toperiodic checks anddaylight</td><td rowspan=1 colspan=1>[47,59,60]</td><td rowspan=1 colspan=1>Continuous verification at night, whenpredation and theft concentrate</td><td rowspan=1 colspan=1>Night and dusk performance onfarms; people at dusk werepoorly detected [114]</td></tr><tr><td rowspan=1 colspan=1>Prevention measuresuntested; deterrentsdecay</td><td rowspan=1 colspan=1>[16,31,90,93]</td><td rowspan=1 colspan=1>A measurement layer against which anyintervention can be tested; detection thatdoes not rely on deterrence</td><td rowspan=1 colspan=1>Effect on loss</td></tr><tr><td rowspan=1 colspan=1>Attribution ofpredation and dogattacks is uncertain</td><td rowspan=1 colspan=1>[47-49,55]</td><td rowspan=1 colspan=1>Species- and class-level imagery to tell fox,badger, dog and person apart</td><td rowspan=1 colspan=1>Accuracy for UK predators infarm settings</td></tr><tr><td rowspan=1 colspan=1>Under-reporting andevidence forenforcement</td><td rowspan=1 colspan=1>[17,21,24,51,54j</td><td rowspan=1 colspan=1>Verified, time-stamped imagery to supportreports, claims and the 2025 Dogs Actpowers</td><td rowspan=1 colspan=1>Whether it changes reporting oroutcomes</td></tr><tr><td rowspan=1 colspan=1>Privacy andproportionality</td><td rowspan=1 colspan=1>[108]</td><td rowspan=1 colspan=1>Anonymise-before-classify workflows; localprocessing</td><td rowspan=1 colspan=1>UK GDPR basis, signage andretention in practice</td></tr></table>

Why we judge the benefit high. Four points carry the judgement. The gaps the approach addresses are the largest in the literature (no event record, no gate evidence, no evaluation of any camera-based measure). Its main output, verified and time-stamped events, has value whether or not it deters. It complements and does not replace the measures with the best durability, physical barriers and management. And the categories moving fastest, livestock theft (up nearly 30%), quad bike theft (up 31%) and dog attacks on livestock (up 10%), are ones in which access routes, people, vehicles and dogs are central [1,2]. A system that produces trusted evidence on these where none existed is, in our view, highly beneficial to farmers and, we expect, to the insurers and police who work with them.

What this does not show. No study has shown that continuous AI-mediated monitoring reduces theft, predation or worrying losses, or that it pays for itself, and the cost-effectiveness record of other measures is mixed [92,100]. However, this could be down to the novelty of the approach and thus, lack of data. Detection is not prevention: a determined organised group may defeat a camera, as one insurer case study shows [16]. In the companion deployment connectivity, not detection accuracy, was the binding constraint on real-time coverage [6]. The judgement above is a reasoned expectation and should be read that way.

## 4.3 What would show that it works

## Box 1 | A minimum evaluation design

• Design. A stepped-wedge or matched-control comparison across farms, with at least 12 months of baseline, pre-registered, and with an independent evaluator who did not build the system.

• Detection quality. Recall and precision for person, vehicle, dog, fox and badger against humanreviewed samples, reported separately for day, dusk and night, with false alarms per cameraweek.

• Operations. Connectivity uptime, time from event to farmer action, and alert fatigue.

• Outcomes. Verified events by class and gate, gate-left-open events and durations, livestock deaths attributed to cause, insured and uninsured losses, and cost per verified event.

• Reporting. Reporting rate and outcome for events with and without verified imagery, with police and insurers.

• Privacy. Lawful basis, signage and retention under UK GDPR, and anonymise-before-classify for people.

• Disclosure. Report every pre-specified outcome including null and negative results, and any commercial interest of the evaluators.

## 4.4 Implications for research and policy

Three actions follow from the evidence. Recording: England and Wales need a way to identify farm and rural crime in recorded data, as Northern Ireland has $[ 3 , 4 , 2 0 ]$ . Evaluation: independent trials of cameras, sensors, trackers and marking, which the sources recommend but none reports, would settle the largest question in the field [46,90]. Access: a first study of gates, public access and livestock, covering gate-left-open frequency, its consequences and any effect of monitoring, would open an empty domain [13,61]. Larger UK samples on crime, reporting and mental health, using probability sampling, would improve every other estimate [12,28].

## 5 Limitations

This review has limits that readers should weigh. Only two bibliographic databases were searched, two International Journal of Rural Criminology articles surfaced only through hand-search, so others may exist [9,10]. Query 2 was of low precision, and English-language and 2000–2026 limits exclude other work. Screening was by one reviewer, a language model, with a human spot-check, and the gap rule for resolving Maybe records was set after the first pass.

Data were extracted from abstracts firstly, then texts, so design, sample and findings are only as the abstracts and texts state them, and two book chapters had no abstract [14,15]. Full texts were read firstly for UK records and for many grey sources by an automated reader, and therefore the key figures were checked by the authors against the originals. Several grey sources were secondary web pages, and the 2026 NFU Mutual figures rest on a summary page [1]. Insurer and trade sources have commercial interests, and some survey sources are self-selected. Samples overlap between several records, but we counted them. Publication bias is likely: null results on security measures are less likely to be published. The searches were run on a single week and the evidence has moved fast since 2020. The reasoning in Section 4.2 is an interpretation of the evidence, not a finding of the review.

## 6 Conclusions

Farm security in the UK is a problem with a known cost and an unknown shape. The literature cannot say how many farm crimes happen, how often gates are left open and who or what takes the lambs. Prevention evidence favours barriers and management over devices, and for AI the evidence is of accuracy in ecological settings, with a weaker result for wildlife and for people at dusk using old technology. Those findings are the case for evidence, and they are also the case for continuous, gatelevel monitoring that produces it.

On that basis the authors judge that using Fieldmark AI for farm security would be highly beneficial. It would give a farm the event-level record that no official source holds, verify what periodic checks miss, separate foxes from dogs from people before the farmer acts, and let every other measure be tested against what happens at the gate. That is a reasoned inference from the gaps and not a

demonstrated reduction in loss. The next step is the independent, pre-registered comparison in Box 1, and the authors welcome it.

## Declarations

Use of AI. A large language model (Claude, Anthropic) screened the records, extracted the data and synthesised the findings, under the authors’ direction (Section 2.7). The authors take full responsibility for the final text.

Author contributions. Dr Steve Symes designed and ran the literature searches, directed the AIassisted screening, extraction. Dr Symes drafted and edited the manuscript. Prof Paul Fergus, Mr Mark Carr and Mr Omar McIver each contributed short sections of text and edited the manuscript. Mr Arwyn Jones added his expert knowledge to the processes outlined in this review to ensure accuracy. All authors read and approved the final version.

Funding. This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

Ethics. This is a review of published and publicly available sources; no human participants were recruited and no new primary data were collected.

Data availability. The search strings are in Appendix A. The data-extraction workbook (96 database records and 35 grey sources, with the evidence map, contrary-findings and overlap sheets) and the screening workbooks are available as supplementary files. Exported database records are subject to the licence terms of the providers.

## References

All web addresses were accessed on 3–4 November 2026. Numbers are in order of first citation.

1. NFU (2026) NFU Mutual Rural Crime Report 2026 (covering 2025): summary page. 28 May 2026. https://www.nfuonline.com/news/nfu-mutual-rural-crime-report/

2. NFU (2026) NFU Mutual livestock worrying statistics 2025. 17 February 2026. https://www.nfuonline.com/news/nfu-mutual-livestock-worrying-statistics/

3. Johnson, D. (Home Office) (2025) Written answer on agricultural crime outcomes. UK Parliament, 13 January 2025 (read via https://parallelparliament.co.uk/question/21532/agriculture-crime).

4. Jones, S. (Home Office) (2026) Written answer on recording rural crime as a distinct category. UK Parliament, 27 March 2026 (read via https://westminsterbrief.co.uk/archive/pq/121939).

5. Symes, S., Fergus, P., Carr, M. & McIver, O. (2026) The case for integrated, AI-enabled land management platforms in UK agriculture: a review of the economic, regulatory and technological evidence. Manuscript in preparation.

6. Symes, S., Fergus, P., Carr, M., McIver, O. & Jones, A. (2026) The case for camera-trap AI in on-farm security: a case study of a 14-camera gate-monitoring deployment at Tan Y Pant Farm, North Wales. Manuscript in preparation.

7. Page, M.J., McKenzie, J.E., Bossuyt, P.M., Boutron, I., Hoffmann, T.C., Mulrow, C.D. et al. (2021) The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 372, n71. doi:10.1136/bmj.n71.

8. Rethlefsen, M.L., Kirtley, S., Waffenschmidt, S., Ayala, A.P., Moher, D., Page, M.J. et al. (2021) PRISMA-S: an extension to the PRISMA statement for reporting literature searches in systematic reviews. Systematic Reviews 10, 39. doi:10.1186/s13643-020-01542-z.

9. Nunns, Wills and Montrose (2022) Shutting the gate: a preliminary study of farm crime prevention methods used by the police in rural England and Wales. International Journal of Rural Criminology. https://pure.hartpury.ac.uk/en/publications/shutting-the-gate-a-preliminary-study-of-farm-crimeprevention-me/

10. Smith, K. & Byrne, R. (2017) Farm crime in England and Wales: a preliminary scoping study examining farmer attitudes. International Journal of Rural Criminology 3(2). https://core.ac.uk/download/159555965.pdf

11. Morris, W., Norris, G. & Dowell, D. (2020) The business of farm crime: evaluating trust in the police and reporting of offences. Crime Prevention and Community Safety. doi:10.1057/s41300-019-00083-5.

12. Smith, K. (2020) Desolation in the countryside: How agricultural crime impacts the mental health of British farmers. Journal of Rural Studies. doi:10.1016/j.jrurstud.2020.10.037.

13. Chartres, A., Oultram, J.W.H., Royden, A., Shanks, S. & Tulloch, J.S.P. (2025) Human fatalities associated with livestock in Great Britain (2010-2023). Public Health. doi:10.1016/j.puhe.2025.105965.

14. Barclay, E. (2016) Farm victimisation: the quintessential rural crime. Routledge International Handbook of Rural Criminology.

15. Peterson, J.R., Mulrooney, K. & Harkness, A. (2025) Farm crime victimisation and punitive attitudes: the cases of Australia and the United States. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-32.

16. NFU Mutual (2025) Rural Crime Report 2025 (covering 2024). https://www.nfumutual.co.uk/globalassets/farming/rural-crime/2025/rural-crime-report-2025.pdf

17. Smith, K. (2026) Rural crime figures for 2025: what we need to consider. Harper Adams University commentary, June 2026. https://cdn.harper-adams.ac.uk/document/project/260605-Rural-Crime-Figuresfor-2025-What-we-need-to-cons469430.pdf

18. Home Office (2018) Crime against businesses 2017: Commercial Victimisation Survey, agriculture, forestry and fishing factsheet. April 2018. https://assets.publishing.service.gov.uk/media/5ae32c9fed915d42f7c6bb5e/crime-against-businessesfactsheet-agriculture-forestry-fishing-2017.pdf

19. FarmingUK (2015) Home Office outlines toll of rural crime. 24 April 2015. https://www.farminguk.com/news/home-office-outlines-toll-of-rural-crime\_35488.html

20. PSNI & NISRA (2026) Agricultural and rural crime in Northern Ireland, period ending 31 March 2026. Official Statistics, 14 May 2026. https://www.psni.police.uk/system/files/2026- 05/702263285/Agriculture%20and%20Rural%20Crime%20Bulletin%20Period%20Ending%2031st%20Marc h%202026.pdf

21. National Rural Crime Network (2015) National Rural Crime Survey 2015. 11 September 2015. https://www.northyorkshire-pfcc.gov.uk/documents/national-rural-crime-network-national-rural-crimesurvey/

22. Rural Services Network (2018) National Rural Crime Survey 2018: presentation of the National Rural Crime Network survey. 12 November 2018. https://rsnonline.org.uk/images/meetings/rsn-agms/12-11- 18/RSN\_NRCN\_survey\_presentation.pdf

23. Scottish Land & Estates (2025) Reduction in livestock worrying, but vigilance remains the message (citing Police Scotland). 20 May 2025. https://www.scottishlandandestates.co.uk/news/reduction-livestockworrying-vigilance-remains-message

24. Defra (2026) New law comes into force to protect farm animals from dog attacks. GOV.UK press release, 18 March 2026. https://www.gov.uk/government/news/new-law-comes-into-force-to-protect-farm-animalsfrom-dog-attacks

25. Aberystwyth University (2026) More than a fifth of farms experienced crime in the last 12 months (Farm and Rural Crime Survey, Wales). News release, 29 July 2026. https://www.aber.ac.uk/en/news/archive/2026/07/title-288649-en.html

26. Smith, K. & Byrne, R. (2019) Horizon scanning rural crime in England. Crime Prevention and Community Safety. doi:10.1057/s41300-019-00073-7.

27. Smith, K. (2025) Crime and safety in rural England. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-10.

28. Smith, K. (2021) Conducting Online Crime and Safety Surveys with British Farmers. International Criminal Justice Review. doi:10.1177/10575677211042262.

29. NFU Cymru (2025) NFU Cymru ramps up pressure on rural crime at the Royal Welsh Show. 23 July 2025. https://www.nfu-cymru.org.uk/news-and-information/nfu-cymru-ramps-up-pressure-on-rural-crime-at-theroyal-welsh-show/

30. Barclay, E. & Donnermeyer, J.F. (2011) Crime and security on agricultural operations. Security Journal. doi:10.1057/sj.2008.23.

31. Barclay, E. & Donnermeyer, J.F. (2002) Property crime and crime prevention on farms in Australia. Crime Prevention and Community Safety. doi:10.1057/palgrave.cpcs.8140169.

32. Mears, D.P., Scott, M.L. & Bhati, A.S. (2007) Opportunity theory and agricultural crime victimization. Rural Sociology. doi:10.1526/003601107781170044.

33. Abbas, T., Muhammad, Y., Raza, S., Nasir, A. & Höreth-Böntgen, D.W. (2014) Some facts and issues related to livestock theft in Punjab province of Pakistan: findings of series of cases. Berliner und Münchener Tierärztliche Wochenschrift. doi:10.2376/0005-9366-127-166.

34. Ndzungu, K., Ngoshe, Y.B. & Jaja, I.F. (2025) Quantification of the incidence and financial implications of stock theft in Eastern Cape Province, South Africa. Tropical Animal Health and Production. doi:10.1007/s11250- 025-04634-x.

35. Ndzungu, K. & Jaja, I.F. (2024) Farmers' perceptions on stock theft in some districts of the Eastern Cape Province, South Africa. PLOS ONE. doi:10.1371/journal.pone.0310881.

36. Hughes, N. & Bowden, M. (2025) Crime and safety in rural Ireland. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-13.

37. Meško, G. & Eman, K. (2025) Rural crime and safety in Slovenia. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-19.

38. Peterson, J.R. (2025) Crime and safety in rural United States. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-23.

39. Mulrooney, K. & Harkness, A. (2025) Crime and safety in rural Australia. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-7.

40. Umar, F. (2025) Crime and safety in rural Nigeria. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-17.

41. McCormick, J. & Gibson, L. (2025) Crime and safety in rural Aotearoa New Zealand. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-16.

42. Isaac, W.A., Ganpat, W. & Joseph, M. (2017) Farm security for food security: dealing with farm theft in the Caribbean region. Agricultural Development and Food Security in Developing Nations. doi:10.4018/978-1- 5225-0942-4.ch013.

43. Ceccato, V., Abraham, J. & Lundqvist, P. (2021) Crimes Against Animal Production: Exploring the use of Media Archives. International Criminal Justice Review. doi:10.1177/10575677211041915.

44. Breetzke, G.D., Mosesi, N. & Bester, P. (2023) The 'contestation of crime': Using a spatial theory of crime to examine livestock theft among small-scale farmers in Swartruggens, North West province. South African Geographical Journal. doi:10.1080/03736245.2022.2102062.

45. Maluleke, W., Mokwena, R.J. & Motsepa, L.L. (2016) Rural farmers' perspectives on stock theft: police crime statistics. South African Journal of Agricultural Extension. doi:10.17159/2413-3221/2016/v44n2a393.

46. Home Office (2025) Equipment Theft (Prevention) Act: summary of call for evidence responses and conclusion. 17 October 2025, updated 10 March 2026. https://www.gov.uk/government/calls-for-evidence/equipmenttheft-prevention-bill-call-for-evidence/outcome/equipment-theft-prevention-act-summary-of-call-forevidence-responses-and-conclusion

47. White, P.C.L., Groves, H.L., Savery, J.R., Conington, J. & Hutchings, M.R. (2000) Fox predation as a cause of lamb mortality on hill farms. Veterinary Record. doi:10.1136/vr.147.2.33.

48. Moberly, R.L., White, P.C.L., Webbon, C.C., Baker, P.J. & Harris, S. (2003) Factors associated with fox (Vulpes vulpes) predation of lambs in Britain. Wildlife Research. doi:10.1071/WR02060.

49. George, S., Campbell, S., Martineau, H., Hale, C., Kaden, J. & Webster, L.M.I. (2024) Are Eurasian badgers responsible for lamb predation? An investigation using farmer interviews, post-mortem examination and molecular tools. Ecological Solutions and Evidence. doi:10.1002/2688-8319.12370.

50. Moberly, R.L., White, P.C.L. & Harris, S. (2004) Mortality due to fox predation in free-range poultry flocks in Britain. Veterinary Record. doi:10.1136/vr.155.2.48.

51. Scottish Parliament Information Centre (2020) Dogs (Protection of Livestock) (Amendment) (Scotland) Bill. SPICe briefing SB 20-57, 11 September 2020. https://www.parliament.scot/chamber-andcommittees/research-prepared-for-parliament/research-briefings/2020/9/11/sb-2057

52. Oxley, J.A., Evans, B. & Montrose, V.T. (2017) Prevention of sheep worrying in the UK: Rethinking the approach. Journal of Veterinary Behavior: Clinical Applications and Research. doi:10.1016/j.jveb.2017.02.001.

53. National Sheep Association (2026) Stronger laws, stronger message: new powers to tackle sheep worrying. March 2026. https://nationalsheep.org.uk/for-the-public/culture/sheep-worrying/stronger-laws-strongermessage-new-powers-to-tackle-sheep-worrying/

54. Welsh Government (2025) Wildlife and Rural Crime Strategy for Wales 2025–28. September 2025. https://www.gov.wales/sites/default/files/publications/2025-09/wildlife-and-rural-crime-strategy-2025- 28\_0.pdf

55. Sarenbo, S. & Doane, M. (2025) Dog attacks on livestock: insights from Swedish news articles and experiences of farmers and inspectors. Frontiers in Veterinary Science. doi:10.3389/fvets.2025.1629966.

56. Montecino-Latorre, D. & San Martín, W. (2019) Evidence supporting that human-subsidized free-ranging dogs are the main cause of animal losses in small-scale farms in Chile. Ambio. doi:10.1007/s13280-018-1066-3.

57. Zambra, N., Piaggio, J. & Ungerfeld, R. (2022) Characteristics of sheep farms and livestock practices that influence sheep predation in Uruguay. Mastozoologia Neotropical. doi:10.31687/saremMN.22.29.1.05.e0569.

58. Ruiz, P., Benitez, G., Bobadilla, P.E., Piedracueva, M. & Damian, J.P. (2024) Do predator attacks on productive species and the respective economic losses influence the psychological distress of farmers in Uruguay? A cross sectional study. Rural and Remote Health. doi:10.22605/RRH7614.

59. Fleming, P.A., Dundas, S.J., Lau, Y.Y.W. & Pluske, J.R. (2016) Predation by Red Foxes (Vulpes vulpes) at an Outdoor Piggery. Animals. doi:10.3390/ani6100060.

60. Roddick, S., Kreplins, T.L., Kobryn, H.T. & Fleming, P.A. (2022) Livestock guardian dog protection of free-range poultry from the red fox. Animal Production Science. doi:10.1071/AN21229.

61. Health and Safety Executive (2019) Cattle and public access in England and Wales. Agriculture Information Sheet AIS17(rev). https://www.hse.gov.uk/pubns/ais17ew.pdf

62. FarmingUK (2023) UK's safety watchdog urges public to stay safe around cattle (report of HSE guidance). 3 April 2023. https://www.farminguk.com/news/uk-s-safety-watchdog-urges-public-to-stay-safe-aroundcattle\_62354.htm

63. Health and Safety Executive (2025) Agriculture, forestry and fishing statistics in Great Britain, 2024/25. 20 November 2025. https://www.hse.gov.uk/statistics/assets/docs/agriculture.pdf

64. Natural England & Defra (undated) The Countryside Code (A5 edition). https://assets.publishing.service.gov.uk/media/646f7507243157000c6f42cd/Countryside\_Code\_A5.pdf

65. Mears, D.P., Scott, M.L. & Bhati, A.S. (2007) A process and outcome evaluation of an agricultural crime prevention initiative. Criminal Justice Policy Review. doi:10.1177/0887403406294586.

66. O'Brien, M. & Windle, J. (2022) The victimisation of farms in Ireland: fear of crime, social isolation and crime prevention. Crime Prevention and Community Safety. doi:10.1057/s41300-022-00153-1.

67. Carmarthen Journal (c. 2021) CCTV the norm on our farms (report of an NFU Cymru rural crime poll). https://www.pressreader.com/uk/carmarthen-journal/20210428/282376927431621

68. Farmers Weekly (2020) Trackers help police recover stolen farm machinery. 29 June 2020. https://www.fwi.co.uk/news/crime/trackers-help-police-recover-stolen-farm-machinery

69. Project Plant (2025) NRCN Action Week highlights success of CESAR and Datatag in fight against rural crime. 2 September 2025. https://projectplant.co.uk/2025/09/nrcn-action-week-highlights-success-of-cesar-anddatatag-in-fight-against-rural-crime/

70. Farmers Weekly (2024) How successful is the National Rural Crime Unit one year on? 25 April 2024. https://www.fwi.co.uk/news/crime/how-successful-is-the-national-rural-crime-unit-one-year-on

71. National Vehicle Crime Intelligence Service (2023) NaVCIS: May 2023 successes. 10 June 2023. https://navcis.police.uk/april23-2/

72. NFU (2025) NFU welcomes crackdown on rural organised crime (NPCC wildlife and rural crime strategy 2025– 2028). 25 November 2025. https://www.nfuonline.com/updates-and-information/npcc-announces-threeyear-strategy/

73. Chowdhury, S. et al. (2025) Enhanced IoT-based smart agro-farm security system through multi-sensor fusion and field validation. Smart Agricultural Technology. doi:10.1016/j.atech.2025.101512.

74. Sayem, N.S. et al. (2023) IoT-based smart protection system to address agro-farm security challenges in Bangladesh. Smart Agricultural Technology. doi:10.1016/j.atech.2023.100358.

75. Ali, M.R., Sayem, N.S., Chowdhury, S. & Kamruzzaman, M. (2024) IoT based Robust Agro-Farm Security System in Bangladesh. 2024 ASABE Annual International Meeting. doi:10.13031/aim.202400198.

76. Sandlana, M.V., Mathonsi, T.E., Du, C. & Du Plessis, D.P. (2022) A Wireless Livestock Tracking System Based on Real-time Internet of Things for Theft Prevention. International Conference on Electrical, Computer, Communications and Mechatronics Engineering, ICECCME 2022. doi:10.1109/ICECCME55909.2022.9988459.

77. Sowmiya, P., Sujitra, R., Srinidhi, M.R., Bhuvana, B.P. & Yokesh, V. (2025) IoT- Based Rural Animal Monitoring and Theft Prevention System. 5th IEEE International Conference on Innovations in Power and Advanced Computing Technologies, i-PACT 2025. doi:10.1109/i-PACT65952.2025.11307970.

78. Obien, J.B., Calinao, V., Bautista, M.G., Dadios, E., Jose, J.A. & Concepcion, R. (2022) AEaaS: Artificial Intelligence Edge-of-Things as a Service for Intelligent Remote Farm Security and Intrusion Detection Pre-

alarm System. 2022 IEEE 14th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management, HNICEM 2022. doi:10.1109/HNICEM57413.2022.10109572.

79. Ehlers, S.G. & Tormoehlen, R.L. (2019) Agricultural Security Monitoring and Safety Alert System: Implementation of Wireless Video on the Farmstead. Journal of Agricultural Safety and Health. doi:10.13031/jash.13004.

80. Endo, H. et al. (2025) Design and training of a sound classification model on a resource-constrained edge device for fruit theft prevention. IECON 2025: 51st Annual Conference of the IEEE Industrial Electronics Society. doi:10.1109/IECON58223.2025.11221896.

81. Amemiya, M., Suzuki, A., Shimada, T. & Kurita, H. (2026) Plural policing in rural contexts: a GPS-based evaluation of multi-entity patrols against fruit theft. Security Journal. doi:10.1057/s41284-026-00543-2.

82. Zantsi, S., Mdiya, L., Ngobeni, L. & Madyo, S. (2026) Awareness and Willingness to Adopt Livestock Tracking Devices to Mitigate Stock Theft Among Eastern Cape Communal Farmers. South African Journal of Agricultural Extension. doi:10.17159/2413-3221/2026/v54n1a20517.

83. Zantsi, S. & Nkunjana, T. (2021) A review of possibilities for using animal tracking devices to mitigate stock theft in smallholder livestock farming systems in rural South Africa. South African Journal of Agricultural Extension. doi:10.17159/2413-3221/2021/v49n1a10784.

84. Clack, W. (2026) Community Policing and Evidence-Based Strategies for Livestock Theft Reduction in South Africa. Social Sciences. doi:10.3390/socsci15020108.

85. Mdoda, L. (2025) Impact of Livestock Theft on the Welfare of Smallholder Livestock Farmers in Nyandeni Local Municipality, Eastern Cape Province. South African Journal of Agricultural Extension. doi:10.17159/2413- 3221/2025/v53n2a15998.

86. Lombard, W.A. & Bahta, Y.T. (2019) Stock theft control mechanism and economic impact of livestock theft in the Free State Province of South Africa: Implication for agricultural management policies. International Journal of Agricultural Management. doi:10.5836/ijam/2019-08-98.

87. Eklund, A., López-Bao, J.V., Tourani, M., Chapron, G. & Frank, J. (2017) Limited evidence on the effectiveness of interventions to reduce livestock predation by large carnivores. Scientific Reports. doi:10.1038/s41598-017- 02323-w.

88. Plotsky, K., Alexander, S.M. & Musiani, M. (2024) Canid livestock predation research has become more robust, but gaps remain. Global Ecology and Conservation. doi:10.1016/j.gecco.2024.e02923.

89. Moreira-Arce, D., Ugarte, C.S., Zorondo-Rodríguez, F. & Simonetti, J.A. (2018) Management Tools to Reduce Carnivore-Livestock Conflicts: Current Gap and Future Challenges. Rangeland Ecology & Management. doi:10.1016/j.rama.2018.02.005.

90. Khorozyan, I. & Waltert, M. (2019) How long do anti-predator interventions remain effective? Patterns, thresholds and uncertainty. Royal Society Open Science. doi:10.1098/rsos.190826.

91. Pépin, D., Feuvrier, P., Powolny, T. & Giraudoux, P. (2025) Investigating the effects of red fox management on poultry beyond the controversy, Jura Massif, France. Scientific Reports. doi:10.1038/s41598-025-08500-6.

92. Dawson, S.J., Kreplins, T.L., Kennedy, M.S., Wicks, S. & Fleming, P.A. (2025) An economic analysis of cellfencing in semi-arid rangelands. Rangeland Journal. doi:10.1071/RJ24023.

93. Hall, K.J. & Fleming, P.A. (2021) In the spotlight: can lights be used to mitigate fox predation on a free-range piggery? Applied Animal Behaviour Science. doi:10.1016/j.applanim.2021.105420.

94. Breck, S.W., Schultz, J.T., Prause, D., Krebs, C., Giordano, A.J. & Boots, B. (2023) Integrating robotics into wildlife conservation: testing improvements to predator deterrents through movement. PeerJ. doi:10.7717/peerj.15491.

95. Bott, C.A. et al. (2024) FlashTags: evaluating a novel deterrent to reduce livestock depredation. Human-Wildlife Interactions.

96. Greentree, C., Saunders, G., McLeod, L. & Hone, J. (2000) Lamb predation and fox control in south-eastern Australia. Journal of Applied Ecology. doi:10.1046/j.1365-2664.2000.00530.x.

97. Allen, L.R. (2014) Wild dog control impacts on calf wastage in extensive beef cattle enterprises. Animal Production Science. doi:10.1071/AN12356.

98. Moberly, R.L., White, P.C.L., Webbon, C.C., Baker, P.J. & Harris, S. (2004) Modelling the costs of fox predation and preventive measures on sheep farms in Britain. Journal of Environmental Management. doi:10.1016/j.jenvman.2003.11.005.

99. Heydon, M.J. & Reynolds, J.C. (2000) Demography of rural foxes (Vulpes vulpes) in relation to cull intensity in three contrasting regions of Britain. Journal of Zoology. doi:10.1017/S0952836900006117.

100. Saitone, T.L. & Bruno, E.M. (2020) Cost Effectiveness of Livestock Guardian Dogs for Predator Control. Wildlife Society Bulletin. doi:10.1002/wsb.1063.

101. van Bommel, L. & Johnson, C.N. (2024) Still a good dog! Long-term use and effectiveness of livestock guardian dogs to protect livestock from predators in Australia's extensive grazing systems. Wildlife Research. doi:10.1071/WR23008.

102. Larson, S., McGranahan, D.A. & Timm, R.M. (2019) The Marin County Livestock Protection Program: 15 years in review. Human-Wildlife Interactions.

103. McLeod, L.J., Saunders, G.R., McLeod, S.R., Dawson, M. & van de Ven, R. (2010) The potential for participatory landscape management to reduce the impact of the red fox (Vulpes vulpes) on lamb production. Wildlife Research. doi:10.1071/WR10082.

104. Saunders, G.R., Gentle, M.N. & Dickman, C.R. (2010) The impacts and management of foxes Vulpes vulpes in Australia. Mammal Review. doi:10.1111/j.1365-2907.2010.00159.x.

105. Scasta, J.D., Stam, B. & Windh, J.L. (2017) Rancher-reported efficacy of lethal and non-lethal livestock predation mitigation strategies for a suite of carnivores. Scientific Reports. doi:10.1038/s41598-017-14462-1.

106. Schofield, W. (2021) Wild dog/dingo conservation, control and management: The impact beyond the paddock. Australian Zoologist. doi:10.7882/AZ.2020.037.

107. van Bommel, L. & Johnson, C. (2017) Predation control. Advances in Sheep Welfare. doi:10.1016/B978-0-08- 100718-1.00009-1.

108. Fennell, M., Beirne, C. & Burton, A.C. (2022) Use of object detection in camera trap image identification: Assessing a method to rapidly and accurately classify human and animal detections for research and application in recreation ecology. Global Ecology and Conservation. doi:10.1016/j.gecco.2022.e02104.

109. Meng, D.Y. et al. (2023) A method for automatic identification and separation of wildlife images using ensemble learning. Ecological Informatics. doi:10.1016/j.ecoinf.2023.102262.

110. Beery, S., Wu, G.H., Rathod, V., Votel, R. & Huang, J. (2020) Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/CVPR42600.2020.01309.

111. Korkmaz, A., Agdas, M.T., Kosunalp, S., Iliev, T. & Stoyanov, I. (2024) Detection of threats to farm animals using deep learning models: a comparative study. Applied Sciences. doi:10.3390/app14146098.

112. Lopez, A., Jimenez, E., Valles, D. & Drewery, M.L. (2026) Developing Intelligent Models to Detect and Classify Cattle Behavior on Pasture. Animals. doi:10.3390/ani16162617.

113. Ojija, F. et al. (2025) Artificial intelligence-driven solutions for mitigating human-wildlife conflict in biodiversity hotspots. Science Progress. doi:10.1177/00368504251394584.

114. Westworth, S.O.A., Chalmers, C., Fergus, P., Longmore, S.N., Piel, A.K. & Wich, S.A. (2022) Understanding External Influences on Target Detection and Classification Using Camera Trap Images and Machine Learning. Sensors. doi:10.3390/s22145386.

115. Thampan, A., Muhammed Nahas, N., Nazarudeen, R., Suresh, T. & Varun Chand, H. (2026) Smart Farm Protection Systems based on AI: A Survey of Wildlife and Bird Intrusion Detection and Deterrence Techniques. Proceedings of the 5th International Conference on Sentiment Analysis and Deep Learning, ICSADL 2026. doi:10.1109/ICSADL67539.2026.11451990.

116. Roilo, S., Hofmeester, T.R., Frauendorf, M., Widén, A. & Cord, A.F. (2025) The untapped potential of camera traps for farmland biodiversity monitoring: current practice and outstanding agroecological questions. Remote Sensing in Ecology and Conservation. doi:10.1002/rse2.426.

117. Dede, J. & Förster, A. (2024) Demo: From Pasture to Cloud: AI-Driven Predator Deterrence with Local Action and Cloud-Linked Alerts. International Conference on Embedded Wireless Systems and Networks.

118. Kane, C.A. & Faye, P.F. (2024) Combating Agricultural Challenges with Secure Digital Farming. Proceedings of the International Conference on Informatics in Control, Automation and Robotics. doi:10.5220/0013011800003822.

119. Gokila, E., Muzammil Ali, A., Srinivasan, M.K., Ritesh, V. & Ansilin Kumar, S.M. (2025) A Comparative Study of VGG19, Dense Net and Inception V3 for Wildlife and Human Threat Detection. 2025 International Conference on Data Science and Business Systems, ICDSBS 2025. doi:10.1109/ICDSBS63635.2025.11031571.

120. Jones and Phipps (2012) Policing farm crime in England and Wales. Papers from the British Criminology Conference 12, 3–24. https://research.aber.ac.uk/en/publications/policing-farm-crime-in-england-andwales/

121. Jones (2011) Recording and responding to farm crime: a national study of police forces in England and Wales. Unpublished report, Aberystwyth University. Cited second-hand through Jones and Phipps (2012); not retrieved.

122. The Scottish Farmer (undated) SPARC: Tom French key insights on Scottish rural crime. https://thescottishfarmer.co.uk/features/24550999.sparcs-tom-french-key-insights-scottish-rural-crime

123. Hughes, N. & Bowden, M. (2025) Farm crime and trust in the Police in Ireland. Crime Prevention and Community Safety. doi:10.1057/s41300-025-00227-w.

124. Ceccato, V. & Harkness, A. (2025) Research design for assessing victimisation, safety and policing in rural areas. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-4.

125. FarmingUK (2018) 69% of farmers victim of rural crime in last 12 months, UK-wide survey shows (report of the 2018 National Rural Crime Survey). 23 July 2018. https://www.farminguk.com/news/69-of-farmers-victimof-rural-crime-in-last-12-months-uk-wide-survey-shows\_49850.html

126. Clack, W. (2025) Livestock theft dynamics in the Free State: analysing perceptions, trends and economic implications (2019-2024). South African Crime Quarterly. doi:10.17159/sacq.n74.20909.

127. Barclay, E., Donnermeyer, J.F. & Jobes, P.C. (2004) The dark side of gemeinschaft: Criminality within rural communities. Crime Prevention and Community Safety. doi:10.1057/palgrave.cpcs.8140191.

128. Norris, G. & Morris, W. (2025) Crime and safety in rural Wales. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-24.

129. Peterson, J.R., Ward, K.C. & Lawrie, M. (2024) Rural American farmers' perceptions of crime, safety and policing. Policing: An International Journal of Police Strategies & Management. doi:10.1108/PIJPSM-03- 2024-0030.

130. Eman, K. & Bulovec, T. (2021) A case study of rural crime and policing in Pomurje region in Slovenia. Journal of Rural Studies. doi:10.1016/j.jrurstud.2021.05.012.

131. Clack, W. & Fitz, L. (2026) Livestock theft in the media: quantitative reporting trends across South African provinces (April 2018 to March 2025). International Journal for Crime Justice and Social Democracy. doi:10.5204/ijcjsd.4238.

132. Ceccato, V. & Harkness, A. (2025) Victimisation, safety and policing in rural areas: evidence from the literature. Crime, Peoples and Places: Perspectives on Rural Safety and Justice. doi:10.4324/9781003395522-3.

133. Nyczaj Kyle, S., Barkus, E., Dunne, S. & Jones, A. (2025) A comparison of farm stressors in the UK and Canada. Round Table. doi:10.1080/00358533.2025.2498161.

134. Smith, K. (2026) Agricultural Crime and Farmer Mental Health: Balancing Positive and Negative Affects. Routledge International Handbook of Criminology and Affect. doi:10.4324/9781003543213-14.

135. Ecker, S., Please, P.M. & Maybery, D.J. (2017) Constantly chasing dogs: assessing landholder stress from wild dog attacks on livestock using quantitative and qualitative methods. Australasian Journal of Environmental Management. doi:10.1080/14486563.2016.1251346.

## Appendix A. Full search strings

Searches were run on 4 November 2026 and pasted exactly as run. Web of Science limits were applied in the interface as noted (Article or Review Article; English); Scopus limits are in the string.

<table><tr><td rowspan=1 colspan=1>Query</td><td rowspan=1 colspan=1>Database</td><td rowspan=1 colspan=1>String</td></tr><tr><td rowspan=1 colspan=1>Q1</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>TS=((&quot;livestock theft&quot; OR &quot;sheep theft&quot; OR &quot;stock theft&quot; OR &quot;livestockrustling&quot; OR &quot;farm crime&quot; OR &quot;agricultural crime&quot; OR &quot;rural crime&quot; OR&quot;farm theft&quot; OR &quot;farm machinery theft&quot; OR &quot;agricultural machinery theft&quot;OR &quot;tractor theft&quot; OR &quot;quad bike theft&quot; OR &quot;farm equipment theft&quot;) AND(cost* OR impact* OR victim* OR loss* OR prevalence OR incidence ORburden)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q2</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>(TS=((&quot;livestock worrying&quot; OR &quot;sheep worrying&quot; OR &quot;dog attack*&quot; OR &quot;dogpredation&quot; OR &quot;livestock predation&quot; OR &quot;lamb predation&quot; OR &quot;predation oflambs&quot; OR &quot;fox predation&quot; OR &quot;predator attack*&quot;) AND (livestock OR sheepOR lamb* OR cattle OR poultry OR farm*)) NOT TI=(wolf OR wolves OR lynx ORbear OR bears OR leopard* OR lion* OR tiger* OR jaguar OR cougar ORhyena*)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q3</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>TS=((&quot;gates left open&quot; OR &quot;gate left open&quot; OR &quot;livestock escape*&quot; OR&quot;livestock straying&quot; OR &quot;stray livestock&quot; OR &quot;livestock-vehiclecollision*&quot; OR &quot;dog walker*&quot; OR &quot;public access&quot; OR &quot;recreational access&quot;OR &quot;right of way&quot; OR &quot;rights of way&quot;) AND (livestock OR sheep OR cattle)AND (farm* OR farmer* OR agricultur*)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q4</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>TS=((&quot;farm security&quot; OR &quot;farm crime prevention&quot; OR &quot;rural crimeprevention&quot; OR &quot;situational crime prevention&quot; OR &quot;crime prevention&quot; OR&quot;theft prevention&quot;) AND (farm* OR agricultur* OR rural OR livestock) AND(CCTV OR camera* OR surveillance OR tracking OR &quot;DNA marking&quot; OR &quot;forensicmarking&quot; OR ANPR OR &quot;number plate recognition&quot; OR alarm* OR &quot;securitysystem*&quot; OR &quot;computer vision&quot; OR &quot;deep learning&quot;)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q5</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>TS=((&quot;camera trap*&quot; OR &quot;trail camera*&quot; OR &quot;wildlife camera*&quot; OR &quot;remotecamera*&quot;) AND (&quot;deep learning&quot; OR &quot;computer vision&quot; OR &quot;machine learning&quot;OR &quot;object detection&quot; OR &quot;convolutional neural network*&quot;) AND (predator*OR livestock OR intruder* OR farm* OR &quot;human detection&quot; OR &quot;persondetection&quot; OR &quot;vehicle detection&quot;)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q6</td><td rowspan=1 colspan=1>Web of Science CoreCollection (Editions:All); run 2026-11-04</td><td rowspan=1 colspan=1>TS=((&quot;rural crime&quot; OR &quot;farm crime&quot; OR &quot;agricultural crime&quot; OR &quot;livestocktheft&quot;) AND (underreport* OR &quot;under-reporting&quot; OR &quot;under reporting&quot; ORunreported OR &quot;police recorded&quot; OR &quot;recorded crime&quot; OR &quot;crime survey&quot; OR&quot;dark figure&quot; OR &quot;data quality&quot;)) AND PY=(2000-2026)</td></tr><tr><td rowspan=1 colspan=1>Q1</td><td rowspan=1 colspan=1>Scopus advancedsearch; run 2026-11-04</td><td rowspan=1 colspan=1>TITLE-ABS-KEY((&quot;livestock theft&quot; OR &quot;sheep theft&quot; OR &quot;stock theft&quot; OR&quot;livestock rustling&quot; OR &quot;farm crime&quot; OR &quot;agricultural crime&quot; OR &quot;ruralcrime&quot; OR &quot;farm theft&quot; OR &quot;farm machinery theft&quot; OR &quot;agriculturalmachinery theft&quot; OR &quot;tractor theft&quot; OR &quot;quad bike theft&quot; OR &quot;farmequipment theft&quot;) AND (cost* OR impact* OR victim* OR loss* OR prevalenceOR incidence OR burden)) AND PUBYEAR &gt; 1999 AND PUBYEAR &lt; 2027 ANDLANGUAGE(english) AND DOCTYPE(ar OR re OR ch)</td></tr><tr><td rowspan=1 colspan=1>Q2</td><td rowspan=1 colspan=1>Scopus advancedsearch; run 2026-11-04</td><td rowspan=1 colspan=1>TITLE-ABS-KEY((&quot;livestock worrying&quot; OR &quot;sheep worrying&quot; OR &quot;dog attack*&quot;OR &quot;dog predation&quot; OR &quot;livestock predation&quot; OR &quot;lamb predation&quot; OR&quot;predation of lambs&quot; OR &quot;fox predation&quot; OR &quot;predator attack*&quot;) AND(livestock OR sheep OR lamb* OR cattle OR poultry OR farm*)) AND PUBYEAR &gt;1999 AND PUBYEAR &lt; 2027 AND LANGUAGE(english) AND DOCTYPE(ar OR re OR ch)AND NOT TITLE(wolf OR wolves OR lynx OR bear OR bears OR leopard* OR lion*OR tiger* OR jaguar OR cougar OR hyena*)</td></tr><tr><td rowspan=1 colspan=1>Q3</td><td rowspan=1 colspan=1>Scopus advancedsearch; run 2026-11-04</td><td rowspan=1 colspan=1>TITLE-ABS-KEY((&quot;gates left open&quot; OR &quot;gate left open&quot; OR &quot;livestockescape*&quot; OR &quot;livestock straying&quot; OR &quot;stray livestock&quot; OR &quot;livestock-vehicle collision*&quot; OR &quot;dog walker*&quot; OR &quot;public access&quot; OR &quot;recreationalaccess&quot; OR &quot;right of way&quot; OR &quot;rights of way&quot;) AND (livestock OR sheep ORcattle) AND (farm* OR farmer* OR agricultur*)) AND PUBYEAR &gt; 1999 ANDPUBYEAR &lt; 2027 AND LANGUAGE(english) AND DOCTYPE(ar OR re OR ch)</td></tr><tr><td rowspan=1 colspan=1>Q4</td><td rowspan=1 colspan=1>Scopus advancedsearch; run 2026-11-04</td><td rowspan=1 colspan=1>TITLE-ABS-KEY((&quot;farm security&quot; OR &quot;farm crime prevention&quot; OR &quot;rural crimeprevention&quot; OR &quot;situational crime prevention&quot; OR &quot;crime prevention&quot; OR&quot;theft prevention&quot;) AND (farm* OR agricultur* OR rural OR livestock) AND(CCTV OR camera* OR surveillance OR tracking OR &quot;DNA marking&quot; OR &quot;forensicmarking&quot; OR ANPR OR &quot;number plate recognition&quot; OR alarm* OR &quot;securitysystem*&quot; OR &quot;computer vision&quot; OR &quot;deep learning&quot;)) AND PUBYEAR &gt; 1999 ANDPUBYEAR &lt; 2027 AND LANGUAGE(english) AND DOCTYPE(ar OR re OR cp)</td></tr></table>

<table><tr><td>Q5 04</td><td>Scopus advanced search; run 2026-11-</td><td>TITLE-ABS-KEY((&quot;camera trap*&quot; OR &quot;trail camera*&quot; OR &quot;wildlife camera*&quot; OR &quot;remote camera*&quot;) AND (&quot;deep learning&quot; OR &quot;computer vision&quot; OR &quot;machine learning&quot; OR &quot;object detection&quot; OR &quot;convolutional neural network*&quot;) AND (predator* OR livestock OR intruder* OR farm* OR &quot;human detection&quot; OR</td><td></td><td>&quot;person detection&quot; OR &quot;vehicle detection&quot;)) AND PUBYEAR &gt; 1999 AND PUBYEAR</td><td></td></tr><tr><td>Q6</td><td>Scopus advanced search; run 2026-11- 04</td><td>&lt; 2027 AND LANGUAGE(english) AND DOCTYPE(ar OR re OR cp) TITLE-ABS-KEY((&quot;rural crime&quot; OR &quot;farm crime&quot; OR &quot;agricultural crime&quot; OR &quot;livestock theft&quot;) AND (underreport* OR &quot;under-reporting&quot; OR &quot;under reporting&quot; OR unreported OR &quot;police recorded&quot; OR &quot;recorded crime&quot; OR</td><td></td><td></td><td></td></tr></table>

## Appendix B. PRISMA 2020 checklist: where each item is reported

<table><tr><td colspan="1" rowspan="1">Item</td><td colspan="1" rowspan="1">Topic</td><td colspan="1" rowspan="1">Requirement (abridged)</td><td colspan="1" rowspan="1">Where reported</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Title</td><td colspan="1" rowspan="1">Identify the report as a systematic review</td><td colspan="1" rowspan="1">Title and abstract; presented as a PRISMA-guided review</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">Abstract</td><td colspan="1" rowspan="1">See PRISMA 2020 for Abstracts</td><td colspan="1" rowspan="1">Abstract (unstructured)</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">Rationale</td><td colspan="1" rowspan="1">Describe the rationale</td><td colspan="1" rowspan="1">Section 1</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">Objectives</td><td colspan="1" rowspan="1">State the objectives or questions</td><td colspan="1" rowspan="1">Section 1 (RQ1-5)</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">Eligibility criteria</td><td colspan="1" rowspan="1">Specify inclusion and exclusion criteria</td><td colspan="1" rowspan="1">Section 2.2</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">Information sources</td><td colspan="1" rowspan="1">Specify databases and other sources, withdates</td><td colspan="1" rowspan="1">Section 2.3; Table 1</td></tr><tr><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">Search strategy</td><td colspan="1" rowspan="1">Present full search strategies</td><td colspan="1" rowspan="1">Appendix A</td></tr><tr><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">Selection process</td><td colspan="1" rowspan="1">Methods used to decide eligibility</td><td colspan="1" rowspan="1">Section 2.4 (single AI-assisted reviewer;human spot-check pending)</td></tr><tr><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">Data collectionprocess</td><td colspan="1" rowspan="1">Methods used to collect data</td><td colspan="1" rowspan="1">Section 2.5</td></tr><tr><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">Data items</td><td colspan="1" rowspan="1">List and define outcomes and othervariables</td><td colspan="1" rowspan="1">Section 2.5</td></tr><tr><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">Study risk of biasassessment</td><td colspan="1" rowspan="1">Methods used to assess risk of bias</td><td colspan="1" rowspan="1">Not performed (Sections 2.6, 5)</td></tr><tr><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">Effect measures</td><td colspan="1" rowspan="1">Specify effect measures</td><td colspan="1" rowspan="1">Not applicable; effects as reported bysources</td></tr><tr><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">Synthesis methods</td><td colspan="1" rowspan="1">Methods to tabulate and synthesise</td><td colspan="1" rowspan="1">Section 2.6; Fig. 3</td></tr><tr><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">Reporting biasassessment</td><td colspan="1" rowspan="1">Methods to assess reporting bias</td><td colspan="1" rowspan="1">Not formal; Section 5</td></tr><tr><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">Certainty assessment</td><td colspan="1" rowspan="1">Methods to assess certainty</td><td colspan="1" rowspan="1">Not performed; described in words</td></tr><tr><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">Study selection</td><td colspan="1" rowspan="1">Results of search and selection, with flowdiagram</td><td colspan="1" rowspan="1">Section 3.1; Fig. 1</td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">Study characteristics</td><td colspan="1" rowspan="1">Cite each study and present itscharacteristics</td><td colspan="1" rowspan="1">Section 3.1; Figs. 2–3; supplementaryworkbook</td></tr><tr><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">Risk of bias in studies</td><td colspan="1" rowspan="1">Present assessments of risk of bias</td><td colspan="1" rowspan="1">Not performed</td></tr><tr><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">Results of individualstudies</td><td colspan="1" rowspan="1">Present results for each study</td><td colspan="1" rowspan="1">Sections 3.2–3.8; supplementary workbook</td></tr><tr><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">Results of syntheses</td><td colspan="1" rowspan="1">Present results of all syntheses</td><td colspan="1" rowspan="1">Sections 3.2–3.9; Tables 2–4</td></tr><tr><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">Reporting biases</td><td colspan="1" rowspan="1">Present assessments of reporting bias</td><td colspan="1" rowspan="1">Section 5 (qualitative)</td></tr><tr><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">Certainty of evidence</td><td colspan="1" rowspan="1">Present certainty assessments</td><td colspan="1" rowspan="1">Not performed</td></tr><tr><td colspan="1" rowspan="1">23</td><td colspan="1" rowspan="1">Discussion</td><td colspan="1" rowspan="1">Interpretation, limitations, implications</td><td colspan="1" rowspan="1">Sections 4-5</td></tr><tr><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">Registration andprotocol</td><td colspan="1" rowspan="1">Registration information</td><td colspan="1" rowspan="1">Not registered (Section 2.1)</td></tr><tr><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">Support</td><td colspan="1" rowspan="1">Sources of support</td><td colspan="1" rowspan="1">Declarations (to be completed)</td></tr><tr><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">Competing interests</td><td colspan="1" rowspan="1">Declare competing interests</td><td colspan="1" rowspan="1">Not reported</td></tr><tr><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">Availability of data,code and materials</td><td colspan="1" rowspan="1">Report what is publicly available</td><td colspan="1" rowspan="1">Declarations; Appendix A</td></tr></table>