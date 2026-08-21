# When Saying No Makes Beter Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation

YEARIM KIM<sup>∗</sup>, Seoul National University, Republic of Korea

INJUN BAEK<sup>∗</sup>, Seoul National University, Samsung Electronics, Republic of Korea

NOJUN KWAK<sup>†</sup>, Seoul National University, Republic of Korea

To prevent the adoption of aesthetically polished but pedagogically flawed AI content, we study a video authoring pipeline featuring two layers of structured refusal. The first layer empowers educators to iteratively reshape AI scripts based on multimedia learning theory, while the second employs automated metrics to flag violations in instructional coherence and narrative-visual synchronization. While neither layer is exhaustive, their synergy ensures that principled resistance–the act of deferring AI output until it meets rigorous standards–becomes a catalyst for higher quality. Evaluation combining a study with 23 educators across 3 topics and automated metrics across 7 topics drawn from established science and philosophy curricula shows that both layers independently improve the same instructional dimensions, suggesting that thoughtful resistance and generative AI are not opposites but partners.

CCS Concepts: • Human-centered computing → User studies; • Applied computing → Interactive learning environments.

Additional Key Words and Phrases: Human-AI Collaboration, Educational AI, Generative AI, Multimedia Learning

## ACM Reference Format:

Yearim Kim, Injun Baek, and Nojun Kwak. 2026. When Saying No Makes Better Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation. In Proceedings ofUnderstanding and Engaging Critical Resistance to AI in Education (CHI ’26 Workshop). ACM, New York, NY, USA, 5 pages

## 1 Introduction

While modern AI models [1, 2] synthesizes professional-looking educational videos in a minute, surface-level polish does not guarantee pedagogical rigor. Current video generation pipelines often prioritize visual appeal over instructional essentials, such as precise temporal alignment ofnarration or strategic sequencing ofprerequisite concepts. Consequently, outputs are frequently optimized for looking good, rather than teaching well. This gap matters as educators are increasingly expected to adopt AI-generated materials with minimal intervention. The pressure toward seamless, friction-free adoption treats any slowdown as ineficiency—a stance that risks reducing the educator’s role from professional decision-maker to passive consumer. We argue, however, that pedagogical friction is not a hurdle to be eliminated but a site of professional accountability. Moments of deliberate hesitation–whether an educator questioning a script’s logical flow or an algorithm flagging a narrative truncation–are precisely where instructional quality is forged.

Table 1. Mayer’s 12 CTML principles [3] for reducing extraneous, managing essential, and fostering generative processing.
<table><tr><td>Coherence</td><td>Signaling</td><td>Redundancy</td><td>Spatial Contiguity</td><td>Temporal Contiguity</td><td>Segmenting</td></tr><tr><td>Pre-training</td><td>Modality</td><td>Multimedia</td><td>Personalization</td><td>Voice</td><td>Image</td></tr></table>

We conceptualize this approach as principled resistance: deliberate, theory-grounded pushback against AI outputs that fail pedagogical standards. Rooted in Mayer’s Cognitive Theory of Multimedia Learning (CTML) [3]–a framework of 12 empirically validated principles for efective multimedia instruction (Table 1)–we developed the PedaCo (Pedagogical Cocreation), a human-AI collaborative system that operationalizes this resistance. PedaCo integrates two complementary gatekeeping mechanisms: reviewing scripts through CTML-informed criteria, and an automated metric evaluating finished videos against computationally measurable CTML dimensions. This paper presents the design rationale behind this dual approach, summarizes converging evidence from both human and computational evaluations, and poses open questions about how educational AI systems should balance human agency with automated safeguards.

## 2 Two Layers of Resistance

In our framework, principled resistance takes three concrete forms: rejecting (requesting regeneration), revising (manual editing), and overriding (vetoing automated flags). These are not ad hoc reactions but norm-driven decisions grounded in CTML principles. The framework rests on a simple observation: educators and algorithms are good at catching diferent kinds of problems: while human educators excel at identifying nuanced pedagogical mismatches–such as content being too advanced for a target audience–algorithms provide precise, high-resolution verification of structura integrity, such as identifying temporal misalignments between narration and visuals. Building both checkpoints into the same pipeline creates overlapping coverage that neither could achieve alone.

## 2.1 Layer 1: Human Educator Review at the Script Stage

The first layer intervenes before any video is rendered (Figure 1 left). The educator begins by inputting learning content and configuring which CTML principles the system should enforce. An LLM [4] then generates an initial script, which passes through a structured review cycle. An AI reviewer—itself prompted with CTML principles—produces feedback organized by principle, identifying potential violations rather than definitive judgments (e.g., “Scene 3 introduces technical terms without prior explanation, which may conflict with the Pre-training principle”). The human educator then decides what to accept, what to revise manually, and what to regenerate. This review loop can be repeated unti the educator is satisfied with the script.

This design choice to review at the script level is deliberate. Textual revisions are computationally and laboriously eficient, whereas pedagogical errors baked into a rendered visual narrative are nearly impossible to correct postsynthesis. By placing the human checkpoint at this intermediate stage, we make pedagogical critique economically viable. This ensures the system remains advisory, preserving the educator’s professional authority to say ‘no’ to AI suggestions based on their specific curricular context and pedagogical style.

## 2.2 Layer 2: Automated Metric After Video Synthesis

The second layer performs a post-synthesis evaluation of the finalized video through a composite metric (Figure 1 right). We automate the assessment of five dimensions: coherence, redundancy, temporal contiguity, modality, and image quality. The educator reviews the principle-level scores and decides whether to accept the video or return to the script stage for targeted revision. This boundary between human and automated resistance is a strategic design decision While dimensions like temporal synchronization are amenable to reliable computational measurement, others—such as Manuscript submitted to ACM

When Saying No Makes Better Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation3

![](images/ba85c5000d1484a98f59ecf3de66b6a2c8eba4e50cba83f8b396cf47eff7ad09.jpg)  
Fig. 1. The Dual Gatekeeping Interface. The left panel (Layer 1: Script Level) generates an initial script (�) from learning content (�) and generation principles (�), then scafolds the educator’s revision by providing AI critiques and revised drafts (�) based on review constraints (�). The right panel (Layer 2: Video Level) visualizes invisible pedagogical quality via automated metrics (ℎ, �), allowing users to assess the alignment between the final video (�) and the learning content (�).

Personalization (tone)—require a deep understanding of curricular structures and learner psychology that currently remains the sole domain of the human expert. By automating only where algorithmic feasibility aligns with pedagogical necessity, we create a robust safety net that prevents technical regressions without marginalizing human judgment.

## 3 Evidence from Two Evaluations

We conducted a multi-method evaluation to assess the eficacy of the PedaCo framework, combining a human-centric study with educators (3.1) and an algorithmic assessment via automated metrics (3.2). Together, these evaluations provide converging evidence for the value of dual-layered resistance.

## 3.1 What Educators Found

We conducted a within-subject study with 23 educators—who were priorly briefed on the CTML principles—directly used our system to generate and evaluate videos. Guided by the system, participants simulated with the pre-generated videos by inputting raw learning content and iteratively refining the AI-generated scripts with the system’s CTML feedback. Then, they compared these videos outcomes with baseline videos generated without CTML guidelines. The topics include three diferent cognitive demands: causal reasoning, abstract concepts, and procedural knowledge. Participants rated each condition on 13 items covering all 12 CTML principles and overall instructional validity.

The review-based approach yielded statistically significant improvements across every principle (� < .05, Wilcoxon signed-rank). The mean rating rose from 3.07 to 3.86 on a 5-point scale (+0.79, � < .01), with the most pronounced gains observed in content organization: prerequisite sequencing (+0.86), irrelevant material removal (+0.84), and overall instructional validity (+0.96)ove. Nearly all efects were large (� ≥ .64, computed as �/ �), with one exception (redundancy, � = .42, medium), and consistent across gender and experience level.

Notably, educators did not perceive the review process as slowing them down. They rated production eficiency at 4.26/5 and the validity of the CTML-based guidance at 4.04/5 (with remarkably low variance, �� = 0.62). One participant captured the tension well: the iterative process was “quite challenging” but ultimately for producing “a robust and efective learning tool” (P23). Another explicitly requested “separate functions where teachers can additionally review, edit, and modify” the AI output (P05)—in other words, more resistance, not less.

## 3.2 What the Metrics Found

Independently, we applied our automated metrics to a corpus of 14 videos (7 topics × 2 conditions) drawn from established science and philosophy curricula. The videos were generated with identical structure, isolating the CTMLinformed generation and review as the sole variable. Two of the five metrics showed significant improvement: tempora contiguity (0.294 vs. 0.273, � = .021) and coherence (0.729 vs. 0.646, � = .011). The remaining three showed no significant diference—modality and redundancy scored high in both conditions (near ceiling), and image quality did not difer as both conditions used the same video synthesis model. We retain these metrics as safety nets: current models perform well on these dimensions, but they serve as guardrails against future model regressions or hallucination-induced failures.

## 3.3 Where the Two Evaluations Agree

The most compelling evidence for our dual-layered approach is the high degree of convergence between subjective ratings and objective metrics. Despite being conducted independently with diferent samples and instruments, both evaluations identified coherence and temporal alignment as the dimensions most significantly enhanced by the PedaCo pipeline. Educators ranked coherence among the top three improvements, matching the statistically significant gains identified by the automated system. This triangulation suggests that the two gatekeeping layers are not merely redundant but provide complementary verification of the same underlying instructional quality.

## 4 Discussion: Reframing Resistance in ducational AI

The PedaCo framework ofers a theory-grounded perspective on where the boundaries of Generative AI should be drawn in educational settings. By operationalizing "principled resistance" through CTML, we illustrate how intentional friction can sustain pedagogical integrity without marginalizing the professional authority of educators. Our findings suggest that "productive friction" is most efective when it is: (a) theoretically grounded rather than intuitive; (b) strategically embedded upstream at the script stage; and (c) hybridized across human and computational agents

However, this dual-gatekeeping approach surfaces three emergent tensions for the workshop to consider:

• Negotiating Agency: When automated flags and educator judgments diverge, how should the interface balance algorithmic safeguards with human autonomy?

• Sustainability of Friction: While valued in our short-term study, the long-term viability of iterative review in daily classroom preparation remains unknown. We must identify the threshold where productive friction transitions into "friction fatigue."

• Beyond Proxy Metrics: Future research must move beyond theoretical proxies to evaluate the direct causal impact of structured resistance on student learning outcomes.

## 5 Conclusion

In this paper, we have argued that educational AI needs structured ways to say “not yet”—bridging human pedagogical expertise with computational precision. Our PedaCo framework builds principled resistance into the video generation pipeline through educator review at the script stage and automated pedagogical metrics at the video stage. Early evidence suggests both layers improve the same instructional dimensions, and that educators experience this friction as productive rather than burdensome. We ofer this as one concrete answer to the workshop’s animating question: resistance to AI in education should not be equated with rejection. Instead, it can mean building systems that are designed to push back, on principled grounds, until the output is genuinely ready to teach. Manuscript submitted to ACM

When Saying No Makes Better Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation5

## Acknowledgments

This work was funded by the Korean Government through the grants from IITP (RS-2021-II211343) and KOCCA (RS-2024-00398320).

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https://openai.com/research/video-generation-models-as world-simulator

[2] Google DeepMind. 2024. Veo: Google’s most capable generative video model. https://deepmind.google/models/veo/. Accessed: 2025-09-29

[3] Richard E. Mayer. 2009. Multimedia Principle. Cambridge University Press, 223–241.

[4] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)