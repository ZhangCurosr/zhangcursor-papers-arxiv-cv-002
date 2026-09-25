# When Can Agents Forget Their Reasoning? ICLR for Long-Horizon Agent Context Compression

Mingxuan Wang<sup>1</sup> Fei Luo<sup>1</sup> Bo Wang<sup>1</sup> Guorun Yao<sup>1</sup> Yinglong Guo<sup>1</sup>

Chao Ning<sup>1</sup> Hongyue Chen<sup>1</sup> Yanbiao Ma<sup>2,\*</sup> Jungong Han<sup>3,\*</sup>

<sup>1</sup>TierFlow Team

<sup>2</sup>Gaoling School of Artificial Intelligence, Renmin University of China

<sup>3</sup>Tsinghua University

Corresponding authors. ybma1998@ruc.edu.cn

## ABSTRACT

Long horizon language model agents continually accumulate reasoning history, increasing context length and inference cost even after earlier decisions have been executed and observed. Unlike static Chain of Thought compression, removing historical reasoning can change future actions and the resulting interaction trajectory. We study when such reasoning can be safely forgotten. We propose Interaction Aware Compression for Long Horizon Reasoning (ICLR), a training free online method that ranks reasoning blocks using frozen proxy entropy while preserving actions, tool calls, and observations. On 260 WorkBuddyBench tasks, ICLR improves average reward from 0.699 to 0.718, while reducing input, output, and cache read tokens by 25.5%, 14.4%, and 33.3%, respectively. Ablations reveal trajectory amplification, where local reasoning deletion produces nonlinear changes in total computation by altering subsequent interaction. Representation probing, activation patching, and controlled trajectory analyses further suggest that historical reasoning becomes more replaceable once task relevant derived state has been reliably externalized into code, files, tool outputs, or environmental feedback. These results characterize agent reasoning as dynamic working state rather than permanent interaction history.

## 1 Introduction

Large language models are increasingly used in complex agent tasks, where reasoning, action, and environmental feedback form a repeated interaction loop<sup>1,2,3</sup>. As a task progresses, earlier reasoning is repeatedly included in later requests, increasing context length and inference cost<sup>4,5,6</sup>. Meanwhile, new observations continuously change the task state. This raises a central question for agent context management: once reasoning has produced an action and the environment has returned feedback, does that reasoning still need to remain in the active context?

Figure 1 motivates this question from two observations. The most effective reasoning setting varies with task difficulty, and more reasoning does not consistently yield higher scores. Our online interventions also show a nonmonotonic relation among reasoning removal, reward, and input reduction. Together, these observations suggest that the value of reasoning depends on the current state rather than on reasoning quantity alone.

![](images/e52c0be35c6cc496d06644abb073fa6ff823e20170b1b70ff3d127723d5165c8.jpg)

![](images/9ba2aaad8f8ea668a763d257f3d091e6a5b689609ac9a783daea305335b3d17c.jpg)

![](images/a84973d9a5ab1bfc4134837a0afa7803758e94888f842e8eaefd7bc3a42a52ee.jpg)  
Figure 1: Reasoning utility and cost are state dependent. (a,b) Scores and token usage vary across reasoning settings and task difficulty. (c) Online interventions show a nonmonotonic reward– efficiency tradeoff. See Appendices J and F.

Prior work has shown that explicit Chain of Thought reasoning contains substantial redundancy<sup>7,8,9</sup>, and existing methods can shorten intermediate reasoning while preserving answer quality<sup>9,10,11,12</sup>. These settings are largely static because a completed reasoning trace can be modified and then evaluated against the same final answer. Agent reasoning is different. Changing reasoning at step t may alter the next tool call, observation, and subsequent decisions. Reasoning compression for agents therefore intervenes in a closed loop decision process rather than simply modifying text <sup>13,14,15</sup>.

Reasoning can move across different forms of state. A plan, constraint, or intermediate conclusion may exist only in the reasoning trace, but become available through tool calls, code, files, or environmental feedback. Context management and memory systems use persistent or external state to reduce dependence on the active context<sup>16,17,18,19</sup>. We therefore hypothesize that the future necessity of reasoning depends on the current agent state and on whether task relevant information has already been externalized.

We propose Interaction Aware Compression for Long Horizon Reasoning (ICLR), a training free compression method for long horizon agents. After each interaction step, a frozen proxy model estimates the predictive entropy of reasoning blocks and preferentially compresses low entropy blocks<sup>9</sup>. The compressed history is directly used by the next agent request. Across long horizon agent tasks from multiple domains, ICLR improves reward from 0.699 to 0.718, while reducing input, output, and cache read tokens by 25.5%, 14.4%, and 33.3%.

Our ablations reveal that local deletion and total system computation are not proportional. Small changes to reasoning history can produce much larger changes in token consumption, while removing all reasoning does not necessarily produce the shortest trajectory because local interventions alter later decisions. We refer to this effect as trajectory amplification.

To understand which reasoning should remain available, we further study its future necessity. Using representation probing techniques<sup>20,21,22</sup>, we find that the hidden state at the current context boundary contains decodable information associated with later rederivation, additional reasoning, and replanning, reaching an AUROC of 0.844 under strict task level nested evaluation. Activation patching provides complementary evidence <sup>23</sup>: replacing deleted history representations with their full history counterparts progressively restores the perturbed next token distribution, with KL Repair increasing from 0.047 at layer 4 to 0.459 at layer 20, 0.724 at layer 28, and 0.978 at layer 31.

Trajectory analysis further suggests an internalization and externalization gap. When task specific derived state remains available only in reasoning, future behavioral risk is 67.3%, compared with

36.2% after that state has been externalized. Controlled interventions support the same interpretation: keeping only the latest reasoning preserves the full history rule score, deleting it reduces the score, clearing reasoning after relevant code has been written leaves the final score unchanged, and hiding observed environmental feedback causes the agent to recover the same information through an additional interaction.

Together, these findings suggest that historical reasoning is most valuable when it remains the only reliable carrier of task relevant derived state. As that state is externalized, the original reasoning becomes increasingly replaceable.

Our contributions are threefold.

(1) Training free online reasoning compression for agents. We introduce ICLR, which continuously compresses accumulated reasoning inside a real agent interaction loop while preserving actions, tool calls, and observations.

(2) Trajectory level characterization of agent reasoning compression. Through full benchmark evaluation and systematic ablations, we show that reasoning compression can alter the execution trajectory, producing a nonlinear relation between local deletion and total system computation.

(3) Analysis of future reasoning necessity. We study historical reasoning through hidden state probing, activation patching, trajectory analysis, and information externalization. The results support a view of agent reasoning as dynamic working state whose future value depends on where task relevant information is stored.

## 2 Related Work

## 2.1 Efficient Reasoning and Chain of Thought Compression

Extended Chain of Thought reasoning has become a central mechanism for mathematical reasoning, code generation, and complex decision making, but longer reasoning traces also increase token usage, latency, and inference cost <sup>7,8,24,25</sup>. A broad line of work therefore studies how much explicit reasoning is actually necessary, including adaptive reasoning budgets, early stopping, concise reasoning, and direct pruning of intermediate reasoning steps<sup>9,10,11,26,12</sup>. These studies collectively suggest that reasoning length and task performance are not simply proportional. A complete reasoning trace often mixes critical deductions with repeated checks, confirmations, and procedural elaboration. Identifying which parts can be removed without harming downstream behavior has therefore become an important direction in efficient reasoning.

Recent work further estimates redundancy at the level of individual reasoning steps. Step Entropy<sup>9</sup> ranks reasoning steps according to uncertainty in the predictive distribution and shows that many low entropy steps can be removed from mathematical reasoning traces while preserving final answer accuracy. CONCISE<sup>11</sup>, TokenSkip<sup>12</sup>, and related compression methods similarly reduce explicit reasoning by identifying steps or tokens that contribute less to the final solution. These methods mainly optimize reasoning within a single completed generation. The decision to remove a step is therefore evaluated against a fixed downstream answer rather than an evolving sequence of actions and observations.

Our setting differs because historical reasoning remains part of the agent’s future decision state. Removing a reasoning block can change the next action, which changes the next observation and all later decisions. The relevant objective is therefore not only whether compressed reasoning still supports the same answer, but whether repeated compression remains effective throughout a closed loop interaction.

## 2.2 Context Management and Memory for Language Model Agents

The growth of interaction history has motivated context management methods for long horizon agents<sup>27,28,29,30,31</sup>. Token level prompt compression methods such as LLMLingua<sup>32</sup>, LongLLM-Lingua<sup>33</sup>, LLMLingua-2<sup>34</sup>, context compression<sup>35</sup>, and gist tokens<sup>36</sup> reduce redundant context. Agent oriented approaches manage evolving histories through relevance estimation, summarization, adaptive pruning, explicit context operations, or external memory<sup>37,38,39,40</sup>. Representative examples include ACON<sup>4</sup>, PACE<sup>5</sup>, SWE Pruner<sup>14</sup>, Self Compact<sup>13</sup>, Self GC<sup>41</sup>, Sculptor<sup>15</sup>, ContextBudget<sup>6</sup>, Context as a Tool<sup>42</sup>, ARC<sup>43</sup>, and structured context eviction<sup>44</sup>.

A complementary line of work uses persistent or external memory to reduce dependence on the active context. MemGPT<sup>16</sup>, SAM<sup>17</sup>, CoMem<sup>45</sup>, ACM<sup>18</sup>, Mem1<sup>46</sup>, proactive memory extraction<sup>47</sup>, and transferable agent memory<sup>48</sup> preserve or recover information outside the immediate interaction window. These systems ask what information should remain accessible. We instead ask whether reasoning that has already produced an action still needs to remain after its consequences have been observed.

This distinction follows from the information flow of an agent. Reasoning combines available evidence into plans and intermediate conclusions, while actions and observations can move the same task relevant information into code, files, tool outputs, or environmental feedback<sup>19,42,43</sup>. Historical reasoning may therefore change from being the only carrier of useful derived state to being redundant with information already stored elsewhere. Prior work on hidden state probin g <sup>20,21,22,49</sup> and activation interventions<sup>23</sup> provides tools for studying this transition. We use them to examine whether the current model state predicts future reasoning reuse and whether reasoning becomes more replaceable after task relevant state is externalized.

## 3 Methodology

Figure 2 summarizes the online compression process. ICLR operates inside the interaction loop rather than on a completed trajectory. It reduces accumulated reasoning while preserving the external records that define the evolving task state. We first describe online compression and entropy based selection, then the analyses used to study when historical reasoning remains useful.

## 3.1 Online Reasoning Compression

We consider a long horizon agent that repeatedly reasons, acts, and receives environmental feedback. Let the tth interaction step be

$$
S _ { t } = ( R _ { t } , A _ { t } , O _ { t + 1 } ) ,\tag{1}
$$

where $R _ { t }$ is the generated reasoning, $A _ { t }$ the resulting action or tool call, and $O _ { t + 1 }$ the returned observation. Before the kth model request, the interaction history is

$$
{ \mathcal { H } } _ { k } = ( P , S _ { 1 } , \ldots , S _ { k - 1 } ) ,\tag{2}
$$

where P contains the system prompt, tool definitions, and task instruction.

![](images/30bd879a2547d386cc6e46215a95a5d00729363909a5cd1df368764349be26ba.jpg)  
Figure 2: Overview of ICLR. At each completed agent step, newly generated reasoning is partitioned into blocks while actions, tool calls, and observations are preserved. A frozen proxy scorer estimates token level predictive entropy for each block. Low entropy blocks are removed before the compressed history is written back for the next agent request.

Static Chain of Thought compression modifies a completed reasoning trace before evaluating its final answer. Online compression instead changes the state from which future actions are generated. For the kth request, ICLR induces

$$
\mathcal { H } _ { k } \to \widetilde { \mathcal { H } } _ { k } \to A _ { k } \to O _ { k + 1 } \to \mathcal { H } _ { k + 1 } .\tag{3}
$$

A local compression decision can therefore affect later tool selection, observations, recovery, and termination.

The first agent request is uncompressed. Before each subsequent request, ICLR compresses only the reasoning from the just completed interaction step while preserving its action, tool call, and observation. The compressed reasoning is written back into persistent history and is not restored, so all later model calls operate on the modified trajectory.

Local text removal is not equivalent to total computation saved. A small change to reasoning history can alter later decisions and therefore change both the number and content of future requests. Local deletion should thus be distinguished from total computation over the resulting trajectory.

## 3.2 Entropy Guided and Context Conditioned Pruning

For each completed reasoning trace, we partition the reasoning into blocks,

$$
R _ { t } = ( B _ { 1 } , B _ { 2 } , \ldots , B _ { N } ) .\tag{4}
$$

The current implementation uses double newline boundaries as a deterministic segmentation rule and excludes empty blocks. Compression is applied only to reasoning. Actions, tool arguments, tool outputs, observations, and final answers remain unchanged. This design prevents the compression policy from directly deleting evidence that has already entered the external interaction record.

Proxy entropy. We score each reasoning block using a frozen proxy language model<sup>9</sup>. Given scorer context $c _ { t } ,$ let $z ( c _ { t } )$ denote the logits predicting the next token. The predictive distribution is

$$
p _ { \theta } ( v \mid c _ { t } ) = \mathrm { s o f t m a x } ( z ( c _ { t } ) ) _ { v } .\tag{5}
$$

The token level predictive entropy is

$$
h _ { t } = - \sum _ { v \in \mathcal { V } } p _ { \theta } ( v \mid c _ { t } ) \log _ { 2 } p _ { \theta } ( v \mid c _ { t } ) ,\tag{6}
$$

and the score of block $B _ { i }$ is the mean entropy of its tokens,

$$
H ( B _ { i } ) = { \frac { 1 } { | B _ { i } | } } \sum _ { t \in B _ { i } } h _ { t } .\tag{7}
$$

Lower entropy indicates greater predictive certainty under the frozen proxy model. We use this proxy entropy to rank reasoning blocks for compression. The acting agent is DeepSeek V4 Flash, while the scorer is a frozen Qwen3.5 9B model.

Context conditioned ranking. A reasoning block is not scored in isolation. The scorer receives the system prompt, tool definitions, accumulated interaction history, and the current completed step in their original causal order. The score can therefore be viewed as a context dependent quantity,

$$
H ( B _ { i } \mid \mathcal { H } _ { t } ) ,\tag{8}
$$

so the same reasoning text may receive a different score under a different trajectory state. This is important in agent settings because the relevance of earlier reasoning can change after new actions and observations have modified the task state.

The full context scorer has a 42K token input limit. If this limit is exceeded, pruning is skipped for that step rather than truncating the history observed by the acting agent.

Block selection. Given N reasoning blocks, ICLR removes the lowest entropy fraction,

$$
\mathcal { D } _ { t } = \mathrm { B o t t o m K } _ { \lfloor \rho N \rfloor } \left( \{ H ( B _ { i } ) \} _ { i = 1 } ^ { N } \right) , \qquad \rho = 0 . 8 .\tag{9}
$$

Each block in $\mathcal { D } _ { t }$ is replaced with [SKIP], while all retained blocks preserve their original order and text. When $N = 1$ , no block is removed. Because the rule operates on blocks rather than individual tokens, the realized token reduction varies across interaction steps.

We use a fixed compression ratio rather than tuning a task specific threshold. This choice creates a consistent intervention across trajectories and allows us to study whether agents can repeatedly discard a large fraction of historical reasoning during real interaction.

Intervention variants. We construct three additional variants for controlled comparison. Step Only Entropy scores only the reasoning from the current completed step and removes earlier trajectory context from the scorer. Delete All Thinking removes the complete targeted reasoning trace while preserving actions and observations. Random r removes a deterministic random subset of reasoning tokens with

$$
r \in \{ 5 \% , 1 0 \% , 2 0 \% , 4 0 \% \} .\tag{10}
$$

All variants follow the same online write back protocol. The modified history is therefore consumed by every subsequent real agent request. These interventions separate the effects of context conditioning, selection strategy, and deletion strength.

## 3.3 Future Necessity Probing

The compression policy determines which reasoning is removed, but it does not explain which reasoning will matter again later. We therefore study whether the current model state contains information associated with future reasoning reuse before that later behavior occurs. Following standard representation probing methods<sup>20,21,22</sup>, we analyze the model state available at each compression boundary.

We use a frozen Qwen3.5 9B model as a representation sensor. For sample i, layer l, and pooling rule $p ,$ let $\mathcal { Z } _ { i } ^ { ( l ) }$ denote the hidden states associated with the current context boundary. We define the pooled representation as

$$
z _ { i } ^ { ( l , p ) } = \mathrm { P o o l } _ { p } \left( \mathcal { Z } _ { i } ^ { ( l ) } \right) .\tag{11}
$$

A linear probe then predicts a behavioral target $y _ { i }$ ,

$$
\widehat { p _ { i } } = \sigma \left( w ^ { \top } z _ { i } ^ { ( l , p ) } + b \right) .\tag{12}
$$

The target records whether the later trajectory contains rederivation, additional reasoning, or explicit replanning. We use this behavioral target as an operational measure of future reasoning reuse. The probe characterizes this signal from the current representation.

To prevent task leakage, evaluation uses strict task level nested splits. Layer selection, pooling choice, feature standardization, and PCA are selected using training tasks only. The outer evaluation folds contain held out tasks. This protocol asks whether the current representation contains information that generalizes across tasks rather than information specific to trajectories already seen during probe selection. Full extraction and evaluation details are reported in Appendix G.

## 3.4 Mechanistic Interventions for Reasoning Persistence

Representation probing tests whether future reasoning reuse can be predicted from the current state. We complement this analysis with interventions that examine how historical reasoning affects model predictions and how its role changes after task relevant information becomes available outside the reasoning trace.

Activation patching. We first test whether representations conditioned on historical reasoning participate in the next token prediction of the frozen sensor model <sup>23</sup>. For sample i, let $P _ { i } ^ { \mathrm { f u l l } }$ denote the next token distribution under the full reasoning history and $P _ { i } ^ { \mathrm { d e l } }$ the corresponding distribution after historical reasoning is removed. At layer l, we replace the prompt boundary residual under the deleted history condition with the residual from the full history condition. The resulting distribution is denoted by $P _ { i , l } ^ { \mathrm { p a t c h } }$

Let

$$
d _ { i } = D _ { \mathrm { K L } } \left( P _ { i } ^ { \mathrm { f u l l } } \Vert P _ { i } ^ { \mathrm { d e l } } \right) .\tag{13}
$$

For samples with $d _ { i } > 0$ , we measure distributional recovery as

$$
{ \mathrm { R e p a i r } } _ { i } ( l ) = 1 - { \frac { D _ { \mathrm { K L } } \left( P _ { i } ^ { \mathrm { f u l l } } \Vert P _ { i , l } ^ { \mathrm { p a t c h } } \right) } { d _ { i } } } .\tag{14}
$$

Larger values indicate that the patched representation more closely recovers the full history next token distribution. The intervention is performed only on the frozen sensor model and does not modify the online ICLR policy.

State externalization. We next examine whether reasoning becomes more replaceable after task specific derived state has been written into a persistent external carrier. Let $\mathcal { U } _ { i }$ denote the unexternalized task specific derived state at compression boundary i. We define

$$
G _ { i } = \mathbb { I } \left( \mathcal { U } _ { i } \neq \emptyset \right) .\tag{15}
$$

Here, derived state refers to an intermediate plan, relation, constraint, synthesis, or task specific conclusion rather than a direct copy of the task instruction or a raw observation. External carriers include code, files, tool outputs, and explicit environmental feedback.

The indicator $G _ { i }$ separates states in which reasoning remains the only available carrier of task relevant derived information from states in which that information has already been materialized elsewhere. This allows us to study whether future reasoning reuse changes as information moves from internal reasoning into persistent external state.

Controlled state interventions. Finally, we construct fixed state continuations that modify the available reasoning or feedback while keeping the surrounding task state unchanged. The interventions compare retaining or deleting recent reasoning, clearing reasoning after relevant code has been written to disk, and retaining or hiding previously observed environmental feedback.

These interventions test whether the next decision still depends on information available only in the reasoning trace. Together, probing, activation patching, state externalization analysis, and controlled interventions provide complementary views of reasoning persistence. The full protocols are reported in Appendices H and I.

## 4 Experiments

We evaluate ICLR on WorkBuddyBench using DeepSeek V4 Flash as the task executing agent and a frozen Qwen3.5 9B model as the proxy entropy scorer. WorkBuddyBench contains long horizon tasks from Code, Office, Security, and Web domains. We report official task reward together with input tokens, output tokens, cache read tokens, and wins, ties, and losses. The full benchmark contains 260 tasks, consisting of 80 Code, 50 Office, 60 Security, and 70 Web tasks. A fixed 80 task set with 20 tasks from each domain is used for ablation analysis.

## 4.1 Overall Performance

We first compare ICLR with representative context management baselines on the fixed Pilot40 task set used by prior WorkBuddyBench context management experiments. Table 1 includes simple sliding window baselines, periodic summarization, and representative context management methods including PACE, SelfCompact, ACON Core, SAM, and SWE Pruner. This comparison focuses on the tradeoff between task quality and token consumption. ICLR directly prunes low entropy reasoning from the online interaction history using a frozen scorer, providing a lightweight alternative to summary based or external memory based context management.

We next evaluate ICLR on the complete 260 task benchmark. Table 2 follows the score and token presentation used by prior WorkBuddyBench context management evaluations. Average reward increases from 0.699 to 0.718. Detailed token accounting is reported separately in the appendix. The method obtains 102/76/82 wins, ties, and losses. These results show that aggressive online pruning of reasoning can reduce repeated processing of historical context without sacrificing aggregate task performance across domains.

Domain-level results and component-wise token changes are reported in Appendix E.

Table 1: Comparison with context management methods on WorkBuddyBench Pilot40. All methods use the same fixed 40 tasks. Scores are multiplied by 100. Parenthesized values show changes from the DeepSeek-V4-Flash base agent.
<table><tr><td>Method</td><td>Code</td><td>Office</td><td>Sec.</td><td>Web</td><td>Avg.</td><td>Tokens↓</td></tr><tr><td colspan="7">Base and Ours</td></tr><tr><td>DeepSeek-V4-Flash</td><td>72.9</td><td>81.6</td><td>44.6</td><td>71.0</td><td>67.5</td><td>1.89M</td></tr><tr><td>ICLR (Ours)</td><td>74.1 (+1.2)</td><td>81.1 (-0.5)</td><td>64.1 (+19.5)</td><td>62.0 (-9.0)</td><td>70.3 (+2.8)</td><td>2.02M (+6.9%)</td></tr><tr><td colspan="7">Simple Context Baselines</td></tr><tr><td></td><td rowspan="6">66.8 (-6.1) 67.4 (-14.2) 74.4 (+1.4) 73.1 (-8.6)</td><td rowspan="6"></td><td rowspan="6">32.3 (-12.3)</td><td rowspan="6">59.0 (-12.0)</td><td rowspan="6">56.4 (-11.2) 62.1 (-5.4)</td><td rowspan="6">2.95M (+56.3%) 1.98M (+5.1%)</td></tr><tr><td>Sliding Window (K = 5)</td></tr><tr><td>Sliding Window (K = 10)</td></tr><tr><td>Sliding Window (K = 20) 79.3 (+6.4) 85.1 (+3.5)</td><td>2.02M (+7.0%)</td></tr><tr><td>Periodic Summary (n = 3)</td><td>70.0 (-1.0) 68.7 (+1.1) 82.7 (+11.7) 60.5 (-7.0) 2.21M (+17.2%)</td></tr><tr><td>Periodic Summary (n = 5)</td><td>70.2 (+2.6) 2.37M (+25.5%)</td></tr><tr><td colspan="7">API-based Methods</td></tr><tr><td>PACE⁵</td><td rowspan="6">70.4 (-2.6)</td><td rowspan="6">55.8 (-25.9) 43.5 (-1.0)</td><td rowspan="6">64.0 (-7.0)</td><td rowspan="6">58.4 (-9.1) 70.0 (-1.0) 62.0 (-5.5)</td><td rowspan="6">1.10M (-41.6%)</td><td rowspan="6">2.40M (+27.3%)</td></tr><tr><td>LLMLingua-234</td></tr><tr><td>SelfCompact13</td></tr><tr><td>ACON-Core⁴</td></tr><tr><td>71.8 (-1.1) Self-GC41 63.5 (-9.5)</td></tr><tr><td>71.7 (-10.0) 62.6 (-10.3)</td></tr><tr><td>LRE50 CoMem45</td><td rowspan="5">63.4 (-18.2) 56.4 (-16.5)</td><td>27.1 (-17.4) 2.5 (-42.1)</td><td rowspan="5">68.0 (-3.0) 58.0 (-13.0)</td><td rowspan="5">55.3 (-12.2) 44.2 (-23.3)</td><td rowspan="5">2.29M (+21.3%)</td><td rowspan="5">0.77M (-59.5%)</td></tr><tr><td></td><td>59.9 (-21.8)</td></tr><tr><td>SAM17</td><td>81.6 (0.0) 47.2 (+2.7) 67.0 (-4.0)</td></tr><tr><td>SWE-Pruner 14</td><td>40.9 (-3.7) 65.0 (-6.0)</td></tr><tr><td>Sculptor 15</td><td>35.2 (-9.3) 69.0 (-2.0)</td></tr><tr><td colspan="5">51.2 (-21.7) 81.6 (0.0) 59.3 (-8.3)</td></tr><tr><td>ACM18</td><td colspan="5">Released-policy Method 38.7 (-34.2) 55.9 (-25.7) 6.2 (-38.4)</td></tr></table>

Table 2: Full benchmark system performance and token usage. Scores are multiplied by 100. Green and red indicate favorable and unfavorable changes from the base system.
<table><tr><td>Model</td><td>Code</td><td>Office</td><td>Sec.</td><td>Web</td><td>Avg.</td><td>Tokens ↓</td></tr><tr><td>DeepSeek-V4-Flash</td><td>77.0</td><td>81.8</td><td>47.8</td><td>72.1</td><td>69.9</td><td>2.69M</td></tr><tr><td>ICLR (Ours)</td><td>70.6 (-6.4)</td><td>77.6 (-4.2)</td><td>70.7 (+22.9)</td><td>69.9 (-2.2)</td><td>71.8 (+1.9)</td><td>2.19M (-18.5%)</td></tr></table>

## 4.2 Ablations and Trajectory Amplification

We next study whether the observed effect depends on entropy based selection, access to full history during scoring, or simply reducing the amount of reasoning. Table 3 reports results on the fixed 80 task ablation set. ICLR reaches a reward of 0.738, compared with 0.627 for the baseline. Step only entropy reaches 0.711, while deleting all thinking reaches 0.718. Random deletion also changes task performance, but none of the reported random variants reaches the reward of full history entropy. These results suggest that reasoning history contains substantial redundancy while also indicating that context conditioned ranking provides useful information beyond deletion alone. Appendix F reports the full intervention and token summaries used for this analysis.

![](images/58eac450083e5c1e02800a04bd0ec08b6a3f0884541d210fb20495dd76a785d9.jpg)

Figure 3: Trajectory amplification. Token savings do not track deletion.  
Table 3: Ablation results on the fixed 80 task WorkBuddyBench set. All variants use the same fixed 80 tasks, with 20 Code, 20 Office, 20 Security, and 20 Web tasks. Scores are multiplied by 100. Parenthesized values show changes from the Base Agent.
<table><tr><td>Method</td><td>Code</td><td>Office</td><td>Sec.</td><td>Web</td><td>Avg.</td><td>Tokens↓</td></tr><tr><td colspan="7">Base and Ours</td></tr><tr><td>Base Agent</td><td>69.09</td><td>78.94</td><td>44.17</td><td>58.50</td><td>62.67</td><td>3.00M</td></tr><tr><td>ICLR (Ours)</td><td>63.85 (-5.24)</td><td>81.14 (+2.20)</td><td>72.03 (+27.86)</td><td>78.00 (+19.50)</td><td>73.76 (+11.09)</td><td>2.49M (-17.0%)</td></tr><tr><td colspan="7">Reasoning Removal Controls</td></tr><tr><td>Delete All Thinking</td><td>69.20 (+0.11)</td><td>82.04 (+3.10) 67.41 (+23.24) 68.50 (+10.00)</td><td></td><td></td><td>71.78 (+9.11)</td><td>1.11M (-62.9%)</td></tr><tr><td>Random 5%</td><td>59.81 (-9.28)</td><td>81.12 (+2.18) 64.17 (+20.00)</td><td></td><td>66.83 (+8.33)</td><td>67.98 (+5.31)</td><td>1.40M (-53.5%)</td></tr><tr><td>Random 10%</td><td>55.08 (-14.01)</td><td>82.65 (+3.71) 71.91 (+27.74)</td><td></td><td>60.61 (+2.11)</td><td>67.56 (+4.89)</td><td>1.19M (-60.3%)</td></tr><tr><td>Random 20%</td><td>67.23 (-1.86)</td><td>76.92 (-2.02)</td><td>64.82 (+20.65)</td><td>61.50 (+3.00)</td><td>67.62 (+4.95)</td><td>1.69M (-43.8%)</td></tr><tr><td>Random 40%</td><td>61.35 (-7.74)</td><td>76.17 (-2.77) 61.37 (+17.20) 71.62 (+13.12)</td><td></td><td></td><td>67.63 (+4.96)</td><td>1.47M (-51.0%)</td></tr><tr><td colspan="7">Scoring Context Ablation</td></tr><tr><td>Step Only Entropy</td><td>60.48 (-8.61) 80.58 (+1.64) 73.10 (+28.93) 70.38 (+11.88)</td><td></td><td></td><td></td><td>71.14 (+8.47)</td><td>0.98M (-67.5%)</td></tr></table>

The most striking observation is that the local deletion ratio is not monotonically related to final system level

computation. Random deletion of 10% of reasoning tokens reduces total input by 60.5%, whereas random deletion of 20% reduces input by only 44.0%. Deleting all thinking reduces input by 63.6%, which is not qualitatively proportional to deleting all reasoning text. Different interventions also produce substantially different numbers of agent calls. We refer to this behavior as trajectory amplification: a local change to reasoning history can be amplified through tool selection, observations, repeated reasoning, recovery, and termination, producing a much larger change in the final trajectory. Detailed intervention and token results are reported in Appendix F.

## 4.3 Future Necessity Analysis

We evaluate future necessity on 265 usable compression boundary samples from 31 tasks, with 112 positive behavioral labels. Under strict task level nested evaluation, the probe reaches an AUROC of 0.844, an AP of 0.811, and a balanced accuracy of 0.760. The prompt last representation is selected in all outer folds, while the selected layers span middle and later parts of the network. These results show that the current representation contains a readable signal associated with later rederivation, additional reasoning, and replanning. Full cohort and layer selection details are reported in Appendix G.1.

Activation patching provides complementary evidence at the representation level. On the 47 sample patching cohort, replacing deleted history residuals with their full history counterparts progressively restores the next token distribution. Mean KL Repair rises from 0.047 at layer 4 to 0.459 at layer 20, 0.724 at layer 28, and 0.978 at layer 31. The complete layerwise results, confidence intervals, and top one agreement values are reported in Appendix H, Table 13.

## 4.4 State Externalization and Controlled Interventions

Trajectory analysis provides a state based explanation for when historical reasoning remains useful. When task specific derived state is still available only in reasoning, the future behavioral risk rate is 67.3%, compared with 36.2% when that state has already been externalized. The within task increase is 19.8 percentage points, and the corresponding hidden score difference is 0.283. Detailed group statistics and task level intervals are reported in Appendix I.1.

Controlled interventions support the same interpretation. Keeping only the latest reasoning preserves the full history rule score of 0.872, whereas deleting only the latest reasoning reduces the score to 0.760, and deleting all reasoning reduces it to 0.675. Once relevant code has been written to disk, clearing subsequent reasoning leaves the rule score unchanged at 0.759. Environmental feedback shows the same pattern. When an observed stderr remains available, the agent proceeds directly to Edit. When the same feedback is hidden, the agent issues another Bash call and observes the same error again. The fixed state experiments are reported in Appendix I.2, and the environmental feedback intervention is reported in Appendix I.3.

Taken together, these results suggest that historical reasoning is most useful while it remains the only reliable carrier of task relevant derived state. As that state is transferred into persistent artifacts or environmental feedback, the original reasoning becomes increasingly replaceable.

## 5 Conclusion

We study when a long horizon agent can forget its own reasoning. Unlike static Chain of Thought compression, online compression changes future decisions and the resulting trajectory. ICLR uses frozen proxy entropy to remove low entropy reasoning while preserving actions and observations, reducing token consumption and improving aggregate reward on the full 260 task evaluation. Ablations reveal trajectory amplification, where local deletion and final system savings are not proportional. Representation probing and activation patching show that current states encode information associated with later reasoning reuse, while controlled analyses indicate that historical reasoning becomes more replaceable as task relevant derived state moves into reliable external carriers. These results more clearly characterize agent reasoning as dynamic working state whose value changes throughout interaction.

## AI use statement

Generative AI tools were used for language editing, drafting assistance, and literature discovery during preparation of this manuscript. All AI assisted text, claims, citations, and experimental descriptions were reviewed by the authors. Experimental results and interpretations were checked against the underlying records, and the authors take full responsibility for the final content of the paper.

## Reproducibility statement

The main paper specifies the online compression protocol, proxy entropy definition, pruning rule, benchmark composition, and principal ablations. The appendix provides implementation details, evaluation set composition, extended system and ablation results, hidden state probing and activation patching protocols, controlled state interventions, and theoretical analysis.

## References

1. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

2. Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

3. Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

4. Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, and Saravan Rajmohan. Acon: Optimizing context compression for long-horizon llm agents. arXiv preprint arXiv:2510.00615, 2025.

5. Lei Wei, Xiao Peng, Guannan Zhang, Chenhao Jiang, Hongyu Li, Lanbo Lin, Yuanwu Xu, Jiayao Liu, Kesu Wang, Bin Wang, et al. Pace: Predictive adaptive context extraction for long-horizon llm agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27184–27199, 2026.

6. Yong Wu, YanZhao Zheng, TianZe Xu, ZhenTao Zhang, YuanQiang Yu, JiHuai Zhu, Chao Ma, BinBin Lin, BaoHua Dong, HangCheng Zhu, et al. Contextbudget: Budget-aware context management for long-horizon search agents. arXiv preprint arXiv:2604.01664, 2026.

7. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

8. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

9. Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, and Qiang Xu. Making slow thinking faster: Compressing llm chain-of-thought via step entropy. In International Conference on Learning Representations, volume 2026, pages 38331–38349, 2026.

10. Juncai Li, Ru Li, Yuxiang Zhou, Boxiang Ma, and Jeff Z Pan. Chain of thought compression: A theoretical analysis. arXiv preprint arXiv:2601.21576, 2026.

11. Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Guanbo Wang, Fandong Meng, Jie Zhou, Ju Ren, and Yaoxue Zhang. Concise: Confidence-guided compression in step-by-step efficient reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8021–8040, 2025.

12. Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3351–3363, 2025.

13. Tianjian Li, Jingyu Zhang, William Jurayj, Xi Wang, Chuanyang Jin, Mehrdad Farajtabar, Eric Nalisnick, and Daniel Khashabi. Self-compacting language model agents. arXiv preprint arXiv:2606.23525, 2026.

14. Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, and Xiaodong Gu. Swe-pruner: Self-adaptive context pruning for coding agents. arXiv preprint arXiv:2601.16746, 2026.

15. Mo Li, LH Xu, Qitai Tan, Long Ma, Hongyong Song, Ting Cao, and Yunxin Liu. Sculptor: Empowering llms with cognitive agency via active context management. In International Conference on Learning Representations, volume 2026, pages 153411–153440, 2026.

16. Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

17. Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Ziliang Zhao, Jiejun Tan, Zheng Liu, and Zhicheng Dou. Sam: State-adaptive memory for long-horizon reasoning agent. arXiv preprint arXiv:2605.24468, 2026.

18. Xiaochuan Li, Ryan Ming, Meng Chu, Shuai Shao, Rong Jin, and Chenyan Xiong. Acm: Agentic context management for long horizon tasks. arXiv preprint arXiv:2607.23809, 2026.

19. Yin Lin, Elaine Ang, Erkang Zhu, Bolin Ding, and Jingren Zhou. Context as an environment: Programmatic context management for long-horizon agents. arXiv preprint arXiv:2608.21690, 2026.

20. Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

21. John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (emnlp-ijcnlp), pages 2733–2743, 2019.

22. Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.

23. Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

24. Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

25. Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman. Quiet-star: Language models can teach themselves to think before speaking. arXiv preprint arXiv:2403.09629, 2024.

26. Yi Sui, Chaozhuo Li, and Dawei Song. Think less, know more: State-aware reasoning compression with knowledge guidance for efficient reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 22470–22491, 2026.

27. Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.

28. Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. In International Conference on Learning Representations, volume 2024, pages 8220–8238, 2024.

29. Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. arXiv preprint arXiv:2404.07143, 2024.

30. Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, volume 2024, pages 21875–21895, 2024.

31. Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in neural information processing systems, 36:34661–34710, 2023.

32. Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 13358–13376, 2023.

33. Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1658–1677, 2024.

34. Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, et al. Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 963–981, 2024.

35. Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. Compressing context to enhance inference efficiency of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 6342–6353, 2023.

36. Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327–19352, 2023.

37. Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. Scaling long-horizon llm agent via context-folding. arXiv preprint arXiv:2510.11967, 2025.

38. Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025.

39. Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, et al. Agentfold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699, 2025.

40. Nikhil Verma. Active context compression: Autonomous memory management in llm agents. arXiv preprint arXiv:2601.07190, 2026.

41. Xubin Hao, Hongjin Meng, Xin Yin, Jiawei Zhu, and Chenpeng Cao. Self-gc: Self-governing context for long-horizon llm agents. arXiv preprint arXiv:2607.00692, 2026.

42. Shukai Liu, Bo Jiang, Jian Yang, Yizhi Li, Jinyang Guo, Xianglong Liu, and Bryan Dai. Context as a tool: Context management for long-horizon swe-agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 20604–20617, 2026.

43. Yilun Yao, Shan Huang, Elsie Dai, Zhewen Tan, Zhenyu Duan, Shousheng Jia, Yanbing Jiang, and Tong Yang. Arc: Active and reflection-driven context management for long-horizon information seeking agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 18644–18659, 2026.

44. Andrew Semenov and Svyatoslav Dorofeev. Beyond compaction: Structured context eviction for long-horizon agents. arXiv preprint arXiv:2606.11213, 2026.

45. Yuwei Zhang, Chengyu Dong, Shuowei Jin, Changlong Yu, Hejie Cui, Hongye Jin, Xinyang Zhang, Hamed Bonab, Colin Lockard, Jianshu Chen, et al. Comem: Context management with a decoupled long-context model. arXiv preprint arXiv:2605.30842, 2026.

46. Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low, and Paul Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. In International Conference on Learning Representations, volume 2026, pages 58413–58438, 2026.

47. Chengyuan Yang, Zequn Sun, Wei Wei, and Wei Hu. Beyond static summarization: Proactive memory extraction for llm agents. arXiv preprint arXiv:2601.04463, 2026.

48. Sirui Liang, Pengfei Cao, Jian Zhao, Wenhao Teng, Xiangwen Liao, Jun Zhao, and Kang Liu. Learning how to remember: A meta-cognitive management method for structured and transferable agent memory. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 30733–30753, 2026.

49. Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827, 2022.

50. Nusrat Jahan Lia and Aritra Mazumder. Learning what not to forget: Long-horizon agent memory from a few kilobytes of learning. arXiv e-prints, pages arXiv–2606, 2026.

## A The Use of Large Language Models

Generative AI tools were used for language editing, drafting assistance, literature discovery, LaTeX preparation, and organization of supplementary explanations. All claims, citations, mathematical statements, and experimental descriptions were reviewed by the authors.

Scope. The supplementary material expands the definitions, protocols, theoretical properties, and experimental results reported in the main paper.

## B Theoretical Analysis

## B.1 Notation and scope

At one compression boundary, let C denote the context preceding the completed reasoning, $B =$ $\left( \boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { N } \right)$ the ordered reasoning blocks, and $U$ the unchanged nonreasoning records of the completed step, including its action, tool record, and returned observation. Let Y denote a downstream target under a fixed reference distribution $P .$ . We write $B _ { < j } = ( B _ { 1 } , \dotsc , B _ { j - 1 } )$ $B _ { - j }$ for all reasoning blocks except $B _ { j }$ , and $B _ { D }$ for the chronologically ordered tuple indexed by $D$

We use $\mathsf { H } _ { P }$ and $I _ { P }$ for population Shannon entropy and mutual information, while $H ( B _ { j } )$ in Section 3.2 denotes the realized proxy score. The following results connect these quantities through the information structure of retained and removed reasoning.

## B.2 Residual information of a reasoning block

Lemma B.1 (Residual information bound). For every block $B _ { j }$

$$
I _ { P } ( B _ { j } ; Y \mid C , U , B _ { - j } ) \le { \sf H } _ { P } ( B _ { j } \mid C , B _ { < j } ) .\tag{B.1}
$$

Proof. By the definition of conditional mutual information,

$$
I _ { P } ( B _ { j } ; Y \mid C , U , B _ { - j } ) = { \sf H } _ { P } ( B _ { j } \mid C , U , B _ { - j } ) - { \sf H } _ { P } ( B _ { j } \mid C , U , B _ { - j } , Y )\tag{B.2}
$$

$$
\leq \mathsf { H } _ { P } ( B _ { j } \mid C , U , B _ { - j } )\tag{B.3}
$$

$$
\leq { \mathsf { H } } _ { P } ( B _ { j } \mid C , B _ { < j } ) .\tag{B.4}
$$

The first inequality follows from nonnegativity of conditional entropy. The second follows because conditioning on $U$ and all blocks in $B _ { - j }$ includes the earlier blocks $B _ { < j }$ and can only reduce conditional entropy. □

Interpretation. Under the reference distribution, a block with low true conditional entropy has a correspondingly limited residual information contribution about $Y$ . This bound motivates conditional uncertainty as a compression signal and provides an information theoretic view of why low uncertainty reasoning can be more replaceable.

## B.3 Several removed blocks

Theorem B.2 (Fixed subset information bound). For any index set $D \subseteq \{ 1 , \dots , N \}$ fixed independently ofthe realized block values,

$$
I _ { P } ( B _ { D } ; Y \mid C , U , B _ { \bar { D } } ) \leq \sum _ { j \in D } { \sf H } _ { P } ( B _ { j } \mid C , B _ { < j } ) ,\tag{B.5}
$$

where $\bar { D }$ is the complement of $D _ { \ l }$

Proof. Write $D = \{ j _ { 1 } < \cdots < j _ { m } \}$ . Then

$$
I _ { P } ( B _ { D } ; Y \mid C , U , B _ { \bar { D } } ) \le { \sf H } _ { P } ( B _ { D } \mid C , U , B _ { \bar { D } } )\tag{B.6}
$$

$$
= \sum _ { r = 1 } ^ { m } \mathsf { H } _ { P } \left( B _ { j _ { r } } \mid C , U , B _ { \bar { D } } , B _ { j _ { 1 } } , \ldots , B _ { j _ { r - 1 } } \right)\tag{B.7}
$$

$$
\leq \sum _ { r = 1 } ^ { m } \mathsf { H } _ { P } ( B _ { j _ { r } } \mid C , B _ { < j _ { r } } ) .\tag{B.8}
$$

The first inequality bounds mutual information by conditional entropy. The equality is the entropy chain rule. In each term, the conditioning set contains every earlier block, either because that block is retained in $B _ { \bar { D } }$ or because it appears among the previously selected blocks. □

Adaptive ranking. The implemented deletion set is selected from observed proxy entropy scores and is therefore data dependent. Theorem B.2 characterizes the fixed subset information structure, while ICLR extends this principle through online ranking of observed reasoning blocks.

## B.4 Online trajectory perturbation

The main paper emphasizes that a local edit changes the state from which later actions are generated. The following bound makes this dependence explicit. Consider a common finite horizon K and two policies interacting with the same environment from the same initial state. At any common observable history $h ,$ let $\pi _ { t } ( \cdot \mid h )$ and $\widetilde { \pi } _ { t } ( \cdot \mid h )$ denote the next action laws of the base and compressed systems.

Theorem B.3 (Finite horizon trajectory perturbation). Suppose that for every jointly reachable common history,

$$
\mathrm { T V } \left( \pi _ { t } ( \cdot  { | } h ) , \pi _ { t } ( \cdot  { | } h ) \right) \le \epsilon _ { t } , \qquad 0 \le \epsilon _ { t } \le 1 .\tag{B.9}
$$

Then the observable trajectory laws satisfy

$$
\mathrm { T V } \left( \mathbb { P } _ { \pi } , \mathbb { P } _ { \widetilde { \pi } } \right) \leq 1 - \prod _ { t = 1 } ^ { K } ( 1 - \epsilon _ { t } ) \leq \operatorname* { m i n } \left\{ 1 , \sum _ { t = 1 } ^ { K } \epsilon _ { t } \right\} .\tag{B.10}
$$

For any terminal score $F \in [ 0 , 1 ]$

$$
\left| \mathbb { E } _ { \pi } F - \mathbb { E } _ { \widetilde { \pi } } F \right| \leq \operatorname* { m i n } \left\{ 1 , \sum _ { t = 1 } ^ { K } \epsilon _ { t } \right\} .\tag{B.11}
$$

Proof. While the two histories agree, couple their next actions maximally. The coupled actions agree with probability at least $1 - \epsilon _ { t }$ . Conditional on equal actions and equal observable histories, couple the next observation identically through the common environment transition. Induction gives a probability of complete trajectory agreement of at least $\prod _ { t } ( 1 - \epsilon _ { t } )$ . Total variation is bounded by the mismatch probability of any coupling, which gives the first inequality. The second follows from the union bound. Since $F \in [ 0 , 1 ]$ , its values can differ by at most one on the mismatch event, which yields Equation (B.11). □

Interpretation. This result formalizes how repeated local policy changes can accumulate across an interaction trajectory and provides a theoretical basis for trajectory amplification.

## B.5 A sufficient condition for forgetting externalized state

Let $Z _ { t }$ denote task relevant state that is available from the retained external interaction history, and let $W _ { t }$ denote reasoning content that would be removed.

Corollary B.4 (Externalized state sufficiency). Suppose that at every reachable history the full history and compressed policies share the same retained state $Z _ { t } ,$ , and that

$$
\pi _ { t } ( a \mid Z _ { t } , W _ { t } ) = q _ { t } ( a \mid Z _ { t } ) = { \widetilde { \pi } } _ { t } ( a \mid Z _ { t } ) .\tag{B.12}
$$

Under the common environment assumptions of Theorem $B . 3 ,$ , the two systems induce identical observable trajectory and terminal score distributions.

Proof. Equation (B.12) implies zero total variation between the two next action laws at every common history. Thus $\epsilon _ { t } = 0$ for all t in Theorem B.3, and both the trajectory and terminal score distances are zero. □

Interpretation. The corollary gives a sufficient condition under which externalized state can fully replace discarded reasoning. The controlled experiments in Section 4 examine local cases in which useful derived state moves from reasoning into code, files, tool outputs, or environmental feedback.

## B.6 Exact token accounting and trajectory amplification

Let

$$
T ^ { \mathrm { b } } = \sum _ { k = 1 } ^ { K _ { b } } \left( I _ { k } ^ { \mathrm { b } } + { \cal O } _ { k } ^ { \mathrm { b } } \right) , \qquad T ^ { \mathrm { c } } = \sum _ { k = 1 } ^ { K _ { c } } \left( I _ { k } ^ { \mathrm { c } } + { \cal O } _ { k } ^ { \mathrm { c } } \right)\tag{B.13}
$$

be the processed token totals of the base and compressed trajectories, where $I _ { k }$ and $O _ { k }$ denote input and output tokens of request k.

Proposition B.5 (Trajectory level token decomposition). Let $K _ { * } = \operatorname* { m i n } ( K _ { b } , K _ { c } )$ . Then

$$
\begin{array} { r l } { \displaystyle T ^ { \mathrm { c } } - T ^ { \mathrm { b } } = \sum _ { k = 1 } ^ { K _ { * } } \big [ \big ( I _ { k } ^ { \mathrm { c } } - I _ { k } ^ { \mathrm { b } } \big ) + \big ( O _ { k } ^ { \mathrm { c } } - O _ { k } ^ { \mathrm { b } } \big ) \big ] } & { } \\ { + \displaystyle \sum _ { k = K _ { * } + 1 } ^ { K _ { c } } \big ( I _ { k } ^ { \mathrm { c } } + O _ { k } ^ { \mathrm { c } } \big ) - \sum _ { k = K _ { * } + 1 } ^ { K _ { b } } \big ( I _ { k } ^ { \mathrm { b } } + O _ { k } ^ { \mathrm { b } } \big ) . } \end{array}\tag{B.14}
$$

Proof. Split each total into the first $K _ { * }$ requests and its unmatched suffix, then subtract the two sums term by term. □

Interpretation. Direct deletion affects the serialized context of one request, while Equation (B.14) also captures changes to later input, later output, and the number of requests. Whole trajectory savings therefore depend on the induced interaction path as well as the local deletion itself. This identity gives the accounting basis for the trajectory amplification analysis in Section 4.

## C Implementation and Online Compression Protocol

## C.1 Runtime settings

Table 4: Recorded execution settings for the acting agent and frozen proxy scorer.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Acting model</td><td>DeepSeek-V4-Flash</td></tr><tr><td>Thinking / reasoning effort</td><td>Enabled / high</td></tr><tr><td>Requested temperature</td><td></td></tr><tr><td>Requested maximum tokens 384,000</td><td></td></tr><tr><td>Streaming</td><td>Enabled</td></tr><tr><td>Frozen scorer</td><td>acm-browsecompplus-qwen3.5-9b- opd-iter3</td></tr><tr><td>Scorer execution</td><td>eval() and inference_mode()</td></tr><tr><td>Full context scorer limit</td><td>42,000 tokens</td></tr><tr><td>Deletion rule</td><td>Lowest [0.8N] nonempty reasoning blocks</td></tr><tr><td>Replacement text</td><td>Literal [SKIP]</td></tr></table>

Interpretation. The acting model generates actions and reasoning, while the frozen scorer is used only to compute the ranking signal. No scorer update is performed during evaluation. When the scorer input exceeds its 42K limit, pruning is skipped for that step rather than truncating the acting agent’s history.

## C.2 Reasoning extraction and entropy alignment

Reasoning is taken from thinking when available. The remaining fallbacks are reasoning\_ content, reasoning, and a <think>...</think> parser. Nonempty segments separated by \n\n define candidate blocks. Only reasoning spans are editable. Actions, tool arguments, tool outputs, observations, and final answers remain unchanged.

For scorer token $x _ { t } .$ , let $z _ { t - 1 }$ denote the logits that predict that token. The implementation computes

$$
\ell _ { t } ( v ) = \mathrm { l o g s o f t m a x } ( z _ { t - 1 } ) _ { v } ,\tag{C.1}
$$

$$
h _ { t } = - \frac { 1 } { \ln 2 } \sum _ { v \in \mathcal { V } } \exp ( \ell _ { t } ( v ) ) \ell _ { t } ( v ) ,\tag{C.2}
$$

$$
H ( B _ { j } ) = | \mathcal { T } _ { j } | ^ { - 1 } \sum _ { t \in \mathcal { T } _ { j } } h _ { t } ,\tag{C.3}
$$

where $\mathcal { T } _ { j }$ is the set of scorer token positions assigned to block $B _ { j }$

Interpretation. Equation (C.3) is predictive entropy rather than observed token surprisal. The score is averaged within each block and used only for ranking candidate blocks.

## C.3 Online write back

The first agent request is executed without compression. Before each subsequent request, the reasoning from the just completed interaction step is partitioned, scored, and rewritten in persistent history. All later requests consume this rewritten history rather than the original reasoning.

For N nonempty reasoning blocks, ICLR replaces the lowest ⌊0.8N⌋ blocks with [SKIP]. When N = 1, no block is removed. The nominal fraction is defined over blocks, so the realized token reduction varies with block length.

## C.4 Intervention definitions

Table 5: Intervention families used in the ablation study.
<table><tr><td>Variant</td><td>Operation</td></tr><tr><td></td><td>Full history entropy Score with the causal full context and replace the lowest 80% of reasoning blocks.</td></tr><tr><td>Step only entropy</td><td>Score the just completed reasoning without earlier trajectory context.</td></tr><tr><td>Delete all thinking</td><td>Remove the complete targeted reasoning while preserving actions and observations.</td></tr><tr><td>Random r</td><td>Remove a deterministic random subset of reasoning tokens for r ∈ {0.05, 0.10, 0.20, 0.40}.</td></tr></table>

Interpretation. The controls separate three questions. Delete all thinking tests wholesale removal, Random r isolates deletion without entropy ranking, and Step only entropy measures the contribution of earlier trajectory context to scoring. Random removal operates at token level, whereas ICLR operates at block level, so the percentages describe different intervention granularities.

## D Evaluation Protocol

## D.1 Evaluation sets

Table 6: Evaluation sets used for the reported system and representation analyses.
<table><tr><td>Set</td><td>Scope</td></tr><tr><td>Full benchmark</td><td>260 tasks: Code 80, Office 50, Security 60, Web 70</td></tr><tr><td>Pilot40</td><td>Fixed 40 task context management comparison</td></tr><tr><td>Ablation set</td><td>Fixed 80 tasks: 20 per domain</td></tr><tr><td>Representation cohort</td><td>265 usable boundary samples from 31 tasks</td></tr><tr><td>Patching cohort</td><td>47 samples from 8 tasks</td></tr><tr><td></td><td>Fixed state diagnostics Controlled Office and error state continuations</td></tr></table>

Interpretation. These rows correspond to different evaluation units. Whole task results measure end to end agent behavior, while representation and fixed state analyses operate on local states sampled from trajectories. Each analysis is reported with its corresponding denominator.

## D.2 Metrics and token accounting

For a consistently defined usage metric $X$ , we report the relative change

$$
\Delta _ { X } = 1 0 0 \left( \frac { X _ { \mathrm { m e t h o d } } } { X _ { \mathrm { b a s e } } } - 1 \right) .\tag{D.1}
$$

The processed token total in the main tables is input plus output tokens. Cache read usage is reported separately rather than added again to input.

Task performance is measured by the official WorkBuddyBench reward. Domain means are reported together with the aggregate task result. Wins, ties, and losses compare the method and base system on the same benchmark tasks.

## D.3 Task level uncertainty

For the externalization analysis, reported confidence intervals use task level resampling so that multiple states from the same task remain grouped. If $\widehat { \theta }$ is a statistic computed from M task clusters, a percentile interval has the form

$$
\left[ Q _ { 0 . 0 2 5 } ( \widehat { \theta ^ { * } } ) , Q _ { 0 . 9 7 5 } ( \widehat { \theta ^ { * } } ) \right] ,\tag{D.2}
$$

where each bootstrap replicate resamples complete tasks with replacement. The appendix reports the recorded task level intervals for the externalization analysis.

## E Extended System Results

## E.1 Full benchmark domain results

Table 7: Full260 domain results. Rewards are shown on the original [0, 1] scale. Input and output columns report relative changes from the base system.
<table><tr><td>Domain</td><td>n</td><td>Base</td><td>ICLR</td><td> $\Delta R$ </td><td> $\Delta I ( \% )$ </td><td> $\Delta O \left( \% \right)$ </td><td>W/T/L</td></tr><tr><td>Code</td><td>80</td><td>0.770000</td><td>0.706329</td><td>-0.063671</td><td>+6.35</td><td>+16.21</td><td>16/40/24</td></tr><tr><td>Office</td><td>50</td><td>0.818000</td><td>0.776152</td><td>-0.041848</td><td>-2.50</td><td>-2.55</td><td>16/8/26</td></tr><tr><td>Security</td><td>60</td><td>0.478000</td><td>0.706816</td><td>+0.228816</td><td>-35.23</td><td>-31.02</td><td>39/9/12</td></tr><tr><td>Web</td><td>70</td><td>0.721000</td><td>0.698551</td><td>-0.022449</td><td>-37.16</td><td>-19.04</td><td>31/19/20</td></tr><tr><td>Overall</td><td>260</td><td>0.698654</td><td>0.717775</td><td>+0.019121</td><td>-25.48</td><td>-14.41</td><td>102/76/82</td></tr></table>

Interpretation. The aggregate reward increases from 0.699 to 0.718, with the largest positive domain change appearing in Security. Code, Office, and Web show smaller domain scores than their base runs, and Code also uses more input. The complete 260 task aggregate combines these domain specific effects. The aggregate cache read reduction is 33.28%.

## E.2 Pilot40 precision

Interpretation. The Pilot40 table corresponds to the fixed forty task context management comparison used in Table 1, while Full260 reports the complete benchmark evaluation.

Table 8: Recorded Pilot40 precision for ICLR.
<table><tr><td>Code</td><td>Office</td><td>Security</td><td>Web</td><td>Mean score</td><td>Total I + O</td></tr><tr><td>74.10</td><td>81.06</td><td>64.11</td><td>62.00</td><td>70.32</td><td>80,812,679</td></tr></table>

## F Ablation Details

## F.1 Reward summary

Table 9: Ablation reward summary on the fixed 80 task set. Values correspond to the main ablation table and are shown on the original [0, 1] scale.
<table><tr><td>Intervention</td><td>Code</td><td>Office</td><td>Security</td><td>Web</td><td>Average</td></tr><tr><td>Base Agent</td><td>0.6909</td><td>0.7894</td><td>0.4417</td><td>0.5850</td><td>0.6267</td></tr><tr><td>ICLR</td><td>0.6385</td><td>0.8114</td><td>0.7203</td><td>0.7800</td><td>0.7376</td></tr><tr><td>Delete All Thinking</td><td>0.6920</td><td>0.8204</td><td>0.6741</td><td>0.6850</td><td>0.7178</td></tr><tr><td>Random 5%</td><td>0.5981</td><td>0.8112</td><td>0.6417</td><td>0.6683</td><td>0.6798</td></tr><tr><td>Random 10%</td><td>0.5508</td><td>0.8265</td><td>0.7191</td><td>0.6061</td><td>0.6756</td></tr><tr><td>Random 20%</td><td>0.6723</td><td>0.7692</td><td>0.6482</td><td>0.6150</td><td>0.6762</td></tr><tr><td>Random 40%</td><td>0.6135</td><td>0.7617</td><td>0.6137</td><td>0.7162</td><td>0.6763</td></tr><tr><td>Step Only Entropy</td><td>0.6048</td><td>0.8058</td><td>0.7310</td><td>0.7038</td><td>0.7114</td></tr></table>

Interpretation. ICLR has the highest reported average reward in this ablation summary. Delete All Thinking also performs above the displayed base average, showing that the history contains substantial removable redundancy. Step Only Entropy remains competitive but below full history entropy, which is consistent with the use of earlier trajectory context in the scoring rule. The random controls show that deletion itself can change task performance, with outcomes depending on more than the configured random removal percentage.

## F.2 Token reduction and trajectory amplification

Table 10: Input and output reductions used in the trajectory amplification analysis.
<table><tr><td>Intervention</td><td>Input reduction (%) Output reduction (%)</td></tr><tr><td>ICLR</td><td>17.03 12.05</td></tr><tr><td>Delete All Thinking</td><td>63.59 13.55</td></tr><tr><td>Random 5%</td><td>53.68 38.46</td></tr><tr><td>Random 10%</td><td>60.53 46.74</td></tr><tr><td>Random 20%</td><td>44.01 26.35</td></tr><tr><td>Random 40%</td><td>51.21 34.22</td></tr><tr><td>Step Only Entropy</td><td>67.88 38.87</td></tr></table>

Interpretation. Input reduction varies nonmonotonically with the configured local deletion ratio. Random 10% reduces input by 60.53%, while Random 20% reduces input by 44.01%. This is the empirical pattern referred to as trajectory amplification. Proposition B.5 explains this behavior through changes to subsequent requests as well as the current serialized history.

Relation to Figure 3. The main figure visualizes the mismatch between local intervention strength and whole trajectory consequences, showing that final token and reward outcomes depend on the trajectory induced after the intervention.

## G Hidden State Probing

## G.1 Cohort and target

A frozen Qwen3.5 9B model reads the context visible at each compression boundary. For sample i, layer l, and pooling rule $p ,$

$$
z _ { i } ^ { ( l , p ) } = \mathrm { P o o l } _ { p } \left( \mathcal { Z } _ { i } ^ { ( l ) } \right) , \qquad \widehat { p } _ { i } = \sigma \left( w ^ { \top } z _ { i } ^ { ( l , p ) } + b \right) .\tag{G.1}
$$

The behavioral target records whether the later trajectory contains rederivation, additional reasoning, or explicit replanning and serves as an operational measure of future reasoning reuse.

Table 11: Representation extraction cohort used for future necessity probing.
<table><tr><td>Quantity Value</td></tr><tr><td>Usable boundary samples 265</td></tr><tr><td>Distinct tasks 31</td></tr><tr><td>Positive behavioral labels 112</td></tr><tr><td>Transformer layers 32</td></tr><tr><td>Pooling candidates 5 Outer task folds 5</td></tr></table>

Interpretation. The unit of analysis is a compression boundary rather than a complete benchmark task. Multiple boundary samples can come from the same task, which is why model selection and evaluation are separated at the task level.

## G.2 Nested task level evaluation

Layer selection, pooling choice, feature standardization, and PCA are chosen using training tasks only. The outer folds contain held out tasks. This design prevents samples from the same task from appearing on both sides of the outer evaluation split.

Table 12: Strict nested task level probe results.
<table><tr><td>AUROC</td><td></td><td>AP Balanced accuracy Selected pooling</td><td></td></tr><tr><td>0.8444 0.8108</td><td></td><td></td><td>0.7597 prompt_last</td></tr></table>

Interpretation. The probe results show that the current hidden state contains a readable cross task signal associated with later reasoning reuse. The linear probe serves as an analysis instrument for characterizing that signal.

![](images/5af42ba7d2c3752166c6237b9c57668f120e6a10b54682e1da22aa976222f90b.jpg)  
Figure 4: Activation patching across depth. Full history residuals progressively restore the perturbed next token distribution when they replace the corresponding deleted history residual at the prompt boundary.

## H Activation Patching

Figure interpretation. The curve summarizes a local intervention on the frozen representation sensor. Larger KL Repair means that the patched representation produces a next token distribution closer to the full history condition. Recovery increases toward later layers, showing that full history information is progressively expressed in representations closer to the output readout.

## H.1 Intervention and measurement

The patching analysis uses 47 samples from 8 tasks on the frozen Qwen3.5 9B sensor. For sample $i ,$ let $P _ { i } ^ { \mathrm { f u l l } }$ be the next token distribution under full history, $P _ { i } ^ { \mathrm { d e l } }$ the distribution after historical reasoning is removed, and $P _ { i , l } ^ { \mathrm { p a t c h } }$ the distribution obtained after replacing the deleted history prompt boundary residual at layer l with the corresponding full history residual.

Define

$$
d _ { i } = D _ { \mathrm { K L } } \left( P _ { i } ^ { \mathrm { f u l l } } \Vert P _ { i } ^ { \mathrm { d e l } } \right) .\tag{H.1}
$$

For $d _ { i } > 0$ , KL Repair is

$$
{ \mathrm { R e p a i r } } _ { i } ( l ) = 1 - { \frac { D _ { \mathrm { K L } } \left( P _ { i } ^ { \mathrm { f u l l } } \Vert P _ { i , l } ^ { \mathrm { p a t c h } } \right) } { d _ { i } } } .\tag{H.2}
$$

A value near one indicates strong distributional restoration relative to the deleted history condition.

## H.2 Layerwise results

Table 13: Activation patching results on the 47 sample cohort.
<table><tr><td>Layer</td><td>Mean KL Repair</td><td>95% CI Top 1 agreement</td></tr><tr><td>4</td><td>0.047</td><td>[0.027, 0.076] 0.681</td></tr><tr><td>16</td><td>0.195</td><td>[0.170, 0.235] 0.745</td></tr><tr><td>20</td><td>0.459</td><td>[0.392, 0.537] 0.851</td></tr><tr><td>24</td><td>0.622</td><td>[0.545, 0.700] 0.851</td></tr><tr><td>28</td><td>0.724</td><td>[0.667, 0.788] 0.872</td></tr><tr><td>31</td><td>0.978</td><td>[0.964, 0.986] 0.936</td></tr><tr><td>32</td><td>1.000</td><td>[1.000, 1.000] 1.000</td></tr></table>

Interpretation. Repair rises substantially from early to late layers, matching the main text summary at layers 4, 20, 28, and 31. Top 1 agreement increases in the same direction. Layer 32 is an implementation positive control because patching the final residual immediately before a deterministic readout restores the full history readout input.

Lemma H.1 (Final layer patching control). Suppose that the final residual $h _ { L }$ is followed by the same deterministic readout map g in the full and deleted history conditions. If the deleted run is patched with $h _ { L } ^ { \mathrm { f u l l } }$ , then

$$
P _ { i , L } ^ { \mathrm { p a t c h } } = P _ { i } ^ { \mathrm { f u l l } } .\tag{H.3}
$$

Whenever $d _ { i } > 0 ,$ , this implies Repair $( L ) = 1$

Proof. After patching, both conditions pass the same residual $h _ { L } ^ { \mathrm { f u l l } }$ through the same deterministic map g. Their resulting next token distributions are therefore identical. Substituting the equality into Equation (H.2) gives zero numerator and repair equal to one. □

## I Externalization and Fixed State Diagnostics

## I.1 Internalization and externalization gap

Let $\mathcal { U } _ { i }$ denote the unexternalized task specific derived state at compression boundary i. We define

$$
G _ { i } = \mathbb { I } \left( \mathcal { U } _ { i } \neq \emptyset \right) .\tag{I.1}
$$

Derived state includes intermediate plans, relations, constraints, syntheses, or task specific conclusions that remain internal to reasoning before materialization in a persistent external carrier.

Table 14: Grouping by the internalization and externalization gap.
<table><tr><td>Group</td><td>n</td><td>Future risk Mean hidden score</td></tr><tr><td>No gap</td><td>213</td><td>36.15% 0.4369</td></tr><tr><td>Gap present</td><td>52</td><td>67.31% 0.6662</td></tr></table>

Interpretation. Future behavioral risk is higher when the relevant derived state remains available only in reasoning. The hidden score shows the same ordering. These pooled values are descriptive and motivate the within task comparison below.

Table 15: Within task contrasts for gap present minus no gap states.
<table><tr><td>Quantity</td><td></td><td>Mean difference Task bootstrap 95% CI</td></tr><tr><td>Future behavioral risk</td><td>+19.75 pp</td><td>[1.00, 37.05]</td></tr><tr><td>Hidden score</td><td>+0.2831</td><td>[0.1703, 0.3977]</td></tr></table>

Interpretation. The within task contrasts preserve the same direction after comparing states from the same tasks, strengthening the state based interpretation in the main text.

## I.2 Controlled state interventions

Table 16: Controlled state interventions. The three groups answer different questions about where useful task state is stored.
<table><tr><td>Intervention</td><td>Rule score</td><td>Reasoning</td><td>New requests</td></tr><tr><td colspan="4">Fixed intermediate Office state</td></tr><tr><td>Keep</td><td>0.8717</td><td>1,817</td><td>5.00</td></tr><tr><td>Delete all</td><td>0.8146</td><td>7,491</td><td>5.33</td></tr><tr><td>Random 5%</td><td>0.8756</td><td>516</td><td>4.00</td></tr><tr><td>Random 10%</td><td>0.8756</td><td>562</td><td>7.33</td></tr><tr><td colspan="4">Position intervention</td></tr><tr><td>Keep all</td><td>0.8717</td><td>1</td><td></td></tr><tr><td>Delete all</td><td>0.6749</td><td>一</td><td>一</td></tr><tr><td>Keep latest only</td><td>0.8717</td><td>一</td><td>1</td></tr><tr><td>Delete latest only</td><td>0.7598</td><td>一</td><td>一</td></tr><tr><td colspan="4">After code is written to disk</td></tr><tr><td>Keep thinking</td><td>0.7587</td><td>1,289</td><td></td></tr><tr><td>Clear thinking</td><td>0.7587</td><td>3,222</td><td></td></tr></table>

Interpretation. The fixed intermediate state shows that complete deletion can trigger substantially more reasoning while lowering the rule score. The position intervention shows that keeping only the latest reasoning matches the keep all score, whereas deleting the latest reasoning causes a larger drop. After relevant code has been written to disk, clearing subsequent reasoning leaves the rule score unchanged at 0.7587. Together, these local interventions support the claim that the importance of reasoning depends on whether task relevant state has already been externalized.

## I.3 Environmental feedback intervention

Table 17: Next action after retaining or hiding the already observed stderr.
<table><tr><td>Feedback</td><td></td><td></td><td>Next action Count Subsequent observation</td></tr><tr><td>stderr visible</td><td>Edit</td><td>9/9</td><td>No repeated error observation</td></tr><tr><td>stderr hidden Bash</td><td></td><td>9/9</td><td>The same ValueError</td></tr><tr><td></td><td></td><td></td><td>appears again</td></tr></table>

Interpretation. When the error remains visible, the agent can act directly on the observed state. When the same feedback is hidden, the agent first performs another Bash call and recovers the same error. This provides a concrete example in which environmental feedback functions as an external information carrier.

## J Motivation Figure Details

Figure 1 is used to motivate the state dependent value of reasoning. Panels (a) and (b) compare task score and token usage across different reasoning settings and difficulty levels. Panel (c) summarizes the empirical reward and input reduction tradeoff under online interventions.

Interpretation. The figure highlights two motivating patterns. First, the relation between reasoning intensity and task score varies across difficulty levels. Second, the relation between local deletion strength and end to end savings varies across interventions. Appendix F provides the corresponding intervention results.

## K Main Text to Appendix Map

Interpretation. The table organizes the paper’s supporting evidence into system level benchmark results, local mechanistic diagnostics, and mathematical analysis.

Table 18: Correspondence between major main text claims and supplementary evidence.
<table><tr><td>Main text claim</td><td>Supplementary location</td><td>Supporting material</td></tr><tr><td>Online compression changes fu- ture interaction</td><td>Appendix B.4</td><td>Finite horizon perturbation theo- rem</td></tr><tr><td>Low entropy ranking is an infor- mation motivated heuristic</td><td>Appendices B.2 and B.3</td><td>Conditional information bounds and qualification</td></tr><tr><td>Local deletion and total compu- tation differ</td><td>Appendix B.6</td><td>Exact trajectory token decomposi- tion</td></tr><tr><td>Full benchmark result</td><td>Appendix E.1</td><td>Domain rewards, token changes, and W/T/L</td></tr><tr><td>Trajectory amplification</td><td>Appendix F.2</td><td>Intervention specific input and out- put reductions</td></tr><tr><td>Future necessity is decodable</td><td>Appendix G</td><td>Cohort, nested protocol, and probe metrics</td></tr><tr><td>Historical reasoning affects rep- resentations</td><td>Appendix H</td><td>Activation patching curve and lay- erwise results</td></tr><tr><td>Externalization changes reason- ing usefulness</td><td>Appendix I</td><td>Gap analysis and controlled state interventions</td></tr></table>

## L Broader Impact and Future Directions

Reducing repeated processing of agent history lowers active context usage and can improve inference efficiency. ICLR preserves actions, tool calls, and observations while selectively compressing reasoning, keeping externally recorded task state available throughout interaction.

Several aspects of the current study motivate future work. Proxy entropy is computed by a frozen model distinct from the acting agent, motivating tighter alignment between scorer and policy representations. The future necessity target is behavioral, suggesting richer labels for different forms of reasoning reuse. Activation patching characterizes the frozen representation sensor, while future studies can extend the same analysis to additional acting models and longer closed loop interventions. Externalization analysis also motivates explicit modeling of when code, files, tool outputs, and observations become sufficient carriers of task relevant derived state.

Raw trajectory access should respect the confidentiality of prompts, credentials, tool outputs, and generated artifacts.

## M Task Manifest and Reproducibility Checklist

## M.1 Task set summary

Interpretation. The Full260, fixed 80 task ablation set, and Pilot40 comparison provide complementary views of complete benchmark performance, controlled ablation behavior, and context management comparison.

Table 19: Task set composition used in the main experiments.
<table><tr><td>Set</td><td>Code</td><td>Office</td><td>Security</td><td>Web Total</td></tr><tr><td>Full benchmark</td><td>80</td><td>50</td><td>60 70</td><td>260</td></tr><tr><td>Ablation set</td><td>20</td><td>20</td><td>20 20</td><td>80</td></tr><tr><td>Pilot40</td><td></td><td></td><td>Fixed forty task comparison</td><td>40</td></tr></table>

## M.2 Available records

Table 20: Reproducibility checklist for the experiments reported in this paper.
<table><tr><td>Item</td><td>Available material</td></tr><tr><td>Method definition</td><td>Block segmentation, proxy entropy, context conditioned scorer, deletion rule, placeholder, and online write back</td></tr><tr><td>Runtime</td><td>Acting model, frozen scorer, scorer context limit, and principal client settings</td></tr><tr><td>Evaluation</td><td>Full260, Pilot40, and fixed 80 task ablation composition and aggregate results</td></tr><tr><td>Probing</td><td>Cohort size, target definition, nested task level protocol, pooling choice, and aggregate metrics</td></tr><tr><td>Patching</td><td>Cohort size, intervention definition, KL Repair, confidence intervals, and layerwise results</td></tr><tr><td>State interventions</td><td>Gap statistics, fixed state rule scores, and environmental feedback intervention</td></tr><tr><td>Theory</td><td>Information bounds, trajectory perturbation bound, externalization sufficiency condition, and token decomposition</td></tr></table>

Interpretation. The checklist summarizes the implementation, evaluation, representation analysis, intervention results, and theoretical material documented in the paper and appendix.