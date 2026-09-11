# RECURSIVE CODE WORLD MODELS: BUILDING COMPLEX WORLDS THROUGH RECURSIVE SCENE PROGRAMS

Zhiqi Li∗ Yuxuan Liao Bo Zhu Georgia Institute of Technology

![](images/d51329794a320c9949a511bc3f813819f063113a35fdc95f59fb1fb5aaee4aa2.jpg)  
Figure 1: For image-to-3D code world reconstruction, Recursive Code World Models use recursive global– local–global construction to progressively refine executable 3D scene programs, producing increasingly accurate local geometry and appearance while preserving whole-scene consistency.

## ABSTRACT

Code world models represent worlds as executable programs, but this representation alone does not determine how to construct a complex world. We introduce Recursive Code World Models (RCWM), a framework for reconstructing complex 3D worlds in code from a single reference image. RCWM couples a Recursive Scene Program (RSP) representation with a construction solver that recursively calls itself. An RSP represents the executable world as compositional scene code, while each solver call follows the same complete process: establish the whole, recursively reconstruct unresolved parts, and revisit the whole to refine their composition. This global–local–global recursion gives fine-scale structures their own perception-and-editing loops while preserving scene-wide geometry and relationships. Reference-aligned views propagate a shared camera projection across levels, while parent revisitation addresses boundaries, spatial relations, and shared errors that emerge after local refinement. A vision-language coding agent directly compares reference images with scene renders to guide refinement, recursive descent, and return. Across complex scenes, RCWM outperforms prior code-based imageto-scene reconstruction methods. Ablation studies further support the benefits of recursive construction and suggest that deeper calls can improve finer-scale reconstruction. RCWM provides a recursive construction principle for building complex executable worlds from visual evidence.

## 1 INTRODUCTION

Code offers an executable and compositional representation of a world, exposing its structure and state for direct inspection and manipulation rather than leaving them implicit in generated pixels (Chen et al., 2026). By expressing geometry, materials, and spatial relationships as program elements, scene code supports targeted editing, component reuse, and integration with physical simulation (Yin et al., 2026). Recent vision-language coding agents have begun to realize this potential by reconstructing scenes from images through program generation and render-based visual inspection (Yin et al., 2026; He et al., 2026; img2threejs contributors, 2026). Yet constructing a complex world requires more than an expressive representation: it requires resolving structures at multiple scales while maintaining the relationships among its parts. This raises a fundamental question: how should a coding model organize its computation when the world contains more structure than a single reconstruction trajectory can resolve?

Consider rebuilding a town from an image. Streets and terrain define its layout; buildings contain facades, and facades contain windows, signs, and ornaments. A scene-wide reconstruction establishes a shared spatial context, but repeated whole-scene reviews may still overlook these small structures. Giving them focused reconstruction tasks makes their details easier to inspect and refine, yet local edits are not independent: a sign may overlap a neighboring window incorrectly, or a refined bridge may no longer meet the riverbanks. These conflicts can emerge even from a coherent initial layout and may only become apparent when the refined parts are inspected together. The same tension arises at every scale, from buildings within a town to windows and signs within a facade.

Our central claim is that complex-world construction should be recursive. A part is not merely an asset to generate and insert. It is a subworld that deserves its own complete reconstruction process, with inherited context, focused visual evidence, and a return to the whole that contains it. The essential recurrence is

![](images/561268c742f406a73c834890f07d4ed237c35dd861c1367367e5fb5edb257c60.jpg)  
: the same solver at every level

(1)

The first whole establishes a shared spatial context for local construction. The second inspects and refines the assembled whole, correcting inconsistencies among its parts and their relationships. Crucially, the same complete cycle applies within a building, a facade, or a terrain region. This is not a fixed scene–object pipeline followed by a final cleanup, but a construction process that recursively calls itself: each subworld establishes its own whole, reconstructs its unresolved parts through the same solver, and refines their composition before returning to its parent.

We introduce Recursive Code World Models (RCWM) and their output representation, Recursive Scene Programs (RSPs). An RSP represents an executable world as compositional scene code organized over nested subworlds. RCWM constructs this representation through a complete recursive process. At each node, a vision-language coding agent establishes a whole, invokes the same solver on unresolved visual subproblems, and then inspects and refines their composition. Each child receives focused reference evidence and its own perception-and-editing loop, while inheriting the parent’s camera projection and spatial conventions. Parent revisitation addresses boundaries, spatial relationships, and shared causes of error that isolated local reviews may overlook. Together, these mechanisms allow local detail and cross-part consistency to be addressed at every level, with visual feedback guiding further descent where finer structure remains unresolved. Recursion determines which subproblems receive a complete solve, whereas parallelism only changes how compatible branches are scheduled. Our contributions are threefold:

• Recursive Scene Programs. We introduce an executable scene representation that organizes complex worlds as compositional subworld programs with explicit child references, editable structure, and shared dependencies.

• Recursive Code World Construction. We introduce a complete recursive construction process in which the same visual solver establishes each subworld, recursively reconstructs its unresolved parts, and revisits their composition after return, combining inherited observation geometry, visually guided recursive calls, and active parent-level refinement.

• Reference-driven reconstruction and evaluation. We instantiate the framework with a parameterized Three.js compiler and evaluate whole-scene and local reconstruction against image-to-sceneprogram baselines using the same base model. Ablation studies examine the roles of construction order, recursive depth, and parent revisitation under the same local visual access.

![](images/29eaa3457b0e78c80a67172d245e1b202ce21682bea5c3c6d4548092fec14c4b.jpg)  
Figure 2: Recursive Code World Models. Top: the recursive construction process. At level k, the solver establishes the current whole, calls the same solver on unresolved subworlds at level k + 1, and revisits and refines the whole after their results return. Each child can in turn make further calls at level k + 2 and beyond, producing a recursive call tree. Bottom: one recorded construction path on medieval-village: the scene calls a solver for the west village, which in turn calls a solver for the tall house. Each level receives a matched reference crop and camera view, returns its constructed subprogram to the parent, and is then revisited in the parent composition.

## 2 RECURSIVE CODE WORLD MODELS

## 2.1 EXECUTABLE RECURSIVE SCENE PROGRAMS

Given a single reference image I<sup>⋆</sup>, our goal is to reconstruct its depicted world as an executable, parameterized scene program P. We represent this output as a Recursive Scene Program (RSP), containing source procedures, editable parameters, component references, and the shared dependencies needed to construct the world. Let Exec denote program execution, S the resulting 3D scene, the rendering operation, and $\widehat { I }$ its output image. With a reference-view camera configuration κ, execution and rendering give

$$
S = \operatorname { E x e c } ( P ) , \qquad { \widehat { I } } = { \mathcal { R } } ( S ; \kappa ) .\tag{2}
$$

The program specifies geometry, materials, lighting, and component placement; κ specifies the camera pose, projection, and viewport dimensions. We seek a program whose rendered layout, local appearance, and spatial relationships agree with the visible evidence in $I ^ { \star }$ . The delivered code supports direct execution, parameter editing, and rendering from additional views.

Compositional program structure. The world program is organized into nested subworlds, each representing an object, an interacting assembly, or a continuous surface region. A node $u _ { i } ^ { ( k ) }$ denotes a subworld at depth k with program $P _ { i } ^ { ( k ) }$ , where i is unique across the hierarchy and the root is $u _ { 0 } ^ { ( 0 ) }$ . The set Children(i) contains its direct child identifiers, so $j \in \mathrm { C h i l d r e n } ( i )$ identifies a child $u _ { j } ^ { ( k + 1 ) }$ with program $P _ { j } ^ { ( k + 1 ) }$ . Each parent program contains local construction procedures, editable parameters, and references to these child programs, together with the rules for their placement and interfaces. We write $P _ { i } ^ { ( k ) } [ j ]$ for fetching a child program: it follows the reference labeled $j$ in the parent code and retrieves the corresponding child code, giving $P _ { i } ^ { ( k ) } [ j ] = P _ { j } ^ { ( k + 1 ) }$ . For selected children, let $Q _ { j }$ denote a candidate replacement for child program $P _ { j } ^ { ( k + 1 ) }$ . The operation Compose( $P _ { i } ^ { ( k ) } , \{ j \mapsto Q _ { j } \} )$ produces an updated parent program whose selected child references point to $Q _ { j } ,$ preserving the parent’s local code, placement rules, and remaining children. Shared generators and assets are accessed through code references, and spatial relationships can connect components across branches. The root program $P = P _ { 0 } ^ { ( 0 ) }$ forms the executable world, which can be executed in Three.js to construct the complete world.

## 2.2 RECURSIVE WORLD CONSTRUCTION

Algorithm 1 Recursive construction of a scene program   
Require: Reference image $I ^ { \star } ;$ total budget b; coding agent A   
Ensure: Executable program P and reference-view camera κ   
1: Initialize the root state $q _ { 0 } ^ { ( 0 ) }$ and identity context ${ \mathcal C } _ { 0 } ^ { ( 0 ) }$ ▷ §2.2.1   
2: $q _ { 0 } ^ { ( 0 ) + }  \mathcal { F } ( q _ { 0 } ^ { ( 0 ) } , \mathcal { C } _ { 0 } ^ { ( 0 ) } ; A )$   
3: return the program P and camera κ from $q _ { 0 } ^ { ( 0 ) + }$ ▷ §2.2.4   
4: function $\mathcal { F } ( q _ { i } ^ { ( k ) } , \mathcal { C } _ { i } ^ { ( k ) } ; A )$   
5: while budget remains in $q _ { i } ^ { ( k ) }$ do   
1. Establish the whole ▷ §2.2.2   
6: Compare the current contextual render with its reference using A.   
7: Refine the layout, shared structures, and component interfaces.   
2. Prepare and recursively construct the parts ▷ §2.2.3   
8: Select unresolved children ${ \mathcal { U } } _ { i } ^ { ( k ) }$ ; create or reopen their programs.   
9: Prepare child states $\boldsymbol q _ { j } ^ { ( k + 1 ) }$ and contexts $\mathcal { C } _ { j } ^ { ( k + \mathrm { \bar { 1 } } ) }$ from the updated parent snapshot, with   
code, matched views, and allocated budgets.   
10: if the budget in $q _ { i } ^ { ( k ) }$ is exhausted then   
11: return $q _ { i } ^ { ( k ) }$   
12: end if   
13: for all $\boldsymbol { j } \in \mathcal { U } _ { i } ^ { ( k ) }$ do   
14: $\check { q } _ { j } ^ { ( k + 1 ) + } \gets \mathcal { F } ( q _ { j } ^ { ( k + 1 ) } , \mathcal { C } _ { j } ^ { ( k + 1 ) } ; A )$ ▷ The same complete solver   
15: end for   
3. Compose, refine, and return ▷ §2.2.4   
16: Integrate returned child programs and update the parent’s budget.   
17: if the budget in $q _ { i } ^ { ( k ) }$ is exhausted then   
18: return $q _ { i } ^ { ( k ) }$   
19: end if   
20: Render the assembled whole and refine cross-part relationships.   
21: Review objects, relationships, and contact boundaries together.   
22: Update the refined program and remaining budget in $q _ { i } ^ { ( k ) }$   
23: if a complete review finds no new actionable discrepancy then   
24: return $q _ { i } ^ { ( k ) }$   
25: end if   
26: end while   
27: return $\mathbf { \overline { { { q } } } } _ { i } ^ { ( \mathbf { \overline { { { k } } } } ) }$   
28: end function

We construct the scene program with a recursive solver $\mathcal { F }$ driven by a frozen vision-language coding agent A. Each call follows three stages: establish the whole, prepare and recursively construct the parts, and compose, refine, and return. At depth k, the solver establishes the subworld’s layout, gives selected components complete solves at depth $k + 1$ , and returns to depth k to refine their composition. Every child follows this same complete process. Throughout construction, $P _ { i } ^ { ( k ) }$ denotes the current version of its component code, and visual feedback guides local edits, further child calls, and the return to the parent.

In Algorithm 1, b is the total inference budget and $q _ { i } ^ { ( k ) }$ contains the current component code, reference view, matched camera, and remaining budget. The inherited assembly function $\mathcal { C } _ { i } ^ { ( k ) }$ places candidate component code in the surrounding world using the parent program snapshot. The selected unresolved child identifiers form $\mathcal { U } _ { i } ^ { ( k ) } \subseteq \mathrm { C h i l d r e n } ( i )$ , and superscript + marks a returned version. The state and context are defined in Sections 2.2.1 and 2.2.2.

Algorithm 1 gives the complete workflow from initialization to the final program; the single instruction that realizes it at every node of our implementation is reproduced in Appendix A.1. Each call receives its local working state and inherited context as separate inputs:

$$
\begin{array} { r } { q _ { i } ^ { ( k ) + } = \mathcal { F } ( q _ { i } ^ { ( k ) } , \mathcal { C } _ { i } ^ { ( k ) } ; A ) . } \end{array}\tag{3}
$$

The following subsections define the state and expand the stages indicated in the algorithm.

## 2.2.1 CONSTRUCTION STATE AND INITIALIZATION

Line 1 of Algorithm 1 initializes the root. Each call maintains a local working state during construction, combining its candidate code with the reference evidence, camera, and resources used to revise it, which are constructed in Section 2.2.3.

Working state. Let $I _ { i } ^ { ( k ) }$ be the call’s reference view, $\kappa _ { i } ^ { ( k ) }$ its matched camera, and $b _ { i } ^ { ( k ) }$ its remaining inference budget, including descendant calls. The state is

$$
\begin{array} { r } { q _ { i } ^ { ( k ) } = \left( P _ { i } ^ { ( k ) } , I _ { i } ^ { ( k ) } , \kappa _ { i } ^ { ( k ) } , b _ { i } ^ { ( k ) } \right) . } \end{array}\tag{4}
$$

The program field holds the evolving component code; the other fields specify the evidence and resources for its construction. Repeated calls on a component retain its identifier and depth, with their individual visits recorded in the construction trace.

Root initialization. The procedure Initialize creates a seed program, establishes world-coordinate and unit conventions, and calibrates the reference camera through full-frame visual inspection. It produces

$$
\begin{array} { c } { { \left( P _ { 0 } ^ { \left( 0 \right) } , \kappa _ { 0 } ^ { \left( 0 \right) } , b _ { 0 } ^ { \left( 0 \right) } \right) = \mathrm { I n i t i a l i z e } ( I ^ { \star } , b ; A ) , } } \\ { { q _ { 0 } ^ { \left( 0 \right) } = \left( P _ { 0 } ^ { \left( 0 \right) } , I _ { 0 } ^ { \left( 0 \right) } = I ^ { \star } , \kappa _ { 0 } ^ { \left( 0 \right) } , b _ { 0 } ^ { \left( 0 \right) } \right) . } } \end{array}\tag{5}
$$

Here $b _ { 0 } ^ { ( 0 ) }$ is the budget remaining after initialization. Child calls inherit the root’s coordinate conventions and derive their cameras from its projection. The root controls camera revisions; each revision is followed by refreshed observations and affected child states before further descent. (see Section 2.2.3).

## 2.2.2 ESTABLISH THE WHOLE

Lines 6–7 of Algorithm 1 establish the current subworld’s layout and the interfaces among its parts through visual comparison. The agent renders the subworld within its inherited scene context, which provides the surrounding geometry, shared structures, and placement rules. Comparing this render with the reference guides revisions to the subworld and establishes a shared spatial arrangement for subsequent child work.

Observe and refine. Let $\mathcal { C } _ { i } ^ { ( k ) } [ \cdot ]$ denote the inherited program context. For candidate component code $Q ,$ , the expression ${ \mathcal { C } } _ { i } ^ { ( k ) } [ Q ]$ assembles a complete world program by placing $Q$ in the current component’s slot while preserving the surrounding scene code. The observation operator Observe executes this assembled program and pairs its render with the reference:

$$
\mathrm { O b s e r v e } ( q _ { i } ^ { ( k ) } , \mathcal { C } _ { i } ^ { ( k ) } ) = \left[ I _ { i } ^ { ( k ) } , \mathcal { R } \Bigl ( \mathrm { E x e c } \Bigl ( \mathcal { C } _ { i } ^ { ( k ) } [ P _ { i } ^ { ( k ) } ] \Bigr ) ; \kappa _ { i } ^ { ( k ) } \Bigr ) \right] .\tag{6}
$$

The agent directly examines this pair, edits $P _ { i } ^ { ( k ) }$ , and renders the updated program for another comparison. Each pass can contain several observation–edit–render iterations. A building establishes its facade arrangement before individual details are reconstructed; a terrain region establishes the surfaces and interfaces needed by local structures. Shared geometry is authored at a scope that contains its dependencies.

Context defined by the parent program. The inherited context is constructed from the parent program when the child call is prepared. Let $\bar { P } _ { i } ^ { ( k ) }$ denote the parent program snapshot after its whole-construction and child-preparation phase. For each child $j \in \mathrm { \bar { C h i l d r e n } } ( i )$ , the assembly function is defined recursively as

$$
\begin{array} { r l } & { ~ \mathcal { C } _ { 0 } ^ { ( 0 ) } [ Q ] : = Q , } \\ & { \mathcal { C } _ { j } ^ { ( k + 1 ) } [ Q ] : = \mathcal { C } _ { i } ^ { ( k ) } \left[ \mathrm { C o m p o s e } \left( \bar { P } _ { i } ^ { ( k ) } , \{ j \mapsto Q \} \right) \right] . } \end{array}\tag{7}
$$

The inner Compose replaces child $j$ in the parent snapshot with $Q ;$ the outer context places the updated parent in its inherited surroundings. At the root, the candidate program already describes the entire world. At deeper levels, the same rule assembles the candidate through the program snapshots along its ancestor path. Thus $\mathcal { C } _ { j } ^ { ( k + 1 ) }$ is fully determined by those snapshots and the child references.

During a call, the surrounding snapshots remain fixed while the candidate component is edited. These snapshots and the assembly function define the construction-time inspection environment. The component code, surrounding structures, and shared dependencies are retained through the world program’s references.

## 2.2.3 PREPARE AND RECURSIVELY CONSTRUCT THE PARTS

Lines 8–14 of Algorithm 1 select unresolved children $\mathcal { U } _ { i } ^ { ( k ) } \subseteq \mathrm { C h i l d r e n } ( i )$ and create or reopen their programs. Let $\mathcal { G } ^ { - }$ denote the preceding whole-construction pass together with child preparation, and let an overbar ¯ mark the updated parent state and its fields. This phase produces the parent state and the prepared child state–context pairs:

$$
\left( \bar { q } _ { i } ^ { ( k ) } , \{ ( q _ { j } ^ { ( k + 1 ) } , \mathcal { C } _ { j } ^ { ( k + 1 ) } ) \} _ { j \in \mathcal { U } _ { i } ^ { ( k ) } } \right) = \mathcal { G } ^ { - } \left( q _ { i } ^ { ( k ) } , \mathcal { C } _ { i } ^ { ( k ) } ; A \right) .\tag{8}
$$

Matched reference and camera crops. Each child receives a reference crop and a camera view of the corresponding scene region. Let $\mathbf { \bar { \boldsymbol { W } } } = ( x , y , w , h )$ denote a window with upper-left coordinate $( x , y )$ , width w, and height h, and let $s > 0$ be its magnification. For an input image I and output coordinates $( \alpha , \beta )$ , the image crop is

$$
\left[ \mathrm { C r o p } _ { W , s } ( I ) \right] ( \alpha , \beta ) = I \left( x + \frac { \alpha } { s } , y + \frac { \beta } { s } \right) ,\tag{9}
$$

with output size $s w \times s h$

For a perspective camera $\kappa = ( R , t , K , ( n _ { x } , n _ { y } ) )$ , where $R , i$ t specify its world-to-camera pose, K its intrinsic matrix, and $( n _ { x } , n _ { y } )$ ) its viewport size, define

$$
H _ { W , s } = \left[ \begin{array} { c c c } { { s } } & { { 0 } } & { { - s x } } \\ { { 0 } } & { { s } } & { { - s y } } \\ { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right] ,\tag{10}
$$

$$
\mathrm { C r o p C a m e r a } _ { W , s } ( \kappa ) = ( R , t , H _ { W , s } K , ( s w , s h ) ) .\tag{11}
$$

These paired operations preserve the camera pose and match the reference crop’s coordinates and magnification. Orthographic cameras use the corresponding projection subwindow. Windows can overlap to inspect objects, relationships, and contact boundaries (Figure 3).

![](images/17af293ecbace82ca53147df27a2530accf8220ddad5454bbb9fad44880e1cb7.jpg)  
Figure 3: Visual inspection for recursive world construction. (a) Reference and camera crops use matching coordinates and magnification, with the camera pose inherited from the parent. (b) Overlapping windows inspect object details, spatial relationships, and contact boundaries. Both panels are schematic.

Child inputs and recursive calls. For each selected child j, the agent chooses a window $W _ { j } ^ { ( k + 1 ) }$ magnification $s _ { j } ^ { ( k + 1 ) }$ , and budget $b _ { j } ^ { ( k + 1 ) }$ , reserving resources for parent revisitation. Its context ${ \mathcal { C } } _ { i } ^ { ( k + 1 ) }$ is derived from the updated parent program through Equation 7. The child starts from the referenced component code and matched visual evidence:

$$
\begin{array} { r l } & { P _ { j } ^ { ( k + 1 ) } = \bar { P } _ { i } ^ { ( k ) } [ j ] , } \\ & { I _ { j } ^ { ( k + 1 ) } = \mathrm { C r o p } _ { W _ { j } ^ { ( k + 1 ) } , s _ { j } ^ { ( k + 1 ) } } \left( I _ { i } ^ { ( k ) } \right) , } \\ & { \kappa _ { j } ^ { ( k + 1 ) } = \mathrm { C r o p C a m e r a } _ { W _ { j } ^ { ( k + 1 ) } , s _ { j } ^ { ( k + 1 ) } } \left( \bar { \kappa } _ { i } ^ { ( k ) } \right) , } \\ & { q _ { j } ^ { ( k + 1 ) } = \left( P _ { j } ^ { ( k + 1 ) } , I _ { j } ^ { ( k + 1 ) } , \kappa _ { j } ^ { ( k + 1 ) } , b _ { j } ^ { ( k + 1 ) } \right) . } \end{array}\tag{12}
$$

The parent then calls the same complete solver (Algorithm 1, line 14):

$$
\begin{array} { r } { q _ { j } ^ { ( k + 1 ) + } = \mathcal { F } ( q _ { j } ^ { ( k + 1 ) } , \mathcal { C } _ { j } ^ { ( k + 1 ) } ; A ) , \qquad j \in \mathcal { U } _ { i } ^ { ( k ) } . } \end{array}\tag{13}
$$

Each child establishes its own whole, recursively solves finer parts where needed, and refines their composition before returning. Visual evidence guides further descent, allowing the same construction process to continue at depth $k + 2$ and beyond.

## 2.2.4 COMPOSE, REFINE, AND RETURN

Lines 16–24 of Algorithm 1 combine the returned child programs and refine their relationships at the parent’s depth k.

Compose the returned programs. Let $P _ { j } ^ { ( k + 1 ) + }$ be the program returned by child j. A tilde marks the assembled parent before refinement, and $\widetilde { b } _ { i } ^ { ( k ) }$ is the remaining budget after deducting all child-call usage, including descendants, from $\bar { b } _ { i } ^ { ( k ) }$ . The assembled program and state are

$$
\begin{array} { r l } & { \widetilde { P } _ { i } ^ { ( k ) } = \mathrm { C o m p o s e } \Big ( \bar { P } _ { i } ^ { ( k ) } , \{ j \mapsto P _ { j } ^ { ( k + 1 ) + } \} _ { j \in \mathcal { U } _ { i } ^ { ( k ) } } \Big ) , } \\ & { \widetilde { q } _ { i } ^ { ( k ) } = \Big ( \widetilde { P } _ { i } ^ { ( k ) } , I _ { i } ^ { ( k ) } , \bar { \kappa } _ { i } ^ { ( k ) } , \widetilde { b } _ { i } ^ { ( k ) } \Big ) . } \end{array}\tag{14}
$$

Joint visual refinement. Let $\mathcal { G } ^ { + }$ denote the parent’s inspection-and-editing phase and $\sigma _ { i } ^ { ( k ) }$ ∈ CONTINUE, RETURN its visual review decision:

$$
\left( q _ { i } ^ { \left( k \right) + } , \sigma _ { i } ^ { \left( k \right) } \right) = \mathcal { G } ^ { + } \left( \tilde { q } _ { i } ^ { \left( k \right) } , \mathcal { C } _ { i } ^ { \left( k \right) } ; A \right) .\tag{15}
$$

Using $\mathrm { O b s e r v e } ( \widetilde { q } _ { i } ^ { ( k ) } , \mathcal { C } _ { i } ^ { ( k ) } )$ , the agent compares the assembled scene with its reference and revises geometry, placement, and shared structures. For example, a refined sign may need repositioning to avoid covering a neighboring window. Comparisons are refreshed after edits.

The review checks both whether reference structures are correctly represented and whether generated elements have reference support or an explicit completion assumption. Shared structures are repaired by their owning ancestor, and camera corrections are handled at the root. Affected child contexts and views are refreshed before further calls.

Continue or return. A complete review with no new actionable discrepancy produces $\sigma _ { i } ^ { ( k ) } = $ RETURN. Otherwise, another cycle starts from $q _ { i } ^ { ( k ) } \gets q _ { i } ^ { ( k ) + }$ while budget remains. Budget exhaustion returns the latest executable state. The final root state $q _ { 0 } ^ { ( 0 ) + }$ supplies the outputs $P = P _ { 0 } ^ { ( 0 ) + }$ and $\kappa = \kappa _ { 0 } ^ { ( 0 ) + }$ (Algorithm 1, line 3).

## 3 RELATED WORK

Code-based world representation and generation. Code represents worlds as executable, editable, and compositional programs rather than only rendered appearances. By exposing geometry, materials, and spatial relationships as explicit program elements, it supports targeted editing, component reuse, and integration with downstream simulation (Yin et al., 2026). Code World Model further couples code-maintained world state with visually generated observations, separating executable world evolution from visual realization (Chen et al., 2026). Infinigen demonstrates the richness of procedural world construction through compositional geometry and materials (Raistrick et al., 2023). For text-driven generation, SceneCraft produces Blender programs through planning, visual refinement, and reusable code (Hu et al., 2024), while WorldClaw constructs open worlds through global planning, terrain construction, local content generation, and render-based refinement (Guo et al., 2026). These approaches demonstrate how code can support the construction of structured worlds that remain accessible beyond rendering. Unlike open-ended generation from procedural or textual specifications, our setting seeks an executable scene program whose rendering faithfully matches a particular reference image.

Image-to-code world reconstruction. Reconstructing a world as code from an image requires more than semantic plausibility: the program must reproduce specific visible shapes, spatial arrangements, occlusions, and fine details under a consistent camera, despite ambiguity in the underlying 3D geometry (Yin et al., 2026; He et al., 2026). Recent vision-language coding agents address this problem through execution-grounded visual feedback. VIGA reconstructs and edits scenes through interleaved program generation, execution, and visual verification (Yin et al., 2026). Thinking in Blender combines scene-graph initialization with factor-specific reconstruction stages for geometry, materials, composition, and lighting (He et al., 2026). The img2threejs project provides a reference-driven, code-only workflow for procedural object reconstruction in Three.js (img2threejs contributors, 2026). RCWM constructs Recursive Scene Programs through a recursive solver: unresolved subworlds receive the same complete solve, and each return triggers renewed inspection and refinement of the assembled parent scene. The recurrence therefore encompasses the entire construction process, rather than only scene decomposition, repeated program editing, or a fixed sequence of reconstruction factors.

Recursive inference and concurrent work. Recursive Language Models organize inference through programmatic examination of external context and recursive model calls (Zhang et al., 2025). In our setting, the external environment is an evolving executable world: child calls construct subprograms whose return changes the scene that their parent must inspect. Concurrent work, FuncRoom-Agent, introduces a recursive DSL and distills construction traces into an expert specialized for indoor scene generation (Feng et al., 2026). Its representation, construction stages, and training objectives are tailored to room structure, functional furniture arrangements, and nested indoor objects. In contrast, RCWM formulates recursion as a general world-construction principle rather than a room-specific generation procedure. A frozen coding model applies the same complete visual solver to objects, interacting assemblies, and continuous terrain, spanning architectural details, buildings, towns, and terrain-rich worlds without prescribing a domain-specific semantic hierarchy.

Recursion remains an active inference-time process: each subworld can invoke further solves and refine its assembled whole before returning to its parent.

## 4 EXPERIMENTS

We evaluate how faithfully RCWM reconstructs complex worlds as executable scene programs, considering scene-wide appearance and fine-scale detail. We first compare its reconstructions with image-to-scene-program baselines (Section 4.2), then examine the roles of initial whole-scene construction, recursive child solves, and parent revisitation through ablations (Section 4.3). Additional views and recorded call trees illustrate the generated geometry and the construction process.

## 4.1 SETUP

Reference images. We use five whole-scene references and five local crops. The city-full reference and four crops—school-block, police-corner, park-lake, and shop-row—come from the example composition of the CC0 “Isometric city” sprite pack by JanaChumi (JanaChumi, 2017). The other references are WorldClaw demonstration renders (Guo et al., 2026): island-harbor (Fig. 4), medieval village (Fig. 9), snow-village (Fig. 10), japan-island (Fig. 12), and the valley-village crop from Fig. 15. These supply reference images only; reconstruction uses the images without their source prompts, terrain, or assets. All methods receive the same raw images at the same resolution. View tabs, biome badges, and location pins remain in the inputs and evaluation references.

Methods and execution. We compare RCWM with SEIG (He et al., 2026), VIGA (Yin et al., 2026), and img2threejs (img2threejs contributors, 2026), using the same base model and reasoningeffort setting: every method is driven by OpenAI’s gpt-6-astra through the Codex command-line agent at high reasoning effort, and RCWM runs each node of the recursion as one Codex session. RCWM uses the instruction in Appendix A.1. SEIG is our reproduction from the paper. VIGA and img2threejs use their released pipelines with the shared model endpoint.

Evaluation. We report PSNR, SSIM, edge $F _ { 1 }$ using Canny edges with a three-pixel tolerance, LPIPS with the AlexNet backbone (Zhang et al., 2018), and cosine similarity of CLIP ViT-B/32 image embeddings. Metrics are computed against the unmodified reference at its native resolution; best values per scene are shown in bold. Visual comparisons show full views and magnified details at matched scales. The ablation additionally reports mean SSIM over eight fixed 2 detail windows. Spatial relationships and program editability are inspected qualitatively in the renders and source programs. The solver uses visual feedback during construction; these metrics are computed for evaluation.

## 4.2 RECONSTRUCTION RESULTS

Reference-view fidelity. Table 1 compares all four methods on the whole scenes and local crops. In these runs, RCWM obtains the highest PSNR and lowest LPIPS on every reference. It also has the highest SSIM on nine references; on valley-village, VIGA obtains 0.41 and RCWM obtains 0.38. The PSNR and LPIPS improvements span both the whole-scene and local-crop groups.

Whole scenes and local detail. Figures 4 and 5 show all methods on city-full and medieval-village. Full views show the overall reconstruction, followed by six affine-aligned magnified windows selected to highlight differences between methods. Figures 6–9 pair the remaining eight references with our results and four magnified windows per case. Together, these views allow scene layout and smaller structures, including individual buildings, groves, and shorelines, to be inspected at their respective scales.

Additional views and construction traces. To demonstrate the 3D structure of the reconstructed worlds, Figure 10 renders the delivered programs from rotated, elevated, and close-up cameras without further optimization. These views reveal the geometry and spatial arrangement encoded in the same scene program beyond the reference projection. Figure 11 shows the recursive call trees recorded during construction, including nested calls to deeper subworlds and returns to their parents. These traces demonstrate the recursive execution of the same solver across multiple levels.

Table 1: Reconstruction metrics for every method with a completed run; best per scene in bold.
<table><tr><td>Scene</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>Edge F1 ↑</td><td>LPIPS↓</td><td>CLIP↑</td></tr><tr><td colspan="7">Whole scenes</td></tr><tr><td>city-full</td><td>Ours</td><td>18.2</td><td>0.66</td><td>0.94</td><td>0.158</td><td>0.95</td></tr><tr><td></td><td>SEIG</td><td>14.5</td><td>0.51</td><td>0.83</td><td>0.339</td><td>0.90</td></tr><tr><td></td><td>VIGA</td><td>9.5</td><td>0.48</td><td>0.58</td><td>0.670</td><td>0.83</td></tr><tr><td></td><td>img2threejs</td><td>15.5</td><td>0.55</td><td>0.91</td><td>0.243</td><td>0.93</td></tr><tr><td>snow-village</td><td>Ours</td><td>17.9</td><td>0.73</td><td>0.85</td><td>0.237</td><td>0.83</td></tr><tr><td></td><td>SEIG</td><td>13.4</td><td>0.64</td><td>0.37</td><td>0.520</td><td>0.77</td></tr><tr><td></td><td>VIGA</td><td>13.5</td><td>0.61</td><td>0.40</td><td>0.541</td><td>0.75</td></tr><tr><td></td><td>img2threejs</td><td>15.4</td><td>0.64</td><td>0.65</td><td>0.351</td><td>0.81</td></tr><tr><td>island-harbor</td><td>Ours</td><td>17.4</td><td>0.74</td><td>0.90</td><td>0.186</td><td>0.93</td></tr><tr><td></td><td>SEIG</td><td>11.7</td><td>0.63</td><td>0.57</td><td>0.516</td><td>0.80</td></tr><tr><td></td><td>VIGA</td><td>10.6</td><td>0.65</td><td>0.48</td><td>0.526</td><td>0.83</td></tr><tr><td></td><td>img2threejs</td><td>15.0</td><td>0.68</td><td>0.81</td><td>0.272</td><td>0.91</td></tr><tr><td>medieval-village</td><td>Ours</td><td>18.7</td><td>0.71</td><td>0.89</td><td>0.200</td><td>0.93</td></tr><tr><td></td><td>SEIG</td><td>10.9</td><td>0.60</td><td>0.50</td><td>0.541</td><td>0.85</td></tr><tr><td></td><td>VIGA</td><td>10.5</td><td>0.58</td><td>0.37</td><td>0.601</td><td>0.83</td></tr><tr><td></td><td>img2threejs</td><td>15.4</td><td>0.62</td><td>0.77</td><td>0.325</td><td>0.89</td></tr><tr><td>japan-island</td><td>Ours</td><td>19.3</td><td>0.73</td><td>0.87</td><td>0.201</td><td>0.96</td></tr><tr><td></td><td>SEIG</td><td>13.3</td><td>0.64</td><td>0.47</td><td>0.513</td><td>0.93</td></tr><tr><td></td><td>VIGA</td><td>12.9</td><td>0.65</td><td>0.43</td><td>0.534</td><td>0.80</td></tr><tr><td></td><td>img2threejs</td><td>16.6</td><td>0.66</td><td>0.76</td><td>0.302</td><td>0.95</td></tr><tr><td colspan="7">Local crops</td></tr><tr><td>school-block</td><td>Ours</td><td>16.4</td><td>0.56</td><td>0.97</td><td>0.171</td><td>0.94</td></tr><tr><td></td><td>SEIG</td><td>11.8</td><td>0.23</td><td>0.80</td><td>0.448</td><td>0.88</td></tr><tr><td></td><td>VIGA</td><td>11.2</td><td>0.19</td><td>0.77</td><td>0.587</td><td>0.85</td></tr><tr><td></td><td>img2threejs</td><td>12.8</td><td>0.26</td><td>0.88</td><td>0.310</td><td>0.91</td></tr><tr><td>police-corner</td><td>Ours</td><td>16.6</td><td>0.44</td><td>0.97</td><td>0.159</td><td>0.91</td></tr><tr><td></td><td>SEIG</td><td>12.7</td><td>0.24</td><td>0.75</td><td>0.415</td><td>0.90</td></tr><tr><td></td><td>VIGA</td><td>12.4</td><td>0.27</td><td>0.82</td><td>0.473</td><td>0.91</td></tr><tr><td></td><td>img2threejs</td><td>13.7</td><td>0.30</td><td>0.86</td><td>0.285</td><td>0.90</td></tr><tr><td>park-lake</td><td>Ours</td><td>23.3</td><td>0.83</td><td>0.99</td><td>0.075</td><td>0.95</td></tr><tr><td></td><td>SEIG</td><td>12.3</td><td>0.48</td><td>0.61</td><td>0.490</td><td>0.90</td></tr><tr><td></td><td>VIGA</td><td>15.1</td><td>0.54</td><td>0.74</td><td>0.330</td><td>0.92</td></tr><tr><td></td><td>img2threejs</td><td>18.4</td><td>0.66</td><td>0.92</td><td>0.169</td><td>0.93</td></tr><tr><td>shop-row</td><td>Ours</td><td>17.4</td><td>0.65</td><td>0.96</td><td>0.120</td><td>0.96</td></tr><tr><td></td><td>SEIG</td><td>12.5</td><td>0.50</td><td>0.79</td><td>0.400</td><td>0.89</td></tr><tr><td></td><td>VIGA</td><td>15.2</td><td>0.54</td><td>0.91</td><td>0.249</td><td>0.91</td></tr><tr><td></td><td>img2threejs</td><td>16.9</td><td>0.59</td><td>0.94</td><td>0.145</td><td>0.95</td></tr><tr><td>valley-village</td><td>Ours</td><td>14.1</td><td>0.38</td><td>0.48</td><td>0.583</td><td>0.87</td></tr><tr><td></td><td>SEIG</td><td>9.7</td><td>0.39</td><td>0.22</td><td>0.742</td><td>0.64</td></tr><tr><td></td><td>VIGA</td><td>10.2</td><td>0.41</td><td>0.20</td><td>0.743</td><td>0.68</td></tr><tr><td></td><td>img2threejs</td><td>11.3</td><td>0.34</td><td>0.41</td><td>0.726</td><td>0.70</td></tr></table>

## 4.3 ABLATION STUDY

Construction organization. We compare five reconstruction workflows that vary the order of wholescene and local construction, refinement after composition, and the permitted depth of recursive calls. Flat + zoom reconstructs the entire scene within one editing process, using crop and zoom tools to inspect local details. Local  global first reconstructs components independently and then assembles them into the whole scene. Global local first establishes the scene layout, then refines individual components, and finishes by assembling their results without further editing of the combined scene. Recursive,fixed 2-level first establishes the whole, gives selected components their own solver calls, and then renders and refines the assembled scene. This cycle can repeat, with calls restricted to the root and its direct children; each child completes its work locally. Recursive, free depth uses the same solver and additionally allows each child to call it on its own unresolved parts. These deeper calls follow the same process: establish their whole, construct their parts, and refine the composition before returning.

![](images/64d5857e2aadd955eeafb66abfea2978641bdaa6afa610e16d75d77298f8af65.jpg)  
Figure 4: Whole-city case: full views (top) and six affine-aligned magnified windows from medium to small (below), all methods on the same base model and raw input. Reference: example composition of the CC0 “Isometric city” sprite pack (JanaChumi, 2017).

All variants use the same base model, scene representation, modeling tools, reference resolution, and crop access, with variant-specific instructions executed by the same runner. The two recursive variants use the same solver and differ in their depth restriction.

Results. Table 2 compares the five variants on school-block and medieval-village, with one run per variant and scene under the shared ablation setting. On school-block, the free-depth run uses only two levels—the root and its direct children—and achieves scores similar to the fixed two-level variant. On medieval-village, it expands to three levels with 33 nodes and achieves higher fidelity: compared with the fixed two-level variant, whole-frame PSNR increases from 16.8 to 19.0 and local SSIM from 0.52 to 0.60. It also outperforms the other three workflows on both measures. Together, these results suggest that deeper recursive solves can improve both local detail and whole-scene fidelity, with the gains varying across reconstruction tasks.

## 5 CONCLUSION AND LIMITATIONS

We introduced Recursive Code World Models and Recursive Scene Programs for reconstructing complex, executable 3D worlds from a single reference image. The framework applies the same complete construction process to each selected subworld: establish the whole, recursively reconstruct unresolved parts, and revisit and refine their composition. Reference-aligned camera views support detailed local inspection, while parent revisitation addresses the spatial relationships among the returned components. RCWM therefore combines the compositional RSP representation with a recursive construction process that addresses local detail and whole-scene consistency at every scale.

![](images/aab2b3ece772f373648ce3840a75de971fe259ecab8fb4e62fc0035201e716c8.jpg)  
Figure 5: Medieval-village case: full views (top) and six affine-aligned magnified windows from medium to small (below), all methods on the same base model and raw input. Reference: WorldClaw Fig. 9 (Guo et al., 2026).

Several limitations remain. Single-view input leaves hidden geometry and absolute scale ambiguous, and visual inspection can overlook errors. Repeated refinement and interactions among components can require substantial computation, and convergence is not guaranteed. Our evaluation covers a limited set of scenes, with single runs on one base model and ablations on two scenes. Larger benchmarks, repeated runs, and human evaluation of spatial relationships would provide a stronger assessment of reconstruction quality and reliability.

![](images/6a9d6df524f869534132be49b39030bf7a0d80f321253ef62f6580362e5d4617.jpg)  
Figure 6: Reference and our result, full view on top and four magnified windows side by side: snow-village (left; WorldClaw Fig. 10) and island-harbor (right; WorldClaw Fig. 4) (Guo et al., 2026).

Table 2: Ablation study on two scenes, single runs on the same base model and scene compiler. Global fidelity is whole-frame PSNR/SSIM/LPIPS against the reference; local fidelity is the mean SSIM over eight fixed 2 detail windows. Best per scene in bold.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>Local SSIM↑</td><td>LPIPS↓</td></tr><tr><td>school-block</td><td></td><td></td><td></td><td></td></tr><tr><td>Flat + zoom</td><td>13.3</td><td>0.28</td><td>0.27</td><td>0.294</td></tr><tr><td>Local → global</td><td>16.3</td><td>0.53</td><td>0.52</td><td>0.179</td></tr><tr><td>Global → local</td><td>13.4</td><td>0.28</td><td>0.28</td><td>0.288</td></tr><tr><td>Recursive, fixed 2-level</td><td>16.8</td><td>0.55</td><td>0.57</td><td>0.166</td></tr><tr><td>Recursive, free depth</td><td>16.5</td><td>0.52</td><td>0.52</td><td>0.169</td></tr><tr><td>medieval-village</td><td></td><td></td><td></td><td></td></tr><tr><td>Flat + zoom</td><td>16.4</td><td>0.66</td><td>0.50</td><td>0.261</td></tr><tr><td>Local → global</td><td>14.3</td><td>0.64</td><td>0.52</td><td>0.402</td></tr><tr><td>Global → local</td><td>14.9</td><td>0.65</td><td>0.52</td><td>0.377</td></tr><tr><td>Recursive, fixed 2-level</td><td>16.8</td><td>0.67</td><td>0.52</td><td>0.252</td></tr><tr><td>Recursive, free depth</td><td>19.0</td><td>0.73</td><td>0.60</td><td>0.187</td></tr></table>

![](images/3b36fd74cce2d307221970042a955c2d63402225672adc9a0dbc21293f80a380.jpg)  
Figure 7: Reference and our result, full view on top and four magnified windows side by side: japan-island (left; WorldClaw Fig. 12) and valley-village (right; a crop of WorldClaw Fig. 15) (Guo et al., 2026).

![](images/81cf264c17133311381743f5a90a54bc9a1c2ced73eac52be0657216e61bb3a9.jpg)  
Figure 8: Reference and our result, full view on top and four magnified windows side by side: schoolblock (left) and police-corner (right), crops of the “Isometric city” sprite-pack example (JanaChumi, 2017).

![](images/138722493ef17da2778f8b117d29f33d024a112aa5f70f56bc6701a6abaab8c0.jpg)

![](images/c33c29bfbaaa640cc747a800b8f4bf1334a008a2f27bafa49760bdd7d7b00f16.jpg)  
Figure 9: Reference and our result, full view on top and four magnified windows side by side: park-lake (left) and shop-row (right), crops of the “Isometric city” sprite-pack example (JanaChumi, 2017).

![](images/bc60b0fe7cb1d88dbd93a35d3db8e674d2b7d5ea643faad8ac371940765d131a.jpg)  
Figure 10: Additional views of the delivered scene programs. Each row renders one program from the reference camera, two azimuth rotations of 35◦, a raised orbit, and two close-ups, through one generic camera harness and without any re-optimization.Crop-based scenes only contain content visible in their reference, so unseen regions are absent when viewed from behind.

![](images/780360beb08ae8237223391e9ca5ab17ac4cc1e7c2ecb97ce1986f2a7358b7db.jpg)  
Figure 11: Solver-call trees reconstructed from runner-logged call and return events for four representative cases. The logged internal nodes follow the complete whole–parts–whole cycle. Trees for all ten scenes are released with the code.

## REFERENCES

Yiwen Chen, Guosheng Lin, and Chi Zhang. Code world model: Coding agent as world brain, 2026. URL https://arxiv.org/abs/2608.25927.

Hao Feng, Zhi Zuo, MingJian Liang, Jingyu Hu, Xiaowei Hu, Liupengfei Wu, Dian Zhang, Guoxin Fang, and Zhengzhe Liu. FuncRoom-Agent: Sequential feed-forward 3D functional indoor scene generation, 2026. URL https://arxiv.org/abs/2608.29519.

Chunchao Guo, Jinpeng Li, Yang Li, and Zilong Huang. WorldClaw: Agentic 3D open-world generation at scale, 2026. URL https://arxiv.org/abs/2608.05248.

Guangzhao He, Rundong Luo, Wei-Chiu Ma, and Hadar Averbuch-Elor. Thinking in Blender: Staged executable inverse graphics with vision-language models, 2026. URL https://arxiv.org/ abs/2606.02580.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A. Ross, Cordelia Schmid, and Alireza Fathi. SceneCraft: An LLM agent for synthesizing 3D scene as Blender code, 2024. URL https://arxiv.org/abs/2403.01248.

img2threejs contributors. img2threejs. GitHub repository, 2026. URL https://github.com/ img2threejs/img2threejs. Accessed September 8, 2026.

JanaChumi. Isometric city. OpenGameArt.org asset pack, CC0 1.0, 2017. URL https:// opengameart.org/content/isometric-city-0. Published December 9, 2017; pack contains roads, vegetation, a fountain, a lake, civic buildings, other buildings, houses, and a shop.

Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, and Jia Deng. Infinite photorealistic worlds using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12630–12641, June 2023. URL https://openaccess.thecvf.com/ content/CVPR2023/html/Raistrick\_Infinite\_Photorealistic\_Worlds\_ Using\_Procedural\_Generation\_CVPR\_2023\_paper.html.

Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning, 2026. URL https://arxiv.org/abs/2601.11109. Version 3, revised April 6, 2026.

Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models, 2025. URL https: //arxiv.org/abs/2512.24601. Version 3, revised May 11, 2026.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, June 2018. URL https://openaccess.thecvf.com/content\_cvpr\_2018/html/ Zhang\_The\_Unreasonable\_Effectiveness\_CVPR\_2018\_paper.html.

## A REPRODUCIBILITY DETAILS

This appendix provides the solver instruction and the execution settings used in the reported experiments.

Execution settings. Each solver call receives a reference crop, a matched camera specification, a task brief, and the inherited parent snapshot. When the agent requests children, the runner starts their solver calls and resumes the parent after their components are returned. The runner allows five levels in total. At this limit, the agent completes the current component through local refinement. The fixed two-level ablation sets the maximum two level.

## A.1 THE SOLVER INSTRUCTION

The instruction below is used at every node in the main reconstruction experiments. The node name, depth, and directory paths are filled in for each call. Its inputs are target.png, view.json, brief.md, and a read-only manifest.json describing the inherited reference, camera, and parent snapshot.

## Solver instruction (released prompt, verbatim)

solve(node) — the one recursive solver, at depth d. This is an approved work order. Make reversible choices on your own recommendation; do not stop for approval. Chain directory: chain; your node directory: chain/fractal/node/. Materials: target.png (this level’s target image) / view.json (the viewport) / brief.md; manifest.json records the reference, camera and parent-snapshot versions you inherited (read-only). Camera contract: chain/camera-contract.json (read-only; if it does not exist and this is the root, solve and calibrate one first and verify the full-frame overlay by eye before locking it).

What you do at this level (one loop, identical at every level). (1) Whole: render this level’s viewport and put it side by side with target.png at matched magnification; look for the conspicuous residuals by eye; repeat as needed. Also take one or two rotated views — any flatness a side view exposes (paper walls, billboard trees, parts without thickness) is a real residual and goes on the repair list right away. (2) Parts: repair here what can be repaired here. A sub-problem that deserves its own solve (one sub-object or sub-assembly that patching at this level cannot fix and that needs its own target image and loop) is not to be forced through. The test: take each part of the side-by-side where you can still see a difference and magnify it on its own. If what you see is a group of objects and the layout relations between them (arrangement, spacing, orientation, attachment) that this level has not yet produced, or a single thing whose internal structure is complex enough to need its own loop, it is a sub-problem: descend. If it is only a simple single thing or scattered small differences, finish it at this level. A level usually cuts two to four children; most scenes close in two or three levels. For a sub-problem, prepare its materials under chain/fractal/<child>/ (target.png cut from your target and magnified / view.json / brief.md), list the child in children.json in your directory (a JSON array, e.g. ["part-a","part-b"]), then end the session — the runner starts this same solver for every child and wakes you with the results when they are done. (3) Whole again (after being woken): the children’s part.json files are in place; integrate them into your level, go back to the side-by-side of the whole at this level and settle continuity and relations; if more descent is needed, update children.json and end again, otherwise finish. (4) Finish: write chain/fractal/node/part.json (this level’s final component, with child references) + account.md (this level’s round-by-round account). part.json on disk means this level is done.

Completion (at every level). First identify: what this is, in what style, and what a complete instance of its kind looks like — imagine the whole thing in your mind, then calibrate that mental image against the visible evidence in the reference (two or three sentences in account.md). Then: what is visible, match to the reference pixel by pixel; what is not visible (back faces, occluded parts, boundary continuations), complete from the calibrated mental image with world knowledge in the same style — a house has four walls and a full roof, a tree crown goes all the way round, a road leads somewhere, terrain continues to the boundary. Self-check: render the scene from the four compass directions and look; stage-set feel and flatness (empty backs, dead-end roads, floating objects, paper parts) must be gone before completion counts; re-verify the visible region against the reference in the same window and do not let the completion pull it away.

Discipline: eyes decide, metrics are only recorded; free-form boxes; matched magnification; no git; a child’s write scope is its own fractal/<child>/. Everything else about the craft is yours to decide. Begin.

The model retains discretion over geometry, materials, decomposition, and repair. Resource limits, schema checks, and evidence recording are automatic runtime responsibilities. The instruction neither authorizes external actions nor uses a generated appearance score as a success gate.

## B DESIGN PHILOSOPHY

Complex-world reconstruction requires both detailed work on individual components and repeated checks of how those components fit together. The design in Section 2 organizes these tasks around a complete reconstruction process that can be applied at every scale.

A complete solve for each subworld. A subworld has its own reference view, editable code, and reconstruction loop. A facade, for example, needs an arrangement of windows and signs as well as a coherent placement within its building. Giving the facade a complete solve allows the agent to establish that arrangement, reconstruct its details through further calls, and inspect the assembled result. The same procedure applies to a building, an interacting assembly, or a terrain region. The agent chooses the decomposition from the visible structure and the remaining work. Examples in the instruction illustrate modeling choices; the recursive procedure remains applicable to other components and scene types.

A shared projection for local inspection. A child needs a target that preserves its relationship to the surrounding scene. We therefore pair each reference crop with the corresponding camera viewport, inheriting the parent’s camera pose and applying the same crop coordinates and magnification. This gives local comparisons a consistent frame for judging shape, scale, placement, and occlusion. The crop focuses inspection on evidence already present in the input. When the root camera or a shared scene dependency changes, affected child views are refreshed before further work.

Trusting visual judgment. We rely on the model’s visual understanding and reasoning to guide reconstruction. The agent examines the reference and current render at matched magnification, identifies discrepancies in shape, placement, and spatial relationships, and decides how to address them. A missing window may call for a local edit; a complex facade may need its own complete solve. After edits and child returns, the agent looks again to assess the result and determine the next step. The recursive structure supports this judgment by giving the model focused views of unresolved details and repeated opportunities to inspect how the parts fit together. Numerical appearance metrics are recorded for evaluation; the model’s visual comparisons guide reconstruction and stopping.

Completing unobserved structure. A single image leaves hidden surfaces and out-of-frame continuations ambiguous. Our completion instruction asks the agent to use its knowledge of the depicted objects and style when constructing such geometry. These additions express assumptions about the unobserved scene. Rotated views help inspect thickness, support, and continuity, and the resulting program is checked again from the reference view to identify changes in visible silhouettes, occlusions, or shading. We assess visible fidelity against the reference and use the other views to examine the geometry produced by these assumptions.

Refining relationships after composition. Local reviews cover the structures and relationships they actually inspect. A review of building shape and facade detail can leave the entrance’s orientation toward a shared street unexamined. Changes made by separate child calls can also affect each other: two components edited against earlier surroundings may overlap after composition. The parent therefore renders the assembled result and revisits placement, contact, occlusion, and shared geometry. It can edit these relationships directly or reopen affected child tasks. Shared errors are repaired at the ancestor that controls their cause. Compatible children may run concurrently, followed by this same joint refinement step. The purpose of returning is to evaluate and improve the composition produced by the child calls.

Code as the world representation. We represent the world as executable code that describes how its geometry, materials, and spatial arrangement are constructed. The code exposes both the procedures that generate each component and the parameters that control its properties. This gives the agent concrete ways to act on visual observations: change a dimension, revise a modeling procedure, or adjust how components are placed together. It also supports recursive construction. Each child returns a subprogram that the parent can compose into the world and continue to edit. For example, a parent can reposition a reconstructed building while retaining the code that generates its windows and roof. The completed world preserves these procedures and component references, supporting further editing, reuse, and rendering from new viewpoints.