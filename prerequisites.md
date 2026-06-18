# Conceptual Readiness Checklist
## Phase 0.5 → Phase 1 Transition
**Project:** Hardware-aware GNN cost model for TPUGraphs
**Status:** Pre-PI meeting, pre-Phase 1 implementation

---

# Category 1 — The Dataset and Compiler

---

## 1.1 What TPUGraphs Actually Is

**Rigorous explanation:**
TPUGraphs is a performance prediction dataset released by Google containing pairs of (computation graph, compiler configuration, execution time). Each computation graph represents one ML program — a full training step or inference pass — expressed as an XLA HLO (High Level Operations) graph. Each node in the HLO graph is a tensor operation. Each edge is a data dependency. The dataset has two collections: layout (31 million graph-config pairs, avg 7,700 nodes per graph) and tile (13 million pairs, avg 40 nodes per kernel subgraph). You are working on layout:xla:random — the hardest split, where configurations are sampled randomly across the full layout search space rather than near the compiler default.

**Why it matters:**
The random split is harder precisely because the configurations are diverse — the model cannot cheat by learning "configurations near the default are probably good." It must actually learn what makes a configuration good from graph structure. This is what makes it a meaningful research problem rather than a pattern-matching exercise.

**Probing questions:**
- "Why is Layout:XLA:Random harder than Layout:XLA:Default, given that both use the same graphs? What does the random search space reveal about the model's ability to generalize?"
- "The paper reports Kendall's Tau = 0.34 as the baseline for Layout:XLA:Random. Why is this number so much lower than the NLP collections, and what does that tell you about the structural diversity of XLA graphs versus transformer graphs?"

---

## 1.2 What XLA Does and What a Layout Configuration Is

**Rigorous explanation:**
XLA (Accelerated Linear Algebra) is Google's domain-specific compiler for ML programs. It takes programs expressed as HLO computation graphs and applies a sequence of optimization passes before generating executable code for a target hardware device. A layout configuration controls how the dimensions of each tensor are physically arranged in memory — specifically, the ordering of minor-to-major dimensions for each tensor at each operation node. For a 4D tensor with dimensions [batch, height, width, channels], a layout specifies whether channels are stored as the innermost (fastest-varying) or outermost (slowest-varying) dimension. XLA inserts explicit copy (transpose) operations at any edge where adjacent operations expect different layouts, creating memory movement overhead. The autotuner searches for the layout assignment that minimizes total execution time including these copy costs.

**Why it matters:**
Layout decisions determine data locality, memory access patterns, and whether tensor operations can exploit the TPU's matrix multiply unit efficiently. A bad layout forces unnecessary transposes and breaks the pipeline. Your cost model predicts which layout configurations minimize these costs without requiring exhaustive hardware evaluation.

**Probing questions:**
- "If a layout configuration changes the dimension ordering of a tensor at one node, what are the downstream consequences for adjacent nodes in the HLO graph, and how does XLA handle the mismatch?"
- "Why is layout optimization a graph-level problem rather than a node-level problem? What property of the HLO graph makes local layout decisions insufficient?"

---

## 1.3 Early-Join Configuration Scattering

**Rigorous explanation:**
Each graph in the TPUGraphs dataset has multiple compiler configurations. A configuration assigns layout choices to a subset of nodes — the configurable nodes, identified by node_config_ids. The 18-dimensional configuration feature vector encodes these layout choices. The early-join approach scatters these 18-dim config features into a per-node zero matrix using node_config_ids, then concatenates with the 140-dim base node features, producing a 158-dim node feature vector. The alternative, late-join, would encode configs separately and combine them only at the graph readout stage. Early-join wins empirically because it allows the GNN's message passing to operate on configuration-aware node features — the 158-dim representation propagates configuration information through the graph topology during forward passes, rather than ignoring it until the final pooling step.

**Why it matters:**
The early-join is not just a data engineering choice — it is an inductive bias. It assumes that the effect of a configuration choice at one node propagates through the computational neighborhood of that node in a way that affects execution cost. This is physically motivated: a layout choice at a matrix multiply node affects the cost of adjacent nodes that must match or transpose to interface with it.

**Probing questions:**
- "Why does early-join outperform late-join empirically, and can you give a concrete example of a scenario where the difference in performance would be most pronounced?"
- "Your early-join scatters config features only onto node_config_ids positions, leaving other nodes as zero. What implicit assumption does this make about which nodes are cost-sensitive, and is this assumption always valid?"

---

## 1.4 The Random vs Default Split

**Rigorous explanation:**
The default split samples configurations near XLA's own default layout choices — the configurations are clustered in a small region of the layout search space. The random split samples uniformly across the full search space. The default split is actually harder to rank in some respects because configurations are very similar to each other — small differences in runtime between nearly identical configurations. The random split has higher runtime variance across configurations, making relative rankings more pronounced, but requires the model to generalize across a much wider and more diverse configuration space.

**Why it matters:**
You chose Layout:XLA:Random because it is the most structurally diverse and most practically relevant split — it tests whether your model can guide search across the full configuration space, not just near the compiler's default. This is the setting where a learned cost model provides the most value.

**Probing questions:**
- "Your Phase 0.5 found ρ = −0.52 between spectral gap and runtime variance across 84 graphs. Does this correlation hold equally across both random and default splits, or would you expect it to be stronger in one? Why?"
- "If a graph has high runtime variance across configurations in the random split, what does that tell you about the sensitivity of its execution time to layout decisions?"

---

# Category 2 — The Applied Mathematics

---

## 2.1 The Normalized Graph Laplacian

**Rigorous explanation:**
The normalized graph Laplacian is $\mathcal{L} = I - D^{-1/2}AD^{-1/2}$, where $A$ is the adjacency matrix and $D$ is the diagonal degree matrix. Applied to a signal vector $x$, it computes at each node the difference between that node's value and the degree-normalized average of its neighbors. The quadratic form $x^\top \mathcal{L} x = \sum_{(i,j) \in E} (x_i - x_j)^2 / \sqrt{d_i d_j}$ equals the total squared variation of signal $x$ across edges — a direct measure of signal roughness on the graph. $\mathcal{L}$ is symmetric positive semidefinite, guaranteeing real non-negative eigenvalues $0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{N-1}$.

**Why it matters:**
The Laplacian is the mathematical object that connects graph topology to GNN information flow. Its eigenvalues are the natural diffusion rates of the graph — they determine how quickly signals spread across the graph during message passing, and therefore how much information a GNN can gather from distant nodes within a fixed number of layers.

**Probing questions:**
- "Why do we use the normalized Laplacian rather than the unnormalized version $L = D - A$ for this analysis? What bias does normalization correct for, and is that bias relevant for XLA computation graphs?"
- "The Laplacian is defined for undirected graphs, but XLA HLO graphs are directed. How did you handle this in your Phase 0.5 analysis, and what information is lost or distorted by treating a directed graph as undirected?"

---

## 2.2 The Spectral Gap $\lambda_1$

**Rigorous explanation:**
$\lambda_1$ is the smallest nonzero eigenvalue of $\mathcal{L}$, found via the Rayleigh quotient minimization:

$$\lambda_1 = \min_{x \perp \mathbf{1}} \frac{x^\top \mathcal{L} x}{x^\top x} = \min_{x \perp \mathbf{1}} \frac{\sum_{(i,j) \in E} (x_i - x_j)^2 / \sqrt{d_i d_j}}{\sum_i x_i^2}$$

It identifies the smoothest possible non-trivial signal on the graph — the signal that varies least across edges while still being non-uniform. Its corresponding eigenvector is the global mode of slowest diffusion. For AlexNet you measured $\lambda_1 = 0.0047$, which is extremely small, indicating a signal pattern exists that the graph can barely smooth out at all.

**Why it matters:**
$\lambda_1$ governs the decay rate of GNN message passing. Each layer of GraphSAGE applies the normalized adjacency $\hat{A} = I - \mathcal{L}$, whose eigenvalues are $1 - \lambda_i$. After $L$ layers, the slowest-decaying mode has eigenvalue $(1 - \lambda_1)^L = (0.9953)^L$. This directly controls the Jacobian norm between distant nodes. A tiny $\lambda_1$ means long-range gradient flow is severely attenuated — over-squashing.

**Probing questions:**
- "You measured λ₁ = 0.0047 for AlexNet. How does this compare to what you would expect for a well-connected random graph of the same size and density? What does the gap between those values tell you about the structural pathology of XLA computation graphs?"
- "λ₁ is a global property of the graph. But over-squashing is a local phenomenon — it affects specific pairs of distant nodes. Can a graph have a small λ₁ but no practically relevant over-squashing, and vice versa?"

---

## 2.3 The Cheeger Inequality

**Rigorous explanation:**
The Cheeger constant $h(G) = \min_{S:|S| \leq N/2} |\partial S| / \text{vol}(S)$ measures the minimum edge-cut ratio over all graph partitions — the minimum fraction of edges you must remove relative to the smaller partition's volume to disconnect the graph into two large pieces. The Cheeger inequality:

$$\frac{\lambda_1}{2} \leq h(G) \leq \sqrt{2\lambda_1}$$

provides a two-sided bound connecting the algebraic spectral gap to the geometric bottleneck. For your AlexNet graph: $0.00235 \leq h(G) \leq 0.097$. This proves the existence of a structural bottleneck — a narrow bridge of edges through which all cross-graph information must pass.

**Why it matters:**
The Cheeger inequality is what gives physical meaning to the spectral gap measurement. $\lambda_1 = 0.0047$ is an algebraic number — the Cheeger bound translates it into a geometric statement about the graph's topology. It confirms that the over-squashing you observe is caused by a real structural bottleneck in the computation graph, not an artifact of the eigenvalue computation.

**Probing questions:**
- "The Cheeger inequality gives you bounds on h(G), not the exact value. For your research narrative, is the bound sufficient or do you need to compute h(G) exactly? What additional information would the exact Cheeger constant give you?"
- "If you added your Granite-style hardware contention edges to the graph, what would you expect to happen to both h(G) and λ₁? Can you quantify the expected change?"

---

## 2.4 Jacobian Decay — The Formal Statement of Over-Squashing

**Rigorous explanation:**
In a GraphSAGE network with $L$ layers and mean aggregation, the influence of node $u$'s initial features on node $v$'s final representation is measured by the Jacobian $\partial h_v^{(L)} / \partial h_u^{(0)}$. Applying the chain rule through $L$ layers and using the spectral decomposition of the message passing operator $\hat{A} = I - \mathcal{L}$:

$$\left\|\frac{\partial h_v^{(L)}}{\partial h_u^{(0)}}\right\|_F \leq (1 - \lambda_1)^d = (0.9953)^d$$

where $d$ is the graph distance between $u$ and $v$. If $d > L$, the Jacobian is exactly zero — the signal never arrived. If $d \leq L$ but the path routes through the Cheeger bottleneck, the effective resistance is far higher than $d$ suggests and the Jacobian is near zero.

**Why it matters:**
This is the formal mathematical statement of why your cost model fails on high-diameter computation graphs. Hardware execution cost is a global property — it depends on operations far apart in the graph competing for the same TPU resources. But the GNN's Jacobian shows that distant operations have zero or near-zero influence on each other's learned representations, making it impossible for the model to capture those global interactions.

**Probing questions:**
- "The Jacobian bound $(1 - λ₁)^d$ applies to the global spectral gap. But the Cheeger constant tells us the bottleneck is localized — only specific paths are constricted. How would you bound the Jacobian for nodes whose shortest path specifically routes through the Cheeger bottleneck?"
- "Your bound is an upper bound on the Jacobian norm. Can you construct a graph where the bound is tight — where the actual Jacobian equals the bound — and a graph where the bound is loose? What structural property determines the tightness?"

---

## 2.5 What Your Correlation Actually Means Physically

**Rigorous explanation:**
Your finding: $\rho(\lambda_1, \sigma_{\text{runtime}}) = -0.52$, $p < 0.001$ across 84 Layout:XLA:Random graphs. Spearman rank correlation of $-0.52$ means that as spectral gap decreases, runtime standard deviation tends to increase — graphs with more severe structural bottlenecks have more volatile execution times across compiler configurations. The $p < 0.001$ means the probability of observing this correlation by chance under the null hypothesis of no relationship is less than 0.1%. The effect size of 0.52 in Spearman terms represents a moderately strong monotonic relationship.

**The physical interpretation:** A graph with small $\lambda_1$ has a structural bottleneck. That bottleneck corresponds to a sequential dependency chain in the computation — a series of operations that must execute in order before the rest of the computation can proceed. The layout configuration choices for operations in and around this bottleneck have an outsized effect on total execution time, because they determine how efficiently data can flow through this critical path. This creates high runtime variance. Conversely, well-connected graphs (large $\lambda_1$) have no single critical path — layout decisions are more evenly distributed and the total runtime is more stable.

**Why it matters:**
This correlation is your empirical proof that the over-squashing problem is not theoretical — it directly causes prediction failures on real graphs. The graphs where your GNN is most blind (small $\lambda_1$, high Jacobian decay) are the same graphs where accurate cost prediction matters most (high runtime variance, large differences between good and bad configurations).

**Probing questions:**
- "Your correlation is ρ = −0.52, which leaves 73% of variance unexplained. What other graph properties do you hypothesize are responsible for the remaining variance, and how would you test those hypotheses?"
- "The correlation is between spectral gap and runtime standard deviation. But your model is evaluated on Kendall's Tau — a rank correlation metric. Can a graph have high runtime standard deviation but low Kendall's Tau difficulty, and vice versa? Does your Phase 0.5 finding directly predict model difficulty?"

---

# Category 3 — The Loss Function

---

## 3.1 Why Ranking Loss and Not Regression

**Rigorous explanation:**
The task is: given a graph and $K$ compiler configurations, identify which configurations produce the fastest execution times. This is a ranking problem — the ground truth is a total ordering of configurations by runtime, not a set of absolute runtime values. Mean Squared Error (MSE) treats the task as regression — it penalizes the model for predicting $\hat{C}(\pi) = 150$ms when the true value is $C(\pi) = 200$ms. But if the model correctly identifies that configuration A is faster than B faster than C, the absolute values don't matter — only the relative ordering does. MSE optimizes a proxy objective (minimize absolute prediction error) that doesn't directly correspond to the true objective (correct ranking). This mismatch causes MSE-trained models to achieve lower Kendall's Tau despite lower prediction error, as confirmed in the TPUGraphs paper.

**Why it matters:**
In practice, your cost model is used to select the top-$K$ configurations to evaluate on real hardware, not to predict exact runtimes. A model that correctly ranks configurations is useful even if its absolute predictions are wrong. MSE-training can produce a model that predicts runtimes accurately but fails to rank them correctly — completely useless for autotuning.

**Probing questions:**
- "MSE and ListMLE both provide gradient signal to the model. Can you construct a specific example — a set of three configurations with specific predicted and true runtimes — where MSE gradient points in the opposite direction from ListMLE gradient? What does this imply about which configurations MSE incorrectly encourages the model to prioritize?"
- "ListMLE is a listwise ranking loss. How does it differ from pairwise losses like RankNet or LambdaRank, and what computational advantage or disadvantage does the listwise approach have when $K$ — the number of configurations per graph — is large?"

---

## 3.2 ListMLE — The Mathematical Formulation

**Rigorous explanation:**
ListMLE maximizes the log-likelihood of observing the correct permutation of items ranked by their predicted scores. Given predicted scores $\hat{s}_1, \hat{s}_2, \ldots, \hat{s}_K$ for $K$ configurations, and the true optimal ordering $\pi^*$ (sorted by actual runtime ascending), ListMLE defines the probability of observing permutation $\pi^*$ using the Plackett-Luce model:

$$P(\pi^* | \hat{s}) = \prod_{k=1}^{K} \frac{\exp(\hat{s}_{\pi^*(k)})}{\sum_{j=k}^{K} \exp(\hat{s}_{\pi^*(j)})}$$

The loss is the negative log-likelihood:

$$\mathcal{L}_{\text{ListMLE}} = -\log P(\pi^* | \hat{s}) = -\sum_{k=1}^{K} \left[\hat{s}_{\pi^*(k)} - \log \sum_{j=k}^{K} \exp(\hat{s}_{\pi^*(j)})\right]$$

This is differentiable with respect to all predicted scores $\hat{s}$, making it suitable for gradient descent. It directly optimizes the probability that the model assigns the correct ranking, making it a tighter surrogate for Kendall's Tau than MSE.

**Why it matters:**
The connection between ListMLE and Kendall's Tau is not just intuitive — it is a formal alignment between the training objective and the evaluation metric. MSE optimizes a completely different objective from evaluation. ListMLE minimizes the gap between what the model is trained to do and what it is evaluated on.

**Probing questions:**
- "ListMLE is a surrogate for Kendall's Tau. Why can't you directly optimize Kendall's Tau with gradient descent, and what properties must a surrogate loss satisfy to be a valid substitute?"
- "The Plackett-Luce probability model assumes configurations are ranked by independent draws from an exponential distribution parameterized by predicted scores. Is this assumption plausible for compiler configurations of the same graph? What happens to the loss when two configurations have nearly identical runtimes?"

---

## 3.3 Kendall's Tau as Evaluation Metric

**Rigorous explanation:**
Kendall's Tau measures rank correlation between two orderings:

$$\tau = \frac{C - D}{\binom{K}{2}}$$

where $C$ is the number of concordant pairs (pairs ranked in the same relative order by both prediction and ground truth) and $D$ is the number of discordant pairs (pairs ranked in opposite order). $\tau \in [-1, 1]$: perfect ranking gives $\tau = 1$, random ranking gives $\tau \approx 0$, perfectly reversed ranking gives $\tau = -1$. The TPUGraphs baseline achieves $\tau = 0.34$ on Layout:XLA:Random — meaning the model gets about 67% of pairwise comparisons correct on the test set.

**Why it matters:**
Kendall's Tau directly measures ranking quality — the thing you actually care about. A model with $\tau = 0.34$ that you use for top-1 selection will find a configuration that is on average 10-20% slower than the true optimum, per the TPUGraphs paper results. Improving $\tau$ to 0.50+ would meaningfully reduce this suboptimality gap and produce a practically useful autotuner improvement.

**Probing questions:**
- "Kendall's Tau is computed per-graph and then averaged across graphs. Why does averaging matter — could a model achieve high average Tau while being useless in practice, and what type of graph distribution would cause this?"
- "Your baseline target is τ > 0.34. If your Phase 3 hardware augmentation improves τ from 0.34 to 0.38, how do you determine whether this improvement is statistically significant given the variance across 2,605 test graphs?"

---

# Category 4 — The Hardware

---

## 4.1 TPU v3 Architecture Fundamentals

**Rigorous explanation:**
The TPU v3 is a domain-specific accelerator designed for dense matrix computations in ML workloads. Its compute hierarchy has two primary units: the Matrix Multiply Unit (MXU) and the Vector Processing Unit (VPU). The MXU is a 128×128 systolic array performing 128×128 matrix multiplications in bfloat16 per clock cycle — approximately 90 TFLOPS peak. The VPU handles elementwise operations (activations, normalization, elementwise add) and operates on 128-element vectors. HBM (High Bandwidth Memory) provides approximately 900 GB/s bandwidth, and the on-chip SRAM scratchpad serves as a fast intermediate buffer. The TPU v3 is organized in pods with inter-chip interconnects, but for single-device inference the relevant hierarchy is MXU → VPU → SRAM → HBM.

**Why it matters:**
Your hardware-aware graph augmentation in Phase 3 requires knowing which operations compete for which hardware resources. Matrix multiplications compete for the MXU. Elementwise and reduction operations compete for the VPU. All large tensor operations compete for HBM bandwidth. Without knowing this hardware anatomy, you cannot design meaningful resource contention edges.

**Probing questions:**
- "The MXU operates on 128×128 tiles. What happens when a matrix multiply operation in the HLO graph has dimensions that are not multiples of 128? How does XLA handle this, and what is the cost implication for your cost model?"
- "The TPU v3 MXU uses bfloat16. How does the choice of numerical precision at the hardware level interact with the layout configuration choices your model is trying to optimize? Are there layout decisions that are more or less sensitive to the precision constraint?"

---

## 4.2 MXU vs VPU — Why the Distinction Matters for Layout

**Rigorous explanation:**
The MXU is deeply sensitive to data layout. It expects inputs as contiguous 128×128 tiles in memory — if the tensor dimensions are laid out with the wrong dimension as innermost (minor), the TPU must perform a physical transpose before the MXU can operate, adding latency and HBM bandwidth overhead. The VPU, operating on 128-element vectors, is less sensitive to multi-dimensional layout but still requires contiguous vector-aligned data. The critical insight is that adjacent operations in the HLO graph may have conflicting layout preferences — a MatMul wants one layout for its output, but the subsequent elementwise operation wants a different layout for its input. XLA resolves this by inserting explicit copy operations, and the layout autotuner minimizes the total cost of these copies.

**Why it matters:**
This is the physical motivation for why layout configurations affect execution time and why the cost model needs to understand graph structure, not just individual nodes. A layout decision at node $i$ propagates a constraint to node $j$ through the edge between them — if they disagree, a copy is inserted. The total copy cost depends on the global layout assignment across the entire graph, not just local decisions. This is why your GNN needs to see the full graph.

**Probing questions:**
- "You described layout configurations as controlling dimension ordering. But the TPU v3 also has tile size configurations in the tile collection. How do layout and tile decisions interact, and why did you focus on layout rather than tile configurations?"
- "If two operations in the HLO graph have the same layout preference — both want the same dimension as innermost — does that guarantee no copy overhead between them? What other factors determine whether a copy is required at a given edge?"

---

## 4.3 Why Layout Choices Cause Bottlenecks — Physical Mechanism

**Rigorous explanation:**
A layout bottleneck occurs when a frequently-executed operation — especially one on the critical path of the computation — requires a tensor layout that conflicts with its input's layout. The copy operation XLA inserts to resolve this conflict has three costs: memory bandwidth to read the source tensor, compute to perform the transpose, and memory bandwidth to write the transposed tensor. For a tensor of shape $[B, H, W, C]$ with large $B$ or $C$, this copy can take as long as the actual computation it precedes. When such a copy sits on the computation's critical path — the longest chain of sequential operations — it directly increases total execution time. Poor layout assignments can introduce multiple such bottlenecks along the critical path, causing execution times 2-5× slower than the optimal layout.

**Why it matters:**
This is why the runtime variance in Layout:XLA:Random is so high — the difference between a layout that avoids critical-path copies and one that creates them can be enormous. And this is why the over-squashing problem is practically important: if your GNN cannot propagate layout information across the graph, it cannot detect that a layout choice at one node is creating a copy bottleneck several hops away on the critical path.

**Probing questions:**
- "You found ρ = −0.52 between spectral gap and runtime variance. Now that you understand the physical mechanism of layout bottlenecks, can you propose a more specific mechanistic hypothesis for why graphs with low spectral gap have high runtime variance? What graph-structural property mediates this relationship?"
- "Your Phase 3 plan adds MXU and HBM contention edges to the computation graph. But layout bottlenecks are caused by copy operations inserted at edges between nodes with conflicting layouts — not by two nodes competing for the same hardware unit. Is there a mismatch between your proposed edge types and the physical source of layout cost? How would you address this?"

---

# Summary — What You Should Be Able to Say Fluently

Before walking into your PI meeting, you should be able to answer these five synthesis questions without hesitation:

**1.** "Walk me from a raw .npz file to a trained cost model prediction. Every step."

**2.** "You found ρ = −0.52. Explain to me, without any equations, why a computation graph that is hard for information to flow across is also a computation graph where layout choices matter more."

**3.** "Why is ListMLE the right loss for this problem? Give me both the mathematical reason and the practical reason."

**4.** "You're about to add hardware-aware edges in Phase 3. What physical property of the TPU are those edges encoding, and how do you know those edges are the right ones to add?"

**5.** "If DiffPool increases the effective spectral gap of the coarsened graph, how does that directly translate into better Kendall's Tau on the test set?"

If you can answer all five clearly and without hedging, you are ready for the meeting.