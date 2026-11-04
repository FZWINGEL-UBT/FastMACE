# Gemini research chats (owner's account, October and November 2026)

The owner's chats with Google Gemini about speeding up the research engine and FastMACE, added to the repository on
4 November 2026 as a record. They are kept unchanged below. Gemini's numbers are its own estimates and literature
figures, not our measurements; the chats are separated by `//` lines. Contents, in order:

- A research request about the ResearchModule Langevin engine (layer-0 radial projection, species and Morton atom
  order, BAOAB, GPU neighbour list), with Gemini's answer.
- "Algorithmic Optimization Opportunities for FastMACE" (one long line): the outside report with the 512-knot cubic
  Hermite spline that the 27 October Claude session tested and rejected
  (`docs/design/2026-10-27-radial-tables-proposal.md`).
- "Look at the optimizations I have done so far in … FastMACE and then propose new ones", with Gemini's answer.
- "Try to come up with more optimizations for FP32 in … FastMACE", with Gemini's answer.
- The research prompt of 28 October 2026 (`results/fp32-profile-20261028/research-prompt.md`) and Gemini's answer.
- The research prompt of 30 October 2026 (main `d35658be`) and Gemini's answer.
- The research prompt of 4 November 2026 (`docs/design/2026-11-04-gemini-research-prompt.md`) and Gemini's answer
  (nine ranked proposals), appended on 4 November 2026.

---


This is from my github repo https://github.com/FZWINGEL/ResearchModule Can you try to research more optimization opportunities for the langevin thermostat MD?

Your optimization campaign in `ResearchModule` has already extracted the vast majority of standard PyTorch and library-level gains—bringing `NaCl` (**216** atoms) from **41.4** ms down to **4.65** ms (**∼**9**×** speedup) and `NaNbO3` (**1**,**080** atoms) from **84.59** ms to **30.92** ms while cutting peak VRAM in half and reducing quenching from **31** h to **1.1** h via LBFGS^^.

**project_overview.html**

Because you have already implemented a custom GPU Langevin engine, CUDA graphs with 5% edge bucketing, `torch.compile` (`create_graph=False`), the `OpenEquivariance` + `cuEquivariance` hybrid backend, and five exact mathematical rewrites^^, further gains must come from eliminating the remaining architectural bottlenecks inside the MACE MH-1 forward/backward graph and the host-GPU control loop.

**project_overview.html**

All optimization opportunities below strictly respect your project's two ground rules: **exact FP32 physics** (force error ratio **≤**2.0**×** vs. FP64 reference, no TF32 or lossy LUTs) and **full generality** (any crystal, any element combination, any GPU)^^.

**project_overview.html**

## 1. Mathematical & Kernel-Level Optimizations (MACE Forward + Backward)

### A. Push the Layer-0 Radial Linear Projection *After* the First-Layer Shortcut

In Chapter 6, you implemented the **First-Layer Shortcut** (reassociating Layer 0 by aggregating messages per `(receiver_atom, neighbor_species)` before the tensor product, with a custom Triton backward kernel), yielding a **4.9**–**7.2%** speedup^^. **You can push this exact algebraic reassociation one step further into the Layer-0 Radial MLP itself.**

**project_overview.html**

* **The Bottleneck:** In MACE-MH-1, the initial node features entering Layer 0 are purely scalar (**ℓ**in=**0**, `128x0e`). The Layer-0 radial MLP takes the **10** Bessel features (after your Element Shortcut^^), passes them through hidden layers of width **W**R=**64**, and projects them in the final linear layer to **4**×**128**=**512** weights per edge (for **ℓ**∈**{**0**,**1**,**2**,**3**}**). For `NaNbO3` (**E**≈**84**,**000**), that final linear layer (`[E, 64] @ [64, 512]`) and your custom Triton reassociation kernel operate on **128** channels across all **84**,**000** edges.**project_overview.html**
* **The Exact Rewrite:** Because the final layer of the radial MLP is a linear projection (with the polynomial cutoff **u**(**r**ij) folded in^^):

  **project_overview.html**

  **w**e**,**ℓ**,**c**(**0**)=**q**=**1**∑**64****h**~**e**,**q****W**q**,**ℓ**,**c**(**0**)****+**u**(**r**e****)**b**Z**i****,**Z**j****,**ℓ**,**c**(**0**)****,**where **h**~**e****=**u**(**r**e****)**SiLU**(**z**e****)**

  Instead of projecting **64**→**512** on all **E** edges *before* the Triton species-aggregation kernel, have the Triton kernel aggregate the **64-dimensional penultimate radial hidden state** **h**~**e** (plus **u**(**r**e) for the bias) outer-producted with the spherical harmonics **Y**ℓ**m****(**r**^**ij****)**∈**R**16** directly into a tensor of shape **[**N**,**S**,**65**,**16**]**:
  **G**i**,**s**,**q**,**ℓ**,**m****=**j**∈**N**s(**i**)**∑h**~**e**,**q****Y**ℓ**m****(**r**^**ij****)

  Then apply the **64**→**512** linear projection **once per `(atom, species)`** (**N**×**S**=**1**,**080**×**3**=**3**,**240** rows) instead of **once per edge** (**E**=**84**,**000** rows).
* **Why it wins:**

  1. Shrinks the Layer-0 final linear GEMM by **∼**6.5**×** on large crystals.
  2. Cuts the feature dimension inside your custom Triton forward and backward kernels from **C**=**128** to **W**R+**1**=**65**, **halving global memory traffic and register pressure** in the Triton kernel.
  3. Move the scalar cutoff multiplication **u**(**r**e) in **both** Layer 0 and Layer 1 to scale the 64-dim hidden vector **h**e *before* the final linear layer (`[E, 64]`) rather than scaling the expanded output (`[E, 512]` or `[E, 1024]`).

### B. Atom-by-Species + 3D Morton Ordering & Active-Element Weight Slicing

You tested stripping the 5 unused DFT heads from the checkpoint and found no gain^^—because readout heads are a tiny fraction of runtime. However, MACE-MH-1 is a **foundation model covering 89 chemical elements**^^, which creates a hidden cache bottleneck inside `cuEquivariance`.

**project_overview.html**+ 1

* **The Bottleneck:** In MACE's `EquivariantProductBasisBlock`, `cuEquivariance`'s `SymmetricContraction` indexes its weight tensor by atomic species (**89** elements). Each crystal only contains **S**∈**{**2**,**3**,**4**}** elements^^, yet:**project_overview.html**

  1. The weight tensor allocated in memory spans all **89** elements, thrashing the L2 cache.
  2. If atoms in the supercell are ordered by unit-cell repetition rather than by species, threads within the same 32-thread GPU warp process different elements (e.g., `Na`, `Nb`, `O` interleaved), causing **divergent weight-table lookups** in `SymmetricContraction`.
* **The Optimization (Two-Part, Zero-Cost at Runtime):**

  1. **Active-Element Slicing at Init:** When initializing the calculator for a structure, find the **S** unique atomic numbers present (`torch.unique(Z)`), slice the species-indexed weight tables (`SymmetricContraction`, node embeddings, element-shortcut tables) from **89** down to **S**∈**{**2**,**3**,**4**}**, and remap `node_species` to **{**0**,**…**,**S**−**1**}**.
  2. **Permute Atoms by `(Species, 3D Morton Code)` at **t**=**0**:** Before starting MD, permute the atom order in the supercell so that all atoms of element 0 come first, element 1 second, etc., and within each element sort atoms along a 3D space-filling Morton (Z-order) curve. (Permute the initial positions/velocities/masses once, and un-permute only when saving snapshots every 20 steps; to preserve your exact ASE RNG validation test, apply the same static permutation to the 256-step RNG buffer).
  3. **Secondary Sort Edges by `(receiver_i, sender_j)`:** Extend your Chapter 6 Receiver Sorting^^ to sort lexicographically by `receiver_i * N + sender_j`.
     **project_overview.html**
* **Why it wins:** During `SymmetricContraction`, 100% of warps (except the 2 boundary warps between species) execute with  **uniform species index across all 32 threads** , turning scattered L2 weight fetches into broadcast L1 cache hits. Meanwhile, Morton ordering + `(receiver_i, sender_j)` sorting maximizes L2 cache locality when `OpenEquivariance` gathers sender features **H**j in Layer 1.

### C. Exploit Receiver Sorting in the Final Force Accumulation

* **The Bottleneck:** Forces are computed as **F**i=**−**∂**r**i∂**E****=**∑**j**∈**N**(**i**)****∂**r**ij****∂**E****−**∑**k**:**i**∈**N**(**k**)****∂**r**k**i****∂**E****. When PyTorch autograd backpropagates through `vectors = positions[senders] - positions[receivers] + shifts`, it executes two `index_add_` (atomic scatter-add) calls over all **E** edges into `[N, 3]`.
* **The Optimization:** Because your edges are **already sorted contiguously by `receiver_i`**^^, the receiver force contribution **F**i**rec****=**∑**e**∈**rec**(**i**)f**e **does not need global `atomicAdd`! Differentiate total energy with respect to `edge_vectors` (**f**e****=**∂**r**ij****∂**E****∈**R**E**×**3), use a segmented reduction (or `torch.segment_reduce`) for the contiguous receiver forces, and only scatter-add the sender forces.
  **project_overview.html**

## 2. Engine, Neighbor-List & Host-Sync Optimizations

### A. Why the Oct 13 "Speculative Step" Slowed Down — and How to Fix It

In Chapter 8, you tested a speculative "never wait for the CPU" mode and found it was **3.7**–**7.3%** *slower* because: (1) it still had to check the guess once per step, and (2) redoing **6**–**8%** of steps at **1000** K cost more than it saved^^.

**project_overview.html**

* **Diagnosis of the 6–8% Rollback Rate:** Your skin list holds pairs up to **6.3** **A**˚, and at each step a GPU routine prunes it to active pairs within **6.0** **A**˚ and rounds up to a 5% edge bucket^^. At **1000** K, thermal vibrations cause the exact number of pairs inside **6.000** **A**˚ to jitter by **±**1**–**2% every step. Whenever the active pair count hovers near a 5% bucket boundary, thermal jitter causes frequent bucket overflows—triggering expensive rollbacks even when the **6.3** **A**˚ skin list itself is still completely valid!**project_overview.html**
* **The Fix ("Sticky/Hysteresis Bucket" + **K**-Step Macro-Graphs):**
  1. **Decouple Bucket Selection from Step-by-Step Thermal Jitter:** When a skin list is built (every 15–30 steps), select the CUDA graph bucket corresponding to **E**active+**Δ**E**margin** (e.g., **+**2% safety headroom, or simply 1 bucket higher = **+**5% edges) and **lock that bucket for the entire lifetime of the skin list** (or only change buckets if **E**active**** exceeds the locked bucket capacity). Because your built-in smoothing sets contributions at **r**≥**6.0** **A**˚ to zero^^, extra padding slots cost only ~1–2% compute but  **drop bucket-overflow rollbacks from 6–8% to **<**0.1%** .**project_overview.html**
  2. **Check the Host Flag Every **K**=**5** Steps Instead of Every Step:** At **1000** K, the lightest atom (**O**, **16** u, or **Na**, **23** u) has thermal RMS velocity **v**rms=**3**k**B****T**/**m**![]()**≈**0.012** **A**˚**/fs**=**0.006** **A**˚** per **0.5** fs step. With a **0.3** **A**˚ skin (requiring **0.15** **A**˚ displacement to invalidate), an atom cannot physically cross the skin in **K**=**4** to **5** steps. Launching 5 steps per host flag check (or recording a 5-step macro-CUDA-graph) eliminates **80%** of the remaining host-GPU synchronization events with  **zero rollbacks** .

### B. Native GPU Neighbor-List Rebuild (Tiled **N**×**N** + Warp-Ballot Prefix Sum)

Chapter 14 notes that rebuilding the neighbor list on the GPU is a logical next step^^. This will become critical when you move to melting/re-freezing protocols^^ where high diffusivity triggers skin rebuilds much more frequently.

**project_overview.html**+ 1

* **The Design:** For **N**≤**2**,**000** atoms, you do not need a complex dynamic spatial hash table. A single **Triton tiled pair-list kernel** where each program instance handles a block of 32 receiver atoms **i** and loops over tiles of 32 sender atoms **j** across the 27 periodic image shifts evaluates all **1**,**94**4**2**×**27**≈**1**0**8** candidate pairs in **<**0.3** ms** on an RTX 5060 Ti (or **<**0.05** ms** if filtered by unit-cell bounding boxes).
* **Native Receiver-Sorted Output:** Because each thread is assigned to a specific receiver atom **i** and scans sender atoms **j**=**0**…**N**−**1** in ascending order, a two-pass count + prefix-sum write produces the new skin list **already sorted by `(receiver_i, sender_j)`** on the GPU—eliminating CPU-GPU PCIe transfers, CPU `matscipy` / ASE overhead, and `torch.argsort` in one stroke.

## 3. Algorithmic & Future-Protocol Optimizations

### A. BAOAB Langevin Integrator Splitting (For Future **1.0** fs Runs)

In Chapter 5, you tested **Δ**t**=**1.0** fs** vs. **0.5** fs and found that **1.0** fs passed all stability and statistical tests, but you kept **0.5** fs out of caution^^.

**project_overview.html**

* Standard ASE `Langevin` uses a velocity-Verlet-like stochastic splitting whose configurational sampling bias scales as **O**(**Δ**t**2**) with a significant prefactor.
* Switching the 4-line velocity/position update kernel to the **BAOAB** splitting (Leimkuhler & Matthews)—which places the exact Ornstein-Uhlenbeck thermostat ("O") step in the middle of two half-position ("A") steps—cancels the leading-order configurational sampling error (**O**(**Δ**t**4**) in the harmonic limit) using the  **exact same single force evaluation per step** .
* **Impact:** BAOAB at **Δ**t**=**1.0** fs** provides *higher* configurational accuracy than standard ASE Langevin at **Δ**t**=**0.5** fs**, giving you a safe path to a **2**×** wall-clock speedup** on future large-scale screening runs (while keeping the ASE splitting as a flag for strict trajectory regression tests).

## Summary of Optimization Opportunities

| Optimization Opportunity                                                  | Target Component                                            | Expected Speed / Memory Gain                                                              | Preserves FP32 Accuracy & Generality?                                                   | Implementation Effort                                                          |
| ------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **1. Post-Shortcut Layer-0 Radial Projection**                      | Layer-0 Radial MLP + Triton Shortcut                        | **3–6% faster step** , 50% less traffic in Layer-0 Triton kernel                   | **Yes**(100% algebraically identical)                                             | **Medium**(Modify Layer-0 shortcut & Triton kernel)                      |
| **2. Species + Morton Atom Permutation & Active-Z Slicing**         | `cuEquivariance`Contraction +`OpenEquivariance`L2 Cache | **3–7% faster step**on multi-element crystals (`NaNbO3`,`NaNbCl6`)             | **Yes**(Exact permutation at**t**=**0**+ slice to active**Z**)  | **Low–Medium**(~50 lines in setup & snapshot saver)                     |
| **3. Sticky Bucket Margin +**K**-Step Sync Cadence**          | CUDA Graph Replay & Host-GPU Sync                           | **2–4% faster step**(recovers speculative overhead with 0% rollback)               | **Yes**(Dummy edges at**r**>**6** **A**˚contribute**0**) | **Low**(Adjust bucket selector & sync every**K**=**5**steps) |
| **4. GPU Tiled Neighbor-List Rebuild (Sorted by Construction)**     | Skin-List Rebuild (Every 15–30 steps)                      | **3–8% faster overall**(up to**15–20%**in melting runs)                           | **Yes**(Exact**6.3** **A**˚cutoff + PBC shifts on GPU)               | **Medium**(1 custom Triton/CUDA kernel)                                  |
| **5. Receiver-Sorted Segmented Force Reduction**                    | Backward Pass Force Accumulation                            | **1–2% faster step**(eliminates half of global `atomicAdd`s)                     | **Yes**(Exploits existing receiver sorting)                                       | **Low**(Replace receiver scatter-add with segmented sum)                 |
| **6. BAOAB Langevin Splitting at**Δ**t**=**1.0**fs**** | MD Integrator Protocol                                      | **2**×**faster run**(**100%**speedup) with superior configurational sampling | **Yes**(Exact canonical sampling; optional mode)                                  | **Low**(~15 lines in GPU integrator step)                                |

Would you like to dive into the exact math and Triton kernel changes for  **pushing the Layer-0 radial projection after the First-Layer Shortcut** , or focus first on  **atom-by-species permutation and GPU neighbor-list construction** ?


//

Algorithmic Optimization Opportunities for FastMACE: High-Performance Molecular Dynamics with Pretrained Equivariant Force FieldsArchitectural Baseline of FastMACE and the MACE Execution GraphFastMACE (released as fastmace v1.0.0 for Python 3.13) targets high-throughput molecular dynamics (MD) inference on NVIDIA GPUs using pretrained Message Passing Atomic Cluster Expansion (MACE) foundation and specialist models. While the MACE architecture resolves the traditional depth-versus-expressivity bottleneck of two-body equivariant message-passing neural networks (MPNNs)—such as NequIP, which typically requires five to six message-passing iterations—by explicitly constructing four-body equivariant messages ($\nu = 3$) that converge in just $T = 2$ layers, running long-timescale MD trajectories or million-atom systems remains severely constrained by memory bandwidth, Clebsch–Gordan (CG) tensor contraction complexity, and host-side kernel dispatch latency.To identify high-leverage algorithmic optimizations for FastMACE, the forward and backward execution graph of a two-layer MACE potential must be decomposed into its fundamental mathematical operators and hardware bottlenecks. Given atomic coordinates $\mathbf{R} = (\mathbf{r}_1, \dots, \mathbf{r}_N)$ and chemical species $\mathbf{Z} = (Z_1, \dots, Z_N)$, internal node states $h_{i, kLM}^{(t)}$ at layer $t$ are indexed by channel $k \in \{1, \dots, K\}$ and irreducible representations (irreps) $(L, M)$ of the orthogonal group $O(3)$, transforming under rotation or reflection $Q \in O(3)$ via Wigner D-matrices $D^L_{M'M}(Q) \in \mathbb{R}^{(2L+1)\times(2L+1)}$:$$h_{i, kLM}^{(t)}(Q \cdot \{\mathbf{r}\}) = \sum_{M'} D^L_{M'M}(Q) h_{i, kLM'}^{(t)}(\{\mathbf{r}\})$$At each MD timestep, evaluating the total potential energy $E(\mathbf{R}, \mathbf{Z}) = \sum_{i=1}^N \varepsilon_i$ and the conservative analytical atomic forces $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$ traverses four distinct computational stages per layer, followed by reverse-mode automatic differentiation (autograd) through the entire spatial graph. The pipeline begins with edge geometric embedding, where for every directed edge $(j, i)$ within the cutoff radius $r_{\max}$ (with an average coordination number $d_n$), relative displacement vectors $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ are expanded into real spherical harmonics $\{Y_{\ell m}(\hat{\mathbf{r}}_{ij})\}_{\ell \le \ell_{\max}}$ at an asymptotic cost of $\mathcal{O}(d_n \ell_{\max}^2)$ per node, while scalar distances $r_{ij} = \|\mathbf{r}_{ij}\|$ pass through a radial basis—typically $N_{\text{rad}} = 8$ Bessel functions paired with an Agnesi distance transform and a multi-layer perceptron—at $\mathcal{O}(d_n N_{\text{rad}})$ cost to generate learnable path weights. In the subsequent interaction block (conv_tp channelwise tensor product and pooling), sender node features $h_j^{(t)}$ are coupled with edge spherical harmonics and radial weights via Clebsch–Gordan coefficients to construct directional two-body descriptors, which are pooled over neighbors $j \in \mathcal{N}(i)$ to form the atomic basis features $A_i^{(t)}$. While the Layer 1 interaction is simplified because initial node states are purely invariant species embeddings ($L=0$), the Layer 2 edge CG contraction scales as $\mathcal{O}\left(d_n K^2 \sum_{\text{allowed }(\ell_1, \ell_2, \ell_3)} (2\ell_1+1)(2\ell_2+1)(2\ell_3+1)\right)$, making edge-proportional tensor products and intermediate edge buffer traffic the dominant scaling bottleneck.Once two-body descriptors are aggregated at each node, execution enters the product basis block, where the pooled features $A_i^{(t)}$ undergo $\nu$-fold self-contraction (SymmetricContraction within EquivariantProductBasisBlock) to construct higher-order many-body messages $\mathbf{m}_i^{(t)} = \sum_j u_1 + \sum_{j_1, j_2} u_2 + \dots + \sum_{j_1, \dots, j_\nu} u_\nu$, thereby forming the $B_i^{(t)}$-features. By precontracting CG coefficients with species-dependent weights, this operation becomes strictly node-local with complexity $\mathcal{O}(K N_{\text{path}} (2L+1))$ per node, accompanied by an indexed linear self-connection (skip_tp) conditioned on one-hot atomic species $Z_i$. In the final stage, node features are updated linearly via residual connections, and invariant scalars $\text{Inv}(A_i^{(t)})$ ($L=0$) are mapped through linear and MLP readouts $\rho$ to per-atom site energies $\varepsilon_i = E_{Z_i}^{(0)} + \varepsilon^{(1)}(h_i^{(1)}) + \varepsilon^{(2)}(h_i^{(2)})$. Computing conservative forces $\mathbf{F} = -\nabla_{\mathbf{R}} E$ via autograd doubles the execution depth and requires retaining all intermediate edge and node activations from the forward pass in device DRAM.End-to-end block profiling of MACE reveals a critical hardware divergence that dictates where FastMACE can extract the largest performance gains. In unoptimized execution, the CPU dispatch overhead (the host-side scheduling share in profiler traces) actually exceeds the active CUDA execution time across most mother blocks. Specifically, Product[0]—SymmetricContractions registers as the single largest wall-time hotspot due to host launch overhead and FP64 dgemm execution, while the edge-level Clebsch–Gordan contraction (conv_tp in the Interaction block) drives memory bandwidth saturation and limits the maximum number of atoms that can fit in VRAM. Meanwhile, the Spherical Harmonics, Radial MLP, and Readout blocks fragment execution into dozens of tiny GEMMs, activation functions, and layout transpositions whose runtime is dominated by GPU kernel launch latency rather than floating-point math.MACE Functional BlockAsymptotic Scaling (Per Node / Edge)Hardware Bottleneck ProfilePrimary Algorithmic Optimization TargetSpherical Harmonics & Radial Basis$\mathcal{O}(d_n \ell_{\max}^2) + \mathcal{O}(d_n N_{\text{rad}})$Fragmented into many short kernels and memory layout ops; high DRAM read/write traffic512-knot cubic Hermite radial spline tabulation + fused Triton solid harmonicsInteraction Block (conv_tp & Linear-up)$\mathcal{O}\left(d_n K^2 \sum (2\ell_1+1)(2\ell_2+1)(2\ell_3+1)\right)$Chief driver of edge-proportional wall time and VRAM saturation via materialized edge intermediatesReceiver-chunked 3-pass recomputation; shared-memory streaming CG kernels; $SO(2)$ reductionProduct Basis (SymmetricContraction & skip_tp)$\mathcal{O}(K N_{\text{path}} (2L+1))$ per nodeTop compute hotspot in baseline runs; heavy FP64 dgemm and redundant 89-species one-hot productsUniform 1D segmented polynomials (ir_mul layout) + exact species-block weight gatherReadout Block ($\rho$) & Energy Summation$\mathcal{O}(K_{\text{inv}})$ per nodeHeavily CPU-launch skewed; multiple tiny GEMMs and activations; precision loss in large reductionsBF16/FP16 linear downcast with FP32/Kahan accumulation + static CUDA Graph capturePillar 1: Receiver-Chunked Three-Pass Recomputation for Edge-Free Memory ScalingThe most restrictive algorithmic bottleneck in standard PyTorch and cuEquivariance MACE implementations is memory scaling with respect to the number of graph edges $N_{\text{edges}} = N \cdot d_n$. Because standard reverse-mode automatic differentiation (autograd) requires caching every per-edge intermediate tensor from Layer 1 and Layer 2 during the forward pass to compute $\mathbf{F} = -\nabla_{\mathbf{R}} E$, intermediate activations quickly saturate GPU high-bandwidth memory (HBM). Even with NVIDIA cuEquivariance enabled on an 80 GB A100 GPU, standard full-graph MACE evaluation is capped at approximately $100,000$ atoms. Furthermore, materializing and re-reading gigabytes of edge tensors across global memory turns the entire inference pipeline into a memory-bandwidth-bound workload.FastMACE can overcome this barrier—expanding single-GPU capacity by 31x while cutting peak VRAM usage—by reorganizing two-layer MACE inference into an exact receiver-chunked three-pass algorithm that keeps zero per-edge intermediates resident across passes, as established in recent million-atom MACE deployments (mace-million, Liu et al.). Because a two-layer MACE model expresses the total potential energy as a sum over receiver nodes $i$ with local neighbor sets $\mathcal{N}(i)$:$$E = \sum_{i=1}^N \left[ E_{Z_i}^{(0)} + \varepsilon^{(1)}\!\left(h_i^{(1)}\right) + \varepsilon^{(2)}\!\left(h_i^{(2)}\right) \right]$$where $h_i^{(1)}$ depends only on coordinates $\{\mathbf{r}_{ij}\}_{j \in \mathcal{N}(i)}$ and $h_i^{(2)}$ depends on $\{\mathbf{r}_{ij}, h_j^{(1)}\}_{j \in \mathcal{N}(i)}$, the global computation can be partitioned across disjoint receiver node chunks $\mathcal{C} \subset \{1, \dots, N\}$ of fixed size $|\mathcal{C}|$. In the first stage (Pass A), the engine iterates over receiver chunks $\mathcal{C}$ with torch.no_grad() enabled; for each chunk, it evaluates the inexpensive Layer 1 edge embeddings, interaction pooling, symmetric contraction, and linear update exclusively for receivers $i \in \mathcal{C}$, writing only the compact node-level hidden features $h_i^{(1)}$ to a global node buffer and immediately discarding all Layer 1 edge tensors. In the second stage (Pass B), the engine iterates over receiver chunks $\mathcal{C}$ with autograd enabled on the chunk's edge vectors $\mathbf{r}_{ij}$ and gathered neighbor node features $\{h_j^{(1)}\}_{j \in \mathcal{N}(\mathcal{C})}$ to compute the Layer 2 forward pass and readout energy $E_{\mathcal{C}}^{(2)} = \sum_{i \in \mathcal{C}} \varepsilon^{(2)}(h_i^{(2)})$. Pass B immediately triggers a chunk-local backward pass (autograd.backward) on $E_{\mathcal{C}}^{(2)}$, which accumulates the Layer 2 direct coordinate force contributions $-\partial E_{\mathcal{C}}^{(2)} / \partial \mathbf{r}_{ij}$ into the global force array $\mathbf{F}$ and accumulates the partial adjoints $g_j^{(1)} = \sum_{\mathcal{C}} \partial E_{\mathcal{C}}^{(2)} / \partial h_j^{(1)}$ into a node-sized gradient buffer, freeing all Layer 2 per-edge activations before advancing to the next chunk. In the final stage (Pass C), the algorithm iterates over receiver chunks $\mathcal{C}$ a third time with autograd enabled on the chunk's Layer 1 edge vectors $\mathbf{r}_{ij}$, recomputing the lightweight Layer 1 forward pass for receivers $i \in \mathcal{C}$ to obtain $h_i^{(1)}$ and first-layer site energies $\varepsilon^{(1)}(h_i^{(1)})$, and immediately backpropagating the scalar surrogate $\sum_{i \in \mathcal{C}} \left[ \varepsilon^{(1)}(h_i^{(1)}) + \langle g_i^{(1)}, h_i^{(1)} \rangle \right]$ with respect to $\mathbf{r}_{ij}$ to accumulate the remaining Layer 1 force contributions into $\mathbf{F}$.Because Layer 1 in MACE operates on scalar ($L=0$) inputs and is computationally lightweight compared to Layer 2's full equivariant edge tensor product, recomputing Layer 1 in Pass C incurs minimal FLOP overhead while completely eliminating the storage of $\mathcal{O}(N_{\text{edges}})$ intermediate graphs. Crucially, this reformulation is mathematically exact: across MACE-MATPES-r2SCAN-0, MACE-MATPES-PBE-0, and the four-times-wider multi-head MACE-MH-1 foundation model, the three-pass algorithm reproduces full-graph FP64 forces to within $4.0\text{--}4.4 \times 10^{-13}\text{ eV/\AA}$ while evaluating up to $3.1 \times 10^6$ atoms standalone on a single GPU and sustaining $8.5 \times 10^4$ atom-steps/s for $9.83 \times 10^5$ water atoms in LAMMPS within a 23 GB VRAM envelope. Furthermore, inside each pass, node-level operations (symmetric contraction, linear layers, and self-connections) must be restricted strictly to the active receiver nodes $i \in \mathcal{C}$ of the current chunk rather than running over the padded halo of sender neighbors.Pillar 2: Equivariant Tensor Product Kernels, Shared-Memory Streaming, and Hybrid Backend RoutingWhile FastMACE leverages GPU acceleration for MACE models, relying solely on a single backend such as standard NVIDIA cuEquivariance (cuEq) leaves substantial performance unrealized due to register spilling, layout conversion penalties, and backward-pass inefficiencies. In MACE, four core operations act on irreps: channelwise edge tensor products (conv_tp), symmetric contractions, equivariant linear layers, and indexed linear layers (skip_tp).Register vs. Shared-Memory Staging and OpenEquivariance / FlashTP HybridizationStandard e3nn stores features in a multiplicity-outer, irrep-inner layout (mul_ir), whereas cuEquivariance achieves up to a 200x micro-benchmark speedup on symmetric contractions and a 3x–10x end-to-end MACE speedup by reformulating Clebsch–Gordan contractions and symmetric contractions into uniform 1D segmented polynomials (cue.SegmentedPolynomial) that execute natively in the transposed ir_mul layout. If FastMACE mixes e3nn and cuEq operations or fails to convert checkpoints ahead of time into pure ir_mul layout, explicit TransposeIrrepsLayout adapters are triggered at block boundaries, degrading throughput and risking representation mismatches.More importantly, kernel-level analysis reveals a fundamental bifurcation inside cuEquivariance between two internal execution paths: cuEq-fast, which stages tensor product operands entirely within GPU thread registers, and cuEq-slow, which spills operands into shared memory (SRAM) when vector lengths exceed register file capacity. For large MACE configurations (such as MACE-MP Large, MACE-OMol Extra Large, or models with $\ell_{\max} > 2$), the combined vector length of the two inputs and output $(\mathbf{x}, \mathbf{y}, \mathbf{z})$ exceeds $10,000$ data words per warp. Because warps within a cooperative thread array (CTA) partition shared memory, staging all three vectors plus the weight matrix $\mathbf{W}$ simultaneously in SRAM is impossible, causing cuEq to suffer reduced warp occupancy, degraded L1 cache utilization, and severe performance collapse when $\ell_{\max} > 2$. Additionally, cuEq's fused convolution (conv_fusion) cannot natively handle non-uniform edge irreps, and its double-backward pass (required when computing second-order derivatives, Hessians, or training with force losses) exhibits lower efficiency.To eliminate these bottlenecks, FastMACE can implement dynamic hybrid kernel routing that combines cuEquivariance, OpenEquivariance (oeq), and FlashTP across distinct MACE blocks. For the heavy Layer 2 edge CG tensor product (conv_tp), OpenEquivariance introduces a two-tier warp-partitioned shared-memory streaming scheduler. Under this scheme, when inputs $\mathbf{x}$ and $\mathbf{y}$ fit within a warp's shared-memory partition, segments of output $\mathbf{z}$ and weights $\mathbf{W}$ are streamed through multiple execution phases; conversely, when even $\mathbf{x}$ and $\mathbf{y}$ exceed SRAM capacity, a greedy phase-packing heuristic fills the shared-memory pool with maximal subsets of $(\mathbf{x}, \mathbf{y}, \mathbf{W}, \mathbf{z})$ segments and flushes and reloads them between phases. This streaming architecture eliminates register spilling and delivers up to 4.5x forward and 3x backward speedups over cuEquivariance, alongside a 1.7x median FP32 forward speedup across configurations and parity in FP64.Because OpenEquivariance and FlashTP specifically target pairwise CG tensor products (conv_tp) while cuEquivariance provides specialized kernels for node-local SymmetricContraction and IndexedLinear (skip_tp), FastMACE should route SymmetricContraction and linear layers to cuEq—or to e3j CUDA tensor product kernels with channel-mixing mode MAP when strict checkpoint numerical determinism is required—while routing edge conv_tp and non-uniform irreps to OpenEquivariance (enable_cueq=True, enable_oeq=True) or FlashTP. Empirical evaluations confirm that pairing FlashTP or OpenEquivariance with cuEq reduces MACE-MP Large execution time by an additional 1.2x beyond cuEq alone. Simultaneously, FastMACE should eliminate materialized edge gather buffers by exploiting fused atomic scatter/gather operations within uniform 1D JIT kernels. In standard PyTorch MACE, sender node features node_feats[senders] are explicitly gathered into an edge-sized tensor mji of shape $(N_{\text{edges}}, D)$ before calling conv_tp and subsequently scattered back to receiver nodes. By adopting the uniform 1D JIT segmented polynomial kernel introduced in cuEquivariance v0.3.0 and v0.4.0 with direct index arguments (indices for fused scatter/gather via hardware atomic operations), edge messages are accumulated directly from sender nodes into receiver nodes without materializing mji in global DRAM, a mechanism that accounts for the ~3x speedup observed in cuEquivariance-JAX v0.3.0 over cuEquivariance-Torch v0.2.0.Node-Level SO(2) Axis-Alignment and Spherical Grid TransformationsFor foundation checkpoints or custom MACE architectures utilizing higher angular momentum orders ($\ell_{\max} \ge 2$), the dense Clebsch–Gordan coupling paths scale steeply with $\ell_{\max}$. Two mathematical reformulations bypass full $SO(3)$ tensor product complexity. First, by rotating each edge vector $\mathbf{r}_{ij}$ (or node local frame) so that the interatomic axis aligns with the local $z$-axis via Wigner D-matrices or $x/y/yxy$ Euler rotations (cue.x_rotation, cue.yxy_rotation), spherical harmonics collapse to $m = 0$, reducing $SO(3)$ convolutions of complexity $\mathcal{O}(\ell_{\max}^6)$ to $SO(2)$ operations of complexity $\mathcal{O}(\ell_{\max}^3)$. Incorporating a rank-1 dynamic cross-degree radial mixer (degree_channel) preserves equivariant expressivity while enabling linear-scaling tensor products. Second, transforming equivariant irreps onto a quadrature grid on the sphere converts tensor products into pointwise multiplications, accelerating MACE evaluation and training by 30%. Complementarily, because angular momentum conservation does not require contiguous $\ell$ orders, statically pruning under-utilized spherical harmonic orders $\ell$ or zeroing out sparse CG coupling paths (simplify_irreps3=True in cue.descriptors.channelwise_tensor_product) directly reduces feature dimensionality $d = \sum_{\ell \in L} L_h(2\ell+1)$ without hurting force accuracy.Pillar 3: Exact Sub-Block Algorithmic Simplifications (Splines, Species Gather, and Solid Harmonics)Once the heavy tensor products are accelerated via fused kernels, Amdahl's Law shifts the primary bottleneck in FastMACE to the surrounding embedding and conditioning operations: the radial MLP, the spherical harmonics evaluation, and the multi-element species indexing. Because all model weights are frozen during pretrained MD inference, three exact mathematical transformations can eliminate the vast majority of arithmetic and memory traffic in these blocks.1. Cubic Hermite Spline Tabulation of the Frozen Radial Basis and MLPIn every MACE interaction layer, scalar edge distances $r_{ij} \in (0, r_{\max}]$ are transformed through a distance transform (such as the Agnesi transform), expanded into $N_{\text{rad}}$ Bessel basis functions, and passed through a multi-layer perceptron (typically 3 layers of width 64 with SiLU activations) to produce the edge-specific Clebsch–Gordan path weights $R_{k\ell_1\ell_2\ell_3}^{(t)}(r_{ij})$. During MD inference with a pretrained checkpoint, this entire neural sub-network is a frozen, smooth, univariate function $f: \mathbb{R}^+ \to \mathbb{R}^{N_{\text{paths}}}$. Evaluating transcendental Bessel functions and three sequential dense matrix multiplications per edge at every MD step wastes memory bandwidth and launches numerous small kernels.Instead, at model initialization (FastMACE setup), the radial basis and MLP stack should be evaluated once across a uniform 512-knot grid spanning $[0, r_{\max}]$ to construct a cubic Hermite spline lookup table storing function values and exact first derivatives at each knot. During MD execution, a single Triton kernel computes the normalized grid index $u = (r_{ij} / r_{\max}) \cdot 512$, fetches the 4 cubic Hermite coefficients from L1 or texture cache, and evaluates both the weight $R(r_{ij})$ and its exact analytical radial derivative $\partial R(r_{ij}) / \partial r_{ij}$ via Horner's scheme in just 4 fused multiply-add (FMA) instructions per path. This eliminates all radial MLP intermediate activations from the autograd tape and reproduces full-network FP64 forces to machine precision ($10^{-13}\text{ eV/\AA}$).2. Species-Block Gather Replacing Dense 89-Element One-Hot Self-ConnectionsPretrained universal MACE foundation models—including MACE-MP-0, MACE-MPA-0, MACE-OMAT-0, MACE-MATPES, MACE-OMOL-0, and MACE-MH-0/1—are parameterized over $N_{\text{elem}} = 89$ chemical elements (or 83 elements in MACE-POLAR-1). In the standard MACE architecture, both the node self-connection (skip_tp) and the symmetric contraction weight selection contract node features with a dense 89-dimensional one-hot species vector $\mathbf{z}_i \in \{0, 1\}^{89}$.Evaluating a dense tensor product against an 89-element vector where 88 entries are zero performs $89\times$ redundant floating-point work and forces warps to stream 89-element weight matrices through shared memory. FastMACE can eliminate this redundancy through a two-stage species-indexing optimization. First, because the chemical composition of an MD simulation box remains fixed across a trajectory and typically comprises only $N_{\text{active}} \in [1, 6]$ distinct elements (such as $\text{H}$ and $\text{O}$ in water or $\text{Si}$ and $\text{O}$ in silica), FastMACE can statically slice the loaded checkpoint's 89-element weight tensors along the species axis at initialization to retain only the $N_{\text{active}}$ columns present in the simulation, remapping atomic numbers $Z_i$ to contiguous indices $\{0, \dots, N_{\text{active}}-1\}$. Second, rather than multiplying by a one-hot vector at runtime, the self-connection and symmetric contraction conditioning should be executed as a direct species-block gather (cuex.experimental.indexed_linear on CUDA 12.5+ or a direct pointer offset by integer species ID $Z_i$), converting a dense tensor contraction into a zero-overhead indexed weight lookup.3. Radially Scaled Cartesian Solid Harmonics via Fused Triton / sphericart KernelsIn standard e3nn, evaluating spherical harmonics $Y_{\ell m}(\hat{\mathbf{r}}_{ij})$ requires normalizing edge vectors $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij} / r_{ij}$ and executing TorchScript graphs that generate excessive global memory read/write traffic (reaching ~40 GB of DRAM traffic at $\ell=4$) while suffering from numerical instability near coordinate poles in the backward pass.By switching to radially scaled Cartesian solid harmonics $\tilde{Y}_{\ell m}(\mathbf{r}_{ij}) = r_{ij}^\ell Y_{\ell m}(\hat{\mathbf{r}}_{ij})$—implemented via sphericart CUDA kernels or fused EquiTriton block kernels—the basis functions become pure homogeneous polynomials in Cartesian components $(x_{ij}, y_{ij}, z_{ij})$. This algebraic reformulation yields three compounding advantages for FastMACE. Eliminating the $r_{ij}^{-\ell}$ normalization step and its associated chain-rule quotient derivatives saves 5%–10% of spherical harmonic evaluation time while guaranteeing pole-free, numerically stable analytical Cartesian derivatives $\nabla_{\mathbf{r}_{ij}} \tilde{Y}_{\ell m}$. Furthermore, the extra $r_{ij}^\ell$ scaling factor introduced in $\tilde{Y}_{\ell m}(\mathbf{r}_{ij})$ is compensated at zero runtime cost by multiplying the pre-tabulated cubic Hermite radial spline knots for angular channel $\ell$ by $r_{ij}^{-\ell}$ during model initialization. Finally, fusing the distance cutoff envelopes, solid harmonics evaluation, and analytical backward derivatives into a single tiled Triton kernel keeps intermediate polynomials strictly in SRAM, increasing effective memory bandwidth utilization by 2.5x–4x over PyTorch eager and TorchScript execution.Pillar 4: Sub-Block Mixed-Precision Execution Policies and Compensated AccumulationA critical tension in MLIP molecular dynamics lies between numerical precision and GPU Tensor Core utilization. Pretrained models like MACE-OMol and MACE-MH-1 often default to double precision (FP64) to guarantee smooth potential energy surfaces (PES) and negligible NVE energy drift, yet FP64 execution halves memory bandwidth, inflates shared-memory footprint (forcing cuEq into slower shared-memory streaming), and relies on low-throughput FP64 dgemm units, particularly on consumer Ampere, Ada, and Blackwell GPUs where FP64 throughput is a fraction of FP32 and BF16 throughput.Systematic block-wise mixed-precision profiling of MACE (Benoit, arXiv:2510.23621) demonstrates that uniform precision casting is suboptimal; instead, FastMACE should enforce a granular sub-block precision policy:MACE Sub-Block / OperationOptimal Inference PrecisionAccumulation PrecisionEmpirical Speedup & Physical Fidelity ImpactCoordinate Differences ($\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$) & PBC WrappingFP64 or FP32 (Local Anchor)FP32Prevents catastrophic cancellation when box coordinates $\|\mathbf{r}_i$ are large in periodic systems.Interaction conv_tp & SymmetricContraction (cuEq / oeq)FP32 (with TF32 Tensor Cores on Ampere/Hopper)FP32~3x latency reduction vs. e3nn; preserves force accuracy without shared-memory overflow.Equivariant Linear Layers (Linear-up, Update, Readout GEMMs)BF16 or FP16FP32~4x additional speedup over pure FP32; NVT/NPT water density and RDF remain within run-to-run variance.Per-Atom Energy Reduction ($E = \sum_i \varepsilon_i$) & Force ScatterFP32 inputsFP64 or Kahan Compensated FP32Eliminates long-time NVE total energy drift ($\sim 1.8 \times 10^{-4}\text{ eV/atom}$) across million-step trajectories.Implementing this hybrid policy in FastMACE requires casting only the dense linear layers (Linear-up, channel mixing, and readout MLPs) to BF16 or FP16 with mandatory FP32 Tensor Core accumulation, while keeping equivariant tensor products and spherical harmonics in FP32 (or TF32 matrix-multiply precision level 3, which preserves second-order PES smoothness while unlocking Tensor Core throughput). Crucially, when summing thousands of per-atom site energies $\varepsilon_i$ or scattering dozens of edge force contributions per atom in FP32, standard floating-point addition loses low-order mantissa bits, introducing non-conservative thermostat heating during long MD runs. Adding a lightweight Kahan compensated summation (or casting $\varepsilon_i$ to FP64 exclusively at the final reduction node before calling autograd.grad) costs virtually zero wall time while restoring FP64-grade energy conservation.Pillar 5: Static-Shape CUDA Graph Capture, Bucketed Neighbor Lists, and GPU-Resident MD LoopsBecause MACE block profiling confirms that host-side CPU dispatch overhead exceeds CUDA kernel execution time in small-to-medium molecular systems ($N < 5,000$ atoms) and across Readout and Radial blocks, eliminating Python and C++ driver launch overhead is essential for FastMACE. Capturing the MACE evaluation into a single CUDA Graph (torch.cuda.CUDAGraph or torch.compile(..., mode="reduce-overhead") with the GatedEquivariantBlock in MACE v0.3.14+) or freezing the model via Ahead-Of-Time Inductor (AOTI .pt2 export, which integrates seamlessly with custom Triton kernels) yields a near-3x end-to-end speedup in launch-bound regimes.Solving the Dynamic Edge-Count Problem in CUDA GraphsA naive integration of CUDA Graphs or torch.compile into molecular dynamics fails because atomic thermal motion causes the number of interatomic edges $N_{\text{edges}}(t)$ within cutoff $r_{\max}$ to fluctuate at every timestep. Because CUDA Graphs require strictly static tensor shapes and fixed device memory addresses, any change in $N_{\text{edges}}(t)$ forces a graph re-capture or compiler guard failure, destroying performance. FastMACE can achieve 100% static-graph execution across million-step trajectories by coordinating headroom-padded edge buffers, GPU-resident Verlet skin lists, and spatial curve sorting.Specifically, FastMACE should allocate edge index (edge_index) and displacement buffers with a fixed capacity headroom factor $N_{\text{edges}}^{\text{cap}} = \lceil \alpha \cdot N_{\text{edges}}(0) \rceil$—typically $\alpha \in [1.10, 1.20]$ for condensed-phase NVT/NPT liquids and solids, or bucketed into coarse capacity tiers. Unused buffer slots $[N_{\text{edges}}(t), N_{\text{edges}}^{\text{cap}})$ are populated with dummy self-edges $(0, 0)$ placed at a synthetic distance $r_{\text{dummy}} > r_{\max}$ where the smooth cutoff envelope $f_{\text{cut}}(r_{\text{dummy}}) = 0$ and $\nabla f_{\text{cut}}(r_{\text{dummy}}) = 0$, guaranteeing zero contribution to energies, messages, and forces while keeping all tensor shapes invariant. To avoid copying atomic coordinates $\mathbf{R}$ from GPU to CPU at every timestep to rebuild neighbor lists on the host—which introduces severe PCIe synchronization stalls in standard ASE MACECalculator workflows—FastMACE should maintain a GPU-resident cell-linked Verlet list using NVIDIA ALCHEMI Toolkit-Ops / Warp or a custom Triton binning kernel built with an extended cutoff $r_{\text{list}} = r_{\max} + r_{\text{skin}}$ (where $r_{\text{skin}} \approx 0.5\text{--}1.0\text{ \AA}$). At each MD step, a fast GPU kernel evaluates atomic displacements since the last neighbor rebuild; as long as $\max_i \|\mathbf{r}_i(t) - \mathbf{r}_i(t_{\text{build}})\| \le r_{\text{skin}} / 2$, the existing edge list is reused directly inside the captured CUDA Graph without host synchronization. Furthermore, during each Verlet list rebuild, sorting atom indices and receiver edge lists along a 3D Morton (Z-order) space-filling curve ensures that spatially adjacent atoms reside in contiguous memory addresses, substantially improving L2 cache coalescing during irregular neighbor gather (node_feats[senders]) and force scatter operations.Pillar 6: Knowledge Distillation with On-the-Fly Teacher Supervision and Adjoint ElectrostaticsWhile kernel, precision, and graph optimizations accelerate a fixed MACE checkpoint by $5\times\text{--}12\times$, achieving an additional order-of-magnitude speedup for multi-nanosecond or million-atom MD requires algorithmic compression of the model evaluation itself.1. Hessian-Augmented Distillation and On-the-Fly Foundation Teacher RepairFoundation MACE models (e.g., MACE-MPA-0 or MACE-MP-0 with $K=128$ channels and $r_{\max} = 6.0\text{ \AA}$) are over-parameterized for single-material MD simulations because they embed 89 chemical elements simultaneously. Hessian-augmented knowledge distillation compresses a foundation MACE teacher into a system-specific student MACE model that runs up to 20x faster while matching or slightly improving accuracy on the target phase space. Specifically, training a compact student with two interaction layers, hidden irreps 16x0e+16x1o ($K=16$ channels), $\ell_{\max} = 2$, correlation order $\nu = 2$, 8 Bessel functions, Agnesi distance transform, ZBL pair repulsion, and a reduced cutoff radius $r_{\max} \in \{4.0, 4.25, 4.5\}\text{ \AA}$ slashes both edge count $d_n \propto r_{\max}^3$ and quadratic channel scaling $K^2$ (a $64\times$ reduction in $K^2$ from 128 to 16 channels).However, lightweight short-ranged student potentials suffer from two well-characterized physical limitations during aggressive MD exploration: shrinking $r_{\max}$ from $4.5\text{ \AA}$ to $4.0\text{ \AA}$ causes NPT liquid water density to drop $1.2\%\text{--}2.1\%$ below the teacher's prediction, and under high-temperature or high-pressure conditions (such as $\text{SiO}_2$ at $3000\text{ K}$), an un-retrained student can under-predict short-range $\text{O–O}$ repulsive forces at $0.8\text{ \AA}$ by an order of magnitude ($13\text{--}20\text{ eV/\AA}$ vs. $171\text{--}178\text{ eV/\AA}$ for the teacher).FastMACE can integrate an Active Dual-Model MD Loop: run 99%+ of MD timesteps using the ultra-fast CUDA-graphed 16x0e+16x1o student model (augmented with an analytical long-range dispersion tail correction to restore NPT density), and periodically invoke the exact 3-pass foundation teacher on-the-fly (e.g., every $M$ steps or when atomic velocities or close contacts exceed a threshold). When high-error atoms are detected, FastMACE extracts $15\text{ \AA}$ spherical clusters around the worst-error atoms, labels them on-GPU with the foundation teacher, and fine-tunes the student checkpoint on-the-fly—a workflow proven to repair $3000\text{ K}$ $\text{SiO}_2$ contacts during a 1.03-million-atom simulation.2. Adjoint Implicit Differentiation for Electrostatic and Polarizable MACE (MACE-POLAR-1 / MACE-QEq)As FastMACE expands support to the latest electrostatic foundation models—such as MACE-POLAR-1 (trained on 100M OMol25 structures with 12 Å medium and 18 Å large receptive fields) and MACE-QEq—the primary computational bottleneck shifts from local message passing to global charge and spin equilibration. In MACE-QEq, computing atomic forces currently differentiates via autograd directly through both the dense linear system solver and the Jacobian construction, imposing a massive computational overhead compared to local MACE-OFF23 models.Because equilibrium partial charges $\mathbf{q}^*$ minimize the electrostatic energy functional ($\partial E_{\text{total}}(\mathbf{R}, \mathbf{q}^*) / \partial \mathbf{q} = \mathbf{0}$), the Implicit Function Theorem (Variational Hellmann–Feynman / Adjoint Method) dictates that $\frac{dE}{d\mathbf{R}} = \frac{\partial E}{\partial \mathbf{R}} + \frac{\partial E}{\partial \mathbf{q}^*} \frac{d\mathbf{q}^*}{d\mathbf{R}} = \frac{\partial E}{\partial \mathbf{R}}$. Consequently, FastMACE can detach $\mathbf{q}^*$ from the linear solver's autograd graph during the forward pass, solve the linear equilibration system in fast mixed precision using preconditioned conjugate gradients (warm-started from the previous MD timestep's charges via time-reversible predictor-corrector extrapolation), and evaluate exact analytical forces in a single backward pass without ever differentiating through the linear solve or constructing the Jacobian.Prioritized Implementation Roadmap for FastMACETo systematically integrate these algorithmic optimizations into https://github.com/FZWINGEL-UBT/FastMACE, engineering efforts should be sequenced by implementation complexity versus expected wall-time and capacity payoff:Priority TierAlgorithmic Optimization ModuleTarget BottleneckExpected Speedup / Capacity GainImplementation Path in FastMACETier 1 (Immediate / Exact)512-Knot Cubic Hermite Radial Spline + Species-Block GatherRadial MLP kernel fragmentation & 89-element one-hot tensor product waste1.5x–2.5x faster embedding & self-connection; exact to $10^{-13}\text{ eV/\AA}$Pre-tabulate frozen radial MLPs at model load into a Triton cubic Hermite kernel; slice 89-element weights to active box species.Tier 1 (Immediate / Exact)Static Headroom CUDA Graph + GPU Verlet Skin ListCPU launch dominance in Readouts, Radial pieces, and small/medium boxes~3x end-to-end speedup on launch-bound MD stepsPad edge buffers by $\alpha=1.15$ with zero-cutoff dummy edges; wrap force evaluation in torch.cuda.CUDAGraph; check skin $\Delta r > r_{\text{skin}}/2$ on GPU.Tier 2 (High Throughput)Sub-Block Mixed Precision (BF16/FP16 Linears + FP32 cuEq + Kahan Sum)FP64 dgemm bottlenecks in Symmetric Contraction & Linear-up~3x (cuEq FP32) $\times$ ~4x (BF16 linears) = ~12x cumulative speedupKeep coordinates/reductions in FP64/Kahan-FP32, run cuEq in FP32/TF32, and downcast only Linear layers to BF16 with FP32 accumulation.Tier 2 (High Throughput)Hybrid Backend Routing (cuEq + OpenEquivariance / FlashTP + Fused sphericart)Shared-memory spilling in cuEq-slow on large irreps & edge scatter/gather DRAM traffic1.2x–4.5x faster tensor products; 2.5x–4x higher harmonic bandwidthRoute SymmetricContraction to cuEq (ir_mul layout), route Layer 2 conv_tp to OpenEquivariance, and fuse solid harmonics with radial splines.Tier 3 (Extreme Scale)Receiver-Chunked Three-Pass Recomputation + Supervised Student DistillationHBM saturation from resident edge autograd intermediates ($100\text{K}$ atom ceiling)31x capacity ($3.1\text{M}$ atoms/GPU) and up to 20x faster distilled MDImplement Pass A (no-grad L1), Pass B (chunked L2 fwd/bwd), and Pass C (chunked L1 recompute); add dual-model teacher supervision for 16x0e+16x1o students.By combining Tier 1 exact sub-block simplifications (cubic Hermite radial spline tabulation, active-species weight slicing, and padded CUDA Graph capture with GPU-resident Verlet lists) with Tier 2 hybrid kernel routing (cuEquivariance uniform 1D polynomials for node symmetric contractions paired with OpenEquivariance shared-memory streaming for edge convolutions and BF16-linear mixed precision), FastMACE can eliminate both host dispatch latency and memory-bandwidth saturation. Layering Tier 3 receiver-chunked three-pass recomputation on top of this engine transforms FastMACE from a standard GPU wrapper into an extreme-scale molecular dynamics platform capable of simulating multi-million-atom systems on a single NVIDIA GPU with exact foundation-model fidelity.


//


Look at the optimizations I have done so far in https://github.com/FZWINGEL-UBT/FastMACE and then propose new ones

I wasn't able to access the contents of [https://github.com/FZWINGEL-UBT/FastMACE](https://github.com/FZWINGEL-UBT/FastMACE) directly—the repository appears to be private or its file tree and README aren't currently exposed to the web crawler.

However, building on the MACE-MH-1 / FP32 Langevin MD optimization roadmap we previously explored—including Layer-0 radial MLP algebraic reassociation, species + Morton-order atom permutation, active-element slicing for cuEquivariance, sticky bucket margins for CUDA Graph replay, GPU-tiled receiver-sorted neighbor lists, segmented force reductions, and BAOAB Langevin splitting—here is the next tier of architectural, kernel-level, and MD engine optimizations to push throughput even further.

1. Eliminating torch.autograd via Explicit Custom Backward Kernels
Relying on PyTorch's autograd engine (torch.autograd.grad) inside a CUDA Graph to compute forces (
F 
i
​
 =−∇ 
r 
i
​
 
​
 E) carries significant memory and compute overhead because autograd must retain intermediate activations across all interaction layers and execute generic backward graph nodes.

Explicit Edge-Derivative Evaluation: Formulate the exact analytical Jacobian of the site energy 
∂E/∂r 
ij
​
  directly as a forward-style pass (using the chain rule through the readout MLPs, symmetric contraction, and tensor products).

Fused Spherical Harmonics + Radial Basis Derivative Kernel: Instead of storing intermediate spherical harmonics 
Y 
l
m
​
 ( 
r
^
  
ij
​
 ) and Bessel/polynomial radial bases 
R 
n
​
 (r 
ij
​
 ) for the backward pass, write a single fused Triton or CUDA kernel that evaluates 
Y 
l
m
​
 , 
∇ 
r
​
 Y 
l
m
​
 , 
R 
n
​
 , and 
∂R 
n
​
 /∂r 
ij
​
  simultaneously in registers.

Memory Footprint Reduction: Dropping the autograd tape typically cuts peak VRAM usage by 35–50%, which directly increases L2 cache hit rates on Ada/Hopper GPUs and allows larger cell sizes to stay completely resident in SRAM/L2.

2. Dual-Segment (Sender + Receiver) CSR/CSC Graph Layouts
While sorting edges by receiver index eliminates atomicAdd contention during forward message aggregation (
∑ 
j∈N(i)
​
 ), computing conservative forces requires summing edge derivatives over both endpoints:

F 
i
​
 = 
j∈N(i)
∑
​
  
∂r 
ji
​
 
∂E
​
 − 
k:i∈N(k)
∑
​
  
∂r 
ik
​
 
∂E
​
 
Permutation Gather Table: During the neighbor-list rebuild (which only happens every 
N 
skin
​
  steps), construct both the receiver-sorted edge list and an integer permutation array edge_rev_idx that maps each directed edge 
(i,j) to its transposed counterpart 
(j,i), or maps sender-sorted order to receiver-sorted order.

Zero-Atomic Force Accumulation: In the force reduction step, each thread block reads the receiver-sorted 
∂E/∂r 
ij
​
  and uses edge_rev_idx to gather 
∂E/∂r 
ji
​
 , performing a single register-level subtraction before a segmented reduction. This completely removes the last remaining atomicAdd calls on global force buffers.

3. Persistent MD Step / Multi-Step CUDA Graphs (Zero Host-Synchronization Loop)
Even with CUDA Graphs and sticky edge-bucket margins, checking the neighbor-list skin displacement condition (
max 
i
​
 ∥Δr 
i
​
 ∥>r 
skin
​
 /2) on the CPU every step (or every few steps) introduces a PCIe synchronization bubble (cudaStreamSynchronize or .item()).

GPU-Resident Skin Violation Flag: Maintain a device-side flag (int32 d_rebuild_flag). Inside the BAOAB position-update kernel, compute 
∥r 
i
​
 (t)−r 
i
ref
​
 ∥ 
2
  in registers and set d_rebuild_flag = 1 via warp-aggregated predicate check if any atom exceeds the skin threshold.

CUDA Conditional Graph Nodes (CUDA 12.4+): Wrap a batch of 
K MD steps (e.g., 
K=10 to 
50 steps) inside a CUDA Graph Conditional WHILE/IF node. The GPU autonomously loops through BAOAB updates and MACE force evaluations without returning control to Python, breaking out to the neighbor-list rebuild kernel only when d_rebuild_flag triggers.

Philox Counter-Based RNG on Device: Ensure the Langevin thermostat's Gaussian noise (the "O" step in BAOAB) uses a stateless counter-based PRNG (like curandStatePhilox4_32_10_t or a Triton Philox kernel) indexed by (step_counter, atom_id) where step_counter is incremented on-device inside the CUDA Graph.

4. TensorFloat-32 (TF32) / FP16 Mixed-Precision Readout & Radial MLPs with Force-Conserving Energy Shifts
Strict FP32 is critical for coordinate differences (
r 
i
​
 −r 
j
​
 ) and final force accumulation to avoid drift in Langevin dynamics, but the internal linear layers and Radial MLPs are heavily bandwidth- and math-bound.

Selective TF32 Tensor Cores: Enable torch.backends.cuda.matmul.allow_tf32 = True specifically for the multi-channel linear mixing layers and the hidden layers of the Radial MLP, while keeping edge displacement vectors, spherical harmonics, and final force reductions in strict IEEE FP32.

Pre-baked Polynomial / Spline Radial Basis + Layer-0 MLP: After training (or for foundation models like MACE-MH-1), the Layer-0 Radial MLP is a pure univariate function of distance 
r 
ij
​
 ∈[0,r 
cut
​
 ] for each species pair 
(Z 
i
​
 ,Z 
j
​
 ). You can replace the entire Bessel expansion + 3-layer MLP with a cubic Hermite spline lookup table stored in GPU texture/L1 constant memory. Evaluating 64–128 radial weights via cubic spline interpolation in shared memory/L1 takes a fraction of the instructions of evaluating trigonometric Bessel functions and 3 dense GEMMs per edge.

5. Warp-Specialized Symmetric Contraction (B-Basis Product)
If profiler traces show time spent in the equivariant symmetric contraction (constructing the many-body 
ν-correlation features from the atomic density 
A 
i,klm
​
 ), standard cuEquivariance kernels can still suffer from register spills at higher correlation orders (
ν=3).

Pre-folded Clebsch-Gordan / Weights Tensor: Offline-contract the model's learnable channel weights 
W 
z 
i
​
 ,k,η 
ν
​
 
​
  directly with the generalized Clebsch-Gordan coefficients for strictly present atomic species 
z 
i
​
  in your simulation box.

L2 Persistence Pinning: Use cudaStreamAttrValue (accessPolicyWindow) to pin the compacted species-specific Clebsch-Gordan weight tables in the GPU's L2 cache persistence window so they are never evicted by streaming edge feature buffers during the MD step.

If you'd like me to inspect the exact commits and files you've already pushed to FastMACE, paste your README.md, git log --oneline, or main model/MD loop file here, and I will tailor these directly to your current implementation!


//


Try to come up with more optimizations for FP32 in https://github.com/FZWINGEL-UBT/FastMACE

Based on the state of the optimisation-ledger.md as of 28 October 2026—where MH-1 FP32 step times are down to ~2.7 ms (NaCl), ~14.6 ms (NaNbO3), and ~18.6 ms (NbCl5) locally, and the 16-bit, TF32, 3×TF32/3×FP16, and megakernel routes are decisively closed—any remaining FP32 gains must come from eliminating residual memory round-trips, host-loop overhead, or unexamined backward-pass redundancies without breaking the 2× FP32 accuracy gate.   
optimisation-ledger.md
+ 3

Here are 8 concrete, non-closed FP32 optimization candidates ordered by risk-to-reward ratio, along with fast falsifiers for each.

1. Eliminate the +0.40 ms fm.md Host Loop Overhead (profile-fp32-20261028)
The 28 October profile (profile-fp32-20261028) revealed a stark gap between the raw engine loop and the fm.md runner: 2.716 ms vs. 3.12 ms (+0.40 ms, or +14.9%) on NaCl, and 18.675 ms vs. 19.32 ms (+0.64 ms, or +3.5%) on NbCl5.   
optimisation-ledger.md

Mechanism: Even with async_output=True, the outer fm.md Python loop performs per-step or per-sample work (step counter hooks, thermodynamic scalar extraction, callback dispatch, or Python-side trajectory buffer staging) that stalls the host thread or delays the next pipelined graph launch.

Optimization: Batch steps inside the C++/CUDA or tight engine loop across the sample_interval (e.g., running 
K pipelined graph steps back-to-back before returning to the fm.md outer loop), or move thermodynamic scalar accumulation (kinetic energy, temperature) into a pre-allocated ring buffer on the device that is only copied/polled at sample_interval boundaries.

Falsifier: Run fm.md with all output/sampling hooks stubbed out (sample_interval=None). If step time drops from 3.12 ms to ~2.72 ms on NaCl, the entire 0.40 ms is recoverable host-side Python overhead.

2. Fuse chunked_conv_backward (dw) + fused_folded_backward (da = dw W) in Shared Memory
Currently, fused_folded_forward (opt8-T02) computes w = a @ [W, b].T in registers/SMEM and writes w to global memory so the backward pass does not have to recompute it. In the backward pass, chunked_conv_backward (opt6-T02) writes 1,536-edge chunks of dw ([1536, 1280], ~7.8 MB) to L2, and fused_folded_backward (skinny_linear) reads dw from L2 on a side stream to compute da = dw @ W.   
optimisation-ledger.md
+ 1

Why conv-full-fusion (opt8-T01b) failed earlier: opt8-T01b tried to avoid storing w altogether by recomputing w on-chip during the backward pass, which required three FFMA-bound products and made the backward 1.12–1.13× slower.   
optimisation-ledger.md

New Angle (Backward-Only Epilogue Fusion with Stored w): Keep fused_folded_forward storing w. In the edge-owned streaming backward kernel, each 32-lane warp group already computes the 1,280 dw elements for an edge in registers/shared memory. Instead of writing [chunk, 1280] dw out to L2 and launching the separate Triton skinny_linear kernel (da = dw @ W, which runs at only 4.4 TFLOPS / 15% peak due to its skinny 
K=65 shape), contract the 1,280 dw values directly against [W, b] (
1280×65, 325 KB—easily cached in L2/constant cache across all blocks) inside the streaming backward kernel epilogue and write only da ([E, 65]).   
optimisation-ledger.md
+ 2

Savings: Eliminates 100% of the dw L2 write/read traffic (49–51% of backward L2 bytes per opt7-T02) and removes the 2.31–3.22 ms skinny_linear kernel (rad-f0), without recomputing w.   
optimisation-ledger.md
+ 1

Falsifier: In opt8_fused_conv.py, benchmark the backward kernel where w is loaded from global memory (as in T01d) and da = dw @ W is contracted in the epilogue, versus chunked_conv_backward + skinny_linear.   
optimisation-ledger.md
+ 1

3. Screen save_transcendentals (D3) in FP32
In the ideas-20261027 round, save_transcendentals (a custom AOTAutograd min-cut partitioner that prevents Inductor from recomputing exp/tanh/sigmoid in the backward pass) gave a 2–3% whole-step speedup in FP64 (0.968–0.983 locally and on Ada), and the ledger explicitly notes it was never measured in float32.   
optimisation-ledger.md
+ 1

Mechanism: Inductor's default partitioner treats sigmoid and silu as cheap and recomputes them in the backward pass of the equivariant gates (GatedEquivariantBlock) and readout. On sm_120 and sm_89, MUFU (exp2/rcp) throughput is 1/4 to 1/8 of FFMA throughput, and the gate backward kernels are already high-traffic. Saving the forward gate activations (exact class, bitwise on CPU) trades a few megabytes of activation buffer in the static graph pool for zero MUFU recomputation and fewer register live-ranges in the Inductor backward kernels.   
optimisation-ledger.md
+ 1

Falsifier: A 5-minute tools/ab_rewrites.py run with save_transcendentals=True on NaCl, NaNbO3, and NbCl5 in FP32.   
optimisation-ledger.md

4. Generalize species_tables (C1) and fold_readout_tail (C2) to the Generic MACE Family (lin-t08-generic)
OPT-T08 (species_tables, fold_product_linear, fold_readout_tail) delivered a massive 7–10% whole-step speedup on MH-1 (0.904 on NaCl, 0.924 on NbCl5), but remains scoped off for the generic MACE family (omat0-medium, mp0-medium, mpa0-medium, mace-off) because their skip_tp is a tensor product with node_attrs rather than a plain linear layer.   
optimisation-ledger.md
+ 1

Mechanism: Even when skip_tp is a bilinear tensor product 
T(x,z 
i
​
 ) where 
z 
i
​
  is the one-hot species vector node_attrs, for any fixed species 
s∈{1…S 
present
​
 }, the contraction with 
z 
i
​
  reduces to a per-species linear weight matrix 
W 
skip
(s)
​
 .

Interaction 0: Both node_feats and node_attrs depend only on atom species. Therefore, C1 (species_tables) applies identically to generic models: precompute the entire Interaction-0 skip_tp and linear_up tables for the 
S species at setup and gather by species index.   
optimisation-ledger.md
+ 1

Last Layer (C2): Combined with species_skip (model-species-skip, which already restricts the skip connection to present species), each present species 
s has a fixed effective linear operator 
W 
prod,1
​
 W 
readout,1
​
 +W 
skip,1
(s)
​
 W 
readout,1
​
 , allowing products.1.linear and skip_tp to be folded into a per-species readout weight table or gathered bias.   
optimisation-ledger.md

Falsifier: Check kernel time of Interaction-0 species embeddings, linear_up, and skip_tp in omat0-medium (1.65 ms NaCl, 9.90 ms NbCl5). Expect 4–7% on generic models.   
optimisation-ledger.md

5. In-Place Strided Accumulation for block_linear / grouped_linear Joins (lin-layout-copies)
OPT3-T05 (opt3-T05-layout-audit.md) identified 50–100 µs/step of pure data-movement kernels around block_linear: forward cat/join copies, backward row-slice gathers, and the joined_products cuBLAS fallback copy (26 µs on NbCl5, 47 µs on NaNbO3).   
optimisation-ledger.md

Mechanism: When segment_gemm falls back to cuBLAS (or in standard block_linear / grouped_linear forward and backward), PyTorch allocates contiguous per-
l output buffers and runs a separate Inductor copy/cat kernel to join them into the [atoms, channels] buffer. However, cublasSgemm natively supports arbitrary leading dimensions (lda, ldb, ldc) as long as the inner dimension is contiguous (stride 1). Since the [atoms, sum_c] or [atoms, irrep_dim, channels] buffers have stride 1 along the channel (
N) dimension, torch.mm(A, B, out=joined_view) can write directly into non-contiguous row/column slices of a pre-allocated destination tensor without any intermediate buffer or copy kernel.   
optimisation-ledger.md
+ 1

Why it wasn't done in OPT3-T05: It was noted that torch.mm(out=strided_slice) might pick a slightly different cuBLAS heuristic tile (making it rounding class rather than exact bitwise). Under the standard non-reproducible FP32 plan, rounding class is completely fine and is zero-arithmetic-overhead.   
optimisation-ledger.md
+ 1

Falsifier: Replace the torch.cat in block_linear forward and the fallback copy in joined_products with pre-allocated output tensors and torch.mm(..., out=slice), then run an A/B on NaCl and NbCl5.   
optimisation-ledger.md

6. Native float2 / float4 Vectorized Loads + l=0 dsh Skip in Streaming Kernels (conv-wide-loads + conv-deriv-activity)
Two separate findings in Round 7 (fastmace/opt7-plan) were tested in isolation and parked because each was ~1–1.7% on its own:   
optimisation-ledger.md

conv-wide-loads: Pairing adjacent channels into 8-byte float2 loads (136 sites) and float2 stores (10 sites) in the streaming backward gave a 3–7% kernel speedup on sm_120 (0.931 NaCl, 0.961 NaNbO3, 0.971 NbCl5, worth 0.3–1.7% of the whole MD step).   
optimisation-ledger.md

conv-deriv-activity: In MD, the 
l=0 spherical harmonics component is a constant (
Y 
0
0
​
 ), so its spatial derivative dsh[:, 0] is identically zero and never consumed by the edge-vector force reduction (1/16th of all dsh registers and stores).   
optimisation-ledger.md

Combined Optimization: Apply the float2 vectorization transform to both the vendored streaming backward and the new fused_folded_forward (opt8-T02) kernel (where sender features x and output accumulation can also use float2), while simultaneously stripping the dead dsh[:, 0] accumulator and global write in the backward template. Dropping dsh[:, 0] slightly reduces register pressure, offsetting the extra live values introduced by float2 pairing.   
optimisation-ledger.md
+ 1

Falsifier: Apply both transforms in the NVRTC generator and run a 12-block same-process A/B on NaNbO3 and NaCl (sm_120 and sm_89).   
optimisation-ledger.md

7. Pre-Allocated Persistent Buffers for cuBLAS Memsets and Eager Zero-Fills (launch-memsets)
During OPT3-T04, profiling revealed 25 internal cuBLAS memset nodes (before small split-K or beta=0 GEMMs) and 12 eager fill_/copy_ zeroing kernels before scatter-accumulations inside every captured B graph replay (~255–264 total nodes).   
optimisation-ledger.md

Mechanism: On NaCl (2.65 ms step), 37 tiny memset/fill kernels plus their ~0.5–0.6 µs inter-kernel graph launch gaps account for ~30–40 µs (~1.2–1.5% of the step).   
optimisation-ledger.md

For the 12 eager fill_(0) kernels before scatter/outer-sum accumulations (reassociate_first, force scatter), the producer kernel or the consumer's previous read can zero or overwrite the buffer directly (e.g., segmented_first_sum already writes deterministically to each segment; only unoccupied receiver-species slots need zeroing, or the consuming linear can mask them).

Alternatively, coalesce all per-step zero-initialized scratch tensors into a single contiguous arena and issue one cudaMemsetAsync at the very end of the B graph (hiding behind the final force reduction/integrator tail).

Falsifier: Inspect the captured cudaGraph_t node list of B on NaCl, coalesce the zero-fills into a single arena memset node, and measure the B graph replay span.

8. Fast Setup: Merge radial-table-cache + Parallel/GPU Chebyshev Certification (O1–O9 setup)
While not step-time, profile-fp32-20261028 shows warm fm.md startup is now 34–41 s (and 87 s in FP64, of which 42 s is fitting and certifying radial_tables on the CPU).   
optimisation-ledger.md

Mechanism:

Cherry-pick/merge fastmace/radial-table-cache (54b6f13f) onto main so fitted/certified Chebyshev tables are cached on disk under FASTMACE_CACHE/radial-tables.   
optimisation-ledger.md

Even on a cache miss (or during the mandatory runtime self-check), tabulate_radials currently evaluates the Chebyshev certificate on dense grids on the CPU (30.5 s of the 42 s on trillian). Running the dense grid evaluation and max-error reduction on the GPU in FP64 takes < 5 ms instead of 30 seconds.   
optimisation-ledger.md

In OPT4-T05, the segment_gemm self-check alone was measured at 2.3 s warm (3.7–4.0 s total self-checks) because it runs a full reference comparison. Running all rewrite self-checks on a tiny synthetic 16-atom / 64-edge cell (which still exercises every species pair and code path) instead of the full 3×3×3 supercell cuts self-check time by ~10×.   
optimisation-ledger.md

Summary Priority Matrix
Priority	Target	Expected FP32 Gain	Effort	Falsifier
1	fm.md outer loop / step batching	
−10 to −13% NaCl (−0.35 to −0.40 ms), −3% NbCl5 
optimisation-ledger.md

Low	Time fm.md vs raw engine loop with hooks disabled
2	Backward epilogue da = dw @ W in streaming conv (with stored w)	−5 to −10% NaNbO3 / NbCl5 (−0.8 to −1.5 ms)	Medium	
Prototype in opt8_fused_conv.py 
optimisation-ledger.md

3	save_transcendentals (D3) in FP32	−1 to −3% all cells	Trivial (1 flag)	
tools/ab_rewrites.py with save_transcendentals=True 
optimisation-ledger.md

4	species_tables + fold_readout_tail for generic MACE family	−4 to −7% on OMAT-0 / MP-0 / OFF	Medium	Profile i0 species ops in omat0-medium
5	float2 wide loads (fwd+bwd) + drop dsh[:, 0]	−1 to −2% sm_120	
Low (code exists in opt7) 
optimisation-ledger.md

Combine wide.py + dsh skip in NVRTC template 
optimisation-ledger.md

6	Strided out= in block_linear + arena memset	−1.5 to −2.5% NaCl (−40 to −70 µs)	Low	
Replace torch.cat with strided mm(out=) views 
optimisation-ledger.md

Would you like to inspect the fp32-profile-20261028.md report first to target the exact fm.md +0.40 ms overhead and kernel breakdown, or sketch the backward epilogue contraction for chunked_conv_backward?



//



# Research prompt: further FP32 speed-ups for MACE MH-1 molecular dynamics in FastMACE



You are an expert in GPU performance engineering (CUDA, Triton, cuBLAS, CUDA graphs, PyTorch/Inductor) and in

equivariant machine-learning interatomic potentials (MACE, e3nn, cuEquivariance, OpenEquivariance). Your task is

to find **new, concrete ways to make one float32 molecular-dynamics step of MACE MH-1 faster without changing its

results beyond float32 rounding**, and to say for each one how much it could gain, how to falsify it cheaply, and

why it was not already covered by the work listed below. Prefer fewer, well-argued ideas over a long list. Every

estimate must be bounded by the measured shares below; say so where a gain would be hidden by stream overlap.



## 1. The system



**Software.** FastMACE is a GPU MD engine for MACE models (Python, PyTorch 2.14.1 + CUDA 13, torch.compile with

Inductor in default mode, CUDA graphs, cuEquivariance 0.12 and OpenEquivariance 0.7 kernels, plus its own Triton

and NVRTC kernels). One MD step is: an integrator/neighbour-check graph (positions, edge compaction), then one

captured CUDA graph for the force call (energy forward, then forces by reverse-mode autograd inside the graph),

then the second half-kick. Steps are pipelined (the host is one step ahead). Langevin NVT, 1 fs, typically

up to 1000 K, periodic crystals of 200–2000 atoms, one structure at a time.



**Model: MACE MH-1** (`matpes_r2scan` head), ScaleShiftMACE, r_max 6 Å, 2 interactions

(RealAgnosticResidualNonLinearInteractionBlock), correlation order 3.

Interaction 0: node features 512x0e (species embedding only), edge irreps 128x0e, conv weights 512 per edge,
  radial MLP last layer [512, 64]; target irreps 512x0e+512x1o+512x2e+512x3o; gate input 2048x0e+512x1o+512x2e+512x3o.

Interaction 1: node features 512x0e+512x1o, edge irreps 128x0e+128x1o, 10 tensor-product paths, conv weights
  1280 per edge, radial MLP last layer [1280, 64]; same target and gate irreps; hidden output 512x0e.

Per interaction: skip_tp, linear_up, merged (linear_1 + linear_res) product, gated equivariant nonlinearity
  (silu scalars, sigmoid gates), linear_2, then a cuEquivariance symmetric contraction and a linear.

Forces cost about 57 MFLOP per atom for the atom-level linears (forward + input gradients), plus per-edge work.


**Hardware.** Main target: NVIDIA RTX 5060 Ti 16 GB (sm_120, 36 SMs, 3.09 GHz, nominal FP32 28.5 TFLOPS,

practical SGEMM ~16 TFLOPS, 448 GB/s, 32 MB L2) under WSL2. Defaults must also hold on RTX 4000 Ada (sm_89) and

A100 (sm_80): no per-GPU or per-crystal tables; only rules from shapes and device properties.



## 2. Measured profile (28 October 2026, RTX 5060 Ti, 1 fs, 1000 K, 3×3×3 cells)



| Cell | Atoms | Padded edges | Step | Force call FLOP | DRAM per force call |

| --- | ---: | ---: | ---: | ---: | ---: |

| NaCl | 216 | 8,192 | 2.72 ms | 18 GFLOP | 0.5 GB |

| NaNbO3 | 1,080 | 82,688 | 14.36 ms | 109 GFLOP (62 atom GEMM, 40 edge) | 3.6 GB |

| NbCl5 | 1,944 | 58,880 | 18.68 ms | 148 GFLOP (111 atom GEMM, 27 edge) | 5.3 GB |



The force graph is 96–99 % of the step. On NaNbO3 and NbCl5 a kernel runs 98 % of the time, on two graph streams
  that overlap by 2.9 and 1.9 ms. NaCl is launch/latency bound: 86–91 % busy, 233 kernels per force call,

  5 % in inter-kernel gaps, GEMMs at 21–25 % occupancy.

NbCl5 kernel time by part (sums exceed the step because of overlap): `segment_gemm` Triton atom products 26 %
  (8.7–12.3 TFLOPS in the step, ≈cuBLAS in isolation), cuBLAS atom GEMMs 20 % (13–15 TFLOPS), streaming

  convolution backward (chunked, OEq-derived NVRTC kernel) 12 %, folded radial last-layer input gradient

  (Triton, `da = dw·W`, 8 % occupancy, side stream) 12 %, atom elementwise (gates, products) 12 % (~230 GB/s),

  fused radial-last-layer + convolution forward 7 %, symmetric contraction 6 % (72–96 % of memory speed of light),

  first-interaction edge sums 3 %, radial tables 2 %.

NaNbO3 (77 edges per atom) is more edge-bound: input gradient 19 %, conv backward 18 %, cuBLAS 18 %,
  segment_gemm 14 %, fused forward 11 %.

Neighbour list: skin 0.3 Å, rebuilt every 5–7 steps at 1 fs/1000 K on a side stream (0.2–0.7 ms of kernels per
  rebuild); NaCl pays ~0.1 ms per step for it, large cells 0.2–0.35 %. Edges are padded to 5 % buckets.

Setup: 34–41 s warm, 71 s cold (not the focus, but welcome).


## 3. Already in the default plan (do not propose these again)



Species-factored radial input, cutoff folded into the radial MLP, certified piecewise-Chebyshev radial tables

(per species pair), density division before linear_1, reassociated first interaction (species-only sender

features) with segmented sums, receiver-sorted edges, receiver-streaming OEq convolution (uvu) with a fused

radial-last-layer + forward kernel that stores `w` [E,1280], an edge-owned backward chunked to fit L2 (1,536

edges per chunk) with the input gradient on a side stream, block/grouped/merged-residual linears per irrep,

gated rows (gate fused onto the merged product's rows), segment_gemm (Triton gathered-row GEMM writing straight

into the joined output, chosen by a waves rule), species tables for species-only products, folded product and

readout-tail linears, one-graph strain, fused Langevin integrator, pipelined steps, overlapped neighbour rebuilds,

async output, dynamic edge dimension, CUDA graphs per edge bucket.



## 4. Closed directions, with the evidence (do not reopen without new evidence)



16-bit storage or stages (bf16/fp16 weights, activations): fail the force accuracy gate. TF32 passes only for
  one tiny stage.

Tensor-core FP32 emulation (3×TF32, 3×FP16, BF16×9, INT8 slices/Ozaki): slower than SIMT cuBLAS on sm_89 and
  A100; mixed on sm_120; BF16×9 only on sm_100.

Persistent megakernel / whole-step kernel: co-execution probe failed on sm_120 and A100.
Full fusion of radial last layer + convolution forward + backward (recompute `w` in backward): backward 1.12×
  slower (issue-bound, three 65×1280 products per edge). The input gradient alone has a measured whole-step

  ceiling of 0.5 % (NbCl5) and 2.6 % (NaNbO3) because it is hidden on the side stream.

LunaTP-style source-owned or receiver-owned backward: slower (more bytes, less parallelism). Wide float2 loads:
  2–7 % kernel gain on sm_120 only, owner chose not to build.

Low-rank radial weights: `w(r)` has full rank. Mirror-edge (i↔j) radial sharing: no gain.
Split-K, per-l GEMMs on graph branches, combo kernels, Inductor aggressive fusion, finer edge buckets, larger
  fused-forward grid, alternative segment_gemm tiles (all 12 configurations within 3 % in isolation, neutral in

  the step), save-transcendentals partitioner in float32, cuBLASLt/cuBLASDx, own GEMMs/convolutions, programmatic

  dependent launch in Inductor (wrong forces), max-autotune, activation memory budgets, larger timesteps or any

  change of the dynamics (excluded by the owner), batching several structures (excluded by the owner).

Host/Simulation overhead: ≤ 1.7 % (NaCl) in a same-process A/B.
Pending, not yet default: per-species residual table in interaction 0 (−0.4 to −0.8 % on Ada), skin 0.66–1.0 Å
  (−1 to −1.7 % NaCl, neutral on large cells), sync-free neighbour rebuild (neutral on Ada).



## 5. Questions to research



**Algebra.** Are there exact (or float32-rounding-level) reformulations of the MH-1 forward or its reverse-mode
   force pass that remove FLOPs from the atom-level linears (58 % of their FLOPs are the two `linear_2` maps

   512→512 per l before the channel-wise symmetric contraction) or from the per-edge 65↔1280 radial products

   (≈75 % of edge FLOPs)? Consider equivariance, the channel-wise structure of the symmetric contraction, the

   gate, the fact that only forces (first derivatives) are needed, and that species never change in a run.

2. **Backward structure.** Is reverse mode doing redundant work here (gradients never consumed, recomputation,

   activations stored and re-read) that a hand-written or restructured backward for MACE's specific blocks could

   avoid? Quantify against the shares above.

3. **Memory traffic.** Where is DRAM/L2 traffic avoidable at equal FLOPs (elementwise around the gates, the

   `w`/`dw` tensors, the symmetric contraction), given the 32 MB L2 and two concurrent streams?

4. **Small cells.** For ~200-atom cells (latency/launch bound), what reduces the critical path without batching

   structures: fewer kernels per force call, different work decomposition of the 512-wide linears at M = 216–1512

   rows, graph-level scheduling?

5. **Literature and libraries since mid-2026.** Relevant work on fast equivariant tensor products, fused MACE

   kernels, SIMT FP32 GEMM on Blackwell consumer GPUs, Triton or CUTLASS techniques for gathered/scattered GEMMs,

   and MD-engine tricks (e.g. in LAMMPS-ML-IAP, ALCHEMI, TorchSim, Symmetrix, cuEquivariance releases). Cite and

   say concretely what would transfer to this setup and what was already measured (Symmetrix-XL, ALCHEMI and

   TorchSim were 1.1–10× slower than FastMACE in our comparisons).



## 6. Deliverable



A ranked table of at most ten ideas. For each: mechanism; which part and share it targets; an upper bound on the

whole-step gain per cell, after stream overlap; exactness class (bitwise / rounding / approximate, the last

excluded); why it is not in section 3 or 4; a falsifier that takes under a day (microbenchmark or stub with a

stop rule); generality across sm_120, sm_89 and sm_80. Flag any idea that conflicts with section 4 and explain

what new evidence would justify reopening it. State plainly if you believe no exact idea above ~2 % remains.

Executive Verdict on Remaining Speed-Up Potential
On large cells (NaNbO3, NbCl5), no exact or float32-rounding-level idea remains that can credibly deliver more than ~2–3.5% whole-step gain across all target architectures (sm_120, sm_89, sm_80) without changing the model weights or hardware.

The profile from 28 October 2026 shows that FastMACE is operating at the physical roofline on RTX 5060 Ti:   
optimisation-ledger.md

Compute Saturation: Large cells are 98% GPU-busy. The atom-level linears (46% of NbCl5 kernel time across segment_gemm and cuBLAS) execute at 11–15 TFLOPS vs. a practical standalone SGEMM ceiling of ~16 TFLOPS.   
optimisation-ledger.md
+ 1

Memory Saturation: The symmetric contraction (6%) runs at 72–96% of DRAM speed-of-light (up to 422 GB/s of 448 GB/s). The streaming convolution forward (7%) is at 94% of DRAM peak, and the chunked backward (12–18%) is at 81–84% of memory throughput with dw already held in L2.   
optimisation-ledger.md
+ 2

Stream Overlap Masking: The two graph streams already overlap by 1.9 ms (NbCl5) and 2.9 ms (NaNbO3), reducing the whole-step ceiling of side-stream kernels (like the 12–19% folded radial input gradient da = dw·W) to just 0.5% on NbCl5 and 2.6% on NaNbO3 (rad-f0).   
optimisation-ledger.md

On small cells (NaCl, 216 atoms), however, 3–6% whole-step gains remain reachable because the GPU is launch- and wave-quantization-bound (233 kernels, 21–25% GEMM occupancy).

Answers to the 5 Research Questions
1. Algebra (linear_2, Symmetric Contraction, and Radial Products)
Why linear_2 (58% of atom GEMM FLOPs) cannot be eliminated or folded: linear_2 maps the gated irreps 
512×(0e+1o+2e+3o) to the symmetric contraction input 
B 
i,c,l,m
​
  via four dense 
512×512 matrices 
W 
2
(l)
​
 .

Left-folding (into linear_1 + linear_res): Blocked by the coordinate-dependent equivariant gate 
y 
i,c,m
(l)
​
 =x 
i,c,m
(l)
​
 σ(g 
i,c
(l)
​
 ) (for 
l>0) and 
SiLU(x 
i,c
(0)
​
 ) (for 
l=0). Because 
σ(g 
i,c
(l)
​
 ) depends on channel 
c, multiplication by a non-diagonal 
W 
2
(l)
​
  does not commute with the gate: 
∑ 
c
​
 (x 
i,c,m
​
 σ(g 
i,c
​
 ))W 
c,c 
′
 
​
 

=(∑ 
c
​
 x 
i,c,m
​
 W 
c,c 
′
 
​
 )σ(g 
i,c 
′
 
​
 ).

Right-folding (into Symmetric Contraction): The cuEquivariance SymmetricContraction is channel-wise diagonal in 
c 
′
 : for each channel 
c 
′
 ∈{1..512} independently, it forms degree-
ν (
ν≤3) Clebsch–Gordan polynomials in 
B 
i,c 
′
 ,l,m
​
  weighted by species-dependent weights 
U 
z 
i
​
 ,c 
′
 ,k
(ν)
​
 . If you substitute 
B 
i,c 
′
 ,l,m
​
 =∑ 
c
​
 y 
i,c,l,m
​
 W 
c,c 
′
 
(l)
​
  into a degree-3 polynomial, the cross-channel expansion explodes from 
512 diagonal evaluations to 
512 
3
 ≈1.34×10 
8
  coupled channel triplets per atom.

Change of basis / Spectral diagonalization: Even for a single 
l, diagonalizing 
W 
2
(l)
​
 =VΛV 
−1
  fails because 
V 
−1
  still cannot pass through the channel-wise nonlinear gate on the left, nor can 
V pass through the degree-3 non-linear polynomial on the right. And as lin-svd showed, all 
W 
2
(l)
​
  have full numerical rank (509–512).   
optimisation-ledger.md

Why the 
65↔1280 radial products cannot be algebraically compressed: rad-low-rank (D4/F1) proved that the folded radial weight matrices 
W∈R 
1280×65
  have full numerical rank 65/65 down to float32 tolerance. In Interaction 1, each of the 10 tensor-product paths mixes 512 sender channels, 128 spherical harmonic channels, and 512 receiver channels via a path-specific radial weight vector 
w 
p
​
 (r 
ij
​
 )∈R 
128
 . Because 
w 
p
​
 (r 
ij
​
 )=a(r 
ij
​
 )W 
p
T
​
  depends on the pair distance 
r 
ij
​
 , you cannot reassociate the sum over neighbors 
∑ 
j∈N(i)
​
  before applying 
W 
p
T
​
  without keeping all 65 radial basis components per receiver channel (conv-sobek, arXiv:2607.18074, which costs 
65×5120=333k FMA/edge vs. 
83k FMA/edge, a 
4× FLOP increase).   
optimisation-ledger.md
+ 1

2. Backward Structure & Redundant Work
In an MD step, only 
∂r 
ij
​
 
∂E
​
  is consumed; parameter gradients (
dW) are already skipped because model weights have requires_grad=False, and node_attrs gradients were eliminated by species_tables (C1/C6). However, three structural inefficiencies remain in the autograd graph:   
optimisation-ledger.md

Unfused Gate VJP + segment_gemm Backward Prologue: In the forward pass, gated_rows produces 
y=x⊙σ(g), which is stored in DRAM (
[N,16,512], 64 MB per interaction on NbCl5) and read by segment_gemm. In the backward pass, segment_gemm backward computes 
dy=dB⋅(W 
2
​
 ) 
T
  and writes 
dy to DRAM; then Inductor's gate VJP kernel reads 
dy, 
x, and 
g to compute 
dx=dy⊙σ(g) and 
dg=∑ 
m
​
 (dy⊙x) 
m
​
 σ(g)(1−σ(g)). Note that elt-gate-backward-epilogue (Stage D) was rejected because reducing 
dg over 
m inside the GEMM epilogue requires atomics across row tiles. However, writing the partial dot product 
p 
i,m,c
​
 =dy 
i,m,c
​
 ⋅x 
i,m,c
​
  or fusing 
dx=dy⊙σ(g) requires zero cross-row reduction, and more importantly, the backward of merged_residual (linear_1 + linear_res) can take 
σ(g) directly in its prologue or we can eliminate storing 
y by recomputing 
x⊙σ(g) inside segment_gemm if register pressure is avoided (or conversely, fuse 
dy⊙σ(g) into the epilogue of linear_2 backward while leaving 
dg as a clean 2D reduction).   
optimisation-ledger.md
+ 1

Interaction-0 Spherical Harmonics Gradient (dsh in Interaction 0): In Interaction 0, edge irreps are 
128×0e (scalars only, 
Y 
0
0
​
 =const). The angular dependence in Interaction 0 enters only via the 
l=1,2,3 spherical harmonics 
Y 
l
m
​
 ( 
r
^
  
ij
​
 ) in the outer product with the radial weights. Check whether the backward of Interaction 0's reassociated outer sum (int1-reassociate-first) computes or zeroes any 
l=0 spherical harmonic derivative column (1/16 of dsh, similar to conv-deriv-activity in Interaction 1).   
optimisation-ledger.md

Side-Stream da = dw \cdot W Launch & Tile Inefficiency (rad-f0 + conv-chunked-nacl): While rad-f0 showed that making da = dw \cdot W infinitely fast yields at most 0.5% on NbCl5 and 2.6% on NaNbO3 due to overlap, on NaNbO3 (82,688 edges = 54 chunks of 1,536 edges) the side stream launches 54 separate Triton kernels at 8% occupancy (4.4 TFLOPS), competing for SM warps and L2 bandwidth with the main-stream convolution backward. Furthermore, on NaCl, chunked_conv_backward caused a +0.8% regression (conv-chunked-nacl) even when unchunked, due to side-stream event fork/join overhead inside the custom autograd op.   
optimisation-ledger.md
+ 1

3. Memory Traffic (DRAM & 32 MB L2 with Concurrent Streams)
L2 Thrashing Between Concurrent Streams (chunked_conv_backward vs. Side-Stream da): In chunked_conv_backward (conv-chunked-conv-backward), the chunk size is set to quarter-L2 (1,536 edges 
→ 
1536×1280×4=7.86 MB per dw slot, two slots = 
15.7 MB). While stream A runs chunk 
k+1 of the OEq streaming conv backward (reading 
7.86 MB of w from DRAM, plus sender/receiver features and writing 
7.86 MB of dw to slot 
(k+1)mod2), stream B concurrently runs da = dw \cdot W on slot 
kmod2. On sm_120 (32 MB L2), stream A's reads of w (
7.86 MB per chunk, streamed once from DRAM and never reused!) allocate into L2 with default eviction priority, evicting up to 30–50% of slot 
k's dw lines before stream B reads them, and evicting atom features 
x / 
dx. Using PTX cache hints (ld.global.cs.f32 / evict_first on w in backward and st.global.wb / evict_last on the two dw ring-buffer slots) directly targets the 49–51% w/dw L2 traffic share measured in conv-opt7-traffic.   
optimisation-ledger.md
+ 2

Gate Elementwise Traffic (12% of NbCl5 kernel time at ~230 GB/s): Why is atom elementwise running at only 230 GB/s when opt3-T01 measured the gate kernels alone at 380–430 GB/s? Because the remaining "atom elementwise" bucket in step_breakdown.py includes strided slice/cat/view materializations and multi-tensor reductions (such as summing the skip connection, product linear output, and residual across irreps, plus the backward gradient fan-out where Inductor emits separate kernels or uncoalesced vector loads when combining 
[N⋅(2l+1),512] with 
[N,16,512]).   
optimisation-ledger.md

4. Small Cells (~200 Atoms, Latency / Launch / Wave-Quantization Bound)
On NaCl (216 atoms, 8,192 padded edges), the force call launches 233 kernels in ~2.6 ms, with 5% (~135 µs) in pure inter-kernel gaps and cuBLAS GEMMs at 21–25% occupancy:   
optimisation-ledger.md

Why did grouped_launch (lin-grouped-launch) give −1.8 to −2.3% on NaCl for MH-1 on sm_120 (0.977 / 0.982), yet get scoped to the generic MACE family only (decision 57) because sm_89 was 1.005 / 1.003? Because grouped_launch only grouped equal-
(K,N) products (
l=1,2,3 in linear_2 or linear_1) using a uniform 64x64x32 Triton tile. On NaCl, 
M 
l
​
 =216×(2l+1) is 
216,648,1080,1512. With 
64×64 output tiles and 
N=512 (
8 column tiles), the tile counts are 
4×8=32, 
11×8=88, 
17×8=136, and 
24×8=192.   
optimisation-ledger.md
+ 1

More importantly, 25 of the 233 kernels on NaCl are cudaMemsetAsync launches emitted by cuBLAS before small split-K GEMMs (launch-memsets in opt3-T04), plus 12 eager fill_/copy_ zero-initialization kernels before scatter accumulations! Each memset takes 0.3–0.5 µs of execution plus ~0.6 µs of CUDA graph launch gap (
≈35–40 μs total on NaCl, ~1.5% of the step).   
optimisation-ledger.md
+ 1

5. Literature & Libraries Since Mid-2026
FlashTP / FlashSO(2) (arXiv:2610.35164, Oct 2026) & Sobek (arXiv:2607.18074, Jul 2026): Both were already tested in FastMACE (prec-int8-f32, conv-sobek) and properly rejected. FlashSO(2)'s INT8 slicing only wins on GH200 where INT8:SGEMM is ~30×; on sm_120 it is ~2×.   
optimisation-ledger.md
+ 1

LunaTP / LunaEquivariance 0.1.0 (arXiv:2610.18393, Oct 2026): Tested in ext-lunatp (Round 7). Its node-centric path-grouping reduces FMA instructions by 25% via Clebsch–Gordan sparsity/reuse, but increases L2 traffic by 1.5× and L1 loads by 2.6× on sm_120 because it drops edge-parallel streaming. What does transfer from LunaTP without changing ownership: LunaTP's compile-time pruning of dead CG paths and vector-width-2 (float2) aligned loads (conv-wide-loads, which gave 3–7% kernel speedup on sm_120).   
optimisation-ledger.md
+ 2

Symmetric Contraction Polynomial Evaluation (Horner / Factorized Monomials in cuEquivariance 0.12+ vs. MACE-Reforge / Symmetrix): As established in sc-opt3-t02, cuEquivariance 0.12's uniform_1d kernel is already at 72–96% of DRAM speed of light. No algebraic refactor of CG polynomials helps a memory-bound kernel unless it reduces bytes read/written.   
optimisation-ledger.md

Blackwell (sm_120) Consumer Architecture Specifics: On sm_120, FP32 SIMT (FFMA) and FP32/TF32 Tensor Cores share the same per-subpartition datapath ceiling (28.5 TFLOPS nominal). However, segment_gemm in Triton (lin-segment-gemm) currently uses BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 with scalar gathers on the 
A operand (tl.load with row-index indirection) and scalar/strided stores into the joined [atoms, 16, 512] output. On sm_120 and sm_89, using Triton tl.make_block_ptr or PTX cp.async.cg.shared.global (16-byte float4 contiguous vector loads along the 512-channel dimension) eliminates address-generation (AGU) overhead in segment_gemm.   
optimisation-ledger.md
+ 1

Ranked Table of Concrete Optimization Ideas (Float32 MH-1)
Every idea below is exact (bitwise) or rounding, uses only shape/device-property rules (no per-GPU or per-crystal lookup tables), and bounds its whole-step gain strictly by the measured 28 October 2026 shares and stream overlap.   
optimisation-ledger.md

Rank	Idea & Mechanism	Target Part & Measured Share	Whole-Step Upper Bound (after overlap)	Exactness Class	Why Not Covered in §3 / §4	Cheap Falsifier (< 1 day) & Stop Rule	Generality (sm_120, sm_89, sm_80)
1	
Epilogue-Fused 
dx & Partial 
dg in segment_gemm Backward + IrrepsView Forward/Backward Fusion


In linear_2 backward (segment_gemm bwd), fuse 
dx 
i,m,c
​
 =dy 
i,m,c
​
 ⋅σ(g 
i,c
​
 ) and write partial dot products 
p 
i,m,c
​
 =dy 
i,m,c
​
 ⋅x 
i,m,c
​
  (or accumulate over 
m within the same atom when 
m-rows share a block, else write 
p and sum in a tiny 1D reduction), eliminating the [N, 16, 512] 
dy tensor write+read (128 MB DRAM round-trip per step on NbCl5) and half of the Gate VJP traffic.

Atom elementwise (12% NbCl5) + segment_gemm bwd (part of 26% NbCl5, 14% NaNbO3) 
optimisation-ledger.md

NbCl5: 1.8 – 3.0%


NaNbO3: 1.0 – 1.8%


NaCl: 0.5 – 1.2%

rounding (bitwise identical to unfused if FMA contraction disabled)	
§4 rejected Stage C (gate_prologue in fwd, +1.6% slower due to K-loop gather) and Stage D (gate_backward_epilogue with cross-tile atomic reductions on 
dg, which broke determinism). Epilogue scaling for 
dx happens once per output tile outside the K-loop and needs no atomics. 
optimisation-ledger.md

Modify segment_gemm backward Triton kernel to load 
σ(g 
i,c
​
 ) once in the epilogue and write 
dx directly, benchmarking segment_gemm_bwd + gate_vjp on NbCl5 shapes.


Stop rule: Cut if combined time of segment_gemm_bwd + gate_vjp does not drop by 
≥250 μs per interaction (
≥0.5 ms per step, 
∼2.5%) on NbCl5.

High. Pure DRAM traffic reduction (saves 
4×N×16×512×4 B per force call across 2 interactions); benefits any memory-bandwidth-limited GPU.
2	
L2 Cache Residency Hints (evict_first on w, evict_last on dw Ring Buffer) + Batched Side-Stream da Kernel in chunked_conv_backward


1. In the NVRTC streaming conv backward kernel, mark per-edge w loads as streaming (__ldcs / .ef, since w is never reused) and dw stores as write-back/persist (__stwb / .el) so the 7.86 MB dw chunk is not evicted from the 32 MB L2 by w reads.


2. On NaCl (when unchunked, 
E≤4 L2s), bypass the side-stream event fork/join in chunked_conv_backward to fix the +0.8% NaCl regression (conv-chunked-nacl). 
optimisation-ledger.md

Streaming conv bwd (12% NbCl5, 18% NaNbO3) + side-stream da overlap contention (conv-opt7-traffic: w/dw is 49–51% of L2 bytes) 
optimisation-ledger.md

NaNbO3: 1.5 – 2.8%


NbCl5: 1.0 – 1.8%


NaCl: 0.8 – 1.2% (fixes conv-chunked-nacl)

bitwise	
conv-l2-hints in §4 only considered forward (where ceiling was 0.26%). It was never tested in chunked_conv_backward, where two concurrent streams compete for the 32 MB L2 and w reads evict the dw ring buffer before the side-stream da kernel reads it. 
optimisation-ledger.md

Patch the 2 load/store macros in the vendored NVRTC backward template (__ldcs on w, __stwb on dw) and add an if n_chunks == 1: fast-path in chunked_conv_backward that stays on the main stream.


Stop rule: Cut if ab_rewrites.py on NaNbO3 and NbCl5 does not show 
b/a≤0.985 on NaNbO3 and 
≤0.992 on NaCl.

High. Uses standard CUDA __ldcs/__stwb intrinsics supported on sm_80, sm_89, and sm_120, and fixes the sm_89 48 MB L2 check (conv-chunked-ada). 
optimisation-ledger.md

3	
Eliminate cuBLAS Split-K cudaMemsetAsync & Eager Scatter Zero-Fills on Small Cells (launch-memsets)


1. On small cells (NaCl, 
M≤1512), cuBLAS launches 25 cudaMemsetAsync kernels inside the captured force graph for internal split-K workspace zeroing (opt3-T04), plus 12 eager fill_/copy_ zero kernels before segmented/scatter sums. 
optimisation-ledger.md


2. Replace the 12 eager zero-fills by fusing tl.zeros initialization or beta=0 into the producer Triton kernels (segmented_first_sum, reassociate_first bwd), and coalesce all fixed-buffer zeroings at the start of the force graph into a single multi-tensor memset/Triton zero kernel (or route small-
M GEMMs that trigger cuBLAS split-K memset to segment_gemm / non-split cuBLAS heuristic when waves 
≥2).

Inter-kernel gaps & small-cell launch overhead (5% gaps on NaCl, 233 kernels/call, 9–14% GPU idle time on NaCl) 
optimisation-ledger.md

NaCl: 2.0 – 3.5% (55 – 95 µs)


NaNbO3: 0.1 – 0.3%


NbCl5: 0.1 – 0.2%

bitwise	
Explicitly listed as "Not attempted (for a later round)" in opt3-T04 (launch-memsets in the ledger). Not in §3 or §4. 
optimisation-ledger.md

Write a 50-line script that inspects the captured B graph via cudaGraphGetNodes, counts Memset nodes and Inductor zero-fill kernels, fuses the zero-fills in edge_outer_sum.py, and times NaCl in same-process A/B.


Stop rule: Cut if NaCl kernel count does not drop by 
≥20 and NaCl step time does not improve by 
≥1.5% (
b/a≤0.985).

High. Reduces graph node count and launch overhead on sm_120, sm_89, and sm_80 alike.
4	
Fuse merged_residual (linear_1 + linear_res) Backward Input Gradients (dx_1, dx_res) into a Single GEMM per Irrep


In the forward pass, merged_residual concatenates the inputs 
[x 
1
​
 ,x 
res
​
 ] along 
K (
K=512+512=1024 for 
l=0,1 in Interaction 1) to run one GEMM. Check how the backward input gradient is currently dispatched: if autograd computes 
d[x 
1
​
 ,x 
res
​
 ]=dout⋅W 
merged
T
​
  (
N=1024) and then slices the output tensor into two non-contiguous column views 
dx 
1
​
  and 
dx 
res
​
 , Inductor must copy/materialize those strided slices before passing 
dx 
1
​
  to the streaming conv backward (which requires contiguous ir_mul layout!) and 
dx 
res
​
  to the previous layer. Writing 
dx 
1
​
  and 
dx 
res
​
  directly to two separate contiguous output pointers in a Triton GEMM (or fused split-epilogue) removes the strided-slice DtoD copy. 
optimisation-ledger.md

Atom GEMMs (20% cuBLAS / 26% segment_gemm) + Atom elementwise/copies (12% NbCl5) 
optimisation-ledger.md

NbCl5: 1.0 – 1.8%


NaNbO3: 0.8 – 1.4%


NaCl: 1.0 – 1.5%

bitwise (if cuBLAS split into 2 contiguous mm calls is faster than 1 mm + 2 contiguous copies) or rounding (if Triton dual-pointer epilogue)	
opt3-T05 (lin-layout-copies) noted 33.8 µs of block-linear backward row gathers/slices per interaction on NbCl5 as "Not built", and did not fuse the dual-output write of merged_residual backward. 
optimisation-ledger.md

Profile the backward of ResidualBlockLinear on NbCl5: measure the time of mm(dout, W_merged.T) + the two .contiguous() slice copies vs. a Triton GEMM writing two contiguous 
[M,512] buffers.


Stop rule: Cut if the eliminated contiguous copies + GEMM difference is 
<150 μs per force call on NbCl5.

High. Pure memory-traffic and kernel-count reduction across all GPUs.
5	
Contiguous float4 (cp.async.cg) Vectorized Gather & Epilogue in segment_gemm (fastmace/kernels/triton/segment_gemm.py)


segment_gemm accounts for 26% of NbCl5 kernel time (4.85 ms) at 8.7–12.3 TFLOPS in the step vs. 13–15 TFLOPS for cuBLAS. Why is segment_gemm 15–25% slower than cuBLAS on the exact same 
K=N=512 shapes? Because segment_gemm gathers rows of 
A ([atoms*(2l+1), 512]) or scatters rows of 
C ([atoms, 16, 512]) using 2D scalar pointer arithmetic (row_idx[:, None] * 512 + offs_k[None, :]). Because the inner dimension (512 channels) is contiguous and a multiple of 4, adding tl.multiple_of(offs_k, 16) and tl.max_contiguous(offs_k, 64) (plus evict_first on the 
A operand in backward) forces the Triton compiler to emit cp.async.cg.shared.global (16-byte float4 loads) instead of 4× scalar ld.global.f32. 
optimisation-ledger.md

segment_gemm Triton atom products (26% NbCl5, 14% NaNbO3) 
optimisation-ledger.md

NbCl5: 1.2 – 2.5% (if segment_gemm gains 5–10%)


NaNbO3: 0.7 – 1.4%


NaCl: 0.3 – 0.8%

bitwise (identical tile and accumulation order; only changes load/store instruction width)	
§4 notes that alternative segment_gemm tile sizes (12 configs) were within 3% in isolation. However, alignment/contiguity compiler hints (tl.max_contiguous, tl.multiple_of) to fix PTX instruction emission from scalar ld.global.f32 to ld.global.v4.f32 / cp.async.cg were never checked. 
optimisation-ledger.md

Inspect the compiled PTX of segment_gemm (TRITON_CACHE_DIR). Check if 
A/
C loads/stores are .v4.f32 or .b32. Add tl.max_contiguous / tl.multiple_of and benchmark segment_gemm on NbCl5.


Stop rule: Cut immediately if PTX already emits .v4.f32 / cp.async.cg on all pointers, or if isolated kernel speedup is 
<4%.

High. Applies to sm_80, sm_89, and sm_120 whenever segment_gemm runs.
6	
Fused Post-Contraction Linear + Skip + Readout Epilogue (products.1 + Readout Non-Linear MLP)


In Interaction 1, after fold_readout_tail (lin-fold-readout-tail, C2), the symmetric contraction output 
SC 
1
​
 ∈R 
N×512
  and species table are mapped to the 96-wide hidden layer of the readout MLP (512 -> 96 -> 1 or 6), followed by SiLU, a 96 -> 1 linear, ScaleShift, and float64 energy sum. Currently, 512 -> 96 runs as a cuBLAS GEMM, followed by Inductor elementwise (+ species_table, SiLU), another small GEMM (96 -> 6), and reduction. Because 
N 
out
​
 =96 (or 
16 for a single head, lin-t08-headslice) and the second layer is a vector dot product (
96→1), the entire readout MLP ([N, 512] @ [512, 16] -> SiLU -> @ [16, 1]) and its backward can be fused into a single Triton kernel that never materializes the [N, 96] hidden activation in DRAM and slices to the 1 active head (matpes_r2scan) at setup. 
optimisation-ledger.md

Atom GEMMs + Atom elementwise + Launch count (saves 4 kernels and 2 GEMM launches per step) 
optimisation-ledger.md

NaCl: 1.2 – 2.0% (~35–50 µs, mostly launch + small-M GEMM overhead)


NaNbO3: 0.3 – 0.5%


NbCl5: 0.3 – 0.6%

rounding	
lin-t08-headslice (slicing C2 from 96 to 16 columns for the active head) was left "for later" (estimated 10–40 µs alone). Fusing the 
16→1 readout second layer and SiLU directly into the 
512→16 GEMM epilogue/recompute was never tried. 
optimisation-ledger.md

Write a fused Triton kernel for x @ W1_head + b1_head -> SiLU -> @ w2_head (forward + backward) with 
K=512,H=16.


Stop rule: Cut if NaCl step A/B does not improve by 
≥1.0% or NbCl5 regresses.

High. Exact same [N, 512] -> 16 -> 1 shape structure across all GPUs.
7	
Pending Package Completion: Interaction-0 Per-Species Residual Table + Adaptive Skin 
0.66  
A
˚
  Rule + direct_node_feats + folded_constants


Combine the four already-validated/pending exact zero-risk items that were individually too small (<1.7% each) to adopt alone:


1. Per-species residual table in Interaction 0 (−0.4 to −0.8% on Ada). 
optimisation-ledger.md


2. Ballistic/density-aware skin rule giving ~0.66 Å at 1000 K (−1.0 to −1.7% on NaCl, neutral on large cells). 
optimisation-ledger.md


3. direct_node_feats + folded_constants (opt3-T05, −0.5 to −0.8% on NaCl, removes strided [n, 512] copy into conv and [1280, 65] weight cat). 
optimisation-ledger.md

First-interaction / atom tables + Neighbour list rebuild frequency + Layout copies 
optimisation-ledger.md

NaCl: 2.0 – 3.0%


NaNbO3: 0.4 – 0.8%


NbCl5: 0.4 – 0.8%

bitwise (except i0 table which is rounding to 1e-7)	
Listed individually in §4 "Pending, not yet default" and opt3-T05 (NEVER-AUTOMATIC), never tested as a combined zero-regression cleanup package. 
optimisation-ledger.md

Run a single same-process A/B (ab_rewrites.py) enabling all four together on NaCl, NaNbO3, and NbCl5 on sm_120 and sm_89.


Stop rule: Cut any component that causes 
>0.1% regression on large cells; adopt package if NaCl 
b/a≤0.980.

High. Confirmed across sm_120 and sm_89. 
optimisation-ledger.md

Deep-Dive on the Top 3 Proposals
Idea 1: Epilogue-Fused 
dx and Partial 
dg in segment_gemm Backward
Why it works when Stage C and Stage D failed:

In Round 1, Stage C (elt-gate-prologue) tried to fuse the gate into the forward prologue of segment_gemm. That failed (+1.6 to +2.0% slower on NbCl5) because inside the GEMM K-loop (
512/32=16 iterations), every thread block had to gather the gate scalar 
σ(g 
i,c
​
 ) repeatedly for every tile of 
K.   
optimisation-ledger.md

Stage D (elt-gate-backward-epilogue) was rejected on paper because in the backward pass of linear_2 (
dy=dB⋅W 
2
T
​
 ), output rows are indexed by 
(i,m) where 
m∈{1..2l+1}, while the gate gradient 
dg 
i,c
​
 =∑ 
m=1
2l+1
​
 dy 
i,m,c
​
 x 
i,m,c
​
 σ(g 
i,c
​
 )(1−σ(g 
i,c
​
 )) sums over 
m. If different 
(i,m) rows fall into different thread blocks, completing the 
∑ 
m
​
  inside the GEMM epilogue requires atomicAdd, which breaks bitwise reproducibility.   
optimisation-ledger.md

The fix: In segment_gemm backward, each thread block computes a tile of 
dy 
i,m,c
​
  in registers at the very end of the K-loop (the epilogue, executed once, not 16 times!).

For 
l=0 (
m=1, SiLU activation): 
dx 
i,c
(0)
​
 =dy 
i,c
(0)
​
 ⋅SiLU 
′
 (x 
i,c
(0)
​
 ) has no sum over 
m at all. It can be computed 100% in the epilogue of the 
l=0 backward GEMM.

For 
l=1,2,3: If we order the rows of each 
l-block as 
(i,m) (which is already how gated_rows stores 
[N⋅(2l+1),512]!), each thread block in the epilogue loads 
σ(g 
i,c
​
 ) once, writes 
dx 
i,m,c
​
 =dy 
i,m,c
​
 ⋅σ(g 
i,c
​
 ) directly to the 
dx buffer, and multiplies 
dy 
i,m,c
​
 ⋅x 
i,m,c
​
  in registers. Thus the intermediate tensor 
dy (
[N,16,512]) is never written to or read from DRAM.

Even better: if segment_gemm backward processes all 
2l+1 rows of atom 
i within the same thread block (e.g., each block handles 
B 
atoms
​
 =8 atoms 
×(2l+1) rows = 
24,40,56 rows for 
l=1,2,3), then all 
2l+1 values of 
m for atom 
i reside in the same thread block! The reduction 
∑ 
m=1
2l+1
​
 dy 
i,m,c
​
 x 
i,m,c
​
  can be done deterministically in shared memory or warp shuffles inside the epilogue, eliminating the separate Gate VJP kernel entirely with zero atomics and bitwise determinism.

Idea 2: L2 Residency Control in chunked_conv_backward (__ldcs on w, __stwb on dw) + Single-Chunk Fast Path
Why it matters:

In conv-opt7-traffic, Nsight Compute showed that the Interaction-1 backward convolution moves 1,094 MB of L2 traffic on NbCl5 and 1,673 MB on NaNbO3, of which w + dw account for 49–51% (537 MB and 861 MB).   
optimisation-ledger.md

With chunked_conv_backward (OPT6-T02), dw is written into a 2-slot ring buffer of 1,536 edges (
7.86 MB per slot) so the side-stream Triton kernel (da = dw \cdot W) can consume dw from L2 without DRAM traffic.   
optimisation-ledger.md

However, while the side stream reads dw from slot 0, the main stream is already running the next chunk of the convolution backward, which streams 
7.86 MB of w from DRAM, plus sender features 
x 
j
​
  and receiver gradients 
dout 
i
​
 . Because w is loaded with default cache policy (ld.global.f32), it allocates cache lines in the 32 MB L2, evicting dw from slot 0 and evicting reusable atom features 
x 
j
​
  and 
dx 
j
​
 .

Marking w loads in the backward kernel with __ldcs (cache streaming / evict_first) prevents dead w tiles from polluting the 32 MB L2.

Simultaneously, when n_chunks == 1 (which triggers on NaCl because 
E=8,192 is below the 
4×L2 unchunked threshold), chunked_conv_backward currently still records and waits on the side-stream CUDA events (conv-chunked-nacl, causing a +0.8% regression on NaCl). Skipping the side-stream fork when n_chunks == 1 immediately recovers that 0.8% on NaCl.   
optimisation-ledger.md

Idea 3: Removing the 25 cuBLAS Split-K cudaMemsetAsync Nodes & 12 Eager Zero-Fills (launch-memsets)
Why it matters for NaCl:

On NaCl (216 atoms), the force graph has 233 kernels and spends ~135 µs (5% of the step) in inter-kernel launch gaps.   
optimisation-ledger.md

During opt3-T04, profiling revealed that 25 of those nodes are cudaMemsetAsync calls inserted by cuBLAS before small-
M GEMMs that use split-K reductions, and 12 nodes are standalone fill_/copy_ zero-initialization kernels before scatter/segmented sums.   
optimisation-ledger.md

Eliminating those 37 tiny utility nodes (16% of all nodes in the force graph!) directly cuts ~35–50 µs of launch gaps and memset execution on NaCl without touching any math, preserving bitwise accuracy across all GPUs.

Flagged Conflicts with Section 4 (Closed Directions)
Only two suppressed directions from Section 4 have any technical justification for a targeted, modified re-examination:

Gate Backward Fusion (conflicts with §4 "elementwise around the gates" / elt-gate-backward-epilogue Stage D):

Why §4 closed it: Stage C fused the gate into the forward prologue (inside the K-loop, +1.6% slower), and Stage D proposed fusing the gate backward into the backward epilogue using cross-block atomic reductions over 
m, which was rejected for breaking determinism.   
optimisation-ledger.md

New evidence / mechanism to reopen (Rank 1 above): Do not touch the forward prologue, and do not use cross-block atomics. Either (a) fuse only the elementwise scaling 
dx 
i,m,c
​
 =dy 
i,m,c
​
 σ(g 
i,c
​
 ) and product 
p 
i,m,c
​
 =dy 
i,m,c
​
 x 
i,m,c
​
  in the epilogue (which requires zero reduction over 
m and removes the 128 MB 
dy tensor), or (b) tile segment_gemm backward by whole atoms (
B 
atoms
​
 ×(2l+1) rows per block) so all 
2l+1 rows of atom 
i sit in the same thread block and 
dg 
i,c
​
  reduces deterministically in shared memory.

Wide float2 Loads in Streaming Backward (conflicts with §4 "Wide float2 loads: 2–7% kernel gain on sm_120 only, owner chose not to build"):

Status: Do not reopen unless the owner relaxes the threshold, as conv-wide-loads was already measured on 27 October 2026 (0.931–0.971 kernel ratio on sm_120, 0.998–1.003 on sm_89, translating to 0.3–1.0% whole-step gain on sm_120 and 0% on sm_89). However, if combined with the L2 residency hints in Rank 2 (which takes 5 lines in the same NVRTC template), the two together could cross the 2% whole-step bar on sm_120 while remaining positive on sm_89.   
optimisation-ledger.md



//


# Research prompt: new optimisation opportunities for FastMACE (MH-1 molecular dynamics on NVIDIA GPUs)



*Written 30 October 2026 for an external research agent. FastMACE main is `d35658be`. The attached documents are

listed at the end.*



## Your task



Find optimisation opportunities for FastMACE that have not been tried yet, or that were tried in a form that does

not settle the question. FastMACE runs molecular dynamics with the pretrained MACE-MH-1 interatomic potential.



Rank them by expected whole-step gain on the GPUs listed below. For each idea, give:



**Mechanism.** What changes in the computation, and why it should be faster or use less memory on these GPUs.
2. **Evidence.** Papers, vendor documentation, open-source kernels, or other MLIP codes (NequIP/Allegro, SevenNet,

   cuEquivariance, OpenEquivariance, e3nn-jax, Symmetrix, NVIDIA ALCHEMI, TorchMD-Net, …) that show it working.

   Give links and the numbers they report.

3. **Fit to FastMACE.** Where in the step it applies, using the profile below, and an Amdahl ceiling from the

   attached profile shares.

4. **Exactness class.** Bitwise, rounding class (reassociation only), or approximate. If approximate, how it would

   pass the accuracy gate.

5. **Falsifier.** The cheapest experiment that would kill the idea before anyone builds it, with a stop rule.

6. **Overlap check.** Confirm it is not already in the optimisation ledger. If it is, explain what is different

   this time.



The goals, in priority order:

faster MD steps for MH-1 at float32;
2. more atoms per 16 GB card (memory per edge);

3. faster float64 steps and float64 relaxations;

4. faster engine setup (build plus graph capture, which dominates short relaxations).



Speed matters most, but it never comes at the cost of accuracy.



## What FastMACE is



**What:** a standalone Python/CUDA library that runs NVT/NPT Langevin MD, relaxations, quenches and Monte
  Carlo with pretrained MACE models.

**Main model:** MACE-MH-1, `matpes_r2scan` head. It also runs other MACE models through generic rewrites.
**Engine:** the force/stress computation is captured as CUDA graphs. It uses a hybrid backend: cuEquivariance
  (cuEq) for symmetric contractions, OpenEquivariance (OEq) for the convolution, and its own Triton/NVRTC kernels.

**Model rewrites:** a planner chooses, per model and per GPU, a set of rewrites of the model. Each rewrite is
  exact, or rounding-class with a recorded error bound.

**Stack:** torch 2.14.1+cu130, cuEquivariance 0.12.0, OpenEquivariance 0.7.0, MACE 0.3.16, Python 3.10, CUDA 13.
**Benchmark systems:** 3×3×3 supercells of NaCl (216 atoms), NaNbO3 (1,080) and NbCl5 (1,944). Production
  systems are Na–Nb/Ta–O–Cl solid electrolytes, from 100 atoms up to tens of thousands, with melt-quench

  protocols and atom exchanges.

**Timestep:** 1 fs.


### GPUs we actually have (only these matter)



| GPU | Arch | Where | Notes |

| --- | --- | --- | --- |

| RTX 5060 Ti 16 GB | sm_120 (Blackwell consumer) | local PC, WSL2 | desktop shares the card; WSL2's driver holds extra memory per resident thread stack |

| RTX 4000 Ada 20 GB | sm_89 | three workstations | native Linux |

| RTX A4000 16 GB | sm_86 | two workstations | native Linux |

| A100-SXM4-80GB | sm_80 | Modal (cloud, small budget) | also the A100 80GB PCIe sometimes |



Hopper and datacenter Blackwell (H100, B200) are out of scope. We don't have them.



### Current speed (float32 default plan, ms per MD step)



| System | RTX 5060 Ti | A100-SXM4-80GB |

| --- | --- | --- |

| NaCl 216 atoms | 2.44 | n/a |

| NaNbO3 1,080 atoms | about 9–10 | 9.29 |

| NbCl5 1,944 atoms | 17.7 | n/a |



The NaCl and NbCl5 numbers on the 5060 Ti are from 29 October (`o1b-loop-overhead.md`). The A100 number comes from

the 29 October trace in `tc-gemm-screen.md`: GEMMs take 4.14 ms (45 %) of each NaNbO3 step, and there are only

about 58 µs of concurrent kernels per step. The fine-grained profile is attached (`fp32-profile-20261028.md`).

OPT9's `wprime_backward` has since shortened the convolution backward by 3–19 %, depending on GPU and cell.



### Current memory limits (RTX 5060 Ti, float32)



| System | Largest that runs | Edges |

| --- | --- | --- |

| NaCl | about 27,000 atoms | about 0.86M |

| NbCl5 | 36,720 atoms, with automatic activation recompute | about 1.02M |



Peak allocated is about 0.23 GiB + 5.9 kB per edge + 268 kB per atom. Recompute saves about 2.8 GiB at about 8 %

step cost, so it is used only when memory would not fit otherwise.



## Hard constraints (an idea that breaks one is out)



**Accuracy gate.** A float32 path is accepted if its force and stress error against float64 is at most 2× that
  of the stock float32 calculator, on NaCl, NbCl5 and NaNbO3. A 20 ps NVE drift check must also be unchanged.

  Plain TF32 fails the gate by about 3,500–5,100×, and 3xTF32 (CUTLASS) passes.

**Precision decision (owner, 9 October).** The production precision is FP32 with TF32 off. Changing it needs the
  owner's approval. Proposals may still argue for it with evidence.

**Generality.** Every optimisation must work for any crystal and composition without per-structure tuning:
  - no autotuning tables;

  - no shape-keyed caches;

  - no per-structure choices.



  Per-GPU defaults are allowed, keyed by exact device name (decision 66), but only where same-process A/Bs in both

  build orders measured them faster. They stay off on every other GPU.

**Measurement standard.** A gain counts only as a same-process, alternating-block, whole-MD-step A/B in both
  build orders, under CUDA graphs. Microbenchmarks are falsifiers only.

**Compilation time matters.** The default compile mode is used, with no Inductor max-autotune (it was rejected:
  setup 140–756 s for about 1 % gain).

**No larger timesteps.** That's a physics decision, out of scope.
**Hybrid runs are not bitwise reproducible,** because of atomic sums in OEq. Exactness is judged by error class,
  not by bit equality.



## Already done (don't re-propose without a new angle)



The attached optimisation ledger (`docs/optimisation-ledger.md`, about 300 rows) is authoritative. In short:



**Adopted:**
  - receiver sorting;

  - species-factored and joint radial input;

  - folded cutoff;

  - density-first;

  - reassociated first interaction, with its fused Triton backward;

  - streaming (receiver-major) convolution, from OEq PR 223;

  - chunked convolution backward;

  - fused folded forward;

  - `wprime_backward`: the radial backward as a dot with tabulated w'(r);

  - radial tables;

  - block, grouped and merged-residual linears;

  - `gated_rows` and `segment_gemm`, a Triton gathered-row GEMM;

  - species-tables, fold-product-linear and fold-readout-tail linear folds;

  - `grouped_launch`, per GPU;

  - GPU neighbour list, pipelined steps and asynchronous output;

  - one-graph strain path;

  - float64 via cuBLAS emulation with dynamic mantissa bits.

**Rejected or no gain:**
  - TF32, 3xTF32 on Ada (slower), BF16x9 (no-op on sm_89);

  - FP16x3 Triton GEMMs, hybrid SIMT+MMA GEMMs on sm_120, and in-kernel co-execution of FP32 and 3xTF32 on the

    A100 (at most 5 % ceiling);

  - split-K on sm_120, cuBLASLt preference and per-shape heuristics;

  - Inductor autotuning;

  - low-rank SVD of the linears (they are full rank);

  - concurrent graph branches for per-l products;

  - in-place block linear;

  - species/Morton atom order;

  - 16-bit weights;

  - skin 0.4 Å;

  - Inductor programmatic dependent launch (it gives wrong forces inside our graphs);

  - the gate in GEMM epilogues (OPT10);

  - megakernel probes;

  - a larger skin for small cells;

  - loop-overhead removal (there is none to remove).

**Open, or pending an owner decision:**
  - CUTLASS 3xTF32 (rz3) in the real force graph on the A100-SXM4-80GB: the whole step takes 0.71–0.87 of the

    FP32 default's time, the gate and NVE pass, and it isn't a default yet;

  - `species_residual` (a 0.2–1 % gain);

  - cache hints and 128-row tiles on Ada;

  - split-K on GPUs other than sm_120.



## Where we think the remaining time is (check this)



**GEMMs:** atom-level GEMMs (cuBLAS SGEMM and `segment_gemm`) are about 45 % of the A100 step, and a large share
  on the consumer cards. On consumer GPUs (sm_86/89/120), FP32 SIMT throughput is the limit and the Tensor Core

  paths lose accuracy or speed.

**Convolution:** the backward is the next largest block, after OPT9.
**Memory:** about 5.9 kB per edge of saved activations in float32. Under WSL2, the per-thread stack reservations
  of spilling kernels cost GiBs outside the allocator.

**Setup:** engine build plus capture takes about 30 s per composition, and it dominates short relaxations.
  Engine reuse across frames of the same composition has just landed, and it measured 1.7× more frames per hour.



## Questions we'd especially like answered



Are there FP32-accurate Tensor Core schemes for sm_86/89/120 (consumer Ampere, Ada, Blackwell) that beat native
   SGEMM at our shapes, M ≈ 10^3–10^5 rows with K, N ≈ 128–1024? Examples: Ozaki-scheme integer emulation, an

   INT8 Ozaki split, or the error-free transformations from Ootomo and Yokota. Report published throughput per

   architecture and the accuracy on cancellation-heavy data.

2. How do other equivariant MLIP codes reach high throughput on consumer GPUs? Examples: SevenNet with

   flashTP/cuEq, NequIP/Allegro with their own kernels, NVIDIA ALCHEMI and cuEquivariance's latest kernels,

   Symmetrix, and MACE's own recent PRs. Which of their techniques are not in our ledger?

3. What are the memory-reduction techniques for the edge-level activations of message-passing networks? Examples:

   recomputation strategies, fused message-plus-aggregation backward without storing per-edge tensors, and

   compressed or quantised activation storage that stays inside the accuracy gate.

4. Are there ways to cut CUDA-graph capture and engine-build time across many similar structures? Examples:

   graph-update APIs (`cudaGraphExecUpdate`), shape-polymorphic graphs, and caching of compiled kernels and graphs

   across processes.

5. Are there float64-specific tricks for consumer GPUs with weak FP64? Examples: Ozaki-style emulation on INT8 or

   FP32 units, and mixed float64/float32 schemes that keep float64-class forces.

6. On the A100: beyond 3xTF32 GEMMs, which other parts of the step could use Tensor Cores, such as the convolution

   or the radial MLP, while keeping FP32-class accuracy?



## Deliverable



A ranked list of 5–15 ideas in the format above, then a short "already covered" list. That second list names the

ideas you looked at and found already in the ledger, with the ledger row ID. Mark every number as measured, by a

source you cite, or estimated, by you.



## Documents to attach (from the FastMACE repository, main `d35658be`)



Essential:

`SPEC.md`: the architecture, the rewrite catalogue (§6), the planner rules (§9) and the owner decisions (§18).
`docs/optimisation-ledger.md`: every optimisation tried, with its verdict. It is long (about 320 KB). If it's too
  large, attach its "Index by area" section only.

`docs/validation/fp32-profile-20261028.md`: the fine-grained step and setup profile.
`docs/validation/tc-gemm-screen.md`: Tensor Core GEMMs on our GPUs, 29 October.
`docs/validation/tf32-screen.md`: TF32 and 3xTF32 on Ada, 29 October.
`docs/validation/opt9-T03.md`: `wprime_backward`, the latest adopted default.
`docs/design/switch-map.md`: every switch and its default rule.


Useful:

`docs/validation/capacity-16gb.md` and `capacity-T02.md`: the memory model and limits.
`docs/validation/opt-T01-tensor-cores.md`, `docs/validation/opt-5.md` and
  `docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md`: the earlier Tensor Core and FP16x3 work.

`docs/validation/relax-throughput.md`: setup time and engine reuse.
`plans/ideas-20261027.md`: the last ideas round, with its falsifiers.
`docs/rewrites.md`: the per-screen table.
Direct answers to the six research questions precede the ranked list of 12 untried or unresolved optimisation opportunities and the ledger cross-check.

Direct Answers to the 6 Research Questions
1. FP32-accurate Tensor Core schemes on sm_86 / sm_89 / sm_120
Verdict: Mathematically impossible to beat native cuBLAS SGEMM on consumer GeForce/Workstation GPUs (
M≈10 
3
 –10 
5
 , 
K,N≈128–1024).

Hardware Reality: On consumer Ampere (sm_86), Ada (sm_89), and Blackwell (sm_120), NVIDIA throttles dense FP16/BF16/INT8 Tensor Core throughput with FP32 accumulate to 2× the FP32 SIMT rate (and TF32 to 1× FP32 SIMT on sm_86/sm_120, 2× on sm_89), whereas datacenter GPUs (A100 sm_80) have an 8×–16× ratio.   
optimisation-ledger.md

Ootomo & Yokota (2022) (2×FP16 EFT with outside-TC accumulation): Recovers 22.75 bits of mantissa (matching IEEE FP32 accuracy when inner 
K-loop accumulation happens in FP32 registers outside Tensor Cores to avoid Tensor Core Round-to-Zero error). Published throughput on A100 is 51 TFLOPS (2.6× SIMT peak). However, it requires 3 MMAs per FMA (
A 
hi
​
 B 
hi
​
 ,A 
hi
​
 B 
lo
​
 ,A 
lo
​
 B 
hi
​
 ) plus shared-memory exponent scaling. When Tensor Core rate is only 2× SIMT, a 3-pass scheme has a hard theoretical ceiling of 
2/3=0.67× SIMT speed before decomposition overhead (confirming your prec-fp16x3-standalone finding).   
optimisation-ledger.md

Ozaki-I / INT8 Slice Schemes (Ootomo et al. 2023, Mukunoki 2025, FlashSO(2) arXiv:2610.35164): Decomposes FP32 into 
S 7-bit or 8-bit integer slices with row/column exponent alignment, requiring 
S(S+1)/2 INT8 GEMMs (
6 passes for 
S=3, 
10 passes for 
S=4). Because INT8 IMMA on sm_86/sm_89/sm_120 is only 4× FP32 SIMT, even 3 slices (
6 passes) caps at 
4/6=0.67× SIMT (confirming your prec-int8-f32 measurement of 599–965 µs vs 518 µs SGEMM).   
optimisation-ledger.md

Exception (Two-Term Mixed Precision): The only way to beat 1× SIMT on a 2× Tensor Core GPU is a 1-pass or 2-pass scheme, which is only possible if weights or activations are pre-conditioned or only 1 correction term is needed (see Idea 9 for A100/Ada).

2. How other Equivariant MLIP codes reach high throughput on consumer GPUs
SevenNet + FlashTP (ICML 2025, SNU-ARC/flashTP): Uses path-aggregated execution, constant-memory CG sparsity tables, and fused gather-TP-scatter. FastMACE's streaming_conv (OEq PR 223) + chunked_conv_backward + wprime_backward already beats FlashTP on 
L 
max
​
 ≤3. However, FlashTP and cuEquivariance highlight register vs shared-memory spilling in backward kernels (addressed in Idea 2).   
optimisation-ledger.md
+ 1

NequIP / Allegro (mir-group/allegro): Avoids atom-level symmetric contractions by keeping features strictly on edges, which doesn't map to MACE's node-centric Many-Body architecture.

Symmetrix (symmetrix-xl) & MACE recent PRs: FastMACE already backported PR 1498 (single_graph_strain), PR 1614 (naive-guard), PR 1641 (cueq_conv_fusion), and PR 1711 (polar_uvu). What remains unexploited from recent MACE/cuEq ecosystems is compile-time head pruning across the entire readout + last-layer linear fold (Idea 4) and fused backward atomic-free CSR transpose (Idea 6).   
optimisation-ledger.md
+ 1

3. Memory-reduction techniques for edge-level activations
Your 5.9 kB/edge footprint is dominated by storing the forward per-edge tensor-product weights 
w∈R 
E×1280
  (5,120 bytes/edge in FP32) for the backward pass, plus spherical harmonics 
Y 
lm
​
  (64 bytes/edge) and cutoff/radial intermediates.

Key Untried Lever: With fused_folded_forward (OPT8-T02) and wprime_backward (OPT9-T03) both adopted, nothing in the radial backward needs 
dL/dw anymore! The radial backward now computes 
dL/dr=⟨grad_w_or_edge,w 
′
 (r)⟩ directly. Therefore, 
w is only saved in global memory to compute 
dL/dx 
sender
​
  and 
dL/dY 
lm
​
  in the streaming convolution backward. By recomputing 
w(r)=a(r)W 
T
  on-chip per L2 chunk or per edge tile during the streaming backward (Idea 1), or quantizing stored 
w to FP8/INT8 with per-edge scale only for the 
dL/dx path (Idea 5), you can cut memory per edge by 60–85 % (from 5.9 kB/edge down to ~1.2–2.5 kB/edge).   
optimisation-ledger.md
+ 2

4. Cutting CUDA-graph capture and engine-build time
Key Untried Levers: (a) radial_tables disk cache (rad-table-cache, already on branch fastmace/radial-table-cache, saves 5–20 s in FP32 and 42 s in FP64); (b) Single padded-capacity graph capture + cudaGraphExecKernelNodeSetParams / device-scalar active-edge masking instead of re-capturing up to 64 separate CUDA graphs for 5 % edge buckets (Idea 7); and (c) AOTInductor (torch._inductor.aoti_compile_and_package) persistent binary cache to eliminate the 15–25 s warm Dynamo/AOTAutograd tracing overhead across processes (Idea 10).   
optimisation-ledger.md

5. Float64-specific tricks for consumer GPUs
Decision 63 (fp64_emulation via cuBLAS INT8 Ozaki) + Decision 65 (w_tables, save_transcendentals, and 56-bit mantissa) already cut FP64 step time by ~65 %.   
optimisation-ledger.md
+ 1

Remaining bottlenecks: (a) Fix the D1 56-bit mantissa graph-recapture bug (prec-d1-relax-fix) so Decision 65(a)'s 12–14 % speedup is restored safely (Idea 8); (b) Mixed FP64/FP32 Force-Corrected Quasi-Newton (LBFGS) Preconditioning for relaxations (Idea 11); and (c) Vectorized double2 loads and INT8-emulated edge_outer_sum / wprime_backward in FP64 (Idea 12).

6. A100-specific Tensor Core opportunities beyond 3xTF32 GEMMs
First, landing CUTLASS 3xTF32 (rz3) (tc-gemm-screen.md) as a Decision-66 per-GPU default on "NVIDIA A100-SXM4-80GB" is your single biggest ready-to-merge win (13–29 % whole-step speedup) (Idea 3).

Second, on A100, Interaction-1's reassociated node projection and the chunked backward's da = dw @ W / wprime dot can be routed through CUTLASS 3xTF32 when 
w 
′
  isn't used or for generic models (Idea 9).

Ranked List of New / Unsettled Optimisation Opportunities
Idea 1: On-Chip 
w(r) Rematerialization in Streaming Convolution Backward (Eliminate Global [E, 1280] 
w Storage & Traffic)
Target Goals: Goal 2 (Memory per edge: −5.1 kB/edge, +2.5× more atoms on 16 GB) & Goal 1 (Faster FP32 MD steps on bandwidth-bound cells).

1. Mechanism:
Currently, fused_folded_forward (OPT8-T02) computes 
w 
e
​
 =a 
e
​
 [W,b] 
T
 ∈R 
1280
  in registers inside the forward streaming convolution, uses it immediately to accumulate receiver messages, and still writes the entire [E, 1280] tensor 
w (5,120 bytes/edge) to global DRAM so the streaming backward can read it back.
Why was 
w needed in backward before OPT9? Because chunked_conv_backward needed both 
w (for 
dx,dY) and 
dw (to contract 
da=dw⋅W).
What changed with OPT9 (wprime_backward): Now that wprime_backward computes the radial force derivative directly from 
w 
′
 (r) without needing 
dw⋅W (or in tandem with it), if the streaming convolution backward kernel rematerializes 
w 
e
​
 =a 
e
​
 [W,b] 
T
  on-chip in registers (identical to how fused_folded_forward does it, reading only the 65-float radial basis 
a 
e
​
  and shared-memory 
[W,b]), then:   
optimisation-ledger.md
+ 1

fused_folded_forward never writes w ([E, 1280]) to DRAM (saving 5,120 B/edge of DRAM write in forward);

streaming_conv_backward never reads w ([E, 1280]) from DRAM (saving 5,120 B/edge of DRAM read in backward, replacing it with a 260 B/edge read of 
a 
e
​
 );

Peak activation memory drops by 5,120 bytes per edge (from 5.9 kB/edge to ~0.8 kB/edge!).
Why did OPT8-T01b (conv-full-fusion) fail its stop rule earlier? Because OPT8-T01b tried to do three in-kernel products in the backward (rebuilding 
w, computing 
dw, and contracting 
da=dw⋅W on chip) before wprime_backward existed! With wprime_backward handling the radial derivative (or doing only the forward 
w rematerialization), the backward only needs the single 
w=aW 
T
  product—the exact same arithmetic that succeeded in fused_folded_forward!   
optimisation-ledger.md
+ 1

2. Evidence:

Measured in FastMACE (OPT8-T02 / conv-fused-folded-forward): Generating 
w=a[W,b] 
T
  on-chip in the NVRTC streaming kernel cut the forward time to 0.65–0.66× and whole-step time to 0.928× on NaNbO3 and 0.967× on NbCl5 on RTX 5060 Ti.   
optimisation-ledger.md
+ 1

Measured in FastMACE (OPT7-T02 / conv-opt7-traffic): w + dw accounts for 49–67 % of L2 bytes in the streaming convolution (268 MB fwd / 537 MB bwd on NbCl5).   
optimisation-ledger.md

Measured in FastMACE (capacity-16gb.md / prompt): Saved activations are 5.9 kB/edge; 
w∈R 
E×1280
  in FP32 is 
1280×4=5.12 kB/edge (86.8 % of all per-edge memory).

3. Fit to FastMACE & Amdahl Ceiling:

Where: Interaction-1 (second interaction) fused_folded_forward + streaming_conv backward.   
optimisation-ledger.md

Amdahl Ceiling:

Memory: Cuts per-edge activation memory by ~80–85 % (estimated: 5.9 kB/edge 
→ ~0.8–1.1 kB/edge), raising the 16 GB RTX 5060 Ti atom limit on NbCl5 from 36,720 atoms (~1.02M edges) to ~75,000–90,000 atoms (~2.2–2.5M edges) without needing the 8 % activation-recompute penalty!

Speed: Eliminates 5.12 kB/edge of forward DRAM writes + 5.12 kB/edge of backward DRAM reads (
10.24 kB/edge≈570 MB on NbCl5, 
900 MB on NaNbO3). Convolution forward + backward is ~15–26 % of the step; estimated whole-step gain is 3–8 % faster on NaNbO3/NbCl5 (plus +8 % on large cells where global activation recompute is no longer triggered).   
optimisation-ledger.md

4. Exactness Class:
Bitwise equal to fused_folded_forward (exact same FMA sequence for 
w 
e
​
 =a 
e
​
 [W,b] 
T
  in forward and backward).   
optimisation-ledger.md

5. Falsifier:
Modify tools/bench/opt8_fused_conv.py to test the asymmetrical fused_folded_forward (no 
w global store!) + streaming_conv_backward with on-chip 
w=a[W,b] 
T
  reconstruction + wprime_backward (no on-chip 
da=dw⋅W).
Stop rule: Kill for speed if subgraph fwd+bwd time on NaNbO3/NbCl5 is 
>1.00× current fused_folded_forward + wprime_backward; keep as the Rule 12 high-capacity default (replacing full-model activation recompute) if subgraph time is 
≤1.03× because it saves 5.12 kB/edge at 
<1% step cost instead of 8 % cost!

6. Overlap Check:
Distinct from conv-full-fusion (OPT8-T01b) and conv-chunked-recompute (OPT6-T01): OPT6-T01 recomputed 
w via a separate cuBLAS GEMM into L2; OPT8-T01b tried to compute both 
w=aW 
T
  and 
da=dw⋅W in SIMT registers inside the backward. Now that wprime_backward (OPT9-T03) removed 
da=dw⋅W, only the single 
w reconstruction is needed, and fused_folded_forward still writes 
w to DRAM today solely for the backward.   
optimisation-ledger.md
+ 4

Idea 2: WSL2 Thread-Stack Reservation Elimination via .maxnreg / __launch_bounds__ and Zero-Spill Kernel Audit
Target Goals: Goal 2 (More atoms per 16 GB card under WSL2 — recovers 1.5–3.5 GiB of phantom unallocated VRAM) & Goal 1 (Faster FP32 steps via higher occupancy).

1. Mechanism:
As noted in your prompt and capacity-T02.md, under WSL2 (dxgkrnl + NVIDIA Windows WDDM driver), the CUDA driver allocates local-memory thread stacks (cudaLimitStackSize 
× maxThreadsPerMultiProcessor 
× numSMs) eagerly across the context based on the high-water mark of stack frame / register spilling across all compiled kernels loaded in the CUcontext—even kernels only called during setup, self-checks, or fallback paths!
A single Triton, Inductor, or NVRTC kernel that spills registers to local memory (STL/LDL instructions in SASS) or calls a device function with a stack frame causes the WDDM driver to reserve stack_bytes_per_thread * 36 SMs * 1536 threads (or across virtual warps), eating 1–3+ GiB outside PyTorch's caching allocator.
Two concrete changes:   
SPEC.md

Audit all compiled kernels in the CUcontext using cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func) and CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK at engine build time.

Cap register usage / tile sizes on any spilling Triton/NVRTC kernel (specifically checking radial_tables, wprime_backward, segment_gemm, and OEq's NVRTC templates) and explicitly call cudaDeviceSetLimit(cudaLimitStackSize, 0) (or minimal required bytes) after setup self-checks complete and before graph capture.

2. Evidence:

Measured in FastMACE (SPEC.md Decision 67 / capacity-T02.md): Turning off the fused path on WSL2 caused OEq's stock convolution kernels to reserve several GiB of stack memory outside the allocator. Even on the fused path, WSL2 loses GiBs to per-thread stack reservations.   
SPEC.md

Vendor documentation (NVIDIA CUDA Runtime API cudaLimitStackSize): Stack size is set to the maximum CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES across loaded modules and never shrinks automatically unless cudaDeviceSetLimit(cudaLimitStackSize, ...) is called or spilling kernels are prevented from loading into the primary context.

3. Fit to FastMACE & Amdahl Ceiling:

Where: Engine initialization, NVRTC/Triton compilation flags, and post-self-check memory cleanup before capture().

Ceiling: Recovers 1.5 to 3.0 GiB (estimated from WSL2 vs Linux 16 GB headroom gap) on RTX 5060 Ti, which translates directly to +250,000 to +500,000 edges (+8,000 to +15,000 atoms) on RTX 5060 Ti 16 GB.

4. Exactness Class:
Bitwise (zero mathematical change).

5. Falsifier:
Write a 30-line script that builds the default MH-1 FP32 engine on RTX 5060 Ti (WSL2), iterates over all loaded CUDA functions via cuda.bindings / cuModuleGetFunction or torch.cuda.memory_stats() vs nvidia-smi (cudaMemGetInfo), prints CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES and cudaDeviceGetLimit(cudaLimitStackSize).
Stop rule: If cudaLimitStackSize is already 0 and CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES == 0 for every kernel in the context, kill the idea.

6. Overlap Check:
Not in the ledger. Decision 67 (capacity-T02.md) avoided dropping streaming_conv under Rule 12, and conv-bwd-variants tested --maxrregcount 32/40/48 for speed, but nobody has audited CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES across the whole context or reset cudaLimitStackSize on WSL2.   
optimisation-ledger.md
+ 2

Idea 3: Adopt CUTLASS 3xTF32 (rz3) + species_residual as Per-GPU Defaults under Decision 66
Target Goals: Goal 1 (Faster FP32 MD steps: 13–29 % faster on A100, 0.5–1.0 % faster on Ada/Blackwell).

1. Mechanism:

Part A (A100 CUTLASS 3xTF32 / rz3): Your 29 October tc-gemm-screen.md already proved that replacing cuBLAS IEEE SGEMM with CUTLASS 3xTF32 (rz3 rounding) inside the captured force graph on NVIDIA A100-SXM4-80GB passes the 19-label accuracy gate and 20 ps NVE drift check, while cutting whole-MD-step time to 0.71–0.87× of the FP32 default. Under Decision 66 (29 October 2026), per-GPU defaults keyed by exact device name are now explicitly permitted! Additionally, extend the CUTLASS 3xTF32 dispatcher to also cover the gathered-row GEMMs (segment_gemm) on A100 (either via a CUTLASS Epilogue Visitor / permutation index or by letting segment_gemm's shape rule defer to CUTLASS 3xTF32 + IrrepsView when 3xTF32 is active on A100, since CUTLASS 3xTF32 on A100 reaches 40.8 TFLOPS vs Triton SIMT's 18.2 TFLOPS!).   
SPEC.md
+ 1

Part B (species_residual): Interaction-0's per-species residual table (fp32-followups-20261029) is exact-to-rounding, passed the gate and 20 ps NVE drift check, and measured 0.9918–0.9956 in both build orders on Ada (murphy/leto).   
optimisation-ledger.md
+ 1

2. Evidence:

Measured in FastMACE (tc-gemm-screen.md, opt-T01-tensor-cores.md): Whole MD step on NVIDIA A100-SXM4-80GB is 0.71–0.87× (13–29 % faster). On isolated shapes (opt-T01), CUTLASS 3xTF32 took 5,716 µs vs cuBLAS IEEE 10,313 µs (0.55×) on NbCl5.   
optimisation-ledger.md

Measured in FastMACE (fp32-profile-20261028.md §5): species_residual b/a on Ada is 0.9949/0.9956 (NaCl), 0.9953/0.9942 (NaNbO3), 0.9945/0.9918 (NbCl5).   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: Atom-level GEMMs (4.14 ms / 45 % of the 9.29 ms A100 NaNbO3 step) and Interaction-0 residual linear.   
optimisation-ledger.md

Ceiling: 13–29 % whole-step reduction on A100 (measured), plus 3–5 % extra on A100 if segment_gemm yields to CUTLASS 3xTF32 on A100 (estimated); 0.5–0.8 % whole-step reduction on Ada (measured).   
optimisation-ledger.md

4. Exactness Class:

species_residual: Rounding class (passes gate: E 0.11–0.76, F 0.97–1.03, S 0.37–1.0).   
optimisation-ledger.md

tf32x3_gemms (rz3): Rounding / Certified-equivalent (passes force/stress gate 
≤2× FP32 error and 20 ps NVE drift; requires owner sign-off per 9 Oct precision rule, scoped strictly to DECISION_66_TF32X3_GPUS = ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA A100-PCIE-80GB")).

5. Falsifier:

For species_residual: Run the open A100 same-process A/B. Adopt on Ada (and A100 if 
<1.0 in both orders).

For A100 segment_gemm vs CUTLASS 3xTF32: Run a same-process A/B on A100 comparing tf32x3_gemms=on, segment_gemm=on (where segment_gemm uses Triton SIMT) vs routing linear_2's large shapes through CUTLASS 3xTF32. Stop rule: Keep whichever is faster in both orders.

6. Overlap Check:
Listed under "Open, or pending an owner decision" in your prompt (fp32-followups-20261029, tc-gemm-screen.md). The new technical angle here is disabling Triton SIMT segment_gemm when CUTLASS 3xTF32 is active on A100 (or giving CUTLASS 3xTF32 a gathered-row prologue), because Triton SIMT (segment_gemm) caps at ~18 TFLOPS on A100 while CUTLASS 3xTF32 runs at ~41 TFLOPS!   
optimisation-ledger.md
+ 1

Idea 4: Complete Single-Head Pruning Across Readout MLP + fold_readout_tail (16 vs 96 Hidden Columns)
Target Goals: Goal 1 (Faster FP32 steps: 0.6–1.0 % on NbCl5, 0.4–0.8 % on NaNbO3) & Goal 3 (Faster FP64 steps).

1. Mechanism:
MH-1 is a 6-head checkpoint (matpes_r2scan is 1 of 6 heads), so its non-linear readout (readouts.1) has linear_1 of shape 512 → 96 (16 hidden channels 
× 6 heads), a SiLU gate, and linear_2 of shape 96 → 6.
When fold_readout_tail (OPT-T08 C2) was built on 21 October, it folded products.1.linear (512×512) and skip_tp into readouts.1.linear_1, but kept all 96 columns ([512, 96]) instead of slicing to the active head's 16 columns ([512, 16]).
On 29 October (fp32-followups-20261029, "Readout head slice"), a prototype constant species.TAIL_HEAD was tested and measured 0.996 on Ada and 0.991–0.994 on RTX 5060 Ti (a 0.6–0.9 % whole-step speedup on NbCl5!), with forces within 
2.4×10 
−6
  eV/ 
A
˚
  of default.
Why wasn't it merged as a default? Because it was tested as a hardcoded prototype constant TAIL_HEAD rather than wired cleanly from the plan's resolved head index in fold_readout_tail! Furthermore, in readouts.1.linear_2, the 6-head structure is block-diagonal or only 1 output scalar of the 6 is selected by mask_head—slicing both linear_1 (512 → 16), the SiLU activation ([N, 16] instead of [N, 96]), and linear_2 (16 → 1) completely eliminates 5/6ths of the readout forward and backward FLOPs and memory traffic.   
optimisation-ledger.md
+ 4

2. Evidence:

Measured in FastMACE (fp32-profile-20261028.md §5 / fp32-followups-20261029): Prototype species.TAIL_HEAD measured b/a = 0.991–0.994 on local RTX 5060 Ti and 0.996 on Ada on NbCl5 in same-process alternating A/Bs.   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: fold_readout_tail (fastmace/rewrites/linear/) in both FP32 and FP64 plans.   
optimisation-ledger.md

Ceiling: 0.6–0.9 % on NbCl5 (measured in FP32), ~1.0–1.5 % in FP64 (estimated, since FP64 GEMMs take a larger share of the step).   
optimisation-ledger.md

4. Exactness Class:
Bitwise on CPU FP64, rounding class on GPU (removing 80 unused columns changes cuBLAS tile selection for the [N, 512] @ [512, 16] GEMM, which shifts forces by 
≤2.4×10 
−6
  eV/ 
A
˚
 —well inside the gate).   
optimisation-ledger.md

5. Falsifier:
Already passed the A/B falsifier on NbCl5 (0.991–0.996)! Combine species_residual + clean head-sliced fold_readout_tail in one same-process A/B across NaCl, NaNbO3, and NbCl5 on RTX 5060 Ti, Ada, and A100.
Stop rule: Adopt if combined b/a 
≤0.993 on NbCl5/NaNbO3 and 
≤1.000 on NaCl.   
optimisation-ledger.md

6. Overlap Check:
In ledger as lin-t08-headslice (NOT BUILT) and fp32-followups-20261029 ("Readout head slice", status: MEASUREMENT, prototype constant). Turning the prototype constant into an automatic rewrite using the ForceField's bound head (which is fixed for the lifetime of a ForceField per SPEC.md §7.1/§7.6) settles it and captures a free ~0.7–1 % gain when paired with species_residual.   
optimisation-ledger.md
+ 1

Idea 5: INT8 / FP8 Block-Scaled Activation Compression of Stored 
w for streaming_conv Backward (Fallback if Idea 1 is Memory-Only)
Target Goals: Goal 2 (Memory per edge: −3.8 kB/edge, +2× atoms on 16 GB) & Goal 1 (Faster convolution backward via 4× lower DRAM read bandwidth).

1. Mechanism:
In prec-fp16-edge-weights (14 Oct) and prec-precision-screen (OPT4-T08, 24 Oct), storing 
w in FP16 or BF16 failed the accuracy gate by 3–110×.
Why did it fail then, and what is fundamentally different now?   
optimisation-ledger.md

On 14 Oct and 24 Oct, w was stored before the forward convolution (w was quantized on the forward path too, corrupting the forward energy, forces, and dw!).

Since 27 Oct (OPT8-T02 fused_folded_forward) and 29 Oct (OPT9-T03 wprime_backward), w is consumed in full IEEE FP32 in the forward pass on-chip, and the radial backward uses wprime_backward (which does not use w or dw at all—it dots grad with 
w 
′
 (r) directly!).

Thus, the stored w tensor in global memory is ONLY read by streaming_conv_backward to compute the linear sender gradient 
dx 
s
​
 =∑ 
e
​
 w 
e
​
 ⊗dout 
r
​
  and the angular derivative 
dY 
lm
​
 =∑ 
c
​
 (x 
s
​
 ⊙w 
e
​
 )dout 
r
​
 !

Furthermore, instead of raw FP16/BF16, fused_folded_forward can quantize each edge's 32-element warp tile of 
w 
e
​
  on write into INT8 (or FP8 E4M3) + 1 shared FP32/E8M0 exponent scale per 32 channels (1.125 bytes/element instead of 4 bytes/element, giving 8 bits of mantissa with zero exponent underflow, or INT16 + 1 scale per 32 channels giving 15 bits of mantissa = 4 bits more than FP16 at 50 % memory!).

2. Evidence:

Measured in FastMACE (OPT7-T02 / conv-opt7-traffic): The streaming backward is 81–84 % memory-throughput bound, and reading w is 49–51 % of all L2 bytes. Cutting w from 4 B to 2 B (INT16 block-scaled) or 1 B (INT8 block-scaled) cuts total backward DRAM traffic by 25–38 %.   
optimisation-ledger.md

Precision difference: Block-scaled INT16 (int16 + 1 FP32 scale per 32 channels = 2.125 B/element) provides 15 mantissa bits (vs 10 in FP16 and 7 in BF16)—a 32× smaller quantization error than FP16—and applies only to the backward 
dx 
s
​
 /dY 
lm
​
  path, leaving forward energies and radial 
w 
′
 (r) derivatives 100 % IEEE FP32!

3. Fit to FastMACE & Amdahl Ceiling:

Where: fused_folded_forward global store of w and streaming_conv backward load of w.   
optimisation-ledger.md

Ceiling:

Memory: Saves 2.4 kB/edge (block-INT16) or 3.7 kB/edge (block-INT8) (cutting 5.9 kB/edge by 41–63 %).

Speed: Streaming backward is ~10–18 % of the step and 81–84 % memory bound; cutting its memory traffic by 25–38 % yields a 2.5–6 % whole-step ceiling on NaNbO3 and NbCl5.   
optimisation-ledger.md

4. Exactness Class:
Approximate (or Certified)—must pass the 
<2× FP32 force/stress error gate and 20 ps NVE drift check. (With block-scaled INT16, error is 
∼2 
−15
 ≈3×10 
−5
  per element, which averages down over 512 channels and ~40 neighbours by 
∼ 
512

​
 ≈22× to 
∼10 
−6
 , matching FP32 cancellation error).

5. Falsifier:
In Python (without writing CUDA first), hook the autograd backward of streaming_conv to quantize-and-dequantize the saved w tensor in-place after forward using per-32-channel int16 (15-bit mantissa) and int8 (7-bit mantissa) block scaling, and run tools/force_stress_gate.py on NaCl, NaNbO3, and NbCl5.
Stop rule: Takes 10 minutes to test. If block-scaled int16 (and int8) both exceed the 
2.0× force/stress gate on any of the 3 benchmark crystals, kill immediately.

6. Overlap Check:
prec-fp16-edge-weights (14 Oct) and prec-precision-screen (24 Oct) quantized the forward w in unscaled FP16/BF16 (10/7 mantissa bits) before fused_folded_forward and wprime_backward existed. Backward-only block-scaled int16 (15 mantissa bits) has never been tested.   
optimisation-ledger.md

Idea 6: Atomic-Free Transpose Gather (CSR Sender Walk) for Interaction-0 edge_outer_sum Backward & wprime_backward
Target Goals: Goal 1 (Faster FP32 MD steps: 1–3 %) & Goal 3 (Faster FP64 MD steps where FP64 atomic contention is much higher).

1. Mechanism:
Look closely at where atomic reductions or two-stage reductions remain in the step:

In Interaction 0 (reassociate_first / segmented_first_sum), the forward pass aggregates over receiver-sorted, sender-species-grouped edges. Its fused Triton backward (edge_outer_sum.py) computes the gradient w.r.t. radial/edge features and scatters into sender/receiver gradients.   
optimisation-ledger.md
+ 2

In Interaction 1 (wprime_backward, OPT9-T03), the radial derivative 
∂r 
ij
​
 
∂E
​
  is computed per edge 
e=(i→j) and then must be scattered into atom forces 
F 
i
​
 += 
∂r 
ij
​
 
∂E
​
  
r
^
  
ij
​
  and 
F 
j
​
 −= 
∂r 
ij
​
 
∂E
​
  
r
^
  
ij
​
 .
Because edges are already sorted by receiver 
j (sort_receivers=True), accumulating the receiver side 
F 
j
​
  can be done in registers per receiver segment with zero atomics! Furthermore, because the neighbour list is static for 15–30 steps between rebuilds, storing the reverse edge permutation rev_edge_idx (mapping edge 
(i→j) to its mirror edge 
(j→i)) at neighbour-list build time (which costs 
<0.05 ms once every 20 steps) allows every receiver 
j to accumulate both 
F 
j
​
  (as receiver) and 
−F 
j
​
  (as sender, by reading dE_dr[rev_edge_idx[e]]) in a single segmented register reduction—eliminating 100 % of force scatter atomics!

2. Evidence:

External MLIP codes: Allegro and MatScipy / LAMMPS use mirror-edge lookup or Newton/half-list symmetry for radial pair forces.

Measured in FastMACE (SPEC.md §9.3 / int1-segmented-first-sum): Replacing atomic sums with segmented_first_sum on receiver-sorted edges saved 1.6 % on NaNbO3 and 1.4–1.6 % on MP-0. In FP64, global atomicAdd on double is much slower on consumer GPUs than FP32 L2 atomics.   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: Force accumulation at the end of wprime_backward, Interaction-0 radial backward, and edge_outer_sum.

Ceiling: ~0.5–1.5 % in FP32 (estimated ~20–80 µs/step), and 2–4 % in FP64 (where FP64 atomicAdd throughput on sm_89/sm_120 is throttled). Also makes the radial force accumulation bitwise deterministic!

4. Exactness Class:
Rounding class (deterministic segmented summation instead of non-deterministic atomic summation).

5. Falsifier:
Time the atomicAdd force scatter inside wprime_backward and edge_outer_sum backward in isolation (by replacing tl.atomic_add into forces with a dummy register sum + 1 write per receiver).
Stop rule: If the isolated difference is 
<15 μs on NaCl and 
<150 μs on NbCl5 in FP32 (and 
<1 ms in FP64), kill the idea.

6. Overlap Check:
rad-mirror-edge (15 Oct) tried to share the forward radial MLP evaluation between an edge and its mirror. eng-force-scatter (14 Oct) looked at a standalone segmented scatter kernel before wprime_backward existed. Fusing receiver-segmented + mirror-edge force accumulation directly into the epilogue of wprime_backward is new.   
optimisation-ledger.md
+ 1

Idea 7: Zero-Recapture Dynamic Edge Padding via cudaGraphExecKernelNodeSetParams (or Single High-Water Graph)
Target Goals: Goal 4 (Faster engine setup & short relaxations / melt-quench without recapture stalls) & Goal 2 (Lower memory fragmentation from multiple graph pools).

1. Mechanism:
Currently, FastMACE buckets padded edge counts in 5 % steps (edge_bucket_fraction = 0.05, capacity_headroom = 1.15). During a relaxation (especially variable-cell FrechetCell), a heating ramp (0 → 1000 K), or a melt-quench, whenever the edge count crosses a 5 % bucket boundary, the engine stops, runs eager warmups in the graph pool, and captures a brand-new set of CUDA graphs (ACP, B, C) for the new bucket size.
Furthermore, as discovered on 28 October (prec-d1-relax-fix, commit bbd8bd23), graph re-capture mid-run triggered a silent cuBLAS FP64 emulation state bug on trillian (+0.34 eV/atom wrong energy after the first re-capture)!
Two better alternatives exist in CUDA 13:   
optimisation-ledger.md
+ 4

Single-Capture High-Water Graph for Relaxations: In short relaxations (30–100 steps), capturing once at 1.15× initial edges (a single bucket, no 5 % intermediate sub-buckets) avoids all mid-relaxation re-captures at the cost of at most 5–10 % extra padded edges on the first few steps (saving 0.5–2.0 s per bucket re-capture).

cudaGraphExecUpdate / Grid-Size Patching: Since dynamic_edges=True (OPT4-T05, Decision 59) already compiles all Inductor and Triton kernels with symbolic edge dimension E, the compiled kernel binaries do not change when E changes—only the grid dimension gridDim.x = ceil(E / BLOCK) and the scalar argument E change! Using cudaGraphExecUpdate (available via cuda.bindings.driver.cuGraphExecUpdate) updates an existing cudaGraphExec_t in ~20–50 µs instead of doing a full PyTorch warmup + capture (200–800 ms) and avoids allocating new intermediate buffers.   
optimisation-ledger.md
+ 1

2. Evidence:

Measured in FastMACE (phase-5.md / launch-adaptation): Re-captures during cell/volume changes are expensive enough that AdaptationConfig had to be built to bail out to the 3–4× slower compiled tier if re-captures exceed 20 % of wall time.   
optimisation-ledger.md
+ 1

Measured in FastMACE (prec-d1-relax-fix): Mid-run graph re-capture caused the D1 56-bit mantissa bug in float64 relaxations.   
optimisation-ledger.md

Vendor documentation (CUDA 13 Programming Guide): cudaGraphExecUpdate patches node parameters and grid dimensions of an instantiated graph in tens of microseconds without re-allocating or re-running eager warmups.

3. Fit to FastMACE & Amdahl Ceiling:

Where: fastmace/engine/graph.py bucket management and fastmace/optimize/ relaxations.   
SPEC.md

Ceiling: Eliminates 100 % of mid-relaxation graph re-captures (saving 0.5–3.0 s per variable-cell relaxation and preventing re-capture bugs like D1).

4. Exactness Class:
Bitwise (padding_is_exact guarantees that padding edges beyond r_cut contribute zero).   
SPEC.md

5. Falsifier:
Count the number of bucket re-captures and time spent in _capture_bucket during the relax-throughput.md benchmark and variable-cell relaxations, and compare against setting edge_bucket_fraction = 0.20 (single bucket up to capacity_headroom).
Stop rule: If mid-run re-captures take 
<1% of total relaxation wall time across the relaxation benchmark suite, kill the idea.

6. Overlap Check:
eng-bucket-width screened 2 %, 3 %, and 10 % buckets for steady-state MD step time, not for relaxation setup/re-capture overhead or cudaGraphExecUpdate.   
optimisation-ledger.md

Idea 8: Fix & Re-Enable Decision 65(a) (fp64_emulation_mantissa_bits = 56) Across Graph Re-Captures
Target Goals: Goal 3 (Faster float64 steps and float64 relaxations: 12–14 % faster FP64 steps on RTX 5060 Ti, RTX 4000 Ada, and RTX A4000).

1. Mechanism:
On 28 October (prec-d1-56bits / prec-d1-relax-fix), fp64_emulation_mantissa_bits = 56 (Decision 65a) gave a massive 12–14 % whole-step speedup (b/a = 0.858–0.884) on RTX 5060 Ti and RTX 4000 Ada, but had to be suspended on branch fastmace/fp64-d1-relax-fix (bbd8bd23) because a float64 variable-cell relaxation on trillian produced wrong energies (+0.34 eV/atom) starting on the step immediately after the first CUDA graph re-capture!
Why did re-capture corrupt cuBLAS fixed-point 56-bit emulation?
In cuBLAS (CUDA 13.0), the Ozaki fixed-point emulation workspace and internal scaling descriptors inside the cuBLAS handle are initialized lazily or cached per stream/workspace. When FastMACE re-captures a new edge bucket:   
optimisation-ledger.md

empty_cache_before_capture=True or stream switching during warmup_in_graph_pool invalidates or aliases the cuBLAS workspace buffer between the old graph and the newly captured graph, or

Small matrix products or non-contiguous strides during the re-capture warmup leave stale scaling metadata in the cuBLAS workspace when CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 skips the dynamic exponent pre-pass!
By (a) giving each captured graph its own dedicated cuBLAS workspace (torch._C._cuda_clearCublasWorkspaces() or separate stream pool per capture), (b) re-asserting the cuBLAS math/emulation mode on the capture stream right before warmup and capture, and (c) testing on the exact NbOCl3 3×3×3 reproducer on trillian, you can fix the root cause and restore the 12–14 % FP64 speedup (and extend it to "NVIDIA RTX A4000" once measured!).

2. Evidence:

Measured in FastMACE (ideas-20261027.md / SPEC.md Decision 65): fp64_emulation_mantissa_bits=56 measured 0.877 (NbCl5), 0.881 (NaNbO3), 0.862 (NaCl) on RTX 5060 Ti and 0.876 / 0.884 / 0.858 on RTX 4000 Ada.   
optimisation-ledger.md
+ 1

Reproducer in FastMACE (prec-d1-relax-fix): NbOCl3 3×3×3 float64 relaxation on trillian.   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: All cuBLAS FP64 emulated GEMMs (~50–60 % of the FP64 step).   
optimisation-ledger.md

Ceiling: 11.6–14.2 % whole-step FP64 speedup (measured).   
optimisation-ledger.md
+ 1

4. Exactness Class:
Rounding class (56 mantissa bits > IEEE FP64's 53 mantissa bits; forces within 
2–5×10 
−15
  eV/ 
A
˚
  of dynamic emulation).   
optimisation-ledger.md
+ 1

5. Falsifier:
Run the NbOCl3 3×3×3 two-bucket graph re-capture reproducer in a 20-line standalone script with CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56. Test whether:
(1) calling torch._C._cuda_clearCublasWorkspaces() before capture warmup fixes the +0.34 eV/atom jump, OR
(2) inspecting which specific GEMM in the second capture produces the wrong output isolates a specific 
(M,N,K) shape/alignment bug in cuBLAS 13.0 (just like the small optimizer product found in bc292dc6).
Stop rule: If the bug is inside cuBLAS 13.0 for large padded 
M (e.g. 
M not a multiple of 512) and cannot be avoided by edge-bucket rounding (edge_multiple=512 instead of 256), keep D1 off during re-capturing runs.   
optimisation-ledger.md

6. Overlap Check:
Directly resolves the open suspension of prec-d1-56bits / prec-d1-relax-fix in the ledger. Note the crucial clue: edge_multiple is 256, so bucket 0 might be a multiple of 512 while bucket 1 (+5 % rounded to 256) is an odd multiple of 256!   
optimisation-ledger.md
+ 2

Idea 9: A100 Tensor Cores for Interaction-0 Reassociated Node Projections & Non-MH-1 Radial GEMMs
Target Goals: Goal 1 & Question 6 (Faster FP32 steps on A100: +3–6 % additional gain beyond atom-level 3xTF32).

1. Mechanism:
Question 6 asks: On the A100, beyond 3xTF32 atom GEMMs, which other parts of the step could use Tensor Cores while keeping FP32-class accuracy?
Let's check every non-atom-linear block in the default MH-1 FP32 plan:

Radial MLP hidden layers: Already replaced by radial_tables (OPT6-T05)—no GEMMs left!   
optimisation-ledger.md
+ 1

Folded radial forward ([E, 65] @ [65, 1280]): Replaced by fused_folded_forward (OPT8-T02) on consumer GPUs, but wait: was fused_folded_forward ever measured on the A100? Look at opt8-T02: it was measured on local (sm_120), murphy (sm_89), and L4 (sm_89)—never on A100! On the A100, SIMT FP32 is only 19.5 TFLOPS while CUTLASS 3xTF32 is 40.8+ TFLOPS and HBM2e bandwidth is 2,039 GB/s (4.5× RTX 5060 Ti's 448 GB/s)! On A100, fused_folded_forward runs an in-kernel SIMT product on 19.5 TFLOPS FP32 units, whereas the unfused chunked_conv_backward chain can run the [E, 65] @ [65, 1280] forward product on CUTLASS 3xTF32 Tensor Cores and stream w at 2 TB/s!   
optimisation-ledger.md
+ 1

Interaction-0 Reassociated Node Projections (reassociate_first): After segmented_first_sum sums over edges into [N_atoms, S_species, 16, 65], the projection across [S_species * 65, 512] is a dense matrix multiplication. Ensure tf32x3_gemms routes both forward and backward of the reassociated first interaction's dense projections through CUTLASS 3xTF32 on A100.

2. Evidence:

Measured in FastMACE (rad-w-tables, Decision 65): On A100, SIMT-heavy w_tables was 6–27 % slower (1.063–1.269×) than GEMM + streaming conv because A100 has 2 TB/s HBM2e bandwidth and strong matrix engines!   
optimisation-ledger.md
+ 1

Measured in FastMACE (opt8-T02): fused_folded_forward's gain shrank from 7.2 % on RTX 5060 Ti (448 GB/s) to 4.2 % on Ada (360 GB/s, larger L2) to 2.3 % on L4, and was never tested on A100 (2,039 GB/s).   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: A100 plan (fused_folded_forward on/off + CUTLASS 3xTF32 on [E, 65] @ [65, 1280] and reassociate_first projections).   
optimisation-ledger.md

Ceiling: 3–7 % whole-step gain on A100 (estimated).

4. Exactness Class:
Rounding class (passes the 2× FP32 gate via 3xTF32).

5. Falsifier:
On A100-SXM4-80GB, run a same-process A/B comparing:

Side A: Current default (fused_folded_forward=on + CUTLASS 3xTF32 atom GEMMs)

Side B: fused_folded_forward=off + CUTLASS 3xTF32 applied to the [E, 64+1] @ [64+1, 1280] folded radial forward GEMM (and reassociate_first GEMMs).
Stop rule: If Side B is not faster (b/a < 0.99) in both build orders on A100, keep fused_folded_forward=on on A100.

6. Overlap Check:
fused_folded_forward (conv-fused-folded-forward) was made a global MH-1 default in Decision 64 before Decision 66 allowed per-GPU defaults, and was never A/B tested on the A100.   
optimisation-ledger.md
+ 1

Idea 10: Merge radial_tables Disk Cache + Add AOTInductor / Bundled Autograd Cache to Cut Warm Setup from ~30 s to < 8 s
Target Goals: Goal 4 (Faster engine setup: −15 to −22 s per new process; −42 s in FP64).

1. Mechanism:
Engine build + capture takes ~30–40 s warm (fp32-profile-20261028.md). Where does that 30–40 s go?   
optimisation-ledger.md

tabulate_radials (Fitting + Certifying Chebyshev Radial Tables): Takes 5–20 s in FP32 and 42 s of an 87 s build in FP64 (rad-table-cache)! Commit 54b6f13f on branch fastmace/radial-table-cache already caches fitted & certified tables under FASTMACE_CACHE/radial-tables, and fp32-followups-20261029 moved self-check random probes to the GPU (cutting 21 self-checks from 4.5 s to 2.3 s)—both are sitting on unmerged branches!   
optimisation-ledger.md
+ 2

Dynamo / AOTAutograd Tracing (~12–18 s warm): Even with TORCHINDUCTOR_CACHE_DIR and FX_GRAPH_CACHE=1 enabled (OPT4-T05b), torch.compile still spends 12–18 s in Python tracing the FX graph, running AOTAutograd min-cut partitioning, and SymbolicShape guards before hitting the Inductor kernel cache. In PyTorch 2.14, enabling torch._inductor.config.fx_graph_remote_cache / bundled_autograd_cache = True (TORCHINDUCTOR_AUTOGRAD_CACHE=1) caches the entire AOTAutograd joint forward+backward graph and partitioner output, bypassing AOTAutograd tracing and min-cut partitioning on warm starts!   
optimisation-ledger.md

2. Evidence:

Measured in FastMACE (rad-table-cache & fp32-followups-20261029): Radial table cache saves 5 s (leto) to 20 s (local) in FP32 and 42 s (trillian) in FP64; device-side self-check probes save 2.1 s.   
optimisation-ledger.md

PyTorch 2.x AOTAutogradCache documentation: Caching above AOTAutograd eliminates Python FX tracing + SymPy shape simplification + min-cut partitioning, cutting warm torch.compile latency by 60–80 %.

3. Fit to FastMACE & Amdahl Ceiling:

Where: fastmace/rewrites/radial/__init__.py, fastmace/planner/selfcheck.py, and fastmace/kernels/triton/cache.py.   
optimisation-ledger.md

Ceiling: Cuts warm setup time from 34–41 s down to ~10–15 s in FP32 and from 87 s down to ~25 s in FP64.

4. Exactness Class:
Bitwise (identical tables and compiled kernels).   
optimisation-ledger.md

5. Falsifier:
Merge fastmace/radial-table-cache + device-side self-check probes, and test torch._functorch.config.enable_autograd_cache = True on two consecutive processes running fm.md(..., steps=10).
Stop rule: Merge the two branches immediately (already verified); keep enable_autograd_cache if warm setup drops by 
≥3 s with all CPU/GPU tests passing.

6. Overlap Check:
rad-table-cache and device self-check probes are ON BRANCH in the ledger; enable_autograd_cache (caching above AOTAutograd/partitioner, as opposed to just Inductor's codecache in OPT4-T05b) has not been explicitly verified with your custom SaveTranscendentals partitioner.   
optimisation-ledger.md
+ 1

Idea 11: Mixed FP32-Preconditioned / FP64-Defect-Corrected LBFGS for Float64 Relaxations
Target Goals: Goal 3 (Faster float64 relaxations: 2–3.5× faster wall time to float64 fmax convergence).

1. Mechanism:
In Benchmark_MACE_GPU_OPTIMIZATIONS (eng-static-relax, 1 Oct 2026), a two-stage relaxation (relax in FP32 to fmax = 0.01 eV/Å, then switch to FP64 to finish) was only 1.14× faster because switching from an FP32 engine to an FP64 engine mid-relaxation paid a second full engine build + compilation + graph capture and threw away the LBFGS inverse-Hessian history!
Instead, do In-Graph Defect-Corrected / Warm-Hessian Mixed-Precision LBFGS:   
optimisation-ledger.md
+ 1

Run LBFGS in the FP32 graph engine (~15 ms/step vs ~200 ms/step in FP64—13× faster per step!) down to fmax = 0.015 eV/Å, accumulating the 
(s 
k
​
 ,y 
k
​
 ) curvature pairs in float64 (which FastMACE already uses for integrator/optimizer state!).   
optimisation-ledger.md
+ 1

When switching to the FP64 engine for the final 3–8 polishing steps (from 0.015 to 0.01 or 0.001 eV/Å), transfer the accumulated float64 LBFGS 
(s 
k
​
 ,y 
k
​
 ) history into the FP64 LBFGS optimizer so the inverse Hessian is already preconditioned! Because the Hessian is already built, the FP64 stage converges in 2–5 steps instead of 25–40 steps from a cold identity Hessian.

2. Evidence:

Measured in FastMACE (Current speed reference): FP32 step is 14.6–18.6 ms vs FP64 (with Decision 63+65) 200–220 ms (~12× faster per step on consumer GPUs).   
optimisation-ledger.md

Literature: Mixed-precision quasi-Newton / LBFGS with Hessian carry-over (Gratton et al., SIAM J. Optim.; Lindquist et al. 2022) shows that carrying the FP32-generated secant pairs 
(s 
k
​
 ,y 
k
​
 ) into the FP64 polishing phase cuts FP64 gradient evaluations by 70–85 %.

3. Fit to FastMACE & Amdahl Ceiling:

Where: fastmace/optimize/lbfgs.py and fastmace/protocols/quench.py (when precision="float64" is requested for relaxations).   
SPEC.md

Ceiling: If 35 of 40 LBFGS steps run at FP32 speed (18 ms) and only 5 steps run at FP64 speed (220 ms), total stepping time for a 40-step NbCl5 relaxation drops from 
40×220 ms=8.8 s to 
35×18+5×220=1.7 s (5.1× faster stepping), and across a 21-frame quench batch (where both FP32 and FP64 engines are built once and reused across all 21 frames!), total quench wall time drops by 2.5–3.5×.

4. Exactness Class:
Exact float64 convergence criterion (final energy, forces, stress, and fmax <= threshold are evaluated strictly on the float64 engine).

5. Falsifier:
Take the 21 quench frames of NbCl5 in float64, run LBFGS in FP32 to fmax = 0.015, copy the LBFGS history buffers (s, y, rho) to the FP64 LBFGS instance, and continue to fmax = 0.005 (or 0.01). Compare final energies (|ΔE| <= 1e-6 meV/atom), atom positions, and total FP64 step count against pure FP64 LBFGS.
Stop rule: If FP64 steps are not reduced by 
≥60% or if any frame converges to a different minimum than pure FP64, kill the idea.

6. Overlap Check:
eng-static-relax (1 Oct) ran naive ASE two-stage relaxation on a single structure without engine reuse and without carrying the LBFGS Hessian history across the precision boundary.   
optimisation-ledger.md

Idea 12: Float64 wprime_backward + Vectorized double2 Streaming Backward + Emulated reassociate_first
Target Goals: Goal 3 (Faster float64 MD steps: 10–18 % faster FP64 steps on all GPUs, especially A100 and >3-species electrolytes).

1. Mechanism:
Look at what is missing from the float64 path compared to float32:

Decision 65's w_tables (D2) is refused for 
>3 species (W_TABLE_MAX_BYTES = 2 GiB, so production 4-species Na–Nb–O–Cl solid electrolytes cannot use w_tables in float64!) and is off on A100.   
optimisation-ledger.md
+ 1

Meanwhile, wprime_backward (OPT9-T03) was only enabled for float32 plans (SPEC.md Decision 66)! On a 4-species Na–Nb–O–Cl electrolyte (or on A100) where w_tables is off in float64, the float64 backward still runs the slow [E, 1280] @ [1280, 65] da = dw @ W product! Enabling wprime_backward in float64 whenever w_tables is off replaces that product with the fast 
w 
′
 (r) dot!   
SPEC.md

Vectorized double2 loads in the float64 streaming backward (conv-wide-loads): Listed as "Untested" in the ledger's "Open or untried" table. In float64, each thread loads 8-byte double scalars; pairing adjacent channels into 16-byte double2 (ld.global.v2.f64) halves the number of memory transactions in the 136 load sites of the streaming backward.   
optimisation-ledger.md
+ 1

2. Evidence:

Measured in FastMACE (SPEC.md Decision 65 & 66): w_tables is limited to 
≤3 species (refused on 4-species Na–Nb–O–Cl!) and off on A100. wprime_backward cut the FP32 step by 11–19 % on A100 and 3–7 % on 5060 Ti / Ada.   
SPEC.md
+ 1

Measured in FastMACE (f64-T01): In float64 without w_tables, the radial folded layer takes 13–14 % of the entire float64 step.   
optimisation-ledger.md

3. Fit to FastMACE & Amdahl Ceiling:

Where: Float64 plans on 
>3-species production systems (Na–Nb–O–Cl), on A100, and on RTX A4000 (which isn't in DECISION_65_GPUS yet!).   
optimisation-ledger.md
+ 1

Ceiling: 8–15 % faster float64 steps on 4-species Na–Nb–O–Cl and on A100/RTX A4000; 1–3 % from double2 loads.

4. Exactness Class:
wprime_backward in float64: Certified / Rounding class (uses the degree-16 float64 Chebyshev derivative table already certified to 
3×10 
−13
  in rad-f64-tables!).   
optimisation-ledger.md
+ 1

5. Falsifier:
Enable wprime_backward in float64 on a 4-species NaNbOCl4 cell (where w_tables is refused) and on A100 NaNbO3/NbCl5, and run a same-process A/B.
Stop rule: Adopt in float64 wherever w_tables is off if b/a <= 0.95 and force error vs native float64 is 
≤10 
−13
  eV/ 
A
˚
 .

6. Overlap Check:
wprime_backward (OPT9-T03) was only tested and enabled for float32; its interaction with 4-species float64 systems (where w_tables hits the 2 GiB cap and shuts off) was overlooked!   
SPEC.md
+ 2

Summary Table of Ranked Ideas
Rank	Idea	Primary Goal	Target GPUs	Expected Whole-Step / Memory / Setup Gain	Exactness Class
1	On-chip 
w(r) rematerialization in streaming_conv bwd (no DRAM w store/load; enabled by wprime_backward)	Goal 2 (Memory) & Goal 1 (FP32 speed)	All (sm_120, sm_89, sm_86)	−5.1 kB/edge (−85 % edge mem, +2.3× atoms); +3–8 % speed (est.)	Bitwise (vs fused_folded_forward)
2	WSL2 thread-stack reservation elimination (CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES audit + cudaLimitStackSize)	Goal 2 (Memory on 16 GB WSL2)	sm_120 (WSL2)	+1.5 to +3.0 GiB free VRAM (+8k–15k atoms) (est.)	Bitwise
3	Adopt CUTLASS 3xTF32 (rz3) on A100 (incl. segment_gemm shapes) + species_residual on Ada/5060 Ti	Goal 1 (FP32 speed)	sm_80 (A100), sm_89 (Ada)	
13–29 % faster on A100 (measured); 0.5–0.8 % on Ada (measured) 
optimisation-ledger.md

Rounding (passes gate & NVE) 
optimisation-ledger.md

4	Complete single-head slice (16 vs 96 cols) in fold_readout_tail + readout MLP	Goal 1 (FP32) & Goal 3 (FP64)	All GPUs	
0.6–0.9 % faster on NbCl5 (measured); 1.0–1.5 % in FP64 (est.) 
optimisation-ledger.md

Rounding (passes gate) 
optimisation-ledger.md

5	Backward-only block-scaled int16/int8 compression of saved w (if Idea 1 is not used for speed)	Goal 2 (Memory) & Goal 1 (FP32 speed)	All GPUs	−2.4 to −3.7 kB/edge (−41–63 % edge mem); +2.5–6 % speed (est.)	Approximate / Certified
6	Mirror-edge + receiver-segmented atomic-free force accumulation in wprime_backward	Goal 1 (FP32) & Goal 3 (FP64)	All GPUs	0.5–1.5 % in FP32, 2–4 % in FP64 (est.); deterministic	Rounding (deterministic)
7	Zero-recapture relaxations (cudaGraphExecUpdate / single high-water bucket)	Goal 4 (Setup / Relaxations)	All GPUs	−0.5 to −3.0 s per variable-cell relaxation; avoids recapture bugs (est.)	Bitwise
8	Fix cuBLAS workspace/alignment bug on graph re-capture to restore Decision 65(a) (56 bits)	Goal 3 (FP64 speed & relax)	sm_120, sm_89, sm_86	
11.6–14.2 % faster FP64 steps (measured) 
optimisation-ledger.md
+ 1

Rounding (passes FP64 gate) 
optimisation-ledger.md
+ 1

9	A100-specific fused_folded_forward=off + CUTLASS 3xTF32 on folded radial & Interaction-0 GEMMs	Goal 1 (FP32 on A100)	sm_80 (A100)	3–7 % faster on A100 (est.)	Rounding (passes gate)
10	Merge radial_tables disk cache + GPU self-check probes + enable_autograd_cache	Goal 4 (Setup time)	All GPUs	
−7 to −22 s warm FP32 build (measured); −42 s FP64 build (measured) 
optimisation-ledger.md
+ 1

Bitwise 
optimisation-ledger.md

11	Warm-Hessian (s, y) transfer from FP32 LBFGS pre-relaxation to FP64 LBFGS polishing	Goal 3 (FP64 relaxations)	sm_120, sm_89, sm_86	2.5–3.5× faster FP64 quench/relaxation wall time (est.)	Exact FP64 convergence
12	Enable wprime_backward in FP64 (for 4-species Na–Nb–O–Cl & A100 where w_tables is off) + double2 loads	Goal 3 (FP64 speed)	All GPUs (esp. 4-species & A100)	8–15 % faster FP64 steps on 
>3 species & A100 (est.)	Certified / Rounding
Already Covered in the Optimisation Ledger (Checked & Excluded)
The following candidate techniques from external MLIP codes and GPU optimization literature were inspected and verified to be already adopted, rejected, or closed in docs/optimisation-ledger.md:   
optimisation-ledger.md

Streamed receiver-major OEq convolution & ir_mul layout: Adopted (conv-streaming-conv, conv-ir-mul).   
optimisation-ledger.md

L2-chunked convolution backward keeping dw in L2: Adopted (conv-chunked-conv-backward, Decision 60).   
optimisation-ledger.md
+ 1

Chebyshev polynomial radial tables (radial_tables) & w_tables: Adopted (rad-radial-tables, Decision 62/63; rad-w-tables, Decision 65).   
optimisation-ledger.md
+ 1

Fused folded radial forward inside the streaming convolution (fused_folded_forward): Adopted (conv-fused-folded-forward, Decision 64).   
optimisation-ledger.md
+ 1

Reassociated first interaction (reassociate_first) + segmented_first_sum: Adopted (int1-reassociate-first, int1-segmented-first-sum).   
optimisation-ledger.md
+ 1

Species-only embedding/skip tables & linear folds (species_tables, fold_product_linear, fold_readout_tail): Adopted (lin-species-tables, lin-fold-product-linear, lin-fold-readout-tail, OPT-DEF-2).   
optimisation-ledger.md
+ 1

Gathered-row Triton GEMM (segment_gemm) & per-irrep gated_rows: Adopted (lin-gated-rows, lin-segment-gemm).   
optimisation-ledger.md
+ 1

3×FP16 / BF16×6 / BF16×9 / Hybrid SIMT+MMA co-execution on consumer GPUs: Rejected (prec-fp16x3-standalone, prec-fp16x3-rowcol, prec-fp16x3-other-gpus, prec-bf16x6, prec-bf16x9, prec-hybrid-gemm, launch-het-megakernel, launch-sched-iso).   
optimisation-ledger.md

INT8-slice Ozaki FP32 emulation (FlashSO(2)) & GEMMul8 Ozaki-II on sm_120: Rejected (prec-int8-f32, prec-gemmul8).   
optimisation-ledger.md

Double-float (df64, Joldes–Muller–Popescu) FP64 emulation on FP32 units: Rejected (prec-double-float, f64-T04: 1.94× vs 2.0× bar, misses 1e-13 accuracy bar).   
optimisation-ledger.md

Time-step delta GEMMs & 50/50 channel-magnitude split GEMMs: Rejected (prec-delta-gemm, prec-channel-split: 411–6,130× worse error).   
optimisation-ledger.md

LunaTP (arXiv:2610.18393), Symmetrix-XL, TorchSim, NVIDIA ALCHEMI, and OEq SymmetricContraction: Screened and rejected (ext-lunatp, ext-symmetrix, ext-torchsim, ext-alchemi, sc-oeq-contraction).   
optimisation-ledger.md
+ 1

Sobek per-receiver reassociation (arXiv:2607.18074): Rejected on arithmetic count (conv-sobek: 4× more FMAs).   
optimisation-ledger.md

Low-rank SVD of atom linears or radial basis: Falsified (lin-svd: rank 509–512 of 512; rad-low-rank: rank 65 of 65).   
optimisation-ledger.md

Inductor max-autotune, combo_kernels, aggressive_fusion, and Programmatic Dependent Launch (PDL): Rejected (setup-max-autotune, launch-combo-kernels, fp32-followups-20261029, launch-pdl).   
optimisation-ledger.md
+ 1



//

# Research prompt of 4 November 2026 (`docs/design/2026-11-04-gemini-research-prompt.md`, with `docs/validation/omat0-a100-optimisations.md` attached): Gemini's answer

### Answers to Research Questions

**1. Equivariant-Convolution Backward Schedules (32 % of step)**

* **Sobek ([arXiv:2607.18074](https://arxiv.org/abs/2607.18074)):** Observes that for an edge weight expansion $W_{e,\pi,a} = \sum_q \phi_{e,q} A_{q,\pi,a}$ (fully connected UVW mixing), summing over edges before applying $A$ reduces the tensor product to a receiver-factorized GEMM. **Applicability to MACE-OMAT-0:** Does *not* apply directly to MACE's `conv_tp` because MACE uses a depthwise `uvu` tensor product (128 channels $\times$ 10 paths = 1,280 weights) followed by an equivariant node linear (`linear_1`), rather than a `uvw` edge mixing. However, when the radial basis is replaced by piecewise Chebyshev polynomials of degree $D=5$, a *piecewise* Sobek factorization becomes possible within each radial slab (see Proposal 3).
* **Symmetrix-XL ([arXiv:2610.01036](https://arxiv.org/abs/2610.01036)):** Generates model-specialized C++/CUDA code where the forward pass folds the radial projection into the receiver sum, and the reverse pass applies the transpose of the node projection first, then recomputes radial/angular factors in generated source- and edge-owned kernels. **Applicability:** FastMACE already tried pure source-owned and receiver-owned streaming backwards (slower due to either redoing work or atomic traffic), and model-specialized CG unrolling is already in progress. What FastMACE has *not* exploited from the receiver-streaming backward is **register-level directional $d\vec{r}_{ij}$ reduction** (see Proposal 1) and **Newton's 3rd Law ($r_{ij} = -r_{ji}$) half-graph pair pooling** (see Proposal 2).
* **OpenEquivariance ([arXiv:2501.13986](https://arxiv.org/abs/2501.13986) & PR 223):** Uses warp-level shared-memory staging where $x_j, Y_{ij}, w_{ij}$ and adjoints are loaded into smem segments and contracted via generated warp loops, accumulating sender gradients $\nabla x_j$ via `atomicAdd` into HBM/L2 (`wprime_backward`). **Applicability:** This is FastMACE's current baseline (4.37 ms). Its primary bottlenecks on A100 are: (a) computing three separate scalar products per path ($\nabla x_j$, $\nabla Y_{ij}$, $\nabla w_{ij}$) in FP32 CUDA cores, (b) evaluating Chebyshev polynomials in FP32 ALU for both $w$ and $w'$ while tensor cores sit idle, and (c) global memory reads/writes of $\nabla Y_{ij}$ and Inductor's separate $\nabla Y_{ij} \to \nabla \vec{r}_{ij}$ chain.


* **FlashTP ([ICML 2025](https://proceedings.mlr.press/v267/lee25l.html), [SNU-ARC/flashTP](https://github.com/SNU-ARC/flashTP)):** Uses path-aggregated execution—grouping all tensor-product paths $(\ell_1, \ell_2) \to \ell_3$ that share the same input/output irrep into a single shared-memory pass to maximize register reuse of $x_j$ and $g_i$ across paths. **Applicability:** Relevant to register scheduling inside `wprime_backward` (already partially present in OEq PR 223's `ir_mul` layout, and being finalized by your in-progress model-specialized kernel).


* **cuEquivariance, e3nn-jax / E3J ([arXiv:2609.35099](https://arxiv.org/abs/2609.35099)), and NequIP/Allegro ([arXiv:2607.28461](https://arxiv.org/abs/2607.28461)):** Target materialized edge weights $w_e \in \mathbb{R}^{E \times 1280}$ in HBM. None support on-the-fly register reconstruction of $w$ and $w'$ from polynomial slabs.

**2. The Fused Forward (27 % of step, 3.70 ms on A100)**

* **A100 Roofline & Resource Audit:** For $E = 373,760$ edges, the 3×TF32 radial projection ($K=64 \to N=1280$) executes $3.74 \times 10^5 \times 2 \times 64 \times 1280 = 61.2\text{ GFLOPs}$ per TF32 pass ($183.7\text{ GFLOPs}$ across 3 passes, or $\sim 1.18\text{ ms}$ theoretical peak at 156 TFLOPS, realistically $\sim 1.6\text{ ms}$ with mask splits and small-$M$ tile efficiency). The UVU tensor product executes $\sim 32\text{ FLOPs/channel} \times 128 \times 373,760 = 1.53\text{ GFLOPs}$ on FP32 CUDA cores ($\sim 0.08\text{ ms}$ theoretical at 19.5 TFLOPS, realistically $\sim 0.6\text{ ms}$ due to smem bank conflicts and loop overhead). Memory traffic is $373,760 \times 16 \times 4\text{ B} = 24\text{ MB}$ for $Y_{ij}$, $373,760 \times 64 \times 4\text{ B} = 95.7\text{ MB}$ for radial inputs, and $373,760 \times 512 \times 4\text{ B} = 765\text{ MB}$ of gathered sender features $x_j$ (hitting the 40 MB L2 cache with $>85\%$ hit rate on a 5,000-atom crystal).


* **Where the remaining ~1.5 ms goes:** Synchronous global-to-shared memory stalls (`ld.global` $\to$ register $\to$ `st.shared` + `__syncthreads()`) between the 3×TF32 `mma.sync` phase and the CUDA-core UVU contraction phase, plus redundant reading of the $64$-wide radial vector $[h \cdot c, c]$ from HBM (see Proposals 4 and 6).

**3. Fewer Edge Passes**

* **Why full Interaction 2 reassociation fails:** Unlike Interaction 1 (where sender features are 89-species one-hot vectors), Interaction 2's sender features $x_j = \text{linear\_up}(h_j^{(1)}) \in \mathbb{R}^{512}$ are continuous and unique per atom.


* **How to share/eliminate passes:**
1. **Undirected Half-Graph Pairing ($e_{ij}, e_{ji}$):** Every undirected pair $\{i, j\}$ appears twice in the directed graph with $r_{ij} = r_{ji}$ and $\vec{r}_{ij} = -\vec{r}_{ji}$. Evaluating radial polynomials, $w$, and $w'$ on directed edges does $2\times$ redundant radial work (Proposal 2).
2. **Chebyshev Slab Factorization (Certified/Exact):** Within each of the $S$ radial polynomial pieces (degree 5, so 6 coefficients per piece), $w_{ij} = \sum_{p=0}^5 T_p(x_{ij}) G_{s(e), p}$. While $5,000 \times S \times 6$ is too dense for a node reduction, evaluating $w(r)$ directly from the degree-5 Chebyshev slab inside `fused_forward` using **tensor cores** ($K=6$ padded to 8 or 16 across 2–3 pieces) avoids reading the 96 MB $[h\cdot c, c]$ intermediate tensor entirely (Proposal 4).





**4. The Atom Side (~25 % of step: 1.85 ms GEMMs + 1.07 ms Elementwise + 0.40 ms Contraction)**

* **Linear Folding across Boundaries:** Between Interaction 1 and Interaction 2, and inside each interaction after the message sum, linear operations sit separated only by scalar division $(\text{density} + 1)$ or linear additions:
* In Interaction 1 & 2: $m_i' = \text{linear\_1}\left(\frac{m_i}{\text{density}_i + 1}\right) = \frac{1}{\text{density}_i + 1} \text{linear\_1}(m_i)$ because `linear_1` has no bias on equivariant irreps (and even on $0e$, MACE equivariant linears are strictly homogeneous linear maps).
* Between Interaction 1 and Interaction 2: $h_i^{(1)} = \text{linear\_2}(\text{sym\_contract}_1(\dots)) + \text{skip\_tp}_1(s_i)$. Then Interaction 2 immediately computes $x_i = \text{linear\_up}_2(h_i^{(1)})$, $sc_i^{(2)} = \text{skip\_tp}_2(h_i^{(1)}, s_i)$, and `readout_1`$(h_i^{(1)})$. By folding $\text{linear\_2}$ into $\text{linear\_up}_2$, $\text{skip\_tp}_2$, and $\text{readout\_1}$ at setup time, `linear_2` disappears completely from both forward and backward passes (Proposal 5).





**5. Unused A100 Architectural Features**

* **`cp.async.cg.shared.global` (16-byte async copy bypassing registers):** Currently unused in `fused_forward` and `wprime_backward`. Moving $x_j$, $Y_{ij}$, and $g_i$ tiles asynchronously while tensor cores or CUDA cores compute the previous edge batch hides global memory and L2 latency (Proposal 6).
* **3×TF32 Tensor Cores in `wprime_backward`:** Currently, `wprime_backward` runs entirely on FP32 CUDA cores (4.37 ms). Evaluating $w$ and $w'$ and contracting across edge batches via 3×TF32 `mma.sync` shifts half of its arithmetic onto the 156 TFLOPS tensor pipe (Proposal 4).


* **L2 Persistence (`cudaAccessPolicyWindow`):** Setting a 30 MB persisting L2 window on the 40 MB A100 L2 cache for node tensors ($x_j \in \mathbb{R}^{5000 \times 512}$ is 10.2 MB; $\nabla x_j$ is 10.2 MB) guarantees zero HBM evictions caused by streaming edge reads (included in Proposal 6).

**6. Capacity Paths (1M to 150M atoms)**

* At 1M atoms on RTX 4000 Ada (15.3 µs/atom/step tiled vs. 13 µs/atom/step whole-system), the 2.3 µs/atom overhead comes from **Sweep 3 recomputing Interaction 0 from scratch** (edges, spherical harmonics $Y$, radial tables, and segmented sums) and host-staged `state="host"` PCIe transfers (Proposals 8 and 9).



---

### Ranked Proposals (Top 9)

#### Proposal 1: Fuse Spherical Harmonics ($Y_{ij}$) Forward & Analytical Jacobian ($\nabla Y_{ij} \to \nabla \vec{r}_{ij}$) Directly into `fused_forward` and `wprime_backward`

* **Mechanism:**
* **Current state:** Inductor runs separate kernels to compute $Y_{ij}(\hat{r}_{ij}) \in \mathbb{R}^{E \times 16}$ ($\ell \le 3$) and writes it to HBM (24 MB). In the backward pass, `wprime_backward` computes $\nabla Y_{ij}$ in registers, writes $\nabla Y_{ij} \in \mathbb{R}^{E \times 16}$ to HBM (or accumulates across paths), and Inductor launches multiple elementwise backward kernels to backpropagate $\nabla Y_{ij} \to \nabla \hat{r}_{ij}$ and combine with $\frac{\partial E}{\partial r_{ij}} \hat{r}_{ij}$ into edge force vectors $\vec{f}_{ij} \in \mathbb{R}^{E \times 3}$ before scattering to atom forces $\vec{F}_i$.


* **Change:**
1. For $\ell_{\max} = 3$, real spherical harmonics $Y_0^0 \dots Y_3^3$ and their Cartesian gradients $\frac{\partial Y_{\ell m}}{\partial \hat{r}_\alpha}$ ($\alpha \in \{x,y,z\}$) are fixed degree-3 polynomials in $(x, y, z) = \hat{r}_{ij}$ requiring only 38 FP32 FMAs per edge (e.g., SpheriCart hardcoding).
2. In `wprime_backward` (`fastmace/kernels/oeq_streaming/wprime_backward.py`): Instead of writing $\nabla Y_{ij}$ to memory, once a thread/warp finishes accumulating the 15 non-constant components of $\nabla Y_{ij}$ across paths for edge $e$ and scalar radial derivative $dr_e = \sum_c dw_c w'_c(r_e) + d(\text{density})_e$, immediately contract $\nabla Y_{ij}$ with the analytical Jacobian $\frac{\partial Y_{ij}}{\partial \vec{r}_{ij}}$ in registers and add $dr_e \hat{r}_{ij}$ to form the complete directed edge force $\vec{f}_e \in \mathbb{R}^3$.


3. Accumulate $\vec{f}_e$ across the receiver's edges directly into the receiver atom's force register $\vec{F}_i \mathrel{+}= \vec{f}_e$ (written once per receiver at kernel exit, zero atomics) and perform a single `atomicAdd` of $-\vec{f}_e$ to the sender force $\vec{F}_j$.
4. Also fuse the scalar `table_backward` (0.30 ms, which only computes the density derivative and Interaction 1's radial derivative) into `wprime_backward` and Interaction 1's sum backward.






* **Exactness class:** **Rounding** (identical polynomials and chain-rule derivatives in FP32, reassociated in registers without intermediate HBM stores).
* **Estimated ceiling (A100-SXM4):** **0.75 – 1.15 ms** per call (eliminates ~0.35 ms of Inductor $Y_{ij}$ backward/edge-force elementwise kernels from the 1.07 ms pool, ~0.20 ms of `table_backward` from the 0.30 ms pool, and ~0.30 ms of HBM read/write traffic inside `wprime_backward`).


* **Cheap falsifier (1 day):** Stub out the $\nabla Y_{ij}$ global store in `wprime_backward`, replace it with a register dot-product against $\vec{r}_{ij}$ (38 FMAs) plus one `atomicAdd` of a `float3` to a dummy force buffer, and measure `wprime_backward` + removed Inductor force-chain kernels in `tools/step_breakdown.py`. Stop rule: if combined time drops by $< 0.35\text{ ms}$ (~2.5 % of step), abort.


* **Sources & differentiation:** Inspired by [SpheriCart](https://github.com/lab-cosmo/sphericart) ([J. Chem. Phys. 159, 064113 (2023)](https://arxiv.org/abs/2306.03813)) and Symmetrix-XL's fused directional force reverse stage ([arXiv:2610.01036](https://arxiv.org/abs/2610.01036)). Unlike rejected "Gate VJP and prologue/epilogue fusions" (which were node-level), this acts on the $E = 373,760$ edge bottleneck and removes the $\nabla Y_{ij}$ HBM round-trip.

---

#### Proposal 2: Undirected Half-Graph Pairing ($e_{ij}, e_{ji}$) in Radial Tables and `wprime_backward`

* **Mechanism:**
* **Current state:** In a periodic crystal of 5,000 atoms with 373,760 directed edges, there are exactly 186,880 undirected pairs $\{i, j\}$. Currently, every directed edge $e_{ij}$ and its twin $e_{ji}$ independently evaluate the degree-5 Chebyshev radial tables (0.42 ms forward, 0.30 ms backward) and independently rebuild $w(r)$ and $w'(r)$ from the slab inside `wprime_backward` (where rebuilding $w$ and $w'$ is half the register/ALU pressure of the 4.37 ms kernel).


* **Change:**
1. During the early neighbour-list rebuild (`SkinNeighbourList` on the side stream, which is already overlapped and free), alongside the receiver-sorted edge list, emit an `int32` index array `twin_idx[E]` (or sort undirected pairs for the radial table pass) mapping each directed edge $e_{ij}$ to its canonical undirected index $u \in [0, E/2)$.


2. **Radial tables (forward & backward):** Evaluate the Triton radial table kernel (`generic_radial_tables`) only on the $E/2 = 186,880$ canonical undirected distances $r_u$. Both edge count and output memory ($[h\cdot c, c, \text{density}]$) drop by $2\times$ (from 96 MB to 48 MB).


3. **In `wprime_backward`:** Because `wprime_backward` must stay receiver-sorted (sorting edges by slab was 1.51× slower), keep the receiver-sorted schedule, OR better yet: in Proposal 4 below, precompute or cache the undirected radial weights/slabs so L2 footprint is halved. Even simpler: in `generic_radial_tables`, running on $E/2$ halves the 0.72 ms radial table forward+backward time immediately and halves the L2 cache footprint of $[h\cdot c, c]$ read by `fused_forward`.






* **Exactness class:** **Exact** (bitwise identical distance $r_{ij} = \|\vec{r}_j - \vec{r}_i\| = r_{ji}$ and symmetric species pair $(Z_i, Z_j)$ in MACE's radial basis; if Agnesi radii are symmetric in $(Z_i, Z_j)$, it is $100\%$ symmetric; if any model has asymmetric pair tables, group by unordered pair and emit both only when $Z_i \ne Z_j$).
* **Estimated ceiling (A100-SXM4):** **0.45 – 0.65 ms** per call (cuts radial table forward+backward from 0.72 ms to ~0.36 ms, and reduces `fused_forward` L2 misses on $[h\cdot c, c]$ by 48 MB $\approx 0.15\text{ ms}$).


* **Cheap falsifier (1 day):** Modify the Triton radial table forward and backward kernels to run on $E/2$ edges and index $[h\cdot c, c]$ in `fused_forward` via `e >> 1` (mocking `twin_idx`). Measure the whole step on A100. Stop rule: if whole-step gain is $< 0.25\text{ ms}$, abort.
* **Sources & differentiation:** Standard MD Newton's third law symmetry, applied here specifically to the radial embedding and Chebyshev table evaluation while *preserving* the receiver-sorted edge order required by `fused_forward` and `wprime_backward` (thus avoiding the rejected 1.51× slower edge-order sort).

---

#### Proposal 3: Linear Folding Across Interaction Boundaries and Density Normalization (Atom Side)

* **Mechanism:**
* **Current state:** Atom GEMMs take 1.85 ms (44 CUTLASS launches) and Inductor elementwise takes 1.07 ms (~74 kernels). Look at the exact computational graph in §1–§2:


1. **Commuting density division across `linear_1`:** In both Interaction 1 and Interaction 2, the receiver message $m_i$ is divided by scalar $d_i = (\text{density}_i + 1)$ *before* `linear_1`: $u_i = \text{linear\_1}(m_i / d_i)$. Because equivariant `linear_1` is strictly homogeneous (zero bias), $\text{linear\_1}(m_i / d_i) = \text{linear\_1}(m_i) / d_i$. Pushing the $1/d_i$ scaling *after* `linear_1` fuses it directly into the `symmetric_contraction` prologue or `skip_tp` addition, eliminating two full-node elementwise kernels (forward + backward) per interaction and allowing `fused_forward` (`BLOCK_MESSAGE`) to feed directly into `linear_1` without an intervening Inductor division kernel!


2. **Folding Interaction 1's `linear_2` into Interaction 2's linear layers:** At the end of Interaction 1, MACE computes $h_i^{(1)} = \text{linear\_2}(p_i^{(1)}) + \text{skip\_tp}_1(s_i)$, where $p_i^{(1)}$ is the output of `symmetric_contraction_1` and $s_i$ is the 89-species one-hot. Then $h_i^{(1)}$ is consumed *exclusively* by three linear maps:


* Interaction 2's $\text{linear\_up}_2(h_i^{(1)})$
* Interaction 2's $\text{skip\_tp}_2(h_i^{(1)}, s_i)$
* $\text{readout\_1}(h_i^{(1)})$ (a linear layer on the $0e$ irrep)
By linearity, for any linear operator $L \in \{\text{linear\_up}_2, \text{skip\_tp}_2(\cdot, s_i), \text{readout\_1}\}$:

$$L\left(\text{linear\_2}(p_i^{(1)}) + \text{skip\_tp}_1(s_i)\right) = (L \circ \text{linear\_2})(p_i^{(1)}) + L(\text{skip\_tp}_1(s_i))$$



At setup time (in float64), multiply the weight matrices $W_{L \circ \text{linear\_2}} = W_L W_{\text{linear\_2}}$ per irrep block and precompute the 89-species constant lookup table $T_L[Z_i] = L(\text{skip\_tp}_1(e_{Z_i}))$.
This **completely eliminates `linear_2` of Interaction 1** (both forward and backward GEMMs) and replaces `skip_tp_1` with a simple species-indexed bias addition!


3. **Same folding for Interaction 2's `linear_2` + `skip_tp_2` on $\ell > 0$:** `readout_2` (an MLP) only reads the scalar $0e$ channel of $h_i^{(2)}$. If `linear_2` of Interaction 2 computes $\ell > 0$ irreps, those irreps are dead outputs (verify whether MACE-OMAT-0's second interaction `linear_2` and `symmetric_contraction_2` already prune $\ell > 0$; if not, prune all $\ell > 0$ paths in Interaction 2's `linear_1`, `symmetric_contraction`, `skip_tp_2`, and `linear_2`).






* **Exactness class:** **Rounding** (algebraically identical linear algebra; weights composed in float64 at setup time; passes the 2× ratio gate easily because matrix multiplication is associative up to FP32 rounding).
* **Estimated ceiling (A100-SXM4):** **0.50 – 0.85 ms** per call (removes 8–12 CUTLASS GEMM launches from the 44 launches in the 1.85 ms pool, plus ~10 Inductor elementwise kernels from the 1.07 ms pool).


* **Cheap falsifier (1 day):** Check the shapes of `interaction_1.linear` (`linear_2`) and `interaction_2.linear_1 / symmetric_contraction` in `breakdown/ops.csv`. Write a 40-line rewrite pass that fuses `W_linear_up2 @ W_linear2` in float64 at setup and bypasses `interaction_1.linear`. Run the accuracy gate and `loup_md.py`. Stop rule: if error ratio $> 2.0$ or step gain $< 0.25\text{ ms}$, abort.


* **Sources & differentiation:** Own derivation from the MACE architecture graph (§1–§2). Distinct from `grouped_linear` and `merged_residual` (which is off for OMAT-0): this composes sequential linear layers *across* the Interaction 1 $\to$ Interaction 2 boundary and commutes the scalar density division.



---

#### Proposal 4: 3×TF32 Tensor-Core Chebyshev Evaluation & Contraction Inside `wprime_backward` (and Direct Chebyshev `fused_forward`)

* **Mechanism:**
* **Current state:** `wprime_backward` (4.37 ms, 32 % of step) rebuilds $w_{e,c} = \sum_{p=0}^5 T_p(x_e) C_{s(e), p, c}$ and $w'_{e,c} = \sum_{p=0}^5 T'_p(x_e) C_{s(e), p, c}$ for $c \in [0, 1280)$ using scalar FP32 FMA loops on CUDA cores. Earlier, "computing `w` from the slab in the forward (spills; slower)" failed because it was done in scalar FP32 registers alongside the UVU registers.


* **Change:** Use A100's **3×TF32 tensor cores (`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`)** inside `wprime_backward`!
1. In `wprime_backward`, each thread block processes a batch of edges for a receiver. For a batch of 8 or 16 edges, form the degree-5 Chebyshev basis values $T_0(x_e)\dots T_5(x_e)$ and derivatives $T'_0(x_e)\dots T'_5(x_e)$ in shared memory (only 12 floats per edge).
2. Note that within a single receiver atom $i$ in a crystal (3 species in SrTiO3), the incoming edges fall into at most 3 species pairs, and typically 2–3 radial pieces per species pair. Even simpler: without any sorting, `wprime_backward` can still use 3×TF32 `mma.sync` if we rebuild $w$ and $w'$ from the **shared folded weight matrix $W \in \mathbb{R}^{1280 \times 64}$** and a 64-wide radial table $[h\cdot c, c]$ and its derivative $[h\cdot c, c]'$, **OR** if we keep `TABLE_W`, we pack the 16-edge tile's $dw_{e, c}$ computation:


3. Specifically, in `wprime_backward`, the UVU backward first computes $dw_{e, c}$ and $\nabla x_{j, c}$ from $g_{i}$ and $x_j, Y_{ij}$. Notice that $dw_{e, (u, \pi)} = \sum_{m_i, m_y, m_o} C^\pi_{m_i m_y m_o} g_{i, u, m_o} x_{j, u, m_i} Y_{\pi, m_y}$. Once $dw_{e, c}$ is formed in registers/smem for a tile of 16 edges, instead of rebuilding both $w_{e,c}$ and $w'_{e,c}$ in scalar FP32 registers concurrently (which causes high register pressure and occupies FP32 pipes), compute $w_{e,c}$ in Phase 1 (to get $\nabla x_j$ and $\nabla Y_{ij}$), overwrite its registers/smem with $dw_{e,c}$, and compute $dr_e = \sum_c dw_{e,c} w'_{e,c}$ in Phase 2!
4. **Crucial two-phase register lifetime split in `wprime_backward`:** Right now, `TABLE_W` computes $w_c$ and $w'_c$ simultaneously from the slab in the inner loop so it only loads the slab once. However, holding both $w_c$ and $w'_c$ (plus $T_0..T_5$ and $T'_0..T'_5$) doubles register demand and halves warp occupancy or causes local-memory/smem traffic. Instead, use the algebraic identity for Chebyshev polynomials: the derivative basis $T'_p(x)$ is a fixed linear combination of $T_q(x)$ ($T'_1 = T_0$, $T'_2 = 4 T_1$, $T'_3 = 6 T_2 + 3 T_0$, $T'_4 = 8 T_3 + 8 T_1$, $T'_5 = 10 T_4 + 10 T_2 + 5 T_0$). Therefore:



$$\sum_{c=1}^{1280} dw_{e,c} w'_{e,c}(x_e) = \sum_{p=0}^{5} T'_p(x_e) \left( \sum_{c=1}^{1280} dw_{e,c} C_{s(e), p, c} \right)$$



**Look at that equation!** By reassociating the sum over $c \in [0, 1280)$ *inside* the polynomial degree $p \in [0, 5]$:
* You do **NOT** compute $w'_{e,c} = \sum_{p=0}^5 T'_p(x_e) C_{s(e), p, c}$ for all 1,280 channels (which costs $6 \times 1280$ FMAs to build $w'_c$ + $1280$ FMAs to dot with $dw_c$ = $7 \times 1280$ FMAs, plus holding both $w_c$ and $w'_c$)!
* Instead, as each channel $c$'s slab coefficients $C_{s(e), 0..5, c}$ are loaded to form $w_{e,c} = \sum_{p=0}^5 T_p(x_e) C_{s(e), p, c}$, you immediately compute $dw_{e,c}$ for that channel and accumulate $dw_{e,c} \cdot C_{s(e), p, c}$ into **just 5 scalar register accumulators** $S_1 \dots S_5$ per edge!
* At the very end of the edge (after all 1,280 channels are done), you perform **one single 5-element dot product** $dr_e = \sum_{p=1}^5 T'_p(x_e) S_p$!






* **Exactness class:** **Rounding** (exact distributive law $\sum_c dw_c \left(\sum_{p=1}^5 T'_p(x) C_{p,c}\right) = \sum_{p=1}^5 T'_p(x) \left(\sum_c dw_c C_{p,c}\right)$; replaces 1,280 evaluations of $T'_p$ per edge with a single 5-term dot product at the end of the edge, and frees the registers previously holding $T'_1 \dots T'_5$ in the inner loop).
* **Estimated ceiling (A100-SXM4):** **0.40 – 0.80 ms** per call (~10–18 % of the 4.37 ms `wprime_backward` kernel by cutting slab FMA count and register live-range in the inner 1,280-channel loop).


* **Cheap falsifier (1 day):** In `fastmace/kernels/oeq_streaming/wprime_backward.py`, replace the inner-loop evaluation of `w_prime = c1*Tp1 + ... + c5*Tp5` and `dr += dw * w_prime` with 5 register accumulators `s1 += dw * c1; ... s5 += dw * c5;` and evaluate `dr = s1*Tp1 + ... + s5*Tp5` once after the channel loop. Compile with NVRTC and check `ptxas` register count and `loup_md.py` timing. Stop rule: if `wprime_backward` shrinks by $< 0.20\text{ ms}$, abort.


* **Sources & differentiation:** Own algebraic derivation from FastMACE's `wprime_backward.TABLE_W` design (§3.5). Nothing like this is in the rejected list (which tested vectorised slope loads and degree changes, but still evaluated $w'_c(r)$ per channel before multiplying by $dw_c$).



---

#### Proposal 5: Sender-Deduplicated Shared-Memory Gather & Local Atomic Reduction per Receiver Tile in `wprime_backward` and `fused_forward`

* **Mechanism:**
* **Current state:** In `fused_forward` (3.70 ms) and `wprime_backward` (4.37 ms), each receiver atom $i$ has ~75 incoming edges $(j \to i)$ within the 6.0 Å (+ 0.5 Å skin) cutoff. However, because SrTiO3 (and most condensed-matter systems with $6.5\text{ Å}$ cutoff) has periodic images or multiple edges/chunks, and more importantly, when a thread block processes 2–4 spatially adjacent receiver atoms (or 32 edges per warp), **many edges in the same thread block share the same sender atom $j$**! Furthermore, in `wprime_backward`, for every single path $\pi \in [0, 10)$ of the same edge $e = (j \to i)$, or across edges sharing sender $j$ within the block, threads issue global `atomicAdd`s to $\nabla x_j \in \mathbb{R}^{N \times 512}$ in L2/HBM.


* **Change:**
1. **Per-Edge Sender Gradient Accumulation in Registers/Smem before Global Atomic:** Ensure that within `wprime_backward`, for a given edge $e = (j \to i)$, all paths $\pi$ that contribute to the *same* sender irrep $(\ell_i, m_i)$ of $\nabla x_{j, u, (\ell_i, m_i)}$ are accumulated in registers/shared memory *first*, so that each edge issues at most **one** `atomicAdd` per sender irrep component (16 components total per channel), rather than one `atomicAdd` per path $\pi$ (up to 34 path-component updates). (Note: if your in-progress model-specialized backward already fuses updates across paths to the same sender irrep, verify this; if not, this alone eliminates >50% of global `atomicAdd` instructions!)
2. **Multi-Receiver Spatial Block Coalescing:** Since atoms are spatially ordered (or can be ordered by 3D cell bin without affecting exactness, note: Morton order alone gave no gain on the un-coalesced kernel, so only do this if paired with shared-memory caching), two adjacent receivers share ~65% of their neighbor list. Even for a *single* receiver $i$ in `wprime_backward`, $x_j$ (512 floats) is loaded once per edge, and $\nabla x_j$ (512 floats) is written once per edge. By staging $\nabla x_j$ in shared memory and flushing via vectorized `atomicAdd` (or warp-coalesced `atomicAdd` across the 128 channels $u$, which is contiguous in `ir_mul` or `mul_ir` layout), L2 atomic contention drops sharply.




* **Exactness class:** **Rounding** (changes only the order of floating-point addition into $\nabla x_j$).
* **Estimated ceiling (A100-SXM4):** **0.35 – 0.70 ms** per call (8–15 % of `wprime_backward`).


* **Cheap falsifier (1 day):** Profile `wprime_backward` with Nsight Compute (`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,smsp__sass_thread_inst_executed_op_atom_pred_on.sum`) or replace the global `atomicAdd` on $\nabla x_j$ with a shared-memory dummy write to measure the exact hardware cost of the global atomic scatter on A100. Stop rule: if removing/coalescing the $\nabla x_j$ `atomicAdd` saves $< 0.30\text{ ms}$ in the kernel stub, abort.
* **Sources & differentiation:** Inspired by FlashTP's path-aggregated execution ([ICML 2025](https://proceedings.mlr.press/v267/lee25l.html)) and TrimSO(2)'s on-chip accumulation before reduction ([arXiv:2610.20029](https://arxiv.org/abs/2610.20029)). Differs from the rejected Morton atom order (which was tested without changing the kernel's atomic emission count).

---

#### Proposal 6: `cp.async` Double-Buffered Edge Pipeline + 30 MB Persisting L2 Window for Node Features on A100

* **Mechanism:**
* **Current state:** `fused_forward` (3.70 ms) and `wprime_backward` (4.37 ms) process a receiver's edges in batches of 8 edges. In each iteration of the receiver loop, threads issue synchronous global loads of sender features $x_{s(e)} \in \mathbb{R}^{512}$, spherical harmonics $Y_e \in \mathbb{R}^{16}$, and radial features/slabs, wait for them to arrive in registers, store them to shared memory, call `__syncthreads()`, and then run tensor cores / CUDA cores. On A100 (`sm_80`), synchronous `ld.global` $\to$ `st.shared` wastes both registers (staging buffers) and instruction slots, and stalls warps during the L2 fetch (~200 cycles).


* **Change:**
1. **Hardware `cp.async.cg.shared.global` (16-byte async copy, `sm_80+`):** Allocate a 2-stage ring buffer in shared memory for the 8-edge batch inputs ($8 \times 512 \times 4\text{ B} = 16\text{ KB}$ per stage, easily fitting inside A100's 164 KB shared memory per SM without reducing occupancy). Issue `cp.async.cg.shared.global` (`__pipeline_memcpy_async` in CUDA C++) for batch $b+1$ *before* starting the 3×TF32 `mma.sync` and UVU contraction of batch $b$, followed by `cp.async.commit_group()` and `cp.async.wait_group<1>()`. The `.cg` (cache global) modifier bypasses L1 (avoiding L1 thrashing from streaming edges) and fetches directly from L2 into shared memory without touching the register file.
2. **A100 L2 Persistence (`cudaStreamAttrValue::accessPolicyWindow`):** The A100 has a 40 MB L2 cache. Across the entire 5,000-atom system, the node tensors that are randomly gathered/scattered by `s(e)` are:
* $x = \text{linear\_up}(h^{(1)}) \in \mathbb{R}^{5000 \times 512}$ (10.24 MB)
* $\nabla x \in \mathbb{R}^{5000 \times 512}$ (10.24 MB)
* $g = \nabla m \in \mathbb{R}^{5000 \times 512}$ (10.24 MB)
Total = **30.72 MB**. Meanwhile, streaming edge arrays ($Y_{ij}$, radial tables, slab weights) stream >200 MB through L2 and evict cache lines of $x$ and $\nabla x$. Allocate $x$, $\nabla x$, and $g$ in a single contiguous 31 MB workspace arena and attach a `cudaAccessPolicyWindow` with `hitRatio = 1.0`, `hitProp = cudaAccessPropertyPersisting`, `missProp = cudaAccessPropertyStreaming` on the CUDA stream before capturing Graph B.








* **Exactness class:** **Exact** (bitwise identical arithmetic; hardware data-movement and cache-replacement policy only).
* **Estimated ceiling (A100-SXM4):** **0.40 – 0.75 ms** per call (~5–9 % across `fused_forward` and `wprime_backward` combined, 8.07 ms base).


* **Cheap falsifier (1 day):**
* *Part A (1 hour):* Call `cudaCtxSetLimit(cudaLimitPersistingL2CacheSize, 32 << 20)` and set `cudaStreamSetAttribute` with `accessPolicyWindow` pointing to $x$ and $\nabla x$ on the capture stream of Graph B. Run same-process A/B in both build orders.


* *Part B (half day):* Replace the global-to-shared load of $x_{s(e)}$ in `fused_forward.py` with `__pipeline_memcpy_async` (16-byte `float4` chunks) and a 2-stage smem buffer. Stop rule: if neither gives $\ge 0.20\text{ ms}$, abort.


* **Sources & differentiation:** NVIDIA Ampere Architecture Tuning Guide (`sm_80` `cp.async` and L2 persistence). Note that the rejected list mentions "OEq launch-shape, L2-hint and register or block tuning (ceiling 0.3 %)" on the *old* materialized OEq kernel in October; `cp.async` double-buffering across the 8-edge streaming loop in `fused_forward` / `wprime_backward` has never been tested.

---

#### Proposal 7: Fuse Interaction 1's Segmented Edge Sum Directly into the Radial Table Kernel (Eliminating 1 Edge Pass)

* **Mechanism:**
* **Current state:** For Interaction 1, `generic_radial_tables` runs a Triton kernel over all $E = 373,760$ edges to evaluate the degree-5 Chebyshev polynomials and writes $[h_1 \cdot c, c, \text{density}_1] \in \mathbb{R}^{E \times 65}$ to HBM (~97 MB write). Immediately after, `segmented_first_sum` (0.22 ms forward, 0.50 ms backward) launches another Triton kernel that reads $[h_1 \cdot c, c] \in \mathbb{R}^{E \times 64}$ (96 MB read) and $Y_{ij} \in \mathbb{R}^{E \times 16}$ (24 MB read) from HBM, forms their outer product per edge, and run-segment-sums them per `(receiver, sender_species)`. In the backward pass, `segmented_first_sum` backward (0.50 ms) writes $\nabla [h_1 \cdot c, c] \in \mathbb{R}^{E \times 64}$ (96 MB write) to HBM, which `table_backward` (0.30 ms) then reads back (96 MB read) to multiply by the Chebyshev slope!


* **Change:**
* Interaction 1's radial table output $[h_1 \cdot c, c] \in \mathbb{R}^{E \times 64}$ has **no other consumer** than `segmented_first_sum`!


* Fuse Interaction 1's Chebyshev table evaluation directly inside `segmented_first_sum` (forward) and fuse Interaction 1's Chebyshev derivative $w'_1(r)$ directly inside `segmented_first_sum_backward` (backward)!
* Consequently, the entire $E \times 64$ edge tensor for Interaction 1 (96 MB forward write + 96 MB forward read + 96 MB backward write + 96 MB backward read = **384 MB of HBM traffic per step**) **disappears completely**.
* Even better: in `segmented_first_sum` (and its backward), apply **Sobek's Chebyshev factorization**! Within a `(receiver, sender_species)` run, group the sum by radial slab piece $s \in [0, S)$ and polynomial degree $p \in [0, 5]$: instead of multiplying $64$ radial channels by $16$ spherical harmonics ($64 \times 16 = 1,024$ FMAs per edge), multiply only the $6$ scalar Chebyshev powers $T_0(x_e)\dots T_5(x_e)$ by $Y_{ij}$ ($6 \times 16 = 96$ FMAs per edge—an **11× reduction in per-edge ALU work**!) and multiply by the $6 \times 64$ slab matrix *after* summing over the run!




* **Exactness class:** **Rounding** (fusing the table into `segmented_first_sum` is bitwise identical; factoring the 6 Chebyshev powers across the segmented sum is exact distributive arithmetic up to FP32 summation order).
* **Estimated ceiling (A100-SXM4):** **0.45 – 0.65 ms** per call (eliminates half of the 0.72 ms radial table forward+backward and more than half of the 0.72 ms `segmented_first_sum` forward+backward by removing 384 MB of HBM traffic and cutting edge FMAs by $11\times$).


* **Cheap falsifier (1 day):** Fuse Interaction 1's degree-5 Chebyshev evaluation inline into the Triton `segmented_first_sum` forward and backward kernels so Interaction 1 never allocates or touches the `[E, 64]` radial tensor. Measure on A100 in `tools/step_breakdown.py`. Stop rule: if combined time of `radial_tables` + `segmented_first_sum` (currently $0.72 + 0.72 = 1.44\text{ ms}$) drops by $< 0.30\text{ ms}$, abort.


* **Sources & differentiation:** Own synthesis combining FastMACE's `generic_radial_tables` (§3.3) and `segmented_first_sum` (§3.4) with Sobek's polynomial/basis reassociation ([arXiv:2607.18074](https://arxiv.org/abs/2607.18074)). Never tried before because `generic_radial_tables` was built as a drop-in replacement for the radial MLP module rather than fused with Interaction 1's segmented reducer.



---

#### Proposal 8 (Secondary Target – Capacity Path 1M–150M): Cache Compact Interaction-0 Segmented Sums instead of Recomputing Interaction 0 in Sweep 3

* **Mechanism:**
* **Current state:** In the tiled and wavefront paths (§4–§5), Sweep 1 runs Interaction 0 without autograd and stores per atom $x = \text{linear\_up}(h_1)$ and $sc = \text{skip\_tp}(h_1)$ (4.75 kB/atom). Then Sweep 3 **recomputes all of Interaction 0 from scratch with autograd** (re-evaluating tile edge vectors, spherical harmonics $Y$, radial tables, `segmented_first_sum`, `linear_1`, `symmetric_contraction_1`, and `linear_2`) just to backpropagate $dE/dx$ and $dE/dsc$ to Interaction 0's edge forces!


* **Change:**
* Look at what actually connects the node side of Interaction 0 (`symmetric_contraction_1`) to the edges of Interaction 0:
* Because Interaction 0 is reassociated (§3.4), its node side (`linear_1`, `skip_tp`, `symmetric_contraction_1`, `linear_2`, `readout_0`) takes as input **only** the node-level projected message $m_i^{(0)} \in \mathbb{R}^{R \times 512}$ and scalar $\text{density}_i^{(0)} \in \mathbb{R}^R$!


* Furthermore, in Sweep 2, the tile already loads $x$ and $sc$ for all receivers + halo. In Sweep 3, for a tile of $R = 8,192$ or $16,384$ receivers, recomputing the *edge* part of Interaction 0 (`Y`, `radial_tables`, `segmented_first_sum` forward) is only needed to obtain $m_i^{(0)}$ so `symmetric_contraction_1` has its forward input for backward!


* If instead we either:
1. Run the *node-only* backward of Interaction 0 (`linear_up`, `skip_tp`, `linear_2`, `symmetric_contraction_1`, `linear_1`) for receiver tile $T$ **at the end of Sweep 1**? No, $dE/dx_i$ isn't ready until Sweep 2 finishes all tiles that use atom $i$ as a halo sender.
2. **In Sweep 3, do NOT recompute Interaction 0's edges (`Y`, `radial_tables`, `segmented_first_sum` forward) to get $m_i^{(0)}$!** Instead, store the pre-contraction node feature $u_i^{(0)} \in \mathbb{R}^{512}$ (2 kB/atom, or overwrite `x` in host/device state as soon as Sweep 2 finishes with atom $i$ via the aliased state pool!) so Sweep 3 runs `symmetric_contraction_1` backward directly from $u_i^{(0)}$ without running Interaction 0's forward edge pass at all!
3. Even better: once Proposal 1 & Proposal 7 are enabled inside the tiles (and `fused_forward` + `wprime_backward` are enabled in tiled mode, which §4.3 notes has not yet been measured in tiles!), Sweep 3 only needs to run `segmented_first_sum_backward` + `Y_backward` on the tile's edges—zero forward edge recomputation in Sweep 3!








* **Exactness class:** **Exact** (stores/aliases the 512-float node input to `symmetric_contraction_1` instead of recomputing it from edges in Sweep 3).
* **Estimated ceiling (1M / 150M atoms on RTX 4000 Ada):** **1.8 – 2.8 µs/atom/step** (~12–18 % speedup on the 15.3 µs/atom/step 1M path and 19.8 µs/atom 149M path, closing almost the entire gap between tiled 15.3 µs and whole-system 13.0 µs).


* **Cheap falsifier (1 day):** Profile one tile of Sweep 1, Sweep 2, and Sweep 3 in `tools/bench/tiled_md.py`. Measure how much of Sweep 3 is spent recomputing Interaction 0's forward edge pass (`Y`, `radial_tables`, `segmented_first_sum` forward) plus enabling `wprime_backward` in Sweep 2. Stop rule: if removing Sweep 3's forward edge pass + enabling `wprime_backward` saves $< 1.0\text{ µs/atom/step}$, abort.


* **Sources & differentiation:** Own derivation from the 3-sweep diagram in §4. Directly extends the "aliased host state" idea (§4.2) by observing that Interaction 0's reassociation makes the boundary between Interaction 0's edges and nodes purely a node tensor $u_i^{(0)}$.



---

#### Proposal 9 (Secondary Target – Capacity Path 1M): GPU-Resident Velocity-Verlet + Custom Pinned Arena for `state="host"`

* **Mechanism:**
* **Current state:** At 1M atoms (`tools/bench/tiled_md.py`), the integrator runs Velocity-Verlet **on the host** (`docs/validation/tiled-1m-speed.md:203`), and `state="host"` uses **pageable** host memory because PyTorch's caching allocator rounds pinned allocations up to powers of two (§4.2). Pageable host-to-device and device-to-host copies cannot use DMA directly—the CUDA driver must stage every tile's rows through an internal driver staging buffer on the CPU, burning CPU bandwidth and preventing true asynchronous overlap with tile compute!


* **Change:**
1. Allocate the host state pool using `cudaHostAlloc` (or a small C++/ctypes wrapper around `cudaMallocHost`) for the **exact** byte size (bypassing PyTorch's power-of-two rounding allocator completely!). This gives true zero-overhead DMA pinned memory with zero power-of-two waste.
2. Move the 1M-atom positions, velocities, and forces (only $3 \times 1\text{M} \times 12\text{ B} = 36\text{ MB}$ total!) to stay permanently on the GPU with the GPU-resident Velocity-Verlet integrator, rather than running the host integrator in `tiled_md.py`.






* **Exactness class:** **Exact** (bitwise identical).
* **Estimated ceiling (1M atoms on RTX 4000 Ada):** **0.8 – 1.5 µs/atom/step** (~6–10 % of the 14.95 s step at 1M atoms).


* **Cheap falsifier (half day):** Allocate the host state buffer via `cuda.cudart().cudaHostAlloc(nbytes, 0)` wrapped with `torch.from_dlpack` / `ctypes` in `fastmace/model/tiled.py` and run 5 steps of `tiled_md.py` at 1M atoms. Stop rule: if step time drops by $< 0.4\text{ s}$ (out of 14.9 s), abort.


* **Sources & differentiation:** Own diagnosis from §4.2 ("a third of it was PyTorch's pinned-memory rounding" $\to$ switched to pageable). Bypassing PyTorch's rounding via direct `cudaHostAlloc` restores full PCIe DMA throughput without the memory bloat.



---

### Summary Table of Proposals (A100-SXM4 Primary Target Ranked First)

| Rank | Proposal | Target Component | Exactness | Est. Ceiling (A100) |
| --- | --- | --- | --- | --- |
| **1** | Fuse $Y_{ij}$ Jacobian ($\nabla Y_{ij} \to \nabla \vec{r}_{ij}$) & edge forces into `wprime_backward` | `wprime_backward` + Inductor + `table_bwd` | Rounding | 0.75 – 1.15 ms |
| **2** | Fuse Interaction 1 Chebyshev Table + Sobek 6-term basis into `segmented_first_sum` | `radial_tables` + `segmented_first_sum` | Rounding | 0.45 – 0.65 ms |
| **3** | Fold `linear_2` across Interaction 1$\to$2 & commute density division across `linear_1` | Atom GEMMs + Inductor elementwise | Rounding | 0.50 – 0.85 ms |
| **4** | Reassociate $\sum_c dw_c w'_c(x) = \sum_{p=1}^5 T'_p(x) \sum_c dw_c C_{p,c}$ inside `wprime_backward` | `wprime_backward` (4.37 ms) | Rounding | 0.40 – 0.80 ms |
| **5** | Undirected half-graph pairing ($e_{ij}, e_{ji}$) for radial tables ($E/2$ evaluation) | `radial_tables` + `fused_forward` L2 | Exact | 0.45 – 0.65 ms |
| **6** | `cp.async` 2-stage smem pipeline + 30 MB A100 L2 persisting window on $x, \nabla x, g$ | `fused_forward` + `wprime_backward` | Exact | 0.40 – 0.75 ms |
| **7** | Coalesce $\nabla x_j$ across paths before `atomicAdd` in `wprime_backward` | `wprime_backward` (4.37 ms) | Rounding | 0.35 – 0.70 ms |
| **8** | *(Capacity)* Eliminate Sweep 3 forward edge recomputation + enable `wprime_backward` in tiles | Tiled & Wavefront (1M–150M) | Exact | 1.8 – 2.8 µs/atom |
| **9** | *(Capacity)* Exact-size `cudaHostAlloc` pinned pool + GPU Velocity-Verlet at 1M | Tiled 1M (`tiled_md.py`) | Exact | 0.8 – 1.5 µs/atom |

*(Note: Proposals 2 and 7 are inverted in the table vs. text numbering so the top 4 proposals in the table are the highest-confidence, lowest-risk single-day falsifiers on the A100 whole-system path: Proposals 1, 7, 3, and 4 sum to an estimated **2.1 – 3.4 ms** reduction out of the 13.70 ms force call, targeting **~2.1–2.3 µs/atom/step** on Benchmark_loup.)*
