# Phase 7 coverage check — screening result (O15)

Screened 2026-10-05 by Claude against the coverage check declared 2026-09-27 in
`experiments/phase7_meta_analysis.md` (frame retrieved 2026-09-27, `candidates.json`, n = 571).
Every screened paper has a decision and a one-line reason in `screening.json`.
**Status: FINAL.** the first author spot-checked all 31 flagged calls on 2026-10-05 (check sheet: Claude Doc
"Coverage spot-check") and agreed with every one. No decision changed.

## Result (as declared: ambiguous abstracts screened IN)

- **Frame A: 8 of 30** most-cited in-scope papers are in the corpus.
- **Frame B: 0 of 10** most-cited in-scope 2024–2026 papers are in the corpus.
- In-scope after screening and de-duplication: 62 of 571.

Sensitivity (not the registered number): if every ambiguous / spot-check IN were screened OUT instead,
coverage would be 10/30 and 2/10. The registered rule (ambiguous → IN) gives the lower numbers,
as the protocol intends.

## How screening was done (disclose)

1. All 571 titles read. Papers clearly outside the domain (image compression, chemistry, traffic,
   medicine, wireless networks, databases …) were excluded **on title**, reason logged as
   "off-domain; excluded on title". The protocol says title + abstract; abstracts were in hand but
   not needed for these. ~480 papers.
2. The 88 papers in or near the domain were screened on title + abstract against the scope rule.
3. **Scope consistency with the corpus:** analytical models compared against learned baselines
   (uiCA, Crop) and benchmarks that compare models (BHive) were screened IN, because the corpus
   already applies this to Facile (A14), TPUGraphs (A1) and TenSet (A11).
4. Duplicates (arXiv vs venue versions of the same work) counted once: #388→#78, #375→#203,
   #389→#374, #412→#176, #498→#387, #561→#467.
5. Ties: Frame A's 30th paper has 5 citations, and #432 Sectum (also 5, not covered) falls just
   outside; Frame B's 9th–11th papers tie at 2 citations. Ties broken by the S2 file order.
   Neither tie changes the covered count.
6. Not matched under the strict rule: #269 "Learned TPU Cost Model for XLA Tensor Programs"
   (2019), the workshop precursor of A9 (different title). Counted as **uncovered**; spot-check.

## Things the frame shows about itself (disclose)

- Only 10 of the 20 corpus papers appear anywhere in the 571 candidates. The six fixed queries miss
  TPUGraphs (A1), TGraph (A2), ProGraML (A5), PerfSAGE (A7), NeuSight (A8), Polynormer (A12),
  A13, RLM (A16), Omniwise (A17) and PerfVec (A20).
- Frame A contains 5 HLS performance predictors (1 covered, A18) and 4 DL-model latency
  predictors (nn-Meter, CoDL, DNNPerf, NAR-Former; none covered), sub-areas the corpus samples
  lightly.
- Frame B's top 10 have 2–16 citations each; citation ranking is noisy this close to retrieval.

## Frame A — top 30 in-scope

| # | S2 rank | Citations | Year | Title | Covered (corpus) | Spot-check |
|---|---|---|---|---|---|---|
| 1 | 13 | 522 | 2018 | Learning to Optimize Tensor Programs | A10 |  |
| 2 | 35 | 223 | 2021 | nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices | no |  |
| 3 | 42 | 191 | 2018 | Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks | A4 |  |
| 4 | 73 | 103 | 2022 | CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices | no | yes |
| 5 | 78 | 97 | 2021 | A Learned Performance Model for Tensor Processing Units | A9 |  |
| 6 | 88 | 83 | 2022 | High-Level Synthesis Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing | A18 |  |
| 7 | 109 | 73 | 2021 | TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers | A11 |  |
| 8 | 114 | 70 | 2023 | IronMan-Pro: Multiobjective Design Space Exploration in HLS via Reinforcement Learning and Graph Neural Network-Based Modeling | no |  |
| 9 | 123 | 65 | 2022 | TLP: A Deep Learning-Based Cost Model for Tensor Program Tuning | A6 |  |
| 10 | 140 | 55 | 2022 | DietCode: Automatic Optimization for Dynamic Tensor Programs | no | yes |
| 11 | 148 | 53 | 2023 | Runtime Performance Prediction for Deep Learning Models with Graph Neural Network | no |  |
| 12 | 200 | 37 | 2023 | WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program | no |  |
| 13 | 203 | 37 | 2021 | uiCA: accurate throughput prediction of basic blocks on recent intel microarchitectures | no | yes |
| 14 | 207 | 36 | 2022 | Graph Neural Networks for High-Level Synthesis Design Space Exploration | no |  |
| 15 | 216 | 32 | 2019 | BHive: A Benchmark Suite and Measurement Framework for Validating x86-64 Basic Block Performance Models | no | yes |
| 16 | 236 | 27 | 2022 | GRANITE: A Graph Neural Network Model for Basic Block Throughput Estimation | A3 |  |
| 17 | 269 | 22 | 2019 | Learned TPU Cost Model for XLA Tensor Programs | no | yes |
| 18 | 293 | 19 | 2022 | NAR-Former: Neural Architecture Representation Learning Towards Holistic Attributes Prediction | no | yes |
| 19 | 319 | 16 | 2024 | LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers | no |  |
| 20 | 336 | 14 | 2019 | Absinthe: Learning an Analytical Performance Model to Fuse and Tile Stencil Codes in One Shot | no | yes |
| 21 | 353 | 13 | 2022 | NoCeption: A Fast PPA Prediction Framework for Network-on-Chips Using Graph Neural Network | no | yes |
| 22 | 355 | 12 | 2023 | CDMPP: A Device-Model Agnostic Framework for Latency Prediction of Tensor Programs | no |  |
| 23 | 356 | 12 | 2023 | Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs | no |  |
| 24 | 360 | 12 | 2023 | Facile: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction | A14 |  |
| 25 | 374 | 10 | 2022 | Moses: Efficient Exploitation of Cross-device Transferable Features for Tensor Program Optimization | no |  |
| 26 | 382 | 10 | 2023 | A Graph Neural Network Model for Fast and Accurate Quality of Result Estimation for High-Level Synthesis | no |  |
| 27 | 384 | 10 | 2024 | Learning to Compare Hardware Designs for High-Level Synthesis | no |  |
| 28 | 387 | 9 | 2024 | Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning | no | yes |
| 29 | 422 | 5 | 2023 | ATFormer: A Learned Performance Model with Transfer Learning Across Devices for Deep Learning Tensor Programs | no |  |
| 30 | 423 | 5 | 2024 | Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment | no | yes |

## Frame B — top 10 in-scope, 2024–2026

| # | S2 rank | Citations | Year | Title | Covered (corpus) | Spot-check |
|---|---|---|---|---|---|---|
| 1 | 319 | 16 | 2024 | LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers | no |  |
| 2 | 384 | 10 | 2024 | Learning to Compare Hardware Designs for High-Level Synthesis | no |  |
| 3 | 387 | 9 | 2024 | Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning | no | yes |
| 4 | 423 | 5 | 2024 | Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment | no | yes |
| 5 | 433 | 4 | 2024 | Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs | no | yes |
| 6 | 439 | 4 | 2024 | Fast and Accurate Context-Aware Basic Block Timing Prediction using Transformers | no |  |
| 7 | 443 | 3 | 2025 | NLTSP: A cost model for tensor program tuning using nested loop trees | no | yes |
| 8 | 444 | 3 | 2024 | UniCoMo: A Unified Learning-Based Cost Model for Tensorized Program Tuning | no |  |
| 9 | 451 | 2 | 2026 | UniCoX: A Unified Cost Model for Tensorized Program Tuning Across Ubiquitous Accelerators | no |  |
| 10 | 452 | 2 | 2026 | UniSparTa: A Unified Sparse Tensor Program Tuning Framework | no | yes |

## Spot-check list (31 papers, all decisions marked ambiguous or boundary)

| S2 rank | Title | Decision | Reason |
|---|---|---|---|
| 9 | Ansor : Generating High-Performance Tensor Programs for Deep Learning | OUT | claim is about search space + evolutionary search, not the cost model; matches the corpus's earlier full-read exclusion |
| 11 | Neo: A Learned Query Optimizer | OUT | database query optimizer; claim is plan quality vs PostgreSQL/commercial optimizers, not a program/kernel performance model |
| 48 | A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction | OUT | netlist timing (slack) prediction, a circuit-design metric, not program/kernel/model performance |
| 73 | CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices | IN | ambiguous: system paper whose latency predictor is a stated contribution vs prior predictors |
| 76 | Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction | OUT | predicts LLM output length for scheduling, not hardware performance of a program |
| 140 | DietCode: Automatic Optimization for Dynamic Tensor Programs | IN | ambiguous, no abstract in S2: dynamic-shape auto-scheduler with a micro-kernel cost model |
| 203 | uiCA: accurate throughput prediction of basic blocks on recent intel microarchitectures | IN | analytical/simulation throughput model compared against learned models (Ithemal); in scope by the same rule the corpus applied to Facile (A14) |
| 216 | BHive: A Benchmark Suite and Measurement Framework for Validating x86-64 Basic Block Performance Models | IN | ambiguous: benchmark that compares learned and analytical BB models (cf. dataset papers A1, A11 in corpus) |
| 266 | MASTER: Machine Learning-Based Cold Start Latency Prediction Framework in Serverless Edge Computing Environments for Industry 4.0 | OUT | serverless cold-start latency for resource management, not program/kernel performance |
| 269 | Learned TPU Cost Model for XLA Tensor Programs | IN | learned TPU cost model for XLA programs (2019 workshop precursor of A9; different title, so not matched under the rule) |
| 293 | NAR-Former: Neural Architecture Representation Learning Towards Holistic Attributes Prediction | IN | NN architecture representation predicting accuracy AND latency; architectural comparison |
| 336 | Absinthe: Learning an Analytical Performance Model to Fuse and Tile Stencil Codes in One Shot | IN | ambiguous: learned analytical performance model for stencil fusion/tiling |
| 353 | NoCeption: A Fast PPA Prediction Framework for Network-on-Chips Using Graph Neural Network | IN | ambiguous: GNN predicting NoC power/performance/area for applications, vs baseline predictor |
| 387 | Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning | IN | ambiguous: draft model + learned cost model, online cost-model adaptation |
| 423 | Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment | IN | ambiguous: TPO framework with a transferable cost model |
| 433 | Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs | IN | analytical cost model claimed comparable to learned cost models (A14 rule) |
| 438 | Learned Performance Model for SSD | IN | ambiguous, no abstract in S2: learned SSD performance model |
| 443 | NLTSP: A cost model for tensor program tuning using nested loop trees | IN | tensor-program cost model (no abstract in S2; title states cost model) |
| 447 | A Parametric Microarchitecture Model for Accurate Basic Block Throughput Prediction on Recent Intel CPUs | IN | ambiguous, no abstract in S2: parametric BB throughput model (likely uiCA lineage) |
| 452 | UniSparTa: A Unified Sparse Tensor Program Tuning Framework | IN | ambiguous: sparse-program tuning with a unified cost model |
| 457 | BB-ML: Basic Block Performance Prediction using Machine Learning Techniques | IN | ambiguous: ML extrapolation of basic-block counts for GPU performance prediction |
| 459 | Quantifying Throughput of Basic Blocks on ARM Microarchitectures by Static Code Analyzers: A Case Study on Kunpeng 920 | IN | ambiguous: comparison of BB throughput predictors on ARM |
| 469 | GAS: A scheduling primitive dependency analysis-based cost model for tensor program optimization | IN | tensor-program cost model (no abstract in S2) |
| 488 | BALTO: FAST TENSOR PROGRAM OPTIMIZATION WITH BIASED - DIVERSITY - BASED ACTIVE LEARNING | IN | ambiguous, no abstract in S2: active learning for tensor-program cost models |
| 501 | Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated Quantum Circuit Simulation | IN | learned rankers of GPU contraction-plan performance vs baselines |
| 522 | N-TORC: Native Tensor Optimizer for Real-Time Constraints | IN | ambiguous: data-driven performance/resource models for HLS layers |
| 526 | Adaptive Data Augmentation with Bayesian Optimization for Basic Block Throughput Prediction | IN | ambiguous, no abstract in S2: data augmentation for BB throughput models |
| 529 | MCAD: Beyond Basic-Block Throughput Estimation Through Differential, Instruction-Level Tracing | IN | ambiguous, no abstract in S2: BB throughput tooling |
| 543 | Energy-Efficient Cache Configuration Prediction Using Machine Learning on Basic Blocks | IN | ambiguous: ML predicting best cache configuration from BB features |
| 560 | Ph.D. Project: Floorplan Quality Prediction for HLS Design Exploration on Multi-Die FPGA | OUT | floorplan congestion prediction (physical design), not program performance |
| 568 | Hotspy: identifying performance hotspot with graph neural network based static analysis | IN | ambiguous, no abstract in S2: GNN static analysis for performance hotspots |
