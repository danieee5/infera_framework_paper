5
2
0
2

g
u
A
2
2

]
F
P
.
s
c
[

1
v
2
1
7
6
1
.
8
0
5
2
:
v
i
X
r
a

Systematic Characterization of LLM Quantization: A
Performance, Energy, and Quality Perspective

Tianyao Shi
Purdue University
West Lafayette, IN, USA
shi676@purdue.edu

Yi Ding
Purdue University
West Lafayette, IN, USA
yiding@purdue.edu

Abstract
Large language models (LLMs) have demonstrated remark-
able capabilities across diverse domains, but their heavy
resource demands make quantization—reducing precision to
lower-bit formats—critical for efficient serving. While many
quantization methods exist, a systematic understanding of
their performance, energy, and quality tradeoffs in realistic
serving conditions remains a gap.

In this work, we first develop a fully automated online
characterization framework qMeter, and then conduct an in-
depth characterization of 11 post-training LLM quantization
methods across 4 model sizes (7B–70B) and two GPU archi-
tectures (A100, H100). We evaluate quantization at the ap-
plication, workload, parallelism, and hardware levels under
online serving conditions. Our study reveals highly task- and
method-dependent tradeoffs, strong sensitivity to workload
characteristics, and complex interactions with parallelism
and GPU architecture. We further present three optimization
case studies illustrating deployment challenges in capacity
planning, energy-efficient scheduling, and multi-objective
tuning. To the best of our knowledge, this is one of the first
comprehensive application-, system-, and hardware-level
characterization of LLM quantization from a joint perfor-
mance, energy, and quality perspective.

1 Introduction
Large language models (LLMs) [2, 15, 19, 45] have demon-
strated remarkable capabilities across a wide range of tasks
such as software development [21, 23], scientific discov-
ery [72], and healthcare [50, 52]. Due to their large size, quan-
tization—reducing model precision from full precision to
lower-bit formats (e.g., 8-bit, 4-bit)—has been widely adopted
to shrink model size and improve serving efficiency [22].
In recent years, numerous quantization techniques have
emerged, including weight-only (only weights are quantized
into lower bits) [17, 18, 33, 47, 56] and activation (both ac-
tivation and weights are quantized) [3, 16, 62, 64, 68, 70]
quantization, with more recent approaches incorporating
KV cache compression for additional size reduction [34].

Despite the broad adoption of quantization and extensive
characterization on LLM serving [25, 30, 41, 48, 49, 57, 73],
systematic characterization of LLM quantization remains
underexplored. While recent studies have begun to address
this gap [14, 29, 31], they exhibit the following limitations:

1

• Lack of performance-energy-quality triad analysis.
Existing LLM quantization studies evaluate only partial
tradeoffs: some focus solely on a single metric such as
output quality [31] or performance (latency/throughput),
while others examine two metrics in combination, e.g.,
accuracy–performance [29] or energy–performance [14].
None evaluate performance, energy, and quality together,
leaving the full design tradeoff space unexplored and lim-
iting informed deployment decisions.

• Lack of online characterization. Existing LLM quanti-
zation characterization work focuses on offline profiling
using fixed prompts under controlled conditions rather
than online setting, which fails to capture how quantized
models behave in dynamic, real-world serving scenarios
with varying load and input distributions.

• Lack of system-level optimization. Existing LLM quan-
tization characterization studies are mostly from the ma-
chine learning community, overlooking advanced system-
level optimization techniques such as parallelism (for scal-
ing model serving across multiple devices) and KV cache
compression (for further model size reduction). This makes
it unclear how quantization interacts with distributed ex-
ecution and memory management.

• Limited system design insights. Existing LLM quanti-
zation studies largely focus on application-level metrics
such as latency, throughput, or quality. They rarely trans-
late these findings into implications for architectural and
system design, deployment strategies, and resource allo-
cation in production environments, where performance,
energy, quality, and service-level objectives (SLOs) must
be explicitly managed.
To overcome these limitations, we conduct a comprehen-
sive, online systematic characterization of LLM quantization,
evaluating performance, energy efficiency, and output qual-
ity jointly across diverse model sizes, quantization methods,
and workloads. To facilitate accurate and repeatable online
profiling, we develop qMeter (§3), a fully automated online
characterization framework that detects saturation points,
sweeps large configuration spaces (quantization schemes,
parallelism levels, workloads), and integrates with inference
engines and benchmarking suites. qMeter ensures measure-
ment robustness by continuously monitoring serving engine
health and restarting failed instances, enabling consistent
profiling across diverse load conditions.

Using qMeter, we conduct an in-depth characterization of
11 LLM post-training quantization methods (summarized in
Table 1) on the TensorRT-LLM v0.19.0 inference engine [42].
Our experiments cover the Llama-2 model family [60] at four
sizes (7B, 13B, 34B, and 70B), evaluated on NVIDIA H100
and A100 GPUs. Following prior work [57, 73], we include
chatbot, code generation, and summarization tasks across
diverse benchmarks. The study examines quantization at
four levels: application, workload, parallelism, and hardware.
Our results reveal the following key insights:
• Task- and method-dependent tradeoffs (§4). No sin-
gle quantization method dominates across latency, energy
efficiency, and quality. Larger quantized models can some-
times outperform smaller full-precision ones, but improve-
ments are highly task- and method-dependent.

• Workload Sensitivity (§5). Quantization benefits vary
with input/output length and load intensity: short out-
puts can hurt TTFT, long inputs can increase TPOT, and
optimal configurations shift with request rate.

• Parallelism interaction (§6). Activation quantization
scales with moderate tensor parallelism (TP), even reduc-
ing GPU needs. Weight-only with KV compression incurs
compounded latency and energy overhead. This indicates
that quantization and parallelism should be co-optimized.
• Hardware dependence (§7). GPU architecture affects
quantization tradeoffs: H100 improves latency and scala-
bility, while A100 offers higher energy efficiency at mod-
erate loads; memory and compute capacity jointly shape
saturation behavior.
Building on the characterization results, we present three
optimization case studies for quantization-enabled LLM serv-
ing clusters, motivated by real-world deployment needs and
guided by model–system–hardware co-design (§8). The first
examines saturation point prediction for scheduling and
capacity planning. The second explores energy-optimal con-
figuration by examining how different parallelization strate-
gies impact efficiency under varying traffic loads. The third
highlights the energy–quality imbalance in single-objective
optimization, where prioritizing one metric can degrade the
other. Together, these case studies demonstrate the need for
holistic approaches that balance performance, energy, and
quality across models, systems, and hardware.

To the best of our knowledge, this is one of the first com-
prehensive application-, system-, and hardware-level charac-
terization of LLM quantization from a joint performance, en-
ergy, and quality perspective. We hope to lay the foundation
for future research on principled model, system, hardware
co-design for quantization-enabled LLM serving at scale.

2 Background and Related Work
In this section, we first review the LLM quantization tech-
niques analyzed in this paper, followed by background on

2

Tianyao Shi and Yi Ding

Table 1. Quantization methods studied in this paper.

Category

Method

Name

Weight
Only

Per-Channel INT8 [26] W8A16-INT
W4A16-INT
AWQ [33]

Activation

SmoothQuant [68]
Per-Tensor FP8[36]
AWQ

W8A8-INT
W8A8-FP
W4A8

KV Cache
Compression

QServe [34]
-
-
-

W4A8KV4
W8A16KV8-INT, W4A16KV8-INT,
W8A8KV8-INT, W8A8KV8-FP
W4A8KV8

recent LLM characterization studies to highlight the gap in
quantization-focused characterization.

2.1 LLM Quantization

Quantization techniques can be broadly categorized into
quantization-aware training (QAT), which incorporates quan-
tization into the training process via backpropagation to
update quantized weights [11, 40], and post-training quanti-
zation (PTQ), which is typically training-free [38, 39]. Since
QAT is difficult to scale to large models such as LLMs, PTQ
is the dominant approach for LLM quantization and is the
also focus of this study.

There are three major PTQ techniques. The first is weight-
only quantization, where only weights are quantized into
low-bit integers [17, 18, 33, 47, 56]. For this category, we
study per-channel INT8 [26] and AWQ [33] in this paper.
The second is activation quantization, where both activation
and weights are quantized to INT8 [3, 16, 62, 64, 68, 70].
Activation quantization generally outperforms weight-only
quantization as it reduces memory requirements while also
accelerating token generation in memory-bound workloads.
For this category, we study SmoothQuant [68], per-tensor
FP8 [36], and W4A8-AWQ methods in this paper. The third
is KV cache compression, which quantizes KV cache along
with activation and weights [34]. For this category, we study
QServe and 8-bit KV cache compression variant of all the
weight-only and activation method mentioned above. All
quantization methods considered in this work are widely
adopted in real-world production and are supported by high-
performance inference engines such as TensorRT-LLM.

2.2 LLM Characterization

Recent profiling studies have characterized LLM serving to
understand the complex interplay between model architec-
tures, inference workloads, and system-level characteristics.
Such work provides empirical insights into performance,
scalability, latency, and energy efficiency to inform optimiza-
tion opportunities. On the machine learning side, many stud-
ies benchmark LLMs across various domain-specific tasks
to measure accuracy and speed [9, 28, 37, 51, 63, 65, 69].

Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

LLM quantization profiling, it generalizes to profiling any
LLM workload. We build qMeter in response to real needs
and challenges encountered in our studies:
• Large number of experiment settings and diverse
profiling configurations. Our study spans a wide range
of model sizes, quantization schemes, parallelism strate-
gies, and LLM serving applications. Collecting perfor-
mance, energy, and quality metrics across all these com-
binations efficiently and consistently is a complex and
time-consuming process. An automated approach is criti-
cal—not only to systematically generate and execute these
configurations, but also to identify the saturation point for
each setup, i.e., the maximum load the LLM serving system
can sustain before performance plateaus or degrades.
• Mismatch between existing benchmarking tools and
our evaluation goals. Many existing LLM quantization
profiling tools focus on offline or throughput-centric bench-
marking [14, 29, 31], whereas our evaluation targets online
serving scenarios that measure latency, energy consump-
tion, and output quality simultaneously. This mismatch
means that existing tools often cannot capture the fine-
grained measurements needed for our analysis.

• Fragile serving engines under extreme workloads.
Under high load or stress-testing conditions, existing serv-
ing engines can become unstable, crash, or produce in-
consistent results. This instability makes it challenging
to run large-scale, high-intensity profiling without robust
monitoring and recovery mechanisms.
Figure 1 shows qMeter and how it interacts with the GPU
cluster and the database. The user specifies the mix of model,
quantization method, parallelism strategy, dataset or quality
benchmark in the profiling plan via a JSON file or command
line arguments. Then the Profile Coordinator guides the pro-
filing process by first calling the Engine Handler to create the
corresponding LLM serving instance. For latency and energy
profile, it conducts a request rate (QPS, req/s) range search to
find the saturation point of each system configuration (model
size, quantization method, GPU type, parallelism, dataset).
Specifically, it calls the Benchmarker to run short-period
bursting benchmarks that interact with the LLM serving in-
stances via HTTP get/post and check if the predefined SLO
is violated. The highest SLO-attaining QPS is determined
by binary search and deemed as the saturation point. Based
on the saturation point, the Profile Coordinator generates
the profiling configurations spanning the QPS range and
invokes the Benchmarker in the main profiling loop to col-
lect latency and energy measurements. For quality profile,
qMeter bypasses the QPS range search stage and directly
generates configuration files for benchmarking suites like
lmeval and opencompass, which act as the Benchmarker.

To ensure reliability and avoid redundant runs, the En-
gine Handler continuously monitors the health of serving
instances. Upon detecting that the engine is not responding,
it will send signals to pause profiling, inspect logs for errors,

Figure 1. The flowchart of qMeter (§3.1). We build qMeter
that interacts with the GPU cluster and database to run
controlled tests and profile performance, energy, and quality
metrics.

On the systems side, characterization efforts focus on la-
tency [25, 30, 73], especially detailed latency breakdowns for
TTPT during the prefill phase and TPOT during decoding.
Beyond latency, energy efficiency and the associated trade-
offs are also commonly analyzed [48, 49, 57]. More recently,
there has been growing interest in assessing environmental
impacts of LLM serving such as carbon emissions [41, 67]
and water consumption [66].

Despite extensive characterization of general LLM serving,
only a limited number of studies have examined LLM quanti-
zation. Existing quantization characterization has evaluated
only partial tradeoffs: some focus exclusively on a single
metric such as accuracy [31] or performance (e.g., latency or
throughput), while others investigate pairs of metrics like ac-
curacy and performance [29] or energy and performance [14].
However, none have comprehensively considered all three
metrics together. However, none have jointly examined all
three metrics. In addition, existing studies rely on offline
profiling, which cannot capture quantization behavior under
dynamic, real-world online serving conditions. These gaps
limit our understanding of the full design tradeoffs in LLM
quantization and also motivate this work.

3 Methodology
In this section, we first introduce our newly developed tool,
qMeter, which we use for the characterization study in this
work, and then describe the testbed configurations for all
experiments presented in the paper.

3.1 qMeter

In this work, we introduce qMeter, a tool for automated and
comprehensive LLM profiling. While qMeter is designed for

3

Profiling Plan       qMeterConfigSynthesisProfiling LoopAnalyzerDatabaseEngineHandlerBenchmarkerProfile CoordinatorClusterCreate, Health MonitoringPOSTFailure RecoverLoadWeightsWriteProfileWriteLogDetectErrorReadProfileVisualizationQPS RangeSearch*GETNVMLPipeGPUMonitorTable 2. Datasets for different tasks.

Dataset / Domain

#Input

#Output Mid-Range QPS

ShareGPT (Chat) [55]
HumanEval (Code) [5]
NewsQA (Sum.) [61]

331
193
806

231
67
200

5 req/s
21 req/s
4 req/s

and verify if the engine is corrupted. Once confirmed, it will
kill the old serving processes and restart the new ones to
resume profiling once the new instance is operational.

3.2 Testbed
Configurations. We evaluate both full precision (FP16) and
11 quantization methods (summarized in Table 1) spanning
weight-only, activation, and KV cache quantization schemes.
To broaden the characterization beyond prior LLM quanti-
zation studies [14, 29, 31], we also vary parallelism strate-
gies. We focus on tensor parallelism (TP), which partitions
LLM layers across multiple GPUs to execute them in par-
allel. We prioritize TP due to its superior throughput and
latency compared to pipeline parallelism in single-node de-
ployments [57]. Since most open-source LLMs can fit within
the memory budget of 8 GPUs on a single server, we restrict
experiments to TP configurations over 1, 2, 4, and 8 GPUs,
denoted TP1, TP2, TP4, and TP8, respectively.
Models. We experiment with open-source Llama-2 [60] mod-
els in four sizes: 7B, 13B, 70B, and CodeLlama-34B.
Hardware. We primary experiment with NVIDIA H100 GPU
architecture. For hardware level analysis, we experiment
with both H100 and A100 to capture performance and energy
trade-offs across hardware generations.
Inference engine. We use TensorRT-LLM v0.19.0 [42] due
to its native support of diverse quantization methods, integra-
tion with CUDA kernels optimized for low-latency inference,
and broad adoption in both research and production.
Workloads. Following prior work [57, 73], our workload
suite includes chatbot, code generation, and summarization
tasks, which evaluate on ShareGPT [55], HumanEval [5], and
NewsQA [61] datasets respectively. The details of datasets
are summarized in Table 2. These datasets are used for eval-
uating performance, energy efficiency, and output quality.
The only exception is that for evaluating output quality in
the chatbot task, ShareGPT is not appropriate since it con-
sists of open-ended multi-turn conversations without stan-
dardized groundtruth responses, making objective quality
metrics infeasible. Instead, we use the following benchmarks
to assess chatbot quality, organized into three categories: (1)
chat-S for commonsense knowledge and general QA (Hel-
laSwag [71], ARC-C [12], Winogrande [54], TriviaQA [27]);
(2) chat-R for complex instruction following and reasoning
(BigBench-Hard [58], MMLU [24]); and (3) chat-M for math
problems (GSM8K [13], GPQA Diamond [53]).

Tianyao Shi and Yi Ding

Figure 2. Latency and energy efficiency comparison for
quantized CodeLlama-34B models (§4.1).

Metrics. We quantify performance using Time to First Token
(TTFT) and Time per Output Token (TPOT), both measured
at various percentile latencies (e.g., P50, P90). Energy effi-
ciency is evaluated via energy per token (Joule/token) using
GPU power telemetry. Output quality is assessed via accu-
racy for chatbot tasks, pass@1 for code generation [5], and
ROUGE scores for summarization [32].

4 Application Level Analysis
In this section, we characterize application-level behaviors
by first examining the impact of different quantization meth-
ods on latency, energy efficiency, and output quality inde-
pendently under a fixed model–hardware configuration. We
then analyze tradeoffs among these three metrics.

We quantize Llama-2 (7B, 13B, 70B) and CodeLlama-34B
using TensorRT-LLM’s quantization utilities. Models up to
34B are deployed on a single H100 GPU, while the 70B model
requires TP4, the minimum number of H100s for FP16 infer-
ence. We replay a mid-range request stream defined as 50% of
the saturation throughput of the 13B INT8 model on an H100.
This workload represents a consistent, moderate load: high
enough to enable continuous batching, but below saturation
to keep quantization effects observable. Task characteristics,
including average input/output lengths and mid-range re-
quest rates, are summarized in Table 2. We measure TTFT
and TPOT over a 2-minute trace, energy per token from
H100 telemetry, and output quality (0–100). Due to page
limits, we report P90 tail latency results for the 34B model
as a representative case, since it is the largest model that fits
on a single H100 in FP16 and exhibits trends consistent with
other model sizes and average latencies.

4.1 Latency, Energy, and Quality Results

(Tail) Latency. Figure 2 (top two rows) reports P90 TTFT
and TPOT for each task across quantized 34B models. We

4

ShareGPTHumanEvalNewsQA50100200400P90 TTFT (ms)ShareGPTHumanEvalNewsQA02040P90 TPOT (ms)ShareGPTHumanEvalNewsQA0.00.2Energy/Token (J)FP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV4Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

Figure 3. Quality score percentage change w.r.t. FP16 across quantization methods for 13B, 34B, and 70B models (§4.1).

make the following observations. (1) Weight-only and acti-
vation quantization methods generally reduce tail latency,
with up to a 70% decrease. Among them, W4A8 consistently
delivers the best TTFT and TPOT across all datasets. (2) In
general, greater model compression yields better latency:
e.g., W4A16-INT outperforms W8A16-INT, and activation
methods (W8A8-INT, W8A8-FP, W4A8) achieve lower TTFT
than weight-only methods on ShareGPT and NewsQA, where
input sequences are sufficiently long. In contrast, on Hu-
manEval (short contexts), W8A16-INT and W4A16-INT sur-
pass activation methods, suggesting that context length in-
fluences latency behavior (see more in §5.1). (3) Although
prior work [14, 29, 31] reports substantial throughput gains
in offline batch mode, our results show that at moderate
load, quantization does not always reduce latency in online
serving. (4) KV cache compression is surprisingly harmful,
reducing latency benefits across all methods. Even for W4A8,
applying 8-bit or 4-bit KV cache compression significantly
increases latency, and W4A8KV4 shows negligible TTFT
gains on ShareGPT and NewsQA. For weight-only methods,
KV compression can push latency close to FP16 levels.

Energy. Figure 2 (bottom row) shows energy per token
across methods. We make the following observations. (1)
Quantization improves energy efficiency by up to 30%, but 8-
bit activation methods yield minimal energy savings despite
lower latency. (2) 4-bit quantization delivers larger reduc-
tions, with W4A8KV4 achieving the best energy efficiency
via combining 4-bit weight and 8-bit activation compression.
(3) Exceptions arise for HumanEval again, where W8A16-
INT and W4A8 consume more energy than FP16 despite
latency gains. (4) KV cache compression generally increases
energy consumption, except for W4A8KV4.

Quality. Figure 3 shows normalized benchmark accuracy
score changes relative to FP16 for 13B, 34B, and 70B models
(negative indicates quality loss). Key observations:

5

• Efficiency–quality tradeoffs. Quantization can cause
substantial quality degradation, up to a 92% drop in Hu-
manEval pass rate for the 13B model. Quality-preserving
methods include W8A16-INT, W8A8-FP, and W8A8KV8-
FP, though these still lose 5–10% accuracy on some tasks.
W4A16-INT and W4A8 incur up to 22% loss on larger
models, while W8A8-INT suffers the most severe degra-
dation. KV cache compression typically worsens quality
for weight-only methods and also for W4A8 on 34B.
• Task difficulty sensitivity. Quality losses are smaller
on simpler QA and summarization tasks, but severe for
challenging reasoning tasks like coding and math.

• Model size sensitivity. Smaller models are more vulner-
able towards quantization-induced quality degradation.
This can be confirmed by visually checking the area of red
and yellow cells for each model size. For 13B, quality loss
is widespread and severe; for 34B, degradation is moderate
for chat-S and chat-R and severe in chat-M and coding;
for 70B, chat-S remains nearly lossless, with at most 22%
loss for coding and reasoning tasks except W8A8-INT.

4.2 Tradeoff Analysis

Latency vs. Energy. We examine the latency–energy trade-
offs across 4 model sizes and 12 methods on each dataset
in Figure 4. The x-axes show P90 TTFT and TPOT, the y-axis
shows energy per token at mid-range load, with color and
marker size denoting model size and marker type denoting
quantization method. Configurations closer to the bottom-
left indicate better latency–energy tradeoffs. We want to
find if any quantized larger model lies towards left below a
smaller FP16 model.

Comparing 34B FP16 (yellow circle) with nearby quan-
tized 70B models, we find two cases where 70B W4A8 (red
point-up pentagon) and W4A8KV4 (red hexagon) have simi-
lar latency to 34B FP16 with reduced energy per token on
chatbot and code generation tasks. Similarly, between 13B

 ) 3   :  $    , 1 7 :  $   . 9   , 1 7 :  $    , 1 7 :  $   . 9   , 1 7 :  $   , 1 7 :  $  . 9   , 1 7 :  $   ) 3 :  $  . 9   ) 3 :  $  :  $   . 9  :  $  . 9  ) 3   :  $    , 1 7 :  $   . 9   , 1 7 :  $    , 1 7 :  $   . 9   , 1 7 :  $   , 1 7 :  $  . 9   , 1 7 :  $   ) 3 :  $  . 9   ) 3 :  $  :  $   . 9  :  $  . 9  ) 3   :  $    , 1 7 :  $   . 9   , 1 7 :  $    , 1 7 :  $   . 9   , 1 7 :  $   , 1 7 :  $  . 9   , 1 7 :  $   ) 3 :  $  . 9   ) 3 :  $  :  $   . 9  :  $  . 9  + H O O D 6 Z D J $ 5 &  & : L Q R J U D Q G H 7 U L Y L D 4 $ 0 0 / 8 % % + * 3 4 $  ' L D P R Q G * 6 0  . + X P D Q H Y D O ; V X P                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               / O D P D      % & R G H / O D P D    % / O D P D      % F K D W  6 F K D W  5 F K D W  0Tianyao Shi and Yi Ding

Figure 4. Latency vs. energy tradeoffs across model sizes and quantization methods at mid-range load (§4.2).

Figure 5. Quality vs. latency and energy tradeoffs across model sizes and quantization methods at mid-range load (§4.2).

FP16 (blue circle) and quantized 34B models, we find three
cases where 34B W4A8KV4 (yellow hexagon) and W8A8-INT
(yellow left triangle) on code completion as well as W4A8
(yellow point-up pentagon) on summarization achieve both
lower latency and energy compared to 13B FP16.

Latency and Energy vs. Quality. We further examine the
latency-quality and energy-quality tradeoffs in Figure 5. The
x-axes show P90 TTFT and energy per token at mid-range
load, and the y-axis shows quality scores (chatbot quality
is the arithmetic mean of chat-R benchmarks). Higher and
more leftward points indicate better tradeoffs. Due to page
limits, we present only TTFT results, having checked that
TPOT trends are consistent. We want to find if any quantized
larger model lies above and left of a smaller FP16 model.

Here are the example cases we find. (1) For chatbot work-
loads, we identify two cross-size tradeoff cases: 34B W4A8
(yellow pentagon) achieves higher output quality and lower
latency with only negligible energy overhead compared to
13B FP16 (blue circle); and 70B W4A8 (red pentagon) im-
proves both energy efficiency and quality, with only a mar-
ginal latency increase. (2) For code generation, 34B W4A16-
INT, W8A8-INT, and W4A8KV4 all outperform 13B FP16
by providing higher quality and faster latency without ad-
ditional energy penalties. In fact, CodeLlama-34B shows
superior coding performance even over 70B models, though
its summarization accuracy falls slightly below 13B. (3) For
summarization, however, no quantized larger model is able to

6

improve quality without incurring penalties in latency or en-
ergy efficiency. Overall, these results highlight that the ben-
efits of quantization are highly task- and model-dependent,
with non-trivial cross-size tradeoffs that can be exploited in
scheduling and capacity planning.
Finding #1: (a) No single quantization method dominates
across all three metrics, which are varied by task, model
size, and precision level. (b) Quantized larger models can
match or surpass smaller FP16 models in certain tradeoff
spaces, offering either better energy with comparable la-
tency or better quality/latency without major energy penal-
ties. (c) Task specialization matters; e.g., CodeLlama-34B
significantly outperforms even 70B models on coding tasks
but may underperform on summarization. (d) Tradeoff im-
provements are task- and method-dependent, as gains in
quality often come at the expense of latency or energy, with
few configurations improving all metrics at once.
Recommendation #1: (a) Quantization methods should
aim to improve all three metrics together rather than op-
timizing for one or two metrics, as current approaches
rarely achieve simultaneous improvements. (b) Model size
and precision should be co-optimized rather than indepen-
dently. (c) There is a need for adaptive scheduling and
model selection strategies that tailor quantization choices
to task-specific latency, energy, and quality requirements.
(d) Automated tool is needed to navigate the tradeoff spaces
and select configurations that best meet application SLOs.

1001000P90 TTFT (ms)0.10.20.30.51Energy per Token (J)Chatbot10100P90 TPOT (ms)0.10.20.30.51Chatbot100P90 TTFT (ms)0.10.20.30.51Code Generation10100P90 TPOT (ms)0.10.20.30.51Code Generation100P90 TTFT (ms)0.10.20.30.51Summarization10100P90 TPOT (ms)0.10.20.30.51Summarization34B vs. 70B:Similar latency,better energyefficiency forquantized 70B13B vs. 34B:Better latency &energy efficiencyfor quantized 34BFP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV47B13B34B70B1001000P90 TTFT (ms)20304050Quality ScoreChatbotTTFT vs Quality0.10.20.30.51Energy per Token (J)20304050ChatbotEnergy vs Quality1001000P90 TTFT (ms)01020304050 Code GenerationTTFT vs Quality0.10.20.30.51Energy per Token (J)01020304050 Code GenerationEnergy vs Quality1001000P90 TTFT (ms)14161820SummarizationTTFT vs Quality0.10.20.30.51Energy per Token (J)14161820SummarizationEnergy vs Quality34B vs. 70B:       Similar latency, better quality and energy efficiencyfor quantized 70B         13B vs. 34B:Similar energy efficiency,better quality & latencyfor quantized 70B13B vs. 34B:Better quality &latency, equal or betterenergy efficiencyfor quantized 34BFP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV47B13B34B70BSystematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

Figure 6. Input/output length influence (§5.1). Latency
and energy metrics w.r.t. different input/output lengths
across quantized 34B models at QPS=5 req/s.

Figure 7. Load influence (§5.2). Latency and energy met-
rics of quantized 34B models under variable QPS. The hori-
zontal black dash line represents latency SLO.

5 Workload Level Analysis
Since our study targets real-world online serving, this section
focuses on online workload characteristics and analysis. We
will first examine the impact of input/output lengths, and
then the effect of load intensity.

5.1 Input/Output Lengths

Figure 7 reports P90 TTFT, P90 TPOT, and energy per to-
ken for quantized 34B models across varying input/output
lengths. All experiments are run at a fixed QPS of 5 req/s, cho-
sen based on the saturation point of the longest input/output
lengths on a single H100. The x-axes represent QPS (req/s),
and the y-axes measure latency (ms) or energy per token
(J). For visual clarity, the QPS values of different quantiza-
tion methods are slightly offset, though they correspond to
the same request rate. To control input/output lengths, we
follow the same methodology from prior work [57]. We set
short/medium/long inputs to 128/(256, 512)/1024 tokens, and
short/medium/long outputs to 64/(128, 256)/512 tokens. This
yields 16 input-output combinations in total.

We make the following observations. (1) TTFT can de-
grade for some weight-only methods when output length is
short. This can be observed in input–output pairs such as
(128,64), (256,64/128), (512,64/128), and (1024,64). This indi-
cates that the overhead of weight dequantization and prefill
computation dominates the total runtime and is not well
amortized when the decoding phase is short, making TTFT
less efficient. (2) TPOT can degrade when input length is long.
This can be observed in W4A8KV4, W8A16-INT, W4A16-
INT, and W8A16KV8-INT with 512/1024-token inputs. This
reflects increased inefficiency from quantization in handling
larger activation matrices. (3) Quantization decreases en-
ergy efficiency for short and medium input–output pairs due

Figure 8. Energy efficiency evolution (§5.2). Energy ef-
ficiency rank, best energy, best saving w.r.t. FP16, and best
saving ratio of quantized 34B models under variable QPS.

to dequantization overheads dominating the computation.
However, when input/output lengths grow, quantization im-
proves energy efficiency.

5.2 Load Intensity

Effect of quantization on saturation point. Figure 7
shows P90 TTFT, P90 TPOT, and energy per token for differ-
ent quantized 34B models across different request rates (QPS)
on three tasks. The x-axes represent QPS (req/s), and the
y-axes measure latency (ms) or energy per token (J). Across
all three tasks, quantization generally pushes the saturation
point higher than FP16, meaning that it can allow the sys-
tem to sustain higher QPS before latency spikes. In both
ShareGPT and NewsQA, nearly all quantized methods ex-
tend the saturation point, with more aggressive quantization
yielding the largest improvements. However, HumanEval
shows smaller gains; activation methods still help, but KV
cache compression can offset the benefits.

7

6412825651250100200P90 TTFT (ms)Input Length: 12864128256512102103Input Length: 25664128256512102103Input Length: 51264128256512103104105Input Length: 1024641282565122030P90 TPOT (ms)6412825651220304064128256512501006412825651210020030064128256512Output Length0.250.500.75Energy per Token (J)64128256512Output Length0.20.464128256512Output Length0.150.200.2564128256512Output Length0.10.2FP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV410203050010001500P90 TTFT (ms)ShareGPT25507510050100150200HumanEval51015100030005000NewsQA102030100200300P90 TPOT (ms)ShareGPT255075100100200HumanEval510150200400NewsQA102030QPS0.10.20.5Energy per Token (J)ShareGPT255075100QPS0.050.10.20.5HumanEval51015QPS0.10.20.5NewsQA4680.20.310200.10.23450.150.20FP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV42101618222426123456789101112Energy EfficiencyRank (1 = Best)ShareGPT2101618222426QPS Transition Points135Energy perToken (J)e−12550535558085100HumanEval535558085100QPS Transition Points123e−11520256710131415NewsQA16710131415QPS Transition Points24e−12550Energy Saving %w.r.t. FP16FP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FPW8A8KV8-FPW4A8W4A8KV8W4A8KV4Best Energy (J)Energy Saving (J)Energy Saving Ratio (%)SLO ViolationTianyao Shi and Yi Ding

Impact of traffic load on latency and energy. Traffic
load (request rate) strongly influences how quantization af-
fects latency and energy efficiency as shown in Figure 7. (1)
At low QPS, quantization provides modest latency benefits
but already delivers noticeable energy savings, especially for
lower-bit quantization. (2) As QPS increases, quantization
yields greater latency reductions. Energy savings, shown in
the bottom panel of Figure 8 and discussed in detail below,
grow only when FP16 saturates early; at low QPS, both the
absolute and relative energy savings decrease. (3) KV cache
compression can reduce or even negate latency gains under
high loads, particularly in HumanEval, where short contexts
make KV cache compression overhead more visible.

Shifts in energy-optimal and SLO-compliant config-
urations. The combination of energy efficiency and SLO
compliance changes with request rate. To illustrate this, we
rank the energy-per-token values of all quantization meth-
ods and track how these rankings change with increasing
traffic load, as shown in the top panel of Figure 8. Methods
that violate latency SLO constraints are shaded in gray. For
clarity, we only display QPS transition points where the op-
timal configuration changes, along with the corresponding
best energy efficiency, absolute energy savings relative to
FP16, and percentage savings at each transition shown in
the bottom panel. We make the following observations. (1)
The most energy-efficient SLO-compliant configuration de-
pends on request rate: at low to mid QPS, heavily compressed
models such as W4A8 or W4A8KV8 are often optimal, deliv-
ering 20–50% energy savings while meeting latency SLOs.
(2) Near saturation, lighter compression methods like W8A8-
INT or W8A8KV8-FP become preferable, as they maintain
SLO compliance with slightly reduced energy savings. This
shift underscores the importance of dynamically selecting
quantization configurations based on load.
Finding #2: The benefits of quantization are highly sen-
sitive to input/output length and load intensity. (a) Short
outputs make dequantization and prefill overheads more
noticeable, which can hurt TTFT, while long inputs make
activation handling less efficient, leading to higher TPOT.
(b) Load intensity shifts the balance of energy efficiency
and latency, showing that no single configuration remains
optimal across QPS.
Recommendation #2: (a) Workload-aware and load-
adaptive scheduling are needed to dynamically select quan-
tization methods based on prompt length, expected output
length, and real-time system load. (b) Quantization meth-
ods should be developed to minimize prefill and dequanti-
zation overheads for short outputs and to better optimize
activation handling for long inputs, ensuring more robust
performance across diverse workloads.

Figure 9. Parallelism influence (§6). Latency and energy
trends of quantized Llama-2-70B models compared with
FP16 across parallelism on the chatbot task. Trend group ①:
weight-only; ②: weight-only with KV cache compression; ③:
8-bit weight and 8-bit activation with KV cache compression
including W8A8-FP, W8A8KV8-FP, W8A8-INT, W8A8KV8-
INT, and W4A8; ④:4-bit weight and 8-bit activation with KV
cache compression including W4A8KV8 and W4A8KV4.

6 Parallelism
To serve LLMs of immense size, parallelism is needed to dis-
tribute computation across multiple GPUs. In this section,
we examine how quantization interacts with parallelism. We
focus on tensor parallelism (TP) in this study, running a 70B
model with TP1/2/4/8 on H100 GPUs. TP is chosen because
it performs better than pipeline parallelism in single-node
settings, avoids pipeline bubbles and inter-stage synchro-
nization, and is sufficient to host the model sizes we study
on a 8 GPU node. We also analyze data parallelism in §8.2.
We identify four groups in the latency and energy trends
of quantization methods under parallelism, and choose one
representative per group to illustrate the key findings.

Figure 9 reports P90 TTFT, P90 TPOT, and energy per
token across selected quantization methods from each group.
The x-axis represents QPS and the y-axis shows the metric
values, with colors indicating quantization type and markers
denoting the TP level. We make the following observations.
(1) For weight-only quantization (①), adding more GPUs
(TP2→TP8) consistently reduces both TTFT and TPOT, but
the relative latency and energy improvements from quan-
tization remain fractional. This suggests that while paral-
lelism alleviates compute bottlenecks, memory access and
dequantization overheads still limit efficiency gains. (2) For
weight-only with KV cache compression (②), scaling paral-
lelism amplifies the overhead: latency often increases and en-
ergy efficiency degrades relative to FP16, especially at higher

8

02040102103P90 TTFT (ms)①02040102103②02040102103③02040102103④02040100200P90 TPOT (ms)02040100200020401002000204010020002040QPS100Energy per Token (J)02040QPS10002040QPS10−110002040QPS100FP16 - TP4FP16 - TP8W8A16-INT - TP2W8A16-INT - TP4W8A16-INT - TP8W4A16KV8-INT - TP1W4A16KV8-INT - TP2W4A16KV8-INT - TP4W4A16KV8-INT - TP8W8A8KV8-FP - TP2W8A8KV8-FP - TP4W8A8KV8-FP - TP8W4A8KV8 - TP1W4A8KV8 - TP2W4A8KV8 - TP4W4A8KV8 - TP8Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

Table 3. Hardware specifications of GPUs used in this study.

GPU

Memory
Capacity TFLOPS TOPS Bandwidth

INT8 Memory

FP16

TDP

A100 [43]
H100 [44]

40 GB
80 GB

624
1979

1248
3958

1.6 TB/s
3.35 TB/s

400 W
700 W

TP levels. This indicates that compression overhead inter-
acts poorly with inter-GPU communication and sharded KV
cache. (3) For 8-bit weight and 8-bit activation with KV cache
compression (③), moderate parallelism amplifies the benefits.
These methods consistently reduce both latency and energy
across TP levels. Notably, W8A8KV8-FP at TP4 achieves
latency comparable to FP16 at TP8 under SLO, demonstrat-
ing that quantization combined with moderate parallelism
can replace heavier FP16 scaling. (4) For 4-bit weight and
8-bit activation with KV cache compression (④), results are
mixed. At low to mid QPS, latency often worsens compared
to FP16 due to additional compression/decompression over-
head. However, near FP16’s saturation, these methods show
relative latency gains and consistently better energy effi-
ciency, suggesting that aggressive quantization becomes
more competitive under heavy load. (5) For 4-, 8-bit weight
and 8-bit activation with KV cache compression (③,④), high
TP observes a diminishing return compared to moderate TP.
Doubling GPUs from TP4 to TP8 only increases the satura-
tion point by 20-35% and leads to worse energy efficiency.
Finding #3: (a) Quantization interacts strongly with ten-
sor parallelism: activation quantization scales well under
moderate TP, but weight-only with KV compression incurs
compounded latency and energy overheads. (b) Activation
quantization (e.g., FP8-KV) at TP4 can match FP16 at TP8,
achieving similar latency with fewer GPUs. (c) KV cache
compression can undermine latency at higher TP due to
additional communication and synchronization overheads.
Recommendation #3: (a) The effectiveness of quanti-
zation depends on both precision and parallel execution.
Parallelism-aware quantization is needed to align dequan-
tization and KV handling with sharding layouts. (b) Quan-
tization can act as a substitute for aggressive TP scaling.
We can use quantization as a scaling lever in scheduling to
reduce hardware demand and energy consumption.

7 Hardware Level Analysis
Beyond application- and system-level analysis, we also ex-
amine how hardware platforms influence the behavior of
different quantization methods. Specifically, we compare
NVIDIA A100 and H100 GPUs, with their specifications in
Table 3. To minimize communication and parallelism effects,
we run the Llama-2 13B model on a single GPU in each case,
as one device is sufficient to host the model.

9

Figure 10. Hardware influence (§7). Latency and energy
metrics of quantized 13B models on H100 and A100 GPUs
within A100’s saturation range. Methods marked by * are
only available on H100 for FP8 compute compatibility.

Figure 10 reports P90 TTFT, P90 TPOT, and energy per
token across quantization methods on both GPUs. The x-axis
represents QPS and the y-axis shows the metric values, with
colors indicating GPU type and markers denoting quantiza-
tion method. For visual clarity, the QPS values of different
quantization methods are slightly offset, though they corre-
spond to the same request rate. We make the following obser-
vations. (1) Overall, A100 exhibits higher latency than H100,
even with quantization applied. Quantized models on A100
still saturate earlier than FP16 on H100, reflecting differences
in raw compute capability. (2) Memory capacity also plays
a role: the effective VRAM available for KV cache on A100
running a W4A16-INT 13B model (≈30 GB) is about 2.5×
that of running a FP16 13B (≈12 GB), but still significantly
smaller than running a FP16 13B on H100 (≈53 GB), limiting
the maximum parallelism that can be achieved. (3) A100
shows better energy efficiency at low to mid QPS. Compared
to FP16 on H100, the same quantization method delivers
on average 9.6–35.6% greater energy savings on A100. This
difference stems from the higher TDP of H100, which boosts
performance but also increases power consumption.

1234567102103P90 TTFT (ms)Chatbot369121518212450100150200Code Generation123451021035×103Summarization12345671050100200P90 TPOT (ms)369121518212410501002001234510501002001234567QPS0.10.20.51Energy per Token (J)3691215182124QPS0.050.10.20.512345QPS0.10.20.4FP16W8A16-INTW8A16KV8-INTW4A16-INTW4A16KV8-INTW8A8-INTW8A8KV8-INTW8A8-FP*W8A8KV8-FP*W4A8*W4A8KV8*W4A8KV4A100 40GH100A100 SLO ViolationH100 SLO ViolationFinding #4: (a) Hardware architecture shapes the tradeoffs
of quantization: newer GPUs like H100 improve latency and
scalability, but older GPUs like A100 may deliver higher
energy efficiency under moderate loads. (b) Quantization
helps alleviate the memory capacity bottleneck, especially
on older hardware. But beyond that, but system perfor-
mance and saturation also depend on compute capability.
Recommendation #4: Hardware-aware scheduling is
needed to select quantization configurations based on both
workload characteristics and GPU architecture.

8 Optimizations for LLM Quantization

Serving Systems

Building on the characterization results, this section presents
three optimization case studies for quantization-enabled
LLM serving clusters, motivated by real-world deployment
needs and guided by model–system–hardware co-design.

8.1 Saturation Point Prediction

To optimize the energy efficiency of a quantization-enabled
LLM serving cluster, we want to identify the saturation point,
i.e., the maximum QPS each instance can sustain while meet-
ing latency SLOs. This point determines usable throughput
for scheduling and capacity planning. Exhaustive profiling
provides accurate saturation points at high cost. We therefore
ask: can machine learning models predict the saturation point
for unseen configurations, reducing the need for profiling?

Data sources and setup. We leverage three sources of

benchmarking data from prior profiling efforts:
1. H100 benchmarking data covering 7B/13B/34B/70B mod-
els, ShareGPT/HumanEval/NewsQA workloads, tensor
parallelism (TP) levels 1/2/4/8, and all 12 methods (277
data points).

2. A100 benchmarking data with the same model sizes and
datasets as H100, but restricted to TP levels 1/2/4 and 8
non-FP8 quantization methods (172 data points).

3. Synthetic random-input workloads for the 34B TP1 model
on H100, where we systematically control input and out-
put length (192 data points).

Each record is represented as a feature tuple: (model size,
quantization method, GPU type, input length, output length),
and the prediction target is the measured saturation point.
We train XGBoost [6] regressors under different train-test
split schemes to understand predictability.

Experiments and findings. We experiments under the

following three scenarios to understand predictability.
1. General learnability (random splits). We begin with
random data splits to assess the fundamental learnability
of the prediction task. When using all available data, the
model achieves a mean absolute percentage error (MAPE)
of 31.1% with an 8:2 train-test split, and 33.0% with a 9:1

Tianyao Shi and Yi Ding

split. Excluding the HumanEval dataset—whose unusu-
ally long outputs produce saturation points exceeding 100
req/s and heavily skew the target distribution—reduces er-
ror to 18.9%. A similar improvement is observed when re-
stricting evaluation to H100 data alone (18.8%), indicating
greater consistency within a single hardware domain. Fur-
ther narrowing to random-input workloads yields even
better stability, with MAPE decreasing to 16.2%. The low-
est error, 14.9%, is achieved when combining both re-
strictions: H100-only data while excluding HumanEval.
Overall, these results suggest that heterogeneity across
datasets and hardware introduces distribution shift, but
within a constrained domain, saturation point prediction
is feasible with moderate accuracy.

2. Unseen request lengths. We evaluate extrapolation by
excluding specific input or output lengths during train-
ing. Excluding input lengths yields poor generalization
(MAPE 34.9–85.3%), and excluding output lengths pro-
duces similar errors (16.8–69.5%). Short outputs (64/128
tokens) are predicted more accurately due to overlap with
natural workloads (ShareGPT/HumanEval). Incorporat-
ing higher TP (TP>1) slightly increases error by 1–10%,
indicating parallelism confounds length-dependent satu-
ration behavior. The lowest error (14.9% MAPE) occurs
when restricting to H100 data and excluding HumanEval.
Overall, interpolation across lengths is feasible, but ex-
trapolation to unseen lengths remains unreliable.

3. Cross-GPU transfer. We examine whether knowledge
can transfer across GPU types. Using H100 data with
partial A100 data (HumanEval, NewsQA) to train and
testing on A100 ShareGPT results in a high error (MAPE
73.1%). Alternative configurations, such as swapping the
train–test datasets (e.g., training on A100 ShareGPT and
testing on HumanEval/NewsQA) or augmenting the train-
ing set with random-input workloads, do not improve ac-
curacy and can yield errors exceeding 100%. These results
reveal a significant gap between H100 and A100 GPUs.
Saturation behavior is hardware-specific; models trained
on one GPU type is hard to generalize to another.

Takeaway. Learning-based prediction of saturation points
is feasible within homogeneous data regimes (same GPU
type, similar request lengths, consistent datasets), where er-
rors can be reduced below 15%. However, predictions fail
across domains—especially between different GPUs or un-
seen request lengths—revealing strong domain gaps. This
underscores that profiling remains indispensable for robust
system design, particularly when optimizing energy effi-
ciency across heterogeneous hardware or diverse workloads.

8.2 Energy-Optimal Configuration with Data

Parallelism

While we did not analyze data parallelism in §6, we examine
the role of data parallelism in improving energy efficiency

10

Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

here by evaluating system configurations across different
model sizes and datasets. For each configuration defined
by a pair of quantization method and tensor parallelism
degree, we sweep the QPS range from 0 up to the saturation
point of a single instance under tensor parallelism. When
the target QPS exceeds this saturation point, we provision
⌈ 𝑥
⌉ instances to meet the demand, distributing load evenly
𝑥𝑠
across instances. This load balancing is crucial because the
energy–QPS curve is convex and monotonically decreasing
(e.g., Figure 7), allowing us to maximize energy efficiency. By
repeating this process across all QPS values, we can identify
the energy-optimal configuration for each workload.

Findings. We summarize the findings as follows:

• Data and tensor parallelism tradeoffs. For the profiled
model sizes and datasets, data parallelism combined with
lower tensor parallelism often outperforms a single high
tensor parallelism H100 instance in terms of energy ef-
ficiency. This indicates that, on H100, tensor parallelism
does not necessarily scale quantization benefits.

• Cross-GPU comparison (A100 vs. H100). On A100,
however, combining data and tensor parallelism can occa-
sionally approach or even surpass the energy efficiency
of an H100 configuration. For example, on the 34B Hu-
manEval workload, an A100 TP2 configuration rivals the
energy efficiency of H100 TP1 under moderate load.

Takeaway. Determining energy-optimal system configu-
rations is inherently non-trivial. Simple rules—such as pre-
ferring tensor parallelism over data parallelism or always
using newer GPUs—can lead to suboptimal outcomes.

8.3 Energy–Quality Tradeoffs

To better understand the energy–quality tradeoffs in deploy-
ing quantized models under high traffic, we synthesize a
cluster-level request trace and compare three simple strate-
gies for selecting system configurations (model size, quanti-
zation method, GPU type, tensor parallelism (TP), data paral-
lelism (DP)). The trace includes four request types: chat-S,
chat-R, code generation, and summarization, each with
specific latency and quality SLOs (TTFT/TPOT/Quality Score:
1s/0.2s/55, 3s/0.2s/50, 0.15s/0.2s/35, 5s/0.2s/16). Each request
type is assumed to have a dedicated resource pool. We gen-
erate request lengths by sampling the Azure LLM trace [49],
aligning them with benchmarking datasets and scaling the
QPS to over 100 req/s to simulate cluster-level traffic, with
a chat-S: chat-R ratio of 4:1. Based on profiling, we map
request types to fixed model sizes: chat-S to 13B, chat-R to
70B, code generation to 34B, summarization to 13B.

We evaluate three strategies for system configurations:
1. FP16-Only: Use FP16 models, calculate TP and DP levels
to meet SLOs, pick the energy-optimal (GPU, DP, TP).
2. Quality-First: Select the quantization method with high-
est output quality, then compute TP and DP to meet SLOs,
finally pick energy-optimal (GPU, DP, TP).

11

Table 4. Energy-quality tradeoffs for three strategies.

Strategy

Avg # GPUs

SLO attainment (%)

Energy/Token

FP16-Only
Quality-First
Energy-First

45
53
24

100
100
38.6

0.128 J
0.137 J
0.062 J

3. Energy-First: Select the (quantization method, GPU, TP)
with lowest energy per token at its saturation point, then
compute the DP to meet SLOs.

For each timestamp in the synthetic trace, we apply the three
strategies and record the total GPUs used, SLO attainment,
and cluster-level energy per token.

Findings. Table 4 shows the results. Prioritizing quality
increases GPU allocation and energy consumption due to
dequantization overhead, while prioritizing energy can de-
grade output quality. For example, Energy-First chooses the
13B W8A8KV8-INT model for chat-S requests, violating the
quality SLO: the average score drops from 60 (FP16) to 50.48,
a 16% decline. Since chat-S represents 50–60% of the traffic,
this illustrates the critical need to balance energy efficiency
with quality preservation in real-world LLM serving.

Takeaway. Energy–quality tradeoffs exist. Achieving the
best energy-quality tradeoffs while meeting SLOs requires
adaptive configuration strategies that dynamically balance
model precision, parallelism, and resource allocation.

9 Limitation Discussion
Despite our best efforts to analyze state-of-the-art LLM quan-
tization methods, our study cannot encompass all possible
approaches. The main limitations are: (1) We focus on 8-bit
and 4-bit quantization, and do not cover lower-bit formats
such as 2-bit [4], as well as emerging mixed-precision [8, 59]
and adaptive schemes [46]. These are not included because
they are unsupported by our chosen inference engine for
study, and their adoption in practice remains uncertain. (2)
Our evaluation uses open-source Llama models with the
TensorRT-LLM inference engine, which may limit applica-
bility to other model families or serving frameworks. (3)
Our hardware scope is restricted to NVIDIA H100 and A100
GPUs, without considering other NVIDIA GPU generations
or alternative AI accelerators. (4) We do not consider the
impact of other inference optimizations such as chunked
prefill [1] and disaggregated serving [49, 57, 73], which are
orthogonal to quantization but may interact in interesting
ways (e.g., through communication overhead). (5) Our bench-
marking tasks do not consider very long-context workloads
such as repository-level code completion (≥8K tokens) [10]
that are increasingly common. (6) For parallelism, we do
not consider pipeline [35], 3D [7], or other parallelism tech-
niques. (7) On the model side, we do not consider the GGUF
format [20], which supports lower-bit quantization and is

popular for personal or edge devices, nor do we evaluate
newer models that adopt native mixed-FP8 training rather
than relying solely on post-training quantization.

10 Conclusion
This paper presents an online profiling tool and a joint per-
formance, energy, and quality characterization of LLM quan-
tization, laying the groundwork for model, system, hardware
co-design for quantization-enabled LLM serving at scale.

References
[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun
Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ram-
jee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference
with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems
Design and Implementation (OSDI 24).

[2] Meta AI. [n. d.]. LLaMA: Open and Efficient Foundation Language
Models. https://ai.facebook.com/blog/large-language-model-llama.
[3] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian Croci, Bo Li,
Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and
James Hensman. 2024. Quarot: Outlier-free 4-bit inference in rotated
llms. Advances in Neural Information Processing Systems 37 (2024).
[4] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M
De Sa. 2023. Quip: 2-bit quantization of large language models with
guarantees. Advances in Neural Information Processing Systems 36
(2023), 4396–4429.

[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke
Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz
Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Fe-
lipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios
Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss,
Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin,
Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, An-
drew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa,
Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie
Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCan-
dlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large
Language Models Trained on Code. arXiv:2107.03374 [cs.LG]

[6] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree
boosting system. In Proceedings of the 22nd acm sigkdd international
conference on knowledge discovery and data mining. 785–794.

[7] Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, and Jingren Zhou.
2023. Ee-llm: Large-scale training and inference of early-exit large
language models with 3d parallelism. arXiv preprint arXiv:2312.04916
(2023).

[8] Yidong Chen, Chen Zhang, Rongchao Dong, Haoyuan Zhang, Yonghua
Zhang, Zhonghua Lu, and Jidong Zhai. 2024. Mixq: Taming dynamic
outliers in mixed-precision quantization by online prediction. In SC24:
International Conference for High Performance Computing, Networking,
Storage and Analysis. IEEE, 1–15.

[9] Myra Cheng, Tiziano Piccardi, and Diyi Yang. 2023. CoMPosT: Charac-
terizing and Evaluating Caricature in LLM Simulations. In Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Pro-
cessing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association
for Computational Linguistics, Singapore, 10853–10875.

[10] Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-Guided Retrieval
Augmentation for Repository-Level Code Completion. In Proceed-
ings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Yashar Mehdad, Baobao Li, and

12

Tianyao Shi and Yi Ding

Chengqing Zong (Eds.). Association for Computational Linguistics,
Bangkok, Thailand, 7957–7977.

[11] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen
Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. 2018.
Pact: Parameterized clipping activation for quantized neural networks.
arXiv preprint arXiv:1805.06085 (2018).

[12] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabhar-
wal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have
Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.
arXiv preprint arXiv:1803.05457 (2018).

[13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton,
Christopher Nayak, John Knight, William Chen, and John Schulman.
2021. Training verifiers to solve math word problems. arXiv preprint
arXiv:2110.14168 (2021).

[14] Pepijn de Reus, Ana Oprescu, and Jelle Zuidema. 2024. An exploration
of the effect of quantisation on energy consumption and inference
time of StarCoder2. arXiv preprint arXiv:2411.12758 (2024).

[15] DeepSeek. [n. d.]. DeepSeek. https://chat.deepseek.com/.
[16] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.
2022. Gpt3. int8 (): 8-bit matrix multiplication for transformers at
scale. Advances in Neural Information Processing Systems (NeurIPS)
(2022).

[17] Tim Dettmers and Luke Zettlemoyer. 2023. The case for 4-bit precision:
k-bit inference scaling laws. In International Conference on Machine
Learning. PMLR.

[18] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022.
Gptq: Accurate post-training quantization for generative pre-trained
transformers. arXiv preprint arXiv:2210.17323 (2022).
[19] Gemini. [n. d.]. Gemini. https://gemini.google.com/app.
[20] Georgi Gerganov. 2023. GGUF Specification. https://github.com/ggml-

org/ggml/blob/master/docs/gguf.md

[21] GitHub. [n. d.]. copilot. https://github.com/features/copilot.
[22] Ruihao Gong, Yifu Ding, Zining Wang, Chengtao Lv, Xingyu Zheng,
Jinyang Du, Yang Yong, Shiqiao Gu, Haotong Qin, Jinyang Guo, Dahua
Lin, Michele Magno, and Xianglong Liu. 2025. A survey of low-bit large
language models: Basics, systems, and algorithms. Neural Networks
(2025), 107856.

[23] Junda He, Christoph Treude, and David Lo. 2025. LLM-Based Multi-
Agent Systems for Software Engineering: Literature Review, Vision,
and the Road Ahead. ACM Transactions on Software Engineering and
Methodology 34, 5 (2025), 1–30.

[24] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas
Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive
Multitask Language Understanding. In International Conference on
Learning Representations.

[25] Qinghao Hu, Zhisheng Ye, Zerui Wang, Guoteng Wang, Meng Zhang,
Qiaoling Chen, Peng Sun, Dahua Lin, Xiaolin Wang, Yingwei Luo,
Yonggang Wen, and Tianwei Zhang. 2024. Characterization of large
language model development in the datacenter. In 21st USENIX Sym-
posium on Networked Systems Design and Implementation (NSDI 24).
709–729.

[26] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew
Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko.
2018. Quantization and training of neural networks for efficient
integer-arithmetic-only inference. In Proceedings of the IEEE conference
on computer vision and pattern recognition. 2704–2713.

[27] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer.
2017. Triviaqa: A large scale distantly supervised challenge dataset
for reading comprehension. arXiv preprint arXiv:1705.03551 (2017).

[28] Ching-Yun Ko, Pin-Yu Chen, Payel Das, Yung-Sung Chuang, and Luca
Daniel. 2024. On Robustness-Accuracy Characterization of Language
Models using Synthetic Datasets. In First Conference on Language
Modeling.

Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective

[29] Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, and
Dan Alistarh. 2024. " Give Me BF16 or Give Me Death"? Accuracy-
Performance Trade-Offs in LLM Quantization.
arXiv preprint
arXiv:2411.02355 (2024).

[30] Malgorzata Lazuka, Andreea Anghel, and Thomas Parnell. 2024. Llm-
pilot: Characterize and optimize performance of your llm inference
services. In SC24: International Conference for High Performance Com-
puting, Networking, Storage and Analysis. IEEE, 1–18.

[31] Jemin Lee, Sihyeong Park, Jinse Kwon, Jihun Oh, and Yongin Kwon.
2024. Exploring the Trade-Offs: Quantization Methods, Task Difficulty,
and Model Size in Large Language Models From Edge to Giant. arXiv
preprint arXiv:2409.11055 (2024).

[32] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of

Summaries. In Text Summarization Branches Out. 74–81.

[33] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-
Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song
Han. 2024. Awq: Activation-aware weight quantization for on-device
llm compression and acceleration. Proceedings of Machine Learning
and Systems (2024).

[34] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan
Xiao, Chuang Gan, and Song Han. 2024. Qserve: W4a8kv4 quanti-
zation and system co-design for efficient llm serving. arXiv preprint
arXiv:2405.04532 (2024).

[35] Ruilong Ma, Xiang Yang, Jingyu Wang, Qi Qi, Haifeng Sun, Jing Wang,
Zirui Zhuang, and Jianxin Liao. 2024. Hpipe: Large language model
pipeline parallelism for long context on heterogeneous cost-effective
devices. In Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 6: Industry Track). 1–9.

[36] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea,
Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Hei-
necke, Patrick Judd, John Kamalu, et al. 2022. Fp8 formats for deep
learning. arXiv preprint arXiv:2209.05433 (2022).

[37] Aditi Mishra, Sajjadur Rahman, Kushan Mitra, Hannah Kim, and Es-
tevam Hruschka. 2024. Characterizing Large Language Models as
Rationalizers of Knowledge-intensive Tasks. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational
Linguistics, Bangkok, Thailand, 8117–8139.

[38] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos,
and Tijmen Blankevoort. 2020. Up or down? adaptive rounding for
post-training quantization. In International conference on machine
learning. PMLR.

[39] Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling.
2019. Data-free quantization through weight equalization and bias
correction. In Proceedings of the IEEE/CVF international conference on
computer vision.

[40] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bon-
darenko, Mart Van Baalen, and Tijmen Blankevoort. 2021. A white
paper on neural network quantization. arXiv preprint arXiv:2106.08295
(2021).

[41] Sophia Nguyen, Beihao Zhou, Yi Ding, and Sihang Liu. 2024. Towards
Sustainable Large Language Model Serving. In Proceedings of the 3rd
Workshop on Sustainable Computer Systems (HotCarbon).

[42] NVIDIA Corporation. [n. d.]. TensorRT-LLM: An Open-Source Li-
brary for Accelerating Large Language Model Inference on NVIDIA
GPUs. https://github.com/NVIDIA/TensorRT-LLM. https://github.
com/NVIDIA/TensorRT-LLM

[43] NVIDIA Corporation. 2020. NVIDIA A100 Tensor Core GPU. Product
Datasheet. https://www.nvidia.com/content/dam/en-zz/Solutions/
Data-Center/a100/pdf/nvidia-a100-datasheet.pdf

[44] NVIDIA Corporation. 2022. NVIDIA H100 Tensor Core GPU. Product
Datasheet. https://resources.nvidia.com/en-us-hopper-architecture/
nvidia-tensor-core-gpu-datasheet

13

[45] OpenAI. [n. d.]. ChatGPT. https://chatgpt.com/.
[46] Lin Ou, Jinpeng Xia, Yuewei Zhang, Chuzhan Hao, and Hao Henry
Wang. 2024. Adaptive quantization error reconstruction for llms with
mixed precision. In First Conference on Language Modeling.

[47] Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon
Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee,
and Dongsoo Lee. 2022. Lut-gemm: Quantized matrix multiplication
based on luts for efficient inference in large-scale generative language
models. arXiv preprint arXiv:2206.09557 (2022).

[48] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Brijesh
Warrier, Nithish Mahalingam, and Ricardo Bianchini. 2024. Charac-
terizing Power Management Opportunities for LLMs in the Cloud. In
Proceedings of the 29th ACM International Conference on Architectural
Support for Programming Languages and Operating Systems, Volume 3.
[49] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo
Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient
generative LLM inference using phase splitting. In ISCA.

[50] Cheng Peng, Xi Yang, Aokun Chen, Kaleb E. Smith, Nima PourNejatian,
Anthony B. Costa, Cheryl Martin, Mona G. Flores, Ying Zhang, Tanja
Magoc, Gloria Lipori, Duane A. Mitchell, Naykky S. Ospina, Mustafa M.
Ahmed, William R. Hogan, Elizabeth A. Shenkman, Yi Guo, Jiang Bian,
and Yonghui Wu. 2023. A study of generative large language model
for medical research and healthcare. npj Digital Medicine 6, 1 (2023),
210.

[51] Soham Poddar, Paramita Koley, Janardan Misra, Niloy Ganguly, and
Saptarshi Ghosh. 2025. Brevity is the soul of sustainability: Char-
acterizing LLM response lengths. In Findings of the Association for
Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende,
Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association
for Computational Linguistics, Vienna, Austria, 21848–21864.
[52] Jianing Qiu, Kyle Lam, Guohao Li, Amish Acharya, Tien Yin Wong, Ara
Darzi, Wu Yuan, and Eric J Topol. 2024. LLM-based agentic systems
in medicine and healthcare. Nature Machine Intelligence 6, 12 (2024),
1418–1420.

[53] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty,
Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R
Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark.
In First Conference on Language Modeling.

[54] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin
Choi. 2021. Winogrande: An adversarial winograd schema challenge
at scale. Commun. ACM 64, 9 (2021), 99–106.

[55] ShareGPT. [n. d.]. ShareGPT - Share and Save Your Conversations

with AI. https://sharegpt.com/.

[56] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max
Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and
Ce Zhang. 2023. Flexgen: High-throughput generative inference of
large language models with a single gpu. In International Conference
on Machine Learning. PMLR.

[57] Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Josep Torrellas, and Esha
Choukse. 2025. Dynamollm: Designing llm inference clusters for per-
formance and energy efficiency. In 2025 IEEE International Symposium
on High Performance Computer Architecture (HPCA). IEEE, 1348–1362.
[58] Mirac Suzgun, Nathan Chen, Yonatan Efrat, Hua Li, Austin Chen, Yang
Liu, Xiaodong Zheng, Jun Kasai, Roman Schärli, Weizhe Li, et al. 2022.
Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can
Solve Them. arXiv preprint arXiv:2210.09261 (2022).

[59] Wei Tao, Haocheng Lu, Xiaoyang Qu, Bin Zhang, Kai Lu, Jiguang Wan,
and Jianzong Wang. 2025. MoQAE: Mixed-Precision Quantization
for Long-Context LLM Inference via Mixture of Quantization-Aware
Experts. In Proceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che,
Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar
(Eds.). Association for Computational Linguistics, Vienna, Austria,
10810–10820.

Tianyao Shi and Yi Ding

scientific discovery in molecular property prediction. Nature Machine
Intelligence (2025), 1–11.

[73] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xu-
anzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating
Prefill and Decoding for Goodput-optimized Large Language Model
Serving. In 18th USENIX Symposium on Operating Systems Design and
Implementation (OSDI).

[60] Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad
Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Pra-
jjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cris-
tian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude
Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj
Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui
Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Is-
abel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne
Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yun-
ing Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Moly-
bog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta,
Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subra-
manian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang
Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela
Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert
Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open
Foundation and Fine-Tuned Chat Models. ArXiv (2023).

[61] Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessan-
dro Sordoni, Philip Bachman, and Kaheer Suleman. 2016. Newsqa:
A machine comprehension dataset. arXiv preprint arXiv:1611.09830
(2016).

[62] Jaylen Wang, Udit Gupta, and Akshitha Sriraman. 2023. Peeling back
the carbon curtain: Carbon optimization challenges in cloud com-
puting. In Proceedings of the 2nd Workshop on Sustainable Computer
Systems (HotCarbon).

[63] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebas-
tian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald
Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang,
Jeff Dean, and William Fedus. 2022. Emergent Abilities of Large Lan-
guage Models. Transactions on Machine Learning Research (2022).
[64] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shang-
hang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. 2022. Outlier
suppression: Pushing the limit of low-bit transformer language models.
Advances in Neural Information Processing Systems 35 (2022).
[65] Bingbing Wen, Bill Howe, and Lucy Lu Wang. 2024. Characterizing
LLM Abstention Behavior in Science QA with Context Perturbations.
In Findings of the Association for Computational Linguistics: EMNLP
2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.).
Association for Computational Linguistics, Miami, Florida, USA, 3437–
3450.

[66] Yanran Wu, Inez Hua, and Yi Ding. 2025. Not All Water Consumption
Is Equal: A Water Stress Weighted Metric for Sustainable Computing.
In The 4th Workshop on Sustainable Computer Systems (HotCarbon).

[67] Yanran Wu, Inez Hua, and Yi Ding. 2025. Unveiling Environmental
Impacts of Large Language Model Serving: A Functional Unit View.
In The 63rd Annual Meeting of the Association for Computational Lin-
guistics Main Conference (ACL).

[68] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and
Song Han. 2023. Smoothquant: Accurate and efficient post-training
quantization for large language models. In International Conference
on Machine Learning. PMLR.

[69] Zhuoyan Xu, Zhenmei Shi, and Yingyu Liang. 2024. Do Large Lan-
guage Models Have Compositional Ability? An Investigation into
Limitations and Scalability. In First Conference on Language Modeling.
[70] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu,
Conglong Li, and Yuxiong He. 2022. Zeroquant: Efficient and affordable
post-training quantization for large-scale transformers. Advances in
Neural Information Processing Systems 35 (2022).

[71] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin
Choi. 2019. HellaSwag: Can a Machine Really Finish Your Sentence?.
In Proceedings of the 57th Annual Meeting of the Association for Com-
putational Linguistics.

[72] Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen, Lauren T
May, Geoffrey I Webb, and Shirui Pan. 2025. Large language models for

14

