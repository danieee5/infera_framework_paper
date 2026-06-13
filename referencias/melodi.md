The Price of Prompting: Profiling Energy Use in
Large Language Models Inference

Erik Johannes Husom, Arda Goknil, Lwin Khin Shar, and Sagar Sen

1

6
2
0
2

r
a

M
3

]

Y
C
.
s
c
[

Abstract—In the realm of Artificial Intelligence (AI), the deployment of
Large Language Models (LLMs) has become fundamental to advancing
technology and enriching user experiences across various applications.
However, as these models grow in complexity and size, they pose signif-
icant computational and environmental challenges due to their substan-
tial energy consumption. The increasing deployment of LLMs accentu-
ates the need for sustainable practices to manage their energy demands
effectively, especially in resource-constrained environments. In this pa-
per, we introduce MELODI – Monitoring Energy Levels and Optimization
for Data-driven Inference – a multifaceted framework crafted to monitor
and analyze the energy consumed during LLM inference processes.
Unlike existing tools that measure energy at the system level, MELODI
uniquely combines process-level CPU monitoring via Scaphandre with
GPU tracking via nvidia-smi, employing configurable temporal buffers to
capture complete inference cycles with high precision. MELODI enables
detailed observations of power consumption dynamics and facilitates
the creation of datasets reflective of energy efficiency across various
deployment scenarios. We apply MELODI to profile inference-time en-
ergy consumption under a diverse set of scenarios. Our findings reveal
pronounced energy efficiency disparities: large models (≥70B parame-
ters) consume roughly two orders of magnitude more energy per token
than smaller models, while response token length is a strong predictor of
energy use (R2 > 0.95 in most settings). We further derive a predictive
model with an R2 value of 0.9962, utilizing response length, model type,
and hardware. Prompt characteristics exhibit minimal correlation with
energy consumption. In contrast, hardware choice has a substantial
impact—laptop deployments consistently consume more energy than
workstations for comparable models—highlighting significant opportuni-
ties for optimization and more sustainable LLM deployment.

1 INTRODUCTION
The swift progression of artificial
intelligence (AI) has
precipitated the emergence of advanced natural language
processing (NLP) systems. Large language models (LLMs)
have ascended to prominence among these systems, proving
indispensable across a vast spectrum of applications. Their
utility ranges from simple text predictions to managing
complex dialogues. In various domains, LLMs have shown
great promise in automating and enhancing a variety of
tasks that traditionally require significant human expertise
and effort [1]–[5]. For instance, LLMs can assist in con-
tent generation [6]–[8], customer support [9]–[11], machine
translation [12]–[14], sentiment analysis [15]–[17], and legal

• E.J. Husom, A. Goknil, and S. Sen are affiliated with SINTEF, Norway.
L.K. Shar is affiliated with Singapore Management University, Singapore.
arda.goknil@sintef.no
E-mails:
lkshar@smu.edu.sg sagar.sen@sintef.no

erik.johannes.husom@sintef.no

Manuscript received June 2025; revised January 2026.

2
v
3
9
8
6
1
.
7
0
4
2
:
v
i
X
r
a

✦

document review, thereby streamlining operations across
diverse sectors. The ability of LLMs to generate human-
like text with high accuracy and relevance has made them
invaluable tools for professionals, enabling them to focus on
more complex and creative aspects of their disciplines.

Context and Motivation. As these models grow in com-
putational complexity, their energy demands escalate corre-
spondingly [18]–[20]. The widespread deployment of LLMs
across various applications, from code generation tools used
in iterative workflows to general-purpose chatbots with
hundreds of millions of users, makes individual usage accu-
mulate to massive resource consumption. Unlike training,
which occurs once per model, inference is a continuous
operational cost that scales with usage. In an epoch that
prioritizes sustainable computing, the energy expenditure
of LLMs is a consideration of mounting importance and one
that commands focused attention [2], [21]. Therefore, the
integration of LLMs in diverse applications from healthcare
to customer service without compounding the carbon foot-
print becomes paramount, thus underscoring the need for
techniques that can mitigate the energy demands of LLMs
while maintaining their functionality and accessibility.

Problem Statement. Despite the widespread adoption
of LLMs, understanding their energy implications remains
a significant challenge [22]. Prior studies have predomi-
nantly concentrated on the training phase of traditional
machine learning (ML) models, with the inference phase
often relegated to a secondary focus despite its cumulative
energy footprint over the lifespan of a model [23]. Tools
such as Green Algorithms [24] provide estimations of carbon
emissions for computational tasks and yet lack real-time
monitoring capabilities. Furthermore, while initiatives like
CodeCarbon [25] and its integration to ML pipelines [26]
offer a more automated tracking of emissions, they neither
cater to the nuanced demands of LLMs nor provide granular
data concerning inference-related energy consumption.

Recent research [21], [27]–[33] highlights the signifi-
cant energy and carbon footprints of LLM configurations.
Wilkins et al. [28] introduce an offline, workload-based
framework to optimize energy consumption for LLM infer-
ence on mixed GPU-CPU systems, focusing on energy usage
and runtime based on token interactions. Their method
faces limitations like computational overhead due to fre-
quent sampling and lack of granularity. Conversely, Samsi
et al. [27] examine the energy consumption of GPUs in
LLaMA models, excluding CPU impacts. Prior work neither

2

systematically applies a comprehensive framework for mon-
itoring and analyzing LLM inference energy across diverse
operational conditions, nor investigates real-time optimiza-
tion across heterogeneous hardware. To address this gap,
we aim to develop a holistic energy monitoring framework
tailored to LLM deployments and use it to analyze how
prompt characteristics and hardware configurations influ-
ence energy consumption. The framework should advance
understanding of LLM energy use while enabling data-
driven optimization strategies to support more efficient and
sustainable AI engineering.

Our Design. This paper introduces the MELODI frame-
work designed to monitor and analyze energy usage during
LLM inference. The urgent need for such a framework stems
from the growing environmental and economic costs asso-
ciated with the operation of LLMs. MELODI stands at the
intersection of ML, software engineering, and sustainability,
encapsulating a holistic approach. MELODI leverages two
specialized power usage monitoring tools (Scaphandre [34]
and Nvidia-smi [35]). Scaphandre tracks the CPU’s power
consumption for the LLM process, while nvidia-smi mea-
sures the GPU’s total power usage. For accurate measure-
ments, the GPU must be exclusively used for the LLM
inference, with no concurrent processes. MELODI records
the energy consumption for each inference process, in ad-
dition to the prompt and the LLM’s response. Furthermore,
it facilitates the collection of metadata associated with LLM
inference tasks, enabling a deeper analysis of the relation-
ship between task complexity and energy utilization.

Exploratory Study. We conducted a study using
MELODI to profile inference-time energy consumption un-
der a diverse set of LLM inference scenarios. We deployed
multiple open-source LLMs across heterogeneous hard-
ware platforms, ranging from CPU-only laptops to GPU-
equipped workstations and servers, and evaluated them
on two representative prompt datasets (Alpaca [36] and
Code-Feedback [37]). For each inference request, MELODI
token-level metadata,
recorded prompt–response pairs,
timestamps, and CPU/GPU power traces, which were in-
tegrated into an energy consumption dataset enabling com-
parative analyses across model types, model sizes, hardware
configurations, and workload characteristics.

Our study reveals pronounced variability in LLM infer-
ence energy across deployment conditions: 70B-class models
consume up to two orders of magnitude more energy per
token than smaller models, and laptop deployments are
generally less efficient than workstations, with measurable
differences even between models of similar size. Energy
consumption is driven primarily by response characteristics
(response length and duration), while prompt complexity
features show weak associations. Using response length,
we achieve highly accurate energy prediction and de-
rive an interpretable model with near-perfect performance
(R2 = 0.9962) based on response length, model type, and
hardware. Finally, repeated runs show model-dependent
variability, and comparisons with other monitoring tools
expose substantial measurement discrepancies, highlighting
the need for robust inference-time energy monitoring.

To summarize, our contributions include:

• MELODI Framework: We introduce MELODI, an

open-source and extensible framework for fine-
grained monitoring and analysis of CPU and GPU
energy consumption during LLM inference, enabling
reproducible energy profiling at the level of individ-
ual inference requests.

• Energy Consumption Dataset: Using MELODI, we
construct and release a comprehensive inference-
time energy dataset spanning multiple hardwares,
LLM families and model sizes, and prompt datasets,
supporting systematic benchmarking of energy effi-
ciency across deployment conditions.

• Empirical Characterization of Energy Drivers: We
conduct an extensive exploratory study identifying
how model size, model type, hardware configura-
tion, and prompt/response characteristics influence
inference energy consumption, providing evidence-
based insights for energy-aware deployment.

• Prediction and Interpretable Modeling: We develop
and assess predictive models for inference energy,
demonstrating that energy can be reliably modeled
from response length, model type, and hardware.

The paper is structured as follows. Section 2 presents the
background regarding energy monitoring tools. In Section 3,
we present MELODI. Section 4 reports on the experiments.
Section 5 presents some future work. Section 6 discusses the
related work. We conclude the paper in Section 7.

2 BACKGROUND: MONITORING TECHNOLOGIES

Advanced monitoring tools and technologies are essential
for tracking and optimizing energy consumption. In this
context, MELODI employs the following tools:

Interface

NVIDIA System Management

(nvidia-
smi) [35] is a command-line utility, based on NVIDIA Man-
agement Library (NVML) [38], providing the real-time mon-
itoring of GPU, including power consumption, for systems
equipped with NVIDIA GPUs. This allows administrators to
query GPU device state and with the appropriate privileges,
permits administrators to modify GPU device state [35].

Scaphandre [34] is a metrology agent designed to en-
hance energy management across various computing en-
vironments. It is one of the few tools providing detailed
insights into the energy usage of computing systems at the
process level [39]. It offers comprehensive features including
the ability to measure power consumption directly on bare
metal hosts and within qemu/kvm virtual machines from
the host. Additionally, Scaphandre facilitates the integration
of power metrics into virtual machines as if they were bare
metal setups and supports exporting these metrics as a
Prometheus HTTP exporter [40].

3 MELODI FOR MONITORING ENERGY USAGE

To analyze LLM energy consumption, we developed the
MELODI framework, which monitors energy use during
LLM inference. Figure 1 outlines its architecture, highlight-
ing its process of sampling prompts, submitting them to an
LLM, and recording the prompts, responses, metadata, and
energy metrics. This collected data allows for a comprehen-
sive analysis of energy trends during LLM operations.

3

Nvidia-smi tracks the power usage of the entire GPU and
cannot distinguish among multiple processes concurrently
utilizing the GPU. To guarantee precision, we limit the GPU
to solely operating the LLM inference process. The two
tools measure the power consumption P of a process in
microwatts (µW). We compute the energy consumption E
by integrating the power consumption over the duration t,
applying the trapezoidal rule. With t (in seconds), we con-
vert E into kilowatt-hours (kWh) by dividing by 3.6 × 1012,
facilitating a standardized energy usage assessment.

the process
the

In each inference,
activates
MELODI
Scaphandre to monitor
usage
CPU power
level,
at
LLM
targeting
service
ensure
to
measurement accuracy.
Concurrently,
nvidia-
smi tracks GPU power
the device
usage at
level,
necessitating
the restriction of GPU
activities solely to LLM
inference to eliminate
external
application
interference. MELODI
logs
prompt,
response, token counts,
and the
timestamps
marking the start and
the inference,
end of
the
recording
while
energy
consumption
and power draw as a
time series.

the

(a) Power draw during inference
(M0 = M1 = 0 and R0 = R1 = 0).

(b) Power draw during inference
(M0 = M1 = 0.5 seconds and
R0 = R1 = 0).

(c) Power draw during inference
(M0 = M1 = 0.5 seconds, and
R0 = 0 and R1 = 0.2 seconds).

To enhance energy
measurement accuracy
during inference, we
implement two types of
buffers in MELODI to
compensate for moni-
toring tool delays. The
first, monitoring buffer
(M ), introduces a delay
before and after the in-
ference process to en-
sure Scaphandre and
nvidia-smi are fully op-
erational also before and after the inference. This buffer is
divided into two configurable intervals: M0, the delay from
monitoring startup to inference start, and M1, the delay
from inference end to monitoring shutdown, effectively
preventing data loss at the critical start and end inference
points due to tool response times.

Fig. 2: Power draw during a
prompt inference as monitored
by MELODI with varying buffer
settings. Thick semi-transparent
lines indicate active monitoring
intervals by Scaphandre (CPU)
and nvidia-smi (GPU), while the
solid line with dots are data
points recorded by MELODI.

Figures 2a and 2b demonstrate the impact of employing
a monitoring buffer on data capture. Figure 2a, utilizing
no buffer (M0 = M1 = 0), shows that nvidia-smi fails
to capture data during extremely brief inference runs, re-
sulting in incomplete energy measurements. Conversely,
Figure 2b, using a 0.5-second buffer for pre- and post-

Fig. 1: Overview of the MELODI framework.

•

Input Data (Prompt Dataset in Figure 1): MELODI
extracts prompts from a pre-existing dataset to query
a deployed LLM through an API client.

• Power Usage Monitoring and Data Collection:
MELODI utilizes two complementary power mon-
itoring tools (i.e., Scaphandre [34] and nvidia-
smi [35]) to track the power consumption of LLMs
in Figure 1. Scaphandre
like the Ollama model
records CPU power consumption specific to the LLM
process, while nvidia-smi measures total GPU power
usage, requiring exclusive GPU operation (no other
processes on the GPU) during measurements for
accuracy. MELODI calculates energy consumption
for each prompt, aggregating data on the prompt,
response, and energy metrics for detailed analysis.
• Dataset Generation: MELODI iterates through all
prompts from the prompt dataset. The data obtained
for each prompt are aggregated into a dataset, encap-
sulating the energy consumption of LLM inference.

MELODI relies on LLM services that can be queried via
API endpoints for generating responses. Our implementa-
tion supports services adhering to Ollama or OpenAI API
specifications, covering most LLM deployment services that
provide natural language responses and metadata such as
timestamps and token lengths for prompts and responses.
The landscape of LLM deployment tools is diverse, fea-
turing platforms such as Llama.cpp [41], GPT4All [42],
vLLM [43], Llamafile [44], and Ollama [45]. We utilized
Ollama for our experiments due to its easy setup, compat-
ibility with both CPU and GPU architectures, and support
for several open-source LLMs.

Scaphandre operates with process-level granularity,
monitoring and recording the power usage of active pro-
cesses on a computing device. We apply a regular expression
to the process names to isolate the specific processes associ-
ated with the LLM service. The frequency at which Scaphan-
dre samples power metrics is configurable with granular-
ity fine enough to reach the nanosecond scale. Although
a higher sampling frequency increases the resolution and
accuracy of the data, it also requires more storage for data
retention and increases the energy consumption of Scaphan-
dre. Therefore, we also monitor the energy consumption of
Scaphandre, enabling an assessment of the energy overhead
incurred by the measurement process.

4

TABLE 1: Capture completeness rate (CCR) for different
buffer settings, measured as the percentage of runs (50 per
setting) in which the full power spike was captured. In some
cases, achieving 100% CCR includes excess baseline power
outside the inference window.
(a) M = M0 = M1

(b) R0

(c) R1

M (s)

0.2
0.4
0.5
1.0

CCR

0%
48%
100%
100%

R0 (s)

0.0
0.1
0.2
0.3

CCR

94%
100%
100%
100%

R1 (s)

0.0
0.1
0.2
0.3

CCR

34%
36%
88%
100%

inference (M0 = M1 = 0.5), shows improved reliability
in data capture, accounting even for brief inference periods.
We tested multiple values of M (Table 1a) and found that
M = 0.5 provides optimal data capture without including
unnecessary measurements. A monitoring interval was con-
sidered successful when the power profile was stable both
before and after the inference window.

Even with a monitoring buffer, we observe that GPU
power spikes occasionally persist briefly post-inference, as
depicted in Figure 2b. To address this, we introduce a
recording buffer (R), extending the monitoring period before
and/or after the actual inference window. This buffer is
split into R0, extending the recording time before infer-
ence, and R1, extending after. R1 is crucial for capturing
lingering GPU power spikes after inference concludes, en-
suring comprehensive data collection (see Figure 2c). This
delayed power consumption decline, also observed in other
studies [46], underscores the necessity of an end buffer for
precise measurements.

Determining optimal buffer values of R involves a trade-
off between measurement completeness and estimation bias.
Larger buffers ensure full capture of transient power spikes
but can overestimate energy by including baseline con-
sumption outside the inference window. Smaller buffers
reduce this bias but may miss lingering power decay,
particularly on GPUs (Figure 2c). As shown in Table 1b,
R0 = 0.1 achieves complete capture but consistently in-
cludes 0.1 seconds of pre-inference baseline power, whereas
R0 = 0.0 removes this offset while still capturing 94% of
cases. Similarly, Table 1c shows that R1 = 0.3 guarantees
full capture of post-inference power decay but often ex-
tends beyond the actual spike, while R1 = 0.2 provides
a better balance, achieving 88% completeness with reduced
baseline inclusion. We therefore set R0 = 0.0 and R1 = 0.2
for all experiments, balancing measurement accuracy and
completeness. This choice avoids systematic overestimation
while tolerating a small, controlled loss of data.

4 EXPERIMENTS

We analyze the energy consumption dataset to address three
Research Questions (RQ)s:

• RQ1. How does the energy consumption of LLM
inference vary across different hardware, models, and
prompt datasets?

• RQ2. What is the relationship between prompt
complexity, response characteristics, and the energy
consumption of LLMs during the inference process?

• RQ3. Can we develop a predictive model accurately
forecasting the energy consumption of LLMs based
on prompt features and response characteristics?
• RQ4. To what extent can LLM inference energy con-
sumption be modeled using an interpretable mathe-
matical function of response length, model type, and
hardware configuration?

• RQ5. How does the variability in energy consump-
tion for the same prompt manifest across multiple
executions and different models?

• RQ6. What are the differences in energy measure-
ments among MELODI and other tools, and what
factors contribute to these variations?

4.1 Subjects of the Experiments

This section outlines the energy consumption dataset
collected with MELODI, using diverse computing ma-
chines (from high-end servers to laptops) and multi-
ple prompt datasets for several LLMs. Table 2 lists the
i.e., Alpaca [36]
prompt datasets in dataset collection,
and Code-Feedback [37], and their average energy con-
sumption and response token length. The Alpaca dataset,
produced using OpenAI’s text-davinci-003 engine, contains
52,000 prompts designed to enhance instruction-following
capabilities in language models. The dataset primarily in-
cludes text generation tasks and is tailored to train pre-
trained language models to follow complex instructions
more effectively. The Code-Feedback dataset supports the
OpenCodeInterpreter model [47] designed to refine code
by integrating code execution and human feedback. This
dataset features 68,000 multi-turn interactions, combining
user instructions with compiler responses to enhance model
training in coding scenarios.

TABLE 2: Prompt datasets.

Prompt
dataset
Alpaca
Code-feedback

Average energy
consumption (kWh)
4.49e-04
5.68e-04

Average response
token length

213.98
405.96

TABLE 3: LLMs in the experiments.

Model name/size

codellama-7b
codellama-70b
gemma-2b
gemma-7b
llama3-8b
llama3-70b
phi-2b
qwen2-7b
qwen2-72b

Average energy
consumption (kWh)
2.35e-04
3.33e-03
8.18e-05
1.36e-04
1.34e-04
2.26e-03
3.14e-05
8.37e-05
2.25e-03

Average response
token length

535.03
342.64
279.11
249.33
255.20
251.46
179.30
199.34
210.82

Table 3 lists the LLMs with their average energy con-
sumption and response token lengths. The models range
from codelama-7b to llama3-70b, with energy con-
sumption varying from as low as 8.20 × 10−5 kWh for
gemma-2b to 3.33 × 10−3 kWh for codellama-70b. The
average response token lengths vary, with codelama-7b
generating the longest responses at 535.03 tokens, while
phi-2b records shorter lengths at 179.30 tokens.

Table 4 details the hardware used in our data collection.
The diversity of the hardware covers a wide range of pro-
cessing power, memory capacities, and graphics capabilities,
crucial for a comprehensive analysis of energy consumption
across different computing environments.

TABLE 4: Hardware used in the data collection.

Machine
Server
Workstation
Laptop 1
Laptop 2

CPU
AMD EPYC 7643 48-Core Processor
Intel Xeon W-2223 8-core @ 3.6GHz
Intel i5 11th Gen 12-core @ 2.4GHz
Intel i7 10th Gen 12-core @ 2.7GHz

Memory
528GB
128GB
16GB
32GB

GPU
NVIDIA RTX A5000
NVIDIA RTX A2000
None
NVIDIA Quadro RTX 4000

GPU memory
24GB
12GB
None
8GB

ID

Prompt dataset

Model

Model size

Hardware

No. of prompts

Average energy consumption
per response (kWh)

Average response
token length

TABLE 5: Overview of the energy consumption dataset.

5

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20

code-feedback
code-feedback
code-feedback
code-feedback
alpaca
alpaca
alpaca
code-feedback
code-feedback
alpaca
alpaca
code-feedback
code-feedback
alpaca
alpaca
alpaca
alpaca
alpaca
alpaca
alpaca

codellama
codellama
codellama
codellama
gemma
gemma
gemma
gemma
gemma
gemma
gemma
gemma
gemma
llama3
llama3
phi
phi
qwen2
qwen2
qwen2

7b
7b
7b
70b
2b
2b
2b
2b
2b
7b
7b
7b
7b
8b
70b
2b
2b
7b
7b
72b

laptop1
laptop2
workstation
workstation
laptop1
laptop2
workstation
laptop2
workstation
laptop2
workstation
laptop2
workstation
laptop2
server
laptop2
workstation
laptop2
workstation
workstation

5347
3572
1000
1000
5347
5101
1000
5008
1000
5099
1000
3405
1000
5101
1026
674
1000
821
1000
1000

1.85e-04
2.46e-04
2.73e-04
3.33e-03
1.85e-04
4.70e-05
3.84e-05
7.33e-05
6.60e-05
9.81e-05
8.05e-05
2.00e-04
1.65e-04
1.34e-04
2.26e-03
3.22e-05
3.15e-05
1.22e-04
1.11e-04
2.25e-03

403.63
518.40
685.06
342.64
401.64
181.52
184.30
303.98
324.14
160.60
164.90
332.62
339.21
255.20
251.46
221.63
136.98
300.70
97.99
210.82

Table 2 summarizes the energy dataset collected by
MELODI across multiple hardware setups and LLMs under
diverse operational conditions. Each sample includes the
prompt and response, prompt/response token counts, API
timestamp, time-series power traces for the LLM process
and Scaphandre during monitoring, and the aggregated
inference energy consumption.

4.2 Experiments Setup

Data Preparation. Our raw data consists of power usage
over time, which we aggregate to calculate the energy con-
sumption for each inference process as outlined in Section 3.
These aggregated values are compiled into distinct energy
consumption datasets, categorized by the specific hardware,
model, and prompt dataset utilized during the inference.

Statistical Analysis. Our analysis of the energy con-
sumption datasets focuses on two metrics: energy per token
and energy per response. Energy consumption per token is
crucial as it evaluates resource use across different inference
setups regardless of text volume. Meanwhile, energy per
response is beneficial for estimating resource usage relative
to the input prompt. We conduct statistical analyses on these
metrics to compare across model types, sizes, hardware
setups, and the specific prompt datasets used during infer-
ence, providing a comprehensive view of energy dynamics.
Visualization. We employ box plots to visualize the dis-
tribution of energy consumption for each model and hard-
ware. The box plots depict the median energy consumption
as a red line, the interquartile range (IQR) with the main box,
and extend to 1.5 times the IQR with whiskers, providing a
clear summary of variability and central tendency.

Interpretation. We interpret results through a compara-
tive analysis of energy consumption across different scenar-
ios. The relationship between prompt complexity and en-
ergy consumption is assessed using the Pearson correlation
coefficient, which ranges from 0 to 1 to indicate variable
correlation. Additionally, predictive models for energy con-
sumption are evaluated using the R2 score.

4.3 Results

4.3.1 RQ1 (energy consumption of LLM inference across
different setups)
To address RQ1, we performed a comparative analysis
of energy consumption across various LLMs operating on
different hardware (see Figures 3 and 4).

Model Size. Figures 3a and 4 categorize energy con-
sumption per token by the model utilized for infer-
ence. Notably, the largest models (codellama-70b with
Code-Feedback, qwen2-72b and llama3-70b with
Alpaca) exhibit energy usage approximately 100 times
greater than their smallest counterparts (codellama-7b
with Code-Feedback, qwen2-7b and llama3-8b with
Alpaca). Additionally, the larger models (gemma-7b with
both Alpaca and Code-Feedback) show an energy con-
sumption that is roughly ten times higher than their smaller
versions (gemma-2b with Alpaca and Code-Feedback).

Hardware. Figure 3b organizes results by the hard-
ware. Energy consumption is noticeably higher for the
laptops than the workstation. Laptop 1 (CPU-only) shows
similar energy usage when operating different models
(codellama-7b with Code-Feedback and gemma-2b
with Alpaca), suggesting inefficiencies in CPU-based pro-
cessing for LLM tasks since we would expect a significant
difference when running a 2b model vs a 7b model. Com-
parisons between the workstation and Laptop 2, despite the
former’s more robust GPU (12GB vs. 8GB), reveal roughly
equivalent energy usage, underscoring potential disparities
in hardware efficiency.

Model. In energy assessments of different models of
the same size (codellama-7b with Code-Feedback vs.
gemma-7b with Code-Feedback) run on the worksta-
tion and Laptop 2, gemma-7b displayed higher energy
consumption than codellama-7b. This observation sug-
gests that codellama-7b is a slightly more energy-efficient
model than gemma-7b. phi-2b shows comparable energy
consumption per token to the gemma-2b of similar size but
has a slightly higher consumption on the workstation.

6

(a) Comparison of energy consumption per token across models with 2-8 billion parameters.

(b) Comparison of energy consumption per token across hardware setups (Codellama-70b is excluded
for the workstation to not skew the plot).

Fig. 3: Comparative energy consumption per token across inference setups.

Prompt Datasets. To evaluate the energy consumption
per output token across two prompt datasets, we analyzed
the box plots for gemma-2b and gemma-7b in Figure 3a, the
only models tested with both prompt datasets. The analysis
reveals that, except in one instance, the Alpaca dataset
consistently led to higher energy consumption per token.
The IQR, representing variability in energy consumption, is
notably wider in all cases, suggesting greater fluctuation in
energy use when using the Alpaca dataset.

Table 2 facilitates a comparison of energy consumption
per prompt, revealing that the median consumption for
the Code-Feedback prompt dataset exceeds that of the
Alpaca dataset during inference. This discrepancy likely
arises from longer response length of Code-Feedback
prompts. The response length of Code-Feedback prompts
averages at 405.96 tokens whereas the response length of
Alpaca prompts averages at 213.98 tokens.

RQ1 Conclusion. Larger models like codellama-70b
and llama3-70b use roughly 100 times more energy
per token than smaller ones. Energy demands are higher
for LLMs running on laptops than workstations, partic-
ularly due to inefficiencies in CPU processing. Models
of the same size exhibit varying energy efficiencies. The
Code-Feedback dataset leads to greater energy use per
token than Alpaca, likely due to longer response lengths.

4.3.2 RQ2 (relationship between prompt complexity, re-
sponse characteristics and the energy consumption)

To respond to RQ2, we used Python libraries, namely,
Spacy [48], nltk [49], textblob [50], and textstat [51], to
derive 55 text-based features. We integrated the features

Fig. 4: Comparison of energy consumption per token across
models with 70-72 billion parameters.

with our energy consumption dataset and calculated the
Pearson correlation coefficient between energy consumption
per prompt and each feature.

We extracted prompt features spanning multiple cate-
gories, including lexical richness and word-level complexity
(e.g., average word length, long/polysyllabic word counts,
lexical diversity), part-of-speech frequencies (e.g., noun and
adjective counts), readability indices (e.g., Flesch, SMOG,
Coleman–Liau, Dale–Chall), sentence-level structure (e.g.,
mean sentence length, punctuation counts), and sentiment
scores. Table 6 reports the top 20 correlated features.

Our analysis revealed that significant correlations with
energy consumption per prompt are primarily linked to
response characteristics rather than the prompt complexity.
Key factors such as response duration, total duration, and
response token length show strong positive correlations
with energy consumption (with coefficients of 0.770, 0.766,
and 0.755, respectively). This observation suggests that en-
ergy consumption escalates with an increase in response

laptop1laptop2workstationcodellama_7b456energy_per_token (kWh)1e7workstationworkstationlaptop1laptop2laptop2gemma_2b24681e7workstationlaptop2workstationlaptop2gemma_7b0.51.01.51e6laptop2llama3_8b456781e7laptop2workstationphi_2b02461e7workstationlaptop2qwen2_7b121e6codellama_7bgemma_2blaptop1456energy_per_token (kWh)1e7codellama_7bgemma_2bgemma_2bgemma_7bgemma_7bllama3_8bphi_2bqwen2_7blaptop20.00.51.01.51e6codellama_7bgemma_2bgemma_2bgemma_7bgemma_7bphi_2bqwen2_7bworkstation1231e6llama3_70bserver0.81.01.21.4energy_per_token (kWh)1e5codellama_70bqwen2_72bworkstation0.91.01.11.21e5TABLE 6: Correlations between text features and energy
consumption per response across all samples. Features are
derived from the prompt unless marked with (*), which
indicates response-based features.

7

Feature
*response duration
*total duration
*response token length
prompt duration
adj count
polysyllabcount
long word count
syllable count
letter count
char count

Feature
reading time
lexicon count
adverb count
prompt token length
stop word count
noun count
verb count

Correlation
0.770
0.766
0.755
0.129
0.101
0.087
0.085
0.079 monosyllabcount
0.079
0.076

word count
sentence count

Correlation
0.076
0.075
0.073
0.073
0.070
0.070
0.068
0.067
0.065
0.058

tokens due to more extensive processing within the model,
highlighting that longer and more time-intensive responses
drive higher energy usage.

We observe only low positive correlations with prompt
complexity features. Attributes like prompt duration, adjec-
tive count, and syllable count demonstrate modest correla-
tions with energy consumption, with coefficients of 0.129,
0.101, and 0.087, respectively. These results indicate that
prompt complexity (given our set of features) has a minimal
impact on energy consumption. Instead, the length and du-
ration of responses are more significant factors. This insight
suggests that optimizing the response generation process
could be more effective in reducing energy consumption
than merely simplifying the input prompts.

RQ2 Conclusion. The results show that response char-
acteristics such as token length and duration are strongly
correlated with energy usage, indicating higher consump-
tion with longer responses. Conversely, prompt complex-
ity features like adjective and syllable counts exhibit low
correlations, suggesting minimal impact on energy use.
Managing response generation may offer more significant
energy savings than simplifying prompts.

4.3.3 RQ3 (predictive model for LLM energy consumption)

To address RQ3, we trained ML models on the energy
dataset using either prompt features or response character-
istics as inputs. We evaluated Random Forest (RF), Gradi-
ent Boosting (GB), Linear Regression (LR), XGBoost (XGB),
Decision Tree (DT), and Support Vector Machine (SVM),
training two models per dataset: one using response token
length and one using prompt-derived features.

Table 6 shows that energy consumption is strongly corre-
lated with response characteristics such as token length and
duration, suggesting high potential for accurate prediction.
Figure 5 indicates that several ML models can predict en-
ergy consumption from response length with high accuracy:
the best model achieves R2 > 0.78 across all datasets, and
R2 > 0.95 for all but the bottom four. As summarized in
Table 7, LR performs best overall, with an average R2 = 0.96
and top performance in 13 of 20 datasets. The lowest scores
are observed on Laptop1 (R2 = 0.77), indicating possible
measurement inaccuracies in this CPU-only setup.

Our results suggest that limiting response length (e.g., by
specifying a maximum output size in the prompt) can sub-
stantially reduce inference energy consumption. In contrast,
models trained solely on prompt-derived features show
limited predictive performance (Figure 7).

Fig. 5: R2 scores across different ML algorithms, using only
response length for forecasting energy consumption per
response. Negative R2 scores are clipped at -1.0.

Fig. 6: R2 scores across different ML algorithms, using
prompt characteristics for forecasting energy consumption
per response. Negative R2 scores are clipped at -1.0.

Fig. 7: R2 scores of GB models using prompt characteristics
for forecasting energy consumption per response.

Figure 6 reports R2 scores for models trained solely on
prompt-derived features, excluding any response informa-
tion, to assess whether inference energy can be predicted
a priori. Overall performance is low: only four datasets
achieve R2 > 0.44, and none exceed 0.50. As summarized
in Table 7, LR attains the highest average R2, while GB
performs best in roughly half of the datasets, indicating
that predicting inference energy from prompt characteris-
tics alone remains challenging. Notably, predictive accuracy
varies substantially across datasets and appears primarily
driven by model type: in Figure 7, GB models yield mostly

1.00.50.00.51.0R2 scorealpaca_gemma_2b_workstationalpaca_phi_2b_workstationalpaca_gemma_7b_laptop2alpaca_gemma_2b_laptop2codefeedback_codellama_7b_workstationalpaca_qwen2_72b_workstationalpaca_llama3_8b_laptop2alpaca_gemma_7b_workstationcodefeedback_codellama_70b_workstationalpaca_qwen2_7b_workstationcodefeedback_gemma_2b_laptop2codefeedback_gemma_2b_workstationcodefeedback_codellama_7b_laptop2codefeedback_gemma_7b_laptop2codefeedback_gemma_7b_workstationalpaca_llama3_70b_serveralpaca_qwen2_7b_laptop2alpaca_phi_2b_laptop2codefeedback_codellama_7b_laptop1alpaca_gemma_2b_laptop1Energy consumption datasetML ModelGBRFDTLRXGBSVM1.000.750.500.250.000.250.50R2 scorecodefeedback_gemma_7b_laptop2codefeedback_gemma_7b_workstationcodefeedback_gemma_2b_laptop2codefeedback_gemma_2b_workstationalpaca_llama3_8b_laptop2alpaca_gemma_7b_laptop2alpaca_gemma_2b_laptop2alpaca_llama3_70b_serveralpaca_gemma_7b_workstationalpaca_qwen2_7b_laptop2alpaca_gemma_2b_workstationcodefeedback_codellama_70b_workstationalpaca_qwen2_72b_workstationcodefeedback_codellama_7b_workstationalpaca_phi_2b_workstationalpaca_phi_2b_laptop2codefeedback_codellama_7b_laptop2codefeedback_codellama_7b_laptop1alpaca_qwen2_7b_workstationalpaca_gemma_2b_laptop1Energy consumption datasetML ModelRFGBLRXGBDTSVM1.000.750.500.250.000.250.50R2 scorecodefeedback_gemma_7b_laptop2codefeedback_gemma_2b_laptop2codefeedback_gemma_7b_workstationcodefeedback_gemma_2b_workstationalpaca_llama3_8b_laptop2alpaca_gemma_7b_laptop2alpaca_gemma_2b_laptop2alpaca_llama3_70b_serveralpaca_gemma_7b_workstationalpaca_qwen2_7b_laptop2alpaca_gemma_2b_workstationcodefeedback_codellama_70b_workstationalpaca_qwen2_72b_workstationalpaca_phi_2b_laptop2alpaca_phi_2b_workstationalpaca_gemma_2b_laptop1codefeedback_codellama_7b_laptop1alpaca_qwen2_7b_workstationcodefeedback_codellama_7b_workstationcodefeedback_codellama_7b_laptop2Energy consumption datasetClipped (-2.24)Clipped (-5.54)Clipped (-26.41)Clipped (-30.52)Clipped (-2632.82)LLM typegemmallama3qwen2codellamaphi8
TABLE 7: Average R2 scores and number of wins for various
ML algorithms using two feature sets: (i) only response
length, and (ii) prompt characteristics.

ML alg.

Response length only

Prompt characteristics

Avg. R2

No. of wins

Avg. R2

No. of wins

DT
GB
LR
RF
SVM
XGB

0.87
0.89
0.96
0.89
-12392.08
-1.17

0
5
13
2
0
0

-156.14
-134.72
-18.34
-48.01
-12392.08
-0.47

0
10
0
6
0
4

positive R2 for Gemma, low negative scores for Phi, and the
worst performance for CodeLlama. In contrast, hardware,
model size, and prompt dataset show no consistent ef-
fect. We hypothesize that model-specific training differences
affect the predictability of response length from prompt
features, but more controlled experiments are needed.

RQ3 Conclusion. ML models, particularly Linear Re-
gression, predict inference energy accurately when using
response length as input, with most datasets achieving
R2 > 0.95. In contrast, models trained solely on prompt
features perform poorly,
though results vary widely
across datasets. This suggests that some LLM families
exhibit a stronger, more predictable relationship between
prompt characteristics and response length than others.

4.3.4 RQ4 (mathematical model for energy consumption
based on response length, model and hardware)

The consistently high predictive performance of the LR
models in RQ3 suggests that inference-time energy con-
sumption can be approximated through a compact and in-
terpretable mathematical formulation. To investigate this in
RQ4, we aggregated data from all experiments and trained
an LR model with response length, model attributes, and
hardware configuration as predictors. We model the energy
consumption E (kWh) of a single inference instance as:

E = β0 + β1 · ntokens + β2 · smodel +

(cid:88)

j

γj · Ihwj +

(cid:88)

i,j

where:

(cid:88)

i

αi · Itypei

+

δi,j · (ntokens × Icati,j )

(1)

•

β0 is the base energy cost (intercept),
β1 is the baseline energy per response token,
•
• ntokens is the number of generated response tokens,
β2 is the coefficient for model size (parameters),
•
smodel is the model size,

•
• αi are model type effects encoded as categorical
indicators (codellama, gemma, llama3, phi, qwen2)
Itypei are indicator variables for model types,
γj are hardware effects encoded as categorical indi-
cators (laptop, workstation, server),
Ihwj are indicator variables for hardware setups,
δi,j are interaction coefficients that allow the energy-
per-token rate to vary across model types and hard-
ware configurations.

•

•

•

•

Interaction terms were included because the empirical
results indicate that token-level energy scaling differs across
model architectures and hardware setups, making a single

global slope insufficient to capture observed behavior. Nu-
merical predictors include response length (primary driver)
and model size, while categorical predictors include 11
models (codellama, codellama-70b, codellama-7b,
gemma-2b, gemma-7b, llama3, llama3-70b, phi,
qwen2, qwen2-72b, qwen2-7b) and 4 hardware configura-
tions (Laptop 1, Laptop 2, Workstation, Server). We encoded
categorical variables with one-hot encoding and removed
the first category to avoid multicollinearity.

Model Performance. The fitted LR model achieves very
strong performance across all combined datasets, reaching
R2 = 0.9962. This result indicates that inference energy
can be predicted accurately using a simple, interpretable
formulation grounded in response length, model identity,
and deployment context.

Estimated Coefficients and Interpretation. The esti-
mated baseline coefficient for response length was β1 =
5.28 × 10−7 kWh/token, capturing the average marginal
cost per generated token across all configurations. In con-
trast, model size exhibited only a negligible direct effect
(β2 = 3.96 × 10−18 kWh/billion parameters), suggesting
that model size primarily impacts energy indirectly through
model-family and configuration-specific token-level scal-
ing captured by categorical and interaction terms.

Model family effects showed substantial variability in

intercept shifts. For example:

•

•

•

codellama-70b exhibited a strong negative inter-
cept shift (−5.67 × 10−5 kWh - energy offset),
qwen2-7b showed the largest positive shift (+3.98×
10−5),
gemma and llama3 showed more moderate shifts in
the range of +2.4 to +3.2 × 10−5 kWh.

Hardware-specific intercept terms differed, with server
having the highest baseline energy, consistent with higher
system-level power draw even for comparable workloads.
• Server: +3.24 × 10−5 kWh (highest baseline cost,

likely due to higher power draw)

• Workstation: −1.42 × 10−6 kWh (slight reduction in

baseline cost)

• Laptop 2: +2.31 × 10−6 kWh (small positive shift)
More importantly, token-level scaling varied substan-

tially across configurations. For example:

•

•

•

qwen2-72b and codellama-70b showed the
steepest energy-per-token scaling,
llama3-70b exhibited similarly high scaling,
Smaller models such as gemma-2b and phi-2b
showed considerably lower scaling slopes.

These findings reinforce that inference energy is not
determined solely by parameter count, but also by
architecture-dependent decoding behavior and hardware-
dependent execution efficiency.

Ablation Study. To quantify the relative impact of dif-
ferent feature groups, we performed an ablation study by
fitting reduced LR models:

• Both interactions (full model): R2 = 0.9962
• Hardware interactions only: R2 = 0.8361
• Model type interactions only: R2 = 0.9954
These results indicate that model type and response
length explain most of the variance, while hardware con-
tributes a smaller but non-negligible portion. The findings

suggest that, for practical energy-aware deployment, opti-
mization should prioritize selecting more efficient model
families, and controlling response length, while hardware
choice plays a secondary role compared to these factors.

RQ4 Conclusion. The mathematical model demonstrates
that LLM inference energy consumption is highly pre-
dictable (R2 = 0.9962), with response length being the
dominant factor and model type accounting for nearly
all categorical variation. Our ablation study reveals
that hardware configuration has minimal impact once
model type is known, indicating that energy optimization
should prioritize model selection and response length
control over hardware choices.

4.3.5 RQ5 (variability in energy consumption for the same
prompt across multiple executions and different models)
To address RQ5, we an-
alyzed the energy con-
sumption and response
token length for 100 it-
erations of a prompt
(Give three tips for staying
healthy) from the Alpaca
dataset on seven mod-
els using laptop 2. We
evaluated variability by
using the prompt across
multiple trials. This en-
sures uniformity in our
analysis, as the chosen
prompt is unlikely to in-
fluence variations in en-
ergy consumption and
response length when re-
peatedly processed. Fig-
ure 8 presents differences
in energy consumption
and response variability
among models.
Figure

(a) Variation in energy consump-
tion per prompt
for the same
prompt run multiple times on
seven models.

(b) Variation in energy consump-
tion per
the same
token for
prompt run multiple times on
seven models.

prompt

exhibits

8a presents
the energy consumption
per
across
seven models, revealing
variability.
substantial
Qwen2-7b,
the most
expensive per prompt
overall,
the
highest
IQR of energy
consumption. Gemma-7b
consistently
exhibits
similar consumption per
prompt as the smaller
version of
the model
(gemma-2b). This aligns
with its much lower
response
token length
in Figure 8c. Shorter
generally
responses
reduce the total energy
The
processing.
for

(c) Variation in response token
length for the same prompt run
multiple times on seven models.

Fig. 8: Variability in energy con-
sumption and response token
length for 100 iterations of the
prompt ’Give three tips for stay-
ing healthy’ from Alpaca.

9

variability is the smallest in all the models (see its IQR),
suggesting similar energy use across repeated inferences.

Figure 8b displays the energy consumption per to-
ken, noting a wide IQR for gemma-2b, which is signifi-
cantly greater than that of other models. While gemma-7b
also shows a higher IQR,
is less pronounced than
it
gemma-2b. The largest models (gemma-7b, llama3-8b,
and qwen2-7b) exhibit the highest mean energy consump-
tion per token, as they require more computation. This direct
relationship between model size and energy expenditure is
consistent with our findings, underscoring that the models
with 7B and 8B parameters are more energy-intensive than
their counterparts due to their complex inference processes.
Llama3-8b and qwen2-7b produce the longest re-
sponses, averaging 249 and 264 tokens, respectively (see
Figure 8c), contributing to their higher energy consumption
per prompt. Despite their lengthy outputs, these models
display moderate variability in response length, indicating
consistent response generation for similar prompts.

RQ5 Conclusion. The results revealed significant differ-
ences in energy usage and response length variability
among models. Qwen2-7b demonstrated larger fluctu-
ations in energy consumption per prompt than others,
suggesting factors such as model architecture and opera-
tional conditions may influence energy usage consistency.

4.3.6 RQ6 (comparative analysis of energy measurement
tools in LLM deployments)

To address RQ6, we evaluated per-token energy consump-
tion using 467 to 480 prompts from the Alpaca dataset across
four tools: MELODI, PyJoules [52], CodeCarbon [25], and
EnergyMeter [33]. We selected these tools for comparison
due to their diverse methodologies in monitoring and re-
porting energy consumption in LLM deployments, allowing
us to thoroughly assess and benchmark the performance of
MELODI against established tools. We run phi2-2b [53] on
laptop 2 for the analysis. We opted for phi2-2b primarily
for its rapid inference capabilities. Given that our study
prioritizes the assessment of measurement tools over model
performance, utilizing a fast-processing model ensures uni-
form results across various tools and minimizes delays that
could arise from complex model computations. The model
choice is not expected to influence the observed differences
between the monitoring tools, as the effects are normalized
by averaging over the number of tokens.

Figure 9 displays considerable variation in per-token
energy consumption across the tools. For CPU energy
consumption, CodeCarbon recorded a mean of 1.40 ×
10−7 kWh, EnergyMeter logged 8.13 × 10−8 kWh, and
PyJoules reported 1.46 × 10−7 kWh. Notably, the process-
level measurements of MELODI showed the lowest aver-
age CPU energy at 6.40 × 10−11 kWh. For GPU, Code-
Carbon and EnergyMeter were closely aligned with means
around 2.57 × 10−7 kWh and 2.53 × 10−7 kWh, respectively.
MELODI reported a mean GPU energy consumption of
1.22 × 10−7 kWh, while PyJoules recorded a significantly
lower mean GPU energy at 2.81 × 10−10 kWh, possibly
reflecting differences in how idle states or sampling frequen-
cies impact measurements.

gemma_2bgemma_7bgemma2_2bllama3_8bllama3.2_1bphi_2bqwen2_7blaptop20.000000.000050.000100.00015energy_consumption_llm (kWh)gemma_2bgemma_7bgemma2_2bllama3_8bllama3.2_1bphi_2bqwen2_7blaptop2234energy_per_token (kWh)1e7gemma_2bgemma_7bgemma2_2bllama3_8bllama3.2_1bphi_2bqwen2_7blaptop2100200300response_token_length10

TABLE 8: Comparison of energy measurement tools for LLM deployments.

Criteria
CPU Tracking
GPU Tracking
Default sampling freq.

MELODI
RAPL (via Scaphandre)
NVIDIA-SMI (via NVML)
10 Hz

PyJoules
RAPL
NVML
Hardware-dependent

CodeCarbon
RAPL
pynvml (NVML)
1/15 Hz

EnergyMeter
RAPL (via pyRAPL)
NVIDIA-SMI (via NVML)
Not specified

The variability in energy measurements among different
tools suggests tool-specific biases or variations in interface
handling for CPU and GPU measurements. For CPU, the
per-process measurement approach of MELODI yielded
significantly lower energy readings, potentially isolating
energy consumption more effectively in the mixed-use envi-
ronments of laptops. In contrast, PyJoules and CodeCarbon,
which measure CPU energy at the system level, reported
higher averages, likely including additional background
activities that influence overall energy assessments.

The similar GPU energy consumption readings between
CodeCarbon and EnergyMeter suggest similar NVML-
based GPU data handling (Table 8). Conversely, MELODI re-
ported a slightly lower mean GPU energy consumption, in-
dicating a more restrictive approach to GPU monitoring and
potentially reflecting a more precise monitoring method.
Notably, the lower IQR in MELODI suggests more accurate
data collection, supported by its higher sampling frequency
of 10 Hz compared to the lower frequency of CodeCarbon
(Table 8). PyJoules, having dramatically lower readings and
outdated maintenance (its last update was in October 2021),
raises concerns about its measurement accuracy. The vari-
ations in GPU energy measurements across tools highlight
the potential impact of how each tool interacts with NVML-
based monitoring on the reliability of GPU data.

MELODI distinguishes itself by its capability to measure
CPU energy consumption at the per-process level, effec-
tively isolating model-specific usage more accurately than
whole-system methods. Its precision minimizes the interfer-
ence from background activities prevalent in multitasking
environments, enhancing its reliability.

RQ6 Conclusion. The analysis of energy measurement
tools for LLM deployments revealed significant differ-
ences in per-token energy consumption among MELODI,
PyJoules, CodeCarbon, and EnergyMeter. MELODI’s per-
process CPU monitoring yielded more precise and gen-
erally lower energy readings compared to whole-system
methods, pinpointing model-specific consumption. More-
over, the variability in GPU energy measurements, espe-
cially the abnormally low readings from PyJoules, sug-
gests discrepancies that could stem from how the tools
interact with NVIDIA’s GPU measurements, potentially
affected by sampling frequencies and recording intervals.

4.4 Threats to validity

Internal Validity. The accuracy of the power usage moni-
toring tools significantly impacts the internal validity of our
study. The nvidia-smi tool, according to NVIDIA documen-
tation [54], has an accuracy of ±5 Watts, though Yang et al.
suggest this error margin could be as high as ±5%. Ensuring
no concurrent GPU processes is crucial for accurate read-
ings, and while most experiments isolated the GPU, residual
base processes had minimal impact. Scaphandre’s reliance

Fig. 9: Comparison of the energy consumption per token, as
measured by four different monitoring frameworks.

on Intel’s RAPL interface for power reasings may also affect
data reliability. Variability in hardware power efficiency
could further skew results, underscoring the need for more
precise measurement tools and standardized testing across
different hardware configurations in future research.

External Validity. The generalizability of our findings is
limited by the scope of our inference scenarios, particularly
concerning the hardware setups and LLMs tested. Given the
sample size, drawing broad conclusions is challenging. The
ability to extend our results more widely hinges on whether
the hardware and model configurations used accurately
reflect the diversity in larger-scale LLM deployments.

5 DISCUSSION
Scope of Our Study. Our study concentrates on CPU and
GPU energy consumption for LLM inference, chosen for its
measurable impact. While significant, this focus omits other
energy contributors in data center environments like net-
working, memory management, orchestration, and cooling,
which can exceed computational energy use. This research
addresses only direct CPU and GPU power use, emphasiz-
ing the need for broader energy evaluations in future work.
Response Token Length as an Energy Indicator. Our re-
sults reveal that response token length is a reliable indicator
of energy consumption, suggesting that managing response
lengths could effectively control energy use. However, the
connection between prompt complexity and energy con-
sumption is less definitive. While models based on prompt
characteristics show potential, their performance varies, in-
dicating these correlations may not be strong enough for
reliable predictions in all contexts. We plan to explore the
potential of using LLMs or other NLP models to analyze
prompts for energy consumption prediction, balancing the
benefits against the potential energy costs of such analyses.
Exploring Response Quality and Energy Use. Our
current analysis does not address the quality of responses
generated. Our dataset, however, provides a rich foundation
to explore the relationship between response quality and en-
ergy consumption. Understanding whether higher-quality

codecarbonenergymetermelodipyjoulesphi_2.7b0.00.51.01.52.0energy_per_token (kWh)1e7CPUcodecarbonenergymetermelodipyjoulesphi_2.7b0123energy_per_token (kWh)1e7GPUresponses inherently require more energy could inform the
development of more efficient models and prompt designs.
Enhancing Generalizability and Model Comparisons.
To enhance our study’s generalizability, we plan to test a
wider array of hardware configurations and model types,
allowing us to better understand energy consumption varia-
tions. Comparing the energy efficiency of task-specific mod-
els against general-purpose models in different applications
would also be valuable. This analysis could reveal if spe-
cialized models yield significant energy savings for specific
tasks, ensuring that comparisons maintain a consistent base
model for accuracy.

Investigating Energy Consumption Patterns. Future re-
search could explore patterns in energy consumption across
models, focusing beyond mere size to include architectural
differences. This could reveal how specific design choices
impact energy demands and help optimize models for better
energy efficiency without sacrificing performance.

6 RELATED WORK
6.1 Energy and Carbon Footprint of AI and LLMs

The advancement of deep neural networks, particularly
in natural language processing (NLP), has intensified con-
cerns about their environmental impact. Strubell et al. [23]
quantified the energy cost of training large NLP models,
showing that a single training run can emit as much CO2 as
multiple passenger vehicles over their lifetimes. Subsequent
studies expanded this perspective: Luccioni et al. [55] an-
alyzed inference-time energy consumption for large trans-
former models such as BLOOM, while Luccioni et al. [56]
demonstrated that general-purpose generative models incur
orders-of-magnitude higher energy costs than task-specific
alternatives, even when controlling for model size.

Recent work has shifted attention from training to the
operational footprint of LLMs. De Vries [57] highlights the
substantial contribution of AI-driven data center operations
to global energy demand. Similarly, Jegham et al. [58] exam-
ined energy, water, and carbon metrics across state-of-the-art
LLM inference deployments, and Poddar et al. [59] iden-
tify strong correlations between inference energy, output
length, model size, and hardware configuration. Meanwhile,
Google Cloud’s analysis of their Gemini model estimate
median prompt energy usage at 0.24 W·h and CO2 emission
at 0.03 g, underscoring how inference, though individually
modest, scales massively [60].

Collectively, this body of work underscores a critical
insight: while training remains energy-intensive, inference
increasingly dominates the lifecycle energy and carbon
footprint of LLMs due to continuous, large-scale deploy-
ment. Despite this shift, most prior studies still emphasize
training or rely on coarse-grained measurements, motivat-
ing the need for precise, deployment-aware methodologies
to analyze inference-time energy consumption and support
sustainable AI engineering.

6.2 Energy Monitoring Frameworks for LLMs

Prior work has proposed several frameworks for monitor-
ing and analyzing the energy use and carbon footprint of
LLMs, revealing substantial variation across model config-
urations [21], [27]–[29], [33]. Wilkins et al. [28] introduce an

11

offline, workload-based framework for LLM inference on
heterogeneous CPU–GPU systems, modeling energy con-
sumption and runtime as functions of input and output
tokens. However, their reliance on AMDµProf entails no-
table limitations, including measurement overhead from fre-
quent polling, limited real-time granularity, and hardware
specificity to AMD CPUs, which restricts generalizability
and increases setup complexity. In contrast, Samsi et al. [27]
focus exclusively on GPU energy consumption for LLaMA
models, omitting CPU contributions.

Our work complements these efforts above by providing
a more comprehensive framework that measures energy
consumption across multiple LLMs, using Scaphandre and
nvidia-smi for real-time tracking of both CPU and GPU
energy usage. This broader approach enables a more holistic
understanding of energy dynamics during LLM inference.

Argerich et al. [33] present a related framework based
on pyRAPL [61], nvidia-smi [54], and eBPF [62] to esti-
mate CPU, memory, GPU, and storage energy on Linux,
supported by extensive analyses across many LLMs and
model sizes. In contrast, MELODI supports a broader range
of hardware configurations beyond a single bare-metal
environment and extends energy monitoring with predic-
tive analysis, providing a more flexible and comprehensive
framework for studying LLM inference energy consump-
tion. Henderson et al. [63] utilize RAPL in a similar manner
to Scaphandre to monitor energy use but derive per-process
energy consumption based on the CPU’s relative utilization.
Scaphandre, however, provides a more comprehensive ex-
planation of how it calculates per-process energy metrics.

6.3 Inference-Time Energy Profiling and Optimization

The growing deployment of LLMs in production highlights
the importance of inference-time energy profiling and opti-
mization. Samsi et al. [27] analyze inference energy costs for
different LLaMA model sizes on NVIDIA V100 and A100
GPUs, demonstrating how model scale, hardware, and task
characteristics influence energy consumption. Similarly, Fer-
nandez et al. [64] show that inference optimizations—such
as quantization, batching strategies, and hardware-aware
mapping—can reduce energy usage by up to 73% compared
to unoptimized deployments. Wilhelm et al. [65] intro-
duce the energy-per-token metric and demonstrate strong
correlations between response length, model size, and en-
ergy consumption, while also exploring dynamic reasoning
strategies to reduce energy without significant accuracy loss.
Stojkovic et al. [66] introduce a dynamic inference-cluster
management framework that exploits workload heterogene-
ity to optimize inference energy, achieving average energy
savings of 52% under service-level objectives.

Despite these advances, most existing approaches lack
fine-grained attribution of energy usage to specific pro-
cesses (e.g., model decoding) or integrations across di-
verse hardware and prompt workloads. This gap motivates
frameworks capable of capturing per-token, per-process,
and hardware-aware energy profiles for real-world LLM
deployment scenarios.

6.4 Hardware-Level Power Monitoring Techniques

Hardware-level power monitoring is fundamental for an-
alyzing and optimizing energy consumption in computing

12

systems. Intel’s RAPL (Running Average Power Limit) inter-
face is widely used, providing on-chip energy counters for
CPU and DRAM domains. Prior studies show that RAPL
measurements strongly correlate with external power me-
ters, although limitations remain due to driver support, up-
date timing, and domain-specific inaccuracies [67]. In par-
ticular, RAPL’s memory-domain estimates can overestimate
power consumption and exhibit temporal inconsistencies on
heterogeneous memory systems [68], highlighting the need
for careful interpretation of such measurements.

On the GPU side, power monitoring commonly uses
NVML (NVIDIA Management Library) and the nvidia-smi
interface. You et al. [69] show that GPU power limits
strongly affect the energy–performance trade-off of DNN
workloads. Another study by Yang et al.
[70] reveals
that nvidia-smi often samples only a portion of runtime,
leading to substantial under- or over-estimation of energy
use—sometimes by as much as 35–65%. Furthermore, Arafa
et al. [71] demonstrate how hardware counters and instru-
mentation can measure energy consumption at the instruc-
tion level on NVIDIA GPUs, reinforcing that architecture-
specific factors (e.g., micro-architecture generation, instruc-
tion type) influence energy behavior significantly.

In summary, hardware-level measurement techniques
provide essential visibility into component-level power us-
age (CPU, DRAM, GPU), but each comes with its own
limitations—granularity, accuracy, sampling frequency, and
applicability vary. For energy-aware systems especially in
the context of LLM deployment, selecting and validating
these techniques is critical before interpreting results.

6.5 Sustainable AI Engineering and Future Directions

Recent work on sustainable AI increasingly emphasizes
energy- and resource-aware design, deployment, and op-
eration of large models. A comprehensive survey on Green
AI [72] highlights the need to account for lifecycle impacts,
including carbon and water usage, when evaluating AI
systems. Another study on LLM sustainability [73] extend
this perspective by considering economic and resource costs
alongside computation, advocating greater transparency
and standardized benchmarking in AI energy and carbon
accounting. Fernandez et al. [64] show that inference-level
optimizations can reduce energy consumption by up to
73% with negligible impact on accuracy. Similarly, S´anchez-
Momp ´o et al. [74] empirically demonstrate that larger mod-
els do not necessarily incur higher energy per request,
particularly under low serving utilization.

Collectively, these studies underscore that sustainable AI
engineering demands a holistic perspective encompassing
model design, hardware selection, inference architecture,
measurement transparency, and deployment practices. They
point to key directions such as standardized efficiency
metrics, energy-aware architectures, cross-layer co-design,
and system-level optimizations. MELODI contributes to this
agenda by providing fine-grained inference-time energy
measurements supporting energy-conscious AI systems.

6.6 Research Gaps and Motivation for Our Framework

Although significant advances have been made in quanti-
fying the environmental footprint of artificial intelligence

systems, several research gaps remain—especially regard-
ing fine-grained, inference-time energy measurement of
LLMs. Current literature and tools leave important method-
ological, technical, and analytical challenges unresolved,
motivating the need for a more comprehensive framework.
(1) Lack of inference-level granularity. Most prior re-
search emphasizes training-phase emissions and provides
only coarse-grained reporting for inference. Existing tools
such as CodeCarbon, PyJoules, and EnergyMeter measure
energy at the system level, aggregating consumption across
all active processes. This approach obscures the specific
energy cost of the LLM process, especially in multi-tasking
environments, thereby limiting the interpretability of results
and their use for optimization.

(2) Hardware-dependent and non-standardized moni-
toring. Many tools rely on vendor-specific interfaces (e.g.,
Intel’s RAPL for CPUs and NVIDIA’s NVML for GPUs)
whose accuracy, update frequency, and calibration differ
across architectures. These inconsistencies introduce mea-
surement bias and hinder the reproducibility and com-
parability of studies. Moreover, most frameworks do not
jointly monitor CPU and GPU energy, overlooking cross-
component dependencies shaping overall power dynamics.
(3) Limited linkage between operational factors and
energy behavior. Empirical analyses rarely explore how
prompt features (e.g., length, complexity), model configu-
rations (e.g., size, architecture), or deployment hardware
collectively shape energy consumption. This lack of fine-
grained correlation hinders the development of targeted
optimization strategies for sustainable inference.

(4) Absence of unified, open, and process-level mea-
surement solutions. While recent works call for trans-
parency and standardized reporting of energy metrics, there
is no openly available framework that provides process-
level monitoring, configurable sampling, and reproducible
data collection across heterogeneous hardware setups.

Rationale. To address these challenges, we propose an
extensible framework for high-resolution monitoring and
analysis of LLM inference energy consumption. MELODI is
designed to (i) capture CPU and GPU energy at the process
level with configurable sampling intervals; (ii) log infer-
ence metadata, including token counts and timestamps; and
(iii) enable empirical studies linking energy use to model,
prompt, and hardware characteristics. By bridging mea-
surement accuracy, analytical depth, and reproducibility,
our framework advances the methodological foundations of
sustainable AI engineering, enabling more transparent and
data-driven optimization of LLM inference efficiency.

7 CONCLUSION

In conclusion, MELODI represents a noteworthy leap for-
ward in advancing energy-aware practices within the do-
main of LLMs. By formulating a precise methodology and
unveiling an open-source instrument tailored for real-time
monitoring of energy consumption throughout the LLM
inference process, MELODI effectively bridges a vital chasm
in sustainable computing. Our data collection, encompass-
ing various LLMs, hardware setups, and prompt datasets,
provides a robust platform for rigorous comparison and
nuanced analysis. This facilitates a deeper understanding

of the energy dynamics associated with LLM deployment
scenarios, offering critical insights that contribute to the
sustainability of LLMs.

ACKNOWLEDGMENT
The work has been conducted as part of the ENFIELD
project (101120657) funded by the European Commission
within the HEU Programme.

REFERENCES

[1] A. Vaswani, “Attention is all you need,” Advances in Neural Infor-

mation Processing Systems, 2017.

[2] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von
Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., “On
the opportunities and risks of foundation models,” arXiv preprint
arXiv:2108.07258, 2021.

[3] T. B. Brown, “Language models are few-shot learners,” arXiv

preprint arXiv:2005.14165, 2020.

[4] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell,
“On the dangers of stochastic parrots: Can language models be
too big?” in ACM FAccT’21, 2021, pp. 610–623.

[5] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al.,
“Language models are unsupervised multitask learners,” OpenAI
blog, vol. 1, no. 8, p. 9, 2019.
S. Sudhakaran, M. Gonz´alez-Duque, M. Freiberger, C. Glanois,
E. Najarro, and S. Risi, “Mariogpt: Open-ended text2level gen-
eration through large language models,” Advances in Neural Infor-
mation Processing Systems, vol. 36, 2024.

[6]

[8]

[7] X. Liang, H. Wang, Y. Wang, S. Song, J. Yang, S. Niu, J. Hu,
D. Liu, S. Yao, F. Xiong et al., “Controllable text generation for
large language models: A survey,” arXiv preprint arXiv:2408.12599,
2024.
J. Li, T. Tang, W. X. Zhao, J.-Y. Nie, and J.-R. Wen, “Pre-trained
language models for text generation: A survey,” ACM Computing
Surveys, vol. 56, no. 9, pp. 1–39, 2024.
J. Wulf and J. Meierhofer, “Exploring the potential of large lan-
guage models for automation in technical customer service,” arXiv
preprint arXiv:2405.09161, 2024.

[9]

[10] S. Kolasani, “Optimizing natural language processing, large lan-
guage models (llms) for efficient customer service, and hyper-
personalization to enable sustainable growth and revenue,” Trans-
actions on Latest Trends in Artificial Intelligence, vol. 4, no. 4, 2023.

[11] J. Chen, Z. Liu, X. Huang, C. Wu, Q. Liu, G. Jiang, Y. Pu, Y. Lei,
X. Chen, X. Wang et al., “When large language models meet
personalization: Perspectives of challenges and opportunities,”
World Wide Web, vol. 27, no. 4, p. 42, 2024.

[12] J. Li, H. Zhou, S. Huang, S. Cheng, and J. Chen, “Eliciting the
translation ability of large language models via multilingual fine-
tuning with translation instructions,” Transactions of the Association
for Computational Linguistics, vol. 12, pp. 576–592, 2024.

[13] B. Zhang, B. Haddow, and A. Birch, “Prompting large language
model for machine translation: A case study,” in International
Conference on Machine Learning. PMLR, 2023, pp. 41 092–41 110.

[14] L. Wang, C. Lyu, T. Ji, Z. Zhang, D. Yu, S. Shi, and Z. Tu,
“Document-level machine translation with large language mod-
els,” arXiv preprint arXiv:2304.02210, 2023.

[15] Q. Zhong, L. Ding, J. Liu, B. Du, and D. Tao, “Can chatgpt
understand too? a comparative study on chatgpt and fine-tuned
bert,” arXiv preprint arXiv:2302.10198, 2023.

[16] X. Zhu, S. Gardiner, T. Rold´an, and D. Rossouw, “The model arena
for cross-lingual sentiment analysis: A comparative study in the
era of large language models,” arXiv preprint arXiv:2406.19358,
2024.

[17] W. Zhang, Y. Deng, B. Liu, S. J. Pan, and L. Bing, “Sentiment
analysis in the era of large language models: A reality check,”
arXiv preprint arXiv:2305.15005, 2023.

[18] M. C. Rillig, M. ˚Agerstrand, M. Bi, K. A. Gould, and U. Sauerland,
“Risks and benefits of large language models for the environ-
ment,” Environmental Science & Technology, vol. 57, no. 9, pp. 3464–
3466, 2023.

[19] L. Weidinger, J. Uesato, M. Rauh, C. Griffin, P.-S. Huang, J. Mellor,
A. Glaese, M. Cheng, B. Balle, A. Kasirzadeh et al., “Taxonomy
of risks posed by language models,” in ACM FAccT’22, 2022, pp.
214–229.

13

[20] L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang,
M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh et al., “Ethical
and social risks of harm from language models,” arXiv preprint
arXiv:2112.04359, 2021.

[21] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga,
Y. Zhang, D. Narayanan, Y. Wu, A. Kumar et al., “Holistic evalua-
tion of language models,” arXiv preprint arXiv:2211.09110, 2022.
[22] E. J. Husom, A. Goknil, M. Astekin, L. K. Shar, A. K ˜A¥ sen,
S. Sen, B. A. Mithassel, and A. Soylu, “Sustainable llm inference
for edge ai: Evaluating quantized llms for energy efficiency, output
accuracy, and inference latency,” ACM Transactions on Internet of
Things, vol. 6, no. 4, pp. 1–35, 2025.

[23] E. Strubell, A. Ganesh, and A. McCallum, “Energy and pol-
icy considerations for deep learning in nlp,” arXiv preprint
arXiv:1906.02243, 2019.

[24] L. Lannelongue, J. Grealey, and M. Inouye, “Green algorithms:
quantifying the carbon footprint of computation,” Advanced sci-
ence, vol. 8, no. 12, p. 2100707, 2021.

[25] CodeCarbon, https://codecarbon.io/, Visited in 2024.
[26] E. J. Husom, S. Sen, and A. Goknil, “Engineering carbon emission-
aware machine learning pipelines,” in CAIN’24, 2024, pp. 118–128.
[27] S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones,
W. Bergeron, J. Kepner, D. Tiwari, and V. Gadepally, “From words
to watts: Benchmarking the energy costs of large language model
inference,” in IEEE HPEC’23, 2023, pp. 1–9.

[28] G. Wilkins, S. Keshav, and R. Mortier, “Offline energy-optimal
llm serving: Workload-based energy models for llm inference on
heterogeneous systems,” arXiv preprint arXiv:2407.04014, 2024.
[29] ——, “Hybrid heterogeneous clusters can lower the energy con-
sumption of llm inference workloads,” in ACM e-Energy’24, 2024,
pp. 506–513.

[30] A. A. Chien, L. Lin, H. Nguyen, V. Rao, T. Sharma, and R. Wi-
jayawardana, “Reducing the carbon impact of generative ai infer-
ence (today and in 2035),” in HotCarbon’23, 2023, pp. 1–7.

[31] J. Stojkovic, E. Choukse, C. Zhang, I. Goiri, and J. Torrellas,
“Towards greener llms: Bringing energy-efficiency to the forefront
of llm inference,” arXiv preprint arXiv:2403.20306, 2024.

[32] B. Everman, T. Villwock, D. Chen, N. Soto, O. Zhang, and Z. Zong,
“Evaluating the carbon impact of large language models at the
inference stage,” in IEEE IPCCC’23, 2023, pp. 150–157.

[33] M. F. Argerich and M. Pati ˜no-Mart´ınez, “Measuring and improv-
ing the energy efficiency of large language models inference,”
IEEE Access, vol. 12, pp. 80 194–80 207, 2024.

[34] Scaphandre, https://github.com/hubblo-org/scaphandre.
[35] NVIDIA,

https://developer.nvidia.com/nvidia-system-

management-interface, Visited in 2024.

[36] Alpaca, https://huggingface.co/datasets/tatsu-lab/alpaca.
[37] Code-Feedback, https://huggingface.co/datasets/m-a-p/Code-

Feedback, Visited in 2026.

[38] NVIDIA,

https://developer.nvidia.com/management-library-

nvml, Visited in 2024.

[39] M. Jay, V. Ostapenco, L. Lef`evre, D. Trystram, A.-C. Orgerie, and
B. Fichel, “An experimental comparison of software-based power
meters: focus on cpu and gpu,” in IEEE/ACM CCGRID’23, 2023,
pp. 106–118.

[40] Prometheus, https://prometheus.io/docs/instrumenting/exporters/.
[41] Llama.cpp, https://github.com/ggerganov/llama.cpp.
[42] Y. Anand, Z. Nussbaum, A. Treat, A. Miller, R. Guo, B. Schmidt,
G. Community, B. Duderstadt, and A. Mulyar, “Gpt4all: An
ecosystem of open source compressed language models,” arXiv
preprint arXiv:2311.04931, 2023.

[43] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gon-
zalez, H. Zhang, and I. Stoica, “Efficient memory management for
large language model serving with pagedattention,” in SOSP’23,
2023, pp. 611–626.

[44] Llamafile, https://github.com/Mozilla-Ocho/llamafile.
[45] Ollama, https://github.com/ollama/ollama, Visited in 2026.
[46] Z. Yang, K. Adamek, and W. Armour, “Part-time power measure-

ments: nvidia-smi’s lack of attention,” 2024.

[47] T. Zheng, G. Zhang, T. Shen, X. Liu, B. Y. Lin, J. Fu, W. Chen, and
X. Yue, “Opencodeinterpreter: Integrating code generation with
execution and refinement,” https://arxiv.org/abs/2402.14658, 2024.

[48] M. Honnibal and I. Montani, “spaCy 2: Natural language under-
standing with Bloom embeddings, convolutional neural networks
and incremental parsing,” 2017.

14

[49] S. Bird, E. Klein, and E. Loper, Natural language processing with
” O’Reilly

Python: analyzing text with the natural language toolkit.
Media, Inc.”, 2009.

[50] S. Loria, “textblob documentation,” Release 0.15, vol. 2, 2018.
[51] textstat, https://github.com/textstat/textstat, Visited in 2024.
[52] PowerAPI, “Pyjoules: Python-based energy measurement library,”
https://github.com/powerapi-ng/pyJoules, Visited in 2024.
[53] phi 2, https://huggingface.co/microsoft/phi-2, Visited in 2025.
[54] NVIDIA

Management

Interface,

System

https://docs.nvidia.com/deploy/nvidia-smi/index.html.

[55] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, “Estimating the
carbon footprint of bloom, a 176b parameter language model,”
Journal of machine learning research, vol. 24, no. 253, pp. 1–15, 2023.
[56] S. Luccioni, Y. Jernite, and E. Strubell, “Power hungry processing:
Watts driving the cost of ai deployment?” in ACM FAccT’24, 2024,
pp. 85–99.

[57] A. De Vries, “The growing energy footprint of artificial intelli-

gence,” Joule, vol. 7, no. 10, pp. 2191–2194, 2023.

[58] N. Jegham, M. Abdelatti, L. Elmoubarki, and A. Hendawi, “How
hungry is ai? benchmarking energy, water, and carbon footprint of
llm inference,” arXiv preprint arXiv:2505.09598, 2025.

[59] S. Poddar, P. Koley, J. Misra, S. Podder, N. Ganguly, and S. Ghosh,
“Towards sustainable nlp: Insights from benchmarking inference
energy in large language models,” arXiv preprint arXiv:2502.05610,
2025.

[60] Google, https://cloud.google.com/blog/products/infrastructure/measuring-

the-environmental-impact-of-ai-inference/, Visited in 2025.
[61] pyRAPL, https://pypi.org/project/pyRAPL/, Visited in 2026.
[62] eBPF, https://ebpf.io/, Visited in 2026.
[63] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, and
J. Pineau, “Towards the systematic reporting of the energy and
carbon footprints of machine learning,” Journal of Machine Learning
Research, vol. 21, no. 248, pp. 1–43, 2020.

[64] J. Fernandez, C. Na, V. Tiwari, Y. Bisk, S. Luccioni, and E. Strubell,
“Energy considerations of large language model inference and
efficiency optimizations,” arXiv preprint arXiv:2504.17674, 2025.
[65] P. Wilhelm, T. Wittkopp, and O. Kao, “Beyond test-time compute
strategies: Advocating energy-per-token in llm inference,” in Eu-
roMLSys’25, 2025, pp. 208–215.

[66] J. Stojkovic, C. Zhang, ´I. Goiri, J. Torrellas, and E. Choukse,
“Dynamollm: Designing llm inference clusters for performance
and energy efficiency,” in IEEE HPCA’25, 2025, pp. 1348–1362.
[67] K. N. Khan, M. Hirki, T. Niemi, J. K. Nurminen, and Z. Ou, “Rapl
in action: Experiences in using rapl for power measurements,”
ACM Transactions on Modeling and Performance Evaluation of Com-
puting Systems, vol. 3, no. 2, pp. 1–26, 2018.

[68] L. Alt, A. Kozhokanova, T. Ilsche, C. Terboven, and M. S. Mueller,
“An experimental setup to evaluate rapl energy counters for
heterogeneous memory,” in ICPE’24, 2024, pp. 71–82.

[69] J. You, J.-W. Chung, and M. Chowdhury, “Zeus: Understanding
and optimizing {GPU} energy consumption of {DNN} training,”
in NSDI’23, 2023, pp. 119–139.

[70] Z. Yang, K. Adamek, and W. Armour, “Part-time power
measurements: nvidia-smi’s lack of attention,” arXiv preprint
arXiv:2312.02741, 2023.

[71] Y. Arafa, A. ElWazir, A. ElKanishy, Y. Aly, A. Elsayed, A.-H.
Badawy, G. Chennupati, S. Eidenbenz, and N. Santhi, “Verified
instruction-level energy consumption measurement for nvidia
gpus,” in ACM CF’20, 2020, pp. 60–70.

[72] V. Bol ´on-Canedo, L. Mor´an-Fern´andez, B. Cancela, and A. Alonso-
Betanzos, “A review of green artificial intelligence: Towards a
more sustainable future,” Neurocomputing, vol. 599, p. 128096,
2024.

[73] A. Singh, N. P. Patel, A. Ehtesham, S. Kumar, and T. T. Khoei, “A
survey of sustainability in large language models: Applications,
economics, and challenges,” in IEEE CCWC’25, 2025, pp. 00 008–
00 014.

[74] A. S´anchez-Momp ´o,

I. Mavromatis, P. Li, K. Katsaros, and
A. Khan, “Green mlops to green genops: An empirical study of
energy consumption in discriminative and generative ai opera-
tions,” Information, vol. 16, no. 4, p. 281, 2025.

Erik Johannes Husom is a Research Scientist
in the Trustworthy Green IoT Software research
group at SINTEF Digital, 0373 Oslo, Norway. His
research interests include applied ML, AI engi-
neering, and responsible AI. Husom received his
M.Sc. in computational physics from the Univer-
sity of Oslo.

Arda Goknil received his Ph.D. degree in com-
puter science from the University of Twente, in
the Netherlands, in 2011. He is a senior research
scientist at SINTEF, Norway. He was a research
associate at the University of Luxembourg. His
research concerns AI Engineering, Sustainable
AI, Software Testing, Software Security, and In-
termittent Computing. He is active on EU-funded
and national research projects with several aca-
demic and industry partners.

Lwin Khin Shar (Member,
IEEE) received a
Ph.D. degree in software engineering from
Nanyang Technological University (NTU), Singa-
pore, in 2014. He is an Associate Professor at
the School of Computing and Information Sys-
tems, Singapore Management University, Singa-
pore. He was a Postdoctoral Research Asso-
ciate with SnT of the University of Luxembourg,
Esch-sur-Alzette, Luxembourg, and then a Re-
search Scientist with NTU. His research interests
include software engineering, security & privacy,

and machine learning.

Sagar Sen is a senior research scientist at SIN-
TEF Digital, 0373 Oslo, Norway. His research
interests are in engineering and testing of life-
long AI systems for application domains such
as manufacturing and health. Sen received his
Ph.D. in computer science from the University of
Rennes 1/INRIA. He is a Member of IEEE.

