2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA)

MLPerf Power: Benchmarking the Energy
Efficiency of Machine Learning Systems from
µWatts to MWatts for Sustainable AI

Arya Tschand1∗ Arun Tejusve Raghunath Rajan2∗ Sachin Idgunji3∗ Anirban Ghosh3 Jeremy
Holleman4 Csaba Kiraly5 Pawan Ambalkar6 Ritika Borkar3 Ramesh Chukka7 Trevor Cockrell6 Oliver
Curtis8 Grigori Fursin9 Miro Hodak10 Hiwot Kassa2 Anton Lokhmotov11 Dejan Miskovic3 Yuechao
Pan12 Manu Prasad Manmathan7 Liz Raymond6 Tom St. John13 Arjun Suresh14 Rowan Taubitz8 Sean
Zhan8 Scott Wasson15 David Kanter15 Vijay Janapa Reddi1

∗Equal contribution 1Harvard University 2Meta 3NVIDIA 4UNC Charlotte / Syntiant 5Codex 6Dell 7Intel
8SMC 9FlexAI / cTuning 10AMD 11KRAI 12Google 13Decompute 14GATE Overflow 15MLCommons

Abstract—Rapid adoption of machine learning (ML) technolo-
gies has led to a surge in power consumption across diverse
systems, from tiny IoT devices to massive datacenter clusters.
Benchmarking the energy efficiency of these systems is crucial for
optimization, but presents novel challenges due to the variety of
hardware platforms, workload characteristics, and system-level
interactions. This paper introduces MLPerf® Power, a compre-
hensive benchmarking methodology with capabilities to evaluate
the energy efficiency of ML systems at power levels ranging from
microwatts to megawatts. Developed by a consortium of industry
professionals from more than 20 organizations, coupled with
insights from academia, MLPerf Power establishes rules and best
practices to ensure comparability across diverse architectures.
We use representative workloads from the MLPerf benchmark
suite to collect 1,841 reproducible measurements from 60 systems
across the entire range of ML deployment scales. Our analysis
reveals trade-offs between performance, complexity, and energy
efficiency across this wide range of systems, providing actionable
insights for designing optimized ML solutions from the smallest
edge devices to the largest cloud infrastructures. This work
emphasizes the importance of energy efficiency as a key metric
in the evaluation and comparison of the ML system, laying the
foundation for future research in this critical area. We discuss
the implications for developing sustainable AI solutions and
standardizing energy efficiency benchmarking for ML systems.

I. INTRODUCTION

In recent years, machine learning (ML) technologies have
transformed a wide range of fields, from high-performance
computing centers to edge devices and small IoT systems.
The exceptional improvement in ML performance is clearly
demonstrated by the MLPerf Training benchmark results [46],
as shown in Figure 1. Since the debut of this benchmark in
2018, performance has soared, showing an incredible 32-fold
increase. This rapid pace of advancement is not limited to
established benchmarks; newly introduced benchmarks also
show consistent performance enhancements. Such exponential
growth has led us to surpass Moore’s law, which highlights
the extraordinary pace of innovation in ML systems driven by
hardware, algorithm, and software enhancements.

The advancement in ML capabilities has coincided with a
notable increase in concerns about the power consumption of

Fig. 1: MLPerf performance improvements have outpaced
Moore’s Law. This trend highlights the rapid evolution of
AI systems, prompting the development of MLPerf Power to
address emerging concerns over their energy efficiency.

ML systems [11], [65], [75], [76], [78]. As the ecological
impact and operational expenses of these systems continue
to increase, they have become a significant concern for both
industry and academia [15], [72], [80], [82]. The critical
nature of this challenge highlights the need for a standardized
and thorough benchmarking approach to measure the energy
efficiency of ML systems, from small IoT devices to large-
scale cloud infrastructures. This method must also guarantee
accurate reproducibility, facilitate meaningful comparisons,
and encourage advances in energy-efficient ML technologies.
However, measuring and comparing the energy efficiency
of ML systems presents unique challenges that differentiate
them from traditional computing systems. Unlike conven-
tional general-purpose workloads, ML systems span various
hardware platforms, from specialized accelerators to general-
purpose processors [44], each with their own power con-
sumption characteristics [28], [41]. Moreover, ML workloads
exhibit diverse computational patterns and resource require-
ments, such as high data parallelism, memory intensity, and
communication-bound operations, which differ significantly

5
2
0
2

b
e
F
6

]

R
A
.
s
c
[

2
v
2
3
0
2
1
.
0
1
4
2
:
v
i
X
r
a

from traditional CPU-centric applications. The complexity is
further compounded by system-level interactions, such as data
movement between memory hierarchies and communication
overheads in distributed systems, which have a more profound
impact on ML systems’ energy efficiency than traditional
setups. These characteristics emphasize the need for a compre-
hensive benchmarking methodology that can accurately cap-
ture and analyze energy efficiency across different hardware
configurations, workload types, and system scales.

In addition,

the wide spectrum of ML systems, rang-
ing from IoT devices to expansive data centers, presents
a formidable challenge in developing a universally applica-
ble power measurement methodology. This diversity requires
a scalable approach capable of accurately capturing power
consumption characteristics across heterogeneous domains.
Although the fundamental principles of power measurement
remain consistent, the specific techniques and considerations
vary significantly depending on the scale and nature of the ML
system under evaluation, which we elaborate in this paper.

These diverse requirements emphasize the need for a com-
prehensive and adaptable power measurement methodology.
To address the multifaceted challenges of measuring and
comparing the energy efficiency of ML systems, we developed
MLPerf Power, a comprehensive benchmarking methodology.
This initiative is the result of collaborative efforts among
20 organizations actively participating in the MLPerf Train-
ing [46] and Inference [70] benchmarks, which have long
served as the industry standard for measuring ML system per-
formance. MLPerf Power extends this framework, recognizing
that performance metrics alone are insufficient to enable the
development of sustainable AI systems for the future.

MLPerf Power represents a concerted industry-wide effort
to establish a standardized approach for evaluating the power
consumption of ML systems across an unprecedented range
of energy levels, from microwatts in resource-constrained IoT
devices to megawatts in high-performance computing clusters.
Using the collective expertise and industry experience of
its contributors, MLPerf Power defines a consistent set of
benchmarking rules, measurement techniques, and reporting
guidelines. This framework enables accurate and fair com-
parisons between various ML systems, irrespective of their
hardware configurations or workload characteristics.

We provide a detailed overview of the MLPerf Power
methodology and its application across various ML systems.
The methodology is designed to account for the unique aspects
of ML workloads, including power consumption implications
of data preparation, model
training, and inference phases.
MLPerf Power places a strong emphasis on transparency
and reproducibility, mandating detailed disclosure of system
specifications and power measurement setups. We describe the
MLPerf Power methodology’s foundational design principles
and key components, exploring the benchmarking rules that
establish a level playing field for diverse ML systems, the
measurement techniques ensuring accurate power consump-
tion data across different scales, and the reporting guidelines
promoting transparency and enabling meaningful comparisons.

The application of the MLPerf Power methodology to a
diverse range of ML systems has yielded powerful insights
into energy efficiency trends and optimization strategies. Our
comprehensive analysis reveals both promising advances and
emerging challenges in the search for more sustainable AI
solutions. Energy efficiency trends across MLPerf versions
show a general positive trajectory, but exhibit recent plateaus
in improvements for certain workloads and scales. In partic-
ular, older and better optimized workloads such as ResNet
and RNN-T have reached a point of diminishing returns in
recent versions. In contrast, newer workloads like GPT-J
and Llama2 demonstrate rapid improvements, with energy
efficiency improving over 100× in about a year between
versions, driven by intense commercial
interest and high-
impact optimizations.

Our study of datacenter-scale training submissions for mod-
els such as Llama2-70b highlights the complex trade-offs
between system scale, energy efficiency, and performance. As
the number of accelerators scales by a factor of 64×, we
observe that energy consumption increases only 3.8×, while
training time plummets by 93%. This nonlinear relationship
underscores the importance of considering both energy and
time when evaluating large-scale ML systems. Furthermore,
these high-level
to provide actionable insights, we dissect
energy efficiency improvements, isolating the contributions
of various optimization strategies. Through a case study on
progressive versions of neural network ASICs with compa-
rable software stacks, we demonstrate that while performance
improvements plateau at around 1.5×, significant reductions in
power consumption enable a remarkable 4× improvement in
overall energy efficiency. In contrast, our analysis of software
isolated improvements in comparable hardware submissions
reveals that negligible performance sacrifices (less than 1%)
can yield substantial energy efficiency gains of up to 28%.
Furthermore, we illustrate how more aggressive quantization
techniques, when carefully applied to maintain benchmark
accuracy targets, can increase energy efficiency by up to 70%.
As ML continues to evolve and expand, the importance of
energy efficiency in ML systems cannot be overstated. As
newer and more complex generative models are consuming
orders of magnitude more energy per inference, our findings
emphasize the importance of considering energy efficiency as
a first-class metric in system design and optimization. The
insights and best practices presented in this work will help
guide the design, deployment, and optimization of energy-
efficient ML systems across various domains, from edge
devices to cloud-scale infrastructures.

In summary, this paper makes the following contributions:
• Standardization of power measurement for ML sys-
tems across the industry: MLPerf Power introduces a
comprehensive, industry-wide benchmarking methodol-
ogy for measuring power consumption in ML systems.
This enables fair comparisons across diverse hardware
platforms and workloads, from tiny IoT devices to large
datacenter clusters. MLPerf Power measures the true
power consumption during standardized workloads to

track improvements in ML hardware and software opti-
mizations, and promote transparency across the industry.
• First large-scale study with practical guidance on
energy efficiency in ML systems: We present a com-
prehensive analysis of over 1,800 energy measurement
results across diverse ML systems in production. We also
evaluate the real-world impact of various optimization
techniques, including hardware improvements, software
optimizations, and quantization. This dual approach offers
insights into the current state of energy efficiency in ML
and provides actionable guidance for practitioners and re-
searchers to make informed decisions about optimization
strategies for their specific use cases.

• Industry-wide analysis of ML system energy efficiency
trends: The paper presents a longitudinal study of energy
efficiency improvements across multiple generations of
MLPerf Power submissions from various industry play-
ers. This analysis provides insights into the state of
energy efficiency in commercial ML systems, highlight-
ing areas of progress and identifying opportunities for
further optimization. It offers a unique perspective on
how different sectors of the industry (datacenter, edge,
tiny) are addressing energy efficiency challenges.

II. BACKGROUND, MOTIVATION AND PRIOR WORK

This section establishes the requirements for benchmarking
and measuring the energy efficiency of ML systems and
demonstrates how our approach advances the field. We outline
the key requirements and then gives an overview of the limi-
tations of existing methods using Table I. MLPerf Power was
specifically designed to meet all the identified requirements.
Power Measurements

Req 1. Compatible with a wide range of platforms and config-
urations (and their power consumption characteristics).
Req 2. Account for system-level interactions and shared re-
sources that impact ML (i.e., system energy efficiency,
heterogeneous systems accelerator + host + etc.).

Req 3. Adapt measurements to capture power consumption
characteristics across different scales of ML systems.

Req 4. Physical power measurements with an analyzer.

Diverse ML Benchmarking

Req 5. Consistent benchmarking rules and reporting guide-

lines (for accurate and fair comparisons).

Req 6. Representative workloads from the industry-standard

MLPerf Benchmarking Suite (tiny to datacenter).

Req 7. Collect data on power consumption and performance
metrics (e.g., throughput, latency) at different through-
puts and ML model accuracy/quality targets.

Evaluation, Trends, and Impact

Req 8. Communicate power consumption analysis and opti-
mizations for sustainable AI (patterns and bottlenecks
differ significantly across scales of ML systems).

Req 9. Be driven and audited by the industry, which fosters
fair and representative performance evaluations.

Req 10. Communicate energy efficiency trends

(observed
across a wide spectrum of ML platform scales, power
levels, workload characteristics, hardware configura-
tions, system-level optimizations).

Req 11. Case studies showcasing the application of MLPerf
Power to real-world scenarios on production-grade
systems (demonstrate its effectiveness in identifying
energy efficiency bottlenecks and guiding the devel-
opment of more sustainable AI solutions).

MLPerf Power addresses all necessary requirements, com-
bining the strengths of innovative academic approaches with
industry-standard best practices. This sets a new benchmark
for measuring energy efficiency in machine learning systems.

III. ML SYSTEM POWER MEASUREMENT CHALLENGES

The goal of MLPerf is to be a system level performance
benchmark for machine learning (ML) systems. Therefore, our
goal while developing the power measurement methodologies
for MLPerf was to develop a full system power measurement
methodology that is applicable across the diverse hardware
platforms that MLPerf supports and scalable across different
MLPerf categories that target different system scales. This
section highlights the challenges encountered while developing
the MLPerf power measurement methodology.

A. Scale of ML Systems

MLPerf Power is designed to stretch from microwatts to
megawatts. Figure 2 samples from the official MLPerf Power
submission results to showcase this diverse range of systems,
which poses a challenge in creating a unified approach to mea-
suring full system power. MLPerf-Tiny systems operate with a
power consumption as low as 5.64 mW. They typically process
incoming (sensor) data before entering a low-power standby
mode until the next data frame arrives, resulting in duty cycles
that frequently fall below 5%. Consequently, even systems
with milliwatt-level peak power consumption often draw just
microwatts in real-world average power consumption. At the
other end of the scale, explicit power submissions for MLPerf-
Training can reach up to 500 kW [46]. However, in large-
scale submissions that currently only submit performance
measurements, we estimate power consumption around 10
MW in MLPerf-Training and 30 MW in MLPerf-HPC.

To address this diversity in ML applications and bench-
marks, it is essential to tailor methodologies for three distinct
scales while measuring system power consumption as com-
prehensively as possible for each scale.

Datacenter Training / HPC To measure the full power of
a system, we must account for all components that contribute
to training performance. For multi-node training, this includes
the compute nodes, interconnect fabric, and any cooling infras-
tructure. Due to the scale (10K+ GPUs) and nature (secure
HPC, on-premise or cloud datacenters) of these systems, it
is impractical to use conventional power meters to measure
system power. Instead, we use a combination of hardware
counters for compute and estimation for switching subsystems.

TABLE I: Related work. We identify several key requirements that are necessary for an industry-standard ML power benchmark.

Category

Paper

Benchmark
Benchmark
Benchmark
Measurement
Measurement
Measurement
Measurement
Measurement
Optimization
Trends
Trends
All of the above

SPEC Power [40]
Green500 [17]
Top500 [14]
Carbon [64]
Systematic [25]
Carbontracker [4]
AI Training [81]
USB [43]
Zeus [86]
Trends DL [12]
Survey [71]
MLPerf Power

Power Measurement

Req. 1
✓
✓
✓
X
✓
X
✓
X
✓
-
-
✓

Req. 2
X
X
X
X
✓
X
✓
X
X
-
-
✓

Req. 3
X
X
X
✓
X
✓
X
✓
X
-
-
✓

Req. 4
✓
✓
✓
X
X
X
X
✓
X
-
-
✓

Diverse ML Benchmarking
Req. 7
Req. 6
Req. 5
-
-
-
-
-
-
-
-
-
✓
✓
✓
✓
X
X
✓
X
X
✓
X
X
✓
✓
✓
✓
X
X
-
-
-
-
-
-
✓
✓
✓

Real-World Evaluation, Trends, Impact
Req. 11
Req. 8
-
-
-
-
-
-
✓
✓
✓
X
✓
X
X
X
X
X
✓
X
✓
X
X
X
✓
✓

Req. 10
-
-
-
X
X
X
X
X
X
✓
✓
✓

Req. 9
-
-
-
X
X
X
X
X
X
X
X
✓

e.g. WiFi transceivers, on-board sensors, idle IO interfaces,
etc. Our approach for edge systems involves using external
power analyzers, carefully calibrated to capture the full range
of power consumption without introducing measurement error.
Mobile MLPerf Mobile benchmarks the performance of
mobile ML systems [32]. However, we currently do not sup-
port the measurement of power of mobile systems in MLPerf
Power due to the complexity and variability of smartphones.
Mobile devices comprise a wide range of components, includ-
ing communication interfaces, sensors, and peripherals, each
with significant fluctuations in power consumption patterns
and active components [60]. Additionally, the battery-powered
nature of these devices means they cannot be simply connected
to an external power source for measurement without altering
normal operation. Thermal throttling further complicates the
relationship between power consumption and performance
metrics [66]. We have piloted efforts using DC-based low-
power analyzers, but we have not reached the point of stan-
dardization. Given these complexities, establishing a fair and
transparent benchmark for measuring power consumption in
battery-powered devices is a future goal for MLPerf Power.

Tiny Embedded ML systems relying on microcontrollers
often function as detectors, recognizing human presence in a
room via camera input or identifying spoken wake words (e.g.,
“Alexa” or “Hey Siri”). Compared to datacenter or edge, they
require a different measurement paradigm. The challenge here
lies in capturing extremely low power consumption accurately
without the measurement setup itself influencing the results.
We need specialized micro-power instrumentation and metic-
ulous setup to capture milliwatt-level fluctuations accurately.
We also need to implement strategies to prevent parasitic
powering, where the system under test might inadvertently
draw power from measurement or communication lines.

B. Heterogeneity and Comparability in Power Measurements

Heterogeneous Hardware Handling diverse hardware plat-
forms presents another layer of complexity. MLPerf submis-
sions span a wide range of architectures, from general-purpose
CPUs to specialized AI accelerators, each with unique power
consumption characteristics. To ensure fair comparability, we
must develop a flexible methodology that can adapt to different
hardware configurations while maintaining consistent mea-
surement principles. This includes guidelines for measuring

Fig. 2: The power consumption range across MLPerf divisions,
highlighting the need for scalable power measurement.

MLPerf supports a large diversity of submitters, each with
their own telemetry and monitoring setup. However, mea-
surement of interconnect power is challenging, as switches
may not have as much telemetry to report power and cloud
service providers often do not want to reveal their network
topology to competitors. To address this, we need to develop
a software-based approach that leverages estimation methods
and existing telemetry systems, allowing for comprehensive
power measurement without compromising security or requir-
ing extensive hardware modifications.

Measuring cooling power fairly is another challenge. Air-
cooled servers have fans whose power is measured when
measuring node power, but immersion or direct liquid cooled
systems pay their cooling cost at the datacenter level, making
it difficult to attribute what fraction was used by the ML
job under test. Measuring power consumption from cooling
remains future work.

Edge Edge and embedded systems have their own unique
challenges when measuring power. For MLPerf-Inference, the
Edge devices range from SoCs rated for tens of watts to
edge servers that can draw several kilowatts. The inference
methodology requires the use of a SPEC-certified power meter
such as the Yokogawa WT310. When measuring AC wall
power on very low power devices, we noticed that
these
power meters tend to have a higher error bar due to the high
crest factor caused by power adapters under 75W. Measuring
full system power for low power devices may also lead to
measuring power of sub-systems that cannot be powered off
on the system but do not contribute to ML performance –

full system power, accounting for both compute and auxiliary
components, and standardized reporting formats that capture
hardware-specific details relevant to power consumption.

Comparability Maintaining comparability between sub-
missions is crucial
to the integrity of the MLPerf Power
benchmarks. To address this challenge, we employ a multi-
dimensional strategy. First, we have defined stringent criteria
for valid power measurements, such as measurement duration,
sampling frequencies, and accuracy standards. Second, we
have adopted a uniform logging format to ensure power data
from various systems can be consistently analyzed. Third, we
mandate comprehensive documentation of measurement con-
figurations and any estimation methods employed, facilitating
thorough peer evaluation of submissions.

C. Myths and Pitfalls in ML System Power Measurement

Before diving into our methodology, we want to address
prevalent misconceptions and potential pitfalls in ML system
power measurement that contribute to its inherent complexity.
Myth #1: Isolating ML Component Power is Sufficient
A common misconception is that measuring the power con-
sumption of specific ML components, such as accelerators
or GPUs, is adequate to assess system efficiency. In reality,
overall system power consumption is crucial. Different com-
ponents are active at various stages of ML workloads with
varying duty cycles. For example, during training, accelerators
might be active during forward and backward propagation,
networking during gradient exchange, and CPUs during data
loading. The isolation of component power fails to capture
this dynamic interplay, leading to inaccurate representations
of energy usage. System-level measurement provides a more
comprehensive and realistic view of power consumption.

Myth #2: TDP and PSU Ratings Reflect Power Usage
Another prevalent myth is the reliance on Thermal Design
Power (TDP) or Power Supply Unit (PSU) ratings as proxies
for power measurement. In reality, TDP only represents the
thermal design limit, not the actual power usage in typical
workloads. Similarly, PSU ratings include significant mar-
gins for power spikes and redundancy, especially in compute
servers. These metrics often grossly overestimate actual power
consumption, making them poor approximations for real-
world usage. Accurate power measurement requires direct
monitoring of power consumption during actual ML tasks,
rather than relying on these theoretical maximum values.

Myth #3: PUE is Suitable for ML System Efficiency
The use of Power Usage Effectiveness (PUE) to evaluate ML
system energy efficiency is a misguided approach. While PUE
is crucial for data center efficiency, it is not appropriate for
MLPerf methodology for two key reasons. First, PUE reflects
datacenter efficiency, not ML system efficiency. Second, ver-
ifying PUE claims across diverse submitters is impractical,
potentially skewing comparisons between ML systems. Fo-
cusing on PUE could mask poor system efficiency with good
datacenter efficiency, or vice versa, detracting from MLPerf’s
goal of tracking ML system improvements. MLPerf’s scope

is to track improvements in ML system efficiency; building or
infrastructure efficiency is beyond its purview.

Myth #4: Precise Power Measurement is Always Possible
There is a common assumption that high-granularity power
measurements are universally achievable, which is not always
the case. Power measurement granularity is often limited by
available infrastructure. Some facilities only measure at the
Power Distribution Unit (PDU) level, encompassing multi-
ple nodes, making it challenging to isolate the exact power
demands of a specific ML system. This is particularly prob-
lematic in cloud environments or shared computing facilities
where dynamic job allocation can lead to measurements in-
cluding idle or unrelated nodes. Additionally, shared cooling
resources can make it difficult to isolate the power consump-
tion of specific components. These limitations can result in
inaccurate or overly coarse power consumption data, especially
in cloud or shared computing environments, highlighting the
need for careful consideration of measurement methodologies.

IV. MLPERF POWER MEASUREMENT METHODOLOGY

The methodology is designed to provide a comprehensive
and standardized approach to measuring power consumption in
ML systems across a wide range of scales, from microwatts
to megawatts. This section outlines the core concepts, met-
rics, and philosophy behind the MLPerf power measurement
approach.

A. Common Principles Across All Segments

MLPerf Power is driven by a core set of engineering
principles, whether we are dealing with tiny IoT devices or
massive datacenters. These principles help ensure that our
power measurements are reliable, comparable, and relevant.
By adhering to five principles, we are able to provide accurate
and meaningful data across a diverse range of systems.

First, a fundamental tenet of MLPerf Power is the emphasis
on measuring full system power consumption. This approach
recognizes that ML workloads involve various components
beyond the primary computing units. In datacenter systems,
we account for compute nodes and interconnect fabric. As
of today, we do not yet a methodology for including the
power for the cooling devices and the storage nodes, but
any other drives within the compute blades are accounted
for. For edge devices, we consider the entire system-on-chip,
including any peripherals that cannot be powered down during
operation. In tiny systems, we measure the power of the
entire device, including any always-on components. Figure 3
illustrates our measurement for the different scales of systems.
Our measurements closely reflect
the true energy cost of
running ML workloads in real-world scenarios, rather than
focusing solely on idealized component-level efficiencies.

Second, ML workloads typically consist of multiple phases:
initialization, execution, and result assimilation. The MLPerf
Power methodology emphasizes the precision of aligning
power measurements with these phases, particularly focusing
on the execution phase. This alignment is crucial because
it allows for accurate attribution of power consumption to

Fig. 3: ML system components within the MLPerf Power measurement scope are outlined in green.

specific parts of the workload, ensures comparability across
different systems, and provides insight into the energy ef-
ficiency of the core ML computations. To achieve this in
tiny systems, we use hardware pins to signal the start and
end of inference. In larger systems, we use software-based
timestamps to demarcate the execution phase. For training
workloads, we parse performance logs to determine the start
and stop times of the run.

Third, unlike evaluating overall system performance, the
execution phase is more important than end-to-end runtime
in power measurement considerations. The execution phase
represents the core ML computations, which are typically the
most energy-intensive part of the workload. It also provides
the most relevant power measurement data to draw actionable
insights to improve the energy efficiency of ML systems where
it matters most. For inference tasks we measure the power
during the actual inference computations. For training tasks,
we focus on the power consumed during the training iterations
alone. Any of the preceding phases including data assimilation,
data setup, offline and online pre-processing steps, are not
accounted for. This approach, combined with full system
power measurement (as explained in our first principle) and
precise alignment with workload phases, forms the foundation
of MLPerf Power’s methodology across all segments, from the
smallest IoT devices to the largest datacenter installations.

Fourth, we ensure comprehensive data collection by im-
plementing a minimum measurement duration. If a workload
finishes executing before the 60-second mark, it is run in a loop
until this threshold is reached. For workloads that take more
than 60 seconds, we collect power data until the workload
is completed. We then average the power over the duration of
valid samples to report. This approach ensures that we capture
the system’s full power consumption during execution.

Finally, we employ a unified approach to evaluating en-
ergy efficiency on different scales and types of systems.
For throughput benchmarks, such as datacenter and offline
edge inference [70], we measure throughput in Samples/s and
power consumption in Watts. We calculate energy efficiency
as (Samples/s)/(Watts), yielding Samples/Joule. For latency
benchmarks such as tiny inference [5], we use the inverse of
energy in 1/Joules. We treat these two metrics as comparable,
despite their different units, because the benchmarks fix dif-
ferent execution factors and thus require different performance

evaluations. This standardized approach allows for meaningful
comparisons across the diverse range of systems.

B. MLPerf Inference Setup and Operation

Despite the vast difference in the scale of ML inference
systems, the core principles of measurement remain consistent,
with adaptations made to address the unique challenges of each
size of the system. For tiny inference systems, which often
function as always-on detectors like wake-word recognizers or
presence detectors, the energy per inference is the key metric.
These systems align well with the single-stream scenario in
MLPerf Inference, which is the time to process one sample.
The energy measurement setup for systems in the Tiny
division is illustrated in Figure 4a. To accurately measure
such low-power devices, we utilize an I/O Manager that sits
between the host and the System Under Test (SUT). The I/O
Manager is implemented with an Arduino UNO board that
captures timing information with resolution that the host (a
Windows, Linux, or Mac) cannot. The I/O Manager, along
with the level shifter, also electrically isolates the SUT from
the host to avoid parasitic powering from the host’s signal
lines, skewing the measurements. The I/O Manager provides
a USB connection to the host and connects to the SUT’s Uni-
versal Asynchronous Receiver-Transmitter (UART) through an
isolating level shifter, ensuring minimal power transfer.

To measure the energy, the host issues an infer command,
which is relayed to the SUT via the I/O manager and includes
a number of inferences to be performed. The SUT signals the
beginning and end of the inference activity by toggling the
Timestamp pin. This timing information is captured by the
external energy monitor along with the current and voltage
waveforms and transmitted to the host. The host then integrates
the power over the indicated interval and divides by the
number of inferences to calculate the energy per inference.

For larger-scale inference systems, we rely on a client-
server architecture where the SUT acts as the client, and
a separate system, called the Director, serves as the server.
This setup allows for more complex measurement scenar-
ios and better control over the benchmarking process. The
process begins with an NTP (Network Time Protocol) sync
between the SUT and the Director to align their timestamp
references. The Director then initiates communication with a
SPEC-approved power analyzer through PTD (Power-Thermal

Daemon) API calls. Once the connection is established, the
Director commands the SUT to start executing the loadgen,
which runs the workload of interest. During execution, the
Director logs power data from the analyzer, typically in a
text file later converted to CSV format for easier processing.
This file contains detailed information on the current, voltage,
phase, and power usage of the SUT, all with time stamps.

Currently,

the SUT collects performance log files that
timestamp different phases of execution. To help us improve
measurement accuracy, we employ a range mode that involves
an initial run to determine the maximum current and voltage
levels for a particular workload. Subsequent runs then use fixed
analyzer ranges based on these observed peaks, allowing for
higher accuracy within that range. For multi-node inference
configurations, we adapt our methodology to handle multiple
SUTs connected to a single analyzer or multiple analyzers
connected to a single SUT, as shown in Figure 4b. This
flexibility allows us to accurately measure power consumption
in more complex distributed inference scenarios.

Across all scales of inference systems, from tiny to data-
center, we focus on measuring the power consumption during
the active portion of the workload. By aligning power mea-
surements with the execution phase and employing techniques
appropriate to each scale, we ensure accurate and comparable
results across the entire spectrum of inference deployments.

C. MLPerf Training and HPC Setup and Operation

MLPerf Training and HPC [16] Power Measurement
methodology addresses the unique challenges posed by large-
scale ML training systems. These systems range from single
nodes to clusters comprising thousands of nodes and tens of
thousands of accelerators, often located in secure on-premises
datacenters, cloud environments, or national laboratories. The
scale, complexity, and security requirements of these systems
require a different approach compared to inference setups.

At the core of our methodology is a software-based mea-
surement scheme. This approach leverages the existing teleme-
try infrastructure of large clusters to measure system power,
bypassing the impracticality of using separate power analyzers
for each node and interconnect component. The scheme is
designed to be flexible, accommodating the diverse range
of telemetry systems used by different submitters while still
adhering to MLPerf’s goal of measuring full system power.

Power measurement at the node level is the foundation
of our approach, as shown in Figure 4c. For each partici-
pating compute node, power measurements are taken using
the submitter’s own telemetry systems, such as IPMI [48]
or RedFish [21]. We recommend out-of-band measurement
techniques to minimize interference with the ML job itself.
In cases where individual node measurement is not feasible,
power can be measured at the Power Distribution Unit (PDU)
level, provided that all nodes supplied by that PDU are
involved in the ML job. Submitters are required to thoroughly
document the accuracy of their telemetry systems for review.
Interconnect power measurement presents its own set of
challenges. For interconnect switches, power can be measured

similarly to that of compute nodes. However, in cases where
direct measurement is not possible, we allow estimated power
values. When estimation is used, submitters must provide
detailed documentation of their estimation methodology, en-
suring transparency and comparability between submissions.
To ensure consistency and comparability across submissions
with different hardware platforms and configurations, we have
implemented a standardized data logging process. Submitters
are required to convert their node and interconnect logs to a
standardized format using the MLPerf Logging Library.

The energy-to-train calculation is a key component of our
methodology. Our result summarizer parses both performance
and power logs to derive this metric. The process involves
aggregating power consumption data for the duration of the
ML training process. The summarizer determines the ML
training process window by extracting the timestamps of the
run start and run stop log lines in the performance log. It then
parses each node’s log file to extract power samples that fall
within this window. The energy for each node is calculated
by integrating power samples over the run’s time window, and
the total energy is computed by summing the energy across
all compute and interconnect components.

D. Validation and Reporting

The MLPerf Power methodology places a significant em-
phasis on the validation of measurements and the standardiza-
tion of reporting. This is done to ensure accuracy, integrity,
comparability, and usefulness of the benchmarking results.

Quality Accuracy requirements form the foundation of our
validation process. Although we recognize that the diverse
range of systems and scales in ML workloads requires different
approaches to power measurement, all must meet stringent
accuracy standards as described by MLPerf Inference [70] and
Training [46]. For datacenter and high-performance computing
systems, we require documentation of the telemetry systems
used, including their accuracy specifications and calibration
procedures. Edge systems using external power analyzers
must employ SPEC-approved devices, known for their high
accuracy and reliability. For TinyML, we mandate specialized
micro-power instruments with target accuracy ratings [5].

Sampling Rates We also specify minimum measurement
durations and sampling rates to ensure that the power data ad-
equately captures the system’s behavior under the benchmark
workload. For instance, we require a minimum of 60 seconds
of power data collection for most scenarios, with provisions
for longer durations in cases of extended workloads. This
ensures that temporary fluctuations or anomalies do not unduly
influence the reported power consumption.

Reporting To facilitate comparison between diverse sub-
missions, we have developed a standardized reporting for-
mat. This format is designed to capture all relevant details
of the power measurement setup, system configuration, and
benchmark results. Submitters are required to use the MLPerf
Logging Library to convert their raw power logs into this
standardized format. The reporting template includes fields

(a) Tiny System

(b) Multi-SUT Inference System

(c) Training System

Fig. 4: Measurement diagrams for Tiny, Multi-SUT Inference, and Training systems.

for system specifications, power measurement methodology,
workload details, and actual power and performance metrics.
The HPC and training reporting and submission process is
designed to more comprehensive, flexible, and contextually
rich. Submitters are required to report power consumption
data along with additional information that could affect these
measurements, including details on cooling solutions, power
management techniques, and environmental conditions during
the testing. The process allows for reporting at either the PDU
or node level, accommodating different measurement setups.
Submitters run the benchmark with their chosen telemetry
option, estimate switch power if necessary, and post-process
power logs to conform to the MLPerf Logging Library format.
For systems where certain components’ power consumption
must be estimated, such as interconnect power in large-
scale training sets, dedicated sections are provided in the
reporting format to document the estimation methodology.
The submission package must
include the processed data
in a specified directory structure, along with documentation
on telemetry accuracy. This comprehensive approach ensures
transparency and comparability between different submissions,
while allowing for the flexibility needed to accommodate
various system configurations and measurement approaches.

Review Our review process [52], [57] is rigorous and mul-
tilayered, designed to catch any inconsistencies or anomalies
in the submissions. Upon receiving a submission, our review
committee, composed of experts in ML systems and power
measurement, examines the reported data and documentation.
They verify that all required information is present, that the re-
ported measurements comply with the accuracy requirements,
and that the power consumption figures are consistent with the
system specifications and workload characteristics.

Since the ecosystem is still evolving, in cases where novel
measurement techniques or system architectures are involved,
the review process may include additional steps. This might
involve consultations with domain experts or requests for more
detailed explanations of the measurement methodology. The
goal is to ensure that innovative approaches can be included
in the benchmarks while maintaining the high standards of
accuracy and comparability that MLPerf is known for.

Transparency is a key principle of our validation and re-
porting process. All accepted submissions are made publicly
available, along with their full documentation. This allows
community scrutiny and fosters trust in the benchmark results.

It also provides valuable data for researchers and engineers
working to improve the energy efficiency of ML systems [19].

V. RESULTS/EVALUATION

We analyze this diverse range of data to gain an under-
standing of comparative trends across different workloads and
system scales, shedding light on the areas of focus within
the industry. MLPerf benchmarks span a diverse range of
ML workloads, from resource-constrained keyword spotting to
complex computer vision, sophisticated recommendation sys-
tems, and state-of-the-art large language models (LLMs) [47],
[49]. With each version, we make changes to the MLPerf
benchmarks to adapt to the changing ML landscape, such
as updating datasets, adding new benchmarks, updating the
network architectures, and refining our methodology and hy-
perparameters [69]. A notable example is the addition of LLMs
in version 3.1 to address the growing interest in this area.

The MLPerf Power submissions offer a large dataset that
spans several years and four workload versions. This exten-
sive collection, comprising 1,841 submission results, provides
useful insight into industry-wide trends in energy efficiency
improvements. The data set
includes 590 data center re-
sults [50], 792 edge results [51], 447 tiny MLPerf results [53]
for inference, and 12 training submission results [56]. Un-
less otherwise specified, all results are verified MLPerf sub-
missions [55]. Configurations (performance-optimized, energy
efficiency-optimized, or default) and logs of individual MLPerf
submissions are publicly available on Github.

A. Overall Energy Efficiency Improvements in Inference

Following our earlier examination of performance trends
shown in Figure 1, we now shift our focus to advancements
in AI energy efficiency. Figure 5 highlights significant im-
provements across the categories of data center, edge, and tiny
devices. These advancements are expressed using normalized
samples per joule, a measure that encapsulates enhancements
in processing power alongside an industry-wide emphasis on
optimizing AI technology for energy efficiency.

In the data center category, RetinaNet [45] shows the
strongest increase in samples per joule amongst older bench-
marks. However, it is clear to see that the new generative AI
benchmarks exhibit massive energy efficiency improvements
in recent versions. GPT [6] and Llama2 [79] both reach
over 100× improvements in energy efficiency, which can be
attributed to the significant attention given to LLM computing

(a) Datacenter

(b) Edge

(c) Tiny

Fig. 5: Comparison of energy efficiency trends for datacenter, edge, and tiny inference.

efficiency as a result of its significant impact on scale. Overall,
data center models show large gains in energy efficiency, with
notable improvements in the later iterations.

In the edge category, improvements in energy efficiency
are moderate. BERT at 99% accuracy, or BERT-99.0, and
RNN-T are at the forefront with substantial early improve-
ments, reaching up to four times their initial efficiency. In
contrast, ResNet shows a more steady enhancement, leveling
at approximately 1.5 normalized samples per joule. Progress in
edge devices underscores the emphasis on refining AI models
for real-time, on-device execution, crucial for applications that
demand low latency and high energy efficiency.

In the tiny category, we observe significant improvements in
energy efficiency across every model. ResNet [24] achieves
a normalized efficiency gain over 1000×, while the other 3
workloads all reach between 79× and 596×. These results
underscore the rapid advances in AI efficiency for small-
scale, resource-constrained environments. In particular, tiny-
scale computing has experienced massive improvements in
energy efficiency over its early versions, with gains between
2 and 3 orders of magnitude across every workload.

The trends across all three categories highlight the ongoing
efforts and successes in enhancing AI models’ energy effi-
ciency, ensuring that high performance can be achieved sus-
tainably across different environments and use cases. However,
for edge and tiny systems, energy efficiency advancements
have plateaued in the last 3 MLPerf versions, indicating a
need for more innovative approaches to continue improving
efficiency. Improving their energy efficiency remains crucial
for the scalability and sustainability of future IoT systems.

B. Scalability Analysis of Training

As ML models grow in complexity, the time to solution
(train) becomes a critical factor in the design of the ML
system [8], [10]. Researchers and industry practitioners have
developed various parallelization strategies to reduce training
time by leveraging additional computational resources. How-
ever, our MLPerf Power benchmarking reveals that this pursuit
of faster training times introduces a new dimension of com-
plexity: energy consumption at scale. To this end, we present
a novel analysis of the intricate relationship between system
scale, training time, and energy consumption, drawing from
our recently released MLPerf Training power benchmarks.

Figure 6 shows the time-to-train and energy consumption
metrics from our v4.0 training power submissions for the

Fig. 6: Energy consumption and time-to-train for Llama2-70b
LoRA fine-tuning across different numbers of accelerators.

Llama2-70b [79] model on three distinct system scales.
This data provides insights into the non-linear nature of en-
ergy scaling in large-scale ML training systems. With perfect
scaling, increasing the number of accelerators should lead to a
proportional reduction in training time and the energy to train
should remain flat. But our results reveal a complex reality.

As the system scale increases, we observe diminishing
returns in training time reduction. The improvement in training
time becomes less pronounced as we add more accelerators,
due to increased inter-accelerator communication and reduced
FLOPs utilization in each accelerator. This results in higher
accelerator-hours to train despite absolute time-to-train being
lower. Simultaneously, we see an increase in energy to train.
This is due to the increase in total accelerator-hours to train
and the added energy cost of interconnect components, such as
networking switches, required for inter-node communication.
Our analysis suggests several key implications for future
ML system design. First, architects must carefully consider
the energy costs associated with scaling up training systems,
which can lead to the development of more energy-efficient
accelerators and interconnects. This energy-aware scaling ap-
proach is crucial for sustainable AI development.

Second, understanding the parallelization characteristics of
different ML workloads can help to determine the optimal
system scale that balances performance gains with energy
efficiency. This workload-specific optimization will be key to
maximizing efficiency across various ML tasks. Lastly, future
ML systems should optimize not just compute energy, but
also other types of energy consumption, such as interconnect
energy, which becomes increasingly significant at larger scales.

C. Workload-Specific Insights

Unlike raw power measurements, which are impacted by
how submitters scale their hardware in relation to model
size, energy consumption is agnostic to constant power or
latency requirements. For this reason, we use the energy per
benchmark inference to evaluate its computational demands
and efficiency. We provide insights into the relationships
between model size, workload characteristics, and energy
consumption of each benchmark submission by evaluating two
divisions of inference benchmarks, datacenter and tiny. Note
that while there is no perfectly comparable metric for model
size, we use total matrix accumulation (MAC) operations as
an estimate. For an in-depth description of the workloads and
deeper insight into the computation of each benchmark, refer
to the MLPerf-Inference [70] and MLPerf-Tiny [5] papers.

Datacenter The blue bars in Figure 7 show the energy
per inference sample and workload computation size for each
MLPerf Inference benchmark on a large-scale datacenter sys-
tem. ResNet [24], [59], a classic image classification model
widely used as a baseline for computer vision (CV) tasks,
consumes the least energy per inference at 8.7 mJ/Sample.
As computer vision models get larger and more capable, the
energy for a single inference also grows. RetinaNet [45]
that uses multiple ResNet
is an object detection model
inferences for multiple classifications within a single image
and thus a multiple order-of-magnitude growth in inference
energy. The computer vision benchmark, 3D U-Net [7] is
a 3-dimensional medical image segmentation model that uses
even more energy per inference due to the extra visual dimen-
sion and required precision.

Compared to CV models, recommendation models such
as DLRM-v2 [62] exhibit very different data movement
patterns [26]. Recommendation models are known to have
memory bandwidth bottlenecks and therefore exert a greater
percentage of total energy on data movement than compute
bound models. Although DLRM-v2 only has 1.06× more
MACs than ResNet, the 2.3× increase in J/Sample can be
attributed to the data movement energy not captured in pure
computation measurements.

We also evaluate two NLP benchmarks. RNN-T [33] is
a speech recognition model designed to transcribe spoken
language into text, while BERT [13] is a multipurpose encoder-
only transformer language model capable of various tasks
such as question answering. Despite their model architecture
similarities, these workloads likely have much lower inference
energy requirements than an LLM like GPT-J due to 1 order
of magnitude fewer parameters and and 2 orders of magnitude
fewer MAC computations.

The three generative AI models in MLPerf Inference are
significantly more energy-hungry than every benchmark other
than 3D U-Net. Generative transformer-based model archi-
tectures are much more complex than other models in both
raw MAC computation and data movement [36]. Stable
Diffusion [73] computes 673× more MACs and con-
sumes 1082× more energy for a single image generation

Fig. 7: Energy consumption and total MAC operations per
inference for each MLPerf Inference and MLPerf Tiny bench-
marks. Each benchmark is categorized by its workload type.

inference compared to RetinaNet’s image object detection
network. LLM inference is the most widely used generative AI
model in industry and is benchmarked by the 70B parameter
Llama2 [79] and the 6B parameter GPT-J [6]. Interestingly,
despite the 11.7× increase in parameters, Llama2 computes
4.4× more MACs and uses 4.2× more energy per inference
than GPT-J. This discrepancy could be due to the non-linear
scaling of energy efficiency in language models by parameters,
better energy efficiency of Llama2, or differences in the
benchmark datasets.

It is important to note that autoregressive language models
exhibit variability in their inference lengths. In MLPerf Infer-
ence v4.0, each of the two LLM benchmarks uses a different
metric. GPT-J is reported in samples/second, while Llama2
is given in tokens/second. From version 4.1 onward, all LLM
benchmarks will be reported in tokens/second, as is common
in industry. However, for better alignment with how humans
interact with LLMs and comparability with other MLPerf
Inference benchmarks, we use samples/second to evaluate
LLM performance for energy efficiency calculations in Fig. 7.
To do this, we convert tokens into samples using an estimated
median sequence length of 69 tokens/sample for GPT-J and
292 tokens/sample for Llama2, with an error margin due to
sequence length variation, user pairs, and other factors.

As newer ML models grow in their computational com-
plexity, their energy consumption scales by multiple orders
of magnitude. While our largest benchmarked LLM Llama2
consumes 111.4 J/Sample at 70B parameters, current models
are growing into the multi-trillion parameter scales [30]. Due
to propriety and compute resource limitations, we cannot
explicitly benchmark these large language models in MLPerf.
However,
is clear that as energy consumption reaches
multi-KJs for a single question-answer inference, it is more
important than ever to consider energy when designing and
optimizing models for more complex tasks.

it

are

Tiny

trends

Similar

in MLPerf-Tiny.
AutoEncoder [38],
[67], an anomoly detection
model, consumes the least energy at 20 mJ per inference on
the reference hardware and computes the fewest MACs at
265k (without batch norm operations). MobileNet [27],

[39],

seen

Fig. 8: Distribution of BERT energy efficiency drop when
increasing accuracy from 99% to 99.9% for datacenter offline
submissions in MLPerf v3.1, v4.0, and v4.1.

a visual wake-word detection model, and DSCNN [87], a
keyword-spotting model, follow similar trends with model
complexity leading to higher energy per inference.

CV with ResNet [24] is an interesting comparison between
datacenter and tiny systems. The datacenter ResNet bench-
mark uses the 224x224 ImageNet dataset at 99% inference
accuracy, while the tiny ResNet uses the 32x32 CIFAR10
dataset at 85% inference accuracy. With the 49× larger input
dataset, 14% increase in the accuracy target and 326× increase
in the total MACs computed, the datacenter system consumes
321.7× more energy per inference than the tiny system. While
ML ASICs in datacenter scale systems and embedded neural
processors in tiny scale systems encounter different energy
bottlenecks, there is strong continuity between workload com-
plexity and inference energy across hardware scales.

D. Energy Efficiency Costs of High Accuracy Inference

Achieving high accuracy in an ML model generally requires
more computational resources due to the increased complexity
in calculations, data processing, and additional training itera-
tions. Each MLPerf benchmark includes a minimum accuracy
target that must be achieved to qualify for a successful submis-
sion. Several MLPerf workloads have separate benchmarks set
at 99% and 99.9% accuracy. For example, in the low accuracy
BERT-99.0 benchmark, participants must achieve 99% of
the original model’s single precision FP32 accuracy.

Figure 8 shows the energy efficiency cost distribution on the
BERT benchmark as accuracy increases from 99% to 99.9%
in offline datacenter submissions for versions 3.1 and 4.0.
Energy efficiency cost is measured as the percentage change
in Samples/Joule when transitioning from BERT-99.0 to
BERT-99.9. We observe that datacenter systems running
inference at 99.9% accuracy are, on average, half as efficient as
those running inference at 99% accuracy. However, the three
samples on the left showcase the capabilities of recent sub-
missions that utilize more advanced quantization techniques
to maintain strong energy efficiency at higher accuracy. This
trend indicates that running inference at higher accuracy is
becoming cheaper and more sustainable.

E. Low Precision Efficiency Improvement Analysis

Using quantization improves performance [29], [61] and
[20], [23] by decreasing the amount

reduces power usage

Fig. 9: Energy efficiency for high and low accuracy targets
for BERT MLPerf-Inference. INT8 quantization is used for
BERT-99.0 submissions, while BERT-99.9 submissions
initially used FP16 and later adopted FP8 starting from v3.1.

of computation, either directly or by enhancing parallelism.
However, due to the stringent accuracy standards set by the
MLPerf benchmarks, participants must find a careful balance
between aggressive quantization to boost performance and
energy efficiency while still meeting the accuracy criteria.
Figure 9 examines the normalized energy efficiency of systems
using workloads BERT-99.0 and BERT-99.9 on the same
hardware platform within the same MLPerf-Inference version.
BERT-99.0 submissions consistently use INT8 quanti-
zation, as it is sufficient to meet the 99% accuracy target.
However, INT8 is inadequate for the higher accuracy target of
BERT-99.9, which requires the use of 16-bit floating point
(FP16) until version 3.1. Starting from v3.1, the new hardware
that supports 8-bit floating point (FP8) formats allowed sub-
mitters to achieve higher performance and energy efficiency
while still meeting the 99.9% accuracy target. Consequently,
the energy efficiency of BERT-99.9 submissions improved
significantly from approximately 50% of BERT-99.0 in older
submissions to around 85% in newer ones.

Thus, the adoption of lower precision floating point formats,
such as FP8, leads to reduced energy consumption per opera-
tion and energy usage in memory data transfers. Our analysis
strongly suggests that innovation in quantization techniques
will continue being a driving force in energy efficient systems.

F. Software versus Hardware-driven Efficiency Improvements

As discussed in Section V-A, while significant improve-
ments in efficiency have been observed in the past, recent
versions of MLPerf show signs of plateauing (Figure 5).
We analyze these advancements by isolating the sources of
efficiency gains, measuring their extent, and evaluating their
pros and cons. Our goal is to provide hardware and software
engineers with a better understanding of the origins of energy
enhancements to enable more informed decision making.

Figure 10 illustrates submissions with identical hardware
configurations (host CPU, number of accelerators, type of
accelerator) in two successive versions of any workload. The
histogram shows that 89% of the data is positive, suggesting
that most submissions on similar hardware have some degree
of improvement in software efficiency. We propose that the
small percentage of data showing a minor decline in efficiency

Fig. 10: Distribution of energy efficiency change when opti-
mizing only the software between identical hardware system
submissions in consecutive MLPerf datacenter offline versions.

might be due to configuration changes not considered in our
hardware equivalency or vendor modifications that do not
focus on energy efficiency. Furthermore, it is notable that 18%
of the data achieves improvements in energy efficiency that
exceed 50%, providing strong evidence that software optimiza-
tions have significantly improved energy efficient systems.

We analyze a case study on software enhancements in
Figure 11a,
identifying an identical edge system in four
consecutive versions of ResNet inference workloads. With
negligible performance loss, power consumption has decreased
at a fast pace, resulting in 1.28× energy efficiency improve-
ment. Submission logs show that improvements in software
development kits (SDKs) and total cost of ownership (TCO)
contribute to the gains in energy efficiency. Sacrificing a small
amount of performance for these software-level optimization
additions results in energy efficiency enhancements.

In addition to software improvements, we see improvements
in hardware to reduce training time and energy. In Figure 11b,
we evaluate a case study on hardware where we use a constant
software stack, including quantization, and evaluate subse-
quent versions of the ASIC on the BERT benchmark.1 Perfor-
mance increases to 1.72× and power consumption decreases
to 0.38×, giving a 4× increase in energy efficiency in the
latest hardware version of the system. The computing units in
newer hardware versions have features that yield better energy
efficiency: support for lower numeric precision, more efficient
designs for the same precision, and better handling of memory-
compute coordination [34], [35]. We also see improvements
in the fast interconnects between ML accelerators in terms
of higher bandwidth, lower latency, and support for more
diverse communication patterns. Furthermore, power man-
agement techniques such as dynamic voltage and frequency
scaling allow processors to adjust their power consumption
based on workload demands. Our goal in analyzing these
advancements is to provide architects and ML engineers with
a better understanding of the origins of energy enhancements
to enable better and more informed decision making.

VI. RECOMMENDATIONS FOR FUTURE DIRECTIONS

We have identified four main areas that require the attention

of the ML systems research community and industry.

1Figure 11b results are unverified by MLPerf Power. This data is used here
only to show general trends in energy efficiency from hardware optimizations.

(a) Software Optimizations

(b) Hardware Optimizations

Fig. 11: Normalized energy efficiency, performance, and
power improvements from progressive versions of software-
isolated and hardware-isolated optimizations.

Mitigating Efficiency Plateau. Our analysis quantitatively
shows that significant improvements in energy efficiency in
new generative AI benchmarks in the datacenter category due
to spiking commercial interest in its performance and energy
efficiency. However, recent MLPerf versions are indicating a
plateau in advancements in the edge and tiny categories. To
overcome this, we recommend developing granular, workload-
specific energy efficiency metrics that account for unique
model characteristics, training regimes, and energy cost per
accuracy improvement. We also suggest analyzing the energy
efficiency of individual training and inference phases.

Promoting AI Sustainability. We recommend integrating
environmental
impact considerations into AI development
practices and and ethical frameworks. Beyond including en-
ergy efficiency as a key metric, we recommend developing
and standardizing tools to estimate the carbon footprint of
AI model training and inference, based on existing work in
this area. Existing [2], [3], [9], [31], [42], [85] and new
tools should be integrated into AI frameworks to provide real-
time feedback on power consumption during the development
process. Furthermore, we suggest exploring the creation of
incentive structures, such as energy efficiency certifications or
awards, to encourage energy-efficient ML system design.

Embracing AI Systems Policy Our work directly supports
emerging regulations like the European Union AI Act’s Article
53 [1], which requires AI model providers to document energy
usage. The MLPerf Power benchmarking methodology can

help meet these regulatory demands by providing standard-
ized and reproducible energy efficiency measurements. As
sustainability becomes a key focus in AI, this benchmark
will be crucial for promoting transparency and compliance.
Future research should explore integrating these benchmarks
into regulatory frameworks [37], [83], [88] and expanding the
policy to cover new AI workloads and hardware systems.

Nurturing Industry-Wide Collaboration. The diverse na-
ture of AI workloads and rapid pace of technological advance-
ment require continued coordinated efforts to drive improve-
ments in energy efficiency. Building on the existing work of
MLCommons [49] and its MLPerf Working Groups [53], [54],
[56], [58], we recommend to growing these collaborations to
address emerging challenges in AI energy efficiency. In addi-
tion to expanding the scope of current benchmarks to include
more diverse and emerging AI workloads, such as computer
vision [18], multi-model [77], [84] and multimodal [63], [68]
workloads, we suggest developing guidelines for reporting the
environmental impact of AI model development and deploy-
ment that complement the energy efficiency metrics.

Large-Scale AI Training Optimization. Our analysis of
Llama2-70b fine-tuning, reveals a complex relationship
between system scale, training time, and energy consumption.
We recommend developing granular, workload-specific energy
efficiency metrics for large-scale AI training that account
for the unique characteristics of different AI models and
training regimes, going beyond overall system efficiency to
examine component-level energy consumption patterns. We
also recommend creating a framework for analyzing the energy
efficiency of individual training phases, such as data loading,
forward passes, backward passes, and parameter updates. This
granular approach could help identify specific bottlenecks in
large-scale training systems and guide targeted optimizations.
Additionally, we suggest incorporating scaling efficiency met-
rics that quantify the energy cost of distributed training.

Accuracy-Efficiency Trade-offs. Our analysis of the BERT
and DLRM-v2 benchmarks demonstrates the significant im-
pact of quantization on energy efficiency and reveals sub-
stantial energy costs associated with high-accuracy inference.
To address the accuracy-efficiency trade-off more explicitly,
we recommend extending the MLPerf Power framework to
include fine-grained metrics that quantify energy cost per
accuracy improvement and analyze the submission results.

Additionally, incorporating methodologies for tiered accu-
racy offerings would enable dynamic adjustment of model
accuracy based on application requirements and energy con-
straints [22], [74]. These enhancements would provide a more
comprehensive framework for evaluating and optimizing the
energy efficiency of AI models across various accuracy levels
and deployment scenarios, ultimately guiding the development
of more energy-efficient AI systems.

Multimodal Deployments. As part of MLCommons’ on-
going effort to add a benchmark for the audiovisual (AV)
domain [89], we are incorporating multiple multimodal (cam-
era + LiDAR, for example) neural networks covering key
elements of the AV stack which will execute concurrently. As

power consumption is primary concern for such use cases, we
are currently devising an appropriate methodology for power
measurement.

Software-Hardware Co-design Paradigm. As we show,
software optimizations alone can lead to significant energy
efficiency improvements. We recommend establishing formal
collaborative frameworks for software-hardware co-design in
AI system development to facilitate early and ongoing inter-
action between hardware designers and software engineers,
with shared energy efficiency targets guiding development
processes.

VII. CONCLUSION

MLPerf Power introduces a standardized methodology for
benchmarking energy efficiency in machine learning systems
across an unprecedented scale, from microwatts to megawatts.
This work addresses the critical need for understanding and
optimizing power consumption in ML as the field advances
rapidly. Our key contributions include establishing an industry-
wide standard for ML system power measurement, presenting
the first large-scale study of ML system energy efficiency,
analyzing longitudinal energy efficiency trends across multiple
generations of MLPerf, and providing actionable insights on
energy optimization techniques. Our findings emphasize the
importance of energy efficiency as a primary metric in ML
system design. As the field progresses, the insights gained
from MLPerf Power will serve as a stepping stone towards
the future of sustainable machine learning systems.

ACKNOWLEDGEMENTS

is

MLPerf Power

the collective effort of numerous
individuals from various organizations. In this section, we
would like to acknowledge all
those who contributed to
producing the initial set of results or supported the overall
benchmark development. This work was supported in part by
funding from SRC and NSF.

Broadcom - Ravi Soundarajan.
Databricks - Hanlin Tang.
Fujitsu - Takahiro Notsu.
Google - Tom Jablin, David Patterson.
Infineon - Peter Torelli.
Intel - Remya Jayaraman.
KRAI - Leo Gordon.
Meta - Carole-Jean Wu, Whitney Zhou.
MLCommons - Pablo Gonzales Mesa.
NVIDIA - Ashwin Nanjappa, Ashutosh Dhar, Dilip Sequeira.
Xored - Albert Safin, Yaroslav Kurlaev, Julia Mozhar, Daniil
Efremov, Andrey Atangulov, Sergei Lunin.

REFERENCES

[1] “Regulation of the european parliament and of the council laying down
harmonised rules on artificial intelligence (ai act) and amending certain
union legislative acts, article 53,” https://eur-lex.europa.eu/legal-content/
EN/TXT/?uri=CELEX%3A52021PC0206, 2024, accessed: 2024-09-05.

[2] B. Acun, B. Lee, F. Kazhamiaka, K. Maeng, U. Gupta, M. Chakkar-
avarthy, D. Brooks, and C.-J. Wu, “Carbon explorer: A holistic frame-
work for designing carbon aware datacenters,” in Proceedings of the 28th
ACM International Conference on Architectural Support for Program-
ming Languages and Operating Systems, Volume 2, 2023, pp. 118–132.
[3] B. Acun, B. Lee, F. Kazhamiaka, A. Sundarrajan, K. Maeng,
M. Chakkaravarthy, D. Brooks, and C.-J. Wu, “Carbon dependencies
in datacenter design and management,” ACM SIGENERGY Energy
Informatics Review, vol. 3, no. 3, pp. 21–26, 2023.

[4] L. F. W. Anthony, B. Kanding, and R. Selvan, “Carbontracker: Tracking
and predicting the carbon footprint of training deep learning models,”
arXiv preprint arXiv:2007.03051, 2020.

[5] C. R. Banbury, V. J. Reddi, P. Torelli, J. Holleman, N. Jeffries,
C. Király, P. Montino, D. Kanter, S. Ahmed, D. Pau, U. Thakker,
A. Torrini, P. Warden, J. Cordaro, G. D. Guglielmo, J. M. Duarte,
S. Gibellini, V. Parekh, H. Tran, N. Tran, W. Niu, and X. Xu,
“Mlperf tiny benchmark,” CoRR, vol. abs/2106.07597, 2021. [Online].
Available: https://arxiv.org/abs/2106.07597

[6] T. B. Brown, “Language models are few-shot learners,” arXiv preprint

ArXiv:2005.14165, 2020.

[7] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
“3d u-net: Learning dense volumetric segmentation from sparse annota-
tion,” in Medical Image Computing and Computer-Assisted Intervention
– MICCAI 2016, S. Ourselin, L. Joskowicz, M. R. Sabuncu, G. Unal,
and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp.
424–432.

[8] C. Coleman, D. Narayanan, D. Kang, T. Zhao, J. Zhang, L. Nardi,
P. Bailis, K. Olukotun, C. Ré, and M. Zaharia, “Dawnbench: An end-
to-end deep learning benchmark and competition,” Training, vol. 100,
no. 101, p. 102, 2017.

[9] B. Courty, V. Schmidt, S. Luccioni, Goyal-Kamal, MarionCoutarel,
Inimaz, supatomic,
B. Feld, J. Lecourt, LiamConnell, A. Saboni,
M. Léval, L. Blanche, A. Cruveiller, ouminasara, F. Zhao, A. Joshi,
A. Bogroff, H. de Lavoreille, N. Laskaris, E. Abati, D. Blank,
Z. Wang, A. Catovic, M. Alencon, M. St˛echły, C. Bauer, L. O. N.
de Araújo, JPW, and MinervaBooks, “mlco2/codecarbon: v2.4.1,” May
2024. [Online]. Available: https://doi.org/10.5281/zenodo.11171501
[10] J. Dean, D. Patterson, and C. Young, “A new golden age in computer ar-
chitecture: Empowering the machine-learning revolution,” IEEE Micro,
vol. 38, no. 2, pp. 21–29, 2018.

[11] R. Desislavov, F. Martínez-Plumed, and J. Hernández-Orallo, “Compute
and energy consumption trends in deep learning inference,” arXiv
preprint arXiv:2109.05472, 2021.

[12] R. Desislavov, F. Martínez-Plumed, and J. Hernández-Orallo, “Compute
and energy consumption trends in deep learning inference,” arXiv
preprint arXiv:2109.05472, 2021.

[13] J. Devlin, “Bert: Pre-training of deep bidirectional transformers for
language understanding,” arXiv preprint arXiv:1810.04805, 2018.
[14] J. J. Dongarra, H. W. Meuer, E. Strohmaier, H. D. Simon, and A. van der
Steen, “Top500 supercomputer sites,” Supercomputer, vol. 13, pp. 89–
111, 1997.

[15] L. Eeckhout, “Toward sustainable computer systems,” Computer, vol. 57,

no. 02, pp. 101–104, feb 2024.

[16] S. Farrell, M. Emani, J. Balma, L. Drescher, A. Drozd, A. Fink, G. Fox,
D. Kanter, T. Kurth, P. Mattson, D. Mu, A. Ruhela, K. Sato, K. Shirahata,
T. Tabaru, A. Tsaris, J. Balewski, B. Cumming, T. Danjo, J. Domke,
T. Fukai, N. Fukumoto, T. Fukushi, B. Gerofi, T. Honda, T. Imamura,
A. Kasagi, K. Kawakami, S. Kudo, A. Kuroda, M. Martinasso, S. Mat-
suoka, H. Mendonça, K. Minami, P. Ram, T. Sawada, M. Shankar,
T. S. John, A. Tabuchi, V. Vishwanath, M. Wahib, M. Yamazaki, and
J. Yin, “Mlperf™ hpc: A holistic benchmark suite for scientific machine
learning on hpc systems,” in 2021 IEEE/ACM Workshop on Machine
Learning in High Performance Computing Environments (MLHPC),
2021, pp. 33–45.

[17] W.-c. Feng and K. Cameron, “The green500 list: Encouraging sustain-
able supercomputing,” Computer, vol. 40, no. 12, pp. 50–55, 2007.
[18] A. Fu, M. S. Hosseini, and K. N. Plataniotis, “Reconsidering co2
emissions from computer vision,” in 2021 IEEE/CVF Conference
on Computer Vision and Pattern Recognition Workshops (CVPRW).
Los Alamitos, CA, USA: IEEE Computer Society,
jun 2021, pp.
2311–2317. [Online]. Available: https://doi.ieeecomputersociety.org/10.
1109/CVPRW53098.2021.00262

[19] G. Fursin,

and cost-effective AI/ML
systems with Collective Mind, virtualized MLOps, MLPerf, Collective

“Enabling more

efficient

Knowledge Playground and reproducible optimization tournaments,”
2024. [Online]. Available: https://arxiv.org/abs/2406.16791

[20] A. Garofalo, M. Rusci, F. Conti, D. Rossi, and L. Benini, “Pulp-nn:
Accelerating quantized neural networks on parallel ultra-low-power risc-
v processors,” Philosophical Transactions of the Royal Society A, vol.
378, no. 2164, p. 20190155, 2020.

[21] G. Gonçalves, D. Rosendo, L. Ferreira, G. L. Santos, D. Gomes,
A. Moreira, J. Kelner, D. Sadok, M. Wildeman, and P. T. Endo, “A
standard to rule them all: Redfish,” IEEE Communications Standards
Magazine, vol. 3, no. 2, pp. 36–43, 2019.

[22] M. Halpern, B. Boroujerdian, T. Mummert, E. Duesterwald, and V. J.
Reddi, “One size does not fit all: Quantifying and exposing the accuracy-
latency trade-off in machine learning cloud service apis via tolerance
tiers,” arXiv preprint arXiv:1906.11307, 2019.

[23] S. Hashemi, N. Anthony, H. Tann, R. I. Bahar, and S. Reda, “Un-
derstanding the impact of precision quantization on the accuracy and
energy of neural networks,” in Design, Automation and Test in Europe
Conference and Exhibition (DATE), 2017, 2017, pp. 1474–1479.
[24] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2016, pp. 770–778.

[25] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, and J. Pineau,
“Towards the systematic reporting of the energy and carbon footprints
of machine learning,” Journal of Machine Learning Research, vol. 21,
no. 248, pp. 1–43, 2020.

[26] M. Hildebrand, J. Lowe-Power, and V. Akella, “Efficient large scale dlrm
implementation on heterogeneous memory systems,” in International
Conference on High Performance Computing. Springer, 2023, pp. 42–
61.

[27] A. Howard, “Mobilenets: Efficient convolu-tional neural networks for

mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[28] S. Hsia, A. Golden, B. Acun, N. Ardalani, Z. DeVito, G.-Y. Wei,
D. Brooks, and C.-J. Wu, “Mad-max beyond single-node: Enabling
large machine learning model acceleration on distributed systems,” in
2024 ACM/IEEE 51st Annual International Symposium on Computer
Architecture (ISCA), 2024, pp. 818–833.

[29] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio,
“Quantized neural networks: Training neural networks with low
precision weights and activations,” Journal of Machine Learning
Research, vol. 18, no. 187, pp. 1–30, 2018.
[Online]. Available:
http://jmlr.org/papers/v18/16-456.html

[30] N. C. Hudson,

J. G. Pauloski, M. Baughman, A. Kamatar,
M. Sakarvadia, L. Ward, R. Chard, A. Bauer, M. Levental, W. Wang,
W. Engler, O. Price Skelly, B. Blaiszik, R. Stevens, K. Chard,
serving infrastructure for
and I. Foster, “Trillion parameter ai
scientific discovery: A survey and vision,” in Proceedings of
the
IEEE/ACM 10th International Conference on Big Data Computing,
Applications and Technologies, ser. BDCAT ’23. New York, NY,
USA: Association for Computing Machinery, 2024. [Online]. Available:
https://doi.org/10.1145/3632366.3632396

[31] A. Imran, T. Kosar, J. Zola, and M. Bulut, “Towards sustainable
cloud software systems through energy-aware code smell refactoring,”
in 2024 IEEE 17th International Conference on Cloud Computing
(CLOUD). Los Alamitos, CA, USA: IEEE Computer Society,
jul
2024, pp. 223–234. [Online]. Available: https://doi.ieeecomputersociety.
org/10.1109/CLOUD62652.2024.00034

[32] V. Janapa Reddi, D. Kanter, P. Mattson, J. Duke, T. Nguyen, R. Chukka,
K. Shiring, K.-S. Tan, M. Charlebois, W. Chou et al., “Mlperf mobile
inference benchmark: An industry-standard open-source machine learn-
ing benchmark for on-device ai,” Proceedings of Machine Learning and
Systems, vol. 4, pp. 352–369, 2022.

[33] M. Johnson, M. Schuster, Q. V. Le, M. Krikun, Y. Wu, Z. Chen,
N. Thorat, F. Viégas, M. Wattenberg, G. Corrado, M. Hughes, and
J. Dean, “Google’s Multilingual Neural Machine Translation System:
Enabling Zero-Shot Translation,” Transactions of the Association for
Computational Linguistics, vol. 5, pp. 339–351, 10 2017. [Online].
Available: https://doi.org/10.1162/tacl_a_00065

[34] N. Jouppi, G. Kurian, S. Li, P. Ma, R. Nagarajan, L. Nai, N. Patil,
S. Subramanian, A. Swing, B. Towles, C. Young, X. Zhou, Z. Zhou, and
D. A. Patterson, “Tpu v4: An optically reconfigurable supercomputer
for machine
embeddings,”
in Proceedings of
International Symposium on
Computer Architecture,
ISCA ’23. New York, NY, USA:

learning with hardware

the 50th Annual

support

ser.

for

[Online]. Available:

[51] MLCommons,

“MLPerf

Inference Edge,” https://mlcommons.org/

Association for Computing Machinery, 2023.
https://doi.org/10.1145/3579371.3589350

[35] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa,
S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P.-l. Cantin,
C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V.
Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho,
D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski,
A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy,
J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin,
G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan,
R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick,
N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani,
C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing,
M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan,
R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter
performance analysis of a tensor processing unit,” SIGARCH Comput.
Archit. News, vol. 45, no. 2, p. 1–12, jun 2017. [Online]. Available:
https://doi.org/10.1145/3140659.3080246

[36] F. D. Keles, P. M. Wijewardena, and C. Hegde, “On the computational
complexity of self-attention,” in International Conference on Algorith-
mic Learning Theory. PMLR, 2023, pp. 597–619.

[37] I. Kindylidi and T. S. Cabral, “Sustainability of ai: The case of provision
of information to consumers,” Sustainability, vol. 13, no. 21, p. 12064,
2021.

[38] Y. Koizumi, Y. Kawaguchi, K. Imoto, T. Nakamura, Y. Nikaido,
R. Tanabe, H. Purohit, K. Suefusa, T. Endo, M. Yasuda, and
N. Harada, “Description and discussion on dcase2020 challenge
task2: Unsupervised anomalous sound detection for machine condition
monitoring,” 2020. [Online]. Available: https://arxiv.org/abs/2006.05822
[39] Y. Koizumi, S. Saito, H. Uematsu, N. Harada, and K. Imoto, “Toyadmos:
A dataset of miniature-machine operating sounds for anomalous sound
detection,” in 2019 IEEE Workshop on Applications of Signal Processing
to Audio and Acoustics (WASPAA), 2019, pp. 313–317.

[40] K.-D. Lange, “Identifying shades of green: The specpower benchmarks,”

Computer, vol. 42, no. 03, pp. 95–97, 2009.

[41] J. Lee and H.-J. Yoo, “An overview of energy-efficient hardware
accelerators for on-device deep-neural-network training,” IEEE Open
Journal of the Solid-State Circuits Society, vol. 1, pp. 115–128, 2021.
[42] B. Li, S. Samsi, V. Gadepally, and D. Tiwari, “Clover: Toward
sustainable ai with carbon-aware machine learning inference service,”
in SC23: International Conference for High Performance Computing,
Networking, Storage and Analysis.
Los Alamitos, CA, USA:
IEEE Computer Society, nov 2023, pp. 1–15. [Online]. Available:
https://doi.ieeecomputersociety.org/10.1145/3581784.3607034

[43] L. A. Libutti, F. D. Igual, L. Pinuel, L. De Giusti, and M. Naiouf,
“Benchmarking performance and power of usb accelerators for inference
with mlperf,” in Proc. 2nd Workshop Accelerated Mach. Learn.(AccML),
2020, pp. 1–15.

[44] S. Lim, Y. P. Liu, L. Benini, T. Karnik, and H.-C. Chang, “F1: Striking
the balance between energy efficiency and flexibility: General-purpose
vs special-purpose ml processors,” in 2021 IEEE International Solid-
State Circuits Conference (ISSCC), vol. 64.
IEEE, 2021, pp. 513–516.
[45] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss
for dense object detection,” in Proceedings of the IEEE international
conference on computer vision, 2017, pp. 2980–2988.

[46] P. Mattson, C. Cheng, G. Diamos, C. Coleman, P. Micikevicius,
D. Patterson, H. Tang, G.-Y. Wei, P. Bailis, V. Bittorf, D. Brooks,
D. Chen, D. Dutta, U. Gupta, K. Hazelwood, A. Hock, X. Huang,
D. Kang, D. Kanter, N. Kumar, J. Liao, D. Narayanan, T. Oguntebi,
G. Pekhimenko, L. Pentecost, V. Janapa Reddi, T. Robie, T. St John,
C.-J. Wu, L. Xu, C. Young, and M. Zaharia, “Mlperf
training
benchmark,” in Proceedings of Machine Learning and Systems,
I. Dhillon, D. Papailiopoulos, and V. Sze, Eds., vol. 2, 2020, pp.
336–349. [Online]. Available: https://proceedings.mlsys.org/paper_files/
paper/2020/file/411e39b117e885341f25efb8912945f7-Paper.pdf

[47] P. Mattson, V. J. Reddi, C. Cheng, C. Coleman, G. Diamos, D. Kanter,
P. Micikevicius, D. Patterson, G. Schmuelling, H. Tang, G.-Y. Wei, and
C.-J. Wu, “Mlperf: An industry standard benchmark suite for machine
learning performance,” IEEE Micro, vol. 40, no. 2, pp. 8–16, 2020.
[48] C. Minyard, “Ipmi–a gentle introduction with openipmi,” XP055165227,

Software Montavista, pp. 1–238, 2006.

[49] MLCommons, “MLCommons Website,” https://mlcommons.org.
[50] MLCommons, “MLPerf Inference Datacenter,” https://mlcommons.org/

benchmarks/inference-datacenter/.

benchmarks/inference-edge/.

[52] MLCommons,

“MLPerf

Inference

Rules,”

https://github.com/

mlcommons/inference_policies/blob/master/inference_rules.adoc#
closed-division.

[53] MLCommons,

“MLPerf

Inference Tiny,” https://mlcommons.org/

benchmarks/inference-tiny/.

[54] MLCommons, “MLPerf Inference Working Group,” https://mlcommons.

org/working-groups/benchmarks/inference/.

[55] MLCommons,

“MLPerf

Results

Messaging

Guidelines,”

https://github.com/mlcommons/policies/blob/master/MLPerf_Results_
Messaging_Guidelines.adoc.

[56] MLCommons, “MLPerf Training,” https://mlcommons.org/benchmarks/

training/.
[57] MLCommons,

“MLPerf

Training

Rules,”

https://github.com/

mlcommons/training_policies/blob/master/training_rules.adoc#11-
run-results.

[58] MLCommons,

“Power Working Group,”

https://mlcommons.org/

working-groups/benchmarks/power/.

[59] MLPerf, “Resnet in tensorflow,” https://github.com/mlperf/training/tree/

master/image_classification/tensorflow/official, 2019.

[60] R. Murmuria, J. Medsger, A. Stavrou, and J. M. Voas, “Mobile ap-
plication and device power usage measurements,” in 2012 IEEE Sixth
International Conference on Software Security and Reliability, 2012, pp.
147–156.

[61] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van
Baalen, and T. Blankevoort, “A white paper on neural network
quantization,” CoRR, vol. abs/2106.08295, 2021. [Online]. Available:
https://arxiv.org/abs/2106.08295

[62] M. Naumov, D. Mudigere, H. M. Shi, J. Huang, N. Sundaraman,
J. Park, X. Wang, U. Gupta, C. Wu, A. G. Azzolini, D. Dzhulgakov,
A. Mallevich, I. Cherniavskii, Y. Lu, R. Krishnamoorthi, A. Yu,
V. Kondratenko, S. Pereira, X. Chen, W. Chen, V. Rao, B. Jia,
L. Xiong, and M. Smelyanskiy, “Deep learning recommendation
model for personalization and recommendation systems,” CoRR, vol.
abs/1906.00091, 2019. [Online]. Available: http://arxiv.org/abs/1906.
00091

[63] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng,
“Multimodal deep learning,” in Proceedings of the 28th international
conference on machine learning (ICML-11), 2011, pp. 689–696.

[64] D. Patterson,

J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia,
D. Rothchild, D. So, M. Texier, and J. Dean, “Carbon emissions and
large neural network training,” arXiv preprint arXiv:2104.10350, 2021.
[65] S. Prakash, M. Stewart, C. Banbury, M. Mazumder, P. Warden,
B. Plancher, and V. J. Reddi, “Is tinyml sustainable?” Communications
of the ACM, vol. 66, no. 11, pp. 68–77, 2023.

[66] P. K. D. Pramanik, N. Sinhababu, B. Mukherjee, S. Padmanaban,
A. Maity, B. K. Upadhyaya, J. B. Holm-Nielsen, and P. Choudhury,
“Power consumption analysis, measurement, management, and issues:
A state-of-the-art review of smartphone battery and energy usage,” ieee
Access, vol. 7, pp. 182 113–182 172, 2019.

[67] H. Purohit, R. Tanabe, K. Ichige, T. Endo, Y. Nikaido, K. Suefusa,
and Y. Kawaguchi, “Mimii dataset: Sound dataset for malfunction-
ing industrial machine investigation and inspection,” arXiv preprint
arXiv:1909.09347, 2019.

[68] D. Ramachandram and G. W. Taylor, “Deep multimodal

learning:
A survey on recent advances and trends,” IEEE signal processing
magazine, vol. 34, no. 6, pp. 96–108, 2017.

[69] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, G. Schmuelling, and
C.-J. Wu, “The vision behind mlperf: Understanding ai inference per-
formance,” IEEE Micro, vol. 41, no. 3, pp. 10–18, 2021.

[70] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, G. Schmuelling, C.-J.
Wu, B. Anderson, M. Breughe, M. Charlebois, W. Chou, R. Chukka,
C. Coleman, S. Davis, P. Deng, G. Diamos, J. Duke, D. Fick, J. S.
Gardner, I. Hubara, S. Idgunji, T. B. Jablin, J. Jiao, T. S. John, P. Kanwar,
D. Lee, J. Liao, A. Lokhmotov, F. Massa, P. Meng, P. Micikevicius,
C. Osborne, G. Pekhimenko, A. T. R. Rajan, D. Sequeira, A. Sirasao,
F. Sun, H. Tang, M. Thomson, F. Wei, E. Wu, L. Xu, K. Yamada,
B. Yu, G. Yuan, A. Zhong, P. Zhang, and Y. Zhou, “Mlperf inference
benchmark,” in 2020 ACM/IEEE 47th Annual International Symposium
on Computer Architecture (ISCA), 2020, pp. 446–459.

[71] A. Reuther, P. Michaleas, M. Jones, V. Gadepally, S. Samsi, and
J. Kepner, “Survey and benchmarking of machine learning accelerators,”

in 2019 IEEE high performance extreme computing conference (HPEC).
IEEE, 2019, pp. 1–9.

USENIX Symposium on Networked Systems Design and Implementation
(NSDI 23), 2023, pp. 119–139.

[87] Y. Zhang, N. Suda, L. Lai, and V. Chandra, “Hello edge: Keyword
spotting on microcontrollers,” arXiv preprint arXiv:1711.07128, 2017.
[88] J. Zhao and B. Gómez Fariñas, “Artificial intelligence and sustainable
decisions,” European Business Organization Law Review, vol. 24, no. 1,
pp. 1–39, 2023.

[89] H. Zhu, M.-D. Luo, R. Wang, A.-H. Zheng, and R. He, “Deep audio-
visual learning: A survey,” International Journal of Automation and
Computing, vol. 18, no. 3, pp. 351–376, 2021.

[72] M. C. Rillig, M. Ågerstrand, M. Bi, K. A. Gould, and U. Sauerland,
“Risks and benefits of large language models for the environment,”
Environmental Science and Technology, vol. 57, no. 9, pp. 3464–3466,
2023.

[73] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-
resolution image synthesis with latent diffusion models,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 10 684–10 695.

[74] F. Romero, Q. Li, N. J. Yadwadkar, and C. Kozyrakis, “{INFaaS}:
Automated model-less inference serving,” in 2021 USENIX Annual
Technical Conference (USENIX ATC 21), 2021, pp. 397–411.

[75] S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones,
W. Bergeron, J. Kepner, D. Tiwari, and V. Gadepally, “From words
to watts: Benchmarking the energy costs of large language model infer-
ence,” in 2023 IEEE High Performance Extreme Computing Conference
(HPEC).

IEEE, 2023, pp. 1–9.

[76] J. Sevilla, L. Heim, A. Ho, T. Besiroglu, M. Hobbhahn, and P. Villalobos,
“Compute trends across three eras of machine learning,” in 2022
International Joint Conference on Neural Networks (IJCNN).
IEEE,
2022, pp. 1–8.

[77] Z. Shen, Z. He, and X. Xue, “Meal: Multi-model ensemble via ad-
versarial learning,” in Proceedings of the AAAI conference on artificial
intelligence, vol. 33, no. 01, 2019, pp. 4886–4893.

[78] J. Stojkovic, E. Choukse, C. Zhang, I. Goiri, and J. Torrellas, “Towards
greener llms: Bringing energy-efficiency to the forefront of llm infer-
ence,” arXiv preprint arXiv:2403.20306, 2024.

[79] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher,
C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu,
W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn,
S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa,
I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril,
J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov,
P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta,
K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian,
X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu,
Z. Yan,
I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang,
A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, “Llama 2: Open
foundation and fine-tuned chat models,” 2023. [Online]. Available:
https://arxiv.org/abs/2307.09288

[80] A. Vahdat, X. Ma, and D. Patterson, “New computer evaluation metrics

for a changing world,” Communications of the ACM, 2024.

[81] Y. Wang, Q. Wang, S. Shi, X. He, Z. Tang, K. Zhao, and X. Chu,
“Benchmarking the performance and energy efficiency of ai accelerators
for ai training,” in 2020 20th IEEE/ACM International Symposium on
Cluster, Cloud and Internet Computing (CCGRID).
IEEE, 2020, pp.
744–751.

[82] C.-J. Wu, R. Raghavendra, U. Gupta, B. Acun, N. Ardalani, K. Maeng,
G. Chang, F. Aga, J. Huang, C. Bai, M. Gschwind, A. Gupta, M. Ott,
A. Melnikov, S. Candido, D. Brooks, G. Chauhan, B. Lee, H.-H.
Lee, B. Akyildiz, M. Balandat, J. Spisak, R. Jain, M. Rabbat, and
K. Hazelwood, “Sustainable ai: Environmental implications, challenges
and opportunities,” in Proceedings of Machine Learning and Systems,
D. Marculescu, Y. Chi, and C. Wu, Eds., vol. 4, 2022, pp.
795–813. [Online]. Available: https://proceedings.mlsys.org/paper_files/
paper/2022/file/462211f67c7d858f663355eff93b745e-Paper.pdf

[83] M. Xiangchun, Y. Lingling, Z. Ming, and C. Jingchun, “Carbon
emissions trading and sustainable development of power industry,”
in Electrical and Control Engineering, International Conference on.
Los Alamitos, CA, USA: IEEE Computer Society,
jun 2010, pp.
3443–3446. [Online]. Available: https://doi.ieeecomputersociety.org/10.
1109/iCECE.2010.839

[84] Y. Xiao, J. Wu, Z. Lin, and X. Zhao, “A deep learning-based multi-
model ensemble method for cancer prediction,” Computer methods and
programs in biomedicine, vol. 153, pp. 1–9, 2018.

[85] J. Xing, B. Acun, A. Sundarrajan, D. Brooks, M. Chakkaravarthy,
N. Avila, C.-J. Wu, and B. C. Lee, “Carbon responder: Coordi-
the datacenter fleet,” arXiv preprint
nating demand response for
arXiv:2311.08589, 2023.

[86] J. You, J.-W. Chung, and M. Chowdhury, “Zeus: Understanding and
optimizing {GPU} energy consumption of {DNN} training,” in 20th

