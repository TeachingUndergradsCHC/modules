# Multi-tasking Heterogeneous Computers 

## Description
Heterogeneous computers are particularly good at performing a variety of tasks concurrently. But in
order for them to do a good job, the software must ensure that tasks are appropriately distributed to the specialized
cores.

In this assignment, you will investigate performance (and energy) issues of a heterogeneous
computing system. You will be given a set of four programs with different characteristics. Your goal
is to determine the best mapping of these programs to the different processing cores via
experimentation and analysis.  


## Environment

You will be running experiments on `megatron`, a heterogeneous multicore
system. megatron has four processing cores and each core has been configured to do a specific type
of job. Although each core can do any type of computation it will perform certain tasks really
well.

## Tools

Familiarize yourself with the following tools. 

  * `perf`
  * `taskset`
  * `likwid`
  * `cpufreutils`
  * `cpupower`

## Instructions

### 1. Log in to megatron.

megatron is a server behind the firewall. From within the Texas State network, you can ssh into
megatron as follows

     ssh netid@megatron.cs.txstate.edu

From an off-campus network, you will first need to ssh into a gateway server (e.g.,
zeus.cs.txstate.edu) and then ssh into megatron. 


### 2. Download code samples.

After you logged into megatron, clone the following git repository into your home directory 

     git clone https://git.txstate.edu/aq10/hetero/assg1.git`

Create a directory for the codes to reside and unzip the codes into that directory. You should see
four executables and a README. The four executables are designed to perform the following tasks

  * p0: numeric computation (e.g., excel)
  * p1: graphics (e.g., game)
  * p2: play music (e.g, music app)
  * p3: communicate with the internet (e.g., web browser) 

The README has more information about each application and their characteristics. 

### 3. Conduct Performance Experiments

Launch the four programs, at the same time, with different thread mapping configurations. You can do this in one step using the mapper tool (installed in
`/usr/local/bin/mapper`). For example, 

     /usr/local/bin/mapper p0 p1 p3 p4 3 1 0 2

The above command will launch the four programs at the same time and map `p0`, `p1`, `p3`, `p4` to
processing cores `3`, `1`, `0` and `2` respectively. The program arguments must be the fully qualified name
of the executable and the processor arguments must be in the range 0-3. Type the following to see
more options

     /usr/local/bin/mapper --help 

For each configuration, record the performance of the individual cores and the overall workload. You
can use the perf tool for this purpose.

     perf stat /usr/local/bin/mapper p0 p1 p3 p4 3 1 0 2

perf will report a bunch of performance metrics. The ones that you want to pay particular attention
to are `CPUs Utilized` and instructions for cycle. Instructions per cycle (IPC) is a throughput
metric that normalizes performance across different workloads. 

Repeat the experiments and measure the energy consumption. You can use likwid (located in
/usr/local/bin/likwid) to do this

     likwid-perctr -c 0-3 -g ENERGY /usr/local/bin/mapper p0 p1 p3 p4 3 1 0 2

### 4. Analyze the data

Create charts showing performance (as measured using the metrics described above), power and energy
for different configurations. Analyze the data and create a report answering the following questions

* Which processor is good numeric computation?
* Which processor is good at graphics?
* Which processor is good at playing music?
* Which processor is good when there is a need to communicate over the network?
* Do the answers hold for power as well?
* What is the configuration the provides the best performance?
* What is the configuration that consumes least power?
* What is the configurations that is most energy efficient?

