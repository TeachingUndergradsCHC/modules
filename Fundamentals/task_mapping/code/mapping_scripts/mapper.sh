#!/bin/bash
#
#  Script to map workloads to heterogeneous cores
#
#  @author: Apan Qasem <apan@txstate.edu>
#  @date: 04/02/20 
# 
# @update: 03/12/21
#

function usage() {
/bin/cat 2>&1 << "EOF"

    Run hybrid workloads on heterogeneous cores. 

    Usage: mapper [options] workloads ... cores ... 

        workloads are specified with "wki". For this installation, i is between 0-3 
        cores are specified using integers. On this systems available cores are 0-3
     
        number of workload and core arguments must match 
				executable workloads must be available in the user's path 

    Options: 
      --help          print this help message     
      --parallel      run workloads in parallel 
EOF
exit 1
}


if [ "$1" = "--help" ] || [ $# -lt 1 ]; then
	usage 
fi		

if [ "$1" = "--parallel" ]; then
	 par=true
	 shift 
fi		

function check_arg_wkld() {
	wkld=$1
	[ $wkld == "wk0" ] || 	
	[ $wkld == "wk1" ] ||
	[ $wkld == "wk2" ] || 
	[ $wkld == "wk3" ] ||
	{ echo "workload not recognized: $wkld"; exit 0; }

	[ -e ./$wkld ] || { echo "did not find workload in path: $wkld"; exit 0; }
	return
}
										

i=0
j=0


while [ $# -gt 0 ]; do
	arg="$1"

	# check if arg is workload (wk0, ...) or core (integer)
	if ! [ "$arg" -eq "$arg" ] 2> /dev/null ; then
			check_arg_wkld $arg
			wklds[$i]=$arg
			i=$(($i+1))
	else
		cores[$j]=$arg
		j=$(($j+1))
  fi
	shift
done 

[ $i -gt 0 ] || { echo "no workload speficied"; exit 0;}
[ $j == $i ] || { echo "number of workloads and cores don't match"; exit 0;}


#
# Heterogeneous System Simulation
# mapper assumes target system has at least 4 cores
# and it has already been set up to simulate a heterogeneous system
# See ToUCH scripts available at https://github.com/TeachingUndergradsCHC/modules.git
# for instructions on simulating a heterogeneous system on a multicore heterogeneous system
#
hetero_cores[0]="0-1"   # "little" cores; cores running at minimum frequency 
hetero_cores[1]="6-7"   # "big" cores 
hetero_cores[2]="3,11"  # hyper-threaded cores with larger cache capacity
hetero_cores[3]="4-5"   # gpu 

#
# Workloads 
#
big=0        # FP compute-intensive code; best performance achieved when run on "big" cores
little=1     # minimal computationa; performance is least affected by core frequency
gpu=2        # GPU workload; will offload to GPU only when the simulated GPU core is specified;
             # otherwise runs on CPU; best performance achieved when run on GPU
cache=3      # code with inter-thread data locality; best performance is attained when mapped to hyperthreaded cores sh             # sharing the same L1 and L2 cache

wkld_args[$big]="2000000 2"  
wkld_args[$little]="5000 20 2" 
wkld_args[$gpu]="100000000" 
wkld_args[$cache]="3000 2" 

# timer code from stackoverflow [https://stackoverflow.com/questions/42356299/bash-timer-in-milliseconds]
start_at=$(date +%s,%N)
_s1=$(echo $start_at | cut -d',' -f1)   # sec
_s2=$(echo $start_at | cut -d',' -f2)   # nano sec

start=$SECONDS
for ((k=0; $k < $i; k++)); do
	length=${#wklds[$k]}
	length=$(($length-1))

	wkld_index=${wklds[$k]:$length}
  assigned_core=${hetero_cores[${cores[$k]}]} 
	args=${wkld_args[${wkld_index}]}
	
	if [ ${wklds[$k]} = "wk2" ]; then   # GPU 
			if [ ${cores[$k]} = "3" ]; then
					args=${wkld_args[${wkld_index}]}" 1"
			else
					args=${wkld_args[${wkld_index}]}" 0"
			fi
	fi
	echo "Launching workload ${wklds[$k]} on ${assigned_core} ..."
	if [ "$par" ]; then  
			taskset -c ${assigned_core} ./${wklds[$k]} $args &
	else
			taskset -c ${assigned_core} ./${wklds[$k]} $args
	fi
done

# if running the workloads in parallel, wait for all of them to finish
if [ "$par" ]; then 
		wait
fi
end_at=$(date +%s,%N)
_e1=$(echo $end_at | cut -d',' -f1)
_e2=$(echo $end_at | cut -d',' -f2)
time=$(bc <<< "scale=3; $_e1 - $_s1 + ($_e2 -$_s2)/1000000000")

echo "All workloads completed in ${time} seconds."
