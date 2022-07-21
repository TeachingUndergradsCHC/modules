#!/bin/bash 

#
# @author: Apan Qasem
# @date: 07/03/20
#
# This script creates a Heterogeneous Computing (HC) environment within a homogenous multicore system.
# Script takes advantage of software-driven DVFS avaialble on modern Linux distrubutions
# Script must be run in sudo mode since frequency setting is only available to root
#
# TODO
#   - generalize to N available DVFS settings
#   - add other configrations for three-state case
#

function usage() {
/bin/cat 2>&1 <<"EOF"                 

Usage:  build_hc_env.sh [ OPTIONS ]         

Options: 
   --help          print this help message                                 
   --restore       restore original state of system              
EOF
  exit 0
}

function set_hc_mode() {

	cpus=$1
	shift
	states=$@

	# copy frequency states in an array 
	i=0
	for s in $states; do
		freqs[$i]=$s
		i=$(($i+1))
	done
	num_states=$i

	# find max and min states 
	max=${freqs[0]}
	min=${freqs[$(($num_states - 1))]}

	# set all governors userspace 
	for (( i = 0; i < $cpus; i++ )); do  
		cpufreq-set -c $i -g userspace
	done


  # three-state case 
	if [ ${num_states} -eq 3 ]; then 
		# Configuration: binary-alternate  
		#    CPU0,CPU2,... = max
		#    CPU1,CPU3,... = min
		for (( i = 0; i < $cpus; i++ )); do
			if [ $((i % 2)) -eq 0 ]; then
				cpufreq-set -c $i -f $max
			else 
				cpufreq-set -c $i -f $min
			fi
		done
	fi
}

#
# restore default settings for CPU cores
#
function restore() {
	for (( i = 0; i < $cpus; i++ )); do  
		cpufreq-set -c $i -g ondemand
	done
}

if [ "$1" = "--help" ]; then
   usage
fi

while [ $# -gt 0 ]; do
	arg="$1"
	case $arg in 
		-h|--help)
			usage
			exit 0
			;;
		-r|--restore)
			restore=true
			;;
		*)
			echo Unknow option: $arg
			usage
			exit 0
			;;
	esac
	shift
done 


# check if needed utilites are available 
[ `which cpufreq-info` ] || { echo "cpufreq-info not installed"; exit 0;  }

# should be there if cpufreq-info is there, but still ...
[ `which cpufreq-set` ] || { echo "cpufreq-set not installed"; exit 0;  }

# check if script is being invoked in sudo mode, bail otherwise 
[ `whoami` == "root" ] || { echo "must run this script with sudo"; exit 0; }

#
# get frequency info on this system 
#
freq_data=/tmp/freq_data.txt
cpufreq-info > ${freq_data}

# number of controllable CPU cores in the system 
cpus=`cat ${freq_data} | grep "analyzing CPU" | wc -l`

if [ "$restore" ]; then 
	# restore default settings and return 
	restore $cpus
	exit 0
fi

# available frequency states (assuming same number of states available on all cores)
states=`cat ${freq_data} | grep "available frequency steps" | awk -F ":" '{print $2}' | head -1` 
states=`echo $states | sed 's/\ G/G/g' | awk -F "," '{print $1 $2 $3}'`
states=`echo $states | sed 's/\ G/G/g' | awk -F "," '{ for(i=1;i<=NF;i++) print $i}'`

if [ $verbose ]; then 
	echo $states
fi

# make sure userspace governor is avaialble on all cpus
user_governors=`cat ${freq_data} | grep "governors" | grep "userspace" | wc -l`
if [ $cpus != ${user_governors} ]; then 
	echo "userspace governor is not available on all cpu"
	exit 0
fi

# save old policy 
for (( i = 0; i < $cpus; i++ )); do  
	old_policy[$i]=`cpufreq-info -c $i --policy | awk '{print $NF}'`
	old_freq[$i]=`cpufreq-info -c $i --freq | awk '{print $NF}'`
	old_min_max[$i]=`cpufreq-info -c 0 -l`
done

echo "foo" > /var/my_saved_file 

if [ $verbose ]; then
	for (( i = 0; i < $cpus; i++ )); do  
		echo ${old_policy[$i]} ":" ${old_freq[$i]}
	done
fi

# set HC mode 
set_hc_mode $cpus ${states}

# clean up 
rm -rf ${freq_data}
