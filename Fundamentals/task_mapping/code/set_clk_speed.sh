#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
Usage: ./set_clk_speed.sh [OPTIONS] <cpulist> <frequency>
  
  set CPU freqeuncies to specified values 

  <cpulist> is a list of CPU IDs. ID can be sepated by commas;  
  consecutive range of CPU IDs can be specified using a dash
  
	<frequency> floating-point val expressed in GHZ (e.g., 3.33)

Options:
   --help   print this help message

Options with values: 
 
EOF
exit 1
}   

if [ $# -ne 2 ] || [ $1 == "--help" ]; then
		usage
fi

if [ `whoami` != "root" ]; then
		echo "Need root privileges to set core frequencies"
		exit 0
fi

cpu=$1
user_freq=$2

freq=`echo ${user_freq} | awk '{ print $1 * 1000000}'`

if [ $DEBUG ]; then 
		echo $cpu
		echo $user_freq
		echo $freq
fi

# check frequency limits of CPU
lb=`cpufreq-info -c ${cpu} | grep limits | awk '{print $3 * 1000000}'`
ub=`cpufreq-info -c ${cpu} | grep limits | awk '{print $6 * 1000000}'`

if [ $freq -lt $lb ] || [ $freq -gt $ub ]; then
		echo "Illegal frequecy setting " ${user_freq} ". Not supported by hardware"
		exit 0
fi

# set frequency
# both upper and lower bounds set to same frequency 
cpupower -c $cpu frequency-set -u $freq
cpupower -c $cpu frequency-set -d $freq
