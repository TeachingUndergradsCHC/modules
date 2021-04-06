#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
Usage: ./reset_clk_speed.sh [OPTIONS] <cpulist> 
  
  reset CPU freqeuncies to default values (wrapper around cpupower

  <cpulist> is a list of CPU IDs. ID can be sepated by commas;  
  consecutive range of CPU IDs can be specified using a dash
  
Options:
   --help   print this help message

Options with values: 
 
EOF
exit 1
}   

if [ $# -ne 1 ] || [ $1 == "--help" ]; then
		usage
fi

if [ `whoami` != "root" ]; then
		echo "Need root privileges to reset core frequencies"
		exit 0
fi

cpu=$1

freq=`echo ${user_freq} | awk '{ print $1 * 1000000}'`

if [ $DEBUG ]; then 
		echo $cpu
		echo $user_freq
		echo $freq
fi

# get limits 
lb=`cpufreq-info -c ${cpu} | grep limits | awk '{print $3 * 1000000}'`
ub=`cpufreq-info -c ${cpu} | grep limits | awk '{print $6 * 1000000}'`


# restore defaults 
cpupower -c $cpu frequency-set -d $lb &> /dev/null
cpupower -c $cpu frequency-set -u $ub
