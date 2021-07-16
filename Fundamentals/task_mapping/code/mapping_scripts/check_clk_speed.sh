#!/bin/bash

cpufreq-info | grep "analyzing CPU\|policy" > /tmp/freqs
cat /tmp/freqs | grep "CPU\|policy" | sed 's/\.//g' | awk '{if ($1 == "analyzing") printf $2 $3; else printf "\t" $7 " " $8 " - " $10" "$11 "\n"}'



