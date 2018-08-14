#!/bin/bash
a=$1
b=$2
for i in $(seq $a $b) ; do
    echo "screen -X -S session$i quit"
    screen -X -S session$i quit
done
