#!/bin/bash
cp /dev/null jobsRunning.txt
cp /dev/null jobsCompleted.txt

#clone git repo
rm -rf ./cloned_repo
mkdir cloned_repo
git clone .. ./cloned_repo
a=$1
b=$2
for i in $(seq $a $b) ; do
    echo "screen -dmS session$i ./clusterClient.pl -d ."
    screen -dmS session$i ./clusterClient.pl -d . 
done
