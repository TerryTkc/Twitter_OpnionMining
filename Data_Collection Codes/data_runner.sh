#! /bin/bash

num=$1
count=0


while true ; do
    echo "count: $count"
    script=$(eval "python3 Cities.py")
    echo $script
    echo
    count=$(($count+1))
    sleep 1800s
done
