#!/bin/bash
mkdir -p res
mkdir -p chan
mkdir -p sa

zpp_task(){

if [ ! -f "res/$1.res" ]
then
	network -ha -res res/$1.res $2
fi

if [ ! -f "chan/$1.chan" ]
then
	network -ha -chan 1.5 chan/$1.chan $2
fi

if [ ! -f "sa/$1.sa" ]
then
	network -ha -sa 1.2 1.2 2000 sa/$1.sa $2
fi
}


open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=100
open_sem $N
for cssr_file in cssr/*.cssr
do
    filename=$(basename "$cssr_file" .cssr)
    run_with_lock zpp_task $filename $cssr_file
done 


# # test `zpp_task`
# for cssr_file in ../data/cssr/*.cssr
# do
# 	filename=$(basename "$cssr_file" .cssr)
# 	zpp_task $filename $cssr_file
# done
