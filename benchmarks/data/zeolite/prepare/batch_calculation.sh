#!/bin/bash
mkdir -p res
mkdir -p chan
mkdir -p sa

for cssr_file in ../data/cssr/*.cssr
do
	filename=$(basename "$cssr_file" .cssr)
	network -ha -res res/$filename.res $cssr_file
	network -ha -chan 1.5 chan/$filename.chan $cssr_file
	network -ha -sa 1.2 1.2 2000 sa/$filename.sa $cssr_file
done
