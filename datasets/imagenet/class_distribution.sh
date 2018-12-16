#!/bin/bash

for d0 in $(ls); do
    echo "${d0} $(ls ${d0} | wc -l)" >> class_distribution.txt
done;
