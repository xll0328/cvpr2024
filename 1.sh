#!/bin/bash
n=1
mv ./models/multiTask/SELF_MM_model2.py ./models/multiTask/SELF_MM.py
echo $n
python run.py --num=1 --a1=0.5 --a2=1.5 --num_workers=12 --emb_size=9> results/test1_$n.log 2>&1 
((n=n+1))
echo $n
python run.py --num=1 --a1=0.5 --a2=1.5 --num_workers=12 --emb_size=11> results/test1_$n.log 2>&1 
((n=n+1))
echo $n
mv ./models/multiTask/SELF_MM.py ./models/multiTask/SELF_MM_model2.py
mv ./models/multiTask/SELF_MM_model3.py ./models/multiTask/SELF_MM.py
python run.py --num=1 --a1=0.5 --a2=1.5 --num_workers=12 --emb_size=9> results/test1_$n.log 2>&1 
((n=n+1))
echo $n
python run.py --num=1 --a1=0.5 --a2=1.5 --num_workers=4 --emb_size=9> results/test2_1.log 2>&1 
shutdown -h now