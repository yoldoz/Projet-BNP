# PIC-PROJ
Projet BNP

## To download the results of SROIE
We put the results on google drive. You can download these result by using the following link.

* https://drive.google.com/file/d/12mwaHgljXO1QLDxycitsQwuLxaRX53Gd/view?usp=sharing

## Training
~~~bash
python main.py  --data_root './data_loading/funsd' \
                --dataset 'funsd' \
                --model 'LayoutLM' \
                --test_only False \
                --num_train_epochs 5 \
                --lr 5e-5 \
                --batch_size 2 \
                --val_batch_size 1 \
                --gpu_id 0 \
                > train_funsd.out
~~~

## Evaluation
~~~bash
python main.py  --data_root './data_loading/funsd' \
                --dataset 'funsd' \
                --model 'LayoutLM \
                --test_only True \
                --val_batch_size 1 \
                --gpu_id 0 \
                > test_funsd.out
~~~
