python main.py  --data_root './data_loading/FUNSD' \
                --dataset 'funsd' \
                --model 'LayoutLM' \
                --num_train_epochs 5 \
                --lr 5e-5 \
                --batch_size 2 \
                --val_batch_size 1 \
                --gpu_id 0 \
                > train_funsd.out
