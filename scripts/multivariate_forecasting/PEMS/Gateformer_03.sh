export CUDA_VISIBLE_DEVICES=0

model_name=Gateformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_3 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS03_3_out.txt" 2> "PEMS03_3_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_6 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS03_6_out.txt" 2> "PEMS03_6_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS03_12_out.txt" 2> "PEMS03_12_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS03_24_out.txt" 2> "PEMS03_24_err.txt"