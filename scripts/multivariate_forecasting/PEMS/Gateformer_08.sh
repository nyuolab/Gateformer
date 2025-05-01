export CUDA_VISIBLE_DEVICES=0

model_name=Gateformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_3 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS08_3_out.txt" 2> "PEMS08_3_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_6 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS08_6_out.txt" 2> "PEMS08_6_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS08_12_out.txt" 2> "PEMS08_12_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "PEMS08_24_out.txt" 2> "PEMS08_24_err.txt"