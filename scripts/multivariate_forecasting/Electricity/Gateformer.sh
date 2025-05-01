export CUDA_VISIBLE_DEVICES=0

model_name=Gateformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "electricity_96_out.txt" 2> "electricity_96_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 1024 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "electricity_192_out.txt" 2> "electricity_192_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 1024 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "electricity_336_out.txt" 2> "electricity_336_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 512 \
  --d_ff 1024 \
  --learning_rate 0.0005 \
  --n_heads 8 \
  > "electricity_720_out.txt" 2> "electricity_720_err.txt"