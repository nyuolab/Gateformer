export CUDA_VISIBLE_DEVICES=0

model_name=Gateformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 64 \
  --d_ff 64 \
  --learning_rate 0.0001 \
  --n_heads 4 \
  > "exchange_rate_96_out.txt" 2> "exchange_rate_96_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 64 \
  --d_ff 64 \
  --learning_rate 0.001 \
  --n_heads 4 \
  > "exchange_rate_192_out.txt" 2> "exchange_rate_192_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 64 \
  --d_ff 64 \
  --learning_rate 0.0001 \
  --n_heads 4 \
  > "exchange_rate_336_out.txt" 2> "exchange_rate_336_err.txt"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 8 \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --n_heads 4 \
  > "exchange_rate_720_out.txt" 2> "exchange_rate_720_err.txt"