lora_model_save_path='/root/paddlejob/workspace/env_run/output/Qwen2.5-7B/repllama'
model_path='/root/paddlejob/workspace/env_run/model/Qwen2.5-7B'
corpus_path='/root/paddlejob/workspace/env_run/data/msmarco-pass/corpus.jsonl'
train_path="/root/paddlejob/workspace/env_run/data/msmarco-pass/repllama-train-tevatron.jsonl"
dev_query_path='/root/paddlejob/workspace/env_run/data/msmarco-pass/dev.jsonl'

dev_qrel_path='/root/paddlejob/workspace/env_run/data/msmarco-pass/qrels.dev.tsv'

encode_path=$lora_model_save_path'/encode'
mkdir -p $lora_model_save_path
mkdir $encode_path
q_max_len=32
p_max_len=156
echo $model_path
query_prefix="Query: "
passage_prefix="Passage: "
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60001 --module tevatron.retriever.driver.train \
  --deepspeed /root/paddlejob/workspace/env_run/output/tevatron-main/deepspeed/ds_zero3_config.json \
  --output_dir  $lora_model_save_path \
  --model_name_or_path $model_path \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 200 \
  --lora_r 32 \
  --dataset_path $train_path \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_prefix $query_prefix \
  --passage_prefix $passage_prefix \
  --query_max_len $q_max_len \
  --passage_max_len $p_max_len \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 4
sleep 10s
rm -rf $lora_model_save_path'/checkpoint-'*

echo "====== encode query"

CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $lora_model_save_path \
  --normalize \
  --encode_is_query \
  --query_prefix $query_prefix \
  --passage_prefix $passage_prefix \
  --fp16 \
  --per_device_eval_batch_size 100 \
  --passage_max_len $p_max_len \
  --pooling eos \
  --append_eos_token \
  --query_max_len $q_max_len \
  --dataset_path $dev_query_path \
  --encode_output_path $encode_path/dev_query_emb.pkl

sleep 10s
echo "====== encode corpus"

for s in 0 1 2 3 4 5 6 7;
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $lora_model_save_path \
  --normalize \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --pooling eos \
  --query_prefix $query_prefix \
  --passage_prefix $passage_prefix \
  --append_eos_token \
  --passage_max_len $p_max_len \
  --dataset_path $corpus_path \
  --query_max_len $q_max_len \
  --dataset_number_of_shards 8 \
  --dataset_shard_index ${s} \
  --encode_output_path $encode_path/corpus_emb.${s}.pkl   &
  # 等待最后一个循环完成
  if [ "$s" == "7" ]; then
      wait
  fi
done
sleep 10s
echo "====== Search the Corpus"
set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
--query_reps $encode_path/dev_query_emb.pkl \
--passage_reps $encode_path'/corpus_emb.*.pkl' \
--depth 1000 \
--batch_size 1024 \
--save_text \
--save_ranking_to $encode_path/dev_rank.txt
    
echo "====== Evaluation"
python tevatron/utils/format/convert_result_to_trec.py \
--input $encode_path/dev_rank.txt \
--output $encode_path/dev_rank.txt.trec

./trec_eval -m recall $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.50 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.1000 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recip_rank.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.20 $dev_qrel_path $encode_path/dev_rank.txt.trec
python trec_eval.py --qrel_file $dev_qrel_path  --run_file $encode_path/dev_rank.txt.trec



encode_path=$lora_model_save_path'/encode_trec-19'
mkdir $encode_path
dev_query_path='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/msmarco-test2019-queries.json'
dev_qrel_path='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/2019qrels-pass.txt'

echo "====== encode query"
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $lora_model_save_path \
  --normalize \
  --encode_is_query \
  --query_prefix $query_prefix \
  --passage_prefix $passage_prefix \
  --fp16 \
  --per_device_eval_batch_size 100 \
  --passage_max_len $p_max_len \
  --pooling eos \
  --append_eos_token \
  --query_max_len $q_max_len \
  --dataset_path $dev_query_path \
  --encode_output_path $encode_path/dev_query_emb.pkl

echo "====== Search the Corpus"
set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
--query_reps $encode_path/dev_query_emb.pkl \
--passage_reps $lora_model_save_path'/encode/corpus_emb.*.pkl' \
--depth 1000 \
--batch_size 512 \
--save_text \
--save_ranking_to $encode_path/dev_rank.txt
    
echo "====== Evaluation"
python tevatron/utils/format/convert_result_to_trec.py \
--input $encode_path/dev_rank.txt \
--output $encode_path/dev_rank.txt.trec

./trec_eval -m recall $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.50 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.1000 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recip_rank.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.20 $dev_qrel_path $encode_path/dev_rank.txt.trec




encode_path=$lora_model_save_path'/encode_trec20'
mkdir $encode_path
dev_query_path='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/msmarco-test2020-queries.json'
dev_qrel_path='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/2020qrels-pass.txt'

echo "====== encode query"
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $lora_model_save_path \
  --normalize \
  --encode_is_query \
  --query_prefix $query_prefix \
  --passage_prefix $passage_prefix \
  --fp16 \
  --per_device_eval_batch_size 100 \
  --passage_max_len $p_max_len \
  --pooling eos \
  --append_eos_token \
  --query_max_len $q_max_len \
  --dataset_path $dev_query_path \
  --encode_output_path $encode_path/dev_query_emb.pkl

echo "====== Search the Corpus"
set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
--query_reps $encode_path/dev_query_emb.pkl \
--passage_reps $lora_model_save_path'/encode/corpus_emb.*.pkl' \
--depth 1000 \
--batch_size 512 \
--save_text \
--save_ranking_to $encode_path/dev_rank.txt
    
echo "====== Evaluation"
python tevatron/utils/format/convert_result_to_trec.py \
--input $encode_path/dev_rank.txt \
--output $encode_path/dev_rank.txt.trec

./trec_eval -m recall $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.50 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recall.1000 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m recip_rank.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
./trec_eval -m ndcg_cut.20 $dev_qrel_path $encode_path/dev_rank.txt.trec
sh Qwen2.5-7b-instruct.sh > Qwen2.5-7b-instruct.log
sh Qwen2.5-coder-7B-instruct.sh > Qwen2.5-coder-7B-instruct.log
sh Qwen2.5-coder-7B.sh > Qwen2.5-coder-7B.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /root/paddlejob/workspace/env_run/run.py --size 40000 --gpus 8 --interval 0.0
