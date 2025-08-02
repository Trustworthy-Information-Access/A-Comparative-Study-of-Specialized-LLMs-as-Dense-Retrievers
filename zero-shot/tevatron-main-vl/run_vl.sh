for model in   "Qwen2.5-VL-7B-Instruct" 
do
    echo $model
    model_name_or_path='/root/paddlejob/workspace/env_run/model/'$model
    # for data_name in  "trec-covid-beir"  "nfcorpus" "quora"   "arguana";
    # do
    for data_name in   "fever" "nq" "hotpotqa" "climate-fever" "dbpedia-entity"  "scifact" "webis-touche2020";
    do
    echo $data_name
    python beir_qrels_ch.py $data_name"/"$data_name
    python delete_metadata.py $data_name"/"$data_name
    python select_query.py $data_name"/"$data_name
    p_max_len=512
    q_max_len=512
    dataset_name="/root/paddlejob/workspace/env_run/data/beir/BEIR/"$data_name"/"$data_name
    dev_qrel_path=$dataset_name'/qrels/test_qrels.tsv'
    encode_path="zero-shot/"$model"/"$data_name
    mkdir "zero-shot/"$model
    mkdir $encode_path
    echo "====== encode corpus"

    for s in 0 1 2 3 4 5 6 7;
    do
    gpuid=$s
    CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode_mm \
    --output_dir=temp \
    --model_name_or_path $model_name_or_path \
    --normalize \
    --fp16 \
    --query_data $data_name \
    --per_device_eval_batch_size 1 \
    --pooling eos \
    --append_eos_token \
    --passage_max_len $p_max_len \
    --dataset_path $dataset_name'/corpus.jsonl' \
    --query_max_len $q_max_len \
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${s} \
    --encode_output_path $encode_path/corpus_emb.${s}.pkl   &
    # 等待最后一个循环完成
    if [ "$s" == "7" ]; then
        wait
    fi
    done
    CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode_mm \
    --output_dir=temp \
    --model_name_or_path $model_name_or_path \
    --normalize \
    --encode_is_query \
    --fp16 \
    --per_device_eval_batch_size 1 \
    --query_data $data_name \
    --passage_max_len $p_max_len \
    --pooling eos \
    --append_eos_token \
    --query_max_len $q_max_len \
    --dataset_path $dataset_name'/queries.jsonl' \
    --encode_output_path $encode_path/dev_query_emb.pkl

    sleep 10s
    echo "====== Search the Corpus"
    set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
    --query_reps $encode_path'/dev_query_emb.pkl' \
    --passage_reps $encode_path'/corpus_emb.*.pkl' \
    --depth 100 \
    --batch_size 1024 \
    --save_text \
    --save_ranking_to $encode_path/dev_rank.txt

    
    echo "====== Evaluation"
    python tevatron/utils/format/convert_result_to_trec.py \
    --input $encode_path'/dev_rank.txt' \
    --output $encode_path'/dev_rank.txt.trec'

    ./trec_eval -m ndcg_cut.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
    ./trec_eval -m ndcg_cut.100 $dev_qrel_path $encode_path/dev_rank.txt.trec
    ./trec_eval -m recall $dev_qrel_path $encode_path/dev_rank.txt.trec
    python trec_eval.py --qrel_file $dev_qrel_path  --run_file $encode_path/dev_rank.txt.trec
    # 等待最后一个循环完成
    if [ "$data_name" == "arguana" ]; then
        wait
    fi
    done
done
sh run_cqa.sh > run_cqa.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /root/paddlejob/workspace/env_run/run.py --size 40000 --gpus 8 --interval 0.0
