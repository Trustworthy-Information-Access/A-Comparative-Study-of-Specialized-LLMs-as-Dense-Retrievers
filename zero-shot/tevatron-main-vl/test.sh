model="Qwen2.5-VL-7B-Instruct" 
echo $model
model_name_or_path='/root/paddlejob/workspace/env_run/model/'$model
# for data_name in  "apps"  "codetrans-contest" "codefeedback-mt";
for code_types in "go" "java" "javascript" "php" "python" "ruby";
do 
    for data_name in  "CodeSearchNet" "CodeSearchNet-ccr";
    do
    echo $data_name'/'$code_types
    p_max_len=512
    q_max_len=512
    dataset_name="/root/paddlejob/workspace/env_run/output/CoIR/coir/"$data_name'/'$code_types
    dev_qrel_path=$dataset_name'-qrels/test_qrels.tsv'
    encode_path="zero-shot/"$model"/"$data_name'/'$code_types
    mkdir "zero-shot/"$model
    mkdir "zero-shot/"$model"/"$data_name
    mkdir $encode_path

    # sleep 10s
    # echo "====== Search the Corpus"
    # for id in "0" "1" "2" "3" "4" "5" "6" "7";
    # do 
    # set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
    # --query_reps $encode_path'/dev_query_emb.'$id'.pkl' \
    # --passage_reps $encode_path'/corpus_emb.*.pkl' \
    # --depth 100 \
    # --batch_size 1024 \
    # --save_text \
    # --save_ranking_to $encode_path'/dev_rank'$id'.txt'
    # if [ "$id" == "7" ]; then
    # wait
    # fi
    # done
    # cat $encode_path/dev_rank{0..7}.txt > $encode_path/dev_rank.txt

    
    echo "====== Evaluation"
    python tevatron/utils/format/convert_result_to_trec.py \
    --input $encode_path'/dev_rank.txt' \
    --output $encode_path'/dev_rank.txt.trec'

    ./trec_eval -m ndcg_cut.10 $dev_qrel_path $encode_path/dev_rank.txt.trec
    # 等待最后一个循环完成
    if [ "$data_name" == "CodeSearchNet-ccr" ]; then
        wait
    fi
    done
if [ "$code_types" == "ruby" ]; then
    wait
fi
done
