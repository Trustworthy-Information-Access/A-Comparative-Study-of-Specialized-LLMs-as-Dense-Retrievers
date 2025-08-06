Code for our paper 'A Comparative Study of Specialized LLMs as Dense Retrievers'. 

# Zero-shot 
## Datasets 
(BEIR)[https://github.com/beir-cellar/beir] and (CoIR)[https://huggingface.co/CoIR-Retrieval] datasets. 
## Inference
```
sh zero-shot/tevatron-main-vl/run_vl.sh
```


# SFT on MS MARCO
## Dataset
Training data are provided by [Tevatron](https://www.dropbox.com/scl/fi/pkm1mtgfobae9kuesp7dr/train-tevatron.jsonl?rlkey=2thutc4zkozr9jp4zbbrz5rvi&dl=0)
Test data can be downloaded on [MS MARCO Dev](https://microsoft.github.io/msmarco/Datasets) 
## Training and Inference
Take Qwen2.5-7b-instruct as an example:  
```
sh SFT/tevatron-main/src/Qwen2.5-7b-instruct.sh 
```
All the checkpoints of the trained specialized LLM-based dense retrievers on MS MARCO are released on [Specialized_LLMs_as_Dense_Retrievers](https://huggingface.co/hengranZhang/Specialized_LLMs_as_Dense_Retrievers). 




