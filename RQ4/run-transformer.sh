GPU=2
export CUDA_VISIBLE_DEVICES=${GPU}

periodicType=2seq_conv
Layers=3
modelName=Qwen2.5Embedding-Transformer

timestamp=$(date "+%m-%d_%H-%M")
path="./RQ4_res/${Layers}_${periodicType}_${modelName}_${timestamp}"
python3 -u ./test.py \
--layers ${Layers} \
--model_name ${modelName} \
--periodic_type ${periodicType} \
--path ${path}

wait $!