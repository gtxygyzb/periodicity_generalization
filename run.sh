GPU=0
export CUDA_VISIBLE_DEVICES=${GPU}

periodicType=2seq_add
Layers=2
modelName=RWKV

timestamp=$(date "+%m-%d_%H-%M")
path="./final_res/${periodicType}_${modelName}_rope_2-17_${timestamp}"
python3 -u ./test.py \
--layers ${Layers} \
--model_name ${modelName} \
--periodic_type ${periodicType} \
--path ${path}

wait $!