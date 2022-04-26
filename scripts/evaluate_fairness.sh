#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

if [ $1 = "datagen" ]
then
    python /workspace/fairDL/src/stylegan2_generate_syndata.py --num=2000 --outdir=/workspace/fairDL/data/synthface
    python /workspace/fairDL/src/preprocess.py --source_path="/workspace/fairDL/data/synthface" --target_path="/workspace/fairDL/data/synthface_processed" --size=144 --num=2000
    python /workspace/fairDL/src/stylegan2_generate_syndata.py --num=500 --outdir=/workspace/fairDL/data/synthfaceval
    python /workspace/fairDL/src/preprocess.py --source_path="/workspace/fairDL/data/synthfaceval" --target_path="/workspace/fairDL/data/synthfaceval_processed" --size=144 --num=500
elif [ $1 = "finetune" ]
then
    #python -u /workspace/fairDL/src/fine_tune.py --save_path=/workspace/fairDL/checkpoints/ --model="LightCNN_29" --dataset="LFW" --num_classes=1180 --arch="LightCNN_29" --epochs=30 --batch_size=256 --learning_rate=1e-4 --weight_decay=1e-4 --momentum=0.9 >> /workspace/fairDL/results/LightCNN29_out_$current_time.log
    python -u /workspace/fairDL/src/fine_tune.py --save_path=/workspace/fairDL/checkpoints/ --model="LightCNN_29" --dataset="LFW" --num_classes=1180 --arch="LightCNN_29" --epochs=50 --batch_size=256 --learning_rate=1e-2 --weight_decay=1e-2 --momentum=0.9 >> /workspace/fairDL/results/LightCNN29_out_$current_time.log
elif [ $1 = "evaluate-pretrained" ]
then
    python /workspace/fairDL/src/predict.py --model="LightCNN_29" --state="Pretrained" --file="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv" --root_path="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/" --output_file="/workspace/fairDL/results/pre_trained_pred.csv" --model_checkpoint="/workspace/fairDL/models/LightCNN_29Layers_checkpoint.pth.tar"
    python -u /workspace/fairDL/src/evaluate.py --state="Pretrained" --predfile="/workspace/fairDL/results/pre_trained_pred.csv" --outdir="/workspace/fairDL/results" >> /workspace/fairDL/results/out_eval_pretrained_$current_time.log
elif [ $1 = "evaluate-finetuned" ]
then
    ##python /workspace/fairDL/src/predict.py --model="LightCNN_29"  --state="finetuned" --file="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv" --root_path="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/" --output_file="/workspace/fairDL/results/fine_tuned_pred.csv" --model_checkpoint="/workspace/fairDL/checkpoints/model_22_checkpoint.pth.tar"
    #####python /workspace/fairDL/src/predict.py --model="LightCNN_29"  --state="finetuned" --file="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv" --root_path="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/" --output_file="/workspace/fairDL/results/fine_tuned_pred.csv" --model_checkpoint="/workspace/fairDL/checkpoints/model_23_checkpoint.pth.tar"
    python /workspace/fairDL/src/predict.py --model="LightCNN_29"  --state="finetuned" --file="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv" --root_path="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/" --output_file="/workspace/fairDL/results/fine_tuned_pred.csv" --model_checkpoint="/workspace/fairDL/checkpoints/model_41_checkpoint.pth.tar"
    python -u /workspace/fairDL/src/evaluate.py --state="finetuned" --predfile="/workspace/fairDL/results/fine_tuned_pred.csv" --outdir="/workspace/fairDL/results" >> /workspace/fairDL/results/out_eval_finetuned_$current_time.log
elif [ $1 = "plots" ]
then
    python /workspace/fairDL/src/generate_plot.py --srcdir="/workspace/fairDL/results" --outdir="/workspace/fairDL/results"
else
    echo "TODO"
fi