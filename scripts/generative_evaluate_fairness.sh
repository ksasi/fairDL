#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
if [ $1 = "stylegan2" ]
then
    python /workspace/fairDL/src/stylegan2_generator.py --num=8000 --outdir=/workspace/fairDL/data/stylegan2
    python /workspace/fairDL/src/generate_csv.py --imgdir=/workspace/fairDL/data/stylegan2 --outdir=/workspace/fairDL/results
elif [ $1 = "vae" ]
then
    rmdir /workspace/fairDL/data/nvae
    mkdir /workspace/fairDL/data/nvae
    cd /workspace/NVAE
    python evaluate.py --checkpoint /workspace/fairDL/src/checkpoint.pt --eval_mode=sample --temp=0.5 --readjust_bn --save=/workspace/fairDL/data/nvae
    python /workspace/fairDL/src/generate_csv.py --imgdir=/workspace/fairDL/data/nvae --outdir=/workspace/fairDL/results
    rm /workspace/fairDL/data/nvae/log.txt
elif [ $1 = "FlowModel" ]
then
    echo "TO DO"
elif [ $1 = "DiffusionModel" ]
then
    echo "TO DO"
elif [ $1 = "ffhq" ]
then
    python /workspace/fairDL/src/generate_csv.py --imgdir=/workspace/fairDL/data/ffhq/images1024x1024 --outdir=/workspace/fairDL/results
elif [ $1 = "All" ]
then
    echo "TO DO"
    exit 1
else
    echo "Incorrect Arguments"
fi


cd /workspace/fairDL/results
python /workspace/FairFace/predict.py --csv /workspace/fairDL/results/test_imgs.csv
mv /workspace/fairDL/results/test_outputs.csv /workspace/fairDL/results/test_outputs_$current_time.csv
python /workspace/fairDL/src/generative_generate_plot.py --src=/workspace/fairDL/results/test_outputs_$current_time.csv --outdir=/workspace/fairDL/results
mv /workspace/fairDL/results/plot_race.pdf /workspace/fairDL/results/plot_race_$1_$current_time.pdf
mv /workspace/fairDL/results/plot_race4.pdf /workspace/fairDL/results/plot_race4_$1_$current_time.pdf
mv /workspace/fairDL/results/plot_gender.pdf /workspace/fairDL/results/plot_gender_$1_$current_time.pdf
mv /workspace/fairDL/results/plot_age.pdf /workspace/fairDL/results/plot_age_$1_$current_time.pdf