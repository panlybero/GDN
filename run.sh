gpu_n=$1
DATASET=$2

groupsearch=${7:-0}
masks=${6:-0}
seed=${5:-0}
BATCH_SIZE=64
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=15
out_layer_inter_dim=64
val_ratio=0.2
decay=0

echo $4
path_pattern="${DATASET}"
COMMENT="${DATASET}"

EPOCH=${3-30}
report='best'
echo "Running masks $masks groupsearch $groupsearch\n"
if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -custom_edges ${4-false} \
        -n_masks $masks \
        -group_search $groupsearch \
        
fi