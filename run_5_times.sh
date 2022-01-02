masks=${1-0}
groupsearch=${2:-0}
runid=${3:-0}
dataset=${4:-swat}
custom_edges=${5-false}
log_save_dir=${6-"./runs_no_edges/"}

for i in 0 1 2 3 
do
    log_save_dir_i="${log_save_dir}masks_${dataset}_${masks}_${runid}_$i.txt"
    echo $log_save_dir_i
    echo $custom_edges
    bash run.sh $(($i%4)) $dataset 50 $custom_edges $i $masks $groupsearch> $log_save_dir_i &
done

sleep 5s

done="1"
while true;
do
    lines=$(ps aux | grep plymper | grep main.py | grep $dataset | wc -l)
    
    if [ $lines -eq 0 ] 
    then
        echo "should break"
        break
    fi
    echo "running $lines"
    sleep 10s
done

echo "DONE"