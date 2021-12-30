masks=${1-0}
groupsearch=${2:-0}
custom_edges=${3-false}
log_save_dir=${4-"./runs_no_edges/"}
for i in 0 
do
    log_save_dir_i="${log_save_dir}$i.txt"
    echo $log_save_dir_i
    echo $custom_edges
    bash run.sh $(($i%4)) swat 30 $custom_edges $i $masks $groupsearch > $log_save_dir_i 
done

#python slack_message.py "once"
