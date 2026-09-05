salloc -A bii_dsc_community \
       -p bii-gpu \
       --reservation=bi_fox_dgx \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=64G \
       --time=02:00:00