ijob -A bii_dsc_community \
     -p bii-gpu \
     --reservation=bi_fox_dgx \
     --gres=gpu:1 \
     -c 8 \
     --mem=64G \
     -t 0-02:00:00