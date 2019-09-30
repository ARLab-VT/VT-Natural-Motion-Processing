python train-seq2seq.py -f /home/jack/research/ARL/xsens/position-inference/ \
                        --batch-size=32 \
                        --encoder-feature-size=12 \
                        --decoder-feature-size=3 \
                        --num-epochs=5 \
                        --bidirectional \
                        --attention