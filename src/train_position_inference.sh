
python train-seq2seq.py -f /home/jack/work/research/ARL/xsens/position-inference/ \
                        --model-file-path ./models/trained-model.pt \
                        --batch-size=32 \
                        --encoder-feature-size=12 \
                        --decoder-feature-size=3 \
                        --num-epochs=20 \
                        --bidirectional \
                        --attention=biased-general