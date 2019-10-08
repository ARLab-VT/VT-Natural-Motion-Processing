
python train-seq2seq.py -f ./data/ \
                        --model-file-path ./models/trained-model.pt \
                        --batch-size=32 \
                        --encoder-feature-size=12 \
                        --decoder-feature-size=3 \
                        --num-epochs=20 \
                        --bidirectional \
                        --attention
