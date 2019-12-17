
python train-seq2seq.py --data-path ./sensor-transfer \
                        --model-file-path ./models/trained-model.pt \
                        --batch-size=32 \
                        --seq-length=10 \
                        --learning-rate=0.1 \
                        --num-epochs=50 \
                        --hidden-size=512 \
                        --bidirectional \
                        --attention=general
