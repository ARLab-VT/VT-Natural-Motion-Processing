
python train-seq2seq.py --data-path ./elbow-prediction \
                        --model-file-path ./models/trained-model.pt \
                        --batch-size=32 \
                        --num-epochs=20 \
                        --hidden-size=364 \
                        --bidirectional \
                        --attention=biased-general
