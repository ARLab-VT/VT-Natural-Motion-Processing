
python train-seq2seq.py --task prediction \
                        --data-path ./prediction-dataset \
                        --model-file-path ./models/seq2seq-trained-model.pt \
                        --batch-size=32 \
                        --seq-length=160 \
                        --stride=20 \
                        --learning-rate=0.001 \
                        --num-epochs=30 \
                        --hidden-size=256 \
                        --bidirectional \
                        --attention=general
