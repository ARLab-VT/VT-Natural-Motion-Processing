
python train-seq2seq.py --task prediction \
                        --data-path ./test-dataset \
                        --model-file-path ./models/seq2seq-trained-model.pt \
                        --batch-size=32 \
                        --seq-length=10 \
                        --learning-rate=0.1 \
                        --num-epochs=10 \
                        --hidden-size=512 \
                        --bidirectional \
                        --attention=general
