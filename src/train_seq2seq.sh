
python train-seq2seq.py --task prediction \
                        --data-path ./prediction-dataset \
                        --model-file-path ./models/seq2seq-trained-model-dim-1000.pt \
                        --batch-size=32 \
                        --seq-length=160 \
                        --stride=20 \
                        --learning-rate=0.001 \
                        --num-epochs=10 \
                        --hidden-size=1000 \
                        --attention=dot
