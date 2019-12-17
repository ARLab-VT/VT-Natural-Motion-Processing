
python stride-train-transformer.py --data-path ./full-body-transfer \
                        --model-file-path ./models/full-body-trained-model-60-stride.pt \
                        --batch-size=32 \
                        --seq-length=10 \
                        --learning-rate=0.1 \
                        --num-epochs=100 \
                        --num-heads=4 \
                        --dim-feedforward=248 \
                        --dropout=0.2 \
                        --num-layers=2
