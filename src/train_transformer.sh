
python train-transformer.py --data-path ./orientation-dataset \
                        --model-file-path ./models/full-body-trained-model-10-stride.pt \
                        --batch-size=32 \
                        --seq-length=10 \
                        --learning-rate=0.1 \
                        --num-epochs=100 \
                        --num-heads=8 \
                        --dim-feedforward=64 \
                        --dropout=0.1 \
                        --num-layers=2
