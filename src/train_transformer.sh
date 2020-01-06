
python train-transformer.py --task conversion \
                        --data-path ./big-dataset \
                        --model-file-path ./models/full-body-trained-model-10-stride.pt \
                        --batch-size=32 \
                        --seq-length=10 \
                        --stride=3 \
                        --learning-rate=0.1 \
                        --num-epochs=100 \
                        --num-heads=4 \
                        --dim-feedforward=64 \
                        --dropout=0.1 \
                        --num-layers=2
