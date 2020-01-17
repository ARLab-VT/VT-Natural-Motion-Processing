
python train-transformer.py --task prediction \
                        --data-path ./jointAngles-dataset \
                        --model-file-path ./models/transformer-test.pt \
                        --batch-size=32 \
                        --seq-length=20 \
                        --stride=5 \
                        --learning-rate=0.1 \
                        --num-epochs=100 \
                        --num-heads=1 \
                        --dim-feedforward=64 \
                        --dropout=0.1 \
                        --num-layers=2
