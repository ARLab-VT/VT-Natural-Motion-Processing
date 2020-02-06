
python train-transformer.py --task prediction \
                        --data-path ./prediction-dataset \
                        --model-file-path ./models/orientation-model-160-adamw.pt \
                        --batch-size=16 \
                        --seq-length=160 \
                        --stride=32 \
                        --learning-rate=0.1 \
                        --num-epochs=20 \
                        --num-heads=4 \
                        --dim-feedforward=512 \
                        --dropout=0.1 \
                        --num-layers=6
