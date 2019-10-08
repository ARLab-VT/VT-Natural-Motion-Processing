python build-position-inference-dataset.py --csv-files "../../CSV Files" \
                                           --output-file-path "data/cross-validation" \
                                           --max-file-count 20 \
                                           --seq-length 20 \
                                           --batch-size 512 \
                                           --split-size 0.8
