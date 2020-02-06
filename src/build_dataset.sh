python build-dataset.py --data-path "h5-files" \
                        --output-path "prediction-dataset" \
                        --training "P1 P2 P3 P4 P5 P6 P7 P9 P10 P12 W1 W2" \
                        --validation "P8 W3" \
                        --testing "P5" \
                        --task-input "normOrientation" \
                        --input-label-request "all" \
                        --task-output "normOrientation"
