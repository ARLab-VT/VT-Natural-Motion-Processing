python build-dataset.py --data-path "h5-files" \
                        --output-path "orientation-dataset" \
                        --training "W1 W2 W3 P1 P2 P3 P4 P5 P6 P7" \
                        --validation "P8 P9 P10 P12" \
                        --testing "P5" \
                        --task-input "smoothedOrientation" \
                        --input-label-request "all" \
                        --task-output "smoothedOrientation"
