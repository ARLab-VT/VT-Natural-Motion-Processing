python build-dataset.py --data-path "h5-files" \
                        --output-path "test-dataset" \
                        --training "W1" \
                        --validation "W3" \
                        --testing "P5" \
                        --task-input "smoothedOrientation" \
                        --input-label-request "all" \
                        --task-output "jointAngle" \
                        --output-label-request "all"
