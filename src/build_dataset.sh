python build-dataset.py --data-path "h5-files" \
                        --output-path "test-folder" \
                        --training "P1" \
                        --validation "P5" \
                        --testing "P10" \
                        --task-input "normOrientation normAcceleration" \
                        --input-label-request "T8 RightForeArm" \
                        --task-output "normOrientation" \
                        --output-label-request "RightUpperArm"
