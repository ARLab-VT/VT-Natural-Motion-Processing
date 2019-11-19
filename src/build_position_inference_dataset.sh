python build-dataset.py --data-path "h5-files" \
                        --output-path "elbow-prediction" \
                        --training "P1 P2 P3 P4" \
                        --validation "W1 W2 W3" \
                        --testing "P5 P7" \
                        --task-input "orientation" \
                        --input-label-request "RightUpperArm RightForeArm T8" \
                        --task-output "jointAngle" \
                        --output-label-request "jRightElbow"
