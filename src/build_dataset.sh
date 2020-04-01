python build-dataset.py --data-path "h5-files" \
                        --output-path "/groups/MotionPred/motion-inference/full-body/full-set-2" \
                        --training "P1 P2 P3 P4 P6 P7 P8 P9 P11 P12 W1 W4" \
                        --validation "P5 W2 W3" \
                        --testing "P5" \
                        --task-input "normOrientation normAcceleration" \
                        --input-label-request "T8 RightForeArm RightLowerLeg LeftUpperArm LeftLowerLeg" \
                        --task-output "normOrientation" \
                        --output-label-request "all"
