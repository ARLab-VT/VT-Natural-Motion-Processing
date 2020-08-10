python build-dataset.py --data-path "/groups/MotionPred/h5-dataset" \
                        --output-path "/home/jackg7/VT-Natural-Motion-Processing/data/set-2" \
                        --training "P1" \
                        --validation "P5" \
                        --testing "P10" \
                        --task-input "normOrientation normAcceleration" \
                        --input-label-request "T8 RightForeArm RightLowerLeg LeftForeArm LeftLowerLeg" \
                        --task-output "normOrientation" \
                        --output-label-request "all"
