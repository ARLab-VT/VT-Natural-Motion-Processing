python build-dataset.py --data-path "h5-files" \
                        --output-path "sensor-test-dataset" \
                        --training "W1" \
                        --validation "P8" \
                        --testing "P5" \
                        --task-input "sensorFreeAcceleration sensorMagneticField sensorOrientation" \
                        --input-label-request "all" \
                        --task-output "jointAngle" \
                        --output-label-request "all"
