% Copyright (c) 2020-present, Assistive Robotics Lab
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

function mvnx_to_hdf(file)

input_file = [get_folder_path(), '/mvnx-files/', file];
output_file = [get_folder_path(), '/h5-files/', file(1:end-5), '.h5'];

dataset_names = {'orientation', 'position', 'velocity', 'acceleration', 'angularVelocity', ...
		 'angularAcceleration', 'footContacts', 'sensorFreeAcceleration', ...
		 'sensorMagneticField', 'sensorOrientation', 'jointAngle', 'jointAngleXZY', ...
                 'jointAngleErgo', 'jointAngleErgoXZY', 'centerOfMass'};

disp(input_file);
             
tic
for i = 1:length(dataset_names)
    [data, ~, ~, ~] = load_partial_mvnx(input_file, dataset_names(i)); 
    data = struct2cell(data);
    
    dataset = cell2mat(data);
    dataset = reshape(dataset, length(data), length(dataset)/length(data));
    
    h5create(output_file, ['/', dataset_names{i}], size(dataset));
    h5write(output_file, ['/', dataset_names{i}], dataset);
end
toc

h5disp(output_file);
end
