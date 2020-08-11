% Copyright (c) 2020-present, Assistive Robotics Lab
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

function data = mvnx_to_csv(file)
%MVNX_TO_CSV File for reading in .mvnx files and converting them to .csv files.
%   Reading the .mvnx files is expensive, so this converts orientation, position, and
%   joint angle data to csv format.

filename = [get_folder_path(), '/mvnx-files/', file];

[data, ~, ~, ~] = load_partial_mvnx(filename, {'orientation', 'position', 'jointAngle'}); 

csvData = zeros(size(data,1), 227);

for i = 1:size(data,1)
    
    csvData(i, :) = [data(i).orientation(:)' data(i).position(:)' data(i).jointAngle(:)'];

end
output = [get_folder_path(), '/', 'csv-files', '/', file(1:end-5), '.csv'];

csvwrite(output, csvData);

end