function data = mvnx_to_csv(file)
%MVNX_TO_CSV Summary of this function goes here
%   Detailed explanation goes here

filename = [get_folder_path(), '/', 'MVNX Files', '/', file];

[data, ~, ~, ~] = load_partial_mvnx(filename, {'orientation', 'position', 'jointAngle'}); 


csvData = zeros(size(data,1), 227);

for i = 1:size(data,1)
    
    csvData(i, :) = [data(i).orientation(:)' data(i).position(:)' data(i).jointAngle(:)'];

end
output = [get_folder_path(), '/', 'CSV Files', '/', file(1:end-5), '.csv'];

csvwrite(output, csvData);

end