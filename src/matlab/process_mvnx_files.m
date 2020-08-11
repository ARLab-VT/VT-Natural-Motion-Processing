% Copyright (c) 2020-present, Assistive Robotics Lab
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

files = dir([get_folder_path(), '/mvnx-files/*.mvnx']);
i = 0;

for file = files'
  i = i + 1;
  disp(i);
  mvnx_to_hdf(file.name);
end
