% Copyright (c) 2020-present, Assistive Robotics Lab
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

function[matrix]= joint_angles(values, frames, index)
% JDEG Re-shuffles each joint angle in data to a joint
%
%   matrix = JDEG(values, frames, index) will associate flexion, abduction,
%   and extension to the proper joint.
%
%   See also READ_DATA
    matrix= zeros(frames, 3);
    for i= 1:frames
        matrix(i, :)= values(i).jointAngle(index:index+2);
    end
end