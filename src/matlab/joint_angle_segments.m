function [jointAngleSegments] = joint_angle_segments()
% JOINT_ANGLE_SEGMENTS Returns a cell array containing the joint angles of
% the XSens data file in proper order.
%
%   jointAngleSegments = JOINT_ANGLE_SEGMENTS() is used elsewhere to
%   construct joint angle maps and fill in data.
%
%   See also INIT_JOINT_ANGLE_MAP, READ_DATA
%
    jointAngleSegments = {'jL5S1','jL4L3','jL1T12','jT9T8','jT1C7','jC1Head','jRightT4Shoulder',...
        'jRightShoulder','jRightElbow','jRightWrist','jLeftT4Shoulder','jLeftShoulder','jLeftElbow',...
        'jLeftWrist','jRightHip','jRightKnee','jRightAnkle','jRightBallFoot','jLeftHip','jLeftKnee','jLeftAnkle',...
        'jLeftBallFoot'};
end