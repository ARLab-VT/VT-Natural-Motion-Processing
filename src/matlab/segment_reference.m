function [segmentOrientationMap, segmentPositionMap, jointMap]= segment_reference
% SEGREF Places all of the segments and joints into maps for
% accessing. Necessary for reading data from MVNX files.
%
%   [segor_map, segpos_map, joint_map] = segRef will return a map for
%   segment orientation, segment position, and joint angles. These maps can
%   then be used to access the proper index in the XSens data for
%   processing.
%
%   See also READ_DATA, LOAD_PARTIAL_MVNX

orientationIndices = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89};
positionIndices = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67};
jointIndices = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64};

segmentKeys= {'Pelvis';'L5';'L3';'T12';'T8';'Neck';'Head';'RightShoulder';'RightUpperArm';'RightForeArm';'RightHand';'LeftShoulder';'LeftUpperArm';'LeftForeArm';'LeftHand';'RightUpperLeg';'RightLowerLeg';'RightFoot';'RightToe';'LeftUpperLeg';'LeftLowerLeg';'LeftFoot';'LeftToe'};
jointKeys= {'jL5S1';'jL4L3';'jL1T12';'jT9T8';'jT1C7';'jC1Head';'jRightC7Shoulder';'jRightShoulder';'jRightElbow';'jRightWrist';'jLeftC7Shoulder';'jLeftShoulder';'jLeftElbow';'jLeftWrist';'jRightHip';'jRightKnee';'jRightAnkle';'jRightBallFoot';'jLeftHip';'jLeftKnee';'jLeftAnkle';'jLeftBallFoot'};

segmentOrientationMap = containers.Map(segmentKeys, orientationIndices);
segmentPositionMap = containers.Map(segmentKeys, positionIndices);
jointMap = containers.Map(jointKeys, jointIndices);
