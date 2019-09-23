function[matrix]= segment_orientation(values, frames, index)
% SEGMENT_ORIENTATION Re-shuffles each quaternion in data to a segment
%
%   matrix = SEGMENT_ORIENTATION(values, frames, index) will associate 4
%   parts of quaternion to the proper segment using XSens'
%   segmentation indices.
%
%   See also READ_DATA, GET_ROLL, GET_PITCH, GET_YAW
    matrix= zeros(frames, 4);
    for i= 1:frames
        matrix(i, :)= values(i).orientation(index:index+3); 
    end
end