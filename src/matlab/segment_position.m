function[matrix]= segment_position(values, frames, index)
% SEGMENT_POSITION Re-shuffles each x,y,z position in data to a segment
%
%   matrix = SEGMENT_POSITION(values, frames, index) will associate 3
%   parts of position to the proper segment using XSens'
%   segmentation indices.
%
%   See also READ_DATA
    matrix= zeros(frames, 3);
    for i= 1:frames
        matrix(i, :)= values(i).position(index:index + 2);
    end
end