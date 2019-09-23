function [parentpath] = get_folder_path()
% GET_FOLDER_PATH Returns the path to the parent folder.
%
%   parentPath = get_folder_path() returns the path to the parent folder
%   for ease of use in other files.
%
%   See also MVNX_TO_CSV
    parentpath = cd(cd('..'));
end