% Copyright (c) 2020-present, Assistive Robotics Lab
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

  clear

  fprintf('Starting job');
%
%  BATCH defines the job and sends it for execution.
%
  job = batch ('process_mvnx_files', 'Profile', 'dtlogin_R2018a', 'AttachedFiles', {'get_folder_path', 'joint_angle_segments', 'joint_angles', ... 
											 'load_partial_mvnx', 'mvnx_to_hdf', ...
                                                                                         'segment_orientation', 'segment_position', 'segment_reference' }, ...
                                                                                         'CurrentFolder', '.', 'Pool', 9);
%
%  WAIT pauses the MATLAB session til the job completes.
%
  wait (job);

%
%  DIARY displays any messages printed during execution.
%
  diary (job);

%
%  These commands clean up data about the job we no longer need.
  delete ( job ); %Use delete() for R2012a or later

  fprintf(1, '\n');
  fprintf(1, 'Done');
