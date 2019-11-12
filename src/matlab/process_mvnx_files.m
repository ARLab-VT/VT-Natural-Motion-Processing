files = dir([get_folder_path(), '/mvnx-files/*.mvnx']);
i = 0;

for file = files'
  i = i + 1;
  disp(i);
  mvnx_to_hdf(file.name);
end
