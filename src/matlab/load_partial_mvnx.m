function [data,time,cal_data,extra] = load_partial_mvnx(filename,section,varargin)
% This function allows matlab to read in larger mvnx files since it does
% not try to load the entire file into memory which can be several GB
% large. 
%
%
% data = load_partial_mvnx(filename,section)
%   Loads all of the frame data from the fields specified in section, with
%   multiple fields passed as a cell array. 
%
% data = load_partial_mvnx(filename,section,n)
%   Loads n frames of the data. Where n<=0 and n=0 indicates to
%   read to the end of the file. If n is greater than the number of frames
%   then the data is read until the end of the file.
%
% data = load_partial_mvnx(filename,section,m,n)
%   Loads n frames of the data starting at frame m.
%
% Inputs
%   filename - Type:string, path to the file to be read 
%
%   section - Types:string,cell array, labels of data to read
%       Valid value for field are 'segments','joints','orientation','position','velocity',
%       'acceleration','angularVelocity','angularAcceleration','jointAngle','jointAngleXYZ'
%   
%       'segments' and 'joints' cannot be used in the cell array.
%       Order does not matter in the cell array
%       Example section = 'position' or {'acceleration','position',...} 
%
%   m,n - Type:integers, m>=0, n>=0
%
% Outputs
%   data - Type:Varies based on arguements
%
%   time - Type:struct time values for the frames, output only for sections
%   'orientation','position','velocity','acceleration','angularVelocity',
%   'angularAcceleration','jointAngle','jointAngleXYZ'
%
%   cal_data - Type:struct calibration pose data, output only for sections
%   'orientation','position','velocity','acceleration','angularVelocity',
%   'angularAcceleration','jointAngle','jointAngleXYZ'
%
%   extra - Type:struct other data from sections 'orientation','position',
%   'velocity','acceleration','angularVelocity','angularAcceleration',
%   'jointAngle','jointAngleXYZ'


% preallocate some space probably will not be enough 
N=1000000;
cells=cell(N,1);

% number of calibration poses, aka frames with type other than 'normal',
% this might need to be made variable in the future
n_cal = 3;

% error handling and setting up initial parameters
frame_data = {'orientation', 'position', 'velocity', 'acceleration', 'angularVelocity', ...
              'angularAcceleration', 'footContacts', 'sensorFreeAcceleration', ...
		      'sensorMagneticField', 'sensorOrientation', 'jointAngle', 'jointAngleXZY', ...
              'jointAngleErgo', 'jointAngleErgoXZY', 'centerOfMass'};

if iscell(section)         
    parfor i=1:length(section)
        f_test(i)=any(strcmp(frame_data, section{i}));
    end
else
    f_test=any(strcmp(frame_data, section));
end

if iscell(section)
    ME=MException('Input:error','Supplied arg %s %s %s\nSyntax: load_partial_mvnx(filename,section,[[limit_start],][limit_stop])\nSupported section values are:\n Singular Arguments - segments, joints\n These can be applied individually or as a cell array - orientation, position, velocity, acceleration, angularVelocity, angularAcceleration, jointAngle, jointAngleXYZ',filename,section{:},varargin{:});
else
    ME=MException('Input:error','Supplied arg %s %s %s\nSyntax: load_partial_mvnx(filename,section,[[limit_start],][limit_stop])\nSupported section values are:\n Singular Arguments - segments, joints\n These can be applied individually or as a cell array - orientation, position, velocity, acceleration, angularVelocity, angularAcceleration, jointAngle, jointAngleXYZ',filename,section,varargin{:});
end
LE=MException('Limit:error','0<=limit_start<=limit_end');

if nargin-length(varargin)<2 || nargin>4
    throw(ME)
end
run_cal=1;
% set up to get calibration data
if length(varargin)==1 && ischar(varargin{1})
    limit_start=0;
    limit_stop=n_cal;
    section={'orientation','position'};
    f_test=1;
    run_cal=0;
%set up to get segment/joint info
elseif ~iscell(section) && (strcmp(section,'segments') || strcmp(section,'joints'))
    limit_start=0;
    limit_stop=0;
% set converting limits for all other formats
elseif length(varargin)==1
    if varargin{1}<0
        throw(LE)
    end
    limit_start=n_cal;
    limit_stop=varargin{1}+n_cal;
elseif length(varargin)==2
    if varargin{1}<0 || varargin{2}<varargin{1}
        throw(LE)
    end
    limit_start=varargin{1}+n_cal;
    limit_stop=varargin{2}+varargin{1}+n_cal;
else
    limit_start=n_cal;
    limit_stop=0;
end

% Set parsing values
if ~iscell(section) && strcmp(section,'segments')
    start = '<segments>';
    st = {'<segment','<points>','<point'};
    key = {'<pos_b>'};
    en = {'</segment>','</points>','</point>'};
    stop = '</segments>'; 
    st_depth = 1;
    en_depth = 0;
    max_depth=length(st);
elseif ~iscell(section) && strcmp(section,'joints')
    start = '<joints>';
    st = {'<joint'};
    key ={'<connector1>','<connector2>'};
    en = {'</joint>'};
    stop = '</joints>'; 
    st_depth = 1;
    en_depth = 0;
    max_depth=length(st);
elseif  all(f_test) && ~any(strcmp('segments', section)) && ~any(strcmp('joints', section))
    start = '<frames';
    st = {'<frame'};
    if iscell(section)
        key = section;
    else
        key = {section};
    end
    en = {'</frame>'};
    stop = '</frames>'; 
    st_depth = 1;
    en_depth = 0;
    max_depth=length(st);
else
    throw(ME);
end

% Check file name
if isempty(strfind(filename,'mvnx'))
    filename = [get_folder_path() '\' filename '.mvnx'];        
end
if ~exist(filename,'file')
    error([mfilename ':xsens:filename'],['No file with filename: ' filename ', file is not present or file has wrong format (function only reads .mvnx)'])
end


% Open file
disp('Reading file');
fid = fopen(filename, 'r', 'n', 'UTF-8');

l = fgetl(fid);
k = 1;

% find start
while isempty(strfind(l,start))
    l = fgetl(fid);
end
cells{k} = l;
k=k+1;
l = fgetl(fid);
count = 0;
while ~feof(fid)
    if ~isempty(strfind(l,stop))
        cells{k} = l;
        k = k+1;
        break
    end
    % test for the start key at the current depth
    found = 0;
    if st_depth<=max_depth && ~isempty(strfind(l,st{st_depth}))
        found = 1;
        if count>=limit_start
            cells{k} = l;
            k = k+1;
        end
        st_depth = st_depth+1;
        en_depth = en_depth+1;
    elseif st_depth>max_depth
        for i=1:length(key)
            if ~isempty(strfind(l,key{i}))
                if count>=limit_start
                    cells{k} = l;
                    k = k+1;
                end
                break 
            end
        end
    end
    % if a start key was not fould 
    if en_depth>0 && ~isempty(strfind(l,en{en_depth}))
        if found==0
            if count>=limit_start
                cells{k} = l;
                k = k+1;
            end
            if en_depth == 1
                count=count+1;
            end
            st_depth = st_depth-1;
            en_depth = en_depth-1;
        else
            if en_depth == 1
                count=count+1;
            end
            st_depth = st_depth-1;
            en_depth = en_depth-1;
        end
    end
    if limit_stop~=0 && count>=limit_stop
        cells{k}=stop;
        break
    end
    l = fgetl(fid);
end
if k<N
    cells(cellfun(@isempty,cells))=[];
end

if isempty(cells)
    error('start value is too large');
end
fclose(fid);
%%
cellContent=cells;

% look for comment lines and remove them
cellContent(cellfun(@(x) x(1)=='-' || (x(1)=='<' && (x(2)=='?' || x(2)=='!')),cellContent)) = [];

% get start of text and clean up some
index = cellfun(@(x) find(x==60,1),cellContent);
openandclose = cellfun(@(x) sum(x==60)==2,cellContent); % lines with have an opening and closing statement in one line
cellContent = cellfun(@(x) x(find(x==60):end), cellContent,'UniformOutput',false);

%% get start and end words
disp('Parsing file');
%n = 0;
clear value name
iWord = 0;
word = cell(1,length(cellContent));
wordindex = cell(1,length(cellContent));
wordvalue = cell(1,length(cellContent));
wordfields = cell(1,length(cellContent));
parfor n = 1:1:length(cellContent)
    %n=n+1;
    line = cellContent{n};oneword = false;
    hooks = find(line == 62);
    iWord = iWord+1;
    wordindex{n} = index(n);
    if any(line == 32) && ~openandclose(n)
        if ~isempty(hooks) && hooks(1) < find(line==32,1)
            word{n} = line(2:hooks(1)-1);
            iLine = hooks(1)+1;
        else
            word{n} = line(2:find(line==32,1)-1);
            iLine = find(line==32,1)+1;
        end
    elseif ~isempty(hooks) && length(line)>=8 && strcmp('comment',line(2:8))
        % add exception for comment
        word{n} = line(2:hooks(1)-1);
        iLine = hooks(1)+1;
    elseif openandclose(n)
        word{n} = line(2:find(line==62,1)-1);
        iLine = find(line==62,1)+1;
    else
        word{n} = line(2:end-1);
        oneword = true;
    end
    if word{n}(1) ~= '/'
        if ~oneword && ~openandclose(n)
            k = find(line == 34);
            k = reshape(k,2,length(k)/2)';
            l = [iLine find(line(iLine:end) == 61)+iLine-2];
            fieldname = cell(1,length(l)-1); value = cell(1,length(l)-1);
            if ~isempty(k)
                for il=1:size(k,1)
                    fieldname{il} = line(iLine:find(line(iLine:end) == 61,1)+iLine-2);
                    if size(k,1) > 1 && il < size(k,1)
                        a = strfind(line(iLine:end),'" ')+iLine+1;
                        iLine = a(1);
                    end
                    value{il} = line(k(il,1)+1:k(il,2)-1);
                end
            else
                value = []; fieldname =[];
                value = line(find(line == 62,1)+1:end);
            end
        elseif ~oneword && openandclose(n)
            value = []; fieldname =[];
            value = line(find(line == 62,1)+1:find(line==60,1,'last')-1);
        else
            value = NaN;fieldname = [];
        end
        wordvalue{n} = value;
        wordfields{n} = fieldname;
    end
end
%% get values
parfor n=1:length(wordvalue)
    if iscell(wordvalue{n})
        if length(wordvalue{n}) == 1
            B = [];
            try
                B = str2double(wordvalue{n}{1});
            end
            if ~isempty(B)
                wordvalue{n} = B;
            else
                wordvalue{n} = wordvalue{n}{1};
            end
        else
            for m=1:length(wordvalue{n})
                try
                    B = str2double(wordvalue{n}{m});
                    if ~isempty(B)
                        wordvalue{n}{m} = B;
                    end
                end
            end
        end
    else
        try
            B = str2num(wordvalue{n});
            if ~isempty(B)
                wordvalue{n} = B;
            end
        end
    end
end
%%
disp('Logging data');
% form tree structure for segment and joint data
if ~iscell(section) && (strcmp(section,'segments') || strcmp(section,'joints'))
    data=rec_struct(word,wordfields,wordvalue,wordindex,1,length(section));
    extra = [];
    cal_data = [];
    time=[];
else
    %store extra values and get calibration data
    extra = struct(wordfields{1,1}{1},wordvalue{1,1}{1},wordfields{1,1}{2},wordvalue{1,1}{2});
    if run_cal
        cal_data = load_partial_mvnx(filename,section,'cal');
    else
        cal_data=[];
    end
    
    f = strcmp(word,'frame');
    fi=find(f);
    n_f = length(wordfields{fi(1)});
    fr_lab = wordfields{f};

    n_k=length(key);

    for i=1:length(key)
        k = strcmp(word,key{i});
        kf = find(k);
        kc{i}=kf;
    end

    %preallocate
    val=cell(length(fi),n_f);
    val2=cell(length(fi),n_k);

    for i=1:length(fi)
        for j=1:n_f
            if isnumeric(wordvalue{fi(i)}{j})
                val{i,j}=wordvalue{fi(i)}{j};
            elseif ischar(wordvalue{fi(i)}{j})
                val{i,j}=wordvalue{fi(i)}{j};
            else
                disp('error');
            end
        end
        for j=1:n_k
            val2{i,j}=wordvalue{kc{j}(i)};
        end
    end

    % put data into structures
    time=form_struct(fr_lab,val);
    data=form_struct(key,val2);

end
end
function data = form_struct(lab,val)
    switch length(lab)
        case 1
            data=struct(lab{1},val(:,1));
        case 2
            data=struct(lab{1},val(:,1),lab{2},val(:,2));
        case 3
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3));
        case 4
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3),lab{4},val(:,4));
        case 5
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3),lab{4},val(:,4),lab{5}, val(:,5));
        case 6
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3),lab{4},val(:,4),lab{5},val(:,5),...
                        lab{6},val(:,6));
        case 7
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3),lab{4},val(:,4),lab{5},val(:,5),...
                        lab{6},val(:,6),lab{7},val(:,7));
        case 8
            data=struct(lab{1},val(:,1),lab{2},val(:,2),lab{3},val(:,3),lab{4},val(:,4),lab{5},val(:,5),...
                        lab{6},val(:,6),lab{7},val(:,7),lab{8},val(:,8));
        otherwise
            disp('data format error');
    end
end
% Recursively builds up structs for segments and joints
function [s,j] = rec_struct(word,field,value,index,i,len)
    in=index{i};
    numfields = length(field{i});
    j=i+1;
    k=1;
    si={};
    while index{j}>in
        [sn,j]=rec_struct(word,field,value,index,j,len);
        if ~isempty(sn)
            si{k}=sn;
            k=k+1;
        end
    end
    if isempty(si)
        si=value{i};
    end
    switch numfields
        case 0
            s=si;
        case 1
            s=struct(char(field{i}),value{i},word{i+1},si);
        case 2
            s=struct(field{i}{1},value{i}{1},field{i}{2},value{i}{2},word{i+1},si);
        case 3
            s=struct(field{i}{1},value{i}{1},field{i}{2},value{i}{2},field{i}{3},value{i}{3},word{i+1},si);
        case 4
            s=struct(field{i}{1},value{i}{1},field{i}{2},value{i}{2},field{i}{3},value{i}{3},field{i}{4},value{i}{4},word{i+1},si);
        case 5
            s=struct(field{i}{1},value{i}{1},field{i}{2},value{i}{2},field{i}{3},value{i}{3},field{i}{4},value{i}{4},field{i}{5},value{i}{5},word{i+1},si);
        otherwise
            disp('error')
    end
end
