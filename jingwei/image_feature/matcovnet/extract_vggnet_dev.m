clear

data_folder = '/home/xiaojie/Projects/data/yfcc100m/';
% data_folder = '/Users/xiaojiew1/Projects/data/yfcc100m/';
ds = 'yfcc8k';

%data_folder = '/fishtank/urix/survey/';   % corresponds to SURVEY_DATA
%ds = 'train10k';                          % dataset to be processed
%ds = 'train1m';
%ds = 'nuswide';
%ds = 'train10k';
%ds = 'train100k';
%ds = 'mirflickr08';
%ds = 'imagenet166';
relu = 1;                                 % Do relu after fc7

% Job splitting
this_part = 1;                            % Part to be processed. One could get this from command line or environment.
parts = 1;                                % Total parts.
%%%gpuDevice(this_part);                     % Use this GPU id

cnn_nets_folder = 'cnn_models/';
data_output_folder = 'FeatureData';
img_folder = 'ImageData';
                                                                                
%dataset_path = [img_folder, ds, '/images/'];
                                                                                                    
%% --------------------SELECT PRETRAINED MODEL-------------------
%net_name = 'vgg-m-128';
%net_name = 'vgg-f';
%net_name = 'vgg-s';
%net_name = 'vgg-verydeep-16';
net_name = 'vgg-verydeep-16';

if strfind(net_name, 'verydeep-16')
    layer_selected = 35 + relu; %fc7 verydeep-16
    chunk_size = 40;
elseif strfind(net_name, 'verydeep-19')
    layer_selected = 41 + relu; %fc7 verydeep-19
    chunk_size = 20;    
else
    layer_selected = 19 + relu; %fc7
    chunk_size = 150;
end
if strfind(net_name, 'm-128')
    descrSize=128;
else
    descrSize=4096;
end

preTrainedModel=['imagenet-', net_name, '.mat'];
preTrainedModel = [cnn_nets_folder, preTrainedModel];

%% --------------------SELECT DATASET-------------------
%load file list
images_paths = importdata([data_folder, ds, '/', img_folder, '/', ds, '.txt']);
datasetSize=length(images_paths);
images_paths = cellfun(@(x) [data_folder, ds, '/', img_folder, '/', x], images_paths, 'UniformOutput', false);

if parts > 1
    part_size = floor(length(images_paths) / parts);
    if this_part == parts
        images_paths = images_paths((this_part-1) * part_size + 1 : end);
    else
        images_paths = images_paths((this_part-1) * part_size + 1 : (this_part) * part_size);
    end
end

fc7 = zeros(descrSize,length(images_paths),'single');

%% --------------------LOAD MODEL-------------------
run matconvnet-1.0-beta8/matlab/vl_setupnn
net = load(preTrainedModel);
%%%net=vl_simplenn_move(net,'gpu');

%% --------------- NUMBER OF CORES IN THE MACHINE-----------------
numCores=12;

%% --------------- MEMORY TO USE IN GB-----------------
mem2use=5;
%% --------------- ESTIMATED IMAGE SIZE IN GB-----------------
imageSize=0.0011;
numImgsForTurn=floor(mem2use/imageSize);
numTurns=ceil(length(images_paths)/numImgsForTurn);

nextBatch=cellstr(images_paths(1:min(numImgsForTurn, length(images_paths))));
vl_imreadjpeg(nextBatch,'numThreads',numCores,'Prefetch');

failedToRead = {};
failed = [];
k=1;
for j=1:numTurns
    fprintf('%d TURN of %d\n*******************\n', j, numTurns);
  
    tic;
    images=vl_imreadjpeg(nextBatch,'numThreads',numCores);
    previousBatch = nextBatch;
    if j ~= numTurns
        if j==(numTurns-1)
           nextBatch=cellstr(images_paths(((j)*numImgsForTurn+1):end));
        else
           nextBatch=cellstr(images_paths(((j)*numImgsForTurn+1):numImgsForTurn*(j+1)));
        end
        vl_imreadjpeg(nextBatch,'numThreads',numCores,'Prefetch');
    end
    toc;
  

end
