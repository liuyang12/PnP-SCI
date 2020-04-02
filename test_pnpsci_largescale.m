function test_pnpsci_largescale(dataname)
%TEST_PNPSCI_LARGESCALE Test Plug-and-Play algorithms for Snapshot 
%Compressive Imaging (PnP-SCI) of large-scale datasets. Here we include the 
%deep denoiser FFDNet as image/video priors for four simulated largescale
%video-SCI datasets (in Bayer-RGB color).
%   TEST_PNPSCI_LARGESCALE(dataname) runs the PnP-SCI algorithms with
%   FFDNet as denoising priors for largescale color video SCI, where 
%   dataname denotes the name of the dataset for reconstruction with 
%   `messi` data as default.
% Reference
%   [1] X. Yuan, Y. Liu, J. Suo, and Q. Dai, Plug-and-play Algorithms for  
%       Large-scale Snapshot Compressive Imaging, in IEEE/CVF Conf. Comput. 
%       Vis. Pattern Recognit. (CVPR), 2020.
%   [2] X. Yuan, P. Llull, X. Liao, J. Yang, D. J. Brady, G. Sapiro, and 
%       L. Carin, Low-Cost Compressive Sensing for Color Video and Depth, 
%       in IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 3318-3325, 
%       2014. 
% Dataset
%   Please refer to the readme file in `dataset` folder.
% Contact
%   Xin Yuan, Bell Labs, xyuan@bell-labs.com, initial version Jul 2, 2015.
%   Yang Liu, MIT CSAIL, yliu@csail.mit.edu, last update Apr 1, 2020.
%   
%   See also GAPDENOISE_CACTI, GAPDENOISE, TEST_PNPSCI.

% [0] environment configuration
addpath(genpath('./algorithms')); % algorithms
addpath(genpath('./packages'));   % packages
addpath(genpath('./utils'));      % utilities

datasetdir = './dataset/simdata/largescale'; % dataset
resultdir  = './results'; % results

% [1] load dataset
if nargin<1
    dataname = 'messi'; % [default] data name
end
datapath = sprintf('%s/%s.mat',datasetdir,dataname);

if exist(datapath,'file')
    load(datapath,'meas_bayer','mask_bayer','orig_bayer');
else
    error('File %s does not exist, please check dataset directory!',datapath);
end

nframe = 1;                   % number of coded frames to be reconstructed
nmask  = size(mask_bayer, 3); % number of masks (or compression ratio B)
MAXB   = 255;                 % maximum pixel value of the image (8-bit -> 255)

para.nframe = nframe;
para.MAXB   = MAXB;

para.maskdirection   = 'plain'; % direction of the mask
para.sensorAlignment = 'bggr'; % sensor alignment of Bayer pattern
para.rotnum          = 0; % number of 90 degrees for rotation of each frame

% [2] apply PnP-SCI for reconstruction
% [2.0] common parameters
mask_bayer = single(mask_bayer);
orig_bayer = single(orig_bayer);

para.projmeth =  'gap'; % projection method 
para.lambda   =      1; % correction coefficiency
para.acc      =      1; % enable acceleration
para.flag_iqa =   true; % enable image quality assessments in iterations

% [2.1] GAP-FFDNet
para.denoiser = 'ffdnet'; % FFDNet denoising
  load(fullfile('models','FFDNet_gray.mat'),'net');
  para.net = vl_simplenn_tidy(net);
  para.useGPU = true;
  if para.useGPU
      para.net = vl_simplenn_move(para.net, 'gpu') ;
  end
  para.ffdnetvnorm_init = true; % use normalized video for the first 10 iterations
  para.ffdnetvnorm = false; % normalize the video before FFDNet video denoising
  para.sigma   = [50 25 12  6]/MAXB; % noise deviation (to be estimated and adapted)
  para.maxiter = [10 10 10 10];
  
[vgapffdnet_bayer,psnr_gapffdnet,ssim_gapffdnet,tgapffdnet,psnrall_ffdnet] = ...
    admmdenoise_cacti_bayer(mask_bayer,meas_bayer,orig_bayer,[],para);

vgapffdnet = vdemosaic(vgapffdnet_bayer,para); % video demosaic

fprintf('%s-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.projmeth),upper(para.denoiser),mean(psnr_gapffdnet),...
    mean(ssim_gapffdnet),tgapffdnet);

% [3] save results as .mat file
framerate = 5;

matdir = [resultdir '/savedmat'];
if ~exist(matdir,'dir')
    mkdir(matdir);
end

bayervideodir = sprintf('%s/video/sim_color/%s%d',...
    resultdir,dataname,nframe*nmask);
if ~exist(bayervideodir,'dir')
    mkdir(bayervideodir);
end

write_video(vgapffdnet,framerate,sprintf('%s/%s%d_%s-%s.avi',bayervideodir,...
    dataname,nframe*nmask,upper(para.projmeth),upper(para.denoiser)));

save([matdir '/pnpsci_largescale_' dataname num2str(nframe*nmask) ...
      '.mat'],'-v7.3');

end

