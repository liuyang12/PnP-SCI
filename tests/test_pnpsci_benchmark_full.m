function test_pnpsci_benchmark_full(dataname)
%TEST_PNPSCI_BENCHMARK_FULL Test Plug-and-Play algorithms for Snapshot 
%Compressive Imaging (PnP-SCI). Here we include the deep denoiser FFDNet as 
%image/video priors along with TV, VBM4D, WNNM denoisers/priors for six 
%simulated benchmark video-SCI datasets (in grayscale). A full list and 
%comparison of PnP-SCI algorithms are in Tab. 1 and Figs. 2, 4.
%   TEST_PNPSCI_BENCHMARK_FULL(dataname) runs the PnP-SCI algorithms with 
%   TV and FFDNet as denoising priors for grayscale video SCI, where 
%   dataname denotes the name of the dataset for reconstruction with `kobe` 
%   data as default.
% Reference
%   [1] X. Yuan, Y. Liu, J. Suo, and Q. Dai, Plug-and-play Algorithms for  
%       Large-scale Snapshot Compressive Imaging, in IEEE/CVF Conf. Comput. 
%       Vis. Pattern Recognit. (CVPR), 2020.
%   [2] X. Yuan, Generalized alternating projection based total variation 
%       minimization for compressive sensing, in Proc. IEEE Int. Conf. 
%       Image Process. (ICIP), pp. 2539-2543, 2016.
% Dataset
%   Please refer to the readme file in `dataset` folder.
% Contact
%   Xin Yuan, Bell Labs, xyuan@bell-labs.com, initial version Jul 2, 2015.
%   Yang Liu, MIT CSAIL, yliu@csail.mit.edu, last update Apr 1, 2020.
%   
%   See also GAPDENOISE_CACTI, GAPDENOISE, TEST_PNPSCI_LARGESCALE.

% [0] environment configuration
addpath(genpath('../algorithms')); % algorithms
addpath(genpath('../packages'));   % packages
addpath(genpath('../utils'));      % utilities

datasetdir = '../dataset/simdata/benchmark'; % benchmark simulation dataset
resultdir  = '../results';                   % results

% [1] load dataset
if nargin<1
    dataname = 'kobe'; % [default] data name
end
datapath = sprintf('%s/%s.mat',datasetdir,dataname);

if exist(datapath,'file')
    load(datapath,'meas','mask','orig'); % meas, mask, orig
else
    error('File %s does not exist, please check dataset directory!',datapath);
end

nframe = size(meas, 3); % number of coded frames to be reconstructed
nmask  = size(mask, 3); % number of masks (or compression ratio B)
MAXB   = 255;           % maximum pixel value of the image (8-bit -> 255)

para.nframe = nframe; 
para.MAXB   = MAXB;

% [2] apply PnP-SCI for reconstruction
para.Mfunc  = @(z) A_xy(z,mask);
para.Mtfunc = @(z) At_xy_nonorm(z,mask);

para.Phisum = sum(mask.^2,3);
para.Phisum(para.Phisum==0) = 1;

% [2.0] common parameters
mask = single(mask);
orig = single(orig);

para.lambda   =    1; % correction coefficiency
para.acc      =    1; % enable acceleration
para.flag_iqa = true; % enable image quality assessments in iterations

%% [2.1] GAP-TV
para.denoiser = 'tv'; % TV denoising
  para.maxiter  = 100; % maximum iteration
  para.tvweight = 0.07*255/MAXB; % weight for TV denoising
  para.tviter   = 5; % number of iteration for TV denoising
  
[vgaptv,psnr_gaptv,ssim_gaptv,tgaptv,psnrall_tv] = ...
    gapdenoise_cacti(mask,meas,orig,[],para);

fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gaptv),mean(ssim_gaptv),tgaptv);

%% [2.2] GAP-VBM4D
para.denoiser = 'vbm4d'; % VBM4D denoising
  para.maxiter = 170; % maximum iteration

[vgapvbm4d,psnr_gapvbm4d,ssim_gapvbm4d,tgapvbm4d,psnrall_vbm4d] ...
= gapdenoise_cacti(mask,meas,orig,[],para);

fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapvbm4d),mean(ssim_gapvbm4d),tgapvbm4d);
 
%% [2.3] GAP-FFDNet
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
  
[vgapffdnet,psnr_gapffdnet,ssim_gapffdnet,tgapffdnet,psnrall_ffdnet] = ...
    gapdenoise_cacti(mask,meas,orig,[],para);

fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapffdnet),mean(ssim_gapffdnet),tgapffdnet);

%% [2.4] GAP-WNNM
para.denoiser = 'wnnm'; % WNNM denoising
  para.wnnm_int = true; % enable GAP-WNNM integrated
    para.blockmatch_period = 20; % period of block matching
  para.sigma   = [100 50 25 12  6]/MAXB; % noise deviation (to be estimated and adapted)
  para.vrange  = 1; % value range
  para.maxiter = [ 60 60 60 60 60];
  para.iternum = 1; % iteration number in WNNM
  para.enparfor = true; % enable parfor
  if para.enparfor % if parfor is enabled, start parpool in advance
      mycluster = parcluster('local');
      delete(gcp('nocreate')); % delete current parpool
      od = 1;
      while nmask/od > mycluster.NumWorkers
          od = od+1;
      end
      poolobj = parpool(mycluster,max(floor(nmask/od),1));
  end

[vgapwnnm,psnr_gapwnnm,ssim_gapwnnm,tgapwnnm,psnrall_wnnm] = ...
    gapdenoise_cacti(mask,meas,orig,[],para);

delete(poolobj); % delete pool object

fprintf('GAP-%s mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapwnnm),mean(ssim_gapwnnm),tgapwnnm);

%% [3.1] GAP-WNNM-TV
para.denoiser = 'wnnm'; % WNNM denoising
  para.wnnm_int = true; % enable GAP-WNNM integrated
    para.blockmatch_period = 20; % period of block matching
  para.sigma   = [12  6]/MAXB; % noise deviation (to be estimated and adapted)
  para.vrange  = 1; % value range
  para.maxiter = [60 80];
  para.iternum = 1; % iteration number in WNNM
  para.enparfor = true; % enable parfor
  if para.enparfor % if parfor is enabled, start parpool in advance
      mycluster = parcluster('local');
      delete(gcp('nocreate')); % delete current parpool
      od = 1;
      while nmask/od > mycluster.NumWorkers
          od = od+1;
      end
      poolobj = parpool(mycluster,max(floor(nmask/od),1));
  end

[vgapwnnm_tv,psnr_gapwnnm_tv,ssim_gapwnnm_tv,tgapwnnm3,psnrall_wnnm3] = ...
    gapdenoise_cacti(mask,meas,orig,vgaptv,para);

tgapwnnm_tv = tgapwnnm3 + tgaptv;
psnrall_wnnm_tv = [psnrall_tv, psnrall_wnnm3];
delete(poolobj); % delete pool object

fprintf('GAP-%s-TV mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapwnnm_tv),mean(ssim_gapwnnm_tv),tgapwnnm3);


%% [3.2] GAP-WNNM-VBM4D
para.denoiser = 'wnnm'; % WNNM denoising
  para.wnnm_int = true; % enable GAP-WNNM integrated
    para.blockmatch_period = 20; % period of block matching
  para.sigma   = [ 6]/MAXB; % noise deviation (to be estimated and adapted)
  para.vrange  = 1; % range of the input signal for denoising
  para.maxiter = [60];
  para.iternum = 1; % iteration number in WNNM
  para.enparfor = true; % enable parfor
  if para.enparfor % if parfor is enabled, start parpool in advance
      mycluster = parcluster('local');
      delete(gcp('nocreate')); % delete current parpool
      od = 1;
      while nmask/od > mycluster.NumWorkers
          od = od+1;
      end
      poolobj = parpool(mycluster,max(floor(nmask/od),1));
  end

[vgapwnnm_vbm4d,psnr_gapwnnm_vbm4d,ssim_gapwnnm_vbm4d,tgapwnnm2,psnrall_wnnm2] = ...
    gapdenoise_cacti(mask,meas,orig,vgapvbm4d,para);
  delete(poolobj); % delete pool object

tgapwnnm_vbm4d = tgapwnnm2 + tgapvbm4d;
psnrall_wnnm_vbm4d = [psnrall_vbm4d,psnrall_wnnm2];

fprintf('GAP-%s-VBM4D mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapwnnm_vbm4d),mean(ssim_gapwnnm_vbm4d),tgapwnnm2);

%% [3.3] GAP-WNNM-FFDNet
para.denoiser = 'wnnm'; % WNNM denoising
  para.wnnm_int = true; % enable GAP-WNNM integrated
    para.blockmatch_period = 20; % period of block matching
  para.sigma   = [ 6]/MAXB; % noise deviation (to be estimated and adapted)
  para.vrange  = 1; % range of the input signal for denoising
  para.maxiter = [60];
  para.iternum = 1; % iteration number in WNNM
  para.enparfor = true; % enable parfor
  if para.enparfor % if parfor is enabled, start parpool in advance
      mycluster = parcluster('local');
      delete(gcp('nocreate')); % delete current parpool
      od = 1;
      while nmask/od > mycluster.NumWorkers
          od = od+1;
      end
      poolobj = parpool(mycluster,max(floor(nmask/od),1));
  end

[vgapwnnm_ffdnet,psnr_gapwnnm_ffdnet,ssim_gapwnnm_ffdnet,tgapwnnm4,psnrall_wnnm4] = ...
    gapdenoise_cacti(mask,meas,orig,vgapffdnet,para);
  delete(poolobj); % delete pool object

tgapwnnm_ffdnet = tgapwnnm4 + tgapffdnet;
psnrall_wnnm_ffdnet = [psnrall_ffdnet,psnrall_wnnm4];

fprintf('GAP-%s-FFDNet mean PSNR %2.2f dB, mean SSIM %.4f, total time % 4.1f s.\n',...
    upper(para.denoiser),mean(psnr_gapwnnm_ffdnet),mean(ssim_gapwnnm_ffdnet),tgapwnnm4);

%% [3] save as the result .mat file and run the demonstration code
%       demo_gapdenoise.
matdir = [resultdir '/savedmat'];
if ~exist(matdir,'dir')
    mkdir(matdir);
end

save([matdir '/pnpsci_benchmark_full_' dataname num2str(nframe*nmask) '.mat']);

end

