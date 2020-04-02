function [denoisedim,psnrall] = ffdnet_imdenoise(noisyim,orgim,para)
%FFDNET_IMDENOISE Fast and flexible denoising convolutional neural network 
%(FFDNet)-based image denoising.
format compact;
global sigmas; % input noise level or input noise level map
para.useGPU = false;
ffdnetnorm = true;
if isfield(para,'useGPU'),         useGPU = para.useGPU;             end
if isfield(para,'ffdnetnorm'), ffdnetnorm = para.ffdnetnorm;         end

if ffdnetnorm
    %%%%% [start] normalization, like VBM4D %%%%%
    maxz = max(noisyim(:));
    minz = min(noisyim(:));
    scale = 0.7;
    shift = (1-scale)/2;
    noisyim = (noisyim-minz)/(maxz-minz);
    noisyim = noisyim*scale+shift;

    sigmas = para.sigma/(maxz-minz)*scale;
    %%%%% [start] normalization, like VBM4D %%%%%
else
    % set noise level map
    sigmas = para.sigma; % see "vl_simplenn.m".
end

if isfield(para,'net') && ~isempty(para.net)
    net = para.net;
else
    load(fullfile('models','FFDNet_gray.mat'),'net');
    net = vl_simplenn_tidy(net);
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end
end

input = double(noisyim);

% perform denoising
% res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
if useGPU
    input = gpuArray(input);
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    output = res(end).x;
    output = gather(output);
else
    res    = vl_simplenn(net,single(input),[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    output = res(end).x;
end
% res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
% res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow

denoisedim = double(output);

if ffdnetnorm
    %%%%% [start] de-normalization, like VBM4D %%%%%
    denoisedim = (denoisedim-shift)/scale;
    denoisedim = denoisedim*(maxz-minz)+minz;
    %%%%% [start] de-normalization, like VBM4D %%%%%
end


end

