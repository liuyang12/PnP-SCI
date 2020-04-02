function [v, psnrall] = gapdenoise( y , opt )
%GAPDENOISE Generalized alternating projection (GAP)-based denoising
%framework for compressive sensing reconstruction.
%   v=GAPDENOISE(y,opt) returns the reconstruction result v of the
%   measurements with CASSI or CACTI coding, where y is the measurement
%   matrix, opt is the parameters for the GAP-Denoise algorithm, typically
%   the denoiser applied in the framework.
% Reference
%   [1] Y. Liu, X. Yuan, J. Suo, D.J. Brady, and Q. Dai, Rank Minimization 
%       for Snapshot Compressive Imaging, IEEE Trans. Pattern Anal. Mach. 
%       Intell. (TPAMI), DOI:10.1109/TPAMI.2018.2873587, 2018.
%   [2] X. Yuan, Generalized alternating projection based total variation 
%       minimization for compressive sensing, in Proc. IEEE Int. Conf. 
%       Image Process. (ICIP), pp. 2539-2543, 2016.
% Code credit
%   Xin Yuan, Bell Labs, xyuan@bell-labs.com, initial version Jul 2, 2015.
%   Yang Liu, Tsinghua University, y-liu16@mails.tsinghua.edu.cn, last
%     update Jul 13, 2018.
% 
%   See also GAPDENOISE_CACTI.
if nargin<2
    opt = [];
end
% [0] default parameter configuration, to be specified
A  = @(x) M_func(x);
At = @(z) Mt_func(z);
if isfield(opt,'Mfunc'),  A  = @(x) opt.Mfunc(x);  end
if isfield(opt,'Mtfunc'), At = @(z) opt.Mtfunc(z); end

denoiser = 'tv';  % video denoiser
% v0       = At(y); % start point (initialization of iteration)
lambda   = 0.2;     % correction coefficiency
maxiter  = 100;     % maximum number of iteration
sigma    = 10/255;  % noise deviation 
acc      = 1;       % enable acceleration
tvweight = 0.07;    % weight for TV denoising
tviter   = 5;       % number of iteration for TV denoising
nosestim = true;    % enable noise estimation (if possible)
flag_iqa = true;    % flag of showing image quality assessments (be sure 
                    %  to turn it off for benchmark)
ffdnetvnorm_init = true; % use normalized video as input for initialization
                         %  (with the first 10 iterations)
                    
if isfield(opt,'Phisum'),     Phisum = opt.Phisum;   end
if isfield(opt,'denoiser'), denoiser = opt.denoiser; end
if isfield(opt,'v0'),             v0 = opt.v0;       end
if isfield(opt,'lambda'),   lambda   = opt.lambda;   end
if isfield(opt,'maxiter'),   maxiter = opt.maxiter;  end
if isfield(opt,'acc'),           acc = opt.acc;      end
if isfield(opt,'tvweight'), tvweight = opt.tvweight; end
if isfield(opt,'tviter'),     tviter = opt.tviter;   end
if isfield(opt,'nosestim'), nosestim = opt.nosestim; end
if isfield(opt,'sigma'),       sigma = opt.sigma;    end
if isfield(opt,'flag_iqa'), flag_iqa = opt.flag_iqa; end
if isfield(opt,'ffdnetvnorm_init'), ffdnetvnorm_init = opt.ffdnetvnorm_init; end

if ~exist('v0','var') || isempty(v0)
    v0 = At(y); % start point (initialization of iteration)
end

if  isfield(opt,'ffdnetvnorm') && ffdnetvnorm_init
    sigma = [50/255 sigma];
    maxiter = [10 maxiter];
    ffdnetvnorm = opt.ffdnetvnorm;
end

y1 = zeros(size(y),'like',y);

% [1] start iteration
v = v0; % initialization
k = 1; % current number of iteration
psnrall = []; % return empty with no ground truth
for isig = 1:length(maxiter) % extension for a series of noise levels
    nsigma = sigma(isig); 
    opt.sigma = nsigma;
    for iter = 1:maxiter(isig)
        % [1.1] Euclidean projection
        yb = A(v);
        if acc % enable acceleration
            y1 = y1+(y-yb);
            v = v+lambda*(At((y1-yb)./Phisum)); % v=v+lambda*(At*A)^-1*At*dy
        else
            v = v+lambda*(At((y-yb)./Phisum));
        end
        % [1.2] Denoising to match the video prior
        switch lower(denoiser)
            case 'tv' % TV denoising
                v = TV_denoising(v,tvweight,tviter);
            case 'vbm3d' % VBM3D denoising (normalization required)
                [~,v] = VBM3D(v,nsigma,0,0); % nsigma
            case 'vbm4d' % VBM4D denoising
                if nosestim % noise estimation enabled
                    v = vbm4d(v,-1,'lc',1,1,1,0); % -1 to enable noise estimation
                else % noise estimation disabled
                    v = vbm4d(v,nsigma,'lc',1,1,1,0); % -1 to enable noise estimation
                end
            case 'bm4d' % BM4D denoising
                if nosestim % noise estimation enabled
                    v = bm4d(v,'Gauss',0,'lc',1,0); % 0 to enable noise estimation
                else % noise estimation disabled
                    v = bm4d(v,'Gauss',nsigma,'lc',1,0); % 0 to enable noise estimation
                end
            case 'wnnm' % WNNM video denoising (MATLAB-style matrix version)
                v = wnnm_vdenoise(v,[],opt); % opt.sigma
            case 'ffdnet' % FFDNet video denoising (frame-wise)
                if ffdnetvnorm_init
                    if isig==1
                        opt.ffdnetvnorm = true;
                    else
                        opt.ffdnetvnorm = ffdnetvnorm;
                    end
                end
                v = ffdnet_vdenoise(v,[],opt); % opt.sigma
            otherwise
                error('Unsupported denoiser %s!',denoiser);
        end
        % [1.3] save and show intermediate results of psnr (and ssim)
        if flag_iqa && isfield(opt,'orig') && (~isempty(opt.orig))
            psnrall(k) = psnr(double(v),double(opt.orig)); % record all psnr
            if (mod(k,5)==0) 
                fprintf('  GAP-%s iteration % 4d, sigma % 3d, PSNR %2.2f dB.\n',...
                    upper(opt.denoiser),k,nsigma*255,psnrall(k));
            end
        end
        k = k+1;
    end % GAP loop [maxiter]
end % sigma loop [length(maxiter)]

end            
            
            