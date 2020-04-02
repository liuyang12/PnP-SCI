function [v,psnrall] = admmwnnm_int( y, opt )
%ADMMWNNM_INT_RES Integrated ADMM and WNNM for snapshot compressive imaging.
%   Note integrated version of ADMM-WNNM instead of embedded for
%   acceleration of WNNM with less block-match process for multiple
%   iterations.
%   See also ADMMDENOISE, GAPWNNM_INT, ADMMWNNM_INT.
if nargin<2
    opt = [];
end
% [0] default parameter configuration, to be specified
A  = @(x) M_func(x);
At = @(z) Mt_func(z);
if isfield(opt,'Mfunc'),  A  = @(x) opt.Mfunc(x);  end
if isfield(opt,'Mtfunc'), At = @(z) opt.Mtfunc(z); end
% Phisum   = ;
% v0       = At(y); % start point (initialization of iteration)
lambda   = 1;     % correction coefficiency
gamma    = 1e-3;    % regularization factor for the noise entry
maxiter  = 300;     % maximum number of iteration
acc      = 1;       % enable acceleration
sigma    = 10/255;  % noise deviation 
flag_iqa = true;    % flag of showing image quality assessments

if isfield(opt,'Phisum'),     Phisum = opt.Phisum;   end
if isfield(opt,'v0'),             v0 = opt.v0;       end
if isfield(opt,'lambda'),   lambda   = opt.lambda;   end
if isfield(opt,'gamma'),       gamma = opt.gamma;    end
if isfield(opt,'maxiter'),   maxiter = opt.maxiter;  end
if isfield(opt,'acc'),           acc = opt.acc;      end
if isfield(opt,'sigma'),       sigma = opt.sigma;    end
if isfield(opt,'flag_iqa'), flag_iqa = opt.flag_iqa; end

if ~exist('v0','var') || isempty(v0)
    v0 = At(y); % start point (initialization of iteration)
end
y1 = zeros(size(y),'like',y);
psnrall = []; % return empty with no ground truth
% ssimall = []; % return empty with no ground truth
% [1] start iteration
v = v0; % initialization
theta = v0;
b = zeros(size(v0),'like',v0);
% opt.sigma = sigma;
k = 1; % current number of iteration
nonlocalarr = uint32.empty;
vindarr     = uint8.empty;
for isig = 1:length(sigma)
    opt.sigma = sigma(isig);
    for iter = 1:maxiter(isig)
        for ii = 1:1
            % [1.1] Euclidean projection
            yb = A(theta+b);
            v = (theta+b)+lambda*(At((y-yb)./(Phisum+gamma))); % lambda=1 for ADMM
        end
        % [1.2] Denoising to match the video prior
        % WNNM video denoising (MATLAB-style matrix version)
        % v = wnnm_vdenoise(v,[],opt); % opt.sigma
        noisyv = v-b;
        para = vdefparaconf(opt); % default parameter configuration

        [nrow,ncol,nframe] = size(noisyv); % grayscale video (color todo)
        % [1] pre-calculation of the indexes of the neighbors within the search
        %     window
        [neighborindarr,neighbornumarr,selfindarr] = neighborind([nrow ncol],para);

        % [2] WNNM denoisng for several iterations
        estv = noisyv;
        for iit = 1:para.iternum
            % correction between adjacent iterations
            if ~para.adaptboost % 
                estv = estv + para.delta*(noisyv-estv); 
            else % adaptive boosting for WNNM-based denoising
                Eestv = mean(abs(estv(:)).^2);
                Enoise  = para.abeta*abs(para.nsigma^2-var(noisyv(:)-estv(:)));
                rho = sqrt(Eestv)/(sqrt(Eestv)+sqrt(max(Eestv-Enoise,0)));
                fprintf('    Iteration % 2d, rho = %.3f.\n',iit,rho);
                estv = estv + (1-rho)*(noisyv-estv); 
            end
            % use all frames for denoising
            curvall = zeros(nrow,ncol,nframe,nframe);
            curfall = zeros(nrow,ncol,nframe,nframe);
            
            % [2.1] video to patches
            [rawpatchmat,nsigmamat] = v2patch(estv,noisyv,para);
            blockmatch_period = para.blockmatch_period;
            % inner loop to reuse the block-matching results
                if mod(k,blockmatch_period) == 1 % block matching periodically
                    nonlocalarr = uint32.empty;
                    vindarr     = uint8.empty;
                    parfor (iframe = 1:nframe,nframe) % maximum nframe parpool for optimal
                    % [2.2] calculate the patches with non-local similarity for each 
                    %       key patch
                    [curnonlocalarr,curvindarr] = vblockmatch(iframe,rawpatchmat,...
                        neighborindarr,neighbornumarr,selfindarr,para);
                    nonlocalarr(:,:,iframe) = curnonlocalarr;
                    vindarr(:,:,iframe) = curvindarr;
                    end
                end
                if iit==1 % initial noise level of each patch
                    nsigmamat = para.nsigma*ones(size(nsigmamat));
                end
            % frame-wise denosing using parfor instead of for to get accelerated
            % performance, requiring Parallel Computing Toolbox.
            
            parfor (iframe = 1:nframe,nframe) % maximum nframe parpool for optimal
            % for iframe = 1:nframe % maximum nframe parpool for optimal
                % use all frames for denoising
                % inner loop to reuse the block-matching results
                % if mod(k,blockmatch_period) == 1 % block matching periodically
                %     % [2.2] calculate the patches with non-local similarity for each 
                %     %       key patch
                %     [nonlocalarr(:,:,iframe),vindarr(:,:,iframe)] = vblockmatch(iframe,rawpatchmat,...
                %         neighborindarr,neighbornumarr,selfindarr,para);
                % end
                curnonlocalarr = nonlocalarr(:,:,iframe);
                curvindarr = vindarr(:,:,iframe);
                % [2.3] patch estimation by means of WNNM
                [estpatchmat,frqpatchmat] = vpatchestimate(iframe,curnonlocalarr,...
                    curvindarr,rawpatchmat,nsigmamat,selfindarr,para);
                % [2.4] aggregate overlapped patches to the whole image
                [curv,curf] = patch2v(estpatchmat,frqpatchmat,size(noisyv),...
                    para.patchsize);
                % use all frames for denoising
                curvall(:,:,:,iframe) = curv;
                curfall(:,:,:,iframe) = curf;
            end
%             %%%%%%%%%%%%%%%%% [start] temporal trial %%%%%%%%%%%%%%%%%%%%%%
%             %%%%% devide the parfor loop into two seperate one to %%%%%
%             %%%%% to fit the maximum number of parpool workers,   %%%%%
%             %%%%% that is CPU cores available.                    %%%%%
%             % inner loop to reuse the block-matching results
%                 if mod(k,blockmatch_period) == 1 % block matching periodically
%                     parfor (iframe = 1:nframe/2,nframe/2) % maximum nframe parpool for optimal
%                     % [2.2] calculate the patches with non-local similarity for each 
%                     %       key patch
%                     [nonlocalarr(:,:,iframe),vindarr(:,:,iframe)] = vblockmatch(iframe,rawpatchmat,...
%                         neighborindarr,neighbornumarr,selfindarr,para);
%                     end
%                     parfor (iframe = nframe/2+1:nframe,nframe/2) % maximum nframe parpool for optimal
%                     % [2.2] calculate the patches with non-local similarity for each 
%                     %       key patch
%                     [nonlocalarr(:,:,iframe),vindarr(:,:,iframe)] = vblockmatch(iframe,rawpatchmat,...
%                         neighborindarr,neighbornumarr,selfindarr,para);
%                     end
%                 end
%                 if iit==1 % initial noise level of each patch
%                     nsigmamat = para.nsigma*ones(size(nsigmamat));
%                 end
%             % frame-wise denosing using parfor instead of for to get accelerated
%             % performance, requiring Parallel Computing Toolbox.
%             
%             parfor (iframe = 1:nframe/2,nframe/2) % maximum nframe parpool for optimal
%             % for iframe = 1:nframe % maximum nframe parpool for optimal
%                 % use all frames for denoising
%                 % inner loop to reuse the block-matching results
%                 % if mod(k,blockmatch_period) == 1 % block matching periodically
%                 %     % [2.2] calculate the patches with non-local similarity for each 
%                 %     %       key patch
%                 %     [nonlocalarr(:,:,iframe),vindarr(:,:,iframe)] = vblockmatch(iframe,rawpatchmat,...
%                 %         neighborindarr,neighbornumarr,selfindarr,para);
%                 % end
%                 % [2.3] patch estimation by means of WNNM
%                 [estpatchmat,frqpatchmat] = vpatchestimate(iframe,nonlocalarr(:,:,iframe),...
%                     vindarr(:,:,iframe),rawpatchmat,nsigmamat,selfindarr,para);
%                 % [2.4] aggregate overlapped patches to the whole image
%                 [curv,curf] = patch2v(estpatchmat,frqpatchmat,size(noisyv),...
%                     para.patchsize);
%                 % use all frames for denoising
%                 curvall(:,:,:,iframe) = curv;
%                 curfall(:,:,:,iframe) = curf;
%             end
%             parfor (iframe = nframe/2+1:nframe,nframe/2) % maximum nframe parpool for optimal
%             % for iframe = 1:nframe % maximum nframe parpool for optimal
%                 % use all frames for denoising
%                 % inner loop to reuse the block-matching results
%                 % if mod(k,blockmatch_period) == 1 % block matching periodically
%                 %     % [2.2] calculate the patches with non-local similarity for each 
%                 %     %       key patch
%                 %     [nonlocalarr(:,:,iframe),vindarr(:,:,iframe)] = vblockmatch(iframe,rawpatchmat,...
%                 %         neighborindarr,neighbornumarr,selfindarr,para);
%                 % end
%                 % [2.3] patch estimation by means of WNNM
%                 [estpatchmat,frqpatchmat] = vpatchestimate(iframe,nonlocalarr(:,:,iframe),...
%                     vindarr(:,:,iframe),rawpatchmat,nsigmamat,selfindarr,para);
%                 % [2.4] aggregate overlapped patches to the whole image
%                 [curv,curf] = patch2v(estpatchmat,frqpatchmat,size(noisyv),...
%                     para.patchsize);
%                 % use all frames for denoising
%                 curvall(:,:,:,iframe) = curv;
%                 curfall(:,:,:,iframe) = curf;
%             end
%             %%%%%%%%%%%%%%%%% [end] temporal trial %%%%%%%%%%%%%%%%%%%%%%

            % use all frames for denoising
            v_ = sum(curvall,ndims(curvall));
            f_ = sum(curfall,ndims(curfall));

            estv = v_./(f_+eps);

            if mod(iit-1,para.innerloop)==0 
                % % [2.2] calculate the patches with non-local similarity for 
                % %       each key patch
                % [nonlocalarr,vindarr] = vblockmatch(cframe,rawpatchmat,...
                %     neighborindarr,neighbornumarr,selfindarr,para);
                % less non-local patches with lower noise level
                para.patchnum = para.patchnum-10; 
            end
            % [1.4] save and show intermediate results of psnr and ssim
            if flag_iqa && isfield(opt,'orig') && (~isempty(opt.orig))
                psnrall(k) = psnr(double(estv),opt.orig); % record all psnr
                % ssimall(k) = ssim(double(estv),opt.orig); % record all ssim
                if (mod(k,5)==0) 
                    fprintf('  ADMM-%s iteration % 4d, sigma %.1f, PSNR %2.2f dB.\n',...
                        upper(opt.denoiser),k,opt.sigma*255,psnrall(k));
                end
            end
            k = k+1;
        end % WNNM loop [iternum]
        theta = estv;
        b = b-(v-theta); % update residual
        % % [1.3] update noise standard deviation
        % nsigma = eta*sqrt(abs(sigma^2-var(v(:)-v0(:))));
        % opt.sigma = nsigma;
        
    end % GAP loop [maxiter]
end % sigma loop [length(sigma)]

end            
            
            
