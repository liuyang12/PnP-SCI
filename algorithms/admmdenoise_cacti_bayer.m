function [v_,psnr_,ssim_,t_,psnrall] = admmdenoise_cacti_bayer( ...
    mask,meas,orig,v0,para)
%ADMMDENOISE_CACTI_BAYER ADMM-Denoise framework for recontruction of CACTI 
%high-speed imaging with RGB Bayer sensor.
%   See also ADMMDENOISE, TEST_ADMMDENOISE.
iframe   = 1; % start frame number
maskdirection  = 'updown'; % direction of the mask
projmeth = 'admm'; % projection method (ADMM for default)
if isfield(para,'iframe');               iframe = para.iframe; end
if isfield(para,'projmeth');           projmeth = para.projmeth; end
if isfield(para,'maskdirection'); maskdirection = para.maskdirection; end
[nrow,ncol,nmask]  = size(mask);
nframe = para.nframe;
MAXB   = para.MAXB;

bayer = [1 1; 1 2; 2 1; 2 2]; % Bayer pattern (R G1 G2 B)
[nbayer,~] = size(bayer);

psnrall = [];
% ssimall = [];
v_bayer = zeros([nrow ncol nmask],'like',meas);
v_ = zeros([nrow ncol nmask*nframe],'like',meas);
tic
% coded-frame-wise denoising
for kf = 1 : nframe
    fprintf('%s-%s Reconstruction frame-block %d of %d ...\n',...
        upper(para.projmeth),upper(para.denoiser),kf,nframe);
    % if ~isempty(orig)
    %     % para.orig = orig(:,:,(kf-1+iframe-1)*nmask+(1:nmask))/MAXB;
    %     origall = orig(:,:,(kf-1+iframe-1)*nmask+(1:nmask))/MAXB;
    % end
    y = meas(:,:,kf+iframe-1)/MAXB;
    
    for ibayer = 1:nbayer % Bayer element-wise
        b = bayer(ibayer,:);
        yb = y( b(1):2:end, b(2):2:end ); % single-channel measurement
        Phib = mask( b(1):2:end, b(2):2:end, : ); % single-channel mask
        
        if ~isempty(orig)
            switch lower(maskdirection)
                case 'plain'
                    para.orig = orig(b(1):2:end,b(2):2:end,(kf-1+iframe-1)*nmask+(1:nmask))/MAXB;
                case 'updown'
                    if mod(kf+iframe-1,2) == 0 % even frame (falling of triangular wave)
                        para.orig = orig(b(1):2:end,b(2):2:end,(kf-1+iframe-1)*nmask+(1:nmask))/MAXB;
                    else % odd frame (rising of triangular wave)
                        para.orig = origall(b(1):2:end,b(2):2:end,(kf-1+iframe-1)*nmask+(nmask:-1:1))/MAXB;
                    end
                case 'downup'
                    if mod(kf+iframe-1,2) == 1 % odd frame (rising of triangular wave)
                        para.orig = orig(b(1):2:end,b(2):2:end,(kf-1+iframe-1)*nmask+(1:nmask))/MAXB;
                    else % even frame (falling of triangular wave)
                        para.orig = orig(b(1):2:end,b(2):2:end,(kf-1+iframe-1)*nmask+(nmask:-1:1))/MAXB;
                    end
                otherwise
                    error('Unsupported mask direction %s!',lower(maskdirection));
            end
        end
        if isempty(v0) % raw initialization
            para.v0 = [];
        else % given initialization
            switch lower(maskdirection)
                case 'plain'
                    para.v0 = v0(b(1):2:end,b(2):2:end,(kf-1)*nmask+(1:nmask));
                case 'updown'
                    if mod(kf+iframe-1,2) == 0 % even frame (falling of triangular wave)
                        para.v0 = v0(b(1):2:end,b(2):2:end,(kf-1)*nmask+(1:nmask));
                    else % odd frame (rising of triangular wave)
                        para.v0 = v0(b(1):2:end,b(2):2:end,(kf-1)*nmask+(nmask:-1:1));
                    end
                case 'downup'
                    if mod(kf+iframe-1,2) == 1 % odd frame (rising of triangular wave)
                        para.v0 = v0(b(1):2:end,b(2):2:end,(kf-1)*nmask+(1:nmask));
                    else % even frame (falling of triangular wave)
                        para.v0 = v0(b(1):2:end,b(2):2:end,(kf-1)*nmask+(nmask:-1:1));
                    end
                otherwise
                    error('Unsupported mask direction %s!',lower(maskdirection));
            end
        end
        
        para.Mfunc  = @(z) A_xy(z,Phib);
        para.Mtfunc = @(z) At_xy_nonorm(z,Phib);

        para.Phisum = sum(Phib.^2,3);
        para.Phisum(para.Phisum==0) = 1;
        switch lower(projmeth)
            case 'gap' % GAP-Denoise
                if isfield(para,'wnnm_int') && para.wnnm_int % GAP-WNNM integrated
                    if isempty(orig) || (isfield(para,'flag_iqa') && ~para.flag_iqa) % ImQualAss disabled
                        v = gapwnnm_int(yb,para);
                    else
                        [v,psnrall(kf,:)] = gapwnnm_int(yb,para);
                    end
                else
                    if isempty(orig) || (isfield(para,'flag_iqa') && ~para.flag_iqa) % ImQualAss disabled
                        v = gapdenoise(yb,para);
                    else
                        [v,psnrall(kf,:)] = gapdenoise(yb,para);
                    end
                end
            case 'admm' % ADMM-Denoise
                if isfield(para,'wnnm_int') && para.wnnm_int % GAP-WNNM integrated
                    if isempty(orig) || (isfield(para,'flag_iqa') && ~para.flag_iqa) % ImQualAss disabled
                        v = admmwnnm_int(yb,para);
                    else
                        [v,psnrall(kf,:)] = admmwnnm_int(yb,para);
                    end
                else
                    if isempty(orig) || (isfield(para,'flag_iqa') && ~para.flag_iqa) % ImQualAss disabled
                        v = admmdenoise(yb,para);
                    else
                        [v,psnrall(kf,:)] = admmdenoise(yb,para);
                    end
                end
            otherwise
                error('Unsupported projection method %s!',projmeth);
        end
        v_bayer( b(1):2:end, b(2):2:end, : ) = v; % bayered video frame
    end % bayer loop [4]
    switch lower(maskdirection)
        case 'plain'
            v_(:,:,(kf-1)*nmask+(1:nmask)) = v_bayer;
        case 'updown'
            if mod(kf+iframe-1,2) == 0 % even frame (falling of triangular wave)
                v_(:,:,(kf-1)*nmask+(1:nmask)) = v_bayer;
            else % odd frame (rising of triangular wave)
                v_(:,:,(kf-1)*nmask+(nmask:-1:1)) = v_bayer;
            end
        case 'downup'
            if mod(kf+iframe-1,2) == 1 % odd frame (rising of triangular wave)
                v_(:,:,(kf-1)*nmask+(1:nmask)) = v_bayer;
            else % even frame (falling of triangular wave)
                v_(:,:,(kf-1)*nmask+(nmask:-1:1)) = v_bayer;
            end
        otherwise
            error('Unsupported mask direction %s!',lower(maskdirection));    
    end
end % frame loop [nframe]
t_ = toc;
% image quality assessments
psnr_ = zeros([1 nmask*nframe]);
ssim_ = zeros([1 nmask*nframe]);
if ~isempty(orig)
    for kv = 1:nmask*nframe
        psnr_(kv) = psnr(double(v_(:,:,kv)),double(orig(:,:,kv+(iframe-1)*nmask))/MAXB,255/MAXB);
        ssim_(kv) = ssim(double(v_(:,:,kv)),double(orig(:,:,kv+(iframe-1)*nmask))/MAXB);
    end
end

end

