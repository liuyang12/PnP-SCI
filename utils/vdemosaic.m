function vdemo = vdemosaic( vmo, opt )
%VDEMOSAIC Video demosaic.
nc              = 3; % number of color [no input]
rotnum          = 0; % number of 90 degrees for rotation of each frame
sensorAlignment = 'rggb'; % sensorAlignment of RGB Bayer pattern
if isfield(opt,'rotnum'),                   rotnum = opt.rotnum; end
if isfield(opt,'sensorAlignment'), sensorAlignment = opt.sensorAlignment; end

[nrow,ncol,nim] = size(vmo);

vdescend = sort(vmo(:),'descend'); % sort by descending order
vmax = vdescend(round( 0.002*length(vdescend) )); % why 0.002?

vdemo_rot = zeros([nrow ncol nc nim],'uint8');
% frame-wise demosaic
for iim = 1:nim
    im = uint8(min(vmo(:,:,iim)/vmax,1)*255);
    vdemo_rot(:,:,:,iim) = demosaic(im,sensorAlignment);
end

% RGB frame-wise rotation
for iim = 1:nim
    for ic = 1:nc
        vdemo(:,:,ic,iim) = rot90(vdemo_rot(:,:,ic,iim),rotnum);
    end
end

end

