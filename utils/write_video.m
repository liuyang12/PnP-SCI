function write_video(imseq,fps,vdir)
%WRITE_VIDEO Write the image sequence to a video.
%   WRITE_VIDEO(imseq,fps,vdir) write the image sequence imseq to a 
%   destinated video directory vdir.
outvideo = VideoWriter(vdir,'Uncompressed AVI');
outvideo.FrameRate = fps;
open(outvideo);

if isstruct(imseq) % 1*1 structure or 1*F structure array
    writeVideo(outvideo,imseq);
else % 3-D array (grayscale) or 4-D array (color) with the last dimension 
     % denoting the frame number.
    otherdims = repmat({':'},1,ndims(imseq)-1);
    for ii = 1:size(imseq,ndims(imseq))
        im = imseq(otherdims{:},ii);
        writeVideo(outvideo,uint8(im));
    end
end

close(outvideo);
end

