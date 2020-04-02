# Plug-and-Play algorithms for Snapshot Compressive Imaging (PnP-SCI)
This repository contains the MATLAB code for the paper **Plug-and-play Algorithms for Large-scale Snapshot Compressive Imaging** in ***IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*** 2020 (**Oral**) by [Xin Yuan](https://www.bell-labs.com/usr/x.yuan), [Yang Liu](https://liuyang12.github.io/), [Jinli Suo](https://sites.google.com/site/suojinli/), and [Qionghai Dai](http://media.au.tsinghua.edu.cn/).
[[pdf]](https://arxiv.org/pdf/2003.13654 "arXiv preprint")   [[github]](https://github.com/liuyang12/PnP-SCI "github repository")   [[arXiv]](https://arxiv.org/abs/2003.13654 "arXiv preprint")  

<p align="center">
<img src="https://github.com/liuyang12/PnP-SCI/blob/master/results/video/pnpsci_largescale_football48.gif?raw=true">
</p>

Figure 1. Reconstructed large-scale `Football` video **(3840 × 1644 × 48)** using the proposed PnP-SCI algorithm with the deep denoiser [FFDNet](https://github.com/cszn/FFDNet) as the image/video prior, which is denoted as PnP-FFDNet (bottom-right). The ground truth and the results using GAP-TV (ICIP'16) are shown on the bottom-left and top-right, respectively for comparison. The captured image (top-left) size is **UHD (3840 × 1644)** and **48 frames** are recovered from **a snapshot measurement**. The `Football` video is from [a slow-motion 4K video clip](https://www.youtube.com/watch?v=EGAuWZYe2No "Falcons in 4K Super Slow Motion | YouTube").

## Snapshot compressive imaging (SCI)
Snapshot compressive imaging (SCI) asks the question that can we encode multi-dimensional visual information into low-dimensional sampling. Thus, SCI refers to encoding the three- or higher- dimensional data in a snapshot with a distinct mask (or coded aperture) for each slice of the data, as shown in Fig. 2. Typical applications are high-speed imaging (with temporally-variant masks), hyperspectral imaging (with spectrally-variant masks), light-field imaging (with angularly-variant masks), and simultaneously multidimensional imaging and sensing.

<p align="center">
<img src="https://github.com/liuyang12/PnP-SCI/blob/master/results/image/sci_system_demo.jpg?raw=true">
</p>

Figure 2. Sensing process of video SCI (left) and the reconstruction results using the proposed PnP-FFDNet (bottom-right). The captured
image (middle-top) size is **UHD (3840 × 1644)** and **48 frames** are recovered from **a snapshot measurement**. GAP-TV (top-right) takes
180 mins and PnP-FFDNet takes 55 mins for the reconstruction. All other methods are too slow (more than 12 hours) to be used.

### Large-scale SCI
The key challenge for SCI is the trade-off of the performance in terms of reconstruction quality and speed, especially when it comes to large-scale data (eg. UHD video data here).

Plug-and-play approach uses image/video deniosers as priors. Therefore, it could bridge the image/video processing community and the inverse problem community directly. PnP-SCI enjoys this benefit. Figure 3 shows the trade-off of quality and speed of various plug-and-play denoising algorithms for SCI reconstruction.

<p align="center">
<img src="https://github.com/liuyang12/PnP-SCI/blob/master/results/image/pnpsci_performance_tradeoff.jpg?raw=true" width="600">
</p>

Fig. 3.  Trade-off of quality (peak signal-to-noise ratio, PSNR in dB) and speed (1/runtime in 1/min) of various plug-and-play denoising algorithms for SCI reconstruction. The benchmark `Kobe` data (in grayscale) is used here for full comparison.
 
As we can see clearly, the deep denoiser FFDNet exhibits a better trade-off between the reconstruction quality and the speed. Therefore, PnP-FFDNet can be used as an *efficient baseline* in SCI reconstruction.

*Code for the video-SCI data from real systems would be available sooner.*

## Usage
### Download the PnP-SCI repository
0. The platform is MATLAB(R). [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html "Parallel Computing Toolbox | MathWorks(R)") (`parfor` for multi-CPU acceleration) is required to apply WNNM as the denoiser (as in DeSCI). [MatConvNet](https://www.vlfeat.org/matconvnet/ "MatConvNet: CNNs for MATLAB") is required to apply FFDNet as the denoiser (as in PnP-FFDNet), where compiling the GPU support is **strongly recommended**. *Note that you might find [this issue](https://github.com/vlfeat/matconvnet/issues/1143 "Problem Compiling with GPU Support on MATLAB 2018a") useful if you compile it on MATLAB 2018a and above.*
1. Download this repository via git or download the [zip file](https://github.com/liuyang12/PnP-SCI/archive/master.zip "PnP-SCI") manually.
```
git clone https://github.com/liuyang12/PnP-SCI
```

2. Download the large-scale dataset via [this link on Dropbox](https://www.dropbox.com/sh/6pzqxgv9aw1qqc2/AAABTmqSfTEA_i4E-p-TQJ0Sa?dl=0) and put the data in `./datasets/simdata/largescale`.

### Run PnP-SCI on benchmark video-SCI dataset
3. Test the PnP-SCI algorithm (on `Kobe` benchmark video-SCI dataset as default) via
```matlab
test_pnpsci
```
or (optionally) include the `dataname` when running the `test_pnpsci()` function, *i.e.*,
```matlab
test_pnpsci('traffic')
```

Note that we have decreased the number of iterations in PnP-SCI for some benchmark datasets (*e.g.*, to 5 in `Traffic` and `Crash` data, and 3 in `Aerial` data) to avoid the over-smoothing effect because of the mismatch between the estimated noise level and the real noise level for each iteration. 

### Run PnP-SCI on large-scale video-SCI dataset
4. Test the PnP-SCI algorithm (on `Messi` large-scale video-SCI dataset as default) via
```matlab
test_pnpsci_largescale
```
or (optionally) include the `dataname` when running the `test_pnpsci_largescale()` function, *i.e.*,
```matlab
test_pnpsci_largescale('football')
```

### Run full-comparison of PnP-SCI algorithms
5. [Optional] 
```matlab
tests/test_pnpsci_benchmark_full              % benchmark `Kobe` data
tests/test_pnpsci_benchmark_full('traffic')   % benchmark `Traffic` data
tests/test_pnpsci_largescale_full             % large-scale `Messi` data
tests/test_pnpsci_largescale_full('football') % large-scale `Football` datat
```

*Notice: Please donot wait for the results immediately after getting the full-comparison code to run. Because patch-based denoising priors are slow. That is why we need deep priors!*

## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `algorithms` | MATLAB functions of main algorithms proposed in the paper (original) | 
| `tests`    | MATLAB scripts to reproduce the results in the paper (original) |
| `packages`   | algorithms adapted from the state-of-art algorithms (adapted)|
| `dataset`    | data used for reconstruction (simulated) |
| `results`    | results of reconstruction (after reconstruction) |
| `utils`      | utility functions |

## Platform
The test platform is MATLAB(R) 2019b operating on Ubuntu 18.04 LTS (x64) with an Intel(R) Core(TM) 10-core processor at 3.00 GHz and 64 GB RAM. It can run on any machine with MATLAB(R) and Parallel Computing Toolbox, operating on Windows(R), Linux, or Mac OS. GPU is not required (but *recommended* for PnP-FFDNet) to run this code.

## Citation
```
@inproceedings{Yuan20PnPSCI,
   author    = {Yuan, Xin and Liu, Yang and Suo, Jinli and Dai, Qionghai},
   title     = {Plug-and-play Algorithms for Large-scale Snapshot Compressive Imaging},
   booktitle = {IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)},
   publisher = {IEEE/CVF},
   year      = {2020},
   arxiv     = {2003.13654},
   type      = {Conference Proceedings}
}
```

## Contact
[Xin Yuan, Bell Labs](mailto:xyuan@bell-labs.com "Xin Yuan, Bell labs")   

[Yang Liu, MIT CSAIL](mailto:yliu@csail.mit.edu "Yang Liu, MIT CSAIL") 
