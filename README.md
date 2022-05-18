# CA-MSN-action-recognition
Ronghao Dang, Chengju Liu, Ming Liu, Qijun Chen (Under review for AI COMMUNICATIONS)

## Abstract
3D skeleton data has been widely used in action recognition as skeleton-based action
recognition has achieved good performance in complex dynamic environments. With
the rise of spatio-temporal graph convolution, it has been attracted much attention to
use graph convolution to extract spatial and temporal features together in the field of
skeleton-based action recognition. However, due to the huge difference in the focus of
spatial and temporal features, it is difficult to improve the efficiency of combined
extraction to spatiotemporal features. In this paper, we propose a channel attention
and multi-scale neural network (CA-MSN) for skeleton-based action recognition with a
series of spatio-temporal extraction modules. We exploit the relationship of body joints
hierarchically through two modules, i.e., a spatial module which uses the residual GCN
network with the channel attention block to extract the high-level spatial feature
between the body joints in each frame, and a temporal module which uses the multiscale
TCN network to extract the temporal feature at different scales. We perform
extensive experiments on NTU-RGBD60 and NTU-RGBD120 datasets and verify the
effectiveness of our network. The comparison results show that our method achieves
state-of-the-art performance under the premise of ensuring the calculation speed.

## Prerequisites
-Python 3.6
-Anaconda
-Pytorch 1.3


