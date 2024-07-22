# A Channel-Split Medical Image Semantic Segmentation Network Integrating Fusion of Differentiated Information

paper address:https://github.com/YF-W/MIR-CSNet

***Semantic segmentation is crucial for advancing intelligent healthcare by enabling accurate labeling and delineation of lesion areas in medical images. To address limitations in traditional U-shaped networks, which only scale channels at encoding and decoding stages and show inadequate attention to background information, we developed MIR-CSNet (Multiscale Information Reorganization with Channel-Split Networks). This network includes IACI Net (Interactively Arranging Channel Information Network), which enhances multilevel information fusion by dynamically splitting, merging, and rearranging channels. Additionally, two pluggable modules, FB-SR (Both Foreground and Background with Semantic Recovery Module) and SSOC (Symmetrical Split Overlap Capture Module), are introduced to improve background extraction and feature integration. These innovations address the insufficient information fusion and low attention to background details observed in traditional approaches.***

## Paper:MIR-CSNet(Multiscale Information Reorganization with Channel-Split Networks)

**Authors:Yuefei Wang, Li Zhang, Yutong Zhang, Yuquan Xu, Zhixuan Chen, Yuxuan Wan, Zongyuan Liu, Pingwen Shi, Ruixin Cao, Guokai Chen, Binxiong Li, Yixi Yang, Xi Yu**

### 1. Architecture Overview

![MIR-CSNet](https://github.com/YF-W/MIR-CSNet/blob/12c93619b4ce022acbb99adeeebec143d6d94b12/MIR-CSNet.png)

***MIR-CSNet builds on the foundation of IACI Net by creating a single encoder-decoder network with channel-split multi-scale information extraction. The encoding section uses IACI Net's channel-split merging structure to combine multi-level semantic information through various convolutional layers. The decoding end replaces IACI Net with FB-SR to integrate both foreground and background information for image restoration. Additionally, the skip-connection component incorporates SSOC to address the loss of channel information integrity due to repeated channel-splitting, facilitating differentiated feature extraction and aggregation of key semantics.***

### 2. Our network baseline

![MIR-CSNet_Baseline](https://github.com/YF-W/MIR-CSNet/blob/1683c3bedf227b29d1386532e3d90a8d839677c9/MIR-CSNet_Baseline.png)

***We introduce the IACI Net, which utilizes a channel-split single encoder-single decoder architecture. Unlike traditional networks where channels progressively deepen in the encoding part and reduce in the decoding part, our approach involves repeatedly splitting and merging the channels. Following each merge, we use channel shuffle to rearrange the feature channels, disrupting the original channel boundaries. This method enables more effective mixing and sharing of features across different convolutional layers, facilitating multi-level information fusion and preventing channel-wise information homogenization. As a result, the diversity of feature maps is significantly enhanced, boosting the network’s representational capacity. For semantic information extraction, we also employ multi-scale convolutions to capture various image details.***

### 3. Module 1: FB-SR

![MIR-CSNet_Module 1_FB-SR](https://github.com/YF-W/MIR-CSNet/blob/c516553fe2dd95ac4849535ec555117f4cdd5ee6/MIR-CSNet_Module%201_FB-SR.png)

**We propose FB-SR, a model that processes foreground and background information through two parallel channels, which are then merged for final processing. This model integrates into the decoding part of IACI Net to enhance semantic recovery. FB-SR improves performance by two key methods: adjusting channel ratios to focus on primary and secondary features, and employing advanced techniques to separately enhance foreground and background attention. These strategies allow the model to extract and represent features more effectively, improving its generalization capability and adaptation to varying data conditions. **

| Layer Name | Module structure                         |
| :--------- | ---------------------------------------- |
| Layer 1    | F(x1) =  DepthwiseSeparableConv[3,3]     |
| Layer 2    | F(x2), D1(x3) = Split[1/8C,7/8C](D1(x1)) |
| Layer 3    | F(x4) = ones_like(D1(x2))                |
| Layer 4    | F(x5) = D1(x4)-D1(x2)                    |
| Layer 5    | F(x6) = DoubleConv[3,3](D1(x5))          |
| Layer 6    | F(x7) = DoubleConv[3,3](D1(x3))          |
| Layer 7    | F(x8) = F(x6) + F(x7)                    |

### 4. Module 2 : SSOC

![MIR-CSNet_Module 2 _SSOC](https://github.com/YF-W/MIR-CSNet/blob/78c8816865e18117a5bfb55a35cf6003cf7d8b20/MIR-CSNet_Module%202%20_SSOC.png)

***We design a feature fusion module, SSOC, to achieve cross-layer feature fusion by aggregating local and central key semantics. (1) Grid Segmentation and Convolution Branch: This branch segments the input feature map into a 4×4 grid, processes each sub-region through dual convolution operations, and applies global max pooling and average pooling to generate detailed feature maps. (2) Sub-region Padding and Overlay Branch: This branch pads and overlays sub-region feature maps to restore global features and aggregate critical information while ensuring comprehensive coverage. (3) Dilated Convolution Enhancement Branch: This branch applies dilated convolution with a dilation rate of 2 to further extract key information, enhancing edge and texture details from multiple sub-regions. The SSOC module effectively combines these branches to enhance local and global feature expression, strengthen cross-level feature fusion, and improve image segmentation accuracy.***



| Layer Name  | Module structure                          |                                                          |                                                              |                                                |
| ----------- | :---------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| Layer 1     | x1_1 = x[:, :, :h/4*3, :w/4*3]            | x1_2 = x[:, :, :h/4*3, -w/4*3:]                          | x2_1 = x[:, :, -h/4*3:, :w/4*3]                              | x2_2 = x[:, :, -h/4*3:, -w/4*3:]               |
| Output size | [4, c, h/4*3, w/4*3]                      | [4, c, h/4*3, w/4*3]                                     | [4, c, h/4*3, w/4*3]                                         | [4, c, h/4*3, w/4*3]                           |
| Layer 2     | M1_1(x1_1) = GMP(Conv[3,3])*x1_1          | M1_2(x1_2) = GMP(Conv[3,3])*x1_2                         | M(x2_1) = GMP(Conv[3,3])*x2_1                                | M(x2_2) = GMP(Conv[3,3])*x2_2                  |
| Layer 2     | A1_1(x1_1) = GAP(Conv[3,3])*x1_1          | A1_2(x1_2) = GAP(Conv[3,3])*x1_2                         | A(x2_1) = GAP(Conv[3,3])*x2_1                                | A(x2_2) = GAP(Conv[3,3])*x2_2                  |
| Output size | [4, c, h/4*3, w/4*3]                      | [4, c, h/4*3, w/4*3]                                     | [4, c, h/4*3, w/4*3]                                         | [4, c, h/4*3, w/4*3]                           |
| Layer 3     | x1_1M, x1_2M, x2_2M, x2_1M = pad(w_1//4)  | x_up = x1_1M + x1_2M;x_down = x2_1M + x2_2M              | x2_1A, x1_1A, x1_2A, x1_2A = pad(h_1//4)                     | x_left = x1_1A + x2_1A;x_right = x1_2A + x2_2A |
| Output size |                                           | x_up, x_down = [4, c, h/4*3, w]                          |                                                              | x_left, x_right = [4, c, h, w/4*3]             |
| Layer 4     | x_up, x_down, x_left, x_right = Conv[3,3] | x_up, x_down = pad(h_1//4);x_left, x_right = pad(w_1//4) | x1(x_up + x_down) = Conv[3,3, r=2];x2(x_left + x_right) = Conv[3,3, r=2] | x = Concat(x1, x2);x(x)=Conv[3,3]              |
| Output size | [4, c, h, w]                              |                                                          |                                                              |                                                |

### Datasets:

1. LUNG Dataset:https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2. Digital Retinal Images for Vessel Extraction Dataset, abbreviated as "DRIVE":https://drive.grand-challenge.org/
3. Skin Lesion Dataset, abbreviated as "SKIN LESION":https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset
4. MICCAI – Tooth Dataset, abbreviated as "TOOTH":https://tianchi.aliyun.com/dataset/156596
