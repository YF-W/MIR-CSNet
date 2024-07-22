import torch

# 加载.pth.tar文件
checkpoint = torch.load('SCB_DRIVE_fold0.pth.tar')

# 提取模型状态字典
state_dict = checkpoint['state_dict']

# 保存状态字典到.pth文件
torch.save(state_dict, 'SCB_DRIVE_fold0.pth')