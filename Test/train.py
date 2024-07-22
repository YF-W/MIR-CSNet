import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from numpy import *
# from Time import Timer as Time_tool
# from Models.model_FCN import FCN
# from SwinUnet import SwinUnet
# from UNetFormer import UNetFormer
# from Models.SmaAt_UNet import SmaAt_UNet
# from model_unet_plusplus import UNet_2Plus
# from model_pspnet import PSPNet
# from module_deeplabv2 import DeepLabV2
# from module_deeplabv3 import DeepLabV3
# from model_deeplabv3 import DeepLabv3Plus
# from model_segnet import SegNet
# from FAT_Net import FAT_Net
# from model_sunet import SUNet
# from model_unet import UNET
# from MTUnet import MTUNet,configs
# from model_resunet import build_resunet
# from module_deeplabv1 import DeepLabV1
# from model_multiresunet import MultiResUnet
# from model_pspnet import PSPNet
# from refinenet import RefineNet
# from refinenet import Bottleneck
# from model_OAUNet import oaunet
# from TransMUNet import TransMUNet
# from model_FTUNet import FTUNet
# from model_Subregion_Unet import fianlModel
# from model_MU_Net import MUNet


from torch.utils.data import Subset

from Model_test.UNet_old import UNet





from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    DiceLoss
)
from torch.utils.data import DataLoader
from dataset import OurDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset
import warnings



warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Hyperparameters etc.
#默认学习1e-4
LEARNING_RATE =1e-4
DEVICE = "cuda:0"
BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224#
PIN_MEMORY = False
LOAD_MODEL = False

# TRAIN_IMG_DIR = "data/data_skin/train_images"
# TRAIN_MASK_DIR = "data/data_skin/train_masks"
# VAL_IMG_DIR = "data/data_skin/val_images"
# VAL_MASK_DIR = "data/data_skin/val_masks"

TRAIN_IMG_DIR = "data/data_lung/train_images"
TRAIN_MASK_DIR = "data/data_lung/train_masks"
VAL_IMG_DIR = "data/data_lung/val_images"
VAL_MASK_DIR = "data/data_lung/val_masks"



def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    print(f"---Epoch:{epoch}---")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():


    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    ########

    kfold = KFold(n_splits = 5, shuffle = False)

    # Start print
    print('--------------------------------')

    # 改动过LungDataset
    train_dataset = OurDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )


    val_dataset = OurDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=train_transform,
    )
    all_dataset = ConcatDataset([train_dataset, val_dataset])

    time_total=[]       #用于记录每一折的时间，共5折,在最后统一输出（2023/3/20添加）

    # K-fold Cross Validation model evaluation
    test_ids: object
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.            #V_1：已经被我注释
        # train_subsampler = torch.utils.data.SequentialSampler(train_ids)
        # test_subsampler = torch.utils.data.SequentialSampler(test_ids)

        train_subsampler = train_ids                            #V_2：可行，
        test_subsampler = test_ids

        # train_subsampler=Subset(train_ids,all_dataset)           #V_3：许
        # test_subsampler=Subset(test_ids,all_dataset)


        # Define data loaders for training and testing data in this fold

        trainloader = DataLoader(
            all_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY,
            drop_last=True,
            sampler = train_subsampler,
        )
        testloader = DataLoader(
            all_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY,
            drop_last=True,
            sampler = test_subsampler
        )


        # model = Myall(in_channels=3, out_channels=1).to(DEVICE)
        # model = FCN(num_classes=1).to(DEVICE)
        # model = SegNet(in_channels=3, out_channels=1).to(DEVICE)
        #model = DeepLabv3Plus(num_classes=1).to(DEVICE)
        #model = PSPNet(num_classes=1).to(DEVICE)
        # model = build_resunet().to(DEVICE)
        #model = UNet_2Plus().to(DEVICE)
        #model = DcUnet(input_channels=3).to(DEVICE)
        model = SmaAt_UNet(3, 1).to(DEVICE)
        #model = UNetFormer(num_classes=1).to(DEVICE)
        #model = Myattention(in_channels=3, out_channels=1).to(DEVICE)
        # model = FAT_Net(n_channels=3).to(DEVICE)#224*224
        # model = FTUNet(n_classes = 1).to(DEVICE)
        #model = DFANet(ch_cfg, 3, 1).to(DEVICE)
        #model = DeepLabV2(n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24] ).to(DEVICE)
        # model = DeepLabV2(n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]).to(DEVICE)
        #model = DANET().to(DEVICE)
        # model = DeepLabV3(n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4],output_stride=8).to(DEVICE)
        #model = RefineNet(Bottleneck, [3, 4, 23, 3], num_classes=1).to(DEVICE)
        # model = DeepLabV1(n_classes=1, n_blocks=[3, 4, 23, 3]).to(DEVICE)
        #model = MultiResUnet(channels=3).to(DEVICE)
        #model = CBAMNet(in_channels=3, out_channels=1).to(DEVICE)
        #model = Nest_oneunet().to(DEVICE)
        #model = Refinenet(1).to(DEVICE)
        #model = SUNet(in_chans=3, out_chans=1,).to(DEVICE)
        #model=TransUnet_baseline(n_channels=3).to(DEVICE)
        #model = oaunet(in_channels=3, out_channels=1).to(DEVICE)
        # model = MTUNet(out_ch=1).to(DEVICE)
        #model = SwinUnet(num_classes=1)
        #model = TransMUNet(n_classes=1, patch_size=16, emb_size=512, img_size=256, n_channels=3, depth=4,n_regions=(256 // 16) ** 2)
        # model = fianlModel(inchannel=3, nclass=1).to(DEVICE)
        # model = MUNet(in_channels=3, out_channels=1).to(DEVICE)
        #model = SuperConvUNet(in_channels=3, out_channels=1).to(DEVICE)



        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



        # if LOAD_MODEL:
        #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        check_accuracy(testloader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()
        epoch_list = []
        acc_list = []
        dice_list = []
        jaccard_list = []
        precision_list=[]
        recall_list=[]
        f_score_list=[]
        specificity_list=[]
        auc_list=[]
        tp_list=[]
        tn_list=[]
        fp_list=[]
        fn_list=[]

        # time1 = Time_tool()
        # time1.start()  # time start
        for epoch in range(NUM_EPOCHS):

            epoch_list.append(epoch)
            train_fn(trainloader, model, optimizer, loss_fn, scaler, epoch)

            # # save model
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint)

            # check accuracy
            # check_accuracy(val_loader, model, device=DEVICE)
            acc, dice, jaccard, precision, recall,f_score,specificity,auc,tp,tn,fp,fn = check_accuracy(testloader, model, device = DEVICE)
            acc_list.append(acc)
            dice_list.append(dice)
            jaccard_list.append(jaccard)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
            specificity_list.append(specificity)
            auc_list.append(auc)
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            # plt.plot(epoch_list, acc_list)

            # print some examples to a folder
            save_predictions_as_imgs(
                testloader, model, folder="saved_images_cell{}".format(str(fold)), device=DEVICE
            )
        # time1.stop()
        # time_total.append(time1.times[-1])      #将时间添加到汇总列表中


        plt.plot(epoch_list, jaccard_list,precision_list,recall_list)
        # plt.plot(epoch_list, dice_list)
        plt.show()
        print("------------Mean-----------")
        print(f"Mean Accuracy:{mean(acc_list)}")
        print(f"Mean Jaccard:{mean(jaccard_list)}")
        print(f"Mean Dice:{mean(dice_list)}")
        print(f"Mean Precision:{mean(precision_list)}")
        print(f"Mean Recall:{mean(recall_list)}")
        print(f"Mean F1-score:{mean(f_score_list)}")
        print(f"Mean Specificity:{mean(specificity_list)}")
        print(f"Mean AUC:{mean(auc_list)}")

        print("------------MAX-----------")
        print(f"Max Accuracy:{max(acc_list)}")
        print(f"Max Jaccard:{max(jaccard_list)}")
        print(f"Max Dice:{max(dice_list)}")
        print(f"Max Precision:{max(precision_list)}")
        print(f"Max Recall:{max(recall_list)}")
        print(f"Max F1-score:{max(f_score_list)}")
        print(f"Max Specificity:{max(specificity_list)}")
        print(f"Max AUC:{max(auc_list)}")

        if '/' in str(TRAIN_IMG_DIR):
            data = TRAIN_IMG_DIR.split("/", 2)[1]
        print("{}".format(str(data)))
        if '(' in str(model):
            models = str(model).split("(", 1)[0]
        print("{}".format(str(models)))
        print("LEARNING_RATE= ",LEARNING_RATE)

    # for i in range(len(time_total)):
    #     print(f'FOLD {i} total times:{time_total[i]}s')



if __name__ == "__main__":
    main()

