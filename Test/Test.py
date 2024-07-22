import argparse
import torch


from utils import (
    get_test_loader,
    get_transform,
    # get_transform_test,         #暂时用于ISBI测试，后需要删除，无用处
    check_accuracy,
    load_checkpoint,
    save_predictions_as_imgs
)


from Model_test.CNet_module1 import CNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='dataset2', help='experiment_name, please use all capital letters')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as png!')
parser.add_argument('--cuda', type=str, default="cpu", help='cuda')
parser.add_argument('--pin_memory', type=bool,  default=False, help='DataLoader.pin_memory')
# parser.add_argument('--drop_list', type=bool,  default=True, help='DataLoader.drop_last')

args = parser.parse_args()

if __name__ =='__main__':

    DEVICE = args.cuda

    model =CNet()
    model.to(device=DEVICE)

    load_checkpoint(torch.load("Model_test/CNet_module1.pth.tar"), model)     #use checkpoint
    # model.load_state_dict(torch.load('Model_test/UNet_WELDING_fold4.pth', map_location=DEVICE))        #use .pth

    #Dataset
    TEST_IMG_DIR = "data/{}/image".format(args.dataset)
    TEST_MASK_DIR = "data/{}/mask".format(args.dataset)

    #DataLoader
    test_loader=get_test_loader(TEST_IMG_DIR,
                                TEST_MASK_DIR,
                                args.batch_size,
                                get_transform(),
                                args.num_classes,
                                args.pin_memory,
                                # args.drop_list
                                )

    check_accuracy(test_loader, model, device=DEVICE)

    save_predictions_as_imgs(
        test_loader, model, folder=args.test_save_dir, device=DEVICE)





