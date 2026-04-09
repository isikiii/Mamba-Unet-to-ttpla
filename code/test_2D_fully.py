import argparse
import os
import shutil
import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from scipy.ndimage import zoom
from networks.net_factory import net_factory
from medpy import metric

from networks.vision_mamba import MambaUnet as ViM_seg
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/TTPLA_Processed')
parser.add_argument('--exp', type=str, default='TTPLA_Mamba_Final')
parser.add_argument('--model', type=str, default='mambaunet')
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--labeled_num', type=int, default=140)

# 最关键的一行：让脚本认识这个参数
parser.add_argument('--save_mode_path', type=str, default=None)

# 兼容参数
parser.add_argument('--cfg', type=str, default="./code/configs/vmamba_tiny.yaml")
parser.add_argument('--patch_size', type=list, default=[512, 512])
parser.add_argument('--opts', help="Modify config options", default=None, nargs='+')
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache-mode', type=str, default='part')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int)
parser.add_argument('--use-checkpoint', action='store_true')
parser.add_argument('--amp-opt-level', type=str, default='O1')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt)
    return 0


def test_single_volume(case, net, test_save_path, FLAGS):
    img_path = os.path.join(FLAGS.root_path, "test/images", "{}.png".format(case))
    lbl_path = os.path.join(FLAGS.root_path,"test/masks", "{}.png".format(case))

    image = np.array(Image.open(img_path).convert('RGB'))
    label = np.array(Image.open(lbl_path).convert('L'))

    h, w, _ = image.shape
    patch_size = FLAGS.patch_size

    img_resized = zoom(image, (patch_size[0] / h, patch_size[1] / w, 1), order=0)
    input = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = zoom(out.cpu().detach().numpy(), (h / patch_size[0], w / patch_size[1]), order=0)

    metrics = []
    for i in range(1, FLAGS.num_classes):
        metrics.append(calculate_metric_percase(prediction == i, label == i))

    vis_pred = np.zeros((h, w, 3), dtype=np.uint8)
    vis_pred[prediction == 1] = [255, 0, 0]
    vis_pred[prediction == 2] = [0, 255, 0]
    cv2.imwrite(os.path.join(test_save_path, case + "_pred.png"), cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR))

    return metrics


def Inference(FLAGS):
    with open(os.path.join(FLAGS.root_path, 'test.list'), 'r') as f:
        image_list = [item.strip().split(".")[0] for item in f.readlines() if item.strip()]

    test_save_path = "../model/{}/{}_predictions/".format(FLAGS.exp, FLAGS.model)
    if not os.path.exists(test_save_path): os.makedirs(test_save_path)

    config = get_config(FLAGS)
    # 根据命令行参数判断：如果是 mambaunet 就用原逻辑，否则去工厂里找（比如 unet）
    if FLAGS.model == 'mambaunet':
        net = ViM_seg(config, img_size=FLAGS.patch_size[0], num_classes=FLAGS.num_classes).cuda()
    else:
        # 注意：这里传入 in_chns=3 以适配你的三通道优化 [cite: 309, 404]
        net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)

    load_path = FLAGS.save_mode_path if FLAGS.save_mode_path else os.path.join(
        "../model/{}_{}/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model), 'mambaunet_best_model.pth')
    print(">>> Loading weight from: {}".format(load_path))
    net.load_state_dict(torch.load(load_path))
    net.eval()

    results = []
    for case in tqdm(image_list):
        results.append(test_single_volume(case, net, test_save_path, FLAGS))

    avg_dice = np.mean(results, axis=0)
    return avg_dice


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    avg_dice = Inference(FLAGS)
    print("\n" + "=" * 30)
    print(f"Test Set Dice - Tower: {avg_dice[0]:.4f}, Line: {avg_dice[1]:.4f}")
    print(f"Mean Dice: {np.mean(avg_dice):.4f}")
    print("=" * 30)