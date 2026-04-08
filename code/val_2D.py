import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    # 针对 2D RGB 图像的验证代码
    image = image.cpu().detach().numpy()  # [1, 3, H, W]
    label = label.squeeze(0).cpu().detach().numpy()  # [H, W]

    # 提取单张图像
    img = image[0]  # [3, H, W]
    c, x, y = img.shape

    # 缩放到网络要求的尺寸 (注意：通道维度 c=3 保持不变，缩放因子设为 1)
    if x != patch_size[0] or y != patch_size[1]:
        img = zoom(img, (1, patch_size[0] / x, patch_size[1] / y), order=0)

    # 转为 Tensor 并送入 GPU
    input = torch.from_numpy(img).unsqueeze(0).float().cuda()  # [1, 3, 512, 512]

    net.eval()
    with torch.no_grad():
        # 获取网络输出
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        # 将预测结果恢复到原始尺寸进行精度计算
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    # 用于 Deep Supervision 的版本
    image = image.cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    img = image[0]
    c, x, y = img.shape

    if x != patch_size[0] or y != patch_size[1]:
        img = zoom(img, (1, patch_size[0] / x, patch_size[1] / y), order=0)

    input = torch.from_numpy(img).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        output_main, _, _, _ = net(input)
        out = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list