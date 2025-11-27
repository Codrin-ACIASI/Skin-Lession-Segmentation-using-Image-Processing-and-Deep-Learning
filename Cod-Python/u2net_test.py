import os
import glob
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import f1_score, jaccard_score

from preprocesare import (
    ResizeFixed,
    RandomFlips,
    RandomBrightnessContrast,
    Normalize01,
    ToTensorDict,
    HairRemoval,
    BorderRemoval,
    ComposeDict,
    SalObjDataset
)
from u2net import U2NET, U2NETP


# ------------------ 1. Normalize prediction ------------------
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)
    return dn


# ------------------ 2. Postprocessing mask ------------------
def postprocess_mask(mask, threshold=0.5, min_size=100):
    mask_bin = (mask > threshold).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask_bin)
    mask_clean = np.zeros_like(mask_bin)
    for i in range(1, num_labels):
        component = (labels_im == i).astype(np.uint8)
        if component.sum() >= min_size:
            mask_clean += component
    kernel = np.ones((5, 5), np.uint8)
    mask_morph = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_final = cv2.GaussianBlur(mask_morph.astype(np.float32), (5, 5), 0)
    mask_final = (mask_final > 0.5).astype(np.uint8)
    return mask_final


# ------------------ 3. Save mask ------------------
def save_mask(image_name, mask, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mask_img = (mask * 255).astype(np.uint8)
    im = Image.fromarray(mask_img)
    base_name = os.path.basename(image_name)
    save_path = os.path.join(save_dir, os.path.splitext(base_name)[0] + '.png')
    im.save(save_path)


# ------------------ 4. Calculate metrics ------------------
def calculate_metrics(pred_mask, gt_mask):
    # Resize predicted mask to match GT
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    dice = f1_score(gt_flat, pred_flat)
    jaccard = jaccard_score(gt_flat, pred_flat)
    return dice, jaccard


# ------------------ 5. Main testing function ------------------
def main():
    model_name = 'u2net'  # sau 'u2netp'

    image_dir = os.path.join(os.getcwd(), 'Dataset_test', 'ISBI2016_ISIC_Part1_Test_Data')
    gt_dir = os.path.join(os.getcwd(), 'Dataset_test', 'ISBI2016_ISIC_Part1_Test_GroundTruth')
    prediction_dir = os.path.join(os.getcwd(), 'Dataset_test', model_name + '_results')
    model_path = r"C:\Users\Codrin\Desktop\PI-Proiect\CodPI\Cod-Python\saved_models\u2net\u2net_bce_itr_1469_train_0.951_tar_0.127_interrupted.pth"

    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    print(f"{len(img_name_list)} imagini gasite pentru test.")

    # Dataset + DataLoader cu transformari noi
    test_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=ComposeDict([
            BorderRemoval(threshold=10),
            HairRemoval(kernel_size=17, threshold=10),
            ResizeFixed(256),
            RandomFlips(p_h=0.0, p_v=0.0),
            RandomBrightnessContrast(brightness=0.0, contrast=0.0, p=0.0),
            Normalize01(),
            ToTensorDict()
        ])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)
    try:
        net.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        print(f"GPU loading failed ({e}), incercare CPU...")
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.to(device)
    net.eval()

    all_dice = []
    all_jaccard = []

    for i_test, data_test in enumerate(test_dataloader):
        inputs_test = data_test['image'].float().to(device)
        batch_names = data_test['imidx']  # lista de indici ai imaginilor din batch

        try:
            with torch.no_grad():
                d1, *_ = net(inputs_test)
                pred = torch.sigmoid(d1[:, 0, :, :])
                pred = normPRED(pred)
                pred_np_batch = pred.cpu().numpy()
        except RuntimeError as e:
            print(f"CUDA runtime error: {e}, mutare pe CPU...")
            device_cpu = torch.device('cpu')
            net.to(device_cpu)
            inputs_test = inputs_test.to(device_cpu)
            with torch.no_grad():
                d1, *_ = net(inputs_test)
                pred = torch.sigmoid(d1[:, 0, :, :])
                pred = normPRED(pred)
                pred_np_batch = pred.cpu().numpy()

        for j, idx in enumerate(batch_names):
            img_name = img_name_list[idx.item()]
            mask_refined = postprocess_mask(pred_np_batch[j])
            save_mask(img_name, mask_refined, prediction_dir)

            gt_name = os.path.splitext(os.path.basename(img_name))[0] + '_Segmentation.png'
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 127).astype(np.uint8)
                dice, jaccard = calculate_metrics(mask_refined, gt_mask)
                all_dice.append(dice)
                all_jaccard.append(jaccard)
                print(f"{img_name} -> Dice: {dice:.4f}, Jaccard: {jaccard:.4f}")
            else:
                print(f"{img_name} -> Ground truth missing")

        # Curatare GPU batch
        del inputs_test, d1, pred, pred_np_batch
        torch.cuda.empty_cache()

    if all_dice:
        print("=== Rezumat metrics test ===")
        print(f"Mean Dice: {np.mean(all_dice):.4f}")
        print(f"Mean Jaccard: {np.mean(all_jaccard):.4f}")


if __name__ == "__main__":
    main()
