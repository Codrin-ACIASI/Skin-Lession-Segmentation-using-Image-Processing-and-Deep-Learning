import os
import glob
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import f1_score, jaccard_score
from skimage import io

from preprocesare import RescaleT, ToTensorLab, SalObjDataset
from u2net import U2NET, U2NETP


# ------------------ 1. Normalize prediction ------------------
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)  # evitare impartire la 0
    return dn


# ------------------ 2. Postprocessing mask ------------------
def postprocess_mask(mask, threshold=0.5, min_size=100):
    mask_bin = (mask > threshold).astype(np.uint8)

    # Eliminare zgomot prin connected components
    num_labels, labels_im = cv2.connectedComponents(mask_bin)
    mask_clean = np.zeros_like(mask_bin)
    for i in range(1, num_labels):
        component = (labels_im == i).astype(np.uint8)
        if component.sum() >= min_size:
            mask_clean += component

    # Morfologie (closing) pentru umplerea golurilor
    kernel = np.ones((5, 5), np.uint8)
    mask_morph = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Smooth edges cu Gaussian blur si reconvertire binara
    mask_final = cv2.GaussianBlur(mask_morph.astype(np.float32), (5, 5), 0)
    mask_final = (mask_final > 0.5).astype(np.uint8)

    return mask_final


# ------------------ 3. Save mask ------------------
def save_mask(image_name, mask, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    mask_img = (mask * 255).astype(np.uint8)
    im = Image.fromarray(mask_img)

    base_name = os.path.basename(image_name)
    save_path = os.path.join(save_dir, os.path.splitext(base_name)[0] + '.png')
    im.save(save_path)


# ------------------ 4. Calculate metrics ------------------
def calculate_metrics(pred_mask, gt_mask):
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
    model_path = r"C:\Users\Codrin\Desktop\PI-Proiect\Cod-Python\saved_models\u2net\u2net_bce_itr_1041_train_1.627_tar_0.210_interrupted.pth"

    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    print(f"{len(img_name_list)} imagini gasite pentru test.")

    # Dataset + DataLoader
    test_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=ToTensorLab(flag=0)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if model_name == 'u2net':
        net = U2NET(3, 1)
    else:
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.to(device)
    net.eval()

    all_dice = []
    all_jaccard = []

    for i_test, data_test in enumerate(test_dataloader):
        print("Inferencing:", os.path.basename(img_name_list[i_test]))

        inputs_test = data_test['image'].type(torch.FloatTensor).to(device)

        # Forward pass - folosim doar d1
        with torch.no_grad():
            d1, *_ = net(inputs_test)
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)
            pred_np = pred.cpu().data.numpy()[0]

        # Postprocesare
        mask_refined = postprocess_mask(pred_np)

        # Salvare masca
        save_mask(img_name_list[i_test], mask_refined, prediction_dir)


        # Construim numele ground truth folosind sufixul "_Segmentation" si extensia .png
        gt_name = os.path.splitext(os.path.basename(img_name_list[i_test]))[0] + '_Segmentation.png'
        gt_path = os.path.join(gt_dir, gt_name)

        if os.path.exists(gt_path):
            gt_mask = io.imread(gt_path)
            gt_mask = (gt_mask > 127).astype(np.uint8)
            dice, jaccard = calculate_metrics(mask_refined, gt_mask)
            all_dice.append(dice)
            all_jaccard.append(jaccard)
            print(f"Dice: {dice:.4f}, Jaccard: {jaccard:.4f}")
        else:
            print("Ground truth missing, skipping metrics.")

        # Curatare GPU
        del inputs_test, d1, pred, pred_np
        torch.cuda.empty_cache()

    # Rezumat metrics
    if all_dice:
        print("=== Rezumat metrics test ===")
        print(f"Mean Dice: {np.mean(all_dice):.4f}")
        print(f"Mean Jaccard: {np.mean(all_jaccard):.4f}")


if __name__ == "__main__":
    main()
