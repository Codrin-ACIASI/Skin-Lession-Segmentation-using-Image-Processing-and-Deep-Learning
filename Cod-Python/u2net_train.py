import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob

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

# ------------------ 1. LOSS FUNCTION ------------------
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = torch.clamp(labels_v, 0.0, 1.0)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f" %
          (loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
    return loss0, loss


# ------------------ 2. CONFIG ------------------
model_name = 'u2net'  # sau 'u2netp'
data_dir = r"C:\Users\Codrin\Desktop\PI-Proiect\CodPI\Cod-Python\Dataset_training"
tra_image_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_Data")
tra_label_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_GroundTruth")
image_ext = '.jpg'
label_ext = '.png'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
os.makedirs(model_dir, exist_ok=True)

epoch_num = 100000
batch_size_train = 12
save_frq = 2000


# ------------------ 3. MAIN FUNCTION ------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- dataset ----
    tra_img_name_list = glob.glob(os.path.join(tra_image_dir, '*' + image_ext))
    tra_lbl_name_list = []

    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        imidx = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(tra_label_dir, imidx + "_Segmentation" + label_ext)
        if os.path.exists(lbl_path):
            tra_lbl_name_list.append(lbl_path)
        else:
            print(f"Warning: Label missing for image {img_name}")

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=ComposeDict([
            BorderRemoval(threshold=10),
            HairRemoval(kernel_size=17, threshold=10),
            ResizeFixed(256),
            RandomFlips(p_h=0.5, p_v=0.2),
            RandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.5),
            Normalize01(),
            ToTensorDict()
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=1
    )

    # ---- model ----
    net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)

    checkpoint_path = r"C:\Users\Codrin\Desktop\PI-Proiect\CodPI\Cod-Python\saved_models\u2net\u2net_bce_itr_2442_train_4.283_tar_0.611_interrupted.pth"

    # fallback CPU/GPU
    try:
        net.load_state_dict(torch.load(checkpoint_path))
    except RuntimeError as e:
        print(f"GPU loading failed ({e}), incercare CPU...")
        net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    net.to(device)

    # ---- optimizer ----
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_checkpoint_path = checkpoint_path.replace('.pth', '_optimizer.pth')
    if os.path.exists(optimizer_checkpoint_path):
        try:
            optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))
        except RuntimeError as e:
            print(f"GPU optimizer loading failed ({e}), incercare CPU...")
            optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, map_location='cpu'))

    # ---- training loop ----
    ite_num = 2047
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    try:
        for epoch in range(epoch_num):
            net.train()
            for i, data in enumerate(salobj_dataloader):
                ite_num += 1
                ite_num4val += 1

                inputs, labels = data['image'].float().to(device), data['label'].float().to(device)

                # Debug label values
                print("Label min/max:", labels.min().item(), labels.max().item())
                labels = torch.clamp(labels, 0.0, 1.0)

                optimizer.zero_grad()
                d0, d1, d2, d3, d4, d5, d6 = net(inputs)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_tar_loss += loss2.item()

                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " %
                      (epoch + 1, epoch_num, (i + 1) * batch_size_train, len(tra_img_name_list),
                       ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

                if ite_num % save_frq == 0:
                    save_path = os.path.join(
                        model_dir,
                        f"{model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth"
                    )
                    torch.save(net.state_dict(), save_path)
                    torch.save(optimizer.state_dict(), save_path.replace('.pth', '_optimizer.pth'))
                    print(f"Model salvat: {save_path}")

                    running_loss = 0.0
                    running_tar_loss = 0.0
                    net.train()
                    ite_num4val = 0

    except KeyboardInterrupt:
        print("\nCtrl+C detectat! Salvare model curent...")
        save_path = os.path.join(
            model_dir,
            f"{model_name}_bce_itr_{ite_num}_train_{running_loss / max(1, ite_num4val):.3f}_tar_{running_tar_loss / max(1, ite_num4val):.3f}_interrupted.pth"
        )
        torch.save(net.state_dict(), save_path)
        print(f"Model salvat la întrerupere: {save_path}")


if __name__ == '__main__':
    main()
