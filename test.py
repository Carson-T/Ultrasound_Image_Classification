import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm          # version >= 0.92
import albumentations
from albumentations import pytorch as AT
import ttach as tta
from sklearn.metrics import roc_auc_score, confusion_matrix
from utils.plot import plot_matrix
import cv2
import os
import pandas as pd
import collections
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, testpath, transform, mode):
        super(TestDataset, self).__init__()
        self.testpath = testpath
        if mode == 'J':
            self.class_dict = {"1.静息-标准": 0, "2.静息-非标准": 1}  # label dictionary 要把各个测试集里的类别文件夹名字统一
        else:
            self.class_dict = {"3.Valsalva-标准": 0, "4.Valsalva-非标准": 1}  # label dictionary
        self.groups = [
                        "白银",
                        "佛山市一",
                        "广医附三", 
                        "湖南省妇幼",    
                        "岭南迈瑞"
                        ]
        self.transform = transform
        self.img_paths, self.labels = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)      # linux 可直接读取中文路径图片
        # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # windows 不可直接用imread
        if self.transform:
            # img = self.transform(img)
            img = self.transform(image=img)["image"]
        return img, label, img_path[20:]

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group in self.groups:
            group_path = os.path.join(self.testpath, group)
            for class_name in self.class_dict:
                class_path = os.path.join(group_path, class_name)
                label = self.class_dict[class_name]
                for file_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, file_name)
                    img_paths.append(img_path)
                    labels.append(label)

        return img_paths, labels

class Convnext(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Convnext, self).__init__()
        self.model = backbone
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def get_head(self):
        return self.model.head

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = collections.OrderedDict()
    for name, params in state_dict.items():
        if "module" in name:
            name = name[7:]
            new_state_dict[name] = params
        else:
            new_state_dict = state_dict
            break
    del state_dict
    model.load_state_dict(new_state_dict)


def test(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        for i, (images, targets, img_paths) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            if i == 0:
                all_outputs = output
                all_targets = targets
                all_img_paths = list(img_paths)
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
                all_img_paths.extend(list(img_paths))
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach(), all_img_paths


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transforms = albumentations.Compose([
        albumentations.Resize(160, 315),
        albumentations.Normalize([0.22570096, 0.21439962, 0.20035854],
                                 [0.22492044, 0.21637627, 0.20286143]),
        AT.ToTensorV2()
    ])

    Test_path = "../new_data/TestSet"   # 测试集路径，该路径下包含5个测试集文件夹
    test_loader = DataLoader(TestDataset(Test_path, test_transforms, mode="J"),  # J:静息  V:Valsalva
                             batch_size=64, shuffle=False, num_workers=8, drop_last=False)

    model1 = Convnext(timm.create_model("convnextv2_nano.fcmae_ft_in1k"), 2).to(device)
    models = [model1]*5

    model_paths = [                  # J 静息 model weights
        "../saved_model/J/convnext/convnextv2_n-new_data-J-fold1-v1.pth",
        "../saved_model/J/convnext/convnextv2_n-new_data-J-fold2-v1.pth",
        "../saved_model/J/convnext/convnextv2_n-new_data-J-fold3-v1.pth",
        "../saved_model/J/convnext/convnextv2_n-new_data-J-fold4-v1.pth",
        "../saved_model/J/convnext/convnextv2_n-new_data-J-fold5-v1.pth"
        ]

    # model_paths = [                # V Valsalva model weights
    #     "../saved_model/V/convnext/convnextv2_n-new_data-V-fold1-v1.pth",
    #     "../saved_model/V/convnext/convnextv2_n-new_data-V-fold2-v1.pth",
    #     "../saved_model/V/convnext/convnextv2_n-new_data-V-fold3-v1.pth",
    #     "../saved_model/V/convnext/convnextv2_n-new_data-V-fold4-v1.pth",
    #     "../saved_model/V/convnext/convnextv2_n-new_data-V-fold5-v1.pth"
    #     ]


    tta_transforms = tta.Compose(    #Test Time Augmentation
        [
            # tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[0.8, 1, 1.2]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    all_outputs = []
    all_preds = []
    all_acc = []
    all_auc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        tta_model = tta.ClassificationTTAWrapper(models[i], tta_transforms)
        outputs, targets, img_paths = test(test_loader, tta_model, device)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).sum().item() / len(targets)
        auc = roc_auc_score(targets, outputs[:, 1])
        all_outputs.append(outputs)
        all_preds.append(preds)
        all_acc.append(acc)
        all_auc.append(auc)

    print("all_acc:",all_acc)
    print("all_auc:",all_auc)

    # vote
    voted_preds = []
    for i in range(len(targets)):
        if all_preds[0][i] + all_preds[1][i] + all_preds[2][i] + all_preds[3][i] + all_preds[4][i] >= 3:
            voted_preds.append(1)
        else:
            voted_preds.append(0)

    average_outputs = sum(all_outputs)/len(all_outputs)

    cm = confusion_matrix(targets, voted_preds)
    print(cm)
    tp = cm[0][0]
    fp = cm[1][0]
    tn = cm[1][1]
    fn = cm[0][1]
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    npv = tn/(tn+fn)

    average_acc = (torch.tensor(voted_preds) == targets).sum().item() / len(targets)
    average_auc = roc_auc_score(targets, average_outputs[:, 1])
    print("recall:",recall)
    print("precision:",precision)
    print("specificity:",specificity)
    print("npv:",npv)
    print("average_acc:",average_acc)
    print("average_auc:",average_auc)

    # plot_matrix(targets, voted_preds, [0, 1],
    #                     "jingxi_confusion_matrix.jpg",
    #                     ['standards', 'non-standards'])
    
    probility, _ = torch.max(average_outputs,1)
    df = pd.DataFrame({"img_path":img_paths,"prob":probility.tolist(),"pred":voted_preds,"label":targets})
    df.to_csv("../jingxi_predict.csv",index=False,encoding="gbk")
