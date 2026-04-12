
import os
import random
import argparse
import numpy as np
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from data.pets_dataset import get_dataloaders, PetDataset, val_transform, IMG_SIZE
from models.vgg11 import VGG11
from models.layers import CustomDropout
from models.classification import ClassificationModel
from models.localization import LocalizationModel
from models.segmentation import UNetVGG11, DiceCELoss
from losses.iou_loss import IoULoss

#Reproducibility 
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_ROOT   = "./data"
NUM_CLASSES = 37

print(f"Device: {DEVICE}")

#Helpers
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def unnorm(img_t):
    return (img_t.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

def compute_iou_batch(pred, target, eps=1e-6):
    def to_xyxy(b):
        return b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    px1,py1,px2,py2 = to_xyxy(pred)
    tx1,ty1,tx2,ty2 = to_xyxy(target)
    inter = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) * \
            (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
    union = (px2-px1).clamp(0)*(py2-py1).clamp(0) + \
            (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0) - inter + eps
    return (inter/union).mean().item()

def dice_score(logits, targets, num_classes=3, eps=1e-6):
    preds = logits.argmax(1)
    oh_p  = F.one_hot(preds,   num_classes).permute(0,3,1,2).float()
    oh_t  = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
    inter = (oh_p*oh_t).sum(dim=(0,2,3))
    total = oh_p.sum(dim=(0,2,3)) + oh_t.sum(dim=(0,2,3))
    return ((2*inter+eps)/(total+eps)).mean().item()

def mask_to_rgb(mask_np):
    palette = {0:[0,0,255], 1:[0,255,0], 2:[255,0,0]}
    rgb = np.zeros((*mask_np.shape,3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask_np==cls] = color
    return rgb


#Task 1: Train VGG11 Classifier
def train_classifier(train_loader, val_loader, dropout_p=0.5,
                     epochs=30, run_name="task1_vgg11_classification"):

    model = VGG11(num_classes=NUM_CLASSES, dropout_p=dropout_p).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    wandb.init(project="da6401_assignment2", name=run_name,
               config={"epochs":epochs,"lr":1e-3,"dropout":dropout_p,"batchnorm":True})

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss, correct = 0.0, 0
        for imgs, labels, _, _ in tqdm(train_loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss   = crit(logits, labels)
            loss.backward(); opt.step()
            tr_loss += loss.item()*imgs.size(0)
            correct += (logits.argmax(1)==labels).sum().item()
        sch.step()

        model.eval()
        va_loss, all_p, all_l = 0.0, [], []
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                va_loss += crit(logits,labels).item()*imgs.size(0)
                all_p.extend(logits.argmax(1).cpu().tolist())
                all_l.extend(labels.cpu().tolist())

        n    = len(train_loader.dataset)
        nv   = len(val_loader.dataset)
        va_f1 = f1_score(all_l, all_p, average="macro", zero_division=0)

        # Log activation histogram (3rd conv = block2[0][0]) every 5 epochs
        if epoch % 5 == 0:
            acts = []
            def _h(m,i,o): acts.append(o.detach().cpu().flatten())
            h = model.block2[0][0].register_forward_hook(_h)
            imgs_s,*_ = next(iter(val_loader))
            with torch.no_grad(): model(imgs_s[:8].to(DEVICE))
            h.remove()
            wandb.log({"activation_hist/3rd_conv": wandb.Histogram(torch.cat(acts).numpy()),
                       "epoch": epoch})
            print(f"Epoch {epoch:3d} | loss {tr_loss/n:.4f} acc {correct/n:.3f} | val_f1 {va_f1:.3f}")

        wandb.log({"cls/train_loss":tr_loss/n, "cls/train_acc":correct/n,
                   "cls/val_loss":va_loss/nv,  "cls/val_f1":va_f1, "epoch":epoch})

    torch.save(model.state_dict(), "checkpoints/classifier.pth")
    wandb.finish()
    print("Saved checkpoints/classifier.pth")
    return model


#Task 2: Train Localization
def train_localizer(train_loader, val_loader, epochs=20):
    vgg = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    vgg.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=DEVICE))

    model = LocalizationModel(vgg).to(DEVICE)
    iou_fn = IoULoss(reduction="mean")
    mse_fn = nn.MSELoss()

    opt = optim.Adam([
        {"params": model.encoder.parameters(),   "lr": 1e-4},
        {"params": model.regressor.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    wandb.init(project="da6401_assignment2", name="task2_localization",
               config={"epochs":epochs,"lr_enc":1e-4,"lr_head":1e-3})

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for imgs, _, bboxes, _ in tqdm(train_loader, leave=False):
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            opt.zero_grad()
            pred  = model(imgs)
            loss  = mse_fn(pred, bboxes) + iou_fn(pred, bboxes)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        sch.step()

        model.eval()
        va_loss, va_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, _, bboxes, _ in val_loader:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                pred = model(imgs)
                va_loss += (mse_fn(pred,bboxes)+iou_fn(pred,bboxes)).item()
                va_iou  += compute_iou_batch(pred, bboxes)
        n = len(val_loader)
        wandb.log({"loc/train_loss":tr_loss/len(train_loader),
                   "loc/val_loss":va_loss/n, "loc/val_iou":va_iou/n, "epoch":epoch})
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | val_iou {va_iou/n:.3f}")

    torch.save(model.state_dict(), "checkpoints/localizer.pth")
    wandb.finish()
    print("Saved checkpoints/localizer.pth")
    return model


#Task 3: Train Segmentation
def train_segmentation(train_loader, val_loader, epochs=25):
    vgg = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    vgg.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=DEVICE))

    model    = UNetVGG11(vgg).to(DEVICE)
    loss_fn  = DiceCELoss(num_classes=3)
    opt      = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch      = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    wandb.init(project="da6401_assignment2", name="task3_segmentation",
               config={"epochs":epochs,"lr":1e-3,"loss":"DiceCE"})

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for imgs, _, _, masks in tqdm(train_loader, leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, masks)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        sch.step()

        model.eval()
        va_loss, va_dice, va_pxacc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, _, _, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits   = model(imgs)
                va_loss  += loss_fn(logits, masks).item()
                va_dice  += dice_score(logits, masks)
                va_pxacc += (logits.argmax(1)==masks).float().mean().item()
        n = len(val_loader)
        wandb.log({"seg/train_loss":tr_loss/len(train_loader),
                   "seg/val_loss":va_loss/n, "seg/val_dice":va_dice/n,
                   "seg/val_px_acc":va_pxacc/n, "epoch":epoch})
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Dice {va_dice/n:.3f} | Px-acc {va_pxacc/n:.3f}")

    torch.save(model.state_dict(), "checkpoints/unet.pth")
    wandb.finish()
    print("Saved checkpoints/unet.pth")
    return model


# Task 4: Train Multitask
def train_multitask(train_loader, val_loader, epochs=30):
    from models.multitask import MultiTaskPerceptionModel

    # Build fresh model WITHOUT gdown (training mode)
    from models.segmentation import DoubleConv
    from models.vgg11 import conv_bn_relu

    class MultiTaskTrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(conv_bn_relu(3,64),    nn.MaxPool2d(2,2))
            self.block2 = nn.Sequential(conv_bn_relu(64,128),  nn.MaxPool2d(2,2))
            self.block3 = nn.Sequential(conv_bn_relu(128,256), conv_bn_relu(256,256), nn.MaxPool2d(2,2))
            self.block4 = nn.Sequential(conv_bn_relu(256,512), conv_bn_relu(512,512), nn.MaxPool2d(2,2))
            self.block5 = nn.Sequential(conv_bn_relu(512,512), conv_bn_relu(512,512), nn.MaxPool2d(2,2))
            self.avgpool = nn.AdaptiveAvgPool2d((7,7))
            self.cls_head = nn.Sequential(
                nn.Linear(512*7*7,4096),nn.BatchNorm1d(4096),nn.ReLU(True),CustomDropout(0.5),
                nn.Linear(4096,4096),   nn.BatchNorm1d(4096),nn.ReLU(True),CustomDropout(0.5),
                nn.Linear(4096,NUM_CLASSES))
            self.loc_head = nn.Sequential(
                nn.Linear(512*7*7,1024),nn.BatchNorm1d(1024),nn.ReLU(True),CustomDropout(0.3),
                nn.Linear(1024,256),nn.ReLU(True),nn.Linear(256,4),nn.ReLU(True))
            self.up5=nn.ConvTranspose2d(512,512,2,stride=2); self.dec5=DoubleConv(1024,512)
            self.up4=nn.ConvTranspose2d(512,256,2,stride=2); self.dec4=DoubleConv(512,256)
            self.up3=nn.ConvTranspose2d(256,128,2,stride=2); self.dec3=DoubleConv(256,128)
            self.up2=nn.ConvTranspose2d(128,64,2,stride=2);  self.dec2=DoubleConv(128,64)
            self.up1=nn.ConvTranspose2d(64,32,2,stride=2);   self.dec1=DoubleConv(32,32)
            self.seg_out=nn.Conv2d(32,3,1)

        def forward(self,x):
            s1=self.block1(x);s2=self.block2(s1);s3=self.block3(s2)
            s4=self.block4(s3);s5=self.block5(s4)
            p=torch.flatten(self.avgpool(s5),1)
            c=self.cls_head(p);l=self.loc_head(p)
            d=self.dec5(torch.cat([self.up5(s5),s4],1))
            d=self.dec4(torch.cat([self.up4(d),s3],1))
            d=self.dec3(torch.cat([self.up3(d),s2],1))
            d=self.dec2(torch.cat([self.up2(d),s1],1))
            d=self.dec1(self.up1(d))
            return c,l,self.seg_out(d)

    model   = MultiTaskTrain().to(DEVICE)
    ce_fn   = nn.CrossEntropyLoss()
    iou_fn  = IoULoss()
    mse_fn  = nn.MSELoss()
    seg_fn  = DiceCELoss()
    opt     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    wandb.init(project="da6401_assignment2", name="task4_multitask",
               config={"epochs":epochs,"lr":1e-3})

    for epoch in range(1, epochs+1):
        model.train()
        tr_cls,tr_loc,tr_seg = 0.0,0.0,0.0
        for imgs,labels,bboxes,masks in tqdm(train_loader,leave=False):
            imgs,labels,bboxes,masks = imgs.to(DEVICE),labels.to(DEVICE),bboxes.to(DEVICE),masks.to(DEVICE)
            opt.zero_grad()
            c,l,s = model(imgs)
            lc=ce_fn(c,labels); ll=mse_fn(l,bboxes)+iou_fn(l,bboxes); ls=seg_fn(s,masks)
            (lc+ll+ls).backward(); opt.step()
            tr_cls+=lc.item();tr_loc+=ll.item();tr_seg+=ls.item()
        sch.step()

        model.eval()
        all_p,all_l,va_iou,va_dice=[],[],0.0,0.0
        with torch.no_grad():
            for imgs,labels,bboxes,masks in val_loader:
                imgs,labels,bboxes,masks=imgs.to(DEVICE),labels.to(DEVICE),bboxes.to(DEVICE),masks.to(DEVICE)
                c,l,s=model(imgs)
                all_p.extend(c.argmax(1).cpu().tolist())
                all_l.extend(labels.cpu().tolist())
                va_iou+=compute_iou_batch(l,bboxes)
                va_dice+=dice_score(s,masks)
        n=len(val_loader); NTR=len(train_loader)
        va_f1=f1_score(all_l,all_p,average="macro",zero_division=0)
        wandb.log({"mt/train_cls":tr_cls/NTR,"mt/train_loc":tr_loc/NTR,"mt/train_seg":tr_seg/NTR,
                   "mt/train_loss":(tr_cls+tr_loc+tr_seg)/NTR,
                   "mt/val_f1":va_f1,"mt/val_iou":va_iou/n,"mt/val_dice":va_dice/n,"epoch":epoch})
        if epoch%5==0:
            print(f"Epoch {epoch:2d} | F1 {va_f1:.3f} | IoU {va_iou/n:.3f} | Dice {va_dice/n:.3f}")

    torch.save(model.state_dict(), "checkpoints/multitask.pth")
    wandb.finish()
    print("Saved checkpoints/multitask.pth")


# W&B Report
# Section 2.1: No-BatchNorm ablation
def run_report_experiments(train_loader, val_loader, test_loader):
    crit = nn.CrossEntropyLoss()

    if False:
        crit = nn.CrossEntropyLoss()
        wandb.init(project="da6401_assignment2",name="task1_no_batchnorm",
                   config={"epochs":15,"batchnorm":False})
        m=VGG11NoBN().to(DEVICE)
        o=optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)

        for epoch in range(1,16):
            m.train()
            tl,tc=0.0,0
            for imgs,labels,_,_ in tqdm(train_loader,leave=False):
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                o.zero_grad()
                logits=m(imgs)
                loss=crit(logits,labels)
                loss.backward()
                o.step()
                tl+=loss.item()*imgs.size(0)
                tc+=(logits.argmax(1)==labels).sum().item()

            vl2, vp, vl = 0.0, [], []
            m.eval()
            with torch.no_grad():
                for imgs,labels,_,_ in val_loader:
                    imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                    lg=m(imgs)
                    vl2 += crit(lg,labels).item()*imgs.size(0)
                    vp.extend(lg.argmax(1).cpu().tolist())
                    vl.extend(labels.cpu().tolist())

        wandb.finish()

    # Section 2.2: Dropout
    for dp in [0.0, 0.2, 0.5]:
        rn = f"dropout_p{str(dp).replace('.','_')}"
        wandb.init(project="da6401_assignment2",name=rn,
                   config={"epochs":5,"dropout_p":dp})

        m=VGG11(num_classes=NUM_CLASSES,dropout_p=dp).to(DEVICE)
        o=optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)

        
        s=optim.lr_scheduler.CosineAnnealingLR(o,T_max=5)

        for epoch in range(1,6):
            m.train()
            tl,tc=0.0,0
            for imgs,labels,_,_ in tqdm(train_loader,leave=False):
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                o.zero_grad()
                lg=m(imgs)
                loss=crit(lg,labels)
                loss.backward()
                o.step()
                tl+=loss.item()*imgs.size(0)
                tc+=(lg.argmax(1)==labels).sum().item()

            s.step()

            m.eval()
            vl2,vp,vl=0.0,[],[]
            with torch.no_grad():
                for imgs,labels,_,_ in val_loader:
                    imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                    lg=m(imgs)
                    vl2+=crit(lg,labels).item()*imgs.size(0)
                    vp.extend(lg.argmax(1).cpu().tolist())
                    vl.extend(labels.cpu().tolist())

            n=len(train_loader.dataset)
            nv=len(val_loader.dataset)

            vf1=f1_score(vl,vp,average="macro",zero_division=0)

            wandb.log({
                "cls/train_loss":tl/n,
                "cls/val_loss":vl2/nv,
                "cls/generalization_gap":vl2/nv - tl/n,
                "cls/val_f1":vf1,
                "epoch":epoch
            })

        wandb.finish()

    # Section 2.3: Transfer learning showdown
    vgg_base = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    vgg_base.load_state_dict(torch.load("checkpoints/classifier.pth",map_location=DEVICE))
    seg_fn = DiceCELoss()

    for strategy,rn in [("freeze_all","tl_frozen"),("freeze_early","tl_partial"),("full","tl_full_finetune")]:
        vgg_tmp = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
        vgg_tmp.load_state_dict(torch.load("checkpoints/classifier.pth",map_location=DEVICE))
        model_tl = UNetVGG11(vgg_tmp).to(DEVICE)

        if strategy=="freeze_all":
            for blk in [model_tl.enc1,model_tl.enc2,model_tl.enc3,model_tl.enc4,model_tl.enc5]:
                for p in blk.parameters(): p.requires_grad=False
        elif strategy=="freeze_early":
            for blk in [model_tl.enc1,model_tl.enc2,model_tl.enc3]:
                for p in blk.parameters(): p.requires_grad=False

        wandb.init(project="da6401_assignment2",name=rn,
                   config={"epochs":5,"strategy":strategy})
        o=optim.Adam(filter(lambda p:p.requires_grad,model_tl.parameters()),lr=1e-3,weight_decay=1e-4)
        s=optim.lr_scheduler.CosineAnnealingLR(o,T_max=5)
        for epoch in range(1,6):
            model_tl.train(); tl=0.0
            for imgs,_,_,masks in tqdm(train_loader,leave=False):
                imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
                o.zero_grad(); lg=model_tl(imgs); loss=seg_fn(lg,masks)
                loss.backward(); o.step(); tl+=loss.item()
            s.step()
            model_tl.eval(); vl,vd,vp=0.0,0.0,0.0
            with torch.no_grad():
                for imgs,_,_,masks in val_loader:
                    imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
                    lg=model_tl(imgs); vl+=seg_fn(lg,masks).item()
                    vd+=dice_score(lg,masks)
                    vp+=(lg.argmax(1)==masks).float().mean().item()
            n=len(val_loader)
            wandb.log({"seg/train_loss":tl/len(train_loader),
                       "seg/val_loss":vl/n,"seg/val_dice":vd/n,"seg/val_px_acc":vp/n,"epoch":epoch})
        wandb.finish()

    #Sections 2.4-2.7:
    wandb.init(project="da6401_assignment2",name="wb_report_visuals")

    # 2.4 Feature maps
    vgg = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    vgg.load_state_dict(torch.load("checkpoints/classifier.pth",map_location=DEVICE))
    vgg.eval()
    sample_img,*_ = next(iter(val_loader))
    sample_img = sample_img[:1].to(DEVICE)
    acts={}
    def hk(name):
        def fn(m,i,o): acts[name]=o.detach().cpu()
        return fn
    h1=vgg.block1[0][0].register_forward_hook(hk("first"))
    h5=vgg.block5[0][0].register_forward_hook(hk("last"))
    with torch.no_grad(): vgg(sample_img)
    h1.remove(); h5.remove()
    fig,axes=plt.subplots(2,8,figsize=(20,5))
    for i in range(8):
        axes[0,i].imshow(acts["first"][0,i].numpy(),cmap="viridis"); axes[0,i].axis("off")
        axes[1,i].imshow(acts["last"][0,i].numpy(), cmap="viridis"); axes[1,i].axis("off")
    axes[0,0].set_title("First conv layer",loc="left",fontsize=10)
    axes[1,0].set_title("Last conv layer", loc="left",fontsize=10)
    plt.tight_layout(); plt.savefig("/tmp/feat_maps.png",dpi=100); plt.close()
    wandb.log({"Feature Maps":wandb.Image("/tmp/feat_maps.png")})

    # 2.5 BBox table
    loc_vgg = VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    loc_vgg.load_state_dict(torch.load("checkpoints/classifier.pth",map_location=DEVICE))
    model_loc = LocalizationModel(loc_vgg).to(DEVICE)
    model_loc.load_state_dict(torch.load("checkpoints/localizer.pth",map_location=DEVICE))
    model_loc.eval()
    bb_table=wandb.Table(columns=["image","gt_box","pred_box","confidence","iou"])
    imgs_t,_,bboxes_t,_=next(iter(test_loader))
    imgs_t,bboxes_t=imgs_t.to(DEVICE),bboxes_t.to(DEVICE)
    with torch.no_grad(): preds_t=model_loc(imgs_t)
    for i in range(min(10,imgs_t.size(0))):
        pred_box=preds_t[i].cpu(); gt_box=bboxes_t[i].cpu()
        siou=compute_iou_batch(pred_box.unsqueeze(0),gt_box.unsqueeze(0))
        fig,ax=plt.subplots(1,figsize=(4,4)); ax.imshow(unnorm(imgs_t[i]))
        for box,color,lbl in [(gt_box,"green","GT"),(pred_box,"red",f"IoU={siou:.2f}")]:
            cx,cy,w,h=box; x1,y1=cx-w/2,cy-h/2
            ax.add_patch(patches.Rectangle((x1,y1),w,h,linewidth=2,edgecolor=color,facecolor="none"))
            ax.text(x1,y1-2,lbl,color=color,fontsize=8)
        ax.axis("off"); plt.tight_layout()
        plt.savefig(f"/tmp/bbox_{i}.png",dpi=80); plt.close()
        bb_table.add_data(wandb.Image(f"/tmp/bbox_{i}.png"),str(gt_box.tolist()),
                          str(pred_box.tolist()),f"{preds_t[i].max().item():.3f}",f"{siou:.3f}")
    wandb.log({"Bounding Box Predictions":bb_table})

    # 2.6 Segmentation samples
    seg_vgg=VGG11(num_classes=NUM_CLASSES).to(DEVICE)
    seg_vgg.load_state_dict(torch.load("checkpoints/classifier.pth",map_location=DEVICE))
    model_seg=UNetVGG11(seg_vgg).to(DEVICE)
    model_seg.load_state_dict(torch.load("checkpoints/unet.pth",map_location=DEVICE))
    model_seg.eval()
    seg_imgs=[]
    imgs_s,_,_,masks_s=next(iter(val_loader))
    imgs_s,masks_s=imgs_s[:5].to(DEVICE),masks_s[:5].to(DEVICE)
    with torch.no_grad(): seg_logits=model_seg(imgs_s)
    preds_s=seg_logits.argmax(1)
    for i in range(5):
        fig,axes=plt.subplots(1,3,figsize=(9,3))
        axes[0].imshow(unnorm(imgs_s[i]));                          axes[0].set_title("Original");  axes[0].axis("off")
        axes[1].imshow(mask_to_rgb(masks_s[i].cpu().numpy()));      axes[1].set_title("GT Trimap"); axes[1].axis("off")
        axes[2].imshow(mask_to_rgb(preds_s[i].cpu().numpy()));      axes[2].set_title("Prediction");axes[2].axis("off")
        plt.tight_layout(); plt.savefig(f"/tmp/seg_{i}.png",dpi=80); plt.close()
        seg_imgs.append(wandb.Image(f"/tmp/seg_{i}.png"))
    wandb.log({"Segmentation Samples":seg_imgs})

    # 2.7 In-the-wild
    import urllib.request, torchvision.transforms as T
    wild_urls=[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/800px-YellowLabradorLooking_new.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CyprusShorthair.jpg/800px-CyprusShorthair.jpg",
    ]
    from PIL import Image as PILImage
    norm=T.Compose([T.Resize((224,224)),T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    from models.multitask import MultiTaskPerceptionModel
    # load multitask weights for inference
    wild_imgs_wb=[]
    for j,url in enumerate(wild_urls):
        try:
            fname=f"/tmp/wild_{j}.jpg"
            urllib.request.urlretrieve(url,fname)
            img_pil=PILImage.open(fname).convert("RGB")
            img_t=norm(img_pil).unsqueeze(0).to(DEVICE)
            # quick inference using individual models
            with torch.no_grad():
                cls_out=vgg(img_t)
                loc_out=model_loc(img_t)
                seg_out=model_seg(img_t)
            pred_cls=cls_out.argmax(1).item()
            pred_box=loc_out[0].cpu()
            pred_mask=seg_out.argmax(1)[0].cpu().numpy()
            img_disp=np.array(img_pil.resize((224,224)))
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            axes[0].imshow(img_disp)
            cx,cy,w,h=pred_box.tolist()
            axes[0].add_patch(patches.Rectangle((cx-w/2,cy-h/2),w,h,linewidth=2,edgecolor="red",facecolor="none"))
            axes[0].set_title(f"Class {pred_cls}"); axes[0].axis("off")
            axes[1].imshow(img_disp); axes[1].set_title("Original"); axes[1].axis("off")
            axes[2].imshow(mask_to_rgb(pred_mask)); axes[2].set_title("Seg Mask"); axes[2].axis("off")
            plt.tight_layout(); plt.savefig(f"/tmp/wild_{j}.png",dpi=80); plt.close()
            wild_imgs_wb.append(wandb.Image(f"/tmp/wild_{j}.png",caption=f"Wild {j+1} class {pred_cls}"))
        except Exception as e:
            print(f"Wild image {j} failed: {e}")
    wandb.log({"In-the-Wild Results":wild_imgs_wb})
    wandb.finish()
    print("All report visuals logged.")


# Main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["all","1","2","3","4","report"])
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers",   type=int, default=2)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, args.batch_size, args.workers)

    os.makedirs("checkpoints", exist_ok=True)

    if args.task in ("1", "all"):
        print("\n=== Task 1: VGG11 Classification ===")
        train_classifier(train_loader, val_loader)

    if args.task in ("2", "all"):
        print("\n=== Task 2: Localization ===")
        train_localizer(train_loader, val_loader)

    if args.task in ("3", "all"):
        print("\n=== Task 3: Segmentation ===")
        train_segmentation(train_loader, val_loader)

    if args.task in ("4", "all"):
        print("\n=== Task 4: Multitask ===")
        train_multitask(train_loader, val_loader)

    if args.task in ("report", "all"):
        print("\n=== W&B Report Runs ===")
        run_report_experiments(train_loader, val_loader, test_loader)

    print("\nAll done!")
