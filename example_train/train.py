import argparse, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import torch
import math, random
import torch.backends.cudnn as cudnn
import kornia.augmentation as K
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from example_train.utils import MIMICTextCLIPDataset
from example_train.model import MIMIC_SwinTransformer
from torch.cuda.amp import autocast, GradScaler
import logging

logging.disable(logging.WARNING)

device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="CT clip training")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=40, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=5)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=6, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.9, help='Learning Rate decay')

train_transforms = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomGaussianNoise(mean=0.0, std=0.05),
    K.RandomBrightness(brightness=0.2),
)


def train():
    opt = parser.parse_args()
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    # ====== 三个模型 ======
    model_0 = MIMIC_SwinTransformer(in_channels=1)
    model_1 = MIMIC_SwinTransformer(in_channels=1)
    model_2 = MIMIC_SwinTransformer(in_channels=1)

    # ckpt_root = "/data_backup/Project/Text_transfer/MIMIC"
    # model_0.load_state_dict(torch.load(f"{ckpt_root}/m0_epoch5.pth"))
    # model_1.load_state_dict(torch.load(f"{ckpt_root}/m1_epoch5.pth"))
    # model_2.load_state_dict(torch.load(f"{ckpt_root}/m2_epoch5.pth"))

    models = [model_0, model_1, model_2]

    for m in models:
        m.to(device)
        # 冻结 text_encoder
        for p in m.text_encoder.parameters():
            p.requires_grad = False

    # ====== 每个模型一个 optimizer ======
    optimizers = [
        torch.optim.AdamW(models[0].parameters(), lr=opt.lr),
        torch.optim.AdamW(models[1].parameters(), lr=opt.lr),
        torch.optim.AdamW(models[2].parameters(), lr=opt.lr),
    ]

    # ====== 每个 optimizer 一个 scheduler ======
    schedulers = [
        torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=opt.step, gamma=opt.gamma),
        torch.optim.lr_scheduler.StepLR(optimizers[1], step_size=opt.step, gamma=opt.gamma),
        torch.optim.lr_scheduler.StepLR(optimizers[2], step_size=opt.step, gamma=opt.gamma),
    ]

    scalers = [GradScaler(), GradScaler(), GradScaler()]

    # ====== 数据集 & DataLoader ======
    data_set = MIMICTextCLIPDataset(
        device=device,
        ratio=0.02
    )

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    log_file = "/data_backup/Project/Text_transfer/MIMIC/loss_log.csv"
    # 如果不存在就创建并写入表头
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("epoch,iteration,loss_1,loss_2,loss_3\n")

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch:", epoch)

        trainor(
            data_loader=data_loader,
            models=models,
            optimizers=optimizers,
            scalers=scalers,
            epoch=epoch,
            device=device,
            save_path="/data_backup/Project/Text_transfer/MIMIC",
            print_every=100,
        )

        # 每个 optimizer 都 step 一下
        for scheduler in schedulers:
            scheduler.step()


def trainor(
        data_loader,
        models,
        optimizers,
        scalers,
        epoch,
        device,
        save_path=None,
        print_every=100,
):
    print(f"Epoch={epoch}, lr={optimizers[0].param_groups[0]['lr']}")

    for m in models:
        m.train()

    epoch_loss = [0.0, 0.0, 0.0]
    num_batches = 0

    for iteration, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = batch["image"].to(device, non_blocking=True)
        images = train_transforms(images)

        text = (
            batch["input_ids_1"].to(device, non_blocking=True),
            batch["input_ids_2"].to(device, non_blocking=True),
            batch["input_ids_3"].to(device, non_blocking=True)
        )

        for i in range(3):
            optimizers[i].zero_grad()
            with autocast():
                outputs = models[i](images, text[i])
                loss = outputs["loss_value"]

            scalers[i].scale(loss).backward()
            scalers[i].step(optimizers[i])
            scalers[i].update()

            epoch_loss[i] += loss.item()

        num_batches += 1

        if (iteration + 1) % print_every == 0:
            loss_1 = epoch_loss[0] / num_batches
            loss_2 = epoch_loss[1] / num_batches
            loss_3 = epoch_loss[2] / num_batches

            print(
                f"===> Epoch[{epoch}] Iter[{iteration + 1}/{len(data_loader)}]: "
                f"loss_1={loss_1:.5f}; "
                f"loss_2={loss_2:.5f}; "
                f"loss_3={loss_3:.5f}; "
            )

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                for i in range(3):
                    torch.save(
                        models[i].state_dict(),
                        f"{save_path}/m01_{i}_epoch{epoch}.pth"
                    )
                print(f"Saved models at epoch {epoch}.")


if __name__ == '__main__':
    from torch.multiprocessing import set_start_method

    set_start_method("spawn", force=True)
    train()
