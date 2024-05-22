import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator

from torchvision.utils import save_image
import cv2

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(cfg):
    backbone_name = cfg.backbone
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

        self.best_acc = 0
        self.is_best = False

        self.normal_label = cfg.normal_label

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand


        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        #change to non_crop
        transform_train = transforms.Compose([
            #transforms.Resize((resolution,resolution)),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize((resolution,resolution)),
            # transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.test_ensemble:
            transform_test = transforms.Compose([
                transforms.Resize((resolution + expand,resolution + expand)),
                transforms.FiveCrop(resolution),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize(mean, std),
            ])           
        else:
            transform_test = transforms.Compose([
                transforms.Resize((resolution,resolution)),
                # transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                # transforms.Resize(resolution * 8 // 7),
                # transforms.CenterCrop(resolution),
                # transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])
        if self.cfg.tsne:
            transform_test_check = transforms.Compose([
                transforms.Resize((resolution,resolution)),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])  
        else:
            transform_test_check = transforms.Compose([
                transforms.Resize((resolution,resolution)),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])               


        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train, head_num=cfg.head_num)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain, head_num=cfg.head_num)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test, head_num=cfg.head_num)
        
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test, head_num=None)
        test_dataset_check = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test_check, head_num=None)


        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames


        # print("train_dataset",train_dataset.num_classes)
        # print("train_dataset",len(train_dataset.labels))

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)


        self.test_loader_check = DataLoader(test_dataset_check,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
             
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        if cfg.loss_type == "BCE":
            num_classes = 1

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            prompts = self.get_tokenized_prompts(classnames)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                    #   {"params": self.head.parameters()}],
                                      {"params": self.head.parameters(), 'weight_decay': cfg.weight_decay_head}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "BCE":
            self.criterion = nn.BCELoss()
        elif cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        
    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        cfg = self.cfg
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        if cfg.loss_type == "BCE":
            num_classes = 1

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        idx = 0
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)


                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                        
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()
            self.test(cfg = cfg)

            if self.is_best == True:
                self.save_model(cfg.output_dir)
                self.is_best = False

            # if (epoch_idx+1) % 5 == 0 and epoch_idx !=0:
            #     self.test(cfg = cfg)

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        # self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, cfg, mode="test", image_check=False):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
            data_loader_check = self.test_loader_check

        num_batch = 0

        split_output_dir = cfg.output_dir.split('/')
        split_output_dir = split_output_dir[-1]

        os.makedirs('./image_check/fnr_image/{}'.format(split_output_dir), exist_ok=True)
        os.makedirs('./image_check/fpr_image/{}'.format(split_output_dir), exist_ok=True)
        os.makedirs('./image_check/top_1_image/{}'.format(split_output_dir), exist_ok=True)

        # print("===========================================attention map ================================")

        for batch, batch_test in tqdm(zip(data_loader,data_loader_check), ascii=True):
        # for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            image_test = batch_test[0]
            label_test = batch_test[1]

            image_test = image_test.to(self.device)
            label_test = label_test.to(self.device)


            # print("=================image.size : {}==================".format(image.size()))

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            # print("=================image.view.size : {}==================".format(image.size()))

            output = self.model(image)

            # print("=================output.size : {}==================".format(output.size()))

            output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            # print("=================output.view.size : {}==================".format(output.size()))

            self.evaluator.process(image_test, output, label, split_output_dir, self.is_best, image_check)

            fpr_image_idx = self.evaluator.fpr_process(image_test, output, label, self.normal_label, split_output_dir, image_check)
            fnr_image_idx = self.evaluator.fnr_process(image_test, output, label, self.normal_label, split_output_dir, image_check)

            num_batch += output.shape[0]
            # print("=================num_batch : {}==================".format(num_batch))

        fpr_results = self.evaluator.fpr_evaluate()
        fnr_results = self.evaluator.fnr_evaluate()

        results = self.evaluator.evaluate()

        if self.best_acc <= results["accuracy"]:
            self.best_acc = results["accuracy"]
            self.is_best = True

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        if self.is_best == False:
            save_path = os.path.join(directory, "checkpoint_last.pth.tar")
            print(f"Checkpoint saved to {save_path} at the ene of training")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)

    @torch.no_grad()
    def tsne(self, mode="test", tsne_mode="dots"):

        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
            data_loader_check = self.test_loader_check

        features = []
        labels = []
        image_tests = []

        fpr_image_idxs = np.array([])
        fnr_image_idxs = np.array([])
        i=0
        index_i_fnr = 0
        index_i_fpr = 0

        for batch, batch_test in tqdm(zip(data_loader, data_loader_check), ascii=True):
            image = batch[0]
            label = batch[1]

            # print("image check")
            # print(image.shape)

            # print("label check")
            # print(label.shape)

            image = image.to(self.device)
            label = label.to(self.device)

            image_test = batch_test[0]
            label_test = batch_test[1]

            # print("image check")
            # print(image.shape)

            # print("label check")
            # print(label.shape)

            image_test = image_test.to(self.device)
            label_test = label_test.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            _bsz_test, _ncrops_test, _c_test, _h_test, _w_test = image_test.size()

            image_test = image_test.view(_bsz * _ncrops_test, _c_test, _h_test, _w_test)
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            feature, output = self.model(image)

            feature = feature.view(_bsz, _ncrops, -1).mean(dim=1)
            output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            # print("================check2==============")
            # print(output.shape)
            # output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            # print("================check3==============")
            # print(output.shape)

            self.evaluator.process(image_test, output, label, None)

            fpr_image_idx = self.evaluator.fpr_process(image_test, output, label, self.normal_label, None)
            fnr_image_idx = self.evaluator.fnr_process(image_test, output, label, self.normal_label, None)

            # fpr_image_idx = fpr_image_idx.numpy()
            # fnr_image_idx = fnr_image_idx.numpy()

            if fpr_image_idx.nelement() !=0:
                if fpr_image_idx.dim() ==0 :
                    fpr_image_idx = fpr_image_idx.unsqueeze(0)

                if index_i_fpr >=1:
                    # print("\n =================fpr_image_idx 2================")
                    # print(label_test[fpr_image_idx])
                    # print(fpr_image_idx)
                    # print(fpr_image_idxs)
                    fpr_image_idxs = torch.cat((fpr_image_idxs, fpr_image_idx+64*i),0)
                else:
                    fpr_image_idxs = fpr_image_idx
                    # print("\n =================fpr_image_idx check3================")
                    # print(fpr_image_idx)
                    # print(fpr_image_idxs)
                    index_i_fpr +=1


            if fnr_image_idx.nelement() !=0:
                if fnr_image_idx.dim() ==0 :
                    fnr_image_idx = fnr_image_idx.unsqueeze(0)

                if index_i_fnr >=1:
                    print("\n =================fpr_image_idx check2================")
                    print("label_test[fpr_image_idx]")
                    print(label_test[fpr_image_idx])
                    print("fnr_image_idxs")
                    print(fnr_image_idxs)
                    print("fnr_image_idx")
                    print(fnr_image_idx)
                    print("_bsz")
                    print(_bsz)
                    print(i)
                    print(fnr_image_idx+64*i)
                    print("fnr_image_idxs")
                    fnr_image_idxs = torch.cat((fnr_image_idxs, fnr_image_idx+64*i),0)
                else:
                    fnr_image_idxs = fnr_image_idx+64*i
                    print("\n =================fnr_image_idxs check3================")
                    print("fnr_image_idx")
                    print(fnr_image_idx)
                    print("fnr_image_idxs")
                    print(fnr_image_idxs)
                    index_i_fnr +=1

            if i==0: 
                features = feature
                labels = label
                image_tests = image_test
            else:
                features = torch.cat((features,feature),0)
                labels = torch.cat((labels, label),0)
                image_tests = torch.cat((image_tests, image_test),0)

            i+=1
            
        # Define the mean and std used in normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        # Create a denormalization transform
        denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        # image_tests = torch.squeeze(image_tests)

        # Denormalize the tensor_images
        image_tests = denormalize(image_tests)

        # Convert the PyTorch tensor to NumPy array
        image_tests = image_tests.cpu().detach().numpy().transpose((0, 2, 3, 1))

        # Clip values to be in the valid range [0, 1]
        image_tests = np.clip(image_tests, 0, 1)

        # Convert to uint8 (assuming the images are in the range [0, 1])
        image_tests = (image_tests * 255).astype(np.uint8)

        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        if torch.is_tensor(fpr_image_idxs):
            fpr_image_idxs = fpr_image_idxs.cpu().detach().numpy()
            image_tests_fpr = image_tests[fpr_image_idxs]
        if torch.is_tensor(fnr_image_idxs):
            fnr_image_idxs = fnr_image_idxs.cpu().detach().numpy()
            image_tests_fnr = image_tests[fnr_image_idxs]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('t-SNE')
        coords = TSNE(n_components=2,random_state=42, perplexity=10).fit_transform(features)

        # Save the images
        from PIL import Image

        # for i, image in enumerate(image_tests_fnr):
        #     pil_image = Image.fromarray(image)
        #     pil_image.save(f'./figure/increase_normal/normal_10_ensemble_o/image_fnr_test_normal_10_ensemble_o_{i}.png')  # Change the format

        print(coords.shape)
        print(fpr_image_idxs)
        print(fnr_image_idxs)

        # fnr_image_idxs = np.array([364, 384, 395, 403, 420, 425, 426, 427, 433, 496, 506, 581, 585, 589, 597])


        label_idx = [0,1,2,3,4,5,6,7,8,9]
        # label_idx = [0,1,2,3,4,5,6,7,8,9,"normal"]

        if tsne_mode == 'imgs':
            # for image, (x, y) in zip(img_inputs.cpu(), coords):
            #     im = OffsetImage(image.reshape(28, 28), zoom=1, cmap='gray')
            #     ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            #     ax.add_artist(ab)
            # ax.update_datalim(coords)
            # ax.autoscale()
            pass
        elif tsne_mode == 'dots':
            ax.set_title('t-SNE for class token (10 normal train images)', fontsize=20)

            for i, label in zip(range(len(label_idx)), label_idx):
                idx = np.where(labels == i)
                if label != "normal":
                    ax.scatter(coords[idx, 0], coords[idx, 1], marker='.', label=label)
                    # plt.scatter(coords[idx, 0], coords[idx, 1], marker='.', label=label)
                else:
                    ax.scatter(coords[idx, 0], coords[idx, 1], marker='.', label=label, color = 'black')
                    # plt.scatter(coords[idx, 0], coords[idx, 1], marker='.', label=label, color = 'black')



            # figure for fnr 

            # coords_fnr = coords[fnr_image_idxs,:]
            

            # i = 0
            # for image, (x, y) in zip(image_tests_fnr, coords_fnr):
            #     im = OffsetImage(image.reshape(_w_test, _h_test, _c_test), zoom=0.2)
            #     # ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            #     ab = AnnotationBbox(im, (x,y),
            #         # xybox=(x + int(np.random.randint(-100, 100, size=1)), y + +int(np.random.randint(-100, 100, size=1))),
            #         xybox=(120., -80.),
            #         xycoords='data',
            #         boxcoords="offset points",
            #         pad=0.3,
            #         arrowprops=dict(
            #             arrowstyle="->",
            #             connectionstyle="angle,angleA=0,angleB=90,rad=3")
            #         )
                    
            #     ax.add_artist(ab)
            #     plt.legend()

            #     plt.savefig("./figure/increase_normal/normal_10_ensemble_o/tsne_confirm_ver2_increase_normal_10_ensemble_o_defect_{i}.png".format(i=i))
            #     ab.remove()

            #     i+=1
                
                # if i>=10:
                #     break


            # # cdict = {1: 'purple', 2: 'blue', 3: 'green', 7: 'yellow', }
            # # print(classes)
            # # scatter = ax.scatter(coords[:,0], coords[:,1])
            # # color check
            # for i in range(11):
            #     if i==0:
            #         scatter = ax.scatter(coords[i:50*(i+1), 0], coords[i:50*(i+1), 1])
            #     if i==10:
            #         scatter = ax.scatter(coords[i:50*(i+1), 0], coords[i:50*(i+1), 1])
            #     else:
            #         scatter = ax.scatter(coords[i:50*(i+1), 0], coords[i:50*(i+1), 1])
            # # plt.colorbar()

            "set axes range"
            # plt.xlim(0, 100)
            # plt.ylim(-100, 50)
            plt.legend()
            # for i in range(11):
            #     legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes{i}".format(i=i),fontsize=20)
            #     ax.add_artist(legend1) 

                # if i==0:
                #     legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes", c='yellow',fontsize=20)
                #     ax.add_artist(legend1) 
                # else:
                #     legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes", fontsize=20)
                #     ax.add_artist(legend1) 
                # class_center = np.mean(coords[classes == i], axis=0)
                # text = TextArea('{}'.format(i))
                # ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)
                # ax.add_artist(ab)

            plt.savefig("./figure/tsne_test.png")
            # plt.savefig("./figure/increase_normal/patchcore.png")


    @torch.no_grad()
    def rollout(self, attentions, discard_ratio, head_fusion):

        # attn_bias = torch.zeros(L, S, dtype=query.dtype).to("cuda")
        print("================================rollout check===========================")
        print("len(attentions),",len(attentions))
        print("=================one_attentions_map.size : {}==================".format(attentions[0].size()))

        result = torch.eye(attentions[0].size(-1)).to("cuda")
        for attention in attentions:
            # if head_fusion == "mean":
            #     attention_heads_fused = attention.mean(axis=1)
            # elif head_fusion == "max":
            #     attention_heads_fused = attention.max(axis=1)[0]
            # elif head_fusion == "min":
            #     attention_heads_fused = attention.min(axis=1)[0]
            # else:
            #     raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = attention.topk(int(attention.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            attention[0, indices] = 0

            I = torch.eye(attention.size(-1)).to("cuda")
            a = (attention + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
        
        # Look at the total attention between the class token,
        # and the image patches
        print("=================result.size : {}==================".format(result.size()))

        # mask = result[0, 0 , 1 :]
        mask = result[0,1 :]
        print("=================class_token.size : {}==================".format(mask.size()))
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        print("=================class_token.size : {}==================".format(mask.shape))
        mask = mask / np.max(mask)
        return mask    

    def show_mask_on_image(self, img, mask):
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return img, np.uint8(255 * cam)

    @torch.no_grad()
    def attention_map(self, cfg, mode="test"):
        import matplotlib.pyplot as plt

        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        # self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        split_output_dir = cfg.output_dir
        # split_output_dir = cfg.output_dir.split('/')
        # split_output_dir = split_output_dir[-1]

        # os.makedirs('./attention_map/NORMAL_{}'.format(split_output_dir), exist_ok=True)
        os.makedirs('./attention_map/ABNORMAL_{}'.format(split_output_dir), exist_ok=True)

        i=0

        for batch in tqdm(data_loader, ascii=True):
            images = batch[0]
            labels = batch[1]

            # image = batch[0][0].unsqueeze(0)
            # label = batch[1][0].unsqueeze(0)

            for image, label in zip(images, labels):

                image = image.unsqueeze(0)
                label = label.unsqueeze(0)


                image = image.to(self.device)
                label = label.to(self.device)

                _bsz, _ncrops, _c, _h, _w = image.size()
                image = image.view(_bsz * _ncrops, _c, _h, _w)

                output, attention_map = self.model(image)

                # shit
                # output = output.view(_bsz, _ncrops, -1).mean(dim=1)
                # attention_map = attention_map.view(_bsz, _ncrops, -1).mean(dim=1)


                print("===================attention_map check==================")

                mask = self.rollout(attention_map, 0.9, "mean")

                print("label check, {}".format(label))
                print("mask check, {}".format(label == cfg.normal_label))
                print("cfg.label check, {}".format(cfg.normal_label))

                # if label == cfg.normal_label:
                if label != cfg.normal_label:

                    # grayscale = transforms.Grayscale()
                    # gray_image = grayscale(image)
                    # save_image(gray_image, './attention_map/NORMAL_{}/{}_class_{}_image.png'.format(split_output_dir, label.item(), i))    
                    save_image(image, './attention_map/ABNORMAL_{}/{}_class_{}_image.png'.format(split_output_dir, label.item(), i))    
                    # save_image(mask, './mask.png')

                    image = image.squeeze().transpose(0,1)
                    image = image.transpose(1,2)

                    np_img = image.cpu().numpy()
                    
                    mask = cv2.resize(mask, (np_img.shape[0], np_img.shape[1]))
                    image, mask = self.show_mask_on_image(np_img, mask)

                    # cv2.imshow("Input Image", np_img)
                    # cv2.imwrite("./attention_map/NORMAL_{}/{}_class_{}_mask.png".format(split_output_dir, label.item(), i), mask)
                    cv2.imwrite("./attention_map/ABNORMAL_{}/{}_class_{}_mask.png".format(split_output_dir, label.item(), i), mask)
                    
                    i+=1
                    # cv2.waitKey(-1)

    @torch.no_grad()
    def weight_norm(self, cfg, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        print("========================weight norm check========================")

        print(self.head.weight.shape)

        weight_norms = torch.norm(self.head.weight, dim=1)

        print(weight_norms.shape)

        class_numbers = list(range(1, self.head.weight.shape[0] + 1))
        plt.plot(class_numbers, weight_norms.cpu().numpy())
        plt.xlabel('Class Number')
        plt.ylabel('Norm')
        plt.title('Norm of Weight Vectors for Each Class')
        plt.show()

        plt.savefig("./figure/wafer_confirm_ver7_ver3_mean_plus_6class_1step_patch_classifier_norm_10_adapter_dim_128.png")