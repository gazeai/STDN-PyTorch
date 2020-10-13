from utils.scripts import *
from utils.transforms import GenerateTranslatedKP
import torch
from utils.scripts import save_checkpoint
from tqdm import tqdm
import numpy as np
import copy

torch.autograd.set_detect_anomaly(True)


class STDNTrainer:

    def __init__(self, config, loss, arch, optimizer, scheduler):
        self.config = config
        self.model = arch['model']
        self.device = config.device
        #         print(self.device)
        self.model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.start_epoch = arch['start_epoch']
        self.losses = arch['losses']
        self.scheduler = scheduler
        self.do_crossval = config.train_config.do_crossval
        self.save_dir = config.train_config.checkpoint.save_path
        self.preferred_metric = config.train_config.checkpoint.preferred_metric
        if self.do_crossval:
            self.phase = ['train', 'val']
        else:
            self.phase = ['train']

        #         print(self.model.device)
        #         self.model.cuda()
        #         self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 3])
        self.model.cuda(self.device)
        self.config = config
        anchor_pts = [[0, 0], [0, 256], [256, 0], [256, 256],
                           [0, 128], [128, 0], [256, 128], [128, 256],
                           [0, 64], [0, 192], [256, 64], [256, 192],
                           [64, 0], [192, 0], [64, 256], [192, 256]]
        self.generate_offset_map = GenerateTranslatedKP(anchor_pts)

    def custom_backward(self, loss):
        sum_loss, gan_loss, disc_loss = loss
        gan_loss.backward(retain_graph=True)
        disc_loss.backward()

    def rearrange_images(self, data):

        rgb = data['rgb_path']
        rgb = torch.stack(rgb)
        gt_binary = data['label'].to(self.device)
        # gt_binary = torch.unsqueeze(gt_binary[:, 0], dim=1)
        # print(f"Shape of rgb: {rgb.shape}")
        rgb = torch.squeeze(rgb, dim=0).to(self.device)
        # print(len(rgb))
        # print(f"Shape of rgb: {rgb.shape}")
        sp_indices = torch.where(gt_binary == 1)[0]
        re_indices = torch.where(gt_binary == 0)[0]
        # print(sp_indices, len(re_indices))
        spoof_images = rgb[sp_indices]
        real_images = rgb[re_indices]
        keypoints = torch.stack(data['keypoints']).squeeze(0)
        # print(f"Shape of spoof: {spoof_images.shape}, shape of real: {real_images.shape}, lenth of keypints: {len(keypoints)}")
        # print(f"shape of kp: {keypoints.shape}")
        # print(sp_indices, re_indices)
        spoof_kp = keypoints[sp_indices]
        real_kp = keypoints[re_indices]
        # print(len(spoof_kp), len(real_kp), spoof_kp[0].shape)
        # print(spoof_images[0])
        # print(real_images[0])
        # print(spoof_kp[0])
        reg_map_spoof = []
        for i in range(len(spoof_kp)):
            offset = self.generate_offset_map(spoof_kp[i].squeeze(0).numpy(), real_kp[i].squeeze(0).numpy())
            # print(type(offset))
            offset = torch.from_numpy(offset)
            # print(offset.shape)
            reg_map_spoof.append(offset.permute([2, 0, 1]))  # b/2, 3, 256, 256
        # print(reg_map_spoof[0].shape)
        reg_map_spoof = torch.stack(reg_map_spoof).to(self.device)
        # reg_map_spoof = torch.from_numpy(reg_map_spoof).to(self.device).permute([0, 3, 1, 2])
        new_labels = torch.cat([gt_binary[re_indices], gt_binary[sp_indices]])
        new_images = torch.cat([real_images, spoof_images], dim=0)

        return new_labels, new_images, reg_map_spoof

    def infer(self, data, **kwargs):
        # images, gt_binary = data
        if len(kwargs.keys()):
            pass
        gt_binary, rgb, reg_map_spoof = self.rearrange_images(data)
        if len(rgb.shape) > 4:
            rgb = rgb.permute([1, 0, 2, 3, 4])
        esr_feat, spoof_trace, dis_features, esr_feat_syn, spoof_trace_syn, recon_syn, trace_warp, trace_unwarp, synth = self.model(rgb, reg_map_spoof)

        return gt_binary, esr_feat, spoof_trace, dis_features, esr_feat_syn, spoof_trace_syn, recon_syn, trace_warp, trace_unwarp, synth

    def track_pred(self, out):
        gt_binary, esr_feat, _, _, _, _, _, _, trace_unwarp, _ = out
        # print(esr_feat.shape, trace_unwarp.shape)
        # print(torch.mean(trace_unwarp, dim=[1, 2, 3]).detach().cpu(), torch.mean(esr_feat, dim=[1, 2, 3]).detach().cpu(), gt_binary.cpu().flatten().tolist())
        score = torch.mean(esr_feat, dim=[1, 2, 3]) + 0.5 * torch.mean(trace_unwarp, dim=[1, 2, 3])
        pred = (score >= 0.5).int().cpu().numpy().tolist()
        gt = gt_binary.cpu().flatten().tolist()
        preds = {'binary': pred}

        return preds, gt

    def train(self, dataloader, n_epochs=50, **kwargs):
        if self.start_epoch:
            print(f'STARTING TRAINING AT: {self.start_epoch} - LAST LOSS VALUE IS {self.losses[-1]}')
        else:
            print("STARTING TRAINING FROM SCRATCH")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f'LAYER TO BE ADAPTED FROM GRAD CHECK : {name}')

        best_weight = copy.deepcopy(self.model.state_dict())
        best_acer = float("inf")

        for epoch in tqdm(range(self.start_epoch, n_epochs)):
            if self.config.train_config.backbone.freeze:
                # print(self.model.no_grad_gen)
                # print(eval(f"self.model.{self.config.train_config.backbone.name}"))
                backbone = getattr(self.model, f"{self.config.train_config.backbone.name}")
                if self.config.train_config.backbone.freeze_epoch == epoch:
                    for params in backbone.parameters():
                        params.requires_grad = False
                unfreeze_epoch = self.config.train_config.backbone.unfreeze_epoch
                if unfreeze_epoch is not None:
                    if epoch == unfreeze_epoch:
                        for params in backbone:
                            params.requires_grad = True
            train_loss_hist = []
            val_loss_hist = []
            train_labels = {'gt': [], 'pred': {}}
            val_labels = {'gt': [], 'pred': {}}

            for phase in self.phase:
                print("phase is ", phase)
                if phase == 'train':
                    self.model.train()
                    # print(dataloader)
                    for i, data in enumerate(dataloader[phase]):
                        self.optimizer.zero_grad()
                        out = self.infer(data, **kwargs)
                        loss = self.loss(*out)
                        if isinstance(loss, tuple) or isinstance(loss, list):
                            single_loss = loss[0]
                        elif isinstance(loss, int):
                            single_loss = torch.tensor(loss)
                        else:
                            single_loss = loss
                        self.custom_backward(loss)
                        self.optimizer.step()

                        if i % 4 == 0:
                            # print(type(loss))

                            print(
                                f"[{epoch + 1}/{n_epochs}][{i}/{len(dataloader[phase])}] => LOSS: {single_loss.item()}"
                                f" PHASE: {phase}")
                        self.losses.append(single_loss.item())
                        train_loss_hist.append(single_loss.item())
                        preds, gts = self.track_pred(out)

                        train_labels['gt'] += gts
                        for k, v in preds.items():
                            if k not in train_labels['pred'].keys():
                                train_labels['pred'][k] = v
                            else:
                                train_labels['pred'][k] += v
                        if i % 100 == 0:
                            metrics = self.evaluate(train_labels)
                            for key in metrics.keys():
                                print(metrics[key])
                else:
                    self.model.eval()
                    for i, data in enumerate(dataloader[phase]):
                        out = self.infer(data)
                        loss = self.loss(*out)
                        if isinstance(loss, tuple) or isinstance(loss, list):
                            single_loss = loss[0]
                        elif isinstance(loss, int):
                            single_loss = torch.tensor(loss)
                        else:
                            single_loss = loss
                        if i % 20 == 0:
                            print(
                                f"[{epoch + 1}/{n_epochs}][{i}/{len(dataloader[phase])}] => LOSS: {single_loss.item()} "
                                f"PHASE: {phase}")
                        val_loss_hist.append(single_loss.item())
                        preds, gts = self.track_pred(out)

                        val_labels['gt'] += gts
                        for k, v in preds.items():
                            if k not in val_labels['pred'].keys():
                                val_labels['pred'][k] = v
                            else:
                                val_labels['pred'][k] += v
                        if i % 100 == 0:
                            metrics = self.evaluate(val_labels)
                            for key in metrics.keys():
                                print(metrics[key])
                if self.scheduler is not None:
                    self.scheduler.step()

            train_metrics = self.evaluate(train_labels)
            val_metrics = self.evaluate(val_labels)
            acc_train = "train ".upper()
            acc_val = "val ".upper()
            for k in train_metrics.keys():
                acc_train += f"for {k} - acc: {train_metrics[k]['acc'] * 100:.2f} acer: {train_metrics[k]['acer'] * 100:.2f} ".upper()
                acc_val += f"for {k} - acc: {val_metrics[k]['acc'] * 100:.2f} acer: {val_metrics[k]['acer'] * 100:.2f} ".upper()
            print(f"EPOCH: {epoch + 1} TRAIN LOSS: {np.mean(train_loss_hist)} "
                  f"{acc_train}")
            print(f"EPOCH: {epoch + 1} VAL LOSS: {np.mean(val_loss_hist)} "
                  f"{acc_val}")

            info_dict = {}
            for k, v in val_metrics.items():
                info_dict[k] = {'epoch': epoch + 1,
                                'model': self.model,
                                'loss': self.losses,
                                'apcer': v['apcer'],
                                'bpcer': v['bpcer'],
                                'acc': v['acc']
                                }

            save_checkpoint(self.save_dir, info_dict[self.preferred_metric])
            self.model.to(self.device)
            if val_metrics[self.preferred_metric]['acer'] < best_acer:
                print(
                    f"VAL ACER IMPROVED FROM: {best_acer} TO: {val_metrics[self.preferred_metric]['acer']}, COPYING OVER NEW WEIGHTS")
                best_acer = val_metrics[self.preferred_metric]['acer']
                best_weight = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_weight)
        torch.save(self.model, f'{self.save_dir}/best.pt')

    def evaluate(self, labels):

        gt = labels['gt']
        pred = labels['pred']
        metrics = {}
        # print(gt, pred)
        for k, v in pred.items():
            if len(v):
                ACC = accuracy(gt, v)
                APCER = calculate_fpr(gt, v)
                BPCER = calculate_fnr(gt, v)
                ACER = (APCER + BPCER) / 2
                metrics[k] = {'acc': ACC, 'apcer': APCER, 'bpcer': BPCER, 'acer': ACER}

        return metrics
