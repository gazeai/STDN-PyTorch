from torch import nn
import torch


class STDNLoss:
    def __init__(self, **kwargs):
        device = kwargs['device']
        self.device = 'cpu' if device is None else device
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.esr_labels = kwargs['esr_labels'].split("_")
        self.gan_labels = kwargs['gan_labels'].split("_")
        self.disc_labels = kwargs['disc_labels'].split("_")
        self.step_one = kwargs['step_one'].split("_")
        self.step_two = kwargs['step_two'].split("_")
        self.step_three = kwargs['step_three'].split("_")
        self.spoof_trace_labels = kwargs['spoof_trace_labels'].split("_")

    def _create_new_labels(self, labels, dis_features):
        # print(len(dis_features))
        dls, dss = dis_features
        # print(f'dls: {len(dls)}, dss: {len(dss)}')
        self.dl_sl = []
        self.dl_rl = []
        self.ds_ss = []
        self.ds_rs = []
        for i in range(3):
            # print(f'shape of dls: {dls[i].shape}, shape of dss: {dss[i].shape}')
            rl, _, sl, _ = torch.split(dls[i], len(dls[i]) // 4)
            self.dl_sl.append(
                sl)  # only live matters and it will have 2 batchs where first batch is images and second batch is synthesized one
            self.dl_rl.append(rl)
            _, rs, _, ss = torch.split(dss[i], len(dss[i]) // 4)
            self.ds_ss.append(ss)
            self.ds_rs.append(rs)

        self.spoof_labels, self.real_labels = torch.split(labels, len(labels) // 2)

    # esr_loss from code
    def _loss_ESR(self, esr_feats):

        esr_re, esr_sp = torch.split(esr_feats, len(esr_feats) // 2)
        # print(f"esr shape: {esr_re.shape}")
        if self.esr_labels[0] == "0":
            esr_labels_re = torch.zeros(esr_re.shape).to(self.device)
        elif self.esr_labels[0] == "1":
            esr_labels_re = torch.ones(esr_re.shape).to(self.device)
        elif self.esr_labels[0] == "-1":
            esr_labels_re = -1 * torch.ones(esr_re.shape).to(self.device)
        else:
            esr_labels_re = float(self.esr_labels[0]) * torch.ones(esr_re.shape).to(self.device)

        if self.esr_labels[1] == "0":
            esr_labels_sp = torch.zeros(esr_sp.shape).to(self.device)
        elif self.esr_labels[1] == "1":
            esr_labels_sp = torch.ones(esr_sp.shape).to(self.device)
        elif self.esr_labels[1] == "-1":
            esr_labels_sp = -1 * torch.ones(esr_sp.shape).to(self.device)
        else:
            esr_labels_sp = float(self.esr_labels[1]) * torch.ones(esr_sp.shape).to(self.device)

        #         esr_labels_sp = torch.ones(esr_sp.shape).to(self.device)
        loss = self.l1_loss(esr_re, esr_labels_re) + self.l1_loss(esr_sp, esr_labels_sp)
        return loss

    # gan_loss from code
    def _loss_G(self):
        loss = 0

        for i in range(3):
            #             print(f'gan loss: {self.dl_sl[i].shape}, {self.dl_rl[i].shape}')
            #             ones = torch.ones(self.dl_sl[i].shape).to(self.device)
            if self.gan_labels[0] == "0":
                gan_labels_re = torch.zeros(self.dl_sl[i].shape).to(self.device)
            elif self.gan_labels[0] == "1":
                gan_labels_re = torch.ones(self.dl_sl[i].shape).to(self.device)
            elif self.gan_labels[0] == "-1":
                gan_labels_re = -1 * torch.ones(self.dl_sl[i].shape).to(self.device)
            else:
                gan_labels_re = float(self.gan_labels[0]) * torch.ones(self.dl_sl[i].shape).to(self.device)

            if self.gan_labels[1] == "0":
                gan_labels_sp = torch.zeros(self.dl_sl[i].shape).to(self.device)
            elif self.gan_labels[1] == "1":
                gan_labels_sp = torch.ones(self.dl_sl[i].shape).to(self.device)
            elif self.gan_labels[1] == "-1":
                gan_labels_sp = -1 * torch.ones(self.dl_sl[i].shape).to(self.device)
            else:
                gan_labels_sp = float(self.gan_labels[1]) * torch.ones(self.dl_sl[i].shape).to(self.device)

            loss += self.l2_loss(self.dl_sl[i], gan_labels_re) + self.l2_loss(self.ds_ss[i], gan_labels_sp)
        return loss

    # reg_loss from code
    def _loss_R(self, spoof_trace_feats):
        s, b, C, T = spoof_trace_feats

        if self.spoof_trace_labels[1] == "0":
            s_re = torch.zeros(s[:len(s) // 2].shape).to(self.device)
            b_re = torch.zeros(b[:len(b) // 2].shape).to(self.device)
            C_re = torch.zeros(C[:len(C) // 2].shape).to(self.device)
            T_re = torch.zeros(T[:len(T) // 2].shape).to(self.device)
        elif self.spoof_trace_labels[1] == "1":
            s_re = torch.ones(s[:len(s) // 2].shape).to(self.device)
            b_re = torch.ones(b[:len(b) // 2].shape).to(self.device)
            C_re = torch.ones(C[:len(C) // 2].shape).to(self.device)
            T_re = torch.ones(T[:len(T) // 2].shape).to(self.device)
        elif self.spoof_trace_labels[1] == "-1":
            s_re = -1 * torch.ones(s[:len(s) // 2].shape).to(self.device)
            b_re = -1 * torch.ones(b[:len(b) // 2].shape).to(self.device)
            C_re = -1 * torch.ones(C[:len(C) // 2].shape).to(self.device)
            T_re = -1 * torch.ones(T[:len(T) // 2].shape).to(self.device)
        else:
            s_re = float(self.spoof_trace_labels[1]) * torch.ones(s[:len(s) // 2].shape).to(self.device)
            b_re = float(self.spoof_trace_labels[1]) * torch.ones(b[:len(b) // 2].shape).to(self.device)
            C_re = float(self.spoof_trace_labels[1]) * torch.ones(C[:len(C) // 2].shape).to(self.device)
            T_re = float(self.spoof_trace_labels[1]) * torch.ones(T[:len(T) // 2].shape).to(self.device)

        if self.spoof_trace_labels[1] == "0":
            s_sp = torch.zeros(s[len(s) // 2:].shape).to(self.device)
            b_sp = torch.zeros(b[len(b) // 2:].shape).to(self.device)
            C_sp = torch.zeros(C[len(C) // 2:].shape).to(self.device)
            T_sp = torch.zeros(T[len(T) // 2:].shape).to(self.device)
        elif self.spoof_trace_labels[1] == "1":
            s_sp = torch.ones(s[len(s) // 2:].shape).to(self.device)
            b_sp = torch.ones(b[len(b) // 2:].shape).to(self.device)
            C_sp = torch.ones(C[len(C) // 2:].shape).to(self.device)
            T_sp = torch.ones(T[len(T) // 2:].shape).to(self.device)
        elif self.spoof_trace_labels[1] == "-1":
            s_sp = -1 * torch.ones(s[len(s) // 2:].shape).to(self.device)
            b_sp = -1 * torch.ones(b[len(b) // 2:].shape).to(self.device)
            C_sp = -1 * torch.ones(C[len(C) // 2:].shape).to(self.device)
            T_sp = -1 * torch.ones(T[len(T) // 2:].shape).to(self.device)
        else:
            s_sp = float(self.spoof_trace_labels[1]) * torch.ones(s[len(s) // 2:].shape).to(self.device)
            b_sp = float(self.spoof_trace_labels[1]) * torch.ones(b[len(b) // 2:].shape).to(self.device)
            C_sp = float(self.spoof_trace_labels[1]) * torch.ones(C[len(C) // 2:].shape).to(self.device)
            T_sp = float(self.spoof_trace_labels[1]) * torch.ones(T[len(T) // 2:].shape).to(self.device)

        reg_loss_spoof = self.l2_loss(s[len(s) // 2:], s_sp) + self.l2_loss(b[len(b) // 2:], b_sp) + self.l2_loss(
            C[len(C) // 2:], C_sp) + self.l2_loss(T[len(T) // 2:], T_sp)
        reg_loss_real = self.l2_loss(s[:len(s) // 2], s_re) + self.l2_loss(b[:len(b) // 2], b_re) + self.l2_loss(
            C[:len(C) // 2], C_re) + self.l2_loss(T[:len(T) // 2], T_re)
        reg_loss = reg_loss_real * 10 + reg_loss_spoof * 1e-4
        return reg_loss

    # d_loss from code
    def _loss_D(self):
        loss = 0
        for i in range(3):
            # print(f'disc loss: {self.dl_sl[i].shape}, {self.dl_rl[i].shape}')
            if self.disc_labels[0] == "0":
                disc_labels_re = torch.zeros(self.dl_rl[i].shape).to(self.device)
            elif self.disc_labels[0] == "1":
                disc_labels_re = torch.ones(self.dl_rl[i].shape).to(self.device)
            elif self.disc_labels[0] == "-1":
                disc_labels_re = -1 * torch.ones(self.dl_rl[i].shape).to(self.device)
            else:
                disc_labels_re = float(self.disc_labels[0]) * torch.ones(self.dl_rl[i].shape).to(self.device)

            if self.disc_labels[1] == "0":
                disc_labels_sp = torch.zeros(self.dl_sl[i].shape).to(self.device)
            elif self.disc_labels[1] == "1":
                disc_labels_sp = torch.ones(self.dl_sl[i].shape).to(self.device)
            elif self.disc_labels[1] == "-1":
                disc_labels_sp = -1 * torch.ones(self.dl_sl[i].shape).to(self.device)
            else:
                disc_labels_sp = float(self.disc_labels[1]) * torch.ones(self.dl_sl[i].shape).to(self.device)

            ones = torch.ones(self.dl_rl[i].shape).to(self.device)
            zeros = torch.zeros(self.dl_sl[i].shape).to(self.device)
            loss += self.l2_loss(self.dl_rl[i], disc_labels_re) + self.l2_loss(self.dl_sl[i],
                                                                               disc_labels_sp) + self.l2_loss(
                self.ds_rs[i], disc_labels_re) + self.l2_loss(self.ds_ss[i], disc_labels_sp)
        return loss

    # esr_loss_a from code

    def _loss_ESR_syn(self, esr_feats_syn):
        # print(f'synth esr loss: ', esr_feats_syn.shape)
        esr_sp, esr_re = torch.split(esr_feats_syn, len(esr_feats_syn) // 2)
        if self.esr_labels[0] == "0":
            esr_labels_re = torch.zeros(esr_re.shape).to(self.device)
        elif self.esr_labels[0] == "1":
            esr_labels_re = torch.ones(esr_re.shape).to(self.device)
        elif self.esr_labels[0] == "-1":
            esr_labels_re = -1 * torch.ones(esr_re.shape).to(self.device)
        else:
            esr_labels_re = float(self.esr_labels[0]) * torch.ones(esr_re.shape).to(self.device)

        if self.esr_labels[1] == "0":
            esr_labels_sp = torch.zeros(esr_sp.shape).to(self.device)
        elif self.esr_labels[1] == "1":
            esr_labels_sp = torch.ones(esr_sp.shape).to(self.device)
        elif self.esr_labels[1] == "-1":
            esr_labels_sp = -1 * torch.ones(esr_sp.shape).to(self.device)
        else:
            esr_labels_sp = float(self.esr_labels[1]) * torch.ones(esr_sp.shape).to(self.device)

        loss = self.l1_loss(esr_re, esr_labels_re) + self.l1_loss(esr_sp, esr_labels_sp)
        return loss

    # pixel_loss from code
    def _loss_P(self, recon_syn, trace_warp_orig):
        # print(f'pixel loss: {recon_syn[:len(recon_syn) // 2].shape}, {trace_warp_orig.shape}')
        p = self.l1_loss(recon_syn[:len(recon_syn) // 2], trace_warp_orig)
        return p

    def _loss_for_one(self, esr_feats, spoof_trace_feats):
        gan_loss = self._loss_G()
        esr_loss = self._loss_ESR(esr_feats)
        reg_loss = self._loss_R(spoof_trace_feats)
        step_one_loss = esr_loss * float(self.step_one[0]) + gan_loss * float(self.step_one[1]) + reg_loss * float(
            self.step_one[2])
        return step_one_loss

    def _loss_for_two(self):
        step_two_loss = self._loss_D() * float(self.step_two[0])
        return step_two_loss

    def _loss_for_three(self, esr_feats_syn, recon_syn, trace_warp_orig):
        synth_loss = self._loss_ESR_syn(esr_feats_syn)
        pixel_loss = self._loss_P(recon_syn, trace_warp_orig)
        step_three_loss = synth_loss * float(self.step_three[0]) + pixel_loss * float(self.step_three[1])

        return step_three_loss

    # esr_feat, spoof_trace, dis_features, esr_feat_syn, spoof_trace_syn, recon_syn, trace_warp_orig
    def __call__(self, labels, esr_feats, trace_feats, dis_features, esr_feats_syn, trace_feats_syn, recon_syn,
                 trace_warp_orig, trace_unwarp_orig, synth):
        self._create_new_labels(labels, dis_features)

        step_1 = self._loss_for_one(esr_feats, trace_feats)
        step_2 = self._loss_for_two()
        step_3 = self._loss_for_three(esr_feats_syn, recon_syn, trace_warp_orig)
        gan_loss = step_3 + step_1
        disc_loss = step_2
        sum_loss = step_1 + step_2 + step_3
        return sum_loss, gan_loss, disc_loss