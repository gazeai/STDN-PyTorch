from torch import nn
import torch
from skimage.color import rgb2yuv
# import tensorflow as tf
from torch.nn import functional as F
from config import flags
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
physical_devices = tf.config.list_physical_devices('GPU')
try:
    # Disable first GPU
    gpu_flag = int(flags.device.split(':')[-1])
    tf.config.set_visible_devices(physical_devices[gpu_flag:gpu_flag + 1], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')

    assert len(logical_devices) == len(physical_devices) - 3
except:
    print("Gpu is not found!")
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def resize(x, size):
    res = nn.Upsample(size=size, mode='bilinear')
    return res(x)


def normalize(x, p=2):
    return F.normalize(x, p=2, dim=1)


def convert_from_rgb2yuv(x):
    device = x.device
    x = x.cpu()
    x = x.permute(0, 2, 3, 1).numpy()
    x = rgb2yuv(x)
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
    x = x.to(device)
    return x


class ESR(nn.Module):
    def __init__(self, in_c, out_c=1):
        super(ESR, self).__init__()
        # self.resize =
        self.conv1 = DownSample(in_c * 3, 64, mul=2, drop=True)
        # self.conv2 = EncConv(96, 64, drop=True)
        self.conv2 = EncConv(64, out_c, act=False, norm=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(len(x)):
            x[i] = resize(x[i], (32, 32))
            # print(f'esr-1: {x[i].shape}')
        x = torch.cat(x, dim=1)
        # print(f'esr concatanated: {x.shape}')
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)   # b, 1, 16, 16
        # print(x.shape)
        # x = self.conv3(x)
        return x


class EncConv(nn.Module):
    def __init__(self, in_c, out_c, act=True, norm=True, drop=False):
        super(EncConv, self).__init__()
        self.norm = norm
        self.act = act
        self.apply_drop = drop

        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
        if self.norm:
            self.bn = nn.BatchNorm2d(out_c)
        if self.act:
            self.leaky_relu = nn.PReLU()
        if self.apply_drop:
            self.dropout = nn.Dropout(1 - 0.7)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.act:
            x = self.leaky_relu(x)
        if self.apply_drop:
            x = self.dropout(x)

        return x


class DownSample(nn.Module):
    def __init__(self, in_c, out_c, mul=2, act=True, norm=True, drop=False):
        super(DownSample, self).__init__()
        self.norm = norm
        self.act = act
        self.apply_drop = drop
        self.conv = nn.Conv2d(in_c, out_c, 3, mul, 1)
        if self.norm:
            self.bn = nn.BatchNorm2d(out_c)
        if self.act:
            self.leaky_relu = nn.PReLU()
        if self.apply_drop:
            self.dropout = nn.Dropout(1 - 0.7)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.act:
            x = self.leaky_relu(x)
        if self.apply_drop:
            x = self.dropout(x)

        return x


class UpSample(nn.Module):
    def __init__(self, in_c, out_c, norm=True, act=True, drop=False):
        super(UpSample, self).__init__()
        self.norm = norm
        self.act = act
        self.apply_drop = drop
        self.deconv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        if self.norm:
            self.bn = nn.BatchNorm2d(out_c)
        if self.act:
            self.leaky_realu = nn.PReLU()
        if self.apply_drop:
            self.dropout = nn.Dropout(1 - 0.7)

    def forward(self, x):
        x = self.deconv(x)
        if self.norm:
            x = self.bn(x)
        if self.act:
            x = self.leaky_realu(x)
        if self.apply_drop:
            x = self.dropout(x)

        return x


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        ## encoders
        self.enc1 = nn.Sequential(EncConv(3, 64),
                                  # EncConv(6, 64),
                                  EncConv(64, 96),
                                  EncConv(96, 128),
                                  EncConv(128, 96))
        self.down_1 = DownSample(96, 96)
        self.enc2 = nn.Sequential(EncConv(96, 128),
                                  EncConv(128, 96))
        self.down_2 = DownSample(96, 96)
        self.enc3 = nn.Sequential(EncConv(96, 128),
                                  EncConv(128, 96))
        self.down_3 = DownSample(96, 96)

        ## decoders

        self.dec1 = UpSample(96, 64)
        self.dec2 = UpSample(96 + 64, 64)
        self.dec3 = UpSample(96 + 64, 64)

    def _encoder(self, x):
        esr_feat = []
        dec_feat = []
        '''
        input: x: (b, 3, 256, 256)
        process:
        1:x=256: (b, 64, x, x) -> (b, 96, x, x) -> (b, 128, x, x) -> (b, 96, x, x) -> (b, 96, x/2, x/2)
        2:x=128: (b, 96, x, x) -> (b, 128, x, x) ->  (b, 96, x/2, x/2)
        2:x=64: (b, 96, x, x) -> (b, 128, x, x) ->  (b, 96, x/2, x/2)
        '''
        for i in range(1, 4):
            x = getattr(self, f'enc{i}')(x)
            # print(f'Enc-{i}: {x.shape}')
            x = getattr(self, f"down_1")(x)
            # print(f'down-{i}: {x.shape}')
            dec_feat.append(x)
            esr_feat.append(x)

        return x, esr_feat, dec_feat[:-1]

    def _decoder(self, x, dec_feats):
        '''

        :param x: shape: (b, 96, 32, 32)
        :param dec_feats: len=2, 1: shape(b, 96, 64, 64), 0: shape(b, 96, 128, 128)
        :return:
        '''
        # print(f'Decoder-1 input: {x.shape}')
        x = self.dec1(x)
        # shape: (b, 64, 64, 64)
        # print(f'decoder-1 out: {x.shape}')
        sb = x
        x = torch.cat([x, dec_feats[1]], 1)
        # print(f'decoder-2 input: {x.shape}')
        x = self.dec2(x)  # input: (b, 96+64, 64, 64)
        # print(f'decoder-2 output: {x.shape}')
        # shape: (b, 64, 128, 128)
        C = x
        x = torch.cat([x, dec_feats[0]], 1)
        # print(f'decoder-3 input: {x.shape}')
        x = self.dec3(x)  ## input: (b, 96+64, 128, 128)
        # shape: (b, 64, 256, 256)
        # print(f'decoder-3 output: {x.shape}')
        T = x

        #### These are not sbct
        return sb, C, T

    def forward(self, x):
        # x_yuv = convert_from_rgb2yuv(x)
        # x = torch.cat([x, x_yuv], dim=1)
        # x: (b, 3, 256, 256)
        # print(f"Encoder input: {x.shape}")
        x, esr_feat, dec_feats = self._encoder(x)
        sb_feat, C_feat, T_feat = self._decoder(x, dec_feats)
        return (sb_feat, C_feat, T_feat), esr_feat


class STDNGen(nn.Module):
    def __init__(self):
        super(STDNGen, self).__init__()
        ## To get spoof trace
        self.unet = UNET()

        # for s and b
        self.conv_sb1 = EncConv(64, 16)
        self.conv_sb2 = EncConv(16, 6, act=False, norm=False)

        # for C
        self.conv_C1 = EncConv(64, 16)
        self.conv_C2 = EncConv(16, 3, act=False, norm=False)
        self.down_avg = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # for T
        self.conv_T1 = EncConv(64, 16)

        self.conv_T2 = EncConv(16, 3, act=False, norm=False)

        ## Early Spoof Regressor

        self.tanh = nn.Tanh()
        self.esr = ESR(96, 1)
        self.normalize = normalize

    def _get_spoof_trace(self, feats):
        sb_feat, C_feat, T_feat = feats

        sb = self.conv_sb2(self.conv_sb1(sb_feat))
        sb = self.tanh(sb)

        s = sb[:, 3:, :, :]
        s = torch.mean(s, dim=[2, 3], keepdim=True)
        #         s = self.normalize(s)

        b = sb[:, :3, :, :]
        b = torch.mean(b, dim=[2, 3], keepdim=True)
        #         b = self.normalize(b)

        C = self.conv_C2(self.conv_C1(C_feat))
        C = self.tanh(C)
        C = self.down_avg(C)
        #         C = self.normalize(C)

        T = self.conv_T2(self.conv_T1(T_feat))
        T = self.tanh(T)
        #         T = self.normalize(T)

        return s, b, C, T

    def forward(self, x):
        # print(f"Unet input: {x.shape}")
        spoof_trace_feats, esr_feat_lis = self.unet(x)
        s, b, C, T = self._get_spoof_trace(spoof_trace_feats)
        # print(f's: {s.shape}, b: {b.shape}, C: {C.shape}, T: {T.shape}')
        esr_feat = self.esr(esr_feat_lis)
        return esr_feat, (s, b, C, T)


class STDNSingleDis(nn.Module):
    def __init__(self, in_C):
        super(STDNSingleDis, self).__init__()
        self.conv1 = EncConv(in_C, 32)
        self.down1 = DownSample(32, 32)

        self.conv2 = EncConv(32, 64)
        self.down2 = DownSample(64, 64)

        self.conv3 = EncConv(64, 64)
        self.down3 = DownSample(64, 96)

        self.conv4 = EncConv(96, 96)
        self.conv5 = EncConv(96, 1, act=False, norm=False)
        self.conv6 = EncConv(96, 1, act=False, norm=False)

    def forward(self, x):
        # x_yuv = convert_from_rgb2yuv(x)
        # x = torch.cat([x, x_yuv], dim=1)
        # print(f"Input shape of single Discriminator: {x.shape}")
        # x er input er bhitore ase --
        # actual input, fake live, fake spoof
        x = self.down1(self.conv1(x))
        x = self.down2(self.conv2(x))
        x = self.down3(self.conv3(x))
        x = self.conv4(x)

        x_l = self.conv5(x)
        x_s = self.conv6(x)
        # print(f"Output of Single Discriminator: {x_l.shape}, {x_s.shape}")
        return x_l, x_s


class STDNMultiDis(nn.Module):
    def __init__(self):
        super(STDNMultiDis, self).__init__()

        self.dis1 = STDNSingleDis(3)
        # self.dis1 = STDNSingleDis(3*2, 32)

        self.dis2 = STDNSingleDis(3)
        # self.dis2 = STDNSingleDis(3*2, 64)

        self.dis3 = STDNSingleDis(3)
        # self.dis3 = STDNSingleDis(3*2, 96)

    def forward(self, images):
        xls, xss = [], []
        for i in range(1, len(images) + 1):
            images[i - 1] = images[i - 1].float()
            # print(images[i-1].shape)
            x_l, x_s = getattr(self, f'dis{i}')(images[i - 1])
            xls.append(x_l)
            xss.append(x_s)

        return xls, xss


def gather_nd(params, indices):
    _y = tf.gather_nd(tf.convert_to_tensor(params.clone().detach().cpu().numpy()),
                      tf.convert_to_tensor(indices.clone().detach().cpu().numpy()))
    _y = torch.from_numpy(_y.numpy())

    return _y


def warping(trace, offsets):
    trace = trace.permute([0, 2, 3, 1])
    offsets = offsets.permute([0, 2, 3, 1])
    # print("IN WARPING:")
    # print(f"trace: {trace.device}, offset: {offsets.device}")
    bsize = trace.shape[0]
    xsize = trace.shape[1]  # 256
    offsets = offsets * imsize
    offsets = offsets[:, :, :, 0:2].view(bsize, -1, 2)  # 3rd channel is discared
    # print(offsets.shape)
    # first build the grid for target face coordinates
    t_coords = torch.meshgrid([torch.arange(xsize), torch.arange(xsize)])  # 2 tensors of (256, 256)
    # print(t_coords)
    t_coords = torch.stack(t_coords, dim=-1).float()  # (2, 256, 256)
    t_coords = t_coords.view((-1, 2))  # flattening (2, 256*256)
    t_coords = t_coords.unsqueeze(0).repeat(bsize, 1, 1).to(offsets.device)
    # print(f"t_coords: {t_coords.device}, offsets: {offsets.device}")
    # find the coordinates in the source image to copy pixels
    s_coords = t_coords + offsets
    s_coords = torch.clamp(s_coords, 0, (xsize - 1))
    n_coords = s_coords.shape[1]
    idx = torch.arange(bsize).unsqueeze(-1)
    idx = idx.repeat(1, n_coords)
    idx = idx.view(-1).to(s_coords.device)

    def _gather_pixel(_x, coords):
        coords = coords.float()
        xcoords = coords[:, :, 0].view(-1)
        ycoords = coords[:, :, 1].view(-1)
        ind = torch.stack([idx, xcoords, ycoords], dim=-1).long()

        _y = gather_nd(_x, ind).to(ind.device)
        # shape of x: (batch_size, 256, 256, 3)
        # shape of ind: (batch_size, 256*256, 3)
        # _y = gather_nd(_x, ind)
        _y = _y.view(bsize, n_coords, _x.shape[3])
        return _y

    # solve fractional coordinates via bilinear interpolation
    s_coords_lu = torch.floor(s_coords)
    s_coords_rb = torch.ceil(s_coords)
    s_coords_lb = torch.stack([s_coords_lu[:, :, 0], s_coords_rb[:, :, 1]], dim=-1)
    s_coords_ru = torch.stack([s_coords_rb[:, :, 0], s_coords_lu[:, :, 1]], dim=-1)
    _x_lu = _gather_pixel(trace, s_coords_lu)
    _x_rb = _gather_pixel(trace, s_coords_rb)
    _x_lb = _gather_pixel(trace, s_coords_lb)
    _x_ru = _gather_pixel(trace, s_coords_ru)
    # bilinear interpolation
    s_coords_fraction = s_coords - s_coords_lu.float()
    s_coords_fraction_x = s_coords_fraction[..., 0]
    s_coords_fraction_y = s_coords_fraction[..., 1]
    _xs, _ys = s_coords_fraction_x.shape
    s_coords_fraction_x = s_coords_fraction_x.view(_xs, _ys, 1)
    s_coords_fraction_y = s_coords_fraction_y.view(_xs, _ys, 1)
    _x_u = _x_lu + (_x_ru - _x_lu) * s_coords_fraction_x
    _x_b = _x_lb + (_x_rb - _x_lb) * s_coords_fraction_x
    warped_x = _x_u + (_x_b - _x_u) * s_coords_fraction_y
    warped_x = warped_x.view(bsize, xsize, xsize, -1)
    warped_x = warped_x.permute([0, 3, 1, 2])
    return warped_x


class ConstructImages:
    def __init__(self):
        super(ConstructImages, self).__init__()
        self.size = [256, 128, 64]
        self.size = [256, 160, 40]

    def __call__(self, x, spoof_trace_feats, warp_map):
        # print(f'input in construct: {x.shape}')
        s, b, C, T = spoof_trace_feats
        # print(f's: {s.shape}, b: {b.shape}, C: {C.shape}, T: {T.shape}')
        C = resize(C, (self.size[0], self.size[0]))
        # print(f's: {s.shape}, b: {b.shape}, C: {C.shape}, T: {T.shape}')
        recon1 = (1 - s) * x - b - C - T        ### reconstruction of an image wchich should be live
        trace = x - recon1          ## finding spoof trace
        #         trace = normalize(trace)
        # print(f"Recon shape: {recon1.shape}, trace shape: {trace.shape}")
        # trace: (b, 3, 256, 256)
        trace_warp = warping(trace[len(trace) // 2:], warp_map)         ## spoofed spoof_trace warped to match live alignment
        synth1 = x[:len(trace) // 2] + trace_warp           ### constructed spoof face from live
        im_d1 = torch.cat([x, recon1[len(trace) // 2:], synth1], 0)

        recon2 = resize(recon1, (self.size[1], self.size[1]))
        synth2 = resize(synth1, (self.size[1], self.size[1]))
        im_d2 = torch.cat([resize(x, (self.size[1], self.size[1])), recon2[len(trace) // 2:], synth2], 0)

        recon3 = resize(recon1, (self.size[2], self.size[2]))
        synth3 = resize(synth1, (self.size[2], self.size[2]))
        im_d3 = torch.cat([resize(x, (self.size[2], self.size[2])), recon3[len(trace) // 2:], synth3], 0)

        return [synth1, synth2, synth3], [im_d1, im_d2, im_d3], [recon1, recon2, recon3], trace_warp, trace

        ### synth: bsize/2
        ### recon: bsize
        ### im_d: bsize*2


class STDN(nn.Module):
    def __init__(self, **kwargs):
        super(STDN, self).__init__()
        self.gen = STDNGen()
        self.reconstruct = ConstructImages()
        self.disc = STDNMultiDis()
        self.no_grad_gen = STDNGen()

    def forward(self, x, warp):
        shape = x.size()
        # b, 3, 256, 256
        # real: b_1, 3, 256, 256
        # spoof: b_2, 3, 256, 256
        # b_1 + b_2 = b
        # print(f"inputs shape {x.shape}, {warp.shape}")
        esr_feat, (s, b, C, T) = self.gen(x)
        synthesized_images, disc_input, reconstructed_images, trace_warped, trace_unwarped = self.reconstruct(x, (
        s, b, C, T), warp)
        xls, xss = self.disc(disc_input)

        synth_cat = torch.cat([synthesized_images[0], reconstructed_images[0][len(x) // 2:]], dim=0).float()
        esr_feat_syn, (s_syn, b_syn, C_syn, T_syn) = self.no_grad_gen(synth_cat)
        recon_syn = s_syn * x + b_syn + resize(C_syn, (256, 256)) + T_syn
        return esr_feat, (s, b, C, T), (xls, xss), esr_feat_syn, (
        s_syn, b_syn, C_syn, T_syn), recon_syn, trace_warped, trace_unwarped, synth_cat

class STDNTest(nn.Module):
    def __init__(self, **kwargs):
        super(STDNTest, self).__init__()
        self.gen = STDNGen()
        self.size = 256

    def forward(self, x):
        shape = x.size()
        # b, 3, 256, 256
        # real: b_1, 3, 256, 256
        # spoof: b_2, 3, 256, 256
        # b_1 + b_2 = b
        # print(f"inputs shape {x.shape}, {warp.shape}")
        esr_feat, (s, b, C, T) = self.gen(x)
        C = resize(C, (self.size, self.size))
        # print(f's: {s.shape}, b: {b.shape}, C: {C.shape}, T: {T.shape}')
        recon1 = (1 - s) * x - b - C - T  ### reconstruction of an image wchich should be live
        unwarped_trace = x - recon1
        return esr_feat, (s, b, C, T), unwarped_trace
