import torchvision as tv
from torch import optim
from loss import STDNLoss
from utils import transforms as tf
from utils import joint_transforms as j_tf
from easydict import EasyDict as edict
from dataloader import STDNDataset
from utils.samplers import BalancedBatchSampler

flags = edict()

flags.device = 'cuda:0'
from model import STDN as STDNModel

flags.train_config = edict()
flags.data_config = edict()
flags.arch_config = edict()
flags.param_config = edict()


flags.labels = edict()
flags.labels.real = 0
flags.labels.spoof = 1


'''
Specify configurations for Dataloader
'''

flags.data_config.name = 'dataset name'
flags.data_config.protocol = 'protocol'

flags.data_config.needed_columns = ['rgb_path', 'keypoints', 'label']
flags.data_config.data_columns = ['rgb_path']
flags.data_config.target_columns = ['label']
flags.data_config.order = 0

flags.data_config.shape = edict()
flags.data_config.shape.rgb = (256, 256)

flags.data_config.color_mode = edict()
flags.data_config.color_mode.rgb_path = 'rgb'

flags.data_config.num_workers = 8
flags.data_config.sampler = None
flags.data_config.batch_sampler = BalancedBatchSampler

flags.data_config.frame_interval = 3

flags.data_config.dataloader = edict()
flags.data_config.dataloader.name = STDNDataset

flags.data_config.train = edict()
flags.data_config.train.path = 'data/train.csv'
flags.data_config.train.batch_size = 2
flags.data_config.train.shuffle = True

def t():
    kp = tv.transforms.Compose([
        tf.Transform4EachKey([tf.Transform4EachElement([tf.LoadNP()])], key_list=["keypoints"]),
        tf.JointTransforms([
            j_tf.ScaleTensor(base_key=0, trans_key=1, size=flags.data_config.shape.rgb[0]),
            j_tf.ToPILImage(),
            j_tf.RandomHorizontalFlip(p=0.5)],
            key_list=['rgb_path', 'keypoints']),
        tf.Transform4EachKey([
            tf.Transform4EachElement([
                tv.transforms.ToTensor(),
            ])
        ], key_list=["keypoints"]),
    ])
    rgb = tf.Transform4EachKey([
        tf.Transform4EachElement(
            [
                tv.transforms.Compose([
                    tv.transforms.Resize(flags.data_config.shape.rgb)]),
                tv.transforms.ToTensor()
            ]
        )], key_list=['rgb_path'])
    return tv.transforms.Compose([kp, rgb])

flags.data_config.train.transforms = t

flags.data_config.test = edict()
flags.data_config.test.path = 'your test path in Dataframe'

flags.data_config.test.batch_size = 2
flags.data_config.test.shuffle = False

def t():
    kp = tv.transforms.Compose([
        tf.Transform4EachKey([tf.Transform4EachElement([tf.LoadNP()])], key_list=["keypoints"]),
        tf.JointTransforms([
            j_tf.ScaleTensor(base_key=0, trans_key=1, size=flags.data_config.shape.rgb[0]),
            j_tf.ToPILImage()
        ],
            key_list=['rgb_path', 'keypoints']),
        tf.Transform4EachKey([
            tf.Transform4EachElement([
                tv.transforms.ToTensor(),
            ])
        ], key_list=["keypoints"]),

    ])
    rgb = tf.Transform4EachKey([
        tf.Transform4EachElement(
            [
                # tf.ConvertColor(flags.data_config.color_mode),
                tv.transforms.Compose([
                    # tv.transforms.ToPILImage(),
                    tv.transforms.Resize(flags.data_config.shape.rgb)]),
                # tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.ToTensor()
                # tv.transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )], key_list=['rgb_path'])

    return tv.transforms.Compose([kp, rgb])

flags.data_config.test.transforms = t
flags.data_config.val = edict()
flags.data_config.val.path = 'data/train.csv'
flags.data_config.val.batch_size = 2
flags.data_config.val.shuffle = True

def t():
    rgb = tf.Transform4EachKey([
        tf.Transform4EachElement(
            [
                tv.transforms.Compose([
                    tv.transforms.Resize(flags.data_config.shape.rgb)
                ]),
                tv.transforms.ToTensor()
            ]
        )], key_list=['rgb_path'])

    kp = tv.transforms.Compose([
        tf.Transform4EachKey([tf.Transform4EachElement([tf.LoadNP()])], key_list=["keypoints"]),
        tf.JointTransforms([j_tf.ScaleTensor(base_key=0, trans_key=1, size=flags.data_config.shape.rgb[0]),
                            j_tf.ToPILImage(),
                            # j_tf.ResizeByScale(),
                            # j_tf.RandomHorizontalFlip(p=0.5)
                            ], key_list=['rgb_path', 'keypoints']),
        tf.Transform4EachKey([
            tf.Transform4EachElement([
                tv.transforms.ToTensor(),
            ])
        ], key_list=["keypoints"]),
    ])

    return tv.transforms.Compose([kp, rgb])

flags.data_config.val.transforms = t

'''
specify all types of parameters
'''

flags.param_config.num_classes = 1

flags.param_config.optimizer = edict()
flags.param_config.optimizer.name = optim.Adam
#     flags.param_config.optimizer.lr = 6e-5
flags.param_config.optimizer.lr = 1e-4
flags.param_config.optimizer.betas = (0.9, 0.999)
flags.param_config.optimizer.eps = 1e-08
flags.param_config.optimizer.weight_decay = 0

flags.param_config.scheduler = edict()
flags.param_config.scheduler.name = optim.lr_scheduler.StepLR
flags.param_config.scheduler.gamma = 0.9
flags.param_config.scheduler.step_size = 1

# flags.param_config.frame_interval = 3

flags.param_config.loss = edict()
flags.param_config.loss.name = STDNLoss
flags.param_config.loss.gan_labels = "1_1"
flags.param_config.loss.esr_labels = "-1_1"
flags.param_config.loss.disc_labels = "1_0"
flags.param_config.loss.spoof_trace_labels = "0_0"
flags.param_config.loss.step_one = "50_1_1"
flags.param_config.loss.step_two = "1"
flags.param_config.loss.step_three = "5_0.1"
flags.param_config.loss.device = flags.device

flags.param_config.epoch = 50
# flags.param_config.eval_interval = 1

'''
Specify model name and its parameters
'''

flags.arch_config.name = STDNModel
flags.arch_config.nf = 1
flags.arch_config.multiframe = False
flags.arch_config.alpha = 0.5
flags.arch_config.num_classes = flags.param_config.num_classes

flags.train_config.do_crossval = True

'''
Specify training configuration
'''

## when frozen backbone is required
flags.train_config.backbone = edict()
flags.train_config.backbone.freeze = False
flags.train_config.backbone.name = "no_grad_gen"
flags.train_config.backbone.freeze_epoch = 0
flags.train_config.backbone.unfreeze_epoch = None

flags.train_config.checkpoint = edict()
flags.train_config.checkpoint.resume = False
flags.train_config.checkpoint.save_path = f"ckpts/stdn/{flags.data_config.name}/{flags.data_config.protocol}/batch_size_{flags.data_config.train.batch_size}_gan_{flags.param_config.loss.gan_labels}_disc_{flags.param_config.loss.disc_labels}_spoof_trace_{flags.param_config.loss.spoof_trace_labels}_esr_{flags.param_config.loss.esr_labels}"
flags.train_config.checkpoint.load_path = None
flags.train_config.checkpoint.preferred_metric = 'binary'



