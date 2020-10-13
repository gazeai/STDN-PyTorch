import torch


def read_config(flags, phase="train"):
    # ckpt_path = f"{flags.checkpoint.path}/{flags.checkpoint.format}"
    labels = flags.labels

    data_config = flags.data_config
    data_loaders = get_dataloader(data_config, labels, phase)

    arch_config = flags.arch_config

    train_config = flags.train_config
    arch_info = get_arch_info(arch_config, train_config)

    optim_info = flags.param_config.optimizer
    optimizer = get_optimizer(optim_info, arch_info['model'].parameters())

    if flags.param_config.scheduler is not None:
        scheduler = get_scheduler(optimizer, flags.param_config.scheduler)
    else:
        scheduler = None

    loss_info = flags.param_config.loss
    loss = get_loss(loss_info)

    return arch_info, optimizer, data_loaders, loss, scheduler, flags


def get_scheduler(optimizer, scheduler_info):
    params = {key: value for key, value in scheduler_info.items()}
    scheduler = params["name"]
    del params["name"]
    return scheduler(optimizer, **params)
def get_loss(loss_info):
    # params = [x for x in loss_info.values()]
    params = {key: value for key, value in loss_info.items()}
    loss = params['name']
    del params['name']
    return loss(**params)


def get_optimizer(optim_info, parameters):
    params = {key: value for key, value in optim_info.items()}
    opt = params['name']
    del params['name']
    return opt(parameters, **params)

def get_arch_info(arch_info, train_info):

    params = {key: value for key, value in arch_info.items()}
    model_name = params['name']
    del params['name']
    architecture = model_name(**params)

    arch_info = {"model": architecture,
            "losses": [],
            "start_epoch": 0}
    if train_info.checkpoint.resume:
        from utils.scripts import load_checkpoint
        arch_info = load_checkpoint(arch_info['model'], train_info.checkpoint)
    return arch_info


def get_dataloader(data_info, labels, phase):
    loader_info = {key: value for key, value in data_info.dataloader.items()}
    cls_name = loader_info["name"]
    del loader_info['name']
    datasets = {}
    if phase == "train":
        for p in ['train', 'val']:
            datasets[p] = cls_name(data_info, p, **loader_info)
    else:
        for p in [phase]:
            datasets[p] = cls_name(data_info, p, **loader_info)

    dataloaders = {}
    num_workers = data_info.num_workers
    print(f'Number of workers: {num_workers}')
    SAMPLER = data_info.sampler
    BATCH_SAMPLER = data_info.batch_sampler
    for key in datasets.keys():
        batch_size = eval(f"data_info.{key}.batch_size")
        # shuffle = eval(f"data_info.{key}.shuffle")
        print(key, batch_size)
        print(f'Number of samples in {key}: {len(datasets[key])}')
        if BATCH_SAMPLER is not None:
            # from utils.dataloader.sampler import BalancedBatchSampler
            # print(batch_sampler, BalancedBatchSampler)
            batch_sampler = BATCH_SAMPLER(datasets[key], len(labels), batch_size//len(labels))
            # print(list(batch_sampler))
            dataloaders[key] = torch.utils.data.DataLoader(datasets[key],
                                                           batch_sampler=batch_sampler,
                                                           num_workers=num_workers)
        else:
            if SAMPLER is not None:
                sampler = SAMPLER
            else:
                sampler = None
            dataloaders[key] = torch.utils.data.DataLoader(datasets[key],
                                                       batch_size=batch_size,
                                                       shuffle=(sampler is None and key is not "test"),
                                                       num_workers=num_workers,
                                                       sampler=sampler)
        # sampler=sampler)

    return dataloaders
