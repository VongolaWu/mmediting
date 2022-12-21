_base_ = '../_base_/default_runtime.py'

experiment_name = 'nafnet_c64eb11128mb1db1111_lr1e-3_400k_gopro_diy'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='NAFNetLocal',
        img_channel=3,
        mid_channels=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    ),
    pixel_loss=dict(type='PSNRLoss'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='AvgFrames', keys=['img']),
    dict(type='PackEditInputs')
]

# dataset settings
train_dataset_type = 'MultipleFramesDataset'
val_dataset_type = 'MultipleFramesDataset'
data_root = '../datasets/Adobe240fps/'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,  # 8 gpu
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=train_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='mfi'),
        data_root=data_root + 'test',
        data_prefix=dict(img='full_sharp', gt='full_sharp'),
        pipeline=val_pipeline,
        depth=2,
        num_overlapped=0,
        load_frames_list=dict(img=[0, 1, 2, 3, 4, 5, 6, 7], gt=[4])))

val_dataloader = dict(
    num_workers=8,
    batch_size=4,  # 8 gpu
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='mfi'),
        data_root=data_root + 'test',
        data_prefix=dict(img='full_sharp', gt='full_sharp'),
        pipeline=val_pipeline,
        depth=2,
        num_overlapped=0,
        load_frames_list=dict(img=[0, 1, 2, 3, 4, 5, 6], gt=[3])))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400_000, val_interval=20000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9)))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=False, T_max=400_000, eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(bgr2rgb=False)

randomness = dict(seed=10, diff_rank_seed=True)
