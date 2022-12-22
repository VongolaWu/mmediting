_base_ = '../_base_/default_runtime.py'

experiment_name = 'mmp-rnn_adobe'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MMPRNN',
        para=dict(
            do_skip=True,
            n_blocks_a=9,
            n_blocks_b=10,
            n_features=18,
            past_frames=2,
            future_frames=2,
            centralize=True,
            normalize=True,
            activation='gelu')),
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
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

# dataset settings
train_dataset_type = 'MultipleFramesDataset'
val_dataset_type = 'MultipleFramesDataset'
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=train_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='../datasets/gopro/train',
        data_prefix=dict(gt='sharp', img='blur'),
        pipeline=train_pipeline,
        num_overlapped=4,
        depth=2,
        load_frames_list=dict(img=[0, 1, 2, 3, 4, 5, 6, 7], gt=[2, 3, 4, 5])))

val_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='../datasets/gopro/test',
        data_prefix=dict(gt='sharp', img='blur_gamma'),
        pipeline=val_pipeline,
        num_overlapped=4,
        depth=2,
        load_frames_list=dict(img=[0, 1, 2, 3, 4, 5, 6, 7], gt=[2, 3, 4, 5])))

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

visualizer = dict(img_keys=['input', 'gt_img', 'pred_img'], fn_key='key', bgr2rgb=True)

randomness = dict(seed=10, diff_rank_seed=True)
