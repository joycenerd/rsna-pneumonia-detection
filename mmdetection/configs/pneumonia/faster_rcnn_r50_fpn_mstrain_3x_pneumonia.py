_base_='../faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py'

data_root = '/eva_data/zchin/rsna_data/'

model=dict(
    type='FasterRCNN',
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1
        )
    ),
    test_cfg=dict(
        rcnn=dict(score_thr=0.75)
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instance_train.json',
            img_prefix=data_root + 'images/train')),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instance_val.json',
        img_prefix=data_root + 'images/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instance_test.json',
        img_prefix=data_root + 'images/test',
        pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=40)
load_from='/home/zchin/rsna-pneumonia-detection/mmdetection/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'