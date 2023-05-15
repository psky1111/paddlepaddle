cfg = dict(
    model='CVLP_r50',
    desc_path='./data/imagenet',
    pretrained_clip='./pretrained_models/RN50.pdparams',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    data_set='IMNET_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,
    train_mode=False,

    lr=5e-5,
    min_lr=0.,

    epochs=30,
    batch_size=128,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',

    eval_pretrain=True,
    test=False
)