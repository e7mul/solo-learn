python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --data_dir /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 400 \
    --devices 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --dali \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --num_crops_per_aug 2 \
    --name swav-400ep-imagenet100 \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --proj_output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2
