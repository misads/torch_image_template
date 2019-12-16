# torch_image_template
A template for pytorch image handling project.

### File structure

```yaml
.
├── checkpoints
│   └── tag_1  # Saved checkpoints
├── logs
│   └── tag_1  # Log and tensorboard files
├── README.md
├── backbone
│   └── linknet.py
├── dataloader
│   ├── dual_residual_dataset.py
│   ├── image_folder.py  # Folder image dataloader
│   ├── reside_dataset.py
│   └── transforms.py
├── eval.py
├── loss
│   └── content_loss.py
├── network
│   ├── DuRN_Pure_Conv.py
│   ├── Model.py
│   ├── Ms_Discriminator.py
│   ├── base_model.py
│   ├── metrics.py
│   └── norm.py
├── options
│   ├── __init__.py
│   └── options.py  # Args
├── scripts
│   ├── test_scipt.py
│   └── train_script.py
├── test.py
├── train.py
└── utils
    ├── misc_utils.py
    └── torch_utils.py

```

### Train your own network
```bash
    python3 train.py --tag tag_1 --batch_size 16 --epochs 100 [--load <pretrained models folder> --which-epoch 500] --gpu_ids 0
```

### Test the model
```shell script
    python3 test.py --dataset voc --load <pretrained models folder> --which-epoch 500
```

### Visulization
```shell script
   tensorboard --logdir logs/tag_1
```

