# torch_image_template
A template for pytorch image handling project.

### File structure

```yaml
.
├── checkpoint
│   └── model_0001.ckpt
├── config.py
├── data_loader
│   ├── datadb.py
│   ├── data_loader.py
│   ├── imdb.py
│   └── pipeline.py  # image loading pipeline
├── main.py
├── models
│   ├── base_model.py
│   ├── layers.py  # layers (e.g. conv, devonv, batchnorm)
│   ├── losses.py  # common losses
│   ├── module.py  # network modules
│   ├── mynet.py
│   └── process.py  # image (pre)process
├── README.md
├── scripts
│   └── test_batch.py
└── utils
    └── misc_utils.py

```

### Train your own network
```shell script
    python3 main.py --train --input_dir train/ --output_dir checkpoint/ --epochs 100 --which_direction BtoA [--resume]
```

### Test the model
```shell script
    python3 main.py --test --input_dir val/ --output_dir test_results/ --checkpoint checkpoint/ 
```

### Visulization
```shell script
   visulization code here
```

