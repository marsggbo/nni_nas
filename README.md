NNI_NAS v0.1

# Dependences

- nni>=1.2
- albumentations
- torch>=1.2
- tensorboard

# debug

```bash
CUDA_VISIBLE_DEVICES=0 python search.py --config_file ./connfig/search.yaml --debug dataset.name fakedata
```

# search

```bash
CUDA_VISIBLE_DEVICES=0 python search.py --config_file ./connfig/search.yaml
```


# retrain
after search, the arch checkpoint will be saved in, for example, `output/checkpoint_0/epoch_3.json`.

```
CUDA_VISIBLE_DEVICES=0 python retrain.py --config_file ./config/retrain.yaml --arc-checkpoint output/checkpoint_0/epoch_3.json
```

# Features

## 1. knowledge distilltation for re-training

By default, we expect you to load a teacher network that is trained by using [torchline](https://github.com/marsggbo/torchline). You can also use your own method to load the teacher network by modifying `kd_model.py`.

# Todo

- [ ] add DARTS
- [ ] update documents
- [ ] use [torchline](https://github.com/marsggbo/torchline) API?