NNI_NAS v0.2

# Dependences

- nni>=1.2
- albumentations
- torch>=1.2
- tensorboard

# debug

```bash
CUDA_VISIBLE_DEVICES=0 python search.py --config_file ./configs/search.yaml debug dataset.name fakedata
```

# search

```bash
CUDA_VISIBLE_DEVICES=0 python search.py --config_file ./configs/search.yaml
```


# retrain
after search, the arch checkpoint will be saved in, for example, `output/checkpoint_0/epoch_3.json`.

```
CUDA_VISIBLE_DEVICES=0 python retrain.py --config_file ./configs/retrain.yaml --arc_path output/checkpoint_0/epoch_3.json
```


# Todo

- [x] add DARTS (todo in sample_final)
- [ ] update documents
- [ ] use [torchline](https://github.com/marsggbo/torchline) API?