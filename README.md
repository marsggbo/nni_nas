# debug

```bash
CUDA_VISIBLE_DEVICES=0 python -m ipdb search.py --debug dataset.name fakedata
```

# search

```bash
CUDA_VISIBLE_DEVICES=0 python search.py
```


# retrain
after search, the arch checkpoint will be saved in, for example, `output/checkpoint_0/epoch_3.json`.

```
CUDA_VISIBLE_DEVICES=0 python retrain.py --config_file ./config/skin10.yaml --arc-checkpoint output/checkpoint_0/epoch_3.json
```

# Todo

- [x] 数据集读取
- [x] 模型构建
- [x] 训练流程搭建
- [x] 验证流程搭建
- [x] 保存模型结构
- [ ] 改进loss
  - [x] 将模型大小或者flops考虑进去: add model size as loss with 0.02 weight
- [ ] 理清搜索和训练流程
- [x] warm start
- [x] save the best model and mutator
- [x] resume training
- [x] retrain model
- [ ] 借鉴GhostNet设计思路
- [ ] 使用torchline接口
- [x] 使用torchline中的albumentations
- [ ] DARTS
- [ ] 多存一些json，绘制模型大小约束下，model size和acc散点图，
- [ ] 搜索阶段逐渐增加网络层数
- [x] retrain之前先对生成的所有网络简单eval一下，然后选择最好的retrain
- [ ] 添加测试demo(网络、数据集、mutator、outputs)，方便调试代码
- [x] retrain mixup
- [ ] 完善evaluator.py中的finetune部分代码，提高兼容性

