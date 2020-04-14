
# CheckpointCallback

The flag of whether to save a cheeckpoint is the value of `save_metric` provided in trainer, as below.

```python
def train(self, validate=True):
    ...

    for callback in self.callbacks:
        if isinstance(callback, CheckpointCallback):
            callback.update_best_metric(meters.meters['save_metric'].avg)
        callback.on_epoch_end(epoch)
```

The `save_metric` is calculated in the function `metrics` of ``utils.utils.py`, as below:

```python
def metrics(outputs, targets, topk=(1, 3)):
    """Computes the precision@k for the specified values of k"""
    res = {}
    ...
    res['acc1'] = ...
    res['save_metric'] = ...
    return res
```


# knowledge distilltation for re-training

By default, we expect you to load a teacher network that is trained by using [torchline](https://github.com/marsggbo/torchline). You can also use your own method to load the teacher network by modifying `kd_model.py`.

