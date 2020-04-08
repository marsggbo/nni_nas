from yacs.config import CfgNode as _CfgNode

__all__ = [
    'CfgNode',
    'get_cfg',
    'add_config',
    'CN'
]

class CfgNode(_CfgNode):
    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            v = f"'{v}'" if isinstance(v, str) else v
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

global_cfg = CfgNode()
CN = CfgNode

def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    from .default import _C
    return _C.clone()

def add_config(cfg):
    '''
    add new node to cfg
    '''
    cfg.mixup = CN()
    cfg.mixup.enable = 0
    cfg.mixup.alpha = 0.4

    cfg.kd = CN()
    cfg.kd.enable = 0
    cfg.kd.model = CN()
    cfg.kd.model.name = 'Nasnetamobile'
    cfg.kd.model.path = 'teacher_net.pt'
    cfg.kd.loss = CN()
    cfg.kd.loss.alpha = 0.5
    cfg.kd.loss.temperature = 2

    cfg.loss.focal_loss = CN()
    cfg.loss.focal_loss.alpha = [2.03316646, 3.4860515 , 5.50677966, 1., 6.33333333, 8.24619289, 3.32889344, 2.75338983, 7.98280098, 8.57255937]
    cfg.loss.focal_loss.gamma = 2
    cfg.loss.focal_loss.size_average = True