"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.config import Config

if not hasattr(Config().trainer, 'use_mindspore'):
    from plato.trainers.mindspore import (
        basic as basic_mindspore, )

    registered_trainers = OrderedDict([
        ('basic', basic_mindspore.Trainer),
    ])
elif hasattr(Config().trainer, 'use_tensorflow'):
    from plato.trainers.tensorflow import (
        basic as basic_tensorflow, )

    registered_trainers = OrderedDict([
        ('basic', basic_tensorflow.Trainer),
    ])
else:
    try:
        from plato.trainers import (
            basic,
            pascal_voc,
        )

        registered_trainers = OrderedDict([
            ('basic', basic.Trainer),
            ('pascal_voc', pascal_voc.Trainer),
        ])
    except:
        pass


def get(model=None):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)
    # print('Config().trainer:', Config().trainer, flush=True)
    # print('Config().trainer,use_mindspore:', Config().trainer.use_mindspore, hasattr(Config().trainer, 'use_mindspore'),  flush=True)
    if Config().trainer.model_name == 'yolov5':
        if not hasattr(Config().trainer, 'use_mindspore'):
            from plato.trainers.mindspore import ms_yolo
            return ms_yolo.Trainer(model)
        else:
            from plato.trainers import yolo
            return yolo.Trainer()
    elif Config().trainer.type == 'HuggingFace':
        from plato.trainers import huggingface
        return huggingface.Trainer(model)
    elif trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name](model)
    else:
        raise ValueError('No such trainer: {}'.format(trainer_name))

    return registered_trainer
