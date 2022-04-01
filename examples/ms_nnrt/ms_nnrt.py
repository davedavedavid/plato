import os

import ms_nnrt_datasource_yolo as nnrt_datasource
import ms_nnrt_trainer_yolo as nnrt_trainer
from ms_nnrt_algorithms import ms_mistnet
from ms_nnrt_models import ms_acl_inference
from plato.config import Config
from plato.clients.mistnet import Client

os.environ['config_file'] = './ms_mistnet_yolov5.yml'


print('test.')

def main():
    """ A Plato mistnet training sesstion using a nnrt yolo model, datasource and trainer. """
    datasource = nnrt_datasource.DataSource(
    )  # special datasource for yolo model

    model = ms_acl_inference.Inference(int(Config().trainer.deviceID),
                                    Config().trainer.om_path,
                                    Config().data.input_height,
                                    Config().data.input_width)

    trainer = nnrt_trainer.Trainer(model=model)
    algorithm = ms_mistnet.Algorithm(trainer)

    client = Client(model=model,
                    datasource=datasource,
                    algorithm=algorithm,
                    trainer=trainer)
    client.load_data()
    _, features = client.train()


if __name__ == "__main__":
    main()