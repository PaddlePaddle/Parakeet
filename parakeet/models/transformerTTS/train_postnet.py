from network import *
from preprocess import batch_examples_postnet, LJSpeech
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from parakeet.data.datacargo import DataCargo
from pathlib import Path
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint

class MyDataParallel(dg.parallel.DataParallel):
    """
    A data parallel proxy for model.
    """

    def __init__(self, layers, strategy):
        super(MyDataParallel, self).__init__(layers, strategy)

    def __getattr__(self, key):
        if key in self.__dict__:
            return object.__getattribute__(self, key)
        elif key is "_layers":
            return object.__getattribute__(self, "_sub_layers")["_layers"]
        else:
            return getattr(
                object.__getattribute__(self, "_sub_layers")["_layers"], key)


def main():
    parser = jsonargparse.ArgumentParser(description="Train postnet model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/train_postnet.yaml'.split())
    
    local_rank = dg.parallel.Env().local_rank

    if local_rank == 0:
        # Print the whole config setting.
        pprint(jsonargparse.namespace_to_dict(cfg))

    LJSPEECH_ROOT = Path(cfg.data_path)
    dataset = LJSpeech(LJSPEECH_ROOT)
    dataloader = DataCargo(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=batch_examples_postnet, drop_last=True)
    
    global_step = 0
    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if cfg.use_data_parallel else fluid.CUDAPlace(0)
             if cfg.use_gpu else fluid.CPUPlace())
    
    if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
    path = os.path.join(cfg.log_dir,'postnet')
    writer = SummaryWriter(path)

    with dg.guard(place):
         # dataloader
        input_fields = {
                'names': ['mel', 'mag'],
                'shapes':
                [[cfg.batch_size, None, 80], [cfg.batch_size, None, 257]], 
                'dtypes': ['float32', 'float32'],
                'lod_levels': [0, 0]
            }

        inputs = [
            fluid.data(
                name=input_fields['names'][i],
                shape=input_fields['shapes'][i],
                dtype=input_fields['dtypes'][i],
                lod_level=input_fields['lod_levels'][i])
            for i in range(len(input_fields['names']))
        ]

        reader = fluid.io.DataLoader.from_generator(
            feed_list=inputs,
            capacity=32,
            iterable=True,
            use_double_buffer=True,
            return_list=True)

        
        model = ModelPostNet('postnet', cfg)

        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=dg.NoamDecay(1/(4000 *( cfg.lr ** 2)), 4000))

        if cfg.checkpoint_path is not None:
            model_dict, opti_dict = fluid.dygraph.load_dygraph(cfg.checkpoint_path)
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            print("load checkpoint!!!")

        if cfg.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = MyDataParallel(model, strategy)

        for epoch in range(cfg.epochs):
            reader.set_batch_generator(dataloader, place)
            pbar = tqdm(reader())
            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d'%epoch)
                mel, mag = data
                mag = dg.to_variable(mag.numpy())
                mel = dg.to_variable(mel.numpy())
                global_step += 1

                mag_pred = model(mel)
                
                loss = layers.mean(layers.abs(layers.elementwise_sub(mag_pred, mag)))
                if cfg.use_data_parallel:
                    loss = model.scale_loss(loss)

                writer.add_scalars('training_loss',{
                    'loss':loss.numpy(),
                }, global_step)

                loss.backward()
                if cfg.use_data_parallel:
                    model.apply_collective_grads()
                optimizer.minimize(loss, grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(1))
                model.clear_gradients()

                if global_step % cfg.save_step == 0:
                    if not os.path.exists(cfg.save_path):
                        os.mkdir(cfg.save_path)
                    save_path = os.path.join(cfg.save_path,'postnet/%d' % global_step)
                    dg.save_dygraph(model.state_dict(), save_path)
                    dg.save_dygraph(optimizer.state_dict(), save_path)

                



if __name__ == '__main__':
    main()