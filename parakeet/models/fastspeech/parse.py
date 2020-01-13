import jsonargparse

def add_config_options_to_parser(parser):
    parser.add_argument('--audio.num_mels', type=int, default=80,
        help="the number of mel bands when calculating mel spectrograms.")
    parser.add_argument('--audio.n_fft', type=int, default=2048,
        help="the number of fft components.")
    parser.add_argument('--audio.sr', type=int, default=22050,
        help="the sampling rate of audio data file.")
    parser.add_argument('--audio.preemphasis', type=float, default=0.97,
        help="the preemphasis coefficient.")
    parser.add_argument('--audio.hop_length', type=int, default=128,
        help="the number of samples to advance between frames.")
    parser.add_argument('--audio.win_length', type=int, default=1024,
        help="the length (width) of the window function.")
    parser.add_argument('--audio.power', type=float, default=1.4,
        help="the power to raise before griffin-lim.")
    parser.add_argument('--audio.min_level_db', type=int, default=-100,
        help="the minimum level db.")
    parser.add_argument('--audio.ref_level_db', type=int, default=20,
        help="the reference level db.")
    parser.add_argument('--audio.outputs_per_step', type=int, default=1,
        help="the outputs per step.")
    
    parser.add_argument('--fs_embedding_size', type=int, default=256,
        help="the dim size of embedding of fastspeech.")
    parser.add_argument('--encoder_n_layer', type=int, default=6,
        help="the number of FFT Block in encoder.")
    parser.add_argument('--encoder_head', type=int, default=2,
        help="the attention head number in encoder.")
    parser.add_argument('--encoder_conv1d_filter_size', type=int, default=1024,
        help="the filter size of conv1d in encoder.")
    parser.add_argument('--max_sep_len', type=int, default=2048,
        help="the max length of sequence.")
    parser.add_argument('--decoder_n_layer', type=int, default=6,
        help="the number of FFT Block in decoder.")
    parser.add_argument('--decoder_head', type=int, default=2,
        help="the attention head number in decoder.")
    parser.add_argument('--decoder_conv1d_filter_size', type=int, default=1024,
        help="the filter size of conv1d in decoder.")
    parser.add_argument('--fs_hidden_size', type=int, default=256,
        help="the hidden size in model of fastspeech.")
    parser.add_argument('--duration_predictor_output_size', type=int, default=256,
        help="the output size of duration predictior.")
    parser.add_argument('--duration_predictor_filter_size', type=int, default=3,
        help="the filter size of conv1d in duration prediction.")
    parser.add_argument('--fft_conv1d_filter', type=int, default=3,
        help="the filter size of conv1d in fft.")
    parser.add_argument('--fft_conv1d_padding', type=int, default=1,
        help="the padding size of conv1d in fft.")
    parser.add_argument('--dropout', type=float, default=0.1,
        help="the dropout in network.")
    parser.add_argument('--transformer_head', type=int, default=4,
        help="the attention head num of transformerTTS.")

    parser.add_argument('--hidden_size', type=int, default=256,
        help="the hidden size in model of transformerTTS.")
    parser.add_argument('--embedding_size', type=int, default=256,
        help="the dim size of embedding of transformerTTS.")

    parser.add_argument('--warm_up_step', type=int, default=4000,
        help="the warm up step of learning rate.")
    parser.add_argument('--grad_clip_thresh', type=float, default=1.0,
        help="the threshold of grad clip.")
    parser.add_argument('--batch_size', type=int, default=32,
        help="batch size for training.")
    parser.add_argument('--epochs', type=int, default=10000,
        help="the number of epoch for training.")
    parser.add_argument('--lr', type=float, default=0.001,
        help="the learning rate for training.")
    parser.add_argument('--save_step', type=int, default=500,
        help="checkpointing interval during training.")
    parser.add_argument('--use_gpu', type=bool, default=True,
        help="use gpu or not during training.")
    parser.add_argument('--use_data_parallel', type=bool, default=False,
        help="use data parallel or not during training.")

    parser.add_argument('--data_path', type=str, default='./dataset/LJSpeech-1.1',
        help="the path of dataset.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
        help="the path to load checkpoint or pretrain model.")
    parser.add_argument('--save_path', type=str, default='./checkpoint',
        help="the path to save checkpoint.")
    parser.add_argument('--log_dir', type=str, default='./log',
        help="the directory to save tensorboard log.")
    parser.add_argument('--sample_path', type=str, default='./sample',
        help="the directory to save audio sample in synthesis.")
    parser.add_argument('--transtts_path', type=str, default='./log',
        help="the directory to load pretrain transformerTTS model.")
    parser.add_argument('--transformer_step', type=int, default=70000,
        help="the step to load transformerTTS model.")
    

    parser.add_argument('-c', '--config', action=jsonargparse.ActionConfigFile)
