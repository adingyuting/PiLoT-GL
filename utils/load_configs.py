import argparse

from constant import SEED, DatasetType


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_link_prediction_args():
    parser = argparse.ArgumentParser(description="Train M1 prior-conditioned LoRA PL-DisGLSL.")

    parser.add_argument("--dataset_name", type=str, default=DatasetType.WIKI_GL.name.lower(),
                        choices=[ds.name.lower() for ds in DatasetType])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="SGD-DyG-M1-prior-conditioned-lora-PL-DisGLSL")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cuda", type=str2bool, default=True)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lam", type=float, default=0.0005)
    parser.add_argument("--tau", type=float, default=0.1)

    parser.add_argument("--num_feature", type=int, default=8)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--hidden_feature", type=int, default=16)
    parser.add_argument("--bandwidth", type=int, default=20)
    parser.add_argument("--m_choice", type=int, default=2)
    parser.add_argument("--tgc_dropout", type=float, default=0.75)
    parser.add_argument("--fft_dropout", type=float, default=0.75)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--r_mid", type=int, default=3)
    parser.add_argument("--r_long", type=int, default=8)
    parser.add_argument("--cdpss_hidden_dim", type=int, default=64)
    parser.add_argument("--cdpss_dropout", type=float, default=0.2)
    parser.add_argument("--attention_temperature", type=float, default=1.5)

    parser.add_argument("--cl_powerlaw_eta", type=float, default=0.5)
    parser.add_argument("--cl_weight_min", type=float, default=0.5)
    parser.add_argument("--cl_weight_max", type=float, default=2.0)
    parser.add_argument("--cl_degree_bins", default="0,1,2,4,8,16,32,64,128")
    parser.add_argument("--cl_disagreement_alpha", type=float, default=0.5)
    parser.add_argument("--cl_disagreement_min", type=float, default=0.2)
    parser.add_argument("--cl_disagreement_max", type=float, default=1.0)
    parser.add_argument("--cl_disagreement_normalize", type=str2bool, default=True)

    args = parser.parse_args()
    args.hidden_features = [args.hidden_feature] * args.layer
    return args
