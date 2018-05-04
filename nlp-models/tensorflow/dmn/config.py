"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=80)
parser.add_argument('--hidden_size', type=int, default=80)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--n_hops', type=int, default=2)
parser.add_argument('--clip_norm', type=float, default=5.0)

args = parser.parse_args()
"""
from bunch import Bunch

args = Bunch({
    'n_epochs': 10,
    'batch_size': 64,
    'embed_dim': 80,
    'hidden_size': 80,
    'dropout_rate': 0.1,
    'n_hops': 2,
    'clip_norm': 5.0,
})