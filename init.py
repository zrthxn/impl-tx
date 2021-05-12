import logging
from sys import argv
from config import build_configuration

from src.training import train
from src.preprocess import build_dataset


def main():
	src, tgt = build_dataset(
		src_path="data/toy-ende/src-train.txt",
		tgt_path="data/toy-ende/tgt-train.txt"
	)

	train()

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	build_configuration(argv[1:])
	main()
