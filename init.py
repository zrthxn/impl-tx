import logging
from sys import argv
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from config import build_configuration

from src.preprocess import build_dataset
from src.models.transformer import Transformer
from src.training import train

def main(actions, args):

	vocab, datas = build_dataset(
		src_path="data/toy-ende/src-train.txt",
		tgt_path="data/toy-ende/tgt-train.txt")

	src_vocab, tgt_vocab = vocab

	Model = Transformer(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
	if actions.__contains__("train"):
		train(Model, datasets=datas, args=args)

	return

if __name__ == "__main__":
	actions = list()
	for i, arg in enumerate(argv[1:]):
		if arg.find("=") == -1:
			actions.append(arg)
			argv.pop(i)
		else:
			continue

	logging.basicConfig(level=logging.INFO)
	build_configuration(argv[1:])

	parser = ArgumentParser()
	parser = Trainer.add_argparse_args(parser)
	args = parser.parse_args()

	main(actions, args)
