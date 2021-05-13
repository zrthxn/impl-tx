import logging
from sys import argv
from config import build_configuration

from src.training import train_session
from src.preprocess import build_dataset
from src.models import transformer, bert

def main():
	vocab, iters = build_dataset(
		src_path="data/toy-ende/src-train.txt",
		tgt_path="data/toy-ende/tgt-train.txt")

	src_vocab, tgt_vocab = vocab
	src_iters, tgt_iters = iters

	Model = transformer.Transformer(
		src_vocab=src_vocab,
		tgt_vocab=tgt_vocab,
		seq_length=2)

	# Model = train_session(Model, train)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	build_configuration(argv[1:])
	main()
