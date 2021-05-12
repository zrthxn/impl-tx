import logging
from sys import argv

from torch.tensor import Tensor
from config import build_configuration

from src.training import train_session
from src.preprocess import build_dataset
from src.models import transformer

def main():
	# train, valid = build_dataset(
	# 	src_path="data/toy-ende/src-train.txt",
	# 	tgt_path="data/toy-ende/tgt-train.txt"
	# )

	Model = transformer.SelfAttention(5, 5)

	x = [
		[0.1, 0.1, 0.1, 0.1, 0.1],
		[0.1, 0.1, 0.1, 0.1, 0.1],
	]

	batch = Tensor([x, x, x])

	Model(batch, batch, batch, None)

	# train_session(Model, train, valid)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	build_configuration(argv[1:])
	main()
