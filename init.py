from sys import argv
from config import defaults, build_configuration
from src.training import train

def main():
	train()

if __name__ == "__main__":
	build_configuration(argv[1:])
	main()
