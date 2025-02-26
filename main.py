import sys

from src.runner import Runner


if __name__ == "__main__":
    args = sys.argv[1:]
    configs_dir = args[0]
    main_runner = Runner(configs_dir) 
    main_runner.train()
