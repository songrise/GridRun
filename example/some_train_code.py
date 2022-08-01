import argparse
import time
parser = argparse.ArgumentParser(description='Example of argparse')
parser.add_argument('--name', type=str, default='John',
                    help='Name of the experiment')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

args = parser.parse_args()
print(args.name+" Started")
time.sleep(5)
print(args.name+" Finished")
