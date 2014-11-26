import sys
from data_loader import *

def main(argv):
    data_loader = DataLoader("data/train/")
    print(data_loader.get_raw_data())

if __name__ == '__main__':
    main(sys.argv[1:])