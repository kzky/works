from lds.svhn.datasets import Converter

def main():
    converter = Converter()

    fpath_inp="/home/kzk/datasets/svhn/train_32x32.mat"
    fpath_out="/home/kzk/datasets/svhn/train.mat"
    converter.convert_then_save(fpath_inp, fpath_out)

    fpath_inp="/home/kzk/datasets/svhn/test_32x32.mat"
    fpath_out="/home/kzk/datasets/svhn/test.mat"
    converter.convert_then_save(fpath_inp, fpath_out)
    
if __name__ == '__main__':
    main()
