import itertools
import csv

def main():
    a = ["MLP", "Elman", "LSTM"]
    b = ["Parameter tying", "Reconstruct"]
    c = ["Use all inputs", "First one"]
    d = ["Use all output", "Last one", "Confidence"]
    e = ["Use soft", "Hard value"]
    f = ["MH-RNN", "Not MH-RNN"]

    categories = [a, b, c, d, e, f]
    fname = "./experiment_list.csv"
    with open(fname, "w") as fpout:
        writer = csv.writer(fpout)
        for elm in itertools.product(*categories):
            writer.writerow(elm)
    
if __name__ == '__main__':
    main()
