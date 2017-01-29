import itertools
import csv

def main():
    a = ["MLP", "CNN", "STN"]
    b = ["None", "Residual", "Residual+Skip"]
    c = ["None", "Conditioned"]
    d = ["None", "Patch"]
    
    categories = [a, b, c, d]
    fname = "./experiment_list.csv"
    with open(fname, "w") as fpout:
        writer = csv.writer(fpout)
        for elm in itertools.product(*categories):
            writer.writerow(elm)
    
if __name__ == '__main__':
    main()
