import itertools
import csv

def main():
    a = ["MLP", "CNN"]
    b = ["RC on X", "RC on all"]
    c = ["LDS on Y", "LDS on all"]
    d = ["Scale for RC", "None"]
    e = ["Scale for LDS", "None"]
    f = ["Clean", "Noise"]
    g = ["GN for W", "None"]
    h = ["One Opt", "Two Opts"]
    
    categories = [a, b, c, d, e, f, g, h]
    fname = "./experiment_list.csv"
    with open(fname, "w") as fpout:
        writer = csv.writer(fpout)
        for elm in itertools.product(*categories):
            writer.writerow(elm)
    
if __name__ == '__main__':
    main()
