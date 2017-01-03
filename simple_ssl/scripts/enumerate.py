import itertools
import csv

def main():
    a = ["MLP", "CNN"]
    b = ["RC on X", "RC on all"]
    c = ["LDS on Y", "LDS on all"]
    d = ["Scale for RC", "Scale for LDS"]
    e = ["Clean", "Noise"]
    f = ["GN for W", "None"]
    g = ["One Opt", "Two Opts"]
    
    categories = [a, b, c, d]
    fname = "./experiment_list.csv"
    with open(fname, "w") as fpout:
        writer = csv.writer(fpout)
        for elm in itertools.product(*categories):
            writer.writerow(elm)
    
if __name__ == '__main__':
    main()
