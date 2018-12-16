import os



def main():
    # config
    d2l_fpath = "dirname_to_label.txt"
    picked_label_fpath = "cherry-picked-class-id.txt"
    

    l2d_map = {}
    with open(d2l_fpath) as fp:
        for l in fp:
            d, l = l.rstrip().split(" ")
            l2d_map[l] = d

    with open(picked_label_fpath) as fp:
        for l in fp:
            l = l.rstrip()
            print("{} {}".format(l2d_map[l], l))
            
if __name__ == '__main__':
    main()
