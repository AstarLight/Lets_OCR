import os

ROOT = './extra/extra_labels'
labels_names = os.listdir(ROOT)
for label in labels_names:
    full_name = os.path.join(ROOT, label)
    full_out = os.path.join("./extra/labels", label)
    fout = open(full_out, "w+")
    with open(full_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            x = int(line[0])
            y = int(line[1])
            w = int(line[2])
            h = int(line[3])
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y
            x3 = x + w
            y3 = y + h
            x4 = x
            y4 = y + h
            loc = "%d,%d,%d,%d,%d,%d,%d,%d\n" % (x1,y1,x2,y2,x3,y3,x4,y4)
            fout.write(loc)
    fout.close()
