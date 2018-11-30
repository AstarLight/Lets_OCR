###
#   transform original gt to location gt
###

import os


def rawGT_to_locGT(in_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    files_list = os.listdir(in_path)
    for name in files_list:
        in_file = os.path.join(in_path, name)
        out_file = os.path.join(out_path, 'gt_'+name)
        f1 = open(in_file, 'r')
        #f1 = codecs.open(in_file, 'r', 'utf-8-sig')
        lines = f1.readlines()
        f1.close()
        f2 = open(out_file, 'w+')
        #print("img %s %s" % (in_file, lines))

        for line in lines:
            line.strip()
            if line.split(',')[-2] == 'Arabic':
                continue
            loc = line.split(',')[:8]
            str1 = ",".join(loc)
            str1.strip()
            #print("img %s raw str is %s" % (in_file, line))
            #print("img %s aft str is %s" % (in_file, str1))
            f2.write(str1)
            f2.write('\n')
        f2.close()


rawGT_to_locGT('/home/ljs/OCR_dataset/ali_ocr/train_1000/txt_1000', '/home/ljs/data_ready/ali_icpr/gt_1000')
#rawGT_to_locGT('/home/ljs/OCR_dataset/MLT/val_gt', '/home/ljs/OCR_dataset/MLT/val_loc_gt')