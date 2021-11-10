import os
import json

func = 'read'

if func == 'save':
    root_dir = '/Midgard/Data/hfang/sn7_winner_split/'

    aois = sorted([f for f in os.listdir(os.path.join(root_dir, 'train'))
                if os.path.isdir(os.path.join(root_dir, 'train', f))])

    dic = {}
    for i, aoi in enumerate(aois):
        print(i, "aoi:", aoi)
        im_dir = os.path.join(root_dir, 'train', aoi, 'images_masked/')
        im_files = sorted([f
                    for f in os.listdir(os.path.join(im_dir))
                    if f.endswith('.tif') and os.path.exists(os.path.join(im_dir, f))])
        list = []
        for j, f in enumerate(im_files):
            month = f.split('_mosaic')[0].split('monthly_')[-1]
            list.append(month)
        dic.update({aoi: list})

    file = open("month.json", "w")
    json.dump(dic, file)
    file.close()

if func == 'read':
    a_file = open("/Midgard/home/hfang/temporal_CD/HRNet_SN7/lib/datasets/month.json", "r")
    dic = json.load(a_file)
    a_file.close()

    aoi = 'L15-0331E-1257N_1327_3160_13'

    print(type(dic[aoi]))