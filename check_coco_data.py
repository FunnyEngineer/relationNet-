import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
anno_file = './datasets/coco/annotations/instances_val2017.json'
f = open(anno_file)

data = json.load(f)

cocoGt = COCO(anno_file) 
cocoDt = cocoGt.loadRes('output/test_result_retina.json')  
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# cocoGt = COCO(anno_file) 
# import pdb; pdb.set_trace()

count_dict = {}
# create image to image id dict 
for img_info in data['images']:
    count_dict[img_info['id']] = 0

for anno_info in data['annotations']:
    count_dict[anno_info['image_id']] += 1
count = 0
for i in count_dict:
    # print(i)
    if count_dict[i] == 0:
        count +=1

print('There are {} zero annotationsin total {} images.'.format(count, len(data['images'])))

# cat count
# import pdb; pdb.set_trace()
cat_num = 0
cat_list= []
for anno_info in data['annotations']:
    if anno_info['category_id'] not in cat_list:
        cat_list.append(anno_info['category_id'])

print('there are {} category in the training set'.format(len(cat_list)))
# import pdb; pdb.set_trace()