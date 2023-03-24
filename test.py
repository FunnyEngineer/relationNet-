import torch
import torchvision
import argparse
import torchvision.transforms as transforms
from model.BVRRetina import BVRRetina
from coco_utils import _coco_remove_images_without_annotations
from itertools import compress
import json
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
from eval import evaluate

def collate_fn(batch):
    return tuple(zip(*batch))

def tuneTargetformat(targets, device):
    total = []
    for target in targets:
        w = 480
        h = 640
        new = {}
        boxes = [obj["bbox"] for obj in target]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        new['boxes'] = boxes[keep].to(device)

        classes = [t['category_id'] for t in target ]
        classes = torch.tensor(classes, dtype=torch.long)
        new['labels'] = classes[keep].to(device)

        total.append(new)
    return total

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        # transforms.Resize((480, 640)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Pad(32),
    ])

    totensor = transforms.ToTensor()

    valset = torchvision.datasets.CocoDetection('./datasets/coco/val2017',
        './datasets/coco/annotations/instances_val2017.json', transform = train_transform)
    valset = _coco_remove_images_without_annotations(valset)
    valloader = torch.utils.data.DataLoader(valset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn)

    # model
    if args.model == 'Retina':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        model.load_state_dict(torch.load('./model_zoo/Retina.pkl'))
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif args.model == 'BVR':
        model = BVRRetina()
    model = model.to(device)

    # optimizetion
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=args.lr,
    #                             momentum=0.9, weight_decay=0.0001)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,11], gamma=0.1)

    # print_every_step = args.print_every_step
    # model.train()
    # # training phase
    # for epoch in range(1, args.epochs + 1):

    #     model.train()
    #     for bat_idx, (images, targets) in enumerate(valloader):
            
    #         targets = tuneTargetformat(targets, device)
    #         images = torch.stack(images).to(device)
    #         mask = []
    #         new_tar = []
    #         for tar in targets:
    #             mask.append(len(tar['labels']) != 0)
    #             if len(tar['labels']) != 0:
    #                 new_tar.append(tar)

    #         targets = new_tar
    #         images = images[mask]
    #         optimizer.zero_grad()

    #         output = model(images,targets)

    #         loss = output['classification'] + output['bbox_regression'] \
    #         + output['centerKeyLoss'] + output['centerOffsetLoss'] \
    #         + output['cornerKeyLoss'] + output['cornerOffsetLoss']
    #         loss.backward()
    #         optimizer.step()

    #         if bat_idx % print_every_step == 0:
    #             # for print
    #             sep_loss = []
    #             for sep in output.values():
    #                 sep_loss.append(sep.item())

    #             print('Epoch {} [Step {} / {} ({:.2f}%)]: Total loss: {:.3f} Classification loss: {:.3f} bbox_regression loss: {:.3f} centerKeyLoss loss: {:.3f} centerOffsetLoss loss: {:.3f} cornerKeyLoss loss: {:.3f} cornerOffsetLoss loss: {:.3f}'.format(
    #                 epoch, bat_idx, len(valloader), bat_idx / len(valloader) * 100.0, loss.item(),
    #                 sep_loss[0], sep_loss[1], sep_loss[2] , sep_loss[3], sep_loss[4], sep_loss[5]
    #             ), flush = True)

    #     # lr step
    #     lr_scheduler.step()
    #     print('Epoch {} [Step {} / {} ({:.2f}%)]: Total loss: {:.6f}'.format(
    #                 epoch, bat_idx, len(trainloader), bat_idx / len(trainloader) * 100.0, loss.item()
    #             ), flush = True)

        # testing phase
    model.eval()
    total_json = []
    for bat_idx, (images, targets) in enumerate(valloader):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for img_index, per_iamge in enumerate(outputs):
            img_id = targets[img_index][0]['image_id']
            # print(per_iamge)
            box_list = per_iamge['boxes']
            labels_list = per_iamge['labels']
            score_list = per_iamge['scores']
            for obj_index in range(len(box_list)):
                sin = {}
                sin['image_id'] = img_id
                sin['category_id'] = labels_list[obj_index].item()
                sin['bbox'] = box_list[obj_index].tolist()
                sin['score'] = score_list[obj_index].item()
                total_json.append(sin)
    with open('./output/test_result_retina.json', 'w') as outputfile:
        json.dump(total_json, outputfile)
    # evaluate(model, valloader, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="choose which model", choices = ['Retina', 'BVR']
                    , default = 'BVR', type=str)
    parser.add_argument("--epochs", help="how many epochs"
                    , default = 12, type=int)
    parser.add_argument("--save_path", help="save model path"
                    , default = './model_zoo/', type=str)
    parser.add_argument("--lr", help="learning rate"
                    , default = 0.02, type=float)
    parser.add_argument("--print_every_step", help="how many steps to print loss"
                    , default = 1, type=int)
    args = parser.parse_args()
    main(args)
