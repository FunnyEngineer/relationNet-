import torch
from torch.utils.data import Subset
import argparse
import torchvision
import torchvision.transforms as transforms
from model.BVRRetina import BVRRetina
from coco_utils import _coco_remove_images_without_annotations
from itertools import compress
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
        transforms.Resize((1333, 800)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Pad(32),
    ])

    totensor = transforms.ToTensor()

    trainset = torchvision.datasets.CocoDetection('./datasets/coco/train2017',
        './datasets/coco/annotations/instances_train2017.json', transform = train_transform)
    trainset = _coco_remove_images_without_annotations(trainset)
    trainset = Subset(trainset, list(range(80000)))
    valset = Subset(trainset, list(range(80000, 115000)))
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn)
    # valset = torchvision.datasets.CocoDetection('./datasets/coco/val2017',
    #     './datasets/coco/annotations/instances_val2017.json', transform = train_transform)
    valloader = torch.utils.data.DataLoader(valset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn)
    if args.model == 'Retina':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False)
    elif args.model == 'BVR':
        model = BVRRetina()
    model = model.to(device)

    # optimizetion
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,11], gamma=0.1)

    #config
    model.train()
    for epoch in range(1, args.epochs + 1):
        # training phase
        model.train()
        for bat_idx, (images, targets) in enumerate(trainloader):

            targets = tuneTargetformat(targets, device)
            images = torch.stack(images).to(device)
            mask = []
            new_tar = []
            for tar in targets:
                mask.append(len(tar['labels']) != 0)
                if len(tar['labels']) != 0:
                    new_tar.append(tar)

            targets = new_tar
            images = images[mask]
            optimizer.zero_grad()
            output = model(images,targets)
            if args.model == 'Retina':
                loss = output['classification'] + output['bbox_regression']
            else:
                loss = output['classification'] + output['bbox_regression'] \
                + output['centerKeyLoss'] + output['centerOffsetLoss'] \
                + output['cornerKeyLoss'] + output['cornerOffsetLoss']

            loss.backward()
            optimizer.step()

            if bat_idx % args.print_every_step == 0:
                if args.model == 'Retina':
                    # for print
                    sep_loss = []
                    for sep in output.values():
                        sep_loss.append(sep.item())

                    print('Epoch {} [Step {} / {} ({:.2f}%)]: Current loss: {:.3f} Classification loss: {:.3f} bbox_regression loss: {:.3f} '.format(
                        epoch, bat_idx, len(trainloader), bat_idx / len(trainloader) * 100.0, loss.item(),
                        sep_loss[0], sep_loss[1]
                    ), flush = True)
                else:
                    # for print
                    sep_loss = []
                    for sep in output.values():
                        sep_loss.append(sep.item())

                    print('Epoch {} [Step {} / {} ({:.2f}%)]: Current loss: {:.3f} Classification loss: {:.3f} bbox_regression loss: {:.3f} centerKeyLoss loss: {:.3f} centerOffsetLoss loss: {:.3f} cornerKeyLoss loss: {:.3f} cornerOffsetLoss loss: {:.3f}'.format(
                        epoch, bat_idx, len(trainloader), bat_idx / len(trainloader) * 100.0, loss.item(),
                        sep_loss[0], sep_loss[1], sep_loss[2] , sep_loss[3], sep_loss[4], sep_loss[5]
                    ), flush = True)
        # lr step
        lr_scheduler.step()
        if args.model == 'Retina':
            print('Epoch {} [Step {} / {} ({:.2f}%)]: Current loss: {:.3f} Classification loss: {:.3f} bbox_regression loss: {:.3f} '.format(
                epoch, bat_idx, len(trainloader), bat_idx / len(trainloader) * 100.0, loss.item(),
                sep_loss[0], sep_loss[1]
            ), flush = True)
        else:
            print('Epoch {} [Step {} / {} ({:.2f}%)]: Current loss: {:.3f} Classification loss: {:.3f} bbox_regression loss: {:.3f} centerKeyLoss loss: {:.3f} centerOffsetLoss loss: {:.3f} cornerKeyLoss loss: {:.3f} cornerOffsetLoss loss: {:.3f}'.format(
                epoch, bat_idx, len(trainloader), bat_idx / len(trainloader) * 100.0, loss.item(),
                sep_loss[0], sep_loss[1], sep_loss[2] , sep_loss[3], sep_loss[4], sep_loss[5]
            ), flush = True)

        # testing phase
        model.eval()
        with torch.no_grad():
            evaluate(model, valloader, device=device)
        #save model
        torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="choose which model", choices = ['Retina', 'BVR']
                    , default = 'BVR', type=str)
    parser.add_argument("--epochs", help="how many epochs"
                    , default = 12, type=int)
    parser.add_argument("--save_path", help="save model path"
                    , default = './model_zoo/bvrRetina.pkl', type=str)
    parser.add_argument("--lr", help="learning rate"
                    , default = 5e-4, type=float)
    parser.add_argument("--print_every_step", help="learning rate"
                    , default = 1, type=int)
    args = parser.parse_args()
    main(args)
