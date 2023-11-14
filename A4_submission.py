import numpy as np
import torch
import os
import shutil
import pathlib
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand
        
def generate_pred_bboxes(result):
    # print(result)
    pred_box = np.empty((2, 4), dtype=np.float64)
    pred_clas = np.empty((2), dtype=np.int32)
    boxes = result[0]['boxes'].detach().round().cpu().numpy()
    labels = result[0]['labels'].detach().cpu().numpy().astype(np.int32)
    if len(boxes) >= 2:
        labels = result[0]['labels'].detach().cpu().numpy().astype(np.int32)
        if labels[0] < labels[1]:
            pred_box[0] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_box[1] = np.array([boxes[1][1], boxes[1][0], boxes[1][3], boxes[1][2]])
            pred_clas = np.array([labels[0]-1, labels[1]-1], dtype=np.int32)
        else:
            pred_box[0] = np.array([boxes[1][1], boxes[1][0], boxes[1][3], boxes[1][2]])
            pred_box[1] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_clas = np.array([labels[1], labels[0]], dtype=np.int32)
    else:
        if len(boxes) > 0:
            pred_box = np.empty((2, 4), dtype=np.float64)
            pred_box[0] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_box[1] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_clas = np.array([0, labels[0]], dtype=np.int32)
            
    return pred_box, pred_clas.reshape(2)




class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

# A Nano backbone.
class NanoBackbone(nn.Module):
    def __init__(self, initialize_weights=True, num_classes=1000):
        super(NanoBackbone, self).__init__()

        self.num_classes = num_classes
        self.features = self._create_conv_layers()

        if initialize_weights:
            # Random initialization of the weights
            # just like the original paper.
            self._initialize_weights()

    def _create_conv_layers(self):
        conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the backbone features.
    backbone = NanoBackbone(num_classes=11).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    backbone.out_channels = 256

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    representation_size = 128

    # Box head.
    box_head = TwoMLPHead(
        in_channels=backbone.out_channels * roi_pooler.output_size[0] ** 2,
        representation_size=representation_size
    )

    # Box predictor.
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None, # Num classes shoule be None when `box_predictor` is provided.
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor
    )
    return model

def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]
    

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4.), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    model = UNET(3, 11).to(device).eval()
    dir_path = (Path(__file__).parent.resolve())
    goal_dir = os.path.join(dir_path, "model.pth")
    checkpoint = torch.load(goal_dir)
    model.load_state_dict(checkpoint["model_state_dict"])

    trans_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    images = np.array(images, dtype=np.float32)
    images = torch.from_numpy(images)
    
    FasterRcnnModel = create_model(num_classes=11, pretrained=True, coco_model=False).to(device)
    FasterRcnnModel.eval()
    checkpoint2 = torch.load(faster_rcnn_MODEL_PATH)
    FasterRcnnModel.load_state_dict(checkpoint2['model_state_dict'])

    for i in range(N):
        # print(images[i].shape)
        image = images[i].reshape(64, 64, 3)
        image = np.array(image, dtype=np.float32)
        image = trans_image(image)
        image2 = [image.clone().to(device)]
        image = image.reshape(1, 3, 64, 64)
        image = image.to(device)
        pred_seg[i] = model(image).argmax(dim=1).flatten().cpu()
        output = faster_rcnn_model(image2)
        pred_bboxes[i], pred_class[i] = generate_pred_bboxes(output)


    # add your code here to fill in pred_class and pred_bboxes

    return pred_class, pred_bboxes, pred_seg