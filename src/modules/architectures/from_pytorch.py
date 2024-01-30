import torch
import torchvision


def get_efficientnet_b0(num_classes, input_channels, img_height, img_width):
    model = torchvision.models.efficientnet_b0(num_classes=num_classes, progress=False)

    model.input_channels = input_channels
    model.img_height = img_height
    model.img_height = img_width
    model.num_classes = num_classes
    return model


def get_efficientnet_v2_s(num_classes, input_channels, img_height, img_width, dropout, stochastic_depth_prob, whether_batchnorm_layers):
    model = torchvision.models.efficientnet_v2_s(num_classes=num_classes, progress=False, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)

    model.input_channels = input_channels
    model.img_height = img_height
    model.img_height = img_width
    model.num_classes = num_classes
    return model


def get_convnext_t(num_classes, input_channels, img_height, img_width, stochastic_depth_prob):
    model = torchvision.models.convnext_tiny(num_classes=num_classes, progress=False, stochastic_depth_prob=stochastic_depth_prob)

    model.input_channels = input_channels
    model.img_height = img_height
    model.img_height = img_width
    model.num_classes = num_classes
    return model


def get_resnet(num_classes, input_channels, img_height, img_width, model_name, nonstandard_first_conv, whether_batchnorm_layers):
    match model_name:
        case 'resnet18':
            model = torchvision.models.resnet18(num_classes=num_classes, progress=False)
        case 'resnet34':
            model = torchvision.models.resnet34(num_classes=num_classes, progress=False)
        case 'resnet50':
            model = torchvision.models.resnet50(num_classes=num_classes, progress=False)
        case 'resnet101':
            model = torchvision.models.resnet101(num_classes=num_classes, progress=False)
        case 'resnet152':
            model = torchvision.models.resnet152(num_classes=num_classes, progress=False)
        case _:
            raise ValueError(f"Invalid model_name {model_name}. Expected one of: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'")
    
    if nonstandard_first_conv is True:
        model.maxpool = torch.nn.Identity()
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        
    model.input_channels = input_channels
    model.img_height = img_height
    model.img_height = img_width
    model.num_classes = num_classes
    return model