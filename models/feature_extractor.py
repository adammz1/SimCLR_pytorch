import torch
import copy
import timm
import torchvision.models as models
from models.Unfold.unfold import UnfoldNd


def load(name, pretrained):
    args = _BACKBONES[name][1]
    if 'pretrained' in args:
        args['pretrained'] = pretrained
    else:
        args['weights'] = None
    model = _BACKBONES[name][0](**args)
    return model


class PatchifyBackBone(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device):
        super(PatchifyBackBone, self).__init__()
        self.backbone = NetworkFeatureAggregator(backbone=backbone,
                                                 layers_to_extract_from=layers_to_extract_from,
                                                 device=device)

    def patchify(self, feature_map, superpixel, patch_size, step_size, apply_patch_padding=False):
        nx = int(patch_size[0] / superpixel[0])
        ny = int(patch_size[1] / superpixel[1])
        step_x = int(step_size[0] / superpixel[0])
        step_y = int(step_size[1] / superpixel[1])

        # Unfold feature maps
        channels = feature_map.shape[0]
        padding_y = self.get_padding(feature_map.shape[2], step_y, ny) if apply_patch_padding else 0
        padding_x = self.get_padding(feature_map.shape[3], step_x, nx) if apply_patch_padding else 0
        b = feature_map.shape[1]
        unfolder = UnfoldNd(kernel_size=(ny, nx), dilation=1, padding=(padding_y, padding_x), stride=(step_y, step_x))
        unfolded_features = unfolder(feature_map)
        unfolded_features = unfolded_features.reshape(b, channels, ny, nx, -1)
        unfolded_features = unfolded_features.permute(1, 0, 4, 2, 3)
        return unfolded_features

    def forward(self, x):
        x = self.backbone(x)
        x = self.patchify(x['layer3'], patch_size=[128, 128], step_size=[64, 64], superpixel=[16, 16])
        x = torch.mean(x.reshape((x.shape[0], x.shape[1], x.shape[2], -1)), dim=3).permute(0, 2, 1).flatten(0, 1)
        return x


class NetworkFeatureAggregator(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))
        self.to(self.device)
        self.eval()

    def forward(self, images):
        self.outputs.clear()
        try:
            last_output = self.backbone(images)
        except LastLayerToExtractReachedException:
            pass
        if len(self.layers_to_extract_from) == 0:
            return last_output
        else:
            return self.outputs

    def feature_dimensions(self, input_shape):
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


_BACKBONES = {
    "alexnet": (models.alexnet, dict(pretrained=True)),
    "resnet18": (models.resnet18, dict(pretrained=False)),
    "resnet50": (models.resnet50, dict(weights="IMAGENET1K_V2")),
    "resnet101": (models.resnet101, dict(pretrained=True)),
    "resnext101": (models.resnext101_32x8d, dict(pretrained=True)),
    "resnext50": (models.resnext50_32x4d, dict(weights='IMAGENET1K_V2')),
    "resnet200": (timm.create_model, dict(model_name="resnet200", pretrained=True)),
    "resnest50": (timm.create_model, dict(model_name="resnest50d_4s2x40d", pretrained=True)),
    "resnetv2_50_bit": (timm.create_model, dict(model_name="resnetv2_50x3_bitm", pretrained=True)),
    "resnetv2_50_21k": (timm.create_model, dict(model_name="resnetv2_50x3_bitm_in21k", pretrained=True)),
    "resnetv2_101_bit": (timm.create_model, dict(model_name="resnetv2_101x3_bitm", pretrained=True)),
    "resnetv2_101_21k": (timm.create_model, dict(model_name="resnetv2_101x3_bitm_in21k", pretrained=True)),
    "resnetv2_152_bit": (timm.create_model, dict(model_name="resnetv2_152x4_bitm", pretrained=True)),
    "resnetv2_152_21k": (timm.create_model, dict(model_name="resnetv2_152x4_bitm_in21k", pretrained=True)),
    "resnetv2_152_384": (timm.create_model, dict(model_name="resnetv2_152x2_bit_teacher_384", pretrained=True)),
    "resnetv2_101": (timm.create_model, dict(model_name="resnetv2_101", pretrained=True)),
    "vgg11": (models.vgg11, dict(pretrained=True)),
    "vgg19": (models.vgg19, dict(pretrained=True)),
    "vgg19_bn": (models.vgg19_bn, dict(pretrained=True)),
    "wideresnet50": (models.wide_resnet50_2, dict(weights="IMAGENET1K_V1")),
    "wideresnet101": (models.wide_resnet101_2, dict(pretrained=True)),
    "mnasnet_100": (timm.create_model, dict(model_name="mnasnet_100", pretrained=True)),
    "mnasnet_a1": (timm.create_model, dict(model_name="mnasnet_a1", pretrained=True)),
    "mnasnet_b1": (timm.create_model, dict(model_name="mnasnet_b1", pretrained=True)),
    "densenet121": (timm.create_model, dict(model_name="densenet121", pretrained=True)),
    "densenet201": (timm.create_model, dict(model_name="densenet201", pretrained=True)),
    "inception_v4": (timm.create_model, dict(model_name="inception_v4", pretrained=True)),
    "vit_small": (timm.create_model, dict(model_name="vit_small_patch16_224", pretrained=True)),
    "vit_base": (timm.create_model, dict(model_name="vit_base_patch16_224", pretrained=True)),
    "vit_large": (timm.create_model, dict(model_name="vit_large_patch16_224", pretrained=True)),
    "vit_r50": (timm.create_model, dict(model_name="vit_large_r50_s32_224", pretrained=True)),
    "vit_deit_base": (timm.create_model, dict(model_name="deit_base_patch16_224", pretrained=True)),
    "vit_deit_distilled": (timm.create_model, dict(model_name="deit_base_distilled_patch16_224", pretrained=True)),
    "vit_swin_base": (timm.create_model, dict(model_name="swin_base_patch4_window7_224", pretrained=True)),
    "vit_swin_large": (timm.create_model, dict(model_name="swin_large_patch4_window7_224", pretrained=True)),
    "vit_h_14": (models.vit_h_14, dict(weights="IMAGENET1K_SWAG_E2E_V1")),
    "efficientnet_b7": (timm.create_model, dict(model_name="tf_efficientnet_b7", pretrained=True)),
    "efficientnet_b5": (timm.create_model, dict(model_name="tf_efficientnet_b5", pretrained=True)),
    "efficientnet_b3": (timm.create_model, dict(model_name="tf_efficientnet_b3", pretrained=True)),
    "efficientnet_b1": (timm.create_model, dict(model_name="tf_efficientnet_b1", pretrained=True)),
    "efficientnetv2_m": (timm.create_model, dict(model_name="tf_efficientnetv2_m", pretrained=True)),
    "efficientnetv2_l": (timm.create_model, dict(model_name="tf_efficientnetv2_l", pretrained=True)),
    "efficientnetv2_s": (timm.create_model, dict(model_name="tf_efficientnetv2_s", pretrained=True)),
    "efficientnet_b3a": (timm.create_model, dict(model_name="efficientnet_b3a", pretrained=True)),
    "ssl_resnet50": (timm.create_model, dict(model_name="resnet50.fb_ssl_yfcc100m_ft_in1k", pretrained=True))
}