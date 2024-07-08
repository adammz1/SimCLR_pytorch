import os
import shutil
import torch
import yaml
from torchvision import transforms


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_weights(base_model, new_model, str_to_replace, replacement_str):
    base_model = {key.replace(str_to_replace, replacement_str): value for key, value in base_model.items()}
    for name, param in base_model.items():
        if name in new_model.state_dict():
            new_model.state_dict()[name].copy_(param)
    return new_model


class RandomCropWithCoords:
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        self.coords = None
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = transforms.functional.get_image_size(img)
        th, tw = output_size, output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def forward(self, img):
        if self.padding is not None:
            img = transforms.functional.pad(img, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = transforms.functional.pad(img, [self.size[1] - img.size[0], 0], self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = transforms.functional.pad(img, [0, self.size[0] - img.size[1]], self.fill, self.padding_mode)

        if self.coords is None:
            i, j, h, w = self.get_params(img, self.size)
            self.coords = (i, j, h, w)
            return transforms.functional.crop(img, i, j, h, w)
        else:
            i, j, h, w = self.coords
            return transforms.functional.crop(img, i, j, h, w)

    def get_crop_coords(self):
        return self.coords

    def __call__(self, img):
        return self.forward(img)
