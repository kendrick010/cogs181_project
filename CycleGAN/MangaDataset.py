import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MangaDataset(Dataset):

    def __init__(self, train_test_root, transforms_=None, mode='train'):
        self.color_path = train_test_root + '/' + mode + '_color'
        self.gray_path = train_test_root + '/' + mode + '_gray'

        self.color = os.listdir(self.color_path)
        self.gray = os.listdir(self.gray_path)

        self.transforms_ = transforms_

    def __getitem__(self, index):
        color_panel_path = os.path.join(self.color_path, self.color[index])
        gray_panel_path = os.path.join(self.gray_path, self.gray[index])
        
        color_panel = Image.open(color_panel_path).convert('RGB')
        gray_panel = Image.open(gray_panel_path).convert('RGB')

        color_panel = self.remove_transparency(color_panel)
        gray_panel = self.remove_transparency(gray_panel)

        if self.transforms_:
            color_panel = self.transforms_(color_panel)
            gray_panel = self.transforms_(gray_panel)

        return color_panel, gray_panel

    def __len__(self):
        return min(len(self.color), len(self.gray))
    
    @staticmethod
    def remove_transparency(image):
        background = Image.new('New_RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])

        return background