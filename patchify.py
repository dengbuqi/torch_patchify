import torch.nn.functional as F

class ImagePatchify:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def patchify(self, img):
        batch_size, channels, height, width = img.size()
        ph, pw = self.patch_size[0], patch_size[1]

        # Pad the input image to ensure its dimensions are divisible by patch size
        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Extract patches from the padded image
        patches = padded_img.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(batch_size, channels, -1, ph, pw).permute(0,2,1,3,4)
        # patches -> B, C, No.of patches, ph, pw
        return patches

    def unpatchify(self, patches, original_size):
        B, C, H, W = original_size
        ph, pw = self.patch_size[0], self.patch_size[1]
        sH, sW = H//ph, W//pw
        # Reshape patches to match the original image size
        reconstructed_img = patches.view(B, sH, sW, C, ph, pw).permute(0,3,1,4,2,5).contiguous().view(B, C, H, W)
        print(reconstructed_img.shape)
        return reconstructed_img
        # print(patches.shape)
        # # Reconstruct the image from patches
        # reconstructed_img = patches.permute(0, 1, 3, 4, 2).contiguous()
        # print(reconstructed_img.shape)
        # reconstructed_img = reconstructed_img.view(B, C, H, W)

        # return reconstructed_img
    
if __name__=='__main__':
    from torchvision.io import read_image
    from torchvision.utils import save_image

    # image = torch.randn(2, 3, 400, 600)
    image = read_image('/LLE_DEFOG/data/LOL/LOL-v2/Real_captured/Test/Normal/normal00690.png').unsqueeze(0)

    p_factor = 8
    b, c, h, w = image.shape
    patch_size = [int(h/p_factor),int(w/p_factor)]

    p = ImagePatchify(patch_size)

    patches = p.patchify(image)
    print('patches', patches.shape)
    out_patchs = patches[0]
    save_image(out_patchs/255, 'patchs.png', nrow=p_factor)
    reconstructed_image = p.unpatchify(patches, image.shape)
    print('reconstructed_image', reconstructed_image.shape)
    save_image(reconstructed_image[0]/255, 'rec.png')

