import torch.nn.functional as F

class ImagePatchify:
    def __init__(self):
        pass

    def patchify(self, img, patch_size):
        batch_size, channels, height, width = img.size()
        ph, pw = patch_size[0], patch_size[1]

        # Pad the input image to ensure its dimensions are divisible by patch size
        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Extract patches from the padded image
        patches = padded_img.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(batch_size, channels, -1, ph, pw)
        # patches -> B, C, No.of patches, ph, pw
        return patches

    def unpatchify(self, patches, patch_size, original_size):
        B, C, H, W = original_size
        ph, pw = patch_size[0], patch_size[1]
        sH, sW = H//ph, W//pw
        # Reshape patches to match the original image size
        reconstructed_img = patches.view(B, C, sH, sW, ph, pw).permute(0,1,2,4,3,5).contiguous().view(B, C, H, W)
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
    image = read_image('xxx.png').unsqueeze(0)

    p_sacle_factor = 8
    b, c, h, w = image.shape
    patch_size = [int(h/p_sacle_factor),int(w/p_sacle_factor)]

    p = ImagePatchify()

    patches = p.patchify(image, patch_size)
    print('patches', patches.shape)
    out_patchs = patches[0].permute(1, 0, 2, 3)
    save_image(out_patchs/255, 'patchs.png', nrow=p_sacle_factor)
    reconstructed_image = p.unpatchify(patches, patch_size, image.shape)
    print('reconstructed_image', reconstructed_image.shape)
    save_image(reconstructed_image[0]/255, 'rec.png')

