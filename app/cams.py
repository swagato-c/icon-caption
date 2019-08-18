from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from skimage.transform import rescale, resize
from skimage import data, color
from matplotlib import cm
import base64 as b64

class ActGrad(object):
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers

    def __call__(self, x, cat):
        self.model.zero_grad()
        with hook_output(self.layers) as hook_a:
            with hook_output(self.layers, grad=True) as hook_g:
                preds = self.model(x)
                preds[0,int(cat)].backward()
        return hook_a.stored[0].cpu(),hook_g.stored[0][0].cpu() # returns (ch,h,w)

class GradCAM(ActGrad):
    def __init__(self, model, layers):
        super(GradCAM, self).__init__(model, layers)

    def __call__(self, x, y):
        acts, grads = ActGrad(self.model, self.layers)(x, y)
        wts = grads.mean((-2, -1))  #global avg pooling
        L = F.relu((wts[..., None, None] * acts).sum(-3))  #saliency map
        L = resize(L, (x.shape[-2], x.shape[-1]), anti_aliasing=True)
        L = (L - L.min()) / (L.max() - L.min())
        return np.uint8(L * 255)


class GradCAMpp(GradCAM):
    def __init__(self, models, layers):
        super(GradCAMpp, self).__init__(models, layers)

    def __call__(self, x, y):
        acts, grads = ActGrad(self.model, self.layers)(x, y)
        alpha_n = grads.pow(2)
        alpha_d = grads.pow(2).mul(2) + acts.mul(grads.pow(3)).sum(
            (-1, -2), keepdim=True)
        alpha_d = torch.where(alpha_d != 0.0, alpha_d, torch.ones_like(alpha_d))
        alpha = alpha_n.div(alpha_d + 1e-7)
        pgrads = F.relu(grads)
        # F.relu(score.exp()*grads) where score = m(x)[0,int(y)]
        wts = (pgrads * alpha).sum((-1, -2))
        L = F.relu((wts[..., None, None] * acts).sum(-3))
        L = resize(L, (x.shape[-2], x.shape[-1]), anti_aliasing=True)
        L = (L - L.min()) / (L.max() - L.min())
        return np.uint8(L * 255)


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.relus = [
            module[1] for module in self.model.named_modules()
            if "ReLU(inplace)" == str(module[1])
        ]
        self.hook = Hooks(self.relus, self.clamp_gradients_hook, is_forward=False)

    def clamp_gradients_hook(self, module, grad_in, grad_out):
        for grad in grad_in:
            torch.clamp_(grad, min=0.0)

    def __call__(self, image, cat):
        image.requires_grad_()
        self.model.zero_grad()
        preds = self.model(image)
        preds[0, int(cat)].backward()
        return image.grad.squeeze().cpu().numpy()

def repr_image_format(npim, format_str="PNG"):
    with BytesIO() as str_buffer:
        plt.imsave(str_buffer, npim, format=format_str)
        return str_buffer.getvalue()

def grad_to_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    gradient = gradient[..., ::-1]
    return gradient

def to_grayscale(tensor_im):
    grayscale_im = np.sum(np.abs(tensor_im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def ggcams(grad, sm):
    ggcam = (grad * sm)
    img_ggcam = grad_to_image(ggcam)
    ggcam_gray = to_grayscale(ggcam)
    img_ggcam_gray = np.squeeze(grad_to_image(ggcam_gray))
    return img_ggcam, img_ggcam_gray

def to_dataURI(img):
    return "data:image/png;base64," + b64.b64encode(repr_image_format(img)).decode('utf-8')

def get_gradcam(learn, x, y, plus=False):
    xb, _ = learn.data.one_item(x)
    xb_im = Image(learn.data.denorm(xb)[0])
    m = learn.model.eval()
    CAM = GradCAM if not plus else GradCAMpp

    sm = CAM(m, m[0])(xb, y)
    smn = sm.astype(float) / 255.0
    grad = GuidedBackprop(m)(xb, y)

    hmi = PIL.Image.fromarray(np.uint8(cm.jet(smn) * 255))
    #gcami = PIL.Image.blend(hmi,PIL.Image.fromarray(image2np(xb_im.data*255).astype(np.uint8)),0.6)
    #gcam = np.asarray(gcami)
    #hm = plt.imshow(sm, alpha=0.6,cmap='jet')
    cggcam = (smn * xb_im.data.numpy()).transpose(1, 2, 0)
    #hm = np.uint8(cm.jet(smn) * 255)
    hmi = hmi.convert("RGB")
    hm = np.asarray(hmi)
    gcam = (hm * 0.6 + image2np(xb_im.data*255) * 0.4).astype(np.uint8)
    q = [gcam,*ggcams(grad,sm),cggcam]
    return list(map(to_dataURI,q))
