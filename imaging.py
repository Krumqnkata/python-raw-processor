"""Resolution-independent, non-destructive editing operations (float RGB)."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class LocalAdjustment:
    kind: str = 'brush'
    points: tuple[tuple[float, float], ...] = ()
    radius: float = .08
    exposure: float = .5
    shadows: float = 0.0
    saturation: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'points', tuple(tuple(p) for p in self.points))
        if self.kind not in {'brush', 'gradient'} or not self.points:
            raise ValueError('Маската трябва да е четка или градиент с точки.')
        if self.kind == 'gradient' and len(self.points) != 2:
            raise ValueError('Градиентът изисква две точки.')
        if len(self.points) > 5000 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in self.points):
            raise ValueError('Невалидни координати на маската.')
        for name, low, high in [('radius', .005, .5), ('exposure', -3, 3), ('shadows', -1, 1), ('saturation', 0, 2)]:
            v = getattr(self, name)
            if not math.isfinite(v) or not low <= v <= high:
                raise ValueError('Невалидна локална корекция.')


def tonal(rgb, shadows=0., highlights=0., whites=0., blacks=0.):
    luma = rgb @ np.array([.2126, .7152, .0722], np.float32)
    delta = (shadows * .35 * (1-luma)**3 * (1-np.exp(-luma*12)) +
             highlights * .3 * luma**3 + whites * .2 * luma**6 +
             blacks * .15 * (1-luma)**6)
    return np.clip(rgb + delta[..., None], 0, 1).astype(np.float32)


def colour(rgb, saturation=1., vibrance=0., monochrome=False):
    luma = (rgb @ np.array([.2126, .7152, .0722], np.float32))[..., None]
    chroma = rgb.max(axis=2, keepdims=True) - rgb.min(axis=2, keepdims=True)
    gain = saturation * (1 + vibrance * (1-chroma))
    return np.repeat(luma,3,axis=2).astype(np.float32) if monochrome else np.clip(luma + (rgb-luma)*gain,0,1).astype(np.float32)


def mask_weights(shape, mask):
    h, w = shape[:2]
    if mask.kind == 'gradient':
        (x0,y0),(x1,y1) = mask.points
        yy,xx = np.ogrid[:h,:w]
        dx,dy = (x1-x0)*w,(y1-y0)*h
        distance = dx*dx+dy*dy
        if distance < 1e-6:
            return np.zeros((h,w), np.float32)
        return np.clip(((xx-x0*w)*dx+(yy-y0*h)*dy)/distance,0,1).astype(np.float32)
    weights = np.zeros((h,w), np.float32)
    radius = max(1., mask.radius * min(w,h))
    # Work only inside the brush bounding boxes, not H*W*number_of_points.
    points = list(mask.points)
    dense = [points[0]]
    for a,b in zip(points, points[1:]):
        steps = max(1, math.ceil(math.hypot((b[0]-a[0])*w,(b[1]-a[1])*h)/(radius*.4)))
        dense.extend((a[0]+(b[0]-a[0])*t/steps,a[1]+(b[1]-a[1])*t/steps) for t in range(1,steps+1))
    for x,y in dense:
        cx,cy = x*(w-1),y*(h-1)
        x0,x1 = max(0,int(cx-radius)),min(w,int(cx+radius)+2)
        y0,y1 = max(0,int(cy-radius)),min(h,int(cy+radius)+2)
        yy,xx = np.ogrid[y0:y1,x0:x1]
        stamp = np.clip(1-((xx-cx)**2+(yy-cy)**2)/radius**2,0,1)**2
        np.maximum(weights[y0:y1,x0:x1],stamp,out=weights[y0:y1,x0:x1])
    return weights


def local_adjustments(rgb, masks, check=lambda: None):
    for mask in masks:
        check()
        alpha = mask_weights(rgb.shape, mask)[...,None]
        linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        linear = np.clip(linear*2**mask.exposure,0,1)
        exposed = np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055).astype(np.float32)
        adjusted = tonal(exposed,shadows=mask.shadows)
        adjusted = colour(adjusted, mask.saturation)
        rgb = rgb*(1-alpha)+adjusted*alpha
    return np.ascontiguousarray(rgb,np.float32)


def lens_correct(rgb, p, check=lambda: None):
    if not p.lens_enabled:
        return rgb
    check()
    if not any((p.lens_k1,p.lens_k2,p.vignette,p.ca_red,p.ca_blue)):
        return rgb
    h,w = rgb.shape[:2]
    # Row blocks bound temporary coordinate memory for large RAW images.
    result = np.empty_like(rgb)
    for start in range(0,h,256):
        check()
        end = min(h,start+256)
        yy,xx = np.mgrid[start:end,:w].astype(np.float32)
        nx,ny = (xx-(w-1)/2)/max(w,h)*2,(yy-(h-1)/2)/max(w,h)*2
        r2 = nx*nx+ny*ny
        scale = 1+p.lens_k1*r2+p.lens_k2*r2*r2
        for channel,ca in enumerate((p.ca_red,0,p.ca_blue)):
            mx = ((nx*scale*(1+ca))*max(w,h)/2+(w-1)/2).astype(np.float32)
            my = ((ny*scale*(1+ca))*max(w,h)/2+(h-1)/2).astype(np.float32)
            result[start:end,:,channel] = cv2.remap(rgb[:,:,channel],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        result[start:end] *= np.exp(p.vignette*r2)[...,None]
    return np.clip(result,0,1)


def geometry(rgb, p):
    if p.rotation:
        rgb = np.rot90(rgb, -p.rotation)
    if p.flip_horizontal:
        rgb = rgb[:,::-1]
    if p.flip_vertical:
        rgb = rgb[::-1]
    if p.straighten:
        h,w = rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),p.straighten,1)
        rgb = cv2.warpAffine(np.ascontiguousarray(rgb),matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    x0,y0,x1,y1 = p.crop
    h,w = rgb.shape[:2]
    left,top = min(w-1,round(x0*w)),min(h-1,round(y0*h))
    right,bottom = max(left+1,round(x1*w)),max(top+1,round(y1*h))
    return np.ascontiguousarray(rgb[top:bottom,left:right])


def resize_export(rgb, max_edge):
    h,w = rgb.shape[:2]
    if max_edge and max(h,w) > max_edge:
        scale = max_edge/max(h,w)
        rgb = cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    return rgb


def histogram(image):
    pixels = np.asarray(image.convert('RGB'))
    h,w = pixels.shape[:2]
    step = max(1,int(math.ceil(math.sqrt(h*w/500000))))
    sample = pixels[::step,::step]
    channels = [np.bincount(sample[:,:,i].ravel(),minlength=256).tolist() for i in range(3)]
    luma = np.rint(sample @ np.array([.2126,.7152,.0722])).astype(np.uint8)
    return {'rgb':channels,'luma':np.bincount(luma.ravel(),minlength=256).tolist(),
            'shadows':float((sample.min(axis=2)==0).mean()),'highlights':float((sample.max(axis=2)==255).mean())}
