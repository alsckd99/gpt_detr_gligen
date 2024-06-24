from huggingface_hub.fastai_utils import PYPROJECT_TEMPLATE
import torch
import sys
import argparse
from omegaconf import OmegaConf
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt
from ldm.util import default_device
import math
import requests
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import math
from contextlib import nullcontext
from torchvision.models import resnet50
import torchvision.transforms as T
import time
import openai
import ipywidgets as widgets
from IPython.display import display, clear_output
from dataset.generate_txt_dataset import generate_instruction
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

from typing import Optional

from huggingface_hub import hf_hub_download
hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")

arg_bool = lambda x: x.lower() == 'true'
device = default_device()

print(f"GLIGEN uses {device.upper()} device.")
if device == "cpu":
    print("It will be sloooow. Consider using GPU support with CUDA or (in case of M1/M2 Apple Silicon) MPS.")
elif device == "mps":
    print("The fastest you can get on M1/2 Apple Silicon. Yet, still many opimizations are switched off and it will is much slower than CUDA.")

def parse_option():
    parser = argparse.ArgumentParser('GLIGen Demo', add_help=False)
    parser.add_argument("--folder", type=str,  default="create_samples", help="path to OUTPUT")
    parser.add_argument("--official_ckpt", type=str,  default='ckpts/sd-v1-4.ckpt', help="")
    parser.add_argument("--guidance_scale", type=float,  default=5, help="")
    parser.add_argument("--alpha_scale", type=float,  default=1, help="scale tanh(alpha). If 0, the behaviour is same as original model")
    parser.add_argument("--load-text-box-generation", type=arg_bool, default=False, help="Load text-box generation pipeline.")
    parser.add_argument("--load-text-box-inpainting", type=arg_bool, default=True, help="Load text-box inpainting pipeline.")
    parser.add_argument("--load-text-image-box-generation", type=arg_bool, default=False, help="Load text-image-box generation pipeline.")
    args = parser.parse_args()
    return args
args = parse_option()


def load_from_hf(repo_id, filename='diffusion_pytorch_model.bin'):
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return torch.load(cache_file, map_location='cpu')

def load_ckpt_config_from_hf(modality):
    ckpt = load_from_hf(f'gligen/{modality}')
    config = load_from_hf('gligen/demo_config_legacy', filename=f'{modality}.pth')
    return ckpt, config


if args.load_text_box_generation:
    pretrained_ckpt_gligen, config = load_ckpt_config_from_hf('gligen-generation-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False
    config.model['params']['is_style'] = False
    loaded_model_list = load_ckpt(config, pretrained_ckpt_gligen) 


if args.load_text_box_inpainting:
    pretrained_ckpt_gligen_inpaint, config = load_ckpt_config_from_hf('gligen-inpainting-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = True 
    config.model['params']['is_style'] = False
    loaded_model_list_inpaint = load_ckpt(config, pretrained_ckpt_gligen_inpaint)


if args.load_text_image_box_generation:
    pretrained_ckpt_gligen_style, config = load_ckpt_config_from_hf('gligen-generation-text-image-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False 
    config.model['params']['is_style'] = True
    loaded_model_list_style = load_ckpt(config, pretrained_ckpt_gligen_style)


def load_clip_model():
    from transformers import CLIPProcessor, CLIPModel
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).to(device)
    processor = CLIPProcessor.from_pretrained(version)

    return {
        'version': version,
        'model': model,
        'processor': processor,
    }

clip_model = load_clip_model()

'''
inference model
'''
def calculate_fid(real_features, generated_features):

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    diff = mu1 - mu2
    try:
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except ValueError:
        covmean = 0  # 또는 적절한 오류 처리 또는 대체 값

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# 이미지 특징 추출 함수
def extract_features(images, model):
    transform_image = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(299),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_image = [transform_image(image) for image in images]  # transform the single image
    images_tensor = torch.stack(transformed_image)
    with torch.no_grad():
        features = model(images_tensor)
    return features.numpy()

@torch.no_grad()
def inference(task,prompt, language_instruction,img_path, grounding_instruction, inpainting_boxes_nodrop, image,
              alpha_sample, guidance_scale, batch_size,
              fix_seed, rand_seed, actual_mask, style_image,
              *args, **kwargs):
    grounding_instruction = json.loads(grounding_instruction)
    phrase_list, location_list = [], []
    for k, v  in grounding_instruction.items():
        phrase_list.append(k)
        location_list.append(v)
    placeholder_image = Image.open(img_path).convert("RGB")    
    image_list = [placeholder_image] * len(phrase_list) # placeholder input for visual prompt, which is disabled
    batch_size = int(batch_size)
    if not 1 <= batch_size <= 4:
        batch_size = 2
    if style_image == False:
        has_text_mask = 1 
        has_image_mask = 0 # then we hack above 'image_list' 
    else:
        valid_phrase_len = len(phrase_list)

        phrase_list += ['placeholder']
        has_text_mask = [1]*valid_phrase_len + [0]

        image_list = [placeholder_image]*valid_phrase_len + [style_image]
        has_image_mask = [0]*valid_phrase_len + [1]
        
        location_list += [ [0.0, 0.0, 1, 0.01]  ] # style image grounding location
    if task == 'Grounded Inpainting':
        alpha_sample = 1.0
    instruction = dict(
        prompt = prompt,
        phrases = phrase_list,
        images = image_list,
        locations = location_list,
        alpha_type = [alpha_sample, 0, 1.0 - alpha_sample], 
        has_text_mask = has_text_mask,
        has_image_mask = has_image_mask,
        save_folder_name = language_instruction,
        guidance_scale = guidance_scale,
        batch_size = batch_size,
        fix_seed = bool(fix_seed),
        rand_seed = int(rand_seed),
        actual_mask = actual_mask,
        inpainting_boxes_nodrop = inpainting_boxes_nodrop,
    )
    # float16 autocasting only CUDA device
    with torch.autocast(device_type='cuda', dtype=torch.float16) if device == "cuda" else nullcontext():
        if task == 'Grounded Inpainting':
            assert image is not None
            instruction['input_image'] = image.convert("RGB")
            return grounded_generation_box(loaded_model_list_inpaint, instruction, *args, **kwargs)

def auto_append_grounding(prompt, grounding_texts):
    for grounding_text in grounding_texts:
        if grounding_text not in prompt and grounding_text != 'auto':
            prompt += "; " + grounding_text
    return prompt


def generate(task, prompt,language_instruction, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state,img_path):
    if 'boxes' not in state:
        state['boxes'] = []
    
    boxes = state['boxes']
    assert len(boxes) == len(grounding_texts)
    W,H = sketch_pad['image'].size
    boxes = np.asarray(boxes) 
    boxes[:, 0] /= W 
    boxes[:, 1] /= H 
    boxes[:, 2] /= W  
    boxes[:, 3] /= H 
    boxes = boxes.tolist()
    grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes)})
    image = None
    actual_mask = None
    if task == 'Grounded Inpainting':
        image = state.get('original_image', sketch_pad['image']).copy()

        if use_actual_mask:
            actual_mask = sketch_pad['mask'].copy()
            if actual_mask.ndim == 3:
                actual_mask = actual_mask[..., 0]
            actual_mask = torch.from_numpy(actual_mask == 0).float()

        if state.get('inpaint_hw', None):
            boxes = np.asarray(boxes) * 0.9 + 0.05
            boxes = boxes.tolist()
            grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes) if obj != 'auto'})
    
    if append_grounding:
        prompt = auto_append_grounding(prompt, grounding_texts)
    sample = inference(
        task, prompt,language_instruction,img_path, grounding_instruction, boxes, image,
        alpha_sample, guidance_scale, batch_size,
        fix_seed, rand_seed, actual_mask, style_cond_image, clip_model=clip_model
    )

    return sample

torch.set_grad_enabled(False);
img_path="/content/drive/MyDrive/123.jpg"
transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()
CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
def plot_results(pil_img, prob, boxes,filter_labels=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    label_list=[]
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        if filter_labels==None:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
        
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
            label_list.append(CLASSES[cl])
        else:
          if CLASSES[cl] in filter_labels:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    return label_list

def resize_bbox(bbox, orig_dim, new_dim):
    x, y, w, h = bbox
    orig_width, orig_height = orig_dim
    new_width, new_height = new_dim
    
    x_min = x * (new_width / orig_width)
    y_min = y * (new_height / orig_height)
    x_max = (x + w) * (new_width / orig_width)
    y_max = (y + h) * (new_height / orig_height)
    
    return x_min, y_min, x_max, y_max

def main():
    torch.cuda.empty_cache()
    state={}
    state['boxes']=[]
    state['mask']=[]
    sketch_pad = {}
    sketch_pad['image']=[]
    sketch_pad['mask']=[]
    prompt=None

    im=Image.open(img_path)
    img = transform(im).unsqueeze(0)
    resize_transform_image=im.resize((512,512))
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], resize_transform_image.size)
    label_list=plot_results(resize_transform_image, probas[keep], bboxes_scaled)
    
    
    sketch_pad['image'] = resize_transform_image
    sketch_pad['mask'] = ""

    task = "Grounded Inpainting"
    #language_instruction = input("Language instruction: ")
    #select_instruction = input("Select instruction (Separated by semicolon): ")
    #grounding_instruction = input("Grounding instruction (Separated by semicolon): ")
    language_instruction="Make a cartoon-style duck character."
    select_instruction="cell phone"
    grounding_instruction="character"
    alpha_sample = 0.3
    guidance_scale = 7.5
    batch_size = 1
    append_grounding = True
    use_actual_mask = False

    fix_seed = True
    rand_seed = 0

    style_cond_image = False
    openai_model="gpt-3.5-turbo-instruct"

    print(f"Language Instruction: {language_instruction}, Select Instruction: {select_instruction} Grounding Instruction: {grounding_instruction}")
    select_texts = [x.strip() for x in select_instruction.split(';')]
    grounding_texts = [x.strip() for x in grounding_instruction.split(';')]
    for label, bboxes in zip(label_list,bboxes_scaled):
      for select_text in select_texts:
        if select_text==label:
          bboxes = bboxes.tolist()
          state['boxes'].append(bboxes)
    prompt_list= generate_instruction(openai_model, language_instruction)
    prompt=', '.join(prompt_list[1:])
    
    if prompt==None:
      prompt=language_instruction
    else:
      print("prompt : "+ prompt)
    generate(task, prompt,language_instruction, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state)
    
    # torch.cuda.empty_cache()
    # sketch_pad = {}
    # state={}
    # sketch_pad['mask']=[]
    # sketch_pad['mask'] = ""
    # resize_transform_image_list=[]
    # sample_list=[]
    # state['mask']=[]
    
    
    
    # sketch_pad['image']=[]
    # task = "Grounded Inpainting"
    # alpha_sample = 0.3
    # guidance_scale = 7.5
    # batch_size = 1
    # append_grounding = True
    # use_actual_mask = False

    # fix_seed = True
    # rand_seed = 0

    # style_cond_image = False
    # openai_model="gpt-3.5-turbo-instruct"


    # from pycocotools.coco import COCO
    # import os
    # import random
    # from matplotlib.patches import Rectangle

    # ann_dir = '/content/drive/MyDrive/annotations/instances_val2017.json'
    # cap_dir = '/content/drive/MyDrive/annotations/captions_val2017.json'
    # coco_instances=COCO(ann_dir)
    # coco_captions = COCO(cap_dir)
    # data_dir = '/content/drive/MyDrive/val2017'
    # img_dir = os.path.join(data_dir, 'val2017')
    # img_ids=coco_instances.getImgIds()
    # img_dicts=coco_instances.loadImgs(img_ids)
    # for index,img_inform in enumerate(img_dicts):
    #   state['boxes']=[]
    #   grounding_texts=[]
    #   img_path = os.path.join(img_dir, img_inform['file_name'])
    #   image = Image.open(img_path)
    #   resize_transform_image=image.resize((512,512))
      
    #   sketch_pad['image'] = resize_transform_image
    #   annIds = coco_instances.getAnnIds(imgIds=img_inform['id'], iscrowd=None)
    #   anns = coco_instances.loadAnns(annIds)
      
    #   if anns: 
    #     random_ann = random.choice(anns)
    #   else:
    #     continue
    #   bbox = random_ann['bbox']
    #   resize_box=resize_bbox(bbox, (img_inform['width'],img_inform['height']), (512,512))
    #   state['boxes'].append(resize_box)
    #   rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    #   cat_id = random_ann['category_id']
    #   cat_name = coco_instances.loadCats(cat_id)[0]['name']
    #   grounding_texts.append(cat_name)
    #   cap_ids = coco_captions.getAnnIds(imgIds=img_inform['id'])
    #   caps = coco_captions.loadAnns(cap_ids)
    #   language_instruction=f"Make a {grounding_texts}."

    #   prompt_list= generate_instruction(openai_model, language_instruction)
    #   prompt=', '.join(prompt_list[1:])
    
    #   prompt=language_instruction 
    #   sample=generate(task, prompt,language_instruction, grounding_texts, sketch_pad,
    #             alpha_sample, guidance_scale, batch_size,
    #             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
    #             state,img_path)
    #   resize_transform_image_list.append(resize_transform_image)
    #   sample_list.append(sample)
    #   print(f"{index} : {language_instruction}")
    #   if index==100:
    #       break
    # model = inception_v3(pretrained=True, transform_input=False)
    # model.fc = torch.nn.Identity()
    # model.eval()
    # real_features = extract_features(resize_transform_image_list, model)
    # generated_features = extract_features(sample_list, model)

    # fid_score = calculate_fid(real_features, generated_features)
    # print(f'FID: {fid_score}')  

if __name__ == "__main__":
    main()    



