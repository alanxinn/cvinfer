"""Compare export wrappers with official Qwen/WeVisOne modules using small random weights.

These are architecture/ONNX checks, not real-checkpoint task accuracy results.
"""
import argparse
import copy
import sys,json
from pathlib import Path
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path('cache/wevisone/source/9a6bcd85404505df0d05cfb5d97512ccacf35dee'))
parser.add_argument('--output',type=Path,default=Path('cache/wevisone/validation'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.source.resolve()))
from export_wevisone import wrappers
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from wevisone_vl.feature_bridge import VisualFeatureCapture,per_sample_premerge_feature_maps,LastHiddenCapture
torch.set_num_threads(4)
torch.manual_seed(2026)
config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
    num_attention_heads=4,num_key_value_heads=2,head_dim=16,rope_parameters=dict(rope_type='default',rope_theta=10000.,mrope_section=[4,2,2],mrope_interleaved=True)),
    vision_config=dict(depth=4,hidden_size=64,intermediate_size=128,num_heads=4,patch_size=16,temporal_patch_size=2,
        spatial_merge_size=2,out_hidden_size=64,deepstack_visual_indexes=[0,1,2],num_position_embeddings=16),image_token_id=120)
model=Qwen3VLForConditionalGeneration(config).eval()
Vision,Language,Pixel=wrappers(torch)
vision=Vision(model.model.visual,4,4,2).eval()
language=Language(model).eval()
grid=torch.tensor([[1,4,4]])
pixels=torch.randn(16,3*2*16*16)
results=[]
def check(name,a,b,tol=1e-5):
    error=(a-b).abs().max().item()
    results.append(dict(name=name,max_abs=error,passed=error<=tol))
    print(name,'max_abs',error,'PASS' if error<=tol else 'FAIL',flush=True)
    if error>tol:raise AssertionError(name)
with torch.no_grad():
    with VisualFeatureCapture(model,feature_layers=(0,1,2,3)) as capture:
        ref=model.model.visual(pixels,grid_thw=grid)
    maps=per_sample_premerge_feature_maps(capture.premerge_features,(0,1,2,3),grid,2)[0]
    actual=vision(pixels)
    for name,a,b in zip(['merged']+[f'deep_{i}' for i in range(3)]+[f'map_{i}' for i in range(4)],actual,[ref.pooler_output,*ref.deepstack_features,*maps]):
        check(name,a,b)
    seq=8
    ids=torch.tensor([[1,120,120,2,3,4,5,6]])
    pos=torch.arange(seq).view(1,1,seq).expand(3,1,seq).clone()
    pos[1,0,2]+=1
    mask=(ids==120)[:,:,None].float()
    img=torch.randn(1,seq,64)*mask
    deep=torch.randn(3,1,seq,64)*mask[None]
    embeds=model.model.language_model.embed_tokens(ids)*(1-mask)+img
    with LastHiddenCapture(model) as hidden:
        reference=model.model.language_model(inputs_embeds=embeds,position_ids=pos,use_cache=True,
            visual_pos_masks=mask[:,:,0].bool(),deepstack_visual_embeds=[d[mask[:,:,0].bool()] for d in deep])
    actual=language(ids,pos,img,deep,mask,torch.zeros(4,2,2,1,16))
    check('prefill_logits',actual[0],model.lm_head(reference.last_hidden_state[:,-1]))
    check('prefill_OBJ_hidden',actual[1],hidden.hidden[:,-1])
    for i,layer in enumerate(reference.past_key_values.layers):
        check(f'prefill_cache_{i}_K',actual[2][i,0,:,1:],layer.keys[0])
        check(f'prefill_cache_{i}_V',actual[2][i,1,:,1:],layer.values[0])
    for step in range(3):
        ids=torch.tensor([[7+step]])
        pos=torch.full((3,1,1),seq+step)
        with LastHiddenCapture(model) as hidden:
            reference=model.model.language_model(input_ids=ids,position_ids=pos,past_key_values=reference.past_key_values,use_cache=True)
        actual=language(ids,pos,torch.zeros(1,1,64),torch.zeros(3,1,1,64),torch.zeros(1,1,1),actual[2])
        check(f'decode_{step}_logits',actual[0],model.lm_head(reference.last_hidden_state[:,-1]))
        check(f'decode_{step}_OBJ_hidden',actual[1],hidden.hidden[:,-1])
(args.output/'wrapper_parity.json').write_text(json.dumps(results,indent=2))
print('ALL WRAPPER CHECKS PASSED',len(results),flush=True)

from wevisone_vl.config import Qwen3VLSegConfig
from wevisone_vl.decoder import build_mask_decoder_from_qwen
torch.backends.mha.set_fastpath_enabled(False)
cfg=Qwen3VLSegConfig(cnn_image_size=64,vision_feature_layers=(0,1,2,3))
decoder=build_mask_decoder_from_qwen(model,cfg).eval()
pixel=Pixel(decoder,4,4).eval()
image=torch.rand(1,3,64,64)
maps=[torch.randn(1,64,4,4) for _ in range(4)]
tseg=torch.randn(2,64)
boxes=torch.tensor([[.1,.1,.8,.9],[.3,.3,.7,.7]])
with torch.no_grad():
    a=pixel(image,*maps,tseg,boxes)
    b=decoder(image,maps,tseg,boxes,(4,4))['mask_logits']
    check('pixel_cascade_N2',a,b,1e-4)
half_model=copy.deepcopy(model).half()
with torch.no_grad():
    expected=half_model.model.visual(pixels.half(),grid_thw=grid)
    half_vision=Vision(half_model.model.visual,4,4,2).eval()
    observed=half_vision(pixels)
    check('vision_fp16_merged',observed[0],expected.pooler_output.float(),2e-3)
    for i in range(3):check(f'vision_fp16_deep_{i}',observed[i+1],expected.deepstack_features[i].float(),2e-3)
del half_model,half_vision
root=args.output/'tiny';root.mkdir(parents=True,exist_ok=True)
seq=8
ids=torch.ones(1,seq,dtype=torch.long)
pos=torch.arange(seq).reshape(1,1,seq).expand(3,1,seq)
s=torch.export.Dim('sequence',min=1,max=64)
p=torch.export.Dim('past_sequence',min=1,max=128)
with torch.no_grad():
    torch.onnx.export(language,(ids,pos,torch.zeros(1,seq,64),torch.zeros(3,1,seq,64),torch.zeros(1,seq,1),torch.zeros(4,2,2,2,16)),
        str(root/'language.onnx'),dynamo=True,external_data=True,opset_version=18,
        input_names=['input_ids','position_ids','image_features','deepstack','visual_mask','past'],
        output_names=['logits','last_hidden','present'],
        dynamic_shapes={'input_ids':{1:s},'position_ids':{2:s},'image_features':{1:s},'deepstack':{2:s},'visual_mask':{1:s},'past':{3:p}})
    print('PASS dynamic language ONNX export',flush=True)
    torch.onnx.export(pixel,(image,*maps,tseg,boxes),str(root/'pixel.onnx'),dynamo=False,opset_version=18,
        input_names=['image','visual_0','visual_1','visual_2','visual_3','t_seg','boxes'],output_names=['mask_logits'],
        dynamic_axes={'t_seg':{0:'objects'},'boxes':{0:'objects'},'mask_logits':{0:'objects'}})
    print('PASS dynamic pixel ONNX export',flush=True)
(args.output/'wrapper_parity.json').write_text(json.dumps(results,indent=2))
