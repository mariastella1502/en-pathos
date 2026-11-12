#This script runs the discovery model on a video, extracting frames and audio, transcribing the audio, and generating observations based on a taxonomy of empathy gaps. The results are saved to JSON files for further analysis.

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import mlx_whisper
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

GAPS = {
    'rigidity': 'Avoid scripted, rigid responses; adapt naturally to the person.',
    'information_relevance': 'Select only relevant information, without unnecessary background.',
    'harshness': 'Be delicate, respectful, nonjudgmental, without shaming.',
    'negativity': 'Avoid needless pessimism while remaining realistic and truthful.',
    'genericity': 'Respond specifically to the situation, not with generic platitudes.',
    'overconfidence': 'Show calibrated uncertainty and respect boundaries and autonomy.',
    'automatic_agreement': 'Disagree constructively rather than automatically endorsing claims.'
}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def duration(path):
    info = json.loads(run(['ffprobe','-v','error','-show_format','-of','json',str(path)]))
    return float(info['format']['duration'])


def extract_audio(video, audio):
    run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio)])


def extract_frames(video, start, end, folder, interval=3.0, max_frames=4):
    times = np.linspace(start + 0.25, max(start + 0.25, end - 0.25),
                        min(max_frames, max(1, math.ceil((end-start)/interval))))
    paths=[]
    for i,t in enumerate(times):
        dst=folder/f'frame_{i:02d}.jpg'
        run(['ffmpeg','-y','-loglevel','error','-ss',str(float(t)),'-i',str(video),
             '-frames:v','1','-vf','scale=640:-2',str(dst)])
        if dst.exists(): paths.append((round(float(t),2),str(dst)))
    return paths


def acoustic_features(audio_path, start, end):
    import wave
    with wave.open(str(audio_path),'rb') as w:
        sr=w.getframerate(); w.setpos(min(w.getnframes(),int(start*sr)))
        raw=w.readframes(max(0,int((end-start)*sr)))
    signal=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    if not len(signal): return {'rms':None,'silence_fraction':None}
    rms=float(np.sqrt(np.mean(signal**2)))
    chunks=[signal[i:i+int(sr*.1)] for i in range(0,len(signal),int(sr*.1))]
    silence=float(np.mean([np.sqrt(np.mean(x*x))<0.012 for x in chunks if len(x)]))
    return {'rms':round(rms,4),'silence_fraction':round(silence,3),
            'note':'Whole-mix acoustics; not speaker-specific or emotion labels.'}


def parse_json(text):
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    try: return json.loads(text)
    except json.JSONDecodeError:
        decoder=json.JSONDecoder()
        for i,c in enumerate(text):
            if c=='{':
                try: return decoder.raw_decode(text[i:])[0]
                except json.JSONDecodeError: pass
    return {'parse_error':True,'raw_response':text}


def build_prompt(mode, transcript, acoustics, frames, taxonomy):
    if mode == 'guided':
        instruction = ('GUIDED: Use ONLY behavior IDs defined under the matching gap in the taxonomy. '
                       'Do not force an observation if none is supported.\n'
                       + json.dumps(taxonomy['gaps'], ensure_ascii=False))
    else:
        instruction = ('OPEN: Discover behaviors without any behavior taxonomy. '
                       'Use snake_case names and do not assume any gap is present.')
    return f"""You are a conservative evidence-based multimodal conversation annotator.
Frames are sparse stills, not continuous video. Transcript has no reliable speaker diarization.
Do not infer speaker identity, gaze shifts, emotions, or outcomes without direct evidence.
Do not equate apologies, feelings, or agreement with empathy automatically.
Do not duplicate one observation under multiple gaps or frames.
Seven gap definitions: {json.dumps(GAPS, ensure_ascii=False)}
{instruction}
FRAME TIMES IN SEQUENCE: {[t for t,_ in frames]}
TRANSCRIPT SEGMENTS: {json.dumps(transcript, ensure_ascii=False)}
ACOUSTIC MEASUREMENTS: {json.dumps(acoustics)}
Return ONLY a JSON object with exactly these top-level keys: scene_summary, observations, unknowns.
observations must be an array (may be empty), maximum 8. Each observation has:
gap (one of seven keys), behavior (snake_case), polarity (helpful|unhelpful|ambiguous),
modality (verbal|visual|audio|multimodal), evidence_quote (exact transcript substring or null),
evidence_frame_time (one of FRAME TIMES IN SEQUENCE or null),
evidence_description (directly observable fact, NOT interpretation),
interpretation (cautious hypothesis), confidence (0..1), interlocutor_response (string or null).
For visual evidence select ONLY a provided frame time and describe visible features concretely.
For verbal-only evidence you may set evidence_frame_time=null. Do not invent screenshots or paths.
Prefer no observations to unsupported ones."""


def normalize_analysis(analysis, mode, taxonomy, frames, segments, screenshot_by_time):
    if not isinstance(analysis, dict):
        return {'scene_summary':'', 'observations':[], 'unknowns':['Model did not return a JSON object'], 'validation_errors':['not_an_object']}
    errors=[]
    if not isinstance(analysis.get('observations'), list):
        errors.append('missing_or_invalid_observations_array')
        observations=[]
    else:
        observations=analysis['observations']
    normalized=[]
    valid_gaps=set(GAPS)
    transcript_text=' '.join(str(s.get('text','')) for s in segments)
    seen=set()
    for i,obs in enumerate(observations):
        if not isinstance(obs,dict):
            errors.append(f'observation_{i}_not_object'); continue
        gap=obs.get('gap'); behavior=obs.get('behavior')
        if gap not in valid_gaps or not isinstance(behavior,str):
            errors.append(f'observation_{i}_invalid_gap_or_behavior'); continue
        if mode=='guided' and behavior not in taxonomy['gaps'][gap]['behaviors']:
            errors.append(f'observation_{i}_behavior_not_in_taxonomy'); continue
        quote=obs.get('evidence_quote')
        if quote is not None and (not isinstance(quote,str) or quote not in transcript_text):
            errors.append(f'observation_{i}_quote_not_in_transcript'); continue
        t=obs.get('evidence_frame_time')
        if t is not None:
            if not isinstance(t,(float,int)) or not screenshot_by_time:
                errors.append(f'observation_{i}_invalid_frame_time'); continue
            # Model times can be slightly rounded; never pretend an unprovided time was actually analyzed.
            nearest=min(screenshot_by_time,key=lambda x:abs(x-float(t)))
            if abs(nearest-float(t))>0.03:
                errors.append(f'observation_{i}_frame_time_not_supplied'); continue
            t=nearest
        if obs.get('modality') in ('visual','multimodal') and t is None:
            errors.append(f'observation_{i}_visual_without_frame'); continue
        if not quote and t is None:
            errors.append(f'observation_{i}_no_evidence'); continue
        key=(gap,behavior,quote,t)
        if key in seen:
            errors.append(f'observation_{i}_duplicate');continue
        seen.add(key)
        obs=dict(obs)
        obs['evidence_frame_time']=t
        obs['screenshot_path']=screenshot_by_time.get(t) if t is not None else None
        obs['screenshot_status']='model_observed_frame' if t is not None else 'no_visual_evidence'
        if mode=='guided':
            definition=taxonomy['gaps'][gap]['behaviors'][behavior]
            obs['behavior_definition_it']=definition['definition_it']
            obs['gap_label_it']=taxonomy['gaps'][gap]['label_it']
        normalized.append(obs)
    return {'scene_summary':analysis.get('scene_summary',''),
            'observations':normalized,
            'unknowns':analysis.get('unknowns',[]) if isinstance(analysis.get('unknowns'),list) else [],
            'validation_errors':errors}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',default='data/processed/meld/scenes/train_4.mp4')
    p.add_argument('--out',default='data/processed/meld/annotations')
    p.add_argument('--taxonomy',default='data/benchmarks/empathy_taxonomy_detailed.json')
    p.add_argument('--model',default='mlx-community/Qwen2.5-VL-7B-Instruct-4bit')
    p.add_argument('--window',type=float,default=12)
    p.add_argument('--stride',type=float,default=9)
    p.add_argument('--mode',choices=['guided','open','both'],default='both')
    p.add_argument('--max-tokens',type=int,default=2200)
    args=p.parse_args()
    if args.window<=0 or args.stride>args.window or args.stride<=0:
        p.error('Require window > 0 and 0 < stride <= window')
    video=Path(args.video).resolve()
    if not video.is_file(): p.error(f'Missing video: {video}')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    taxonomy=json.loads(Path(args.taxonomy).read_text(encoding='utf-8'))
    screenshot_root=out/'frames'/video.stem
    screenshot_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp); audio=tmp/'audio.wav';extract_audio(video,audio)
        transcription=mlx_whisper.transcribe(str(audio),path_or_hf_repo='mlx-community/whisper-large-v3-turbo',word_timestamps=False)
        (out/f'{video.stem}_transcript.json').write_text(json.dumps(transcription,indent=2,ensure_ascii=False))
        model,processor=load(args.model);config=load_config(args.model)
        total=duration(video)
        starts=np.arange(0,total,args.stride)
        modes=['guided','open'] if args.mode=='both' else [args.mode]
        for mode in modes:
            results=[]
            for index,start in enumerate(starts):
                end=min(total,float(start)+args.window)
                frame_dir=screenshot_root/f'window_{index:03d}'
                frame_dir.mkdir(parents=True,exist_ok=True)
                frames=extract_frames(video,float(start),end,frame_dir)
                screenshots={t:str(Path(path).resolve()) for t,path in frames}
                segments=[{'start':seg['start'],'end':seg['end'],'text':seg['text']}
                          for seg in transcription['segments'] if seg['end']>=start and seg['start']<=end]
                acoustics=acoustic_features(audio,float(start),end)
                prompt=build_prompt(mode,segments,acoustics,frames,taxonomy)
                paths=[path for _,path in frames]
                formatted=apply_chat_template(processor,config,prompt,num_images=len(paths))
                output=generate(model,processor,formatted,paths,max_tokens=args.max_tokens,temperature=0.0,verbose=False)
                text=output.text if hasattr(output,'text') else str(output)
                analysis=normalize_analysis(parse_json(text),mode,taxonomy,frames,segments,screenshots)
                record={'window_id':index,'start':round(float(start),2),'end':round(end,2),
                        'frames':[{'timestamp_seconds':t,'screenshot_path':str(Path(path).resolve())} for t,path in frames],
                        'analysis':analysis}
                results.append(record)
                target=out/f'{video.stem}_{mode}.json'
                target.write_text(json.dumps({'video':str(video),'mode':mode,'model':args.model,
                    'taxonomy_version':taxonomy.get('version') if mode=='guided' else None,
                    'windows':results},indent=2,ensure_ascii=False),encoding='utf-8')
                print(f'{mode}: window {index+1}/{len(starts)} -> {target}',flush=True)
    print('Completed. Screenshots are preserved under the annotations/frames directory. Human review required.')

if __name__=='__main__': main()
