"""Official CLAP audio branch, strict checkpoint loading, bounded local decoding."""
from pathlib import Path
import json
import subprocess
import numpy as np
from .sound import MODEL,REVISION,WEIGHTS_SHA256,sha256

MODEL_DIR=Path(__file__).resolve().parent/'data/cache/clap-music-model'


def duration(path):
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                      '-of','json',str(Path(path).resolve())],capture_output=True,timeout=20,check=True)
    seconds=float(json.loads(p.stdout)['format']['duration'])
    if not np.isfinite(seconds) or seconds<=0 or seconds>7200:
        raise ValueError('Audio duration must be positive and at most two hours')
    return seconds


def decode_segment(path,start):
    p=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(Path(path).resolve()),
                      '-t','10','-vn','-ac','1','-ar','48000','-f','f32le','pipe:1'],
                     capture_output=True,timeout=45,check=True)
    v=np.frombuffer(p.stdout,dtype='<f4').copy()
    if len(v)<4800 or len(v)>480001 or not np.isfinite(v).all():
        raise ValueError('Invalid or too short audio segment')
    # Exactly 10s avoids the extractor's random long-audio cropping.
    return np.tile(v,int(np.ceil(480000/len(v))))[:480000]


def download_model(destination=MODEL_DIR):
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL,revision=REVISION,local_dir=str(destination),
                      allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin'],max_workers=1)
    if sha256(Path(destination)/'pytorch_model.bin')!=WEIGHTS_SHA256:
        raise ValueError('Official checkpoint checksum mismatch')


class ClapEncoder:
    def __init__(self,model_dir=MODEL_DIR,device='auto'):
        import torch
        from importlib.metadata import version
        if version('transformers')!='4.57.6':raise ValueError('Use transformers==4.57.6 for this cache contract')
        from transformers import ClapConfig,ClapFeatureExtractor,ClapAudioModelWithProjection
        model_dir=Path(model_dir)
        if sha256(model_dir/'pytorch_model.bin')!=WEIGHTS_SHA256:
            raise ValueError('Wrong CLAP weights; run sound_import download')
        expected={'config.json':'2d7722d338bb83ea8824272b1431f088954d3425d79eb3c2d39489478516dc03',
                  'preprocessor_config.json':'9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753'}
        for name,digest in expected.items():
            if sha256(model_dir/name)!=digest:raise ValueError(f'Unexpected CLAP configuration: {name}')
        # Validate configuration against the pinned repository files as well.
        config=ClapConfig.from_pretrained(model_dir,local_files_only=True)
        if config.audio_config.hidden_size!=1024 or config.projection_dim!=512 or config.audio_config.depths!=[2,2,12,2]:
            raise ValueError('Unexpected CLAP architecture')
        config.audio_config.projection_dim=config.projection_dim
        config.audio_config.projection_hidden_act=config.projection_hidden_act
        self.model=ClapAudioModelWithProjection(config.audio_config)
        state=torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
        audio={k:v for k,v in state.items() if k.startswith(('audio_model.','audio_projection.'))}
        self.model.load_state_dict(audio,strict=True)
        del state,audio
        self.extractor=ClapFeatureExtractor.from_pretrained(model_dir,local_files_only=True)
        if self.extractor.sampling_rate!=48000 or self.extractor.nb_max_samples!=480000:
            raise ValueError('Unexpected CLAP preprocessing')
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        torch.set_num_threads(min(4,torch.get_num_threads()))
        torch.backends.cudnn.benchmark=False
        self.model.to(self.device).eval()

    def encode(self,path):
        import torch
        seconds=duration(path);vectors=[]
        for fraction in (.50,):
            waveform=decode_segment(path,max(0,seconds-10)*fraction)
            inputs=self.extractor(waveform,sampling_rate=48000,return_tensors='pt')
            with torch.inference_mode():
                v=self.model(**{k:v.to(self.device) for k,v in inputs.items()}).audio_embeds[0]
                v=torch.nn.functional.normalize(v,dim=0).cpu().numpy()
            vectors.append(v)
        return np.mean(vectors,axis=0)
