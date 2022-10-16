# def download_model_n_scorer():
#   from wget import download
#   download("https://coqui.gateway.scarf.sh/english/coqui/v1.0.0-large-vocab/model.tflite")
#   download("https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer")
#   download("https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm")
#   download("https://coqui.gateway.scarf.sh/english/coqui/v1.0.0-large-vocab/large_vocabulary.scorer")


def make():
  from os import mkdir, path
  if not path.exists("Timings"):
    mkdir("Timings")
  if not path.exists("splitted_audio"):
    mkdir("splitted_audio")
  



import pandas as pd
import numpy as np
import os
import wave

def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        print("Rate:", rate)
        print("Frames:", frames)
        print("Buffer Len:", len(buffer))

    return buffer, rate

def transcribe_batch(audio_file, model):
    buffer, rate = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    pred = model.sttWithMetadata(data16)
    sentence = ""
    confidence = pred.transcripts[0].confidence
    for charecter in pred.transcripts[0].tokens:
      sentence = sentence+charecter.text
    
    return sentence, confidence


def init_model_stt():
  from stt import Model
  from deepspeech import Model as ds_model
  model_file = 'model.tflite'
  lm_file_path = "large_vocabulary.scorer"
  # ds_model_file = ""
  ds_model_file = 'deepspeech-0.9.3-models.pbmm'
  ds_lm_file_path = "deepspeech-0.9.3-models.scorer"
  beam_width = 2000
  lm_alpha = 0.93
  lm_beta = 1.18
  #import the acoustic model
  model = Model(model_file)
  model.enableExternalScorer(lm_file_path)
  model.setScorerAlphaBeta(lm_alpha, lm_beta)
  model.setBeamWidth(beam_width)
  ds_model = ds_model(ds_model_file)
  ds_model.enableExternalScorer(ds_lm_file_path)
  ds_model.setScorerAlphaBeta(lm_alpha, lm_beta)
  ds_model.setBeamWidth(beam_width)
  print("Initialized the stt model successfully")
  return model, ds_model

def init_diarize_pipline():
  from pyannote.audio import Pipeline
  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
  return pipeline

def create_df(rows: list):
  df = pd.DataFrame()
  for i in rows:
    df[i] = []
  return df




def aud_dn(links: list, data_dir):
  from wget import download
  for i in links:
    download(i, data_dir)


def diarize(audio_file_list, pipeline):
  timing_path = f"{os.getcwd()}{os.sep}Timings"
  if not os.path.exists(timing_path):
    os.mkdir(timing_path)
  file_counter = 0
  for file in audio_file_list:
    diarization = pipeline(f"{file}")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
      #if speaker =="SPEAKER_01":
      #  continue
      # print(f"{i}: start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
      with open(f"{timing_path}{os.sep}{file.split('/')[-1][:-4]}.txt" ,"a") as f:
        f.writelines((f"{turn.start:.1f} {turn.end:.1f} {speaker}\n"))

    


def audio_splitter(ROOT, data_dir, audio_file_list, model, df):
  from pydub import AudioSegment
  import glob
  timing_dir = os.path.join(ROOT,'Timings')
  timings_list = glob.glob(f"{timing_dir}/*.txt")
  data_list = os.listdir(data_dir)
  data_list = [x for x in data_list if x!= ".ipynb_checkpoints"]
  for timing, audio, filename in zip(sorted(timings_list), sorted(audio_file_list), sorted(data_list)):
    if filename==".ipynb_checkpoints":
      continue
    with open(f"{timing}") as f:
      audio_file = AudioSegment.from_file(audio)
      export_path = os.path.join(ROOT,"splitted_audio") #make splitted auido folder
      for num, line in enumerate(f):
        time = line.split(" ")
        start = float(time[0])
        end = float(time[1])
        sliced_audio = audio_file[start*1000: end*1000]
        export_name = export_path + os.sep + filename[:-4]+ "_" + str(num)
        transcript, confidence = transcribe_batch(sliced_audio.export(f"{export_name}.wav", format="wav"), model=model) #This splits and saves the audio file and saves it to directory and get the transcription
        df = df.append({"filename": filename[:-4]+ "_" + str(num) , "transcript": transcript , "confidence":confidence, "loudness":0, "noisiness":0, "coloration":0 , "discontinuity":0, "age":0, "gender":0, "accent":0}, ignore_index=True)
        if num == 3:
          break
  return df
        










