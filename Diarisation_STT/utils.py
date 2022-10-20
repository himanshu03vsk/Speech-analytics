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
    with wave.open(filename, "rb") as w:
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
        sentence = sentence + charecter.text

    return sentence, confidence


def init_model_stt(model_file, lm_file):
    from stt import Model

    # from deepspeech import Model as ds_model
    model_file = model_file
    lm_file_path = lm_file
    # ds_model_file = ""
    # ds_model_file = 'deepspeech-0.9.3-models.pbmm'
    # ds_lm_file_path = "deepspeech-0.9.3-models.scorer"
    beam_width = 2000
    lm_alpha = 0.93
    lm_beta = 1.18
    # import the acoustic model
    model = Model(model_file)
    model.enableExternalScorer(lm_file_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)
    # ds_model = ds_model(ds_model_file)
    # ds_model.enableExternalScorer(ds_lm_file_path)
    # ds_model.setScorerAlphaBeta(lm_alpha, lm_beta)
    # ds_model.setBeamWidth(beam_width)
    ds_model = None
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


def diarize(audio_file_list, pipeline, output_dir):
    # timing_path = f"{os.getcwd()}{os.sep}Timings"
    timing_path = f"{output_dir}"
    if not os.path.exists(timing_path):
        os.mkdir(timing_path)
    file_counter = 0
    for file in audio_file_list:
        diarization = pipeline(f"{file}")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # if speaker =="SPEAKER_01":
            #  continue
            # print(f"{i}: start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            with open(f"{timing_path}{os.sep}{file.split('/')[-1][:-4]}.txt", "a") as f:
                f.writelines((f"{turn.start:.1f} {turn.end:.1f} {speaker}\n"))


def audio_splitter(ROOT, data_dir, audio_file_list, model, df):
    from pydub import AudioSegment
    import glob

    timing_dir = os.path.join(ROOT, "Timings")
    timings_list = glob.glob(f"{timing_dir}/*.txt")
    data_list = os.listdir(data_dir)
    data_list = [x for x in data_list if x != ".ipynb_checkpoints"]
    for timing, audio, filename in zip(
        sorted(timings_list), sorted(audio_file_list), sorted(data_list)
    ):
        if filename == ".ipynb_checkpoints":
            continue
        with open(f"{timing}") as f:
            audio_file = AudioSegment.from_file(audio)
            export_path = os.path.join(
                ROOT, "splitted_audio"
            )  # make splitted auido folder
            for num, line in enumerate(f):
                time = line.split(" ")
                start = float(time[0])
                end = float(time[1])
                sliced_audio = audio_file[start * 1000 : end * 1000]
                export_name = export_path + os.sep + filename[:-4] + "_" + str(num)
                transcript, confidence = transcribe_batch(
                    sliced_audio.export(f"{export_name}.wav", format="wav"), model=model
                )  # This splits and saves the audio file and saves it to directory and get the transcription

                #! TODO
                # * Read the exported wav file and do the nisqa check
                # * Add logic of splitting the audios which are greater in length

                df = df.append(
                    {
                        "filename": filename[:-4] + "_" + str(num),
                        "transcript": transcript,
                        "confidence": confidence,
                        "loudness": 0,
                        "noisiness": 0,
                        "coloration": 0,
                        "discontinuity": 0,
                        "age": 0,
                        "gender": 0,
                        "accent": 0,
                    },
                    ignore_index=True,
                )
                if num == 3:
                    break
    return df


"""Notes:
Figure out way to do nisqa on >5sec samples
"""


def audio_splitter(ROOT, data_dir, audio_file_list, model, df, output_dir):
    """
    This function will basically contain all the parellel part that was modified in MIRO Board, It will:
    1. Do the splitting
    2. Do the Nisqa Assessment
    3. Do the acoustic assessment
    4. Noise Classification
    5. Do the Vocab assessment
    6. Bias Detection
    ROOT: your cwd
    data_dir: Directory of audio calls
    """
    from pydub import AudioSegment  # To read the WAV files
    import glob  # Kind of a regex to find any type of extension files in a folder
    from pydub.utils import make_chunks  # To make equal chunks of audio

    args = {
        "mode": "predict_file",
        "deg": "",
        "pretrained_model": r"C:\Users\himan\Documents\Speech\nisqa\NISQA\weights\nisqa.tar",
        "output_dir": None,
        "ms_channel": None,
    }

    timing_dir = os.path.join(
        ROOT, "Timings"
    )  # This specifies where to look for the timing file of the particular audio
    timings_list = glob.glob(
        f"{timing_dir}/*.txt"
    )  # Find all the txt files in the foldr
    data_list = os.listdir(data_dir)  # Get the audio calls present in the directory
    data_list = [x for x in data_list if x != ".ipynb_checkpoints"]
    # Init the nisqa model here
    from nisqa.NISQA.nisqa.NISQA_model import nisqaModel

    for timing, audio, filename in zip(
        sorted(timings_list), sorted(audio_file_list), sorted(data_list)
    ):  # This is the main loop and will do all the task mentioned in the docstring
        # if filename==".ipynb_checkpoints":
        #   continue
        with open(f"{timing}") as f:  # Read the timing file

            audio_file = AudioSegment.from_file(audio)
            export_path = os.path.join(
                ROOT, "splitted_audio"
            )  # make splitted auido folder

            for num, line in enumerate(f):

                time = line.split(" ")
                start = float(time[0])
                end = float(time[1])
                sliced_audio = audio_file[start * 1000 : end * 1000]
                slice_audio_name = "{1}_{0}.wav".format(num, audio[:-4])

                if sliced_audio.duration_seconds > 5.0:  # Make chunks
                    chunks = make_chunks(sliced_audio, 5000)

                    for i, chunk in enumerate(chunks):
                        # print(big_chunk[:-4])
                        # chunk_name = "{1}_{0}.wav".format(i, big_chunk[:-4])
                        chunk_name = f"{1}_{0}.wav".format(
                            i, slice_audio_name.split(".")[0]
                        )
                        # print(audio_dir)
                        # print(chunk_name)

                        print("exporting", chunk_name)
                        exp_file = chunk.export(
                            os.path.join(output_dir, chunk_name), format="wav"
                        )
                        args["deg"] = exp_file

                        # ? For now I am taking only noise values but will make container for different parameters as well
                        # DO NISQA assessment on splitted audio file
                        model = nisqaModel(args)
                        nisqa_params = model.predict().noi_pred.values[0]
                        transcript, confidence = transcribe_batch(exp_file, model=model)

                        # TODO PESQ CODE HERE

                        # TODO BIAS CODE HERE

                        # TODO ADD the parameters into a dataframe so we can get info about these audio files

                else:
                    exp_file = sliced_audio.export(
                        os.path.join(output_dir, slice_audio_name), format="wav"
                    )
                    args["deg"] = exp_file
                    model = nisqaModel(args)
                    nisqa_params = model.predict().noi_pred.values[0]
                    transcript, confidence = transcribe_batch(exp_file, model=model)

                    # TODO PESQ CODE HERE

                    # TODO BIAS CODE HERE

                    # TODO ADD the parameters into a dataframe so we can get info about these audio files
                #! TODO
                # * Read the exported wav file and do the nisqa check
                # * Add logic of splitting the audios which are greater in length

                df = df.append(
                    {
                        "filename": filename[:-4] + "_" + str(num),
                        "transcript": transcript,
                        "confidence": confidence,
                        "loudness": 0,
                        "noisiness": 0,
                        "coloration": 0,
                        "discontinuity": 0,
                        "age": 0,
                        "gender": 0,
                        "accent": 0,
                    },
                    ignore_index=True,
                )
                if num == 3:
                    break
    return df


#  This argument list must be passed from the notebook.
{
    "mode": "predict_file",
    "pretrained_model": r"C:\Users\himan\Documents\Speech\nisqa\NISQA\weights\nisqa.tar",
    "output_dir": None,
    "ms_channel": None,
}
