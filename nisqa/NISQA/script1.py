# The less the NISQA score the more the presence of it


args = {
    "mode": "predict_dir",
    "pretrained_model": "weights/nisqa.tar",  # This argument list must be passed from the notebook.
    "data_dir": "recordings/",
    "output_dir": "results/",
    "ms_channel": None,
}

df_splitter_dir = r"df.csv"
df_nisqa_dir = r"results\NISQA_results.csv"


def assess(input_dir="split_audio", **kwargs):
    from nisqa.NISQA_model import nisqaModel

    nisqa = nisqaModel(kwargs)
    nisqa.predict()


def chunk_splitter(audio_dir, output_dir="recordings", split_length=5, threshold=5):
    """This function takes splits the audio files that are > split_length seconds in length and stores them in the audio_dir directory
    audio_dir: Directory path can be relative or full path.
    split_length: argument for how much duration(seconds) the splitted chunks must be."""
    import os
    import shutil
    from pydub import AudioSegment
    from pydub.utils import make_chunks

    chunk_length_ms = 5000  # pydub calculates in millisec
    os.listdir(audio_dir)
    for filename in os.listdir(audio_dir):
        # big_chunk = audio_dir+os.sep+ filename
        big_chunk = filename
        file = AudioSegment.from_file(big_chunk, "wav")
        if file.duration_seconds > 5:
            chunks = make_chunks(file, chunk_length_ms)  # Make chunks of five sec
            for i, chunk in enumerate(chunks):
                # print(big_chunk[:-4])
                # chunk_name = "{1}_{0}.wav".format(i, big_chunk[:-4])
                chunk_name = "{1}_{0}.wav".format(i, big_chunk[:-4])
                # print(audio_dir)
                # print(chunk_name)
                print("exporting", chunk_name)
                chunk.export(os.path.join(output_dir, chunk_name), format="wav")
            # os.remove(big_chunk)
        else:
            shutil.copy(
                os.path.join(audio_dir, filename), os.path.join(output_dir, filename)
            )


def make_chunks(transcript: str, chunk_length: int, number_of_chunks: int):
    return [
        transcript[i * chunk_length : (i + 1) * chunk_length]
        for i in range(int(number_of_chunks))
    ]


def df_unify(df_nisqa, df_splitter, noi_thresh: float, save_locally=False):
    """
    Please provide full path or absolute path to the dataframe
    df_nisqa: Dataframe produced by nisqa module
    df_splitter: Dataframe produced by diarization pipeline
    noi_thresh: At how much percentage the threshold value to set to seperate noisy inputs.
    save_locally: Weather to save the produced dataframe locally or not
    """
    import pandas as pd
    from math import ceil

    df_nisqa = pd.read_csv(df_nisqa)
    df_nisqa.reset_index(inplace=True)
    df_nisqa.index = range(df_nisqa.shape[0])
    df_splitter = pd.read_csv(df_splitter)
    df_splitter = df_splitter.astype(
        {"filename": "string", "transcript": "string", "accent": "string"}
    )
    df_nisqa["filename"] = df_nisqa["deg"]
    df_nisqa = df_nisqa.drop(["deg"], axis=1)
    df_new = pd.DataFrame()
    csv_nisqa_params = [
        "filename",
        "duration",
        "transcript",
        "confidence",
        "age",
        "gender",
        "accent",
        "mos_pred",
        "noi_pred",
        "dis_pred",
        "col_pred",
        "loud_pred",
        "model",
    ]
    for fields in csv_nisqa_params:
        df_new[fields] = ""
    extra_nisqa_params = [
        "filename",
        "duration",
        "transcript",
        "confidence",
        "age",
        "gender",
        "accent",
    ]

    for i in df_splitter.index:
        # print("iteration ", i)
        parent_params = df_splitter.loc[i]
        filename = parent_params.loc["filename"]
        concerned_params = df_splitter.loc[
            i, ["filename", "duration", "confidence", "age", "gender", "accent"]
        ]
        transcript = df_splitter.loc[i, "transcript"]
        chunks = df_nisqa[df_nisqa["filename"].str.contains(filename)]
        chunks.reset_index(inplace=True)
        for fields in extra_nisqa_params:
            chunks[fields] = ""
        no_of_sub_calls = chunks.shape[0]
        chunk_length = ceil(len(transcript) / no_of_sub_calls)
        # print(no_of_sub_calls)
        sub_transcripts = make_chunks(transcript, chunk_length, no_of_sub_calls)
        # print(sub_transcripts)
        # print(len(chunks.index), "\n",len(sub_transcripts))
        # break
        # print(i)
        for sub_chunks in chunks.index:
            # print("Sub_ ITERATION ", sub_chunks)
            # print(sub_chunks)
            # print(chunks.loc[sub_chunks, "transcript"])
            chunks.loc[sub_chunks, "transcript"] = sub_transcripts[sub_chunks]
            # final_parameters = pd.concat([concerned_params, chunks.loc[sub_chunks]], axis=1, ignore_index=True)
            df_new = df_new.append(
                {
                    "filename": df_nisqa.loc[sub_chunks, "filename"],
                    "duration": df_splitter.loc[i, "duration"],
                    "confidence": concerned_params.loc["confidence"],
                    "age": concerned_params.loc["age"],
                    "gender": concerned_params.loc["gender"],
                    "accent": concerned_params.loc["accent"],
                    "mos_pred": chunks.loc[sub_chunks, "mos_pred"],
                    "noi_pred": chunks.loc[sub_chunks, "noi_pred"],
                    "dis_pred": chunks.loc[sub_chunks, "dis_pred"],
                    "col_pred": chunks.loc[sub_chunks, "col_pred"],
                    "loud_pred": chunks.loc[sub_chunks, "loud_pred"],
                    "model": chunks.loc[sub_chunks, "model"],
                    "transcript": chunks.loc[sub_chunks, "transcript"],
                },
                ignore_index=True,
            )
    df_new = df_new.sort_values("noi_pred")
    df_new.reset_index(inplace=True)
    df_new.drop(["index"], axis=1, inplace=True)
    noise_threshold = ceil(noi_thresh * df_new.shape[0])
    noise_threshold_value = df_new.loc[noise_threshold, "noi_pred"].values
    noisy_audio = df_new[df_new["noi_pred" < noise_threshold_value]]
    if save_locally:
        df_new.to_csv("final_data.csv")
    return df_new, noise_threshold_value, noisy_audio


chunk_splitter(audio_dir="D:\\deepspeech\\nisqa\\NISQA\\reserve_recs")
assess(**args)
df_unify(df_nisqa_dir, df_splitter_dir, 0.2)
