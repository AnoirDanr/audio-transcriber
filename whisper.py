from faster_whisper import WhisperModel
import sys,os,ffmpeg,math

def split_audio(input_file, chunk_duration=300):
    probe = ffmpeg.probe(input_file)
    duration = float(probe["format"]["duration"])
    os.makedirs("chunks", exist_ok=True)
    chunks = []
    total_chunks = math.ceil(duration / chunk_duration)
    for i in range(total_chunks):
        start = i * chunk_duration
        output = f"chunks/chunk_{i:03d}.wav"
        (
            ffmpeg
            .input(input_file, ss=start, t=chunk_duration)
            .output(output, ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        chunks.append(output)
    return chunks

def transcript_chunks(chunks, model:WhisperModel,chunk_duration=300):
    for i,chunk in enumerate(chunks):
        offset = i*chunk_duration
        for segment in whisper_model.transcribe(chunk, language="it",beam_size=1)[0]:
            print(f"[{format_hms(segment.start+offset) } -> {format_hms(segment.end+offset)}] {segment.text}")





assert len(sys.argv) == 2, "transcript [nome_file_audio]"


def format_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

file_name = sys.argv[1]


whisper_model = WhisperModel(model_size_or_path="small",device="cpu")
chunks = split_audio(file_name)
transcript_chunks(chunks,model=whisper_model)
