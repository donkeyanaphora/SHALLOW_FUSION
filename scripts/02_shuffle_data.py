import os
import subprocess
import shutil
import random

def make_seedfile(seed: int, seedfile_path: str, size_in_bytes: int = 100_000_000):
    # No need to make 'staging/' if we know it already exists:
    # dir_name = os.path.dirname(seedfile_path)
    # if dir_name:
    #     os.makedirs(dir_name, exist_ok=True)

    r = random.Random(seed)
    data = bytes(r.getrandbits(8) for _ in range(size_in_bytes))

    with open(seedfile_path, "wb") as sf:
        sf.write(data)


def shuffle_file_with_seed(input_file: str, output_file: str, seedfile_path: str, seed: int = 42):
    """
    Shuffles 'input_file' into 'output_file' using a deterministic seed
    via --random-source=seedfile. Works on Linux (shuf) or macOS (gshuf).
    """
    # 1) Check if shuf or gshuf is installed
    shuf_command = shutil.which("shuf") or shutil.which("gshuf")
    if shuf_command is None:
        raise EnvironmentError(
            "Neither 'shuf' nor 'gshuf' is found. "
            "Install GNU coreutils (macOS) or ensure shuf is on PATH."
        )

    # 2) Create a seedfile for reproducible random bytes
    make_seedfile(seed, seedfile_path)

    # 3) Run shuf with --random-source=seedfile
    cmd = [
        shuf_command,
        "--random-source=" + seedfile_path,
        input_file,
        "-o",
        output_file
    ]

    print(f"Shuffling {input_file} -> {output_file} with seed={seed}")
    # 4) Run command
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error during shuffling:")
        print(e.stderr)
        raise

    print(f"Shuffled file saved at: {output_file}")
    # Optionally remove the seedfile if you don't want to keep it
    # os.remove(seedfile_path)

if __name__ == "__main__":
    # Example usage
    os.makedirs("staging", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    seedfile_path = "staging/seedfile.dat"
    input_jsonl = "staging/pubmed_abstracts.jsonl"
    output_jsonl = "data/shuffled_pubmed_abstracts.jsonl"

    shuffle_file_with_seed(input_jsonl, output_jsonl, seedfile_path, seed=42)
