# Hieu Minh Truong moddified
import json
from Bio import SeqIO
import os
import argparse

# HMT: Receive file paths from the wrapper script
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infasta', required=True)
parser.add_argument('-o', '--outdir', required=True)
args = parser.parse_args()

infasta = args.infasta
outdir = args.outdir
# HMT: Get the basename by removing the very last extension
# e.g., file.example.fasta -> file.example
infasta_base = os.path.splitext(os.path.basename(infasta))[0]
outjson = os.path.join(outdir, infasta_base + '.json')


def fasta_to_json(fasta_file_path, json_file_path):
    # Create a list to store the entries
    sequence_data = []

    # Read the FASTA file
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequence_entry = {
            "id": record.id,             # Name/ID of the sequence
            "sequence": str(record.seq), # Sequence as a string
            "seq_len": len(record.seq)   # Length of the sequence
        }
        sequence_data.append(sequence_entry)

    # Save the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(sequence_data, json_file, indent=4)

    print(f"FASTA: {fasta_file_path}")
    print(f"JSON : {json_file_path}")
    print("Conversion completed.")


# Do the conversion
fasta_to_json(infasta, outjson)
