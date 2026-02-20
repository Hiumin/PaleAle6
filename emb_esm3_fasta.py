# Hieu Minh Truong modified
import os
import argparse
import torch
import subprocess
import numpy as np  # Import NumPy for saving arrays
from Bio import SeqIO  # Use Biopython to read FASTA files

try:
    import esm
except ImportError:
    # If esm is not found, install it
    subprocess.check_call(["pip", "install", "fair-esm"])
    import esm

torch.set_grad_enabled(False)


# HMT: Receive file paths from the wrapper script
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infasta', required=True)
parser.add_argument('-o', '--outdir', required=True)
args = parser.parse_args()

infasta = args.infasta
infasta_base = os.path.splitext(os.path.basename(infasta))[0]
outdir = os.path.join(args.outdir, infasta_base)
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Function to read sequences from a FASTA file
def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append({
            'id': record.id,
            'sequence': str(record.seq)
        })
    return sequences

# Read sequences from the FASTA file
list_entity = read_fasta(infasta)


# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# Iterate through each sequence in the FASTA file
for entity in list_entity:
    print(entity['id'])
    data = [(entity['id'], entity['sequence'])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    embedded_seq = token_representations[0, 1 : - 1].numpy()

    # Save the embedding using numpy.save
    np.save(os.path.join(outdir, f"{entity['id']}.npy"), embedded_seq)

print(f"ESM-2 embeddings successfully generated at\n{outdir}/")
