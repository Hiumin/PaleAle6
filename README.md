# PaleAle6.0 (forked)  

This repository is a fork of [PaleAle6.0][paleale6], a program for predicting relative solvent accessibility (RSA) in 2 states, 4 states, or real values.  

[paleale6]: https://github.com/WafaAlanazi/PaleAle6  


Table of Contents  

<!-- Table of Contents GFM -->

* [1. Modifications](#1-modifications)
* [2. Installation](#2-installation)
* [3. Usage](#3-usage)

<!-- /Table of Contents -->


## 1. Modifications  

Main modifications:  

- Implemented a wrapper script with a customizable command-line interface.  
- Re-organized and renamed some files and directories for clarity.  
- Implemented a dynamic path finder (as opposed to hard-coded paths).  
- Implemented a prediction parser to convert RSA tendencies into FASTA-format sequences (for 2- and 4-state predictions).  
- Output progress messages more explicitly.  

New or heavily modified components:  

| Component                      | Purpose                                                     |
| ------------------------------ | ----------------------------------------------------------- |
| `PaleAle6.sh`                  | Main wrapper + CLI.                                         |
| `parse_solvacc_pred.py`        | Parse tendency predictions into FASTA-format sequences.     |
| `fasta2json.py`                | Convert input fasta into json.                              |
| `emb_esm3_fasta.py`            | Generate ESM-2 feature embedding.                           |
| `RSA_*/new_test_ensemble.py`   | Lower-level secondary wrapper for the predictions.          |
| `RSA_*/utils/ensemble.py`      | RSA prediction script.                                      |
| `RSA_*/params/filePath.py`     | Dynamic path finder.                                        |
| `RSA_*/training`               | Pre-trained models. renamed from `output`.                  |
| `original`                     | Everything from the original repository.                    |

## 2. Installation  

Installing from source:  

``` bash
git clone https://github.com/Hiumin/PaleAle6.git
cd PaleAle6
micromamba create -n PaleAle6 -f env_PaleAle6.yml
micromamba activate PaleAle6
pip install -r env_PaleAle6.txt
ln -s PaleAle6.sh PaleAle6
export PATH=$(pwd):$PATH
PaleAle6 -h
```

Verifying the installation:  

```bash
PaleAle6 -i test/TIGR04045_dealn.FASTA -o solvacc -p TIGR04045 -2 -4 -r 
```

## 3. Usage  

```
Predicts per-residue relative solvent accessibility from protein sequence.
Returns predicted tendencies in JSON and sequences in FASTA format.

Version: 6.0

Usage: PaleAle6 -i fasta -o outdir {-2 -4 -r} [-p prefix]

Arguments:
    -i, --infasta           [Required] An input file in fasta format.
                            May contain one or multiple (unaligned) sequences.
    -o, --outdir            [Required] Path to the output directory (will be created if not existing).
                            Prediction results will be placed in the corresponding subdirectory
                            for each run mode (2-state, 4-state, and/or real-value).
    -p, --outprefix         A prefix for naming the output prediction files (tendencies and states).
                            Note: Provide only a base name, no paths.
    -2e, --outext-2state    An extension for naming the final fasta-format 2-state predictions.
                            Note: Do not include the dot. Default: rsa2c.
    -4e, --outext-4state    An extension for naming the final fasta-format 4-state predictions.
                            Note: Do not include the dot. Default: rsa4c.

Options:
    -2, --rsa2c         Predict in 2 states (exposed or buried). Default: off.
                        Exposure threshold: 25%.
                        Can be combined with other prediction modes.
    -4, --rsa4c         Predict in 4 states. Default: off.
                        Exposure threshold: 4%, 25%, and 50%.
                        Can be combined with other prediction modes.
    -r, --rsarv         Predict in real values. Default: off.
                        Can be combined with other prediction modes.
    -s, --skip          Skip sequence format conversion and embedding generation (if the files already exist).
    --cleanup           Remove intermediate files (JSON sequences, embeddings, etc.)
```
