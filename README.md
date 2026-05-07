# PaleAle6.0 (forked)  

This repository is a fork of [PaleAle6.0][paleale6], a program for predicting relative solvent accessibility (RSA) in 2 states, 4 states, or real values.  

[paleale6]: https://github.com/WafaAlanazi/PaleAle6  


**Table of Contents**  

<!-- Table of Contents GFM -->

* [1. Modifications](#1-modifications)
* [2. Installation](#2-installation)
* [3. Usage](#3-usage)

<!-- /Table of Contents -->


## 1. Modifications  

Main modifications:  

- Implemented a wrapper script with a customizable command-line interface.  
- Implemented a dynamic path finder (as opposed to hard-coded paths).  
- Implemented a prediction parser to convert RSA tendencies into FASTA-format sequences (for 2- and 4-state predictions).  
- Re-organized and renamed some files and directories for clarity.  
- Load pre-saved models safely in compliance with newer `torch` (>= 2.6). Relevant sources: [1][torch load], [2][solution p1], [3][solution p2].  
- Enable GPU acceleration for MacOS.  
- Output progress messages more explicitly.  

[torch load]: https://docs.pytorch.org/docs/main/notes/serialization.html#weights-only-security  
[solution p1]: https://github.com/suno-ai/bark/issues/626#issuecomment-3198148041  
[solution p2]: https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error  

New or heavily modified components:  

| Component                      | Purpose                                                     |
| ------------------------------ | ----------------------------------------------------------- |
| `PaleAle6.sh`                  | Main wrapper + CLI.                                         |
| `fasta2json.py`                | Convert input fasta into json.                              |
| `emb_esm3_fasta.py`            | Generate ESM-2 feature embedding.                           |
| `RSA_*/params/filePath.py`     | Dynamic path finder.                                        |
| `RSA_*/new_test_ensemble.py`   | Lower-level secondary wrapper for the predictions.          |
| `RSA_*/utils/ensemble.py`      | RSA prediction script.                                      |
| `RSA_*/training`               | Pre-trained models. Previously named `output`.              |
| `parse_solvacc_pred.py`        | Parse tendency predictions into FASTA-format sequences.     |
| `archive`                      | Everything from the original repository.                    |

## 2. Installation  

Installing from source, with `conda`/`mamba`:  

``` bash
git clone https://github.com/Hiumin/PaleAle6.git
cd PaleAle6
conda create -n PaleAle6 -f env_PaleAle6.yaml
conda activate PaleAle6
pip install -r env_PaleAle6.pip
chmod +x PaleAle6.sh
ln -s PaleAle6.sh PaleAle6
```

Verify the installation by running it on a test sequence:  

```bash
PaleAle6 -i test/TIGR02284_dealigned.FASTA -o solvacc -p TIGR02284 -2 -4 -r 
```

To be able to call the program from anywhere, add the `PaleAle6` directory to `PATH` by adding the following line to `~/.bashrc`:  

```bash
export PATH=$PATH:$(pwd)
```

## 3. Usage  

```
Predicts per-residue relative solvent accessibility from protein sequences.
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
                            Note: Do not include the dot. Default: 2sa.
    -4e, --outext-4state    An extension for naming the final fasta-format 4-state predictions.
                            Note: Do not include the dot. Default: 4sa.

Options:
    -2, --rsa2c         Predict in 2 states (exposed or buried). Default: off.
                        Exposure threshold: 25%.
                        Can be combined with other prediction modes.
    -4, --rsa4c         Predict in 4 states. Default: off.
                        Exposure threshold: 4%, 25%, and 50%.
                        Can be combined with other prediction modes.
    -r, --rsarv         Predict in real values. Default: off.
                        Can be combined with other prediction modes.
    --skip              Skip sequence format conversion and embedding generation (if the files already exist).
    --cleanup           Remove intermediate files (JSON sequences, embeddings, etc.)
```

