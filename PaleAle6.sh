#!/bin/bash


showhelp () {
    cat <<- '#EOF'
Predict per-residue solvent accessibility from protein sequence using PaleAle6.0.
Return predictions in JSON format.

Usage: doggo_drink [options] -i fasta -d outdir [-o prefix]

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
#EOF
    exit 1
}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parse command-line arguments
if [ $# -eq 0 ]; then
    showhelp
fi

# Set default values
rsa2c=0
rsa4c=0
rsarv=0
skip=0
cleanup=0
outext_2state='rsa2c'
outext_4state='rsa4c'

while [ $# -ge 1 ]; do
    case $1 in
        -i|--infasta)
            infasta="$(readlink -e "$2")"
            shift; shift ;;
        -o|--outdir)
            outdir="$(readlink -m "$2")"
            mkdir -p "$outdir"
            shift; shift ;;
        -p|--outprefix)
            outprefix="$2"
            shift; shift ;;
        -2e|--outext-2state)
            outext_2state="$2"
            shift; shift ;;
        -4e|--outext-4state)
            outext_4state="$2"
            shift; shift ;;

        -2|--rsa2c)
            rsa2c=1
            shift ;;
        -4|--rsa4c)
            rsa4c=1
            shift ;;
        -r|--rsarv)
            rsarv=1
            shift ;;

        -s|--skip)
            skip=1
            shift ;;
        --cleanup)
            cleanup=1
            shift ;;

        -h|--help)
            showhelp ;;
        -*|--*)
            echo "Option not recognized: $1"
            echo "Use -h or --help for help."
            exit 1 ;;
        *)
            break ;;
    esac
done

# Check if any prediction mode is specified.
checkmode=$(expr $rsa2c + $rsa4c + $rsarv)
if [ $checkmode -eq 0 ]; then
    printf "No prediction mode specified.\nExiting.\n"
    exit 1
fi

# Read and create file paths
scriptdir="$(dirname "$(readlink -e "$0")")"
# Remove the very last extension to get the basename
infasta_base="$(basename "${infasta%.*}")"
if [ -z "$outprefix" ]; then
    outprefix="$infasta_base"
fi

#Summarize input size
seq_count="$(grep -o ">" "$infasta" | wc -l)"

calc_mean_seq_len () {
    awk -c -v count=$seq_count '
        BEGIN { RS=">"; FS="\n"; }
        NR >= 2 { for (i=2;i<=NF;i++) { sum+=length($i) } }
        END { printf "%d", sum/count }
        ' "${1:-/dev/stdin}"
}

mean_length="$(calc_mean_seq_len "$infasta")"

echo "Sequences:   $infasta"
echo "Count:       $seq_count"
echo "Mean length: $mean_length"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
printf "\nStep 1 - Convert FASTA to JSON\n"
mkdir -p "${outdir}/json"
# Note: The output json file is named using the basename of the input fasta
# i.e., without the very last extension.
if [ $skip -eq 0 ]; then
    python $scriptdir/fasta2json.py -i "$infasta" -o "$outdir/json"
    if [ $? -ne 0 ]; then echo "Something went wrong..."; exit 1; fi
fi


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
printf "\nStep 2 - Generate feature embeddings with ESM-2\n"
mkdir -p "${outdir}"/features/{esm2,onehot,protTrans,evaluation}
# Note: Each sequence in the input fasta gets its own embedding.
if [ $skip -eq 0 ]; then
    python $scriptdir/emb_esm3_fasta.py -i "$infasta" -o "$outdir/features/esm2"
    if [ $? -ne 0 ]; then echo "Something went wrong..."; exit 1; fi
fi


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
printf "\nStep 3 - Predict solvent accessibility using pre-trained models\n"
if [ $rsa2c -eq 1 ]; then
    predmode="RSA_2C"
    SECONDS=0
    mkdir -p "$outdir/$predmode"
    
    printf "\nPrediction mode: 2-state\n"
    python $scriptdir/$predmode/new_test_ensemble.py -i "$outdir/json/$infasta_base.json" \
        -m "$predmode" -o "$outdir" -p "$outprefix"
    if [ $? -ne 0 ]; then echo "Something went wrong..."; exit 1; fi

    python parse_solvacc_pred.py -i "$outdir/$predmode/$outprefix.json" -m "$predmode" -e "$outext_2state"
    echo "2-state predictions completed in $SECONDS s for $seq_count sequences x $mean_length residues. Result:"
    echo "$outdir/$predmode/$outprefix.$outext_2state"
fi

if [ $rsa4c -eq 1 ]; then
    predmode="RSA_4C"
    SECONDS=0
    mkdir -p "$outdir/$predmode"

    printf "\nPrediction mode: 4-state\n"
    python $scriptdir/$predmode/new_test_ensemble.py -i "$outdir/json/$infasta_base.json" \
        -m "$predmode" -o "$outdir" -p "$outprefix"
    if [ $? -ne 0 ]; then echo "Something went wrong..."; exit 1; fi

    python parse_solvacc_pred.py -i "$outdir/$predmode/$outprefix.json" -m "$predmode" -e "$outext_4state"
    echo "4-state predictions completed in $SECONDS s for $seq_count sequences x $mean_length residues. Result:"
    echo "$outdir/$predmode/$outprefix.$outext_4state"
fi

if [ $rsarv -eq 1 ]; then
    predmode="RSA_realValue"
    SECONDS=0
    mkdir -p "$outdir/$predmode"

    printf "\nPrediction mode: Real-value\n"
    python $scriptdir/$predmode/new_test_ensemble.py -i "$outdir/json/$infasta_base.json" \
        -m "$predmode" -o "$outdir" -p "$outprefix"
    if [ $? -ne 0 ]; then echo "Something went wrong..."; exit 1; fi

    echo "Real-value predictions completed in $SECONDS s for $seq_count sequences x $mean_length residues. Result:"
    echo "$outdir/$predmode/$outprefix.json"
fi


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if [ $cleanup -eq 1 ]; then
    echo "Removing intermediate files..."
    find "$outdir/features/"{esm2,onehot,protTrans} -name "*.npy" -delete
    find "$outdir/json" -name "*.json" -delete
fi
echo "All steps completed successfully!"
