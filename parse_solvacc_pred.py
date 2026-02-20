# Author: Hieu Minh Truong
import os
import argparse
import json


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def solvacc_2(tendency):
    tendency = float(tendency)
    if tendency < 0.25:
        return 0 # Buried
    else:
        return 1 # Exposed

def solvacc_4(tendency_list):
    return tendency_list.index(max(tendency_list))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Receive file paths from the wrapper script
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--injson', required=True)
parser.add_argument('-m', '--mode', required=True, choices=['RSA_2C', 'RSA_4C'])
parser.add_argument('-p', '--outprefix')
parser.add_argument('-e', '--outextension', help='Default: rsa2c (2-state), rsa4c (4-state).')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()
default_extension = {
    'RSA_2C' : 'rsa2c',
    'RSA_4C' : 'rsa4c'
}

if args.mode is None:
    raise ValueError('Please select a prediction mode to parse (RSA_2C, RSA_4C).')
else:
    predmode = args.mode

injson = os.path.abspath(args.injson)
injson_base = os.path.splitext(injson)[0]

if args.outprefix is None:
    outprefix = injson_base
else:
    outprefix = args.outprefix

if args.outextension is None:
    outextension = default_extension[args.outextension]
else:
    outextension = '.' + args.outextension
outsequences = outprefix + outextension
with open(outsequences, 'w') as handle:
    pass


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read solvent accessibility tendencies
with open(injson) as handle:
    predictions = json.loads(handle.read())

# Convert tendencies to character states
for record in predictions:
    if predmode == 'RSA_2C':
        record['solvacc'] = [solvacc_2(x) for x in record['true_pred']]
    if predmode == 'RSA_4C':
        record['solvacc'] = [solvacc_4(x) for x in record['true_pred']]
    message = ">{}\n{}\n".format(record['id'], "".join(str(x) for x in record['solvacc']))
    with open(outsequences, 'a') as handle:
        handle.write(message)
    if args.verbose:
        print("{:<45}{:5}".format(record['id'], record['seq_len']))
        #print(">", record['id'], "\n", *record['solvacc'], sep='')


