def fasta_reader(filepath):
    """
    Reads in a fasta file located at the filepath and returns a list of the sequences in the file

    Arguments:
        filepath::str
            The path to the fasta file that is being read
    
    Returns:
        sequences::[string]
            List of strings representing the DNA sequences encoded in the fasta file
    """
    sequences = []
    with open(filepath, "r") as infile:
        seq = ''
        for i, line in enumerate(infile):
            if line.startswith(">"):
                if i != 0:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line.strip().upper()
        sequences.append(seq)
    
    return sequences


def line_reader(filepath):
    """
    Reads in a sequence file holding one sequence per line. Returns a list of these sequences
    
    Arguments:
        filepath::str
            The path to the sequence file that is being read
    
    Returns:
        sequences::[string]
            List of strings representing the DNA sequences encoded in the sequence file
    """
    sequences = []
    with open(filepath, "r") as infile:
        for line in infile:
            sequences.append(line.strip().upper())
    
    return sequences

def write_output(outfile_path, sequences, predictions):
    """
    Specific function for writing out the data for part 5 in the requested format (2 columns, first column is the sequence, second is the predicted value)
    
    Arguments:
        outfile_path::str
            The desired path of the output file
        sequences::[str]
            List of the sequences (in DNA string format) that have had predictions performed for them.
        predictions::[float]
            List of predicted values for the sequences in "sequences". Must be in the same order and the same length of the sequences list

    Returns:
        None
    """
    with open(outfile_path, 'w') as f:
        for seq, pred in zip(sequences,predictions):
            f.write(seq+'\t'+str(pred)+'\n')
    