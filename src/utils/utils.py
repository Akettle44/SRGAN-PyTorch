
import torch
import os

# Utility functions for bert pruning
class Utils():

    @staticmethod
    def loadHypsFromDisk(path):
        """ Load hyperparameters from disk

        Args:
            path (str): Path to hyperparameter file

        Returns:
            dict: Hyp name mapped to its value
        """

        hyps = {}
        if os.path.exists(path):
            # Read each line from file and store as hyp
            with open(path, 'r') as f:
                for line in f:
                    sp = line.split('=')
                    if len(sp) > 2:
                        exit(f"Hyperparameter file is ill-formatted")
                    key = sp[0].strip()
                    val,ty = sp[1].strip().split('$')
                    if ty == 'i':
                        val = int(val)
                    elif ty == 'f':
                        val = float(val)
                    elif ty == 'l':
                        items = val.strip('[]').split(',')
                        val = [int(item) for item in items]
                    else:
                        exit(f"Unsupported object type: {ty}")

                    hyps[key] = val

        else:
            exit(f"Hyperparameter file could not be found at {path}")

        return hyps