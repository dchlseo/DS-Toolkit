# NUMPY VERSION OF SINUSOIDAL POSITIONAL ENCODING

import numpy as np

def positional_encoding(max_position, d_model, min_freq=1e-4):
    """
    Generates a matrix of sinusoidal positional encodings.

    Parameters:
    - max_position (int): The maximum position index, i.e., the maximum length of the sequence.
    - d_model (int): The dimensionality of the model's embeddings; the number of dimensions 
                      in the positional encoding vectors.
    - min_freq (float): The minimum frequency to use for the sinusoidal calculations. It is 
                        used as a base for the geometric decay of frequencies along the 
                        dimensions. Default is 1e-4.

    Returns:
    - pos_enc (np.array): A matrix of shape (max_position, d_model) where each row corresponds 
                          to a position in the sequence and each column corresponds to the 
                          positional encoding for that dimension.
    """
    
    # Create an array of position indices (0 to max_position-1)
    position = np.arange(max_position)
    
    # Generate the frequencies for the sinusoidal functions
    # Using a geometric progression from 1 to min_freq, with a total number of steps equal to d_model//2
    # The exponentiation by 2 and division by d_model ensures that the frequencies decay
    # geometrically from the highest frequency (1) to the lowest frequency (min_freq).
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    
    # Calculate the arguments for the sinusoidal functions
    # By multiplying the position indices with the frequencies, we effectively create a
    # "scaled time" for each dimension, where each dimension oscillates at its own frequency.
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    
    # Apply the cosine function to even indices (0, 2, 4, ...)
    # This fills the even-indexed columns of the pos_enc matrix with the cosine of the scaled times.
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    
    # Apply the sine function to odd indices (1, 3, 5, ...)
    # This fills the odd-indexed columns of the pos_enc matrix with the sine of the scaled times.
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    
    # Return the matrix of positional encodings.
    return pos_enc
