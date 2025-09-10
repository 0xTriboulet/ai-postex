'''Stand-alone helpper script to use as reference when implementing other encoding mechanics.'''

import string 

# tokenization
chars = set(''.join(string.printable)) # get the reference chars from printable chars
char2idx = {c: i+2 for i, c in enumerate(sorted(chars))}
char2idx['<PAD>'] = 0
char2idx['<UNK>'] = 1

min_len = 7
max_len = 32

# print(''.join(sorted(string.printable)))

def encode_word(w: str) -> list:
    encoded = [char2idx.get(ch, char2idx['<UNK>']) for ch in w]
    encoded = encoded[:max_len]
    if len(encoded) < max_len:
        encoded += [0]*(max_len - len(encoded)) # pad if our list is not the correct length
    return encoded
    
    
# print(encode_word("morning"))