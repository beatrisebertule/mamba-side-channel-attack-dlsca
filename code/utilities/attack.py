import numpy as np

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

def calculate_HW(data):
  """
  convert labels (sbox ouput values) to HW

  """
  hw = [bin(x).count("1") for x in range(256)]
  return [hw[int(s)] for s in data]


def score_keys(predictions, plaintext_byte, leakage_model):
    key_scores = np.zeros(256)                                 
    for k in range(256):                                        # loop over every possible key byte value k
        hypothetical_labels = AES_Sbox[plaintext_byte ^ k]                      # compute a hypothetical leakage label if the key byte were k for the known plaintext (target) byte
        if leakage_model == "HW":
            hypothetical_labels = calculate_HW(hypothetical_labels)
        
        for trace in range(predictions.shape[0]):                               # for every attack (test) trace trace
            key_scores[k] += predictions[trace, hypothetical_labels[trace]]            # accumulate the model’s probability assigned to the hypothetical label for a particular trace

    return key_scores

def guessing_entropy(predictions, plaintexts, correct_key, nb_traces, leakage_model, nb_attacks=100, eps=1e-36):
    ranks = np.zeros(nb_attacks) 
    predictions_log = np.log(predictions + eps)

    # take a random subset of traces
    for attack in range(nb_attacks):
        r = np.random.choice(
                range(predictions_log.shape[0]), nb_traces, replace=False)

        key_scores = score_keys(predictions_log[r], plaintexts[r], leakage_model)
        order_keys = np.argsort(key_scores)[::-1]
        ranks[attack] = np.where(order_keys == correct_key)[0][0]
    return np.median(ranks), np.average(ranks) 

def score_keys_convergence(predictions, plaintexts, leakage_model):
  scores_keys = np.zeros((256, predictions.shape[0]))           # scores_keys shape (256, nb_traces)
  for k in range(256):
    # Generate Hypothetical labels for a key-candidate k
    hypothetical_labels = AES_Sbox[plaintexts ^ k]              # hypothetical labels shape (nb_traces, )
    if leakage_model == "HW":
      hypothetical_labels = calculate_HW(hypothetical_labels)

    # handle the 1st trace, simply return the prediction
    scores_keys[k, 0] = predictions[0, hypothetical_labels[0]]
    for trace in range(1, predictions.shape[0]):  # how scores_keys evolve as more traces are evaluated, at some point signal should exceed the noise (?)
      scores_keys[k, trace] = scores_keys[k,trace-1] + predictions[trace, hypothetical_labels[trace]]     # predictions shape (nb_traces, 256)
  return scores_keys

def guessing_entropy_convergence(predictions, plaintexts, correct_key, leakage_model, nb_traces, nb_attacks=100):
  ranks = np.zeros((nb_attacks, nb_traces))       # ranks shape (100, 1000)

  # take log of probablities to sum later with small addition for numeric stability
  predictions_log = np.log(predictions + 1e-36)

  for attack in range(nb_attacks):     # repeat the attack several times (100 times) with a different subset of attack traces (bootstrapping - )
    # Take random subset of traces
    r = np.random.choice(
            range(predictions_log.shape[0]), nb_traces, replace=False)
    # print(predictions_log[r].shape)

    key_scores = score_keys_convergence(predictions_log[r], plaintexts[r], leakage_model)  # key_socres has shape of (256, nb_traces), where nb_traces hold the accumulated probabilities
    # print(key_scores.shape)

    for n in range(nb_traces):
      order_keys = np.argsort(key_scores[:, n])[::-1]                       # (guessing vector) key_scores[:, n] returns all 256 keys but only n-th trace, shape (256, )
      # print(order_keys[1])
      ranks[attack, n]  = np.where(order_keys == correct_key)[0][0]
  return np.median(ranks, axis=0), np.average(ranks, axis=0)                # compute the median and average rank of the correct key across all 100 attack repetitions for each trace count
  # median vector shape (nb_traces,) and average vector shape (nb_traces,) both hold convergabce rates - how the correct key's rank slowly move up the guessing vector
