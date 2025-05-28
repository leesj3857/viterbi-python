# universal_hmm.py

from collections import defaultdict

def parse_bigram_file(bigram_path):
    bigram = defaultdict(dict)
    with open(bigram_path, 'r') as f:
        for line in f:
            from_word, to_word, prob = line.strip().split()
            bigram[from_word][to_word] = float(prob)
    return bigram

def build_universal_hmm(word_hmm_models, bigram_probs):
    global_states = []
    global_trans = []
    word_offsets = {}
    state_index = 0

    # 모든 단어 상태를 global state로 이어붙임
    for word, hmm in word_hmm_models.items():
        word_offsets[word] = state_index
        global_states.extend(hmm['states'])
        state_index += len(hmm['states'])

    total_states = len(global_states)
    global_trans = [[0.0 for _ in range(total_states)] for _ in range(total_states)]

    # 각 단어 내부 전이 복사
    for word, hmm in word_hmm_models.items():
        offset = word_offsets[word]
        local_trans = hmm['trans']
        for i in range(len(local_trans)):
            for j in range(len(local_trans[i])):
                global_trans[offset + i][offset + j] = local_trans[i][j]

    # 단어 간 전이 연결
    for from_word in bigram_probs:
        if from_word not in word_offsets:
            continue
        from_offset = word_offsets[from_word]
        from_len = len(word_hmm_models[from_word]['states'])
        from_last = from_offset + from_len - 1

        for to_word, prob in bigram_probs[from_word].items():
            if to_word not in word_offsets:
                continue
            to_offset = word_offsets[to_word]
            global_trans[from_last][to_offset] += prob

    for i, row in enumerate(global_trans):
        row_sum = sum(row)
        if row_sum > 0:
            global_trans[i] = [x / row_sum for x in row]
            
    return {
        'states': global_states,
        'trans': global_trans,
        'word_offsets': word_offsets
    }
