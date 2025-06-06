import re
from collections import defaultdict
from build_universal import parse_bigram_file, build_universal_hmm
import numpy as np
import math
import os

def parse_hmm_file(hmm_path):
    hmm_dict = {}
    with open(hmm_path, 'r') as f:
        content = f.read()
    
    hmm_blocks = re.findall(r'~h "(.*?)"\s+<BEGINHMM>(.*?)<ENDHMM>', content, re.DOTALL)
    
    for phoneme, body in hmm_blocks:
        lines = body.strip().split('\n')
        num_states = int(re.search(r'<NUMSTATES> (\d+)', body).group(1))
        states = defaultdict(list)
        trans_matrix = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<STATE>'):
                state_id = int(line.split()[1])
                i += 1
                mixes = []
                num_mixes = int(lines[i].strip().split()[1])
                i += 1
                for _ in range(num_mixes):
                    mix_line = lines[i].strip()
                    _, weight = mix_line.split()[1:]
                    weight = float(weight)
                    i += 1
                    dim = int(lines[i].strip().split()[1])
                    i += 1
                    mean_vals = list(map(float, lines[i].strip().split()))
                    i += 1
                    dim_check = int(lines[i].strip().split()[1])
                    i += 1
                    var_vals = list(map(float, lines[i].strip().split()))
                    i += 1
                    mixes.append({
                        'weight': weight,
                        'mean': mean_vals,
                        'variance': var_vals
                    })
                states[state_id] = mixes
            elif line.startswith('<TRANSP>'):
                size = int(line.split()[1])
                for _ in range(size):
                    i += 1
                    trans_matrix.append(list(map(float, lines[i].strip().split())))
            else:
                i += 1

        hmm_dict[phoneme] = {
            'num_states': num_states,
            'states': states,
            'trans': trans_matrix
        }
    return hmm_dict

def parse_dictionary_file(dict_path):
    word_dict = {}
    with open(dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                phonemes = parts[1:]
                word_dict[word] = phonemes
    return word_dict

def build_word_hmm(word_dict, phoneme_hmms):
    word_hmms = {}
    for word, phonemes in word_dict.items():
        all_states = []
        trans_blocks = []
        state_map = []

        state_index = 0
        for phoneme in phonemes:
            hmm = phoneme_hmms[phoneme]
            num_states = hmm['num_states']  # <NUMSTATES> N
            states = hmm['states']
            trans = hmm['trans']  # (N+2) x (N+2)

            # 실제 내부 상태: state 2 ~ state N-1 (총 N-2개)
            internal_state_ids = list(range(2, num_states))
            for s_id in internal_state_ids:
                all_states.append(states[s_id])

            # 내부 상태 간 전이만 추출 (N-2) x (N-2)
            internal_trans = []
            for i in range(2, num_states):
                row = []
                for j in range(2, num_states):
                    row.append(trans[i][j])
                internal_trans.append(row)

            trans_blocks.append({
                'trans': internal_trans,
                'exit_prob': trans[num_states-1][num_states],  # 마지막 상태에서 종료 상태로 가는 확률
                'entry_prob': trans[1][2]  # 시작 상태에서 첫 상태로 가는 확률
            })
            state_map.append(state_index)
            state_index += len(internal_state_ids)

        # 전체 전이행렬 초기화
        total_states = len(all_states)
        total_trans = [[0.0 for _ in range(total_states)] for _ in range(total_states)]

        # 각 음소의 전이행렬을 해당 위치에 삽입
        for block, start in zip(trans_blocks, state_map):
            for i in range(len(block['trans'])):
                for j in range(len(block['trans'][i])):
                    total_trans[start + i][start + j] = block['trans'][i][j]

        # 음소 사이 연결: 이전 음소 마지막 상태 → 다음 음소 첫 상태
        for i in range(len(state_map) - 1):
            if len(trans_blocks[i]['trans']) == 0 or len(trans_blocks[i + 1]['trans']) == 0:
                continue  # 내부 상태가 없으면 skip
            prev_last = state_map[i] + len(trans_blocks[i]['trans']) - 1
            next_first = state_map[i + 1]
            # 이전 음소의 종료 확률과 다음 음소의 시작 확률을 곱함
            total_trans[prev_last][next_first] = trans_blocks[i]['exit_prob'] * trans_blocks[i + 1]['entry_prob']

        word_hmms[word] = {
            'states': all_states,
            'trans': total_trans
        }
    return word_hmms









LOG_ZERO = -float('inf')

def read_mfcc_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    n_rows, n_dims = map(int, lines[0].split())
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    return np.array(data)  # shape: (n_rows, n_dims)

def log_dot(vec, mixtures):
    total = 0.0
    for m in mixtures:
        diff = np.array(vec) - np.array(m['mean'])
        exponent = -0.5 * np.dot(diff, diff) / np.array(m['variance']).mean()
        total += m['weight'] * np.exp(exponent)
    return np.log(total + 1e-10)  # ← small constant to prevent log(0)

def viterbi(observations, hmm):
    N = len(hmm['states'])
    T = len(observations)
    trans = hmm['trans']
    emit = hmm['states']
    lambda1 = 0.2  # emission weight
    lambda2 = 1.0  # transition weight
    delta = [[LOG_ZERO for _ in range(N)] for _ in range(T)]
    psi = [[-1 for _ in range(N)] for _ in range(T)]

    for i in range(N):
        delta[0][i] = log_dot(observations[0], emit[i])

    for t in range(1, T):
        for j in range(N):
            max_prob, max_state = LOG_ZERO, -1
            for i in range(N):
                if trans[i][j] > 0:
                    prob = delta[t-1][i] + lambda2 * math.log(trans[i][j])
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
            delta[t][j] = max_prob + lambda1 * log_dot(observations[t], emit[j])
            psi[t][j] = max_state

    last_state = max(range(N), key=lambda i: delta[T-1][i])
    path = [last_state]
    for t in range(T-1, 0, -1):
        last_state = psi[t][last_state]
        path.append(last_state)
    path.reverse()
    return path

def state_seq_to_word_seq(state_seq, word_offsets, word_hmms, min_duration=15):
    sorted_offsets = sorted(word_offsets.items(), key=lambda x: x[1])
    word_seq = []
    last_word = None
    word_duration = 0
    last_state = -1

    for s in state_seq:
        current_word = None
        for word, offset in sorted_offsets:
            length = len(word_hmms[word]['states'])
            if offset <= s < offset + length:
                current_word = word
                break

        if current_word == last_word:
            word_duration += 1
        else:
            # 이전 단어가 있고, 최소 지속 시간을 만족하면 추가
            if last_word and word_duration >= min_duration:
                word_seq.append(last_word)
            last_word = current_word
            word_duration = 1
            last_state = s

    # 마지막 단어 처리
    if last_word and word_duration >= min_duration:
        word_seq.append(last_word)

    return word_seq

def collect_txt_files(root_dir, limit=100):
    txt_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
                if len(txt_files) >= limit:
                    return txt_files
    return txt_files








# === 실행 ===
if __name__ == "__main__":
    hmm_path = 'hmm.txt'
    dict_path = 'dictionary.txt'

    phoneme_hmms = parse_hmm_file(hmm_path)
    dictionary = parse_dictionary_file(dict_path)
    word_hmm_models = build_word_hmm(dictionary, phoneme_hmms)
    bigram_probs = parse_bigram_file('bigram.txt')
    universal_hmm = build_universal_hmm(word_hmm_models, bigram_probs)
    # obs = read_mfcc_file("test.txt")
    # state_seq = viterbi(obs, universal_hmm)

    # # Convert to word sequence
    # word_seq = state_seq_to_word_seq(state_seq, universal_hmm['word_offsets'])

    mfcc_files = collect_txt_files("mfc", limit=100)

    # MLF 생성
    with open("recognized.txt", "w") as f:
        f.write("#!MLF!#\n")
        for filepath in mfcc_files:
            obs = read_mfcc_file(filepath)
            state_seq = viterbi(obs, universal_hmm)
            word_seq = state_seq_to_word_seq(state_seq, universal_hmm['word_offsets'], word_hmm_models)
            cleaned_word_seq = [w for w in word_seq if w != '']
            rel_path = filepath.replace("\\", "/").replace(".txt", ".rec")
            f.write(f"\"{rel_path}\"\n")
            for word in cleaned_word_seq:
                f.write(word + "\n")
            f.write(".\n")

    print("✅ MLF format decoding complete → saved to recognized.txt")