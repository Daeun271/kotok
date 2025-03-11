# TODO:
#   Add more common typo patterns
#   Support single vocals/consonants via pre/post processing
#   Support latin characters

import os
from typing import Optional
import hangul_jamo

COST_KEY_ADJACENT = 1.25
COST_KEY_LAYER = 1.0
COST_SIMILAR_SOUND = 0.65
COST_COMBO_VOVEL_MISSING = 1.5
COST_COMBO_VOVEL_EXTRA = 1.75
COST_COMBO_BATCHIM_MISSING = 2.5
COST_COMBO_BATCHIM_EXTRA = 2.5
COST_BATCHIM_SHIFT = 0.5
COST_LENGTHEN = 0.25
COST_BATCHIM_REMOVE = 2.0

HANGUL_RANGES = [
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF)   # Hangul Jamo Extended-B
]

def is_hangul(char):
    return any(start <= ord(char) <= end for start, end in HANGUL_RANGES)

key_map_con = [
    ['ㅂㅃq', 'ㅈㅉw', 'ㄷㄸe', 'ㄱㄲr', 'ㅅㅆt'],
    ['ㅁㅁa', 'ㄴㄴs', 'ㅇㅇd', 'ㄹㄹf', 'ㅎㅎg'],
    ['ㅋㅋz', 'ㅌㅌc', 'ㅊㅊv', 'ㅍㅍb'],
]

key_map_vov = [
    ['ㅛㅛy', 'ㅕㅕu', 'ㅑㅑi', 'ㅐㅒo', 'ㅔㅖp'],
    ['ㅗㅗh', 'ㅓㅓj', 'ㅏㅏk', 'ㅣㅣl'],
    ['ㅠㅠb', 'ㅜㅜn', 'ㅡㅡm'],
]

similar_sound_con = [
    "ㄱㅋㄲ",  # Velar sounds
    "ㄷㅌㄸ",  # Alveolar stops
    "ㅂㅍㅃ",  # Bilabial stops
    "ㅈㅊㅉ",  # Affricates
    "ㅅㅆ",   # Sibilants
    "ㄴㄹ",   # Nasal and lateral sounds (sometimes confused in pronunciation)
    "ㅇㅎ",   # Glottal sounds
]

similar_sound_bat = [
    "ㄱㅋㄲㄳㄺ",  # Pronounced as [k]
    "ㄷㅌㅅㅆㅈㅊㅎ",  # Pronounced as [t]
    "ㅂㅍㅄ",  # Pronounced as [p]
    "ㄴㄵㄶ",  # Pronounced as [n]
    "ㅁㄻ",  # Pronounced as [m]
    "ㄹㄽㄾㄿㅀ",  # Pronounced as [l]
]

similar_sound_vov = [
    "ㅏㅓ",   # Similar open vowels
    "ㅏㅑ",
    "ㅓㅕ",
    "ㅑㅕ",   # Similar open vowels with y-sound
    "ㅗㅛ",   # Similar rounded vowels
    "ㅜㅠ",
    "ㅛㅠ",   # Similar rounded vowels with y-sound
    "ㅡㅜㅗ",   # Close back vowels
    "ㅐㅔ",   # Similar mid vowels
    "ㅐㅒ",
    "ㅔㅖ",
    "ㅒㅖ"    # Similar mid vowels with y-sound
]

batchim_split = {
    'ㄳ': 'ㄱㅅ',
    'ㄵ': 'ㄴㅈ',
    'ㄶ': 'ㄴㅎ',
    'ㄺ': 'ㄹㄱ',
    'ㄻ': 'ㄹㅁ',
    'ㄼ': 'ㄹㅂ',
    'ㄽ': 'ㄹㅅ',
    'ㄾ': 'ㄹㅌ',
    'ㄿ': 'ㄹㅍ',
    'ㅀ': 'ㄹㅎ',
    'ㅄ': 'ㅂㅅ',
}

batchim_split_inv = {v: k for k, v in batchim_split.items()}

vovel_split = {
    'ㅘ': 'ㅗㅏ',
    'ㅙ': 'ㅗㅐ',
    'ㅚ': 'ㅗㅣ',
    'ㅝ': 'ㅜㅓ',
    'ㅞ': 'ㅜㅔ',
    'ㅟ': 'ㅜㅣ',
    'ㅢ': 'ㅡㅣ',
}

vovel_split_inv = {v: k for k, v in vovel_split.items()}

def combine_cost(cost1, cost2):
    if cost2 is None:
        return cost1
    if cost1 is None:
        return cost2
    return min(cost1, cost2)

def add_cost(dict, jamo_from, jamo_to, cost):
    if jamo_from not in dict:
        dict[jamo_from] = {}
    dict[jamo_from][jamo_to] = combine_cost(dict[jamo_from].get(jamo_to, None), cost)

def add_cost_dict(dict, dict_add):
    for jamo_from, jamo_to_dict in dict_add.items():
        for jamo_to, cost in jamo_to_dict.items():
            add_cost(dict, jamo_from, jamo_to, cost)

def gen_keyboard_cost(map):
    dist = {}
    
    for y in range(len(map)):
        for x in range(len(map[y])):
            this_key = map[y][x]

            this_a, this_b, _this_c = this_key
            dist[this_a] = {}
            dist[this_b] = {}

            unique_keys = list(set(this_key))
            for a in unique_keys:
                for b in unique_keys:
                    if a == b or not is_hangul(a) or not is_hangul(b):
                        continue
                    add_cost(dist, a, b, COST_KEY_LAYER)

            other_indices = [
                (y, x - 1), # left
                (y, x + 1), # right
                (y - 1, x), # top
                (y + 1, x), # bottom
                (y - 1, x - 1), # top left
                (y - 1, x + 1), # top right
            ]

            for oy, ox in other_indices:
                if oy < 0 or oy >= len(map) or ox < 0 or ox >= len(map[oy]):
                    continue

                other_key = map[oy][ox]

                for (this_char, other_char) in zip(this_key, other_key):
                    if not is_hangul(this_char) or not is_hangul(other_char):
                        continue
                    add_cost(dist, this_char, other_char, COST_KEY_ADJACENT)

    return dist


def gen_similar_sound_cost(similar_sound):
    dist = {}
    for group in similar_sound:
        for jamo in group:
            for other_jamo in group:
                if jamo == other_jamo:
                    continue
                add_cost(dist, jamo, other_jamo, COST_SIMILAR_SOUND)
    return dist

def gen_double_vovel_cost():
    dist = {}
    for double_vovel, split in vovel_split.items():
        for single_vovel in split:
            add_cost(dist, double_vovel, single_vovel, COST_COMBO_VOVEL_EXTRA)
            add_cost(dist, single_vovel, double_vovel, COST_COMBO_VOVEL_MISSING)
    return dist

def gen_double_batchim_cost():
    dist = {}
    for double_batchim, split in batchim_split.items():
        for single_batchim in split:
            add_cost(dist, double_batchim, single_batchim, COST_COMBO_BATCHIM_EXTRA)
            add_cost(dist, single_batchim, double_batchim, COST_COMBO_BATCHIM_MISSING)
    return dist 


swap_costs_con = {}
swap_costs_vov = {}
swap_costs_bat = {}


add_cost_dict(swap_costs_con, gen_keyboard_cost(key_map_con))
add_cost_dict(swap_costs_con, gen_similar_sound_cost(similar_sound_con))
add_cost_dict(swap_costs_vov, gen_keyboard_cost(key_map_vov))
add_cost_dict(swap_costs_vov, gen_similar_sound_cost(similar_sound_vov))
add_cost_dict(swap_costs_vov, gen_double_vovel_cost())
add_cost_dict(swap_costs_bat, gen_keyboard_cost(key_map_con))
add_cost_dict(swap_costs_bat, gen_similar_sound_cost(similar_sound_bat))
add_cost_dict(swap_costs_bat, gen_double_batchim_cost())


# shifts in both directions
def batchim_shift(s1, s2):
    (c1, v1, b1) = s1
    (c2, v2, b2) = s2

    r = []

    # shift batchim to the next consonant
    if c2 == 'ㅇ' and b1:
        if b1 in batchim_split:
            b1, c2 = batchim_split[b1]
            r.append(((c1, v1, b1), (c2, v2, b2)))
        else:
            r.append(((c1, v1, None), (b1, v2, b2)))
    
    # recombine batchim
    if b1 and c2:
        combined = b1 + c2
        if combined in batchim_split_inv:
            b1 = batchim_split_inv[combined]
            r.append(((c1, v1, b1), ('ㅇ', v2, b2)))
        
    if not b1 and c2:
        r.append(((c1, v1, c2), ('ㅇ', v2, b2)))

    return r


def lengthening_add(s):
    # returns possible lengthening syllable
    (_c, v, b) = s
    if b:
        return None
    return ('ㅇ', v, None)

def lengthening_remove(s1, s2):
    # returns True if s2 can be removed
    (_c1, v1, _b1) = s1
    (c2, v2, _b2) = s2

    if c2 != 'ㅇ' or v1 != v2:
        return False
    
    return True

def batchim_remove(s):
    # returns possible batchim removed syllable
    (c, v, b) = s
    if not b:
        return None
    return (c, v, None)

def typo(syllables: tuple[tuple[str, str, Optional[str]]], max_depth = 5, max_cost = 4.5, candidates = None, current_cost = 0.0):
    if candidates is None:
        candidates = {}
    if max_depth == 0 or current_cost > max_cost:
        return candidates
    
    def add_candidate(syllables, cost):
        actual_cost = current_cost + cost
        if actual_cost > max_cost:
            return
        if syllables not in candidates or candidates[syllables] > actual_cost:
            candidates[syllables] = actual_cost
            typo(syllables, max_depth - 1, max_cost, candidates, actual_cost)


    num_s = len(syllables)
    for i, s in enumerate(syllables):
        s_next = syllables[i+1] if i < num_s-1 else None

        # Attempt Batchim shift
        if s_next:
            for s_pair in batchim_shift(s, s_next):
                candidate = syllables[:i] + s_pair + syllables[i + 2:]
                add_candidate(candidate, COST_BATCHIM_SHIFT)

        # Attempt lengthening removal
        if s_next and lengthening_remove(s, s_next):
            candidate = syllables[:i] + syllables[i+1:]
            add_candidate(candidate, COST_LENGTHEN)

        # Attempt lengthening
        s_lengthening = lengthening_add(s)
        if s_lengthening:
            candidate = syllables[:i] + (s, s_lengthening) + syllables[i+1:]
            add_candidate(candidate, COST_LENGTHEN)

        # Attempt batchim removal
        s_batchim_removed = batchim_remove(s)
        if s_batchim_removed:
            candidate = syllables[:i] + (s_batchim_removed,) + syllables[i+1:]
            add_candidate(candidate, COST_BATCHIM_REMOVE)

        c, v, b = s

        # Consonant typos
        if c in swap_costs_con:
            for c2, cost in swap_costs_con[c].items():
                candidate = syllables[:i] + ((c2, v, b),) + syllables[i+1:]
                add_candidate(candidate, cost)

        # Vowel typos
        if v in swap_costs_vov:
            for v2, cost in swap_costs_vov[v].items():
                candidate = syllables[:i] + ((c, v2, b),) + syllables[i+1:]
                add_candidate(candidate, cost)

        # Batchim typos
        if b in swap_costs_bat:
            for b2, cost in swap_costs_bat[b].items():
                candidate = syllables[:i] + ((c, v, b2),) + syllables[i+1:]
                add_candidate(candidate, cost)

    return candidates


lengthenable_jamo = {
    'ㅏ',
    'ㅓ',
    'ㅗ',
    'ㅜ',
    'ㅡ',
    'ㅣ',
}

def typo_text(text: str, max_depth = 4, max_cost = 4.0, return_suffix = False):
    suffix = ''
    syllables = []

    for char in text:
        try:
            jamo = hangul_jamo.decompose_syllable(char)
            syllables.append(jamo)
        except ValueError:
            if char in lengthenable_jamo:
                syllables.append(('ㅇ', char, None))
            else:
                break
    suffix = text[len(syllables):]

    result = typo(tuple(syllables), max_depth, max_cost).items()
    candidates = {}

    for syllables, cost in result:
        try:
            typoed_text = ''.join(hangul_jamo.compose_jamo_characters(*s) for s in syllables)
        except ValueError:
            # invalid jamo combination
            continue
        if return_suffix:
            candidates[typoed_text] = cost
        else:
            candidates[typoed_text + suffix] = cost
    
    if return_suffix:
        return candidates, suffix
    return candidates


class TypoCorrector:
    def __init__(self, correction_data_path: str):
        self.correction_data_path = correction_data_path
        self.load_correction_data()

    def load_correction_data(self):
        self.clean_data = {}
        clean_path = os.path.join(self.correction_data_path, 'clean.txt')

        with open(clean_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean, freq = line.strip().split()
                self.clean_data[clean] = int(freq)

    def correct(self, text: str, max_depth = 4, max_cost = 4.0):
        candidates, suffix = typo_text(text, max_depth, max_cost, return_suffix=True)
        r = []
        for candidate, cost in candidates.items():
            if not candidate in self.clean_data:
                continue
            r.append((candidate + suffix, cost))
        return r


def main():
    while True:
        try:
            text = input('> ')
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        text = text.strip()
        
        to_text = None
        colon_idx = text.find(':')
        if colon_idx != -1:
            to_text = text[colon_idx+1:].lstrip()
            text = text[:colon_idx].rstrip()
        
        candidates = typo_text(text)
        candidates_sorted = sorted(candidates.items(), key=lambda x: -x[1])

        for candidate, cost in candidates_sorted:
            if to_text:
                if candidate == to_text:
                    print(f'{candidate} ({cost})')
                    break
            else:
                print(f'{candidate} ({cost})')
        else:
            if to_text:
                print(f'{to_text} (not found)')


if __name__ == '__main__':
    main()
