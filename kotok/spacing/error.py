import random

high_frequency_error = [
    '이',
    '마저',
    '에',
    '에서',
    '밖에',
    '부터',
    '만',
    '이나마',
    '입니',
    '처럼',
    '까지나',
    '도',
    '는',
    '만큼',
    '것이',
    '할',
    '수'
    '바',
    '지',
    '개',
    '명',
    '사람',
    '분',
    '가구',
    '쌍',
    '마리',
    '필',
    '두',
    '수',
    '암',
    '대',
    '권',
    '장',
    '병',
    '잔',
    '그릇',
    '판',
    '송이',
    '척',
    '채',
    '동',
    '자루',
    '가닥',
    '줄',
    '벌',
    '켤레',
    '조각',
    '통',
    '단',
    '리터',
    '미터',
    '센티미터',
    '킬로그램',
    '그램',
    '번',
    '차례',
    '살',
    '군데',
    '배',
    '바퀴',
]

def add_spacing_errors(text: str, add_spacing_chance: float=0.1, remove_spacing_chance: float=0.15, high_frequency_factor: float=3.0):
    # new_text -> text map
    new_text = ''
    offset_map = {}
    spacing_removed = set()
    spacing_added = set()

    i = 0
    while i < len(text):
        char = text[i]
        
        next_char = text[i + 1] if i + 1 < len(text) else ''
        if not next_char:
            offset_map[len(new_text)] = i
            new_text += char
            i += 1
            continue

        if next_char == ' ':
            chance = remove_spacing_chance

            after_space = text[i + 2] if i + 2 < len(text) else ''
            if after_space:
                for error in high_frequency_error:
                    if not after_space.startswith(error):
                        continue
                    chance *= high_frequency_factor
                    break
            
            if random.random() < chance:
                # remove the space
                offset_map[len(new_text)] = i
                new_text += char
                spacing_removed.add(i + 1)
                i += 2
                continue

        else:
            chance = add_spacing_chance
            next_text = text[i:i+1]

            for error in high_frequency_error:
                if not next_text.startswith(error):
                    continue
                chance *= high_frequency_factor
                break

            if random.random() < chance:
                # add a space
                offset_map[len(new_text)] = i
                offset_map[len(new_text) + 1] = i
                new_text += char + ' '
                spacing_added.add(i + 1)
                i += 1
                continue
        
        # just add the character
        offset_map[len(new_text)] = i
        new_text += char
        i += 1

    offset_map[len(new_text)] = len(text)

    return new_text, offset_map, spacing_removed, spacing_added


if __name__ == '__main__':
    import sys

    if sys.argv[1:]:
        text = ' '.join(sys.argv[1:])
        new_text, offset_map = add_spacing_errors(text)
    else:
        while True:
            try:
                text = input('> ')
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            new_text, offset_map = add_spacing_errors(text)
