import random
import hangul_jamo

class TypoGenerator:
    def __init__(self, char_typo_probability=0.3, word_typo_probability=0.5, multiple_component_chance=0.4):
        """
        Initialize the Korean typo generator with customizable parameters.
        
        Args:
            char_typo_probability (float): Probability of introducing a typo for each character (0-1)
            word_typo_probability (float): Probability of replacing known words with common typos (0-1)
            multiple_component_chance (float): Chance to modify multiple components of a single character (0-1)
        """
        self.char_typo_probability = char_typo_probability
        self.word_typo_probability = word_typo_probability
        self.multiple_component_chance = multiple_component_chance
        self.initialize_typo_sets()
        
    def initialize_typo_sets(self):
        """Initialize the typo sets based on the reference data."""
        # Common vowel confusions
        self.vowel_typos = {
            'ㅏ': ['ㅑ', 'ㅓ'],
            'ㅑ': ['ㅏ', 'ㅕ'],
            'ㅓ': ['ㅏ', 'ㅕ'],
            'ㅕ': ['ㅑ', 'ㅓ'],
            'ㅗ': ['ㅛ', 'ㅜ'],
            'ㅛ': ['ㅗ', 'ㅠ'],
            'ㅜ': ['ㅗ', 'ㅠ'],
            'ㅠ': ['ㅛ', 'ㅜ'],
            'ㅐ': ['ㅔ', 'ㅒ'],
            'ㅔ': ['ㅐ', 'ㅖ'],
            'ㅒ': ['ㅐ', 'ㅖ'],
            'ㅖ': ['ㅔ', 'ㅒ'],
            'ㅘ': ['ㅝ', 'ㅙ'],
            'ㅝ': ['ㅘ', 'ㅞ', 'ㅗ', 'ㅓ'],
            'ㅙ': ['ㅘ', 'ㅚ', 'ㅞ', 'ㅐ', 'ㅔ'],
            'ㅞ': ['ㅝ', 'ㅙ', 'ㅚ', 'ㅐ', 'ㅔ'],
            'ㅚ': ['ㅙ', 'ㅞ', 'ㅐ', 'ㅔ'],
            'ㅟ': ['ㅣ', 'ㅜ'],
            'ㅢ': ['ㅣ', 'ㅡ'],
            'ㅡ': ['ㅣ', 'ㅜ', 'ㅓ'],
            'ㅣ': ['ㅡ', 'ㅟ', 'ㅢ'],
        }
        
        # Consonant confusions (initial consonants)
        self.consonant_typos = {
            'ᄀ': ['ᄁ', 'ᄂ', 'ᄏ'],
            'ᄁ': ['ᄀ', 'ᄃ'],
            'ᄂ': ['ᄀ', 'ᄃ', 'ᄆ'],
            'ᄃ': ['ᄄ', 'ᄂ', 'ᄐ'],
            'ᄄ': ['ᄃ', 'ᄌ'],
            'ᄅ': ['ᄂ', 'ᄆ'],
            'ᄆ': ['ᄂ', 'ᄇ', 'ᄅ'],
            'ᄇ': ['ᄈ', 'ᄆ', 'ᄑ'],
            'ᄈ': ['ᄇ', 'ᄊ'],
            'ᄉ': ['ᄊ', 'ᄌ'],
            'ᄊ': ['ᄉ', 'ᄍ'],
            'ᄌ': ['ᄍ', 'ᄎ', 'ᄉ'],
            'ᄍ': ['ᄌ', 'ᄊ'],
            'ᄎ': ['ᄌ', 'ᄏ', 'ᄐ'],
            'ᄏ': ['ᄀ', 'ᄎ', 'ᄐ'],
            'ᄐ': ['ᄃ', 'ᄏ', 'ᄑ'],
            'ᄑ': ['ᄇ', 'ᄐ'],
            'ᄒ': ['ᄋ'],
            'ᄋ': ['ᄒ'],
        }
        
        # Trailing consonant (batchim) confusions
        self.batchim_typos = {
            'ᆨ': ['ᆩ', 'ᆪ', 'ᆮ', 'ᆺ', 'ᆿ', ''],
            'ᆩ': ['ᆨ', 'ᆪ', ''],
            'ᆪ': ['ᆨ', 'ᆩ', 'ᆺ', ''],
            'ᆫ': ['ᆬ', 'ᆭ', 'ᆮ', 'ᆷ', ''],
            'ᆬ': ['ᆫ', 'ᆭ', 'ᆽ', ''],
            'ᆭ': ['ᆫ', 'ᆬ', 'ᇂ', ''],
            'ᆮ': ['ᆨ', 'ᆺ', 'ᆽ', 'ᇀ', ''],
            'ᆯ': ['ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆫ', ''],
            'ᆰ': ['ᆯ', 'ᆨ', ''],
            'ᆱ': ['ᆯ', 'ᆷ', ''],
            'ᆲ': ['ᆯ', 'ᆸ', ''],
            'ᆳ': ['ᆯ', 'ᆺ', ''],
            'ᆴ': ['ᆯ', 'ᇀ', ''],
            'ᆵ': ['ᆯ', 'ᇁ', ''],
            'ᆶ': ['ᆯ', 'ᇂ', ''],
            'ᆷ': ['ᆫ', 'ᆸ', 'ᆱ', ''],
            'ᆸ': ['ᆷ', 'ᆹ', 'ᇁ', ''],
            'ᆹ': ['ᆸ', 'ᆺ', ''],
            'ᆺ': ['ᆻ', 'ᆮ', 'ᆽ', 'ᆨ', ''],
            'ᆻ': ['ᆺ', ''],
            'ᆽ': ['ᆺ', 'ᆮ', 'ᆾ', ''],
            'ᆾ': ['ᆽ', 'ᇀ', ''],
            'ᆿ': ['ᆨ', 'ᇀ', ''],
            'ᇀ': ['ᆮ', 'ᆾ', 'ᆿ', ''],
            'ᇁ': ['ᆸ', 'ᆵ', ''],
            'ᇂ': ['ᆭ', 'ᆶ', ''],
            '': ['ᆨ', 'ᆫ', 'ᆮ', 'ᆯ', 'ᆷ', 'ᆸ', 'ᆺ', 'ᆽ', 'ᇂ'],
        }
        
        # Common word typos
        self.word_typos = {
            '안': ['않'],
            '않': ['안'],
            '던': ['든'],
            '든': ['던'],
            '때': ['데'],
            '데': ['때'],
            '빛': ['빚'],
            '빚': ['빛'],
            '맞추': ['맞히'],
            '맞히': ['맞추'],
            '맞춰': ['맞혀'],
            '맞혀': ['맞춰'],
            '받치': ['바치', '받히'],
            '바치': ['받치', '받히'],
            '받히': ['받치', '바치'],
            '자': ['쟈'],
            '쟈': ['자'],
            '저': ['져'],
            '져': ['저'],
            '제': ['졔'],
            '졔': ['제'],
            '조': ['죠', '줘'],
            '죠': ['조', '줘'],
            '줘': ['조', '죠'],
            '주': ['쥬'],
            '쥬': ['주'],
            '므': ['무'],
            '무': ['므'],
            '브': ['부'],
            '부': ['브'],
            '프': ['푸'],
            '푸': ['프'],
            '르': ['루'],
            '루': ['르'],
            '러': ['뤄'],
            '뤄': ['러'],
        }
        
        # Article/particle typos
        self.article_typos = {
            '은': ['는'],
            '는': ['은'],
            '을': ['를'],
            '를': ['을'],
            '이': ['가'],
            '가': ['이'],
        }

        # Character swaps mapping - characters that might be swapped by mistake
        self.swap_candidates = [
            ('ㅏ', 'ㅓ'), ('ㅑ', 'ㅕ'), ('ㅗ', 'ㅜ'), ('ㅛ', 'ㅠ'), 
            ('ㅐ', 'ㅔ'), ('ㅒ', 'ㅖ'), ('ㄱ', 'ㅋ'), ('ㄷ', 'ㅌ'),
            ('ㅂ', 'ㅍ'), ('ㅅ', 'ㅎ'), ('ㅈ', 'ㅊ'), ('ㄴ', 'ㄹ'),
            ('ㅁ', 'ㅇ'), ('ㅏ', 'ㅑ'), ('ㅓ', 'ㅕ'), ('ㅗ', 'ㅛ'),
            ('ㅜ', 'ㅠ'), ('ㅣ', 'ㅡ')
        ]
        
    def get_random_typo(self, char_type, jamo):
        """Get a random typo substitution for the given jamo based on its type."""
        if char_type == "LEADING" and jamo in self.consonant_typos:
            return random.choice(self.consonant_typos[jamo])
        elif char_type == "VOWEL" and jamo in self.vowel_typos:
            return random.choice(self.vowel_typos[jamo])
        elif char_type == "TRAILING" and jamo in self.batchim_typos:
            return random.choice(self.batchim_typos[jamo])
        return jamo
    
    def should_apply_typo(self, probability=None):
        """Determine if a typo should be applied based on the probability."""
        if probability is None:
            probability = self.char_typo_probability
        return random.random() < probability
    
    def find_word_typos(self, text):
        """Find all possible word-level typo replacements in the text."""
        replacements = []
        
        # Check for whole words
        for word, typos in self.word_typos.items():
            start = 0
            while True:
                start = text.find(word, start)
                if start == -1:
                    break
                if self.should_apply_typo(self.word_typo_probability):
                    replacements.append((start, len(word), random.choice(typos)))
                start += len(word)
        
        # Check for particles
        for particle, typos in self.article_typos.items():
            start = 0
            while True:
                start = text.find(particle, start)
                if start == -1:
                    break
                
                # Only replace if it looks like a particle (preceded by a Hangul character)
                if start > 0 and 0xAC00 <= ord(text[start-1]) <= 0xD7A3:
                    if self.should_apply_typo(self.word_typo_probability):
                        replacements.append((start, len(particle), random.choice(typos)))
                start += len(particle)
        
        # Sort by position in descending order to apply replacements from right to left
        # (so earlier replacements don't affect positions of later ones)
        return sorted(replacements, key=lambda x: x[0], reverse=True)
    
    def apply_word_level_typos(self, text):
        """Apply word-level typos to the text."""
        replacements = self.find_word_typos(text)
        
        # Apply replacements
        for start, length, replacement in replacements:
            text = text[:start] + replacement + text[start+length:]
        
        return text
    
    def apply_character_typos(self, text):
        """Apply character-level typos to Korean text."""
        result = []
        
        for char in text:
            # Skip non-Hangul characters
            if not (ord('가') <= ord(char) <= ord('힣')):
                result.append(char)
                continue
            
            # Apply typo with probability
            if self.should_apply_typo():
                # Decompose the character
                leading, vowel, trailing = hangul_jamo.decompose_syllable(char)
                
                # Possibly modify multiple components
                components_to_alter = []
                
                # Always modify at least one component
                components_to_alter.append(random.choice(['LEADING', 'VOWEL', 'TRAILING']))
                
                # Possibly modify additional components
                remaining_components = [c for c in ['LEADING', 'VOWEL', 'TRAILING'] if c not in components_to_alter]
                for component in remaining_components:
                    if self.should_apply_typo(self.multiple_component_chance):
                        components_to_alter.append(component)
                
                # Apply modifications
                for component in components_to_alter:
                    if component == 'LEADING':
                        leading = self.get_random_typo('LEADING', leading)
                    elif component == 'VOWEL':
                        vowel = self.get_random_typo('VOWEL', vowel)
                    elif component == 'TRAILING':
                        # Even if no trailing consonant, might add one
                        trailing = self.get_random_typo('TRAILING', trailing)
                
                # Compose the character back
                try:
                    char = hangul_jamo.compose_jamo_characters(leading, vowel, trailing)
                except:
                    # Fallback if composition fails
                    pass
            
            result.append(char)
        
        return ''.join(result)

    def swap_adjacent_chars(self, text, swap_probability=0.05):
        """Swap adjacent characters with a given probability."""
        chars = list(text)
        i = 0
        
        while i < len(chars) - 1:
            # Only consider swapping Hangul characters
            if (ord('가') <= ord(chars[i]) <= ord('힣') and 
                ord('가') <= ord(chars[i+1]) <= ord('힣') and
                self.should_apply_typo(swap_probability)):
                # Swap characters
                chars[i], chars[i+1] = chars[i+1], chars[i]
                # Skip the swapped character
                i += 2
            else:
                i += 1
                
        return ''.join(chars)
    
    def add_typos(self, text):
        """Add various types of typos to Korean text."""
        # First apply word-level typos
        text = self.apply_word_level_typos(text)
        
        # Then apply character-level typos
        text = self.apply_character_typos(text)
        
        # Finally, potentially swap some adjacent characters
        # text = self.swap_adjacent_chars(text)
        
        return text

# Example usage
if __name__ == "__main__":
    # Create a typo generator with high probability settings
    typo_generator = TypoGenerator(
        char_typo_probability=0.3,
        word_typo_probability=0.4,
        multiple_component_chance=0.1,
    )
    
    # Test with some Korean text
    sample_texts = [
        "안녕하세요, 반갑습니다.",
        "한국어 오타 생성기입니다.",
        "이 프로그램은 한글 자모 라이브러리를 사용합니다.",
        "프로그래밍은 재미있습니다.",
        "저는 한국에 가본 적이 없습니다."
    ]
    
    for text in sample_texts:
        typo_text = typo_generator.add_typos(text)

        typo_indices = [i for i, (a, b) in enumerate(zip(text, typo_text)) if a != b]

        print(f"Original:   {text}")
        print(f"With typos: {typo_text}")
        print(f"Typos:      {typo_indices}")
        print()
