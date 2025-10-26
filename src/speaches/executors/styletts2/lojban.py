# credits: gleki
from __future__ import annotations
import sys
import os

from re import sub, compile
from itertools import islice

def krulermorna(text: str) -> str:
    text = sub(r"\.", "", text)
    text = sub(r"^", ".", text)
    text = sub(r"u([aeiouy])", r"w\1", text)
    text = sub(r"i([aeiouy])", r"ɩ\1", text)
    text = sub(r"au", "ḁ", text)
    text = sub(r"ai", "ą", text)
    text = sub(r"ei", "ę", text)
    text = sub(r"oi", "ǫ", text)
    text = sub(r"\.", "", text)
    return text

def krulermornaize(words: list[str]) -> list[str]:
    return [krulermorna(word) for word in words]

ipa_vits = {
    "a$": 'aː',
    "a": 'aː',
    # "e(?=v)": 'ɛːʔ',
    # "e$": 'ɛːʔ',
    "e": 'ɛː',
    "i": 'iː',
    "o": 'oː',
    "u": 'ʊu',
    # "u": 'ʊː',
    "y": 'əː',
    "ą": 'aɪ',
    "ę": 'ɛɪ',
    # "ę(?=\b)(?!')": 'ɛɪʔ',
    "ǫ": 'ɔɪ',
    "ḁ": 'aʊ',
    "ɩa": 'jaː',
    "ɩe": 'jɛː',
    "ɩi": 'jiː',
    "ɩo": 'jɔː',
    "ɩu": 'juː',
    "ɩy": 'jəː',
    "ɩ": 'j',
    "wa": 'waː',
    "we": 'wɛː',
    "wi": 'wiː',
    "wo": 'wɔː',
    "wu": 'wuː',
    "wy": 'wəː',
    "w": 'w',
    "c": 'ʃ',
    # "bj": 'bʒ',
    "j": 'ʒ',
    "s": 's',
    "z": 'z',
    "f": 'f',
    "v": 'v',
    "x": 'hhh',
    "'": 'h',
    # "dj":'dʒ',
    # "tc":'tʃ',
    # "dz":'ʣ',
    # "ts":'ʦ',
    'r': 'ɹ',
    'r(?![ˈaeiouyḁąęǫ])': 'ɹɹ',
    # 'r(?=[ˈaeiouyḁąęǫ])': 'ɹ',
    "nˈu": 'nˈʊuː',
    "nu": 'nʊuː',
    "ng": 'n.g',
    "n": 'n',
    "m": 'm',
    "l": 'l',
    "b": 'b',
    "d": 'd',
    "g": 'ɡ',
    "k": 'k',
    "p": 'p',
    "t": 't',
    "h": 'h'
}

ipa_nix = {
    "a$": 'aː',
    "a": 'aː',
    # "e(?=v)": 'ɛːʔ',
    # "e$": 'ɛːʔ',
    "e": 'ɛː',
    "i": 'iː',
    "o": 'oː',
    "u": 'ʊu',
    # "u": 'ʊː',
    "y": 'əː',
    "ą": 'aɪ',
    "ę": 'ɛɪ',
    # "ę(?=\b)(?!')": 'ɛɪʔ',
    "ǫ": 'ɔɪ',
    "ḁ": 'aʊ',
    "ɩa": 'jaː',
    "ɩe": 'jɛː',
    "ɩi": 'jiː',
    "ɩo": 'jɔː',
    "ɩu": 'juː',
    "ɩy": 'jəː',
    "ɩ": 'j',
    "wa": 'waː',
    "we": 'wɛː',
    "wi": 'wiː',
    "wo": 'wɔː',
    "wu": 'wuː',
    "wy": 'wəː',
    "w": 'w',
    "c": 'ʃ',
    "gj": 'gɪʒ',
    "bj": 'bɪʒ',
    "j": 'ʒ',
    "s": 's',
    "z": 'z',
    "f": 'f',
    "v": 'v',
    "x": 'hh',
    "'": 'h',
    # "dj":'dʒ',
    # "tc":'tʃ',
    # "dz":'ʣ',
    # "ts":'ʦ',
    'r': 'ɹ',
    'r(?![ˈaeiouyḁąęǫ])': 'ɹɹɹɪ',
    # 'r(?=[ˈaeiouyḁąęǫ])': 'ɹ',
    "nˈu": 'nˈʊuː',
    "nu": 'nʊuː',
    "ng": 'ng',
    "n": 'n',
    "m": 'm',
    "l": 'l',
    "b": 'b',
    "d": 'd',
    "g": 'ɡ',
    "k": 'k',
    "p": 'p',
    "t": 't',
    "h": 'h'
}

vowel_pattern = compile("[aeiouyąęǫḁ]")
vowel_coming_pattern = compile("(?=[aeiouyąęǫḁ])")
diphthong_coming_pattern = compile("(?=[ąęǫḁ])")

question_words = krulermornaize(["ma", "mo", "xu"])
starter_words = krulermornaize(["le", "lo", "lei", "loi"])
terminator_words = krulermornaize(["kei", "ku'o", "vau", "li'u"])

def lojban2ipa(text: str, mode: str) -> str:
    if mode == 'vits':
        return lojban2ipa_vits(text)
    if mode == 'nix':
        return lojban2ipa_nix(text)
    return lojban2ipa_vits(text)
    
def lojban2ipa_vits(text: str) -> str:
    text = krulermorna(text.strip())
    words = text.split(' ')
    rebuilt_words = []
    question_sentence = False
    for index, word in enumerate([*words]):
        modified_word = word
        prefix, postfix = "", ""

        if word in question_words:
            postfix = "?"
            prefix=" " + prefix
            # question_sentence = True

        if word in starter_words:
            prefix=" " + prefix
            # question_sentence = True

        if word in terminator_words:
            postfix = ", "
        # if not vowel_pattern.match(word[-1:][0]):
        #     postfix += "ʔ"
        #     # cmevla
        #     if not vowel_pattern.match(word[0]):
        #         prefix += "ʔ"

        # if vowel_pattern.match(word[0]):
        #     prefix = "ʔ" + prefix

        if index == 0 or word in ["ni'o", "i"]:
            prefix = ", " + prefix

        split_word = vowel_coming_pattern.split(word)
        tail_word = split_word[-2:]
        # add stress to {klama}, {ni'o}
        if len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(vowel_pattern.match(tail_word[0][0])) and bool(vowel_pattern.match(tail_word[1][0])):
            head_word = split_word[:-2]
            modified_word = "".join(head_word) + "ˈ" + "".join(tail_word)
            # prefix=" " + prefix
            # add a pause after two-syllable words
            postfix = postfix + " "
        # add stress to {lau}, {coi}
        elif len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(diphthong_coming_pattern.match(tail_word[1][0])):
            head_word = split_word[:-2]
            modified_word = "".join(head_word) + tail_word[0] + "ˈ" + tail_word[1]
            # prefix=" " + prefix
            postfix = postfix + " "
        # add stress to {le}
        # elif len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(vowel_pattern.match(tail_word[1][0])):
        #     head_word = split_word[:-2]
        #     modified_word = "".join(head_word) + tail_word[0] + "ˈ" + tail_word[1]+" "
        #     postfix =postfix +" "
        
        # add a pause even after a cmavo
        if not (index - 1 >= 0 and words[index-1] in starter_words):
            prefix = " " + prefix

        # # add a pause before {.alis}
        # if bool(vowel_pattern.match(word[0])):
        #     word = ", " + word

        """
        for each letter: if the slice matches then convert the letter
        """
        rebuilt_word = ""
        lit = enumerate([*modified_word])
        for idx, x in lit:
            tail = modified_word[idx:]
            matched = False
            consumed = 1
            for attr, val in sorted(ipa_vits.items(), key=lambda x: len(str(x[0])), reverse=True):
                pattern = compile("^"+attr)
                matches = pattern.findall(tail)
                if len(matches)>0:
                    match = matches[0]
                    consumed = len(match)
                    rebuilt_word += val
                    matched = True
                    break
            if not matched:
                rebuilt_word += x
            [next(lit, None) for _ in range(consumed - 1)]

        rebuilt_words.append(prefix+rebuilt_word+postfix)

    output = "".join(rebuilt_words).strip()
    output = sub(r" {2,}", " ", output)
    output = sub(r", ?(?=,)", "", output)

    if question_sentence == True:
        output += "?"
    elif bool(vowel_pattern.match(text[-1:][0])):
        output += "."

    return output

def lojban2ipa_nix(text: str) -> str:
    text = krulermorna(text.strip())
    words = text.split(' ')
    rebuilt_words = []
    question_sentence = False
    for index, word in enumerate([*words]):
        modified_word = word
        prefix, postfix = "", ""

        if word in question_words:
            # postfix = "?"
            prefix=" " + prefix
            # question_sentence = True

        if word in starter_words:
            prefix=" " + prefix
            # question_sentence = True

        if word in terminator_words:
            postfix = ", "
        # if not vowel_pattern.match(word[-1:][0]):
        #     postfix += "ʔ"
        #     # cmevla
        #     if not vowel_pattern.match(word[0]):
        #         prefix += "ʔ"

        # if vowel_pattern.match(word[0]):
        #     prefix = "ʔ" + prefix

        if index == 0 or word in ["ni'o", "i"]:
            prefix = ", " + prefix

        split_word = vowel_coming_pattern.split(word)
        tail_word = split_word[-2:]
        # add stress to {klama}, {ni'o}
        if len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(vowel_pattern.match(tail_word[0][0])) and bool(vowel_pattern.match(tail_word[1][0])):
            head_word = split_word[:-2]
            modified_word = "".join(head_word) + "ˈ" + "".join(tail_word)
            # prefix=" " + prefix
            # add a pause after two-syllable words
            postfix = postfix + " "
        # add stress to {lau}, {coi}
        elif len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(diphthong_coming_pattern.match(tail_word[1][0])):
            head_word = split_word[:-2]
            modified_word = "".join(head_word) + tail_word[0] + "ˈ" + tail_word[1]
            # prefix=" " + prefix
            postfix = postfix + " "
        # add stress to {le}
        # elif len(tail_word) == 2 and len(tail_word[0]) > 0 and bool(vowel_pattern.match(tail_word[1][0])):
        #     head_word = split_word[:-2]
        #     modified_word = "".join(head_word) + tail_word[0] + "ˈ" + tail_word[1]+" "
        #     postfix =postfix +" "
        
        # add a pause even after a cmavo
        if not (index - 1 >= 0 and words[index-1] in starter_words):
            prefix = " " + prefix

        # # add a pause before {.alis}
        # if bool(vowel_pattern.match(word[0])):
        #     word = ", " + word

        """
        for each letter: if the slice matches then convert the letter
        """
        rebuilt_word = ""
        lit = enumerate([*modified_word])
        for idx, x in lit:
            tail = modified_word[idx:]
            matched = False
            consumed = 1
            for attr, val in sorted(ipa_nix.items(), key=lambda x: len(str(x[0])), reverse=True):
                pattern = compile("^"+attr)
                matches = pattern.findall(tail)
                if len(matches)>0:
                    match = matches[0]
                    consumed = len(match)
                    rebuilt_word += val
                    matched = True
                    break
            if not matched:
                rebuilt_word += x
            [next(lit, None) for _ in range(consumed - 1)]

        rebuilt_words.append(prefix+rebuilt_word+postfix)

    output = "".join(rebuilt_words).strip()
    output = sub(r" {2,}", " ", output)
    output = sub(r", ?(?=,)", "", output)

    if question_sentence == True:
        output += "?"
    elif bool(vowel_pattern.match(text[-1:][0])):
        output += "."

    return output

# print(lojban2ipa("ni'o le pa tirxu be me'e zo .teris. pu ki kansa le za'u pendo be le nei le ka xabju le foldi be loi spati"))
