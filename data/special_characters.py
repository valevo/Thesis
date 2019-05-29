# -*- coding: utf-8 -*-
import re


def chars_from_unicode_block(block_prefix, block_rng, convert_to_hex=True, convert_to_char=True):
    if convert_to_hex:
        block_rng = list(map(lambda i: hex(i)[2:], block_rng))

    # construct the hex string codes of the characters in the given range
    chars = ["0x" + str(block_prefix) + char_ind for char_ind in block_rng]

    if convert_to_char:
        # first convert each hex representation to an int and then to a char
        chars = [chr(int(hex_code, 16)) for hex_code in chars]
    # make sure that white space is not returned (hex(ord(" ")) == "0x20")
    return list(filter(lambda c: not c.rstrip() == "0x20", chars))


BASIC_LATIN_SPECIAL = chars_from_unicode_block(2, range(16)) + \
                      chars_from_unicode_block(3, range(10, 16)) + \
                      chars_from_unicode_block(5, range(11, 16)) + \
                      chars_from_unicode_block(7, range(11, 16))
LATIN1_SUPPLEMENT_SPECIAL = chars_from_unicode_block("a", range(16)) +\
                            chars_from_unicode_block("b", range(16))
AR_SPECIAL = chars_from_unicode_block(60, range(16)) + chars_from_unicode_block(6, range(16, 32))
CJK_SPECIAL = chars_from_unicode_block(300, range(16)) + chars_from_unicode_block(30, range(16, 64))

RE_METACHARS = ".^$*+?{}[]\|()"


UNASSIGNED = chars_from_unicode_block(0, range(9)) + \
                    chars_from_unicode_block(1, range(16)) + \
                    chars_from_unicode_block(8, range(16)) + \
                    chars_from_unicode_block(9, range(16))


def get_special_char_regexp():
    RE_METACHARS = ".^$*+?{}[]\|()-"

    SPECIAL_CHARS = BASIC_LATIN_SPECIAL + LATIN1_SUPPLEMENT_SPECIAL + CJK_SPECIAL + AR_SPECIAL

    SPECIAL_CHARS = "".join([c for c in SPECIAL_CHARS
                             if not (c in RE_METACHARS or c.rstrip() == "")])
    
    RE_METACHARS = re.escape(RE_METACHARS)

    return re.compile("[" + RE_METACHARS + SPECIAL_CHARS + "]")


def special_char_remover():
    special_chars = get_special_char_regexp()
    return lambda s: special_chars.sub(' ', s)
    

def unassigned_char_remover():
    UNASSIGNED_CHARS = "".join(UNASSIGNED)
    unassigned_re = re.compile("[" + UNASSIGNED_CHARS + "]")
    return lambda s: unassigned_re.sub("", s)
    



if __name__ == '__main__':
    re_special = get_special_char_regexp()

    print(re_special)
    print(len(re_special.pattern))


    print(re_special.sub("F", "a b.c{d〽e〟f⟩〉g\nh"))


