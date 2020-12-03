def full2half_width(ustr):
    half = []
    for u in ustr:
        num = ord(u)
        if num == 0x3000:    # 全角空格变半角
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        u = chr(num)
        half.append(u)
    return ''.join(half)

def half2full_width(ustr):
    full = []
    for u in ustr:
        num = ord(u)
        if num == 32:    # 半角空格变全角
            num = 0x3000
        elif 0x21 <= num <= 0x7E:
            num += 0xfee0
        u = chr(num)    # to unicode
        full.append(u)
        
    return ''.join(full)