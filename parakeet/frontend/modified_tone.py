# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style


class ModifiedTone():
    def __init__(self):
        self.must_neural_tone_words = {'麻烦', '麻利', '鸳鸯', '高粱', '骨头', '骆驼', '马虎', '首饰', '馒头', '馄饨', '风筝', '难为', '队伍',
                                       '阔气', '闺女', '门道', '锄头', '铺盖', '铃铛', '铁匠', '钥匙', '里脊', '里头', '部分', '那么', '道士',
                                       '造化', '迷糊', '连累', '这么', '这个', '运气', '过去', '软和', '转悠', '踏实', '跳蚤', '跟头', '趔趄',
                                       '财主', '豆腐', '讲究', '记性', '记号', '认识', '规矩', '见识', '裁缝', '补丁', '衣裳', '衣服', '衙门',
                                       '街坊', '行李', '行当', '蛤蟆', '蘑菇', '薄荷', '葫芦', '葡萄', '萝卜', '荸荠', '苗条', '苗头', '苍蝇',
                                       '芝麻', '舒服', '舒坦', '舌头', '自在', '膏药', '脾气', '脑袋', '脊梁', '能耐', '胳膊', '胭脂', '胡萝',
                                       '胡琴', '胡同', '聪明', '耽误', '耽搁', '耷拉', '耳朵', '老爷', '老实', '老婆', '老头', '老太', '翻腾',
                                       '罗嗦', '罐头', '编辑', '结实', '红火', '累赘', '糨糊', '糊涂', '精神', '粮食', '簸箕', '篱笆', '算计',
                                       '算盘', '答应', '笤帚', '笑语', '笑话', '窟窿', '窝囊', '窗户', '稳当', '稀罕', '称呼', '秧歌', '秀气',
                                       '秀才', '福气', '祖宗', '砚台', '码头', '石榴', '石头', '石匠', '知识', '眼睛', '眯缝', '眨巴', '眉毛',
                                       '相声', '盘算', '白净', '痢疾', '痛快', '疟疾', '疙瘩', '疏忽', '畜生', '生意', '甘蔗', '琵琶', '琢磨',
                                       '琉璃', '玻璃', '玫瑰', '玄乎', '狐狸', '状元', '特务', '牲口', '牙碜', '牌楼', '爽快', '爱人', '热闹',
                                       '烧饼', '烟筒', '烂糊', '点心', '炊帚', '灯笼', '火候', '漂亮', '滑溜', '溜达', '温和', '清楚', '消息',
                                       '浪头', '活泼', '比方', '正经', '欺负', '模糊', '槟榔', '棺材', '棒槌', '棉花', '核桃', '栅栏', '柴火',
                                       '架势', '枕头', '枇杷', '机灵', '本事', '木头', '木匠', '朋友', '月饼', '月亮', '暖和', '明白', '时候',
                                       '新鲜', '故事', '收拾', '收成', '提防', '挖苦', '挑剔', '指甲', '指头', '拾掇', '拳头', '拨弄', '招牌',
                                       '招呼', '抬举', '护士', '折腾', '扫帚', '打量', '打算', '打点', '打扮', '打听', '打发', '扎实', '扁担',
                                       '戒指', '懒得', '意识', '意思', '情形', '悟性', '怪物', '思量', '怎么', '念头', '念叨', '快活', '忙活',
                                       '志气', '心思', '得罪', '张罗', '弟兄', '开通', '应酬', '庄稼', '干事', '帮手', '帐篷', '希罕', '师父',
                                       '师傅', '巴结', '巴掌', '差事', '工夫', '岁数', '屁股', '尾巴', '少爷', '小气', '小伙', '将就', '对头',
                                       '对付', '寡妇', '家伙', '客气', '实在', '官司', '学问', '学生', '字号', '嫁妆', '媳妇', '媒人', '婆家',
                                       '娘家', '委屈', '姑娘', '姐夫', '妯娌', '妥当', '妖精', '奴才', '女婿', '头发', '太阳', '大爷', '大方',
                                       '大意', '大夫', '多少', '多么', '外甥', '壮实', '地道', '地方', '在乎', '困难', '嘴巴', '嘱咐', '嘟囔',
                                       '嘀咕', '喜欢', '喇嘛', '喇叭', '商量', '唾沫', '哑巴', '哈欠', '哆嗦', '咳嗽', '和尚', '告诉', '告示',
                                       '含糊', '吓唬', '后头', '名字', '名堂', '合同', '吆喝', '叫唤', '口袋', '厚道', '厉害', '千斤', '包袱',
                                       '包涵', '匀称', '勤快', '动静', '动弹', '功夫', '力气', '前头', '刺猬', '刺激', '别扭', '利落', '利索',
                                       '利害', '分析', '出息', '凑合', '凉快', '冷战', '冤枉', '冒失', '养活', '关系', '先生', '兄弟', '便宜',
                                       '使唤', '佩服', '作坊', '体面', '位置', '似的', '伙计', '休息', '什么', '人家', '亲戚', '亲家', '交情',
                                       '云彩', '事情', '买卖', '主意', '丫头', '丧气', '两口', '东西', '东家', '世故', '不由', '不在', '下水',
                                       '下巴', '上头', '上司', '丈夫', '丈人', '一辈', '那个'}

    def _neural_tone(self, word, pos, sub_finals):
        ge_idx = word.find("个")
        if len(word) == 1 and word in "吧呢啊嘛" and pos == 'y':
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        elif len(word) == 1 and word in "的地得" and pos in {"ud", "uj", "uv"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        # eg: 走了, 看着, 去过
        elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        elif len(word) > 1 and word[-1] in "们子" and pos in {"r", "n"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        # eg: 桌上, 地下, 家里
        elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        # eg: 上来, 下去
        elif len(word) > 1 and word[-1] in "来去" and pos[0] in {"v"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        # 个做量词
        elif ge_idx >= 1 and word[ge_idx - 1].isnumeric():
            sub_finals[ge_idx] = sub_finals[ge_idx][:-1] + "5"
        # reduplication words for n. and v. eg: 奶奶, 试试
        elif len(word) >= 2 and word[-1] == word[-2] and pos[0] in {"n", "v"}:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"
        # conventional tone5 in Chinese
        elif word in self.must_neural_tone_words or word[-2:] in self.must_neural_tone_words:
            sub_finals[-1] = sub_finals[-1][:-1] + "5"

        return sub_finals

    def _bu_tone(self, word, sub_finals):
        # "不" before tone4 should be bu2, eg: 不怕
        if len(word) > 1 and word[0] == "不" and sub_finals[1][-1] == "4":
            sub_finals[0] = sub_finals[0][:-1] + "2"
        # eg: 看不懂
        elif len(word) == 3 and word[1] == "不":
            sub_finals[1] = sub_finals[1][:-1] + "5"

        return sub_finals

    def _yi_tone(self, word, sub_finals):
        # "一" in number sequences, eg: 一零零
        if len(word) > 1 and word[0] == "一" and all([item.isnumeric() for item in word]):
            return sub_finals
        # "一" before tone4 should be yi2, eg: 一段
        elif len(word) > 1 and word[0] == "一" and sub_finals[1][-1] == "4":
            sub_finals[0] = sub_finals[0][:-1] + "2"
        # "一" before non-tone4 should be yi4, eg: 一天
        elif len(word) > 1 and word[0] == "一"  and  sub_finals[1][-1]!= "4":
            sub_finals[0] = sub_finals[0][:-1] + "4"
        # "一" beturn reduplication words shold be yi5, eg: 看一看
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            sub_finals[1] = sub_finals[1][:-1] + "5"
        # when "一" is oedinal word, it should be yi1
        elif word.startswith("第一"):
            sub_finals[1] = sub_finals[1][:-1] + "1"
        return sub_finals

    # 我给你讲个故事  没处理
    def _three_tone(self, word, sub_finals):
        if len(word) == 2 and self._all_tone_three(sub_finals):
            sub_finals[0] = sub_finals[0][:-1] + "2"
        elif len(word) == 3:
            word_list = jieba.cut_for_search(word)
            word_list = sorted(word_list, key=lambda i: len(i), reverse=False)
            new_word_list = []
            first_subword = word_list[0]
            first_begin_idx = word.find(first_subword)
            if first_begin_idx == 0:
                second_subword = word[len(first_subword):]
                new_word_list = [first_subword, second_subword]
            else:
                second_subword = word[:-len(first_subword)]

                new_word_list = [second_subword, first_subword]
            if self._all_tone_three(sub_finals):
                #  disyllabic + monosyllabic, eg: 蒙古/包
                if len(new_word_list[0]) == 2:
                    sub_finals[0] = sub_finals[0][:-1] + "2"
                    sub_finals[1] = sub_finals[1][:-1] + "2"
                #  monosyllabic + disyllabic, eg: 纸/老虎
                elif len(new_word_list[0]) == 1:
                    sub_finals[1] = sub_finals[1][:-1] + "2"
            else:
                sub_finals_list = [sub_finals[:len(new_word_list[0])], sub_finals[len(new_word_list[0]):]]
                if len(sub_finals_list) == 2:
                    for i, sub in enumerate(sub_finals_list):
                        # eg: 所有/人
                        if self._all_tone_three(sub) and len(sub) == 2:
                            sub_finals_list[i][0] = sub_finals_list[i][0][:-1] + "2"
                        # eg: 好/喜欢
                        elif i == 1 and not self._all_tone_three(sub) and sub_finals_list[i][0][-1] == "3" and \
                                sub_finals_list[0][-1][-1] == "3":

                            sub_finals_list[0][-1] = sub_finals_list[0][-1][:-1] + "2"
                        sub_finals = sum(sub_finals_list, [])
        # split idiom into two words who's length is 2
        elif len(word) == 4:
            sub_finals_list = [sub_finals[:2], sub_finals[2:]]
            sub_finals = []
            for sub in sub_finals_list:
                if self._all_tone_three(sub):
                    sub[0] = sub[0][:-1] + "2"
                sub_finals += sub

        return sub_finals

    def _all_tone_three(self, finals):
        return all(x[-1] == "3" for x in finals)

    # merge "不" and the word behind it
    def _merge_bu(self, seg):
        new_seg = []
        last_word = ""
        for word, pos in seg:
            if last_word == "不":
                word = last_word + word
            if word != "不":
                new_seg.append((word, pos))
            last_word = word[:]
        if last_word == "不":
            new_seg.append((last_word, 'd'))
            last_word = ""
        seg = new_seg
        return seg

    # function 1: merge "一" and reduplication words in it's left and right,eg: "看","一","看" ->"看一看"
    # function 2: merge single  "一" and the word behind it
    def _merge_yi(self, seg):
        new_seg = []
        # function 1
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and word == "一" and i + 1 < len(seg) and seg[i - 1][0] == seg[i + 1][0] and seg[i - 1][
                1] == "v":
                new_seg[i - 1][0] = new_seg[i - 1][0] + "一" + new_seg[i - 1][0]
            else:
                if i - 2 >= 0 and seg[i - 1][0] == "一" and seg[i - 2][0] == word and pos == "v":
                    continue
                else:
                    new_seg.append([word, pos])
        seg = new_seg
        new_seg = []
        # function 2
        for i, (word, pos) in enumerate(seg):
            if new_seg and new_seg[-1][0] == "一":
                new_seg[-1][0] = new_seg[-1][0] + word
            else:
                new_seg.append([word, pos])

        seg = new_seg
        return seg

    def _merge_continuous_three_tones(self, seg):
        new_seg = []
        sub_finals_list = [lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3) for (word, pos)
                           in seg]
        assert len(sub_finals_list) == len(seg)
        merge_last = [False] * len(seg)
        for i, (word, pos) in enumerate(seg):
            if i - 1 >= 0 and self._all_tone_three(sub_finals_list[i - 1]) and self._all_tone_three(
                    sub_finals_list[i]) and not merge_last[i - 1]:
                if len(seg[i - 1][0]) + len(seg[i][0]) <= 3:
                    new_seg[-1][0] = new_seg[-1][0] + seg[i][0]
                    merge_last[i] = True
                else:
                    new_seg.append([word, pos])
            else:
                new_seg.append([word, pos])
        seg = new_seg
        return seg

    def pre_merge_for_modify(self, seg):
        seg = self._merge_bu(seg)
        seg = self._merge_yi(seg)
        seg = self._merge_continuous_three_tones(seg)
        return seg

    def modified_tone(self, word, pos, finals):
        finals = self._bu_tone(word, finals)
        finals = self._yi_tone(word, finals)
        finals = self._neural_tone(word, pos, finals)
        finals = self._three_tone(word, finals)
        return finals
