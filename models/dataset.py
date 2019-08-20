from torchtext import data
import xml.etree.ElementTree as ET
import random
import re


class SemEval2014(data.Dataset):
    def __init__(self, text_field, label_field, switch, examples=None, **kwargs):

        # super().__init__()
        # self.data_list = []
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        my_field = data.Field()
        text_field.preprocessing = data.Pipeline(clean_str)
        if switch == 1:
            fields = [('left', text_field), ('right', text_field), ('aspect', text_field), ('label', label_field)]
        else:
            if switch == 2:
                fields = [('text', text_field), ('aspect', text_field), ('label', label_field)]
            else:
                print('switch= 3')
                fields = [('text', text_field), ('aspect', text_field), ('label', label_field),
                          ('start', my_field), ('end', my_field)]
        for i in range(len(examples)):
            examples[i] = data.Example.fromlist(examples[i], fields)
        super(SemEval2014, self).__init__(examples, fields, **kwargs)

        # print
    @classmethod
    def load_data(cls):
        train_file = ['../semeval_data/Laptop_Train_v2.xml', '../semeval_data/laptops-trial.xml',
                      '../semeval_data/Restaurants_Train_v2.xml', '../semeval_data/Restaurants_Train.xml',
                      '../semeval_data/restaurants-trial.xml']
        # super().__init__()
        data_list1 = []
        data_list2 = []
        data_list3 = []
        for file in train_file:
            tree = ET.parse(file)
            root = tree.getroot()
            for child in root:
                for gchild in child:
                    if gchild.tag == 'aspectTerms':
                        for ggchild in gchild:
                            if 'polarity' in ggchild.attrib.keys():
                                # data_list += [[child[0].text, ggchild.attrib['term'], ggchild.attrib['polarity']]]
                                text = child[0].text
                                aspect = ggchild.attrib['term']
                                polar = ggchild.attrib['polarity']
                                # print('Org text:',text)
                                # print('Org asp:', aspect)
                                text = cls.replace(text)
                                aspect = cls.replace(aspect)
                                leftIdx = text.find(aspect)
                                rightIdx = leftIdx+len(aspect)
                                left = text[0:rightIdx]
                                right = text[leftIdx:-1]
                                list_number = list(right.split(' '))  # 由于reverse无法对字符串进行操作，故对输入字符串以空格为单位分割，然后转为列表
                                list_number.reverse()
                                right = " ".join(list_number)
                                data_list1 += [[left, right, aspect, polar]]
                                data_list2 +=[[text, aspect, polar]]
                                text1 = list(text.split(' '))
                                y = aspect.split(' ')
                                try:lStart = text1.index(y[0])
                                except ValueError:
                                    lStart = 0
                                end = len(y)+lStart
                                data_list3 += [[text, aspect, polar, lStart, end]]
        return data_list1, data_list2, data_list3

    @classmethod
    def replace(cls, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string

    @classmethod
    def split(cls, text_field, label_field, dev_ratio=1/6, shuffle = True):
        """"
        ratio  训练数据:测试数据
        """
        data_list1, data_list2, data_list3 = cls.load_data()
        # print(data_list)
        if shuffle:
            random.shuffle(data_list1)
            random.shuffle(data_list2)
            random.shuffle(data_list3)
        # train_fields = [('text', text_field), ('aspect', aspect_field), ('label', label_field)]
        # for i in range(len(self.data_list)):
            # self.data_list[i] = data.Example.fromlist(self.data_list[i], train_fields)
        dev_index = -1 * int(dev_ratio*len(data_list1))
        # return self.data_list[:dev_index], self.data_list[dev_index:]
        # print('dev_index', dev_index, 'len', len(data_list),len(data_list[:dev_index]), len(data_list[dev_index:]))
        return (cls(text_field, label_field, 1, examples=data_list1[:dev_index]),
                cls(text_field, label_field, 1, examples=data_list1[dev_index:]),
                cls(text_field, label_field, 2, examples=data_list2[:dev_index]),
                cls(text_field, label_field, 2, examples=data_list2[dev_index:]),
                cls(text_field, label_field, 3, examples=data_list3[:dev_index]),
                cls(text_field, label_field, 3, examples=data_list3[dev_index:]))






