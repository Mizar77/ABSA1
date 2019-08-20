from torchtext import data
import xml.etree.ElementTree as ET
import random
import re


class SemEval2014(data.Dataset):
    def __init__(self, text_field, label_field, examples=None, **kwargs):

        # super().__init__()
        # self.data_list = []
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

            string = re.sub(r"\'s", "", string)   ###" \'s"
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", "", string)   ##" , "
            string = re.sub(r"!", "", string)  ## " ! "
            string = re.sub(r"\(", "", string)
            string = re.sub(r"\)", "", string)
            string = re.sub(r" \)","", string)
            string = re.sub(r"\?", "", string)
            string = re.sub(r"\s{2,}", "", string)
            string = re.sub(r"\"", "", string)
            string = re.sub(r"\'", "", string)
            string = re.sub(r"\\","",string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)

        fields = [('text', text_field), ('aspect', text_field), ('label', label_field)]
        for i in range(len(examples)):
            examples[i] = data.Example.fromlist(examples[i], fields)
        print(examples[i].text)
        super(SemEval2014, self).__init__(examples, fields, **kwargs)

        # print
    @classmethod
    def load_data(cls):
        train_file =['semeval_data/Laptop_Train_v2.xml', 'semeval_data/Laptops_Test_Data_phaseB.xml',
                     'semeval_data/Restaurants_Train.xml','semeval_data/Restaurants_Train_v2.xml',
                     'semeval_data/laptops-trial.xml','semeval_data/restaurants-trial.xml']
        # super().__init__()
        data_list = []
        for file in train_file:
            tree = ET.parse(file)
            root = tree.getroot()
            for child in root:
                for gchild in child:
                    if gchild.tag == 'aspectTerms':
                        for ggchild in gchild:
                            if 'polarity' in ggchild.attrib.keys():
                                data_list += [[child[0].text, ggchild.attrib['term'], ggchild.attrib['polarity']]]
        return data_list

    @classmethod
    def split(cls, text_field, label_field, dev_ratio=.1, shuffle = True):
        """"
        ratio  训练数据:测试数据
        """
        data_list = cls.load_data()
        # print(data_list)
        if shuffle:
            random.shuffle(data_list)
        # train_fields = [('text', text_field), ('aspect', aspect_field), ('label', label_field)]
        # for i in range(len(self.data_list)):
            # self.data_list[i] = data.Example.fromlist(self.data_list[i], train_fields)
        dev_index = -1 * int(dev_ratio*len(data_list))
        # return self.data_list[:dev_index], self.data_list[dev_index:]
        # print('dev_index', dev_index, 'len', len(data_list),len(data_list[:dev_index]), len(data_list[dev_index:]))
        return (cls(text_field, label_field, examples=data_list[:dev_index]),
                cls(text_field, label_field, examples=data_list[dev_index:]))






