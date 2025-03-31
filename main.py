# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy

import torch
import numpy as np
from six.moves import cPickle as pickle
import dill
from torch.utils.data import DataLoader

import operator
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision import models
import torch.nn.functional as fn
from collections import defaultdict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from six.moves import cPickle as pickle
from torchvision.datasets.folder import accimage_loader
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F

#from external.googlenet.googlenet import googlenet

class VQA: ## open the dataset JSON
    def __init__(self, annotation_file=None, question_file=None):
        """
           Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:
            print('loading VQA annotations and questions into memory...')
            #time_t = datetime.datetime.utcnow()
            time_t = datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            print(datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        #print('imgToQA is ', imgToQA)
        #print(qa)
        #print(qqa)
        for ann in self.dataset['annotations']:
            imgToQA[ann['image_id']] += [ann] # list of image IDs and annotation of each image ID
            qa[ann['question_id']] = ann # List of question ID <-> annotations
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques # list of question ID <-> Question
        print('index created!')
        #print(imgToQA)
        #print(qa[1])

        """
        first_n_values = list(imgToQA.values())[0]
        first= list(imgToQA.keys())[0]
        print('Example of imgToQA')
        print('key', first)
        print('Values:', first_n_values)
        """
        """
        Example of imgToQA
        key 458752
        Values: [
            {
                'question_type': 'what is this', 
                'multiple_choice_answer': 'net', 
                'answers': [
                    {'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3}, 
                    {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6}, 
                    {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9}, 
                    {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}
                ],
                'image_id': 458752, 
                'answer_type': 'other',
                'question_id': 458752000
            }, 
            {
                'question_type': 'what', 
                'multiple_choice_answer': 'pitcher', 
                'answers': [
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 1}, 
                    {'answer': 'catcher', 'answer_confidence': 'no', 'answer_id': 2}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 3}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 4}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 5}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 6}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 7}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 8}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 9}, 
                    {'answer': 'pitcher', 'answer_confidence': 'yes', 'answer_id': 10}
                ], 
                'image_id': 458752, 
                'answer_type': 'other', 
                'question_id': 458752001
            }, 
            {
            'answer_type': 'other', 
            'multiple_choice_answer': 'orange', 
            'answers': [
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 1}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 2}, 
                {'answer': 'orange', 'answer_confidence': 'maybe', 'answer_id': 3}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 4}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 5}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 6}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 7}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 8}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 9}, 
                {'answer': 'orange', 'answer_confidence': 'yes', 'answer_id': 10}
            ], 
            'image_id': 458752, 
            'question_type': 'what color is the', 
            'question_id': 458752002
            }, 
            {
            'question_type': 'is this', 
            'multiple_choice_answer': 'yes', 
            'answers': [
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 1}, 
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 2}, 
                {'answer': 'yes', 'answer_confidence': 'maybe', 'answer_id': 3}, 
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 4}, 
                {'answer': 'yes', 'answer_confidence': 'maybe', 'answer_id': 5}, 
                {'answer': 'no', 'answer_confidence': 'maybe', 'answer_id': 6}, 
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 7}, 
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 8}, 
                {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 9}, 
                {'answer': 'yes', 'answer_confidence': 'maybe', 'answer_id': 10}
            ], 
            'image_id': 458752, 
            'answer_type': 'yes/no', 
            'question_id': 458752003
            }
        ]
        """
        """
        first_n_values = list(qa.values())[0]
        first = list(qa.keys())[0]
        print('Example of qa')
        print('key', first)
        print('Values:', first_n_values)
        """
        """
        Example of qa
        key 458752000
        Values: {
            'question_type': 'what is this', 
            'multiple_choice_answer': 'net', 
            'answers': [
                {'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3}, 
                {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6}, 
                {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9}, 
                {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}], 
            'image_id': 458752, 
            'answer_type': 'other', 
            'question_id': 458752000
        }
        """
        """
        first_n_values = list(qqa.values())[0]
        first = list(qqa.keys())[0]
        print('Example of qqa')
        print('key', first)
        print('Values:', first_n_values)
        """
        """
        Example of qqa
        key 458752000
        Values: {'image_id': 458752, 'question': 'What is this photo taken looking through?', 'question_id': 458752000}
        """

        # create class members
        self.qa = qa #each question ID with its answers
        self.qqa = qqa #Each question ID with the question itself and image ID
        self.imgToQA = imgToQA # Each image ID with a list of question IDs, each question ID answers

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print('%s: %s' % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(quesIds) == 0:
                anns = sum([[self.qa[quesId]] for quesId in quesIds if quesId in self.qa], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def loadQQA(self, ids=[]):
        """
        Load questions with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qqa[id] for id in ids]
        elif type(ids) == int:
            return [self.qqa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" % (self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, 'results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), \
            'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
        for ann in anns:
            quesId = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId][
                    'multiple_choices'], 'predicted answer is not one of the multiple choices'
            qaAnn = self.qa[quesId]
            ann['image_id'] = qaAnn['image_id']
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type'] = qaAnn['answer_type']
        print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


def pil_loader(path): # PIL Loader is simply RGB
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img


def default_loader(path): # accimage loader is faster but does not have all operations as PIL, uses intel IPP library
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pre_process_val(image_dir, qjson, ajson, img_prefix):
    with open('a2i.pkl', 'rb') as f:
        a2i = pickle.load(f)
        #print('a2i is ', a2i)

    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    ques_ids = vqa.getQuesIds(img_ids)

    ques_ids_modif = []
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        print('annotation is ', qa)
        qqa = vqa.loadQQA(ques_id)[0]
        print('answer is ', qqa)

        ques = qqa['question'][:-1]
        answers = qa['answers']
        answer = ""
        for ans in answers:
            ans = ans['answer'].lower()
            if ans in a2i:
                answer = ans
                break

        if not answer == "":
            ques_ids_modif.append(ques_id)
    return ques_ids_modif, img_names, img_ids


def pre_process_val_Ahmed(image_dir, qjson, ajson, img_prefix):
    with open('a2i.pkl', 'rb') as f:
        a2i = pickle.load(f)
        print('a2i is ', a2i)

    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if 'COCO_val2014_000000000428.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    ques_ids = vqa.getQuesIds(img_ids)

    ques_ids_modif = []
    counter = 0
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        print('annotation is ', qa)
        qqa = vqa.loadQQA(ques_id)[0]
        print('answer is ', qqa)

        ques = qqa['question'][:-1]
        answers = qa['answers']
        answer = ""
        for ans in answers:
            ans = ans['answer'].lower()
            if ans in a2i:
                answer = ans
                break

        if not answer == "" and counter <= 1:
            ques_ids_modif.append(ques_id)
        counter = counter + 1
    print('question id selected is ', ques_ids_modif)
    return ques_ids_modif, img_names, img_ids


def pre_process_dataset2(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset method #2. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    ques_ids = vqa.getQuesIds(img_ids)

    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    start = q2i["<sos>"]
    end = q2i["<eos>"]
    UNK = q2i["<unk>"]

    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]  # [0] means all answers for a question id, array has already one item
        qqa = vqa.loadQQA(ques_id)[0]  # [0] means question

        ques = qqa['question'][:-1]
        [q2i[x] for x in ques.lower().strip().split(" ")]

        answers = qa['answers']
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
            if ans not in a2i_count:
                a2i_count[ans] = 1
            else:
                a2i_count[ans] = a2i_count[ans] + 1

    a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)

    i2a = {}
    count = 0
    a2i = defaultdict(lambda: len(a2i))
    for word, _ in a_sort:
        a2i[word]
        i2a[a2i[word]] = word
        count = count + 1
        if count == 1000:
            break

    ques_ids_modif = []
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        qqa = vqa.loadQQA(ques_id)[0]

        ques = qqa['question'][:-1]
        answers = qa['answers']
        answer = ""
        for ans in answers:
            ans = ans['answer'].lower()
            if ans in a2i:
                answer = ans
                break

        if answer == "":
            continue
        ques_ids_modif.append(ques_id)
    return q2i, a2i, i2a, a2i_count, ques_ids_modif, ques_ids, img_names, img_ids


def pre_process_dataset2_Ahmed(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset method #2. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if 'COCO_train2014_000000000532.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    ques_ids = vqa.getQuesIds(img_ids)

    q2i = defaultdict(lambda: len(q2i))
    pad = q2i["<pad>"]
    start = q2i["<sos>"]
    end = q2i["<eos>"]
    UNK = q2i["<unk>"]

    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]  # [0] means all answers for a question id, array has already one item
        qqa = vqa.loadQQA(ques_id)[0]  # [0] means question

        ques = qqa['question'][:-1]
        [q2i[x] for x in ques.lower().strip().split(" ")]

        answers = qa['answers']
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
            if ans not in a2i_count:
                a2i_count[ans] = 1
            else:
                a2i_count[ans] = a2i_count[ans] + 1

    a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)

    i2a = {}
    count = 0
    a2i = defaultdict(lambda: len(a2i))
    for word, _ in a_sort:
        a2i[word]
        i2a[a2i[word]] = word
        count = count + 1
        if count == 1000:
            break

    ques_ids_modif = []
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        qqa = vqa.loadQQA(ques_id)[0]

        ques = qqa['question'][:-1]
        answers = qa['answers']
        answer = ""
        for ans in answers:
            ans = ans['answer'].lower()
            if ans in a2i:
                answer = ans
                break

        if answer == "":
            continue
        ques_ids_modif.append(ques_id)
    return q2i, a2i, i2a, a2i_count, ques_ids_modif, ques_ids, img_names, img_ids



def pre_process_dataset_Ahmed(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset for Ahmed. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if 'COCO_train2014_000000000532.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    # print('img_ids is', img_ids)
    ques_ids = vqa.getQuesIds(img_ids)
    # global q2i
    q2i = defaultdict(lambda: len(q2i))  # default for non-existant key will be len(q2i)
    pad = q2i["<pad>"]
    # print('pad is ', pad)
    # print('length of q2i is ', len(q2i))
    start = q2i["<sos>"]
    # print('start is ', start)
    # print('length of q2i is ', len(q2i))
    end = q2i["<eos>"]
    # print('end is ', end)
    # print('length of q2i is ', len(q2i))
    UNK = q2i["<unk>"]
    # print('UNK is ', UNK)
    # print('length of q2i is ', len(q2i))

    # global a2i_count
    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        print('qa is ', qa)
        qqa = vqa.loadQQA(ques_id)[0]
        print('qqa is ', qqa)

        ques = qqa['question'][:-1]  # -1 to omit the last character from the ques which is the question mark ?
        print('ques is ', ques)
        [q2i[x] for x in ques.lower().strip().split(" ")]  # q2i contains each word existing in a question, each one with an index.
        answers = qa['answers']
        print('answers are ', answers)
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
            if ans not in a2i_count:
                a2i_count[ans] = 1
            else:
                a2i_count[ans] = a2i_count[ans] + 1

    a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)
    # print('sorted answer length is ', len(a_sort))
    # print('sorted answers by occurance is ')
    # for x in range(10):#len(a_sort)):
    # print(a_sort[x])

    # global i2a
    i2a = {}
    count = 0
    # global a2i
    a2i = defaultdict(lambda: len(a2i))
    for word, _ in a_sort:
        a2i[ word]  # put each answer from the a_sort in a defaultdict a2i with len(a2i) which means each new word gets an iterative index because the words does not exist already
        i2a[a2i[ word]] = word  # make a copy list i2a similar to the a2i so putting each answer with an index sorted to the frequency of the answer
        count = count + 1
        if count == 1000:
            break

    print('a2i length is ', len(a2i))
    iterator = iter(a2i.items())
    for i in range(10):
        print(next(iterator))

    # for item in a2i.items():
    #    print(item)
    print('i2a length is ', len(i2a))
    iterator = iter(i2a.items())
    for i in range(10):
        print(next(iterator))
    # for item in i2a.items():
    #    print(item)
    print('q2i length is ', len(q2i))
    iterator = iter(q2i.items())
    for i in range(10):
        print(next(iterator))
    # for item in q2i.items():
    #    print(item)
    print('a2i_count length is ', len(a2i_count))
    iterator = iter(a2i_count.items())
    for i in range(10):
        print(next(iterator))
    # for item in a2i_count.items():
    #    print(item)
    return q2i, a2i, i2a, a2i_count


def pre_process_dataset(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))

    # print('img_ids is', img_ids)
    ques_ids = vqa.getQuesIds(img_ids)
    # global q2i
    q2i = defaultdict(lambda: len(q2i))  # default for non-existant key will be len(q2i)
    pad = q2i["<pad>"]
    # print('pad is ', pad)
    # print('length of q2i is ', len(q2i))
    start = q2i["<sos>"]
    # print('start is ', start)
    # print('length of q2i is ', len(q2i))
    end = q2i["<eos>"]
    # print('end is ', end)
    # print('length of q2i is ', len(q2i))
    UNK = q2i["<unk>"]
    # print('UNK is ', UNK)
    # print('length of q2i is ', len(q2i))

    # global a2i_count
    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        # print('qa is ', qa)
        qqa = vqa.loadQQA(ques_id)[0]
        # print('qqa is ', qqa)
        """
        qa is  {
            'question_type': 'are the', 
            'multiple_choice_answer': 'no', 
            'answers': [
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 1}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 2}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 3}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 4}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 5}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 6}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 7}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 8}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 9}, 
                {'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 10}
            ], 
            'image_id': 225041, 
            'answer_type': 'yes/no', 
            'question_id': 225041000
        }
        qqa is  {
            'image_id': 225041, 
            'question': 'Are the people getting off the bus?', 
            'question_id': 225041000
        }
        """
        ques = qqa['question'][:-1]  # -1 to omit the last character from the ques which is the question mark ?
        # print('ques is ', ques)
        [q2i[x] for x in ques.lower().strip().split(" ")]  # q2i contains each word existing in a question, each one with an index.

        """
        q2i length is  15196
        ('<pad>', 0)
        ('<sos>', 1)
        ('<eos>', 2)
        ('<unk>', 3)
        ('is', 4)
        ('the', 5)
        ('player', 6)
        ('left', 7)
        ('or', 8)
        ('right', 9)
        """
        answers = qa['answers']
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
            if ans not in a2i_count:
                a2i_count[ans] = 1
            else:
                a2i_count[ans] = a2i_count[ans] + 1

    a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)
    # print('sorted answer length is ', len(a_sort))
    # print('sorted answers by occurance is ')
    # for x in range(10):#len(a_sort)):
    # print(a_sort[x])

    """
    sorted answer length is  120299
    sorted answers by occurance is 
    ('no', 720451)
    ('yes', 704532)
    ('1', 108698)
    ('2', 108314)
    ('white', 71814)
    ('3', 51803)
    ('red', 43869)
    ('blue', 41489)
    ('black', 39940)
    ('0', 39628)
    """

    # global i2a
    i2a = {}
    count = 0
    # global a2i
    a2i = defaultdict(lambda: len(a2i))
    for word, _ in a_sort:
        a2i[ word]  # put each answer from the a_sort in a defaultdict a2i with len(a2i) which means each new word gets an iterative index because the words does not exist already
        i2a[a2i[ word]] = word  # make a copy list i2a similar to the a2i so putting each answer with an index sorted to the frequency of the answer
        count = count + 1
        if count == 1000:
            break
    """
    a2i length is  1000
    ('no', 0)
    ('yes', 1)
    ('1', 2)
    ('2', 3)
    ('white', 4)
    ('3', 5)
    ('red', 6)
    ('blue', 7)
    ('black', 8)
    ('0', 9)

    i2a length is  1000
    (0, 'no')
    (1, 'yes')
    (2, '1')
    (3, '2')
    (4, 'white')
    (5, '3')
    (6, 'red')
    (7, 'blue')
    (8, 'black')
    (9, '0')
    """
    print('a2i length is ', len(a2i))
    iterator = iter(a2i.items())
    for i in range(10):
        print(next(iterator))

    # for item in a2i.items():
    #    print(item)
    print('i2a length is ', len(i2a))
    iterator = iter(i2a.items())
    for i in range(10):
        print(next(iterator))
    # for item in i2a.items():
    #    print(item)
    print('q2i length is ', len(q2i))
    iterator = iter(q2i.items())
    for i in range(10):
        print(next(iterator))
    # for item in q2i.items():
    #    print(item)
    print('a2i_count length is ', len(a2i_count))
    iterator = iter(a2i_count.items())
    for i in range(10):
        print(next(iterator))
    # for item in a2i_count.items():
    #    print(item)
    return q2i, a2i, i2a, a2i_count


def collate_lines(seq_list):
    imgT, quesT, gT = zip(*seq_list)
    lens = [len(ques) for ques in quesT]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    imgT = torch.stack([imgT[i] for i in seq_order])
    quesT = [quesT[i] for i in seq_order]
    gT = torch.stack([gT[i] for i in seq_order])
    return imgT, quesT, gT

"""
class VqaDataset(Dataset):
    
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path,
                 image_filename_pattern, collate=False, q2i=None, a2i=None, i2a=None,
                 a2i_count=None, img_names=None, img_ids=None, ques_ids=None,
                 method='simple', dataset_type='train', enc_dir=''):
        
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        

        print('inside VqaDataset')
        print('Method is ', method)
        self.image_dir = image_dir
        self.qjson = question_json_file_path
        self.ajson = annotation_json_file_path
        img_prefix = image_filename_pattern.split('{}')[0]#img_prefix=COCO_train2014_
        self.collate = collate
        self.q2i = q2i
        self.a2i = a2i
        self.i2a = i2a
        self.a2i_count = a2i_count
        self.img_ids = img_ids
        self.ques_ids = ques_ids
        self.img_names = img_names
        self.method = method
        self.vqa = VQA(self.ajson, self.qjson)

        if self.method == 'simple':
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

        if not collate: ## what is collate?
            print('not collate')
            self.img_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f] # Get the image name: list form
            self.img_ids = [] #for:img_ids is full of pictures id:000000000001-nnnnnnnnn
            for fname in self.img_names: #fname = "COC0_train2014_000000000009. jpg"
                img_id = fname.split('.')[0].rpartition(img_prefix)[-1] #img_id =000000000009 (string type)
                self.img_ids.append(int(img_id))

            self.ques_ids = self.vqa.getQuesIds(self.img_ids)

            self.q2i, self.a2i, self.i2a, self.a2i_count = pre_process_dataset(image_dir, self.qjson, self.ajson, img_prefix)
            print('length of q2i is ', len(q2i), 'of folder ' , image_dir)
            print('length of a2i is ', len(a2i), 'of folder ' , image_dir)
            print('length of i2a is ', len(i2a), 'of folder ' , image_dir)
            print('length of a2i_count is ', len(a2i_count), 'of folder ' , image_dir)

        self.q2i_len = len(self.q2i)
        self.a2i_len = len(self.a2i.keys())
        self.q2i_keys = self.q2i.keys() #Get the keys of q2i, q2i: {"where": 0, "is":1, "he":2, "looking":3}), that is, get where, is, he, looking. . .
        self.enc_dir = enc_dir
        print('length of q2i is ', len(q2i), 'of folder ', image_dir)
        print('length of a2i is ', len(a2i), 'of folder ', image_dir)
        print('length of i2a is ', len(i2a), 'of folder ', image_dir)
        print('length of a2i_count is ', len(a2i_count), 'of folder ', image_dir)


        if collate and dataset_type == 'train':
            print('collate and train')
            with open('train_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)
        elif collate and dataset_type == 'val':
            print('collate and val')
            with open('val_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)

    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, idx):
        ques_id = self.ques_ids[idx]
        #print('ques_id is ',ques_id)
        img_id = self.vqa.getImgIds([ques_id])[0]
        #print('img_id is ', img_id)

        qa = self.vqa.loadQA(ques_id)[0]
        qqa = self.vqa.loadQQA(ques_id)[0]
        img_name = self.img_names[self.img_ids.index(img_id)]

        if self.method == 'simple':
            img = default_loader(self.image_dir + '/' + img_name)
            #imgT = self.transform(img).permute(1, 2, 0)
            imgT = self.transform(img).float()
        else:
            #file_idx = self.enc_idx[img_id] // 50
            #arr_idx = self.enc_idx[img_id] % 50
            #path = self.enc_dir + '/' + str(file_idx) + '.npz'
            #img = np.load(path)['out'][arr_idx, :, :]               # 512 x 196
            #imgT = torch.from_numpy(img).float()

            img = default_loader(self.image_dir + '/' + img_name)
            imgT = self.transform(img).float()
            #print('image tensor size after transform is ', imgT.size())

        ques = qqa['question'][:-1] #Get the question with the question mark removed
        quesI = [self.q2i["<sos>"]] + [self.q2i[x.lower()] for x in ques.split(" ") if x.lower() in self.q2i_keys] + [self.q2i["<eos>"]]
        # q2i={dict:11471}, q2i={"<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3, 'what':4, 'color':5. . . }
        # quesI = [1,4,454,54,843,6,11,2]
        if not self.collate:
            quesI = quesI + [self.q2i["<pad>"]]*(8 - len(quesI)) #????? why 8? What if the question len(quesI greater than 8)
        if self.method == 'simple':
            quesT = torch.zeros(self.q2i_len).float()#quesT:tensor([0. ,1. ,1. ,. . 0.0.0])
            for idx in quesI:
                quesT[idx] = 1
        else:
            quesT = torch.from_numpy(np.array(quesI)).long()
            #print('inside get_item, quesT is ', quesT)
            #print('inside get item, questT size is ', quesT.size())

        answers = qa['answers']
        max_count = 0
        answer = ""
        for ans in answers:
            #if not ans['answer_confidence'] == 'yes': # try running it with this condition enabled
            #    continue
            ans = ans['answer'].lower()
            if ans in self.a2i.keys() and self.a2i_count[ans] > max_count: # ans has to be in the top 1000 a2i.keys
                max_count = self.a2i_count[ans]
                answer = ans # take the answer of the question as the most dominant answer in the a2i_count dict
                #print('answer is ', answer)
        #print('the chosen answer is ', answer)

        if answer == "":                                              # only for validation
            gT = torch.from_numpy(np.array([self.a2i_len])).long()
        else:
            gT = torch.from_numpy(np.array([self.a2i[answer]])).long()

        if not self.collate:
            return {'img' : imgT, 'ques' : quesT, 'gt': gT}

        return imgT, quesT, gT
"""

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path,
                 image_filename_pattern, collate=False, q2i=None, a2i=None, i2a=None,
                 a2i_count=None, img_names=None, img_ids=None, ques_ids=None,
                 method='simple', dataset_type='train', enc_dir=''):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """

        print('inside VqaDataset')
        print('Method is ', method)
        self.image_dir = image_dir
        self.qjson = question_json_file_path
        self.ajson = annotation_json_file_path
        img_prefix = image_filename_pattern.split('{}')[0]#img_prefix=COCO_train2014_
        self.collate = collate
        self.q2i = q2i
        self.a2i = a2i
        self.i2a = i2a
        self.a2i_count = a2i_count
        self.img_ids = img_ids
        self.ques_ids = ques_ids
        self.img_names = img_names
        self.method = method
        self.vqa = VQA(self.ajson, self.qjson)

        if self.method == 'simple':
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

        if not collate: ## what is collate?
            print('not collate')
            self.img_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f] # Get the image name: list form
            self.img_ids = [] #for:img_ids is full of pictures id:000000000001-nnnnnnnnn
            for fname in self.img_names: #fname = "COC0_train2014_000000000009. jpg"
                img_id = fname.split('.')[0].rpartition(img_prefix)[-1] #img_id =000000000009 (string type)
                self.img_ids.append(int(img_id))

            self.ques_ids = self.vqa.getQuesIds(self.img_ids)

            self.q2i, self.a2i, self.i2a, self.a2i_count = pre_process_dataset(image_dir, self.qjson, self.ajson, img_prefix)
            print('length of q2i is ', len(q2i), 'of folder ' , image_dir)
            print('length of a2i is ', len(a2i), 'of folder ' , image_dir)
            print('length of i2a is ', len(i2a), 'of folder ' , image_dir)
            print('length of a2i_count is ', len(a2i_count), 'of folder ' , image_dir)

        self.q2i_len = len(self.q2i)
        self.a2i_len = len(self.a2i.keys())
        self.q2i_keys = self.q2i.keys() #Get the keys of q2i, q2i: {"where": 0, "is":1, "he":2, "looking":3}), that is, get where, is, he, looking. . .
        self.enc_dir = enc_dir
        print('length of q2i is ', len(q2i), 'of folder ', image_dir)
        print('length of a2i is ', len(a2i), 'of folder ', image_dir)
        print('length of i2a is ', len(i2a), 'of folder ', image_dir)
        print('length of a2i_count is ', len(a2i_count), 'of folder ', image_dir)


        if collate and dataset_type == 'train':
            print('collate and train')
            with open('train_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)
        elif collate and dataset_type == 'val':
            print('collate and val')
            with open('val_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)

    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, idx):
        ques_id = self.ques_ids[idx]
        #print('ques_id is ',ques_id)
        img_id = self.vqa.getImgIds([ques_id])[0]
        #print('img_id is ', img_id)

        qa = self.vqa.loadQA(ques_id)[0]
        qqa = self.vqa.loadQQA(ques_id)[0]
        img_name = self.img_names[self.img_ids.index(img_id)]

        if self.method == 'simple':
            img = default_loader(self.image_dir + '/' + img_name)
            #imgT = self.transform(img).permute(1, 2, 0)
            imgT = self.transform(img).float()
        else:
            #file_idx = self.enc_idx[img_id] // 50
            #arr_idx = self.enc_idx[img_id] % 50
            #path = self.enc_dir + '/' + str(file_idx) + '.npz'
            #img = np.load(path)['out'][arr_idx, :, :]               # 512 x 196
            #imgT = torch.from_numpy(img).float()

            img = default_loader(self.image_dir + '/' + img_name)
            imgT = self.transform(img).float()
            #print('image tensor size after transform is ', imgT.size())

        ques = qqa['question'][:-1] #Get the question with the question mark removed
        #print('ques is ', ques)
        #ques = '[CLS] ' + ques + ' [SEP] '
        #print('ques is ', ques)
        #print('length of tokenizer is ', len(tokenizer))
        #tokens = tokenizer.tokenize(ques)
        #print(tokens)
        #cls_token = tokenizer.cls_token
        #sep_token = tokenizer.sep_token
        #pad_token = tokenizer.pad_token
        #unk_token = tokenizer.unk_token
        #print(cls_token, sep_token, pad_token, unk_token)
        #cls_token_idx = tokenizer.cls_token_id
        #sep_token_idx = tokenizer.sep_token_id
        #pad_token_idx = tokenizer.pad_token_id
        #unk_token_idx = tokenizer.unk_token_id
        #print(cls_token_idx, sep_token_idx, pad_token_idx, unk_token_idx)
        #indexes = tokenizer.convert_tokens_to_ids(tokens)
        #print('indexes are ', indexes)
        #encoded_input = tokenizer(ques, return_tensors='pt')
        #print('encoded inout is ', encoded_input['input_ids'])
        #output = model(**encoded_input)
        #print('output size is ', output['last_hidden_state'].size())
        quesI = [self.q2i["<sos>"]] + [self.q2i[x.lower()] for x in ques.split(" ") if x.lower() in self.q2i_keys] + [self.q2i["<eos>"]]
        # q2i={dict:11471}, q2i={"<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3, 'what':4, 'color':5. . . }
        # quesI = [1,4,454,54,843,6,11,2]
        if not self.collate:
            quesI = quesI + [self.q2i["<pad>"]]*(8 - len(quesI)) #????? why 8? What if the question len(quesI greater than 8)
        if self.method == 'simple':
            quesT = torch.zeros(self.q2i_len).float()#quesT:tensor([0. ,1. ,1. ,. . 0.0.0])
            for idx in quesI:
                quesT[idx] = 1
        else:
            quesT = torch.from_numpy(np.array(quesI)).long()
            #quesT = output['last_hidden_state']
            #print('inside get_item, quesT is ', quesT)
            #print('inside get item, questT size is ', quesT.size())

        answers = qa['answers']
        max_count = 0
        answer = ""
        for ans in answers:
            #if not ans['answer_confidence'] == 'yes': # try running it with this condition enabled
            #    continue
            ans = ans['answer'].lower()
            if ans in self.a2i.keys() and self.a2i_count[ans] > max_count: # ans has to be in the top 1000 a2i.keys
                max_count = self.a2i_count[ans]
                answer = ans # take the answer of the question as the most dominant answer in the a2i_count dict
                #print('answer is ', answer)
        #print('the chosen answer is ', answer)

        if answer == "":                                              # only for validation
            gT = torch.from_numpy(np.array([self.a2i_len])).long()
        else:
            gT = torch.from_numpy(np.array([self.a2i[answer]])).long()

        if not self.collate:
            return {'img' : imgT, 'ques' : quesT, 'gt': gT}

        return imgT, quesT, gT



class VqaImgDataset(Dataset):

    def __init__(self, image_dir, name, img_prefix):
        self.image_dir = image_dir
        self.img_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f]
        self.transform = transforms.Compose([transforms.Resize((448, 448)),
                                             transforms.ToTensor()])

        img_ids = {}
        for idx, fname in enumerate(self.img_names):
            img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
            img_ids[int(img_id)] = idx

        with open(name + '_enc_idx.npy', 'wb') as f:
            pickle.dump(img_ids, f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = default_loader(self.image_dir + '/' + self.img_names[idx])
        imgT = self.transform(img)

        return imgT.float()


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, num_embeddings, num_classes, embed_dim=512, k=30):#num_embeddings = len(q2i), num_classes = 1000, why embed_dim is 512?
        super().__init__()
        # nn. Embedding: The input is a list of numbers, and the output is a list of corresponding symbol embedding vectors
        # num_embeddings: The size of the dictionary, for example, if there are 5000 words in total, then enter 5000
        # embed_dim: The dimension of the input vector, that is, how many dimensions are used to represent a symbol
        self.embed = nn.Embedding(num_embeddings, embed_dim) # len(q2i) number of rows for words, 512 vector length or the number of columns
                                    #    512,      512 kernel size
        self.unigram_conv = nn.Conv1d(embed_dim, embed_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(embed_dim, embed_dim, 2, stride=1, padding=1, dilation=2) #bigram means each KS=2
        # so each word will be convoluted twice, for the first and last word to be convoulted twice means we need an extra 1 word padding on each side
        # padding is usually kernel -1 # why dilation is 2? try with dilation = 1
        self.trigram_conv = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding=2, dilation=2)# same as above trigram
        # so KS=3 and stride 1 means each word will be convoluted three times, for each word to be convoluted three times then we need 2 extra padding on each side
        self.max_pool = nn.MaxPool2d((3, 1)) # kernel =3,1 which is equivalent to 3 rows and 1 column to get the max value of each word t in question q
        # Maximum pooling layer: The maximum pool is used in this article to obtain phrase-level characteristics
        # After maximum pooling, lstm is used to encode the sequence problem phrase level, and the corresponding problem-level feature qst is the hidden vector of time tLSTM.
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=3, dropout=0.4) # "Vanilla LSTM" 3 layers
        # mean single input layer, single hidden layer and single output layer
        self.tanh = nn.Tanh()
        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim)) # why k =30????????????????????????????
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        #self.W_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #self.W_p = nn.Parameter(torch.randn(embed_dim*2, embed_dim))
        #self.W_s = nn.Parameter(torch.randn(embed_dim*2, embed_dim))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim) # why Ws for sentence or question is not embed_dim*3?

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, image, question):                    # Image: B x 512 x 196 # is the question here passed after
        # one-hot-encoding? question is a long tensor of 8 long numbers representing the question words
        #print('question type is ', type(question))
        #print('question is ', question)
        #print('question size is ', question.size())
        #print('sorted tensors by length')
        #print(sorted(question, key=lambda x: x.size()[0], reverse=True))
        question, lens = rnn.pad_packed_sequence(question)
        #print('question after padding is ', question)
        print('question after padding size is ', question.size())
        question = question.permute(1, 0)                  # Ques : B x L where B stands for batch and batch is 100
        #print('question after permute is ', question)
        print('question after permute size is ', question.size())
        #      question after permute size is  torch.Size([100, 17])
        words = self.embed(question).permute(0, 2, 1)      # Words: B x L x 512

        #delete from here
        #words = self.embed(question)
        #print('words size is ', words.size())
        #words = words.permute(0,2,1)
        print('words size after permute is ', words.size())
        # till here

        #torch.set_printoptions(profile="full")
        #print('words is ', words)
        #print('words after embed and permute size is ', words.size())

        #      words after embed and permute size is torch.Size([100, 512, 13])
        #unigrams size is torch.Size([100, 512, 1, 13])
        #bigrams size is torch.Size([100, 512, 1, 13])
        #trigrams size is torch.Size([100, 512, 1, 13])

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2) # B x 512 x L
        #print('unigrams size is ', unigrams.size())
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # B x 512 x L
        #print('bigrams size is ', bigrams.size())
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2) # B x 512 x L
        #print('trigrams size is ', trigrams.size())
        words = words.permute(0, 2, 1)
        #print('words after convolution and unsqueeze and permute size is ', words.size())
        #      words after convolution and unsqueeze and permute size is  torch.Size([100, 13, 512])

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        #print('phrase before permute size is ', phrase.size())
        phrase = phrase.permute(0, 2, 1)                                    # B x L x 512
        #print('phrase after permute size is ', phrase.size())

        hidden = None
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)                          # B x L x 512

        print('image tensor size is ', image.size())
        #print('question tensor size is ', words.size())
        #print('phrase tensor size is ', phrase.size())
        #print('sentence tensor size is ', sentence.size())

        v_word, q_word = self.parallel_co_attention(image, words)
        #print('v_word size is ', v_word.size())
        #print('q_word size is ', q_word.size())
        v_phrase, q_phrase = self.parallel_co_attention(image, phrase)
        #print('v_phrase size is ', v_phrase.size())
        #print('q_phrase size is ', q_phrase.size())
        v_sent, q_sent = self.parallel_co_attention(image, sentence)
        #print('v_sent size is ', v_sent.size())
        #print('q_sent size is ', v_sent.size())

        #h_w = self.tanh(torch.matmul((q_word + v_word), self.W_w))
        #h_p = self.tanh(torch.matmul(torch.cat(((q_phrase + v_phrase), h_w), dim=1), self.W_p))
        #h_s = self.tanh(torch.matmul(torch.cat(((q_sent + v_sent), h_p), dim=1), self.W_s))

        h_w = self.tanh(self.W_w(q_word + v_word))
        #print('h_w size is ', h_w.size())
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        #print('h_p size is ', h_p.size())
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))
        #print('h_s size is ', h_s.size())

        logits = self.fc(h_s)

        return logits

    def parallel_co_attention(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 196
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L
        #print('size of a_v is ', a_v.size())
        #print('size of a_q is ', a_q.size())
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, lr=0.001):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10             # Steps
        self._test_freq = 250*4         # Steps
        self._save_freq = 1 #2          # Epochs
        self._print_freq = 50
        self._batch_size = batch_size
        self._lr = lr
        print('Inside ExperminetRunnerBase Constructor')

        # Use the GPU if it's available.
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if self.DEVICE == "cuda":
            self._model = self._model.cuda()

        if self.method == 'simple':
            #self.optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
            #self.optimizer = optim.SGD([{'params': self._model.embed.parameters(), 'lr': 0.8},
            #                            {'params': self._model.gnet.parameters(), 'lr': 1e-2},
            #                            {'params': self._model.fc.parameters(), 'lr': 1e-2}
            #                           ], momentum=0.9)
            self.optimizer = optim.Adam([{'params': self._model.embed.parameters(), 'lr': 0.08},
                                        {'params': self._model.gnet.parameters(), 'lr': 1e-3},
                                        {'params': self._model.fc.parameters(), 'lr': 1e-3}
                                       ], weight_decay=1e-8)
        else:
            self.optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1e-8)
        self.criterion = nn.CrossEntropyLoss()
        self.initialize_weights()

        #Logger for tensorboard
        self.writer = SummaryWriter()

        self.total_validation_questions = 214354.0 #121512.0

        if self.method == 'simple':
            self.chk_dir = './chk_simple/'
        else:
            self.chk_dir = './chk_coattention/'
            print('Creating Image Encoder')
            self.img_enc = models.resnet18(pretrained=True)
            modules = list(self.img_enc.children())[:-2] # resnet feature extractor, all layer except last 2 or from layer 0 to layer 7, can use [0:8] which means 0 --> 7
            self.img_enc = nn.Sequential(*modules)
            for params in self.img_enc.parameters():
                params.requires_grad = False
            if self.DEVICE == "cuda":
                self.img_enc = self.img_enc.cuda()
            self.img_enc.eval()

        if not os.path.exists(self.chk_dir):
            os.makedirs(self.chk_dir)

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        # TODO. Should return your validation accuracy
        accuracy = 0.0
        for batch_id, (imgT, quesT, gT) in enumerate(self._val_dataset_loader):
            self._model.eval()  # Set the model to train mode

            if not self.method == 'simple':
                print('questT type is ', quesT.type())
                quesT = rnn.pack_sequence(quesT)
                #print('1- image tensor size is ', imgT.size())
                imgT = imgT.to(self.DEVICE)
                #print('2- image tensor size is ', imgT.size())
                imgT = self.img_enc(imgT)
                #print('3- image tensor size is ', imgT.size())
                imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
                #print('4- image tensor size is ', imgT.size())

            imgT, quesT, gT = imgT.to(self.DEVICE), quesT.to(self.DEVICE), gT.to(self.DEVICE)
            gT = torch.squeeze(gT)
            pd_ans = self._model(imgT, quesT) # TODO
            for i in range(gT.shape[0]):
                if torch.argmax(pd_ans[i]).item() == gT[i]:
                    print('matching answers. Predicted is ', torch.argmax(pd_ans[i]).item(), ' and ground truth is ', gT[i].item())
                    accuracy = accuracy + 1.0

            if (batch_id + 1) % self._print_freq == 0:
                print('Validation Accuracy: %f' % (accuracy / ((batch_id + 1)*self._batch_size)))
        print('number of accurate answers is ', accuracy, ' out of the total number of validation questions ' , self.total_validation_questions)
        accuracy = accuracy / self.total_validation_questions
        return accuracy

    def train(self):
        print('Started Training.\n')
        tr_iter = 0
        val_iter = 0
        best_prec = 0.0
        for epoch in range(self._num_epochs):
            if (epoch + 1) // 3 == 0:
                print('epoch is ', epoch)
                self.adjust_learning_rate(epoch + 1)
            num_batches = len(self._train_dataset_loader)
            for batch_id, (imgT, quesT, gT) in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                if not self.method == 'simple':
                    #print('quesT is ', quesT)
                    print('quesT rows is ', len(quesT))
                    print('quesT column is ', len(quesT[0]))
                    #print('questT type is ', type(quesT))
                    quesT = rnn.pack_sequence(quesT)
                    #print('1- image tensor size is ', imgT.size())
                    imgT = imgT.to(self.DEVICE)
                    #print('2- image tensor size is ', imgT.size())
                    imgT = self.img_enc(imgT)
                    #print('3- image tensor size is ', imgT.size())
                    imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
                    #print('4- image tensor size is ', imgT.size())

                    """
                    1- image tensor size is  torch.Size([100, 3, 448, 448])
                    2- image tensor size is  torch.Size([100, 3, 448, 448])
                    3- image tensor size is  torch.Size([100, 512, 14, 14])
                    4- image tensor size is  torch.Size([100, 512, 196])
                    """
                else:
                    imgT = imgT.to(self.DEVICE)

                quesT, gT = quesT.to(self.DEVICE), gT.to(self.DEVICE)
                predicted_answer = self._model(imgT, quesT) # TODO
                ground_truth_answer = torch.squeeze(gT)     # TODO
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if (current_step + 1) % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))

                    # TODO: you probably want to plot something here
                    self.writer.add_scalar('train/loss', loss.item(), tr_iter)
                    tr_iter = tr_iter + 1

#                if (current_step + 1) % self._test_freq == 0:
#                    self._model.eval()
#                    val_accuracy = self.validate()
#                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
#
#                    # TODO: you probably want to plot something here
#                    self.writer.add_scalar('valid/accuracy', val_accuracy, val_iter)
#                    val_iter = val_iter + 1

            if (epoch + 1) % self._save_freq == 0 or epoch == self._num_epochs - 1:
                val_accuracy = self.validate()
                print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                self.writer.add_scalar('valid/accuracy', val_accuracy, val_iter)
                val_iter = val_iter + 1

                # remember best val_accuracy and save checkpoint
                is_best = val_accuracy > best_prec
                best_prec = max(val_accuracy, best_prec)
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self._model.state_dict(),
                                      'best_prec': best_prec},
                                      #'optimizer': optimizer.state_dict()}, is_best,
                                      is_best, self.chk_dir + 'checkpoint_' + str(epoch + 1) + '.pth.tar')

        # Closing tensorboard logger
        logdir = os.path.join('./tb_', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer.export_scalars_to_json(logdir + 'tb_summary.json')
        self.writer.close()

    def initialize_weights(self):
      for layer in self._model.modules():
          if not isinstance(layer, (nn.Conv2d, nn.Linear)):
              continue
          try:
              torch.nn.init.xavier_normal_(layer.weight)
              try:
                  nn.init.constant_(layer.bias.data, 0)
              except:
                  pass
          except:
              pass

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        self.method = 'coattention'
        print('Loading coattention numpy files. \n')
        with open('q2i.pkl', 'rb') as f:
            q2i = pickle.load(f)
        with open('a2i.pkl', 'rb') as f:
            a2i = pickle.load(f)
        with open('i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)
        with open('a2i_count.pkl', 'rb') as f:
            a2i_count = pickle.load(f)

        #print q2i, a2i, i2a length after being loaded from pickles as they will be for the whole dataset and then print their length at the preprocess once again as they will be
        #only for the tr images and validation images
        print('length of imported q2i.pkl is ', len(q2i))
        print('length of imported a2i.pkl is ', len(a2i))
        print('length of imported i2a.pkl is ', len(i2a))
        print('length of imported a2i_count.pkl is ', len(a2i_count))

        tr_img_names = np.load('tr_img_names.npy', encoding='latin1').tolist()
        tr_img_ids = np.load('tr_img_ids.npy', encoding='latin1').tolist()
        tr_ques_ids = np.load('tr_ques_ids.npy', encoding='latin1').tolist()

        va_img_names = np.load('va_img_names.npy', encoding='latin1').tolist()
        va_img_ids = np.load('va_img_ids.npy', encoding='latin1').tolist()
        va_ques_ids = np.load('va_ques_ids.npy', encoding='latin1').tolist()
        #va_ques_ids = np.load('va_ques_ids_orig.npy', encoding='latin1').tolist()

        print('Creating Datasets.')
        train_dataset = VqaDataset(image_dir=train_image_dir, collate=True,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                   img_names=tr_img_names, img_ids=tr_img_ids,
                                   ques_ids=tr_ques_ids, method=self.method,
                                   dataset_type="train", enc_dir='tr_enc')

        val_dataset = VqaDataset(image_dir=test_image_dir, collate=True,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                 img_names=va_img_names, img_ids=va_img_ids,
                                 ques_ids=va_ques_ids, method=self.method,
                                 dataset_type="validation", enc_dir='va_enc')

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_lines)

        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_lines)


        print('Creating Co Attention Model.')
        model = CoattentionNet(len(q2i), 1000).float()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)


    def _optimize(self, predicted_answers, true_answer_ids):
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_answers, true_answer_ids) # crossentropy_loss
        loss.backward()
        self.optimizer.step()

        return loss


from torch.utils import model_zoo

__all__ = ['GoogLeNet', 'googlenet']

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}


def googlenet(pretrained=False, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        transform_input (bool): If True, preprocesses the input according to the method with which it was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']))
        return model

    return GoogLeNet(**kwargs)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                       init_weights=True, remove_fc=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.remove_fc = remove_fc
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits and not self.remove_fc:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits and not self.remove_fc:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)

        if not self.remove_fc:
            # N x 1024
            x = self.dropout(x)
            x = self.fc(x)

        # N x 1000 (num_classes)
        if self.training and self.aux_logits and not self.remove_fc:
            return aux1, aux2, x

        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.fc2(x)
        # N x 1024

        return x


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, num_embeddings, num_classes):
        super().__init__()

        self.gnet = googlenet(pretrained=True, remove_fc=True)
        #self.embed = nn.Embedding(num_embeddings, 1024)
        self.embed = nn.Linear(num_embeddings, 1024)
        self.fc = nn.Linear(1024 + 1024, num_classes)

    def forward(self, image, question_encoding):
        img = self.gnet(image)
        ques = self.embed(question_encoding)

        con = torch.cat((img, ques), dim=1)
        return self.fc(con)


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):
        print('inside SimpleBaseLineExperimentRunner')
        self.method = 'simple'
        print('Loading simple numpy files. \n')
        with open('q2i.pkl', 'rb') as f:
            q2i = pickle.load(f)
        with open('a2i.pkl', 'rb') as f:
            a2i = pickle.load(f)
        with open('i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)
        with open('a2i_count.pkl', 'rb') as f:
            a2i_count = pickle.load(f)

        tr_img_names = np.load('tr_img_names.npy', encoding='latin1').tolist()
        tr_img_ids = np.load('tr_img_ids.npy', encoding='latin1').tolist()
        tr_ques_ids = np.load('tr_ques_ids.npy', encoding='latin1').tolist()

        va_img_names = np.load('va_img_names.npy', encoding='latin1').tolist()
        va_img_ids = np.load('va_img_ids.npy', encoding='latin1').tolist()
        va_ques_ids = np.load('va_ques_ids.npy', encoding='latin1').tolist()
        #va_ques_ids = np.load('va_ques_ids_orig.npy', encoding='latin1').tolist()

        #va_ques_ids_complete = np.load('va_ques_ids_orig.npy', encoding='latin1').tolist()
        #print('Total Validation Questions: %d' % len(va_ques_ids_complete))
        #print('Total training validation Questions: %d' % len(va_ques_ids))

        print('Creating Datasets.')
        train_dataset = VqaDataset(image_dir=train_image_dir, collate=True,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                   img_names=tr_img_names, img_ids=tr_img_ids,
                                   ques_ids=tr_ques_ids, method="simple",
                                   dataset_type="train")

        val_dataset = VqaDataset(image_dir=test_image_dir, collate=True,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                 img_names=va_img_names, img_ids=va_img_ids,
                                 ques_ids=va_ques_ids, method="simple",
                                 dataset_type="validation")

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)#num_data_loader_workers)

        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)#num_data_loader_workers)


        print('Creating Simple Baseline Model.')
        model = SimpleBaselineNet(len(q2i), 1000).float()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answer_ids):
        self.optimizer.zero_grad() #optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
        loss = self.criterion(predicted_answers, true_answer_ids) #criterion is CrossEntropyLoss
        loss.backward() #loss.backward() computes dloss/dx for every parameter x
        self.optimizer.step() #optimizer.step updates the value of x using the gradient x.grad

        return loss


# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    # print_hi('PyCharm')
    image_dir = "train2014"
    img_prefix = "COCO_train2014_"
    qjson = "v2_OpenEnded_mscoco_train2014_questions.json"
    ajson = "v2_mscoco_train2014_annotations.json"

    # global q2i, i2a, a2i, a2i_count
    """
    q2i, a2i, i2a, a2i_count = pre_process_dataset_Ahmed(image_dir, qjson, ajson, img_prefix)
    
    print("saving q2i")
    np.save('q2i.npy', dict(q2i))
    print("saved q2i")
    print("saving a2i")
    np.save('a2i.npy', dict(a2i))
    print("saved a2i")
    print("saving i2a")
    np.save('i2a.npy', i2a)
    print("saved i2a")
    print("saving a2i count")
    np.save('a2i_count.npy', a2i_count)
    print("saved a2i count")
    """
    # can be deleted
    """
    print("Trying dill dump")
    dill.dumps(q2i, 'q2i-dill.npy')
    print("Trying dill dump")

    print("Trying pickle.dump")
    pickle.dump(q2i, 'q2i-pickle.np')
    print("Trying pickle.dump")
    """

    """
    q2i, a2i, i2a, a2i_count, ques_ids_modif, ques_ids, img_names, img_ids  = pre_process_dataset2(image_dir, qjson, ajson, img_prefix)
    print('question IDs length is ', len(ques_ids))
    print('modified question IDs length is ', len(ques_ids_modif))
    
    question IDs length is  443757
    modified question IDs length is  422334
    """

    """""
    with open('q2i.pkl', 'wb') as f:
        pickle.dump(dict(q2i), f)
    with open('a2i.pkl', 'wb') as f:
        pickle.dump(dict(a2i), f)
    with open('i2a.pkl', 'wb') as f:
        pickle.dump(i2a, f)
    with open('a2i_count.pkl', 'wb') as f:
        pickle.dump(a2i_count, f)

    np.save('q2i.npy', dict(q2i))
    np.save('a2i.npy', dict(a2i))
    np.save('i2a.npy', i2a)
    np.save('a2i_count.npy', a2i_count)

    np.save('tr_img_names.npy', img_names)
    np.save('tr_img_ids.npy', img_ids)
    np.save('tr_ques_ids.npy', ques_ids_modif)
    print("done")
    """

    """
    q2i, a2i, i2a, a2i_count, ques_ids_modif, ques_ids, img_names, img_ids = pre_process_dataset2_Ahmed(image_dir, qjson,
                                                                                                  ajson, img_prefix)
    print('question IDs length is ', len(ques_ids))
    print('modified question IDs length is ', len(ques_ids_modif))




    np.save('tr_img_names2.npy', img_names)
    np.save('tr_img_ids2.npy', img_ids)
    np.save('tr_ques_ids2.npy', ques_ids_modif)
    print("done")
    """


    """
    image_dir = "val2014"
    img_prefix = "COCO_val2014_"
    qjson = "v2_OpenEnded_mscoco_val2014_questions.json"
    ajson = "v2_mscoco_val2014_annotations.json"

    ques_ids_modif_val, img_names_val, img_ids_val = pre_process_val(image_dir, qjson, ajson, img_prefix)

    np.save('va_img_names.npy', img_names_val)
    np.save('va_img_ids.npy', img_ids_val)
    np.save('va_ques_ids.npy', ques_ids_modif_val)
    """


    image_dir = "val2014-copy"
    img_prefix = "COCO_val2014_"
    qjson = "v2_OpenEnded_mscoco_val2014_questions.json"
    ajson = "v2_mscoco_val2014_annotations.json"

    ques_ids_modif_val2, img_names_val2, img_ids_val2 = pre_process_val_Ahmed(image_dir, qjson, ajson, img_prefix)

    #np.save('va_img_names2.npy', img_names_val2)
    #np.save('va_img_ids2.npy', img_ids_val2)
    #np.save('va_ques_ids2.npy', ques_ids_modif_val2)


    """
    tr_image_dir = 'train2014'
    va_image_dir = 'val2014'
    tr_out_dir = 'tr_enc'
    va_out_dir = 'va_enc'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=True)
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    for params in model.parameters():
        params.requires_grad = False

    if DEVICE == 'cuda':
        model = model.cuda()

    tr_img_dataset = VqaImgDataset(image_dir=tr_image_dir, name='train', img_prefix="COCO_train2014_")
    tr_img_dataset_loader = DataLoader(tr_img_dataset, batch_size=50, shuffle=False, num_workers=10)

    va_img_dataset = VqaImgDataset(image_dir=va_image_dir, name='val', img_prefix="COCO_val2014_")
    va_img_dataset_loader = DataLoader(va_img_dataset, batch_size=50, shuffle=False, num_workers=10)

    print('Dumping Training images encodings.')
    for idx, imgT in enumerate(tr_img_dataset_loader):
        imgT = imgT.to(DEVICE)
        out = model(imgT)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.cpu().numpy()

        path = tr_out_dir + '/' + str(idx) + '.npz'
        # np.savez(path, out=out)
        np.savez_compressed(path, out=out)
        print(path)
    
    print('Dumping Validation images encodings.')
    for idx, imgT in enumerate(va_img_dataset_loader):
        imgT = imgT.to(DEVICE)
        out = model(imgT)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.cpu().numpy()

        path = va_out_dir + '/' + str(idx) + '.npz'
        # np.savez(path, out=out)
        np.savez_compressed(path, out=out)
        print(path)
    """



    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='coattention')
    parser.add_argument('--train_image_dir', type=str, default='train2014')
    parser.add_argument('--train_question_path', type=str, default='v2_OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--train_annotation_path', type=str, default='v2_mscoco_train2014_annotations.json')
    parser.add_argument('--test_image_dir', type=str, default='val2014')
    parser.add_argument('--test_question_path', type=str, default='v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--test_annotation_path', type=str, default='v2_mscoco_val2014_annotations.json')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers)
    experiment_runner.train()

