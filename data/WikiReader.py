# -*- coding: utf-8 -*-
import os
#from polyglot.text import Text
import re
import logging
import pickle
from data.special_characters import special_char_remover, unassigned_char_remover



class WikiReader_polyglot:
    def __init__(self, directory, language_code=None, char_cleaner=None, do_lower=True, disable_logger=True):
        # variables for file iteration
        os_walk = list(os.walk(directory))
        if not os_walk:
            raise FileNotFoundError("os.walk(" + directory + ") was empty. Does it exist?")
        all_files = (cur_dir + "/" + f for cur_dir, _, dir_files in os_walk for f in dir_files)
        relevant_name = re.compile("wiki_[0-9]+$")
        self.filenames = sorted(filter(relevant_name.search, all_files))

        # variables for line matching inside a file
        self.WikiExtractor_doc_open = re.compile('<doc id="([0-9]+)" url="(.*)" title="(.*)">')
        self.WikiExtractor_doc_close = re.compile('</doc>')

        self.language_code = language_code
        self.do_lower = do_lower


        if disable_logger:
            lang_detect_logger = logging.getLogger("polyglot.detect.base")
            lang_detect_logger.disabled = True


        # variable for preparing lines for tokenisation and removing unwanted characters
        if char_cleaner is None:
            self.char_cleaner = lambda sentence: sentence
        else:
            self.char_cleaner = char_cleaner()
            
        self.utf8_cleaner = unassigned_char_remover()
        
        self.errors = []

    # matches a given line with the regexp above
    def match_xml_opening(self, xml_line):
        return self.WikiExtractor_doc_open.match(xml_line)

    # matches a given line with the regexp above
    def match_xml_closing(self, xml_line):
        return self.WikiExtractor_doc_close.match(xml_line)

    # opens a file, makes the iterator available and closes the file
    @staticmethod
    def get_file_iter(filename):
        f_handle = open(filename)
        yield from f_handle
        f_handle.close()

    # iterates over the files in the directory,
    # yielding from the iterator below
    def article_iter(self):
        for file in self.filenames:
            print('Opening ', file)
            cur_file_iter = self.get_file_iter(file)
            yield from self.single_file_iter(cur_file_iter)

    # takes an iterator over lines from a file,
    # assuming the output format of WikiExtractor,
    # and yields the title and text of each article
    def single_file_iter(self, cur_file_iter):
        for line in cur_file_iter:
            line = line.rstrip()
            if line == "":
                continue

            open_match = self.match_xml_opening(line)
            if open_match:
                article_title = open_match.groups()[2]
            # text extraction in the else section
            # makes the iterator skip the line under the title
            # (this line is a repetition of the title)
            else:
                article_text = list(self.get_text(cur_file_iter))
                # don't yield articles with empty texts
                if article_text:
                    yield article_title, article_text

    # takes a file reader object and yields
    # lines in it, after tokenisation and punctuation removal,
    # until the line matches the xml closing tag
    def get_text(self, cur_file_iter):
        for line in cur_file_iter:
            line = line.rstrip()
            if line == "":
                continue

            if self.match_xml_closing(line):
                return
            
            line = self.utf8_cleaner(line)
            
            for sent in Text(line, hint_language_code=self.language_code).sentences:
                tokenised = " ".join(sent.words)
                clean = self.char_cleaner(tokenised)
                clean = clean.lower() if self.do_lower else clean
                yield clean.split()

def read_and_pickle_wiki(lang, directory="./", out_dir="./", size_per_file=10**5):
    corpus_dir = directory + lang
    w = WikiReader_polyglot(corpus_dir, special_char_remover)
    w_iter = w.article_iter()

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    n = 0
    
    cur_part = [a for i, a in zip(range(size_per_file), w_iter)]
    i = 0
    
    while cur_part:
        file_name = out_dir + "/" + lang + "_sent_wiki" + str(i) +".pkl"
        with open(file_name, "wb") as handle:
            pickle.dump(cur_part, handle)
        print("Dumped ", len(cur_part), " articles to ", file_name)
        n += len(cur_part)
        i += 1
        cur_part = [a for i, a in zip(range(size_per_file), w_iter)]
        
    print("Pickled ", n, " articles in total")

        
def wiki_from_pickles(pkls_dir, n_articles=None):
    pkl_files = os.listdir(pkls_dir)
    
    for f in pkl_files:
        with open(pkls_dir + "/" + f, "rb") as handle:
            cur_articles = pickle.load(handle)
            yield from cur_articles # [:min(len(cur_articles), n_articles)]     
#            n_articles -= len(cur_articles)
#            
#        if n_articles <= 0:
#            break
#                
            
    
    
    
#if __name__ == "__main__":
#    lang = "ALS"
#    lang_dir = "./" + lang
#    
#    out_dir_name = lang + "_pkl"
#    
#    read_and_pickle_wiki(lang, out_dir=out_dir_name)
#    
#    
    