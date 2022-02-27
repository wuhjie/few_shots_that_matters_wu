# def udpos_parse(self, lang, input_file, which_split, update_label_list=True):
        # or just use nltk
sentence_egs = []
language = 'german'

with open("test-fa.tsv", "r") as f:
    lines = f.read().strip().split("\n\n")
    for line in lines:
        sent_vec = line.strip().split("\n")
        token_tag_vec = [wt.strip().split("\t") for wt in sent_vec]
        if True:
            for _, tag in token_tag_vec:
                # self.label_list.append(tag)
                print(1)
        # sentence_egs.append((language, which_split, token_tag_vec,))
# return sentence_egs

