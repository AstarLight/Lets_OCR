import collections
import torch


class StrConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

    # def encode(self, text):
    #     if isinstance(text, str):
    #         text_label = [self.dict[char] for char in text]
    #         text_length = [len(text_label)]
    #     elif isinstance(text, collections.Iterable):
    #         text_label = []
    #         text_length = []
    #         for t in text:
    #             label, length = self.encode(t)
    #             text_label.append(label)
    #             text_length.append(length)
    #     else:
    #         raise TypeError()
    #     return text_label, text_length

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), \
                "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
