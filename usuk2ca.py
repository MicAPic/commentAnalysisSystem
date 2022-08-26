dictionary = {}
with open("usuk2ca-dictionary.txt") as text_file:
    for line in text_file:
        (key, value) = line.split()
        dictionary[key.rstrip()] = value


def canadize(word):
    if word in dictionary.keys():
        word = dictionary[word]
    return word
