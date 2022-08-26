dictionary = {}
with open("usuk2ca-dictionary.txt") as text_file:
    for line in text_file:
        (key, value) = line.split()
        dictionary[key.rstrip()] = value


def canadize(
        word: str
) -> str:
    """
    "Canadize" the given word. Uses a heavily modified version of a dictionary found at
    https://www.tysto.com/uk-us-spelling-list.html

    :param word: An English word in its American/British form
    :return: The Canadian form of the word given
    """
    if word in dictionary.keys():
        word = dictionary[word]
    return word
