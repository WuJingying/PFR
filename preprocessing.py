from sklearn.model_selection import train_test_split
import time

def clear(filename):
    file = open(filename, encoding='ansi')
    texts = file.readlines()

    newfile = open("renmin.txt", 'w', encoding='ansi')

    for sentence in texts:
        words = sentence.strip().split(' ')
        words.pop(0)

        for word in words:
            if word.strip() != '':
                if word.startswith('['):
                    word = word[1:]
                elif ']' in word:
                    word = word[0:word.index(']')]

                if '{' in word:
                    word = word[0:word.index('{')]

                newfile.write(word + ' ')

        newfile.write('\n')


if __name__ == '__main__':
    start = time.time()
    clear('199801_clear.txt')
    end = time.time()
    print("清洗完成！所用时间为：", end - start, "s")