file = open("../data/lexicons/bigsci_venues.txt", "r", encoding="utf8")
flag = False
for i, line in enumerate(file.readlines()):
    if line == '':
        if not flag:
            flag = True
        else:
            print("Extra empty line!")
    else:
        flag = False
    word = line.split()
    if len(word) == 1:
        print("Empty word!")
        print(i)
file.close()
