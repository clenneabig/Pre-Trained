
print("/home/clenneabig/Desktop/AIML591/FirstYear/Full Dataset/Lime-PurpleBlue/GH015970_1.jpg".split("/"))

outputs = []

with open(r"/home/clenneabig/Desktop/AIML591/Pre-Trained/query_paths.txt", 'r') as fp:
    for line in fp:
        splits = line.split("/")
        outputs.append(splits[7])

print(set(outputs))