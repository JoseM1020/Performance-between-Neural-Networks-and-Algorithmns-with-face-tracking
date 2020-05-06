import glob

paths = glob.glob("./Data/*.pgm")
f = open("yaleFaceInfo.info", "w")

for p in paths:
    f.write(p[2:] + " 1 0 0 168 192" + "\n")

f.close()


pathsNeg = glob.glob("./Negatives/*")
f = open("bg.txt", "w")

for p in pathsNeg:
    f.write(p[2:] + "\n")

f.close()