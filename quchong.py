#coding:utf-8
import shutil
readDir = "/home/sleeve/桌面/TCwork_git/data/wikisql_tok1/test_tok/train_tok.jsonl"
writeDir = "/home/sleeve/桌面/TCwork_git/data/wikisql_tok1/test_tok/train_tok2.jsonl"
lines_seen = set()
outfile=open(writeDir,"w")
f = open(readDir,"r")
for line in f:
    if line not in lines_seen:
        outfile.write(line)
        lines_seen.add(line)
outfile.close()
print ("success")
