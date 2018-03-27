import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

lines = sys.stdin
for line in lines:
    line = line.strip()
    data = json.loads(line)
    if data['mestext'] == "":
        continue
    print data['mestext']



