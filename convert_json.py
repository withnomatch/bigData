import json

input_file = r'd:\OneDrive - shoutoutuoadi325\桌面\大数据\PJ\StackOverFlow_Oracle_Database\oracle_database_questions.json'
output_file = r'd:\OneDrive - shoutoutuoadi325\桌面\大数据\PJ\oracle_database_questions_jsonlines.json'

print("Reading JSON array from: %s" % input_file)

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Total records: %d" % len(data))

print("Writing JSON Lines to: %s" % output_file)

with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Done! Converted %d records to JSON Lines format." % len(data))
