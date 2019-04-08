import csv

mm = '0'
with open('train.csv', 'w') as f:
    f_csv = csv.writer(f)
    headers = next(f_csv)
    for row in f_csv:
        # Process row
        if row[2] == 'NA':
            row[2] = f.write(mm)
            # print(row[2])

        # print(len(row[2]))
        # for i in range(len(row[2])):
        #     # print(len)
        #     if row[2][i] == 'NA':
        #         # row[2][i] = '0'
        #         print(i)
