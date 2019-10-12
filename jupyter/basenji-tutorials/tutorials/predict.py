lines = [['index','identifier','file','clip','sum_stat','description']]
lines.append(['0', 'CNhs11760', 'data/CNhs11760.bw', '384', 'sum', 'aorta'])
lines.append(['1', 'CNhs12843', 'data/CNhs12843.bw', '384', 'sum', 'artery'])
lines.append(['2', 'CNhs12856', 'data/CNhs12856.bw', '384', 'sum', 'pulmonic_valve'])

samples_out = open('data/heart_wigs.txt', 'w')
for line in lines:
    print('\t'.join(line), file=samples_out)
samples_out.close()
