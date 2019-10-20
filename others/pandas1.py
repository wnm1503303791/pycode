import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gff3_file = "./data/Homo_sapiens.GRCh38.97.chromosome.16.gene.gff3"
df = pd.read_csv(gff3_file, sep='\t', header=None)  # 读取文件，分隔符为 '\t'
df.columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
gene_length = df['end'] - df['start']

# print(df.head(10),df.shape)
# print(gene_length,gene_length.describe())

gene_length.plot.box(label="tz1")
plt.yscale('log')
plt.ylabel("length")


fig, ax = plt.subplots(figsize=(20, 4))
gene_length.plot(color="#ff5c5c", alpha=0.7)

df.groupby('source').count()['seqid']
ensids = df.attributes.str.extract("gene:(.*);Name")
df['ENSID'] = ensids
print(df.head(5))

plt.show()