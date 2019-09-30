import subprocess
import os
import random
import glob
import re
import pysam
import pandas as pd
#subprocess.call('export SENTIEON_LICENSE=mn01:9000',shell=True)  ##should export outside the python script.
SENTIEON_DIR='/public/home/software/opt/bio/software/Sentieon/201808.07/bin/sentieon'
input_file = '/public/home/WBXie/Brassica_napus/505_genome_raw_data/cleandata/sample_info.txt'
#input_file = '/public/home/hzhao/temp/sample_info_test.txt'
ref = '/public/home/WBXie/genome/zs11.genome.fa'
outputpath = '/public/home/hzhao/data/rape_ZS11'
bwapath='%s/bwa_map' %outputpath
gatkpath = '%s/GVCF' %outputpath
vcfpath = '%s/VCF' %outputpath
nc=12
dd = pd.read_csv(input_file, sep='\s+',names=['vars','fq'])
samples_num = len(dd.loc[:,'vars'].unique())
#######################################################################################
##Do BWA###############################################################################
#######################################################################################
def do_bwa(SENTIEON_DIR, bwapath, ref, input_file, nc):
    samples = open(input_file,'r')
    if not os.path.exists(bwapath):
        os.mkdir(bwapath)
    lines = samples.readlines()
    samples.close()
    info_dict = {}
    for l in lines:
        infos = l.strip().split()
        if infos[0] in list(info_dict.keys()):
            info_dict[infos[0]].append(infos[1])
        else:
            info_dict[infos[0]] = [infos[1]]
    for k,v in list(info_dict.items()):
        #print(v)
        log_file = '%s/bwa_map/%s.bam.log' %(outputpath,k)
        if os.path.isfile(log_file):
            continue
        else:
            open(log_file, 'w').close()
            if len(v) >2:
                p =re.compile('.*[/-](.*_L\d+)_.*')
                sample_info_dict ={}
                for f in v:
                    exp_name = p.findall(f)[0]
                    if exp_name in list(sample_info_dict.keys()):
                        sample_info_dict[exp_name].append(f)
                    else:
                        sample_info_dict[exp_name] = [f]
                    tmp_bam_list = []
                for exp,fi in list(sample_info_dict.items()):
                    rand_n = str(random.random()*1e7)[0:6]
                    print(('bwa pair: %s + %s' %(fi[0],fi[1])))
                    subprocess.call("%s bwa mem -M -R  '@RG\\tID:%s%s\\tSM:%s\\tLB:%s\\tPL:illumin' -k 32 -t %s %s %s %s  2>>%s|samtools fixmate -cm -O bam - -|samtools view -ubhS - 2>>%s|samtools sort -m 20G -O bam -o %s/%s_%s.bam -T /public/home/WBXie/temp/temp/%s%s.sorted -" %(SENTIEON_DIR, rand_n,k,k,exp,nc,ref,fi[0],fi[1],log_file,log_file,bwapath,k,exp,rand_n,k),shell=True)
                    tmp_bam_list.append('%s/%s_%s.bam' %(bwapath, k, exp))
                subprocess.call("samtools merge -r -f %s/%s.bam %s" %(bwapath, k, ' '.join(tmp_bam_list)), shell=True)
                subprocess.call("samtools markdup -r -O bam %s/%s.bam %s/%s_markdup.bam" %(bwapath, k,bwapath, k),shell=True)
                subprocess.call(["samtools","index","%s/%s_markdup.bam" %(bwapath,k)])
                os.remove('%s/%s.bam' %(bwapath, k))
                for f in tmp_bam_list:
                    os.remove(f)
            else:
                rand_n = str(random.random()*1e7)[0:6]
                print(('bwa pair: %s + %s' %(v[0],v[1])))
                subprocess.call("%s bwa mem -M -R '@RG\\tID:%s%s\\tSM:%s\\tLB:%s\\tPL:illumin' -k 32 -t %s %s %s %s 2>>%s|samtools fixmate -cm -O bam - -|samtools view -ubhS - 2>>%s|samtools sort -m 20G -O bam -o %s/%s.bam -T /public/home/WBXie/temp/temp/%s%s.sorted -" %(SENTIEON_DIR, rand_n,k,k,k,nc,ref,v[0],v[1],log_file,log_file,bwapath,k,rand_n,k),shell=True)
                subprocess.call("samtools markdup -r -O bam %s/%s.bam %s/%s_markdup.bam" %(bwapath, k, bwapath, k),shell=True)
                subprocess.call(["samtools","index","%s/%s_markdup.bam" %(bwapath,k)])
                os.remove('%s/%s.bam' %(bwapath, k))
    return
##########################################################################################################################################
def check_bam(bam_path):
    logs = glob.glob('%s/*.log' %bam_path)
    bais = glob.glob('%s/*.bai' %bam_path)
    logs_run = [re.sub('_markdup.bam.bai','.bam.log',x) for x in bais]
    not_run = set(logs) - set(logs_run)
    for x in not_run:
        print(x)
        os.remove(x)
    return

def do_qc(bwapath, ref, nc):
    bam_files = glob.glob('%/*.bam' %bwapath)
    for bam in bam_files:
        if os.path.isfile('%s.bai' %bam):
            continue
        else:
            os.remove(re.sub('_markdup.bam','.log', bam))
            do_bwa(SENTIEON_DIR, bwapath, ref, input_file, nc=4) ##if bam file is incomplete, re-run bwa.
###########################################################################################################################################
####Do GATK################################################################################################################################
###########################################################################################################################################
def do_gatk(bwapath, gatkpath, ref, nc):
    if not os.path.exists(gatkpath):
        os.mkdir(gatkpath)
    #print(bwapath)
    bam_files = glob.glob('%s/*.bam' %bwapath)
    for bam in bam_files:
        if not os.path.isfile('%s.bai' %bam):
            os.remove(re.sub('_markdup.bam','.log', bam))
            do_bwa(SENTIEON_DIR, bwapath, ref, input_file, nc=nc) ##if bam file is incomplete, re-run bwa.
        name = re.sub('%s|_markdup.bam' %(bwapath),'',bam)
        print('GATK GVCF start..., %s' %name)
        log_file = '%s/%s.log' %(gatkpath,name)
        if os.path.isfile(log_file):
            continue
        else:
            open(log_file, 'w').close()
            #print(bam)
            subprocess.call("%s driver -r %s -t %s -i %s --algo Haplotyper --call_conf 30 --emit_conf 30 --genotype_model multinomial --emit_mode gvcf %s/%s.g.vcf.gz" %(SENTIEON_DIR, ref, nc, bam, gatkpath, name),shell=True)
    return

def merge_gvcf(bwapath, gatkpath, vcfpath, ref, nc):
    if not os.path.exists(vcfpath):
        os.mkdir(vcfpath)
    gvcf_files = glob.glob('%s/*.g.vcf.gz' %gatkpath)
    for gvcf in gvcf_files:
        if not os.path.isfile('%s.tbi' %gvcf):
            os.remove(re.sub('.g.vcf.gz','.log', gvcf))
            do_gatk(bwapath, gatkpath, ref, nc)
    gvcf_list = ' -v '.join(gvcf_files)
    genome = pysam.Fastafile(ref)
    chroms = genome.references
    for chrom in chroms:
        log_file = '%s/%s.log' %(vcfpath, chrom)
        if os.path.isfile(log_file):
            continue
        else:
            open(log_file, 'w').close()
            #print(gvcf_list)
            subprocess.call("%s driver -r %s --interval %s -t %s --algo GVCFtyper --emit_conf 30 --call_conf 30 --emit_mode VARIANT --genotype_model multinomial -v %s %s/%s.vcf" %(SENTIEON_DIR, ref, chrom, nc, gvcf_list, vcfpath, chrom),shell=True)
    return
if __name__==  '__main__':
    do_bwa(SENTIEON_DIR, bwapath, ref, input_file, nc)
    bam_num = len(glob.glob('%s/*.bai' %bwapath))
    if bam_num == samples_num:
        do_gatk(bwapath, gatkpath, ref, nc)
        gvcf_num = len(glob.glob('%s/*.tbi' %gatkpath))
        if gvcf_num == samples_num:
            merge_gvcf(bwapath, gatkpath, vcfpath, ref, nc)
