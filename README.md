# **GTasm: A Genome Assembly Method Using Graph Transformers and HiFi Reads**

## Requirements



- conda 4.9+
- python 3.9+
- pytorch==2.1.2+cu118
- dgl==2.1.0+cu118

## Installation

### 1. Create a conda virtual environment

```shell
git clone https://github.com/chu-xuezhe/GTasm.git
cd GTasm
conda env create -f requirements.yml
conda activate gtasm
```

### 2.Get tools

```shell
python get_tools.py
```



## Usage

```shell
python GTasm.py --reads <hifi_reads> --out <output path>
	--reads <hifi_reads>
	input hifi reads in FASTA/FASTQ format
	--out <output path>
	path of the result will be saved
	Optional:
	--threads <threads>
	Number of threads(default: 8)
	--model <model>
	Model path(default:pretrained/pretrained_model.pt)
	

```







