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
python GTasm.py --reads <hifi_reads> --out <output_path>
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



## Training

Train the model

```shell
python train.py --train <train_data> --valid <valid_data>
	--train <train_data>
	The data used to train
	--valid <valid_data>
	The data used to valid
	Optional:
	--mn <modelname>
	The name of the model generated through training(default: test)
	--dropout <dropout>
	Dropout rate(default: 0.0)
	--seed <seed>
	Random seed(default: 1)
	
```



## Tested data

| dataset     | link                                                        |
| ----------- | ----------------------------------------------------------- |
| CHM13       | https://github.com/marbl/CHM13                              |
| HG002       | https://github.com/human-pangenomics/HG002_Data_Freeze_v1.0 |
| A. thaliana | https://ngdc.cncb.ac.cn/gsa/browse/CRA004538/CRX257574      |
| G. gallus   | https://www.genomeark.org/genomeark-all/Gallus_gallus.html  |

