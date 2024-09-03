import argparse
import os
import subprocess

from to_dgl import to_dgl
from inference import inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reads', required=True, type=str, help='Hifi reads path')
    parser.add_argument('--out', type=str, default='.', help='Output path')
    parser.add_argument('--threads', type=str, default=8, help='Number of threads')
    parser.add_argument('--model', type=str, default='pretrained/pretrained_model.pt', help='Model path')
    args = parser.parse_args()

    reads = args.reads
    out = args.out
    threads = args.threads
    model = args.model

    print(f'Generate the init assembly graph...')
    init_graph=f'{out}/hifiasm/output'
    if not os.path.isdir(init_graph):
        os.makedirs(init_graph)

    subprocess.run(f'./vendor/hifiasm-0.18.8/hifiasm --prt-raw -o {init_graph}/asm -t{threads} -l0 {reads}',
                   shell=True)

    print(f'to DGL...')

    gfa_file = f'{init_graph}/asm.bp.raw.r_utg.gfa'
    to_dgl(gfa_file, reads, out)

    print(f'Getting assembly result...')
    inference(data_path=out, model_path=model, savedir=os.path.join(out, 'hifiasm'))

    result_dir = f'{out}/hifiasm/assembly'
    print(f'\nDone!')
    print(f'Assembly result saved in: {result_dir}/0_assembly.fasta')

