import os
import subprocess

from info import get_config


def get_tools():
    tools_dir = get_config()['tool_dir']
    if not os.path.isdir(tools_dir):
        os.mkdir(tools_dir)

    hifiasm = f'hifiasm-0.18.8'
    if os.path.isfile(os.path.join(tools_dir, hifiasm, 'hifiasm')):
        print(f'\nhifiasm Done...\n')
    else:
        print(f'\nGetting hifisam...')
        subprocess.run(
            f'git clone https://github.com/chhylp123/hifiasm.git --branch 0.18.8 --single-branch {hifiasm}',
            shell=True, cwd=tools_dir)
        hifiasm_dir = os.path.join(tools_dir, hifiasm)
        subprocess.run(f'make', shell=True, cwd=hifiasm_dir)
        print(f'\nhifiasm Done...\n')

    pbsim3 = f'pbsim3'
    if os.path.isfile(os.path.join(tools_dir, pbsim3, 'src', 'pbsim')):
        print(f'\npbsim Done!\n')
    else:
        print(f'\nGetting PBSIM3...')
        subprocess.run(f'git clone https://github.com/yukiteruono/pbsim3.git {pbsim3}', shell=True, cwd=tools_dir)
        pbsim_dir = os.path.join(tools_dir, pbsim3)
        subprocess.run(f'./configure; make; make install', shell=True, cwd=pbsim_dir)
        print(f'\npbsim Done!\n')


if __name__ == '__main__':
    get_tools()
