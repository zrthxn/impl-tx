# Implement Transformer
An attempt to implement a Transformer model in PyTorch, purely for the sake of deeper 
understanding of the architecture and the internals. **Definitely not for production use.**

## Setup

Create a virtual environment and activate it.
```bash
$ virtualenv tx -p python3
  created virtual environment CPython3.6.9.final.0-64 in 4451ms
  creator CPython3Posix(dest=/mnt/d/home/user/impl-tx/tx, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/zrthxn/.local/share/virtualenv)
    added seed packages: pip==21.0.1, setuptools==56.0.0, wheel==0.36.2
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator

$ source tx/bin/activate
```

Install dependencies from the file.
```bash
(tx) $ pip install -r requirements.txt
```

Download and extract data files.
```bash
(tx) $ wget -P data https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz

# Extract gzipped tarball
(tx) $ cd data && tar -xzf toy-ende.tar.gz
```

## Resources

https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
https://github.com/SamLynnEvans/Transformer
https://www.youtube.com/watch?v=U0s0f995w14
