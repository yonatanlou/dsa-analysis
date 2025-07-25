# DSA Analysis

In this repo, i will do some ad hoc analysis to the DSA dataset.
The [shantay](https://github.com/apparebit/shantay/tree/boss) is great. 
But i needed only a good sample of the data, so /data_prep can sampl
## Download data
For getting the relevant data i used the very easy to use shantay package:


```bash
mkdir data
uvx shantay download --first 2025-01-01 --last 2025-01-10 --archive data
```