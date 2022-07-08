# AGImpute
A composite structure model for single-cell RNA-seq imputation
## Tabel of Contents
- [Install AGImpute](#installAGImpute)
- [Install dependences](#installdependences)

## <a name="installAGImpute"></a>Install AGImpute
- **Download** 
`git clone https://github.com/MengShuang-ping/AGImpute.git`
- `cd AGImpute`
## <a name="installdependences"></a>Install dependences
- **Install** 
AGImpute is implemented in `python`(>3.8) and `pytorch`(>10.1) or `cuda`(11.4),Please install `python`(>3.8) and `pytorch`(>10.1) or cuda dependencies before run AGImpute.Users can either use pre-configured conda environment(recommended)or build your own environmen manually.
- `pip install -r requirements.txt `

###Use pre-configured conda environment(recommended)
1. Install conda(>4.0)
2. Install pytorch and torchvision
3. Install scanpy
4. Install leidenalg

## Use AGImpute
```
python3 AGImpute.py --help
```
You should see the following output:
```
Usage:enhance.py [OPRIONS]

    Options:
    -f, --fpath TEXT            The input UMI-count matrix.
    -o, --saveto TEXT           The output matrix.
    --transcript-count INTEGER  The target median transcript count for
                                determining thenumber of neighbors to use for
                                aggregation.(Ignored if "--num-neighbors" is
                                specified.)
    --max-neighbor-frac FLOAT   The maximum number of neighbors to use for
                                aggregation, relative to the total number of
                                cells in the dataset. (Ignored if "--num-
                                neighbors" is specified.)
    --pc-var-fold-thresh FLOAT  The fold difference in variance required for
                                relevant PCs, relative to the variance of the
                                first PC of a simulated dataset containing only
                                noise.
    --max-components INTEGER    The maximum number of principal components to
                                use.
    --num-neighbors INTEGER     The number of neighbors to use for aggregation.
    --sep TEXT                  Separator used in input file. The output file
                                will use this separator as well.  [default: \t]
    --use-double-precision      Whether to use double-precision floating point
                                format. (This doubles the amount of memory
                                required.)
    --seed INTEGER              Seed for pseudo-random number generator.
                                [default: 0]
    --test                      Test if results for test data are correct.
    --help                      Show this message and exit.
```

### Commands and options
### Input file format
### Output files
### Run with Testdata
