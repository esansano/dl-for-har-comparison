# har-dl
### Comparison of deep learning architectures for human activity recognition

To run a particular experiment:

<code>run_experiment.py -e <experiment_file> -d <dataset> -f <number of folds> -s</code>

Use the -s flag to save the experiment results in a log file.  
Example:

<code>run_experiment.py -e bilstm/exp01_bilstm -d usc-had -f 5 -s</code>

runs the experiment 1 using 5-fold cross validation and saving the results in the file <code>log/usc-had_exp01_bilstm.csv</code>

Scripts <code>run_bilstm_experiments.sh</code>, <code>run_lstm_experiments.sh</code>, <code>run_gru_experiments.sh</code> and <code>run_cnn_experiments.sh</code> run all the experiments for each architecture.

The experiments were conducted on two identical machines with Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz, NVIDIA GeForce GTX 1080 Ti GPU and running Ubuntu v17.10 as the operating system, using Pytorch version 0.3 and CUDA version 9.0. It took more than 3000 hours to complete the total set of experiments.


- The exp folder contains the definition of each individual experiment using two json files, one to describe the architecture of the model and the other to set the hyper-parameter values.
- The log folder contains the logs of the experiments.
- The file data/data_structure.txt describes the structure of the folders containing the datasets used for this experiment.
The datasets can be obtained from these links:

    - uci-har:
https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29

    - swell:
https://www.utwente.nl/en/eemcs/ps/research/dataset/

    - hhar:
https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition

    - realworld:
https://sensor.informatik.uni-mannheim.de/#dataset_realworld

    - activemiles:
http://hamlyn.doc.ic.ac.uk/activemiles/datasets.html

    - opportunity:
https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition

    - fusion:
https://www.utwente.nl/en/eemcs/ps/research/dataset/

    - mhealth:
http://archive.ics.uci.edu/ml/datasets/mhealth+dataset

    - usc-had:
http://sipi.usc.edu/HAD/

- The python scripts included under the folder src/plot generate the plots from the log files of the original experiments.