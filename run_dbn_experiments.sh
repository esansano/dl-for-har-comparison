#!/usr/bin/env bash

cd src
python run_experiment.py -e dbn/exp01_dbn -d activemiles -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d activemiles -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d activemiles -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d activemiles -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d activemiles -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d activemiles -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d fusion -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d fusion -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d fusion -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d fusion -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d fusion -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d fusion -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d uci-har -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d uci-har -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d uci-har -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d uci-har -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d uci-har -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d uci-har -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d mhealth -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d mhealth -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d mhealth -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d mhealth -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d mhealth -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d mhealth -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d opportunity -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d opportunity -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d opportunity -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d opportunity -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d opportunity -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d opportunity -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d swell -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d swell -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d swell -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d swell -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d swell -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d swell -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d usc-had -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d usc-had -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d usc-had -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d usc-had -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d usc-had -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d usc-had -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d hhar -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d hhar -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d hhar -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d hhar -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d hhar -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d hhar -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d pamap2 -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d pamap2 -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d pamap2 -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d pamap2 -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d pamap2 -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d pamap2 -f 5 -s

python run_experiment.py -e dbn/exp01_dbn -d realworld -f 5 -s
python run_experiment.py -e dbn/exp02_dbn -d realworld -f 5 -s
python run_experiment.py -e dbn/exp03_dbn -d realworld -f 5 -s
python run_experiment.py -e dbn/exp04_dbn -d realworld -f 5 -s
python run_experiment.py -e dbn/exp05_dbn -d realworld -f 5 -s
python run_experiment.py -e dbn/exp06_dbn -d realworld -f 5 -s
