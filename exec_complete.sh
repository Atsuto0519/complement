# !/bin/sh

python ./complete_data.py
python ./calc_initialAIC.py
python ./calc_RMSE.py -i 0
python ./calc_RMSE.py -i 1
