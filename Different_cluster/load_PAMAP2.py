import os
import numpy as nP
from pandas import Series
### timestamp | activityID | hr | temp+IMU hand (17 cols) | IMU chest (17 cols) | IMU ankle (17 cols)

def Loading_PAMAP2(filepath,id,table):
    with open(filepath, 'r') as f:
         for line in f:
             # print('New Line')
             tokens= line.split()
             # print(tokens)
             assert len(tokens) == 54
                 # Skip the line if no current session and cannot adopt activity from row
             if not tokens[1].isdigit() or tokens[1] == "0" or tokens[6]== "NaN" or tokens[4]== "NaN" or tokens[5]== "NaN":
                 continue
             table['timestamp'].append(float(tokens[0]))
             table['x'].append(float(tokens[4]))
             table['y'].append(float(tokens[5]))
             table['z'].append(float(tokens[6]))
             table['activity'].append(int(tokens[1]))
             table['User'].append(id)

    return table

