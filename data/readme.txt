
# ===== ampds ===== #
https: // ieeexplore.ieee.org / document / 6802949

input - aggregate measurements
labelInp / labelOut - column names

column      meaning                     unit
0           time                        seconds
1           id                          none
2           active power (aggregate)    W
3           current (aggregate)         A
4           reactive power              VAr
5           apparent power              VA

output - appliance-level measurements
unitInp / unitOut - units
-- 22 Channels
-- channel 0 - time
-- channel 1 id
-- remaining 20 channels are appliances
-- 4 features per channel
-- -- Active Power (W)
-- -- Current (A)
-- -- Reactive Power (VAr)
-- -- Apparent Power (VA)

Output Index        Appliance
2                   RSE
3                   GRE
4                   B1E
5                   BME
6                   CWE
7                   DWE
8                   EQE
9                   FRE
10                  HPE
11                  OFE
12                  UTE
13                  WOE
14                  B2E
15                  CDE
16                  DNE
17                  EBE
18                  FGE
19                  HTE
20                  OUE
21                  TVE

Deferrable Loads:
-- 7. DWE (dishwasher)
-- 9. FRE (fridge)
-- 10. HPE (heat pump)
-- 13. WOE (wall oven)
-- 15. CDE (clothes dryer)

# ===== eco ===== #
Input - aggregate features
Output - appliance-level signals

Input
3 Aggregate Channels, Each channel has 5 electrical measurements:
-- P: Active Power
-- Q: Reactive Power
-- V: Voltage
-- A: Apparent Power
-- I: Current
Plus time, id. Total 17 columns.

Output
2D output
7 Appliances, each appliance has only 1 feature (P)

Index       Appliance
2           FRE (fridge)
3           DRW (dryer)
4           COM
5           KET (kettle)
6           WME (washing machine)
7           CSE
8           FRZ (freezer)
plus time, id, 9 columns total

6 files
All X have shape (x, 15) where x ranges from 273600 to 352800
Y have different # columns
eco1, eco3, eco6 have 7 columns ['time', 'id  ', 'FRE ', 'DRE ', 'COM ', 'KET ', 'WME ', 'CSE ', 'FRZ '], ['time', 'id  ', 'TAB ', 'FRZ ', 'COM ', 'CSE ', 'FRE ', 'KET ', 'ENT ']
eco4, eco5 have 8 columns
eco2 have 12 columns: outputs ['time', 'id  ', 'TAB ', 'DWE ', 'AIR ', 'FRE ', 'ENT ', 'FRZ ', 'KET ', 'LAM ', 'LAP ', 'STO ', 'TV0 ', 'STE ']
most likely becuase different houses have different appliances.

# ===== iawe ===== #

Input

Column      Meaning
0           time
1           id
2           P-agg
only one aggregate feature

Output
8 columns plus time and id

Index           Appliance                   Units
2               FRE (Fridge)                W
3               AC1 (AC)                    W
4               WME (Washing Machine)       W
5               PCE                         W
6               IRE                         W
7               UNK (Unknown Load)          W
8               TV                          W
9               WAP                         W

# ===== redd ===== #

Input
Column              Meaning             Unit
0                   time                sec
1                   id                  -
2                   total aggregate     W
3                   phase 1 aggregate   W
4                   phase 2 aggregate   W

X_t = [P_total, P_phase1, P_phase2]

Output

Index           Appliance
0               time
1               id
2               OVE
3               FRE
4               DWE
5               SO1
6               SO2
7               LT1
8               WAD
9               MIC
10              BAT
11              ESH
12              EST
13              SO3
14              SO4
15              LI2
16              LI3

input features not consistent


# ===== refit ===== #

Input
time, id, P_agg (W)

Output
Index       Appliance (all W)
0           time
1           id
2           FRE
3           FZE1
4           FZE2
5           TUD
6           WME
7           DWE
8           CSE
9           TVE
10          EHE

# ===== ukdale ===== #

Input
time, id, P_agg (W)

Output
Index           Appliace
0               time
1               id
2               KTE
3               MIC
4               FRE
5               WME
