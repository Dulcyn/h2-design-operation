
import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time()

path_scenarios = 'Cenarios_Renovavel.csv'
# path_scenarios = 'Cenario_Medio.csv'

df = pd.read_csv(path_scenarios, encoding='utf-8')
df.drop(columns=['year'], inplace=True)

def fPV(scenario,hour,df):
  return df[(df['scenario']==scenario)&(df['hour']==hour)]['Solar'].values[0]

def fWD(scenario,hour,df):
  return df[(df['scenario']==scenario)&(df['hour']==hour)]['Eólica'].values[0]

def require_data(scenario,hour,energy_type,df):
  return df[(df['scenario']==scenario)&(df['hour']==hour)][energy_type].values[0]


PARAM = {
    "sale_H": 4,              #US$/kg
    "purchase_H2O": 0.6,      #US$/m³
    "TUST": 0.0198,           #US$/kW.ano
    "rate": 0.09,       
}
EL = {
    "capacity": 1000000,          #kW
    "power_demand": 50,           #kW/kg H2         
    "water_consumption": 0.05,    #m³/kg H2
    "capex": 1400,                #US$/kW
    "opex": 42,            #US$/kW
    "lifespan": 25                #anos
}
WD = {
    "capex": 1110,                #US$/kW
    "opex": 22.2,            #US$/kW
    "lifespan": 30                #anos
}
PV = {
    "capex": 770,                 #US$/kW
    "opex": 15.4,             #US$/kW
    "lifespan": 25                # anos
}

BESS = {
    "capex_power": 700,
    "capex_energy": 15,
    "opex": 14,
    "lifespan": 25,
    "E0": 0,
    "η": 0.8
}

EDS = {
    "spot_price": 0.0106 ,
    "purchase": 0.058 ,
}


import numpy as np
class Sets:
    def __init__(self, Nsc, Nhours, Nyears):
        self.scenarios = np.array(range(1, Nsc + 1))
        self.hours = np.array(range(1, Nhours + 1))
        self.years = np.array(range(1, Nyears + 1))
        return

sets = Sets(1, 8760, 25)

Δt = 1

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo

model = pyo.ConcreteModel()

# Variáveis de custos
model.CAPEX    = pyo.Var(within=pyo.NonNegativeReals)
model.OPEX     = pyo.Var(within=pyo.Reals)
model.OPEX_in  = pyo.Var(within=pyo.NonNegativeReals)
model.OPEX_out = pyo.Var(within=pyo.NonNegativeReals)
# Variáveis de decisões para potência instalada
model.MaxPPV    = pyo.Var(within=pyo.NonNegativeReals)
model.MaxPWD    = pyo.Var(within=pyo.NonNegativeReals)
model.MaxPbess  = pyo.Var(within=pyo.NonNegativeReals)
model.MaxEbess  = pyo.Var(within=pyo.NonNegativeReals)
# Variáveis de decisões para operação (potência)
model.Ppeds     = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)
model.Pneds     = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)
model.Pbess     = pyo.Var(sets.scenarios, sets.hours, within=pyo.Reals)
model.Pbess_c   = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)
model.Pbess_d   = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)
model.γbess_c   = pyo.Var(sets.scenarios, sets.hours, within=pyo.Binary)
model.γbess_d   = pyo.Var(sets.scenarios, sets.hours, within=pyo.Binary)
model.Pel       = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)

# Variáveis de decisões para operação (energia)
model.Ebess     = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)

# Variáveis de decisões para operação (outros)
model.mh2       = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)
model.Vh2o      = pyo.Var(sets.scenarios, sets.hours, within=pyo.NonNegativeReals)

#   Objective Function
model.value = pyo.Objective(expr = model.CAPEX + model.OPEX, sense = pyo.minimize)

#   Subject to
model.costs   = pyo.ConstraintList()
model.costs.add(expr = model.CAPEX == EL["capex"] * EL["capacity"] + PV["capex"] * model.MaxPPV + WD['capex'] * model.MaxPWD + BESS["capex_power"] * model.MaxPbess + BESS["capex_energy"] * model.MaxEbess)

model.costs.add(expr = model.OPEX_out == PARAM["TUST"] * (EL["capacity"] + model.MaxPbess) +
  EL["opex"] * EL["capacity"] + PV["opex"] * model.MaxPPV + WD["opex"] * model.MaxPWD + BESS["opex"] * model.MaxPbess +
  (1/len(sets.scenarios)) * sum(sum(PARAM["purchase_H2O"] * model.Vh2o[s, h] + EDS["purchase"] * model.Ppeds[s, h]  for s in sets.scenarios) for h in sets.hours)
)
model.costs.add(expr = model.OPEX_in == (1/len(sets.scenarios)) * sum(sum(PARAM['sale_H'] * model.mh2[s, h] + EDS["spot_price"] * model.Pneds[s, h] * Δt for h in sets.hours) for s in sets.scenarios))
model.costs.add(expr = model.OPEX == sum((1/(1 + PARAM['rate'])**y)*(model.OPEX_out - model.OPEX_in) for y in sets.years))

# Power flow and EDS constraints
model.powerflow = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.powerflow.add(expr = model.Ppeds[s, h] - model.Pneds[s, h] == model.Pel[s, h] + model.Pbess[s, h] - model.MaxPPV * fPV(s,h, df) - model.MaxPWD * fWD(s,h, df))

# water and hydrogen constraints
model.h20_const = pyo.ConstraintList()
model.h2_const  = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.h20_const.add(expr = model.Vh2o[s, h] == EL['water_consumption'] * model.mh2[s, h])
    model.h2_const.add(expr = model.mh2[s, h] == Δt * model.Pel[s, h]/ EL['power_demand'])

# ELectrolyzer Constraints
model.elet_const = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.elet_const.add(expr = model.Pel[s, h] <= EL["capacity"])


# BESS Constraints
model.bess_const = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.bess_const.add(expr = model.Pbess[s, h] == model.Pbess_c[s, h] - model.Pbess_d[s, h])
    model.bess_const.add(expr = model.Pbess[s, h] >= -model.MaxPbess)
    model.bess_const.add(expr = model.Pbess[s, h] <= model.MaxPbess)
    model.bess_const.add(expr = model.Pbess_c[s, h] <= 1e6 * model.γbess_c[s, h])
    model.bess_const.add(expr = model.Pbess_d[s, h] <= 1e6 * model.γbess_d[s, h])
    model.bess_const.add(expr = model.γbess_c[s, h] + model.γbess_d[s, h] <= 1)
    model.bess_const.add(expr = model.Ebess[s, h] <= model.MaxEbess)
    if h > 1:
      model.bess_const.add(expr = model.Ebess[s, h] == model.Ebess[s, h-1] + Δt * (model.Pbess_c[s, h] * BESS["η"] - model.Pbess_d[s, h]/BESS["η"]))
    else:
      model.bess_const.add(expr = model.Ebess[s, h] == BESS["E0"])


solution = SolverFactory("gurobi")
results = solution.solve(model)

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
  print ("this is feasible and optimal")
elif results.solver.termination_condition == TerminationCondition.infeasible:
  print ("do something about it? or exit?")
else:
 # something else is wrong
  print (str(results.solver))

print(f'Total cost: {(model.CAPEX.get_values()[None] + model.OPEX.get_values()[None])/1e6} MUS$')

print(f'CAPEX: {model.CAPEX.get_values()[None]/1e6} MUS$')
print(f'OPEX total: {model.OPEX.get_values()[None]/1e6} MUS$')
print(f'OPEX yearly cost: {model.OPEX_out.get_values()[None]/1e6} MUS$')
print(f'OPEX yearly profit: {model.OPEX_in.get_values()[None]/1e6} MUS$')
print(f'Potência PV máxima instalada: {model.MaxPPV.get_values()[None]/1e3}MW')
print(f'Potência WD máxima instalada: {model.MaxPWD.get_values()[None]/1e3}MW')
print(f'Potência BESS máxima instalada: {model.MaxPbess.get_values()[None]/1e3}MW')
print(f'Energia BESS máxima instalada:  {model.MaxEbess.get_values()[None]/1e3}MWh')


for s in sets.scenarios:
  power = {
    'EDS': [], 'PV': [], 'WD': [], 'BESS': [], "EL": []
  }
  PPEDS = model.Ppeds.get_values()
  PNEDS = model.Pneds.get_values()
  PBESS = model.Pbess.get_values()
  PEL = model.Pel.get_values()

  for h in sets.hours:
    power['EDS'].append(PPEDS[s,h] - PNEDS[s,h])
    power['PV'].append(model.MaxPPV.get_values()[None] * fPV(s, h, df))
    power['WD'].append(model.MaxPWD.get_values()[None] * fWD(s, h, df))
    power['BESS'].append(PBESS[s, h])
    power['EL'].append(PEL[s, h])
  
  plt.figure()
  for key in power.keys():
    plt.plot(power[key], label = key)
  plt.tight_layout()
  plt.xlabel('Hour')
  plt.ylabel('Power [kW]')
  plt.legend()
  plt.savefig(f'Results/scenario-{s}.png')

end = time.time()

print(f'tempo de execução em segundos: {end - start}' )

print('fim do programa')
