########### Codigo Nova Versao - Exato

import matplotlib.pyplot as plt   #biblioteca para criacao de graficos
import pandas as pd               #biblioteca para planilhas, tabelas, dados estruturados...  
import time                       #biblioteca para manipulacao de tempo  

start = time.time()               #tempo inicial de rodar o codigo 

path_scenarios = 'Cenarios_Renovavel.csv'    #variavel para guardar o arquivo dos cenarios   
# path_scenarios = 'Cenario_Medio.csv'

df = pd.read_csv(path_scenarios, encoding='utf-8')    #le o arquivo csv e cria um dataframe 'df'  
df.drop(columns=['year'], inplace=True)               #remove a coluna chamada 'year' da tabela df e modifica o objeto diretamente 

def fPV(scenario,hour,df):
  return df[(df['scenario']==scenario)&(df['hour']==hour)]['Solar'].values[0]   #filtra o df por scenario, hour e retorna o valor da coluna `Solar`

#def fWD(scenario,hour,df):
  #return df[(df['scenario']==scenario)&(df['hour']==hour)]['Eolica'].values[0]           ENERGIA EÓLICA

def require_data(scenario,hour,energy_type,df):
  return df[(df['scenario']==scenario)&(df['hour']==hour)][energy_type].values[0]     #filtra valor especifico
  

#Parametros fornecidos do problema organizados em Dicionarios

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
    "opex": 15.4,             #US$/kWs
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

import numpy as np                                            #biblioteca para matematica e arrays (estruturas que armazenam colecoes de valores)
class Sets:                                                   #cria uma classe `Sets` com 3 conjuntos numericos
    def __init__(self, Nsc, Nhours, Nyears):                  #cria uma funcao com o metodo construtor, que e executado automat
        self.scenarios = np.array(range(1, Nsc + 1))          #usada para facilitar iteracoes para otimizacao                          
        self.hours = np.array(range(1, Nhours + 1))
        self.years = np.array(range(1, Nyears + 1))
        return

sets = Sets(1, 8760, 25)                                      #arrays: num de cenarios, num de horas por ano, num de anos
Δt = 1                                                        #variando de 1 em 1 hora

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition  #importa funcoes para usar o otimizador
import pyomo.environ as pyo                                              #importa pacote do pyomo  

model = pyo.ConcreteModel()                                              #cria um modelo de otimizcao concreto

#### Criacao das variaveis
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

##### Funcao Objetivo
#   Objective Function
model.value = pyo.Objective(expr = model.CAPEX + model.OPEX, sense = pyo.minimize)                            #minimizar o CAPEX + OPEX


#### Equacoes
#   Subject to
model.costs   = pyo.ConstraintList()
model.costs.add(expr = model.CAPEX == EL["capex"] * EL["capacity"] + PV["capex"] * model.MaxPPV + BESS["capex_power"] * model.MaxPbess + BESS["capex_energy"] * model.MaxEbess)   #expressao CAPEX sem eolica

#expressao OPEX saida
model.costs.add(expr = model.OPEX_out == PARAM["TUST"] * (EL["capacity"] + model.MaxPbess) +
  EL["opex"] * EL["capacity"] + PV["opex"] * model.MaxPPV + BESS["opex"] * model.MaxPbess +
  (1/len(sets.scenarios)) * sum(sum(PARAM["purchase_H2O"] * model.Vh2o[s, h] + EDS["purchase"] * model.Ppeds[s, h]  for s in sets.scenarios) for h in sets.hours)
)

#expressao OPEX entrada
model.costs.add(expr = model.OPEX_in == (1/len(sets.scenarios)) * sum(sum(PARAM['sale_H'] * model.mh2[s, h] + EDS["spot_price"] * model.Pneds[s, h] * Δt for h in sets.hours) for s in sets.scenarios))

#expressao OPEX total
model.costs.add(expr = model.OPEX == sum((1/(1 + PARAM['rate'])**y)*(model.OPEX_out - model.OPEX_in) for y in sets.years))


#Balanco de potencia - conservacao da energia
# Power flow and EDS constraints
model.powerflow = pyo.ConstraintList()                          #lista de restricoes
for s in sets.scenarios:
  for h in sets.hours:
    model.powerflow.add(expr = model.Ppeds[s, h] - model.Pneds[s, h] == model.Pel[s, h] + model.Pbess[s, h] - model.MaxPPV * fPV(s,h, df))

#equacoes da taxa de hidrogenio e da agua utilizada no eletrolizador
# water and hydrogen constraints
model.h20_const = pyo.ConstraintList()
model.h2_const  = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.h20_const.add(expr = model.Vh2o[s, h] == EL['water_consumption'] * model.mh2[s, h])               #volume de agua
    model.h2_const.add(expr = model.mh2[s, h] == Δt * model.Pel[s, h]/ EL['power_demand'])                  #massa de hidrogenio


# Restricoes eletrolisador
# ELectrolyzer Constraints
model.elet_const = pyo.ConstraintList()
for s in sets.scenarios:
  for h in sets.hours:
    model.elet_const.add(expr = model.Pel[s, h] <= EL["capacity"])

####### Restricoes do sistema de baterias
# BESS Constraints
model.bess_const = pyo.ConstraintList()                         #cria lista de restricoes, permite add varias restricoes dentro do loop
for s in sets.scenarios:
  for h in sets.hours:
    model.bess_const.add(expr = model.Pbess[s, h] == model.Pbess_c[s, h] - model.Pbess_d[s, h])         #potencia liquida do BESS
    model.bess_const.add(expr = model.Pbess[s, h] >= -model.MaxPbess)                                   #limite inferior da potencia da bateria
    model.bess_const.add(expr = model.Pbess[s, h] <= model.MaxPbess)                                    #limite superior da potencia da bateria
    model.bess_const.add(expr = model.Pbess_c[s, h] <= 1e6 * model.γbess_c[s, h])
    model.bess_const.add(expr = model.Pbess_d[s, h] <= 1e6 * model.γbess_d[s, h])
    model.bess_const.add(expr = model.γbess_c[s, h] + model.γbess_d[s, h] <= 1)                         #condicao para evitar carga/descarg simultaneamente
    model.bess_const.add(expr = model.Ebess[s, h] <= model.MaxEbess)                                    #limite max de energia armazenada na bateria     

#equacao balanco de energia da bateria          
    if h > 1:
      model.bess_const.add(expr = model.Ebess[s, h] == model.Ebess[s, h-1] + Δt * (model.Pbess_c[s, h] * BESS["η"] - model.Pbess_d[s, h]/BESS["η"]))
    else:
      model.bess_const.add(expr = model.Ebess[s, h] == BESS["E0"]) #energia inicial da bateria


solution = SolverFactory("gurobi")                                    #escolhendo o solver que resolvera o problema de otimizacao - Gurobi
results = solution.solve(model)                                       #envia o modelo matematico para o solver executar a otimizacao

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):       #se for viavel e otima
  print ("this is feasible and optimal")
elif results.solver.termination_condition == TerminationCondition.infeasible:                                                   #se for inviavel
  print ("do something about it? or exit?")
else:
 # something else is wrong
  print (str(results.solver))                                                                               #mostra o status completo do solver

print(f'Total cost: {(model.CAPEX.get_values()[None] + model.OPEX.get_values()[None])/1e6} MUS$')

print(f'CAPEX: {model.CAPEX.get_values()[None]/1e6} MUS$')
print(f'OPEX total: {model.OPEX.get_values()[None]/1e6} MUS$')
print(f'OPEX yearly cost: {model.OPEX_out.get_values()[None]/1e6} MUS$')
print(f'OPEX yearly profit: {model.OPEX_in.get_values()[None]/1e6} MUS$')
print(f'Potência PV máxima instalada: {model.MaxPPV.get_values()[None]/1e3}MW')
#print(f'Potência WD máxima instalada: {model.MaxPWD.get_values()[None]/1e3}MW')
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
    #power['WD'].append(model.MaxPWD.get_values()[None] * fWD(s, h, df))
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