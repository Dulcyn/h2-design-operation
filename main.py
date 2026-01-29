import json
from datetime import datetime, timedelta   #biblioteca para manuseio do tempo: h, dia e intervalo de tempo
#from xml.parsers.expat import model       # ?? le texto estruturado
import pandas as pd
import pyomo.environ as pyo

class Battery:                                  #criei uma classe
    def __init__(self, bess_config):            #objeto criado e dicionario dos seus atributos
        self.capexP=bess_config["capex_power"]   #atributo armazenado
        self.capexE=bess_config["capex_energy"]
        self.opex=bess_config["opex"]           #atributo armazenado
        self.life=bess_config["lifespan"]
        self.einitial=bess_config["E0"]
        self.efficiency=bess_config["η"]
        self.Emax=40   #kWh                      atributo fixo
        self.E=20      #kWh
        return

    def get_soc(self):
        return self.E/self.Emax
    
    def update_energy(self, potencia, Δt):
        ΔE=potencia*Δt
        self.E=self.E+ΔE
        return

class Eletrolisador:
    def __init__(self,eletr_config):
        self.capex=eletr_config["capex"]
        self.opex=eletr_config["opex"]
        self.capac=eletr_config["capacity"]
        self.powerd=eletr_config["power_demand"]
        self.lifespan=eletr_config["lifespan"]
        self.waterc=eletr_config["water_consumption"]
        return

class Photovoltaic:
    def __init__(self,pv_config):
        self.capex=pv_config["capex"]
        self.opex=pv_config["opex"]
        self.life=pv_config["lifespan"]
        return

class General:
    def __init__(self,par_config):
        self.sale=par_config["sale_H"]
        self.purc=par_config["purchase_H2O"]
        self.tust=par_config["TUST"]
        self.rate=par_config["rate"]


class H2sizer:
    def __init__(self,data, dataframe_path):
        self.config=data
        self.df=pd.read_csv(dataframe_path, encoding='utf-8', index_col="hour")
        self.bess=Battery(data["BESS"])
        self.eletr=Eletrolisador(data["EL"])
        self.pv=Photovoltaic(data["PV"])
        self.par=General(data["PARAM"])
    
    def build(self, soc):
        model=pyo.ConcreteModel()

        model.H = pyo.Set(initialize=pd.RangeIndex(start=1, stop=49, step=1, name='hour'))

        model.Pgrid=pyo.Var(model.H, within=pyo.Reals)
        model.Pbess_c=pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Pbess_d=pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Ebess=pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Ppvmax=pyo.Var(within=pyo.NonNegativeReals)
        model.Ebessmax = pyo.Var(within=pyo.NonNegativeReals)
        model.Pbessmax = pyo.Var(within=pyo.NonNegativeReals)
        model.Pel = pyo.Var(model.H, within=pyo.NonNegativeReals)


        def objective_rule(model):
            capex = (self.pv.capex * model.Ppvmax +
                     self.bess.capexP * model.Pbessmax + 
                        self.bess.capexE * model.Ebessmax +
                        self.eletr.capex * self.eletr.capac)
            opex = 00
            return capex + opex
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


        # constrains would be added here
        def power_balance_rule(model, h):
            return model.Pgrid[h] + model.Pbess_d[h] + model.Ppvmax * (self.df.at[h, 'Solar']) == model.Pbess_c[h] + model.Pel[h]
        model.power_balance = pyo.Constraint(model.H, rule=power_balance_rule)


        return

    def solve(self):
        return 5

def main():
    with open('data/parameters.json', 'r', encoding='utf8') as file:
        data = json.load(file)
    
    dataframe_path = "Cenarios.csv"
    h2sizer = H2sizer(data, dataframe_path)
    h2sizer.build(0.5)
    resultado=h2sizer.solve()

    a=1

if __name__ == "__main__":
    main()  