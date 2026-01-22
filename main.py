import json


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
        self.life=eletr_config["lifespan"]
        self.waterc=eletr_config["water_consumption"]
        return

class Photovoltaic:
    def __init__(self,pv_config):
        self.capex=pv_config["capex"]
        self.opex=pv_config["opex"]
        self.life=pv_config["lifespan"]
        return



with open('data/parameters.json', 'r', encoding='utf8') as file:
    data = json.load(file)


bess=Battery(data["BESS"])

bess.update_energy(5,5/60)


eletr=Eletrolisador(data["EL"])

pv=Photovoltaic(data["PV"])


a=1
