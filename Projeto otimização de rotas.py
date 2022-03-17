
"""
Created on Tue Mar 15 03:29:00 2022

@author: ferna söndahl
"""

##### Distribuição de serviços


import pandas as pd

from haversine import haversine, Unit, haversine_vector
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
import requests
import zipfile
from io import BytesIO
import time

# %%

dic_comb = {'origem': [], 'destino': [], 'loc_orig': [], 'loc_dest': []}

locplan = "C:/Users/ferna/OneDrive/Área de Trabalho/Curso OR em Python/Municipios.xlsx"

geo_df = pd.read_excel(io=locplan)
geo_df['cords'] = geo_df['Latitude'].astype(str) + ", " + geo_df['Longitude'].astype(str)
geo_df['cords_inv'] = geo_df['Longitude'].astype(str) + ", " + geo_df['Latitude'].astype(str)
geo_df['chave'] = geo_df.index

for i in range(len(geo_df)):
    for j in range(len(geo_df)):
        dic_comb['origem'].append(geo_df['chave'][i])
        dic_comb['destino'].append(geo_df['chave'][j])
        dic_comb['loc_orig'].append(geo_df['cords'][i])
        dic_comb['loc_dest'].append(geo_df['cords'][j])

#### dist - DataFrame de combinação de todas as distâncias das coordenadas entre si
    
dist_df = pd.DataFrame(data=dic_comb)


for col in['loc_orig', 'loc_dest']:
  dist_df[col] = dist_df[col].apply(lambda lin: tuple((list(map(float, lin.split(','))))))

dist_df['Distancia'] = haversine_vector(list(dist_df['loc_orig']), list(dist_df['loc_dest'])).astype(float)

#### dist2 - DataFrame De Para

dist2_df = pd.DataFrame() 

tst = list(dist_df['origem'].drop_duplicates())


for local in tst:
    dist2_df[str(local)] = list(dist_df.query(f'origem == {local}')['Distancia'])
    
# %%

ciclo = []
subrotas = []
subciclo = []

durac = time.time()

while len(ciclo) != len(geo_df): 
        
    
    model = pyo.ConcreteModel()
    
    
    model.servic = pyo.Var(range(len(dist2_df)), range(len(dist2_df)), bounds=(0, 1), within=Integers)
    servic = model.servic
    
    
    model.C1 = pyo.ConstraintList()
    for i in range(len(dist2_df)):
        model.C1.add(expr= sum(servic[i, j] for j in range(len(dist2_df))) == 1)  #Uma só na origem
        model.C1.add(expr= sum(servic[j, i] for j in range(len(dist2_df))) == 1) #Um só no destino
    
    model.C2 = pyo.ConstraintList()
    for i in range(len(dist2_df)):
        model.C2.add(expr= servic[i, i] ==0)    #Imperdir nó de sair e chegar em si próprio
    
    
    model.restS = pyo.ConstraintList()    
    for s in subciclo:
        model.restS.add(expr =sum(servic[i, j] for i in s for j in s) - len(s) <=  -1)
        
    
    model.obj = pyo.Objective(expr= sum(servic[i, j]*dist2_df[str(i)][j] for i in range(len(geo_df)) for j in range(len(geo_df))), sense=minimize)
    
    
    
    opt = SolverFactory('gurobi')
    opt.solve(model)
    

    
    result = []
    resultado = []
    
    
    vet_mult = list(range(len(dist2_df)))
    for i in range(len(dist2_df)):
        resultado.append(int(sum(pyo.value(servic[i,j])*vet_mult[j] for j in range(len(dist2_df)))))
        
    geo_df['Destino'] = resultado
    
    
    ciclo = [geo_df['chave'][0]]
    subrotas = []
    arestas = []
    j=0
    
    control = ''
    
    while control != 'done':
    
        for i in range(len(geo_df)):
            ciclo.append(geo_df['Destino'][j])
            arestas.append(geo_df['Destino'][j])
            j=geo_df['Destino'][j]
        
        ciclo = list(dict.fromkeys(ciclo))
        
        subrotas.append(ciclo)
        arest_tot = sum(list(map(lambda elemento: len(elemento), subrotas)))
        dif = list(set(arestas).symmetric_difference(set(list(geo_df['chave']))))
        try:
            j = dif[0]
            ciclo = [dif[0]]
        except IndexError:
            control = 'done'
    
    subciclo +=subrotas

print(f'Otimização concluída em {round(time.time() - durac, 2)} segundos\nDistância total: {round(pyo.value(model.obj), 2)}')

# %%

#### mapa dos trajetos

# Mapa Rj

map_url = 'https://geoftp.ibge.gov.br/cartas_e_mapas/bases_cartograficas_continuas/bc25/rj/versao2016/shapefile/lim.zip'


map_zip = BytesIO(requests.get(map_url).content)
map_zip = zipfile.ZipFile(map_zip)

map_zip.extract(    
    member='LIM_Municipio_A.shp')

map_zip.extract(
    member='LIM_Municipio_A.shx')

mapa = geopandas.read_file('LIM_Municipio_A.shp')


trajetos_df = geo_df[['chave', 'Latitude', 'Longitude', 'Destino']].rename(
    columns={'chave': 'Origem'})

trajetos_df = pd.merge(left=trajetos_df, right=geo_df[['Longitude', 'Latitude']], left_on='Destino', right_index=True, how='left')


with sns.axes_style('dark'):
    # Plotagem do mapa
    
    fig, ax = plt.subplots()
    # Layer RJ
    mapa.plot(ax=ax, alpha=0.8,color='slategrey')
    # Coordenadas dos serviços
    plt.plot(list(geo_df['Longitude']), list(geo_df['Latitude']), 'ko')
    
    # Trajetos
    for i in range(len(trajetos_df)):
        a = [trajetos_df["Longitude_x"][i], trajetos_df["Longitude_y"][i]]
        b = [trajetos_df["Latitude_x"][i], trajetos_df["Latitude_y"][i]]
        plt.plot(a,b, linewidth = 2, linestyle = "-", color = "chocolate")
        
    plt.title("Menor trajeto - RJ", fontdict={'fontsize': 20, 'horizontalalignment': 'center','verticalalignment': 'bottom'})
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    
    
    fig.set_figheight(10, forward=True)
    fig.set_figwidth(15, forward=True)
    
    plt.savefig("C:/Users/ferna/OneDrive/Área de Trabalho/Curso OR em Python/trajetos.png")

