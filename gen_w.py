"""
For each event the script selects W-bosons based on pgid:24, mass>0, isHardProcess : 7.
Calculates delta R to each GenJetAK8, and selects closest GenJetAK8 in event.
Plots masses and deltaR's of these GenJetAK8s

Author: Niklas H
Date: 20.10.2020

"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'   #12 threads in hefaistos
try:
    import sys
    import uproot
    import logging
    import operator
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import multiprocessing
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from itertools import combinations
    from timeit import default_timer as timer
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    #logging levels: critical : 50, error : 40, warning: 30, info: 20, debug: 10
except ModuleNotFoundError as error:
    logging.WARNING(error)
    sys.exit()


def main():
    """
    Comparing W-bosons to GenJetAK8s, and selecting closest ones.
    """
    #TEST = 'JMEPFNanoAOD/testSamples/ll_Plain_Large'
    FILE_LL = 'JMEPFNanoAOD/WWjj_ll_hadronic_HT1000Inf'
    FILE_TT = 'JMEPFNanoAOD/WWjj_tt_hadronic_HT1000Inf'

    key = ['GenPart_pdgId', 'GenPart_eta', 'GenPart_phi', 'GenPart_mass', 'GenPart_statusFlags', 'GenJetAK8_mass', 'GenJetAK8_pt', 'GenJetAK8_eta', 'GenJetAK8_phi']

    """
    df_test = data_loader(key, TEST)
    df_test = _200cut(df_test)
    mass_LL, delr_LL = particle_iterator(w_fj_deltar, df_test)
    print(len(mass_LL))
    print(len(delr_LL))
    """
    
    df_LL = data_loader(key, FILE_LL)
    df_TT = data_loader(key, FILE_TT)

    print(f"Cut-Check: Before 200GeV cut {df_LL.shape[0]}")
    df_LL = _200cut(df_LL)
    print(f"After 200GeV cut {df_LL.shape[0]}")
    df_TT = _200cut(df_TT)

    mass_LL, delr_LL = particle_iterator(df=df_LL, func=w_fj_deltar)
    mass_TT, delr_TT = particle_iterator(df=df_TT, func=w_fj_deltar)

    plot_hist(mass_LL, mass_TT, title='PFNano GenJetAK8 mass closest to W', density=False, \
              x_lbl='mass [GeV]', range=[0, 420], max_ticks=420, ticks=10, bins=210)

    plot_hist(delr_LL, delr_TT, title='PFNano GenJetAK8 del_R closest to W', density=False, \
              x_lbl='delta R', range=[0, 7], max_ticks=7, ticks=0.2, bins=40)

def _200cut(df):
    """
    Cutting PFNano pt to [200, inf) to better compare plots with Deepntuplizer
    """
    pt = df['GenJetAK8_pt'].explode()
    mass = df['GenJetAK8_mass']
    print(f"DF shape before {df.shape[0]}")
    df = df.drop(pt[pt<200].index) #dropping dataframe rows with GenJetAK8_pt's less than 200
    print(f"DF shape after 200GeV cut {df.shape[0]}")
    df = df.drop(mass[mass.str.len().eq(0)].index)
    print(f"DF shape after removing events with len(GenJetAK8s)==0 {df.shape[0]}")
    print(df.shape[0])

    return df

def particle_iterator(func, df):
    """
    Partitions dataframe into 20*(number of cores-1) for Pooling.
    Uses thus max available threads-1 for calculations.
    """
    start = timer()

    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    logging.info(f"Available cores {num_cores}")
    num_partitions = 20*num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)

    logging.info("Loading multiprocessing pool...")

    #combos = pool.map(func, df_split) #if order matters

    calculated_list = []
    for item_in_list in tqdm(pool.imap_unordered(func, df_split), total=len(df_split), desc='Processing: '):
        calculated_list.extend(item_in_list)

    pool.close()
    pool.join()

    end = timer()
    print(f"Elapsed time {end - start}.")

    mass, delr = zip(*calculated_list)

    return mass, delr


def w_fj_deltar(dataf):
    """
    Compared W-boson del_R to GenJetAK8s, and select the closest one.

    Dataframe columns: ['GenPart_pdgId', 'GenPart_eta', 'GenPart_phi', 'GenPart_mass', 'GenPart_statusFlags',
                       'GenJetAK8_mass', 'GenJetAK8_pt', 'GenJetAK8_eta', 'GenJetAK8_phi']
    """
    del_r = []
    gen_mass = []

    for event in dataf.itertuples():
        gen_eta, gen_phi, w_eta, w_phi = [], [], [], []
        bitwise_list = np.bitwise_and(event.GenPart_statusFlags, 0b1000_0000)

        for i, w_boson in enumerate(event.GenPart_pdgId):
            if w_boson==24 and (event.GenPart_mass[i]>0) and (bitwise_list[i] == 128 ):
                w_eta.append(event.GenPart_eta[i])
                w_phi.append(event.GenPart_phi[i])

        for j, gen_m in enumerate(event.GenJetAK8_mass):
                gen_eta.append(event.GenJetAK8_eta[j])
                gen_phi.append(event.GenJetAK8_phi[j])

        if len(w_eta)>0:
            for part_eta, part_phi in zip(w_eta, w_phi):
                temp = []
                for gen_e, gen_p in zip(gen_eta, gen_phi):
                    temp.append(deltaR(eta1=part_eta, phi1=part_phi, eta2=gen_e, phi2=gen_p))


                min_index, min_value = min(enumerate(temp), key=operator.itemgetter(1))  #selecting closest one
                gen_mass.append((event.GenJetAK8_mass[min_index]).item())
                del_r.append(min_value)


    combined_list = zip(gen_mass, del_r) #need to do for Pool to work (?)

   # print(f"type of combined list {combined_list}")
    return combined_list

def deltaR(eta1, phi1, eta2, phi2):
    eta_r = eta1 - eta2
    phi_r = phi1 - phi2
    d_R = np.sqrt((eta_r * eta_r) + (phi_r * phi_r))
    return d_R.item()


def plot_hist(data1, data2=False, title='title', x_lbl='missing', range=[0,200], max_ticks = 200, ticks = 10, bins=100, density = False):
    sns.set()
    sns.color_palette("Paired")
    logging.info("Plotting histogram ...")
    np.set_printoptions(threshold=sys.maxsize)

    plt.hist(data2, alpha = 0.4, label='TT', bins=bins, range = range, density = density)
    plt.hist(data1, alpha = 0.5, bins=bins,label='LL', range=range, density = density)

    plt.xlabel(x_lbl)
    plt.legend()
    plt.xticks(np.arange(0, max_ticks, ticks))
    plt.title(title)
    #plt.savefig('fjmass_nano1.png')
    plt.show()

def data_loader(KEYS, data_file):
    """
    ~~~CURRENTLY SAVING IS REMOVED~~~
    Root file with the selected keys is simply loaded to memory.
    """

    filePath = '/work/data/VBS/'+data_file+'.root'
    rootFile = uproot.open(filePath)['Events']

    loaded_data = pd.DataFrame()
    logging.info(f"Loading data from root file: {filePath}")
    for key in KEYS:
        loaded_data[key] = np.array(rootFile.array(key))

    return loaded_data

if __name__ == '__main__':
    main()










