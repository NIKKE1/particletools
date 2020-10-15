"""
Event inspector, made for hefaistos server with 12 threads.
Plots delta_R for W-bosons, and particle distributions.
Author: Niklas H
15.10.2020
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'   #12 threads in hefaistos
try:
    import sys
    import uproot
    import logging
    import pandas as pd
    import numpy as np
    import multiprocessing
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from matplotlib.image import NonUniformImage
    from itertools import combinations  #, chain
    from timeit import default_timer as timer
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

except ModuleNotFoundError as error:
    print(error)

np.set_printoptions(threshold=sys.maxsize)

#Data root files: WWjj_SS_ll_hadronic_Extension WWjj_tt_hadronic_HT1000Inf WWjj_ll_hadronic_HT1000Inf_80PercentDone

FILE = 'WWjj_ll_hadronic_HT1000Inf_80PercentDone'
#FILE = 'll_Plain_Small.root' #test sample


def main():
    df = data_loader()
    answer = input("\n\tChoose\n\tCalculate deltaR for W Bosons [1]\n \t   Plot particle distribution [2]: ")
    if answer == '1':
        print(f"Calculating delta_R for particles in {FILE}")
        del_R = particle_iterator(deltaRs, df)
        plot_hist(del_R)
    elif answer == '2':
        mass, num = particle_iterator(particle_ids, df)
        plot_2dhist(mass, num)
    else:
        print("Wrong input, exiting...")
        sys.exit()


def particle_iterator(func, df):
    """
    Partitions dataframe into 20*(number of cores-1) for Pooling.
    Uses thus max available threads-1 for calculations.
    """
    start = timer()

    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    num_partitions = 20*num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)

    print("Loading multiprocessing pool...")

    #combos = pool.map(func, df_split) #if order matters

    calculated_list = []
    for mass_num_list in tqdm(pool.imap_unordered(func, df_split), total=len(df_split), desc = 'Processing: '):
        calculated_list.append(mass_num_list)

    pool.close()
    pool.join()

    end = timer()
    print(f"Elapsed time {end - start}.\n Plotting histogram...")
    if func == particle_ids:
        mass_list, num_list = zip(*calculated_list)
        mass = sum(mass_list, [])
        num = sum(num_list, [])
        return mass, num
    else:
        calculated_list = sum(calculated_list, [])
    return calculated_list

def deltaRs(dataf):
    """
    Will go event by event (row at a time) to find W-bosons (pid == 24, mass > 0)

    Using combinations() for each event to calculate delta R: r-length tuples, in sorted order, no repeated elements.
    Example: print(list(combinations(range(5), 2)))
    >> [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    Dataframe columns: ['GenPart_pdgId', 'GenPart_eta', 'GenPart_phi', 'GenPart_mass', 'GenPart_statusFlags'
    Bitwise: 128 for 7 : isHardProcess  (2^7 = 128)
    Column orders: 'GenPart_pdgId', 'GenPart_eta', 'GenPart_phi', 'GenPart_mass', 'GenPart_statusFlags'
    """
    del_r = []
    for event in range(dataf.shape[0]):
        eta, phi = [], []
        for i, particle_id in enumerate(dataf.iloc[event,0]):
            if particle_id==24 and dataf.iloc[event, 3][i]>0 and np.bitwise_and(dataf.iloc[event,4][i], 0b1000_0000) == 128:

                eta.append(dataf.iloc[event, 1][i])
                phi.append(dataf.iloc[event, 2][i])

        del_r += [deltaR(eta1=eta[particle1], phi1=phi[particle1], eta2=eta[particle2], phi2=phi[particle2])
                        for (particle1, particle2) in combinations(range(len(eta)), 2)]
    return del_r

def deltaR(eta1, phi1, eta2, phi2):
    eta_r = eta1 - eta2
    phi_r = phi1 - phi2
    d_R = np.sqrt((eta_r * eta_r) + (phi_r * phi_r))
    return d_R


def particle_ids(dataf):
    particle_mass, particle_number = [], []
    counter = 0
    for event in range(dataf.shape[0]):
        for i, particle_id in enumerate(dataf.iloc[event,0]):
                 counter += 1
                 if particle_id==24 and (dataf.iloc[event, 3][i]>0): # and (np.bitwise_and(dataf.iloc[event,4][i], 0b1000_0000) == 128 )):
                     particle_number.append(particle_id)
                     particle_mass.append(dataf.iloc[event, 3][i])

    if len(particle_number)>0:
        print(f"Found {len(particle_number)} W+ bosons. Ratio compared to all particles: {len(particle_number)/counter}")

    return particle_mass, particle_number


def data_loader():
    """  ~~~CURRENTLY SAVING IS COMMENTED OUT~~~
    This function will load the data and create a folder called /data where the
    location the python file is run. If the data exists, it will just load it.
    """
    destdir = 'data'
    if not os.path.exists(destdir):
        print(f"{destdir}-folder does not exist, creating...")
        os.makedirs(destdir)
    output = os.path.join(destdir, '%s.h5' % (FILE))
    if os.path.exists(output):
        loaded_data = pd.read_hdf(output, key='table')
        printf(f"Loading data from... {output}")
        return loaded_data
    else:
        #filePathll = '/work/data/VBS/JMEPFNanoAOD/testSamples/ll_Plain_Small.root'
        filePathll = '/work/data/VBS/JMEPFNanoAOD/'+FILE+'.root'
        rootFilell = uproot.open(filePathll)['Events']
        keys = ['GenPart_pdgId', 'GenPart_eta', 'GenPart_phi', 'GenPart_mass', 'GenPart_statusFlags']
        loaded_data = pd.DataFrame()
        print(f"Loading data from {filePathll}")
        for key in keys:
            loaded_data[key] = np.array(rootFilell.array(key))

       # loaded_data.to_hdf(output, key='table', mode='w')

    return loaded_data

def plot_hist(deltaR):
    #10 different numbers should produce 45 combinations for instance
    plt.hist(deltaR, bins=100)
    plt.xlabel('delta R')
    plt.xticks(np.arange(0, max(deltaR), 0.5))
    plt.title(FILE)
    plt.show()

def plot_2dhist(mass, num, pic_side_len = 50):
    titlez = FILE
    histo2d, xedges, yedges = np.histogram2d(mass, num, bins=[pic_side_len, pic_side_len], range=[[70, 120],[18,30]]) #55 - 155
    #histo2d = plt.hist2d(x[0], y[0], bins = [64, 64], range = [[-, 200,],[-55, 55]], weights = v['part_ptrel'][0]) #if matplotlib preferred
    H = histo2d.T  # Let each row list bins with common y range.

    fig = plt.figure(figsize=(100, 100))  # figsize can obviously be smaller/larger
    ax0 = fig.add_subplot(111, title=f"{titlez}")
    ax0.spines['top'].set_color('none')
    ax0.spines['bottom'].set_color('none')
    ax0.spines['left'].set_color('none')
    ax0.spines['right'].set_color('none')
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax1 = fig.add_subplot(121, title='imshow: square bins') #131
    plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap='gray')
    plt.xticks(np.arange(70, 120, 5))
    plt.yticks(np.arange(18, 30, 6))

    ax2 = fig.add_subplot(122, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = NonUniformImage(ax2, interpolation='bilinear')
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax2.images.append(im)
    plt.xticks(np.arange(70, 120, 5))
    plt.yticks(np.arange(18, 30, 6))

    ax1.set_xlabel('Mass [GeV]')
    ax1.set_ylabel('Particle ID')
    ax2.set_xlabel('Mass [GeV]')
    ax2.set_ylabel('Particle ID')

    plt.show()


if __name__ == '__main__':
    main()
