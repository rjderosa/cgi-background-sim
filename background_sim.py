import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
import multiprocessing as mp
from astropy import units as u
from astropy import constants
from scipy import interpolate as interp

def regrid(data, param):

    # Create coordinate array of size (s, length) (i.e. (3,100) for 3 parameters of 100 models)
    length = np.shape(data)

    # Calculate axes for n-D grid
    model_axes = [np.unique(i) for i in param]
    model_coords = [np.arange(len(i), dtype = int) for i in model_axes]

    # Get the lengths of each axis
    s = [len(i) for i in model_axes]#.append(length[1])

    final_coords = np.full((len(s), length[0]), -1, dtype=int)
    for i in xrange(0, len(model_axes)):
        for j in xrange(0, len(model_axes[i])):
            ind = np.where(param[i] == model_axes[i][j])[0]
            if len(ind) > 0:
                final_coords[i][ind] = model_coords[i][j]

    if np.shape(final_coords)[1] != np.shape(np.unique(final_coords, axis=1))[1]:
        print 'WARNING: Multiple models for a given coordinate! Using last model in list, but this should be avoided'

    final_coords = tuple([x for x in final_coords])
    if np.ndim(data) == 1:
        regular_grid = np.full(s, -np.inf, dtype=np.float64)
        regular_grid[final_coords] = data
    else:
        tmp_grid = np.full(s, -np.inf, dtype=np.float64)
        s.append(length[1])
        regular_grid = np.full(s, -np.inf, dtype=np.float64)

        for i in xrange(0, length[1]):
            tmp_grid[final_coords] = data[:, i]
            regular_grid[..., i] = tmp_grid

    return regular_grid, model_axes

def read_besancon(file, synth_logflux, synth_logflux_lowtemp, zp, vm, al_av):

    """
    Read besancon simlulation output and returns fundamental star properties
    
    Args:
        file: File name

    Returns:
        some variables

    """

    data = fits.open(file)[1].data
    logteff = np.log10(data['teff']) # Kelvin
    logg = data['logg'] # dex
    mh = data['[M/H]'].clip(-4.0, 0.5) # dex
    rad = (data['radius']*u.R_sun).to(u.m).value # meters
    av = data['av']
    dist = (data['dist']*u.kpc).to(u.m).value
    vmag = data['V']

    return data['Mass'], data['V'], np.transpose(synthetic_mags(logteff, logg, mh, rad, av, dist, vmag, synth_logflux, synth_logflux_lowtemp, zp, vm, al_av))

def read_trilegal(file, synth_logflux, synth_logflux_lowtemp, zp, vm, al_av):

    """
    Read TRILEGAL simlulation output and returns fundamental star properties
    
    Args:
        file: File name

    Returns:
        some variables

    """

    data = ascii.read(file)
    logteff = data['logTe'].data
    logg = data['logg'].data
    mh = data['[M/H]'].data.clip(-4.0, 0.5)
    mass_si = (((data['Mact'].data)*u.M_sun).to(u.kg)).value
    rad = np.sqrt((constants.G.value * mass_si) / ((10**logg)/100.0)) # meters
    av = data['Av'].data
    dist = ((10**((data['m-M0']/5.0)+1.0))*u.pc).to(u.m).value
    vmag = data['V'].data

    return data['Mact'].data, data['V'].data, np.transpose(synthetic_mags(logteff, logg, mh, rad, av, dist, vmag, synth_logflux, synth_logflux_lowtemp, zp, vm, al_av))

def synthetic_mags(logteff, logg, mh, rad, av, dist, vmag, synth_logflux, synth_logflux_lowtemp, zp, vm, al_av):

    synth_mag = []
    for i in range(0, len(logteff)):
        if logg[i] < 6.0:
            if 10**logteff[i] < 2600.0:
                result = synth_logflux_lowtemp((logteff[i], np.clip(logg[i], 4.50001, 5.49999)))
            else:
                result = synth_logflux((logteff[i], np.clip(logg[i], -0.5, 5.5), mh[i]))

            if np.any(np.isinf(result)) | np.any(np.isnan(result)):
                # Out of bounds
                print(10**logteff[i], logg[i], mh[i], rad[i])
            else:
                result = (10**result) * ((rad[i]**2.0)/(dist[i]**2.0)) # Apparent flux
                result = (-2.5*np.log10(result/zp)) + vm # Apparent magnitude
            synth_mag.append(result + (av[i]*al_av))
        else:
            # Going to use Vmag for white dwarfs
            synth_mag.append(np.repeat(vmag[i], len(zp)))

    return synth_mag


def main():

    d = 2.37*u.m

    ## Load narrow contrast curve
    l_narrow = 575*u.nm # 10%
    ld_narrow = (((l_narrow/d).to(1))*(u.rad).to(u.deg)).value # degrees
    wfirst_narrow = ascii.read('WFIRST_pred_imaging.txt')
    wfirst_narrow_sep = wfirst_narrow['l/D'].data * ld_narrow
    wfirst_narrow_con = -2.5*np.log10(wfirst_narrow['contr_snr5'].data)
    wfirst_narrow_int = interp.interp1d(wfirst_narrow_sep, wfirst_narrow_con, bounds_error=False, fill_value=1.0)

    ## Load wide contrast curve
    l_wide = 852*u.nm # 10%
    ld_wide = (((l_wide/d).to(1))*(u.rad).to(u.deg)).value # degrees
    wfirst_wide = ascii.read('WFIRST_pred_disk.txt')
    wfirst_wide_sep = wfirst_wide['Rho(as)'].data / 3600.0 # degrees
    wfirst_wide_con = -2.5*np.log10(wfirst_wide['Band4_contr_snr5'].data)
    wfirst_wide_int = interp.interp1d(wfirst_wide_sep, wfirst_wide_con, bounds_error=False, fill_value=1.0)

    ## Load IFS contrast curve
    l_spec = 660*u.nm # 18%
    ld_spec = (((l_spec/d).to(1))*(u.rad).to(u.deg)).value # degrees
    wfirst_spec = ascii.read('WFIRST_pred_spec.txt')
    wfirst_spec_sep = wfirst_spec['l/D'].data * ld_spec
    wfirst_spec_con = -2.5*np.log10(wfirst_spec['contr_snr10'].data/2.0) #convert 10 to 5 sigma
    wfirst_spec1_int = interp.interp1d(wfirst_spec_sep, wfirst_spec_con, bounds_error=False, fill_value=1.0)

    l_spec = 730*u.nm # 18%
    ld_spec = (((l_spec/d).to(1))*(u.rad).to(u.deg)).value # degrees
    wfirst_spec_sep = wfirst_spec['l/D'].data * ld_spec
    wfirst_spec_con = -2.5*np.log10(wfirst_spec['contr_snr10'].data/2.0) #convert 10 to 5 sigma
    wfirst_spec2_int = interp.interp1d(wfirst_spec_sep, wfirst_spec_con, bounds_error=False, fill_value=1.0)

    l_spec = 760*u.nm # 18%
    ld_spec = (((l_spec/d).to(1))*(u.rad).to(u.deg)).value # degrees
    wfirst_spec_sep = wfirst_spec['l/D'].data * ld_spec
    wfirst_spec_con = -2.5*np.log10(wfirst_spec['contr_snr10'].data/2.0) #convert 10 to 5 sigma
    wfirst_spec3_int = interp.interp1d(wfirst_spec_sep, wfirst_spec_con, bounds_error=False, fill_value=1.0)
    wfirst_spec_int = (wfirst_spec1_int, wfirst_spec2_int, wfirst_spec3_int)

    ## Opening angle of bowtie    
    wedge_angle = 65.0 * np.pi/180.0

    ## Contrast curves for other instruments
    ## STIS
    stis = ascii.read('HST_STIS.txt')
    stis_sep = stis['Rho(as)'].data / 3600.0
    stis_con = stis['KLIP_Contr'].data

    ## Extend STIS contrast curve to 20"
    ## log(con)=-2.8277*log(sep[deg]) - 16.557
    extension = np.linspace(np.max(stis_sep), 20.0/3600.0, 100.0)
    stis_sep = np.append(stis_sep, extension)
    stis_con = np.append(stis_con, 10**((-2.8277*np.log10(extension)) - 16.557))

    # Store remaining instrument contrast objects and filter names in a list
    inst_con = []
    inst_filter = []
    inst_name = []

    ## CHARIS
    charis = ascii.read('CHARIS_IFS_H_contrast.txt')
    charis_con = interp.interp1d(charis['arcsec'].data/3600.0, -2.5*np.log10(charis['5sigHcontr'].data), kind='linear', bounds_error=False, fill_value=1.0)
    inst_con.append(charis_con)
    inst_filter.append('2MASS_H')
    inst_name.append('CHARIS_IFS')

    ## NIRC2 (PALMS)
    nirc2 = ascii.read('NIRC2_PALMS.txt')
    nirc2_con = interp.interp1d(nirc2['arcsec'].data/3600.0, nirc2['dmag'].data, kind='linear', bounds_error=False, fill_value=1.0)
    inst_con.append(nirc2_con)
    inst_filter.append('2MASS_H')
    inst_name.append('NIRC2_PALMS')


    n_inst = len(inst_name)


    ## Define filters and compute extinction from Cardelli et al. 1989
    filters = ['STIS_50CCD', 'CGI_narrow', 'CGI_wide', 'CGI_IFS660', 'CGI_IFS730', 'CGI_IFS760', '2MASS_H']
    y = (1.0/np.array([0.556, 0.574, 0.819, 0.654, 0.721, 0.749, 1.64])) - 1.82
    a = 1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    al_av = a + (b/3.1)

    ## Read in synthetic magnitudes
    synth = ascii.read('WFIRST_fluxes_BTSettlAGSS2009.txt')
    logteff = np.log10(synth['teff'].data)
    logg = synth['logg'].data
    mh = synth['mh'].data

    synth_logflux = np.zeros((len(logteff), len(filters)))
    for i, f in enumerate(filters):
        synth_logflux[:, i] = synth[f].data

    ## Load zero points
    data = ascii.read('WFIRST_zeropoints.txt')
    zp = []
    vm = []
    for f in filters:
        zp.append(data['zp'].data[list(data['filter'].data).index(f)])
        vm.append(data['vm'].data[list(data['filter'].data).index(f)])

    # Create interpolator object
    grid, axes = regrid(synth_logflux, (logteff, logg, mh))
    synth_interp = interp.RegularGridInterpolator(axes, grid, method='linear', bounds_error =False, fill_value=-np.inf)

    # Create low-temperature object
    ind = np.where(mh == 0.0)
    grid, axes = regrid(synth_logflux[ind], (logteff[ind], logg[ind]))
    synth_interp_lowtemp = interp.RegularGridInterpolator(axes, grid, method='linear', bounds_error=False, fill_value=-np.inf)

    starlist = ascii.read('starlist.txt')
    n = len(starlist['Name'].data)

    ## star proper motion converted from mas/yr to deg/yr
    print_header = True
    for dupl, name, star_pmra, star_pmde, star_vmag, size_b, size_t, besancon, trilegal in zip(starlist['Dupl?'].data, starlist['Name'].data, \
                                                                            starlist['st_pmra'].data/(3600.0*1e3), starlist['st_pmdec'].data/(3600.0*1e3), \
                                                                            starlist['st_vmag'].data, starlist['size_b'].data, starlist['size_t'].data,
                                                                            starlist['Besancon'].data, starlist['Trilegal'].data):


        ## For STIS curve
        stis_bkg = 10**((26.0-star_vmag)/(-2.5))

        ## Fit straight line to contrast curve in log-log space, then extrapolate to 20"
        stis_con_noclip = stis_con/(10**((3.60-star_vmag)/(-2.5)))
        stis_con_clip =np.clip(stis_con/(10**((3.60-star_vmag)/(-2.5))), stis_bkg, None) 
        stis_int = interp.interp1d(stis_sep, -2.5*np.log10(stis_con_clip), bounds_error=False, fill_value=1.0)
        
        fig, ax = plt.subplots(1, figsize=(3, 3))
        ax.plot(stis_sep*3600.0, stis_con_noclip, color='gray')
        ax.plot(stis_sep*3600.0, stis_con_clip, color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Separation (asec)')
        ax.set_ylabel('Contrast')
        fig.savefig('STIScontrast-'+name.replace(' ','_')+'.png', dpi=300, bbox_inches='tight')
        plt.close('all')

        sim_type = 'Besancon'

        ## Only do non-duplicate entries on the list
        if (dupl == 0) and (name == 'HD 150706'):

            ## For each star, run n_sim simulations with both besancon and trilegal
            n_sim = 20000

            if sim_type == 'Besancon':
                besancon_name = 'Besancon/output'+str(besancon)+'.fits.gz'
                mass, vmag, synth_mag = read_besancon(besancon_name, synth_interp, synth_interp_lowtemp, zp, vm, al_av)
                size = size_b
            elif sim_type == 'TRILEGAL':
                trilegal_name = 'TRILEGAL/'+str(trilegal)+'.dat.gz'
                mass, vmag, synth_mag = read_trilegal(trilegal_name, synth_interp, synth_interp_lowtemp, zp, vm, al_av)
                size = size_t

            #mass, vmag = read_trilegal(trilegal_name)
            delta_vmag = vmag - star_vmag
            delta_mag = synth_mag - star_vmag

            wfirst_ra, wfirst_de = 0.0, 0.0 # Star is at (0, 0) in 2028
            baseline = 8.0 # years between STIS and WFIRST observations
            current_ra, current_de = -(star_pmra*baseline), -(star_pmde*baseline) # Location of star in 2019

            ## Random orientation for IFS bowtie
            spec_orient = np.random.uniform(0.0, 2.0*np.pi, n_sim)

            ## Save results in a (6+n_inst, n) array
            ## [narrow, wide, ifs660, ifs730, ifs760, STIS, inst1, inst2]
            flag = np.zeros((6+n_inst, 0), dtype=bool) #660

            for i in range(0, n_sim):
                bkg_ra, bkg_de = np.random.uniform(-np.sqrt(size)/2.0, np.sqrt(size)/2.0, len(mass)), np.random.uniform(-np.sqrt(size)/2.0, np.sqrt(size)/2.0, len(mass))
                rho = np.sqrt(bkg_ra**2.0 + bkg_de**2.0)
                
                ## How many are within 2 asec
                indx = np.where((rho < (2.0/3600.0)))[0]
                if len(indx) > 0:
                    
                    new_ra = bkg_ra[indx] - current_ra
                    new_de = bkg_de[indx] - current_de
                    current_rho = np.sqrt(new_ra**2.0 + new_de**2.0)

                    stis_indx = filters.index('STIS_50CCD')
                    stis_detected = np.array(stis_int(current_rho) > delta_mag[stis_indx][indx], dtype=bool)

                    inst_flags = []
                    for j in range(0, n_inst):
                        inst_indx = filters.index(inst_filter[j])
                        inst_flags.append(np.array(inst_con[j](current_rho) > delta_mag[inst_indx][indx], dtype=bool))

                    ## Narrow
                    narrow_indx = filters.index('CGI_narrow')
                    narrow_detected = np.array(wfirst_narrow_int(rho[indx]) > delta_mag[narrow_indx][indx], dtype=bool)

                    ## Wide
                    wide_indx = filters.index('CGI_wide')
                    wide_detected = np.array(wfirst_wide_int(rho[indx]) > delta_mag[wide_indx][indx], dtype=bool)

                    ## IFS
                    for j in range(0, 3):
                        if j == 0: ifs_indx = filters.index('CGI_IFS660')
                        if j == 1: ifs_indx = filters.index('CGI_IFS730')
                        if j == 2: ifs_indx = filters.index('CGI_IFS760')

                        detected = wfirst_spec_int[j](rho[indx]) > delta_mag[ifs_indx][indx]

                        # Spec FoV is (0+orient) pm 32.5, (180+orient) pm 32.5 (0.57 radians)
                        # Work out angle between background star and the midpoint of each of the wedges
                        pa = np.arctan2(-bkg_ra[indx], bkg_de[indx]) % (2.0*np.pi)
                        delta_pa1 = np.abs(np.arctan2(np.sin(pa - (0.0+spec_orient[i])), np.cos(pa - (0.0+spec_orient[i]))))
                        delta_pa2 = np.abs(np.arctan2(np.sin(pa - (np.pi+spec_orient[i])), np.cos(pa - (np.pi+spec_orient[i]))))

                        detected = detected & ((delta_pa1 <= (wedge_angle/2.0)) | (delta_pa2 <= (wedge_angle/2.0)))

                        if j == 0: ifs_detected1 = np.array(detected, dtype=bool)
                        if j == 1: ifs_detected2 = np.array(detected, dtype=bool)
                        if j == 2: ifs_detected3 = np.array(detected, dtype=bool)

                    ## Now append flags to results array
                    flag = np.hstack((flag, np.vstack((narrow_detected, wide_detected, ifs_detected1, ifs_detected2, ifs_detected3, stis_detected, inst_flags))))

            ## For each CGI mode: print number of background sources detected by CGI, and number detected by STIS, inst1, ..., instn
            n_sim = float(n_sim)
            result_header = 'name, n_bkg'
            cgi_modes = ('CGI_narrow', 'CGI_wide', 'CGI_660', 'CGI_730', 'CGI_760')
            for j in range(0, len(cgi_modes)):
                result_header += ', '+cgi_modes[j]+', +STIS'
                for k in range(0, n_inst):
                    result_header += ', +'+inst_name[k]

            # Print name, % of sims with background objects within 2"
            result_string = '{:s}, {:.2f}%'.format(name, float(len(flag[0]))/n_sim*100.)

            for j in range(0, len(cgi_modes)):
                # % of sims with background objects detected in this CGI mode, and those recovered with STIS
                result_string += ', {:.2f}%, {:.2f}%'.format(np.sum(flag[j])/n_sim*100.0, np.sum(flag[j] & flag[5])/n_sim*100.0)
                for k in range(0, n_inst):
                    # and those recovered with the other instruments...
                    result_string += ', {:.2f}%'.format(np.sum(flag[j] & flag[6+k])/n_sim*100.0)

            if print_header:
                print result_header
                print_header = False
            print result_string


    return 0

if __name__ == '__main__':
    main()
