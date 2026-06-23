from pesummary.utils.utils import logger

def pesummary_to_pycbc(samples, include_extra_keys=True):
    """
    Converts the keys in data_dict from PESummary format to PyCBC format

    Parameters:
    -----------
    samples: dict
        dictionary with parameter samples and keys in PESummary format
    include_extra_keys: bool, optional [Default: True]
        If True, return dictionary includes keys which were not mapped to PyCBC key words as it is.

    Returns:
    --------
    dictionary containing the parameter samples but with key names modified according to PyCBC
    """

    PESummary_to_PyCBC_map = {}
    
    for key in ['mass_1', 'mass_2', 'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'lambda_1', 'lambda_2']:
        PESummary_to_PyCBC_map[key] = ''.join(key.split('_'))
        
    PESummary_to_PyCBC_map['chirp_mass'] = 'mchirp'
    PESummary_to_PyCBC_map['mass_ratio'] = 'q'
    PESummary_to_PyCBC_map['symmetric_mass_ratio'] = 'eta'
    PESummary_to_PyCBC_map['total_mass'] = 'mtotal'
    PESummary_to_PyCBC_map['final_mass'] = 'final_mass'

    PESummary_to_PyCBC_map['phase'] = 'coa_phase'
    PESummary_to_PyCBC_map['geocent_time'] = 'trigger_time'
    PESummary_to_PyCBC_map['ra'] = 'ra'
    PESummary_to_PyCBC_map['dec'] = 'dec'
    PESummary_to_PyCBC_map['psi'] = 'polarization'
    PESummary_to_PyCBC_map['reference_frequency'] = 'f_ref'
    
    PESummary_to_PyCBC_map['a_1'] = 'spin1_a'
    PESummary_to_PyCBC_map['a_2'] = 'spin2_a'
    PESummary_to_PyCBC_map['tilt_1'] = 'spin1_polar'
    PESummary_to_PyCBC_map['cos_tilt_1'] = 'cos_spin1_polar'
    PESummary_to_PyCBC_map['tilt_2'] = 'spin2_polar'
    PESummary_to_PyCBC_map['cos_tilt_2'] = 'cos_spin2_polar'
    PESummary_to_PyCBC_map['phi_1'] = 'spin1_azimuthal'
    PESummary_to_PyCBC_map['phi_2'] = 'spin2_azimuthal'
    PESummary_to_PyCBC_map['chi_eff'] = 'chi_eff'
    PESummary_to_PyCBC_map['chi_p'] = 'chi_p'

    PESummary_to_PyCBC_map['luminosity_distance'] = 'distance'
    PESummary_to_PyCBC_map['redshift'] = 'redshift'

    for key in samples.keys():
        if '_source' in key:
            PESummary_to_PyCBC_map[key] = 'src' + PESummary_to_PyCBC_map[key.split('_source')[0]]
    
    if 'iota' in samples.keys():
        PESummary_to_PyCBC_map['iota'] = 'inclination'
    else:
        PESummary_to_PyCBC_map['theta_jn'] = 'inclination'

    pycbc_samples = {}
    extra_samples = {}
    
    for key in samples.keys():
        if key not in PESummary_to_PyCBC_map.keys():
            extra_samples[key] = samples[key]
        else:
            if key == 'mass_ratio':
                pycbc_samples[PESummary_to_PyCBC_map[key]] = 1/samples[key]
            else:
                pycbc_samples[PESummary_to_PyCBC_map[key]] = samples[key]
    logger.info(f"Keys without corresponding PyCBC names: {extra_samples.keys()}")

    if include_extra_keys:
        pycbc_samples.update(extra_samples)

    return(pycbc_samples)
