import numpy as np
from unyt import (
    hydrogen_mass,
    boltzmann_constant,
    gravitational_constant,
    parsec,
    solar_mass,
)

# Delete the units from Unyt constants
hydrogen_mass = float(hydrogen_mass.value)
boltzmann_constant = float(boltzmann_constant.value)
G_SI = float(gravitational_constant.value)
G_astro = float(gravitational_constant.in_units('(1e6*pc)*(km/s)**2/(1e10*solar_mass)').value)
parsec = float((1 * parsec).in_units('m').value)
solar_mass = float(solar_mass.value)


#####################################################
#													#
#				COMOVING UNITS      				#
# 									 				#
#####################################################


def comoving_density(header, density):
    scale_factor = 1 / (header['zred'] + 1)
    return np.multiply(density,  header['Hub'] ** 2 * scale_factor ** -3)

def comoving_length(header, coord):
    scale_factor = 1 / (header['zred'] + 1)
    return np.multiply(coord, scale_factor / header['Hub'])

def comoving_velocity(header, vel):
    scale_factor = 1 / (header['zred'] + 1)
    return np.multiply(vel, np.sqrt(scale_factor))

def comoving_mass(header, mass):
    return np.divide(mass, header['Hub'])

def comoving_kinetic_energy(header, kinetic_energy):
    scale_factor = 1 / (header['zred'] + 1)
    _kinetic_energy = np.multiply(kinetic_energy, scale_factor)
    _kinetic_energy = np.divide(_kinetic_energy, header['Hub'])
    return _kinetic_energy

def comoving_momentum(header, mom):
    scale_factor = 1 / (header['zred'] + 1)
    return np.multiply(mom, np.sqrt(scale_factor) / header['Hub'])

def comoving_ang_momentum(header, angmom):
    scale_factor = 1 / (header['zred'] + 1)
    return np.multiply(angmom, np.sqrt(scale_factor**3) / np.power(header['Hub'], 2.))


#####################################################
#													#
#			    UNITS CONEVRSION     				#
# 									 				#
#####################################################

def density_units(density, unit_system='SI'):
    if unit_system == 'SI':
        # kg*m^-3
        conv_factor = 6.769911178294543 * 10 ** -28
    elif unit_system == 'cgs':
        # g*cm^-3
        conv_factor = 6.769911178294543 * 10 ** -31
    elif unit_system == 'astro':
        # solar masses / (parsec)^3
        conv_factor = 6.769911178294543 * np.power(3.086, 3.) / 1.9891 * 10 ** -10
    elif unit_system == 'nHcgs':
        conv_factor = 6.769911178294543 * 10 ** -31 / (1.674 * 10 ** -24.)
    else:
        raise("[ERROR] Trying to convert SPH density to an unknown metric system.")
    return np.multiply(density, conv_factor)

def velocity_units(velocity, unit_system='SI'):
    if unit_system == 'SI':
        # m/s
        conv_factor =  10 ** 3
    elif unit_system == 'cgs':
        # cm/s
        conv_factor =  10 ** 5
    elif unit_system == 'astro':
        # km/s
        conv_factor =  1
    else:
        raise("[ERROR] Trying to convert velocity to an unknown metric system.")
    return np.multiply(velocity, conv_factor)

def length_units(len, unit_system='SI'):
    if unit_system == 'SI':
        conv_factor = 1e6 * parsec
    elif unit_system == 'cgs':
        conv_factor = 1e8 * parsec
    elif unit_system == 'astro':
        conv_factor = 1
    else:
        raise ("[ERROR] Trying to convert length to an unknown metric system.")
    return np.multiply(len, conv_factor)

def mass_units(mass, unit_system='SI'):
    if unit_system == 'SI':
        conv_factor = 1e10 * solar_mass
    elif unit_system == 'cgs':
        conv_factor = 1e13 * solar_mass
    elif unit_system == 'astro':
        conv_factor = 1e10
    else:
        raise("[ERROR] Trying to convert mass to an unknown metric system.")
    return np.multiply(mass, conv_factor)

def momentum_units(momentum, unit_system='SI'):
    if unit_system == 'SI':
        conv_factor = 1.9891 * 10 ** 43
    elif unit_system == 'cgs':
        conv_factor = 1.9891 * 10 ** 48
    elif unit_system == 'astro':
        conv_factor = 10 ** 10
    else:
        raise("[ERROR] Trying to convert mass to an unknown metric system.")
    return np.multiply(momentum, conv_factor)

def energy_units(energy, unit_system='SI'):
    if unit_system == 'SI':
        conv_factor = np.power(10., 46)

    else:
        raise ("[ERROR] Trying to convert mass to an unknown metric system.")
    return np.multiply(energy, conv_factor)