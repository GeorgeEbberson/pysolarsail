"""
Constant values and conversions between units.
"""
from numba import float64, jit
from spiceypy import clight, spd

# Number of seconds in a 24-hour day.
SECS_PER_DAY = spd()

# Number of metres per astronomical unit.
# https://www.iau.org/static/resolutions/IAU2012_English.pdf
# Resolution B2 of the XXVIII General Assembly of the IAU.
M_PER_AU = float(149_597_870_700)

# Speed of light in a vacuum.
# https://www.bipm.org/en/publications/si-brochure/
# The International System of Units, 9th Edition, BIPM.
# SPICE gives it in km/s.
SPEED_OF_LIGHT_M_S = clight() * 1000

# Total solar irradiance.
# http://dx.doi.org/10.1029/2010GL045777
# Kopp and Lean, 2011, Geophysical Research Letters Vol 38.
SOLAR_IRRADIANCE_W_M2 = 1360.8


@jit(float64(float64))
def m_to_au(metres: float) -> float:
    """Convert distance in metres to astronomical units."""
    return metres / M_PER_AU


@jit(float64(float64))
def au_to_m(au: float) -> float:
    """Convert distance in astronomical units to metres."""
    return au * M_PER_AU
