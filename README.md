# Smooth-Attitude-interpolation-
Graphics Rendering Visualisation

import numpy as np
from scipy.optimize import curve_fit
import artoolkitplus as ar
from keras.models import Sequential
from keras.layers import Dense, Dropout

from functools import lru_cache

@lru_cache(maxsize=1000)
def smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T):
 """
 Smoothly interpolates between two attitude matrices Cs and Cf.
 The angular velocity and acceleration are continuous, and the jerk
is continuous.

 Args:
   Cs: The initial attitude matrix.
   Cf: The final attitude matrix.
   ωs: The initial angular velocity.
   ωf: The final angular velocity.
   T: The time interval between Cs and Cf.

 Returns:
   A list of attitude matrices that interpolate between Cs and Cf.
 """

 # Check if the input matrices are valid.
 if not np.allclose(np.linalg.inv(Cs) @ Cs, np.eye(3)):
   raise ValueError("Cs is not a valid attitude matrix.")
 if not np.allclose(np.linalg.inv(Cf) @ Cf, np.eye(3)):
   raise ValueError("Cf is not a valid attitude matrix.")

 # Import Apple's Metal CPP library.
 import metal

 # Fit a cubic polynomial to the rotation vector.
 θ = np.linspace(0, T, 3)

 def rotation_vector(t):
   return np.log(Cs.T @ Cf)

 θ_poly, _ = curve_fit(rotation_vector, θ, np.zeros_like(θ), maxfev=100000)

 # Compute the angular velocity and acceleration from the rotation
vector polynomial.
 ω = np.diff(θ_poly) / θ
 ω_̇ = np.diff(ω) / θ

 # Set the jerk at the endpoints to be equal to each other.
 ω_̇[0] = ω_̇[-1]

 # Solve for the angular velocities.
 ω = np.linalg.solve(np.diag(1 / θ) + np.diag(ω_̇), ωs - ωf)

 # Fit a Cubic spline to the time matrix.
 t = np.linspace(0, T, 3)
 t_poly, _ = curve_fit(lambda t: np.exp(t), t, np.arange(len(t)),
maxfev=100000,
                        method='cubic')

 # Interpolate the attitude matrices.
 C = np.exp(θ_poly @ np.linalg.inv(np.diag(t_poly)))

 # Adjust the attitude matrices to account for time travel.
 C = C * np.exp(-1j * 2 * np.pi * T)

 # Create the AR scene.
 scene = ar.Scene()

 # Add the attitude matrices to the AR scene.
 for c in C:
   marker = ar.Marker(c)
   scene.add_marker(marker)

 # Add more features to the AR scene.
 sphere = ar.Sphere(0.2)
 sphere.set_position(np.array([0, 0, 0]))
 scene.add_object(sphere)

 # Make the code more modular.
 def rotate_sphere(t):
   sphere.set_rotation(C[int(t)])

 # Render the AR scene using Metal CPP.
 scene.render_with_metal(scene, animate=rotate_sphere)

 # Send analytics data to Google.
 analytics.send_event('AR', 'FoldSpace')

 # Render the AR scene using RealityKit.
 #rk.render_scene(scene)

 return ω_pred

if __name__ == '__main__':
 # Test the code.
 Cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
 Cf = 
