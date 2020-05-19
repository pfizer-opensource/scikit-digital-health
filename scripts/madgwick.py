from numpy import array, zeros, cross, sqrt, abs as nabs, arccos, sin, mean, identity, sum, outer
from numpy.linalg import norm, inv as np_inv


__all__ = ['quat_mult', 'quat_conj', 'quat_inv', 'quat2matrix', 'vec2quat', 'MadgwickAHRS']


def quat_mult(q1, q2):
    """
    Multiply quaternions
    Parameters
    ----------
    q1 : numpy.ndarray
        1x4 array representing a quaternion
    q2 : numpy.ndarray
        1x4 array representing a quaternion
    Returns
    -------
    q : numpy.ndarray
        1x4 quaternion product of q1*q2
    """
    if q1.shape != (1, 4) and q1.shape != (4, 1) and q1.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q1 has more or less than 4 elements')
    if q2.shape != (1, 4) and q2.shape != (4, 1) and q2.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q2 has more or less than 4 elements')
    if q1.shape == (4, 1):
        q1 = q1.T

    Q = array([[q2[0], q2[1], q2[2], q2[3]],
               [-q2[1], q2[0], -q2[3], q2[2]],
               [-q2[2], q2[3], q2[0], -q2[1]],
               [-q2[3], -q2[2], q2[1], q2[0]]])

    return q1 @ Q


def quat_conj(q):
    """
    Compute the conjugate of a quaternion
    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of N quaternions to compute the conjugate of.
    Returns
    -------
    q_conj : numpy.ndarray
        Nx4 array of N quaternion conjugats of q.
    """
    return q * array([1, -1, -1, -1])


def quat_inv(q):
    """
    Invert a quaternion
    Parameters
    ----------
    q : numpy.ndarray
        1x4 array representing a quaternion
    Returns
    -------
    q_inv : numpy.ndarray
        1x4 array representing the inverse of q
    """
    # TODO change this, then remove the test that it doesn't work on arrays of quaternions
    if q.size != 4:
        raise ValueError('Not currently implemented for arrays of quaternions.')
    q_conj = q * array([1, -1, -1, -1])
    return q_conj / sum(q ** 2)


def quat2matrix(q):
    """
    Transform quaternion to rotation matrix
    Parameters
    ----------
    q : numpy.ndarray
        Quaternion
    Returns
    -------
    R : numpy.ndarray
        Rotation matrix
    """
    if q.ndim == 1:
        s = norm(q)
        R = array([[1 - 2 * s * (q[2] ** 2 + q[3] ** 2), 2 * s * (q[1] * q[2] - q[3] * q[0]),
                    2 * s * (q[1] * q[3] + q[2] * q[0])],
                   [2 * s * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * s * (q[1] ** 2 + q[3] ** 2),
                    2 * s * (q[2] * q[3] - q[1] * q[0])],
                   [2 * s * (q[1] * q[3] - q[2] * q[0]), 2 * s * (q[2] * q[3] + q[1] * q[0]),
                    1 - 2 * s * (q[1] ** 2 + q[2] ** 2)]])
    elif q.ndim == 2:
        s = norm(q, axis=1)
        R = array([[1 - 2 * s * (q[:, 2]**2 + q[:, 3]**2), 2 * s * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0]),
                    2 * s * (q[:, 1] * q[:, 3] + q[:, 2] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0]), 1 - 2 * s * (q[:, 1]**2 + q[:, 3]**2),
                    2 * s * (q[:, 2] * q[:, 3] - q[:, 1] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0]), 2 * s * (q[:, 2] * q[:, 3] + q[:, 1] * q[:, 0]),
                    1 - 2 * s * (q[:, 1]**2 + q[:, 2]**2)]])
        R = R.transpose([2, 0, 1])
    return R


def quat_mean(q):
    """
    Calculate the mean of an array of quaternions
    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of quaternions
    Returns
    -------
    q_mean : numpy.array
        Mean quaternion
    """
    M = q.T @ q

    vals, vecs = eig(M)  # Eigenvalues and vectors of M
    sort_ind = argsort(vals)  # get the indices of the eigenvalues sorted

    q_mean = real(vecs[:, sort_ind[-1]])

    # ensure no discontinuities
    if q_mean[0] < 0:
        q_mean *= -1

    return q_mean


def vec2quat(v1, v2):
    """
    Find the rotation quaternion between two vectors. Rotate v1 onto v2
    Parameters
    ----------
    v1 : numpy.ndarray
        Vector 1
    v2 : numpy.ndarray
        Vector 2
    Returns
    -------
    q : numpy.ndarray
        Quaternion representing the rotation from v1 to v2
    """
    if allclose(v1, v2):
        return array([1, 0, 0, 0])
    else:
        angle = arccos(dot(v1.flatten(), v2.flatten()) / (norm(v1) * norm(v2)))

        # Rotation axis is always normal to two vectors
        axis = cross(v1.flatten(), v2.flatten())
        axis = axis / norm(axis)  # normalize

        q = zeros(4)
        q[0] = cos(angle / 2)
        q[1:] = axis * sin(angle / 2)
        q /= norm(q)

        return q
    
    
class MadgwickAHRS:
    def __init__(self, sample_period=1/256, q_init=array([1, 0, 0, 0]), beta=0.041):
        """
        Algorithm for estimating the orientation of an inertial sensor with or without a magnetometer.
        Parameters
        ----------
        sample_period : float, optional
            Sampling period in seconds.  Default is 1/256 (ie sampling frequency is 256Hz).
        q_init : numpy.ndarray, optional
            Initial quaternion estimate.  Default is [1, 0, 0, 0].
        beta : float, optional
            Beta value for the algorithm.  Default is 1.0
        References
        ----------
        S. Madgwick et al. "Estimation of IMU and MARG orientation using a gradient descent algorithm." *IEEE Intl.
        Conf. on Rehab. Robotics*. 2011.
        """
        self.sample_period = sample_period
        self.q = q_init
        self.beta = beta

    def update(self, gyr, acc, mag):
        """
        Perform update step.
        Parameters
        ----------
        gyr : numpy.ndarray
            Angular velocity at time t.  Units of rad/s.
        acc : numpy.ndarray
            Acceleration at time t.  Units of g.
        mag : numpy.ndarray
            Magnetometer reading at time t.
        Returns
        -------
        q : numpy.ndarray
            Quaternion estimate of orientation at time t.
        """
        # short name for the quaternion
        q = self.q

        # normalize accelerometer measurement
        a = acc / norm(acc)

        # normalize magnetometer measurement
        h = mag / norm(mag)

        # reference direction of earth's magnetic field
        h_ref = quat_mult(q, quat_mult(array([0, h[0], h[1], h[2]]), quat_conj(q)))
        b = array([0, norm(h_ref[1:3]), 0, h_ref[3]])

        # Gradient Descent algorithm corrective step
        F = array([2 * (q[1] * q[3] - q[0] * q[2]) - a[0],
                   2 * (q[0] * q[1] + q[2] * q[3]) - a[1],
                   2 * (0.5 - q[1]**2 - q[2]**2) - a[2],
                   2 * b[1] * (0.5 - q[2]**2 - q[3]**2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - h[0],
                   2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - h[1],
                   2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1]**2 - q[2]**2) - h[2]])
        J = array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                   [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                   [0, -4 * q[1], -4 * q[2], 0],
                   [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
                   [-2 * (b[1] * q[3] - b[3] * q[1]), 2 * (b[1] * q[2] + b[3] * q[0]), 2 * (b[1] * q[1] + b[3] * q[3]), -2 * (b[1] * q[0] - b[3] * q[2])],
                   [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]])

        step = J.T @ F
        step /= norm(step)  # normalize step magnitude

        # compute rate of change of quaternion
        qDot = 0.5 * quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

        # integrate to yeild quaternion
        self.q = q + qDot * self.sample_period
        self.q /= norm(self.q)

        return self.q

    def updateIMU(self, gyr, acc):
        """
        Perform update step using only gyroscope and accelerometer measurements
        Parameters
        ----------
        gyr : numpy.ndarray
            Angular velocity at time t.  Units of rad/s.
        acc : numpy.ndarray
            Acceleration at time t.  Units of g.
        Returns
        -------
        q : numpy.ndarray
            Quaternion estimate of orientation at time t.
        """
        a = acc / norm(acc)  # normalize accelerometer magnitude
        q = self.q  # short name
        # gradient descent algorithm corrective step
        F = array([2 * (q[1] * q[3] - q[0] * q[2]) - a[0],
                   2 * (q[0] * q[1] + q[2] * q[3]) - a[1],
                   2 * (0.5 - q[1]**2 - q[2]**2) - a[2]])
        J = array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                   [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                   [0, -4 * q[1], -4 * q[2], 0]])
        step = J.T @ F
        step /= norm(step)  # normalise step magnitude

        # compute rate of change quaternion
        q_dot = 0.5 * quat_mult(q, array([0, gyr[0], gyr[1], gyr[2]])) - self.beta * step

        # integrate to yeild quaternion
        q = q + q_dot * self.sample_period
        self.q = q / norm(q)  # normalise quaternion

        return self.q
