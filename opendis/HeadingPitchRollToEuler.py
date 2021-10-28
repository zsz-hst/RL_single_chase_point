from math import sin, cos, atan2, sqrt, pow, radians





def RotateAboutAxis(d, s, n, t):
    st = sin(t)
    ct = cos(t)
    d[0] = (1.0 - ct) * (n[0] * n[0] * s[0] + n[0] * n[1] * s[1] + n[0] * n[2] * s[2]) + ct * s[0] + st * (n[1] * s[2] - n[2] * s[1])
    d[1] = (1.0 - ct) * (n[0] * n[1] * s[0] + n[1] * n[1] * s[1] + n[1] * n[2] * s[2]) + ct * s[1] + st * (n[2] * s[0] - n[0] * s[2])
    d[2] = (1.0 - ct) * (n[0] * n[2] * s[0] + n[1] * n[2] * s[1] + n[2] * n[2] * s[2]) + ct * s[2] + st * (n[0] * s[1] - n[1] * s[0])



def Cross(d, a, b):
    d[0] = a[1] * b[2] - b[1] * a[2]
    d[1] = b[0] * a[2] - a[0] * b[2]
    d[2] = a[0] * b[1] - b[0] * a[1]



def memcpy(D1, D, size):
    D1[0] = D[0]
    D1[1] = D[1]
    D1[2] = D[2]


def Dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]



def HeadingPitchRollToEuler(H, P, R, Lat, Lon):
    '''
    :param H:
    :param P:
    :param R:
    :param Lat:
    :param Lon:
    :return: radians
    '''
    D0 = [1.0, 0.0, 0.0]
    E0 = [0.0, 1.0, 0.0]
    N0 = [0.0, 0.0, 1.0]
    me = [0, 0, 0]
    N = [0., 0., 0.]
    E = [0., 0., 0.]
    D = [0., 0., 0.]

    RotateAboutAxis(E, E0, N0, Lon)
    me[0] = -E[0]
    me[1] = -E[1]
    me[2] = -E[2]

    RotateAboutAxis(N, N0, me, Lat)

    Cross(D, N, E)
    N1 = [0., 0., 0.]
    E1 = [0., 0., 0.]
    D1 = [0., 0., 0.]
    RotateAboutAxis(N1, N, D, H)
    RotateAboutAxis(E1, E, D, H)

    size = 3
    memcpy(D1, D, size)#有问题

    N2 = [0., 0., 0.]
    E2 = [0., 0., 0.]
    D2 = [0., 0., 0.]
    RotateAboutAxis(N2, N1, E1, P)

    memcpy(E2, E1, size)
    RotateAboutAxis(D2, D1, E1, P)

    N3 = [0., 0., 0.]
    E3 = [0., 0., 0.]
    D3 = [0., 0., 0.]

    memcpy(N3, N2, size)
    RotateAboutAxis(E3, E2, N2, R)
    RotateAboutAxis(D3, D2, N2, R)

    x0 = [1.0, 0.0, 0.0]
    y0 = [0.0, 1.0, 0.0]
    z0 = [0.0, 0.0, 1.0]

    y2 = [0., 0., 0.]
    z2 = [0., 0., 0.]

    Psi = atan2(Dot(N3, y0), Dot(N3, x0))
    Theta = atan2(-Dot(N3, z0), sqrt(pow(Dot(N3, x0), 2) + pow(Dot(N3, y0), 2)))
    RotateAboutAxis(y2, y0, z0, Psi)
    RotateAboutAxis(z2, z0, y2, Theta)
    Phi = atan2(Dot(E3, z2), Dot(E3, y2))
    return Psi, Theta, Phi


if __name__ == "__main__":
    H = radians(180)
    P = radians(12)
    R = radians(12)
    print((H,P,R))
    print(HeadingPitchRollToEuler(H, P, R, 30.3, -120.1))
    print(sin(1.57))
