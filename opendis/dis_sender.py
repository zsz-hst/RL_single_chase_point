import socket
from io import BytesIO
from time import sleep

#from bs.fdthdler.aline import fdtAline
from data.performance_custom.OpenAP.acdict import acdict
from opendis.DataOutputStream import DataOutputStream
from opendis.dis7 import *
from opendis.RangeCoordinates import GPS

vtbxIP= "255.255.255.255"


def get_ownip():
    local_addrs = socket.gethostbyname_ex(socket.gethostname())[-1]
    for addr in local_addrs:
        if not addr.startswith('127'):
            return addr
    return '127.0.0.1'
try:
    vaddr=get_ownip()
    vtbxIP =vaddr[:vaddr.rfind(".")]+'.255'
except Exception as e:
    print(e)

class DisSet:
    def __init__(self, data=None):
        self.IP = vtbxIP
        self.SiteID = 0
        self.ExerciseID=1
        self.ApplicationID=0
        if data:
            self.set(data)
    def set(self,data):
        try:
            self.IP = data['IP']
            self.SiteID = int(data['SiteID'])
            self.ExerciseID = int(data['ExerciseID'])
            self.ApplicationID = int(data['ApplicationID'])
        except:
            print('Error setting dis config')
diset = DisSet()

UDP_PORT = 3000
class Count:
    def __init__(self):
        self.id = 10
count = Count()


def env_send_dis(id, lvpos, psi, theta, phi):
    UDP_PORT = 3000
    DESTINATION_ADDRESS = "255.255.255.255"
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    gps = GPS()
    pdu = EntityStatePdu()

    pdu.entityID.entityID = int(id)

    pdu.entityID.siteID = 0
    pdu.entityID.applicationID = pdu.entityID.entityID
    x = lvpos[0]
    y = lvpos[1]
    z = lvpos[2]

    montereyLocation = gps.lla2ecef((x, y, z))
    pdu.entityLocation.x = montereyLocation[0]
    pdu.entityLocation.y = montereyLocation[1]
    pdu.entityLocation.z = montereyLocation[2]

    pdu.entityOrientation.psi = psi
    pdu.entityOrientation.theta = theta
    pdu.entityOrientation.phi = phi
    get_entityType(pdu, 'f16')
 
    pdu.deadReckoningParameters.deadReckoningAlgorithm = 4
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (DESTINATION_ADDRESS, UDP_PORT))

def send_dis(id, lvpos, psi, theta, phi, alt_anh, actype, up):

    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    gps = GPS()
    pdu = EntityStatePdu()

    pdu.entityID.entityID = int(id)

    pdu.entityID.siteID = 0
    pdu.entityID.applicationID = pdu.entityID.entityID
    x = lvpos[0]
    y = lvpos[1]
    z = lvpos[2]

    montereyLocation = gps.lla2ecef((x, y, z))
    pdu.entityLocation.x = montereyLocation[0]
    pdu.entityLocation.y = montereyLocation[1]
    pdu.entityLocation.z = montereyLocation[2]
    #up -1 下降， 1上升
    pdu.entityAppearance = int(0x82666200)  # 放起落架
    if alt_anh >15 and up >= 0:
        pdu.entityAppearance = int(0x21403200)#收起落架
    elif up<0 and alt_anh<200:
        pdu.entityAppearance = int(0x82666200)#放起落架
    # pdu.marking.characterSet = id
    # pdu.marking.characters = [ id, id, id, id, id, id, id, id, id, id, id]
    pdu.entityOrientation.psi = psi
    pdu.entityOrientation.theta = theta
    pdu.entityOrientation.phi = phi
    get_entityType(pdu, actype)
    pdu.variableParameters.append(get_rubber())
    pdu.variableParameters.append(get_local())
    pdu.deadReckoningParameters.deadReckoningAlgorithm = 4
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))

def get_entityType(pdu, actype):
    actype = actype.upper()
    try:
        pdu.entityType = EntityType(acdict[actype])
        pdu.alternativeEntityType = EntityType(acdict[actype])
    except:
        pdu.entityType = EntityType(acdict['A320'])
        pdu.alternativeEntityType = EntityType(acdict['J11'])


def get_local():
    pdu = VariableParameter()
    pdu.variableParameterFields2=957
    return pdu

def get_rubber():
    pdu = ArticulatedParts()
    pdu.recordType = 1
    pdu.changeIndicator = 1
    pdu.partAttachedTo = 1
    pdu.parameterType = 1
    pdu.parameterValue = 1
    return pdu



def send_observer(ICAO, observerType, entityID, positionLLA, orientationPBH, relativeXYZ, relativePBH):


    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = ObserverPdu()
    pdu.ICAO = str.encode(ICAO)
    pdu.observerType = observerType
    pdu.entityID = entityID
    pdu.positionLLA.x = positionLLA.x
    pdu.positionLLA.y = positionLLA.y
    pdu.positionLLA.z = positionLLA.z
    pdu.orientationPBH.x = orientationPBH.x
    pdu.orientationPBH.y = orientationPBH.y
    pdu.orientationPBH.z = orientationPBH.z
    pdu.relativeXYZ.x = relativeXYZ.x
    pdu.relativeXYZ.y = relativeXYZ.y
    pdu.relativeXYZ.z = relativeXYZ.z
    pdu.relativePBH.x = relativePBH.x
    pdu.relativePBH.y = relativePBH.y
    pdu.relativePBH.z = relativePBH.z
   
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))



def send_radarinfo(nID, nState, dLon, dLat, dAlt, dSize):

    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = DrawRadarPDU()

    pdu.nID = nID
    pdu.nState = nState
    pdu.dLon = dLon
    pdu.dLat = dLat
    pdu.dAlt = dAlt
    pdu.dSize = dSize
  
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))

def send_time(year, month, day, hour, minute, second):
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = TimeInfoPDU()

    pdu.year = year
    pdu.month = month
    pdu.day = day
    pdu.hour = hour
    pdu.minute = minute
    pdu.second = second

    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))


def send_weather(windSpeed, windDir, enableRain, enableSnow, visibility, season):


    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = WeatherStatePdu()
    pdu.windSpeed = windSpeed
    pdu.windDir = windDir
    pdu.enableRain = enableRain
    pdu.enableSnow = enableSnow
    pdu.visibility = visibility
    pdu.season = season
    # print((windSpeed, windDir, enableRain, enableSnow, visibility, season))

    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))
def send_timepoint(test):
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = timepointPDU()
    pdu.name = test
    pdu.len = len(test)
    pdu.pduType = 110
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))

def send_route(test):
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = routePDU()
    pdu.name = test
    pdu.len = len(test)
    pdu.pduType = 111
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))
def send_pdu(test):
    udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    pdu = mypdu()
    pdu.len = len(list(test))
    pdu.name = list(test)
    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    udpSocket.sendto(data, (diset.IP, UDP_PORT))

if __name__ == "__main__":
    # send_pdu('ack_ucac')
    # while 1:
    #     send_pdu('request_ucac')
    while 1:
        udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        pdu = FdtInfoPDU()
        pdu({'aline':{'array':np.array([1.1,2.2,3.3],dtype=np.double),\
                        'timestamps':np.array(['10:11:22,','10:11:22,','10:11:22,'],dtype=np.string_)
                      },
             'aline2': {'array': np.array([1.1, 2.2], dtype=np.double), \
                       'timestamps': np.array(['10:11:22,', '10:11:22,'], dtype=np.string_)
                       }
             })
        # pdu.name = list('ack')
        # pdu.len = len(pdu.name)
        memoryStream = BytesIO()
        outputStream = DataOutputStream(memoryStream)
        pdu.serialize(outputStream)
        data = memoryStream.getvalue()
        # print('send Pdu')
        udpSocket.sendto(data, ('192.168.57.43', 3000))




