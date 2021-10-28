#!python
from time import sleep

__author__ = "mcgredo"
__date__ = "$Jun 25, 2015 12:10:26 PM$"

import socket
from opendis.RangeCoordinates import GPS
from opendis.PduFactory import createPdu



UDP_PORT = 3000
udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
udpSocket.bind(("", UDP_PORT))
# udpSocket.settimeout(5) # exit if we get nothing in this many seconds

print("Created UDP socket {}".format(UDP_PORT))

gps = GPS();


def recv():
    try:
        data = udpSocket.recv(1024)  # buffer size in bytes
        pdu = createPdu(data)
        print(pdu.pduType)
        if pdu.pduType==112:
            print(pdu.name)


    except Exception as e:
        print(e)



if __name__ == "__main__":
    while 1:
        # recv()
        if udpSocket.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16):
            print('hhhhhhh')
        else:
            print('nnnnnnn')
