from collections import defaultdict
import sys
import random
import json

def generateBits(length):
    bits = ''
    for i in range(length):
        bits += str(random.getrandbits(1))

    return bits

def generateMessages(number):
    messages = defaultdict(dict)
    for i in range(number):
        if(bool(random.getrandbits(1))):
            messages[i]['start'] = '0'
            messages[i]['id'] = generateBits(11)
            messages[i]['rtr'] = '0'
            messages[i]['ide'] = '0'
            messages[i]['reservado'] = '0'
            messages[i]['dlc'] = generateBits(4)
            messages[i]['data'] = generateBits(8)
            messages[i]['crc'] = generateBits(15)
            messages[i]['dlm_crc'] = '1'
            messages[i]['ack'] = '1'
            messages[i]['dlm_ack'] = '1'
            messages[i]['eof'] = '1111111'
        else:
            messages[i]['start'] = generateBits(1)
            messages[i]['id'] = generateBits(11)
            messages[i]['rtr'] = generateBits(1)
            messages[i]['ide'] = generateBits(1)
            messages[i]['reservado'] = generateBits(1)
            messages[i]['dlc'] = generateBits(4)
            messages[i]['data'] = generateBits(8)
            messages[i]['crc'] = generateBits(15)
            messages[i]['dlm_crc'] = generateBits(1)
            messages[i]['ack'] = generateBits(1)
            messages[i]['dlm_ack'] = generateBits(1)
            messages[i]['eof'] = generateBits(7)

    return messages

def dumpMessages(messages, path):
    (open(path, 'w')).write(json.dumps(messages))

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Error con los argumentos.')
        sys.exit()

    number = int(sys.argv[1])
    path = sys.argv[2]

    messages = generateMessages(number)
    dumpMessages(messages, path)