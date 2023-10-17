import qi
import naoqi
import argparse
import sys
import random
import math
import json
import csv
import time
import pandas as pd
import numpy as np

def word_recognized(value):
    if value != []:
        for i in range( len(value)-1 ):
            print("word recognized is " + value[0] + " ,confidence is: " + value[1])

def main(session):
    
    memory = session.service("ALMemory")
    subscriber = memory.subscriber("WordRecognized")
    subscriber.signal.connect(word_recognized)

    tts = session.service("ALTextToSpeech")
    asr = session.service("ALSpeechRecognition")

    asr.setLanguage("English")

    vocabulary = ["yes", "no", "please", "hello"]
    asr.setVocabulary(vocabulary, False)

    # Start the speech recognition engine with user Test_ASR
    asr.subscribe("Test_ASR")

    # tts.say("a knee mo bumbaclat dog")
    # tts.say("do do do do do")
    tts.say("volume")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.22.1.21",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)