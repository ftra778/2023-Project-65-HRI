
import dialogflow

import qi
import naoqi
import argparse
import sys

def detect_intent_texts(project_id, session_id, texts, language_code):

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    response_text = ""
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        print('=' * 20)
        for i in response.query_result.fulfillment_messages:
            response_text = response_text + (str(i.text)[7:-2])

        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        # print('Fulfillment text: {}\n'.format(
        #    response.query_result.fulfillment_text))
        print("Response text: {}\n".format(response_text))

def main(session):

    project_id = "pepperrobot-ppgw"
    session_id = "1234567890"
    texts = [""]
    language_code = "en-US"

    while(True):
        texts = [""]
        texts[0] = raw_input("Talk:")
        detect_intent_texts(project_id, session_id, texts, language_code)
    
    # max_rt_s = 20
    # motion_service = session.service("ALMotion")
    # posture_service = session.service("ALRobotPosture")
    # tts = session.service("ALTextToSpeech")

    # tts.say("")
    # tts.say("do do do do do")
    # tts.say("volume")



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